"""Best-effort Slack crawler using only authenticated read-only MCP tools."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol, cast

import anyio
from pydantic import Field

from autobrain.mcp.policy import ToolSnapshot, UntrustedToolResult
from autobrain.models import (
    CoverageCompleteness,
    Sha256,
    SourceId,
    SourceKind,
    Status,
    StrictModel,
)

_CHANNEL_LIST_TOOLS = ("slack-channel-list", "slack_channels_list")
_HISTORY_TOOLS = ("slack-channel-history", "slack_channel_history")
_THREAD_TOOLS = ("slack-thread-replies", "slack_thread_replies")
_FILE_TOOLS = ("slack-file-read", "slack_file_read")
_CANVAS_TOOLS = ("slack-canvas-read", "slack_canvas_read")
_DENIED_ERRORS = {
    "access_denied",
    "channel_not_found",
    "missing_scope",
    "not_allowed",
    "not_in_channel",
    "permission_denied",
}
_TRUNCATED_ERRORS = {"cursor_expired", "invalid_cursor", "response_too_large"}
_TEXT_MIME_PREFIXES = ("text/",)
_TEXT_MIME_TYPES = {
    "application/json",
    "application/rtf",
    "application/xml",
    "application/x-yaml",
}
_SECRET_RE = re.compile(r"\bxox[baprs]-[A-Za-z0-9-]+\b")
_HOLDOUT_RE = re.compile(r"\b(?:HOLDOUT|ORACLE)-[A-Za-z0-9_-]+\b", re.IGNORECASE)


class SlackMcpClient(Protocol):
    @property
    def snapshot(self) -> ToolSnapshot: ...

    async def call(self, name: str, arguments: dict[str, Any]) -> UntrustedToolResult: ...


class SlackCoverageRecord(StrictModel):
    channel_id: str | None = None
    channel_name: str | None = None
    channel_type: str | None = None
    content_type: str
    completeness: CoverageCompleteness = CoverageCompleteness.UNKNOWN
    discovered: int = Field(default=0, ge=0)
    fetched: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    truncated: int = Field(default=0, ge=0)
    denied: int = Field(default=0, ge=0)
    rate_limited: int = Field(default=0, ge=0)
    unsupported: int = Field(default=0, ge=0)


class SlackCoverageLedger(StrictModel):
    completeness: CoverageCompleteness = CoverageCompleteness.UNKNOWN
    exhaustive_organization: bool = False
    scopes: tuple[str, ...]
    channel_types: tuple[str, ...]
    records: tuple[SlackCoverageRecord, ...]
    warnings: tuple[str, ...] = ()
    interrupted: bool = False

    def record(self, *, channel_id: str | None, content_type: str) -> SlackCoverageRecord:
        matches = [
            item
            for item in self.records
            if item.channel_id == channel_id and item.content_type == content_type
        ]
        if len(matches) != 1:
            raise LookupError(
                f"expected one coverage record for channel={channel_id!r}, "
                f"content_type={content_type!r}; found {len(matches)}"
            )
        return matches[0]


class RawSlackDocument(StrictModel):
    source_id: SourceId
    source_kind: SourceKind
    canonical_url: str = Field(pattern=r"^https?://")
    title: str = Field(min_length=1)
    text: str
    content_hash: Sha256
    channel_id: str
    channel_name: str
    channel_type: str
    channel_archived: bool = False
    message_ts: str | None = None
    thread_ts: str | None = None
    parent_source_id: SourceId | None = None
    user_id: str | None = None
    user_name: str | None = None
    bot: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None
    edited: bool = False
    deleted: bool = False
    untrusted: bool = True
    crawl_provenance: dict[str, str]
    metadata: dict[str, str] = Field(default_factory=dict)


class SlackCrawlResult(StrictModel):
    status: Status
    documents: tuple[RawSlackDocument, ...]
    coverage: SlackCoverageLedger


@dataclass
class _MutableCoverage:
    channel_id: str | None
    content_type: str
    channel_name: str | None = None
    channel_type: str | None = None
    discovered: int = 0
    fetched: int = 0
    skipped: int = 0
    truncated: int = 0
    denied: int = 0
    rate_limited: int = 0
    unsupported: int = 0

    def freeze(self) -> SlackCoverageRecord:
        return SlackCoverageRecord(
            channel_id=self.channel_id,
            channel_name=self.channel_name,
            channel_type=self.channel_type,
            content_type=self.content_type,
            discovered=self.discovered,
            fetched=self.fetched,
            skipped=self.skipped,
            truncated=self.truncated,
            denied=self.denied,
            rate_limited=self.rate_limited,
            unsupported=self.unsupported,
        )


@dataclass
class _State:
    records: dict[tuple[str | None, str], _MutableCoverage] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    interrupted: bool = False
    partial: bool = False

    def record(
        self,
        channel_id: str | None,
        content_type: str,
        *,
        channel_name: str | None = None,
        channel_type: str | None = None,
    ) -> _MutableCoverage:
        key = (channel_id, content_type)
        current = self.records.get(key)
        if current is None:
            current = _MutableCoverage(
                channel_id=channel_id,
                channel_name=channel_name,
                channel_type=channel_type,
                content_type=content_type,
            )
            self.records[key] = current
        elif channel_name is not None:
            current.channel_name = channel_name
            current.channel_type = channel_type
        return current


@dataclass(frozen=True)
class _Channel:
    channel_id: str
    name: str
    channel_type: str
    archived: bool


@dataclass(frozen=True)
class _RetrievedMessage:
    raw: dict[str, object]
    page: int
    cursor_lineage: tuple[str, ...]
    retry_count: int


class _MalformedResponse(ValueError):
    pass


class _DeniedResponse(PermissionError):
    pass


class _TruncatedResponse(RuntimeError):
    pass


class _PartialMessages(_TruncatedResponse):
    def __init__(self, messages: list[_RetrievedMessage], reason: Exception) -> None:
        super().__init__(str(reason))
        self.messages = messages
        self.interrupted = isinstance(reason, InterruptedError)


async def _no_wait(_seconds: float) -> None:
    """Default deterministic retry clock without wall-clock sleeping."""


class SlackCrawler:
    """Crawl the Slack surfaces advertised by an authenticated MCP connection."""

    def __init__(
        self,
        client: SlackMcpClient,
        *,
        include_dms: bool = False,
        concurrency: int = 4,
        max_retries: int = 3,
        retry_wait: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least one")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.client = client
        self.include_dms = include_dms
        self.max_retries = max_retries
        self.retry_wait = retry_wait or _no_wait
        self._limit = anyio.Semaphore(concurrency)
        allowed = set(client.snapshot.allowed)
        self._channel_list = self._select_tool(allowed, _CHANNEL_LIST_TOOLS)
        self._history = self._select_tool(allowed, _HISTORY_TOOLS)
        self._threads = self._select_tool(allowed, _THREAD_TOOLS)
        self._files = self._select_tool(allowed, _FILE_TOOLS)
        self._canvases = self._select_tool(allowed, _CANVAS_TOOLS)

    @staticmethod
    def _select_tool(allowed: set[str], candidates: tuple[str, ...]) -> str | None:
        return next((name for name in candidates if name in allowed), None)

    async def crawl(self, *, scopes: tuple[str, ...]) -> SlackCrawlResult:
        state = _State()
        channel_types = ["public_channel", "private_channel"]
        if self.include_dms:
            channel_types.extend(("im", "mpim"))
        global_channels = state.record(None, "channel")
        if self._channel_list is None or self._history is None:
            global_channels.unsupported += 1
            state.warnings.append("Slack MCP lacks channel-list or channel-history capability")
            return self._result(
                Status.CAPABILITY_UNAVAILABLE,
                (),
                state,
                scopes=scopes,
                channel_types=channel_types,
            )

        try:
            discovered = await self._discover_channels(channel_types, global_channels)
        except _MalformedResponse as exc:
            global_channels.truncated += 1
            state.partial = True
            state.warnings.append(f"Malformed Slack channel discovery response: {exc}")
            return self._result(
                Status.FAILED, (), state, scopes=scopes, channel_types=channel_types
            )
        except (_DeniedResponse, _TruncatedResponse) as exc:
            global_channels.truncated += 1
            state.partial = True
            state.warnings.append(f"Slack channel discovery was incomplete: {exc}")
            return self._result(
                Status.FAILED, (), state, scopes=scopes, channel_types=channel_types
            )

        global_channels.discovered = len(discovered)
        readable = [item for item in discovered if item.channel_type in channel_types]
        global_channels.skipped = len(discovered) - len(readable)
        per_channel: dict[str, list[RawSlackDocument]] = {}

        async def crawl_channel(channel: _Channel) -> None:
            try:
                documents = await self._crawl_channel(channel, state)
            except InterruptedError as exc:
                state.interrupted = True
                state.partial = True
                state.warnings.append(
                    f"Slack crawl interrupted for channel {channel.channel_id}: {exc}"
                )
                record = state.record(channel.channel_id, "message")
                record.truncated += 1
                return
            per_channel[channel.channel_id] = documents
            history = state.record(channel.channel_id, "message")
            if history.denied == 0 and (history.truncated == 0 or history.discovered > 0):
                global_channels.fetched += 1

        async with anyio.create_task_group() as tasks:
            for item in sorted(readable, key=lambda value: value.channel_id):
                tasks.start_soon(crawl_channel, item)

        documents = tuple(
            document for channel_id in sorted(per_channel) for document in per_channel[channel_id]
        )
        status = Status.FAILED if state.partial else Status.OK
        return self._result(
            status,
            documents,
            state,
            scopes=scopes,
            channel_types=channel_types,
        )

    def _result(
        self,
        status: Status,
        documents: tuple[RawSlackDocument, ...],
        state: _State,
        *,
        scopes: tuple[str, ...],
        channel_types: list[str],
    ) -> SlackCrawlResult:
        records = tuple(
            state.records[key].freeze()
            for key in sorted(
                state.records,
                key=lambda value: (value[0] or "", value[1]),
            )
        )
        return SlackCrawlResult(
            status=status,
            documents=documents,
            coverage=SlackCoverageLedger(
                scopes=tuple(sorted(set(scopes))),
                channel_types=tuple(sorted(channel_types)),
                records=records,
                warnings=tuple(sorted(state.warnings)),
                interrupted=state.interrupted,
            ),
        )

    async def _discover_channels(
        self,
        channel_types: list[str],
        coverage: _MutableCoverage,
    ) -> list[_Channel]:
        assert self._channel_list is not None
        cursor: str | None = None
        seen_cursors: set[str] = set()
        result: list[_Channel] = []
        while True:
            arguments: dict[str, Any] = {"types": channel_types, "limit": 200}
            if cursor is not None:
                arguments["cursor"] = cursor
            payload = await self._call(self._channel_list, arguments, coverage)
            channels = payload.get("channels")
            if not isinstance(channels, list):
                raise _MalformedResponse("channels was not a list")
            for raw_item in cast(list[object], channels):
                if not isinstance(raw_item, dict):
                    raise _MalformedResponse("channel item was not an object")
                raw = cast(dict[str, object], raw_item)
                channel_id = raw.get("id")
                name = raw.get("name")
                channel_type = raw.get("type")
                if not all(
                    isinstance(value, str) and value for value in (channel_id, name, channel_type)
                ):
                    raise _MalformedResponse("channel identity was missing")
                assert isinstance(channel_id, str)
                assert isinstance(name, str)
                assert isinstance(channel_type, str)
                result.append(
                    _Channel(
                        channel_id=channel_id,
                        name=name,
                        channel_type=channel_type,
                        archived=bool(raw.get("is_archived", False)),
                    )
                )
            cursor = self._next_cursor(payload)
            if cursor is None:
                return result
            if cursor in seen_cursors:
                raise _TruncatedResponse("channel cursor repeated")
            seen_cursors.add(cursor)

    async def _crawl_channel(
        self,
        channel: _Channel,
        state: _State,
    ) -> list[RawSlackDocument]:
        assert self._history is not None
        history = state.record(
            channel.channel_id,
            "message",
            channel_name=channel.name,
            channel_type=channel.channel_type,
        )
        history_truncated = False
        history_interrupted = False
        try:
            messages = await self._paginate_messages(
                self._history,
                {"channel_id": channel.channel_id, "limit": 200},
                history,
            )
        except _DeniedResponse:
            history.channel_name = None
            history.channel_type = channel.channel_type
            history.denied += 1
            state.partial = True
            state.warnings.append(f"Slack channel {channel.channel_id} was denied")
            return []
        except _PartialMessages as exc:
            messages = exc.messages
            history.truncated += 1
            state.partial = True
            state.interrupted = state.interrupted or exc.interrupted
            history_interrupted = exc.interrupted
            state.warnings.append(
                f"Slack channel {channel.channel_id} history was truncated: {exc}"
            )
            history_truncated = True
        except (_MalformedResponse, _TruncatedResponse) as exc:
            history.truncated += 1
            state.partial = True
            state.warnings.append(
                f"Slack channel {channel.channel_id} history was truncated: {exc}"
            )
            return []

        documents: list[RawSlackDocument] = []
        seen_attachments: set[tuple[str, str]] = set()
        for retrieved in messages:
            raw = retrieved.raw
            history.discovered += 1
            try:
                document = self._message_document(
                    channel,
                    retrieved,
                    tool=self._history,
                    surface="channel_history",
                    partial=history_truncated,
                    interrupted=history_interrupted,
                )
            except _MalformedResponse as exc:
                history.skipped += 1
                history.truncated += 1
                state.partial = True
                state.warnings.append(
                    f"Malformed Slack message in channel {channel.channel_id}: {exc}"
                )
                continue
            documents.append(document)
            history.fetched += 1
            documents.extend(
                await self._fetch_attachments(
                    channel,
                    raw,
                    parent_source_id=document.source_id,
                    origin=retrieved,
                    origin_partial=history_truncated,
                    origin_interrupted=history_interrupted,
                    state=state,
                    seen=seen_attachments,
                )
            )
            reply_count = raw.get("reply_count", 0)
            if isinstance(reply_count, int) and reply_count > 0:
                documents.extend(
                    await self._fetch_thread(
                        channel,
                        raw,
                        root_source_id=document.source_id,
                        state=state,
                    )
                )
        if history_truncated and not documents:
            state.warnings.append(
                f"Slack channel {channel.channel_id} yielded no usable documents before truncation"
            )
        return documents

    async def _fetch_thread(
        self,
        channel: _Channel,
        root: dict[str, object],
        *,
        root_source_id: SourceId,
        state: _State,
    ) -> list[RawSlackDocument]:
        coverage = state.record(
            channel.channel_id,
            "thread_reply",
            channel_name=channel.name,
            channel_type=channel.channel_type,
        )
        root_ts = root.get("ts")
        if not isinstance(root_ts, str):
            coverage.truncated += 1
            state.partial = True
            return []
        if self._threads is None:
            coverage.unsupported += 1
            coverage.truncated += 1
            state.partial = True
            state.warnings.append("Slack MCP does not advertise thread retrieval")
            return []
        thread_interrupted = False
        try:
            messages = await self._paginate_messages(
                self._threads,
                {"channel_id": channel.channel_id, "thread_ts": root_ts, "limit": 200},
                coverage,
            )
        except _DeniedResponse:
            coverage.denied += 1
            coverage.truncated += 1
            state.partial = True
            return []
        except _PartialMessages as exc:
            messages = exc.messages
            coverage.truncated += 1
            state.partial = True
            state.interrupted = state.interrupted or exc.interrupted
            thread_interrupted = exc.interrupted
            state.warnings.append(
                f"Malformed or truncated Slack thread {channel.channel_id}:{root_ts}: {exc}"
            )
        except (_MalformedResponse, _TruncatedResponse) as exc:
            coverage.truncated += 1
            state.partial = True
            state.warnings.append(
                f"Malformed or truncated Slack thread {channel.channel_id}:{root_ts}: {exc}"
            )
            return []
        documents: list[RawSlackDocument] = []
        thread_partial = coverage.truncated > 0
        for retrieved in messages:
            raw = retrieved.raw
            if raw.get("ts") == root_ts:
                continue
            coverage.discovered += 1
            documents.append(
                self._message_document(
                    channel,
                    retrieved,
                    tool=self._threads,
                    surface="thread_replies",
                    partial=thread_partial,
                    interrupted=thread_interrupted,
                    parent_source_id=root_source_id,
                    forced_thread_ts=root_ts,
                    root_source_id=root_source_id,
                )
            )
            coverage.fetched += 1
        return documents

    async def _fetch_attachments(
        self,
        channel: _Channel,
        raw: dict[str, object],
        *,
        parent_source_id: SourceId,
        origin: _RetrievedMessage,
        origin_partial: bool,
        origin_interrupted: bool,
        state: _State,
        seen: set[tuple[str, str]],
    ) -> list[RawSlackDocument]:
        documents: list[RawSlackDocument] = []
        files = raw.get("files", [])
        if files is not None and not isinstance(files, list):
            state.record(channel.channel_id, "file").truncated += 1
            state.partial = True
            files = []
        for raw_item in cast(list[object], files):
            if not isinstance(raw_item, dict):
                state.record(channel.channel_id, "file").truncated += 1
                state.partial = True
                continue
            item = cast(dict[str, object], raw_item)
            coverage = state.record(
                channel.channel_id,
                "file",
                channel_name=channel.name,
                channel_type=channel.channel_type,
            )
            coverage.discovered += 1
            file_id = item.get("id")
            mime = item.get("mimetype", "")
            if not isinstance(file_id, str) or not file_id:
                coverage.truncated += 1
                state.partial = True
                continue
            if ("file", file_id) in seen:
                coverage.skipped += 1
                continue
            seen.add(("file", file_id))
            if not isinstance(mime, str) or not self._is_text_mime(mime):
                coverage.skipped += 1
                coverage.unsupported += 1
                continue
            if self._files is None:
                coverage.skipped += 1
                coverage.unsupported += 1
                state.partial = True
                continue
            try:
                payload, retry_count = await self._call_with_attempts(
                    self._files,
                    {"file_id": file_id},
                    coverage,
                )
                text = payload.get("text")
                if not isinstance(text, str):
                    raise _MalformedResponse("text file body was missing")
            except _DeniedResponse:
                coverage.denied += 1
                state.partial = True
                continue
            except (_MalformedResponse, _TruncatedResponse) as exc:
                coverage.truncated += 1
                state.partial = True
                state.warnings.append(f"Slack file {file_id} was truncated: {exc}")
                continue
            try:
                documents.append(
                    self._attachment_document(
                        channel,
                        source_kind=SourceKind.SLACK_FILE,
                        attachment_id=file_id,
                        title=self._string(item.get("name")) or f"Slack file {file_id}",
                        text=text,
                        canonical_url=self._required_url(item.get("permalink"), "file", file_id),
                        parent_source_id=parent_source_id,
                        metadata={"mimetype": mime},
                        tool=self._files,
                        surface="file_read",
                        retrieval="file_attachment",
                        retry_count=retry_count,
                        partial=origin_partial,
                        interrupted=origin_interrupted,
                        origin=origin,
                    )
                )
            except _MalformedResponse as exc:
                coverage.skipped += 1
                coverage.truncated += 1
                state.partial = True
                state.warnings.append(f"Slack file {file_id} was malformed: {exc}")
                continue
            coverage.fetched += 1

        canvases = raw.get("canvases", [])
        if canvases is not None and not isinstance(canvases, list):
            state.record(channel.channel_id, "canvas").truncated += 1
            state.partial = True
            canvases = []
        for raw_item in cast(list[object], canvases):
            if not isinstance(raw_item, dict):
                state.record(channel.channel_id, "canvas").truncated += 1
                state.partial = True
                continue
            item = cast(dict[str, object], raw_item)
            coverage = state.record(
                channel.channel_id,
                "canvas",
                channel_name=channel.name,
                channel_type=channel.channel_type,
            )
            coverage.discovered += 1
            canvas_id = item.get("id")
            if not isinstance(canvas_id, str) or not canvas_id:
                coverage.truncated += 1
                state.partial = True
                continue
            if ("canvas", canvas_id) in seen:
                coverage.skipped += 1
                continue
            seen.add(("canvas", canvas_id))
            if self._canvases is None:
                coverage.skipped += 1
                coverage.unsupported += 1
                state.partial = True
                continue
            try:
                payload, retry_count = await self._call_with_attempts(
                    self._canvases,
                    {"canvas_id": canvas_id},
                    coverage,
                )
                text = payload.get("markdown", payload.get("text"))
                if not isinstance(text, str):
                    raise _MalformedResponse("canvas body was missing")
            except _DeniedResponse:
                coverage.denied += 1
                state.partial = True
                continue
            except (_MalformedResponse, _TruncatedResponse) as exc:
                coverage.truncated += 1
                state.partial = True
                state.warnings.append(f"Slack canvas {canvas_id} was truncated: {exc}")
                continue
            try:
                documents.append(
                    self._attachment_document(
                        channel,
                        source_kind=SourceKind.SLACK_CANVAS,
                        attachment_id=canvas_id,
                        title=self._string(item.get("title")) or f"Slack canvas {canvas_id}",
                        text=text,
                        canonical_url=self._required_url(
                            item.get("permalink"), "canvas", canvas_id
                        ),
                        parent_source_id=parent_source_id,
                        metadata={},
                        tool=self._canvases,
                        surface="canvas_read",
                        retrieval="canvas_attachment",
                        retry_count=retry_count,
                        partial=origin_partial,
                        interrupted=origin_interrupted,
                        origin=origin,
                    )
                )
            except _MalformedResponse as exc:
                coverage.skipped += 1
                coverage.truncated += 1
                state.partial = True
                state.warnings.append(f"Slack canvas {canvas_id} was malformed: {exc}")
                continue
            coverage.fetched += 1
        return documents

    async def _paginate_messages(
        self,
        tool: str,
        base_arguments: dict[str, Any],
        coverage: _MutableCoverage,
    ) -> list[_RetrievedMessage]:
        cursor: str | None = None
        seen_cursors: set[str] = set()
        cursor_lineage = ["start"]
        messages: list[_RetrievedMessage] = []
        page_number = 1
        while True:
            arguments = dict(base_arguments)
            if cursor is not None:
                arguments["cursor"] = cursor
            try:
                payload, retry_count = await self._call_with_attempts(tool, arguments, coverage)
            except (
                InterruptedError,
                _MalformedResponse,
                _DeniedResponse,
                _TruncatedResponse,
            ) as exc:
                if messages:
                    raise _PartialMessages(messages, exc) from exc
                raise
            page = payload.get("messages")
            if not isinstance(page, list):
                raise _MalformedResponse("messages was not a list")
            for raw_item in cast(list[object], page):
                if not isinstance(raw_item, dict):
                    raise _MalformedResponse("message item was not an object")
                messages.append(
                    _RetrievedMessage(
                        raw=cast(dict[str, object], raw_item),
                        page=page_number,
                        cursor_lineage=tuple(cursor_lineage),
                        retry_count=retry_count,
                    )
                )
            cursor = self._next_cursor(payload)
            if cursor is None:
                return messages
            if cursor in seen_cursors:
                raise _TruncatedResponse("cursor repeated")
            seen_cursors.add(cursor)
            cursor_lineage.append(self._cursor_fingerprint(cursor))
            page_number += 1

    async def _call(
        self,
        tool: str,
        arguments: dict[str, Any],
        coverage: _MutableCoverage,
    ) -> dict[str, object]:
        payload, _retry_count = await self._call_with_attempts(tool, arguments, coverage)
        return payload

    async def _call_with_attempts(
        self,
        tool: str,
        arguments: dict[str, Any],
        coverage: _MutableCoverage,
    ) -> tuple[dict[str, object], int]:
        retries = 0
        while True:
            async with self._limit:
                result = await self.client.call(tool, arguments)
            payload = self._decode_payload(result)
            if not result.is_error:
                return payload, retries
            code = payload.get("error", payload.get("code"))
            if code == "rate_limited":
                coverage.rate_limited += 1
                if retries >= self.max_retries:
                    raise _TruncatedResponse("rate limit retry budget exhausted")
                retry_after = payload.get("retry_after", 0)
                if not isinstance(retry_after, int | float) or retry_after < 0:
                    raise _MalformedResponse("Retry-After was invalid")
                retries += 1
                await self.retry_wait(float(retry_after))
                continue
            if isinstance(code, str) and code in _DENIED_ERRORS:
                raise _DeniedResponse(code)
            if isinstance(code, str) and code in _TRUNCATED_ERRORS:
                raise _TruncatedResponse(code)
            raise _MalformedResponse(f"MCP tool {tool} returned error {code!r}")

    @staticmethod
    def _decode_payload(result: UntrustedToolResult) -> dict[str, object]:
        content = result.content
        if isinstance(content, dict):
            return cast(dict[str, object], content)
        if not isinstance(content, list):
            raise _MalformedResponse("MCP content was not a list or object")
        decoded: list[object] = []
        for raw_item in cast(list[object], content):
            if not isinstance(raw_item, dict):
                continue
            item = cast(dict[str, object], raw_item)
            text = item.get("text")
            if isinstance(text, str):
                try:
                    decoded.append(json.loads(text))
                except json.JSONDecodeError as exc:
                    raise _MalformedResponse("MCP text content was not valid JSON") from exc
            elif "json" in item:
                decoded.append(item["json"])
        if len(decoded) != 1 or not isinstance(decoded[0], dict):
            raise _MalformedResponse("MCP response did not contain one JSON object")
        return cast(dict[str, object], decoded[0])

    def _message_document(
        self,
        channel: _Channel,
        retrieved: _RetrievedMessage,
        *,
        tool: str,
        surface: str,
        partial: bool,
        interrupted: bool,
        parent_source_id: SourceId | None = None,
        forced_thread_ts: str | None = None,
        root_source_id: SourceId | None = None,
    ) -> RawSlackDocument:
        raw = retrieved.raw
        ts = raw.get("ts")
        if not isinstance(ts, str) or not ts:
            raise _MalformedResponse("message timestamp was missing")
        reply_count = raw.get("reply_count", 0)
        is_root = parent_source_id is None and isinstance(reply_count, int) and reply_count > 0
        source_kind = SourceKind.SLACK_THREAD if is_root else SourceKind.SLACK_MESSAGE
        prefix = "slack-thread" if is_root else "slack-message"
        source_id: SourceId = f"{prefix}:{channel.channel_id}:{ts}"
        thread_ts = forced_thread_ts or self._string(raw.get("thread_ts"))
        text = self._redact_source_text(self._string(raw.get("text")) or "")
        subtype = self._string(raw.get("subtype"))
        deleted = subtype == "message_deleted"
        edited = raw.get("edited")
        edited_data = cast(dict[str, object], edited) if isinstance(edited, dict) else None
        edited_ts = edited_data.get("ts") if edited_data is not None else None
        canonical_url = self._required_url(
            raw.get("permalink"), "message", f"{channel.channel_id}:{ts}"
        )
        metadata = {"subtype": subtype} if subtype else {}
        return RawSlackDocument(
            source_id=source_id,
            source_kind=source_kind,
            canonical_url=canonical_url,
            title=f"#{channel.name} at {ts}",
            text=text,
            content_hash=self._hash(
                {
                    "source_id": source_id,
                    "canonical_url": canonical_url,
                    "text": text,
                    "thread_ts": thread_ts,
                    "parent_source_id": parent_source_id,
                    "edited_ts": edited_ts,
                    "subtype": subtype,
                }
            ),
            channel_id=channel.channel_id,
            channel_name=channel.name,
            channel_type=channel.channel_type,
            channel_archived=channel.archived,
            message_ts=ts,
            thread_ts=thread_ts,
            parent_source_id=parent_source_id,
            user_id=self._string(raw.get("user")),
            user_name=self._string(raw.get("user_name")),
            bot=bool(raw.get("bot_id")) or subtype == "bot_message",
            created_at=self._timestamp(ts),
            updated_at=self._timestamp(edited_ts),
            edited=edited_data is not None,
            deleted=deleted,
            crawl_provenance=self._provenance(
                channel,
                tool=tool,
                surface=surface,
                retrieval=(
                    "thread_reply"
                    if parent_source_id is not None
                    else "thread_root"
                    if is_root
                    else "standalone_message"
                ),
                page=retrieved.page,
                cursor_lineage=retrieved.cursor_lineage,
                retry_count=retrieved.retry_count,
                partial=partial,
                interrupted=interrupted,
                parent_source_id=parent_source_id,
                root_source_id=root_source_id,
            ),
            metadata=metadata,
        )

    def _attachment_document(
        self,
        channel: _Channel,
        *,
        source_kind: SourceKind,
        attachment_id: str,
        title: str,
        text: str,
        canonical_url: str,
        parent_source_id: SourceId,
        metadata: dict[str, str],
        tool: str,
        surface: str,
        retrieval: str,
        retry_count: int,
        partial: bool,
        interrupted: bool,
        origin: _RetrievedMessage,
    ) -> RawSlackDocument:
        prefix = "slack-file" if source_kind is SourceKind.SLACK_FILE else "slack-canvas"
        source_id: SourceId = f"{prefix}:{attachment_id}"
        redacted = self._redact_source_text(text)
        return RawSlackDocument(
            source_id=source_id,
            source_kind=source_kind,
            canonical_url=canonical_url,
            title=title,
            text=redacted,
            content_hash=self._hash(
                {
                    "source_id": source_id,
                    "canonical_url": canonical_url,
                    "text": redacted,
                    "parent_source_id": parent_source_id,
                    "metadata": metadata,
                }
            ),
            channel_id=channel.channel_id,
            channel_name=channel.name,
            channel_type=channel.channel_type,
            channel_archived=channel.archived,
            parent_source_id=parent_source_id,
            crawl_provenance=self._provenance(
                channel,
                tool=tool,
                surface=surface,
                retrieval=retrieval,
                page=1,
                cursor_lineage=("direct",),
                retry_count=retry_count,
                partial=partial,
                interrupted=interrupted,
                parent_source_id=parent_source_id,
                origin=origin,
            ),
            metadata=metadata,
        )

    @classmethod
    def _provenance(
        cls,
        channel: _Channel,
        *,
        tool: str,
        surface: str,
        retrieval: str,
        page: int,
        cursor_lineage: tuple[str, ...],
        retry_count: int,
        partial: bool,
        interrupted: bool,
        parent_source_id: SourceId | None = None,
        root_source_id: SourceId | None = None,
        origin: _RetrievedMessage | None = None,
    ) -> dict[str, str]:
        provenance = {
            "attempt_count": str(retry_count + 1),
            "channel_id": channel.channel_id,
            "channel_type": channel.channel_type,
            "connector": "slack-mcp",
            "cursor_lineage": ">".join(cursor_lineage),
            "interrupted": str(interrupted).lower(),
            "page": str(page),
            "partial": str(partial).lower(),
            "retrieval": retrieval,
            "retried": str(retry_count > 0).lower(),
            "retry_count": str(retry_count),
            "surface": surface,
            "tool": tool,
        }
        if parent_source_id is not None:
            provenance["parent_source_id"] = parent_source_id
        if root_source_id is not None:
            provenance["root_source_id"] = root_source_id
        if origin is not None:
            provenance.update(
                {
                    "origin_cursor_lineage": ">".join(origin.cursor_lineage),
                    "origin_page": str(origin.page),
                    "origin_surface": "channel_history",
                }
            )
        return provenance

    @staticmethod
    def _cursor_fingerprint(cursor: str) -> str:
        return f"sha256:{hashlib.sha256(cursor.encode()).hexdigest()}"

    @staticmethod
    def _next_cursor(payload: dict[str, object]) -> str | None:
        direct = payload.get("next_cursor")
        if direct in (None, ""):
            metadata = payload.get("response_metadata")
            if isinstance(metadata, dict):
                direct = cast(dict[str, object], metadata).get("next_cursor")
        if direct in (None, ""):
            return None
        if not isinstance(direct, str):
            raise _MalformedResponse("next cursor was not a string")
        return direct

    @staticmethod
    def _required_url(value: object, kind: str, identity: str) -> str:
        if isinstance(value, str) and value.startswith(("https://", "http://")):
            return value
        raise _MalformedResponse(f"{kind} {identity} lacked a canonical permalink")

    @staticmethod
    def _timestamp(value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromtimestamp(float(Decimal(value)), tz=UTC)
        except (InvalidOperation, ValueError, OverflowError):
            return None

    @staticmethod
    def _string(value: object) -> str | None:
        return value if isinstance(value, str) else None

    @staticmethod
    def _is_text_mime(value: str) -> bool:
        return value.startswith(_TEXT_MIME_PREFIXES) or value in _TEXT_MIME_TYPES

    @staticmethod
    def _redact_source_text(value: str) -> str:
        return _HOLDOUT_RE.sub(
            "[REDACTED_HOLDOUT]",
            _SECRET_RE.sub("[REDACTED_SECRET]", value),
        )

    @staticmethod
    def _hash(value: dict[str, object]) -> Sha256:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()
