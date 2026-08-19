"""Notion MCP discovery and whole-document extraction."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Protocol, cast

from autobrain.mcp.policy import UntrustedToolResult
from autobrain.models import (
    CoverageCompleteness,
    CoverageRecord,
    NormalizedDocument,
    SourceKind,
    Status,
)

_PROMPT_LIKE = re.compile(
    r"\b(ignore|disregard|override|follow|system prompt|assistant|call a (?:write|tool))\b",
    re.IGNORECASE,
)
_UNSAFE_ACCESS_METADATA = re.compile(
    r"(?:write|create|update|delete|insert|patch|mutat|token|secret|password|bearer|api[_ -]?key)",
    re.IGNORECASE,
)


class NotionMcpClient(Protocol):
    """Minimal MCP boundary used by the crawler and deterministic fakes."""

    async def call(self, name: str, arguments: dict[str, Any]) -> UntrustedToolResult: ...


class NotionMcpError(RuntimeError):
    """A recoverable or terminal error returned by the Notion MCP surface."""


class NotionCapabilityError(NotionMcpError):
    """The authenticated Notion MCP session cannot provide a required capability."""

    status: ClassVar[Status] = Status.CAPABILITY_UNAVAILABLE


class NotionSearchCapabilityError(NotionCapabilityError):
    """The authenticated Notion MCP session cannot search."""


class NotionUpgradeRequiredError(NotionCapabilityError):
    """The Notion MCP surface requires an unavailable upgrade."""

    terminal: ClassVar[bool] = True


@dataclass(frozen=True)
class NotionCrawlResult:
    documents: list[NormalizedDocument]
    coverage: CoverageRecord
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Counters:
    discovered: int = 0
    fetched: int = 0
    denied: int = 0
    truncated: int = 0
    unsupported: int = 0


class NotionCrawler:
    """Crawl only search-discovered Notion entities through read-only MCP tools."""

    def __init__(
        self,
        client: NotionMcpClient,
        *,
        call: Callable[[str, dict[str, Any]], Awaitable[UntrustedToolResult]] | None = None,
    ) -> None:
        self.client = client
        self._call = call or client.call

    async def crawl(self) -> NotionCrawlResult:
        try:
            identity = await self._fetch("self")
        except NotionMcpError as exc:
            if "returned no JSON object" in str(exc):
                raise NotionSearchCapabilityError(
                    "Notion MCP identity/tool access is unavailable"
                ) from exc
            raise
        access = _tool_access(identity.get("current_tool_access"))
        if not any(name in access for name in ("notion-search", "notion_search", "search")):
            raise NotionSearchCapabilityError("Notion MCP search capability is unavailable")
        if not any(name in access for name in ("notion-fetch", "notion_fetch", "fetch")):
            raise NotionCapabilityError("Notion MCP fetch capability is unavailable")

        counters = _Counters()
        warnings: list[str] = []
        identity_provenance = _identity_provenance(identity)
        discovered: list[dict[str, Any]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            arguments = {} if cursor is None else {"cursor": cursor}
            try:
                payload = await self._call_payload("notion-search", arguments)
            except NotionMcpError as exc:
                if cursor is None:
                    raise
                counters.unsupported += 1
                warnings.append(f"search pagination cursor {cursor} was invalid or expired: {exc}")
                break
            results = _as_dicts(payload.get("results"))
            discovered.extend(results)
            cursor_value = payload.get("next_cursor")
            if not isinstance(cursor_value, str) or not cursor_value:
                break
            if cursor_value in seen_cursors:
                counters.unsupported += 1
                warnings.append(f"search pagination repeated cursor {cursor_value}")
                break
            seen_cursors.add(cursor_value)
            cursor = cursor_value

        documents: list[NormalizedDocument] = []
        seen: set[str] = set()
        for entity in discovered:
            entity_id = _string(entity.get("id"))
            if entity_id is None or entity_id in seen:
                continue
            seen.add(entity_id)
            counters.discovered += 1
            try:
                payload = await self._fetch_entity(entity_id)
            except NotionMcpError as exc:
                counters.denied += 1
                warnings.append(f"{entity_id}: {exc}")
                continue
            counters.fetched += 1
            document, unknown_ids, entity_warnings = _document_from_payload(
                payload, crawl_provenance=identity_provenance
            )
            warnings.extend(f"{entity_id}: {warning}" for warning in entity_warnings)
            if payload.get("truncated") and not unknown_ids:
                counters.unsupported += 1
                warnings.append(f"{entity_id}: truncated response omitted unknown_block_ids")
            if unknown_ids:
                counters.truncated += 1
                related: list[str] = []
                recovered_text: list[str] = []
                pending = list(unknown_ids)
                visited_blocks: set[str] = set()
                while pending:
                    unknown_id = pending.pop(0)
                    if unknown_id in visited_blocks:
                        continue
                    visited_blocks.add(unknown_id)
                    related.append(_source_id_for("block", unknown_id))
                    try:
                        block = await self._fetch_entity(unknown_id)
                    except NotionMcpError as exc:
                        counters.unsupported += 1
                        warnings.append(
                            f"{entity_id}: inaccessible unknown block {unknown_id}: {exc}"
                        )
                        continue
                    block_text = _text_from_payload(block)
                    if block_text:
                        recovered_text.append(block_text)
                    nested_ids = [
                        value
                        for value in _as_list(block.get("unknown_block_ids"))
                        if isinstance(value, str)
                    ]
                    if block.get("truncated") and not nested_ids:
                        counters.unsupported += 1
                        warnings.append(
                            f"{entity_id}: truncated block {unknown_id} omitted unknown_block_ids"
                        )
                    pending.extend(nested_ids)
                document = document.model_copy(
                    update={
                        "text": _join_text([document.text, *recovered_text]),
                        "related_source_ids": sorted({*document.related_source_ids, *related}),
                        "content_hash": _hash_text(_join_text([document.text, *recovered_text])),
                    }
                )
            documents.append(document)

        completeness = (
            CoverageCompleteness.UNKNOWN
            if counters.unsupported
            else (
                CoverageCompleteness.EXHAUSTIVE
                if identity.get("coverage") == CoverageCompleteness.EXHAUSTIVE
                else CoverageCompleteness.SEARCH_DISCOVERED
            )
        )
        coverage = CoverageRecord(
            source=SourceKind.NOTION_PAGE,
            completeness=completeness,
            discovered=counters.discovered,
            fetched=counters.fetched,
            denied=counters.denied,
            truncated=counters.truncated,
            unsupported=counters.unsupported,
            crawl_provenance=identity_provenance,
        )
        return NotionCrawlResult(documents=documents, coverage=coverage, warnings=warnings)

    async def _fetch(self, entity_id: str) -> dict[str, Any]:
        return await self._call_payload("notion-fetch", {"id": entity_id})

    async def _fetch_entity(self, entity_id: str) -> dict[str, Any]:
        return await self._fetch(entity_id)

    async def _call_payload(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self._call(name, arguments)
        if result.is_error:
            decoded = _decode_content(result.content)
            error_code: object = decoded
            if isinstance(decoded, dict):
                decoded_payload = cast(dict[str, Any], decoded)
                error_code = decoded_payload.get("error", decoded_payload.get("code"))
            if isinstance(error_code, dict):
                nested_error = cast(dict[str, Any], error_code)
                error_code = nested_error.get("code", nested_error.get("message"))
            if error_code == "upgrade_required" or "upgrade_required" in str(error_code):
                raise NotionUpgradeRequiredError("Notion MCP upgrade_required")
            message = (
                str(error_code)
                if isinstance(error_code, str | int | float)
                else _text_from_content(result.content)
            ) or "Notion MCP call failed"
            raise NotionMcpError(message)
        payload = _decode_content(result.content)
        if not isinstance(payload, dict):
            raise NotionMcpError(f"{name} returned no JSON object")
        payload = cast(dict[str, Any], payload)
        error_code = payload.get("error", payload.get("code"))
        if payload.get("upgrade_required") or error_code == "upgrade_required":
            raise NotionUpgradeRequiredError("Notion MCP upgrade_required")
        if payload.get("success") is False or payload.get("error") is not None:
            error = payload.get("error")
            if isinstance(error, dict):
                error = cast(dict[str, Any], error)
                code = error.get("code")
                if isinstance(code, str):
                    if code == "upgrade_required":
                        raise NotionUpgradeRequiredError("Notion MCP upgrade_required")
                    raise NotionMcpError(code)
            raise NotionMcpError(str(error or "Notion MCP reported failure"))
        return payload


def _decode_content(content: object) -> object:
    if isinstance(content, dict):
        return cast(dict[str, Any], content)
    if isinstance(content, list):
        for raw_item in cast(list[object], content):
            if isinstance(raw_item, dict):
                item = cast(dict[str, Any], raw_item)
                if not isinstance(item.get("text"), str):
                    continue
                text = item["text"]
                try:
                    return cast(object, json.loads(text))
                except json.JSONDecodeError:
                    return text
    return cast(object, content)


def _text_from_content(content: object) -> str:
    decoded = _decode_content(content)
    return decoded if isinstance(decoded, str) else ""


def _text_from_payload(payload: dict[str, Any]) -> str:
    for key in ("markdown", "text", "content"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return _join_text([item for item in cast(list[object], value) if isinstance(item, str)])
    return ""


def _document_from_payload(
    payload: dict[str, Any],
    *,
    crawl_provenance: dict[str, str] | None = None,
) -> tuple[NormalizedDocument, list[str], list[str]]:
    entity_id = _string(payload.get("id")) or "unknown"
    object_type = _string(payload.get("object")) or "page"
    source_kind = (
        SourceKind.NOTION_DATA_SOURCE
        if object_type in {"data_source", "database"}
        else SourceKind.NOTION_PAGE
    )
    source_id = _source_id_for(
        "data-source" if source_kind is SourceKind.NOTION_DATA_SOURCE else "page", entity_id
    )
    text = _text_from_payload(payload)
    warnings = (
        ["prompt-like source instructions preserved as inert data"]
        if _PROMPT_LIKE.search(text)
        else []
    )
    unknown_ids = [
        block_id
        for block_id in _as_list(payload.get("unknown_block_ids"))
        if isinstance(block_id, str)
    ]
    related = [_source_id_for("block", block_id) for block_id in unknown_ids]
    metadata = _metadata_from_payload(payload)
    document = NormalizedDocument(
        source_id=source_id,
        source_kind=source_kind,
        canonical_url=_url_for(payload, entity_id),
        title=_title(payload.get("title")) or entity_id,
        text=text,
        content_hash=_hash_text(text),
        created_at=_parse_time(payload.get("created_time")),
        updated_at=_parse_time(payload.get("last_edited_time")),
        container=_parent_label(payload.get("parent")),
        authors=_authors(payload),
        related_source_ids=related,
        crawl_provenance={
            "connector": "notion-mcp",
            "discovery": "search",
            **(crawl_provenance or {}),
        },
        warnings=warnings,
        metadata=metadata,
    )
    return document, unknown_ids, warnings


def _identity_provenance(identity: dict[str, Any]) -> dict[str, str]:
    provenance: dict[str, str] = {}
    aliases = {
        "id": "authenticated_user_id",
        "user_id": "authenticated_user_id",
        "name": "authenticated_user_name",
        "user_name": "authenticated_user_name",
        "workspace_id": "workspace_id",
        "workspace_name": "workspace_name",
    }
    for source, target in aliases.items():
        value = identity.get(source)
        if isinstance(value, str | int | bool):
            provenance[target] = str(value)
    for key, value in (("user", identity.get("user")), ("workspace", identity.get("workspace"))):
        if isinstance(value, dict):
            nested = cast(dict[str, Any], value)
            for source, target in (
                ("id", f"{key}_id" if key == "workspace" else "authenticated_user_id"),
                ("name", f"{key}_name" if key == "workspace" else "authenticated_user_name"),
            ):
                nested_value = nested.get(source)
                if isinstance(nested_value, str | int | bool):
                    provenance[target] = str(nested_value)
    current_tool_access = _safe_tool_access_metadata(identity.get("current_tool_access"))
    if current_tool_access is not None:
        provenance["current_tool_access"] = current_tool_access
    return dict(sorted(provenance.items()))


def _metadata_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    parent = payload.get("parent")
    metadata: dict[str, str] = {}
    if isinstance(parent, dict):
        parent = cast(dict[str, Any], parent)
        for key in ("type", "page_id", "database_id", "data_source_id"):
            value = parent.get(key)
            if isinstance(value, str | int | bool):
                metadata[f"parent_{key}"] = str(value)
    links = payload.get("links")
    if isinstance(links, list):
        links = cast(list[object], links)
        metadata["links"] = ",".join(sorted(str(link) for link in links if isinstance(link, str)))
    return metadata


def _parent_label(parent: object) -> str | None:
    if not isinstance(parent, dict):
        return None
    parent = cast(dict[str, Any], parent)
    for key in ("page_id", "database_id", "data_source_id", "type"):
        value = parent.get(key)
        if isinstance(value, str):
            return value
    return None


def _url_for(payload: dict[str, Any], entity_id: str) -> str:
    url = cast(object, payload.get("url"))
    return (
        url
        if isinstance(url, str) and url.startswith(("http://", "https://"))
        else f"https://notion.so/{entity_id}"
    )


def _source_id_for(kind: str, entity_id: str) -> str:
    return f"notion:{kind}:{entity_id}"


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _title(value: object) -> str | None:
    if isinstance(value, str):
        return value or None
    if not isinstance(value, list):
        return None
    parts: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, dict):
            continue
        rich_text = cast(dict[str, Any], item)
        plain_text = rich_text.get("plain_text")
        if isinstance(plain_text, str):
            parts.append(plain_text)
    title = "".join(parts)
    return title or None


def _authors(payload: dict[str, Any]) -> list[str]:
    authors: list[str] = []
    for key in ("created_by", "last_edited_by", "author"):
        value = payload.get(key)
        if isinstance(value, str):
            authors.append(value)
        elif isinstance(value, dict):
            identity = cast(dict[str, Any], value)
            label = identity.get("id") or identity.get("name")
            if isinstance(label, str):
                authors.append(label)
    return sorted(set(authors))


def _tool_access(value: object) -> set[str]:
    if isinstance(value, list):
        return {item for item in cast(list[object], value) if isinstance(item, str)}
    if isinstance(value, dict):
        access = cast(dict[str, Any], value)
        return {key for key, enabled in access.items() if enabled is True}
    return set()


def _safe_tool_access_metadata(value: object) -> str | None:
    """Serialize read-only access names without copying sensitive tool metadata."""
    if isinstance(value, list):
        access = cast(list[object], value)
        if not all(isinstance(item, str) for item in access):
            return None
        safe_access = [
            item for item in cast(list[str], access) if not _UNSAFE_ACCESS_METADATA.search(item)
        ]
        return json.dumps(safe_access, separators=(",", ":")) if safe_access else None
    if isinstance(value, dict):
        access = cast(dict[object, object], value)
        if not all(
            isinstance(key, str) and isinstance(enabled, bool) for key, enabled in access.items()
        ):
            return None
        safe_access = {
            key: enabled
            for key, enabled in cast(dict[str, bool], access).items()
            if not _UNSAFE_ACCESS_METADATA.search(key)
        }
        return (
            json.dumps(safe_access, sort_keys=True, separators=(",", ":")) if safe_access else None
        )
    return None


def _as_list(value: object) -> list[object]:
    return cast(list[object], value) if isinstance(value, list) else []


def _as_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        cast(dict[str, Any], item) for item in cast(list[object], value) if isinstance(item, dict)
    ]


def _join_text(parts: list[str]) -> str:
    return "\n\n".join(part for part in parts if part)


def _hash_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
