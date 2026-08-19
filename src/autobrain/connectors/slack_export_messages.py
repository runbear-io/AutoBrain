"""Convert typed Slack export message JSON into RawSlackDocument records."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256

from pydantic import TypeAdapter, ValidationError

from autobrain.connectors.slack import RawSlackDocument
from autobrain.models import SourceKind

_OBJECT_ADAPTER = TypeAdapter(dict[str, object])
_ARRAY_ADAPTER = TypeAdapter(list[object])


def message_document(
    message: Mapping[str, object],
    channel_name: str,
    channel: tuple[str, str, bool],
    users: Mapping[str, str],
    team: Mapping[str, object],
    digest: str,
) -> RawSlackDocument | None:
    timestamp = text_value(message.get("ts"))
    if not timestamp:
        return None
    file_names, file_urls = _file_metadata(array_value(message.get("files")))
    text = text_value(message.get("text")) or (
        f"[Files] {', '.join(file_names)}" if file_names else ""
    )
    if not text:
        return None
    channel_id, channel_type, archived = channel
    thread_ts = text_value(message.get("thread_ts"))
    parent_ts = thread_ts if thread_ts and thread_ts != timestamp else None
    source_id = f"slack-message:{channel_id}:{timestamp}"
    user_id = text_value(message.get("user"))
    edited = object_value(message.get("edited"))
    metadata = {"archive_sha256": digest}
    if file_names:
        metadata["file_names"] = ",".join(file_names)
    if file_urls:
        metadata["file_urls"] = ",".join(file_urls)
    return RawSlackDocument(
        source_id=source_id,
        source_kind=SourceKind.SLACK_THREAD if parent_ts else SourceKind.SLACK_MESSAGE,
        canonical_url=_slack_url(team, channel_id, timestamp),
        title=f"#{channel_name} at {timestamp}",
        text=text,
        content_hash=sha256(text.encode("utf-8")).hexdigest(),
        created_at=_timestamp(timestamp),
        channel_id=channel_id,
        channel_name=channel_name,
        channel_type=channel_type,
        channel_archived=archived,
        message_ts=timestamp,
        thread_ts=thread_ts,
        parent_source_id=(
            f"slack-message:{channel_id}:{parent_ts}" if parent_ts is not None else None
        ),
        user_id=user_id,
        user_name=users.get(user_id or ""),
        bot=bool(message.get("bot_id")) or message.get("subtype") == "bot_message",
        edited=bool(edited),
        deleted=message.get("subtype") == "message_deleted",
        untrusted=True,
        crawl_provenance={"connector": "slack-export", "archive_sha256": digest},
        metadata=metadata,
    )


def array_value(value: object) -> list[object]:
    try:
        return _ARRAY_ADAPTER.validate_python(value, strict=True)
    except ValidationError:
        return []


def object_value(value: object) -> dict[str, object]:
    try:
        return _OBJECT_ADAPTER.validate_python(value, strict=True)
    except ValidationError:
        return {}


def text_value(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _file_metadata(records: list[object]) -> tuple[list[str], list[str]]:
    names: list[str] = []
    urls: list[str] = []
    for record in records:
        file_record = object_value(record)
        name = text_value(file_record.get("name")) or text_value(file_record.get("title"))
        url = text_value(file_record.get("url_private")) or text_value(file_record.get("permalink"))
        if name:
            names.append(name)
        if url:
            urls.append(url)
    return names, urls


def _slack_url(team: Mapping[str, object], channel_id: str, timestamp: str) -> str:
    compact_timestamp = timestamp.replace(".", "")
    domain = text_value(team.get("domain"))
    if domain:
        return f"https://{domain}.slack.com/archives/{channel_id}/p{compact_timestamp}"
    team_id = text_value(team.get("id")) or "workspace"
    return f"https://app.slack.com/client/{team_id}/{channel_id}/thread/{compact_timestamp}"


def _timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(value), tz=UTC)
    except ValueError:
        return None
