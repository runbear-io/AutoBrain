"""Validate and parse official Slack export archive structure."""

from __future__ import annotations

import json
import stat
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile, ZipInfo

from autobrain.connectors.slack import RawSlackDocument
from autobrain.connectors.slack_export_messages import (
    array_value,
    message_document,
    object_value,
    text_value,
)
from autobrain.connectors.slack_export_types import (
    SlackExportError,
    SlackExportSourceChangedError,
    SlackExportSummary,
)

MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024 * 1024
MAX_MEMBER_COUNT = 100_000
_CATALOG_NAMES = frozenset(
    {"team.json", "users.json", "channels.json", "groups.json", "dms.json", "mpims.json"}
)


def archive_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as archive_file:
        for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_slack_export(
    path: Path,
) -> tuple[SlackExportSummary, tuple[RawSlackDocument, ...]]:
    if path.is_symlink():
        raise SlackExportError("Slack export input cannot be a symlink")
    if not path.is_file():
        raise SlackExportError(f"Slack export not found: {path}")
    try:
        digest = archive_sha256(path)
    except OSError as error:
        raise SlackExportError(f"cannot read Slack export: {path}") from error
    try:
        with ZipFile(path) as archive:
            members = _validated_members(archive)
            root = _archive_root(members)
            team = object_value(_load_optional(archive, members, root / "team.json"))
            users = _user_names(array_value(_load_optional(archive, members, root / "users.json")))
            channels = _channel_catalog(archive, members, root)
            documents = _message_documents(
                archive=archive,
                members=members,
                root=root,
                channels=channels,
                users=users,
                team=team,
                digest=digest,
            )
        try:
            final_digest = archive_sha256(path)
        except OSError as error:
            raise SlackExportSourceChangedError(
                f"Slack export changed during parse: {path}"
            ) from error
        if final_digest != digest:
            raise SlackExportSourceChangedError(f"Slack export changed during parse: {path}")
    except BadZipFile as error:
        raise SlackExportError("invalid Slack export ZIP") from error
    except OSError as error:
        raise SlackExportError(f"cannot read Slack export: {path}") from error
    if not documents:
        raise SlackExportError("Slack export contains no Slack messages")
    file_links = sum(bool(document.metadata.get("file_urls")) for document in documents)
    summary = SlackExportSummary(
        archive_path=str(path),
        archive_sha256=digest,
        workspace_id=text_value(team.get("id")),
        workspace_name=text_value(team.get("name")),
        workspace_domain=text_value(team.get("domain")),
        channel_count=len(channels),
        user_count=len(users),
        message_count=len(documents),
        file_link_count=file_links,
    )
    return summary, tuple(sorted(documents, key=lambda document: document.source_id))


def _validated_members(archive: ZipFile) -> dict[PurePosixPath, ZipInfo]:
    members: dict[PurePosixPath, ZipInfo] = {}
    total_size = 0
    infos = archive.infolist()
    if len(infos) > MAX_MEMBER_COUNT:
        raise SlackExportError("Slack export contains too many archive members")
    for info in infos:
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise SlackExportError(f"unsafe archive member: {info.filename}")
        if info.flag_bits & 0x1:
            raise SlackExportError(f"encrypted archive member: {info.filename}")
        if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
            raise SlackExportError(f"symlink archive member: {info.filename}")
        if info.file_size > MAX_MEMBER_BYTES:
            raise SlackExportError(f"archive member is too large: {info.filename}")
        total_size += info.file_size
        if total_size > MAX_TOTAL_BYTES:
            raise SlackExportError("Slack export is too large")
        if member in members:
            raise SlackExportError(f"duplicate archive member: {info.filename}")
        members[member] = info
    return members


def _archive_root(members: Mapping[PurePosixPath, ZipInfo]) -> PurePosixPath:
    catalogs = [path for path in members if path.name == "channels.json"]
    if len(catalogs) != 1:
        raise SlackExportError("Slack export must contain one channels.json catalog")
    return catalogs[0].parent


def _load_optional(
    archive: ZipFile,
    members: Mapping[PurePosixPath, ZipInfo],
    path: PurePosixPath,
) -> object:
    info = members.get(path)
    if info is None:
        return None
    try:
        with archive.open(info) as source:
            return json.load(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SlackExportError(f"invalid JSON in {path}") from error


def _channel_catalog(
    archive: ZipFile,
    members: Mapping[PurePosixPath, ZipInfo],
    root: PurePosixPath,
) -> dict[str, tuple[str, str, bool]]:
    catalog: dict[str, tuple[str, str, bool]] = {}
    for filename, channel_type in (
        ("channels.json", "public_channel"),
        ("groups.json", "private_channel"),
        ("dms.json", "im"),
        ("mpims.json", "mpim"),
    ):
        for record in array_value(_load_optional(archive, members, root / filename)):
            item = object_value(record)
            channel_id = text_value(item.get("id"))
            name = text_value(item.get("name")) or channel_id
            if channel_id and name:
                catalog[name] = (channel_id, channel_type, bool(item.get("is_archived", False)))
    if not catalog:
        raise SlackExportError("Slack export contains no channel catalog")
    return catalog


def _user_names(records: list[object]) -> dict[str, str]:
    users: dict[str, str] = {}
    for record in records:
        user = object_value(record)
        user_id = text_value(user.get("id"))
        profile = object_value(user.get("profile"))
        name = (
            text_value(profile.get("display_name"))
            or text_value(profile.get("real_name"))
            or text_value(user.get("real_name"))
            or text_value(user.get("name"))
        )
        if user_id:
            users[user_id] = name or user_id
    return users


def _message_documents(
    *,
    archive: ZipFile,
    members: Mapping[PurePosixPath, ZipInfo],
    root: PurePosixPath,
    channels: Mapping[str, tuple[str, str, bool]],
    users: Mapping[str, str],
    team: Mapping[str, object],
    digest: str,
) -> list[RawSlackDocument]:
    documents: list[RawSlackDocument] = []
    for path in sorted(members):
        relative = path.relative_to(root)
        if path.name in _CATALOG_NAMES or len(relative.parts) < 2 or path.suffix != ".json":
            continue
        channel_name = relative.parts[-2]
        channel = channels.get(channel_name)
        if channel is None:
            continue
        for record in array_value(_load_optional(archive, members, path)):
            document = message_document(
                object_value(record), channel_name, channel, users, team, digest
            )
            if document is not None:
                documents.append(document)
    return documents
