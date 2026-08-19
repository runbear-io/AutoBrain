from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from autobrain.connectors.slack_export import (
    SlackExportConnector,
    SlackExportError,
    inspect_slack_export,
)
from autobrain.models import SourceKind, Status


def _write_json(archive: ZipFile, name: str, payload: object) -> None:
    archive.writestr(name, json.dumps(payload, ensure_ascii=False))


def _valid_export(path: Path) -> Path:
    with ZipFile(path, "w") as archive:
        _write_json(
            archive,
            "team.json",
            {"id": "T1", "name": "Acme", "domain": "acme"},
        )
        _write_json(
            archive,
            "users.json",
            [
                {
                    "id": "U1",
                    "name": "ada",
                    "profile": {"display_name": "Ada Lovelace", "real_name": "Ada"},
                },
                {
                    "id": "U2",
                    "name": "grace",
                    "profile": {"display_name": "", "real_name": "Grace Hopper"},
                },
            ],
        )
        _write_json(
            archive,
            "channels.json",
            [{"id": "C1", "name": "general", "is_archived": False}],
        )
        _write_json(
            archive,
            "general/2026-08-18.json",
            [
                {
                    "type": "message",
                    "user": "U2",
                    "text": "How should memory retention work?",
                    "ts": "1700000000.000001",
                    "files": [
                        {
                            "id": "F1",
                            "name": "retention-guide.pdf",
                            "url_private": "https://files.slack.com/files-pri/T1-F1/guide.pdf",
                        }
                    ],
                },
                {
                    "type": "message",
                    "user": "U1",
                    "text": "Keep verified decisions for 90 days.",
                    "ts": "1700000001.000002",
                    "thread_ts": "1700000000.000001",
                },
            ],
        )
    return path


def test_slack_export_resolves_channels_users_threads_and_file_links(tmp_path: Path) -> None:
    archive_path = _valid_export(tmp_path / "slack-export.zip")

    summary = inspect_slack_export(archive_path)
    result = asyncio.run(SlackExportConnector(archive_path).crawl())

    assert summary.workspace_name == "Acme"
    assert summary.channel_count == 1
    assert summary.user_count == 2
    assert summary.message_count == 2
    assert result.status is Status.OK
    assert [document.source_id for document in result.documents] == [
        "slack-message:C1:1700000000.000001",
        "slack-message:C1:1700000001.000002",
    ]
    root, reply = result.documents
    assert root.source_kind is SourceKind.SLACK_MESSAGE
    assert root.user_name == "Grace Hopper"
    assert root.canonical_url == "https://acme.slack.com/archives/C1/p1700000000000001"
    assert root.metadata["file_names"] == "retention-guide.pdf"
    assert root.metadata["file_urls"] == "https://files.slack.com/files-pri/T1-F1/guide.pdf"
    assert reply.source_kind is SourceKind.SLACK_THREAD
    assert reply.user_name == "Ada Lovelace"
    assert reply.parent_source_id == root.source_id


def test_slack_export_output_is_deterministic(tmp_path: Path) -> None:
    archive_path = _valid_export(tmp_path / "slack-export.zip")

    first = asyncio.run(SlackExportConnector(archive_path).crawl())
    second = asyncio.run(SlackExportConnector(archive_path).crawl())

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


@pytest.mark.parametrize("member_name", ["../escape.json", "/absolute.json"])
def test_slack_export_rejects_unsafe_member_paths(tmp_path: Path, member_name: str) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(member_name, "{}")

    with pytest.raises(SlackExportError, match="unsafe archive member"):
        inspect_slack_export(archive_path)


def test_slack_export_rejects_symlink_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "symlink.zip"
    symlink = ZipInfo("general/link.json")
    symlink.create_system = 3
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(symlink, "target")

    with pytest.raises(SlackExportError, match="symlink"):
        inspect_slack_export(archive_path)


def test_slack_export_rejects_oversized_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autobrain.connectors import slack_export_archive

    archive_path = _valid_export(tmp_path / "oversized.zip")
    monkeypatch.setattr(slack_export_archive, "MAX_MEMBER_BYTES", 8)

    with pytest.raises(SlackExportError, match="too large"):
        inspect_slack_export(archive_path)


def test_slack_export_rejects_malformed_json(tmp_path: Path) -> None:
    archive_path = tmp_path / "malformed.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("channels.json", "[")

    with pytest.raises(SlackExportError, match="invalid JSON"):
        inspect_slack_export(archive_path)


def test_slack_export_rejects_archives_without_messages(tmp_path: Path) -> None:
    archive_path = tmp_path / "empty.zip"
    with ZipFile(archive_path, "w") as archive:
        _write_json(archive, "team.json", {"id": "T1", "domain": "acme"})
        _write_json(archive, "users.json", [])
        _write_json(archive, "channels.json", [{"id": "C1", "name": "general"}])

    with pytest.raises(SlackExportError, match="no Slack messages"):
        inspect_slack_export(archive_path)
