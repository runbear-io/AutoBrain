"""Synthetic, credential-free source replay inputs for connector tests."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile


def write_official_shaped_slack_export(path: Path) -> Path:
    """Write a small Slack Workspace Export-shaped ZIP with synthetic content."""
    with ZipFile(path, "w") as archive:
        _write_json(
            archive,
            "team.json",
            {"id": "T-SYNTHETIC", "name": "Example", "domain": "example"},
        )
        _write_json(
            archive,
            "users.json",
            [
                {
                    "id": "U-SYNTHETIC-1",
                    "name": "alex",
                    "real_name": "Alex Example",
                    "profile": {"display_name": "Alex Example"},
                },
                {
                    "id": "U-SYNTHETIC-2",
                    "name": "sam",
                    "real_name": "Sam Example",
                    "profile": {"display_name": "Sam Example"},
                },
            ],
        )
        _write_json(
            archive,
            "channels.json",
            [{"id": "C-SYNTHETIC", "name": "general", "is_archived": False}],
        )
        _write_json(archive, "groups.json", [])
        _write_json(archive, "dms.json", [])
        _write_json(archive, "mpims.json", [])
        _write_json(
            archive,
            "general/2026-08-30.json",
            [
                {
                    "type": "message",
                    "user": "U-SYNTHETIC-1",
                    "text": "Synthetic launch checklist: verify the rollback plan.",
                    "ts": "1724976000.000001",
                },
                {
                    "type": "message",
                    "user": "U-SYNTHETIC-2",
                    "text": "The rollback owner is the release captain.",
                    "ts": "1724976001.000002",
                    "thread_ts": "1724976000.000001",
                },
            ],
        )
    return path


def write_notion_snapshot(path: Path) -> Path:
    """Write a valid bounded Notion MCP snapshot containing synthetic content."""
    payload = {
        "schema_version": 1,
        "source": "notion-mcp-snapshot",
        "fetched_at": "2026-08-30T12:00:00Z",
        "documents": [
            {
                "page_id": "synthetic-page-1",
                "page_url": "https://www.notion.so/synthetic-page-1",
                "title": "Synthetic release notes",
                "fetched_at": "2026-08-30T11:59:00Z",
                "content": "Synthetic release notes: rollback owner is the release captain.",
                "metadata": {"origin": "synthetic-test-fixture"},
            }
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_json(archive: ZipFile, name: str, payload: object) -> None:
    archive.writestr(name, json.dumps(payload, sort_keys=True))
