from __future__ import annotations

import json
import stat
from pathlib import Path
from zipfile import ZipFile

from autobrain.contracts import SourceConnectionState, SourceConnectionStatusProjectionV1
from autobrain.paths import AutoBrainPaths
from autobrain.source_store import SlackSourceState, SlackSourceStore


def _write_json(archive: ZipFile, name: str, payload: object) -> None:
    archive.writestr(name, json.dumps(payload))


def _export(path: Path) -> Path:
    with ZipFile(path, "w") as archive:
        _write_json(archive, "team.json", {"id": "T1", "name": "Acme", "domain": "acme"})
        _write_json(archive, "users.json", [{"id": "U1", "name": "ada"}])
        _write_json(archive, "channels.json", [{"id": "C1", "name": "general"}])
        _write_json(
            archive,
            "general/2026-08-19.json",
            [{"type": "message", "user": "U1", "text": "What changed?", "ts": "1.1"}],
        )
    return path


def test_slack_source_store_saves_confined_0600_reference(tmp_path: Path) -> None:
    paths = AutoBrainPaths.from_home(tmp_path)
    archive_path = _export(tmp_path / "slack-export.zip")
    store = SlackSourceStore(paths.sources)

    configured = store.configure_export(archive_path)
    status = store.status()

    assert configured.archive_path == str(archive_path.resolve())
    assert configured.summary.message_count == 1
    assert stat.S_IMODE(store.config_path.stat().st_mode) == 0o600
    assert status.state is SlackSourceState.READY
    assert status.ready is True
    assert status.archive_path == archive_path.resolve()
    assert status.config == configured
    assert isinstance(status.projection, SourceConnectionStatusProjectionV1)
    assert status.projection.state is SourceConnectionState.READY


def test_slack_source_store_detects_changed_archive(tmp_path: Path) -> None:
    paths = AutoBrainPaths.from_home(tmp_path)
    archive_path = _export(tmp_path / "slack-export.zip")
    store = SlackSourceStore(paths.sources)
    store.configure_export(archive_path)

    with ZipFile(archive_path, "a") as archive:
        archive.writestr("notes.txt", "changed")

    status = store.status()

    assert status.state is SlackSourceState.ARCHIVE_CHANGED
    assert status.ready is False
    assert status.projection.state is SourceConnectionState.FAILED
    assert status.projection.diagnostics == ["archive_changed"]


def test_slack_source_store_detects_deleted_archive(tmp_path: Path) -> None:
    paths = AutoBrainPaths.from_home(tmp_path)
    archive_path = _export(tmp_path / "slack-export.zip")
    store = SlackSourceStore(paths.sources)
    store.configure_export(archive_path)
    archive_path.unlink()

    status = store.status()

    assert status.state is SlackSourceState.ARCHIVE_MISSING
    assert status.ready is False
    assert status.projection.state is SourceConnectionState.FAILED
    assert status.projection.diagnostics == ["archive_missing"]


def test_slack_source_store_remove_returns_to_not_configured(tmp_path: Path) -> None:
    paths = AutoBrainPaths.from_home(tmp_path)
    archive_path = _export(tmp_path / "slack-export.zip")
    store = SlackSourceStore(paths.sources)
    store.configure_export(archive_path)

    store.remove()

    status = store.status()
    assert status.state is SlackSourceState.NOT_CONFIGURED
    assert status.projection.state is SourceConnectionState.AWAITING_LOCAL_INPUT
    assert status.projection.diagnostics == ["archive_not_configured"]
    assert not store.config_path.exists()
