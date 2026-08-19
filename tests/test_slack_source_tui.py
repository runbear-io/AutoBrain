from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from zipfile import ZipFile

import pytest

from autobrain.auth.models import AuthStatusReport, ConnectionStatus, Provider
from autobrain.models import ConnectionState, Status
from autobrain.source_store import SlackSourceStore
from autobrain.subscription import SubscriptionStatus
from autobrain.tui_runtime import connection_snapshot, run_connection_flow


class _ConnectionManager:
    def status(self) -> AuthStatusReport:
        return AuthStatusReport(
            connections=(
                ConnectionStatus(
                    provider=Provider.SLACK,
                    state=ConnectionState.DISCONNECTED,
                    status=Status.MCP_AUTH_UNAVAILABLE,
                ),
                ConnectionStatus(
                    provider=Provider.NOTION,
                    state=ConnectionState.DISCONNECTED,
                    status=Status.MCP_AUTH_UNAVAILABLE,
                ),
            )
        )


class _Subscription:
    def status(self) -> SubscriptionStatus:
        return SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE


class _Screen:
    def refresh(self) -> None:
        return None


def _slack_export(path: Path) -> Path:
    with ZipFile(path, "w") as archive:
        archive.writestr("team.json", '{"id":"T1","name":"Acme","domain":"acme"}')
        archive.writestr("users.json", '[{"id":"U1","name":"ada"}]')
        archive.writestr("channels.json", '[{"id":"C1","name":"general"}]')
        archive.writestr(
            "general/2026-08-19.json",
            json.dumps([{"type": "message", "user": "U1", "text": "What changed?", "ts": "1.1"}]),
        )
    return path


def test_connection_snapshot_marks_valid_slack_export_ready(tmp_path: Path) -> None:
    store = SlackSourceStore(tmp_path / "sources")
    config = store.configure_export(_slack_export(tmp_path / "slack-export.zip"))

    snapshot = connection_snapshot(
        manager=_ConnectionManager(),  # type: ignore[arg-type]
        subscription_client=_Subscription(),  # type: ignore[arg-type]
        source_store=store,
    )

    assert snapshot.sources[Provider.SLACK] is ConnectionState.CONNECTED
    assert snapshot.source_details is not None
    assert snapshot.source_details[Provider.SLACK] == "export ready"
    assert snapshot.slack_export_path == Path(config.archive_path)
    assert snapshot.slack_export_sha256 == config.archive_sha256


def test_slack_connection_key_opens_source_setup_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []

    def runner(
        command: Sequence[str],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del check
        commands.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("autobrain.tui_runtime.curses.def_prog_mode", lambda: None)
    monkeypatch.setattr("autobrain.tui_runtime.curses.endwin", lambda: None)
    monkeypatch.setattr("autobrain.tui_runtime.curses.reset_prog_mode", lambda: None)

    run_connection_flow(_Screen(), Provider.SLACK, runner=runner)  # type: ignore[arg-type]

    assert commands[0][-2:] == ("source", "slack")
