from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from zipfile import ZipFile

import pytest

from autobrain.auth.models import AuthStatusReport, ConnectionStatus, Provider
from autobrain.contracts import SourceConnectionState, SourceConnectionStatusProjectionV1
from autobrain.models import ConnectionState, Status
from autobrain.source_store import SlackSourceStore
from autobrain.subscription import ProviderId, SubscriptionStatus
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


class _ConnectedSlackManager:
    def status(self) -> AuthStatusReport:
        return AuthStatusReport(
            connections=(
                ConnectionStatus(
                    provider=Provider.SLACK,
                    state=ConnectionState.CONNECTED,
                    status=Status.OK,
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
        manager=_ConnectionManager(),
        subscription_client=_Subscription(),
        source_store=store,
    )

    assert snapshot.subscription_provider is ProviderId.CODEX
    assert snapshot.sources[Provider.SLACK] is ConnectionState.CONNECTED
    assert snapshot.source_projections is not None
    assert snapshot.source_projections[Provider.SLACK].state is SourceConnectionState.READY
    assert snapshot.source_details is not None
    assert snapshot.source_details[Provider.SLACK] == "export ready"
    assert snapshot.slack_export_path == Path(config.archive_path)
    assert snapshot.slack_export_sha256 == config.archive_sha256


def test_connection_snapshot_does_not_override_changed_archive_as_connected(tmp_path: Path) -> None:
    store = SlackSourceStore(tmp_path / "sources")
    archive = _slack_export(tmp_path / "slack-export.zip")
    store.configure_export(archive)
    with ZipFile(archive, "a") as changed:
        changed.writestr("stale.txt", "dirty")

    snapshot = connection_snapshot(
        manager=_ConnectedSlackManager(),
        subscription_client=_Subscription(),
        source_store=store,
    )

    assert snapshot.sources[Provider.SLACK] is not ConnectionState.CONNECTED
    assert snapshot.source_projections is not None
    projection = snapshot.source_projections[Provider.SLACK]
    assert projection.state is SourceConnectionState.FAILED
    assert projection.ready is False
    assert "archive_changed" in projection.diagnostics
    assert snapshot.slack_export_path is None


def test_connection_snapshot_exposes_missing_archive_as_actionable_projection(
    tmp_path: Path,
) -> None:
    store = SlackSourceStore(tmp_path / "sources")
    archive = _slack_export(tmp_path / "slack-export.zip")
    store.configure_export(archive)
    archive.unlink()

    snapshot = connection_snapshot(
        manager=_ConnectionManager(),
        subscription_client=_Subscription(),
        source_store=store,
    )

    projections = snapshot.source_projections
    assert projections is not None
    projection: SourceConnectionStatusProjectionV1 = projections[Provider.SLACK]
    assert projection.state is SourceConnectionState.FAILED
    assert projection.diagnostics == ["archive_missing"]
    details = snapshot.source_details
    assert details is not None
    assert "no longer exists" in details[Provider.SLACK]


@pytest.mark.parametrize("provider", tuple(ProviderId))
def test_subscription_connection_flow_passes_exact_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: ProviderId,
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

    run_connection_flow(_Screen(), provider, runner=runner)  # type: ignore[arg-type]

    assert commands[0][-4:] == ("subscription", "setup", "--provider", provider.value)


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

    run_connection_flow(_Screen(), Provider.SLACK, runner=runner)  # type: ignore[arg-type]

    assert commands[0][-2:] == ("source", "slack")
