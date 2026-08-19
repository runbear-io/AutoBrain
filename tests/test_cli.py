import json
from pathlib import Path
from zipfile import ZipFile

from pytest import MonkeyPatch
from typer.testing import CliRunner

from autobrain.auth.models import Provider
from autobrain.cli import app
from autobrain.orchestration import RunConfig
from autobrain.source_store import SlackSourceState


def _slack_export(path: Path) -> Path:
    with ZipFile(path, "w") as archive:
        archive.writestr("team.json", '{"id":"T1","name":"Acme","domain":"acme"}')
        archive.writestr("users.json", '[{"id":"U1","name":"ada"}]')
        archive.writestr("channels.json", '[{"id":"C1","name":"general"}]')
        archive.writestr(
            "general/2026-08-19.json",
            '[{"type":"message","user":"U1","text":"What changed?","ts":"1.1"}]',
        )
    return path


def test_cli_exposes_doctor_subcommand() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout
    assert "source" in result.stdout


def test_doctor_json_is_machine_readable(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    assert '"schema_version": 1' in result.stdout
    assert "MISSING_PROVIDER" in result.stdout


def test_malformed_callback_port_is_a_typed_independent_check(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOBRAIN_CALLBACK_PORT", "not-a-port")
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["callback"]["status"] == "CAPABILITY_UNAVAILABLE"
    assert "integer from 1 to 65535" in checks["callback"]["detail"]
    assert checks["candidate_pins"]["status"] == "OK"
    assert checks["python"]["status"] == "OK"


def test_source_slack_export_configures_local_archive(tmp_path: Path) -> None:
    archive_path = _slack_export(tmp_path / "slack-export.zip")

    configured = CliRunner().invoke(
        app,
        ["source", "slack", "--export", str(archive_path)],
        env={"HOME": str(tmp_path)},
    )
    status = CliRunner().invoke(
        app,
        ["source", "status", "--json"],
        env={"HOME": str(tmp_path)},
    )

    assert configured.exit_code == 0
    assert "Slack export ready" in configured.stdout
    report = json.loads(status.stdout)
    assert report["state"] == SlackSourceState.READY.value
    assert report["ready"] is True
    assert report["config"]["summary"]["message_count"] == 1


def test_source_slack_interactive_defaults_to_export(tmp_path: Path) -> None:
    archive_path = _slack_export(tmp_path / "slack-export.zip")

    result = CliRunner().invoke(
        app,
        ["source", "slack"],
        input=f"\n{archive_path}\n",
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert "Slack export ready" in result.stdout


def test_source_slack_live_uses_existing_oauth_flow(monkeypatch: MonkeyPatch) -> None:
    authorized: list[str] = []

    def authorize(provider: Provider) -> None:
        authorized.append(provider.value)

    monkeypatch.setattr(
        "autobrain.source_cli._authorize_source",
        authorize,
    )

    result = CliRunner().invoke(app, ["source", "slack", "--live"])

    assert result.exit_code == 0
    assert authorized == ["slack"]


def test_source_slack_remove_clears_configured_export(tmp_path: Path) -> None:
    archive_path = _slack_export(tmp_path / "slack-export.zip")
    runner = CliRunner()
    runner.invoke(
        app,
        ["source", "slack", "--export", str(archive_path)],
        env={"HOME": str(tmp_path)},
    )

    removed = runner.invoke(
        app,
        ["source", "slack", "--remove"],
        env={"HOME": str(tmp_path)},
    )
    status = runner.invoke(
        app,
        ["source", "status", "--json"],
        env={"HOME": str(tmp_path)},
    )

    assert removed.exit_code == 0
    assert json.loads(status.stdout)["state"] == SlackSourceState.NOT_CONFIGURED.value


def test_run_uses_configured_slack_export(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    archive_path = _slack_export(tmp_path / "slack-export.zip")
    runner = CliRunner()
    runner.invoke(
        app,
        ["source", "slack", "--export", str(archive_path)],
        env={"HOME": str(tmp_path)},
    )
    captured: list[RunConfig] = []

    def capture(config: RunConfig) -> None:
        captured.append(config)
        raise ValueError("stop after config capture")

    monkeypatch.setattr("autobrain.cli.RunOrchestrator.local", capture)

    result = runner.invoke(
        app,
        ["run", "--no-open"],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 1
    assert captured[0].slack_export_path == archive_path.resolve()
    assert captured[0].slack_export_sha256 is not None
