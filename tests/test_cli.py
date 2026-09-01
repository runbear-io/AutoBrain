import json
from collections.abc import Callable
from pathlib import Path
from zipfile import ZipFile

from pytest import MonkeyPatch
from typer.testing import CliRunner

from autobrain.auth.models import Provider
from autobrain.cli import app
from autobrain.models import Status
from autobrain.orchestration import RunConfig, RunResult, StageEvent
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


def test_doctor_offline_is_explicit_and_machine_readable(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = CliRunner().invoke(app, ["doctor", "--offline", "--json"])
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["schema_version"] == 1
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["browser_open"]["status"] == "NOT_PROBED"
    assert checks["chatgpt_subscription"]["status"] in {"NOT_PROBED", "UNAVAILABLE"}
    assert "remediation" in checks["chatgpt_subscription"]["detail"]


def test_doctor_json_is_machine_readable(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    assert '"schema_version": 1' in result.stdout
    assert "MISSING_PROVIDER" in result.stdout


def test_run_uses_provider_native_gemini_key_when_gbrain_key_is_absent(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUTOBRAIN_GBRAIN_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-gemini-key")

    captured: dict[str, object] = {}

    class _Config:
        pass

    def fake_semantic(
        provider: object,
        *,
        model: str | None,
        dimensions: int | None,
        endpoint: str | None,
        credential: str | None,
    ) -> _Config:
        captured["provider"] = provider
        captured["credential"] = credential
        return _Config()

    monkeypatch.setattr("autobrain.cli.GBrainExecutionConfig.semantic", fake_semantic)

    def fake_orchestrator(**_: object) -> None:
        return None

    monkeypatch.setattr("autobrain.cli.RunOrchestrator", fake_orchestrator)

    result = CliRunner().invoke(
        app,
        ["run", "--gbrain-provider", "gemini", "--no-open"],
    )

    assert result.exit_code != 0
    assert captured["credential"] == "fixture-gemini-key"


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


def test_source_slack_export_reports_typed_archive_failure(tmp_path: Path) -> None:
    archive_path = _slack_export(tmp_path / "slack-export.zip")
    archive_path.write_bytes(archive_path.read_bytes()[:-12])

    result = CliRunner().invoke(
        app,
        ["source", "slack", "--export", str(archive_path)],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 1
    assert "SOURCE_AUTH_UNAVAILABLE: invalid Slack export ZIP" in result.output
    assert "Traceback" not in result.output


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


def test_source_status_reports_unsupported_provider_readiness(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["source", "status", "--json"],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0
    connectors = json.loads(result.stdout)["connectors"]
    for provider in ("confluence", "google_drive", "sharepoint"):
        assert connectors[provider]["state"] == "UNSUPPORTED"
        assert connectors[provider]["ready"] is False
        assert connectors[provider]["mode"] == "UNSUPPORTED"


def test_source_notion_snapshot_import_and_status(tmp_path: Path) -> None:
    snapshot = tmp_path / "notion-snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "notion-mcp-snapshot",
                "fetched_at": "2026-08-20T10:00:00Z",
                "documents": [
                    {
                        "page_id": "page-1",
                        "page_url": "https://www.notion.so/page-1",
                        "title": "Synthetic page",
                        "fetched_at": "2026-08-20T09:59:00Z",
                        "content": "Synthetic safe content.",
                        "metadata": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    imported = runner.invoke(
        app,
        ["source", "notion-snapshot", "--import", str(snapshot)],
        env={"HOME": str(tmp_path)},
    )
    status = runner.invoke(
        app,
        ["source", "status", "--json"],
        env={"HOME": str(tmp_path)},
    )

    assert imported.exit_code == 0
    assert "partial/non-final" in imported.stdout
    report = json.loads(status.stdout)
    assert report["notion_snapshot"]["ready"] is True
    assert report["notion_snapshot"]["coverage"]["completeness"] == "UNKNOWN"


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


def test_run_notion_only_selects_snapshot_without_slack_auth(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    snapshot = tmp_path / "notion-snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "notion-mcp-snapshot",
                "fetched_at": "2026-08-20T10:00:00Z",
                "documents": [
                    {
                        "page_id": "page-1",
                        "page_url": "https://www.notion.so/page-1",
                        "title": "Synthetic page",
                        "fetched_at": "2026-08-20T09:59:00Z",
                        "content": "Synthetic safe content.",
                        "metadata": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    runner.invoke(
        app,
        ["source", "notion-snapshot", "--import", str(snapshot)],
        env={"HOME": str(tmp_path)},
    )
    captured: list[RunConfig] = []

    def capture(config: RunConfig) -> None:
        captured.append(config)
        raise ValueError("stop after config capture")

    monkeypatch.setattr("autobrain.cli.RunOrchestrator.local", capture)

    result = runner.invoke(
        app,
        ["run", "--notion-only", "--no-open"],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 1
    assert captured[0].selected_sources == (Provider.NOTION,)
    assert captured[0].slack_export_path is None
    assert captured[0].notion_snapshot_path is not None


def test_run_stage_events_option_writes_jsonl_from_persisted_sink(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    event_path = tmp_path / "stages.jsonl"
    run_dir = tmp_path / "runs" / "fixture-run"
    run_dir.mkdir(parents=True)

    class _FakeOrchestrator:
        def run(self) -> RunResult:
            return RunResult(
                run_id="fixture-run",
                run_dir=run_dir,
                status=Status.OK,
                report_path=None,
                candidate_results=(),
                verdict="NO_DECISION",
            )

    def build(
        config: RunConfig,
        *,
        stage_event_sink: Callable[[StageEvent], None],
    ) -> _FakeOrchestrator:
        del config
        stage_event_sink(
            StageEvent(
                sequence=1,
                run_id="fixture-run",
                name="preflight",
                status=Status.OK,
                detail="persisted",
                started_at="2026-08-19T00:00:00+00:00",
            )
        )
        return _FakeOrchestrator()

    monkeypatch.setattr("autobrain.cli.RunOrchestrator.local", build)

    result = CliRunner().invoke(
        app,
        ["run", "--no-open", "--stage-events", str(event_path)],
        env={"HOME": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert json.loads(event_path.read_text()) == {
        "sequence": 1,
        "run_id": "fixture-run",
        "name": "preflight",
        "status": "OK",
        "detail": "persisted",
        "started_at": "2026-08-19T00:00:00+00:00",
    }


def test_setup_command_is_registered() -> None:
    from typer.testing import CliRunner

    from autobrain.cli import app

    result = CliRunner().invoke(app, ["setup", "--help"])
    assert result.exit_code == 0
    assert "onboarding" in result.stdout.lower() or "reconnect" in result.stdout.lower()
