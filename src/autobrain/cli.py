import json
import os
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from autobrain.auth.models import Provider
from autobrain.candidates.gbrain_config import (
    GBrainEmbeddingProvider,
    GBrainExecutionConfig,
)
from autobrain.connectors.notion_snapshot import NotionSnapshotStore
from autobrain.embedding import EmbeddingBackend, production_embedding_registry
from autobrain.fixture import (
    FixtureValidationError,
    fixture_candidates,
    fixture_connectors,
    load_fixture,
)
from autobrain.local_server import (
    DEFAULT_LOCAL_PORT,
    PROJECTION_PATH,
    LocalRunServer,
    RunOutcomeStatus,
    outcome_for_run_dir,
)
from autobrain.model_access import inspect_model_access, render_model_access_human
from autobrain.models import Status
from autobrain.orchestration import (
    DEFAULT_BUDGET_USD,
    RunConfig,
    RunOrchestrator,
    StageEvent,
    locate_run,
)
from autobrain.paths import AutoBrainPaths, PathConfinementError
from autobrain.preflight import Preflight
from autobrain.runs import RunInspectionError, compare_runs, list_runs
from autobrain.secrets import RuntimeEnvironment, RuntimeSettings
from autobrain.source_cli import auth_app, source_app
from autobrain.source_store import SlackSourceStore
from autobrain.subscription import (
    ProviderId,
    SubscriptionError,
    SubscriptionStatus,
    provider_registry,
)
from autobrain.tui import run_tui

app = typer.Typer(
    name="autobrain",
    help="Compare pinned company-brain candidates locally.",
    no_args_is_help=False,
    invoke_without_command=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
app.add_typer(auth_app, name="auth")
subscription_app = typer.Typer(help="Use an explicitly selected local consumer subscription CLI.")
app.add_typer(subscription_app, name="subscription")
app.add_typer(source_app, name="source")
runs_app = typer.Typer(help="Inspect and compare immutable evaluation runs.")
app.add_typer(runs_app, name="runs")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    provider: Annotated[
        ProviderId,
        typer.Option("--provider", help="Subscription provider for the TUI."),
    ] = ProviderId.CODEX,
) -> None:
    if ctx.invoked_subcommand is None:
        try:
            run_tui(provider=provider)
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from None


@app.command("setup")
def setup(
    provider: Annotated[
        ProviderId,
        typer.Option("--provider", help="Initial subscription provider for setup."),
    ] = ProviderId.CODEX,
) -> None:
    """Run first-time onboarding, or reconnect a provider, Slack, and Notion."""
    try:
        run_tui(force_setup=True, provider=provider)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None


@subscription_app.command("status")
def subscription_status(
    provider: Annotated[
        ProviderId,
        typer.Option("--provider", help="Subscription provider to inspect."),
    ] = ProviderId.CODEX,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON status."),
    ] = False,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Bypass a recent bounded status probe."),
    ] = False,
) -> None:
    """Check the selected local consumer-subscription boundary."""
    registry = provider_registry()
    report = registry.probe(provider, refresh=refresh)
    client = registry.get(provider)
    detail = {
        "provider": provider.value,
        "status": report.status.value,
        "reason": report.reason.value if report.reason is not None else None,
        "detail": report.detail,
        "identity": {
            "provider": client.identity.provider.value,
            "model": client.identity.model,
            "cli_version": client.identity.cli_version,
            "auth_kind": client.identity.auth_kind.value,
        },
    }
    if json_output:
        typer.echo(json.dumps(detail, indent=2))
        return
    typer.echo(f"{report.status.value}: {provider.value}")
    if report.detail:
        typer.echo(report.detail, err=report.status is not SubscriptionStatus.READY)


@subscription_app.command("setup")
def subscription_setup(
    provider: Annotated[
        ProviderId,
        typer.Option("--provider", help="Subscription provider to authenticate."),
    ] = ProviderId.CODEX,
) -> None:
    """Open the selected vendor's user-driven consumer login flow."""
    registry = provider_registry()
    client = registry.get(provider)
    try:
        return_code = client.login()
    except SubscriptionError as exc:
        typer.echo(f"{exc.status.value}: {exc.detail}", err=True)
        raise typer.Exit(1) from None
    if return_code != 0:
        typer.echo(
            f"SUBSCRIPTION_AUTH_UNAVAILABLE: {provider.value} login did not complete",
            err=True,
        )
        raise typer.Exit(return_code or 1)
    registry.invalidate(provider)
    report = registry.probe(provider, refresh=True)
    if report.status is not SubscriptionStatus.READY:
        typer.echo(f"{report.status.value}: {report.detail}", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"{provider.value} consumer subscription login completed; "
        f"run `autobrain subscription status --provider {provider.value}`."
    )


@subscription_app.command("ask")
def subscription_ask(
    prompt: Annotated[str, typer.Argument(help="Prompt sent through subscription mode.")],
    provider: Annotated[
        ProviderId,
        typer.Option("--provider", help="Subscription provider to execute."),
    ] = ProviderId.CODEX,
) -> None:
    """Run one safe, tool-free prompt through the selected local subscription."""
    try:
        client = provider_registry().get(provider)
        answer = client.answer(prompt).text
    except (SubscriptionError, ValueError) as exc:
        if isinstance(exc, SubscriptionError):
            typer.echo(f"{exc.status.value}: {exc.detail}", err=True)
        else:
            typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(answer)


model_access_app = typer.Typer(help="Inspect local model capability access.")
app.add_typer(model_access_app, name="model-access")


@model_access_app.command("status")
def model_access_status(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the complete capability matrix as JSON."),
    ] = False,
) -> None:
    """Report chat, embedding, verifier, and metering capability separately."""
    status = inspect_model_access(os.environ)
    if json_output:
        typer.echo(json.dumps(status.as_dict(), indent=2, sort_keys=True))
        return
    typer.echo(render_model_access_human(status))


@app.command()
def doctor(
    provider: Annotated[
        ProviderId,
        typer.Option("--provider", help="Subscription provider to diagnose."),
    ] = ProviderId.CODEX,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the typed doctor report as JSON."),
    ] = False,
) -> None:
    """Check local runtimes, credentials, paths, pins, and callback capabilities."""
    settings = RuntimeSettings.from_environ(os.environ)
    report = Preflight(
        paths=AutoBrainPaths.from_home(),
        environment=RuntimeEnvironment.from_environ(os.environ),
        callback_port=settings.callback_port,
        callback_port_error=settings.callback_port_error,
        subscription_provider=provider,
        embedding_environ=os.environ,
    ).run()
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        return
    typer.echo(f"AutoBrain doctor: {report.status.value}")
    typer.echo(
        "Precredential readiness: "
        f"{'READY' if report.readiness.ready else 'BLOCKED'} "
        f"(governance={','.join(report.readiness.governance_codes) or 'none'})"
    )
    for check in report.checks:
        typer.echo(f"{check.status.value:24} {check.name}: {check.detail}")
    typer.echo(f"State root: {Path(report.paths.root)}")


@app.command("run")
def run_comparison(
    budget_usd: Annotated[
        float,
        typer.Option("--budget-usd", min=0.0, help="Hard local/provider budget cap."),
    ] = DEFAULT_BUDGET_USD,
    max_questions: Annotated[
        int,
        typer.Option("--max-questions", min=1, help="Maximum benchmark questions."),
    ] = 30,
    include_dms: Annotated[
        bool,
        typer.Option("--include-dms", help="Include permission-scoped Slack DMs."),
    ] = False,
    no_open: Annotated[
        bool,
        typer.Option("--no-open", help="Do not open the local HTML report."),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Run output root; defaults to ~/.autobrain/runs."),
    ] = None,
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            help="Provider mode: api or <codex|claude|kimi|grok>-subscription.",
            case_sensitive=False,
        ),
    ] = "api",
    embedding_backend: Annotated[
        str | None,
        typer.Option(
            "--embedding-backend",
            help="Explicit embedding backend: openai or local-hash.",
            case_sensitive=False,
        ),
    ] = None,
    stage_events: Annotated[
        Path | None,
        typer.Option(
            "--stage-events",
            help="Write persisted stage events as JSONL to this file.",
        ),
    ] = None,
    notion_only: Annotated[
        bool,
        typer.Option(
            "--notion-only",
            help=(
                "Run only the configured Notion snapshot; Slack is recorded absent "
                "and the result is non-final."
            ),
        ),
    ] = False,
    gbrain_provider: Annotated[
        GBrainEmbeddingProvider,
        typer.Option(
            "--gbrain-provider",
            help=(
                "GBrain mode/provider. keyword-only is keyless; hosted credentials are read "
                "from AUTOBRAIN_GBRAIN_API_KEY."
            ),
        ),
    ] = GBrainEmbeddingProvider.KEYWORD_ONLY,
    gbrain_model: Annotated[
        str | None,
        typer.Option("--gbrain-model", help="Explicit GBrain embedding model override."),
    ] = None,
    gbrain_dimensions: Annotated[
        int | None,
        typer.Option("--gbrain-dimensions", min=1, help="Embedding dimensions override."),
    ] = None,
    gbrain_endpoint: Annotated[
        str | None,
        typer.Option("--gbrain-endpoint", help="HTTP(S) endpoint without URL userinfo."),
    ] = None,
) -> None:
    """Run one immutable Slack/Notion comparison and report its run ID."""
    try:
        paths = AutoBrainPaths.from_home()
        slack_status = SlackSourceStore(paths.sources).status()
        notion_snapshot_status = NotionSnapshotStore(paths.sources).status()
        fixture_path_raw = os.environ.get("AUTOBRAIN_TEST_FIXTURE_PATH")
        fixture_allowed = os.environ.get("AUTOBRAIN_ALLOW_TEST_FIXTURE") == "1"
        embedding_registry = production_embedding_registry()
        if fixture_allowed and os.environ.get("AUTOBRAIN_ENABLE_TEST_SEMANTIC_EMBEDDING") == "1":
            embedding_registry = embedding_registry.with_test_semantic_backend()
        if notion_only and not notion_snapshot_status.ready:
            raise ValueError("SOURCE_AUTH_UNAVAILABLE: --notion-only requires an imported snapshot")
        gbrain_config = (
            GBrainExecutionConfig.quick_start()
            if gbrain_provider is GBrainEmbeddingProvider.KEYWORD_ONLY
            else GBrainExecutionConfig.semantic(
                gbrain_provider,
                model=gbrain_model,
                dimensions=gbrain_dimensions,
                endpoint=gbrain_endpoint,
                credential=(
                    os.environ.get("AUTOBRAIN_GBRAIN_API_KEY")
                    or os.environ.get(
                        {
                            GBrainEmbeddingProvider.GEMINI: "GEMINI_API_KEY",
                            GBrainEmbeddingProvider.OPENAI: "OPENAI_API_KEY",
                            GBrainEmbeddingProvider.VOYAGE: "VOYAGE_API_KEY",
                        }.get(gbrain_provider, ""),
                    )
                ),
            )
        )
        config = RunConfig(
            budget_usd=budget_usd,
            max_questions=max_questions,
            include_dms=include_dms,
            open_report=not no_open,
            output=output,
            provider_mode=provider.lower(),
            embedding_backend=(
                embedding_backend.lower()
                if embedding_backend is not None
                else os.environ.get("AUTOBRAIN_EMBEDDING_BACKEND")
                or (
                    EmbeddingBackend.OPENAI.value
                    if provider.lower() == "api"
                    else EmbeddingBackend.LOCAL_HASH.value
                )
            ),
            embedding_registry=embedding_registry,
            slack_export_path=slack_status.archive_path if slack_status.ready else None,
            slack_export_sha256=(
                slack_status.config.archive_sha256
                if slack_status.ready and slack_status.config is not None
                else None
            ),
            notion_snapshot_path=(
                notion_snapshot_status.snapshot_path if notion_snapshot_status.ready else None
            ),
            gbrain_config=gbrain_config,
            selected_sources=(
                (Provider.NOTION,) if notion_only else (Provider.SLACK, Provider.NOTION)
            ),
        )
        if fixture_path_raw and not fixture_allowed:
            raise ValueError(
                "MCP_AUTH_UNAVAILABLE: test fixture requires AUTOBRAIN_ALLOW_TEST_FIXTURE=1"
            )
        event_file = stage_events.open("w", encoding="utf-8") if stage_events else None

        def emit_stage_event(event: StageEvent) -> None:
            if event_file is None:
                return
            event_file.write(json.dumps(event.as_manifest_entry(), separators=(",", ":")) + "\n")
            event_file.flush()

        try:
            if fixture_allowed:
                if not fixture_path_raw:
                    raise ValueError(
                        "MCP_AUTH_UNAVAILABLE: AUTOBRAIN_TEST_FIXTURE_PATH is required"
                    )
                fixture = load_fixture(Path(fixture_path_raw))
                result = RunOrchestrator.fixture(
                    config,
                    fixture_id=fixture.fixture_id,
                    fixture_sha256=fixture.fixture_sha256,
                    connectors=fixture_connectors(fixture),
                    candidates=fixture_candidates(fixture),
                    stage_event_sink=emit_stage_event if event_file else None,
                ).run()
            else:
                orchestrator = (
                    RunOrchestrator.local(config, stage_event_sink=emit_stage_event)
                    if event_file is not None
                    else RunOrchestrator.local(config)
                )
                result = orchestrator.run()
        finally:
            if event_file is not None:
                event_file.close()
    except (FixtureValidationError, ValueError, OSError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"run-id: {result.run_id}")
    typer.echo(f"status: {result.status.value}")
    typer.echo(f"run-dir: {result.run_dir}")
    if result.report_path is not None:
        typer.echo(f"report: {result.report_path}")
    for diagnostic in result.event_sink_errors:
        typer.echo(f"stage-event-error: {diagnostic}", err=True)
    if result.status is not Status.OK or result.event_sink_errors:
        raise typer.Exit(1)


def _emit_run_error(error: RunInspectionError | PathConfinementError) -> NoReturn:
    if isinstance(error, RunInspectionError):
        payload = error.payload()
    else:
        payload = {"status": "PATH_ESCAPE", "detail": str(error)}
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    raise typer.Exit(1)


@runs_app.command("list")
def runs_list(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the validated run inventory as JSON."),
    ] = False,
) -> None:
    """List immutable local evaluations, including failed and incomplete runs."""
    try:
        inventory = list_runs(AutoBrainPaths.from_home().runs)
    except (RunInspectionError, PathConfinementError) as error:
        _emit_run_error(error)
    if json_output:
        typer.echo(inventory.model_dump_json(indent=2))
        return
    if not inventory.runs:
        typer.echo("No immutable runs found.")
        return
    for run in inventory.runs:
        typer.echo(f"{run.run_id}  {run.status}  {run.artifact_status}")


@runs_app.command("compare")
def runs_compare(
    left_run_id: Annotated[str, typer.Argument(help="First immutable run ID.")],
    right_run_id: Annotated[str, typer.Argument(help="Second immutable run ID.")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the typed comparison as JSON."),
    ] = False,
    allow_different_corpus: Annotated[
        bool,
        typer.Option(
            "--allow-different-corpus",
            help="Inspect different corpus/benchmark hashes as explicitly non-equivalent.",
        ),
    ] = False,
) -> None:
    """Compare provenance, hashes, metrics, eligibility, and verdict."""
    try:
        comparison = compare_runs(
            AutoBrainPaths.from_home().runs,
            left_run_id,
            right_run_id,
            allow_different_corpus=allow_different_corpus,
        )
    except (RunInspectionError, PathConfinementError) as error:
        _emit_run_error(error)
    if json_output:
        typer.echo(comparison.model_dump_json(indent=2))
        return
    typer.echo(f"status: {comparison.status}")
    typer.echo(f"equivalent: {str(comparison.equivalent).lower()}")
    typer.echo(f"comparable: {str(comparison.comparable).lower()}")
    for difference in comparison.differences:
        typer.echo(f"{difference.path}: {difference.left!r} -> {difference.right!r}")


@app.command("report")
def reopen_report(
    run_id: Annotated[str, typer.Argument(help="Immutable run ID to reopen.")],
    no_open: Annotated[
        bool,
        typer.Option("--no-open", help="Print the report path without opening it."),
    ] = False,
) -> None:
    """Reopen a local report by run ID without rerunning or changing evidence."""
    paths = AutoBrainPaths.from_home()
    roots = [paths.runs]
    configured_root = os.environ.get("AUTOBRAIN_RUN_ROOT")
    if configured_root:
        roots.insert(0, Path(configured_root))
    run_dir = locate_run(run_id, roots=roots)
    if run_dir is None:
        typer.echo(f"FAILED: run not found: {run_id}", err=True)
        raise typer.Exit(1)
    report_path = run_dir / "report.html"
    if not report_path.is_file():
        typer.echo(f"FAILED: report not found for run: {run_id}", err=True)
        raise typer.Exit(1)
    if not no_open:
        import webbrowser

        webbrowser.open(report_path.as_uri())
    typer.echo(f"report: {report_path}")


if __name__ == "__main__":
    app()


@app.command("serve")
def serve_local_fixture(
    run_dir: Annotated[
        Path | None,
        typer.Option("--run-dir", help="Directory holding the comparison.json to publish."),
    ] = None,
    port: Annotated[
        int,
        typer.Option("--port", help=f"Loopback port to bind. Default {DEFAULT_LOCAL_PORT}."),
    ] = DEFAULT_LOCAL_PORT,
    check: Annotated[
        bool,
        typer.Option("--check", help="Report what would be published, then exit."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the outcome payload as JSON."),
    ] = False,
) -> None:
    """Serve one redacted run projection on 127.0.0.1 for a local browser client.

    This is a local, unauthenticated developer fixture. It binds loopback only,
    is never exposed to a network, and is not a hosted deployment. Only browser
    origins on localhost or 127.0.0.1 are granted CORS access.
    """
    target = run_dir if run_dir is not None else AutoBrainPaths.from_home().runs
    outcome = outcome_for_run_dir(target)
    if check:
        if json_output:
            typer.echo(json.dumps(outcome.to_payload(), indent=2))
        else:
            typer.echo(f"{outcome.status.value}: {outcome.error or target}")
        if outcome.status is RunOutcomeStatus.FAILED and "no comparison.json" in (
            outcome.error or ""
        ):
            raise typer.Exit(1)
        return
    with LocalRunServer(lambda: outcome_for_run_dir(target), port=port) as server:
        typer.echo(f"AutoBrain local fixture (unauthenticated, loopback only): {server.base_url}")
        typer.echo(f"projection: {server.base_url}{PROJECTION_PATH}")
        typer.echo("Press Ctrl+C to stop.")
        try:
            server.wait_forever()
        except KeyboardInterrupt:
            typer.echo("stopped")
