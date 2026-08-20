import json
import os
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from autobrain.fixture import (
    FixtureValidationError,
    fixture_candidates,
    fixture_connectors,
    load_fixture,
)
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
    CodexSubscriptionClient,
    CodexSubscriptionConfig,
    SubscriptionError,
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
subscription_app = typer.Typer(help="Use a locally authenticated ChatGPT subscription.")
app.add_typer(subscription_app, name="subscription")
app.add_typer(source_app, name="source")
runs_app = typer.Typer(help="Inspect and compare immutable evaluation runs.")
app.add_typer(runs_app, name="runs")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        try:
            run_tui()
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from None


@app.command("setup")
def setup() -> None:
    """Run first-time onboarding, or reconnect ChatGPT, Slack, and Notion."""
    try:
        run_tui(force_setup=True)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None


@subscription_app.command("status")
def subscription_status(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON status."),
    ] = False,
) -> None:
    """Check the local Codex/ChatGPT subscription login boundary."""
    client = CodexSubscriptionClient(CodexSubscriptionConfig.from_environ())
    status = client.status()
    detail = {
        "provider": "codex",
        "status": status.value,
        "command": client.config.command,
    }
    if json_output:
        typer.echo(json.dumps(detail, indent=2))
        return
    typer.echo(f"{status.value}: {client.config.command}")
    if status.value != "READY":
        typer.echo("Run `codex login` with the ChatGPT account.", err=True)


@subscription_app.command("setup")
def subscription_setup() -> None:
    """Open the user-driven ChatGPT subscription login flow."""
    client = CodexSubscriptionClient(CodexSubscriptionConfig.from_environ())
    try:
        return_code = client.login()
    except SubscriptionError as exc:
        typer.echo(f"{exc.status.value}: {exc.detail}", err=True)
        raise typer.Exit(1) from None
    if return_code != 0:
        typer.echo("SUBSCRIPTION_AUTH_UNAVAILABLE: Codex login did not complete", err=True)
        raise typer.Exit(return_code or 1)
    typer.echo("ChatGPT subscription login completed; run `autobrain subscription status`.")


@subscription_app.command("ask")
def subscription_ask(
    prompt: Annotated[str, typer.Argument(help="Prompt sent through Codex subscription mode.")],
) -> None:
    """Run one read-only prompt through the local ChatGPT subscription."""
    try:
        answer = CodexSubscriptionClient(CodexSubscriptionConfig.from_environ()).ask(prompt)
    except (SubscriptionError, ValueError) as exc:
        if isinstance(exc, SubscriptionError):
            typer.echo(f"{exc.status.value}: {exc.detail}", err=True)
        else:
            typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(answer)


@app.command()
def doctor(
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
    ).run()
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        return
    typer.echo(f"AutoBrain doctor: {report.status.value}")
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
            help="Provider mode: api or codex-subscription.",
            case_sensitive=False,
        ),
    ] = "api",
    stage_events: Annotated[
        Path | None,
        typer.Option(
            "--stage-events",
            help="Write persisted stage events as JSONL to this file.",
        ),
    ] = None,
) -> None:
    """Run one immutable Slack/Notion comparison and report its run ID."""
    try:
        paths = AutoBrainPaths.from_home()
        slack_status = SlackSourceStore(paths.sources).status()
        config = RunConfig(
            budget_usd=budget_usd,
            max_questions=max_questions,
            include_dms=include_dms,
            open_report=not no_open,
            output=output,
            provider_mode=provider.lower(),
            slack_export_path=slack_status.archive_path if slack_status.ready else None,
            slack_export_sha256=(
                slack_status.config.archive_sha256
                if slack_status.ready and slack_status.config is not None
                else None
            ),
        )
        fixture_path_raw = os.environ.get("AUTOBRAIN_TEST_FIXTURE_PATH")
        fixture_allowed = os.environ.get("AUTOBRAIN_ALLOW_TEST_FIXTURE") == "1"
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
