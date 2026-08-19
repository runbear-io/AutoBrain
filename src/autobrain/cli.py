"""AutoBrain command-line entry point."""

import os
from pathlib import Path
from typing import Annotated

import typer

from autobrain.auth.callback import LocalCallback
from autobrain.auth.models import OAuthError, Provider
from autobrain.auth.oauth import OAuthManager
from autobrain.auth.service import ConnectionManager
from autobrain.auth.storage import TokenStore
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
    locate_run,
)
from autobrain.paths import AutoBrainPaths
from autobrain.preflight import Preflight
from autobrain.secrets import RuntimeEnvironment, RuntimeSettings
from autobrain.subscription import (
    CodexSubscriptionClient,
    CodexSubscriptionConfig,
    SubscriptionError,
)

app = typer.Typer(
    name="autobrain",
    help="Compare pinned company-brain candidates locally.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
auth_app = typer.Typer(help="Connect and manage hosted read-only MCP sources.")
app.add_typer(auth_app, name="auth")
subscription_app = typer.Typer(help="Use a locally authenticated ChatGPT subscription.")
app.add_typer(subscription_app, name="subscription")


@app.callback()
def main() -> None:
    """Compare pinned company-brain candidates locally."""


def _authorize_source(source: Provider) -> None:
    paths = AutoBrainPaths.from_home()
    settings = RuntimeSettings.from_environ(os.environ)
    if settings.callback_port_error is not None:
        typer.echo(f"MCP_AUTH_UNAVAILABLE: {settings.callback_port_error}", err=True)
        raise typer.Exit(1)
    environment = RuntimeEnvironment.from_environ(os.environ)
    manager = OAuthManager(
        TokenStore(paths.root / "auth"),
        callback_factory=lambda state: LocalCallback("127.0.0.1", settings.callback_port, state),
    )
    try:
        token = manager.authorize(
            source,
            slack_client_id=(
                environment.slack_client_id.get_secret_value()
                if environment.slack_client_id
                else None
            ),
            slack_client_secret=(
                environment.slack_client_secret.get_secret_value()
                if environment.slack_client_secret
                else None
            ),
        )
    except OAuthError as exc:
        typer.echo(f"MCP_AUTH_UNAVAILABLE: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Connected {source.value} workspace {token.workspace_id} as {token.user_id}")
    if manager.store.degraded_warning:
        typer.echo(f"WARNING: {manager.store.degraded_warning}", err=True)


@auth_app.command("slack")
def authorize_slack() -> None:
    """Authorize the fixed internal Slack app for read-only MCP access."""
    _authorize_source(Provider.SLACK)


@auth_app.command("notion")
def authorize_notion() -> None:
    """Authorize Notion MCP using discovery, DCR, and PKCE."""
    _authorize_source(Provider.NOTION)


@auth_app.command("status")
def auth_status(
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON status.")] = False,
) -> None:
    """Show Slack and Notion connection state independently."""
    report = ConnectionManager(AutoBrainPaths.from_home().root).status()
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        return
    for connection in report.connections:
        typer.echo(f"{connection.provider.value:8} {connection.state.value}")
        if connection.warning:
            typer.echo(f"WARNING: {connection.warning}", err=True)


@auth_app.command("logout")
def auth_logout(provider: Provider) -> None:
    """Remove every stored identity for one source."""
    try:
        removed = ConnectionManager(AutoBrainPaths.from_home().root).logout(provider)
    except OAuthError as exc:
        typer.echo(f"MCP_AUTH_UNAVAILABLE: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(
        f"Logged out {provider.value} ({removed} credential{'s' if removed != 1 else ''} removed)"
    )


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
        import json

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
) -> None:
    """Run one immutable Slack/Notion comparison and report its run ID."""
    try:
        config = RunConfig(
            budget_usd=budget_usd,
            max_questions=max_questions,
            include_dms=include_dms,
            open_report=not no_open,
            output=output,
            provider_mode=provider.lower(),
        )
        fixture_path_raw = os.environ.get("AUTOBRAIN_TEST_FIXTURE_PATH")
        fixture_allowed = os.environ.get("AUTOBRAIN_ALLOW_TEST_FIXTURE") == "1"
        if fixture_path_raw and not fixture_allowed:
            raise ValueError(
                "MCP_AUTH_UNAVAILABLE: test fixture requires AUTOBRAIN_ALLOW_TEST_FIXTURE=1"
            )
        if fixture_allowed:
            if not fixture_path_raw:
                raise ValueError("MCP_AUTH_UNAVAILABLE: AUTOBRAIN_TEST_FIXTURE_PATH is required")
            fixture = load_fixture(Path(fixture_path_raw))
            result = RunOrchestrator.fixture(
                config,
                fixture_id=fixture.fixture_id,
                fixture_sha256=fixture.fixture_sha256,
                connectors=fixture_connectors(fixture),
                candidates=fixture_candidates(fixture),
            ).run()
        else:
            result = RunOrchestrator.local(config).run()
    except (FixtureValidationError, ValueError, OSError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"run-id: {result.run_id}")
    typer.echo(f"status: {result.status.value}")
    typer.echo(f"run-dir: {result.run_dir}")
    if result.report_path is not None:
        typer.echo(f"report: {result.report_path}")
    if result.status is not Status.OK:
        raise typer.Exit(1)


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
