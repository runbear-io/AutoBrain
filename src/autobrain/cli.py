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
from autobrain.custom_provider import (
    CustomProviderConfig,
    CustomProviderError,
    CustomProviderRegistry,
)
from autobrain.embedding import EmbeddingBackend, production_embedding_registry
from autobrain.fixture import (
    FixtureValidationError,
    fixture_candidates,
    fixture_connectors,
    load_fixture,
    write_fixture,
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
from autobrain.paths import (
    AutoBrainPaths,
    PathConfinementError,
    is_valid_run_id,
    resolve_run_root,
)
from autobrain.preflight import Preflight
from autobrain.runs import (
    RunInspectionError,
    RunVerificationStatus,
    compare_runs,
    explain_run,
    list_runs,
    verify_run,
)
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
provider_app = typer.Typer(help="Manage local OpenAI-compatible provider registrations.")
app.add_typer(provider_app, name="provider")
app.add_typer(source_app, name="source")
runs_app = typer.Typer(help="Inspect and compare immutable evaluation runs.")
app.add_typer(runs_app, name="runs")
fixture_app = typer.Typer(help="Create local, deterministic test fixtures.")
app.add_typer(fixture_app, name="fixture")


@fixture_app.command("generate")
def fixture_generate(
    seed: Annotated[int, typer.Option("--seed", help="Deterministic fixture seed.")],
    output: Annotated[Path, typer.Option("--output", help="Output JSON path.")] = Path(
        "fixture.json"
    ),
) -> None:
    """Generate a schema-v1 fixture; available only in explicit test mode."""
    if os.environ.get("AUTOBRAIN_ALLOW_TEST_FIXTURE") != "1":
        typer.echo(
            "MCP_AUTH_UNAVAILABLE: fixture generation requires AUTOBRAIN_ALLOW_TEST_FIXTURE=1",
            err=True,
        )
        raise typer.Exit(1)
    try:
        path = write_fixture(output, seed=seed)
    except (ValueError, OSError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from None
    spec = load_fixture(path)
    typer.echo(f"fixture-id: {spec.fixture_id}")
    typer.echo(f"fixture-sha256: {spec.fixture_sha256}")
    typer.echo(f"fixture-path: {path}")


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


@provider_app.command("add")
def provider_add(
    provider_id: Annotated[str, typer.Argument(help="Unique local provider name.")],
    endpoint: Annotated[str, typer.Option("--endpoint", help="Provider HTTP(S) base URL.")],
    model: Annotated[str, typer.Option("--model", help="Default chat model.")],
    api_key_env: Annotated[
        str, typer.Option("--api-key-env", help="Environment fallback variable.")
    ],
    api_key: Annotated[str | None, typer.Option("--api-key", hidden=True)] = None,
) -> None:
    """Register a provider locally; its API key is stored in the OS keychain."""
    try:
        config = CustomProviderConfig(
            provider_id=provider_id,
            name=provider_id,
            endpoint=endpoint,
            model=model,
            api_key_env=api_key_env,
        )
        secret = api_key or typer.prompt("API key", hide_input=True)
        CustomProviderRegistry(AutoBrainPaths.from_home().root).add(config, secret)
    except (CustomProviderError, ValueError, OSError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"custom provider registered: {config.provider_id}")


@provider_app.command("status")
def provider_status(
    provider_id: Annotated[str | None, typer.Argument(help="Provider name; omit for all.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON status.")] = False,
) -> None:
    """Show local registration and whether a key is available (never the key)."""
    try:
        registry = CustomProviderRegistry(AutoBrainPaths.from_home().root)
        statuses = (
            [registry.status(provider_id, dict(os.environ))]
            if provider_id
            else [registry.status(item.provider_id, dict(os.environ)) for item in registry.list()]
        )
    except (CustomProviderError, ValueError, OSError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from None
    if json_output:
        typer.echo(json.dumps([item.model_dump(mode="json") for item in statuses], indent=2))
        return
    for item in statuses:
        typer.echo(f"{item.status}: {item.provider_id} ({item.endpoint}, {item.model})")
        if item.detail:
            typer.echo(item.detail)


@provider_app.command("verify")
def provider_verify(
    provider_id: Annotated[str, typer.Argument(help="Provider name to verify.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON status.")] = False,
) -> None:
    """Verify the configured endpoint using the locally stored API key."""
    try:
        result = CustomProviderRegistry(AutoBrainPaths.from_home().root).verify(
            provider_id, dict(os.environ)
        )
    except (CustomProviderError, ValueError, OSError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from None
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(f"{result.status}: {result.provider_id}")
        typer.echo(result.detail)
    if result.status != "READY":
        raise typer.Exit(1)


@provider_app.command("remove")
def provider_remove(
    provider_id: Annotated[str, typer.Argument(help="Provider name to remove.")],
    yes: Annotated[bool, typer.Option("--yes", help="Do not prompt for confirmation.")] = False,
) -> None:
    """Remove a local provider registration and its stored API key."""
    if not yes and not typer.confirm(f"Remove custom provider {provider_id!r}?", default=False):
        raise typer.Abort()
    try:
        CustomProviderRegistry(AutoBrainPaths.from_home().root).remove(provider_id)
    except (CustomProviderError, ValueError, OSError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"custom provider removed: {provider_id.casefold()}")


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
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            help=(
                "Inspect only local config, filesystem, and executable availability; "
                "never probe providers."
            ),
        ),
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
    ).run(offline=offline)
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
        custom_provider = None
        if provider.casefold().startswith("custom:"):
            custom_provider = CustomProviderRegistry(paths.root).get(provider.split(":", 1)[1])
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
            custom_provider=custom_provider,
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
    typer.echo(f"run-root: {result.run_dir.parent}")
    typer.echo(f"run-dir: {result.run_dir}")
    typer.echo(f"inspect: autobrain runs list --run-root {result.run_dir.parent}")
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
    run_root: Annotated[
        Path | None,
        typer.Option("--run-root", help="Run output root; defaults to ~/.autobrain/runs."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the validated run inventory as JSON."),
    ] = False,
) -> None:
    """List immutable local evaluations, including failed and incomplete runs."""
    try:
        inventory = list_runs(resolve_run_root(run_root))
    except (RunInspectionError, PathConfinementError) as error:
        _emit_run_error(error)
    exit_after_output = inventory.status != "OK"
    if json_output:
        typer.echo(inventory.model_dump_json(indent=2))
    else:
        typer.echo(f"run-root: {resolve_run_root(run_root)}")
        if not inventory.runs:
            typer.echo("No immutable runs found.")
        else:
            for run in inventory.runs:
                detail = f"  {run.detail}" if run.detail else ""
                typer.echo(f"{run.run_id}  {run.status}  {run.artifact_status}{detail}")
    if exit_after_output:
        raise typer.Exit(1)


@runs_app.command("explain")
def runs_explain(
    run_id: Annotated[str, typer.Argument(help="Immutable run ID to explain.")],
    run_root: Annotated[
        Path | None,
        typer.Option("--run-root", help="Run output root; defaults to ~/.autobrain/runs."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the persisted eligibility explanation as JSON."),
    ] = False,
) -> None:
    """Explain why each persisted candidate was or was not eligible."""
    try:
        explanation = explain_run(resolve_run_root(run_root), run_id)
    except (RunInspectionError, PathConfinementError) as error:
        _emit_run_error(error)
    if json_output:
        typer.echo(explanation.model_dump_json(indent=2))
        return
    typer.echo(f"run-id: {explanation.run_id}")
    typer.echo(f"status: {explanation.run_status}")
    typer.echo(f"verdict: {explanation.verdict}")
    typer.echo(f"decision-status: {explanation.decision_status}")
    typer.echo(f"rationale: {explanation.rationale}")
    for candidate in explanation.candidates:
        candidate_id = candidate.candidate
        reasons = candidate.eligibility_reasons
        if candidate.eligible:
            typer.echo(f"{candidate_id}: eligible")
        else:
            typer.echo(f"{candidate_id}: ineligible")
            for reason in reasons:
                typer.echo(f"  - {reason}")


@runs_app.command("verify")
def runs_verify(
    run_id: Annotated[str, typer.Argument(help="Immutable run ID to verify.")],
    run_root: Annotated[
        Path | None,
        typer.Option("--run-root", help="Run output root; defaults to ~/.autobrain/runs."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the typed verification as JSON."),
    ] = False,
) -> None:
    """Verify recorded immutable run artifact hashes and cross-artifact integrity."""
    try:
        verification = verify_run(resolve_run_root(run_root), run_id)
    except (RunInspectionError, PathConfinementError) as error:
        _emit_run_error(error)
    if json_output:
        typer.echo(verification.model_dump_json(indent=2))
    else:
        typer.echo(f"status: {verification.status.value}")
        typer.echo(f"run-id: {verification.run_id}")
        if verification.detail:
            typer.echo(verification.detail)
        for artifact in verification.artifacts:
            state = (
                "NOT_RECORDED"
                if artifact.matches is None
                else "MATCH"
                if artifact.matches
                else "MISMATCH"
            )
            typer.echo(f"{artifact.path}: {state}")
    if verification.status is not RunVerificationStatus.VALID:
        raise typer.Exit(1)


@runs_app.command("compare")
def runs_compare(
    left_run_id: Annotated[str, typer.Argument(help="First immutable run ID.")],
    right_run_id: Annotated[str, typer.Argument(help="Second immutable run ID.")],
    run_root: Annotated[
        Path | None,
        typer.Option("--run-root", help="Run output root; defaults to ~/.autobrain/runs."),
    ] = None,
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
            resolve_run_root(run_root),
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
    run_root: Annotated[
        Path | None,
        typer.Option("--run-root", help="Run output root; defaults to ~/.autobrain/runs."),
    ] = None,
    no_open: Annotated[
        bool,
        typer.Option("--no-open", help="Print the report path without opening it."),
    ] = False,
) -> None:
    """Reopen a local report by run ID without rerunning or changing evidence."""
    try:
        root = resolve_run_root(run_root)
    except PathConfinementError as error:
        _emit_run_error(error)
    run_dir = locate_run(run_id, roots=[root])
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


@app.command("serve")
def serve_local_fixture(
    run_dir: Annotated[
        Path | None,
        typer.Option("--run-dir", help="Directory holding the comparison.json to publish."),
    ] = None,
    run_root: Annotated[
        Path | None,
        typer.Option("--run-root", help="Run output root; defaults to ~/.autobrain/runs."),
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
    try:
        configured_root = run_root is not None or os.environ.get("AUTOBRAIN_RUN_ROOT") is not None
        if run_dir is None:
            target = resolve_run_root(run_root)
        else:
            target = run_dir
            if target.is_symlink():
                raise PathConfinementError(f"run directory cannot be a symlink: {target}")
            if configured_root:
                root = resolve_run_root(run_root)
                canonical_root = root.resolve()
                canonical_target = target.resolve(strict=False)
                if (
                    not target.is_dir()
                    or not is_valid_run_id(target.name)
                    or target.resolve().parent != canonical_root
                    or not canonical_target.is_relative_to(canonical_root)
                ):
                    raise PathConfinementError(
                        f"run directory is not a valid run under configured root: {target}"
                    )
    except PathConfinementError as error:
        _emit_run_error(error)
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
    try:
        with LocalRunServer(lambda: outcome_for_run_dir(target), port=port) as server:
            if json_output:
                typer.echo(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "status": "LISTENING",
                            "base_url": server.base_url,
                            "projection_url": f"{server.base_url}{PROJECTION_PATH}",
                            "scope": "loopback",
                        },
                        sort_keys=True,
                    )
                )
            else:
                typer.echo(
                    f"AutoBrain local fixture (unauthenticated, loopback only): {server.base_url}"
                )
                typer.echo(f"projection: {server.base_url}{PROJECTION_PATH}")
                typer.echo("Press Ctrl+C to stop.")
            try:
                server.wait_forever()
            except KeyboardInterrupt:
                if json_output:
                    typer.echo(
                        json.dumps({"schema_version": 1, "status": "STOPPED"}, sort_keys=True)
                    )
                else:
                    typer.echo("stopped")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app()
