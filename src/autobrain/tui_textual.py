"""Actual Textual application, screens, widgets, workers, and messages."""

from __future__ import annotations

import logging
import queue
import time
from collections.abc import Callable
from typing import ClassVar, cast

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Label, Select, Static
from textual.worker import Worker, WorkerState

from autobrain.auth.models import Provider
from autobrain.cancellation import RunCancellation
from autobrain.candidates.gbrain_config import (
    GBrainEmbeddingProvider,
    GBrainExecutionConfig,
)
from autobrain.models import CandidateId
from autobrain.orchestration import RunResult, StageEvent
from autobrain.subscription_domain import ProviderId
from autobrain.subscription_process import sanitize_diagnostic
from autobrain.tui_actions import (
    BeginSetup,
    CancelRun,
    ConnectionsLoaded,
    GBrainValidated,
    GoBack,
    LoginSettled,
    Navigate,
    OpenReport,
    RefreshConnections,
    RequestLogin,
    RequestQuit,
    ResetRun,
    RunCompleted,
    RunFailed,
    RunStarted,
    SelectGBrainMode,
    SelectProvider,
    StageObserved,
    StartRun,
    ToggleCandidate,
    ToggleSource,
    UiAction,
    ValidateGBrain,
)
from autobrain.tui_effects import (
    CancelActiveRun,
    EffectRegistry,
    InteractiveLogin,
    LoadConnections,
    OpenExactReport,
    RunExperiment,
    UiEffect,
    ValidateGBrainProvider,
    open_exact_report,
    run_login_process,
)
from autobrain.tui_runtime import connection_snapshot, execute_plan
from autobrain.tui_state import UiScreen, UiState, reduce_ui
from autobrain.tui_viewmodels import build_view_model

_LOGGER = logging.getLogger(__name__)


def validate_gbrain_connection(config: GBrainExecutionConfig) -> GBrainExecutionConfig:
    """Run a bounded, secret-safe setup validation before Review."""
    endpoint = config.embedding.endpoint
    if endpoint is not None:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(endpoint, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=2):
                pass
        except urllib.error.HTTPError:
            pass  # The daemon answered; endpoint-specific methods may reject HEAD.
        except OSError:
            raise RuntimeError(
                "GBRAIN_LOCAL_ENDPOINT_UNAVAILABLE: start the local provider and retry"
            ) from None
    return config


def _safe_worker_error_name(operation: str, error: BaseException | None) -> str:
    """Classify worker failures without exposing exception text to UI or logs."""
    error_name = sanitize_diagnostic(
        type(error).__name__ if error is not None else "UnknownError",
        limit=100,
    )
    _LOGGER.warning("%s: %s", operation, error_name)
    return error_name


class ActionRequested(Message):
    """The only message emitted by cockpit widgets."""

    def __init__(self, action: UiAction) -> None:
        super().__init__()
        self.action = action


class ActionButton(Button):
    """Button carrying a semantic action rather than an I/O callback."""

    def __init__(self, label: str, action: UiAction, *, id: str) -> None:
        super().__init__(label, id=id)
        self.ui_action = action

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(ActionRequested(self.ui_action))


class CockpitScreen(Screen[None]):
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("q", "request_quit", "Quit"),
        ("escape", "go_back", "Back"),
    ]
    screen_id: UiScreen = UiScreen.HOME

    def compose(self) -> ComposeResult:
        yield Static("AutoBrain", id="app-header")
        with VerticalScroll(id="body"):
            yield Label(id="screen-title")
            yield Static(id="summary", markup=False)
            with Horizontal(id="actions"):
                yield ActionButton("Back", GoBack(), id="back")
                yield ActionButton("Refresh", RefreshConnections(), id="refresh")
                yield from self.screen_actions()
            yield Static(id="details", markup=False)
        yield Footer()

    def screen_actions(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.call_after_refresh(self.refresh_view)

    def action_request_quit(self) -> None:
        self.post_message(ActionRequested(RequestQuit()))

    def action_go_back(self) -> None:
        self.post_message(ActionRequested(GoBack()))

    def screen_title(self, model: object) -> str:
        del model
        return self.screen_id.value.title()

    def compact_summary(self, model: object) -> str:
        del model
        return ""

    def diagnostic_details(self, model: object) -> str:
        del model
        app = cast(AutoBrainApp, self.app)
        current = build_view_model(app.state)
        return (
            f"Provider: {current.provider.value} ({current.provider_status.value})\n"
            f"Evaluator embeddings: {current.embedding_status} - {current.embedding_detail}\n"
            f"Stage: {current.stage} {current.stage_detail}  Elapsed: {current.elapsed}\n"
            f"Terminal: {current.terminal_reason or '-'}  Report: {current.report_path or '-'}"
        )

    def refresh_view(self) -> None:
        app = cast(AutoBrainApp, self.app)
        model = build_view_model(app.state)
        self.query_one("#screen-title", Label).update(self.screen_title(model))
        self.query_one("#summary", Static).update(self.compact_summary(model))
        self.query_one("#details", Static).update(self.diagnostic_details(model))


class HomeScreen(CockpitScreen):
    screen_id = UiScreen.HOME

    def screen_actions(self) -> ComposeResult:
        yield ActionButton("Setup", BeginSetup(), id="setup")
        yield ActionButton("Run", StartRun(), id="run")


class ConnectionsScreen(CockpitScreen):
    screen_id = UiScreen.CONNECTIONS

    def screen_actions(self) -> ComposeResult:
        for provider in ProviderId:
            yield ActionButton(
                provider.value.title(),
                SelectProvider(provider),
                id=f"provider-{provider.value}",
            )
        for provider in ProviderId:
            yield ActionButton(
                f"Login {provider.value.title()}",
                RequestLogin(provider),
                id=f"login-{provider.value}",
            )
        yield ActionButton("Continue", Navigate(UiScreen.SLACK.value), id="continue")


class SlackScreen(CockpitScreen):
    screen_id = UiScreen.SLACK

    def screen_actions(self) -> ComposeResult:
        yield ActionButton("Toggle Slack", ToggleSource(Provider.SLACK), id="toggle-slack")
        yield ActionButton("Configure Slack", RequestLogin(Provider.SLACK), id="login-slack")
        yield ActionButton("Continue", Navigate(UiScreen.NOTION.value), id="continue")


class NotionScreen(CockpitScreen):
    screen_id = UiScreen.NOTION

    def screen_actions(self) -> ComposeResult:
        yield ActionButton("Toggle Notion", ToggleSource(Provider.NOTION), id="toggle-notion")
        yield ActionButton("Connect Notion", RequestLogin(Provider.NOTION), id="login-notion")
        yield ActionButton("Continue", Navigate(UiScreen.CANDIDATES.value), id="continue")


class CandidatesScreen(CockpitScreen):
    screen_id = UiScreen.CANDIDATES

    def screen_title(self, model: object) -> str:
        del model
        return "Candidates"

    def compact_summary(self, model: object) -> str:
        del model
        app = cast(AutoBrainApp, self.app)
        return (
            "Choose candidates\n"
            + "\n".join(
                (
                    f"{'[x]' if candidate in app.state.selected_candidates else '[ ]'} "
                    f"{candidate.value}"
                )
                for candidate in CandidateId
            )
            + "\n\nQuick Start: keyword-only / Not measured\nSemantic Setup: provider embeddings"
        )

    def screen_actions(self) -> ComposeResult:
        for candidate in CandidateId:
            yield ActionButton(
                candidate.value, ToggleCandidate(candidate), id=f"candidate-{candidate.value}"
            )
        yield ActionButton("Quick Start", SelectGBrainMode(False), id="quick-start")
        yield ActionButton("Semantic Setup", SelectGBrainMode(True), id="semantic-setup")


class GBrainScreen(CockpitScreen):
    screen_id = UiScreen.GBRAIN

    def screen_title(self, model: object) -> str:
        del model
        return "Semantic Setup (BYOK/local)"

    def compact_summary(self, model: object) -> str:
        del model
        return (
            "Choose one provider explicitly. Hosted keys stay in run memory; "
            "local daemons are never auto-selected."
        )

    def diagnostic_details(self, model: object) -> str:
        del model
        app = cast(AutoBrainApp, self.app)
        return app.state.gbrain_error or (
            "Validation checks endpoint availability and required fields."
        )

    def compose(self) -> ComposeResult:
        yield Static("AutoBrain", id="app-header")
        with VerticalScroll(id="body"):
            yield Label("Semantic Setup (BYOK/local)", id="screen-title")
            yield Static(
                "Choose one provider explicitly. Hosted keys stay in run memory; "
                "local daemons are never auto-selected.",
                id="summary",
            )
            yield Select(
                [
                    (provider.value, provider.value)
                    for provider in GBrainEmbeddingProvider
                    if provider is not GBrainEmbeddingProvider.KEYWORD_ONLY
                ],
                value=GBrainEmbeddingProvider.OPENAI.value,
                id="gbrain-provider",
            )
            yield Input(
                placeholder="API key (hosted providers only)", password=True, id="gbrain-key"
            )
            yield Input(placeholder="Model (required for llama-server)", id="gbrain-model")
            yield Input(
                placeholder="Dimensions (positive integer for llama-server)", id="gbrain-dimensions"
            )
            yield Input(placeholder="Endpoint (local/custom providers)", id="gbrain-endpoint")
            yield Static(id="details", markup=False)
            with Horizontal(id="actions"):
                yield ActionButton("Back", GoBack(), id="back")
                yield Button("Validate", id="validate-gbrain")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "validate-gbrain":
            return
        event.stop()
        provider_select = cast(Select[str], self.query_one("#gbrain-provider", Select))
        provider_value = provider_select.value
        if not isinstance(provider_value, str):
            self.post_message(ActionRequested(GoBack()))
            return
        provider = GBrainEmbeddingProvider(provider_value)
        raw_dimensions = self.query_one("#gbrain-dimensions", Input).value.strip()
        key_input = self.query_one("#gbrain-key", Input)
        from pydantic import SecretStr

        self.post_message(
            ActionRequested(
                ValidateGBrain(
                    provider,
                    model=self.query_one("#gbrain-model", Input).value.strip(),
                    dimensions=int(raw_dimensions) if raw_dimensions.isdigit() else None,
                    endpoint=self.query_one("#gbrain-endpoint", Input).value.strip(),
                    credential=SecretStr(key_input.value) if key_input.value else None,
                )
            )
        )
        key_input.value = ""


class ReviewScreen(CockpitScreen):
    screen_id = UiScreen.REVIEW

    def screen_title(self, model: object) -> str:
        del model
        return "Review"

    def compact_summary(self, model: object) -> str:
        del model
        app = cast(AutoBrainApp, self.app)
        config = app.state.gbrain_config
        if config.keyword_only:
            semantic = "Quick Start / keyword-only / Not measured"
        else:
            spec = config.embedding
            endpoint = spec.endpoint or "default"
            semantic = (
                f"{spec.provider.value} / {spec.model or '-'} / "
                f"{spec.dimensions or '-'} / {endpoint}"
            )
        status = (
            "Runnable"
            if app.state.plan is not None
            else "Blocked: " + (app.state.setup_error or "PLAN_UNAVAILABLE")
        )
        return (
            f"Setup: {semantic}\n{status}\n"
            f"Plan: {app.state.plan.title if app.state.plan else '-'}\n"
            f"{app.state.plan.description if app.state.plan else ''}"
        )

    def screen_actions(self) -> ComposeResult:
        yield ActionButton("Run experiment", StartRun(), id="run")

    def on_mount(self) -> None:
        super().on_mount()
        self.call_after_refresh(self._sync_run_button)

    def refresh_view(self) -> None:
        super().refresh_view()
        self._sync_run_button()

    def _sync_run_button(self) -> None:
        self.query_one("#run", Button).disabled = not build_view_model(
            cast(AutoBrainApp, self.app).state
        ).can_run


class RunningScreen(CockpitScreen):
    screen_id = UiScreen.RUNNING
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("q", "cancel_and_quit", "Cancel + quit"),
        ("c", "cancel", "Cancel"),
    ]

    def screen_actions(self) -> ComposeResult:
        yield ActionButton("Cancel", CancelRun(), id="cancel")

    def action_cancel(self) -> None:
        self.post_message(ActionRequested(CancelRun()))

    def action_cancel_and_quit(self) -> None:
        self.post_message(ActionRequested(RequestQuit()))


class ResultsScreen(CockpitScreen):
    screen_id = UiScreen.RESULTS

    def screen_actions(self) -> ComposeResult:
        yield ActionButton("Open exact report", OpenReport(), id="open-report")
        yield ActionButton("Home", ResetRun(), id="home")


SCREEN_TYPES = {
    UiScreen.HOME: HomeScreen,
    UiScreen.CONNECTIONS: ConnectionsScreen,
    UiScreen.SLACK: SlackScreen,
    UiScreen.NOTION: NotionScreen,
    UiScreen.CANDIDATES: CandidatesScreen,
    UiScreen.GBRAIN: GBrainScreen,
    UiScreen.REVIEW: ReviewScreen,
    UiScreen.RUNNING: RunningScreen,
    UiScreen.RESULTS: ResultsScreen,
}


class AutoBrainApp(App[None]):
    TITLE = "AutoBrain"
    CSS = """
    Screen { min-width: 60; }
    #app-header { height: 1; text-style: bold; background: $primary; color: $text; padding: 0 1; }
    #body { width: 100%; padding: 0 1; }
    #screen-title { text-style: bold; color: $accent; }
    #summary { width: 100%; height: auto; }
    #details { width: 100%; height: auto; color: $text-muted; }
    #actions { height: auto; layout: grid; grid-size: 3; grid-gutter: 0 1; }
    Button { margin: 0; min-width: 12; height: 1; border: none; padding: 0 1; }
    GBrainScreen Input, GBrainScreen Select { height: 1; border: none; padding: 0 1; }
    """

    def __init__(self, *, force_setup: bool, provider: ProviderId) -> None:
        super().__init__()
        self.state = UiState(
            screen=UiScreen.CONNECTIONS if force_setup else UiScreen.HOME,
            provider=provider,
        )
        self.effect_registry = EffectRegistry()

    def on_mount(self) -> None:
        screen_type = SCREEN_TYPES[self.state.screen]
        self.push_screen(cast(Screen[None], screen_type()))
        self.call_after_refresh(self.dispatch_ui, RefreshConnections())

    def on_action_requested(self, message: ActionRequested) -> None:
        self.dispatch_ui(message.action)

    def dispatch_ui(self, action: UiAction) -> None:
        reduction = reduce_ui(self.state, action)
        previous_screen = self.state.screen
        self.state = reduction.state
        for effect in reduction.effects:
            self._execute(effect)
        if self.state.screen is not previous_screen:
            self._show_state_screen()
        elif isinstance(self.screen, CockpitScreen):
            self.screen.refresh_view()

    def _show_state_screen(self) -> None:
        screen_type = SCREEN_TYPES[self.state.screen]
        switch_screen = cast(Callable[[Screen[None]], object], self.switch_screen)
        switch_screen(cast(Screen[None], screen_type()))

    def _execute(self, effect: UiEffect) -> None:
        if isinstance(effect, LoadConnections):
            self.load_connections(effect)
        elif isinstance(effect, InteractiveLogin):
            suspension = self.suspend()
            suspension.__enter__()
            self.effect_registry.register_login(effect.handle, suspension)
            try:
                self.interactive_login(effect)
            except BaseException:
                registered = self.effect_registry.settle_login(effect.handle)
                if registered is not None:
                    registered.__exit__(None, None, None)
                raise
        elif isinstance(effect, ValidateGBrainProvider):
            self.validate_gbrain(effect)
        elif isinstance(effect, RunExperiment):
            started_at = time.monotonic()
            cancellation = self.effect_registry.register_run(effect.handle, started_at=started_at)
            self.dispatch_ui(RunStarted(effect.handle, started_at))
            self.run_experiment(effect, cancellation)
        elif isinstance(effect, CancelActiveRun):
            self.effect_registry.cancel_run(effect.handle)
        elif isinstance(effect, OpenExactReport):
            open_exact_report(effect)
        else:
            self.exit()

    @work(thread=True, exclusive=True, group="connections")
    def load_connections(self, effect: LoadConnections) -> None:
        try:
            snapshot = connection_snapshot(
                subscription_provider=effect.provider,
                refresh_subscription=effect.refresh,
            )
            self.call_from_thread(self.dispatch_ui, ConnectionsLoaded(snapshot))
        except Exception as exc:
            self.call_from_thread(
                self.dispatch_ui,
                RunFailed(
                    "CONNECTION_PROBE_FAILED: "
                    f"{_safe_worker_error_name('CONNECTION_PROBE_FAILED', exc)}"
                ),
            )

    @work(thread=True, exclusive=True, group="login", exit_on_error=False)
    def interactive_login(self, effect: InteractiveLogin) -> None:
        run_login_process(effect)

    @work(thread=True, exclusive=True, group="gbrain", exit_on_error=False)
    def validate_gbrain(self, effect: ValidateGBrainProvider) -> GBrainExecutionConfig:
        return validate_gbrain_connection(effect.config)

    @work(thread=True, exclusive=True, group="run", exit_on_error=False)
    def run_experiment(
        self,
        effect: RunExperiment,
        cancellation: RunCancellation,
    ) -> RunResult:
        result_queue: queue.Queue[RunResult | BaseException] = queue.Queue(maxsize=1)

        def observed(event: StageEvent) -> None:
            self.call_from_thread(
                self.dispatch_ui,
                StageObserved(event, observed_at=time.monotonic()),
            )

        execute_plan(
            effect.plan,
            result_queue,
            effect.provider,
            observed,
            cancellation,
        )
        completed = result_queue.get()
        if isinstance(completed, BaseException):
            raise completed
        return completed

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state not in {WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED}:
            return
        worker = cast(Worker[RunResult | None], event.worker)
        if worker.group == "login":
            handle = self.state.active_login_handle
            if handle is None:
                return
            suspension = self.effect_registry.settle_login(handle)
            if suspension is not None:
                suspension.__exit__(None, None, None)
            error = (
                _safe_worker_error_name("LOGIN_FAILED", worker.error)
                if event.state is WorkerState.ERROR
                else ""
            )
            self.dispatch_ui(LoginSettled(handle, error))
            return
        if worker.group == "gbrain":
            if event.state is WorkerState.SUCCESS:
                self.dispatch_ui(GBrainValidated(cast(GBrainExecutionConfig, worker.result)))
            else:
                safe_error = str(worker.error or "")
                if not safe_error.startswith("GBRAIN_LOCAL_ENDPOINT_UNAVAILABLE:"):
                    error_name = _safe_worker_error_name(
                        "GBRAIN_PROVIDER_UNAVAILABLE", worker.error
                    )
                    safe_error = f"GBRAIN_PROVIDER_UNAVAILABLE: {error_name}"
                self.dispatch_ui(GBrainValidated(error=safe_error))
            return
        if worker.group != "run":
            return
        handle = self.state.active_run_handle
        if handle is None:
            return
        self.effect_registry.settle_run(handle)
        quit_after = self.state.quit_after_settlement
        if event.state is WorkerState.SUCCESS:
            self.dispatch_ui(RunCompleted(cast(RunResult, worker.result)))
        elif event.state is WorkerState.ERROR:
            self.dispatch_ui(
                RunFailed(f"RUN_FAILED: {_safe_worker_error_name('RUN_FAILED', worker.error)}")
            )
        else:
            self.dispatch_ui(RunFailed("RUN_CANCELLED: worker cancelled"))
        if quit_after:
            self.dispatch_ui(RequestQuit())


def run_textual(*, force_setup: bool = False, provider: ProviderId = ProviderId.CODEX) -> None:
    AutoBrainApp(force_setup=force_setup, provider=provider).run()
