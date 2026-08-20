"""Actual Textual application, screens, widgets, workers, and messages."""

from __future__ import annotations

import queue
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import ClassVar

from autobrain.auth.models import Provider
from autobrain.models import CandidateId
from autobrain.orchestration import RunCancellation, RunResult, StageEvent
from autobrain.subscription_domain import ProviderId
from autobrain.tui_actions import (
    BeginSetup,
    CancelRun,
    ConnectionsLoaded,
    GoBack,
    Navigate,
    OpenReport,
    RefreshConnections,
    RequestLogin,
    ResetRun,
    RunCompleted,
    RunFailed,
    SelectProvider,
    StageObserved,
    StartRun,
    ToggleCandidate,
    ToggleSource,
    UiAction,
)
from autobrain.tui_effects import (
    CancelActiveRun,
    ExitApplication,
    InteractiveLogin,
    LoadConnections,
    OpenExactReport,
    RunExperiment,
    UiEffect,
    execute_interactive_login,
    open_exact_report,
)
from autobrain.tui_runtime import connection_snapshot, execute_plan
from autobrain.tui_state import UiScreen, UiState, reduce_ui
from autobrain.tui_viewmodels import build_view_model

try:
    from textual import work
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, VerticalScroll
    from textual.message import Message
    from textual.screen import Screen
    from textual.widgets import Button, Footer, Header, Label, Static
    from textual.worker import Worker, WorkerState
except ImportError as exc:  # dependency is an explicit runtime requirement
    _TEXTUAL_IMPORT_ERROR = exc
else:
    _TEXTUAL_IMPORT_ERROR = None


if _TEXTUAL_IMPORT_ERROR is None:

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
            yield Header(show_clock=True)
            with VerticalScroll(id="body"):
                yield Label(id="screen-title")
                yield Static(id="summary")
                with Horizontal(id="actions"):
                    yield ActionButton("Back", GoBack(), id="back")
                    yield ActionButton("Refresh", RefreshConnections(), id="refresh")
                    yield from self.screen_actions()
            yield Footer()

        def screen_actions(self) -> ComposeResult:
            yield from ()

        def on_mount(self) -> None:
            self.refresh_view()

        def on_action_requested(self, message: ActionRequested) -> None:
            self.app.dispatch_ui(message.action)  # type: ignore[attr-defined]

        def action_request_quit(self) -> None:
            if self.app.state.running:  # type: ignore[attr-defined]
                self.app.dispatch_ui(CancelRun())  # type: ignore[attr-defined]
            else:
                self.app.exit()

        def action_go_back(self) -> None:
            self.app.dispatch_ui(GoBack())  # type: ignore[attr-defined]

        def refresh_view(self) -> None:
            model = build_view_model(self.app.state)  # type: ignore[attr-defined]
            self.query_one("#screen-title", Label).update(
                f"{self.screen_id.value.title()}  |  provider: {model.provider.value} "
                f"({model.provider_status.value})"
            )
            sources = "\n".join(
                f"{'[x]' if selected else '[ ]'} {source.value}: {status.value} {detail}"
                for source, status, selected, detail in model.sources
            )
            candidates = "\n".join(
                f"{'[x]' if item.selected else '[ ]'} {candidate.value}: {item.status}"
                for candidate, item in model.candidates.items()
            )
            self.query_one("#summary", Static).update(
                f"{sources}\n\n{candidates}\n\n"
                f"Plan: {model.plan_title or '-'}\n{model.plan_description}\n"
                f"Stage: {model.stage} {model.stage_detail}\nElapsed: {model.elapsed}\n"
                f"Terminal: {model.terminal_reason or '-'}\n"
                f"Report: {model.report_path or '-'}\n{model.setup_error}"
            )

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

        def screen_actions(self) -> ComposeResult:
            for candidate in CandidateId:
                yield ActionButton(
                    candidate.value, ToggleCandidate(candidate), id=f"candidate-{candidate.value}"
                )
            yield ActionButton("Review", Navigate(UiScreen.REVIEW.value), id="review")

    class ReviewScreen(CockpitScreen):
        screen_id = UiScreen.REVIEW

        def screen_actions(self) -> ComposeResult:
            yield ActionButton("Run experiment", StartRun(), id="run")

    class RunningScreen(CockpitScreen):
        screen_id = UiScreen.RUNNING
        BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
            ("q", "cancel_and_quit", "Cancel + quit"),
            ("c", "cancel", "Cancel"),
        ]

        def screen_actions(self) -> ComposeResult:
            yield ActionButton("Cancel", CancelRun(), id="cancel")

        def action_cancel(self) -> None:
            self.app.dispatch_ui(CancelRun())  # type: ignore[attr-defined]

        def action_cancel_and_quit(self) -> None:
            self.app.quit_after_settlement = True  # type: ignore[attr-defined]
            self.app.dispatch_ui(CancelRun())  # type: ignore[attr-defined]

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
        UiScreen.REVIEW: ReviewScreen,
        UiScreen.RUNNING: RunningScreen,
        UiScreen.RESULTS: ResultsScreen,
    }

    class AutoBrainApp(App[None]):
        TITLE = "AutoBrain"
        CSS = """
        Screen { min-width: 60; }
        #body { width: 100%; padding: 1 2; }
        #screen-title { text-style: bold; color: $accent; margin-bottom: 1; }
        #summary { width: 100%; min-height: 12; }
        #actions { height: auto; }
        Button { margin: 0 1 1 0; }
        """

        def __init__(self, *, force_setup: bool, provider: ProviderId) -> None:
            super().__init__()
            self.state = UiState(
                screen=UiScreen.CONNECTIONS if force_setup else UiScreen.HOME,
                provider=provider,
            )
            self.run_cancellation: RunCancellation | None = None
            self.run_started_at = 0.0
            self.quit_after_settlement = False

        def on_mount(self) -> None:
            self._show_state_screen()
            self.dispatch_ui(RefreshConnections())

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
            if self.screen_stack:
                self.switch_screen(screen_type())
            else:
                self.push_screen(screen_type())

        def _execute(self, effect: UiEffect) -> None:
            if isinstance(effect, LoadConnections):
                self.load_connections(effect)
            elif isinstance(effect, InteractiveLogin):
                self.interactive_login(effect)
            elif isinstance(effect, RunExperiment):
                self.run_experiment(effect)
            elif isinstance(effect, CancelActiveRun):
                if self.run_cancellation is not None:
                    self.run_cancellation.cancel()
            elif isinstance(effect, OpenExactReport):
                open_exact_report(effect)
            elif isinstance(effect, ExitApplication):
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
                    self.dispatch_ui, RunFailed(f"CONNECTION_PROBE_FAILED: {exc}")
                )

        @contextmanager
        def suspended(self) -> Iterator[None]:
            with self.suspend():
                yield

        @work(thread=True, exclusive=True, group="login")
        def interactive_login(self, effect: InteractiveLogin) -> None:
            try:
                execute_interactive_login(effect, terminal=self)
                snapshot = connection_snapshot(
                    subscription_provider=self.state.provider,
                    refresh_subscription=True,
                )
                self.call_from_thread(self.dispatch_ui, ConnectionsLoaded(snapshot))
            except Exception as exc:
                self.call_from_thread(self.dispatch_ui, RunFailed(f"LOGIN_FAILED: {exc}"))

        @work(thread=True, exclusive=True, group="run", exit_on_error=False)
        def run_experiment(self, effect: RunExperiment) -> RunResult:
            result_queue: queue.Queue[RunResult | BaseException] = queue.Queue(maxsize=1)
            cancellation = RunCancellation()
            self.run_cancellation = cancellation
            self.run_started_at = time.monotonic()

            def observed(event: StageEvent) -> None:
                elapsed = int(time.monotonic() - self.run_started_at)
                self.call_from_thread(self.dispatch_ui, StageObserved(event, elapsed))

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
            if event.worker.group != "run":
                return
            if event.state is WorkerState.SUCCESS:
                self.dispatch_ui(RunCompleted(event.worker.result))
                if self.quit_after_settlement:
                    self.exit()
            elif event.state is WorkerState.ERROR:
                error = event.worker.error
                self.dispatch_ui(RunFailed(f"RUN_FAILED: {type(error).__name__}: {error}"))
                if self.quit_after_settlement:
                    self.exit()


def run_textual(*, force_setup: bool = False, provider: ProviderId = ProviderId.CODEX) -> None:
    if _TEXTUAL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "TUI_UNAVAILABLE: install the Textual dependency to launch the default UI"
        ) from _TEXTUAL_IMPORT_ERROR
    AutoBrainApp(force_setup=force_setup, provider=provider).run()
