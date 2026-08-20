"""Keyboard-first terminal cockpit for AutoBrain experiments."""

from __future__ import annotations

import curses
import queue
import sys
import time
import webbrowser
from collections.abc import Callable

from autobrain.auth.models import Provider
from autobrain.experiment import ExperimentPlan
from autobrain.models import CandidateId, ConnectionState
from autobrain.onboarding import is_onboarded, mark_onboarded
from autobrain.orchestration import RunResult, StageEvent
from autobrain.paths import AutoBrainPaths
from autobrain.subscription import ProviderId, SubscriptionStatus
from autobrain.tui_legacy_runtime import (
    ConnectionSnapshot,
    PlanWorker,
    connection_snapshot,
    resolve_plan,
    start_plan_worker,
)
from autobrain.tui_legacy_runtime import (
    run_connection_flow as _run_connection_flow,
)
from autobrain.tui_render import render_dashboard, terminal_too_small
from autobrain.tui_state import TUIState, WizardSection
from autobrain.tui_style import line_style


def run_connection_flow(screen: curses.window, provider: Provider | ProviderId) -> None:
    curses.def_prog_mode()
    curses.endwin()
    try:
        _run_connection_flow(screen, provider)
    finally:
        curses.reset_prog_mode()
        screen.refresh()


def run_tui(*, force_setup: bool = False, provider: ProviderId = ProviderId.CODEX) -> None:
    """Launch the full-screen terminal UI."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError(
            "TUI_UNAVAILABLE: run AutoBrain in an interactive terminal or use `autobrain run`"
        )
    curses.wrapper(lambda screen: _run(screen, force_setup=force_setup, provider=provider))


def accepts_key_at_size(key: int, *, width: int, height: int) -> bool:
    if not terminal_too_small(width=width, height=height):
        return True
    return key in {ord("q"), ord("Q")}


def subscription_provider_key(key: int) -> ProviderId | None:
    return {
        ord("1"): ProviderId.CODEX,
        ord("2"): ProviderId.CLAUDE,
        ord("3"): ProviderId.KIMI,
        ord("4"): ProviderId.GROK,
    }.get(key)


def select_subscription_provider(
    state: TUIState,
    key: int,
    *,
    snapshot: Callable[..., ConnectionSnapshot] = connection_snapshot,
) -> tuple[TUIState, ConnectionSnapshot | None]:
    provider = subscription_provider_key(key)
    if provider is None:
        return state, None
    selected = state.with_subscription_provider(provider)
    return (
        selected,
        snapshot(
            subscription_provider=provider,
            refresh_subscription=True,
        ),
    )


def accepts_key_for_state(
    state: TUIState,
    key: int,
    *,
    width: int,
    height: int,
) -> bool:
    if not accepts_key_at_size(key, width=width, height=height):
        return False
    if state.section is WizardSection.RUNNING:
        return key in {ord("q"), ord("Q"), ord("c"), ord("C")}
    return True


def _run(screen: curses.window, *, force_setup: bool, provider: ProviderId) -> None:
    curses.curs_set(0)
    curses.use_default_colors()
    if curses.has_colors():
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
    screen.timeout(120)
    paths = AutoBrainPaths.from_home()
    if force_setup or not is_onboarded(paths):
        state = TUIState(subscription_provider=provider)
    else:
        state = TUIState(section=WizardSection.HOME, subscription_provider=provider)
    connections = connection_snapshot(subscription_provider=state.subscription_provider)
    result_queue: queue.Queue[RunResult | BaseException] = queue.Queue(maxsize=1)
    result: RunResult | None = None
    started_at = 0.0
    runtime_error = ""
    latest_stage: StageEvent | None = None
    runtime_worker: PlanWorker | None = None

    def record_stage(event: StageEvent) -> None:
        nonlocal latest_stage
        latest_stage = event

    while True:
        plan, setup_error = resolve_plan(
            selected_sources=state.selected_sources,
            selected_candidates=state.selected_candidates,
            connections=connections,
            subscription_provider=state.subscription_provider,
        )
        setup_error = runtime_error or setup_error
        if state.section is WizardSection.REVIEW and plan is not None:
            mark_onboarded(paths)
        _draw(
            screen,
            state,
            connections,
            plan,
            setup_error,
            result,
            int(time.monotonic() - started_at) if started_at else 0,
            latest_stage,
        )
        if state.section is WizardSection.RUNNING:
            try:
                completed = result_queue.get_nowait()
            except queue.Empty:
                completed = None
            if completed is not None and runtime_worker is not None:
                if not runtime_worker.join(timeout=0.5):
                    runtime_error = "RUN_SETTLEMENT_TIMEOUT: worker did not exit after result"
                runtime_worker = None
            if isinstance(completed, BaseException):
                runtime_error = f"RUN_FAILED: {type(completed).__name__}: {completed}"
                state = state.with_section(WizardSection.REVIEW)
            elif completed is not None:
                result = completed
                runtime_error = ""
                state = state.with_section(WizardSection.RESULTS)

        key = screen.getch()
        current_height, current_width = screen.getmaxyx()
        if not accepts_key_for_state(
            state,
            key,
            width=current_width,
            height=current_height,
        ):
            continue
        if state.section is WizardSection.RUNNING and key in {
            ord("q"),
            ord("Q"),
            ord("c"),
            ord("C"),
        }:
            settled = runtime_worker is not None and runtime_worker.cancel_and_join(timeout=2.0)
            if not settled:
                runtime_error = "RUN_CANCELLATION_TIMEOUT: worker did not settle"
                continue
            if key in {ord("q"), ord("Q")}:
                return
            continue
        if key in {ord("q"), ord("Q")}:
            return
        if key in {ord("b"), ord("B"), curses.KEY_BACKSPACE}:
            runtime_error = ""
            state = state.back()
            continue
        if key in {curses.KEY_UP, curses.KEY_BTAB}:
            runtime_error = ""
            state = state.back()
            continue
        if key in {curses.KEY_DOWN, 9}:
            runtime_error = ""
            state = state.advance()
            continue

        if (
            state.section is WizardSection.CONNECTIONS
            and subscription_provider_key(key) is not None
        ):
            state, selected_connections = select_subscription_provider(state, key)
            assert selected_connections is not None
            connections = selected_connections
            runtime_error = ""
            continue
        if state.section is WizardSection.CONNECTIONS and key in {ord("r"), ord("R")}:
            connections = connection_snapshot(
                subscription_provider=state.subscription_provider,
                refresh_subscription=True,
            )
            runtime_error = ""
            continue
        if state.section is WizardSection.HOME and key in {ord("s"), ord("S")}:
            state = state.start_setup()
            runtime_error = ""
            continue
        if state.section is WizardSection.SLACK and key in {ord("s"), ord("S")}:
            state = state.skip_source(Provider.SLACK)
            runtime_error = ""
            continue
        if state.section is WizardSection.NOTION and key in {ord("s"), ord("S")}:
            state = state.skip_source(Provider.NOTION)
            runtime_error = ""
            continue
        if state.section is WizardSection.CANDIDATES:
            candidates = {
                ord("1"): CandidateId.LLM_WIKI,
                ord("2"): CandidateId.MEM0,
                ord("3"): CandidateId.GBRAIN,
            }
            if key in candidates:
                state = state.toggle_candidate(candidates[key])
                runtime_error = ""
        elif state.section is WizardSection.RESULTS:
            if key in {ord("o"), ord("O")} and result and result.report_path:
                webbrowser.open(result.report_path.as_uri())
            elif key in {ord("r"), ord("R")}:
                result = None
                state = TUIState(
                    section=WizardSection.HOME,
                    subscription_provider=state.subscription_provider,
                )

        if key in {10, 13, curses.KEY_ENTER}:
            if state.section is WizardSection.HOME:
                if plan is None:
                    state = state.start_setup()
                else:
                    state = state.with_section(WizardSection.RUNNING)
                    latest_stage = None
                    started_at = time.monotonic()
                    runtime_worker = start_plan_worker(
                        plan,
                        result_queue,
                        record_stage,
                        state.subscription_provider,
                    )
            elif state.section is WizardSection.CONNECTIONS:
                if connections.subscription is not SubscriptionStatus.READY:
                    run_connection_flow(screen, state.subscription_provider)
                    connections = connection_snapshot(
                        subscription_provider=state.subscription_provider,
                        refresh_subscription=True,
                    )
                    runtime_error = ""
                    if connections.subscription is SubscriptionStatus.READY:
                        state = state.advance()
                else:
                    state = state.advance()
            elif state.section is WizardSection.SLACK:
                slack_ready = connections.sources.get(Provider.SLACK) is ConnectionState.CONNECTED
                if Provider.SLACK in state.selected_sources and not slack_ready:
                    run_connection_flow(screen, Provider.SLACK)
                    connections = connection_snapshot(
                        subscription_provider=state.subscription_provider,
                    )
                    runtime_error = ""
                    if connections.sources.get(Provider.SLACK) is ConnectionState.CONNECTED:
                        state = state.advance()
                else:
                    state = state.advance()
            elif state.section is WizardSection.NOTION:
                notion_ready = connections.sources.get(Provider.NOTION) is ConnectionState.CONNECTED
                if Provider.NOTION in state.selected_sources and not notion_ready:
                    run_connection_flow(screen, Provider.NOTION)
                    connections = connection_snapshot(
                        subscription_provider=state.subscription_provider,
                    )
                    runtime_error = ""
                    if connections.sources.get(Provider.NOTION) is ConnectionState.CONNECTED:
                        state = state.advance()
                else:
                    state = state.advance()
            elif state.section is WizardSection.REVIEW and plan is not None:
                state = state.with_section(WizardSection.RUNNING)
                latest_stage = None
                started_at = time.monotonic()
                runtime_worker = start_plan_worker(
                    plan,
                    result_queue,
                    record_stage,
                    state.subscription_provider,
                )
            else:
                state = state.advance()


def _draw(
    screen: curses.window,
    state: TUIState,
    connections: ConnectionSnapshot,
    plan: ExperimentPlan | None,
    setup_error: str,
    result: RunResult | None,
    elapsed_seconds: int,
    latest_stage: StageEvent | None,
) -> None:
    screen.erase()
    height, width = screen.getmaxyx()
    lines = render_dashboard(
        section=state.section.value,
        selected_sources=state.selected_sources,
        selected_candidates=state.selected_candidates,
        source_states=connections.sources,
        subscription_status=connections.subscription,
        plan=plan,
        setup_error=setup_error,
        result=result,
        elapsed_seconds=elapsed_seconds,
        width=width,
        height=height,
        source_details=connections.source_details,
        latest_stage=latest_stage,
        subscription_provider=state.subscription_provider,
    )
    column = 1 if width > 2 else 0
    for row, line in enumerate(lines[: max(1, height - 1)]):
        try:
            screen.addstr(row, column, line, line_style(line, row))
        except curses.error:
            break
    screen.refresh()
