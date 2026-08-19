"""Keyboard-first terminal cockpit for AutoBrain experiments."""

from __future__ import annotations

import curses
import queue
import sys
import threading
import time
import webbrowser

from autobrain.auth.models import Provider
from autobrain.experiment import ExperimentPlan
from autobrain.models import CandidateId
from autobrain.orchestration import RunResult
from autobrain.tui_render import render_dashboard, terminal_too_small
from autobrain.tui_runtime import (
    ConnectionSnapshot,
    connection_key,
    connection_snapshot,
    execute_plan,
    resolve_plan,
    run_connection_flow,
)
from autobrain.tui_state import TUIState, WizardSection


def run_tui() -> None:
    """Launch the full-screen terminal UI."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError(
            "TUI_UNAVAILABLE: run AutoBrain in an interactive terminal or use `autobrain run`"
        )
    curses.wrapper(_run)


def accepts_key_at_size(key: int, *, width: int, height: int) -> bool:
    if not terminal_too_small(width=width, height=height):
        return True
    return key in {ord("q"), ord("Q")}


def accepts_key_for_state(
    state: TUIState,
    key: int,
    *,
    width: int,
    height: int,
) -> bool:
    if not accepts_key_at_size(key, width=width, height=height):
        return False
    return state.section is not WizardSection.RUNNING


def _run(screen: curses.window) -> None:
    curses.curs_set(0)
    curses.use_default_colors()
    screen.timeout(120)
    state = TUIState()
    connections = connection_snapshot()
    result_queue: queue.Queue[RunResult | BaseException] = queue.Queue(maxsize=1)
    result: RunResult | None = None
    started_at = 0.0
    runtime_error = ""

    while True:
        plan, setup_error = resolve_plan(
            selected_sources=state.selected_sources,
            selected_candidates=state.selected_candidates,
            connections=connections,
        )
        setup_error = runtime_error or setup_error
        _draw(
            screen,
            state,
            connections,
            plan,
            setup_error,
            result,
            int(time.monotonic() - started_at) if started_at else 0,
        )
        if state.section is WizardSection.RUNNING:
            try:
                completed = result_queue.get_nowait()
            except queue.Empty:
                completed = None
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
        if key in {ord("q"), ord("Q")} and state.section is not WizardSection.RUNNING:
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

        if state.section is WizardSection.CONNECTIONS:
            provider = connection_key(key)
            if provider is not None:
                run_connection_flow(screen, provider)
                connections = connection_snapshot()
                runtime_error = ""
        elif state.section is WizardSection.KNOWLEDGE_SOURCES:
            if key == ord("1"):
                state = state.toggle_source(Provider.SLACK)
                runtime_error = ""
            elif key == ord("2"):
                state = state.toggle_source(Provider.NOTION)
                runtime_error = ""
        elif state.section is WizardSection.CANDIDATES:
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
                state = state.with_section(WizardSection.REVIEW)

        if key in {10, 13, curses.KEY_ENTER}:
            if state.section is WizardSection.REVIEW and plan is not None:
                state = state.with_section(WizardSection.RUNNING)
                started_at = time.monotonic()
                threading.Thread(
                    target=execute_plan,
                    args=(plan, result_queue),
                    daemon=True,
                ).start()
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
    )
    column = 1 if width > 2 else 0
    for row, line in enumerate(lines[: max(1, height - 1)]):
        try:
            style = curses.A_BOLD if row == 0 or line.startswith(">") else curses.A_NORMAL
            screen.addstr(row, column, line, style)
        except curses.error:
            break
    screen.refresh()
