from __future__ import annotations

import ast
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path

from autobrain.auth.models import Provider
from autobrain.models import CandidateId, ConnectionState, Status
from autobrain.orchestration import RunResult, StageEvent
from autobrain.subscription import ProviderId, SubscriptionStatus
from autobrain.tui_actions import (
    CancelRun,
    ConnectionsLoaded,
    OpenReport,
    RunCompleted,
    SelectProvider,
    StageObserved,
    StartRun,
    ToggleCandidate,
)
from autobrain.tui_effects import (
    CancelActiveRun,
    InteractiveLogin,
    OpenExactReport,
    RunExperiment,
    execute_interactive_login,
)
from autobrain.tui_runtime import ConnectionSnapshot
from autobrain.tui_state import UiScreen, UiState, reduce_ui
from autobrain.tui_viewmodels import build_view_model


def _connections(provider: ProviderId = ProviderId.CODEX) -> ConnectionSnapshot:
    return ConnectionSnapshot(
        subscription=SubscriptionStatus.READY,
        sources={
            Provider.SLACK: ConnectionState.CONNECTED,
            Provider.NOTION: ConnectionState.CONNECTED,
        },
        subscription_provider=provider,
    )


def test_reducer_is_immutable_and_emits_semantic_effects() -> None:
    initial = UiState(screen=UiScreen.CONNECTIONS)

    selected = reduce_ui(initial, SelectProvider(ProviderId.CLAUDE))

    assert initial.provider is ProviderId.CODEX
    assert selected.state.provider is ProviderId.CLAUDE
    assert selected.effects
    assert initial is not selected.state

    loaded = reduce_ui(selected.state, ConnectionsLoaded(_connections(ProviderId.CLAUDE)))
    candidate = loaded.state.selected_candidates[-1]
    toggled = reduce_ui(loaded.state, ToggleCandidate(candidate))
    assert candidate in loaded.state.selected_candidates
    assert candidate not in toggled.state.selected_candidates


def test_run_stage_cancel_completion_and_exact_report_are_reduced_purely(tmp_path: Path) -> None:
    state = reduce_ui(UiState(), ConnectionsLoaded(_connections())).state
    started = reduce_ui(state, StartRun())
    assert started.state.screen is UiScreen.RUNNING
    assert isinstance(started.effects[-1], RunExperiment)

    event = StageEvent(
        sequence=7,
        run_id="run-1",
        name="candidate:mem0",
        status=Status.OK,
        detail="20 cases",
        started_at="2026-08-20T00:00:00+00:00",
    )
    observed = reduce_ui(started.state, StageObserved(event, elapsed_seconds=9))
    running = build_view_model(observed.state)
    assert running.stage == "candidate:mem0"
    assert running.elapsed == "00:09"
    assert running.candidates[CandidateId.MEM0].status == "OK"

    cancelling = reduce_ui(observed.state, CancelRun())
    assert cancelling.state.cancelling is True
    assert cancelling.effects == (CancelActiveRun(),)

    report = tmp_path / "report.html"
    result = RunResult(
        run_id="run-1",
        run_dir=tmp_path,
        status=Status.CANCELLED,
        report_path=report,
        candidate_results=(),
        verdict="NO_DECISION",
    )
    completed = reduce_ui(cancelling.state, RunCompleted(result))
    assert completed.state.screen is UiScreen.RESULTS
    assert completed.state.terminal_reason == Status.CANCELLED.value
    opening = reduce_ui(completed.state, OpenReport())
    assert opening.effects == (OpenExactReport(report),)


def test_interactive_login_always_restores_terminal() -> None:
    lifecycle: list[str] = []

    class Terminal:
        @contextmanager
        def suspended(self) -> Generator[None]:
            lifecycle.append("suspend")
            try:
                yield
            finally:
                lifecycle.append("restore")

    def fail(effect: InteractiveLogin) -> None:
        del effect
        lifecycle.append("login")
        raise RuntimeError("login failed")

    with suppress(RuntimeError):
        execute_interactive_login(
            InteractiveLogin(ProviderId.CLAUDE),
            terminal=Terminal(),
            login=fail,
        )

    assert lifecycle == ["suspend", "login", "restore"]


def test_default_and_shared_tui_modules_do_not_import_curses() -> None:
    root = Path(__file__).parents[1] / "src" / "autobrain"
    shared = {
        "tui.py",
        "tui_actions.py",
        "tui_effects.py",
        "tui_runtime.py",
        "tui_state.py",
        "tui_textual.py",
        "tui_viewmodels.py",
    }
    for name in shared:
        tree = ast.parse((root / name).read_text(encoding="utf-8"), filename=name)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        )
        assert "curses" not in imported, name

    legacy = ast.parse((root / "tui_legacy.py").read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.Import) and any(alias.name == "curses" for alias in node.names)
        for node in ast.walk(legacy)
    )
