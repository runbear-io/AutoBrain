from __future__ import annotations

import ast
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path

from autobrain.auth.models import Provider
from autobrain.embedding import EmbeddingBackendConfig
from autobrain.models import CandidateId, ConnectionState, Status
from autobrain.orchestration import RunResult, StageEvent
from autobrain.subscription import ProviderId, SubscriptionStatus
from autobrain.tui_actions import (
    CancelRun,
    ConnectionsLoaded,
    LoginSettled,
    OpenReport,
    RequestLogin,
    RequestQuit,
    ResetRun,
    RunCompleted,
    RunStarted,
    SelectProvider,
    StageObserved,
    StartRun,
    ToggleCandidate,
)
from autobrain.tui_effects import (
    CancelActiveRun,
    EffectHandle,
    EffectRegistry,
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
        embeddings=EmbeddingBackendConfig.from_environ(
            {"OPENAI_API_KEY": "fixture"}, requested="openai"
        ).readiness(),
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


def test_lifecycle_identity_timestamps_and_quit_intent_live_in_immutable_state() -> None:
    state = reduce_ui(UiState(), ConnectionsLoaded(_connections())).state
    started = reduce_ui(state, StartRun())
    effect = started.effects[-1]
    assert isinstance(effect, RunExperiment)
    assert started.state.active_run_handle == effect.handle
    assert started.state.run_started_at is None

    running = reduce_ui(started.state, RunStarted(effect.handle, started_at=41.5))
    assert running.state.run_started_at == 41.5
    quitting = reduce_ui(running.state, RequestQuit())
    assert quitting.state.quit_after_settlement is True
    assert quitting.effects == (CancelActiveRun(effect.handle),)

    idle_quit = reduce_ui(UiState(), RequestQuit())
    assert idle_quit.effects[-1].__class__.__name__ == "ExitApplication"


def test_login_lifecycle_is_reduced_by_handle_and_refreshes_after_restore() -> None:
    requested = reduce_ui(UiState(), RequestLogin(ProviderId.CLAUDE))
    effect = requested.effects[-1]
    assert isinstance(effect, InteractiveLogin)
    assert requested.state.active_login_handle == effect.handle
    settled = reduce_ui(requested.state, LoginSettled(effect.handle))
    assert settled.state.active_login_handle is None
    assert settled.effects[-1].__class__.__name__ == "LoadConnections"


def test_effect_registry_owns_cancellation_handles() -> None:
    registry = EffectRegistry()
    handle = EffectHandle("run-1")
    cancellation = registry.register_run(handle, started_at=10.0)
    assert registry.run_started_at(handle) == 10.0
    assert registry.cancel_run(handle) is True
    assert cancellation.cancelled is True
    assert registry.settle_run(handle) is cancellation
    assert registry.cancel_run(handle) is False


def test_reset_is_explicit_and_unknown_actions_are_rejected() -> None:
    reset = reduce_ui(UiState(screen=UiScreen.RESULTS), ResetRun())
    assert reset.state.screen is UiScreen.HOME
    with __import__("pytest").raises(AssertionError):
        reduce_ui(UiState(), object())  # type: ignore[arg-type]


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
    effect = started.effects[-1]
    assert isinstance(effect, RunExperiment)
    running_state = reduce_ui(started.state, RunStarted(effect.handle, started_at=10.0)).state
    observed = reduce_ui(running_state, StageObserved(event, observed_at=19.0))
    running = build_view_model(observed.state)
    assert running.stage == "candidate:mem0"
    assert running.elapsed == "00:09"
    assert running.candidates[CandidateId.MEM0].status == "OK"

    cancelling = reduce_ui(observed.state, CancelRun())
    assert cancelling.state.cancelling is True
    assert cancelling.effects == (CancelActiveRun(effect.handle),)

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
            InteractiveLogin(ProviderId.CLAUDE, EffectHandle("login-test")),
            terminal=Terminal(),
            login=fail,
        )

    assert lifecycle == ["suspend", "login", "restore"]


def test_textual_widgets_have_no_direct_exit_or_parallel_lifecycle_mutation() -> None:
    path = Path(__file__).parents[1] / "src" / "autobrain" / "tui_textual.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    forbidden_attributes = {"run_cancellation", "run_started_at", "quit_after_settlement"}
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr in forbidden_attributes
        for node in ast.walk(tree)
    )
    screen_classes = {
        node.name
        for node in tree.body
        if isinstance(node, ast.If)
        for child in node.body
        if isinstance(child, ast.ClassDef)
        for node in [child]
        if any(
            isinstance(base, ast.Name) and base.id in {"Screen", "CockpitScreen", "Button"}
            for base in child.bases
        )
    }
    methods = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    execute = methods["_execute"]
    login_worker = methods["interactive_login"]
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "suspend"
        for node in ast.walk(execute)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"suspend", "call_from_thread", "dispatch_ui"}
        for node in ast.walk(login_worker)
    )
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name not in screen_classes:
            continue
        assert not any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "exit"
            for child in ast.walk(node)
        ), node.name


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
