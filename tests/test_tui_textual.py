from __future__ import annotations

import asyncio
import html
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from textual.pilot import Pilot
from textual.worker import Worker, WorkerState

from autobrain.auth.models import Provider
from autobrain.experiment import ExperimentPlan
from autobrain.models import CandidateId, Status
from autobrain.orchestration import RunResult
from autobrain.subscription import ProviderId
from autobrain.tui_actions import Navigate, RequestLogin, RunCompleted, StartRun, ToggleCandidate
from autobrain.tui_effects import LoadConnections, OpenExactReport
from autobrain.tui_state import UiScreen, reduce_ui
from autobrain.tui_textual import AutoBrainApp

_RAW_SECRET_ERROR = (
    "raw provider explosion Authorization: Bearer auth-secret-12345678 "
    "api_key=sk-test-secret-12345678 password=hunter-secret-12345678"
)


def _cell_text(app: AutoBrainApp) -> str:
    svg = app.export_screenshot()
    return html.unescape(re.sub(r"<[^>]+>", "", svg)).replace("\xa0", " ")


_FORBIDDEN_ERROR_FRAGMENTS = (
    "raw provider explosion",
    "auth-secret-12345678",
    "sk-test-secret-12345678",
    "hunter-secret-12345678",
    "authorization: bearer",
    "api_key=sk-test",
    "password=hunter",
)


def _assert_error_is_safe(
    app: AutoBrainApp,
    caplog: pytest.LogCaptureFixture,
    expected: str,
) -> None:
    rendered = f"{app.state.setup_error}\n{app.state.terminal_reason}\n{app.export_screenshot()}"
    exposed = f"{rendered}\n{caplog.text}".casefold()
    assert expected in rendered
    assert all(fragment not in exposed for fragment in _FORBIDDEN_ERROR_FRAGMENTS)


def _failed_worker_event(group: str, error: BaseException) -> Worker.StateChanged:
    worker = cast(Worker[object], SimpleNamespace(group=group, error=error, result=None))
    return Worker.StateChanged(worker, WorkerState.ERROR)


async def _activate_button(pilot: Pilot[None], tabs: int) -> None:
    for _ in range(tabs):
        await pilot.press("tab")
    await pilot.press("enter")
    await pilot.pause()


def test_textual_pilot_navigates_every_setup_screen_by_keyboard() -> None:
    async def exercise() -> None:
        app = AutoBrainApp(force_setup=True, provider=ProviderId.CODEX)
        async with app.run_test(size=(80, 24)) as pilot:
            assert app.state.screen is UiScreen.CONNECTIONS
            await _activate_button(pilot, 11)
            assert app.state.screen is UiScreen.SLACK
            await _activate_button(pilot, 5)
            assert app.state.screen is UiScreen.NOTION
            await _activate_button(pilot, 5)
            assert app.state.screen is UiScreen.CANDIDATES
            await _activate_button(pilot, 6)
            assert app.state.screen is UiScreen.REVIEW
            app.state = replace(
                app.state,
                plan=ExperimentPlan(
                    title="Pilot plan",
                    description="Keyboard run path",
                    provider_mode="codex-subscription",
                    embedding_backend="openai",
                    sources=(Provider.SLACK,),
                    candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
                    budget_usd=25.0,
                    max_questions=20,
                ),
            )
            app._execute = lambda effect: None  # type: ignore[method-assign]
            app.screen.refresh_view()  # type: ignore[attr-defined]
            await _activate_button(pilot, 3)
            assert app.state.screen is UiScreen.RUNNING

    asyncio.run(exercise())


def test_connection_probe_exception_is_classified_without_rendering_raw_message(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_probe(**kwargs: object) -> None:
        del kwargs
        raise RuntimeError(_RAW_SECRET_ERROR)

    monkeypatch.setattr("autobrain.tui_textual.connection_snapshot", fail_probe)

    async def exercise() -> None:
        app = AutoBrainApp(force_setup=True, provider=ProviderId.CODEX)
        app._execute = lambda effect: None  # type: ignore[method-assign]
        async with app.run_test(size=(80, 24)) as pilot:
            worker = app.load_connections(LoadConnections(ProviderId.CODEX))
            await worker.wait()
            await pilot.pause()
            _assert_error_is_safe(app, caplog, "CONNECTION_PROBE_FAILED: RuntimeError")

    asyncio.run(exercise())


def test_login_exception_is_classified_without_rendering_raw_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def exercise() -> None:
        app = AutoBrainApp(force_setup=True, provider=ProviderId.CODEX)
        app._execute = lambda effect: None  # type: ignore[method-assign]
        async with app.run_test(size=(80, 24)) as pilot:
            app.state = reduce_ui(app.state, RequestLogin(ProviderId.CODEX)).state
            app.on_worker_state_changed(
                _failed_worker_event("login", ValueError(_RAW_SECRET_ERROR))
            )
            await pilot.pause()
            _assert_error_is_safe(app, caplog, "LOGIN_FAILED: ValueError")

    asyncio.run(exercise())


def test_run_exception_is_classified_without_rendering_raw_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def exercise() -> None:
        app = AutoBrainApp(force_setup=False, provider=ProviderId.CODEX)
        app._execute = lambda effect: None  # type: ignore[method-assign]
        async with app.run_test(size=(80, 24)) as pilot:
            app.state = replace(
                app.state,
                plan=ExperimentPlan(
                    title="Adversarial run",
                    description="Synthetic exception path",
                    provider_mode="codex-subscription",
                    embedding_backend="openai",
                    sources=(Provider.SLACK,),
                    candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
                    budget_usd=25.0,
                    max_questions=20,
                ),
            )
            app.state = reduce_ui(app.state, StartRun()).state
            app.on_worker_state_changed(
                _failed_worker_event("run", RuntimeError(_RAW_SECRET_ERROR))
            )
            await pilot.pause()
            _assert_error_is_safe(app, caplog, "RUN_FAILED: RuntimeError")

    asyncio.run(exercise())


def test_specialized_screen_copy_and_selected_labels_survive_refresh() -> None:
    async def exercise() -> None:
        app = AutoBrainApp(force_setup=True, provider=ProviderId.CODEX)
        app._execute = lambda effect: None  # type: ignore[method-assign]
        async with app.run_test(size=(60, 22)) as pilot:
            app.dispatch_ui(Navigate(UiScreen.CANDIDATES.value))
            await pilot.pause()
            rendered = _cell_text(app)
            assert "Choose candidates" in rendered
            assert "Quick Start" in rendered
            assert "[x] gbrain" in rendered
            app.dispatch_ui(ToggleCandidate(CandidateId.GBRAIN))
            await pilot.pause()
            assert "[ ] gbrain" in _cell_text(app)

    asyncio.run(exercise())


def test_review_initial_viewport_is_complete_runnable_and_cjk_safe() -> None:
    async def exercise() -> None:
        app = AutoBrainApp(force_setup=True, provider=ProviderId.CODEX)
        app._execute = lambda effect: None  # type: ignore[method-assign]
        async with app.run_test(size=(60, 22)) as pilot:
            app.dispatch_ui(Navigate(UiScreen.CANDIDATES.value))
            await pilot.pause()
            app.state = replace(
                app.state,
                plan=ExperimentPlan(
                    title="서울 지식베이스 검토",
                    description="한국어 질문으로 후보를 비교합니다.",
                    provider_mode="codex-subscription",
                    embedding_backend="local-hash",
                    sources=(Provider.SLACK,),
                    candidates=(CandidateId.LLM_WIKI, CandidateId.GBRAIN),
                    budget_usd=25.0,
                    max_questions=20,
                ),
            )
            app.dispatch_ui(Navigate(UiScreen.REVIEW.value))
            await pilot.pause()
            rendered = _cell_text(app)
            for copy in (
                "Review",
                "Quick Start",
                "keyword-only",
                "Not measured",
                "Runnable",
                "Run experiment",
                "서울 지식베이스 검토",
                "한국어 질문으로 후보를 비교합니다.",
            ):
                assert copy in rendered

    asyncio.run(exercise())


def test_review_blocks_run_visibly_when_plan_is_absent() -> None:
    async def exercise() -> None:
        app = AutoBrainApp(force_setup=True, provider=ProviderId.CODEX)
        app._execute = lambda effect: None  # type: ignore[method-assign]
        async with app.run_test(size=(60, 22)) as pilot:
            app.dispatch_ui(Navigate(UiScreen.REVIEW.value))
            await pilot.pause()
            run = app.screen.query_one("#run")
            assert run.disabled is True
            assert "Blocked" in _cell_text(app)
            app.dispatch_ui(StartRun())
            assert app.state.setup_error.startswith("PLAN_UNAVAILABLE:")

    asyncio.run(exercise())


def test_textual_pilot_home_and_results_paths_are_keyboard_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Path] = []

    def record_open(effect: OpenExactReport) -> bool:
        opened.append(effect.path)
        return True

    monkeypatch.setattr("autobrain.tui_textual.open_exact_report", record_open)

    async def exercise() -> None:
        app = AutoBrainApp(force_setup=False, provider=ProviderId.CLAUDE)
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.state.screen is UiScreen.HOME
            await _activate_button(pilot, 3)
            assert app.state.screen is UiScreen.CONNECTIONS

            report = tmp_path / "exact-report.html"
            app.dispatch_ui(
                RunCompleted(
                    RunResult(
                        run_id="pilot",
                        run_dir=tmp_path,
                        status=Status.OK,
                        report_path=report,
                        candidate_results=(),
                        verdict="NO_RECOMMENDATION",
                    )
                )
            )
            await pilot.pause()
            assert app.state.screen is UiScreen.RESULTS
            await _activate_button(pilot, 3)
            assert opened == [report]

    asyncio.run(exercise())
