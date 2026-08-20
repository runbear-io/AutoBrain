from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from textual.pilot import Pilot

from autobrain.auth.models import Provider
from autobrain.experiment import ExperimentPlan
from autobrain.models import CandidateId, Status
from autobrain.orchestration import RunResult
from autobrain.subscription import ProviderId
from autobrain.tui_actions import RunCompleted
from autobrain.tui_effects import OpenExactReport
from autobrain.tui_state import UiScreen
from autobrain.tui_textual import AutoBrainApp


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
            await _activate_button(pilot, 3)
            assert app.state.screen is UiScreen.RUNNING

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
