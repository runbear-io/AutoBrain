from __future__ import annotations

import asyncio
from pathlib import Path

from autobrain.models import Status
from autobrain.orchestration import RunResult
from autobrain.subscription import ProviderId
from autobrain.tui_actions import RunCompleted
from autobrain.tui_state import UiScreen
from autobrain.tui_textual import AutoBrainApp


def test_textual_pilot_navigates_setup_screens_keyboard_only() -> None:
    async def exercise() -> None:
        app = AutoBrainApp(force_setup=True, provider=ProviderId.CODEX)
        async with app.run_test(size=(80, 24)) as pilot:
            assert app.state.screen is UiScreen.CONNECTIONS
            await pilot.click("#continue")
            assert app.state.screen is UiScreen.SLACK
            await pilot.click("#continue")
            assert app.state.screen is UiScreen.NOTION
            await pilot.click("#continue")
            assert app.state.screen is UiScreen.CANDIDATES
            await pilot.click("#review")
            assert app.state.screen is UiScreen.REVIEW

    asyncio.run(exercise())


def test_textual_pilot_results_opens_only_completed_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    opened: list[Path] = []
    monkeypatch.setattr(
        "autobrain.tui_textual.open_exact_report",
        lambda effect: opened.append(effect.path) or True,
    )

    async def exercise() -> None:
        app = AutoBrainApp(force_setup=False, provider=ProviderId.CLAUDE)
        async with app.run_test(size=(120, 40)) as pilot:
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
            await pilot.click("#open-report")
            assert opened == [report]

    asyncio.run(exercise())
