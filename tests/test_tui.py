from __future__ import annotations

import curses
import queue
from pathlib import Path

import pytest
from typer.testing import CliRunner

import autobrain.cli as cli
from autobrain.auth.models import Provider
from autobrain.cli import app
from autobrain.experiment import ExperimentPlan, build_automatic_plan
from autobrain.models import CandidateId, ConnectionState, Status
from autobrain.orchestration import RunConfig, RunResult, StageEvent
from autobrain.subscription import ProviderId, SubscriptionStatus
from autobrain.terminal_text import terminal_width
from autobrain.tui import (
    TUIState,
    WizardSection,
    accepts_key_at_size,
    accepts_key_for_state,
    select_subscription_provider,
    subscription_provider_key,
)
from autobrain.tui_render import (
    MIN_TERMINAL_HEIGHT,
    MIN_TERMINAL_WIDTH,
    render_dashboard,
)
from autobrain.tui_runtime import ConnectionSnapshot, execute_plan


def test_tui_navigation_only_exposes_requested_setup_sections() -> None:
    state = TUIState()

    assert state.section is WizardSection.CONNECTIONS
    assert state.advance().section is WizardSection.SLACK
    assert state.advance().advance().section is WizardSection.NOTION
    assert state.advance().advance().advance().section is WizardSection.CANDIDATES
    assert state.advance().advance().advance().advance().section is WizardSection.REVIEW


@pytest.mark.parametrize(
    ("key", "provider"),
    [
        (ord("1"), ProviderId.CODEX),
        (ord("2"), ProviderId.CLAUDE),
        (ord("3"), ProviderId.KIMI),
        (ord("4"), ProviderId.GROK),
    ],
)
def test_legacy_tui_provider_key_reprobes_exact_selection(
    key: int,
    provider: ProviderId,
) -> None:
    calls: list[tuple[ProviderId, bool]] = []

    def snapshot(**kwargs: object) -> ConnectionSnapshot:
        selected = kwargs["subscription_provider"]
        assert isinstance(selected, ProviderId)
        refresh = kwargs["refresh_subscription"] is True
        calls.append((selected, refresh))
        return ConnectionSnapshot(
            subscription=SubscriptionStatus.UNSUPPORTED,
            sources={},
            subscription_provider=selected,
        )

    state, connections = select_subscription_provider(TUIState(), key, snapshot=snapshot)

    assert state.subscription_provider is provider
    assert connections is not None
    assert connections.subscription_provider is provider
    assert calls == [(provider, True)]


def test_tui_provider_selection_survives_navigation_and_refresh_state() -> None:
    state = TUIState().with_subscription_provider(ProviderId.CLAUDE)

    assert state.advance().back().subscription_provider is ProviderId.CLAUDE
    assert state.start_setup().subscription_provider is ProviderId.CLAUDE
    assert state.with_section(WizardSection.HOME).subscription_provider is ProviderId.CLAUDE
    assert subscription_provider_key(ord("1")) is ProviderId.CODEX
    assert subscription_provider_key(ord("2")) is ProviderId.CLAUDE
    assert subscription_provider_key(ord("3")) is ProviderId.KIMI
    assert subscription_provider_key(ord("4")) is ProviderId.GROK
    assert subscription_provider_key(ord("x")) is None


def test_execute_plan_preserves_selected_provider_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[RunConfig] = []

    class Orchestrator:
        def run(self) -> RunResult:
            raise RuntimeError("stop after config capture")

    def local(
        config: RunConfig,
        *,
        stage_event_sink: object | None = None,
        cancellation: object | None = None,
    ) -> Orchestrator:
        del stage_event_sink, cancellation
        captured.append(config)
        return Orchestrator()

    monkeypatch.setattr("autobrain.tui_runtime.RunOrchestrator.local", local)

    class SlackStatus:
        ready = False

    def slack_status(_self: object) -> SlackStatus:
        return SlackStatus()

    monkeypatch.setattr("autobrain.tui_runtime.SlackSourceStore.status", slack_status)
    result_queue: queue.Queue[RunResult | BaseException] = queue.Queue()
    plan = ExperimentPlan(
        title="Claude plan",
        description="explicit provider",
        provider_mode="claude-subscription",
        sources=(Provider.NOTION,),
        candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
        budget_usd=25.0,
        max_questions=20,
    )

    execute_plan(plan, result_queue, ProviderId.CLAUDE)

    assert captured[0].provider_mode == "claude-subscription"
    assert isinstance(result_queue.get_nowait(), RuntimeError)


def test_tui_toggles_sources_and_candidates_without_manual_budget_or_questions() -> None:
    state = TUIState()

    state = state.toggle_source(Provider.NOTION)
    state = state.toggle_candidate(CandidateId.GBRAIN)

    assert state.selected_sources == (Provider.SLACK,)
    assert state.selected_candidates == (CandidateId.LLM_WIKI, CandidateId.MEM0)
    assert not hasattr(state, "budget_usd")
    assert not hasattr(state, "max_questions")


def test_default_cli_launches_tui_and_headless_commands_remain_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[tuple[bool, ProviderId]] = []

    def capture(*, force_setup: bool = False, provider: ProviderId = ProviderId.CODEX) -> None:
        launched.append((force_setup, provider))

    monkeypatch.setattr(cli, "run_tui", capture)

    default = CliRunner().invoke(app, [])
    claude = CliRunner().invoke(app, ["--provider", "claude"])
    setup = CliRunner().invoke(app, ["setup", "--provider", "grok"])

    assert default.exit_code == 0
    assert claude.exit_code == 0
    assert setup.exit_code == 0
    assert launched == [
        (False, ProviderId.CODEX),
        (False, ProviderId.CLAUDE),
        (True, ProviderId.GROK),
    ]
    assert CliRunner().invoke(app, ["run", "--help"]).exit_code == 0


def test_provider_menu_renders_explicit_selection_and_unsupported_choices() -> None:
    lines = render_dashboard(
        section=WizardSection.CONNECTIONS.value,
        selected_sources=(Provider.SLACK,),
        selected_candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
        source_states={},
        subscription_status=SubscriptionStatus.UNSUPPORTED,
        subscription_provider=ProviderId.KIMI,
        plan=None,
        setup_error="UNSUPPORTED",
        result=None,
        elapsed_seconds=0,
        width=100,
        height=30,
    )

    assert any("[3] [x] Kimi" in line for line in lines)
    assert any("[2] [ ] Claude" in line for line in lines)
    assert any("never falls back" in line for line in lines)
    assert "1/2/3/4 select" in lines[-1]


def test_setup_dashboard_fits_standard_terminal() -> None:
    plan = build_automatic_plan(
        sources=(Provider.SLACK, Provider.NOTION),
        candidates=tuple(CandidateId),
        subscription_status=SubscriptionStatus.READY,
    )

    lines = render_dashboard(
        section=WizardSection.REVIEW.value,
        selected_sources=plan.sources,
        selected_candidates=plan.candidates,
        source_states={
            Provider.SLACK: ConnectionState.CONNECTED,
            Provider.NOTION: ConnectionState.CONNECTED,
        },
        subscription_status=SubscriptionStatus.READY,
        plan=plan,
        setup_error="",
        result=None,
        elapsed_seconds=0,
        width=80,
    )

    assert len(lines) <= 23
    assert max(map(len, lines)) <= 78


def test_narrow_terminal_uses_explicit_resize_state() -> None:
    lines = render_dashboard(
        section=WizardSection.CONNECTIONS.value,
        selected_sources=(Provider.SLACK,),
        selected_candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
        source_states={},
        subscription_status=SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE,
        plan=None,
        setup_error="",
        result=None,
        elapsed_seconds=0,
        width=50,
        height=20,
    )

    assert "TERMINAL_TOO_SMALL" in lines
    assert max(map(len, lines)) <= 48


def test_advertised_minimum_height_preserves_keyboard_footer() -> None:
    plan = build_automatic_plan(
        sources=(Provider.SLACK, Provider.NOTION),
        candidates=tuple(CandidateId),
        subscription_status=SubscriptionStatus.READY,
    )

    lines = render_dashboard(
        section=WizardSection.REVIEW.value,
        selected_sources=plan.sources,
        selected_candidates=plan.candidates,
        source_states={
            Provider.SLACK: ConnectionState.CONNECTED,
            Provider.NOTION: ConnectionState.CONNECTED,
        },
        subscription_status=SubscriptionStatus.READY,
        plan=plan,
        setup_error="",
        result=None,
        elapsed_seconds=0,
        width=MIN_TERMINAL_WIDTH,
        height=MIN_TERMINAL_HEIGHT,
    )

    assert len(lines) <= MIN_TERMINAL_HEIGHT - 1
    assert "Q" in lines[-1]


def test_results_hide_open_action_when_report_is_unavailable(tmp_path: Path) -> None:
    result = RunResult(
        run_id="demo",
        run_dir=tmp_path,
        status=Status.OK,
        report_path=None,
        candidate_results=(),
        verdict="NO_DECISION",
    )

    lines = render_dashboard(
        section=WizardSection.RESULTS.value,
        selected_sources=(Provider.SLACK,),
        selected_candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
        source_states={Provider.SLACK: ConnectionState.CONNECTED},
        subscription_status=SubscriptionStatus.READY,
        plan=None,
        setup_error="",
        result=result,
        elapsed_seconds=0,
        width=80,
        height=24,
    )

    assert all(not line.startswith("O ") for line in lines)


def test_narrow_terminal_only_accepts_quit() -> None:
    assert not accepts_key_at_size(ord("1"), width=50, height=20)
    assert not accepts_key_at_size(ord("c"), width=50, height=20)
    assert not accepts_key_at_size(10, width=50, height=20)
    assert accepts_key_at_size(ord("q"), width=50, height=20)
    assert accepts_key_at_size(ord("1"), width=80, height=24)


def test_extremely_narrow_renderer_respects_physical_width() -> None:
    lines = render_dashboard(
        section=WizardSection.CONNECTIONS.value,
        selected_sources=(Provider.SLACK,),
        selected_candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
        source_states={},
        subscription_status=SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE,
        plan=None,
        setup_error="",
        result=None,
        elapsed_seconds=0,
        width=10,
        height=10,
    )

    assert max(map(len, lines)) <= 8
    assert "TOO" in lines
    assert "SMALL" in lines
    assert ">=60x22" in lines
    assert "Q quit" in lines


def test_setup_footers_only_advertise_available_actions() -> None:
    source_lines = render_dashboard(
        section=WizardSection.SLACK.value,
        selected_sources=(Provider.SLACK,),
        selected_candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
        source_states={},
        subscription_status=SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE,
        plan=None,
        setup_error="PROVIDER_UNAVAILABLE",
        result=None,
        elapsed_seconds=0,
        width=80,
        height=24,
    )
    review_lines = render_dashboard(
        section=WizardSection.REVIEW.value,
        selected_sources=(Provider.SLACK,),
        selected_candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
        source_states={},
        subscription_status=SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE,
        plan=None,
        setup_error="PROVIDER_UNAVAILABLE",
        result=None,
        elapsed_seconds=0,
        width=80,
        height=24,
    )

    assert "S skip" in source_lines[-1]
    assert "Enter connect" in source_lines[-1]
    assert "3 toggle" not in source_lines[-1]
    assert "Enter run experiment" not in review_lines[-1]
    assert any("Add Slack knowledge" in line for line in source_lines)
    assert not any("Connections" in line for line in source_lines)


def test_running_dashboard_shows_latest_persisted_stage() -> None:
    lines = render_dashboard(
        section=WizardSection.RUNNING.value,
        selected_sources=(Provider.SLACK,),
        selected_candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
        source_states={Provider.SLACK: ConnectionState.CONNECTED},
        subscription_status=SubscriptionStatus.READY,
        plan=None,
        setup_error="",
        result=None,
        elapsed_seconds=3,
        width=80,
        height=24,
        latest_stage=StageEvent(
            sequence=4,
            run_id="demo",
            name="coverage",
            status=Status.OK,
            detail="24 documents",
            started_at="2026-08-19T00:00:00+00:00",
        ),
    )

    assert "Stage       coverage" in lines
    assert "Status      OK - 24 documents" in lines


def test_running_state_only_accepts_cooperative_cancel_keys() -> None:
    state = TUIState(section=WizardSection.RUNNING)

    for key in (ord("c"), ord("C"), ord("q"), ord("Q")):
        assert accepts_key_for_state(state, key, width=80, height=24)
    for key in (
        ord("b"),
        curses.KEY_BACKSPACE,
        curses.KEY_UP,
        curses.KEY_BTAB,
        10,
        ord("1"),
    ):
        assert not accepts_key_for_state(state, key, width=80, height=24)


def test_wide_character_content_is_terminal_cell_bounded(tmp_path: Path) -> None:
    assert terminal_width("✈️") == 2
    assert terminal_width("1️⃣") == 2

    result = RunResult(
        run_id="demo",
        run_dir=tmp_path,
        status=Status.OK,
        report_path=tmp_path / ("보고서🙂e\u0301✈️1️⃣" * 20),
        candidate_results=(),
        verdict="NO_DECISION",
    )

    lines = render_dashboard(
        section=WizardSection.RESULTS.value,
        selected_sources=(Provider.SLACK,),
        selected_candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
        source_states={Provider.SLACK: ConnectionState.CONNECTED},
        subscription_status=SubscriptionStatus.READY,
        plan=None,
        setup_error="",
        result=result,
        elapsed_seconds=0,
        width=40,
        height=24,
    )

    assert max(terminal_width(line) for line in lines) <= 38


def test_dashboard_labels_configured_slack_export() -> None:
    lines = render_dashboard(
        section=WizardSection.SLACK.value,
        selected_sources=(Provider.SLACK, Provider.NOTION),
        selected_candidates=tuple(CandidateId),
        source_states={
            Provider.SLACK: ConnectionState.CONNECTED,
            Provider.NOTION: ConnectionState.DISCONNECTED,
        },
        subscription_status=SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE,
        plan=None,
        setup_error="",
        result=None,
        elapsed_seconds=0,
        width=100,
        height=30,
        source_details={Provider.SLACK: "export ready"},
    )

    assert any("export ready" in line for line in lines)
    assert any("Add Slack knowledge" in line for line in lines)
    assert any("Provider" in line for line in lines)
