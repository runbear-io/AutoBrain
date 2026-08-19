from __future__ import annotations

import curses
from pathlib import Path

import pytest
from typer.testing import CliRunner

import autobrain.cli as cli
from autobrain.auth.models import Provider
from autobrain.cli import app
from autobrain.experiment import build_automatic_plan
from autobrain.models import CandidateId, ConnectionState, Status
from autobrain.orchestration import RunResult
from autobrain.subscription import SubscriptionStatus
from autobrain.terminal_text import terminal_width
from autobrain.tui import (
    TUIState,
    WizardSection,
    accepts_key_at_size,
    accepts_key_for_state,
)
from autobrain.tui_render import (
    MIN_TERMINAL_HEIGHT,
    MIN_TERMINAL_WIDTH,
    render_dashboard,
)


def test_tui_navigation_only_exposes_requested_setup_sections() -> None:
    state = TUIState()

    assert state.section is WizardSection.CONNECTIONS
    assert state.advance().section is WizardSection.KNOWLEDGE_SOURCES
    assert state.advance().advance().section is WizardSection.CANDIDATES
    assert state.advance().advance().advance().section is WizardSection.REVIEW


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
    launched: list[bool] = []
    monkeypatch.setattr(cli, "run_tui", lambda: launched.append(True))

    result = CliRunner().invoke(app, [])

    assert result.exit_code == 0
    assert launched == [True]
    assert CliRunner().invoke(app, ["run", "--help"]).exit_code == 0


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
    assert ">=60x23" in lines
    assert "Q quit" in lines


def test_setup_footers_only_advertise_available_actions() -> None:
    source_lines = render_dashboard(
        section=WizardSection.KNOWLEDGE_SOURCES.value,
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

    assert "S/N connect" in source_lines[-1]
    assert "1/2 toggle" in source_lines[-1]
    assert "3 toggle" not in source_lines[-1]
    assert "Enter run experiment" not in review_lines[-1]
    assert any(line.startswith("> 2  Knowledge") for line in source_lines)
    assert not any("Connections" in line for line in source_lines)
    assert any("[S]" in line and "[x]" in line and "Slack" in line for line in source_lines)


def test_running_state_rejects_navigation_and_duplicate_run_keys() -> None:
    state = TUIState(section=WizardSection.RUNNING)

    for key in (
        ord("b"),
        curses.KEY_BACKSPACE,
        curses.KEY_UP,
        curses.KEY_BTAB,
        10,
        ord("1"),
        ord("c"),
        ord("q"),
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
        section=WizardSection.CONNECTIONS.value,
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

    assert any("Slack" in line and "export ready" in line for line in lines)
    assert any("[S]" in line and "[x]" in line and "export ready" in line for line in lines)
    assert not any("1  Connections" in line for line in lines)
    assert any("1  ChatGPT" in line for line in lines)
    assert any("2  Knowledge" in line for line in lines)
