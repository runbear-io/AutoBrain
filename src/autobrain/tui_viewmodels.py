"""Presentation-only view models derived from immutable UI state."""

from __future__ import annotations

from dataclasses import dataclass

from autobrain.auth.models import Provider
from autobrain.models import CandidateId, ConnectionState
from autobrain.subscription_domain import ProviderId, SubscriptionStatus
from autobrain.tui_state import UiScreen, UiState


@dataclass(frozen=True)
class CandidateView:
    candidate: CandidateId
    selected: bool
    status: str


@dataclass(frozen=True)
class UiViewModel:
    screen: UiScreen
    title: str
    provider: ProviderId
    provider_status: SubscriptionStatus
    sources: tuple[tuple[Provider, ConnectionState, bool, str], ...]
    candidates: dict[CandidateId, CandidateView]
    plan_title: str
    plan_description: str
    setup_error: str
    stage: str
    stage_detail: str
    elapsed: str
    terminal_reason: str
    report_path: str
    can_run: bool
    cancelling: bool


def build_view_model(state: UiState) -> UiViewModel:
    source_states = dict(state.source_states)
    details = dict(state.source_details)
    statuses = dict(state.candidate_statuses)
    minutes, seconds = divmod(max(0, state.elapsed_seconds), 60)
    result = state.result
    return UiViewModel(
        screen=state.screen,
        title="AutoBrain",
        provider=state.provider,
        provider_status=state.subscription_status,
        sources=tuple(
            (
                provider,
                source_states.get(provider, ConnectionState.DISCONNECTED),
                provider in state.selected_sources,
                details.get(provider, ""),
            )
            for provider in Provider
        ),
        candidates={
            candidate: CandidateView(
                candidate=candidate,
                selected=candidate in state.selected_candidates,
                status=statuses.get(candidate, "PENDING"),
            )
            for candidate in CandidateId
        },
        plan_title=state.plan.title if state.plan else "",
        plan_description=state.plan.description if state.plan else "",
        setup_error=state.setup_error,
        stage=state.latest_stage.name if state.latest_stage else "waiting",
        stage_detail=state.latest_stage.detail if state.latest_stage else "",
        elapsed=f"{minutes:02d}:{seconds:02d}",
        terminal_reason=(state.terminal_reason or (result.status.value if result else "")),
        report_path=str(result.report_path) if result and result.report_path else "",
        can_run=state.plan is not None and not state.running,
        cancelling=state.cancelling,
    )
