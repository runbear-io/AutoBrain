"""Immutable, framework-neutral state and reducer for the AutoBrain UI."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from autobrain.auth.models import Provider
from autobrain.experiment import ExperimentPlan
from autobrain.models import CandidateId, ConnectionState
from autobrain.orchestration import RunResult, StageEvent
from autobrain.subscription_domain import ProviderId, SubscriptionStatus
from autobrain.tui_actions import (
    BeginSetup,
    CancelRun,
    ConnectionsLoaded,
    GoBack,
    Navigate,
    OpenReport,
    RefreshConnections,
    RequestLogin,
    RunCompleted,
    RunFailed,
    SelectProvider,
    SkipSource,
    StageObserved,
    StartRun,
    ToggleCandidate,
    ToggleSource,
    UiAction,
)
from autobrain.tui_effects import (
    CancelActiveRun,
    InteractiveLogin,
    LoadConnections,
    OpenExactReport,
    RunExperiment,
    UiEffect,
)


class UiScreen(StrEnum):
    HOME = "home"
    CONNECTIONS = "connections"
    SLACK = "slack"
    NOTION = "notion"
    CANDIDATES = "candidates"
    REVIEW = "review"
    RUNNING = "running"
    RESULTS = "results"


WizardSection = UiScreen
_SETUP_SCREENS = (
    UiScreen.CONNECTIONS,
    UiScreen.SLACK,
    UiScreen.NOTION,
    UiScreen.CANDIDATES,
    UiScreen.REVIEW,
)


@dataclass(frozen=True)
class UiState:
    screen: UiScreen = UiScreen.CONNECTIONS
    selected_sources: tuple[Provider, ...] = (Provider.SLACK, Provider.NOTION)
    selected_candidates: tuple[CandidateId, ...] = tuple(CandidateId)
    provider: ProviderId = ProviderId.CODEX
    subscription_status: SubscriptionStatus = SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE
    source_states: tuple[tuple[Provider, ConnectionState], ...] = ()
    source_details: tuple[tuple[Provider, str], ...] = ()
    plan: ExperimentPlan | None = None
    setup_error: str = ""
    latest_stage: StageEvent | None = None
    elapsed_seconds: int = 0
    candidate_statuses: tuple[tuple[CandidateId, str], ...] = ()
    result: RunResult | None = None
    terminal_reason: str = ""
    running: bool = False
    cancelling: bool = False
    return_home: bool = False

    @property
    def section(self) -> UiScreen:
        return self.screen

    @property
    def subscription_provider(self) -> ProviderId:
        return self.provider

    def advance(self) -> UiState:
        if self.screen not in _SETUP_SCREENS:
            return self
        index = _SETUP_SCREENS.index(self.screen)
        return replace(self, screen=_SETUP_SCREENS[min(index + 1, len(_SETUP_SCREENS) - 1)])

    def back(self) -> UiState:
        if self.screen is UiScreen.HOME:
            return self
        if self.screen not in _SETUP_SCREENS:
            return replace(self, screen=UiScreen.HOME)
        index = _SETUP_SCREENS.index(self.screen)
        if index == 0 and self.return_home:
            return replace(self, screen=UiScreen.HOME)
        return replace(self, screen=_SETUP_SCREENS[max(0, index - 1)])

    def start_setup(self) -> UiState:
        return replace(self, screen=UiScreen.CONNECTIONS, return_home=True)

    def with_section(self, section: UiScreen) -> UiState:
        return replace(self, screen=section)

    def with_subscription_provider(self, provider: ProviderId) -> UiState:
        return replace(self, provider=provider)

    def skip_source(self, provider: Provider) -> UiState:
        values = tuple(item for item in self.selected_sources if item is not provider)
        return replace(self, selected_sources=values).advance()

    def toggle_source(self, provider: Provider) -> UiState:
        values = set(self.selected_sources)
        values.symmetric_difference_update({provider})
        return replace(self, selected_sources=tuple(item for item in Provider if item in values))

    def toggle_candidate(self, candidate: CandidateId) -> UiState:
        values = set(self.selected_candidates)
        values.symmetric_difference_update({candidate})
        return replace(
            self,
            selected_candidates=tuple(item for item in CandidateId if item in values),
        )


@dataclass(frozen=True)
class TUIState:
    """One-release immutable compatibility facade for legacy policy callers."""

    section: WizardSection = WizardSection.CONNECTIONS
    selected_sources: tuple[Provider, ...] = (Provider.SLACK, Provider.NOTION)
    selected_candidates: tuple[CandidateId, ...] = tuple(CandidateId)
    subscription_provider: ProviderId = ProviderId.CODEX
    return_home: bool = False

    def advance(self) -> TUIState:
        if self.section not in _SETUP_SCREENS:
            return self
        index = _SETUP_SCREENS.index(self.section)
        return replace(
            self,
            section=_SETUP_SCREENS[min(index + 1, len(_SETUP_SCREENS) - 1)],
        )

    def back(self) -> TUIState:
        if self.section is WizardSection.HOME:
            return self
        if self.section not in _SETUP_SCREENS:
            return replace(self, section=WizardSection.HOME)
        index = _SETUP_SCREENS.index(self.section)
        if index == 0 and self.return_home:
            return replace(self, section=WizardSection.HOME)
        return replace(self, section=_SETUP_SCREENS[max(0, index - 1)])

    def start_setup(self) -> TUIState:
        return replace(self, section=WizardSection.CONNECTIONS, return_home=True)

    def with_section(self, section: WizardSection) -> TUIState:
        return replace(self, section=section)

    def with_subscription_provider(self, provider: ProviderId) -> TUIState:
        return replace(self, subscription_provider=provider)

    def skip_source(self, provider: Provider) -> TUIState:
        values = tuple(item for item in self.selected_sources if item is not provider)
        return replace(self, selected_sources=values).advance()

    def toggle_source(self, provider: Provider) -> TUIState:
        values = set(self.selected_sources)
        values.symmetric_difference_update({provider})
        return replace(self, selected_sources=tuple(item for item in Provider if item in values))

    def toggle_candidate(self, candidate: CandidateId) -> TUIState:
        values = set(self.selected_candidates)
        values.symmetric_difference_update({candidate})
        return replace(
            self,
            selected_candidates=tuple(item for item in CandidateId if item in values),
        )


@dataclass(frozen=True)
class Reduction:
    state: UiState
    effects: tuple[UiEffect, ...] = ()


def _resolved(state: UiState) -> UiState:
    from autobrain.tui_runtime import ConnectionSnapshot, resolve_plan

    snapshot = ConnectionSnapshot(
        subscription=state.subscription_status,
        sources=dict(state.source_states),
        subscription_provider=state.provider,
        source_details=dict(state.source_details),
    )
    plan, error = resolve_plan(
        selected_sources=state.selected_sources,
        selected_candidates=state.selected_candidates,
        connections=snapshot,
        subscription_provider=state.provider,
    )
    return replace(state, plan=plan, setup_error=error)


def _candidate_stage_statuses(
    current: tuple[tuple[CandidateId, str], ...], event: StageEvent
) -> tuple[tuple[CandidateId, str], ...]:
    values = dict(current)
    normalized = event.name.casefold().replace("_", "-")
    for candidate in CandidateId:
        if candidate.value in normalized:
            values[candidate] = event.status.value
    return tuple((candidate, values.get(candidate, "PENDING")) for candidate in CandidateId)


def reduce_ui(state: UiState, action: UiAction) -> Reduction:
    """Apply one semantic action without performing I/O."""
    if isinstance(action, BeginSetup):
        return Reduction(replace(state, screen=UiScreen.CONNECTIONS, return_home=True))
    if isinstance(action, Navigate):
        return Reduction(replace(state, screen=UiScreen(action.screen)))
    if isinstance(action, GoBack):
        return Reduction(state.back())
    if isinstance(action, SelectProvider):
        selected = replace(
            state,
            provider=action.provider,
            plan=None,
            setup_error="Refreshing selected provider...",
        )
        return Reduction(selected, (LoadConnections(action.provider, refresh=True),))
    if isinstance(action, ToggleSource):
        return Reduction(_resolved(state.toggle_source(action.provider)))
    if isinstance(action, SkipSource):
        return Reduction(_resolved(state.skip_source(action.provider)))
    if isinstance(action, ToggleCandidate):
        return Reduction(_resolved(state.toggle_candidate(action.candidate)))
    if isinstance(action, RefreshConnections):
        return Reduction(state, (LoadConnections(state.provider, refresh=True),))
    if isinstance(action, RequestLogin):
        return Reduction(state, (InteractiveLogin(action.provider),))
    if isinstance(action, ConnectionsLoaded):
        snapshot = action.snapshot
        loaded = replace(
            state,
            provider=snapshot.subscription_provider,
            subscription_status=snapshot.subscription,
            source_states=tuple(snapshot.sources.items()),
            source_details=tuple((snapshot.source_details or {}).items()),
        )
        return Reduction(_resolved(loaded))
    if isinstance(action, StartRun):
        if state.plan is None:
            return Reduction(replace(state, setup_error=state.setup_error or "PLAN_UNAVAILABLE"))
        started = replace(
            state,
            screen=UiScreen.RUNNING,
            running=True,
            cancelling=False,
            result=None,
            latest_stage=None,
            elapsed_seconds=0,
            candidate_statuses=tuple((item, "PENDING") for item in state.selected_candidates),
            terminal_reason="",
        )
        return Reduction(started, (RunExperiment(state.plan, state.provider),))
    if isinstance(action, StageObserved):
        return Reduction(
            replace(
                state,
                latest_stage=action.event,
                elapsed_seconds=max(0, action.elapsed_seconds),
                candidate_statuses=_candidate_stage_statuses(
                    state.candidate_statuses, action.event
                ),
            )
        )
    if isinstance(action, CancelRun):
        if not state.running or state.cancelling:
            return Reduction(state)
        return Reduction(replace(state, cancelling=True), (CancelActiveRun(),))
    if isinstance(action, RunCompleted):
        result = action.result
        statuses = dict(state.candidate_statuses)
        for outcome in result.candidate_results:
            try:
                candidate = CandidateId(outcome.candidate)
            except ValueError:
                continue
            statuses[candidate] = outcome.status.value
        return Reduction(
            replace(
                state,
                screen=UiScreen.RESULTS,
                running=False,
                cancelling=False,
                result=result,
                terminal_reason=result.status.value,
                candidate_statuses=tuple(statuses.items()),
            )
        )
    if isinstance(action, RunFailed):
        return Reduction(
            replace(
                state,
                screen=UiScreen.RESULTS,
                running=False,
                cancelling=False,
                terminal_reason=action.reason,
                setup_error=action.reason,
            )
        )
    if isinstance(action, OpenReport):
        if state.result is None or state.result.report_path is None:
            return Reduction(state)
        return Reduction(state, (OpenExactReport(state.result.report_path),))
    return Reduction(
        replace(
            state,
            screen=UiScreen.HOME,
            result=None,
            latest_stage=None,
            terminal_reason="",
            elapsed_seconds=0,
        )
    )
