"""Pure setup state machine for the AutoBrain interview cockpit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from autobrain.auth.models import Provider
from autobrain.models import CandidateId
from autobrain.subscription import ProviderId


class WizardSection(StrEnum):
    HOME = "home"
    CONNECTIONS = "connections"
    SLACK = "slack"
    NOTION = "notion"
    CANDIDATES = "candidates"
    REVIEW = "review"
    RUNNING = "running"
    RESULTS = "results"


_SETUP_SECTIONS = (
    WizardSection.CONNECTIONS,
    WizardSection.SLACK,
    WizardSection.NOTION,
    WizardSection.CANDIDATES,
    WizardSection.REVIEW,
)


@dataclass(frozen=True)
class TUIState:
    section: WizardSection = WizardSection.CONNECTIONS
    selected_sources: tuple[Provider, ...] = (Provider.SLACK, Provider.NOTION)
    selected_candidates: tuple[CandidateId, ...] = tuple(CandidateId)
    subscription_provider: ProviderId = ProviderId.CODEX
    return_home: bool = False

    def _clone(
        self,
        *,
        section: WizardSection | None = None,
        selected_sources: tuple[Provider, ...] | None = None,
        selected_candidates: tuple[CandidateId, ...] | None = None,
        subscription_provider: ProviderId | None = None,
        return_home: bool | None = None,
    ) -> TUIState:
        return TUIState(
            section=self.section if section is None else section,
            selected_sources=(
                self.selected_sources if selected_sources is None else selected_sources
            ),
            selected_candidates=(
                self.selected_candidates if selected_candidates is None else selected_candidates
            ),
            subscription_provider=(
                self.subscription_provider
                if subscription_provider is None
                else subscription_provider
            ),
            return_home=self.return_home if return_home is None else return_home,
        )

    def advance(self) -> TUIState:
        if self.section not in _SETUP_SECTIONS:
            return self
        index = _SETUP_SECTIONS.index(self.section)
        return self._clone(section=_SETUP_SECTIONS[min(index + 1, len(_SETUP_SECTIONS) - 1)])

    def start_setup(self) -> TUIState:
        return self._clone(section=WizardSection.CONNECTIONS, return_home=True)

    def back(self) -> TUIState:
        if self.section is WizardSection.HOME:
            return self
        if self.section not in _SETUP_SECTIONS:
            return self._clone(section=WizardSection.HOME)
        index = _SETUP_SECTIONS.index(self.section)
        if index == 0 and self.return_home:
            return self._clone(section=WizardSection.HOME)
        return self._clone(section=_SETUP_SECTIONS[max(0, index - 1)])

    def with_section(self, section: WizardSection) -> TUIState:
        return self._clone(section=section)

    def with_subscription_provider(self, provider: ProviderId) -> TUIState:
        return self._clone(subscription_provider=provider)

    def skip_source(self, provider: Provider) -> TUIState:
        values = tuple(item for item in self.selected_sources if item is not provider)
        return self._clone(selected_sources=values).advance()

    def toggle_source(self, provider: Provider) -> TUIState:
        values = set(self.selected_sources)
        if provider in values:
            values.remove(provider)
        else:
            values.add(provider)
        return self._clone(selected_sources=tuple(item for item in Provider if item in values))

    def toggle_candidate(self, candidate: CandidateId) -> TUIState:
        values = set(self.selected_candidates)
        if candidate in values:
            values.remove(candidate)
        else:
            values.add(candidate)
        return self._clone(
            selected_candidates=tuple(item for item in CandidateId if item in values)
        )
