"""Pure setup state machine for the AutoBrain terminal cockpit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from autobrain.auth.models import Provider
from autobrain.models import CandidateId


class WizardSection(StrEnum):
    CONNECTIONS = "connections"
    KNOWLEDGE_SOURCES = "knowledge_sources"
    CANDIDATES = "candidates"
    REVIEW = "review"
    RUNNING = "running"
    RESULTS = "results"


_SETUP_SECTIONS = (
    WizardSection.CONNECTIONS,
    WizardSection.KNOWLEDGE_SOURCES,
    WizardSection.CANDIDATES,
    WizardSection.REVIEW,
)


@dataclass(frozen=True)
class TUIState:
    section: WizardSection = WizardSection.CONNECTIONS
    selected_sources: tuple[Provider, ...] = (Provider.SLACK, Provider.NOTION)
    selected_candidates: tuple[CandidateId, ...] = tuple(CandidateId)

    def advance(self) -> TUIState:
        if self.section not in _SETUP_SECTIONS:
            return self
        index = _SETUP_SECTIONS.index(self.section)
        return TUIState(
            section=_SETUP_SECTIONS[min(index + 1, len(_SETUP_SECTIONS) - 1)],
            selected_sources=self.selected_sources,
            selected_candidates=self.selected_candidates,
        )

    def back(self) -> TUIState:
        if self.section not in _SETUP_SECTIONS:
            return self.with_section(WizardSection.REVIEW)
        index = _SETUP_SECTIONS.index(self.section)
        return self.with_section(_SETUP_SECTIONS[max(0, index - 1)])

    def with_section(self, section: WizardSection) -> TUIState:
        return TUIState(
            section=section,
            selected_sources=self.selected_sources,
            selected_candidates=self.selected_candidates,
        )

    def toggle_source(self, provider: Provider) -> TUIState:
        values = set(self.selected_sources)
        if provider in values:
            values.remove(provider)
        else:
            values.add(provider)
        return TUIState(
            section=self.section,
            selected_sources=tuple(item for item in Provider if item in values),
            selected_candidates=self.selected_candidates,
        )

    def toggle_candidate(self, candidate: CandidateId) -> TUIState:
        values = set(self.selected_candidates)
        if candidate in values:
            values.remove(candidate)
        else:
            values.add(candidate)
        return TUIState(
            section=self.section,
            selected_sources=self.selected_sources,
            selected_candidates=tuple(item for item in CandidateId if item in values),
        )
