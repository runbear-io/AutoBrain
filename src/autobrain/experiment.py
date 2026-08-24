"""Automatic experiment planning for the interactive AutoBrain flow."""

from __future__ import annotations

from dataclasses import dataclass, field

from autobrain.auth.models import Provider
from autobrain.candidates.gbrain_config import GBrainExecutionConfig
from autobrain.embedding import EmbeddingReadiness
from autobrain.models import CandidateId
from autobrain.subscription import ProviderId, SubscriptionStatus


class ExperimentSetupError(ValueError):
    """A typed setup failure that the TUI can explain to the user."""


@dataclass(frozen=True)
class ExperimentPlan:
    title: str
    description: str
    provider_mode: str
    embedding_backend: str
    sources: tuple[Provider, ...]
    candidates: tuple[CandidateId, ...]
    budget_usd: float
    max_questions: int
    gbrain_config: GBrainExecutionConfig = field(default_factory=GBrainExecutionConfig.quick_start)


_CANDIDATE_LABELS = {
    CandidateId.LLM_WIKI: "LLM Wiki",
    CandidateId.MEM0: "Mem0 OSS",
    CandidateId.GBRAIN: "GBrain",
}


def automatic_experiment_copy(
    *,
    sources: tuple[Provider, ...],
    candidates: tuple[CandidateId, ...],
) -> tuple[str, str]:
    source_label = " + ".join(source.value.title() for source in sources)
    candidate_label = ", ".join(_CANDIDATE_LABELS[candidate] for candidate in candidates)
    return (
        f"Find the best knowledge system for {source_label}",
        f"Compare {candidate_label} on grounded questions from {source_label}.",
    )


def build_automatic_plan(
    *,
    sources: tuple[Provider, ...],
    candidates: tuple[CandidateId, ...],
    subscription_status: SubscriptionStatus,
    embedding_readiness: EmbeddingReadiness,
    subscription_provider: ProviderId = ProviderId.CODEX,
    gbrain_config: GBrainExecutionConfig | None = None,
) -> ExperimentPlan:
    """Own all experiment decisions except sources and candidate scope."""
    if not sources:
        raise ExperimentSetupError("KNOWLEDGE_SOURCE_REQUIRED: select Slack or Notion")
    if len(candidates) < 2:
        raise ExperimentSetupError("TWO_CANDIDATES_REQUIRED: select at least two candidates")

    if subscription_status is not SubscriptionStatus.READY:
        provider_label = (
            "ChatGPT" if subscription_provider is ProviderId.CODEX else subscription_provider.value
        )
        raise ExperimentSetupError(
            f"{subscription_status.value}: connect {provider_label} subscription"
        )
    selected_gbrain = gbrain_config or GBrainExecutionConfig.quick_start()
    if embedding_readiness.backend == "local-hash":
        raise ExperimentSetupError(embedding_readiness.detail)
    if not selected_gbrain.keyword_only and (
        not embedding_readiness.recommendation_ready or embedding_readiness.backend is None
    ):
        raise ExperimentSetupError(embedding_readiness.detail)

    title, description = automatic_experiment_copy(sources=sources, candidates=candidates)
    return ExperimentPlan(
        title=title,
        description=description,
        provider_mode=f"{subscription_provider.value}-subscription",
        embedding_backend=embedding_readiness.backend or "unavailable",
        sources=sources,
        candidates=candidates,
        budget_usd=25.0,
        max_questions=20 if len(sources) == 1 else 30,
        gbrain_config=selected_gbrain,
    )
