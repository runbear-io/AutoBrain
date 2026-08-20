import pytest

from autobrain.decision import select_winner
from autobrain.embedding import EmbeddingBackendConfig, production_embedding_registry
from autobrain.models import (
    CandidateEvaluation,
    CandidateId,
    CostStatus,
    EmbeddingProvenance,
    EmbeddingQuality,
    Status,
    UsageSource,
    Verdict,
)

_SEMANTIC_EMBEDDING = EmbeddingBackendConfig.from_environ(
    {"OPENAI_API_KEY": "fixture-embedding-key"},
    requested="openai",
).descriptor


def _select(candidates: list[CandidateEvaluation]):
    return select_winner(candidates, embedding=_SEMANTIC_EMBEDDING)


def candidate(
    name: CandidateId,
    quality: float,
    *,
    cost: float | None = 1.0,
    cost_status: CostStatus = CostStatus.COMPLETE,
    latency: float = 100.0,
    eligible: bool = True,
    direct_leakage: bool = False,
    usage_source: UsageSource = UsageSource.MEASURED,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate=name,
        status=Status.OK,
        scored_cases=20,
        answered_cases=20,
        quality_score=quality,
        answer_success_rate=1.0,
        source_support_rate=0.8,
        contradiction_count=0,
        total_input_tokens=100,
        total_output_tokens=100,
        total_cost_usd=cost,
        cost_status=cost_status,
        usage_source=usage_source,
        query_p50_ms=latency / 2,
        query_p95_ms=latency,
        workspace_bytes=100,
        operating_burden=2.0,
        valid_pin=True,
        corpus_hash="a" * 64,
        direct_leakage=direct_leakage,
        eligible_override=eligible,
    )


def test_quality_floor_and_quality_outside_epsilon_win() -> None:
    result = _select(
        [
            candidate(CandidateId.LLM_WIKI, 74),
            candidate(CandidateId.MEM0, 80),
            candidate(CandidateId.GBRAIN, 59),
        ]
    )
    assert result.verdict == Verdict.MEM0
    assert result.status == Status.OK
    assert "quality" in result.rationale.lower()


def test_complete_cost_breaks_close_quality_tie() -> None:
    result = _select(
        [
            candidate(CandidateId.LLM_WIKI, 80, cost=2.0),
            candidate(CandidateId.MEM0, 78, cost=1.0),
        ]
    )
    assert result.verdict == Verdict.MEM0


def test_each_candidate_has_a_canonical_winner_label() -> None:
    for winner in CandidateId:
        result = _select([candidate(winner, 90)])
        assert result.verdict == Verdict(winner.value)


def test_latency_breaks_a_complete_cost_tie() -> None:
    result = _select(
        [
            candidate(CandidateId.LLM_WIKI, 80, cost=1.0, latency=200),
            candidate(CandidateId.MEM0, 80, cost=1.0, latency=100),
        ]
    )
    assert result.verdict == Verdict.MEM0


@pytest.mark.parametrize("usage_source", [UsageSource.ESTIMATED, UsageSource.UNAVAILABLE])
def test_unmeasured_usage_is_ineligible_even_with_claimed_complete_cost(
    usage_source: UsageSource,
) -> None:
    result = _select(
        [
            candidate(
                CandidateId.MEM0,
                99,
                cost=0.01,
                cost_status=CostStatus.COMPLETE,
                usage_source=usage_source,
            )
        ]
    )

    assert result.verdict == Verdict.NO_RECOMMENDATION
    assert result.status == Status.NO_RECOMMENDATION
    assert usage_source.value in result.rationale


def test_holdout_leakage_is_ineligible_without_override() -> None:
    result = _select([candidate(CandidateId.GBRAIN, 90, direct_leakage=True)])
    assert result.verdict == Verdict.NO_RECOMMENDATION


@pytest.mark.parametrize(
    ("embedding", "reason"),
    [
        (
            EmbeddingProvenance(
                backend="local-hash-embedding",
                quality=EmbeddingQuality.SEMANTIC,
            ),
            "local-hash-embedding is smoke-only",
        ),
        (
            EmbeddingProvenance(
                backend="forged-semantic-backend",
                quality=EmbeddingQuality.SEMANTIC,
            ),
            "forged-semantic-backend is not registered",
        ),
    ],
)
def test_embedding_quality_claims_cannot_forge_recommendation_eligibility(
    embedding: EmbeddingProvenance,
    reason: str,
) -> None:
    result = select_winner(
        [candidate(CandidateId.MEM0, 90)],
        embedding=embedding,
    )

    assert result.status is Status.NO_RECOMMENDATION
    assert result.verdict is Verdict.NO_RECOMMENDATION
    assert reason in result.rationale


def test_explicitly_registered_test_semantic_backend_is_eligible() -> None:
    registry = production_embedding_registry().with_test_semantic_backend()
    config = EmbeddingBackendConfig.from_environ(
        {},
        requested="test-semantic",
        registry=registry,
    )

    result = select_winner(
        [candidate(CandidateId.MEM0, 90)],
        embedding=config.descriptor,
        embedding_registry=registry,
    )

    assert result.status is Status.OK
    assert result.verdict is Verdict.MEM0


def test_incomplete_cost_and_no_eligible_have_typed_verdicts() -> None:
    resolved_without_cost = _select(
        [
            candidate(CandidateId.LLM_WIKI, 80, cost=None, cost_status=CostStatus.INCOMPLETE),
            candidate(CandidateId.MEM0, 78, cost=None, cost_status=CostStatus.INCOMPLETE),
        ]
    )
    assert resolved_without_cost.status == Status.NO_RECOMMENDATION
    assert resolved_without_cost.verdict == Verdict.NO_RECOMMENDATION
    assert "cost is incomplete" in resolved_without_cost.rationale
    no_recommendation = _select(
        [
            candidate(CandidateId.LLM_WIKI, 59, eligible=False),
            candidate(CandidateId.MEM0, 50, eligible=False),
        ]
    )
    assert no_recommendation.verdict == Verdict.NO_RECOMMENDATION
