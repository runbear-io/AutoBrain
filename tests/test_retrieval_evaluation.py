from __future__ import annotations

from typing import cast

from autobrain.evaluate import evaluate_candidate, evaluate_case
from autobrain.experiment_contracts import RetrievalMetrics
from autobrain.models import BenchmarkCase, CandidateId, CostStatus, EvaluationMode


def _case(
    *,
    case_id: str = "case-001",
    source_ids: list[str] | None = None,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        question="When is the launch?",
        source_ids=source_ids or ["slack:launch"],
        expected_claims=[],
        forbidden_contradictions=[],
    )


def test_retrieval_only_is_default_and_does_not_require_answer_generation() -> None:
    result = evaluate_case(
        _case(),
        answer="",
        cited_source_ids=["slack:launch"],
    )

    assert result.evaluation_mode is EvaluationMode.RETRIEVAL_ONLY
    assert result.score == 100.0
    assert result.retrieval_recall == 100.0
    assert result.required_claims == 0
    assert result.covered_claims == 0


def test_retrieval_only_results_are_comparable_across_fixture_candidates() -> None:
    cases: list[tuple[BenchmarkCase, str, list[str], float, int]] = [
        (_case(), "", ["slack:launch"], 1.0, 10),
        (_case(case_id="case-002", source_ids=["notion:policy"]), "", cast(list[str], []), 1.0, 12),
    ]

    evaluations = [
        evaluate_candidate(
            candidate,
            cases,
            total_cost_usd=1.0,
            cost_status=CostStatus.COMPLETE,
        )
        for candidate in CandidateId
    ]

    assert [item.quality_score for item in evaluations] == [50.0] * 3
    assert [item.source_support_rate for item in evaluations] == [0.5] * 3
    assert [item.answer_success_rate for item in evaluations] == [1.0] * 3
    assert [item.query_p50_ms for item in evaluations] == [11.0] * 3
    assert [item.query_p95_ms for item in evaluations] == [11.9] * 3
    assert [item.total_cost_usd for item in evaluations] == [1.0] * 3
    assert [item.cost_status for item in evaluations] == [CostStatus.COMPLETE] * 3


def test_retrieval_metrics_expose_recall_precision_missing_evidence_and_noise() -> None:
    metrics = RetrievalMetrics(
        relevant_retrieved=2,
        retrieved=4,
        relevant_available=3,
        missing_evidence=1,
        noise=2,
        latency_ms=12.5,
        cost_status=CostStatus.UNAVAILABLE,
    )

    assert metrics.recall == 2 / 3
    assert metrics.precision == 0.5
    assert metrics.missing_evidence == 1
    assert metrics.noise == 2
    assert metrics.latency_ms == 12.5
    assert metrics.cost_status is CostStatus.UNAVAILABLE
