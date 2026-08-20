"""Recall-based retrieval evaluation over gold source IDs."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import quantiles
from typing import Any

from autobrain.models import (
    BenchmarkCase,
    CandidateEvaluation,
    CandidateId,
    CaseEvaluation,
    CostStatus,
    QualityComponents,
    Status,
    UsageSource,
)


def _percentile(values: Sequence[int], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return float(quantiles(values, n=100, method="inclusive")[int(percentile) - 1])


def evaluate_case(
    case: BenchmarkCase,
    *,
    answer: str,
    cited_source_ids: Sequence[str],
    reference_confidence: float = 1.0,
    status: Status = Status.OK,
    failure_detail: str = "",
    latency_ms: int = 0,
    candidate: CandidateId = CandidateId.MEM0,
) -> CaseEvaluation:
    """Score retrieval recall against the case gold source IDs."""
    del answer
    if not 0 <= reference_confidence <= 1:
        raise ValueError("reference_confidence must be between 0 and 1")
    gold = set(case.source_ids)
    retrieved = set(cited_source_ids)
    hits = len(gold & retrieved)
    recall = hits / len(gold) if gold else 0.0
    score = 0.0 if status != Status.OK else round(100 * recall, 4)
    components = QualityComponents(retrieval_recall=score)
    return CaseEvaluation(
        candidate=candidate,
        case_id=case.case_id,
        status=status if status != Status.OK else Status.OK,
        score=score,
        components=components,
        required_claims=len(gold),
        covered_claims=hits if status == Status.OK else 0,
        cited_claims=hits if status == Status.OK else 0,
        forbidden_matches=0,
        source_ids=list(cited_source_ids),
        generated=case.generated,
        reference_confidence=reference_confidence,
        failure_detail=failure_detail,
        latency_ms=latency_ms,
    )


def evaluate_candidate(
    candidate: CandidateId,
    cases: Sequence[tuple[Any, ...]],
    *,
    total_cost_usd: float | None,
    cost_status: CostStatus | str,
    usage_source: UsageSource = UsageSource.UNAVAILABLE,
    valid_pin: bool = True,
    corpus_hash: str | None = "a" * 64,
    direct_leakage: bool = False,
    total_input_tokens: int = 0,
    total_output_tokens: int = 0,
    ingest_wall_time_ms: int = 0,
    query_wall_time_ms: int = 0,
    workspace_bytes: int | None = None,
    operating_burden: float | None = None,
) -> CandidateEvaluation:
    """Aggregate typed case results while retaining partial failures."""
    evaluated: list[CaseEvaluation] = []
    for item in cases:
        if len(item) < 5:
            raise ValueError(
                "candidate case tuple needs case, answer, sources, confidence, latency"
            )
        benchmark_case, answer, cited_sources, confidence, latency, *rest = item
        case_status = rest[0] if rest else Status.OK
        failure_detail = rest[1] if len(rest) > 1 else ""
        evaluated.append(
            evaluate_case(
                benchmark_case,
                answer=answer,
                cited_source_ids=cited_sources,
                reference_confidence=confidence,
                status=case_status,
                failure_detail=failure_detail,
                latency_ms=latency,
                candidate=candidate,
            )
        )
    scored = len(evaluated)
    answered = sum(result.status == Status.OK for result in evaluated)
    quality = sum(result.score for result in evaluated) / scored if scored else 0.0
    latencies = [
        result.latency_ms
        for result in evaluated
        if result.status == Status.OK and result.latency_ms > 0
    ]
    return CandidateEvaluation(
        candidate=candidate,
        status=Status.OK if scored else Status.FAILED,
        scored_cases=scored,
        answered_cases=answered,
        quality_score=round(quality, 4),
        answer_success_rate=answered / scored if scored else 0.0,
        source_support_rate=quality / 100 if scored else 0.0,
        contradiction_count=sum(result.forbidden_matches for result in evaluated),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_usd=total_cost_usd,
        cost_status=CostStatus(cost_status),
        usage_source=usage_source,
        ingest_wall_time_ms=ingest_wall_time_ms,
        query_wall_time_ms=query_wall_time_ms,
        query_p50_ms=_percentile(latencies, 50),
        query_p95_ms=_percentile(latencies, 95),
        workspace_bytes=workspace_bytes,
        operating_burden=operating_burden,
        valid_pin=valid_pin,
        corpus_hash=corpus_hash,
        direct_leakage=direct_leakage,
        generated_cases=sum(getattr(item[0], "generated", False) for item in cases),
        partial_failures=scored - answered,
    )
