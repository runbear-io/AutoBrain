"""Deterministic typed quality evaluation for candidate answers."""

from __future__ import annotations

import re
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
)

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(value: str) -> set[str]:
    return set(_WORD.findall(value.lower()))


def _claim_present(claim: str, answer: str) -> bool:
    claim_tokens = _tokens(claim)
    answer_tokens = _tokens(answer)
    return bool(claim_tokens) and claim_tokens <= answer_tokens


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
    """Score one answer with the fixed 45/25/20/10 rubric."""
    if not 0 <= reference_confidence <= 1:
        raise ValueError("reference_confidence must be between 0 and 1")
    if status != Status.OK:
        components = QualityComponents(
            required_claim_coverage=0,
            cited_source_support=0,
            contradiction_safety=0,
            supplementary_style=0,
        )
        return CaseEvaluation(
            candidate=candidate,
            case_id=case.case_id,
            status=status,
            score=0,
            components=components,
            required_claims=len(case.expected_claims),
            covered_claims=0,
            cited_claims=0,
            forbidden_matches=0,
            source_ids=list(cited_source_ids),
            generated=case.generated,
            reference_confidence=reference_confidence,
            failure_detail=failure_detail,
            latency_ms=latency_ms,
        )
    covered = sum(_claim_present(claim, answer) for claim in case.expected_claims)
    cited = sum(
        _claim_present(claim, answer) and bool(set(cited_source_ids).intersection(case.source_ids))
        for claim in case.expected_claims
    )
    forbidden_matches = sum(
        _claim_present(claim, answer) for claim in case.forbidden_contradictions
    )
    required_score = 45 * covered / len(case.expected_claims) if case.expected_claims else 0
    if case.expected_claims:
        cited_score = 25 * cited / len(case.expected_claims) * reference_confidence
    else:
        cited_score = 0
    safety_score = 0 if forbidden_matches else 20
    style_score = 10 if answer.strip() and cited_source_ids else 0
    components = QualityComponents(
        required_claim_coverage=round(required_score, 4),
        cited_source_support=round(cited_score, 4),
        contradiction_safety=safety_score,
        supplementary_style=style_score,
    )
    return CaseEvaluation(
        candidate=candidate,
        case_id=case.case_id,
        status=Status.OK,
        score=components.total,
        components=components,
        required_claims=len(case.expected_claims),
        covered_claims=covered,
        cited_claims=cited,
        forbidden_matches=forbidden_matches,
        source_ids=list(cited_source_ids),
        generated=case.generated,
        reference_confidence=reference_confidence,
        latency_ms=latency_ms,
    )


def evaluate_candidate(
    candidate: CandidateId,
    cases: Sequence[tuple[Any, ...]],
    *,
    total_cost_usd: float | None,
    cost_status: CostStatus | str,
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
    required = sum(result.required_claims for result in evaluated)
    cited = sum(result.cited_claims for result in evaluated)
    latencies = [result.latency_ms for result in evaluated if result.status == Status.OK]
    return CandidateEvaluation(
        candidate=candidate,
        status=Status.OK if scored else Status.FAILED,
        scored_cases=scored,
        answered_cases=answered,
        quality_score=round(quality, 4),
        answer_success_rate=answered / scored if scored else 0.0,
        source_support_rate=cited / required if required else 0.0,
        contradiction_count=sum(result.forbidden_matches for result in evaluated),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_usd=total_cost_usd,
        cost_status=CostStatus(cost_status),
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
