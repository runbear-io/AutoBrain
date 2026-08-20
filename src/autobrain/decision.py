"""Deterministic quality-first candidate selection."""

from __future__ import annotations

from collections.abc import Sequence

from autobrain.models import (
    CandidateEvaluation,
    CostStatus,
    DecisionResult,
    Status,
    UsageSource,
    Verdict,
)

QUALITY_FLOOR = 60.0
CLOSE_QUALITY_EPSILON = 5.0
MIN_SCORED_CASES = 20
MIN_SUCCESS_RATE = 0.9
MIN_SOURCE_SUPPORT_RATE = 0.5


def _eligibility(candidate: CandidateEvaluation, quality_floor: float) -> list[str]:
    reasons: list[str] = []
    if candidate.eligible_override is False:
        reasons.append("explicitly marked ineligible")
    if candidate.scored_cases < MIN_SCORED_CASES:
        reasons.append(f"fewer than {MIN_SCORED_CASES} scored cases")
    if candidate.answer_success_rate < MIN_SUCCESS_RATE:
        reasons.append("answer success rate below 90%")
    if candidate.quality_score < quality_floor:
        reasons.append(f"mean quality below {quality_floor:g}")
    if candidate.source_support_rate < MIN_SOURCE_SUPPORT_RATE:
        reasons.append("source-support rate below 50%")
    if not candidate.valid_pin:
        reasons.append("candidate pin is invalid")
    if not candidate.corpus_hash:
        reasons.append("corpus hash is missing")
    if candidate.direct_leakage:
        reasons.append("direct holdout/oracle leakage detected")
    if candidate.usage_source is not UsageSource.MEASURED:
        reasons.append(f"usage is {candidate.usage_source.value}")
    if candidate.cost_status is not CostStatus.COMPLETE:
        reasons.append(f"candidate cost is {candidate.cost_status.value.lower()}")
    elif candidate.total_cost_usd is None:
        reasons.append("candidate cost is missing")
    if candidate.status != Status.OK:
        reasons.append(f"candidate status is {candidate.status.value}")
    if candidate.partial_failures > 0:
        reasons.append("partial candidate failures are not eligible to win")
    return reasons


def select_winner(
    candidates: Sequence[CandidateEvaluation],
    *,
    quality_floor: float = QUALITY_FLOOR,
    close_quality_epsilon: float = CLOSE_QUALITY_EPSILON,
) -> DecisionResult:
    """Apply the canonical eligibility, quality, cost, latency, burden rules."""
    if not candidates:
        return DecisionResult(
            status=Status.NO_RECOMMENDATION,
            verdict=Verdict.NO_RECOMMENDATION,
            rationale="No candidate results were available.",
            tie_break_metric="candidate_query_p95_ms",
        )
    eligible: list[CandidateEvaluation] = []
    incomplete_cost_candidates: list[str] = []
    unmeasured_usage_candidates: list[str] = []
    considered = [candidate.candidate for candidate in candidates]
    for candidate in candidates:
        reasons = _eligibility(candidate, quality_floor)
        if candidate.cost_status is not CostStatus.COMPLETE or candidate.total_cost_usd is None:
            incomplete_cost_candidates.append(candidate.candidate.value)
        if candidate.usage_source is not UsageSource.MEASURED:
            unmeasured_usage_candidates.append(
                f"{candidate.candidate.value} ({candidate.usage_source.value})"
            )
        candidate = candidate.model_copy(update={"eligibility_reasons": reasons})
        if not reasons:
            eligible.append(candidate)
    if not eligible:
        cost_detail = (
            " cost is incomplete for: " + ", ".join(sorted(incomplete_cost_candidates)) + "."
            if incomplete_cost_candidates
            else ""
        )
        usage_detail = (
            " usage is not measured for: " + ", ".join(sorted(unmeasured_usage_candidates)) + "."
            if unmeasured_usage_candidates
            else ""
        )
        return DecisionResult(
            status=Status.NO_RECOMMENDATION,
            verdict=Verdict.NO_RECOMMENDATION,
            rationale=(
                "No candidate met the quality, reliability, provenance, leakage, "
                f"and complete-cost gates.{cost_detail}{usage_detail}"
            ),
            considered_candidates=considered,
            quality_floor=quality_floor,
            close_quality_epsilon=close_quality_epsilon,
            tie_break_metric="candidate_query_p95_ms",
        )
    highest_quality = max(candidate.quality_score for candidate in eligible)
    close = [
        candidate
        for candidate in eligible
        if highest_quality - candidate.quality_score <= close_quality_epsilon
    ]
    if len(close) == 1:
        winner = close[0]
        rationale = (
            f"{winner.candidate.value} has the highest eligible quality "
            f"({winner.quality_score:.2f}/100)."
        )
    else:
        winner = min(
            close,
            key=lambda candidate: (
                candidate.total_cost_usd if candidate.total_cost_usd is not None else float("inf"),
                candidate.query_p95_ms if candidate.query_p95_ms is not None else float("inf"),
                candidate.operating_burden
                if candidate.operating_burden is not None
                else float("inf"),
                candidate.candidate.value,
            ),
        )
        rationale = (
            f"Quality is within {close_quality_epsilon:g} points; {winner.candidate.value} "
            "wins on complete measured cost, latency, operating burden, and stable ID order."
        )
    if incomplete_cost_candidates:
        rationale += (
            " cost is incomplete and therefore ineligible for: "
            + ", ".join(sorted(incomplete_cost_candidates))
            + "."
        )
    return DecisionResult(
        status=Status.OK,
        verdict=Verdict(winner.candidate.value),
        rationale=rationale,
        eligible_candidates=[candidate.candidate for candidate in eligible],
        considered_candidates=considered,
        quality_floor=quality_floor,
        close_quality_epsilon=close_quality_epsilon,
        tie_break_metric="candidate_query_p95_ms",
    )
