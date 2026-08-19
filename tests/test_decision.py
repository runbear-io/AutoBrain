from autobrain.decision import select_winner
from autobrain.models import CandidateEvaluation, CandidateId, CostStatus, Status, Verdict


def candidate(
    name: CandidateId,
    quality: float,
    *,
    cost: float | None = 1.0,
    cost_status: CostStatus = CostStatus.COMPLETE,
    latency: float = 100.0,
    eligible: bool = True,
    direct_leakage: bool = False,
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
    result = select_winner(
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
    result = select_winner(
        [
            candidate(CandidateId.LLM_WIKI, 80, cost=2.0),
            candidate(CandidateId.MEM0, 78, cost=1.0),
        ]
    )
    assert result.verdict == Verdict.MEM0


def test_each_candidate_has_a_canonical_winner_label() -> None:
    for winner in CandidateId:
        result = select_winner([candidate(winner, 90)])
        assert result.verdict == Verdict(winner.value)


def test_latency_breaks_a_complete_cost_tie() -> None:
    result = select_winner(
        [
            candidate(CandidateId.LLM_WIKI, 80, cost=1.0, latency=200),
            candidate(CandidateId.MEM0, 80, cost=1.0, latency=100),
        ]
    )
    assert result.verdict == Verdict.MEM0


def test_holdout_leakage_is_ineligible_without_override() -> None:
    result = select_winner([candidate(CandidateId.GBRAIN, 90, direct_leakage=True)])
    assert result.verdict == Verdict.NO_RECOMMENDATION


def test_incomplete_cost_and_no_eligible_have_typed_verdicts() -> None:
    resolved_without_cost = select_winner(
        [
            candidate(CandidateId.LLM_WIKI, 80, cost=None, cost_status=CostStatus.INCOMPLETE),
            candidate(CandidateId.MEM0, 78, cost=None, cost_status=CostStatus.INCOMPLETE),
        ]
    )
    assert resolved_without_cost.status == Status.NO_RECOMMENDATION
    assert resolved_without_cost.verdict == Verdict.NO_RECOMMENDATION
    assert "cost is incomplete" in resolved_without_cost.rationale
    no_recommendation = select_winner(
        [
            candidate(CandidateId.LLM_WIKI, 59, eligible=False),
            candidate(CandidateId.MEM0, 50, eligible=False),
        ]
    )
    assert no_recommendation.verdict == Verdict.NO_RECOMMENDATION
