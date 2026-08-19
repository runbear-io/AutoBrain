from autobrain.evaluate import evaluate_candidate, evaluate_case
from autobrain.models import BenchmarkCase, CandidateId, Status


def case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="case-001",
        question="When is the launch?",
        source_ids=["slack:launch"],
        expected_claims=["The launch is Tuesday."],
        forbidden_contradictions=["The launch is Friday."],
    )


def test_quality_uses_the_four_fixed_weight_components() -> None:
    result = evaluate_case(
        case(),
        answer="The launch is Tuesday.",
        cited_source_ids=["slack:launch"],
        reference_confidence=1.0,
    )
    assert result.status == Status.OK
    assert result.required_claim_coverage == 45.0
    assert result.cited_source_support == 25.0
    assert result.contradiction_safety == 20.0
    assert result.supplementary_style == 10.0
    assert result.score == 100.0


def test_partial_failure_and_low_confidence_reference_remain_typed() -> None:
    result = evaluate_case(
        case(),
        answer="I cannot answer.",
        cited_source_ids=[],
        reference_confidence=0.2,
        status=Status.FAILED,
        failure_detail="candidate timed out",
    )
    assert result.status == Status.FAILED
    assert result.score == 0.0
    assert result.failure_detail == "candidate timed out"
    assert result.reference_confidence == 0.2


def test_forbidden_claim_removes_contradiction_safety_points() -> None:
    result = evaluate_case(
        case(),
        answer="The launch is Friday.",
        cited_source_ids=["slack:launch"],
    )
    assert result.forbidden_matches == 1
    assert result.contradiction_safety == 0


def test_candidate_aggregation_counts_successes_and_mixed_case_provenance() -> None:
    aggregate = evaluate_candidate(
        CandidateId.MEM0,
        [
            (case(), "The launch is Tuesday.", ["slack:launch"], 1.0, 10),
            (
                case().model_copy(update={"case_id": "case-002", "generated": True}),
                "",
                [],
                0.6,
                0,
                Status.FAILED,
                "provider unavailable",
            ),
        ],
        total_cost_usd=0.12,
        cost_status="COST_COMPLETE",
    )
    assert aggregate.scored_cases == 2
    assert aggregate.answered_cases == 1
    assert aggregate.generated_cases == 1
    assert aggregate.status == Status.OK
    assert aggregate.quality_score == 50.0
