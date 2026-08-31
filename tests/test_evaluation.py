from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from autobrain.evaluate import evaluate_candidate, evaluate_case
from autobrain.models import (
    BenchmarkCase,
    CandidateId,
    EvaluationMode,
    NormalizedDocument,
    SourceKind,
    Status,
)
from autobrain.retrieval_ids import document_slug, provenance_map, resolve_retrieved_source_ids


def case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="case-001",
        question="When is the launch?",
        source_ids=["slack:launch"],
        expected_claims=["The launch is Tuesday."],
        forbidden_contradictions=["The launch is Friday."],
    )


def test_quality_scores_expected_answer_claims_even_when_citation_matches() -> None:
    result = evaluate_case(
        case(),
        answer="The launch is Tuesday.",
        cited_source_ids=["slack:launch"],
        mode=EvaluationMode.ANSWER_AWARE,
    )
    assert result.status == Status.OK
    assert result.score == 100.0
    assert result.components.retrieval_recall == 100.0
    assert result.required_claims == 1
    assert result.covered_claims == 1


def test_wrong_answer_scores_lower_than_correct_answer_with_same_citation() -> None:
    correct = evaluate_case(
        case(),
        answer="The launch is Tuesday.",
        cited_source_ids=["slack:launch"],
        mode=EvaluationMode.ANSWER_AWARE,
    )
    wrong = evaluate_case(
        case(),
        answer="The launch is Friday.",
        cited_source_ids=["slack:launch"],
        mode=EvaluationMode.ANSWER_AWARE,
    )
    assert correct.score == 100.0
    assert wrong.score < correct.score
    assert wrong.covered_claims == 0
    assert wrong.forbidden_matches == 1
    assert wrong.components.retrieval_recall == correct.components.retrieval_recall


def test_missing_gold_source_keeps_answer_score_but_has_no_citation_support() -> None:
    result = evaluate_case(
        case(),
        answer="The launch is Tuesday.",
        cited_source_ids=["slack:other"],
        mode=EvaluationMode.ANSWER_AWARE,
    )
    assert result.score == 75.0
    assert result.components.retrieval_recall == 0.0
    assert result.cited_claims == 0


def test_partial_recall_is_the_fraction_of_gold_sources_retrieved() -> None:
    multi = case().model_copy(update={"source_ids": ["slack:launch", "notion:policy"]})
    result = evaluate_case(
        multi,
        answer="",
        cited_source_ids=["slack:launch", "slack:noise"],
        mode=EvaluationMode.ANSWER_AWARE,
    )
    assert result.score == 20.0
    assert result.components.retrieval_recall == 50.0
    assert result.cited_claims == 0
    assert result.required_claims == 1


def test_failed_retrieval_stays_zero() -> None:
    result = evaluate_case(
        case(),
        answer="",
        cited_source_ids=[],
        status=Status.FAILED,
        failure_detail="candidate timed out",
    )
    assert result.status == Status.FAILED
    assert result.score == 0.0
    assert result.failure_detail == "candidate timed out"


def test_candidate_aggregation_accepts_streaming_cases() -> None:
    def cases() -> Iterator[tuple[Any, ...]]:
        yield (case(), "The launch is Tuesday.", ["slack:launch"], 1.0, 10)
        yield (
            case().model_copy(update={"case_id": "case-002", "generated": True}),
            "",
            [],
            0.6,
            0,
            Status.FAILED,
            "provider unavailable",
        )

    aggregate = evaluate_candidate(
        CandidateId.MEM0,
        cases(),
        total_cost_usd=0.12,
        cost_status="COST_COMPLETE",
        mode=EvaluationMode.ANSWER_AWARE,
    )
    assert aggregate.scored_cases == 2
    assert aggregate.answered_cases == 1
    assert aggregate.generated_cases == 1
    assert aggregate.quality_score == 50.0
    assert aggregate.partial_failures == 1


def test_candidate_aggregation_averages_recall_and_keeps_failures() -> None:
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
        mode=EvaluationMode.ANSWER_AWARE,
    )
    assert aggregate.scored_cases == 2
    assert aggregate.answered_cases == 1
    assert aggregate.generated_cases == 1
    assert aggregate.status == Status.OK
    assert aggregate.quality_score == 50.0
    assert aggregate.source_support_rate == 0.5


def test_retrieval_only_is_default_and_does_not_require_answer_generation() -> None:
    retrieval_case = case().model_copy(update={"expected_claims": []})

    result = evaluate_case(
        retrieval_case,
        answer="",
        cited_source_ids=["slack:launch"],
    )

    assert result.evaluation_mode is EvaluationMode.RETRIEVAL_ONLY
    assert result.score == 100.0
    assert result.retrieval_recall == 100.0
    assert result.required_claims == 0
    assert result.covered_claims == 0


def test_retrieval_only_metrics_are_comparable_across_fixture_candidates() -> None:
    retrieval_case = case().model_copy(update={"expected_claims": []})
    cases: list[tuple[Any, ...]] = [
        (retrieval_case, "", ["slack:launch"], 1.0, 10),
        (
            retrieval_case.model_copy(
                update={"case_id": "case-002", "source_ids": ["notion:policy"]}
            ),
            "",
            [],
            1.0,
            12,
        ),
    ]

    evaluations = [
        evaluate_candidate(
            candidate,
            cases,
            total_cost_usd=1.0,
            cost_status="COST_COMPLETE",
        )
        for candidate in CandidateId
    ]

    assert [item.quality_score for item in evaluations] == [50.0] * 3
    assert [item.source_support_rate for item in evaluations] == [0.5] * 3
    assert [item.answer_success_rate for item in evaluations] == [1.0] * 3


def test_answer_evaluation_remains_explicitly_separate() -> None:
    result = evaluate_case(
        case(),
        answer="The launch is Friday.",
        cited_source_ids=["slack:launch"],
        mode=EvaluationMode.ANSWER_AWARE,
    )

    assert result.evaluation_mode is EvaluationMode.ANSWER_AWARE
    assert result.score == 10.0
    assert result.forbidden_matches == 1


def test_retrieved_slugs_map_back_to_gold_source_ids() -> None:
    document = NormalizedDocument(
        source_id="notion:page-1",
        source_kind=SourceKind.NOTION_PAGE,
        canonical_url="https://notion.example/page-1",
        title="Launch",
        text="Launch is Tuesday.",
        content_hash="a" * 64,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    mapping = provenance_map([document])
    slug = document_slug("notion:page-1")
    assert resolve_retrieved_source_ids(
        [{"page_slug": slug}, {"slug": "unrelated"}],
        mapping,
    ) == ["notion:page-1"]
