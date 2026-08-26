from datetime import UTC, datetime

from autobrain.evaluate import evaluate_candidate, evaluate_case
from autobrain.models import BenchmarkCase, CandidateId, NormalizedDocument, SourceKind, Status
from autobrain.retrieval_ids import document_slug, provenance_map, resolve_retrieved_source_ids


def case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="case-001",
        question="When is the launch?",
        source_ids=["slack:launch"],
        expected_claims=["The launch is Tuesday."],
        forbidden_contradictions=["The launch is Friday."],
    )


def test_quality_is_recall_of_gold_source_ids() -> None:
    result = evaluate_case(
        case(),
        answer="ignored generation text",
        cited_source_ids=["slack:launch"],
    )
    assert result.status == Status.OK
    assert result.score == 100.0
    assert result.components.retrieval_recall == 100.0


def test_missing_gold_source_is_zero_recall_even_with_a_fluent_answer() -> None:
    result = evaluate_case(
        case(),
        answer="The launch is Tuesday.",
        cited_source_ids=["slack:other"],
    )
    assert result.score == 0.0
    assert result.components.retrieval_recall == 0.0


def test_partial_recall_is_the_fraction_of_gold_sources_retrieved() -> None:
    multi = case().model_copy(update={"source_ids": ["slack:launch", "notion:policy"]})
    result = evaluate_case(
        multi,
        answer="",
        cited_source_ids=["slack:launch", "slack:noise"],
    )
    assert result.score == 50.0
    assert result.cited_claims == 1
    assert result.required_claims == 2


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


def test_candidate_aggregation_averages_recall_and_keeps_failures() -> None:
    aggregate = evaluate_candidate(
        CandidateId.MEM0,
        [
            (case(), "ignored", ["slack:launch"], 1.0, 10),
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
    assert aggregate.source_support_rate == 0.5


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
