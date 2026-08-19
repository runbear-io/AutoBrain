from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from autobrain.models import (
    BenchmarkCase,
    CandidateId,
    CandidateObservation,
    ConnectionState,
    CoverageCompleteness,
    CoverageRecord,
    Holdout,
    McpCapability,
    NormalizedDocument,
    QualityResult,
    SourceKind,
    Status,
    UsageCost,
    Verdict,
)


def test_contracts_are_strict_and_round_trip() -> None:
    now = datetime.now(UTC)
    document = NormalizedDocument(
        source_id="slack:channel:C1:message:1.0",
        source_kind=SourceKind.SLACK_MESSAGE,
        canonical_url="https://example.test/message",
        title="Decision",
        text="Use the blue deployment.",
        content_hash="a" * 64,
        created_at=now,
        metadata={"channel_id": "C1"},
    )
    case = BenchmarkCase(
        case_id="case-001",
        question="Which deployment is used?",
        source_ids=[document.source_id],
        expected_claims=["The blue deployment is used."],
    )
    observation = CandidateObservation(
        candidate=CandidateId.MEM0,
        status=Status.OK,
        answer="Blue.",
        source_ids=[document.source_id],
        usage=UsageCost(input_tokens=1, output_tokens=1, usd=0.01),
        latency_ms=2,
    )
    quality = QualityResult(score=0.9, supported_claims=1, total_claims=1)
    assert case.model_validate_json(case.model_dump_json()) == case
    assert observation.usage is not None
    assert observation.usage.usd == 0.01
    assert quality.score == 0.9
    assert Holdout(case_id=case.case_id, source_ids=case.source_ids, reference_text="Blue")
    assert CoverageRecord(
        source=SourceKind.SLACK_THREAD,
        completeness=CoverageCompleteness.SEARCH_DISCOVERED,
        discovered=1,
        fetched=1,
    )
    assert ConnectionState.CONNECTED.value == "CONNECTED"
    assert McpCapability.SEARCH.value == "SEARCH"
    assert Verdict.NO_DECISION.value == "NO_DECISION"


def test_extra_fields_and_coercion_are_rejected() -> None:
    with pytest.raises(ValidationError):
        UsageCost(input_tokens="1", output_tokens=1, usd=0)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        QualityResult(score=1, supported_claims=1, total_claims=1, surprise=True)  # type: ignore[call-arg]
