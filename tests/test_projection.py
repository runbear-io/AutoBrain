"""Versioned redacted projection of a comparison artifact.

The projection is the only shape allowed to cross the local HTTP boundary. It
is deliberately narrower than `ComparisonArtifact`: it carries no evidence
text, no oracle fields and no raw source identifiers, so a browser client can
render a run summary without ever seeing corpus content.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from autobrain.models import (
    BenchmarkProvenance,
    CandidateEvaluation,
    CandidateId,
    ComparisonArtifact,
    CostStatus,
    DecisionResult,
    Status,
    Verdict,
)
from autobrain.projection import (
    PROJECTION_SCHEMA_VERSION,
    RunProjection,
    project_comparison,
)
from autobrain.report import build_comparison, redact_payload

CORPUS_HASH = "a" * 64
BENCHMARK_HASH = "b" * 64


def evaluation(
    candidate: CandidateId,
    *,
    quality: float,
    status: Status = Status.OK,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate=candidate,
        status=status,
        scored_cases=30,
        answered_cases=28,
        quality_score=quality,
        answer_success_rate=0.93,
        source_support_rate=0.79,
        contradiction_count=1,
        total_input_tokens=1200,
        total_output_tokens=340,
        total_cost_usd=1.25,
        cost_status=CostStatus.COMPLETE,
        query_p50_ms=820.0,
        query_p95_ms=2460.0,
        operating_burden=2.0,
        valid_pin=True,
        corpus_hash=CORPUS_HASH,
    )


def artifact(
    *,
    status: Status = Status.OK,
    verdict: Verdict = Verdict.GBRAIN,
    rationale: str = "GBrain leads grounded recall.",
):
    return build_comparison(
        run_id="RUN-A41F",
        status=status,
        corpus_hash=CORPUS_HASH,
        benchmark_hash=BENCHMARK_HASH,
        coverage=[],
        candidates=[
            evaluation(CandidateId.GBRAIN, quality=93.6),
            evaluation(CandidateId.MEM0, quality=82.4),
        ],
        decision=DecisionResult(
            status=status,
            verdict=verdict,
            rationale=rationale,
            eligible_candidates=[CandidateId.GBRAIN, CandidateId.MEM0],
            considered_candidates=[CandidateId.GBRAIN, CandidateId.MEM0],
        ),
        evidence=[],
    )


def test_projection_is_versioned_and_carries_the_run_summary() -> None:
    projection = project_comparison(artifact())

    assert projection.schema_version == PROJECTION_SCHEMA_VERSION
    assert projection.run_id == "RUN-A41F"
    assert projection.status is Status.OK
    assert projection.verdict is Verdict.GBRAIN
    assert projection.corpus_hash == CORPUS_HASH
    assert [candidate.candidate for candidate in projection.candidates] == [
        CandidateId.GBRAIN,
        CandidateId.MEM0,
    ]


def test_projection_schema_version_is_pinned_and_rejects_other_versions() -> None:
    payload = project_comparison(artifact()).model_dump(mode="json")
    payload["schema_version"] = PROJECTION_SCHEMA_VERSION + 1

    with pytest.raises(ValidationError):
        RunProjection.model_validate(payload)


def test_projection_never_exposes_evidence_or_oracle_fields() -> None:
    payload = project_comparison(artifact()).model_dump(mode="json")

    assert "evidence" not in payload
    for forbidden in ("expected_claims", "reference_text", "holdout", "answer"):
        assert forbidden not in payload

    for candidate in payload["candidates"]:
        assert "answer" not in candidate
        assert "source_ids" not in candidate


def test_projection_redacts_secrets_carried_in_free_text() -> None:
    """The projection redacts independently of how its input was produced.

    `build_comparison` already redacts, so this builds the artifact directly to
    prove the projection is not merely inheriting an upstream guarantee. That
    matters because an artifact can also arrive from `load_comparison` of a
    file written by an older version.
    """
    secret = "sk-abcdef0123456789"
    leaky = ComparisonArtifact(
        schema_version=2,
        run_id="RUN-A41F",
        status=Status.OK,
        corpus_hash=CORPUS_HASH,
        benchmark_hash=BENCHMARK_HASH,
        verdict=Verdict.GBRAIN,
        decision=DecisionResult(
            status=Status.OK,
            verdict=Verdict.GBRAIN,
            rationale=f"selected using {secret} from the operator shell",
        ),
        candidates=[evaluation(CandidateId.GBRAIN, quality=93.6)],
        provenance=BenchmarkProvenance(),
        warnings=[f"operator exported {secret}"],
    )
    assert secret in leaky.decision.rationale

    projection = project_comparison(leaky)

    assert secret not in projection.rationale
    assert "[REDACTED]" in projection.rationale
    assert all(secret not in warning for warning in projection.warnings)


def test_projection_redaction_is_recursive_for_nested_secret_and_oracle_values() -> None:
    secret = "sk-nested-secret-12345678"
    cleaned = redact_payload(
        {
            "outer": [
                {"authorization": f"Bearer {secret}"},
                {"details": {"oracle_text": "evaluator-only answer"}},
            ]
        }
    )
    serialized = json.dumps(cleaned, sort_keys=True)
    assert secret not in serialized
    assert "evaluator-only" not in serialized
    assert "[REDACTED]" in serialized


def test_projection_preserves_non_ok_run_status() -> None:
    cancelled = artifact(
        status=Status.CANCELLED,
        verdict=Verdict.NO_DECISION,
        rationale="operator cancelled run",
    )

    projection = project_comparison(cancelled)

    assert projection.status is Status.CANCELLED
    assert projection.verdict is Verdict.NO_DECISION


def test_projection_is_frozen_and_rejects_unknown_fields() -> None:
    projection = project_comparison(artifact())

    with pytest.raises(ValidationError):
        RunProjection.model_validate(projection.model_dump(mode="json") | {"corpus_text": "leaked"})
