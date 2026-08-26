"""Versioned redacted projection of a comparison artifact.

This is the only shape allowed to cross the local HTTP boundary in
`autobrain.local_server`. It is deliberately narrower than
`ComparisonArtifact`: per-case evidence, expected claims, holdout references
and raw source identifiers are dropped entirely rather than redacted, so a
browser client can render a run summary without ever receiving corpus content.

Versioning is explicit and pinned. A client that understands
`PROJECTION_SCHEMA_VERSION` can rely on this payload; any other version fails
validation instead of being silently coerced.
"""

from __future__ import annotations

import json
from typing import Final, Literal, cast

from pydantic import Field

from autobrain.models import (
    CandidateId,
    ComparisonArtifact,
    CostStatus,
    Sha256,
    Status,
    StrictModel,
    Verdict,
)
from autobrain.report import redact_payload, redact_text

PROJECTION_SCHEMA_VERSION: Final = 1


class CandidateProjection(StrictModel):
    """Scoreboard row for a single candidate. Carries metrics, never answers."""

    candidate: CandidateId
    status: Status
    quality_score: float = Field(ge=0, le=100)
    answer_success_rate: float = Field(ge=0, le=1)
    source_support_rate: float = Field(ge=0, le=1)
    contradiction_count: int = Field(ge=0)
    scored_cases: int = Field(ge=0)
    answered_cases: int = Field(ge=0)
    cost_status: CostStatus
    total_cost_usd: float | None = Field(default=None, ge=0)
    query_p50_ms: float | None = Field(default=None, ge=0)
    query_p95_ms: float | None = Field(default=None, ge=0)
    operating_burden: float | None = Field(default=None, ge=0)


class RunProjection(StrictModel):
    """Redacted, versioned run summary safe to serve over the local boundary."""

    schema_version: Literal[1]
    run_id: str = Field(min_length=1)
    status: Status
    verdict: Verdict
    rationale: str
    corpus_hash: Sha256
    benchmark_hash: Sha256
    candidates: list[CandidateProjection] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)


def project_comparison(artifact: ComparisonArtifact) -> RunProjection:
    """Build the redacted projection for `artifact`.

    Redaction reuses the report pipeline so the boundary and the on-disk report
    can never drift into disagreeing about what counts as sensitive.
    """
    candidates = [
        CandidateProjection(
            candidate=evaluation.candidate,
            status=evaluation.status,
            quality_score=evaluation.quality_score,
            answer_success_rate=evaluation.answer_success_rate,
            source_support_rate=evaluation.source_support_rate,
            contradiction_count=evaluation.contradiction_count,
            scored_cases=evaluation.scored_cases,
            answered_cases=evaluation.answered_cases,
            cost_status=evaluation.cost_status,
            total_cost_usd=evaluation.total_cost_usd,
            query_p50_ms=evaluation.query_p50_ms,
            query_p95_ms=evaluation.query_p95_ms,
            operating_burden=evaluation.operating_burden,
        )
        for evaluation in artifact.candidates
    ]
    projection = RunProjection(
        schema_version=PROJECTION_SCHEMA_VERSION,
        run_id=artifact.run_id,
        status=artifact.status,
        verdict=artifact.verdict,
        rationale=redact_text(artifact.decision.rationale),
        corpus_hash=artifact.corpus_hash,
        benchmark_hash=artifact.benchmark_hash,
        candidates=candidates,
        warnings=[redact_text(warning) for warning in artifact.warnings],
    )
    payload = cast(dict[str, object], redact_payload(projection.model_dump(mode="json")))
    return RunProjection.model_validate_json(json.dumps(payload), strict=True)
