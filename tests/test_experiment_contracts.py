from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from autobrain.experiment_contracts import (
    ExperimentLifecycle,
    ExperimentLifecycleStatus,
    ExperimentReadiness,
    ExperimentReadinessState,
    ExperimentRequest,
    ReadinessCheckState,
    RetrievalMetrics,
    StableExperimentError,
    StableExperimentErrorCode,
)
from autobrain.models import CandidateId, CorpusIdentity, CostStatus, ExperimentIdentity


HASH = "a" * 64


def test_experiment_request_and_identity_are_strict_and_immutable() -> None:
    identity = ExperimentIdentity(
        corpus=CorpusIdentity(sha256=HASH, document_count=2),
        benchmark_sha256="b" * 64,
        protocol="retrieval-v1",
        evaluator="retrieval",
        provider="codex",
        model="gpt-fixture",
        configuration_hash="c" * 64,
        code_version="d" * 40,
    )
    request = ExperimentRequest(
        schema_version=1,
        experiment_id="exp-1",
        identity=identity,
        candidates=[CandidateId.GBRAIN, CandidateId.MEM0],
        evaluation_mode="retrieval_only",
    )
    with pytest.raises(ValidationError):
        request.candidates = []  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ExperimentRequest.model_validate({**request.model_dump(mode="json"), "secret": "nope"})


def test_readiness_is_explicit_and_cannot_claim_ready_with_blockers() -> None:
    blocked = ExperimentReadiness(
        state=ExperimentReadinessState.BLOCKED,
        checks={"source": ReadinessCheckState.READY, "embedding": ReadinessCheckState.NOT_CONFIGURED},
        blockers=["SEMANTIC_EMBEDDING_REQUIRED"],
    )
    assert not blocked.ready
    with pytest.raises(ValidationError, match="blockers"):
        ExperimentReadiness(
            state=ExperimentReadinessState.READY,
            checks={"source": ReadinessCheckState.READY},
            blockers=["unexpected"],
        )


def test_retrieval_metrics_reject_malformed_or_inconsistent_values() -> None:
    metrics = RetrievalMetrics(
        relevant_retrieved=2,
        retrieved=4,
        relevant_available=3,
        missing_evidence=1,
        noise=2,
        latency_ms=12.5,
        cost_status=CostStatus.UNAVAILABLE,
    )
    assert abs(metrics.recall - (2 / 3)) < 1e-12
    assert abs(metrics.precision - 0.5) < 1e-12
    with pytest.raises(ValidationError, match="noise"):
        RetrievalMetrics(
            relevant_retrieved=2,
            retrieved=4,
            relevant_available=3,
            missing_evidence=1,
            noise=1,
            latency_ms=1,
            cost_status=CostStatus.UNAVAILABLE,
        )


def test_lifecycle_transitions_are_typed_and_stable() -> None:
    now = datetime(2026, 8, 31, tzinfo=UTC)
    lifecycle = ExperimentLifecycle(experiment_id="exp-1", status=ExperimentLifecycleStatus.CREATED)
    validated = lifecycle.transition(ExperimentLifecycleStatus.VALIDATING, now=now)
    ready = validated.transition(ExperimentLifecycleStatus.READY, now=now)
    started = ready.transition(ExperimentLifecycleStatus.RUNNING, now=now)
    assert started.status is ExperimentLifecycleStatus.RUNNING
    with pytest.raises(StableExperimentError) as error:
        started.transition(ExperimentLifecycleStatus.CREATED, now=now)
    assert error.value.code is StableExperimentErrorCode.INVALID_TRANSITION
    assert str(error.value) == "INVALID_TRANSITION: RUNNING -> CREATED"
