"""Canonical typed contracts for the local Web experiment boundary.

The Web serializes these models; it does not reimplement evaluation or lifecycle
policy. Contracts are strict and immutable so a completed experiment can be
reopened and compared without identity drift.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, cast

from pydantic import Field, field_validator, model_validator

from autobrain.models import CandidateId, CostStatus, ExperimentIdentity, StrictModel


class ExperimentLifecycleStatus(StrEnum):
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ExperimentReadinessState(StrEnum):
    UNKNOWN = "UNKNOWN"
    READY = "READY"
    BLOCKED = "BLOCKED"


class ReadinessCheckState(StrEnum):
    READY = "READY"
    CONFIGURED = "CONFIGURED"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    UNSUPPORTED = "UNSUPPORTED"
    BLOCKED = "BLOCKED"


class StableExperimentErrorCode(StrEnum):
    INVALID_TRANSITION = "INVALID_TRANSITION"
    INVALID_REQUEST = "INVALID_REQUEST"
    NOT_READY = "NOT_READY"
    NOT_FOUND = "NOT_FOUND"
    RUN_FAILED = "RUN_FAILED"
    CANCELLED = "CANCELLED"


class StableExperimentError(ValueError):
    """Machine-readable error whose code is safe to expose at the Web boundary."""

    def __init__(self, code: StableExperimentErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


class ExperimentReadiness(StrictModel):
    state: ExperimentReadinessState
    checks: dict[str, ReadinessCheckState] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.state is ExperimentReadinessState.READY

    @model_validator(mode="after")
    def state_matches_blockers(self) -> ExperimentReadiness:
        if self.state is ExperimentReadinessState.READY and self.blockers:
            raise ValueError("ready readiness cannot contain blockers")
        if self.state is ExperimentReadinessState.BLOCKED and not self.blockers:
            raise ValueError("blocked readiness requires blockers")
        return self


class ExperimentRequest(StrictModel):
    schema_version: Literal[1]
    experiment_id: str = Field(min_length=1, max_length=128)
    identity: ExperimentIdentity
    candidates: list[CandidateId] = Field(min_length=1)
    evaluation_mode: Literal["retrieval_only", "answer_aware"] = "retrieval_only"

    @field_validator("candidates", mode="before")
    @classmethod
    def parse_candidate_ids(cls, value: object) -> object:
        if isinstance(value, list):
            parsed: list[CandidateId | object] = []
            for item in cast(list[Any], value):
                parsed.append(CandidateId(item) if isinstance(item, str) else item)
            return parsed
        return value

    @model_validator(mode="after")
    def candidates_are_unique(self) -> ExperimentRequest:
        if len(self.candidates) != len(set(self.candidates)):
            raise ValueError("candidates must be unique")
        return self


class RetrievalMetrics(StrictModel):
    """Retrieval-only metrics; answer generation is deliberately absent."""

    relevant_retrieved: int = Field(ge=0)
    retrieved: int = Field(ge=0)
    relevant_available: int = Field(ge=0)
    missing_evidence: int = Field(ge=0)
    noise: int = Field(ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    cost_status: CostStatus
    freshness_score: float | None = Field(default=None, ge=0, le=1)

    @property
    def recall(self) -> float:
        return self.relevant_retrieved / self.relevant_available if self.relevant_available else 0.0

    @property
    def precision(self) -> float:
        return self.relevant_retrieved / self.retrieved if self.retrieved else 0.0

    @model_validator(mode="after")
    def counts_are_consistent(self) -> RetrievalMetrics:
        if self.relevant_retrieved > self.retrieved:
            raise ValueError("relevant_retrieved cannot exceed retrieved")
        if self.relevant_retrieved > self.relevant_available:
            raise ValueError("relevant_retrieved cannot exceed relevant_available")
        if self.missing_evidence != self.relevant_available - self.relevant_retrieved:
            raise ValueError("missing_evidence must equal relevant_available - relevant_retrieved")
        if self.noise != self.retrieved - self.relevant_retrieved:
            raise ValueError("noise must equal retrieved - relevant_retrieved")
        return self


class RetrievalResult(StrictModel):
    """One candidate's retrieval output and reconciled metrics."""

    candidate: CandidateId
    case_id: str = Field(min_length=1)
    retrieved_source_ids: list[str] = Field(default_factory=list)
    metrics: RetrievalMetrics


_ALLOWED_TRANSITIONS: dict[ExperimentLifecycleStatus, frozenset[ExperimentLifecycleStatus]] = {
    ExperimentLifecycleStatus.CREATED: frozenset(
        {ExperimentLifecycleStatus.VALIDATING, ExperimentLifecycleStatus.CANCELLED}
    ),
    ExperimentLifecycleStatus.VALIDATING: frozenset(
        {
            ExperimentLifecycleStatus.READY,
            ExperimentLifecycleStatus.FAILED,
            ExperimentLifecycleStatus.CANCELLED,
        }
    ),
    ExperimentLifecycleStatus.READY: frozenset(
        {ExperimentLifecycleStatus.RUNNING, ExperimentLifecycleStatus.CANCELLED}
    ),
    ExperimentLifecycleStatus.RUNNING: frozenset(
        {
            ExperimentLifecycleStatus.SUCCEEDED,
            ExperimentLifecycleStatus.FAILED,
            ExperimentLifecycleStatus.CANCELLED,
        }
    ),
    ExperimentLifecycleStatus.SUCCEEDED: frozenset(),
    ExperimentLifecycleStatus.FAILED: frozenset(),
    ExperimentLifecycleStatus.CANCELLED: frozenset(),
}


class ExperimentLifecycle(StrictModel):
    experiment_id: str = Field(min_length=1, max_length=128)
    status: ExperimentLifecycleStatus
    updated_at: datetime | None = None

    def transition(
        self, status: ExperimentLifecycleStatus, *, now: datetime | None = None
    ) -> ExperimentLifecycle:
        if status not in _ALLOWED_TRANSITIONS[self.status]:
            raise StableExperimentError(
                StableExperimentErrorCode.INVALID_TRANSITION,
                f"{self.status.value} -> {status.value}",
            )
        timestamp = now or datetime.now(UTC)
        return self.model_copy(update={"status": status, "updated_at": timestamp})
