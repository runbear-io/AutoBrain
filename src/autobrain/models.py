"""Strict domain contracts shared by AutoBrain components."""

import re
from datetime import datetime
from enum import StrEnum
from ipaddress import ip_address
from typing import Annotated
from urllib.parse import urlsplit, urlunsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

SourceId = Annotated[
    str, StringConstraints(min_length=3, max_length=512, pattern=r"^[a-z][a-z0-9_-]*:.+")
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def normalize_safe_source_url(value: str) -> str | None:
    """Return a canonical safe HTTP(S) URL, or omit an unsafe value."""
    if (
        not value
        or value != value.strip()
        or "\\" in value
        or any(character.isspace() or ord(character) < 0x20 for character in value)
    ):
        return None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except (UnicodeError, ValueError):
        return None
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None

    try:
        canonical_host = ip_address(hostname).compressed
    except ValueError:
        try:
            canonical_host = hostname.rstrip(".").encode("idna").decode("ascii").lower()
        except UnicodeError:
            return None
        if (
            not canonical_host
            or len(canonical_host) > 253
            or all(character.isdigit() or character == "." for character in canonical_host)
            or any(_HOST_LABEL.fullmatch(label) is None for label in canonical_host.split("."))
        ):
            return None

    authority = f"[{canonical_host}]" if ":" in canonical_host else canonical_host
    if port is not None:
        authority = f"{authority}:{port}"
    return urlunsplit(
        (
            parsed.scheme.lower(),
            authority,
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Status(StrEnum):
    OK = "OK"
    ENV_UNAVAILABLE = "ENV_UNAVAILABLE"
    MISSING_PROVIDER = "MISSING_PROVIDER"
    MCP_AUTH_UNAVAILABLE = "MCP_AUTH_UNAVAILABLE"
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    INSUFFICIENT_BENCHMARK = "INSUFFICIENT_BENCHMARK"
    LEAKAGE_DETECTED = "LEAKAGE_DETECTED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    NO_DECISION = "NO_DECISION"
    NO_RECOMMENDATION = "NO_RECOMMENDATION"


class QACapabilityStatus(StrEnum):
    BROWSER_UNAVAILABLE = "BROWSER_UNAVAILABLE"


class SourceKind(StrEnum):
    SLACK_MESSAGE = "SLACK_MESSAGE"
    SLACK_THREAD = "SLACK_THREAD"
    SLACK_CANVAS = "SLACK_CANVAS"
    SLACK_FILE = "SLACK_FILE"
    NOTION_PAGE = "NOTION_PAGE"
    NOTION_DATA_SOURCE = "NOTION_DATA_SOURCE"


class McpCapability(StrEnum):
    SEARCH = "SEARCH"
    FETCH = "FETCH"
    HISTORY = "HISTORY"
    THREADS = "THREADS"
    FILES = "FILES"


class ConnectionState(StrEnum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTED = "CONNECTED"
    EXPIRED = "EXPIRED"
    REAUTHORIZATION_REQUIRED = "REAUTHORIZATION_REQUIRED"


class CoverageCompleteness(StrEnum):
    EXHAUSTIVE = "EXHAUSTIVE"
    SEARCH_DISCOVERED = "SEARCH_DISCOVERED"
    UNKNOWN = "UNKNOWN"


class CandidateId(StrEnum):
    LLM_WIKI = "llm-wiki"
    MEM0 = "mem0"
    GBRAIN = "gbrain"


class Verdict(StrEnum):
    LLM_WIKI = "llm-wiki"
    MEM0 = "mem0"
    GBRAIN = "gbrain"
    NO_DECISION = "NO_DECISION"
    NO_RECOMMENDATION = "NO_RECOMMENDATION"


class CoverageRecord(StrictModel):
    source: SourceKind
    completeness: CoverageCompleteness
    discovered: int = Field(ge=0)
    fetched: int = Field(ge=0)
    skipped: int = Field(default=0, ge=0)
    truncated: int = Field(default=0, ge=0)
    denied: int = Field(default=0, ge=0)
    rate_limited: int = Field(default=0, ge=0)
    unsupported: int = Field(default=0, ge=0)
    crawl_provenance: dict[str, str] = Field(default_factory=dict)


class NormalizedDocument(StrictModel):
    source_id: SourceId
    source_kind: SourceKind
    canonical_url: str = Field(pattern=r"^https?://")
    title: str = Field(min_length=1)
    text: str
    content_hash: Sha256
    created_at: datetime | None = None
    updated_at: datetime | None = None
    container: str | None = None
    authors: list[str] = Field(default_factory=list)
    related_source_ids: list[SourceId] = Field(default_factory=list)
    source_references: list[str] = Field(default_factory=list)
    crawl_provenance: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class NormalizedThread(StrictModel):
    source_id: SourceId
    root: NormalizedDocument
    replies: list[NormalizedDocument]


class BenchmarkCase(StrictModel):
    case_id: str = Field(pattern=r"^case-[A-Za-z0-9_-]+$")
    question: str = Field(min_length=1)
    source_ids: list[SourceId] = Field(min_length=1)
    expected_claims: list[str] = Field(min_length=1)
    forbidden_contradictions: list[str] = Field(default_factory=list)
    generated: bool = False


class CandidateQuery(StrictModel):
    case_id: str = Field(pattern=r"^case-[A-Za-z0-9_-]+$")
    question: str = Field(min_length=1)


class Holdout(StrictModel):
    case_id: str = Field(pattern=r"^case-[A-Za-z0-9_-]+$")
    source_ids: list[SourceId] = Field(min_length=1)
    reference_text: str = Field(min_length=1)
    reply_ids: list[SourceId] = Field(default_factory=list)


class UsageCost(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    usd: float = Field(ge=0)


class CostStatus(StrEnum):
    COMPLETE = "COST_COMPLETE"
    INCOMPLETE = "COST_INCOMPLETE"
    UNAVAILABLE = "COST_UNAVAILABLE"


class QualityComponents(StrictModel):
    required_claim_coverage: float = Field(ge=0, le=45)
    cited_source_support: float = Field(ge=0, le=25)
    contradiction_safety: float = Field(ge=0, le=20)
    supplementary_style: float = Field(ge=0, le=10)

    @property
    def total(self) -> float:
        return round(
            self.required_claim_coverage
            + self.cited_source_support
            + self.contradiction_safety
            + self.supplementary_style,
            4,
        )


class CaseEvaluation(StrictModel):
    candidate: CandidateId
    case_id: str
    status: Status
    score: float = Field(ge=0, le=100)
    components: QualityComponents
    required_claims: int = Field(ge=0)
    covered_claims: int = Field(ge=0)
    cited_claims: int = Field(ge=0)
    forbidden_matches: int = Field(ge=0)
    source_ids: list[SourceId] = Field(default_factory=list)
    generated: bool = False
    reference_confidence: float = Field(ge=0, le=1, default=1.0)
    failure_detail: str = ""
    latency_ms: int = Field(ge=0, default=0)

    @property
    def required_claim_coverage(self) -> float:
        return self.components.required_claim_coverage

    @property
    def cited_source_support(self) -> float:
        return self.components.cited_source_support

    @property
    def contradiction_safety(self) -> float:
        return self.components.contradiction_safety

    @property
    def supplementary_style(self) -> float:
        return self.components.supplementary_style


class CandidateCaseEvidence(StrictModel):
    candidate: CandidateId
    case_id: str
    status: Status
    score: float = Field(ge=0, le=100)
    source_ids: list[SourceId] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    cited_claims: int = Field(ge=0)
    required_claims: int = Field(ge=0)
    failure_detail: str = ""

    @field_validator("source_urls")
    @classmethod
    def source_urls_are_safe(cls, values: list[str]) -> list[str]:
        return [
            normalized
            for value in values
            if (normalized := normalize_safe_source_url(value)) is not None
        ]


class CandidateEvaluation(StrictModel):
    candidate: CandidateId
    status: Status
    scored_cases: int = Field(ge=0)
    answered_cases: int = Field(ge=0)
    quality_score: float = Field(ge=0, le=100)
    answer_success_rate: float = Field(ge=0, le=1)
    source_support_rate: float = Field(ge=0, le=1)
    contradiction_count: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_cost_usd: float | None = Field(default=None, ge=0)
    cost_status: CostStatus
    ingest_wall_time_ms: int = Field(ge=0, default=0)
    query_wall_time_ms: int = Field(ge=0, default=0)
    query_p50_ms: float | None = Field(default=None, ge=0)
    query_p95_ms: float | None = Field(default=None, ge=0)
    workspace_bytes: int | None = Field(default=None, ge=0)
    operating_burden: float | None = Field(default=None, ge=0)
    valid_pin: bool = False
    corpus_hash: Sha256 | None = None
    direct_leakage: bool = False
    generated_cases: int = Field(ge=0, default=0)
    partial_failures: int = Field(ge=0, default=0)
    eligibility_reasons: list[str] = Field(default_factory=list)
    eligible_override: bool | None = None

    @model_validator(mode="after")
    def complete_cost_has_a_value(self) -> "CandidateEvaluation":
        if self.cost_status is CostStatus.COMPLETE and self.total_cost_usd is None:
            raise ValueError("COST_COMPLETE requires total_cost_usd")
        if self.cost_status is not CostStatus.COMPLETE and self.total_cost_usd is not None:
            raise ValueError("incomplete or unavailable cost must not expose a total")
        return self


class DecisionResult(StrictModel):
    status: Status
    verdict: Verdict
    rationale: str
    eligible_candidates: list[CandidateId] = Field(default_factory=list)
    considered_candidates: list[CandidateId] = Field(default_factory=list)
    quality_floor: float = 60.0
    close_quality_epsilon: float = 5.0


class ComparisonArtifact(StrictModel):
    schema_version: int = Field(default=1, ge=1)
    run_id: str = Field(min_length=1)
    status: Status = Status.OK
    corpus_hash: Sha256
    benchmark_hash: Sha256
    verdict: Verdict
    decision: DecisionResult
    coverage: list[CoverageRecord] = Field(default_factory=list)
    candidates: list[CandidateEvaluation] = Field(min_length=1)
    evidence: list[CandidateCaseEvidence] = Field(default_factory=list)
    methodology: dict[str, str] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    price_sheet_version: str | None = None
    warnings: list[str] = Field(default_factory=list)


class CandidateObservation(StrictModel):
    candidate: CandidateId
    status: Status
    case_id: str = ""
    answer: str = ""
    source_ids: list[SourceId] = Field(default_factory=list)
    usage: UsageCost | None = None
    latency_ms: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class QualityResult(StrictModel):
    score: float = Field(ge=0, le=1)
    supported_claims: int = Field(ge=0)
    total_claims: int = Field(ge=0)

    @model_validator(mode="after")
    def supported_not_greater_than_total(self) -> "QualityResult":
        if self.supported_claims > self.total_claims:
            raise ValueError("supported_claims cannot exceed total_claims")
        return self


class CheckResult(StrictModel):
    name: str
    status: Status
    detail: str
    version: str | None = None
    path: str | None = None


class DoctorPaths(StrictModel):
    root: str
    runs: str
    tools: str
    cache: str


class CandidatePin(StrictModel):
    id: CandidateId
    distribution: str = Field(min_length=1)
    version: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    repository: str = Field(pattern=r"^https://github\.com/")
    license: str = Field(min_length=1)


class CandidatePins(StrictModel):
    schema_version: int = Field(ge=1)
    candidates: list[CandidatePin] = Field(min_length=1)

    @model_validator(mode="after")
    def candidate_ids_are_unique(self) -> "CandidatePins":
        ids = [candidate.id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate candidate IDs are not allowed")
        return self
