"""Strict domain contracts shared by AutoBrain components."""

import re
from datetime import datetime
from enum import StrEnum
from ipaddress import ip_address
from typing import Annotated, Literal
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
    UNSUPPORTED = "UNSUPPORTED"
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
    CONFLUENCE_PAGE = "CONFLUENCE_PAGE"
    GOOGLE_DRIVE_FILE = "GOOGLE_DRIVE_FILE"


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


class EmbeddingQuality(StrEnum):
    SEMANTIC = "semantic"
    SMOKE_ONLY = "smoke_only"


class NativeMode(StrEnum):
    KEYWORD_ONLY = "keyword_only"
    SEMANTIC = "semantic"
    SMOKE = "smoke"


class CapabilityClass(StrEnum):
    RETRIEVAL_ONLY = "retrieval_only"
    RETRIEVAL_AND_ANSWER = "retrieval_and_answer"


class EvidenceStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    SMOKE = "smoke"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


class UsageSource(StrEnum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


class SourceMutability(StrEnum):
    FROZEN_EXPORT = "frozen_export"
    LIVE_MCP_CAPTURED = "live_mcp_captured"


class LatencySpanKind(StrEnum):
    END_TO_END = "end_to_end"
    PROVIDER_EXECUTION = "provider_execution"
    CANDIDATE_INGEST = "candidate_ingest"
    CANDIDATE_QUERY = "candidate_query"
    PROCESS_STARTUP = "process_startup"
    UNCLASSIFIED_OVERHEAD = "unclassified_overhead"


class ChatProvenance(StrictModel):
    provider: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)
    cli_version: str | None = Field(default=None, min_length=1)
    auth_kind: str | None = Field(default=None, min_length=1)


class EmbeddingProvenance(StrictModel):
    backend: str | None = Field(default=None, min_length=1)
    quality: EmbeddingQuality | None = None


class SourceProvenance(StrictModel):
    source: str = Field(min_length=1)
    mutability: SourceMutability


class BackendIdentity(StrictModel):
    name: str = Field(min_length=1)
    version: str | None = Field(default=None, min_length=1)
    commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")


class CorpusIdentity(StrictModel):
    sha256: Sha256
    document_count: int = Field(ge=0)


class NativeCandidateResult(StrictModel):
    """Typed, diagnostic contract for a candidate's native execution result."""

    candidate: CandidateId
    mode: NativeMode
    backend: BackendIdentity
    capability: CapabilityClass
    evidence_status: EvidenceStatus
    corpus: CorpusIdentity | None = None
    recommendation_eligible: bool = False
    eligibility_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contract(self) -> "NativeCandidateResult":
        expected_backend = {
            CandidateId.GBRAIN: "gbrain",
            CandidateId.LLM_WIKI: "llm-wiki-compiler",
            CandidateId.MEM0: "mem0ai",
        }[self.candidate]
        if self.backend.name != expected_backend:
            raise ValueError(
                f"backend {self.backend.name!r} is not allowed for {self.candidate.value}"
            )
        if self.candidate is not CandidateId.GBRAIN and self.mode is not NativeMode.SEMANTIC:
            raise ValueError(f"{self.candidate.value} only supports semantic mode")
        if self.mode is NativeMode.KEYWORD_ONLY:
            if self.capability is not CapabilityClass.RETRIEVAL_ONLY:
                raise ValueError("keyword_only mode must use retrieval_only capability")
            if self.recommendation_eligible:
                raise ValueError("keyword_only mode cannot be recommendation eligible")
        if self.evidence_status is not EvidenceStatus.COMPLETE and self.recommendation_eligible:
            raise ValueError("incomplete native evidence cannot be recommendation eligible")
        if self.capability is CapabilityClass.RETRIEVAL_ONLY and self.recommendation_eligible:
            raise ValueError("retrieval_only capability cannot be recommendation eligible")
        if self.recommendation_eligible and self.eligibility_reasons:
            raise ValueError("eligible native result must not have eligibility_reasons")
        if not self.recommendation_eligible and not self.eligibility_reasons:
            raise ValueError("ineligible native result requires eligibility_reasons")
        return self


class LatencySpan(StrictModel):
    name: LatencySpanKind
    duration_ms: float | None = Field(default=None, ge=0)
    candidate: CandidateId | None = None

    @model_validator(mode="after")
    def unavailable_is_not_zero(self) -> "LatencySpan":
        if self.duration_ms == 0:
            raise ValueError("zero-duration latency spans are unavailable, not measured")
        return self


class IntegrationReuse(StrEnum):
    DIRECT_REUSE = "direct_reuse"
    PROTOCOL_REUSE = "protocol_reuse"
    THIN_ADAPTER = "thin_adapter"
    GATED = "gated"


class IntegrationStatus(StrEnum):
    CURRENT = "current"
    GATED = "gated"


class IntegrationProvenance(StrictModel):
    id: str = Field(min_length=1)
    provider: str | None = Field(default=None, min_length=1)
    source: str | None = Field(default=None, min_length=1)
    backend: str = Field(min_length=1)
    version: str | None = Field(default=None, min_length=1)
    license: str | None = Field(default=None, min_length=1)
    auth_kind: str = Field(min_length=1)
    capabilities: tuple[str, ...]
    usage_provenance: str = Field(min_length=1)
    reuse: IntegrationReuse
    status: IntegrationStatus
    evidence: str = Field(min_length=1)


class DesignPartnerGateEvidence(StrictModel):
    """Machine-readable typed refusal for a design-partner integration gate.

    Every boolean field defaults to ``False`` because the gate is fail-closed:
    absent evidence is recorded as ``False``, never inferred or guessed.  The
    model validator enforces that a closed gate cannot be recommendation
    eligible.
    """

    integration_id: str = Field(min_length=1)
    design_partner_evidence: bool = False
    verified_license: bool = False
    public_api: bool = False
    runtime_proof: bool = False
    acl_proof: bool = False
    resource_proof: bool = False
    network_proof: bool = False
    teardown_proof: bool = False
    recommendation_eligible: bool = False
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def closed_gate_is_not_eligible(self) -> "DesignPartnerGateEvidence":
        if not self.recommendation_eligible:
            return self
        if not (
            self.design_partner_evidence
            and self.verified_license
            and self.public_api
            and self.runtime_proof
            and self.acl_proof
            and self.resource_proof
            and self.network_proof
            and self.teardown_proof
        ):
            raise ValueError("recommendation_eligible requires all evidence fields to be true")
        return self


class BenchmarkProvenance(StrictModel):
    chat: ChatProvenance = Field(default_factory=ChatProvenance)
    embedding: EmbeddingProvenance = Field(default_factory=EmbeddingProvenance)
    usage_source: UsageSource = UsageSource.UNAVAILABLE
    sources: list[SourceProvenance] = Field(default_factory=list)
    latency_spans: list[LatencySpan] = Field(default_factory=list)
    integrations: list["IntegrationProvenance"] = Field(default_factory=list)


class RunHashes(StrictModel):
    corpus_sha256: Sha256 | None = None
    benchmark_sha256: Sha256 | None = None


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, frozen=True)

    schema_version: Literal[1, 2]
    run_id: str = Field(min_length=1)
    created_at: datetime | None = None
    status: Status | None = None
    verdict: str | None = None
    hashes: RunHashes | None = None
    provenance: BenchmarkProvenance
    decision: "DecisionResult | None" = None
    evaluations: list["CandidateEvaluation"] | None = None
    candidates: list[dict[str, object]] | None = None


class QualityComponents(StrictModel):
    retrieval_recall: float = Field(ge=0, le=100)
    required_claim_coverage: float = Field(default=0, ge=0, le=45)
    cited_source_support: float = Field(default=0, ge=0, le=25)
    contradiction_safety: float = Field(default=0, ge=0, le=20)
    supplementary_style: float = Field(default=0, ge=0, le=10)

    @property
    def total(self) -> float:
        return round(self.retrieval_recall, 4)


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
    def retrieval_recall(self) -> float:
        return self.components.retrieval_recall

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
    usage_source: UsageSource = UsageSource.MEASURED
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
    native_result: NativeCandidateResult | None = None

    @model_validator(mode="after")
    def complete_cost_has_a_value(self) -> "CandidateEvaluation":
        if self.cost_status is CostStatus.COMPLETE and self.total_cost_usd is None:
            raise ValueError("COST_COMPLETE requires total_cost_usd")
        if self.cost_status is not CostStatus.COMPLETE and self.total_cost_usd is not None:
            raise ValueError("incomplete or unavailable cost must not expose a total")
        if self.native_result is not None:
            if self.native_result.candidate is not self.candidate:
                raise ValueError("native_result candidate must match evaluation candidate")
            if (
                self.corpus_hash is not None
                and self.native_result.corpus is not None
                and self.native_result.corpus.sha256 != self.corpus_hash
            ):
                raise ValueError("native_result corpus must match evaluation corpus")
        return self


class DecisionResult(StrictModel):
    status: Status
    verdict: Verdict
    rationale: str
    eligible_candidates: list[CandidateId] = Field(default_factory=list)
    ineligible_candidates: dict[CandidateId, list[str]] = Field(default_factory=dict)
    considered_candidates: list[CandidateId] = Field(default_factory=list)
    quality_floor: float = 60.0
    close_quality_epsilon: float = 5.0
    tie_break_metric: str = "candidate_query_p95_ms"


class ComparisonArtifact(StrictModel):
    schema_version: Literal[2]
    run_id: str = Field(min_length=1)
    status: Status = Status.OK
    corpus_hash: Sha256
    benchmark_hash: Sha256
    verdict: Verdict
    decision: DecisionResult
    coverage: list[CoverageRecord] = Field(default_factory=list)
    candidates: list[CandidateEvaluation] = Field(min_length=1)
    evidence: list[CandidateCaseEvidence] = Field(default_factory=list)
    provenance: BenchmarkProvenance
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
