"""Strict, data-only fake-MCP fixture runtime for installed CLI QA."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, field_validator, model_validator

from autobrain.cancellation import RunCancellation
from autobrain.models import (
    CandidateId,
    CandidateObservation,
    CostStatus,
    SourceKind,
    Status,
    StrictModel,
    normalize_safe_source_url,
)
from autobrain.orchestration import CandidateContext, CandidateOutcome, ConnectorSnapshot
from autobrain.secrets import contains_secret

_FORBIDDEN = re.compile(
    r"(?i)\b(?:oracle|holdout|reference[\s_-]*answer|expected[\s_-]*claim|raw[\s_-]*reply)\b"
)


class FixtureValidationError(ValueError):
    """A local fixture failed its strict, non-executable contract."""


class FixtureFaultCode(StrEnum):
    """Closed vocabulary for describing injected fixture defects.

    Faults are metadata only. They are never interpreted as instructions or
    executed by the fixture runtime.
    """

    CONTENT_HASH_MISMATCH = "content_hash_mismatch"
    DUPLICATE_SOURCE_ID = "duplicate_source_id"
    MISSING_DOCUMENT = "missing_document"
    SECRET_LIKE_CONTENT = "secret_like_content"
    UNSAFE_URL = "unsafe_url"


class FixtureFault(StrictModel):
    code: FixtureFaultCode
    target: str | None = Field(default=None, min_length=1, max_length=512)

    @field_validator("code", mode="before")
    @classmethod
    def parse_code(cls, value: object) -> FixtureFaultCode:
        return value if isinstance(value, FixtureFaultCode) else FixtureFaultCode(str(value))

    detail: str | None = Field(default=None, min_length=1, max_length=4096)


class FixtureDocument(StrictModel):
    provider: Literal["slack", "notion"]
    source_id: str = Field(pattern=r"^(?:slack|notion):[A-Za-z0-9:_-]+$")
    source_kind: Literal["SLACK_MESSAGE", "NOTION_PAGE"]
    canonical_url: str
    title: str = Field(min_length=1, max_length=4096)
    text: str = Field(max_length=1_000_000)
    question: str = Field(min_length=1, max_length=4096)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity_and_url(self) -> FixtureDocument:
        if not self.source_id.startswith(f"{self.provider}:"):
            raise ValueError("source_id provider prefix does not match provider")
        expected_kind = (
            SourceKind.SLACK_MESSAGE if self.provider == "slack" else SourceKind.NOTION_PAGE
        )
        if self.source_kind != expected_kind.value:
            raise ValueError("source_kind does not match provider")
        if normalize_safe_source_url(self.canonical_url) != self.canonical_url:
            raise ValueError("canonical_url must be a safe canonical HTTP(S) URL")
        if hashlib.sha256(self.text.encode()).hexdigest() != self.content_hash:
            raise ValueError("content_hash does not match text")
        return self


class FixtureCandidateSpec(StrictModel):
    id: Literal["llm-wiki", "mem0", "gbrain"]
    score: float = Field(ge=0, le=100)
    cost_usd: float = Field(ge=0)


class FixtureSpec(StrictModel):
    schema_version: Literal[1]
    fixture_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    documents: list[FixtureDocument] = Field(min_length=20)
    candidates: list[FixtureCandidateSpec] = Field(min_length=3, max_length=3)
    faults: list[FixtureFault] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def validate_fixture(self) -> FixtureSpec:
        source_ids = [document.source_id for document in self.documents]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("fixture source IDs must be unique")
        candidate_ids = [candidate.id for candidate in self.candidates]
        if candidate_ids != ["llm-wiki", "mem0", "gbrain"]:
            raise ValueError("fixture candidates must be exactly llm-wiki, mem0, gbrain")
        serialized = self.model_dump(mode="json")
        claimed = serialized["fixture_sha256"]
        if hashlib.sha256(_fixture_hash_payload(serialized)).hexdigest() != claimed:
            raise ValueError("fixture_sha256 does not match canonical fixture content")
        for value in _walk_strings(serialized):
            if contains_secret(value) or _FORBIDDEN.search(value):
                raise ValueError("fixture contains secret or evaluator-only marker")
        return self


def _walk_strings(value: object) -> Sequence[str]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return tuple(item for child in mapping.values() for item in _walk_strings(child))
    if isinstance(value, list):
        children = cast(list[object], value)
        return tuple(item for child in children for item in _walk_strings(child))
    return ()


def _fixture_hash_payload(payload: Mapping[str, object]) -> bytes:
    without_hash = dict(payload)
    without_hash.pop("fixture_sha256", None)
    if without_hash.get("faults") == []:
        without_hash.pop("faults")
    return json.dumps(without_hash, sort_keys=True, separators=(",", ":")).encode()


def fixture_json_bytes(spec: FixtureSpec) -> bytes:
    """Serialize a fixture with stable formatting for reproducible artifacts."""
    return (json.dumps(spec.model_dump(mode="json"), sort_keys=True, indent=2) + "\n").encode()


def build_fixture(
    *,
    seed: int,
    fixture_id: str | None = None,
    faults: Sequence[FixtureFaultCode | FixtureFault] = (),
) -> FixtureSpec:
    """Build a deterministic, secret-free schema-v1 fixture."""
    identifier = fixture_id or f"generated-fixture-{seed}"
    documents: list[dict[str, object]] = []
    for index in range(24):
        provider = "slack" if index % 2 == 0 else "notion"
        text = f"Project Atlas fact {index} has stable value {index}."
        documents.append(
            {
                "provider": provider,
                "source_id": f"{provider}:fixture:{index}",
                "source_kind": "SLACK_MESSAGE" if provider == "slack" else "NOTION_PAGE",
                "canonical_url": f"https://fixture.example.test/source/{index}",
                "title": f"Fixture fact {index}",
                "text": text,
                "question": f"What is Project Atlas fact {index}?",
                "content_hash": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    normalized_faults = [
        item if isinstance(item, FixtureFault) else FixtureFault(code=item) for item in faults
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "fixture_id": identifier,
        "documents": documents,
        "candidates": [
            {"id": "llm-wiki", "score": 92.0, "cost_usd": 1.0},
            {"id": "mem0", "score": 88.0, "cost_usd": 1.0},
            {"id": "gbrain", "score": 86.0, "cost_usd": 1.0},
        ],
        "faults": [item.model_dump(mode="python") for item in normalized_faults],
    }
    payload["fixture_sha256"] = hashlib.sha256(_fixture_hash_payload(payload)).hexdigest()
    return FixtureSpec.model_validate(payload)


def generate_fixture(
    *,
    seed: int,
    fixture_id: str | None = None,
    faults: Sequence[FixtureFaultCode | FixtureFault] = (),
) -> FixtureSpec:
    """Public generator alias for callers that prefer generator terminology."""
    return build_fixture(seed=seed, fixture_id=fixture_id, faults=faults)


def write_fixture(
    path: Path,
    *,
    seed: int,
    fixture_id: str | None = None,
    faults: Sequence[FixtureFaultCode | FixtureFault] = (),
) -> Path:
    """Write one deterministic fixture and return its path."""
    spec = build_fixture(seed=seed, fixture_id=fixture_id, faults=faults)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fixture_json_bytes(spec))
    return path


def load_fixture(path: Path) -> FixtureSpec:
    """Load only an absolute, regular local JSON fixture."""
    if not path.is_absolute():
        raise FixtureValidationError("test fixture path must be absolute")
    if ".." in path.parts:
        raise FixtureValidationError("test fixture path must not contain parent traversal")
    if path.is_symlink() or not path.is_file():
        raise FixtureValidationError("test fixture path must be a regular local file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return FixtureSpec.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise FixtureValidationError(f"invalid test fixture: {path}") from error


class FixtureConnector:
    def __init__(self, provider: str, documents: Sequence[FixtureDocument]) -> None:
        self.provider = provider
        self.documents = tuple(
            {
                "provider": item.provider,
                "source_id": item.source_id,
                "source_kind": item.source_kind,
                "canonical_url": item.canonical_url,
                "title": item.title,
                "text": item.text,
                "question": item.question,
                "evidence_reply": item.text,
            }
            for item in documents
            if item.provider == provider
        )

    def probe(self, cancellation: RunCancellation | None = None) -> dict[str, list[str]]:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        return {"advertised": ["search", "fetch"], "allowed": ["search", "fetch"]}

    def crawl(self, *, cancellation: RunCancellation | None = None) -> ConnectorSnapshot:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        return ConnectorSnapshot(
            provider=self.provider,
            documents=self.documents,
            coverage={
                "completeness": "SEARCH_DISCOVERED",
                "discovered": len(self.documents),
                "fetched": len(self.documents),
                "test_mode": True,
            },
        )


class FixtureCandidate:
    def __init__(self, candidate_id: str, score: float, cost_usd: float, fixture_id: str) -> None:
        self.candidate_id = candidate_id
        self.score = score
        self.cost_usd = cost_usd
        self.fixture_id = fixture_id

    def run(self, context: CandidateContext) -> CandidateOutcome:
        source_urls = sorted(
            {
                str(document["canonical_url"])
                for document in context.documents
                if isinstance(document.get("canonical_url"), str)
            }
        )
        candidate = CandidateId(self.candidate_id)
        observations = tuple(
            CandidateObservation(
                candidate=candidate,
                case_id=case_id,
                status=Status.OK if self.score >= 90 and document is not None else Status.FAILED,
                answer=str(document["text"]) if self.score >= 90 and document is not None else "",
                source_ids=[str(document["source_id"])] if document is not None else [],
                latency_ms=1,
            )
            for case_id, question in zip(context.case_ids, context.questions, strict=True)
            for document in (_document_for_fixture_question(context, question),)
        )
        return CandidateOutcome(
            candidate=self.candidate_id,
            status=Status.OK,
            score=self.score,
            answered_cases=len(context.questions),
            scored_cases=len(context.questions),
            cost_usd=self.cost_usd,
            latency_ms=1,
            artifact={
                "test_mode": True,
                "fixture_id": self.fixture_id,
                "source_urls": source_urls,
            },
            observations=observations,
            cost_status=CostStatus.COMPLETE,
        )

    def cleanup(self) -> None:
        return


def _document_for_fixture_question(
    context: CandidateContext,
    question: str,
) -> Mapping[str, object] | None:
    """Resolve the deterministic fixture answer to its source document."""
    normalized = question.casefold()
    match = re.search(r"(?:fact|value) (\d+)", normalized)
    index = match.group(1) if match is not None else None
    return next(
        (
            document
            for document in context.documents
            if str(document.get("source_id", "")).casefold() in normalized
            or (index is not None and index in str(document.get("text", "")))
        ),
        None,
    )


def fixture_connectors(spec: FixtureSpec) -> tuple[FixtureConnector, FixtureConnector]:
    return (
        FixtureConnector("slack", spec.documents),
        FixtureConnector("notion", spec.documents),
    )


def fixture_candidates(spec: FixtureSpec) -> tuple[FixtureCandidate, ...]:
    return tuple(
        FixtureCandidate(item.id, item.score, item.cost_usd, spec.fixture_id)
        for item in spec.candidates
    )
