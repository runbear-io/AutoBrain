"""Strict, data-only fake-MCP fixture runtime for installed CLI QA."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, model_validator

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

_SECRET = re.compile(
    r"(?i)(?<![a-z0-9])(?:sk-[a-z0-9_-]{8,}|xox[a-z]-[a-z0-9-]{8,}|"
    r"bearer\s+[a-z0-9._~+/=-]{8,}|"
    r"(?:api[_-]?key|token|password|authorization)[=: ]+[a-z0-9._~+/=-]{8,})"
)
_FORBIDDEN = re.compile(
    r"(?i)\b(?:oracle|holdout|reference[\s_-]*answer|expected[\s_-]*claim|raw[\s_-]*reply)\b"
)


class FixtureValidationError(ValueError):
    """A local fixture failed its strict, non-executable contract."""


class FixtureDocument(StrictModel):
    provider: Literal["slack", "notion"]
    source_id: str = Field(pattern=r"^(?:slack|notion):[A-Za-z0-9:_-]+$")
    source_kind: Literal["SLACK_MESSAGE", "NOTION_PAGE"]
    canonical_url: str
    title: str = Field(min_length=1)
    text: str
    question: str = Field(min_length=1)
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

    @model_validator(mode="after")
    def validate_fixture(self) -> FixtureSpec:
        source_ids = [document.source_id for document in self.documents]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("fixture source IDs must be unique")
        candidate_ids = [candidate.id for candidate in self.candidates]
        if candidate_ids != ["llm-wiki", "mem0", "gbrain"]:
            raise ValueError("fixture candidates must be exactly llm-wiki, mem0, gbrain")
        serialized = self.model_dump(mode="json")
        claimed = serialized.pop("fixture_sha256")
        encoded = json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(encoded).hexdigest() != claimed:
            raise ValueError("fixture_sha256 does not match canonical fixture content")
        for value in _walk_strings(serialized):
            if _SECRET.search(value) or _FORBIDDEN.search(value):
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
                status=Status.OK if self.score >= 90 else Status.FAILED,
                answer=(
                    str(
                        next(
                            document["text"]
                            for document in context.documents
                            if str(index) in str(document.get("text", ""))
                        )
                    )
                    if self.score >= 90
                    else ""
                ),
                source_ids=[
                    str(document["source_id"])
                    for document in context.documents
                    if str(index) in str(document.get("text", ""))
                ],
                latency_ms=1,
            )
            for case_id, index in zip(
                context.case_ids,
                (question.rsplit(" ", 1)[-1].rstrip("?") for question in context.questions),
                strict=True,
            )
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
