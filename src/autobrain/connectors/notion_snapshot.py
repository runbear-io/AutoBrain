"""Strict, read-only imports of externally fetched Notion MCP snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator

from autobrain.cancellation import RunCancellation
from autobrain.models import (
    CoverageCompleteness,
    CoverageRecord,
    NormalizedDocument,
    SourceKind,
    Status,
    StrictModel,
    normalize_safe_source_url,
)
from autobrain.orchestration import ConnectorSnapshot

MAX_SNAPSHOT_BYTES = 1_024 * 1_024
MAX_DOCUMENT_BYTES = 256 * 1_024
MAX_DOCUMENTS = 1_000
MAX_CONTENT_CHARS = MAX_DOCUMENT_BYTES // 4
_PROMPT_LIKE = re.compile(
    r"\b(ignore|disregard|override|system prompt|assistant|"
    r"follow these instructions|call a (?:write|tool))\b",
    re.IGNORECASE,
)
_CONCRETE_CREDENTIAL = re.compile(
    r"(?i)(?<![a-z0-9])(?:"
    r"bearer\s+[a-z0-9._~+/=-]{8,}|"
    r"(?:sk|xox[baprs])-[a-z0-9._-]{8,}|"
    r"(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|client[_ -]?secret|"
    r"password|authorization|credential)\s*[=:]\s*[a-z0-9._~+/=-]{8,}|"
    r"//[^/\s:@]+:[^@\s]+@)"
)
_PLACEHOLDER_VALUE = (
    r"(?:<\s*[A-Z0-9_-]+\s*>|\[\s*[A-Z0-9_-]+\s*\]|"
    r"\{\{?\s*[A-Z0-9_-]+\s*\}?\}|\$\{\s*[A-Z0-9_-]+\s*\}|"
    r"(?:YOUR|EXAMPLE|SAMPLE|DUMMY|TEST)[_-][A-Z0-9_-]+|"
    r"PLACEHOLDER(?:[_-]VALUE)?|CHANGEME)"
)
_CREDENTIAL_PLACEHOLDER = re.compile(
    rf"(?i)(?P<prefix>\bbearer\s+|"
    rf"\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|client[_ -]?secret|"
    rf"password|authorization|credential)\s*[=:]\s*){_PLACEHOLDER_VALUE}"
)
_MUTATION = re.compile(
    r"(?:write|create|update|delete|insert|patch|mutat|remove|append)", re.IGNORECASE
)

PageId = Annotated[str, Field(min_length=1, max_length=256, pattern=r"^[^/\\]+$")]


class SnapshotTextClassification(StrEnum):
    SAFE = "SAFE"
    PLACEHOLDER_NORMALIZED = "PLACEHOLDER_NORMALIZED"
    CONCRETE_CREDENTIAL = "CONCRETE_CREDENTIAL"


@dataclass(frozen=True)
class ClassifiedSnapshotText:
    text: str
    classification: SnapshotTextClassification


def _classify_snapshot_text(value: str) -> ClassifiedSnapshotText:
    def normalize(match: re.Match[str]) -> str:
        return (
            match.group(0)
            if match.group(0).endswith("[REDACTED_PLACEHOLDER]")
            else "[REDACTED_PLACEHOLDER]"
        )

    normalized, placeholder_count = _CREDENTIAL_PLACEHOLDER.subn(normalize, value)
    if _CONCRETE_CREDENTIAL.search(normalized):
        return ClassifiedSnapshotText(
            text=normalized,
            classification=SnapshotTextClassification.CONCRETE_CREDENTIAL,
        )
    return ClassifiedSnapshotText(
        text=normalized,
        classification=(
            SnapshotTextClassification.PLACEHOLDER_NORMALIZED
            if placeholder_count
            else SnapshotTextClassification.SAFE
        ),
    )


class NotionSnapshotError(ValueError):
    """The supplied snapshot is not a safe bounded read-only snapshot."""


class NotionSnapshotDocument(StrictModel):
    page_id: PageId
    page_url: str
    title: str = Field(min_length=1, max_length=10_000)
    fetched_at: datetime
    content: str = Field(max_length=MAX_CONTENT_CHARS)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("page_url")
    @classmethod
    def safe_url(cls, value: str) -> str:
        normalized = normalize_safe_source_url(value)
        if normalized is None or "notion" not in normalized.lower():
            raise ValueError("page_url must be a safe Notion HTTP(S) URL")
        return normalized

    @field_validator("content", "title")
    @classmethod
    def no_secrets(cls, value: str) -> str:
        classified = _classify_snapshot_text(value)
        if classified.classification is SnapshotTextClassification.CONCRETE_CREDENTIAL:
            raise ValueError("concrete credentials are not permitted in snapshot data")
        return classified.text

    @field_validator("metadata")
    @classmethod
    def safe_metadata(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, item in value.items():
            if _PROMPT_LIKE.search(key) or _PROMPT_LIKE.search(item):
                raise ValueError("prompt-like metadata is not permitted")
            classified_key = _classify_snapshot_text(key)
            classified_item = _classify_snapshot_text(item)
            if (
                _MUTATION.search(key)
                or _MUTATION.search(item)
                or any(
                    classified.classification is SnapshotTextClassification.CONCRETE_CREDENTIAL
                    for classified in (classified_key, classified_item)
                )
            ):
                raise ValueError("write/mutation or concrete credential metadata is not permitted")
            normalized[classified_key.text] = classified_item.text
        return normalized


class NotionSnapshot(StrictModel):
    schema_version: int = Field(strict=True, ge=1, le=1)
    source: str = Field(pattern=r"^notion-mcp-snapshot$")
    fetched_at: datetime
    documents: list[NotionSnapshotDocument] = Field(min_length=1, max_length=MAX_DOCUMENTS)

    @field_validator("documents")
    @classmethod
    def unique_pages(cls, value: list[NotionSnapshotDocument]) -> list[NotionSnapshotDocument]:
        ids = [document.page_id for document in value]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate page IDs are not permitted")
        return value


class NotionSnapshotStatus(StrictModel):
    ready: bool
    status: Status
    detail: str
    snapshot_path: Path | None = None
    snapshot_sha256: str | None = None
    document_count: int = 0
    fetched_at: datetime | None = None
    coverage: CoverageRecord = CoverageRecord(
        source=SourceKind.NOTION_PAGE,
        completeness=CoverageCompleteness.UNKNOWN,
        discovered=0,
        fetched=0,
        unsupported=1,
    )


class NotionSnapshotConfig(StrictModel):
    schema_version: int = 1
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_count: int = Field(ge=1)
    fetched_at: datetime


class NotionSnapshotIntegrity(StrictModel):
    schema_version: int = Field(strict=True, ge=1, le=1)
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_count: int = Field(ge=1)
    fetched_at: datetime


def _read_snapshot(path: Path) -> tuple[NotionSnapshot, bytes]:
    if any(part == ".." for part in path.parts):
        raise NotionSnapshotError("snapshot path contains traversal")
    if path.is_symlink():
        raise NotionSnapshotError("snapshot input cannot be a symlink")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise NotionSnapshotError(f"snapshot cannot be read: {exc}") from exc
    if len(raw) > MAX_SNAPSHOT_BYTES:
        raise NotionSnapshotError("snapshot is too large")
    try:
        snapshot = NotionSnapshot.model_validate_json(raw)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise NotionSnapshotError("invalid Notion snapshot") from exc
    encoded = json.dumps(
        snapshot.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    if len(encoded) > MAX_SNAPSHOT_BYTES:
        raise NotionSnapshotError("snapshot is too large")
    for document in snapshot.documents:
        if len(document.content.encode("utf-8")) > MAX_DOCUMENT_BYTES:
            raise NotionSnapshotError("document is too large")
    classified = _classify_snapshot_text(encoded.decode("utf-8", errors="ignore"))
    if classified.classification is SnapshotTextClassification.CONCRETE_CREDENTIAL:
        raise NotionSnapshotError("snapshot contains concrete credential data")
    return snapshot, encoded


class NotionSnapshotStore:
    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root
        self.snapshot_path = source_root / "notion-snapshot.json"
        self.integrity_path = source_root / "notion-snapshot.integrity.json"

    def import_snapshot(self, input_path: Path) -> NotionSnapshotConfig:
        snapshot, encoded = _read_snapshot(input_path.expanduser())
        if self.source_root.is_symlink():
            raise NotionSnapshotError("source directory cannot be a symlink")
        self.source_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.snapshot_path.is_symlink() or self.integrity_path.is_symlink():
            raise NotionSnapshotError("snapshot storage cannot contain symlinks")
        digest = hashlib.sha256(encoded).hexdigest()
        config = NotionSnapshotConfig(
            snapshot_sha256=digest,
            document_count=len(snapshot.documents),
            fetched_at=snapshot.fetched_at,
        )
        self._atomic_write(self.snapshot_path, encoded + b"\n")
        self._atomic_write(
            self.integrity_path,
            json.dumps(
                config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode()
            + b"\n",
        )
        return config

    def verified_snapshot(self) -> tuple[NotionSnapshot, NotionSnapshotIntegrity, str] | None:
        if not self.snapshot_path.exists():
            return None
        if self.snapshot_path.is_symlink() or self.integrity_path.is_symlink():
            raise NotionSnapshotError("snapshot storage cannot contain symlinks")
        if not self.integrity_path.exists():
            raise NotionSnapshotError("snapshot integrity sidecar is missing")
        try:
            sidecar = NotionSnapshotIntegrity.model_validate_json(self.integrity_path.read_bytes())
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise NotionSnapshotError("invalid snapshot integrity sidecar") from exc
        snapshot, encoded = _read_snapshot(self.snapshot_path)
        digest = hashlib.sha256(encoded).hexdigest()
        if digest != sidecar.snapshot_sha256:
            raise NotionSnapshotError("snapshot changed since import")
        if (
            len(snapshot.documents) != sidecar.document_count
            or snapshot.fetched_at != sidecar.fetched_at
        ):
            raise NotionSnapshotError("snapshot integrity metadata does not match")
        return snapshot, sidecar, digest

    def load(self) -> NotionSnapshot | None:
        verified = self.verified_snapshot()
        return verified[0] if verified is not None else None

    def status(self) -> NotionSnapshotStatus:
        try:
            snapshot = self.load()
        except NotionSnapshotError as exc:
            detail = str(exc)
            status = Status.UNVERIFIABLE if "sidecar" in detail else Status.INVALID_CONFIG
            if "changed" in detail:
                status = Status.SNAPSHOT_CHANGED
            return NotionSnapshotStatus(
                ready=False, status=status, detail=detail, snapshot_path=self.snapshot_path
            )
        if snapshot is None:
            return NotionSnapshotStatus(
                ready=False, status=Status.INVALID_CONFIG, detail="No Notion snapshot is configured"
            )
        verified = self.verified_snapshot()
        assert verified is not None
        _, _sidecar, digest = verified
        return NotionSnapshotStatus(
            ready=True,
            status=Status.OK,
            detail=f"{len(snapshot.documents)} pages from external read-only MCP session",
            snapshot_path=self.snapshot_path,
            snapshot_sha256=digest,
            document_count=len(snapshot.documents),
            fetched_at=snapshot.fetched_at,
            coverage=CoverageRecord(
                source=SourceKind.NOTION_PAGE,
                completeness=CoverageCompleteness.UNKNOWN,
                discovered=len(snapshot.documents),
                fetched=len(snapshot.documents),
                crawl_provenance={
                    "connector": "notion-mcp-snapshot",
                    "partial": "true",
                    "snapshot_sha256": digest,
                    "transport_mode": "imported_snapshot",
                },
            ),
        )

    def remove(self) -> None:
        if self.snapshot_path.is_symlink() or self.integrity_path.is_symlink():
            raise NotionSnapshotError("snapshot storage cannot contain symlinks")
        self.snapshot_path.unlink(missing_ok=True)
        self.integrity_path.unlink(missing_ok=True)

    def _atomic_write(self, destination: Path, content: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.source_root, prefix=".notion-snapshot-", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
            temporary.replace(destination)
            destination.chmod(0o600)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def snapshot_documents(
    snapshot: NotionSnapshot, *, snapshot_sha256: str | None = None
) -> tuple[dict[str, object], ...]:
    return tuple(
        NormalizedDocument(
            source_id=f"notion:page:{document.page_id}",
            source_kind=SourceKind.NOTION_PAGE,
            canonical_url=document.page_url,
            title=document.title,
            text=document.content,
            content_hash=hashlib.sha256(document.content.encode("utf-8")).hexdigest(),
            updated_at=document.fetched_at,
            crawl_provenance={
                "connector": "notion-mcp-snapshot",
                "partial": "true",
                **({"snapshot_sha256": snapshot_sha256} if snapshot_sha256 else {}),
                "transport_mode": "imported_snapshot",
            },
            warnings=sorted(
                {
                    "imported from an external MCP session; content is untrusted data",
                    *(
                        ["prompt-like source instructions preserved as inert data"]
                        if _PROMPT_LIKE.search(document.content)
                        else []
                    ),
                }
            ),
            metadata={"fetched_at": document.fetched_at.isoformat(), **document.metadata},
        ).model_dump(mode="json")
        for document in snapshot.documents
    )


class NotionSnapshotConnector:
    provider = "notion"

    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path
        self._cached_verified: tuple[NotionSnapshot, NotionSnapshotIntegrity, str] | None = None

    def probe(self, cancellation: RunCancellation | None = None) -> dict[str, object]:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        store = NotionSnapshotStore(self.snapshot_path.parent)
        try:
            verified = store.verified_snapshot()
            self._cached_verified = verified
        except NotionSnapshotError as exc:
            return {
                "allowed": ["snapshot-read"],
                "capability_available": False,
                "status": store.status().status.value,
                "detail": str(exc),
            }
        return {
            "allowed": ["snapshot-read"],
            "capability_available": verified is not None,
            "status": Status.OK.value if verified is not None else Status.INVALID_CONFIG.value,
        }

    def crawl(self, *, cancellation: RunCancellation | None = None) -> ConnectorSnapshot:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        store = NotionSnapshotStore(self.snapshot_path.parent)
        verified = self._cached_verified
        if verified is None:
            verified = store.verified_snapshot()
        if verified is None:
            raise NotionSnapshotError("No Notion snapshot is configured")
        snapshot, sidecar, digest = verified
        final_snapshot, final_encoded = _read_snapshot(self.snapshot_path)
        final_digest = hashlib.sha256(final_encoded).hexdigest()
        if (
            final_digest != digest
            or len(final_snapshot.documents) != sidecar.document_count
            or final_snapshot.fetched_at != sidecar.fetched_at
        ):
            raise NotionSnapshotError("snapshot changed during run")
        return ConnectorSnapshot(
            provider=self.provider,
            documents=snapshot_documents(snapshot, snapshot_sha256=digest),
            coverage=store.status().coverage.model_dump(mode="json"),
        )
