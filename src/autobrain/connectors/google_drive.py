"""Fail-closed gate for the not-yet-approved Google Drive connector.

No Drive transport belongs here until project policy permits an official Google
Drive SDK/API dependency and its credential contract has been verified.  The
structured gate is intentionally useful to source status and doctor without
pretending that binary files were crawled or OCR'd.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from enum import StrEnum
from hashlib import sha256
from typing import Any, Self

from pydantic import Field, model_validator

from autobrain.auth.models import Provider
from autobrain.models import (
    CoverageCompleteness,
    CoverageRecord,
    NormalizedDocument,
    SourceKind,
    StrictModel,
    normalize_safe_source_url,
)


class ExternalGateReason(StrEnum):
    PROJECT_POLICY = "PROJECT_POLICY"
    OFFICIAL_SDK_UNVERIFIED = "OFFICIAL_SDK_UNVERIFIED"
    CREDENTIALS_UNAVAILABLE = "CREDENTIALS_UNAVAILABLE"
    PUBLIC_API_UNVERIFIED = "PUBLIC_API_UNVERIFIED"


class GoogleDriveExternalGate(StrictModel):
    """Machine-readable evidence that Drive integration is intentionally blocked."""

    provider: Provider = Provider.GOOGLE_DRIVE
    state: str = "EXTERNAL_GATE"
    reason: ExternalGateReason
    official_surface: str = Field(min_length=1)
    dependency: str = Field(min_length=1)
    dependency_available: bool
    credentials_present: bool = False
    public_api_verified: bool = False
    read_only: bool = True
    network_allowed: bool = False
    source_writes_allowed: bool = False
    required_contract: tuple[str, ...] = (
        "explicit text MIME handling",
        "explicit Google Workspace export handling",
        "stable source IDs based on Drive file IDs",
        "ACL/permission denied coverage",
        "rate-limit coverage",
        "truncated response coverage",
        "unsupported MIME coverage",
        "source provenance including file ID and MIME type",
        "no OCR and no silent binary skips",
    )
    detail: str = Field(min_length=1)
    remediation: str = Field(min_length=1)

    @model_validator(mode="after")
    def gate_is_fail_closed(self) -> Self:
        if self.state != "EXTERNAL_GATE" or not self.read_only:
            raise ValueError("Google Drive gate must remain read-only and non-ready")
        if self.network_allowed or self.source_writes_allowed or self.public_api_verified:
            raise ValueError("Google Drive gate cannot claim live access or verified API use")
        return self


def google_drive_external_gate() -> GoogleDriveExternalGate:
    """Return the current fail-closed gate without reading credential contents."""
    sdk_available = all(
        importlib.util.find_spec(module) is not None
        for module in ("googleapiclient", "google.oauth2")
    )
    return GoogleDriveExternalGate(
        reason=ExternalGateReason.PROJECT_POLICY,
        official_surface="Google Drive API v3 via google-api-python-client/google-auth",
        dependency="google-api-python-client and google-auth",
        dependency_available=sdk_available,
        detail=(
            "Google Drive is blocked: project policy does not currently permit an "
            "official Drive API/SDK connector; no REST, MCP, OCR, or binary fallback "
            "is used."
        ),
        remediation=(
            "Keep Drive gated until project policy, the official dependency, and an "
            "authenticated read-only contract are independently approved; use a "
            "sanitized local fixture for offline checks."
        ),
    )


_TEXT_MIME_TYPES = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/csv",
        "application/json",
        "application/xml",
    }
)
_WORKSPACE_EXPORTS = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
}


class GoogleDriveFixtureFile(StrictModel):
    """Sanitized Drive file shape for the offline, read-only contract."""

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    web_view_link: str = Field(pattern=r"^https?://")
    text: str
    export_mime_type: str | None = None

    @model_validator(mode="after")
    def content_mime_is_explicit(self) -> Self:
        if self.mime_type in _WORKSPACE_EXPORTS:
            expected = _WORKSPACE_EXPORTS[self.mime_type]
            if self.export_mime_type != expected:
                raise ValueError(f"Google Workspace files require export_mime_type={expected!r}")
        elif self.mime_type not in _TEXT_MIME_TYPES:
            raise ValueError("unsupported Drive MIME type requires explicit fixture handling")
        elif self.export_mime_type is not None:
            raise ValueError("export_mime_type is only valid for Google Workspace files")
        return self


def _fixture_file(raw: Mapping[str, Any]) -> GoogleDriveFixtureFile:
    """Validate a sanitized file payload without accepting Drive write fields."""
    allowed = {
        "id",
        "name",
        "mime_type",
        "mimeType",
        "web_view_link",
        "webViewLink",
        "text",
        "export_mime_type",
        "exportMimeType",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unsupported Google Drive fixture fields: {sorted(unknown)}")
    if "mime_type" in raw and "mimeType" in raw and raw["mime_type"] != raw["mimeType"]:
        raise ValueError("conflicting Google Drive MIME type fields")
    if (
        "web_view_link" in raw
        and "webViewLink" in raw
        and raw["web_view_link"] != raw["webViewLink"]
    ):
        raise ValueError("conflicting Google Drive link fields")
    if (
        "export_mime_type" in raw
        and "exportMimeType" in raw
        and raw["export_mime_type"] != raw["exportMimeType"]
    ):
        raise ValueError("conflicting Google Drive export MIME fields")
    payload = {
        "id": raw.get("id"),
        "name": raw.get("name"),
        "mime_type": raw.get("mime_type", raw.get("mimeType")),
        "web_view_link": raw.get("web_view_link", raw.get("webViewLink")),
        "text": raw.get("text"),
        "export_mime_type": raw.get("export_mime_type", raw.get("exportMimeType")),
    }
    return GoogleDriveFixtureFile.model_validate(payload, strict=True)


def normalize_google_drive_fixture(
    files: Sequence[Mapping[str, Any] | GoogleDriveFixtureFile],
) -> tuple[NormalizedDocument, ...]:
    """Normalize sanitized text/export fixtures into stable Drive documents."""
    normalized: list[NormalizedDocument] = []
    seen: set[str] = set()
    for item in files:
        file = item if isinstance(item, GoogleDriveFixtureFile) else _fixture_file(item)
        url = normalize_safe_source_url(file.web_view_link)
        if url is None:
            raise ValueError("Google Drive fixture URL is not a safe HTTP(S) URL")
        source_id = f"google_drive:file:{file.id}"
        if source_id in seen:
            raise ValueError(f"duplicate Google Drive file ID: {file.id}")
        seen.add(source_id)
        normalized.append(
            NormalizedDocument(
                source_id=source_id,
                source_kind=SourceKind.GOOGLE_DRIVE_FILE,
                canonical_url=url,
                title=file.name,
                text=file.text,
                content_hash=sha256(file.text.encode("utf-8")).hexdigest(),
                source_references=[source_id],
                crawl_provenance={
                    "connector": "google-drive-read-only-fixture",
                    "source_id": source_id,
                    "file_id": file.id,
                    "mime_type": file.mime_type,
                    "content_mime_type": file.export_mime_type or file.mime_type,
                    "transport": "fixture",
                    "offline_gate": "PROJECT_POLICY",
                },
            )
        )
    return tuple(sorted(normalized, key=lambda document: document.source_id))


def google_drive_fixture_corpus(
    files: Sequence[Mapping[str, Any] | GoogleDriveFixtureFile],
) -> tuple[tuple[NormalizedDocument, ...], CoverageRecord]:
    """Return normalized files and deterministic, provenance-preserving coverage."""
    documents = normalize_google_drive_fixture(files)
    return documents, CoverageRecord(
        source=SourceKind.GOOGLE_DRIVE_FILE,
        completeness=CoverageCompleteness.EXHAUSTIVE,
        discovered=len(documents),
        fetched=len(documents),
        crawl_provenance={
            "connector": "google-drive-read-only-fixture",
            "transport": "fixture",
            "offline_gate": "PROJECT_POLICY",
        },
    )
