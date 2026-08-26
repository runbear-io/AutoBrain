"""Offline Confluence read-only boundary.

This module intentionally has no HTTP client, OAuth flow, or source-write API.
The public Atlassian endpoint observed in the reuse-first probe is represented as
an external gate until an authenticated contract is independently verified.
Sanitized fixture pages can still exercise the stable document/provenance seam.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any, Self, cast
from urllib.parse import urljoin

from pydantic import Field, model_validator

from autobrain.auth.models import Provider
from autobrain.contracts import SourceConnectionState
from autobrain.models import (
    CoverageCompleteness,
    CoverageRecord,
    NormalizedDocument,
    SourceKind,
    StrictModel,
    normalize_safe_source_url,
)


class ConfluenceGateReason(str):
    """String constants used instead of claiming an authenticated connector."""

    AUTHENTICATED_CONTRACT_UNVERIFIED = "AUTHENTICATED_CONTRACT_UNVERIFIED"
    INVALID_TOKEN = "INVALID_TOKEN"


class ConfluenceExternalGate(StrictModel):
    """Receipt for the blocked Confluence live surface."""

    provider: Provider = Provider.CONFLUENCE
    state: str = "EXTERNAL_GATE"
    connection_state: SourceConnectionState = SourceConnectionState.FAILED
    official_surface: str = "https://mcp.atlassian.com/v1/mcp"
    transport: str = "streamable_http"
    observed_status: int = Field(default=401, ge=100, le=599)
    observed_error: str = "invalid_token"
    credential_present: bool = False
    read_only: bool = True
    network_allowed: bool = False
    source_writes_allowed: bool = False
    reason: str = ConfluenceGateReason.AUTHENTICATED_CONTRACT_UNVERIFIED
    detail: str = Field(
        default="Confluence live access is blocked behind an external gate.",
        min_length=1,
    )
    remediation: str = Field(
        default="Use a sanitized local fixture until authenticated evidence exists.",
        min_length=1,
    )

    @model_validator(mode="after")
    def gate_is_fail_closed(self) -> Self:
        if self.state != "EXTERNAL_GATE" or self.connection_state is SourceConnectionState.READY:
            raise ValueError("Confluence external gate must remain non-ready")
        if self.credential_present or self.network_allowed or self.source_writes_allowed:
            raise ValueError("Confluence gate cannot claim credentials, network, or writes")
        return self


def confluence_external_gate() -> ConfluenceExternalGate:
    """Return the recorded 401 gate without touching credentials or the network."""
    return ConfluenceExternalGate(
        reason=ConfluenceGateReason.INVALID_TOKEN,
        detail=(
            "Authenticated Confluence MCP/REST read contract is not verified; the "
            "recorded unauthenticated probe returned HTTP 401 invalid_token."
        ),
        remediation=(
            "Keep Confluence gated until authenticated read-surface evidence is "
            "available; use a sanitized fixture for offline checks."
        ),
    )


class ConfluenceFixturePage(StrictModel):
    """Minimal sanitized page shape accepted by the offline normalizer."""

    id: str = Field(min_length=1, pattern=r"^[0-9]+$")
    title: str = Field(min_length=1)
    text: str
    url: str = Field(pattern=r"^https?://")
    space_key: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    version: int = Field(ge=1)
    status: str = "current"

    @model_validator(mode="after")
    def current_page_only(self) -> Self:
        if self.status != "current":
            raise ValueError("only current Confluence pages may enter a fixture corpus")
        return self


def _fixture_page(raw: Mapping[str, Any]) -> ConfluenceFixturePage:
    """Normalize the two safe fixture forms without accepting API write fields."""
    if "body" in raw or "_links" in raw:
        allowed_api_fields = {"id", "title", "body", "_links", "version", "space", "status"}
        unexpected = set(raw) - allowed_api_fields
        if unexpected:
            raise ValueError(f"unsupported Confluence fixture fields: {sorted(unexpected)}")
        body = raw.get("body")
        if not isinstance(body, Mapping):
            raise ValueError("Confluence fixture body must be an object")
        body_map = cast(Mapping[str, Any], body)
        storage = body_map.get("storage")
        view = body_map.get("view")
        storage_map = cast(Mapping[str, Any], storage) if isinstance(storage, Mapping) else None
        view_map = cast(Mapping[str, Any], view) if isinstance(view, Mapping) else None
        text = storage_map.get("value") if storage_map is not None else None
        if not isinstance(text, str):
            text = view_map.get("value") if view_map is not None else None
        links = raw.get("_links")
        links_map = cast(Mapping[str, Any], links) if isinstance(links, Mapping) else None
        webui = links_map.get("webui") if links_map is not None else None
        base = links_map.get("base") if links_map is not None else None
        url = urljoin(str(base), str(webui)) if isinstance(webui, str) else None
        version = raw.get("version")
        version_map = cast(Mapping[str, Any], version) if isinstance(version, Mapping) else None
        version_value: object = cast(
            object,
            version_map.get("number") if version_map is not None else version,
        )
        version_number = version_value if isinstance(version_value, int) else None
        space = raw.get("space")
        space_map = cast(Mapping[str, Any], space) if isinstance(space, Mapping) else None
        space_key = space_map.get("key") if space_map is not None else None
        payload: dict[str, Any] = {
            "id": raw.get("id"),
            "title": raw.get("title"),
            "text": text,
            "url": url,
            "space_key": space_key,
            "version": version_number,
            "status": raw.get("status", "current"),
        }
    else:
        payload = dict(raw)
    page = ConfluenceFixturePage.model_validate(payload, strict=True)
    safe_url = normalize_safe_source_url(page.url)
    if safe_url is None:
        raise ValueError("Confluence fixture URL is not a safe HTTP(S) URL")
    return page.model_copy(update={"url": safe_url})


def normalize_confluence_fixture(
    pages: Sequence[Mapping[str, Any] | ConfluenceFixturePage],
) -> tuple[NormalizedDocument, ...]:
    """Convert sanitized pages to stable, read-only AutoBrain documents.

    The returned documents contain page text only as corpus input and carry
    source identity/version metadata needed to audit the fixture. No response
    field is interpreted as a command and no write operation is exposed.
    """
    normalized: list[NormalizedDocument] = []
    seen: set[str] = set()
    for item in pages:
        page = item if isinstance(item, ConfluenceFixturePage) else _fixture_page(item)
        source_id = f"confluence:page:{page.id}"
        if source_id in seen:
            raise ValueError(f"duplicate Confluence page ID: {page.id}")
        seen.add(source_id)
        normalized.append(
            NormalizedDocument(
                source_id=source_id,
                source_kind=SourceKind.CONFLUENCE_PAGE,
                canonical_url=page.url,
                title=page.title,
                text=page.text,
                content_hash=sha256(page.text.encode("utf-8")).hexdigest(),
                container=page.space_key,
                source_references=[source_id],
                crawl_provenance={
                    "connector": "confluence-read-only-fixture",
                    "source_id": source_id,
                    "space_key": page.space_key,
                    "page_version": str(page.version),
                    "transport": "fixture",
                    "offline_gate": "W2",
                },
            )
        )
    return tuple(sorted(normalized, key=lambda document: document.source_id))


def confluence_fixture_corpus(
    pages: Sequence[Mapping[str, Any] | ConfluenceFixturePage],
) -> tuple[tuple[NormalizedDocument, ...], CoverageRecord]:
    """Return normalized pages and deterministic coverage for offline probes."""
    documents = normalize_confluence_fixture(pages)
    return documents, CoverageRecord(
        source=SourceKind.CONFLUENCE_PAGE,
        completeness=CoverageCompleteness.EXHAUSTIVE,
        discovered=len(documents),
        fetched=len(documents),
        crawl_provenance={
            "connector": "confluence-read-only-fixture",
            "transport": "fixture",
            "offline_gate": "W2",
        },
    )
