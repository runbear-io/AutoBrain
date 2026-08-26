"""Source-neutral readiness and corpus contracts for connector fixtures.

This module deliberately contains no transport or authentication code.  Fixture
sources can prove normalization and corpus behavior; live OAuth sources remain
explicitly gated until a separately verified connector exists.
"""

from __future__ import annotations

import hashlib
import json
import stat
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from autobrain.auth.models import Provider
from autobrain.auth.providers import config_for
from autobrain.connectors.confluence import ConfluenceExternalGate, confluence_external_gate
from autobrain.connectors.google_drive import GoogleDriveExternalGate, google_drive_external_gate
from autobrain.contracts import (
    SourceOAuthAppConfigV1,
    SourceProvider,
    SourceTransportMode,
)
from autobrain.models import CoverageRecord, NormalizedDocument, SourceKind, StrictModel

# Keep one provider identity across auth, configuration, and connector readiness.
ConnectorSource = Provider


class ReadinessMode(StrEnum):
    FIXTURE = "FIXTURE"
    LIVE = "LIVE"
    OAUTH_GATED = "OAUTH_GATED"
    UNSUPPORTED = "UNSUPPORTED"


class ReadinessState(StrEnum):
    READY = "READY"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    UNSUPPORTED = "UNSUPPORTED"


class ReadinessReason(StrEnum):
    """Stable machine-readable reason for a readiness state."""

    FIXTURE_READY = "FIXTURE_READY"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    UNSUPPORTED_CONNECTOR = "UNSUPPORTED_CONNECTOR"


class TransportReadinessState(StrEnum):
    """Offline transport state; app registration is not authenticated access."""

    READY = "READY"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    UNAVAILABLE = "UNAVAILABLE"
    CONFIG_INVALID = "CONFIG_INVALID"


class TransportGovernanceCode(StrEnum):
    SOURCE_TRANSPORT_READY = "SOURCE_TRANSPORT_READY"
    SOURCE_TRANSPORT_UNAVAILABLE = "SOURCE_TRANSPORT_UNAVAILABLE"
    SOURCE_OAUTH_APP_MISSING = "SOURCE_OAUTH_APP_MISSING"
    SOURCE_OAUTH_CONFIG_INVALID = "SOURCE_OAUTH_CONFIG_INVALID"
    SOURCE_OAUTH_APP_UNSAFE = "SOURCE_OAUTH_APP_UNSAFE"
    SOURCE_REDIRECT_URI_MISMATCH = "SOURCE_REDIRECT_URI_MISMATCH"


class SourceTransportReadiness(StrictModel):
    """Deterministic, credential-value-free source transport status."""

    provider: SourceProvider
    mode: SourceTransportMode
    state: TransportReadinessState
    ready: bool
    governance_code: TransportGovernanceCode
    credential_present: bool = False
    app_configured: bool = False
    detail: str
    remediation: str = ""

    @model_validator(mode="after")
    def readiness_is_fail_closed(self) -> Self:
        if self.ready != (self.state is TransportReadinessState.READY):
            raise ValueError("ready must agree with transport state")
        if (
            self.ready
            and not self.credential_present
            and self.mode is not SourceTransportMode.EXPORT_ARCHIVE
        ):
            raise ValueError("live transport cannot be ready without credentials")
        if (
            self.governance_code is TransportGovernanceCode.SOURCE_TRANSPORT_READY
            and not self.ready
        ):
            raise ValueError("ready governance code requires ready transport")
        return self


class SourceTransportConfigError(ValueError):
    """A local source OAuth app file failed security or schema validation."""


_SOURCE_KINDS: dict[ConnectorSource, SourceKind] = {
    ConnectorSource.CONFLUENCE: SourceKind.CONFLUENCE_PAGE,
    ConnectorSource.GOOGLE_DRIVE: SourceKind.GOOGLE_DRIVE_FILE,
}


class CorpusContract(StrictModel):
    """Deterministic, source-neutral output shared by connector implementations."""

    source: ConnectorSource
    documents: tuple[NormalizedDocument, ...]
    coverage: CoverageRecord
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def source_matches_documents(self) -> Self:
        expected_kind = _SOURCE_KINDS.get(self.source)
        if expected_kind is not None and any(
            document.source_kind is not expected_kind for document in self.documents
        ):
            raise ValueError(f"documents do not match {self.source.value} source")
        if self.coverage.source is not (expected_kind or self.coverage.source):
            raise ValueError("coverage source does not match corpus source")
        expected_hash = _corpus_sha256(self.documents)
        if self.corpus_sha256 != expected_hash:
            raise ValueError(
                f"corpus_sha256 does not match canonical documents (expected {expected_hash})"
            )
        return self

    @classmethod
    def from_documents(
        cls,
        *,
        source: ConnectorSource,
        documents: tuple[NormalizedDocument, ...],
        coverage: CoverageRecord,
    ) -> Self:
        return cls(
            source=source,
            documents=documents,
            coverage=coverage,
            corpus_sha256=_corpus_sha256(documents),
        )


def _corpus_sha256(documents: tuple[NormalizedDocument, ...]) -> str:
    payload = "\n".join(
        json.dumps(document.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        for document in sorted(documents, key=lambda item: item.source_id)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ConnectorRefusal(StrictModel):
    """Typed evidence for an external connector gate.

    This is intentionally not a transport implementation: an endpoint that
    answers unauthenticated does not prove that its authenticated read surface
    is available to AutoBrain.
    """

    code: str = Field(min_length=1)
    endpoint: str = Field(pattern=r"^https://")
    transport: str = Field(min_length=1)
    observed: str = Field(min_length=1)
    credential_present: bool


class ConnectorReadiness(StrictModel):
    """Truthful readiness result; fixture readiness never implies live OAuth."""

    connector: ConnectorSource
    state: ReadinessState
    mode: ReadinessMode
    ready: bool
    live_oauth: bool = False
    detail: str = Field(min_length=1)
    reason: ReadinessReason | None = None
    remediation: str = ""
    corpus: CorpusContract | None = None
    refusal: ConnectorRefusal | None = None
    external_gate: ConfluenceExternalGate | GoogleDriveExternalGate | None = None

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.ready != (self.state is ReadinessState.READY):
            raise ValueError("ready must agree with readiness state")
        if self.mode is ReadinessMode.FIXTURE and self.live_oauth:
            raise ValueError("fixture readiness cannot claim live OAuth")
        if self.ready != (self.corpus is not None):
            raise ValueError("ready connectors must provide a corpus")
        if self.mode is ReadinessMode.FIXTURE and self.reason is not ReadinessReason.FIXTURE_READY:
            raise ValueError("fixture readiness must report FIXTURE_READY reason")
        if (
            self.mode is ReadinessMode.UNSUPPORTED
            and self.reason is not ReadinessReason.UNSUPPORTED_CONNECTOR
        ):
            raise ValueError("unsupported readiness must report UNSUPPORTED_CONNECTOR reason")
        return self


def fixture_readiness(
    source: ConnectorSource,
    *,
    documents: tuple[NormalizedDocument, ...],
    coverage: CoverageRecord,
) -> ConnectorReadiness:
    """Build deterministic readiness for a fixture-only connector surface."""
    if source not in _SOURCE_KINDS:
        raise ValueError(f"fixture readiness is not supported for {source.value}")
    return ConnectorReadiness(
        connector=source,
        state=ReadinessState.READY,
        mode=ReadinessMode.FIXTURE,
        ready=True,
        reason=ReadinessReason.FIXTURE_READY,
        detail=f"{source.value} fixture corpus is ready",
        corpus=CorpusContract.from_documents(
            source=source,
            documents=documents,
            coverage=coverage,
        ),
    )


def readiness_for(source: ConnectorSource) -> ConnectorReadiness:
    """Return truthful readiness from the authoritative provider configuration."""
    config = config_for(source)
    if not config.supported:
        refusal = None
        external_gate = None
        if source is ConnectorSource.GOOGLE_DRIVE:
            external_gate = google_drive_external_gate()
        if source is ConnectorSource.CONFLUENCE:
            refusal = ConnectorRefusal(
                code="OFFICIAL_MCP_UNVERIFIED",
                endpoint="https://mcp.atlassian.com/v1/mcp",
                transport="streamable_http",
                observed="HTTP 401 invalid_token",
                credential_present=False,
            )
            external_gate = confluence_external_gate()
        return ConnectorReadiness(
            connector=source,
            state=ReadinessState.UNSUPPORTED,
            mode=ReadinessMode.UNSUPPORTED,
            ready=False,
            reason=ReadinessReason.UNSUPPORTED_CONNECTOR,
            detail=(
                external_gate.detail
                if external_gate is not None
                else f"{source.value}: {config.detail}"
            ),
            remediation=config.remediation,
            refusal=refusal,
            external_gate=external_gate,
        )
    return ConnectorReadiness(
        connector=source,
        state=ReadinessState.AUTH_REQUIRED,
        mode=ReadinessMode.OAUTH_GATED,
        ready=False,
        reason=ReadinessReason.AUTH_REQUIRED,
        detail=f"{source.value} authentication is required",
    )


def sharepoint_readiness() -> ConnectorReadiness:
    """Backward-compatible named boundary for the planned SharePoint connector."""
    return readiness_for(ConnectorSource.SHAREPOINT)


_DEFAULT_SOURCE_OAUTH_APPS = Path.home() / ".autobrain" / "source-oauth-apps.json"


def _safe_source_oauth_config_path(path: Path) -> Path:
    """Validate the config path before reading it, following auth file hardening."""
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise SourceTransportConfigError("source OAuth app config cannot be a symlink")
    if not expanded.is_absolute():
        raise SourceTransportConfigError("source OAuth app config path must be absolute")
    parent = expanded.parent
    if parent.is_symlink() or not parent.is_dir():
        raise SourceTransportConfigError("source OAuth app config parent is unsafe")
    if not expanded.resolve(strict=False).is_relative_to(parent.resolve()):
        raise SourceTransportConfigError("source OAuth app config escapes its parent")
    if expanded.exists():
        mode = stat.S_IMODE(expanded.stat().st_mode)
        if not stat.S_ISREG(expanded.stat().st_mode):
            raise SourceTransportConfigError("source OAuth app config must be a regular file")
        if mode & 0o077:
            raise SourceTransportConfigError(
                "source OAuth app config permissions must be 0600 or stricter"
            )
    return expanded


def load_source_oauth_apps(path: Path | None = None) -> SourceOAuthAppConfigV1 | None:
    """Load a strictly validated local app registry without reading tokens."""
    config_path = _safe_source_oauth_config_path(path or _DEFAULT_SOURCE_OAUTH_APPS)
    if not config_path.exists():
        return None
    try:
        payload = config_path.read_text(encoding="utf-8")
        return SourceOAuthAppConfigV1.model_validate_json(payload, strict=True)
    except (OSError, ValueError, UnicodeError) as error:
        raise SourceTransportConfigError("source OAuth app config is malformed") from error


class SourceTransportRegistry:
    """Resolve provider/mode/config to offline, typed readiness only."""

    def __init__(self, *, config_path: Path | None = None) -> None:
        self.config_path = config_path or _DEFAULT_SOURCE_OAUTH_APPS

    def resolve(
        self,
        provider: SourceProvider | str,
        mode: SourceTransportMode | str,
        *,
        redirect_uri: str | None = None,
    ) -> SourceTransportReadiness:
        from autobrain.contracts import SourceSelectionV1

        provider = SourceProvider(provider)
        mode = SourceTransportMode(mode)
        try:
            SourceSelectionV1(provider=provider, mode=mode)
        except ValueError:
            return SourceTransportReadiness(
                provider=provider,
                mode=mode,
                state=TransportReadinessState.UNAVAILABLE,
                ready=False,
                governance_code=TransportGovernanceCode.SOURCE_TRANSPORT_UNAVAILABLE,
                detail="source transport mode is not registered for this provider",
                remediation="Select a registered offline source transport mode.",
            )

        if mode is SourceTransportMode.EXPORT_ARCHIVE:
            return SourceTransportReadiness(
                provider=provider,
                mode=mode,
                state=TransportReadinessState.READY,
                ready=True,
                governance_code=TransportGovernanceCode.SOURCE_TRANSPORT_READY,
                detail=(
                    "Slack export transport is available offline; "
                    "archive validation remains separate."
                ),
                remediation="",
            )

        if provider in {
            SourceProvider.CONFLUENCE,
            SourceProvider.GOOGLE_DRIVE,
            SourceProvider.SHAREPOINT,
        }:
            return SourceTransportReadiness(
                provider=provider,
                mode=mode,
                state=TransportReadinessState.UNAVAILABLE,
                ready=False,
                governance_code=TransportGovernanceCode.SOURCE_TRANSPORT_UNAVAILABLE,
                detail=(
                    f"{provider.value} transport is a design-partner/external gate "
                    "and is not implemented."
                ),
                remediation="Keep this source gated or use a sanitized local fixture.",
            )

        try:
            config = load_source_oauth_apps(self.config_path)
        except SourceTransportConfigError as error:
            code = (
                TransportGovernanceCode.SOURCE_OAUTH_APP_UNSAFE
                if "permissions" in str(error) or "symlink" in str(error) or "unsafe" in str(error)
                else TransportGovernanceCode.SOURCE_OAUTH_CONFIG_INVALID
            )
            return self._config_failure(provider, mode, code, str(error))
        if config is None:
            return self._config_failure(
                provider,
                mode,
                TransportGovernanceCode.SOURCE_OAUTH_APP_MISSING,
                "No local OAuth app configuration is present.",
            )
        app = next(
            (item for item in config.apps if item.provider is provider and item.mode is mode),
            None,
        )
        if app is None:
            return self._config_failure(
                provider,
                mode,
                TransportGovernanceCode.SOURCE_OAUTH_APP_MISSING,
                "No local OAuth app is configured for this provider and mode.",
            )
        if redirect_uri is not None and app.redirect_uri != redirect_uri:
            return self._config_failure(
                provider,
                mode,
                TransportGovernanceCode.SOURCE_REDIRECT_URI_MISMATCH,
                "Configured OAuth redirect URI does not match the requested redirect URI.",
            )
        if provider is SourceProvider.SLACK or provider is SourceProvider.NOTION:
            return SourceTransportReadiness(
                provider=provider,
                mode=mode,
                state=TransportReadinessState.AUTH_REQUIRED,
                ready=False,
                governance_code=TransportGovernanceCode.SOURCE_TRANSPORT_UNAVAILABLE,
                app_configured=True,
                detail=(
                    "OAuth app is configured, but no authenticated source connection is present."
                ),
                remediation="Authorize the source through the verified read-only OAuth flow.",
            )
        return SourceTransportReadiness(
            provider=provider,
            mode=mode,
            state=TransportReadinessState.UNAVAILABLE,
            ready=False,
            governance_code=TransportGovernanceCode.SOURCE_TRANSPORT_UNAVAILABLE,
            app_configured=True,
            detail=(
                f"{provider.value} transport is a design-partner/external gate "
                "and is not implemented."
            ),
            remediation="Keep this source gated or use a sanitized local fixture.",
        )

    @staticmethod
    def _config_failure(
        provider: SourceProvider,
        mode: SourceTransportMode,
        code: TransportGovernanceCode,
        detail: str,
    ) -> SourceTransportReadiness:
        return SourceTransportReadiness(
            provider=provider,
            mode=mode,
            state=TransportReadinessState.CONFIG_INVALID,
            ready=False,
            governance_code=code,
            detail=detail,
            remediation=(
                "Create a 0600 source-oauth-apps.json with a registered provider/mode "
                "and HTTPS redirect URI."
            ),
        )
