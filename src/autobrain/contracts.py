"""Strict, versioned source-transport and local projection contracts.

These contracts are intentionally offline: they describe what a caller may
request and what a local runtime can truthfully report.  They do not contain
credentials, corpus material, or evaluator-only fields.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import Field, StringConstraints, field_validator, model_validator

from autobrain.models import Sha256, StrictModel

PublicId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._:-]{0,127}$")]
OpaqueToken = Annotated[str, StringConstraints(min_length=8, max_length=512)]


class SourceProvider(StrEnum):
    SLACK = "slack"
    NOTION = "notion"
    CONFLUENCE = "confluence"
    GOOGLE_DRIVE = "google_drive"
    SHAREPOINT = "sharepoint"


class SourceTransportMode(StrEnum):
    EXPORT_ARCHIVE = "export_archive"
    LIVE_MCP_OAUTH = "live_mcp_oauth"
    OAUTH_3LO_REST = "oauth_3lo_rest"
    OAUTH_REST = "oauth_rest"
    GRAPH_OAUTH_PREVIEW = "graph_oauth_preview"


_ALLOWED_SOURCE_MODES: dict[SourceProvider, frozenset[SourceTransportMode]] = {
    SourceProvider.SLACK: frozenset(
        {SourceTransportMode.EXPORT_ARCHIVE, SourceTransportMode.LIVE_MCP_OAUTH}
    ),
    SourceProvider.NOTION: frozenset({SourceTransportMode.LIVE_MCP_OAUTH}),
    SourceProvider.CONFLUENCE: frozenset({SourceTransportMode.OAUTH_3LO_REST}),
    SourceProvider.GOOGLE_DRIVE: frozenset({SourceTransportMode.OAUTH_REST}),
    SourceProvider.SHAREPOINT: frozenset({SourceTransportMode.GRAPH_OAUTH_PREVIEW}),
}


class ModelAccessMode(StrEnum):
    PROVIDER_API_BYOK = "provider_api_byok"
    SUBSCRIPTION_CLI = "subscription_cli"
    LOCAL_OPENAI_COMPATIBLE = "local_openai_compatible"


class ModelCapabilityStatus(StrEnum):
    READY = "READY"
    UNAVAILABLE = "UNAVAILABLE"
    METERING_INCOMPLETE = "METERING_INCOMPLETE"
    NOT_CONFIGURED = "NOT_CONFIGURED"


class ModelCapability(StrEnum):
    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    RECOMMENDATION_VERIFIER = "recommendation_verifier"


class SourceConnectionState(StrEnum):
    REQUESTED = "REQUESTED"
    CLAIMED = "CLAIMED"
    AWAITING_LOCAL_INPUT = "AWAITING_LOCAL_INPUT"
    AUTHORIZING = "AUTHORIZING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class SourceOAuthAppV1(StrictModel):
    provider: SourceProvider
    mode: SourceTransportMode
    client_id: str = Field(min_length=1)
    client_secret_ref: str | None = Field(default=None, min_length=1)
    redirect_uri: str

    @field_validator("redirect_uri")
    @classmethod
    def redirect_is_safe_https(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("redirect_uri must be an HTTPS URL without credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("redirect_uri cannot contain query or fragment")
        return value

    @model_validator(mode="after")
    def provider_mode_is_registered(self) -> SourceOAuthAppV1:
        if self.mode not in _ALLOWED_SOURCE_MODES[self.provider]:
            raise ValueError("source transport mode is not allowed for provider")
        return self


class SourceOAuthAppConfigV1(StrictModel):
    schema_version: Literal[1]
    apps: list[SourceOAuthAppV1]

    @model_validator(mode="after")
    def app_keys_are_unique(self) -> SourceOAuthAppConfigV1:
        keys = [(app.provider, app.mode) for app in self.apps]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate source OAuth app")
        return self


class SourceSelectionV1(StrictModel):
    provider: SourceProvider
    mode: SourceTransportMode

    @model_validator(mode="after")
    def provider_mode_is_registered(self) -> SourceSelectionV1:
        if self.mode not in _ALLOWED_SOURCE_MODES[self.provider]:
            raise ValueError("source transport mode is not allowed for provider")
        return self


class SourceConnectionStatusProjectionV1(StrictModel):
    schema_version: Literal[1]
    request_id: UUID
    provider: SourceProvider
    mode: SourceTransportMode
    state: SourceConnectionState
    ready: bool
    credential_present: bool
    diagnostics: list[PublicId] = Field(default_factory=list)

    @model_validator(mode="after")
    def readiness_matches_state(self) -> SourceConnectionStatusProjectionV1:
        if self.ready != (self.state is SourceConnectionState.READY):
            raise ValueError("ready must be true only for READY state")
        if (
            self.ready
            and not self.credential_present
            and self.mode is not SourceTransportMode.EXPORT_ARCHIVE
        ):
            raise ValueError("ready authenticated source requires credential_present")
        return self


class ModelAccessProfileV1(StrictModel):
    """Truthful, secret-free capability status for one model access mode."""

    schema_version: Literal[1]
    mode: ModelAccessMode
    chat: ModelCapabilityStatus
    embeddings: ModelCapabilityStatus
    verifier: ModelCapabilityStatus
    metering: Literal["COST_COMPLETE", "COST_INCOMPLETE", "COST_UNAVAILABLE"]
    keyword_only: bool = False
    smoke_only_hash: bool = False
    recommendation_eligible: bool = False
    diagnostics: list[PublicId] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_capability_semantics(self) -> ModelAccessProfileV1:
        if (
            self.keyword_only
            and self.embeddings is not ModelCapabilityStatus.READY
            and self.recommendation_eligible
        ):
            raise ValueError("keyword-only mode cannot be recommendation eligible")
        if self.smoke_only_hash and self.recommendation_eligible:
            raise ValueError("smoke-only hash mode cannot be recommendation eligible")
        if self.recommendation_eligible and not all(
            capability is ModelCapabilityStatus.READY
            for capability in (self.chat, self.embeddings, self.verifier)
        ):
            raise ValueError(
                "recommendation eligibility requires ready chat, embeddings, and verifier"
            )
        return self


class ModelAccessStatusProjectionV1(StrictModel):
    schema_version: Literal[1]
    mode: ModelAccessMode
    chat_ready: bool
    embeddings_ready: bool
    verifier_ready: bool
    metering_status: Literal["COST_COMPLETE", "COST_INCOMPLETE", "COST_UNAVAILABLE"]
    recommendation_eligible: bool
    diagnostics: list[PublicId] = Field(default_factory=list)

    @model_validator(mode="after")
    def eligibility_requires_capabilities(self) -> ModelAccessStatusProjectionV1:
        if self.recommendation_eligible and not (
            self.chat_ready and self.embeddings_ready and self.verifier_ready
        ):
            raise ValueError("recommendation eligibility requires chat, embeddings, and verifier")
        return self


class CandidateModeSelectionV1(StrictModel):
    candidate_id: PublicId
    mode_id: PublicId


class JobSpecV1(StrictModel):
    schema_version: Literal[1]
    job_id: UUID
    run_id: UUID
    workspace_id: UUID
    runner_id: UUID
    issued_at: datetime
    expires_at: datetime
    nonce: OpaqueToken
    source_selections: list[SourceSelectionV1] = Field(min_length=1)
    candidate_modes: list[CandidateModeSelectionV1] = Field(min_length=1)
    required_model_capabilities: list[ModelCapability]
    budget_micro_usd: int = Field(ge=0)
    max_questions: int = Field(ge=1)
    include_dms: Literal[False] = False
    registry_snapshot_hash: Sha256
    projection_schema_version: Literal[1] = 1
    key_id: PublicId
    signature: OpaqueToken

    @model_validator(mode="after")
    def selections_are_unique_and_time_bounded(self) -> JobSpecV1:
        source_keys = [(selection.provider, selection.mode) for selection in self.source_selections]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("duplicate source selection")
        candidate_keys = [
            (selection.candidate_id, selection.mode_id) for selection in self.candidate_modes
        ]
        if len(candidate_keys) != len(set(candidate_keys)):
            raise ValueError("duplicate candidate mode selection")
        if self.expires_at <= self.issued_at:
            raise ValueError("JobSpec expiry must follow issue time")
        return self


_SCHEMA_MODELS = {
    "job-spec-v1.json": JobSpecV1,
    "model-access-profile-v1.json": ModelAccessProfileV1,
    "model-access-status-projection-v1.json": ModelAccessStatusProjectionV1,
    "source-connection-status-projection-v1.json": SourceConnectionStatusProjectionV1,
    "source-oauth-app-config-v1.json": SourceOAuthAppConfigV1,
}


def exported_json_schemas() -> dict[str, dict[str, object]]:
    return {name: model.model_json_schema() for name, model in sorted(_SCHEMA_MODELS.items())}


def json_schema_bytes(schema: dict[str, object]) -> bytes:
    return (json.dumps(schema, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("utf-8")


def export_json_schemas(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, schema in exported_json_schemas().items():
        (output_dir / filename).write_bytes(json_schema_bytes(schema))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    exporter = subparsers.add_parser("export-json-schemas")
    exporter.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "export-json-schemas":
        export_json_schemas(arguments.output_dir)


if __name__ == "__main__":
    main()
