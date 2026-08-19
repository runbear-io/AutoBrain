"""Typed results for local Slack export ingestion."""

from __future__ import annotations

from pydantic import Field

from autobrain.connectors.slack import RawSlackDocument
from autobrain.models import Status, StrictModel


class SlackExportError(ValueError):
    """Raised when a Slack export is unsafe or unsupported."""


class SlackExportSummary(StrictModel):
    archive_path: str
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    workspace_id: str | None = None
    workspace_name: str | None = None
    workspace_domain: str | None = None
    channel_count: int = Field(ge=0)
    user_count: int = Field(ge=0)
    message_count: int = Field(ge=0)
    file_link_count: int = Field(ge=0)


class SlackExportCrawlResult(StrictModel):
    status: Status
    documents: tuple[RawSlackDocument, ...]
    coverage: dict[str, str | int]
    summary: SlackExportSummary
