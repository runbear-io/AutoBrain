"""Confined local source references for user-provided knowledge exports."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from autobrain.connectors.slack_export import inspect_slack_export
from autobrain.connectors.slack_export_archive import archive_sha256
from autobrain.connectors.slack_export_types import (
    SlackExportError,
    SlackExportSummary,
)
from autobrain.contracts import (
    SourceConnectionState,
    SourceConnectionStatusProjectionV1,
    SourceProvider,
    SourceTransportMode,
)
from autobrain.models import StrictModel


class SlackSourceState(StrEnum):
    NOT_CONFIGURED = "NOT_CONFIGURED"
    READY = "READY"
    ARCHIVE_MISSING = "ARCHIVE_MISSING"
    ARCHIVE_CHANGED = "ARCHIVE_CHANGED"
    INVALID_CONFIG = "INVALID_CONFIG"


class SlackSourceConfig(StrictModel):
    archive_path: str
    archive_sha256: str
    configured_at: datetime
    summary: SlackExportSummary


class SlackSourceStatus(StrictModel):
    state: SlackSourceState
    ready: bool
    detail: str
    projection: SourceConnectionStatusProjectionV1
    archive_path: Path | None = None
    config: SlackSourceConfig | None = None


class SlackSourceStore:
    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root
        self.config_path = source_root / "slack-export.json"

    def configure_export(self, archive_path: Path) -> SlackSourceConfig:
        archive_path = archive_path.expanduser()
        if archive_path.is_symlink():
            raise SlackExportError("Slack export input cannot be a symlink")
        resolved = archive_path.resolve()
        summary = inspect_slack_export(resolved)
        config = SlackSourceConfig(
            archive_path=str(resolved),
            archive_sha256=summary.archive_sha256,
            configured_at=datetime.now(tz=UTC),
            summary=summary,
        )
        self._write_config(config)
        return config

    def load(self) -> SlackSourceConfig | None:
        if not self.config_path.exists():
            return None
        if self.config_path.is_symlink():
            raise SlackExportError("Slack source config cannot be a symlink")
        try:
            return SlackSourceConfig.model_validate_json(
                self.config_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise SlackExportError("invalid Slack source configuration") from error

    def status(self) -> SlackSourceStatus:
        try:
            config = self.load()
        except SlackExportError as error:
            return self._status(
                state=SlackSourceState.INVALID_CONFIG,
                projection_state=SourceConnectionState.FAILED,
                diagnostic="archive_config_invalid",
                detail=str(error),
            )
        if config is None:
            return self._status(
                state=SlackSourceState.NOT_CONFIGURED,
                projection_state=SourceConnectionState.AWAITING_LOCAL_INPUT,
                diagnostic="archive_not_configured",
                detail="No Slack export is configured",
            )
        archive_path = Path(config.archive_path)
        if archive_path.is_symlink():
            return self._status(
                state=SlackSourceState.INVALID_CONFIG,
                projection_state=SourceConnectionState.FAILED,
                diagnostic="archive_symlink",
                detail="Configured Slack export cannot be a symlink",
                archive_path=archive_path,
                config=config,
            )
        if not archive_path.is_file():
            return self._status(
                state=SlackSourceState.ARCHIVE_MISSING,
                projection_state=SourceConnectionState.FAILED,
                diagnostic="archive_missing",
                detail="Configured Slack export no longer exists",
                archive_path=archive_path,
                config=config,
            )
        try:
            actual_sha256 = archive_sha256(archive_path)
        except OSError as error:
            return self._status(
                state=SlackSourceState.ARCHIVE_MISSING,
                projection_state=SourceConnectionState.FAILED,
                diagnostic="archive_missing",
                detail=f"Configured Slack export cannot be read: {error}",
                archive_path=archive_path,
                config=config,
            )
        if actual_sha256 != config.archive_sha256:
            return self._status(
                state=SlackSourceState.ARCHIVE_CHANGED,
                projection_state=SourceConnectionState.FAILED,
                diagnostic="archive_changed",
                detail="Configured Slack export changed; configure it again",
                archive_path=archive_path,
                config=config,
            )
        return self._status(
            state=SlackSourceState.READY,
            projection_state=SourceConnectionState.READY,
            diagnostic="archive_valid",
            detail=(
                f"{config.summary.message_count} messages from "
                f"{config.summary.channel_count} channels"
            ),
            archive_path=archive_path,
            config=config,
        )

    @staticmethod
    def _status(
        *,
        state: SlackSourceState,
        projection_state: SourceConnectionState,
        diagnostic: str,
        detail: str,
        archive_path: Path | None = None,
        config: SlackSourceConfig | None = None,
    ) -> SlackSourceStatus:
        return SlackSourceStatus(
            state=state,
            ready=projection_state is SourceConnectionState.READY,
            detail=detail,
            projection=SourceConnectionStatusProjectionV1(
                schema_version=1,
                request_id=uuid4(),
                provider=SourceProvider.SLACK,
                mode=SourceTransportMode.EXPORT_ARCHIVE,
                state=projection_state,
                ready=projection_state is SourceConnectionState.READY,
                credential_present=False,
                diagnostics=[diagnostic],
            ),
            archive_path=archive_path,
            config=config,
        )

    def remove(self) -> None:
        if self.config_path.is_symlink():
            raise SlackExportError("Slack source config cannot be a symlink")
        self.config_path.unlink(missing_ok=True)

    def _write_config(self, config: SlackSourceConfig) -> None:
        self.source_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.source_root.is_symlink():
            raise SlackExportError("Slack source directory cannot be a symlink")
        payload = json.dumps(
            config.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.source_root,
            prefix=".slack-export-",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as target:
                target.write(payload)
                target.write("\n")
            temporary_path.replace(self.config_path)
            self.config_path.chmod(0o600)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
