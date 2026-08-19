"""Confined local source references for user-provided knowledge exports."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from autobrain.connectors.slack_export import inspect_slack_export
from autobrain.connectors.slack_export_archive import archive_sha256
from autobrain.connectors.slack_export_types import (
    SlackExportError,
    SlackExportSummary,
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
    archive_path: Path | None = None
    config: SlackSourceConfig | None = None


class SlackSourceStore:
    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root
        self.config_path = source_root / "slack-export.json"

    def configure_export(self, archive_path: Path) -> SlackSourceConfig:
        resolved = archive_path.expanduser().resolve()
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
            return SlackSourceStatus(
                state=SlackSourceState.INVALID_CONFIG,
                ready=False,
                detail=str(error),
            )
        if config is None:
            return SlackSourceStatus(
                state=SlackSourceState.NOT_CONFIGURED,
                ready=False,
                detail="No Slack export is configured",
            )
        archive_path = Path(config.archive_path)
        if not archive_path.is_file():
            return SlackSourceStatus(
                state=SlackSourceState.ARCHIVE_MISSING,
                ready=False,
                detail="Configured Slack export no longer exists",
                archive_path=archive_path,
                config=config,
            )
        try:
            actual_sha256 = archive_sha256(archive_path)
        except OSError as error:
            return SlackSourceStatus(
                state=SlackSourceState.ARCHIVE_MISSING,
                ready=False,
                detail=f"Configured Slack export cannot be read: {error}",
                archive_path=archive_path,
                config=config,
            )
        if actual_sha256 != config.archive_sha256:
            return SlackSourceStatus(
                state=SlackSourceState.ARCHIVE_CHANGED,
                ready=False,
                detail="Configured Slack export changed; configure it again",
                archive_path=archive_path,
                config=config,
            )
        return SlackSourceStatus(
            state=SlackSourceState.READY,
            ready=True,
            detail=(
                f"{config.summary.message_count} messages from "
                f"{config.summary.channel_count} channels"
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
