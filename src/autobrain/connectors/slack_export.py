"""Public connector for local official Slack Workspace Export ZIP files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autobrain.connectors.slack_export_archive import archive_sha256, parse_slack_export
from autobrain.connectors.slack_export_types import (
    SlackExportCrawlResult,
    SlackExportError,
    SlackExportSourceChangedError,
    SlackExportSummary,
)
from autobrain.models import Status

__all__ = [
    "SlackExportConnector",
    "SlackExportCrawlResult",
    "SlackExportError",
    "SlackExportSourceChangedError",
    "SlackExportSummary",
    "inspect_slack_export",
]


class SlackExportConnector:
    def __init__(self, archive_path: Path, *, expected_sha256: str | None = None) -> None:
        archive_path = archive_path.expanduser()
        if archive_path.is_symlink():
            raise SlackExportError("Slack export input cannot be a symlink")
        self.archive_path = archive_path.resolve()
        self.expected_sha256 = expected_sha256
        self._cached: tuple[SlackExportSummary, tuple[Any, ...]] | None = None

    async def probe(self) -> dict[str, str | int]:
        summary, documents = parse_slack_export(self.archive_path)
        self._cached = (summary, documents)
        self._verify_hash(summary.archive_sha256)
        return {
            "status": Status.OK.value,
            "archive_sha256": summary.archive_sha256,
            "messages": summary.message_count,
        }

    async def crawl(self) -> SlackExportCrawlResult:
        cached = self._cached
        if cached is None:
            summary, documents = parse_slack_export(self.archive_path)
        else:
            summary, documents = cached
            self._verify_hash(archive_sha256(self.archive_path))
        self._verify_hash(summary.archive_sha256)
        return SlackExportCrawlResult(
            status=Status.OK,
            documents=documents,
            coverage={
                "source": "slack-export",
                "completeness": "EXHAUSTIVE",
                "discovered": summary.message_count,
                "fetched": summary.message_count,
                "archive_sha256": summary.archive_sha256,
            },
            summary=summary,
        )

    def _verify_hash(self, actual_sha256: str) -> None:
        if self.expected_sha256 is not None and actual_sha256 != self.expected_sha256:
            raise SlackExportError("Slack export changed after it was configured")


def inspect_slack_export(path: Path) -> SlackExportSummary:
    summary, _ = parse_slack_export(path.expanduser())
    return summary
