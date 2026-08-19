"""Public connector for local official Slack Workspace Export ZIP files."""

from __future__ import annotations

from pathlib import Path

from autobrain.connectors.slack_export_archive import parse_slack_export
from autobrain.connectors.slack_export_types import (
    SlackExportCrawlResult,
    SlackExportError,
    SlackExportSummary,
)
from autobrain.models import Status

__all__ = [
    "SlackExportConnector",
    "SlackExportCrawlResult",
    "SlackExportError",
    "SlackExportSummary",
    "inspect_slack_export",
]


class SlackExportConnector:
    def __init__(self, archive_path: Path, *, expected_sha256: str | None = None) -> None:
        self.archive_path = archive_path.expanduser().resolve()
        self.expected_sha256 = expected_sha256

    async def probe(self) -> dict[str, str | int]:
        summary = inspect_slack_export(self.archive_path)
        self._verify_hash(summary.archive_sha256)
        return {
            "status": Status.OK.value,
            "archive_sha256": summary.archive_sha256,
            "messages": summary.message_count,
        }

    async def crawl(self) -> SlackExportCrawlResult:
        summary, documents = parse_slack_export(self.archive_path)
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
    summary, _ = parse_slack_export(path.expanduser().resolve())
    return summary
