"""Slack source readiness for AutoBrain doctor."""

from __future__ import annotations

from autobrain.models import CheckResult, Status
from autobrain.paths import AutoBrainPaths
from autobrain.secrets import EnvironmentReadiness
from autobrain.source_store import SlackSourceStore


def check_slack_source(
    *,
    paths: AutoBrainPaths,
    readiness: EnvironmentReadiness,
) -> CheckResult:
    status = SlackSourceStore(paths.sources).status()
    if status.ready:
        return CheckResult(name="slack_source", status=Status.OK, detail="export ready")
    if readiness.slack_client_id and readiness.slack_client_secret:
        return CheckResult(
            name="slack_source",
            status=Status.OK,
            detail="live MCP credentials configured",
        )
    return CheckResult(
        name="slack_source",
        status=Status.MCP_AUTH_UNAVAILABLE,
        detail="configure a Slack export ZIP or Slack app credentials",
    )
