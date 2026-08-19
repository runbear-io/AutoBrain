"""Runtime boundaries used by the AutoBrain terminal cockpit."""

from __future__ import annotations

import curses
import queue
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from autobrain.auth.models import Provider
from autobrain.auth.service import ConnectionManager
from autobrain.experiment import ExperimentPlan, ExperimentSetupError, build_automatic_plan
from autobrain.models import CandidateId, ConnectionState
from autobrain.orchestration import RunConfig, RunOrchestrator, RunResult
from autobrain.paths import AutoBrainPaths
from autobrain.source_store import SlackSourceStore
from autobrain.subscription import (
    CodexSubscriptionClient,
    CodexSubscriptionConfig,
    SubscriptionStatus,
)


@dataclass(frozen=True)
class ConnectionSnapshot:
    subscription: SubscriptionStatus
    sources: dict[Provider, ConnectionState]
    source_details: dict[Provider, str] | None = None
    slack_export_path: Path | None = None
    slack_export_sha256: str | None = None


class ConnectionFlowRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]: ...


def _subprocess_runner(
    command: Sequence[str],
    *,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True)


def connection_snapshot(
    *,
    manager: ConnectionManager | None = None,
    subscription_client: CodexSubscriptionClient | None = None,
    source_store: SlackSourceStore | None = None,
) -> ConnectionSnapshot:
    paths = AutoBrainPaths.from_home()
    report = (manager or ConnectionManager(paths.root)).status()
    source_states = {item.provider: item.state for item in report.connections}
    subscription = (
        subscription_client or CodexSubscriptionClient(CodexSubscriptionConfig.from_environ())
    ).status()
    slack_status = (source_store or SlackSourceStore(paths.sources)).status()
    source_details: dict[Provider, str] = {}
    if slack_status.ready and slack_status.config is not None:
        source_states[Provider.SLACK] = ConnectionState.CONNECTED
        source_details[Provider.SLACK] = "export ready"
        return ConnectionSnapshot(
            subscription=subscription,
            sources=source_states,
            source_details=source_details,
            slack_export_path=slack_status.archive_path,
            slack_export_sha256=slack_status.config.archive_sha256,
        )
    return ConnectionSnapshot(
        subscription=subscription,
        sources=source_states,
        source_details=source_details,
    )


def resolve_plan(
    *,
    selected_sources: tuple[Provider, ...],
    selected_candidates: tuple[CandidateId, ...],
    connections: ConnectionSnapshot,
) -> tuple[ExperimentPlan | None, str]:
    disconnected = [
        provider.value
        for provider in selected_sources
        if connections.sources.get(provider) is not ConnectionState.CONNECTED
    ]
    if disconnected:
        return None, "SOURCE_AUTH_UNAVAILABLE: connect " + ", ".join(disconnected)
    try:
        return (
            build_automatic_plan(
                sources=selected_sources,
                candidates=selected_candidates,
                subscription_status=connections.subscription,
            ),
            "",
        )
    except ExperimentSetupError as exc:
        return None, str(exc)


def execute_plan(
    plan: ExperimentPlan,
    result_queue: queue.Queue[RunResult | BaseException],
) -> None:
    try:
        paths = AutoBrainPaths.from_home()
        slack_status = SlackSourceStore(paths.sources).status()
        slack_export_selected = Provider.SLACK in plan.sources and slack_status.ready
        config = RunConfig(
            budget_usd=plan.budget_usd,
            max_questions=plan.max_questions,
            open_report=False,
            provider_mode=plan.provider_mode,
            selected_sources=plan.sources,
            selected_candidates=plan.candidates,
            slack_export_path=(slack_status.archive_path if slack_export_selected else None),
            slack_export_sha256=(
                slack_status.config.archive_sha256
                if slack_export_selected and slack_status.config is not None
                else None
            ),
            experiment_title=plan.title,
            experiment_description=plan.description,
        )
        result_queue.put(RunOrchestrator.local(config).run())
    except BaseException as exc:
        result_queue.put(exc)


def run_connection_flow(
    screen: curses.window,
    provider: Provider | Literal["subscription"],
    *,
    runner: ConnectionFlowRunner = _subprocess_runner,
) -> None:
    if provider is Provider.SLACK:
        command = [sys.executable, "-m", "autobrain.cli", "source", "slack"]
    elif isinstance(provider, Provider):
        command = [sys.executable, "-m", "autobrain.cli", "auth", provider.value]
    else:
        command = [sys.executable, "-m", "autobrain.cli", "subscription", "setup"]
    curses.def_prog_mode()
    curses.endwin()
    runner(command, check=False)
    curses.reset_prog_mode()
    screen.refresh()


def connection_key(key: int) -> Provider | Literal["subscription"] | None:
    if key in {ord("c"), ord("C")}:
        return "subscription"
    if key in {ord("s"), ord("S")}:
        return Provider.SLACK
    if key in {ord("n"), ord("N")}:
        return Provider.NOTION
    return None
