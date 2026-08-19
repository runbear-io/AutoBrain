"""Runtime boundaries used by the AutoBrain terminal cockpit."""

from __future__ import annotations

import curses
import queue
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

from autobrain.auth.models import Provider
from autobrain.auth.service import ConnectionManager
from autobrain.experiment import ExperimentPlan, ExperimentSetupError, build_automatic_plan
from autobrain.models import CandidateId, ConnectionState
from autobrain.orchestration import RunConfig, RunOrchestrator, RunResult
from autobrain.paths import AutoBrainPaths
from autobrain.subscription import (
    CodexSubscriptionClient,
    CodexSubscriptionConfig,
    SubscriptionStatus,
)


@dataclass(frozen=True)
class ConnectionSnapshot:
    subscription: SubscriptionStatus
    sources: dict[Provider, ConnectionState]


def connection_snapshot() -> ConnectionSnapshot:
    paths = AutoBrainPaths.from_home()
    report = ConnectionManager(paths.root).status()
    source_states = {item.provider: item.state for item in report.connections}
    subscription = CodexSubscriptionClient(CodexSubscriptionConfig.from_environ()).status()
    return ConnectionSnapshot(subscription=subscription, sources=source_states)


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
        config = RunConfig(
            budget_usd=plan.budget_usd,
            max_questions=plan.max_questions,
            open_report=False,
            provider_mode=plan.provider_mode,
            selected_sources=plan.sources,
            selected_candidates=plan.candidates,
            experiment_title=plan.title,
            experiment_description=plan.description,
        )
        result_queue.put(RunOrchestrator.local(config).run())
    except BaseException as exc:
        result_queue.put(exc)


def run_connection_flow(
    screen: curses.window,
    provider: Provider | Literal["subscription"],
) -> None:
    if isinstance(provider, Provider):
        command = [sys.executable, "-m", "autobrain.cli", "auth", provider.value]
    else:
        command = [sys.executable, "-m", "autobrain.cli", "subscription", "setup"]
    curses.def_prog_mode()
    curses.endwin()
    subprocess.run(command, check=False)
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
