"""Framework-neutral runtime boundaries used by terminal UI hosts."""

from __future__ import annotations

import os
import queue
import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from autobrain.auth.models import Provider
from autobrain.auth.service import ConnectionManager
from autobrain.embedding import EmbeddingReadiness, inspect_embedding_backend
from autobrain.experiment import ExperimentPlan, ExperimentSetupError, build_automatic_plan
from autobrain.models import CandidateId, ConnectionState
from autobrain.orchestration import (
    RunCancellation,
    RunConfig,
    RunOrchestrator,
    RunResult,
    StageEventSink,
)
from autobrain.paths import AutoBrainPaths
from autobrain.source_store import SlackSourceStore
from autobrain.subscription import (
    ProviderId,
    SubscriptionProvider,
    SubscriptionStatus,
    provider_registry,
)
from autobrain.tui_effects import login_command


class RefreshableScreen(Protocol):
    def refresh(self) -> None: ...


class ConnectionFlowRunner(Protocol):
    def __call__(
        self, command: Sequence[str], *, check: bool
    ) -> subprocess.CompletedProcess[str]: ...


def _subprocess_runner(command: Sequence[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True)


@dataclass(frozen=True)
class ConnectionSnapshot:
    subscription: SubscriptionStatus
    embeddings: EmbeddingReadiness
    sources: dict[Provider, ConnectionState]
    subscription_provider: ProviderId = ProviderId.CODEX
    source_details: dict[Provider, str] | None = None
    slack_export_path: Path | None = None
    slack_export_sha256: str | None = None


def connection_snapshot(
    *,
    manager: ConnectionManager | None = None,
    subscription_client: SubscriptionProvider | None = None,
    subscription_provider: ProviderId = ProviderId.CODEX,
    refresh_subscription: bool = False,
    source_store: SlackSourceStore | None = None,
) -> ConnectionSnapshot:
    paths = AutoBrainPaths.from_home()
    report = (manager or ConnectionManager(paths.root)).status()
    source_states = {item.provider: item.state for item in report.connections}
    subscription = (
        subscription_client.status()
        if subscription_client is not None
        else provider_registry().probe(subscription_provider, refresh=refresh_subscription).status
    )
    slack_status = (source_store or SlackSourceStore(paths.sources)).status()
    embeddings = inspect_embedding_backend(os.environ)
    source_details: dict[Provider, str] = {}
    if slack_status.ready and slack_status.config is not None:
        source_states[Provider.SLACK] = ConnectionState.CONNECTED
        source_details[Provider.SLACK] = "export ready"
        return ConnectionSnapshot(
            subscription_provider=subscription_provider,
            subscription=subscription,
            embeddings=embeddings,
            sources=source_states,
            source_details=source_details,
            slack_export_path=slack_status.archive_path,
            slack_export_sha256=slack_status.config.archive_sha256,
        )
    return ConnectionSnapshot(
        subscription_provider=subscription_provider,
        subscription=subscription,
        embeddings=embeddings,
        sources=source_states,
        source_details=source_details,
    )


def run_connection_flow(
    screen: RefreshableScreen,
    provider: Provider | ProviderId,
    *,
    runner: ConnectionFlowRunner = _subprocess_runner,
) -> None:
    """Compatibility command boundary without terminal-framework ownership."""
    runner(login_command(provider), check=False)


def resolve_plan(
    *,
    selected_sources: tuple[Provider, ...],
    selected_candidates: tuple[CandidateId, ...],
    connections: ConnectionSnapshot,
    subscription_provider: ProviderId = ProviderId.CODEX,
) -> tuple[ExperimentPlan | None, str]:
    if connections.subscription_provider is not subscription_provider:
        return None, "SUBSCRIPTION_PROVIDER_MISMATCH: refresh the selected provider"
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
                embedding_readiness=connections.embeddings,
                subscription_provider=subscription_provider,
            ),
            "",
        )
    except ExperimentSetupError as exc:
        return None, str(exc)


def execute_plan(
    plan: ExperimentPlan,
    result_queue: queue.Queue[RunResult | BaseException],
    subscription_provider: ProviderId = ProviderId.CODEX,
    stage_event_sink: StageEventSink | None = None,
    cancellation: RunCancellation | None = None,
) -> None:
    try:
        expected_mode = f"{subscription_provider.value}-subscription"
        if plan.provider_mode != expected_mode:
            raise ValueError(
                "SUBSCRIPTION_PROVIDER_MISMATCH: plan does not match selected provider"
            )
        paths = AutoBrainPaths.from_home()
        slack_status = SlackSourceStore(paths.sources).status()
        slack_export_selected = Provider.SLACK in plan.sources and slack_status.ready
        config = RunConfig(
            budget_usd=plan.budget_usd,
            max_questions=plan.max_questions,
            open_report=False,
            provider_mode=plan.provider_mode,
            embedding_backend=plan.embedding_backend,
            selected_sources=plan.sources,
            selected_candidates=plan.candidates,
            slack_export_path=slack_status.archive_path if slack_export_selected else None,
            slack_export_sha256=(
                slack_status.config.archive_sha256
                if slack_export_selected and slack_status.config is not None
                else None
            ),
            experiment_title=plan.title,
            experiment_description=plan.description,
        )
        result_queue.put(
            RunOrchestrator.local(
                config,
                stage_event_sink=stage_event_sink,
                cancellation=cancellation,
            ).run()
        )
    except BaseException as exc:
        result_queue.put(exc)


@dataclass(frozen=True)
class PlanWorker:
    thread: threading.Thread
    cancellation: RunCancellation

    def cancel_and_join(self, *, timeout: float) -> bool:
        self.cancellation.cancel()
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()

    def join(self, *, timeout: float) -> bool:
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()


def start_plan_worker(
    plan: ExperimentPlan,
    result_queue: queue.Queue[RunResult | BaseException],
    stage_event_sink: StageEventSink | None = None,
    subscription_provider: ProviderId = ProviderId.CODEX,
) -> PlanWorker:
    cancellation = RunCancellation()
    thread = threading.Thread(
        target=execute_plan,
        args=(plan, result_queue, subscription_provider, stage_event_sink, cancellation),
    )
    thread.start()
    return PlanWorker(thread=thread, cancellation=cancellation)
