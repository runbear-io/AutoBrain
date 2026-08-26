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

from autobrain.auth.models import AuthStatusReport, Provider
from autobrain.auth.service import ConnectionManager
from autobrain.candidates.gbrain_config import GBrainExecutionConfig, GBrainReadiness
from autobrain.contracts import SourceConnectionState, SourceConnectionStatusProjectionV1
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
    SubscriptionStatus,
    provider_registry,
)
from autobrain.tui_effects import login_command


class RefreshableScreen(Protocol):
    def refresh(self) -> None: ...


class ConnectionStatusProvider(Protocol):
    def status(self) -> AuthStatusReport: ...


class SubscriptionStatusProvider(Protocol):
    def status(self) -> SubscriptionStatus: ...


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
    source_projections: dict[Provider, SourceConnectionStatusProjectionV1] | None = None
    slack_export_path: Path | None = None
    slack_export_sha256: str | None = None


def connection_snapshot(
    *,
    manager: ConnectionStatusProvider | None = None,
    subscription_client: SubscriptionStatusProvider | None = None,
    subscription_provider: ProviderId = ProviderId.CODEX,
    refresh_subscription: bool = False,
    source_store: SlackSourceStore | None = None,
) -> ConnectionSnapshot:
    paths = AutoBrainPaths.from_home()
    report = (manager or ConnectionManager(paths.root)).status()
    auth_report = report.with_projections()
    projections: dict[Provider, SourceConnectionStatusProjectionV1] = {
        Provider(item.provider.value): item for item in auth_report.projections
    }
    source_states: dict[Provider, ConnectionState] = {
        provider: _legacy_connection_state(projection.state)
        for provider, projection in projections.items()
    }
    subscription = (
        subscription_client.status()
        if subscription_client is not None
        else provider_registry().probe(subscription_provider, refresh=refresh_subscription).status
    )
    slack_status = (source_store or SlackSourceStore(paths.sources)).status()
    projections[Provider.SLACK] = slack_status.projection
    source_states[Provider.SLACK] = _legacy_connection_state(slack_status.projection.state)
    embeddings = inspect_embedding_backend(os.environ)
    source_details: dict[Provider, str] = {
        Provider.SLACK: "export ready" if slack_status.projection.ready else slack_status.detail
    }
    return ConnectionSnapshot(
        subscription_provider=subscription_provider,
        subscription=subscription,
        embeddings=embeddings,
        sources=source_states,
        source_details=source_details,
        source_projections=projections,
        slack_export_path=(slack_status.archive_path if slack_status.projection.ready else None),
        slack_export_sha256=(
            slack_status.config.archive_sha256
            if slack_status.projection.ready and slack_status.config is not None
            else None
        ),
    )


def _legacy_connection_state(state: SourceConnectionState) -> ConnectionState:
    if state is SourceConnectionState.READY:
        return ConnectionState.CONNECTED
    if state is SourceConnectionState.EXPIRED:
        return ConnectionState.EXPIRED
    if state is SourceConnectionState.AWAITING_LOCAL_INPUT:
        return ConnectionState.DISCONNECTED
    return ConnectionState.REAUTHORIZATION_REQUIRED


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
    gbrain_config: GBrainExecutionConfig | None = None,
    gbrain_readiness: GBrainReadiness | None = None,
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
                gbrain_config=gbrain_config,
                gbrain_readiness=gbrain_readiness,
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
            gbrain_config=plan.gbrain_config,
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
