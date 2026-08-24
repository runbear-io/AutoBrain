from __future__ import annotations

from pathlib import Path

import pytest

from autobrain.auth.models import AuthStatusReport, ConnectionStatus, Provider
from autobrain.embedding import EmbeddingBackendConfig, EmbeddingReadiness
from autobrain.experiment import ExperimentSetupError, build_automatic_plan
from autobrain.models import CandidateId, ConnectionState, Status
from autobrain.orchestration import (
    CandidateContext,
    CandidateOutcome,
    ConnectorSnapshot,
    RunConfig,
    RunOrchestrator,
    retain_selected_candidates,
)
from autobrain.production import build_production_candidates
from autobrain.subscription import SubscriptionStatus
from autobrain.tui_runtime import ConnectionSnapshot, resolve_plan


def _semantic_readiness() -> EmbeddingReadiness:
    return EmbeddingBackendConfig.from_environ(
        {"OPENAI_API_KEY": "fixture-embedding-key"},
        requested="openai",
    ).readiness()


def _smoke_readiness() -> EmbeddingReadiness:
    return EmbeddingBackendConfig.from_environ({}, requested="local-hash").readiness()


class FakeConnector:
    provider = Provider.SLACK.value

    def probe(self) -> dict[str, object]:
        return {"ok": True}

    def crawl(self, *, include_dms: bool) -> ConnectorSnapshot:
        del include_dms
        return ConnectorSnapshot(provider=self.provider, documents=(), coverage={})


class FakeConnectionManager:
    def status(self) -> AuthStatusReport:
        return AuthStatusReport(
            connections=(
                ConnectionStatus(
                    provider=Provider.SLACK,
                    state=ConnectionState.CONNECTED,
                    status=Status.OK,
                ),
                ConnectionStatus(
                    provider=Provider.NOTION,
                    state=ConnectionState.DISCONNECTED,
                    status=Status.MCP_AUTH_UNAVAILABLE,
                ),
            )
        )


class TrackingConnectionManager:
    def __init__(self, *, connected: bool) -> None:
        self.connected = connected
        self.status_calls = 0

    def status(self) -> AuthStatusReport:
        self.status_calls += 1
        return AuthStatusReport(
            connections=(
                ConnectionStatus(
                    provider=Provider.SLACK,
                    state=(
                        ConnectionState.CONNECTED
                        if self.connected
                        else ConnectionState.DISCONNECTED
                    ),
                    status=Status.OK if self.connected else Status.MCP_AUTH_UNAVAILABLE,
                ),
            )
        )


def test_selected_source_does_not_require_unselected_connection(tmp_path: Path) -> None:
    orchestrator = RunOrchestrator.local(
        RunConfig(
            output=tmp_path / "runs",
            open_report=False,
            selected_sources=(Provider.SLACK,),
        ),
        connection_manager=FakeConnectionManager(),  # type: ignore[arg-type]
        connector_builder=lambda _manager, _include_dms: (FakeConnector(),),
        api_key="provider-key",
    )

    assert orchestrator.provider_available is True
    assert [connector.provider for connector in orchestrator.connectors] == ["slack"]


def test_source_auth_unavailable_precedes_missing_semantic_embedding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    manager = TrackingConnectionManager(connected=False)
    builders: list[str] = []

    def connector_builder(*_args: object, **_kwargs: object):
        builders.append("connector")
        raise AssertionError("source auth failure must not construct connector clients")

    def candidate_builder(*_args: object, **_kwargs: object):
        builders.append("candidate")
        raise AssertionError("source auth failure must not construct candidate clients")

    orchestrator = RunOrchestrator.local(
        RunConfig(
            output=tmp_path / "runs",
            open_report=False,
            provider_mode="codex-subscription",
            embedding_backend="openai",
            selected_sources=(Provider.SLACK,),
        ),
        connection_manager=manager,  # type: ignore[arg-type]
        connector_builder=connector_builder,
        candidate_builder=candidate_builder,
    )
    result = orchestrator.run()

    assert manager.status_calls == 1
    assert builders == []
    assert orchestrator.provider_detail == "MCP_AUTH_UNAVAILABLE: slack"
    assert result.status is Status.MCP_AUTH_UNAVAILABLE


def test_missing_semantic_embedding_is_checked_after_source_auth_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    manager = TrackingConnectionManager(connected=True)
    builders: list[str] = []

    def connector_builder(*_args: object, **_kwargs: object):
        builders.append("connector")
        raise AssertionError("embedding failure must stop before connector construction")

    def candidate_builder(*_args: object, **_kwargs: object):
        builders.append("candidate")
        raise AssertionError("embedding failure must stop before candidate construction")

    orchestrator = RunOrchestrator.local(
        RunConfig(
            output=tmp_path / "runs",
            open_report=False,
            provider_mode="codex-subscription",
            embedding_backend="openai",
            selected_sources=(Provider.SLACK,),
        ),
        connection_manager=manager,  # type: ignore[arg-type]
        connector_builder=connector_builder,
        candidate_builder=candidate_builder,
    )
    result = orchestrator.run()

    assert manager.status_calls == 1
    assert builders == []
    assert "SEMANTIC_EMBEDDING_UNAVAILABLE" in orchestrator.provider_detail
    assert result.status is Status.MISSING_PROVIDER


def test_missing_semantic_embedding_config_stops_before_ingestion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    candidate_builder_called = False

    def candidate_builder(*_args: object, **_kwargs: object):
        nonlocal candidate_builder_called
        candidate_builder_called = True
        raise AssertionError("candidate construction must not begin")

    orchestrator = RunOrchestrator.local(
        RunConfig(
            output=tmp_path / "runs",
            open_report=False,
            provider_mode="codex-subscription",
            embedding_backend="openai",
            selected_sources=(Provider.SLACK,),
        ),
        connection_manager=FakeConnectionManager(),  # type: ignore[arg-type]
        connector_builder=lambda _manager, _include_dms: (FakeConnector(),),
        candidate_builder=candidate_builder,
    )
    result = orchestrator.run()

    assert orchestrator.provider_available is False
    assert "SEMANTIC_EMBEDDING_UNAVAILABLE" in orchestrator.provider_detail
    assert candidate_builder_called is False
    assert result.status is Status.MISSING_PROVIDER
    assert result.candidate_results == ()


def test_production_builder_creates_only_selected_candidates(tmp_path: Path) -> None:
    candidates = build_production_candidates(
        tmp_path / "run",
        api_key="provider-key",
        candidate_ids=(CandidateId.LLM_WIKI, CandidateId.MEM0),
        provider_upstream=lambda _payload: {},
    )
    try:
        assert [candidate.candidate_id for candidate in candidates] == [
            CandidateId.LLM_WIKI.value,
            CandidateId.MEM0.value,
        ]
    finally:
        for candidate in candidates:
            candidate.cleanup()


def test_automatic_plan_owns_questions_budget_and_first_experiment() -> None:
    plan = build_automatic_plan(
        sources=(Provider.SLACK, Provider.NOTION),
        candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0, CandidateId.GBRAIN),
        subscription_status=SubscriptionStatus.READY,
        embedding_readiness=_semantic_readiness(),
    )

    assert plan.provider_mode == "codex-subscription"
    assert plan.max_questions == 30
    assert plan.budget_usd == 25.0
    assert plan.title == "Find the best knowledge system for Slack + Notion"
    assert plan.description == (
        "Compare LLM Wiki, Mem0 OSS, GBrain on grounded questions from Slack + Notion."
    )


def test_legacy_candidate_builder_cleans_unselected_native_candidates() -> None:
    class LifecycleCandidate:
        def __init__(self, candidate_id: CandidateId) -> None:
            self.candidate_id = candidate_id.value
            self.cleaned = 0

        def run(self, context: CandidateContext) -> CandidateOutcome:
            del context
            raise AssertionError("candidate execution is outside this construction test")

        def cleanup(self) -> None:
            self.cleaned += 1

    candidates = {candidate_id: LifecycleCandidate(candidate_id) for candidate_id in CandidateId}

    selected = retain_selected_candidates(
        tuple(candidates.values()),
        selected_ids={CandidateId.LLM_WIKI.value, CandidateId.MEM0.value},
    )

    assert [candidate.candidate_id for candidate in selected] == ["llm-wiki", "mem0"]
    assert candidates[CandidateId.GBRAIN].cleaned == 1
    assert candidates[CandidateId.LLM_WIKI].cleaned == 0
    assert candidates[CandidateId.MEM0].cleaned == 0


def test_automatic_plan_requires_source_and_two_candidates() -> None:
    with pytest.raises(ExperimentSetupError, match="KNOWLEDGE_SOURCE_REQUIRED"):
        build_automatic_plan(
            sources=(),
            candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
            subscription_status=SubscriptionStatus.READY,
            embedding_readiness=_semantic_readiness(),
        )

    with pytest.raises(ExperimentSetupError, match="TWO_CANDIDATES_REQUIRED"):
        build_automatic_plan(
            sources=(Provider.SLACK,),
            candidates=(CandidateId.GBRAIN,),
            subscription_status=SubscriptionStatus.READY,
            embedding_readiness=_semantic_readiness(),
        )


def test_tui_allows_smoke_only_evaluator_for_runnable_non_recommendation_run() -> None:
    plan, error = resolve_plan(
        selected_sources=(Provider.SLACK,),
        selected_candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
        connections=ConnectionSnapshot(
            subscription=SubscriptionStatus.READY,
            embeddings=_smoke_readiness(),
            sources={Provider.SLACK: ConnectionState.CONNECTED},
        ),
    )

    assert plan is not None
    assert plan.embedding_backend == "local-hash"
    assert error == ""


def test_tui_never_falls_back_to_openai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enable-the-tui")

    plan, error = resolve_plan(
        selected_sources=(Provider.SLACK,),
        selected_candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
        connections=ConnectionSnapshot(
            subscription=SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE,
            embeddings=_semantic_readiness(),
            sources={Provider.SLACK: ConnectionState.CONNECTED},
        ),
    )

    assert plan is None
    assert error == "SUBSCRIPTION_AUTH_UNAVAILABLE: connect ChatGPT subscription"
