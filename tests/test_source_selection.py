from __future__ import annotations

from pathlib import Path

from autobrain.auth.models import Provider
from autobrain.auth.service import ConnectionManager
from autobrain.cancellation import RunCancellation
from autobrain.models import CandidateId, Status
from autobrain.orchestration import ConnectorSnapshot, RunConfig, RunOrchestrator
from autobrain.production import SlackExportSourceConnector, build_production_connectors


class _Connector:
    provider = "slack"

    def probe(self, cancellation: RunCancellation | None = None) -> dict[str, object]:
        return {"ok": True}

    def crawl(self, *, cancellation: RunCancellation | None = None) -> ConnectorSnapshot:
        return ConnectorSnapshot(provider=self.provider, documents=(), coverage={})


def _config(tmp_path: Path, *, sources: tuple[Provider, ...]) -> RunConfig:
    return RunConfig(
        output=tmp_path / "runs",
        provider_mode="api",
        selected_sources=sources,
        selected_candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
        slack_export_path=tmp_path / "slack-export.zip",
        slack_export_sha256="a" * 64,
    )


def test_slack_export_bypasses_slack_oauth_preflight(tmp_path: Path) -> None:
    manager = ConnectionManager(tmp_path / "state")
    connector = _Connector()

    orchestrator = RunOrchestrator.local(
        _config(tmp_path, sources=(Provider.SLACK,)),
        connection_manager=manager,
        connector_builder=lambda _manager, _include_dms: (connector,),
        candidate_builder=lambda *_args, **_kwargs: (),
        api_key="test-provider-key",
    )

    assert orchestrator.provider_available is True
    assert orchestrator.connectors == (connector,)


def test_slack_export_does_not_bypass_notion_oauth(tmp_path: Path) -> None:
    manager = ConnectionManager(tmp_path / "state")

    orchestrator = RunOrchestrator.local(
        _config(tmp_path, sources=(Provider.SLACK, Provider.NOTION)),
        connection_manager=manager,
        connector_builder=lambda _manager, _include_dms: (_Connector(),),
        candidate_builder=lambda *_args, **_kwargs: (),
        api_key="test-provider-key",
    )

    assert orchestrator.provider_available is False
    assert orchestrator.provider_status is Status.MCP_AUTH_UNAVAILABLE
    assert orchestrator.provider_detail == "MCP_AUTH_UNAVAILABLE: notion"


def test_production_builder_prefers_slack_export_connector(tmp_path: Path) -> None:
    manager = ConnectionManager(tmp_path / "state")
    archive_path = tmp_path / "slack-export.zip"

    connectors = build_production_connectors(
        manager,
        providers=(Provider.SLACK,),
        slack_export_path=archive_path,
        slack_export_sha256="b" * 64,
    )

    assert len(connectors) == 1
    connector = connectors[0]
    assert isinstance(connector, SlackExportSourceConnector)
    assert connector.archive_path == archive_path
