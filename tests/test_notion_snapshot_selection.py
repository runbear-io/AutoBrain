from __future__ import annotations

from pathlib import Path

from autobrain.auth.models import AuthStatusReport, ConnectionStatus, Provider
from autobrain.models import CandidateId, ConnectionState, Status
from autobrain.orchestration import ConnectorSnapshot, RunConfig, RunOrchestrator


class _Manager:
    def status(self) -> AuthStatusReport:
        return AuthStatusReport(
            connections=(
                ConnectionStatus(
                    provider=Provider.SLACK,
                    state=ConnectionState.DISCONNECTED,
                    status=Status.MCP_AUTH_UNAVAILABLE,
                ),
                ConnectionStatus(
                    provider=Provider.NOTION,
                    state=ConnectionState.DISCONNECTED,
                    status=Status.MCP_AUTH_UNAVAILABLE,
                ),
            )
        )


class _NotionConnector:
    provider = "notion"

    def probe(self) -> dict[str, object]:
        return {"allowed": ["snapshot-read"], "capability_available": True}

    def crawl(self, *, include_dms: bool) -> ConnectorSnapshot:
        del include_dms
        return ConnectorSnapshot(
            provider="notion",
            documents=(),
            coverage={"completeness": "UNKNOWN", "fetched": 1},
        )


def _config(tmp_path: Path, *, sources: tuple[Provider, ...]) -> RunConfig:
    return RunConfig(
        output=tmp_path / "runs",
        provider_mode="api",
        selected_sources=sources,
        selected_candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
        notion_snapshot_path=tmp_path / "notion-snapshot.json",
    )


def test_snapshot_bypasses_only_notion_auth_not_slack_auth(tmp_path: Path) -> None:
    notion_only = RunOrchestrator.local(
        _config(tmp_path, sources=(Provider.NOTION,)),
        connection_manager=_Manager(),  # type: ignore[arg-type]
        connector_builder=lambda *_args: (_NotionConnector(),),
        candidate_builder=lambda *_args, **_kwargs: (),
        api_key="provider-key",
    )
    with_slack = RunOrchestrator.local(
        _config(tmp_path, sources=(Provider.SLACK, Provider.NOTION)),
        connection_manager=_Manager(),  # type: ignore[arg-type]
        connector_builder=lambda *_args: (_NotionConnector(),),
        candidate_builder=lambda *_args, **_kwargs: (),
        api_key="provider-key",
    )

    assert notion_only.provider_available is True
    assert with_slack.provider_available is False
    assert with_slack.provider_detail == "MCP_AUTH_UNAVAILABLE: slack"


def test_notion_only_snapshot_marks_recommendation_ineligible(tmp_path: Path) -> None:
    orchestrator = RunOrchestrator.local(
        _config(tmp_path, sources=(Provider.NOTION,)),
        connection_manager=_Manager(),  # type: ignore[arg-type]
        connector_builder=lambda *_args: (_NotionConnector(),),
        candidate_builder=lambda *_args, **_kwargs: (),
        api_key="provider-key",
    )

    assert orchestrator.coverage_eligibility_reasons() == [
        "Slack source absent; source coverage is partial and non-final",
        "Notion snapshot coverage is partial and non-final",
    ]
