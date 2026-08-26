"""Conformance tests pinning the `Connector` seam declared in `autobrain.orchestration`.

The seam previously lied about itself: it declared `probe()`/`crawl(*, include_dms)`
while every production connector actually accepted a `cancellation` argument, and
orchestration bridged the gap with signature inspection plus a `type: ignore`. These
tests keep the declared Protocol and the real implementers in agreement, so a new
connector cannot silently ship without cancellation support.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import SecretStr

from autobrain.auth.models import Provider, TokenRecord
from autobrain.auth.providers import CONFIGS, ProviderConfig, config_for
from autobrain.auth.service import ConnectionManager
from autobrain.cancellation import RunCancellation, RunCancelled
from autobrain.connectors.notion_snapshot import NotionSnapshotConnector
from autobrain.fixture import FixtureConnector
from autobrain.mcp.transport import StreamableHttpConnection
from autobrain.models import Status
from autobrain.orchestration import (
    Candidate,
    CandidateOutcome,
    Connector,
    ConnectorSnapshot,
    RunConfig,
    RunOrchestrator,
)
from autobrain.production import (
    NotionMcpConnector,
    SlackExportSourceConnector,
    SlackMcpConnector,
    build_production_connectors,
)


def _token(provider: Provider) -> TokenRecord:
    return TokenRecord(
        provider=provider,
        workspace_id="fixture-workspace",
        user_id="fixture-user",
        audience=cast(str, config_for(provider).resource),
        access_token=SecretStr("fixture-token"),
    )


def _connection(provider: Provider) -> StreamableHttpConnection:
    """Build an unopened MCP connection; construction performs no network I/O."""
    return StreamableHttpConnection(
        provider,
        cast(str, config_for(provider).resource),
        _token(provider),
    )


def _implementers() -> tuple[Connector, ...]:
    """Every in-tree `Connector` implementer, constructed without I/O.

    The `tuple[Connector, ...]` annotation is itself the static half of this
    conformance check: basedpyright rejects the assignment if any implementer
    drifts from the declared Protocol.
    """
    return (
        SlackMcpConnector(_connection(Provider.SLACK)),
        NotionMcpConnector(_connection(Provider.NOTION)),
        SlackExportSourceConnector(Path("slack-export.zip")),
        NotionSnapshotConnector(Path("notion-snapshot.json")),
        FixtureConnector("slack", ()),
    )


@pytest.mark.parametrize(
    "connector",
    _implementers(),
    ids=lambda connector: type(connector).__name__,
)
@pytest.mark.parametrize("method_name", ["probe", "crawl"])
def test_connector_takes_optional_cancellation_and_no_slack_specific_include_dms(
    connector: Connector,
    method_name: str,
) -> None:
    parameters = inspect.signature(getattr(connector, method_name)).parameters

    assert "cancellation" in parameters, (
        f"{type(connector).__name__}.{method_name} must accept cancellation"
    )
    assert parameters["cancellation"].default is None, (
        f"{type(connector).__name__}.{method_name} must make cancellation optional"
    )
    assert "include_dms" not in parameters, (
        f"{type(connector).__name__}.{method_name} must not carry the Slack-specific "
        "include_dms flag on the source-neutral seam"
    )


def test_snapshot_connector_refuses_probe_and_crawl_on_a_cancelled_run(tmp_path: Path) -> None:
    cancellation = RunCancellation()
    cancellation.cancel()
    connector = NotionSnapshotConnector(tmp_path / "notion-snapshot.json")

    with pytest.raises(RunCancelled):
        connector.probe(cancellation=cancellation)
    with pytest.raises(RunCancelled):
        connector.crawl(cancellation=cancellation)


def test_snapshot_connector_probe_still_answers_without_cancellation(tmp_path: Path) -> None:
    connector = NotionSnapshotConnector(tmp_path / "notion-snapshot.json")

    assert connector.probe()["capability_available"] is False


class _RecordingConnector:
    """A conformant connector that records the cancellation it was handed."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.probe_cancellations: list[RunCancellation | None] = []
        self.crawl_cancellations: list[RunCancellation | None] = []

    def probe(self, cancellation: RunCancellation | None = None) -> Mapping[str, Any]:
        self.probe_cancellations.append(cancellation)
        return {"allowed": ["search", "fetch"]}

    def crawl(self, *, cancellation: RunCancellation | None = None) -> ConnectorSnapshot:
        self.crawl_cancellations.append(cancellation)
        documents = tuple(_records(self.provider))
        return ConnectorSnapshot(
            provider=self.provider,
            documents=documents,
            coverage={"completeness": "SEARCH_DISCOVERED", "discovered": len(documents)},
        )


class _Candidate:
    candidate_id = "llm-wiki"

    def __init__(self) -> None:
        self.cleaned = False

    def run(self, context: Any) -> CandidateOutcome:
        del context
        return CandidateOutcome(
            candidate=self.candidate_id,
            status=Status.OK,
            answered_cases=24,
            scored_cases=24,
            cost_usd=1,
        )

    def cleanup(self) -> None:
        self.cleaned = True


def _records(provider: str, count: int = 24) -> list[dict[str, Any]]:
    source_kind = "SLACK_MESSAGE" if provider == "slack" else "NOTION_PAGE"
    return [
        {
            "provider": provider,
            "source_id": f"{provider}:contract:{index}",
            "source_kind": source_kind,
            "canonical_url": f"https://fixture.example.test/{provider}/{index}",
            "title": f"Policy {index}",
            "text": f"Use the documented policy for service {index} within five minutes.",
            "question": f"How do we handle service {index} incidents?",
            "evidence_reply": f"The on-call follows policy {index} and records the incident.",
        }
        for index in range(count)
    ]


def test_orchestrator_hands_its_cancellation_to_every_connector(tmp_path: Path) -> None:
    """Orchestration must plumb cancellation unconditionally, without signature sniffing."""
    cancellation = RunCancellation()
    connectors = [_RecordingConnector("slack"), _RecordingConnector("notion")]

    result = RunOrchestrator(
        config=RunConfig(output=tmp_path / "runs", open_report=False),
        connectors=connectors,
        candidates=cast(Sequence[Candidate], [_Candidate()]),
        provider_available=True,
        cancellation=cancellation,
    ).run()

    assert result.status is not Status.CAPABILITY_UNAVAILABLE
    for connector in connectors:
        assert connector.probe_cancellations == [cancellation]
        assert connector.crawl_cancellations == [cancellation]


class _TokenedManager(ConnectionManager):
    """A connection manager holding tokens without touching real OAuth storage."""

    def token_for(self, provider: Provider) -> TokenRecord | None:
        return _token(provider)


def test_mcp_required_tool_groups_are_provider_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = config_for(Provider.NOTION)
    custom = ProviderConfig(
        original.provider,
        original.resource,
        original.scopes,
        original.allowlist | {"synthetic-search", "synthetic-fetch"},
        original.dynamic_registration,
        original.fixed_client,
        (frozenset({"synthetic-search"}), frozenset({"synthetic-fetch"})),
    )

    class SnapshotConnection:
        snapshot = type(
            "Snapshot",
            (),
            {"advertised": ("synthetic-search",), "allowed": ("synthetic-search",)},
        )()

        async def __aenter__(self) -> SnapshotConnection:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

    monkeypatch.setitem(CONFIGS, Provider.NOTION, custom)
    connector = NotionMcpConnector(SnapshotConnection())  # type: ignore[arg-type]

    probe = connector.probe()
    assert probe["required"] == [["synthetic-search"], ["synthetic-fetch"]]
    assert probe["capability_available"] is False


def test_synthetic_provider_config_does_not_default_to_notion() -> None:
    assert config_for(Provider.CONFLUENCE).required_tool_groups == ()


def test_mcp_required_tool_groups_have_read_only_aliases_and_no_writes() -> None:
    for provider in (Provider.SLACK, Provider.NOTION):
        config = config_for(provider)
        assert config.required_tool_groups
        assert all(group <= config.allowlist for group in config.required_tool_groups)
        assert not any(
            "write" in tool or "create" in tool or "delete" in tool
            for group in config.required_tool_groups
            for tool in group
        )


def test_slack_dm_opt_in_is_carried_by_construction_not_the_crawl_seam(tmp_path: Path) -> None:
    connectors = build_production_connectors(
        _TokenedManager(tmp_path / "state"),
        include_dms=True,
        providers=(Provider.SLACK,),
    )

    connector = connectors[0]
    assert isinstance(connector, SlackMcpConnector)
    assert connector.include_dms is True
