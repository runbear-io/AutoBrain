from __future__ import annotations

from pathlib import Path

import pytest

from autobrain.auth.models import Provider
from autobrain.auth.service import ConnectionManager
from autobrain.connectors.notion_snapshot import NotionSnapshotStore
from autobrain.models import SourceKind, Status
from autobrain.production import (
    NotionSnapshotConnector,
    SlackExportSourceConnector,
    build_production_connectors,
)
from tests.support.source_replay import (
    write_notion_snapshot,
    write_official_shaped_slack_export,
)


def test_official_shaped_slack_export_replays_without_credentials(tmp_path: Path) -> None:
    archive = write_official_shaped_slack_export(tmp_path / "slack-export.zip")
    manager = ConnectionManager(tmp_path / "auth-state")

    (connector,) = build_production_connectors(
        manager,
        providers=(Provider.SLACK,),
        slack_export_path=archive,
    )

    assert isinstance(connector, SlackExportSourceConnector)
    assert connector.probe()["status"] == Status.OK.value
    snapshot = connector.crawl()
    assert snapshot.provider == Provider.SLACK.value
    assert snapshot.coverage["source"] == "slack-export"
    assert snapshot.coverage["completeness"] == "EXHAUSTIVE"
    assert [document["source_kind"] for document in snapshot.documents] == [
        SourceKind.SLACK_MESSAGE.value,
        SourceKind.SLACK_THREAD.value,
    ]
    assert snapshot.documents[1]["parent_source_id"] == snapshot.documents[0]["source_id"]


def test_notion_snapshot_replays_without_credentials(tmp_path: Path) -> None:
    incoming = write_notion_snapshot(tmp_path / "incoming-notion-snapshot.json")
    store = NotionSnapshotStore(tmp_path / "sources")
    config = store.import_snapshot(incoming)
    manager = ConnectionManager(tmp_path / "auth-state")

    (connector,) = build_production_connectors(
        manager,
        providers=(Provider.NOTION,),
        notion_snapshot_path=store.snapshot_path,
    )

    assert isinstance(connector, NotionSnapshotConnector)
    assert connector.probe()["capability_available"] is True
    snapshot = connector.crawl()
    assert snapshot.provider == Provider.NOTION.value
    assert snapshot.coverage["completeness"] == "UNKNOWN"
    assert snapshot.coverage["crawl_provenance"]["snapshot_sha256"] == config.snapshot_sha256
    assert snapshot.documents[0]["source_kind"] == SourceKind.NOTION_PAGE.value
    assert snapshot.documents[0]["crawl_provenance"]["transport_mode"] == "imported_snapshot"


@pytest.mark.parametrize("provider", [Provider.SLACK, Provider.NOTION])
def test_live_provider_path_is_unavailable_without_credentials(
    tmp_path: Path,
    provider: Provider,
) -> None:
    manager = ConnectionManager(tmp_path / "auth-state")

    expected = rf"^MCP_AUTH_UNAVAILABLE: authenticated {provider.value.title()}"
    with pytest.raises(ValueError, match=expected):
        build_production_connectors(manager, providers=(provider,))
