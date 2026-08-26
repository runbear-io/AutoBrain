"""Permission-scoped source crawlers."""

from autobrain.connectors.confluence import (
    ConfluenceExternalGate,
    ConfluenceFixturePage,
    confluence_external_gate,
    confluence_fixture_corpus,
    normalize_confluence_fixture,
)
from autobrain.connectors.google_drive import (
    ExternalGateReason,
    GoogleDriveExternalGate,
    GoogleDriveFixtureFile,
    google_drive_external_gate,
    google_drive_fixture_corpus,
    normalize_google_drive_fixture,
)
from autobrain.connectors.readiness import (
    ConnectorReadiness,
    ConnectorRefusal,
    ConnectorSource,
    CorpusContract,
    ReadinessMode,
    ReadinessReason,
    ReadinessState,
    fixture_readiness,
    readiness_for,
    sharepoint_readiness,
)
from autobrain.connectors.slack import SlackCrawler, SlackCrawlResult

__all__ = [
    "ConfluenceExternalGate",
    "ConfluenceFixturePage",
    "ConnectorReadiness",
    "ConnectorRefusal",
    "ConnectorSource",
    "CorpusContract",
    "ExternalGateReason",
    "GoogleDriveExternalGate",
    "GoogleDriveFixtureFile",
    "ReadinessMode",
    "ReadinessReason",
    "ReadinessState",
    "SlackCrawlResult",
    "SlackCrawler",
    "confluence_external_gate",
    "confluence_fixture_corpus",
    "fixture_readiness",
    "google_drive_external_gate",
    "google_drive_fixture_corpus",
    "normalize_confluence_fixture",
    "normalize_google_drive_fixture",
    "readiness_for",
    "sharepoint_readiness",
]
