from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from autobrain.auth.models import Provider
from autobrain.connectors.confluence import (
    ConfluenceExternalGate,
    confluence_external_gate,
    confluence_fixture_corpus,
    normalize_confluence_fixture,
)
from autobrain.connectors.readiness import readiness_for
from autobrain.contracts import SourceConnectionState

_PAGE: dict[str, Any] = {
    "id": "12345",
    "title": "Incident runbook",
    "text": "Escalate incidents to the on-call team.",
    "url": "https://example.atlassian.net/wiki/spaces/OPS/pages/12345/runbook",
    "space_key": "OPS",
    "version": 7,
    "status": "current",
}


def test_sanitized_fixture_is_read_only_stable_and_provenance_preserving() -> None:
    documents, coverage = confluence_fixture_corpus([_PAGE])

    assert [item.source_id for item in documents] == ["confluence:page:12345"]
    assert documents[0].source_kind.value == "CONFLUENCE_PAGE"
    assert documents[0].crawl_provenance == {
        "connector": "confluence-read-only-fixture",
        "source_id": "confluence:page:12345",
        "space_key": "OPS",
        "page_version": "7",
        "transport": "fixture",
        "offline_gate": "W2",
    }
    assert coverage.crawl_provenance["offline_gate"] == "W2"
    assert coverage.fetched == coverage.discovered == 1


def test_public_read_shape_normalizes_without_claiming_live_access() -> None:
    api_page = {
        "id": "12345",
        "title": "Incident runbook",
        "status": "current",
        "version": {"number": 7},
        "space": {"key": "OPS"},
        "body": {"storage": {"value": "Escalate incidents to the on-call team."}},
        "_links": {
            "base": "https://example.atlassian.net",
            "webui": "/wiki/spaces/OPS/pages/12345/runbook",
        },
    }
    documents = normalize_confluence_fixture([api_page])
    assert documents[0].canonical_url == _PAGE["url"]
    assert documents[0].crawl_provenance["transport"] == "fixture"


def test_fixture_normalization_orders_pages_and_rejects_duplicate_or_stale_ids() -> None:
    second = {**_PAGE, "id": "2", "title": "Second"}
    first = {**_PAGE, "id": "1", "title": "First"}
    documents = normalize_confluence_fixture([second, first])
    assert [item.source_id for item in documents] == [
        "confluence:page:1",
        "confluence:page:2",
    ]

    with pytest.raises(ValueError, match="duplicate"):
        normalize_confluence_fixture([first, first])
    with pytest.raises(ValidationError, match="current"):
        normalize_confluence_fixture([{**first, "status": "historical"}])


def test_fixture_rejects_misleading_write_or_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        normalize_confluence_fixture([{**_PAGE, "operation": "update"}])
    with pytest.raises(ValidationError, match="extra_forbidden"):
        normalize_confluence_fixture([{**_PAGE, "unknown_field": "dirty"}])


def test_fixture_rejects_dirty_urls_and_missing_content() -> None:
    with pytest.raises(ValueError, match="safe HTTP"):
        normalize_confluence_fixture([{**_PAGE, "url": "https://example.test/a\\nwrite"}])
    with pytest.raises(ValidationError):
        normalize_confluence_fixture(
            [{key: value for key, value in _PAGE.items() if key != "text"}]
        )


def test_confluence_live_surface_is_external_gate_and_never_ready() -> None:
    gate = confluence_external_gate()
    assert isinstance(gate, ConfluenceExternalGate)
    assert gate.state == "EXTERNAL_GATE"
    assert gate.connection_state is SourceConnectionState.FAILED
    assert gate.observed_status == 401
    assert gate.observed_error == "invalid_token"
    assert gate.credential_present is False
    assert gate.network_allowed is False
    assert gate.source_writes_allowed is False

    readiness = readiness_for(Provider.CONFLUENCE)
    assert readiness.ready is False
    assert readiness.external_gate == gate
    assert readiness.refusal is not None
    assert readiness.refusal.observed == "HTTP 401 invalid_token"


def test_external_gate_rejects_any_attempt_to_claim_live_access() -> None:
    with pytest.raises(ValidationError):
        ConfluenceExternalGate(
            credential_present=True,
            detail="blocked",
            remediation="fixture",
        )
    with pytest.raises(ValidationError):
        ConfluenceExternalGate(network_allowed=True, detail="blocked", remediation="fixture")
    with pytest.raises(ValidationError):
        ConfluenceExternalGate(
            source_writes_allowed=True,
            detail="blocked",
            remediation="fixture",
        )
