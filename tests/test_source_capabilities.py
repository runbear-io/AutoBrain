from __future__ import annotations

from datetime import UTC, datetime

import pytest

from autobrain.auth.models import Provider
from autobrain.connectors.readiness import (
    ReadinessMode,
    ReadinessReason,
    ReadinessState,
    fixture_readiness,
    readiness_for,
)
from autobrain.models import CoverageCompleteness, CoverageRecord, NormalizedDocument, SourceKind


def _document(
    source_id: str = "confluence:page:1",
    source_kind: SourceKind = SourceKind.CONFLUENCE_PAGE,
) -> NormalizedDocument:
    return NormalizedDocument(
        source_id=source_id,
        source_kind=source_kind,
        canonical_url="https://fixture.example.test/page-1",
        title="Launch",
        text="The launch is Tuesday.",
        content_hash="a" * 64,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )


def test_fixture_source_is_ready_without_claiming_live_oauth() -> None:
    document = _document()
    readiness = fixture_readiness(
        Provider.CONFLUENCE,
        documents=(document,),
        coverage=CoverageRecord(
            source=SourceKind.CONFLUENCE_PAGE,
            completeness=CoverageCompleteness.EXHAUSTIVE,
            discovered=1,
            fetched=1,
        ),
    )

    assert readiness.state is ReadinessState.READY
    assert readiness.mode is ReadinessMode.FIXTURE
    assert readiness.reason is ReadinessReason.FIXTURE_READY
    assert readiness.live_oauth is False
    assert readiness.corpus is not None
    assert readiness.corpus.documents[0].source_id == document.source_id


def test_unsupported_source_fails_closed_with_stable_reason() -> None:
    readiness = readiness_for(Provider.CONFLUENCE)

    assert readiness.state is ReadinessState.UNSUPPORTED
    assert readiness.mode is ReadinessMode.UNSUPPORTED
    assert readiness.ready is False
    assert readiness.reason is ReadinessReason.UNSUPPORTED_CONNECTOR
    assert readiness.corpus is None
    assert readiness.refusal is not None


def test_fixture_readiness_rejects_a_document_from_the_wrong_source() -> None:
    with pytest.raises(ValueError, match="documents do not match"):
        fixture_readiness(
            Provider.CONFLUENCE,
            documents=(_document("slack:message:1", SourceKind.SLACK_MESSAGE),),
            coverage=CoverageRecord(
                source=SourceKind.CONFLUENCE_PAGE,
                completeness=CoverageCompleteness.EXHAUSTIVE,
                discovered=1,
                fetched=1,
            ),
        )


def test_live_readiness_does_not_report_ready_without_a_verified_transport() -> None:
    readiness = readiness_for(Provider.GOOGLE_DRIVE)

    assert readiness.ready is False
    assert readiness.corpus is None
    assert readiness.reason is ReadinessReason.UNSUPPORTED_CONNECTOR
    assert readiness.live_oauth is False
