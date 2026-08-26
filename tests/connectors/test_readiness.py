import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from autobrain.auth.models import Provider
from autobrain.auth.providers import CONFIGS
from autobrain.connectors.google_drive import ExternalGateReason
from autobrain.connectors.readiness import (
    ConnectorRefusal,
    ConnectorSource,
    CorpusContract,
    ReadinessMode,
    ReadinessState,
    SourceTransportRegistry,
    TransportGovernanceCode,
    TransportReadinessState,
    fixture_readiness,
    readiness_for,
    sharepoint_readiness,
)
from autobrain.models import CoverageCompleteness, CoverageRecord, NormalizedDocument, SourceKind


def document(source_kind: SourceKind, source_id: str, title: str) -> NormalizedDocument:
    return NormalizedDocument(
        source_id=source_id,
        source_kind=source_kind,
        canonical_url=f"https://example.test/{source_id.replace(':', '/')}",
        title=title,
        text=f"Content for {title}.",
        content_hash="a" * 64,
    )


def test_fixture_corpus_is_source_neutral_and_hash_is_deterministic() -> None:
    records = (
        document(SourceKind.CONFLUENCE_PAGE, "confluence:page:2", "Second"),
        document(SourceKind.CONFLUENCE_PAGE, "confluence:page:1", "First"),
    )
    coverage = CoverageRecord(
        source=SourceKind.CONFLUENCE_PAGE,
        completeness=CoverageCompleteness.EXHAUSTIVE,
        discovered=2,
        fetched=2,
    )

    first = CorpusContract.from_documents(
        source=ConnectorSource.CONFLUENCE,
        documents=records,
        coverage=coverage,
    )
    second = CorpusContract.from_documents(
        source=ConnectorSource.CONFLUENCE,
        documents=tuple(reversed(records)),
        coverage=coverage,
    )

    assert first.documents == records
    assert first.corpus_sha256 == second.corpus_sha256
    assert len(first.corpus_sha256) == 64


def test_corpus_rejects_forged_and_stale_hashes() -> None:
    records = (document(SourceKind.CONFLUENCE_PAGE, "confluence:page:1", "Runbook"),)
    coverage = CoverageRecord(
        source=SourceKind.CONFLUENCE_PAGE,
        completeness=CoverageCompleteness.EXHAUSTIVE,
        discovered=1,
        fetched=1,
    )
    valid = CorpusContract.from_documents(
        source=ConnectorSource.CONFLUENCE,
        documents=records,
        coverage=coverage,
    )

    with pytest.raises(ValidationError, match="corpus_sha256"):
        CorpusContract(
            source=valid.source,
            documents=valid.documents,
            coverage=valid.coverage,
            corpus_sha256="b" * 64,
        )
    with pytest.raises(ValidationError, match="corpus_sha256"):
        CorpusContract(
            source=valid.source,
            documents=(document(SourceKind.CONFLUENCE_PAGE, "confluence:page:1", "Changed"),),
            coverage=valid.coverage,
            corpus_sha256=valid.corpus_sha256,
        )


def test_confluence_and_drive_fixture_readiness_is_ready_without_live_auth() -> None:
    confluence = fixture_readiness(
        ConnectorSource.CONFLUENCE,
        documents=(document(SourceKind.CONFLUENCE_PAGE, "confluence:page:1", "Runbook"),),
        coverage=CoverageRecord(
            source=SourceKind.CONFLUENCE_PAGE,
            completeness=CoverageCompleteness.SEARCH_DISCOVERED,
            discovered=1,
            fetched=1,
        ),
    )
    drive = fixture_readiness(
        ConnectorSource.GOOGLE_DRIVE,
        documents=(document(SourceKind.GOOGLE_DRIVE_FILE, "drive:file:1", "Policy"),),
        coverage=CoverageRecord(
            source=SourceKind.GOOGLE_DRIVE_FILE,
            completeness=CoverageCompleteness.SEARCH_DISCOVERED,
            discovered=1,
            fetched=1,
        ),
    )

    assert confluence.ready is True
    assert confluence.mode.value == "FIXTURE"
    assert confluence.live_oauth is False
    assert drive.ready is True
    assert drive.corpus is not None


def test_sharepoint_is_explicitly_gated_and_has_no_fixture_corpus() -> None:
    readiness = sharepoint_readiness()

    assert readiness.connector is ConnectorSource.SHAREPOINT
    assert readiness.state is ReadinessState.UNSUPPORTED
    assert readiness.ready is False
    assert readiness.mode.value == "UNSUPPORTED"
    assert readiness.live_oauth is False
    assert readiness.corpus is None


def test_auth_provider_table_is_authoritative_for_all_connectors() -> None:
    assert Provider is ConnectorSource
    assert tuple(Provider) == tuple(ConnectorSource)
    assert set(CONFIGS) == set(Provider)
    assert CONFIGS[Provider.CONFLUENCE].supported is False
    assert CONFIGS[Provider.GOOGLE_DRIVE].supported is False
    drive_readiness = readiness_for(Provider.GOOGLE_DRIVE)
    drive_gate = drive_readiness.external_gate
    assert drive_gate is not None
    assert drive_gate.reason is ExternalGateReason.PROJECT_POLICY
    assert CONFIGS[Provider.SHAREPOINT].supported is False


def test_confluence_refusal_records_official_mcp_external_gate() -> None:
    readiness = readiness_for(ConnectorSource.CONFLUENCE)

    assert readiness.refusal == ConnectorRefusal(
        code="OFFICIAL_MCP_UNVERIFIED",
        endpoint="https://mcp.atlassian.com/v1/mcp",
        transport="streamable_http",
        observed="HTTP 401 invalid_token",
        credential_present=False,
    )


def test_offline_transport_registry_covers_all_planned_modes_without_network() -> None:
    registry = SourceTransportRegistry(config_path=Path("/tmp/does-not-exist-autobrain-oauth.json"))
    cases = (
        ("slack", "export_archive", TransportReadinessState.READY),
        ("slack", "live_mcp_oauth", TransportReadinessState.CONFIG_INVALID),
        ("notion", "live_mcp_oauth", TransportReadinessState.CONFIG_INVALID),
        ("confluence", "oauth_3lo_rest", TransportReadinessState.UNAVAILABLE),
        ("google_drive", "oauth_rest", TransportReadinessState.UNAVAILABLE),
        ("sharepoint", "graph_oauth_preview", TransportReadinessState.UNAVAILABLE),
    )
    for provider, mode, expected_state in cases:
        result = registry.resolve(provider, mode)  # type: ignore[arg-type]
        assert result.state is expected_state
        assert result.ready is (expected_state is TransportReadinessState.READY)
        assert result.governance_code


def _write_oauth_config(
    path: Path, *, redirect_uri: str = "https://app.example.test/callback"
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "apps": [
                    {
                        "provider": "slack",
                        "mode": "live_mcp_oauth",
                        "client_id": "fixture-client-id",
                        "client_secret_ref": "fixture-secret-ref",
                        "redirect_uri": redirect_uri,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)


def test_oauth_app_config_is_local_strict_and_redirect_mismatch_is_typed(tmp_path: Path) -> None:
    config_path = tmp_path / "source-oauth-apps.json"
    _write_oauth_config(config_path)
    registry = SourceTransportRegistry(config_path=config_path)

    configured = registry.resolve(
        "slack",
        "live_mcp_oauth",
        redirect_uri="https://app.example.test/callback",  # type: ignore[arg-type]
    )
    assert configured.state is TransportReadinessState.AUTH_REQUIRED
    assert configured.app_configured is True
    assert configured.credential_present is False

    mismatch = registry.resolve(
        "slack",
        "live_mcp_oauth",
        redirect_uri="https://other.example.test/callback",  # type: ignore[arg-type]
    )
    assert mismatch.governance_code is TransportGovernanceCode.SOURCE_REDIRECT_URI_MISMATCH
    assert mismatch.state is TransportReadinessState.CONFIG_INVALID


def test_oauth_app_config_rejects_malformed_unsafe_permissions_and_symlink(tmp_path: Path) -> None:
    config_path = tmp_path / "source-oauth-apps.json"
    config_path.write_text("{not-json", encoding="utf-8")
    config_path.chmod(0o600)
    malformed = SourceTransportRegistry(config_path=config_path).resolve("slack", "live_mcp_oauth")  # type: ignore[arg-type]
    assert malformed.governance_code is TransportGovernanceCode.SOURCE_OAUTH_CONFIG_INVALID

    _write_oauth_config(config_path)
    config_path.chmod(0o644)
    unsafe = SourceTransportRegistry(config_path=config_path).resolve("slack", "live_mcp_oauth")  # type: ignore[arg-type]
    assert unsafe.governance_code is TransportGovernanceCode.SOURCE_OAUTH_APP_UNSAFE

    outside = tmp_path / "outside.json"
    _write_oauth_config(outside)
    config_path.unlink()
    config_path.symlink_to(outside)
    symlink = SourceTransportRegistry(config_path=config_path).resolve("slack", "live_mcp_oauth")  # type: ignore[arg-type]
    assert symlink.governance_code is TransportGovernanceCode.SOURCE_OAUTH_APP_UNSAFE


def test_oauth_app_config_missing_is_actionable_and_never_ready(tmp_path: Path) -> None:
    result = SourceTransportRegistry(config_path=tmp_path / "missing.json").resolve(
        "notion",
        "live_mcp_oauth",  # type: ignore[arg-type]
    )
    assert result.governance_code is TransportGovernanceCode.SOURCE_OAUTH_APP_MISSING
    assert result.ready is False
    assert result.state is TransportReadinessState.CONFIG_INVALID


def test_unsupported_connectors_are_explicitly_not_ready() -> None:
    for source in (
        ConnectorSource.CONFLUENCE,
        ConnectorSource.GOOGLE_DRIVE,
        ConnectorSource.SHAREPOINT,
    ):
        readiness = readiness_for(source)
        assert readiness.ready is False
        assert readiness.state is ReadinessState.UNSUPPORTED
        assert readiness.mode is ReadinessMode.UNSUPPORTED
        assert readiness.corpus is None
        if source is ConnectorSource.GOOGLE_DRIVE:
            assert readiness.external_gate is not None
            assert readiness.external_gate.read_only is True
