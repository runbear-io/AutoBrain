"""Gating tests for SharePoint, Grok, Kimi, and unverified custom providers.

These providers are explicitly unsupported: they expose stable machine-readable
status, reason, and remediation guidance, and no production constructor path
exists for them.  Verified Codex/Claude/Gemini behavior is preserved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from autobrain.auth.models import Provider
from autobrain.auth.providers import config_for
from autobrain.cli import app
from autobrain.connectors.readiness import (
    ConnectorSource,
    ReadinessMode,
    ReadinessReason,
    ReadinessState,
    readiness_for,
    sharepoint_readiness,
)
from autobrain.models import Status
from autobrain.orchestration import RunConfig, RunOrchestrator
from autobrain.production import build_production_connectors
from autobrain.subscription import (
    ProviderId,
    SubscriptionFailureReason,
    SubscriptionStatus,
    provider_registry,
)

# ---------------------------------------------------------------------------
# Readiness: SharePoint and unsupported connectors have structured status
# ---------------------------------------------------------------------------


def test_sharepoint_readiness_has_machine_readable_reason_and_remediation() -> None:
    readiness = sharepoint_readiness()

    assert readiness.connector is ConnectorSource.SHAREPOINT
    assert readiness.state is ReadinessState.UNSUPPORTED
    assert readiness.mode is ReadinessMode.UNSUPPORTED
    assert readiness.ready is False
    assert readiness.live_oauth is False
    assert readiness.corpus is None
    assert readiness.reason is ReadinessReason.UNSUPPORTED_CONNECTOR
    assert readiness.remediation, "remediation guidance must not be empty"
    assert (
        "supported" in readiness.remediation.lower() or "implement" in readiness.remediation.lower()
    )


def test_all_unsupported_connectors_have_structured_reason_and_remediation() -> None:
    for source in (
        ConnectorSource.SHAREPOINT,
        ConnectorSource.CONFLUENCE,
        ConnectorSource.GOOGLE_DRIVE,
    ):
        readiness = readiness_for(source)
        assert readiness.state is ReadinessState.UNSUPPORTED
        assert readiness.reason is ReadinessReason.UNSUPPORTED_CONNECTOR
        assert readiness.remediation, f"{source.value} must have remediation guidance"
        assert readiness.corpus is None
        assert readiness.live_oauth is False


# ---------------------------------------------------------------------------
# No production constructor path for SharePoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", [Provider.GOOGLE_DRIVE, Provider.SHAREPOINT])
def test_gated_source_has_no_production_connector_constructor(
    provider: Provider,
) -> None:
    """Gated sources fail before any live connector can be constructed."""
    from autobrain.auth.service import ConnectionManager

    manager = ConnectionManager(Path(f"/tmp/autobrain-test-no-{provider.value}"))
    with pytest.raises((ValueError, KeyError)):
        build_production_connectors(manager, providers=(provider,))


def test_sharepoint_provider_config_is_unsupported_with_no_resource_or_scopes() -> None:
    config = config_for(Provider.SHAREPOINT)
    assert config.supported is False
    assert config.resource is None
    assert config.scopes == ()
    assert config.allowlist == frozenset()
    assert config.required_tool_groups == ()


# ---------------------------------------------------------------------------
# Doctor: kimi/grok show UNSUPPORTED with guidance
# ---------------------------------------------------------------------------


def test_doctor_reports_kimi_and_grok_as_unsupported_with_guidance(
    tmp_path: Path,
) -> None:
    from autobrain.paths import AutoBrainPaths
    from autobrain.preflight import CommandResult, Preflight
    from autobrain.secrets import RuntimeEnvironment

    def runner(command: tuple[str, ...], timeout: float) -> CommandResult:
        del timeout
        name = Path(command[0]).name
        versions = {"python": "Python 3.12.7", "node": "v25.9.0", "bun": "1.3.14"}
        return CommandResult(
            returncode=0,
            stdout=versions.get("python" if name.startswith("python") else name, "ok"),
            stderr="",
        )

    report = Preflight(
        paths=AutoBrainPaths.from_home(tmp_path),
        environment=RuntimeEnvironment.from_environ({}),
        command_runner=runner,
        executable_finder=lambda name: f"/fake/{name}",
        keyring_available=lambda: True,
        callback_available=lambda _h, _p: True,
        browser_available=lambda: True,
    ).run()
    checks = {check.name: check for check in report.checks}
    for name in ("kimi_subscription", "grok_subscription"):
        assert checks[name].status is Status.UNSUPPORTED
        assert "unsupported" in checks[name].detail.lower()


# ---------------------------------------------------------------------------
# Registry: kimi/grok are UNSUPPORTED with PROVIDER_UNSUPPORTED reason
# ---------------------------------------------------------------------------


def test_registry_kimi_and_grok_have_unsupported_reason_and_guidance() -> None:
    registry = provider_registry()
    for provider_id in (ProviderId.KIMI, ProviderId.GROK):
        report = registry.get(provider_id).probe_status()
        assert report.status is SubscriptionStatus.UNSUPPORTED
        assert report.reason is SubscriptionFailureReason.PROVIDER_UNSUPPORTED
        assert report.detail, f"{provider_id.value} must have guidance detail"


def test_registry_rejects_unverified_custom_provider_name() -> None:
    """An unknown provider name must not silently resolve to a supported adapter."""
    registry = provider_registry()
    with pytest.raises(ValueError):
        registry.get("custom-llm")
    with pytest.raises(ValueError):
        registry.get("openai")


# ---------------------------------------------------------------------------
# Attempted run fails before construction for kimi/grok subscription
# ---------------------------------------------------------------------------


def test_kimi_subscription_run_fails_before_candidate_construction(
    tmp_path: Path,
) -> None:
    config = RunConfig(
        provider_mode="kimi-subscription",
        output=tmp_path / "runs",
        open_report=False,
        selected_sources=(Provider.SLACK,),
        slack_export_path=tmp_path / "slack.zip",
        slack_export_sha256="a" * 64,
    )
    orchestrator = RunOrchestrator.local(
        config,
        connector_builder=lambda _manager, _include_dms: (),
        candidate_builder=lambda *_args, **_kwargs: (),
    )
    assert orchestrator.provider_available is False
    assert orchestrator.candidates == ()
    result = orchestrator.run()
    assert result.status is Status.CAPABILITY_UNAVAILABLE
    assert result.run_dir.exists()  # manifest persisted even on failure


def test_grok_subscription_run_fails_before_candidate_construction(
    tmp_path: Path,
) -> None:
    config = RunConfig(
        provider_mode="grok-subscription",
        output=tmp_path / "runs",
        open_report=False,
        selected_sources=(Provider.SLACK,),
        slack_export_path=tmp_path / "slack.zip",
        slack_export_sha256="a" * 64,
    )
    orchestrator = RunOrchestrator.local(
        config,
        connector_builder=lambda _manager, _include_dms: (),
        candidate_builder=lambda *_args, **_kwargs: (),
    )
    assert orchestrator.provider_available is False
    result = orchestrator.run()
    assert result.status is Status.CAPABILITY_UNAVAILABLE


def test_unverified_custom_provider_mode_is_rejected_by_run_config() -> None:
    with pytest.raises(ValueError, match="provider_mode"):
        RunConfig(provider_mode="custom-llm-subscription")
    with pytest.raises(ValueError, match="provider_mode"):
        RunConfig(provider_mode="azure-subscription")


# ---------------------------------------------------------------------------
# Fixture readiness never claims live OAuth
# ---------------------------------------------------------------------------


def test_fixture_readiness_mode_is_fixture_and_live_oauth_is_false() -> None:
    from autobrain.connectors.readiness import fixture_readiness
    from autobrain.models import (
        CoverageCompleteness,
        CoverageRecord,
        NormalizedDocument,
        SourceKind,
    )

    doc = NormalizedDocument(
        source_id="confluence:page:1",
        source_kind=SourceKind.CONFLUENCE_PAGE,
        canonical_url="https://example.test/confluence/page/1",
        title="Runbook",
        text="Content for Runbook.",
        content_hash="a" * 64,
    )
    coverage = CoverageRecord(
        source=SourceKind.CONFLUENCE_PAGE,
        completeness=CoverageCompleteness.SEARCH_DISCOVERED,
        discovered=1,
        fetched=1,
    )
    readiness = fixture_readiness(
        ConnectorSource.CONFLUENCE,
        documents=(doc,),
        coverage=coverage,
    )
    assert readiness.ready is True
    assert readiness.mode is ReadinessMode.FIXTURE
    assert readiness.live_oauth is False
    assert readiness.reason is ReadinessReason.FIXTURE_READY
    assert readiness.corpus is not None


# ---------------------------------------------------------------------------
# CLI: subscription status for kimi/grok shows UNSUPPORTED
# ---------------------------------------------------------------------------


def test_subscription_status_cli_shows_unsupported_for_kimi_and_grok() -> None:
    import json

    runner = CliRunner()
    for provider in ("kimi", "grok"):
        result = runner.invoke(app, ["subscription", "status", "--provider", provider, "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["status"] == SubscriptionStatus.UNSUPPORTED.value
        assert payload["reason"] == SubscriptionFailureReason.PROVIDER_UNSUPPORTED.value
        assert payload["detail"], f"{provider} must have guidance detail"


# ---------------------------------------------------------------------------
# Verified Codex/Claude behavior is preserved
# ---------------------------------------------------------------------------


def test_verified_providers_remain_supported_in_registry() -> None:
    registry = provider_registry()
    assert ProviderId.CODEX in registry.provider_ids
    assert ProviderId.CLAUDE in registry.provider_ids
    # Codex and Claude are real adapters, not UnsupportedSubscriptionProvider
    from autobrain.subscription_registry import UnsupportedSubscriptionProvider

    for provider_id in (ProviderId.CODEX, ProviderId.CLAUDE):
        assert not isinstance(registry.get(provider_id), UnsupportedSubscriptionProvider), (
            f"{provider_id.value} must remain a verified adapter"
        )


def test_verified_source_providers_remain_supported() -> None:
    assert config_for(Provider.SLACK).supported is True
    assert config_for(Provider.NOTION).supported is True
    assert config_for(Provider.SLACK).resource is not None
    assert config_for(Provider.NOTION).resource is not None
