"""Verified model-access provider gates and unsupported CLI status.

This module is the single consolidated evidence source for the readiness gate
that records which local subscription adapters are verified, which embedding
backends are separate BYOK capabilities, and which providers remain typed
unsupported with remediation.

Gates under test:

1. **Codex and Claude** are the only verified local subscription adapters.
   They are real adapter instances (not ``UnsupportedSubscriptionProvider``)
   and expose the full ``SubscriptionProvider`` protocol surface.

2. **Gemini BYOK embedding** is a separate capability, not a subscription
   adapter.  It is registered in the embedding backend registry under the
   ``gemini`` selector with ``gemini_transport`` and requires
   ``GEMINI_API_KEY``.  It is never present in the subscription provider
   registry.

3. **Kimi, Grok, and any custom/unverified provider name** remain typed
   ``UNSUPPORTED`` with ``PROVIDER_UNSUPPORTED`` reason and non-empty
   remediation guidance.  No production constructor path or run config
   accepts them as verified.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autobrain.cli import app
from autobrain.embedding import production_embedding_registry
from autobrain.subscription import (
    AuthKind,
    ProviderId,
    SubscriptionError,
    SubscriptionFailureReason,
    SubscriptionStatus,
    provider_registry,
)
from autobrain.subscription_registry import UnsupportedSubscriptionProvider

# ---------------------------------------------------------------------------
# Gate 1: Codex and Claude are the only verified local subscription adapters
# ---------------------------------------------------------------------------


def test_only_codex_and_claude_are_verified_subscription_adapters() -> None:
    """The registry contains exactly four providers; only Codex and Claude are real adapters."""
    registry = provider_registry()
    assert tuple(registry.provider_ids) == (
        ProviderId.CODEX,
        ProviderId.CLAUDE,
        ProviderId.KIMI,
        ProviderId.GROK,
    )

    verified: list[ProviderId] = []
    unsupported: list[ProviderId] = []
    for pid in registry.provider_ids:
        provider = registry.get(pid)
        if isinstance(provider, UnsupportedSubscriptionProvider):
            unsupported.append(pid)
        else:
            verified.append(pid)

    assert verified == [ProviderId.CODEX, ProviderId.CLAUDE]
    assert unsupported == [ProviderId.KIMI, ProviderId.GROK]


def test_verified_adapters_expose_consumer_subscription_auth_kind() -> None:
    registry = provider_registry()
    for pid in (ProviderId.CODEX, ProviderId.CLAUDE):
        identity = registry.get(pid).probe_identity()
        assert identity.auth_kind is AuthKind.CONSUMER_SUBSCRIPTION, (
            f"{pid.value} must use consumer subscription auth"
        )


def test_verified_adapters_are_not_unsupported_provider_instances() -> None:
    registry = provider_registry()
    for pid in (ProviderId.CODEX, ProviderId.CLAUDE):
        assert not isinstance(registry.get(pid), UnsupportedSubscriptionProvider)


def test_no_extra_verified_provider_ids_exist_beyond_codex_and_claude() -> None:
    """No gemini, openai, or custom provider id is in the subscription registry."""
    registry = provider_registry()
    for name in ("gemini", "openai", "azure", "custom-llm", "mistral"):
        with pytest.raises((ValueError, KeyError)):
            registry.get(name)


# ---------------------------------------------------------------------------
# Gate 2: Gemini BYOK embedding is separate from subscription adapters
# ---------------------------------------------------------------------------


def test_gemini_embedding_is_registered_separately_from_subscription() -> None:
    registry = production_embedding_registry()
    descriptor = registry.resolve_selector("gemini")
    assert descriptor is not None
    assert descriptor.gemini_transport is True
    assert descriptor.openai_transport is False
    assert descriptor.requires_api_key is True
    assert descriptor.api_key_env == "GEMINI_API_KEY"
    assert descriptor.provenance_backend == "google:gemini-embedding-001"


def test_gemini_is_not_a_subscription_provider_id() -> None:
    """Gemini is an embedding backend, not a subscription ProviderId."""
    provider_ids = {pid.value for pid in ProviderId}
    assert "gemini" not in provider_ids


def test_embedding_registry_contains_only_local_hash_openai_and_gemini() -> None:
    selectors = production_embedding_registry().selectors
    assert set(selectors) == {"local-hash", "openai", "gemini"}


# ---------------------------------------------------------------------------
# Gate 3: Kimi, Grok, and custom providers remain typed unsupported
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_id", [ProviderId.KIMI, ProviderId.GROK])
def test_unsupported_providers_have_unsupported_status_and_reason(
    provider_id: ProviderId,
) -> None:
    registry = provider_registry()
    report = registry.get(provider_id).probe_status()
    assert report.status is SubscriptionStatus.UNSUPPORTED
    assert report.reason is SubscriptionFailureReason.PROVIDER_UNSUPPORTED
    assert report.detail, f"{provider_id.value} must have remediation guidance"
    assert "verified official" in report.detail


@pytest.mark.parametrize("provider_id", [ProviderId.KIMI, ProviderId.GROK])
def test_unsupported_provider_identity_has_unsupported_auth_kind(
    provider_id: ProviderId,
) -> None:
    registry = provider_registry()
    identity = registry.get(provider_id).probe_identity()
    assert identity.auth_kind is AuthKind.UNSUPPORTED
    assert identity.model is None
    assert identity.cli_version is None


@pytest.mark.parametrize("provider_id", [ProviderId.KIMI, ProviderId.GROK])
def test_unsupported_provider_login_raises_typed_error(provider_id: ProviderId) -> None:
    registry = provider_registry()
    with pytest.raises(SubscriptionError) as exc_info:
        registry.get(provider_id).login()
    assert exc_info.value.status is SubscriptionStatus.UNSUPPORTED
    assert exc_info.value.reason is SubscriptionFailureReason.PROVIDER_UNSUPPORTED


@pytest.mark.parametrize("provider_id", [ProviderId.KIMI, ProviderId.GROK])
def test_unsupported_provider_answer_raises_typed_error(provider_id: ProviderId) -> None:
    registry = provider_registry()
    with pytest.raises(SubscriptionError) as exc_info:
        registry.get(provider_id).answer("test prompt")
    assert exc_info.value.status is SubscriptionStatus.UNSUPPORTED
    assert exc_info.value.reason is SubscriptionFailureReason.PROVIDER_UNSUPPORTED


def test_custom_unverified_provider_name_is_rejected_by_registry() -> None:
    registry = provider_registry()
    for name in ("custom-llm", "openai", "azure", "mistral"):
        with pytest.raises((ValueError, KeyError)):
            registry.get(name)


# ---------------------------------------------------------------------------
# CLI gate: subscription status and ask reflect the typed gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["kimi", "grok"])
def test_cli_subscription_status_reports_unsupported(provider: str) -> None:
    result = CliRunner().invoke(app, ["subscription", "status", "--provider", provider, "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == SubscriptionStatus.UNSUPPORTED.value
    assert payload["reason"] == SubscriptionFailureReason.PROVIDER_UNSUPPORTED.value
    assert payload["detail"]


@pytest.mark.parametrize("provider", ["kimi", "grok"])
def test_cli_subscription_ask_rejects_unsupported_provider(provider: str) -> None:
    result = CliRunner().invoke(app, ["subscription", "ask", "hello", "--provider", provider])
    assert result.exit_code == 1
    assert SubscriptionStatus.UNSUPPORTED.value in result.stderr


# ---------------------------------------------------------------------------
# Doctor gate: kimi/grok show UNSUPPORTED in the doctor report
# ---------------------------------------------------------------------------


def test_doctor_reports_kimi_and_grok_as_unsupported(tmp_path: Path) -> None:
    from autobrain.models import Status
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
