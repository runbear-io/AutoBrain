"""Machine-readable inventory of reused and gated integration surfaces.

The catalog records only identities verified in repository pins or explicit
runtime contracts.  Unknown version and license fields remain null rather than
being guessed.  It is embedded in run provenance so reports and readiness
surfaces describe the same integration boundary.
"""

from __future__ import annotations

from autobrain.models import (
    DesignPartnerGateEvidence,
    IntegrationProvenance,
    IntegrationReuse,
    IntegrationStatus,
)

# Version/license values are populated only where candidate-pins.json or an
# explicit project dependency verifies them.  Consumer CLI versions are runtime
# observations and therefore intentionally null in the static catalog.
_CATALOG: tuple[IntegrationProvenance, ...] = (
    IntegrationProvenance(
        id="source.slack-export",
        provider="slack",
        source="slack",
        backend="autobrain-slack-export",
        version=None,
        license=None,
        auth_kind="local_export",
        capabilities=("read_only", "frozen_snapshot"),
        usage_provenance="unavailable",
        reuse=IntegrationReuse.THIN_ADAPTER,
        status=IntegrationStatus.CURRENT,
        evidence="repository source connector and frozen export contract",
    ),
    IntegrationProvenance(
        id="source.notion-mcp",
        provider="notion",
        source="notion",
        backend="hosted-notion-mcp",
        version=None,
        license=None,
        auth_kind="oauth",
        capabilities=("read_only", "search", "fetch", "live_mcp_capture"),
        usage_provenance="unavailable",
        reuse=IntegrationReuse.PROTOCOL_REUSE,
        status=IntegrationStatus.CURRENT,
        evidence="official hosted MCP endpoint and read-tool allowlist",
    ),
    IntegrationProvenance(
        id="candidate.llm-wiki",
        provider="llm-wiki-compiler",
        source=None,
        backend="llm-wiki-compiler",
        version="1.1.0",
        license="MIT",
        auth_kind="api_key",
        capabilities=("ingest", "retrieval", "answer"),
        usage_provenance="measured_or_unavailable",
        reuse=IntegrationReuse.THIN_ADAPTER,
        status=IntegrationStatus.CURRENT,
        evidence="candidate-pins.json and native lifecycle adapter",
    ),
    IntegrationProvenance(
        id="candidate.mem0",
        provider="mem0ai",
        source=None,
        backend="mem0ai",
        version="2.0.18",
        license="Apache-2.0",
        auth_kind="api_key",
        capabilities=("ingest", "retrieval", "answer"),
        usage_provenance="measured_or_unavailable",
        reuse=IntegrationReuse.THIN_ADAPTER,
        status=IntegrationStatus.CURRENT,
        evidence="candidate-pins.json and native lifecycle adapter",
    ),
    IntegrationProvenance(
        id="candidate.gbrain",
        provider="gbrain",
        source=None,
        backend="gbrain",
        version="0.46.19.0",
        license="MIT",
        auth_kind="api_key",
        capabilities=("ingest", "retrieval", "answer"),
        usage_provenance="measured_or_unavailable",
        reuse=IntegrationReuse.THIN_ADAPTER,
        status=IntegrationStatus.CURRENT,
        evidence="candidate-pins.json and pinned native CLI adapter",
    ),
    IntegrationProvenance(
        id="subscription.codex",
        provider="codex",
        source=None,
        backend="codex-cli",
        version=None,
        license=None,
        auth_kind="consumer_subscription",
        capabilities=("status", "login", "structured_answer", "read_only"),
        usage_provenance="native_or_estimated_or_unavailable",
        reuse=IntegrationReuse.PROTOCOL_REUSE,
        status=IntegrationStatus.CURRENT,
        evidence="verified provider protocol and local CLI contract",
    ),
    IntegrationProvenance(
        id="subscription.claude",
        provider="claude",
        source=None,
        backend="claude-code-cli",
        version=None,
        license=None,
        auth_kind="consumer_subscription",
        capabilities=("status", "login", "structured_answer", "read_only"),
        usage_provenance="native_or_unavailable",
        reuse=IntegrationReuse.PROTOCOL_REUSE,
        status=IntegrationStatus.CURRENT,
        evidence="verified first-party consumer login and JSON CLI contract",
    ),
    IntegrationProvenance(
        id="subscription.kimi",
        provider="kimi",
        source=None,
        backend="kimi-cli",
        version=None,
        license=None,
        auth_kind="unsupported",
        capabilities=(),
        usage_provenance="unavailable",
        reuse=IntegrationReuse.GATED,
        status=IntegrationStatus.GATED,
        evidence="no verified official consumer CLI contract",
    ),
    IntegrationProvenance(
        id="subscription.grok",
        provider="grok",
        source=None,
        backend="grok-cli",
        version=None,
        license=None,
        auth_kind="unsupported",
        capabilities=(),
        usage_provenance="unavailable",
        reuse=IntegrationReuse.GATED,
        status=IntegrationStatus.GATED,
        evidence="no verified official consumer CLI contract",
    ),
    IntegrationProvenance(
        id="embedding.local-hash",
        provider=None,
        source=None,
        backend="local-hash-embedding",
        version=None,
        license=None,
        auth_kind="none",
        capabilities=("embedding", "smoke_only"),
        usage_provenance="unavailable",
        reuse=IntegrationReuse.DIRECT_REUSE,
        status=IntegrationStatus.CURRENT,
        evidence="deterministic local implementation; never recommendation eligible",
    ),
    IntegrationProvenance(
        id="embedding.openai",
        provider="openai",
        source=None,
        backend="openai:text-embedding-3-small",
        version=None,
        license=None,
        auth_kind="api_key",
        capabilities=("embedding", "semantic"),
        usage_provenance="measured_or_unavailable",
        reuse=IntegrationReuse.PROTOCOL_REUSE,
        status=IntegrationStatus.CURRENT,
        evidence="registered OpenAI-compatible embedding transport",
    ),
    IntegrationProvenance(
        id="embedding.gemini",
        provider="google",
        source=None,
        backend="google:gemini-embedding-001",
        version=None,
        license=None,
        auth_kind="api_key",
        capabilities=("embedding", "semantic"),
        usage_provenance="measured_or_unavailable",
        reuse=IntegrationReuse.PROTOCOL_REUSE,
        status=IntegrationStatus.CURRENT,
        evidence="registered Gemini embedding transport; BYOK capability",
    ),
    IntegrationProvenance(
        id="source.google-drive",
        provider="google_drive",
        source="google_drive",
        backend="google-drive-connector",
        version=None,
        license=None,
        auth_kind="oauth",
        capabilities=(),
        usage_provenance="unavailable",
        reuse=IntegrationReuse.GATED,
        status=IntegrationStatus.GATED,
        evidence="connector is explicitly unsupported by current project policy",
    ),
    IntegrationProvenance(
        id="source.confluence",
        provider="confluence",
        source="confluence",
        backend="atlassian-mcp",
        version=None,
        license=None,
        auth_kind="oauth",
        capabilities=(),
        usage_provenance="unavailable",
        reuse=IntegrationReuse.GATED,
        status=IntegrationStatus.GATED,
        evidence="official MCP authentication contract is unverified",
    ),
    IntegrationProvenance(
        id="source.onyx",
        provider="onyx",
        source="onyx",
        backend="onyx",
        version=None,
        license=None,
        auth_kind="unsupported",
        capabilities=(),
        usage_provenance="unavailable",
        reuse=IntegrationReuse.GATED,
        status=IntegrationStatus.GATED,
        evidence=(
            "no design-partner evidence; no verified license, public API, "
            "runtime, ACL, resource, network, or teardown proof"
        ),
    ),
)


# Typed, fail-closed gate evidence for the Onyx design-partner evaluation.
# Every field is False because no verified evidence exists.  The model
# validator in DesignPartnerGateEvidence prevents recommendation_eligible=True
# while any field is False.
ONYX_GATE_EVIDENCE = DesignPartnerGateEvidence(
    integration_id="source.onyx",
    reason=(
        "Onyx design-partner gate is closed: no design-partner evidence, verified "
        "license, public API, runtime, ACL, resource, network, or teardown proof "
        "is present"
    ),
)


def integration_catalog() -> tuple[IntegrationProvenance, ...]:
    """Return the immutable catalog used by artifacts and documentation."""
    return _CATALOG


def design_partner_gate_evidence() -> tuple[DesignPartnerGateEvidence, ...]:
    """Return typed gate evidence for all design-partner-gated integrations."""
    return (ONYX_GATE_EVIDENCE,)
