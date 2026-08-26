"""Fail-closed gate tests for the Onyx design-partner evaluation.

Onyx is not a verified integration: no design-partner evidence, verified
license, public API, runtime, ACL, resource, network, or teardown proof
exists.  These tests prove the gate is machine-readable, fail-closed, and
that Onyx has no production candidate construction path.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autobrain.integration_provenance import (
    ONYX_GATE_EVIDENCE,
    design_partner_gate_evidence,
    integration_catalog,
)
from autobrain.models import (
    CandidateId,
    DesignPartnerGateEvidence,
    IntegrationReuse,
    IntegrationStatus,
    NativeCandidateResult,
)


def test_onyx_is_gated_in_provenance_catalog() -> None:
    catalog = {item.id: item for item in integration_catalog()}
    onyx = catalog["source.onyx"]

    assert onyx.status is IntegrationStatus.GATED
    assert onyx.reuse is IntegrationReuse.GATED
    assert onyx.capabilities == ()
    assert onyx.usage_provenance == "unavailable"
    assert onyx.version is None
    assert onyx.license is None
    assert onyx.auth_kind == "unsupported"


def test_onyx_gate_evidence_is_fail_closed_with_all_fields_false() -> None:
    evidence = ONYX_GATE_EVIDENCE

    assert evidence.integration_id == "source.onyx"
    assert evidence.recommendation_eligible is False
    assert evidence.design_partner_evidence is False
    assert evidence.verified_license is False
    assert evidence.public_api is False
    assert evidence.runtime_proof is False
    assert evidence.acl_proof is False
    assert evidence.resource_proof is False
    assert evidence.network_proof is False
    assert evidence.teardown_proof is False
    assert evidence.reason


def test_design_partner_gate_evidence_returns_onyx() -> None:
    evidence = design_partner_gate_evidence()

    assert len(evidence) == 1
    assert evidence[0].integration_id == "source.onyx"
    assert evidence[0].recommendation_eligible is False


def test_gate_evidence_rejects_eligible_with_any_field_false() -> None:
    with pytest.raises(ValidationError, match="recommendation_eligible"):
        DesignPartnerGateEvidence(
            integration_id="source.onyx",
            design_partner_evidence=True,
            verified_license=False,
            public_api=True,
            runtime_proof=True,
            acl_proof=True,
            resource_proof=True,
            network_proof=True,
            teardown_proof=True,
            recommendation_eligible=True,
            reason="partial evidence",
        )


def test_gate_evidence_accepts_eligible_only_when_all_fields_true() -> None:
    evidence = DesignPartnerGateEvidence(
        integration_id="source.example",
        design_partner_evidence=True,
        verified_license=True,
        public_api=True,
        runtime_proof=True,
        acl_proof=True,
        resource_proof=True,
        network_proof=True,
        teardown_proof=True,
        recommendation_eligible=True,
        reason="all evidence present",
    )

    assert evidence.recommendation_eligible is True


def test_onyx_is_not_a_candidate_id() -> None:
    assert "onyx" not in {candidate.value for candidate in CandidateId}
    with pytest.raises(ValueError):
        CandidateId("onyx")


def test_onyx_is_absent_from_candidate_pins() -> None:
    from autobrain.preflight_support import load_candidate_pins

    pins = load_candidate_pins()

    assert {candidate.id for candidate in pins.candidates} == {
        CandidateId.LLM_WIKI,
        CandidateId.MEM0,
        CandidateId.GBRAIN,
    }


def test_native_candidate_result_rejects_onyx() -> None:
    from autobrain.models import BackendIdentity, CapabilityClass, EvidenceStatus, NativeMode

    with pytest.raises((ValidationError, ValueError)):
        NativeCandidateResult(
            candidate="onyx",  # type: ignore[arg-type]
            mode=NativeMode.SEMANTIC,
            backend=BackendIdentity(name="onyx"),
            capability=CapabilityClass.RETRIEVAL_AND_ANSWER,
            evidence_status=EvidenceStatus.COMPLETE,
            recommendation_eligible=False,
            eligibility_reasons=["onyx is not a verified candidate"],
        )
