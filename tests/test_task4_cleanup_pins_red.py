from pathlib import Path
from typing import get_type_hints

from autobrain.models import CandidateId
from autobrain.preflight_support import load_candidate_pins


def test_native_candidates_expose_verified_cleanup_receipts() -> None:
    from autobrain.production import GBrainCandidate, LLMWikiCandidate, Mem0Candidate

    assert get_type_hints(GBrainCandidate.cleanup)["return"].__name__ == "CleanupReceipt"
    assert get_type_hints(LLMWikiCandidate.cleanup)["return"].__name__ == "CleanupReceipt"
    assert get_type_hints(Mem0Candidate.cleanup)["return"].__name__ == "CleanupReceipt"


def test_pin_eligibility_is_derived_from_native_metadata_not_candidate_name() -> None:
    from autobrain.preflight_support import candidate_pin_matches

    pins = load_candidate_pins()
    pin = next(item for item in pins.candidates if item.id is CandidateId.MEM0)
    assert candidate_pin_matches(pin, distribution="mem0ai", version=pin.version, commit=pin.commit)
    assert not candidate_pin_matches(
        pin, distribution="mem0ai", version=pin.version, commit="0" * 40
    )


def test_cleanup_receipt_is_incomplete_when_workspace_remains(tmp_path: Path) -> None:
    from autobrain.lifecycle import CleanupReceipt, cleanup_receipt_complete

    receipt = CleanupReceipt(
        candidate=CandidateId.GBRAIN,
        removed_paths=[str(tmp_path)],
        remaining_paths=[str(tmp_path / "still-there")],
    )
    assert not cleanup_receipt_complete(receipt)


def test_new_candidate_without_complete_pin_or_cleanup_receipt_cannot_be_eligible() -> None:
    from autobrain.decision import eligibility_reasons
    from autobrain.models import (
        BackendIdentity,
        CandidateEvaluation,
        CapabilityClass,
        CorpusIdentity,
        CostStatus,
        EvidenceStatus,
        NativeCandidateResult,
        NativeMode,
        Status,
        UsageSource,
    )

    native = NativeCandidateResult(
        candidate=CandidateId.MEM0,
        mode=NativeMode.SEMANTIC,
        backend=BackendIdentity(name="mem0ai", version="2.0.18", commit="1" * 40),
        capability=CapabilityClass.RETRIEVAL_AND_ANSWER,
        evidence_status=EvidenceStatus.COMPLETE,
        corpus=CorpusIdentity(sha256="a" * 64, document_count=1),
        recommendation_eligible=True,
    )
    evaluation = CandidateEvaluation(
        candidate=CandidateId.MEM0,
        status=Status.OK,
        scored_cases=20,
        answered_cases=20,
        quality_score=90,
        answer_success_rate=1,
        source_support_rate=1,
        contradiction_count=0,
        total_input_tokens=1,
        total_output_tokens=1,
        total_cost_usd=1,
        cost_status=CostStatus.COMPLETE,
        usage_source=UsageSource.MEASURED,
        valid_pin=False,
        corpus_hash="a" * 64,
        native_result=native,
    )
    reasons = eligibility_reasons(evaluation)
    assert any("pin" in reason for reason in reasons)
