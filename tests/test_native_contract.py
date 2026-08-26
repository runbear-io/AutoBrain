import pytest
from pydantic import ValidationError

from autobrain.decision import select_winner
from autobrain.embedding import EmbeddingBackendConfig
from autobrain.models import (
    BackendIdentity,
    CandidateEvaluation,
    CandidateId,
    CapabilityClass,
    CorpusIdentity,
    CostStatus,
    EvidenceStatus,
    NativeCandidateResult,
    NativeMode,
    Status,
    UsageSource,
    Verdict,
)


def _native(candidate: CandidateId, *, eligible: bool = True) -> NativeCandidateResult:
    backend_name = {
        CandidateId.GBRAIN: "gbrain",
        CandidateId.LLM_WIKI: "llm-wiki-compiler",
        CandidateId.MEM0: "mem0ai",
    }[candidate]
    return NativeCandidateResult(
        candidate=candidate,
        mode=NativeMode.SEMANTIC,
        backend=BackendIdentity(name=backend_name, version="1.0", commit="a" * 40),
        capability=CapabilityClass.RETRIEVAL_AND_ANSWER,
        evidence_status=EvidenceStatus.COMPLETE,
        corpus=CorpusIdentity(sha256="b" * 64, document_count=20),
        recommendation_eligible=eligible,
        eligibility_reasons=[] if eligible else ["native evidence is partial"],
    )


def _evaluation(native: NativeCandidateResult, quality: float) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate=native.candidate,
        status=Status.OK,
        scored_cases=20,
        answered_cases=20,
        quality_score=quality,
        answer_success_rate=1,
        source_support_rate=0.8,
        contradiction_count=0,
        total_input_tokens=100,
        total_output_tokens=100,
        total_cost_usd=1,
        cost_status=CostStatus.COMPLETE,
        usage_source=UsageSource.MEASURED,
        valid_pin=True,
        corpus_hash="b" * 64,
        native_result=native,
    )


def test_native_result_is_strict_and_round_trips() -> None:
    result = _native(CandidateId.GBRAIN)
    assert NativeCandidateResult.model_validate_json(result.model_dump_json()) == result


def test_native_result_rejects_unknown_capability_and_bad_identity() -> None:
    with pytest.raises(ValidationError):
        NativeCandidateResult.model_validate(
            {
                **_native(CandidateId.MEM0).model_dump(mode="json"),
                "capability": "invented",
            }
        )


def test_native_result_rejects_forged_backend_identity() -> None:
    with pytest.raises(ValidationError, match="backend"):
        NativeCandidateResult.model_validate(
            _native(CandidateId.MEM0)
            .model_copy(update={"backend": BackendIdentity(name="gbrain", version="1.0")})
            .model_dump(mode="json"),
            strict=False,
        )


def test_native_result_rejects_keyword_answer_capability() -> None:
    with pytest.raises(ValidationError, match="keyword_only"):
        NativeCandidateResult.model_validate(
            _native(CandidateId.GBRAIN)
            .model_copy(
                update={
                    "mode": NativeMode.KEYWORD_ONLY,
                    "capability": CapabilityClass.RETRIEVAL_AND_ANSWER,
                }
            )
            .model_dump(mode="json"),
            strict=False,
        )


@pytest.mark.parametrize("evidence_status", list(EvidenceStatus)[1:])
def test_native_result_rejects_eligibility_without_complete_evidence(
    evidence_status: EvidenceStatus,
) -> None:
    with pytest.raises(ValidationError, match="eligible"):
        NativeCandidateResult.model_validate(
            _native(CandidateId.MEM0)
            .model_copy(update={"evidence_status": evidence_status})
            .model_dump(mode="json"),
            strict=False,
        )


def test_native_result_requires_reasons_to_agree_with_eligibility() -> None:
    with pytest.raises(ValidationError, match="eligibility_reasons"):
        NativeCandidateResult.model_validate(
            _native(CandidateId.MEM0)
            .model_copy(update={"recommendation_eligible": True, "eligibility_reasons": ["stale"]})
            .model_dump(mode="json"),
            strict=False,
        )


def test_candidate_evaluation_binds_native_candidate_and_corpus() -> None:
    with pytest.raises(ValidationError, match="candidate"):
        CandidateEvaluation.model_validate(
            _evaluation(_native(CandidateId.MEM0), 80)
            .model_copy(update={"candidate": CandidateId.GBRAIN})
            .model_dump(mode="json"),
            strict=False,
        )
    with pytest.raises(ValidationError, match="corpus"):
        CandidateEvaluation.model_validate(
            _evaluation(_native(CandidateId.MEM0), 80)
            .model_copy(update={"corpus_hash": "c" * 64})
            .model_dump(mode="json"),
            strict=False,
        )


def test_legacy_candidate_evaluation_migrates_without_native_result() -> None:
    payload = _evaluation(_native(CandidateId.LLM_WIKI), 80).model_dump(mode="json")
    payload.pop("native_result")
    migrated = CandidateEvaluation.model_validate(payload, strict=False)
    assert migrated.native_result is None


def test_native_diagnostics_do_not_replace_recall_quality_order() -> None:
    embedding = EmbeddingBackendConfig.from_environ(
        {"OPENAI_API_KEY": "fixture-key"}, requested="openai"
    ).descriptor
    winner = select_winner(
        [
            _evaluation(_native(CandidateId.LLM_WIKI), 91),
            _evaluation(_native(CandidateId.MEM0), 80),
        ],
        embedding=embedding,
    )
    assert winner.status is Status.OK
    assert winner.verdict is Verdict.LLM_WIKI


def test_ineligible_native_result_is_not_recommendation_eligible() -> None:
    evaluation = _evaluation(_native(CandidateId.GBRAIN, eligible=False), 99)
    result = select_winner([evaluation])
    assert result.verdict is Verdict.NO_RECOMMENDATION
    assert "native evidence is partial" in result.ineligible_candidates[CandidateId.GBRAIN]
