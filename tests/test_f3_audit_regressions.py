from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from autobrain.cancellation import RunCancellation
from autobrain.models import (
    CandidateEvaluation,
    CandidateId,
    CandidateObservation,
    CostStatus,
    Status,
)
from autobrain.orchestration import (
    CandidateContext,
    CandidateOutcome,
    ConnectorSnapshot,
    RunConfig,
    RunOrchestrator,
)


def _documents(count: int = 24) -> list[dict[str, Any]]:
    return [
        {
            "provider": "slack",
            "source_id": f"slack:message:{index}",
            "source_kind": "SLACK_MESSAGE",
            "canonical_url": f"https://fixture.example.test/{index}",
            "title": f"Policy {index}",
            "text": f"Use policy {index} for the service incident response.",
            "question": f"How do we handle service {index} incidents?",
            "evidence_reply": f"The on-call follows policy {index} and records the incident.",
        }
        for index in range(count)
    ]


class _Connector:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.provider = "slack"
        self.records = records

    def probe(self, cancellation: RunCancellation | None = None) -> dict[str, list[str]]:
        return {"advertised": ["search"], "allowed": ["search"]}

    def crawl(self, *, cancellation: RunCancellation | None = None) -> ConnectorSnapshot:
        return ConnectorSnapshot(
            provider=self.provider,
            documents=tuple(self.records),
            coverage={"completeness": "SEARCH_DISCOVERED", "discovered": len(self.records)},
        )


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig(output=tmp_path / "runs", open_report=False)


def _evaluation(
    candidate: CandidateId,
    *,
    quality: float,
    cost: float | None = 1.0,
    cost_status: CostStatus = CostStatus.COMPLETE,
    latency: float = 20,
    burden: float = 1,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate=candidate,
        status=Status.OK,
        scored_cases=20,
        answered_cases=20,
        quality_score=quality,
        answer_success_rate=1.0,
        source_support_rate=0.8,
        contradiction_count=0,
        total_input_tokens=100,
        total_output_tokens=100,
        total_cost_usd=cost,
        cost_status=cost_status,
        query_p50_ms=latency / 2,
        query_p95_ms=latency,
        workspace_bytes=100,
        operating_burden=burden,
        valid_pin=True,
        corpus_hash="a" * 64,
        eligible_override=True,
    )


class _EvaluatingCandidate:
    def __init__(self, candidate_id: CandidateId, evaluation: CandidateEvaluation) -> None:
        self.candidate_id = candidate_id.value
        self.evaluation = evaluation
        self.calls = 0
        self.cleaned = 0
        self.context: CandidateContext | None = None

    def run(self, context: CandidateContext) -> CandidateOutcome:
        self.calls += 1
        self.context = context
        observations = tuple(
            CandidateObservation(
                candidate=CandidateId(self.candidate_id),
                case_id=case_id,
                status=Status.OK,
                answer=(
                    f"The on-call follows policy {int(question.rsplit(' ', 2)[1])} "
                    "and records the incident."
                    if self.evaluation.quality_score >= 78
                    else question
                ),
                source_ids=(
                    [f"slack:message:{int(question.rsplit(' ', 2)[1])}"]
                    if self.evaluation.quality_score >= 78
                    else []
                ),
                latency_ms=int(self.evaluation.query_p95_ms or 1),
            )
            for case_id, question in zip(context.case_ids, context.questions, strict=True)
        )
        return CandidateOutcome(
            candidate=self.candidate_id,
            status=Status.OK,
            # Deliberately disagree with the canonical evaluation. The
            # production path must not select from this legacy shortcut.
            score=100 - self.evaluation.quality_score,
            answered_cases=len(context.questions),
            scored_cases=len(context.questions),
            cost_usd=self.evaluation.total_cost_usd,
            latency_ms=1,
            observations=observations,
            evaluation=self.evaluation,
        )

    def cleanup(self) -> None:
        self.cleaned += 1


@pytest.mark.parametrize(
    ("pre_holdout_count", "expected_candidate_count"),
    (
        (20, 0),
        (21, 20),
        (30, 27),
        (31, 28),
        (40, 30),
    ),
)
def test_normal_benchmark_contract_is_exactly_twenty_to_thirty_after_holdout(
    tmp_path: Path,
    pre_holdout_count: int,
    expected_candidate_count: int,
) -> None:
    records = _documents(pre_holdout_count)

    def run(records: list[dict[str, Any]], directory: Path) -> tuple[Any, Any]:
        candidate = _EvaluatingCandidate(
            CandidateId.MEM0,
            _evaluation(CandidateId.MEM0, quality=90),
        )
        result = RunOrchestrator(
            config=_config(directory),
            connectors=[_Connector(records), _Connector([])],
            candidates=[candidate],
            provider_available=True,
        ).run()
        return result, candidate

    forward, forward_candidate = run(records, tmp_path / "forward")
    reverse, reverse_candidate = run(list(reversed(records)), tmp_path / "reverse")

    assert forward.status is reverse.status
    assert forward_candidate.cleaned == reverse_candidate.cleaned == 1
    forward_manifest = json.loads((forward.run_dir / "manifest.json").read_text(encoding="utf-8"))
    reverse_manifest = json.loads((reverse.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert forward_manifest["benchmark"]["case_count"] == expected_candidate_count
    assert reverse_manifest["benchmark"]["case_count"] == expected_candidate_count
    assert (
        forward_manifest["hashes"]["benchmark_sha256"]
        == reverse_manifest["hashes"]["benchmark_sha256"]
    )
    if expected_candidate_count:
        assert forward.status is Status.OK
        assert forward_candidate.context is not None
        assert reverse_candidate.context is not None
        assert len(forward_candidate.context.questions) == expected_candidate_count
        assert forward_candidate.context.questions == reverse_candidate.context.questions
    else:
        assert forward.status is Status.INSUFFICIENT_BENCHMARK
        assert forward_candidate.calls == reverse_candidate.calls == 0


def test_exactly_twenty_pre_holdout_cases_fail_closed_before_candidate_start(
    tmp_path: Path,
) -> None:
    candidate = _EvaluatingCandidate(
        CandidateId.MEM0,
        _evaluation(CandidateId.MEM0, quality=90),
    )

    result = RunOrchestrator(
        config=_config(tmp_path),
        connectors=[_Connector(_documents(20)), _Connector([])],
        candidates=[candidate],
        provider_available=True,
    ).run()

    assert result.status is Status.INSUFFICIENT_BENCHMARK
    assert result.candidate_results == ()
    assert candidate.calls == 0
    assert candidate.cleaned == 1
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["benchmark"]["case_count"] == 0
    assert any(
        stage["name"] == "cleanup" and stage["status"] == Status.OK.value
        for stage in manifest["stages"]
    )


def test_benchmark_cap_is_order_independent_after_holdout(tmp_path: Path) -> None:
    records = _documents(40)
    forward_candidate = _EvaluatingCandidate(
        CandidateId.MEM0,
        _evaluation(CandidateId.MEM0, quality=90),
    )
    reverse_candidate = _EvaluatingCandidate(
        CandidateId.MEM0,
        _evaluation(CandidateId.MEM0, quality=90),
    )

    forward = RunOrchestrator(
        config=_config(tmp_path / "forward"),
        connectors=[_Connector(records), _Connector([])],
        candidates=[forward_candidate],
        provider_available=True,
    ).run()
    reverse = RunOrchestrator(
        config=_config(tmp_path / "reverse"),
        connectors=[_Connector(list(reversed(records))), _Connector([])],
        candidates=[reverse_candidate],
        provider_available=True,
    ).run()

    assert forward_candidate.context is not None
    assert reverse_candidate.context is not None
    assert len(forward_candidate.context.questions) == 30
    assert forward_candidate.context.questions == reverse_candidate.context.questions
    forward_manifest = json.loads((forward.run_dir / "manifest.json").read_text(encoding="utf-8"))
    reverse_manifest = json.loads((reverse.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert (
        forward_manifest["hashes"]["benchmark_sha256"]
        == reverse_manifest["hashes"]["benchmark_sha256"]
    )
    assert forward.verdict == reverse.verdict == CandidateId.MEM0.value


def test_candidate_boundary_removes_holdout_and_oracle_content_before_start(
    tmp_path: Path,
) -> None:
    records = _documents()
    leaked_reply = records[-1]["evidence_reply"]
    # Copy evaluator content into an otherwise candidate-visible record to
    # prove reference-content scanning happens before any candidate starts.
    records[0]["text"] = f"Candidate-visible copy: {leaked_reply}"
    candidates = [
        _EvaluatingCandidate(
            CandidateId.MEM0,
            _evaluation(CandidateId.MEM0, quality=90),
        )
    ]

    result = RunOrchestrator(
        config=_config(tmp_path),
        connectors=[_Connector(records), _Connector([])],
        candidates=candidates,
        provider_available=True,
    ).run()

    assert result.status is Status.LEAKAGE_DETECTED
    assert candidates[0].calls == 0
    assert candidates[0].cleaned == 1


def test_normal_orchestration_uses_canonical_evaluation_not_score_shortcut(
    tmp_path: Path,
) -> None:
    records = _documents()
    first = _EvaluatingCandidate(
        CandidateId.LLM_WIKI,
        _evaluation(CandidateId.LLM_WIKI, quality=70),
    )
    second = _EvaluatingCandidate(
        CandidateId.MEM0,
        _evaluation(CandidateId.MEM0, quality=90),
    )

    result = RunOrchestrator(
        config=_config(tmp_path),
        connectors=[_Connector(records), _Connector([])],
        candidates=[first, second],
        provider_available=True,
    ).run()

    assert result.status is Status.OK
    assert result.verdict == CandidateId.MEM0.value


def test_normal_orchestration_uses_full_deterministic_tie_break_chain(
    tmp_path: Path,
) -> None:
    records = _documents()
    candidates = [
        _EvaluatingCandidate(
            CandidateId.LLM_WIKI,
            _evaluation(
                CandidateId.LLM_WIKI,
                quality=80,
                cost=1,
                latency=10,
                burden=1,
            ),
        ),
        _EvaluatingCandidate(
            CandidateId.MEM0,
            _evaluation(
                CandidateId.MEM0,
                quality=79,
                cost=0.5,
                latency=100,
                burden=5,
            ),
        ),
        _EvaluatingCandidate(
            CandidateId.GBRAIN,
            _evaluation(
                CandidateId.GBRAIN,
                quality=78,
                cost=0.5,
                latency=100,
                burden=5,
            ),
        ),
    ]

    result = RunOrchestrator(
        config=_config(tmp_path),
        connectors=[_Connector(records), _Connector([])],
        candidates=candidates,
        provider_available=True,
    ).run()

    # All candidates are inside epsilon. Cost removes llm-wiki; latency and
    # operations tie; the stable candidate ID is the final deterministic key.
    assert result.verdict == CandidateId.GBRAIN.value
    assert all(candidate.cleaned == 1 for candidate in candidates)


class _CrashingCandidate:
    candidate_id = CandidateId.LLM_WIKI.value

    def __init__(self) -> None:
        self.calls = 0
        self.cleaned = 0

    def run(self, context: CandidateContext) -> CandidateOutcome:
        del context
        self.calls += 1
        raise RuntimeError("candidate crashed")

    def cleanup(self) -> None:
        self.cleaned += 1


class _MisidentifiedCandidate:
    candidate_id = CandidateId.MEM0.value

    def __init__(self) -> None:
        self.cleaned = 0

    def run(self, context: CandidateContext) -> CandidateOutcome:
        observations: list[CandidateObservation] = []
        for index, question in enumerate(context.questions):
            source_index = int(question.rsplit(" ", 2)[1])
            source_id = f"slack:message:{source_index}"
            case_id = (
                "case-wrong"
                if index == 0
                else f"case-{hashlib.sha256(source_id.encode()).hexdigest()[:16]}"
            )
            observations.append(
                CandidateObservation(
                    candidate=CandidateId.GBRAIN if index == 0 else CandidateId.MEM0,
                    case_id=case_id,
                    status=Status.OK,
                    answer=(f"The on-call follows policy {source_index} and records the incident."),
                    source_ids=[source_id],
                    latency_ms=1,
                )
            )
        return CandidateOutcome(
            candidate=self.candidate_id,
            status=Status.OK,
            answered_cases=len(observations),
            scored_cases=len(observations),
            cost_usd=1.0,
            observations=tuple(observations),
            cost_status=CostStatus.COMPLETE,
        )

    def cleanup(self) -> None:
        self.cleaned += 1


class _CleanupFailingCandidate(_EvaluatingCandidate):
    def cleanup(self) -> None:
        self.cleaned += 1
        raise RuntimeError("cleanup failed")


def test_candidate_failure_is_typed_and_does_not_break_continuity_or_cleanup(
    tmp_path: Path,
) -> None:
    crashed = _CrashingCandidate()
    healthy = _EvaluatingCandidate(
        CandidateId.MEM0,
        _evaluation(CandidateId.MEM0, quality=90),
    )

    result = RunOrchestrator(
        config=_config(tmp_path),
        connectors=[_Connector(_documents()), _Connector([])],
        candidates=[crashed, healthy],
        provider_available=True,
    ).run()

    assert [outcome.status for outcome in result.candidate_results] == [
        Status.FAILED,
        Status.OK,
    ]
    assert result.verdict == CandidateId.MEM0.value
    assert crashed.calls == healthy.calls == 1
    assert crashed.cleaned == healthy.cleaned == 1


def test_observation_identity_mismatch_does_not_rebind_by_position(
    tmp_path: Path,
) -> None:
    candidate = _MisidentifiedCandidate()

    result = RunOrchestrator(
        config=_config(tmp_path),
        connectors=[_Connector(_documents()), _Connector([])],
        candidates=[candidate],
        provider_available=True,
    ).run()

    expected_case_id = f"case-{hashlib.sha256(b'slack:message:0').hexdigest()[:16]}"
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    evaluation = manifest["evaluations"][0]
    evidence = json.loads((result.run_dir / "comparison.json").read_text(encoding="utf-8"))[
        "evidence"
    ]
    first_evidence = next(item for item in evidence if item["case_id"] == expected_case_id)

    assert evaluation["status"] == Status.FAILED.value
    assert evaluation["partial_failures"] == 1
    assert first_evidence["status"] == Status.FAILED.value
    assert first_evidence["case_id"] == expected_case_id
    assert result.verdict == "NO_RECOMMENDATION"


def test_cleanup_failure_propagates_to_result_and_persisted_artifacts(
    tmp_path: Path,
) -> None:
    candidate = _CleanupFailingCandidate(
        CandidateId.MEM0,
        _evaluation(CandidateId.MEM0, quality=90),
    )

    result = RunOrchestrator(
        config=_config(tmp_path),
        connectors=[_Connector(_documents()), _Connector([])],
        candidates=[candidate],
        provider_available=True,
    ).run()

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    comparison = json.loads((result.run_dir / "comparison.json").read_text(encoding="utf-8"))
    cleanup_stage = next(stage for stage in manifest["stages"] if stage["name"] == "cleanup")
    cleanup_ledger = next(entry for entry in manifest["commands"] if entry.get("kind") == "cleanup")

    assert result.status is Status.FAILED
    assert manifest["status"] == Status.FAILED.value
    assert cleanup_stage["status"] == Status.FAILED.value
    assert cleanup_ledger["status"] == Status.FAILED.value
    assert comparison["status"] == Status.FAILED.value
    assert "cleanup failed" in comparison["warnings"][-1]
    report_path = result.report_path
    assert report_path is not None
    assert "FAILED" in report_path.read_text(encoding="utf-8")


def test_normal_orchestration_does_not_eligibilize_incomplete_cost(
    tmp_path: Path,
) -> None:
    records = _documents()
    incomplete = _EvaluatingCandidate(
        CandidateId.LLM_WIKI,
        _evaluation(
            CandidateId.LLM_WIKI,
            quality=99,
            cost=None,
            cost_status=CostStatus.INCOMPLETE,
        ),
    )
    complete = _EvaluatingCandidate(
        CandidateId.MEM0,
        _evaluation(CandidateId.MEM0, quality=80),
    )

    result = RunOrchestrator(
        config=_config(tmp_path),
        connectors=[_Connector(records), _Connector([])],
        candidates=[incomplete, complete],
        provider_available=True,
    ).run()

    assert result.status is Status.OK
    assert result.verdict == CandidateId.MEM0.value

    assert complete.context is not None
    assert set(vars(complete.context)) == {
        "documents",
        "questions",
        "case_ids",
        "cancellation",
    }
    assert 20 <= len(complete.context.questions) <= 30
    candidate_corpus = json.dumps(
        complete.context.documents,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert "evidence_reply" not in candidate_corpus
    assert "question" not in candidate_corpus

    holdout = json.loads((result.run_dir / "evaluator" / "holdout.json").read_text())
    assert holdout["source_ids"]
    assert holdout["documents"]
    report = result.report_path.read_text() if result.report_path is not None else ""
    assert "unknown (COST_INCOMPLETE)" in report


def test_normal_orchestration_applies_the_canonical_quality_floor(
    tmp_path: Path,
) -> None:
    candidate = _EvaluatingCandidate(
        CandidateId.MEM0,
        _evaluation(CandidateId.MEM0, quality=59),
    )

    result = RunOrchestrator(
        config=_config(tmp_path),
        connectors=[_Connector(_documents()), _Connector([])],
        candidates=[candidate],
        provider_available=True,
    ).run()

    assert result.status is Status.OK
    assert result.verdict == "NO_RECOMMENDATION"
