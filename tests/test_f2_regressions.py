from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

import autobrain.production as production
from autobrain.cancellation import RunCancellation
from autobrain.candidates.llm_wiki import LLMWikiObservation, LLMWikiRunResult
from autobrain.metering import BudgetExceededError, LoopbackMeteringProxy
from autobrain.models import (
    CandidateEvaluation,
    CandidateId,
    CandidateObservation,
    CandidatePin,
    CostStatus,
    Status,
)
from autobrain.orchestration import (
    CandidateContext,
    CandidateOutcome,
    ConnectorSnapshot,
    RunConfig,
    RunOrchestrator,
    locate_run,
)
from autobrain.report import load_manifest
from autobrain.runs import compare_runs, list_runs


def _documents(count: int = 24) -> list[dict[str, str]]:
    return [
        {
            "provider": "slack",
            "source_id": f"slack:message:{index}",
            "source_kind": "SLACK_MESSAGE",
            "canonical_url": f"https://fixture.example.test/{index}",
            "title": f"Policy {index}",
            "text": f"Use policy {index} for service incidents.",
            "question": f"How do we handle service {index} incidents?",
            "evidence_reply": f"The on-call follows policy {index}.",
        }
        for index in range(count)
    ]


class _Connector:
    provider = "slack"

    def __init__(self, records: list[dict[str, str]]) -> None:
        self.records = records

    def probe(self, cancellation: RunCancellation | None = None) -> dict[str, list[str]]:
        return {"advertised": ["search"], "allowed": ["search"]}

    def crawl(self, *, cancellation: RunCancellation | None = None) -> ConnectorSnapshot:
        return ConnectorSnapshot(
            provider=self.provider,
            documents=tuple(self.records),
            coverage={"completeness": "SEARCH_DISCOVERED", "discovered": len(self.records)},
        )


def _evaluation(
    candidate: CandidateId,
    *,
    quality: float,
    cost: float = 1.0,
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
        cost_status=CostStatus.COMPLETE,
        query_p50_ms=1,
        query_p95_ms=1,
        workspace_bytes=1,
        operating_burden=1,
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

    def run(self, context: CandidateContext) -> CandidateOutcome:
        self.calls += 1
        observations = tuple(
            CandidateObservation(
                candidate=CandidateId(self.candidate_id),
                case_id=case_id,
                status=Status.OK,
                answer=f"The on-call follows policy {question.rsplit(' ', 2)[1]}.",
                source_ids=[f"slack:message:{question.rsplit(' ', 2)[1]}"],
                latency_ms=1,
            )
            for case_id, question in zip(context.case_ids, context.questions, strict=True)
        )
        return CandidateOutcome(
            candidate=self.candidate_id,
            status=Status.OK,
            answered_cases=len(observations),
            scored_cases=len(observations),
            cost_usd=self.evaluation.total_cost_usd,
            latency_ms=1,
            observations=observations,
            evaluation=self.evaluation,
        )

    def cleanup(self) -> None:
        self.cleaned += 1


class _InterruptingCandidate(_EvaluatingCandidate):
    def run(self, context: CandidateContext) -> CandidateOutcome:
        self.calls += 1
        del context
        raise KeyboardInterrupt


def _orchestrator(
    tmp_path: Path,
    candidates: list[_EvaluatingCandidate],
    *,
    budget_usd: float = 25.0,
) -> RunOrchestrator:
    return RunOrchestrator(
        config=RunConfig(
            output=tmp_path / "runs",
            budget_usd=budget_usd,
            open_report=False,
        ),
        connectors=[_Connector(_documents()), _Connector([])],
        candidates=candidates,
        provider_available=True,
    )


def test_keyboard_interrupt_finalizes_every_cleanup_and_typed_artifacts(
    tmp_path: Path,
) -> None:
    interrupted = _InterruptingCandidate(
        CandidateId.LLM_WIKI,
        _evaluation(CandidateId.LLM_WIKI, quality=99),
    )
    later = [
        _EvaluatingCandidate(
            candidate,
            _evaluation(candidate, quality=90),
        )
        for candidate in (CandidateId.MEM0, CandidateId.GBRAIN)
    ]

    result = _orchestrator(tmp_path, [interrupted, *later]).run()

    assert result.status is Status.CANCELLED
    assert result.status is not Status.OK
    assert [candidate.calls for candidate in [interrupted, *later]] == [1, 0, 0]
    assert all(candidate.cleaned == 1 for candidate in [interrupted, *later])

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    comparison = json.loads((result.run_dir / "comparison.json").read_text(encoding="utf-8"))
    assert manifest["status"] == Status.CANCELLED.value
    assert comparison["status"] == Status.CANCELLED.value
    assert any("interrupt" in warning.lower() for warning in comparison["warnings"])
    assert result.report_path is not None and result.report_path.is_file()
    assert Status.CANCELLED.value in result.report_path.read_text(encoding="utf-8")


def test_crossing_budget_is_typed_and_stops_later_candidates_without_winner(
    tmp_path: Path,
) -> None:
    candidates = [
        _EvaluatingCandidate(
            CandidateId.LLM_WIKI,
            _evaluation(CandidateId.LLM_WIKI, quality=99, cost=2.0),
        ),
        _EvaluatingCandidate(
            CandidateId.MEM0,
            _evaluation(CandidateId.MEM0, quality=98, cost=0.1),
        ),
        _EvaluatingCandidate(
            CandidateId.GBRAIN,
            _evaluation(CandidateId.GBRAIN, quality=97, cost=0.1),
        ),
    ]

    result = _orchestrator(tmp_path, candidates, budget_usd=1.5).run()

    assert result.status is Status.BUDGET_EXCEEDED
    assert result.verdict == "NO_RECOMMENDATION"
    assert result.candidate_results[0].status is Status.BUDGET_EXCEEDED
    assert [candidate.calls for candidate in candidates] == [1, 0, 0]
    assert all(candidate.cleaned == 1 for candidate in candidates)

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    comparison = json.loads((result.run_dir / "comparison.json").read_text(encoding="utf-8"))
    assert manifest["status"] == Status.BUDGET_EXCEEDED.value
    assert comparison["status"] == Status.BUDGET_EXCEEDED.value
    assert comparison["decision"]["verdict"] == "NO_RECOMMENDATION"
    assert not comparison["decision"]["eligible_candidates"]


def test_loopback_metering_rejects_crossing_request_and_blocks_later_requests() -> None:
    calls = 0

    def upstream(_payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "id": "request-1",
            "usage": {"prompt_tokens": 5_000_000, "completion_tokens": 0},
        }

    with LoopbackMeteringProxy(upstream, budget_usd=1.0) as proxy:
        with pytest.raises(BudgetExceededError):
            proxy.chat(
                {"model": "gpt-5-mini", "messages": []},
                candidate="mem0",
                phase="query",
            )
        with pytest.raises(BudgetExceededError):
            proxy.chat(
                {"model": "gpt-5-mini", "messages": []},
                candidate="mem0",
                phase="query",
            )

    assert calls == 1


def test_production_candidates_share_budget_boundary_and_stop_after_measured_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def upstream(_payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "id": "production-request-1",
            "usage": {"prompt_tokens": 5_000_000, "completion_tokens": 0},
        }

    class FakeLLMWikiAdapter:
        def __init__(self, config: object) -> None:
            self.config = config

        def cleanup(self) -> None:
            return

    class FakeMem0Adapter:
        def __init__(self, config: object) -> None:
            self.config = config

        def cleanup(self) -> None:
            return

    class FakeGBrainAdapter:
        def __init__(self, **_: object) -> None:
            return

        def cleanup(self) -> None:
            return

    monkeypatch.setattr(production, "LLMWikiAdapter", FakeLLMWikiAdapter)
    monkeypatch.setattr(production, "Mem0Adapter", FakeMem0Adapter)
    monkeypatch.setattr(production, "GBrainAdapter", FakeGBrainAdapter)

    candidates = production.build_production_candidates(
        tmp_path / "run",
        api_key="fixture-provider-key",
        budget_usd=1.0,
        provider_upstream=upstream,
    )

    assert [candidate.candidate_id for candidate in candidates] == [
        CandidateId.LLM_WIKI.value,
        CandidateId.MEM0.value,
        CandidateId.GBRAIN.value,
    ]
    assert len(candidates) == 3
    assert all(getattr(candidate, "metering_proxy", None) is not None for candidate in candidates)

    first = cast(production.LLMWikiCandidate, candidates[0])
    first_proxy = first.metering_proxy
    with pytest.raises(BudgetExceededError):
        first_proxy.chat(
            {"model": "gpt-5-mini", "messages": []},
            candidate=first.candidate_id,
            phase="query",
        )
    with pytest.raises(BudgetExceededError):
        cast(production.Mem0Candidate, candidates[1]).metering_proxy.chat(
            {"model": "gpt-5-mini", "messages": []},
            candidate=candidates[1].candidate_id,
            phase="query",
        )

    assert calls == 1
    assert first_proxy.spent_usd > 1.0


def test_persistence_boundary_redacts_nested_candidate_details_and_report(
    tmp_path: Path,
) -> None:
    secret = "sk-synthetic-provider-secret-123456789"

    class SecretCandidate:
        candidate_id = CandidateId.LLM_WIKI.value

        def run(self, context: CandidateContext) -> CandidateOutcome:
            observations = tuple(
                CandidateObservation(
                    candidate=CandidateId.LLM_WIKI,
                    case_id=case_id,
                    status=Status.OK,
                    answer=f"safe answer {secret}",
                    source_ids=[],
                    latency_ms=1,
                )
                for case_id in context.case_ids
            )
            return CandidateOutcome(
                candidate=CandidateId.LLM_WIKI.value,
                status=Status.OK,
                answered_cases=len(observations),
                scored_cases=len(observations),
                cost_usd=1.0,
                latency_ms=1,
                detail=f"provider exception detail: {secret}",
                artifact={
                    "provider_exception": {"detail": secret},
                    "nested": [{"observation_detail": secret}],
                },
                observations=observations,
                cost_status=CostStatus.COMPLETE,
            )

        def cleanup(self) -> None:
            return

    result = _orchestrator(
        tmp_path,
        [SecretCandidate()],  # type: ignore[list-item]
    ).run()

    assert result.status is Status.OK
    persisted = [
        path.read_text(encoding="utf-8") for path in (result.run_dir / "candidates").glob("*.json")
    ]
    persisted.append((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert persisted
    assert all(secret not in text for text in persisted)
    assert all("[REDACTED]" in text for text in persisted)
    assert result.report_path is not None
    assert secret not in result.report_path.read_text(encoding="utf-8")


def test_llm_wiki_candidate_pin_survives_full_run_persistence_and_reopen(
    tmp_path: Path,
) -> None:
    pin = CandidatePin(
        id=CandidateId.LLM_WIKI,
        distribution="llm-wiki-compiler",
        version="1.1.0",
        commit="3e17bcfe8b50f24c14c6bcda0cb9224d94fd8206",
        repository="https://github.com/atomicstrata/llm-wiki-compiler",
        license="MIT",
    )

    class FakeLLMWikiAdapter:
        def run(self, *_args: object, **_kwargs: object) -> LLMWikiRunResult:
            observations = tuple(
                LLMWikiObservation(
                    case_id=f"case-{index}",
                    question=f"How do we handle service {index} incidents?",
                    answer=f"The on-call follows policy {index}.",
                    citations=(f"slack:message:{index}",),
                    source_ids=(f"slack:message:{index}",),
                    page_ids=(),
                    latency_ms=1,
                    raw_result_path=f"results/{index}.json",
                )
                for index in range(20)
            )
            return LLMWikiRunResult(
                status=Status.OK,
                skipped=False,
                pin=pin,
                workspace=str(tmp_path / "workspace"),
                environment={},
                commands=(),
                observations=observations,
                artifacts=(),
                warnings=(),
                metering_events=(),
                measured_cost_usd=None,
                elapsed_ms=20,
                workspace_bytes=1,
                workspace_seal_sha256="a" * 64,
            )

        def cleanup(self) -> None:
            return

    candidate = production.LLMWikiCandidate(
        FakeLLMWikiAdapter(),  # type: ignore[arg-type]
        api_key="fixture-provider-key",
        metering_proxy=LoopbackMeteringProxy(lambda _payload: {}),
    )
    result = _orchestrator(tmp_path, [candidate]).run()  # type: ignore[list-item]

    assert result.status is Status.OK
    candidate_path = result.run_dir / "candidates" / "llm-wiki.json"
    manifest_path = result.run_dir / "manifest.json"
    comparison_path = result.run_dir / "comparison.json"
    assert candidate_path.is_file()
    assert manifest_path.is_file()
    assert comparison_path.is_file()
    assert result.report_path is not None and result.report_path.is_file()

    candidate_artifact = json.loads(candidate_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    expected_pin = pin.model_dump(mode="json")
    assert candidate_artifact["artifact"]["pin"] == expected_pin
    assert manifest["candidates"][0]["artifact"]["pin"] == expected_pin
    assert manifest["schema_version"] == 2
    assert comparison["schema_version"] == 2
    assert manifest["hashes"] == {
        "benchmark_sha256": comparison["benchmark_hash"],
        "corpus_sha256": comparison["corpus_hash"],
    }

    before = {path: path.read_bytes() for path in result.run_dir.rglob("*") if path.is_file()}
    loaded = load_manifest(manifest_path)
    inventory = list_runs(tmp_path / "runs")
    self_comparison = compare_runs(tmp_path / "runs", result.run_id, result.run_id)
    reopened = locate_run(result.run_id, roots=[tmp_path / "runs"])

    assert loaded.run_id == result.run_id
    assert [item.run_id for item in inventory.runs] == [result.run_id]
    assert self_comparison.status == "EQUIVALENT"
    assert self_comparison.equivalent is True
    assert reopened == result.run_dir
    assert reopened is not None
    assert (reopened / "report.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    assert {
        path: path.read_bytes() for path in result.run_dir.rglob("*") if path.is_file()
    } == before


def test_persisted_token_metrics_remain_strictly_loadable_at_inventory_boundaries(
    tmp_path: Path,
) -> None:
    first = _orchestrator(
        tmp_path,
        [_EvaluatingCandidate(CandidateId.LLM_WIKI, _evaluation(CandidateId.LLM_WIKI, quality=90))],
    ).run()
    second = _orchestrator(
        tmp_path,
        [_EvaluatingCandidate(CandidateId.LLM_WIKI, _evaluation(CandidateId.LLM_WIKI, quality=90))],
    ).run()
    manifest = json.loads((first.run_dir / "manifest.json").read_text(encoding="utf-8"))

    evaluation = manifest["evaluations"][0]
    candidate = manifest["candidates"][0]["evaluation"]
    assert evaluation["total_input_tokens"] == 0
    assert evaluation["total_output_tokens"] == 0
    assert candidate["total_input_tokens"] == 0
    assert candidate["total_output_tokens"] == 0
    assert isinstance(manifest["schema_version"], int)
    assert isinstance(manifest["config"]["open_report"], bool)
    assert manifest["timings"]["total_ms"] is None or isinstance(
        manifest["timings"]["total_ms"], int | float
    )

    loaded = load_manifest(first.run_dir / "manifest.json")
    inventory = list_runs(tmp_path / "runs")
    comparison = compare_runs(tmp_path / "runs", first.run_id, second.run_id)
    reopened = locate_run(first.run_id, roots=[tmp_path / "runs"])

    assert loaded.run_id == first.run_id
    assert {item.run_id for item in inventory.runs} == {first.run_id, second.run_id}
    assert comparison.status == "EQUIVALENT"
    assert reopened == first.run_dir
    assert reopened is not None
    assert (reopened / "report.html").is_file()
