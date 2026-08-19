from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

import autobrain.production as production
from autobrain.metering import BudgetExceededError, LoopbackMeteringProxy
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

    def probe(self) -> dict[str, list[str]]:
        return {"advertised": ["search"], "allowed": ["search"]}

    def crawl(self, *, include_dms: bool) -> ConnectorSnapshot:
        del include_dms
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
