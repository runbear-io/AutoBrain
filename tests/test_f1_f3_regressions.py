from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from pydantic import SecretStr

from autobrain.auth.models import AuthStatusReport, ConnectionStatus, Provider, TokenRecord
from autobrain.models import CandidateId, CandidateObservation, ConnectionState, CostStatus, Status
from autobrain.orchestration import (
    CandidateContext,
    CandidateOutcome,
    ConnectorSnapshot,
    RunConfig,
    RunOrchestrator,
)


def _records(count: int = 24) -> list[dict[str, Any]]:
    return [
        {
            "provider": "slack",
            "source_id": f"slack:message:{index}",
            "source_kind": "SLACK_MESSAGE",
            "canonical_url": f"https://fixture.example.test/{index}",
            "title": f"Policy {index}",
            "text": f"Use the documented policy for service {index} within five minutes.",
            "question": f"How do we handle service {index} incidents?",
            "evidence_reply": f"The on-call follows policy {index} and records the incident.",
        }
        for index in range(count)
    ]


class _Connector:
    def __init__(self, provider: str, records: list[dict[str, Any]]) -> None:
        self.provider = provider
        self.records = records
        self.probe_calls = 0
        self.crawl_calls = 0

    def probe(self) -> dict[str, Any]:
        self.probe_calls += 1
        return {"advertised": ["search", "fetch"], "allowed": ["search", "fetch"]}

    def crawl(self, *, include_dms: bool) -> ConnectorSnapshot:
        del include_dms
        self.crawl_calls += 1
        return ConnectorSnapshot(
            provider=self.provider,
            documents=tuple(self.records),
            coverage={"completeness": "SEARCH_DISCOVERED", "discovered": len(self.records)},
        )


class _Candidate:
    def __init__(self, candidate_id: str, *, fail: bool = False) -> None:
        self.candidate_id = candidate_id
        self.fail = fail
        self.calls = 0
        self.cleaned = 0
        self.questions: tuple[str, ...] = ()

    def run(self, context: Any) -> CandidateOutcome:
        self.calls += 1
        self.questions = context.questions
        if self.fail:
            raise RuntimeError("candidate failure")
        return CandidateOutcome(
            candidate=self.candidate_id,
            status=Status.OK,
            score=90,
            answered_cases=len(context.questions),
            scored_cases=len(context.questions),
            cost_usd=1,
        )

    def cleanup(self) -> None:
        self.cleaned += 1


class _ConnectedManager:
    def __init__(self, *, connected: bool = True) -> None:
        state = ConnectionState.CONNECTED if connected else ConnectionState.DISCONNECTED
        status = Status.OK if connected else Status.MCP_AUTH_UNAVAILABLE
        self._status = AuthStatusReport(
            connections=tuple(
                ConnectionStatus(provider=provider, state=state, status=status)
                for provider in Provider
            )
        )
        self._token = TokenRecord(
            provider=Provider.SLACK,
            workspace_id="fixture-workspace",
            user_id="fixture-user",
            audience="https://mcp.slack.com/mcp",
            access_token=SecretStr("fixture-token"),
        )

    def status(self) -> AuthStatusReport:
        return self._status

    def token_for(self, provider: Provider) -> TokenRecord | None:
        if self._status.connections[0].state is not ConnectionState.CONNECTED:
            return None
        if provider is Provider.SLACK:
            return self._token
        return self._token.model_copy(
            update={
                "provider": Provider.NOTION,
                "audience": "https://mcp.notion.com/mcp",
            }
        )


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig(output=tmp_path / "runs", open_report=False)


def test_local_wiring_invokes_injected_connector_and_pinned_candidate_builders(
    tmp_path: Path,
) -> None:
    records = _records()
    connector_calls: list[tuple[str, bool]] = []
    candidate_calls: list[tuple[Path, str, tuple[str, ...]]] = []
    candidates = [_Candidate("llm-wiki", fail=True), _Candidate("mem0"), _Candidate("gbrain")]

    def build_connectors(manager: Any, include_dms: bool) -> tuple[_Connector, _Connector]:
        del manager
        connector_calls.append(("connectors", include_dms))
        return _Connector("slack", records), _Connector("notion", [])

    def build_candidates(run_dir: Path, api_key: str) -> tuple[_Candidate, ...]:
        candidate_calls.append((run_dir, api_key, tuple(item.candidate_id for item in candidates)))
        return tuple(candidates)

    orchestrator = RunOrchestrator.local(
        _config(tmp_path),
        connection_manager=cast(Any, _ConnectedManager()),
        connector_builder=build_connectors,
        candidate_builder=build_candidates,
        api_key="fixture-openai-key",
    )

    result = orchestrator.run()

    assert result.status is Status.OK
    assert connector_calls == [("connectors", False)]
    assert candidate_calls and candidate_calls[0][1] == "fixture-openai-key"
    assert [candidate.calls for candidate in candidates] == [1, 1, 1]
    assert [candidate.cleaned for candidate in candidates] == [1, 1, 1]
    assert result.candidate_results[0].status is Status.FAILED
    assert result.report_path is not None and result.report_path.is_file()
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["experiment_title"]
    assert manifest["config"]["experiment_description"]
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["experiment_title"]
    assert manifest["config"]["experiment_description"]


def test_qualified_question_is_accepted_but_thanks_is_rejected_before_start(
    tmp_path: Path,
) -> None:
    records = _records()
    records[0]["question"] = "Thanks!"
    candidate = _Candidate("llm-wiki")
    orchestrator = RunOrchestrator(
        config=_config(tmp_path),
        connectors=[_Connector("slack", records), _Connector("notion", [])],
        candidates=[candidate],
        provider_available=True,
    )

    result = orchestrator.run()

    assert result.status is Status.OK
    assert candidate.calls == 1
    assert "Thanks!" not in candidate.questions
    assert "How do we handle service 1 incidents?" in candidate.questions


def test_raw_question_without_reply_evidence_is_rejected_before_candidate_start(
    tmp_path: Path,
) -> None:
    records = _records()
    for record in records:
        record.pop("evidence_reply")
        record["text"] = "How do we handle this service incident?"
    candidate = _Candidate("llm-wiki")
    orchestrator = RunOrchestrator(
        config=_config(tmp_path),
        connectors=[_Connector("slack", records), _Connector("notion", [])],
        candidates=[candidate],
        provider_available=True,
    )

    result = orchestrator.run()

    assert result.status is Status.INSUFFICIENT_BENCHMARK
    assert candidate.calls == 0


def test_holdout_leakage_blocks_all_candidates_before_start(tmp_path: Path) -> None:
    records = _records()
    records[0]["text"] = "This note accidentally mentions slack:message:23."
    candidates = [_Candidate("llm-wiki"), _Candidate("mem0"), _Candidate("gbrain")]
    orchestrator = RunOrchestrator(
        config=_config(tmp_path),
        connectors=[_Connector("slack", records), _Connector("notion", [])],
        candidates=candidates,
        provider_available=True,
    )

    result = orchestrator.run()

    assert result.status is Status.LEAKAGE_DETECTED
    assert all(candidate.calls == 0 for candidate in candidates)
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert any(stage["name"] == "leakage-gate" for stage in manifest["stages"])


def test_missing_auth_remains_typed_and_does_not_build_clients(tmp_path: Path) -> None:
    calls: list[str] = []

    def forbidden_builder(*_: Any) -> tuple[_Connector, _Connector]:
        calls.append("connector")
        raise AssertionError("missing auth must not construct MCP clients")

    orchestrator = RunOrchestrator.local(
        _config(tmp_path),
        connection_manager=cast(Any, _ConnectedManager(connected=False)),
        connector_builder=forbidden_builder,
    )

    result = orchestrator.run()

    assert result.status is Status.MCP_AUTH_UNAVAILABLE
    assert calls == []


class _ObservationCandidate:
    def __init__(
        self,
        candidate_id: CandidateId,
        answers: dict[str, str],
        *,
        cost_usd: float | None,
    ) -> None:
        self.candidate_id = candidate_id.value
        self.answers = answers
        self.cost_usd = cost_usd
        self.context: CandidateContext | None = None
        self.cleaned = 0

    def run(self, context: Any) -> CandidateOutcome:
        self.context = context
        observations = tuple(
            CandidateObservation(
                candidate=CandidateId(self.candidate_id),
                case_id=case_id,
                status=Status.OK,
                answer=self.answers[question],
                source_ids=(
                    [f"slack:message:{int(question.rsplit(' ', 2)[1])}"]
                    if "follows policy" in self.answers[question]
                    else ["slack:noise"]
                ),
                latency_ms=5,
            )
            for case_id, question in zip(context.case_ids, context.questions, strict=True)
        )
        return CandidateOutcome(
            candidate=self.candidate_id,
            status=Status.OK,
            score=100.0,
            answered_cases=len(observations),
            scored_cases=len(observations),
            cost_usd=self.cost_usd,
            observations=observations,
        )

    def cleanup(self) -> None:
        self.cleaned += 1


def test_fake_production_scores_reference_claims_not_question_text(
    tmp_path: Path,
) -> None:
    records = _records()
    answers = {record["question"]: record["evidence_reply"] for record in records}
    reference_grounded = _ObservationCandidate(
        CandidateId.MEM0,
        answers,
        cost_usd=1.0,
    )
    question_echo = _ObservationCandidate(
        CandidateId.LLM_WIKI,
        {record["question"]: record["question"] for record in records},
        cost_usd=1.0,
    )

    result = RunOrchestrator(
        config=_config(tmp_path),
        connectors=[_Connector("slack", records), _Connector("notion", [])],
        candidates=[question_echo, reference_grounded],
        provider_available=True,
    ).run()

    assert result.status is Status.OK
    assert result.verdict == CandidateId.MEM0.value
    assert result.candidate_results[0].score < result.candidate_results[1].score
    assert result.candidate_results[1].evaluation is not None
    assert result.candidate_results[1].evaluation.source_support_rate == 1.0


def test_candidate_context_contains_only_documents_and_questions(
    tmp_path: Path,
) -> None:
    records = _records()
    candidate = _ObservationCandidate(
        CandidateId.MEM0,
        {record["question"]: record["evidence_reply"] for record in records},
        cost_usd=1.0,
    )

    result = RunOrchestrator(
        config=_config(tmp_path),
        connectors=[_Connector("slack", records), _Connector("notion", [])],
        candidates=[candidate],
        provider_available=True,
    ).run()

    assert result.status is Status.OK
    assert candidate.context is not None
    assert set(vars(candidate.context)) == {"documents", "questions", "case_ids"}
    serialized_context = json.dumps(vars(candidate.context), default=str).casefold()
    assert "expected_claim" not in serialized_context
    assert "reference_answer" not in serialized_context
    assert "oracle" not in serialized_context
    assert "holdout" not in serialized_context
    assert "forbidden" not in serialized_context
    assert "evidence_reply" not in serialized_context


def test_incomplete_cost_is_excluded_but_complete_fake_metering_can_win(
    tmp_path: Path,
) -> None:
    records = _records()
    answers = {record["question"]: record["evidence_reply"] for record in records}
    incomplete = _ObservationCandidate(CandidateId.LLM_WIKI, answers, cost_usd=None)
    complete = _ObservationCandidate(CandidateId.MEM0, answers, cost_usd=1.0)

    result = RunOrchestrator(
        config=_config(tmp_path),
        connectors=[_Connector("slack", records), _Connector("notion", [])],
        candidates=[incomplete, complete],
        provider_available=True,
    ).run()

    assert result.status is Status.OK
    assert result.verdict == CandidateId.MEM0.value
    assert result.candidate_results[0].evaluation is not None
    assert result.candidate_results[0].evaluation.cost_status is CostStatus.INCOMPLETE

    no_cost = RunOrchestrator(
        config=_config(tmp_path / "no-cost"),
        connectors=[_Connector("slack", records), _Connector("notion", [])],
        candidates=[_ObservationCandidate(CandidateId.MEM0, answers, cost_usd=None)],
        provider_available=True,
    ).run()
    assert no_cost.status is Status.OK
    assert no_cost.verdict == "NO_RECOMMENDATION"
