from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from autobrain.auth.models import Provider
from autobrain.cli import app
from autobrain.models import CandidateId, CandidateObservation, SourceKind, Status
from autobrain.orchestration import (
    CandidateOutcome,
    ConnectorSnapshot,
    RunConfig,
    RunOrchestrator,
)
from autobrain.report import load_comparison


def _documents(count: int = 24) -> list[dict[str, Any]]:
    return [
        {
            "provider": "slack" if index % 2 == 0 else "notion",
            "source_id": f"slack:message:{index}" if index % 2 == 0 else f"notion:page:{index}",
            "canonical_url": f"https://fixture.example.test/source/{index}",
            "title": f"Fact {index}",
            "text": f"Project Atlas fact {index} is value {index}.",
            "question": f"What is Project Atlas fact {index}?",
            "expected": f"Project Atlas fact {index} is value {index}.",
            "source_kind": (
                SourceKind.SLACK_MESSAGE.value if index % 2 == 0 else SourceKind.NOTION_PAGE.value
            ),
        }
        for index in range(count)
    ]


class FakeConnector:
    def __init__(self, provider: str, records: list[dict[str, Any]]) -> None:
        self.provider = provider
        self.records: list[Mapping[str, Any]] = [
            record for record in records if record["provider"] == provider
        ]
        self.probe_calls = 0
        self.crawl_calls = 0

    def probe(self) -> dict[str, Any]:
        self.probe_calls += 1
        return {"advertised": ["search", "fetch"], "allowed": ["search", "fetch"]}

    def crawl(self, *, include_dms: bool) -> ConnectorSnapshot:
        self.crawl_calls += 1
        del include_dms
        return ConnectorSnapshot(
            provider=self.provider,
            documents=self.records,
            coverage={"completeness": "SEARCH_DISCOVERED", "discovered": len(self.records)},
        )


class FakeCandidate:
    def __init__(
        self,
        candidate_id: str,
        *,
        score: float = 90.0,
        cost_usd: float = 1.0,
        error: Exception | None = None,
        malformed: bool = False,
    ) -> None:
        self.candidate_id = candidate_id
        self.score = score
        self.cost_usd = cost_usd
        self.error = error
        self.malformed = malformed
        self.calls = 0
        self.cleaned = 0

    def run(self, context: Any) -> CandidateOutcome:
        self.calls += 1
        assert not any("holdout" in str(item).lower() for item in context.documents)
        if self.error is not None:
            raise self.error
        if self.malformed:
            return CandidateOutcome(
                candidate=self.candidate_id,
                status=Status.FAILED,
                detail="malformed answer",
                cost_usd=None,
            )
        return CandidateOutcome(
            candidate=self.candidate_id,
            status=Status.OK,
            score=self.score,
            answered_cases=len(context.questions),
            scored_cases=len(context.questions),
            cost_usd=self.cost_usd,
            artifact={"answer_format": "fake"},
            observations=tuple(
                CandidateObservation(
                    candidate=CandidateId(self.candidate_id),
                    case_id=case_id,
                    status=Status.OK,
                    answer=(
                        f"Project Atlas fact {int(question.split()[-1].rstrip('?'))} "
                        f"is value {int(question.split()[-1].rstrip('?'))}."
                    ),
                    source_ids=[
                        (
                            f"slack:message:{int(question.split()[-1].rstrip('?'))}"
                            if int(question.split()[-1].rstrip("?")) % 2 == 0
                            else f"notion:page:{int(question.split()[-1].rstrip('?'))}"
                        )
                    ],
                    latency_ms=1,
                )
                for case_id, question in zip(context.case_ids, context.questions, strict=True)
            ),
        )

    def cleanup(self) -> None:
        self.cleaned += 1


def _config(tmp_path: Path, **kwargs: Any) -> RunConfig:
    values: dict[str, Any] = {
        "output": tmp_path / "run",
        "budget_usd": 25.0,
        "max_questions": 30,
        "open_report": False,
    }
    values.update(kwargs)
    return RunConfig(**values)


def _orchestrator(
    tmp_path: Path,
    candidates: list[FakeCandidate] | None = None,
    records: list[dict[str, Any]] | None = None,
) -> tuple[RunOrchestrator, list[FakeCandidate]]:
    records = records or _documents()
    candidates = candidates or [
        FakeCandidate("llm-wiki", score=92),
        FakeCandidate("mem0", score=88),
        FakeCandidate("gbrain", score=86),
    ]
    sources = [FakeConnector("slack", records), FakeConnector("notion", records)]
    return (
        RunOrchestrator(
            config=_config(tmp_path),
            connectors=sources,
            candidates=candidates,
            provider_available=True,
        ),
        candidates,
    )


def test_notion_only_snapshot_run_is_partial_non_final_and_has_no_recommendation(
    tmp_path: Path,
) -> None:
    records = _documents(50)
    candidates = [
        FakeCandidate("llm-wiki", score=92),
        FakeCandidate("mem0", score=88),
    ]
    orchestrator = RunOrchestrator(
        config=_config(
            tmp_path,
            selected_sources=(Provider.NOTION,),
            selected_candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
            notion_snapshot_path=tmp_path / "notion-snapshot.json",
        ),
        connectors=[FakeConnector("notion", records)],
        candidates=candidates,
        provider_available=True,
    )

    result = orchestrator.run()

    assert result.status is Status.OK
    assert result.verdict == "NO_RECOMMENDATION"
    comparison = load_comparison(result.run_dir / "comparison.json")
    assert comparison.decision.status is Status.NO_RECOMMENDATION
    slack_coverage = next(
        record for record in comparison.coverage if record.source is SourceKind.SLACK_MESSAGE
    )
    assert slack_coverage.completeness.value == "UNKNOWN"
    assert slack_coverage.discovered == 0
    assert all(candidate.eligible_override is False for candidate in comparison.candidates)
    assert all(
        "Slack source absent; source coverage is partial and non-final"
        in candidate.eligibility_reasons
        for candidate in comparison.candidates
    )


def test_fake_mcp_e2e_persists_one_immutable_run_and_report(tmp_path: Path) -> None:
    orchestrator, candidates = _orchestrator(tmp_path)

    result = orchestrator.run()

    assert result.status is Status.OK
    assert result.run_id
    assert result.report_path is not None and result.report_path.is_file()
    report_html = result.report_path.read_text(encoding="utf-8")
    assert '<script src="http' not in report_html
    assert '<link href="http' not in report_html
    assert "https://fixture.example.test/source/0" in report_html
    assert len(result.candidate_results) == 3
    assert all(candidate.cleaned == 1 for candidate in candidates)
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"][0]["name"] == "preflight"
    assert manifest["stages"][-1]["name"] == "cleanup"
    assert manifest["coverage"]["slack"]["completeness"] == "SEARCH_DISCOVERED"
    assert manifest["benchmark"]["case_count"] >= 20
    assert manifest["hashes"]["corpus_sha256"]
    assert manifest["pins"]["candidates"] == ["llm-wiki", "mem0", "gbrain"]


def test_fake_mcp_run_writes_loadable_canonical_comparison_and_evidence(
    tmp_path: Path,
) -> None:
    orchestrator, _ = _orchestrator(tmp_path)

    result = orchestrator.run()

    assert result.status is Status.OK
    comparison_path = result.run_dir / "comparison.json"
    artifact = load_comparison(comparison_path)
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert {
        "benchmark_hash",
        "decision",
        "evidence",
        "methodology",
        "artifact_paths",
        "warnings",
        "coverage",
        "candidates",
    } <= payload.keys()
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert artifact.benchmark_hash == manifest["hashes"]["benchmark_sha256"]
    assert artifact.decision.verdict.value == result.verdict
    assert len(artifact.evidence) == len(artifact.candidates) * manifest["benchmark"]["case_count"]
    assert artifact.evidence[0].source_ids
    serialized = comparison_path.read_text(encoding="utf-8").casefold()
    assert not any(
        source_id.casefold() in serialized
        for source_id in manifest["benchmark"].get("holdout_source_ids", [])
    )
    assert "expected" not in serialized
    assert artifact.methodology["quality_weights"]
    assert artifact.artifact_paths == {
        "comparison_json": "comparison.json",
        "corpus_freeze": "corpus-freeze.json",
        "manifest": "manifest.json",
        "report_html": "report.html",
    }

    report = result.run_dir / "report.html"
    html = report.read_text(encoding="utf-8")
    assert "Methodology and caveats" in html
    assert "Per-case evidence" in html
    assert "case-" in html
    assert "Sources" in html


def test_fake_e2e_is_repeatable_without_overwriting_prior_run(tmp_path: Path) -> None:
    first, _ = _orchestrator(tmp_path)
    second, _ = _orchestrator(tmp_path)

    first_result = first.run()
    second_result = second.run()

    assert first_result.run_id != second_result.run_id
    assert first_result.run_dir != second_result.run_dir
    assert first_result.report_path is not None
    assert second_result.report_path is not None
    assert first_result.report_path.read_bytes() != b""


def test_candidate_failure_settles_and_later_candidates_run(tmp_path: Path) -> None:
    failed = FakeCandidate("llm-wiki", error=RuntimeError("process interrupted"))
    healthy = [FakeCandidate("mem0"), FakeCandidate("gbrain")]
    orchestrator, candidates = _orchestrator(tmp_path, [failed, *healthy])

    result = orchestrator.run()

    assert result.status is Status.OK
    assert result.candidate_results[0].status is Status.FAILED
    assert [candidate.calls for candidate in candidates] == [1, 1, 1]
    assert all(candidate.cleaned == 1 for candidate in candidates)
    assert "process interrupted" in result.candidate_results[0].detail


def test_budget_exceeded_after_first_candidate_does_not_start_later(tmp_path: Path) -> None:
    candidates = [
        FakeCandidate("llm-wiki", cost_usd=24),
        FakeCandidate("mem0", cost_usd=1),
        FakeCandidate("gbrain", cost_usd=1),
    ]
    orchestrator, candidates = _orchestrator(tmp_path, candidates)

    result = orchestrator.run()

    assert result.candidate_results[0].status is Status.OK
    assert result.candidate_results[1].status is Status.BUDGET_EXCEEDED
    assert result.candidate_results[2].status is Status.BUDGET_EXCEEDED
    assert [candidate.calls for candidate in candidates] == [1, 0, 0]


def test_insufficient_benchmark_is_typed_and_candidates_do_not_start(tmp_path: Path) -> None:
    candidates = [FakeCandidate("llm-wiki"), FakeCandidate("mem0"), FakeCandidate("gbrain")]
    orchestrator, candidates = _orchestrator(tmp_path, candidates, _documents(6))

    result = orchestrator.run()

    assert result.status is Status.INSUFFICIENT_BENCHMARK
    assert all(candidate.calls == 0 for candidate in candidates)
    assert (
        json.loads((result.run_dir / "manifest.json").read_text())["benchmark"]["case_count"] == 0
    )


def test_unknown_notion_and_partial_rate_limit_coverage_survive_report(tmp_path: Path) -> None:
    records = _documents()
    connectors = [FakeConnector("slack", records), FakeConnector("notion", records)]
    connectors[0].crawl = lambda *, include_dms: ConnectorSnapshot(  # type: ignore[method-assign]
        provider="slack",
        documents=[record for record in records if record["provider"] == "slack"],
        coverage={"completeness": "PARTIAL_RATE_LIMIT", "discovered": 12},
    )
    connectors[1].crawl = lambda *, include_dms: ConnectorSnapshot(  # type: ignore[method-assign]
        provider="notion",
        documents=[record for record in records if record["provider"] == "notion"],
        coverage={"completeness": "UNKNOWN", "discovered": 12},
    )
    orchestrator = RunOrchestrator(
        config=_config(tmp_path),
        connectors=connectors,
        candidates=[FakeCandidate("llm-wiki"), FakeCandidate("mem0"), FakeCandidate("gbrain")],
        provider_available=True,
    )

    result = orchestrator.run()

    assert result.status is Status.OK
    report = result.report_path.read_text(encoding="utf-8") if result.report_path else ""
    assert "PARTIAL_RATE_LIMIT" in report
    assert "UNKNOWN" in report


def test_missing_read_capability_stops_before_crawl(tmp_path: Path) -> None:
    class NoReadConnector(FakeConnector):
        def probe(self) -> dict[str, Any]:
            self.probe_calls += 1
            return {"advertised": ["write"], "allowed": []}

    slack = NoReadConnector("slack", _documents())
    notion = FakeConnector("notion", _documents())
    orchestrator = RunOrchestrator(
        config=_config(tmp_path),
        connectors=[slack, notion],
        candidates=[FakeCandidate("llm-wiki"), FakeCandidate("mem0"), FakeCandidate("gbrain")],
        provider_available=True,
    )

    result = orchestrator.run()

    assert result.status is Status.CAPABILITY_UNAVAILABLE
    assert slack.crawl_calls == 0
    assert notion.crawl_calls == 0


def test_incomplete_cost_is_preserved_as_unknown_and_does_not_fabricate_price(
    tmp_path: Path,
) -> None:
    candidates = [
        FakeCandidate("llm-wiki", cost_usd=0),
        FakeCandidate("mem0", cost_usd=0),
        FakeCandidate("gbrain", cost_usd=0),
    ]
    candidates[1].cost_usd = None  # type: ignore[assignment]
    orchestrator, _ = _orchestrator(tmp_path, candidates)

    result = orchestrator.run()

    assert result.status is Status.OK
    assert result.candidate_results[1].cost_usd is None
    assert "unknown" in (
        result.report_path.read_text(encoding="utf-8") if result.report_path else ""
    )


def test_max_questions_caps_benchmark_without_resume_or_overwrite(tmp_path: Path) -> None:
    orchestrator = RunOrchestrator(
        config=_config(tmp_path, max_questions=20),
        connectors=[FakeConnector("slack", _documents()), FakeConnector("notion", _documents())],
        candidates=[FakeCandidate("llm-wiki"), FakeCandidate("mem0"), FakeCandidate("gbrain")],
        provider_available=True,
    )

    result = orchestrator.run()

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["benchmark"]["case_count"] == 20
    assert result.status is Status.OK


@pytest.mark.parametrize(
    ("argv", "needle"),
    [
        (["does-not-exist"], "No such command"),
        (["run", "--budget-usd", "0"], "greater than 0"),
    ],
)
def test_cli_rejects_unknown_command_and_invalid_budget(argv: list[str], needle: str) -> None:
    result = CliRunner().invoke(app, argv)
    assert result.exit_code != 0
    assert needle in result.output


def test_cli_run_missing_provider_is_typed_without_auth_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = CliRunner().invoke(app, ["run", "--no-open"], env={"HOME": str(tmp_path)})
    assert result.exit_code != 0
    assert "MCP_AUTH_UNAVAILABLE" in result.output
    assert "OPENAI_API_KEY" not in result.output


def test_report_command_reopens_local_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _ = _orchestrator(tmp_path)
    run = orchestrator.run()
    monkeypatch.setenv("AUTOBRAIN_RUN_ROOT", str(tmp_path))

    result = CliRunner().invoke(app, ["report", run.run_id, "--no-open"])

    assert result.exit_code == 0
    assert str(run.report_path) in result.stdout


def test_scope_fidelity_has_two_connectors_three_candidates_and_no_server() -> None:
    package_root = Path(__file__).parents[2] / "src" / "autobrain"
    source = (package_root / "orchestration.py").read_text(encoding="utf-8")
    assert source.count('provider="slack"') <= 1
    assert source.count('provider="notion"') <= 1
    assert "requests." not in source
    assert "uvicorn" not in source
    assert "plugbear" not in source.casefold()
