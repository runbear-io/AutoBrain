from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

from autobrain.decision import select_winner
from autobrain.models import (
    CandidateCaseEvidence,
    CandidateEvaluation,
    CandidateId,
    CostStatus,
    CoverageCompleteness,
    CoverageRecord,
    SourceKind,
    Status,
    Verdict,
)
from autobrain.report import build_comparison, render_report, write_artifacts


class _LinkAndTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value is not None:
                self.hrefs.append(value)

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def _candidate(
    name: CandidateId,
    quality: float,
    *,
    cost: float | None = 1.0,
    cost_status: CostStatus = CostStatus.COMPLETE,
    latency: float = 100.0,
    burden: float = 2.0,
    status: Status = Status.OK,
    partial_failures: int = 0,
    eligible_override: bool | None = True,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate=name,
        status=status,
        scored_cases=20,
        answered_cases=20 - partial_failures,
        quality_score=quality,
        answer_success_rate=(20 - partial_failures) / 20,
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
        partial_failures=partial_failures,
        eligible_override=eligible_override,
    )


def _comparison(
    *,
    candidates: list[CandidateEvaluation],
    evidence: list[CandidateCaseEvidence] | None = None,
    artifact_paths: dict[str, str] | None = None,
    warnings: list[str] | None = None,
):
    return build_comparison(
        run_id="run-001",
        corpus_hash="b" * 64,
        benchmark_hash="c" * 64,
        coverage=[
            CoverageRecord(
                source=SourceKind.NOTION_PAGE,
                completeness=CoverageCompleteness.UNKNOWN,
                discovered=4,
                fetched=3,
                denied=1,
            )
        ],
        candidates=candidates,
        decision=select_winner(candidates),
        evidence=evidence or [],
        artifact_paths=artifact_paths,
        warnings=warnings,
    )


def _assert_320px_layout_contract(html_text: str) -> None:
    css = html_text.split("<style>", 1)[1].split("</style>", 1)[0]
    assert ".page" in css and "min-width:0" in css
    assert ".table-wrap" in css and "max-width:100%" in css
    assert "overflow-wrap:anywhere" in css
    assert "html, body { max-width:100%; overflow-x:hidden; }" not in css


def test_report_boundary_color_meets_design_contrast_target() -> None:
    html_text = render_report(
        _comparison(
            candidates=[
                _candidate(CandidateId.LLM_WIKI, 90),
                _candidate(CandidateId.MEM0, 80),
            ]
        )
    )
    css = html_text.split("<style>", 1)[1].split("</style>", 1)[0]
    assert "--line:#64748b" in css


def test_complete_and_incomplete_reports_have_no_document_overflow_at_320px(
    tmp_path: Path,
) -> None:
    long_path = "artifacts/" + ("very-long-evidence-segment-" * 30) + ".json"
    evidence = [
        CandidateCaseEvidence(
            candidate=CandidateId.LLM_WIKI,
            case_id="case-001",
            status=Status.FAILED,
            score=0,
            source_ids=["slack:launch"],
            source_urls=["https://example.test/" + ("source-segment-" * 30)],
            cited_claims=0,
            required_claims=1,
            failure_detail="partial provider failure",
        )
    ]
    complete = _comparison(
        candidates=[
            _candidate(CandidateId.LLM_WIKI, 90),
            _candidate(CandidateId.MEM0, 80),
            _candidate(CandidateId.GBRAIN, 70),
        ],
        evidence=evidence,
        artifact_paths={"native": long_path},
    )
    incomplete = _comparison(
        candidates=[
            _candidate(
                CandidateId.LLM_WIKI,
                90,
                cost=None,
                cost_status=CostStatus.INCOMPLETE,
            ),
            _candidate(
                CandidateId.MEM0,
                80,
                cost=None,
                cost_status=CostStatus.INCOMPLETE,
            ),
            _candidate(
                CandidateId.GBRAIN,
                70,
                cost=None,
                cost_status=CostStatus.UNAVAILABLE,
            ),
        ],
        evidence=evidence,
        artifact_paths={"native": long_path},
    )

    for artifact in (complete, incomplete):
        _assert_320px_layout_contract(render_report(artifact))


def test_partial_or_failed_candidates_never_win_but_remain_visible() -> None:
    result = select_winner(
        [
            _candidate(CandidateId.LLM_WIKI, 99, partial_failures=1),
            _candidate(CandidateId.MEM0, 80),
        ]
    )
    assert result.verdict == CandidateId.MEM0
    artifact = _comparison(
        candidates=[
            _candidate(CandidateId.LLM_WIKI, 99, partial_failures=1),
            _candidate(CandidateId.MEM0, 80),
        ]
    )
    report = render_report(artifact)
    assert "llm-wiki" in report
    assert "partial" in report.lower()


def test_no_valid_candidates_returns_no_recommendation() -> None:
    result = select_winner(
        [
            _candidate(CandidateId.LLM_WIKI, 99, status=Status.FAILED),
            _candidate(CandidateId.MEM0, 80, eligible_override=False),
        ]
    )
    assert result.verdict.value == "NO_RECOMMENDATION"


def test_incomplete_cost_is_ineligible_even_with_better_latency() -> None:
    result = select_winner(
        [
            _candidate(
                CandidateId.LLM_WIKI,
                80,
                cost=None,
                cost_status=CostStatus.INCOMPLETE,
                latency=200,
                burden=1,
            ),
            _candidate(
                CandidateId.MEM0,
                78,
                cost=None,
                cost_status=CostStatus.UNAVAILABLE,
                latency=100,
                burden=3,
            ),
        ]
    )
    assert result.verdict == Verdict.NO_RECOMMENDATION
    assert result.status == Status.NO_RECOMMENDATION
    assert "cost is incomplete" in result.rationale


def test_incomplete_cost_is_ineligible_even_with_better_operating_burden() -> None:
    result = select_winner(
        [
            _candidate(
                CandidateId.LLM_WIKI,
                80,
                cost=None,
                cost_status=CostStatus.INCOMPLETE,
                latency=100,
                burden=2,
            ),
            _candidate(
                CandidateId.MEM0,
                80,
                cost=None,
                cost_status=CostStatus.INCOMPLETE,
                latency=100,
                burden=1,
            ),
        ]
    )
    assert result.verdict == Verdict.NO_RECOMMENDATION
    assert "cost is incomplete" in result.rationale


def test_quality_outside_epsilon_still_requires_complete_cost() -> None:
    result = select_winner(
        [
            _candidate(
                CandidateId.LLM_WIKI,
                86,
                cost=None,
                cost_status=CostStatus.INCOMPLETE,
                latency=500,
                burden=10,
            ),
            _candidate(
                CandidateId.MEM0,
                80,
                cost=None,
                cost_status=CostStatus.INCOMPLETE,
                latency=1,
                burden=0,
            ),
        ]
    )
    assert result.verdict == Verdict.NO_RECOMMENDATION
    assert "cost is incomplete" in result.rationale


def test_browser_unavailable_is_qa_metadata_not_comparison_vocabulary() -> None:
    artifact = _comparison(
        candidates=[
            _candidate(CandidateId.LLM_WIKI, 90),
            _candidate(CandidateId.MEM0, 80),
        ],
        warnings=["BROWSER_UNAVAILABLE: screenshots skipped"],
    )
    serialized = artifact.model_dump_json()
    report = render_report(artifact)
    assert "BROWSER_UNAVAILABLE" not in serialized
    assert "BROWSER_UNAVAILABLE" not in report


def test_metering_roles_are_explicit_filterable_and_mixed_roles_do_not_aggregate() -> None:
    from autobrain.metering import (
        MeteringEvent,
        MeteringRole,
        PriceQuote,
        PriceSheet,
        reconcile_usage,
    )

    prices = PriceSheet(
        version="test",
        effective_date="2026-08-01",
        models={"gpt-5-mini": PriceQuote(input_usd_per_million=1, output_usd_per_million=1)},
    )
    events = [
        MeteringEvent(
            event_id="candidate-query",
            request_id="candidate-query",
            candidate="mem0",
            phase="query",
            role=MeteringRole.CANDIDATE,
            model="gpt-5-mini",
            input_tokens=10,
            output_tokens=5,
            tags={"candidate": "mem0", "phase": "query", "role": "candidate"},
        ),
        MeteringEvent(
            event_id="generator-query",
            request_id="generator-query",
            candidate="benchmark",
            phase="query",
            role=MeteringRole.GENERATOR,
            model="gpt-5-mini",
            input_tokens=1000,
            output_tokens=1000,
            tags={"candidate": "benchmark", "phase": "query", "role": "generator"},
        ),
    ]
    mixed = reconcile_usage(events, prices)
    assert mixed.cost_status == CostStatus.INCOMPLETE
    candidate_only = reconcile_usage(events, prices, role=MeteringRole.CANDIDATE, candidate="mem0")
    assert candidate_only.input_tokens == 10
    assert candidate_only.output_tokens == 5
    reversed_candidate_only = reconcile_usage(
        list(reversed(events)),
        prices,
        role=MeteringRole.CANDIDATE,
        candidate="mem0",
    )
    assert reversed_candidate_only.model_dump(mode="json") == candidate_only.model_dump(mode="json")


def test_free_local_event_does_not_make_paid_run_cost_complete() -> None:
    from autobrain.metering import MeteringEvent, PriceQuote, PriceSheet, reconcile_usage

    prices = PriceSheet(
        version="test",
        effective_date="2026-08-01",
        models={"local-free": PriceQuote(input_usd_per_million=0, output_usd_per_million=0)},
    )
    result = reconcile_usage(
        [
            MeteringEvent(
                event_id="local",
                request_id="local",
                candidate="mem0",
                phase="local",
                model="local-free",
                input_tokens=0,
                output_tokens=0,
                tags={"candidate": "mem0", "phase": "local", "role": "candidate"},
            )
        ],
        prices,
    )
    assert result.cost_status == CostStatus.INCOMPLETE
    assert result.usd is None


def test_unsafe_source_urls_are_not_emitted_as_hrefs() -> None:
    evidence = [
        CandidateCaseEvidence(
            candidate=CandidateId.MEM0,
            case_id="case-unsafe",
            status=Status.OK,
            score=80,
            source_urls=[
                "javascript:alert(1)",
                "data:text/html,<script>alert(1)</script>",
                "file:///etc/passwd",
                "https://user:password@example.test/private",
                "https://safe.example/source",
            ],
            cited_claims=1,
            required_claims=1,
        )
    ]
    report = render_report(
        _comparison(candidates=[_candidate(CandidateId.MEM0, 80)], evidence=evidence)
    )
    assert "javascript:" not in report
    assert "data:" not in report
    assert "file:" not in report
    assert "user:password@" not in report
    assert 'href="https://safe.example/source"' in report


def test_comparison_and_report_redact_nested_secrets_and_oracle_text() -> None:
    artifact = _comparison(
        candidates=[
            _candidate(CandidateId.MEM0, 80),
        ],
        evidence=[
            CandidateCaseEvidence(
                candidate=CandidateId.MEM0,
                case_id="case-001",
                status=Status.FAILED,
                score=0,
                source_ids=["oracle:secret-source"],
                source_urls=["https://api-key:sk-live-secret-123456789@example.test"],
                cited_claims=0,
                required_claims=1,
                failure_detail=(
                    "reference answer: evaluator-only oracle text sk-live-secret-123456789"
                ),
            )
        ],
        artifact_paths={"raw_metering": "oracle-reference/sk-live-secret-123456789.json"},
        warnings=["nested detail bearer super-secret-token"],
    )
    serialized = artifact.model_dump_json()
    report = render_report(artifact)
    for value in (
        "sk-live-secret-123456789",
        "super-secret-token",
        "reference answer",
        "evaluator-only oracle text",
    ):
        assert value not in serialized
        assert value not in report
    assert "[REDACTED]" in serialized
    assert "[REDACTED]" in report
    assert "<script" not in report.lower()


def test_unsafe_nested_source_urls_never_reach_serialized_artifacts(
    tmp_path: Path,
) -> None:
    safe_url = "https://safe.example/source?q=1&label=%3Cok%3E"
    unsafe_urls = [
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "file:///etc/passwd",
        "https://user:password@example.test/private",
        "//example.test/protocol-relative",
        "ftp://example.test/unsupported",
        "https://[broken.example/path",
        "https://bad host.example/path",
    ]
    source_ids = ["slack:launch", "notion:page-1"]
    artifact = _comparison(
        candidates=[_candidate(CandidateId.MEM0, 80)],
        evidence=[
            CandidateCaseEvidence(
                candidate=CandidateId.MEM0,
                case_id="case-url-policy",
                status=Status.OK,
                score=80,
                source_ids=source_ids,
                source_urls=[*unsafe_urls, safe_url],
                cited_claims=1,
                required_claims=1,
            )
        ],
    )

    model_payload = artifact.model_dump(mode="json")
    model_text = json.dumps(model_payload, sort_keys=True)
    assert model_payload["evidence"][0]["source_ids"] == source_ids
    assert model_payload["evidence"][0]["source_urls"] == [safe_url]

    paths = write_artifacts(artifact, tmp_path)
    comparison_text = paths.comparison_json.read_text(encoding="utf-8")
    comparison_payload = json.loads(comparison_text)
    report = paths.report_html.read_text(encoding="utf-8")
    parser = _LinkAndTextParser()
    parser.feed(report)

    assert comparison_payload["evidence"][0]["source_ids"] == source_ids
    assert comparison_payload["evidence"][0]["source_urls"] == [safe_url]
    assert parser.hrefs == [safe_url]
    assert safe_url in comparison_text
    assert "https://safe.example/source?q=1&amp;label=%3Cok%3E" in report

    rendered_text = "".join(parser.text)
    for unsafe_url in unsafe_urls:
        assert unsafe_url not in model_text
        assert unsafe_url not in comparison_text
        assert unsafe_url not in report
        assert unsafe_url not in parser.hrefs
        assert unsafe_url not in rendered_text
