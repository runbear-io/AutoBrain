from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from autobrain.decision import select_winner
from autobrain.models import (
    CandidateCaseEvidence,
    CandidateEvaluation,
    CandidateId,
    ComparisonArtifact,
    CostStatus,
    CoverageCompleteness,
    CoverageRecord,
    SourceKind,
    Status,
)
from autobrain.report import (
    build_comparison,
    load_comparison,
    render_report,
    write_artifacts,
)


def comparison() -> ComparisonArtifact:
    candidates = [
        CandidateEvaluation(
            candidate=name,
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
            query_p50_ms=50,
            query_p95_ms=100,
            workspace_bytes=1000,
            operating_burden=2,
            valid_pin=True,
            corpus_hash="a" * 64,
            direct_leakage=False,
        )
        for name, quality, cost in (
            (CandidateId.LLM_WIKI, 90.0, 1.2),
            (CandidateId.MEM0, 80.0, 0.8),
            (CandidateId.GBRAIN, 70.0, 2.0),
        )
    ]
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
        evidence=[
            CandidateCaseEvidence(
                candidate=CandidateId.LLM_WIKI,
                case_id="case-001",
                status=Status.OK,
                score=84,
                source_ids=["slack:launch"],
                source_urls=["https://example.test/launch?q=<script>"],
                cited_claims=1,
                required_claims=1,
            )
        ],
    )


def test_report_is_deterministic_accessible_escaped_and_offline() -> None:
    artifact = comparison()
    html = render_report(artifact)
    assert html == render_report(artifact)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert '<main aria-label="Comparison report">' in html
    assert "COST_COMPLETE" in html
    assert "UNKNOWN" in html
    assert "https://" in html
    assert "http://" not in html.replace("https://", "")
    assert "<link" not in html
    assert "overflow-x: auto" in html
    assert "Generated cases" in html


def test_report_redacts_secrets_in_warnings() -> None:
    artifact = comparison().model_copy(
        update={"warnings": ["provider error sk-live-secret-123456789"]}
    )
    assert "sk-live-secret-123456789" not in render_report(artifact)
    assert "[REDACTED]" in render_report(artifact)


def test_atomic_artifacts_have_stable_hashes_and_corruption_is_typed(tmp_path: Path) -> None:
    artifact = comparison()
    paths = write_artifacts(artifact, tmp_path)
    comparison_bytes = paths.comparison_json.read_bytes()
    assert hashlib.sha256(comparison_bytes).hexdigest() == paths.comparison_sha256
    assert paths.report_html.exists()
    assert json.loads(comparison_bytes)["verdict"] == "llm-wiki"
    assert load_comparison(paths.comparison_json).run_id == "run-001"
    paths.comparison_json.write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        load_comparison(paths.comparison_json)
