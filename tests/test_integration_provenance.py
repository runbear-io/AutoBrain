from __future__ import annotations

import json
from pathlib import Path

from autobrain.integration_provenance import integration_catalog
from autobrain.models import IntegrationReuse, IntegrationStatus
from autobrain.orchestration import RunConfig, RunOrchestrator
from autobrain.report import render_report


def test_documented_catalog_matches_runtime_catalog() -> None:
    documented = json.loads(Path("docs/integration-provenance.json").read_text(encoding="utf-8"))

    assert documented["schema_version"] == 1
    assert documented["integrations"] == [
        item.model_dump(mode="json") for item in integration_catalog()
    ]


def test_catalog_has_complete_machine_readable_fields_without_invented_metadata() -> None:
    records = {item.id: item for item in integration_catalog()}

    assert records["candidate.llm-wiki"].version == "1.1.0"
    assert records["candidate.llm-wiki"].license == "MIT"
    assert records["candidate.mem0"].license == "Apache-2.0"
    assert records["candidate.gbrain"].version == "0.46.19.0"
    assert records["subscription.claude"].version is None
    assert records["subscription.claude"].license is None
    assert records["source.notion-mcp"].license is None
    assert records["embedding.local-hash"].license is None
    assert records["candidate.llm-wiki"].reuse is IntegrationReuse.THIN_ADAPTER
    assert records["subscription.codex"].reuse is IntegrationReuse.PROTOCOL_REUSE
    assert records["embedding.local-hash"].reuse is IntegrationReuse.DIRECT_REUSE


def test_gated_surfaces_have_no_claimed_capability_or_usage() -> None:
    gated = [item for item in integration_catalog() if item.status is IntegrationStatus.GATED]

    assert {item.id for item in gated} == {
        "source.google-drive",
        "source.confluence",
        "source.onyx",
        "subscription.kimi",
        "subscription.grok",
    }
    assert all(item.reuse is IntegrationReuse.GATED for item in gated)
    assert all(item.capabilities == () for item in gated)
    assert all(item.usage_provenance == "unavailable" for item in gated)
    assert all(item.version is None and item.license is None for item in gated)


def test_runtime_provenance_and_html_report_round_trip_catalog(tmp_path: Path) -> None:
    orchestrator = RunOrchestrator(
        config=RunConfig(output=tmp_path / "runs", open_report=False),
        connectors=(),
        candidates=(),
        provider_available=False,
    )
    provenance = orchestrator.benchmark_provenance()

    assert provenance.integrations == list(integration_catalog())
    from autobrain.decision import select_winner
    from autobrain.models import (
        CandidateEvaluation,
        CandidateId,
        CostStatus,
        EmbeddingProvenance,
        EmbeddingQuality,
        Status,
        UsageSource,
    )
    from autobrain.report import build_comparison

    candidate = CandidateEvaluation(
        candidate=CandidateId.MEM0,
        status=Status.OK,
        scored_cases=20,
        answered_cases=20,
        quality_score=90,
        answer_success_rate=1,
        source_support_rate=1,
        contradiction_count=0,
        total_input_tokens=10,
        total_output_tokens=5,
        total_cost_usd=0.01,
        cost_status=CostStatus.COMPLETE,
        usage_source=UsageSource.MEASURED,
        valid_pin=True,
        corpus_hash="a" * 64,
    )
    embedding = EmbeddingProvenance(
        backend="openai:text-embedding-3-small", quality=EmbeddingQuality.SEMANTIC
    )
    artifact = build_comparison(
        run_id="integration-round-trip",
        corpus_hash="b" * 64,
        benchmark_hash="c" * 64,
        coverage=[],
        candidates=[candidate],
        decision=select_winner([candidate], embedding=embedding),
        evidence=[],
        provenance=provenance,
    )
    html = render_report(artifact)

    assert "subscription.claude" in html
    assert "protocol_reuse" in html
    assert "source.google-drive" in html
    assert "gated" in html
