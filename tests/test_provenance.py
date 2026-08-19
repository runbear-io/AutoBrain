from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from autobrain.auth.models import Provider
from autobrain.decision import select_winner
from autobrain.models import (
    BenchmarkProvenance,
    CandidateEvaluation,
    CandidateId,
    ChatProvenance,
    CostStatus,
    EmbeddingProvenance,
    EmbeddingQuality,
    LatencySpan,
    LatencySpanKind,
    SourceMutability,
    SourceProvenance,
    Status,
    UsageSource,
)
from autobrain.orchestration import RunConfig, RunOrchestrator
from autobrain.report import build_comparison, load_comparison, load_manifest, render_report


def _candidate() -> CandidateEvaluation:
    return CandidateEvaluation(
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
        valid_pin=True,
        corpus_hash="a" * 64,
    )


def _provenance() -> BenchmarkProvenance:
    return BenchmarkProvenance(
        chat=ChatProvenance(provider="openai", model="gpt-5-mini"),
        embedding=EmbeddingProvenance(
            backend="openai:text-embedding-3-small",
            quality=EmbeddingQuality.SEMANTIC,
        ),
        usage_source=UsageSource.MEASURED,
        sources=[
            SourceProvenance(source="slack", mutability=SourceMutability.FROZEN_EXPORT),
            SourceProvenance(source="notion", mutability=SourceMutability.LIVE_MCP_CAPTURED),
        ],
        latency_spans=[
            LatencySpan(
                name=LatencySpanKind.CANDIDATE_QUERY,
                duration_ms=12.5,
                candidate=CandidateId.MEM0,
            ),
            LatencySpan(name=LatencySpanKind.PROVIDER_EXECUTION, duration_ms=None),
        ],
    )


def test_provenance_round_trips_with_json_html_parity_and_null_preservation(
    tmp_path: Path,
) -> None:
    candidate = _candidate()
    artifact = build_comparison(
        run_id="run-provenance",
        corpus_hash="b" * 64,
        benchmark_hash="c" * 64,
        coverage=[],
        candidates=[candidate],
        decision=select_winner([candidate]),
        evidence=[],
        provenance=_provenance(),
    )
    path = tmp_path / "comparison.json"
    path.write_text(artifact.model_dump_json(), encoding="utf-8")

    loaded = load_comparison(path)
    payload = loaded.model_dump(mode="json")
    html = render_report(loaded)

    assert payload["schema_version"] == 2
    assert payload["provenance"]["chat"] == {
        "provider": "openai",
        "model": "gpt-5-mini",
    }
    assert payload["provenance"]["embedding"] == {
        "backend": "openai:text-embedding-3-small",
        "quality": "semantic",
    }
    assert payload["provenance"]["usage_source"] == "measured"
    assert payload["provenance"]["sources"] == [
        {"source": "slack", "mutability": "frozen_export"},
        {"source": "notion", "mutability": "live_mcp_captured"},
    ]
    assert payload["provenance"]["latency_spans"][1]["duration_ms"] is None
    for value in (
        "openai",
        "gpt-5-mini",
        "openai:text-embedding-3-small",
        "semantic",
        "measured",
        "slack",
        "frozen_export",
        "notion",
        "live_mcp_captured",
        "candidate_query",
        "12.5 ms",
        "provider_execution",
        "unavailable",
    ):
        assert value in html


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_quality", "approximate"),
        ("usage_source", "reconciled-ish"),
        ("source_mutability", "mutable"),
        ("latency_span", "wall_clock_guess"),
    ],
)
def test_provenance_rejects_invalid_enums(field: str, value: str) -> None:
    payload = _provenance().model_dump(mode="json")
    if field == "embedding_quality":
        payload["embedding"]["quality"] = value
    elif field == "usage_source":
        payload["usage_source"] = value
    elif field == "source_mutability":
        payload["sources"][0]["mutability"] = value
    else:
        payload["latency_spans"][0]["name"] = value
    with pytest.raises(ValidationError):
        BenchmarkProvenance.model_validate(payload)


def test_schema_v1_comparison_loads_with_unavailable_defaults(tmp_path: Path) -> None:
    candidate = _candidate()
    artifact = build_comparison(
        run_id="old-run",
        corpus_hash="b" * 64,
        benchmark_hash="c" * 64,
        coverage=[],
        candidates=[candidate],
        decision=select_winner([candidate]),
        evidence=[],
        provenance=_provenance(),
    )
    payload = artifact.model_dump(mode="json")
    payload["schema_version"] = 1
    del payload["provenance"]
    path = tmp_path / "comparison-v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_comparison(path)

    assert loaded.schema_version == 2
    assert loaded.provenance.chat.provider is None
    assert loaded.provenance.chat.model is None
    assert loaded.provenance.embedding.backend is None
    assert loaded.provenance.embedding.quality is None
    assert loaded.provenance.usage_source is UsageSource.UNAVAILABLE
    assert loaded.provenance.sources == []
    assert loaded.provenance.latency_spans == []


def _write_comparison_payload(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_comparison_missing_schema_version_is_rejected(tmp_path: Path) -> None:
    candidate = _candidate()
    payload = build_comparison(
        run_id="missing-schema",
        corpus_hash="b" * 64,
        benchmark_hash="c" * 64,
        coverage=[],
        candidates=[candidate],
        decision=select_winner([candidate]),
        evidence=[],
        provenance=_provenance(),
    ).model_dump(mode="json")
    del payload["schema_version"]

    with pytest.raises(ValueError, match="corrupt comparison artifact"):
        load_comparison(_write_comparison_payload(tmp_path, payload))


@pytest.mark.parametrize("schema_version", [True, False, 1.0, 2.0, "1", "2"])
def test_comparison_rejects_non_integer_schema_versions(
    tmp_path: Path,
    schema_version: object,
) -> None:
    candidate = _candidate()
    payload = build_comparison(
        run_id="invalid-schema-type",
        corpus_hash="b" * 64,
        benchmark_hash="c" * 64,
        coverage=[],
        candidates=[candidate],
        decision=select_winner([candidate]),
        evidence=[],
        provenance=_provenance(),
    ).model_dump(mode="json")
    payload["schema_version"] = schema_version

    with pytest.raises(ValueError, match="corrupt comparison artifact"):
        load_comparison(_write_comparison_payload(tmp_path, payload))


def test_schema_v2_comparison_missing_provenance_is_rejected(tmp_path: Path) -> None:
    candidate = _candidate()
    payload = build_comparison(
        run_id="missing-provenance",
        corpus_hash="b" * 64,
        benchmark_hash="c" * 64,
        coverage=[],
        candidates=[candidate],
        decision=select_winner([candidate]),
        evidence=[],
        provenance=_provenance(),
    ).model_dump(mode="json")
    del payload["provenance"]

    with pytest.raises(ValueError, match="corrupt comparison artifact"):
        load_comparison(_write_comparison_payload(tmp_path, payload))


def test_future_manifest_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 999,
                "run_id": "future-run",
                "provenance": _provenance().model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="corrupt run manifest"):
        load_manifest(path)


@pytest.mark.parametrize("schema_version", [True, False, 1.0, 2.0, "1", "2"])
def test_manifest_rejects_non_integer_schema_versions(
    tmp_path: Path,
    schema_version: object,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "run_id": "invalid-schema-type",
                "provenance": _provenance().model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="corrupt run manifest"):
        load_manifest(path)


def test_schema_v2_manifest_missing_provenance_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"schema_version": 2, "run_id": "missing-provenance"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="corrupt run manifest"):
        load_manifest(path)


def test_orchestration_records_only_known_runtime_provenance(tmp_path: Path) -> None:
    subscription = RunOrchestrator(
        config=RunConfig(
            output=tmp_path,
            provider_mode="codex-subscription",
            selected_sources=(Provider.SLACK, Provider.NOTION),
            slack_export_path=tmp_path / "slack.zip",
            slack_export_sha256="a" * 64,
        ),
        connectors=(),
        candidates=(),
        provider_available=False,
    )
    before_usage = subscription.benchmark_provenance()
    after_usage = subscription.benchmark_provenance([_candidate()])

    assert before_usage.chat.provider == "codex"
    assert before_usage.chat.model is None
    assert before_usage.embedding.backend == "local-hash-embedding"
    assert before_usage.embedding.quality is EmbeddingQuality.SMOKE_ONLY
    assert before_usage.usage_source is UsageSource.UNAVAILABLE
    assert before_usage.sources == [
        SourceProvenance(source="slack", mutability=SourceMutability.FROZEN_EXPORT),
        SourceProvenance(source="notion", mutability=SourceMutability.LIVE_MCP_CAPTURED),
    ]
    assert after_usage.usage_source is UsageSource.ESTIMATED
    assert after_usage.latency_spans == [
        LatencySpan(
            name=LatencySpanKind.CANDIDATE_QUERY,
            duration_ms=0,
            candidate=CandidateId.MEM0,
        )
    ]

    api = RunOrchestrator(
        config=RunConfig(output=tmp_path),
        connectors=(),
        candidates=(),
        provider_available=False,
    ).benchmark_provenance([_candidate()])
    assert api.chat == ChatProvenance(provider="openai", model="gpt-5-mini")
    assert api.embedding == EmbeddingProvenance(
        backend="openai:text-embedding-3-small",
        quality=EmbeddingQuality.SEMANTIC,
    )
    assert api.usage_source is UsageSource.MEASURED


def test_existing_schema_v1_manifest_reopens_read_only() -> None:
    path = Path("/Users/kimbwook/.autobrain/runs/20260819T011402Z-d1c62299/manifest.json")
    before = path.read_bytes()

    loaded = load_manifest(path)

    assert loaded.schema_version == 1
    assert loaded.run_id == "20260819T011402Z-d1c62299"
    assert loaded.provenance.usage_source is UsageSource.UNAVAILABLE
    assert loaded.provenance.chat.provider is None
    assert path.read_bytes() == before
