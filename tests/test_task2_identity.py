from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from autobrain.corpus import canonical_corpus_identity, freeze_corpus
from autobrain.decision import select_winner
from autobrain.embedding import EmbeddingBackendConfig
from autobrain.models import (
    BenchmarkProvenance,
    CandidateEvaluation,
    CandidateId,
    CostStatus,
    CoverageCompleteness,
    ExperimentIdentity,
    NormalizedDocument,
    SourceKind,
    Status,
)
from autobrain.report import build_comparison
from autobrain.runs import RunInspectionError, compare_runs

_EMBEDDING = EmbeddingBackendConfig.from_environ(
    {"OPENAI_API_KEY": "fixture-embedding-key"}, requested="openai"
).descriptor


def _document(source_id: str, text: str) -> NormalizedDocument:
    return NormalizedDocument(
        source_id=source_id,
        source_kind=SourceKind.NOTION_PAGE,
        canonical_url=f"https://example.test/{source_id.split(':')[-1]}",
        title=source_id,
        text=text,
        content_hash="0" * 64,
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
    )


def test_corpus_identity_is_the_same_for_freeze_and_candidate_identity(tmp_path: Path) -> None:
    documents = [_document("notion:page:b", "beta"), _document("notion:page:a", "alpha")]
    expected = canonical_corpus_identity(documents)
    result = freeze_corpus(
        documents,
        tmp_path / "corpus",
        completeness=CoverageCompleteness.EXHAUSTIVE,
        coverage={"source": "notion", "discovered": 2},
    )

    manifest = json.loads((tmp_path / "corpus" / "manifest.json").read_text())
    assert result.identity == expected
    assert manifest["corpus_identity"] == expected.model_dump(mode="json")
    assert manifest["completeness"] == "EXHAUSTIVE"
    assert manifest["coverage"] == {"source": "notion", "discovered": 2}
    assert manifest["immutable"] is True
    assert manifest["provenance"] == {"normalization": "autobrain.corpus.canonical.v1"}


def test_experiment_identity_is_typed_and_shared_by_artifact() -> None:
    embedding = _EMBEDDING
    identity = ExperimentIdentity(
        corpus=canonical_corpus_identity([_document("notion:page:a", "alpha")]),
        benchmark_sha256="b" * 64,
        protocol="retrieval-recall-v1",
    )
    candidate = CandidateEvaluation(
        candidate=CandidateId.MEM0,
        status=Status.OK,
        scored_cases=20,
        answered_cases=20,
        quality_score=90,
        answer_success_rate=1,
        source_support_rate=1,
        contradiction_count=0,
        total_input_tokens=1,
        total_output_tokens=1,
        total_cost_usd=0.01,
        cost_status=CostStatus.COMPLETE,
        valid_pin=True,
        corpus_hash=identity.corpus.sha256,
    )
    artifact = build_comparison(
        run_id="run-identity",
        corpus_hash=identity.corpus.sha256,
        benchmark_hash=identity.benchmark_sha256,
        experiment_identity=identity,
        coverage=[],
        candidates=[candidate],
        decision=select_winner([candidate], embedding=embedding),
        evidence=[],
        provenance=BenchmarkProvenance(),
    )
    assert artifact.experiment_identity == identity


def test_legacy_runs_are_explicitly_non_comparable(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    for run_id in ("legacy-a", "legacy-b"):
        run_dir = root / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "hashes": {"corpus_sha256": "a" * 64, "benchmark_sha256": "b" * 64},
                }
            )
        )
        candidate = CandidateEvaluation(
            candidate=CandidateId.MEM0,
            status=Status.OK,
            scored_cases=20,
            answered_cases=20,
            quality_score=90,
            answer_success_rate=1,
            source_support_rate=1,
            contradiction_count=0,
            total_input_tokens=1,
            total_output_tokens=1,
            total_cost_usd=0.01,
            cost_status=CostStatus.COMPLETE,
            valid_pin=True,
            corpus_hash="a" * 64,
        )
        artifact = build_comparison(
            run_id=run_id,
            corpus_hash="a" * 64,
            benchmark_hash="b" * 64,
            coverage=[],
            candidates=[candidate],
            decision=select_winner([candidate], embedding=_EMBEDDING),
            evidence=[],
        )
        payload = artifact.model_dump(mode="json")
        payload["schema_version"] = 1
        (run_dir / "comparison.json").write_text(json.dumps(payload))

    with pytest.raises(RunInspectionError, match=r"legacy.*non-comparable"):
        compare_runs(root, "legacy-a", "legacy-b")
