from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autobrain.cli import app
from autobrain.decision import select_winner
from autobrain.embedding import EmbeddingBackendConfig
from autobrain.models import (
    BackendIdentity,
    BenchmarkProvenance,
    CandidateEvaluation,
    CandidateId,
    CapabilityClass,
    CorpusIdentity,
    CostStatus,
    EvidenceStatus,
    NativeCandidateResult,
    NativeMode,
    Status,
    UsageSource,
)
from autobrain.report import build_comparison, write_artifacts
from autobrain.runs import (
    RunVerificationStatus,
    _candidate_snapshot,  # pyright: ignore[reportPrivateUsage]
    verify_run,
)

_SEMANTIC_EMBEDDING = EmbeddingBackendConfig.from_environ(
    {"OPENAI_API_KEY": "fixture-embedding-key"},
    requested="openai",
).descriptor
_SEMANTIC_PROVENANCE = _SEMANTIC_EMBEDDING.provenance


def _select(candidates: list[CandidateEvaluation]):
    return select_winner(candidates, embedding=_SEMANTIC_EMBEDDING)


def _candidate(*, quality: float = 90.0, eligible: bool = True) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate=CandidateId.MEM0,
        status=Status.OK,
        scored_cases=20,
        answered_cases=20,
        quality_score=quality,
        answer_success_rate=1,
        source_support_rate=1,
        contradiction_count=0,
        total_input_tokens=10,
        total_output_tokens=5,
        total_cost_usd=0.01,
        cost_status=CostStatus.COMPLETE,
        query_p50_ms=10,
        query_p95_ms=20,
        workspace_bytes=100,
        operating_burden=1,
        valid_pin=True,
        corpus_hash="a" * 64,
        eligible_override=eligible,
        eligibility_reasons=[] if eligible else ["fixture ineligible"],
    )


def _write_run(
    home: Path,
    run_id: str,
    *,
    status: Status = Status.OK,
    corpus_hash: str = "a" * 64,
    benchmark_hash: str = "b" * 64,
    quality: float = 90.0,
    eligible: bool = True,
    comparison: bool = True,
    schema_version: int = 2,
) -> Path:
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    candidate = _candidate(quality=quality, eligible=eligible)
    decision = _select([candidate])
    manifest = {
        "schema_version": schema_version,
        "run_id": run_id,
        "created_at": "2026-08-20T12:00:00+00:00",
        "status": status.value,
        "verdict": decision.verdict.value,
        "hashes": {
            "corpus_sha256": corpus_hash,
            "benchmark_sha256": benchmark_hash,
        },
        "provenance": BenchmarkProvenance(embedding=_SEMANTIC_PROVENANCE).model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
        "evaluations": [candidate.model_dump(mode="json")],
        "candidates": [{"candidate": candidate.candidate.value}],
    }
    if schema_version == 1:
        manifest.pop("provenance")
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if comparison:
        artifact = build_comparison(
            run_id=run_id,
            status=status,
            corpus_hash=corpus_hash,
            benchmark_hash=benchmark_hash,
            coverage=[],
            candidates=[candidate],
            decision=decision,
            evidence=[],
            provenance=BenchmarkProvenance(embedding=_SEMANTIC_PROVENANCE),
        )
        write_artifacts(artifact, run_dir)
    return run_dir


def _env(home: Path) -> dict[str, str]:
    return {"AUTOBRAIN_HOME": str(home)}


def _json_result(arguments: list[str], home: Path):
    result = CliRunner().invoke(app, arguments, env=_env(home))
    return result, json.loads(result.stdout)


def test_verify_run_checks_recorded_artifact_hashes(tmp_path: Path) -> None:
    home = tmp_path / "state"
    run_dir = _write_run(home, "run-verified")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["comparison"] = {
        "path": "comparison.json",
        "sha256": hashlib.sha256((run_dir / "comparison.json").read_bytes()).hexdigest(),
    }
    manifest["report"] = {
        "path": "report.html",
        "sha256": hashlib.sha256((run_dir / "report.html").read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    verification = verify_run(home / "runs", "run-verified")

    assert verification.status is RunVerificationStatus.VALID
    assert {item.path for item in verification.artifacts} == {"comparison.json", "report.html"}


def test_verify_run_detects_tampered_recorded_artifact(tmp_path: Path) -> None:
    home = tmp_path / "state"
    run_dir = _write_run(home, "run-tampered")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["comparison"] = {
        "path": "comparison.json",
        "sha256": "0" * 64,
    }
    manifest["report"] = {
        "path": "report.html",
        "sha256": "0" * 64,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    verification = verify_run(home / "runs", "run-tampered")

    assert verification.status is RunVerificationStatus.CORRUPT
    assert any(not item.matches for item in verification.artifacts)


def test_verify_run_marks_legacy_artifacts_without_recorded_hashes_unverifiable(
    tmp_path: Path,
) -> None:
    home = tmp_path / "state"
    _write_run(home, "run-legacy", schema_version=1)

    verification = verify_run(home / "runs", "run-legacy")

    assert verification.status is RunVerificationStatus.UNVERIFIABLE


def test_runs_verify_cli_returns_typed_status_and_nonzero_for_corruption(tmp_path: Path) -> None:
    home = tmp_path / "state"
    run_dir = _write_run(home, "run-cli")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["comparison"] = {"path": "comparison.json", "sha256": "0" * 64}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["runs", "verify", "run-cli", "--run-root", str(home / "runs"), "--json"],
        env=_env(home),
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "CORRUPT"
    assert payload["run_id"] == "run-cli"


def test_runs_list_json_reports_complete_failed_incomplete_and_legacy_runs(
    tmp_path: Path,
) -> None:
    home = tmp_path / "state"
    complete = _write_run(home, "run-ok")
    failed = _write_run(home, "run-failed", status=Status.FAILED)
    incomplete = _write_run(home, "run-incomplete", comparison=False)
    legacy = _write_run(
        home,
        "run-legacy-failed",
        status=Status.MCP_AUTH_UNAVAILABLE,
        comparison=False,
        schema_version=1,
    )
    before = {path: path.read_bytes() for path in complete.rglob("*") if path.is_file()}
    before.update({path: path.read_bytes() for path in failed.rglob("*") if path.is_file()})
    before.update({path: path.read_bytes() for path in incomplete.rglob("*") if path.is_file()})
    before.update({path: path.read_bytes() for path in legacy.rglob("*") if path.is_file()})

    result, payload = _json_result(["runs", "list", "--json"], home)

    assert result.exit_code == 0
    assert payload["status"] == "OK"
    runs = {item["run_id"]: item for item in payload["runs"]}
    assert runs["run-ok"]["artifact_status"] == "COMPLETE"
    assert runs["run-ok"]["status"] == "OK"
    assert runs["run-failed"]["artifact_status"] == "COMPLETE"
    assert runs["run-failed"]["status"] == "FAILED"
    assert runs["run-incomplete"]["artifact_status"] == "INCOMPLETE"
    assert runs["run-legacy-failed"]["schema_version"] == 1
    assert runs["run-legacy-failed"]["status"] == "MCP_AUTH_UNAVAILABLE"
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("artifact", ["manifest.json", "comparison.json"])
def test_runs_list_does_not_silently_skip_corrupt_runs(tmp_path: Path, artifact: str) -> None:
    home = tmp_path / "state"
    run_dir = _write_run(home, "run-corrupt")
    (run_dir / artifact).write_text("{bad", encoding="utf-8")

    result, payload = _json_result(["runs", "list", "--json"], home)

    assert result.exit_code == 1
    assert payload["status"] == "CORRUPT_RUN"
    corrupt = next(item for item in payload["runs"] if item["run_id"] == "run-corrupt")
    assert corrupt["artifact_status"] == "CORRUPT_RUN"
    assert artifact in corrupt["detail"]


_CANDIDATE_FIELD_MUTATIONS: dict[str, object] = {
    "candidate": CandidateId.GBRAIN,
    "status": Status.FAILED,
    "scored_cases": 21,
    "answered_cases": 19,
    "quality_score": 89.0,
    "answer_success_rate": 0.95,
    "source_support_rate": 0.9,
    "contradiction_count": 1,
    "total_input_tokens": 11,
    "total_output_tokens": 6,
    "total_cost_usd": 0.02,
    "cost_status": CostStatus.INCOMPLETE,
    "usage_source": UsageSource.ESTIMATED,
    "ingest_wall_time_ms": 1,
    "query_wall_time_ms": 1,
    "query_p50_ms": 11,
    "query_p95_ms": 21,
    "workspace_bytes": 101,
    "operating_burden": 2,
    "valid_pin": False,
    "corpus_hash": "c" * 64,
    "direct_leakage": True,
    "generated_cases": 1,
    "partial_failures": 1,
    "eligibility_reasons": ["changed"],
    "eligible_override": False,
    "native_result": NativeCandidateResult(
        candidate=CandidateId.GBRAIN,
        mode=NativeMode.SEMANTIC,
        backend=BackendIdentity(name="gbrain", version="1.0"),
        capability=CapabilityClass.RETRIEVAL_AND_ANSWER,
        evidence_status=EvidenceStatus.COMPLETE,
        corpus=CorpusIdentity(sha256="a" * 64, document_count=20),
        recommendation_eligible=True,
    ),
}


@pytest.mark.parametrize("field", sorted(_CANDIDATE_FIELD_MUTATIONS))
def test_candidate_snapshot_compares_every_persisted_evaluation_field(field: str) -> None:
    candidate = _candidate()
    artifact = build_comparison(
        run_id="run-a",
        corpus_hash="a" * 64,
        benchmark_hash="b" * 64,
        coverage=[],
        candidates=[candidate],
        decision=_select([candidate]),
        evidence=[],
    )
    update = {field: _CANDIDATE_FIELD_MUTATIONS[field]}
    changed = candidate.model_copy(update=update)
    changed_artifact = artifact.model_copy(update={"candidates": [changed]})

    assert set(CandidateEvaluation.model_fields) == set(_CANDIDATE_FIELD_MUTATIONS)
    assert _candidate_snapshot(artifact) != _candidate_snapshot(changed_artifact)


def test_candidate_snapshot_compares_derived_eligibility() -> None:
    candidate = _candidate()
    artifact = build_comparison(
        run_id="run-a",
        corpus_hash="a" * 64,
        benchmark_hash="b" * 64,
        coverage=[],
        candidates=[candidate],
        decision=_select([candidate]),
        evidence=[],
    )
    changed_decision = artifact.decision.model_copy(update={"eligible_candidates": []})
    changed = artifact.model_copy(update={"decision": changed_decision})

    assert _candidate_snapshot(artifact) != _candidate_snapshot(changed)


_CROSS_ARTIFACT_MUTATIONS: tuple[tuple[str, tuple[object, ...], object, str], ...] = (
    ("manifest.json", ("run_id",), "other-run", "run_id"),
    ("comparison.json", ("run_id",), "other-run", "run_id"),
    ("manifest.json", ("status",), Status.FAILED.value, "status"),
    ("comparison.json", ("status",), Status.FAILED.value, "status"),
    ("manifest.json", ("hashes", "corpus_sha256"), "c" * 64, "corpus"),
    ("comparison.json", ("corpus_hash",), "c" * 64, "corpus"),
    ("manifest.json", ("hashes", "benchmark_sha256"), "d" * 64, "benchmark"),
    ("comparison.json", ("benchmark_hash",), "d" * 64, "benchmark"),
    ("manifest.json", ("provenance", "chat", "provider"), "manifest-only", "provenance"),
    ("comparison.json", ("provenance", "chat", "provider"), "comparison-only", "provenance"),
    ("manifest.json", ("verdict",), CandidateId.GBRAIN.value, "verdict"),
    ("comparison.json", ("verdict",), CandidateId.GBRAIN.value, "verdict"),
    ("manifest.json", ("decision", "status"), Status.FAILED.value, "decision"),
    ("comparison.json", ("decision", "status"), Status.FAILED.value, "decision"),
    ("manifest.json", ("decision", "verdict"), CandidateId.GBRAIN.value, "decision"),
    ("comparison.json", ("decision", "verdict"), CandidateId.GBRAIN.value, "decision"),
    ("manifest.json", ("evaluations", 0, "candidate"), CandidateId.GBRAIN.value, "candidate"),
    ("comparison.json", ("candidates", 0, "candidate"), CandidateId.GBRAIN.value, "candidate"),
    ("manifest.json", ("candidates", 0, "candidate"), CandidateId.GBRAIN.value, "candidate"),
)


def _mutate_json(path: Path, keys: tuple[object, ...], value: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = payload
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    ("artifact", "keys", "value", "detail"),
    _CROSS_ARTIFACT_MUTATIONS,
    ids=lambda value: str(value),
)
def test_inventory_rejects_cross_artifact_integrity_mismatch(
    tmp_path: Path,
    artifact: str,
    keys: tuple[object, ...],
    value: object,
    detail: str,
) -> None:
    home = tmp_path / "state"
    run_dir = _write_run(home, "run-dirty")
    _mutate_json(run_dir / artifact, keys, value)

    result, payload = _json_result(["runs", "list", "--json"], home)

    assert result.exit_code == 1
    assert payload["status"] == "CORRUPT_RUN"
    assert detail in payload["detail"]


def test_equivalence_rejects_dirty_manifest(tmp_path: Path) -> None:
    home = tmp_path / "state"
    _write_run(home, "run-a")
    dirty = _write_run(home, "run-b")
    _mutate_json(dirty / "manifest.json", ("provenance", "chat", "provider"), "dirty")

    result, payload = _json_result(["runs", "compare", "run-a", "run-b", "--json"], home)

    assert result.exit_code == 1
    assert payload["status"] == "CORRUPT_RUN"
    assert payload["run_id"] == "run-b"


def test_runs_explain_json_uses_persisted_eligibility_reasons_and_redacts(
    tmp_path: Path,
) -> None:
    home = tmp_path / "state"
    run_dir = _write_run(home, "run-explain", eligible=False)
    persisted_reason = "operator token=fixture-secret-123456 marked this candidate ineligible"
    for artifact_name, candidate_key in (
        ("manifest.json", "evaluations"),
        ("comparison.json", "candidates"),
    ):
        artifact = json.loads((run_dir / artifact_name).read_text(encoding="utf-8"))
        artifact[candidate_key][0]["eligibility_reasons"] = [persisted_reason]
        (run_dir / artifact_name).write_text(json.dumps(artifact), encoding="utf-8")

    result, payload = _json_result(["runs", "explain", "run-explain", "--json"], home)

    assert result.exit_code == 0
    assert payload == {
        "status": "OK",
        "run_id": "run-explain",
        "run_status": "OK",
        "verdict": "NO_RECOMMENDATION",
        "decision_status": "NO_RECOMMENDATION",
        "rationale": payload["rationale"],
        "candidates": [
            {
                "candidate": "mem0",
                "eligible": False,
                "eligibility_reasons": ["operator [REDACTED] marked this candidate ineligible"],
            }
        ],
    }
    assert "fixture-secret-123456" not in result.stdout
    assert "explicitly marked ineligible" not in result.stdout


def test_runs_explain_human_output_and_manifest_only_run(tmp_path: Path) -> None:
    home = tmp_path / "state"
    _write_run(home, "run-explain", eligible=False, comparison=False)

    result = CliRunner().invoke(
        app,
        ["runs", "explain", "run-explain"],
        env=_env(home),
    )

    assert result.exit_code == 0
    assert "run-id: run-explain" in result.stdout
    assert "verdict: NO_RECOMMENDATION" in result.stdout
    assert "mem0: ineligible" in result.stdout
    assert "fixture ineligible" in result.stdout


def test_runs_explain_rejects_run_without_persisted_decision_or_evaluations(
    tmp_path: Path,
) -> None:
    home = tmp_path / "state"
    run_dir = _write_run(home, "run-incomplete", comparison=False)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["decision"] = None
    manifest["evaluations"] = None
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result, payload = _json_result(["runs", "explain", "run-incomplete", "--json"], home)

    assert result.exit_code == 1
    assert payload["status"] == "INCOMPLETE_RUN"
    assert payload["run_id"] == "run-incomplete"


def test_runs_explain_rejects_traversal_and_symlink_escape(tmp_path: Path) -> None:
    home = tmp_path / "state"
    outside = tmp_path / "outside"
    _write_run(outside, "escaped")
    (home / "runs").mkdir(parents=True)
    (home / "runs" / "escaped").symlink_to(outside / "runs" / "escaped", target_is_directory=True)

    malformed, malformed_payload = _json_result(["runs", "explain", "../escape", "--json"], home)
    escaped, escaped_payload = _json_result(["runs", "explain", "escaped", "--json"], home)

    assert malformed.exit_code == 1
    assert malformed_payload["status"] == "INVALID_RUN_ID"
    assert escaped.exit_code == 1
    assert escaped_payload["status"] == "PATH_ESCAPE"


def test_runs_compare_equivalent_and_metric_or_eligibility_differences(tmp_path: Path) -> None:
    home = tmp_path / "state"
    _write_run(home, "run-a")
    _write_run(home, "run-b")
    _write_run(home, "run-c", quality=81, eligible=False)

    same, equivalent = _json_result(["runs", "compare", "run-a", "run-b", "--json"], home)
    changed, different = _json_result(["runs", "compare", "run-a", "run-c", "--json"], home)

    assert same.exit_code == 0
    assert equivalent["status"] == "EQUIVALENT"
    assert equivalent["equivalent"] is True
    assert equivalent["comparable"] is True
    assert equivalent["differences"] == []
    assert changed.exit_code == 0
    assert different["status"] == "NON_EQUIVALENT"
    assert different["equivalent"] is False
    paths = {item["path"] for item in different["differences"]}
    assert "candidates.mem0.quality_score" in paths
    assert "candidates.mem0.eligible" in paths


def test_runs_compare_refuses_different_hashes_unless_explicitly_allowed(
    tmp_path: Path,
) -> None:
    home = tmp_path / "state"
    _write_run(home, "run-a")
    _write_run(home, "run-other-corpus", corpus_hash="c" * 64)
    _write_run(home, "run-other-benchmark", benchmark_hash="d" * 64)

    for other in ("run-other-corpus", "run-other-benchmark"):
        refused, refusal = _json_result(["runs", "compare", "run-a", other, "--json"], home)
        allowed, comparison = _json_result(
            [
                "runs",
                "compare",
                "run-a",
                other,
                "--allow-different-corpus",
                "--json",
            ],
            home,
        )
        assert refused.exit_code == 1
        assert refusal["status"] == "DIFFERENT_CORPUS"
        assert refusal["equivalent"] is False
        assert refused.stderr == ""
        assert allowed.exit_code == 0
        assert comparison["status"] == "NON_EQUIVALENT"
        assert comparison["equivalent"] is False
        assert comparison["comparable"] is False
        assert comparison["comparison_basis"] == "DIFFERENT_CORPUS_ALLOWED"


def test_runs_compare_rejects_incomplete_and_mismatched_artifact_ids(tmp_path: Path) -> None:
    home = tmp_path / "state"
    _write_run(home, "run-a")
    _write_run(home, "run-incomplete", comparison=False)
    mismatched = _write_run(home, "run-mismatch")
    payload = json.loads((mismatched / "comparison.json").read_text(encoding="utf-8"))
    payload["run_id"] = "some-other-run"
    (mismatched / "comparison.json").write_text(json.dumps(payload), encoding="utf-8")

    incomplete, incomplete_payload = _json_result(
        ["runs", "compare", "run-a", "run-incomplete", "--json"], home
    )
    corrupt, corrupt_payload = _json_result(
        ["runs", "compare", "run-a", "run-mismatch", "--json"], home
    )

    assert incomplete.exit_code == 1
    assert incomplete_payload["status"] == "INCOMPLETE_RUN"
    assert corrupt.exit_code == 1
    assert corrupt_payload["status"] == "CORRUPT_RUN"
    assert "run_id" in corrupt_payload["detail"]


@pytest.mark.parametrize("run_id", ["../escape", "a/b", ".", "..", "/tmp/escape"])
def test_runs_compare_rejects_traversal_and_malformed_run_ids(tmp_path: Path, run_id: str) -> None:
    home = tmp_path / "state"
    _write_run(home, "run-a")

    result, payload = _json_result(["runs", "compare", "run-a", run_id, "--json"], home)

    assert result.exit_code == 1
    assert payload["status"] == "INVALID_RUN_ID"


def test_runs_commands_reject_symlink_escape(tmp_path: Path) -> None:
    home = tmp_path / "state"
    outside = tmp_path / "outside"
    _write_run(outside, "escaped")
    (home / "runs").mkdir(parents=True)
    (home / "runs" / "escaped").symlink_to(outside / "runs" / "escaped", target_is_directory=True)

    listed, list_payload = _json_result(["runs", "list", "--json"], home)
    compared, compare_payload = _json_result(
        ["runs", "compare", "escaped", "escaped", "--json"], home
    )

    assert listed.exit_code == 1
    assert list_payload["status"] == "PATH_ESCAPE"
    assert compared.exit_code == 1
    assert compare_payload["status"] == "PATH_ESCAPE"


def test_runs_compare_rereads_artifacts_instead_of_using_stale_cache(tmp_path: Path) -> None:
    home = tmp_path / "state"
    _write_run(home, "run-a")
    changed_dir = _write_run(home, "run-b")

    first, first_payload = _json_result(["runs", "compare", "run-a", "run-b", "--json"], home)
    artifact = json.loads((changed_dir / "comparison.json").read_text(encoding="utf-8"))
    artifact["candidates"][0]["quality_score"] = 70
    (changed_dir / "comparison.json").write_text(json.dumps(artifact), encoding="utf-8")
    second, second_payload = _json_result(["runs", "compare", "run-a", "run-b", "--json"], home)

    assert first.exit_code == 0
    assert first_payload["status"] == "EQUIVALENT"
    assert second.exit_code == 1
    assert second_payload["status"] == "CORRUPT_RUN"
    assert "candidate evaluations" in second_payload["detail"]


def test_autobrain_home_is_a_confined_explicit_state_root(tmp_path: Path) -> None:
    state = tmp_path / "isolated-autobrain"
    _write_run(state, "run-a")

    result, payload = _json_result(["runs", "list", "--json"], state)

    assert result.exit_code == 0
    assert [item["run_id"] for item in payload["runs"]] == ["run-a"]
    assert not (tmp_path / ".autobrain").exists()


def test_configured_autobrain_home_rejects_symlinked_state_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside-state"
    outside.mkdir()
    state = tmp_path / "state-link"
    state.symlink_to(outside, target_is_directory=True)

    result = CliRunner().invoke(
        app,
        ["runs", "list", "--json"],
        env={"AUTOBRAIN_HOME": str(state)},
    )
    payload = json.loads(result.stdout)

    assert result.exit_code == 1
    assert payload["status"] == "PATH_ESCAPE"
    assert "state root cannot be a symlink" in payload["detail"]


def test_runs_list_rejects_symlinked_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _write_run(outside, "run-a")
    state = tmp_path / "state"
    state.mkdir()
    (state / "runs").symlink_to(outside / "runs", target_is_directory=True)

    result, payload = _json_result(["runs", "list", "--json"], state)

    assert result.exit_code == 1
    assert payload["status"] == "PATH_ESCAPE"


def test_runs_list_retains_healthy_runs_when_one_run_is_corrupt(tmp_path: Path) -> None:
    home = tmp_path / "state"
    _write_run(home, "run-healthy")
    corrupt = _write_run(home, "run-corrupt")
    (corrupt / "manifest.json").write_text("{bad", encoding="utf-8")

    result, payload = _json_result(
        ["runs", "list", "--run-root", str(home / "runs"), "--json"], home
    )

    assert result.exit_code == 1
    assert payload["status"] == "CORRUPT_RUN"
    assert {item["run_id"] for item in payload["runs"]} == {"run-healthy", "run-corrupt"}
    healthy = next(item for item in payload["runs"] if item["run_id"] == "run-healthy")
    assert healthy["artifact_status"] == "COMPLETE"


def test_custom_run_root_is_shared_by_list_compare_explain_and_report(tmp_path: Path) -> None:
    home = tmp_path / "state"
    _write_run(home, "run-custom")
    custom_root = tmp_path / "custom-runs"
    custom_root.mkdir()
    (home / "runs" / "run-custom").rename(custom_root / "run-custom")

    listed, listed_payload = _json_result(
        ["runs", "list", "--run-root", str(custom_root), "--json"], home
    )
    compared, compared_payload = _json_result(
        [
            "runs",
            "compare",
            "run-custom",
            "run-custom",
            "--run-root",
            str(custom_root),
            "--json",
        ],
        home,
    )
    explained, explained_payload = _json_result(
        [
            "runs",
            "explain",
            "run-custom",
            "--run-root",
            str(custom_root),
            "--json",
        ],
        home,
    )
    reopened = CliRunner().invoke(
        app,
        ["report", "run-custom", "--run-root", str(custom_root), "--no-open"],
        env=_env(home),
    )

    assert listed.exit_code == 0
    assert listed_payload["runs"][0]["run_id"] == "run-custom"
    assert compared.exit_code == 0
    assert compared_payload["status"] == "EQUIVALENT"
    assert explained.exit_code == 0
    assert explained_payload["run_id"] == "run-custom"
    assert reopened.exit_code == 0
    assert str(custom_root / "run-custom" / "report.html") in reopened.stdout


def test_runs_list_preserves_unrelated_dirty_files(tmp_path: Path) -> None:
    home = tmp_path / "state"
    _write_run(home, "run-a")
    dirty = tmp_path / "dirty.txt"
    dirty.write_text("do not touch", encoding="utf-8")
    os.chmod(home / "runs" / "run-a" / "manifest.json", 0o400)

    result, _ = _json_result(["runs", "list", "--json"], home)

    assert result.exit_code == 0
    assert dirty.read_text(encoding="utf-8") == "do not touch"
