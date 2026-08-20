from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autobrain.cli import app
from autobrain.decision import select_winner
from autobrain.models import (
    BenchmarkProvenance,
    CandidateEvaluation,
    CandidateId,
    CostStatus,
    Status,
    UsageSource,
)
from autobrain.report import build_comparison, write_artifacts
from autobrain.runs import _candidate_snapshot  # pyright: ignore[reportPrivateUsage]


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
    decision = select_winner([candidate])
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
        "provenance": BenchmarkProvenance().model_dump(mode="json"),
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
        )
        write_artifacts(artifact, run_dir)
    return run_dir


def _env(home: Path) -> dict[str, str]:
    return {"AUTOBRAIN_HOME": str(home)}


def _json_result(arguments: list[str], home: Path):
    result = CliRunner().invoke(app, arguments, env=_env(home))
    return result, json.loads(result.stdout)


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
    assert payload["run_id"] == "run-corrupt"
    assert artifact in payload["detail"]


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
        decision=select_winner([candidate]),
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
        decision=select_winner([candidate]),
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


def test_runs_list_preserves_unrelated_dirty_files(tmp_path: Path) -> None:
    home = tmp_path / "state"
    _write_run(home, "run-a")
    dirty = tmp_path / "dirty.txt"
    dirty.write_text("do not touch", encoding="utf-8")
    os.chmod(home / "runs" / "run-a" / "manifest.json", 0o400)

    result, _ = _json_result(["runs", "list", "--json"], home)

    assert result.exit_code == 0
    assert dirty.read_text(encoding="utf-8") == "do not touch"
