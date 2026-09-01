"""Read-only inventory and comparison of immutable evaluation runs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict

from autobrain.models import ComparisonArtifact, RunManifest
from autobrain.paths import is_valid_run_id, resolve_run_root
from autobrain.report import load_comparison, load_manifest, redact_payload


class RunInspectionStatus(StrEnum):
    OK = "OK"
    CORRUPT_RUN = "CORRUPT_RUN"
    INCOMPLETE_RUN = "INCOMPLETE_RUN"
    INVALID_RUN_ID = "INVALID_RUN_ID"
    RUN_NOT_FOUND = "RUN_NOT_FOUND"
    PATH_ESCAPE = "PATH_ESCAPE"
    DIFFERENT_CORPUS = "DIFFERENT_CORPUS"
    EQUIVALENT = "EQUIVALENT"
    NON_EQUIVALENT = "NON_EQUIVALENT"


class RunVerificationStatus(StrEnum):
    VALID = "VALID"
    CORRUPT = "CORRUPT"
    INCOMPLETE = "INCOMPLETE"
    UNVERIFIABLE = "UNVERIFIABLE"


class VerifiedArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    path: str
    recorded_sha256: str | None = None
    actual_sha256: str | None = None
    matches: bool | None = None


class RunVerification(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    status: RunVerificationStatus
    run_id: str
    artifacts: list[VerifiedArtifact]
    detail: str | None = None


class RunInspectionError(ValueError):
    """Typed user-facing failure while reading immutable run artifacts."""

    def __init__(self, status: RunInspectionStatus, detail: str, *, run_id: str | None = None):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.run_id = run_id

    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"status": self.status.value, "detail": self.detail}
        if self.run_id is not None:
            payload["run_id"] = self.run_id
        if self.status is RunInspectionStatus.DIFFERENT_CORPUS:
            payload.update({"equivalent": False, "comparable": False})
        return payload


class RunRecord(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    run_id: str
    schema_version: int
    created_at: str | None
    status: str
    artifact_status: str
    corpus_hash: str | None
    benchmark_hash: str | None
    verdict: str | None
    provenance: dict[str, object]
    detail: str | None = None


class RunInventory(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    status: str = RunInspectionStatus.OK.value
    runs: list[RunRecord]
    run_id: str | None = None
    detail: str | None = None


class RunDifference(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    path: str
    left: object
    right: object


class RunComparison(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    status: str
    equivalent: bool
    comparable: bool
    comparison_basis: str
    left_run_id: str
    right_run_id: str
    left: dict[str, object]
    right: dict[str, object]
    differences: list[RunDifference]


class CandidateEligibilityExplanation(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    candidate: str
    eligible: bool
    eligibility_reasons: list[str]


class RunExplanation(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    status: str
    run_id: str
    run_status: str
    verdict: str
    decision_status: str
    rationale: str
    candidates: list[CandidateEligibilityExplanation]


@dataclass(frozen=True)
class _LoadedRun:
    run_dir: Path
    manifest: RunManifest
    comparison: ComparisonArtifact | None


def _confined_runs_root(root: Path) -> Path:
    try:
        resolved = resolve_run_root(root)
    except ValueError as error:
        raise RunInspectionError(RunInspectionStatus.PATH_ESCAPE, str(error)) from error
    return resolved.resolve() if resolved.exists() else resolved


def _confined_run_dir(root: Path, run_id: str) -> Path:
    if not is_valid_run_id(run_id):
        raise RunInspectionError(
            RunInspectionStatus.INVALID_RUN_ID,
            f"invalid run id: {run_id!r}",
            run_id=run_id,
        )
    canonical_root = _confined_runs_root(root)
    run_dir = root / run_id
    if not run_dir.exists():
        raise RunInspectionError(
            RunInspectionStatus.RUN_NOT_FOUND,
            f"run not found: {run_id}",
            run_id=run_id,
        )
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RunInspectionError(
            RunInspectionStatus.PATH_ESCAPE,
            f"run path is not a confined directory: {run_dir}",
            run_id=run_id,
        )
    if not run_dir.resolve().is_relative_to(canonical_root):
        raise RunInspectionError(
            RunInspectionStatus.PATH_ESCAPE,
            f"run path escapes runs root: {run_dir}",
            run_id=run_id,
        )
    return run_dir


def _artifact_path(run_dir: Path, name: str) -> Path:
    path = run_dir / name
    if path.is_symlink() or not path.resolve(strict=False).is_relative_to(run_dir.resolve()):
        raise RunInspectionError(
            RunInspectionStatus.PATH_ESCAPE,
            f"run artifact escapes run directory: {path}",
            run_id=run_dir.name,
        )
    return path


def _load_run(root: Path, run_id: str) -> _LoadedRun:
    run_dir = _confined_run_dir(root, run_id)
    manifest_path = _artifact_path(run_dir, "manifest.json")
    if not manifest_path.is_file():
        raise RunInspectionError(
            RunInspectionStatus.CORRUPT_RUN,
            f"missing run manifest: {manifest_path}",
            run_id=run_id,
        )
    try:
        manifest = load_manifest(manifest_path)
    except ValueError as error:
        raise RunInspectionError(
            RunInspectionStatus.CORRUPT_RUN,
            str(error),
            run_id=run_id,
        ) from error
    if manifest.run_id != run_id:
        raise RunInspectionError(
            RunInspectionStatus.CORRUPT_RUN,
            f"manifest run_id {manifest.run_id!r} does not match directory {run_id!r}",
            run_id=run_id,
        )

    comparison_path = _artifact_path(run_dir, "comparison.json")
    comparison: ComparisonArtifact | None = None
    if comparison_path.exists():
        if not comparison_path.is_file():
            raise RunInspectionError(
                RunInspectionStatus.CORRUPT_RUN,
                f"comparison artifact is not a file: {comparison_path}",
                run_id=run_id,
            )
        try:
            comparison = load_comparison(comparison_path)
        except ValueError as error:
            raise RunInspectionError(
                RunInspectionStatus.CORRUPT_RUN,
                str(error),
                run_id=run_id,
            ) from error
        if comparison.run_id != run_id:
            raise RunInspectionError(
                RunInspectionStatus.CORRUPT_RUN,
                f"comparison run_id {comparison.run_id!r} does not match directory {run_id!r}",
                run_id=run_id,
            )
        _validate_manifest_comparison(manifest, comparison, run_id)
    return _LoadedRun(run_dir=run_dir, manifest=manifest, comparison=comparison)


def _integrity_mismatch(run_id: str, field: str) -> RunInspectionError:
    return RunInspectionError(
        RunInspectionStatus.CORRUPT_RUN,
        f"manifest and comparison {field} disagree",
        run_id=run_id,
    )


def _manifest_candidate_ids(manifest: RunManifest, run_id: str) -> list[str] | None:
    if manifest.candidates is None:
        return None
    candidate_ids: list[str] = []
    for candidate in manifest.candidates:
        candidate_id = candidate.get("candidate")
        if not isinstance(candidate_id, str):
            raise _integrity_mismatch(run_id, "candidate identities")
        candidate_ids.append(candidate_id)
    return candidate_ids


def _validate_manifest_comparison(
    manifest: RunManifest, comparison: ComparisonArtifact, run_id: str
) -> None:
    if manifest.status is not None and manifest.status != comparison.status:
        raise _integrity_mismatch(run_id, "status")
    if (
        manifest.experiment_identity is not None
        and comparison.experiment_identity is not None
        and manifest.experiment_identity != comparison.experiment_identity
    ):
        raise _integrity_mismatch(run_id, "experiment identity")
    if manifest.hashes is not None:
        if (
            manifest.hashes.corpus_sha256 is not None
            and manifest.hashes.corpus_sha256 != comparison.corpus_hash
        ):
            raise _integrity_mismatch(run_id, "corpus hashes")
        if (
            manifest.hashes.benchmark_sha256 is not None
            and manifest.hashes.benchmark_sha256 != comparison.benchmark_hash
        ):
            raise _integrity_mismatch(run_id, "benchmark hashes")
    if manifest.provenance != comparison.provenance:
        raise _integrity_mismatch(run_id, "provenance")
    if manifest.verdict is not None and manifest.verdict != comparison.verdict.value:
        raise _integrity_mismatch(run_id, "verdict")
    if manifest.decision is not None and manifest.decision != comparison.decision:
        raise _integrity_mismatch(run_id, "decision")
    comparison_candidate_ids = [candidate.candidate.value for candidate in comparison.candidates]
    if manifest.evaluations is not None:
        evaluation_candidate_ids = [candidate.candidate.value for candidate in manifest.evaluations]
        if evaluation_candidate_ids != comparison_candidate_ids:
            raise _integrity_mismatch(run_id, "candidate identities")
        if manifest.evaluations != comparison.candidates:
            raise _integrity_mismatch(run_id, "candidate evaluations")
    manifest_candidate_ids = _manifest_candidate_ids(manifest, run_id)
    if manifest_candidate_ids is not None and manifest_candidate_ids != comparison_candidate_ids:
        raise _integrity_mismatch(run_id, "candidate identities")


def _recorded_artifact_hash(
    manifest: RunManifest, artifact_name: str
) -> tuple[str | None, str | None]:
    """Return an explicitly recorded artifact path and digest, if present."""
    extra = manifest.model_extra or {}
    record = extra.get("comparison" if artifact_name == "comparison.json" else "report")
    if isinstance(record, Mapping):
        typed_record = cast(Mapping[str, object], record)
        recorded_path = typed_record.get("path")
        recorded_hash = typed_record.get("sha256")
        if isinstance(recorded_path, str) and isinstance(recorded_hash, str):
            return recorded_path, recorded_hash
    hashes = extra.get("hashes")
    if isinstance(hashes, Mapping):
        typed_hashes = cast(Mapping[str, object], hashes)
        recorded_hash = typed_hashes.get(artifact_name)
        if isinstance(recorded_hash, str):
            return artifact_name, recorded_hash
    return None, None


def verify_run(root: Path, run_id: str) -> RunVerification:
    """Verify one immutable run without mutating or exposing its contents."""
    try:
        loaded = _load_run(root, run_id)
    except RunInspectionError as error:
        if error.status in {
            RunInspectionStatus.INVALID_RUN_ID,
            RunInspectionStatus.PATH_ESCAPE,
            RunInspectionStatus.RUN_NOT_FOUND,
        }:
            raise
        status = (
            RunVerificationStatus.INCOMPLETE
            if error.status is RunInspectionStatus.INCOMPLETE_RUN
            else RunVerificationStatus.CORRUPT
        )
        if error.status is RunInspectionStatus.CORRUPT_RUN and "provenance" in error.detail:
            try:
                legacy_dir = _confined_run_dir(root, run_id)
                legacy_manifest = load_manifest(legacy_dir / "manifest.json")
            except (RunInspectionError, ValueError):
                pass
            else:
                if legacy_manifest.schema_version == 1:
                    status = RunVerificationStatus.UNVERIFIABLE
        return RunVerification(
            status=status,
            run_id=run_id,
            artifacts=[],
            detail=error.detail,
        )

    if loaded.comparison is None:
        return RunVerification(
            status=RunVerificationStatus.INCOMPLETE,
            run_id=run_id,
            artifacts=[],
            detail="comparison artifact is missing",
        )

    artifacts: list[VerifiedArtifact] = []
    missing_recording = False
    mismatch = False
    for artifact_name in ("comparison.json", "report.html"):
        recorded_path, recorded_hash = _recorded_artifact_hash(loaded.manifest, artifact_name)
        if recorded_path is None or recorded_hash is None:
            missing_recording = True
            artifacts.append(VerifiedArtifact(path=artifact_name))
            continue
        artifact_path = _artifact_path(loaded.run_dir, recorded_path)
        if not artifact_path.is_file():
            mismatch = True
            artifacts.append(
                VerifiedArtifact(path=artifact_name, recorded_sha256=recorded_hash, matches=False)
            )
            continue
        actual_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        matches = actual_hash == recorded_hash
        mismatch |= not matches
        artifacts.append(
            VerifiedArtifact(
                path=artifact_name,
                recorded_sha256=recorded_hash,
                actual_sha256=actual_hash,
                matches=matches,
            )
        )

    if mismatch:
        return RunVerification(
            status=RunVerificationStatus.CORRUPT,
            run_id=run_id,
            artifacts=artifacts,
            detail="recorded artifact hash mismatch",
        )
    if missing_recording:
        return RunVerification(
            status=RunVerificationStatus.UNVERIFIABLE,
            run_id=run_id,
            artifacts=artifacts,
            detail="one or more artifact hashes were not recorded",
        )
    return RunVerification(
        status=RunVerificationStatus.VALID,
        run_id=run_id,
        artifacts=artifacts,
    )


def list_runs(root: Path) -> RunInventory:
    """List runs, retaining healthy records when an ordinary run is corrupt."""
    root = resolve_run_root(root)
    canonical_root = _confined_runs_root(root)
    if not root.exists():
        return RunInventory(runs=[])
    records: list[RunRecord] = []
    corrupt_run_id: str | None = None
    corrupt_detail: str | None = None
    for entry in sorted(root.iterdir(), key=lambda item: item.name, reverse=True):
        if entry.name.startswith("."):
            continue
        if entry.is_symlink():
            raise RunInspectionError(
                RunInspectionStatus.PATH_ESCAPE,
                f"run path cannot be a symlink: {entry}",
                run_id=entry.name,
            )
        if not entry.is_dir():
            continue
        if not entry.resolve().is_relative_to(canonical_root):
            raise RunInspectionError(
                RunInspectionStatus.PATH_ESCAPE,
                f"run path escapes runs root: {entry}",
                run_id=entry.name,
            )
        try:
            loaded = _load_run(root, entry.name)
        except RunInspectionError as error:
            if error.status is not RunInspectionStatus.CORRUPT_RUN:
                raise
            if corrupt_run_id is None:
                corrupt_run_id = entry.name
                corrupt_detail = error.detail
            detail = error.detail if len(error.detail) <= 500 else error.detail[:497] + "..."
            records.append(
                RunRecord(
                    run_id=entry.name,
                    schema_version=0,
                    created_at=None,
                    status=RunInspectionStatus.CORRUPT_RUN.value,
                    artifact_status=RunInspectionStatus.CORRUPT_RUN.value,
                    corpus_hash=None,
                    benchmark_hash=None,
                    verdict=None,
                    provenance={},
                    detail=detail,
                )
            )
            continue
        manifest = loaded.manifest
        comparison = loaded.comparison
        records.append(
            RunRecord(
                run_id=manifest.run_id,
                schema_version=manifest.schema_version,
                created_at=manifest.created_at.isoformat() if manifest.created_at else None,
                status=(
                    comparison.status.value
                    if comparison is not None
                    else manifest.status.value
                    if manifest.status is not None
                    else "INCOMPLETE"
                ),
                artifact_status="COMPLETE" if comparison is not None else "INCOMPLETE",
                corpus_hash=(
                    comparison.corpus_hash
                    if comparison is not None
                    else manifest.hashes.corpus_sha256
                    if manifest.hashes is not None
                    else None
                ),
                benchmark_hash=(
                    comparison.benchmark_hash
                    if comparison is not None
                    else manifest.hashes.benchmark_sha256
                    if manifest.hashes is not None
                    else None
                ),
                verdict=(comparison.verdict.value if comparison is not None else manifest.verdict),
                provenance=manifest.provenance.model_dump(mode="json"),
            )
        )
    return RunInventory(
        status=(
            RunInspectionStatus.CORRUPT_RUN.value
            if corrupt_run_id is not None
            else RunInspectionStatus.OK.value
        ),
        runs=records,
        run_id=corrupt_run_id,
        detail=corrupt_detail,
    )


def _candidate_snapshot(artifact: ComparisonArtifact) -> dict[str, object]:
    eligible = set(artifact.decision.eligible_candidates)
    snapshots: dict[str, object] = {}
    for candidate in sorted(artifact.candidates, key=lambda item: item.candidate.value):
        metrics = candidate.model_dump(mode="json")
        candidate_id = str(metrics.pop("candidate"))
        metrics["eligible"] = candidate.candidate in eligible
        snapshots[candidate_id] = metrics
    return snapshots


def _snapshot(artifact: ComparisonArtifact) -> dict[str, object]:
    return {
        "status": artifact.status.value,
        "corpus_hash": artifact.corpus_hash,
        "benchmark_hash": artifact.benchmark_hash,
        "provenance": artifact.provenance.model_dump(mode="json"),
        "candidates": _candidate_snapshot(artifact),
        "eligibility": {
            "eligible_candidates": sorted(
                item.value for item in artifact.decision.eligible_candidates
            ),
            "considered_candidates": sorted(
                item.value for item in artifact.decision.considered_candidates
            ),
            "quality_floor": artifact.decision.quality_floor,
            "close_quality_epsilon": artifact.decision.close_quality_epsilon,
        },
        "verdict": artifact.verdict.value,
        "decision_status": artifact.decision.status.value,
    }


def _differences(left: object, right: object, path: str = "") -> list[RunDifference]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[RunDifference] = []
        left_mapping = cast(Mapping[str, object], left)
        right_mapping = cast(Mapping[str, object], right)
        for key in sorted(set(left_mapping) | set(right_mapping)):
            child_path = f"{path}.{key}" if path else str(key)
            differences.extend(
                _differences(left_mapping.get(key), right_mapping.get(key), child_path)
            )
        return differences
    if left != right:
        left_value = cast(object, left)
        right_value = right
        return [RunDifference(path=path, left=left_value, right=right_value)]
    return []


def explain_run(root: Path, run_id: str) -> RunExplanation:
    """Explain a persisted run without recomputing eligibility or mutating it."""
    loaded = _load_run(root, run_id)
    manifest = loaded.manifest
    artifact = loaded.comparison
    decision = artifact.decision if artifact is not None else manifest.decision
    evaluations = artifact.candidates if artifact is not None else manifest.evaluations
    if decision is None or evaluations is None:
        raise RunInspectionError(
            RunInspectionStatus.INCOMPLETE_RUN,
            f"persisted decision or candidate evaluations are missing for run: {run_id}",
            run_id=run_id,
        )

    eligible_candidates = set(decision.eligible_candidates)
    candidates = [
        {
            "candidate": evaluation.candidate.value,
            "eligible": evaluation.candidate in eligible_candidates,
            "eligibility_reasons": list(evaluation.eligibility_reasons),
        }
        for evaluation in evaluations
    ]
    payload = {
        "status": RunInspectionStatus.OK.value,
        "run_id": run_id,
        "run_status": (
            artifact.status.value
            if artifact is not None
            else manifest.status.value
            if manifest.status is not None
            else "INCOMPLETE"
        ),
        "verdict": decision.verdict.value,
        "decision_status": decision.status.value,
        "rationale": decision.rationale,
        "candidates": candidates,
    }
    redacted = cast(dict[str, object], redact_payload(payload))
    return RunExplanation.model_validate(redacted, strict=True)


def compare_runs(
    root: Path,
    left_run_id: str,
    right_run_id: str,
    *,
    allow_different_corpus: bool = False,
) -> RunComparison:
    """Compare two complete run artifacts without mutating either run."""
    left_loaded = _load_run(root, left_run_id)
    right_loaded = _load_run(root, right_run_id)
    if left_loaded.comparison is None:
        raise RunInspectionError(
            RunInspectionStatus.INCOMPLETE_RUN,
            f"comparison artifact is missing for run: {left_run_id}",
            run_id=left_run_id,
        )
    if right_loaded.comparison is None:
        raise RunInspectionError(
            RunInspectionStatus.INCOMPLETE_RUN,
            f"comparison artifact is missing for run: {right_run_id}",
            run_id=right_run_id,
        )
    left = _snapshot(left_loaded.comparison)
    right = _snapshot(right_loaded.comparison)
    if left_loaded.manifest.schema_version == 1 or right_loaded.manifest.schema_version == 1:
        raise RunInspectionError(
            RunInspectionStatus.DIFFERENT_CORPUS,
            "legacy artifacts are readable but non-comparable; regenerate both runs",
        )
    left_identity = left_loaded.comparison.experiment_identity
    right_identity = right_loaded.comparison.experiment_identity
    same_corpus = (
        left_identity == right_identity
        if left_identity is not None and right_identity is not None
        else (
            left_loaded.comparison.corpus_hash == right_loaded.comparison.corpus_hash
            and left_loaded.comparison.benchmark_hash == right_loaded.comparison.benchmark_hash
        )
    )
    if not same_corpus and not allow_different_corpus:
        raise RunInspectionError(
            RunInspectionStatus.DIFFERENT_CORPUS,
            "corpus or benchmark hashes differ; pass --allow-different-corpus to inspect a "
            "non-equivalent comparison",
        )
    differences = _differences(left, right)
    equivalent = same_corpus and not differences
    return RunComparison(
        status=(
            RunInspectionStatus.EQUIVALENT.value
            if equivalent
            else RunInspectionStatus.NON_EQUIVALENT.value
        ),
        equivalent=equivalent,
        comparable=same_corpus,
        comparison_basis=("SAME_CORPUS" if same_corpus else "DIFFERENT_CORPUS_ALLOWED"),
        left_run_id=left_run_id,
        right_run_id=right_run_id,
        left=left,
        right=right,
        differences=differences,
    )
