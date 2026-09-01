"""Operator-facing `autobrain serve` command for the local run fixture.

The command exists so a browser client has a documented, fixed port to talk to
instead of an ephemeral one nobody can predict. It must state plainly that it
is local and unauthenticated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autobrain.cli import app
from autobrain.local_server import DEFAULT_LOCAL_PORT, RunOutcome, RunOutcomeStatus
from autobrain.projection import PROJECTION_SCHEMA_VERSION
from tests.test_projection import artifact, project_comparison


def test_serve_help_documents_the_local_unauthenticated_scope() -> None:
    result = CliRunner().invoke(app, ["serve", "--help"])

    assert result.exit_code == 0
    combined = result.stdout.lower()
    assert "local" in combined
    assert str(DEFAULT_LOCAL_PORT) in result.stdout


def test_serve_rejects_a_directory_outside_the_configured_run_root(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    result = CliRunner().invoke(
        app,
        ["serve", "--run-root", str(run_root), "--run-dir", str(outside), "--check"],
    )
    assert result.exit_code != 0
    assert "PATH_ESCAPE" in result.stdout


def test_serve_rejects_a_symlinked_run_directory(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (run_root / "run-1").symlink_to(outside, target_is_directory=True)
    result = CliRunner().invoke(
        app,
        ["serve", "--run-root", str(run_root), "--run-dir", str(run_root / "run-1"), "--check"],
    )
    assert result.exit_code != 0
    assert "PATH_ESCAPE" in result.stdout


def test_serve_reports_no_run_when_the_directory_is_empty(tmp_path: Path) -> None:
    """With nothing to serve the command fails closed instead of inventing a run."""
    result = CliRunner().invoke(
        app,
        ["serve", "--run-dir", str(tmp_path), "--check"],
    )

    assert result.exit_code != 0
    assert "no comparison" in result.stdout.lower()


def test_serve_check_reports_the_projection_it_would_publish(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    comparison.write_text(artifact().model_dump_json(), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["serve", "--run-dir", str(tmp_path), "--check", "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == RunOutcomeStatus.SUCCEEDED.value
    assert payload["projection"]["schema_version"] == PROJECTION_SCHEMA_VERSION
    assert payload["projection"]["run_id"] == "RUN-A41F"


def test_serve_check_reports_failure_for_an_unreadable_comparison(tmp_path: Path) -> None:
    (tmp_path / "comparison.json").write_text("{ not json", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["serve", "--run-dir", str(tmp_path), "--check", "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == RunOutcomeStatus.FAILED.value
    assert payload["projection"] is None
    assert payload["error"]


@pytest.mark.parametrize("port", [0, DEFAULT_LOCAL_PORT, 9000])
def test_serve_accepts_an_explicit_port(tmp_path: Path, port: int) -> None:
    comparison = tmp_path / "comparison.json"
    comparison.write_text(artifact().model_dump_json(), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["serve", "--run-dir", str(tmp_path), "--port", str(port), "--check"],
    )

    assert result.exit_code == 0


def test_outcome_for_run_dir_maps_a_valid_comparison_to_success(tmp_path: Path) -> None:
    from autobrain.local_server import outcome_for_run_dir

    (tmp_path / "comparison.json").write_text(artifact().model_dump_json(), encoding="utf-8")

    outcome = outcome_for_run_dir(tmp_path)

    assert outcome.status is RunOutcomeStatus.SUCCEEDED
    assert outcome.projection is not None
    assert outcome.projection.run_id == "RUN-A41F"


def test_outcome_for_run_dir_reports_a_missing_file_as_failure(tmp_path: Path) -> None:
    from autobrain.local_server import outcome_for_run_dir

    outcome = outcome_for_run_dir(tmp_path)

    assert outcome.status is RunOutcomeStatus.FAILED
    assert outcome.projection is None
    assert "comparison" in (outcome.error or "").lower()


def test_outcome_for_run_dir_projects_the_same_shape_as_direct_projection(tmp_path: Path) -> None:
    """The served payload must equal the in-process projection of the same run."""
    from autobrain.local_server import outcome_for_run_dir

    source = artifact()
    (tmp_path / "comparison.json").write_text(source.model_dump_json(), encoding="utf-8")

    served = outcome_for_run_dir(tmp_path)
    direct = RunOutcome.succeeded(project_comparison(source))

    assert served.to_payload() == direct.to_payload()
