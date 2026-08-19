from pathlib import Path

import pytest

from autobrain.paths import AutoBrainPaths, OccupiedRunError, PathConfinementError


def test_layout_is_confined_and_created(tmp_path: Path) -> None:
    paths = AutoBrainPaths.from_home(tmp_path)
    paths.ensure_base_dirs()
    run = paths.create_run("run-20260818")
    assert run == tmp_path / ".autobrain" / "runs" / "run-20260818"
    assert paths.tools == tmp_path / ".autobrain" / "tools"
    assert paths.cache.is_dir()
    assert paths.sources.is_dir()


@pytest.mark.parametrize("run_id", ["../escape", "/tmp/escape", "a/b", "", ".", ".."])
def test_malformed_run_ids_are_rejected(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(PathConfinementError):
        AutoBrainPaths.from_home(tmp_path).create_run(run_id)


def test_nonempty_run_is_not_overwritten_and_dirty_file_is_preserved(tmp_path: Path) -> None:
    paths = AutoBrainPaths.from_home(tmp_path)
    dirty = paths.runs / "stale" / "keep.txt"
    dirty.parent.mkdir(parents=True)
    dirty.write_text("preserve me", encoding="utf-8")
    with pytest.raises(OccupiedRunError):
        paths.create_run("stale")
    assert dirty.read_text(encoding="utf-8") == "preserve me"


def test_symlinked_runs_root_is_rejected_before_writing_outside(tmp_path: Path) -> None:
    paths = AutoBrainPaths.from_home(tmp_path)
    outside = tmp_path / "outside-runs"
    outside.mkdir()
    paths.root.mkdir()
    paths.runs.symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathConfinementError):
        paths.create_run("escaped")
    assert list(outside.iterdir()) == []


def test_symlinked_state_root_is_rejected(tmp_path: Path) -> None:
    paths = AutoBrainPaths.from_home(tmp_path)
    outside = tmp_path / "outside-root"
    outside.mkdir()
    paths.root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathConfinementError):
        paths.create_run("escaped")
    assert list(outside.iterdir()) == []


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    paths = AutoBrainPaths.from_home(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    paths.runs.mkdir(parents=True)
    (paths.runs / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathConfinementError):
        paths.create_run("linked")


def test_output_root_rejects_lexical_traversal_before_creation(tmp_path: Path) -> None:
    output = tmp_path / "confined" / ".." / "target"
    with pytest.raises(PathConfinementError):
        AutoBrainPaths.validate_output_root(output)
    assert not (tmp_path / "confined").exists()
    assert not (tmp_path / "target").exists()
