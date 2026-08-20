"""Confined filesystem layout for AutoBrain state."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PathConfinementError(ValueError):
    """A requested path escapes the AutoBrain state root."""


class OccupiedRunError(FileExistsError):
    """A non-empty run directory already exists."""


def is_valid_run_id(run_id: str) -> bool:
    """Return whether a run ID is safe as one confined path component."""
    return bool(_RUN_ID.fullmatch(run_id)) and run_id not in {".", ".."}


def validate_state_root(root: Path, *, explicit: bool = False) -> None:
    """Validate an explicitly configured state root before inspecting it."""
    if any(part == ".." for part in root.parts):
        raise PathConfinementError(f"state root contains lexical traversal: {root}")
    if explicit:
        if root.is_symlink():
            raise PathConfinementError(f"state root cannot be a symlink: {root}")
        if not root.is_dir():
            raise PathConfinementError(f"state root is not a directory: {root}")


@dataclass(frozen=True)
class AutoBrainPaths:
    root: Path
    runs: Path
    tools: Path
    cache: Path

    @classmethod
    def from_home(cls, home: Path | None = None) -> "AutoBrainPaths":
        configured = os.environ.get("AUTOBRAIN_HOME") if home is None else None
        root = Path(configured).expanduser() if configured else (home or Path.home()) / ".autobrain"
        validate_state_root(root, explicit=configured is not None)
        return cls(root=root, runs=root / "runs", tools=root / "tools", cache=root / "cache")

    @property
    def sources(self) -> Path:
        return self.root / "sources"

    @staticmethod
    def validate_output_root(root: Path) -> None:
        """Reject lexical traversal before any output path is created."""
        if any(part == ".." for part in root.parts):
            raise PathConfinementError(f"output root contains lexical traversal: {root}")

    def ensure_base_dirs(self) -> None:
        expected_root = self.root.parent.resolve() / self.root.name
        if self.root.is_symlink():
            raise PathConfinementError(f"state root cannot be a symlink: {self.root}")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        canonical_root = self.root.resolve()
        if canonical_root != expected_root:
            raise PathConfinementError(f"state root escapes its canonical parent: {self.root}")
        for directory in (self.runs, self.tools, self.cache, self.sources):
            if directory.is_symlink():
                raise PathConfinementError(f"state directory cannot be a symlink: {directory}")
            directory.mkdir(mode=0o700, exist_ok=True)
            if directory.resolve().parent != canonical_root:
                raise PathConfinementError(f"state directory escapes root: {directory}")

    def create_run(self, run_id: str) -> Path:
        if not is_valid_run_id(run_id):
            raise PathConfinementError(f"invalid run id: {run_id!r}")
        self.ensure_base_dirs()
        runs_root = self.runs.resolve()
        candidate = self.runs / run_id
        if candidate.is_symlink():
            raise PathConfinementError(f"run path cannot be a symlink: {candidate}")
        if not candidate.resolve(strict=False).is_relative_to(runs_root):
            raise PathConfinementError(f"run path escapes {runs_root}")
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            if not candidate.is_dir():
                raise PathConfinementError(
                    f"run path is not a confined directory: {candidate}"
                ) from None
            if any(candidate.iterdir()):
                raise OccupiedRunError(f"run already contains files: {candidate}") from None
        return candidate

    def tool_dir(self, candidate_id: str) -> Path:
        if not _RUN_ID.fullmatch(candidate_id):
            raise PathConfinementError(f"invalid candidate id: {candidate_id!r}")
        self.ensure_base_dirs()
        path = self.tools / candidate_id
        if path.is_symlink() or not path.resolve(strict=False).is_relative_to(self.tools.resolve()):
            raise PathConfinementError("tool path escaped tool cache")
        return path
