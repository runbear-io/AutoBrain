from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from autobrain.fixture import write_fixture


def resolve_autobrain_python(environ: Mapping[str, str]) -> Path:
    value = environ.get("AUTOBRAIN_PYTHON")
    if not value:
        raise RuntimeError("AUTOBRAIN_PYTHON is required for web E2E interpreter resolution")
    path = Path(value).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"AUTOBRAIN_PYTHON is not an executable file: {path}")
    return path


def _fixture(path: Path) -> None:
    write_fixture(path, seed=5, fixture_id="installed-binary-e2e")


@dataclass
class Result:
    completed: subprocess.CompletedProcess[str]
    json: Any = None

    @property
    def returncode(self) -> int:
        return self.completed.returncode

    @property
    def stdout(self) -> str:
        return self.completed.stdout

    @property
    def stderr(self) -> str:
        return self.completed.stderr

    @property
    def output(self) -> str:
        return self.stdout + self.stderr

    @property
    def lines(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in self.stdout.splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                result[key] = value
        return result

    @property
    def process_returncode(self) -> int | None:
        return self.returncode

    @property
    def timed_out(self) -> bool:
        return False


@dataclass
class TimeoutResult(Result):
    _timed_out: bool = True

    @property
    def timed_out(self) -> bool:
        return self._timed_out


class E2EHarness:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.home = root / "home"
        self.autobrain_home = root / "autobrain-home"
        self.run_root = root / "runs"
        self.fixture_path = root / "fixture.json"
        self.fake_codex = root / "fake-codex"
        self.fake_codex.write_text(
            Path(__file__).with_name("fake_codex.py").read_text(), encoding="utf-8"
        )
        self.fake_codex.chmod(0o755)
        self.empty_run_dir = self.run_root / "empty-run"
        for directory in (self.home, self.autobrain_home, self.run_root, self.empty_run_dir):
            directory.mkdir(parents=True)
        _fixture(self.fixture_path)
        configured_binary = os.environ.get("AUTOBRAIN_BINARY")
        self.binary = (
            Path(configured_binary)
            if configured_binary
            else Path(os.environ.get("AUTOBRAIN_WORKTREE", Path.cwd())) / ".venv/bin/autobrain"
        )

    def environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "HOME": str(self.home),
                "AUTOBRAIN_HOME": str(self.autobrain_home),
                "AUTOBRAIN_RUN_ROOT": str(self.run_root),
                "AUTOBRAIN_ALLOW_TEST_FIXTURE": "1",
                "AUTOBRAIN_TEST_FIXTURE_PATH": str(self.fixture_path),
                "AUTOBRAIN_ENABLE_TEST_SEMANTIC_EMBEDDING": "1",
                "AUTOBRAIN_EMBEDDING_BACKEND": "local-hash",
                "AUTOBRAIN_CODEX_COMMAND": str(self.fake_codex),
            }
        )
        return env

    def command(self, *args: str) -> list[str]:
        return [str(self.binary), *args]

    def run(self, *args: str) -> Result:
        completed = subprocess.run(
            self.command(*args), env=self.environment(), capture_output=True, text=True, check=False
        )
        parsed = None
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(completed.stdout)
        return Result(completed, parsed)

    def run_with_timeout(self, *args: str, timeout: float) -> TimeoutResult:
        try:
            completed = subprocess.run(
                self.command(*args),
                env=self.environment(),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
            return TimeoutResult(completed, _timed_out=False)
        except subprocess.TimeoutExpired as error:
            return TimeoutResult(
                subprocess.CompletedProcess[str](
                    self.command(*args),
                    -signal.SIGTERM,
                    cast(str, error.stdout or ""),
                    cast(str, error.stderr or ""),
                )
            )

    def cancel_after_ready(self, *args: str) -> Result:
        process = subprocess.Popen(
            self.command(*args),
            env=self.environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        first_line = process.stdout.readline()
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate()
        return Result(
            subprocess.CompletedProcess(
                process.args, process.returncode, first_line + stdout, stderr
            )
        )

    def live_children(self) -> list[int]:
        return []
