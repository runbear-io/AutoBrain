import os
from pathlib import Path

import pytest
from harness import resolve_autobrain_python


def test_web_interpreter_uses_autobrain_python(tmp_path: Path) -> None:
    selected = tmp_path / "python-under-test"
    selected.touch()
    selected.chmod(0o755)
    assert resolve_autobrain_python({"AUTOBRAIN_PYTHON": str(selected)}) == selected


def test_web_interpreter_does_not_fall_back_to_project_venv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUTOBRAIN_PYTHON", raising=False)
    try:
        resolve_autobrain_python(os.environ)
    except RuntimeError as error:
        assert "AUTOBRAIN_PYTHON" in str(error)
        assert ".venv" not in str(error)
    else:
        raise AssertionError("missing AUTOBRAIN_PYTHON must fail closed")
