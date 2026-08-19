import json
import subprocess
import sys
from collections.abc import Callable
from types import ModuleType

import pytest

from autobrain import bootstrap


class FakeCli(ModuleType):
    app: Callable[[], None]


def test_bootstrap_imports_no_third_party_cli_dependencies() -> None:
    script = (
        "import sys; import autobrain.bootstrap; "
        "assert not any(name == 'pydantic' or name.startswith('typer') for name in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_python_314_fails_closed_as_typed_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = bootstrap.main(["doctor", "--json"], version_info=(3, 14, 0))
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert code == 1
    assert captured.err == ""
    assert report["status"] == "ENV_UNAVAILABLE"
    assert report["checks"][0]["name"] == "python"
    assert "Python >=3.14 is unsupported" in report["checks"][0]["detail"]


def test_python_314_help_fails_without_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    code = bootstrap.main(["--help"], version_info=(3, 14, 0))
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err == ""
    assert captured.out.startswith("ENV_UNAVAILABLE:")
    assert "Traceback" not in captured.out


@pytest.mark.parametrize("version_info", [(3, 12, 0), (3, 13, 9)])
def test_supported_python_delegates_to_cli(
    monkeypatch: pytest.MonkeyPatch,
    version_info: tuple[int, int, int],
) -> None:
    calls: list[bool] = []
    fake_cli = FakeCli("autobrain.cli")

    def fake_app() -> None:
        calls.append(True)

    fake_cli.app = fake_app
    monkeypatch.setitem(sys.modules, "autobrain.cli", fake_cli)
    assert bootstrap.main([], version_info=version_info) == 0
    assert calls == [True]
