import json
import tomllib
from importlib import resources
from pathlib import Path


def test_supported_python_range_excludes_unverified_314() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["requires-python"] == ">=3.12,<3.14.0a0"


def test_candidate_pins_are_an_importable_package_resource() -> None:
    pins = resources.files("autobrain").joinpath("candidate-pins.json")
    assert pins.is_file()
    assert '"schema_version": 1' in pins.read_text(encoding="utf-8")


def test_sdist_excludes_stale_senpi_evidence_and_keeps_current_allowlist() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    sdist = project["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert ".senpi/task-10-final-qa/**" in sdist["include"]
    assert ".senpi/task-10-final-qa-20260818/**" in sdist["exclude"]
    assert ".senpi/task9-final-qa-20260818/**" in sdist["exclude"]
    assert ".senpi/task9-qa-20260818/**" in sdist["exclude"]
    assert ".senpi/task10-browser-fixture/**" in sdist["exclude"]


def test_redacted_final_attempt_contains_every_required_evidence_reference() -> None:
    root = Path(".senpi/task-10-final-qa")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert all((root / relative).is_file() for relative in manifest["allowlist"])
