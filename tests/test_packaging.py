import json
import os
import subprocess
import tarfile
import tomllib
from importlib import resources
from pathlib import Path

EVIDENCE_ROOT = ".senpi/task-10-final-qa"
EVIDENCE_ALLOWLIST = {
    "comparison.json",
    "manifest.json",
    "screenshots/report-1280.png",
    "screenshots/report-375.png",
    "screenshots/report-768.png",
    "task-10-final-qa.txt",
}


def test_supported_python_range_excludes_unverified_314() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["requires-python"] == ">=3.12,<3.14.0a0"


def test_textual_is_a_direct_runtime_dependency() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert "textual>=1,<8" in project["dependencies"]


def test_textual_runtime_dependency_is_locked_from_pypi() -> None:
    lock = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))
    packages = {package["name"]: package for package in lock["package"]}

    assert {"name": "textual"} in packages["autobrain"]["dependencies"]
    assert packages["textual"]["source"] == {"registry": "https://pypi.org/simple"}


def test_candidate_pins_are_an_importable_package_resource() -> None:
    pins = resources.files("autobrain").joinpath("candidate-pins.json")
    assert pins.is_file()
    assert '"schema_version": 1' in pins.read_text(encoding="utf-8")


def test_sdist_configuration_force_includes_only_current_evidence_allowlist() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    sdist = project["tool"]["hatch"]["build"]["targets"]["sdist"]
    expected = {f"{EVIDENCE_ROOT}/{relative}" for relative in EVIDENCE_ALLOWLIST}

    assert sdist["force-include"] == {path: path for path in expected}
    assert all("*" not in path for path in sdist["force-include"])
    assert not any(path == ".senpi" or path.startswith(".senpi/") for path in sdist["include"])


def test_built_sdist_contains_exactly_the_redacted_evidence_allowlist(tmp_path: Path) -> None:
    subprocess.run(
        [os.environ.get("AUTOBRAIN_TEST_UV", "uv"), "build", "--sdist", "--out-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    sdist = next(tmp_path.glob("autobrain-*.tar.gz"))
    with tarfile.open(sdist, "r:gz") as archive:
        evidence = {
            member.name.split(f"{EVIDENCE_ROOT}/", 1)[1]
            for member in archive.getmembers()
            if f"{EVIDENCE_ROOT}/" in member.name and member.isfile()
        }

    assert evidence == EVIDENCE_ALLOWLIST


def test_redacted_final_attempt_contains_every_required_evidence_reference() -> None:
    root = Path(".senpi/task-10-final-qa")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert all((root / relative).is_file() for relative in manifest["allowlist"])
