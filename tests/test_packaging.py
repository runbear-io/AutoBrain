import hashlib
import json
import os
import re
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
ZERO_SHA256 = "0" * 64
SECRET_PATTERNS = (
    re.compile(rb"sk-(?:live|test|proj)-[A-Za-z0-9_-]{8,}"),
    re.compile(rb"xox[baprs]-[A-Za-z0-9-]{8,}"),
    re.compile(rb"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._~-]{8,}"),
    re.compile(rb"(?i)(?:api[_-]?key|client[_-]?secret|password)\s*[=:]\s*[^\s\"']{8,}"),
)


def _manifest_self_hash(raw: bytes, claimed_digest: str) -> str:
    needle = claimed_digest.encode("ascii")
    assert len(needle) == 64
    assert raw.count(needle) == 1
    return hashlib.sha256(raw.replace(needle, ZERO_SHA256.encode("ascii"), 1)).hexdigest()


def _evidence_members(archive: tarfile.TarFile) -> dict[str, bytes]:
    evidence: dict[str, bytes] = {}
    for member in archive.getmembers():
        marker = f"{EVIDENCE_ROOT}/"
        if marker not in member.name or not member.isfile():
            continue
        extracted = archive.extractfile(member)
        assert extracted is not None
        evidence[member.name.split(marker, 1)[1]] = extracted.read()
    return evidence


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


def test_release_evidence_manifest_is_complete_canonical_and_fail_closed() -> None:
    root = Path(EVIDENCE_ROOT)
    manifest_path = root / "manifest.json"
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}

    assert actual == EVIDENCE_ALLOWLIST
    assert manifest["allowlist"] == sorted(EVIDENCE_ALLOWLIST)
    assert set(manifest["hashes"]) == EVIDENCE_ALLOWLIST
    assert manifest["hash_basis"] == (
        "SHA-256 of the exact UTF-8 manifest.json bytes after replacing the single "
        "hashes.manifest.json value with 64 ASCII zeroes"
    )

    for relative in sorted(EVIDENCE_ALLOWLIST - {"manifest.json"}):
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert manifest["hashes"][relative] == digest
    assert manifest["hashes"]["manifest.json"] == _manifest_self_hash(
        raw_manifest, manifest["hashes"]["manifest.json"]
    )

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    release = manifest["release"]
    assert release["version"] == project["version"]
    assert release["candidate_commit"] == "036714b91df82d1eb538d39d8251f86ae9f55a4e"
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", release["candidate_commit"], "HEAD"],
        check=True,
        capture_output=True,
    )
    assert manifest["runtime_evidence"] == {
        "status": "UNBOUND_CURRENT_RELEASE",
        "reason": "SOURCE_COMMIT_AND_0.1.1_WHEEL_NOT_RECORDED",
        "observed_installed_wheel": "autobrain-0.1.0-py3-none-any.whl",
        "external_access": "NOT_ATTEMPTED",
    }

    retained = b"\n".join((root / relative).read_bytes() for relative in sorted(actual))
    assert all(pattern.search(retained) is None for pattern in SECRET_PATTERNS)


def test_built_sdist_closes_over_exact_verified_release_evidence(tmp_path: Path) -> None:
    subprocess.run(
        [os.environ.get("AUTOBRAIN_TEST_UV", "uv"), "build", "--sdist", "--out-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    sdist = next(tmp_path.glob("autobrain-*.tar.gz"))
    with tarfile.open(sdist, "r:gz") as archive:
        evidence = _evidence_members(archive)

    assert set(evidence) == EVIDENCE_ALLOWLIST
    manifest_raw = evidence["manifest.json"]
    manifest = json.loads(manifest_raw)
    for relative in sorted(EVIDENCE_ALLOWLIST - {"manifest.json"}):
        assert manifest["hashes"][relative] == hashlib.sha256(evidence[relative]).hexdigest()
    assert manifest["hashes"]["manifest.json"] == _manifest_self_hash(
        manifest_raw, manifest["hashes"]["manifest.json"]
    )
    assert all(
        pattern.search(payload) is None
        for payload in evidence.values()
        for pattern in SECRET_PATTERNS
    )
