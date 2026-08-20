import hashlib
import json
import os
import re
import shutil
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
RELEASE_SOURCE_FIXED_FILES = (
    "pyproject.toml",
    "uv.lock",
    "candidate-pins.json",
)
RELEASE_SOURCE_TREE = "src/autobrain"
RELEASE_SOURCE_EXCLUDES = ("**/__pycache__/**", "**/*.pyc")
RELEASE_SOURCE_HASH_BASIS = (
    "SHA-256 over each sorted UTF-8 repository-relative path, a NUL byte, the "
    "SHA-256 digest bytes of the exact file content, and a trailing NUL byte"
)
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


def _release_source_files(root: Path = Path(".")) -> tuple[str, ...]:
    tree = root / RELEASE_SOURCE_TREE
    tree_files = (
        path.relative_to(root).as_posix()
        for path in tree.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    return tuple(sorted((*RELEASE_SOURCE_FIXED_FILES, *tree_files)))


def _release_source_digest(root: Path = Path(".")) -> str:
    digest = hashlib.sha256()
    for relative in _release_source_files(root):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256((root / relative).read_bytes()).digest())
        digest.update(b"\0")
    return digest.hexdigest()


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


def test_release_version_is_coherent_across_runtime_package_formula_and_evidence() -> None:
    from autobrain import __version__

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    formula = json.loads(Path("release/homebrew-formula.json").read_text(encoding="utf-8"))
    evidence = json.loads(Path(EVIDENCE_ROOT, "manifest.json").read_text(encoding="utf-8"))

    assert __version__ == project["version"] == formula["source"]["version"]
    assert evidence["release"]["version"] == __version__


def test_release_source_digest_has_explicit_scope_and_expected_mutation_behavior(
    tmp_path: Path,
) -> None:
    for relative in _release_source_files():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative, target)

    baseline = _release_source_digest(tmp_path)
    assert _release_source_files(tmp_path) == _release_source_files()

    for unrelated in (
        ".senpi/task-10-final-qa/manifest.json",
        "docs/security-and-privacy.md",
        "tests/test_packaging.py",
    ):
        target = tmp_path / unrelated
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("unrelated mutation\n", encoding="utf-8")
    assert _release_source_digest(tmp_path) == baseline

    for relative in (*RELEASE_SOURCE_FIXED_FILES, "src/autobrain/__init__.py"):
        target = tmp_path / relative
        original = target.read_bytes()
        target.write_bytes(original + b"\n")
        assert _release_source_digest(tmp_path) != baseline
        target.write_bytes(original)
    assert _release_source_digest(tmp_path) == baseline


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
    assert release["source_digest"] == {
        "algorithm": "sha256",
        "hash_basis": RELEASE_SOURCE_HASH_BASIS,
        "fixed_files": list(RELEASE_SOURCE_FIXED_FILES),
        "recursive_tree": RELEASE_SOURCE_TREE,
        "excludes": list(RELEASE_SOURCE_EXCLUDES),
        "sha256": _release_source_digest(),
    }
    assert manifest["runtime_evidence"] == {
        "status": "UNBOUND_CURRENT_RELEASE",
        "reason": "RUNTIME_SOURCE_DIGEST_AND_0.1.1_WHEEL_NOT_RECORDED",
        "observed_installed_wheel": "autobrain-0.1.0-py3-none-any.whl",
        "external_access": "NOT_ATTEMPTED",
    }

    retained = b"\n".join((root / relative).read_bytes() for relative in sorted(actual))
    assert all(pattern.search(retained) is None for pattern in SECRET_PATTERNS)


def test_retained_comparison_uses_authoritative_release_source_digest() -> None:
    formula = json.loads(Path("release/homebrew-formula.json").read_text(encoding="utf-8"))
    manifest = json.loads(Path(EVIDENCE_ROOT, "manifest.json").read_text(encoding="utf-8"))
    comparison = json.loads(Path(EVIDENCE_ROOT, "comparison.json").read_text(encoding="utf-8"))
    expected = _release_source_digest()

    assert formula["source"]["tree_sha256"] == expected
    assert manifest["release"]["source_digest"]["sha256"] == expected
    assert comparison["runtime_evidence"]["reviewed_source_sha256"] == expected


def test_retained_task_text_uses_authoritative_release_source_digest() -> None:
    task_text = Path(EVIDENCE_ROOT, "task-10-final-qa.txt").read_text(encoding="utf-8")
    expected = _release_source_digest()

    assert f"`sha256:{expected}`" in task_text


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
