from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from autobrain.release_formula import (
    ArchiveKind,
    FormulaManifest,
    FormulaParseError,
    ReleaseSourceError,
    classify_archive,
    load_formula_manifest,
    parse_formula,
    run_cli,
    verify_formula,
    write_formula_atomic,
)
from autobrain.release_formula import generate_formula as _raw_generate_formula

ROOT = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "release"
MANIFEST = ROOT / "release" / "homebrew-formula.json"
TAP_FORMULA = Path(
    os.environ.get(
        "AUTOBRAIN_PREPARED_TAP_FORMULA",
        ROOT.parent / "homebrew-autobrain-wt-completeness" / "Formula" / "autobrain.rb",
    )
)
PUBLISHED_PLATFORM_WHEELS = {
    "cffi",
    "cryptography",
    "greenlet",
    "grpcio",
    "jiter",
    "markupsafe",
    "numpy",
    "pydantic-core",
    "rpds-py",
}
TARGET_PLATFORM_WHEELS = PUBLISHED_PLATFORM_WHEELS - {"greenlet"}


def test_valid_fixture_accepts_universal_wheels_and_sdists() -> None:
    verification = verify_formula((FIXTURES / "valid_formula.rb").read_text())

    assert verification.valid
    assert {resource.archive_kind for resource in verification.resources} == {
        ArchiveKind.UNIVERSAL_WHEEL,
        ArchiveKind.SDIST,
    }
    assert verification.platform_wheel_violations == ()


def test_custom_install_route_must_audit_every_platform_wheel() -> None:
    formula = """
resource "native" do
  url "https://example.test/native-1.0-cp313-cp313-macosx_11_0_arm64.whl"
  sha256 "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
end

def install
  resource("native").stage { virtualenv.pip_install Pathname.pwd.glob("*.whl") }
end
"""

    verification = verify_formula(formula)

    assert not verification.valid
    assert verification.platform_wheel_violations == ("native",)


def test_published_v0_1_1_formula_identifies_complete_platform_wheel_set() -> None:
    verification = verify_formula((FIXTURES / "autobrain-v0.1.1.rb").read_text())

    platform_wheels = {
        resource.name
        for resource in verification.resources
        if resource.archive_kind is ArchiveKind.PLATFORM_WHEEL
    }
    assert platform_wheels == PUBLISHED_PLATFORM_WHEELS
    assert verification.platform_wheel_violations == tuple(sorted(PUBLISHED_PLATFORM_WHEELS))
    assert not verification.valid


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.test/demo-1.0-py3-none-any.whl", ArchiveKind.UNIVERSAL_WHEEL),
        (
            "https://example.test/demo-1.0-cp313-cp313-macosx_11_0_arm64.whl",
            ArchiveKind.PLATFORM_WHEEL,
        ),
        ("https://example.test/demo-1.0.tar.gz", ArchiveKind.SDIST),
        ("https://example.test/demo-1.0.zip", ArchiveKind.SDIST),
    ],
)
def test_archive_classifier_uses_standard_distribution_tags(
    url: str, expected: ArchiveKind
) -> None:
    assert classify_archive(url) is expected


@pytest.mark.parametrize(
    "formula",
    [
        'resource "broken" do\n  sha256 "missing-url"\nend\n',
        'resource "broken" do\n  url "https://example.test/not-an-archive"\nend\n',
        'resource "broken" do\n  url "https://example.test/demo.whl"\nend\n',
        (
            'resource "duplicate" do\n  url "https://example.test/a-1-py3-none-any.whl"\nend\n'
            'resource "duplicate" do\n  url "https://example.test/b-1.tar.gz"\nend\n'
        ),
        (
            'resource "nested" do\n'
            '  resource "inner" do\n'
            '    url "https://example.test/inner-1.tar.gz"\n'
            "  end\n"
            "end\n"
        ),
    ],
)
def test_malformed_formula_is_rejected(formula: str) -> None:
    with pytest.raises(FormulaParseError):
        parse_formula(formula)


def _write_manifest(tmp_path: Path, data: object) -> Path:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(data))
    return manifest_path


def _release_paths(root: Path) -> list[Path]:
    paths = [
        root / "pyproject.toml",
        root / "uv.lock",
        root / "candidate-pins.json",
    ]
    paths.extend(
        path
        for path in (root / "src" / "autobrain").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _release_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _release_paths(root):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_release_archive(path: Path, root: Path) -> None:
    tar_payload = io.BytesIO()
    with tarfile.open(fileobj=tar_payload, mode="w") as archive:
        for source in _release_paths(root):
            relative = source.relative_to(root)
            payload = source.read_bytes()
            info = tarfile.TarInfo(f"AutoBrain-release/{relative.as_posix()}")
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    with (
        path.open("wb") as destination,
        gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as compressed,
    ):
        compressed.write(tar_payload.getvalue())


def generate_formula(
    manifest: FormulaManifest,
    *,
    uv_lock_path: Path,
    pyproject_path: Path,
) -> str:
    with tempfile.TemporaryDirectory() as temporary:
        archive_path = Path(temporary) / "release.tar.gz"
        _write_release_archive(archive_path, pyproject_path.parent)
        source = replace(
            manifest.source,
            status="approved",
            reviewed_commit="036714b91df82d1eb538d39d8251f86ae9f55a4e",
            tree_sha256=_release_tree_sha256(pyproject_path.parent),
            sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        )
        return _raw_generate_formula(
            replace(manifest, source=source),
            uv_lock_path=uv_lock_path,
            pyproject_path=pyproject_path,
            source_archive_path=archive_path,
        )


def _approved_manifest(tmp_path: Path, archive_path: Path) -> Path:
    data = json.loads(MANIFEST.read_text())
    data["source"].update(
        {
            "status": "approved",
            "reviewed_commit": "036714b91df82d1eb538d39d8251f86ae9f55a4e",
            "tree_sha256": _release_tree_sha256(ROOT),
            "url": "https://github.com/runbear-io/AutoBrain/archive/refs/tags/v0.1.1.tar.gz",
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        }
    )
    return _write_manifest(tmp_path, data)


def test_basic_formula_verification_cannot_detect_source_tree_mismatch() -> None:
    formula = (FIXTURES / "valid_formula.rb").read_text()

    assert verify_formula(formula).valid
    assert "pyproject.toml" not in formula
    assert "uv.lock" not in formula
    assert "candidate-pins.json" not in formula


@pytest.mark.parametrize(
    "relative",
    [
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("candidate-pins.json"),
        Path("src/autobrain/release_formula.py"),
    ],
)
def test_generator_rejects_archive_from_a_different_source_tree(
    tmp_path: Path, relative: Path
) -> None:
    stale_root = tmp_path / "stale"
    stale_root.mkdir()
    for source in _release_paths(ROOT):
        target = stale_root / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    changed = stale_root / relative
    changed.write_bytes(changed.read_bytes() + b"\n# stale release tree\n")
    archive_path = tmp_path / "stale.tar.gz"
    _write_release_archive(archive_path, stale_root)
    manifest_path = _approved_manifest(tmp_path, archive_path)

    mismatch = rf"source archive.*{re.escape(relative.as_posix())}"
    with pytest.raises(FormulaParseError, match=mismatch):
        _raw_generate_formula(
            load_formula_manifest(manifest_path),
            uv_lock_path=ROOT / "uv.lock",
            pyproject_path=ROOT / "pyproject.toml",
            source_archive_path=archive_path,
        )


def test_checked_in_approved_release_requires_the_bound_source_archive() -> None:
    with pytest.raises(
        ReleaseSourceError,
        match=r"approved release generation requires --source-archive",
    ):
        _raw_generate_formula(
            load_formula_manifest(MANIFEST),
            uv_lock_path=ROOT / "uv.lock",
            pyproject_path=ROOT / "pyproject.toml",
        )


def test_checked_in_manifest_binds_current_release_tree_digest() -> None:
    source = json.loads(MANIFEST.read_text())["source"]

    assert source["tree_sha256"] == _release_tree_sha256(ROOT)


def test_checked_in_manifest_binds_the_approved_v0_1_2_release() -> None:
    source = json.loads(MANIFEST.read_text())["source"]

    assert source["version"] == "0.1.2"
    assert source["status"] == "approved"
    assert source["url"] == (
        "https://github.com/runbear-io/AutoBrain/releases/download/v0.1.2/autobrain-0.1.2.tar.gz"
    )
    assert source["reviewed_commit"] == "95c3336e06e40dd033a22a5d620cf05ee0be0860"
    assert source["tree_sha256"] == _release_tree_sha256(ROOT)
    assert len(source["sha256"]) == 64
    assert source["sha256"] != "0" * 64


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"status": "approved", "reviewed_commit": None}, "approved source requires"),
        ({"status": "approved", "reviewed_commit": "a" * 40, "sha256": "0" * 64}, "non-zero"),
        (
            {
                "status": "approved",
                "reviewed_commit": "a" * 40,
                "sha256": "a" * 64,
                "url": "https://example.test/UNAPPROVED/source.tar.gz",
            },
            "UNAPPROVED",
        ),
        ({"status": "prepared", "reviewed_commit": "a" * 40}, "prepared source must not"),
        (
            {
                "status": "prepared",
                "sha256": "a" * 64,
                "reviewed_commit": None,
                "url": "https://example.test/UNAPPROVED/source.tar.gz",
            },
            "prepared source SHA-256",
        ),
    ],
)
def test_manifest_source_state_is_fail_closed(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    data = json.loads(MANIFEST.read_text())
    data["source"].update(updates)

    with pytest.raises(FormulaParseError, match=message):
        load_formula_manifest(_write_manifest(tmp_path, data))


def test_approved_archive_rejects_unreviewed_source_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "extra-source.tar.gz"
    tar_payload = io.BytesIO()
    with tarfile.open(fileobj=tar_payload, mode="w") as archive:
        for source in _release_paths(ROOT):
            payload = source.read_bytes()
            info = tarfile.TarInfo(f"AutoBrain-release/{source.relative_to(ROOT).as_posix()}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        payload = b"print('unreviewed')\n"
        info = tarfile.TarInfo("AutoBrain-release/src/autobrain/unreviewed.py")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with (
        archive_path.open("wb") as destination,
        gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as compressed,
    ):
        compressed.write(tar_payload.getvalue())
    manifest_path = _approved_manifest(tmp_path, archive_path)

    with pytest.raises(ReleaseSourceError, match="unreviewed packaged source file"):
        _raw_generate_formula(
            load_formula_manifest(manifest_path),
            uv_lock_path=ROOT / "uv.lock",
            pyproject_path=ROOT / "pyproject.toml",
            source_archive_path=archive_path,
        )


@pytest.mark.parametrize("member_name", ["../escape", "/absolute", "root/../../escape"])
def test_approved_archive_rejects_unsafe_member_paths(tmp_path: Path, member_name: str) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    tar_payload = io.BytesIO()
    with tarfile.open(fileobj=tar_payload, mode="w") as archive:
        payload = b"unsafe"
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with (
        archive_path.open("wb") as destination,
        gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as compressed,
    ):
        compressed.write(tar_payload.getvalue())
    manifest_path = _approved_manifest(tmp_path, archive_path)

    with pytest.raises(ReleaseSourceError, match="unsafe member"):
        _raw_generate_formula(
            load_formula_manifest(manifest_path),
            uv_lock_path=ROOT / "uv.lock",
            pyproject_path=ROOT / "pyproject.toml",
            source_archive_path=archive_path,
        )


def test_generator_rejects_missing_macos_marked_runtime_dependency(tmp_path: Path) -> None:
    data = json.loads(MANIFEST.read_text())
    data["resources"] = [
        resource for resource in data["resources"] if resource["name"] != "uvicorn"
    ]

    with pytest.raises(FormulaParseError, match=r"missing target runtime resources.*uvicorn"):
        generate_formula(
            load_formula_manifest(_write_manifest(tmp_path, data)),
            uv_lock_path=ROOT / "uv.lock",
            pyproject_path=ROOT / "pyproject.toml",
        )


def test_generator_rejects_omitted_synthetic_macos_marked_resource(tmp_path: Path) -> None:
    lock_bytes = (
        (ROOT / "uv.lock")
        .read_bytes()
        .replace(
            b'{ name = "uvicorn", marker = "sys_platform != \'emscripten\'" },',
            b'{ name = "uvicorn", marker = "sys_platform != \'emscripten\'" },\n'
            b'    { name = "colorama", marker = "sys_platform == \'darwin\'" },',
        )
    )
    assert lock_bytes != (ROOT / "uv.lock").read_bytes()
    lock_path = tmp_path / "uv.lock"
    lock_path.write_bytes(lock_bytes)
    data = json.loads(MANIFEST.read_text())
    data["uv_lock_sha256"] = hashlib.sha256(lock_bytes).hexdigest()

    with pytest.raises(FormulaParseError, match=r"missing target runtime resources.*colorama"):
        generate_formula(
            load_formula_manifest(_write_manifest(tmp_path, data)),
            uv_lock_path=lock_path,
            pyproject_path=ROOT / "pyproject.toml",
        )


def test_generator_rejects_arbitrary_extra_runtime_resource(tmp_path: Path) -> None:
    data = json.loads(MANIFEST.read_text())
    package = next(
        package
        for package in tomllib.loads((ROOT / "uv.lock").read_text())["package"]
        if package["name"] == "colorama"
    )
    wheel = package["wheels"][0]
    data["resources"].append(
        {
            "name": "colorama",
            "url": wheel["url"],
            "sha256": wheel["hash"].removeprefix("sha256:"),
            "role": "runtime",
        }
    )

    with pytest.raises(FormulaParseError, match=r"extra target runtime resources.*colorama"):
        generate_formula(
            load_formula_manifest(_write_manifest(tmp_path, data)),
            uv_lock_path=ROOT / "uv.lock",
            pyproject_path=ROOT / "pyproject.toml",
        )


def test_generator_rejects_unknown_requested_extra(tmp_path: Path) -> None:
    lock_bytes = (
        (ROOT / "uv.lock")
        .read_bytes()
        .replace(
            b'{ name = "markdown-it-py", extra = ["linkify"] }',
            b'{ name = "markdown-it-py", extra = ["arbitrary"] }',
        )
    )
    assert lock_bytes != (ROOT / "uv.lock").read_bytes()
    lock_path = tmp_path / "uv.lock"
    lock_path.write_bytes(lock_bytes)
    data = json.loads(MANIFEST.read_text())
    data["uv_lock_sha256"] = hashlib.sha256(lock_bytes).hexdigest()

    with pytest.raises(FormulaParseError, match=r'unknown extra "markdown-it-py\[arbitrary\]"'):
        generate_formula(
            load_formula_manifest(_write_manifest(tmp_path, data)),
            uv_lock_path=lock_path,
            pyproject_path=ROOT / "pyproject.toml",
        )


def test_generator_rejects_linux_only_runtime_resource(tmp_path: Path) -> None:
    data = json.loads(MANIFEST.read_text())
    package = next(
        package
        for package in tomllib.loads((ROOT / "uv.lock").read_text())["package"]
        if package["name"] == "secretstorage"
    )
    wheel = package["wheels"][0]
    data["resources"].append(
        {
            "name": "secretstorage",
            "url": wheel["url"],
            "sha256": wheel["hash"].removeprefix("sha256:"),
            "role": "runtime",
        }
    )

    with pytest.raises(FormulaParseError, match=r"extra target runtime resources.*secretstorage"):
        generate_formula(
            load_formula_manifest(_write_manifest(tmp_path, data)),
            uv_lock_path=ROOT / "uv.lock",
            pyproject_path=ROOT / "pyproject.toml",
        )


def test_generated_formula_uses_current_homebrew_style() -> None:
    generated = generate_formula(
        load_formula_manifest(MANIFEST),
        uv_lock_path=ROOT / "uv.lock",
        pyproject_path=ROOT / "pyproject.toml",
    )

    assert 'desc "Compare LLM Wiki, Mem0 OSS, and GBrain on Slack and Notion"' in generated
    assert 'resource "hatchling" do' in generated
    assert 'url "https://' in generated
    assert 'sha256 "' in generated
    assert "depends_on arch: :arm64\ndepends_on :macos" not in generated
    assert '  depends_on arch: :arm64\n  depends_on :macos\n  depends_on "python@3.13"' in generated
    non_url_lines = [
        line for line in generated.splitlines() if not line.lstrip().startswith("url ")
    ]
    assert max(map(len, non_url_lines)) <= 118
    assert "'" not in generated


def test_generated_formula_installs_locked_build_backend_without_index_lookup() -> None:
    generated = generate_formula(
        load_formula_manifest(MANIFEST),
        uv_lock_path=ROOT / "uv.lock",
        pyproject_path=ROOT / "pyproject.toml",
    )

    assert generated.index('resource "hatchling"') < generated.index('resource "annotated-doc"')
    assert "venv.pip_install_and_link buildpath, build_isolation: false" in generated
    assert "venv.pip_install_and_link buildpath\n" not in generated


def test_generated_formula_uses_one_explicit_audited_route_for_all_native_wheels() -> None:
    manifest = load_formula_manifest(MANIFEST)
    generated = generate_formula(
        manifest,
        uv_lock_path=ROOT / "uv.lock",
        pyproject_path=ROOT / "pyproject.toml",
    )

    verification = verify_formula(generated)

    assert verification.valid
    assert verification.audited_platform_wheels == tuple(sorted(TARGET_PLATFORM_WHEELS))
    assert verification.platform_wheel_violations == ()
    assert "virtualenv_install_with_resources" not in generated
    assert "resource(name).cached_download" in generated


def test_published_tap_formula_matches_approved_source_metadata() -> None:
    source = json.loads(MANIFEST.read_text())["source"]
    formula = TAP_FORMULA.read_text(encoding="utf-8")

    assert f'  url "{source["url"]}"' in formula
    assert f'  sha256 "{source["sha256"]}"' in formula
    assert "NON-PUBLISHABLE" not in formula


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        (("description",), 'Injected "quote"'),
        (("description",), "Injected\nnewline"),
        (("description",), "Injected\x00control"),
        (("class_name",), "Autobrain; system('owned')"),
        (("source", "version"), "v0.1.1\nend"),
        (("python",), 'python@3.13"; system "owned"'),
        (("head", "branch"), "main\nend"),
        (("homepage",), "javascript:alert(1)"),
    ],
)
def test_manifest_rejects_unsafe_strings_at_boundary(
    tmp_path: Path, field_path: tuple[str, ...], value: str
) -> None:
    data = json.loads(MANIFEST.read_text())
    target = data
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(data))

    with pytest.raises(FormulaParseError):
        load_formula_manifest(manifest_path)


@pytest.mark.parametrize(
    "payload",
    [
        "#{system(1)}",
        "#@ivar",
        "#$global",
        "`whoami`",
        "%x(whoami)",
        "%Q{owned}",
        "\\#{system('owned')}",
        "\x1b[31mowned",
    ],
)
def test_every_manifest_string_rejects_ruby_control_syntax(tmp_path: Path, payload: str) -> None:
    original = json.loads(MANIFEST.read_text())
    paths: list[tuple[str | int, ...]] = []

    def collect(value: object, path: tuple[str | int, ...] = ()) -> None:
        if isinstance(value, str):
            paths.append(path)
        elif isinstance(value, dict):
            for key, child in cast(dict[str, object], value).items():
                collect(child, (*path, key))
        elif isinstance(value, list):
            for index, child in enumerate(cast(list[object], value)):
                collect(child, (*path, index))

    collect(original)
    for index, field_path in enumerate(paths):
        data = json.loads(MANIFEST.read_text())
        target: object = data
        for key in field_path[:-1]:
            if isinstance(key, str):
                target = cast(dict[str, object], target)[key]
            else:
                target = cast(list[object], target)[key]
        key = field_path[-1]
        if isinstance(key, str):
            mapping = cast(dict[str, object], target)
            mapping[key] = f"{mapping[key]}{payload}"
        else:
            items = cast(list[object], target)
            items[key] = f"{items[key]}{payload}"
        manifest_path = tmp_path / f"manifest-{index}.json"
        manifest_path.write_text(json.dumps(data))
        with pytest.raises(FormulaParseError):
            load_formula_manifest(manifest_path)


@pytest.mark.parametrize(
    "name",
    ["../cffi", "a/b", ".", "..", "CFFI", "cffi;end", "cffi name", "cffi\nend"],
)
def test_manifest_rejects_unsafe_resource_names(tmp_path: Path, name: str) -> None:
    data = json.loads(MANIFEST.read_text())
    data["resources"][3]["name"] = name
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(data))

    with pytest.raises(FormulaParseError, match="resource name"):
        load_formula_manifest(manifest_path)


def test_manifest_rejects_normalized_resource_name_collisions(tmp_path: Path) -> None:
    data = json.loads(MANIFEST.read_text())
    data["resources"][4]["name"] = "annotated_doc"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(data))

    with pytest.raises(FormulaParseError, match="collision"):
        load_formula_manifest(manifest_path)


def test_atomic_formula_write_uses_conventional_readable_mode(tmp_path: Path) -> None:
    output = tmp_path / "autobrain.rb"

    write_formula_atomic(output, b"formula\n")

    assert output.stat().st_mode & 0o777 == 0o644


def test_atomic_formula_write_preserves_previous_file_when_replace_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "autobrain.rb"
    output.write_bytes(b"previous formula\n")

    def interrupt_replace(_source: Path | str, _target: Path | str) -> None:
        raise InterruptedError("simulated interruption")

    monkeypatch.setattr("autobrain.release_formula.os.replace", interrupt_replace)

    with pytest.raises(InterruptedError, match="simulated interruption"):
        write_formula_atomic(output, b"replacement formula\n")

    assert output.read_bytes() == b"previous formula\n"
    assert list(tmp_path.iterdir()) == [output]


def test_generator_rejects_stale_lock_metadata(tmp_path: Path) -> None:
    stale_lock = tmp_path / "uv.lock"
    stale_lock.write_bytes((ROOT / "uv.lock").read_bytes() + b"\n# stale\n")

    with pytest.raises(FormulaParseError, match=r"uv\.lock SHA-256"):
        generate_formula(
            load_formula_manifest(MANIFEST),
            uv_lock_path=stale_lock,
            pyproject_path=ROOT / "pyproject.toml",
        )


@pytest.mark.parametrize(
    "spoof",
    [
        """
PLATFORM_WHEEL_RESOURCES = %w[native].freeze
resource "native" do
  url "https://example.test/native-1.0-cp313-cp313-macosx_11_0_arm64.whl"
  sha256 "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
end
# def install
#   wheel = resource(name).cached_download
#   venv.pip_install wheel
# end
def install
end
""",
        """
PLATFORM_WHEEL_RESOURCES = %w[native].freeze
resource "native" do
  url "https://example.test/native-1.0-cp313-cp313-macosx_11_0_arm64.whl"
  sha256 "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
end
def audit_spoof
  wheel = resource(name).cached_download
  venv.pip_install wheel
end
def install
  puts "not installing resources"
end
""",
        """
PLATFORM_WHEEL_RESOURCES = %w[native].freeze
resource "native" do
  url "https://example.test/native-1.0-cp313-cp313-macosx_11_0_arm64.whl"
  sha256 "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
end
def install
  if false
    resources.each do |res|
      name = res.name
      wheel = resource(name).cached_download
      venv.pip_install wheel
    end
  end
  venv.pip_install_and_link buildpath, build_isolation: false
end
""",
    ],
)
def test_comments_and_dead_methods_cannot_spoof_audited_install_route(spoof: str) -> None:
    assert verify_formula(spoof).platform_wheel_violations == ("native",)


def test_install_route_requires_project_build_isolation_disabled() -> None:
    generated = generate_formula(
        load_formula_manifest(MANIFEST),
        uv_lock_path=ROOT / "uv.lock",
        pyproject_path=ROOT / "pyproject.toml",
    )
    unsafe = generated.replace(
        "venv.pip_install_and_link buildpath, build_isolation: false",
        "venv.pip_install_and_link buildpath",
    )

    assert verify_formula(unsafe).platform_wheel_violations == tuple(sorted(TARGET_PLATFORM_WHEELS))


def test_parenthesized_virtualenv_install_is_detected() -> None:
    formula = """
resource "native" do
  url "https://example.test/native-1.0-cp313-cp313-macosx_11_0_arm64.whl"
end

def install
  virtualenv_install_with_resources(without: "other")
end
"""

    assert verify_formula(formula).platform_wheel_violations == ("native",)


def test_cli_reports_success_without_misleading_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_cli(FIXTURES / "valid_formula.rb") == 0
    output = capsys.readouterr().out
    assert output.endswith(
        "OK: formula Python resources use a Homebrew-compatible installation route\n"
    )
    assert "ERROR:" not in output


def test_cli_reports_every_platform_wheel_and_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_cli(FIXTURES / "autobrain-v0.1.1.rb") == 1
    output = capsys.readouterr().out
    assert "OK:" not in output
    assert "platform wheels routed through virtualenv_install_with_resources" in output
    for package in PUBLISHED_PLATFORM_WHEELS:
        assert package in output
