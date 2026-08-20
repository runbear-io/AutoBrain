from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from autobrain.release_formula import (
    ArchiveKind,
    FormulaParseError,
    classify_archive,
    generate_formula,
    load_formula_manifest,
    parse_formula,
    run_cli,
    verify_formula,
    write_formula_atomic,
)

ROOT = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "release"
MANIFEST = ROOT / "release" / "homebrew-formula.json"
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


def test_generated_formula_installs_locked_build_backend_without_index_lookup() -> None:
    generated = generate_formula(
        load_formula_manifest(MANIFEST),
        uv_lock_path=ROOT / "uv.lock",
        pyproject_path=ROOT / "pyproject.toml",
    )

    assert generated.index("resource 'hatchling'") < generated.index("resource 'annotated-doc'")
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
    assert verification.audited_platform_wheels == tuple(sorted(PUBLISHED_PLATFORM_WHEELS))
    assert verification.platform_wheel_violations == ()
    assert "virtualenv_install_with_resources" not in generated
    assert "resource(name).cached_download" in generated


def test_generator_is_byte_idempotent_and_matches_tap_formula() -> None:
    manifest = load_formula_manifest(MANIFEST)
    generated_once = generate_formula(
        manifest,
        uv_lock_path=ROOT / "uv.lock",
        pyproject_path=ROOT / "pyproject.toml",
    )
    generated_twice = generate_formula(
        manifest,
        uv_lock_path=ROOT / "uv.lock",
        pyproject_path=ROOT / "pyproject.toml",
    )

    assert generated_once.encode() == generated_twice.encode()


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

    assert verify_formula(unsafe).platform_wheel_violations == tuple(
        sorted(PUBLISHED_PLATFORM_WHEELS)
    )


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
