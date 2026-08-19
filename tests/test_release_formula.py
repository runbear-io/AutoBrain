from __future__ import annotations

from pathlib import Path

import pytest

from autobrain.release_formula import (
    ArchiveKind,
    FormulaParseError,
    classify_archive,
    parse_formula,
    run_cli,
    verify_formula,
)

FIXTURES = Path(__file__).parent / "fixtures" / "release"
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


def test_custom_install_route_can_handle_platform_wheels_explicitly() -> None:
    formula = """
resource "native" do
  url "https://example.test/native-1.0-cp313-cp313-macosx_11_0_arm64.whl"
end

def install
  resource("native").stage { virtualenv.pip_install Pathname.pwd.glob("*.whl") }
end
"""

    verification = verify_formula(formula)

    assert verification.valid
    assert verification.platform_wheel_violations == ()


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
