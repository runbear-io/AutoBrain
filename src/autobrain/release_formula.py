"""Deterministic validation for Python resources in a Homebrew formula."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import unquote, urlsplit

_RESOURCE_START = re.compile(r'^\s*resource\s+"(?P<name>[^"]+)"\s+do\s*$')
_URL = re.compile(r'^\s*url\s+"(?P<url>[^"]+)"\s*$')
_END = re.compile(r"^\s*end\s*$")
_VIRTUALENV_INSTALL = re.compile(r"^\s*virtualenv_install_with_resources(?:\s|\(|$)", re.MULTILINE)
_SDIST_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip")


class FormulaParseError(ValueError):
    """Raised when the formula resource declarations cannot be verified."""


class ArchiveKind(StrEnum):
    """Python distribution archive classes relevant to Homebrew installation."""

    UNIVERSAL_WHEEL = "universal-wheel"
    PLATFORM_WHEEL = "platform-wheel"
    SDIST = "sdist"


@dataclass(frozen=True)
class FormulaResource:
    """One named Homebrew resource and its classified Python archive."""

    name: str
    url: str
    archive_kind: ArchiveKind


@dataclass(frozen=True)
class FormulaVerification:
    """Parsed resources and any unsafe virtualenv installation routes."""

    resources: tuple[FormulaResource, ...]
    uses_virtualenv_install_with_resources: bool
    platform_wheel_violations: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.platform_wheel_violations


def classify_archive(url: str) -> ArchiveKind:
    """Classify a Python distribution URL from its standard archive filename."""
    filename = Path(unquote(urlsplit(url).path)).name
    if filename.endswith(".whl"):
        wheel_parts = filename.removesuffix(".whl").split("-")
        if len(wheel_parts) < 5:
            raise FormulaParseError(f"malformed wheel filename: {filename}")
        platform_tags = wheel_parts[-1].split(".")
        if platform_tags and all(tag == "any" for tag in platform_tags):
            return ArchiveKind.UNIVERSAL_WHEEL
        return ArchiveKind.PLATFORM_WHEEL
    if filename.endswith(_SDIST_SUFFIXES):
        return ArchiveKind.SDIST
    raise FormulaParseError(f"unsupported Python resource archive: {filename or url}")


def parse_formula(text: str) -> tuple[FormulaResource, ...]:
    """Parse top-level Homebrew resource blocks without evaluating Ruby."""
    resources: list[FormulaResource] = []
    seen_names: set[str] = set()
    current_name: str | None = None
    current_url: str | None = None

    for line_number, line in enumerate(text.splitlines(), start=1):
        resource_match = _RESOURCE_START.match(line)
        if current_name is None:
            if resource_match is not None:
                current_name = resource_match.group("name")
                current_url = None
            continue

        if resource_match is not None:
            raise FormulaParseError(f"nested resource block at line {line_number}")
        url_match = _URL.match(line)
        if url_match is not None:
            if current_url is not None:
                raise FormulaParseError(
                    f'resource "{current_name}" has multiple URLs at line {line_number}'
                )
            current_url = url_match.group("url")
            continue
        if _END.match(line) is None:
            continue
        if current_url is None:
            raise FormulaParseError(f'resource "{current_name}" has no URL')
        if current_name in seen_names:
            raise FormulaParseError(f'duplicate resource name: "{current_name}"')
        seen_names.add(current_name)
        resources.append(
            FormulaResource(
                name=current_name,
                url=current_url,
                archive_kind=classify_archive(current_url),
            )
        )
        current_name = None
        current_url = None

    if current_name is not None:
        raise FormulaParseError(f'unterminated resource block: "{current_name}"')
    if not resources:
        raise FormulaParseError("formula has no Python resources")
    return tuple(resources)


def verify_formula(text: str) -> FormulaVerification:
    """Verify the formula's Python resource installation strategy."""
    resources = parse_formula(text)
    uses_virtualenv_resources = _VIRTUALENV_INSTALL.search(text) is not None
    violations = (
        tuple(
            sorted(
                resource.name
                for resource in resources
                if resource.archive_kind is ArchiveKind.PLATFORM_WHEEL
            )
        )
        if uses_virtualenv_resources
        else ()
    )
    return FormulaVerification(
        resources=resources,
        uses_virtualenv_install_with_resources=uses_virtualenv_resources,
        platform_wheel_violations=violations,
    )


def _format_names(verification: FormulaVerification, kind: ArchiveKind) -> str:
    names = [resource.name for resource in verification.resources if resource.archive_kind is kind]
    return ", ".join(names) if names else "(none)"


def run_cli(formula_path: Path) -> int:
    """Validate one formula and print a deterministic human-readable report."""
    try:
        verification = verify_formula(formula_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, FormulaParseError) as error:
        print(f"ERROR: {error}")
        return 2

    print(f"formula: {formula_path}")
    strategy = (
        "virtualenv_install_with_resources"
        if verification.uses_virtualenv_install_with_resources
        else "custom"
    )
    print(f"install-strategy: {strategy}")
    for kind in ArchiveKind:
        print(f"{kind.value}: {_format_names(verification, kind)}")
    if verification.platform_wheel_violations:
        print(
            "ERROR: platform wheels routed through virtualenv_install_with_resources: "
            + ", ".join(verification.platform_wheel_violations)
        )
        return 1
    print("OK: formula Python resources use a Homebrew-compatible installation route")
    return 0


def main() -> None:
    """Console entry point for the release formula verifier."""
    parser = argparse.ArgumentParser(
        description="Verify Homebrew Python resource archive installation compatibility."
    )
    parser.add_argument("formula", type=Path, help="path to Formula/autobrain.rb")
    arguments = parser.parse_args()
    raise SystemExit(run_cli(arguments.formula))


if __name__ == "__main__":
    main()
