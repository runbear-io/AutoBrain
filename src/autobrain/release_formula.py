"""Deterministic generation and validation for the Homebrew formula."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

from packaging.markers import Marker

_RESOURCE_START = re.compile(r"^\s*resource\s+(?P<quote>['\"])(?P<name>[^'\"]+)(?P=quote)\s+do\s*$")
_URL = re.compile(r"^\s*url\s+(?P<quote>['\"])(?P<url>[^'\"]+)(?P=quote)\s*$")
_SHA256 = re.compile(r"^\s*sha256\s+(?P<quote>['\"])(?P<sha256>[^'\"]+)(?P=quote)\s*$")
_END = re.compile(r"^\s*end\s*$")
_VIRTUALENV_INSTALL = re.compile(r"^\s*virtualenv_install_with_resources(?:\s|\(|$)", re.MULTILINE)
_AUDITED_WHEELS = re.compile(
    r"^\s*PLATFORM_WHEEL_RESOURCES\s*=\s*%w\[\s*(?P<names>.*?)\s*\]\.freeze\s*$",
    re.MULTILINE | re.DOTALL,
)
_SDIST_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip")
_SHA256_LENGTH = 64
_RESOURCE_NAME = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_CLASS_NAME = re.compile(r"^[A-Z][A-Za-z0-9]*$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[a-z0-9.-]+)?$")
_PYTHON_FORMULA = re.compile(r"^python@[0-9]+\.[0-9]+$")
_BRANCH = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._/-]*[A-Za-z0-9])?$")
_LICENSE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-.+]*$")
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .,()&+:/_-]*$")
_RUBY_CONTROL = re.compile(r"#(?:\{|@|\$)|`|%[qQwWiIxrs]?(?=[^A-Za-z0-9\s%])")
_BUILD_RESOURCES = (
    "hatchling",
    "packaging",
    "pathspec",
    "pluggy",
    "trove-classifiers",
)


class FormulaParseError(ValueError):
    """Raised when formula generation inputs or declarations cannot be verified."""


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
    sha256: str
    archive_kind: ArchiveKind


@dataclass(frozen=True)
class FormulaVerification:
    """Parsed resources and compatibility of their installation routes."""

    resources: tuple[FormulaResource, ...]
    uses_virtualenv_install_with_resources: bool
    audited_platform_wheels: tuple[str, ...]
    platform_wheel_violations: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.platform_wheel_violations


@dataclass(frozen=True)
class FormulaSource:
    version: str
    url: str
    sha256: str


@dataclass(frozen=True)
class FormulaHead:
    url: str
    branch: str


@dataclass(frozen=True)
class FormulaManifestResource:
    name: str
    url: str
    sha256: str
    role: str


@dataclass(frozen=True)
class FormulaManifest:
    schema_version: int
    class_name: str
    description: str
    homepage: str
    license: str
    source: FormulaSource
    head: FormulaHead
    python: str
    uv_lock_sha256: str
    candidate_pins_sha256: str
    platform_wheel_resources: tuple[str, ...]
    resources: tuple[FormulaManifestResource, ...]


def _required_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise FormulaParseError(f'manifest field "{key}" must be a non-empty string')
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise FormulaParseError(f'manifest field "{key}" contains control characters')
    if _RUBY_CONTROL.search(value) is not None or "\\" in value:
        raise FormulaParseError(
            f'manifest field "{key}" contains Ruby interpolation or literal syntax'
        )
    return value


def _matching_string(
    mapping: dict[str, Any], key: str, pattern: re.Pattern[str], description: str
) -> str:
    value = _required_string(mapping, key)
    if pattern.fullmatch(value) is None:
        raise FormulaParseError(f'manifest field "{key}" must be {description}')
    return value


def _https_url(mapping: dict[str, Any], key: str) -> str:
    value = _required_string(mapping, key)
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise FormulaParseError(f'manifest field "{key}" must be an absolute HTTPS URL')
    if any(character in value for character in ('"', "'", "\\")):
        raise FormulaParseError(f'manifest field "{key}" contains unsafe URL characters')
    return value


def _branch_name(mapping: dict[str, Any]) -> str:
    value = _matching_string(mapping, "branch", _BRANCH, "a Git branch name")
    if any(segment in {".", ".."} for segment in value.split("/")):
        raise FormulaParseError('manifest field "branch" contains a dot segment')
    return value


def _resource_name(mapping: dict[str, Any]) -> str:
    try:
        return _matching_string(
            mapping,
            "name",
            _RESOURCE_NAME,
            "a normalized lowercase Python package name",
        )
    except FormulaParseError as error:
        raise FormulaParseError(f"unsafe resource name: {error}") from error


def _normalized_resource_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name)


def _ruby_string(value: str) -> str:
    """Double-quote a validated manifest value without Ruby interpolation."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _required_sha256(mapping: dict[str, Any], key: str) -> str:
    value = _required_string(mapping, key)
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise FormulaParseError(f'manifest field "{key}" must be a lowercase SHA-256')
    return value


def load_formula_manifest(path: Path) -> FormulaManifest:
    """Load the checked-in formula generation lock with strict field validation."""
    try:
        raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormulaParseError(f"cannot read formula manifest: {error}") from error
    if not isinstance(raw, dict):
        raise FormulaParseError("formula manifest must be a JSON object")
    data = cast(dict[str, Any], raw)
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise FormulaParseError("formula manifest schema_version must be integer 1")

    source_raw = data.get("source")
    head_raw = data.get("head")
    resources_raw = data.get("resources")
    platform_raw = data.get("platform_wheel_resources")
    if not isinstance(source_raw, dict) or not isinstance(head_raw, dict):
        raise FormulaParseError("manifest source and head must be objects")
    if not isinstance(resources_raw, list) or not resources_raw:
        raise FormulaParseError("manifest resources must be a non-empty array")
    resource_items = cast(list[object], resources_raw)
    if not isinstance(platform_raw, list):
        raise FormulaParseError("manifest platform_wheel_resources must be a string array")
    platform_items = cast(list[object], platform_raw)
    if not all(isinstance(name, str) for name in platform_items):
        raise FormulaParseError("manifest platform_wheel_resources must be a string array")

    resources: list[FormulaManifestResource] = []
    seen_names: set[str] = set()
    seen_normalized_names: set[str] = set()
    for index, raw_resource in enumerate(resource_items):
        if not isinstance(raw_resource, dict):
            raise FormulaParseError(f"manifest resource {index} must be an object")
        resource = cast(dict[str, Any], raw_resource)
        name = _resource_name(resource)
        if name in seen_names:
            raise FormulaParseError(f'duplicate manifest resource name: "{name}"')
        normalized_name = _normalized_resource_name(name)
        if normalized_name in seen_normalized_names:
            raise FormulaParseError(f'normalized resource name collision: "{name}"')
        seen_names.add(name)
        seen_normalized_names.add(normalized_name)
        role = _required_string(resource, "role")
        if role not in {"build", "runtime"}:
            raise FormulaParseError(f'unsupported role for resource "{name}": {role}')
        resources.append(
            FormulaManifestResource(
                name=name,
                url=_https_url(resource, "url"),
                sha256=_required_sha256(resource, "sha256"),
                role=role,
            )
        )

    build_names = tuple(resource.name for resource in resources if resource.role == "build")
    if build_names != _BUILD_RESOURCES:
        raise FormulaParseError(
            "build resources must be locked in order: " + ", ".join(_BUILD_RESOURCES)
        )
    if any(resource.role == "build" for resource in resources[len(_BUILD_RESOURCES) :]):
        raise FormulaParseError("build resources must precede runtime resources")

    platform_names = tuple(cast(list[str], platform_items))
    for name in platform_names:
        if _RESOURCE_NAME.fullmatch(name) is None:
            raise FormulaParseError(
                "platform_wheel_resources entries must use normalized package names"
            )
    if len(platform_names) != len(set(platform_names)):
        raise FormulaParseError("manifest platform_wheel_resources contains duplicates")
    missing = sorted(set(platform_names) - seen_names)
    if missing:
        raise FormulaParseError("platform wheels missing from resources: " + ", ".join(missing))

    source = cast(dict[str, Any], source_raw)
    head = cast(dict[str, Any], head_raw)
    return FormulaManifest(
        schema_version=1,
        class_name=_matching_string(data, "class_name", _CLASS_NAME, "a Ruby class identifier"),
        description=_matching_string(data, "description", _SAFE_TEXT, "single-line release text"),
        homepage=_https_url(data, "homepage"),
        license=_matching_string(data, "license", _LICENSE, "an SPDX-style identifier"),
        source=FormulaSource(
            version=_matching_string(source, "version", _VERSION, "a release version identifier"),
            url=_https_url(source, "url"),
            sha256=_required_sha256(source, "sha256"),
        ),
        head=FormulaHead(
            url=_https_url(head, "url"),
            branch=_branch_name(head),
        ),
        python=_matching_string(
            data, "python", _PYTHON_FORMULA, "a versioned Homebrew Python formula"
        ),
        uv_lock_sha256=_required_sha256(data, "uv_lock_sha256"),
        candidate_pins_sha256=_required_sha256(data, "candidate_pins_sha256"),
        platform_wheel_resources=platform_names,
        resources=tuple(resources),
    )


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
    current_sha256: str | None = None

    for line_number, line in enumerate(text.splitlines(), start=1):
        resource_match = _RESOURCE_START.match(line)
        if current_name is None:
            if resource_match is not None:
                current_name = resource_match.group("name")
                current_url = None
                current_sha256 = None
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
        sha_match = _SHA256.match(line)
        if sha_match is not None:
            if current_sha256 is not None:
                raise FormulaParseError(
                    f'resource "{current_name}" has multiple SHA-256 values at line {line_number}'
                )
            current_sha256 = sha_match.group("sha256")
            continue
        if _END.match(line) is None:
            continue
        if current_url is None:
            raise FormulaParseError(f'resource "{current_name}" has no URL')
        if _RESOURCE_NAME.fullmatch(current_name) is None:
            raise FormulaParseError(f'unsafe resource name: "{current_name}"')
        if current_name in seen_names:
            raise FormulaParseError(f'duplicate resource name: "{current_name}"')
        seen_names.add(current_name)
        resources.append(
            FormulaResource(
                name=current_name,
                url=current_url,
                sha256=current_sha256 or "",
                archive_kind=classify_archive(current_url),
            )
        )
        current_name = None
        current_url = None
        current_sha256 = None

    if current_name is not None:
        raise FormulaParseError(f'unterminated resource block: "{current_name}"')
    if not resources:
        raise FormulaParseError("formula has no Python resources")
    return tuple(resources)


def _code_without_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _extract_install_method(text: str) -> tuple[str, ...]:
    lines = _code_without_comments(text).splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.strip() == "def install"),
        None,
    )
    if start is None:
        return ()
    body: list[str] = []
    depth = 1
    block_starts = ("def ", "if ", "unless ", "case ", "begin", "while ", "until ", "for ")
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped == "end":
            depth -= 1
            if depth == 0:
                return tuple(body)
            body.append(stripped)
            continue
        body.append(stripped)
        if stripped.startswith(block_starts) or stripped.endswith(" do") or " do |" in stripped:
            depth += 1
    raise FormulaParseError("unterminated install method")


def _parse_audited_wheels(text: str) -> tuple[str, ...]:
    match = _AUDITED_WHEELS.search(_code_without_comments(text))
    if match is None:
        return ()
    names = tuple(match.group("names").split())
    if len(names) != len(set(names)):
        raise FormulaParseError("PLATFORM_WHEEL_RESOURCES contains duplicates")
    return tuple(sorted(names))


def verify_formula(text: str) -> FormulaVerification:
    """Verify the formula's Python resource installation strategy."""
    resources = parse_formula(text)
    uses_virtualenv_resources = _VIRTUALENV_INSTALL.search(text) is not None
    platform_names = {
        resource.name
        for resource in resources
        if resource.archive_kind is ArchiveKind.PLATFORM_WHEEL
    }
    audited_names = set(_parse_audited_wheels(text))
    install_lines = _extract_install_method(text)
    has_explicit_route = (
        len(install_lines) == 7
        and install_lines[0].startswith("venv = virtualenv_create(libexec, ")
        and install_lines[1:6]
        == (
            "resources.each do |res|",
            "name = res.name",
            "wheel = resource(name).cached_download",
            "venv.pip_install wheel",
            "end",
        )
    )
    has_offline_project_build = (
        len(install_lines) == 7
        and install_lines[6] == "venv.pip_install_and_link buildpath, build_isolation: false"
    )
    violations = (
        set(platform_names) if uses_virtualenv_resources else platform_names - audited_names
    )
    if not has_explicit_route or not has_offline_project_build:
        violations.update(platform_names)
    violations.update(audited_names - platform_names)
    return FormulaVerification(
        resources=resources,
        uses_virtualenv_install_with_resources=uses_virtualenv_resources,
        audited_platform_wheels=tuple(sorted(audited_names)),
        platform_wheel_violations=tuple(sorted(violations)),
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _project_version(pyproject_path: Path) -> str:
    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        version = pyproject["project"]["version"]
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise FormulaParseError(f"cannot read project version: {error}") from error
    if not isinstance(version, str):
        raise FormulaParseError("project version must be a string")
    return version


def _validate_locked_resources(lock_bytes: bytes, manifest: FormulaManifest) -> None:
    try:
        lock = tomllib.loads(lock_bytes.decode())
        packages = cast(list[object], lock["package"])
    except (UnicodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise FormulaParseError(f"cannot parse uv.lock packages: {error}") from error

    locked_artifacts: dict[str, set[tuple[str, str]]] = {}
    locked_packages: dict[str, dict[str, Any]] = {}
    for raw_package in packages:
        if not isinstance(raw_package, dict):
            raise FormulaParseError("uv.lock package entries must be tables")
        package = cast(dict[str, Any], raw_package)
        name = package.get("name")
        if not isinstance(name, str):
            raise FormulaParseError("uv.lock package name must be a string")
        artifacts: set[tuple[str, str]] = set()
        raw_wheels = package.get("wheels", [])
        if not isinstance(raw_wheels, list):
            raise FormulaParseError(f'uv.lock wheels for "{name}" must be an array')
        for raw_wheel in cast(list[object], raw_wheels):
            if not isinstance(raw_wheel, dict):
                raise FormulaParseError(f'uv.lock wheel for "{name}" must be a table')
            wheel = cast(dict[str, Any], raw_wheel)
            url = wheel.get("url")
            digest = wheel.get("hash")
            if isinstance(url, str) and isinstance(digest, str) and digest.startswith("sha256:"):
                artifacts.add((url, digest.removeprefix("sha256:")))
        locked_artifacts[name] = artifacts
        locked_packages[name] = package

    marker_environment = {
        "implementation_name": "cpython",
        "implementation_version": "3.13.0",
        "os_name": "posix",
        "platform_machine": "arm64",
        "platform_python_implementation": "CPython",
        "platform_release": "",
        "platform_system": "Darwin",
        "platform_version": "",
        "python_full_version": "3.13.0",
        "python_version": "3.13",
        "sys_platform": "darwin",
    }

    def dependency_target(raw_dependency: object, owner: str) -> tuple[str, frozenset[str]] | None:
        if not isinstance(raw_dependency, dict):
            raise FormulaParseError(f'uv.lock dependency for "{owner}" must be a table')
        dependency = cast(dict[str, Any], raw_dependency)
        dependency_name = dependency.get("name")
        if not isinstance(dependency_name, str):
            raise FormulaParseError(f'uv.lock dependency name for "{owner}" must be a string')
        marker = dependency.get("marker")
        if marker is not None:
            if not isinstance(marker, str):
                raise FormulaParseError(f'uv.lock marker for "{owner}" must be a string')
            try:
                if not Marker(marker).evaluate(environment=marker_environment):
                    return None
            except Exception as error:
                raise FormulaParseError(
                    f'cannot evaluate uv.lock marker for "{owner}": {error}'
                ) from error
        raw_extras = dependency.get("extra", [])
        if not isinstance(raw_extras, list) or not all(
            isinstance(extra, str) for extra in cast(list[object], raw_extras)
        ):
            raise FormulaParseError(f'uv.lock extras for "{owner}" must be a string array')
        return dependency_name, frozenset(cast(list[str], raw_extras))

    required_runtime: set[str] = set()
    visited: set[tuple[str, frozenset[str]]] = set()
    pending: list[tuple[str, frozenset[str]]] = [("autobrain", frozenset())]
    while pending:
        name, extras = pending.pop()
        state = (name, extras)
        if state in visited:
            continue
        visited.add(state)
        package = locked_packages.get(name)
        if package is None:
            raise FormulaParseError(f'uv.lock is missing reachable package "{name}"')
        if name != "autobrain":
            required_runtime.add(name)
        raw_dependencies = package.get("dependencies", [])
        if not isinstance(raw_dependencies, list):
            raise FormulaParseError(f'uv.lock dependencies for "{name}" must be an array')
        dependency_items = list(cast(list[object], raw_dependencies))
        raw_optional = package.get("optional-dependencies", {})
        if not isinstance(raw_optional, dict):
            raise FormulaParseError(f'uv.lock optional dependencies for "{name}" must be a table')
        optional = cast(dict[str, object], raw_optional)
        for extra in extras:
            if extra not in optional:
                raise FormulaParseError(
                    f'uv.lock dependency requests unknown extra "{name}[{extra}]"'
                )
            raw_extra_dependencies = optional[extra]
            if not isinstance(raw_extra_dependencies, list):
                raise FormulaParseError(
                    f'uv.lock optional dependency group "{name}[{extra}]" must be an array'
                )
            dependency_items.extend(cast(list[object], raw_extra_dependencies))
        for raw_dependency in dependency_items:
            target = dependency_target(raw_dependency, name)
            if target is not None:
                pending.append(target)

    manifest_names = {resource.name for resource in manifest.resources}
    required_names = required_runtime | set(_BUILD_RESOURCES)
    missing_runtime = sorted(required_names - manifest_names)
    extra_runtime = sorted(manifest_names - required_names)
    if missing_runtime:
        raise FormulaParseError(
            "manifest is missing target runtime resources: " + ", ".join(missing_runtime)
        )
    if extra_runtime:
        raise FormulaParseError(
            "manifest has extra target runtime resources: " + ", ".join(extra_runtime)
        )

    for resource in manifest.resources:
        if resource.name not in required_runtime:
            continue
        if (resource.url, resource.sha256) not in locked_artifacts.get(resource.name, set()):
            raise FormulaParseError(
                f'resource "{resource.name}" URL/SHA is not an exact uv.lock wheel artifact'
            )


def _validate_generation_inputs(
    manifest: FormulaManifest,
    uv_lock_path: Path,
    pyproject_path: Path,
) -> None:
    try:
        lock_bytes = uv_lock_path.read_bytes()
    except OSError as error:
        raise FormulaParseError(f"cannot read uv.lock: {error}") from error
    actual_lock_sha = _sha256(lock_bytes)
    if actual_lock_sha != manifest.uv_lock_sha256:
        raise FormulaParseError(
            f"uv.lock SHA-256 mismatch: expected {manifest.uv_lock_sha256}, got {actual_lock_sha}"
        )
    _validate_locked_resources(lock_bytes, manifest)
    if _project_version(pyproject_path) != manifest.source.version:
        raise FormulaParseError(
            f"project version does not match formula version {manifest.source.version}"
        )
    candidate_pins_path = pyproject_path.with_name("candidate-pins.json")
    try:
        candidate_sha = _sha256(candidate_pins_path.read_bytes())
    except OSError as error:
        raise FormulaParseError(f"cannot read candidate-pins.json: {error}") from error
    if candidate_sha != manifest.candidate_pins_sha256:
        raise FormulaParseError(
            "candidate-pins.json SHA-256 mismatch: "
            f"expected {manifest.candidate_pins_sha256}, got {candidate_sha}"
        )


def generate_formula(
    manifest: FormulaManifest,
    *,
    uv_lock_path: Path,
    pyproject_path: Path,
) -> str:
    """Render a byte-stable formula from the checked-in release lock."""
    _validate_generation_inputs(manifest, uv_lock_path, pyproject_path)
    lines = [
        "# This file is generated by autobrain-generate-formula. Do not edit.",
        f"# uv.lock sha256: {manifest.uv_lock_sha256}",
        f"# candidate-pins.json sha256: {manifest.candidate_pins_sha256}",
        f"class {manifest.class_name} < Formula",
        "  include Language::Python::Virtualenv",
        "",
        f"  desc {_ruby_string(manifest.description)}",
        f"  homepage {_ruby_string(manifest.homepage)}",
        f"  url {_ruby_string(manifest.source.url)}",
        f"  sha256 {_ruby_string(manifest.source.sha256)}",
        f"  license {_ruby_string(manifest.license)}",
        f"  head {_ruby_string(manifest.head.url)}, branch: {_ruby_string(manifest.head.branch)}",
        "",
        "  depends_on arch: :arm64",
        "  depends_on :macos",
        f"  depends_on {_ruby_string(manifest.python)}",
        "",
        "  PLATFORM_WHEEL_RESOURCES = %w[",
    ]
    lines.extend(f"    {name}" for name in sorted(manifest.platform_wheel_resources))
    lines.extend(["  ].freeze", ""])

    previous_role: str | None = None
    for resource in manifest.resources:
        if previous_role is not None and resource.role != previous_role:
            lines.append("")
        lines.extend(
            [
                f"  resource {_ruby_string(resource.name)} do",
                f"    url {_ruby_string(resource.url)}",
                f"    sha256 {_ruby_string(resource.sha256)}",
                "  end",
            ]
        )
        previous_role = resource.role

    imports = [
        "cffi",
        "cryptography",
        "grpc",
        "jiter",
        "markupsafe",
        "numpy",
        "pydantic_core",
        "rpds",
    ]
    lines.extend(
        [
            "",
            "  def install",
            "    venv = virtualenv_create("
            f"libexec, {_ruby_string(manifest.python.replace('@', ''))})",
            "    resources.each do |res|",
            "      name = res.name",
            "      wheel = resource(name).cached_download",
            "      venv.pip_install wheel",
            "    end",
            "    venv.pip_install_and_link buildpath, build_isolation: false",
            "  end",
            "",
            "  test do",
            '    assert_match "Usage", shell_output(bin/"autobrain --help")',
            '    assert_match "checks", shell_output(bin/"autobrain doctor --json")',
            '    system libexec/"bin/python", "-c",',
            '           "import ' + ", ".join(imports) + '"',
            "  end",
            "end",
            "",
        ]
    )
    generated = "\n".join(lines)
    verification = verify_formula(generated)
    if not verification.valid:
        raise FormulaParseError(
            "generated formula has unsafe platform-wheel routes: "
            + ", ".join(verification.platform_wheel_violations)
        )
    return generated


def verify_downloads(manifest: FormulaManifest, *, timeout: float = 60.0) -> None:
    """Fetch every release artifact and reject the first URL/SHA mismatch."""
    artifacts = [("source", manifest.source.url, manifest.source.sha256)] + [
        (resource.name, resource.url, resource.sha256) for resource in manifest.resources
    ]
    for name, url, expected_sha in artifacts:
        digest = hashlib.sha256()
        try:
            with urlopen(url, timeout=timeout) as response:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as error:
            raise FormulaParseError(f'cannot download "{name}": {error}') from error
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha:
            raise FormulaParseError(
                f'SHA-256 mismatch for "{name}": expected {expected_sha}, got {actual_sha}'
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
        else "explicit-audited-wheel-route"
    )
    print(f"install-strategy: {strategy}")
    for kind in ArchiveKind:
        print(f"{kind.value}: {_format_names(verification, kind)}")
    if verification.platform_wheel_violations:
        print(
            "ERROR: platform wheels routed through virtualenv_install_with_resources "
            "or missing an explicit audited install route: "
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


def write_formula_atomic(path: Path, content: bytes) -> None:
    """Durably replace a formula without exposing a partially written target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def generate_main() -> None:
    """Console entry point for deterministic formula generation and supply-chain checks."""
    parser = argparse.ArgumentParser(description="Generate the locked AutoBrain Homebrew formula.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--uv-lock", type=Path, required=True)
    parser.add_argument("--pyproject", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="fail unless output is byte-identical")
    parser.add_argument(
        "--check-downloads",
        action="store_true",
        help="download every source/resource and verify its locked SHA-256",
    )
    arguments = parser.parse_args()
    try:
        manifest = load_formula_manifest(arguments.manifest)
        generated = generate_formula(
            manifest,
            uv_lock_path=arguments.uv_lock,
            pyproject_path=arguments.pyproject,
        )
        if arguments.check_downloads:
            verify_downloads(manifest)
        if arguments.check:
            if arguments.output.read_bytes() != generated.encode():
                raise FormulaParseError(f"generated formula is stale: {arguments.output}")
        else:
            write_formula_atomic(arguments.output, generated.encode())
    except (OSError, UnicodeError, FormulaParseError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1) from None
    print(f"OK: formula generation contract verified: {arguments.output}")


if __name__ == "__main__":
    main()
