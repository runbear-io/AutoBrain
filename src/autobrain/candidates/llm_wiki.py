"""Pinned, run-local adapter for llm-wiki-compiler 1.1.0."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, cast
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

from autobrain.models import (
    BenchmarkCase,
    CandidateId,
    CandidatePin,
    CandidateQuery,
    Holdout,
    NormalizedDocument,
    Status,
)

APPROVED_DISTRIBUTION: Final = "llm-wiki-compiler"
APPROVED_VERSION: Final = "1.1.0"
APPROVED_COMMIT: Final = "3e17bcfe8b50f24c14c6bcda0cb9224d94fd8206"
APPROVED_REPOSITORY: Final = "https://github.com/atomicstrata/llm-wiki-compiler"
APPROVED_LICENSE: Final = "MIT"
APPROVED_LICENSE_SHA256: Final = "c6bc84c8d6f6d21fb78a51d1b356746c2401934a297a83967914dd66fa4b85cc"
# Reproducible `npm ci --ignore-scripts && npm run build` tree digest at APPROVED_COMMIT.
APPROVED_DIST_TREE_SHA256: Final = (
    "e0e12f625f232c6407a4c0a5a83be89b75bf39711fc0a9535b487d603fd9e804"
)
MAX_SOURCE_CHARS: Final = 100_000
_DRIVER_NAME: Final = "autobrain-driver.mjs"
_PIN_MARKER: Final = "autobrain-pin.json"
_CITATION = re.compile(r"\^\[([^\]#:]+\.md)(?:(?::|#L)[^\]]+)?\]")
_URL = re.compile(r"https?://[^\s<>'\"\\]+")
_SENSITIVE_KEY = re.compile(r"(?:secret|token|password|api[_-]?key|authorization|credential)", re.I)
_SEAL_NAME: Final = "workspace-seal.json"


class ToolCacheError(RuntimeError):
    """The isolated tool cache is dirty, incomplete, or not at the approved pin."""


class NativeArtifactError(RuntimeError):
    """A native process did not produce its required structured artifact."""


class WorkspaceIntegrityError(RuntimeError):
    """A sealed candidate workspace was modified after completion."""


class EmptyAnswerError(NativeArtifactError):
    """The native query returned no usable answer text."""


def _has_symlink_component(path: Path) -> bool:
    current = path.absolute()
    for candidate in (current, *current.parents):
        if candidate.exists() and candidate.is_symlink():
            return True
    return False


def _canonical_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _redact_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return "[REDACTED_URL]"
    hostname = parsed.hostname or ""
    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        return "[REDACTED_URL]"
    userinfo = "[REDACTED]@" if parsed.username is not None or parsed.password is not None else ""
    query = urlencode(
        [(name, "[REDACTED]") for name, _ in parse_qsl(parsed.query, keep_blank_values=True)]
    )
    fragment = "[REDACTED]" if parsed.fragment else ""
    return urlunsplit((parsed.scheme, f"{userinfo}{hostname}{port}", parsed.path, query, fragment))


def _redact_text(text: str, known_secrets: Sequence[str] = ()) -> str:
    cleaned = text
    for secret in sorted((item for item in known_secrets if item), key=len, reverse=True):
        cleaned = cleaned.replace(secret, "[REDACTED]")
    return _URL.sub(lambda match: _redact_url(match.group(0)), cleaned)


def _redact_value(value: object, known_secrets: Sequence[str] = ()) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            str(key): "[REDACTED]"
            if _SENSITIVE_KEY.search(str(key))
            else _redact_value(item, known_secrets)
            for key, item in mapping.items()
        }
    if isinstance(value, list):
        items = cast(list[object], value)
        return [_redact_value(item, known_secrets) for item in items]
    if isinstance(value, tuple):
        items = cast(tuple[object, ...], value)
        return tuple(_redact_value(item, known_secrets) for item in items)
    if isinstance(value, str):
        return _redact_text(value, known_secrets)
    return value


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ToolCacheError(f"integrity path must be a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise ToolCacheError(f"integrity tree must be a real directory: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file() or path.is_symlink())
    if not files:
        raise ToolCacheError(f"integrity tree is empty: {root}")
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ToolCacheError(f"integrity tree contains a non-regular file: {path}")
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


@dataclass(frozen=True)
class LLMWikiConfig:
    """Filesystem, runtime, and bounded-execution settings for one candidate run."""

    workspace: Path
    tool_cache: Path
    node_executable: Path = Path("node")
    git_executable: str = "git"
    npm_executable: str = "npm"
    base_url: str | None = None
    metering_events_path: Path | None = None
    compile_concurrency: int = 2
    timeout_seconds: float = 600.0
    cleanup_grace_seconds: float = 2.0
    additional_env: Mapping[str, str] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        if not 1 <= self.compile_concurrency <= 50:
            raise ValueError("compile_concurrency must be between 1 and 50")
        if self.timeout_seconds <= 0 or self.cleanup_grace_seconds <= 0:
            raise ValueError("process timeouts must be positive")
        workspace = _canonical_path(self.workspace)
        tool_cache = _canonical_path(self.tool_cache)
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "tool_cache", tool_cache)
        if (
            workspace == tool_cache
            or workspace.is_relative_to(tool_cache)
            or tool_cache.is_relative_to(workspace)
        ):
            raise ValueError("workspace and tool cache must be canonically isolated")


@dataclass(frozen=True)
class AdapterWarning:
    code: str
    message: str
    source_id: str | None = None


@dataclass(frozen=True)
class CommandRecord:
    operation: str
    argv: tuple[str, ...]
    returncode: int
    elapsed_ms: int
    stdout_path: str
    stderr_path: str
    timed_out: bool = False
    terminated: bool = False


@dataclass(frozen=True)
class LLMWikiObservation:
    case_id: str
    question: str
    answer: str
    citations: tuple[str, ...]
    source_ids: tuple[str, ...]
    page_ids: tuple[str, ...]
    latency_ms: int
    raw_result_path: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMWikiRunResult:
    status: Status
    skipped: bool
    pin: CandidatePin
    workspace: str
    environment: Mapping[str, str]
    commands: tuple[CommandRecord, ...]
    observations: tuple[LLMWikiObservation, ...]
    artifacts: tuple[str, ...]
    warnings: tuple[AdapterWarning, ...]
    metering_events: tuple[Mapping[str, Any], ...]
    measured_cost_usd: float | None
    elapsed_ms: int
    workspace_bytes: int
    workspace_seal_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return recursively redacted JSON-safe evidence."""
        evidence = {
            "status": self.status.value,
            "skipped": self.skipped,
            "pin": self.pin.model_dump(mode="json"),
            "workspace": self.workspace,
            "environment": dict(self.environment),
            "commands": [asdict(item) for item in self.commands],
            "observations": [asdict(item) for item in self.observations],
            "artifacts": list(self.artifacts),
            "warnings": [asdict(item) for item in self.warnings],
            "metering_events": [dict(item) for item in self.metering_events],
            "measured_cost_usd": self.measured_cost_usd,
            "elapsed_ms": self.elapsed_ms,
            "workspace_bytes": self.workspace_bytes,
            "workspace_seal_sha256": self.workspace_seal_sha256,
        }
        return cast(dict[str, Any], _redact_value(evidence))


@dataclass(frozen=True)
class _ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_ms: int
    timed_out: bool
    terminated: bool


class _BoundedRunner:
    """Run one process in a fresh group and settle its entire tree on failure."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout: float,
        cleanup_grace: float,
    ) -> _ProcessResult:
        started = time.monotonic()
        process = subprocess.Popen(
            [str(part) for part in argv],
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        terminated = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminated = self._terminate_group(process, cleanup_grace)
            stdout, stderr = process.communicate()
        except BaseException:
            self._terminate_group(process, cleanup_grace)
            raise
        elapsed_ms = round((time.monotonic() - started) * 1000)
        return _ProcessResult(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed_ms=elapsed_ms,
            timed_out=timed_out,
            terminated=terminated,
        )

    @staticmethod
    def _terminate_group(process: subprocess.Popen[str], grace: float) -> bool:
        # The group leader may exit while descendants remain. Never use leader
        # liveness as proof that the process group is gone.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        if process.poll() is None:
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=grace)
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return True
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        if process.poll() is None:
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=grace)
        return True


class LLMWikiAdapter:
    """Execute the exact approved LLM Wiki build without global state or servers."""

    def __init__(self, config: LLMWikiConfig) -> None:
        self.config = config
        self._runner = _BoundedRunner()
        self._commands: list[CommandRecord] = []

    def prepare_tool_cache(self) -> Path:
        """Install or verify the exact approved package and return its SDK entry point."""
        return self._ensure_tool_cache()

    def verify_workspace(self) -> str:
        """Verify the completed workspace seal and return its root digest."""
        return self._verify_workspace_seal()

    @property
    def pin(self) -> CandidatePin:
        return CandidatePin(
            id=CandidateId.LLM_WIKI,
            distribution=APPROVED_DISTRIBUTION,
            version=APPROVED_VERSION,
            commit=APPROVED_COMMIT,
            repository=APPROVED_REPOSITORY,
            license=APPROVED_LICENSE,
        )

    def run(
        self,
        documents: Sequence[NormalizedDocument],
        cases: Sequence[BenchmarkCase | CandidateQuery],
        *,
        holdouts: Sequence[Holdout] = (),
        oracle_paths: Sequence[Path] = (),
        api_key: str | None,
    ) -> LLMWikiRunResult:
        """Ingest whole documents and execute native compile/query/export/lint surfaces."""
        started = time.monotonic()
        self._commands = []
        self._validate_inputs(documents, cases, holdouts, oracle_paths)
        if not api_key:
            return self._result(
                status=Status.MISSING_PROVIDER,
                skipped=True,
                started=started,
                environment={},
                observations=[],
                warnings=[
                    AdapterWarning(
                        "MISSING_PROVIDER", "OPENAI_API_KEY is unavailable; candidate skipped"
                    )
                ],
            )

        self._ensure_empty_workspace()
        package_entry = self._ensure_tool_cache()
        self._prepare_workspace()
        environment, reported_environment = self._environment(api_key)
        known_secrets = self._secret_values(environment)
        warnings: list[AdapterWarning] = []
        observations: list[LLMWikiObservation] = []
        source_map: list[dict[str, Any]] = []
        filename_to_source_id: dict[str, str] = {}
        status = Status.OK

        try:
            for index, document in enumerate(documents, start=1):
                mapped = self._map_document(document, index)
                response, command = self._invoke(
                    "ingest",
                    {"root": str(self.config.workspace), "document": mapped},
                    package_entry,
                    environment,
                    known_secrets,
                )
                if command.returncode != 0:
                    raise NativeArtifactError("native ingest failed")
                filename = self._required_string(response, "filename", "ingest")
                truncated = response.get("truncated")
                if not isinstance(truncated, bool):
                    raise NativeArtifactError("native ingest result has invalid truncated field")
                record = {
                    "source_id": document.source_id,
                    "source_kind": document.source_kind.value,
                    "canonical_url": document.canonical_url,
                    "content_hash": document.content_hash,
                    "filename": filename,
                    "native_truncated": truncated,
                    "original_chars": len(mapped["text"]),
                }
                source_map.append(record)
                filename_to_source_id[filename] = document.source_id
                if truncated:
                    warnings.append(
                        AdapterWarning(
                            "SOURCE_TRUNCATED",
                            f"native {APPROVED_DISTRIBUTION} limit truncated the mapped "
                            f"source at {MAX_SOURCE_CHARS} characters",
                            document.source_id,
                        )
                    )
            self._write_json(self.config.workspace / "artifacts" / "source-map.json", source_map)

            compile_response, compile_command = self._invoke(
                "compile",
                {
                    "root": str(self.config.workspace),
                    "concurrency": self.config.compile_concurrency,
                },
                package_entry,
                environment,
                known_secrets,
            )
            if compile_command.returncode != 0:
                raise NativeArtifactError("native compile failed")
            raw_compile_errors = compile_response.get("errors", [])
            if not isinstance(raw_compile_errors, list):
                raise NativeArtifactError("native compile result has invalid errors field")
            compile_errors = cast(list[Any], raw_compile_errors)
            for error in compile_errors:
                warnings.append(AdapterWarning("COMPILE_WARNING", str(error)))
            self._write_json(
                self.config.workspace / "artifacts" / "native-compile.json", compile_response
            )

            for case in cases:
                query_response, query_command = self._invoke(
                    "query",
                    {"root": str(self.config.workspace), "question": case.question},
                    package_entry,
                    environment,
                    known_secrets,
                )
                if query_command.returncode != 0:
                    raise NativeArtifactError(f"native query failed for {case.case_id}")
                answer = self._required_string(query_response, "answer", "query")
                if not answer.strip():
                    raise EmptyAnswerError(
                        f"native query returned an empty answer for {case.case_id}"
                    )
                page_ids = self._string_list(query_response.get("pageIds"), "query pageIds")
                native_warnings = self._native_query_warnings(query_response)
                citations = tuple(dict.fromkeys(_CITATION.findall(answer)))
                source_ids = tuple(
                    dict.fromkeys(
                        filename_to_source_id[citation]
                        for citation in citations
                        if citation in filename_to_source_id
                    )
                )
                observations.append(
                    LLMWikiObservation(
                        case_id=case.case_id,
                        question=case.question,
                        answer=answer,
                        citations=citations,
                        source_ids=source_ids,
                        page_ids=tuple(page_ids),
                        latency_ms=query_command.elapsed_ms,
                        raw_result_path=f"process/{len(self._commands):03d}-query-response.json",
                        warnings=tuple(native_warnings),
                    )
                )

            export_response, export_command = self._invoke(
                "export",
                {"root": str(self.config.workspace)},
                package_entry,
                environment,
                known_secrets,
            )
            if export_command.returncode != 0:
                raise NativeArtifactError("native export failed")
            self._validate_export(export_response)
            self._write_json(
                self.config.workspace / "artifacts" / "native-export.json", export_response
            )

            lint_response, lint_command = self._invoke(
                "lint",
                {"root": str(self.config.workspace)},
                package_entry,
                environment,
                known_secrets,
            )
            if lint_command.returncode != 0:
                raise NativeArtifactError("native lint failed")
            self._validate_lint(lint_response)
            self._write_json(
                self.config.workspace / "artifacts" / "native-lint.json", lint_response
            )
            raw_lint_results = lint_response.get("results", [])
            assert isinstance(raw_lint_results, list)
            for raw_item in cast(list[Any], raw_lint_results):
                if not isinstance(raw_item, dict):
                    continue
                item = cast(dict[str, Any], raw_item)
                if item.get("severity") in {"warning", "error"}:
                    detail = (
                        f"{item.get('severity')}: {item.get('file', 'unknown')}: "
                        f"{item.get('message', '')}"
                    )
                    warnings.append(AdapterWarning("NATIVE_LINT", detail))
        except EmptyAnswerError as error:
            status = Status.FAILED
            warnings.append(AdapterWarning("EMPTY_ANSWER", _redact_text(str(error), known_secrets)))
        except (NativeArtifactError, OSError) as error:
            status = Status.FAILED
            warnings.append(
                AdapterWarning("NATIVE_FAILURE", _redact_text(str(error), known_secrets))
            )

        events, measured_cost, metering_warnings = self._read_metering(known_secrets)
        warnings.extend(metering_warnings)
        self._redact_workspace(known_secrets)
        seal_sha256 = self._seal_workspace()
        return self._result(
            status=status,
            skipped=False,
            started=started,
            environment=reported_environment,
            observations=observations,
            warnings=warnings,
            metering_events=events,
            measured_cost_usd=measured_cost,
            workspace_seal_sha256=seal_sha256,
        )

    def _validate_inputs(
        self,
        documents: Sequence[NormalizedDocument],
        cases: Sequence[BenchmarkCase | CandidateQuery],
        holdouts: Sequence[Holdout],
        oracle_paths: Sequence[Path],
    ) -> None:
        source_ids = [document.source_id for document in documents]
        duplicates = sorted({item for item in source_ids if source_ids.count(item) > 1})
        if duplicates:
            raise ValueError(f"duplicate source_id values: {', '.join(duplicates)}")
        case_ids = [case.case_id for case in cases]
        duplicate_cases = sorted({item for item in case_ids if case_ids.count(item) > 1})
        if duplicate_cases:
            raise ValueError(f"duplicate case_id values: {', '.join(duplicate_cases)}")
        if not documents or not cases:
            raise ValueError("LLM Wiki requires at least one document and one benchmark case")

        markers: set[str] = set()
        held_source_ids: set[str] = set()
        for holdout in holdouts:
            markers.add(holdout.case_id)
            markers.update(reply_id for reply_id in holdout.reply_ids if reply_id)
            if holdout.reference_text:
                markers.add(holdout.reference_text)
            held_source_ids.update(holdout.source_ids)

        workspace = self.config.workspace.resolve(strict=False)
        for oracle_path in oracle_paths:
            resolved = oracle_path.resolve(strict=False)
            if resolved == workspace or resolved.is_relative_to(workspace):
                raise ValueError("oracle path must remain outside the candidate workspace")
        markers.update(self._oracle_markers(oracle_paths))

        for document in documents:
            if document.source_id in held_source_ids:
                raise ValueError(
                    f"holdout/oracle leakage: held-out source {document.source_id} is in corpus"
                )
            if self._contains_forbidden_marker(document.model_dump_json(), markers):
                raise ValueError("holdout/oracle leakage detected in candidate-facing document")
        for case in cases:
            if self._contains_forbidden_marker(case.question, markers):
                raise ValueError("holdout/oracle leakage detected in benchmark question/prompt")

    @staticmethod
    def _contains_forbidden_marker(value: str, markers: Sequence[str] | set[str]) -> bool:
        normalized = unquote(value).replace("\\", "/").casefold()
        return any(
            marker and unquote(marker).replace("\\", "/").casefold() in normalized
            for marker in markers
        )

    @staticmethod
    def _oracle_markers(oracle_paths: Sequence[Path]) -> set[str]:
        markers: set[str] = set()
        for oracle_path in oracle_paths:
            expanded = oracle_path.expanduser()
            resolved = expanded.resolve(strict=False)
            markers.update(
                {
                    str(oracle_path),
                    expanded.as_posix(),
                    resolved.as_posix(),
                    resolved.as_uri(),
                    expanded.name,
                }
            )
            if expanded.is_file() and not expanded.is_symlink():
                try:
                    marker_text = expanded.read_text(encoding="utf-8").strip()
                except (OSError, UnicodeDecodeError):
                    continue
                if marker_text and len(marker_text) <= 4096:
                    markers.add(marker_text)
        return {marker for marker in markers if marker}

    def _ensure_empty_workspace(self) -> None:
        workspace = _canonical_path(self.config.workspace)
        if _has_symlink_component(self.config.workspace):
            raise ValueError("candidate workspace path cannot contain symlinks")
        if workspace.exists():
            if not workspace.is_dir():
                raise ValueError("candidate workspace must be a directory")
            if any(workspace.iterdir()):
                if (workspace / _SEAL_NAME).is_file():
                    self._verify_workspace_seal()
                    raise FileExistsError(
                        f"candidate workspace is sealed and cannot be reused: {workspace}"
                    )
                raise FileExistsError(f"candidate workspace is non-empty: {workspace}")

    def _prepare_workspace(self) -> None:
        self.config.workspace.mkdir(mode=0o700, parents=True, exist_ok=False)
        for name in ("artifacts", "process"):
            (self.config.workspace / name).mkdir(mode=0o700)

    def _ensure_tool_cache(self) -> Path:
        cache = _canonical_path(self.config.tool_cache)
        if _has_symlink_component(self.config.tool_cache):
            raise ToolCacheError("tool cache path cannot contain symlinks")
        if cache.exists() and any(cache.iterdir()):
            return self._verify_tool_cache(cache)
        cache.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._install_tool(cache)
        return self._verify_tool_cache(cache)

    def _verify_tool_cache(self, cache: Path) -> Path:
        source = cache / "source"
        marker_path = cache / _PIN_MARKER
        package_path = source / "package.json"
        license_path = source / "LICENSE"
        entry = source / "dist" / "index.js"
        for path in (source, package_path, license_path, entry):
            if path.is_symlink():
                raise ToolCacheError(f"tool cache contains a symlink: {path}")
        if not marker_path.is_file() or not package_path.is_file() or not entry.is_file():
            raise ToolCacheError(f"dirty or incomplete LLM Wiki tool cache: {cache}")
        marker = self._read_object(marker_path, "tool cache marker")
        package = self._read_object(package_path, "package metadata")
        expected = {
            "distribution": APPROVED_DISTRIBUTION,
            "version": APPROVED_VERSION,
            "commit": APPROVED_COMMIT,
            "license": APPROVED_LICENSE,
            "license_sha256": APPROVED_LICENSE_SHA256,
            "dist_tree_sha256": APPROVED_DIST_TREE_SHA256,
        }
        if marker != expected:
            raise ToolCacheError("LLM Wiki tool cache integrity marker mismatch")
        if (
            package.get("name") != APPROVED_DISTRIBUTION
            or package.get("version") != APPROVED_VERSION
            or package.get("license") != APPROVED_LICENSE
        ):
            raise ToolCacheError("LLM Wiki package identity/version/license mismatch")
        if _sha256_file(license_path) != APPROVED_LICENSE_SHA256:
            raise ToolCacheError("LLM Wiki LICENSE integrity mismatch")
        if _tree_sha256(source / "dist") != APPROVED_DIST_TREE_SHA256:
            raise ToolCacheError("LLM Wiki built SDK integrity mismatch")
        self._verify_git_checkout(source)
        self._verify_sdk_import(entry, cache)
        return entry

    def _verify_git_checkout(self, source: Path) -> None:
        status = self._git(source, "status", "--porcelain=v1", "--untracked-files=all")
        if status:
            raise ToolCacheError("LLM Wiki source checkout is dirty")
        head = self._git(source, "rev-parse", "HEAD")
        if head != APPROVED_COMMIT:
            raise ToolCacheError(f"LLM Wiki source HEAD mismatch: {head}")
        origin = self._git(source, "remote", "get-url", "origin")
        if self._canonical_repo(origin) != self._canonical_repo(APPROVED_REPOSITORY):
            raise ToolCacheError(f"LLM Wiki source origin mismatch: {origin}")

    def _git(self, source: Path, *arguments: str) -> str:
        result = self._runner.run(
            [self.config.git_executable, "-C", str(source), *arguments],
            cwd=source,
            env=self._base_process_environment(),
            timeout=self.config.timeout_seconds,
            cleanup_grace=self.config.cleanup_grace_seconds,
        )
        if result.returncode != 0:
            raise ToolCacheError(
                _redact_text(result.stderr or f"git command failed: {' '.join(arguments)}")
            )
        return result.stdout.strip()

    @staticmethod
    def _canonical_repo(value: str) -> str:
        return value.strip().removesuffix("/").removesuffix(".git")

    def _verify_sdk_import(self, entry: Path, cache: Path) -> None:
        probe = (
            f"import({json.dumps(entry.as_uri())}).then(m => {{ "
            'if (typeof m.createWiki !== "function") process.exit(2); })'
        )
        result = self._runner.run(
            [str(self.config.node_executable), "--input-type=module", "-e", probe],
            cwd=cache,
            env=self._base_process_environment(),
            timeout=self.config.timeout_seconds,
            cleanup_grace=self.config.cleanup_grace_seconds,
        )
        if result.returncode != 0:
            raise ToolCacheError("built LLM Wiki SDK import verification failed")

    @staticmethod
    def _read_object(path: Path, label: str) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ToolCacheError(f"malformed {label}: {error}") from error
        if not isinstance(value, dict):
            raise ToolCacheError(f"malformed {label}")
        return cast(dict[str, Any], value)

    def _install_tool(self, cache: Path) -> None:
        source = cache / "source"
        install_env = self._base_process_environment()
        commands = (
            [self.config.git_executable, "init", str(source)],
            [
                self.config.git_executable,
                "-C",
                str(source),
                "remote",
                "add",
                "origin",
                APPROVED_REPOSITORY,
            ],
            [
                self.config.git_executable,
                "-C",
                str(source),
                "fetch",
                "--depth=1",
                "origin",
                APPROVED_COMMIT,
            ],
            [self.config.git_executable, "-C", str(source), "checkout", "--detach", "FETCH_HEAD"],
            [self.config.npm_executable, "ci", "--ignore-scripts", "--no-audit", "--no-fund"],
            [self.config.npm_executable, "run", "build"],
        )
        log_lines: list[str] = []
        for argv in commands:
            cwd = source if argv[0] == self.config.npm_executable else cache
            result = self._runner.run(
                argv,
                cwd=cwd,
                env=install_env,
                timeout=self.config.timeout_seconds,
                cleanup_grace=self.config.cleanup_grace_seconds,
            )
            log_lines.extend((f"$ {' '.join(map(str, argv))}", result.stdout, result.stderr))
            if result.returncode != 0:
                (cache / "install.log").write_text("\n".join(log_lines), encoding="utf-8")
                raise ToolCacheError(
                    f"failed to install exact LLM Wiki pin: {' '.join(map(str, argv))}"
                )
        revision = self._runner.run(
            [self.config.git_executable, "-C", str(source), "rev-parse", "HEAD"],
            cwd=cache,
            env=install_env,
            timeout=self.config.timeout_seconds,
            cleanup_grace=self.config.cleanup_grace_seconds,
        )
        if revision.returncode != 0 or revision.stdout.strip() != APPROVED_COMMIT:
            raise ToolCacheError("installed LLM Wiki checkout does not match approved commit")
        package = json.loads((source / "package.json").read_text(encoding="utf-8"))
        if (
            package.get("name") != APPROVED_DISTRIBUTION
            or package.get("version") != APPROVED_VERSION
            or package.get("license") != APPROVED_LICENSE
        ):
            raise ToolCacheError(
                "approved commit does not declare the approved package identity/license"
            )
        if _sha256_file(source / "LICENSE") != APPROVED_LICENSE_SHA256:
            raise ToolCacheError("approved checkout LICENSE integrity mismatch")
        if _tree_sha256(source / "dist") != APPROVED_DIST_TREE_SHA256:
            raise ToolCacheError("approved checkout produced an unexpected SDK build")
        (cache / "install.log").write_text("\n".join(log_lines), encoding="utf-8")
        self._write_json(
            cache / _PIN_MARKER,
            {
                "distribution": APPROVED_DISTRIBUTION,
                "version": APPROVED_VERSION,
                "commit": APPROVED_COMMIT,
                "license": APPROVED_LICENSE,
                "license_sha256": APPROVED_LICENSE_SHA256,
                "dist_tree_sha256": APPROVED_DIST_TREE_SHA256,
            },
        )

    def _environment(self, api_key: str) -> tuple[dict[str, str], dict[str, str]]:
        environment = self._base_process_environment()
        settings = {
            "LLMWIKI_PROVIDER": "openai",
            "LLMWIKI_MODEL": "gpt-5-mini",
            "LLMWIKI_EMBEDDING_MODEL": "text-embedding-3-small",
            "LLMWIKI_COMPILE_CONCURRENCY": str(self.config.compile_concurrency),
            "LLMWIKI_VERBOSE": "1",
        }
        if self.config.base_url:
            settings["OPENAI_BASE_URL"] = self.config.base_url
            settings["OPENAI_EMBEDDINGS_BASE_URL"] = self.config.base_url
        environment.update(settings)
        environment["OPENAI_API_KEY"] = api_key
        reported = cast(dict[str, str], _redact_value(settings, (api_key,)))
        return environment, reported

    @staticmethod
    def _secret_values(environment: Mapping[str, str]) -> tuple[str, ...]:
        return tuple(
            value
            for key, value in environment.items()
            if value
            and (key == "OPENAI_API_KEY" or (_SENSITIVE_KEY.search(key) and len(value) >= 8))
        )

    def _base_process_environment(self) -> dict[str, str]:
        allowed = (
            "PATH",
            "TMPDIR",
            "TEMP",
            "TMP",
            "LANG",
            "LC_ALL",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "NODE_EXTRA_CA_CERTS",
            "SYSTEMROOT",
        )
        environment = {name: os.environ[name] for name in allowed if name in os.environ}
        environment.update(self.config.additional_env)
        return environment

    def _map_document(self, document: NormalizedDocument, index: int) -> dict[str, str]:
        safe_id = re.sub(r"[^A-Za-z0-9]+", "-", document.source_id).strip("-").lower()
        filename = f"doc-{safe_id[:60] or index}.md"
        provenance = {
            "source_id": document.source_id,
            "source_kind": document.source_kind.value,
            "canonical_url": document.canonical_url,
            "content_hash": document.content_hash,
            "created_at": document.created_at.isoformat() if document.created_at else None,
            "updated_at": document.updated_at.isoformat() if document.updated_at else None,
            "authors": document.authors,
            "related_source_ids": document.related_source_ids,
            "metadata": document.metadata,
        }
        text = (
            "<!-- autobrain-provenance\n"
            + json.dumps(provenance, sort_keys=True, ensure_ascii=True)
            + "\n-->\n\n"
            + document.text
        )
        return {
            "title": f"{document.title} [{document.source_id}]",
            "text": text,
            "source": document.canonical_url,
            "filename": filename,
        }

    def _invoke(
        self,
        operation: str,
        request: Mapping[str, Any],
        package_entry: Path,
        environment: Mapping[str, str],
        known_secrets: Sequence[str],
    ) -> tuple[dict[str, Any], CommandRecord]:
        number = len(self._commands) + 1
        process_dir = self.config.workspace / "process"
        request_path = process_dir / f"{number:03d}-{operation}-request.json"
        response_path = process_dir / f"{number:03d}-{operation}-response.json"
        stdout_path = process_dir / f"{number:03d}-{operation}.stdout.log"
        stderr_path = process_dir / f"{number:03d}-{operation}.stderr.log"
        self._write_json(request_path, request)
        driver = self._ensure_driver()
        argv = (
            str(self.config.node_executable),
            str(driver),
            str(package_entry),
            operation,
            str(request_path),
            str(response_path),
        )
        try:
            result = self._runner.run(
                argv,
                cwd=self.config.workspace,
                env=environment,
                timeout=self.config.timeout_seconds,
                cleanup_grace=self.config.cleanup_grace_seconds,
            )
        except BaseException:
            self._write_json(request_path, _redact_value(request, known_secrets))
            if response_path.is_file():
                raw_response = response_path.read_text(encoding="utf-8", errors="replace")
                response_path.write_text(
                    _redact_text(raw_response, known_secrets), encoding="utf-8"
                )
            self._redact_workspace(known_secrets)
            raise
        stdout_path.write_text(_redact_text(result.stdout, known_secrets), encoding="utf-8")
        stderr_path.write_text(_redact_text(result.stderr, known_secrets), encoding="utf-8")
        record = CommandRecord(
            operation=operation,
            argv=argv,
            returncode=result.returncode,
            elapsed_ms=result.elapsed_ms,
            stdout_path=str(stdout_path.relative_to(self.config.workspace)),
            stderr_path=str(stderr_path.relative_to(self.config.workspace)),
            timed_out=result.timed_out,
            terminated=result.terminated,
        )
        self._commands.append(record)
        redacted_request = _redact_value(request, known_secrets)
        self._write_json(request_path, redacted_request)
        if response_path.is_file():
            raw_response = response_path.read_text(encoding="utf-8", errors="replace")
            redacted_response = _redact_text(raw_response, known_secrets)
            response_path.write_text(redacted_response, encoding="utf-8")
        if result.returncode != 0:
            return {}, record
        if not response_path.is_file():
            raise NativeArtifactError(f"native {operation} did not write its response artifact")
        try:
            loaded = json.loads(response_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise NativeArtifactError(f"native {operation} response is malformed JSON") from error
        if not isinstance(loaded, dict):
            raise NativeArtifactError(f"native {operation} response must be an object")
        redacted = _redact_value(cast(dict[str, Any], loaded), known_secrets)
        self._write_json(response_path, redacted)
        return cast(dict[str, Any], redacted), record

    def _ensure_driver(self) -> Path:
        driver = self.config.workspace / _DRIVER_NAME
        if not driver.exists():
            driver.write_text(_NODE_DRIVER, encoding="utf-8")
        return driver

    def _read_metering(
        self, known_secrets: Sequence[str]
    ) -> tuple[list[Mapping[str, Any]], float | None, list[AdapterWarning]]:
        path = self.config.metering_events_path
        if path is None:
            if self.config.base_url:
                return (
                    [],
                    None,
                    [
                        AdapterWarning(
                            "METERING_UNAVAILABLE",
                            "metering endpoint configured but no event ledger was supplied",
                        )
                    ],
                )
            return (
                [],
                None,
                [
                    AdapterWarning(
                        "COST_INCOMPLETE", "no run-local metering endpoint was configured"
                    )
                ],
            )
        if not path.is_file():
            return (
                [],
                None,
                [
                    AdapterWarning(
                        "METERING_UNAVAILABLE", "configured metering event ledger is unavailable"
                    )
                ],
            )
        try:
            source_bytes = path.read_bytes()
            source_text = source_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return (
                [],
                None,
                [AdapterWarning("METERING_MALFORMED", "metering ledger could not be decoded")],
            )
        events: list[Mapping[str, Any]] = []
        total = 0.0
        malformed = False
        malformed_records = 0
        for line in source_text.splitlines():
            if not line.strip():
                continue
            try:
                raw_value = json.loads(line)
            except json.JSONDecodeError:
                malformed = True
                malformed_records += 1
                continue
            if not isinstance(raw_value, dict):
                malformed = True
                malformed_records += 1
                continue
            value = cast(dict[str, Any], raw_value)
            if isinstance(value.get("usd"), bool) or not isinstance(value.get("usd"), (int, float)):
                malformed = True
                malformed_records += 1
                continue
            if value.get("candidate") not in {None, CandidateId.LLM_WIKI.value}:
                continue
            usd = float(value["usd"])
            if usd < 0:
                malformed = True
                malformed_records += 1
                continue
            events.append(cast(dict[str, Any], _redact_value(value, known_secrets)))
            total += usd
        artifacts = self.config.workspace / "artifacts"
        self._write_jsonl(artifacts / "metering-events.jsonl", events)
        self._write_json(
            artifacts / "metering-provenance.json",
            {
                "schema_version": 1,
                "source_kind": "external-redacted-copy",
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "raw_retained": False,
                "accepted_events": len(events),
                "malformed_records": malformed_records,
            },
        )
        if malformed or not events:
            return (
                events,
                None,
                [
                    AdapterWarning(
                        "METERING_MALFORMED",
                        "metering ledger was malformed or lacked measured USD events",
                    )
                ],
            )
        return events, total, []

    def _redact_workspace(self, known_secrets: Sequence[str]) -> None:
        for path in self.config.workspace.rglob("*"):
            if path.is_symlink():
                raise WorkspaceIntegrityError(f"workspace contains a symlink: {path}")
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            redacted = _redact_text(text, known_secrets)
            if redacted != text:
                path.write_text(redacted, encoding="utf-8")

    def _seal_workspace(self) -> str:
        workspace = _canonical_path(self.config.workspace)
        seal_path = workspace / _SEAL_NAME
        if seal_path.exists():
            raise WorkspaceIntegrityError("workspace seal already exists")
        entries = self._workspace_entries(workspace)
        root_sha256 = self._entries_digest(entries)
        self._write_json(
            seal_path,
            {"schema_version": 1, "root_sha256": root_sha256, "files": entries},
        )
        return root_sha256

    def _verify_workspace_seal(self) -> str:
        workspace = _canonical_path(self.config.workspace)
        seal_path = workspace / _SEAL_NAME
        if seal_path.is_symlink() or not seal_path.is_file():
            raise WorkspaceIntegrityError("workspace seal is missing or invalid")
        try:
            raw_seal = json.loads(seal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise WorkspaceIntegrityError(f"workspace seal is malformed: {error}") from error
        if not isinstance(raw_seal, dict):
            raise WorkspaceIntegrityError("workspace seal is malformed")
        seal = cast(dict[str, Any], raw_seal)
        files = seal.get("files")
        root_sha256 = seal.get("root_sha256")
        if (
            seal.get("schema_version") != 1
            or not isinstance(files, list)
            or not isinstance(root_sha256, str)
        ):
            raise WorkspaceIntegrityError("workspace seal is malformed")
        expected = cast(list[Any], files)
        actual = self._workspace_entries(workspace)
        if expected != actual or self._entries_digest(actual) != root_sha256:
            raise WorkspaceIntegrityError("workspace integrity seal detected tampering")
        return root_sha256

    @staticmethod
    def _workspace_entries(workspace: Path) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for path in sorted(workspace.rglob("*")):
            if path.name == _SEAL_NAME and path.parent == workspace:
                continue
            if path.is_symlink():
                raise WorkspaceIntegrityError(f"workspace contains a symlink: {path}")
            if not path.is_file():
                continue
            entries.append(
                {
                    "path": path.relative_to(workspace).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        return entries

    @staticmethod
    def _entries_digest(entries: Sequence[Mapping[str, Any]]) -> str:
        payload = json.dumps(list(entries), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def _result(
        self,
        *,
        status: Status,
        skipped: bool,
        started: float,
        environment: Mapping[str, str],
        observations: Sequence[LLMWikiObservation],
        warnings: Sequence[AdapterWarning],
        metering_events: Sequence[Mapping[str, Any]] = (),
        measured_cost_usd: float | None = None,
        workspace_seal_sha256: str | None = None,
    ) -> LLMWikiRunResult:
        workspace_bytes = self._directory_bytes(self.config.workspace)
        artifacts_dir = self.config.workspace / "artifacts"
        artifacts = (
            tuple(
                str(path.relative_to(artifacts_dir))
                for path in sorted(artifacts_dir.rglob("*"))
                if path.is_file()
            )
            if artifacts_dir.is_dir()
            else ()
        )
        return LLMWikiRunResult(
            status=status,
            skipped=skipped,
            pin=self.pin,
            workspace=str(self.config.workspace),
            environment=dict(environment),
            commands=tuple(self._commands),
            observations=tuple(observations),
            artifacts=artifacts,
            warnings=tuple(warnings),
            metering_events=tuple(metering_events),
            measured_cost_usd=measured_cost_usd,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            workspace_bytes=workspace_bytes,
            workspace_seal_sha256=workspace_seal_sha256,
        )

    @staticmethod
    def _required_string(value: Mapping[str, Any], key: str, operation: str) -> str:
        item = value.get(key)
        if not isinstance(item, str):
            raise NativeArtifactError(f"native {operation} result has invalid {key} field")
        return item

    @staticmethod
    def _string_list(value: object, label: str) -> list[str]:
        if not isinstance(value, list):
            raise NativeArtifactError(f"native {label} must be a string array")
        items = cast(list[Any], value)
        if not all(isinstance(item, str) for item in items):
            raise NativeArtifactError(f"native {label} must be a string array")
        return [item for item in items if isinstance(item, str)]

    @staticmethod
    def _native_query_warnings(response: Mapping[str, Any]) -> list[str]:
        value = response.get("warnings", [])
        if not isinstance(value, list):
            raise NativeArtifactError("native query warnings must be an array")
        warnings: list[str] = []
        for raw_item in cast(list[Any], value):
            if not isinstance(raw_item, dict):
                raise NativeArtifactError("native query warning is malformed")
            item = cast(dict[str, Any], raw_item)
            if not isinstance(item.get("code"), str):
                raise NativeArtifactError("native query warning is malformed")
            warnings.append(f"{item['code']}: {item.get('message', '')}")
        return warnings

    @staticmethod
    def _validate_export(response: Mapping[str, Any]) -> None:
        raw_pages = response.get("pages")
        if not isinstance(raw_pages, list):
            raise NativeArtifactError("native export pages artifact is malformed")
        pages = cast(list[Any], raw_pages)
        if not all(isinstance(page, dict) for page in pages):
            raise NativeArtifactError("native export pages artifact is malformed")
        if not isinstance(response.get("pageCount"), int) or response["pageCount"] != len(pages):
            raise NativeArtifactError("native export pageCount does not match pages")

    @staticmethod
    def _validate_lint(response: Mapping[str, Any]) -> None:
        if not isinstance(response.get("results"), list):
            raise NativeArtifactError("native lint results artifact is malformed")
        for key in ("errors", "warnings", "info"):
            if not isinstance(response.get(key), int):
                raise NativeArtifactError(f"native lint {key} count is malformed")

    @staticmethod
    def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        payload = "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n" for value in values
        )
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _directory_bytes(path: Path) -> int:
        if not path.is_dir():
            return 0
        return sum(
            item.stat().st_size
            for item in path.rglob("*")
            if item.is_file() and not item.is_symlink()
        )


_NODE_DRIVER = r"""import { readFile, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

const [, , packageEntry, operation, requestPath, responsePath] = process.argv;
const request = JSON.parse(await readFile(requestPath, "utf8"));
const { createWiki } = await import(pathToFileURL(packageEntry).href);
const wiki = createWiki({ root: request.root });
let result;
switch (operation) {
  case "ingest": {
    const { title, text, source } = request.document;
    result = await wiki.ingestText({ title, text, source });
    break;
  }
  case "compile":
    result = await wiki.compile({ concurrency: request.concurrency });
    break;
  case "query":
    result = await wiki.query(request.question, { debug: true, save: false });
    break;
  case "export":
    result = await wiki.exportJson({ projectId: "autobrain-llm-wiki" });
    break;
  case "lint":
    result = await wiki.lint();
    break;
  default:
    throw new Error(`unsupported AutoBrain LLM Wiki operation: ${operation}`);
}
await writeFile(responsePath, JSON.stringify(result));
"""
