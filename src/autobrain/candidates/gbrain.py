"""Pinned, process-isolated adapter for the GBrain TypeScript CLI.

This module deliberately contains no GBrain implementation code.  It only
materializes whole documents as Markdown and drives the pinned ``src/cli.ts``
entry point through Bun, retaining native JSON and filesystem evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from autobrain.cancellation import RunCancellation
from autobrain.candidates.gbrain_config import GBrainExecutionConfig
from autobrain.lifecycle import CleanupReceipt, remaining_paths
from autobrain.models import CandidateId, NormalizedDocument
from autobrain.retrieval_ids import provenance_map, resolve_retrieved_source_ids

GBRAIN_COMMIT = "f49ca569232dbc0d8e0783d84606115e3bfe5ab1"
GBRAIN_VERSION = "0.46.19.0"
GBRAIN_REPOSITORY = "https://github.com/garrytan/gbrain.git"
EMBEDDING_MODEL = "openai:text-embedding-3-small"
THINK_MODEL = "openai:gpt-5-mini"
_ALLOWED_COMMANDS = {"init", "import", "sync", "dream", "search", "query", "think", "status"}
_FORBIDDEN_SURFACES = (
    "personal-agent",
    "personal_agent",
    "minion",
    "serve",
    "schema",
    "auth",
)


class GBrainError(RuntimeError):
    """A typed adapter failure with enough context for a candidate observation."""


class GBrainIsolationError(GBrainError):
    """Candidate input or state attempted to cross the holdout boundary."""


class GBrainProcessError(GBrainError):
    """The pinned CLI exited unsuccessfully or emitted invalid JSON."""


class GBrainCapabilityError(GBrainProcessError):
    """The pinned CLI does not advertise a capability required by this mode."""


class GBrainMissingProviderError(GBrainProcessError):
    """Native GBrain diagnosed unavailable OpenAI embedding credentials."""

    status = "MISSING_PROVIDER"

    def __init__(self, message: str, *, stdout: str, stderr: str) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


class GBrainInterruptedError(GBrainProcessError):
    """The adapter received an external termination request."""


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_ms: int


@dataclass
class GBrainResult:
    status: str
    commit: str = GBRAIN_COMMIT
    version: str = GBRAIN_VERSION
    answer: str = ""
    evidence: list[Any] = field(default_factory=list)
    search_evidence: list[Any] = field(default_factory=list)
    query_evidence: list[Any] = field(default_factory=list)
    gather_evidence: dict[str, Any] = field(default_factory=dict)
    citations: list[Any] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    pages_gathered: int = 0
    model_used: str = THINK_MODEL
    usage: dict[str, int] | None = None
    cost_usd: float | None = None
    timings_ms: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    footprint_bytes: int = 0
    native: dict[str, Any] = field(default_factory=dict)
    commands: list[CommandResult] = field(default_factory=list)
    cost_status: str = "COST_INCOMPLETE"
    base_url_supported: bool | None = None
    proxy_events: list[dict[str, Any]] = field(default_factory=list)
    proxy_usage: dict[str, int] | None = None
    keyword_only: bool = False
    semantic_enabled: bool = True
    semantic_quality: str = "configured"
    recommendation_eligible: bool = False
    execution_config: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["commands"] = [command.__dict__ for command in self.commands]
        return result


Runner = Callable[[Sequence[str], Path, dict[str, str], float], CommandResult]


def run_process(
    command: Sequence[str],
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    cancellation: RunCancellation | None = None,
) -> CommandResult:
    """Run one process group and tear it down on timeout or external interruption."""
    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    previous_sigterm: Any = None
    owns_signal = threading.current_thread() is threading.main_thread()

    def interrupt(_signum: int, _frame: object) -> None:
        raise GBrainInterruptedError(f"GBrain command interrupted: {' '.join(command)}")

    if owns_signal:
        previous_sigterm = signal.signal(signal.SIGTERM, interrupt)
    remove_callback = (
        cancellation.add_callback(lambda: _terminate_process_tree(process))
        if cancellation is not None
        else lambda: None
    )
    try:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            if cancellation is not None:
                cancellation.raise_if_cancelled()
        except subprocess.TimeoutExpired as error:
            raise GBrainProcessError(f"GBrain command timed out: {' '.join(command)}") from error
    finally:
        remove_callback()
        if process.poll() is None:
            _terminate_process_tree(process)
        if owns_signal and previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
    return CommandResult(
        tuple(command),
        process.returncode,
        stdout,
        stderr,
        round((time.monotonic() - started) * 1000),
    )


_default_runner = run_process


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Bound cleanup to the process group created by this adapter."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=0.5)
    except (ProcessLookupError, OSError):
        with suppress(ProcessLookupError):
            process.kill()


def parse_json_output(stdout: str) -> Any:
    """Parse clean JSON, tolerating native informational lines around it."""
    text = stdout.strip()
    if not text:
        raise GBrainProcessError("GBrain emitted no JSON")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                value, end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if not text[index + end :].strip():
                return value
        raise GBrainProcessError("GBrain emitted corrupt JSON") from None


def document_markdown(document: NormalizedDocument) -> str:
    """Serialize a complete document, retaining stable provenance in frontmatter."""
    metadata = {
        **document.metadata,
        "source_id": document.source_id,
        "source_kind": document.source_kind.value,
        "canonical_url": document.canonical_url,
        "content_hash": document.content_hash,
    }
    frontmatter = "\n".join(
        f"{key}: {json.dumps(str(value), ensure_ascii=True)}" for key, value in metadata.items()
    )
    return f"---\n{frontmatter}\n---\n\n# {document.title}\n\n{document.text.rstrip()}\n"


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _grounded_retrieval_answer(
    search_evidence: Sequence[Any], query_evidence: Sequence[Any]
) -> str:
    for item in (*search_evidence, *query_evidence):
        if not isinstance(item, dict):
            continue
        mapping = cast(dict[str, Any], item)
        for key in ("chunk_text", "text", "content", "snippet"):
            value = mapping.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "No grounded retrieval evidence was returned."


@dataclass(frozen=True)
class _MeteringResult:
    events: list[dict[str, Any]]
    usage: dict[str, int] | None
    cost_usd: float | None
    status: str
    warnings: list[str]


def _reconcile_metering(
    events: Sequence[Mapping[str, Any]] | None,
    native_usage: dict[str, int] | None,
    *,
    question_index: int,
    question_count: int,
) -> _MeteringResult:
    if events is None:
        return _MeteringResult(
            [],
            None,
            None,
            "COST_INCOMPLETE",
            ["COST_INCOMPLETE: no measured proxy event ledger was supplied"],
        )

    selected: list[dict[str, Any]] = []
    recorded: list[dict[str, Any]] = []
    seen: set[str] = set()
    warnings: list[str] = []
    invalid = False
    for raw_event in events:
        event = dict(raw_event)
        event_id = event.get("event_id")
        recorded.append(event)
        if not isinstance(event_id, str) or not event_id:
            warnings.append("METERING_INVALID: event_id is required")
            invalid = True
            continue
        if event_id in seen:
            warnings.append(f"METERING_DUPLICATE: {event_id}")
            invalid = True
            continue
        seen.add(event_id)
        phase = event.get("phase")
        event_question = event.get("question_index")
        if phase == "ingest" or (
            phase == "query"
            and (
                event_question == question_index or (event_question is None and question_count == 1)
            )
        ):
            selected.append(event)

    phases = {event.get("phase") for event in selected}
    for required in ("ingest", "query"):
        if required not in phases:
            warnings.append(f"METERING_MISSING_PHASE: {required}")
            invalid = True

    total_input = 0
    total_output = 0
    total_usd = 0.0
    query_events: list[dict[str, Any]] = []
    for event in selected:
        input_tokens = event.get("input_tokens")
        output_tokens = event.get("output_tokens")
        usd = event.get("usd")
        if (
            isinstance(input_tokens, bool)
            or not isinstance(input_tokens, int)
            or input_tokens < 0
            or isinstance(output_tokens, bool)
            or not isinstance(output_tokens, int)
            or output_tokens < 0
            or isinstance(usd, bool)
            or not isinstance(usd, int | float)
            or usd < 0
        ):
            warnings.append(f"METERING_INVALID: {event.get('event_id')}")
            invalid = True
            continue
        total_input += input_tokens
        total_output += output_tokens
        total_usd += float(usd)
        model = event.get("model")
        if event.get("phase") == "ingest" and model not in {
            "text-embedding-3-small",
            EMBEDDING_MODEL,
        }:
            warnings.append(f"METERING_MODEL_MISMATCH: ingest event uses {model!r}")
            invalid = True
        if event.get("phase") == "query":
            query_events.append(event)
            if model not in {"gpt-5-mini", THINK_MODEL}:
                warnings.append(f"METERING_MODEL_MISMATCH: query event uses {model!r}")
                invalid = True

    comparable = [event for event in query_events if event.get("operation") == "think"]
    if not comparable and len(query_events) == 1:
        comparable = query_events
    if native_usage is not None and comparable:
        proxy_query_input = sum(int(event["input_tokens"]) for event in comparable)
        proxy_query_output = sum(int(event["output_tokens"]) for event in comparable)
        if (
            proxy_query_input != native_usage["input_tokens"]
            or proxy_query_output != native_usage["output_tokens"]
        ):
            warnings.append("METERING_USAGE_MISMATCH: native think usage differs from proxy")
            invalid = True
    elif native_usage is None:
        warnings.append("METERING_NATIVE_USAGE_MISSING: proxy events cannot reconcile native usage")
        invalid = True

    usage = {"input_tokens": total_input, "output_tokens": total_output} if selected else None
    return _MeteringResult(
        recorded,
        usage,
        total_usd if selected else None,
        "COST_INCOMPLETE" if invalid else "COST_COMPLETE",
        warnings,
    )


class GBrainAdapter:
    """Owns one run-local GBrain home and one exact pinned checkout."""

    def __init__(
        self,
        tools_root: Path,
        run_root: Path,
        *,
        bun: str = "bun",
        git: str = "git",
        timeout_seconds: float = 180.0,
        runner: Runner | None = None,
        config: GBrainExecutionConfig | None = None,
    ) -> None:
        self.tools_root = tools_root.resolve()
        self.run_root = run_root.resolve()
        self.bun = bun
        self.git = git
        self.timeout_seconds = timeout_seconds
        self.runner = runner or _default_runner
        self.config = config or GBrainExecutionConfig.quick_start()
        self.checkout = self.tools_root / f"gbrain-{GBRAIN_COMMIT}"
        self.home = self.run_root / "gbrain-home"
        self.sources = self.run_root / "sources"
        self._commands: list[CommandResult] = []
        self._holdout_markers: tuple[str, ...] = ()
        self._lifecycle_warnings: list[str] = []
        self._cancellation: RunCancellation | None = None

    def ensure_checkout(self) -> Path:
        self.tools_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._assert_checkout_confined()
        if not (self.checkout / ".git").is_dir():
            self._run(
                (self.git, "clone", "--no-checkout", GBRAIN_REPOSITORY, str(self.checkout)),
                self.tools_root,
            )
        self._assert_checkout_confined()
        self._run((self.git, "fetch", "--depth", "1", "origin", GBRAIN_COMMIT), self.checkout)
        self._run((self.git, "checkout", "--detach", GBRAIN_COMMIT), self.checkout)
        self._run((self.git, "reset", "--hard", GBRAIN_COMMIT), self.checkout)
        self._run((self.git, "clean", "-fdx"), self.checkout)
        head = self._run((self.git, "rev-parse", "HEAD"), self.checkout).stdout.strip()
        if head != GBRAIN_COMMIT:
            raise GBrainError(f"GBrain checkout is not pinned: {head}")
        package = json.loads((self.checkout / "package.json").read_text(encoding="utf-8"))
        if package.get("version") != GBRAIN_VERSION:
            raise GBrainError(f"GBrain version mismatch: {package.get('version')!r}")
        self._run(
            (self.bun, "install", "--frozen-lockfile"),
            self.checkout,
            timeout=self.timeout_seconds * 2,
        )
        return self.checkout

    def _assert_checkout_confined(self) -> None:
        if self.checkout.is_symlink():
            raise GBrainIsolationError(f"GBrain checkout cannot be a symlink: {self.checkout}")
        resolved = self.checkout.resolve(strict=False)
        if not resolved.is_relative_to(self.tools_root):
            raise GBrainIsolationError(
                f"GBrain checkout resolves outside tools root: {self.checkout}"
            )

    def _run(
        self,
        command: Sequence[str],
        cwd: Path,
        *,
        timeout: float | None = None,
        base_url: str | None = None,
    ) -> CommandResult:
        if (
            command
            and command[0] == self.bun
            and len(command) > 1
            and command[1].endswith("cli.ts")
            and Path(command[1]).name != "cli.ts"
        ):
            raise GBrainError("GBrain must be invoked through pinned src/cli.ts")
        if self.runner is _default_runner:
            result = run_process(
                command,
                cwd,
                self._environment(base_url),
                timeout or self.timeout_seconds,
                self._cancellation,
            )
        else:
            result = self.runner(
                command, cwd, self._environment(base_url), timeout or self.timeout_seconds
            )
        self._commands.append(result)
        if result.returncode != 0:
            detail = self._sanitize(result.stderr.strip()[-1000:] or result.stdout.strip()[-1000:])
            diagnostic = f"{result.stdout}\n{result.stderr}".lower()
            if "openai_api_key" in diagnostic and any(
                marker in diagnostic
                for marker in ("required", "missing", "unavailable", "not configured")
            ):
                raise GBrainMissingProviderError(
                    f"GBrain command failed ({result.returncode}): {detail}",
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            raise GBrainProcessError(f"GBrain command failed ({result.returncode}): {detail}")
        return result

    def _sanitize(self, value: str) -> str:
        secrets = [
            secret.get_secret_value()
            for secret in (self.config.credential, self.config.chat_credential)
            if secret is not None
        ]
        sanitized = value
        for secret in secrets:
            sanitized = sanitized.replace(secret, "[REDACTED]")
        return sanitized

    def _environment(self, base_url: str | None = None) -> dict[str, str]:
        provider_keys = {"OPENAI_API_KEY", "VOYAGE_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"}
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"GBRAIN_HOME", "GBRAIN_DATABASE_URL", *provider_keys}
            and "HOLDOUT" not in key.upper()
            and "ORACLE" not in key.upper()
            and not any(marker and marker in value for marker in self._holdout_markers)
        }
        env["GBRAIN_HOME"] = str(self.home)
        env["GBRAIN_SKIP_STARTUP_HOOKS"] = "1"
        env["NODE_ENV"] = env.get("NODE_ENV", "production")
        env.update(self.config.child_environment())
        endpoint = self.config.embedding.endpoint
        if endpoint:
            env["GBRAIN_EMBEDDING_BASE_URL"] = endpoint
        if base_url and self.config.chat_provider == "openai":
            env["OPENAI_BASE_URL"] = base_url
        if self.config.chat_credential is not None:
            env["OPENAI_API_KEY"] = self.config.chat_credential.get_secret_value()
        return env

    def _cli(self, *args: str) -> tuple[str, ...]:
        if not args or args[0] not in _ALLOWED_COMMANDS:
            raise GBrainError(f"unsupported native GBrain surface: {args[0] if args else ''}")
        if any(part.lower() in _FORBIDDEN_SURFACES for part in args):
            raise GBrainError("forbidden GBrain surface requested")
        return (self.bun, "src/cli.ts", *args)

    def _run_cli(self, *args: str, base_url: str | None = None) -> CommandResult:
        return self._run(
            self._cli(*args),
            self.checkout,
            timeout=self.timeout_seconds,
            base_url=base_url,
        )

    def _init_capabilities(self, *, base_url: str | None = None) -> frozenset[str]:
        """Read supported init flags from the pinned CLI instead of guessing."""
        help_result = self._run_cli("init", "--help", base_url=base_url)
        return frozenset(re.findall(r"--[a-z0-9-]+", help_result.stdout + help_result.stderr))

    def _init_args(self, capabilities: frozenset[str]) -> list[str]:
        args = ["init", "--pglite", "--non-interactive"]
        if self.config.keyword_only:
            required = "--no-embedding"
            if required not in capabilities:
                raise GBrainCapabilityError(
                    "GBRAIN_CAPABILITY_UNAVAILABLE: pinned GBrain init does not support "
                    f"keyword-only mode ({required})"
                )
            args.append(required)
        else:
            embedding = self.config.embedding
            if not {"--embedding-model", "--embedding-dimensions"}.issubset(capabilities):
                raise GBrainCapabilityError(
                    "GBRAIN_CAPABILITY_UNAVAILABLE: pinned GBrain init does not support "
                    "the configured semantic embedding contract"
                )
            assert embedding.model is not None
            assert embedding.dimensions is not None
            args.extend(
                [
                    "--embedding-model",
                    f"{embedding.provider.value}:{embedding.model}",
                    "--embedding-dimensions",
                    str(embedding.dimensions),
                ]
            )
        if self.config.chat_provider == "openai" and self.config.chat_model:
            if "--chat-model" not in capabilities:
                raise GBrainCapabilityError(
                    "GBRAIN_CAPABILITY_UNAVAILABLE: pinned GBrain init does not support "
                    "the configured chat model contract"
                )
            args.extend(["--chat-model", self.config.chat_model])
        args.append("--json")
        return args

    def prepare_sources(
        self, documents: Sequence[NormalizedDocument], holdout_markers: Sequence[str] = ()
    ) -> Path:
        self.sources.mkdir(mode=0o700, parents=True, exist_ok=True)
        markers = tuple(holdout_markers)
        for document in documents:
            markdown = document_markdown(document)
            if any(marker and marker in markdown for marker in markers):
                raise GBrainIsolationError(
                    f"holdout marker found in candidate source {document.source_id}"
                )
            slug = hashlib.sha256(document.source_id.encode()).hexdigest()[:24]
            path = self.sources / f"{slug}.md"
            path.write_text(markdown, encoding="utf-8")
        return self.sources

    def run(
        self,
        documents: Sequence[NormalizedDocument],
        questions: Sequence[str],
        *,
        holdout_markers: Sequence[str] = (),
        base_url: str | None = None,
        proxy_events: Sequence[Mapping[str, Any]] | None = None,
        strict_base_url: bool = False,
        cancellation: RunCancellation | None = None,
    ) -> list[GBrainResult]:
        self._cancellation = cancellation
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        if self.run_root.exists() and any(self.run_root.iterdir()):
            raise GBrainIsolationError(f"GBrain run directory is not empty: {self.run_root}")
        self._holdout_markers = tuple(holdout_markers)
        if any(
            marker and marker in question
            for marker in self._holdout_markers
            for question in questions
        ):
            raise GBrainIsolationError("holdout marker found in candidate question")
        self.run_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lifecycle_warnings = []
        self.ensure_checkout()
        if strict_base_url and base_url is not None and not self.base_url_supported():
            raise GBrainProcessError(
                "BASE_URL_UNSUPPORTED: pinned GBrain runtime cannot expose provider calls "
                "through the run-local metering boundary"
            )
        self.home.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.prepare_sources(documents, holdout_markers)
        gold_sources = provenance_map(documents)
        capabilities = self._init_capabilities(base_url=base_url)
        init_args = self._init_args(capabilities)
        self._run_cli(*init_args, base_url=base_url)
        self._run_cli("import", str(self.sources), "--json", base_url=base_url)
        try:
            self._run_cli(
                "sync",
                "--repo",
                str(self.sources),
                "--no-pull",
                "--json",
                base_url=base_url,
            )
        except GBrainProcessError as error:
            detail = str(error).lower()
            if "no commits in repo" not in detail and "not inside a git repository" not in detail:
                raise
            self._lifecycle_warnings.append(
                "SYNC_UNAVAILABLE: native sync requires a pre-existing git HEAD; "
                "import remains authoritative"
            )
        status = self._run_cli("status", "--json", base_url=base_url)
        status_json = parse_json_output(status.stdout)
        results: list[GBrainResult] = []
        for question_index, question in enumerate(questions):
            started = time.monotonic()
            search = self._run_cli("search", question, "--json", base_url=base_url)
            search_json = parse_json_output(search.stdout)
            query: CommandResult | None = None
            query_json: Any = []
            if self.config.semantic_enabled:
                query = self._run_cli("query", question, "--json", base_url=base_url)
                query_json = parse_json_output(query.stdout)
            think: CommandResult | None = None
            native: dict[str, Any] = {}
            if self.config.chat_provider == "openai" and self.config.chat_credential is not None:
                think_model = self.config.chat_model or THINK_MODEL
                think = self._run_cli(
                    "think", question, "--model", think_model, "--json", base_url=base_url
                )
                native_value = parse_json_output(think.stdout)
                if not isinstance(native_value, dict):
                    raise GBrainProcessError("GBrain think JSON is not an object")
                native = cast(dict[str, Any], native_value)
            usage_value = native.get("usage")
            if isinstance(usage_value, dict):
                usage_object = cast(dict[str, Any], usage_value)
                usage = {
                    "input_tokens": int(usage_object.get("input_tokens", 0)),
                    "output_tokens": int(usage_object.get("output_tokens", 0)),
                }
            else:
                usage = None
            citations_value = native.get("citations", [])
            gaps_value = native.get("gaps", [])
            warnings_value = native.get("warnings", [])
            cost_value = native.get("cost_usd")
            raw_citations = (
                cast(list[Any], citations_value) if isinstance(citations_value, list) else []
            )
            gaps = (
                [str(gap) for gap in cast(list[Any], gaps_value)]
                if isinstance(gaps_value, list)
                else []
            )
            pages_gathered = int(native.get("pagesGathered", 0))
            metering = _reconcile_metering(
                proxy_events,
                usage,
                question_index=question_index,
                question_count=len(questions),
            )
            search_evidence = (
                cast(list[Any], search_json) if isinstance(search_json, list) else [search_json]
            )
            query_evidence = (
                cast(list[Any], query_json) if isinstance(query_json, list) else [query_json]
            )
            citations = resolve_retrieved_source_ids(
                [*raw_citations, *search_evidence, *query_evidence],
                gold_sources,
            )
            gather_evidence = {
                "citations": raw_citations,
                "pages_gathered": pages_gathered,
                "gaps": gaps,
            }
            native_cost = float(cost_value) if isinstance(cost_value, int | float) else None
            base_url_supported = base_url is None or self.base_url_supported()
            metering_status = metering.status
            metering_warnings = list(metering.warnings)
            if proxy_events is not None and base_url is None:
                metering_status = "COST_INCOMPLETE"
                metering_warnings.append(
                    "METERING_BASE_URL_MISSING: proxy events are not attributable to this run"
                )
            if base_url is not None and not base_url_supported:
                metering_status = "COST_INCOMPLETE"
            grounded_answer = str(native.get("answer", "")) or _grounded_retrieval_answer(
                search_evidence, query_evidence
            )
            result = GBrainResult(
                status="OK",
                answer=grounded_answer,
                evidence=citations,
                search_evidence=search_evidence,
                query_evidence=query_evidence,
                gather_evidence=gather_evidence,
                citations=citations,
                gaps=gaps,
                pages_gathered=pages_gathered,
                model_used=str(native.get("modelUsed", THINK_MODEL)),
                usage=usage,
                cost_usd=metering.cost_usd if metering.cost_usd is not None else native_cost,
                timings_ms={
                    "search": search.elapsed_ms,
                    "query": query.elapsed_ms if query is not None else 0,
                    "think": think.elapsed_ms if think is not None else 0,
                    "total_query": round((time.monotonic() - started) * 1000),
                },
                warnings=self._lifecycle_warnings
                + (
                    [str(warning) for warning in cast(list[Any], warnings_value)]
                    if isinstance(warnings_value, list)
                    else []
                )
                + metering_warnings
                + [
                    "RAW_GATHER_UNAVAILABLE: this release exposes citations, pagesGathered, "
                    "and gaps from think, but not raw gathered pages"
                ]
                + (
                    ["BASE_URL_UNSUPPORTED: pinned OpenAI provider ignores the metering URL"]
                    if base_url is not None and not self.base_url_supported()
                    else []
                ),
                files=[
                    str(path.relative_to(self.run_root))
                    for root in (self.home, self.sources)
                    for path in root.rglob("*.md")
                ],
                footprint_bytes=_directory_size(self.home),
                native={
                    "search": search_json,
                    "query": query_json,
                    "think": native,
                    "status": status_json,
                    "execution_config": self.config.safe_metadata(),
                },
                commands=list(self._commands),
                cost_status=metering_status,
                base_url_supported=base_url_supported,
                proxy_events=metering.events,
                proxy_usage=metering.usage,
                keyword_only=self.config.keyword_only,
                semantic_enabled=self.config.semantic_enabled,
                semantic_quality=("not_measured" if self.config.keyword_only else "configured"),
                recommendation_eligible=self.config.recommendation_eligible,
                execution_config=self.config.safe_metadata(),
            )
            results.append(result)
        return results

    def base_url_supported(self) -> bool:
        """Detect support from the pinned provider implementation, not an estimate."""
        gateway = self.checkout / "src/core/ai/gateway.ts"
        return gateway.exists() and "OPENAI_BASE_URL" in gateway.read_text(encoding="utf-8")

    def cleanup(self) -> CleanupReceipt:
        """Remove only this adapter's run-local process/state; never global GBrain state."""
        removed: list[str] = []
        try:
            if self.home.exists():
                for path in sorted(self.home.rglob("*"), reverse=True):
                    if path.is_file() or path.is_symlink():
                        path.unlink(missing_ok=True)
                    elif path.is_dir():
                        path.rmdir()
                self.home.rmdir()
                removed.append(str(self.home))
            return CleanupReceipt(
                candidate=CandidateId.GBRAIN,
                removed_paths=removed,
                remaining_paths=list(remaining_paths(self.home)),
            )
        except KeyboardInterrupt:
            return CleanupReceipt(candidate=CandidateId.GBRAIN, interrupted=True)
        except Exception as exc:
            return CleanupReceipt(
                candidate=CandidateId.GBRAIN, removed_paths=removed, error=str(exc)
            )
