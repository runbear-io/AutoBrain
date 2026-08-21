"""Run-local adapter for the pinned Mem0 OSS runtime."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import threading
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence, Set
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast
from urllib.parse import urlsplit

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

if TYPE_CHECKING:
    Memory: Any
    _mem0_main: Any
    _mem0_telemetry: Any
else:
    # Keep the OSS import explicit; no Platform/MemoryClient surface is used.
    from mem0 import Memory
    from mem0.memory import main as _mem0_main
    from mem0.memory import telemetry as _mem0_telemetry

    # Also cover a process where another module imported Mem0 before this adapter.
    _mem0_main.MEM0_TELEMETRY = False
    _mem0_telemetry.MEM0_TELEMETRY = False
from pydantic import Field, ValidationError, field_validator

from autobrain.cancellation import RunCancellation
from autobrain.models import NormalizedDocument, SourceId, Status, StrictModel
from autobrain.secrets import RuntimeEnvironment

_CHAT_MODEL = "gpt-5-mini"
_EMBEDDING_MODEL = "text-embedding-3-small"
_DEFAULT_TOP_K = 8
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_JSON_FENCE = re.compile(r"\A\s*```(?:json)?\s*\n(?P<payload>.*?)\n```\s*\Z", re.DOTALL)
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bxox[a-z]-[A-Za-z0-9-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(
        r"(?i)\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|client[_ -]?secret|"
        r"password|authorization|credential)\s*[=:]\s*[A-Za-z0-9._~+/=-]{8,}"
    ),
    re.compile(r"//[^/\s:@]+:[^@\s]+@"),
)
_T = TypeVar("_T")


def _redacted_url(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        parsed = urlsplit(value)
        return f"{parsed.scheme or 'configured'}://[REDACTED]"
    except ValueError:
        return "[REDACTED_URL]"


def _error_class(error: BaseException) -> str:
    return type(error).__name__


def _structured_answer_object(content: str) -> dict[str, Any]:
    candidate = content
    if fence := _JSON_FENCE.fullmatch(content):
        candidate = fence.group("payload")
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise TypeError("structured answer must be a JSON object")
    return cast(dict[str, Any], payload)


class Mem0AdapterError(RuntimeError):
    """Base error for Mem0 adapter failures."""


class Mem0PersistenceError(Mem0AdapterError):
    """The run-local stores could not be initialized safely."""


class Mem0MissingProviderError(Mem0AdapterError):
    """The canonical provider credential is unavailable before adapter construction."""

    status = Status.MISSING_PROVIDER

    def __init__(self) -> None:
        super().__init__("MISSING_PROVIDER: OPENAI_API_KEY is not configured")


class Mem0SecretBoundaryError(Mem0AdapterError):
    """Credential-bearing data was rejected before crossing a storage/provider boundary."""


class Mem0OperationTimeout(Mem0AdapterError):
    """A bounded Mem0 native or provider operation exceeded its deadline."""

    def __init__(
        self, operation: str, timeout_seconds: float, *, unavailable: bool = False
    ) -> None:
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.unavailable = unavailable
        detail = "hard timeout unavailable" if unavailable else "operation timed out"
        super().__init__(f"Mem0 {detail} during {operation} after {timeout_seconds:g}s")


class StructuredAnswerError(Mem0AdapterError):
    """The answer model did not return a valid, evidence-bound answer."""


class _EvidenceRetryRequired(ValueError):
    """The first answer is otherwise valid but omitted required evidence IDs."""


class Mem0CleanupError(Mem0AdapterError):
    """Native cleanup failed and retained state remains inspectable."""

    def __init__(self, error_type: str, retained_paths: Sequence[Path]) -> None:
        self.error_type = error_type
        self.retained_paths = tuple(retained_paths)
        joined_paths = ", ".join(str(path) for path in retained_paths)
        super().__init__(f"Mem0 cleanup failed ({error_type}); retained state is at {joined_paths}")


class _MemoryApi(Protocol):
    def add(self, messages: str, **kwargs: Any) -> Mapping[str, Any]: ...

    def search(self, query: str, **kwargs: Any) -> Mapping[str, Any]: ...

    def get(self, memory_id: str) -> Mapping[str, Any] | None: ...

    def get_all(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def delete_all(self, **kwargs: Any) -> Mapping[str, Any]: ...


class Mem0UsageEvent(StrictModel):
    phase: str
    model: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    request_id: str | None = None
    source: str


class Mem0UsageAction(StrEnum):
    APPEND = "APPEND"
    UPSERT = "UPSERT"


class Mem0UsageUpdate(StrictModel):
    """Sink envelope: UPSERT replaces the prior event with the same request ID."""

    action: Mem0UsageAction
    request_id: str | None
    revision: int = Field(ge=1)
    event: Mem0UsageEvent


class Mem0IngestResult(StrictModel):
    native_results: list[dict[str, Any]]
    memory_ids: list[str]


class Mem0CleanupReceipt(StrictModel):
    already_clean: bool = False
    closed_resources: list[str] = Field(default_factory=list)
    removed_paths: list[str] = Field(default_factory=list)


class Mem0TimeoutEvidence(StrictModel):
    timeout_seconds: float = Field(gt=0)
    configured_native_clients: list[str]
    hard_operation_deadline: str
    upstream_config_limitation: str


class Mem0TokenUsage(StrictModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class Mem0Answer(StrictModel):
    answer: str = Field(min_length=1)
    claim_ids: list[str] = Field(min_length=1)
    source_ids: list[SourceId] = Field(min_length=1)
    usage: Mem0TokenUsage | None = None
    usage_available: bool

    @field_validator("answer")
    @classmethod
    def answer_is_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer cannot be blank")
        return value

    @field_validator("claim_ids")
    @classmethod
    def claim_ids_are_nonempty(cls, values: list[str]) -> list[str]:
        if not values or any(not value.strip() for value in values):
            raise ValueError("claim IDs cannot be blank")
        if len(values) != len(set(values)):
            raise ValueError("claim IDs must be unique")
        return values

    @field_validator("source_ids")
    @classmethod
    def source_ids_are_unique(cls, values: list[SourceId]) -> list[SourceId]:
        if not values:
            raise ValueError("at least one source ID is required")
        if len(values) != len(set(values)):
            raise ValueError("source IDs must be unique")
        return values


class Mem0AdapterConfig:
    """Validated run-scoped configuration with credential-redacted representation."""

    def __init__(
        self,
        *,
        run_id: str,
        run_dir: Path,
        heldout_source_ids: Set[str],
        known_secrets: Set[str] = frozenset(),
        forbidden_markers: Set[str] = frozenset(),
        api_key: str | None = None,
        base_url: str | None = None,
        top_k: int = _DEFAULT_TOP_K,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not _SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError("run_id must contain only letters, digits, underscores, and hyphens")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self.run_id = run_id
        self.run_dir = run_dir.expanduser().resolve()
        suffix = hashlib.sha256(run_id.encode()).hexdigest()[:12]
        slug = re.sub(r"[^a-z0-9-]", "-", run_id.lower()).strip("-")[:48] or "run"
        self.user_id = f"autobrain-{slug}-{suffix}-user"
        self.agent_id = f"autobrain-{slug}-{suffix}-mem0"
        self.qdrant_path = self.run_dir / "qdrant"
        self.history_db_path = self.run_dir / "history.db"
        self.top_k = top_k
        self.timeout_seconds = timeout_seconds
        self.heldout_source_ids = frozenset(heldout_source_ids)
        self.known_secrets = frozenset(secret for secret in known_secrets if secret)
        self.forbidden_markers = frozenset(forbidden_markers)
        self._api_key = api_key
        self.base_url = base_url
        self.native_timeout_strategy = (
            "OpenAI clients are rebuilt with timeout_seconds; synchronous Mem0 operations are "
            "additionally bounded by SIGALRM on the CLI main thread because mem0ai 2.0.18 "
            "does not expose a supported LLM/embedder timeout config field."
        )
        self.memory_config: dict[str, Any] = {
            "version": "v1.1",
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": f"autobrain_{suffix}",
                    "path": str(self.qdrant_path),
                    "embedding_model_dims": 1536,
                    "on_disk": True,
                },
            },
            "history_db_path": str(self.history_db_path),
            "llm": {
                "provider": "openai",
                "config": {
                    "model": _CHAT_MODEL,
                    "api_key": api_key,
                    "openai_base_url": base_url,
                    "temperature": 0,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": _EMBEDDING_MODEL,
                    "api_key": api_key,
                    "openai_base_url": base_url,
                    "embedding_dims": 1536,
                },
            },
        }
        self.answer_client_kwargs: dict[str, Any] = {"timeout": timeout_seconds}
        if api_key is not None:
            self.answer_client_kwargs["api_key"] = api_key
        if base_url is not None:
            self.answer_client_kwargs["base_url"] = base_url

    def provider_api_key(self) -> str | None:
        """Return the configured key only for constructing the local provider clients."""
        return self._api_key

    def __repr__(self) -> str:
        return (
            "Mem0AdapterConfig("
            f"run_id={self.run_id!r}, run_dir={str(self.run_dir)!r}, "
            f"base_url={_redacted_url(self.base_url)!r}, top_k={self.top_k!r}, "
            "api_key=<redacted>)"
        )


class Mem0Adapter:
    """Small OSS-only Mem0 adapter preserving native retrieval as first-class evidence."""

    def __init__(
        self,
        config: Mem0AdapterConfig,
        *,
        usage_sink: Callable[[Mem0UsageUpdate], None] | None = None,
    ) -> None:
        self.config = config
        self._usage_sink = usage_sink
        self._known_secrets = frozenset(
            {
                *config.known_secrets,
                *RuntimeEnvironment.from_environ(os.environ).known_secret_values(),
                *(secret for secret in (config.provider_api_key(),) if secret),
            }
        )
        if not (config.provider_api_key() or os.environ.get("OPENAI_API_KEY")):
            raise Mem0MissingProviderError()
        self.usage_events: list[Mem0UsageEvent] = []
        self._usage_event_indexes: dict[str, int] = {}
        self._usage_revisions: dict[str, int] = {}
        self._closed_resource_ids: set[int] = set()
        self._closed_resource_names: list[str] = []
        self._native_deleted = False
        self._ingest_sources: dict[str, str] = {}
        self._cleanup_receipt: Mem0CleanupReceipt | None = None
        self._prepare_local_stores()
        self.config.memory_config["llm"]["config"]["response_callback"] = self._mem0_callback
        memory: Any | None = None
        try:
            factory = getattr(Memory, "from_config", None)
            memory = factory(self.config.memory_config) if callable(factory) else Memory()
            self._memory = cast(_MemoryApi, memory)
            configured_native_clients = self._configure_native_provider_timeouts(memory)
            self._answer_client = OpenAI(**self.config.answer_client_kwargs)
            self.timeout_evidence = Mem0TimeoutEvidence(
                timeout_seconds=self.config.timeout_seconds,
                configured_native_clients=configured_native_clients,
                hard_operation_deadline="SIGALRM on CLI main thread",
                upstream_config_limitation=(
                    "mem0ai 2.0.18 exposes no supported OpenAI LLM/embedder timeout config field"
                ),
            )
        except Exception as exc:
            if memory is not None:
                self._close_failed_initialization(memory)
            raise Mem0PersistenceError(
                f"failed to initialize run-local Mem0 stores ({_error_class(exc)})"
            ) from None

    @staticmethod
    def _close_failed_initialization(memory: Any) -> None:
        vector_store = getattr(memory, "vector_store", None)
        llm = getattr(memory, "llm", None)
        embedder = getattr(memory, "embedding_model", None)
        resources = (
            getattr(vector_store, "client", None),
            getattr(memory, "db", None),
            getattr(llm, "client", None),
            getattr(embedder, "client", None),
        )
        for resource in resources:
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    continue

    def _prepare_local_stores(self) -> None:
        try:
            self.config.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            if self.config.history_db_path.exists():
                raise Mem0PersistenceError(
                    f"refusing dirty Mem0 history store: {self.config.history_db_path}"
                )
            if self.config.qdrant_path.exists() and any(self.config.qdrant_path.iterdir()):
                raise Mem0PersistenceError(
                    f"refusing dirty Mem0 vector store: {self.config.qdrant_path}"
                )
            self.config.qdrant_path.mkdir(mode=0o700, exist_ok=True)
        except Mem0PersistenceError:
            raise
        except OSError as exc:
            raise Mem0PersistenceError(
                f"could not create run-local Mem0 stores ({_error_class(exc)})"
            ) from None

    def _configure_native_provider_timeouts(self, memory: Any) -> list[str]:
        configured: list[str] = []
        for attribute in ("embedding_model", "llm"):
            component = getattr(memory, attribute, None)
            if component is None or not type(component).__module__.startswith("mem0."):
                continue
            previous_client = getattr(component, "client", None)
            replacement = OpenAI(**self.config.answer_client_kwargs)
            component.client = replacement
            close = getattr(previous_client, "close", None)
            if callable(close):
                close()
            configured.append(attribute)
        return configured

    @contextmanager
    def _operation_deadline(self, operation: str) -> Generator[None]:
        if threading.current_thread() is not threading.main_thread() or not hasattr(
            signal, "SIGALRM"
        ):
            raise Mem0OperationTimeout(operation, self.config.timeout_seconds, unavailable=True)

        def expire(_signum: int, _frame: Any) -> None:
            raise Mem0OperationTimeout(operation, self.config.timeout_seconds)

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, expire)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, self.config.timeout_seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)

    def _bounded(
        self,
        operation: str,
        call: Callable[[], _T],
        cancellation: RunCancellation | None = None,
    ) -> _T:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        with self._operation_deadline(operation):
            result = call()
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        return result

    def _assert_no_secrets(self, value: Any, *, boundary: str) -> None:
        serialized = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
        known_secret_found = any(secret in serialized for secret in self._known_secrets)
        patterned_secret_found = any(pattern.search(serialized) for pattern in _SECRET_PATTERNS)
        if known_secret_found or patterned_secret_found:
            raise Mem0SecretBoundaryError(
                f"credential-bearing data rejected before Mem0 {boundary} boundary"
            )

    def ingest(
        self,
        documents: Iterable[NormalizedDocument],
        *,
        heldout_source_ids: set[str] | None = None,
        forbidden_markers: set[str] | None = None,
        cancellation: RunCancellation | None = None,
    ) -> Mem0IngestResult:
        heldout = set(self.config.heldout_source_ids)
        heldout.update(heldout_source_ids or set())
        markers = set(self.config.forbidden_markers)
        markers.update(forbidden_markers or set())
        self._ingest_sources.clear()
        native_results: list[dict[str, Any]] = []
        memory_ids: list[str] = []
        seen_source_ids: set[str] = set()
        for document in documents:
            self._assert_no_secrets(document.model_dump(mode="json"), boundary="native ingest")
            self._assert_candidate_safe(document, heldout, markers)
            if document.source_id in seen_source_ids:
                raise ValueError(f"duplicate source ID during Mem0 ingest: {document.source_id}")
            seen_source_ids.add(document.source_id)
            metadata = self._document_metadata(document)
            try:
                response = self._bounded(
                    "add",
                    lambda document=document, metadata=metadata: self._memory.add(
                        document.text,
                        user_id=self.config.user_id,
                        agent_id=self.config.agent_id,
                        run_id=self.config.run_id,
                        metadata=metadata,
                        infer=True,
                    ),
                    cancellation,
                )
            except Mem0OperationTimeout:
                raise
            except Exception as exc:
                raise Mem0PersistenceError(
                    f"Mem0 failed to persist whole document {document.source_id} "
                    f"({_error_class(exc)})"
                ) from None
            results = self._results(response, operation="add")
            if not results:
                response = self._bounded(
                    "add",
                    lambda document=document, metadata=metadata: self._memory.add(
                        document.text,
                        user_id=self.config.user_id,
                        agent_id=self.config.agent_id,
                        run_id=self.config.run_id,
                        metadata=metadata,
                        infer=False,
                    ),
                    cancellation,
                )
                results = self._results(response, operation="add")
            self._assert_no_secrets(results, boundary="native add artifact emission")
            for item in results:
                memory_id = item.get("id")
                if isinstance(memory_id, str):
                    self._ingest_sources[memory_id] = document.source_id
            self._map_unattributed_memories(document.source_id)
            native_results.extend(results)
            memory_ids.extend(
                str(item["id"]) for item in results if isinstance(item.get("id"), str)
            )
        return Mem0IngestResult(native_results=native_results, memory_ids=memory_ids)

    def search_native(
        self,
        question: str,
        *,
        top_k: int | None = None,
        cancellation: RunCancellation | None = None,
    ) -> dict[str, Any]:
        self._assert_no_secrets(question, boundary="native search prompt")
        limit = self.config.top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k must be positive")
        try:
            response = self._bounded(
                "search",
                lambda: self._memory.search(
                    question,
                    top_k=limit,
                    filters=self._filters,
                    threshold=0.0,
                    rerank=False,
                ),
                cancellation,
            )
        except Mem0OperationTimeout:
            raise
        except Exception as exc:
            raise Mem0AdapterError(f"native Mem0 search failed ({_error_class(exc)})") from None
        results = self._results(response, operation="search")
        self._assert_no_secrets(results, boundary="native search artifact emission")
        results.sort(key=self._deterministic_result_key)
        return {"results": results[:limit]}

    def filter_native_results(
        self,
        results: Sequence[Mapping[str, Any]],
        *,
        heldout_source_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        self._assert_no_secrets(results, boundary="native evidence")
        heldout = set(self.config.heldout_source_ids)
        heldout.update(heldout_source_ids or set())
        filtered: list[dict[str, Any]] = []
        for result in results:
            source_id = self._result_source_id(result)
            if source_id is not None and source_id in heldout:
                raise ValueError(f"held-out source reached Mem0 retrieval: {source_id}")
            filtered.append(dict(result))
        return filtered

    def answer(
        self,
        question: str,
        native_results: Sequence[Mapping[str, Any]],
        *,
        cancellation: RunCancellation | None = None,
    ) -> Mem0Answer:
        self._assert_no_secrets(question, boundary="answer provider prompt")
        validated_results = self.filter_native_results(native_results)
        self._assert_no_secrets(validated_results, boundary="answer provider evidence prompt")
        evidence = [self._answer_evidence(result) for result in validated_results]
        memory_sources: dict[str, set[str]] = {}
        allowed_sources: set[str] = set()
        for result in validated_results:
            memory_id = result.get("id")
            source_id = self._result_source_id(result)
            if source_id is not None:
                allowed_sources.add(source_id)
                if isinstance(memory_id, str):
                    memory_sources.setdefault(memory_id, set()).add(source_id)
        if not allowed_sources:
            raise StructuredAnswerError("Mem0 answer requires at least one retrieved source")
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied native Mem0 results. Return one JSON object "
                    "with string answer, string-array claim_ids, and string-array source_ids. "
                    "Use only source_ids present in the evidence."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "native_mem0_results": evidence},
                    ensure_ascii=True,
                    sort_keys=True,
                ),
            },
        ]
        try:
            response = self._answer_response("answer", messages, cancellation)
            try:
                return self._parse_answer_response(
                    response,
                    memory_sources=memory_sources,
                    allowed_sources=allowed_sources,
                )
            except _EvidenceRetryRequired:
                retry_instruction: ChatCompletionMessageParam = {
                    "role": "system",
                    "content": (
                        "Retry once. Return evidence-backed claim_ids and source_ids from the "
                        "same supplied native Mem0 evidence. Do not infer citations or use any "
                        "source outside that evidence. If evidence is insufficient, honestly "
                        "state that you cannot answer while still returning the required JSON "
                        "object with evidence-backed claim_ids and source_ids."
                    ),
                }
                retry_messages = [*messages, retry_instruction]
                retry_response = self._answer_response("answer-retry", retry_messages, cancellation)
                try:
                    return self._parse_answer_response(
                        retry_response,
                        memory_sources=memory_sources,
                        allowed_sources=allowed_sources,
                    )
                except _EvidenceRetryRequired:
                    raise StructuredAnswerError(
                        "invalid structured Mem0 answer (missing evidence IDs)"
                    ) from None
        except (StructuredAnswerError, Mem0OperationTimeout):
            raise
        except (
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            ValidationError,
            json.JSONDecodeError,
        ) as exc:
            raise StructuredAnswerError(
                f"invalid structured Mem0 answer ({_error_class(exc)})"
            ) from None
        except Exception as exc:
            raise StructuredAnswerError(
                f"Mem0 answer provider failed ({_error_class(exc)})"
            ) from None

    def _answer_response(
        self,
        operation: str,
        messages: list[ChatCompletionMessageParam],
        cancellation: RunCancellation | None,
    ) -> Any:
        response = self._bounded(
            operation,
            lambda: self._answer_client.chat.completions.create(
                model=_CHAT_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            ),
            cancellation,
        )
        self._record_usage_from_response(response, phase=operation, source="answer_callback")
        return response

    def _parse_answer_response(
        self,
        response: Any,
        *,
        memory_sources: Mapping[str, set[str]],
        allowed_sources: set[str],
    ) -> Mem0Answer:
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise StructuredAnswerError("answer model returned no text")
        payload = _structured_answer_object(content)
        if set(payload) != {"answer", "claim_ids", "source_ids"}:
            raise ValueError("structured answer has unsupported or missing fields")
        answer: Any = payload["answer"]
        claim_ids: Any = payload["claim_ids"]
        source_ids: Any = payload["source_ids"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("structured answer text must be nonblank")
        if not isinstance(claim_ids, list):
            raise TypeError("claim_ids must be a string array")
        raw_claim_ids = cast(list[Any], claim_ids)
        if any(not isinstance(value, str) for value in raw_claim_ids):
            raise TypeError("claim_ids must be a string array")
        typed_claim_ids = cast(list[str], raw_claim_ids)
        if not isinstance(source_ids, list):
            raise TypeError("source_ids must be a string array")
        raw_source_ids = cast(list[Any], source_ids)
        if any(not isinstance(value, str) for value in raw_source_ids):
            raise TypeError("source_ids must be a string array")
        typed_source_ids = cast(list[str], raw_source_ids)

        blank_claims = not typed_claim_ids or all(not value.strip() for value in typed_claim_ids)
        blank_sources = not typed_source_ids or all(not value.strip() for value in typed_source_ids)
        if not blank_claims:
            if any(not value.strip() for value in typed_claim_ids):
                raise ValueError("claim IDs cannot mix blank and nonblank values")
            if len(typed_claim_ids) != len(set(typed_claim_ids)):
                raise ValueError("claim IDs must be unique")

        normalized_sources: list[str] = []
        if not blank_sources:
            if any(not value.strip() for value in typed_source_ids):
                raise ValueError("source IDs cannot mix blank and nonblank values")
            for value in typed_source_ids:
                if value in allowed_sources:
                    source_id = value
                else:
                    candidates = memory_sources.get(value, set())
                    if len(candidates) != 1:
                        raise ValueError("answer cited unknown or ambiguous native evidence")
                    source_id = next(iter(candidates))
                if source_id in normalized_sources:
                    raise ValueError("source IDs must be unique")
                normalized_sources.append(source_id)
        if blank_claims or blank_sources:
            raise _EvidenceRetryRequired()

        parsed = Mem0Answer.model_validate(
            {
                **payload,
                "source_ids": normalized_sources,
                "usage": self._usage_cost(response),
                "usage_available": getattr(response, "usage", None) is not None,
            }
        )
        if set(parsed.source_ids) - allowed_sources:
            raise StructuredAnswerError("answer cited source IDs absent from native retrieval")
        return parsed

    def get(self, memory_id: str) -> dict[str, Any] | None:
        result = self._bounded("get", lambda: self._memory.get(memory_id))
        if result is None:
            return None
        emitted = dict(result)
        self._assert_no_secrets(emitted, boundary="native get artifact emission")
        return emitted

    def get_all(self, *, top_k: int = 10_000) -> dict[str, Any]:
        response = self._bounded(
            "get_all", lambda: self._memory.get_all(filters=self._filters, top_k=top_k)
        )
        results = self._results(response, operation="get_all")
        self._assert_no_secrets(results, boundary="native get_all artifact emission")
        results.sort(key=self._deterministic_result_key)
        return {"results": results}

    def record_proxy_event(self, event: Mapping[str, Any], *, phase: str) -> None:
        request_id_value = event.get("request_id") or event.get("id")
        request_id = str(request_id_value) if request_id_value is not None else None
        usage = event.get("usage")
        usage_map: Mapping[str, Any] = (
            cast(Mapping[str, Any], usage) if isinstance(usage, Mapping) else {}
        )
        self._emit_usage(
            Mem0UsageEvent(
                phase=phase,
                model=str(event["model"]) if event.get("model") is not None else None,
                input_tokens=self._optional_int(
                    usage_map.get("prompt_tokens", usage_map.get("input_tokens"))
                ),
                output_tokens=self._optional_int(
                    usage_map.get("completion_tokens", usage_map.get("output_tokens"))
                ),
                total_tokens=self._optional_int(usage_map.get("total_tokens")),
                request_id=request_id,
                source="proxy",
            )
        )

    def cleanup(self) -> Mem0CleanupReceipt:
        if self._cleanup_receipt is not None:
            return self._cleanup_receipt.model_copy(update={"already_clean": True})
        removed_paths: list[str] = []
        try:
            if not self._native_deleted:
                self._bounded(
                    "delete_all", lambda: self._memory.delete_all(run_id=self.config.run_id)
                )
                remaining_response = self._bounded(
                    "cleanup verification",
                    lambda: self._memory.get_all(filters=self._filters, top_k=1),
                )
                remaining = self._results(remaining_response, operation="get_all")
                if remaining:
                    raise RuntimeError("native delete_all left scoped memories behind")
                self._native_deleted = True
            for name, resource in self._native_resources():
                resource_id = id(resource)
                if resource_id in self._closed_resource_ids:
                    continue
                close = getattr(resource, "close", None)
                if not callable(close):
                    continue
                self._bounded(f"close {name}", close)
                self._closed_resource_ids.add(resource_id)
                self._closed_resource_names.append(name)
            if self.config.qdrant_path.exists():
                shutil.rmtree(self.config.qdrant_path)
                removed_paths.append(str(self.config.qdrant_path))
            if self.config.history_db_path.exists():
                self.config.history_db_path.unlink()
                removed_paths.append(str(self.config.history_db_path))
        except Exception as exc:
            raise Mem0CleanupError(
                _error_class(exc),
                [self.config.qdrant_path, self.config.history_db_path],
            ) from None
        self._cleanup_receipt = Mem0CleanupReceipt(
            closed_resources=self._closed_resource_names,
            removed_paths=removed_paths,
        )
        return self._cleanup_receipt

    def _native_resources(self) -> list[tuple[str, Any]]:
        vector_store = getattr(self._memory, "vector_store", None)
        llm = getattr(self._memory, "llm", None)
        embedder = getattr(self._memory, "embedding_model", None)
        return [
            ("qdrant", getattr(vector_store, "client", None)),
            ("sqlite", getattr(self._memory, "db", None)),
            ("mem0_llm", getattr(llm, "client", None)),
            ("mem0_embedder", getattr(embedder, "client", None)),
            ("answer_client", self._answer_client),
        ]

    def __enter__(self) -> Mem0Adapter:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.cleanup()

    @property
    def _filters(self) -> dict[str, str]:
        return {
            "user_id": self.config.user_id,
            "agent_id": self.config.agent_id,
            "run_id": self.config.run_id,
        }

    def _mem0_callback(self, _llm: Any, response: Any, params: Mapping[str, Any]) -> None:
        phase = str(params.get("autobrain_phase", "ingest"))
        self._record_usage_from_response(response, phase=phase, source="mem0_callback")

    def _record_usage_from_response(self, response: Any, *, phase: str, source: str) -> None:
        usage = getattr(response, "usage", None)
        self._emit_usage(
            Mem0UsageEvent(
                phase=phase,
                model=self._response_model(response),
                input_tokens=self._usage_value(usage, "prompt_tokens", "input_tokens"),
                output_tokens=self._usage_value(usage, "completion_tokens", "output_tokens"),
                total_tokens=self._usage_value(usage, "total_tokens"),
                request_id=self._response_id(response),
                source=source,
            )
        )

    def _emit_usage(self, event: Mem0UsageEvent) -> None:
        if event.request_id is not None and event.request_id in self._usage_event_indexes:
            index = self._usage_event_indexes[event.request_id]
            current = self.usage_events[index]
            source_order = {"mem0_callback": 0, "answer_callback": 1, "proxy": 2}
            sources = sorted(
                {*current.source.split("+"), *event.source.split("+")},
                key=lambda source: (source_order.get(source, 99), source),
            )
            current_has_proxy = "proxy" in current.source.split("+")
            incoming_is_proxy = event.source == "proxy"
            preferred = event if incoming_is_proxy or not current_has_proxy else current
            fallback = current if preferred is event else event
            merged = Mem0UsageEvent(
                phase=preferred.phase,
                model=preferred.model or fallback.model,
                input_tokens=preferred.input_tokens
                if preferred.input_tokens is not None
                else fallback.input_tokens,
                output_tokens=preferred.output_tokens
                if preferred.output_tokens is not None
                else fallback.output_tokens,
                total_tokens=preferred.total_tokens
                if preferred.total_tokens is not None
                else fallback.total_tokens,
                request_id=event.request_id,
                source="+".join(sources),
            )
            self.usage_events[index] = merged
            self._publish_usage_update(merged)
            return
        if event.request_id is not None:
            self._usage_event_indexes[event.request_id] = len(self.usage_events)
        self.usage_events.append(event)
        self._publish_usage_update(event)

    def _publish_usage_update(self, event: Mem0UsageEvent) -> None:
        if self._usage_sink is None:
            return
        if event.request_id is None:
            update = Mem0UsageUpdate(
                action=Mem0UsageAction.APPEND,
                request_id=None,
                revision=1,
                event=event,
            )
        else:
            revision = self._usage_revisions.get(event.request_id, 0) + 1
            self._usage_revisions[event.request_id] = revision
            update = Mem0UsageUpdate(
                action=Mem0UsageAction.UPSERT,
                request_id=event.request_id,
                revision=revision,
                event=event,
            )
        self._usage_sink(update)

    @staticmethod
    def _usage_cost(response: Any) -> Mem0TokenUsage | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        input_tokens = Mem0Adapter._usage_value(usage, "prompt_tokens", "input_tokens")
        output_tokens = Mem0Adapter._usage_value(usage, "completion_tokens", "output_tokens")
        if input_tokens is None or output_tokens is None:
            return None
        # Pricing is reconciled centrally; token evidence must not imply a zero-dollar cost.
        return Mem0TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)

    @staticmethod
    def _usage_value(usage: Any, *names: str) -> int | None:
        if usage is None:
            return None
        for name in names:
            value = (
                cast(Mapping[str, Any], usage).get(name)
                if isinstance(usage, Mapping)
                else getattr(usage, name, None)
            )
            parsed = Mem0Adapter._optional_int(value)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return (
            value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
        )

    @staticmethod
    def _response_model(response: Any) -> str | None:
        value = getattr(response, "model", None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _response_id(response: Any) -> str | None:
        value = getattr(response, "id", None)
        return value if isinstance(value, str) else None

    @staticmethod
    def _results(response: Mapping[str, Any], *, operation: str) -> list[dict[str, Any]]:
        raw = response.get("results")
        if not isinstance(raw, list):
            raise Mem0AdapterError(f"native Mem0 {operation} response has no results list")
        typed_raw = cast(list[Any], raw)
        if any(not isinstance(item, Mapping) for item in typed_raw):
            raise Mem0AdapterError(f"native Mem0 {operation} returned malformed result metadata")
        typed_results = cast(list[Mapping[str, Any]], typed_raw)
        return [dict(item) for item in typed_results]

    @staticmethod
    def _document_metadata(document: NormalizedDocument) -> dict[str, Any]:
        metadata: dict[str, Any] = dict(document.metadata)
        metadata.update(
            {
                "source_id": document.source_id,
                "source_kind": document.source_kind.value,
                "canonical_url": document.canonical_url,
                "title": document.title,
                "content_hash": document.content_hash,
                "related_source_ids": json.dumps(sorted(document.related_source_ids)),
                "provenance_schema": "autobrain.normalized-document.v1",
            }
        )
        if any(not Mem0Adapter._metadata_value(value) for value in metadata.values()):
            raise ValueError(f"unsupported Mem0 metadata for source {document.source_id}")
        return metadata

    @staticmethod
    def _metadata_value(value: Any) -> bool:
        return value is None or isinstance(value, str | int | float | bool)

    @staticmethod
    def _assert_candidate_safe(
        document: NormalizedDocument,
        heldout_source_ids: set[str],
        forbidden_markers: set[str],
    ) -> None:
        referenced_ids = {document.source_id, *document.related_source_ids}
        leaked_ids = referenced_ids & heldout_source_ids
        if leaked_ids:
            raise ValueError(f"held-out source cannot enter Mem0: {sorted(leaked_ids)}")
        serialized = json.dumps(document.model_dump(mode="json"), sort_keys=True)
        leaked_markers = sorted(
            marker for marker in forbidden_markers if marker and marker in serialized
        )
        if leaked_markers:
            raise ValueError(f"oracle marker cannot enter Mem0: {leaked_markers}")

    def _map_unattributed_memories(self, source_id: str) -> None:
        try:
            payload = self._bounded(
                "get_all", lambda: self._memory.get_all(filters=self._filters, top_k=10_000)
            )
        except Exception:
            return
        try:
            results = self._results(payload, operation="get_all")
        except Mem0AdapterError:
            return
        for item in results:
            memory_id = item.get("id")
            if isinstance(memory_id, str) and memory_id not in self._ingest_sources:
                self._ingest_sources[memory_id] = source_id

    def _result_source_id(self, result: Mapping[str, Any]) -> str | None:
        metadata = result.get("metadata")
        if isinstance(metadata, Mapping):
            typed_metadata = cast(Mapping[str, Any], metadata)
            if isinstance(typed_metadata.get("source_id"), str):
                return cast(str, typed_metadata["source_id"])
        if isinstance(result.get("source_id"), str):
            return cast(str, result["source_id"])
        memory_id = result.get("id")
        if isinstance(memory_id, str):
            return self._ingest_sources.get(memory_id)
        return None

    def _answer_evidence(self, result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "memory_id": str(result.get("id", "")),
            "memory": str(result.get("memory", "")),
            "score": result.get("score"),
            "source_id": self._result_source_id(result),
        }

    def _deterministic_result_key(self, result: Mapping[str, Any]) -> tuple[float, str, str]:
        score = result.get("score")
        numeric_score = float(score) if isinstance(score, int | float) else float("-inf")
        source_id = self._result_source_id(result) or ""
        return (-numeric_score, source_id, str(result.get("id", "")))
