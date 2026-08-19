from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from autobrain.candidates.mem0 import (
    Mem0Adapter,
    Mem0AdapterConfig,
    Mem0AdapterError,
    Mem0CleanupError,
    Mem0PersistenceError,
    StructuredAnswerError,
)

from .test_mem0 import FakeMemory, FakeOpenAI, FakeResponse, document

_CREATED_ADAPTERS: list[Mem0Adapter] = []


@pytest.fixture(autouse=True)
def cleanup_created_adapters() -> Iterator[None]:
    start = len(_CREATED_ADAPTERS)
    yield
    for memory in FakeMemory.instances:
        memory.cleanup_error = None
    for candidate in reversed(_CREATED_ADAPTERS[start:]):
        candidate.cleanup()
        assert not candidate.config.qdrant_path.exists()
        assert not candidate.config.history_db_path.exists()


def make_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    timeout_seconds: float = 0.05,
    heldout_source_ids: set[str] | None = None,
    base_url: str | None = None,
    memory_class: type[FakeMemory] = FakeMemory,
    openai_class: type[Any] = FakeOpenAI,
) -> Mem0Adapter:
    import autobrain.candidates.mem0 as module

    FakeMemory.instances.clear()
    FakeOpenAI.instances.clear()
    monkeypatch.setattr(module, "Memory", memory_class)
    monkeypatch.setattr(module, "OpenAI", openai_class)
    candidate = Mem0Adapter(
        Mem0AdapterConfig(
            run_id="repair-run",
            run_dir=tmp_path,
            timeout_seconds=timeout_seconds,
            heldout_source_ids=heldout_source_ids or set(),
            api_key="sk-test-provider-123456789",
            base_url=base_url,
        )
    )
    _CREATED_ADAPTERS.append(candidate)
    return candidate


def test_callback_and_proxy_same_request_emit_one_canonical_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = make_adapter(monkeypatch, tmp_path)
    response = FakeResponse(
        '{"answer":"Tuesday.","claim_ids":["claim-1"],"source_ids":["slack:launch"]}',
        usage=type("Usage", (), {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7})(),
        request_id="same-request",
    )
    callback = adapter.config.memory_config["llm"]["config"]["response_callback"]
    callback(None, response, {"autobrain_phase": "ingest"})
    adapter.record_proxy_event(
        {
            "request_id": "same-request",
            "model": "gpt-5-mini",
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        },
        phase="ingest",
    )
    callback(None, response, {"autobrain_phase": "ingest"})
    adapter.record_proxy_event(
        {
            "request_id": "other-request",
            "model": "gpt-5-mini",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
        phase="ingest",
    )
    assert [event.request_id for event in adapter.usage_events] == [
        "same-request",
        "other-request",
    ]
    assert adapter.usage_events[0].source == "mem0_callback+proxy"


def test_proxy_first_then_repeated_callback_keeps_one_proxy_canonical_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = make_adapter(monkeypatch, tmp_path)
    adapter.record_proxy_event(
        {
            "request_id": "matrix-request",
            "model": "proxy-model",
            "usage": {"prompt_tokens": 9, "completion_tokens": 8, "total_tokens": 17},
        },
        phase="search",
    )
    response = FakeResponse(
        "{}",
        usage=type("Usage", (), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})(),
        request_id="matrix-request",
    )
    callback = adapter.config.memory_config["llm"]["config"]["response_callback"]
    callback(None, response, {"autobrain_phase": "search"})
    callback(None, response, {"autobrain_phase": "search"})
    assert len(adapter.usage_events) == 1
    event = adapter.usage_events[0]
    assert event.source == "mem0_callback+proxy"
    assert event.model == "proxy-model"
    assert event.input_tokens == 9
    assert event.output_tokens == 8


def test_slow_mem0_add_is_bounded_and_typed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = threading.Event()

    class SlowAddMemory(FakeMemory):
        def add(self, messages: str, **kwargs: Any) -> dict[str, Any]:
            gate.wait()
            return {"results": []}

    adapter = make_adapter(monkeypatch, tmp_path, memory_class=SlowAddMemory)
    started = time.monotonic()
    try:
        with pytest.raises(Mem0AdapterError, match=r"timed out.*add"):
            adapter.ingest([document("slack:slow")])
        assert time.monotonic() - started < 0.5
    finally:
        adapter.cleanup()
    assert not adapter.config.qdrant_path.exists()
    assert not adapter.config.history_db_path.exists()


def test_slow_mem0_search_is_bounded_and_provider_clients_receive_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = threading.Event()

    class SlowSearchMemory(FakeMemory):
        def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
            gate.wait()
            return {"results": []}

    adapter = make_adapter(
        monkeypatch, tmp_path, timeout_seconds=0.03, memory_class=SlowSearchMemory
    )
    started = time.monotonic()
    try:
        with pytest.raises(Mem0AdapterError, match=r"timed out.*search"):
            adapter.search_native("slow")
        assert time.monotonic() - started < 0.5
        assert adapter.config.native_timeout_strategy
    finally:
        adapter.cleanup()
    assert not adapter.config.qdrant_path.exists()
    assert not adapter.config.history_db_path.exists()


def test_slow_answer_llm_is_bounded_and_typed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = threading.Event()

    class SlowCompletions:
        def create(self, **_kwargs: Any) -> FakeResponse:
            gate.wait()
            return FakeResponse("{}")

    class SlowOpenAI:
        def __init__(self, **_kwargs: Any) -> None:
            self.chat = type("Chat", (), {"completions": SlowCompletions()})()

    adapter = make_adapter(monkeypatch, tmp_path, timeout_seconds=0.03, openai_class=SlowOpenAI)
    started = time.monotonic()
    try:
        with pytest.raises(Mem0AdapterError, match=r"timed out.*answer"):
            adapter.answer(
                "question",
                [
                    {
                        "id": "m-1",
                        "memory": "Tuesday",
                        "metadata": {"source_id": "slack:launch"},
                    }
                ],
            )
        assert time.monotonic() - started < 0.5
    finally:
        adapter.cleanup()
    assert not adapter.config.qdrant_path.exists()
    assert not adapter.config.history_db_path.exists()


def test_answer_provider_error_is_redacted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FailingCompletions:
        def create(self, **_kwargs: Any) -> FakeResponse:
            raise RuntimeError("https://user:answer-secret@example.test/?api_key=answer-secret")

    class FailingOpenAI:
        def __init__(self, **_kwargs: Any) -> None:
            self.chat = type("Chat", (), {"completions": FailingCompletions()})()

    adapter = make_adapter(monkeypatch, tmp_path, openai_class=FailingOpenAI)
    try:
        with pytest.raises(StructuredAnswerError) as error:
            adapter.answer(
                "question",
                [
                    {
                        "id": "m-1",
                        "memory": "Tuesday",
                        "metadata": {"source_id": "slack:launch"},
                    }
                ],
            )
        assert "answer-secret" not in str(error.value)
        assert error.value.__cause__ is None
    finally:
        adapter.cleanup()


def test_direct_search_to_answer_rejects_configured_holdout_before_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = make_adapter(monkeypatch, tmp_path, heldout_source_ids={"slack:heldout"})
    try:
        native = adapter.search_native("question", top_k=2)
        before = len(FakeOpenAI.instances[-1].chat.completions.calls)
        with pytest.raises(ValueError, match="held-out"):
            adapter.answer("question", native["results"])
        assert len(FakeOpenAI.instances[-1].chat.completions.calls) == before
    finally:
        adapter.cleanup()


class FailingMemory(FakeMemory):
    def __init__(self) -> None:
        super().__init__()
        raise RuntimeError("https://user:upstream-secret@example.test/?api_key=upstream-secret")


def test_secret_bearing_urls_and_upstream_errors_are_redacted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import autobrain.candidates.mem0 as module

    monkeypatch.setattr(module, "Memory", FailingMemory)
    monkeypatch.setattr(module, "OpenAI", FakeOpenAI)
    config = Mem0AdapterConfig(
        run_id="secret-run",
        run_dir=tmp_path,
        api_key="adapter-secret",
        base_url="https://user:upstream-secret@example.test/v1?api_key=url-secret",
        heldout_source_ids=set(),
    )
    assert "upstream-secret" not in repr(config)
    assert "url-secret" not in repr(config)
    with pytest.raises(Mem0PersistenceError) as error:
        Mem0Adapter(config)
    assert "upstream-secret" not in str(error.value)
    assert "url-secret" not in str(error.value)
    assert error.value.__cause__ is None
    shutil.rmtree(config.qdrant_path)


def test_cleanup_closes_native_qdrant_and_sqlite_before_removing_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Resource:
        def __init__(self) -> None:
            self.closed = False

        def close(self, **_kwargs: Any) -> None:
            self.closed = True

    class Component:
        def __init__(self, client: Resource) -> None:
            self.client = client

    class Database:
        def __init__(self, owner: ClosableMemory) -> None:
            self.owner = owner

        def close(self) -> None:
            self.owner.db_closed = True

    class ClosableMemory(FakeMemory):
        def __init__(self) -> None:
            super().__init__()
            self.vector_store = Component(Resource())
            self.db = Database(self)
            self.llm = Component(Resource())
            self.embedding_model = Component(Resource())
            self.db_closed = False

    import autobrain.candidates.mem0 as module

    monkeypatch.setattr(module, "Memory", ClosableMemory)
    monkeypatch.setattr(module, "OpenAI", FakeOpenAI)
    adapter = Mem0Adapter(
        Mem0AdapterConfig(
            run_id="close-run",
            run_dir=tmp_path,
            heldout_source_ids=set(),
            api_key="sk-test-provider-123456789",
        )
    )
    receipt = adapter.cleanup()
    native = FakeMemory.instances[-1]
    assert isinstance(native, ClosableMemory)
    assert native.vector_store.client.closed
    assert native.llm.client.closed
    assert native.embedding_model.client.closed
    assert native.db_closed
    assert receipt.removed_paths
    assert not adapter.config.qdrant_path.exists()


def test_close_failure_keeps_typed_retained_state_and_cleanup_can_be_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingResource:
        def __init__(self) -> None:
            self.calls = 0

        def close(self, **_kwargs: Any) -> None:
            self.calls += 1
            if self.calls == 1:
                raise OSError("close-secret")

    class VectorStore:
        def __init__(self, resource: FailingResource) -> None:
            self.client = resource

    class ClosableMemory(FakeMemory):
        def __init__(self) -> None:
            super().__init__()
            self.resource = FailingResource()
            self.vector_store = VectorStore(self.resource)

    import autobrain.candidates.mem0 as module

    monkeypatch.setattr(module, "Memory", ClosableMemory)
    monkeypatch.setattr(module, "OpenAI", FakeOpenAI)
    adapter = Mem0Adapter(
        Mem0AdapterConfig(
            run_id="retry-run",
            run_dir=tmp_path,
            heldout_source_ids=set(),
            api_key="sk-test-provider-123456789",
        )
    )
    with pytest.raises(Mem0CleanupError) as error:
        adapter.cleanup()
    assert "close-secret" not in str(error.value)
    assert adapter.config.qdrant_path.exists()
    receipt = adapter.cleanup()
    assert receipt.removed_paths


def test_interrupted_cleanup_after_qdrant_close_retries_without_native_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Resource:
        def __init__(self) -> None:
            self.closed = False

        def close(self, **_kwargs: Any) -> None:
            self.closed = True

    class InterruptingDatabase:
        def __init__(self) -> None:
            self.calls = 0

        def close(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise InterruptedError("interrupt-secret")

    class VectorStore:
        def __init__(self) -> None:
            self.client = Resource()

    class InterruptingMemory(FakeMemory):
        def __init__(self) -> None:
            super().__init__()
            self.vector_store = VectorStore()
            self.db = InterruptingDatabase()

    import autobrain.candidates.mem0 as module

    monkeypatch.setattr(module, "Memory", InterruptingMemory)
    monkeypatch.setattr(module, "OpenAI", FakeOpenAI)
    adapter = Mem0Adapter(
        Mem0AdapterConfig(
            run_id="interrupt-run",
            run_dir=tmp_path,
            heldout_source_ids=set(),
            api_key="sk-test-provider-123456789",
        )
    )
    with pytest.raises(Mem0CleanupError) as error:
        adapter.cleanup()
    assert error.value.error_type == "InterruptedError"
    assert "interrupt-secret" not in str(error.value)
    receipt = adapter.cleanup()
    native = FakeMemory.instances[-1]
    assert isinstance(native, InterruptingMemory)
    assert native.deleted == [adapter.config.run_id]
    assert "qdrant" in receipt.closed_resources
    assert "sqlite" in receipt.closed_resources


@pytest.mark.parametrize(
    "payload",
    [
        '{"answer":"Tuesday.","claim_ids":[],"source_ids":["slack:launch"]}',
        '{"answer":"Tuesday.","claim_ids":["claim-1"],"source_ids":[]}',
        '{"answer":"Tuesday.","claim_ids":["  "],"source_ids":["slack:launch"]}',
    ],
)
def test_empty_or_blank_structured_evidence_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload: str
) -> None:
    FakeOpenAI.response = FakeResponse(payload)
    adapter = make_adapter(monkeypatch, tmp_path)
    with pytest.raises(StructuredAnswerError):
        adapter.answer(
            "question",
            [{"id": "m-1", "memory": "Tuesday", "metadata": {"source_id": "slack:launch"}}],
        )
