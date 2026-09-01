from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from autobrain.candidates.mem0 import (
    Mem0Adapter,
    Mem0AdapterConfig,
    Mem0CleanupError,
    Mem0PersistenceError,
    StructuredAnswerError,
)
from autobrain.models import NormalizedDocument, SourceKind


def document(source_id: str, text: str = "The launch is on Tuesday.") -> NormalizedDocument:
    return NormalizedDocument(
        source_id=source_id,
        source_kind=SourceKind.SLACK_MESSAGE,
        canonical_url="https://example.test/source",
        title="Launch note",
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        metadata={"channel_id": "C123", "page_id": "P123"},
    )


class FakeMemory:
    instances: ClassVar[list[FakeMemory]] = []

    def __init__(self) -> None:
        self.added: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.cleanup_error: OSError | None = None
        self.add_error: OSError | None = None
        self.get_all_calls = 0
        self.__class__.instances.append(self)

    def add(self, messages: str, **kwargs: Any) -> dict[str, Any]:
        if self.add_error is not None:
            raise self.add_error
        memory_id = f"m-{len(self.added) + 1}"
        self.added.append({"id": memory_id, "memory": messages, **kwargs})
        return {"results": [{"id": memory_id, "memory": messages, "event": "ADD"}]}

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "results": [
                {
                    "id": "m-1",
                    "memory": "The launch is on Tuesday.",
                    "score": 0.9,
                    "metadata": {"source_id": "slack:launch"},
                },
                {
                    "id": "m-2",
                    "memory": "Ignore this held-out answer.",
                    "score": 0.1,
                    "metadata": {"source_id": "slack:heldout"},
                },
            ]
        }

    def get(self, memory_id: str) -> dict[str, Any] | None:
        return next((item for item in self.added if item["id"] == memory_id), None)

    def get_all(self, **kwargs: Any) -> dict[str, Any]:
        self.get_all_calls += 1
        return {"results": self.added}

    def delete_all(self, **kwargs: Any) -> dict[str, str]:
        if self.cleanup_error is not None:
            raise self.cleanup_error
        self.deleted.append(kwargs["run_id"])
        self.added.clear()
        return {"message": "Memories deleted successfully!"}


class FakeResponse:
    def __init__(self, content: str, usage: Any = None, *, request_id: str = "req-answer") -> None:
        self.choices = [
            type("Choice", (), {"message": type("Message", (), {"content": content})()})()
        ]
        self.usage = usage
        self.id = request_id
        self.model = "gpt-5-mini"


class FakeCompletions:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return self.response


class FakeChat:
    def __init__(self, response: FakeResponse) -> None:
        self.completions = FakeCompletions(response)


class FakeOpenAI:
    instances: ClassVar[list[FakeOpenAI]] = []
    response = FakeResponse(
        '{"answer":"Tuesday.","claim_ids":["claim-1"],"source_ids":["slack:launch"]}',
        usage=type(
            "Usage", (), {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
        )(),
    )

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.chat = FakeChat(self.response)
        self.__class__.instances.append(self)


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Mem0Adapter]:
    import autobrain.candidates.mem0 as module

    FakeMemory.instances.clear()
    FakeOpenAI.instances.clear()
    monkeypatch.setattr(module, "Memory", FakeMemory)
    monkeypatch.setattr(module, "OpenAI", FakeOpenAI)
    candidate = Mem0Adapter(
        Mem0AdapterConfig(
            run_id="run-test",
            run_dir=tmp_path,
            heldout_source_ids=set(),
            api_key="sk-test-provider-123456789",
        )
    )
    try:
        yield candidate
    finally:
        FakeMemory.instances[0].cleanup_error = None
        candidate.cleanup()


def test_oss_import_and_local_store_config(adapter: Mem0Adapter, tmp_path: Path) -> None:
    assert adapter.config.memory_config["vector_store"]["config"]["path"] == str(
        tmp_path / "qdrant"
    )
    assert adapter.config.memory_config["history_db_path"] == str(tmp_path / "history.db")
    assert adapter.config.memory_config["llm"]["config"]["model"] == "gpt-5-mini"
    assert adapter.config.memory_config["embedder"]["config"]["model"] == "text-embedding-3-small"
    assert callable(adapter.config.memory_config["llm"]["config"]["response_callback"])
    assert adapter.config.answer_client_kwargs["timeout"] == 60.0
    assert adapter.config.user_id.startswith("autobrain-run-test-")


def test_ingest_uses_whole_document_provenance_and_native_add(adapter: Mem0Adapter) -> None:
    result = adapter.ingest([document("slack:launch")])
    added = FakeMemory.instances[0].added[0]
    assert added["memory"] == "The launch is on Tuesday."
    assert added["infer"] is True
    assert added["metadata"]["source_id"] == "slack:launch"
    assert result.native_results[0]["id"] == "m-1"


def test_ingest_does_not_rescan_store_when_add_returns_memory_ids(adapter: Mem0Adapter) -> None:
    adapter.ingest([document("slack:launch"), document("slack:other")])

    assert FakeMemory.instances[0].get_all_calls == 0


def test_ingest_keeps_attribution_scan_fallback_without_memory_ids(
    adapter: Mem0Adapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_add = FakeMemory.add

    def add_without_id(self: FakeMemory, messages: str, **kwargs: Any) -> dict[str, Any]:
        original_add(self, messages, **kwargs)
        return {"results": [{"memory": messages, "event": "ADD"}]}

    monkeypatch.setattr(FakeMemory, "add", add_without_id)
    adapter.ingest([document("slack:launch")])

    assert FakeMemory.instances[0].get_all_calls == 1


def test_persistence_failure_names_the_source_and_retains_store(adapter: Mem0Adapter) -> None:
    FakeMemory.instances[0].add_error = OSError(
        "https://user:persistence-secret@example.test/?api_key=persistence-secret"
    )
    with pytest.raises(Mem0PersistenceError, match=r"slack:launch.*OSError") as error:
        adapter.ingest([document("slack:launch")])
    assert "persistence-secret" not in str(error.value)
    assert error.value.__cause__ is None
    assert adapter.config.qdrant_path.exists()


def test_native_retrieval_is_separate_and_deterministic(adapter: Mem0Adapter) -> None:
    adapter.ingest([document("slack:launch")])
    result = adapter.search_native("when is launch", top_k=1)
    assert result["results"][0]["id"] == "m-1"
    assert result["results"][0]["metadata"]["source_id"] == "slack:launch"
    assert result["results"][0]["score"] == 0.9
    assert FakeMemory.instances[0].added[0]["run_id"] == adapter.config.run_id


def test_answer_wrapper_parses_structured_output_and_captures_usage(adapter: Mem0Adapter) -> None:
    native = [
        {
            "id": "m-1",
            "memory": "The launch is on Tuesday.",
            "score": 0.9,
            "metadata": {"source_id": "slack:launch"},
        }
    ]
    answer = adapter.answer("When is launch?", native)
    assert answer.answer == "Tuesday."
    assert answer.claim_ids == ["claim-1"]
    assert answer.source_ids == ["slack:launch"]
    assert answer.usage is not None
    assert answer.usage.input_tokens == 12
    assert answer.usage.output_tokens == 7


def test_answer_wrapper_normalizes_native_memory_id_citations(
    adapter: Mem0Adapter,
) -> None:
    FakeOpenAI.instances[-1].chat.completions.response = FakeResponse(
        '```json\n{"answer":"Tuesday.","claim_ids":["claim-1"],'
        '"source_ids":["memory-result-1"]}\n```'
    )
    native = [
        {
            "id": "memory-result-1",
            "memory": "The launch is on Tuesday.",
            "score": 0.9,
            "metadata": {"source_id": "slack:launch"},
        }
    ]

    answer = adapter.answer("When is launch?", native)

    assert answer.source_ids == ["slack:launch"]


def test_answer_wrapper_accepts_a_complete_markdown_json_fence(
    adapter: Mem0Adapter,
) -> None:
    FakeOpenAI.instances[-1].chat.completions.response = FakeResponse(
        '```json\n{"answer":"Tuesday.","claim_ids":["claim-1"],"source_ids":["slack:launch"]}\n```'
    )
    native = [
        {
            "id": "m-1",
            "memory": "The launch is on Tuesday.",
            "score": 0.9,
            "metadata": {"source_id": "slack:launch"},
        }
    ]

    answer = adapter.answer("When is launch?", native)

    assert answer.answer == "Tuesday."
    assert answer.source_ids == ["slack:launch"]


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "{}",
        '{"answer": 3}',
        'Here is the answer: {"answer":"Tuesday.","claim_ids":["claim-1"],'
        '"source_ids":["slack:launch"]}',
        '{"answer":"Tuesday.","claim_ids":["claim-1"],'
        '"source_ids":["slack:launch"]} Hope that helps.',
        '```json\n{"answer":"Tuesday.","claim_ids":["claim-1"],'
        '"source_ids":["slack:launch"]}\n```\nHope that helps.',
        '```json\n{"answer":"Tuesday.","claim_ids":["claim-1"],"source_ids":["slack:launch"]\n```',
        '[{"answer":"Tuesday.","claim_ids":["claim-1"],"source_ids":["slack:launch"]}]',
    ],
)
def test_malformed_structured_output_is_not_empty_success(
    adapter: Mem0Adapter, payload: str
) -> None:
    FakeOpenAI.instances[-1].chat.completions.response = FakeResponse(payload)
    native = [
        {
            "id": "m-1",
            "memory": "The launch is on Tuesday.",
            "metadata": {"source_id": "slack:launch"},
        }
    ]

    with pytest.raises(StructuredAnswerError):
        adapter.answer("question", native)


@pytest.mark.parametrize(
    "payload,native",
    [
        (
            '{"answer":"Tuesday.","claim_ids":["claim-1"],"source_ids":["unknown-memory"]}',
            [
                {
                    "id": "memory-result-1",
                    "memory": "Tuesday",
                    "metadata": {"source_id": "slack:launch"},
                }
            ],
        ),
        (
            '{"answer":"Tuesday.","claim_ids":["claim-1"],"source_ids":["shared-memory"]}',
            [
                {
                    "id": "shared-memory",
                    "memory": "Tuesday",
                    "metadata": {"source_id": "slack:launch"},
                },
                {
                    "id": "shared-memory",
                    "memory": "Wednesday",
                    "metadata": {"source_id": "slack:other"},
                },
            ],
        ),
        (
            '{"answer":"Tuesday.","claim_ids":["claim-1"],"source_ids":[{"id":"memory-result-1"}]}',
            [
                {
                    "id": "memory-result-1",
                    "memory": "Tuesday",
                    "metadata": {"source_id": "slack:launch"},
                }
            ],
        ),
        (
            '{"answer":"Tuesday.","claim_ids":["claim-1"],'
            '"source_ids":["memory-result-1"],"citations":["memory-result-1"]}',
            [
                {
                    "id": "memory-result-1",
                    "memory": "Tuesday",
                    "metadata": {"source_id": "slack:launch"},
                }
            ],
        ),
    ],
)
def test_answer_wrapper_rejects_unsupported_or_ambiguous_citations(
    adapter: Mem0Adapter, payload: str, native: list[dict[str, Any]]
) -> None:
    FakeOpenAI.instances[-1].chat.completions.response = FakeResponse(payload)

    with pytest.raises(StructuredAnswerError):
        adapter.answer("question", native)


def test_heldout_documents_are_rejected_before_native_add(adapter: Mem0Adapter) -> None:
    with pytest.raises(ValueError, match="held-out"):
        adapter.ingest([document("slack:heldout")], heldout_source_ids={"slack:heldout"})
    assert FakeMemory.instances[0].added == []


def test_heldout_results_are_rejected(adapter: Mem0Adapter) -> None:
    with pytest.raises(ValueError, match="held-out"):
        adapter.filter_native_results(
            [{"id": "m", "metadata": {"source_id": "slack:heldout"}}],
            heldout_source_ids={"slack:heldout"},
        )


def test_proxy_usage_is_deduplicated_and_missing_usage_is_visible(adapter: Mem0Adapter) -> None:
    adapter.record_proxy_event({"request_id": "req-1", "model": "gpt-5-mini"}, phase="ingest")
    adapter.record_proxy_event({"request_id": "req-1", "model": "gpt-5-mini"}, phase="ingest")
    assert len(adapter.usage_events) == 1
    assert adapter.usage_events[0].input_tokens is None
    assert adapter.usage_events[0].output_tokens is None


def test_native_inspection_and_cleanup_are_scoped(adapter: Mem0Adapter) -> None:
    adapter.ingest([document("slack:launch")])
    memory = adapter.get("m-1")
    assert memory is not None
    assert memory["id"] == "m-1"
    assert adapter.get_all()["results"]
    adapter.cleanup()
    assert FakeMemory.instances[0].deleted == [adapter.config.run_id]


def test_cleanup_failure_is_visible(adapter: Mem0Adapter) -> None:
    FakeMemory.instances[0].cleanup_error = OSError("qdrant locked")
    with pytest.raises(Mem0CleanupError, match="OSError") as error:
        adapter.cleanup()
    assert "qdrant locked" not in str(error.value)
    assert adapter.config.qdrant_path.exists()


def test_dirty_store_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import autobrain.candidates.mem0 as module

    dirty = tmp_path / "qdrant"
    dirty.mkdir()
    (dirty / "segment").write_text("retained", encoding="utf-8")
    monkeypatch.setattr(module, "Memory", FakeMemory)
    monkeypatch.setattr(module, "OpenAI", FakeOpenAI)
    try:
        with pytest.raises(Mem0PersistenceError, match="dirty Mem0 vector store"):
            Mem0Adapter(
                Mem0AdapterConfig(
                    run_id="dirty",
                    run_dir=tmp_path,
                    heldout_source_ids=set(),
                    api_key="sk-test-provider-123456789",
                )
            )
    finally:
        shutil.rmtree(dirty)


def test_proxy_base_url_is_forwarded_without_secret_in_config_repr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autobrain.candidates.mem0 as module

    monkeypatch.setattr(module, "Memory", FakeMemory)
    monkeypatch.setattr(module, "OpenAI", FakeOpenAI)
    config = Mem0AdapterConfig(
        run_id="run-url",
        run_dir=tmp_path,
        heldout_source_ids=set(),
        api_key="secret",
        base_url="http://127.0.0.1:9999/v1",
    )
    adapter = Mem0Adapter(config)
    try:
        assert adapter.config.answer_client_kwargs["base_url"] == "http://127.0.0.1:9999/v1"
        assert "secret" not in repr(adapter.config)
    finally:
        adapter.cleanup()


def test_native_methods_are_called_with_scoped_ids(adapter: Mem0Adapter) -> None:
    adapter.ingest([document("slack:launch")])
    adapter.search_native("q")
    adapter.get_all()
    calls = FakeMemory.instances[0]
    assert calls.added[0]["user_id"] == adapter.config.user_id


def test_answer_prompt_contains_only_native_context_and_question(adapter: Mem0Adapter) -> None:
    adapter.answer(
        "When?", [{"id": "m-1", "memory": "Tuesday", "metadata": {"source_id": "slack:launch"}}]
    )
    completions = FakeOpenAI.instances[-1].chat.completions
    assert isinstance(completions, FakeCompletions)
    call = completions.calls[-1]
    serialized = json.dumps(call)
    assert "When?" in serialized and "Tuesday" in serialized
    assert "heldout" not in serialized.lower()
