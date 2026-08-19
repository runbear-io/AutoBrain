from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from autobrain.candidates.mem0 import (
    Mem0Adapter,
    Mem0AdapterConfig,
    Mem0MissingProviderError,
    Mem0SecretBoundaryError,
    Mem0UsageUpdate,
)
from autobrain.models import NormalizedDocument, SourceKind, Status

from .test_mem0 import FakeMemory, FakeOpenAI, FakeResponse

_PROVIDER_KEY = "sk-test-provider-123456789"


def secret_document(secret: str, *, field: str) -> NormalizedDocument:
    title = "Safe title"
    canonical_url = "https://example.test/safe"
    text = "Safe text"
    metadata = {"channel": "safe"}
    if field == "metadata":
        metadata = {"authorization": f"Bearer {secret}"}
    elif field == "canonical_url":
        canonical_url = f"https://example.test/{secret}"
    elif field == "title":
        title = secret
    else:
        text = secret
    return NormalizedDocument(
        source_id="slack:secret",
        source_kind=SourceKind.SLACK_MESSAGE,
        canonical_url=canonical_url,
        title=title,
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        metadata=metadata,
    )


def adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    usage_sink: Any = None,
    known_secrets: set[str] | None = None,
    memory_class: type[FakeMemory] = FakeMemory,
) -> Mem0Adapter:
    import autobrain.candidates.mem0 as module

    FakeMemory.instances.clear()
    FakeOpenAI.instances.clear()
    monkeypatch.setattr(module, "Memory", memory_class)
    monkeypatch.setattr(module, "OpenAI", FakeOpenAI)
    return Mem0Adapter(
        Mem0AdapterConfig(
            run_id="proxy-repair",
            run_dir=tmp_path,
            heldout_source_ids=set(),
            api_key=_PROVIDER_KEY,
            known_secrets=known_secrets or set(),
        ),
        usage_sink=usage_sink,
    )


@pytest.mark.parametrize("field", ["text", "title", "canonical_url", "metadata"])
def test_known_secret_corpus_is_rejected_before_memory_add(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    candidate = adapter(monkeypatch, tmp_path, known_secrets={"runtime-company-secret"})
    try:
        with pytest.raises(Mem0SecretBoundaryError):
            candidate.ingest([secret_document("runtime-company-secret", field=field)])
        assert FakeMemory.instances[0].added == []
    finally:
        candidate.cleanup()
    assert not (tmp_path / "qdrant").exists()
    assert not (tmp_path / "history.db").exists()


@pytest.mark.parametrize(
    "secret",
    [
        "sk-openai-secret-123456",
        "xoxb-slack-secret-123456",
        "Bearer bearer-secret-123456",
    ],
)
def test_credential_patterns_are_rejected_before_memory_add(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, secret: str
) -> None:
    candidate = adapter(monkeypatch, tmp_path)
    try:
        with pytest.raises(Mem0SecretBoundaryError):
            candidate.ingest([secret_document(secret, field="text")])
        assert FakeMemory.instances[0].added == []
    finally:
        candidate.cleanup()


def test_runtime_environment_slack_secret_is_rejected_before_add(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AUTOBRAIN_SLACK_CLIENT_SECRET", "runtime-slack-value")
    candidate = adapter(monkeypatch, tmp_path)
    try:
        with pytest.raises(Mem0SecretBoundaryError):
            candidate.ingest([secret_document("runtime-slack-value", field="text")])
        assert FakeMemory.instances[0].added == []
    finally:
        candidate.cleanup()


def test_secret_native_search_result_is_not_emitted_or_sent_to_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class SecretSearchMemory(FakeMemory):
        def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
            return {
                "results": [
                    {
                        "id": "secret-memory",
                        "memory": "native-result-secret",
                        "metadata": {"source_id": "slack:launch"},
                        "score": 1.0,
                    }
                ]
            }

    candidate = adapter(
        monkeypatch,
        tmp_path,
        known_secrets={"native-result-secret"},
        memory_class=SecretSearchMemory,
    )
    try:
        with pytest.raises(Mem0SecretBoundaryError):
            candidate.search_native("question")
        assert FakeOpenAI.instances[-1].chat.completions.calls == []
    finally:
        candidate.cleanup()


def test_secret_native_add_artifact_is_not_returned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class SecretAddMemory(FakeMemory):
        def add(self, messages: str, **kwargs: Any) -> dict[str, Any]:
            self.added.append({"memory": messages, **kwargs})
            return {"results": [{"id": "m-secret", "memory": "artifact-secret"}]}

    candidate = adapter(
        monkeypatch,
        tmp_path,
        known_secrets={"artifact-secret"},
        memory_class=SecretAddMemory,
    )
    try:
        with pytest.raises(Mem0SecretBoundaryError):
            candidate.ingest([secret_document("safe", field="text")])
    finally:
        candidate.cleanup()


def test_secret_question_and_direct_native_evidence_fail_before_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate = adapter(monkeypatch, tmp_path, known_secrets={"prompt-secret"})
    evidence = [
        {
            "id": "m-1",
            "memory": "Safe evidence",
            "metadata": {"source_id": "slack:launch"},
        }
    ]
    try:
        with pytest.raises(Mem0SecretBoundaryError):
            candidate.answer("prompt-secret", evidence)
        with pytest.raises(Mem0SecretBoundaryError):
            candidate.answer(
                "safe question",
                [
                    {
                        "id": "m-2",
                        "memory": "prompt-secret",
                        "metadata": {"source_id": "slack:launch"},
                    }
                ],
            )
        assert FakeOpenAI.instances[-1].chat.completions.calls == []
    finally:
        candidate.cleanup()


def test_missing_provider_is_canonical_before_store_or_client_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import autobrain.candidates.mem0 as module

    class ForbiddenMemory:
        def __init__(self) -> None:
            raise AssertionError("Memory must not be constructed")

    class ForbiddenOpenAI:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError("OpenAI must not be constructed")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(module, "Memory", ForbiddenMemory)
    monkeypatch.setattr(module, "OpenAI", ForbiddenOpenAI)
    config = Mem0AdapterConfig(
        run_id="missing-provider",
        run_dir=tmp_path,
        heldout_source_ids=set(),
        known_secrets=set(),
    )
    with pytest.raises(Mem0MissingProviderError) as error:
        Mem0Adapter(config)
    assert error.value.status is Status.MISSING_PROVIDER
    assert not (tmp_path / "qdrant").exists()
    assert not (tmp_path / "history.db").exists()


def test_usage_sink_receives_upserted_canonical_event_in_both_arrival_orders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    updates: list[Mem0UsageUpdate] = []
    candidate = adapter(monkeypatch, tmp_path, usage_sink=updates.append)
    callback = candidate.config.memory_config["llm"]["config"]["response_callback"]
    callback_response = FakeResponse(
        "{}",
        usage=type("Usage", (), {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})(),
        request_id="callback-first",
    )
    callback(None, callback_response, {"autobrain_phase": "ingest"})
    candidate.record_proxy_event(
        {
            "request_id": "callback-first",
            "model": "gpt-5-mini",
            "usage": {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13},
        },
        phase="ingest",
    )
    candidate.record_proxy_event(
        {
            "request_id": "proxy-first",
            "model": "gpt-5-mini",
            "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
        },
        phase="search",
    )
    proxy_first_response = FakeResponse(
        "{}",
        usage=type("Usage", (), {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})(),
        request_id="proxy-first",
    )
    callback(None, proxy_first_response, {"autobrain_phase": "search"})
    try:
        latest = {update.request_id: update for update in updates}
        assert set(latest) == {"callback-first", "proxy-first"}
        assert all(update.action == "UPSERT" for update in latest.values())
        assert latest["callback-first"].revision == 2
        assert latest["callback-first"].event.source == "mem0_callback+proxy"
        assert latest["callback-first"].event.input_tokens == 8
        assert latest["proxy-first"].revision == 2
        assert latest["proxy-first"].event.source == "mem0_callback+proxy"
        assert latest["proxy-first"].event.input_tokens == 7
    finally:
        candidate.cleanup()
