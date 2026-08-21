from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from openai import OpenAI as RealOpenAI

from autobrain.candidates.mem0 import (
    Mem0Adapter,
    Mem0OperationTimeout,
    Mem0UsageUpdate,
    StructuredAnswerError,
)
from autobrain.metering import LoopbackMeteringProxy, PriceQuote, PriceSheet

from .test_mem0 import FakeMemory, FakeOpenAI, FakeResponse
from .test_mem0_repair import make_adapter

_NATIVE = [
    {
        "id": "memory-result-1",
        "memory": "The launch is on Tuesday.",
        "score": 0.9,
        "metadata": {"source_id": "slack:launch"},
    }
]
_VALID = '{"answer":"Tuesday.","claim_ids":["claim-1"],"source_ids":["slack:launch"]}'


class SequenceCompletions:
    def __init__(self, responses: list[FakeResponse | BaseException]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class SequenceOpenAI:
    instances: ClassVar[list[SequenceOpenAI]] = []
    responses: ClassVar[list[FakeResponse | BaseException]] = []

    def __init__(self, **_kwargs: Any) -> None:
        self.chat = type("Chat", (), {"completions": SequenceCompletions(list(self.responses))})()
        self.__class__.instances.append(self)


def sequence_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    responses: list[FakeResponse | BaseException],
    *,
    usage_sink: Any = None,
    timeout_seconds: float = 0.05,
) -> Mem0Adapter:
    import autobrain.candidates.mem0 as module

    FakeMemory.instances.clear()
    FakeOpenAI.instances.clear()
    SequenceOpenAI.instances.clear()
    SequenceOpenAI.responses = responses
    monkeypatch.setattr(module, "Memory", FakeMemory)
    monkeypatch.setattr(module, "OpenAI", SequenceOpenAI)
    adapter = Mem0Adapter(
        module.Mem0AdapterConfig(
            run_id="answer-retry",
            run_dir=tmp_path,
            heldout_source_ids=set(),
            api_key="sk-test-provider-123456789",
            timeout_seconds=timeout_seconds,
        ),
        usage_sink=usage_sink,
    )
    return adapter


def completions() -> SequenceCompletions:
    chat = cast(Any, SequenceOpenAI.instances[-1].chat)
    return cast(SequenceCompletions, chat.completions)


@pytest.mark.parametrize(
    "first",
    [
        '{"answer":"Tuesday.","claim_ids":[],"source_ids":["slack:launch"]}',
        '{"answer":"Tuesday.","claim_ids":["claim-1"],"source_ids":[]}',
        '{"answer":"Tuesday.","claim_ids":["  "],"source_ids":["slack:launch"]}',
        '{"answer":"Tuesday.","claim_ids":["claim-1"],"source_ids":["  "]}',
    ],
)
def test_retries_once_for_only_nonblank_answer_with_blank_or_empty_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, first: str
) -> None:
    adapter = sequence_adapter(
        monkeypatch,
        tmp_path,
        [FakeResponse(first, request_id="req-first"), FakeResponse(_VALID, request_id="req-retry")],
    )
    try:
        answer = adapter.answer("When is launch?", _NATIVE)
        calls = completions().calls
        assert answer.source_ids == ["slack:launch"]
        assert len(calls) == 2
        assert calls[0]["model"] == calls[1]["model"] == "gpt-5-mini"
        assert calls[0]["temperature"] == calls[1]["temperature"] == 0
        assert calls[0]["response_format"] == calls[1]["response_format"] == {"type": "json_object"}
        retry_prompt = str(calls[1]["messages"])
        assert "evidence-backed claim_ids and source_ids" in retry_prompt
        assert "same supplied native Mem0 evidence" in retry_prompt
        assert "honestly state that you cannot answer" in retry_prompt
        assert "Do not infer citations" in retry_prompt
    finally:
        adapter.cleanup()


def test_retry_failure_is_structured_and_never_gets_a_third_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = '{"answer":"Tuesday.","claim_ids":[],"source_ids":["slack:launch"]}'
    adapter = sequence_adapter(
        monkeypatch,
        tmp_path,
        [
            FakeResponse(first, request_id="req-first"),
            FakeResponse(first, request_id="req-retry"),
            FakeResponse(_VALID, request_id="must-not-run"),
        ],
    )
    try:
        with pytest.raises(StructuredAnswerError):
            adapter.answer("When is launch?", _NATIVE)
        assert len(completions().calls) == 2
    finally:
        adapter.cleanup()


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        '```json\n{"answer":"Tuesday.","claim_ids":[],"source_ids":[]}\n``` trailing',
        '[{"answer":"Tuesday.","claim_ids":[],"source_ids":[]}]',
        '{"answer":"   ","claim_ids":[],"source_ids":[]}',
        '{"answer":"Tuesday.","claim_ids":[],"source_ids":[],"citations":[]}',
        '{"answer":"Tuesday.","claim_ids":null,"source_ids":[]}',
        '{"answer":"Tuesday.","claim_ids":[],"source_ids":[3]}',
    ],
)
def test_does_not_retry_other_structured_validation_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload: str
) -> None:
    adapter = sequence_adapter(
        monkeypatch,
        tmp_path,
        [FakeResponse(payload, request_id="req-first"), FakeResponse(_VALID)],
    )
    try:
        with pytest.raises(StructuredAnswerError):
            adapter.answer("When is launch?", _NATIVE)
        assert len(completions().calls) == 1
    finally:
        adapter.cleanup()


@pytest.mark.parametrize(
    "payload,native",
    [
        (
            '{"answer":"Tuesday.","claim_ids":[],"source_ids":["unknown-memory"]}',
            _NATIVE,
        ),
        (
            '{"answer":"Tuesday.","claim_ids":[],"source_ids":["shared-memory"]}',
            [
                {"id": "shared-memory", "memory": "Tuesday", "source_id": "slack:launch"},
                {"id": "shared-memory", "memory": "Wednesday", "source_id": "slack:other"},
            ],
        ),
        (
            '{"answer":"Tuesday.","claim_ids":[],"source_ids":["slack:launch","slack:launch"]}',
            _NATIVE,
        ),
    ],
)
def test_does_not_retry_unknown_ambiguous_or_duplicate_citations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: str,
    native: list[dict[str, Any]],
) -> None:
    adapter = sequence_adapter(
        monkeypatch,
        tmp_path,
        [FakeResponse(payload, request_id="req-first"), FakeResponse(_VALID)],
    )
    try:
        with pytest.raises(StructuredAnswerError):
            adapter.answer("When is launch?", native)
        assert len(completions().calls) == 1
    finally:
        adapter.cleanup()


def test_provider_error_does_not_retry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter = sequence_adapter(
        monkeypatch,
        tmp_path,
        [RuntimeError("provider failed"), FakeResponse(_VALID)],
    )
    try:
        with pytest.raises(StructuredAnswerError, match="provider failed \\(RuntimeError\\)"):
            adapter.answer("When is launch?", _NATIVE)
        assert len(completions().calls) == 1
    finally:
        adapter.cleanup()


def test_retry_uses_distinct_bounded_operation_and_existing_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = threading.Event()
    first = '{"answer":"Tuesday.","claim_ids":[],"source_ids":["slack:launch"]}'

    class RetryTimeoutCompletions(SequenceCompletions):
        def create(self, **kwargs: Any) -> FakeResponse:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return FakeResponse(first, request_id="req-first")
            gate.wait()
            return FakeResponse(_VALID, request_id="req-retry")

    class RetryTimeoutOpenAI:
        instances: ClassVar[list[RetryTimeoutOpenAI]] = []

        def __init__(self, **_kwargs: Any) -> None:
            self.chat = type("Chat", (), {"completions": RetryTimeoutCompletions([])})()
            self.__class__.instances.append(self)

    adapter = make_adapter(
        monkeypatch,
        tmp_path,
        timeout_seconds=0.03,
        openai_class=RetryTimeoutOpenAI,
    )
    try:
        with pytest.raises(Mem0OperationTimeout) as error:
            adapter.answer("When is launch?", _NATIVE)
        assert error.value.operation == "answer-retry"
    finally:
        adapter.cleanup()


def test_retry_calls_both_affect_the_metered_budget_with_distinct_request_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import autobrain.candidates.mem0 as module

    responses = iter(
        [
            {
                "id": "metered-first",
                "model": "gpt-5-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                '{"answer":"Tuesday.","claim_ids":[],"source_ids":["slack:launch"]}'
                            ),
                        },
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
            {
                "id": "metered-retry",
                "model": "gpt-5-mini",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": _VALID},
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
        ]
    )
    prices = PriceSheet(
        version="test",
        effective_date="2026-08-21",
        models={
            "openai:gpt-5-mini": PriceQuote(
                input_usd_per_million=1.0,
                output_usd_per_million=1.0,
            )
        },
    )
    FakeMemory.instances.clear()
    monkeypatch.setattr(module, "Memory", FakeMemory)
    monkeypatch.setattr(module, "OpenAI", RealOpenAI)
    with LoopbackMeteringProxy(
        lambda _payload: next(responses),
        budget_usd=1.0,
        prices=prices,
        default_candidate="mem0",
    ) as proxy:
        adapter = Mem0Adapter(
            module.Mem0AdapterConfig(
                run_id="metered-answer-retry",
                run_dir=tmp_path,
                heldout_source_ids=set(),
                api_key="loopback-key",
                base_url=proxy.base_url,
            )
        )
        try:
            answer = adapter.answer("When is launch?", _NATIVE)
            assert answer.source_ids == ["slack:launch"]
            assert [event.request_id for event in proxy.events] == [
                "metered-first",
                "metered-retry",
            ]
            assert abs(proxy.spent_usd - 0.000012) < 1e-12
        finally:
            adapter.cleanup()


def test_both_calls_emit_distinct_usage_events_and_sink_updates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    usage = type("Usage", (), {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6})()
    updates: list[Mem0UsageUpdate] = []
    first = '{"answer":"Tuesday.","claim_ids":[],"source_ids":["slack:launch"]}'
    adapter = sequence_adapter(
        monkeypatch,
        tmp_path,
        [
            FakeResponse(first, usage=usage, request_id="req-first"),
            FakeResponse(_VALID, usage=usage, request_id="req-retry"),
        ],
        usage_sink=updates.append,
    )
    try:
        answer = adapter.answer("When is launch?", _NATIVE)
        assert answer.usage is not None
        assert [event.request_id for event in adapter.usage_events] == [
            "req-first",
            "req-retry",
        ]
        assert [event.phase for event in adapter.usage_events] == ["answer", "answer-retry"]
        assert [update.request_id for update in updates] == ["req-first", "req-retry"]
    finally:
        adapter.cleanup()
