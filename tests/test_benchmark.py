from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from autobrain.benchmark import (
    BenchmarkBuildConfig,
    BenchmarkStatus,
    GenerationResponse,
    OpenAICompatibleBenchmarkProvider,
    OracleResponse,
    SlackQuestionThread,
    build_benchmark,
)
from autobrain.models import NormalizedDocument, SourceKind


def _document(
    source_id: str,
    text: str,
    *,
    kind: SourceKind = SourceKind.NOTION_PAGE,
) -> NormalizedDocument:
    return NormalizedDocument(
        source_id=source_id,
        source_kind=kind,
        canonical_url=f"https://example.test/{source_id}",
        title=source_id,
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        metadata={"topic": source_id.split(":")[-1]},
    )


def _thread(number: int, *, topic: str = "ops") -> SlackQuestionThread:
    root_id = f"slack:message:C{number}:root"
    reply_id = f"slack:message:C{number}:reply"
    return SlackQuestionThread(
        root=_document(
            root_id,
            f"How do we handle {topic} incident {number}?",
            kind=SourceKind.SLACK_MESSAGE,
        ),
        replies=(
            _document(
                reply_id,
                (
                    f"Use the documented {topic} runbook and notify the on-call "
                    f"within {number + 5} minutes."
                ),
                kind=SourceKind.SLACK_MESSAGE,
            ),
        ),
        root_is_bot=False,
        reply_is_bot=(False,),
        channel=f"channel-{number % 4}",
        topic=topic,
    )


class DeterministicProvider:
    def __init__(self, generation_responses: list[GenerationResponse] | None = None) -> None:
        self.generation_responses = list(generation_responses or [])
        self.oracle_calls = 0
        self.generation_calls = 0
        self.prompts: list[str] = []

    def extract_oracle(
        self,
        *,
        thread: SlackQuestionThread,
        model: str,
        temperature: int,
        seed: int,
        timeout_seconds: float,
    ) -> OracleResponse:
        assert model == "gpt-5-mini"
        assert temperature == 0
        assert timeout_seconds > 0
        self.oracle_calls += 1
        human_replies = [
            reply.text
            for reply, is_bot in zip(thread.replies, thread.reply_is_bot, strict=True)
            if not is_bot
        ]
        return OracleResponse(
            expected_claims=human_replies,
            forbidden_contradictions=[f"The {thread.topic} guidance does not exist."],
            weak_human_confidence=0.8,
            input_tokens=10 + seed % 3,
            output_tokens=5,
            cost_usd=0,
        )

    def generate(
        self,
        *,
        prompt: str,
        model: str,
        temperature: int,
        seed: int,
        timeout_seconds: float,
    ) -> GenerationResponse:
        assert model == "gpt-5-mini"
        assert temperature == 0
        assert timeout_seconds > 0
        assert seed >= 0
        self.generation_calls += 1
        self.prompts.append(prompt)
        for index, response in enumerate(self.generation_responses):
            if response.source_ids[0] in prompt:
                selected = self.generation_responses.pop(index)
                self.generation_responses.append(selected)
                return selected
        raise AssertionError("unexpected generation source")


def test_fixture_a_selects_all_24_real_cases_and_is_deterministic(tmp_path: Path) -> None:
    threads = tuple(
        _thread(index, topic=("ops", "product", "people")[index % 3]) for index in range(24)
    )
    config = BenchmarkBuildConfig(seed=17, max_cases=30, output_dir=tmp_path / "run")

    first_provider = DeterministicProvider()
    first = build_benchmark(
        threads=threads,
        documents=tuple(),
        config=config,
        provider=first_provider,
    )
    second_provider = DeterministicProvider()
    second = build_benchmark(
        threads=threads,
        documents=tuple(),
        config=config.model_copy(update={"output_dir": tmp_path / "second"}),
        provider=second_provider,
    )

    assert first.status is BenchmarkStatus.OK
    assert len(first.candidate_cases) == 24
    assert all(not case.generated for case in first.candidate_cases)
    assert first.manifest_hash == second.manifest_hash
    assert first.candidate_cases == second.candidate_cases
    assert len(first.candidate_cases) <= 30
    assert first.generation_usage.generated_cases == 0
    assert first_provider.oracle_calls == 24
    assert first_provider.generation_calls == 0
    assert second_provider.oracle_calls == 24


def test_diversity_and_maximum_are_fixed_seeded(tmp_path: Path) -> None:
    threads = tuple(
        _thread(index, topic=("ops", "product", "people", "security")[index % 4])
        for index in range(48)
    )
    config = BenchmarkBuildConfig(seed=99, max_cases=25, output_dir=tmp_path / "run")

    result = build_benchmark(
        threads=threads,
        documents=tuple(),
        config=config,
        provider=DeterministicProvider(),
    )

    assert result.status is BenchmarkStatus.OK
    assert len(result.candidate_cases) == 25
    assert len({case.topic for case in result.candidate_cases}) >= 4
    assert [case.case_id for case in result.candidate_cases] == [
        case.case_id
        for case in build_benchmark(
            threads=threads,
            documents=tuple(),
            config=config.model_copy(update={"output_dir": tmp_path / "second"}),
            provider=DeterministicProvider(),
        ).candidate_cases
    ]


def test_fixture_b_generates_exact_missing_count_and_writes_no_oracle_to_candidates(
    tmp_path: Path,
) -> None:
    threads = tuple(_thread(index) for index in range(7))
    documents = tuple(
        _document(
            f"notion:policy-{index}",
            f"The {topic} policy requires {index + 1} review steps and is owned by team {topic}.",
        )
        for index, topic in enumerate(
            (
                "security",
                "billing",
                "support",
                "product",
                "people",
                "release",
                "legal",
                "finance",
                "data",
                "sales",
                "ops",
                "design",
                "qa",
            )
        )
    )

    responses: list[GenerationResponse] = [
        GenerationResponse(
            question=f"What does the {doc.metadata['topic']} policy require?",
            source_ids=[doc.source_id],
            expected_claims=[doc.text.split(".")[0]],
            forbidden_contradictions=["The policy has no review steps."],
            reference_text=doc.text,
        )
        for doc in documents
    ]

    provider = DeterministicProvider(responses)
    result = build_benchmark(
        threads=threads,
        documents=documents,
        config=BenchmarkBuildConfig(seed=3, output_dir=tmp_path / "run", max_cases=30),
        provider=provider,
    )

    assert result.status is BenchmarkStatus.OK
    assert len(result.candidate_cases) == 20
    assert sum(case.generated for case in result.candidate_cases) == 13
    assert (
        result.manifest_hash
        == build_benchmark(
            threads=threads,
            documents=documents,
            config=BenchmarkBuildConfig(seed=3, output_dir=tmp_path / "second", max_cases=30),
            provider=DeterministicProvider(responses),
        ).manifest_hash
    )
    candidate_payload = json.dumps(
        {
            "cases": [case.model_dump(mode="json") for case in result.candidate_cases],
            "corpus": [document.model_dump(mode="json") for document in result.candidate_documents],
        }
    )
    assert "reference_text" not in candidate_payload
    assert "expected_claims" not in candidate_payload
    assert '"generated"' not in candidate_payload
    assert not any(
        reply.source_id in candidate_payload for thread in threads for reply in thread.replies
    )
    evaluator_payload = (tmp_path / "run" / "evaluator" / "holdouts.jsonl").read_text()
    assert '"generated":true' in evaluator_payload
    assert any(document.text in evaluator_payload for document in documents)
    assert result.generation_usage.generated_cases == 13
    assert provider.oracle_calls == 7
    assert provider.generation_calls == 13


def test_over_cap_real_input_is_capped_and_order_independent(tmp_path: Path) -> None:
    threads = tuple(_thread(index) for index in range(40))
    first = build_benchmark(
        threads=threads,
        documents=tuple(),
        config=BenchmarkBuildConfig(seed=11, max_cases=30, output_dir=tmp_path / "first"),
        provider=DeterministicProvider(),
    )
    second = build_benchmark(
        threads=tuple(reversed(threads)),
        documents=tuple(),
        config=BenchmarkBuildConfig(seed=11, max_cases=30, output_dir=tmp_path / "second"),
        provider=DeterministicProvider(),
    )

    assert first.status is BenchmarkStatus.OK
    assert second.status is BenchmarkStatus.OK
    assert len(first.candidate_cases) == len(second.candidate_cases) == 30
    assert len(first.holdouts) == len(second.holdouts) == 30
    assert first.manifest_hash == second.manifest_hash
    assert [case.case_id for case in first.candidate_cases] == [
        case.case_id for case in second.candidate_cases
    ]


def test_generation_is_bounded_and_provider_output_must_be_structured(tmp_path: Path) -> None:
    class MalformedProvider:
        def extract_oracle(self, **_: Any) -> OracleResponse:
            return OracleResponse(
                expected_claims=["A valid human claim."],
                weak_human_confidence=0.7,
            )

        def generate(self, **_: Any) -> dict[str, str]:
            return {"unexpected": "shape"}

    output_dir = tmp_path / "run"
    result = build_benchmark(
        threads=tuple(_thread(index) for index in range(7)),
        documents=(_document("notion:one", "The rule is one."),),
        config=BenchmarkBuildConfig(seed=1, output_dir=output_dir, max_provider_calls=8),
        provider=MalformedProvider(),
    )

    assert result.status is BenchmarkStatus.PROVIDER_FAILED
    assert result.candidate_start_allowed is False
    assert not output_dir.exists()


def test_missing_provider_is_typed_when_generation_is_required(tmp_path: Path) -> None:
    result = build_benchmark(
        threads=tuple(_thread(index) for index in range(7)),
        documents=(_document("notion:one", "The rule is one."),),
        config=BenchmarkBuildConfig(output_dir=tmp_path / "run"),
        provider=None,
    )

    assert result.status is BenchmarkStatus.MISSING_PROVIDER
    assert result.candidate_cases == ()
    assert not (tmp_path / "run").exists()


def test_source_verification_rejects_unverifiable_generated_answer(tmp_path: Path) -> None:
    provider = DeterministicProvider(
        [
            GenerationResponse(
                question="What is unrelated?",
                source_ids=["notion:one"],
                expected_claims=["The source says something else."],
                forbidden_contradictions=[],
                reference_text="The source says something else.",
            )
        ]
    )
    result = build_benchmark(
        threads=tuple(_thread(index) for index in range(19)),
        documents=(_document("notion:one", "The rule is one."),),
        config=BenchmarkBuildConfig(output_dir=tmp_path / "run"),
        provider=provider,
    )

    assert result.status is BenchmarkStatus.INSUFFICIENT_BENCHMARK
    assert result.rejections
    assert any(
        rejection.reason == "UNVERIFIABLE_GENERATED_ANSWER" for rejection in result.rejections
    )


def test_provider_timeout_cancellation_and_budget_are_typed(tmp_path: Path) -> None:
    class TimeoutProvider:
        def extract_oracle(self, **_: Any) -> OracleResponse:
            while True:
                pass

        def generate(self, **_: Any) -> GenerationResponse:
            raise AssertionError("generation must not start after oracle timeout")

    result = build_benchmark(
        threads=tuple(_thread(index) for index in range(7)),
        documents=tuple(
            _document(f"notion:{index}", f"The rule is {index}.") for index in range(20)
        ),
        config=BenchmarkBuildConfig(
            output_dir=tmp_path / "run",
            max_provider_calls=64,
            generation_budget_usd=0.0001,
            provider_timeout_seconds=0.01,
        ),
        provider=TimeoutProvider(),
    )

    assert result.status is BenchmarkStatus.PROVIDER_TIMEOUT
    assert result.candidate_cases == ()
    assert result.generation_usage.provider_calls == 1
    assert not (tmp_path / "run").exists()


def test_provider_cancellation_is_typed_and_atomic(tmp_path: Path) -> None:
    class CancelledProvider:
        def extract_oracle(self, **_: Any) -> OracleResponse:
            raise asyncio.CancelledError

        def generate(self, **_: Any) -> GenerationResponse:
            raise AssertionError("generation must not start after oracle cancellation")

    output_dir = tmp_path / "cancelled"
    result = build_benchmark(
        threads=tuple(_thread(index) for index in range(7)),
        documents=(_document("notion:one", "One review is required."),),
        config=BenchmarkBuildConfig(output_dir=output_dir),
        provider=CancelledProvider(),
    )

    assert result.status is BenchmarkStatus.PROVIDER_CANCELLED
    assert result.generation_usage.provider_calls == 1
    assert not output_dir.exists()


def test_generation_cost_budget_exhaustion_is_typed(tmp_path: Path) -> None:
    provider = DeterministicProvider(
        [
            GenerationResponse(
                question="What review is required?",
                source_ids=["notion:one"],
                expected_claims=["One review is required."],
                reference_text="One review is required.",
                cost_usd=0.02,
            )
        ]
    )
    result = build_benchmark(
        threads=tuple(_thread(index) for index in range(19)),
        documents=(_document("notion:one", "One review is required."),),
        config=BenchmarkBuildConfig(
            output_dir=tmp_path / "budget",
            generation_budget_usd=0.01,
        ),
        provider=provider,
    )

    assert result.status is BenchmarkStatus.BUDGET_EXCEEDED
    assert result.generation_usage.cost_usd == 0.02
    assert result.candidate_start_allowed is False


def test_failure_does_not_leave_partial_candidate_or_evaluator_artifacts(tmp_path: Path) -> None:
    class MalformedProvider:
        def extract_oracle(self, **_: Any) -> OracleResponse:
            return OracleResponse(
                expected_claims=["A valid human claim."],
                weak_human_confidence=0.7,
            )

        def generate(self, **_: Any) -> dict[str, str]:
            return {"bad": "output"}

    output_dir = tmp_path / "run"
    result = build_benchmark(
        threads=tuple(_thread(index) for index in range(7)),
        documents=tuple(
            _document(f"notion:{index}", f"The rule is {index}.") for index in range(20)
        ),
        config=BenchmarkBuildConfig(output_dir=output_dir, max_provider_calls=8),
        provider=MalformedProvider(),
    )

    assert result.status is BenchmarkStatus.PROVIDER_FAILED
    assert not output_dir.exists()


def test_openai_compatible_provider_uses_fixed_model_temperature_and_metering() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def create(self, **kwargs: Any) -> Any:
            self.kwargs = kwargs
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {
                                        "content": json.dumps(
                                            {
                                                "question": "What is required?",
                                                "source_ids": ["notion:one"],
                                                "expected_claims": ["One review is required."],
                                                "forbidden_contradictions": [],
                                                "reference_text": "One review is required.",
                                            }
                                        )
                                    },
                                )()
                            },
                        )()
                    ],
                    "usage": type(
                        "Usage",
                        (),
                        {"prompt_tokens": 100, "completion_tokens": 25},
                    )(),
                },
            )()

    completions = FakeCompletions()
    fake_client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": completions})()},
    )()
    provider = OpenAICompatibleBenchmarkProvider(client=cast(Any, fake_client))

    response = provider.generate(
        prompt="source",
        model="gpt-5-mini",
        temperature=0,
        seed=11,
        timeout_seconds=2,
    )

    assert completions.kwargs["model"] == "gpt-5-mini"
    assert completions.kwargs["temperature"] == 0
    assert completions.kwargs["seed"] == 11
    assert completions.kwargs["timeout"] == 2
    assert response.input_tokens == 100
    assert response.output_tokens == 25
    assert response.cost_usd > 0


def test_real_holdout_oracles_use_provider_when_available(tmp_path: Path) -> None:
    class OracleProvider:
        calls = 0

        def extract_oracle(
            self,
            *,
            thread: SlackQuestionThread,
            model: str,
            temperature: int,
            seed: int,
            timeout_seconds: float,
        ) -> OracleResponse:
            self.calls += 1
            assert model == "gpt-5-mini"
            assert temperature == 0
            assert seed == 5
            assert timeout_seconds == 60
            return OracleResponse(
                expected_claims=[f"Verified {thread.topic} claim."],
                forbidden_contradictions=["The policy is absent."],
                weak_human_confidence=0.8,
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.001,
            )

        def generate(self, **_: Any) -> GenerationResponse:
            raise AssertionError("generated fallback is not required")

    provider = OracleProvider()
    result = build_benchmark(
        threads=tuple(_thread(index) for index in range(20)),
        documents=tuple(),
        config=BenchmarkBuildConfig(seed=5, output_dir=tmp_path / "run"),
        provider=provider,
    )

    assert result.status is BenchmarkStatus.OK
    assert provider.calls == 20
    assert result.generation_usage.provider_calls == 20
    assert result.generation_usage.generated_cases == 0
    assert all(holdout.expected_claims == ("Verified ops claim.",) for holdout in result.holdouts)


def test_selection_records_chatter_poll_bot_and_speculation_rejections(tmp_path: Path) -> None:
    base = _thread(100)
    chatter = base.model_copy(
        update={
            "root": base.root.model_copy(update={"text": "Happy birthday everyone?"}),
        }
    )
    poll = base.model_copy(
        update={
            "root": base.root.model_copy(
                update={"source_id": "slack:poll:root", "text": "Poll: choose one option?"}
            ),
        }
    )
    bot = base.model_copy(
        update={
            "root": base.root.model_copy(update={"source_id": "slack:bot:root"}),
            "root_is_bot": True,
        }
    )
    speculation = base.model_copy(
        update={
            "root": base.root.model_copy(update={"source_id": "slack:guess:root"}),
            "replies": (
                base.replies[0].model_copy(update={"text": "Maybe it could be handled later."}),
            ),
        }
    )

    result = build_benchmark(
        threads=(chatter, poll, bot, speculation),
        documents=tuple(),
        config=BenchmarkBuildConfig(output_dir=tmp_path / "run"),
    )

    assert result.status is BenchmarkStatus.INSUFFICIENT_BENCHMARK
    assert {item.reason for item in result.rejections} == {
        "SOCIAL_CHATTER_OR_POLL",
        "BOT_ROOT",
        "UNRESOLVED_SPECULATION",
    }


def test_case_bounds_are_always_twenty_to_thirty() -> None:
    with pytest.raises(ValidationError):
        BenchmarkBuildConfig(min_cases=19)
    with pytest.raises(ValidationError):
        BenchmarkBuildConfig(max_cases=19)
    with pytest.raises(ValidationError):
        BenchmarkBuildConfig(max_cases=31)


def test_real_cases_require_the_evaluator_provider() -> None:
    result = build_benchmark(
        threads=tuple(_thread(index) for index in range(20)),
        documents=tuple(),
    )

    assert result.status is BenchmarkStatus.MISSING_PROVIDER
    assert result.candidate_start_allowed is False


def test_strict_builder_never_allows_an_undersized_candidate_benchmark() -> None:
    provider = DeterministicProvider()

    result = build_benchmark(
        threads=tuple(_thread(index) for index in range(19)),
        documents=tuple(),
        provider=provider,
    )

    assert result.status is BenchmarkStatus.INSUFFICIENT_BENCHMARK
    assert result.candidate_cases == ()
    assert result.candidate_start_allowed is False
    assert result.candidate_started is False
    assert provider.oracle_calls == 0


def test_no_fallback_sources_is_insufficient_before_provider_is_required() -> None:
    result = build_benchmark(
        threads=tuple(_thread(index) for index in range(7)),
        documents=tuple(),
    )

    assert result.status is BenchmarkStatus.INSUFFICIENT_BENCHMARK
    assert result.generation_usage.provider_calls == 0
    assert result.candidate_start_allowed is False


def test_generated_reference_text_must_be_source_verifiable(tmp_path: Path) -> None:
    provider = DeterministicProvider(
        [
            GenerationResponse(
                question="What review is required?",
                source_ids=["notion:one"],
                expected_claims=["One review is required."],
                reference_text="Two reviews are required.",
            )
        ]
    )

    result = build_benchmark(
        threads=tuple(_thread(index) for index in range(19)),
        documents=(_document("notion:one", "One review is required."),),
        config=BenchmarkBuildConfig(output_dir=tmp_path / "run"),
        provider=provider,
    )

    assert result.status is BenchmarkStatus.INSUFFICIENT_BENCHMARK
    assert any(
        rejection.reason == "UNVERIFIABLE_GENERATED_ANSWER" for rejection in result.rejections
    )
    assert not (tmp_path / "run").exists()


def test_generated_question_cannot_copy_a_source_sentence(tmp_path: Path) -> None:
    provider = DeterministicProvider(
        [
            GenerationResponse(
                question="One review is required?",
                source_ids=["notion:one"],
                expected_claims=["One review is required."],
                reference_text="One review is required.",
            )
        ]
    )

    result = build_benchmark(
        threads=tuple(_thread(index) for index in range(19)),
        documents=(_document("notion:one", "One review is required."),),
        config=BenchmarkBuildConfig(output_dir=tmp_path / "run"),
        provider=provider,
    )

    assert result.status is BenchmarkStatus.INSUFFICIENT_BENCHMARK
    assert any(rejection.reason == "COPIED_SENTENCE_QUESTION" for rejection in result.rejections)


def test_write_failure_is_atomic_and_cleans_temporary_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write_text = Path.write_text

    def fail_manifest(path: Path, data: str, **kwargs: Any) -> int:
        if path.name == "manifest.json":
            raise OSError("disk full")
        return original_write_text(path, data, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_manifest)
    output_dir = tmp_path / "run"

    with pytest.raises(OSError, match="disk full"):
        build_benchmark(
            threads=tuple(_thread(index) for index in range(20)),
            documents=tuple(),
            config=BenchmarkBuildConfig(output_dir=output_dir),
            provider=DeterministicProvider(),
        )

    assert not output_dir.exists()
    assert list(tmp_path.iterdir()) == []
