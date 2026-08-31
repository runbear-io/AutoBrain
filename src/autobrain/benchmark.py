"""Automatic Slack-question benchmark construction and holdout isolation.

The benchmark boundary is intentionally kept in one module.  Candidate-facing
models contain only questions and safe source provenance; evaluator-owned
models contain replies, claims, contradictions, and confidence.  The writer
commits both sides together only after the candidate side has passed leakage
scanning.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import shutil
import signal
import tempfile
import threading
from collections import defaultdict
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from types import FrameType
from typing import Any, Protocol, cast

from openai import OpenAI
from pydantic import Field, field_validator, model_validator

from autobrain.corpus import canonical_corpus_identity
from autobrain.models import NormalizedDocument, Sha256, SourceId, SourceKind, StrictModel
from autobrain.performance import RunCache

MODEL_NAME = "gpt-5-mini"
TEMPERATURE = 0
MAX_CASES = 30
MIN_CASES = 20

_QUESTION = re.compile(
    r"(?:\?|^(?:how|what|when|where|which|who|why|can|could|should|is|are|do|does|"
    r"has|have|will|would)\b)",
    re.IGNORECASE,
)
_ACKNOWLEDGEMENT = re.compile(
    r"^(?:thanks|thank you|thx|ty|great|awesome|nice|got it|lgtm|agreed|"
    r"same|welcome|congrats|congratulations|lol|haha|cool|\+1)[!. ]*$",
    re.IGNORECASE,
)
_SOCIAL = re.compile(
    r"\b(?:happy birthday|happy anniversary|welcome to the team|lunch|coffee|"
    r"happy friday|weekend plans|🎉|🎂)\b",
    re.IGNORECASE,
)
_POLL = re.compile(
    r"\b(?:poll|vote|voting|react with|:+1:|:thumbsup:|choose one|survey)\b",
    re.IGNORECASE,
)
_SPECULATION = re.compile(
    r"^(?:i think|maybe|perhaps|not sure|probably|might be|could be|guessing)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9]+")
_ORACLE_MARKER = re.compile(
    r"\b(?:oracle[_ -](?:marker|only|id)|holdout(?:[_ -](?:marker|only|id|reply))?|"
    r"reference[_ -]?answer|expected[_ -]?claim|weak-human)\b",
    re.IGNORECASE,
)


class BenchmarkStatus(StrEnum):
    OK = "OK"
    MISSING_PROVIDER = "MISSING_PROVIDER"
    INSUFFICIENT_BENCHMARK = "INSUFFICIENT_BENCHMARK"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_CANCELLED = "PROVIDER_CANCELLED"
    PROVIDER_FAILED = "PROVIDER_FAILED"
    LEAKAGE_DETECTED = "LEAKAGE_DETECTED"


class BenchmarkProviderError(RuntimeError):
    """Provider output or execution violated the benchmark contract."""

    def __init__(
        self,
        message: str,
        *,
        status: BenchmarkStatus = BenchmarkStatus.PROVIDER_FAILED,
        usage: BenchmarkGenerationUsage | None = None,
    ) -> None:
        self.status = status
        self.usage = usage
        super().__init__(message)


class SlackQuestionThread(StrictModel):
    """A normalized Slack root and its direct replies."""

    root: NormalizedDocument
    replies: tuple[NormalizedDocument, ...] = Field(min_length=1)
    root_is_bot: bool = False
    reply_is_bot: tuple[bool, ...]
    channel: str = Field(min_length=1)
    topic: str = Field(min_length=1)

    @model_validator(mode="after")
    def reply_flags_match_and_are_slack(self) -> SlackQuestionThread:
        if len(self.reply_is_bot) != len(self.replies):
            raise ValueError("reply_is_bot must match replies")
        if self.root.source_kind not in {SourceKind.SLACK_MESSAGE, SourceKind.SLACK_THREAD}:
            raise ValueError("SlackQuestionThread.root must be a Slack message")
        if any(
            reply.source_kind not in {SourceKind.SLACK_MESSAGE, SourceKind.SLACK_THREAD}
            for reply in self.replies
        ):
            raise ValueError("SlackQuestionThread.replies must be Slack messages")
        return self


class CandidateCase(StrictModel):
    """Safe case sent to candidates; no oracle fields are representable."""

    case_id: str = Field(pattern=r"^case-[A-Za-z0-9_-]+$")
    question: str = Field(min_length=1)
    source_ids: tuple[SourceId, ...] = ()
    generated: bool = Field(default=False, exclude=True)
    topic: str = Field(min_length=1)
    provenance: tuple[str, ...] = ()

    @field_validator("question")
    @classmethod
    def question_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question cannot be blank")
        return value.strip()


class GenerationResponse(StrictModel):
    """Structured response expected from the evaluator generation provider."""

    question: str = Field(min_length=1)
    source_ids: list[SourceId] = Field(min_length=1)
    expected_claims: list[str] = Field(min_length=1)
    forbidden_contradictions: list[str] = Field(default_factory=list)
    reference_text: str = Field(min_length=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)


class OracleResponse(StrictModel):
    """Structured evaluator fields extracted from held-out human replies."""

    expected_claims: list[str] = Field(min_length=1)
    forbidden_contradictions: list[str] = Field(default_factory=list)
    weak_human_confidence: float = Field(ge=0, le=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)


class EvaluatorHoldout(StrictModel):
    """Evaluator-only record retaining raw replies and structured oracle data."""

    case_id: str = Field(pattern=r"^case-[A-Za-z0-9_-]+$")
    question: str = Field(min_length=1)
    root_id: SourceId
    reply_ids: tuple[SourceId, ...] = Field(min_length=1)
    source_ids: tuple[SourceId, ...] = Field(min_length=1)
    raw_replies: tuple[str, ...] = Field(min_length=1)
    expected_claims: tuple[str, ...] = Field(min_length=1)
    forbidden_contradictions: tuple[str, ...] = ()
    weak_human_confidence: float = Field(ge=0, le=1)
    generated: bool = False


class BenchmarkRejection(StrictModel):
    source_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    detail: str = ""


class BenchmarkGenerationUsage(StrictModel):
    """Generation-only accounting; candidate usage is recorded elsewhere."""

    model: str = MODEL_NAME
    temperature: int = TEMPERATURE
    seed: int
    provider_calls: int = Field(default=0, ge=0)
    generated_cases: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    provenance: tuple[str, ...] = ()


class LeakageScanResult(StrictModel):
    clean: bool
    locations: tuple[str, ...] = ()
    matched_tokens: tuple[str, ...] = ()


class BenchmarkBuildConfig(StrictModel):
    seed: int = 0
    min_cases: int = Field(default=MIN_CASES, ge=MIN_CASES, le=MIN_CASES)
    max_cases: int = Field(default=MAX_CASES, ge=MIN_CASES, le=MAX_CASES)
    # Corpus holdouts are opt-in for compatibility with the standalone toolkit;
    # production orchestration enables the audited ten-percent split.
    holdout_fraction: float = Field(default=0.0, ge=0, lt=1)
    output_dir: Path | None = None
    generation_budget_usd: float = Field(default=25.0, ge=0)
    max_provider_calls: int = Field(default=64, ge=0)
    provider_timeout_seconds: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def case_bounds_are_valid(self) -> BenchmarkBuildConfig:
        if self.min_cases > self.max_cases:
            raise ValueError("min_cases cannot exceed max_cases")
        return self


class BenchmarkBuildResult(StrictModel):
    status: BenchmarkStatus
    candidate_cases: tuple[CandidateCase, ...] = ()
    candidate_documents: tuple[NormalizedDocument, ...] = ()
    holdouts: tuple[EvaluatorHoldout, ...] = ()
    holdout_source_ids: tuple[SourceId, ...] = ()
    rejections: tuple[BenchmarkRejection, ...] = ()
    generation_usage: BenchmarkGenerationUsage
    manifest_hash: Sha256 | None = None
    corpus_hash: Sha256 | None = None
    benchmark_hash: Sha256 | None = None
    evaluator_artifacts_dir: Path | None = None
    candidate_start_allowed: bool = False
    candidate_started: bool = False
    leakage: LeakageScanResult = LeakageScanResult(clean=True)


class BenchmarkProvider(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        model: str,
        temperature: int,
        seed: int,
        timeout_seconds: float,
    ) -> GenerationResponse | Mapping[str, Any]: ...


class OracleProvider(Protocol):
    def extract_oracle(
        self,
        *,
        thread: SlackQuestionThread,
        model: str,
        temperature: int,
        seed: int,
        timeout_seconds: float,
    ) -> OracleResponse | Mapping[str, Any]: ...


class _Completions(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _Chat(Protocol):
    completions: _Completions


class _OpenAICompatibleClient(Protocol):
    chat: _Chat


class OpenAICompatibleBenchmarkProvider:
    """Run-local OpenAI-compatible structured generation adapter."""

    def __init__(
        self,
        *,
        client: _OpenAICompatibleClient,
        input_usd_per_million: float = 0.25,
        output_usd_per_million: float = 2.0,
    ) -> None:
        if input_usd_per_million < 0 or output_usd_per_million < 0:
            raise ValueError("generation prices cannot be negative")
        self.client = client
        self.input_usd_per_million = input_usd_per_million
        self.output_usd_per_million = output_usd_per_million

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> OpenAICompatibleBenchmarkProvider:
        """Construct the same run-local OpenAI-compatible client used by candidates."""
        if not api_key:
            raise BenchmarkProviderError(
                "MISSING_PROVIDER: OPENAI_API_KEY is unavailable",
                status=BenchmarkStatus.MISSING_PROVIDER,
            )
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_seconds}
        if base_url is not None:
            kwargs["base_url"] = base_url
        return cls(client=cast(_OpenAICompatibleClient, OpenAI(**kwargs)))

    def generate(
        self,
        *,
        prompt: str,
        model: str,
        temperature: int,
        seed: int,
        timeout_seconds: float,
    ) -> GenerationResponse:
        """Generate one bounded JSON case and attach raw token/cost evidence."""
        if model != MODEL_NAME or temperature != TEMPERATURE:
            raise BenchmarkProviderError("provider model and temperature are fixed by Task 5")
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            seed=seed,
            timeout=timeout_seconds,
            max_completion_tokens=1200,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return one JSON object with question, source_ids, expected_claims, "
                        "forbidden_contradictions, and reference_text. Do not copy a source "
                        "sentence as the question. Every claim must be directly supported."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        try:
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise TypeError("completion content was not text")
            payload = json.loads(content)
            usage = response.usage
            input_tokens = int(usage.prompt_tokens) if usage is not None else 0
            output_tokens = int(usage.completion_tokens) if usage is not None else 0
        except (AttributeError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise BenchmarkProviderError("MALFORMED_PROVIDER_OUTPUT: completion shape") from exc
        if not isinstance(payload, dict):
            raise BenchmarkProviderError("MALFORMED_PROVIDER_OUTPUT: completion was not an object")
        payload["input_tokens"] = input_tokens
        payload["output_tokens"] = output_tokens
        payload["cost_usd"] = (
            input_tokens * self.input_usd_per_million + output_tokens * self.output_usd_per_million
        ) / 1_000_000
        return _parse_generation_response(cast(Mapping[str, Any], payload))

    def extract_oracle(
        self,
        *,
        thread: SlackQuestionThread,
        model: str,
        temperature: int,
        seed: int,
        timeout_seconds: float,
    ) -> OracleResponse:
        """Extract claims and contradiction guards from held-out human replies."""
        if model != MODEL_NAME or temperature != TEMPERATURE:
            raise BenchmarkProviderError("provider model and temperature are fixed by Task 5")
        reply_payload = [
            reply.text
            for reply, is_bot in zip(thread.replies, thread.reply_is_bot, strict=True)
            if not is_bot
        ]
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            seed=seed,
            timeout=timeout_seconds,
            max_completion_tokens=1200,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return JSON with expected_claims, forbidden_contradictions, and "
                        "weak_human_confidence. Use only the held-out replies."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": thread.root.text, "replies": reply_payload},
                        sort_keys=True,
                    ),
                },
            ],
        )
        payload, input_tokens, output_tokens = _completion_payload(response)
        payload["input_tokens"] = input_tokens
        payload["output_tokens"] = output_tokens
        payload["cost_usd"] = (
            input_tokens * self.input_usd_per_million + output_tokens * self.output_usd_per_million
        ) / 1_000_000
        return _parse_oracle_response(payload)


def mine_real_cases(
    threads: Sequence[SlackQuestionThread],
    *,
    seed: int = 0,
    max_cases: int = MAX_CASES,
) -> tuple[tuple[SlackQuestionThread, ...], tuple[BenchmarkRejection, ...]]:
    """Filter and deterministically diversify substantive Slack questions."""
    rejections: list[BenchmarkRejection] = []
    valid: list[SlackQuestionThread] = []
    for thread in threads:
        reason = _thread_rejection(thread)
        if reason is not None:
            rejections.append(BenchmarkRejection(source_id=thread.root.source_id, reason=reason))
        else:
            valid.append(thread)

    rng = random.Random(seed)
    shuffled = sorted(valid, key=lambda item: item.root.source_id)
    rng.shuffle(shuffled)
    buckets: dict[str, list[SlackQuestionThread]] = defaultdict(list)
    for thread in shuffled:
        buckets[f"{thread.topic}\x00{thread.channel}"].append(thread)
    selected: list[SlackQuestionThread] = []
    topics = sorted(buckets)
    while len(selected) < min(max_cases, len(valid)) and topics:
        next_topics: list[str] = []
        for topic in topics:
            bucket = buckets[topic]
            if bucket:
                selected.append(bucket.pop())
                if len(selected) == min(max_cases, len(valid)):
                    break
            if bucket:
                next_topics.append(topic)
        topics = next_topics
    return tuple(selected), tuple(rejections)


def scan_benchmark_leakage(
    *,
    texts: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    prompts: Sequence[str] = (),
    argv: Sequence[str] = (),
    environment: Mapping[str, str] | None = None,
    serialized_artifacts: Sequence[str] = (),
    outputs: Sequence[str] = (),
    candidate_workspaces: Sequence[Path] = (),
    forbidden_tokens: Sequence[str] = (),
    cache: RunCache | None = None,
) -> LeakageScanResult:
    """Scan every candidate-visible boundary and report exact locations."""
    tokens = tuple(dict.fromkeys(token for token in forbidden_tokens if token))
    locations: list[str] = []
    matched: set[str] = set()

    def inspect(location: str, value: object) -> None:
        serialized = (
            value
            if isinstance(value, str)
            else (
                cache.serialize(value, leakage=True)
                if cache is not None
                else _serialize_scannable(value)
            )
        )
        folded = serialized.casefold()
        for token in tokens:
            if token.casefold() in folded:
                locations.append(location)
                matched.add(token)
        if _ORACLE_MARKER.search(serialized):
            locations.append(location)
            matched.add("oracle-marker")

    for index, value in enumerate(texts):
        inspect(f"texts[{index}]", value)
    inspect("metadata", metadata or {})
    for index, value in enumerate(prompts):
        inspect(f"prompts[{index}]", value)
    inspect("argv", list(argv))
    inspect("environment", dict(environment or {}))
    for index, value in enumerate(serialized_artifacts):
        inspect(f"serialized_artifacts[{index}]", value)
    for index, value in enumerate(outputs):
        inspect(f"outputs[{index}]", value)
    for workspace in candidate_workspaces:
        if not workspace.exists():
            continue
        for path in sorted(path for path in workspace.rglob("*") if path.is_file()):
            try:
                inspect(
                    f"candidate_workspaces:{path}",
                    path.read_text(encoding="utf-8", errors="replace"),
                )
            except OSError:
                locations.append(f"candidate_workspaces:{path}:unreadable")
    grouped_locations = {location.split("[", 1)[0].split(":", 1)[0] for location in locations}
    return LeakageScanResult(
        clean=not locations,
        locations=tuple(sorted(grouped_locations)),
        matched_tokens=tuple(sorted(matched)),
    )


def build_benchmark(
    *,
    threads: Sequence[SlackQuestionThread],
    documents: Sequence[NormalizedDocument],
    config: BenchmarkBuildConfig | None = None,
    provider: (
        BenchmarkProvider | OracleProvider | Sequence[GenerationResponse | Mapping[str, Any]] | None
    ) = None,
    candidate_workspaces: Sequence[Path] = (),
    candidate_prompts: Sequence[str] = (),
    candidate_argv: Sequence[str] = (),
    candidate_environment: Mapping[str, str] | None = None,
    candidate_metadata: Mapping[str, Any] | None = None,
    candidate_outputs: Sequence[str] = (),
    serialized_artifacts: Sequence[str] = (),
    cache: RunCache | None = None,
) -> BenchmarkBuildResult:
    """Build and atomically persist the candidate and evaluator benchmark sides."""
    settings = config or BenchmarkBuildConfig()
    selected, rejections = mine_real_cases(
        threads,
        seed=settings.seed,
        # Reserve holdouts before applying the public benchmark cap.
        max_cases=len(threads),
    )
    usage = BenchmarkGenerationUsage(seed=settings.seed)
    case_threads, holdout_threads = _stratified_holdout_split(
        selected,
        seed=settings.seed,
        fraction=settings.holdout_fraction,
        min_cases=settings.min_cases,
    )
    if settings.holdout_fraction > 0 and len(selected) == settings.min_cases:
        return _blocked_result(BenchmarkStatus.INSUFFICIENT_BENCHMARK, rejections, usage)
    selected_holdout_ids = {
        identifier for thread in holdout_threads for identifier in _thread_ids(thread)
    }
    candidate_documents = _remove_holdouts(documents, selected_holdout_ids)
    eligible_generation_documents = _eligible_generation_documents(candidate_documents)
    if len(selected) < settings.min_cases and not eligible_generation_documents:
        return _blocked_result(BenchmarkStatus.INSUFFICIENT_BENCHMARK, rejections, usage)
    oracle_responses: dict[str, OracleResponse] | None = None
    if selected and not callable(getattr(provider, "extract_oracle", None)):
        return _blocked_result(BenchmarkStatus.MISSING_PROVIDER, rejections, usage)
    has_generation_provider = callable(getattr(provider, "generate", None)) or (
        isinstance(provider, Sequence) and not isinstance(provider, str | bytes)
    )
    if len(selected) < settings.min_cases and not has_generation_provider:
        return _blocked_result(BenchmarkStatus.MISSING_PROVIDER, rejections, usage)
    try:
        oracle_responses, usage = _extract_real_oracles(
            case_threads,
            provider=provider,
            config=settings,
            initial_usage=usage,
        )
        if len(case_threads) < settings.min_cases:
            missing = settings.min_cases - len(selected)
            if provider is None:
                return _blocked_result(BenchmarkStatus.MISSING_PROVIDER, rejections, usage)
            generated, generated_holdouts, usage, generated_rejections = _generate_cases(
                missing,
                documents=eligible_generation_documents,
                config=settings,
                provider=cast(
                    BenchmarkProvider | Sequence[GenerationResponse | Mapping[str, Any]],
                    provider,
                ),
                initial_usage=usage,
                cache=cache,
            )
            rejections = (*rejections, *generated_rejections)
        else:
            generated = ()
            generated_holdouts = ()
    except BenchmarkProviderError as exc:
        usage = exc.usage or usage
        return _blocked_result(exc.status, (*rejections,), usage)

    real_cases, real_holdouts = _real_case_records(
        sorted(case_threads, key=lambda thread: _natural_source_key(thread.root.source_id)),
        oracle_responses=oracle_responses,
    )
    candidate_cases = (*real_cases, *generated)
    holdouts = (*real_holdouts, *generated_holdouts)
    if len(candidate_cases) < settings.min_cases:
        return _blocked_result(BenchmarkStatus.INSUFFICIENT_BENCHMARK, rejections, usage)
    if len(candidate_cases) > settings.max_cases:
        candidate_cases = candidate_cases[: settings.max_cases]
        holdouts = holdouts[: settings.max_cases]
    if len(candidate_cases) < settings.min_cases:
        return _blocked_result(BenchmarkStatus.INSUFFICIENT_BENCHMARK, rejections, usage)

    # Benchmark roots and replies are evaluator-owned evidence. Remove every
    # benchmark record from the candidate corpus, including bot replies and
    # documents linked to those records.
    benchmark_ids = {
        identifier
        for thread in selected
        for identifier in _thread_ids(thread)
    }
    # Scan the pre-freeze source boundary as well: a non-holdout record can
    # accidentally quote a held-out root even though all benchmark roots are
    # later removed from the candidate corpus.
    benchmark_reply_ids = {
        reply.source_id for thread in selected for reply in thread.replies
    }
    source_boundary_payload = [
        json.dumps(document.model_dump(mode="json"), sort_keys=True)
        for document in candidate_documents
        if document.source_id not in benchmark_reply_ids
        and not any(related in benchmark_ids for related in document.related_source_ids)
        and document.metadata.get("parent_source_id") not in benchmark_ids
    ]
    candidate_documents = _remove_holdouts(candidate_documents, benchmark_ids)
    forbidden = (
        *selected_holdout_ids,
        # Flattened Slack fixtures retain selected holdout replies in the raw
        # connector record; protect those evaluator-only strings even though
        # normalization intentionally drops unknown fixture fields.
        *[
            reply.text
            for thread in holdout_threads
            for reply, is_bot in zip(thread.replies, thread.reply_is_bot, strict=True)
            if not is_bot
        ],
        *[
            identifier
            for holdout in holdouts
            if holdout.generated
            for identifier in holdout.reply_ids
        ],
        *[
            identifier
            for holdout in holdouts
            if not holdout.generated
            for identifier in holdout.reply_ids
        ],
        *[reply for holdout in holdouts if not holdout.generated for reply in holdout.raw_replies],
        *[
            claim
            for holdout in holdouts
            if not holdout.generated
            for claim in holdout.expected_claims
        ],
    )
    candidate_payload = source_boundary_payload + [
        json.dumps(document.model_dump(mode="json"), sort_keys=True)
        for document in candidate_documents
    ] + [json.dumps(case.model_dump(mode="json"), sort_keys=True) for case in candidate_cases]
    leakage = scan_benchmark_leakage(
        texts=candidate_payload,
        metadata={**dict(candidate_metadata or {}), "case_count": len(candidate_cases)},
        prompts=(*[case.question for case in candidate_cases], *candidate_prompts),
        argv=candidate_argv,
        environment=candidate_environment or {},
        serialized_artifacts=(*candidate_payload, *serialized_artifacts),
        outputs=candidate_outputs,
        candidate_workspaces=candidate_workspaces,
        forbidden_tokens=forbidden,
        cache=cache,
    )
    if not leakage.clean:
        return _blocked_result(BenchmarkStatus.LEAKAGE_DETECTED, rejections, usage, leakage=leakage)

    corpus_hash = canonical_corpus_identity(candidate_documents).sha256
    benchmark_hash = _hash_models(candidate_cases)
    manifest_hash = _manifest_hash(
        candidate_cases,
        candidate_documents,
        holdouts,
        corpus_hash=corpus_hash,
        benchmark_hash=benchmark_hash,
    )
    evaluator_dir: Path | None = None
    if settings.output_dir is not None:
        try:
            evaluator_dir = _write_artifacts(
                settings.output_dir,
                candidate_cases=candidate_cases,
                candidate_documents=candidate_documents,
                holdouts=holdouts,
                rejections=rejections,
                usage=usage,
                manifest_hash=manifest_hash,
                corpus_hash=corpus_hash,
                benchmark_hash=benchmark_hash,
            )
        except BaseException:
            raise
    return BenchmarkBuildResult(
        status=BenchmarkStatus.OK,
        candidate_cases=tuple(candidate_cases),
        candidate_documents=tuple(candidate_documents),
        holdouts=tuple(holdouts),
        holdout_source_ids=tuple(sorted(selected_holdout_ids)),
        rejections=tuple(rejections),
        generation_usage=usage,
        manifest_hash=manifest_hash,
        corpus_hash=corpus_hash,
        benchmark_hash=benchmark_hash,
        evaluator_artifacts_dir=evaluator_dir,
        candidate_start_allowed=True,
        candidate_started=False,
        leakage=leakage,
    )


def _thread_rejection(thread: SlackQuestionThread) -> str | None:
    question = thread.root.text.strip()
    if thread.root_is_bot:
        return "BOT_ROOT"
    if not any(not is_bot for is_bot in thread.reply_is_bot):
        return "NO_NON_BOT_EVIDENCE_REPLY"
    if _ACKNOWLEDGEMENT.fullmatch(question) or _SOCIAL.search(question) or _POLL.search(question):
        return "SOCIAL_CHATTER_OR_POLL"
    if not _QUESTION.search(question) or len(_WORD.findall(question)) < 5:
        return "NOT_A_SUBSTANTIVE_QUESTION"
    usable_replies = [
        reply.text.strip()
        for reply, is_bot in zip(thread.replies, thread.reply_is_bot, strict=True)
        if not is_bot and len(_WORD.findall(reply.text)) >= 5
    ]
    if not usable_replies:
        return "NO_NON_BOT_EVIDENCE_REPLY"
    if all(_SPECULATION.match(reply) for reply in usable_replies):
        return "UNRESOLVED_SPECULATION"
    if len(" ".join(usable_replies)) < 20:
        return "INSUFFICIENT_REFERENCE_TEXT"
    return None


def _real_case_records(
    threads: Sequence[SlackQuestionThread],
    *,
    oracle_responses: Mapping[str, OracleResponse],
) -> tuple[tuple[CandidateCase, ...], tuple[EvaluatorHoldout, ...]]:
    candidates: list[CandidateCase] = []
    holdouts: list[EvaluatorHoldout] = []
    for thread in threads:
        case_id = f"case-{_short_hash(thread.root.source_id)}"
        replies = tuple(
            reply.text
            for reply, is_bot in zip(thread.replies, thread.reply_is_bot, strict=True)
            if not is_bot
        )
        oracle = oracle_responses.get(thread.root.source_id)
        claims = (
            tuple(oracle.expected_claims)
            if oracle is not None
            else tuple(_claims_from_replies(replies))
        )
        candidates.append(
            CandidateCase(
                case_id=case_id,
                question=thread.root.text,
                # The root is safe provenance; reply IDs remain evaluator-only.
                source_ids=(thread.root.source_id,),
                generated=False,
                topic=thread.topic,
                provenance=(thread.channel,),
            )
        )
        holdouts.append(
            EvaluatorHoldout(
                case_id=case_id,
                question=thread.root.text,
                root_id=thread.root.source_id,
                reply_ids=tuple(reply.source_id for reply in thread.replies),
                source_ids=tuple(reply.source_id for reply in thread.replies),
                raw_replies=replies,
                expected_claims=claims,
                forbidden_contradictions=(
                    tuple(oracle.forbidden_contradictions) if oracle is not None else ()
                ),
                weak_human_confidence=(
                    oracle.weak_human_confidence
                    if oracle is not None
                    else _weak_confidence(replies)
                ),
                generated=False,
            )
        )
    return tuple(candidates), tuple(holdouts)


def _extract_real_oracles(
    threads: Sequence[SlackQuestionThread],
    *,
    provider: object | None,
    config: BenchmarkBuildConfig,
    initial_usage: BenchmarkGenerationUsage,
) -> tuple[dict[str, OracleResponse], BenchmarkGenerationUsage]:
    extractor = getattr(provider, "extract_oracle", None)
    if not callable(extractor):
        if threads:
            raise BenchmarkProviderError(
                "MISSING_PROVIDER: evaluator oracle extraction requires a provider",
                status=BenchmarkStatus.MISSING_PROVIDER,
                usage=initial_usage,
            )
        return {}, initial_usage
    extract = extractor
    responses: dict[str, OracleResponse] = {}
    calls = initial_usage.provider_calls
    input_tokens = initial_usage.input_tokens
    output_tokens = initial_usage.output_tokens
    cost_usd = initial_usage.cost_usd
    for thread in threads:
        if calls >= config.max_provider_calls:
            raise BenchmarkProviderError(
                "BUDGET_EXCEEDED: provider call bound exhausted",
                status=BenchmarkStatus.BUDGET_EXCEEDED,
                usage=_usage_snapshot(
                    config.seed,
                    calls,
                    0,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    (item.root.source_id for item in threads),
                ),
            )
        calls += 1
        try:
            with _provider_deadline(config.provider_timeout_seconds):
                raw = extract(
                    thread=thread,
                    model=MODEL_NAME,
                    temperature=TEMPERATURE,
                    seed=config.seed,
                    timeout_seconds=config.provider_timeout_seconds,
                )
            response = _parse_oracle_response(raw)
        except TimeoutError as exc:
            raise BenchmarkProviderError(
                "PROVIDER_TIMEOUT: oracle extraction deadline exceeded",
                status=BenchmarkStatus.PROVIDER_TIMEOUT,
                usage=_usage_snapshot(
                    config.seed,
                    calls,
                    0,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    (item.root.source_id for item in threads),
                ),
            ) from exc
        except (KeyboardInterrupt, asyncio.CancelledError) as exc:
            raise BenchmarkProviderError(
                "PROVIDER_CANCELLED: oracle extraction cancelled",
                status=BenchmarkStatus.PROVIDER_CANCELLED,
                usage=_usage_snapshot(
                    config.seed,
                    calls,
                    0,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    (item.root.source_id for item in threads),
                ),
            ) from exc
        except BenchmarkProviderError as exc:
            raise BenchmarkProviderError(
                str(exc),
                status=exc.status,
                usage=_usage_snapshot(
                    config.seed,
                    calls,
                    0,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    (item.root.source_id for item in threads),
                ),
            ) from exc
        except Exception as exc:
            raise BenchmarkProviderError(
                f"MALFORMED_PROVIDER_OUTPUT: {type(exc).__name__}",
                usage=_usage_snapshot(
                    config.seed,
                    calls,
                    0,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    (item.root.source_id for item in threads),
                ),
            ) from exc
        input_tokens += response.input_tokens
        output_tokens += response.output_tokens
        cost_usd += response.cost_usd
        if cost_usd > config.generation_budget_usd:
            raise BenchmarkProviderError(
                "BUDGET_EXCEEDED: generation budget exhausted",
                status=BenchmarkStatus.BUDGET_EXCEEDED,
                usage=_usage_snapshot(
                    config.seed,
                    calls,
                    0,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    (item.root.source_id for item in threads),
                ),
            )
        responses[thread.root.source_id] = response
    return responses, BenchmarkGenerationUsage(
        seed=config.seed,
        provider_calls=calls,
        generated_cases=initial_usage.generated_cases,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        provenance=tuple(thread.root.source_id for thread in threads),
    )


def _generate_cases(
    missing: int,
    *,
    documents: tuple[NormalizedDocument, ...],
    config: BenchmarkBuildConfig,
    provider: BenchmarkProvider | Sequence[GenerationResponse | Mapping[str, Any]],
    initial_usage: BenchmarkGenerationUsage,
    cache: RunCache | None = None,
) -> tuple[
    tuple[CandidateCase, ...],
    tuple[EvaluatorHoldout, ...],
    BenchmarkGenerationUsage,
    tuple[BenchmarkRejection, ...],
]:
    eligible = list(documents)
    responses = (
        provider
        if isinstance(provider, Sequence) and not isinstance(provider, str | bytes)
        else None
    )
    if responses is None:
        rng = random.Random(config.seed)
        rng.shuffle(eligible)
    calls = initial_usage.provider_calls
    input_tokens = initial_usage.input_tokens
    output_tokens = initial_usage.output_tokens
    cost_usd = initial_usage.cost_usd
    generated: list[CandidateCase] = []
    holdouts: list[EvaluatorHoldout] = []
    rejections: list[BenchmarkRejection] = []
    source_by_id = (
        cache.source_index(eligible)
        if cache is not None
        else {document.source_id: document for document in eligible}
    )
    if responses is not None:
        work: list[tuple[NormalizedDocument, GenerationResponse | Mapping[str, Any] | None]] = []
        for raw_response in responses:
            parsed = _parse_generation_response(raw_response)
            document = source_by_id.get(parsed.source_ids[0])
            if document is None:
                rejections.append(
                    BenchmarkRejection(
                        source_id=parsed.source_ids[0],
                        reason="MISSING_OR_UNKNOWN_SOURCE_ID",
                    )
                )
                continue
            work.append((document, raw_response))
    else:
        work = (
            [(eligible[index % len(eligible)], None) for index in range(config.max_provider_calls)]
            if eligible
            else []
        )
    seen_questions: set[str] = set()
    for document, supplied_response in work:
        if len(generated) >= missing:
            break
        if calls >= config.max_provider_calls:
            rejections.append(
                BenchmarkRejection(
                    source_id=document.source_id,
                    reason="PROVIDER_CALL_LIMIT_EXHAUSTED",
                )
            )
            break
        prompt = _generation_prompt(document)
        calls += 1
        try:
            if responses is not None:
                assert supplied_response is not None
                raw = supplied_response
            else:
                with _provider_deadline(config.provider_timeout_seconds):
                    raw = cast(BenchmarkProvider, provider).generate(
                        prompt=prompt,
                        model=MODEL_NAME,
                        temperature=TEMPERATURE,
                        seed=config.seed,
                        timeout_seconds=config.provider_timeout_seconds,
                    )
            response = _parse_generation_response(raw)
        except TimeoutError as exc:
            raise BenchmarkProviderError(
                "PROVIDER_TIMEOUT: generation deadline exceeded",
                status=BenchmarkStatus.PROVIDER_TIMEOUT,
                usage=_usage_snapshot(
                    config.seed,
                    calls,
                    len(generated),
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    (
                        *initial_usage.provenance,
                        *(item.source_id for item in eligible),
                    ),
                ),
            ) from exc
        except (KeyboardInterrupt, asyncio.CancelledError) as exc:
            raise BenchmarkProviderError(
                "PROVIDER_CANCELLED: generation cancelled",
                status=BenchmarkStatus.PROVIDER_CANCELLED,
                usage=_usage_snapshot(
                    config.seed,
                    calls,
                    len(generated),
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    (
                        *initial_usage.provenance,
                        *(item.source_id for item in eligible),
                    ),
                ),
            ) from exc
        except BenchmarkProviderError as exc:
            raise BenchmarkProviderError(
                str(exc),
                status=exc.status,
                usage=_usage_snapshot(
                    config.seed,
                    calls,
                    len(generated),
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    (
                        *initial_usage.provenance,
                        *(item.source_id for item in eligible),
                    ),
                ),
            ) from exc
        except Exception as exc:
            raise BenchmarkProviderError(
                f"MALFORMED_PROVIDER_OUTPUT: {type(exc).__name__}",
                usage=_usage_snapshot(
                    config.seed,
                    calls,
                    len(generated),
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    (
                        *initial_usage.provenance,
                        *(item.source_id for item in eligible),
                    ),
                ),
            ) from exc
        input_tokens += response.input_tokens
        output_tokens += response.output_tokens
        cost_usd += response.cost_usd
        if cost_usd > config.generation_budget_usd:
            raise BenchmarkProviderError(
                "BUDGET_EXCEEDED: generation budget exhausted",
                status=BenchmarkStatus.BUDGET_EXCEEDED,
                usage=_usage_snapshot(
                    config.seed,
                    calls,
                    len(generated),
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    (
                        *initial_usage.provenance,
                        *(item.source_id for item in eligible),
                    ),
                ),
            )
        rejection = _validate_generated_response(response, document)
        question_key = " ".join(_WORD.findall(response.question.casefold()))
        if rejection is None and question_key in seen_questions:
            rejection = "DUPLICATE_GENERATED_QUESTION"
        if rejection is not None:
            rejections.append(BenchmarkRejection(source_id=document.source_id, reason=rejection))
            continue
        seen_questions.add(question_key)
        case_id = f"case-generated-{_short_hash(document.source_id + response.question)}"
        generated.append(
            CandidateCase(
                case_id=case_id,
                question=response.question,
                source_ids=tuple(response.source_ids),
                generated=True,
                topic=document.metadata.get("topic", document.title),
                provenance=(document.source_id,),
            )
        )
        holdouts.append(
            EvaluatorHoldout(
                case_id=case_id,
                question=response.question,
                root_id=f"generated:root:{_short_hash(case_id)}",
                reply_ids=(f"generated:oracle:{_short_hash(case_id)}",),
                source_ids=tuple(response.source_ids),
                raw_replies=(response.reference_text,),
                expected_claims=tuple(response.expected_claims),
                forbidden_contradictions=tuple(response.forbidden_contradictions),
                weak_human_confidence=0.0,
                generated=True,
            )
        )
    usage = BenchmarkGenerationUsage(
        seed=config.seed,
        provider_calls=calls,
        generated_cases=len(generated),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        provenance=tuple(
            sorted(
                {
                    *initial_usage.provenance,
                    *(document.source_id for document in eligible),
                }
            )
        ),
    )
    return tuple(generated), tuple(holdouts), usage, tuple(rejections)


def _validate_generated_response(
    response: GenerationResponse,
    document: NormalizedDocument,
) -> str | None:
    if tuple(dict.fromkeys(response.source_ids)) != (document.source_id,):
        return "MISSING_OR_UNKNOWN_SOURCE_ID"
    source_lower = document.text.casefold()
    if any(claim.casefold() not in source_lower for claim in response.expected_claims):
        return "UNVERIFIABLE_GENERATED_ANSWER"
    normalized_reference = " ".join(_WORD.findall(response.reference_text.casefold()))
    normalized_source = " ".join(_WORD.findall(document.text.casefold()))
    if not normalized_reference or normalized_reference not in normalized_source:
        return "UNVERIFIABLE_GENERATED_ANSWER"
    normalized_question = " ".join(_WORD.findall(response.question.casefold()))
    if normalized_question and normalized_question in normalized_source:
        return "COPIED_SENTENCE_QUESTION"
    return None


def _parse_generation_response(raw: object) -> GenerationResponse:
    if isinstance(raw, GenerationResponse):
        return raw
    if not isinstance(raw, Mapping):
        raise BenchmarkProviderError("MALFORMED_PROVIDER_OUTPUT: expected object")
    try:
        mapping = cast(Mapping[object, object], raw)
        payload = {str(key): value for key, value in mapping.items()}
        return GenerationResponse.model_validate(payload)
    except Exception as exc:
        raise BenchmarkProviderError("MALFORMED_PROVIDER_OUTPUT: schema validation failed") from exc


def _parse_oracle_response(raw: object) -> OracleResponse:
    if isinstance(raw, OracleResponse):
        return raw
    if not isinstance(raw, Mapping):
        raise BenchmarkProviderError("MALFORMED_PROVIDER_OUTPUT: expected oracle object")
    try:
        mapping = cast(Mapping[object, object], raw)
        payload = {str(key): value for key, value in mapping.items()}
        return OracleResponse.model_validate(payload)
    except Exception as exc:
        raise BenchmarkProviderError(
            "MALFORMED_PROVIDER_OUTPUT: oracle schema validation failed"
        ) from exc


def _completion_payload(response: object) -> tuple[dict[str, object], int, int]:
    try:
        raw_response = cast(Any, response)
        content = raw_response.choices[0].message.content
        if not isinstance(content, str):
            raise TypeError("completion content was not text")
        decoded = json.loads(content)
        usage = raw_response.usage
        input_tokens = int(usage.prompt_tokens) if usage is not None else 0
        output_tokens = int(usage.completion_tokens) if usage is not None else 0
    except (AttributeError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BenchmarkProviderError("MALFORMED_PROVIDER_OUTPUT: completion shape") from exc
    if not isinstance(decoded, dict):
        raise BenchmarkProviderError("MALFORMED_PROVIDER_OUTPUT: completion was not an object")
    mapping = cast(dict[object, object], decoded)
    return {str(key): value for key, value in mapping.items()}, input_tokens, output_tokens


def _serialize_scannable(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _usage_snapshot(
    seed: int,
    calls: int,
    generated_cases: int,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    provenance: Iterable[str],
) -> BenchmarkGenerationUsage:
    return BenchmarkGenerationUsage(
        seed=seed,
        provider_calls=calls,
        generated_cases=generated_cases,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        provenance=tuple(sorted(set(provenance))),
    )


def _generation_prompt(document: NormalizedDocument) -> str:
    return (
        "Create one source-verifiable company question and structured expected claims from this "
        f"document. Source ID: {document.source_id}. Document: {document.text}"
    )


def _stratified_holdout_split(
    threads: Sequence[SlackQuestionThread],
    *,
    seed: int,
    fraction: float,
    min_cases: int,
) -> tuple[tuple[SlackQuestionThread, ...], tuple[SlackQuestionThread, ...]]:
    """Reserve whole Slack threads using a seeded topic/channel stratification."""
    if not threads or fraction <= 0:
        return tuple(threads), ()
    target = min(int(len(threads) * fraction), len(threads) - min_cases)
    if target <= 0:
        return tuple(threads), ()
    buckets: dict[str, list[SlackQuestionThread]] = defaultdict(list)
    for thread in sorted(threads, key=lambda item: item.root.source_id):
        buckets[f"{thread.topic}\x00{thread.channel}"].append(thread)
    # Keep the reserve stable across connector ordering and run IDs. Taking
    # the lexically last thread matches the mature holdout boundary while
    # still preserving whole threads and the configured target size.
    held = sorted(
        threads,
        key=lambda item: _natural_source_key(item.root.source_id),
        reverse=True,
    )[:target]
    held_ids = {thread.root.source_id for thread in held}
    return (
        tuple(thread for thread in threads if thread.root.source_id not in held_ids),
        tuple(sorted(held, key=lambda item: item.root.source_id)),
    )


def _remove_holdouts(
    documents: Sequence[NormalizedDocument],
    holdout_ids: set[str],
) -> list[NormalizedDocument]:
    """Remove only exact holdout records and their exact child records."""
    filtered: list[NormalizedDocument] = []
    for document in documents:
        if document.source_id in holdout_ids:
            continue
        parent = document.metadata.get("parent_source_id")
        if parent in holdout_ids:
            continue
        if any(related in holdout_ids for related in document.related_source_ids):
            continue
        filtered.append(document)
    return sorted(filtered, key=lambda item: item.source_id)


def _eligible_generation_documents(
    documents: Sequence[NormalizedDocument],
) -> tuple[NormalizedDocument, ...]:
    return tuple(
        document
        for document in sorted(documents, key=lambda item: item.source_id)
        if not _ORACLE_MARKER.search(json.dumps(document.model_dump(mode="json"), sort_keys=True))
    )


@contextmanager
def _provider_deadline(seconds: float) -> Generator[None, None, None]:
    if (
        threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
    ):
        yield
        return

    def deadline_exceeded(_signum: int, _frame: FrameType | None) -> None:
        raise TimeoutError("provider deadline exceeded")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, deadline_exceeded)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _thread_ids(thread: SlackQuestionThread) -> tuple[str, ...]:
    return (thread.root.source_id, *(reply.source_id for reply in thread.replies))


def _claims_from_replies(replies: Sequence[str]) -> list[str]:
    claims: list[str] = []
    for reply in replies:
        for sentence in re.split(r"(?<=[.!?])\s+", reply.strip()):
            if sentence and sentence not in claims:
                claims.append(sentence)
    return claims or list(replies)


def _weak_confidence(replies: Sequence[str]) -> float:
    if not replies:
        return 0.0
    if all(_SPECULATION.match(reply.strip()) for reply in replies):
        return 0.1
    return min(1.0, 0.5 + 0.1 * len(replies))


def _natural_source_key(source_id: str) -> tuple[str, ...]:
    return tuple(
        part.zfill(20) if part.isdigit() else part for part in re.split(r"(\d+)", source_id)
    )


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _manifest_hash(
    cases: Sequence[CandidateCase],
    documents: Sequence[NormalizedDocument],
    holdouts: Sequence[EvaluatorHoldout],
    *,
    corpus_hash: str,
    benchmark_hash: str,
) -> str:
    payload = {
        "cases": [case.model_dump(mode="json") for case in cases],
        "documents": [document.model_dump(mode="json") for document in documents],
        "holdout_case_ids": [holdout.case_id for holdout in holdouts],
        "corpus_hash": corpus_hash,
        "benchmark_hash": benchmark_hash,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def _hash_models(values: Sequence[StrictModel]) -> str:
    payload = "".join(
        json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
        for value in values
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_artifacts(
    output_dir: Path,
    *,
    candidate_cases: Sequence[CandidateCase],
    candidate_documents: Sequence[NormalizedDocument],
    holdouts: Sequence[EvaluatorHoldout],
    rejections: Sequence[BenchmarkRejection],
    usage: BenchmarkGenerationUsage,
    manifest_hash: str,
    corpus_hash: str,
    benchmark_hash: str,
) -> Path:
    if output_dir.exists() and (
        output_dir.is_symlink() or not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(f"benchmark output is not empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    evaluator = temporary / "evaluator"
    benchmark_dir = temporary / "benchmark"
    corpus_dir = temporary / "corpus"
    try:
        evaluator.mkdir()
        benchmark_dir.mkdir()
        corpus_dir.mkdir()
        _write_jsonl(benchmark_dir / "cases.jsonl", candidate_cases)
        _write_jsonl(corpus_dir / "documents.jsonl", candidate_documents)
        _write_jsonl(evaluator / "holdouts.jsonl", holdouts)
        _write_jsonl(evaluator / "oracles.jsonl", holdouts)
        _write_jsonl(evaluator / "rejections.jsonl", rejections)
        (evaluator / "generation-usage.json").write_text(
            json.dumps(usage.model_dump(mode="json"), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "manifest_hash": manifest_hash,
                    "corpus_hash": corpus_hash,
                    "benchmark_hash": benchmark_hash,
                    "case_count": len(candidate_cases),
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            output_dir.rmdir()
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir / "evaluator"


def _write_jsonl(path: Path, values: Iterable[StrictModel]) -> None:
    content = "".join(
        json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
        for value in values
    )
    path.write_text(content, encoding="utf-8")


def _blocked_result(
    status: BenchmarkStatus,
    rejections: Sequence[BenchmarkRejection],
    usage: BenchmarkGenerationUsage,
    *,
    leakage: LeakageScanResult | None = None,
) -> BenchmarkBuildResult:
    return BenchmarkBuildResult(
        status=status,
        rejections=tuple(rejections),
        generation_usage=usage,
        leakage=leakage or LeakageScanResult(clean=True),
        candidate_start_allowed=False,
        candidate_started=False,
    )
