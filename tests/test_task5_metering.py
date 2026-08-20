from __future__ import annotations

from pathlib import Path

import pytest

from autobrain.metering import (
    COST_COMPLETE,
    COST_INCOMPLETE,
    COST_UNAVAILABLE,
    LoopbackMeteringProxy,
    MeteringEvent,
    PriceQuote,
    PriceSheet,
    reconcile_usage,
)
from autobrain.models import (
    CandidateId,
    LatencySpan,
    LatencySpanKind,
    Status,
    UsageSource,
)
from autobrain.orchestration import (
    CandidateContext,
    CandidateOutcome,
    RunConfig,
    RunOrchestrator,
)
from autobrain.subscription_domain import (
    AnswerUsage,
    AuthKind,
    ProviderAnswer,
    ProviderId,
    ProviderIdentity,
    UsageKind,
)
from autobrain.subscription_upstream import build_subscription_upstream


def _prices() -> PriceSheet:
    return PriceSheet(
        version="qualified-2026-08-01",
        effective_date="2026-08-01",
        models={
            "openai:gpt-5-mini": PriceQuote(
                input_usd_per_million=1,
                output_usd_per_million=2,
            )
        },
    )


def _event(
    *,
    provider: str = "openai",
    usage_source: UsageSource = UsageSource.MEASURED,
    input_tokens: int | None = 10,
    output_tokens: int | None = 5,
) -> MeteringEvent:
    return MeteringEvent(
        event_id="event-query",
        request_id="request-query",
        candidate="mem0",
        phase="query",
        provider=provider,
        model="gpt-5-mini",
        usage_source=usage_source,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw_usage={},
    )


def test_measured_native_usage_reconstructs_exact_provider_qualified_cost() -> None:
    result = reconcile_usage([_event()], _prices(), required_phases=("query",))

    assert result.cost_status is COST_COMPLETE
    assert result.usage_source is UsageSource.MEASURED
    assert result.usd == 0.00002


def test_estimated_subscription_word_counts_never_become_cost_complete() -> None:
    result = reconcile_usage(
        [_event(provider="codex", usage_source=UsageSource.ESTIMATED)],
        _prices(),
        required_phases=("query",),
    )

    assert result.cost_status is COST_INCOMPLETE
    assert result.usage_source is UsageSource.ESTIMATED
    assert result.usd is None
    assert "estimated usage" in " ".join(result.warnings)


def test_missing_usage_remains_unavailable_and_never_fabricates_zero_cost() -> None:
    empty = reconcile_usage([], _prices())
    missing = reconcile_usage(
        [_event(input_tokens=None, output_tokens=None)],
        _prices(),
        required_phases=("query",),
    )

    assert empty.cost_status is COST_UNAVAILABLE
    assert empty.usage_source is UsageSource.UNAVAILABLE
    assert empty.usd is None
    assert missing.cost_status is COST_UNAVAILABLE
    assert missing.usage_source is UsageSource.UNAVAILABLE
    assert missing.usd is None


@pytest.mark.parametrize(
    ("usage_sources", "expected"),
    [
        ((UsageSource.MEASURED,), UsageSource.MEASURED),
        ((UsageSource.ESTIMATED,), UsageSource.ESTIMATED),
        ((UsageSource.UNAVAILABLE,), UsageSource.UNAVAILABLE),
        (
            (UsageSource.MEASURED, UsageSource.ESTIMATED),
            UsageSource.ESTIMATED,
        ),
        (
            (UsageSource.MEASURED, UsageSource.UNAVAILABLE),
            UsageSource.UNAVAILABLE,
        ),
        (
            (UsageSource.ESTIMATED, UsageSource.UNAVAILABLE),
            UsageSource.UNAVAILABLE,
        ),
        (
            (
                UsageSource.MEASURED,
                UsageSource.ESTIMATED,
                UsageSource.UNAVAILABLE,
            ),
            UsageSource.UNAVAILABLE,
        ),
    ],
)
def test_usage_source_aggregation_is_explicit_and_conservative(
    usage_sources: tuple[UsageSource, ...],
    expected: UsageSource,
) -> None:
    events = [
        _event(usage_source=usage_source).model_copy(
            update={"event_id": f"event-{index}", "request_id": f"request-{index}"}
        )
        for index, usage_source in enumerate(usage_sources)
    ]

    result = reconcile_usage(events, _prices(), required_phases=("query",))

    assert result.usage_source is expected
    assert result.cost_status is (
        COST_COMPLETE if usage_sources == (UsageSource.MEASURED,) else COST_INCOMPLETE
    )
    assert result.usd == (0.00002 if usage_sources == (UsageSource.MEASURED,) else None)


def test_provider_qualified_model_cannot_use_openai_alias_price() -> None:
    result = reconcile_usage(
        [_event(provider="codex")],
        _prices(),
        required_phases=("query",),
    )

    assert result.cost_status is COST_INCOMPLETE
    assert result.usd is None
    assert "no price for codex:gpt-5-mini" in " ".join(result.warnings)


def test_unavailable_latency_is_none_not_zero() -> None:
    span = LatencySpan(name=LatencySpanKind.PROVIDER_EXECUTION, duration_ms=None)

    assert span.duration_ms is None
    with pytest.raises(ValueError, match="unavailable, not measured"):
        LatencySpan(name=LatencySpanKind.PROVIDER_EXECUTION, duration_ms=0)


def test_subscription_native_usage_and_actual_model_cross_proxy_exactly() -> None:
    class NativeClient:
        def answer(self, prompt: str) -> ProviderAnswer:
            return ProviderAnswer(
                text="answer",
                usage=AnswerUsage(kind=UsageKind.NATIVE, input_tokens=10, output_tokens=5),
                identity=ProviderIdentity(
                    provider=ProviderId.CODEX,
                    model="gpt-5-mini",
                    cli_version="codex 1",
                    auth_kind=AuthKind.CONSUMER_SUBSCRIPTION,
                ),
                execution_ms=7,
            )

        def ask(self, prompt: str) -> str:
            raise AssertionError("typed answer boundary must be used")

    proxy = LoopbackMeteringProxy(build_subscription_upstream(NativeClient()))
    with proxy:
        proxy.chat(
            {"model": "openai-alias", "messages": [{"role": "user", "content": "question"}]},
            candidate="mem0",
            phase="query",
        )

    event = proxy.events[0]
    assert event.provider == "codex"
    assert event.model == "gpt-5-mini"
    assert event.usage_source is UsageSource.MEASURED
    assert event.provider_execution_ms == 7


def test_api_provider_execution_uses_injected_monotonic_clock() -> None:
    ticks = iter((2.0, 2.025))
    proxy = LoopbackMeteringProxy(
        lambda _payload: {
            "id": "api-response",
            "model": "gpt-5-mini",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "choices": [],
        },
        monotonic_clock=lambda: next(ticks),
    )

    with proxy:
        proxy.chat(
            {"model": "gpt-5-mini", "messages": []},
            candidate="mem0",
            phase="query",
        )

    assert proxy.events[0].provider_execution_ms == 25


def test_candidate_end_to_end_span_uses_only_injected_monotonic_clock(tmp_path: Path) -> None:
    ticks = iter((10.0, 10.125))

    class Candidate:
        candidate_id = CandidateId.MEM0.value

        def run(self, context: CandidateContext) -> CandidateOutcome:
            return CandidateOutcome(candidate=self.candidate_id, status=Status.OK)

        def cleanup(self) -> None:
            return None

    class TestOrchestrator(RunOrchestrator):
        def run_candidates(
            self,
            run_dir: Path,
            context: CandidateContext,
        ) -> list[CandidateOutcome]:
            return self._run_candidates(run_dir, context)

    orchestrator = TestOrchestrator(
        config=RunConfig(output=tmp_path),
        connectors=(),
        candidates=(Candidate(),),
        provider_available=True,
        monotonic_clock=lambda: next(ticks),
    )
    context = CandidateContext(documents=(), questions=(), case_ids=())

    outcomes = orchestrator.run_candidates(tmp_path, context)

    assert outcomes[0].latency_spans == (
        LatencySpan(
            name=LatencySpanKind.END_TO_END,
            duration_ms=125,
            candidate=CandidateId.MEM0,
        ),
    )


def test_backward_clock_is_unavailable_instead_of_negative_or_zero(tmp_path: Path) -> None:
    ticks = iter((10.0, 9.0))

    class Candidate:
        candidate_id = CandidateId.MEM0.value

        def run(self, context: CandidateContext) -> CandidateOutcome:
            return CandidateOutcome(candidate=self.candidate_id, status=Status.OK)

        def cleanup(self) -> None:
            return None

    class TestOrchestrator(RunOrchestrator):
        def run_candidates(
            self,
            run_dir: Path,
            context: CandidateContext,
        ) -> list[CandidateOutcome]:
            return self._run_candidates(run_dir, context)

    orchestrator = TestOrchestrator(
        config=RunConfig(output=tmp_path),
        connectors=(),
        candidates=(Candidate(),),
        provider_available=True,
        monotonic_clock=lambda: next(ticks),
    )

    outcome = orchestrator.run_candidates(
        tmp_path,
        CandidateContext(documents=(), questions=(), case_ids=()),
    )[0]

    assert outcome.latency_spans == ()
    assert outcome.latency_ms == 0
