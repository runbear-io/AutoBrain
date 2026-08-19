from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pytest

from autobrain.decision import select_winner
from autobrain.metering import (
    COST_COMPLETE,
    COST_INCOMPLETE,
    LoopbackMeteringProxy,
    MeteringEvent,
    MeteringRole,
    PriceQuote,
    PriceSheet,
    reconcile_usage,
)
from autobrain.models import CandidateEvaluation, CandidateId, CostStatus, Status


def test_reconciles_proxy_events_and_preserves_price_version() -> None:
    prices = PriceSheet(
        version="openai-2026-08-01",
        effective_date="2026-08-01",
        models={
            "gpt-5-mini": PriceQuote(input_usd_per_million=0.25, output_usd_per_million=2.0),
            "text-embedding-3-small": PriceQuote(
                input_usd_per_million=0.02, output_usd_per_million=0.0
            ),
        },
    )
    result = reconcile_usage(
        [
            MeteringEvent(
                event_id="e-ingest",
                request_id="r-ingest",
                candidate="mem0",
                phase="ingest",
                model="text-embedding-3-small",
                input_tokens=1_000,
                output_tokens=0,
                raw_usage={"prompt_tokens": 1_000},
                tags={"run_id": "run-1"},
            ),
            MeteringEvent(
                event_id="e-query",
                request_id="r-query",
                candidate="mem0",
                phase="query",
                model="gpt-5-mini",
                input_tokens=2_000,
                output_tokens=500,
                raw_usage={"prompt_tokens": 2_000, "completion_tokens": 500},
                tags={"run_id": "run-1"},
            ),
        ],
        prices,
    )
    assert result.cost_status == COST_COMPLETE
    assert result.price_sheet_version == prices.version
    assert result.input_tokens == 3_000
    assert result.output_tokens == 500
    assert result.usd is not None
    assert abs(result.usd - 0.00152) < 1e-8
    assert result.raw_events[0]["request_id"] == "r-ingest"


def test_missing_native_usage_is_incomplete_not_zero_cost() -> None:
    result = reconcile_usage(
        [
            MeteringEvent(
                event_id="proxy-query",
                request_id="r-query",
                candidate="gbrain",
                phase="query",
                model="gpt-5-mini",
                input_tokens=10,
                output_tokens=5,
                raw_usage={"prompt_tokens": 10, "completion_tokens": 5},
                tags={},
            )
        ],
        None,
    )
    assert result.cost_status == COST_INCOMPLETE
    assert result.usd is None
    assert any("native" in warning.lower() for warning in result.warnings)


def test_corrupt_price_sheet_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "prices.json"
    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt price sheet"):
        from autobrain.metering import load_price_sheet

        load_price_sheet(path)


def test_proxy_adds_phase_and_candidate_tags_and_reconciles_native_first() -> None:
    proxy = LoopbackMeteringProxy(
        upstream=lambda payload: {
            "id": "chat-1",
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            "choices": [{"message": {"content": "ok"}}],
            "echo": payload,
        }
    )
    with proxy:
        response = proxy.chat(
            {"model": "gpt-5-mini", "messages": []},
            candidate="llm-wiki",
            phase="query",
        )
    assert response["usage"]["prompt_tokens"] == 7
    assert proxy.events[0].tags == {
        "candidate": "llm-wiki",
        "phase": "query",
        "role": "candidate",
    }
    assert proxy.events[0].raw_usage["completion_tokens"] == 3


def test_full_reconciliation_summary_is_arrival_order_independent_for_all_roles() -> None:
    prices = PriceSheet(
        version="task9-order",
        effective_date="2026-08-01",
        models={"gpt-5-mini": PriceQuote(input_usd_per_million=1, output_usd_per_million=2)},
    )
    events = [
        MeteringEvent(
            event_id="candidate-query",
            request_id="candidate-query",
            candidate="mem0",
            phase="query",
            role=MeteringRole.CANDIDATE,
            model="gpt-5-mini",
            input_tokens=20,
            output_tokens=2,
            raw_usage={"completion_tokens": 2, "prompt_tokens": 20, "detail": "candidate-query"},
            tags={"z": "last", "role": "candidate", "phase": "query", "candidate": "mem0"},
        ),
        MeteringEvent(
            event_id="candidate-ingest",
            request_id="candidate-ingest",
            candidate="mem0",
            phase="ingest",
            role=MeteringRole.CANDIDATE,
            model="gpt-5-mini",
            input_tokens=10,
            output_tokens=1,
            raw_usage={"detail": "candidate-ingest", "prompt_tokens": 10, "completion_tokens": 1},
            tags={"candidate": "mem0", "phase": "ingest", "role": "candidate"},
        ),
        MeteringEvent(
            event_id="generator-query",
            request_id="generator-query",
            candidate="benchmark",
            phase="query",
            role=MeteringRole.GENERATOR,
            model="gpt-5-mini",
            input_tokens=None,
            output_tokens=3,
            raw_usage={"completion_tokens": 3, "detail": "generator-query"},
            tags={"candidate": "benchmark", "phase": "query", "role": "generator"},
        ),
        MeteringEvent(
            event_id="generator-ingest",
            request_id="generator-ingest",
            candidate="benchmark",
            phase="ingest",
            role=MeteringRole.GENERATOR,
            model="gpt-5-mini",
            input_tokens=5,
            output_tokens=1,
            raw_usage={"detail": "generator-ingest", "completion_tokens": 1, "prompt_tokens": 5},
            tags={"role": "generator", "candidate": "benchmark", "phase": "ingest"},
        ),
        MeteringEvent(
            event_id="oracle-query",
            request_id="oracle-query",
            candidate="evaluator",
            phase="query",
            role=MeteringRole.ORACLE,
            model="gpt-5-mini",
            input_tokens=7,
            output_tokens=None,
            raw_usage={"prompt_tokens": 7, "detail": "oracle-query"},
            tags={"phase": "query", "candidate": "evaluator", "role": "oracle"},
        ),
        MeteringEvent(
            event_id="oracle-ingest",
            request_id="oracle-ingest",
            candidate="evaluator",
            phase="ingest",
            role=MeteringRole.ORACLE,
            model="gpt-5-mini",
            input_tokens=3,
            output_tokens=1,
            raw_usage={"completion_tokens": 1, "prompt_tokens": 3, "detail": "oracle-ingest"},
            tags={"candidate": "evaluator", "role": "oracle", "phase": "ingest"},
        ),
        MeteringEvent(
            event_id="harness-query",
            request_id="harness-query",
            candidate="runner",
            phase="query",
            role=MeteringRole.HARNESS,
            model="gpt-5-mini",
            input_tokens=4,
            output_tokens=1,
            raw_usage={"detail": "harness-query", "prompt_tokens": 4, "completion_tokens": 1},
            tags={"role": "harness", "phase": "query", "candidate": "runner"},
        ),
        MeteringEvent(
            event_id="harness-ingest",
            request_id="harness-ingest",
            candidate="runner",
            phase="ingest",
            role=MeteringRole.HARNESS,
            model="gpt-5-mini",
            input_tokens=2,
            output_tokens=1,
            raw_usage={"prompt_tokens": 2, "detail": "harness-ingest", "completion_tokens": 1},
            tags={"candidate": "runner", "phase": "ingest", "role": "harness"},
        ),
    ]

    for role in MeteringRole:
        forward = reconcile_usage(events, prices, role=role)
        reverse = reconcile_usage(list(reversed(events)), prices, role=role)
        assert forward.model_dump_json() == reverse.model_dump_json()

    forward_mixed = reconcile_usage(events, prices)
    reverse_mixed = reconcile_usage(list(reversed(events)), prices)
    assert forward_mixed.model_dump_json() == reverse_mixed.model_dump_json()
    assert [event["event_id"] for event in forward_mixed.raw_events] == [
        "candidate-ingest",
        "candidate-query",
        "generator-ingest",
        "generator-query",
        "harness-ingest",
        "harness-query",
        "oracle-ingest",
        "oracle-query",
    ]
    assert all(
        {"candidate", "phase", "role"} <= event["tags"].keys() for event in forward_mixed.raw_events
    )


def _task9_prices() -> PriceSheet:
    return PriceSheet(
        version="task9-mixed-role",
        effective_date="2026-08-01",
        models={"gpt-5-mini": PriceQuote(input_usd_per_million=1, output_usd_per_million=2)},
    )


def _task9_role_event(
    role: MeteringRole,
    phase: str,
    *,
    missing_output: bool = False,
) -> MeteringEvent:
    candidates = {
        MeteringRole.CANDIDATE: "mem0",
        MeteringRole.GENERATOR: "benchmark",
        MeteringRole.ORACLE: "evaluator",
        MeteringRole.HARNESS: "runner",
    }
    event_id = f"{role.value}-{phase}"
    input_tokens = 10 if phase == "ingest" else 20
    output_tokens = None if missing_output else (1 if phase == "ingest" else 2)
    return MeteringEvent(
        event_id=event_id,
        request_id=event_id,
        candidate=candidates[role],
        phase=phase,
        role=role,
        model="gpt-5-mini",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw_usage={"phase": phase, "role": role.value},
        tags={"candidate": candidates[role], "phase": phase, "role": role.value},
    )


ROLE_COMBINATIONS = [
    combination
    for size in range(2, len(MeteringRole) + 1)
    for combination in combinations(MeteringRole, size)
]


@pytest.mark.parametrize("roles", ROLE_COMBINATIONS)
def test_full_usage_mixed_roles_never_produce_a_combined_cost(
    roles: tuple[MeteringRole, ...],
) -> None:
    events = [_task9_role_event(role, phase) for role in roles for phase in ("ingest", "query")]

    result = reconcile_usage(events, _task9_prices())
    reversed_result = reconcile_usage(list(reversed(events)), _task9_prices())

    assert result.cost_status == COST_INCOMPLETE
    assert result.usd is None
    assert "METERING_ROLE_MIXED: pass role/candidate filters before reconciliation" in (
        result.warnings
    )
    assert result.model_dump_json() == reversed_result.model_dump_json()


@pytest.mark.parametrize("roles", ROLE_COMBINATIONS)
def test_missing_usage_mixed_roles_remain_unusable_and_order_independent(
    roles: tuple[MeteringRole, ...],
) -> None:
    events = [
        _task9_role_event(role, phase, missing_output=phase == "query")
        for role in roles
        for phase in ("ingest", "query")
    ]

    result = reconcile_usage(events, _task9_prices())
    reversed_result = reconcile_usage(list(reversed(events)), _task9_prices())

    assert result.cost_status == COST_INCOMPLETE
    assert result.usd is None
    assert any("COST_INCOMPLETE: missing usage" in warning for warning in result.warnings)
    assert any("METERING_ROLE_MIXED:" in warning for warning in result.warnings)
    assert result.model_dump_json() == reversed_result.model_dump_json()


@pytest.mark.parametrize("role", list(MeteringRole))
def test_explicit_role_filters_reconcile_each_role_completely(
    role: MeteringRole,
) -> None:
    events = [
        _task9_role_event(event_role, phase)
        for event_role in MeteringRole
        for phase in ("ingest", "query")
    ]
    candidate = {
        MeteringRole.CANDIDATE: "mem0",
        MeteringRole.GENERATOR: "benchmark",
        MeteringRole.ORACLE: "evaluator",
        MeteringRole.HARNESS: "runner",
    }[role]

    result = reconcile_usage(
        events,
        _task9_prices(),
        role=role,
        candidate=candidate,
    )
    reversed_result = reconcile_usage(
        list(reversed(events)),
        _task9_prices(),
        role=role,
        candidate=candidate,
    )

    assert result.cost_status == COST_COMPLETE
    assert result.usd == 0.000036
    assert result.input_tokens == 30
    assert result.output_tokens == 3
    assert not any("METERING_ROLE_MIXED:" in warning for warning in result.warnings)
    assert result.model_dump_json() == reversed_result.model_dump_json()


def test_mixed_role_summary_cannot_supply_a_complete_winner_cost() -> None:
    events = [
        _task9_role_event(role, phase)
        for role in (MeteringRole.CANDIDATE, MeteringRole.GENERATOR)
        for phase in ("ingest", "query")
    ]
    mixed = reconcile_usage(events, _task9_prices())

    def candidate(
        name: CandidateId,
        *,
        cost: float | None,
        cost_status: CostStatus,
        latency: float,
    ) -> CandidateEvaluation:
        return CandidateEvaluation(
            candidate=name,
            status=Status.OK,
            scored_cases=20,
            answered_cases=20,
            quality_score=80,
            answer_success_rate=1.0,
            source_support_rate=0.8,
            contradiction_count=0,
            total_input_tokens=100,
            total_output_tokens=100,
            total_cost_usd=cost,
            cost_status=cost_status,
            query_p50_ms=latency / 2,
            query_p95_ms=latency,
            workspace_bytes=100,
            operating_burden=2,
            valid_pin=True,
            corpus_hash="a" * 64,
        )

    result = select_winner(
        [
            candidate(
                CandidateId.LLM_WIKI,
                cost=mixed.usd,
                cost_status=mixed.cost_status,
                latency=200,
            ),
            candidate(
                CandidateId.MEM0,
                cost=999,
                cost_status=COST_COMPLETE,
                latency=100,
            ),
        ]
    )

    assert mixed.cost_status == COST_INCOMPLETE
    assert mixed.usd is None
    assert result.verdict == CandidateId.MEM0
    assert "cost is incomplete" in result.rationale
