from __future__ import annotations

import json
import queue
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from autobrain.auth.models import Provider
from autobrain.experiment import build_automatic_plan
from autobrain.models import CandidateId, Status
from autobrain.orchestration import (
    CandidateOutcome,
    ConnectorSnapshot,
    RunCancellation,
    RunCancelled,
    RunConfig,
    RunOrchestrator,
    RunResult,
    StageEvent,
)
from autobrain.subscription import SubscriptionStatus
from autobrain.tui_runtime import start_plan_worker


def _records(count: int = 24) -> list[dict[str, Any]]:
    return [
        {
            "provider": "slack",
            "source_id": f"slack:stage:{index}",
            "source_kind": "SLACK_MESSAGE",
            "canonical_url": f"https://fixture.example.test/{index}",
            "title": f"Policy {index}",
            "text": f"Use the documented policy for service {index} within five minutes.",
            "question": f"How do we handle service {index} incidents?",
            "evidence_reply": f"The on-call follows policy {index} and records the incident.",
        }
        for index in range(count)
    ]


class _Connector:
    provider = "slack"

    def probe(self) -> dict[str, Any]:
        return {"allowed": ["search", "fetch"]}

    def crawl(self, *, include_dms: bool) -> ConnectorSnapshot:
        del include_dms
        return ConnectorSnapshot(
            provider=self.provider,
            documents=tuple(_records()),
            coverage={"completeness": "SEARCH_DISCOVERED", "discovered": 24},
        )


class _Candidate:
    def __init__(self, candidate_id: str = "llm-wiki", *, interrupt: bool = False) -> None:
        self.candidate_id = candidate_id
        self.interrupt = interrupt
        self.cleaned = False

    def run(self, context: Any) -> CandidateOutcome:
        del context
        if self.interrupt:
            raise KeyboardInterrupt
        return CandidateOutcome(
            candidate=self.candidate_id,
            status=Status.OK,
            answered_cases=24,
            scored_cases=24,
            cost_usd=1,
        )

    def cleanup(self) -> None:
        self.cleaned = True


def _orchestrator(tmp_path: Path, candidate: _Candidate, **kwargs: Any) -> RunOrchestrator:
    return RunOrchestrator(
        config=RunConfig(output=tmp_path / "runs", open_report=False),
        connectors=[_Connector(), _Connector()],
        candidates=[candidate],
        provider_available=True,
        **kwargs,
    )


def test_stage_events_are_exact_post_redaction_persisted_entries(tmp_path: Path) -> None:
    secret = "sk-stage-event-secret-value"
    events: list[StageEvent] = []

    class _SecretConnector(_Connector):
        def crawl(self, *, include_dms: bool) -> ConnectorSnapshot:
            raise RuntimeError(f"failed under /private/{secret}/run")

    def sink(event: StageEvent) -> None:
        manifest_path = tmp_path / "runs" / event.run_id / "manifest.json"
        persisted = json.loads(manifest_path.read_text())["stages"][-1]
        assert event.as_manifest_entry() == persisted
        events.append(event)

    result = RunOrchestrator(
        config=RunConfig(output=tmp_path / "runs", open_report=False),
        connectors=[_SecretConnector(), _Connector()],
        candidates=[_Candidate()],
        provider_available=True,
        provider_key=secret,
        stage_event_sink=sink,
    ).run()

    assert result.status is Status.FAILED
    assert secret not in json.dumps([event.as_manifest_entry() for event in events])
    assert any("[REDACTED]" in event.detail for event in events)

    assert events
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert [event.name for event in events] == [
        stage["name"]
        for stage in json.loads((result.run_dir / "manifest.json").read_text())["stages"]
    ]
    assert all(event.run_id == result.run_id for event in events)
    with pytest.raises(AttributeError):
        events[0].name = "mutated"  # type: ignore[misc]


def test_terminal_cancel_and_cleanup_are_published_once(tmp_path: Path) -> None:
    events: list[StageEvent] = []
    candidate = _Candidate(interrupt=True)
    result = _orchestrator(tmp_path, candidate, stage_event_sink=events.append).run()

    assert result.status is Status.CANCELLED
    assert candidate.cleaned
    assert events[-2].name == "cancelled"
    assert events[-2].status is Status.CANCELLED
    assert events[-1].name == "cleanup"
    assert events[-1].status is Status.OK
    assert len({event.sequence for event in events}) == len(events)
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["status"] == Status.CANCELLED.value
    assert [event.name for event in events] == [stage["name"] for stage in manifest["stages"]]


@pytest.mark.parametrize(
    "failure",
    [
        SystemExit("system-exit-secret-12345678"),
        KeyboardInterrupt("keyboard-secret-12345678"),
    ],
)
def test_sink_baseexception_is_bounded_redacted_and_secondary(
    tmp_path: Path,
    failure: BaseException,
) -> None:
    def sink(event: StageEvent) -> None:
        del event
        raise failure

    result = _orchestrator(tmp_path, _Candidate(), stage_event_sink=sink).run()

    assert result.status is Status.OK
    assert result.event_sink_errors
    assert all(len(diagnostic) <= 500 for diagnostic in result.event_sink_errors)
    assert all("secret-12345678" not in diagnostic for diagnostic in result.event_sink_errors)
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["status"] == Status.OK.value
    assert manifest["stages"][-1]["name"] == "cleanup"


def test_sink_failure_is_secondary_and_primary_failure_remains_truthful(tmp_path: Path) -> None:
    seen: list[str] = []

    def sink(event: StageEvent) -> None:
        seen.append(event.name)
        raise RuntimeError("sink exploded")

    result = _orchestrator(
        tmp_path,
        _Candidate(interrupt=False),
        stage_event_sink=sink,
    ).run()

    assert result.status is Status.OK
    assert seen
    assert result.event_sink_errors
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["status"] == Status.OK.value
    assert any("stage event sink failed" in warning for warning in manifest["warnings"])


def test_run_orchestrator_is_single_use(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, _Candidate())

    first = orchestrator.run()
    with pytest.raises(RuntimeError, match="single-use"):
        orchestrator.run()

    manifests = list((tmp_path / "runs").glob("*/manifest.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["run_id"] == first.run_id


def test_tui_runtime_forwards_events_and_worker_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received = threading.Event()
    events: list[StageEvent] = []
    results: queue.Queue[Any] = queue.Queue(maxsize=1)
    fake_result = RunResult(
        run_id="tui-run",
        run_dir=tmp_path,
        status=Status.OK,
        report_path=None,
        candidate_results=(),
        verdict="NO_DECISION",
    )

    class _FakeOrchestrator:
        def __init__(self, sink: Any) -> None:
            self.sink = sink

        def run(self) -> RunResult:
            self.sink(
                StageEvent(
                    sequence=1,
                    run_id="tui-run",
                    name="preflight",
                    status=Status.OK,
                    detail="persisted",
                    started_at="2026-08-19T00:00:00+00:00",
                )
            )
            return fake_result

    def build(
        config: RunConfig,
        *,
        stage_event_sink: Any,
        cancellation: RunCancellation,
    ) -> _FakeOrchestrator:
        del config, cancellation
        return _FakeOrchestrator(stage_event_sink)

    monkeypatch.setattr("autobrain.tui_runtime.RunOrchestrator.local", build)

    def slack_status(_store: object) -> SimpleNamespace:
        return SimpleNamespace(ready=False)

    monkeypatch.setattr("autobrain.tui_runtime.SlackSourceStore.status", slack_status)
    plan = build_automatic_plan(
        sources=(Provider.SLACK, Provider.NOTION),
        candidates=tuple(CandidateId),
        subscription_status=SubscriptionStatus.READY,
    )

    def sink(event: StageEvent) -> None:
        events.append(event)
        received.set()

    worker = start_plan_worker(plan, results, sink)
    assert received.wait(timeout=2)
    assert worker.join(timeout=2)

    assert not worker.thread.is_alive()
    assert results.get_nowait() is fake_result
    assert [event.name for event in events] == ["preflight"]


def test_blocked_tui_worker_cancels_persists_cleanup_queues_result_and_joins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    events: list[StageEvent] = []
    results: queue.Queue[RunResult | BaseException] = queue.Queue(maxsize=1)

    class _BlockedCandidate(_Candidate):
        def run(self, context: Any) -> CandidateOutcome:
            started.set()
            assert context.cancellation.wait(timeout=2)
            raise RunCancelled("operator cancelled run")

    candidate = _BlockedCandidate()

    def build(
        config: RunConfig,
        *,
        stage_event_sink: Any,
        cancellation: RunCancellation,
    ) -> RunOrchestrator:
        return RunOrchestrator(
            config=config,
            connectors=[_Connector(), _Connector()],
            candidates=[candidate],
            provider_available=True,
            stage_event_sink=stage_event_sink,
            cancellation=cancellation,
        )

    def slack_status(_store: object) -> SimpleNamespace:
        return SimpleNamespace(ready=False)

    monkeypatch.setattr("autobrain.tui_runtime.RunOrchestrator.local", build)
    monkeypatch.setattr("autobrain.tui_runtime.SlackSourceStore.status", slack_status)
    plan = build_automatic_plan(
        sources=(Provider.SLACK, Provider.NOTION),
        candidates=tuple(CandidateId),
        subscription_status=SubscriptionStatus.READY,
    )

    worker = start_plan_worker(plan, results, events.append)
    assert not worker.thread.daemon
    assert started.wait(timeout=2)
    assert worker.cancel_and_join(timeout=2)

    assert not worker.thread.is_alive()
    queued = results.get_nowait()
    assert isinstance(queued, RunResult)
    assert queued.status is Status.CANCELLED
    assert candidate.cleaned
    manifest = json.loads((queued.run_dir / "manifest.json").read_text())
    assert manifest["status"] == Status.CANCELLED.value
    assert [stage["name"] for stage in manifest["stages"]][-2:] == ["cancelled", "cleanup"]
    assert [event.name for event in events][-2:] == ["cancelled", "cleanup"]


def test_hung_connector_cancellation_settles_with_cleanup_result_and_no_thread_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    results: queue.Queue[RunResult | BaseException] = queue.Queue(maxsize=1)
    before = {thread.ident for thread in threading.enumerate()}

    class _HungConnector:
        provider = "slack"

        def probe(self, *, cancellation: RunCancellation) -> dict[str, Any]:
            entered.set()
            cancellation.wait()
            cancellation.raise_if_cancelled()
            raise AssertionError("cancelled connector continued")

        def crawl(self, *, include_dms: bool) -> ConnectorSnapshot:
            del include_dms
            raise AssertionError("cancelled connector crawled")

    candidate = _Candidate()

    def build(
        config: RunConfig,
        *,
        stage_event_sink: Any,
        cancellation: RunCancellation,
    ) -> RunOrchestrator:
        return RunOrchestrator(
            config=replace(config, output=tmp_path / "runs"),
            connectors=[cast(Any, _HungConnector()), _Connector()],
            candidates=[candidate],
            provider_available=True,
            stage_event_sink=stage_event_sink,
            cancellation=cancellation,
        )

    monkeypatch.setattr("autobrain.tui_runtime.RunOrchestrator.local", build)

    def slack_status(_store: object) -> SimpleNamespace:
        return SimpleNamespace(ready=False)

    monkeypatch.setattr("autobrain.tui_runtime.SlackSourceStore.status", slack_status)
    plan = build_automatic_plan(
        sources=(Provider.SLACK, Provider.NOTION),
        candidates=tuple(CandidateId),
        subscription_status=SubscriptionStatus.READY,
    )

    worker = start_plan_worker(plan, results)
    assert entered.wait(timeout=1)
    worker.cancellation.cancel()
    worker.cancellation.cancel()
    assert worker.cancel_and_join(timeout=2)

    queued = results.get_nowait()
    assert isinstance(queued, RunResult)
    assert queued.status is Status.CANCELLED
    assert candidate.cleaned
    manifest = json.loads((queued.run_dir / "manifest.json").read_text())
    assert manifest["status"] == Status.CANCELLED.value
    assert [stage["name"] for stage in manifest["stages"]][-2:] == ["cancelled", "cleanup"]
    assert {thread.ident for thread in threading.enumerate()} == before


def test_sink_failure_does_not_replace_primary_runtime_failure(tmp_path: Path) -> None:
    class _FailingConnector(_Connector):
        def crawl(self, *, include_dms: bool) -> ConnectorSnapshot:
            del include_dms
            raise RuntimeError("primary crawl failure")

    def sink(event: StageEvent) -> None:
        raise RuntimeError("secondary sink failure")

    result = RunOrchestrator(
        config=RunConfig(output=tmp_path / "runs", open_report=False),
        connectors=[_FailingConnector(), _Connector()],
        candidates=[_Candidate()],
        provider_available=True,
        stage_event_sink=sink,
    ).run()

    assert result.status is Status.FAILED
    assert result.event_sink_errors
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["status"] == Status.FAILED.value
    assert manifest["stages"][-2]["name"] == "failed"
    assert manifest["stages"][-2]["detail"] == "primary crawl failure"
    assert manifest["stages"][-1]["name"] == "cleanup"
    assert all("secondary sink failure" not in stage["detail"] for stage in manifest["stages"])
    assert all("secondary sink failure" not in error for error in result.event_sink_errors)
