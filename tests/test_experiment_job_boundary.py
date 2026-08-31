from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

import pytest

from autobrain.experiment_contracts import ExperimentRequest
from autobrain.experiment_job import (
    ExperimentJobBoundary,
    ExperimentJobServer,
    JobResult,
)
from autobrain.models import CandidateId, Status

HASH = "a" * 64


def request_payload(*, experiment_id: str = "exp-1") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "identity": {
            "corpus": {"sha256": HASH, "document_count": 2},
            "benchmark_sha256": "b" * 64,
            "protocol": "retrieval-v1",
            "evaluator": "retrieval",
        },
        "candidates": [CandidateId.LLM_WIKI, CandidateId.MEM0],
        "evaluation_mode": "retrieval_only",
    }


class FakeRun:
    def __init__(self, root: Path, *, gate: threading.Event | None = None) -> None:
        self.root = root
        self.cancellation = None
        self.gate = gate
        self.run_id = "run-1"

    def cancel(self) -> None:
        assert self.cancellation is not None
        self.cancellation.cancel()

    def run(self) -> Any:
        if self.gate is not None:
            self.gate.wait(timeout=2)
            assert self.cancellation is not None
            self.cancellation.raise_if_cancelled()
        run_dir = self.root / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "comparison.json").write_text("{}", encoding="utf-8")
        return type(
            "Result", (), {"run_id": self.run_id, "run_dir": run_dir, "status": Status.OK}
        )()


def test_create_validate_start_status_cancel_and_result_use_typed_boundary(tmp_path: Path) -> None:
    gate = threading.Event()
    made: list[FakeRun] = []

    def factory(request: ExperimentRequest, sink: Any, cancellation: Any) -> FakeRun:
        del request, sink
        run = FakeRun(tmp_path / "runs", gate=gate)
        run.cancellation = cancellation
        made.append(run)
        return run

    jobs = ExperimentJobBoundary(
        factory=factory, result_loader=lambda _: JobResult.success("run-1")
    )
    created = jobs.create(ExperimentRequest.model_validate(request_payload()))
    assert created.status.value == "CREATED"
    assert jobs.validate(created.experiment_id).status.value == "READY"
    jobs.start(created.experiment_id)
    assert jobs.status(created.experiment_id).status.value == "RUNNING"
    assert jobs.progress(created.experiment_id).events == ()
    jobs.cancel(created.experiment_id)
    gate.set()
    jobs.wait(created.experiment_id, timeout=2)
    assert jobs.status(created.experiment_id).status.value == "CANCELLED"
    assert jobs.result(created.experiment_id).status == "CANCELLED"
    assert made


def test_http_contract_is_strict_and_does_not_expose_request_secrets(tmp_path: Path) -> None:
    jobs = ExperimentJobBoundary(
        factory=lambda _request, _sink, _cancel: FakeRun(tmp_path / "runs"),
        result_loader=lambda _: JobResult.failed("safe failure"),
    )
    with ExperimentJobServer(jobs, port=0) as server:
        connection = HTTPConnection(server.host, server.port, timeout=2)
        body = json.dumps(request_payload()).encode()
        connection.request(
            "POST", "/api/v1/experiments", body, {"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        assert response.status == 201
        created = json.loads(response.read())
        assert created["status"] == "CREATED"
        experiment_id = created["experiment_id"]

        connection.request("POST", f"/api/v1/experiments/{experiment_id}/validate")
        assert connection.getresponse().status == 200
        connection.request("GET", f"/api/v1/experiments/{experiment_id}/status")
        status = json.loads(connection.getresponse().read())
        assert status["status"] == "READY"

        connection.request("POST", f"/api/v1/experiments/{experiment_id}/start")
        assert connection.getresponse().status == 202
        connection.request("GET", f"/api/v1/experiments/{experiment_id}/progress")
        progress = json.loads(connection.getresponse().read())
        assert progress["experiment_id"] == experiment_id
        assert "identity" not in json.dumps(progress)
        assert connection.request("GET", f"/api/v1/experiments/{experiment_id}/result") is None
        assert connection.getresponse().status in {200, 409}

        connection.request(
            "POST",
            "/api/v1/experiments/nope/compare",
            json.dumps({}).encode(),
            {"Content-Type": "application/json"},
        )
        assert connection.getresponse().status == 404
        connection.close()


def test_rerun_and_compare_are_explicit_operations(tmp_path: Path) -> None:
    jobs = ExperimentJobBoundary(
        factory=lambda _request, _sink, _cancel: FakeRun(tmp_path / "runs"),
        result_loader=lambda run_id: JobResult.success(run_id),
        comparator=lambda left, right: {"equivalent": left == right, "left": left, "right": right},
    )
    first = jobs.create(ExperimentRequest.model_validate(request_payload())).experiment_id
    rerun = jobs.rerun(first)
    assert rerun.experiment_id != first
    assert jobs.compare(first, rerun.experiment_id)["equivalent"] is False


def test_unknown_job_is_stable_error() -> None:
    jobs = ExperimentJobBoundary(factory=lambda _request, _sink, _cancel: FakeRun(Path("/tmp")))
    with pytest.raises(Exception, match="NOT_FOUND"):
        jobs.status("missing")


def test_browser_origin_can_reach_the_job_boundary(tmp_path: Path) -> None:
    """The Web wizard runs in a browser, so loopback origins need a CORS grant.

    Only loopback origins are granted, and a lookalike host is refused, so the
    boundary stays reachable from a local dev server and from nowhere else.
    """
    jobs = ExperimentJobBoundary(
        factory=lambda _request, _sink, _cancel: FakeRun(tmp_path / "runs"),
        result_loader=lambda run_id: JobResult.success(run_id),
    )
    with ExperimentJobServer(jobs, port=0) as server:
        connection = HTTPConnection(server.host, server.port, timeout=2)

        connection.request(
            "OPTIONS",
            "/api/v1/experiments",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        preflight = connection.getresponse()
        preflight.read()
        assert preflight.status == 204
        assert preflight.getheader("Access-Control-Allow-Origin") == "http://localhost:5173"
        assert "POST" in (preflight.getheader("Access-Control-Allow-Methods") or "")
        assert "Content-Type" in (preflight.getheader("Access-Control-Allow-Headers") or "")

        connection.request(
            "POST",
            "/api/v1/experiments",
            json.dumps(request_payload(experiment_id="exp-cors")).encode(),
            {"Content-Type": "application/json", "Origin": "http://127.0.0.1:5173"},
        )
        created = connection.getresponse()
        created.read()
        assert created.status == 201
        assert created.getheader("Access-Control-Allow-Origin") == "http://127.0.0.1:5173"
        assert created.getheader("Vary") == "Origin"

        connection.request(
            "GET",
            "/api/v1/experiments/exp-cors/status",
            headers={"Origin": "http://localhost.evil.example"},
        )
        spoofed = connection.getresponse()
        spoofed.read()
        assert spoofed.status == 200
        assert spoofed.getheader("Access-Control-Allow-Origin") is None
        connection.close()
