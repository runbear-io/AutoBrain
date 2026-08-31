from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from typing import Any

import pytest

from autobrain.experiment_contracts import (
    ExperimentRequest,
    StableExperimentError,
    StableExperimentErrorCode,
)
from autobrain.experiment_job import ExperimentJobBoundary, ExperimentJobServer, JobResult
from autobrain.models import Status
from tests.test_experiment_job_boundary import request_payload


class BlockingRun:
    def __init__(self, cancellation: Any) -> None:
        self.cancellation = cancellation
        self.started = threading.Event()
        self.cancel_called = threading.Event()
        self.release = threading.Event()
        self.run_id = "run-hardening"

    def cancel(self) -> None:
        self.cancel_called.set()
        self.cancellation.cancel()
        self.release.set()

    def run(self) -> Any:
        self.started.set()
        self.release.wait(timeout=2)
        self.cancellation.raise_if_cancelled()
        return type("Result", (), {"run_id": self.run_id, "status": Status.OK})()


def test_validate_and_start_are_idempotent_and_duplicate_create_replays_same_request() -> None:
    made: list[BlockingRun] = []

    def factory(_request: ExperimentRequest, _sink: Any, cancellation: Any) -> BlockingRun:
        run = BlockingRun(cancellation)
        made.append(run)
        return run

    jobs = ExperimentJobBoundary(
        factory=factory, result_loader=lambda run_id: JobResult.success(run_id)
    )
    request = ExperimentRequest.model_validate(request_payload())
    first = jobs.create(request)
    assert jobs.create(request) == first
    assert jobs.validate("exp-1").status.value == "READY"
    assert jobs.validate("exp-1").status.value == "READY"
    assert jobs.start("exp-1").status.value == "RUNNING"
    assert jobs.start("exp-1").status.value == "RUNNING"
    assert len(made) == 1
    jobs.cancel("exp-1")
    assert made[0].cancel_called.wait(timeout=1)
    assert jobs.cancel("exp-1").status.value == "CANCELLED"
    assert jobs.wait("exp-1", timeout=1)


def test_duplicate_create_with_changed_request_is_a_stable_conflict() -> None:
    jobs = ExperimentJobBoundary(
        factory=lambda _request, _sink, cancellation: BlockingRun(cancellation)
    )
    jobs.create(ExperimentRequest.model_validate(request_payload()))
    changed = request_payload()
    changed["candidates"] = ["gbrain"]
    with pytest.raises(StableExperimentError) as error:
        jobs.create(ExperimentRequest.model_validate(changed))
    assert error.value.code is StableExperimentErrorCode.INVALID_REQUEST
    assert error.value.detail == "experiment_id already exists with a different request"


def test_local_job_limit_is_bounded() -> None:
    jobs = ExperimentJobBoundary(
        factory=lambda _request, _sink, cancellation: BlockingRun(cancellation), max_jobs=1
    )
    jobs.create(ExperimentRequest.model_validate(request_payload()))
    with pytest.raises(StableExperimentError) as error:
        jobs.create(ExperimentRequest.model_validate(request_payload(experiment_id="exp-2")))
    assert error.value.code is StableExperimentErrorCode.INVALID_REQUEST
    assert error.value.detail == "local experiment job limit reached"


def test_http_errors_are_stable_json_for_malformed_and_unknown_requests() -> None:
    jobs = ExperimentJobBoundary(
        factory=lambda _request, _sink, cancellation: BlockingRun(cancellation)
    )
    with ExperimentJobServer(jobs, port=0) as server:
        connection = HTTPConnection(server.host, server.port, timeout=2)
        connection.request(
            "POST",
            "/api/v1/experiments",
            b"not-json",
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 400
        assert payload == {"error": "INVALID_REQUEST", "detail": "request failed validation"}
        assert response.getheader("Content-Type") == "application/json"

        connection.request("POST", "/api/v1/experiments/missing/start")
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 404
        assert payload == {"error": "NOT_FOUND", "detail": "missing"}
        connection.close()
