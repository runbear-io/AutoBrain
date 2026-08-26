"""Local loopback HTTP boundary for the run projection.

This server is a developer fixture. It binds 127.0.0.1 on an ephemeral port and
serves the versioned redacted projection. It is not a hosted deployment and
makes no claim of being one, so the tests below pin the local-only contract as
tightly as the payload contract.
"""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from typing import Any
from urllib.request import urlopen

import pytest

from autobrain.cancellation import RunCancellation, RunCancelled
from autobrain.local_server import (
    LocalRunServer,
    RunOutcome,
    RunOutcomeStatus,
    run_locally,
)
from autobrain.models import Status, Verdict
from tests.test_projection import artifact, project_comparison


def get_json(url: str) -> tuple[int, dict[str, Any]]:
    with urlopen(url) as response:
        return response.status, json.loads(response.read())


def test_server_binds_loopback_only_on_an_ephemeral_port() -> None:
    with LocalRunServer(lambda: RunOutcome.succeeded(project_comparison(artifact()))) as server:
        assert server.host == "127.0.0.1"
        assert server.base_url.startswith("http://127.0.0.1:")
        assert server.port > 0


def test_projection_endpoint_serves_the_versioned_payload() -> None:
    projection = project_comparison(artifact())

    with LocalRunServer(lambda: RunOutcome.succeeded(projection)) as server:
        status, payload = get_json(f"{server.base_url}/api/v1/run")

    assert status == 200
    assert payload["status"] == RunOutcomeStatus.SUCCEEDED.value
    assert payload["projection"]["schema_version"] == projection.schema_version
    assert payload["projection"]["run_id"] == "RUN-A41F"


def test_successful_outcome_reports_succeeded_with_a_projection() -> None:
    outcome = run_locally(lambda _: project_comparison(artifact()))

    assert outcome.status is RunOutcomeStatus.SUCCEEDED
    assert outcome.projection is not None
    assert outcome.projection.verdict is Verdict.GBRAIN
    assert outcome.error is None


def test_failed_outcome_reports_failed_with_a_redacted_reason() -> None:
    def explode(_: RunCancellation) -> Any:
        raise RuntimeError("upstream exploded using sk-abcdef0123456789")

    outcome = run_locally(explode)

    assert outcome.status is RunOutcomeStatus.FAILED
    assert outcome.projection is None
    assert outcome.error is not None
    assert "sk-abcdef0123456789" not in outcome.error
    assert "[REDACTED]" in outcome.error


def test_cancelled_outcome_is_distinct_from_failure() -> None:
    cancellation = RunCancellation()

    def cancel_immediately(signal: RunCancellation) -> Any:
        signal.raise_if_cancelled()
        raise AssertionError("run body must not proceed after cancellation")

    cancellation.cancel()
    outcome = run_locally(cancel_immediately, cancellation=cancellation)

    assert outcome.status is RunOutcomeStatus.CANCELLED
    assert outcome.projection is None


def test_cancellation_observed_mid_run_is_reported_as_cancelled() -> None:
    cancellation = RunCancellation()
    entered = threading.Event()

    def slow_body(signal: RunCancellation) -> Any:
        entered.set()
        signal.wait(timeout=5)
        signal.raise_if_cancelled()
        raise AssertionError("cancelled run must not produce a projection")

    result: dict[str, RunOutcome] = {}

    def drive() -> None:
        result["outcome"] = run_locally(slow_body, cancellation=cancellation)

    worker = threading.Thread(target=drive)
    worker.start()
    assert entered.wait(timeout=5)
    cancellation.cancel()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert result["outcome"].status is RunOutcomeStatus.CANCELLED


def test_server_reports_each_terminal_status_over_http() -> None:
    cases = [
        (RunOutcome.succeeded(project_comparison(artifact())), RunOutcomeStatus.SUCCEEDED),
        (RunOutcome.failed("boundary refused the request"), RunOutcomeStatus.FAILED),
        (RunOutcome.cancelled(), RunOutcomeStatus.CANCELLED),
    ]

    for outcome, expected in cases:
        with LocalRunServer(lambda outcome=outcome: outcome) as server:
            _, payload = get_json(f"{server.base_url}/api/v1/run")
        assert payload["status"] == expected.value


def test_cancelled_and_failed_payloads_never_carry_a_projection() -> None:
    for outcome in (RunOutcome.failed("nope"), RunOutcome.cancelled()):
        with LocalRunServer(lambda outcome=outcome: outcome) as server:
            _, payload = get_json(f"{server.base_url}/api/v1/run")
        assert payload["projection"] is None


@pytest.mark.parametrize(
    "path",
    ["/", "/index.html", "/../etc/passwd", "/api/v1/run/../../etc/passwd", "/api/v2/run"],
)
def test_unknown_paths_are_rejected_rather_than_serving_files(path: str) -> None:
    """Routing is exact-match, so no request path is ever mapped to the filesystem.

    The request is issued with a raw connection because urllib normalizes
    traversal segments client-side, which would never exercise the server.
    """
    with LocalRunServer(lambda: RunOutcome.cancelled()) as server:
        connection = HTTPConnection(server.host, server.port, timeout=5)
        try:
            connection.putrequest("GET", path)
            connection.endheaders()
            response = connection.getresponse()
            status, body = response.status, response.read()
        finally:
            connection.close()

    assert status == 404
    assert b"root:" not in body


def test_run_status_is_not_confused_with_engine_status() -> None:
    """A run can complete successfully while reporting a non-OK engine status."""
    projection = project_comparison(
        artifact(status=Status.NO_DECISION, verdict=Verdict.NO_DECISION, rationale="tie")
    )

    outcome = RunOutcome.succeeded(projection)

    assert outcome.status is RunOutcomeStatus.SUCCEEDED
    assert outcome.projection is not None
    assert outcome.projection.status is Status.NO_DECISION


def test_run_body_receives_the_cancellation_signal() -> None:
    seen: list[RunCancellation] = []
    cancellation = RunCancellation()

    def body(signal: RunCancellation) -> Any:
        seen.append(signal)
        return project_comparison(artifact())

    run_locally(body, cancellation=cancellation)

    assert seen == [cancellation]


def test_run_cancelled_exception_from_a_nested_boundary_is_cancelled() -> None:
    def body(_: RunCancellation) -> Any:
        raise RunCancelled("operator cancelled run")

    assert run_locally(body).status is RunOutcomeStatus.CANCELLED
