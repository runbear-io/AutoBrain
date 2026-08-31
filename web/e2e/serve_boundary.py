"""Start the real local experiment job boundary for browser E2E.

This runs the shipped `ExperimentJobBoundary`/`ExperimentJobServer` with an
in-memory fixture runner so the browser exercises the genuine HTTP contract,
CORS policy, and lifecycle transitions without needing provider credentials or
a scored corpus. It binds loopback only and prints its base URL as JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from autobrain.experiment_contracts import ExperimentRequest
from autobrain.experiment_job import (
    ExperimentJobBoundary,
    ExperimentJobServer,
    JobResult,
)
from autobrain.models import CandidateId, CostStatus, Status, Verdict
from autobrain.projection import (
    PROJECTION_SCHEMA_VERSION,
    CandidateProjection,
    RunProjection,
)


class FixtureRun:
    """Deterministic stand-in for the orchestrator, local and credential-free."""

    def __init__(self, request: ExperimentRequest) -> None:
        self.run_id = f"run-{request.experiment_id[:8]}"

    def cancel(self) -> None:
        return None

    def run(self) -> Any:
        return type("Result", (), {"run_id": self.run_id, "status": Status.OK})()


def fixture_projection(run_id: str) -> RunProjection:
    """Build a deterministic but genuinely shaped run projection.

    This uses the shipped `RunProjection`/`CandidateProjection` models, so the
    browser reads exactly the payload the real engine emits - including its
    validation rules - without needing a scored corpus or any credential.
    """
    return RunProjection(
        schema_version=PROJECTION_SCHEMA_VERSION,
        run_id=run_id,
        status=Status.OK,
        verdict=Verdict.GBRAIN,
        rationale=(
            "GBrain retrieved grounded evidence for more cases than the other "
            "candidates on this frozen corpus."
        ),
        corpus_hash="a" * 64,
        benchmark_hash="b" * 64,
        candidates=[
            CandidateProjection(
                candidate=CandidateId.GBRAIN,
                status=Status.OK,
                quality_score=81.5,
                answer_success_rate=0.9,
                source_support_rate=0.82,
                contradiction_count=0,
                scored_cases=20,
                answered_cases=18,
                cost_status=CostStatus.COMPLETE,
                total_cost_usd=1.42,
                query_p50_ms=104.0,
                query_p95_ms=228.0,
                operating_burden=2.0,
            ),
            CandidateProjection(
                candidate=CandidateId.MEM0,
                status=Status.OK,
                quality_score=62.0,
                answer_success_rate=0.65,
                source_support_rate=0.61,
                contradiction_count=1,
                scored_cases=20,
                answered_cases=12,
                cost_status=CostStatus.INCOMPLETE,
                total_cost_usd=None,
                query_p50_ms=180.0,
                query_p95_ms=412.0,
                operating_burden=3.0,
            ),
        ],
        warnings=["cost telemetry was incomplete for mem0"],
    )


def main() -> None:
    jobs = ExperimentJobBoundary(
        factory=lambda request, _sink, _cancellation: FixtureRun(request),
        result_loader=lambda run_id: JobResult.success(run_id, fixture_projection(run_id)),
    )
    with ExperimentJobServer(jobs, port=0) as server:
        print(
            json.dumps({"base_url": f"http://{server.host}:{server.port}"}),
            flush=True,
        )
        try:
            while True:
                sys.stdin.readline()
                if sys.stdin.closed:
                    break
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
