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
from autobrain.models import Status


class FixtureRun:
    """Deterministic stand-in for the orchestrator, local and credential-free."""

    def __init__(self, request: ExperimentRequest) -> None:
        self.run_id = f"run-{request.experiment_id[:8]}"

    def cancel(self) -> None:
        return None

    def run(self) -> Any:
        return type("Result", (), {"run_id": self.run_id, "status": Status.OK})()


def main() -> None:
    jobs = ExperimentJobBoundary(
        factory=lambda request, _sink, _cancellation: FixtureRun(request),
        result_loader=lambda run_id: JobResult.success(run_id),
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
