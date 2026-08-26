"""Spawnable local fixture server used by the web integration test.

Started as a subprocess by `web/src/live/httpIntegration.test.ts`, which then
talks to it over real HTTP. It prints the bound base URL on stdout as the
port handoff, so the caller never has to guess a port.

The `succeeded`, `failed` and `cancelled` modes serve genuine
`LocalRunServer` outcomes. The `malformed` mode serves a deliberately corrupt
payload from a plain HTTP server so the browser client's strict validation can
be exercised against something a drifted or damaged runner might emit.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final, cast

from autobrain.local_server import (
    LOOPBACK_HOST,
    PROJECTION_PATH,
    LocalRunServer,
    RunOutcome,
    is_allowed_origin,
)
from autobrain.models import (
    CandidateEvaluation,
    CandidateId,
    ComparisonArtifact,
    CostStatus,
    DecisionResult,
    Status,
    Verdict,
)
from autobrain.projection import project_comparison
from autobrain.report import build_comparison

CORPUS_HASH: Final = "a" * 64
BENCHMARK_HASH: Final = "b" * 64


def sample_artifact() -> ComparisonArtifact:
    evaluation = CandidateEvaluation(
        candidate=CandidateId.GBRAIN,
        status=Status.OK,
        scored_cases=30,
        answered_cases=28,
        quality_score=93.6,
        answer_success_rate=0.93,
        source_support_rate=0.79,
        contradiction_count=1,
        total_input_tokens=1200,
        total_output_tokens=340,
        total_cost_usd=1.25,
        cost_status=CostStatus.COMPLETE,
        query_p50_ms=820.0,
        query_p95_ms=2460.0,
        operating_burden=2.0,
        valid_pin=True,
        corpus_hash=CORPUS_HASH,
    )
    return build_comparison(
        run_id="RUN-A41F",
        status=Status.OK,
        corpus_hash=CORPUS_HASH,
        benchmark_hash=BENCHMARK_HASH,
        coverage=[],
        candidates=[evaluation],
        decision=DecisionResult(
            status=Status.OK,
            verdict=Verdict.GBRAIN,
            rationale="GBrain leads grounded recall.",
            eligible_candidates=[CandidateId.GBRAIN],
            considered_candidates=[CandidateId.GBRAIN],
        ),
        evidence=[],
    )


class _MalformedHandler(BaseHTTPRequestHandler):
    """Serves a structurally invalid projection with correct CORS headers."""

    def do_GET(self) -> None:
        if self.path != PROJECTION_PATH:
            self._respond(404, {"error": "not found"})
            return
        payload = RunOutcome.succeeded(project_comparison(sample_artifact())).to_payload()
        projection = cast(dict[str, object], payload["projection"])
        projection["corpus_hash"] = "not-a-sha256"
        self._respond(200, payload)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _cors(self) -> None:
        self.send_header("Vary", "Origin")
        origin = self.headers.get("Origin")
        if origin is not None and is_allowed_origin(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")

    def _respond(self, code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def announce(base_url: str) -> None:
    sys.stdout.write(f"listening {base_url}\n")
    sys.stdout.flush()


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "succeeded"
    if mode == "malformed":
        server = ThreadingHTTPServer((LOOPBACK_HOST, 0), _MalformedHandler)
        announce(f"http://{LOOPBACK_HOST}:{server.server_port}")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        threading.Event().wait()
        return

    outcomes = {
        "succeeded": lambda: RunOutcome.succeeded(project_comparison(sample_artifact())),
        "failed": lambda: RunOutcome.failed("candidate runtime exited with code 1"),
        "cancelled": RunOutcome.cancelled,
    }
    build = outcomes[mode]
    with LocalRunServer(build, port=0) as server:
        announce(server.base_url)
        server.wait_forever()


if __name__ == "__main__":
    main()
