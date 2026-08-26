"""Local loopback HTTP boundary serving a redacted run projection.

Scope, stated plainly: this is a developer fixture. It binds 127.0.0.1 on an
ephemeral port, serves exactly one read-only JSON endpoint and exits with the
process. It is not a hosted deployment, has no authentication, no TLS and no
multi-user story, and nothing here should be read as a claim that it does.

The boundary reports a run with three explicit terminal statuses -
SUCCEEDED, FAILED and CANCELLED - which are deliberately distinct from the
engine's `Status` enum carried inside the projection. A run can succeed as an
operation while reporting a non-OK engine status such as NO_DECISION.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, cast
from urllib.parse import urlsplit

from autobrain.cancellation import RunCancellation, RunCancelled
from autobrain.projection import RunProjection, project_comparison
from autobrain.report import load_comparison, redact_text

PROJECTION_PATH: Final = "/api/v1/run"
LOOPBACK_HOST: Final = "127.0.0.1"

#: Port the browser client defaults to. Kept in sync with
#: DEFAULT_LOCAL_RUNNER_URL in web/src/live/runClient.ts.
DEFAULT_LOCAL_PORT: Final = 8765

#: Hosts permitted as a CORS origin. A browser page served from a loopback dev
#: server needs to read this fixture, but nothing else ever should, so the
#: allowlist is exactly loopback and the scheme is pinned to http.
_ALLOWED_ORIGIN_HOSTS: Final = frozenset({"localhost", "127.0.0.1", "[::1]", "::1"})


def is_allowed_origin(origin: str) -> bool:
    """Return True for http loopback origins on any port, False otherwise.

    Parsed rather than prefix-matched so lookalikes such as
    ``http://localhost.evil.example`` are rejected.
    """
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    if parsed.scheme != "http" or parsed.hostname is None:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    if parsed.path or parsed.query or parsed.fragment:
        return False
    try:
        # Accessing .port validates the port syntax; the value itself is
        # irrelevant because any loopback port is a legitimate dev origin.
        _ = parsed.port
    except ValueError:
        return False
    return parsed.hostname in _ALLOWED_ORIGIN_HOSTS


class RunOutcomeStatus(StrEnum):
    """Terminal status of a local run attempt, distinct from engine `Status`."""

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class RunOutcome:
    """Result of one local run attempt.

    A projection is present only on success; failure carries a redacted reason
    and cancellation carries neither, so a cancelled run can never be mistaken
    for a completed one.
    """

    status: RunOutcomeStatus
    projection: RunProjection | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if (self.status is RunOutcomeStatus.SUCCEEDED) != (self.projection is not None):
            raise ValueError("a projection is present exactly when the run succeeded")
        if self.error is not None and self.status is not RunOutcomeStatus.FAILED:
            raise ValueError("only a failed run carries an error reason")

    @classmethod
    def succeeded(cls, projection: RunProjection) -> RunOutcome:
        return cls(RunOutcomeStatus.SUCCEEDED, projection=projection)

    @classmethod
    def failed(cls, reason: str) -> RunOutcome:
        return cls(RunOutcomeStatus.FAILED, error=redact_text(reason))

    @classmethod
    def cancelled(cls) -> RunOutcome:
        return cls(RunOutcomeStatus.CANCELLED)

    def to_payload(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "projection": (
                self.projection.model_dump(mode="json") if self.projection is not None else None
            ),
            "error": self.error,
        }


RunBody = Callable[[RunCancellation], RunProjection]


def run_locally(
    body: RunBody,
    *,
    cancellation: RunCancellation | None = None,
) -> RunOutcome:
    """Execute `body`, mapping its result onto the three terminal statuses.

    Cancellation is checked before and after the body so a signal raised by a
    nested boundary and a signal observed cooperatively both land on CANCELLED
    rather than being reported as a failure.
    """
    signal = cancellation if cancellation is not None else RunCancellation()
    try:
        signal.raise_if_cancelled()
        projection = body(signal)
    except RunCancelled:
        return RunOutcome.cancelled()
    except Exception as error:
        if signal.cancelled:
            return RunOutcome.cancelled()
        return RunOutcome.failed(f"{type(error).__name__}: {error}")
    if signal.cancelled:
        return RunOutcome.cancelled()
    return RunOutcome.succeeded(projection)


def outcome_for_run_dir(run_dir: Path) -> RunOutcome:
    """Project the comparison artifact in `run_dir` into a servable outcome.

    A missing or unreadable artifact is reported as FAILED rather than raised,
    so the server can keep answering and the operator sees the reason in the
    browser instead of a dead connection.
    """
    comparison = run_dir / "comparison.json"
    if not comparison.is_file():
        return RunOutcome.failed(f"no comparison.json in {run_dir}")
    try:
        return RunOutcome.succeeded(project_comparison(load_comparison(comparison)))
    except Exception as error:
        return RunOutcome.failed(f"{type(error).__name__}: {error}")


class _ProjectionHandler(BaseHTTPRequestHandler):
    server_version = "AutoBrainLocalFixture/1"

    def do_GET(self) -> None:
        if self.path != PROJECTION_PATH:
            self._send_json(404, {"error": "not found"})
            return
        server = cast(_ProjectionServer, self.server)
        try:
            outcome = server.resolve()
        except Exception as error:
            outcome = RunOutcome.failed(f"{type(error).__name__}: {error}")
        self._send_json(200, outcome.to_payload())

    def do_OPTIONS(self) -> None:
        """Answer the browser preflight for loopback dev origins only."""
        self.send_response(204)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_cors_headers(self) -> None:
        """Echo back only an allowed loopback origin.

        The origin is echoed verbatim rather than answered with `*` so the
        grant stays scoped to the exact caller, and credentials are never
        allowed because this fixture has no authentication to protect.
        """
        # Vary is set unconditionally: the response body is origin-independent
        # but the CORS headers are not, so caches must key on Origin.
        self.send_header("Vary", "Origin")
        origin = self.headers.get("Origin")
        if origin is None or not is_allowed_origin(origin):
            return
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")

    def _send_json(self, code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # Local fixture: refuse content sniffing and embedding.
        self.send_header("X-Content-Type-Options", "nosniff")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class _ProjectionServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], resolve: Callable[[], RunOutcome]) -> None:
        self.resolve = resolve
        super().__init__(address, _ProjectionHandler)


class LocalRunServer:
    """Loopback-only HTTP fixture exposing one run projection endpoint."""

    def __init__(
        self,
        resolve: Callable[[], RunOutcome],
        *,
        port: int = DEFAULT_LOCAL_PORT,
    ) -> None:
        self._resolve = resolve
        self._requested_port = port
        self._server: _ProjectionServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        return LOOPBACK_HOST

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("local run server is not running")
        return self._server.server_port

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> LocalRunServer:
        self._server = _ProjectionServer((LOOPBACK_HOST, self._requested_port), self._resolve)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def wait_forever(self) -> None:
        """Block until the serving thread stops or the operator interrupts."""
        if self._thread is None:
            raise RuntimeError("local run server is not running")
        while self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def __exit__(self, *_args: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
