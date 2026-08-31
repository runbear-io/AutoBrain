"""Local-only typed experiment job boundary over canonical orchestration.

This module is an in-process adapter, not a hosted service. It accepts only
secret-free experiment contracts and delegates execution, cancellation,
projection, and comparison to the existing AutoBrain domain boundaries.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import unquote, urlsplit

from pydantic import Field, ValidationError

from autobrain.cancellation import RunCancellation, RunCancelled
from autobrain.experiment_contracts import (
    ExperimentLifecycle,
    ExperimentLifecycleStatus,
    ExperimentRequest,
    StableExperimentError,
    StableExperimentErrorCode,
)
from autobrain.local_server import is_allowed_origin
from autobrain.models import StrictModel
from autobrain.projection import RunProjection, project_comparison
from autobrain.report import load_comparison, redact_text
from autobrain.runs import compare_runs


class JobProgress(StrictModel):
    experiment_id: str = Field(min_length=1)
    status: ExperimentLifecycleStatus
    events: tuple[dict[str, Any], ...] = ()


class JobResult(StrictModel):
    status: str
    run_id: str | None = None
    projection: RunProjection | None = None
    error: str | None = None

    @classmethod
    def success(cls, run_id: str, projection: RunProjection | None = None) -> JobResult:
        return cls(status="SUCCEEDED", run_id=run_id, projection=projection)

    @classmethod
    def failed(cls, error: str) -> JobResult:
        return cls(status="FAILED", error=error)


class _Run(Protocol):
    def run(self) -> Any: ...

    def cancel(self) -> None: ...


RunFactory = Callable[[ExperimentRequest, Callable[[Any], None], RunCancellation], _Run]
ResultLoader = Callable[[str], JobResult]
Comparator = Callable[[str, str], dict[str, Any]]


@dataclass
class _Job:
    request: ExperimentRequest
    lifecycle: ExperimentLifecycle
    cancellation: RunCancellation
    run: _Run | None = None
    result: JobResult | None = None
    events: list[dict[str, Any]] | None = None
    thread: threading.Thread | None = None

    def progress(self) -> JobProgress:
        return JobProgress(
            experiment_id=self.request.experiment_id,
            status=self.lifecycle.status,
            events=tuple(self.events or ()),
        )


class ExperimentJobBoundary:
    """Thread-safe registry of local experiment attempts."""

    def __init__(
        self,
        *,
        factory: RunFactory | None = None,
        orchestrator_factory: RunFactory | None = None,
        run_root: Path | None = None,
        result_loader: ResultLoader | None = None,
        comparator: Comparator | None = None,
    ) -> None:
        selected_factory = factory or orchestrator_factory
        if selected_factory is None:
            raise ValueError("factory is required")
        if factory is not None and orchestrator_factory is not None:
            raise ValueError("factory and orchestrator_factory are mutually exclusive")
        self._factory = selected_factory
        self._result_loader = result_loader or (
            result_loader_for_run_root(run_root) if run_root is not None else self._load_result
        )
        self._comparator = comparator or (
            comparator_for_run_root(run_root) if run_root is not None else self._compare
        )
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.RLock()

    def create(self, request: ExperimentRequest) -> ExperimentLifecycle:
        with self._lock:
            if request.experiment_id in self._jobs:
                raise StableExperimentError(
                    StableExperimentErrorCode.INVALID_REQUEST,
                    "experiment_id already exists",
                )
            job = _Job(
                request=request,
                lifecycle=ExperimentLifecycle(
                    experiment_id=request.experiment_id,
                    status=ExperimentLifecycleStatus.CREATED,
                ),
                cancellation=RunCancellation(),
                events=[],
            )
            self._jobs[request.experiment_id] = job
            return job.lifecycle

    def validate(self, experiment_id: str) -> ExperimentLifecycle:
        with self._lock:
            job = self._job(experiment_id)
            if job.lifecycle.status is not ExperimentLifecycleStatus.CREATED:
                return job.lifecycle
            job.lifecycle = job.lifecycle.transition(ExperimentLifecycleStatus.VALIDATING)
        try:
            # ExperimentRequest validation has already enforced the public
            # shape. The runner factory is the canonical readiness boundary.
            run = self._factory(job.request, self._publish(experiment_id), job.cancellation)
        except Exception as exc:
            with self._lock:
                job.lifecycle = job.lifecycle.transition(ExperimentLifecycleStatus.FAILED)
                job.result = JobResult.failed(f"{type(exc).__name__}: {exc}")
                return job.lifecycle
        with self._lock:
            job.run = run
            if job.cancellation.cancelled:
                job.lifecycle = job.lifecycle.transition(ExperimentLifecycleStatus.CANCELLED)
            else:
                job.lifecycle = job.lifecycle.transition(ExperimentLifecycleStatus.READY)
            return job.lifecycle

    def start(self, experiment_id: str) -> ExperimentLifecycle:
        with self._lock:
            job = self._job(experiment_id)
            if job.lifecycle.status is ExperimentLifecycleStatus.CREATED:
                raise StableExperimentError(StableExperimentErrorCode.NOT_READY, "validate first")
            if job.lifecycle.status is not ExperimentLifecycleStatus.READY:
                raise StableExperimentError(
                    StableExperimentErrorCode.INVALID_TRANSITION,
                    f"cannot start from {job.lifecycle.status.value}",
                )
            if job.run is None:
                raise StableExperimentError(
                    StableExperimentErrorCode.NOT_READY,
                    "runner unavailable",
                )
            job.lifecycle = job.lifecycle.transition(ExperimentLifecycleStatus.RUNNING)
            thread = threading.Thread(target=self._execute, args=(job,), daemon=True)
            job.thread = thread
            thread.start()
            return job.lifecycle

    def status(self, experiment_id: str) -> ExperimentLifecycle:
        with self._lock:
            return self._job(experiment_id).lifecycle

    def progress(self, experiment_id: str) -> JobProgress:
        with self._lock:
            return self._job(experiment_id).progress()

    def cancel(self, experiment_id: str) -> ExperimentLifecycle:
        with self._lock:
            job = self._job(experiment_id)
            if job.lifecycle.status in {
                ExperimentLifecycleStatus.SUCCEEDED,
                ExperimentLifecycleStatus.FAILED,
                ExperimentLifecycleStatus.CANCELLED,
            }:
                return job.lifecycle
            job.cancellation.cancel()
            if job.lifecycle.status in {
                ExperimentLifecycleStatus.CREATED,
                ExperimentLifecycleStatus.VALIDATING,
                ExperimentLifecycleStatus.READY,
            }:
                job.lifecycle = job.lifecycle.transition(ExperimentLifecycleStatus.CANCELLED)
            return job.lifecycle

    def result(self, experiment_id: str) -> JobResult:
        with self._lock:
            job = self._job(experiment_id)
            if job.result is not None:
                return job.result
            if job.lifecycle.status not in {
                ExperimentLifecycleStatus.SUCCEEDED,
                ExperimentLifecycleStatus.FAILED,
                ExperimentLifecycleStatus.CANCELLED,
            }:
                raise StableExperimentError(
                    StableExperimentErrorCode.NOT_READY,
                    "result is not ready",
                )
            return JobResult(status=job.lifecycle.status.value)

    def wait(self, experiment_id: str, *, timeout: float) -> bool:
        with self._lock:
            thread = self._job(experiment_id).thread
        if thread is not None:
            thread.join(timeout=timeout)
        return self.status(experiment_id).status in {
            ExperimentLifecycleStatus.SUCCEEDED,
            ExperimentLifecycleStatus.FAILED,
            ExperimentLifecycleStatus.CANCELLED,
        }

    def rerun(self, experiment_id: str) -> ExperimentLifecycle:
        with self._lock:
            source = self._job(experiment_id)
            request = source.request.model_copy(update={"experiment_id": str(uuid.uuid4())})
        self.create(request)
        self.validate(request.experiment_id)
        return self.start(request.experiment_id)

    def compare(self, left: str, right: str) -> dict[str, Any]:
        self._job(left)
        self._job(right)
        return self._comparator(left, right)

    def _execute(self, job: _Job) -> None:
        assert job.run is not None
        try:
            result = job.run.run()
            run_id = str(getattr(result, "run_id", ""))
            if not run_id:
                raise StableExperimentError(
                    StableExperimentErrorCode.RUN_FAILED,
                    "orchestrator returned no run id",
                )
            loaded = self._result_loader(run_id)
            with self._lock:
                job.result = loaded
                result_status = str(getattr(result, "status", "OK"))
                target = (
                    ExperimentLifecycleStatus.CANCELLED
                    if job.cancellation.cancelled
                    else ExperimentLifecycleStatus.FAILED
                    if loaded.status == "FAILED" or result_status in {"FAILED", "Status.FAILED"}
                    else ExperimentLifecycleStatus.SUCCEEDED
                )
                job.lifecycle = job.lifecycle.transition(target)
        except RunCancelled:
            with self._lock:
                job.lifecycle = job.lifecycle.transition(ExperimentLifecycleStatus.CANCELLED)
                job.result = JobResult(status="CANCELLED")
        except Exception as exc:
            with self._lock:
                target = (
                    ExperimentLifecycleStatus.CANCELLED
                    if job.cancellation.cancelled
                    else ExperimentLifecycleStatus.FAILED
                )
                job.lifecycle = job.lifecycle.transition(target)
                job.result = JobResult(
                    status=target.value,
                    error=None
                    if target is ExperimentLifecycleStatus.CANCELLED
                    else f"{type(exc).__name__}: {exc}",
                )

    def _publish(self, experiment_id: str) -> Callable[[Any], None]:
        def publish(event: Any) -> None:
            if hasattr(event, "as_manifest_entry"):
                value = event.as_manifest_entry()
            elif isinstance(event, Mapping):
                value: dict[str, Any] = {
                    str(key): item for key, item in cast(Mapping[Any, Any], event).items()
                }
            else:
                value = {"detail": str(event)}
            with self._lock:
                job = self._job(experiment_id)
                assert job.events is not None
                job.events.append(value)

        return publish

    def _job(self, experiment_id: str) -> _Job:
        job = self._jobs.get(experiment_id)
        if job is None:
            raise StableExperimentError(StableExperimentErrorCode.NOT_FOUND, experiment_id)
        return job

    def _load_result(self, run_id: str) -> JobResult:
        raise StableExperimentError(
            StableExperimentErrorCode.RUN_FAILED,
            f"no result loader for {run_id}",
        )

    def _compare(self, left: str, right: str) -> dict[str, Any]:
        raise StableExperimentError(
            StableExperimentErrorCode.RUN_FAILED,
            "no comparison store configured",
        )


class ExperimentJobServer:
    """Loopback HTTP adapter for :class:`ExperimentJobBoundary`."""

    host = "127.0.0.1"

    def __init__(self, boundary: ExperimentJobBoundary, *, port: int = 0) -> None:
        self.boundary = boundary
        self._server = _JobHTTPServer((self.host, port), self)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._server.server_port

    def __enter__(self) -> ExperimentJobServer:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        assert self._thread is not None
        self._thread.join(timeout=5)


class _JobHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], owner: ExperimentJobServer) -> None:
        self.owner = owner
        super().__init__(address, _JobHandler)


class _JobHandler(BaseHTTPRequestHandler):
    server_version = "AutoBrainLocalJobs/1"

    def do_POST(self) -> None:
        try:
            path = [unquote(item) for item in urlsplit(self.path).path.split("/") if item]
            if path == ["api", "v1", "experiments"]:
                payload = self._body()
                request = ExperimentRequest.model_validate(payload)
                lifecycle = self._owner().boundary.create(request)
                self._json(201, lifecycle.model_dump(mode="json"))
                return
            if len(path) >= 5 and path[:3] == ["api", "v1", "experiments"]:
                experiment_id = path[3]
                operation = path[4]
                boundary = self._owner().boundary
                if operation == "validate":
                    self._json(200, boundary.validate(experiment_id).model_dump(mode="json"))
                elif operation == "start":
                    self._json(202, boundary.start(experiment_id).model_dump(mode="json"))
                elif operation == "cancel":
                    self._json(202, boundary.cancel(experiment_id).model_dump(mode="json"))
                elif operation == "rerun":
                    self._json(202, boundary.rerun(experiment_id).model_dump(mode="json"))
                elif operation == "compare":
                    # Resolve the left ID before parsing the body so unknown
                    # jobs always have the stable NOT_FOUND contract.
                    boundary.status(experiment_id)
                    other = self._body().get("other_experiment_id")
                    if not isinstance(other, str):
                        raise StableExperimentError(
                            StableExperimentErrorCode.INVALID_REQUEST,
                            "other_experiment_id is required",
                        )
                    self._json(200, boundary.compare(experiment_id, other))
                else:
                    self._json(404, {"error": "NOT_FOUND", "detail": "unknown operation"})
            else:
                self._json(404, {"error": "NOT_FOUND", "detail": "not found"})
        except Exception as exc:
            self._error(exc)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._send_cors_headers()
        self.end_headers()

    def _send_cors_headers(self) -> None:
        """Echo back only an allowed loopback origin.

        The Web wizard is served from a local dev server, so it reaches this
        boundary cross-origin. The grant is scoped to the exact loopback caller
        rather than ``*``, and credentials are never allowed because this
        boundary has no authentication to protect.
        """
        # The body is origin-independent but these headers are not, so caches
        # must key on Origin.
        self.send_header("Vary", "Origin")
        origin = self.headers.get("Origin")
        if origin is None or not is_allowed_origin(origin):
            return
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")

    def do_GET(self) -> None:
        try:
            path = [unquote(item) for item in urlsplit(self.path).path.split("/") if item]
            if len(path) in {4, 5} and path[:3] == ["api", "v1", "experiments"]:
                boundary = self._owner().boundary
                experiment_id = path[3]
                operation = path[4] if len(path) == 5 else "status"
                if operation == "status":
                    payload = boundary.status(experiment_id).model_dump(mode="json")
                elif operation == "progress":
                    payload = boundary.progress(experiment_id).model_dump(mode="json")
                elif operation == "result":
                    payload = boundary.result(experiment_id).model_dump(mode="json")
                else:
                    self._json(404, {"error": "NOT_FOUND", "detail": "unknown operation"})
                    return
                self._json(200, payload)
                return
            self._json(404, {"error": "NOT_FOUND", "detail": "not found"})
        except Exception as exc:
            self._error(exc)

    def _owner(self) -> ExperimentJobServer:
        return cast(_JobHTTPServer, self.server).owner

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        decoded: object = json.loads(self.rfile.read(length) or b"{}")
        value: dict[str, Any]
        if isinstance(decoded, dict):
            value = {str(key): item for key, item in cast(dict[Any, Any], decoded).items()}
        else:
            value = {}
        if not value:
            raise StableExperimentError(
                StableExperimentErrorCode.INVALID_REQUEST, "JSON object required"
            )
        return value

    def _error(self, error: Exception) -> None:
        if isinstance(error, StableExperimentError):
            code = (
                404
                if error.code is StableExperimentErrorCode.NOT_FOUND
                else 409
                if error.code
                in {
                    StableExperimentErrorCode.NOT_READY,
                    StableExperimentErrorCode.INVALID_TRANSITION,
                }
                else 400
            )
            self._json(code, {"error": error.code.value, "detail": redact_text(error.detail)})
        elif isinstance(error, ValidationError):
            self._json(400, {"error": "INVALID_REQUEST", "detail": "request failed validation"})
        else:
            self._json(400, {"error": "INVALID_REQUEST", "detail": redact_text(str(error))})

    def _json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def make_orchestrator_factory(
    config_factory: Callable[[ExperimentRequest], Any],
    *,
    connection_manager: Any = None,
) -> RunFactory:
    """Adapt typed requests to the canonical ``RunOrchestrator.local`` entrypoint.

    The request-to-config mapping is deliberately injected because source
    transport selection is product policy owned by the caller. Credentials
    remain in the normal local auth/config managers and are never part of this
    boundary's request or progress payload.
    """
    from autobrain.orchestration import RunOrchestrator

    def build(
        request: ExperimentRequest,
        sink: Callable[[Any], None],
        cancellation: RunCancellation,
    ) -> _Run:
        config = config_factory(request)
        return RunOrchestrator.local(
            config,
            connection_manager=connection_manager,
            stage_event_sink=sink,
            cancellation=cancellation,
        )

    return build


def result_loader_for_run_root(run_root: Path) -> ResultLoader:
    def load(run_id: str) -> JobResult:
        run_dir = run_root / run_id
        artifact = load_comparison(run_dir / "comparison.json")
        return JobResult.success(run_id, project_comparison(artifact))

    return load


def comparator_for_run_root(run_root: Path) -> Comparator:
    def compare(left: str, right: str) -> dict[str, Any]:
        result = compare_runs(run_root, left, right)
        return result.model_dump(mode="json")

    return compare
