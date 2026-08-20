import shutil
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from autobrain.embedding import check_embedding_backend
from autobrain.models import CandidatePin, CheckResult, DoctorPaths, Status, StrictModel
from autobrain.paths import AutoBrainPaths, PathConfinementError
from autobrain.preflight_slack import check_slack_source
from autobrain.preflight_subscription import check_subscription_provider
from autobrain.preflight_support import (
    CommandResult,
    format_version,
    load_candidate_pins,
    parse_version,
    probe_writable,
    run_command,
)
from autobrain.preflight_support import (
    browser_available as default_browser_available,
)
from autobrain.preflight_support import (
    callback_available as default_callback_available,
)
from autobrain.preflight_support import (
    keyring_available as default_keyring_available,
)
from autobrain.secrets import EnvironmentReadiness, RuntimeEnvironment
from autobrain.subscription import ProviderId

_MIN_NODE = (24, 0, 0)
_MIN_BUN = (1, 3, 10)


class DoctorReport(StrictModel):
    schema_version: int = 1
    status: Status
    generated_at: datetime
    checks: list[CheckResult]
    environment: EnvironmentReadiness
    paths: DoctorPaths
    candidate_pins: list[CandidatePin] = Field(default_factory=list)


class Preflight:
    def __init__(
        self,
        *,
        paths: AutoBrainPaths,
        environment: RuntimeEnvironment,
        pins_path: Path | None = None,
        callback_host: str = "127.0.0.1",
        callback_port: int = 8765,
        callback_port_error: str | None = None,
        command_runner: Callable[[tuple[str, ...], float], CommandResult] = run_command,
        executable_finder: Callable[[str], str | None] = shutil.which,
        keyring_available: Callable[[], bool] = default_keyring_available,
        callback_available: Callable[[str, int], bool] = default_callback_available,
        browser_available: Callable[[], bool] = default_browser_available,
        subscription_provider: ProviderId = ProviderId.CODEX,
        embedding_environ: Mapping[str, str] | None = None,
    ) -> None:
        self.paths = paths
        self.environment = environment
        self.pins_path = pins_path
        self.callback_host = callback_host
        self.callback_port = callback_port
        self.callback_port_error = callback_port_error
        self.command_runner = command_runner
        self.executable_finder = executable_finder
        self.keyring_probe = keyring_available
        self.callback_probe = callback_available
        self.browser_probe = browser_available
        self.subscription_provider = subscription_provider
        self.embedding_environ = dict(embedding_environ or {})

    def run(self) -> DoctorReport:
        checks = [self._runtime("python", (3, 12, 0)), self._runtime("node", _MIN_NODE)]
        checks.extend((self._runtime("bun", _MIN_BUN), self._writable_paths()))
        checks.append(self._boolean("keyring", self.keyring_probe, Status.ENV_UNAVAILABLE))
        readiness = self.environment.readiness()
        checks.extend(
            (
                check_subscription_provider(
                    self.subscription_provider,
                    command_runner=self.command_runner,
                    executable_finder=self.executable_finder,
                ),
                check_embedding_backend(self.embedding_environ),
                check_slack_source(paths=self.paths, readiness=readiness),
            )
        )
        pins, pin_check = self._pins()
        checks.append(pin_check)
        callback_check = (
            CheckResult(
                name="callback",
                status=Status.CAPABILITY_UNAVAILABLE,
                detail=self.callback_port_error,
            )
            if self.callback_port_error is not None
            else self._boolean(
                "callback",
                lambda: self.callback_probe(self.callback_host, self.callback_port),
                Status.CAPABILITY_UNAVAILABLE,
                f"{self.callback_host}:{self.callback_port}",
            )
        )
        checks.extend(
            (
                callback_check,
                self._boolean("browser_open", self.browser_probe, Status.CAPABILITY_UNAVAILABLE),
            )
        )
        return DoctorReport(
            status=self._overall(checks),
            generated_at=datetime.now(UTC),
            checks=checks,
            environment=readiness,
            paths=DoctorPaths(
                root=str(self.paths.root),
                runs=str(self.paths.runs),
                tools=str(self.paths.tools),
                cache=str(self.paths.cache),
            ),
            candidate_pins=pins,
        )

    def _runtime(self, name: str, minimum: tuple[int, int, int] | None = None) -> CheckResult:
        executable = sys.executable if name == "python" else self.executable_finder(name)
        if executable is None:
            return CheckResult(name=name, status=Status.ENV_UNAVAILABLE, detail=f"{name} not found")
        try:
            result = self.command_runner((executable, "--version"), 3.0)
            output = result.stdout or result.stderr
            if result.returncode != 0:
                return CheckResult(
                    name=name,
                    status=Status.ENV_UNAVAILABLE,
                    detail=output or "version command failed",
                    path=executable,
                )
            version = parse_version(output)
            if version is None:
                return CheckResult(
                    name=name,
                    status=Status.ENV_UNAVAILABLE,
                    detail=f"unparseable version: {output}",
                    path=executable,
                )
            if minimum is not None and version < minimum:
                return CheckResult(
                    name=name,
                    status=Status.ENV_UNAVAILABLE,
                    detail=f"requires >= {format_version(minimum)}",
                    version=format_version(version),
                    path=executable,
                )
            return CheckResult(
                name=name,
                status=Status.OK,
                detail="available",
                version=format_version(version),
                path=executable,
            )
        except (OSError, RuntimeError, TimeoutError) as error:
            return CheckResult(
                name=name, status=Status.ENV_UNAVAILABLE, detail=str(error), path=executable
            )

    def _writable_paths(self) -> CheckResult:
        try:
            self.paths.ensure_base_dirs()
            probe_writable((self.paths.runs, self.paths.tools, self.paths.cache))
            return CheckResult(
                name="paths",
                status=Status.OK,
                detail="run, tool, and cache directories are writable",
                path=str(self.paths.root),
            )
        except (OSError, PathConfinementError) as error:
            return CheckResult(
                name="paths",
                status=Status.ENV_UNAVAILABLE,
                detail=str(error),
                path=str(self.paths.root),
            )

    def _pins(self) -> tuple[list[CandidatePin], CheckResult]:
        try:
            pins = load_candidate_pins(self.pins_path)
            location = (
                str(self.pins_path)
                if self.pins_path is not None
                else "package:autobrain/candidate-pins.json"
            )
            return pins.candidates, CheckResult(
                name="candidate_pins",
                status=Status.OK,
                detail="exact approved versions, commits, and licenses",
                path=location,
            )
        except (OSError, ValueError) as error:
            location = (
                str(self.pins_path)
                if self.pins_path is not None
                else "package:autobrain/candidate-pins.json"
            )
            return [], CheckResult(
                name="candidate_pins",
                status=Status.FAILED,
                detail=str(error),
                path=location,
            )

    @staticmethod
    def _boolean(
        name: str, probe: Callable[[], bool], failure: Status, detail: str = ""
    ) -> CheckResult:
        try:
            available = probe()
        except (OSError, RuntimeError) as error:
            return CheckResult(name=name, status=failure, detail=str(error))
        return CheckResult(
            name=name,
            status=Status.OK if available else failure,
            detail=detail or ("available" if available else "unavailable"),
        )

    @staticmethod
    def _overall(checks: list[CheckResult]) -> Status:
        for status in (
            Status.FAILED,
            Status.ENV_UNAVAILABLE,
            Status.MISSING_PROVIDER,
            Status.MCP_AUTH_UNAVAILABLE,
            Status.CAPABILITY_UNAVAILABLE,
        ):
            if any(check.status is status for check in checks):
                return status
        return Status.OK
