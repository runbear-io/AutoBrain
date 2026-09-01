import shutil
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from autobrain.connectors.readiness import (
    SourceTransportReadiness,
    SourceTransportRegistry,
    TransportGovernanceCode,
    TransportReadinessState,
)
from autobrain.contracts import (
    ModelAccessMode,
    ModelAccessProfileV1,
    ModelCapabilityStatus,
    SourceProvider,
    SourceTransportMode,
)
from autobrain.embedding import check_embedding_backend
from autobrain.model_access import ModelAccessStatus, inspect_model_access
from autobrain.models import CandidatePin, CheckResult, DoctorPaths, Status, StrictModel
from autobrain.paths import AutoBrainPaths, PathConfinementError
from autobrain.preflight_google_drive import check_google_drive_source
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
from autobrain.source_store import SlackSourceStore
from autobrain.subscription import ProviderId, provider_registry

_MIN_NODE = (24, 0, 0)
_MIN_BUN = (1, 3, 10)


class PrecredentialReadiness(StrictModel):
    """One deterministic, offline readiness result for the local doctor path."""

    schema_version: int = 1
    ready: bool
    source_transport: SourceTransportReadiness
    model_access: ModelAccessProfileV1
    governance_codes: list[str] = Field(default_factory=list)
    remediation: list[str] = Field(default_factory=list)


class DoctorReport(StrictModel):
    schema_version: int = 1
    status: Status
    generated_at: datetime
    checks: list[CheckResult]
    environment: EnvironmentReadiness
    paths: DoctorPaths
    readiness: PrecredentialReadiness
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
        source_provider: SourceProvider = SourceProvider.SLACK,
        source_mode: SourceTransportMode = SourceTransportMode.EXPORT_ARCHIVE,
        source_oauth_config_path: Path | None = None,
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
        self.source_provider = source_provider
        self.source_mode = source_mode
        self.source_oauth_config_path = source_oauth_config_path

    def run(self, *, offline: bool = False) -> DoctorReport:
        if offline:
            return self._run_offline()
        checks = [self._runtime("python", (3, 12, 0)), self._runtime("node", _MIN_NODE)]
        checks.extend((self._runtime("bun", _MIN_BUN), self._writable_paths()))
        checks.append(self._boolean("keyring", self.keyring_probe, Status.ENV_UNAVAILABLE))
        readiness = self.environment.readiness()
        subscription_checks = [
            check_subscription_provider(
                provider,
                command_runner=self.command_runner,
                executable_finder=self.executable_finder,
            )
            for provider in provider_registry().provider_ids
        ]
        slack_check = check_slack_source(paths=self.paths, readiness=readiness)
        checks.extend(
            (
                *subscription_checks,
                check_embedding_backend(self.embedding_environ),
                slack_check,
                check_google_drive_source(),
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
        source_transport = self._source_transport(slack_check)
        model_access = self._model_access()
        readiness_result = self._readiness(source_transport, model_access)
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
            readiness=readiness_result,
            candidate_pins=pins,
        )

    def _run_offline(self) -> DoctorReport:
        checks = [
            self._offline_executable("python", sys.executable),
            self._offline_executable("node", self.executable_finder("node")),
            self._offline_executable("bun", self.executable_finder("bun")),
            self._offline_paths(),
            self._not_probed(
                "keyring",
                "credential backend probe skipped in offline mode; "
                "remediation: configure a local keyring",
            ),
        ]
        for provider in ProviderId:
            name = (
                "chatgpt_subscription"
                if provider is ProviderId.CODEX
                else f"{provider.value}_subscription"
            )
            executable = self.executable_finder(provider.value)
            if executable is None:
                checks.append(
                    CheckResult(
                        name=name,
                        status=Status.UNAVAILABLE,
                        detail=(
                            f"UNAVAILABLE: {provider.value} CLI not found; "
                            f"remediation: install {provider.value}"
                        ),
                        remediation=f"Install the {provider.value} CLI.",
                    )
                )
            else:
                checks.append(
                    self._not_probed(
                        name,
                        (
                            "NOT_PROBED: credential status not probed in offline mode; "
                            f"remediation: run `autobrain subscription status --provider "
                            f"{provider.value}` or `autobrain subscription setup --provider "
                            f"{provider.value}`"
                        ),
                        executable,
                        (
                            f"Run `autobrain subscription status --provider {provider.value}` "
                            "outside offline mode."
                        ),
                    )
                )
        pins, pin_check = self._pins()
        checks.append(pin_check)
        checks.extend(
            (
                self._not_probed(
                    "semantic_embeddings",
                    "embedding capability not probed in offline mode; "
                    "remediation: configure a supported local or provider backend",
                ),
                self._not_probed(
                    "slack_source",
                    "source credentials and transport not probed in offline mode; "
                    "remediation: configure a local Slack export archive",
                ),
                self._not_probed(
                    "google_drive_source",
                    "provider source not probed in offline mode; "
                    "remediation: use a local sanitized fixture",
                ),
                self._not_probed(
                    "callback",
                    "callback binding not probed in offline mode; "
                    "remediation: run normal `autobrain doctor`",
                ),
                self._not_probed(
                    "browser_open",
                    "browser availability not probed in offline mode; "
                    "remediation: run normal `autobrain doctor`",
                ),
            )
        )
        source_transport = SourceTransportRegistry(
            config_path=self.source_oauth_config_path
        ).resolve(self.source_provider, self.source_mode)
        model_access = self._model_access()
        readiness_result = self._readiness(source_transport, model_access)
        readiness_result = readiness_result.model_copy(
            update={
                "ready": False,
                "remediation": [
                    "Offline mode does not probe credentials, providers, or capabilities; "
                    "run normal `autobrain doctor` for live readiness.",
                    *readiness_result.remediation,
                ],
            }
        )
        return DoctorReport(
            status=self._overall(checks),
            generated_at=datetime.now(UTC),
            checks=checks,
            environment=self.environment.readiness(),
            paths=DoctorPaths(
                root=str(self.paths.root),
                runs=str(self.paths.runs),
                tools=str(self.paths.tools),
                cache=str(self.paths.cache),
            ),
            readiness=readiness_result,
            candidate_pins=pins,
        )

    def _offline_paths(self) -> CheckResult:
        directories = (self.paths.root, self.paths.runs, self.paths.tools, self.paths.cache)
        if all(directory.is_dir() and not directory.is_symlink() for directory in directories):
            return CheckResult(
                name="paths",
                status=Status.OK,
                detail="run, tool, and cache directories exist (offline read-only check)",
                path=str(self.paths.root),
            )
        return CheckResult(
            name="paths",
            status=Status.UNAVAILABLE,
            detail=(
                "UNAVAILABLE: required local state directories are absent; "
                "remediation: run `autobrain setup`"
            ),
            path=str(self.paths.root),
            remediation="Run `autobrain setup` to create the local state directories.",
        )

    @staticmethod
    def _offline_executable(name: str, executable: str | None) -> CheckResult:
        if executable is None:
            return CheckResult(
                name=name,
                status=Status.UNAVAILABLE,
                detail=f"UNAVAILABLE: {name} executable not found; remediation: install {name}",
                remediation=f"Install the {name} executable.",
            )
        return CheckResult(
            name=name,
            status=Status.NOT_PROBED,
            detail=(
                "NOT_PROBED: executable found; version not checked in offline mode; "
                "remediation: run normal `autobrain doctor`"
            ),
            path=executable,
            remediation="Run normal `autobrain doctor` to verify the executable version.",
        )

    @staticmethod
    def _not_probed(
        name: str, detail: str, path: str | None = None, remediation: str | None = None
    ) -> CheckResult:
        return CheckResult(
            name=name,
            status=Status.NOT_PROBED,
            detail=detail,
            path=path,
            remediation=remediation or "Run normal `autobrain doctor` to probe this capability.",
        )

    def _source_transport(self, slack_check: CheckResult) -> SourceTransportReadiness:
        del slack_check
        result = SourceTransportRegistry(config_path=self.source_oauth_config_path).resolve(
            self.source_provider, self.source_mode
        )
        if (
            self.source_provider is SourceProvider.SLACK
            and self.source_mode is SourceTransportMode.EXPORT_ARCHIVE
        ):
            archive_status = SlackSourceStore(self.paths.sources).status()
            if archive_status.ready:
                return result
            return result.model_copy(
                update={
                    "state": TransportReadinessState.UNAVAILABLE,
                    "ready": False,
                    "governance_code": TransportGovernanceCode.SOURCE_TRANSPORT_UNAVAILABLE,
                    "detail": archive_status.detail,
                    "remediation": "Configure and verify a local Slack export archive.",
                }
            )
        return result

    def _model_access(self) -> ModelAccessProfileV1:
        status: ModelAccessStatus = inspect_model_access(self.embedding_environ)
        profiles = status.profiles
        # Prefer a verified recommendation-capable profile, then a ready
        # subscription, while keeping selection stable for empty environments.
        return max(
            profiles,
            key=lambda profile: (
                profile.recommendation_eligible,
                profile.chat is ModelCapabilityStatus.READY
                and profile.verifier is ModelCapabilityStatus.READY,
                profile.mode is ModelAccessMode.PROVIDER_API_BYOK,
            ),
        )

    @staticmethod
    def _readiness(
        source_transport: SourceTransportReadiness,
        model_access: ModelAccessProfileV1,
    ) -> PrecredentialReadiness:
        codes = [source_transport.governance_code.value]
        remediation: list[str] = []
        if not source_transport.ready:
            remediation.append(source_transport.remediation or source_transport.detail)
        if not model_access.recommendation_eligible:
            remediation.extend(model_access.diagnostics)
        return PrecredentialReadiness(
            ready=source_transport.ready and model_access.recommendation_eligible,
            source_transport=source_transport,
            model_access=model_access,
            governance_codes=sorted(set(codes)),
            remediation=remediation,
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
            Status.UNAVAILABLE,
            Status.ENV_UNAVAILABLE,
            Status.MISSING_PROVIDER,
            Status.MCP_AUTH_UNAVAILABLE,
            Status.CAPABILITY_UNAVAILABLE,
        ):
            if any(check.status is status for check in checks):
                return status
        return Status.OK
