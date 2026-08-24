"""Typed side effects and the framework-neutral UI executor boundary."""

from __future__ import annotations

import subprocess
import sys
import webbrowser
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from autobrain.auth.models import Provider
from autobrain.cancellation import RunCancellation
from autobrain.candidates.gbrain_config import GBrainExecutionConfig
from autobrain.experiment import ExperimentPlan
from autobrain.subscription_domain import ProviderId


@dataclass(frozen=True, order=True)
class EffectHandle:
    value: str


@dataclass(frozen=True)
class LoadConnections:
    provider: ProviderId
    refresh: bool = False


@dataclass(frozen=True)
class InteractiveLogin:
    provider: Provider | ProviderId
    handle: EffectHandle


@dataclass(frozen=True)
class ValidateGBrainProvider:
    config: GBrainExecutionConfig


@dataclass(frozen=True)
class RunExperiment:
    plan: ExperimentPlan
    provider: ProviderId
    handle: EffectHandle


@dataclass(frozen=True)
class CancelActiveRun:
    handle: EffectHandle


@dataclass(frozen=True)
class OpenExactReport:
    path: Path


@dataclass(frozen=True)
class ExitApplication:
    pass


type UiEffect = (
    LoadConnections
    | InteractiveLogin
    | ValidateGBrainProvider
    | RunExperiment
    | CancelActiveRun
    | OpenExactReport
    | ExitApplication
)


class TerminalLifecycle(Protocol):
    def suspended(self) -> AbstractContextManager[None]: ...


class LoginRunner(Protocol):
    def __call__(self, effect: InteractiveLogin) -> None: ...


@dataclass(frozen=True)
class _RunExecution:
    cancellation: RunCancellation
    started_at: float


class EffectRegistry:
    """The sole mutable registry for host-owned effect resources."""

    def __init__(self) -> None:
        self._runs: dict[EffectHandle, _RunExecution] = {}
        self._logins: dict[EffectHandle, AbstractContextManager[None]] = {}

    def register_run(self, handle: EffectHandle, *, started_at: float) -> RunCancellation:
        if handle in self._runs:
            raise ValueError(f"duplicate run effect handle: {handle.value}")
        cancellation = RunCancellation()
        self._runs[handle] = _RunExecution(cancellation, started_at)
        return cancellation

    def run_started_at(self, handle: EffectHandle) -> float | None:
        execution = self._runs.get(handle)
        return execution.started_at if execution is not None else None

    def cancel_run(self, handle: EffectHandle) -> bool:
        execution = self._runs.get(handle)
        if execution is None:
            return False
        execution.cancellation.cancel()
        return True

    def settle_run(self, handle: EffectHandle) -> RunCancellation | None:
        execution = self._runs.pop(handle, None)
        return execution.cancellation if execution is not None else None

    def register_login(
        self,
        handle: EffectHandle,
        suspension: AbstractContextManager[None],
    ) -> None:
        if handle in self._logins:
            raise ValueError(f"duplicate login effect handle: {handle.value}")
        self._logins[handle] = suspension

    def settle_login(self, handle: EffectHandle) -> AbstractContextManager[None] | None:
        return self._logins.pop(handle, None)


class EffectExecutor(Protocol):
    """Host implementation used by Textual; widgets never receive this object."""

    def submit(self, effect: UiEffect) -> None: ...


def login_command(provider: Provider | ProviderId) -> tuple[str, ...]:
    if provider is Provider.SLACK:
        return (sys.executable, "-m", "autobrain.cli", "source", "slack")
    if isinstance(provider, Provider):
        return (sys.executable, "-m", "autobrain.cli", "auth", provider.value)
    return (
        sys.executable,
        "-m",
        "autobrain.cli",
        "subscription",
        "setup",
        "--provider",
        provider.value,
    )


def run_login_process(
    effect: InteractiveLogin,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    runner(login_command(effect.provider), check=False, text=True)


def execute_interactive_login(
    effect: InteractiveLogin,
    *,
    terminal: TerminalLifecycle,
    login: LoginRunner = run_login_process,
) -> None:
    """Release the alternate screen around a vendor-owned interactive login."""
    with terminal.suspended():
        login(effect)


def open_exact_report(effect: OpenExactReport) -> bool:
    """Open only the immutable report path carried by the completed run."""
    return webbrowser.open(effect.path.resolve().as_uri())
