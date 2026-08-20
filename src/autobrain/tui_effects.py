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
from autobrain.experiment import ExperimentPlan
from autobrain.subscription_domain import ProviderId


@dataclass(frozen=True)
class LoadConnections:
    provider: ProviderId
    refresh: bool = False


@dataclass(frozen=True)
class InteractiveLogin:
    provider: Provider | ProviderId


@dataclass(frozen=True)
class RunExperiment:
    plan: ExperimentPlan
    provider: ProviderId


@dataclass(frozen=True)
class CancelActiveRun:
    pass


@dataclass(frozen=True)
class OpenExactReport:
    path: Path


@dataclass(frozen=True)
class ExitApplication:
    pass


type UiEffect = (
    LoadConnections
    | InteractiveLogin
    | RunExperiment
    | CancelActiveRun
    | OpenExactReport
    | ExitApplication
)


class TerminalLifecycle(Protocol):
    def suspended(self) -> AbstractContextManager[None]: ...


class LoginRunner(Protocol):
    def __call__(self, effect: InteractiveLogin) -> None: ...


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
