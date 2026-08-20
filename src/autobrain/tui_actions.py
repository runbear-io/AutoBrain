"""Semantic actions emitted by the AutoBrain user interface."""

from __future__ import annotations

from dataclasses import dataclass

from autobrain.auth.models import Provider
from autobrain.models import CandidateId
from autobrain.orchestration import RunResult, StageEvent
from autobrain.subscription_domain import ProviderId
from autobrain.tui_effects import EffectHandle
from autobrain.tui_runtime import ConnectionSnapshot


@dataclass(frozen=True)
class Navigate:
    screen: str


@dataclass(frozen=True)
class BeginSetup:
    pass


@dataclass(frozen=True)
class GoBack:
    pass


@dataclass(frozen=True)
class SelectProvider:
    provider: ProviderId


@dataclass(frozen=True)
class ToggleSource:
    provider: Provider


@dataclass(frozen=True)
class SkipSource:
    provider: Provider


@dataclass(frozen=True)
class ToggleCandidate:
    candidate: CandidateId


@dataclass(frozen=True)
class RefreshConnections:
    pass


@dataclass(frozen=True)
class RequestLogin:
    provider: Provider | ProviderId


@dataclass(frozen=True)
class ConnectionsLoaded:
    snapshot: ConnectionSnapshot


@dataclass(frozen=True)
class LoginSettled:
    handle: EffectHandle
    error: str = ""


@dataclass(frozen=True)
class StartRun:
    pass


@dataclass(frozen=True)
class RunStarted:
    handle: EffectHandle
    started_at: float


@dataclass(frozen=True)
class StageObserved:
    event: StageEvent
    observed_at: float


@dataclass(frozen=True)
class CancelRun:
    pass


@dataclass(frozen=True)
class RequestQuit:
    pass


@dataclass(frozen=True)
class RunCompleted:
    result: RunResult


@dataclass(frozen=True)
class RunFailed:
    reason: str


@dataclass(frozen=True)
class OpenReport:
    pass


@dataclass(frozen=True)
class ResetRun:
    pass


type UiAction = (
    Navigate
    | BeginSetup
    | GoBack
    | SelectProvider
    | ToggleSource
    | SkipSource
    | ToggleCandidate
    | RefreshConnections
    | RequestLogin
    | ConnectionsLoaded
    | LoginSettled
    | StartRun
    | RunStarted
    | StageObserved
    | CancelRun
    | RequestQuit
    | RunCompleted
    | RunFailed
    | OpenReport
    | ResetRun
)
