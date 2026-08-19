"""Provider-neutral contracts for local subscription execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class SubscriptionStatus(StrEnum):
    READY = "READY"
    SUBSCRIPTION_CLI_UNAVAILABLE = "SUBSCRIPTION_CLI_UNAVAILABLE"
    SUBSCRIPTION_AUTH_UNAVAILABLE = "SUBSCRIPTION_AUTH_UNAVAILABLE"
    SUBSCRIPTION_EXECUTION_UNAVAILABLE = "SUBSCRIPTION_EXECUTION_UNAVAILABLE"


class SubscriptionError(RuntimeError):
    """A typed failure from the local subscription execution boundary."""

    def __init__(self, status: SubscriptionStatus, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class ProviderId(StrEnum):
    CODEX = "codex"


class AuthKind(StrEnum):
    CONSUMER_SUBSCRIPTION = "consumer_subscription"


class ProviderCapability(StrEnum):
    STATUS = "status"
    LOGIN = "login"
    STRUCTURED_ANSWER = "structured_answer"
    READ_ONLY = "read_only"


class UsageKind(StrEnum):
    NATIVE = "native"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class AnswerUsage:
    kind: UsageKind
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class ProviderIdentity:
    provider: ProviderId
    model: str | None
    cli_version: str | None
    auth_kind: AuthKind


@dataclass(frozen=True)
class ProviderAnswer:
    text: str
    usage: AnswerUsage
    identity: ProviderIdentity


@dataclass(frozen=True)
class ProviderConfig:
    command: str
    model: str | None
    timeout_seconds: float


class SubscriptionProvider(Protocol):
    """The application-facing boundary for a subscription-backed provider."""

    identity: ProviderIdentity

    def login(self) -> int: ...

    def status(self) -> SubscriptionStatus: ...

    def answer(self, prompt: str) -> ProviderAnswer: ...


@dataclass(frozen=True)
class StructuredOutput:
    answer: str
    usage: AnswerUsage
    cli_version: str | None = None
