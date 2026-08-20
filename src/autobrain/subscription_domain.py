"""Provider-neutral contracts for local subscription execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class SubscriptionStatus(StrEnum):
    READY = "READY"
    UNSUPPORTED = "UNSUPPORTED"
    SUBSCRIPTION_CLI_UNAVAILABLE = "SUBSCRIPTION_CLI_UNAVAILABLE"
    SUBSCRIPTION_AUTH_UNAVAILABLE = "SUBSCRIPTION_AUTH_UNAVAILABLE"
    SUBSCRIPTION_EXECUTION_UNAVAILABLE = "SUBSCRIPTION_EXECUTION_UNAVAILABLE"


class SubscriptionFailureReason(StrEnum):
    PROVIDER_UNSUPPORTED = "PROVIDER_UNSUPPORTED"
    LOGIN_UNAVAILABLE = "LOGIN_UNAVAILABLE"
    AUTH_UNAVAILABLE = "AUTH_UNAVAILABLE"
    AUTH_KIND_UNSUPPORTED = "AUTH_KIND_UNSUPPORTED"
    PROBE_IN_PROGRESS = "PROBE_IN_PROGRESS"
    STATUS_TIMEOUT = "STATUS_TIMEOUT"
    STATUS_CANCELLED = "STATUS_CANCELLED"
    STATUS_NONZERO = "STATUS_NONZERO"
    STATUS_MALFORMED_OUTPUT = "STATUS_MALFORMED_OUTPUT"
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"
    EXECUTION_CANCELLED = "EXECUTION_CANCELLED"
    EXECUTION_NONZERO = "EXECUTION_NONZERO"
    EXECUTION_MALFORMED_OUTPUT = "EXECUTION_MALFORMED_OUTPUT"
    EXECUTION_EMPTY_ANSWER = "EXECUTION_EMPTY_ANSWER"


@dataclass(frozen=True)
class SubscriptionStatusReport:
    status: SubscriptionStatus
    reason: SubscriptionFailureReason | None = None
    detail: str = ""


class SubscriptionError(RuntimeError):
    """A typed failure from the local subscription execution boundary."""

    def __init__(
        self,
        status: SubscriptionStatus,
        detail: str,
        *,
        reason: SubscriptionFailureReason | None = None,
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.reason = reason


class ProviderId(StrEnum):
    CODEX = "codex"
    CLAUDE = "claude"
    KIMI = "kimi"
    GROK = "grok"


class AuthKind(StrEnum):
    CONSUMER_SUBSCRIPTION = "consumer_subscription"
    UNSUPPORTED = "unsupported"


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
    execution_ms: float | None = None


@dataclass(frozen=True)
class ProviderConfig:
    command: str
    model: str | None
    timeout_seconds: float


class SubscriptionProvider(Protocol):
    """The application-facing boundary for a subscription-backed provider."""

    @property
    def identity(self) -> ProviderIdentity: ...

    @property
    def last_answer(self) -> ProviderAnswer | None: ...

    def probe_identity(self) -> ProviderIdentity: ...

    def login(self) -> int: ...

    def probe_status(self) -> SubscriptionStatusReport: ...

    def status(self) -> SubscriptionStatus: ...

    def answer(self, prompt: str) -> ProviderAnswer: ...

    def ask(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class StructuredOutput:
    answer: str
    usage: AnswerUsage
    model: str | None = None
    cli_version: str | None = None
