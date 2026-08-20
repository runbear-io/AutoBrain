"""Verified Claude Code CLI adapter for consumer subscription execution."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import cast

from autobrain.cancellation import RunCancellation
from autobrain.subscription_domain import (
    AnswerUsage,
    AuthKind,
    ProviderAnswer,
    ProviderCapability,
    ProviderConfig,
    ProviderId,
    ProviderIdentity,
    SubscriptionError,
    SubscriptionFailureReason,
    SubscriptionStatus,
    SubscriptionStatusReport,
    UsageKind,
)
from autobrain.subscription_process import (
    ProviderProcessCancelled,
    ProviderProcessTimeout,
    run_interactive_provider_process,
    run_provider_process,
    sanitize_diagnostic,
)

Runner = Callable[[Sequence[str], str, float], subprocess.CompletedProcess[str]]
InteractiveRunner = Callable[[Sequence[str]], int]


@dataclass(frozen=True)
class ClaudeSubscriptionConfig:
    command: str = "claude"
    model: str | None = None
    timeout_seconds: float = 120.0

    @classmethod
    def from_environ(cls) -> ClaudeSubscriptionConfig:
        timeout_raw = os.environ.get("AUTOBRAIN_SUBSCRIPTION_TIMEOUT_SECONDS", "120")
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise ValueError("AUTOBRAIN_SUBSCRIPTION_TIMEOUT_SECONDS must be a number") from exc
        if timeout_seconds <= 0:
            raise ValueError("AUTOBRAIN_SUBSCRIPTION_TIMEOUT_SECONDS must be greater than 0")
        return cls(
            command=os.environ.get("AUTOBRAIN_CLAUDE_COMMAND", "claude"),
            model=os.environ.get("AUTOBRAIN_SUBSCRIPTION_MODEL") or None,
            timeout_seconds=timeout_seconds,
        )

    def provider_config(self) -> ProviderConfig:
        return ProviderConfig(
            command=self.command,
            model=self.model,
            timeout_seconds=self.timeout_seconds,
        )


@dataclass
class ClaudeSubscriptionClient:
    config: ClaudeSubscriptionConfig
    runner: Runner = run_provider_process
    interactive_runner: InteractiveRunner = run_interactive_provider_process
    cancellation: RunCancellation | None = None
    last_answer: ProviderAnswer | None = field(default=None, init=False)
    _identity: ProviderIdentity | None = field(default=None, init=False)

    @property
    def identity(self) -> ProviderIdentity:
        if self._identity is not None:
            return self._identity
        return ProviderIdentity(
            provider=ProviderId.CLAUDE,
            model=self.config.model,
            cli_version=None,
            auth_kind=AuthKind.CONSUMER_SUBSCRIPTION,
        )

    @property
    def capabilities(self) -> frozenset[ProviderCapability]:
        return frozenset(
            {
                ProviderCapability.STATUS,
                ProviderCapability.LOGIN,
                ProviderCapability.STRUCTURED_ANSWER,
                ProviderCapability.READ_ONLY,
            }
        )

    def probe_identity(self) -> ProviderIdentity:
        try:
            result = self._run([self.config.command, "--version"], "")
        except (
            subprocess.TimeoutExpired,
            ProviderProcessTimeout,
            ProviderProcessCancelled,
            OSError,
        ):
            return self.identity
        cli_version = sanitize_diagnostic(result.stdout) if result.returncode == 0 else ""
        base = self.identity
        self._identity = ProviderIdentity(
            provider=base.provider,
            model=base.model,
            cli_version=cli_version or None,
            auth_kind=base.auth_kind,
        )
        return self._identity

    def login(self) -> int:
        if shutil.which(self.config.command) is None:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_CLI_UNAVAILABLE,
                f"Claude Code CLI not found: {self.config.command}",
                reason=SubscriptionFailureReason.LOGIN_UNAVAILABLE,
            )
        try:
            return self.interactive_runner([self.config.command, "auth", "login", "--claudeai"])
        except ProviderProcessCancelled as exc:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Claude subscription login cancelled",
                reason=SubscriptionFailureReason.EXECUTION_CANCELLED,
            ) from exc

    def probe_status(self) -> SubscriptionStatusReport:
        if shutil.which(self.config.command) is None:
            return SubscriptionStatusReport(
                status=SubscriptionStatus.SUBSCRIPTION_CLI_UNAVAILABLE,
                reason=SubscriptionFailureReason.LOGIN_UNAVAILABLE,
                detail=f"Claude Code CLI not found: {self.config.command}",
            )
        try:
            result = self._run(
                [self.config.command, "auth", "status", "--json"],
                "",
            )
        except (subprocess.TimeoutExpired, ProviderProcessTimeout):
            return SubscriptionStatusReport(
                status=SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                reason=SubscriptionFailureReason.STATUS_TIMEOUT,
                detail="Claude subscription status timed out",
            )
        except ProviderProcessCancelled:
            return SubscriptionStatusReport(
                status=SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                reason=SubscriptionFailureReason.STATUS_CANCELLED,
                detail="Claude subscription status cancelled",
            )
        if result.returncode != 0:
            return SubscriptionStatusReport(
                status=SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                reason=SubscriptionFailureReason.STATUS_NONZERO,
                detail=sanitize_diagnostic(result.stderr)
                or "Claude subscription status command failed",
            )
        payload = _json_object(result.stdout)
        if payload is None:
            return SubscriptionStatusReport(
                status=SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                reason=SubscriptionFailureReason.STATUS_MALFORMED_OUTPUT,
                detail="Claude subscription status returned malformed JSON",
            )
        logged_in = payload.get("loggedIn")
        auth_method = payload.get("authMethod")
        api_provider = payload.get("apiProvider")
        if (
            type(logged_in) is not bool
            or not isinstance(auth_method, str)
            or not isinstance(api_provider, str)
        ):
            return SubscriptionStatusReport(
                status=SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                reason=SubscriptionFailureReason.STATUS_MALFORMED_OUTPUT,
                detail="Claude subscription status omitted required authentication fields",
            )
        if not logged_in:
            return SubscriptionStatusReport(
                status=SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE,
                reason=SubscriptionFailureReason.AUTH_UNAVAILABLE,
                detail="Run `claude auth login --claudeai` before using Claude subscription mode",
            )
        if auth_method != "oauth_token" or api_provider != "firstParty":
            return SubscriptionStatusReport(
                status=SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE,
                reason=SubscriptionFailureReason.AUTH_KIND_UNSUPPORTED,
                detail=(
                    "Claude is authenticated with a non-consumer credential; run "
                    "`claude auth login --claudeai` and select the Claude subscription"
                ),
            )
        return SubscriptionStatusReport(status=SubscriptionStatus.READY)

    def status(self) -> SubscriptionStatus:
        return self.probe_status().status

    def answer(self, prompt: str) -> ProviderAnswer:
        self.last_answer = None
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        report = self.probe_status()
        if report.status is not SubscriptionStatus.READY:
            raise SubscriptionError(
                report.status,
                report.detail or report.status.value,
                reason=report.reason,
            )
        args = [
            self.config.command,
            "-p",
            "--output-format",
            "json",
            "--tools",
            "",
            "--safe-mode",
            "--no-session-persistence",
            "--permission-mode",
            "dontAsk",
        ]
        if self.config.model is not None:
            args.extend(["--model", self.config.model])
        try:
            result = self._run(args, prompt)
        except (subprocess.TimeoutExpired, ProviderProcessTimeout) as exc:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Claude subscription execution timed out",
                reason=SubscriptionFailureReason.EXECUTION_TIMEOUT,
            ) from exc
        except ProviderProcessCancelled as exc:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Claude subscription execution cancelled",
                reason=SubscriptionFailureReason.EXECUTION_CANCELLED,
            ) from exc
        if result.returncode != 0:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                sanitize_diagnostic(result.stderr) or "Claude subscription execution failed",
                reason=SubscriptionFailureReason.EXECUTION_NONZERO,
            )
        payload = _json_object(result.stdout)
        if payload is None:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Claude returned malformed structured output",
                reason=SubscriptionFailureReason.EXECUTION_MALFORMED_OUTPUT,
            )
        if payload.get("type") != "result" or payload.get("subtype") != "success":
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Claude returned a non-success result",
                reason=SubscriptionFailureReason.EXECUTION_NONZERO,
            )
        text = payload.get("result")
        if not isinstance(text, str):
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Claude returned malformed structured output",
                reason=SubscriptionFailureReason.EXECUTION_MALFORMED_OUTPUT,
            )
        if not text.strip():
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Claude returned no assistant answer",
                reason=SubscriptionFailureReason.EXECUTION_EMPTY_ANSWER,
            )
        usage = _parse_usage(payload)
        model = payload.get("model")
        base = self.identity
        self._identity = ProviderIdentity(
            provider=base.provider,
            model=model if isinstance(model, str) else base.model,
            cli_version=base.cli_version,
            auth_kind=base.auth_kind,
        )
        answer = ProviderAnswer(text=text.strip(), usage=usage, identity=self._identity)
        self.last_answer = answer
        return answer

    def ask(self, prompt: str) -> str:
        return self.answer(prompt).text

    def _run(self, args: Sequence[str], stdin: str) -> subprocess.CompletedProcess[str]:
        if self.runner is run_provider_process:
            return run_provider_process(
                args,
                stdin,
                self.config.timeout_seconds,
                self.cancellation,
            )
        return self.runner(args, stdin, self.config.timeout_seconds)


def _json_object(value: str) -> dict[str, object] | None:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return cast(dict[str, object], payload)


def _parse_usage(payload: dict[str, object]) -> AnswerUsage:
    model_usage = payload.get("modelUsage")
    if not isinstance(model_usage, dict):
        usage = payload.get("usage")
        if isinstance(usage, dict):
            usage_dict = cast(dict[str, object], usage)
            return _native_usage(
                usage_dict.get("input_tokens"),
                usage_dict.get("output_tokens"),
            )
        return AnswerUsage(kind=UsageKind.UNAVAILABLE)
    input_tokens = 0
    output_tokens = 0
    saw_usage = False
    for value in cast(dict[str, object], model_usage).values():
        if not isinstance(value, dict):
            return AnswerUsage(kind=UsageKind.UNAVAILABLE)
        usage_dict = cast(dict[str, object], value)
        item = _native_usage(
            usage_dict.get("inputTokens"),
            usage_dict.get("outputTokens"),
        )
        if item.kind is not UsageKind.NATIVE:
            return AnswerUsage(kind=UsageKind.UNAVAILABLE)
        input_tokens += item.input_tokens or 0
        output_tokens += item.output_tokens or 0
        saw_usage = True
    if not saw_usage:
        return AnswerUsage(kind=UsageKind.UNAVAILABLE)
    return AnswerUsage(
        kind=UsageKind.NATIVE,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _native_usage(input_tokens: object, output_tokens: object) -> AnswerUsage:
    if (
        type(input_tokens) is int
        and input_tokens >= 0
        and type(output_tokens) is int
        and output_tokens >= 0
    ):
        return AnswerUsage(
            kind=UsageKind.NATIVE,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    return AnswerUsage(kind=UsageKind.UNAVAILABLE)
