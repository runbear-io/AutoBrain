"""Codex CLI adapter for the subscription provider protocol."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import cast

from autobrain.subscription_domain import (
    AnswerUsage,
    AuthKind,
    ProviderAnswer,
    ProviderCapability,
    ProviderConfig,
    ProviderId,
    ProviderIdentity,
    StructuredOutput,
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
class CodexSubscriptionConfig:
    command: str = "codex"
    model: str | None = None
    timeout_seconds: float = 120.0

    @classmethod
    def from_environ(cls) -> CodexSubscriptionConfig:
        timeout_raw = os.environ.get("AUTOBRAIN_SUBSCRIPTION_TIMEOUT_SECONDS", "120")
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise ValueError("AUTOBRAIN_SUBSCRIPTION_TIMEOUT_SECONDS must be a number") from exc
        if timeout_seconds <= 0:
            raise ValueError("AUTOBRAIN_SUBSCRIPTION_TIMEOUT_SECONDS must be greater than 0")
        return cls(
            command=os.environ.get("AUTOBRAIN_CODEX_COMMAND", "codex"),
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
class CodexSubscriptionClient:
    config: CodexSubscriptionConfig
    runner: Runner = run_provider_process
    interactive_runner: InteractiveRunner = run_interactive_provider_process
    last_answer: ProviderAnswer | None = field(default=None, init=False)
    _identity: ProviderIdentity | None = field(default=None, init=False)

    @property
    def identity(self) -> ProviderIdentity:
        if self._identity is not None:
            return self._identity
        return ProviderIdentity(
            provider=ProviderId.CODEX,
            model=self.config.model,
            cli_version=None,
            auth_kind=AuthKind.CONSUMER_SUBSCRIPTION,
        )

    def probe_identity(self) -> ProviderIdentity:
        try:
            result = self.runner(
                [self.config.command, "--version"],
                "",
                self.config.timeout_seconds,
            )
        except (
            subprocess.TimeoutExpired,
            ProviderProcessTimeout,
            ProviderProcessCancelled,
            OSError,
        ):
            return self.identity
        cli_version = sanitize_diagnostic(result.stdout) if result.returncode == 0 else ""
        base_identity = self.identity
        self._identity = ProviderIdentity(
            provider=base_identity.provider,
            model=base_identity.model,
            cli_version=cli_version or None,
            auth_kind=base_identity.auth_kind,
        )
        return self._identity

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

    def login(self) -> int:
        """Start the user-driven Codex/ChatGPT browser login flow."""
        if shutil.which(self.config.command) is None:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_CLI_UNAVAILABLE,
                f"Codex CLI not found: {self.config.command}",
                reason=SubscriptionFailureReason.LOGIN_UNAVAILABLE,
            )
        try:
            return self.interactive_runner([self.config.command, "login"])
        except ProviderProcessCancelled as exc:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Codex login cancelled",
                reason=SubscriptionFailureReason.EXECUTION_CANCELLED,
            ) from exc

    def probe_status(self) -> SubscriptionStatusReport:
        if shutil.which(self.config.command) is None:
            return SubscriptionStatusReport(
                status=SubscriptionStatus.SUBSCRIPTION_CLI_UNAVAILABLE,
                reason=SubscriptionFailureReason.LOGIN_UNAVAILABLE,
                detail=f"Codex CLI not found: {self.config.command}",
            )
        try:
            result = self.runner(
                [self.config.command, "login", "status"],
                "",
                self.config.timeout_seconds,
            )
        except (subprocess.TimeoutExpired, ProviderProcessTimeout):
            return SubscriptionStatusReport(
                status=SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                reason=SubscriptionFailureReason.STATUS_TIMEOUT,
                detail="Codex subscription status timed out",
            )
        except ProviderProcessCancelled:
            return SubscriptionStatusReport(
                status=SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                reason=SubscriptionFailureReason.STATUS_CANCELLED,
                detail="Codex subscription status cancelled",
            )
        output = f"{result.stdout}\n{result.stderr}".strip().lower()
        auth_markers = ("not logged", "logged out", "login required", "unauthenticated")
        if any(marker in output for marker in auth_markers):
            return SubscriptionStatusReport(
                status=SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE,
                reason=SubscriptionFailureReason.AUTH_UNAVAILABLE,
                detail="Run `codex login` with the ChatGPT account before using subscription mode",
            )
        if result.returncode != 0:
            return SubscriptionStatusReport(
                status=SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                reason=SubscriptionFailureReason.STATUS_NONZERO,
                detail=sanitize_diagnostic(result.stderr)
                or "Codex subscription status command failed",
            )
        if not any(marker in output for marker in ("logged in", "authenticated")):
            return SubscriptionStatusReport(
                status=SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                reason=SubscriptionFailureReason.STATUS_MALFORMED_OUTPUT,
                detail="Codex subscription status returned unrecognized output",
            )
        return SubscriptionStatusReport(status=SubscriptionStatus.READY)

    def status(self) -> SubscriptionStatus:
        """Compatibility mapping from detailed status reports to the legacy enum."""
        report = self.probe_status()
        if report.reason is SubscriptionFailureReason.STATUS_NONZERO:
            return SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE
        return report.status

    def ask_answer(self, prompt: str) -> ProviderAnswer:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        status_report = self.probe_status()
        if status_report.status is not SubscriptionStatus.READY:
            raise SubscriptionError(
                status_report.status,
                status_report.detail or self._status_detail(status_report.status),
                reason=status_report.reason,
            )

        args = [
            self.config.command,
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
        ]
        if self.config.model is not None:
            args.extend(["--model", self.config.model])
        try:
            # Prompt is deliberately stdin-only; it must never become argv data.
            result = self.runner(args, prompt, self.config.timeout_seconds)
        except (subprocess.TimeoutExpired, ProviderProcessTimeout) as exc:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Codex subscription execution timed out",
                reason=SubscriptionFailureReason.EXECUTION_TIMEOUT,
            ) from exc
        except ProviderProcessCancelled as exc:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Codex subscription execution cancelled",
                reason=SubscriptionFailureReason.EXECUTION_CANCELLED,
            ) from exc
        if result.returncode != 0:
            detail = sanitize_diagnostic(result.stderr) or "Codex subscription execution failed"
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                detail,
                reason=SubscriptionFailureReason.EXECUTION_NONZERO,
            )
        parsed = _parse_structured_output(result.stdout)
        if parsed is None:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Codex returned malformed structured output",
                reason=SubscriptionFailureReason.EXECUTION_MALFORMED_OUTPUT,
            )
        if not parsed.answer:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Codex returned no assistant answer",
                reason=SubscriptionFailureReason.EXECUTION_EMPTY_ANSWER,
            )
        base_identity = self.identity
        self._identity = ProviderIdentity(
            provider=base_identity.provider,
            model=base_identity.model,
            cli_version=parsed.cli_version or base_identity.cli_version,
            auth_kind=base_identity.auth_kind,
        )
        return ProviderAnswer(
            text=parsed.answer,
            usage=parsed.usage,
            identity=self._identity,
        )

    def answer(self, prompt: str) -> ProviderAnswer:
        self.last_answer = None
        result = self.ask_answer(prompt)
        self.last_answer = result
        return result

    def ask(self, prompt: str) -> str:
        """Legacy string-returning facade; use ``answer`` for typed usage."""
        return self.answer(prompt).text

    def _status_detail(self, status: SubscriptionStatus) -> str:
        if status is SubscriptionStatus.SUBSCRIPTION_CLI_UNAVAILABLE:
            return f"Codex CLI not found: {self.config.command}"
        if status is SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE:
            return "Run `codex login` with the ChatGPT account before using subscription mode"
        return status.value


_ALLOWED_EVENT_TYPES = {
    "error",
    "item.completed",
    "item.started",
    "item.updated",
    "message",
    "agent_message",
    "thread.started",
    "turn.completed",
    "turn.failed",
    "turn.started",
}


def _parse_structured_output(stdout: str) -> StructuredOutput | None:
    messages: list[str] = []
    usage = AnswerUsage(kind=UsageKind.UNAVAILABLE)
    cli_version: str | None = None
    saw_json = False
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event_value = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(event_value, dict):
            return None
        event = cast(dict[str, object], event_value)
        event_type = event.get("type")
        if not isinstance(event_type, str) or event_type not in _ALLOWED_EVENT_TYPES:
            return None
        saw_json = True
        version = event.get("version")
        if isinstance(version, str):
            cli_version = version
        usage_value = event.get("usage")
        if isinstance(usage_value, dict):
            usage_dict = cast(dict[str, object], usage_value)
            input_tokens = usage_dict.get("input_tokens")
            output_tokens = usage_dict.get("output_tokens")
            if (
                type(input_tokens) is int
                and input_tokens >= 0
                and type(output_tokens) is int
                and output_tokens >= 0
            ):
                usage = AnswerUsage(
                    kind=UsageKind.NATIVE,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
        item = event.get("item")
        if isinstance(item, dict):
            item_dict = cast(dict[str, object], item)
            text = item_dict.get("text")
            if item_dict.get("type") in {"agent_message", "message"} and isinstance(text, str):
                messages.append(text)
        text = event.get("text")
        if event.get("type") in {"agent_message", "message"} and isinstance(text, str):
            messages.append(text)
    if not saw_json:
        return None
    return StructuredOutput(
        answer="\n".join(messages).strip(),
        usage=usage,
        cli_version=cli_version,
    )
