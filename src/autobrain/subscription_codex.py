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
    SubscriptionStatus,
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

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            provider=ProviderId.CODEX,
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

    def login(self) -> int:
        """Start the user-driven Codex/ChatGPT browser login flow."""
        if shutil.which(self.config.command) is None:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_CLI_UNAVAILABLE,
                f"Codex CLI not found: {self.config.command}",
            )
        try:
            return self.interactive_runner([self.config.command, "login"])
        except ProviderProcessCancelled as exc:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Codex login cancelled",
            ) from exc

    def status(self) -> SubscriptionStatus:
        if shutil.which(self.config.command) is None:
            return SubscriptionStatus.SUBSCRIPTION_CLI_UNAVAILABLE
        try:
            result = self.runner(
                [self.config.command, "login", "status"],
                "",
                self.config.timeout_seconds,
            )
        except (subprocess.TimeoutExpired, ProviderProcessTimeout, ProviderProcessCancelled):
            return SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE
        output = f"{result.stdout}\n{result.stderr}".lower()
        if result.returncode != 0 or any(
            marker in output
            for marker in ("not logged", "logged out", "login required", "unauthenticated")
        ):
            return SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE
        return SubscriptionStatus.READY

    def ask_answer(self, prompt: str) -> ProviderAnswer:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        status = self.status()
        if status is not SubscriptionStatus.READY:
            raise SubscriptionError(status, self._status_detail(status))

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
            ) from exc
        except ProviderProcessCancelled as exc:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Codex subscription execution cancelled",
            ) from exc
        if result.returncode != 0:
            detail = sanitize_diagnostic(result.stderr) or "Codex subscription execution failed"
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                detail,
            )
        parsed = _parse_structured_output(result.stdout)
        if parsed is None:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Codex returned malformed structured output",
            )
        if not parsed.answer:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Codex returned no assistant answer",
            )
        return ProviderAnswer(
            text=parsed.answer,
            usage=parsed.usage,
            identity=ProviderIdentity(
                provider=self.identity.provider,
                model=self.identity.model,
                cli_version=parsed.cli_version,
                auth_kind=self.identity.auth_kind,
            ),
        )

    def answer(self, prompt: str) -> ProviderAnswer:
        return self.ask_answer(prompt)

    def ask(self, prompt: str) -> str:
        """Legacy string-returning facade; use ``answer`` for typed usage."""
        result = self.answer(prompt)
        self.last_answer = result
        return result.text

    def _status_detail(self, status: SubscriptionStatus) -> str:
        if status is SubscriptionStatus.SUBSCRIPTION_CLI_UNAVAILABLE:
            return f"Codex CLI not found: {self.config.command}"
        if status is SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE:
            return "Run `codex login` with the ChatGPT account before using subscription mode"
        return status.value


def _parse_structured_output(stdout: str) -> StructuredOutput | None:
    messages: list[str] = []
    usage = AnswerUsage(kind=UsageKind.UNAVAILABLE)
    cli_version: str | None = None
    saw_json = False
    for line in stdout.splitlines():
        try:
            event_value = json.loads(line)
        except json.JSONDecodeError:
            continue
        saw_json = True
        if not isinstance(event_value, dict):
            continue
        event = cast(dict[str, object], event_value)
        version = event.get("version")
        if isinstance(version, str):
            cli_version = version
        usage_value = event.get("usage")
        if isinstance(usage_value, dict):
            usage_dict = cast(dict[str, object], usage_value)
            input_tokens = usage_dict.get("input_tokens")
            output_tokens = usage_dict.get("output_tokens")
            if isinstance(input_tokens, int) and isinstance(output_tokens, int):
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
