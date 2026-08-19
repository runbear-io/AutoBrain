"""Local subscription-backed execution through the Codex CLI."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast


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


Runner = Callable[
    [Sequence[str], str, float],
    subprocess.CompletedProcess[str],
]


def _run_command(
    args: Sequence[str],
    stdin: str,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        input=stdin,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout_seconds,
    )


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


@dataclass
class CodexSubscriptionClient:
    config: CodexSubscriptionConfig
    runner: Runner = _run_command

    def login(self) -> int:
        """Start the user-driven Codex/ChatGPT browser login flow."""
        if shutil.which(self.config.command) is None:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_CLI_UNAVAILABLE,
                f"Codex CLI not found: {self.config.command}",
            )
        return subprocess.run(
            [self.config.command, "login"],
            check=False,
        ).returncode

    def status(self) -> SubscriptionStatus:
        if shutil.which(self.config.command) is None:
            return SubscriptionStatus.SUBSCRIPTION_CLI_UNAVAILABLE
        result = self.runner(
            [self.config.command, "login", "status"],
            "",
            self.config.timeout_seconds,
        )
        output = f"{result.stdout}\n{result.stderr}".lower()
        if result.returncode != 0 or any(
            marker in output
            for marker in ("not logged", "logged out", "login required", "unauthenticated")
        ):
            return SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE
        return SubscriptionStatus.READY

    def ask(self, prompt: str) -> str:
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
        args.append(prompt)
        try:
            result = self.runner(args, "", self.config.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Codex subscription execution timed out",
            ) from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or "Codex subscription execution failed"
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                detail,
            )
        answer = _extract_answer(result.stdout)
        if not answer:
            raise SubscriptionError(
                SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                "Codex returned no assistant answer",
            )
        return answer

    def _status_detail(self, status: SubscriptionStatus) -> str:
        if status is SubscriptionStatus.SUBSCRIPTION_CLI_UNAVAILABLE:
            return f"Codex CLI not found: {self.config.command}"
        if status is SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE:
            return "Run `codex login` with the ChatGPT account before using subscription mode"
        return status.value


def _extract_answer(stdout: str) -> str:
    messages: list[str] = []
    for line in stdout.splitlines():
        try:
            event_value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event_value, dict):
            continue
        event = cast(dict[str, object], event_value)
        item = event.get("item")
        if isinstance(item, dict):
            item_dict = cast(dict[str, object], item)
            text = item_dict.get("text")
            if item_dict.get("type") in {"agent_message", "message"} and isinstance(text, str):
                messages.append(text)
        text = event.get("text")
        if event.get("type") in {"agent_message", "message"} and isinstance(text, str):
            messages.append(text)
    if messages:
        return "\n".join(messages).strip()
    return stdout.strip()


def local_embedding(text: str, *, dimensions: int = 1536) -> list[float]:
    """Create a deterministic local embedding without provider billing."""
    vector = [0.0] * dimensions
    tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [round(value / norm, 8) for value in vector]


def build_subscription_upstream(
    client: CodexSubscriptionClient,
) -> Callable[[dict[str, object]], dict[str, object]]:
    """Build an OpenAI-compatible upstream backed by subscription chat and local vectors."""

    def upstream(payload: dict[str, object]) -> dict[str, object]:
        model = payload.get("model")
        if isinstance(model, str) and model.startswith("text-embedding"):
            inputs = payload.get("input")
            if isinstance(inputs, str):
                texts = [inputs]
            elif isinstance(inputs, list):
                input_items = cast(list[object], inputs)
                texts = [item for item in input_items if isinstance(item, str)]
            else:
                texts = []
            return {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": local_embedding(text),
                    }
                    for index, text in enumerate(texts)
                ],
                "model": "local-hash-embedding",
                "usage": {
                    "prompt_tokens": sum(_estimate_tokens(text) for text in texts),
                    "total_tokens": sum(_estimate_tokens(text) for text in texts),
                },
            }

        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("subscription chat payload requires messages")
        message_items = cast(list[object], messages)
        message_dicts = [
            cast(dict[str, object], message)
            for message in message_items
            if isinstance(message, dict)
        ]
        prompt_parts = [
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in message_dicts
        ]
        answer = client.ask("\n".join(prompt_parts))
        return {
            "id": "subscription-chat-completion",
            "object": "chat.completion",
            "model": "chatgpt-subscription",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": _estimate_tokens("\n".join(prompt_parts)),
                "completion_tokens": _estimate_tokens(answer),
            },
        }

    return upstream


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))
