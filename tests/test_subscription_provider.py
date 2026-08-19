from __future__ import annotations

import subprocess
from collections.abc import Sequence

import pytest

from autobrain.subscription import (
    AuthKind,
    CodexSubscriptionClient,
    CodexSubscriptionConfig,
    ProviderCapability,
    ProviderId,
    SubscriptionError,
    SubscriptionStatus,
    UsageKind,
)


def test_codex_adapter_exposes_typed_identity_capabilities_and_native_usage() -> None:
    calls: list[tuple[list[str], str]] = []

    def runner(
        args: Sequence[str],
        stdin: str,
        _timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((list(args), stdin))
        if list(args)[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(args, 0, "Logged in", "")
        return subprocess.CompletedProcess(
            args,
            0,
            (
                '{"type":"thread.started","version":"codex-cli 1.2.3"}\n'
                '{"type":"item.completed","item":{"type":"agent_message",'
                '"text":"subscription answer"},'
                '"usage":{"input_tokens":12,"output_tokens":3}}\n'
            ),
            "",
        )

    client = CodexSubscriptionClient(
        CodexSubscriptionConfig(model="gpt-5"),
        runner=runner,
    )

    answer = client.answer("untrusted ; $(touch /tmp/should-not-run)")

    assert client.identity.provider is ProviderId.CODEX
    assert client.identity.auth_kind is AuthKind.CONSUMER_SUBSCRIPTION
    assert ProviderCapability.READ_ONLY in client.capabilities
    assert answer.text == "subscription answer"
    assert answer.usage.kind is UsageKind.NATIVE
    assert answer.usage.input_tokens == 12
    assert answer.usage.output_tokens == 3
    assert answer.identity.cli_version == "codex-cli 1.2.3"
    assert client.ask("second prompt") == "subscription answer"
    assert client.last_answer is not None
    assert client.last_answer.identity.cli_version == "codex-cli 1.2.3"
    assert calls[1][1] == "untrusted ; $(touch /tmp/should-not-run)"
    assert "untrusted" not in calls[1][0]


def test_codex_adapter_rejects_malformed_structured_success() -> None:
    def runner(
        args: Sequence[str],
        _stdin: str,
        _timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        if list(args)[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(args, 0, "Logged in", "")
        return subprocess.CompletedProcess(args, 0, "not-json-but-exit-zero", "")

    client = CodexSubscriptionClient(CodexSubscriptionConfig(), runner=runner)

    with pytest.raises(SubscriptionError) as failure:
        client.answer("hello")

    assert failure.value.status is SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE
    assert failure.value.detail == "Codex returned malformed structured output"


def test_codex_adapter_preserves_timeout_as_typed_execution_failure() -> None:
    def runner(
        args: Sequence[str],
        _stdin: str,
        _timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        if list(args)[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(args, 0, "Logged in", "")
        raise subprocess.TimeoutExpired(args, 1)

    client = CodexSubscriptionClient(CodexSubscriptionConfig(), runner=runner)

    with pytest.raises(SubscriptionError) as failure:
        client.answer("hello")

    assert failure.value.status is SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE
    assert failure.value.detail == "Codex subscription execution timed out"


def test_codex_adapter_preserves_empty_answer_as_typed_execution_failure() -> None:
    def runner(
        args: Sequence[str],
        _stdin: str,
        _timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        if list(args)[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(args, 0, "Logged in", "")
        return subprocess.CompletedProcess(args, 0, '{"type":"turn.completed"}\n', "")

    client = CodexSubscriptionClient(CodexSubscriptionConfig(), runner=runner)

    with pytest.raises(SubscriptionError) as failure:
        client.answer("hello")

    assert failure.value.status is SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE
    assert failure.value.detail == "Codex returned no assistant answer"


def test_codex_adapter_bounds_and_redacts_nonzero_diagnostics() -> None:
    def runner(
        args: Sequence[str],
        _stdin: str,
        _timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        if list(args)[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(args, 0, "Logged in", "")
        return subprocess.CompletedProcess(args, 1, "", "Bearer secret-token-value " + "x" * 5000)

    client = CodexSubscriptionClient(CodexSubscriptionConfig(), runner=runner)

    with pytest.raises(SubscriptionError) as failure:
        client.answer("hello")

    assert "secret-token-value" not in failure.value.detail
    assert "[REDACTED]" in failure.value.detail
    assert len(failure.value.detail) <= 2048
