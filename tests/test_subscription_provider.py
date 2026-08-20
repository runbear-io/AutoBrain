from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence

import pytest

from autobrain.subscription import (
    AuthKind,
    CodexSubscriptionClient,
    CodexSubscriptionConfig,
    ProviderCapability,
    ProviderId,
    ProviderIdentity,
    SubscriptionError,
    SubscriptionFailureReason,
    SubscriptionStatus,
    UsageKind,
)
from autobrain.subscription_process import ProviderProcessCancelled


def test_codex_status_probe_distinguishes_public_failure_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def available(_command: str) -> str:
        return "/codex"

    monkeypatch.setattr("autobrain.subscription_codex.shutil.which", available)
    outcomes: list[subprocess.CompletedProcess[str] | BaseException] = [
        subprocess.TimeoutExpired(["codex"], 1),
        subprocess.CompletedProcess(["codex"], 7, "", "unexpected failure"),
        subprocess.CompletedProcess(["codex"], 0, "ambiguous success", ""),
        subprocess.CompletedProcess(["codex"], 1, "", "login required"),
    ]

    def runner(
        _args: Sequence[str],
        _stdin: str,
        _timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, subprocess.CompletedProcess)
        return outcome

    client = CodexSubscriptionClient(CodexSubscriptionConfig(), runner=runner)

    assert client.probe_status().reason is SubscriptionFailureReason.STATUS_TIMEOUT
    assert client.probe_status().reason is SubscriptionFailureReason.STATUS_NONZERO
    assert client.probe_status().reason is SubscriptionFailureReason.STATUS_MALFORMED_OUTPUT
    assert client.probe_status().reason is SubscriptionFailureReason.AUTH_UNAVAILABLE


def test_legacy_status_maps_generic_nonzero_to_auth_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def available(_command: str) -> str:
        return "/codex"

    monkeypatch.setattr("autobrain.subscription_codex.shutil.which", available)

    def runner(
        args: Sequence[str],
        _stdin: str,
        _timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 7, "", "generic failure")

    client = CodexSubscriptionClient(CodexSubscriptionConfig(), runner=runner)

    assert client.status() is SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE


def test_codex_missing_cli_has_login_unavailable_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_command: str) -> None:
        return None

    monkeypatch.setattr("autobrain.subscription_codex.shutil.which", unavailable)

    report = CodexSubscriptionClient(CodexSubscriptionConfig()).probe_status()

    assert report.status is SubscriptionStatus.SUBSCRIPTION_CLI_UNAVAILABLE
    assert report.reason is SubscriptionFailureReason.LOGIN_UNAVAILABLE


def test_codex_adapter_probes_cli_version_for_persisted_identity() -> None:
    def runner(
        args: Sequence[str],
        _stdin: str,
        _timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        assert list(args) == ["codex", "--version"]
        return subprocess.CompletedProcess(args, 0, "codex-cli 1.2.3\n", "")

    client = CodexSubscriptionClient(
        CodexSubscriptionConfig(model="gpt-5"),
        runner=runner,
    )

    assert client.probe_identity() == ProviderIdentity(
        provider=ProviderId.CODEX,
        model="gpt-5",
        cli_version="codex-cli 1.2.3",
        auth_kind=AuthKind.CONSUMER_SUBSCRIPTION,
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


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [
        (True, 1),
        (1, False),
        (1.5, 2),
        (1, 2.5),
        ("1", 2),
        (1, "2"),
        (-1, 2),
        (1, -2),
    ],
)
def test_codex_usage_rejects_non_integer_or_negative_token_counts(
    input_tokens: object,
    output_tokens: object,
) -> None:
    def runner(
        args: Sequence[str],
        _stdin: str,
        _timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        if list(args)[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(args, 0, "Logged in", "")
        event = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "answer"},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }
        return subprocess.CompletedProcess(args, 0, json.dumps(event) + "\n", "")

    answer = CodexSubscriptionClient(CodexSubscriptionConfig(), runner=runner).answer("hello")

    assert answer.usage.kind is UsageKind.UNAVAILABLE
    assert answer.usage.input_tokens is None
    assert answer.usage.output_tokens is None


@pytest.mark.parametrize(("input_tokens", "output_tokens"), [(0, 0), (12, 3)])
def test_codex_usage_accepts_exact_nonnegative_json_integers(
    input_tokens: int,
    output_tokens: int,
) -> None:
    def runner(
        args: Sequence[str],
        _stdin: str,
        _timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        if list(args)[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(args, 0, "Logged in", "")
        event = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "answer"},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }
        return subprocess.CompletedProcess(args, 0, json.dumps(event) + "\n", "")

    answer = CodexSubscriptionClient(CodexSubscriptionConfig(), runner=runner).answer("hello")

    assert answer.usage.kind is UsageKind.NATIVE
    assert answer.usage.input_tokens == input_tokens
    assert answer.usage.output_tokens == output_tokens


@pytest.mark.parametrize(
    "stdout",
    [
        'preface\n{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\n',
        '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\ntrailing',
        '[]\n{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\n',
        '{"type":"unknown.event","text":"ok"}\n',
    ],
)
def test_codex_adapter_rejects_mixed_or_unknown_structured_output(stdout: str) -> None:
    def runner(
        args: Sequence[str],
        _stdin: str,
        _timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        if list(args)[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(args, 0, "Logged in", "")
        return subprocess.CompletedProcess(args, 0, stdout, "")

    client = CodexSubscriptionClient(CodexSubscriptionConfig(), runner=runner)

    with pytest.raises(SubscriptionError) as failure:
        client.answer("hello")

    assert failure.value.reason is SubscriptionFailureReason.EXECUTION_MALFORMED_OUTPUT


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
    assert failure.value.reason is SubscriptionFailureReason.EXECUTION_MALFORMED_OUTPUT
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
    assert failure.value.reason is SubscriptionFailureReason.EXECUTION_TIMEOUT
    assert failure.value.detail == "Codex subscription execution timed out"


def test_codex_adapter_preserves_cancellation_as_typed_execution_failure() -> None:
    def runner(
        args: Sequence[str],
        _stdin: str,
        _timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        if list(args)[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(args, 0, "Logged in", "")
        raise ProviderProcessCancelled("cancelled")

    client = CodexSubscriptionClient(CodexSubscriptionConfig(), runner=runner)

    with pytest.raises(SubscriptionError) as failure:
        client.answer("hello")

    assert failure.value.reason is SubscriptionFailureReason.EXECUTION_CANCELLED


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
    assert failure.value.reason is SubscriptionFailureReason.EXECUTION_EMPTY_ANSWER
    assert failure.value.detail == "Codex returned no assistant answer"


def test_codex_adapter_clears_stale_last_answer_before_repeated_failure() -> None:
    executions = 0

    def runner(
        args: Sequence[str],
        _stdin: str,
        _timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal executions
        if list(args)[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(args, 0, "Logged in", "")
        executions += 1
        if executions == 1:
            return subprocess.CompletedProcess(
                args,
                0,
                '{"type":"item.completed","item":{"type":"agent_message","text":"first"}}\n',
                "",
            )
        raise subprocess.TimeoutExpired(args, 1)

    client = CodexSubscriptionClient(CodexSubscriptionConfig(), runner=runner)
    assert client.ask("first") == "first"
    assert client.last_answer is not None

    with pytest.raises(SubscriptionError):
        client.ask("second")

    assert client.last_answer is None


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

    assert failure.value.reason is SubscriptionFailureReason.EXECUTION_NONZERO
    assert "secret-token-value" not in failure.value.detail
    assert "[REDACTED]" in failure.value.detail
    assert len(failure.value.detail) <= 2048
