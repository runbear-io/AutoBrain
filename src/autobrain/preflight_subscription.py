"""Selected subscription-provider readiness check for AutoBrain doctor."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

from autobrain.models import CheckResult, Status
from autobrain.preflight_support import CommandResult
from autobrain.subscription import ProviderId, SubscriptionStatus, provider_registry


def check_subscription_provider(
    provider: ProviderId,
    *,
    command_runner: Callable[[tuple[str, ...], float], CommandResult],
    executable_finder: Callable[[str], str | None],
) -> CheckResult:
    check_name = (
        "chatgpt_subscription" if provider is ProviderId.CODEX else f"{provider.value}_subscription"
    )
    if provider in {ProviderId.KIMI, ProviderId.GROK}:
        report = provider_registry().probe(provider, refresh=True)
        return CheckResult(
            name=check_name,
            status=Status.UNSUPPORTED,
            detail=f"{report.status.value}: {report.detail}",
        )
    executable_name = provider.value
    executable = executable_finder(executable_name)
    if executable is None:
        return CheckResult(
            name=check_name,
            status=Status.MISSING_PROVIDER,
            detail=(
                f"{SubscriptionStatus.SUBSCRIPTION_CLI_UNAVAILABLE.value}: "
                f"{provider.value} CLI not found"
            ),
        )
    command = (
        (executable, "login", "status")
        if provider is ProviderId.CODEX
        else (executable, "auth", "status", "--json")
    )
    try:
        result = command_runner(command, 3.0)
    except (OSError, RuntimeError, TimeoutError) as error:
        return CheckResult(
            name=check_name,
            status=Status.MISSING_PROVIDER,
            detail=(f"{SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE.value}: {error}"),
            path=executable,
        )
    if provider is ProviderId.CLAUDE:
        return _claude_check(result, executable)
    output = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode != 0 or any(
        marker in output
        for marker in ("not logged", "logged out", "login required", "unauthenticated")
    ):
        return CheckResult(
            name=check_name,
            status=Status.MISSING_PROVIDER,
            detail=(
                f"{SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE.value}: "
                "run `autobrain subscription setup --provider codex`"
            ),
            path=executable,
        )
    return CheckResult(
        name=check_name,
        status=Status.OK,
        detail=SubscriptionStatus.READY.value,
        path=executable,
    )


def check_chatgpt_subscription(
    *,
    command_runner: Callable[[tuple[str, ...], float], CommandResult],
    executable_finder: Callable[[str], str | None],
) -> CheckResult:
    """One-release compatibility alias for the Codex doctor check."""
    result = check_subscription_provider(
        ProviderId.CODEX,
        command_runner=command_runner,
        executable_finder=executable_finder,
    )
    return result.model_copy(update={"name": "chatgpt_subscription"})


def _claude_check(result: CommandResult, executable: str) -> CheckResult:
    name = "claude_subscription"
    if result.returncode != 0:
        return CheckResult(
            name=name,
            status=Status.MISSING_PROVIDER,
            detail=(
                f"{SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE.value}: "
                "Claude status command failed"
            ),
            path=executable,
        )
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError:
        raw = None
    if not isinstance(raw, dict):
        return CheckResult(
            name=name,
            status=Status.MISSING_PROVIDER,
            detail=(
                f"{SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE.value}: "
                "Claude status returned malformed JSON"
            ),
            path=executable,
        )
    payload = cast(dict[str, object], raw)
    if payload.get("loggedIn") is not True:
        return CheckResult(
            name=name,
            status=Status.MISSING_PROVIDER,
            detail=(
                f"{SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE.value}: "
                "run `autobrain subscription setup --provider claude`"
            ),
            path=executable,
        )
    if payload.get("authMethod") != "oauth_token" or payload.get("apiProvider") != "firstParty":
        return CheckResult(
            name=name,
            status=Status.MISSING_PROVIDER,
            detail=(
                f"{SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE.value}: "
                "Claude must use first-party consumer OAuth, not API-key/enterprise/proxy auth"
            ),
            path=executable,
        )
    return CheckResult(
        name=name,
        status=Status.OK,
        detail=SubscriptionStatus.READY.value,
        path=executable,
    )
