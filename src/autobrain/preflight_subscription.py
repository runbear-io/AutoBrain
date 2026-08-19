"""ChatGPT subscription readiness check for AutoBrain doctor."""

from __future__ import annotations

from collections.abc import Callable

from autobrain.models import CheckResult, Status
from autobrain.preflight_support import CommandResult
from autobrain.subscription import SubscriptionStatus


def check_chatgpt_subscription(
    *,
    command_runner: Callable[[tuple[str, ...], float], CommandResult],
    executable_finder: Callable[[str], str | None],
) -> CheckResult:
    executable = executable_finder("codex")
    if executable is None:
        return CheckResult(
            name="chatgpt_subscription",
            status=Status.MISSING_PROVIDER,
            detail=(
                f"{SubscriptionStatus.SUBSCRIPTION_CLI_UNAVAILABLE.value}: Codex CLI not found"
            ),
        )
    try:
        result = command_runner((executable, "login", "status"), 3.0)
    except (OSError, RuntimeError, TimeoutError) as error:
        return CheckResult(
            name="chatgpt_subscription",
            status=Status.MISSING_PROVIDER,
            detail=(f"{SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE.value}: {error}"),
            path=executable,
        )
    output = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode != 0 or any(
        marker in output
        for marker in ("not logged", "logged out", "login required", "unauthenticated")
    ):
        return CheckResult(
            name="chatgpt_subscription",
            status=Status.MISSING_PROVIDER,
            detail=(
                f"{SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE.value}: "
                "run `autobrain subscription setup`"
            ),
            path=executable,
        )
    return CheckResult(
        name="chatgpt_subscription",
        status=Status.OK,
        detail=SubscriptionStatus.READY.value,
        path=executable,
    )
