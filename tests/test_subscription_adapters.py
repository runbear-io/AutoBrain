from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autobrain.auth.models import Provider
from autobrain.cli import app
from autobrain.embedding import EmbeddingBackendConfig
from autobrain.experiment import build_automatic_plan
from autobrain.models import CandidateId, ChatProvenance
from autobrain.orchestration import RunConfig, RunOrchestrator
from autobrain.subscription import (
    AuthKind,
    ClaudeSubscriptionClient,
    ClaudeSubscriptionConfig,
    ProviderId,
    SubscriptionError,
    SubscriptionFailureReason,
    SubscriptionProviderRegistry,
    SubscriptionStatus,
    SubscriptionStatusReport,
    UsageKind,
    provider_registry,
)
from autobrain.subscription_process import ProviderProcessCancelled, ProviderProcessTimeout


def _completed(
    args: Sequence[str], payload: object, *, returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, json.dumps(payload), "")


def _available(_command: str) -> str:
    return "/claude"


def test_registry_exposes_explicit_vendor_set_and_never_falls_back() -> None:
    registry = provider_registry()

    assert tuple(registry.provider_ids) == (
        ProviderId.CODEX,
        ProviderId.CLAUDE,
        ProviderId.KIMI,
        ProviderId.GROK,
    )
    assert registry.get(ProviderId.CLAUDE).identity.provider is ProviderId.CLAUDE
    assert registry.get(ProviderId.KIMI).probe_status().status is SubscriptionStatus.UNSUPPORTED
    assert "verified official" in registry.get(ProviderId.KIMI).probe_status().detail
    assert registry.get(ProviderId.GROK).probe_status().status is SubscriptionStatus.UNSUPPORTED


def test_claude_status_requires_first_party_consumer_oauth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("autobrain.subscription_claude.shutil.which", _available)
    outcomes = [
        {"loggedIn": True, "authMethod": "oauth_token", "apiProvider": "firstParty"},
        {"loggedIn": False, "authMethod": "none", "apiProvider": "firstParty"},
        {"loggedIn": True, "authMethod": "api_key", "apiProvider": "firstParty"},
        {"loggedIn": True, "authMethod": "oauth_token", "apiProvider": "bedrock"},
        {"loggedIn": True, "authMethod": "oauth_token"},
    ]

    def runner(
        args: Sequence[str], _stdin: str, _timeout: float
    ) -> subprocess.CompletedProcess[str]:
        assert list(args) == ["claude", "auth", "status", "--json"]
        return _completed(args, outcomes.pop(0))

    client = ClaudeSubscriptionClient(ClaudeSubscriptionConfig(), runner=runner)

    assert client.probe_status().status is SubscriptionStatus.READY
    assert client.probe_status().reason is SubscriptionFailureReason.AUTH_UNAVAILABLE
    assert client.probe_status().reason is SubscriptionFailureReason.AUTH_KIND_UNSUPPORTED
    assert client.probe_status().reason is SubscriptionFailureReason.AUTH_KIND_UNSUPPORTED
    assert client.probe_status().reason is SubscriptionFailureReason.STATUS_MALFORMED_OUTPUT


def test_claude_login_is_interactive_consumer_login_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("autobrain.subscription_claude.shutil.which", _available)
    calls: list[list[str]] = []

    client = ClaudeSubscriptionClient(
        ClaudeSubscriptionConfig(),
        interactive_runner=lambda args: calls.append(list(args)) or 0,
    )

    assert client.login() == 0
    assert calls == [["claude", "auth", "login", "--claudeai"]]


def test_claude_answer_is_tool_free_safe_json_stdin_only_and_reports_native_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("autobrain.subscription_claude.shutil.which", _available)
    calls: list[tuple[list[str], str]] = []

    def runner(
        args: Sequence[str], stdin: str, _timeout: float
    ) -> subprocess.CompletedProcess[str]:
        calls.append((list(args), stdin))
        if list(args)[1:3] == ["auth", "status"]:
            return _completed(
                args,
                {"loggedIn": True, "authMethod": "oauth_token", "apiProvider": "firstParty"},
            )
        return _completed(
            args,
            {
                "type": "result",
                "subtype": "success",
                "result": "safe answer",
                "modelUsage": {
                    "claude-sonnet": {
                        "inputTokens": 11,
                        "outputTokens": 4,
                    }
                },
            },
        )

    client = ClaudeSubscriptionClient(ClaudeSubscriptionConfig(model="sonnet"), runner=runner)
    prompt = "hostile ; $(touch /tmp/autobrain-claude-injection)"
    answer = client.answer(prompt)

    execution_argv, execution_stdin = calls[1]
    assert execution_stdin == prompt
    assert prompt not in execution_argv
    assert execution_argv == [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--tools",
        "",
        "--safe-mode",
        "--no-session-persistence",
        "--permission-mode",
        "dontAsk",
        "--model",
        "sonnet",
    ]
    assert answer.text == "safe answer"
    assert answer.usage.kind is UsageKind.NATIVE
    assert answer.usage.input_tokens == 11
    assert answer.usage.output_tokens == 4
    assert answer.identity.provider is ProviderId.CLAUDE
    assert answer.identity.auth_kind is AuthKind.CONSUMER_SUBSCRIPTION


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (
            {"type": "result", "subtype": "success", "result": ""},
            SubscriptionFailureReason.EXECUTION_EMPTY_ANSWER,
        ),
        (
            {"type": "result", "subtype": "error", "result": "no"},
            SubscriptionFailureReason.EXECUTION_NONZERO,
        ),
        (
            {
                "type": "result",
                "subtype": "success",
                "result": "ok",
                "modelUsage": {"m": {"inputTokens": True, "outputTokens": 1}},
            },
            None,
        ),
        ([{"result": "misleading READY"}], SubscriptionFailureReason.EXECUTION_MALFORMED_OUTPUT),
    ],
)
def test_claude_rejects_malformed_empty_and_misleading_json(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    reason: SubscriptionFailureReason | None,
) -> None:
    monkeypatch.setattr("autobrain.subscription_claude.shutil.which", _available)

    def runner(
        args: Sequence[str], _stdin: str, _timeout: float
    ) -> subprocess.CompletedProcess[str]:
        if list(args)[1:3] == ["auth", "status"]:
            return _completed(
                args,
                {"loggedIn": True, "authMethod": "oauth_token", "apiProvider": "firstParty"},
            )
        return _completed(args, payload)

    client = ClaudeSubscriptionClient(ClaudeSubscriptionConfig(), runner=runner)
    if reason is None:
        assert client.answer("hello").usage.kind is UsageKind.UNAVAILABLE
    else:
        with pytest.raises(SubscriptionError) as failure:
            client.answer("hello")
        assert failure.value.reason is reason


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (ProviderProcessTimeout("hung descendant"), SubscriptionFailureReason.EXECUTION_TIMEOUT),
        (ProviderProcessCancelled("cancelled"), SubscriptionFailureReason.EXECUTION_CANCELLED),
        (
            subprocess.CompletedProcess(["claude"], 9, "", "failed"),
            SubscriptionFailureReason.EXECUTION_NONZERO,
        ),
    ],
)
def test_claude_preserves_timeout_cancel_and_nonzero_execution(
    monkeypatch: pytest.MonkeyPatch,
    outcome: BaseException | subprocess.CompletedProcess[str],
    reason: SubscriptionFailureReason,
) -> None:
    monkeypatch.setattr("autobrain.subscription_claude.shutil.which", _available)

    def runner(
        args: Sequence[str], _stdin: str, _timeout: float
    ) -> subprocess.CompletedProcess[str]:
        if list(args)[1:3] == ["auth", "status"]:
            return _completed(
                args,
                {"loggedIn": True, "authMethod": "oauth_token", "apiProvider": "firstParty"},
            )
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    with pytest.raises(SubscriptionError) as failure:
        ClaudeSubscriptionClient(ClaudeSubscriptionConfig(), runner=runner).answer("hello")

    assert failure.value.reason is reason


def test_claude_selection_is_explicit_in_plan_and_provenance() -> None:
    embedding_readiness = EmbeddingBackendConfig.from_environ(
        {"OPENAI_API_KEY": "fixture-embedding-key"},
        requested="openai",
    ).readiness()
    plan = build_automatic_plan(
        sources=(Provider.SLACK,),
        candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
        subscription_status=SubscriptionStatus.READY,
        embedding_readiness=embedding_readiness,
        subscription_provider=ProviderId.CLAUDE,
    )
    config = RunConfig(provider_mode="claude-subscription")
    orchestrator = RunOrchestrator(
        config=config,
        connectors=(),
        candidates=(),
        provider_available=False,
        chat_provenance_provider=lambda: ChatProvenance(
            provider="claude",
            model="sonnet",
            cli_version="2.1.235",
            auth_kind="consumer_subscription",
        ),
    )

    assert plan.provider_mode == "claude-subscription"
    assert config.provider_mode == "claude-subscription"
    assert orchestrator.benchmark_provenance().chat.provider == "claude"


def test_pre_run_reprobes_exact_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = provider_registry()
    refreshed: list[ProviderId] = []
    original_probe = registry.probe

    def probe(
        provider: ProviderId | str,
        *,
        refresh: bool = False,
    ) -> SubscriptionStatusReport:
        provider_id = provider if isinstance(provider, ProviderId) else ProviderId(provider)
        if refresh:
            refreshed.append(provider_id)
        if provider_id is ProviderId.CLAUDE:
            return SubscriptionStatusReport(status=SubscriptionStatus.READY)
        return original_probe(provider_id, refresh=refresh)

    registry.probe = probe  # type: ignore[method-assign]
    claude = registry.get(ProviderId.CLAUDE)
    claude.probe_identity = lambda: claude.identity  # type: ignore[method-assign]

    def selected_registry(**_kwargs: object) -> SubscriptionProviderRegistry:
        return registry

    monkeypatch.setattr("autobrain.subscription.provider_registry", selected_registry)

    orchestrator = RunOrchestrator.local(
        RunConfig(
            provider_mode="claude-subscription",
            selected_sources=(Provider.SLACK,),
            selected_candidates=(CandidateId.LLM_WIKI, CandidateId.MEM0),
            output=tmp_path / "runs",
            slack_export_path=tmp_path / "slack.zip",
            slack_export_sha256="a" * 64,
        ),
        connector_builder=lambda _manager, _include_dms: (),
        candidate_builder=lambda *_args, **_kwargs: (),
    )

    assert refreshed == [ProviderId.CLAUDE]
    assert orchestrator.provider_available is True


def test_registry_probe_cache_is_bounded_non_overlapping_and_refreshable() -> None:
    registry = provider_registry(status_ttl_seconds=60.0)
    client = registry.get(ProviderId.CLAUDE)
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def slow_probe() -> object:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=2)
        return SubscriptionStatusReport(status=SubscriptionStatus.READY)

    client.probe_status = slow_probe  # type: ignore[method-assign]
    result: list[object] = []
    thread = threading.Thread(
        target=lambda: result.append(registry.probe(ProviderId.CLAUDE, refresh=True))
    )
    thread.start()
    assert entered.wait(timeout=1)

    overlap = registry.probe(ProviderId.CLAUDE, refresh=True)
    assert overlap.reason is SubscriptionFailureReason.PROBE_IN_PROGRESS
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert registry.probe(ProviderId.CLAUDE).status is SubscriptionStatus.READY
    assert calls == 1


def test_subscription_cli_requires_explicit_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = provider_registry()
    monkeypatch.setattr("autobrain.cli.provider_registry", lambda: registry)

    kimi = CliRunner().invoke(app, ["subscription", "status", "--provider", "kimi", "--json"])
    grok_ask = CliRunner().invoke(app, ["subscription", "ask", "hello", "--provider", "grok"])

    assert kimi.exit_code == 0
    assert json.loads(kimi.stdout)["status"] == SubscriptionStatus.UNSUPPORTED.value
    assert grok_ask.exit_code == 1
    assert SubscriptionStatus.UNSUPPORTED.value in grok_ask.stderr


def test_subscription_setup_does_not_start_login_without_command_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = provider_registry()
    client = registry.get(ProviderId.CLAUDE)
    called = False

    def login() -> int:
        nonlocal called
        called = True
        return 0

    client.login = login  # type: ignore[method-assign]
    monkeypatch.setattr("autobrain.cli.provider_registry", lambda: registry)

    status = CliRunner().invoke(app, ["subscription", "status", "--provider", "claude", "--json"])

    assert status.exit_code == 0
    assert called is False
