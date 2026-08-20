from __future__ import annotations

import subprocess
from collections.abc import Sequence
from typing import cast

import pytest

from autobrain.metering import LoopbackMeteringProxy
from autobrain.orchestration import RunConfig
from autobrain.subscription import (
    CodexSubscriptionClient,
    CodexSubscriptionConfig,
    SubscriptionStatus,
    build_subscription_upstream,
    local_embedding,
)


def test_subscription_status_reports_auth_unavailable_without_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def which(_command: str) -> str:
        return "/usr/local/bin/codex"

    monkeypatch.setattr("autobrain.subscription.shutil.which", which)

    def runner(
        args: Sequence[str],
        _stdin: str,
        _timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        assert list(args) == ["codex", "login", "status"]
        return subprocess.CompletedProcess(args, 1, "", "Not logged in")

    client = CodexSubscriptionClient(CodexSubscriptionConfig(), runner=runner)

    assert client.status() is SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE


def test_local_embedding_is_deterministic_and_normalized() -> None:
    first = local_embedding("same document")
    second = local_embedding("same document")

    assert first == second
    assert len(first) == 1536
    assert max(abs(value) for value in first) <= 1
    assert any(value != 0 for value in first)


def test_subscription_upstream_handles_local_embeddings_without_codex() -> None:
    client = CodexSubscriptionClient(CodexSubscriptionConfig())
    upstream = build_subscription_upstream(client)

    response = upstream(
        {
            "model": "text-embedding-3-small",
            "input": ["first document", "second document"],
        }
    )

    assert response["model"] == "local-hash-embedding"
    data = cast(list[dict[str, object]], response["data"])
    assert len(data) == 2
    assert len(cast(list[float], data[0]["embedding"])) == 1536


def test_subscription_chat_can_use_an_explicit_semantic_embedding_backend() -> None:
    embedding_calls: list[dict[str, object]] = []
    chat_calls: list[str] = []

    class FakeClient(CodexSubscriptionClient):
        def ask(self, prompt: str) -> str:
            chat_calls.append(prompt)
            return "subscription response"

    def semantic_embedding(payload: dict[str, object]) -> dict[str, object]:
        embedding_calls.append(payload)
        return {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.75]}],
            "model": "fake-semantic-embedding",
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }

    upstream = build_subscription_upstream(
        FakeClient(CodexSubscriptionConfig()),
        embedding_upstream=semantic_embedding,
    )

    embedding_response = upstream(
        {"model": "text-embedding-3-small", "input": ["semantic document"]}
    )
    chat_response = upstream(
        {
            "model": "gpt-5-mini",
            "messages": [{"role": "user", "content": "answer this"}],
        }
    )

    assert embedding_response["model"] == "fake-semantic-embedding"
    assert embedding_calls == [{"model": "text-embedding-3-small", "input": ["semantic document"]}]
    assert chat_calls and "user: answer this" in chat_calls[0]
    choices = cast(list[dict[str, object]], chat_response["choices"])
    message = cast(dict[str, object], choices[0]["message"])
    assert message["content"] == "subscription response"


def test_subscription_upstream_translates_chat_to_codex_messages() -> None:
    calls: list[str] = []

    class FakeClient(CodexSubscriptionClient):
        def ask(self, prompt: str) -> str:
            calls.append(prompt)
            return "subscription response"

    upstream = build_subscription_upstream(FakeClient(CodexSubscriptionConfig()))
    response = upstream(
        {
            "model": "gpt-5-mini",
            "messages": [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "What is new?"},
            ],
        }
    )

    choices = cast(list[dict[str, object]], response["choices"])
    message = cast(dict[str, object], choices[0]["message"])
    assert message["content"] == "subscription response"
    assert "system: Be concise" in calls[0]
    assert "user: What is new?" in calls[0]


def test_run_config_accepts_subscription_provider_mode() -> None:
    config = RunConfig(provider_mode="codex-subscription")

    assert config.provider_mode == "codex-subscription"
    assert config.embedding_backend == "local-hash"
    assert (
        RunConfig(
            provider_mode="codex-subscription",
            embedding_backend="openai",
        ).embedding_backend
        == "openai"
    )
    with pytest.raises(ValueError, match="provider_mode"):
        RunConfig(provider_mode="unsupported")
    with pytest.raises(ValueError, match="AUTOBRAIN_EMBEDDING_BACKEND"):
        RunConfig(provider_mode="codex-subscription", embedding_backend="looks-semantic")
    with pytest.raises(ValueError, match="requires embedding_backend=openai"):
        RunConfig(provider_mode="api", embedding_backend="local-hash")


def test_subscription_ask_uses_read_only_ephemeral_codex_exec() -> None:
    calls: list[list[str]] = []

    def runner(
        args: Sequence[str],
        _stdin: str,
        _timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        if list(args)[1:3] == ["login", "status"]:
            return subprocess.CompletedProcess(args, 0, "Logged in", "")
        return subprocess.CompletedProcess(
            args,
            0,
            (
                '{"type":"item.completed","item":{"type":"agent_message",'
                '"text":"subscription answer"}}\n'
            ),
            "",
        )

    client = CodexSubscriptionClient(
        CodexSubscriptionConfig(model="gpt-5"),
        runner=runner,
    )

    assert client.ask("Answer safely") == "subscription answer"
    assert calls[1] == [
        "codex",
        "exec",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--model",
        "gpt-5",
    ]


def test_subscription_execution_requires_authenticated_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def which(_command: str) -> str:
        return "/usr/local/bin/codex"

    monkeypatch.setattr("autobrain.subscription.shutil.which", which)

    def runner(
        args: Sequence[str],
        _stdin: str,
        _timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, "", "login required")

    client = CodexSubscriptionClient(CodexSubscriptionConfig(), runner=runner)

    assert client.status() is SubscriptionStatus.SUBSCRIPTION_AUTH_UNAVAILABLE


def test_run_local_proxy_exposes_subscription_chat_boundary() -> None:
    class FakeClient(CodexSubscriptionClient):
        def ask(self, prompt: str) -> str:
            assert "user: answer this" in prompt
            return "local subscription answer"

    proxy = LoopbackMeteringProxy(
        build_subscription_upstream(FakeClient(CodexSubscriptionConfig()))
    )
    with proxy:
        response = proxy.chat(
            {
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "answer this"}],
            },
            candidate="llm-wiki",
            phase="query",
        )

    choices = cast(list[dict[str, object]], response["choices"])
    message = cast(dict[str, object], choices[0]["message"])
    assert message["content"] == "local subscription answer"
