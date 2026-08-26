from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from autobrain.candidates.gbrain import CommandResult, GBrainAdapter
from autobrain.candidates.gbrain_config import (
    GBrainEmbeddingProvider,
    GBrainExecutionConfig,
    validate_endpoint,
)
from tests.candidates.test_gbrain import FakeRunner, document, fake_checkout


@pytest.mark.parametrize(
    ("provider", "model", "dimensions", "env_name"),
    [
        ("openai", "text-embedding-3-small", 1536, "OPENAI_API_KEY"),
        ("voyage", "voyage-4", 1024, "VOYAGE_API_KEY"),
        ("gemini", "gemini-embedding-001", 768, "GEMINI_API_KEY"),
        ("openrouter", "openai/text-embedding-3-small", 1536, "OPENROUTER_API_KEY"),
    ],
)
def test_hosted_provider_argv_and_secret_environment(
    tmp_path: Path, provider: str, model: str, dimensions: int, env_name: str
) -> None:
    tools = tmp_path / "tools"
    checkout = fake_checkout(tools)
    runner = FakeRunner(checkout)
    config = GBrainExecutionConfig.semantic(provider, credential="super-secret")
    GBrainAdapter(tools, tmp_path / "run", runner=runner, config=config).run(
        [document()], ["launch?"]
    )
    init = next(
        call for call in runner.calls if call[0][2:3] == ("init",) and "--help" not in call[0]
    )
    assert "--embedding-provider" not in init[0]
    assert f"{provider}:{model}" in init[0]
    assert str(dimensions) in init[0]
    assert init[2][env_name] == "super-secret"
    assert all("super-secret" not in part for part in init[0])


def test_quick_start_is_keyless_and_skips_dream_query_think(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tools = tmp_path / "tools"
    checkout = fake_checkout(tools)
    runner = FakeRunner(checkout)
    result = GBrainAdapter(
        tools, tmp_path / "run", runner=runner, config=GBrainExecutionConfig.quick_start()
    ).run([document()], ["launch?"])[0]
    cli = [call[0] for call in runner.calls if call[0][:2] == ("bun", "src/cli.ts")]
    assert [command[2] for command in cli] == ["init", "init", "import", "sync", "status", "search"]
    assert "--no-embedding" in cli[1]
    assert "--quickstart" not in cli[1]
    assert all("OPENAI_API_KEY" not in call[2] for call in runner.calls)
    assert result.answer
    assert result.keyword_only is True
    assert result.semantic_enabled is False
    assert result.semantic_quality == "not_measured"
    assert result.recommendation_eligible is False


def test_local_providers_are_only_selected_explicitly(tmp_path: Path) -> None:
    assert (
        GBrainExecutionConfig.quick_start().embedding.provider
        is GBrainEmbeddingProvider.KEYWORD_ONLY
    )
    ollama = GBrainExecutionConfig.semantic("ollama")
    assert ollama.embedding.model == "nomic-embed-text"
    assert ollama.embedding.endpoint == "http://127.0.0.1:11434"
    llama = GBrainExecutionConfig.semantic(
        "llama-server", model="embed.gguf", dimensions=384, endpoint="http://127.0.0.1:8080"
    )
    assert llama.embedding.dimensions == 384
    with pytest.raises(ValueError, match="positive dimensions"):
        GBrainExecutionConfig.semantic("llama-server", model="embed.gguf", dimensions=0)


def test_endpoint_rejects_userinfo() -> None:
    with pytest.raises(ValueError, match="userinfo"):
        validate_endpoint("http://user:secret@127.0.0.1:11434")


def test_secret_is_removed_from_process_errors(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    fake_checkout(tools)

    def fail(
        command: Sequence[str], cwd: Path, env: dict[str, str], timeout: float
    ) -> CommandResult:
        del cwd, timeout
        output = "provider refused key=" + env.get("VOYAGE_API_KEY", "")
        if command[-2:] == ("rev-parse", "HEAD"):
            return CommandResult(
                tuple(command), 0, "f49ca569232dbc0d8e0783d84606115e3bfe5ab1", "", 1
            )
        if command[:2] != ("bun", "src/cli.ts"):
            return CommandResult(tuple(command), 0, "{}", "", 1)
        return CommandResult(tuple(command), 1, "", output, 1)

    config = GBrainExecutionConfig.semantic("voyage", credential="never-render-this")
    with pytest.raises(Exception) as caught:
        GBrainAdapter(tools, tmp_path / "run", runner=fail, config=config).run(
            [document()], ["launch?"]
        )
    assert "never-render-this" not in str(caught.value)
    assert "never-render-this" not in repr(config)
