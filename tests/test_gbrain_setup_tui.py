from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from pydantic import SecretStr

from autobrain.candidates.gbrain_config import (
    GBrainEmbeddingProvider,
    GBrainExecutionConfig,
)
from autobrain.tui_actions import (
    GBrainValidated,
    GoBack,
    Navigate,
    SelectGBrainMode,
    ValidateGBrain,
)
from autobrain.tui_effects import ValidateGBrainProvider
from autobrain.subscription_domain import ProviderId
from autobrain.tui_state import UiScreen, UiState, reduce_ui
from autobrain.tui_textual import AutoBrainApp


def test_reducer_quick_start_and_transient_semantic_handoff() -> None:
    quick = reduce_ui(UiState(screen=UiScreen.CANDIDATES), SelectGBrainMode(False))
    assert quick.state.screen is UiScreen.REVIEW
    assert quick.state.gbrain_config.keyword_only

    semantic = reduce_ui(UiState(screen=UiScreen.CANDIDATES), SelectGBrainMode(True))
    assert semantic.state.screen is UiScreen.GBRAIN
    requested = reduce_ui(
        semantic.state,
        ValidateGBrain(
            GBrainEmbeddingProvider.VOYAGE,
            credential=SecretStr("transient-secret"),
        ),
    )
    assert isinstance(requested.effects[0], ValidateGBrainProvider)
    assert "transient-secret" not in repr(requested.effects[0])
    config = requested.effects[0].config
    settled = reduce_ui(requested.state, GBrainValidated(config))
    assert settled.state.screen is UiScreen.REVIEW
    assert settled.state.gbrain_config.embedding.provider is GBrainEmbeddingProvider.VOYAGE

    cleared = reduce_ui(replace(settled.state, screen=UiScreen.GBRAIN), GoBack())
    assert cleared.state.gbrain_config.keyword_only


@pytest.mark.parametrize(
    "provider",
    [
        GBrainEmbeddingProvider.OPENAI,
        GBrainEmbeddingProvider.VOYAGE,
        GBrainEmbeddingProvider.GEMINI,
        GBrainEmbeddingProvider.OPENROUTER,
        GBrainEmbeddingProvider.OLLAMA,
        GBrainEmbeddingProvider.LLAMA_SERVER,
    ],
)
def test_every_semantic_provider_has_deterministic_validation(
    provider: GBrainEmbeddingProvider,
) -> None:
    kwargs: dict[str, object] = {}
    if provider in {
        GBrainEmbeddingProvider.OPENAI,
        GBrainEmbeddingProvider.VOYAGE,
        GBrainEmbeddingProvider.GEMINI,
        GBrainEmbeddingProvider.OPENROUTER,
    }:
        kwargs["credential"] = "hosted-test-key"
    if provider is GBrainEmbeddingProvider.LLAMA_SERVER:
        kwargs.update(model="embed.gguf", dimensions=384)
    config = GBrainExecutionConfig.semantic(provider, **kwargs)
    assert config.embedding.provider is provider
    assert "hosted-test-key" not in repr(config)


def test_quick_start_keyboard_surface_fits_minimum_terminal() -> None:
    async def exercise() -> None:
        app = AutoBrainApp(force_setup=True, provider=ProviderId.CODEX)
        app._execute = lambda effect: None  # type: ignore[method-assign]
        async with app.run_test(size=(60, 22)) as pilot:
            app.dispatch_ui(Navigate(UiScreen.CANDIDATES.value))
            await pilot.pause()
            button = app.screen.query_one("#quick-start")
            button.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert app.state.screen is UiScreen.REVIEW
            assert app.state.gbrain_config.keyword_only

    asyncio.run(exercise())
