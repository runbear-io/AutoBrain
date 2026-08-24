from __future__ import annotations

import asyncio
import html
import re
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import pytest
from pydantic import SecretStr
from textual.widgets import Input, Select

from autobrain.auth.models import Provider
from autobrain.candidates.gbrain_config import (
    GBrainEmbeddingProvider,
    GBrainExecutionConfig,
)
from autobrain.embedding import inspect_embedding_backend
from autobrain.models import ConnectionState
from autobrain.subscription_domain import ProviderId, SubscriptionStatus
from autobrain.tui_actions import (
    GBrainValidated,
    GoBack,
    Navigate,
    SelectGBrainMode,
    StartRun,
    ValidateGBrain,
)
from autobrain.tui_effects import RunExperiment, UiEffect, ValidateGBrainProvider
from autobrain.tui_state import UiScreen, UiState, reduce_ui
from autobrain.tui_textual import AutoBrainApp, validate_gbrain_connection


def _cell_text(app: AutoBrainApp) -> str:
    svg = app.export_screenshot()
    return html.unescape(re.sub(r"<[^>]+>", "", svg)).replace("\xa0", " ")


def _connected_state() -> UiState:
    return UiState(
        screen=UiScreen.CANDIDATES,
        subscription_status=SubscriptionStatus.READY,
        embedding_readiness=inspect_embedding_backend({}),
        source_states=(
            (Provider.SLACK, ConnectionState.CONNECTED),
            (Provider.NOTION, ConnectionState.CONNECTED),
        ),
    )


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
    credential = None
    model = None
    dimensions = None
    if provider in {
        GBrainEmbeddingProvider.OPENAI,
        GBrainEmbeddingProvider.VOYAGE,
        GBrainEmbeddingProvider.GEMINI,
        GBrainEmbeddingProvider.OPENROUTER,
    }:
        credential = "hosted-test-key"
    if provider is GBrainEmbeddingProvider.LLAMA_SERVER:
        model = "embed.gguf"
        dimensions = 384
    config = GBrainExecutionConfig.semantic(
        provider, credential=credential, model=model, dimensions=dimensions
    )
    assert config.embedding.provider is provider
    assert "hosted-test-key" not in repr(config)


def test_validated_semantic_provider_resolves_plan_without_ambient_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    state = _connected_state()
    state = reduce_ui(state, SelectGBrainMode(True)).state
    requested = reduce_ui(
        state,
        ValidateGBrain(GBrainEmbeddingProvider.VOYAGE, credential=SecretStr("transient")),
    )
    effect = requested.effects[0]
    assert isinstance(effect, ValidateGBrainProvider)
    config = effect.config
    settled = reduce_ui(requested.state, GBrainValidated(config))
    assert settled.state.plan is not None
    assert settled.state.plan.gbrain_config.embedding.provider is GBrainEmbeddingProvider.VOYAGE
    assert settled.state.embedding_readiness.recommendation_ready is False
    assert "OPENAI_API_KEY" not in settled.state.setup_error


def test_quick_start_is_runnable_without_semantic_key_and_preserves_evaluator_gate() -> None:
    state = _connected_state()
    review = reduce_ui(state, SelectGBrainMode(False))
    assert review.state.plan is not None
    assert review.state.plan.embedding_backend == "local-hash"
    assert review.state.embedding_readiness.recommendation_ready is False
    assert "OPENAI_API_KEY" not in review.state.setup_error
    started = reduce_ui(review.state, StartRun())
    assert started.state.screen is UiScreen.RUNNING
    assert started.effects


@pytest.mark.parametrize(
    ("action", "code"),
    [
        (ValidateGBrain(GBrainEmbeddingProvider.OPENAI), "GBRAIN_PROVIDER_KEY_REQUIRED"),
        (
            ValidateGBrain(GBrainEmbeddingProvider.OLLAMA, endpoint="ftp://localhost:11434"),
            "GBRAIN_ENDPOINT_SCHEME_INVALID",
        ),
        (
            ValidateGBrain(
                GBrainEmbeddingProvider.OLLAMA,
                endpoint="http://user:password@localhost:11434",
            ),
            "GBRAIN_ENDPOINT_USERINFO_FORBIDDEN",
        ),
        (ValidateGBrain(GBrainEmbeddingProvider.LLAMA_SERVER), "GBRAIN_MODEL_REQUIRED"),
        (
            ValidateGBrain(
                GBrainEmbeddingProvider.LLAMA_SERVER,
                model="embed.gguf",
                dimensions=0,
            ),
            "GBRAIN_DIMENSIONS_INVALID",
        ),
    ],
)
def test_semantic_setup_errors_have_actionable_typed_codes(
    action: ValidateGBrain,
    code: str,
) -> None:
    result = reduce_ui(UiState(screen=UiScreen.GBRAIN), action)
    assert result.effects == ()
    assert result.state.gbrain_error.startswith(f"{code}:")


def test_local_endpoint_unavailable_is_typed_and_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    config = GBrainExecutionConfig.semantic("ollama")

    def unavailable(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("raw socket detail with secret=do-not-render")

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    with pytest.raises(RuntimeError, match="GBRAIN_LOCAL_ENDPOINT_UNAVAILABLE") as error:
        validate_gbrain_connection(config)
    assert "do-not-render" not in str(error.value)


def test_quick_start_keyboard_surface_fits_minimum_terminal() -> None:
    async def exercise() -> None:
        app = AutoBrainApp(force_setup=True, provider=ProviderId.CODEX)
        app._execute = lambda effect: None  # type: ignore[method-assign]
        async with app.run_test(size=(60, 22)) as pilot:
            app.dispatch_ui(Navigate(UiScreen.CANDIDATES.value))
            await pilot.pause()
            screenshot = _cell_text(app)
            assert "Quick Start" in screenshot
            assert "Semantic Setup" in screenshot
            button = app.screen.query_one("#quick-start")
            button.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert app.state.screen is UiScreen.REVIEW
            assert app.state.gbrain_config.keyword_only

    asyncio.run(exercise())


def test_semantic_setup_selected_provider_label_tracks_select_at_minimum_viewport() -> None:
    async def exercise() -> None:
        app = AutoBrainApp(force_setup=True, provider=ProviderId.CODEX)
        app._execute = lambda effect: None  # type: ignore[method-assign]
        async with app.run_test(size=(60, 22)) as pilot:
            app.dispatch_ui(Navigate(UiScreen.GBRAIN.value))
            await pilot.pause()
            assert "Selected provider: OpenAI" in _cell_text(app)

            provider = cast(Select[str], app.screen.query_one("#gbrain-provider", Select))
            provider.value = GBrainEmbeddingProvider.OLLAMA.value
            await pilot.pause()
            assert "Selected provider: Ollama" in _cell_text(app)

    asyncio.run(exercise())


def test_semantic_setup_initial_viewport_shows_fields_validate_and_inline_error() -> None:
    async def exercise() -> None:
        app = AutoBrainApp(force_setup=True, provider=ProviderId.CODEX)
        app._execute = lambda effect: None  # type: ignore[method-assign]
        async with app.run_test(size=(60, 22)) as pilot:
            app.dispatch_ui(Navigate(UiScreen.GBRAIN.value))
            await pilot.pause()
            initial = _cell_text(app)
            for copy in (
                "Semantic Setup",
                "API key",
                "Model",
                "Dimensions",
                "Endpoint",
                "Validate",
            ):
                assert copy in initial
            app.dispatch_ui(ValidateGBrain(GBrainEmbeddingProvider.OPENAI))
            await pilot.pause()
            assert "GBRAIN_PROVIDER_KEY_REQUIRED" in _cell_text(app)

    asyncio.run(exercise())


def test_semantic_setup_keyboard_ollama_validation_emits_runnable_run_effect() -> None:
    class LoopbackHandler(BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), LoopbackHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"

    async def exercise() -> None:
        captured: list[UiEffect] = []
        review_ready = asyncio.Event()
        app = AutoBrainApp(force_setup=True, provider=ProviderId.CODEX)

        def execute(effect: UiEffect) -> None:
            captured.append(effect)
            if isinstance(effect, ValidateGBrainProvider):
                config = validate_gbrain_connection(effect.config)

                def settle_validation() -> None:
                    app.dispatch_ui(GBrainValidated(config))
                    app.call_after_refresh(review_ready.set)

                app.call_later(settle_validation)

        app._execute = execute  # type: ignore[method-assign]
        async with app.run_test(size=(60, 22)) as pilot:
            app.state = _connected_state()
            app.dispatch_ui(Navigate(UiScreen.GBRAIN.value))
            await pilot.pause()

            provider = cast(Select[str], app.screen.query_one("#gbrain-provider", Select))
            provider.focus()
            await pilot.press("enter", "home", "down", "down", "down", "down", "down", "enter")
            await pilot.pause()
            assert provider.value == GBrainEmbeddingProvider.OLLAMA.value
            assert "Selected provider: Ollama" in _cell_text(app)

            await pilot.press("tab", "tab")
            await pilot.press(*"nomic-test")
            await pilot.press("tab")
            await pilot.press(*"384")
            await pilot.press("tab")
            await pilot.press(*endpoint)
            await pilot.press("tab", "tab", "enter")
            await asyncio.wait_for(review_ready.wait(), timeout=2)

            assert app.state.screen is UiScreen.REVIEW
            review = _cell_text(app)
            for copy in ("Review", "Setup: ollama", "nomic-test", "384", endpoint, "Runnable"):
                assert copy in review

            app.screen.query_one("#run").focus()
            await pilot.press("enter")
            await pilot.pause()

        run_effect = cast(RunExperiment, captured[-1])
        assert isinstance(run_effect, RunExperiment)
        config = run_effect.plan.gbrain_config
        assert config.embedding.provider is GBrainEmbeddingProvider.OLLAMA
        assert config.embedding.model == "nomic-test"
        assert config.embedding.dimensions == 384
        assert config.embedding.endpoint == endpoint
        assert config.credential is None
        assert run_effect.plan.sources
        assert len(run_effect.plan.candidates) >= 2
        assert run_effect.plan.provider_mode == "codex-subscription"

    try:
        asyncio.run(exercise())
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


def test_semantic_setup_validate_button_uses_runtime_select_type() -> None:
    async def exercise() -> None:
        app = AutoBrainApp(force_setup=True, provider=ProviderId.CODEX)
        app._execute = lambda effect: None  # type: ignore[method-assign]
        async with app.run_test(size=(60, 22)) as pilot:
            app.dispatch_ui(Navigate(UiScreen.CANDIDATES.value))
            await pilot.pause()
            app.screen.query_one("#semantic-setup").focus()
            await pilot.press("enter")
            await pilot.pause()
            app.screen.query_one("#gbrain-provider", Select).value = "openai"
            app.screen.query_one("#gbrain-key", Input).value = "synthetic-key"
            app._execute = AutoBrainApp._execute.__get__(app, AutoBrainApp)  # type: ignore[method-assign]
            app.screen.query_one("#validate-gbrain").focus()
            await pilot.press("enter")
            await pilot.pause()
            assert app.state.screen is UiScreen.REVIEW
            assert app.state.gbrain_config.embedding.provider is GBrainEmbeddingProvider.OPENAI
            assert "synthetic-key" not in app.export_screenshot()

    asyncio.run(exercise())
