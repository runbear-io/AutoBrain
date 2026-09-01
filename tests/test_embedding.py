"""Tests for the Gemini BYOK embedding backend in :mod:`autobrain.embedding`.

These tests verify config resolution, readiness behaviour, request parsing,
provenance metadata, and secret custody for the ``gemini`` embedding backend
without making live API calls.  Local-hash and OpenAI behaviour is asserted
to be unchanged.
"""

from __future__ import annotations

import json
from typing import cast
from unittest.mock import patch
from urllib.request import Request

import pytest

from autobrain.embedding import (
    EmbeddingBackend,
    EmbeddingBackendConfig,
    EmbeddingBackendDescriptor,
    EmbeddingBackendRegistry,
    build_gemini_embedding_upstream,
    build_openai_embedding_upstream,
    inspect_embedding_backend,
    production_embedding_registry,
    recommendation_eligibility_reason,
)
from autobrain.models import EmbeddingProvenance, EmbeddingQuality, Status

_GEMINI_KEY = "AIza-fixture-gemini-key-DO-NOT-USE"
_OPENAI_KEY = "fixture-openai-key"


# ---------------------------------------------------------------------------
# Registry / descriptor
# ---------------------------------------------------------------------------


class TestGeminiBackendRegistration:
    def test_gemini_is_a_production_backend_selector(self) -> None:
        registry = production_embedding_registry()
        descriptor = registry.resolve_selector("gemini")
        assert descriptor is not None
        assert descriptor.selector == "gemini"

    def test_gemini_descriptor_is_semantic_and_recommendation_eligible(self) -> None:
        descriptor = production_embedding_registry().resolve_selector("gemini")
        assert descriptor is not None
        assert descriptor.quality is EmbeddingQuality.SEMANTIC
        assert descriptor.recommendation_eligible is True
        assert descriptor.requires_api_key is True

    def test_gemini_descriptor_provenance_backend_identifies_google_model(self) -> None:
        descriptor = production_embedding_registry().resolve_selector("gemini")
        assert descriptor is not None
        assert descriptor.provenance_backend == "google:gemini-embedding-001"

    def test_gemini_descriptor_does_not_use_openai_transport(self) -> None:
        descriptor = production_embedding_registry().resolve_selector("gemini")
        assert descriptor is not None
        assert descriptor.openai_transport is False
        assert descriptor.gemini_transport is True

    def test_gemini_descriptor_api_key_env_is_gemini(self) -> None:
        descriptor = production_embedding_registry().resolve_selector("gemini")
        assert descriptor is not None
        assert descriptor.api_key_env == "GEMINI_API_KEY"

    def test_embedding_backend_enum_has_gemini(self) -> None:
        assert EmbeddingBackend.GEMINI.value == "gemini"


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


class TestGeminiConfigResolution:
    def test_from_environ_reads_gemini_api_key(self) -> None:
        config = EmbeddingBackendConfig.from_environ(
            {"GEMINI_API_KEY": _GEMINI_KEY},
            requested="gemini",
        )
        assert config.backend == "gemini"
        assert config.api_key is not None
        assert config.api_key.get_secret_value() == _GEMINI_KEY

    def test_from_environ_does_not_read_openai_key_for_gemini(self) -> None:
        config = EmbeddingBackendConfig.from_environ(
            {"OPENAI_API_KEY": _OPENAI_KEY},
            requested="gemini",
        )
        assert config.backend == "gemini"
        assert config.api_key is None

    def test_from_environ_with_autobrain_embedding_backend_env(self) -> None:
        config = EmbeddingBackendConfig.from_environ(
            {"AUTOBRAIN_EMBEDDING_BACKEND": "gemini", "GEMINI_API_KEY": _GEMINI_KEY},
        )
        assert config.backend == "gemini"
        assert config.api_key is not None
        assert config.api_key.get_secret_value() == _GEMINI_KEY

    def test_from_environ_explicit_api_key_overrides_env(self) -> None:
        config = EmbeddingBackendConfig.from_environ(
            {"GEMINI_API_KEY": "env-value"},
            requested="gemini",
            api_key="explicit-key",
        )
        assert config.api_key is not None
        assert config.api_key.get_secret_value() == "explicit-key"

    def test_provenance_metadata_for_gemini(self) -> None:
        config = EmbeddingBackendConfig.from_environ(
            {"GEMINI_API_KEY": _GEMINI_KEY},
            requested="gemini",
        )
        provenance = config.provenance
        assert provenance.backend == "google:gemini-embedding-001"
        assert provenance.quality is EmbeddingQuality.SEMANTIC


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


class TestGeminiReadiness:
    def test_gemini_with_key_is_ready_and_recommendation_eligible(self) -> None:
        readiness = EmbeddingBackendConfig.from_environ(
            {"GEMINI_API_KEY": _GEMINI_KEY},
            requested="gemini",
        ).readiness()
        assert readiness.status is Status.OK
        assert readiness.recommendation_ready is True
        assert "google:gemini-embedding-001" in readiness.detail

    def test_gemini_without_key_is_missing_provider(self) -> None:
        readiness = EmbeddingBackendConfig.from_environ(
            {},
            requested="gemini",
        ).readiness()
        assert readiness.status is Status.MISSING_PROVIDER
        assert readiness.recommendation_ready is False
        assert "GEMINI_API_KEY" in readiness.detail
        assert "SEMANTIC_EMBEDDING_UNAVAILABLE" in readiness.detail

    def test_gemini_without_key_does_not_recommend(self) -> None:
        config = EmbeddingBackendConfig.from_environ(
            {},
            requested="gemini",
        )
        assert config.recommendation_ready is False

    def test_inspect_embedding_backend_gemini_with_key(self) -> None:
        readiness = inspect_embedding_backend(
            {"GEMINI_API_KEY": _GEMINI_KEY},
            requested="gemini",
        )
        assert readiness.status is Status.OK
        assert readiness.recommendation_ready is True

    def test_inspect_embedding_backend_gemini_without_key(self) -> None:
        readiness = inspect_embedding_backend(
            {},
            requested="gemini",
        )
        assert readiness.status is Status.MISSING_PROVIDER
        assert readiness.recommendation_ready is False
        assert _GEMINI_KEY not in readiness.detail


# ---------------------------------------------------------------------------
# Recommendation eligibility
# ---------------------------------------------------------------------------


class TestGeminiRecommendationEligibility:
    def test_gemini_provenance_is_recommendation_eligible(self) -> None:
        provenance = EmbeddingProvenance(
            backend="google:gemini-embedding-001",
            quality=EmbeddingQuality.SEMANTIC,
        )
        assert recommendation_eligibility_reason(provenance) is None

    def test_unknown_gemini_like_backend_is_not_eligible(self) -> None:
        provenance = EmbeddingProvenance(
            backend="google:unknown-model",
            quality=EmbeddingQuality.SEMANTIC,
        )
        reason = recommendation_eligibility_reason(provenance)
        assert reason is not None
        assert "not registered" in reason


# ---------------------------------------------------------------------------
# Secret custody / repr safety
# ---------------------------------------------------------------------------


class TestGeminiSecretCustody:
    def test_repr_does_not_expose_gemini_key(self) -> None:
        config = EmbeddingBackendConfig.from_environ(
            {"GEMINI_API_KEY": _GEMINI_KEY},
            requested="gemini",
        )
        rendered = repr(config)
        assert _GEMINI_KEY not in rendered

    def test_readiness_detail_does_not_expose_gemini_key(self) -> None:
        readiness = EmbeddingBackendConfig.from_environ(
            {"GEMINI_API_KEY": _GEMINI_KEY},
            requested="gemini",
        ).readiness()
        assert _GEMINI_KEY not in readiness.detail

    def test_invalid_backend_detail_does_not_expose_key(self) -> None:
        readiness = inspect_embedding_backend(
            {"GEMINI_API_KEY": _GEMINI_KEY},
            requested="nonexistent-backend",
        )
        assert readiness.status is Status.ENV_UNAVAILABLE
        assert _GEMINI_KEY not in readiness.detail

    def test_registry_repr_does_not_expose_keys(self) -> None:
        registry = production_embedding_registry()
        assert _GEMINI_KEY not in repr(registry)

    def test_descriptor_repr_does_not_expose_keys(self) -> None:
        descriptor = cast(
            EmbeddingBackendDescriptor,
            production_embedding_registry().resolve_selector("gemini"),
        )
        assert _GEMINI_KEY not in repr(descriptor)


# ---------------------------------------------------------------------------
# Request parsing / upstream
# ---------------------------------------------------------------------------


class TestGeminiUpstreamRequestParsing:
    def test_build_gemini_upstream_requires_gemini_transport(self) -> None:
        openai_config = EmbeddingBackendConfig.from_environ(
            {"OPENAI_API_KEY": _OPENAI_KEY},
            requested="openai",
        )
        with pytest.raises(ValueError, match=r"(?i)gemini"):
            build_gemini_embedding_upstream(openai_config)

    def test_build_gemini_upstream_requires_api_key(self) -> None:
        config = EmbeddingBackendConfig.from_environ(
            {},
            requested="gemini",
        )
        with pytest.raises(ValueError, match="not ready"):
            build_gemini_embedding_upstream(config)

    def test_gemini_upstream_translates_openai_payload_to_gemini_api(
        self,
    ) -> None:
        config = EmbeddingBackendConfig.from_environ(
            {"GEMINI_API_KEY": _GEMINI_KEY},
            requested="gemini",
        )
        upstream = build_gemini_embedding_upstream(config)

        captured_requests: list[dict[str, object]] = []

        class _FakeResponse:
            def __init__(self, body: bytes) -> None:
                self._body = body

            def __enter__(self) -> _FakeResponse:
                return self

            def __exit__(self, *_args: object) -> None:
                pass

            def read(self) -> bytes:
                return self._body

        def fake_urlopen(req: object, timeout: float = 30) -> _FakeResponse:
            del timeout
            req_obj = req
            url = getattr(req_obj, "full_url", "")
            data = getattr(req_obj, "data", b"")
            headers = getattr(req_obj, "headers", {})
            captured_requests.append(
                {
                    "url": url,
                    "data": json.loads(data) if data else {},
                    "headers": dict(headers) if headers else {},
                }
            )
            return _FakeResponse(
                json.dumps(
                    {
                        "embeddings": [
                            {"values": [0.1, 0.2, 0.3]},
                            {"values": [0.4, 0.5, 0.6]},
                        ]
                    }
                ).encode("utf-8"),
            )

        with patch("autobrain.embedding.urllib.request.urlopen", side_effect=fake_urlopen):
            result = upstream(
                {
                    "model": "text-embedding-3-small",
                    "input": ["hello world", "second text"],
                }
            )

        assert len(captured_requests) == 1
        request = captured_requests[0]
        url = cast(str, request["url"])
        assert "generativelanguage.googleapis.com" in url
        assert "gemini-embedding-001" in url
        assert "batchEmbedContent" in url
        body = cast(dict[str, object], request["data"])
        requests_list = cast(list[dict[str, object]], body["requests"])
        assert len(requests_list) == 2
        first_parts = cast(
            list[dict[str, str]],
            cast(dict[str, object], requests_list[0]["content"])["parts"],
        )
        assert first_parts[0]["text"] == "hello world"
        headers = cast(dict[str, str], request["headers"])
        assert any(k.lower() == "x-goog-api-key" for k in headers)
        assert next(v for k, v in headers.items() if k.lower() == "x-goog-api-key") == _GEMINI_KEY

        # Response must be OpenAI-compatible
        assert result["object"] == "list"
        data = cast(list[dict[str, object]], result["data"])
        assert len(data) == 2
        assert data[0]["index"] == 0
        assert data[1]["index"] == 1
        assert cast(list[float], data[0]["embedding"]) == [0.1, 0.2, 0.3]
        assert cast(list[float], data[1]["embedding"]) == [0.4, 0.5, 0.6]

    def test_gemini_upstream_single_input_uses_embed_content(
        self,
    ) -> None:
        config = EmbeddingBackendConfig.from_environ(
            {"GEMINI_API_KEY": _GEMINI_KEY},
            requested="gemini",
        )
        upstream = build_gemini_embedding_upstream(config)

        captured_urls: list[str] = []

        class _FakeResponse:
            def __init__(self, body: bytes) -> None:
                self._body = body

            def __enter__(self) -> _FakeResponse:
                return self

            def __exit__(self, *_args: object) -> None:
                pass

            def read(self) -> bytes:
                return self._body

        def fake_urlopen(req: object, timeout: float = 30) -> _FakeResponse:
            del timeout
            captured_urls.append(getattr(req, "full_url", ""))
            return _FakeResponse(
                json.dumps({"embedding": {"values": [0.1, 0.2, 0.3]}}).encode("utf-8"),
            )

        with patch("autobrain.embedding.urllib.request.urlopen", side_effect=fake_urlopen):
            result = upstream(
                {
                    "model": "text-embedding-3-small",
                    "input": "single text",
                }
            )

        assert len(captured_urls) == 1
        assert "embedContent" in captured_urls[0]
        data = cast(list[dict[str, object]], result["data"])
        assert len(data) == 1
        assert cast(list[float], data[0]["embedding"]) == [0.1, 0.2, 0.3]

    def test_gemini_upstream_raises_runtime_error_on_http_failure(self) -> None:
        config = EmbeddingBackendConfig.from_environ(
            {"GEMINI_API_KEY": _GEMINI_KEY},
            requested="gemini",
        )
        upstream = build_gemini_embedding_upstream(config)

        def fake_urlopen(_req: object, timeout: float = 30) -> object:
            del timeout
            raise OSError("connection refused")

        with (
            patch("autobrain.embedding.urllib.request.urlopen", side_effect=fake_urlopen),
            pytest.raises(RuntimeError, match="semantic embedding request failed"),
        ):
            upstream({"model": "text-embedding-3-small", "input": "text"})

    def test_gemini_upstream_key_not_in_runtime_error(self) -> None:
        config = EmbeddingBackendConfig.from_environ(
            {"GEMINI_API_KEY": _GEMINI_KEY},
            requested="gemini",
        )
        upstream = build_gemini_embedding_upstream(config)

        def fake_urlopen(req: object, timeout: float = 30) -> object:
            del timeout
            raise OSError(f"url was {getattr(req, 'full_url', '')}")

        with (
            patch("autobrain.embedding.urllib.request.urlopen", side_effect=fake_urlopen),
            pytest.raises(RuntimeError) as exc_info,
        ):
            upstream({"model": "text-embedding-3-small", "input": "text"})

        assert _GEMINI_KEY not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Local-hash and OpenAI unchanged
# ---------------------------------------------------------------------------


class TestExistingBackendsUnchanged:
    def test_local_hash_descriptor_unchanged(self) -> None:
        registry = production_embedding_registry()
        descriptor = registry.resolve_selector("local-hash")
        assert descriptor is not None
        assert descriptor.quality is EmbeddingQuality.SMOKE_ONLY
        assert descriptor.recommendation_eligible is False
        assert descriptor.requires_api_key is False
        assert descriptor.openai_transport is False
        assert descriptor.gemini_transport is False
        assert descriptor.provenance_backend == "local-hash-embedding"

    def test_openai_descriptor_unchanged(self) -> None:
        registry = production_embedding_registry()
        descriptor = registry.resolve_selector("openai")
        assert descriptor is not None
        assert descriptor.quality is EmbeddingQuality.SEMANTIC
        assert descriptor.recommendation_eligible is True
        assert descriptor.requires_api_key is True
        assert descriptor.openai_transport is True
        assert descriptor.gemini_transport is False
        assert descriptor.provenance_backend == "openai:text-embedding-3-small"

    def test_local_hash_readiness_unchanged(self) -> None:
        readiness = EmbeddingBackendConfig.from_environ(
            {},
            requested="local-hash",
        ).readiness()
        assert readiness.status is Status.CAPABILITY_UNAVAILABLE
        assert readiness.recommendation_ready is False

    def test_openai_readiness_with_key_unchanged(self) -> None:
        readiness = EmbeddingBackendConfig.from_environ(
            {"OPENAI_API_KEY": _OPENAI_KEY},
            requested="openai",
        ).readiness()
        assert readiness.status is Status.OK
        assert readiness.recommendation_ready is True

    def test_openai_readiness_without_key_unchanged(self) -> None:
        readiness = EmbeddingBackendConfig.from_environ(
            {},
            requested="openai",
        ).readiness()
        assert readiness.status is Status.MISSING_PROVIDER
        assert readiness.recommendation_ready is False
        assert "OPENAI_API_KEY" in readiness.detail

    def test_openai_repr_does_not_expose_key(self) -> None:
        config = EmbeddingBackendConfig.from_environ(
            {"OPENAI_API_KEY": _OPENAI_KEY},
            requested="openai",
        )
        assert _OPENAI_KEY not in repr(config)

    def test_registry_selectors_include_all_three_backends(self) -> None:
        selectors = production_embedding_registry().selectors
        assert "local-hash" in selectors
        assert "openai" in selectors
        assert "gemini" in selectors

    def test_openai_upstream_still_validates_model(self) -> None:
        config = EmbeddingBackendConfig.from_environ(
            {"OPENAI_API_KEY": _OPENAI_KEY},
            requested="openai",
        )
        upstream = build_openai_embedding_upstream(config)

        with pytest.raises(ValueError, match="text-embedding-3-small"):
            upstream({"model": "wrong-model", "input": "text"})

    def test_openai_upstream_accepts_custom_compatible_base_url(self) -> None:
        config = EmbeddingBackendConfig.from_environ(
            {"OPENAI_API_KEY": _OPENAI_KEY},
            requested="openai",
        )
        upstream = build_openai_embedding_upstream(
            config,
            base_url="http://127.0.0.1:1234/v1/",
        )

        captured: list[Request] = []

        class _Response:
            def __enter__(self) -> _Response:
                return self

            def __exit__(self, *_args: object) -> None:
                pass

            def read(self) -> bytes:
                return b'{"object":"list","data":[]}'

        def fake_urlopen(request: Request, timeout: float = 30) -> _Response:
            del timeout
            captured.append(request)
            return _Response()

        with patch("autobrain.embedding.urllib.request.urlopen", side_effect=fake_urlopen):
            upstream({"model": "text-embedding-3-small", "input": "text"})

        assert captured
        assert captured[0].full_url == "http://127.0.0.1:1234/v1/embeddings"

    def test_default_backend_still_openai(self) -> None:
        config = EmbeddingBackendConfig.from_environ(
            {"OPENAI_API_KEY": _OPENAI_KEY},
        )
        assert config.backend == "openai"

    def test_registry_rejects_duplicate_provenance_backends(self) -> None:
        with pytest.raises(ValueError, match="provenance backends must be unique"):
            EmbeddingBackendRegistry(
                (
                    EmbeddingBackendDescriptor(
                        selector="a",
                        provenance_backend="dup:backend",
                        quality=EmbeddingQuality.SEMANTIC,
                        recommendation_eligible=True,
                        requires_api_key=True,
                        api_key_env="KEY_A",
                    ),
                    EmbeddingBackendDescriptor(
                        selector="b",
                        provenance_backend="dup:backend",
                        quality=EmbeddingQuality.SEMANTIC,
                        recommendation_eligible=True,
                        requires_api_key=True,
                        api_key_env="KEY_B",
                    ),
                )
            )
