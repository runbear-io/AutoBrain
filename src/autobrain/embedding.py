"""Explicit embedding configuration and closed recommendation capabilities."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

from pydantic import SecretStr

from autobrain.models import (
    CheckResult,
    EmbeddingProvenance,
    EmbeddingQuality,
    Status,
)

_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
_GEMINI_EMBEDDING_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_LOCAL_HASH_BACKEND = "local-hash-embedding"
_TEST_SEMANTIC_SELECTOR = "test-semantic"
_TEST_SEMANTIC_BACKEND = "test:semantic-fixture"


class EmbeddingBackend(StrEnum):
    """Production embedding selectors accepted by explicit configuration."""

    LOCAL_HASH = "local-hash"
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass(frozen=True)
class EmbeddingBackendDescriptor:
    """Canonical capability associated with one configured backend identity."""

    selector: str
    provenance_backend: str
    quality: EmbeddingQuality
    recommendation_eligible: bool
    requires_api_key: bool
    openai_transport: bool = False
    gemini_transport: bool = False
    api_key_env: str | None = None

    @property
    def provenance(self) -> EmbeddingProvenance:
        return EmbeddingProvenance(
            backend=self.provenance_backend,
            quality=self.quality,
        )


@dataclass(frozen=True)
class EmbeddingBackendRegistry:
    """Immutable registry used to resolve backend identity to trusted capability."""

    descriptors: tuple[EmbeddingBackendDescriptor, ...]

    def __post_init__(self) -> None:
        selectors = [descriptor.selector for descriptor in self.descriptors]
        backends = [descriptor.provenance_backend for descriptor in self.descriptors]
        if len(selectors) != len(set(selectors)):
            raise ValueError("embedding backend selectors must be unique")
        if len(backends) != len(set(backends)):
            raise ValueError("embedding provenance backends must be unique")
        for descriptor in self.descriptors:
            if descriptor.recommendation_eligible is not (
                descriptor.quality is EmbeddingQuality.SEMANTIC
            ):
                raise ValueError("recommendation-eligible embedding backends must be semantic")

    @property
    def selectors(self) -> tuple[str, ...]:
        return tuple(descriptor.selector for descriptor in self.descriptors)

    def resolve_selector(self, selector: str) -> EmbeddingBackendDescriptor | None:
        normalized = selector.casefold()
        return next(
            (descriptor for descriptor in self.descriptors if descriptor.selector == normalized),
            None,
        )

    def resolve_provenance_backend(
        self,
        backend: str | None,
    ) -> EmbeddingBackendDescriptor | None:
        if backend is None:
            return None
        return next(
            (
                descriptor
                for descriptor in self.descriptors
                if descriptor.provenance_backend == backend
            ),
            None,
        )

    def with_test_semantic_backend(self) -> EmbeddingBackendRegistry:
        """Return an explicitly test-scoped registry with one semantic fixture backend."""
        if self.resolve_selector(_TEST_SEMANTIC_SELECTOR) is not None:
            return self
        return EmbeddingBackendRegistry(
            (
                *self.descriptors,
                EmbeddingBackendDescriptor(
                    selector=_TEST_SEMANTIC_SELECTOR,
                    provenance_backend=_TEST_SEMANTIC_BACKEND,
                    quality=EmbeddingQuality.SEMANTIC,
                    recommendation_eligible=True,
                    requires_api_key=False,
                ),
            )
        )


_PRODUCTION_EMBEDDING_REGISTRY = EmbeddingBackendRegistry(
    (
        EmbeddingBackendDescriptor(
            selector=EmbeddingBackend.LOCAL_HASH.value,
            provenance_backend=_LOCAL_HASH_BACKEND,
            quality=EmbeddingQuality.SMOKE_ONLY,
            recommendation_eligible=False,
            requires_api_key=False,
        ),
        EmbeddingBackendDescriptor(
            selector=EmbeddingBackend.OPENAI.value,
            provenance_backend=f"openai:{_OPENAI_EMBEDDING_MODEL}",
            quality=EmbeddingQuality.SEMANTIC,
            recommendation_eligible=True,
            requires_api_key=True,
            openai_transport=True,
            api_key_env="OPENAI_API_KEY",
        ),
        EmbeddingBackendDescriptor(
            selector=EmbeddingBackend.GEMINI.value,
            provenance_backend=f"google:{_GEMINI_EMBEDDING_MODEL}",
            quality=EmbeddingQuality.SEMANTIC,
            recommendation_eligible=True,
            requires_api_key=True,
            gemini_transport=True,
            api_key_env="GEMINI_API_KEY",
        ),
    )
)


def production_embedding_registry() -> EmbeddingBackendRegistry:
    return _PRODUCTION_EMBEDDING_REGISTRY


@dataclass(frozen=True)
class EmbeddingBackendConfig:
    """Run-local embedding selection with secrets excluded from representation."""

    descriptor: EmbeddingBackendDescriptor
    registry: EmbeddingBackendRegistry = field(repr=False)
    api_key: SecretStr | None = field(default=None, repr=False)

    @classmethod
    def from_environ(
        cls,
        environ: Mapping[str, str],
        *,
        requested: EmbeddingBackend | str | None = None,
        api_key: str | None = None,
        registry: EmbeddingBackendRegistry | None = None,
    ) -> EmbeddingBackendConfig:
        selected_registry = registry or production_embedding_registry()
        raw_backend = (
            requested.value
            if isinstance(requested, EmbeddingBackend)
            else requested
            if requested is not None
            else environ.get(
                "AUTOBRAIN_EMBEDDING_BACKEND",
                EmbeddingBackend.OPENAI.value,
            )
        )
        descriptor = selected_registry.resolve_selector(raw_backend)
        if descriptor is None:
            choices = ", ".join(selected_registry.selectors)
            raise ValueError(f"AUTOBRAIN_EMBEDDING_BACKEND must be one of: {choices}")
        resolved_key = (
            api_key
            if api_key is not None
            else environ.get(descriptor.api_key_env)
            if descriptor.api_key_env
            else None
        )
        return cls(
            descriptor=descriptor,
            registry=selected_registry,
            api_key=SecretStr(resolved_key) if resolved_key else None,
        )

    @property
    def backend(self) -> str:
        return self.descriptor.selector

    @property
    def provenance(self) -> EmbeddingProvenance:
        return self.descriptor.provenance

    @property
    def recommendation_ready(self) -> bool:
        return self.descriptor.recommendation_eligible and (
            not self.descriptor.requires_api_key or self.api_key is not None
        )

    @property
    def candidate_api_key(self) -> str:
        if self.api_key is not None:
            return self.api_key.get_secret_value()
        return (
            "autobrain-test-semantic" if self.recommendation_ready else "autobrain-local-hash-smoke"
        )

    def readiness(self) -> EmbeddingReadiness:
        descriptor = self.descriptor
        provenance = descriptor.provenance
        if not descriptor.recommendation_eligible:
            return EmbeddingReadiness(
                backend=descriptor.selector,
                status=Status.CAPABILITY_UNAVAILABLE,
                detail=(
                    "SMOKE_ONLY: local-hash-embedding can exercise the pipeline but cannot "
                    "produce a recommendation"
                ),
                provenance=provenance,
                recommendation_ready=False,
            )
        if descriptor.requires_api_key and self.api_key is None:
            env_name = descriptor.api_key_env or "API_KEY"
            return EmbeddingReadiness(
                backend=descriptor.selector,
                status=Status.MISSING_PROVIDER,
                detail=(
                    f"SEMANTIC_EMBEDDING_UNAVAILABLE: {env_name} is required for "
                    f"{descriptor.provenance_backend}"
                ),
                provenance=provenance,
                recommendation_ready=False,
            )
        return EmbeddingReadiness(
            backend=descriptor.selector,
            status=Status.OK,
            detail=f"READY: {descriptor.provenance_backend} (semantic)",
            provenance=provenance,
            recommendation_ready=True,
        )


@dataclass(frozen=True)
class EmbeddingReadiness:
    backend: str | None
    status: Status
    detail: str
    provenance: EmbeddingProvenance
    recommendation_ready: bool


def inspect_embedding_backend(
    environ: Mapping[str, str],
    *,
    requested: EmbeddingBackend | str | None = None,
    api_key: str | None = None,
    registry: EmbeddingBackendRegistry | None = None,
) -> EmbeddingReadiness:
    """Resolve current configuration without retaining stale prior state."""
    try:
        config = EmbeddingBackendConfig.from_environ(
            environ,
            requested=requested,
            api_key=api_key,
            registry=registry,
        )
    except ValueError as error:
        return EmbeddingReadiness(
            backend=None,
            status=Status.ENV_UNAVAILABLE,
            detail=f"EMBEDDING_CONFIG_INVALID: {error}",
            provenance=EmbeddingProvenance(),
            recommendation_ready=False,
        )
    return config.readiness()


def check_embedding_backend(environ: Mapping[str, str]) -> CheckResult:
    readiness = inspect_embedding_backend(environ)
    return CheckResult(
        name="semantic_embeddings",
        status=readiness.status,
        detail=readiness.detail,
    )


def recommendation_eligibility_reason(
    embedding: EmbeddingProvenance,
    *,
    registry: EmbeddingBackendRegistry | None = None,
) -> str | None:
    """Resolve recommendation capability from registered identity, never claimed quality."""
    selected_registry = registry or production_embedding_registry()
    descriptor = selected_registry.resolve_provenance_backend(embedding.backend)
    backend = embedding.backend or "unavailable"
    if descriptor is None:
        return (
            "recommendation requires semantic embeddings; configured backend "
            f"{backend} is not registered"
        )
    if not descriptor.recommendation_eligible:
        return (
            "recommendation requires semantic embeddings; configured backend "
            f"{descriptor.provenance_backend} is smoke-only"
        )
    return None


def build_openai_embedding_upstream(
    config: EmbeddingBackendConfig,
) -> Callable[[dict[str, object]], dict[str, object]]:
    """Build the explicit OpenAI semantic-embedding request boundary."""
    if not config.descriptor.openai_transport or config.api_key is None:
        raise ValueError("semantic OpenAI embedding backend is not ready")
    api_key = config.api_key.get_secret_value()

    def request(payload: dict[str, object]) -> dict[str, object]:
        model = payload.get("model")
        if model != _OPENAI_EMBEDDING_MODEL:
            raise ValueError(
                f"semantic embedding upstream requires model {_OPENAI_EMBEDDING_MODEL}"
            )
        outgoing = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(outgoing, timeout=30) as response:
                value = json.loads(response.read())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"semantic embedding request failed ({type(error).__name__})"
            ) from None
        if not isinstance(value, dict):
            raise RuntimeError("semantic embedding provider returned a non-object response")
        return cast(dict[str, Any], value)

    return request


def build_gemini_embedding_upstream(
    config: EmbeddingBackendConfig,
) -> Callable[[dict[str, object]], dict[str, object]]:
    """Build the explicit Google Gemini semantic-embedding request boundary.

    Accepts an OpenAI-style embedding payload (``model`` + ``input``) and
    translates it to the Gemini ``embedContent`` / ``batchEmbedContent`` REST
    API, returning an OpenAI-compatible response envelope so callers can treat
    the upstream uniformly.  No live calls are made at construction time.
    """
    if not config.descriptor.gemini_transport or config.api_key is None:
        raise ValueError("semantic Gemini embedding backend is not ready")
    api_key = config.api_key.get_secret_value()
    model = _GEMINI_EMBEDDING_MODEL
    base = _GEMINI_EMBEDDING_API_BASE

    def request(payload: dict[str, object]) -> dict[str, object]:
        raw_input = payload.get("input")
        if isinstance(raw_input, str):
            texts = [raw_input]
        elif isinstance(raw_input, list):
            texts = [item for item in cast(list[object], raw_input) if isinstance(item, str)]
        else:
            raise ValueError("gemini embedding upstream requires 'input' string or list")
        if not texts:
            raise ValueError("gemini embedding upstream requires at least one input text")

        if len(texts) == 1:
            url = f"{base}/{model}:embedContent"
            body = json.dumps(
                {
                    "content": {"parts": [{"text": texts[0]}]},
                    "taskType": "RETRIEVAL_DOCUMENT",
                }
            ).encode("utf-8")
        else:
            url = f"{base}/{model}:batchEmbedContent"
            body = json.dumps(
                {
                    "requests": [
                        {
                            "content": {"parts": [{"text": text}]},
                            "taskType": "RETRIEVAL_DOCUMENT",
                        }
                        for text in texts
                    ],
                }
            ).encode("utf-8")

        outgoing = urllib.request.Request(
            url,
            data=body,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(outgoing, timeout=30) as response:
                value = json.loads(response.read())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"semantic embedding request failed ({type(error).__name__})"
            ) from None
        if not isinstance(value, dict):
            raise RuntimeError("semantic embedding provider returned a non-object response")

        # Translate Gemini response to OpenAI-compatible envelope.
        response = cast(dict[str, object], value)
        if len(texts) == 1:
            embedding_obj = response.get("embedding")
            vectors = (
                [cast(list[float], cast(dict[str, object], embedding_obj)["values"])]
                if isinstance(embedding_obj, dict)
                else []
            )
        else:
            embeddings = response.get("embeddings")
            vectors = (
                [
                    cast(list[float], item["values"])
                    for item in cast(list[dict[str, object]], embeddings)
                ]
                if isinstance(embeddings, list)
                else []
            )
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": vec}
                for i, vec in enumerate(vectors)
            ],
            "model": f"google:{model}",
            "_autobrain_provider": "gemini",
            "_autobrain_usage_source": "measured",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return request
