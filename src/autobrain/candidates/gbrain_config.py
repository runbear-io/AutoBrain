"""Typed, run-local GBrain embedding and capability configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from ipaddress import ip_address
from urllib.parse import urlsplit

from pydantic import SecretStr


class GBrainEmbeddingProvider(StrEnum):
    KEYWORD_ONLY = "keyword-only"
    OPENAI = "openai"
    VOYAGE = "voyage"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    LLAMA_SERVER = "llama-server"


@dataclass(frozen=True)
class GBrainProviderSpec:
    provider: GBrainEmbeddingProvider
    model: str | None
    dimensions: int | None
    requires_key: bool
    endpoint: str | None = None


_DEFAULTS: dict[GBrainEmbeddingProvider, GBrainProviderSpec] = {
    GBrainEmbeddingProvider.OPENAI: GBrainProviderSpec(
        GBrainEmbeddingProvider.OPENAI, "text-embedding-3-small", 1536, True
    ),
    GBrainEmbeddingProvider.VOYAGE: GBrainProviderSpec(
        GBrainEmbeddingProvider.VOYAGE, "voyage-4", 1024, True
    ),
    GBrainEmbeddingProvider.GEMINI: GBrainProviderSpec(
        GBrainEmbeddingProvider.GEMINI, "gemini-embedding-001", 768, True
    ),
    GBrainEmbeddingProvider.OPENROUTER: GBrainProviderSpec(
        GBrainEmbeddingProvider.OPENROUTER, "openai/text-embedding-3-small", 1536, True
    ),
    GBrainEmbeddingProvider.OLLAMA: GBrainProviderSpec(
        GBrainEmbeddingProvider.OLLAMA, "nomic-embed-text", 768, False, "http://127.0.0.1:11434"
    ),
}


@dataclass(frozen=True)
class GBrainExecutionConfig:
    """Immutable run configuration; ``credential`` never participates in repr/artifacts."""

    embedding: GBrainProviderSpec
    credential: SecretStr | None = None
    chat_provider: str | None = None
    chat_model: str | None = None
    chat_credential: SecretStr | None = None

    @classmethod
    def quick_start(cls) -> GBrainExecutionConfig:
        return cls(GBrainProviderSpec(GBrainEmbeddingProvider.KEYWORD_ONLY, None, None, False))

    @classmethod
    def semantic(
        cls,
        provider: GBrainEmbeddingProvider | str,
        *,
        model: str | None = None,
        dimensions: int | None = None,
        endpoint: str | None = None,
        credential: str | None = None,
        chat_credential: str | None = None,
        chat_model: str = "openai:gpt-5-mini",
    ) -> GBrainExecutionConfig:
        selected = GBrainEmbeddingProvider(provider)
        if selected is GBrainEmbeddingProvider.KEYWORD_ONLY:
            raise ValueError("keyword-only is not a semantic provider")
        if selected is GBrainEmbeddingProvider.LLAMA_SERVER:
            if not model:
                raise ValueError("llama-server requires an explicit model")
            if dimensions is None or dimensions <= 0:
                raise ValueError("llama-server requires positive dimensions")
            spec = GBrainProviderSpec(selected, model, dimensions, False, endpoint)
        else:
            default = _DEFAULTS[selected]
            spec = GBrainProviderSpec(
                selected,
                model or default.model,
                dimensions or default.dimensions,
                default.requires_key,
                endpoint or default.endpoint,
            )
        if spec.requires_key and not credential:
            raise ValueError(f"{selected.value} requires a credential")
        if spec.endpoint is not None:
            validate_endpoint(spec.endpoint)
        return cls(
            spec,
            SecretStr(credential) if credential else None,
            "openai" if chat_credential else None,
            chat_model if chat_credential else None,
            SecretStr(chat_credential) if chat_credential else None,
        )

    @property
    def keyword_only(self) -> bool:
        return self.embedding.provider is GBrainEmbeddingProvider.KEYWORD_ONLY

    @property
    def semantic_enabled(self) -> bool:
        return not self.keyword_only

    @property
    def recommendation_eligible(self) -> bool:
        return (
            self.semantic_enabled
            and self.chat_provider == "openai"
            and self.chat_credential is not None
        )

    def safe_metadata(self) -> dict[str, object]:
        return {
            "keyword_only": self.keyword_only,
            "semantic_enabled": self.semantic_enabled,
            "semantic_quality": "not_measured" if self.keyword_only else "configured",
            "recommendation_eligible": self.recommendation_eligible,
            "provider": self.embedding.provider.value,
            "model": self.embedding.model,
            "dimensions": self.embedding.dimensions,
            "endpoint": self.embedding.endpoint,
            "capabilities": {
                "embedding": True,
                "chat": self.chat_provider == "openai",
                "think": False,
            },
        }

    def child_environment(self) -> dict[str, str]:
        if self.credential is None:
            return {}
        names = {
            GBrainEmbeddingProvider.OPENAI: "OPENAI_API_KEY",
            GBrainEmbeddingProvider.VOYAGE: "VOYAGE_API_KEY",
            GBrainEmbeddingProvider.GEMINI: "GEMINI_API_KEY",
            GBrainEmbeddingProvider.OPENROUTER: "OPENROUTER_API_KEY",
        }
        name = names.get(self.embedding.provider)
        return {name: self.credential.get_secret_value()} if name else {}


def validate_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint must be an HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint userinfo is not permitted")
    try:
        port = parsed.port
        del port
        ip_address(parsed.hostname)
    except ValueError:
        if any(character.isspace() for character in parsed.hostname):
            raise ValueError("endpoint host is invalid") from None
    return endpoint


def provider_specs() -> tuple[GBrainProviderSpec, ...]:
    return tuple(_DEFAULTS.values())
