"""Offline, typed model-access capability reporting.

The resolver only inspects local configuration and bounded subscription status
probes. It never sends model requests, reads secret values, or treats a hash
vector as semantic evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from autobrain.contracts import (
    ModelAccessMode,
    ModelAccessProfileV1,
    ModelCapabilityStatus,
)
from autobrain.embedding import production_embedding_registry
from autobrain.subscription_domain import SubscriptionStatus
from autobrain.subscription_registry import SubscriptionProviderRegistry, provider_registry


class ModelCapability(StrEnum):
    """Provider capability requested by an OpenAI-compatible boundary."""

    CHAT = "chat"
    SEMANTIC_EMBEDDING = "semantic_embedding"
    SMOKE_EMBEDDING = "smoke_embedding"


ModelAccessHandler = Callable[[dict[str, object]], dict[str, object]]


@dataclass(frozen=True)
class ModelAccess:
    """One trusted implementation of a model capability."""

    capability: ModelCapability
    provider: str
    handler: ModelAccessHandler


class ModelAccessUnavailable(ValueError):
    """Raised when a requested capability has no registered implementation."""


@dataclass(frozen=True)
class ModelAccessRegistry:
    """Immutable capability registry; capabilities are never inferred from names."""

    entries: Mapping[ModelCapability, ModelAccess]

    def __post_init__(self) -> None:
        for capability, access in self.entries.items():
            if capability is not access.capability:
                raise ValueError("model access entry capability does not match its key")
            if not access.provider.strip():
                raise ValueError("model access provider must not be empty")

    def resolve(self, capability: ModelCapability) -> ModelAccess | None:
        return self.entries.get(capability)

    def require(self, capability: ModelCapability) -> ModelAccess:
        access = self.resolve(capability)
        if access is None:
            raise ModelAccessUnavailable(
                f"{capability.value} capability is unavailable; no model access is registered"
            )
        return access

    def with_access(self, access: ModelAccess) -> ModelAccessRegistry:
        entries = dict(self.entries)
        if access.capability in entries:
            raise ValueError(f"model access already registered for {access.capability.value}")
        entries[access.capability] = access
        return ModelAccessRegistry(entries)


EMPTY_MODEL_ACCESS = ModelAccessRegistry({})


@dataclass(frozen=True)
class ModelAccessStatus:
    """Typed aggregate status exposed by the CLI and safe for JSON serialization."""

    profiles: tuple[ModelAccessProfileV1, ...]

    @property
    def recommendation_eligible(self) -> bool:
        return any(profile.recommendation_eligible for profile in self.profiles)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "recommendation_eligible": self.recommendation_eligible,
            "profiles": [profile.model_dump(mode="json") for profile in self.profiles],
        }


def _subscription_profile(
    environ: Mapping[str, str],
    subscriptions: SubscriptionProviderRegistry,
) -> ModelAccessProfileV1:
    del environ  # Provider clients own their bounded local help/version probes.
    reports = [subscriptions.probe(provider) for provider in subscriptions.provider_ids]
    ready = any(report.status is SubscriptionStatus.READY for report in reports)
    diagnostics = [] if ready else ["chat_unavailable"]
    if not ready:
        diagnostics.extend(
            f"subscription_{report.status.value.lower()}" for report in reports if report.detail
        )
    return ModelAccessProfileV1(
        schema_version=1,
        mode=ModelAccessMode.SUBSCRIPTION_CLI,
        chat=ModelCapabilityStatus.READY if ready else ModelCapabilityStatus.UNAVAILABLE,
        embeddings=ModelCapabilityStatus.UNAVAILABLE,
        verifier=ModelCapabilityStatus.READY if ready else ModelCapabilityStatus.UNAVAILABLE,
        metering="COST_INCOMPLETE" if ready else "COST_UNAVAILABLE",
        diagnostics=diagnostics,
    )


def _byok_profile(environ: Mapping[str, str]) -> ModelAccessProfileV1:
    # Presence-only checks deliberately avoid reading secret values. A key's
    # validity remains an external/provider execution concern.
    selector = environ.get("AUTOBRAIN_EMBEDDING_BACKEND", "openai").casefold()
    descriptor = production_embedding_registry().resolve_selector(selector)
    embedding_ready = bool(
        descriptor is not None
        and descriptor.recommendation_eligible
        and (descriptor.api_key_env is None or descriptor.api_key_env in environ)
    )
    has_chat_key = "OPENAI_API_KEY" in environ or "GEMINI_API_KEY" in environ
    diagnostics: list[str] = []
    if descriptor is None:
        diagnostics.append("embedding_config_invalid")
    elif not embedding_ready:
        diagnostics.append("semantic_embeddings_unavailable")
    if not has_chat_key:
        diagnostics.append("chat_unavailable")
    ready = has_chat_key and embedding_ready
    return ModelAccessProfileV1(
        schema_version=1,
        mode=ModelAccessMode.PROVIDER_API_BYOK,
        chat=ModelCapabilityStatus.READY if has_chat_key else ModelCapabilityStatus.UNAVAILABLE,
        embeddings=ModelCapabilityStatus.READY
        if embedding_ready
        else ModelCapabilityStatus.UNAVAILABLE,
        verifier=ModelCapabilityStatus.READY if ready else ModelCapabilityStatus.UNAVAILABLE,
        metering="COST_INCOMPLETE" if ready else "COST_UNAVAILABLE",
        recommendation_eligible=ready,
        diagnostics=diagnostics,
    )


def _local_profile(environ: Mapping[str, str]) -> ModelAccessProfileV1:
    del environ
    return ModelAccessProfileV1(
        schema_version=1,
        mode=ModelAccessMode.LOCAL_OPENAI_COMPATIBLE,
        chat=ModelCapabilityStatus.NOT_CONFIGURED,
        embeddings=ModelCapabilityStatus.METERING_INCOMPLETE,
        verifier=ModelCapabilityStatus.UNAVAILABLE,
        metering="COST_UNAVAILABLE",
        keyword_only=True,
        smoke_only_hash=True,
        diagnostics=["local_chat_not_configured", "hash_embeddings_smoke_only"],
    )


def inspect_model_access(
    environ: Mapping[str, str],
    *,
    subscriptions: SubscriptionProviderRegistry | None = None,
) -> ModelAccessStatus:
    """Build the complete local fallback matrix without model/network calls."""
    registry = subscriptions or provider_registry()
    return ModelAccessStatus(
        profiles=(
            _byok_profile(environ),
            _subscription_profile(environ, registry),
            _local_profile(environ),
        )
    )


def render_model_access_human(status: ModelAccessStatus) -> str:
    lines = [
        "AutoBrain model access: "
        + (
            "recommendation eligible"
            if status.recommendation_eligible
            else "not recommendation eligible"
        )
    ]
    for profile in status.profiles:
        lines.append(f"{profile.mode.value}: {', '.join(profile.diagnostics) or 'ready'}")
        lines.append(
            f"  chat={profile.chat.value} embeddings={profile.embeddings.value} "
            f"verifier={profile.verifier.value} metering={profile.metering} "
            f"keyword_only={profile.keyword_only} smoke_only_hash={profile.smoke_only_hash} "
            f"recommendation_eligible={profile.recommendation_eligible}"
        )
    return "\n".join(lines)
