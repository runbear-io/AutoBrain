"""One-release compatibility facade for subscription-backed execution.

New code should import provider contracts, adapters, and runner types from the
specialized ``subscription_*`` modules.
"""

from __future__ import annotations

# Kept as a module attribute for callers that historically patched
# ``autobrain.subscription.shutil.which``. The Codex adapter imports the same
# stdlib module object, so that compatibility hook remains effective.
import shutil as shutil

from autobrain.model_access import (
    ModelAccess,
    ModelAccessRegistry,
    ModelCapability,
)
from autobrain.subscription_claude import (
    ClaudeSubscriptionClient,
    ClaudeSubscriptionConfig,
)
from autobrain.subscription_codex import (
    CodexSubscriptionClient,
    CodexSubscriptionConfig,
)
from autobrain.subscription_domain import (
    AnswerUsage,
    AuthKind,
    ProviderAnswer,
    ProviderCapability,
    ProviderConfig,
    ProviderId,
    ProviderIdentity,
    SubscriptionError,
    SubscriptionFailureReason,
    SubscriptionProvider,
    SubscriptionStatus,
    SubscriptionStatusReport,
    UsageKind,
)
from autobrain.subscription_registry import (
    SubscriptionProviderRegistry,
    UnsupportedSubscriptionProvider,
    provider_registry,
)
from autobrain.subscription_upstream import (
    build_subscription_upstream,
    local_embedding,
)

__all__ = [
    "AnswerUsage",
    "AuthKind",
    "ClaudeSubscriptionClient",
    "ClaudeSubscriptionConfig",
    "CodexSubscriptionClient",
    "CodexSubscriptionConfig",
    "ModelAccess",
    "ModelAccessRegistry",
    "ModelCapability",
    "ProviderAnswer",
    "ProviderCapability",
    "ProviderConfig",
    "ProviderId",
    "ProviderIdentity",
    "SubscriptionError",
    "SubscriptionFailureReason",
    "SubscriptionProvider",
    "SubscriptionProviderRegistry",
    "SubscriptionStatus",
    "SubscriptionStatusReport",
    "UnsupportedSubscriptionProvider",
    "UsageKind",
    "build_subscription_upstream",
    "local_embedding",
    "provider_registry",
]
