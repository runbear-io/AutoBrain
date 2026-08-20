"""Explicit registry and bounded status probes for subscription providers."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import cast

from autobrain.cancellation import RunCancellation
from autobrain.subscription_claude import ClaudeSubscriptionClient, ClaudeSubscriptionConfig
from autobrain.subscription_codex import CodexSubscriptionClient, CodexSubscriptionConfig
from autobrain.subscription_domain import (
    AuthKind,
    ProviderAnswer,
    ProviderId,
    ProviderIdentity,
    SubscriptionError,
    SubscriptionFailureReason,
    SubscriptionProvider,
    SubscriptionStatus,
    SubscriptionStatusReport,
)


@dataclass
class UnsupportedSubscriptionProvider:
    provider_id: ProviderId
    guidance: str

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            provider=self.provider_id,
            model=None,
            cli_version=None,
            auth_kind=AuthKind.UNSUPPORTED,
        )

    @property
    def last_answer(self) -> ProviderAnswer | None:
        return None

    def probe_identity(self) -> ProviderIdentity:
        return self.identity

    def login(self) -> int:
        raise SubscriptionError(
            SubscriptionStatus.UNSUPPORTED,
            self.guidance,
            reason=SubscriptionFailureReason.PROVIDER_UNSUPPORTED,
        )

    def probe_status(self) -> SubscriptionStatusReport:
        return SubscriptionStatusReport(
            status=SubscriptionStatus.UNSUPPORTED,
            reason=SubscriptionFailureReason.PROVIDER_UNSUPPORTED,
            detail=self.guidance,
        )

    def status(self) -> SubscriptionStatus:
        return SubscriptionStatus.UNSUPPORTED

    def answer(self, prompt: str) -> ProviderAnswer:
        del prompt
        raise SubscriptionError(
            SubscriptionStatus.UNSUPPORTED,
            self.guidance,
            reason=SubscriptionFailureReason.PROVIDER_UNSUPPORTED,
        )

    def ask(self, prompt: str) -> str:
        return self.answer(prompt).text


_PROBE_LOCK = threading.Lock()
_ACTIVE_PROVIDERS: set[ProviderId] = set()


@dataclass(frozen=True)
class _CachedProbe:
    report: SubscriptionStatusReport
    monotonic_at: float


@dataclass
class SubscriptionProviderRegistry:
    providers: dict[ProviderId, SubscriptionProvider]
    status_ttl_seconds: float = 5.0
    _cache: dict[ProviderId, _CachedProbe] = field(default_factory=dict, init=False)

    @property
    def provider_ids(self) -> tuple[ProviderId, ...]:
        return tuple(self.providers)

    def get(self, provider: ProviderId | str) -> SubscriptionProvider:
        provider_id = (
            provider if isinstance(provider, ProviderId) else ProviderId(provider.casefold())
        )
        return self.providers[provider_id]

    def probe(
        self,
        provider: ProviderId | str,
        *,
        refresh: bool = False,
    ) -> SubscriptionStatusReport:
        provider_id = (
            provider if isinstance(provider, ProviderId) else ProviderId(provider.casefold())
        )
        now = time.monotonic()
        with _PROBE_LOCK:
            cached = self._cache.get(provider_id)
            if (
                not refresh
                and cached is not None
                and now - cached.monotonic_at <= self.status_ttl_seconds
            ):
                return cached.report
            if provider_id in _ACTIVE_PROVIDERS:
                return SubscriptionStatusReport(
                    status=SubscriptionStatus.SUBSCRIPTION_EXECUTION_UNAVAILABLE,
                    reason=SubscriptionFailureReason.PROBE_IN_PROGRESS,
                    detail=f"{provider_id.value} subscription status probe already in progress",
                )
            _ACTIVE_PROVIDERS.add(provider_id)
        try:
            report = self.get(provider_id).probe_status()
        finally:
            with _PROBE_LOCK:
                _ACTIVE_PROVIDERS.discard(provider_id)
        with _PROBE_LOCK:
            self._cache[provider_id] = _CachedProbe(report=report, monotonic_at=time.monotonic())
        return report

    def invalidate(self, provider: ProviderId | str) -> None:
        provider_id = (
            provider if isinstance(provider, ProviderId) else ProviderId(provider.casefold())
        )
        with _PROBE_LOCK:
            self._cache.pop(provider_id, None)


_KIMI_GUIDANCE = (
    "Kimi Code is unsupported: no verified official local CLI contract was available for "
    "consumer-subscription status/auth distinction, interactive login, tool-free structured "
    "JSON with usage, and bounded cancellation. Install a vendor release exposing all of those "
    "surfaces; AutoBrain will not implement custom OAuth or accept API-key/proxy auth."
)
_GROK_GUIDANCE = (
    "Grok is unsupported: no verified official local CLI contract was available for "
    "consumer-subscription status/auth distinction, interactive login, tool-free structured "
    "JSON with usage, and bounded cancellation. Install a vendor release exposing all of those "
    "surfaces; AutoBrain will not implement custom OAuth or accept API-key/proxy auth."
)


def provider_registry(
    *,
    status_ttl_seconds: float = 5.0,
    cancellation: RunCancellation | None = None,
) -> SubscriptionProviderRegistry:
    providers: dict[ProviderId, SubscriptionProvider] = {
        ProviderId.CODEX: cast(
            SubscriptionProvider,
            CodexSubscriptionClient(
                CodexSubscriptionConfig.from_environ(),
                cancellation=cancellation,
            ),
        ),
        ProviderId.CLAUDE: cast(
            SubscriptionProvider,
            ClaudeSubscriptionClient(
                ClaudeSubscriptionConfig.from_environ(),
                cancellation=cancellation,
            ),
        ),
        ProviderId.KIMI: cast(
            SubscriptionProvider,
            UnsupportedSubscriptionProvider(ProviderId.KIMI, _KIMI_GUIDANCE),
        ),
        ProviderId.GROK: cast(
            SubscriptionProvider,
            UnsupportedSubscriptionProvider(ProviderId.GROK, _GROK_GUIDANCE),
        ),
    }
    return SubscriptionProviderRegistry(
        providers=providers,
        status_ttl_seconds=status_ttl_seconds,
    )
