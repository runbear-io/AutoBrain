"""Default Textual entrypoint; legacy curses is opt-in only."""

from __future__ import annotations

import os
from collections.abc import Callable

from autobrain.subscription_domain import ProviderId
from autobrain.tui_runtime import ConnectionSnapshot, connection_snapshot
from autobrain.tui_state import TUIState
from autobrain.tui_state import UiScreen as WizardSection


def accepts_key_at_size(key: int, *, width: int, height: int) -> bool:
    return (width >= 60 and height >= 22) or key in {ord("q"), ord("Q")}


def accepts_key_for_state(state: TUIState, key: int, *, width: int, height: int) -> bool:
    if not accepts_key_at_size(key, width=width, height=height):
        return False
    if state.section is WizardSection.RUNNING:
        return key in {ord("q"), ord("Q"), ord("c"), ord("C")}
    return True


def subscription_provider_key(key: int) -> ProviderId | None:
    return {
        ord("1"): ProviderId.CODEX,
        ord("2"): ProviderId.CLAUDE,
        ord("3"): ProviderId.KIMI,
        ord("4"): ProviderId.GROK,
    }.get(key)


def select_subscription_provider(
    state: TUIState,
    key: int,
    *,
    snapshot: Callable[..., ConnectionSnapshot] = connection_snapshot,
) -> tuple[TUIState, ConnectionSnapshot | None]:
    provider = subscription_provider_key(key)
    if provider is None:
        return state, None
    selected = state.with_subscription_provider(provider)
    return selected, snapshot(subscription_provider=provider, refresh_subscription=True)


def run_tui(*, force_setup: bool = False, provider: ProviderId = ProviderId.CODEX) -> None:
    if os.environ.get("AUTOBRAIN_TUI", "textual").casefold() == "legacy":
        from autobrain.tui_legacy import run_tui as run_legacy

        return run_legacy(force_setup=force_setup, provider=provider)
    from autobrain.tui_textual import run_textual

    run_textual(force_setup=force_setup, provider=provider)
