"""CLI-facing secure connection management."""

from pathlib import Path

from autobrain.auth.models import (
    AuthStatusReport,
    ConnectionStatus,
    OAuthError,
    Provider,
    TokenRecord,
)
from autobrain.auth.providers import config_for
from autobrain.auth.storage import TokenStore
from autobrain.models import ConnectionState, Status


class ConnectionManager:
    def __init__(self, state_root: Path, *, store: TokenStore | None = None) -> None:
        self.store = store or TokenStore(state_root / "auth")

    def status(self) -> AuthStatusReport:
        storage_warning: str | None = None
        try:
            connected = {item.provider: item for item in self.store.statuses()}
        except OAuthError:
            connected = {}
            storage_warning = "OAuth storage is unavailable; credentials were not modified"
        rows = tuple(
            connected.get(provider)
            or ConnectionStatus(
                provider=provider,
                state=ConnectionState.DISCONNECTED,
                status=(
                    Status.MCP_AUTH_UNAVAILABLE
                    if config_for(provider).supported
                    else Status.CAPABILITY_UNAVAILABLE
                ),
                warning=storage_warning or (config_for(provider).detail or None),
            )
            for provider in Provider
        )
        return AuthStatusReport(connections=rows).with_projections()

    def token_for(self, provider: Provider) -> TokenRecord | None:
        """Return one valid stored token without initiating OAuth or network access."""
        try:
            entries = self.store.files.read_mapping(self.store.index)
        except OAuthError:
            return None
        for key, value in sorted(entries.items()):
            if not isinstance(value, dict) or value.get("provider") != provider.value:
                continue
            try:
                token = self.store.get(key)
            except OAuthError:
                return None
            if token is not None and token.provider is provider and not token.needs_refresh():
                return token
        return None

    def logout(self, provider: Provider) -> int:
        index = self.store.index
        if not index.exists():
            return 0
        entries = self.store.files.read_mapping(index)
        keys = [
            key
            for key, value in entries.items()
            if isinstance(value, dict) and value.get("provider") == provider.value
        ]
        for key in keys:
            self.store.delete(key)
        return len(keys)
