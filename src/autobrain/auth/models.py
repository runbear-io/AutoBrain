"""OAuth and connection-management contracts."""

import time
from enum import StrEnum

from pydantic import Field, SecretStr

from autobrain.models import ConnectionState, Status, StrictModel


class Provider(StrEnum):
    SLACK = "slack"
    NOTION = "notion"


class OAuthMetadata(StrictModel):
    resource: str
    authorization_servers: tuple[str, ...]
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str | None = None
    scopes_supported: tuple[str, ...] = ()


class OAuthClient(StrictModel):
    client_id: str = Field(min_length=1)
    client_secret: SecretStr | None = None


class TokenRecord(StrictModel):
    provider: Provider
    workspace_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    audience: str = Field(pattern=r"^https?://")
    access_token: SecretStr
    refresh_token: SecretStr | None = None
    token_type: str = "Bearer"
    expires_at: int | None = None
    scope: str = ""
    oauth_client_id: str | None = None

    @property
    def storage_key(self) -> str:
        return f"{self.provider.value}:{self.workspace_id}:{self.user_id}"

    def needs_refresh(self, *, skew_seconds: int = 60, now: int | None = None) -> bool:
        current = int(time.time()) if now is None else now
        return self.expires_at is not None and self.expires_at <= current + skew_seconds


class ConnectionStatus(StrictModel):
    provider: Provider
    state: ConnectionState
    status: Status
    workspace_id: str | None = None
    user_id: str | None = None
    storage: str | None = None
    warning: str | None = None


class AuthStatusReport(StrictModel):
    schema_version: int = 1
    connections: tuple[ConnectionStatus, ...]


class OAuthError(RuntimeError):
    """A bounded, user-facing OAuth failure."""


class ConsentDeniedError(OAuthError):
    """The resource owner denied consent."""


class StateMismatchError(OAuthError):
    """The callback did not contain the expected anti-CSRF state."""


class ReauthorizationRequired(OAuthError):
    """The refresh grant is terminal and interactive authorization is required."""


class WorkspaceMismatchError(OAuthError):
    """A token response identified a different workspace or user."""
