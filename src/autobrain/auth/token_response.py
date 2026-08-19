"""Strict OAuth token-response parsing and identity binding."""

import time
from typing import cast

import httpx
from pydantic import SecretStr

from autobrain.auth.models import OAuthClient, OAuthError, Provider, TokenRecord


def safe_json(response: httpx.Response) -> dict[str, object]:
    try:
        data = response.json()
    except ValueError:
        return {}
    return cast(dict[str, object], data) if isinstance(data, dict) else {}


def json_success(response: httpx.Response, operation: str) -> dict[str, object]:
    if response.status_code < 200 or response.status_code >= 300:
        raise OAuthError(f"{operation} failed ({response.status_code})")
    data = safe_json(response)
    if not data:
        raise OAuthError(f"{operation} returned malformed JSON")
    return data


def parse_token(
    provider: Provider,
    audience: str,
    response: httpx.Response,
    *,
    prior: TokenRecord | None = None,
    oauth_client: OAuthClient | None = None,
) -> TokenRecord:
    data = json_success(response, "token exchange")
    actual_audience = data.get("audience", data.get("resource", audience))
    if actual_audience != audience:
        raise OAuthError("token audience mismatch")
    access = data.get("access_token")
    if not isinstance(access, str) or not access:
        raise OAuthError("token response omitted access_token")
    workspace, user = _identity(provider, data, prior)
    expires_in = data.get("expires_in")
    expires_at = int(time.time()) + expires_in if isinstance(expires_in, int) else None
    refresh = data.get("refresh_token")
    if refresh is None and prior and prior.refresh_token:
        refresh = prior.refresh_token.get_secret_value()
    return TokenRecord(
        provider=provider,
        workspace_id=workspace,
        user_id=user,
        audience=audience,
        access_token=SecretStr(access),
        refresh_token=SecretStr(refresh) if isinstance(refresh, str) and refresh else None,
        token_type=str(data.get("token_type", "Bearer")),
        expires_at=expires_at,
        scope=str(data.get("scope", "")),
        oauth_client_id=(
            oauth_client.client_id
            if oauth_client is not None
            else (prior.oauth_client_id if prior else None)
        ),
    )


def _identity(
    provider: Provider,
    data: dict[str, object],
    prior: TokenRecord | None,
) -> tuple[str, str]:
    if provider is Provider.SLACK:
        team = data.get("team")
        user = data.get("authed_user")
        team_data = cast(dict[str, object], team) if isinstance(team, dict) else {}
        user_data = cast(dict[str, object], user) if isinstance(user, dict) else {}
        workspace = team_data.get("id", data.get("workspace_id"))
        user_id = user_data.get("id", data.get("user_id"))
    else:
        owner = data.get("owner")
        owner_data = cast(dict[str, object], owner) if isinstance(owner, dict) else {}
        notion_raw = owner_data.get("user")
        notion_user = cast(dict[str, object], notion_raw) if isinstance(notion_raw, dict) else {}
        workspace = data.get("workspace_id")
        user_id = notion_user.get("id", data.get("user_id"))
    workspace = workspace or (prior.workspace_id if prior else None)
    user_id = user_id or (prior.user_id if prior else None)
    if (
        not isinstance(workspace, str)
        or not workspace
        or not isinstance(user_id, str)
        or not user_id
    ):
        raise OAuthError("token response omitted workspace/user identity")
    return workspace, user_id
