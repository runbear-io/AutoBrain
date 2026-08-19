"""Provider-aware authorization-code and refresh orchestration."""

import webbrowser
from collections.abc import Callable
from typing import Protocol
from urllib.parse import urlencode

import httpx
from pydantic import SecretStr

from autobrain.auth.callback import LocalCallback
from autobrain.auth.discovery import discover
from autobrain.auth.models import (
    OAuthClient,
    OAuthError,
    Provider,
    ReauthorizationRequired,
    TokenRecord,
    WorkspaceMismatchError,
)
from autobrain.auth.pkce import create_pkce
from autobrain.auth.providers import config_for
from autobrain.auth.storage import TokenStore
from autobrain.auth.token_response import json_success, parse_token, safe_json


class Callback(Protocol):
    @property
    def redirect_uri(self) -> str: ...

    def wait(self) -> object: ...


def _default_callback(state: str) -> Callback:
    return LocalCallback("127.0.0.1", 8765, state)


class OAuthManager:
    def __init__(
        self,
        store: TokenStore,
        *,
        http: httpx.Client | None = None,
        browser_open: Callable[[str], bool] = webbrowser.open,
        callback_factory: Callable[[str], Callback] | None = None,
        timeout: float = 10.0,
        allow_localhost_http: bool = False,
    ) -> None:
        self.store = store
        self.http = http or httpx.Client(follow_redirects=False, trust_env=False)
        self.browser_open = browser_open
        self.callback_factory: Callable[[str], Callback] = callback_factory or _default_callback
        self.timeout = timeout
        self.allow_localhost_http = allow_localhost_http

    def authorize(
        self,
        provider: Provider,
        *,
        slack_client_id: str | None = None,
        slack_client_secret: str | None = None,
    ) -> TokenRecord:
        config = config_for(provider)
        metadata = discover(
            config.resource,
            self.http,
            timeout=self.timeout,
            allow_localhost_http=self.allow_localhost_http,
        )
        pkce = create_pkce()
        callback = self.callback_factory(pkce.state)
        try:
            oauth_client = self._client(
                provider,
                metadata.registration_endpoint,
                callback.redirect_uri,
                slack_client_id,
                slack_client_secret,
            )
            params = {
                "response_type": "code",
                "client_id": oauth_client.client_id,
                "redirect_uri": callback.redirect_uri,
                "scope": " ".join(config.scopes),
                "state": pkce.state,
                "code_challenge": pkce.challenge,
                "code_challenge_method": "S256",
                "resource": config.resource,
            }
            authorization_url = f"{metadata.authorization_endpoint}?{urlencode(params)}"
            if not self.browser_open(authorization_url):
                raise OAuthError("could not open the authorization URL")
            result = callback.wait()
            code = getattr(result, "code", None)
            state = getattr(result, "state", None)
            if not isinstance(code, str) or state != pkce.state:
                raise OAuthError("invalid callback result")
            body = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": callback.redirect_uri,
                "client_id": oauth_client.client_id,
                "code_verifier": pkce.verifier,
                "resource": config.resource,
            }
            if oauth_client.client_secret:
                body["client_secret"] = oauth_client.client_secret.get_secret_value()
            response = self._post(metadata.token_endpoint, data=body, operation="token exchange")
            token = parse_token(provider, config.resource, response, oauth_client=oauth_client)
            self.store.save(token)
            return token
        finally:
            close = getattr(callback, "close", None)
            if callable(close):
                close()

    def _client(
        self,
        provider: Provider,
        registration_endpoint: str | None,
        redirect_uri: str,
        client_id: str | None,
        client_secret: str | None,
    ) -> OAuthClient:
        if provider is Provider.SLACK:
            if not client_id or not client_secret:
                raise OAuthError("Slack internal-app client credentials are required")
            return OAuthClient(client_id=client_id, client_secret=SecretStr(client_secret))
        if registration_endpoint is None:
            raise OAuthError("Notion authorization server did not advertise client registration")
        response = self._post(
            registration_endpoint,
            json_body={
                "client_name": "AutoBrain",
                "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
            operation="dynamic client registration",
        )
        data = json_success(response, "dynamic client registration")
        registered_id = data.get("client_id")
        if not isinstance(registered_id, str) or not registered_id:
            raise OAuthError("dynamic client registration omitted client_id")
        return OAuthClient(client_id=registered_id)

    def refresh(self, record: TokenRecord, client: OAuthClient | None = None) -> TokenRecord:
        if client is None:
            if record.provider is Provider.SLACK or record.oauth_client_id is None:
                raise OAuthError("OAuth client credentials are required for refresh")
            client = OAuthClient(client_id=record.oauth_client_id)
        refresh = record.refresh_token
        if refresh is None:
            raise ReauthorizationRequired("no refresh token is available")
        with self.store.rotation_lock(record.storage_key):
            current = self.store.get(record.storage_key) or record
            current_refresh = (
                current.refresh_token.get_secret_value() if current.refresh_token else None
            )
            requested_refresh = (
                record.refresh_token.get_secret_value() if record.refresh_token else None
            )
            if current_refresh != requested_refresh:
                return current
            if current.refresh_token is None:
                raise ReauthorizationRequired("no refresh token is available")
            metadata = discover(
                record.audience,
                self.http,
                timeout=self.timeout,
                allow_localhost_http=self.allow_localhost_http,
            )
            body = {
                "grant_type": "refresh_token",
                "refresh_token": current.refresh_token.get_secret_value(),
                "client_id": client.client_id,
                "resource": record.audience,
            }
            if client.client_secret:
                body["client_secret"] = client.client_secret.get_secret_value()
            response = self._post(metadata.token_endpoint, data=body, operation="token refresh")
            if response.status_code >= 400:
                error = safe_json(response).get("error")
                if error == "invalid_grant":
                    self.store.delete(record.storage_key)
                    raise ReauthorizationRequired("refresh grant was revoked or replayed")
                raise OAuthError(f"token refresh failed ({response.status_code})")
            updated = parse_token(
                record.provider,
                record.audience,
                response,
                prior=current,
                oauth_client=client,
            )
            if (updated.workspace_id, updated.user_id) != (record.workspace_id, record.user_id):
                raise WorkspaceMismatchError("refresh response changed workspace or user")
            self.store.save(updated)
            return updated

    def _post(
        self,
        url: str,
        *,
        operation: str,
        data: dict[str, str] | None = None,
        json_body: dict[str, object] | None = None,
    ) -> httpx.Response:
        try:
            return self.http.post(url, data=data, json=json_body, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise OAuthError(f"{operation} failed: {type(exc).__name__}") from exc
