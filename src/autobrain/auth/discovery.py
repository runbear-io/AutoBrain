"""RFC 9728 / RFC 8414 OAuth metadata discovery."""

from typing import cast
from urllib.parse import urljoin, urlparse

import httpx

from autobrain.auth.models import OAuthError, OAuthMetadata


def _secure_url(value: object, *, localhost_http: bool) -> str:
    if not isinstance(value, str):
        raise OAuthError("metadata URL must be a string")
    parsed = urlparse(value)
    local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme != "https" and not (localhost_http and local and parsed.scheme == "http"):
        raise OAuthError("metadata URL must use HTTPS")
    if parsed.username or parsed.password or not parsed.netloc:
        raise OAuthError("malformed metadata URL")
    return value


def _well_known(resource: str) -> str:
    parsed = urlparse(resource)
    suffix = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}/.well-known/oauth-protected-resource{suffix}"


def _get(client: httpx.Client, url: str, timeout: float, operation: str) -> httpx.Response:
    try:
        return client.get(url, timeout=timeout)
    except httpx.HTTPError as exc:
        raise OAuthError(f"{operation} failed: {type(exc).__name__}") from exc


def discover(
    resource: str,
    client: httpx.Client,
    *,
    timeout: float = 5.0,
    allow_localhost_http: bool = False,
) -> OAuthMetadata:
    resource = _secure_url(resource, localhost_http=allow_localhost_http)
    response = _get(client, _well_known(resource), timeout, "protected-resource metadata")
    if response.status_code != 200:
        raise OAuthError(f"protected-resource metadata failed ({response.status_code})")
    try:
        protected = cast(dict[str, object], response.json())
    except ValueError as exc:
        raise OAuthError("protected-resource metadata was not JSON") from exc
    if protected.get("resource") != resource:
        raise OAuthError("protected-resource audience mismatch")
    servers = protected.get("authorization_servers")
    if not isinstance(servers, list) or not servers:
        raise OAuthError("metadata has no authorization server")
    raw_servers = cast(list[object], servers)
    if not all(isinstance(item, str) for item in raw_servers):
        raise OAuthError("metadata has no authorization server")
    typed_servers = cast(list[str], raw_servers)
    server = _secure_url(typed_servers[0], localhost_http=allow_localhost_http).rstrip("/")
    metadata_url = f"{server}/.well-known/oauth-authorization-server"
    auth_response = _get(client, metadata_url, timeout, "authorization metadata")
    if auth_response.status_code != 200:
        oidc_url = urljoin(f"{server}/", ".well-known/openid-configuration")
        auth_response = _get(client, oidc_url, timeout, "OpenID metadata")
    if auth_response.status_code != 200:
        raise OAuthError(f"authorization metadata failed ({auth_response.status_code})")
    try:
        auth = cast(dict[str, object], auth_response.json())
    except ValueError as exc:
        raise OAuthError("authorization metadata was not JSON") from exc
    issuer = _secure_url(auth.get("issuer"), localhost_http=allow_localhost_http).rstrip("/")
    if issuer != server:
        raise OAuthError("authorization-server issuer mismatch")
    endpoint = _secure_url(auth.get("authorization_endpoint"), localhost_http=allow_localhost_http)
    token = _secure_url(auth.get("token_endpoint"), localhost_http=allow_localhost_http)
    registration_raw = auth.get("registration_endpoint")
    registration = (
        _secure_url(registration_raw, localhost_http=allow_localhost_http)
        if registration_raw is not None
        else None
    )
    raw_scopes = auth.get("scopes_supported", [])
    if not isinstance(raw_scopes, list):
        raise OAuthError("invalid scopes_supported metadata")
    scope_values = cast(list[object], raw_scopes)
    if not all(isinstance(item, str) for item in scope_values):
        raise OAuthError("invalid scopes_supported metadata")
    scopes = cast(list[str], scope_values)
    return OAuthMetadata(
        resource=resource,
        authorization_servers=tuple(typed_servers),
        authorization_endpoint=endpoint,
        token_endpoint=token,
        registration_endpoint=registration,
        scopes_supported=tuple(scopes),
    )
