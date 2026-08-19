"""Official-SDK Streamable HTTP transport behind a read-only wrapper."""

from collections.abc import Callable
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Self

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from autobrain.auth.models import OAuthClient, Provider, ReauthorizationRequired, TokenRecord
from autobrain.auth.providers import config_for
from autobrain.mcp.policy import ReadOnlyToolPolicy, ToolSnapshot, UntrustedToolResult

if TYPE_CHECKING:
    from autobrain.auth.oauth import OAuthManager


class AudienceError(PermissionError):
    """A token was presented to a resource other than its bound audience."""


class StreamableHttpConnection:
    def __init__(
        self,
        provider: Provider,
        endpoint: str,
        token: TokenRecord,
        *,
        timeout: float = 10.0,
        expected_endpoint: str | None = None,
        http_transport: httpx.AsyncBaseTransport | None = None,
        token_loader: Callable[[str], TokenRecord | None] | None = None,
        token_refresher: Callable[[TokenRecord], TokenRecord] | None = None,
        refresh_skew_seconds: int = 60,
    ) -> None:
        expected = expected_endpoint or config_for(provider).resource
        if token.provider is not provider or token.audience != endpoint or endpoint != expected:
            raise AudienceError("OAuth token provider/audience does not match the MCP resource")
        self.provider, self.endpoint, self.token, self.timeout = provider, endpoint, token, timeout
        self.expected_endpoint = expected
        self.http_transport = http_transport
        self.token_loader = token_loader
        self.token_refresher = token_refresher
        self.refresh_skew_seconds = refresh_skew_seconds
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._policy: ReadOnlyToolPolicy | None = None

    @classmethod
    def with_oauth(
        cls,
        provider: Provider,
        endpoint: str,
        token: TokenRecord,
        manager: "OAuthManager",
        *,
        oauth_client: OAuthClient | None = None,
        **options: Any,
    ) -> Self:
        return cls(
            provider,
            endpoint,
            token,
            token_loader=manager.store.get,
            token_refresher=lambda current: manager.refresh(current, oauth_client),
            **options,
        )

    async def __aenter__(self) -> "StreamableHttpConnection":
        token = self.token_loader(self.token.storage_key) if self.token_loader else self.token
        token = token or self.token
        if token.needs_refresh(skew_seconds=self.refresh_skew_seconds):
            if self.token_refresher is None:
                raise ReauthorizationRequired("access token expired before MCP connection")
            token = self.token_refresher(token)
        self._validate_token(token)
        stack = AsyncExitStack()
        try:
            headers = {"Authorization": f"Bearer {token.access_token.get_secret_value()}"}
            http = await stack.enter_async_context(
                httpx.AsyncClient(
                    headers=headers,
                    timeout=httpx.Timeout(self.timeout),
                    transport=self.http_transport,
                    trust_env=False,
                )
            )
            streams = await stack.enter_async_context(
                streamable_http_client(self.endpoint, http_client=http)
            )
            read, write, _session_id = streams
            session = await stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=timedelta(seconds=self.timeout))
            )
            await session.initialize()
            tools = await session.list_tools()
            policy = ReadOnlyToolPolicy(self.provider, [tool.name for tool in tools.tools])
        except BaseException:
            await stack.aclose()
            self._session, self._stack, self._policy = None, None, None
            raise
        self.token = token
        self._policy = policy
        self._session, self._stack = session, stack
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack, self._session, self._policy = None, None, None

    def _validate_token(self, token: TokenRecord) -> None:
        if (
            token.provider is not self.provider
            or token.audience != self.endpoint
            or self.endpoint != self.expected_endpoint
        ):
            raise AudienceError("refreshed OAuth token does not match the MCP resource")

    @property
    def snapshot(self) -> ToolSnapshot:
        if self._policy is None:
            raise RuntimeError("MCP connection is not open")
        return self._policy.snapshot()

    async def call(self, name: str, arguments: dict[str, Any]) -> UntrustedToolResult:
        if self._session is None or self._policy is None:
            raise RuntimeError("MCP connection is not open")
        self._policy.require(name)
        result = await self._session.call_tool(name, arguments)
        content = [item.model_dump(mode="json") for item in result.content]
        return self._policy.wrap(content, is_error=bool(result.isError))
