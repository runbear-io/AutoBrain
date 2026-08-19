"""Hard provider-specific MCP tool policy."""

from dataclasses import dataclass
from typing import Any

from autobrain.auth.models import Provider
from autobrain.auth.providers import config_for


class ToolPolicyError(PermissionError):
    """A tool was not both advertised and explicitly read-allowed."""


@dataclass(frozen=True)
class ToolSnapshot:
    provider: Provider
    advertised: tuple[str, ...]
    allowed: tuple[str, ...]
    refused: tuple[str, ...]


@dataclass(frozen=True)
class UntrustedToolResult:
    """MCP-returned content is inert source data, never executable instructions."""

    content: object
    is_error: bool
    trusted: bool = False


class ReadOnlyToolPolicy:
    def __init__(self, provider: Provider, advertised: list[str]) -> None:
        allowlist = config_for(provider).allowlist
        self.provider = provider
        self.advertised = frozenset(advertised)
        self.allowed = self.advertised & allowlist

    def snapshot(self) -> ToolSnapshot:
        return ToolSnapshot(
            provider=self.provider,
            advertised=tuple(sorted(self.advertised)),
            allowed=tuple(sorted(self.allowed)),
            refused=tuple(sorted(self.advertised - self.allowed)),
        )

    def require(self, name: str) -> None:
        if name not in self.advertised:
            raise ToolPolicyError(f"MCP tool was not advertised: {name}")
        if name not in self.allowed:
            raise ToolPolicyError(
                f"MCP tool is not on the {self.provider.value} read allowlist: {name}"
            )

    def wrap(self, content: Any, *, is_error: bool) -> UntrustedToolResult:
        return UntrustedToolResult(content=content, is_error=is_error)
