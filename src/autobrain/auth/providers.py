"""Provider-specific OAuth and read-only policy configuration."""

from dataclasses import dataclass

from autobrain.auth.models import Provider

NOTION_RESOURCE = "https://mcp.notion.com/mcp"
SLACK_RESOURCE = "https://mcp.slack.com/mcp"

# These are deliberately explicit. A name not listed here is never callable,
# even if a hosted server advertises it.
NOTION_READ_TOOLS = frozenset({"notion-search", "notion-fetch", "notion_search", "notion_fetch"})
SLACK_READ_TOOLS = frozenset(
    {
        "slack-search",
        "slack-search-public",
        "slack-search-private",
        "slack-channel-list",
        "slack-channel-history",
        "slack-thread-replies",
        "slack-file-read",
        "slack-canvas-read",
        "slack-user-read",
        "slack_search",
        "slack_search_public",
        "slack_search_private",
        "slack_search_channels",
        "slack_search_users",
        "slack_fetch",
        "slack_channels_list",
        "slack_channel_history",
        "slack_thread_replies",
        "slack_file_read",
        "slack_canvas_read",
        "slack_users_read",
    }
)

# No DM/MPIM scopes appear in the default request.
SLACK_SCOPES = (
    "search:read",
    "channels:read",
    "channels:history",
    "groups:read",
    "groups:history",
    "files:read",
    "canvases:read",
    "users:read",
)


@dataclass(frozen=True)
class ProviderConfig:
    provider: Provider
    resource: str | None
    scopes: tuple[str, ...]
    allowlist: frozenset[str]
    dynamic_registration: bool
    fixed_client: bool
    required_tool_groups: tuple[frozenset[str], ...] = ()
    supported: bool = True
    detail: str = ""
    remediation: str = ""


_UNSUPPORTED_DETAIL = "live OAuth connector is not implemented"
_UNSUPPORTED_REMEDIATION = (
    "No verified official MCP connector exists for this source. "
    "AutoBrain will not implement custom OAuth or accept API-key/proxy auth. "
    "Wait for a verified connector or import a local fixture/snapshot instead."
)


# This is the authoritative provider table. Unsupported entries are deliberate:
# they make configuration and readiness truthful without adding live integrations.
CONFIGS = {
    Provider.SLACK: ProviderConfig(
        Provider.SLACK,
        SLACK_RESOURCE,
        SLACK_SCOPES,
        SLACK_READ_TOOLS,
        False,
        True,
        (
            frozenset({"slack-channel-list", "slack_channels_list"}),
            frozenset({"slack-channel-history", "slack_channel_history"}),
        ),
    ),
    Provider.NOTION: ProviderConfig(
        Provider.NOTION,
        NOTION_RESOURCE,
        ("read_content",),
        NOTION_READ_TOOLS,
        True,
        False,
        (
            frozenset({"notion-search", "notion_search"}),
            frozenset({"notion-fetch", "notion_fetch"}),
        ),
    ),
    Provider.CONFLUENCE: ProviderConfig(
        Provider.CONFLUENCE,
        None,
        (),
        frozenset(),
        False,
        False,
        supported=False,
        detail=_UNSUPPORTED_DETAIL,
        remediation=_UNSUPPORTED_REMEDIATION,
    ),
    Provider.GOOGLE_DRIVE: ProviderConfig(
        Provider.GOOGLE_DRIVE,
        None,
        (),
        frozenset(),
        False,
        False,
        supported=False,
        detail=_UNSUPPORTED_DETAIL,
        remediation=_UNSUPPORTED_REMEDIATION,
    ),
    Provider.SHAREPOINT: ProviderConfig(
        Provider.SHAREPOINT,
        None,
        (),
        frozenset(),
        False,
        False,
        supported=False,
        detail=_UNSUPPORTED_DETAIL,
        remediation=_UNSUPPORTED_REMEDIATION,
    ),
}


def config_for(provider: Provider) -> ProviderConfig:
    return CONFIGS[provider]
