"""CLI groups for Slack/Notion authorization and local source setup."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from autobrain.auth.callback import LocalCallback
from autobrain.auth.models import OAuthError, Provider
from autobrain.auth.oauth import OAuthManager
from autobrain.auth.service import ConnectionManager
from autobrain.auth.storage import TokenStore
from autobrain.connectors.local_file import local_file_readiness
from autobrain.connectors.notion_snapshot import NotionSnapshotError, NotionSnapshotStore
from autobrain.connectors.readiness import readiness_for
from autobrain.connectors.slack_export import SlackExportError
from autobrain.paths import AutoBrainPaths
from autobrain.secrets import RuntimeEnvironment, RuntimeSettings
from autobrain.source_store import SlackSourceStore

auth_app = typer.Typer(help="Connect and manage hosted read-only MCP sources.")
source_app = typer.Typer(help="Configure local knowledge-source inputs.")


def _authorize_source(source: Provider) -> None:
    paths = AutoBrainPaths.from_home()
    settings = RuntimeSettings.from_environ(os.environ)
    if settings.callback_port_error is not None:
        typer.echo(f"MCP_AUTH_UNAVAILABLE: {settings.callback_port_error}", err=True)
        raise typer.Exit(1)
    environment = RuntimeEnvironment.from_environ(os.environ)
    manager = OAuthManager(
        TokenStore(paths.root / "auth"),
        callback_factory=lambda state: LocalCallback("127.0.0.1", settings.callback_port, state),
    )
    try:
        token = manager.authorize(
            source,
            slack_client_id=(
                environment.slack_client_id.get_secret_value()
                if environment.slack_client_id
                else None
            ),
            slack_client_secret=(
                environment.slack_client_secret.get_secret_value()
                if environment.slack_client_secret
                else None
            ),
        )
    except OAuthError as exc:
        typer.echo(f"MCP_AUTH_UNAVAILABLE: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Connected {source.value} workspace {token.workspace_id} as {token.user_id}")
    if manager.store.degraded_warning:
        typer.echo(f"WARNING: {manager.store.degraded_warning}", err=True)


@auth_app.command("slack")
def authorize_slack() -> None:
    """Authorize the fixed internal Slack app for read-only MCP access."""
    _authorize_source(Provider.SLACK)


@auth_app.command("notion")
def authorize_notion() -> None:
    """Authorize Notion MCP using discovery, DCR, and PKCE."""
    _authorize_source(Provider.NOTION)


@auth_app.command("status")
def auth_status(
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON status.")] = False,
) -> None:
    """Show Slack and Notion connection state independently."""
    report = ConnectionManager(AutoBrainPaths.from_home().root).status()
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        return
    for connection in report.connections:
        typer.echo(f"{connection.provider.value:8} {connection.state.value}")
        if connection.warning:
            typer.echo(f"WARNING: {connection.warning}", err=True)


@auth_app.command("logout")
def auth_logout(provider: Provider) -> None:
    """Remove every stored identity for one source."""
    try:
        removed = ConnectionManager(AutoBrainPaths.from_home().root).logout(provider)
    except OAuthError as exc:
        typer.echo(f"MCP_AUTH_UNAVAILABLE: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(
        f"Logged out {provider.value} ({removed} credential{'s' if removed != 1 else ''} removed)"
    )


@source_app.command("slack")
def configure_slack_source(
    export_path: Annotated[
        Path | None,
        typer.Option("--export", help="Official Slack Workspace Export ZIP."),
    ] = None,
    live: Annotated[
        bool,
        typer.Option("--live", help="Use advanced live Slack MCP OAuth instead."),
    ] = False,
    remove: Annotated[
        bool,
        typer.Option("--remove", help="Remove the configured Slack export."),
    ] = False,
) -> None:
    """Configure the recommended local Slack export or advanced live MCP source."""
    if sum((export_path is not None, live, remove)) > 1:
        typer.echo("Choose only one of --export, --live, or --remove.", err=True)
        raise typer.Exit(2)
    store = SlackSourceStore(AutoBrainPaths.from_home().sources)
    if remove:
        store.remove()
        typer.echo("Removed the configured Slack export.")
        return
    if export_path is None and not live:
        choice = typer.prompt(
            "Slack source: [1] Import export ZIP (recommended), [2] Connect live MCP",
            default="1",
        )
        if choice.strip() == "2":
            live = True
        elif choice.strip() == "1":
            export_path = Path(typer.prompt("Path to Slack export ZIP"))
        else:
            typer.echo("Choose 1 or 2.", err=True)
            raise typer.Exit(2)
    if live:
        _authorize_source(Provider.SLACK)
        store.remove()
        return
    if export_path is None:
        raise typer.Exit(2)
    try:
        config = store.configure_export(export_path)
    except (SlackExportError, OSError, ValueError) as exc:
        typer.echo(f"SOURCE_AUTH_UNAVAILABLE: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(
        "Slack export ready: "
        f"{config.summary.message_count} messages from "
        f"{config.summary.channel_count} channels"
    )


@source_app.command("notion-snapshot")
def configure_notion_snapshot(
    import_path: Annotated[
        Path | None,
        typer.Option(
            "--import",
            help="Bounded normalized JSON from an external read-only Notion MCP session.",
        ),
    ] = None,
    remove: Annotated[
        bool,
        typer.Option("--remove", help="Remove the imported Notion snapshot."),
    ] = False,
) -> None:
    """Import or remove a credential-free, read-only Notion MCP snapshot."""
    if (import_path is None) == (not remove):
        typer.echo("Choose exactly one of --import or --remove.", err=True)
        raise typer.Exit(2)
    store = NotionSnapshotStore(AutoBrainPaths.from_home().sources)
    if remove:
        store.remove()
        typer.echo("Removed the imported Notion snapshot.")
        return
    assert import_path is not None
    try:
        config = store.import_snapshot(import_path)
    except (NotionSnapshotError, OSError, ValueError) as exc:
        typer.echo(f"SOURCE_AUTH_UNAVAILABLE: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(
        f"Notion snapshot ready: {config.document_count} pages; "
        "coverage is partial/non-final and content remains untrusted data"
    )


@source_app.command("local-file")
def local_file_status(
    path: Annotated[Path, typer.Argument(help="Absolute Markdown, TXT, or HTML file path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON status.")] = False,
) -> None:
    """Check readiness for one bounded local document source."""
    readiness = local_file_readiness(path.expanduser())
    if json_output:
        typer.echo(json.dumps(readiness.model_dump(mode="json"), indent=2))
    else:
        typer.echo(f"Local file: {readiness.status.value}: {readiness.detail}")
    if not readiness.ready:
        raise typer.Exit(1)


@source_app.command("status")
def source_status(
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON status.")] = False,
) -> None:
    """Show configured local Slack and Notion snapshot state."""
    sources = AutoBrainPaths.from_home().sources
    slack = SlackSourceStore(sources).status()
    notion = NotionSnapshotStore(sources).status()
    readiness = {
        provider.value: readiness_for(provider).model_dump(mode="json")
        for provider in Provider
        if provider not in {Provider.SLACK, Provider.NOTION}
    }
    if json_output:
        typer.echo(
            json.dumps(
                {
                    **slack.model_dump(mode="json"),
                    "slack": slack.model_dump(mode="json"),
                    "notion_snapshot": notion.model_dump(mode="json"),
                    "connectors": {
                        "slack": slack.model_dump(mode="json"),
                        "notion": notion.model_dump(mode="json"),
                        **readiness,
                    },
                },
                indent=2,
            )
        )
        return
    typer.echo(f"Slack: {slack.state.value}: {slack.detail}")
    typer.echo(f"Notion snapshot: {'READY' if notion.ready else 'NOT_CONFIGURED'}: {notion.detail}")
    for provider in Provider:
        if provider in {Provider.SLACK, Provider.NOTION}:
            continue
        status = readiness[provider.value]
        typer.echo(f"{provider.value}: {status['state']}: {status['detail']}")
