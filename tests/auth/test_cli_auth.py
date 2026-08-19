import json
from pathlib import Path

from keyring.errors import KeyringError
from pydantic import SecretStr
from pytest import MonkeyPatch
from typer.testing import CliRunner

import autobrain.cli as cli
from autobrain.auth.models import Provider, TokenRecord
from autobrain.auth.storage import TokenStore
from autobrain.cli import app


def test_auth_cli_exposes_exact_provider_status_and_logout_commands() -> None:
    help_result = CliRunner().invoke(app, ["auth", "--help"])
    assert help_result.exit_code == 0
    for command in ("slack", "notion", "status", "logout"):
        assert command in help_result.stdout


def test_auth_status_json_reports_both_sources_without_tokens(tmp_path: Path) -> None:
    # CliRunner supplies isolated environment without invoking external OAuth.
    result = CliRunner().invoke(app, ["auth", "status", "--json"], env={"HOME": str(tmp_path)})
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert [item["provider"] for item in payload["connections"]] == ["slack", "notion"]
    assert all(item["state"] == "DISCONNECTED" for item in payload["connections"])
    assert "token" not in result.stdout.lower()


def test_initial_authorization_emits_degraded_storage_warning(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    class FailingSetKeyring:
        priority: float = 1.0

        def set_password(self, service: str, username: str, password: str) -> None:
            del service, username, password
            raise KeyringError("backend-secret")

        def get_password(self, service: str, username: str) -> str | None:
            del service, username
            return None

        def delete_password(self, service: str, username: str) -> None:
            del service, username

    store = TokenStore(tmp_path / "auth", backend=FailingSetKeyring())
    token = TokenRecord(
        provider=Provider.NOTION,
        workspace_id="W",
        user_id="U",
        audience="https://mcp.notion.com/mcp",
        access_token=SecretStr("access-secret"),
        refresh_token=SecretStr("refresh-secret"),
    )

    class FakeManager:
        def __init__(self, token_store: TokenStore, **options: object) -> None:
            del options
            self.store = token_store

        def authorize(self, source: Provider, **options: object) -> TokenRecord:
            del source, options
            self.store.save(token)
            return token

    def fake_store(path: Path) -> TokenStore:
        del path
        return store

    monkeypatch.setattr(cli, "TokenStore", fake_store)
    monkeypatch.setattr(cli, "OAuthManager", FakeManager)
    result = CliRunner().invoke(app, ["auth", "notion"], env={"HOME": str(tmp_path)})
    assert result.exit_code == 0
    assert "WARNING:" in result.stderr and "0600" in result.stderr
    assert "backend-secret" not in result.output
    assert "access-secret" not in result.output
    assert "refresh-secret" not in result.output


def test_auth_logout_is_idempotent_without_external_calls(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["auth", "logout", "slack"], env={"HOME": str(tmp_path)})
    assert result.exit_code == 0
    assert "0 credentials removed" in result.stdout
