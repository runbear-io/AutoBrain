from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autobrain.cli import app
from autobrain.custom_provider import CustomProviderConfig, CustomProviderRegistry


class MemoryKeyring:
    priority = 1.0

    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self.values[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self.values.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        self.values.pop((service, username), None)


def config() -> CustomProviderConfig:
    return CustomProviderConfig(
        provider_id="local-vllm",
        name="Local VLLM",
        endpoint="http://127.0.0.1:8000/v1/",
        model="qwen2.5",
        api_key_env="LOCAL_VLLM_API_KEY",
    )


def test_config_validates_endpoint_and_normalizes_id() -> None:
    assert config().endpoint == "http://127.0.0.1:8000/v1"
    with pytest.raises(ValueError, match="username or password"):
        CustomProviderConfig(
            provider_id="safe-name",
            name="x",
            endpoint="https://u:p@example.test/v1",
            model="m",
            api_key_env="KEY",
        )


def test_registry_keeps_key_out_of_metadata_and_supports_keyring(tmp_path: Path) -> None:
    backend = MemoryKeyring()
    registry = CustomProviderRegistry(tmp_path, backend=backend)
    registry.add(config(), "custom-secret-123456")
    raw = (tmp_path / "custom-providers" / "providers.json").read_text()
    assert "custom-secret" not in raw
    assert registry.credential("LOCAL-VLLM") == "custom-secret-123456"
    assert registry.status("local-vllm").credential_present is True
    registry.remove("local-vllm")
    assert registry.list() == ()
    assert registry.list() == ()


def test_registry_fallback_is_confined_and_0600(tmp_path: Path) -> None:
    class NoKeyring(MemoryKeyring):
        priority = 0.0

    registry = CustomProviderRegistry(tmp_path, backend=NoKeyring())
    registry.add(config(), "custom-secret-123456")
    fallback = tmp_path / "custom-providers" / "oauth-tokens.json"
    assert fallback.stat().st_mode & 0o777 == 0o600
    assert json.loads(fallback.read_text())["local-vllm"]["api_key"] == "custom-secret-123456"


def test_cli_provider_lifecycle_never_prints_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    added = CliRunner().invoke(
        app,
        [
            "provider",
            "add",
            "local-vllm",
            "--endpoint",
            "http://127.0.0.1:8000/v1",
            "--model",
            "qwen2.5",
            "--api-key-env",
            "LOCAL_VLLM_API_KEY",
            "--api-key",
            "custom-secret-123456",
        ],
    )
    assert added.exit_code == 0
    assert "custom-secret" not in added.output
    status = CliRunner().invoke(app, ["provider", "status", "--json"], env={"HOME": str(tmp_path)})
    assert status.exit_code == 0
    assert json.loads(status.stdout)[0]["credential_present"] is True
    removed = CliRunner().invoke(
        app, ["provider", "remove", "local-vllm", "--yes"], env={"HOME": str(tmp_path)}
    )
    assert removed.exit_code == 0
