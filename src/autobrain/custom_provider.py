"""Local registration for user-supplied OpenAI-compatible providers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

import keyring
from keyring.errors import KeyringError
from openai import OpenAI
from pydantic import Field, field_validator

from autobrain.auth.file_security import SecureAuthFiles
from autobrain.models import StrictModel

_SERVICE = "autobrain.custom-provider"
_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ENV = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")


class CustomProviderKeyring(Protocol):
    priority: int | float

    def set_password(self, service: str, username: str, password: str) -> None: ...
    def get_password(self, service: str, username: str) -> str | None: ...
    def delete_password(self, service: str, username: str) -> None: ...


class CustomProviderError(ValueError):
    """A user-facing custom provider configuration or storage failure."""


class CustomProviderConfig(StrictModel):
    """Non-secret metadata for one locally registered provider."""

    provider_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    endpoint: str = Field(min_length=1, max_length=2048)
    model: str = Field(min_length=1, max_length=256)
    api_key_env: str = Field(min_length=1, max_length=128)

    @field_validator("provider_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        value = value.casefold()
        if _ID.fullmatch(value) is None or value in {"api", "openai", "custom"}:
            raise ValueError("provider_id must be a unique lowercase name")
        return value

    @field_validator("name", "model", "api_key_env")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("provider fields must not be blank")
        return value

    @field_validator("api_key_env")
    @classmethod
    def valid_env(cls, value: str) -> str:
        if _ENV.fullmatch(value) is None:
            raise ValueError("api_key_env must be an uppercase environment variable name")
        return value

    @field_validator("endpoint")
    @classmethod
    def valid_endpoint(cls, value: str) -> str:
        if value != value.strip() or any(character.isspace() for character in value):
            raise ValueError("endpoint must be an HTTP(S) URL without whitespace")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("endpoint must be an HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("endpoint must not contain username or password")
        try:
            port = parsed.port
        except ValueError:
            raise ValueError("endpoint must contain a valid port") from None
        del port
        return value.rstrip("/")

    @property
    def mode(self) -> str:
        return f"custom:{self.provider_id}"

    def safe_metadata(self) -> dict[str, str]:
        return {
            "provider_id": self.provider_id,
            "name": self.name,
            "endpoint": self.endpoint,
            "model": self.model,
            "api_key_env": self.api_key_env,
        }


class CustomProviderStatus(StrictModel):
    provider_id: str
    name: str
    endpoint: str
    model: str
    api_key_env: str
    credential_present: bool
    status: str
    detail: str = ""


class CustomProviderRegistry:
    """Persist provider metadata locally and credentials in the OS keychain."""

    def __init__(self, root: Path, *, backend: CustomProviderKeyring | None = None) -> None:
        self.files = SecureAuthFiles(root / "custom-providers")
        self.metadata = self.files.root / "providers.json"
        self.backend = backend or cast(CustomProviderKeyring, keyring.get_keyring())
        self.keychain_available = getattr(self.backend, "priority", 0) > 0

    def _read(self) -> dict[str, dict[str, Any]]:
        self.files.ensure_root()
        if not self.metadata.exists():
            return {}
        value = json.loads(self.metadata.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise CustomProviderError("stored custom provider configuration is malformed")
        return cast(dict[str, dict[str, Any]], value)

    def _write(self, value: dict[str, dict[str, Any]]) -> None:
        self.files.write_atomic(self.metadata, value)

    def list(self) -> tuple[CustomProviderConfig, ...]:
        result: list[CustomProviderConfig] = []
        for value in self._read().values():
            try:
                result.append(CustomProviderConfig.model_validate(value))
            except (TypeError, ValueError):
                continue
        return tuple(sorted(result, key=lambda item: item.provider_id))

    def get(self, provider_id: str) -> CustomProviderConfig:
        normalized = provider_id.casefold()
        for provider in self.list():
            if provider.provider_id == normalized:
                return provider
        raise CustomProviderError(f"custom provider is not registered: {provider_id}")

    def add(self, config: CustomProviderConfig, api_key: str) -> None:
        if not api_key.strip():
            raise CustomProviderError("API key must not be blank")
        values = self._read()
        if config.provider_id in values:
            raise CustomProviderError(
                f"custom provider is already registered: {config.provider_id}"
            )
        values[config.provider_id] = config.model_dump(mode="json")
        self._write(values)
        self._set_secret(config.provider_id, api_key)

    def remove(self, provider_id: str) -> None:
        config = self.get(provider_id)
        self._delete_secret(config.provider_id)
        values = self._read()
        values.pop(config.provider_id, None)
        self._write(values)

    def credential(self, provider_id: str, environ: dict[str, str] | None = None) -> str | None:
        config = self.get(provider_id)
        try:
            value = (
                self.backend.get_password(_SERVICE, config.provider_id)
                if self.keychain_available
                else None
            )
        except KeyringError:
            value = None
        if value:
            return value
        fallback = self.files.read_mapping(self.files.tokens).get(config.provider_id)
        if isinstance(fallback, dict) and isinstance(fallback.get("api_key"), str):
            return cast(str, fallback["api_key"])
        return (environ or {}).get(config.api_key_env)

    def status(
        self, provider_id: str, environ: dict[str, str] | None = None
    ) -> CustomProviderStatus:
        config = self.get(provider_id)
        present = self.credential(provider_id, environ) is not None
        return CustomProviderStatus(
            **config.safe_metadata(),
            credential_present=present,
            status="READY" if present else "MISSING_PROVIDER",
            detail="credential is stored locally"
            if present
            else f"{config.api_key_env} is not configured",
        )

    def verify(
        self, provider_id: str, environ: dict[str, str] | None = None
    ) -> CustomProviderStatus:
        config = self.get(provider_id)
        key = self.credential(provider_id, environ)
        if not key:
            return self.status(provider_id, environ)
        try:
            client = OpenAI(api_key=key, base_url=config.endpoint, timeout=10)
            client.models.list()
        except Exception as exc:
            return CustomProviderStatus(
                **config.safe_metadata(),
                credential_present=True,
                status="UNAVAILABLE",
                detail=f"provider verification failed ({type(exc).__name__})",
            )
        return CustomProviderStatus(
            **config.safe_metadata(),
            credential_present=True,
            status="READY",
            detail="provider endpoint accepted the credential",
        )

    def _set_secret(self, provider_id: str, value: str) -> None:
        self.files.ensure_root()
        try:
            if self.keychain_available:
                self.backend.set_password(_SERVICE, provider_id, value)
                return
        except KeyringError:
            pass
        values = self.files.read_mapping(self.files.tokens)
        values[provider_id] = {"api_key": value}
        self.files.write_atomic(self.files.tokens, values)

    def _delete_secret(self, provider_id: str) -> None:
        try:
            if self.keychain_available:
                self.backend.delete_password(_SERVICE, provider_id)
        except KeyringError:
            pass
        values = self.files.read_mapping(self.files.tokens)
        values.pop(provider_id, None)
        self.files.write_atomic(self.files.tokens, values)


def resolve_custom_provider_mode(
    mode: str, registry: CustomProviderRegistry
) -> CustomProviderConfig | None:
    if not mode.casefold().startswith("custom:"):
        return None
    return registry.get(mode.split(":", 1)[1])
