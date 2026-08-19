"""Keychain-backed tokens with confined atomic degraded storage."""

import json
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Protocol, cast

import keyring
from keyring.errors import KeyringError

from autobrain.auth.file_security import SecureAuthFiles
from autobrain.auth.models import ConnectionStatus, OAuthError, Provider, TokenRecord
from autobrain.models import ConnectionState, Status

_SERVICE = "autobrain.oauth"
_EXPIRY_SKEW_SECONDS = 60
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_DEGRADED_WARNING = "OS keychain unavailable; tokens use a confined atomic 0600 fallback file"


class TokenStorageError(OAuthError):
    """Sanitized token-storage failure."""


class KeyringBackend(Protocol):
    @property
    def priority(self) -> float: ...

    def set_password(self, service: str, username: str, password: str) -> None: ...
    def get_password(self, service: str, username: str) -> str | None: ...
    def delete_password(self, service: str, username: str) -> None: ...


def _lock(key: str) -> threading.RLock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def _safe_key(key: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:._-"
    if not key or any(character not in allowed for character in key):
        raise ValueError("invalid token storage key")
    return key


class TokenStore:
    def __init__(self, root: Path, *, backend: KeyringBackend | None = None) -> None:
        self.files = SecureAuthFiles(root)
        self.root = root
        self.fallback = self.files.tokens
        self.index = self.files.index
        self.backend = backend or cast(KeyringBackend, keyring.get_keyring())
        self.keychain_available = self.backend.priority > 0
        self.degraded_warning: str | None = None if self.keychain_available else _DEGRADED_WARNING

    @property
    def using_keychain(self) -> bool:
        return self.keychain_available and self.degraded_warning is None

    def _fallback_values(self) -> dict[str, dict[str, object]]:
        return self.files.read_mapping(self.fallback)

    def _write_fallback(self, key: str, record: TokenRecord | None) -> None:
        with self.files.process_lock("fallback"):
            values = self._fallback_values()
            if record is None:
                values.pop(key, None)
            else:
                values[key] = _raw_record(record)
            self.files.write_atomic(self.fallback, values)

    def _update_index(self, record: TokenRecord | None, key: str, storage: str = "") -> None:
        with self.files.process_lock("index"):
            entries = self.files.read_mapping(self.index)
            if record is None:
                entries.pop(key, None)
            else:
                entries[key] = {
                    "provider": record.provider.value,
                    "workspace_id": record.workspace_id,
                    "user_id": record.user_id,
                    "audience": record.audience,
                    "storage": storage,
                }
            self.files.write_atomic(self.index, entries)

    def save(self, record: TokenRecord) -> None:
        self.files.ensure_root()
        key = _safe_key(record.storage_key)
        with _lock(key):
            storage = "file"
            if self.keychain_available:
                try:
                    self.backend.set_password(_SERVICE, key, json.dumps(_raw_record(record)))
                    if self.fallback.exists():
                        self._write_fallback(key, None)
                    storage = "keychain"
                except KeyringError:
                    self.degraded_warning = _DEGRADED_WARNING
                    self._write_fallback(key, record)
            else:
                self._write_fallback(key, record)
            self._update_index(record, key, storage)

    def get(self, key: str) -> TokenRecord | None:
        self.files.ensure_root()
        key = _safe_key(key)
        with _lock(key):
            raw: str | dict[str, object] | None = None
            if self.keychain_available:
                try:
                    raw = self.backend.get_password(_SERVICE, key)
                except KeyringError as exc:
                    self.degraded_warning = _DEGRADED_WARNING
                    raw = self._fallback_values().get(key)
                    if raw is None:
                        del exc
                        raise TokenStorageError(
                            "OS keychain read failed; stored credentials were preserved"
                        ) from None
            if raw is None:
                raw = self._fallback_values().get(key)
            if raw is None:
                return None
            if isinstance(raw, str):
                try:
                    decoded = json.loads(raw)
                except ValueError as exc:
                    raise TokenStorageError("Stored OAuth token is malformed") from exc
                raw = cast(dict[str, object], decoded)
            raw["provider"] = Provider(str(raw.get("provider")))
            return TokenRecord.model_validate(raw)

    def delete(self, key: str) -> None:
        self.files.ensure_root()
        key = _safe_key(key)
        with _lock(key):
            entries = self.files.read_mapping(self.index)
            storage = entries.get(key, {}).get("storage")
            if self.keychain_available and storage != "file":
                try:
                    self.backend.delete_password(_SERVICE, key)
                except KeyringError as exc:
                    self.degraded_warning = _DEGRADED_WARNING
                    del exc
                    raise TokenStorageError(
                        "OS keychain delete failed; credential index was preserved"
                    ) from None
            if self.fallback.exists():
                self._write_fallback(key, None)
            self._update_index(None, key)

    def rotation_lock(self, key: str) -> ExitStack:
        self.files.ensure_root()
        key = _safe_key(key)
        stack = ExitStack()
        try:
            stack.enter_context(_lock(key))
            stack.enter_context(self.files.process_lock(f"refresh:{key}"))
        except BaseException:
            stack.close()
            raise
        return stack

    def statuses(self, *, now: int | None = None) -> tuple[ConnectionStatus, ...]:
        entries = self.files.read_mapping(self.index)
        result: list[ConnectionStatus] = []
        for key, value in sorted(entries.items()):
            provider_raw = value.get("provider")
            if not isinstance(provider_raw, str) or provider_raw not in {
                item.value for item in Provider
            }:
                continue
            provider = Provider(provider_raw)
            try:
                record = self.get(key)
            except (ValueError, TypeError, TokenStorageError):
                result.append(_reauth_status(provider, self.degraded_warning))
                continue
            if record is None:
                result.append(_reauth_status(provider, self.degraded_warning))
                continue
            current = int(time.time()) if now is None else now
            expired = record.needs_refresh(skew_seconds=_EXPIRY_SKEW_SECONDS, now=current)
            if expired:
                state = (
                    ConnectionState.EXPIRED
                    if record.refresh_token
                    else ConnectionState.REAUTHORIZATION_REQUIRED
                )
                result.append(
                    ConnectionStatus(
                        provider=provider,
                        state=state,
                        status=Status.MCP_AUTH_UNAVAILABLE,
                        workspace_id=record.workspace_id,
                        user_id=record.user_id,
                        storage=str(value.get("storage") or "file"),
                        warning=(
                            "Access token expired; refresh required"
                            if record.refresh_token
                            else "Access token expired; reauthorization required"
                        ),
                    )
                )
                continue
            result.append(
                ConnectionStatus(
                    provider=provider,
                    state=ConnectionState.CONNECTED,
                    status=Status.OK,
                    workspace_id=record.workspace_id,
                    user_id=record.user_id,
                    storage=str(value.get("storage") or "file"),
                    warning=self.degraded_warning,
                )
            )
        return tuple(result)


def _reauth_status(provider: Provider, warning: str | None) -> ConnectionStatus:
    return ConnectionStatus(
        provider=provider,
        state=ConnectionState.REAUTHORIZATION_REQUIRED,
        status=Status.MCP_AUTH_UNAVAILABLE,
        warning=warning or "Stored OAuth state is unavailable; reauthorization is required",
    )


def _raw_record(record: TokenRecord) -> dict[str, object]:
    value = record.model_dump(mode="python")
    value["access_token"] = record.access_token.get_secret_value()
    value["refresh_token"] = (
        record.refresh_token.get_secret_value() if record.refresh_token else None
    )
    value["provider"] = record.provider.value
    return value
