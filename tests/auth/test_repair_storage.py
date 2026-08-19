from pathlib import Path

import pytest
from keyring.errors import KeyringError
from pydantic import SecretStr

from autobrain.auth.file_security import AuthPathError
from autobrain.auth.models import Provider, TokenRecord
from autobrain.auth.storage import TokenStorageError, TokenStore
from tests.auth.fakes import MemoryKeyring


class RuntimeFailingKeyring(MemoryKeyring):
    def __init__(
        self, *, fail_set: bool = False, fail_get: bool = False, fail_delete: bool = False
    ) -> None:
        super().__init__()
        self.fail_set, self.fail_get, self.fail_delete = fail_set, fail_get, fail_delete

    def set_password(self, service: str, username: str, password: str) -> None:
        if self.fail_set:
            raise KeyringError("backend-secret set failure")
        super().set_password(service, username, password)

    def get_password(self, service: str, username: str) -> str | None:
        if self.fail_get:
            raise KeyringError("backend-secret get failure")
        return super().get_password(service, username)

    def delete_password(self, service: str, username: str) -> None:
        if self.fail_delete:
            raise KeyringError("backend-secret delete failure")
        super().delete_password(service, username)


def record(*, expires_at: int | None = None, refresh: bool = True) -> TokenRecord:
    return TokenRecord(
        provider=Provider.NOTION,
        workspace_id="workspace",
        user_id="user",
        audience="https://mcp.notion.com/mcp",
        access_token=SecretStr("access-secret"),
        refresh_token=SecretStr("refresh-secret") if refresh else None,
        expires_at=expires_at,
        oauth_client_id="client",
    )


def test_expired_status_distinguishes_refreshable_and_reauthorization(tmp_path: Path) -> None:
    refreshable = TokenStore(tmp_path / "refreshable", backend=MemoryKeyring())
    refreshable.save(record(expires_at=1050))
    status = refreshable.statuses(now=1000)[0]
    assert status.state.value == "EXPIRED"
    assert status.status.value == "MCP_AUTH_UNAVAILABLE"
    assert "refresh required" in (status.warning or "")

    terminal = TokenStore(tmp_path / "terminal", backend=MemoryKeyring())
    terminal.save(record(expires_at=1000, refresh=False))
    status = terminal.statuses(now=1000)[0]
    assert status.state.value == "REAUTHORIZATION_REQUIRED"
    assert "reauthorization" in (status.warning or "")


def test_runtime_keyring_set_and_get_failures_use_atomic_fallback(tmp_path: Path) -> None:
    backend = RuntimeFailingKeyring(fail_set=True, fail_get=True)
    store = TokenStore(tmp_path, backend=backend)
    store.save(record())
    assert store.fallback.stat().st_mode & 0o777 == 0o600
    assert store.degraded_warning and "0600" in store.degraded_warning
    assert "backend-secret" not in store.degraded_warning
    loaded = store.get(record().storage_key)
    assert loaded == record()
    backend.fail_delete = True
    store.delete(record().storage_key)  # file-backed credential does not call the broken keychain
    assert not store.statuses()


def test_runtime_keyring_get_delete_errors_are_sanitized_and_preserve_index(tmp_path: Path) -> None:
    backend = RuntimeFailingKeyring()
    store = TokenStore(tmp_path, backend=backend)
    store.save(record())
    backend.fail_get = True
    with pytest.raises(TokenStorageError) as get_error:
        store.get(record().storage_key)
    assert "backend-secret" not in str(get_error.value)
    assert store.index.exists()
    backend.fail_get = False
    backend.fail_delete = True
    with pytest.raises(TokenStorageError) as delete_error:
        store.delete(record().storage_key)
    assert "backend-secret" not in str(delete_error.value)
    assert record().storage_key in store.index.read_text(encoding="utf-8")


@pytest.mark.parametrize("target", ["root", "parent", "index", "tokens", "locks"])
def test_symlinked_auth_paths_never_write_outside(tmp_path: Path, target: str) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "auth"
    if target == "root":
        root.symlink_to(outside, target_is_directory=True)
    elif target == "parent":
        parent = tmp_path / "linked-parent"
        parent.symlink_to(outside, target_is_directory=True)
        root = parent / "auth"
    else:
        root.mkdir()
        if target == "locks":
            (root / "locks").symlink_to(outside, target_is_directory=True)
        else:
            outside_file = outside / f"{target}.json"
            outside_file.write_text("preserve", encoding="utf-8")
            name = "oauth-index.json" if target == "index" else "oauth-tokens.json"
            (root / name).symlink_to(outside_file)
    store = TokenStore(root, backend=RuntimeFailingKeyring(fail_set=True))
    with pytest.raises(AuthPathError):
        if target == "locks":
            with store.rotation_lock(record().storage_key):
                pass
        else:
            store.save(record())
    contents = [path.read_text(encoding="utf-8") for path in outside.iterdir() if path.is_file()]
    assert contents in ([], ["preserve"])
