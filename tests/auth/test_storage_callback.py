import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest
from pydantic import SecretStr

from autobrain.auth.callback import LocalCallback
from autobrain.auth.models import (
    ConsentDeniedError,
    OAuthError,
    Provider,
    StateMismatchError,
    TokenRecord,
)
from autobrain.auth.service import ConnectionManager
from autobrain.auth.storage import TokenStore
from tests.auth.fakes import MemoryKeyring


class FailedKeyring:
    priority: float = 0.0

    def set_password(self, service: str, username: str, password: str) -> None:
        del service, username, password
        raise AssertionError("unavailable keyring must not be used")

    def get_password(self, service: str, username: str) -> str | None:
        del service, username
        raise AssertionError("unavailable keyring must not be used")

    def delete_password(self, service: str, username: str) -> None:
        del service, username
        raise AssertionError("unavailable keyring must not be used")


def token() -> TokenRecord:
    return TokenRecord(
        provider=Provider.SLACK,
        workspace_id="T1",
        user_id="U1",
        audience="https://mcp.slack.com/mcp",
        access_token=SecretStr("access-never-print"),
        refresh_token=SecretStr("refresh-never-print"),
    )


def test_keychain_storage_contains_raw_token_but_status_is_redacted(tmp_path: Path) -> None:
    backend = MemoryKeyring()
    store = TokenStore(tmp_path, backend=backend)
    store.save(token())
    assert any("access-never-print" in value for value in backend.values.values())
    report = ConnectionManager(tmp_path.parent, store=store).status().model_dump_json()
    assert "access-never-print" not in report
    assert "refresh-never-print" not in report
    assert json.loads(report)["connections"][0]["provider"] == "slack"


def test_plaintext_fallback_is_0600_and_prominently_warned(tmp_path: Path) -> None:
    store = TokenStore(tmp_path, backend=FailedKeyring())
    store.save(token())
    assert store.fallback.stat().st_mode & 0o777 == 0o600
    assert "access-never-print" in store.fallback.read_text(encoding="utf-8")
    status = store.statuses()[0]
    assert status.storage == "file"
    assert status.warning and "0600" in status.warning
    store.delete(token().storage_key)
    assert token().storage_key not in store.fallback.read_text(encoding="utf-8")


def _callback_request(callback: LocalCallback, query: str) -> tuple[BaseException | None, int, str]:
    response: list[tuple[int, str]] = []

    def request() -> None:
        try:
            opened = urllib.request.urlopen(f"{callback.redirect_uri}?{query}", timeout=2)
            response.append((opened.status, opened.read().decode()))
        except urllib.error.HTTPError as exc:
            response.append((exc.code, exc.read().decode()))

    thread = threading.Thread(target=request)
    thread.start()
    error: BaseException | None = None
    try:
        callback.wait()
    except BaseException as exc:
        error = exc
    thread.join()
    assert len(response) == 1
    return error, response[0][0], response[0][1]


def test_callback_rejects_csrf_state_mismatch_and_cleans_port() -> None:
    callback = LocalCallback("127.0.0.1", 0, "expected", timeout=2)
    error, status, body = _callback_request(callback, "code=ok&state=wrong")
    assert isinstance(error, StateMismatchError)
    assert status == 400
    assert "rejected" in body.lower() and "accepted" not in body.lower()
    replacement = LocalCallback("127.0.0.1", callback.port, "new", timeout=0.1)
    replacement.close()


def test_hung_callback_is_bounded_and_releases_listener() -> None:
    callback = LocalCallback("127.0.0.1", 0, "expected", timeout=0.01)
    port = callback.port
    with pytest.raises(OAuthError, match="timed out"):
        callback.wait()
    replacement = LocalCallback("127.0.0.1", port, "new", timeout=0.01)
    replacement.close()


def test_callback_rejects_non_callback_path_without_consuming_callback() -> None:
    callback = LocalCallback("127.0.0.1", 0, "expected", timeout=0.01)
    try:
        with pytest.raises(urllib.error.HTTPError) as request_error:
            urllib.request.urlopen(
                f"http://127.0.0.1:{callback.port}/wrong-path?code=ok&state=expected",
                timeout=2,
            )
        assert request_error.value.code == 404
        with pytest.raises(OAuthError, match="timed out"):
            callback.wait()
    finally:
        callback.close()


def test_callback_transition_is_one_shot_under_concurrent_requests() -> None:
    callback = LocalCallback("127.0.0.1", 0, "expected", timeout=2)
    barrier = threading.Barrier(3)
    responses: list[int] = []

    def request(code: str) -> None:
        barrier.wait()
        try:
            with urllib.request.urlopen(
                f"{callback.redirect_uri}?code={code}&state=expected", timeout=2
            ) as opened:
                opened.read()
                responses.append(opened.status)
        except urllib.error.HTTPError as exc:
            exc.read()
            responses.append(exc.code)

    threads = [threading.Thread(target=request, args=(f"code-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    result = callback.wait()

    assert sorted(responses) == [200, 409]
    assert result.code in {"code-0", "code-1"}
    assert result.state == "expected"


def test_callback_wait_does_not_shutdown_during_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = LocalCallback("127.0.0.1", 0, "expected", timeout=2)
    response_started = threading.Event()
    release_response = threading.Event()
    response_finished = threading.Event()
    response: list[int] = []
    shutdown_started_before_response = False

    original_finish = BaseHTTPRequestHandler.finish

    def blocked_finish(handler: BaseHTTPRequestHandler) -> None:
        response_started.set()
        assert release_response.wait(2)
        original_finish(handler)
        response_finished.set()

    monkeypatch.setattr(BaseHTTPRequestHandler, "finish", blocked_finish)

    original_shutdown = callback.shutdown

    def shutdown() -> None:
        nonlocal shutdown_started_before_response
        shutdown_started_before_response = not response_finished.is_set()
        original_shutdown()

    monkeypatch.setattr(callback, "shutdown", shutdown)

    def request() -> None:
        with urllib.request.urlopen(
            f"{callback.redirect_uri}?code=ok&state=expected", timeout=2
        ) as opened:
            opened.read()
            response.append(opened.status)

    request_thread = threading.Thread(target=request)
    request_thread.start()
    assert response_started.wait(2)

    wait_thread = threading.Thread(target=callback.wait)
    wait_thread.start()
    release_response.set()
    request_thread.join(2)
    assert not request_thread.is_alive()
    wait_thread.join(2)
    assert not wait_thread.is_alive()
    assert response == [200]
    assert response_finished.is_set()
    assert not shutdown_started_before_response


def test_callback_reports_consent_denial() -> None:
    callback = LocalCallback("127.0.0.1", 0, "expected", timeout=2)
    error, status, body = _callback_request(callback, "error=access_denied&state=expected")
    assert isinstance(error, ConsentDeniedError)
    assert status == 403
    assert "denied" in body.lower() and "accepted" not in body.lower()


def test_callback_success_is_the_only_success_response() -> None:
    callback = LocalCallback("127.0.0.1", 0, "expected", timeout=2)
    error, status, body = _callback_request(callback, "code=ok&state=expected")
    assert error is None
    assert status == 200
    assert "accepted" in body.lower()


def test_stale_token_state_reports_reauthorization_without_leaking(tmp_path: Path) -> None:
    backend = MemoryKeyring()
    store = TokenStore(tmp_path, backend=backend)
    store.save(token())
    backend.values[("autobrain.oauth", token().storage_key)] = "{malformed-secret"
    status = store.statuses()[0]
    assert status.state.value == "REAUTHORIZATION_REQUIRED"
    assert status.warning and "unavailable" in status.warning
    assert "malformed-secret" not in status.model_dump_json()


def test_atomic_write_interruption_leaves_no_partial_secret_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = TokenStore(tmp_path, backend=FailedKeyring())
    real_replace = os.replace

    def interrupt(_source: str, _target: str) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        store.save(token())
    monkeypatch.setattr(os, "replace", real_replace)
    assert not store.fallback.exists()
    assert list(tmp_path.glob(".oauth-tokens.json.*")) == []


def test_callback_rejects_occupied_fixed_port_without_fallback() -> None:
    import socket

    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    busy_port = blocker.getsockname()[1]
    try:
        with pytest.raises(OSError):
            LocalCallback("127.0.0.1", busy_port, "expected", timeout=2)
    finally:
        blocker.close()
