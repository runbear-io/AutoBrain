import base64
import hashlib
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pydantic import SecretStr

from autobrain.auth import providers
from autobrain.auth.callback import CallbackResult as LocalCallbackResult
from autobrain.auth.callback import LocalCallback
from autobrain.auth.discovery import discover
from autobrain.auth.models import (
    OAuthClient,
    OAuthError,
    Provider,
    ReauthorizationRequired,
    TokenRecord,
    WorkspaceMismatchError,
)
from autobrain.auth.oauth import OAuthManager
from autobrain.auth.providers import ProviderConfig
from autobrain.auth.storage import TokenStore
from tests.auth.fakes import MemoryKeyring


class CallbackResult:
    code = "code-1"

    def __init__(self, state: str) -> None:
        self.state = state


class FakeCallback:
    redirect_uri = "http://127.0.0.1:8765/oauth/callback"

    def __init__(self, state: str) -> None:
        self.state = state

    def wait(self) -> CallbackResult:
        return CallbackResult(self.state)


class TrackingCallback(FakeCallback):
    def __init__(self, state: str, *, interrupt: bool = False) -> None:
        super().__init__(state)
        self.interrupt = interrupt
        self.closed = False

    def wait(self) -> CallbackResult:
        if self.interrupt:
            raise KeyboardInterrupt
        return super().wait()

    def close(self) -> None:
        self.closed = True


def fake_config(base: str, provider: Provider = Provider.NOTION) -> ProviderConfig:
    original = providers.config_for(provider)
    return ProviderConfig(
        provider,
        f"{base}/mcp",
        original.scopes,
        original.allowlist,
        provider is Provider.NOTION,
        provider is Provider.SLACK,
    )


def metadata_response(request: httpx.Request, base: str) -> httpx.Response | None:
    if request.url.path == "/.well-known/oauth-protected-resource/mcp":
        return httpx.Response(
            200, json={"resource": f"{base}/mcp", "authorization_servers": [base]}
        )
    if request.url.path == "/.well-known/oauth-authorization-server":
        return httpx.Response(
            200,
            json={
                "issuer": base,
                "authorization_endpoint": f"{base}/authorize",
                "token_endpoint": f"{base}/token",
                "registration_endpoint": f"{base}/register",
                "scopes_supported": ["read_content"],
            },
        )
    return None


def test_notion_dcr_pkce_and_token_exchange(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = "http://127.0.0.1:9001"
    monkeypatch.setitem(providers.CONFIGS, Provider.NOTION, fake_config(base))
    challenge: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        meta = metadata_response(request, base)
        if meta:
            return meta
        if request.url.path == "/register":
            payload = request.read().decode()
            assert "client_name" in payload and "client_secret" not in payload
            return httpx.Response(201, json={"client_id": "dynamic-notion"})
        if request.url.path == "/token":
            form = parse_qs(request.read().decode())
            verifier = form["code_verifier"][0]
            actual = (
                base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
                .rstrip(b"=")
                .decode()
            )
            assert actual == challenge[0]
            assert form["resource"] == [f"{base}/mcp"]
            return httpx.Response(
                200,
                json={
                    "access_token": "notion-access-secret",
                    "refresh_token": "notion-refresh-secret",
                    "workspace_id": "notion-workspace",
                    "user_id": "notion-user",
                    "resource": f"{base}/mcp",
                },
            )
        raise AssertionError(request.url)

    def open_browser(url: str) -> bool:
        query = parse_qs(urlparse(url).query)
        challenge.append(query["code_challenge"][0])
        assert query["code_challenge_method"] == ["S256"]
        assert query["resource"] == [f"{base}/mcp"]
        return True

    store = TokenStore(tmp_path, backend=MemoryKeyring())
    manager = OAuthManager(
        store,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        browser_open=open_browser,
        callback_factory=FakeCallback,
        allow_localhost_http=True,
    )
    token = manager.authorize(Provider.NOTION)
    assert token.storage_key == "notion:notion-workspace:notion-user"
    assert store.get(token.storage_key) == token
    assert "notion-access-secret" not in token.model_dump_json()


def test_slack_uses_fixed_client_and_never_dcr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = "http://127.0.0.1:9002"
    monkeypatch.setitem(providers.CONFIGS, Provider.SLACK, fake_config(base, Provider.SLACK))
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        meta = metadata_response(request, base)
        if meta:
            return meta
        if request.url.path == "/token":
            form = parse_qs(request.read().decode())
            assert form["client_id"] == ["fixed-id"]
            assert form["client_secret"] == ["fixed-secret"]
            return httpx.Response(
                200,
                json={
                    "access_token": "xoxp-secret",
                    "refresh_token": "rotation-1",
                    "team": {"id": "T1"},
                    "authed_user": {"id": "U1"},
                    "audience": f"{base}/mcp",
                },
            )
        raise AssertionError(request.url)

    manager = OAuthManager(
        TokenStore(tmp_path, backend=MemoryKeyring()),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        browser_open=lambda _url: True,
        callback_factory=FakeCallback,
        allow_localhost_http=True,
    )
    manager.authorize(
        Provider.SLACK, slack_client_id="fixed-id", slack_client_secret="fixed-secret"
    )
    assert "/register" not in paths


def test_refresh_rotation_is_serialized_and_replay_coalesced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = "http://127.0.0.1:9003"
    monkeypatch.setitem(providers.CONFIGS, Provider.NOTION, fake_config(base))
    calls = 0
    calls_lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        meta = metadata_response(request, base)
        if meta:
            return meta
        if request.url.path == "/token":
            with calls_lock:
                calls += 1
            form = parse_qs(request.read().decode())
            assert form["refresh_token"] == ["old-refresh"]
            return httpx.Response(
                200,
                json={
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "workspace_id": "W1",
                    "user_id": "U1",
                    "audience": f"{base}/mcp",
                },
            )
        raise AssertionError(request.url)

    store = TokenStore(tmp_path, backend=MemoryKeyring())
    record = TokenRecord(
        provider=Provider.NOTION,
        workspace_id="W1",
        user_id="U1",
        audience=f"{base}/mcp",
        access_token=SecretStr("old-access"),
        refresh_token=SecretStr("old-refresh"),
    )
    store.save(record)
    manager = OAuthManager(
        store, http=httpx.Client(transport=httpx.MockTransport(handler)), allow_localhost_http=True
    )
    barrier = threading.Barrier(3)
    results: list[TokenRecord] = []

    def refresh() -> None:
        barrier.wait()
        results.append(manager.refresh(record, OAuthClient(client_id="dcr")))

    threads = [threading.Thread(target=refresh) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    assert calls == 1
    assert {item.refresh_token.get_secret_value() for item in results if item.refresh_token} == {
        "new-refresh"
    }


def test_invalid_grant_is_terminal_and_clears_stale_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = "http://127.0.0.1:9004"
    monkeypatch.setitem(providers.CONFIGS, Provider.NOTION, fake_config(base))
    token_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_calls
        meta = metadata_response(request, base)
        if meta:
            return meta
        token_calls += 1
        return httpx.Response(400, json={"error": "invalid_grant"})

    store = TokenStore(tmp_path, backend=MemoryKeyring())
    record = TokenRecord(
        provider=Provider.NOTION,
        workspace_id="W",
        user_id="U",
        audience=f"{base}/mcp",
        access_token=SecretStr("a"),
        refresh_token=SecretStr("r"),
    )
    store.save(record)
    manager = OAuthManager(
        store, http=httpx.Client(transport=httpx.MockTransport(handler)), allow_localhost_http=True
    )
    with pytest.raises(ReauthorizationRequired):
        manager.refresh(record, OAuthClient(client_id="client"))
    assert token_calls == 1
    assert store.get(record.storage_key) is None


def test_refresh_workspace_mismatch_preserves_previous_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = "http://127.0.0.1:9005"
    monkeypatch.setitem(providers.CONFIGS, Provider.NOTION, fake_config(base))

    def handler(request: httpx.Request) -> httpx.Response:
        meta = metadata_response(request, base)
        if meta:
            return meta
        return httpx.Response(
            200,
            json={
                "access_token": "new",
                "refresh_token": "rotated",
                "workspace_id": "OTHER",
                "user_id": "U",
                "audience": f"{base}/mcp",
            },
        )

    store = TokenStore(tmp_path, backend=MemoryKeyring())
    record = TokenRecord(
        provider=Provider.NOTION,
        workspace_id="W",
        user_id="U",
        audience=f"{base}/mcp",
        access_token=SecretStr("old"),
        refresh_token=SecretStr("refresh"),
    )
    store.save(record)
    manager = OAuthManager(
        store, http=httpx.Client(transport=httpx.MockTransport(handler)), allow_localhost_http=True
    )
    with pytest.raises(WorkspaceMismatchError):
        manager.refresh(record, OAuthClient(client_id="client"))
    assert store.get(record.storage_key) == record


def test_pkce_rejection_or_misleading_token_success_is_not_stored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = "http://127.0.0.1:9007"
    monkeypatch.setitem(providers.CONFIGS, Provider.NOTION, fake_config(base))

    def rejected(request: httpx.Request) -> httpx.Response:
        meta = metadata_response(request, base)
        if meta:
            return meta
        if request.url.path == "/register":
            return httpx.Response(201, json={"client_id": "dynamic"})
        return httpx.Response(
            400, json={"error": "invalid_grant", "error_description": "PKCE mismatch"}
        )

    store = TokenStore(tmp_path, backend=MemoryKeyring())
    manager = OAuthManager(
        store,
        http=httpx.Client(transport=httpx.MockTransport(rejected)),
        browser_open=lambda _url: True,
        callback_factory=FakeCallback,
        allow_localhost_http=True,
    )
    with pytest.raises(OAuthError, match="token exchange failed"):
        manager.authorize(Provider.NOTION)
    assert not store.index.exists()


def test_token_audience_mismatch_is_rejected_before_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = "http://127.0.0.1:9008"
    monkeypatch.setitem(providers.CONFIGS, Provider.NOTION, fake_config(base))

    def misleading(request: httpx.Request) -> httpx.Response:
        meta = metadata_response(request, base)
        if meta:
            return meta
        if request.url.path == "/register":
            return httpx.Response(201, json={"client_id": "dynamic"})
        return httpx.Response(
            200,
            json={
                "access_token": "must-not-store",
                "workspace_id": "W",
                "user_id": "U",
                "audience": "https://mcp.slack.com/mcp",
            },
        )

    store = TokenStore(tmp_path, backend=MemoryKeyring())
    manager = OAuthManager(
        store,
        http=httpx.Client(transport=httpx.MockTransport(misleading)),
        browser_open=lambda _url: True,
        callback_factory=FakeCallback,
        allow_localhost_http=True,
    )
    with pytest.raises(OAuthError, match="audience mismatch"):
        manager.authorize(Provider.NOTION)
    assert not store.index.exists()


def test_hung_token_endpoint_is_bounded_and_redacted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = "http://127.0.0.1:9010"
    monkeypatch.setitem(providers.CONFIGS, Provider.NOTION, fake_config(base))

    def handler(request: httpx.Request) -> httpx.Response:
        meta = metadata_response(request, base)
        if meta:
            return meta
        if request.url.path == "/register":
            return httpx.Response(201, json={"client_id": "dynamic"})
        raise httpx.ReadTimeout("hung endpoint with secret-do-not-echo", request=request)

    store = TokenStore(tmp_path, backend=MemoryKeyring())
    manager = OAuthManager(
        store,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        browser_open=lambda _url: True,
        callback_factory=FakeCallback,
        allow_localhost_http=True,
        timeout=0.01,
    )
    with pytest.raises(OAuthError, match="ReadTimeout") as error:
        manager.authorize(Provider.NOTION)
    assert "secret-do-not-echo" not in str(error.value)
    assert not store.index.exists()


def test_authorization_interruption_leaves_no_token_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = "http://127.0.0.1:9009"
    monkeypatch.setitem(providers.CONFIGS, Provider.NOTION, fake_config(base))

    def handler(request: httpx.Request) -> httpx.Response:
        meta = metadata_response(request, base)
        if meta:
            return meta
        return httpx.Response(201, json={"client_id": "dynamic"})

    class InterruptedCallback(FakeCallback):
        def wait(self) -> CallbackResult:
            raise KeyboardInterrupt

    store = TokenStore(tmp_path, backend=MemoryKeyring())
    manager = OAuthManager(
        store,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        browser_open=lambda _url: True,
        callback_factory=InterruptedCallback,
        allow_localhost_http=True,
    )
    with pytest.raises(KeyboardInterrupt):
        manager.authorize(Provider.NOTION)
    assert not store.index.exists()


@pytest.mark.parametrize("failure", ["missing-slack", "dcr", "browser", "cancel"])
def test_callback_closes_for_every_pre_wait_and_cancellation_failure(
    failure: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = "http://127.0.0.1:9011"
    provider = Provider.SLACK if failure == "missing-slack" else Provider.NOTION
    monkeypatch.setitem(providers.CONFIGS, provider, fake_config(base, provider))
    callback: TrackingCallback | None = None

    def callback_factory(state: str) -> TrackingCallback:
        nonlocal callback
        callback = TrackingCallback(state, interrupt=failure == "cancel")
        return callback

    def handler(request: httpx.Request) -> httpx.Response:
        meta = metadata_response(request, base)
        if meta:
            return meta
        if request.url.path == "/register":
            if failure == "dcr":
                return httpx.Response(500, json={"error": "backend-secret"})
            return httpx.Response(201, json={"client_id": "dynamic"})
        raise AssertionError(request.url)

    manager = OAuthManager(
        TokenStore(tmp_path, backend=MemoryKeyring()),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        browser_open=(lambda _url: failure != "browser"),
        callback_factory=callback_factory,
        allow_localhost_http=True,
    )
    expected = KeyboardInterrupt if failure == "cancel" else OAuthError
    with pytest.raises(expected):
        manager.authorize(provider)
    assert callback is not None and callback.closed


@pytest.mark.parametrize("failure", ["missing-slack", "dcr", "browser", "cancel"])
def test_real_callback_port_is_released_for_all_pre_browser_failures(
    failure: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = "http://127.0.0.1:9012"
    provider = Provider.SLACK if failure == "missing-slack" else Provider.NOTION
    monkeypatch.setitem(providers.CONFIGS, provider, fake_config(base, provider))
    callback: LocalCallback | None = None

    class InterruptingCallback(LocalCallback):
        def wait(self) -> LocalCallbackResult:
            raise KeyboardInterrupt

    def callback_factory(state: str) -> LocalCallback:
        nonlocal callback
        callback_type = InterruptingCallback if failure == "cancel" else LocalCallback
        callback = callback_type("127.0.0.1", 0, state, timeout=0.1)
        return callback

    def handler(request: httpx.Request) -> httpx.Response:
        meta = metadata_response(request, base)
        if meta:
            return meta
        if request.url.path == "/register":
            return httpx.Response(
                500 if failure == "dcr" else 201,
                json={"error": "failed"} if failure == "dcr" else {"client_id": "dynamic"},
            )
        raise AssertionError(request.url)

    manager = OAuthManager(
        TokenStore(tmp_path, backend=MemoryKeyring()),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        browser_open=lambda _url: failure != "browser",
        callback_factory=callback_factory,
        allow_localhost_http=True,
    )
    expected = KeyboardInterrupt if failure == "cancel" else OAuthError
    with pytest.raises(expected):
        manager.authorize(provider)
    assert callback is not None
    replacement = LocalCallback("127.0.0.1", callback.port, "replacement", timeout=0.01)
    replacement.close()


def test_discovery_rejects_malformed_metadata_and_audience() -> None:
    base = "http://127.0.0.1:9006"
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"resource": "http://127.0.0.1:9006/other", "authorization_servers": [base]},
            )
        )
    )
    with pytest.raises(OAuthError, match="audience mismatch"):
        discover(f"{base}/mcp", client, allow_localhost_http=True)
