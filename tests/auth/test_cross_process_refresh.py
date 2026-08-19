import json
import multiprocessing
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, cast

import httpx
from pydantic import SecretStr

from autobrain.auth import providers
from autobrain.auth.models import OAuthClient, Provider, TokenRecord
from autobrain.auth.oauth import OAuthManager
from autobrain.auth.providers import ProviderConfig
from autobrain.auth.storage import TokenStore


class NoKeyring:
    priority: float = 0.0

    def set_password(self, service: str, username: str, password: str) -> None:
        del service, username, password

    def get_password(self, service: str, username: str) -> str | None:
        del service, username
        return None

    def delete_password(self, service: str, username: str) -> None:
        del service, username


def _refresh_worker(root: str, key: str, base: str, barrier: Any, sender: Connection) -> None:
    original = providers.config_for(Provider.NOTION)
    providers.CONFIGS[Provider.NOTION] = ProviderConfig(
        Provider.NOTION, base + "/mcp", original.scopes, original.allowlist, True, False
    )
    store = TokenStore(Path(root), backend=NoKeyring())
    manager = OAuthManager(
        store,
        http=httpx.Client(trust_env=False),
        allow_localhost_http=True,
        timeout=2,
    )
    record = store.get(key)
    assert record is not None
    barrier.wait(timeout=10)
    updated = manager.refresh(record, OAuthClient(client_id="client"))
    sender.send(updated.refresh_token.get_secret_value() if updated.refresh_token else "")
    sender.close()


def test_two_processes_emit_one_refresh_and_one_atomic_rotation(tmp_path: Path) -> None:
    token_requests = 0
    request_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/.well-known/oauth-protected-resource/mcp":
                payload: dict[str, object] = {
                    "resource": base + "/mcp",
                    "authorization_servers": [base],
                }
            else:
                payload = {
                    "issuer": base,
                    "authorization_endpoint": base + "/authorize",
                    "token_endpoint": base + "/token",
                }
            self._json(200, payload)

        def do_POST(self) -> None:
            nonlocal token_requests
            assert self.path == "/token"
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode()
            assert "refresh_token=old-refresh" in body
            with request_lock:
                token_requests += 1
            self._json(
                200,
                {
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "workspace_id": "W",
                    "user_id": "U",
                    "audience": base + "/mcp",
                },
            )

        def _json(self, status: int, payload: dict[str, object]) -> None:
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original = providers.config_for(Provider.NOTION)
    providers.CONFIGS[Provider.NOTION] = ProviderConfig(
        Provider.NOTION, base + "/mcp", original.scopes, original.allowlist, True, False
    )
    processes: list[BaseProcess] = []
    try:
        store = TokenStore(tmp_path / "auth", backend=NoKeyring())
        record = TokenRecord(
            provider=Provider.NOTION,
            workspace_id="W",
            user_id="U",
            audience=base + "/mcp",
            access_token=SecretStr("old-access"),
            refresh_token=SecretStr("old-refresh"),
            oauth_client_id="client",
        )
        store.save(record)
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(3)
        receivers: list[Connection] = []
        for _ in range(2):
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(
                target=_refresh_worker,
                args=(str(store.root), record.storage_key, base, barrier, sender),
            )
            receivers.append(cast(Connection, receiver))
            processes.append(process)
            process.start()
            sender.close()
        barrier.wait(timeout=10)
        assert all(receiver.poll(10) for receiver in receivers)
        values = [receiver.recv() for receiver in receivers]
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
        assert values == ["new-refresh", "new-refresh"]
        assert token_requests == 1
        persisted = store.get(record.storage_key)
        assert persisted and persisted.refresh_token
        assert persisted.refresh_token.get_secret_value() == "new-refresh"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(2)
        providers.CONFIGS[Provider.NOTION] = original
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
