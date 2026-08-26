"""Browser reachability contract for the local run fixture.

A browser served from the Vite dev origin (http://localhost:5173) must be able
to read the fixture on its own loopback origin. That requires CORS, but the
allowlist stays deliberately narrow: only loopback dev origins are echoed back,
never a wildcard and never a remote origin, and credentials are never allowed
because this fixture has no authentication to protect.
"""

from __future__ import annotations

from http.client import HTTPConnection, HTTPResponse

import pytest

from autobrain.local_server import (
    DEFAULT_LOCAL_PORT,
    LocalRunServer,
    RunOutcome,
    is_allowed_origin,
)
from tests.test_projection import artifact, project_comparison

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:8765",
    "http://[::1]:5173",
]

REJECTED_ORIGINS = [
    "https://evil.example",
    "http://evil.example",
    "http://localhost.evil.example",
    "http://127.0.0.1.evil.example",
    "https://localhost:5173",
    "http://192.168.1.10:5173",
    "http://0.0.0.0:5173",
    "null",
    "file://",
]


def request(
    server: LocalRunServer,
    method: str = "GET",
    path: str = "/api/v1/run",
    origin: str | None = None,
) -> HTTPResponse:
    connection = HTTPConnection(server.host, server.port, timeout=5)
    headers = {"Origin": origin} if origin is not None else {}
    connection.request(method, path, headers=headers)
    return connection.getresponse()


@pytest.fixture
def server():
    outcome = RunOutcome.succeeded(project_comparison(artifact()))
    with LocalRunServer(lambda: outcome, port=0) as running:
        yield running


@pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
def test_loopback_dev_origins_are_echoed_back_exactly(server: LocalRunServer, origin: str) -> None:
    response = request(server, origin=origin)

    assert response.status == 200
    assert response.getheader("Access-Control-Allow-Origin") == origin
    assert "Origin" in (response.getheader("Vary") or "")


@pytest.mark.parametrize("origin", REJECTED_ORIGINS)
def test_non_loopback_origins_receive_no_cors_grant(server: LocalRunServer, origin: str) -> None:
    response = request(server, origin=origin)

    assert response.getheader("Access-Control-Allow-Origin") is None


def test_cors_is_never_a_wildcard(server: LocalRunServer) -> None:
    response = request(server, origin="http://localhost:5173")

    assert response.getheader("Access-Control-Allow-Origin") != "*"


def test_credentials_are_never_allowed(server: LocalRunServer) -> None:
    """The fixture has no auth, so it must never opt into credentialed CORS."""
    response = request(server, origin="http://localhost:5173")

    assert response.getheader("Access-Control-Allow-Credentials") is None


def test_preflight_is_answered_for_a_loopback_origin(server: LocalRunServer) -> None:
    response = request(server, method="OPTIONS", origin="http://localhost:5173")

    assert response.status == 204
    assert response.getheader("Access-Control-Allow-Origin") == "http://localhost:5173"
    assert "GET" in (response.getheader("Access-Control-Allow-Methods") or "")


def test_preflight_from_a_remote_origin_is_refused(server: LocalRunServer) -> None:
    response = request(server, method="OPTIONS", origin="https://evil.example")

    assert response.getheader("Access-Control-Allow-Origin") is None


def test_a_request_without_an_origin_still_works(server: LocalRunServer) -> None:
    """Non-browser clients such as curl send no Origin and must not be broken."""
    response = request(server)

    assert response.status == 200
    assert response.getheader("Access-Control-Allow-Origin") is None


@pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
def test_allowed_origin_predicate_accepts_loopback(origin: str) -> None:
    assert is_allowed_origin(origin) is True


@pytest.mark.parametrize("origin", REJECTED_ORIGINS)
def test_allowed_origin_predicate_rejects_everything_else(origin: str) -> None:
    assert is_allowed_origin(origin) is False


def test_default_port_matches_the_documented_client_default() -> None:
    """The browser client defaults to this port, so the server must too."""
    assert DEFAULT_LOCAL_PORT == 8765


def test_server_binds_the_requested_port() -> None:
    outcome = RunOutcome.cancelled()
    with LocalRunServer(lambda: outcome, port=0) as running:
        chosen = running.port
        assert chosen > 0

    with LocalRunServer(lambda: outcome, port=chosen) as reopened:
        assert reopened.port == chosen


def test_server_defaults_to_the_shared_local_port() -> None:
    with LocalRunServer(lambda: RunOutcome.cancelled()) as running:
        assert running.port == DEFAULT_LOCAL_PORT


def test_server_still_binds_loopback_only_with_cors_enabled(server: LocalRunServer) -> None:
    assert server.host == "127.0.0.1"
    assert server.base_url.startswith("http://127.0.0.1:")
