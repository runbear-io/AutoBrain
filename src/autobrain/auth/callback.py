"""Bounded localhost OAuth callback receiver."""

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Lock, Thread
from urllib.parse import parse_qs, urlparse

from autobrain.auth.models import ConsentDeniedError, OAuthError, StateMismatchError


@dataclass(frozen=True)
class CallbackResult:
    code: str
    state: str


class LocalCallback:
    def __init__(
        self, host: str, port: int, expected_state: str, *, timeout: float = 120.0
    ) -> None:
        self.host, self.port, self.expected_state, self.timeout = (
            host,
            port,
            expected_state,
            timeout,
        )
        self._result: CallbackResult | None = None
        self._error: BaseException | None = None
        self._done = Event()
        self._close_lock = Lock()
        self._closed = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
                state = query.get("state", [""])[0]
                if owner._done.is_set():
                    status, message = 409, b"Authorization callback was already handled."
                elif state != owner.expected_state:
                    owner._error = StateMismatchError("OAuth callback state mismatch")
                    status, message = 400, b"Authorization rejected: invalid state."
                elif query.get("error", [""])[0]:
                    owner._error = ConsentDeniedError("OAuth consent was denied")
                    status, message = 403, b"Authorization was denied. No credentials were stored."
                elif not query.get("code", [""])[0]:
                    owner._error = OAuthError("OAuth callback did not contain a code")
                    status, message = 400, b"Authorization rejected: missing authorization code."
                else:
                    owner._result = CallbackResult(query["code"][0], state)
                    status, message = 200, b"Authorization accepted. You may close this window."
                self.send_response(status)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(message)
                owner._done.set()

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        try:
            self._server = ThreadingHTTPServer((host, port), Handler)
        except OSError:
            if port == 0:
                raise
            self._server = ThreadingHTTPServer((host, 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def redirect_uri(self) -> str:
        return f"http://{self.host}:{self.port}/oauth/callback"

    def wait(self) -> CallbackResult:
        try:
            if not self._done.wait(self.timeout):
                raise OAuthError("OAuth callback timed out")
            if self._error:
                raise self._error
            if self._result is None:
                raise OAuthError("OAuth callback ended without a result")
            return self._result
        finally:
            self.close()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._server.shutdown()
            self._server.server_close()
        self._thread.join(timeout=2)
