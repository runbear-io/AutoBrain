from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

from autobrain.candidates.gbrain import (
    GBRAIN_COMMIT,
    GBRAIN_VERSION,
    GBrainAdapter,
)
from autobrain.candidates.gbrain_config import GBrainExecutionConfig
from autobrain.models import NormalizedDocument, SourceKind


class _Provider(ThreadingHTTPServer):
    events: list[dict[str, Any]]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _write(self, value: dict[str, Any]) -> None:
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._write(
            {
                "object": "list",
                "data": [
                    {"id": "gpt-5-mini", "object": "model"},
                    {"id": "text-embedding-3-small", "object": "model"},
                ],
            }
        )

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        request = cast(dict[str, Any], json.loads(self.rfile.read(length) or b"{}"))
        provider = cast(_Provider, self.server)
        event_id = f"event-{len(provider.events) + 1}"
        if self.path.endswith("/embeddings"):
            raw_input = request.get("input", [])
            inputs = cast(list[Any], raw_input) if isinstance(raw_input, list) else [raw_input]
            tokens = max(1, len(inputs) * 5)
            provider.events.append(
                {
                    "event_id": event_id,
                    "phase": "ingest",
                    "model": "text-embedding-3-small",
                    "input_tokens": tokens,
                    "output_tokens": 0,
                    "usd": tokens / 1_000_000 * 0.02,
                }
            )
            self._write(
                {
                    "object": "list",
                    "model": "text-embedding-3-small",
                    "data": [
                        {
                            "object": "embedding",
                            "index": index,
                            "embedding": [0.01] * 1536,
                        }
                        for index, _value in enumerate(inputs)
                    ],
                    "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
                }
            )
            return

        answer = json.dumps(
            {
                "answer": "Blue [canonical]",
                "citations": [{"page_slug": "canonical", "row_num": None}],
                "gaps": [],
            }
        )
        provider.events.append(
            {
                "event_id": event_id,
                "phase": "query",
                "operation": "think",
                "model": "gpt-5-mini",
                "input_tokens": 11,
                "output_tokens": 3,
                "usd": 0.001,
            }
        )
        if self.path.endswith("/responses"):
            self._write(
                {
                    "id": "response-fixture",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-5-mini",
                    "output": [
                        {
                            "id": "message-fixture",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": answer, "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
                }
            )
            return
        self._write(
            {
                "id": "chat-fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-5-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
            }
        )


@pytest.fixture
def local_provider() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    server = _Provider(("127.0.0.1", 0), _Handler)
    server.events = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", server.events
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _document() -> NormalizedDocument:
    text = "The canonical answer is blue."
    return NormalizedDocument(
        source_id="notion:canonical",
        source_kind=SourceKind.NOTION_PAGE,
        canonical_url="https://example.test/canonical",
        title="Canonical",
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


def test_real_pinned_runtime_reaches_status_search_query_and_think(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_provider: tuple[str, list[dict[str, Any]]],
) -> None:
    base_url, events = local_provider
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    adapter = GBrainAdapter(
        tmp_path / "tools",
        tmp_path / "run",
        timeout_seconds=300,
        config=GBrainExecutionConfig.semantic(
            "openai", credential="fixture", chat_credential="fixture"
        ),
    )
    result = adapter.run(
        [_document()],
        ["What is the canonical answer?"],
        base_url=base_url,
        proxy_events=events,
    )[0]

    assert result.commit == GBRAIN_COMMIT
    assert result.version == GBRAIN_VERSION
    assert result.status == "OK"
    assert result.native["status"]["mode"] == "local"
    assert result.search_evidence
    assert result.query_evidence
    assert result.answer == "Blue [canonical]"
    assert result.gather_evidence["pages_gathered"] == 1
    assert result.cost_status == "COST_COMPLETE"
    assert result.proxy_usage is not None
    assert result.proxy_usage["input_tokens"] >= 11
    sync = next(command.command for command in result.commands if command.command[2] == "sync")
    assert sync[3:] == ("--repo", str(adapter.sources), "--no-pull", "--json")
    assert any("SYNC_UNAVAILABLE" in warning for warning in result.warnings)


def test_semantic_hosted_provider_requires_run_memory_credential() -> None:
    with pytest.raises(ValueError, match="requires a credential"):
        GBrainExecutionConfig.semantic("openai")


def test_keyless_default_has_no_openai_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = GBrainExecutionConfig.quick_start()
    assert config.keyword_only
    assert config.child_environment() == {}
