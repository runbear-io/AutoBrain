from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar, cast

from autobrain.candidates.mem0 import Mem0Adapter, Mem0AdapterConfig
from autobrain.models import NormalizedDocument, SourceKind


class OpenAIHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[str]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        request = cast(dict[str, Any], json.loads(self.rfile.read(length)))
        self.__class__.requests.append(self.path)
        if self.path.endswith("/embeddings"):
            inputs = request["input"]
            count = len(cast(list[Any], inputs)) if isinstance(inputs, list) else 1
            body: dict[str, Any] = {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": index, "embedding": [0.01] * 1536}
                    for index in range(count)
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": count, "total_tokens": count},
            }
        else:
            body = {
                "id": "loopback-answer",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-5-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": (
                                '{"answer":"Tuesday.","claim_ids":["claim-1"],'
                                '"source_ids":["notion:loopback"]}'
                            ),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            }
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def test_real_mem0_loopback_lifecycle_closes_without_shutdown_warning(
    tmp_path: Path, capsys: Any
) -> None:
    OpenAIHandler.requests.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), OpenAIHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    host = str(server.server_address[0])
    port = int(server.server_address[1])
    text = "The loopback project launches on Tuesday."
    document = NormalizedDocument(
        source_id="notion:loopback",
        source_kind=SourceKind.NOTION_PAGE,
        canonical_url="https://example.test/loopback",
        title="Loopback",
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
    adapter: Mem0Adapter | None = None
    try:
        adapter = Mem0Adapter(
            Mem0AdapterConfig(
                run_id="real-loopback",
                run_dir=tmp_path,
                heldout_source_ids=set(),
                api_key="loopback-key",
                base_url=f"http://{host}:{port}/v1",
                timeout_seconds=2,
            )
        )
        ingested = adapter.ingest([document])
        native = adapter.search_native("When does the project launch?", top_k=1)
        answer = adapter.answer("When does the project launch?", native["results"])
        adapter.record_proxy_event(
            {
                "request_id": "loopback-answer",
                "model": "gpt-5-mini",
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            },
            phase="answer",
        )
        receipt = adapter.cleanup()
        repeated = adapter.cleanup()
        assert ingested.memory_ids
        assert answer.source_ids == ["notion:loopback"]
        assert adapter.timeout_evidence.configured_native_clients == ["embedding_model", "llm"]
        answer_events = [
            event for event in adapter.usage_events if event.request_id == "loopback-answer"
        ]
        assert len(answer_events) == 1
        assert answer_events[0].source == "answer_callback+proxy"
        assert "qdrant" in receipt.closed_resources
        assert repeated.already_clean
        assert not adapter.config.qdrant_path.exists()
    finally:
        if adapter is not None and adapter.config.qdrant_path.exists():
            adapter.cleanup()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert not thread.is_alive()
    captured = capsys.readouterr()
    assert "Exception ignored" not in captured.err
    assert "sys.meta_path is None" not in captured.err
    assert OpenAIHandler.requests.count("/v1/embeddings") >= 2
    assert "/v1/chat/completions" in OpenAIHandler.requests
