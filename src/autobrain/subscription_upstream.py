"""Subscription-compatible chat upstream and deterministic local embeddings."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from typing import Protocol, cast


class AnswerClient(Protocol):
    def ask(self, prompt: str) -> str: ...


def local_embedding(text: str, *, dimensions: int = 1536) -> list[float]:
    """Create a deterministic local embedding without provider billing."""
    vector = [0.0] * dimensions
    tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [round(value / norm, 8) for value in vector]


def build_subscription_upstream(
    client: AnswerClient,
) -> Callable[[dict[str, object]], dict[str, object]]:
    """Build an OpenAI-compatible upstream backed by subscription chat and local vectors."""

    def upstream(payload: dict[str, object]) -> dict[str, object]:
        model = payload.get("model")
        if isinstance(model, str) and model.startswith("text-embedding"):
            inputs = payload.get("input")
            if isinstance(inputs, str):
                texts = [inputs]
            elif isinstance(inputs, list):
                input_items = cast(list[object], inputs)
                texts = [item for item in input_items if isinstance(item, str)]
            else:
                texts = []
            return {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": local_embedding(text),
                    }
                    for index, text in enumerate(texts)
                ],
                "model": "local-hash-embedding",
                "usage": {
                    "prompt_tokens": sum(_estimate_tokens(text) for text in texts),
                    "total_tokens": sum(_estimate_tokens(text) for text in texts),
                },
            }

        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("subscription chat payload requires messages")
        message_items = cast(list[object], messages)
        message_dicts = [
            cast(dict[str, object], message)
            for message in message_items
            if isinstance(message, dict)
        ]
        prompt_parts = [
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in message_dicts
        ]
        answer = client.ask("\n".join(prompt_parts))
        return {
            "id": "subscription-chat-completion",
            "object": "chat.completion",
            "model": "chatgpt-subscription",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": _estimate_tokens("\n".join(prompt_parts)),
                "completion_tokens": _estimate_tokens(answer),
            },
        }

    return upstream


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))
