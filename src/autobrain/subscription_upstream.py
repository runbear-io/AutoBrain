"""Subscription-compatible chat upstream and deterministic local embeddings."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from typing import Protocol, cast

from autobrain.subscription_domain import ProviderAnswer, UsageKind


class AnswerClient(Protocol):
    def ask(self, prompt: str) -> str: ...

    def answer(self, prompt: str) -> ProviderAnswer: ...


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
    *,
    embedding_upstream: Callable[[dict[str, object]], dict[str, object]] | None = None,
) -> Callable[[dict[str, object]], dict[str, object]]:
    """Build an OpenAI-compatible upstream with explicit chat and embedding backends."""

    def upstream(payload: dict[str, object]) -> dict[str, object]:
        model = payload.get("model")
        if isinstance(model, str) and model.startswith("text-embedding"):
            if embedding_upstream is not None:
                return embedding_upstream(payload)
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
                "_autobrain_provider": "local",
                "_autobrain_usage_source": "estimated",
                "usage": {
                    "prompt_tokens": sum(_estimate_tokens(text) for text in texts),
                    "completion_tokens": 0,
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
        prompt = "\n".join(prompt_parts)
        answer_method = getattr(client, "answer", None)
        provider_answer = (
            answer_method(prompt)
            if callable(answer_method)
            and ("answer" in type(client).__dict__ or "ask" not in type(client).__dict__)
            else None
        )
        if isinstance(provider_answer, ProviderAnswer):
            answer = provider_answer.text
            identity = provider_answer.identity
            usage = provider_answer.usage
            provider = identity.provider.value
            actual_model = identity.model or "unavailable"
            usage_source = "measured" if usage.kind is UsageKind.NATIVE else usage.kind.value
            raw_usage = (
                {
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                }
                if usage.input_tokens is not None and usage.output_tokens is not None
                else {}
            )
            execution_ms = provider_answer.execution_ms
        else:
            answer = client.ask(prompt)
            provider = "codex"
            actual_model = "unavailable"
            usage_source = "estimated"
            raw_usage = {
                "prompt_tokens": _estimate_tokens(prompt),
                "completion_tokens": _estimate_tokens(answer),
            }
            execution_ms = None
        return {
            "id": "subscription-chat-completion",
            "object": "chat.completion",
            "model": actual_model,
            "_autobrain_provider": provider,
            "_autobrain_usage_source": usage_source,
            "_autobrain_execution_ms": execution_ms,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": raw_usage,
        }

    return upstream


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))
