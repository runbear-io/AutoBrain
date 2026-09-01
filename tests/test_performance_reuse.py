from __future__ import annotations

import hashlib

from autobrain.benchmark import scan_benchmark_leakage
from autobrain.models import NormalizedDocument, SourceKind
from autobrain.performance import RunCache


def _document(source_id: str, text: str = "alpha") -> NormalizedDocument:
    return NormalizedDocument(
        source_id=source_id,
        source_kind=SourceKind.NOTION_PAGE,
        canonical_url=f"https://example.test/{source_id}",
        title=source_id,
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


def test_run_cache_reuses_normalization_serialization_hash_and_source_index() -> None:
    cache = RunCache()
    raw = [_document("notion:page:one")]
    first = cache.normalized_documents(raw)
    second = cache.normalized_documents(raw)
    assert first == second
    assert cache.serialize(first) == cache.serialize(first)
    assert cache.hash_json(first) == cache.hash_json(first)
    assert cache.source_index(first) is cache.source_index(first)
    assert cache.counters.normalized_documents == 1
    assert cache.counters.normalized_cache_hits == 1
    assert cache.counters.serialization_cache_hits >= 1
    assert cache.counters.hash_cache_hits == 1
    assert cache.counters.source_index_builds == 1


def test_answer_tokens_are_cached_without_cross_run_state() -> None:
    first = RunCache()
    second = RunCache()

    def tokenizer(value: str) -> list[str]:
        return value.split()

    assert first.answer_tokens("alpha beta", tokenizer) == frozenset({"alpha", "beta"})
    assert first.answer_tokens("alpha beta", tokenizer) == frozenset({"alpha", "beta"})
    assert first.counters.answer_tokenizations == 1
    assert first.counters.answer_token_cache_hits == 1
    assert second.counters.answer_tokenizations == 0


def test_leakage_scan_can_reuse_one_serialized_payload() -> None:
    cache = RunCache()
    payload = {"answer": "safe"}
    serialized = cache.serialize(payload, leakage=True)
    result = scan_benchmark_leakage(
        texts=(serialized,), serialized_artifacts=(serialized,), forbidden_tokens=("secret",)
    )
    assert result.clean
    assert cache.counters.leakage_serializations == 1
    assert cache.serialize(payload, leakage=True) == serialized
    assert cache.counters.leakage_cache_hits == 1
