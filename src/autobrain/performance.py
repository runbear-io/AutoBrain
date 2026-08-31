"""Run-scoped deterministic reuse for expensive, integrity-neutral work."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from autobrain.models import NormalizedDocument, StrictModel


@dataclass
class PerformanceCounters:
    """Small diagnostic counters; they never affect results or security gates."""

    normalized_documents: int = 0
    normalized_cache_hits: int = 0
    serializations: int = 0
    serialization_cache_hits: int = 0
    hashes: int = 0
    hash_cache_hits: int = 0
    answer_tokenizations: int = 0
    answer_token_cache_hits: int = 0
    source_index_builds: int = 0
    leakage_serializations: int = 0
    leakage_cache_hits: int = 0


@dataclass
class RunCache:
    """Memoize only immutable values within one run/build boundary."""

    counters: PerformanceCounters = field(default_factory=PerformanceCounters)
    _normalized: dict[tuple[object, ...], tuple[NormalizedDocument, ...]] = field(
        default_factory=dict
    )
    _serialized: dict[object, str] = field(default_factory=dict)
    _hashes: dict[object, str] = field(default_factory=dict)
    _answer_tokens: dict[str, frozenset[str]] = field(default_factory=dict)
    _source_indexes: dict[object, Mapping[str, NormalizedDocument]] = field(default_factory=dict)
    _leakage_serialized: dict[object, str] = field(default_factory=dict)

    def normalized_documents(
        self, items: Sequence[NormalizedDocument | dict[str, Any]]
    ) -> tuple[NormalizedDocument, ...]:
        from autobrain.corpus import normalize_raw_items

        key = tuple(_freeze_item(item) for item in items)
        cached = self._normalized.get(key)
        if cached is not None:
            self.counters.normalized_cache_hits += 1
            return cached
        self.counters.normalized_documents += 1
        value = tuple(normalize_raw_items(list(items)))
        self._normalized[key] = value
        return value

    def serialize(self, value: object, *, leakage: bool = False) -> str:
        cache = self._leakage_serialized if leakage else self._serialized
        key = _freeze_item(value)
        cached = cache.get(key)
        if cached is not None:
            if leakage:
                self.counters.leakage_cache_hits += 1
            else:
                self.counters.serialization_cache_hits += 1
            return cached
        encoded = canonical_json(value)
        cache[key] = encoded
        if leakage:
            self.counters.leakage_serializations += 1
        else:
            self.counters.serializations += 1
        return encoded

    def hash_bytes(self, value: bytes) -> str:
        key = ("bytes", value)
        cached = self._hashes.get(key)
        if cached is not None:
            self.counters.hash_cache_hits += 1
            return cached
        self.counters.hashes += 1
        digest = hashlib.sha256(value).hexdigest()
        self._hashes[key] = digest
        return digest

    def hash_json(self, value: object) -> str:
        key = _freeze_item(value)
        cached = self._hashes.get(key)
        if cached is not None:
            self.counters.hash_cache_hits += 1
            return cached
        self.counters.hashes += 1
        digest = hashlib.sha256(self.serialize(value).encode("utf-8")).hexdigest()
        self._hashes[key] = digest
        return digest

    def answer_tokens(self, value: str, tokenizer: Any) -> frozenset[str]:
        cached = self._answer_tokens.get(value)
        if cached is not None:
            self.counters.answer_token_cache_hits += 1
            return cached
        self.counters.answer_tokenizations += 1
        tokens = frozenset(tokenizer(value))
        self._answer_tokens[value] = tokens
        return tokens

    def source_index(
        self, documents: Sequence[NormalizedDocument]
    ) -> Mapping[str, NormalizedDocument]:
        key = tuple(document.source_id for document in documents)
        cached = self._source_indexes.get(key)
        if cached is not None:
            return cached
        self.counters.source_index_builds += 1
        index = {document.source_id: document for document in documents}
        self._source_indexes[key] = index
        return index


def canonical_json(value: object) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, StrictModel) else value
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _freeze_item(value: object) -> object:
    if isinstance(value, StrictModel):
        return _freeze_item(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        entries: list[tuple[str, object]] = []
        mapping = cast(Mapping[object, object], value)
        for key, item in mapping.items():
            entries.append((str(key), _freeze_item(item)))
        return tuple(sorted(entries))
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        sequence = cast(Sequence[object], value)
        return tuple(_freeze_item(item) for item in sequence)
    return value
