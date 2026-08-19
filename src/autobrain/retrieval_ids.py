"""Map candidate retrieval tokens back to frozen gold source IDs."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from autobrain.models import NormalizedDocument


def document_slug(source_id: str) -> str:
    return hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:24]


def provenance_map(documents: Sequence[NormalizedDocument]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for document in documents:
        mapping[document.source_id] = document.source_id
        mapping[document_slug(document.source_id)] = document.source_id
    return mapping


def resolve_retrieved_source_ids(
    raw: Iterable[object],
    provenance: Mapping[str, str],
) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for token in _tokens(raw):
        source_id = provenance.get(token)
        if source_id is None or source_id in seen:
            continue
        seen.add(source_id)
        resolved.append(source_id)
    return resolved


def _tokens(value: object) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, bytes):
        return []
    if isinstance(value, Mapping):
        nested: list[str] = []
        typed = cast(Mapping[str, object], value)
        for item in typed.values():
            nested.extend(_tokens(item))
        return nested
    if isinstance(value, Sequence):
        nested_items: list[str] = []
        typed_sequence = cast(Sequence[object], value)
        for item in typed_sequence:
            nested_items.extend(_tokens(item))
        return nested_items
    return []
