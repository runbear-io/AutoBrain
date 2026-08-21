"""Deterministic whole-document corpus normalization and freezing."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from autobrain.models import CoverageCompleteness, NormalizedDocument, SourceKind


class CorpusBoundaryError(ValueError):
    """Candidate-facing corpus input contains a forbidden artifact."""


class DirtyCorpusError(FileExistsError):
    """A freeze destination contains prior output and cannot be overwritten."""


_SECRET = re.compile(
    r"(?:bearer\s+(?!(?:tokens?|authentication|credentials?)\b)[A-Za-z0-9._~+/=-]+|"
    r"(?:sk|xox[baprs])-[A-Za-z0-9._-]{8,})",
    re.IGNORECASE,
)
_PROTECTED_MARKER = re.compile(r"\b(?:oracle|holdout|reference[_ -]?answer)\b", re.IGNORECASE)
_PROMPT_LIKE = re.compile(
    r"\b(ignore|disregard|override|system prompt|call a (?:write|tool))\b",
    re.IGNORECASE,
)


def normalize_raw_items(
    items: Sequence[NormalizedDocument | dict[str, Any]],
) -> list[NormalizedDocument]:
    """Convert connector records into whole documents and exact-deduplicate content."""
    prepared: list[NormalizedDocument] = []
    for item in items:
        document = item if isinstance(item, NormalizedDocument) else _from_raw(item)
        content_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
        warnings = list(document.warnings)
        if _PROMPT_LIKE.search(document.text):
            warnings.append("prompt-like source instructions preserved as inert data")
        document = document.model_copy(
            update={
                "content_hash": content_hash,
                "warnings": sorted(set(warnings)),
            }
        )
        _reject_boundary(document)
        prepared.append(document)

    by_content: dict[tuple[str, str], NormalizedDocument] = {}
    for document in sorted(prepared, key=lambda value: value.source_id):
        key = (document.content_hash, document.text)
        existing = by_content.get(key)
        if existing is None:
            by_content[key] = document.model_copy(
                update={
                    "source_references": sorted({*document.source_references, document.source_id}),
                    "metadata": {
                        **document.metadata,
                        "source_urls": document.canonical_url,
                    },
                }
            )
            continue
        urls = _split_refs(existing.metadata.get("source_urls", ""))
        urls.append(document.canonical_url)
        by_content[key] = existing.model_copy(
            update={
                "source_references": sorted(
                    {
                        *existing.source_references,
                        *document.source_references,
                        document.source_id,
                    }
                ),
                "metadata": {
                    **existing.metadata,
                    "source_urls": ",".join(sorted(set(urls))),
                },
            }
        )
    return sorted(by_content.values(), key=lambda document: document.source_id)


def freeze_corpus(
    documents: list[NormalizedDocument],
    output_dir: Path,
    *,
    completeness: CoverageCompleteness | str = CoverageCompleteness.UNKNOWN,
    coverage: object | None = None,
) -> FreezeResult:
    """Atomically freeze deterministic JSONL and manifest artifacts."""
    normalized = normalize_raw_items(documents)
    for document in normalized:
        _reject_boundary(document)
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise DirtyCorpusError(f"corpus output is not a safe directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise DirtyCorpusError(f"corpus output is not empty: {output_dir}")

    payload_lines = [
        json.dumps(document.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        for document in normalized
    ]
    documents_jsonl = "\n".join(payload_lines) + ("\n" if payload_lines else "")
    manifest_payload = {
        "schema_version": 1,
        "completeness": str(completeness),
        "document_count": len(normalized),
        "documents_sha256": hashlib.sha256(documents_jsonl.encode("utf-8")).hexdigest(),
        "coverage": _json_value(coverage),
    }
    manifest_hash = _hash_json(manifest_payload)
    manifest = {**manifest_payload, "manifest_hash": manifest_hash}

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        (temporary / "documents.jsonl").write_text(documents_jsonl, encoding="utf-8")
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            if any(output_dir.iterdir()):
                raise DirtyCorpusError(f"corpus output is not empty: {output_dir}")
            output_dir.rmdir()
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return FreezeResult(manifest_hash=manifest_hash, document_count=len(normalized))


class FreezeResult:
    def __init__(self, *, manifest_hash: str, document_count: int) -> None:
        self.manifest_hash = manifest_hash
        self.document_count = document_count


def _from_raw(item: dict[str, Any]) -> NormalizedDocument:
    source_id = str(item["source_id"])
    text = str(item.get("text", ""))
    source_kind = SourceKind(str(item["source_kind"]))
    url = item.get("canonical_url") or item.get("permalink") or item.get("url")
    if not isinstance(url, str):
        raise CorpusBoundaryError(f"missing canonical URL for {source_id}")
    canonical_url = url
    created_at = _parse_datetime(item.get("created_at") or item.get("created_time"))
    updated_at = _parse_datetime(item.get("updated_at") or item.get("last_edited_time"))
    authors = [str(author) for author in item.get("authors", [])]
    if not authors and isinstance(item.get("author"), str):
        authors = [str(item["author"])]
    for key in ("user_name", "user_id"):
        value = item.get(key)
        if isinstance(value, str) and value and value not in authors:
            authors.append(value)
    metadata = {str(k): str(v) for k, v in item.get("metadata", {}).items()}
    for key in (
        "channel_id",
        "channel_name",
        "channel_type",
        "channel_archived",
        "message_ts",
        "thread_ts",
        "parent_source_id",
        "root_source_id",
        "edited_ts",
        "edited",
        "bot",
        "file_id",
        "canvas_id",
        "mimetype",
    ):
        if key in item:
            value = item[key]
            metadata[key] = (
                ""
                if value is None
                else str(value).lower()
                if isinstance(value, bool)
                else str(value)
            )
    relationships = item.get("relationships")
    if relationships is not None:
        metadata["relationships"] = json.dumps(relationships, sort_keys=True, separators=(",", ":"))
    return NormalizedDocument(
        source_id=source_id,
        source_kind=source_kind,
        canonical_url=canonical_url,
        title=str(item.get("title") or source_id),
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        created_at=created_at,
        updated_at=updated_at,
        container=(
            str(item["container"])
            if isinstance(item.get("container"), str)
            else str(item["channel_name"])
            if isinstance(item.get("channel_name"), str)
            else str(item["channel_id"])
            if isinstance(item.get("channel_id"), str)
            else None
        ),
        metadata=metadata,
        authors=authors,
        related_source_ids=[
            *[str(value) for value in item.get("related_source_ids", [])],
            *[
                str(value)
                for key in ("parent_source_id", "root_source_id", "thread_root_source_id")
                if isinstance((value := item.get(key)), str) and value
            ],
        ],
        source_references=[
            str(value) for value in item.get("source_references", []) if isinstance(value, str)
        ],
        crawl_provenance={str(k): str(v) for k, v in item.get("crawl_provenance", {}).items()},
        warnings=[str(value) for value in item.get("warnings", [])],
    )


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _reject_boundary(document: NormalizedDocument) -> None:
    serialized = json.dumps(document.model_dump(mode="json"), sort_keys=True)
    if _SECRET.search(serialized):
        raise CorpusBoundaryError(f"secret-like value in corpus record {document.source_id}")
    if _PROTECTED_MARKER.search(document.text) or _PROTECTED_MARKER.search(document.title):
        raise CorpusBoundaryError(f"evaluator-only marker in corpus record {document.source_id}")


def _split_refs(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def _json_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[no-any-return]
    return value


def _hash_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
