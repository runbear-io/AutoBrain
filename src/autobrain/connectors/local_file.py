"""Bounded, read-only local Markdown, text, and HTML source connector."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote

from markdown_it import MarkdownIt

from autobrain.cancellation import RunCancellation
from autobrain.corpus import normalize_raw_items
from autobrain.models import (
    CoverageCompleteness,
    CoverageRecord,
    NormalizedDocument,
    SourceKind,
)
from autobrain.orchestration import ConnectorSnapshot

MAX_LOCAL_FILE_BYTES = 256 * 1024
# Public aliases make the boundary easy to use without duplicating policy.
MAX_FILE_BYTES = MAX_LOCAL_FILE_BYTES
SUPPORTED_LOCAL_FILE_FORMATS = frozenset({".md", ".markdown", ".txt", ".text", ".html", ".htm"})
UNSUPPORTED_LOCAL_FILE_FORMATS = frozenset({".pdf", ".docx"})


class LocalFileError(ValueError):
    """A local source is unsafe, malformed, too large, or unsupported."""


class LocalFileFormat(StrEnum):
    MARKDOWN = "markdown"
    TXT = "txt"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    UNSUPPORTED = "unsupported"


class LocalFileReadinessState(StrEnum):
    READY = "READY"
    UNSUPPORTED = "UNSUPPORTED"
    INVALID = "INVALID"


class LocalFileStatus(StrEnum):
    READY = "READY"
    UNSUPPORTED = "UNSUPPORTED"
    INVALID_PATH = "INVALID_PATH"
    TOO_LARGE = "TOO_LARGE"
    MALFORMED = "MALFORMED"
    READ_ERROR = "READ_ERROR"


class LocalFileReadiness:
    """Typed, credential-free readiness for one local file."""

    __slots__ = ("detail", "format", "path", "ready", "state", "status")

    def __init__(
        self,
        *,
        path: Path,
        format: LocalFileFormat,
        state: LocalFileReadinessState,
        ready: bool,
        status: LocalFileStatus,
        detail: str,
    ) -> None:
        self.path = path
        self.format = format
        self.state = state
        self.ready = ready
        self.status = status
        self.detail = detail

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        del mode
        return {
            "path": str(self.path),
            "format": self.format.value,
            "state": self.state.value,
            "ready": self.ready,
            "status": self.status.value,
            "detail": self.detail,
        }


def _format_for(path: Path) -> LocalFileFormat:
    suffix = path.suffix.lower()
    return {
        ".md": LocalFileFormat.MARKDOWN,
        ".markdown": LocalFileFormat.MARKDOWN,
        ".txt": LocalFileFormat.TXT,
        ".text": LocalFileFormat.TXT,
        ".html": LocalFileFormat.HTML,
        ".htm": LocalFileFormat.HTML,
        ".pdf": LocalFileFormat.PDF,
        ".docx": LocalFileFormat.DOCX,
    }.get(suffix, LocalFileFormat.UNSUPPORTED)


def _validate_path(path: Path) -> Path:
    if not path.is_absolute():
        raise LocalFileError("local file path must be absolute")
    if ".." in path.parts:
        raise LocalFileError("local file path must not contain parent traversal")
    # The input itself must not be a symlink. Parent directories may be
    # platform aliases (for example /tmp -> /private/tmp on macOS).
    if path.is_symlink():
        raise LocalFileError("local file path cannot be a symlink")
    try:
        if not path.is_file() or path.stat().st_size > MAX_LOCAL_FILE_BYTES:
            raise LocalFileError(
                f"local file must be a regular file no larger than {MAX_LOCAL_FILE_BYTES} bytes"
            )
    except OSError as exc:
        raise LocalFileError(f"local file cannot be inspected: {exc}") from exc
    return path


def _bounded_read(path: Path) -> bytes:
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_LOCAL_FILE_BYTES + 1)
    except OSError as exc:
        raise LocalFileError(f"local file cannot be read: {exc}") from exc
    if len(data) > MAX_LOCAL_FILE_BYTES:
        raise LocalFileError(f"local file exceeds {MAX_LOCAL_FILE_BYTES} bytes")
    return data


def _markdown_text(raw: str) -> str:
    # Rendering is deliberately avoided: token content is the source text and
    # raw HTML is disabled so an HTML payload never becomes executable output.
    tokens = MarkdownIt("commonmark", {"html": False}).parse(raw)
    chunks: list[str] = []
    for token in tokens:
        if token.type == "inline":
            chunks.extend(
                child.content
                for child in token.children or ()
                if child.type in {"text", "code_inline", "math", "softbreak", "hardbreak"}
            )
            chunks.append("\n")
        elif token.type in {"code_block", "fence", "html_block", "html_inline"}:
            if token.type in {"code_block", "fence"}:
                chunks.append(token.content)
        elif token.type == "text":
            chunks.append(token.content)
    return "".join(chunks).strip()


class _VisibleHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in {"script", "style"}:
            self._hidden_depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        # A self-closing script/style has no content and therefore needs no
        # depth change.

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.parts.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


def _html_text(raw: str) -> str:
    parser = _VisibleHTMLParser()
    parser.feed(raw)
    parser.close()
    return parser.text()


def extract_local_file_text(path: Path) -> tuple[LocalFileFormat, str, bytes]:
    """Read and extract one supported local file under the size boundary."""
    validated = _validate_path(path)
    format = _format_for(validated)
    if format in {LocalFileFormat.PDF, LocalFileFormat.DOCX, LocalFileFormat.UNSUPPORTED}:
        raise LocalFileError(f"{format.value} local-file extraction is unsupported")
    raw = _bounded_read(validated)
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LocalFileError("local file is not valid UTF-8") from exc
    if format is LocalFileFormat.MARKDOWN:
        text = _markdown_text(decoded)
    elif format is LocalFileFormat.HTML:
        text = _html_text(decoded)
    else:
        text = decoded
    return format, text, raw


def local_file_document(path: Path) -> NormalizedDocument:
    """Extract one file into the shared normalized-document contract."""
    format, text, raw = extract_local_file_text(path)
    resolved = path.resolve()
    source_id = f"local_file:{hashlib.sha256(str(resolved).encode()).hexdigest()[:32]}"
    canonical_url = "https://local.autobrain.invalid/file/" + quote(str(resolved), safe="")
    return normalize_raw_items(
        [
            {
                "source_id": source_id,
                "source_kind": SourceKind.LOCAL_FILE,
                "canonical_url": canonical_url,
                "title": path.name,
                "text": text,
                "metadata": {
                    "format": format.value,
                    "path": str(resolved),
                    "bytes": str(len(raw)),
                },
                "crawl_provenance": {
                    "connector": "autobrain.connectors.local_file",
                    "source_path": str(resolved),
                    "format": format.value,
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                },
            }
        ]
    )[0]


def local_file_readiness(path: Path) -> LocalFileReadiness:
    """Return deterministic readiness without raising for user input errors."""
    format = _format_for(path)
    if format in {LocalFileFormat.PDF, LocalFileFormat.DOCX, LocalFileFormat.UNSUPPORTED}:
        return LocalFileReadiness(
            path=path,
            format=format,
            state=LocalFileReadinessState.UNSUPPORTED,
            ready=False,
            status=LocalFileStatus.UNSUPPORTED,
            detail=(
                f"{format.value} local-file extraction is unsupported; no dependency is installed"
            ),
        )
    try:
        _validate_path(path)
        _bounded_read(path)
    except LocalFileError as exc:
        status = (
            LocalFileStatus.TOO_LARGE
            if "large" in str(exc) or "exceeds" in str(exc)
            else LocalFileStatus.INVALID_PATH
        )
        return LocalFileReadiness(
            path=path,
            format=format,
            state=LocalFileReadinessState.INVALID,
            ready=False,
            status=status,
            detail=str(exc),
        )
    try:
        extract_local_file_text(path)
    except LocalFileError as exc:
        return LocalFileReadiness(
            path=path,
            format=format,
            state=LocalFileReadinessState.INVALID,
            ready=False,
            status=LocalFileStatus.MALFORMED,
            detail=str(exc),
        )
    return LocalFileReadiness(
        path=path,
        format=format,
        state=LocalFileReadinessState.READY,
        ready=True,
        status=LocalFileStatus.READY,
        detail=f"{format.value} local file is ready",
    )


class LocalFileConnector:
    """Source-neutral connector for exactly one local file."""

    provider = "local_file"

    def __init__(self, path: Path) -> None:
        self.path = path

    def probe(self, cancellation: RunCancellation | None = None) -> dict[str, object]:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        readiness = local_file_readiness(self.path)
        return {"allowed": ["read"], **readiness.model_dump(mode="json")}

    def crawl(self, *, cancellation: RunCancellation | None = None) -> ConnectorSnapshot:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        document = local_file_document(self.path)
        return ConnectorSnapshot(
            provider=self.provider,
            documents=(document.model_dump(mode="json"),),
            coverage=CoverageRecord(
                source=SourceKind.LOCAL_FILE,
                completeness=CoverageCompleteness.EXHAUSTIVE,
                discovered=1,
                fetched=1,
                crawl_provenance={
                    "connector": self.__class__.__name__,
                    "path": str(self.path.resolve()),
                },
            ).model_dump(mode="json"),
        )


# Short compatibility aliases for callers that use the generic source naming.
readiness_for_local_file = local_file_readiness
read_local_file = local_file_document
