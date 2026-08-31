from __future__ import annotations

from pathlib import Path

import pytest

from autobrain.connectors.local_file import (
    MAX_LOCAL_FILE_BYTES,
    LocalFileConnector,
    LocalFileError,
    LocalFileFormat,
    LocalFileReadinessState,
    LocalFileStatus,
    extract_local_file_text,
    local_file_document,
    local_file_readiness,
)


def test_markdown_extracts_tokens_without_rendering_or_raw_html(tmp_path: Path) -> None:
    path = tmp_path / "source.md"
    path.write_text(
        "# Heading\n\n**bold** and [link](https://example.test).\n\n<div>hidden html</div>",
        encoding="utf-8",
    )
    format, text, _ = extract_local_file_text(path)
    assert format is LocalFileFormat.MARKDOWN
    assert "Heading" in text and "bold" in text and "link" in text
    assert "<div>hidden html</div>" in text


def test_html_excludes_script_and_style_even_when_malformed(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text(
        "<html><style>.x{display:none}</style><body>Visible<script>secret()</script> tail",
        encoding="utf-8",
    )
    _, text, _ = extract_local_file_text(path)
    assert text == "Visible tail"


def test_txt_is_bounded_and_utf8(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("héllo\nworld", encoding="utf-8")
    _, text, raw = extract_local_file_text(path)
    assert text == "héllo\nworld"
    assert len(raw) == len("héllo\nworld".encode())

    oversized = tmp_path / "large.txt"
    oversized.write_bytes(b"x" * (MAX_LOCAL_FILE_BYTES + 1))
    readiness = local_file_readiness(oversized)
    assert readiness.state is LocalFileReadinessState.INVALID
    assert readiness.status is LocalFileStatus.TOO_LARGE
    with pytest.raises(LocalFileError, match=r"exceeds|large"):
        extract_local_file_text(oversized)


def test_rejects_traversal_and_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("private", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    with pytest.raises(LocalFileError, match="symlink"):
        extract_local_file_text(link)
    with pytest.raises(LocalFileError, match="absolute"):
        extract_local_file_text(Path("relative.txt"))


def test_document_reuses_normalization_and_preserves_provenance(tmp_path: Path) -> None:
    path = tmp_path / "source.md"
    path.write_text("# Provenance\n\nA fact.", encoding="utf-8")
    document = local_file_document(path)
    assert document.source_kind.value == "LOCAL_FILE"
    assert document.content_hash
    assert document.metadata["format"] == "markdown"
    assert document.metadata["path"] == str(path.resolve())
    assert document.crawl_provenance["connector"] == "autobrain.connectors.local_file"
    assert document.crawl_provenance["raw_sha256"]

    snapshot = LocalFileConnector(path).crawl()
    assert snapshot.provider == "local_file"
    assert snapshot.coverage["source"] == "LOCAL_FILE"
    assert snapshot.documents[0]["content_hash"] == document.content_hash


@pytest.mark.parametrize("suffix", [".pdf", ".docx"])
def test_pdf_and_docx_are_explicitly_typed_unsupported(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / ("document" + suffix)
    path.write_bytes(b"not extracted")
    readiness = local_file_readiness(path)
    assert readiness.state is LocalFileReadinessState.UNSUPPORTED
    assert readiness.ready is False
    assert readiness.status is LocalFileStatus.UNSUPPORTED
    assert readiness.format.value == suffix[1:]
    with pytest.raises(LocalFileError, match="unsupported"):
        extract_local_file_text(path)
