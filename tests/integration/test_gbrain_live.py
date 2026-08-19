from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from autobrain.candidates.gbrain import GBRAIN_COMMIT, GBRAIN_VERSION, THINK_MODEL, GBrainAdapter
from autobrain.models import NormalizedDocument, SourceKind

pytestmark = pytest.mark.live


def test_gbrain_live_invented_micro_corpus(tmp_path: Path) -> None:
    if os.environ.get("AUTOBRAIN_RUN_LIVE_TESTS") != "1":
        pytest.skip("LIVE_DISABLED: set AUTOBRAIN_RUN_LIVE_TESTS=1 for explicit provider access")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("MISSING_PROVIDER: OPENAI_API_KEY is not available or authorized")

    texts = [
        ("Launch", "Project Lantern launches on September 9."),
        ("Owner", "Mina owns the Project Lantern launch checklist."),
        ("Region", "Project Lantern initially launches in Canada."),
    ]
    documents = [
        NormalizedDocument(
            source_id=f"notion:invented-{index}",
            source_kind=SourceKind.NOTION_PAGE,
            canonical_url=f"https://example.invalid/invented-{index}",
            title=title,
            text=text,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
        )
        for index, (title, text) in enumerate(texts)
    ]
    adapter = GBrainAdapter(tmp_path / "tools", tmp_path / "run", timeout_seconds=300)
    results = adapter.run(
        documents,
        ["When and where does Project Lantern launch?", "Who owns its launch checklist?"],
    )

    assert len(results) == 2
    assert all(result.commit == GBRAIN_COMMIT for result in results)
    assert all(result.version == GBRAIN_VERSION for result in results)
    assert all(result.model_used == THINK_MODEL for result in results)
    assert all(result.answer for result in results)
    assert all(result.evidence for result in results)
    assert all(result.footprint_bytes > 0 for result in results)
