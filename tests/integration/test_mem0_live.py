from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from autobrain.candidates.mem0 import Mem0Adapter, Mem0AdapterConfig
from autobrain.models import NormalizedDocument, SourceKind

pytestmark = pytest.mark.live


def test_mem0_live_invented_micro_corpus(tmp_path: Path) -> None:
    if os.environ.get("AUTOBRAIN_RUN_LIVE_TESTS") != "1":
        pytest.skip("LIVE_DISABLED: set AUTOBRAIN_RUN_LIVE_TESTS=1 for explicit provider access")
    if not os.environ.get("OPENAI_API_KEY"):
        from autobrain.candidates.mem0 import Mem0MissingProviderError
        from autobrain.models import Status

        with pytest.raises(Mem0MissingProviderError) as error:
            Mem0Adapter(
                Mem0AdapterConfig(
                    run_id="live-missing-provider",
                    run_dir=tmp_path,
                    heldout_source_ids=set(),
                )
            )
        assert error.value.status is Status.MISSING_PROVIDER
        assert not (tmp_path / "qdrant").exists()
        pytest.skip("MISSING_PROVIDER: OPENAI_API_KEY is not configured")
    text = "The invented project Atlas launches on Tuesday in the test workspace."
    doc = NormalizedDocument(
        source_id="notion:atlas",
        source_kind=SourceKind.NOTION_PAGE,
        canonical_url="https://example.test/atlas",
        title="Atlas test note",
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
    adapter = Mem0Adapter(
        Mem0AdapterConfig(run_id="live-micro", run_dir=tmp_path, heldout_source_ids=set())
    )
    try:
        ingested = adapter.ingest([doc])
        native = adapter.search_native("When does Atlas launch?", top_k=1)
        answer = adapter.answer("When does Atlas launch?", native["results"])
        assert ingested.memory_ids
        assert native["results"]
        assert answer.source_ids == ["notion:atlas"]
    finally:
        adapter.cleanup()
