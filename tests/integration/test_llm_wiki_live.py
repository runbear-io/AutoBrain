from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from autobrain.candidates.llm_wiki import LLMWikiAdapter, LLMWikiConfig
from autobrain.models import BenchmarkCase, NormalizedDocument, SourceKind, Status

pytestmark = pytest.mark.live


def _document(source_id: str, title: str, text: str) -> NormalizedDocument:
    return NormalizedDocument(
        source_id=source_id,
        source_kind=SourceKind.NOTION_PAGE,
        canonical_url=f"https://invented.example/{source_id}",
        title=title,
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


def test_pinned_llm_wiki_live_micro_corpus(tmp_path: Path) -> None:
    if os.environ.get("AUTOBRAIN_RUN_LIVE_TESTS") != "1":
        pytest.skip("LIVE_DISABLED: set AUTOBRAIN_RUN_LIVE_TESTS=1 for explicit provider access")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("MISSING_PROVIDER: OPENAI_API_KEY is unavailable")

    documents = [
        _document(
            "notion:leave-policy",
            "Leave policy",
            "Employees receive 20 vacation days each calendar year.",
        ),
        _document(
            "notion:incident-policy",
            "Incident policy",
            "Critical incidents must be acknowledged within 15 minutes.",
        ),
        _document(
            "notion:expense-policy",
            "Expense policy",
            "Receipts are required for expenses over 25 US dollars.",
        ),
    ]
    cases = [
        BenchmarkCase(
            case_id="case-vacation",
            question="How many vacation days do employees receive?",
            source_ids=["notion:leave-policy"],
            expected_claims=["20 vacation days"],
        ),
        BenchmarkCase(
            case_id="case-incident",
            question="How quickly must a critical incident be acknowledged?",
            source_ids=["notion:incident-policy"],
            expected_claims=["within 15 minutes"],
        ),
    ]
    base_url = os.environ.get("AUTOBRAIN_OPENAI_BASE_URL")
    metering_path_raw = os.environ.get("AUTOBRAIN_METERING_EVENTS_PATH")
    config = LLMWikiConfig(
        workspace=tmp_path / "workspace",
        tool_cache=tmp_path / "node-cache",
        base_url=base_url,
        metering_events_path=Path(metering_path_raw) if metering_path_raw else None,
        timeout_seconds=900,
    )

    result = LLMWikiAdapter(config).run(documents, cases, api_key=api_key)

    assert result.status is Status.OK
    assert len(result.observations) == 2
    assert all(observation.answer for observation in result.observations)
    assert (config.workspace / "artifacts" / "native-export.json").is_file()
    assert (config.workspace / "artifacts" / "native-lint.json").is_file()
    assert result.workspace_bytes > 0
