import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from autobrain.corpus import (
    CorpusBoundaryError,
    DirtyCorpusError,
    freeze_corpus,
    normalize_raw_items,
)
from autobrain.models import NormalizedDocument, SourceKind
from autobrain.orchestration import CandidateContext


def document(
    source_id: str, text: str, *, url: str = "https://example.test/source"
) -> NormalizedDocument:
    return NormalizedDocument(
        source_id=source_id,
        source_kind=SourceKind.NOTION_PAGE,
        canonical_url=url,
        title="Source",
        text=text,
        content_hash="0" * 64,
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
    )


def test_normalizer_exact_deduplicates_and_keeps_all_references() -> None:
    records = normalize_raw_items(
        [
            {
                "source_id": "notion:page:a",
                "source_kind": "NOTION_PAGE",
                "url": "https://notion.so/a",
                "title": "A",
                "text": "same",
            },
            {
                "source_id": "slack:message:b",
                "source_kind": "SLACK_MESSAGE",
                "url": "https://slack.test/b",
                "title": "B",
                "text": "same",
            },
        ]
    )
    assert len(records) == 1
    assert records[0].source_references == [
        "notion:page:a",
        "slack:message:b",
    ]
    assert records[0].content_hash == __import__("hashlib").sha256(b"same").hexdigest()


def test_freeze_is_deterministic_and_writes_jsonl_manifest_hash(tmp_path: Path) -> None:
    first = freeze_corpus(
        [document("notion:page:a", "alpha"), document("notion:page:b", "beta")],
        tmp_path / "first",
        completeness="UNKNOWN",
    )
    second = freeze_corpus(
        [document("notion:page:b", "beta"), document("notion:page:a", "alpha")],
        tmp_path / "second",
        completeness="UNKNOWN",
    )
    assert first.manifest_hash == second.manifest_hash
    assert (tmp_path / "first" / "documents.jsonl").read_bytes() == (
        tmp_path / "second" / "documents.jsonl"
    ).read_bytes()
    manifest = json.loads((tmp_path / "first" / "manifest.json").read_text())
    assert manifest["manifest_hash"] == first.manifest_hash
    assert manifest["completeness"] == "UNKNOWN"


def test_duplicate_order_does_not_change_frozen_hash(tmp_path: Path) -> None:
    duplicate_a = document("notion:page:a", "same", url="https://notion.so/a")
    duplicate_b = document("slack:message:b", "same", url="https://slack.test/b")
    first = freeze_corpus([duplicate_a, duplicate_b], tmp_path / "first")
    second = freeze_corpus([duplicate_b, duplicate_a], tmp_path / "second")
    assert first.manifest_hash == second.manifest_hash


def test_boundary_rejects_secret_or_oracle_artifacts_and_prompt_text_stays() -> None:
    with pytest.raises(CorpusBoundaryError):
        normalize_raw_items(
            [
                {
                    "source_id": "notion:page:a",
                    "source_kind": "NOTION_PAGE",
                    "url": "https://notion.so/a",
                    "title": "A",
                    "text": "bearer super-secret-token",
                }
            ]
        )
    with pytest.raises(CorpusBoundaryError):
        freeze_corpus([document("notion:page:a", "oracle: reference answer")], Path("/tmp/x"))


def test_boundary_allows_conceptual_bearer_token_documentation() -> None:
    normalized = normalize_raw_items(
        [
            {
                "source_id": "notion:page:security",
                "source_kind": "NOTION_PAGE",
                "url": "https://notion.so/security",
                "title": "OAuth security",
                "text": "OAuth clients use bearer tokens; never paste credentials into docs.",
            }
        ]
    )

    assert normalized[0].text.startswith("OAuth clients use bearer tokens")


def test_dirty_output_is_not_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "corpus"
    output.mkdir()
    (output / "keep").write_text("keep")
    with pytest.raises(DirtyCorpusError):
        freeze_corpus([document("notion:page:a", "alpha")], output)


def test_interrupted_freeze_removes_temporary_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Path.write_text

    def interrupt_manifest(path: Path, data: str, **kwargs: object) -> int:
        if path.name == "manifest.json":
            raise KeyboardInterrupt
        return original(path, data, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", interrupt_manifest)
    output = tmp_path / "corpus"
    with pytest.raises(KeyboardInterrupt):
        freeze_corpus([document("notion:page:a", "alpha")], output)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_slack_whole_document_normalization_preserves_adversarial_metadata() -> None:
    record = normalize_raw_items(
        [
            {
                "source_id": "slack-thread:C1:100.001",
                "source_kind": "SLACK_THREAD",
                "canonical_url": "https://acme.slack.com/archives/C1/p100001",
                "title": "#ops at 100.001",
                "text": "root",
                "content_hash": "f" * 64,
                "channel_id": "C1",
                "channel_name": "ops",
                "channel_type": "private_channel",
                "channel_archived": True,
                "message_ts": "100.001",
                "thread_ts": "100.001",
                "parent_source_id": None,
                "user_id": "U1",
                "user_name": "Ada",
                "bot": True,
                "created_at": "2026-08-01T00:01:40+00:00",
                "updated_at": "2026-08-01T00:01:41+00:00",
                "edited": True,
                "metadata": {
                    "mimetype": "text/plain",
                    "file_id": "F1",
                    "canvas_id": "CV1",
                },
                "crawl_provenance": {
                    "connector": "slack-mcp",
                    "scope": "channels:history",
                },
            }
        ]
    )[0]
    assert record.source_id == "slack-thread:C1:100.001"
    assert record.source_kind is SourceKind.SLACK_THREAD
    assert record.canonical_url == "https://acme.slack.com/archives/C1/p100001"
    assert record.container == "ops"
    assert record.created_at == datetime(2026, 8, 1, 0, 1, 40, tzinfo=UTC)
    assert record.updated_at == datetime(2026, 8, 1, 0, 1, 41, tzinfo=UTC)
    assert record.authors == ["Ada", "U1"]
    assert record.metadata == {
        "bot": "true",
        "channel_archived": "true",
        "channel_id": "C1",
        "channel_name": "ops",
        "channel_type": "private_channel",
        "edited": "true",
        "file_id": "F1",
        "canvas_id": "CV1",
        "message_ts": "100.001",
        "mimetype": "text/plain",
        "parent_source_id": "",
        "source_urls": "https://acme.slack.com/archives/C1/p100001",
        "thread_ts": "100.001",
    }
    assert record.crawl_provenance == {
        "connector": "slack-mcp",
        "scope": "channels:history",
    }
    assert record.content_hash == __import__("hashlib").sha256(b"root").hexdigest()


def test_slack_dedup_keeps_every_source_reference_and_provenance() -> None:
    records = normalize_raw_items(
        [
            {
                "source_id": "slack-message:C1:1",
                "source_kind": "SLACK_MESSAGE",
                "url": "https://acme.slack.com/archives/C1/p1",
                "title": "one",
                "text": "same",
                "channel_name": "ops",
                "crawl_provenance": {"page": "1"},
            },
            {
                "source_id": "slack-file:F1",
                "source_kind": "SLACK_FILE",
                "url": "https://acme.slack.com/files/F1",
                "title": "file",
                "text": "same",
                "channel_name": "ops",
                "crawl_provenance": {"page": "2"},
            },
            {
                "source_id": "slack-canvas:CV1",
                "source_kind": "SLACK_CANVAS",
                "url": "https://acme.slack.com/docs/CV1",
                "title": "canvas",
                "text": "same",
                "channel_name": "ops",
                "crawl_provenance": {"page": "3"},
            },
        ]
    )
    assert len(records) == 1
    assert records[0].source_references == [
        "slack-canvas:CV1",
        "slack-file:F1",
        "slack-message:C1:1",
    ]


def test_slack_provenance_survives_jsonl_normalization_and_corpus_freeze(
    tmp_path: Path,
) -> None:
    raw = {
        "source_id": "slack-message:C1:1200.000001",
        "source_kind": "SLACK_MESSAGE",
        "canonical_url": "https://acme.slack.com/archives/C1/p1200000001",
        "title": "#general at 1200.000001",
        "text": "Stable source",
        "content_hash": "0" * 64,
        "channel_id": "C1",
        "channel_name": "general",
        "channel_type": "public_channel",
        "message_ts": "1200.000001",
        "crawl_provenance": {
            "attempt_count": "1",
            "channel_id": "C1",
            "channel_type": "public_channel",
            "connector": "slack-mcp",
            "cursor_lineage": "start>sha256:" + ("a" * 64),
            "interrupted": "false",
            "page": "2",
            "partial": "false",
            "retrieval": "standalone_message",
            "retried": "false",
            "retry_count": "0",
            "surface": "channel_history",
            "tool": "slack-channel-history",
        },
    }
    serialized_raw = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    normalized = normalize_raw_items([json.loads(serialized_raw)])
    freeze_corpus(normalized, tmp_path / "corpus")

    frozen = json.loads((tmp_path / "corpus" / "documents.jsonl").read_text())
    assert frozen["crawl_provenance"] == raw["crawl_provenance"]
    assert normalized[0].crawl_provenance == raw["crawl_provenance"]


def test_candidate_context_normalizes_live_slack_connector_records() -> None:
    raw = {
        "source_id": "slack-message:C1:1200.000001",
        "source_kind": "SLACK_MESSAGE",
        "canonical_url": "https://acme.slack.com/archives/C1/p1200000001",
        "title": "#general at 1200.000001",
        "text": "Stable source",
        "content_hash": "0" * 64,
        "channel_id": "C1",
        "channel_name": "general",
        "channel_type": "public_channel",
        "channel_archived": False,
        "message_ts": "1200.000001",
        "thread_ts": None,
        "parent_source_id": None,
        "user_id": "U1",
        "user_name": "Ada",
        "bot": False,
        "edited": False,
        "deleted": False,
        "untrusted": True,
        "crawl_provenance": {"connector": "slack-mcp"},
        "metadata": {},
    }

    context = CandidateContext(
        documents=(raw,),
        questions=("What is the stable source?",),
        case_ids=("case-stable",),
    )

    assert context.normalized_documents == tuple(normalize_raw_items([raw]))
