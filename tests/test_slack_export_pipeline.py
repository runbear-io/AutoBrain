from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from autobrain.connectors.slack_export import inspect_slack_export
from autobrain.corpus import freeze_corpus, normalize_raw_items
from autobrain.orchestration import CandidateContext
from autobrain.production import SlackExportSourceConnector


def _slack_export(path: Path) -> Path:
    messages = [
        {
            "type": "message",
            "user": "U1",
            "text": f"Question {index}?",
            "ts": f"17000000{index:02d}.000001",
        }
        for index in range(24)
    ]
    with ZipFile(path, "w") as archive:
        archive.writestr("team.json", '{"id":"T1","name":"Acme","domain":"acme"}')
        archive.writestr("users.json", '[{"id":"U1","name":"ada"}]')
        archive.writestr("channels.json", '[{"id":"C1","name":"general"}]')
        archive.writestr("general/2026-08-19.json", json.dumps(messages))
    return path


def test_slack_export_freezes_the_same_documents_candidates_receive(tmp_path: Path) -> None:
    archive_path = _slack_export(tmp_path / "slack-export.zip")
    summary = inspect_slack_export(archive_path)
    snapshot = SlackExportSourceConnector(
        archive_path,
        expected_sha256=summary.archive_sha256,
    ).crawl()
    normalized = normalize_raw_items([dict(document) for document in snapshot.documents])
    frozen_dir = tmp_path / "frozen"

    freeze = freeze_corpus(normalized, frozen_dir, coverage=snapshot.coverage)
    context = CandidateContext(
        documents=tuple(snapshot.documents),
        questions=("Question 0?",),
        case_ids=("case-0",),
    )

    assert freeze.document_count == 24
    assert context.normalized_documents == tuple(normalized)
    frozen_lines = (frozen_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(frozen_lines) == 24
    first = json.loads(frozen_lines[0])
    assert first["metadata"]["channel_name"] == "general"
    assert first["crawl_provenance"]["connector"] == "slack-export"
    assert "channel_id" not in first
