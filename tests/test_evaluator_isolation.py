from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from autobrain.models import CandidateId, CandidateObservation, Status
from autobrain.orchestration import (
    CandidateContext,
    CandidateOutcome,
    ConnectorSnapshot,
    RunConfig,
    RunOrchestrator,
)


class _FakeMcpConnector:
    def __init__(self, provider: str, documents: Sequence[dict[str, Any]]) -> None:
        self.provider = provider
        self.documents = tuple(documents)

    def probe(self) -> dict[str, object]:
        return {"allowed": True, "capability_available": True}

    def crawl(self, *, include_dms: bool) -> ConnectorSnapshot:
        del include_dms
        return ConnectorSnapshot(
            provider=self.provider,
            documents=self.documents,
            coverage={"completeness": "SEARCH_DISCOVERED", "fetched": len(self.documents)},
        )


class _CorpusCapturingCandidate:
    candidate_id = "mem0"

    def __init__(self) -> None:
        self.native_documents: tuple[dict[str, Any], ...] = ()

    def run(self, context: CandidateContext) -> CandidateOutcome:
        self.native_documents = tuple(dict(document) for document in context.documents)
        observations = tuple(
            CandidateObservation(
                candidate=CandidateId.MEM0,
                case_id=case_id,
                status=Status.OK,
                answer="The candidate produced an answer.",
                latency_ms=0,
            )
            for case_id in context.case_ids
        )
        return CandidateOutcome(
            candidate=self.candidate_id,
            status=Status.OK,
            cost_usd=0.01,
            observations=observations,
        )

    def cleanup(self) -> None:
        return


def _benchmark_documents() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for index in range(22):
        root_id = f"slack:benchmark-root:{index:02d}"
        documents.append(
            {
                "provider": "slack",
                "source_id": root_id,
                "source_kind": "SLACK_MESSAGE",
                "canonical_url": f"https://fixture.example.test/{root_id}",
                "title": f"Benchmark question {index}",
                "text": f"How does benchmark policy {index} work?",
            }
        )
        documents.append(
            {
                "provider": "slack",
                "source_id": f"slack:benchmark-reply:{index:02d}",
                "source_kind": "SLACK_MESSAGE",
                "canonical_url": f"https://fixture.example.test/reply/{index}",
                "title": f"Answer {index}",
                "text": (
                    f"Benchmark answer {index} says the policy requires review before release."
                ),
                "parent_source_id": root_id,
            }
        )
    documents.extend(
        [
            {
                "provider": "slack",
                "source_id": "slack:unrelated-chat",
                "source_kind": "SLACK_MESSAGE",
                "canonical_url": "https://fixture.example.test/unrelated-chat",
                "title": "Unrelated Slack chat",
                "text": "FYI team",
            },
            {
                "provider": "notion",
                "source_id": "notion:unrelated-page",
                "source_kind": "NOTION_PAGE",
                "canonical_url": "https://fixture.example.test/unrelated-page",
                "title": "Unrelated Notion page",
                "text": "Draft notes",
            },
        ]
    )
    return documents


def _run_isolation_case(
    tmp_path: Path,
    documents: list[dict[str, Any]] | None = None,
) -> tuple[_CorpusCapturingCandidate, Any]:
    candidate = _CorpusCapturingCandidate()
    result = RunOrchestrator(
        config=RunConfig(
            budget_usd=10.0,
            max_questions=30,
            open_report=False,
            output=tmp_path / "run",
            run_id="isolation",
        ),
        connectors=[
            _FakeMcpConnector("slack", documents or _benchmark_documents()),
            _FakeMcpConnector("notion", []),
        ],
        candidates=[candidate],
        provider_available=True,
    ).run()
    return candidate, result


def test_candidate_payload_excludes_benchmark_replies_and_holdouts(tmp_path: Path) -> None:
    candidate, result = _run_isolation_case(tmp_path)

    assert result.status is Status.OK
    visible_ids = {document["source_id"] for document in candidate.native_documents}
    frozen = json.loads((result.run_dir / "corpus-freeze.json").read_text())
    frozen_ids = {document["source_id"] for document in frozen["documents"]}
    assert frozen_ids == visible_ids
    assert "slack:benchmark-reply:00" not in visible_ids
    assert "slack:benchmark-reply:21" not in visible_ids
    assert "slack:benchmark-root:20" not in visible_ids
    assert "slack:benchmark-root:21" not in visible_ids
    assert "slack:unrelated-chat" in visible_ids
    assert "notion:unrelated-page" in visible_ids


def test_normalized_slack_provenance_preserves_reply_isolation(tmp_path: Path) -> None:
    documents = _benchmark_documents()
    for document in documents:
        if document["source_id"].startswith("slack:benchmark-reply:"):
            parent_source_id = document.pop("parent_source_id")
            document["crawl_provenance"] = {"parent_source_id": parent_source_id}

    candidate, result = _run_isolation_case(tmp_path, documents)

    assert result.status is Status.OK
    visible_ids = {document["source_id"] for document in candidate.native_documents}
    assert "slack:benchmark-reply:00" not in visible_ids
    assert "slack:benchmark-root:20" not in visible_ids


def test_evaluator_records_keep_reply_reference_evidence(tmp_path: Path) -> None:
    documents = _benchmark_documents()
    orchestrator = RunOrchestrator(
        config=RunConfig(output=tmp_path / "run", open_report=False, run_id="records"),
        connectors=[
            _FakeMcpConnector("slack", documents),
            _FakeMcpConnector("notion", []),
        ],
        candidates=[],
        provider_available=True,
    )

    cases, holdout_ids = orchestrator._build_benchmark(  # pyright: ignore[reportPrivateUsage]
        documents, 30
    )
    records = orchestrator._evaluator_cases(  # pyright: ignore[reportPrivateUsage]
        documents, cases
    )

    assert len(cases) == 20
    assert holdout_ids == {
        "slack:benchmark-root:20",
        "slack:benchmark-root:21",
    }
    assert len(records) == len(cases)
    assert all(
        "Benchmark answer" in record.reference_text
        and record.reply_texts
        and "Benchmark answer" in " ".join(record.case.expected_claims)
        for record in records
    )
