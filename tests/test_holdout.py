from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from autobrain.benchmark import (
    BenchmarkBuildConfig,
    BenchmarkStatus,
    GenerationResponse,
    OracleResponse,
    SlackQuestionThread,
    build_benchmark,
    scan_benchmark_leakage,
)
from autobrain.models import NormalizedDocument, SourceKind


def _doc(source_id: str, text: str, kind: SourceKind) -> NormalizedDocument:
    return NormalizedDocument(
        source_id=source_id,
        source_kind=kind,
        canonical_url=f"https://example.test/{source_id}",
        title=source_id,
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


def _real_thread(index: int) -> SlackQuestionThread:
    return SlackQuestionThread(
        root=_doc(
            f"slack:root:{index}",
            f"How is policy {index} applied?",
            SourceKind.SLACK_MESSAGE,
        ),
        replies=(
            _doc(
                f"slack:reply:{index}",
                f"Policy {index} is applied by team {index}.",
                SourceKind.SLACK_MESSAGE,
            ),
        ),
        root_is_bot=False,
        reply_is_bot=(False,),
        channel=f"channel-{index % 3}",
        topic=f"topic-{index % 3}",
    )


class DeterministicOracleProvider:
    def extract_oracle(
        self,
        *,
        thread: SlackQuestionThread,
        model: str,
        temperature: int,
        seed: int,
        timeout_seconds: float,
    ) -> OracleResponse:
        assert model == "gpt-5-mini"
        assert temperature == 0
        assert seed >= 0
        assert timeout_seconds > 0
        human_replies = [
            reply.text
            for reply, is_bot in zip(thread.replies, thread.reply_is_bot, strict=True)
            if not is_bot
        ]
        return OracleResponse(
            expected_claims=human_replies,
            forbidden_contradictions=["The policy is not applied."],
            weak_human_confidence=0.8,
        )

    def generate(self, **_: object) -> GenerationResponse:
        raise AssertionError("generated fallback is not expected")


def test_selected_roots_and_replies_are_removed_before_candidate_freeze(tmp_path: Path) -> None:
    thread = _real_thread(1)
    threads = (thread, *tuple(_real_thread(index) for index in range(2, 21)))
    result = build_benchmark(
        threads=threads,
        documents=(
            _doc("notion:policy", "The policy is documented here.", SourceKind.NOTION_PAGE),
            thread.root,
            *thread.replies,
        ),
        config=BenchmarkBuildConfig(output_dir=tmp_path / "run"),
        provider=DeterministicOracleProvider(),
    )

    assert result.status is BenchmarkStatus.OK
    candidate_ids = {document.source_id for document in result.candidate_documents}
    assert thread.root.source_id not in candidate_ids
    assert all(reply.source_id not in candidate_ids for reply in thread.replies)
    holdout_ids = {source_id for holdout in result.holdouts for source_id in holdout.reply_ids}
    assert holdout_ids == {item.replies[0].source_id for item in threads}
    assert "slack:reply:1" not in json.dumps(
        [document.model_dump(mode="json") for document in result.candidate_documents]
    )


def test_holdout_artifacts_keep_raw_replies_claims_and_confidence(tmp_path: Path) -> None:
    result = build_benchmark(
        threads=tuple(_real_thread(index) for index in range(20)),
        documents=tuple(),
        config=BenchmarkBuildConfig(output_dir=tmp_path / "run"),
        provider=DeterministicOracleProvider(),
    )

    assert result.status is BenchmarkStatus.OK
    assert all(holdout.raw_replies for holdout in result.holdouts)
    assert all(holdout.expected_claims for holdout in result.holdouts)
    assert all(0 <= holdout.weak_human_confidence <= 1 for holdout in result.holdouts)
    assert result.evaluator_artifacts_dir is not None
    assert (result.evaluator_artifacts_dir / "holdouts.jsonl").is_file()
    assert (result.evaluator_artifacts_dir / "oracles.jsonl").is_file()


def test_leakage_scanner_checks_text_metadata_prompts_argv_env_and_serialized_artifacts() -> None:
    finding = scan_benchmark_leakage(
        texts=["safe text", "slack:reply:1"],
        metadata={"x": "oracle-marker"},
        prompts=["question only"],
        argv=["candidate", "--question", "safe"],
        environment={"SAFE": "yes"},
        serialized_artifacts=["reference answer"],
        forbidden_tokens=("slack:reply:1", "oracle-marker", "reference answer"),
    )
    assert finding.clean is False
    assert {"texts", "metadata", "serialized_artifacts"} <= set(finding.locations)


def test_malformed_thread_and_low_information_replies_are_rejected() -> None:
    with pytest.raises(ValueError):
        SlackQuestionThread(
            root=_doc("slack:root:bad", "ok", SourceKind.SLACK_MESSAGE),
            replies=(),
            root_is_bot=False,
            reply_is_bot=(),
            channel="general",
            topic="social",
        )


def test_reference_reply_leak_in_corpus_or_environment_blocks_candidate_start(
    tmp_path: Path,
) -> None:
    threads = tuple(_real_thread(index) for index in range(20))
    leaked_text = threads[1].replies[0].text
    output_dir = tmp_path / "run"

    result = build_benchmark(
        threads=threads,
        documents=(_doc("notion:copied-answer", leaked_text, SourceKind.NOTION_PAGE),),
        config=BenchmarkBuildConfig(output_dir=output_dir),
        provider=DeterministicOracleProvider(),
        candidate_environment={"LEAK": threads[1].replies[0].source_id},
    )

    assert result.status is BenchmarkStatus.LEAKAGE_DETECTED
    assert result.candidate_start_allowed is False
    assert result.candidate_started is False
    assert {"texts", "environment"} <= set(result.leakage.locations)
    assert not output_dir.exists()


def test_every_selected_reply_id_including_bot_replies_is_removed(tmp_path: Path) -> None:
    first = _real_thread(0)
    bot_reply = _doc(
        "slack:bot-reply:0",
        "Automated reminder with internal routing metadata.",
        SourceKind.SLACK_MESSAGE,
    )
    first = first.model_copy(
        update={
            "replies": (*first.replies, bot_reply),
            "reply_is_bot": (False, True),
        }
    )
    related = _doc(
        "notion:related",
        "Safe policy text.",
        SourceKind.NOTION_PAGE,
    ).model_copy(update={"related_source_ids": [bot_reply.source_id]})
    threads = (first, *tuple(_real_thread(index) for index in range(1, 20)))

    result = build_benchmark(
        threads=threads,
        documents=(bot_reply, related),
        config=BenchmarkBuildConfig(output_dir=tmp_path / "run"),
        provider=DeterministicOracleProvider(),
    )

    assert result.status is BenchmarkStatus.OK
    serialized = json.dumps(
        [document.model_dump(mode="json") for document in result.candidate_documents],
        sort_keys=True,
    )
    assert bot_reply.source_id not in serialized
    first_holdout = next(
        holdout for holdout in result.holdouts if holdout.root_id == first.root.source_id
    )
    assert bot_reply.source_id in first_holdout.reply_ids


def test_leakage_scans_metadata_workspaces_and_outputs_before_writing(tmp_path: Path) -> None:
    threads = tuple(_real_thread(index) for index in range(20))
    leaked_id = threads[0].replies[0].source_id
    workspace = tmp_path / "candidate-workspace"
    workspace.mkdir()
    (workspace / "observation.json").write_text(f'{{"reply_id": "{leaked_id}"}}')
    output_dir = tmp_path / "run"

    result = build_benchmark(
        threads=threads,
        documents=tuple(),
        config=BenchmarkBuildConfig(output_dir=output_dir),
        provider=DeterministicOracleProvider(),
        candidate_metadata={"hidden": leaked_id},
        candidate_workspaces=(workspace,),
        candidate_outputs=(f"raw output includes {leaked_id}",),
    )

    assert result.status is BenchmarkStatus.LEAKAGE_DETECTED
    assert {"metadata", "candidate_workspaces", "outputs"} <= set(result.leakage.locations)
    assert not output_dir.exists()
