from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from autobrain.candidates.gbrain import (
    GBRAIN_COMMIT,
    GBRAIN_VERSION,
    THINK_MODEL,
    CommandResult,
    GBrainAdapter,
    GBrainIsolationError,
    GBrainProcessError,
    document_markdown,
    parse_json_output,
)
from autobrain.candidates.gbrain_config import GBrainExecutionConfig
from autobrain.models import NormalizedDocument, SourceKind
from autobrain.retrieval_ids import document_slug


def document(text: str = "The launch date is September 9.") -> NormalizedDocument:
    return NormalizedDocument(
        source_id="notion:page-1",
        source_kind=SourceKind.NOTION_PAGE,
        canonical_url="https://notion.example/page-1",
        title="Launch plan",
        text=text,
        content_hash="a" * 64,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"workspace": "acme"},
    )


@pytest.fixture(autouse=True)
def fixture_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


class FakeRunner:
    def __init__(self, checkout: Path, *, think: dict[str, object] | None = None) -> None:
        self.calls: list[tuple[tuple[str, ...], Path, dict[str, str], float]] = []
        self.checkout = checkout
        self.think = think or {
            "answer": "September 9 [launch-plan]",
            "citations": [{"page_slug": document_slug("notion:page-1"), "row_num": None}],
            "gaps": [],
            "pagesGathered": 1,
            "modelUsed": THINK_MODEL,
            "usage": {"input_tokens": 10, "output_tokens": 4},
            "cost_usd": 0.0001,
            "warnings": [],
        }

    def __call__(
        self, command: Sequence[str], cwd: Path, env: dict[str, str], timeout: float
    ) -> CommandResult:
        call = (tuple(command), cwd, env, timeout)
        self.calls.append(call)
        args = tuple(command)
        if args[-2:] == ("rev-parse", "HEAD"):
            output = GBRAIN_COMMIT
        elif "think" in args:
            output = json.dumps(self.think)
        elif "search" in args or "query" in args:
            output = json.dumps(
                [{"slug": document_slug("notion:page-1"), "chunk_text": "September 9"}]
            )
        elif "status" in args:
            output = json.dumps({"version": GBRAIN_VERSION, "mode": "local"})
        else:
            output = "{}"
        return CommandResult(tuple(command), 0, output, "", 1)


def fake_checkout(root: Path) -> Path:
    checkout = root / f"gbrain-{GBRAIN_COMMIT}"
    (checkout / ".git").mkdir(parents=True)
    (checkout / "src/core/ai").mkdir(parents=True)
    (checkout / "package.json").write_text(json.dumps({"version": GBRAIN_VERSION}))
    (checkout / "src/core/ai/gateway.ts").write_text("const key = 'OPENAI_BASE_URL';")
    return checkout


def test_document_markdown_preserves_whole_document_provenance() -> None:
    rendered = document_markdown(document())
    assert 'source_id: "notion:page-1"' in rendered
    assert 'canonical_url: "https://notion.example/page-1"' in rendered
    assert 'source_kind: "NOTION_PAGE"' in rendered
    assert rendered.endswith("The launch date is September 9.\n")


def test_json_parser_rejects_corruption_and_accepts_native_prefix() -> None:
    assert parse_json_output('progress\n{"answer":"ok"}') == {"answer": "ok"}
    with pytest.raises(GBrainProcessError, match="corrupt JSON"):
        parse_json_output("{definitely broken")


def test_adapter_uses_exact_pin_frozen_bun_and_native_surfaces(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    checkout = fake_checkout(tools)
    runner = FakeRunner(checkout)
    adapter = GBrainAdapter(
        tools,
        tmp_path / "run",
        runner=runner,
        config=GBrainExecutionConfig.semantic(
            "openai", credential="test-key", chat_credential="test-key"
        ),
    )

    results = adapter.run([document()], ["When is launch?"], base_url="http://127.0.0.1:9999")

    commands = [call[0] for call in runner.calls]
    assert ("git", "checkout", "--detach", GBRAIN_COMMIT) in commands
    assert ("bun", "install", "--frozen-lockfile") in commands
    cli = [command for command in commands if command[:2] == ("bun", "src/cli.ts")]
    assert [command[2] for command in cli] == [
        "init",
        "import",
        "sync",
        "status",
        "search",
        "query",
        "think",
    ]
    assert "text-embedding-3-small" in cli[0]
    assert THINK_MODEL in cli[0]
    assert cli[-1][-3:] == ("--model", THINK_MODEL, "--json")
    assert all(call[2]["GBRAIN_HOME"] == str(adapter.home) for call in runner.calls)
    assert all(call[2]["GBRAIN_SKIP_STARTUP_HOOKS"] == "1" for call in runner.calls)
    assert all(
        call[2].get("OPENAI_BASE_URL") == "http://127.0.0.1:9999"
        for call in runner.calls
        if call[0][:2] == ("bun", "src/cli.ts")
    )
    assert results[0].answer.startswith("September 9")
    assert results[0].evidence
    assert results[0].citations
    assert results[0].model_used == THINK_MODEL
    assert results[0].usage == {"input_tokens": 10, "output_tokens": 4}
    assert results[0].cost_status == "COST_INCOMPLETE"
    assert results[0].base_url_supported is True


def test_missing_usage_and_cost_are_explicitly_incomplete(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    checkout = fake_checkout(tools)
    runner = FakeRunner(
        checkout,
        think={
            "answer": "answer",
            "citations": [],
            "gaps": ["missing owner"],
            "pagesGathered": 2,
            "modelUsed": THINK_MODEL,
            "warnings": ["USAGE_UNAVAILABLE"],
        },
    )
    result = GBrainAdapter(
        tools,
        tmp_path / "run",
        runner=runner,
        config=GBrainExecutionConfig.semantic(
            "openai", credential="test-key", chat_credential="test-key"
        ),
    ).run([document()], ["question"])[0]
    assert result.usage is None
    assert result.cost_usd is None
    assert result.cost_status == "COST_INCOMPLETE"
    assert "USAGE_UNAVAILABLE" in result.warnings


def test_holdout_marker_is_rejected_before_any_cli_ingest(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    checkout = fake_checkout(tools)
    runner = FakeRunner(checkout)
    adapter = GBrainAdapter(tools, tmp_path / "run", runner=runner)
    with pytest.raises(GBrainIsolationError, match="holdout marker"):
        adapter.run([document("secret ORACLE-42")], ["question"], holdout_markers=["ORACLE-42"])
    assert not any("import" in call[0] for call in runner.calls)


def test_unsupported_base_url_and_dirty_repeat_are_honest(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    checkout = fake_checkout(tools)
    (checkout / "src/core/ai/gateway.ts").write_text("no provider override")
    runner = FakeRunner(checkout)
    adapter = GBrainAdapter(tools, tmp_path / "run", runner=runner)
    result = adapter.run([document()], ["question"], base_url="http://meter.test")[0]
    assert result.base_url_supported is False
    assert result.cost_status == "COST_INCOMPLETE"
    assert any("BASE_URL_UNSUPPORTED" in warning for warning in result.warnings)
    with pytest.raises(GBrainIsolationError, match="not empty"):
        adapter.run([document()], ["question"])


def test_holdout_is_removed_from_questions_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = tmp_path / "tools"
    checkout = fake_checkout(tools)
    runner = FakeRunner(checkout)
    monkeypatch.setenv("AUTOBRAIN_ORACLE_TEXT", "ORACLE-42")
    adapter = GBrainAdapter(tools, tmp_path / "run", runner=runner)
    with pytest.raises(GBrainIsolationError, match="candidate question"):
        adapter.run([document()], ["Where is ORACLE-42?"], holdout_markers=["ORACLE-42"])

    adapter = GBrainAdapter(tools, tmp_path / "clean-run", runner=runner)
    adapter.run([document()], ["safe question"], holdout_markers=["ORACLE-42"])
    assert all("AUTOBRAIN_ORACLE_TEXT" not in call[2] for call in runner.calls)
    assert all("ORACLE-42" not in value for call in runner.calls for value in call[2].values())


def test_model_rejection_is_contained_and_no_personal_surfaces_exist(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    fake_checkout(tools)
    seen: list[tuple[str, ...]] = []

    def reject(
        command: Sequence[str], cwd: Path, env: dict[str, str], timeout: float
    ) -> CommandResult:
        seen.append(tuple(command))
        if "think" in command:
            return CommandResult(tuple(command), 1, "", "model not usable", 1)
        output = GBRAIN_COMMIT if command[-2:] == ("rev-parse", "HEAD") else "{}"
        if "search" in command or "query" in command:
            output = "[]"
        if "status" in command:
            output = '{"version":"0.46.19.0"}'
        return CommandResult(tuple(command), 0, output, "", 1)

    adapter = GBrainAdapter(
        tools,
        tmp_path / "run",
        runner=reject,
        config=GBrainExecutionConfig.semantic(
            "openai", credential="test-key", chat_credential="test-key"
        ),
    )
    with pytest.raises(GBrainProcessError, match="model not usable"):
        adapter.run([document()], ["question"])
    commands = " ".join(part for command in seen for part in command)
    forbidden = ("minion", "serve", "schema", "personal-agent")
    assert all(surface not in commands for surface in forbidden)
