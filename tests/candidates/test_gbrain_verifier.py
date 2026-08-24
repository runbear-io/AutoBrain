from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from autobrain.candidates.gbrain import (
    GBRAIN_COMMIT,
    GBRAIN_VERSION,
    THINK_MODEL,
    CommandResult,
    GBrainAdapter,
    GBrainIsolationError,
    GBrainMissingProviderError,
    GBrainProcessError,
    document_markdown,
    run_process,
)
from autobrain.candidates.gbrain_config import GBrainExecutionConfig
from autobrain.models import NormalizedDocument, SourceKind


_LEGACY_CONFIG = GBrainExecutionConfig.semantic(
    "openai", credential="test-key", chat_credential="test-key"
)


@pytest.fixture(autouse=True)
def fixture_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _document(metadata: dict[str, str] | None = None) -> NormalizedDocument:
    return NormalizedDocument(
        source_id="notion:canonical",
        source_kind=SourceKind.NOTION_PAGE,
        canonical_url="https://example.test/canonical",
        title="Canonical",
        text="The canonical answer is blue.",
        content_hash="a" * 64,
        metadata=metadata or {},
    )


class LifecycleRunner:
    def __init__(self, checkout: Path, *, failure: str | None = None) -> None:
        self.checkout = checkout
        self.failure = failure
        self.calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def __call__(
        self, command: Sequence[str], cwd: Path, env: dict[str, str], timeout: float
    ) -> CommandResult:
        del cwd, timeout
        argv = tuple(command)
        self.calls.append((argv, env))
        if argv[-2:] == ("rev-parse", "HEAD"):
            stdout = GBRAIN_COMMIT
        elif self.failure == "provider" and "import" in argv:
            return CommandResult(
                argv,
                1,
                '{"status":"embedding_credentials_missing"}',
                "OPENAI_API_KEY is required for openai embeddings",
                1,
            )
        elif "status" in argv:
            stdout = json.dumps({"version": GBRAIN_VERSION, "mode": "local"})
        elif "search" in argv:
            stdout = json.dumps([{"slug": "canonical", "score": 1.0}])
        elif "query" in argv:
            stdout = json.dumps([{"slug": "canonical", "score": 0.9}])
        elif "think" in argv:
            stdout = json.dumps(
                {
                    "answer": "Blue [canonical]",
                    "citations": [{"page_slug": "canonical", "row_num": None}],
                    "gaps": ["Owner is unknown"],
                    "pagesGathered": 1,
                    "modelUsed": THINK_MODEL,
                    "usage": {"input_tokens": 11, "output_tokens": 3},
                    "cost_usd": 0.001,
                    "warnings": [],
                }
            )
        else:
            stdout = "{}"
        return CommandResult(argv, 0, stdout, "", 1)


def _checkout(tools: Path, *, base_url: bool = True) -> Path:
    checkout = tools / f"gbrain-{GBRAIN_COMMIT}"
    (checkout / ".git").mkdir(parents=True)
    (checkout / "src/core/ai").mkdir(parents=True)
    (checkout / "package.json").write_text(json.dumps({"version": GBRAIN_VERSION}))
    gateway = "OPENAI_BASE_URL" if base_url else "no override"
    (checkout / "src/core/ai/gateway.ts").write_text(gateway)
    return checkout


def test_sync_uses_confined_repo_and_query_is_not_called_gather(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    runner = LifecycleRunner(_checkout(tools))
    adapter = GBrainAdapter(tools, tmp_path / "run", runner=runner, config=_LEGACY_CONFIG)
    result = adapter.run([_document()], ["What color?"])[0]

    sync = next(argv for argv, _env in runner.calls if argv[:3] == ("bun", "src/cli.ts", "sync"))
    assert sync == (
        "bun",
        "src/cli.ts",
        "sync",
        "--repo",
        str(adapter.sources),
        "--no-pull",
        "--json",
    )
    assert "query" in result.native
    assert "gather" not in result.native
    assert result.query_evidence == [{"slug": "canonical", "score": 0.9}]
    assert result.gather_evidence == {
        "citations": [{"page_slug": "canonical", "row_num": None}],
        "pages_gathered": 1,
        "gaps": ["Owner is unknown"],
    }
    assert any("RAW_GATHER_UNAVAILABLE" in warning for warning in result.warnings)


def test_canonical_provenance_cannot_be_overridden_by_metadata() -> None:
    rendered = document_markdown(
        _document(
            {
                "source_id": "slack:attacker",
                "source_kind": "SLACK_MESSAGE",
                "canonical_url": "https://evil.invalid",
                "content_hash": "b" * 64,
            }
        )
    )
    assert rendered.count("source_id:") == 1
    assert 'source_id: "notion:canonical"' in rendered
    assert rendered.count("source_kind:") == 1
    assert 'source_kind: "NOTION_PAGE"' in rendered
    assert 'canonical_url: "https://example.test/canonical"' in rendered
    assert f'content_hash: "{"a" * 64}"' in rendered
    assert "slack:attacker" not in rendered
    assert "evil.invalid" not in rendered


def test_checkout_symlink_is_rejected_before_outside_mutation(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "do-not-touch"
    marker.write_text("safe")
    (tools / f"gbrain-{GBRAIN_COMMIT}").symlink_to(outside, target_is_directory=True)
    calls: list[tuple[str, ...]] = []

    def runner(
        command: Sequence[str], cwd: Path, env: dict[str, str], timeout: float
    ) -> CommandResult:
        del cwd, env, timeout
        calls.append(tuple(command))
        return CommandResult(tuple(command), 0, "", "", 1)

    with pytest.raises(GBrainIsolationError, match="checkout"):
        GBrainAdapter(
            tools, tmp_path / "run", runner=runner, config=_LEGACY_CONFIG
        ).ensure_checkout()
    assert calls == []
    assert marker.read_text() == "safe"


def test_proxy_events_reconcile_without_estimating_and_detect_bad_evidence(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    runner = LifecycleRunner(_checkout(tools))
    events = [
        {
            "event_id": "embed-1",
            "phase": "ingest",
            "model": "text-embedding-3-small",
            "input_tokens": 20,
            "output_tokens": 0,
            "usd": 0.000001,
        },
        {
            "event_id": "think-1",
            "phase": "query",
            "model": "gpt-5-mini",
            "input_tokens": 11,
            "output_tokens": 3,
            "usd": 0.001,
        },
    ]
    result = GBrainAdapter(tools, tmp_path / "run", runner=runner, config=_LEGACY_CONFIG).run(
        [_document()], ["What color?"], base_url="http://meter.test", proxy_events=events
    )[0]
    assert result.cost_status == "COST_COMPLETE"
    assert result.cost_usd is not None
    assert abs(result.cost_usd - 0.001001) < 1e-12
    assert result.proxy_usage == {"input_tokens": 31, "output_tokens": 3}
    assert not any("MISMATCH" in warning for warning in result.warnings)

    duplicate = [*events, dict(events[-1])]
    runner = LifecycleRunner(_checkout(tmp_path / "tools-duplicate"))
    bad = GBrainAdapter(
        tmp_path / "tools-duplicate", tmp_path / "duplicate", runner=runner, config=_LEGACY_CONFIG
    ).run([_document()], ["What color?"], base_url="http://meter.test", proxy_events=duplicate)[0]
    assert bad.cost_status == "COST_INCOMPLETE"
    assert len(bad.proxy_events) == 3
    assert any("METERING_DUPLICATE" in warning for warning in bad.warnings)

    runner = LifecycleRunner(_checkout(tmp_path / "tools-missing"))
    missing = GBrainAdapter(
        tmp_path / "tools-missing", tmp_path / "missing", runner=runner, config=_LEGACY_CONFIG
    ).run([_document()], ["What color?"], base_url="http://meter.test", proxy_events=[events[1]])[0]
    assert missing.cost_status == "COST_INCOMPLETE"
    assert any("METERING_MISSING_PHASE" in warning for warning in missing.warnings)

    mismatch = [events[0], {**events[1], "input_tokens": 999}]
    runner = LifecycleRunner(_checkout(tmp_path / "tools-mismatch"))
    bad = GBrainAdapter(
        tmp_path / "tools-mismatch", tmp_path / "mismatch", runner=runner, config=_LEGACY_CONFIG
    ).run([_document()], ["What color?"], base_url="http://meter.test", proxy_events=mismatch)[0]
    assert bad.cost_status == "COST_INCOMPLETE"
    assert any("METERING_USAGE_MISMATCH" in warning for warning in bad.warnings)


def test_missing_provider_is_typed_and_preserves_native_stderr(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    runner = LifecycleRunner(_checkout(tools), failure="provider")
    with pytest.raises(GBrainMissingProviderError) as caught:
        GBrainAdapter(tools, tmp_path / "run", runner=runner, config=_LEGACY_CONFIG).run(
            [_document()], ["What color?"]
        )
    assert caught.value.status == "MISSING_PROVIDER"
    assert "OPENAI_API_KEY" in caught.value.stderr


def test_external_sigterm_cleans_process_group(tmp_path: Path) -> None:
    ready = tmp_path / "ready.fifo"
    os.mkfifo(ready)
    ready_fd = os.open(ready, os.O_RDONLY | os.O_NONBLOCK)
    child_code = f"import os,time; open({str(ready)!r},'w').write(str(os.getpid())); time.sleep(60)"
    helper = tmp_path / "helper.py"
    helper.write_text(
        "import os,sys\n"
        "from pathlib import Path\n"
        "from autobrain.candidates.gbrain import run_process\n"
        f"run_process((sys.executable, '-c', {child_code!r}), Path({str(tmp_path)!r}), "
        "dict(os.environ), 30)\n"
    )
    parent = subprocess.Popen([sys.executable, str(helper)])
    try:
        readable, _writable, _exceptional = select.select([ready_fd], [], [], 5)
        assert readable == [ready_fd]
        child = int(os.read(ready_fd, 64).decode())
        parent.terminate()
        parent.wait(timeout=5)
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)
    finally:
        os.close(ready_fd)
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)


def test_keyboard_interrupt_cleanup_path_is_not_reported_as_timeout(tmp_path: Path) -> None:
    with pytest.raises(KeyboardInterrupt):
        run_process(
            (sys.executable, "-c", "import os,signal; os.kill(os.getppid(), signal.SIGINT)"),
            tmp_path,
            dict(os.environ),
            5,
        )


def test_corrupt_think_json_and_model_rejection_remain_failures(tmp_path: Path) -> None:
    class BadRunner(LifecycleRunner):
        def __call__(
            self, command: Sequence[str], cwd: Path, env: dict[str, str], timeout: float
        ) -> CommandResult:
            result = super().__call__(command, cwd, env, timeout)
            if "think" in command:
                return CommandResult(tuple(command), 0, "{bad", "", 1)
            return result

    tools = tmp_path / "tools"
    with pytest.raises(GBrainProcessError, match="corrupt JSON"):
        GBrainAdapter(
            tools, tmp_path / "run", runner=BadRunner(_checkout(tools)), config=_LEGACY_CONFIG
        ).run([_document()], ["question"])
