from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

import autobrain.candidates.llm_wiki as llm_wiki_module
from autobrain.candidates.llm_wiki import (
    APPROVED_COMMIT,
    APPROVED_LICENSE_SHA256,
    APPROVED_VERSION,
    LLMWikiAdapter,
    LLMWikiConfig,
    ToolCacheError,
)
from autobrain.models import BenchmarkCase, Holdout, NormalizedDocument, SourceKind, Status

_LICENSE_TEXT = """MIT License

Copyright (c) 2026 atomicmemory

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


@pytest.fixture(autouse=True)
def approve_fake_sdk_digest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dist = tmp_path / "approved-fake-dist"
    dist.mkdir()
    (dist / "index.js").write_text("export const fixture = true;\n")
    monkeypatch.setattr(llm_wiki_module, "APPROVED_DIST_TREE_SHA256", _tree_sha256(dist))


def _document(
    source_id: str = "notion:page-1", *, text: str = "Alpha policy is seven days."
) -> NormalizedDocument:
    return NormalizedDocument(
        source_id=source_id,
        source_kind=SourceKind.NOTION_PAGE,
        canonical_url=f"https://example.test/{source_id}",
        title="Alpha policy",
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        metadata={"workspace": "fixture"},
    )


def _case(
    case_id: str = "case-alpha", question: str = "What is the Alpha policy?"
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        question=question,
        source_ids=["notion:page-1"],
        expected_claims=["seven days"],
    )


def _fake_tool_cache(root: Path) -> Path:
    package = root / "source"
    package.mkdir(parents=True)
    (package / "package.json").write_text(
        json.dumps({"name": "llm-wiki-compiler", "version": APPROVED_VERSION, "license": "MIT"}),
        encoding="utf-8",
    )
    (package / "LICENSE").write_text(_LICENSE_TEXT, encoding="utf-8")
    assert hashlib.sha256((package / "LICENSE").read_bytes()).hexdigest() == APPROVED_LICENSE_SHA256
    (package / "src").mkdir()
    (package / "src" / "index.ts").write_text("export const fixture = true;\n")
    (package / "dist").mkdir()
    (package / "dist" / "index.js").write_text("export const fixture = true;\n", encoding="utf-8")
    (root / "autobrain-pin.json").write_text(
        json.dumps(
            {
                "distribution": "llm-wiki-compiler",
                "version": APPROVED_VERSION,
                "commit": APPROVED_COMMIT,
                "license": "MIT",
                "license_sha256": APPROVED_LICENSE_SHA256,
                "dist_tree_sha256": llm_wiki_module.APPROVED_DIST_TREE_SHA256,
            }
        ),
        encoding="utf-8",
    )
    return root


def _fake_git(path: Path) -> Path:
    script = path / "fake-git"
    script.write_text(
        """#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
source = pathlib.Path(args[1])
command = args[2:]
if command[:1] == ["status"]:
    if (source / "src" / "tampered.ts").exists():
        print("?? src/tampered.ts")
elif command == ["rev-parse", "HEAD"]:
    marker = source / ".fake-head"
    print(marker.read_text() if marker.exists() else "3e17bcfe8b50f24c14c6bcda0cb9224d94fd8206")
elif command == ["remote", "get-url", "origin"]:
    marker = source / ".fake-origin"
    print(marker.read_text() if marker.exists() else "https://github.com/atomicstrata/llm-wiki-compiler.git")
else:
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _fake_node(path: Path) -> Path:
    script = path / "fake-node"
    script.write_text(
        """#!/usr/bin/env python3
import json
import os
import pathlib
import signal
import sys
import time

if "--input-type=module" in sys.argv:
    raise SystemExit(0)
_, driver, package_entry, operation, request_path, response_path = sys.argv
request = json.loads(pathlib.Path(request_path).read_text())
workspace = pathlib.Path(request["root"])
if os.environ.get("FAKE_STALE_CHILD") == "1" and operation == "compile":
    child = os.fork()
    if child == 0:
        if os.environ.get("FAKE_CANCEL") != operation:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True:
            time.sleep(1)
    if os.environ.get("FAKE_CANCEL") == operation:
        def settle_child(_signum, _frame):
            os.waitpid(child, 0)
            raise SystemExit(143)
        signal.signal(signal.SIGTERM, settle_child)
    pathlib.Path(os.environ["FAKE_CHILD_PID_FILE"]).write_text(str(child))
if os.environ.get("FAKE_CANCEL") == operation:
    print("__AUTOBRAIN_CANCEL_READY__", flush=True)
    while True:
        time.sleep(1)
if os.environ.get("FAKE_HANG") == operation:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(1)
if os.environ.get("FAKE_PRINT_SECRET") == operation:
    secret = os.environ["OPENAI_API_KEY"]
    credential_url = f"https://user:{secret}@example.test/v1?api_key={secret}"
    print(f"stdout secret={secret} url={credential_url}")
    print(f"stderr secret={secret} url={credential_url}", file=sys.stderr)
if os.environ.get("FAKE_FAIL") == operation:
    secret = os.environ.get("OPENAI_API_KEY", "")
    print(f"fixture failure secret={secret}", file=sys.stderr)
    raise SystemExit(9)
if operation == "ingest":
    sources = workspace / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    filename = request["document"]["filename"]
    (sources / filename).write_text(request["document"]["text"])
    text = request["document"]["text"]
    result = {
        "filename": filename,
        "charCount": min(len(text), 100000),
        "truncated": len(text) > 100000,
        "source": request["document"]["source"],
        "sourceType": "file",
        "writeStatus": "created",
    }
elif operation == "compile":
    wiki = workspace / "wiki" / "concepts"
    wiki.mkdir(parents=True, exist_ok=True)
    page = (
        "---\\ntitle: Alpha\\nsources: [doc-notion-page-1.md]\\n---\\n"
        "Seven days ^[doc-notion-page-1.md]"
    )
    (wiki / "alpha.md").write_text(page)
    result = {
        "compiled": 1,
        "skipped": 0,
        "deleted": 0,
        "concepts": ["alpha"],
        "pages": ["alpha"],
        "errors": [],
    }
elif operation == "query":
    secret = os.environ.get("OPENAI_API_KEY", "")
    answer = "   \\n" if os.environ.get("FAKE_EMPTY_ANSWER") == "1" else (
        f"Seven days ^[doc-notion-page-1.md] secret={secret}"
        if os.environ.get("FAKE_RESPONSE_SECRET") == "1"
        else "Seven days ^[doc-notion-page-1.md]"
    )
    result = {
        "answer": answer,
        "selectedPages": ["alpha"],
        "pageIds": ["concepts/alpha"],
        "refs": [{
            "pageId": "concepts/alpha",
            "slug": "alpha",
            "title": "Alpha",
            "kind": "page",
        }],
        "reasoning": "matched",
        "debug": {
            "usedChunks": True,
            "reranked": False,
            "pages": [],
            "chunks": [],
        },
    }
elif operation == "export":
    result = {
        "schemaVersion": 1,
        "exportedAt": "2025-01-01T00:00:00Z",
        "pageCount": 1,
        "pages": [{
            "slug": "alpha",
            "pageDirectory": "concepts",
            "title": "Alpha",
            "sources": ["doc-notion-page-1.md"],
            "body": "Seven days ^[doc-notion-page-1.md]",
        }],
    }
elif operation == "lint":
    result = {
        "results": [{
            "file": "wiki/concepts/alpha.md",
            "severity": "warning",
            "message": "fixture warning",
        }],
        "errors": 0,
        "warnings": 1,
        "info": 0,
    }
else:
    raise SystemExit(3)
pathlib.Path(response_path).write_text(json.dumps(result))
print(json.dumps({"operation": operation, "ok": True}))
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _config(tmp_path: Path, **kwargs: object) -> LLMWikiConfig:
    values: dict[str, Any] = {
        "workspace": tmp_path / "workspace",
        "tool_cache": _fake_tool_cache(tmp_path / "tool-cache"),
        "node_executable": _fake_node(tmp_path),
        "git_executable": str(_fake_git(tmp_path)),
        "timeout_seconds": 2.0,
        "cleanup_grace_seconds": 0.1,
        "additional_env": {
            key: value for key, value in os.environ.items() if key.startswith("FAKE_")
        },
    }
    values.update(kwargs)
    return LLMWikiConfig(**values)


def test_happy_path_preserves_native_artifacts_provenance_and_environment(tmp_path: Path) -> None:
    metering = tmp_path / "metering.jsonl"
    metering.write_text(
        json.dumps(
            {
                "candidate": "llm-wiki",
                "phase": "query",
                "input_tokens": 10,
                "output_tokens": 4,
                "usd": 0.002,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = _config(tmp_path, base_url="http://127.0.0.1:9911/v1", metering_events_path=metering)
    result = LLMWikiAdapter(config).run([_document()], [_case()], api_key="fixture-key")

    assert result.status is Status.OK
    assert result.pin.commit == APPROVED_COMMIT
    assert result.pin.version == APPROVED_VERSION
    assert [command.operation for command in result.commands] == [
        "ingest",
        "compile",
        "query",
        "export",
        "lint",
    ]
    assert all(
        "view" not in command.argv and "mcp" not in command.argv for command in result.commands
    )
    assert result.environment["LLMWIKI_PROVIDER"] == "openai"
    assert result.environment["LLMWIKI_MODEL"] == "gpt-5-mini"
    assert result.environment["LLMWIKI_EMBEDDING_MODEL"] == "text-embedding-3-small"
    assert result.environment["OPENAI_BASE_URL"] == "http://127.0.0.1:9911/v1"
    assert result.environment["OPENAI_EMBEDDINGS_BASE_URL"] == "http://127.0.0.1:9911/v1"
    assert "OPENAI_API_KEY" not in result.environment
    assert result.observations[0].answer == "Seven days ^[doc-notion-page-1.md]"
    assert result.observations[0].source_ids == ("notion:page-1",)
    assert result.observations[0].citations == ("doc-notion-page-1.md",)
    assert result.measured_cost_usd == 0.002
    assert result.workspace_bytes > 0
    assert "native-export.json" in result.artifacts
    source_map = json.loads((config.workspace / "artifacts" / "source-map.json").read_text())
    assert source_map[0]["source_id"] == "notion:page-1"
    assert source_map[0]["canonical_url"] == "https://example.test/notion:page-1"
    assert (config.workspace / "wiki" / "concepts" / "alpha.md").exists()


def test_missing_provider_is_typed_skip_and_never_starts_node(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = LLMWikiAdapter(config).run([_document()], [_case()], api_key=None)
    assert result.status is Status.MISSING_PROVIDER
    assert result.skipped is True
    assert result.commands == ()
    assert not config.workspace.exists()


def test_native_over_limit_is_reported_without_adapter_chunk_tuning(tmp_path: Path) -> None:
    result = LLMWikiAdapter(_config(tmp_path)).run(
        [_document(text="x" * 100_001)], [_case()], api_key="fixture"
    )
    assert result.status is Status.OK
    assert any(
        w.code == "SOURCE_TRUNCATED" and w.source_id == "notion:page-1" for w in result.warnings
    )
    assert not any("chunk" in " ".join(command.argv).lower() for command in result.commands)


def test_duplicate_ids_and_holdout_leakage_fail_before_workspace_write(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="duplicate source_id"):
        LLMWikiAdapter(config).run([_document(), _document()], [_case()], api_key="fixture")
    assert not config.workspace.exists()

    holdout = Holdout(
        case_id="case-alpha",
        source_ids=["slack:thread-secret"],
        reference_text="ORACLE-SECRET-ANSWER",
        reply_ids=["slack:reply-secret"],
    )
    leaked = _document(text="contains ORACLE-SECRET-ANSWER")
    with pytest.raises(ValueError, match="holdout/oracle leakage"):
        LLMWikiAdapter(config).run([leaked], [_case()], holdouts=[holdout], api_key="fixture")
    assert not config.workspace.exists()


def test_expected_claims_and_oracle_paths_never_reach_candidate(tmp_path: Path) -> None:
    oracle = tmp_path / "evaluator" / "oracle.json"
    oracle.parent.mkdir()
    oracle.write_text("ORACLE-MARKER", encoding="utf-8")
    config = _config(tmp_path)
    result = LLMWikiAdapter(config).run(
        [_document()], [_case()], oracle_paths=[oracle], api_key="fixture"
    )
    assert result.status is Status.OK
    all_bytes = b"".join(
        path.read_bytes() for path in config.workspace.rglob("*") if path.is_file()
    )
    assert b"ORACLE-MARKER" not in all_bytes
    query_request = json.loads(
        (config.workspace / "process" / "003-query-request.json").read_text()
    )
    assert set(query_request) == {"root", "question"}
    assert query_request["question"] == "What is the Alpha policy?"


def test_dirty_or_mismatched_tool_cache_is_refused(tmp_path: Path) -> None:
    dirty = tmp_path / "dirty"
    dirty.mkdir()
    (dirty / "unrelated").write_text("do not delete", encoding="utf-8")
    config = LLMWikiConfig(
        workspace=tmp_path / "workspace",
        tool_cache=dirty,
        node_executable=_fake_node(tmp_path),
    )
    with pytest.raises(ToolCacheError, match="dirty"):
        LLMWikiAdapter(config).run([_document()], [_case()], api_key="fixture")
    assert (dirty / "unrelated").read_text() == "do not delete"

    wrong = _fake_tool_cache(tmp_path / "wrong")
    marker = json.loads((wrong / "autobrain-pin.json").read_text())
    marker["commit"] = "0" * 40
    (wrong / "autobrain-pin.json").write_text(json.dumps(marker), encoding="utf-8")
    wrong_config = LLMWikiConfig(
        workspace=tmp_path / "workspace-2", tool_cache=wrong, node_executable=_fake_node(tmp_path)
    )
    with pytest.raises(ToolCacheError, match="marker mismatch"):
        LLMWikiAdapter(wrong_config).run([_document()], [_case()], api_key="fixture")


def test_timeout_kills_entire_process_group_and_preserves_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_pid_file = tmp_path / "child.pid"
    monkeypatch.setenv("FAKE_HANG", "compile")
    monkeypatch.setenv("FAKE_STALE_CHILD", "1")
    monkeypatch.setenv("FAKE_CHILD_PID_FILE", str(child_pid_file))
    config = _config(tmp_path, timeout_seconds=1.0)
    result = LLMWikiAdapter(config).run([_document()], [_case()], api_key="fixture")
    assert result.status is Status.FAILED
    compile_command = next(command for command in result.commands if command.operation == "compile")
    assert compile_command.timed_out is True
    assert compile_command.terminated is True
    assert (config.workspace / compile_command.stdout_path).exists()
    assert (config.workspace / compile_command.stderr_path).exists()
    pid = int(child_pid_file.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
        pytest.fail("stale child process survived bounded cleanup")


def test_cancellation_kills_process_group_and_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_pid_file = tmp_path / "cancel-child.pid"
    monkeypatch.setenv("FAKE_CANCEL", "compile")
    monkeypatch.setenv("FAKE_STALE_CHILD", "1")
    monkeypatch.setenv("FAKE_CHILD_PID_FILE", str(child_pid_file))
    config = _config(tmp_path)
    original_communicate = cast(
        Callable[[subprocess.Popen[str], str | None, float | None], tuple[str, str]],
        subprocess.Popen.communicate,
    )
    interrupted = False

    def interrupt_compile(
        process: subprocess.Popen[str],
        input: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        nonlocal interrupted
        argv = process.args if isinstance(process.args, list) else []
        if not interrupted and "compile" in argv:
            assert process.stdout is not None
            assert process.stdout.readline().strip() == "__AUTOBRAIN_CANCEL_READY__"
            interrupted = True
            raise KeyboardInterrupt
        return original_communicate(process, input, timeout)

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupt_compile)
    secret = "sk-cancelled-provider-secret"
    with pytest.raises(KeyboardInterrupt):
        LLMWikiAdapter(config).run([_document()], [_case()], api_key=secret)
    persisted = "".join(
        path.read_text(errors="replace") for path in config.workspace.rglob("*") if path.is_file()
    )
    assert secret not in persisted
    pid = int(child_pid_file.read_text())
    state = subprocess.run(
        ["ps", "-o", "state=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert not state or state.startswith("Z"), f"active orphan survived cancellation: {state}"


def test_metering_unavailable_or_malformed_never_fabricates_zero_cost(tmp_path: Path) -> None:
    unavailable = LLMWikiAdapter(_config(tmp_path / "a", base_url="http://127.0.0.1:9/v1")).run(
        [_document()], [_case()], api_key="fixture"
    )
    assert unavailable.measured_cost_usd is None
    assert any(w.code == "METERING_UNAVAILABLE" for w in unavailable.warnings)

    malformed_path = tmp_path / "bad-metering.jsonl"
    malformed_path.write_text('{"usd":"free"}\nnot-json\n', encoding="utf-8")
    malformed = LLMWikiAdapter(_config(tmp_path / "b", metering_events_path=malformed_path)).run(
        [_document()], [_case()], api_key="fixture"
    )
    assert malformed.measured_cost_usd is None
    assert any(w.code == "METERING_MALFORMED" for w in malformed.warnings)


def test_malformed_native_artifact_fails_and_repeated_workspace_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    first = LLMWikiAdapter(config).run([_document()], [_case()], api_key="fixture")
    assert first.status is Status.OK
    with pytest.raises(FileExistsError, match="sealed"):
        LLMWikiAdapter(config).run([_document()], [_case()], api_key="fixture")

    monkeypatch.setenv("FAKE_FAIL", "export")
    malformed_config = _config(tmp_path / "malformed")
    failed = LLMWikiAdapter(malformed_config).run([_document()], [_case()], api_key="fixture")
    assert failed.status is Status.FAILED
    assert failed.observations
    assert failed.measured_cost_usd is None


def test_subprocess_runner_does_not_leak_api_key_in_argv_or_result(tmp_path: Path) -> None:
    config = _config(tmp_path)
    secret = "sk-fixture-super-secret"
    result = LLMWikiAdapter(config).run([_document()], [_case()], api_key=secret)
    serialized = json.dumps(result.to_dict())
    assert secret not in serialized
    assert all(secret not in part for command in result.commands for part in command.argv)


def test_cached_reuse_rejects_tampered_head_origin_license_source_and_dist(tmp_path: Path) -> None:
    attacks: dict[str, Callable[[Path], object]] = {
        "head": lambda cache: (cache / "source" / ".fake-head").write_text("0" * 40),
        "origin": lambda cache: (cache / "source" / ".fake-origin").write_text(
            "https://github.com/attacker/fork"
        ),
        "package-license": lambda cache: _rewrite_json(
            cache / "source" / "package.json", {"license": "GPL-3.0"}
        ),
        "license-file": lambda cache: (cache / "source" / "LICENSE").write_text("forged"),
        "source": lambda cache: (cache / "source" / "src" / "tampered.ts").write_text("evil"),
        "dist": lambda cache: (cache / "source" / "dist" / "index.js").write_text("evil"),
    }
    for name, attack in attacks.items():
        root = tmp_path / name
        config = _config(root)
        attack(config.tool_cache)
        with pytest.raises(ToolCacheError):
            LLMWikiAdapter(config).prepare_tool_cache()
        assert not config.workspace.exists()


def test_registry_only_cache_is_rejected_even_with_forged_marker(tmp_path: Path) -> None:
    cache = _fake_tool_cache(tmp_path / "registry-only")
    config = LLMWikiConfig(
        workspace=tmp_path / "workspace",
        tool_cache=cache,
        node_executable=_fake_node(tmp_path),
        git_executable="git",
    )
    with pytest.raises(ToolCacheError, match=r"not a git|git command|checkout"):
        LLMWikiAdapter(config).prepare_tool_cache()


def test_source_install_uses_supported_lockfile_optional_npm_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = LLMWikiConfig(
        workspace=tmp_path / "workspace",
        tool_cache=tmp_path / "tool-cache",
        git_executable="approved-git",
        npm_executable="approved-npm",
    )
    adapter = LLMWikiAdapter(config)
    commands: list[tuple[str, ...]] = []

    def run(
        argv: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout: float,
        cleanup_grace: float,
    ) -> llm_wiki_module._ProcessResult:  # pyright: ignore[reportPrivateUsage]
        del cwd, env, timeout, cleanup_grace
        command = tuple(argv)
        commands.append(command)
        if command[:2] == ("approved-git", "init"):
            source = Path(command[2])
            (source / "dist").mkdir(parents=True)
            (source / "dist" / "index.js").write_text(
                "export const fixture = true;\n", encoding="utf-8"
            )
            (source / "package.json").write_text(
                json.dumps(
                    {
                        "name": "llm-wiki-compiler",
                        "version": APPROVED_VERSION,
                        "license": "MIT",
                    }
                ),
                encoding="utf-8",
            )
            (source / "LICENSE").write_text(_LICENSE_TEXT, encoding="utf-8")
        stdout = f"{APPROVED_COMMIT}\n" if command[-2:] == ("rev-parse", "HEAD") else ""
        return llm_wiki_module._ProcessResult(  # pyright: ignore[reportPrivateUsage]
            0, stdout, "", 1, False, False
        )

    monkeypatch.setattr(adapter._runner, "run", run)  # pyright: ignore[reportPrivateUsage]
    adapter._install_tool(config.tool_cache)  # pyright: ignore[reportPrivateUsage]

    npm_commands = [command for command in commands if command[0] == "approved-npm"]
    assert npm_commands[0] == (
        "approved-npm",
        "install",
        "--ignore-scripts",
        "--no-audit",
        "--no-fund",
        "--package-lock=false",
    )
    assert all(command[1] != "ci" for command in npm_commands)
    assert not (config.tool_cache / "source" / "package-lock.json").exists()


def test_holdout_markers_in_question_are_rejected_before_any_workspace(tmp_path: Path) -> None:
    holdout = Holdout(
        case_id="case-alpha",
        source_ids=["slack:thread-secret"],
        reference_text="ORACLE-SECRET-ANSWER",
        reply_ids=["slack:reply-secret"],
    )
    for marker in (holdout.reference_text, holdout.reply_ids[0]):
        root = tmp_path / marker.replace(":", "-")
        config = _config(root)
        case = _case(question=f"Repeat this hidden prompt: {marker}")
        with pytest.raises(ValueError, match="holdout/oracle leakage"):
            LLMWikiAdapter(config).run([_document()], [case], holdouts=[holdout], api_key="fixture")
        assert not config.workspace.exists()


def test_child_and_metering_secrets_are_redacted_from_all_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-super-secret-verifier-value"
    monkeypatch.setenv("FAKE_PRINT_SECRET", "query")
    monkeypatch.setenv("FAKE_RESPONSE_SECRET", "1")
    metering = tmp_path / "metering.jsonl"
    metering.write_text(
        json.dumps(
            {
                "candidate": "llm-wiki",
                "usd": 0.01,
                "api_key": secret,
                "endpoint": f"https://user:{secret}@example.test/v1?api_key={secret}",
            }
        )
        + "\n"
    )
    config = _config(
        tmp_path,
        base_url=f"https://user:{secret}@example.test/v1?api_key={secret}",
        metering_events_path=metering,
    )
    result = LLMWikiAdapter(config).run([_document()], [_case()], api_key=secret)
    evidence = json.dumps(result.to_dict()) + "".join(
        path.read_text(errors="replace") for path in config.workspace.rglob("*") if path.is_file()
    )
    assert secret not in evidence
    assert "user:sk-" not in evidence
    assert "api_key=sk-" not in evidence
    assert "[REDACTED]" in evidence


def test_failure_exception_and_stderr_are_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-failing-child-secret"
    monkeypatch.setenv("FAKE_FAIL", "compile")
    config = _config(tmp_path)
    result = LLMWikiAdapter(config).run([_document()], [_case()], api_key=secret)
    assert result.status is Status.FAILED
    evidence = json.dumps(result.to_dict()) + "".join(
        path.read_text(errors="replace") for path in config.workspace.rglob("*") if path.is_file()
    )
    assert secret not in evidence
    assert "[REDACTED]" in evidence


def test_empty_native_answer_cannot_report_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_EMPTY_ANSWER", "1")
    result = LLMWikiAdapter(_config(tmp_path)).run([_document()], [_case()], api_key="fixture")
    assert result.status is Status.FAILED
    assert any(warning.code == "EMPTY_ANSWER" for warning in result.warnings)


def test_canonical_workspace_tool_cache_separation_and_symlink_ancestors(tmp_path: Path) -> None:
    cache = _fake_tool_cache(tmp_path / "cache")
    with pytest.raises(ValueError, match="isolated"):
        LLMWikiConfig(workspace=cache / "workspace", tool_cache=cache)
    with pytest.raises(ValueError, match="isolated"):
        LLMWikiConfig(workspace=tmp_path / "outer", tool_cache=tmp_path / "outer" / "cache")

    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match=r"symlink|isolated"):
        LLMWikiConfig(workspace=alias / "workspace", tool_cache=real / "workspace" / "cache")


def test_external_metering_is_copied_to_redacted_run_local_ledger_without_raw_path(
    tmp_path: Path,
) -> None:
    secret = "sk-metering-only-secret"
    raw_ledger = tmp_path / "immutable" / "provider-ledger.jsonl"
    raw_ledger.parent.mkdir()
    raw_ledger.write_text(
        json.dumps(
            {
                "candidate": "llm-wiki",
                "usd": 0.004,
                "api_key": secret,
                "endpoint": f"https://user:{secret}@billing.example/v1?token={secret}",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = _config(tmp_path, metering_events_path=raw_ledger)
    result = LLMWikiAdapter(config).run([_document()], [_case()], api_key=secret)

    assert result.measured_cost_usd == 0.004
    ledger = config.workspace / "artifacts" / "metering-events.jsonl"
    provenance = config.workspace / "artifacts" / "metering-provenance.json"
    assert ledger.is_file()
    assert provenance.is_file()
    persisted = "".join(
        path.read_text(errors="replace") for path in config.workspace.rglob("*") if path.is_file()
    )
    assert secret not in persisted
    assert str(raw_ledger) not in persisted
    assert str(raw_ledger) not in json.dumps(result.to_dict())
    assert (
        json.loads(provenance.read_text())["source_sha256"]
        == hashlib.sha256(raw_ledger.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("surface", ["document", "question"])
@pytest.mark.parametrize(
    "oracle_text",
    [
        "oracle.json",
        "./evaluator/../evaluator/oracle.json",
        "file:///tmp/evaluator/oracle.json",
        "See /tmp/evaluator/oracle.json for the answer",
        "ORACLE-SECRET-MARKER",
    ],
)
def test_oracle_file_names_paths_and_markers_are_rejected_before_workspace(
    tmp_path: Path, oracle_text: str, surface: str
) -> None:
    oracle = tmp_path / "evaluator" / "oracle.json"
    oracle.parent.mkdir()
    oracle.write_text("ORACLE-SECRET-MARKER", encoding="utf-8")
    config = _config(tmp_path / surface / hashlib.sha256(oracle_text.encode()).hexdigest())
    document_text = f"candidate corpus mentions {oracle_text}" if surface == "document" else "safe"
    question = f"benchmark mentions {oracle_text}" if surface == "question" else "safe?"
    with pytest.raises(ValueError, match="holdout/oracle leakage"):
        LLMWikiAdapter(config).run(
            [_document(text=document_text)],
            [_case(question=question)],
            oracle_paths=[oracle],
            api_key="fixture",
        )
    assert not config.workspace.exists()


def test_holdout_case_id_is_forbidden_in_documents_and_questions_before_workspace(
    tmp_path: Path,
) -> None:
    holdout = Holdout(
        case_id="case-hidden-alpha",
        source_ids=["slack:thread-secret"],
        reference_text="hidden answer",
    )
    for document_text, question in [
        ("mentions case-hidden-alpha", "What is Alpha?"),
        ("Alpha policy", "mentions case-hidden-alpha"),
    ]:
        config = _config(tmp_path / hashlib.sha256(document_text.encode()).hexdigest())
        with pytest.raises(ValueError, match="holdout/oracle leakage"):
            LLMWikiAdapter(config).run(
                [_document(text=document_text)],
                [_case(question=question)],
                holdouts=[holdout],
                api_key="fixture",
            )
        assert not config.workspace.exists()


def test_completed_workspace_is_sealed_and_tampering_is_detected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapter = LLMWikiAdapter(config)
    result = adapter.run([_document()], [_case()], api_key="fixture")
    assert result.status is Status.OK
    seal = config.workspace / "workspace-seal.json"
    assert seal.is_file()
    assert adapter.verify_workspace() == result.workspace_seal_sha256

    page = config.workspace / "wiki" / "concepts" / "alpha.md"
    page.write_text(page.read_text() + "\ntampered")
    with pytest.raises(Exception, match=r"tamper|integrity|seal"):
        adapter.verify_workspace()
    with pytest.raises(Exception, match=r"tamper|integrity|seal"):
        LLMWikiAdapter(config).run([_document()], [_case()], api_key="fixture")


def _rewrite_json(path: Path, updates: dict[str, object]) -> None:
    value = json.loads(path.read_text())
    value.update(updates)
    path.write_text(json.dumps(value))
