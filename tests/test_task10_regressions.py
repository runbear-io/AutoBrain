from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

from typer.testing import CliRunner

from autobrain.cli import app


def _fixture_payload() -> dict[str, object]:
    documents = [
        {
            "provider": "slack" if index % 2 == 0 else "notion",
            "source_id": (
                f"slack:fixture:{index}" if index % 2 == 0 else f"notion:fixture:{index}"
            ),
            "source_kind": "SLACK_MESSAGE" if index % 2 == 0 else "NOTION_PAGE",
            "canonical_url": f"https://fixture.example.test/source/{index}",
            "title": f"Fixture fact {index}",
            "text": f"Project Atlas fact {index} has stable value {index}.",
            "question": f"What is Project Atlas fact {index}?",
            "content_hash": hashlib.sha256(
                f"Project Atlas fact {index} has stable value {index}.".encode()
            ).hexdigest(),
        }
        for index in range(24)
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "fixture_id": "task-10-regression-fixture",
        "documents": documents,
        "candidates": [
            {"id": "llm-wiki", "score": 92.0, "cost_usd": 1.0},
            {"id": "mem0", "score": 88.0, "cost_usd": 1.0},
            {"id": "gbrain", "score": 86.0, "cost_usd": 1.0},
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["fixture_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _write_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(_fixture_payload(), indent=2) + "\n", encoding="utf-8")
    return path


def _run_dir(output: str) -> Path:
    return Path(
        next(line.split(": ", 1)[1] for line in output.splitlines() if line.startswith("run-dir: "))
    )


class _DomParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.nodes: list[tuple[str, dict[str, str], int | None]] = []
        self._stack: list[int] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        parent = self._stack[-1] if self._stack else None
        node = len(self.nodes)
        self.nodes.append((tag, {key: value or "" for key, value in attrs}, parent))
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta"}:
            self._stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self._stack) - 1, -1, -1):
            if self.nodes[self._stack[index]][0] == tag:
                del self._stack[index:]
                return


def test_fixture_mode_requires_explicit_gate_and_absolute_fixture_path(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    output = tmp_path / "runs"
    runner = CliRunner()

    gated = runner.invoke(
        app,
        ["run", "--no-open", "--output", str(output)],
        env={"HOME": str(tmp_path / "home"), "AUTOBRAIN_TEST_FIXTURE_PATH": str(fixture)},
    )
    assert gated.exit_code != 0
    assert "MCP_AUTH_UNAVAILABLE" in gated.output
    assert not output.exists()

    relative = runner.invoke(
        app,
        ["run", "--no-open", "--output", str(output)],
        env={
            "HOME": str(tmp_path / "home"),
            "AUTOBRAIN_ALLOW_TEST_FIXTURE": "1",
            "AUTOBRAIN_TEST_FIXTURE_PATH": "fixture.json",
        },
    )
    assert relative.exit_code != 0
    assert "absolute" in relative.output.lower()
    assert not output.exists()


def test_installed_style_fixture_run_is_local_deterministic_and_source_linked(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    output = tmp_path / "runs"
    env = {
        "HOME": str(tmp_path / "home"),
        "AUTOBRAIN_ALLOW_TEST_FIXTURE": "1",
        "AUTOBRAIN_TEST_FIXTURE_PATH": str(fixture),
    }
    runner = CliRunner()
    first = runner.invoke(app, ["run", "--no-open", "--output", str(output)], env=env)
    second = runner.invoke(app, ["run", "--no-open", "--output", str(output)], env=env)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    first_dir = _run_dir(first.output)
    second_dir = _run_dir(second.output)
    assert first_dir != second_dir
    assert first_dir.name != second_dir.name
    first_manifest = json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second_dir / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["test_mode"]["enabled"] is True
    assert first_manifest["test_mode"]["fixture_id"] == "task-10-regression-fixture"
    assert first_manifest["coverage"]["slack"]["completeness"] == "SEARCH_DISCOVERED"
    assert first_manifest["coverage"]["notion"]["completeness"] == "SEARCH_DISCOVERED"
    assert "https://fixture.example.test" in (first_dir / "report.html").read_text(encoding="utf-8")
    assert "fixture_sha256" in first_manifest["test_mode"]
    assert first_manifest["hashes"]["corpus_sha256"] == second_manifest["hashes"]["corpus_sha256"]
    assert (
        first_manifest["hashes"]["benchmark_sha256"]
        == second_manifest["hashes"]["benchmark_sha256"]
    )
    frozen_corpus = json.loads((first_dir / "corpus-freeze.json").read_text(encoding="utf-8"))
    assert all("provider" not in document for document in frozen_corpus["documents"])
    assert (first_dir / "comparison.json").read_bytes().replace(
        first_dir.name.encode(), b"<RUN_ID>"
    ) == (second_dir / "comparison.json").read_bytes().replace(
        second_dir.name.encode(), b"<RUN_ID>"
    )


def test_fixture_cli_local_hash_persists_smoke_only_no_recommendation(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    output = tmp_path / "local-hash-runs"
    stage_events = tmp_path / "stage-events.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--provider",
            "codex-subscription",
            "--embedding-backend",
            "local-hash",
            "--stage-events",
            str(stage_events),
            "--no-open",
            "--output",
            str(output),
        ],
        env={
            "HOME": str(tmp_path / "home"),
            "AUTOBRAIN_ALLOW_TEST_FIXTURE": "1",
            "AUTOBRAIN_TEST_FIXTURE_PATH": str(fixture),
        },
    )

    assert result.exit_code == 0, result.output
    run_dir = _run_dir(result.output)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    comparison = json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.html").read_text(encoding="utf-8")
    expected_reason = (
        "recommendation requires semantic embeddings; configured backend "
        "local-hash-embedding is smoke-only"
    )

    assert manifest["test_mode"]["enabled"] is True
    assert manifest["provenance"]["embedding"] == {
        "backend": "local-hash-embedding",
        "quality": "smoke_only",
    }
    assert comparison["provenance"]["embedding"] == manifest["provenance"]["embedding"]
    assert comparison["decision"]["verdict"] == "NO_RECOMMENDATION"
    assert expected_reason in comparison["decision"]["ineligible_candidates"]["llm-wiki"]
    assert "Candidate comparison: NO_RECOMMENDATION" in report
    assert expected_reason in report
    assert stage_events.read_text(encoding="utf-8").splitlines()


def test_fixture_cli_explicit_test_semantic_backend_can_recommend(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--provider",
            "codex-subscription",
            "--embedding-backend",
            "test-semantic",
            "--no-open",
            "--output",
            str(tmp_path / "semantic-runs"),
        ],
        env={
            "HOME": str(tmp_path / "home"),
            "AUTOBRAIN_ALLOW_TEST_FIXTURE": "1",
            "AUTOBRAIN_TEST_FIXTURE_PATH": str(fixture),
            "AUTOBRAIN_ENABLE_TEST_SEMANTIC_EMBEDDING": "1",
        },
    )

    assert result.exit_code == 0, result.output
    run_dir = _run_dir(result.output)
    comparison = json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["provenance"]["embedding"] == {
        "backend": "test:semantic-fixture",
        "quality": "semantic",
    }
    assert comparison["decision"]["verdict"] == "llm-wiki"
    assert comparison["decision"]["eligible_candidates"] == ["llm-wiki"]


def test_installed_cli_rejects_absolute_fixture_path_with_lexical_parent_traversal(
    tmp_path: Path,
) -> None:
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    fixture = _write_fixture(tmp_path)
    lexical_path = fixture_dir / ".." / fixture.name
    output = tmp_path / "runs"
    cli = shutil.which("autobrain") or str(Path(sys.executable).with_name("autobrain"))
    assert Path(cli).is_file()
    environment = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "AUTOBRAIN_ALLOW_TEST_FIXTURE": "1",
        "AUTOBRAIN_TEST_FIXTURE_PATH": str(lexical_path),
    }

    result = subprocess.run(
        [cli, "run", "--no-open", "--output", str(output)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "parent traversal" in result.stderr.lower()
    assert not output.exists()


def test_report_warning_markup_is_semantic_and_has_no_li_inside_p(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    result = CliRunner().invoke(
        app,
        ["run", "--no-open", "--output", str(tmp_path / "runs")],
        env={
            "HOME": str(tmp_path / "home"),
            "AUTOBRAIN_ALLOW_TEST_FIXTURE": "1",
            "AUTOBRAIN_TEST_FIXTURE_PATH": str(fixture),
        },
    )
    assert result.exit_code == 0, result.output
    parser = _DomParser()
    parser.feed((_run_dir(result.output) / "report.html").read_text(encoding="utf-8"))
    warning_sections = [
        index
        for index, (tag, attrs, _parent) in enumerate(parser.nodes)
        if tag == "section" and attrs.get("aria-labelledby") == "warnings"
    ]
    assert len(warning_sections) == 1
    warning_section = warning_sections[0]

    def is_descendant(index: int, ancestor: int) -> bool:
        parent = parser.nodes[index][2]
        while parent is not None:
            if parent == ancestor:
                return True
            parent = parser.nodes[parent][2]
        return False

    descendants = [
        (tag, attrs, parent)
        for index, (tag, attrs, parent) in enumerate(parser.nodes)
        if is_descendant(index, warning_section)
    ]
    assert any(tag == "h2" and attrs.get("id") == "warnings" for tag, attrs, _ in descendants)
    assert any(tag == "ul" for tag, _attrs, _parent in descendants)
    assert any(
        tag == "li" and parent is not None and parser.nodes[parent][0] == "ul"
        for tag, _attrs, parent in descendants
    )
    assert not any(
        tag == "li" and parent is not None and parser.nodes[parent][0] == "p"
        for tag, _attrs, parent in parser.nodes
    )


def test_missing_provider_manifest_is_typed_without_report(tmp_path: Path) -> None:
    from autobrain.models import Status
    from autobrain.orchestration import RunConfig, RunOrchestrator

    result = RunOrchestrator(
        config=RunConfig(output=tmp_path / "runs", open_report=False),
        connectors=(),
        candidates=(),
        provider_available=False,
        provider_detail="MISSING_PROVIDER: valid fake auth and capability, provider unavailable",
    ).run()

    assert result.status is Status.MISSING_PROVIDER
    assert result.report_path is None
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    stage = next(item for item in manifest["stages"] if item["name"] == "missing-auth-gate")
    assert stage["status"] == "MISSING_PROVIDER"
    assert "valid fake auth" in stage["detail"]
    assert not (result.run_dir / "report.html").exists()


def test_retained_final_qa_allowlist_and_screenshot_hashes_are_strict() -> None:
    root = Path(".senpi/task-10-final-qa")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "task-10-final-qa.txt",
        "manifest.json",
        "comparison.json",
        "screenshots/report-375.png",
        "screenshots/report-768.png",
        "screenshots/report-1280.png",
    }
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    assert actual == expected
    assert set(manifest["allowlist"]) == expected
    for relative in expected:
        path = root / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if relative == "manifest.json":
            canonical = json.loads(path.read_text(encoding="utf-8"))
            canonical["hashes"]["manifest.json"] = "0" * 64
            digest = hashlib.sha256(
                json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        assert manifest["hashes"][relative] == digest
    for width in (375, 768, 1280):
        png = root / f"screenshots/report-{width}.png"
        png_width, png_height = struct.unpack(">II", png.read_bytes()[16:24])
        assert png_width == width
        assert png_height >= 1000
