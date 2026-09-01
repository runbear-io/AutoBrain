import json
from pathlib import Path

from harness import E2EHarness


def test_installed_binary_cli_surface_and_artifacts(e2e: E2EHarness) -> None:
    generated = e2e.run(
        "fixture",
        "generate",
        "--seed",
        "5",
        "--output",
        str(e2e.fixture_path),
    )
    assert generated.returncode == 0, generated.output
    assert generated.lines["fixture-id"] == "generated-fixture-5"

    help_result = e2e.run("--help")
    assert help_result.returncode == 0
    assert "runs" in help_result.stdout

    doctor = e2e.run("doctor", "--offline", "--json")
    assert doctor.returncode == 0
    assert doctor.json["status"]

    run = e2e.run("run", "--no-open", "--max-questions", "20", "--provider", "codex-subscription")
    assert run.returncode == 0, run.output
    assert run.lines["status"] == "OK"
    run_id = run.lines["run-id"]
    run_dir = Path(run.lines["run-dir"])
    assert run_dir.parent == e2e.run_root
    assert run_dir.is_dir()
    assert "manifest.json" in {item.name for item in run_dir.iterdir()}
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "OK"
    assert manifest["benchmark"]["status"] == "OK"
    assert manifest["benchmark"]["case_count"] == 20
    artifacts = {item.name for item in run_dir.iterdir()}
    assert {"comparison.json", "report.html"} <= artifacts

    inventory = e2e.run("runs", "list", "--json")
    assert inventory.json is not None
    assert any(item["run_id"] == run_id for item in inventory.json["runs"])

    verification = e2e.run("runs", "verify", run_id, "--json")
    assert verification.returncode in (0, 1)
    assert verification.json["status"] in {"VALID", "INCOMPLETE", "CORRUPT", "UNVERIFIABLE"}

    if "comparison.json" in artifacts:
        explanation = e2e.run("runs", "explain", run_id, "--json")
        assert explanation.returncode == 0
        assert explanation.json["run_id"] == run_id

        report = e2e.run("report", run_id, "--no-open")
        assert report.returncode == 0
        assert Path(report.lines["report"]).is_file()

        served = e2e.run("serve", "--run-dir", str(run_dir), "--check", "--json")
        assert served.returncode == 0
        assert served.json["status"] == "SUCCEEDED"


def test_invalid_fixture_is_a_failure_without_artifacts(e2e: E2EHarness) -> None:
    e2e.fixture_path.write_text('{"not": "a fixture"}', encoding="utf-8")
    result = e2e.run("run", "--no-open")
    assert result.returncode != 0
    assert "FAILED:" in result.stderr
    assert not [
        item for item in e2e.run_root.iterdir() if item.is_dir() and item.name != "empty-run"
    ]


def test_timeout_and_cancellation_cleanup_processes(e2e: E2EHarness) -> None:
    timeout = e2e.run_with_timeout("serve", "--run-dir", str(e2e.empty_run_dir), timeout=0.1)
    assert timeout.timed_out
    assert timeout.process_returncode is not None

    cancelled = e2e.cancel_after_ready("serve", "--run-dir", str(e2e.empty_run_dir))
    assert cancelled.returncode == 0
    assert "stopped" in cancelled.stdout
    assert cancelled.process_returncode is not None

    assert not e2e.live_children()
    assert not [
        item for item in e2e.run_root.iterdir() if item.is_dir() and item.name != "empty-run"
    ]
