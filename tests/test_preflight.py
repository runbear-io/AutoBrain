import json
import sys
from pathlib import Path

import pytest

from autobrain.models import Status
from autobrain.paths import AutoBrainPaths
from autobrain.preflight import CommandResult, Preflight
from autobrain.preflight_support import load_candidate_pins
from autobrain.secrets import RuntimeEnvironment


def _runner(command: tuple[str, ...], timeout: float) -> CommandResult:
    del timeout
    name = Path(command[0]).name
    if name == "codex":
        return CommandResult(returncode=0, stdout="Logged in using ChatGPT", stderr="")
    versions = {"python": "Python 3.12.7", "node": "v25.9.0", "bun": "1.3.14"}
    return CommandResult(returncode=0, stdout=versions[name], stderr="")


def test_preflight_reports_independent_missing_requirements(tmp_path: Path) -> None:
    report = Preflight(
        paths=AutoBrainPaths.from_home(tmp_path),
        environment=RuntimeEnvironment.from_environ({}),
        command_runner=_runner,
        executable_finder=lambda name: None if name == "codex" else f"/fake/{name}",
        keyring_available=lambda: False,
        callback_available=lambda _host, _port: False,
        browser_available=lambda: True,
    ).run()
    checks = {check.name: check for check in report.checks}
    assert checks["chatgpt_subscription"].status is Status.MISSING_PROVIDER
    assert checks["chatgpt_subscription"].detail.startswith("SUBSCRIPTION_CLI_UNAVAILABLE")
    assert "openai_api_key" not in report.environment.model_dump()
    assert checks["slack_source"].status is Status.MCP_AUTH_UNAVAILABLE
    assert checks["keyring"].status is Status.ENV_UNAVAILABLE
    assert checks["callback"].status is Status.CAPABILITY_UNAVAILABLE
    assert checks["node"].status is Status.OK
    assert checks["bun"].status is Status.OK
    json.loads(report.model_dump_json())


def test_python_check_uses_the_running_interpreter(tmp_path: Path) -> None:
    commands: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...], timeout: float) -> CommandResult:
        del timeout
        commands.append(command)
        name = Path(command[0]).name
        version = {
            "python": "Python 3.13.7",
            "node": "v25.9.0",
            "bun": "1.3.14",
            "codex": "Logged in using ChatGPT",
        }[name]
        return CommandResult(returncode=0, stdout=version, stderr="")

    Preflight(
        paths=AutoBrainPaths.from_home(tmp_path),
        environment=RuntimeEnvironment.from_environ({}),
        command_runner=runner,
        executable_finder=lambda name: f"/unsupported/{name}",
    ).run()
    assert commands[0][0] == sys.executable


def test_outdated_node_is_rejected_independently(tmp_path: Path) -> None:
    def runner(command: tuple[str, ...], timeout: float) -> CommandResult:
        del timeout
        name = Path(command[0]).name
        version = "v0.1.0" if name == "node" else _runner(command, 0).stdout
        return CommandResult(returncode=0, stdout=version, stderr="")

    report = Preflight(
        paths=AutoBrainPaths.from_home(tmp_path),
        environment=RuntimeEnvironment.from_environ({}),
        command_runner=runner,
        executable_finder=lambda name: f"/fake/{name}",
        keyring_available=lambda: True,
        callback_available=lambda _host, _port: True,
        browser_available=lambda: True,
    ).run()
    checks = {check.name: check for check in report.checks}
    assert checks["node"].status is Status.ENV_UNAVAILABLE
    assert checks["node"].version == "0.1.0"
    assert checks["bun"].status is Status.OK
    assert checks["candidate_pins"].status is Status.OK


def test_outdated_bun_and_bounded_subprocess_failure_are_typed(tmp_path: Path) -> None:
    def runner(command: tuple[str, ...], timeout: float) -> CommandResult:
        assert timeout <= 3
        if Path(command[0]).name == "bun":
            return CommandResult(returncode=0, stdout="1.3.9", stderr="")
        raise TimeoutError("bounded")

    report = Preflight(
        paths=AutoBrainPaths.from_home(tmp_path),
        environment=RuntimeEnvironment.from_environ({}),
        command_runner=runner,
        executable_finder=lambda name: f"/fake/{name}",
        keyring_available=lambda: True,
        callback_available=lambda _host, _port: True,
        browser_available=lambda: True,
    ).run()
    checks = {check.name: check for check in report.checks}
    assert checks["bun"].status is Status.ENV_UNAVAILABLE
    assert checks["python"].status is Status.ENV_UNAVAILABLE
    assert "bounded" in checks["python"].detail


def test_missing_bun_does_not_crash_other_checks(tmp_path: Path) -> None:
    report = Preflight(
        paths=AutoBrainPaths.from_home(tmp_path),
        environment=RuntimeEnvironment.from_environ({}),
        command_runner=_runner,
        executable_finder=lambda name: None if name == "bun" else f"/fake/{name}",
        keyring_available=lambda: True,
        callback_available=lambda _host, _port: True,
        browser_available=lambda: True,
    ).run()
    checks = {check.name: check for check in report.checks}
    assert checks["bun"].status is Status.ENV_UNAVAILABLE
    assert checks["python"].status is Status.OK
    assert checks["candidate_pins"].status is Status.OK


def test_interruption_leaves_no_secret_or_partial_state(tmp_path: Path) -> None:
    def interrupted(_command: tuple[str, ...], _timeout: float) -> CommandResult:
        raise KeyboardInterrupt

    preflight = Preflight(
        paths=AutoBrainPaths.from_home(tmp_path),
        environment=RuntimeEnvironment.from_environ({"OPENAI_API_KEY": "never-written"}),
        command_runner=interrupted,
        executable_finder=lambda name: f"/fake/{name}",
    )
    with pytest.raises(KeyboardInterrupt):
        preflight.run()
    assert list(tmp_path.iterdir()) == []


def test_symlinked_runs_root_is_a_typed_path_failure(tmp_path: Path) -> None:
    paths = AutoBrainPaths.from_home(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    paths.root.mkdir()
    paths.runs.symlink_to(outside, target_is_directory=True)
    report = Preflight(
        paths=paths,
        environment=RuntimeEnvironment.from_environ({}),
        command_runner=_runner,
        executable_finder=lambda name: f"/fake/{name}",
    ).run()
    checks = {check.name: check for check in report.checks}
    assert checks["paths"].status is Status.ENV_UNAVAILABLE
    assert checks["candidate_pins"].status is Status.OK
    assert list(outside.iterdir()) == []


def test_interrupted_writable_probe_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_unlink = Path.unlink
    interrupted = False

    def unlink_once(path: Path, missing_ok: bool = False) -> None:
        nonlocal interrupted
        if path.name.startswith(".doctor-") and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink_once)
    preflight = Preflight(
        paths=AutoBrainPaths.from_home(tmp_path),
        environment=RuntimeEnvironment.from_environ({}),
        command_runner=_runner,
        executable_finder=lambda name: f"/fake/{name}",
    )
    with pytest.raises(KeyboardInterrupt):
        preflight.run()
    assert list(tmp_path.rglob(".doctor-*")) == []


def test_duplicate_candidate_ids_are_rejected_before_collapse(tmp_path: Path) -> None:
    original = json.loads(Path("candidate-pins.json").read_text(encoding="utf-8"))
    original["candidates"].append(original["candidates"][1].copy())
    pins = tmp_path / "duplicate-pins.json"
    pins.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match=r"duplicate|exactly"):
        load_candidate_pins(pins)


@pytest.mark.parametrize(
    ("candidate_index", "wrong_repository"),
    [
        (0, "https://github.com/atomicstrata/not-llm-wiki-compiler"),
        (1, "https://github.com/not-mem0ai/mem0"),
        (2, "https://github.com/garrytan/not-gbrain"),
        (2, "https://github.com/garrytan/gbrain/"),
    ],
)
def test_candidate_repository_must_match_exact_approved_identity(
    tmp_path: Path, candidate_index: int, wrong_repository: str
) -> None:
    original = json.loads(Path("candidate-pins.json").read_text(encoding="utf-8"))
    original["candidates"][candidate_index]["repository"] = wrong_repository
    pins = tmp_path / "wrong-repository-pins.json"
    pins.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match=r"approved set"):
        load_candidate_pins(pins)


def test_candidate_pin_mismatch_cannot_report_success(tmp_path: Path) -> None:
    pins = tmp_path / "pins.json"
    pins.write_text('{"schema_version":1,"candidates":[]}', encoding="utf-8")
    report = Preflight(
        paths=AutoBrainPaths.from_home(tmp_path),
        environment=RuntimeEnvironment.from_environ({}),
        pins_path=pins,
        command_runner=_runner,
        executable_finder=lambda name: f"/fake/{name}",
        keyring_available=lambda: True,
        callback_available=lambda _host, _port: True,
        browser_available=lambda: True,
    ).run()
    check = next(item for item in report.checks if item.name == "candidate_pins")
    assert check.status is Status.FAILED
    assert report.status is Status.FAILED


def test_preflight_accepts_verified_slack_export_without_app_credentials(
    tmp_path: Path,
) -> None:
    from zipfile import ZipFile

    from autobrain.source_store import SlackSourceStore

    paths = AutoBrainPaths.from_home(tmp_path)
    archive = tmp_path / "slack-export.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("team.json", json.dumps({"id": "T1", "name": "Acme", "domain": "acme"}))
        zip_file.writestr("users.json", json.dumps([{"id": "U1", "name": "ada"}]))
        zip_file.writestr("channels.json", json.dumps([{"id": "C1", "name": "general"}]))
        zip_file.writestr(
            "general/2026-08-19.json",
            json.dumps([{"type": "message", "user": "U1", "text": "What changed?", "ts": "1.1"}]),
        )
    SlackSourceStore(paths.sources).configure_export(archive)

    report = Preflight(
        paths=paths,
        environment=RuntimeEnvironment.from_environ({}),
        command_runner=_runner,
        executable_finder=lambda name: f"/fake/{name}",
        keyring_available=lambda: True,
        callback_available=lambda _host, _port: True,
        browser_available=lambda: True,
    ).run()
    checks = {check.name: check for check in report.checks}

    assert checks["slack_source"].status is Status.OK
    assert "export ready" in checks["slack_source"].detail
    assert "slack_credentials" not in checks


def test_node_below_llm_wiki_minimum_is_rejected(tmp_path: Path) -> None:
    def runner(command: tuple[str, ...], timeout: float) -> CommandResult:
        del timeout
        name = Path(command[0]).name
        version = "v22.11.0" if name == "node" else _runner(command, 0).stdout
        return CommandResult(returncode=0, stdout=version, stderr="")

    report = Preflight(
        paths=AutoBrainPaths.from_home(tmp_path),
        environment=RuntimeEnvironment.from_environ({}),
        command_runner=runner,
        executable_finder=lambda name: f"/fake/{name}",
        keyring_available=lambda: True,
        callback_available=lambda _host, _port: True,
        browser_available=lambda: True,
    ).run()
    checks = {check.name: check for check in report.checks}
    assert checks["node"].status is Status.ENV_UNAVAILABLE
    assert checks["node"].version == "22.11.0"
