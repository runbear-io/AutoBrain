import json

from pytest import MonkeyPatch
from typer.testing import CliRunner

from autobrain.cli import app


def test_cli_exposes_doctor_subcommand() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout


def test_doctor_json_is_machine_readable(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    assert '"schema_version": 1' in result.stdout
    assert "MISSING_PROVIDER" in result.stdout


def test_malformed_callback_port_is_a_typed_independent_check(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOBRAIN_CALLBACK_PORT", "not-a-port")
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["callback"]["status"] == "CAPABILITY_UNAVAILABLE"
    assert "integer from 1 to 65535" in checks["callback"]["detail"]
    assert checks["candidate_pins"]["status"] == "OK"
    assert checks["python"]["status"] == "OK"
