"""Todo 4 RED proof: doctor does not yet expose precredential readiness."""

from pathlib import Path
from zipfile import ZipFile

from autobrain.connectors.readiness import TransportGovernanceCode
from autobrain.contracts import SourceProvider, SourceTransportMode
from autobrain.models import Status
from autobrain.paths import AutoBrainPaths
from autobrain.preflight import CommandResult, Preflight
from autobrain.secrets import RuntimeEnvironment


def _runner(command: tuple[str, ...], timeout: float) -> CommandResult:
    del timeout
    name = Path(command[0]).name
    versions = {"python": "Python 3.12.7", "node": "v25.9.0", "bun": "1.3.14"}
    return CommandResult(
        returncode=0,
        stdout=versions.get("python" if name.startswith("python") else name, "provider output"),
        stderr="",
    )


def test_doctor_exposes_typed_precredential_readiness_and_governance(tmp_path: Path) -> None:
    report = Preflight(
        paths=AutoBrainPaths.from_home(tmp_path),
        environment=RuntimeEnvironment.from_environ({}),
        command_runner=_runner,
        executable_finder=lambda name: None if name == "codex" else f"/fake/{name}",
        embedding_environ={},
    ).run()

    readiness = report.readiness
    assert readiness.schema_version == 1
    assert readiness.ready is False
    assert readiness.source_transport.provider is SourceProvider.SLACK
    assert readiness.source_transport.mode is SourceTransportMode.EXPORT_ARCHIVE
    assert readiness.model_access.mode.value in {
        "provider_api_byok",
        "subscription_cli",
        "local_openai_compatible",
    }
    assert readiness.governance_codes
    assert readiness.remediation
    assert report.status in {Status.MCP_AUTH_UNAVAILABLE, Status.MISSING_PROVIDER}


def test_unsupported_transport_and_archive_drift_are_typed_in_aggregate(
    tmp_path: Path,
) -> None:
    unsupported = Preflight(
        paths=AutoBrainPaths.from_home(tmp_path / "unsupported"),
        environment=RuntimeEnvironment.from_environ({}),
        command_runner=_runner,
        executable_finder=lambda name: f"/fake/{name}",
        source_provider=SourceProvider.CONFLUENCE,
        source_mode=SourceTransportMode.OAUTH_3LO_REST,
    ).run()
    assert unsupported.readiness.ready is False
    assert unsupported.readiness.source_transport.governance_code is (
        TransportGovernanceCode.SOURCE_TRANSPORT_UNAVAILABLE
    )

    archive = tmp_path / "slack.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("team.json", '{"id":"T1","name":"Acme","domain":"acme"}')
        output.writestr("users.json", "[]")
        output.writestr("channels.json", '[{"id":"C1","name":"general"}]')
        output.writestr(
            "general/2026-08-19.json",
            '[{"type":"message","user":"U1","text":"fixture","ts":"1.1"}]',
        )
    paths = AutoBrainPaths.from_home(tmp_path / "drift")
    from autobrain.source_store import SlackSourceStore

    SlackSourceStore(paths.sources).configure_export(archive)
    archive.write_bytes(archive.read_bytes() + b"drift")
    drifted = Preflight(
        paths=paths,
        environment=RuntimeEnvironment.from_environ({}),
        command_runner=_runner,
        executable_finder=lambda name: f"/fake/{name}",
    ).run()
    assert drifted.readiness.ready is False
    assert drifted.readiness.source_transport.governance_code is (
        TransportGovernanceCode.SOURCE_TRANSPORT_UNAVAILABLE
    )
    assert "changed" in drifted.readiness.source_transport.detail.lower()
