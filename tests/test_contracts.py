"""Adversarial coverage for the versioned source and projection contracts."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError

from autobrain.contracts import (
    JobSpecV1,
    ModelAccessStatusProjectionV1,
    SourceConnectionStatusProjectionV1,
    SourceOAuthAppConfigV1,
    SourceOAuthAppV1,
    SourceProvider,
    SourceSelectionV1,
    SourceTransportMode,
    exported_json_schemas,
    json_schema_bytes,
)


def _job() -> dict[str, object]:
    return {
        "schema_version": 1,
        "job_id": "00000000-0000-4000-8000-000000000001",
        "run_id": "00000000-0000-4000-8000-000000000002",
        "workspace_id": "00000000-0000-4000-8000-000000000003",
        "runner_id": "00000000-0000-4000-8000-000000000004",
        "issued_at": "2026-08-24T00:00:00Z",
        "expires_at": "2026-08-24T00:10:00Z",
        "nonce": "nonce-value",
        "source_selections": [{"provider": "slack", "mode": "export_archive"}],
        "candidate_modes": [{"candidate_id": "gbrain", "mode_id": "grounded_qa"}],
        "required_model_capabilities": ["chat"],
        "budget_micro_usd": 1,
        "max_questions": 1,
        "registry_snapshot_hash": "a" * 64,
        "key_id": "key-id",
        "signature": "signature",
    }


def test_every_registered_provider_mode_round_trips() -> None:
    allowed = {
        SourceProvider.SLACK: {
            SourceTransportMode.EXPORT_ARCHIVE,
            SourceTransportMode.LIVE_MCP_OAUTH,
        },
        SourceProvider.NOTION: {SourceTransportMode.LIVE_MCP_OAUTH},
        SourceProvider.CONFLUENCE: {SourceTransportMode.OAUTH_3LO_REST},
        SourceProvider.GOOGLE_DRIVE: {SourceTransportMode.OAUTH_REST},
        SourceProvider.SHAREPOINT: {SourceTransportMode.GRAPH_OAUTH_PREVIEW},
    }
    for provider, modes in allowed.items():
        assert modes
        for mode in modes:
            selection = SourceSelectionV1(provider=provider, mode=mode)
            assert SourceSelectionV1.model_validate_json(selection.model_dump_json()) == selection
            app = SourceOAuthAppV1(
                provider=provider,
                mode=mode,
                client_id="fixture-client",
                redirect_uri="https://app.example.test/oauth/callback",
            )
            assert SourceOAuthAppV1.model_validate_json(app.model_dump_json()) == app


@pytest.mark.parametrize(
    ("provider", "mode"),
    [
        (provider, mode)
        for provider, modes in {
            SourceProvider.SLACK: {
                SourceTransportMode.EXPORT_ARCHIVE,
                SourceTransportMode.LIVE_MCP_OAUTH,
            },
            SourceProvider.NOTION: {SourceTransportMode.LIVE_MCP_OAUTH},
            SourceProvider.CONFLUENCE: {SourceTransportMode.OAUTH_3LO_REST},
            SourceProvider.GOOGLE_DRIVE: {SourceTransportMode.OAUTH_REST},
            SourceProvider.SHAREPOINT: {SourceTransportMode.GRAPH_OAUTH_PREVIEW},
        }.items()
        for mode in SourceTransportMode
        if mode not in modes
    ],
)
def test_unregistered_provider_mode_pairs_are_rejected(
    provider: SourceProvider, mode: SourceTransportMode
) -> None:
    with pytest.raises(ValidationError, match="not allowed"):
        SourceSelectionV1(provider=provider, mode=mode)
    with pytest.raises(ValidationError, match="not allowed"):
        SourceOAuthAppV1(
            provider=provider,
            mode=mode,
            client_id="fixture-client",
            redirect_uri="https://app.example.test/oauth/callback",
        )


def test_source_config_is_frozen_strict_and_rejects_unknown_fields() -> None:
    config = SourceOAuthAppConfigV1(
        schema_version=1,
        apps=[
            SourceOAuthAppV1(
                provider=SourceProvider.SLACK,
                mode=SourceTransportMode.EXPORT_ARCHIVE,
                client_id="fixture-client",
                redirect_uri="https://app.example.test/oauth/callback",
            )
        ],
    )
    with pytest.raises(ValidationError):
        config.apps = []  # type: ignore[misc]
    with pytest.raises(ValidationError):
        SourceOAuthAppConfigV1.model_validate(
            {**config.model_dump(mode="json"), "secret": "must-not-cross-boundary"}
        )
    with pytest.raises(ValidationError):
        SourceOAuthAppConfigV1.model_validate({"schema_version": 2, "apps": []})


def test_redirect_uri_rejects_credentials_query_fragment_and_non_https() -> None:
    for redirect_uri in (
        "http://app.example.test/callback",
        "https://user:secret@app.example.test/callback",
        "https://app.example.test/callback?code=leak",
        "https://app.example.test/callback#fragment",
    ):
        with pytest.raises(ValidationError):
            SourceOAuthAppV1(
                provider=SourceProvider.SLACK,
                mode=SourceTransportMode.EXPORT_ARCHIVE,
                client_id="fixture-client",
                redirect_uri=redirect_uri,
            )


def test_connection_projection_is_consistent_and_redacts_no_credentials() -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "request_id": "00000000-0000-4000-8000-000000000001",
        "provider": "slack",
        "mode": "export_archive",
        "state": "READY",
        "ready": True,
        "credential_present": False,
        "diagnostics": [],
    }
    projection = SourceConnectionStatusProjectionV1.model_validate_json(json.dumps(payload))
    assert projection.model_dump(mode="json")["credential_present"] is False
    assert "secret" not in projection.model_dump_json().casefold()


def test_versioned_contracts_reject_omitted_schema_version() -> None:
    cases: list[tuple[type[BaseModel], dict[str, Any]]] = [
        (SourceOAuthAppConfigV1, {"apps": []}),
        (
            SourceConnectionStatusProjectionV1,
            {
                "request_id": "00000000-0000-4000-8000-000000000001",
                "provider": "slack",
                "mode": "export_archive",
                "state": "FAILED",
                "ready": False,
                "credential_present": False,
            },
        ),
        (
            ModelAccessStatusProjectionV1,
            {
                "mode": "local_openai_compatible",
                "chat_ready": False,
                "embeddings_ready": False,
                "verifier_ready": False,
                "metering_status": "COST_UNAVAILABLE",
                "recommendation_eligible": False,
            },
        ),
        (JobSpecV1, {key: value for key, value in _job().items() if key != "schema_version"}),
    ]
    for model, payload in cases:
        with pytest.raises(ValidationError, match="schema_version"):
            model.model_validate_json(json.dumps(payload))


def test_versioned_schema_snapshots_require_schema_version() -> None:
    for schema in exported_json_schemas().values():
        required = cast(list[object], schema.get("required", []))
        assert "schema_version" in required


def test_job_schema_round_trip_and_unknown_fields_fail_closed() -> None:
    job = JobSpecV1.model_validate_json(json.dumps(_job()))
    assert JobSpecV1.model_validate_json(job.model_dump_json()) == job
    with pytest.raises(ValidationError):
        JobSpecV1.model_validate({**_job(), "evidence": "oracle corpus"})


def test_committed_json_schema_snapshots_match_models() -> None:
    for filename, schema in exported_json_schemas().items():
        snapshot = Path("schemas") / filename
        assert snapshot.read_bytes() == json_schema_bytes(schema)


def test_job_rejects_duplicate_source_selection() -> None:
    payload = deepcopy(_job())
    payload["source_selections"] = [
        {"provider": "slack", "mode": "export_archive"},
        {"provider": "slack", "mode": "export_archive"},
    ]
    with pytest.raises(ValidationError, match="duplicate"):
        JobSpecV1.model_validate_json(json.dumps(payload))
