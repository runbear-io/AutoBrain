from __future__ import annotations

import json

from typer.testing import CliRunner

from autobrain.cli import app
from autobrain.contracts import (
    ModelAccessMode,
    ModelAccessProfileV1,
    ModelCapabilityStatus,
)
from autobrain.model_access import inspect_model_access


def test_status_contains_complete_local_fallback_matrix_without_secret_values() -> None:
    secret = "sk-fixture-never-persisted"
    status = inspect_model_access(
        {
            "OPENAI_API_KEY": secret,
            "AUTOBRAIN_EMBEDDING_BACKEND": "local-hash",
        }
    )
    assert {profile.mode for profile in status.profiles} == {
        ModelAccessMode.PROVIDER_API_BYOK,
        ModelAccessMode.SUBSCRIPTION_CLI,
        ModelAccessMode.LOCAL_OPENAI_COMPATIBLE,
    }
    assert secret not in repr(status)
    local = next(p for p in status.profiles if p.mode is ModelAccessMode.LOCAL_OPENAI_COMPATIBLE)
    assert local.keyword_only is True
    assert local.smoke_only_hash is True
    assert local.recommendation_eligible is False
    assert local.embeddings is ModelCapabilityStatus.METERING_INCOMPLETE


def test_hash_only_backend_never_becomes_semantic_or_recommendation_eligible() -> None:
    status = inspect_model_access({"AUTOBRAIN_EMBEDDING_BACKEND": "local-hash"})
    assert status.recommendation_eligible is False
    local = next(p for p in status.profiles if p.mode is ModelAccessMode.LOCAL_OPENAI_COMPATIBLE)
    assert local.embeddings is not ModelCapabilityStatus.READY
    assert "hash_embeddings_smoke_only" in local.diagnostics


def test_profile_rejects_eligible_incomplete_capabilities() -> None:
    try:
        ModelAccessProfileV1(
            schema_version=1,
            mode=ModelAccessMode.LOCAL_OPENAI_COMPATIBLE,
            chat=ModelCapabilityStatus.READY,
            embeddings=ModelCapabilityStatus.UNAVAILABLE,
            verifier=ModelCapabilityStatus.READY,
            metering="COST_COMPLETE",
            recommendation_eligible=True,
        )
    except ValueError as error:
        assert "recommendation eligibility" in str(error)
    else:
        raise AssertionError("incomplete profile was accepted as recommendation eligible")


def test_cli_json_reports_each_capability_and_human_output_is_not_json() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["model-access", "status", "--json"], env={})
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["recommendation_eligible"] is False
    assert {profile["mode"] for profile in payload["profiles"]} == {
        mode.value for mode in ModelAccessMode
    }
    local = next(
        profile
        for profile in payload["profiles"]
        if profile["mode"] == ModelAccessMode.LOCAL_OPENAI_COMPATIBLE.value
    )
    assert local["smoke_only_hash"] is True
    assert local["recommendation_eligible"] is False

    human = runner.invoke(app, ["model-access", "status"], env={})
    assert human.exit_code == 0, human.output
    assert "smoke_only_hash=True" in human.stdout
    assert not human.stdout.lstrip().startswith("{")
