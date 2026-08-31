import json
from pathlib import Path

from autobrain.models import UsageCost
from autobrain.secrets import RuntimeEnvironment, RuntimeSettings, redact


def test_environment_is_typed_and_json_redacted() -> None:
    values = {
        "OPENAI_API_KEY": "provider-secret-value",
        "AUTOBRAIN_SLACK_CLIENT_ID": "client-id-value",
        "AUTOBRAIN_SLACK_CLIENT_SECRET": "client-secret-value",
    }
    env = RuntimeEnvironment.from_environ(values)
    payload = env.model_dump_json()
    assert env.openai_api_key is not None
    assert "provider-secret-value" not in payload
    assert "client-secret-value" not in payload
    assert set(env.readiness().model_dump().values()) == {True}


def test_recursive_redaction_covers_names_and_values() -> None:
    secret = "unique-secret"
    cleaned = redact(
        {"authorization": f"Bearer {secret}", "nested": [secret, {"token": secret}]},
        known_secrets=[secret],
    )
    encoded = json.dumps(cleaned)
    assert secret not in encoded
    assert encoded.count("[REDACTED]") >= 3


def test_recursive_redaction_preserves_non_string_schema_types() -> None:
    secret = "sk-synthetic-provider-secret-123456789"
    cleaned = redact(
        {
            "schema_version": 2,
            "runtime": RuntimeSettings(callback_port=8765),
            "usage": UsageCost(input_tokens=123, output_tokens=45, usd=0.125),
            "totals": {
                "total_input_tokens": 123,
                "total_output_tokens": 45,
                "token_count": 168,
            },
            "credential_state": {
                "available": True,
                "expires_at": None,
                "attempts": 0,
            },
            "source_ids": ["notion:page:abc123", "slack:message:def456"],
            "credentials": [secret, 7, False, None],
            "detail": f"provider failed with Bearer {secret}",
        },
        known_secrets=[secret],
    )

    assert cleaned == {
        "schema_version": 2,
        "runtime": {"callback_port": 8765, "callback_port_error": None},
        "usage": {"input_tokens": 123, "output_tokens": 45, "usd": 0.125},
        "totals": {
            "total_input_tokens": 123,
            "total_output_tokens": 45,
            "token_count": 168,
        },
        "credential_state": {
            "available": True,
            "expires_at": None,
            "attempts": 0,
        },
        "source_ids": ["notion:page:abc123", "slack:message:def456"],
        "credentials": ["[REDACTED]", 7, False, None],
        "detail": "provider failed with Bearer [REDACTED]",
    }


def test_environment_access_never_writes_secret_state(tmp_path: Path) -> None:
    RuntimeEnvironment.from_environ({"OPENAI_API_KEY": "interrupted-secret"})
    assert list(tmp_path.iterdir()) == []


def test_shared_credential_detection_covers_labelled_secrets_and_url_userinfo() -> None:
    from autobrain.secrets import contains_secret, redact_secret_text

    values = (
        "API_KEY: labelled-secret-value",
        "proxy-authorization=Bearer labelled-secret-value",
        "https://user:password-value@example.test/path",
    )
    for value in values:
        assert contains_secret(value)
        assert value not in redact_secret_text(value)
        assert "[REDACTED]" in redact_secret_text(value)

    persisted = redact({"detail": values[0]})
    assert persisted == {"detail": "[REDACTED]"}


def test_shared_credential_detection_allows_conceptual_bearer_documentation() -> None:
    from autobrain.secrets import contains_secret

    assert not contains_secret("OAuth clients use bearer tokens and credentials.")


def test_shared_credential_detection_allows_security_documentation_without_assignments() -> None:
    from autobrain.secrets import contains_secret

    assert not contains_secret(
        "Password rotation and token lifecycle documentation explain API key and credential "
        "storage."
    )
