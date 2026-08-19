import json
from pathlib import Path

from autobrain.secrets import RuntimeEnvironment, redact


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


def test_environment_access_never_writes_secret_state(tmp_path: Path) -> None:
    RuntimeEnvironment.from_environ({"OPENAI_API_KEY": "interrupted-secret"})
    assert list(tmp_path.iterdir()) == []
