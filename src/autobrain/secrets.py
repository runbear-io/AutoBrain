"""Typed environment access and model-boundary redaction."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, SecretStr

from autobrain.models import StrictModel

_SENSITIVE_KEY = re.compile(r"(?:secret|token|password|api[_-]?key|authorization|credential)", re.I)
_SECRET_PATTERN = re.compile(
    r"(?i)(?<![a-z0-9])(?:sk-[a-z0-9_-]{8,}|xox[a-z]-[a-z0-9-]{8,}|"
    r"bearer\s+[a-z0-9._~+/=-]{8,}|"
    r"(?:api[_-]?key|token|password|authorization|credential)[=: ]+[a-z0-9._~+/=-]{8,}|"
    r"//[^/\s:@]+:[^@\s]+@)"
)
type Redactable = (
    bool
    | int
    | float
    | str
    | SecretStr
    | list[Redactable]
    | tuple[Redactable, ...]
    | Mapping[str, Redactable]
    | None
)


class EnvironmentReadiness(StrictModel):
    slack_client_id: bool
    slack_client_secret: bool


class RuntimeSettings(StrictModel):
    callback_port: int = 8765
    callback_port_error: str | None = None

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> "RuntimeSettings":
        raw = environ.get("AUTOBRAIN_CALLBACK_PORT")
        if raw is None:
            return cls()
        try:
            port = int(raw)
        except ValueError:
            return cls(
                callback_port_error="AUTOBRAIN_CALLBACK_PORT must be an integer from 1 to 65535"
            )
        if not 1 <= port <= 65535:
            return cls(
                callback_port_error="AUTOBRAIN_CALLBACK_PORT must be an integer from 1 to 65535"
            )
        return cls(callback_port=port)


class RuntimeEnvironment(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    openai_api_key: SecretStr | None = None
    slack_client_id: SecretStr | None = None
    slack_client_secret: SecretStr | None = None

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> "RuntimeEnvironment":
        def value(name: str) -> SecretStr | None:
            raw = environ.get(name)
            return SecretStr(raw) if raw else None

        return cls(
            openai_api_key=value("OPENAI_API_KEY"),
            slack_client_id=value("AUTOBRAIN_SLACK_CLIENT_ID"),
            slack_client_secret=value("AUTOBRAIN_SLACK_CLIENT_SECRET"),
        )

    def readiness(self) -> EnvironmentReadiness:
        return EnvironmentReadiness(
            slack_client_id=self.slack_client_id is not None,
            slack_client_secret=self.slack_client_secret is not None,
        )

    def known_secret_values(self) -> tuple[str, ...]:
        values = (self.openai_api_key, self.slack_client_id, self.slack_client_secret)
        return tuple(value.get_secret_value() for value in values if value is not None)


def redact(value: object, *, known_secrets: Sequence[str] = ()) -> object:
    """Return a recursively redacted copy suitable for logs and artifacts."""
    if isinstance(value, BaseModel):
        return redact(value.model_dump(mode="python"), known_secrets=known_secrets)
    if is_dataclass(value) and not isinstance(value, type):
        return redact(asdict(cast(Any, value)), known_secrets=known_secrets)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            str(key): "[REDACTED]"
            if _SENSITIVE_KEY.search(str(key))
            else redact(item, known_secrets=known_secrets)
            for key, item in mapping.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        items = [
            redact(item, known_secrets=known_secrets) for item in cast(Sequence[object], value)
        ]
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, SecretStr):
        return "[REDACTED]"
    if isinstance(value, str):
        result = value
        for secret in sorted((item for item in known_secrets if item), key=len, reverse=True):
            if secret:
                result = result.replace(secret, "[REDACTED]")
        return _SECRET_PATTERN.sub("[REDACTED]", result)
    return value
