"""Typed environment access and model-boundary redaction."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, SecretStr

from autobrain.models import StrictModel

_SENSITIVE_KEY = re.compile(r"(?:secret|token|password|api[_-]?key|authorization|credential)", re.I)
_URL_USERINFO = re.compile(r"(?i)(://)([^/@\s:]+):([^/@\s]+)@")
_PLACEHOLDER_VALUE = (
    r"(?:<\s*[A-Z0-9_-]+\s*>|\[\s*[A-Z0-9_-]+\s*\]|"
    r"\{\{?\s*[A-Z0-9_-]+\s*\}?\}|\$\{\s*[A-Z0-9_-]+\s*\}|"
    r"(?:YOUR|EXAMPLE|SAMPLE|DUMMY|TEST)[_-][A-Z0-9_-]+|"
    r"PLACEHOLDER(?:[_-]VALUE)?|CHANGEME)"
)
_SAFE_CREDENTIAL_PLACEHOLDER = re.compile(
    rf"(?i)(?:\b(?:bearer)\s+|\b(?:(?:[A-Za-z0-9]+[_-])*(?:api[_-]?key|key|token|secret|"
    rf"password|cookie)|(?:proxy-)?authorization)\b\s*[:=]\s*(?:bearer\s+)?){_PLACEHOLDER_VALUE}"
)
_LABELLED_SECRET = re.compile(
    r"(?i)(?P<label>\b(?:(?:[A-Za-z0-9]+[_-])*(?:api[_-]?key|key|token|secret|"
    r"password|cookie)|(?:proxy-)?authorization)\b\s*[:=]\s*)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^,;&\s]+)"
)
_SECRET_PATTERN = re.compile(
    r"(?i)(?<![a-z0-9])(?:sk-[a-z0-9_-]{8,}|xox[a-z]-[a-z0-9-]{8,}|"
    r"bearer\s+[a-z0-9._~+/=-]{8,}|"
    r"(?:api[_-]?key|token|password|authorization|credential)\s*[=:]\s*[a-z0-9._~+/=-]{8,})"
)


def redact_secret_text(value: str) -> str:
    """Redact credential-shaped values, including labelled values and URL userinfo."""
    placeholders: list[str] = []

    def protect(match: re.Match[str]) -> str:
        placeholders.append(match.group(0))
        return f"__AUTOBRAIN_SAFE_PLACEHOLDER_{len(placeholders) - 1}__"

    redacted = _SAFE_CREDENTIAL_PLACEHOLDER.sub(protect, value)
    redacted = _URL_USERINFO.sub(r"\1[REDACTED]@", redacted)
    redacted = _LABELLED_SECRET.sub("[REDACTED]", redacted)
    redacted = _SECRET_PATTERN.sub("[REDACTED]", redacted)
    for index, placeholder in enumerate(placeholders):
        redacted = redacted.replace(f"__AUTOBRAIN_SAFE_PLACEHOLDER_{index}__", placeholder)
    return redacted


def contains_secret(value: str) -> bool:
    """Return whether text contains a credential-shaped value."""
    return redact_secret_text(value) != value


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
    return _redact_value(value, known_secrets=known_secrets)


def _redact_value(
    value: object,
    *,
    known_secrets: Sequence[str],
    redact_strings: bool = False,
) -> object:
    if isinstance(value, BaseModel):
        return _redact_value(
            value.model_dump(mode="python"),
            known_secrets=known_secrets,
            redact_strings=redact_strings,
        )
    if is_dataclass(value) and not isinstance(value, type):
        return _redact_value(
            asdict(cast(Any, value)),
            known_secrets=known_secrets,
            redact_strings=redact_strings,
        )
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            str(key): _redact_value(
                item,
                known_secrets=known_secrets,
                redact_strings=redact_strings or bool(_SENSITIVE_KEY.search(str(key))),
            )
            for key, item in mapping.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        items = [
            _redact_value(
                item,
                known_secrets=known_secrets,
                redact_strings=redact_strings,
            )
            for item in cast(Sequence[object], value)
        ]
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, SecretStr):
        return "[REDACTED]"
    if isinstance(value, str):
        if redact_strings:
            return "[REDACTED]"
        result = value
        for secret in sorted((item for item in known_secrets if item), key=len, reverse=True):
            if secret:
                result = result.replace(secret, "[REDACTED]")
        return redact_secret_text(result)
    return value
