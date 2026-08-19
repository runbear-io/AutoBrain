"""PKCE and anti-CSRF values for authorization-code flows."""

import base64
import hashlib
import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class Pkce:
    verifier: str
    challenge: str
    state: str


def create_pkce() -> Pkce:
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )
    return Pkce(verifier=verifier, challenge=challenge, state=secrets.token_urlsafe(32))
