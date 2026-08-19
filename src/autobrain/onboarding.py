"""Persist whether the AutoBrain interview has been completed."""

from __future__ import annotations

import json
import os
from pathlib import Path

from autobrain.paths import AutoBrainPaths

_ONBOARDING_FILE = "onboarding.json"


def onboarding_path(paths: AutoBrainPaths) -> Path:
    return paths.root / _ONBOARDING_FILE


def is_onboarded(paths: AutoBrainPaths) -> bool:
    path = onboarding_path(paths)
    if not path.is_file() or path.is_symlink():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("complete") is True


def mark_onboarded(paths: AutoBrainPaths) -> None:
    paths.ensure_base_dirs()
    path = onboarding_path(paths)
    payload = json.dumps({"complete": True}, indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")
    os.chmod(path, 0o600)
