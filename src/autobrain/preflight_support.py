"""Low-level bounded probes used by doctor."""

import os
import re
import socket
import subprocess
import tempfile
import webbrowser
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import keyring

from autobrain.models import CandidatePin, CandidatePins


def candidate_pin_matches(
    pin: CandidatePin,
    *,
    distribution: str,
    version: str,
    commit: str | None,
) -> bool:
    """Match native identity to the approved registry entry without candidate branches."""
    return (
        pin.distribution == distribution
        and pin.version == version
        and commit is not None
        and pin.commit == commit
    )


EXPECTED_PINS = {
    "llm-wiki": (
        "llm-wiki-compiler",
        "https://github.com/atomicstrata/llm-wiki-compiler",
        "1.1.0",
        "3e17bcfe8b50f24c14c6bcda0cb9224d94fd8206",
        "MIT",
    ),
    "mem0": (
        "mem0ai",
        "https://github.com/mem0ai/mem0",
        "2.0.18",
        "001c235229be8795e3834520467bd0d661ed8f34",
        "Apache-2.0",
    ),
    "gbrain": (
        "gbrain",
        "https://github.com/garrytan/gbrain",
        "0.46.19.0",
        "f49ca569232dbc0d8e0783d84606115e3bfe5ab1",
        "MIT",
    ),
}


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def run_command(command: tuple[str, ...], timeout: float) -> CommandResult:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"command timed out after {timeout:g}s") from error
    return CommandResult(result.returncode, result.stdout.strip(), result.stderr.strip())


def keyring_available() -> bool:
    try:
        return float(keyring.get_keyring().priority) > 0
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def callback_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind((host, port))
        return True
    except OSError:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind((host, 0))
            return True
        except OSError:
            return False


def browser_available() -> bool:
    try:
        webbrowser.get()
        return True
    except webbrowser.Error:
        return False


def parse_version(value: str) -> tuple[int, int, int] | None:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)


def format_version(version: tuple[int, int, int]) -> str:
    return ".".join(map(str, version))


def probe_writable(directories: Iterable[Path]) -> None:
    for directory in directories:
        descriptor: int | None = None
        probe: Path | None = None
        try:
            descriptor, raw_path = tempfile.mkstemp(prefix=".doctor-", dir=directory)
            probe = Path(raw_path)
            os.close(descriptor)
            descriptor = None
            probe.unlink()
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            if probe is not None:
                probe.unlink(missing_ok=True)


def load_candidate_pins(path: Path | None = None) -> CandidatePins:
    if path is None:
        text = resources.files("autobrain").joinpath("candidate-pins.json").read_text("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    pins = CandidatePins.model_validate_json(text)
    expected_ids = set(EXPECTED_PINS)
    actual_ids = [pin.id.value for pin in pins.candidates]
    if len(actual_ids) != len(expected_ids):
        raise ValueError("candidate pins must contain exactly three entries")
    if set(actual_ids) != expected_ids:
        raise ValueError("candidate pin IDs do not match the approved set")
    actual = {
        pin.id.value: (pin.distribution, pin.repository, pin.version, pin.commit, pin.license)
        for pin in pins.candidates
    }
    if pins.schema_version != 1 or actual != EXPECTED_PINS:
        raise ValueError("candidate pins do not match the approved set")
    return pins
