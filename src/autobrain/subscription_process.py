"""Constrained subprocess execution for subscription provider CLIs."""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_MAX_DIAGNOSTIC_CHARS = 2048
_SENSITIVE_NAME = re.compile(
    r"(?:^|[_-])(?:API[_-]?KEY|KEY|ACCESS[_-]?TOKEN|AUTH[_-]?TOKEN|TOKEN|BEARER|"
    r"SECRET|PASSWORD|COOKIE)(?:$|[_-])",
    re.IGNORECASE,
)
_SENSITIVE_VALUE = re.compile(r"(?i)(bearer\s+\S+|(?:sk|key|token|secret)[-_][A-Za-z0-9._-]{8,})")
_LABELLED_SECRET = re.compile(
    r"(?i)(?P<label>\b(?:(?:[A-Za-z0-9]+[_-])*(?:api[_-]?key|key|token|secret|"
    r"password|cookie)|(?:proxy-)?authorization)\b\s*[:=]\s*)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^,;&\s]+(?:\s+[^,;&]+)?)"
)
_URL_USERINFO = re.compile(r"(?i)(://)([^/@\s:]+):([^/@\s]+)@")
_PROVIDER_OVERRIDE_NAMES = {
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AUTOBRAIN_CODEX_COMMAND",
    "AUTOBRAIN_SUBSCRIPTION_MODEL",
    "AUTOBRAIN_SUBSCRIPTION_TIMEOUT_SECONDS",
    "CODEX_HOME",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT",
}


@dataclass(frozen=True)
class ProcessRequest:
    argv: tuple[str, ...]
    stdin: str = ""
    timeout_seconds: float = 120.0
    environment: Mapping[str, str] | None = None


@dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    cwd: Path


class ProviderProcessError(RuntimeError):
    pass


class ProviderProcessTimeout(ProviderProcessError):
    pass


class ProviderProcessCancelled(ProviderProcessError):
    pass


def sanitized_environment(
    environment: Mapping[str, str],
    *,
    selected: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Remove provider overrides and credentials, then add explicit selections."""
    denied_names = {name.casefold() for name in _PROVIDER_OVERRIDE_NAMES}
    clean: dict[str, str] = {}
    for name in environment:
        normalized_name = name.casefold()
        if normalized_name in denied_names or _SENSITIVE_NAME.search(name) is not None:
            continue
        if normalized_name.endswith(("_base_url", "_api_base", "_endpoint")):
            continue
        clean[name] = environment[name]
    if selected is not None:
        clean.update(selected)
    return clean


def sanitize_diagnostic(value: str, *, limit: int = _MAX_DIAGNOSTIC_CHARS) -> str:
    """Return bounded single-line diagnostics with labelled credentials redacted."""
    redacted = _URL_USERINFO.sub(r"\1[REDACTED]@", value)
    redacted = _LABELLED_SECRET.sub(r"\g<label>[REDACTED]", redacted)
    redacted = _SENSITIVE_VALUE.sub("[REDACTED]", redacted)
    compact = " ".join(redacted.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 1]}…"


class ProviderProcessRunner:
    """Run a provider command in an isolated cwd and terminate its process group."""

    def run(self, request: ProcessRequest) -> ProcessResult:
        if not request.argv or not request.argv[0]:
            raise ValueError("provider command must not be empty")
        if request.timeout_seconds <= 0:
            raise ValueError("provider timeout must be greater than 0")

        with tempfile.TemporaryDirectory(prefix="autobrain-provider-") as cwd_raw:
            cwd = Path(cwd_raw)
            process = subprocess.Popen(
                request.argv,
                shell=False,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env=sanitized_environment(
                    os.environ,
                    selected=request.environment,
                ),
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(
                    input=request.stdin,
                    timeout=request.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                self._terminate_group(process)
                stdout, stderr = process.communicate()
                raise ProviderProcessTimeout("provider execution timed out") from exc
            except (KeyboardInterrupt, GeneratorExit) as exc:
                self._terminate_group(process)
                process.communicate()
                raise ProviderProcessCancelled("provider execution cancelled") from exc
            return ProcessResult(
                argv=request.argv,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
                cwd=cwd,
            )

    def run_interactive(self, argv: Sequence[str]) -> int:
        """Run an interactive login without capturing terminal input or output."""
        if not argv or not argv[0]:
            raise ValueError("provider command must not be empty")
        with tempfile.TemporaryDirectory(prefix="autobrain-provider-") as cwd_raw:
            process = subprocess.Popen(
                tuple(argv),
                shell=False,
                text=True,
                cwd=Path(cwd_raw),
                env=sanitized_environment(os.environ),
                start_new_session=True,
            )
            try:
                return process.wait()
            except KeyboardInterrupt as exc:
                self._terminate_group(process)
                raise ProviderProcessCancelled("provider execution cancelled") from exc

    @staticmethod
    def _terminate_group(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=1.0)
            # The leader may exit before descendants. Kill the group once more
            # so an inherited provider child cannot outlive cancellation.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            return


def run_interactive_provider_process(args: Sequence[str]) -> int:
    return ProviderProcessRunner().run_interactive(args)


def run_provider_process(
    args: Sequence[str],
    stdin: str,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    """Compatibility adapter for the original three-argument Runner contract."""
    result = ProviderProcessRunner().run(
        ProcessRequest(tuple(args), stdin=stdin, timeout_seconds=timeout_seconds)
    )
    return subprocess.CompletedProcess(
        args=list(result.argv),
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
