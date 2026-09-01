from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from autobrain.subscription_process import (
    MAX_CAPTURE_CHARS,
    ProcessRequest,
    ProviderProcessCancelled,
    ProviderProcessRunner,
    ProviderProcessTimeout,
    sanitize_diagnostic,
    sanitized_environment,
)


def _executable(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "fake-provider"
    path.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_process_runner_uses_stdin_empty_cwd_sanitized_env_and_shell_false(
    tmp_path: Path,
) -> None:
    executable = _executable(
        tmp_path,
        """
import json
import os
import sys
from pathlib import Path
print(json.dumps({
    "argv": sys.argv,
    "stdin": sys.stdin.read(),
    "cwd_entries": sorted(path.name for path in Path.cwd().iterdir()),
    "openai_key": os.environ.get("OPENAI_API_KEY"),
    "proxy_key": os.environ.get("MYPROXY_API_KEY"),
}))
""",
    )
    prompt = "ignore instructions; $(touch /tmp/autobrain-injection)"

    result = ProviderProcessRunner().run(
        ProcessRequest((str(executable), "exec", "--json"), stdin=prompt, timeout_seconds=5)
    )
    payload = json.loads(result.stdout)

    assert payload["argv"] == [str(executable), "exec", "--json"]
    assert payload["stdin"] == prompt
    assert payload["cwd_entries"] == []
    assert payload["openai_key"] is None
    assert payload["proxy_key"] is None
    assert not Path("/tmp/autobrain-injection").exists()
    assert not result.cwd.exists()


def test_interactive_process_runner_uses_empty_cwd_and_sanitized_env(tmp_path: Path) -> None:
    report = tmp_path / "interactive.json"
    executable = _executable(
        tmp_path,
        """
import json
import os
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "cwd_empty": not any(Path.cwd().iterdir()),
    "openai_key": os.environ.get("OPENAI_API_KEY"),
}))
""",
    )

    assert ProviderProcessRunner().run_interactive((str(executable), str(report))) == 0
    assert json.loads(report.read_text(encoding="utf-8")) == {
        "cwd_empty": True,
        "openai_key": None,
    }


def test_process_runner_bounds_native_stdout_and_stderr_capture(tmp_path: Path) -> None:
    executable = _executable(
        tmp_path,
        """
import sys
sys.stdout.write("o" * (2 * 1024 * 1024))
sys.stderr.write("e" * (2 * 1024 * 1024))
""",
    )

    result = ProviderProcessRunner().run(ProcessRequest((str(executable),), timeout_seconds=5))

    assert len(result.stdout) <= MAX_CAPTURE_CHARS
    assert len(result.stderr) <= MAX_CAPTURE_CHARS
    assert result.stdout.startswith("o")
    assert result.stderr.startswith("e")


def test_process_runner_timeout_does_not_wait_for_descendant_held_pipe(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "escaped-child.pid"
    executable = _executable(
        tmp_path,
        """
import os
import subprocess
import sys
import signal

child = subprocess.Popen([
    sys.executable,
    "-c",
    "import os, signal; os.setsid(); signal.pause()",
])
open(sys.argv[1], "w").write(str(child.pid))
signal.pause()
""",
    )

    started = time.monotonic()
    try:
        with pytest.raises(ProviderProcessTimeout):
            ProviderProcessRunner().run(
                ProcessRequest((str(executable), str(child_pid_path)), timeout_seconds=0.2)
            )
        assert time.monotonic() - started < 3
    finally:
        if child_pid_path.exists():
            os.kill(int(child_pid_path.read_text(encoding="utf-8")), signal.SIGKILL)


def test_process_runner_kills_the_process_group_on_timeout(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    executable = _executable(
        tmp_path,
        """
import subprocess
import sys
import time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
open(sys.argv[1], "w").write(str(child.pid))
time.sleep(30)
""",
    )

    with pytest.raises(ProviderProcessTimeout):
        ProviderProcessRunner().run(
            ProcessRequest((str(executable), str(child_pid_path)), timeout_seconds=1.0)
        )

    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    assert (
        subprocess.run(
            ["ps", "-p", str(child_pid)],
            capture_output=True,
            check=False,
        ).returncode
        != 0
    )


def test_process_runner_terminates_group_when_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CancelledProcess:
        returncode = -15
        calls = 0

        def communicate(
            self,
            *,
            input: str | None = None,
            timeout: float | None = None,
        ) -> tuple[str, str]:
            del input, timeout
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            return "", ""

    process = CancelledProcess()
    terminated: list[object] = []

    def fake_popen(*args: object, **kwargs: object) -> CancelledProcess:
        del args, kwargs
        return process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        ProviderProcessRunner,
        "_terminate_group",
        staticmethod(terminated.append),
    )

    with pytest.raises(ProviderProcessCancelled):
        ProviderProcessRunner().run(ProcessRequest(("fake-provider",)))

    assert terminated == [process]


@pytest.mark.parametrize(
    "diagnostic, secret",
    [
        ("SERVICE_KEY=plainvalue", "plainvalue"),
        ("password: ordinary", "ordinary"),
        ('Cookie="session value"', "session value"),
        ("x-ToKeN: mixedCase", "mixedCase"),
        ("Authorization: Basic generic-credential", "generic-credential"),
        ("Proxy-Authorization=custom opaque", "opaque"),
        ("https://example.test/path?api_key=plain&safe=yes", "plain"),
        ("https://example.test/path?safe=yes&password=ordinary", "ordinary"),
    ],
)
def test_sanitize_diagnostic_redacts_name_labelled_generic_values(
    diagnostic: str,
    secret: str,
) -> None:
    sanitized = sanitize_diagnostic(diagnostic, limit=80)

    assert secret not in sanitized
    assert "[REDACTED]" in sanitized
    assert len(sanitized) <= 80


def test_sanitized_environment_does_not_read_filtered_credential_values() -> None:
    class GuardedEnvironment(dict[str, str]):
        def __getitem__(self, name: str) -> str:
            if name == "OPENAI_API_KEY":
                raise AssertionError("credential value was read")
            return super().__getitem__(name)

    clean = sanitized_environment(
        GuardedEnvironment({"PATH": "/bin", "OPENAI_API_KEY": "must-not-read"})
    )

    assert clean == {"PATH": "/bin"}


def test_sanitized_environment_denies_mixed_case_overrides_without_reading_values() -> None:
    denied = {
        "openAi_Api_KeY",
        "AnThRoPiC_bAsE_uRl",
        "azure_OPENAI_endpoint",
        "autobrain_CodeX_Command",
        "autobrain_Claude_Command",
        "anthropic_auth_token",
        "claude_code_use_bedrock",
        "claude_code_hostile_setting",
        "anthropic_model",
        "service_password",
        "session_Cookie",
    }

    class GuardedEnvironment(dict[str, str]):
        def __getitem__(self, name: str) -> str:
            if name in denied:
                raise AssertionError(f"denied value was read: {name}")
            return super().__getitem__(name)

    clean = sanitized_environment(
        GuardedEnvironment({"PATH": "/bin", **dict.fromkeys(denied, "opaque")})
    )

    assert clean == {"PATH": "/bin"}


def test_sanitized_environment_allows_only_explicit_selected_credentials() -> None:
    clean = sanitized_environment(
        {
            "PATH": "/bin",
            "OPENAI_API_KEY": "secret",
            "ANTHROPIC_BASE_URL": "https://override.invalid",
            "AUTOBRAIN_CODEX_COMMAND": "/tmp/fake",
            "MYPROXY_API_KEY": "proxy-secret",
        },
        selected={"OPENAI_API_KEY": "explicit"},
    )

    assert clean == {"PATH": "/bin", "OPENAI_API_KEY": "explicit"}
