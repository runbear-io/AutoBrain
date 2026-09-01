from __future__ import annotations

import json
import signal
import subprocess
from http.client import HTTPConnection
from pathlib import Path
from urllib.request import urlopen

from harness import E2EHarness


def _serve(e2e: E2EHarness, run_dir: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        e2e.command("serve", "--run-dir", str(run_dir), "--port", "0"),
        env=e2e.environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def _announcement(process: subprocess.Popen[str]) -> str:
    assert process.stdout is not None
    line = process.stdout.readline()
    assert line.startswith("AutoBrain local fixture")
    url_line = process.stdout.readline()
    assert url_line.startswith("projection: http://127.0.0.1:")
    return url_line.removeprefix("projection: ").removesuffix("/api/v1/run\n")


def test_installed_binary_serve_projects_real_http_and_shuts_down(e2e: E2EHarness) -> None:
    run_dir = e2e.run_root / "RUN-A41F"
    run_dir.mkdir()
    comparison = run_dir / "comparison.json"
    # Use the repository's typed artifact generator to avoid hand-maintaining
    # a second comparison schema in this subprocess test.
    from tests.test_projection import artifact

    comparison.write_text(artifact().model_dump_json(), encoding="utf-8")

    process = _serve(e2e, run_dir)
    try:
        base_url = _announcement(process)
        url = f"{base_url}/api/v1/run"
        with urlopen(url, timeout=5) as response:
            assert response.status == 200
            payload = json.loads(response.read())
        assert payload["status"] == "SUCCEEDED"
        assert payload["projection"]["run_id"] == "RUN-A41F"

        connection = HTTPConnection("127.0.0.1", int(base_url.rsplit(":", 1)[1]), timeout=5)
        try:
            connection.request("GET", "/api/v1/run", headers={"Origin": "https://evil.example"})
            forbidden_origin = connection.getresponse()
            assert forbidden_origin.status == 200
            assert forbidden_origin.getheader("Access-Control-Allow-Origin") is None

            connection.request("GET", "/arbitrary-path")
            assert connection.getresponse().status == 404
        finally:
            connection.close()
    finally:
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=5)

    assert process.returncode == 0, stderr
    assert "stopped" in stdout
