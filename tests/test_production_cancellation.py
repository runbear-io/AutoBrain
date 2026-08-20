from __future__ import annotations

import asyncio
import os
import socket
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from autobrain.cancellation import RunCancellation, RunCancelled
from autobrain.candidates.gbrain import run_process
from autobrain.candidates.llm_wiki import _BoundedRunner  # pyright: ignore[reportPrivateUsage]
from autobrain.production import _run_async  # pyright: ignore[reportPrivateUsage]

_HUNG_PROCESS = """
import os
import signal
import socket
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
with socket.create_connection(('127.0.0.1', int(sys.argv[1]))) as ready:
    ready.sendall(f'{child.pid}\\n'.encode())

def stop(_signum, _frame):
    child.wait(timeout=2)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
while True:
    time.sleep(60)
"""


def _assert_pid_absent(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def _run_cancellable(
    tmp_path: Path,
    call: Callable[[Path], None],
    cancellation: RunCancellation,
) -> tuple[int, BaseException]:
    del tmp_path
    outcome: list[BaseException] = []

    with socket.socket() as readiness:
        readiness.bind(("127.0.0.1", 0))
        readiness.listen(1)
        readiness.settimeout(1)
        port = readiness.getsockname()[1]

        def target() -> None:
            try:
                call(Path(str(port)))
            except BaseException as exc:
                outcome.append(exc)

        worker = threading.Thread(target=target, name="production-cancellation-test")
        worker.start()
        connection, _address = readiness.accept()
        with connection:
            child_pid = int(connection.recv(64).decode().strip())
        cancellation.cancel()
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(outcome) == 1
    return child_pid, outcome[0]


def test_llm_wiki_cancellation_terminates_full_process_group(tmp_path: Path) -> None:
    cancellation = RunCancellation()
    runner = _BoundedRunner()
    runner.cancellation = cancellation

    def invoke(socket_path: Path) -> None:
        runner.run(
            [sys.executable, "-c", _HUNG_PROCESS, str(socket_path)],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout=30,
            cleanup_grace=0.5,
        )

    child_pid, error = _run_cancellable(tmp_path, invoke, cancellation)
    assert isinstance(error, RunCancelled)
    _assert_pid_absent(child_pid)


def test_gbrain_cancellation_terminates_full_process_group(tmp_path: Path) -> None:
    cancellation = RunCancellation()

    def invoke(socket_path: Path) -> None:
        run_process(
            [sys.executable, "-c", _HUNG_PROCESS, str(socket_path)],
            tmp_path,
            os.environ.copy(),
            30,
            cancellation,
        )

    child_pid, error = _run_cancellable(tmp_path, invoke, cancellation)
    assert isinstance(error, RunCancelled)
    _assert_pid_absent(child_pid)


def test_async_connector_cancellation_settles_without_thread_leak() -> None:
    cancellation = RunCancellation()
    entered = threading.Event()
    settled = threading.Event()
    before = {thread.ident for thread in threading.enumerate()}
    outcome: list[BaseException] = []

    async def hung_connector() -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            settled.set()

    def target() -> None:
        try:
            _run_async(hung_connector(), cancellation)
        except BaseException as exc:
            outcome.append(exc)

    worker = threading.Thread(target=target, name="hung-connector-test")
    worker.start()
    assert entered.wait(timeout=1)
    cancellation.cancel()
    worker.join(timeout=1)

    assert settled.is_set()
    assert not worker.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], RunCancelled)
    assert {thread.ident for thread in threading.enumerate()} == before
