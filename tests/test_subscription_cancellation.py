from __future__ import annotations

import os
import socket
import sys
import threading

import pytest

from autobrain.cancellation import RunCancellation
from autobrain.subscription_process import (
    ProcessRequest,
    ProviderProcessCancelled,
    ProviderProcessRunner,
)

_HUNG_PROVIDER = """
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


def test_provider_cancellation_terminates_full_process_group() -> None:
    cancellation = RunCancellation()
    outcome: list[BaseException] = []
    before = {thread.ident for thread in threading.enumerate()}

    with socket.socket() as readiness:
        readiness.bind(("127.0.0.1", 0))
        readiness.listen(1)
        readiness.settimeout(1)
        port = readiness.getsockname()[1]

        def target() -> None:
            try:
                ProviderProcessRunner().run(
                    ProcessRequest(
                        (sys.executable, "-c", _HUNG_PROVIDER, str(port)),
                        timeout_seconds=30,
                        cancellation=cancellation,
                    )
                )
            except BaseException as exc:
                outcome.append(exc)

        worker = threading.Thread(target=target, name="hung-provider-test")
        worker.start()
        connection, _address = readiness.accept()
        with connection:
            child_pid = int(connection.recv(64).decode().strip())
        cancellation.cancel()
        cancellation.cancel()
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], ProviderProcessCancelled)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    assert {thread.ident for thread in threading.enumerate()} == before
