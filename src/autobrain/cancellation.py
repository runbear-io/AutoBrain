"""Thread-safe cooperative cancellation with active-operation hooks."""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import suppress


class RunCancelled(Exception):
    """Raised cooperatively when an operator cancels an active run."""


class RunCancellation:
    """Idempotent cancellation signal that can terminate active boundaries."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: dict[int, Callable[[], None]] = {}
        self._next_callback_id = 0

    def cancel(self) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            callbacks = tuple(self._callbacks.values())
        for callback in callbacks:
            with suppress(BaseException):
                callback()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise RunCancelled("operator cancelled run")

    def add_callback(self, callback: Callable[[], None]) -> Callable[[], None]:
        with self._lock:
            if self._event.is_set():
                invoke_now = True
                callback_id = -1
            else:
                invoke_now = False
                callback_id = self._next_callback_id
                self._next_callback_id += 1
                self._callbacks[callback_id] = callback
        if invoke_now:
            callback()

        def remove() -> None:
            if callback_id < 0:
                return
            with self._lock:
                self._callbacks.pop(callback_id, None)

        return remove
