"""Confined atomic files and cross-process locks for OAuth state."""

import fcntl
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Self, cast

from autobrain.auth.models import OAuthError


class AuthPathError(OAuthError):
    """OAuth state path is symlinked or escapes its configured root."""


class ProcessFileLock:
    def __init__(self, files: "SecureAuthFiles", key: str) -> None:
        self.files, self.key = files, key
        self.descriptor: int | None = None

    def __enter__(self) -> Self:
        self.files.ensure_root()
        locks = self.files.locks
        if locks.is_symlink():
            raise AuthPathError("OAuth lock directory cannot be a symlink")
        locks.mkdir(mode=0o700, exist_ok=True)
        self.files.validate(locks)
        digest = hashlib.sha256(self.key.encode()).hexdigest()
        path = locks / f"{digest}.lock"
        self.files.validate(path)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise AuthPathError("OAuth lock path could not be opened safely") from exc
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise AuthPathError("OAuth lock path is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        self.descriptor = descriptor
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self.descriptor is not None:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = None


class SecureAuthFiles:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.index = root / "oauth-index.json"
        self.tokens = root / "oauth-tokens.json"
        self.locks = root / "locks"

    def ensure_root(self) -> None:
        if self.root.is_symlink() or self.root.parent.is_symlink():
            raise AuthPathError("OAuth state root or parent cannot be a symlink")
        expected = self.root.parent.resolve() / self.root.name
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.resolve() != expected:
            raise AuthPathError("OAuth state root escapes its configured parent")
        os.chmod(self.root, 0o700)
        self.validate(self.index)
        self.validate(self.tokens)

    def validate(self, path: Path) -> None:
        if path.is_symlink():
            raise AuthPathError("OAuth state path cannot be a symlink")
        if not path.resolve(strict=False).is_relative_to(self.root.resolve()):
            raise AuthPathError("OAuth state path escapes its configured root")

    def read_mapping(self, path: Path) -> dict[str, dict[str, object]]:
        self.ensure_root()
        self.validate(path)
        if not path.exists():
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("stored OAuth state is malformed")
        return cast(dict[str, dict[str, object]], value)

    def write_atomic(self, path: Path, value: object) -> None:
        self.ensure_root()
        self.validate(path)
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.root)
        temporary = Path(name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            self.validate(path)
            os.replace(temporary, path)
            os.chmod(path, 0o600)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def process_lock(self, key: str) -> ProcessFileLock:
        return ProcessFileLock(self, key)
