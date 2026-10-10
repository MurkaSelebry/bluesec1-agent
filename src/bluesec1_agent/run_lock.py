"""Process lock plus durable intent, shared by every worktree of an account.

A released OS lock says nothing about server state. Unknown/closing intents
survive process exit and require external reconciliation before a new run.
"""

import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Self

from bluesec1_client.remote.errors import is_definite_create_rejection


class RunLockError(RuntimeError):
    """A live owner or unresolved remote intent forbids a new CreateRun."""


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.intent_path = self.path / "intent.json"
        self._descriptor: int | None = None
        self._intent: dict[str, Any] | None = None

    def __enter__(self) -> Self:
        if self._descriptor is not None:
            raise RunLockError("The run lock is already held by this object.")
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(self.path / "run.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RunLockError("Another process holds the shared run lock.") from None
            if self.intent_path.exists():
                raise RunLockError("An unresolved run intent blocks new runs; reconcile it first.")
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        self._intent = None
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
        # Do not unlink the lock inode or the unresolved intent.

    def observe(self, event: str, payload: Any) -> None:
        """Attach directly as RemoteBenchmarkClient.lifecycle_observer.

        create_started is a strict callback invoked before RPC. The other callbacks
        persist received knowledge. No callback queries an invented status RPC.
        """
        if self._descriptor is None:
            raise RunLockError("Run lifecycle requires the shared lock to be held.")
        if event == "create_started":
            if self._intent is not None or self.intent_path.exists():
                raise RunLockError("A previous run intent has not been resolved.")
            self._intent = {
                "schema_version": 1,
                "idempotency_key": payload.idempotency_key,
                "run_id": None,
                "status": "creating",
            }
            self._write_intent()
        elif event == "create_finished":
            self._require_intent()
            self._intent.update(run_id=payload.run_id, status=payload.state)
            self._write_intent()
        elif event == "create_failed":
            self._require_intent()
            if is_definite_create_rejection(payload) and self._intent["run_id"] is None:
                self._clear_intent()
            else:
                self._intent.update(status="unknown", error_type=type(payload).__name__)
                self._write_intent()
        elif event == "cleanup_finished":
            self._require_intent()
            if payload.status == "closed_confirmed" and payload.run_id == self._intent["run_id"]:
                self._clear_intent()
            else:
                self._intent.update(status=payload.status, error_type=payload.error_type)
                self._write_intent()

    def _require_intent(self) -> None:
        if self._intent is None:
            raise RunLockError("A run intent must precede lifecycle outcomes.")

    def _sync_directory(self) -> None:
        descriptor = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _clear_intent(self) -> None:
        self.intent_path.unlink()
        self._sync_directory()
        self._intent = None

    def _write_intent(self) -> None:
        descriptor, name = tempfile.mkstemp(prefix=".intent-", dir=self.path)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self._intent, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.intent_path)
            self._sync_directory()
        finally:
            temporary.unlink(missing_ok=True)
