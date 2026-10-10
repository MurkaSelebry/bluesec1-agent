import asyncio
import json
import subprocess
import sys

import pytest
from test_lifecycle import LifecycleTransport

from bluesec1_client import RemoteBenchmarkClient
from bluesec1_client.remote.errors import RemoteAuthenticationError, RemoteTransportError


def test_second_process_is_blocked_before_creating_a_run(tmp_path):
    from bluesec1_agent.run_lock import RunLock

    script = """
import sys
from pathlib import Path
from bluesec1_agent.run_lock import RunLock, RunLockError
try:
    with RunLock(Path(sys.argv[1])):
        Path(sys.argv[2]).write_text("CreateRun reached")
except RunLockError:
    sys.exit(7)
"""
    reached = tmp_path / "reached"
    with RunLock(tmp_path):
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(reached)])
    assert result.returncode == 7
    assert not reached.exists()


@pytest.mark.parametrize("close_state,blocked", [("closed", False), ("closing", True)])
def test_client_wiring_persists_intent_before_rpc_and_retains_closing(
    tmp_path, close_state, blocked
):
    from bluesec1_agent.run_lock import RunLock, RunLockError

    class IntentTransport(LifecycleTransport):
        async def create_run(self, command):
            intent = json.loads((tmp_path / "intent.json").read_text())
            assert intent["idempotency_key"] == command.idempotency_key
            assert intent["run_id"] is None
            return await super().create_run(command)

    async def exercise():
        with RunLock(tmp_path) as lock:
            client = RemoteBenchmarkClient(
                transport=IntentTransport(close_state=close_state), lifecycle_observer=lock.observe
            )
            async with client:
                assert json.loads((tmp_path / "intent.json").read_text())["run_id"] == "run-1"

    asyncio.run(exercise())
    if blocked:
        with pytest.raises(RunLockError), RunLock(tmp_path):
            pass
    else:
        with RunLock(tmp_path):
            pass


@pytest.mark.parametrize(
    "error,blocked",
    [(RemoteTransportError("lost create"), True), (RemoteAuthenticationError("denied"), False)],
)
def test_create_failure_persists_unknown_intent_without_run_id(tmp_path, error, blocked):
    from bluesec1_agent.run_lock import RunLock, RunLockError

    async def exercise():
        with RunLock(tmp_path) as lock:
            client = RemoteBenchmarkClient(
                transport=LifecycleTransport(create_error=error), lifecycle_observer=lock.observe
            )
            with pytest.raises(type(error)):
                await client.open()

    asyncio.run(exercise())
    if blocked:
        intent = json.loads((tmp_path / "intent.json").read_text())
        assert intent["run_id"] is None
        assert intent["status"] == "unknown"
        with pytest.raises(RunLockError), RunLock(tmp_path):
            pass
    else:
        with RunLock(tmp_path):
            pass


def test_intent_write_failure_prevents_create_rpc(tmp_path):
    from bluesec1_agent.run_lock import RunLock

    async def exercise():
        transport = LifecycleTransport()
        with RunLock(tmp_path) as lock:
            # Remove the empty parent after acquiring the lock: a real filesystem failure.
            (tmp_path / "run.lock").unlink()
            tmp_path.rmdir()
            client = RemoteBenchmarkClient(transport=transport, lifecycle_observer=lock.observe)
            with pytest.raises(OSError):
                await client.open()
            assert transport.creates == []

    asyncio.run(exercise())


def test_crashed_process_intent_blocks_a_new_process_after_os_lock_release(tmp_path):
    create = """
import os, sys
from pathlib import Path
from types import SimpleNamespace
from bluesec1_agent.run_lock import RunLock
with RunLock(Path(sys.argv[1])) as lock:
    lock.observe("create_started", SimpleNamespace(idempotency_key="logical-create"))
    os._exit(0)
"""
    subprocess.run([sys.executable, "-c", create, str(tmp_path)], check=True)
    attempt = """
import sys
from pathlib import Path
from bluesec1_agent.run_lock import RunLock, RunLockError
try:
    with RunLock(Path(sys.argv[1])):
        Path(sys.argv[2]).write_text("CreateRun reached")
except RunLockError:
    sys.exit(7)
"""
    reached = tmp_path / "reached"
    result = subprocess.run([sys.executable, "-c", attempt, str(tmp_path), str(reached)])
    assert result.returncode == 7
    assert not reached.exists()
    assert json.loads((tmp_path / "intent.json").read_text())["run_id"] is None
