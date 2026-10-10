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


@pytest.mark.parametrize(
    "reason", ["SUBJECT_ACTIVE_RUN_LIMIT_REACHED", "PROCESS_ACTIVE_RUN_LIMIT_REACHED"]
)
def test_authoritative_create_capacity_rejection_clears_durable_intent(tmp_path, reason):
    from types import SimpleNamespace

    import grpc
    from test_remote_contract import make_rpc_error

    from bluesec1_agent.run_lock import RunLock
    from bluesec1_client.errors import RunCapacityExceededError
    from bluesec1_client.remote.transport.grpc import GrpcRemoteTransport

    requests = []

    async def reject_create(request, **kwargs):
        requests.append(request)
        intent = json.loads((tmp_path / "intent.json").read_text())
        assert intent["idempotency_key"] == request.idempotency_key
        assert intent["run_id"] is None
        raise make_rpc_error(grpc.StatusCode.RESOURCE_EXHAUSTED, reason=reason)

    async def exercise():
        transport = GrpcRemoteTransport(
            endpoint="unused.invalid",
            token="synthetic",
            stub=SimpleNamespace(CreateRun=reject_create),
        )
        with RunLock(tmp_path) as lock:
            client = RemoteBenchmarkClient(transport=transport, lifecycle_observer=lock.observe)
            assert getattr(client, "run_creation_rejected", False) is False
            with pytest.raises(RunCapacityExceededError):
                await client.open()
            assert client.cleanup_outcome.status == "failed"
            assert client.run_creation_rejected is True
            assert client.run is None
            assert client.cleanup_outcome.run_id is None
        with RunLock(tmp_path):
            pass  # Proved no allocation; this test does not launch another CreateRun.

    asyncio.run(exercise())
    assert len(requests) == 1
    assert not (tmp_path / "intent.json").exists()


@pytest.mark.parametrize("error_kind", ["run_unknown", "task", "operation", "idempotency"])
def test_other_capacity_failures_remain_unknown_and_keep_intent(tmp_path, error_kind):
    from bluesec1_agent.run_lock import RunLock, RunLockError
    from bluesec1_client.errors import (
        IdempotencyCapacityExceededError,
        OperationCapacityExceededError,
        RunCapacityExceededError,
        TaskCapacityExceededError,
    )

    classes = {
        "run_unknown": RunCapacityExceededError,
        "task": TaskCapacityExceededError,
        "operation": OperationCapacityExceededError,
        "idempotency": IdempotencyCapacityExceededError,
    }
    reason = (
        "UNKNOWN_RUN_LIMIT" if error_kind == "run_unknown" else "SUBJECT_ACTIVE_RUN_LIMIT_REACHED"
    )
    error = classes[error_kind]("capacity", reason=reason)

    async def exercise():
        with RunLock(tmp_path) as lock:
            client = RemoteBenchmarkClient(
                transport=LifecycleTransport(create_error=error), lifecycle_observer=lock.observe
            )
            with pytest.raises(type(error)):
                await client.open()
            assert client.run_creation_rejected is False
            assert client.cleanup_outcome.status == "unknown"
        intent = json.loads((tmp_path / "intent.json").read_text())
        assert intent["run_id"] is None and intent["status"] == "unknown"
        with pytest.raises(RunLockError), RunLock(tmp_path):
            pass

    asyncio.run(exercise())


def test_auth_create_rejection_is_explicit_but_post_create_capacity_is_not(tmp_path):
    from bluesec1_agent.run_lock import RunLock
    from bluesec1_client.errors import RunCapacityExceededError

    async def exercise():
        with RunLock(tmp_path) as lock:
            client = RemoteBenchmarkClient(
                transport=LifecycleTransport(create_error=RemoteAuthenticationError("auth")),
                lifecycle_observer=lock.observe,
            )
            with pytest.raises(RemoteAuthenticationError):
                await client.open()
            assert client.run_creation_rejected is True
        with RunLock(tmp_path) as lock:

            def observe(event, payload):
                lock.observe(event, payload)
                if event == "create_finished":
                    raise RunCapacityExceededError(
                        "observer error", reason="SUBJECT_ACTIVE_RUN_LIMIT_REACHED"
                    )

            client = RemoteBenchmarkClient(
                transport=LifecycleTransport(close_state="closing"), lifecycle_observer=observe
            )
            with pytest.raises(RunCapacityExceededError):
                await client.open()
            assert client.run_creation_rejected is False
            assert client.run.run_id == "run-1"
            assert client.cleanup_outcome.status == "closing_acknowledged"
        assert json.loads((tmp_path / "intent.json").read_text())["run_id"] == "run-1"

    asyncio.run(exercise())
