import asyncio

import pytest

from bluesec1_client import RemoteBenchmarkClient
from bluesec1_client.errors import TaskAlreadyFinishedError
from bluesec1_client.protocols import PublicTask, TaskResult, ToolCallResult
from bluesec1_client.remote.errors import RemoteRequestError, RemoteTransportError
from bluesec1_client.remote.models import (
    AbortTaskOutcome,
    CloseRunOutcome,
    LeasedTask,
    RemoteRun,
    ToolCallOutcome,
)


class LifecycleTransport:
    def __init__(
        self, *, finish_error=None, close_state="closed", close_error=None, create_error=None
    ):
        self.finish_error = finish_error
        self.close_state = close_state
        self.close_error = close_error
        self.create_error = create_error
        self.creates = []
        self.calls = []
        self.aborts = []
        self.closes = []

    async def create_run(self, command):
        self.creates.append(command)
        if self.create_error:
            raise self.create_error
        return RemoteRun(run_id="run-1", state="active")

    async def lease_task(self, command):
        return LeasedTask(run_id="run-1", task=PublicTask(task_id="task-1"))

    async def call_tool(self, command):
        self.calls.append(command)
        if self.finish_error:
            raise self.finish_error
        return ToolCallOutcome(
            run_id="run-1",
            call_id=command.call_id,
            result=ToolCallResult(
                task_id="task-1",
                tool_name=command.tool_name,
                terminated=True,
                task_result=TaskResult(task_id="task-1", completion_reason="terminated"),
            ),
        )

    async def abort_task(self, command):
        self.aborts.append(command)
        return AbortTaskOutcome(
            run_id="run-1", task_result=TaskResult(task_id="task-1", completion_reason="aborted")
        )

    async def close_run(self, command):
        self.closes.append(command)
        if self.close_error:
            raise self.close_error
        return CloseRunOutcome(run_id="run-1", state=self.close_state)


@pytest.mark.parametrize(
    "state,status", [("closed", "closed_confirmed"), ("closing", "closing_acknowledged")]
)
def test_client_preserves_authoritative_close_outcome(state, status):
    async def exercise():
        transport = LifecycleTransport(close_state=state)
        client = RemoteBenchmarkClient(transport=transport)
        async with client:
            pass
        assert client.cleanup_outcome.status == status
        assert (await client.aclose()) == client.cleanup_outcome
        assert len(transport.closes) == 1

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "error,status",
    [(RemoteTransportError("lost"), "unknown"), (RemoteRequestError("denied"), "failed")],
)
def test_close_errors_have_truthful_outcomes(error, status):
    async def exercise():
        client = RemoteBenchmarkClient(transport=LifecycleTransport(close_error=error))
        await client.open()
        with pytest.raises(type(error)):
            await client.aclose()
        assert client.cleanup_outcome.status == status

    asyncio.run(exercise())


def test_lost_finish_prevents_new_finish_or_abort():
    async def exercise():
        transport = LifecycleTransport(finish_error=RemoteTransportError("lost response"))
        client = RemoteBenchmarkClient(transport=transport)
        await client.open()
        session = await client.start_task()
        try:
            await session.call_tool("finish_investigation", {})
        except Exception as exc:
            assert type(exc).__name__ == "RemoteOutcomeUnknownError"
        else:
            pytest.fail("finish must signal ambiguous outcome")
        assert session.terminal_state == "unknown"
        for operation in (session.call_tool("finish_investigation", {}), session.abort_task()):
            with pytest.raises(Exception) as blocked:
                await operation
            assert type(blocked.value).__name__ == "RemoteOutcomeUnknownError"
        assert len(transport.calls) == 1
        assert transport.aborts == []
        await client.aclose()

    asyncio.run(exercise())


def test_definite_finish_rejection_keeps_session_active():
    async def exercise():
        transport = LifecycleTransport(finish_error=RemoteRequestError("rejected"))
        client = RemoteBenchmarkClient(transport=transport)
        await client.open()
        session = await client.start_task()
        with pytest.raises(RemoteRequestError):
            await session.call_tool("finish_investigation", {})
        assert session.terminal_state == "active"
        await session.abort_task()
        assert len(transport.aborts) == 1

    asyncio.run(exercise())


def test_terminal_result_survives_recorder_failure_without_abort(tmp_path):
    from bluesec1_agent.recording import RunRecorder, SafeEventSink

    async def exercise():
        transport = LifecycleTransport()
        client = RemoteBenchmarkClient(transport=transport)
        await client.open()
        session = await client.start_task()
        result = await session.call_tool("finish_investigation", {})
        recorder = RunRecorder(tmp_path)
        (recorder.directory / "trace.jsonl").mkdir()
        SafeEventSink(recorder).record("tool_call_finished", result=result.model_dump())
        assert session.terminal_state == "terminal"
        assert session.terminal_result == result.task_result
        with pytest.raises(TaskAlreadyFinishedError):
            await session.abort_task()
        assert transport.aborts == []
        await client.aclose()

    asyncio.run(exercise())


def test_actual_cancellation_marks_finish_unknown_and_context_cleans_up():
    async def exercise():
        entered = asyncio.Event()

        class StalledTransport(LifecycleTransport):
            async def call_tool(self, command):
                self.calls.append(command)
                entered.set()
                await asyncio.Event().wait()

        transport = StalledTransport()
        client = RemoteBenchmarkClient(transport=transport)
        sessions = []

        async def operation():
            async with client:
                session = await client.start_task()
                sessions.append(session)
                await session.call_tool("finish_investigation", {})

        task = asyncio.create_task(operation())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sessions[0].terminal_state == "unknown"
        assert transport.aborts == []
        assert len(transport.closes) == 1
        assert client.cleanup_outcome.status == "closed_confirmed"

    asyncio.run(exercise())


def test_cleanup_timeout_preserves_original_exception_and_unknown():
    async def exercise():
        class StalledTransport(LifecycleTransport):
            async def close_run(self, command):
                self.closes.append(command)
                await asyncio.Event().wait()

        transport = StalledTransport()
        client = RemoteBenchmarkClient(transport=transport, cleanup_timeout_seconds=0.01)
        original = RuntimeError("original synthetic-secret")
        with pytest.raises(RuntimeError) as error:
            async with client:
                raise original
        assert error.value is original
        assert client.cleanup_outcome.status == "unknown"
        assert len(transport.closes) == 1

    asyncio.run(exercise())


def test_durable_intent_failure_closes_owned_transport_without_create(monkeypatch):
    import bluesec1_client.remote.client as module

    class OwnedTransport(LifecycleTransport):
        def __init__(self):
            super().__init__()
            self.closed = False

        async def aclose(self):
            self.closed = True

    transport = OwnedTransport()
    monkeypatch.setattr(module, "GrpcRemoteTransport", lambda **kwargs: transport)
    original = OSError("intent disk failure synthetic-secret")

    def observer(event, payload):
        if event == "create_started":
            raise original

    async def exercise():
        client = RemoteBenchmarkClient(
            endpoint="unused.invalid:443", token="synthetic", lifecycle_observer=observer
        )
        with pytest.raises(OSError) as error:
            await client.open()
        assert error.value is original
        assert transport.creates == []
        assert transport.closed

    asyncio.run(exercise())


def test_run_id_recording_failure_closes_received_run_and_preserves_original():
    original = OSError("disk failure synthetic-secret")

    def observer(event, payload):
        if event == "create_finished":
            raise original

    async def exercise():
        transport = LifecycleTransport()
        client = RemoteBenchmarkClient(transport=transport, lifecycle_observer=observer)
        with pytest.raises(OSError) as error:
            await client.open()
        assert error.value is original
        assert client.run.run_id == "run-1"
        assert client.cleanup_outcome.status == "closed_confirmed"
        assert len(transport.closes) == 1

    asyncio.run(exercise())


def test_confirmed_server_close_survives_owned_channel_error(monkeypatch):
    import bluesec1_client.remote.client as module

    class OwnedTransport(LifecycleTransport):
        async def aclose(self):
            raise OSError("channel close failed synthetic-secret")

    transport = OwnedTransport()
    monkeypatch.setattr(module, "GrpcRemoteTransport", lambda **kwargs: transport)

    async def exercise():
        client = RemoteBenchmarkClient(endpoint="unused.invalid:443", token="synthetic")
        await client.open()
        with pytest.raises(OSError):
            await client.aclose()
        assert client.cleanup_outcome.status == "closed_confirmed"
        await client.aclose()
        assert len(transport.closes) == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("terminal_flag", ["terminated", "truncated"])
def test_terminal_read_without_authoritative_result_blocks_new_mutations(terminal_flag):
    from bluesec1_client.remote.errors import RemoteOutcomeUnknownError

    class MalformedReadTransport(LifecycleTransport):
        async def call_tool(self, command):
            self.calls.append(command)
            return ToolCallOutcome(
                run_id="run-1",
                call_id=command.call_id,
                result=ToolCallResult(
                    task_id="task-1", tool_name=command.tool_name, **{terminal_flag: True}
                ),
            )

    async def exercise():
        transport = MalformedReadTransport()
        client = RemoteBenchmarkClient(transport=transport)
        async with client:
            session = await client.start_task()
            with pytest.raises(RemoteOutcomeUnknownError):
                await session.call_tool("get_entity", {"entity_id": "entity-1"})
            assert session.terminal_state == "unknown"
            assert session.terminal_result is None
            with pytest.raises(RemoteOutcomeUnknownError):
                await session.abort_task()
            with pytest.raises(RemoteOutcomeUnknownError):
                await session.call_tool("finish_investigation", {})
        assert len(transport.calls) == 1
        assert transport.aborts == []

    asyncio.run(exercise())


def test_sdk_rejected_terminal_read_keeps_session_unknown():
    from types import SimpleNamespace

    from bluesec1_client.proto.investigation.runtime.v1 import runtime_pb2
    from bluesec1_client.remote.errors import RemoteOutcomeUnknownError
    from bluesec1_client.remote.transport.grpc import GrpcRemoteTransport

    async def malformed_read(request, **kwargs):
        return runtime_pb2.ToolCallResponse(
            run_id=request.run_id,
            task_id=request.task_id,
            call_id=request.call_id,
            tool_name=request.tool_name,
            truncated=True,
        )

    async def exercise():
        grpc_transport = GrpcRemoteTransport(
            endpoint="unused.invalid",
            token="synthetic",
            stub=SimpleNamespace(CallTool=malformed_read),
        )

        class Transport(LifecycleTransport):
            async def call_tool(self, command):
                self.calls.append(command)
                return await grpc_transport.call_tool(command)

        transport = Transport()
        client = RemoteBenchmarkClient(transport=transport)
        async with client:
            session = await client.start_task()
            with pytest.raises(RemoteOutcomeUnknownError):
                await session.call_tool("get_entity", {"entity_id": "entity-1"})
            assert session.terminal_state == "unknown"
            with pytest.raises(RemoteOutcomeUnknownError):
                await session.abort_task()
        assert transport.aborts == []
        assert len(transport.calls) == 1

    asyncio.run(exercise())


def test_run_capacity_on_finish_is_not_reclassified_as_definite_nonacceptance():
    from bluesec1_client.errors import RunCapacityExceededError
    from bluesec1_client.remote.errors import RemoteOutcomeUnknownError

    async def exercise():
        transport = LifecycleTransport(
            finish_error=RunCapacityExceededError(
                "capacity", reason="SUBJECT_ACTIVE_RUN_LIMIT_REACHED"
            )
        )
        client = RemoteBenchmarkClient(transport=transport)
        async with client:
            session = await client.start_task()
            with pytest.raises(RemoteOutcomeUnknownError):
                await session.call_tool("finish_investigation", {})
            with pytest.raises(RemoteOutcomeUnknownError):
                await session.abort_task()
            assert session.terminal_state == "unknown"
        assert len(transport.calls) == 1
        assert transport.aborts == []

    asyncio.run(exercise())
