import asyncio

import grpc
import pytest
from google.protobuf import any_pb2
from google.rpc import error_details_pb2, status_pb2
from grpc.aio import Metadata
from pydantic import ValidationError

from bluesec1_client import RemoteBenchmarkClient, TaskResult
from bluesec1_client.proto.investigation.runtime.v1 import runtime_pb2
from bluesec1_client.remote.errors import CompetitionClosedError, RemoteRateLimitError
from bluesec1_client.remote.models import (
    CloseRunCommand,
    CloseRunOutcome,
    CreateRunCommand,
    DynamicValueLimits,
    RemoteRun,
    RetryPolicy,
)
from bluesec1_client.remote.transport.grpc import map_grpc_error
from bluesec1_client.remote.transport.mappers import (
    DynamicValueCodec,
    decode_task_result,
    encode_close_run,
    encode_create_run,
)


class FakeTransport:
    """Record lifecycle commands without opening a network connection."""

    def __init__(self) -> None:
        """Initialize command capture lists."""
        self.create_commands: list[CreateRunCommand] = []
        self.close_commands: list[CloseRunCommand] = []

    async def create_run(self, command: CreateRunCommand) -> RemoteRun:
        """Record run creation and return an active run."""
        self.create_commands.append(command)
        return RemoteRun(run_id="run-1", state="active")

    async def close_run(self, command: CloseRunCommand) -> CloseRunOutcome:
        """Record run closure."""
        self.close_commands.append(command)
        return CloseRunOutcome(run_id=command.run_id, state="closed")


def make_rpc_error(code: grpc.StatusCode, *, reason: str) -> grpc.aio.AioRpcError:
    """Build a gRPC failure carrying one stable runtime error reason."""
    packed_reason = any_pb2.Any()
    packed_reason.Pack(error_details_pb2.ErrorInfo(reason=reason, domain="runtime"))
    rich_status = status_pb2.Status(
        code=code.value[0],
        message="server detail",
        details=[packed_reason],
    )
    trailing = Metadata(("grpc-status-details-bin", rich_status.SerializeToString()))
    return grpc.aio.AioRpcError(
        code,
        trailing_metadata=trailing,
        details="server detail",
        debug_error_string="server debug detail",
    )


def test_run_commands_encode_arena_and_close_diagnostics() -> None:
    """Map every new lifecycle field to its protobuf counterpart."""
    codec = DynamicValueCodec(DynamicValueLimits())
    create = encode_create_run(
        CreateRunCommand(idempotency_key="create-1", arena="practice"),
        codec,
    )
    close = encode_close_run(
        CloseRunCommand(
            run_id="run-1",
            idempotency_key="close-1",
            reason="client_error",
            error_message="agent failed",
        )
    )

    assert create.arena == "practice"
    assert close.reason == runtime_pb2.CLOSE_RUN_REASON_CLIENT_ERROR
    assert close.error_message == "agent failed"


def test_task_result_decoder_returns_only_public_scores() -> None:
    """Decode the clean v1 score contract without internal scoring inputs."""
    result = decode_task_result(
        runtime_pb2.TaskResult(
            task_id="task-1",
            completion_reason=runtime_pb2.COMPLETION_REASON_TERMINATED,
            total_reward=42.0,
            steps_taken=8,
            tool_calls=7,
            quality_score=0.9,
            efficiency_score=0.8,
            wall_time_seconds=12.5,
        )
    )

    assert isinstance(result, TaskResult)
    assert result.model_dump() == {
        "task_id": "task-1",
        "completion_reason": "terminated",
        "total_reward": 42.0,
        "steps_taken": 8,
        "tool_calls": 7,
        "quality_score": 0.9,
        "efficiency_score": 0.8,
        "wall_time_seconds": 12.5,
    }

    with pytest.raises(ValidationError):
        TaskResult(
            task_id="task-1",
            completion_reason="terminated",
            max_possible_reward=100.0,
        )


def test_client_propagates_arena_and_context_failure_on_close() -> None:
    """Send arena selection and structured failure details through client lifecycle."""
    transport = FakeTransport()

    async def exercise_client() -> None:
        client = RemoteBenchmarkClient(transport=transport, arena="graded")
        try:
            async with client:
                raise RuntimeError("boom")
        except RuntimeError:
            pass

    asyncio.run(exercise_client())

    assert transport.create_commands[0].arena == "graded"
    assert transport.close_commands[0].reason == "client_error"
    assert transport.close_commands[0].error_message == "RuntimeError: boom"


def test_competition_start_refusals_map_to_dedicated_client_errors() -> None:
    """Distinguish a closed arena from the participant run-start rate limit."""
    closed = map_grpc_error(
        make_rpc_error(
            grpc.StatusCode.FAILED_PRECONDITION,
            reason="COMPETITION_ARENA_CLOSED",
        ),
        retry_policy=RetryPolicy(),
        idempotent=True,
    )
    limited = map_grpc_error(
        make_rpc_error(
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            reason="SUBJECT_RUN_START_RATE_LIMIT_REACHED",
        ),
        retry_policy=RetryPolicy(),
        idempotent=True,
    )

    assert isinstance(closed, CompetitionClosedError)
    assert isinstance(limited, RemoteRateLimitError)
    assert limited.reason == "SUBJECT_RUN_START_RATE_LIMIT_REACHED"


def test_real_sdk_retries_keep_logical_ids_through_client_and_session():
    from bluesec1_client.remote.errors import RemoteOutcomeUnknownError
    from bluesec1_client.remote.transport.grpc import GrpcRemoteTransport

    class Stub:
        def __init__(self):
            self.creates, self.calls, self.closes = [], [], []

        async def CreateRun(self, request, **kwargs):
            self.creates.append(request.idempotency_key)
            if len(self.creates) == 1:
                raise make_rpc_error(grpc.StatusCode.UNAVAILABLE, reason="RUNTIME_UNAVAILABLE")
            return runtime_pb2.Run(run_id="run-1", state=runtime_pb2.RUN_STATE_ACTIVE)

        async def LeaseTask(self, request, **kwargs):
            return runtime_pb2.LeaseTaskResponse(
                leased=runtime_pb2.LeasedTask(
                    run_id="run-1", task=runtime_pb2.PublicTask(task_id="task-1")
                )
            )

        async def CallTool(self, request, **kwargs):
            self.calls.append(request.call_id)
            raise make_rpc_error(grpc.StatusCode.UNAVAILABLE, reason="RUNTIME_UNAVAILABLE")

        async def CloseRun(self, request, **kwargs):
            self.closes.append(request.idempotency_key)
            if len(self.closes) == 1:
                raise make_rpc_error(grpc.StatusCode.UNAVAILABLE, reason="RUNTIME_UNAVAILABLE")
            return runtime_pb2.CloseRunResponse(run_id="run-1", state=runtime_pb2.RUN_STATE_CLOSING)

    async def exercise():
        stub = Stub()
        transport = GrpcRemoteTransport(
            endpoint="unused.invalid:443",
            token="synthetic",
            stub=stub,
            retry_policy=RetryPolicy(initial_backoff_seconds=0, max_backoff_seconds=0),
        )
        client = RemoteBenchmarkClient(transport=transport)
        async with client:
            session = await client.start_task()
            with pytest.raises(RemoteOutcomeUnknownError):
                await session.call_tool("finish_investigation", {})
            with pytest.raises(RemoteOutcomeUnknownError):
                await session.abort_task()
        assert len(stub.creates) == 2 and len(set(stub.creates)) == 1
        assert len(stub.calls) == 3 and len(set(stub.calls)) == 1
        assert len(stub.closes) == 2 and len(set(stub.closes)) == 1
        assert client.cleanup_outcome.status == "closing_acknowledged"

    asyncio.run(exercise())
