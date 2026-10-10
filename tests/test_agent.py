import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from bluesec1_agent.agent import SimpleSGRAgent
from bluesec1_agent.models import (
    FinishInvestigationArtifact,
    FinishInvestigationRequest,
    GetEntityRequest,
    MaliciousFinishInvestigationSubmission,
    NextStep,
)
from bluesec1_client import (
    PublicTask,
    RemoteBenchmarkClient,
    RemoteBenchmarkSession,
    TaskResult,
    ToolCallResult,
)


class FakeStepClient:
    """Deterministic structured steps used by the agent unit test."""

    def __init__(self, steps: list[NextStep]) -> None:
        """Store steps in their expected request order.

        Args:
            steps: Prepared LLM steps.
        """
        self.steps = list(steps)
        self.requests: list[list[dict[str, Any]]] = []

    async def next_step(self, messages: list[dict[str, Any]]) -> NextStep:
        """Record one conversation and return its next prepared step.

        Args:
            messages: Current LLM history.

        Returns:
            Next prepared structured step.
        """
        self.requests.append([dict(message) for message in messages])
        return self.steps.pop(0)


class FakeSession:
    """In-memory remote session used by the structured agent test."""

    def __init__(self) -> None:
        """Create a task with a JSON-encoded remote tool catalog."""
        self.task = PublicTask(
            task_id="task-1",
            observation={
                "alert_text": {"trigger_entities": []},
                "available_tools": json.dumps(
                    [
                        {"name": "get_entity", "parameters": {"type": "object"}},
                        {"name": "finish_investigation", "parameters": {"type": "object"}},
                    ]
                ),
            },
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def task_id(self) -> str:
        """Return the public task identifier.

        Returns:
            Task identifier.
        """
        return self.task.task_id

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolCallResult:
        """Record one call and terminate on the finishing tool.

        Args:
            tool_name: Invoked tool name.
            arguments: Invoked tool arguments.

        Returns:
            Fake server response.
        """
        self.calls.append((tool_name, arguments))
        terminal = tool_name == "finish_investigation"
        task_result = None
        if terminal:
            task_result = TaskResult(
                task_id="task-1",
                completion_reason="terminated",
            )
        return ToolCallResult(
            task_id="task-1",
            tool_name=tool_name,
            reward=1.0,
            terminated=terminal,
            info={
                "tool_status": "ok",
                "tool_result": (
                    [{"entity_id": "process-1", "pid": 42}] if not terminal else "Report accepted"
                ),
            },
            task_result=task_result,
        )

    async def abort_task(self, *, error_message: str | None = None) -> TaskResult:
        """Reject unexpected aborts.

        Args:
            error_message: Agent failure diagnostic.

        Raises:
            AssertionError: Always, because this task must finish normally.
        """
        raise AssertionError(f"Unexpected abort: {error_message}")


class FakeClient:
    """Single-task stand-in for an open RemoteBenchmarkClient."""

    def __init__(self, session: FakeSession) -> None:
        """Store the only session returned before queue exhaustion.

        Args:
            session: Prepared task session.
        """
        self.run = SimpleNamespace(run_id="run-1")
        self.session: FakeSession | None = session

    async def start_task(self) -> RemoteBenchmarkSession | None:
        """Lease the prepared session once.

        Returns:
            Prepared session or `None` after it was leased.
        """
        if self.session is None:
            return None
        session = self.session
        self.session = None
        return cast(RemoteBenchmarkSession, session)


def test_agent_executes_fresh_structured_task_via_remote_session() -> None:
    """Tool evidence should feed the next step before the terminal action."""
    llm = FakeStepClient(
        [
            NextStep(
                current_state="Alert identifies process 42 on host-1.",
                plan_remaining_steps_brief=["Inspect its process ancestry and descendants."],
                task_completed=False,
                function=GetEntityRequest(
                    tool="get_entity",
                    entity_id="entity-1",
                    reasoning="Inspect the alert entity.",
                ),
            ),
            NextStep(
                current_state="The suspicious process is confirmed on host-1.",
                plan_remaining_steps_brief=["Submit the evidence-backed malicious verdict."],
                task_completed=True,
                function=FinishInvestigationRequest(
                    tool="finish_investigation",
                    submission=MaliciousFinishInvestigationSubmission(
                        verdict="malicious",
                        ir_artifacts=[
                            FinishInvestigationArtifact(
                                entity_id="host-1",
                                kind="host_to_isolate",
                            )
                        ],
                        reasoning="The alert process is part of a suspicious process chain.",
                    ),
                ),
            ),
        ]
    )
    session = FakeSession()
    client = FakeClient(session)

    result = asyncio.run(SimpleSGRAgent(llm).run(cast(RemoteBenchmarkClient, client)))

    assert result.run_id == "run-1"
    assert result.successful_tasks == 1
    assert result.failed_tasks == 0
    assert [name for name, _ in session.calls] == [
        "get_entity",
        "finish_investigation",
    ]
    assert session.calls[0][1] == {
        "entity_id": "entity-1",
        "reasoning": "Inspect the alert entity.",
    }
    assert session.calls[1][1] == {
        "submission": {
            "verdict": "malicious",
            "ir_artifacts": [{"entity_id": "host-1", "kind": "host_to_isolate"}],
            "reasoning": "The alert process is part of a suspicious process chain.",
        }
    }
    assert any(
        message.get("role") == "user" and "process-1" in str(message.get("content"))
        for message in llm.requests[1]
    )


def finish_step() -> NextStep:
    return NextStep(
        current_state="A malicious process was observed.",
        plan_remaining_steps_brief=["Submit the supported verdict."],
        task_completed=True,
        function=FinishInvestigationRequest(
            tool="finish_investigation",
            submission=MaliciousFinishInvestigationSubmission(
                verdict="malicious",
                ir_artifacts=[],
                reasoning="The process has a malicious command line.",
            ),
        ),
    )


class QueueClient:
    def __init__(self, sessions: list[FakeSession]) -> None:
        self.run = SimpleNamespace(run_id="run-queue")
        self.sessions = list(sessions)

    async def start_task(self) -> RemoteBenchmarkSession | None:
        return cast(RemoteBenchmarkSession, self.sessions.pop(0)) if self.sessions else None


def test_task_limit_does_not_lease_an_extra_task() -> None:
    client = QueueClient([FakeSession(), FakeSession()])
    llm = FakeStepClient([finish_step()])

    result = asyncio.run(SimpleSGRAgent(llm, max_tasks=1).run(cast(RemoteBenchmarkClient, client)))

    assert len(result.task_results) == 1
    assert len(client.sessions) == 1
    assert result.successful_tasks == 1


def test_truncation_is_a_failure_and_stops_before_next_task() -> None:
    class TruncatedSession(FakeSession):
        async def call_tool(self, tool_name, arguments):
            return ToolCallResult(
                task_id="task-1",
                tool_name=tool_name,
                truncated=True,
                task_result=TaskResult(task_id="task-1", completion_reason="truncated"),
            )

    client = QueueClient([TruncatedSession(), FakeSession()])
    result = asyncio.run(
        SimpleSGRAgent(FakeStepClient([finish_step()])).run(cast(RemoteBenchmarkClient, client))
    )

    assert result.successful_tasks == 0
    assert result.failed_tasks == 1
    assert len(client.sessions) == 1


def test_llm_failure_aborts_current_task_without_consuming_queue() -> None:
    class BrokenLLM:
        async def next_step(self, messages):
            raise RuntimeError("provider unavailable")

    class AbortableSession(FakeSession):
        async def abort_task(self, *, error_message=None):
            return TaskResult(task_id=self.task_id, completion_reason="aborted")

    client = QueueClient([AbortableSession(), FakeSession()])
    result = asyncio.run(SimpleSGRAgent(BrokenLLM()).run(cast(RemoteBenchmarkClient, client)))

    assert result.failed_tasks == 1
    assert len(client.sessions) == 1


def test_repeated_read_reuses_evidence_and_keeps_full_reasoning_history() -> None:
    read = NextStep(
        current_state="Need process ancestry to test the alert hypothesis.",
        plan_remaining_steps_brief=["Read the entity."],
        task_completed=False,
        function=GetEntityRequest(
            tool="get_entity", entity_id="entity-1", reasoning="Inspect ancestry."
        ),
    )
    repeated = read.model_copy(
        update={"function": read.function.model_copy(update={"reasoning": "Check it again."})}
    )
    llm = FakeStepClient([read, repeated, finish_step()])
    session = FakeSession()

    asyncio.run(SimpleSGRAgent(llm).run(cast(RemoteBenchmarkClient, FakeClient(session))))

    assert [name for name, _ in session.calls] == ["get_entity", "finish_investigation"]
    assistant = next(m for m in llm.requests[1] if m["role"] == "assistant")
    assert json.loads(assistant["content"])["current_state"] == read.current_state
    assert "process-1" in llm.requests[2][-1]["content"]
    assert "remaining_steps" in llm.requests[2][-1]["content"]


def test_trace_records_initial_evidence_actions_and_terminal_result(tmp_path: Path) -> None:
    from bluesec1_agent.recording import RunRecorder

    recorder = RunRecorder(tmp_path)
    result = asyncio.run(
        SimpleSGRAgent(FakeStepClient([finish_step()]), recorder=recorder).run(
            cast(RemoteBenchmarkClient, FakeClient(FakeSession()))
        )
    )

    events = [
        json.loads(line) for line in (recorder.directory / "trace.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == [
        "task_started",
        "decision_phase",
        "step_selected",
        "tool_call_started",
        "tool_result",
        "tool_call_finished",
        "provenance_validated",
        "task_finished",
    ]
    assert events[0]["payload"]["observation"]["alert_text"] == {"trigger_entities": []}
    assert events[-1]["payload"]["result"]["completion_reason"] == "terminated"
    assert result.successful_tasks == 1


def test_terminal_result_survives_a_trace_write_failure() -> None:
    class BrokenRecorder:
        def record(self, event, **payload):
            if event == "tool_result":
                raise OSError("disk full")

    result = asyncio.run(
        SimpleSGRAgent(FakeStepClient([finish_step()]), recorder=BrokenRecorder()).run(
            cast(RemoteBenchmarkClient, FakeClient(FakeSession()))
        )
    )

    assert result.successful_tasks == 1
    assert result.task_results[0].completion_reason == "terminated"


def test_incompatible_catalog_stops_before_llm_and_remote_action() -> None:
    class IncompatibleSession(FakeSession):
        async def abort_task(self, *, error_message=None):
            return TaskResult(task_id=self.task_id, completion_reason="aborted")

    session = IncompatibleSession()
    session.task.observation["available_tools"] = json.dumps(
        [
            {
                "name": "finish_investigation",
                "parameters": {"type": "object", "required": ["submission", "unknown_required"]},
            }
        ]
    )
    llm = FakeStepClient([finish_step()])
    result = asyncio.run(SimpleSGRAgent(llm).run(cast(RemoteBenchmarkClient, FakeClient(session))))
    assert result.failed_tasks == 1
    assert llm.requests == []
    assert session.calls == []
