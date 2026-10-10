import asyncio
import json
from types import SimpleNamespace

import pytest

from bluesec1_agent.agent import SimpleSGRAgent
from bluesec1_agent.models import NextStep
from bluesec1_client import PublicTask, TaskResult, ToolCallResult
from bluesec1_client.remote.errors import RemoteOutcomeUnknownError


def step(tool, **arguments):
    return NextStep.model_validate(
        {
            "current_state": "Inspect observed evidence and unresolved relations.",
            "plan_remaining_steps_brief": ["Test the remaining explanation."],
            "task_completed": tool == "finish_investigation",
            "function": {"tool": tool, **arguments},
        }
    )


def benign(entity_id="p1", field="image_path"):
    return step(
        "finish_investigation",
        submission={
            "verdict": "benign",
            "reasoning": "Observed authorized process image.",
            "legitimacy_evidence": [
                {"anchor": "entity", "entity_id": entity_id, "property_fields": [field]}
            ],
        },
    )


class ScriptedLLM:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.messages = []

    async def next_step(self, messages):
        self.messages.append(list(messages))
        return next(self.steps)


class Session:
    def __init__(self):
        self.task_id = "synthetic-task"
        self.terminal_state = "active"
        self.calls = []
        self.abort_count = 0
        self.task = PublicTask(
            task_id=self.task_id,
            observation={
                "alert_text": json.dumps(
                    {"trigger_entities": [{"id": "p1", "type": "windows_process"}]}
                ),
                "available_tools": json.dumps(
                    [
                        {"name": name, "parameters": {"type": "object"}}
                        for name in ["get_entity", "get_relation", "finish_investigation"]
                    ]
                ),
            },
        )

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        terminal = tool_name == "finish_investigation"
        if tool_name == "get_entity":
            data = {
                "id": "p1",
                "type": "windows_process",
                "properties": {"image_path": "C:/Windows/explorer.exe"},
                "relation_ids": {"incoming": ["r1"], "outgoing": []},
            }
        elif tool_name == "get_relation":
            data = {"id": "r1", "type": "runs_process", "source_id": "u1", "target_id": "p1"}
        else:
            data = "accepted"
            self.terminal_state = "terminal"
        return ToolCallResult(
            task_id=self.task_id,
            tool_name=tool_name,
            terminated=terminal,
            info={"tool_status": "ok", "tool_result": data},
            task_result=TaskResult(task_id=self.task_id, completion_reason="terminated")
            if terminal
            else None,
        )

    async def abort_task(self, **kwargs):
        self.abort_count += 1
        return TaskResult(task_id=self.task_id, completion_reason="aborted")


class Client:
    def __init__(self, session):
        self.run = SimpleNamespace(run_id="synthetic-run")
        self.session = session

    async def start_task(self):
        session, self.session = self.session, None
        return session


def test_v2_blocks_fabricated_citation_then_accepts_observed_property():
    session = Session()
    llm = ScriptedLLM(
        [
            step("get_entity", entity_id="p1", reasoning="Read trigger."),
            benign("invented-host"),
            benign(),
        ]
    )
    result = asyncio.run(SimpleSGRAgent(llm, strategy="v2", max_steps=3).run(Client(session)))
    assert result.successful_tasks == 1
    assert [name for name, _ in session.calls] == ["get_entity", "finish_investigation"]
    assert session.calls[-1][1]["submission"]["legitimacy_evidence"][0]["entity_id"] == "p1"
    assert "submission_rejected" in llm.messages[-1][-1]["content"]
    assert all(len(messages) <= 4 for messages in llm.messages)


def test_v2_blocks_relation_id_sent_to_get_entity():
    session = Session()
    llm = ScriptedLLM(
        [
            step("get_entity", entity_id="p1", reasoning="Read trigger."),
            step("get_entity", entity_id="r1", reasoning="Wrong namespace."),
            step("get_relation", relation_id="r1", reasoning="Inspect known relation."),
            benign(),
        ]
    )
    result = asyncio.run(SimpleSGRAgent(llm, strategy="v2", max_steps=4).run(Client(session)))
    assert result.successful_tasks == 1
    assert [name for name, _ in session.calls] == [
        "get_entity",
        "get_relation",
        "finish_investigation",
    ]


def test_deadline_cancels_model_wait_without_an_extra_abort():
    class WaitingLLM:
        async def next_step(self, messages):
            await asyncio.Event().wait()

    session = Session()
    with pytest.raises(TimeoutError):
        asyncio.run(
            SimpleSGRAgent(WaitingLLM(), strategy="v2", task_timeout_seconds=0.01).run(
                Client(session)
            )
        )
    assert session.calls == []
    assert session.abort_count == 0


def test_ambiguous_terminal_call_is_never_followed_by_abort():
    class AmbiguousSession(Session):
        async def call_tool(self, tool_name, arguments):
            if tool_name == "finish_investigation":
                self.terminal_state = "unknown"
                raise RemoteOutcomeUnknownError("lost terminal response")
            return await super().call_tool(tool_name, arguments)

    session = AmbiguousSession()
    llm = ScriptedLLM([step("get_entity", entity_id="p1", reasoning="Inspect"), benign()])
    with pytest.raises(RemoteOutcomeUnknownError):
        asyncio.run(SimpleSGRAgent(llm, strategy="v2").run(Client(session)))
    assert session.abort_count == 0


def test_baseline_audits_provenance_without_changing_legacy_decisions():
    session = Session()
    agent = SimpleSGRAgent(ScriptedLLM([benign("fabricated")]), strategy="baseline")
    result = asyncio.run(agent.run(Client(session)))
    assert result.successful_tasks == 1
    assert agent.provenance_checks == [False]
    assert session.calls[0][0] == "finish_investigation"


def test_terminal_timeout_preserves_unknown_outcome_for_cli():
    class CancelledFinish(Session):
        async def call_tool(self, tool_name, arguments):
            if tool_name == "finish_investigation":
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.terminal_state = "unknown"
                    raise
            return await super().call_tool(tool_name, arguments)

    session = CancelledFinish()
    agent = SimpleSGRAgent(
        ScriptedLLM([step("get_entity", entity_id="p1", reasoning="Inspect"), benign()]),
        strategy="v2",
        task_timeout_seconds=0.02,
    )
    with pytest.raises(TimeoutError):
        asyncio.run(agent.run(Client(session)))
    assert agent.remote_outcome_unknown is True
    assert session.abort_count == 0
