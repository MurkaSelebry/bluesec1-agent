"""Voluntary completion must check a bounded, relevant response scope."""

import asyncio
import json
from copy import deepcopy

import pytest
from test_agent_v2 import Client, ScriptedLLM, Session, step

from bluesec1_agent import agent as agent_module
from bluesec1_agent.agent import SimpleSGRAgent
from bluesec1_agent.budget import Budget
from bluesec1_client import TaskResult, ToolCallResult


def entity(object_id, *, kind="windows_process", incoming=(), outgoing=()):
    return {
        "id": object_id,
        "type": kind,
        "properties": {"approved": True},
        "relation_ids": {"incoming": list(incoming), "outgoing": list(outgoing)},
    }


def relation(object_id, source="p1", target="p1"):
    return {"id": object_id, "type": "parent_of", "source_id": source, "target_id": target}


def finish(verdict="malicious"):
    evidence = (
        {"ir_artifacts": [{"entity_id": "p1", "kind": "process_observed"}]}
        if verdict == "malicious"
        else {
            "legitimacy_evidence": [
                {
                    "anchor": "entity",
                    "entity_id": "p1",
                    "property_fields": ["approved"],
                }
            ]
        }
    )
    return step(
        "finish_investigation",
        submission={
            "verdict": verdict,
            "reasoning": "Synthetic supported conclusion.",
            **evidence,
        },
    )


class GraphSession(Session):
    def __init__(self, entities, relations):
        super().__init__()
        self.entities = {item["id"]: item for item in entities}
        self.relations = {item["id"]: item for item in relations}

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, deepcopy(arguments)))
        terminal = tool_name == "finish_investigation"
        if terminal:
            data = "accepted"
            self.terminal_state = "terminal"
        elif tool_name == "get_entity":
            data = self.entities[arguments["entity_id"]]
        else:
            data = self.relations[arguments["relation_id"]]
        return ToolCallResult(
            task_id=self.task_id,
            tool_name=tool_name,
            terminated=terminal,
            info={"tool_status": "ok", "tool_result": deepcopy(data)},
            task_result=TaskResult(task_id=self.task_id, completion_reason="terminated")
            if terminal
            else None,
        )


class Events:
    def __init__(self):
        self.items = []

    def record(self, event, **payload):
        self.items.append((event, payload))


def run_scope(session, *, max_steps=20, strategy="v2", verdict="malicious", recorder=None):
    llm = ScriptedLLM(
        [
            step("get_entity", entity_id="p1", reasoning="Read alert trigger"),
            *[finish(verdict)] * 20,
        ]
    )
    agent = SimpleSGRAgent(llm, strategy=strategy, max_steps=max_steps, recorder=recorder)
    result = asyncio.run(agent.run(Client(session)))
    assert result.successful_tasks == 1
    return llm


@pytest.mark.parametrize("verdict", ["malicious", "benign"])
def test_supported_voluntary_finish_reads_trigger_and_active_neighbor_links(verdict):
    session = GraphSession(
        [
            entity("p1", outgoing=["r1"]),
            entity("p2", incoming=["r1"], outgoing=["r2"]),
        ],
        [relation("r1", target="p2"), relation("r2", source="p2")],
    )
    events = Events()
    llm = run_scope(session, verdict=verdict, recorder=events)
    assert [
        (name, args.get("entity_id") or args.get("relation_id")) for name, args in session.calls
    ] == [
        ("get_entity", "p1"),
        ("get_relation", "r1"),
        ("get_entity", "p2"),
        ("get_relation", "r2"),
        ("finish_investigation", None),
    ]
    deferrals = [payload for event, payload in events.items if event == "finish_deferred"]
    assert len(deferrals) == 3
    assert all(payload["proposed_decision"]["task_completed"] is True for payload in deferrals)
    assert all(payload["executed_decision"]["task_completed"] is False for payload in deferrals)
    assert all(
        payload["executed_decision"]["function"]["tool"] != "finish_investigation"
        for payload in deferrals
    )
    feedback = json.loads(llm.messages[2][-1]["content"])["feedback"]
    assert feedback["executed_action"]["reason"] == "response_scope_check"


def test_scope_check_does_not_expand_broad_host_neighbor_relations():
    session = GraphSession(
        [
            entity("p1", outgoing=["r1"]),
            entity("host", kind="host", incoming=["r1"], outgoing=["unrelated"]),
        ],
        [relation("r1", target="host"), relation("unrelated", source="host")],
    )
    run_scope(session)
    assert [args.get("relation_id") for name, args in session.calls if name == "get_relation"] == [
        "r1"
    ]
    assert len(session.calls) == 4  # trigger, relevant relation, host detail, final; no host fanout


def test_scope_check_has_six_read_cap_even_with_many_direct_links():
    ids = [f"r{index:02}" for index in range(20)]
    session = GraphSession([entity("p1", outgoing=ids)], [relation(rid) for rid in ids])
    llm = run_scope(session)
    assert [args["relation_id"] for name, args in session.calls if name == "get_relation"] == ids[
        :6
    ]
    assert len(llm.messages) == 8
    assert session.calls[-1][0] == "finish_investigation"


@pytest.mark.parametrize("cause", ["steps", "time"])
def test_forced_finalization_bypasses_unread_scope(cause, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        agent_module, "Budget", lambda seconds: Budget(seconds, clock=lambda: now[0])
    )

    class SlowSession(GraphSession):
        async def call_tool(self, tool_name, arguments):
            response = await super().call_tool(tool_name, arguments)
            if cause == "time" and tool_name == "get_entity":
                now[0] = 650
            return response

    session = SlowSession([entity("p1", outgoing=["r1"])], [relation("r1")])
    llm = run_scope(session, max_steps=2 if cause == "steps" else 20)
    assert [name for name, _ in session.calls] == ["get_entity", "finish_investigation"]
    assert json.loads(llm.messages[-1][-1]["content"])["phase"] == "finalization"


def test_scope_reads_cannot_consume_reserved_decision_slots():
    ids = [f"r{index:02}" for index in range(20)]
    session = GraphSession([entity("p1", outgoing=ids)], [relation(rid) for rid in ids])
    llm = run_scope(session, max_steps=6)
    assert [args["relation_id"] for name, args in session.calls if name == "get_relation"] == ids[
        :2
    ]
    assert len(llm.messages) == 4
    assert json.loads(llm.messages[-1][-1]["content"])["phase"] == "finalization"


@pytest.mark.parametrize("strategy", ["baseline", "v2"])
def test_finish_without_relevant_frontier_is_accepted(strategy):
    session = GraphSession([entity("p1")], [])
    run_scope(session, strategy=strategy)
    assert [name for name, _ in session.calls] == ["get_entity", "finish_investigation"]


def test_baseline_keeps_voluntary_finish_behavior():
    session = GraphSession([entity("p1", outgoing=["r1"])], [relation("r1")])
    run_scope(session, strategy="baseline")
    assert [name for name, _ in session.calls] == ["get_entity", "finish_investigation"]


def test_invalid_final_is_rejected_before_scope_check_can_redirect_it():
    session = GraphSession([entity("p1", outgoing=["r1"])], [relation("r1")])
    invalid = step(
        "finish_investigation",
        submission={
            "verdict": "malicious",
            "reasoning": "Synthetic unsupported citation.",
            "ir_artifacts": [{"entity_id": "invented", "kind": "process_observed"}],
        },
    )
    llm = ScriptedLLM(
        [
            step("get_entity", entity_id="p1", reasoning="Read trigger"),
            invalid,
            finish(),
            finish(),
        ]
    )
    events = Events()
    result = asyncio.run(SimpleSGRAgent(llm, strategy="v2", recorder=events).run(Client(session)))
    assert result.successful_tasks == 1
    assert [payload["step"] for event, payload in events.items if event == "finish_deferred"] == [3]
    assert json.loads(llm.messages[2][-1]["content"])["feedback"]["reason"] == "submission_rejected"
    assert [name for name, _ in session.calls] == [
        "get_entity",
        "get_relation",
        "finish_investigation",
    ]


def test_failed_scope_read_is_not_repeated_by_later_finish_proposals():
    class FailedRelation(GraphSession):
        async def call_tool(self, tool_name, arguments):
            response = await super().call_tool(tool_name, arguments)
            if tool_name == "get_relation":
                response.info["tool_status"] = "error"
            return response

    session = FailedRelation([entity("p1", outgoing=["r1"])], [relation("r1")])
    run_scope(session)
    assert [name for name, _ in session.calls] == [
        "get_entity",
        "get_relation",
        "finish_investigation",
    ]
