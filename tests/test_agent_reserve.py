"""Exercise decision deadlines through the real SDK and agent, without network access."""

import asyncio
import json

import httpx
import pytest
from jsonschema import ValidationError, validate
from openai import AsyncOpenAI
from test_agent_v2 import Client, ScriptedLLM, Session, benign, step

from bluesec1_agent import agent as agent_module
from bluesec1_agent.agent import SimpleSGRAgent
from bluesec1_agent.budget import Budget
from bluesec1_agent.llm import OpenAIStructuredStepClient
from bluesec1_agent.models import NEXT_STEP_RESPONSE_FORMAT


@pytest.mark.parametrize("strategy", ["baseline", "v2"])
def test_elapsed_investigation_enters_finish_only_and_repairs_invalid_final(strategy, monkeypatch):
    requests = []
    request_timeouts = []
    now = [0.0]
    monkeypatch.setattr(
        agent_module, "Budget", lambda seconds: Budget(seconds, clock=lambda: now[0]), raising=False
    )

    class SlowReadSession(Session):
        async def call_tool(self, tool_name, arguments):
            result = await super().call_tool(tool_name, arguments)
            if tool_name == "get_entity":
                now[0] += 215
            return result

    session = SlowReadSession()
    client = None

    def reply(request):
        request_body = json.loads(request.content)
        requests.append(request_body)
        request_timeouts.append(request.extensions["timeout"]["read"])
        if len(requests) == 1:
            proposal = step("get_entity", entity_id="p1", reasoning="Read trigger")
        elif len(requests) == 2:
            proposal = step("get_relation", relation_id="r1", reasoning="Try more reading")
        else:
            proposal = benign()
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": proposal.model_dump_json()},
                    }
                ],
            },
        )

    async def exercise():
        nonlocal client
        async with AsyncOpenAI(
            api_key="synthetic",
            base_url="https://test.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(reply)),
        ) as sdk:
            client = OpenAIStructuredStepClient(
                base_url="https://test.invalid/v1",
                api_key="synthetic",
                model="test",
                timeout_seconds=120,
                max_completion_tokens=1000,
                max_retries=3,
                client=sdk,
            )
            return await SimpleSGRAgent(
                client,
                strategy=strategy,
                max_steps=20,
                task_timeout_seconds=300,
            ).run(Client(session))

    result = asyncio.run(exercise())
    assert result.successful_tasks == 1
    assert [name for name, _ in session.calls] == ["get_entity", "finish_investigation"]
    final_schema = requests[1]["response_format"]["json_schema"]["schema"]
    validate(benign().model_dump(mode="json"), final_schema)
    with pytest.raises(ValidationError):
        validate(step("get_entity", entity_id="p1", reasoning="Repeat").model_dump(), final_schema)
    validate(
        step("get_entity", entity_id="p1", reasoning="Read").model_dump(),
        NEXT_STEP_RESPONSE_FORMAT["json_schema"]["schema"],
    )
    assert request_timeouts[0] == 120
    assert 0 < request_timeouts[1] <= 35  # preserve 15 seconds plus a repair decision
    guidance = json.loads(requests[1]["messages"][-1]["content"])
    assert guidance["phase"] == "finalization"
    assert guidance["remaining_seconds"] == 85
    assert len(requests) == 3  # provider ignoring the final schema is repaired locally


def test_reserved_final_steps_repair_rejected_citation_without_more_navigation():
    session = Session()
    llm = ScriptedLLM(
        [
            step("get_entity", entity_id="p1", reasoning="Read trigger"),
            benign("fabricated"),
            benign(),
        ]
    )
    result = asyncio.run(SimpleSGRAgent(llm, strategy="v2", max_steps=4).run(Client(session)))
    assert result.successful_tasks == 1
    assert [name for name, _ in session.calls] == ["get_entity", "finish_investigation"]
    assert session.abort_count == 0


def test_repeated_cached_reads_recover_to_unseen_relation_and_keep_action_history():
    class SelfLoopSession(Session):
        async def call_tool(self, tool_name, arguments):
            response = await super().call_tool(tool_name, arguments)
            if tool_name == "get_relation":
                response.info["tool_result"]["source_id"] = "p1"
            return response

    session = SelfLoopSession()
    read = step("get_entity", entity_id="p1", reasoning="Read trigger")
    llm = ScriptedLLM([read, read, read, read, benign()])
    result = asyncio.run(SimpleSGRAgent(llm, strategy="v2", max_steps=8).run(Client(session)))
    assert result.successful_tasks == 1
    assert [name for name, _ in session.calls] == [
        "get_entity",
        "get_relation",
        "finish_investigation",
    ]
    context = json.loads(llm.messages[-1][-1]["content"])
    assert context["action_history"]["no_progress_streak"] == 0
    assert any(item["outcome"] == "cached" for item in context["action_history"]["recent"])
    assert context["remaining_seconds"] > 0
    assert context["phase"] == "investigate"


def test_final_repair_is_bounded_and_never_invents_a_verdict():
    session = Session()
    llm = ScriptedLLM(
        [
            step("get_entity", entity_id="p1", reasoning="Inspect"),
            *[benign("fabricated")] * 10,
        ]
    )
    result = asyncio.run(SimpleSGRAgent(llm, strategy="v2", max_steps=4).run(Client(session)))
    assert result.failed_tasks == 1
    assert len(llm.messages) == 4
    assert [name for name, _ in session.calls] == ["get_entity"]
    assert session.abort_count == 1


def test_final_phase_terminal_timeout_marks_real_session_unknown_and_never_aborts():
    from bluesec1_client import RemoteBenchmarkSession
    from bluesec1_client.remote.models import DynamicValueLimits

    class StalledTransport:
        def __init__(self):
            self.calls = []
            self.aborts = []

        async def call_tool(self, command):
            self.calls.append(command)
            await asyncio.Event().wait()

        async def abort_task(self, command):
            self.aborts.append(command)
            raise AssertionError("An ambiguous terminal request cannot be aborted")

    transport = StalledTransport()
    session = RemoteBenchmarkSession(
        run_id="test-run",
        task=Session().task,
        transport=transport,
        is_client_open=lambda: True,
        value_limits=DynamicValueLimits(),
    )
    agent = SimpleSGRAgent(
        ScriptedLLM([benign()]),
        max_steps=1,
        task_timeout_seconds=0.05,
    )
    with pytest.raises(TimeoutError):
        asyncio.run(agent.run(Client(session)))
    assert session.terminal_state == "unknown"
    assert agent.remote_outcome_unknown is True
    assert len(transport.calls) == 1
    assert transport.calls[0].tool_name == "finish_investigation"
    assert transport.aborts == []


def test_slow_model_cannot_start_read_after_investigation_cutoff(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        agent_module, "Budget", lambda seconds: Budget(seconds, clock=lambda: now[0])
    )

    class SlowLLM(ScriptedLLM):
        async def next_step(self, messages):
            answer = await super().next_step(messages)
            if len(self.messages) == 1:
                now[0] = 215.0
            return answer

    session = Session()
    # Unsupported final evidence must still be rejected after time-reserve enforcement.
    llm = SlowLLM(
        [
            step("get_entity", entity_id="p1", reasoning="Late read"),
            benign("fabricated"),
            benign("fabricated"),
            benign("fabricated"),
        ]
    )
    result = asyncio.run(
        SimpleSGRAgent(
            llm,
            strategy="v2",
            max_steps=20,
            task_timeout_seconds=300,
        ).run(Client(session))
    )
    assert result.failed_tasks == 1
    assert session.calls == []
    assert session.abort_count == 1
    assert json.loads(llm.messages[1][-1]["content"])["phase"] == "finalization"


def test_rejected_final_does_not_extend_selected_decision_limit():
    session = Session()
    llm = ScriptedLLM(
        [
            step("get_entity", entity_id="p1", reasoning="Inspect"),
            benign("fabricated"),
            benign(),
        ]
    )
    result = asyncio.run(SimpleSGRAgent(llm, strategy="v2", max_steps=2).run(Client(session)))
    assert result.failed_tasks == 1
    assert len(llm.messages) == 2
    assert [name for name, _ in session.calls] == ["get_entity"]


def test_repeated_failed_lead_is_not_selected_as_no_progress_recovery():
    class FailedSession(Session):
        async def call_tool(self, tool_name, arguments):
            result = await super().call_tool(tool_name, arguments)
            if tool_name == "get_relation":
                return result.model_copy(
                    update={
                        "info": {
                            "tool_status": "error",
                            "tool_result": "unavailable",
                        }
                    }
                )
            return result

    session = FailedSession()
    read_relation = step("get_relation", relation_id="r1", reasoning="Inspect")
    llm = ScriptedLLM(
        [
            step("get_entity", entity_id="p1", reasoning="Inspect"),
            read_relation,
            read_relation,
            read_relation,
            benign(),
        ]
    )
    result = asyncio.run(SimpleSGRAgent(llm, strategy="v2", max_steps=10).run(Client(session)))
    assert result.successful_tasks == 1
    assert [name for name, _ in session.calls].count("get_relation") == 2
    assert json.loads(llm.messages[3][-1]["content"])["phase"] == "finalization"


def test_successful_unsupported_results_do_not_reset_evidence_progress():
    class UnsupportedSession(Session):
        async def call_tool(self, tool_name, arguments):
            result = await super().call_tool(tool_name, arguments)
            if tool_name == "get_entity":
                result.info["tool_result"]["relation_ids"]["incoming"] = ["r1", "r2", "r3"]
            elif tool_name == "get_relation":
                result.info["tool_result"] = "unsupported opaque result"
            return result

    session = UnsupportedSession()
    r2 = step("get_relation", relation_id="r2", reasoning="Inspect")
    llm = ScriptedLLM(
        [
            step("get_entity", entity_id="p1", reasoning="Inspect"),
            step("get_relation", relation_id="r1", reasoning="Inspect"),
            r2,
            r2,
            benign(),
        ]
    )
    result = asyncio.run(SimpleSGRAgent(llm, strategy="v2", max_steps=10).run(Client(session)))
    assert result.successful_tasks == 1
    assert [args["relation_id"] for name, args in session.calls if name == "get_relation"] == [
        "r1",
        "r2",
        "r3",
    ]
    context = json.loads(llm.messages[-1][-1]["content"])
    assert context["action_history"]["no_progress_streak"] == 3
    assert context["evidence"]["coverage"]["unsupported_results"] == 3
