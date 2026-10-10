import asyncio

import pytest
from test_agent_v2 import Client, ScriptedLLM, Session, benign

from bluesec1_agent.deadline import DeadlineSubmissionAgent
from bluesec1_client.remote.errors import RemoteOutcomeUnknownError


def test_deadline_model_finish_keeps_observed_citations():
    session = Session()
    result = asyncio.run(DeadlineSubmissionAgent(ScriptedLLM([benign()])).run(Client(session)))
    assert result.successful_tasks == 1
    assert [name for name, _ in session.calls] == ["get_entity", "finish_investigation"]
    assert session.calls[-1][1]["submission"]["verdict"] == "benign"


@pytest.mark.parametrize("failure", ["timeout", "provenance"])
def test_deadline_fallback_is_explicit_and_has_no_invented_targets(failure):
    class FailedModel:
        async def next_step(self, messages):
            if failure == "timeout":
                raise TimeoutError()
            return benign("invented")

    session = Session()
    result = asyncio.run(DeadlineSubmissionAgent(FailedModel()).run(Client(session)))
    assert result.successful_tasks == 1
    submission = session.calls[-1][1]["submission"]
    assert submission["verdict"] == "malicious"
    assert submission["ir_artifacts"] == []
    assert "investigation incomplete" in submission["reasoning"]


def test_deadline_unknown_finish_is_never_retried_or_aborted():
    class UnknownSession(Session):
        async def call_tool(self, tool_name, arguments):
            if tool_name == "finish_investigation":
                self.calls.append((tool_name, arguments))
                self.terminal_state = "unknown"
                raise RemoteOutcomeUnknownError("unknown")
            return await super().call_tool(tool_name, arguments)

    session = UnknownSession()
    with pytest.raises(RemoteOutcomeUnknownError):
        asyncio.run(DeadlineSubmissionAgent(ScriptedLLM([benign()])).run(Client(session)))
    assert len(session.calls) == 2
    assert session.abort_count == 0


def test_deadline_consumes_forty_synthetic_tasks():
    class Queue(Client):
        def __init__(self):
            super().__init__(None)
            self.sessions = [Session() for _ in range(40)]
            self.remaining = iter(self.sessions)

        async def start_task(self):
            return next(self.remaining, None)

    queue = Queue()
    result = asyncio.run(
        DeadlineSubmissionAgent(
            ScriptedLLM([benign()] * 40),
            max_tasks=40,
        ).run(queue)
    )
    assert result.successful_tasks == 40
    assert all(len(session.calls) == 2 and session.abort_count == 0 for session in queue.sessions)


def test_submission_only_finishes_forty_tasks_without_model_or_graph_reads():
    from bluesec1_agent.deadline import SubmissionOnlyAgent

    class NoModel:
        async def next_step(self, messages):
            raise AssertionError("Submission-only mode must not call a model")

    class Queue(Client):
        def __init__(self):
            super().__init__(None)
            self.sessions = [Session() for _ in range(40)]
            self.remaining = iter(self.sessions)

        async def start_task(self):
            return next(self.remaining, None)

    queue = Queue()
    result = asyncio.run(SubmissionOnlyAgent(NoModel(), max_tasks=40).run(queue))
    assert result.successful_tasks == 40
    assert all(
        [name for name, _ in session.calls] == ["finish_investigation"]
        for session in queue.sessions
    )
    assert all(
        session.calls[0][1]["submission"]["ir_artifacts"] == [] for session in queue.sessions
    )
