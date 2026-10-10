import asyncio
import json
import time

import pytest
from test_agent_v2 import Client, ScriptedLLM, Session, benign

from bluesec1_agent import cli
from bluesec1_agent.budget import BudgetExceededError
from bluesec1_agent.evidence import EvidenceStore
from bluesec1_client.remote.errors import RemoteOutcomeUnknownError


def quality(llm, **kwargs):
    from bluesec1_agent.quality import QualitySubmissionAgent

    return QualitySubmissionAgent(llm, deadline_epoch=time.time() + 1800, max_tasks=40, **kwargs)


def test_quality_cli_requires_explicit_deadline_and_bounded_queue():
    args = cli.build_parser().parse_args(
        ["--quality-submit", "--quality-deadline", "2026-10-10T12:55:00Z", "--max-tasks", "40"]
    )
    assert args.quality_submit
    assert args.max_tasks == 40


def test_quality_reads_graph_before_final_and_repairs_unobserved_citation():
    session = Session()
    llm = ScriptedLLM([benign("unobserved"), benign()])
    result = asyncio.run(quality(llm).run(Client(session)))
    assert result.successful_tasks == 1
    assert (
        "get_relation",
        {"relation_id": "r1", "reasoning": "Read bounded graph evidence."},
    ) in session.calls
    assert session.calls[-1][1]["submission"]["verdict"] == "benign"
    assert len(llm.messages) == 2
    assert "unknown_object" in json.dumps(llm.messages[-1])
    assert sum(name == "finish_investigation" for name, _ in session.calls) == 1


@pytest.mark.parametrize("process_id,host_id", [("z", "a"), ("a", "z")])
@pytest.mark.parametrize("reverse", [False, True])
def test_process_priority_survives_a_later_empty_reference(process_id, host_id, reverse):
    from bluesec1_agent.quality import quality_frontier

    store = EvidenceStore()
    triggers = [
        {"id": process_id, "type": "windows_process", "host_id": host_id},
        {"id": host_id, "type": "host"},
    ]
    store.seed_observation(
        {
            "alert_text": {
                "trigger_entities": triggers[::-1] if reverse else triggers,
                "trigger_relations": [
                    {"id": "edge", "type": "hosts", "source_id": host_id, "target_id": process_id}
                ],
            }
        }
    )
    assert store.get(process_id, "entity").fields == {}
    assert quality_frontier(store, set())[0].object_id == process_id


def test_quality_reads_neighbor_details_before_exhausting_large_relation_fanout():
    from bluesec1_agent.quality import quality_frontier

    store = EvidenceStore()
    store.seed_observation(
        {"alert_text": {"trigger_entities": [{"id": "process", "type": "windows_process"}]}}
    )
    store.ingest(
        step=1,
        tool_name="get_entity",
        arguments={"entity_id": "process"},
        status="ok",
        result={
            "id": "process",
            "type": "windows_process",
            "properties": {},
            "relation_ids": {"outgoing": ["e1", "e2", "e3"]},
        },
    )
    for index in (1, 2):
        store.ingest(
            step=index + 1,
            tool_name="get_relation",
            arguments={"relation_id": f"e{index}"},
            status="ok",
            result={
                "id": f"e{index}",
                "type": "wrote_file",
                "source_id": "process",
                "target_id": f"file{index}",
            },
        )
    assert quality_frontier(store, set())[0].kind == "entity"


@pytest.mark.parametrize(
    "argv",
    [
        ["--quality-submit"],
        ["--quality-submit", "--quality-deadline", "2000-01-01T00:00:00Z"],
        ["--quality-submit", "--quality-deadline", "2099-01-01T00:00:00Z", "--all-tasks"],
    ],
)
def test_invalid_quality_controls_fail_before_provider_or_runtime(monkeypatch, argv):
    from test_cli import _settings

    monkeypatch.setattr(cli, "_load_settings", _settings)
    monkeypatch.setattr(cli.truststore, "inject_into_ssl", lambda: None)

    def forbidden(**kwargs):
        pytest.fail("No provider/runtime allocation is allowed for invalid controls")

    monkeypatch.setattr(cli, "OpenAIStructuredStepClient", forbidden)
    monkeypatch.setattr(cli, "RemoteBenchmarkClient", forbidden)
    with pytest.raises((ValueError, BudgetExceededError)):
        asyncio.run(cli.run(cli.build_parser().parse_args(argv)))


def test_dynamic_budget_includes_lease_remaining_tasks_and_cleanup():
    from bluesec1_agent.quality import allocate_task_seconds

    assert allocate_task_seconds(1800, 40, cleanup_seconds=30, lease_seconds=8) == 36.25
    assert allocate_task_seconds(100, 2, cleanup_seconds=30, lease_seconds=8) == 27
    assert allocate_task_seconds(100, 1, cleanup_seconds=30, lease_seconds=8) == 60
    with pytest.raises(BudgetExceededError):
        allocate_task_seconds(50, 40, cleanup_seconds=30, lease_seconds=8)


def test_quality_unknown_finish_never_retries_or_aborts():
    class Unknown(Session):
        async def call_tool(self, name, arguments):
            if name == "finish_investigation":
                self.calls.append((name, arguments))
                self.terminal_state = "unknown"
                raise RemoteOutcomeUnknownError("uncertain")
            return await super().call_tool(name, arguments)

    session = Unknown()
    with pytest.raises(RemoteOutcomeUnknownError):
        asyncio.run(quality(ScriptedLLM([benign()])).run(Client(session)))
    assert session.abort_count == 0
    assert sum(name == "finish_investigation" for name, _ in session.calls) == 1


@pytest.mark.parametrize("mismatch", [None, "run_id", "task_id", "evidence_sha256"])
def test_advisory_is_bound_to_exact_case_and_validated(tmp_path, mismatch):
    class FailedModel:
        async def next_step(self, messages):
            request = json.loads((tmp_path / "request.json").read_text())
            response = {key: request[key] for key in ("run_id", "task_id", "evidence_sha256")}
            response["decision"] = benign().model_dump(mode="json")
            if mismatch:
                response[mismatch] = "wrong"
            (tmp_path / "response.json").write_text(json.dumps(response))
            raise TimeoutError()

    session = Session()
    result = asyncio.run(quality(FailedModel(), advisory_dir=tmp_path).run(Client(session)))
    assert result.successful_tasks == 1
    submission = session.calls[-1][1]["submission"]
    if mismatch:
        assert "incomplete" in submission["reasoning"]
        assert submission["ir_artifacts"] == []
    else:
        assert submission["verdict"] == "benign"


def test_quality_consumes_forty_tasks_with_one_terminal_call_each():
    class Queue(Client):
        def __init__(self):
            super().__init__(None)
            self.sessions = [Session() for _ in range(40)]
            self.remaining = iter(self.sessions)

        async def start_task(self):
            return next(self.remaining, None)

    queue = Queue()
    result = asyncio.run(quality(ScriptedLLM([benign()] * 40)).run(queue))
    assert result.successful_tasks == 40
    assert all(sum(n == "finish_investigation" for n, _ in s.calls) == 1 for s in queue.sessions)
