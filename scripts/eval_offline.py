"""Validate complete synthetic fixture graphs and replay contracts without a provider."""

import argparse
import asyncio
import json
from pathlib import Path

from bluesec1_agent.evaluation import PartialReplay, sha256_file
from bluesec1_client import PublicTask, RemoteBenchmarkClient, TaskResult, ToolCallResult
from bluesec1_client.remote.models import (
    AbortTaskOutcome,
    CloseRunOutcome,
    LeasedTask,
    RemoteRun,
    ToolCallOutcome,
)
from bluesec1_client.remote.transport.base import RemoteTransport


class GraphTransport(RemoteTransport):
    """A full synthetic graph behind the existing SDK transport seam.

    Expected labels/invariants are deliberately not accepted by this constructor.
    Fault metadata is retained for later integrated behavioral tests. This graph
    read driver does not simulate the agent's fault-handling behavior.
    """

    def __init__(self, scenario: dict) -> None:
        self.scenario = scenario
        self.entities = {e["entity_id"]: e for e in scenario["graph"]["entities"]}
        self.relations = {r["relation_id"]: r for r in scenario["graph"]["relations"]}

    async def create_run(self, command):
        return RemoteRun(run_id="synthetic-run", state="active")

    async def lease_task(self, command):
        return LeasedTask(
            run_id=command.run_id,
            task=PublicTask(task_id="synthetic-task", observation=self.scenario["observation"]),
        )

    async def call_tool(self, command):
        data = None
        if command.tool_name == "get_entity":
            data = self.entities.get(command.arguments.get("entity_id"))
        elif command.tool_name == "get_relation":
            data = self.relations.get(command.arguments.get("relation_id"))
        return ToolCallOutcome(
            run_id=command.run_id,
            call_id=command.call_id,
            result=ToolCallResult(
                task_id=command.task_id,
                tool_name=command.tool_name,
                info={
                    "tool_status": "ok" if data is not None else "error",
                    "tool_result": data if data is not None else {"error": "unsupported_action"},
                },
            ),
        )

    async def abort_task(self, command):
        return AbortTaskOutcome(
            run_id=command.run_id,
            task_result=TaskResult(task_id=command.task_id, completion_reason="aborted"),
        )

    async def close_run(self, command):
        return CloseRunOutcome(run_id=command.run_id, state="closed")

    async def aclose(self):
        pass


async def exercise_graph(scenario: dict) -> dict:
    """Read every graph branch through real SDK sessions using only in-memory I/O."""
    transport = GraphTransport(scenario)
    async with RemoteBenchmarkClient(transport=transport) as client:
        session = await client.start_task()
        for object_id, expected in transport.entities.items():
            result = await session.call_tool("get_entity", {"entity_id": object_id})
            assert result.info["tool_result"] == expected
        for object_id, expected in transport.relations.items():
            result = await session.call_tool("get_relation", {"relation_id": object_id})
            assert result.info["tool_result"] == expected
        unsupported = await session.call_tool("get_entity", {"entity_id": "unvisited"})
        assert unsupported.info["tool_result"]["error"] == "unsupported_action"
    return {
        "entities_read": len(transport.entities),
        "relations_read": len(transport.relations),
        "unsupported_action_status": unsupported.info["tool_status"],
    }


def evaluate_fixture(path: Path) -> dict:
    scenario = json.loads(path.read_text())
    expected_path = path.with_name("expected.json")
    expected = json.loads(expected_path.read_text())
    graph = scenario["graph"]
    entities = {e["entity_id"]: e for e in graph["entities"]}
    relations = {r["relation_id"]: r for r in graph["relations"]}
    assert len(entities) == len(graph["entities"]), "duplicate entity IDs"
    assert len(relations) == len(graph["relations"]), "duplicate relation IDs"
    assert not entities.keys() & relations.keys(), "namespace collision"
    for relation in relations.values():
        assert relation["source_id"] in entities and relation["target_id"] in entities
    for entity in entities.values():
        rid = entity["relation_ids"]
        assert set(rid["incoming"]) == {
            r["relation_id"] for r in relations.values() if r["target_id"] == entity["entity_id"]
        }
        assert set(rid["outgoing"]) == {
            r["relation_id"] for r in relations.values() if r["source_id"] == entity["entity_id"]
        }
    assert len(entities) == expected["entity_count"]
    assert len(relations) == expected["relation_count"]
    assert scenario["origin"]["synthetic"] is True
    assert scenario["origin"]["partition"] == "development"
    assert expected["invariants"]
    assert "synthetic_label" not in scenario
    known = next(iter(entities))
    replay = PartialReplay(
        [
            {
                "tool_name": "get_entity",
                "arguments": {"entity_id": known},
                "response": {"status": "ok", "result": entities[known]},
            }
        ]
    )
    assert replay.call("get_entity", {"entity_id": known})["result"] == entities[known]
    assert replay.call("get_entity", {"entity_id": "unvisited"})["error"] == "unsupported_action"
    return {
        "case_id": scenario["case_id"],
        "status": "passed",
        "scenario_sha256": sha256_file(path),
        "expected_sha256": sha256_file(expected_path),
        "entity_count": len(entities),
        "relation_count": len(relations),
        "sdk_graph_replay": asyncio.run(exercise_graph(scenario)),
        "invariants_for_integrated_driver": expected["invariants"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cases = [evaluate_fixture(path) for path in sorted(args.fixtures.glob("*/scenario.json"))]
    if len(cases) < 12:
        raise ValueError("at least 12 synthetic fixture cases required")
    result = {
        "schema_version": 1,
        "evaluation_kind": "synthetic_contracts_not_model_accuracy",
        "cases_passed": len(cases),
        "cases": cases,
        "integration_status": "A/H/C behavioral drivers require integrated revision",
        "official_bluesec_score": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(str(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
