"""Validate complete synthetic fixture graphs and replay contracts without a provider."""

import argparse
import asyncio
import json
import subprocess
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from bluesec1_agent.evaluation import PartialReplay, read_events, sha256_file, summarize_run
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
    Behavioral mode injects declared faults at the transport boundary while the
    real agent/SDK implement provenance, caching, budgets and lifecycle.
    """

    def __init__(self, scenario: dict, *, behavioral: bool = False) -> None:
        self.scenario = scenario
        self.entities = {e["id"]: e for e in scenario["graph"]["entities"]}
        self.relations = {r["id"]: r for r in scenario["graph"]["relations"]}
        self.behavioral = behavioral
        self.calls = []
        self.abort_calls = 0
        self.close_calls = 0
        self.read_errors = scenario["faults"].get("tool_error_count", 0) if behavioral else 0

    async def create_run(self, command):
        return RemoteRun(run_id="synthetic-run", state="active")

    async def lease_task(self, command):
        observation = deepcopy(self.scenario["observation"])
        catalog = [
            {"name": name, "parameters": {"type": "object"}}
            for name in (
                "get_entity",
                "get_relation",
                "search",
                "get_ad_object_info",
                "finish_investigation",
            )
        ]
        required = self.scenario["faults"].get("unsupported_required_field")
        if self.behavioral and required:
            catalog[0]["parameters"]["required"] = [required]
        observation["available_tools"] = json.dumps(catalog)
        return LeasedTask(
            run_id=command.run_id,
            task=PublicTask(task_id="synthetic-task", observation=observation),
        )

    async def call_tool(self, command):
        self.calls.append({"tool": command.tool_name, **deepcopy(command.arguments)})
        data = None
        terminal = self.behavioral and command.tool_name == "finish_investigation"
        if terminal:
            if self.scenario["faults"].get("lost_terminal_response"):
                raise TimeoutError("Synthetic loss after remote terminal commit")
            return ToolCallOutcome(
                run_id=command.run_id,
                call_id=command.call_id,
                result=ToolCallResult(
                    task_id=command.task_id,
                    tool_name=command.tool_name,
                    terminated=True,
                    info={"tool_status": "ok"},
                    task_result=TaskResult(task_id=command.task_id, completion_reason="terminated"),
                ),
            )
        if command.tool_name == "get_entity":
            data = self.entities.get(command.arguments.get("entity_id"))
        elif command.tool_name == "get_relation":
            data = self.relations.get(command.arguments.get("relation_id"))
        status = "ok" if data is not None else "error"
        if self.read_errors and command.tool_name == "get_entity" and data is not None:
            self.read_errors -= 1
            data = deepcopy(data)
            data["properties"]["approved"] = True
            status = "error"
        return ToolCallOutcome(
            run_id=command.run_id,
            call_id=command.call_id,
            result=ToolCallResult(
                task_id=command.task_id,
                tool_name=command.tool_name,
                info={
                    "tool_status": status,
                    "tool_result": data if data is not None else {"error": "unsupported_action"},
                },
            ),
        )

    async def abort_task(self, command):
        self.abort_calls += 1
        return AbortTaskOutcome(
            run_id=command.run_id,
            task_result=TaskResult(task_id=command.task_id, completion_reason="aborted"),
        )

    async def close_run(self, command):
        self.close_calls += 1
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
    entities = {e["id"]: e for e in graph["entities"]}
    relations = {r["id"]: r for r in graph["relations"]}
    assert len(entities) == len(graph["entities"]), "duplicate entity IDs"
    assert len(relations) == len(graph["relations"]), "duplicate relation IDs"
    assert not entities.keys() & relations.keys(), "namespace collision"
    for relation in relations.values():
        assert relation["source_id"] in entities and relation["target_id"] in entities
    for entity in entities.values():
        rid = entity["relation_ids"]
        assert set(rid["incoming"]) == {
            r["id"] for r in relations.values() if r["target_id"] == entity["id"]
        }
        assert set(rid["outgoing"]) == {
            r["id"] for r in relations.values() if r["source_id"] == entity["id"]
        }
    assert len(entities) == expected["entity_count"]
    assert len(relations) == expected["relation_count"]
    assert scenario["origin"]["synthetic"] is True
    assert scenario["origin"]["partition"] in ("development", "hidden-control")
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


class ScriptedStepClient:
    """Implements StructuredStepClient without loading a provider or oracle labels."""

    def __init__(self, actions: list[dict]) -> None:
        self.actions = deepcopy(actions)
        self.messages = []

    async def next_step(self, messages):
        from bluesec1_agent.models import NextStep

        self.messages.append(deepcopy(messages))
        if len(self.messages) > len(self.actions):
            raise AssertionError("Unexpected decision beyond the prerecorded script")
        function = self.actions[len(self.messages) - 1]
        return NextStep.model_validate(
            {
                "current_state": "Deterministic offline contract probe.",
                "plan_remaining_steps_brief": ["Inspect the next declared probe."],
                "task_completed": function["tool"] == "finish_investigation",
                "function": function,
            }
        )


def _evidence(messages):
    for message in reversed(messages):
        if message["role"] == "user":
            try:
                value = json.loads(message["content"])
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(value, dict) and "evidence" in value:
                return value["evidence"]
    return {}


async def _run_behavior(scenario: dict, output: Path):
    from bluesec1_agent.agent import SimpleSGRAgent
    from bluesec1_agent.recording import RunRecorder, SafeEventSink

    recorder = RunRecorder(output)
    sink = SafeEventSink(recorder)
    llm = ScriptedStepClient(scenario["scripted_actions"])
    transport = GraphTransport(scenario, behavioral=True)
    limits = scenario["driver_limits"]
    agent = SimpleSGRAgent(
        llm,
        strategy="v2",
        max_tasks=1,
        recorder=sink,
        max_steps=limits["max_steps"],
        max_context_chars=limits["max_context_chars"],
    )
    client = RemoteBenchmarkClient(transport=transport)
    error = None
    result = None
    try:
        async with client:
            result = await agent.run(client)
    except Exception as exc:
        error = type(exc).__name__
        sink.record("run_failed", error_type=error)
    if agent.last_error is not None:
        error = type(agent.last_error).__name__
    payload = (
        result.model_dump(mode="json")
        if result
        else {
            "run_id": "synthetic-run",
            "task_results": [r.model_dump(mode="json") for r in agent.task_results],
        }
    )
    payload["cleanup"] = asdict(client.cleanup_outcome) if client.cleanup_outcome else None
    payload["provenance_verified"] = (
        all(agent.provenance_checks) if agent.provenance_checks else None
    )
    sink.write_result(payload)
    summary = summarize_run(recorder.directory / "trace.jsonl", recorder.directory / "result.json")
    events, _ = read_events(recorder.directory / "trace.jsonl")
    contexts = [_evidence(messages) for messages in llm.messages]
    violation_codes = {
        v["code"]
        for row in events
        for v in row.get("payload", {}).get("violations", [])
        if isinstance(v, dict) and "code" in v
    }
    observed = {
        "accepted_final": summary["accepted_final"],
        "quality_score": summary["quality_score"],
        "remote_call_count": len(transport.calls),
        "abort_calls": transport.abort_calls,
        "close_calls": transport.close_calls,
        "remote_finish_calls": sum(c["tool"] == "finish_investigation" for c in transport.calls),
        "cache_hits": summary["cache_hits"],
        "llm_calls": len(llm.messages),
        "task_result_count": len(summary["task_results"]),
        "error_type": error,
        "cleanup_status": client.cleanup_outcome.status if client.cleanup_outcome else None,
        "provenance_verified": summary["provenance_verified"],
        "violation_codes": sorted(violation_codes),
        "max_evidence_context_chars": max(
            (
                len(json.dumps(c, ensure_ascii=False, indent=2, sort_keys=True, default=str))
                for c in contexts
            ),
            default=0,
        ),
        "known_relations": max(
            (c.get("coverage", {}).get("known_relations", 0) for c in contexts), default=0
        ),
        "failed_results": max(
            (c.get("coverage", {}).get("failed_results", 0) for c in contexts), default=0
        ),
        "context_omissions": any(
            c.get("evidence", {}).get("omitted_count", 0) > 0
            or c.get("frontier", {}).get("omitted_count", 0) > 0
            for c in contexts
        ),
    }
    return observed, transport.calls, contexts, llm.messages, recorder.directory


def evaluate_behavior(path: Path, output: Path) -> dict:
    """Run real A/H/C code, then compare against a separate expectation artifact."""
    scenario = json.loads(path.read_text())
    # The runner receives only the scenario/actions, never the oracle file.
    observed, calls, contexts, messages, artifacts = asyncio.run(_run_behavior(scenario, output))
    expected_path = path.with_name("expected.json")
    expected = json.loads(expected_path.read_text())
    failures = []
    for key, want in expected["checks"].items():
        if key == "violation_codes":
            actual = observed[key]
            okay = set(want) <= set(actual)
        elif key == "forbidden_calls":

            def action_key(call):
                return {k: v for k, v in call.items() if k != "reasoning"}

            actual = [c for c in calls if action_key(c) in [action_key(w) for w in want]]
            okay = not actual
        elif key == "frontier_at_decision":
            index = want["decision_index"]
            actual = (
                contexts[index].get("frontier", {}).get("items", [])
                if len(contexts) > index
                else []
            )
            okay = any(item["object_id"] == want["object_id"] for item in actual)
        elif key == "untrusted_text_stays_user_data":
            text = next(
                e["properties"]["command_line"]
                for e in scenario["graph"]["entities"]
                if "command_line" in e["properties"]
            )
            actual = {
                "seen_in_user": any(
                    text in m["content"]
                    for request in messages
                    for m in request
                    if m["role"] == "user"
                ),
                "seen_in_system": any(
                    text in m["content"]
                    for request in messages
                    for m in request
                    if m["role"] == "system"
                ),
            }
            okay = actual == {"seen_in_user": True, "seen_in_system": False}
        else:
            actual = observed[key]
            okay = actual == want
        if not okay:
            failures.append({"check": key, "expected": want, "actual": actual})
    if observed["max_evidence_context_chars"] > scenario["driver_limits"]["max_context_chars"]:
        failures.append(
            {"check": "bounded_evidence_context", "actual": observed["max_evidence_context_chars"]}
        )
    if observed["close_calls"] != 1:
        failures.append({"check": "exactly_one_owned_close", "actual": observed["close_calls"]})
    return {
        "case_id": scenario["case_id"],
        "status": "failed" if failures else "passed",
        "checked_invariants": list(expected["checks"])
        + ["bounded_evidence_context", "exactly_one_owned_close"],
        "declared_invariants": expected["invariants"],
        "observed": observed,
        "failures": failures,
        "artifacts": str(artifacts.resolve()),
        "scenario_sha256": sha256_file(path),
        "expected_sha256": sha256_file(expected_path),
        "model_accuracy_evaluated": False,
        "official_bluesec_score": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--behavioral",
        action="store_true",
        help="Exercise integrated A/H/C with scripted LLM and injected transport.",
    )
    args = parser.parse_args(argv)
    paths = sorted(args.fixtures.glob("*/scenario.json"))
    cases = [evaluate_fixture(path) for path in paths]
    if len(cases) < 12:
        raise ValueError("at least 12 synthetic fixture cases required")
    if args.behavioral:
        for path, case in zip(paths, cases, strict=True):
            case["behavioral"] = evaluate_behavior(
                path, args.output.parent / "behavioral-artifacts"
            )
            case["status"] = case["behavioral"]["status"]
    root = Path(__file__).resolve().parents[1]
    source_paths = subprocess.check_output(
        ["git", "ls-files", "--", "src", "scripts", "pyproject.toml", "uv.lock"],
        cwd=root,
        text=True,
    ).splitlines()
    source_identity = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_fingerprints": {name: sha256_file(root / name) for name in source_paths},
        "dirty_production": bool(
            subprocess.check_output(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--",
                    "src",
                    "scripts",
                    "pyproject.toml",
                    "uv.lock",
                ],
                cwd=root,
                text=True,
            ).strip()
        ),
    }
    passed = sum(case["status"] == "passed" for case in cases)
    result = {
        "schema_version": 1,
        "evaluation_kind": "synthetic_contracts_not_model_accuracy",
        "cases_passed": passed,
        "cases": cases,
        "source_identity": source_identity,
        "integration_status": "scripted_behavioral_checks_executed"
        if args.behavioral
        else "A/H/C behavioral drivers require --behavioral",
        "official_bluesec_score": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(str(args.output.resolve()))
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
