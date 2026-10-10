"""Behavioral checks for offline accounting, never model accuracy tests."""

import importlib.util
import json
import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRAFT_MANIFEST = ROOT / "tests/fixtures/evaluation/draft-manifest.json"


@pytest.fixture(autouse=True)
def deny_mutable_report_access(monkeypatch):
    """Offline contract tests must not depend on or mutate user experiment state."""
    original_open = Path.open

    def isolated_open(path, *args, **kwargs):
        if path.resolve().is_relative_to(ROOT / "reports"):
            raise AssertionError("evaluation tests must use fixtures, not mutable reports")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", isolated_open)


def module():
    from bluesec1_agent import evaluation

    return evaluation


def write_trace(tmp_path, events):
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(json.dumps(e) + "\n" for e in events))
    return trace


def event(name, **payload):
    return {"schema_version": 2, "event": name, "payload": payload}


def test_absent_result_and_started_without_finish_are_unknown(tmp_path):
    trace = write_trace(tmp_path, [event("task_started", task_id="t")])
    summary = module().summarize_run(trace, tmp_path / "missing.json")
    assert summary["accepted_final"] is None
    assert summary["quality_score"] is None
    assert summary["usage"]["total_tokens"] is None
    assert summary["cleanup_confirmed"] is None
    assert summary["status"] == "incomplete"
    assert summary["task_ids"] == ["t"]


@pytest.mark.parametrize("version", [None, 2])
def test_accepted_zero_and_missing_json_score_are_distinct(tmp_path, version):
    terminal = event(
        "task_finished",
        task_id="t",
        result={
            "task_id": "t",
            "completion_reason": "terminated",
            "quality_score": 0,
            "efficiency_score": 0,
            "total_reward": 0,
        },
    )
    if version is None:
        terminal.pop("schema_version")
    trace = write_trace(tmp_path, [terminal])
    summary = module().summarize_run(trace, None)
    assert summary["accepted_final"] is True
    assert summary["quality_score"] == 0
    assert summary["reward"] == 0
    terminal["payload"]["result"].pop("quality_score")
    trace = write_trace(tmp_path, [terminal])
    assert module().summarize_run(trace, None)["quality_score"] is None
    assert "nonoptional" in summary["score_semantics"]


def test_llm_attempts_usage_partial_and_cache_separate_from_remote_calls(tmp_path):
    trace = write_trace(
        tmp_path,
        [
            event("llm_attempt_started", attempt_id="a"),
            event("llm_attempt_finished", attempt_id="a", usage={"total_tokens": 5}),
            event("llm_attempt_started", attempt_id="b"),
            event("cache_hit", tool_name="get_entity"),
            event("tool_result", response={"tool_name": "get_entity"}),
            event("action_rejected", reason="provenance"),
        ],
    )
    summary = module().summarize_run(trace, None)
    assert summary["llm_attempts"] == 2
    assert summary["usage"]["total_tokens"] is None
    assert summary["usage"]["known_total_tokens"] == 5
    assert summary["usage"]["complete"] is False
    assert summary["remote_tool_calls"] == 1
    assert summary["cache_hits"] == 1
    assert summary["provenance_rejections"] == 1


def test_duplicate_attempt_event_counted_once_and_conflict_rejected(tmp_path):
    entry = {"event_id": "e", "event": "reserved", "slot_id": "01"}
    path = write_trace(tmp_path, [entry, entry])
    events, warnings = module().read_events(path)
    assert len(events) == 1
    assert warnings == ["duplicate_event:e"]
    path = write_trace(tmp_path, [entry, {**entry, "slot_id": "02"}])
    with pytest.raises(ValueError, match="conflicting"):
        module().read_events(path)


def test_partial_replay_never_invents_unvisited_branch():
    replay = module().PartialReplay(
        [
            {
                "tool_name": "get_entity",
                "arguments": {"entity_id": "a"},
                "response": {"status": "ok", "result": {"id": "a"}},
            }
        ]
    )
    assert replay.call("get_entity", {"entity_id": "a", "reasoning": "read"})["status"] == "ok"
    assert replay.call("get_entity", {"entity_id": "b"}) == {
        "status": "error",
        "error": "unsupported_action",
        "tool_name": "get_entity",
    }
    assert replay.call("finish_investigation", {})["error"] == "unsupported_action"


def test_reservation_is_durable_before_executor_and_failed_create_consumes_slot(tmp_path):
    journal = module().AttemptJournal(tmp_path / "attempts.jsonl", "abc")
    seen = []

    def execute(slot):
        lines, _ = module().read_events(journal.path)
        seen.append((slot["slot_id"], lines[0]["event"]))
        return {"error_category": "auth", "cleanup_confirmed": True, "run_id": None}

    slots = [{"slot_id": f"{i:02d}"} for i in range(1, 17)]
    module().execute_series(slots, journal, execute)
    assert seen == [("01", "reserved")]
    rows = module().reconcile_attempts(slots, journal.path)["attempts"]
    assert len(rows) == 16
    assert rows[0]["attempted"] is True and rows[0]["run_id"] is None
    assert all(r["status"] == "not_started" and not r["attempted"] for r in rows[1:])
    with pytest.raises(ValueError, match="resume"):
        module().execute_series(slots, journal, execute)


def test_failed_fsync_forbids_executor(tmp_path, monkeypatch):
    journal = module().AttemptJournal(tmp_path / "attempts.jsonl", "abc")

    def fail(_):
        raise OSError("disk failure")

    monkeypatch.setattr(module().os, "fsync", fail)
    calls = []
    with pytest.raises(OSError, match="disk failure"):
        module().execute_series([{"slot_id": "01"}], journal, lambda s: calls.append(s))
    assert calls == []


@pytest.mark.parametrize(
    "outcomes, expected",
    [
        ([{"cleanup_confirmed": None}], 1),
        ([{"cleanup_confirmed": True, "error_category": "schema"}], 1),
        ([{"cleanup_confirmed": True, "error_category": "infrastructure"}] * 2, 2),
    ],
)
def test_stop_rules_prevent_extra_attempts(tmp_path, outcomes, expected):
    journal = module().AttemptJournal(tmp_path / "attempts.jsonl", "abc")
    count = []

    def execute(slot):
        result = outcomes[len(count)]
        count.append(slot)
        return result

    slots = [{"slot_id": f"{i:02d}"} for i in range(1, 17)]
    module().execute_series(slots, journal, execute)
    assert len(count) == expected
    assert module().reconcile_attempts(slots, journal.path)["attempted_count"] == expected


def test_all_sixteen_slots_preserved_and_hard_limit(tmp_path):
    journal = module().AttemptJournal(tmp_path / "attempts.jsonl", "abc")
    slots = [{"slot_id": f"{i:02d}"} for i in range(1, 17)]
    module().execute_series(slots, journal, lambda _: {"cleanup_confirmed": True})
    result = module().reconcile_attempts(slots, journal.path)
    assert result["attempted_count"] == 16
    with pytest.raises(ValueError, match="16"):
        module().execute_series(slots + [{"slot_id": "17"}], journal, lambda _: {})


def test_fixture_contracts_and_scripts_are_offline(tmp_path, monkeypatch, capsys):
    def no_network(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    for name, argv in [
        (
            "eval_offline",
            [
                "--fixtures",
                str(ROOT / "tests/fixtures/evaluation"),
                "--output",
                str(tmp_path / "offline.json"),
            ],
        ),
        (
            "run_experiments",
            ["--manifest", str(DRAFT_MANIFEST), "--dry-run"],
        ),
    ]:
        spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        assert script.main(argv) == 0
    report = json.loads((tmp_path / "offline.json").read_text())
    assert report["cases_passed"] >= 12
    assert report["evaluation_kind"] == "synthetic_contracts_not_model_accuracy"
    assert "dry_run" in capsys.readouterr().out


def test_manifest_gate_rejects_draft_and_wrong_review_sha(tmp_path):
    manifest_path = DRAFT_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    module().validate_manifest(manifest)
    with pytest.raises(ValueError, match="frozen"):
        module().validate_live_gate(manifest, "abc", {}, "1" * 40)
    manifest.update(freeze_state="frozen", integrated_sha="1" * 40)
    with pytest.raises(ValueError, match="gate"):
        module().validate_live_gate(
            manifest,
            "abc",
            {
                "live_ready": True,
                "integrated_sha": "2" * 40,
                "manifest_sha256": "abc",
                "reviewer": "R",
                "review_approved": True,
            },
            "1" * 40,
        )


def test_flat_h_usage_does_not_double_count_subset_tokens(tmp_path):
    trace = write_trace(
        tmp_path,
        [
            event("llm_attempt_started", attempt_id="a"),
            event(
                "llm_attempt_finished",
                attempt_id="a",
                total_tokens=10,
                prompt_tokens=6,
                completion_tokens=4,
                cached_tokens=2,
                reasoning_tokens=3,
                outcome="success",
            ),
            event("cleanup_finished", status="closed_confirmed"),
        ],
    )
    summary = module().summarize_run(trace, None)
    assert summary["usage"]["total_tokens"] == 10
    assert summary["usage"]["complete"] is True
    assert summary["cleanup_confirmed"] is True


def test_v2_rpc_count_includes_lost_response_without_counting_legacy_twice(tmp_path):
    trace = write_trace(
        tmp_path,
        [
            event("tool_call_started", task_id="t", step=1),
            event("tool_call_finished", task_id="t", step=1),
            event("tool_result", task_id="t", step=1, response={"tool_name": "get_entity"}),
            event("tool_call_started", task_id="t", step=2),
        ],
    )
    assert module().summarize_run(trace, None)["remote_tool_calls"] == 2


def test_torn_ledger_preserves_reservation_and_reports_partial_tail(tmp_path):
    journal = module().AttemptJournal(tmp_path / "attempts.jsonl", "abc")
    journal.append("reserved", slot_id="01")
    with journal.path.open("a") as stream:
        stream.write('{"unfinished')
    report = module().reconcile_attempts([{"slot_id": "01"}], journal.path)
    assert report["attempted_count"] == 1
    assert report["attempts"][0]["status"] == "reserved"
    assert report["warnings"][0].startswith("torn_final_record:")


def test_manifest_hash_never_changes_with_journal_updates(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"frozen": true}\n')
    digest = module().sha256_file(path)
    journal = module().AttemptJournal(tmp_path / "attempts.jsonl", digest)
    module().execute_series(
        [{"slot_id": "01"}], journal, lambda _: {"cleanup_confirmed": True, "quality_score": 0}
    )
    assert module().sha256_file(path) == digest
    with pytest.raises(ValueError, match="hash mismatch"):
        module().reconcile_attempts([{"slot_id": "01"}], journal.path, "different")


def test_comparison_preserves_failed_attempt_denominator_and_zero(tmp_path):
    manifest = json.loads(DRAFT_MANIFEST.read_text())
    journal = module().AttemptJournal(tmp_path / "attempts.jsonl", "abc")
    count = []

    def execute(_):
        count.append(1)
        return {
            "cleanup_confirmed": True,
            "accepted_final": len(count) % 2 == 0,
            "quality_score": 0 if len(count) % 2 == 0 else None,
        }

    module().execute_series(manifest["slots"], journal, execute)
    report = module().compare_attempts(
        manifest, module().reconcile_attempts(manifest["slots"], journal.path)
    )
    assert report["attempted_count"] == 16
    assert sum(g["attempts"] for g in report["groups"].values()) == 16
    assert sum(g["incomplete_count"] for g in report["groups"].values()) == 8
    assert report["recommendation"] == "investigate_then_collect_more"
    assert report["changes_default"] is False
    assert report["precision"] is None


def test_incomplete_walltime_cannot_rank_candidate_as_if_measurement_complete():
    manifest = {"configurations": {"v2": {"model": "m", "strategy": "v2"}}}
    rows = [
        {
            "configuration_id": "v2",
            "attempted": True,
            "accepted_final": True,
            "quality_score": 0.5,
            "cleanup_confirmed": True,
            "provenance_verified": True,
            "open_p1_p2": 0,
        }
        for _ in range(3)
    ]
    report = module().compare_attempts(
        manifest,
        {"attempts": rows, "attempted_count": 3, "series_complete": True, "stop_reason": None},
    )
    assert report["recommendation"] == "investigate_then_collect_more"


def test_full_synthetic_graph_driver_uses_real_client_without_network(monkeypatch):
    import asyncio

    def no_network(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    spec = importlib.util.spec_from_file_location(
        "offline_driver", ROOT / "scripts/eval_offline.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    scenario = json.loads(
        (ROOT / "tests/fixtures/evaluation/04-500-relations/scenario.json").read_text()
    )
    report = asyncio.run(driver.exercise_graph(scenario))
    assert report == {
        "entities_read": 501,
        "relations_read": 500,
        "unsupported_action_status": "error",
    }


def test_c_result_error_category_preserved_without_guessing_exception_text(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "status": "failed",
                "error_category": "schema",
                "task_results": [],
                "cleanup": {"status": "closed_confirmed"},
                "artifact_errors": ["trace-write"],
                "metadata": {"strategy": "v2"},
            }
        )
    )
    summary = module().summarize_run(write_trace(tmp_path, []), result)
    assert summary["error_category"] == "schema"
    assert summary["quality_score"] is None
    assert summary["artifact_errors"] == ["trace-write"]
    assert summary["metadata"]["strategy"] == "v2"


def test_recommendation_uses_fixed_policy_tie_and_never_changes_default():
    manifest = {"configurations": {"a": {"model": "a"}, "b": {"model": "b"}}}
    rows = [
        {
            "configuration_id": c,
            "attempted": True,
            "accepted_final": True,
            "quality_score": 0.5,
            "efficiency_score": 0.7,
            "walltime_seconds": 10,
            "cleanup_confirmed": True,
            "provenance_verified": True,
            "open_p1_p2": 0,
        }
        for c in ("a", "b")
        for _ in range(3)
    ]
    data = {"attempts": rows, "attempted_count": 6, "series_complete": True, "stop_reason": None}
    report = module().compare_attempts(manifest, data)
    assert report["recommendation"] == "no_preference"
    assert report["changes_default"] is False
    rows[0]["quality_score"] = 0.8
    rows[1]["quality_score"] = 0.8
    report = module().compare_attempts(manifest, data)
    assert report["candidate_ranking"] == ["a", "b"]
    assert report["recommendation"] == "next_experiment_candidate"
    assert report["changes_default"] is False


@pytest.fixture(autouse=True)
def deny_evaluation_network(monkeypatch):
    import grpc

    def forbidden(*args, **kwargs):
        raise AssertionError("offline evaluation attempted network I/O")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    for target in (grpc, grpc.aio):
        monkeypatch.setattr(target, "secure_channel", forbidden)
        monkeypatch.setattr(target, "insecure_channel", forbidden)
    for key in ("LLM_API_KEY", "SCENARIO_RUNTIME_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def test_all_launchers_use_manifest_canonical_accounting_location(tmp_path):
    manifest = json.loads(DRAFT_MANIFEST.read_text())
    manifest["attempts_path"] = "relative/attempts.jsonl"
    with pytest.raises(ValueError, match="absolute"):
        module().validate_manifest(manifest)


def test_launcher_rejects_unfrozen_manifest_before_process_creation(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("launcher", ROOT / "scripts/run_experiments.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    gate = tmp_path / "gate.json"
    gate.write_text("{}")
    monkeypatch.setattr(launcher, "_git", lambda *args: "1" * 40)

    def forbidden(*args, **kwargs):
        raise AssertionError("child process must not launch before gate")

    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="frozen"):
        launcher.main(
            [
                "--manifest",
                str(DRAFT_MANIFEST),
                "--gate",
                str(gate),
            ]
        )


def test_explicit_unknown_cleanup_is_not_false_confirmation(tmp_path):
    trace = write_trace(tmp_path, [event("cleanup_finished", status="unknown")])
    summary = module().summarize_run(trace, None)
    assert summary["cleanup"]["status"] == "unknown"
    assert summary["cleanup_confirmed"] is None


@pytest.mark.parametrize("case_number", range(1, 13))
def test_integrated_behavioral_case(case_number, tmp_path):
    spec = importlib.util.spec_from_file_location("behavioral", ROOT / "scripts/eval_offline.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    case = next((ROOT / "tests/fixtures/evaluation").glob(f"{case_number:02d}-*/scenario.json"))
    report = driver.evaluate_behavior(case, tmp_path)
    assert report["status"] == "passed", report
    assert report["checked_invariants"]
    assert report["official_bluesec_score"] is None
    assert report["model_accuracy_evaluated"] is False


@pytest.mark.parametrize("damage", ["directory", "corrupt"])
def test_salvage_terminal_zero_and_partial_result_after_artifact_damage(tmp_path, damage):
    result = tmp_path / "result.json"
    if damage == "directory":
        result.mkdir()
    else:
        result.write_text('{"broken')
    partial = tmp_path / "partial-result.json"
    partial.write_text(
        json.dumps(
            {
                "run_id": "r",
                "cleanup": {"status": "closed_confirmed"},
                "task_results": [],
                "artifact_errors": [{"operation": "write_result"}],
            }
        )
    )
    trace = write_trace(
        tmp_path,
        [
            event(
                "task_finished",
                task_id="t",
                result={"task_id": "t", "completion_reason": "terminated", "quality_score": 0},
            )
        ],
    )
    summary = module().summarize_run(trace, result)
    assert summary["accepted_final"] is True
    assert summary["quality_score"] == 0
    assert summary["cleanup_confirmed"] is True
    assert summary["artifact_errors"]
    assert summary["source_sha256"]["partial_result"] == module().sha256_file(partial)


def test_wrong_namespace_uses_structured_violation_code_not_reason_spelling(tmp_path):
    trace = write_trace(
        tmp_path,
        [
            event(
                "action_rejected",
                reason="action_rejected",
                violations=[{"code": "wrong_kind", "object_id": "r"}],
            ),
            event("action_rejected", reason="runtime_schema", violations=["invalid schema"]),
        ],
    )
    assert module().summarize_run(trace, None)["provenance_rejections"] == 1


@pytest.mark.parametrize("category", ["capacity", "auth", "schema"])
def test_definite_creation_rejection_consumes_slot_and_keeps_stop_reason(tmp_path, category):
    journal = module().AttemptJournal(tmp_path / "attempts.jsonl", "abc")
    slots = [{"slot_id": f"{i:02d}"} for i in range(1, 17)]
    module().execute_series(
        slots,
        journal,
        lambda _: {
            "error_category": category,
            "run_creation_rejected": True,
            "cleanup_confirmed": False,
            "run_id": None,
        },
    )
    report = module().reconcile_attempts(slots, journal.path)
    assert report["attempted_count"] == 1
    assert report["stop_reason"] == category
    assert all(row["status"] == "not_started" for row in report["attempts"][1:])


def synthetic_settings(**overrides):
    from bluesec1_agent.settings import Settings

    return Settings(
        _env_file=None,
        scenario_runtime_token="synthetic",
        llm_api_key="synthetic",
        llm_base_url="https://offline.invalid/v1",
        llm_default_model="gpt-6-luna",
        **overrides,
    )


def test_effective_configuration_rejects_environment_drift_without_exposing_keys():
    reference = module().sanitized_effective_configuration(synthetic_settings())
    assert "synthetic" not in json.dumps(reference)
    changed = synthetic_settings(llm_max_completion_tokens=17)
    with pytest.raises(ValueError, match="configuration"):
        module().verify_effective_configuration(reference, changed)
    module().verify_effective_configuration(reference, synthetic_settings())


def test_frozen_gate_requires_reviewed_config_and_hidden_control_hashes():
    manifest = json.loads(DRAFT_MANIFEST.read_text())
    manifest.update(freeze_state="frozen", integrated_sha="1" * 40)
    gate = {
        "live_ready": True,
        "integrated_sha": "1" * 40,
        "manifest_sha256": "abc",
        "reviewer": "R",
        "review_approved": True,
        "review_artifact": "/review.md",
    }
    with pytest.raises(ValueError, match="configuration"):
        module().validate_live_gate(manifest, "abc", gate, "1" * 40)
    manifest["sanitized_configuration"] = module().sanitized_effective_configuration(
        synthetic_settings()
    )
    with pytest.raises(ValueError, match="hidden"):
        module().validate_live_gate(manifest, "abc", gate, "1" * 40)


def test_child_config_guard_precedes_cli_even_after_parent_check(tmp_path, monkeypatch):
    import sys

    from bluesec1_agent import cli
    from bluesec1_agent.settings import Settings

    spec = importlib.util.spec_from_file_location(
        "checked_launcher", ROOT / "scripts/run_experiments.py"
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    expected = module().sanitized_effective_configuration(synthetic_settings())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"sanitized_configuration": expected}))
    monkeypatch.setenv("SCENARIO_RUNTIME_TOKEN", "synthetic")
    monkeypatch.setenv("LLM_API_KEY", "synthetic")
    monkeypatch.setenv("LLM_BASE_URL", "https://offline.invalid/v1")
    monkeypatch.setenv("LLM_DEFAULT_MODEL", "gpt-6-luna")
    monkeypatch.setenv("LLM_MAX_COMPLETION_TOKENS", "17")
    monkeypatch.setattr(Settings, "model_config", {**Settings.model_config, "env_file": None})
    monkeypatch.setattr(sys, "argv", ["-c", str(manifest_path)])

    def forbidden():
        raise AssertionError("CLI must not run after configuration drift")

    monkeypatch.setattr(cli, "main", forbidden)
    with pytest.raises(SystemExit, match="configuration refused"):
        exec(launcher._CHECKED_CLI, {})


def test_child_command_uses_canonical_env_file_and_fixed_cli_limits(tmp_path):
    spec = importlib.util.spec_from_file_location("commands", ROOT / "scripts/run_experiments.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    manifest_path = DRAFT_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    command = launcher.command_for(manifest, manifest["slots"][0], tmp_path, manifest_path)
    assert command[0] == "uv"
    assert command[command.index("--env-file") + 1] == manifest["canonical_env_file"]
    assert command[command.index("--max-steps") + 1] == "60"
    assert command[command.index("--task-timeout-seconds") + 1] == "900"
    assert command[command.index("--runtime-state-dir") + 1] == manifest["runtime_state_dir"]
