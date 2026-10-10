"""Offline evidence accounting and a fail-closed, append-only experiment ledger.

This module has no runtime/provider client dependency and never loads credentials.
Scores are authoritative only when present in a canonical server task result.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCORE_SEMANTICS = (
    "TaskResult protobuf scores are nonoptional scalars (has_presence=false): absent wire "
    "and explicit zero cannot be distinguished. Canonical terminal JSON zero is retained; "
    "missing task_result or missing JSON score is unknown. No ListFields presence inference."
)
MODELS = ("gpt-6-luna", "gpt-6-astra", "claude-opus-5-5", "gpt-6.1-sol")
POLICY = {
    "id": "cycle-01-conservative-v1",
    "changes_default": False,
    "min_accepted_with_quality": 3,
    "requires_verified_provenance": True,
    "requires_confirmed_cleanup": True,
    "requires_no_open_p1_p2": True,
    "ranking": ["median_quality_desc", "median_efficiency_desc_if_complete", "median_walltime_asc"],
    "incomplete_or_stopped_or_all_zero": "investigate_then_collect_more",
    "tie": "no_preference",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_events(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read legacy/v2 JSONL; report a torn last record, reject ambiguous duplicates."""
    if not path.exists():
        return [], ["missing_trace"]
    lines = path.read_text(encoding="utf-8").splitlines()
    events, warnings, seen = [], [], {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                warnings.append(f"torn_final_record:{index + 1}")
                break
            raise ValueError(f"invalid JSONL line {index + 1}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"non-object JSONL line {index + 1}")
        if item.get("schema_version", 2) not in (1, 2):
            raise ValueError("unsupported schema_version")
        event_id = item.get("event_id")
        if event_id is not None and event_id in seen:
            if seen[event_id] != item:
                raise ValueError(f"conflicting event_id: {event_id}")
            warnings.append(f"duplicate_event:{event_id}")
            continue
        if event_id is not None:
            seen[event_id] = item
        events.append(item)
    return events, warnings


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _time(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return None


def summarize_run(trace_path: Path, result_path: Path | None) -> dict[str, Any]:
    """Preserve unknowns and terminal outcomes even with missing/failed artifact writes."""
    events, warnings = read_events(trace_path)
    result: dict[str, Any] = {}
    artifact_errors = []
    partial_path = result_path.with_name("partial-result.json") if result_path else None
    for candidate in (result_path, partial_path):
        if candidate is None or not candidate.exists():
            continue
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("result must be an object")
        except (OSError, ValueError) as error:
            artifact_errors.append(
                {
                    "operation": "read_result",
                    "file": candidate.name,
                    "error_type": type(error).__name__,
                }
            )
            continue
        result = loaded
        if candidate == partial_path:
            warnings.append("partial_result_fallback")
        break
    if not result:
        warnings.append("missing_result")
    tasks, task_ids, attempts, finished = {}, [], {}, {}
    remote_calls, cache_hits, repairs, rejections, decisions = 0, 0, 0, 0, 0
    rpc_starts = set()
    run_id = result.get("run_id")
    cleanup = result.get("cleanup")
    cleanup_confirmed = result.get("cleanup_confirmed")
    errors = []
    provenance_verified = result.get("provenance_verified")
    for index, row in enumerate(events):
        name, p = row.get("event"), row.get("payload", {})
        if not isinstance(p, dict):
            raise ValueError("event payload must be an object")
        run_id = p.get("run_id") or run_id
        task_id = p.get("task_id")
        if task_id and task_id not in task_ids:
            task_ids.append(task_id)
        if name == "task_finished" and isinstance(p.get("result"), dict):
            tasks[task_id or p["result"].get("task_id") or "unknown"] = p["result"]
        if name == "tool_result":
            remote_calls += 1
            response = p.get("response", {})
            if isinstance(response, dict) and isinstance(response.get("task_result"), dict):
                task = response["task_result"]
                tasks[task_id or task.get("task_id") or "unknown"] = task
        if name == "tool_call_started":
            rpc_starts.add(p.get("call_id") or (task_id, p.get("step", index)))
        if name == "llm_attempt_started":
            identity = p.get("attempt_id") or f"legacy-start:{index}"
            if identity in attempts:
                if attempts[identity] != p:
                    raise ValueError(f"conflicting LLM attempt: {identity}")
                warnings.append(f"duplicate_llm_start:{identity}")
            attempts[identity] = p
        if name == "llm_attempt_finished":
            identity = p.get("attempt_id") or f"legacy-finish:{index}"
            if identity in finished and finished[identity] != p:
                raise ValueError(f"conflicting LLM outcome: {identity}")
            finished[identity] = p
        cache_hits += name == "cache_hit"
        repairs += name == "schema_repair"
        decisions += name == "step_selected"
        rejections += name == "submission_rejected" or (
            name == "action_rejected"
            and (
                p.get("reason") in ("provenance", "unknown_id", "wrong_namespace", "evidence")
                or any(
                    isinstance(v, dict)
                    and v.get("code")
                    in {
                        "unknown_object",
                        "wrong_kind",
                        "not_observed",
                        "unknown_field",
                        "artifact_type_mismatch",
                        "conflicting_type",
                    }
                    for v in p.get("violations", [])
                )
            )
        )
        if name in ("run_failed", "task_error"):
            errors.append(p.get("error_type", "unknown"))
        if name in ("cleanup_finished", "run_cleanup"):
            cleanup = p.get("cleanup", p)
        if name == "run_finished":
            cleanup = p.get("cleanup", cleanup)
            if "cleanup_succeeded" in p:
                # Legacy records do not establish server-confirmed closure.
                warnings.append("legacy_cleanup_succeeded_not_confirmation")
        if name == "provenance_validated":
            provenance_verified = p.get("valid") is True
    for task in result.get("task_results", []):
        if not isinstance(task, dict):
            raise ValueError("invalid task result")
        key = task.get("task_id") or (task_ids[0] if len(task_ids) == 1 else "unknown")
        existing = tasks.get(key, {})
        # Do not erase authoritative terminal fields on a partial final artifact.
        tasks[key] = {**existing, **task}
    if isinstance(cleanup, dict):
        cleanup_confirmed = (
            None
            if cleanup.get("status") in (None, "unknown")
            else cleanup.get("status") == "closed_confirmed"
        )
    elif isinstance(cleanup, str):
        cleanup_confirmed = None if cleanup == "unknown" else cleanup == "closed_confirmed"
    terminal = list(tasks.values())
    for key in tasks:
        if key != "unknown" and key not in task_ids:
            task_ids.append(key)
    primary = terminal[0] if len(terminal) == 1 else {}
    accepted = primary.get("completion_reason") == "terminated" if primary else None
    identities = set(attempts) | set(finished)
    usage_fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
    )
    usage: dict[str, Any] = {"attempts": len(identities), "complete": False}
    for field in usage_fields:
        values = []
        for identity in identities:
            item = finished.get(identity, {})
            raw_usage = item.get("usage") or item
            value = _number(raw_usage.get(field))
            if value is not None:
                values.append(value)
        usage[field] = sum(values) if identities and len(values) == len(identities) else None
        usage["known_" + field] = sum(values) if values else None
        usage[field + "_known_attempts"] = len(values)
    usage["complete"] = bool(identities) and usage["total_tokens"] is not None
    times = [t for row in events if (t := _time(row.get("timestamp"))) is not None]
    walltime = _number(result.get("walltime_seconds"))
    if walltime is None and len(times) > 1:
        try:
            walltime = max(0, (max(times) - min(times)).total_seconds())
        except TypeError:
            warnings.append("incompatible_timestamps")
    return {
        "schema_version": 1,
        "trace_path": str(trace_path.resolve()),
        "result_path": str(result_path.resolve()) if result_path else None,
        "source_sha256": {
            "trace": sha256_file(trace_path) if trace_path.exists() else None,
            "result": sha256_file(result_path) if result_path and result_path.is_file() else None,
            "partial_result": sha256_file(partial_path)
            if partial_path and partial_path.is_file()
            else None,
        },
        "run_id": run_id,
        "task_ids": task_ids,
        "task_results": terminal,
        "status": "accepted" if accepted else "not_accepted" if primary else "incomplete",
        "accepted_final": accepted,
        "quality_score": _number(primary.get("quality_score")),
        "efficiency_score": _number(primary.get("efficiency_score")),
        "reward": _number(primary.get("total_reward", primary.get("reward"))),
        "score_semantics": SCORE_SEMANTICS,
        "walltime_seconds": walltime,
        "llm_attempts": len(identities) if identities else None,
        "llm_decisions": decisions,
        "remote_tool_calls": len(rpc_starts) or remote_calls,
        "cache_hits": cache_hits,
        "repairs": repairs,
        "provenance_rejections": rejections,
        "provenance_verified": provenance_verified,
        "usage": usage,
        "cleanup": cleanup,
        "cleanup_confirmed": cleanup_confirmed,
        "error_category": result.get("error_category"),
        "run_creation_rejected": result.get("run_creation_rejected"),
        "artifact_errors": [*result.get("artifact_errors", []), *artifact_errors],
        "metadata": result.get("metadata", {}),
        "errors": errors,
        "warnings": warnings,
        "cost": None,
    }


class PartialReplay:
    """Exact observed action lookup; it cannot stand in for the unseen full graph."""

    def __init__(self, observations: list[dict[str, Any]]) -> None:
        self._responses: dict[str, Any] = {}
        for observation in observations:
            key = self._key(observation["tool_name"], observation["arguments"])
            value = observation["response"]
            if key in self._responses and self._responses[key] != value:
                raise ValueError("conflicting replay response")
            self._responses[key] = value

    @staticmethod
    def _key(tool_name: str, arguments: dict[str, Any]) -> str:
        return json.dumps(
            [tool_name, {k: v for k, v in arguments.items() if k != "reasoning"}], sort_keys=True
        )

    def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        value = self._responses.get(self._key(tool_name, arguments))
        if value is None:
            return {"status": "error", "error": "unsupported_action", "tool_name": tool_name}
        return json.loads(json.dumps(value))


class AttemptJournal:
    """Append and fsync both the event and directory before returning to the caller.

    The caller must hold the series lock. A torn or unwriteable ledger forbids launch.
    This lock is deliberately independent of the CLI's canonical runtime RunLock.
    """

    def __init__(self, path: Path, manifest_sha256: str) -> None:
        self.path = path
        self.manifest_sha256 = manifest_sha256

    def append(self, event: str, **payload: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "schema_version": 1,
            "event_id": uuid.uuid4().hex,
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "manifest_sha256": self.manifest_sha256,
            **payload,
        }
        line = json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n"
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def execute_series(
    slots: list[dict[str, Any]],
    journal: AttemptJournal,
    execute: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    """A fresh series only; reservation consumes a slot even if creation fails."""
    if not 1 <= len(slots) <= 16:
        raise ValueError("series must have between 1 and 16 slots")
    if len({s["slot_id"] for s in slots}) != len(slots):
        raise ValueError("duplicate slot")
    if journal.path.exists() and journal.path.stat().st_size:
        raise ValueError("refusing to resume or replace an existing series")
    consecutive_infrastructure = 0
    for index, slot in enumerate(slots):
        journal.append("reserved", slot_id=slot["slot_id"])
        # If either durable write fails, no CreateRun can occur.
        journal.append("started", slot_id=slot["slot_id"])
        try:
            summary = execute(slot)
        except BaseException as exc:
            summary = {
                "error_category": "unknown_run_state",
                "error_type": type(exc).__name__,
                "cleanup_confirmed": None,
                "accepted_final": None,
            }
            journal.append("finished", slot_id=slot["slot_id"], summary=summary)
            _stop_series(slots[index + 1 :], journal, "executor_failed_unknown_state")
            raise
        journal.append("finished", slot_id=slot["slot_id"], summary=summary)
        category = summary.get("error_category")
        consecutive_infrastructure = (
            consecutive_infrastructure + 1 if category == "infrastructure" else 0
        )
        reason = None
        if category == "capacity" or (
            category in ("auth", "schema") and summary.get("run_creation_rejected") is True
        ):
            reason = category
        elif summary.get("cleanup_confirmed") is not True:
            reason = "unknown_run_state"
        elif category in ("auth", "schema", "unknown_run_state"):
            reason = category
        elif consecutive_infrastructure >= 2:
            reason = "two_consecutive_infrastructure_failures"
        if reason:
            _stop_series(slots[index + 1 :], journal, reason)
            return
    journal.append("series_completed")


def _stop_series(slots: list[dict[str, Any]], journal: AttemptJournal, reason: str) -> None:
    journal.append("series_stopped", reason=reason)
    for slot in slots:
        journal.append("not_started", slot_id=slot["slot_id"], reason=reason)


def reconcile_attempts(
    slots: list[dict[str, Any]], path: Path, manifest_sha256: str | None = None
) -> dict[str, Any]:
    """Retain all scheduled slots and all reserved failures, including no run_id."""
    events, warnings = read_events(path)
    rows = {
        slot["slot_id"]: {**slot, "attempted": False, "status": "planned", "run_id": None}
        for slot in slots
    }
    stopped = None
    completed = False
    for event in events:
        if manifest_sha256 and event.get("manifest_sha256") != manifest_sha256:
            raise ValueError("attempt ledger manifest hash mismatch")
        name, slot_id = event.get("event"), event.get("slot_id")
        if name == "series_stopped":
            stopped = event["reason"]
            continue
        if name == "series_completed":
            completed = True
            continue
        if slot_id not in rows:
            raise ValueError(f"unknown slot: {slot_id}")
        row = rows[slot_id]
        if name == "reserved":
            if row["status"] != "planned":
                raise ValueError(f"duplicate/out-of-order reservation: {slot_id}")
            row.update(attempted=True, status="reserved")
        elif name == "started":
            if row["status"] != "reserved":
                raise ValueError("started without reservation")
            row["status"] = "started"
        elif name == "finished":
            if row["status"] not in ("reserved", "started"):
                raise ValueError("finished without reservation or duplicate finish")
            row.update(event.get("summary", {}))
            row.update(attempted=True, status="finished")
        elif name == "not_started":
            if row["status"] != "planned":
                raise ValueError("not_started cannot erase an attempt")
            row.update(status="not_started", stop_reason=event.get("reason"))
        else:
            raise ValueError(f"unknown journal event: {name}")
    attempts = list(rows.values())
    return {
        "attempts": attempts,
        "attempted_count": sum(r["attempted"] for r in attempts),
        "series_complete": completed,
        "stop_reason": stopped,
        "warnings": warnings,
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("cycle_id") != "cycle-01":
        raise ValueError("unsupported manifest")
    if manifest.get("freeze_state") not in ("draft", "frozen"):
        raise ValueError("invalid freeze state")
    required_limits = {
        "max_attempts": 16,
        "max_tasks": 1,
        "max_steps": 60,
        "task_timeout_seconds": 900,
        "cleanup_timeout_seconds": 30,
        "max_context_chars": 24000,
    }
    if manifest.get("limits") != required_limits:
        raise ValueError("manifest limits differ from approved cycle")
    if manifest.get("selection_policy") != POLICY:
        raise ValueError("selection policy differs from preregistration")
    slots = manifest.get("slots", [])
    if len(slots) != 16 or [s.get("slot_id") for s in slots] != [f"{i:02d}" for i in range(1, 17)]:
        raise ValueError("manifest requires 16 ordered unique slots")
    configs = manifest.get("configurations", {})
    expected = [(m, "v2", "smoke") for m in MODELS]
    expected += [("gpt-6-luna", s, "behavior") for s in ("baseline", "v2") * 3]
    expected += [(m, "v2", "strong_models") for m in MODELS[1:] * 2]
    for slot, want in zip(slots, expected, strict=True):
        config = configs.get(slot.get("configuration_id"), {})
        if (config.get("model"), config.get("strategy"), slot.get("phase")) != want:
            raise ValueError("configuration/order differs from approved schedule")
    if manifest.get("stop_policy") != [
        "unknown_run_state",
        "auth",
        "schema",
        "capacity",
        "two_consecutive_infrastructure_failures",
    ]:
        raise ValueError("invalid stop policy")
    for key in ("runtime_state_dir", "run_artifacts_dir", "attempts_path"):
        if not Path(manifest.get(key, "")).is_absolute():
            raise ValueError(f"{key} must be absolute")


def validate_live_gate(
    manifest: dict[str, Any], digest: str, gate: dict[str, Any], checkout_sha: str
) -> None:
    validate_manifest(manifest)
    if manifest["freeze_state"] != "frozen":
        raise ValueError("live requires frozen manifest")
    sha = manifest.get("integrated_sha")
    if (
        not isinstance(sha, str)
        or len(sha) != 40
        or sha != checkout_sha
        or gate.get("integrated_sha") != sha
        or gate.get("manifest_sha256") != digest
        or gate.get("live_ready") is not True
        or gate.get("reviewer") != "R"
        or gate.get("review_approved") is not True
        or not gate.get("review_artifact")
    ):
        raise ValueError("live gate must approve exact integrated SHA and manifest SHA256")
    if not manifest.get("sanitized_configuration"):
        raise ValueError("frozen manifest requires sanitized effective configuration")
    controls = manifest.get("hidden_control_hashes")
    if (
        not isinstance(controls, dict)
        or not controls
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
            for value in controls.values()
        )
    ):
        raise ValueError("frozen manifest requires hidden control hashes")
    if not Path(manifest.get("canonical_env_file", "")).is_absolute():
        raise ValueError("canonical env-file path must be absolute")


def sanitized_effective_configuration(settings: Any) -> dict[str, Any]:
    """Explicit non-secret allowlist; never dump Settings or environment variables."""
    from bluesec1_agent.recording import sanitize_endpoint
    from bluesec1_client.remote.models import DeadlinePolicy, RetryPolicy

    return {
        "llm_base_url": sanitize_endpoint(settings.llm_base_url),
        "runtime_endpoint": sanitize_endpoint(settings.scenario_runtime_endpoint),
        "verify_tls": settings.scenario_runtime_verify_tls,
        "arena": settings.scenario_runtime_arena,
        "configured_default_model": settings.llm_default_model,
        "max_llm_attempts": settings.llm_max_step_retries,
        "max_completion_tokens": settings.llm_max_completion_tokens,
        "llm_timeout_seconds": settings.llm_timeout_seconds,
        "agent_name": settings.agent_name,
        "run_label": settings.remote_run_label,
        "sdk_retry_policy": RetryPolicy().model_dump(mode="json"),
        "sdk_deadlines": DeadlinePolicy().model_dump(mode="json"),
    }


def verify_effective_configuration(expected: dict[str, Any], settings: Any) -> None:
    """Refuse drift before any provider/runtime construction; values stay unprinted."""
    actual = sanitized_effective_configuration(settings)
    if expected != actual:
        raise ValueError("effective configuration differs from frozen manifest")


def compare_attempts(manifest: dict[str, Any], accounting: dict[str, Any]) -> dict[str, Any]:
    """Small unpaired samples: report raw values, missingness, medians and no accuracy."""
    groups = {}
    for config_id, config in manifest["configurations"].items():
        rows = [
            r
            for r in accounting["attempts"]
            if r.get("configuration_id") == config_id and r["attempted"]
        ]
        accepted = [r for r in rows if r.get("accepted_final") is True]
        group = {
            **config,
            "attempts": len(rows),
            "accepted_finals": len(accepted),
            "incomplete_count": len(rows) - len(accepted),
            "incomplete_rate": (len(rows) - len(accepted)) / len(rows) if rows else None,
        }
        for field in ("quality_score", "efficiency_score", "reward", "walltime_seconds"):
            values = [r[field] for r in accepted if _number(r.get(field)) is not None]
            group[field] = {
                "values": values,
                "missing_accepted": len(accepted) - len(values),
                "missing_all_attempts": sum(r.get(field) is None for r in rows),
                "median": statistics.median(values) if values else None,
            }
        group["eligible"] = len(accepted) >= 3 and all(
            r.get("quality_score") is not None
            and r.get("cleanup_confirmed") is True
            and r.get("provenance_verified") is True
            and r.get("open_p1_p2") == 0
            for r in accepted
        )
        groups[config_id] = group
    candidates = [key for key, group in groups.items() if group["eligible"]]
    incomplete = any(
        r["attempted"]
        and (
            r.get("accepted_final") is not True
            or r.get("quality_score") is None
            or r.get("cleanup_confirmed") is not True
            or r.get("walltime_seconds") is None
        )
        for r in accounting["attempts"]
    )
    all_zero = not any(g["quality_score"]["median"] for g in groups.values())
    recommendation = "investigate_then_collect_more"
    ranking = []
    if (
        accounting["series_complete"]
        and not accounting["stop_reason"]
        and not incomplete
        and not all_zero
        and candidates
    ):
        efficiency_complete = all(
            groups[c]["efficiency_score"]["missing_accepted"] == 0 for c in candidates
        )

        def rank(key):
            g = groups[key]
            return (
                -g["quality_score"]["median"],
                -g["efficiency_score"]["median"] if efficiency_complete else 0,
                g["walltime_seconds"]["median"]
                if g["walltime_seconds"]["missing_accepted"] == 0
                else float("inf"),
            )

        ranking = sorted(candidates, key=rank)
        recommendation = "next_experiment_candidate"
        if len(ranking) > 1 and rank(ranking[0]) == rank(ranking[1]):
            recommendation = "no_preference"
    return {
        "schema_version": 1,
        **accounting,
        "groups": groups,
        "recommendation": recommendation,
        "candidate_ranking": ranking,
        "changes_default": False,
        "precision": None,
        "recall": None,
        "limitations": [
            "small_unpaired_random_live_tasks",
            "no_ground_truth",
            "synthetic_checks_do_not_measure_model_accuracy",
            "smoke_tasks_not_independent_after_prompt_tuning",
        ],
        "score_semantics": SCORE_SEMANTICS,
    }
