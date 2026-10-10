"""Bounded graph investigation with a shared run deadline and optional case advisor."""

import asyncio
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from bluesec1_agent.agent import SYSTEM_PROMPT, SimpleSGRAgent
from bluesec1_agent.budget import Budget, BudgetExceededError
from bluesec1_agent.catalog import CatalogCompatibilityError, check_catalog, validate_arguments
from bluesec1_agent.context import build_context
from bluesec1_agent.deadline import FALLBACK_REASON
from bluesec1_agent.evidence import EvidenceStore
from bluesec1_agent.frontier import navigation_groups, rank_frontier
from bluesec1_agent.llm import render_json
from bluesec1_agent.models import FinishInvestigationRequest, NextStep
from bluesec1_agent.provenance import validate_navigation, validate_submission
from bluesec1_client import RemoteBenchmarkSession, TaskResult

QUALITY_PROMPT = (
    SYSTEM_PROMPT
    + """

The deterministic reader has already gathered the bounded graph evidence below.
You are now in FINALIZATION: return only finish_investigation, task_completed=true.
Compare the malicious and authorized explanations using the actual properties and
relations. Cite exact observed fields. Enumerate all justified response artifacts
that were observed, with their appropriate action kinds; don't invent missing scope.
An alert is a hypothesis, and missing evidence is unknown. A valid signature, familiar
binary name or normal parent chain alone does not establish benign intent. Explain
the concrete observed behavior that supports your verdict and any unresolved scope.
Keep reasoning concise so the structured decision fits in the output budget.
"""
)


def allocate_task_seconds(
    remaining_seconds: float, remaining_tasks: int, *, cleanup_seconds: float, lease_seconds: float
) -> float:
    """Reserve cleanup once and an expected lease allowance for remaining tasks.

    The actual lease may take longer; investigation is clamped again after it.
    """
    seconds = (remaining_seconds - cleanup_seconds) / remaining_tasks - lease_seconds
    if not math.isfinite(seconds) or seconds < 12:
        raise BudgetExceededError("Insufficient run time for remaining tasks and cleanup.")
    return min(60.0, seconds)


def quality_frontier(store: EvidenceStore, attempted: set[tuple[str, str]]):
    """Prioritize original process triggers even after later empty ID references."""
    groups = navigation_groups(store, trigger_ids=store.trigger_ids)
    observed_links = sum(
        store.get(rid, "relation").state == "observed" for rid in groups["focused"]
    )
    observed_neighbors = sum(
        store.get(eid, "entity").state == "observed"
        for eid in groups["neighbors"] - set(store.trigger_ids)
    )

    def priority(lead):
        types = {
            ref.fields.get("type")
            for ref in store.history(lead.object_id, lead.kind)
            if ref.fields.get("type")
        }
        if lead.object_id in store.trigger_ids:
            if types & {"windows_process", "linux_process"}:
                return 0
            if types & {"host", "windows_user", "linux_user"}:
                return 4
            return 1
        if lead.kind == "entity" and lead.object_id in groups["neighbors"]:
            return 1.5 if observed_links >= 2 and observed_links > 2 * observed_neighbors else 3
        if lead.kind == "relation" and lead.object_id in groups["focused"]:
            return 2
        return lead.priority

    return sorted(
        [
            lead
            for lead in rank_frontier(store, trigger_ids=store.trigger_ids)
            if (lead.kind, lead.object_id) not in attempted
        ],
        key=lambda lead: (priority(lead), lead.object_id),
    )


class QualitySubmissionAgent(SimpleSGRAgent):
    def __init__(
        self,
        *args,
        deadline_epoch: float,
        advisory_dir: Path | None = None,
        cleanup_reserve_seconds: float = 30.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.max_tasks is None:
            raise ValueError("Quality mode requires an explicit task count.")
        self.run_budget = Budget(deadline_epoch - time.time())
        self.advisory_dir = advisory_dir
        self.cleanup_reserve_seconds = cleanup_reserve_seconds
        self.run_id = None

    async def run(self, client):
        self.run_id = client.run.run_id
        return await super().run(client)

    def _next_task_timeout(self):
        seconds = allocate_task_seconds(
            self.run_budget.remaining_seconds(),
            self.max_tasks - len(self.task_results),
            cleanup_seconds=self.cleanup_reserve_seconds,
            lease_seconds=8,
        )
        self._record(
            "quality_budget",
            remaining_tasks=self.max_tasks - len(self.task_results),
            run_remaining_seconds=self.run_budget.remaining_seconds(),
            task_seconds=seconds,
        )
        return seconds

    def _publish_brief(self, session, body, budget):
        digest = hashlib.sha256(render_json(body).encode()).hexdigest()
        identity = {"run_id": self.run_id, "task_id": session.task_id, "evidence_sha256": digest}
        request = {
            **identity,
            "case": body,
            "response_due_epoch": time.time() + max(0, budget.remaining_seconds() - 9),
        }
        self._record("quality_case_brief", **request)
        if self.advisory_dir is not None:
            try:
                self.advisory_dir.mkdir(parents=True, exist_ok=True)
                temporary = self.advisory_dir / ".request.tmp"
                temporary.write_text(render_json(request))
                temporary.replace(self.advisory_dir / "request.json")
            except OSError as error:
                self._record("quality_advisory_write_failed", error_type=type(error).__name__)
        return identity

    def _advisory(self, identity):
        if self.advisory_dir is None:
            return None
        try:
            path = self.advisory_dir / "response.json"
            if path.stat().st_size > 262144:
                return None
            response = json.loads(path.read_text())
            if set(response) != {*identity, "decision"}:
                return None
            if any(response.get(key) != value for key, value in identity.items()):
                return None
            return NextStep.model_validate(response["decision"])
        except (OSError, ValueError, TypeError):
            return None

    @staticmethod
    def _validate_decision(decision, schema, store):
        if (
            not isinstance(decision.function, FinishInvestigationRequest)
            or not decision.task_completed
        ):
            return ["Only a completed finish_investigation decision is permitted."]
        arguments = decision.function.model_dump(mode="json", exclude={"tool"}, exclude_none=True)
        return validate_arguments(arguments, schema) + [
            asdict(error) for error in validate_submission(decision.function.submission, store)
        ]

    async def _investigate(self, session: RemoteBenchmarkSession, budget: Budget) -> TaskResult:
        global_available = max(
            0.0, self.run_budget.remaining_seconds() - self.cleanup_reserve_seconds
        )
        budget = budget.with_reserve(max(0.0, budget.remaining_seconds() - global_available))
        specs = self._tool_specs(session.task.observation)
        schemas = check_catalog(specs)
        if "finish_investigation" not in schemas:
            raise CatalogCompatibilityError("The runtime did not advertise finish_investigation.")
        store = EvidenceStore()
        store.seed_observation(session.task.observation)
        attempted = set()
        read_seconds = min(8.0, budget.remaining_seconds() * 0.25)
        read_budget = budget.with_reserve(budget.remaining_seconds() - read_seconds)
        step = 0
        for step in range(1, min(24, self.max_steps - 1) + 1):
            if read_budget.remaining_seconds() < 0.1:
                break
            leads = quality_frontier(store, attempted)
            if not leads:
                break
            lead = leads[0]
            attempted.add((lead.kind, lead.object_id))
            name = "get_" + lead.kind
            arguments = {
                lead.kind + "_id": lead.object_id,
                "reasoning": "Read bounded graph evidence.",
            }
            if name not in schemas or validate_arguments(arguments, schemas[name]):
                continue
            if validate_navigation(name, arguments, store):
                continue
            self._update_context(task_id=session.task_id, step=step)
            self._record(
                "tool_call_started",
                task_id=session.task_id,
                step=step,
                tool_name=name,
                arguments=arguments,
            )
            try:
                async with asyncio.timeout(min(3.0, read_budget.remaining_seconds())):
                    response = await session.call_tool(name, arguments)
            except Exception as error:
                self._record(
                    "tool_call_finished",
                    task_id=session.task_id,
                    step=step,
                    tool_name=name,
                    status="error",
                    error_type=type(error).__name__,
                )
                if getattr(session, "terminal_state", "active") != "active":
                    raise
                continue
            self._record(
                "tool_result",
                task_id=session.task_id,
                step=step,
                response=response.model_dump(mode="json"),
            )
            feedback = self._tool_feedback(response)
            self._record(
                "tool_call_finished",
                task_id=session.task_id,
                step=step,
                tool_name=name,
                status=feedback["status"],
            )
            if response.task_result is not None:
                return TaskResult.model_validate(response.task_result)
            if response.terminated or response.truncated:
                raise RuntimeError("Read ended without authoritative task result.")
            store.ingest(
                step=step,
                tool_name=name,
                arguments=arguments,
                result=feedback.get("result"),
                status=feedback["status"],
            )

        step += 1
        self._update_context(task_id=session.task_id, step=step)
        body = {
            "alert": str(session.task.observation.get("alert_text", ""))[:12000],
            "evidence": build_context(
                store,
                trigger_ids=store.trigger_ids,
                max_chars=self.max_context_chars,
                remaining_steps=1,
            ),
            "available_tools": [spec for spec in specs if spec["name"] == "finish_investigation"],
        }
        identity = self._publish_brief(session, body, budget)
        messages = [
            {"role": "system", "content": QUALITY_PROMPT},
            {"role": "user", "content": render_json(body)},
        ]
        model_budget = budget.with_reserve(9.0)
        for name, value in (
            ("finish_only", True),
            ("max_retries", 1),
            ("max_completion_tokens", 4096),
        ):
            if hasattr(self.llm, name):
                setattr(self.llm, name, value)
        if hasattr(self.llm, "budget"):
            self.llm.budget = model_budget
        decision = None
        origin = "model"
        failures = []
        for attempt in range(2):
            advisor = self._advisory(identity)
            if advisor is not None and not self._validate_decision(
                advisor, schemas["finish_investigation"], store
            ):
                decision, origin = advisor, "coordinator_advisory"
                break
            if model_budget.remaining_seconds() < 2:
                break
            # A fast invalid answer can be repaired; a slow first answer keeps all
            # remaining reasoning time. The final RPC reserve is never borrowed.
            seconds = model_budget.remaining_seconds()
            if hasattr(self.llm, "timeout_seconds"):
                self.llm.timeout_seconds = seconds
            try:
                async with asyncio.timeout(seconds):
                    candidate = await self.llm.next_step(messages)
                errors = self._validate_decision(candidate, schemas["finish_investigation"], store)
                self._record(
                    "quality_model_decision",
                    task_id=session.task_id,
                    attempt=attempt + 1,
                    decision=candidate.model_dump(mode="json"),
                    validation_errors=errors,
                )
                if not errors:
                    decision = candidate
                    break
                messages += [
                    {
                        "role": "assistant",
                        "content": render_json(candidate.model_dump(mode="json")),
                    },
                    {
                        "role": "user",
                        "content": "Repair only using observed evidence: " + render_json(errors),
                    },
                ]
                failures.append("provenance_or_schema")
            except Exception as error:
                failures.append(type(error).__name__)
                self._record(
                    "quality_model_failed",
                    task_id=session.task_id,
                    attempt=attempt + 1,
                    error_type=type(error).__name__,
                )
        if decision is None:
            advisor = self._advisory(identity)
            if advisor is not None and not self._validate_decision(
                advisor, schemas["finish_investigation"], store
            ):
                decision, origin = advisor, "coordinator_advisory"
        if decision is None:
            origin = "incomplete_fallback"
            decision = NextStep.model_validate(
                {
                    "current_state": "Investigation incomplete; no validated final model answer.",
                    "plan_remaining_steps_brief": ["Submit explicit incomplete fallback."],
                    "task_completed": True,
                    "function": {
                        "tool": "finish_investigation",
                        "submission": {
                            "verdict": "malicious",
                            "ir_artifacts": [],
                            "reasoning": FALLBACK_REASON,
                        },
                    },
                }
            )
            self._record(
                "deadline_fallback",
                task_id=session.task_id,
                reason=failures,
                decision=decision.model_dump(mode="json"),
            )
        if self._validate_decision(decision, schemas["finish_investigation"], store):
            raise CatalogCompatibilityError("No supported quality submission.")
        arguments = decision.function.model_dump(mode="json", exclude={"tool"}, exclude_none=True)
        budget.check()
        self._record(
            "step_selected",
            task_id=session.task_id,
            step=step,
            decision=decision.model_dump(mode="json"),
            decision_origin=origin,
        )
        self._record(
            "tool_call_started",
            task_id=session.task_id,
            step=step,
            tool_name="finish_investigation",
            arguments=arguments,
        )
        try:
            async with asyncio.timeout(budget.remaining_seconds()):
                response = await session.call_tool("finish_investigation", arguments)
        except BaseException as error:
            self._record(
                "tool_call_finished",
                task_id=session.task_id,
                step=step,
                tool_name="finish_investigation",
                status="error",
                error_type=type(error).__name__,
                terminal_state=getattr(session, "terminal_state", "unknown"),
            )
            raise
        self._record(
            "tool_result",
            task_id=session.task_id,
            step=step,
            response=response.model_dump(mode="json"),
        )
        self._record(
            "tool_call_finished",
            task_id=session.task_id,
            step=step,
            tool_name="finish_investigation",
            status=self._tool_feedback(response)["status"],
        )
        if response.task_result is None:
            raise RuntimeError("Final submission returned no authoritative task result.")
        self.provenance_checks.append(True)
        self._record("provenance_validated", task_id=session.task_id, step=step, valid=True)
        return TaskResult.model_validate(response.task_result)
