import argparse
import asyncio
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import truststore
from pydantic import ValidationError

from bluesec1_agent.agent import SYSTEM_PROMPT, AgentRunResult, SimpleSGRAgent
from bluesec1_agent.catalog import CatalogCompatibilityError
from bluesec1_agent.llm import (
    STRUCTURED_OUTPUT_INSTRUCTION,
    NextStepRequestError,
    OpenAIStructuredStepClient,
)
from bluesec1_agent.models import NEXT_STEP_RESPONSE_FORMAT
from bluesec1_agent.recording import RunRecorder, SafeEventSink, build_run_metadata, redact
from bluesec1_agent.run_lock import RunLock, RunLockError
from bluesec1_agent.settings import Settings
from bluesec1_client import (
    CompetitionClosedError,
    RemoteAuthenticationError,
    RemoteBenchmarkClient,
    RemoteRateLimitError,
    RemoteUnavailableError,
    RunCapacityExceededError,
)
from bluesec1_client.remote.errors import RemoteAuthorizationError, RemoteOutcomeUnknownError


def build_parser() -> argparse.ArgumentParser:
    """Build the remote-runtime command-line interface.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run the structured-generation agent against scenario-runtime."
    )
    parser.add_argument("--endpoint", help="Override SCENARIO_RUNTIME_ENDPOINT.")
    parser.add_argument("--token", help="Override SCENARIO_RUNTIME_TOKEN.")
    parser.add_argument("--run-label", help="Override REMOTE_RUN_LABEL.")
    parser.add_argument("--agent-name", help="Override AGENT_NAME.")
    parser.add_argument("--arena", help="Override SCENARIO_RUNTIME_ARENA.")
    parser.add_argument("--model", help="Override LLM_DEFAULT_MODEL.")
    parser.add_argument("--strategy", choices=("baseline", "v2"), default=None)
    parser.add_argument("--task-timeout-seconds", type=_positive_float, default=None)
    parser.add_argument("--cleanup-timeout-seconds", type=_positive_float, default=None)
    parser.add_argument("--max-context-chars", type=_positive_int, default=None)
    parser.add_argument("--runtime-state-dir", type=Path, default=None)
    task_limit = parser.add_mutually_exclusive_group()
    task_limit.add_argument(
        "--max-tasks",
        type=_positive_int,
        default=1,
        help="Maximum tasks to investigate in this run (default: 1).",
    )
    task_limit.add_argument(
        "--all-tasks",
        action="store_true",
        help="Consume the full task queue instead of using a task limit.",
    )
    parser.add_argument(
        "--max-steps",
        type=_step_limit,
        help="Maximum investigation steps per task (1-100; default: configured value).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/runs"),
        help="Parent directory for run artifacts (default: reports/runs).",
    )
    parser.add_argument(
        "--verify-tls",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override SCENARIO_RUNTIME_VERIFY_TLS.",
    )
    return parser


def _runtime_state_directory(repository: Path) -> Path:
    """All linked worktrees resolve to the same account run lock by default."""
    try:
        common = subprocess.check_output(
            [
                "git",
                "-C",
                str(repository),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return Path(common).parent / "reports" / "goals" / "runtime-state"
    except (OSError, subprocess.CalledProcessError):
        return repository / "reports" / "goals" / "runtime-state"


def _error_category(error: BaseException | None) -> str | None:
    if error is None:
        return None
    if isinstance(error, (RemoteOutcomeUnknownError, RunLockError)):
        return "unknown_run_state"
    if isinstance(error, (RemoteAuthenticationError, RemoteAuthorizationError)):
        return "auth"
    if isinstance(error, CatalogCompatibilityError):
        return "schema"
    if isinstance(error, NextStepRequestError):
        category = getattr(error, "category", None)
        if category in {"http_401", "http_403"}:
            return "auth"
        if category in {"schema", "truncated", "empty_response", "http_400"}:
            return "schema"
    return "infrastructure"


async def run(args: argparse.Namespace) -> AgentRunResult:
    """Run with shared durable ownership and record the received cleanup outcome."""
    truststore.inject_into_ssl()
    settings = _load_settings()
    repository = Path(__file__).resolve().parents[2]
    agent_name = args.agent_name or settings.agent_name
    model = args.model or settings.llm_default_model
    max_steps = args.max_steps if args.max_steps is not None else settings.agent_max_steps
    max_tasks = None if args.all_tasks else args.max_tasks
    strategy = args.strategy or settings.agent_strategy
    task_timeout = args.task_timeout_seconds or settings.task_timeout_seconds
    cleanup_timeout = args.cleanup_timeout_seconds or settings.cleanup_timeout_seconds
    max_context = args.max_context_chars or settings.max_context_chars
    runtime_state = (
        args.runtime_state_dir or settings.runtime_state_dir or _runtime_state_directory(repository)
    )
    # Validate local controls before allocating any remote resources.
    if max_context < 1000:
        raise ValueError("max_context_chars must be at least 1000")
    runtime_token = args.token or settings.scenario_runtime_token.get_secret_value()
    llm_api_key = settings.llm_api_key.get_secret_value()
    secrets = (runtime_token, llm_api_key)
    verify_tls = (
        settings.scenario_runtime_verify_tls if args.verify_tls is None else args.verify_tls
    )
    recorder = RunRecorder(args.output_dir, secrets=secrets)
    sink = SafeEventSink(recorder, secrets=secrets)
    configuration = {
        "model": model,
        "strategy": strategy,
        "max_tasks": max_tasks,
        "max_steps": max_steps,
        "task_timeout_seconds": task_timeout,
        "cleanup_timeout_seconds": cleanup_timeout,
        "max_context_chars": max_context,
        "max_llm_attempts": settings.llm_max_step_retries,
        "llm_timeout_seconds": settings.llm_timeout_seconds,
        "max_completion_tokens": settings.llm_max_completion_tokens,
        "runtime_endpoint": args.endpoint or settings.scenario_runtime_endpoint,
        "arena": args.arena or settings.scenario_runtime_arena,
        "llm_base_url": settings.llm_base_url,
        "verify_tls": verify_tls,
    }
    metadata = build_run_metadata(
        repository,
        configuration=configuration,
        prompt=SYSTEM_PROMPT + "\n\n" + STRUCTURED_OUTPUT_INSTRUCTION,
        schema=NEXT_STEP_RESPONSE_FORMAT,
        secrets=secrets,
    )
    sink.record("run_prepared", metadata=metadata)
    client = None
    agent = None
    result = None
    failure: BaseException | None = None
    try:
        with RunLock(runtime_state) as lock:

            def observe(event: str, payload: Any) -> None:
                lock.observe(event, payload)
                if event in {"cleanup_started", "cleanup_finished"}:
                    sink.record(event, **asdict(payload))
                elif event == "create_started":
                    sink.record(event, idempotency_key=payload.idempotency_key, run_id=None)
                elif event == "create_finished":
                    sink.update_context(run_id=payload.run_id)
                    sink.record(event, run_id=payload.run_id, state=payload.state)
                elif event == "create_failed":
                    sink.record(event, error_type=type(payload).__name__)

            llm = OpenAIStructuredStepClient(
                base_url=settings.llm_base_url,
                api_key=llm_api_key,
                model=model,
                timeout_seconds=settings.llm_timeout_seconds,
                max_completion_tokens=settings.llm_max_completion_tokens,
                max_retries=settings.llm_max_step_retries,
                event_sink=sink,
            )
            # LLM enters first so constructor/entry failures still close its resources.
            async with llm:
                client = RemoteBenchmarkClient(
                    endpoint=configuration["runtime_endpoint"],
                    token=runtime_token,
                    verify_tls=verify_tls,
                    run_label=args.run_label or settings.remote_run_label,
                    agent_name=agent_name,
                    model_name=model,
                    arena=args.arena or settings.scenario_runtime_arena,
                    metadata={
                        "agent": agent_name,
                        "implementation": "structured-generation-scenario-runtime-grpc",
                        "model": model,
                    },
                    lifecycle_observer=observe,
                    cleanup_timeout_seconds=cleanup_timeout,
                )
                async with client:
                    if client.run is None:
                        raise RuntimeError("Remote runtime did not provide an active run.")
                    sink.update_context(run_id=client.run.run_id)
                    sink.record(
                        "run_started",
                        model=model,
                        agent_name=agent_name,
                        limits={"max_tasks": max_tasks, "max_steps": max_steps},
                        strategy=strategy,
                    )
                    agent = SimpleSGRAgent(
                        llm,
                        max_steps=max_steps,
                        max_tasks=max_tasks,
                        recorder=sink,
                        strategy=strategy,
                        max_context_chars=max_context,
                        task_timeout_seconds=task_timeout,
                    )
                    result = await agent.run(client)
    except BaseException as error:
        failure = error
        sink.record("run_failed", error_type=type(error).__name__)
    finally:
        cleanup = getattr(client, "cleanup_outcome", None)
        cleanup_payload = asdict(cleanup) if cleanup is not None else None
        category = _error_category(failure or getattr(agent, "last_error", None))
        if getattr(agent, "remote_outcome_unknown", False):
            category = "unknown_run_state"
        if cleanup is not None and (
            cleanup.status in {"unknown", "closing_acknowledged"}
            or (cleanup.status == "failed" and cleanup.run_id is not None)
        ):
            category = "unknown_run_state"
        if category is None and result is not None and result.failed_tasks:
            category = "task_incomplete"
        task_results = getattr(agent, "task_results", [])
        payload = (
            result.model_dump(mode="json")
            if result is not None
            else {
                "run_id": sink.context.get("run_id"),
                "task_results": [item.model_dump(mode="json") for item in task_results],
                "successful_tasks": sum(
                    item.completion_reason == "terminated" for item in task_results
                ),
                "failed_tasks": sum(
                    item.completion_reason != "terminated" for item in task_results
                ),
            }
        )
        checks = getattr(agent, "provenance_checks", [])
        successful = payload["successful_tasks"]
        payload["provenance_verified"] = (
            all(checks) if successful and len(checks) == successful else None
        )
        payload.update(
            status="completed" if failure is None and category is None else "failed",
            error_category=category,
            cleanup=cleanup_payload,
            metadata=metadata,
            error_type=type(failure).__name__ if failure is not None else None,
        )
        sink.record("run_finished", status=payload["status"], cleanup=cleanup_payload)
        sink.write_result(payload)
        payload["artifact_errors"] = sink.artifact_errors
        print(json.dumps(redact(payload, secrets), indent=2, sort_keys=True))
        print(f"Artifacts: {recorder.directory}", file=sys.stderr)
    if failure is not None:
        raise failure
    if cleanup is not None and cleanup.status != "closed_confirmed":
        raise RemoteOutcomeUnknownError("Remote cleanup was not confirmed; see run artifacts.")
    if result is None:
        raise RuntimeError("No authoritative run result was received.")
    return result


def main() -> None:
    """Parse arguments and execute the async remote agent."""
    try:
        result = asyncio.run(run(build_parser().parse_args()))
    except RunCapacityExceededError as error:
        print(_run_capacity_message(error), file=sys.stderr)
        raise SystemExit(1) from error
    except CompetitionClosedError as error:
        print(
            "The selected competition arena is closed. Check SCENARIO_RUNTIME_ARENA or retry "
            "when the arena opens.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    except RemoteRateLimitError as error:
        print(
            "Run starts are temporarily rate-limited. Wait before starting another run.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    except RemoteAuthenticationError as error:
        print(
            "The runtime rejected your key. Check SCENARIO_RUNTIME_TOKEN in .env.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    except RemoteUnavailableError as error:
        print(
            "Cannot reach the runtime. Check SCENARIO_RUNTIME_ENDPOINT and "
            "SCENARIO_RUNTIME_VERIFY_TLS in .env.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    except KeyboardInterrupt as error:
        print(
            "Agent run interrupted; cleanup was attempted. See the run artifacts.",
            file=sys.stderr,
        )
        raise SystemExit(130) from error
    except Exception as error:
        print(f"Agent run failed ({type(error).__name__}). See the run artifacts.", file=sys.stderr)
        raise SystemExit(1) from error

    if result.failed_tasks > 0:
        raise SystemExit(1)


def _positive_int(value: str) -> int:
    """Parse a positive integer for a bounded task count."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive finite number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _step_limit(value: str) -> int:
    """Parse a per-task step limit accepted by the agent."""
    parsed = _positive_int(value)
    if parsed > 100:
        raise argparse.ArgumentTypeError("must be between 1 and 100")
    return parsed


def _load_settings() -> Settings:
    """Load settings and explain what is missing instead of raising a raw error.

    Returns:
        Validated settings.
    """
    try:
        return Settings()
    except ValidationError as error:
        missing = [str(item["loc"][0]) for item in error.errors() if item["type"] == "missing"]
        if not missing:
            raise
        print("Missing configuration: " + ", ".join(missing), file=sys.stderr)
        print(
            "Copy .env.example to .env and fill it in. "
            "Runtime address and key: https://bluesec.team/",
            file=sys.stderr,
        )
        raise SystemExit(1) from error


def _run_capacity_message(error: RunCapacityExceededError) -> str:
    """Explain which run limit was hit and what to do about it.

    Args:
        error: Capacity failure reported by the runtime.

    Returns:
        Message for the participant.
    """
    if getattr(error, "reason", None) == "PROCESS_ACTIVE_RUN_LIMIT_REACHED":
        lines = ["The runtime is at its overall capacity right now. This is not your limit."]
    else:
        lines = [
            "You have reached your limit of concurrent runs.",
            "Wait for a running one to finish. If an agent was killed mid-run, its run stays "
            "open until it is released.",
        ]
    return "\n".join(lines)
