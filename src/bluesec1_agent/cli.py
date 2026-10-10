import argparse
import asyncio
import json
import sys
from pathlib import Path

import truststore
from pydantic import ValidationError

from bluesec1_agent.agent import AgentRunResult, SimpleSGRAgent
from bluesec1_agent.llm import OpenAIStructuredStepClient
from bluesec1_agent.recording import RunRecorder
from bluesec1_agent.settings import Settings
from bluesec1_client import (
    CompetitionClosedError,
    RemoteAuthenticationError,
    RemoteBenchmarkClient,
    RemoteRateLimitError,
    RemoteUnavailableError,
    RunCapacityExceededError,
)


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


async def run(args: argparse.Namespace) -> AgentRunResult:
    """Execute one new remote run and print its task results.

    Args:
        args: Parsed command-line arguments.
    """
    truststore.inject_into_ssl()
    settings = _load_settings()
    agent_name = args.agent_name or settings.agent_name
    model = args.model or settings.llm_default_model
    max_steps = args.max_steps if args.max_steps is not None else settings.agent_max_steps
    max_tasks = None if args.all_tasks else args.max_tasks
    runtime_token = args.token or settings.scenario_runtime_token.get_secret_value()
    llm_api_key = settings.llm_api_key.get_secret_value()
    verify_tls = (
        settings.scenario_runtime_verify_tls if args.verify_tls is None else args.verify_tls
    )
    recorder = RunRecorder(args.output_dir, secrets=(runtime_token, llm_api_key))

    try:
        llm = OpenAIStructuredStepClient(
            base_url=settings.llm_base_url,
            api_key=llm_api_key,
            model=model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_completion_tokens=settings.llm_max_completion_tokens,
            max_retries=settings.llm_max_step_retries,
        )
        client = RemoteBenchmarkClient(
            endpoint=args.endpoint or settings.scenario_runtime_endpoint,
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
        )
        async with llm, client:
            if client.run is None:
                raise RuntimeError("Remote runtime did not provide an active run.")
            recorder.record(
                "run_started",
                run_id=client.run.run_id,
                model=model,
                agent_name=agent_name,
                limits={"max_tasks": max_tasks, "max_steps": max_steps},
            )
            result = await SimpleSGRAgent(
                llm,
                max_steps=max_steps,
                max_tasks=max_tasks,
                recorder=recorder,
            ).run(client)
    except BaseException as error:
        recorder.record("run_failed", error_type=type(error).__name__)
        print(f"Artifacts: {recorder.directory}", file=sys.stderr)
        raise

    result_payload = result.model_dump(mode="json")
    recorder.record("run_finished", run_id=result.run_id, cleanup_succeeded=True)
    recorder.write_result(result_payload)
    print(json.dumps(result_payload, indent=2, sort_keys=True))
    print(f"Artifacts: {recorder.directory}", file=sys.stderr)
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
