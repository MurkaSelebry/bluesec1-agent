import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY

import pytest
from pydantic import SecretStr

from bluesec1_agent import cli
from bluesec1_agent.agent import AgentRunResult


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        scenario_runtime_endpoint="runtime.example:443",
        scenario_runtime_token=SecretStr("runtime-setting-secret"),
        scenario_runtime_verify_tls=True,
        scenario_runtime_arena="practice",
        remote_run_label="baseline",
        agent_name="settings-agent",
        llm_base_url="https://llm.example/v1",
        llm_api_key=SecretStr("llm-secret"),
        llm_default_model="settings-model",
        llm_timeout_seconds=30.0,
        llm_max_completion_tokens=4096,
        llm_max_step_retries=2,
        agent_max_steps=80,
    )


@pytest.mark.parametrize("value", ["0", "-1", "101", "text"])
def test_parser_rejects_out_of_range_step_limits(value: str) -> None:
    """A bad CLI limit must fail before any remote run is created."""
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--max-steps", value])

    assert exit_info.value.code == 2


@pytest.mark.parametrize("value", ["0", "-1", "text"])
def test_parser_rejects_non_positive_task_limits(value: str) -> None:
    """A bounded run must always request at least one task."""
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--max-tasks", value])

    assert exit_info.value.code == 2


def test_parser_defaults_to_one_task_and_requires_explicit_full_queue() -> None:
    """A bare invocation must be bounded, while full-queue mode is an explicit choice."""
    parser = cli.build_parser()

    bounded = parser.parse_args([])
    unbounded = parser.parse_args(["--all-tasks"])

    assert bounded.max_tasks == 1
    assert bounded.all_tasks is False
    assert bounded.max_steps is None
    assert bounded.output_dir == Path("reports/runs")
    assert unbounded.all_tasks is True
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--max-tasks", "3", "--all-tasks"])
    assert exit_info.value.code == 2


def test_parser_selects_baseline_or_v2_with_bounded_execution(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args([
        "--strategy", "baseline", "--task-timeout-seconds", "900",
        "--cleanup-timeout-seconds", "30", "--max-context-chars", "24000",
        "--runtime-state-dir", str(tmp_path),
    ])
    assert args.strategy == "baseline"
    assert args.task_timeout_seconds == 900
    assert args.cleanup_timeout_seconds == 30
    assert args.max_context_chars == 24000
    assert args.runtime_state_dir == tmp_path


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "text"])
def test_task_deadline_must_be_finite_and_positive(value: str) -> None:
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(["--task-timeout-seconds", value])
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("task_args", "expected_max_tasks", "step_args", "expected_max_steps"),
    [
        (["--max-tasks", "3"], 3, ["--max-steps", "60"], 60),
        (["--all-tasks"], None, [], 80),
    ],
)
def test_run_records_lifecycle_after_cleanup_and_prints_artifact_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    task_args: list[str],
    expected_max_tasks: int | None,
    step_args: list[str],
    expected_max_steps: int,
) -> None:
    """A successful bounded run must persist its summary only after both clients close."""
    sequence: list[object] = []
    captured: dict[str, object] = {}

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            captured["llm_kwargs"] = kwargs

        async def __aenter__(self) -> "FakeLLM":
            sequence.append("llm_entered")
            return self

        async def __aexit__(self, *args: object) -> None:
            sequence.append("llm_exited")

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["client_kwargs"] = kwargs
            self.run = SimpleNamespace(run_id="run-123")

        async def __aenter__(self) -> "FakeClient":
            sequence.append("client_entered")
            return self

        async def __aexit__(self, *args: object) -> None:
            sequence.append("client_exited")

    class FakeRecorder:
        def __init__(self, output_dir: Path, *, secrets: tuple[str, ...]) -> None:
            captured["recorder_args"] = (output_dir, secrets)
            self.directory = output_dir / "run-artifacts"

        def record(self, event: str, **payload: object) -> None:
            sequence.append((event, payload))

        def write_result(self, result: dict[str, object]) -> None:
            sequence.append(("write_result", result))

    class FakeAgent:
        def __init__(self, llm: FakeLLM, **kwargs: object) -> None:
            captured["agent_args"] = (llm, kwargs)

        async def run(self, client: FakeClient) -> AgentRunResult:
            sequence.append("agent_ran")
            return AgentRunResult(
                run_id=client.run.run_id,
                successful_tasks=1,
                failed_tasks=0,
            )

    monkeypatch.setattr(cli, "_load_settings", _settings)
    monkeypatch.setattr(cli, "OpenAIStructuredStepClient", FakeLLM)
    monkeypatch.setattr(cli, "RemoteBenchmarkClient", FakeClient)
    monkeypatch.setattr(cli, "RunRecorder", FakeRecorder)
    monkeypatch.setattr(cli, "SimpleSGRAgent", FakeAgent)
    monkeypatch.setattr(cli.truststore, "inject_into_ssl", lambda: None)
    args = cli.build_parser().parse_args(
        [
            "--endpoint",
            "cli-runtime.example:443",
            "--token",
            "runtime-cli-secret",
            "--agent-name",
            "cli-agent",
            "--model",
            "cli-model",
            *task_args,
            *step_args,
            "--output-dir",
            str(tmp_path),
        ]
    )

    result = asyncio.run(cli.run(args))

    assert result.failed_tasks == 0
    assert captured["recorder_args"] == (
        tmp_path,
        ("runtime-cli-secret", "llm-secret"),
    )
    assert captured["llm_kwargs"] == {
        "base_url": "https://llm.example/v1",
        "api_key": "llm-secret",
        "model": "cli-model",
        "timeout_seconds": 30.0,
        "max_completion_tokens": 4096,
        "max_retries": 2,
    }
    client_kwargs = captured["client_kwargs"]
    assert isinstance(client_kwargs, dict)
    assert client_kwargs["endpoint"] == "cli-runtime.example:443"
    assert client_kwargs["token"] == "runtime-cli-secret"
    assert client_kwargs["agent_name"] == "cli-agent"
    assert client_kwargs["model_name"] == "cli-model"
    assert client_kwargs["metadata"] == {
        "agent": "cli-agent",
        "implementation": "structured-generation-scenario-runtime-grpc",
        "model": "cli-model",
    }
    agent_args = captured["agent_args"]
    assert isinstance(agent_args, tuple)
    assert agent_args[1] == {
        "max_steps": expected_max_steps,
        "max_tasks": expected_max_tasks,
        "recorder": ANY,
    }
    assert sequence == [
        "llm_entered",
        "client_entered",
        (
            "run_started",
            {
                "run_id": "run-123",
                "model": "cli-model",
                "agent_name": "cli-agent",
                "limits": {
                    "max_tasks": expected_max_tasks,
                    "max_steps": expected_max_steps,
                },
            },
        ),
        "agent_ran",
        "client_exited",
        "llm_exited",
        ("run_finished", {"run_id": "run-123", "cleanup_succeeded": True}),
        (
            "write_result",
            {
                "run_id": "run-123",
                "successful_tasks": 1,
                "failed_tasks": 0,
                "task_results": [],
            },
        ),
    ]
    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "run_id": "run-123",
        "successful_tasks": 1,
        "failed_tasks": 0,
        "task_results": [],
    }
    assert output.err == f"Artifacts: {tmp_path / 'run-artifacts'}\n"


@pytest.mark.parametrize("failure", [RuntimeError("provider failed"), KeyboardInterrupt()])
def test_run_records_minimal_failure_and_closes_contexts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
) -> None:
    """Provider failures and interrupts must close the remote run without logging secrets."""
    sequence: list[object] = []

    class FakeContext:
        def __init__(self, **kwargs: object) -> None:
            self.run = SimpleNamespace(run_id="run-failure")

        async def __aenter__(self) -> "FakeContext":
            sequence.append("entered")
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: object,
        ) -> None:
            sequence.append(("exited", exc_type))

    class FakeRecorder:
        def __init__(self, output_dir: Path, *, secrets: tuple[str, ...]) -> None:
            self.directory = output_dir / "failure-artifacts"

        def record(self, event: str, **payload: object) -> None:
            sequence.append((event, payload))

        def write_result(self, result: dict[str, object]) -> None:
            raise AssertionError("a failed run has no final result")

    class FailingAgent:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run(self, client: FakeContext) -> AgentRunResult:
            raise failure

    monkeypatch.setattr(cli, "_load_settings", _settings)
    monkeypatch.setattr(cli, "OpenAIStructuredStepClient", FakeContext)
    monkeypatch.setattr(cli, "RemoteBenchmarkClient", FakeContext)
    monkeypatch.setattr(cli, "RunRecorder", FakeRecorder)
    monkeypatch.setattr(cli, "SimpleSGRAgent", FailingAgent)
    monkeypatch.setattr(cli.truststore, "inject_into_ssl", lambda: None)
    args = cli.build_parser().parse_args(["--output-dir", str(tmp_path)])

    with pytest.raises(type(failure)):
        asyncio.run(cli.run(args))

    assert sequence[-3:] == [
        ("exited", type(failure)),
        ("exited", type(failure)),
        ("run_failed", {"error_type": type(failure).__name__}),
    ]
    output = capsys.readouterr()
    assert "runtime-setting-secret" not in output.err
    assert "llm-secret" not in output.err
    assert output.err == f"Artifacts: {tmp_path / 'failure-artifacts'}\n"


def test_main_exits_nonzero_after_a_persisted_failed_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed run with failed tasks must still report a failing process status."""

    async def fake_run(args: argparse.Namespace) -> AgentRunResult:
        return AgentRunResult(
            run_id="run-with-failure",
            successful_tasks=0,
            failed_tasks=1,
        )

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["bluesec1-agent"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 1
