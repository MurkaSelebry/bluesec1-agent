import asyncio
import json

import pytest

from bluesec1_agent import cli
from bluesec1_agent.agent import AgentRunResult
from bluesec1_agent.run_lock import RunLockError
from bluesec1_agent.settings import Settings
from bluesec1_client import RemoteBenchmarkClient
from bluesec1_client.remote.errors import RemoteOutcomeUnknownError
from bluesec1_client.remote.models import CloseRunOutcome, RemoteRun


@pytest.mark.parametrize("close_state", ["closed", "closing"])
def test_cli_composes_durable_lock_sdk_cleanup_and_result(monkeypatch, tmp_path, close_state):
    state = tmp_path / "state"
    creates = []

    class Transport:
        async def create_run(self, command):
            intent = json.loads((state / "intent.json").read_text())
            assert intent["idempotency_key"] == command.idempotency_key
            assert intent["run_id"] is None
            creates.append(command)
            return RemoteRun(run_id="test-run", state="active")

        async def close_run(self, command):
            assert json.loads((state / "intent.json").read_text())["run_id"] == "test-run"
            return CloseRunOutcome(run_id="test-run", state=close_state)

    class LLM:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, client):
            return AgentRunResult(run_id=client.run.run_id, successful_tasks=0, failed_tasks=0)

    settings = Settings(
        _env_file=None,
        scenario_runtime_token="synthetic-runtime",
        llm_api_key="synthetic-llm",
        llm_base_url="https://invalid.example/v1",
        llm_default_model="test-model",
    )
    monkeypatch.setattr(cli, "_load_settings", lambda: settings)
    monkeypatch.setattr(cli.truststore, "inject_into_ssl", lambda: None)
    monkeypatch.setattr(cli, "OpenAIStructuredStepClient", LLM)
    monkeypatch.setattr(cli, "SimpleSGRAgent", Agent)
    monkeypatch.setattr(
        cli,
        "RemoteBenchmarkClient",
        lambda **kwargs: RemoteBenchmarkClient(
            transport=Transport(),
            lifecycle_observer=kwargs["lifecycle_observer"],
            cleanup_timeout_seconds=kwargs["cleanup_timeout_seconds"],
        ),
    )
    args = cli.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "runs"),
            "--runtime-state-dir",
            str(state),
        ]
    )
    if close_state == "closing":
        with pytest.raises(RemoteOutcomeUnknownError):
            asyncio.run(cli.run(args))
        with pytest.raises(RunLockError):
            asyncio.run(cli.run(args))
        assert len(creates) == 1
        assert (state / "intent.json").exists()
    else:
        asyncio.run(cli.run(args))
        assert not (state / "intent.json").exists()
    results = [json.loads(path.read_text()) for path in (tmp_path / "runs").glob("*/result.json")]
    first = next(result for result in results if result["run_id"] == "test-run")
    assert first["cleanup"]["status"] == (
        "closed_confirmed" if close_state == "closed" else "closing_acknowledged"
    )
    assert first["error_category"] == (None if close_state == "closed" else "unknown_run_state")
