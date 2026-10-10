from pathlib import Path

import pytest

from bluesec1_agent import cli
from bluesec1_agent.settings import Settings
from bluesec1_client import RunCapacityExceededError


@pytest.fixture(autouse=True)
def isolate_configuration(monkeypatch, tmp_path):
    """Settings tests must not inherit the developer's process or .env secrets."""
    for field in Settings.model_fields.values():
        if isinstance(field.validation_alias, str):
            monkeypatch.delenv(field.validation_alias, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", tmp_path / "absent.env")


def test_settings_accept_explicit_runtime_and_provider_values() -> None:
    """Explicit values should satisfy all required credentials."""
    settings = Settings(
        scenario_runtime_endpoint="runtime.example:443",
        scenario_runtime_token="runtime-secret",
        scenario_runtime_arena="practice",
        llm_base_url="https://llm.example/v1",
        llm_api_key="llm-secret",
        llm_default_model="model-1",
    )

    assert settings.scenario_runtime_endpoint == "runtime.example:443"
    assert settings.scenario_runtime_token.get_secret_value() == "runtime-secret"
    assert settings.scenario_runtime_arena == "practice"
    assert settings.llm_base_url == "https://llm.example/v1"
    assert settings.llm_api_key.get_secret_value() == "llm-secret"
    assert settings.llm_default_model == "model-1"


def test_env_example_contains_complete_runnable_configuration() -> None:
    """The example must be parseable, including with placeholder credentials."""
    env_example = Path(__file__).resolve().parents[1] / ".env.example"

    settings = Settings(_env_file=env_example)

    assert settings.scenario_runtime_endpoint
    assert settings.scenario_runtime_verify_tls is True
    assert settings.scenario_runtime_arena is None
    assert settings.llm_base_url
    assert settings.llm_api_key.get_secret_value()
    assert settings.scenario_runtime_token.get_secret_value()
    assert settings.llm_default_model
    assert settings.agent_max_steps == 100


def test_settings_load_synthetic_env_without_user_secrets(tmp_path) -> None:
    env_file = tmp_path / "offline.env"
    env_file.write_text(
        "SCENARIO_RUNTIME_ENDPOINT=runtime.invalid:443\n"
        "SCENARIO_RUNTIME_TOKEN=offline-runtime-secret\n"
        "LLM_BASE_URL=https://provider.invalid/v1\n"
        "LLM_API_KEY=offline-llm-secret\n"
        "LLM_DEFAULT_MODEL=offline-model\n"
        "LLM_TIMEOUT_SECONDS=17\n"
        "BLUESEC_AGENT_MAX_STEPS=12\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=env_file)
    assert settings.llm_default_model == "offline-model"
    assert settings.llm_api_key.get_secret_value() == "offline-llm-secret"
    assert settings.scenario_runtime_token.get_secret_value() == "offline-runtime-secret"
    assert settings.llm_timeout_seconds == 17
    assert settings.agent_max_steps == 12


def test_missing_configuration_is_reported_without_traceback(monkeypatch, capsys):
    for name in ("SCENARIO_RUNTIME_TOKEN", "LLM_BASE_URL", "LLM_API_KEY", "LLM_DEFAULT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", Path("/nonexistent/.env"))

    with pytest.raises(SystemExit) as exit_info:
        cli._load_settings()

    assert exit_info.value.code == 1
    message = capsys.readouterr().err
    assert "Missing configuration" in message
    assert "SCENARIO_RUNTIME_TOKEN" in message
    assert "Traceback" not in message


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("SUBJECT_ACTIVE_RUN_LIMIT_REACHED", "your limit of concurrent runs"),
        ("PROCESS_ACTIVE_RUN_LIMIT_REACHED", "not your limit"),
    ],
)
def test_run_capacity_message_depends_on_reason(reason: str, expected: str) -> None:
    error = RunCapacityExceededError("capacity reached", reason=reason)

    message = cli._run_capacity_message(error)

    assert expected in message
    assert "One run per participant" not in message
