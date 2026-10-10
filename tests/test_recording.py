import json
from datetime import datetime
from pathlib import Path

from bluesec1_agent.recording import RunRecorder


def read_trace(directory: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]


def test_record_appends_events_that_are_immediately_persisted(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path)

    recorder.record("run_started", run_id="run-1")
    assert [entry["event"] for entry in read_trace(recorder.directory)] == ["run_started"]

    recorder.record("step_completed", step=1)
    entries = read_trace(recorder.directory)

    assert [entry["event"] for entry in entries] == ["run_started", "step_completed"]
    assert entries[0]["payload"] == {"run_id": "run-1"}
    assert entries[1]["payload"] == {"step": 1}
    assert all(
        datetime.fromisoformat(str(entry["timestamp"])).utcoffset().total_seconds() == 0
        for entry in entries
    )


def test_record_and_result_redact_secrets_recursively(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path, secrets=("token-123", ""))

    recorder.record(
        "request_failed",
        headers={"authorization": "Bearer token-123"},
        errors=["upstream rejected token-123", {"detail": "token-123 expired"}],
    )
    recorder.write_result(
        {
            "status": "failed",
            "error": "credential token-123 was rejected",
            "nested": {"messages": ["safe", "token-123"]},
        }
    )

    trace_text = (recorder.directory / "trace.jsonl").read_text()
    result_text = (recorder.directory / "result.json").read_text()
    assert "token-123" not in trace_text
    assert "token-123" not in result_text
    assert read_trace(recorder.directory)[0]["payload"] == {
        "headers": {"authorization": "Bearer [REDACTED]"},
        "errors": [
            "upstream rejected [REDACTED]",
            {"detail": "[REDACTED] expired"},
        ],
    }
    assert json.loads(result_text)["nested"] == {"messages": ["safe", "[REDACTED]"]}


def test_each_recorder_gets_a_unique_run_directory(tmp_path: Path) -> None:
    first = RunRecorder(tmp_path)
    second = RunRecorder(tmp_path)

    assert first.directory != second.directory
    assert first.directory.parent == tmp_path
    assert second.directory.parent == tmp_path
    assert first.directory.is_dir()
    assert second.directory.is_dir()


def test_write_result_creates_pretty_utf8_json(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path)
    result = {"verdict": "benign", "summary": "Проверено", "score": 0.75}

    recorder.write_result(result)

    result_path = recorder.directory / "result.json"
    assert json.loads(result_path.read_text(encoding="utf-8")) == result
    assert result_path.read_text(encoding="utf-8") == (
        '{\n  "verdict": "benign",\n  "summary": "Проверено",\n  "score": 0.75\n}\n'
    )
