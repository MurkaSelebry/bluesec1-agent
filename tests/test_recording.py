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


def test_redacts_keys_event_names_and_exceptions(tmp_path):
    recorder = RunRecorder(tmp_path, secrets=("synthetic-secret",))
    recorder.record("synthetic-secret", **{"synthetic-secret": ValueError("synthetic-secret")})
    raw = (recorder.directory / "trace.jsonl").read_text()
    assert "synthetic-secret" not in raw
    assert json.loads(raw)["schema_version"] == 2


def test_safe_sink_preserves_original_error_and_reports_artifact_failure(tmp_path, capsys):
    from bluesec1_agent.recording import SafeEventSink

    recorder = RunRecorder(tmp_path, secrets=("synthetic-secret",))
    sink = SafeEventSink(recorder, secrets=("synthetic-secret",))
    sink.update_context(run_id="run-1", task_id="task-1", step=2)
    trace = recorder.directory / "trace.jsonl"
    trace.mkdir()  # a real write failure; no mock of recorder semantics
    original = ValueError("original synthetic-secret")
    try:
        raise original
    except ValueError as caught:
        sink.record("task_error", error=caught)
        assert caught is original
    trace.rmdir()
    sink.record("recovered")
    sink.write_result({"status": "failed", "score": None})
    events = read_trace(recorder.directory)
    assert any(e["event"] == "artifact_error" for e in events)
    assert events[-1]["payload"]["run_id"] == "run-1"
    assert sink.artifact_errors
    result = json.loads((recorder.directory / "result.json").read_text())
    assert result["score"] is None
    assert result["artifact_errors"]
    assert "synthetic-secret" not in capsys.readouterr().err


def test_safe_sink_writes_partial_result_if_result_target_is_unwritable(tmp_path):
    from bluesec1_agent.recording import SafeEventSink

    recorder = RunRecorder(tmp_path)
    (recorder.directory / "result.json").mkdir()
    sink = SafeEventSink(recorder)
    sink.write_result({"status": "completed", "score": None})
    partial = json.loads((recorder.directory / "partial-result.json").read_text())
    assert partial["status"] == "completed"
    assert partial["artifact_errors"]


def test_run_metadata_hashes_source_and_sanitizes_configuration(tmp_path):
    import hashlib
    import subprocess

    from bluesec1_agent.recording import build_run_metadata

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "main.py"
    source.write_text("print('source')\n")
    (tmp_path / ".env").write_text("DO_NOT_READ=synthetic-secret")
    metadata = build_run_metadata(
        tmp_path,
        configuration={
            "api_key": "synthetic-secret",
            "endpoint": "https://u:p@llm.invalid/v1?token=secret",
            "max_steps": 60,
            "nested": {"token": "synthetic-secret"},
        },
        prompt="prompt",
        schema={"type": "object"},
    )
    assert "synthetic-secret" not in json.dumps(metadata)
    assert metadata["configuration"]["endpoint"] == "https://llm.invalid/v1"
    assert (
        metadata["source_hashes"]["src/main.py"] == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert ".env" not in metadata["source_hashes"]
    assert metadata["dirty"] is True
    assert metadata["commit"] is None
