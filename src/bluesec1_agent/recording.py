import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RunRecorder:
    """Persist a bounded agent run as redacted local JSON artifacts."""

    def __init__(self, output_dir: Path, *, secrets: tuple[str, ...] = ()) -> None:
        output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        prefix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ-")
        self.directory = Path(tempfile.mkdtemp(prefix=prefix, dir=output_dir))
        self.directory.chmod(0o700)
        self._secrets = tuple(
            sorted({secret for secret in secrets if secret}, key=len, reverse=True)
        )

    def record(self, event: str, **payload: Any) -> None:
        """Append one redacted event to the run trace and flush it immediately."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "payload": self._redact(payload),
        }
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
        trace_path = self.directory / "trace.jsonl"
        descriptor = os.open(trace_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as trace_file:
            trace_file.write(line)
            trace_file.flush()

    def write_result(self, result: dict[str, Any]) -> None:
        """Atomically replace the run result with redacted, pretty JSON."""
        content = json.dumps(self._redact(result), ensure_ascii=False, indent=2) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(prefix=".result-", dir=self.directory)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as result_file:
                result_file.write(content)
                result_file.flush()
                os.fsync(result_file.fileno())
            os.replace(temporary_path, self.directory / "result.json")
        finally:
            temporary_path.unlink(missing_ok=True)

    def _redact(self, value: Any) -> Any:
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {key: self._redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._redact(item) for item in value)
        return value
