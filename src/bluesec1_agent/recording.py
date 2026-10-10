import json
import os
import sys
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from bluesec1_agent.contracts import EventSink


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
            "schema_version": 2,
            "timestamp": datetime.now(UTC).isoformat(),
            "event": self._redact(event),
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
        return redact(value, self._secrets)


def redact(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    """Redact registered credentials even in JSON keys and exception text."""
    if isinstance(value, BaseException):
        value = f"{type(value).__name__}: {value}"
    if isinstance(value, str):
        for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        entries = [(str(key), redact(str(key), secrets), item) for key, item in value.items()]
        original_keys = {original for original, _, _ in entries}
        reserved_keys = {safe for _, safe, _ in entries}
        result: dict[str, Any] = {}
        for original, safe, item in entries:
            key = safe
            if key in result or (original != safe and safe in original_keys):
                suffix = 2
                while f"{safe}#{suffix}" in reserved_keys or f"{safe}#{suffix}" in result:
                    suffix += 1
                key = f"{safe}#{suffix}"
            result[key] = redact(item, secrets)
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item, secrets) for item in value]
    return value


def sanitize_endpoint(endpoint: str) -> str:
    """Keep endpoint routing information, removing userinfo, query and fragment."""
    parts = urlsplit(endpoint)
    return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))


class SafeEventSink:
    """Best-effort telemetry; recording errors never replace investigation outcomes.

    The recorder itself remains strict for callers requiring durable reservations.
    Use RunLock for the fail-closed pre-CreateRun intent, never this adapter.
    """

    def __init__(self, recorder: EventSink | None, *, secrets: tuple[str, ...] = ()) -> None:
        self.recorder = recorder
        self.secrets = secrets
        self.context: dict[str, Any] = {}
        self.artifact_errors: list[dict[str, Any]] = []
        self._pending: list[dict[str, Any]] = []

    def update_context(self, **payload: Any) -> None:
        self.context.update(payload)

    def sanitize_error(self, error: BaseException) -> str:
        return redact(error, self.secrets)

    def _failed(self, operation: str, error: Exception) -> None:
        detail = {"operation": operation, "error_type": type(error).__name__}
        self.artifact_errors.append(detail)
        self._pending.append(detail)
        # Do not print filesystem paths, payloads, exception values or credentials.
        with suppress(Exception):
            print(f"artifact_error: {operation} ({type(error).__name__})", file=sys.stderr)

    def record(self, event: str, **payload: Any) -> None:
        if self.recorder is None:
            return
        try:
            while self._pending:
                self.recorder.record("artifact_error", **self._pending[0])
                self._pending.pop(0)
            self.recorder.record(
                redact(event, self.secrets), **redact({**self.context, **payload}, self.secrets)
            )
        except Exception as exc:
            self._failed("record", exc)

    def write_result(self, result: dict[str, Any]) -> None:
        if self.recorder is None:
            return
        value = redact({**result, "artifact_errors": self.artifact_errors}, self.secrets)
        try:
            self.recorder.write_result(value)
        except Exception as exc:
            self._failed("write_result", exc)
            directory = getattr(self.recorder, "directory", None)
            if directory is not None:
                try:
                    value["artifact_errors"] = self.artifact_errors
                    descriptor = os.open(
                        Path(directory) / "partial-result.json",
                        os.O_CREAT | os.O_TRUNC | os.O_WRONLY,
                        0o600,
                    )
                    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                        json.dump(value, stream, ensure_ascii=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                except Exception as fallback_error:
                    self._failed("partial_result", fallback_error)


def build_run_metadata(
    repository: Path,
    *,
    configuration: dict[str, Any],
    prompt: str,
    schema: dict[str, Any],
    secrets: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Capture reproducibility inputs without reading environment files or keys."""
    import hashlib
    import subprocess

    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repository), *args],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def clean(value: Any, key: str = "") -> Any:
        name = key.lower()
        if name in {"api_key", "token", "password", "secret", "authorization"} or name.endswith(
            ("_api_key", "_token", "_password", "_secret")
        ):
            return "[REDACTED]"
        if isinstance(value, dict):
            return redact({str(k): clean(v, str(k)) for k, v in value.items()}, secrets)
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str) and ("endpoint" in name or "url" in name):
            value = sanitize_endpoint(value)
        return redact(value, secrets)

    sources = sorted((repository / "src").rglob("*.py"))
    hashes = {
        str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
        if path.is_file() and not path.is_symlink()
    }
    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "source_hashes": hashes,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(
            json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest(),
        "configuration": clean(configuration),
    }
