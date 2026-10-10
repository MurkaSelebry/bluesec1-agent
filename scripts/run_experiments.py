"""Run the fixed cycle only after an exact C/R gate, or print a network-free plan."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

from bluesec1_agent.evaluation import (
    AttemptJournal,
    execute_series,
    sha256_file,
    summarize_run,
    validate_live_gate,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]

# The child checks the exact effective Settings in the same process before CLI
# imports create any provider/runtime clients. Only allowlisted configuration is
# emitted by the separate preflight probe; neither command serializes secrets.
_CONFIG_PROBE = """
import json
from bluesec1_agent.evaluation import sanitized_effective_configuration
from bluesec1_agent.settings import Settings
try:
    print(json.dumps(sanitized_effective_configuration(Settings()), sort_keys=True))
except Exception:
    raise SystemExit('Cannot load effective configuration') from None
"""
_CHECKED_CLI = """
import json, sys
from pathlib import Path
from bluesec1_agent.evaluation import verify_effective_configuration
from bluesec1_agent.settings import Settings
manifest = json.loads(Path(sys.argv[1]).read_text())
try:
    verify_effective_configuration(manifest['sanitized_configuration'], Settings())
except Exception:
    raise SystemExit('Effective configuration refused before launch') from None
sys.argv = ['bluesec1-agent', *sys.argv[2:]]
from bluesec1_agent.cli import main
main()
"""


def _env_command(manifest: dict) -> list[str]:
    return [
        "uv",
        "run",
        "--offline",
        "--locked",
        "--env-file",
        manifest["canonical_env_file"],
        "--",
        sys.executable,
    ]


def probe_effective_configuration(manifest: dict) -> dict:
    """Read non-secret effective values under exactly the future child env-file."""
    try:
        output = subprocess.check_output(
            _env_command(manifest) + ["-c", _CONFIG_PROBE],
            cwd=ROOT,
            text=True,
            stderr=subprocess.PIPE,
        )
        return json.loads(output)
    except (OSError, ValueError, subprocess.CalledProcessError):
        raise ValueError("effective configuration probe failed") from None


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify_source(manifest: dict, digest: str, path: Path, gate_path: Path) -> None:
    if sha256_file(path) != digest:
        raise ValueError("frozen manifest changed during series")
    gate = json.loads(gate_path.read_text())
    validate_live_gate(manifest, digest, gate, _git("rev-parse", "HEAD"))
    review = Path(gate["review_artifact"])
    if not review.is_file() or sha256_file(review) != gate.get("review_artifact_sha256"):
        raise ValueError("review artifact hash mismatch")
    if gate.get("coordinator_thread_id") != "01a1251e-a7d3-72a2-bf6b-fe9ecaefb27d":
        raise ValueError("coordinator identity mismatch")
    dirty = _git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "src",
        "scripts",
        "pyproject.toml",
        "uv.lock",
    )
    if dirty:
        raise ValueError("uncommitted production source changes")
    fingerprints = manifest.get("source_fingerprints", {})
    source_paths = set(
        _git("ls-files", "--", "src", "scripts", "pyproject.toml", "uv.lock").splitlines()
    )
    if not source_paths or set(fingerprints) != source_paths:
        raise ValueError("frozen manifest must fingerprint all production files and scripts")
    for relative, fingerprint in fingerprints.items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT) or sha256_file(source) != fingerprint:
            raise ValueError(f"source fingerprint mismatch: {relative}")
    for control, fingerprint in manifest["hidden_control_hashes"].items():
        if not Path(control).is_absolute() or sha256_file(Path(control)) != fingerprint:
            raise ValueError("hidden control hash mismatch")
    if probe_effective_configuration(manifest) != manifest["sanitized_configuration"]:
        raise ValueError("effective configuration differs from frozen manifest")


def command_for(manifest: dict, slot: dict, output: Path, manifest_path: Path) -> list[str]:
    config = manifest["configurations"][slot["configuration_id"]]
    limits = manifest["limits"]
    command = _env_command(manifest) + [
        "-c",
        _CHECKED_CLI,
        str(manifest_path.resolve()),
        "--model",
        config["model"],
        "--strategy",
        config["strategy"],
        "--runtime-state-dir",
        manifest["runtime_state_dir"],
        "--output-dir",
        str(output),
    ]
    for flag in (
        "max_tasks",
        "max_steps",
        "task_timeout_seconds",
        "cleanup_timeout_seconds",
        "max_context_chars",
    ):
        command.extend(["--" + flag.replace("_", "-"), str(limits[flag])])
    return command


def _classify(summary: dict, returncode: int) -> str | None:
    if summary.get("error_category") is not None:
        return summary["error_category"]
    names = set(summary["errors"])
    if names & {"RemoteAuthenticationError", "AuthenticationError", "PermissionDeniedError"}:
        return "auth"
    if names & {"CatalogError", "CatalogCompatibilityError", "SchemaError"}:
        return "schema"
    if summary["cleanup_confirmed"] is not True:
        return "unknown_run_state"
    if returncode:
        return "infrastructure"
    if summary["accepted_final"] is not True:
        return "task_incomplete"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    digest = sha256_file(args.manifest)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "manifest_sha256": digest,
                    "freeze_state": manifest["freeze_state"],
                    "limits": manifest["limits"],
                    "slots": manifest["slots"],
                },
                indent=2,
            )
        )
        return 0
    gate_path = args.gate or args.manifest.parent / "live-gate.json"
    verify_source(manifest, digest, args.manifest, gate_path)
    state = Path(manifest["runtime_state_dir"])
    state.mkdir(parents=True, exist_ok=True)
    # Runtime RunLock is acquired ONLY by the child CLI. This independent lock
    # serializes launchers/attempt accounting and avoids parent-child deadlock.
    with (state / "cycle-01-series.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = AttemptJournal(Path(manifest["attempts_path"]), digest)

        def execute(slot: dict) -> dict:
            verify_source(manifest, digest, args.manifest, gate_path)
            output = Path(manifest["run_artifacts_dir"]) / slot["slot_id"]
            output.mkdir(parents=True, exist_ok=False)
            # Never copy provider responses or secrets into versioned experiment data.
            with (
                (output / "launcher.stdout").open("w") as stdout,
                (output / "launcher.stderr").open("w") as stderr,
            ):
                completed = subprocess.run(
                    command_for(manifest, slot, output, args.manifest),
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            directories = [p for p in output.iterdir() if p.is_dir()]
            if len(directories) != 1:
                return {
                    "accepted_final": None,
                    "cleanup_confirmed": None,
                    "error_category": "unknown_run_state",
                    "artifact_error": "expected_one_run_directory",
                    "output_path": str(output),
                    "exit_code": completed.returncode,
                }
            run = directories[0]
            summary = summarize_run(run / "trace.jsonl", run / "result.json")
            summary["error_category"] = _classify(summary, completed.returncode)
            summary["exit_code"] = completed.returncode
            summary["open_p1_p2"] = 0  # Gate is valid only after R has cleared blockers.
            return summary

        execute_series(manifest["slots"], journal, execute)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f"Experiment refused/stopped: {error}", file=sys.stderr)
        raise SystemExit(1) from error
