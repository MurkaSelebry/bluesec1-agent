"""Reproduce all-attempt accounting from an immutable manifest and its event ledger."""

import argparse
import json
from pathlib import Path

from bluesec1_agent.evaluation import (
    compare_attempts,
    reconcile_attempts,
    sha256_file,
    summarize_run,
    validate_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    digest = sha256_file(args.manifest)
    accounting = reconcile_attempts(manifest["slots"], Path(manifest["attempts_path"]), digest)
    for row in accounting["attempts"]:
        if row.get("trace_path"):
            summary = summarize_run(
                Path(row["trace_path"]),
                Path(row["result_path"]) if row.get("result_path") else None,
            )
            if summary["source_sha256"] != row.get("source_sha256"):
                raise ValueError(f"source artifact changed for slot {row['slot_id']}")
            # Independently recompute metrics; do not trust an edited journal summary.
            row.update(summary)
            row["status"] = "finished"
    report = compare_attempts(manifest, accounting)
    report.update(manifest_sha256=digest, integrated_sha=manifest.get("integrated_sha"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(str(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
