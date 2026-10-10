# E offline handoff, cycle 01

This directory contains a **draft** preregistration, not authorization to run.
No LLM or BlueSec request occurred during its preparation. The accepted plan and
C/R gate remain the authority for the later phase.

## Commands

```
uv run pytest tests/test_evaluation.py -q
uv run python scripts/eval_offline.py --fixtures tests/fixtures/evaluation --output reports/experiments/cycle-01/offline.json
uv run python scripts/run_experiments.py --manifest reports/experiments/cycle-01/manifest.json --dry-run
uv run python scripts/compare_runs.py --manifest reports/experiments/cycle-01/manifest.json --output reports/experiments/cycle-01/comparison.json
```

## Frozen manifest contract

JSON schema version 1: `cycle_id`, `freeze_state`, `integrated_sha`, baseline/prepared
SHAs, `source_fingerprints`, `configurations`, `slots`, `limits`, `stop_policy`,
`selection_policy`, `runtime_state_dir`, `run_artifacts_dir`, `attempts_path`.
Slots 01–04 are v2 smoke tests in Luna/Astra/Opus/Sol order; 05–10 alternate
Luna baseline/v2; 11–16 alternate Astra/Opus/Sol v2 twice. The validator enforces
this fixed order and the limits 16 attempts, 1 task, 60 steps, 900 task seconds,
30 cleanup seconds, 24000 context characters. It also enforces the unchanged
selection policy. Statuses never belong to the manifest.

The current `integrated_sha=null` and `freeze_state=draft` intentionally prevent
live execution. C/E must replace the source/fixture fingerprints with hashes of
the final integrated revision, record effective sanitized provider/runtime
configuration and hidden-control hashes, then freeze. This draft's fingerprints
only identify the offline preparation content. Secrets stay in canonical `.env`;
review must verify the final checkout loads that configuration without copying
secrets into the manifest. A reviewed manifest is never changed during the series.

Only C writes `live-gate.json`, with:

```
{
  "live_ready": true,
  "integrated_sha": "<40-character integrated commit>",
  "manifest_sha256": "<64-character frozen manifest hash>",
  "reviewer": "R",
  "review_approved": true,
  "review_artifact": "<absolute immutable review path>",
  "review_artifact_sha256": "<review file SHA-256>",
  "coordinator_thread_id": "01a1251e-a7d3-72a2-bf6b-fe9ecaefb27d"
}
```

The launcher checks the exact checkout, manifest, review content hash and source
fingerprints before the series and before every attempt. Uncommitted/untracked
`src`, `scripts`, `pyproject.toml`, `uv.lock` block live; local `.env.example`,
ignored reports and `.venv` are allowed. CLI owns H's runtime `RunLock`; launcher
owns only `cycle-01-series.lock`, avoiding parent/child lock deadlock.

## Attempt ledger

The manifest pins one absolute canonical `attempts_path` across worktrees.
JSONL version 1 events contain `event_id`, UTC `timestamp`, `manifest_sha256`,
`event` and applicable `slot_id`. Sequence per slot: `reserved` (file+directory
fsync), `started` (fsync), `finished` with `summary`. No child/CreateRun is invoked
until those writes succeed. Failed create/lease still consumes the reserved slot.
Series terminates with `series_completed` or `series_stopped(reason)` followed by
`not_started` for remaining slots. Existing nonempty journals cannot be resumed
or overwritten by the launcher. A torn final line preserves earlier reservations
and is explicitly reported. Identical duplicate event IDs deduplicate; conflicting
IDs, duplicate reservations, unknown slots and hash mismatches fail closed.

## Summary contract and limits

`summarize_run(trace_path, result_path)` accepts legacy and v2 event envelopes.
It emits `accepted_final`, nullable `quality_score`, `efficiency_score`, `reward`,
`walltime_seconds`, `task_results`, run/task IDs, counters, usage completeness,
cleanup, provenance, error category, artifact errors, metadata and source hashes.
`llm_attempts` deduplicates UUID identities. `remote_tool_calls` counts distinct
v2 starts by `(task_id,step)`; legacy `tool_result` is the fallback, not an added
second count. Cache hits and schema repairs remain separate. Flat H usage and
nested usage are accepted; cached/reasoning tokens are subsets, not added to total.
Partial usage provides known sums and known-attempt counts while total remains
null. No money cost is invented without a provider tariff.

A canonical terminal score 0 remains 0. Missing task_result/JSON field remains
null. Protobuf nonoptional scalars cannot distinguish absent wire values from
explicit zero; the report states that limitation. Neither `ListFields` nor an
accepted final is used to invent metric presence or quality. Legacy
`cleanup_succeeded` does not establish `closed_confirmed`.

C top-level result fields: legacy run_id/successful_tasks/failed_tasks/task_results,
plus status, error_category (`auth|schema|infrastructure|task_incomplete|
unknown_run_state|null`), cleanup.status, artifact_errors, metadata and
provenance_verified. `provenance_validated {valid:true}` is also accepted, but
provenance means reference provenance, not verdict correctness.

`compare_runs.py` includes all 16 slots and all reserved failures, verifies raw
artifact hashes, recomputes available summaries, and shows accepted-only metric
values/medians alongside all-attempt missingness and noncompletion rates. No
precision/recall or p95 is fabricated. The policy proposes only the next experiment
candidate; it never changes the working default. Missing required data, all-zero
scores or a stopped series produce `investigate_then_collect_more`.

The 12 graph fixtures and their SDK read/replay contracts pass offline. Behavioral
A/H/C runs, hidden variants, the final integrated SHA, sanitized runtime settings,
R review and C LIVE_READY remain dependencies for the next phase. The phase is
not evidence of model accuracy or repaired integrated lifecycle behavior.

## Integrated-phase additions

`eval_offline.py --behavioral` executes all 12 scenario scripts through actual
A/H/C code and outputs checked invariants, failures, source commit plus byte
fingerprints and absolute per-case traces. It returns nonzero for any failed
check. Graph-only evaluation remains separately available.

Result aggregation now tolerates corrupt/non-file result.json, records artifact
read errors, falls back to partial-result.json, and salvages authoritative terminal
trace results including zero. Partial-result content has its own source hash.
Structured wrong_kind/unknown_object/etc. violations count as provenance failures;
runtime schema strings do not inflate that count.

The conservative stop policy also contains `capacity`. A definite creation rejection
reserves/consumes its attempt and stops with capacity/auth/schema as appropriate,
while cleanup remains failed/unknown (never invented closed).

Frozen gate now requires `sanitized_configuration`, nonempty `hidden_control_hashes`,
and absolute `canonical_env_file`. Source fingerprints must cover exactly every
Git-tracked file under src/ and scripts/, plus pyproject.toml and uv.lock, including
proto and stub files. Control-file hashes are checked before each attempt.

The E allowlist helper `sanitized_effective_configuration(Settings())` includes
provider/runtime endpoints, TLS, arena, default model, LLM retry/tokens/timeout,
agent/run name and SDK retry/deadline defaults. It never dumps Settings or secrets.
Launcher probes these values using `uv run --offline --locked --env-file <canonical>`
and compares with the frozen object before every attempt. Its child uses the same
env-file and repeats the comparison in-process before invoking CLI.main, so an
environment change between parent preflight and child start cannot silently alter
the experiment. No real configuration probe has been executed in E's offline phase.

Private control preparation tooling is ignored, unexecuted and withheld from A/C.
Actual variants/hashes and any hidden behavioral execution wait for C's final SHA
and freeze instruction. Keep the preceding draft source fingerprints separate from
that final freeze; this commit is not LIVE_READY.
