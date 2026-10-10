# Cycle 01 synthetic development fixtures

Each case has a complete `scenario.json` graph and a separate `expected.json`.
All identities, activities, labels and scores in these files are synthetic. The
expected label is an author-supplied fixture assumption, never BlueSec ground
truth. `unknown` means the small example intentionally does not prescribe a class.

`scenario.json` contains only public observation, a complete graph, origin metadata,
and explicit fault injection settings. `expected.json` contains graph cardinalities
and behavior invariants. Never include the latter in a model request. The graph
transport accepts only the former. These cases are development material, so they
cannot subsequently be described as a held-out model-quality test.

| Case | Purpose |
| --- | --- |
| 01 | Malicious outgoing child behind an ordinary process parent |
| 02 | Administrative command with positive authorization fields |
| 03 | Suspicious executable name without proof of class |
| 04 | 501 entities and exactly 500 outgoing relations |
| 05 | Directed two-node cycle |
| 06 | Entity/relation namespace confusion |
| 07 | A sibling's field cannot justify the trigger |
| 08 | Instructions and a forged ID inside untrusted data |
| 09 | Tool error must not promote evidence |
| 10 | Decision budget ends before a final response |
| 11 | Terminal response is lost |
| 12 | New required catalog field is incompatible |

`uv run python scripts/eval_offline.py --fixtures tests/fixtures/evaluation
--output reports/experiments/cycle-01/offline.json` checks complete graph integrity,
reads every entity/relation through an injected `RemoteTransport` and the real SDK,
and rejects unobserved partial replay actions as `unsupported_action`. It does not
run a model, compute classification accuracy, or assert that A/H invariants have
passed. Its output explicitly records the integrated behavioral driver dependency.
`tests/test_evaluation.py` independently tests aggregation, missingness, accounting,
stop conditions, dry-run isolation and recommendation policy.

After C freezes the integrated revision, E/R create hidden control variants in an
ignored directory, register their hashes before execution, and keep expectations
out of agent input. R receives the variants after freeze. If a variant is used for
prompt/code tuning, it becomes development data for the next cycle. No hidden
control has been generated or exposed during this offline preparation phase.

Origin: E, plan section 6, 2026-10-10; source base
`e29a16c4ae5b02d8f4d2e5247915210035d550fa`. Content fingerprints accompany the draft
manifest and offline result. Review all commits and failures; do not select only
successful cases.
