# Append/prune artifact notes for Paper II

This note records the current repository evidence for actual checkpoint append/prune events. It is manuscript-facing support only; do not cite file names in paper prose.

## Positive append-count artifacts found

The following artifacts contain actual append events, but they are **exact-assisted diagnostics**, not QPU-faithful controller evidence. They should not be used to support measurement-compatible append/prune claims.

| Artifact | Append count | Prune count | Final parameters / blocks | Metrics present | Decision-data status |
|---|---:|---:|---:|---|---|
| `artifacts/json/l2_anchor_std_high.json` | 88 | 0 | runtime parameters 278; logical blocks 124 | final energy error 3.4031; final fidelity 0.0176; final site-occupation max error 0.02223; spectral mean absolute error 0.1961 | exact-assisted: `mode=exact_v1`, `decision_backend=exact`, `exact_decision_checkpoints=201` |
| `artifacts/json/l2_anchor_std_low.json` | 80 | 0 | runtime parameters 267; logical blocks 116 | final energy error 2.9154; final fidelity 0.0857; final site-occupation max error 0.01395; spectral mean absolute error 0.1726 | exact-assisted: `mode=exact_v1`, `decision_backend=exact`, `exact_decision_checkpoints=201` |
| `artifacts/json/l2_anchor_std_real.json` | 78 | 0 | runtime parameters 274; logical blocks 114 | final energy error 3.3984; final fidelity 0.1205; final site-occupation max error 0.06524; spectral mean absolute error 0.1990 | exact-assisted: `mode=exact_v1`, `decision_backend=exact`, `exact_decision_checkpoints=201` |

## Positive prune-count artifact found

The current clean pruning evidence is the aggregate paired pilot in:

- `tmp/dyn_controller_cost_smoke/tab_dyn_ablation_matrix.json`
- `tmp/dyn_controller_cost_smoke/tables_summary.json`

Recorded pilot values:

- pruning enabled: prune count 9, final parameters 32, mean energy error `7.85982671584634e-6`;
- no-prune paired row: prune count 0, final parameters 56, mean energy error `7.860575173008284e-6`;
- prune-minus-no-prune delta: final parameters `-24`, mean energy-error change `-7.484571619448293e-10`.

The aggregate does not expose strict/QPU-faithful decision-contract fields for the pilot. Use it as a paired pruning diagnostic unless a stricter source artifact is located.

## Current manuscript-use rule

- Safe now: quantitative prune-pilot diagnostic sentence with the 9-deletion / 32-vs-56-parameter values.
- Not safe now: claiming a QPU-faithful append demonstration from the `l2_anchor_std_*` files.
- Needed for stronger Paper II claims: strict or measurement-compatible append/prune artifacts with `controller_exact_input_mode=off`, `diagnostic_exact_reference_mode=benchmark_exact`, no exact decision backend, and a passing strict decision contract.
