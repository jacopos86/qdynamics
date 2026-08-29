# Paper-III Peierls--Hubbard and SIAM generality evidence

## Result

The fresh cross-family campaign completed on source commit
`7245d26eabb0435fe42bd2742e9e33c91a70f3f3`.  The locked selected-QSE route
certified the six-root window in all three Peierls--Hubbard regimes and in the
non-smoke ten-qubit SIAM case (`N_b=4`).  Direct Davidson also certified every
case.  No selected-QSE-to-Davidson fallback exists in this campaign.

The full machine-readable artifact is
`output/diagnostics/paper_iii_qse_generality_20260829_v1/generality_evidence.json`
(SHA-256
`50572d7cdb4bdc003e44329d2e689f975f76ce1099aa075b8fb43820ffd230ef`).
Its adjacent `run_receipt.json` locks the source revision, case list, and
artifact digest.

## Locked scientific contract

- Selector: `full_pencil_ky_fan_cost_ratio_novelty_floor_no_reward_v1`.
- Online cost: `K =` graph-span two-qubit count.
- Novelty remains a hard floor; there is no additive novelty reward.
- Every recorded selected-QSE prefix receives certified atomic exchange.
- SIAM uses the declared
  `siam_physical_plus_generalized_singles_paired_doubles_v1` pool.  The pool
  is derived from generalized spin-preserving one-body directions and paired
  cross-spin doubles; it does not use exact excited states.
- Direct Davidson is separately initialized from low-diagonal computational
  probes and uses synthesized residual directions.  It is not an implicit
  fallback inside selected QSE.
- Exact ground states isolate the static-QSE layer.  Exact excited roots and
  vectors are post-run audit data only.

## Matched `10^-6` energy-window crossing

`N2q(graph)` is the online graph-span coordinate.  `N2q(Qiskit)` is the
offline FakeMarrakesh transpilation coordinate.  Estimator queries use the
declared record-pencil QWC cover for selected QSE and the declared
synthesized-basis pencil convention for Davidson.

| case | route | growth dimension | retained rank | max six-root error | N2q(graph) | N2q(Qiskit) | estimator queries |
|---|---:|---:|---:|---:|---:|---:|---:|
| Peierls weak--weak | selected QSE | 15 | 15 | `1.32e-9` | 186 | 195 | 225 |
| Peierls weak--weak | direct Davidson | 12 | 12 | `1.55e-15` | 400 | 556 | 2340 |
| Peierls weak--strong | selected QSE | 15 | 15 | `5.99e-8` | 202 | 244 | 225 |
| Peierls weak--strong | direct Davidson | 12 | 12 | `1.55e-15` | 300 | 417 | 2340 |
| Peierls `U=8`, weak bond coupling | selected QSE | 12 | 12 | `2.78e-9` | 154 | 229 | 225 |
| Peierls `U=8`, weak bond coupling | direct Davidson | 12 | 12 | `3.11e-15` | 300 | 417 | 2340 |
| SIAM `N_b=4` | selected QSE | 64 | 44 | `5.07e-7` | 622 | 622 | 4666 |
| SIAM `N_b=4` | direct Davidson | 71 | 71 | `9.60e-7` | 2142 | 2142 | 46008 |

The exact errors in the last two rows are the errors at the first recorded
`10^-6` crossing; their terminal errors are `2.51e-7` for selected QSE at
growth dimension 77 (retained rank 55) and `1.78e-15` for Davidson at
dimension 95.

Thus, on these four fresh cases, selected QSE reaches the common `10^-6`
window target at lower graph-span cost, lower Qiskit-transpiled two-qubit
cost, and fewer declared estimator queries than direct Davidson.  Davidson
continues to essentially exact roots when allowed to grow further, at the
larger charged cost.

## Fixed-class comparator

The fixed Peierls class retains rank 5, so it cannot form the requested
six-root window in any of the three regimes.  The fixed SIAM class retains
rank 16 but has maximum six-root error `3.64e-2`.  It is cheaper (Peierls:
76 compiled two-qubit gates; SIAM: 66) because it does not solve the declared
task.  It reaches none of the `10^-2`, `10^-4`, or `10^-6` window targets.

## Static spectral audit

Two observables per family were evaluated from each terminal route:
site-0 density and bond displacement for Peierls, and impurity magnetization
and impurity double occupancy for SIAM.  The selected SIAM terminal spectra
have normalized broadened-L1 errors `5.21e-7` and `5.01e-7`; the direct route
is at numerical precision.  Selected Peierls spectra are likewise at or near
numerical precision.  Fixed-class spectral errors are materially larger, and
the artifact marks a normalized ratio as unavailable when the exact reference
weight is essentially zero rather than dividing by a numerical floor.

## Resource-accounting boundary

Qiskit numbers are deterministic structural Pauli-product compilation costs:
every nonzero Pauli word is preserved and complex coefficients are replaced
by their magnitudes before transpilation.  This is necessary because several
QSE records (for example `b` and current quadratures) are not themselves
Hermitian ansatz generators.  The artifact records this coefficient policy on
every compiled table.  These numbers are circuit-structure costs, not an LCU
state-preparation claim.

The selector was replayed independently for every case.  Selected supports,
stop reasons, and final selector residuals were identical in all four replay
locks.  The complete evidence artifact also records every exchange-maintained
growth support, all per-root errors, terminal statuses, Qiskit record rows,
and estimator conventions.

## Reproduction

```bash
PYTHONPATH=. python -m pipelines.exact_bench.paper_iii_qse_generality_campaign \
  --output-json output/diagnostics/paper_iii_qse_generality_20260829_v1/generality_evidence.json
```

The manuscript and existing Paper-III PDF were not edited by this campaign.
