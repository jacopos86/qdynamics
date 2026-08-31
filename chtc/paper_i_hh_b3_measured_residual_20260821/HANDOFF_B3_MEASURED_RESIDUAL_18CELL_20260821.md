# B3-MR: the Bundle-3 matrix with the stationarity assumption dropped

**Decision already made — do not reopen:** the variant flips ONE coupled axis.
Bundle 3 ran the semantic-closure selector: stationary source response, raw
joint gain.  B3-MR runs the measured-residual selector: the active block's
residual gradient is measured every round, and a candidate is credited only
with its marginal gain, joint-minus-active-only.  These are one commitment,
not two knobs — you cannot subtract a baseline you have assumed is zero.
Everything else stays B3.

Written at `appendix-demos-20260820` commit `9c6103b7e016eb026210f2f5e6a1da6e3bcfbbf2`.
Contract conventions: `agent_guidance/shared/agent-handoff-contract.md`.

## Why (evidence, not argument)

L=3 pure-Hubbard one-factor comparison, same 24-generator pool, same physics,
same horizon, shown insensitive to Powell tolerances
(`chtc/ra_adapt_powell_tol_diag_20260821/results/`):

| selector | first <=1e-6 | terminal N2q/D2q/Dc |
|---|---|---|
| measured + marginal | 8 | 127/109/441 |
| stationary + raw (B3's) | 9 | 304-324 / 279-312 / 1149-1284 |

Mechanism: under stationarity the active-only baseline is zero BY ASSUMPTION,
so every candidate inherits the active block's descent — expensive doubles
score their full joint gain and get admitted round after round.  Measuring
the residual makes the subtraction real: near-redundant doubles score ~0 and
cheap hops win.  Expectation for HH: materially more compact circuits at
equal or better error, with the RA-vs-AAVQE margins widening.

## The executable route — no code changes

The canonical facade already materializes this route on HH problems:

```python
problem = build_paper_i_ra_hh_regime_problem(regime_id)
request = RAAdaptRequest(
    method=SRMethodPolicy(insertion=<per policy below>),
    execution=SRExecutionPolicy(stop=SRStopPolicy(maximum_controller_rounds=N)),
)
result = run_ra_adapt(problem, request)
```

Verified at this commit: the resolved protocol reports
`algorithm_id = paper_i_ra_adapt_nonstationary_incremental_full_response_v2`,
`active_gradient_policy = measured_residual_response_v1`,
`phase3_candidate_gain_policy = joint_minus_active_only_supported_trust_v1`.
**Preflight gate (mandatory per cell):** assert those three strings on the
materialized protocol before execution; refuse the cell otherwise.  Do NOT
use the `build_paper_i_ra_*semantic*` request builders — they hard-lock
stationarity and will either raise or silently give you B3 again.

## Matrix — 18 cells exactly as B3

- Regimes: W-W, I-W, S-W, W-S, I-S, S-S (U/t in {0.25, 1.25, 8} x lambda in
  {0.25, 1.25}; omega0/t = 1).  Same nph cutoffs as B3 (weak sector nph=3,
  strong nph=7 — same-cutoff comparisons mandatory).
- Insertion policies: append_only, plateau_commutation, always — 3 per regime.
- Pool: the fixed symmetry- and padding-valid single-Pauli pools B3 used
  (948 weak / 6508 strong).  Same pool, same seeds (adapt 7, transpiler 7).
- Horizons: as B3 reported — weak sector through 45 (run 75 where B3 did),
  strong sector through 75.
- Compile convention: FakeMarrakesh, optimization_level 1, seed_transpiler 7,
  full circuit including reference preparation.

## Differences from B3 you must DISCLOSE, not remove

1. **Refit layer rides the route.** The full-response route carries the
   whitened chart PLUS the exact-guarded joint-step warm start
   (`exact_applied_joint_step_guarded_v1`).  B3 had the chart but no
   initialization concept.  This is part of the route identity — report it as
   such; do not strip it to force one-factor purity, and do not let anyone
   attribute the whole effect to the selector without naming this rider.
2. **Powell tolerances.** RA accepted refits dispatch through engine_support
   at xtol 1e-4 / ftol 1e-8 — same as B3 actually ran, so B3-MR vs B3 is
   matched.  But the AAVQE comparator's own refit runs 1e-5/1e-12
   (append.py), so any RA-vs-AAVQE statement inherits the existing asymmetry.
   Disclose it; changing it is a separate decision with its own rerun.
   The env knob `STATIC_ADAPT_POWELL_TOL_OVERRIDE="xtol:ftol"` exists for
   diagnostics only — canonical cells must NOT set it.
3. **Estimator cost.** Measuring the residual buys gradient/metric work the
   stationary route never paid.  At L=3 it was repaid; report per-cell E so
   the HH trade is measured, not asserted.

## Per-cell report-back

For each of the 18 cells: regime, insertion policy, protocol sha256, the
three preflight strings; k* under B3's own k* rule; error at k* and terminal;
(N2q, D2q, Dc) at k* and terminal; cumulative E; the accepted generator
sequence; and the doubles-vs-singles admission count as a direct check of the
mechanism above.  Compare each cell against its B3 counterpart in one table.

## Traps

- CHTC packaging omits `docs/`; every job dies on
  `import docs.reports.pdf_utils` unless the package includes it.
- One heavy job per machine locally; 10 GB agent ceiling; big sweeps to CHTC.
- Bundle receipts: B3 cells are anchored artifacts — never overwrite them;
  B3-MR gets its own campaign directory and fresh receipts.
- The full-response route runs to horizon; B3's k* selection rule is a
  REPORTING convention, not a stopping rule — keep the runs full length.
