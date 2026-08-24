# Repaired L=3 pure-Hubbard RA route — results source map (2026-08-23)

Profile for every run under `results/` unless the name says otherwise:

- adaptive shortlists: `phase123_active_score_inverse_simpson_adaptive_shortlist_v1`
- phase 0: declared active, cap 24. **Inert at L=3** (cap == pool size); on/off
  give bit-identical ledgers. Recorded for parity with HH, not for effect.
- compiled cost: `phase123_compiled_marginal_qiskit_delta_e_over_one_plus_k_v1`
  (Paper-I Eq. compiled_marginal, real transpiled marginals — NOT the
  `marrakesh_graph_span_v1` c_hat estimates the route used before)
- inner optimizer: Powell maxiter 200, **xtol 1e-5 / ftol 1e-12 on BOTH RA and
  AVQE** unless the directory name says `loose` (= 1e-4 / 1e-8 on both).
- prune delete-refit: always 1e-5 / 1e-12 (`PRUNE_DELETE_REFIT_POWELL_*`);
  it is a recoverability measurement, not an accepted refit.
- compile contract: FakeMarrakesh, optimization level 1, seed_transpiler 7,
  full circuit including reference-state preparation.

## Three code fixes these runs depend on (all uncommitted in this worktree)

1. `ra_adapt/marginal_qiskit_cost.py` (new) + `TranspiledMarginalCostOracle`
   swap in `adapt_pipeline._default_no_prune_backend_compile_oracle`, declared
   by `engine.py` as `ra_marginal_qiskit_cost_all_phases` for every appendix
   run. Before this, all five `raw_delta_compiled_*` fields were `None` and
   phases scored on graph-span estimates (~3x underestimate).
2. Inner-optimizer parity: `engine_support.RA_ACCEPTED_REFIT_POWELL_*` and
   `append.AAVQE_COMPARATOR_POWELL_*` are matched named constants. The
   published state had RA at 1e-4/1e-8 against the comparator's 1e-5/1e-12.
3. `PRUNE_DELETE_REFIT_POWELL_*` threaded through
   `optimizer_routes.run_deterministic(powell_tolerances=...)`. Without it the
   delete-refit stopped at a ~1e-8 residual that the recoverability gate read
   as a real energy regression and refused every deletion.

## Runs

| directory | arm | tolerance | k |
|---|---|---|---|
| `results/baseline_k12` | RA-Append | tight | 12 |
| `results/beam_escalated_k12` | RA + beam | tight | 12 |
| `results/metric_ablation_gated_k12` | RA + prune (theta 1e-3) | tight | 12 |
| `results/batching_k12` | RA + batch | tight | 12 |
| `results/batching_k6` | RA + batch at its own marker | tight | 6 |
| `results/prune_no_screen_h15` | RA + prune, no theta screen | tight | 15 |
| `results/avqe_{tight,loose}_k12` | AVQE comparator | both | 12 |
| `results/ra_{fixed,adaptive}_{tight,loose}_k12` | RA baseline | both | 12 |

## Headline numbers (k=12, tight, all on one build)

| method | E | N2q | D2q | Dc |
|---|---|---|---|---|
| AVQE | 7477 | 344 | 331 | 1393 |
| RA fixed shortlists | 9130 | 112 | 91 | 379 |
| RA adaptive | 7234 | 112 | 84 | 351 |
| RA adaptive + beam | 7219 | 108 | 80 | 335 |
| RA adaptive + prune | 7547 | 112 | 84 | 351 |
| RA adaptive + batch | 8633 | 232 | 203 | 844 |
| RA adaptive + batch (k=6) | 4236 | 208 | 191 | 790 |

Validation: AVQE tight reproduces the published comparator exactly — 7477 at
k=12 and 8522 at terminal, matching `ra_appendix_l3_v9_provenance.json`.

## Known limits

- **Prune has no advantage at k=12** in either screen variant. It wins only at
  k=9-11 (`prune_no_screen_h15`: 92/71/298 vs RA-Append 100/76/315 at k=9) and
  costs E 23651 vs 7234. Its canonical variant accepts a single deletion at
  r15 only.
- Batch wins on rounds-to-plateau (r4 vs r8) and on E at its k=6 marker; its
  circuits are ~1.9x RA-Append.
- Beam is the only extension that improves all four axes at k=12.
- The repo currently sits at the **loose** tolerance for both sides. Reproducing
  the tight rows requires setting both named constants to 1e-5/1e-12.

Reproduce: `marker_arms.py` (arms at marker), `k12.py` (AVQE/RA 2x2),
`prune_theta2.py` (screen variants), `plot_traj.py` (figure).
