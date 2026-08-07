# Paper-I Table III Hubbard-Holstein convergence audit

Date: 2026-05-26

## Scope

This audit covers the current Paper-I main-body Table III, `tab:fixed_accuracy_hh_cartesian`, for the Hubbard-Holstein benchmark. No manuscript table cells were changed.

Target and display convention:

- Shared target: `E_T = 2.0e-4`.
- Displayed error: `delta E = max(0, |E_alg(n_work) - E_ref(n_ED)| - E_T)`.
- Table III regimes and cutoff pairs remain `(3,6)`, `(2,5)`, `(5,8)`, and `(4,7)`.

## Executive finding

The Table III failures are not caused by the phonon cutoff floor alone. All four cutoff floors are below `E_T`. The completed non-hit rows are algorithm-limited by same-cutoff error that is much larger than the remaining target budget. The strongest immediate blockers are:

- Several comparator result JSONs do not carry `exact_reference_energy` even when the declared metric is the higher-cutoff reference; independent recomputation shows the displayed non-hit status is unchanged, but the source fields are incomplete for promotion-quality provenance.
- The current SNAKE HH numeric cells are sourced from `output/pdf/paper_i_three_model_hh_snake_completed_optuna_update_20260526.json`, not from the earlier partial CHTC table update. That SNAKE support file contains only two completed HH SNAKE rows, both non-hits.
- Strong-Holstein SNAKE rows remain incomplete locally. Pos-Geo strong-Holstein rows are still marked running or resource-guarded depending on the source artifact.
- The selected-logical HH pool has only 9 source records derived from an `n_ph=1` HH run. This is plausible as a reduced-pool recovery route, but it is a strong candidate root cause for high-cutoff strong-Holstein misses or stalls.

## Cutoff floor and algorithmic budget

| Regime | `(U/t, lambda)` | Cutoffs | `E_exact(n_work)` | `E_exact(n_ED)` | Cutoff floor | Same-cutoff algorithm budget |
|---|---:|---:|---:|---:|---:|---:|
| weak-weak | `(0.5, 0.5)` | `(3,6)` | `-0.840095743947573` | `-0.840104437698401` | `8.69e-6` | `1.91e-4` |
| strong-weak | `(1.5, 0.5)` | `(2,5)` | `-0.431402049779593` | `-0.431474623655604` | `7.26e-5` | `1.27e-4` |
| weak-strong | `(0.5, 1.5)` | `(5,8)` | `-1.065903893373882` | `-1.065946593456677` | `4.27e-5` | `1.57e-4` |
| strong-strong | `(1.5, 1.5)` | `(4,7)` | `-0.557570091679972` | `-0.557641581216973` | `7.15e-5` | `1.29e-4` |

Interpretation: to hit the external target, a method must get below the same-cutoff budget in the final column. None of the completed adaptive non-hit rows are close enough.

## Source inventory

Artifacts inspected:

- `output/pdf/paper_i_three_model_partial_chtc_v3_v4_table_update_20260525.json`
- `output/pdf/paper_i_three_model_terminal_cost_recovery_20260525.json`
- `output/pdf/paper_i_three_model_fidelity_sources_20260525.json`
- `output/pdf/paper_i_three_model_hh_snake_completed_optuna_update_20260526.json`
- `chtc/phase3_optuna/input/routeA_paper_i_three_model_selected_logical_20260525_v4/paper_i_three_model_routeA_records.tsv`
- `chtc/phase3_optuna/input/paper_i_three_model_comparators_selected_logical_20260525_v3/generic_static_table_records.tsv`
- `chtc/phase3_optuna/input/paper_i_three_model_reduced_pool_selected_logical_20260525_v1/manifest.json`
- `chtc/phase3_optuna/input/paper_i_three_model_reduced_pool_selected_logical_20260525_v1/hh_L2_from_result.selected_logical.json`

Important source-map facts:

- The partial table update has `comparator_cluster=6945817` and `snake_cluster=6945865`.
- In the partial table update, all four HH SNAKE jobs appear under `still_running_or_idle` as `static_family_native_adapt_phase3`.
- The newer SNAKE update file has two completed SNAKE HH rows: weak-weak and strong-weak. Both are completed non-hits.
- The Route-A SNAKE TSV uses `static_route_id=route_a`, `meta_feature_profile=paper_i_production_v1`, `required_target_profile=paper_i_phys_v1`, `discovery_objective_mode=discovery_first_crossing`, `fixed_inner_optimizer=SPSA`, and selected logical route `historical_selected` with transfer mode `exact_match_v1`.
- Comparator adaptive rows use selected logical source `hh_L2_from_result.selected_logical.json` with transfer mode `boundary_v1` for append/TETRIS/Pos-Geo. HEA, family VQE, and Qubit/QEB do not use that selected-logical source.

## Row-level decomposition

Values below recompute the external higher-cutoff error from `energy` and the exact ED energies above when source result JSONs are locally available. `source-missing-ref` means the result JSON did not include `exact_reference_energy`, even though the row declares the higher-cutoff metric.

| Regime | Method | Status | Same-cutoff error | External error | Required same-cutoff budget | Classification |
|---|---|---:|---:|---:|---:|---|
| weak-weak | HEA VQE | completed | `9.80e-1` | `9.80e-1` | `1.91e-4` | algorithm-limited, source-missing-ref |
| weak-weak | family VQE | completed | `9.18e-1` | `9.18e-1` | `1.91e-4` | algorithm-limited, source-missing-ref |
| weak-weak | append ADAPT | completed | `5.57e-3` | `5.57e-3` | `1.91e-4` | algorithm-limited, stopped by gradient threshold |
| weak-weak | TETRIS-ADAPT | completed | `5.57e-3` | `5.57e-3` | `1.91e-4` | algorithm-limited, stopped by gradient threshold |
| weak-weak | Pos-Geo-ADAPT | completed | `2.44e-3` | `2.45e-3` | `1.91e-4` | algorithm-limited, natural-gradient threshold |
| weak-weak | Qubit/QEB | completed | `9.01e-2` | `9.01e-2` | `1.91e-4` | structurally algorithm-limited |
| weak-weak | SNAKE | completed non-hit | `2.35e-3` | `2.35e-3` | `1.91e-4` | algorithm-limited |
| strong-weak | HEA VQE | completed | `2.68e-1` | `2.68e-1` | `1.27e-4` | algorithm-limited, source-missing-ref |
| strong-weak | family VQE | completed | `7.81e-1` | `7.81e-1` | `1.27e-4` | algorithm-limited, source-missing-ref |
| strong-weak | append ADAPT | completed | `9.73e-2` | `9.74e-2` | `1.27e-4` | algorithm-limited, stopped by gradient threshold |
| strong-weak | TETRIS-ADAPT | completed | `9.73e-2` | `9.74e-2` | `1.27e-4` | algorithm-limited, stopped by gradient threshold |
| strong-weak | Pos-Geo-ADAPT | completed | `1.19e-3` | `1.26e-3` | `1.27e-4` | algorithm-limited, natural-gradient threshold |
| strong-weak | Qubit/QEB | completed | `1.81e-1` | `1.81e-1` | `1.27e-4` | structurally algorithm-limited |
| strong-weak | SNAKE | completed non-hit | `1.19e-3` | `1.26e-3` | `1.27e-4` | algorithm-limited |
| weak-strong | HEA VQE | completed | `3.46e-1` | `3.46e-1` | `1.57e-4` | algorithm-limited, source-missing-ref |
| weak-strong | family VQE | completed | `7.87e-1` | `7.87e-1` | `1.57e-4` | algorithm-limited, source-missing-ref |
| weak-strong | append ADAPT | completed | `5.45e-2` | `5.45e-2` | `1.57e-4` | algorithm-limited, stopped by gradient threshold |
| weak-strong | TETRIS-ADAPT | skipped/resource-guarded in recovery source | n/a | n/a | `1.57e-4` | incomplete/source conflict |
| weak-strong | Pos-Geo-ADAPT | running or skipped/resource-guarded depending on source | n/a | n/a | `1.57e-4` | incomplete |
| weak-strong | Qubit/QEB | skipped/resource-guarded in recovery source | n/a | n/a | `1.57e-4` | incomplete/source conflict |
| weak-strong | SNAKE | running/missing locally | n/a | n/a | `1.57e-4` | incomplete |
| strong-strong | HEA VQE | completed | `1.28e-1` | `1.28e-1` | `1.29e-4` | algorithm-limited, source-missing-ref |
| strong-strong | family VQE | completed | `7.96e-1` | `7.97e-1` | `1.29e-4` | algorithm-limited, source-missing-ref |
| strong-strong | append ADAPT | completed | `4.66e-2` | `4.67e-2` | `1.29e-4` | algorithm-limited, stopped by gradient threshold |
| strong-strong | TETRIS-ADAPT | skipped/resource-guarded in recovery source | n/a | n/a | `1.29e-4` | incomplete/source conflict |
| strong-strong | Pos-Geo-ADAPT | running or skipped/resource-guarded depending on source | n/a | n/a | `1.29e-4` | incomplete |
| strong-strong | Qubit/QEB | skipped/resource-guarded in recovery source | n/a | n/a | `1.29e-4` | incomplete/source conflict |
| strong-strong | SNAKE | running/missing locally | n/a | n/a | `1.29e-4` | incomplete |

## SNAKE-specific audit

Completed SNAKE support rows from `paper_i_three_model_hh_snake_completed_optuna_update_20260526.json`:

| Regime | Trial | Same-cutoff error | External error | Display delta | Compiled `N2q` | Compiled `D2q` | Compiled `Dc` | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| weak-weak | `trial_0004` | `2.35e-3` | `2.35e-3` | `2.15e-3` | `2174` | `1670` | `5992` | completed non-hit |
| strong-weak | `trial_0006` | `1.19e-3` | `1.26e-3` | `1.06e-3` | `838` | `625` | `2077` | completed non-hit |

Local blockers for deeper SNAKE trajectory audit:

- The two `source_result_json` paths in the SNAKE support file are not present at their raw `raw_outputs/...` paths locally.
- The exact trial history, route identity payload, selected generator sequence, prune events, and beam events therefore could not be inspected locally from those result JSONs.
- The partial CHTC support artifact still lists all four HH SNAKE rows as `still_running_or_idle`, while the newer 2026-05-26 support file lists two completed non-hits. The audit should treat the newer SNAKE support as partial completion evidence, not complete Table III evidence.

Likely SNAKE failure mode from available evidence:

- Not cutoff-limited: same-cutoff error is `~10x` to `~18x` above the available same-cutoff budget for the two completed SNAKE rows.
- Either selected-logical pool incompleteness or optimizer/refit stagnation is likely. The available support cannot distinguish these without the missing per-trial result JSONs.
- The huge weak-weak SNAKE resource cells paired with non-hit status suggest expensive exploration/refit without reaching the target; this should be audited for selected-pool coverage, SPSA polish failure, or cost/objective misranking.

## Selected-logical HH pool audit

The HH selected-logical source manifest entry is:

- Source file: `chtc/phase3_optuna/input/paper_i_three_model_reduced_pool_selected_logical_20260525_v1/hh_L2_from_result.selected_logical.json`
- Source label: `artifacts/agent_runs/20260508_hh_l2_nph1_spsa_algebraic_warmstart_v1/hh_L2/trial_0013/hh_L2/json/result.json`
- Source kind: `adapt_vqe.continuation.selected_generator_metadata`
- Record count: `9`
- Family IDs listed by manifest: `hh_fermionic_reusable::exchange_current_nn`, `hh_termwise_ham_quadrature_term`, `paop_full:paop_cloud_p`, `paop_full:paop_disp`, `uccsd`

Sample selected labels:

- `hh_termwise_ham_quadrature_term(yezeee)`
- `hh_termwise_ham_quadrature_term(yeeeze)`
- `uccsd_ferm_lifted::uccsd_sing(alpha:0->1)`
- `uccsd_ferm_lifted::uccsd_sing(beta:2->3)`
- `hh_fermionic_reusable::exchange_current_nn(0,1)`
- `paop_full:paop_disp(site=0)`
- `paop_full:paop_disp(site=1)`
- `paop_full:paop_cloud_p(site=1->phonon=0)`
- `paop_full:paop_cloud_p(site=0->phonon=1)`

Interpretation:

- This is a very small reduced pool for `n_ph=2..5` work cutoffs and `n_ED=5..8` references.
- Because it was derived from an `n_ph=1` HH warm-start result, it may under-cover strong-Holstein or high-cutoff phonon dressing directions.
- This is the first concrete repair target for SNAKE and selected-logical adaptive comparators: run a full-meta diagnostic or build an expanded HH selected-logical manifest from higher-cutoff successful evidence.

## Comparator audit

Adaptive comparator behavior:

- Append ADAPT and TETRIS stop by `gradient_threshold` while still far above the required same-cutoff budget. This is an algorithm/selection stopping failure, not a cutoff-floor failure.
- Pos-Geo is the closest completed non-SNAKE adaptive method in weak-weak and strong-weak, but still misses by roughly one order of magnitude relative to the allowed same-cutoff budget. It stops by `geo_natural_gradient_norm_threshold` in completed rows.
- Strong-Holstein Pos-Geo rows are incomplete locally, with source artifacts alternating between `running_or_missing_payload` and `skipped_resource_guard` depending on artifact generation path.
- HEA and Qubit/QEB are structurally poor for HH here and should remain baselines unless the goal becomes improving those baselines specifically.
- Family VQE is also far off, which suggests the fixed family ansatz is not expressive enough under the current high-cutoff HH parameterization.

Source/provenance issue:

- Several comparator result JSONs set `primary_energy_metric = higher_cutoff_reference_abs_delta_e` and `same_cutoff_error_role = diagnostic_only`, but `exact_reference_energy` and `exact_reference_n_ph_max` are null.
- Independent recomputation of external errors does not change the non-hit classification, but promotion-quality support maps should carry the higher-cutoff reference energy explicitly.

## Root-cause classification

| Failure class | Evidence | Applies to |
|---|---|---|
| Cutoff-floor impossible | Not supported; all floors below `2e-4` | None |
| Algorithm-limited same-cutoff error | Same-cutoff error exceeds remaining budget by `~6x` to `>7000x` | Completed rows broadly |
| Optimizer/threshold stagnation | `gradient_threshold` or `geo_natural_gradient_norm_threshold` stops while still above target | append, TETRIS, Pos-Geo |
| Reduced-pool incompleteness | HH selected-logical pool has only 9 records from `n_ph=1` source | SNAKE, append, TETRIS, Pos-Geo selected-logical rows |
| Source-field mismatch | Higher-cutoff reference fields missing in comparator JSONs | Many comparator rows |
| Incomplete evidence | Running/missing or resource-guarded rows | SNAKE strong-Holstein, Pos-Geo strong-Holstein, some recovery-source rows |

## Recommended diagnostics before rerun promotion

1. Fetch or locate the two completed SNAKE result JSONs named in `paper_i_three_model_hh_snake_completed_optuna_update_20260526.json`, plus the active strong-Holstein SNAKE outputs if completed on CHTC. Required fields: full history, selected generator metadata, route identity, selected-logical filter metadata, prune events, beam events, first-crossing status, and per-trial summary.
2. Run a full-meta-vs-selected-logical pool audit for all four HH regimes without changing table cells. Compare pool size, family coverage, and top gradient families.
3. Run fixed-sequence optimizer polish on the two completed SNAKE non-hit ansätze. If polish reaches the same-cutoff budget, optimizer/refit is the blocker. If polish fails, the selected sequence/pool is likely insufficient.
4. Run one shallow full-meta SNAKE diagnostic per HH regime to test whether early energy slope improves relative to selected-logical recovery. These should be diagnostic only.
5. Regenerate support maps only after source rows include explicit higher-cutoff reference energies and compiled displayed-ansatz sidecars.

## Repair levers

SNAKE:

- First repair or audit selected-logical coverage. The current HH source is likely too narrow for high-cutoff strong-Holstein regimes.
- If fixed-sequence polish helps, increase final/refit optimizer budget and preserve the same Route-A production identity.
- If full-meta diagnostic improves early slope, build an expanded selected-logical HH source manifest rather than silently broadening the current one.
- Preserve `route_a`, `paper_i_production_v1`, `paper_i_phys_v1`, first-crossing reporting, and the current cutoff pairs unless explicitly reopened.

Append/TETRIS:

- Increase depth and reconsider gradient threshold behavior; current stops are far above target.
- Test whether selected-logical reduction suppresses essential phonon dressing terms by comparing to full-meta diagnostic.

Pos-Geo:

- Prioritize finishing or fetching strong-Holstein rows.
- Audit metric conditioning and natural-gradient threshold. Completed rows are close relative to other comparators but still not target hits.

Fixed baselines:

- HEA/Qubit-QEB/family VQE are not the first repair priority. They mostly document baseline weakness under this HH regime.

## Promotion criteria for a repaired Table III cell

A row should not replace the manuscript cell until all of the following hold:

- The source artifact is complete, not `running`, `idle`, `skipped_resource_guard`, or missing locally.
- The row case ID and cutoff pair match Table III exactly.
- `required_target_profile = paper_i_phys_v1` and the external target is `2e-4`.
- The source carries both `same_cutoff_exact_gs_energy` and `exact_reference_energy` or an auditable support map computes them explicitly.
- Target-hit rows use first-crossing compiled resources.
- Non-hit rows use terminal compiled resources and remain visibly non-hit.
- SNAKE rows validate Route-A identity and `paper_i_production_v1`.
- The support map records source result path, source hash when available, cutoff pair, threshold, first-hit/terminal role, and compiled-cost sidecar path.

## Bottom line

The current Table III HH problem is not that the chosen work cutoffs force near-machine-precision convergence. The completed rows fail because the algorithms are not close enough even at the same cutoff. The most plausible fix path is not raising cutoffs; it is auditing selected-logical pool coverage and optimizer/refit behavior, then rerunning targeted diagnostics before any table-cell promotion.
