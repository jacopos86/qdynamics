# Hubbard-Holstein fixed-scaffold motif inventory - 2026-06-14

Purpose: compact reference for later fixed-scaffold VQE / Optuna discussions. This is not a manuscript table update and not a promotion record.

## Key distinction

Replaying a final SNAKE scaffold is only a control. It tests whether the discovered operator sequence is expressive after a fixed-ansatz refit. It does not discover a new scaffold, and it should not be counted as a fresh adaptive-selection result.

The useful follow-up is to build a small scaffold grammar from motifs SNAKE already exposed, then let Optuna choose a small number of structural choices around those motifs.

## Existing fixed-scaffold machinery

- Diagnostic CLI: `pipelines/exact_bench/fixed_scaffold_expressivity_audit.py`
- Runner: `pipelines.exact_bench.generic_static_adapt_variants.run_fixed_scaffold_expressivity_audit_single`
- Current behavior: seeds from an existing `adapt_vqe` JSON, assembles a fixed full-meta scaffold, optionally pads from the pool, and refits with selected optimizers including `spsa`.
- Scope: diagnostic / candidate evidence only unless later rerun under the Paper-I run gates and user-approved evidence workflow.

## Recovered scaffold motifs

### Weak-strong, cheap current visible plateau

Source sidecar: `output/pdf/paper_i_table_iii_snake_weak_strong_trial0015_depth6_depth13_qiskit_cost_20260614.json`

Depth-6 plateau sequence:

```text
1. hh_fermionic_reusable::bond_charge_current_nn_up(0,1)
2. hh_fermionic_reusable::bond_charge_current_nn_dn(0,1)
3. paop_full:paop_disp(site=1)
4. paop_full:paop_disp(site=0)
5. hh_phonon::s(site=1)
6. hh_phonon::s(site=0)
```

Interpretation: clean three-block motif: electronic current -> local phonon displacement -> phonon squeeze. Good low-cost control scaffold; not enough evidence that it breaks the deeper weak-strong plateau.

### Weak-strong, depth-13 flat-arm terminal prefix

Same source sidecar as above.

Depth-13 terminal sequence is displacement-heavy:

```text
1-4. paop_full:paop_disp(site=1) repeated
5.   hh_fermionic_reusable::bond_charge_current_nn_up(0,1)
6.   hh_fermionic_reusable::bond_charge_current_nn_dn(0,1)
7.   paop_full:paop_disp(site=1)
8.   paop_full:paop_disp(site=0)
9.   hh_phonon::s(site=1)
10.  hh_phonon::s(site=0)
11-13. paop_full:paop_disp(site=0) repeated
```

Interpretation: useful as a baseline for the flat-arm CHTC behavior, but probably not a good general scaffold grammar by itself because it mostly repeats plain displacements.

### Weak-strong, deeper replayable continuation

Source JSON: `raw_outputs/routeA_paper_i_hh_weak_strong_snake_depth42_reprobe_20260613_v1/weak_strong/json/result.json`

Observed depth-42 motif counts:

```text
electron-phonon hopdrag:        1
density/phonon squared term:    1
electronic single/current:     10
phonon displacement:            9
UCC double:                     6
dressed phonon/correlation:    10
phonon squeeze:                 3
other:                          2
```

First twelve operators:

```text
1.  paop_full:paop_hopdrag(0,1)
2.  paop_sq_std:paop_dens_sq(site=1)
3.  uccsd_ferm_lifted::uccsd_sing(alpha:0->1)
4.  uccsd_ferm_lifted::uccsd_sing(beta:2->3)
5.  paop_full:paop_disp(site=1)
6.  paop_full:paop_disp(site=1)
7.  uccsd_ferm_lifted::uccsd_dbl(ab:0,2->1,3)
8.  paop_full:paop_disp(site=0)
9.  paop_full:paop_disp(site=0)
10. paop_full:paop_disp(site=1)
11. paop_lf_full:paop_dbl_p(site=0->phonon=0)
12. hh_phonon::s(site=1)
```

Interpretation: better source for a plateau-breaking scaffold grammar than the depth-6 prefix. It introduces correlation/dressed phonon terms, not only currents and plain displacements.

### Strong-weak, replayable k=11 source

Source JSON: `raw_outputs/chtc_fetches/hh_snake_strong_weak_trial0011_20260530_113725/raw_outputs/routeA_paper_i_three_model_hh_l2_nph2_three_model_sym_strong_weak_full_meta_energygeom_nocost_routefix_v6/run/hh_L2_nph2_three_model_sym_strong_weak/trial_0011/hh_L2_nph2_three_model_sym_strong_weak/json/result.json`

Sequence:

```text
1.  uccsd_ferm_lifted::uccsd_dbl(ab:0,2->1,3)
2.  hh_phonon::s(site=0)
3.  hh_fermionic_reusable::bond_charge_current_nn_up(0,1)
4.  hh_fermionic_reusable::bond_charge_current_nn_dn(0,1)
5.  paop_full:paop_disp(site=1)
6.  paop_full:paop_disp(site=1)
7.  paop_full:paop_disp(site=0)
8.  paop_full:paop_disp(site=0)
9.  hh_phonon::s(site=1)
10. uccsd_ferm_lifted::uccsd_sing(beta:2->3)
11. uccsd_ferm_lifted::uccsd_sing(beta:2->3)
```

Interpretation: clean successful motif for stronger Hubbard character at low phonon cutoff: UCC double/correlation appears early, then current/displacement/squeeze, then repeated beta single.

### Strong-strong, base k=12 source

Source JSON: `raw_outputs/chtc_fetches/hh_snake_all_time_best_ws_ss_20260531/hh_ss_trial0001_result.json`

Sequence:

```text
1.  uccsd_ferm_lifted::uccsd_dbl(ab:0,2->1,3)
2.  uccsd_ferm_lifted::uccsd_sing(alpha:0->1)
3.  uccsd_ferm_lifted::uccsd_sing(beta:2->3)
4.  paop_full:paop_cloud_p(site=1->phonon=0)
5.  paop_full:paop_cloud_p(site=1->phonon=0)
6.  paop_full:paop_disp(site=1)
7.  paop_full:paop_disp(site=1)
8.  paop_full:paop_disp(site=1)
9.  paop_lf_full:paop_dbl_p(site=1->phonon=1)
10. hh_phonon::s(site=0)
11. hh_phonon::s(site=1)
12. hh_phonon::s(site=1)
```

Interpretation: stronger Hubbard sectors appear to need electronic correlation plus dressed phonon/cloud terms. This supports using `cloud_p` / `dbl_p` / UCC double as the first-pass strong-Hubbard scaffold knobs.

### Weak-weak limitation

The local paper-facing weak-weak source currently preserves prefix/error/resource provenance but not selected generator labels:

- `MATH/paper_facing/paper_I_static_scaffold/history_sources/hh_tableiii_weak_weak_snake_flatnovelty_nocost_bounded_v3_trial0000_current_slim_20260531.json`
- `output/pdf/paper_i_table_iii_snake_weak_weak_live_prefix_promotion_20260530.json`

To recover the exact weak-weak scaffold, refetch or locate the original `current.json` from the CHTC route named in the promotion artifact.

## Scaffold grammar for later testing

Small grammar candidates:

1. Electronic current block:
   - `bond_charge_current_nn_up(0,1)`
   - `bond_charge_current_nn_dn(0,1)`

2. Plain phonon dressing block:
   - `paop_disp(site=0)`
   - `paop_disp(site=1)`

3. Phonon relaxation block:
   - `hh_phonon::s(site=0)`
   - `hh_phonon::s(site=1)`

4. Hubbard-correlation block:
   - `uccsd_dbl(ab:0,2->1,3)`
   - optionally alpha/beta UCC singles

5. Dressed phonon/correlation block:
   - `paop_cloud_p(site=...->phonon=...)`
   - `paop_dbl_p(site=...->phonon=...)`
   - `paop_hopdrag(0,1)`
   - `paop_dens_sq(site=...)`

First-pass Optuna should not search arbitrary large sequences. Prefer a small number of block/order/repeat choices, then SPSA inner optimization.

## Recommended use later

- Use exact discovered scaffolds as controls.
- Use motif-derived block grammars for actual scaffold search.
- Keep scaffold-search evidence separate from adaptive SNAKE evidence unless a later user-approved run/report workflow integrates it.
