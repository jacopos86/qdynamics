# Paper II Dynamics CHTC Artifact Handoff — 2026-05-11

Purpose: agent-facing provenance note for updating Paper II time-dynamics benchmark tables from the recent CHTC dynamics runs. This file is not a manuscript section. It tells the table-update agent which artifacts to trust, which records to exclude, and where to find raw outputs.

## Current decision: harmonic/Kerr chain is excluded

Do **not** include `harmonic_kerr_chain` / harmonic-Kerr / harmonic Kerr records in Paper II benchmark aggregates, bosonic class averages, all-class averages, or paper-facing table cells.

Reason: this pathway is treated as killed/nonconvergent for the current paper-facing dynamics benchmark set. Even if a local summary JSON exists or a trial appears numerically feasible, the seed/path is not accepted as a robust benchmark fixture for the current Paper II table. It should remain diagnostic only until a verified static seed and both low-drive and high-drive dynamics passes exist.

Code support for this Hamiltonian may remain in the repo. The exclusion is a paper-facing benchmark decision, not a request to delete generic Hamiltonian support.

## Manuscript/doc edits already made

- `MATH/paper_details/time_dynamics_paper_II.tex`: `tab:dyn_claims` caption no longer says harmonic/Kerr is retained as a stress row.
- `MATH/paper_details/main_condensed.tex`: `tab:dyn_claims` caption no longer says harmonic/Kerr is retained as a stress row.
- `MATH/paper_details/main_condensed.tex`: benchmark-suite row now lists boson-only controls as `Bose--Hubbard` only.
- Active CHTC record lists no longer include harmonic/Kerr:
  - `chtc/time_dynamics_optuna/input/full_records.txt`
  - `chtc/time_dynamics_optuna/input/full_dt321_append_prune_records.txt`

## CHTC run families

Remote base used by this workflow:

```text
Holstein_time_dynamics_optuna_chtc
```

Local fetched root:

```text
raw_outputs/chtc_time_dynamics_optuna/
```

Main queue IDs from this run series:

| Cluster | Intent | Current paper-facing use |
|---|---|---|
| `6308626` | full class-tuned dynamics Optuna matrix | candidate source for Table III / benchmark rows after exclusions |
| `6308627` | dt=321 append/prune pressure matrix | candidate source for append/prune evidence and ablations, but only passed/non-excluded records |

If the SSH control master is active at `/tmp/chtc-master`, fetch with:

```bash
CHTC_SSH_CONTROL_PATH=/tmp/chtc-master \
  bash chtc/time_dynamics_optuna/fetch_from_chtc.sh
```

Validate fetched outputs with:

```bash
python chtc/time_dynamics_optuna/validate_outputs.py \
  --root raw_outputs/chtc_time_dynamics_optuna \
  --record-list chtc/time_dynamics_optuna/input/full_records.txt

python chtc/time_dynamics_optuna/validate_outputs.py \
  --root raw_outputs/chtc_time_dynamics_optuna \
  --record-list chtc/time_dynamics_optuna/input/full_dt321_append_prune_records.txt
```

As of this handoff, after harmonic/Kerr exclusion, local validator status was:

| Record list | Passed | Failed | Notes |
|---|---:|---:|---|
| `full_records.txt` | 13 / 15 | 2 / 15 | Failed: `td_ttprime_hubbard_A0p2_t8_exact_v1`, `td_ttprime_hubbard_A0p6_t8_exact_v1` had zero completed/feasible trials locally. |
| `full_dt321_append_prune_records.txt` | 12 / 16 | 4 / 16 | Failed: missing TT-prime dt321 outputs and zero-feasible HH dt321 outputs locally. |

Do not use failed records for table aggregation unless they are repaired and revalidated.

## Records to treat as candidate include/exclude

### Candidate include: full matrix, after validation

Use only records that pass validation and are not harmonic/Kerr. The full matrix currently targets:

```text
td_hubbard_A0p2_t8_exact_v1
td_hubbard_A0p6_t8_exact_v1
td_ionic_hubbard_A0p2_t8_exact_v1
td_ionic_hubbard_A0p6_t8_exact_v1
td_extended_hubbard_A0p2_t8_exact_v1
td_extended_hubbard_A0p6_t8_exact_v1
td_ttprime_hubbard_A0p2_t8_exact_v1        # exclude until repaired/revalidated
td_ttprime_hubbard_A0p6_t8_exact_v1        # exclude until repaired/revalidated
td_spinless_tv_A0p2_t8_exact_v1
td_spinless_tv_A0p6_t8_exact_v1
td_spin_boson_A0p2_t8_exact_v1
td_spin_boson_A0p6_t8_exact_v1
td_bose_hubbard_A0p2_t8_exact_v1
td_bose_hubbard_A0p6_t8_exact_v1
td_hubbard_strict_static_t8_qpu_faithful    # include only if strict validation passes in the local report
```

### Candidate include: dt321 append/prune matrix, after validation

Use only records that pass validation and are not harmonic/Kerr. The dt321 list currently targets:

```text
td_hubbard_A0p2_t8_dt321_append_prune_v2
td_hubbard_A0p6_t8_dt321_append_prune_v2
td_ionic_hubbard_A0p2_t8_dt321_append_prune_v2
td_ionic_hubbard_A0p6_t8_dt321_append_prune_v2
td_extended_hubbard_A0p2_t8_dt321_append_prune_v2
td_extended_hubbard_A0p6_t8_dt321_append_prune_v2
td_ttprime_hubbard_A0p2_t8_dt321_append_prune_v2   # exclude until output exists/revalidates
td_ttprime_hubbard_A0p6_t8_dt321_append_prune_v2   # exclude until output exists/revalidates
td_spinless_tv_A0p2_t8_dt321_append_prune_v2
td_spinless_tv_A0p6_t8_dt321_append_prune_v2
td_spin_boson_A0p2_t8_dt321_append_prune_v2
td_spin_boson_A0p6_t8_dt321_append_prune_v2
td_bose_hubbard_A0p2_t8_dt321_append_prune_v2
td_bose_hubbard_A0p6_t8_dt321_append_prune_v2
td_hh_A0p2_t8_dt321_append_prune_v2                 # exclude until repaired/revalidated locally
td_hh_A0p6_t8_dt321_append_prune_v2                 # exclude until repaired/revalidated locally
```

### Mandatory exclude

```text
td_harmonic_kerr_chain_A0p2_t8_exact_v1
td_harmonic_kerr_chain_A0p6_t8_exact_v1
td_harmonic_kerr_chain_A0p2_t8_dt321_append_prune_v2
td_harmonic_kerr_chain_A0p6_t8_dt321_append_prune_v2
```

## Artifact structure per record

For a record ID `<RID>`, inspect:

```text
raw_outputs/chtc_time_dynamics_optuna/<RID>/record.json
raw_outputs/chtc_time_dynamics_optuna/<RID>/summary.json
raw_outputs/chtc_time_dynamics_optuna/<RID>/chtc_status.json
raw_outputs/chtc_time_dynamics_optuna/<RID>/run/progress.json
raw_outputs/chtc_time_dynamics_optuna/<RID>/task_result.json
raw_outputs/chtc_time_dynamics_optuna/<RID>/command.sh
```

The Optuna best trial is in:

```text
summary.json -> best_objective_trial -> metrics
```

Common metric keys include:

```text
mean_abs_energy_total_error
primary_observable_mae_over_exact_span
mean_abs_site_occupations_error
min_fidelity_exact
compiled_count_2q
compiled_depth
compile_backend
append_count
prune_count
full_horizon_gate_passed
full_horizon_early_stop_reason
final_runtime_parameter_count
```

Use `compiled_count_2q`, `compiled_depth`, and any available two-qubit-depth field emitted by the summary/reporting layer for hardware-cost table columns. If a field is missing, compile the final scaffold with the existing FakeMarrakesh/Marrakesh reporting path rather than inventing a proxy.

## Paper-table rules for the other agent

1. Exclude harmonic/Kerr from all Paper II benchmark aggregates.
2. Do not include failed validator records.
3. Do not average failed, zero-feasible, or missing-output rows into class means/medians.
4. Keep class-tuning provenance explicit: McLachlan/controller settings should be tuned at coarse Hamiltonian-class granularity (`fermionic`, `bosonic`, `mixed_fermion_boson`), not per Hamiltonian instance.
5. It is acceptable for the initial static ADAPT ansatz/seed to be benchmark-point-specific, because Paper II compares time-dynamics propagation methods on the same prepared initial ansatz.
6. For QPU-faithful claims, enforce the repo contract: controller decisions must use measurement-compatible prepared-state observables/tangent geometry only. Exact ED/reference trajectories are allowed for diagnostics, plots, and error columns, not as controller-decision inputs.
7. If using `exact_v1` rows, label them internally as fast/provisional dynamics benchmark rows unless the strict decision-data-flow guard proves the controller path is measurement-compatible. Do not let exact-assisted rows silently support strict QPU-faithful claims.
8. Do not claim useful pruning from this CHTC matrix unless a best validated non-excluded trial has `prune_count > 0`. Current best validated non-excluded evidence does not yet establish that.
9. Append evidence should be reported only from non-excluded records and only if `append_count > 0` in the selected best trial.
10. Preserve machine-readable provenance: every table cell should be traceable to record ID, trial number, and raw `summary.json` path.

## What this chat has established for Paper II

- The dynamics benchmark workflow now has class-tuning semantics: static ansatz can be benchmark-specific, while McLachlan/controller parameters should be coarse-class-tuned rather than Hamiltonian-instance-tuned.
- The recent CHTC outputs are useful for provisional Paper II table filling, but they are not all locked final benchmark evidence.
- Harmonic/Kerr is excluded from paper-facing dynamics aggregates.
- TT-prime Hubbard and HH dt321 records need repair/revalidation before they can support table claims from this specific fetched output set.
- Append/prune remains an unresolved evidence gap for Paper II: these runs should not be used to claim robust pruning benefit unless a later validated non-excluded best trial demonstrates it.

## Suggested next step for the table-update agent

Create or refresh a small aggregation artifact before touching `.tex`:

```text
raw_outputs/chtc_time_dynamics_optuna/dynamics_table_fill/aggregated_metrics_excluding_harmonic.json
raw_outputs/chtc_time_dynamics_optuna/dynamics_table_fill/table_dyn_claims_fragment_excluding_harmonic.tex
raw_outputs/chtc_time_dynamics_optuna/dynamics_table_fill/table_dyn_ablation_fragment_excluding_harmonic.tex
```

The aggregation artifact should record:

- included record IDs;
- excluded record IDs and reason;
- selected trial number per included record;
- aggregation rule: mean or median, matching the table caption;
- Hamiltonian class and source-table class;
- metric fields used for energy, observables, fidelity, spectra, compiled costs, and shots.

## Table-lock same-seed refresh — 2026-05-12

Purpose: this is the current follow-up sweep intended to repair the Paper II
table provenance problem.  It uses the current available Phase-3 static ADAPT
seed artifacts as the per-benchmark-point seed source, then runs the dynamics
methods from that same seed so checkpoint McLachlan, fixed McLachlan, and the
benchmark comparators are not mixed across different initial ansatz records.

Local table-lock inputs:

```text
chtc/generic_time_dynamics_table/input/table_lock_available_phase3_cases.json
chtc/generic_time_dynamics_table/input/table_lock_available_phase3_records.tsv
chtc/generic_time_dynamics_table/input/table_lock_available_phase3_record_ids.txt
chtc/generic_time_dynamics_table/input/table_lock_available_phase3_smoke_record_ids.txt
chtc/generic_time_dynamics_table/input/seed_artifacts_table_lock_available_phase3/
```

Local submission note:

```text
prompt-exports/2026-05-12-dynamics-table-lock-submission.md
```

Remote base remains:

```text
Holstein_time_dynamics_optuna_chtc
```

Current CHTC clusters:

| Cluster | Intent | Status |
|---|---|---|
| `6335327` | table-lock smoke after argument/env fix | passed |
| `6335328` | full table-lock matrix, 144 rows | scheduler-finished; 100 rows emitted success status; 44 rows failed mechanically and are being repaired/resubmitted |
| `6335333` | three-row repair smoke: exact-reference, ablation, HH table-lock | passed |
| `6335334` | non-exact failed-row retry subset | passed, 28 rows |
| `6335335` | remaining exact-reference retry subset | passed, 13 rows |

Final remote status after repair:

```text
table_lock_available_phase3_record_ids.txt: 144 / 144 rows have chtc_status.json return_code = 0
```

Observed mechanical failures in `6335328`:

1. `dyn_exact_reference` rows were dispatched through realtime controller mode
   `off`, which driven realtime correctly rejects.  Repair: dispatch the
   diagnostic exact-reference row through the diagnostic exact route rather
   than controller-off driven realtime.
2. `dyn_controller_ablation_matrix` rows did not receive the table-lock case
   manifest in the CHTC wrapper.  Repair: pass `--case-manifest` to
   `generic_dynamics_ablation_matrix` just as benchmark rows already do.
3. HH table-lock case IDs were incorrectly routed through the legacy HH wrapper,
   which only knows the legacy anchor case ID.  Repair: if a table-lock HH case
   is present in the case manifest, dispatch it through the generic isolated
   benchmark runner; use the legacy HH wrapper only for legacy HH anchor cases.

Do not interpret the original 44 failed rows as failed physics.  They were
dispatch/manifest failures and have now been repaired.  The table-lock matrix
is complete at the job-status level; the next step is aggregation and table-cell
provenance extraction from the result JSONs.

Compact aggregation artifacts have been generated and fetched locally:

```text
raw_outputs/dynamics_table_fill/table_lock_available_phase3_20260512/aggregation_manifest.json
raw_outputs/dynamics_table_fill/table_lock_available_phase3_20260512/paper_agent_summary.md
raw_outputs/dynamics_table_fill/table_lock_available_phase3_20260512/tables_summary.json
raw_outputs/dynamics_table_fill/table_lock_available_phase3_20260512/tab_dyn_claims.json
raw_outputs/dynamics_table_fill/table_lock_available_phase3_20260512/tab_dyn_ablation_matrix.json
```

Aggregation status:

```text
source summary files: 144
loaded normalized rows: 224
tab:dyn_claims aggregate rows: 42
tab:dyn_ablation_matrix rows: 96
missing source summaries: 0
```

Expected repaired outputs:

```text
raw_outputs/generic_time_dynamics_table/<record_id>/record.json
raw_outputs/generic_time_dynamics_table/<record_id>/command.sh
raw_outputs/generic_time_dynamics_table/<record_id>/chtc_status.json
raw_outputs/generic_time_dynamics_table/<record_id>/result/summary.json
raw_outputs/generic_time_dynamics_table/<record_id>/result/tab_dyn_claims.json
raw_outputs/generic_time_dynamics_table/<record_id>/result/tab_dyn_ablation_matrix.json
```

For the table-update agent, the same-seed contract is the important point:
aggregate only rows whose `seed_lock.same_seed_comparator_group_id`,
`static_seed_artifact_sha256`, physical drive/time-grid metadata, and compile
backend match within a benchmark point.  If a row is missing or marked failed,
exclude the paired benchmark point from the class aggregate or leave the table
cell provisional rather than mixing in a different seed.
