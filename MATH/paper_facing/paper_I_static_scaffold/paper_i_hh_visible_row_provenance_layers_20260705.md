# Paper-I HH Visible-Row Provenance Layers

Created: 2026-07-05

Purpose: prevent agents from mixing the currently rendered Hubbard--Holstein
Paper-I row with older source-lock ancestry or fresh diagnostic reruns.

## Rule

For recovery of an already displayed Paper-I HH row, use this priority:

1. `visible_row`: support CSV/JSON plus result/effective-command JSON that
   produced the currently rendered manuscript/PDF row.
2. `parent_source_lock`: older ancestry captured in command-audit sidecars.
3. `diagnostic_rerun`: local or CHTC reruns created to test recovery or
   perturbations.

The default recovery anchor is `visible_row`.  Do not use
`parent_source_lock` settings as the default just because they are present in
`source_lock_command_audit.json`.

Current recovery/candidate settings contract:

```text
MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_powell_visible_recovery_candidate_settings_20260706.md
```

That settings contract keeps the resolved visible-row Phase-III child-set cap
`1`, adds the metric-regularized prune route, and fixes
`adapt_beam_lambda=0.005`.  It still requires adaptive SNAKE selection; fixed
selected-prefix refits are not recovery evidence.

The older 2026-07-05 run-stock/config that used cap-3 is historical/diagnostic
for this line unless the user explicitly reopens a child-subset-size diagnostic.
Do not use cap-3 as the default current visible-row recovery setting.

## Current Page-8 Anchor

Current visible block in `MATH/paper_details/static_adapt_paper_I.tex`:

```text
BEGIN_MACHINE_READABLE_HH_POWELL_POOL_EXPOSURE_MAIN_UPDATE_20260702
```

Current support CSV:

```text
output/pdf/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630_powell_pool_exposure_support.csv
```

Weak-weak SNAKE visible-row source result:

```text
output/chtc_retrievals/paper_i_hh_fullmeta_singleton_symmetry_20260630_current_fetch/raw_outputs/paper_i_hh_fullmeta_singleton_symmetry_20260630_schedfix_powell/paper_i_hh_fullmeta_singleton_symmetry_20260630_schedfix_powell__weak_weak__snake__A_native_staged_singleton_hard_guard__native_forced__powell200__depth30_noearlystop__fullmeta_singleton_symmetry/json/result.json
```

Resolved visible-row settings for the weak-weak SNAKE anchor:

```text
optimizer = POWELL
adapt_pool = full_meta
adapt_pool_class_filter_json = None
pool interpretation = unfiltered full_meta, HVA included
phase3_runtime_split_mode = shortlist_pauli_children_v1
phase3_runtime_split_selection_mode = archival_child_set_forward_v1
phase3_runtime_split_child_set_symmetry_policy = hard_guard
phase3_runtime_split_max_subset_size = 1
shared_pauli_pool_mode = off
adapt_maxiter = 200
adapt_final_refit_maxiter = 200
adapt_max_depth = 30
phase3_source_lock_preferred_sequence = empty
```

Important enforcement nuance:

- The row label and effective command use `hard_guard`.
- The selected SNAKE records in the visible weak-weak anchor are
  `child_set[...]` representatives.
- Those selected child-set representatives carry `symmetry_spec = None`.
- Their runtime-split metadata records
  `symmetry_gate.checked = false`,
  `symmetry_gate.passed = true`, and
  `symmetry_gate.skipped_reason = runtime_split_symmetry_spec_missing`.

Therefore the historically accurate recovery route is not a newly enforced
child-set hard guard.  It is an archival route that hard-guards child
construction but forwards the child-set representative with missing
child-set symmetry spec.  Recovery reruns must preserve that effective
metadata behavior unless the user explicitly asks for a true child-set
hard-guard ablation.

The older ancestry row recorded in command-audit sidecars used SPSA and
subset size `3`.  That is `parent_source_lock`, not the default current
visible-row recovery target.

## Required Rerun Manifest Fields

Every recovery or perturbation rerun should record:

```text
provenance_layer = visible_row | parent_source_lock | diagnostic_rerun
visible_support_csv
visible_source_result_json
visible_effective_command_json
settings_reused
settings_changed
settings_change_reason
```

For the current recovery line, a fresh adaptive rerun must not set
`--phase3-source-lock-preferred-sequence`; that would be fixed-ansatz replay,
not adaptive SNAKE recovery.

## Weak-Weak Adaptive Recovery Audit (2026-07-05)

Bounded adaptive recovery run:

```text
raw_outputs/paper_i_hh_algorithmic_recovery_20260705_v4/weak_weak_powell_visible_a1_subset1_archival_missing_spec_depth11/json/result.json
```

This run is an adaptive SNAKE rerun, not fixed-sequence replay:

```text
phase3_source_lock_preferred_sequence = absent
phase3_runtime_split_selection_mode = archival_child_set_forward_v1
phase3_runtime_split_child_set_symmetry_policy = hard_guard
phase3_runtime_split_max_subset_size = 1
adapt_max_depth = 11
```

The bounded run recovers the visible weak-weak plateau energy at its terminal
accepted depth, not at the same displayed prefix index:

```text
source visible-row selected-prefix k        = 10
source visible-row selected-prefix error    = 3.968023977350965e-4
source visible-row selected-prefix costs    = N2q 34, D2q 28, D_c 148, S_alg 6531
adaptive recovery terminal k/depth          = 11 / 11
adaptive recovery final error               = 3.968023969321832e-4
adaptive recovery terminal Qiskit costs     = N2q 40, D2q 34, D_c 179, S_alg 7287, S_beam_search_total 9037
adaptive recovery k=10 error                = 5.306791688701740e-4
adaptive recovery k=10 Qiskit costs         = N2q 32, D2q 26, D_c 151, S_alg 6548
```

The terminal Qiskit costs in this note use the current report-builder sidecar
convention `table_i_basis_gate_transpile_v1`, with circuit scope
`ansatz_circuit_including_reference_state`.  Each local diagnostic root has a
generated `paper_i_terminal_qiskit_cost.json` sidecar with the source JSON hash.
Selected-prefix rows use the same compiler through the history-prefix path.
For beam-enabled SNAKE rows, row-facing `S_alg` is the winner-lineage terminal
work aligned with the displayed terminal ansatz, energy error, and Qiskit costs.
The all-expanded beam-search work is retained separately as
`S_beam_search_total`; it is not the row-facing `S_alg`.

The selected labels and selected-prefix index are not required to match exactly
for this recovery gate.  Several selected records differ by spin-degenerate or
near-degenerate child-set labels, and the local rerun reaches the visible
plateau energy one accepted step later.  It preserves the archival child-set
representative metadata and recovers the same plateau energy without a
preferred-sequence source lock.  Future prune, beam, or batch perturbations
should use this bounded adaptive recovery as the energy-equivalent anchor unless
the user asks for strict selected-label identity or strict selected-prefix
resource identity.

First perturbation check:

```text
raw_outputs/paper_i_hh_algorithmic_recovery_20260705_v5/weak_weak_powell_visible_a1_subset1_archival_missing_spec_depth11_beam_lambda_0p005/json/result.json
```

This run keeps the recovered route and enables `adapt_beam_lambda=0.005` with
the recoverability-prune controls still active.  It also recovers the same
bounded plateau error:

```text
beam-lambda/prune final error = 3.968023969321832e-4
max observed batch_size       = 1
post-admission prune executed = 0 times
terminal Qiskit costs         = N2q 40, D2q 34, D_c 179, S_alg 7287, S_beam_search_total 9037
```

Thus nonzero beam cost weighting at this small value did not disturb the
weak-weak source recovery.  It did not test actual multi-record batch
admission, because every recorded admission remained singleton.

Ordered-batch diagnostic:

```text
raw_outputs/paper_i_hh_algorithmic_recovery_20260705_v7/weak_weak_powell_visible_a1_subset1_archival_missing_spec_depth11_beam_lambda_0p005_greedy_ordered_batch_route_unspecified/json/result.json
```

This run keeps the same bounded weak-weak source-recovery target and enables
the greedy ordered-batch admission path with `adapt_beam_lambda=0.005`.
Because the historical route guard rejects `greedy_reduced_plane` under
`static_route_id=route_a`, this diagnostic used `static_route_id=unspecified`.
It did not preserve the recovered plateau:

```text
source/recovery plateau error        = 3.968023969321832e-4
greedy ordered-batch diagnostic error = 5.045663335210282e-4
absolute degradation                  = 1.0776393658884498e-4
max observed batch_size               = 1
post-admission prune executed         = 0 times
terminal Qiskit costs                 = N2q 38, D2q 32, D_c 157, S_alg 6266, S_beam_search_total 13149
```

Since every recorded admission remained singleton and prune did not execute,
the observed drift is attributable to the ordered-batch/beam selection path
itself, not to an accepted multi-record batch or to post-admission pruning.

Combinatorial ordered-batch diagnostic:

```text
raw_outputs/paper_i_hh_algorithmic_recovery_20260705_v8/weak_weak_powell_visible_a1_subset1_archival_missing_spec_depth11_beam_lambda_0p005_combinatorial_ordered_batch_route_unspecified/json/result.json
```

This run differs from the greedy diagnostic only by changing
`greedy_reduced_plane` to `combinatorial_reduced_plane`.  It gives the same
degraded depth-11 value:

```text
source/recovery plateau error              = 3.968023969321832e-4
combinatorial ordered-batch diagnostic error = 5.045663335210282e-4
absolute degradation                        = 1.0776393658884498e-4
max observed batch_size                     = 1
post-admission prune executed               = 0 times
batch_selected                              = false for every admission
terminal Qiskit costs                       = N2q 38, D2q 32, D_c 157, S_alg 6266, S_beam_search_total 13149
```

The first two admissions match the recovered anchor.  The first trajectory
change occurs at admission 3:

```text
recovered anchor admission 3 = paop_lf_full:paop_dbl_p(site=0->phonon=0)::child_set[4]
ordered-batch admission 3    = paop_lf_full:paop_dbl_p(site=1->phonon=0)::child_set[4]
```

Lambda-zero ordered-batch isolation:

```text
raw_outputs/paper_i_hh_algorithmic_recovery_20260705_v9/weak_weak_powell_visible_a1_subset1_archival_missing_spec_depth11_beam_lambda_0_combinatorial_ordered_batch_route_unspecified/json/result.json
```

This run keeps the combinatorial ordered-batch route but sets
`adapt_beam_lambda=0.0`.  It recovers the source-quality plateau:

```text
source/recovery plateau error                 = 3.968023969321832e-4
combinatorial ordered-batch lambda=0 error    = 3.968023961944400e-4
combinatorial ordered-batch lambda=0.005 error = 5.045663335210282e-4
max observed batch_size                       = 1
post-admission prune executed                 = 0 times
batch_selected                                = false for every admission
```

The lambda-zero ordered-batch trajectory is not strict selected-label identity:
its third admitted child is the mirrored
`paop_lf_full:paop_dbl_p(site=1->phonon=0)::child_set[4]`, while the recovered
anchor used `site=0`.  Nevertheless it reaches the same weak-weak depth-11
energy/fidelity plateau.  Under the current terminal Qiskit sidecar convention,
its two-qubit count/depth match the anchor but total circuit depth and
algorithmic work differ:

```text
ordered-batch lambda=0 costs = N2q 40, D2q 34, D_c 184, S_alg 7322, S_beam_search_total 15240
recovered anchor costs       = N2q 40, D2q 34, D_c 179, S_alg 7287, S_beam_search_total 9037
```

Thus, at least for this bounded weak-weak check, the ordered-batch plumbing by
itself does not force an energy degradation when `adapt_beam_lambda=0.0` and
the batch cardinality remains one.  It is not resource-identical to the
recovered anchor, and later cap-3 runs show that actual batch admission can still
degrade the plateau.

Actual-batch cap-3 diagnostic:

```text
raw_outputs/paper_i_hh_algorithmic_recovery_20260705_v10/weak_weak_powell_visible_a1_subset1_archival_missing_spec_depth11_beam_lambda_0_combinatorial_batch_cap3_children3/json/result.json
```

This run keeps `adapt_beam_lambda=0.0` and uses the combinatorial ordered-batch
path, but widens the new route to the intended batch/beam shell by setting
`adapt_beam_children_per_parent=3` and `phase2_batch_target_size =
phase2_batch_size_cap = 3`.  Therefore it is a diagnostic of the new cap-3
beam/batch surface, not a pure one-field perturbation from the recovered
visible-row anchor.

```text
source/recovery plateau error          = 3.968023969321832e-4
cap-3 actual-batch lambda=0 error      = 4.771404844567950e-4
controller iterations                  = 11
final logical depth                    = 12
batch-size sequence                    = 1,1,1,1,1,1,1,2,1,1,1
post-admission prune executed          = 0 times
terminal Qiskit costs                  = N2q 42, D2q 36, D_c 178, S_alg 7808, S_beam_search_total 19253
```

The first admissions no longer match the recovered source-quality anchor:

```text
recovered anchor admission 1 = uccsd_ferm_lifted::uccsd_sing(alpha:0->1)::child_set[0]
cap-3 diagnostic admission 1 = hh_fermionic_reusable::opp_spin_assist_current_nn_up(0,1)::child_set[0]
```

This shows that widening the beam child shell to 3 changes the adaptive
trajectory before the first accepted multi-record batch.  A size-2 batch is
eventually admitted at controller iteration 8, but the energy degradation cannot
be assigned to batching alone; it is the combined cap-3 beam/batch route.

Beam-shell-width isolation:

```text
raw_outputs/paper_i_hh_algorithmic_recovery_20260705_v11/weak_weak_powell_visible_a1_subset1_archival_missing_spec_depth11_children3_only_legacy_batch_route/json/result.json
```

This run keeps the recovered `route_a` legacy admission path and changes only
`adapt_beam_children_per_parent` from 2 to 3.  It recovers the anchor exactly:

```text
source/recovery plateau error             = 3.968023969321832e-4
children-per-parent=3 legacy-route error  = 3.968023969321832e-4
max observed batch_size                   = 1
post-admission prune executed             = 0 times
terminal Qiskit costs                     = N2q 40, D2q 34, D_c 179, S_alg 7287, S_beam_search_total 9037
first three admitted labels               = identical to recovered anchor
```

Therefore the cap-3 degradation above is not caused by increasing the beam child
shell width alone.  It appears only when the widened shell is coupled to the
new ordered-batch admission route that can select a size-2 batch.

Ordered-batch cap-1 isolation:

```text
raw_outputs/paper_i_hh_algorithmic_recovery_20260705_v12/weak_weak_powell_visible_a1_subset1_archival_missing_spec_depth11_beam_lambda_0_ordered_children3_batch_cap1/json/result.json
```

This run keeps the combinatorial ordered-batch route, keeps
`adapt_beam_lambda=0.0`, and keeps `adapt_beam_children_per_parent=3`, but
forces `phase2_batch_target_size = phase2_batch_size_cap = 1`.  Thus it tests
the ordered-batch route with the widened child shell while preventing actual
multi-record batch admission.

```text
source/recovery plateau error                 = 3.968023969321832e-4
children=3 ordered-batch cap=1 error          = 3.968023961944400e-4
max observed batch_size                       = 1
batch_selected                                = false for every admission
post-admission prune executed                 = 0 times
terminal Qiskit costs                         = N2q 40, D2q 34, D_c 184, S_alg 7322, S_beam_search_total 15240
```

The cap-1 ordered-batch run follows the same energy/cost outcome as the
lambda-zero ordered-batch isolation, not the degraded cap-3 actual-batch run.
Therefore the weak-weak degradation in the cap-3 diagnostic is attributable to
actual multi-record batch admission, not to `adapt_beam_children_per_parent=3`
alone and not to ordered-batch plumbing with batch cardinality fixed at one.

Greedy cap-3 actual-batch diagnostic:

```text
raw_outputs/paper_i_hh_algorithmic_recovery_20260705_v13/weak_weak_powell_visible_a1_subset1_archival_missing_spec_depth11_beam_lambda_0_greedy_batch_cap3_children3/json/result.json
```

This run differs from the cap-3 combinatorial diagnostic only by changing
`phase2_batch_selection_mode` and `phase3_batch_selection_mode` from
`combinatorial_reduced_plane` to `greedy_reduced_plane`.  It gives the same
weak-weak outcome:

```text
source/recovery plateau error          = 3.968023969321832e-4
greedy cap-3 actual-batch error        = 4.771404844567950e-4
controller iterations                  = 11
final logical depth                    = 12
batch-size sequence                    = 1,1,1,1,1,1,1,2,1,1,1
post-admission prune executed          = 0 times
terminal Qiskit costs                  = N2q 42, D2q 36, D_c 178, S_alg 7808, S_beam_search_total 20268
```

The size-2 batch selected at admission 8 is the same in the greedy and
combinatorial cap-3 runs:

```text
hh_phonon::s(site=1)::child_set[0]
hh_phonon::s(site=0)::child_set[0]
```

Thus the observed cap-3 weak-weak degradation is not a greedy-versus-
combinatorial implementation discrepancy.  For this source-locked diagnostic,
both ordered-batch modes choose the same actual two-record batch and land at the
same degraded plateau relative to the recovered singleton source route.

Metric-prune route plus nonzero beam-lambda diagnostic:

```text
raw_outputs/paper_i_hh_algorithmic_recovery_20260705_v14/weak_weak_powell_visible_a1_subset1_archival_missing_spec_depth11_beam_lambda_0p005_metric_prune_route/json/result.json
```

This run starts from the nonzero-beam-lambda recovery row and changes only the
Schur prune nomination route from the historical Hessian-coupling route to the
metric-regularized route:

```text
--adapt-beam-lambda 0.005
--phase1-prune-schur-nomination-route metric_regularized_v1
```

The metric route is active in runtime telemetry at every depth, but it still
does not admit a deletion under the source-locked weak-weak plateau prefix:

```text
source/recovery plateau error               = 3.968023969321832e-4
lambda=0.005 + metric-prune-route error     = 3.968023969321832e-4
controller iterations                       = 11
final logical depth                         = 11
batch-size sequence                         = 1,1,1,1,1,1,1,1,1,1,1
metric-prune route active                   = true
post-admission prune candidate total        = 0
post-admission prune accepted total         = 0
terminal Qiskit costs                       = N2q 40, D2q 34, D_c 179, S_alg 7287, S_beam_search_total 9037
```

The final `adapt_vqe.operators` list is identical to the recovered anchor:

```text
uccsd_ferm_lifted::uccsd_sing(alpha:0->1)::child_set[0]
uccsd_ferm_lifted::uccsd_sing(beta:2->3)::child_set[0]
paop_lf_full:paop_dbl_p(site=0->phonon=0)::child_set[4]
paop_full:paop_disp(site=1)::child_set[4]
paop_full:paop_cloud_p(site=1->phonon=0)::child_set[0]
paop_lf_full:paop_dbl_p(site=1->phonon=1)::child_set[8]
hh_fermionic_reusable::bond_charge_current_nn_up(0,1)::child_set[2]
hh_phonon::s(site=0)::child_set[0]
paop_sq_full:paop_hop_sq(0,1)::child_set[8]
hh_fermionic_reusable::opp_spin_assist_current_nn_up(0,1)::child_set[2]
paop_sq_full:paop_pair_sq(0,1)::child_set[7]
```

Therefore `--adapt-beam-lambda 0.005` together with the metric-prune nomination
route is non-disruptive for this recovered weak-weak prefix, but only because
the prune ladder has zero candidates and no deletion is accepted.  This is not
yet evidence that metric pruning improves the route; it is evidence that the
new prune route can be enabled without changing the recovered plateau when no
prune candidate is nominated.

Combined metric-prune, nonzero beam-lambda, and ordered-batch diagnostic:

```text
raw_outputs/paper_i_hh_algorithmic_recovery_20260705_v15/weak_weak_powell_visible_a1_subset1_archival_missing_spec_depth11_beam_lambda_0p005_metric_prune_combinatorial_batch_cap3_children3/json/result.json
```

This run is the first combined new-route diagnostic after the recovered anchor.
It enables:

```text
--adapt-beam-lambda 0.005
--phase1-prune-schur-nomination-route metric_regularized_v1
--phase2-enable-batching
--phase3-enable-batching
--phase2-batch-selection-mode combinatorial_reduced_plane
--phase3-batch-selection-mode combinatorial_reduced_plane
--phase2-batch-target-size 3
--phase2-batch-size-cap 3
--adapt-beam-children-per-parent 3
```

Observed weak-weak result:

```text
source/recovery plateau error                    = 3.968023969321832e-4
lambda=0.005 + metric-prune + ordered batch      = 5.045663342047035e-4
controller iterations                            = 11
final logical depth                              = 11
batch-size sequence                              = 1,1,1,1,1,1,1,1,1,1,1
metric-prune route active                        = true
post-admission prune candidate total             = 0
post-admission prune accepted total              = 0
terminal Qiskit costs                            = N2q 38, D2q 32, D_c 157, S_alg 6369, S_beam_search_total 19014
first admitted label                             = hh_fermionic_reusable::opp_spin_assist_current_nn_up(0,1)::child_set[0]
```

Thus the combined route does not recover the source plateau, even though no
multi-record batch is ultimately admitted and no prune candidate is nominated.
The degradation is tied to entering the ordered-batch admission family with
nonzero beam cost, not to an accepted deletion and not to an observed
`batch_size > 1` event in this particular run.  The immediately safe setting
pair for weak-weak is therefore:

```text
--adapt-beam-lambda 0.005
--phase1-prune-schur-nomination-route metric_regularized_v1
```

with ordered batching still treated as an unsettled diagnostic variable rather
than part of the recovered source-equivalent route.

Greedy counterpart for the combined diagnostic:

```text
raw_outputs/paper_i_hh_algorithmic_recovery_20260705_v16/weak_weak_powell_visible_a1_subset1_archival_missing_spec_depth11_beam_lambda_0p005_metric_prune_greedy_batch_cap3_children3/json/result.json
```

This run changes only the ordered-batch selection mode relative to `v15`:

```text
--phase2-batch-selection-mode greedy_reduced_plane
--phase3-batch-selection-mode greedy_reduced_plane
```

Observed weak-weak result:

```text
source/recovery plateau error               = 3.968023969321832e-4
lambda=0.005 + metric-prune + greedy batch  = 4.771404844567950e-4
controller iterations                       = 11
final logical depth                         = 12
batch-size sequence                         = 1,1,1,1,1,1,1,2,1,1,1
metric-prune route active                   = true
post-admission prune candidate total        = 0
post-admission prune accepted total         = 0
terminal Qiskit costs                       = N2q 42, D2q 36, D_c 178, S_alg 7808, S_beam_search_total 20268
first admitted label                        = hh_fermionic_reusable::opp_spin_assist_current_nn_up(0,1)::child_set[0]
```

The greedy combined diagnostic matches the earlier cap-3 actual-batch
degradation pattern: a size-2 batch is admitted at controller iteration 8, the
logical depth becomes 12, and the plateau error is worse than the recovered
source route.  Because both combined diagnostics have zero prune candidates,
the current weak-weak evidence separates the knobs as follows:

```text
metric-prune route + lambda=0.005, no ordered batching: recovered
ordered batching with cap=3, lambda=0.0: degraded when size-2 batch fires
ordered batching with cap=3, lambda=0.005: degraded, with or without final size-2 admission depending on mode
```

The next source-locked batching work should therefore isolate and repair the
ordered-batch admission comparator before expanding to broader regimes or
claiming a batch-improved route.

## S_alg Work-Accounting Robustness Gate

For beam-enabled SNAKE rows, never read a terminal scalar `S_alg` without first
checking its semantics.  Paper-I report sidecars must distinguish:

```text
S_alg               = winner-lineage terminal work aligned with the displayed
                      ansatz, energy error, and Qiskit costs
S_beam_search_total = aggregate all-expanded beam-search work, including losing
                      branches
```

The aggregate beam-search total is useful provenance, but it is not the
row-facing `S_alg` in Paper-I tables, plots, or human PDFs.

Accepted terminal sidecars should carry:

```text
work_semantics_version = snake_terminal_s_alg_winner_lineage_v1
S_alg_work_scope       = winner_lineage_terminal
S_alg_row_policy       = beam_terminal_winner_history_v1
S_beam_search_scope    = all_expanded_scored_branches
```

Before reusing old Paper-I SNAKE sidecars in a PDF/table update, run the audit
command on the exact report or raw-output roots being consumed:

```bash
python3 agent_guidance/skills/paper-i-results/scripts/audit_paper_i_snake_s_alg_sidecars.py \
  <sidecar-file-or-report-root> \
  --output-json output/pdf/paper_i_snake_s_alg_sidecar_audit.json \
  --output-csv output/pdf/paper_i_snake_s_alg_sidecar_audit.csv
```

The command fails nonzero when it finds stale or unsafe work semantics.  Use
`--allow-issues` only when intentionally producing an audit report of known
stale evidence; do not use stale rows as displayed Paper-I work values.
