# Paper-I HH POWELL Weak-Weak SNAKE Ablation Matrix Audit

Created: 2026-07-07

Purpose: lock the weak-weak Hubbard--Holstein SNAKE ablation matrix schedule
before any additional runs are launched. This audit is agent-facing. It is not
a manuscript edit, not a PDF update, and not a promotion decision.

This audit was prepared after RepoPrompt context-builder review and Oracle
review. The Oracle review agreed with the schedule structure and added two
requirements used below:

- include the no-batch row explicitly as `nobatch_fullv2`, with batch fields
  marked `N/A`;
- require both Phase-2 and Phase-3 batch target/cap fields for batch-on rows.

## Scope And Hard Stops

Scope for the first ablation matrix:

```text
Hamiltonian/regime: weak-weak Hubbard--Holstein
Method:             SNAKE only
Optimizer:          POWELL
Target context:     static_adapt_paper_I_snake_nobatch_promoted_20260707.pdf
Run class:          audit first; no launch implied
```

Hard stops:

```text
Do not launch local jobs from this audit.
Do not submit CHTC jobs from this audit.
Do not edit static_adapt_paper_I.tex from this audit.
Do not edit static_adapt_paper_I_snake_nobatch_promoted_20260707.tex from this audit.
Do not rebuild or overwrite static_adapt_paper_I_snake_nobatch_promoted_20260707.pdf from this audit.
Do not classify historical diagnostics as active weak-weak matrix completion evidence.
Do not use promotion language; the user decides whether any evidence is promoted.
```

## Authority

Executable settings authority:

```text
MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_powell_visible_recovery_candidate_settings_20260706.md
MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_visible_row_provenance_layers_20260705.md
```

Those files override older SPSA/source-lock ancestry, cap-3 diagnostic notes,
and older canonical/shared-pool notes when they disagree.

The duplicate manuscript/PDF is target context and no-batch evidence context,
not the settings authority:

```text
MATH/paper_details/static_adapt_paper_I_snake_nobatch_promoted_20260707.tex
MATH/paper_details/static_adapt_paper_I_snake_nobatch_promoted_20260707.pdf
MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_snake_nobatch_duplicate_promotion_20260707.json
```

Current visible Paper-I HH POWELL support anchor:

```text
MATH/paper_details/static_adapt_paper_I.tex
BEGIN_MACHINE_READABLE_HH_POWELL_POOL_EXPOSURE_MAIN_UPDATE_20260702
BEGIN_MACHINE_READABLE_HH_POWELL_POOL_EXPOSURE_SNAKE_RUNTIME_SPLIT_TRACE_20260705
```

Visible support artifacts:

```text
output/pdf/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630_powell_pool_exposure_support.csv
output/pdf/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630_powell_pool_exposure_support.json
output/pdf/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630_powell_pool_exposure_support.pdf
```

## Fixed Settings Lock

All weak-weak rows in this first ablation matrix must preserve these settings
unless a later user-approved audit explicitly changes them:

```text
optimizer                                      POWELL
adapt_maxiter                                  200
adapt_final_refit_maxiter                      200
adapt_max_depth                                30
adapt_pool                                     full_meta
adapt_pool_class_filter_json                   None
pool interpretation                            unfiltered full_meta, HVA included
phase3_runtime_split_mode                      shortlist_pauli_children_v1
phase3_runtime_split_selection_mode            archival_child_set_forward_v1
phase3_runtime_split_max_subset_size           1
phase3_runtime_split_child_set_symmetry_policy hard_guard label only
phase3_source_lock_preferred_sequence          absent
adapt_beam_lambda                              0.005
adapt_beam_live_branches                       3
adapt_beam_children_per_parent                 2
phase1_prune_schur_nomination_route            metric_regularized_v1
```

Symmetry nuance:

```text
Preserve archival missing-spec child-set behavior.
Do not reinterpret this as a newly enforced child-set hard-guard experiment.
```

The historically visible route uses child-set representatives whose runtime
split metadata can show:

```text
symmetry_gate.checked = false
symmetry_gate.passed = true
symmetry_gate.skipped_reason = runtime_split_symmetry_spec_missing
```

## First Weak-Weak Matrix Rows

The first weak-weak ablation matrix has exactly three rows:

| Row ID | Batch route | Required batch settings | Evidence status |
|---|---|---|---|
| `nobatch_fullv2` | no batch | `phase2_enable_batching=false`; `phase3_enable_batching=false`; target/cap `N/A` | completed, plateau row used in no-batch duplicate |
| `greedy_maxb1` | ordered batch | `phase2/3_enable_batching=true`; Phase-2 and Phase-3 mode `greedy_reduced_plane`; Phase-2 and Phase-3 target/cap `1/1` | completed run; plateau energy and compiled circuit costs present; plateau `S_alg` missing/blank in overlay provenance |
| `combinatorial_maxb1` | ordered batch | `phase2/3_enable_batching=true`; Phase-2 and Phase-3 mode `combinatorial_reduced_plane`; Phase-2 and Phase-3 target/cap `1/1` | completed run; plateau energy and compiled circuit costs present; plateau `S_alg` missing/blank in overlay provenance |

Terminology guard:

```text
maxB=1 prevents multi-record admission, but it does not restore the old
singleton/no-batch route. Future reports must verify beam_structural_mode and
batch_selection_mode before interpreting the ablation.
```

## Completed Evidence Snapshot

The table below records verified weak-weak result JSONs and plateau values from
the overlay report provenance:

```text
output/pdf/paper_i_hh_powell_visible_batchroute_with_paper1_overlay_20260707/paper_i_hh_powell_visible_batchroute_with_paper1_overlay_20260707_provenance.json
SHA256 daeba94c6e4547b6b1af1b1b617701147ee2b74827d0fe91175348eb482062a0

output/pdf/paper_i_hh_powell_visible_batchroute_with_paper1_overlay_20260707/paper_i_hh_powell_visible_batchroute_with_paper1_overlay_20260707_provenance.csv
SHA256 a0befad1e881181703b88945ccf10525c5feff07e3e236c2c61753b1ce938bc1

output/pdf/paper_i_hh_powell_visible_batchroute_with_paper1_overlay_20260707/paper_i_hh_powell_visible_batchroute_with_paper1_overlay_20260707.pdf
SHA256 d92728282064287ffca7a9b47ce3fb0e5cdee0f9e771378b96e1d37369ef95f1
```

Terminal values exist in the earlier 20260706 report, but the ablation matrix
for the no-batch duplicate should use plateau-consistent values unless the user
explicitly asks for a terminal-depth table.

| Row ID | Result JSON | Result SHA256 | k_pl | d_pl | abs(Delta E) at plateau | N2q | D2q | D_c | S_alg plateau status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `nobatch_fullv2` | `raw_outputs/paper_i_hh_powell_visible_batchroute_nobatch_20260706/weak_weak__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__nobatch_fullv2/json/result.json` | `ee696a09816d99c07cf4ca82e165848ea6cfee55eb7014ee45e2c2491ca45262` | 12 | 12 | `0.00039160562082241057` | 42 | 33 | 178 | validated in no-batch-vs-Paper-I provenance as `8012` |
| `greedy_maxb1` | `raw_outputs/paper_i_hh_powell_visible_batchroute_nobatch_20260706/weak_weak__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__greedy_maxb1/json/result.json` | `db2319cab61aa10c548e8ad1ac77d087733624cdc7d4fd585cf9687ea3067d6c` | 12 | 12 | `0.0004001743582682238` | 44 | 37 | 191 | missing from overlay; recompute/validate before table use |
| `combinatorial_maxb1` | `raw_outputs/paper_i_hh_powell_visible_batchroute_nobatch_20260706/weak_weak__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__combinatorial_maxb1/json/result.json` | `f23d83496503eb697b1489503306bb189dfdac2db4700c5643f596da2be55d8f` | 12 | 12 | `0.0004001743582682238` | 44 | 37 | 191 | missing from overlay; recompute/validate before table use |

Terminal-depth values from the 20260706 provenance are allowed only in a
separate terminal-depth table:

| Row ID | terminal k | terminal d | terminal abs(Delta E) | terminal N2q | terminal D2q | terminal D_c | terminal S_alg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nobatch_fullv2` | 30 | 20 | `0.0003766174550794421` | 72 | 51 | 255 | 8689 |
| `greedy_maxb1` | 30 | 20 | `0.0003727777887689854` | 74 | 63 | 322 | 19867 |
| `combinatorial_maxb1` | 30 | 20 | `0.0003727777887689854` | 74 | 63 | 322 | 19867 |

## Historical Diagnostics That Are Not This Matrix

The following are provenance or negative context only. Do not count them as
completed rows for the active weak-weak ablation matrix unless a later audit
explicitly changes the target:

```text
older SPSA parent-source-lock rows
subset-size/cap-3 ancestry rows
lambda-zero diagnostics
fixed-prefix or fixed-ansatz replay/refit checks
old reduced_plane batch route with target/cap 8/16
cap-3 ordered-batch diagnostics in paper_i_hh_visible_row_provenance_layers_20260705.md
row definitions embedded in scripts without exact result JSON and hash validation
```

Paths mentioned in Markdown or scripts are not completion evidence unless an
exact result JSON, source hash, and settings-compatible report/provenance row
are verified.

## Validation Required Before Table Use

Before inserting or replacing any ablation table values, validate:

```text
result_json exists
result_sha256 matches audit/support record
run_command_json exists
run_command_sha256 recorded
regime = weak_weak
optimizer = POWELL
maxiter = 200
final/refit maxiter = 200
depth cap = 30
pool = full_meta
class filter = None
HVA included
runtime split mode = shortlist_pauli_children_v1
runtime split selection = archival_child_set_forward_v1
runtime split max subset size = 1
phase3_source_lock_preferred_sequence absent
adapt_beam_lambda = 0.005
adapt_beam_live_branches = 3
adapt_beam_children_per_parent = 2
metric prune route = metric_regularized_v1
batching fields match the row ID exactly
static_route_id drift is a hard stop unless an exact verified sidecar proves the row and the user explicitly approved that route-label exception
```

For the two batch-on rows, also validate:

```text
phase2_enable_batching = true
phase3_enable_batching = true
phase2_batch_selection_mode = greedy_reduced_plane or combinatorial_reduced_plane
phase3_batch_selection_mode = same as Phase-2
phase2_batch_target_size = 1
phase2_batch_size_cap = 1
phase3_batch_target_size = 1, either emitted natively or recorded as a derived effective value with source/status
phase3_batch_size_cap = 1, either emitted natively or recorded as a derived effective value with source/status
max_observed_batch_size = 1
beam_structural_mode = ordered_batch_admission in the selected trajectory
```

For the no-batch row, validate:

```text
phase2_enable_batching = false
phase3_enable_batching = false
batch target/cap fields are N/A
beam_structural_mode is not ordered_batch_admission; accepted values include single_admission or stop_or_single_admission with batching disabled
```

## S-Work And Cost Semantics

Human-facing ablation tables must keep compiled circuit costs separate from
algorithmic work:

```text
N2q, D2q, D_c = Qiskit compiled costs for the displayed prefix/terminal choice
S_alg         = winner-lineage/display-prefix algorithmic measurement work
S_beam_search_total = total expanded beam-search work, not row-facing S_alg
```

For this weak-weak matrix, the immediate missing reporting work is:

```text
Recompute or validate plateau S_alg for greedy_maxb1.
Recompute or validate plateau S_alg for combinatorial_maxb1.
Keep terminal S_alg separate from plateau S_alg.
```

## Future Report Contract

Expected future support report root, not created by this audit:

```text
output/pdf/paper_i_hh_powell_weak_weak_snake_ablation_matrix_20260707/
```

Expected future files:

```text
paper_i_hh_powell_weak_weak_snake_ablation_matrix_20260707.pdf
paper_i_hh_powell_weak_weak_snake_ablation_matrix_20260707.tex
paper_i_hh_powell_weak_weak_snake_ablation_matrix_20260707.json
paper_i_hh_powell_weak_weak_snake_ablation_matrix_20260707.csv
paper_i_hh_powell_weak_weak_snake_ablation_matrix_20260707_preflight_manifest.json
```

Required future row fields:

```text
row_id
status
evidence_class
provenance_layer
result_json
result_sha256
run_command_json
run_command_sha256
settings_reused
settings_changed
settings_change_reason
batching_enabled
phase2_batch_selection_mode
phase3_batch_selection_mode
phase2_batch_target_size
phase2_batch_size_cap
phase3_batch_target_size
phase3_batch_size_cap
beam_structural_mode
batch_selection_mode
max_observed_batch_size
post_admission_prune_candidate_total
post_admission_prune_accepted_total
k_pl
d_pl
abs_delta_e
one_minus_f
N2q
D2q
D_c
S_alg
S_alg_work_scope
S_beam_search_total
```

## Next Allowed Step

No adaptive rerun is currently required for weak-weak based on this audit.
This audit authorizes no shell commands, report generation, TeX/PDF edits, or
job launches. With later user approval, the next step is report-side validation:

```text
1. Recompute or validate plateau S_alg for greedy_maxb1 and combinatorial_maxb1.
2. Generate a weak-weak-only ablation support PDF/CSV/JSON under the future report root.
3. Do not update the manuscript duplicate until the user approves the support report values.
```
