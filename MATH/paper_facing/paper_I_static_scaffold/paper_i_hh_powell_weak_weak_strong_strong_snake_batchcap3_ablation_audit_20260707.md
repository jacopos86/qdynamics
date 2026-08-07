# Paper-I HH POWELL SNAKE Batch-Cap-3 Ablation Audit

Created: 2026-07-07

Purpose: lock the local SNAKE batch-cap-3 ablation schedule before launching
jobs. This audit is agent-facing and is the active run contract for the next
local work. It does not edit the manuscript, does not update any PDF by itself,
and does not make a promotion decision.

## Scope

Run order:

```text
1. weak-weak only
2. build a standalone support PDF/CSV/JSON after weak-weak completes
3. only after the weak-weak support report is complete, proceed to strong-strong
```

Execution scope:

```text
local only
at most two local adaptive jobs active at once
no CHTC
no manuscript TeX edits
no target duplicate PDF edits
```

Target paper context:

```text
MATH/paper_details/static_adapt_paper_I_snake_nobatch_promoted_20260707.pdf
MATH/paper_details/static_adapt_paper_I_snake_nobatch_promoted_20260707.tex
```

## Authority

Executable baseline authority:

```text
MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_powell_visible_recovery_candidate_settings_20260706.md
MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_visible_row_provenance_layers_20260705.md
```

Visible Paper-I HH POWELL support anchor:

```text
MATH/paper_details/static_adapt_paper_I.tex
BEGIN_MACHINE_READABLE_HH_POWELL_POOL_EXPOSURE_MAIN_UPDATE_20260702
BEGIN_MACHINE_READABLE_HH_POWELL_POOL_EXPOSURE_SNAKE_RUNTIME_SPLIT_TRACE_20260705
```

No-batch duplicate support manifest:

```text
MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_snake_nobatch_duplicate_promotion_20260707.json
```

The duplicate manuscript/PDF is the comparison target. It is not the executable
settings authority.

## Fixed Baseline Settings

All rows in this ablation keep the current copied-PDF SNAKE settings except for
the explicitly approved batch route/cap variable:

```text
method                                         SNAKE
optimizer                                      POWELL
adapt_maxiter                                  200
adapt_final_refit_maxiter                      200
adapt_max_depth                                30
adapt_pool                                     full_meta
adapt_pool_class_filter_json                   None
pool interpretation                            unfiltered full_meta, HVA included
phase3_runtime_split_mode                      shortlist_pauli_children_v1
phase3_runtime_split_selection_mode            archival_child_set_forward_v1
phase3_runtime_split_child_set_symmetry_policy hard_guard label; preserve archival missing-spec behavior
phase3_runtime_split_max_subset_size           1
phase3_source_lock_preferred_sequence          absent
adapt_beam_lambda                              0.005
adapt_beam_live_branches                       3
adapt_beam_children_per_parent                 2
adapt_beam_parent_workers                      2
phase1_prune_schur_nomination_route            metric_regularized_v1
```

The Pauli-child subset cap remains `1`. The newly approved `3` is the batch
target/size cap, not the Pauli-child subset cap.

## Weak-Weak Matrix

Rows for weak-weak:

| Row ID | Launch? | Batch route | Batch settings | Notes |
|---|---:|---|---|---|
| `nobatch_fullv2` | no | no batch | Phase-2/3 batching disabled; target/cap `N/A` | existing anchor from copied PDF support |
| `greedy_cap3` | yes | ordered batch | Phase-2/3 mode `greedy_reduced_plane`; Phase-2/3 target/cap `3/3` | local adaptive run |
| `combinatorial_cap3` | yes | ordered batch | Phase-2/3 mode `combinatorial_reduced_plane`; Phase-2/3 target/cap `3/3` | local adaptive run |

The weak-weak no-batch anchor is:

```text
raw_outputs/paper_i_hh_powell_visible_batchroute_nobatch_20260706/weak_weak__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__nobatch_fullv2/json/result.json
SHA256 ee696a09816d99c07cf4ca82e165848ea6cfee55eb7014ee45e2c2491ca45262
```

## Strong-Strong Matrix

Strong-strong must not start until the weak-weak support report exists. The
planned rows are the same three rows:

| Row ID | Launch? | Batch route | Batch settings | Notes |
|---|---:|---|---|---|
| `nobatch_fullv2` | no | no batch | Phase-2/3 batching disabled; target/cap `N/A` | existing anchor from copied PDF support |
| `greedy_cap3` | later | ordered batch | Phase-2/3 mode `greedy_reduced_plane`; Phase-2/3 target/cap `3/3` | local adaptive run after weak-weak report |
| `combinatorial_cap3` | later | ordered batch | Phase-2/3 mode `combinatorial_reduced_plane`; Phase-2/3 target/cap `3/3` | local adaptive run after weak-weak report |

## Route-Label Exception

Batch-enabled `greedy_reduced_plane` and `combinatorial_reduced_plane` rows may
use:

```text
--static-route-id unspecified
```

This exception is allowed only for the batch-enabled cap-3 rows because the
batch route intentionally differs from the visible `route_a` batch surface. It
does not authorize other route drift.

## Validation Gates

Before launch, each generated weak-weak command must satisfy:

```text
problem = hh
regime = weak-weak
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
phase2_enable_batching = true
phase3_enable_batching = true
phase2_batch_selection_mode = row-specific greedy/combinatorial
phase3_batch_selection_mode = same as Phase-2
phase2_batch_target_size = 3
phase2_batch_size_cap = 3
phase3_batch_target_size = 3
phase3_batch_size_cap = 3
```

Validation after completion:

```text
result_json exists
current_json exists or result_json contains enough trajectory data
max observed batch size recorded
beam_structural_mode recorded
post-admission prune candidate/accepted counts recorded when available
plateau prefix selected by the same support-report rule used for the copied PDF
plateau and terminal Qiskit costs computed separately
plateau and terminal S_alg computed separately
S_alg is winner-lineage/display-prefix work, not aggregate beam-search work
S_beam_search_total retained separately when available
```

## Reporting Contract

First deliverable after weak-weak completes:

```text
output/pdf/paper_i_hh_powell_weak_weak_snake_batchcap3_ablation_20260707/
```

Expected files:

```text
paper_i_hh_powell_weak_weak_snake_batchcap3_ablation_20260707.pdf
paper_i_hh_powell_weak_weak_snake_batchcap3_ablation_20260707.tex
paper_i_hh_powell_weak_weak_snake_batchcap3_ablation_20260707.csv
paper_i_hh_powell_weak_weak_snake_batchcap3_ablation_20260707.json
paper_i_hh_powell_weak_weak_snake_batchcap3_ablation_20260707_manifest.json
```

The PDF should contain at least two human-facing sections/pages:

```text
1. plateau-prefix comparison
2. terminal-depth comparison
```

Each row should report:

```text
row_id
batch route
k
d_ans
abs(Delta E)
1-F when computable
N2q
D2q
D_c
S_alg
S_beam_search_total when available
result_json
result_sha256
settings_changed
```

Do not update the manuscript duplicate until the support PDF values are reviewed
and the user explicitly asks for a manuscript/PDF update.
