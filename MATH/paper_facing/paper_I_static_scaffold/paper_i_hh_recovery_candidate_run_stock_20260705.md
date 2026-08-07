# Paper-I HH Recovery/Candidate Run Stock

Created: 2026-07-05

Purpose: define the current Hubbard--Holstein SNAKE recovery/candidate run
contract so agents do not mix visible-row provenance, older source-lock ancestry,
fixed-prefix replay, or diagnostic ordered-batch results.

## Source Layer

Use `visible_row` as the recovery anchor.

```text
visible provenance doc =
  MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_visible_row_provenance_layers_20260705.md

visible support CSV =
  output/pdf/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630_powell_pool_exposure_support.csv
```

Do not use `parent_source_lock` as the default merely because it is present in
source-lock command-audit sidecars.  Do not use fixed selected-generator replay
as adaptive SNAKE recovery.  Recovery and candidate rows must not set
`--phase3-source-lock-preferred-sequence`.

## Approved Anchor Perturbation

The current no-batch SNAKE anchor is an adaptive rerun of the visible-row route
with only the user-approved changes below:

```text
pool_contract                              = full_meta_unfiltered
HVA policy                                 = included_unfiltered_full_meta
phase3_runtime_split_mode                  = shortlist_pauli_children_v1
phase3_runtime_split_selection_mode        = archival_child_set_forward_v1
phase3_runtime_split_child_set_symmetry    = hard_guard
phase3_runtime_split_max_subset_size       = 3
phase1_prune_schur_nomination_route        = metric_regularized_v1
adapt_beam_lambda                          = 0.005
batching                                   = disabled for anchor rows
maxiter/final_refit/max_depth              = 200 / 200 / 30
```

The row label may retain historical words such as `singleton` in older
generators.  For this run stock, the executable contract is cap 3:

```text
--phase3-runtime-split-max-subset-size 3
```

The source visible row requested `hard_guard`, but selected child-set metadata
can have `symmetry_spec = None` and `symmetry_gate.checked = false` because of
the archival missing-spec forwarding path.  Do not relabel that as a new true
child-set hard guard unless the runtime sidecar proves it.

## Staging Order

Run no-batch anchors before any batching variant.

```text
1. POWELL SNAKE no-batch anchors
2. SPSA SNAKE no-batch anchors, using the Paper-I HH SPSA schedule
3. ROTOSOLVE SNAKE no-batch anchors
4. ROTOSOLVE Geo/append comparators with Paper-I historical comparator pools
```

At most two regimes should be active at once.  The machine-readable waves are:

```text
wave0: weak-weak, weak-strong
wave1: intermediate-weak, intermediate-strong
wave2: strong-weak, strong-strong
```

## Batching Variants

Batching is a gated follow-on, not part of the anchor.  Generate it only after
the matching no-batch anchor has completed and passed the settings/result gate.

Approved follow-on batch variants:

```text
greedy_batch_cap3:
  phase2/phase3 batch selection = greedy_reduced_plane
  phase2 batch target/cap       = 3 / 3

combinatorial_batch_cap3:
  phase2/phase3 batch selection = combinatorial_reduced_plane
  phase2 batch target/cap       = 3 / 3
```

Do not identify `B_max=1` or observed `batch_size=1` with the old singleton
admission route.  The admission route must be checked through
`beam_structural_mode` and `batch_selection_mode`.

## ROTOSOLVE Comparators

For ROTOSOLVE only, add comparator rows after or alongside the ROTOSOLVE SNAKE
anchors with the same two-regime wave discipline:

```text
Geo-ADAPT   = Paper-I historical macro-only pool
append-only = Paper-I historical macro-filtered plus Pauli-child filtered pool
```

These comparator rows are not full-meta/HVA SNAKE rows and must not be generated
by silently substituting the SNAKE pool contract.

## Reporting Contract

Every completed result batch must update a LaTeX-built PDF report and matching
machine-readable sidecars.  Raw run outputs and chat summaries are insufficient.

```text
PDF/sidecar root =
  output/pdf/paper_i_hh_recovery_candidate_run_stock_20260705/
```

The PDF must include provenance, settings diffs, iteration plots, Qiskit
compiled terminal costs, and S-work fields.  For beam-enabled SNAKE rows:

```text
S_alg               = winner-lineage terminal work
S_beam_search_total = aggregate all-expanded beam-search work
```

Do not report all-expanded beam work as row-facing `S_alg`.

## Blocking Checks

Block submission or result ingestion if any of the following fail:

```text
--phase3-source-lock-preferred-sequence is absent
full_meta_unfiltered is used and no minus-HVA class filter is present
--phase3-runtime-split-max-subset-size 3 is present for SNAKE
--phase1-prune-schur-nomination-route metric_regularized_v1 is present
--adapt-beam-lambda 0.005 is present
no batching flags are active for no-batch anchors
settings-diff audit has no unapproved non-worker drift
S_alg sidecars use winner-lineage terminal semantics
```

Machine-readable source:

```text
chtc/phase3_optuna/config/paper_i_hh_recovery_candidate_run_stock_20260705_v1.json
```
