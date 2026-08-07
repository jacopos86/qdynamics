# Paper-I HH Weak-Weak SNAKE Mechanism Ablation CHTC Plan

Status: agent-facing run contract, not manuscript prose.

## Objective

Prepare a CHTC batch for the Paper-I Hubbard--Holstein weak-weak SNAKE mechanism-ablation matrix. This batch intentionally has two source-anchor families:

1. `batch_cap3_combinatorial`: the existing weak-weak combinatorial ordered-batch cap-3 row.
2. `physical_operator_lane`: a queued weak-weak physical-operator-lane source-anchor rebuild with batching enabled, combinatorial reduced-plane selection, and batch target/cap `3/3`. The July 8 physical-operator-lane support PDF provides the no-batch parent command/provenance only.

The two families use the same ablation definitions wherever meaningful. Reference rows that already exist locally are recorded in the manifest but are not queued.

## Hard Scope

- Run class: `candidate`.
- Hamiltonian regime: weak-weak only.
- Method: SNAKE only.
- Optimizer: POWELL.
- Optimizer budget: maxiter 200; final/refit maxiter 200.
- Depth cap: 30.
- Pool: unfiltered `full_meta`; HVA included; no class-filter JSON.
- SNAKE child policy: native Phase-III archival Pauli-child split unless the row explicitly tests Phase-I-only macro or singleton pools.
- Runtime split mode: `shortlist_pauli_children_v1`.
- Runtime split selection: `archival_child_set_forward_v1`.
- Pauli-child subset cap: `1`.
- Beam: live branches `3`, children per parent `2`, beam lambda `0.005`.
- Metric prune route: `metric_regularized_v1` where the source route carries metric prune.
- No CHTC submission should occur until generated records and preflight pass.

## Source Anchors

### Combinatorial batch cap-3 anchor

- Source result:
  `raw_outputs/paper_i_hh_powell_weak_weak_snake_batchcap3_ablation_20260707/weak_weak__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__combinatorial_cap3/json/result.json`
- Source result SHA-256:
  `bfc80890b7f1086b43dc9d6e82838273644a1bec5c14b584242ea0918c49b1aa`
- Source command:
  `raw_outputs/paper_i_hh_powell_weak_weak_snake_batchcap3_ablation_20260707/weak_weak__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__combinatorial_cap3/run_command.json`
- Source command SHA-256:
  `ca8b79809835f2b9a327c51e2c153ecbb8a6f138f2204c407b13d7b09e62c9a5`
- Anchor semantics:
  source has Phase-II/Phase-III batching enabled with `combinatorial_reduced_plane`, target/cap `3/3`, but Pauli-child subset cap remains `1`.

### Physical operator lane parent and queued batch source anchor

- Source support PDF:
  `output/pdf/paper_i_hh_physical_operator_lane_comparison_20260708/paper_i_hh_physical_operator_lane_comparison_20260708.pdf`
- Existing no-batch parent result:
  `raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/weak_weak/json/result.json`
- Existing no-batch parent result SHA-256:
  `bb51341389bac493f99fac05bd425f6cdfca28a1d87983aa812d979b6301d1cb`
- Existing no-batch parent command manifest:
  `raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/commands.json`
- Existing no-batch parent command manifest SHA-256:
  `aaaef244b7f2a7dbe71bbc2a2062ab2b1855bf5e684d0c2cf768815bfb5d6238`
- Source lock manifest:
  `raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/source_lock_manifest.json`
- Source lock manifest SHA-256:
  `efec56ddc81c65f8dbeaa1aa00129e24f57b62f2f79f923845a4d6193b987c76`
- Parent semantics:
  the existing parent command uses `--static-lane-route physical_operator_type`, `--physical-lane-shortlist-aggressiveness 3`, and no batching.
- Queued source-anchor semantics:
  the physical-lane source anchor to be used for this doubled matrix is the queued `physical_operator_lane__combinatorial_cap3` row. It enables Phase-II/Phase-III batching with `combinatorial_reduced_plane`, target/cap `3/3`, while preserving Pauli-child subset cap `1`.

## Queue Matrix

Rows are generated for each source-anchor family. Reference-only rows are kept in TSV/manifest provenance but are not queued.

| Variant | Queued for combinatorial anchor | Queued for physical anchor | Notes |
|---|---:|---:|---|
| `full_anchor_reference` | no | no | Existing source row. |
| `no_batching_reference` | no | no | Existing no-batch reference for combinatorial family; duplicate of physical anchor for physical family. |
| `greedy_cap3` | no | yes | Existing weak-weak greedy cap-3 row for combinatorial family; physical-lane greedy batch comparison is queued. |
| `combinatorial_cap3` | no | yes | Existing anchor for combinatorial family; physical-lane combinatorial batch-cap-3 source-anchor rebuild is queued. |
| `no_prune` | yes | yes | Disable Phase-I prune/recoverability deletion. |
| `no_cost_term` | yes | yes | Zero selector cost denominators in Phase I/II/III; keep beam lambda `0.005`. |
| `no_novelty` | yes | yes | Disable Phase-II and Phase-III novelty. |
| `phase2_novelty_only_no_second_order` | yes | yes | Disable Phase-II second-order gain and disable Phase III. |
| `phase2_second_order_only_no_novelty` | yes | yes | Disable Phase-II novelty and disable Phase III. |
| `no_phase3` | yes | yes | Use Phase-I+II continuation only. |
| `phase1_only_macro_pool` | yes | yes | Phase-I-only macro/operator pool, no Pauli-child split. |
| `phase1_only_singleton_pool` | yes | yes | Phase-I-only shared singleton Pauli-child pool. |
| `full_geometry_window` | yes | yes | Use raw exact geometry selector instead of reduced selector. |
| `no_shortlisting` | no | no | Blocked until a separate audited no-shortlisting route opens all gates and maturity caps. |

Expected runnable rows: 20.

## Preflight Requirements

Every queued row must pass:

- `display_regime=weak-weak`.
- `family=hh`.
- `case_id=hh_L2_nph2_three_model_sym_weak_weak`.
- `suite_profile=paper_i_three_model_hh_symmetric_20260527_v1`.
- `pool_contract=full_meta_unfiltered`.
- no `hh_full_meta_minus_hva_class_filter.json`.
- `optimizer=POWELL`; `adapt_optimizer_kind=powell`.
- `budget=200`; `max_depth=30`.
- native Phase-III rows use `shortlist_pauli_children_v1`, `archival_child_set_forward_v1`, and subset cap `1`.
- batch variants use batch target/cap `3/3` without changing Pauli-child subset cap.
- no-cost variants preserve `--adapt-beam-lambda 0.005`.
- physical family rows preserve `--static-lane-route physical_operator_type` unless the row explicitly changes to Phase-I-only pool semantics.

## Files

- Generator:
  `chtc/phase3_optuna/generate_paper_i_hh_weak_weak_snake_mechanism_ablation_records.py`
- Preflight:
  `chtc/phase3_optuna/preflight_submit.py`
- Test:
  `test/chtc/test_paper_i_hh_weak_weak_snake_mechanism_ablation_records.py`
