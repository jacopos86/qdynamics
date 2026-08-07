# Oracle Plan

## Delta plan

### Validated interpretation

- **Yes:** `physical_operator_lane` family uses the existing physical-lane weak-weak row as the **Full anchor**.
- **Yes:** physical-lane **No batching** is a duplicate of that anchor because source argv already has `phase2-no-batching` and `phase3-no-batching`; do **not** queue it separately.
- **Yes:** physical-lane ablation rows preserve `static_lane_route=physical_operator_type` and `physical_lane_shortlist_aggressiveness=3` unless the specific variant is explicitly defined as a shortlisting / Phase-I mechanics ablation.
- **Yes:** `phase2_v1` rows must override `--phase3-backend-cost-mode proxy`; apply the same rule to any `phase1_v1` row if present.
- **Yes:** no-cost ablation zeros only Phase-I/II/III **score-denominator cost weights**. It must preserve `--adapt-beam-lambda 0.005`.
- **Yes:** no-shortlisting stays blocked for both families unless a dedicated audited no-shortlisting route is added.

### Generator changes

Use a new mechanism-ablation batch prefix, not the existing recovery/ordered-batch prefixes:

```text
paper_i_hh_weak_weak_mechanism_ablation_*
```

This avoids the current `preflight_submit.py` recovery logic that expects SNAKE Phase-III child subset cap `3`.

Add two source-anchor families to the weak-weak mechanism ablation generator/config:

1. `combinatorial_batch_cap3`
   - source anchor: existing completed weak-weak combinatorial batch cap-3 row.
   - invariant: Pauli child subset cap `1`.
   - invariant: batch target/cap `3/3`.
   - Full anchor is reference-only unless prior plan explicitly requires re-running it.

2. `physical_operator_lane`
   - source anchor:
     - `source_json = raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/weak_weak/json/result.json`
     - `source_sha256 = bb51341389bac493f99fac05bd425f6cdfca28a1d87983aa812d979b6301d1cb`
     - `commands_json = raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/commands.json`
     - `commands_json_sha256 = aaaef244b7f2a7dbe71bbc2a2062ab2b1855bf5e684d0c2cf768815bfb5d6238`
   - extract exact weak-weak argv from `commands.json` into `source_command_args_json`.
   - Full anchor is reference-only.
   - No-batching variant is omitted / marked reference-duplicate, not queued.

For the prior variant list:
- Generate each active ablation variant for both families.
- Skip physical `no_batching`.
- Block `no_shortlisting` for both families (`runnable=false`, not in queue).
- For no-cost, remove any prior override that set `--adapt-beam-lambda 0`; assert effective beam lambda remains `0.005`.
- For phase2-only, set:
  - `--adapt-continuation-mode phase2_v1`
  - `--phase3-backend-cost-mode proxy`

### Preflight changes

In `chtc/phase3_optuna/preflight_submit.py`, add a dedicated mechanism-ablation detector/blocker set for the new prefix.

Required checks:
- `display_regime == weak-weak`
- `method_key == snake`
- `optimizer == POWELL`
- `adapt_optimizer_kind == powell`
- `budget == 200`
- `max_depth == 30`
- `pool_contract == full_meta_unfiltered`
- `adapt_pool_class_filter_json in {"", "off"}`
- `snake_phase3_runtime_split_mode == shortlist_pauli_children_v1`
- `snake_phase3_runtime_split_selection_mode == archival_child_set_forward_v1`
- `snake_phase3_runtime_split_child_set_symmetry_policy == hard_guard`
- `snake_phase3_runtime_split_max_subset_size == 1`

Family-specific checks:
- `combinatorial_batch_cap3`: batch rows must use combinatorial cap-3 source semantics unless the variant explicitly disables batching.
- `physical_operator_lane`: source command hash must match; effective command must preserve physical lane/aggressiveness except explicitly allowed variants.
- physical `no_batching` must not appear in queued record IDs.
- `no_shortlisting` must fail if queued.
- `phase2_v1` / `phase1_v1` rows must have `--phase3-backend-cost-mode proxy`.
- no-cost rows must have denominator cost weights zeroed and `--adapt-beam-lambda 0.005`.

### Report changes

Report output should group rows by `source_anchor_family`, not merge them:

```text
Family 1: combinatorial_batch_cap3
Family 2: physical_operator_lane
```

For each family, show:
- source anchor result path/hash;
- command source path/hash;
- settings reused;
- settings changed;
- queued vs reference-only rows;
- blocked rows, especially `no_shortlisting`.

Physical family report rule:
- one row labeled `Full / no-batching existing anchor`;
- do not display a separate no-batching result as missing.

No-cost report rule:
- explicitly state beam lambda remained `0.005`;
- report only denominator cost weights as zeroed.

## Blockers

1. The exact completed `combinatorial_batch_cap3` source result path, SHA, and command argv/source-command SHA must be recorded in the generator config before preflight can pass.
2. `commands.json` extraction for the physical row must be deterministic; fail closed if the weak-weak argv cannot be found or hash-verified.
3. Do not reuse the existing `paper_i_hh_recovery_candidate_*` prefix; its preflight expects Phase-III subset cap `3`, which conflicts with the new cap-`1` interpretation.
4. No CHTC submit until generated queue excludes physical no-batching and all no-shortlisting rows, and the new preflight bundle passes.

Files to edit:
- `chtc/phase3_optuna/generate_paper_i_hh_weak_weak_mechanism_ablation_records.py`
- `chtc/phase3_optuna/preflight_submit.py`
- `pipelines/reporting/build_paper_i_hh_weak_weak_mechanism_ablation_report.py`
- `test/chtc/test_paper_i_hh_weak_weak_mechanism_ablation_records.py`
- Optional: `chtc/phase3_optuna/upload_submit_chtc.sh` for a narrowly named submit action.