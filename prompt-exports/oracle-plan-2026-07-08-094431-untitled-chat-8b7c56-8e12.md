# Oracle Plan

## Row-to-CLI-override map

Assume every new row starts from the **existing weak-weak combinatorial batch cap-3 completed anchor command**, not from the recovery-candidate generator defaults.

```yaml
full_anchor_existing:
  submit: false
  source: existing weak-weak combinatorial cap-3 row
  overrides: {}

no_batching_existing:
  submit: false
  source: existing no-batch row fills table "No batching"
  overrides: {}

greedy_cap3_existing:
  submit: false
  source: existing greedy cap-3 row is supplemental
  overrides: {}

no_prune:
  submit: true
  enable_flags:
    - --phase1-no-prune
  remove_bool_flags:
    - --phase1-prune-enabled

no_cost_term:
  submit: true
  set_flags:
    # Phase-I score burden
    --phase1-compile-cx-proxy-weight: "0"
    --phase1-compile-sq-proxy-weight: "0"
    --phase1-compile-rotation-step-weight: "0"
    --phase1-compile-position-shift-weight: "0"
    --phase1-compile-refit-active-weight: "0"
    --phase1-measure-groups-weight: "0"
    --phase1-measure-shots-weight: "0"
    --phase1-measure-reuse-weight: "0"
    --phase1-opt-dim-cost-scale: "0"
    --phase1-family-repeat-cost-scale: "0"

    # Phase-II / FullScoreConfig score burden; also feeds Phase-III full-score burden.
    --phase2-compile-cx-proxy-weight: "0"
    --phase2-compile-sq-proxy-weight: "0"
    --phase2-compile-rotation-step-weight: "0"
    --phase2-compile-position-shift-weight: "0"
    --phase2-compile-refit-active-weight: "0"
    --phase2-measure-groups-weight: "0"
    --phase2-measure-shots-weight: "0"
    --phase2-measure-reuse-weight: "0"
    --phase2-opt-dim-cost-scale: "0"
    --phase2-family-repeat-cost-scale: "0"

    # Phase-III backend-aware selector cost.
    --phase3-backend-w-2q: "0"
    --phase3-backend-w-depth: "0"
    --phase3-backend-w-size: "0"

    # Required if the row is truly "no cost term", not merely "no score-denominator cost".
    --adapt-beam-lambda: "0"

no_novelty:
  submit: true
  set_flags:
    --phase3-novelty-ablation-mode: "all"

phase2_novelty_only:
  submit: true
  set_flags:
    --adapt-continuation-mode: "phase2_v1"
    --phase2-selector-gain-mode: "unit_gain_v1"
    --phase3-novelty-ablation-mode: "off"

phase2_second_order_only:
  submit: true
  set_flags:
    --adapt-continuation-mode: "phase2_v1"
    --phase3-novelty-ablation-mode: "phase2"
  note: keep normal Phase-II selector gain; do not set unit_gain_v1.

no_phase3:
  submit: true
  set_flags:
    --adapt-continuation-mode: "phase2_v1"
  note: full Phase-II scoring is preserved; do not change novelty or selector-gain mode.

phase1_only_macro:
  submit: true
  set_flags:
    --adapt-continuation-mode: "phase1_v1"
    --phase3-runtime-split-mode: "off"
  remove_bool_flags:
    - --allow-archival-phase3-runtime-split
  remove_value_flags:
    - --phase3-runtime-split-selection-mode
    - --phase3-runtime-split-child-set-symmetry-policy
    - --phase3-runtime-split-max-subset-size
    - --phase3-source-lock-preferred-sequence

phase1_only_singleton:
  submit: conditional
  set_flags:
    --adapt-continuation-mode: "phase1_v1"
    --shared-pauli-pool-mode: "shared_pauli_child_sets_v1"
    --shared-pauli-pool-symmetry-policy: "hard_guard"
    --shared-pauli-pool-max-subset-size: "1"
    --phase3-runtime-split-mode: "off"
  remove_bool_flags:
    - --allow-archival-phase3-runtime-split
  blocker_if: shared-pauli-pool CLI/runner flags are not accepted by deployed source-lock runner.

full_geometry_window:
  submit: conditional
  set_flags:
    --phase3-geometry-window-size: "0"
  blocker_if: anchor already has phase3_geometry_window_size=0 or otherwise already uses legacy/full coupled geometry.
  note: if intended change is selector geometry mode rather than window size, block unless a validated --phase3-selector-geometry-mode full value exists.

no_shortlisting:
  submit: false
  status: blocked
```

## Blockers

1. **Existing generator is unsafe as-is.**  
   `generate_paper_i_hh_recovery_candidate_run_stock_records.py` hardcodes `SUBSET_SIZE = "3"` and emits `--phase3-runtime-split-max-subset-size 3`, while the completed batch-cap-3 anchor is `phase3_archival_subset1`. That changes the Pauli-child subset cap, not the batch cap, so it violates the source lock.

2. **Existing generator also drifts other anchor settings.**  
   Its batch variant sets `--adapt-beam-children-per-parent 3`; the current audit anchor says `2`. Do not reuse it unmodified for this weak-weak ablation batch.

3. **`no_shortlisting` should block.**  
   There is no single safe CLI switch for “no shortlisting.” A real attempt would need opening Phase-1/2/3 caps and frontier gates together, at minimum `--phase1-shortlist-size`, `--phase1-probe-max-positions`, `--phase2-shortlist-size`, `--phase2-shortlist-fraction 1.0`, `--phase2-frontier-ratio 0`, and `--phase3-frontier-ratio 0`, plus any controller maturity caps. Without a source-audited finite cap bound, this is not source-locked.

4. **`full_geometry_window` is conditional.**  
   Submit only if the anchor command/result proves a non-full positive `phase3_geometry_window_size`. If the anchor already has `0`/legacy/full coupled geometry, this row is duplicate drift and should not run.

5. **`phase1_only_singleton` is conditional/fairness-labeled.**  
   It is expressible only if the shared Pauli-child pool flags are supported by the current CLI/source-lock runner. It is not the same mechanism as Phase-III archival runtime split; label it `phase1_only_singleton_child_pool`, not a pure Phase-III ablation.