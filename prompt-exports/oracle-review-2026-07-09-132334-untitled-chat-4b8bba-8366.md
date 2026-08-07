# Oracle Review

## Summary

The proposed rerun is **allowed as an explicit diagnostic/ablation rerun**, not as the canonical no-batching Paper-I HH route. Do **not** patch finite-amplitude HVA scoring without explicit user approval; that would be an algorithmic scoring change, not a launch-flag change.

## Findings

### P0 — Launch blockers

- **`pipelines/static_adapt/adapt_pipeline.py` / `pipelines/static_adapt/cli_config.py` — incompatible inherited flags can make the rerun fail before launch**
  - `--adapt-child-pool-expansion-mode global_pauli_child_sets_v1` cannot be combined with any retained non-off `--phase3-runtime-split-mode`.
  - `--phase3-enable-batching` will conflict if inherited command still has `--phase2-no-batching` or mismatched legacy `--phase2-batch-*` values.
  - **Suggestion:** explicitly remove old no-batching/runtime-split flags, or set:
    ```bash
    --phase3-runtime-split-mode off
    --phase3-enable-batching
    --phase3-batch-selection-mode combinatorial_reduced_plane
    --phase3-batch-target-size 3
    --phase3-batch-size-cap 3
    ```
    and ensure no conflicting `--phase2-*` aliases remain.

- **`pipelines/static_adapt/adapt_pipeline.py` — protected Hubbard QEB/HVA pools require physical lanes**
  - For `--adapt-pool uccsd_qeb_hva_blocks`, the new guard requires `static_lane_route='physical_operator_type'`.
  - `--physical-lane-shortlist-aggressiveness 2` alone is insufficient/inert unless the physical lane route is active.
  - **Suggestion:** include:
    ```bash
    --static-lane-route physical_operator_type
    --physical-lane-shortlist-aggressiveness 2
    ```
    and verify the post-expansion physical-lane audit still has `other_count=0`.

### P1 — Should fix / approval required

- **`pipelines/static_adapt/output_artifacts.py` — provenance may misreport batching when only phase3 aliases are used**
  - Runtime resolves `--phase3-enable-batching`, but top-level `settings.phase2_enable_batching` is serialized from raw `args.phase2_enable_batching`, which may remain `None → False`.
  - **Suggestion:** either fix serialization to record effective alias-resolved batching values, or launch with matching legacy aliases too for provenance consistency.

- **Finite-amplitude HVA scoring changes require explicit approval**
  - The diagnostic shows HVA has finite drop from the bare reference but **no finite drop from the final plateau state**, so changing selector/scoring code is not justified as a silent launch repair.
  - **Suggestion:** rerun with flags first. Any finite-amplitude HVA scoring patch should be a separate, user-approved opt-in algorithmic change with tests and provenance labels.