# Oracle Review

## Summary

The proposed approach is scientifically reasonable and narrow: add a distinct Hubbard-only `uccsd_qeb_hva_blocks` pool, build three grouped plain-Hubbard HVA block polynomials, classify them into a new physical lane, and require a zero-`other` prelaunch audit before running Hubbard weak to depth 10. I would not launch until the source-locked baseline is explicitly resolved and the new pool’s final emitted labels pass the physical-lane audit.

## Findings

### P0 — Must Fix Before Launch

1. **Run provenance / source-locked settings**
   - **Reference:** `run_plan`, `adapt_pipeline.py` result settings fields
   - **Issue:** The run must not be launched from the candidate command shape or memory of the prior `uccsd_qeb` result. The prior Hubbard weak depth-15 `uccsd_qeb` settings need to be resolved from the actual artifact/result JSON, then only `adapt_pool` and `adapt_max_depth=10` changed.
   - **Suggestion:** Fail closed unless the baseline artifact is found and the effective settings diff shows only:
     - `uccsd_qeb` → `uccsd_qeb_hva_blocks`
     - max depth → `10`
     - required route/lane metadata version changes from the new pool.

2. **Prelaunch gate must cover the new pool**
   - **Reference:** `pipelines/static_adapt/adapt_pipeline.py`
   - **Issue:** Current gate protects only `problem_key == "hubbard" and pool_key == "uccsd_qeb"`. If the new pool is added without extending this gate, a mislabeled HVA/block term could enter the run as `other`.
   - **Suggestion:** Protect `{ "uccsd_qeb", "uccsd_qeb_hva_blocks" }`; require physical route, `other_count == 0`, empty `exact_other_labels`, positive QEB count, and for the new pool positive `hva_hamiltonian_blocks` count.

### P1 — Should Fix

1. **Avoid settings drift from physical-lane cap scaling**
   - **Reference:** `pipelines/static_adapt/adapt_pipeline.py` around shortlist base/effective cap handling
   - **Issue:** The code distinguishes base shortlist caps from effective caps after `physical_lane_shortlist_aggressiveness`. For source-locking, it is easy to accidentally copy effective caps as CLI inputs and divide them again.
   - **Suggestion:** Compare both base and effective fields from the prior result. Pass the same CLI/base values as the source run, not manually divided values.

2. **Builder/classifier label contract must be exact**
   - **Reference:** `primitive_pools.py`, `static_provenance.py`
   - **Issue:** The plan is safe only if emitted labels and classifier labels match exactly. Any mismatch such as `hva_block::potential_layer` vs `ham_block::pot(...)` would fail or silently become `other` if the gate is incomplete.
   - **Suggestion:** Choose one emitted label set, preferably:
     - `hva_block::hop_layer`
     - `hva_block::onsite_layer`
     - `hva_block::potential_layer`
     
     Then test those exact labels through `summarize_static_physical_operator_pool_labels(..., problem="hubbard")`.

3. **Keep the new pool additive**
   - **Reference:** `problem_registry.py`, `pool_resolution.py`
   - **Issue:** Accidentally replacing `uccsd_qeb`, changing Hubbard default `uccsd`, or enabling generic `pool='hva'` for plain Hubbard would violate the requested narrow change.
   - **Suggestion:** Add only `uccsd_qeb_hva_blocks`; leave `uccsd`, `uccsd_qeb`, and generic Hubbard `hva` behavior unchanged.

4. **Ensure grouped HVA blocks stay grouped**
   - **Reference:** `primitive_pools.py`, `adapt_pipeline.py`
   - **Issue:** If `execution_mode="grouped_exact"` is missing or not respected, the HVA block may be split into Pauli-term candidates, changing both science and lane audit semantics.
   - **Suggestion:** Add a focused smoke test asserting each HVA block term has `execution_mode == "grouped_exact"` and survives as a single final pool label before audit.

### P2 — Consider

1. **Unrelated dirty-tree edits increase launch risk**
   - **Reference:** `agent_guidance/skills/paper-i-run/SKILL.md`, unrelated molecular parameterization additions in `adapt_pipeline.py`
   - **Issue:** These are not part of the Hubbard HVA lane objective and make review harder.
   - **Suggestion:** Do not add further unrelated edits in this patch. If possible, isolate the HVA patch from guidance/manuscript/run-doc changes.

2. **Family IDs may be too coarse for HVA blocks**
   - **Reference:** `pool_resolution.py`, planned `_hubbard_uccsd_qeb_hva_family_id_for_label`
   - **Issue:** Mapping all HVA blocks to one family, `hva_hamiltonian_blocks`, may make family-repeat penalties treat hop/onsite/potential as the same family.
   - **Suggestion:** This is not a launch blocker if preserving current coarse `uccsd_qeb` behavior is desired, but consider `hva_hamiltonian_blocks:hop`, `:onsite`, `:potential` if family-repeat diversity matters.

## Bottom Line

No conceptual blocker to the HVA-block lane design. The safer narrow path is: implement only the new key/builder/classifier/route/gate/tests, prove the final pool audit has zero `other`, resolve the prior Hubbard weak artifact, verify the settings diff, then launch depth 10.