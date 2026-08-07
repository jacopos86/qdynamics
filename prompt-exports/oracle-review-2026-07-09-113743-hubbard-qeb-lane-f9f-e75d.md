# Oracle Review

## Summary

The proposed direction is sound: add a Hubbard-only `uccsd_qeb` pool by lifting the existing QEB/qubit-excitation construction into static pool builders, combine it with Hubbard UCCSD, structurally deduplicate, add a first-class Hubbard `qeb_excitation` physical lane, and block launch unless a final emitted-pool audit has `other_count == 0`. However, the current diff/context does **not yet implement the QEB path**, and it includes unrelated changes that should be split out before this provenance-sensitive rerun.

## P0 — Must Fix / Launch Blockers

- **`pipelines/static_adapt/builders/problem_registry.py`, `pool_resolution.py`, `primitive_pools.py` — no `uccsd_qeb` path exists yet**
  - The CLI cannot expose or resolve the intended combined Hubbard UCCSD+QEB pool.
  - **Suggestion:** Add a Hubbard-only key, e.g. `uccsd_qeb`, to `_HUBBARD_POOL_KEYS`; implement `_build_hubbard_uccsd_qeb_pool()`; add a `resolve_pool_plan()` branch with distinct `method_name`, `pool_stage_family`, and `pool_family_ids`.

- **`pipelines/contracts/static_provenance.py` — QEB labels still classify as `other`**
  - Current Hubbard classifier only recognizes `uccsd_sing(...)` and `uccsd_dbl(...)`. `qeb_pair(...)` and `qeb_double(...)` would fail the required lane audit.
  - **Suggestion:** Add `HUBBARD_PHYSICAL_OPERATOR_LANE_QEB_EXCITATION = "qeb_excitation"`, bump the Hubbard classifier version, add the lane to `HUBBARD_PHYSICAL_OPERATOR_LANES`, and classify only anchored canonical labels:
    - `qeb_pair(\d+,\d+)`
    - `qeb_double(\d+,\d+->\d+,\d+)`

- **`pipelines/static_adapt/adapt_pipeline.py` — no final pool pre-launch audit/block yet**
  - Existing `other_count` is candidate-evaluation telemetry, not a pre-launch proof over final emitted pool labels.
  - **Suggestion:** Add a separate `prelaunch_pool_audit` after all pool mutations/expansions and before scoring. For `problem=hubbard, adapt_pool=uccsd_qeb`, raise before launch unless:
    - `static_lane_route == physical_operator_type`
    - `other_count == 0`
    - `exact_other_labels == []`
    - `lane_counts["qeb_excitation"] > 0`

- **Launch gate — do not launch from inferred settings**
  - The 1.75 weak source root must be inspected directly and source-locked before any run.
  - **Suggestion:** Stop after patch/tests/audit until Oracle/user approval. Then reuse exact source settings from `raw_outputs/paper_i_alt_hamiltonian_physical_operator_lanes_1p75_fullreopt_agent_repair1_20260709`, changing only the pool/lane addition.

## P1 — Should Fix

- **Avoid importing `exact_bench` into `static_adapt`**
  - Importing `pipelines.exact_bench.generic_static_adapt_variants` from static builders would be unsafe and likely circular, because `generic_static_adapt_variants` already imports static pool resolution.
  - **Suggestion:** Extract only the pure QEB construction into `primitive_pools.py` or a new static builder module, then make `generic_static_adapt_variants.build_pairwise_qubit_excitation_pool()` delegate to it.

- **Dedup policy must prove QEB survives**
  - UCCSD-first dedup is sensible for label stability, but it may drop QEB terms that duplicate UCCSD. The audit must not silently allow a “combined” pool with zero QEB-lane terms.
  - **Suggestion:** First occurrence wins; UCCSD labels win duplicates. Add an explicit post-dedup guard/test that at least one final label classifies as `qeb_excitation`.

- **Route version must reflect new Hubbard semantics**
  - `lane_routes.py` still has Hubbard route id `v1_uccsd_split`.
  - **Suggestion:** update to something like `route_a_hubbard_physical_operator_lanes_v2_uccsd_qeb_split`.

- **Scope creep in current diff**
  - The diff includes unrelated `generic_static_adapt_variants.py` fidelity/runtime-seed changes and Paper-I skill-doc updates. These are not part of the QEB lane task.
  - **Suggestion:** split or revert unrelated changes for this patch so the review/run provenance is focused.

- **`first_eps_energy_termination_condition` prefix is risky for plateau cost**
  - It is appropriate only if it means “first prefix satisfying the same source-locked absolute target threshold” and does not cut through a batch. It is not appropriate if it refers merely to optimizer `eps_energy`/small energy-change termination.
  - **Suggestion:** For later plateau cost, use a strict replayable selected-prefix tied to `abs_delta_e < 1e-5`, with batch-boundary validation and source JSON hash recorded.

## P2 — Consider / Test Gaps

- Add parser test for `--problem hubbard --adapt-pool uccsd_qeb`.
- Add resolve-pool test proving final `uccsd_qeb` labels have no duplicate `_polynomial_signature()` values.
- Add classifier tests for:
  - `qeb_pair(0,1)` → `qeb_excitation`
  - `qeb_double(0,3->1,2)` → `qeb_excitation`
  - `qeb_pair_alt` → `other`
- Add lane-route test that Hubbard lanes include `qeb_excitation`.
- Add negative test that non-Hubbard `uccsd_qeb` resolution raises.
- Add audit test over intended Hubbard weak settings: `other_count == 0`, no exact other labels, and positive QEB count.