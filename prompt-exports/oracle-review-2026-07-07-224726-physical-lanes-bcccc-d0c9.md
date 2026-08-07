# Oracle Review

## Summary

The plan is directionally safe: treating `physical_operator_type` as an opt-in Route-A lane-route variant is better than minting a new route id, as long as canonical Route A semantics/tests remain unchanged and variant telemetry is explicit. The main blockers are cache-key isolation, child-label inheritance, deterministic shortlist caps, and a hard source-lock gate before using the named promoted-copy/PDF baseline.

## P1 Findings

1. **`pipelines/static_adapt/route_identity.py` — Variant may be mistaken for canonical Route A**
   - Keeping `static_route_id=route_a` is safer than a new route id, but only if physical lanes are clearly marked non-canonical.
   - **Adjust:** add explicit `route_variant_id="route_a_physical_operator_lanes_v1"` and `static_lane_route_is_route_identity=False`; ensure paper-facing labels/manifests cannot describe this as plain canonical Route A.

2. **`pipelines/static_adapt/algebraic_metadata.py` — Generalization must not alter algebraic lane semantics**
   - The helper refactor is acceptable only if existing `flat/curv/disj/mix` behavior, lane order, fallback-to-`mix`, budget allocation, and telemetry keys remain byte-for-byte compatible.
   - **Adjust:** keep algebraic wrappers as compatibility shells using `LANES_PHASE1` and `LANE_MIX`; add regression tests that old algebraic tests pass without expected-output changes.

3. **`pipelines/static_adapt/adapt_candidate_record_cache.py` / `adapt_pipeline.py` — Cache bump alone is not enough**
   - Bumping cache version invalidates stale entries once, but does not prevent cross-route reuse between algebraic and physical runs under the new version.
   - **Adjust:** include `static_lane_route`, lane classifier version, lane key, physical source label/parent label, and effective shortlist settings in cache identity or cached selector payload. If missing physical metadata is detected on a physical run, fail or recompute rather than fallback silently.

4. **`pipelines/static_adapt/adapt_pipeline.py` — Child-set labels will collapse to `other` unless parent metadata is propagated**
   - Label probing after runtime split is fragile; Pauli-child labels often lack HH motif information.
   - **Adjust:** propagate physical lane payload from the parent candidate at child construction/split time using parent pool index/label metadata. Add a gate/telemetry warning if `other` exceeds an expected threshold for full-meta HH pools.

5. **`pipelines/static_adapt/adapt_pipeline.py` / `cli_config.py` — 3x aggressiveness should be reproducible, not implicit**
   - Dividing caps/fractions is reasonable as the default derivation: `ceil(cap / factor)` and `fraction / factor`. But paper-facing runs need explicit effective values recorded.
   - **Adjust:** derive defaults from resolved source settings, then record both base and effective caps/fractions. Consider explicit override flags for effective physical caps, with validation that overrides either match the factor-derived values or are clearly marked as intentional deviations.

6. **Run gate — Named promoted-copy/PDF baseline must be source-locked before use**
   - The plan mentions resolver use, but this should be a hard pre-run gate, not guidance.
   - **Adjust:** require successful `resolve_visible_settings.py` output before any paper-facing physical-lane run. Validate method/source fields: POWELL, no batching, `full_meta_unfiltered_hva_included`, Pauli-child split, children-per-parent 2, beam lambda `0.005`, live branches 3, subset size 1, metric prune enabled, and exact HH regime parameters. Fail closed on any mismatch.