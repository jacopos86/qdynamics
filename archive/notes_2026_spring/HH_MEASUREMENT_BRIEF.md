# HH Measurement-First Brief

Scope:
- HH only
- Optimize the main noiseless staged algorithm first
- Evaluate under staged ADAPT `phase3_v1`
- Assume "beam" means widened shortlist plus phase-2 batching
- Assume windowed refit is on
- Objective: improve the `|Delta E| / (cost, measurements)` Pareto surface

## Short answer

Yes. The noisy layer is downstream of the noiseless staged HH algorithm.
If you want the algorithm's real measurement and depth footprint to improve,
the first work should be in the main phase-3 ADAPT path, not in the noisy
wrapper.

The noisy/oracle code still matters, but mostly as a second layer:
- it inherits the chosen ansatz and generator scaffold
- it inherits the propagated operator structure
- it only adds extra measurement protocol overhead on top

## Main finding

The current HH phase-3 core already has the right control points, but its
cost proxies are still too weak for a true measurement-first Pareto search.

The main upstream gaps are:

1. ADAPT measurement cost is label-based, not basis/group-based.
   `MeasurementCacheAudit` is fed `[candidate_label]`, so reuse is tracked at
   the label level instead of the Pauli-basis level.

2. The depth/compile proxy is too crude.
   `Phase1CompileCostOracle` mainly counts new Pauli actions / rotations and
   position shifts. It does not reflect Pauli weight or a gate-like CX burden.

3. Phase-2 batching is not measurement-aware enough.
   `compatibility_penalty(...)` models overlap, commutation, curvature, and
   schedule coupling, but not shared measurement basis structure.

4. An archival/internal runtime-split implementation still exists, but the
   canonical manuscript/public path keeps runtime split off.
   If that archival path is revisited for internal testing, selection still
   is not yet driven by a strong grouped-measurement plus gate-aware proxy.

## Recommended order

1. Make staged ADAPT measurement proxy group-aware.
   Highest-value upstream change.

2. Replace the crude compile/depth proxy with a gate-weighted proxy.
   This is the main depth-first improvement in the noiseless core.

3. Add measurement-basis affinity into phase-2 batch selection.
   Important when "beam" plus batching is on.

4. Keep the canonical manuscript/public path unchanged (`phase3_v1` with runtime split off).
   If the archival runtime-split path is revisited internally, let improved scoring prefer cheaper children there.

5. Only after that, optimize noisy/oracle batching and shared counts.

## Why this is the right first slice

It targets the place where the HH scaffold is actually chosen:
- which generators enter the ansatz
- how expensive they are to realize
- how much measurement burden they imply downstream

That moves the real frontier.
Oracle batching by itself mostly reduces wrapper overhead after the core
structure has already been chosen.

## Files most implicated

- `pipelines/hardcoded/hh_continuation_scoring.py`
- `pipelines/hardcoded/adapt_pipeline.py`
- `test/test_hh_continuation_scoring.py`

Secondary follow-on files:

- `pipelines/exact_bench/noise_oracle_runtime.py`
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py`
- `test/test_hh_noise_oracle_runtime.py`
- `test/test_hh_noise_robustness_benchmarks.py`

## Short recommendation

Optimize the noiseless HH phase-3 selection surface first:
- measurement-group-aware reuse
- gate-aware depth proxy
- batch compatibility that rewards shared measurement structure

Then update the noisy/oracle layer to exploit the cheaper scaffold more
efficiently.
