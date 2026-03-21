# HH Measurement Reduction Implementation Guide

Scope:
- HH only
- Optimize the upstream noiseless staged algorithm first
- Target staged ADAPT `phase3_v1`
- Keep windowed refit on
- Keep phase-2 batching on
- Treat widened shortlist plus batching as the repo's practical "beam" mode
- Preserve the current HH staged pool curriculum

## Goal

Improve the HH Pareto surface for:

`|Delta E_abs|` versus
- measurement burden
- depth / CX proxy burden
- optimizer burden

before touching downstream noisy-wrapper overhead.

## Key premise

The noisy layer is downstream of the main HH algorithm.
If the staged phase-3 core selects a bad scaffold, noisy batching only makes a
bad scaffold cheaper to read out.

So the first implementation slice should change how phase-3 HH scores and
selects candidates in the noiseless core.

## Current core architecture summary

Primary files:
- `pipelines/hardcoded/hh_continuation_scoring.py`
- `pipelines/hardcoded/adapt_pipeline.py`

Relevant current mechanisms:
- `Phase1CompileCostOracle`
- `MeasurementCacheAudit`
- `build_candidate_features(...)`
- `compatibility_penalty(...)`
- `greedy_batch_select(...)`
- phase-3 runtime split: `shortlist_pauli_children_v1`
- windowed reopt helpers in `adapt_pipeline.py`

### 1. Current measurement proxy is too weak

Current behavior:
- `MeasurementCacheAudit` tracks reused versus new group keys.
- In staged ADAPT, those keys are currently passed as `[candidate_label]`.

Implication:
- two distinct candidate labels with identical measurement structure do not
  reuse
- phase-3 measurement scoring is not aligned with the true downstream burden

### 2. Current depth proxy is too weak

Current behavior:
- `Phase1CompileCostOracle` uses a simple proxy:
  `new_pauli_actions + new_rotation_steps + position_shift_span + refit_active`

Implication:
- term count matters
- Pauli weight is mostly invisible
- a heavy high-weight child can look too similar to a light child

That is a poor fit if runtime split is on and you want the scorer to favor the
best `|Delta E| / (depth, measurement)` tradeoff.

### 3. Current batch compatibility is not measurement-aware

Current behavior:
- `compatibility_penalty(...)` uses:
  - support overlap
  - noncommutation
  - cross-curvature
  - schedule overlap

Implication:
- phase-2 batching can prefer numerically compatible candidates
- but it does not explicitly reward shared measurement basis structure

## Recommended implementation order

### Phase A: Make staged ADAPT measurement proxy group-aware

Priority: highest
Risk: medium
Expected win: better upstream measurement-aware scaffold selection

#### Change

Replace label-based measurement keys with derived measurement signatures from
the candidate polynomial.

Start with deterministic, cheap signatures:

1. Exact Pauli-label set signature
2. Diagonal versus non-diagonal split
3. QWC-style basis signature per Pauli term

Recommended minimum viable change:
- add `measurement_group_keys_for_term(...)`
- feed those keys into `MeasurementCacheAudit.estimate(...)`
- commit those keys for selected candidates and selected batches

#### Files to edit

- `pipelines/hardcoded/hh_continuation_scoring.py`
- `pipelines/hardcoded/adapt_pipeline.py`

Functions/classes to alter:
- `MeasurementCacheAudit`
- add helper such as `measurement_group_keys_for_term(...)`
- phase-1 candidate build path
- phase-2 candidate rebuild path
- selected-batch commit path

#### Acceptance

- different labels with the same basis signature show reuse
- existing staged ADAPT JSON still reports measurement cache stats
- score behavior is unchanged when measurement weights are zero

### Phase B: Replace the crude compile proxy with a gate-aware depth proxy

Priority: highest
Risk: medium
Expected win: better depth-aware upstream selection

#### Change

Upgrade `Phase1CompileCostOracle` so it reflects something closer to the
existing benchmark gate proxies.

Recommended direction:
- compute per-term Pauli weight
- compute per-term X/Y count
- derive a CX-like proxy and single-qubit proxy
- aggregate over the candidate polynomial

Do not remove the old components immediately.
Instead:
- keep the old fields for compatibility
- add richer burden components
- let `depth_cost` and related score inputs read from the richer proxy

Suggested alignment:
- use the same spirit as `_cx_proxy_term(...)` in the benchmark layer
- avoid introducing a second unrelated notion of gate burden

#### Files to edit

- `pipelines/hardcoded/hh_continuation_scoring.py`

Functions/classes to alter:
- `Phase1CompileCostOracle`
- `CompileCostEstimate`
- `build_candidate_features(...)`

#### Acceptance

- high-weight candidates score as more expensive than low-weight candidates
- deterministic behavior is preserved
- existing tests still pass or are updated with tighter expectations

### Phase C: Add measurement-basis affinity to phase-2 batching

Priority: high
Risk: medium
Expected win: better batched shortlist behavior under your "beam" setting

#### Change

Extend `compatibility_penalty(...)` so it can reward candidates that share
measurement structure, not just numerical/geometric compatibility.

Suggested new component:
- measurement-basis overlap or group-sharing score

Use it in:
- `CompatibilityPenaltyOracle`
- `greedy_batch_select(...)`

This matters because with batching on, you do not just care whether two
candidates can coexist in the ansatz. You also care whether they create shared
measurement burden or shared measurement savings.

#### Files to edit

- `pipelines/hardcoded/hh_continuation_scoring.py`
- `pipelines/hardcoded/adapt_pipeline.py`

Functions/classes to alter:
- `compatibility_penalty(...)`
- `CompatibilityPenaltyOracle`
- batch-selection call path in `adapt_pipeline.py`

#### Acceptance

- batching decisions become sensitive to measurement affinity
- existing non-measurement compatibility behavior is not broken

### Phase D: Keep runtime split on, but make it actually useful

Priority: high
Risk: low to medium
Expected win: shallower selected children when macro generators are too costly

#### Change

Continue to use:
- `--phase3-runtime-split-mode shortlist_pauli_children_v1`

But make the scorer above it good enough that:
- split children are preferred when they dominate the parent on the
  `|Delta E| / (cost, measurements)` frontier
- heavy macro generators are no longer protected by a weak proxy

This phase is mostly the payoff of Phases A-C, not a separate feature build.

#### Files to edit

- mainly the same files as Phases A-C

#### Acceptance

- runtime-split children win only when they are actually cheaper on the new
  proxy surface
- parent/child provenance remains intact

### Phase E: Add benchmark metrics that match the new core proxy

Priority: medium
Risk: low
Expected win: honest Pareto comparisons after the core changes land

#### Change

Keep the old benchmark fields, but add grouped / measurement-aware metrics that
match the new phase-3 proxy logic.

Examples:
- grouped measurement basis count
- grouped measurement burden proxy
- gate-aware continuation burden proxy

This is important because otherwise the core scorer and the report card will
optimize different things.

#### Files to edit

- `pipelines/exact_bench/hh_noise_robustness_seq_report.py`
- related benchmark tests

#### Acceptance

- backward-compatible output
- new metrics are deterministic
- grouped metrics are no worse than legacy term-count proxies when grouping is
  actually available

### Phase F: Only then optimize the noisy/oracle wrapper

Priority: follow-on
Risk: low to medium
Expected win: lower downstream overhead after the core scaffold is better

This is where to add:
- `ExpectationOracle.evaluate_many(...)`
- shared diagonal counts
- grouped sampler fallback

Primary files:
- `pipelines/exact_bench/noise_oracle_runtime.py`
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py`
- `pipelines/exact_bench/hh_noise_hardware_validation.py`

This is still worth doing.
It is just not the first move.

## Recommended baseline mode for this work

For investigation and implementation, use the staged HH path:
- `--adapt-continuation-mode phase3_v1`
- `--adapt-reopt-policy windowed`
- `--phase2-enable-batching`
- `--phase3-runtime-split-mode shortlist_pauli_children_v1`
- keep the default HH staged pool resolution

Important:
- do not turn this into a new depth-0 `full_meta` default
- keep the current HH staged curriculum:
  narrow core first, residual widening later

## Tests to add or extend first

### Core scoring tests

File:
- `test/test_hh_continuation_scoring.py`

Add coverage for:
- measurement-group key derivation
- reuse across different labels with the same basis signature
- gate-aware compile/depth ranking
- batching compatibility sensitivity to measurement overlap

### Integration tests

File:
- `test/test_adapt_vqe_integration.py`

Add or extend coverage for:
- staged HH `phase3_v1` selection metadata
- runtime split with the new proxy fields
- measurement cache summary consistency

### Benchmark follow-on tests

Files:
- `test/test_hh_noise_robustness_benchmarks.py`
- `test/test_hh_noise_oracle_runtime.py`

These are follow-on after the core slice lands.

## Suggested implementation sequence for an agent

1. Add measurement-group key derivation to staged ADAPT.
2. Upgrade the compile/depth proxy to be gate-aware.
3. Add measurement affinity into phase-2 compatibility / batching.
4. Re-run staged HH tests and inspect selection metadata.
5. Add grouped benchmark metrics.
6. Then optimize noisy/oracle batching.

## Success criteria

Primary:
- staged HH phase-3 selections become more measurement-aware
- staged HH phase-3 selections become more depth-aware
- runtime split chooses cheaper children more often when justified
- no regression in energy accuracy contracts

Secondary:
- downstream noisy/oracle overhead drops once wrapper batching is added

Non-goals for the first slice:
- changing operator algebra core files
- changing JW conventions
- changing the compiled ADAPT gradient contract
- changing the staged HH pool curriculum

## Files most likely to change in the first implementation slice

- `pipelines/hardcoded/hh_continuation_scoring.py`
- `pipelines/hardcoded/adapt_pipeline.py`
- `test/test_hh_continuation_scoring.py`
- `test/test_adapt_vqe_integration.py`

Follow-on files:

- `pipelines/exact_bench/noise_oracle_runtime.py`
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py`
- `pipelines/exact_bench/hh_noise_hardware_validation.py`
- `test/test_hh_noise_oracle_runtime.py`
- `test/test_hh_noise_robustness_benchmarks.py`
