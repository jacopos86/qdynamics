# Static ADAPT Route Alignment

Purpose: capture the agreed target architecture for refactoring
`pipelines/static_adapt/adapt_pipeline.py`.

This document is a refactor contract. It does not authorize behavior changes or
legacy deletion by itself.

## Agreed Route Stance

Route A is the canonical Paper-I SNAKE route.

Current code anchor:

- `pipelines/static_adapt/route_identity.py` says Route A is the current static
  SNAKE controller route.
- It defines Route A required components, including `full_meta`,
  `phase3_selector_policy=algebraic_nested_v1`, reduced Phase-III geometry,
  batching enabled, and prune policy `recoverability_ladder_v1`.
- It marks Route A as `canonical_snake_eligible=True`.

Route B is legacy pairwise behavior.

- Current code name: `route_b_legacy_pairwise`.
- It differs from Route A by changing Phase-II novelty and raw score formula to
  pairwise legacy variants.
- Target: quarantine as legacy route if retained.

Route C is plateau acquisition behavior.

- Current code name: `route_c`.
- It differs from Route A by enabling plateau acquisition:
  `phase3_plateau_acquisition_mode=novelty_cost_v1` and log-volume scoring.
- Target: quarantine as legacy/diagnostic route if retained.

## Route Dependency Rule

Route A owns canonical SNAKE definitions.

Allowed dependencies:

- Route A may depend on generic shared utilities:
  - problem registry
  - Hamiltonian construction
  - pool construction
  - Pauli/JW mapping
  - optimizer dispatcher
  - shared math/scoring primitives
  - telemetry primitives

- Route B and Route C may import or call Route-A-owned shared primitives.

Forbidden dependencies:

- Route A must not import from Route B.
- Route A must not import from Route C.
- Route A must not branch internally on Route-B or Route-C behavior.
- Route B/C must not mutate Route-A config, score definitions, or route-local
  defaults.

Target dependency direction:

```text
shared primitives -> route_a_snake
shared primitives + route_a_snake -> route_b_legacy_pairwise
shared primitives + route_a_snake -> route_c_plateau
```

Route B/C may reuse Route A, but Route A must remain complete without them.
Route B and Route C do not depend on each other unless a future explicitly
approved legacy reproduction requires it.

## Canonical Route-A Score

Current code anchor:

- Route identity contract records canonical score formula:
  `DeltaE_TR * N3 / (1 + K3)`.
- Primary selector key: `full_v2_score`.

Paper math anchor:

- The Paper-I math defines a shared phase-local cost \(K_k(r;t)\) using the same
  cost structure for \(k\in\{1,2,3\}\):

```text
K_k(r;t)
  = lambda_2q z_2q(r;t)
  + lambda_d z_d(r;t)
  + lambda_theta z_theta(r;t)
  + lambda_shot z_shot(r;t)
  + beta_wt z_wt(r;t)
  + beta_XY z_XY(r;t)
```

Agreed extension:

- The same cost object should also apply to Phase 0 when Phase 0 scores or
  filters candidate records.
- Phase 0 may have a reduced/partial record feature set, but it should still
  emit the same canonical cost-component shape where possible.

Route-A score contract:

```text
S_k(r;t) = Gain_k(r;t) * Novelty_k(r;t) / (1 + K_k(r;t))
```

where:

- \(r\) is a candidate-position record.
- \(k\) is the stage/phase index.
- \(K_k\) is the shared cost object.
- phase differences come from the input record set and available measured
  features, not from hidden score-equation changes.

Implementation note:

- Existing names such as `phase2_raw_score`, `full_v2_score`,
  `phase3_canonical_score_formula`, and `cheap_score_eps` may remain as
  compatibility fields during migration.
- The canonical route should expose one score primitive with aliases/adapters,
  rather than independent score equations per phase.

## Batching And Beam Score Aggregation

Batching belongs to Route A when used by canonical SNAKE.

Agreed rule:

- Batching uses the same record score/cost primitive.
- Batch-level burden is additive over selected batch records unless an explicitly
  documented joint geometry term replaces the corresponding component.
- Batch payloads must record which terms are additive and which terms are joint.

Beam belongs to Route A when used by canonical SNAKE.

Agreed rule:

- Beam survival uses the same Route-A cost primitive.
- Beam branch burden is additive along the branch lineage.
- Ordered-batch beam may use a branch cumulative cost, but the underlying unit
  cost must be the Route-A canonical cost object.

Forbidden:

- Beam must not introduce a different silent score equation.
- Batch must not introduce a different silent score equation.
- Legacy Route-B pairwise scoring must not leak into Route-A batch or beam.

## Problem And Pool Separation

Agreed direction:

- `adapt_pipeline.py` should not hardcode Hubbard, Hubbard-Holstein, molecular,
  or any other Hamiltonian-specific construction inside the algorithm runner.
- Problem construction should be resolved through a problem registry/spec layer.
- The SNAKE runner should receive:
  - Hamiltonian/operator representation
  - reference state/input state
  - exact/reference metadata when available
  - problem-local pool or pool key

Target modules:

- `problem_routes.py` or existing `builders/problem_registry.py`
- `pool_routes.py` or existing `builders/pool_resolution.py`
- Route A should consume resolved `ProblemSpec` and `PoolSpec`.

Comparator methods:

- Append-only ADAPT, Geo-ADAPT, TETRIS-ADAPT, and Qubit/QEB-ADAPT are
  comparator routes, not Route-A internals.
- They may share problem/pool/mapping machinery with Route A.
- They should live in separate route files or thin comparator entrypoints.
- They should not make the SNAKE runner larger.

## Noise And Oracle Separation

Agreed direction:

- Canonical Route A is noiseless unless a noise route is explicitly selected.
- Noise/oracle-gradient/final-noise-audit behavior should be separated from the
  no-noise Route-A hot path.
- Noise routes may call Route A as the clean baseline.
- Route A should not need to understand all Qiskit/noisy backend variants to run
  the canonical no-noise algorithm.

Allowed shared boundary:

- Route A can emit enough telemetry for later noise diagnostics.
- Noise routes can consume Route-A states/history/ansatz data.
- Noise routes cannot alter the canonical no-noise selection rule.

## Refactor Target Layout

Near-term target files:

- `route_a_snake.py`
  - canonical SNAKE route orchestration
  - no Route-B or Route-C branches

- `route_b_legacy_pairwise.py`
  - optional legacy pairwise behavior
  - imports Route-A primitives only as needed

- `route_c_plateau.py`
  - optional plateau acquisition behavior
  - imports Route-A primitives only as needed

- `score_routes.py`
  - canonical score/cost primitive
  - compatibility aliases
  - legacy score shims quarantined behind explicit route names

- `problem_routes.py`
  - problem registry/spec resolution

- `pool_routes.py`
  - pool registry/spec resolution

- `noise_routes.py`
  - oracle gradient, value noise, final noise audit, noisy backend variants

- `optimizer_routes.py`
  - inner optimizer dispatcher and optimizer-memory policies

- `checkpoint_telemetry.py`
  - checkpoint/resume/replay/output telemetry helpers

Existing `beam_search.py` remains the active beam extraction target.

## Route-A Must Stay Simple

Route A should read as:

```text
resolved_problem = resolve_problem(problem_spec)
resolved_pool = resolve_pool(pool_spec, resolved_problem)
score_spec = canonical_route_a_score_spec(...)
optimizer_spec = resolve_optimizer(...)
run_control = resolve_run_control(...)

state = initialize_route_a_state(...)
for k in range(max_adapt_iterations):
    records_0 = build_phase0_records(...)
    records_1 = score_phase(records_0, score_spec)
    records_2 = score_phase(shortlist(records_1), score_spec)
    records_3 = score_phase(shortlist(records_2), score_spec)
    admission = select_admission(records_3, batching=..., beam=...)
    state = apply_admission_and_refit(state, admission, optimizer_spec)
    state = maybe_prune_route_a(state, score_spec)
    checkpoint_telemetry(...)
```

This is schematic, not a required exact API.

The important point: route identity, problem identity, score identity,
optimizer identity, noise identity, and telemetry identity must be visible
instead of being hidden in one local function.

## Legacy Quarantine Rule

Any feature with `legacy`, Route B, Route C, shadow legacy geometry, legacy
pairwise novelty, legacy pairwise score, or route-C plateau naming must be
classified as one of:

- `route_b_legacy`
- `route_c_legacy_or_diagnostic`
- `compatibility_payload_only`
- `delete_after_user_approval`
- `unknown_needs_evidence`

Default classification:

- Route B: `route_b_legacy`
- Route C: `route_c_legacy_or_diagnostic`
- Shadow legacy geometry: `compatibility_payload_only` unless evidence shows it
  changes selected Route-A behavior
- Legacy pairwise scoring: `route_b_legacy`

## Open Work Items

1. Create a feature inventory for `_run_hardcoded_adapt_vqe`.
   - Classify each parameter/block as Route A, shared primitive, Route B,
     Route C, comparator, noise route, telemetry, problem route, pool route, or
     unknown.

2. Create a score inventory.
   - Map code fields to the canonical score primitive:
     `DeltaE_TR`, novelty, `K`, denominator, additive batch/beam burden.
   - Mark legacy pairwise formulas for quarantine.

3. Create a problem/pool inventory.
   - Identify all hardcoded problem-specific logic still inside
     `adapt_pipeline.py`.
   - Map it to existing builder/registry files where possible.

4. Continue behavior-preserving beam extraction.
   - Keep updating `beam_refactor_migration.md`.
   - Do not delete or quarantine legacy behavior in beam extraction commits.

5. After the inventories, implement Route-A extraction first.
   - Route B/C extraction comes after Route A is stable.
   - B/C must depend on A or shared primitives, not the other way around.
