# Static ADAPT Pipeline Refactor Plan

Purpose: make `pipelines/static_adapt/adapt_pipeline.py` understandable and
refactorable without changing Paper-I behavior by accident.

This is a planning/source-of-truth document, not a deletion authorization. A
feature listed as "suspected archaic" still needs an explicit user decision and
tests before removal.

## Current State

Primary target:

- `pipelines/static_adapt/adapt_pipeline.py`
- `_run_hardcoded_adapt_vqe`, currently starting around line 2141
- current file size after initial beam helper extraction: about 28k lines

The function currently mixes all of these concerns:

- route identity and CLI/config normalization
- problem construction, including Hubbard, Hubbard-Holstein, and molecular
  fixture routes
- pool construction and pool filtering
- score/cost term configuration across phase 1, phase 2, phase 3, batching,
  pruning, and beam
- optimizer dispatch and optimizer-specific state, including SPSA/QNSPSA and
  deterministic optimizers
- controller snapshots, maturity caps, phase shots, and measurement work
- runtime prune logic and prune derivative propagation
- route-C plateau acquisition
- noise/oracle-gradient/final-noise-audit paths
- resume/current-checkpoint/replay telemetry
- beam search execution
- final output payload assembly

That is the core design problem: the function is not just "running" the
algorithm. It is defining problem identity, algorithm identity, feature routing,
scoring policy, optimizer policy, noise policy, telemetry, and output format in
one local scope.

## Target Shape

Split the pipeline into explicit route layers:

1. `ProblemSpec`
   - Hamiltonian/problem family.
   - Input state/reference state.
   - Problem-specific fixtures.
   - Hubbard/Hubbard-Holstein/molecular routing belongs here, not in the main
     runner body.

2. `PoolSpec`
   - Pool family and pool filters.
   - Full-meta/HVA/UCCSD/CSE/PAOP/etc.
   - Pool generation should be independent from optimizer and telemetry.

3. `AdaptAlgorithmSpec`
   - Method identity: append-only ADAPT, Geo-ADAPT, SNAKE/TETRIS-like routes,
     beam, batching, pruning.
   - This should express "which algorithm are we running?" without embedding
     low-level CLI fallback logic.

4. `ScoreSpec`
   - Paper-I canonical intent: one cost/score term form shared across phase 1,
     phase 2, phase 3, batching, pruning, and beam.
   - Phase-specific behavior should come from the record set being considered,
     not from different hidden scoring equations unless explicitly routed.

5. `OptimizerSpec`
   - Inner optimizer choice and parameters.
   - Shared dispatcher for SPSA/QNSPSA/ROTOSOLVE/POWELL/COBYLA/BFGS.
   - Optimizer memory reuse and reduced/full refit policy should be explicit
     route settings, not scattered local branches.

6. `NoiseSpec`
   - Noise/oracle-gradient/final-noise-audit behavior.
   - Paper-I noiseless canonical routes should not carry noise-route machinery
     in the hot path except as disabled metadata.
   - Shot/noise fields that are not part of the Paper-I cost term should live
     under the noise route.

7. `RunControlSpec`
   - Max ADAPT iterations.
   - Optional gradient stop/target stop/segment cap/wallclock cap.
   - Paper-I canonical route can simply stop at the configured iteration cap
     unless the user enables another stop policy.

8. `TelemetrySpec`
   - Logs, checkpoint, resume, debug rows, replay telemetry, and output payload
     assembly.
   - These are important but should not be interleaved with algorithm logic.

## Initial Module Plan

Candidate modules under `pipelines/static_adapt/`:

- `adapt_run_config.py`
  - dataclasses for `ProblemSpec`, `PoolSpec`, `AdaptAlgorithmSpec`,
    `ScoreSpec`, `OptimizerSpec`, `NoiseSpec`, `RunControlSpec`,
    `TelemetrySpec`
  - CLI/config validation currently embedded near the top of
    `_run_hardcoded_adapt_vqe`

- `problem_routes.py`
  - Hubbard and Hubbard-Holstein construction
  - molecular fixture route dispatch
  - exact/reference-state inputs

- `pool_routes.py`
  - pool construction and filtering
  - full-meta/HVA/UCCSD/CSE/PAOP selection

- `score_routes.py`
  - unified Paper-I score/cost term
  - legacy score shims, if still needed, behind explicit names

- `optimizer_routes.py`
  - `_run_stochastic_inner_optimizer`
  - `_run_deterministic_inner_optimizer`
  - SPSA/QNSPSA memory payloads

- `noise_routes.py`
  - phase-3 oracle-gradient config
  - final noise audit
  - analytic/value-noise path
  - Qiskit/noisy backend route hooks

- `controller_routes.py`
  - controller snapshot payloads
  - maturity/shot scheduling
  - measurement work accounting

- `prune_routes.py`
  - runtime prune derivatives
  - prune refit window selection
  - prune audit payloads

- `route_c_plateau.py`
  - route-C plateau acquisition state, events, seed probes, and unlock logic

- `checkpoint_telemetry.py`
  - current checkpoint writer
  - replay/current payload helpers
  - final output telemetry assembly

- `beam_search.py`
  - already started for behavior-preserving beam helper extraction

## Suspected Archaic Or Noncanonical Paper-I Features

These should be audited, not immediately deleted:

| Area | Current examples | Paper-I canonical stance |
|---|---|---|
| Phase-3 motif route | `phase3_motif_source_json`, motif usage/output library | likely archaic unless a visible Paper-I run still uses it |
| Shadow legacy geometry | `phase3_shadow_legacy_geometry_mode`, legacy bridge | likely quarantine behind legacy diagnostics |
| Legacy pairwise scoring | `PHASE2_LEGACY_PAIRWISE_SCORE_FORMULA`, `phase2_*legacy*` names | likely replace with one canonical score route |
| Phase shortlist legacy hook | `_phase_shortlist_with_legacy_hook`, `_phase1_lane_shortlist_with_legacy_hook`, `_phase2_lane_health_shortlist_with_legacy_hook` | audit whether this is real behavior or compatibility telemetry |
| Phase-2 cheap-score epsilon | `phase2_cheap_score_eps` | likely a numerical guard, not a Paper-I concept; decide whether it belongs in score config |
| Backend shortlist | `phase3_backend_shortlist` | likely diagnostics/hardware route, not canonical Paper-I scoring |
| Oracle gradient/noise route | `phase3_oracle_gradient_config`, final noise audit, value noise | move to `noise_routes.py`; disabled in canonical noiseless route |
| Segment controls | `adapt_segment_target_depth`, `adapt_segment_max_new_admissions`, `adapt_segment_wallclock_cap_s` | run-control/resume feature, not core algorithm |
| Route C plateau | `route_c_plateau_*` | separate route module; decide whether Paper-I canonical uses it |
| Full/refit window complexity | nested/refit-window policy, legacy coupled windows | likely split optimizer refit policy from geometry measurement windows |
| Controller shots/maturity | `phase*_maturity_shot_cap`, phase shots, controller snapshot shots | separate controller/noise/accounting route; preserve if used for measurement-work accounting |

## Keep Or Move, Not Delete

These are probably real behavior even if they should move:

- problem construction
- pool construction
- optimizer dispatch
- score/cost term implementation
- pruning behavior
- beam behavior
- checkpoint/resume telemetry
- output payload compatibility

The fact that something is not in Paper I as prose does not mean it can be
deleted if old JSON consumers, tests, or active run scripts depend on it.

## Paper-I Canonical Route Contract

Desired route should eventually become explicit, for example:

```text
problem: Hubbard or Hubbard-Holstein
pool: explicit pool key/filter
algorithm: Paper-I SNAKE/static ADAPT route
score: unified Paper-I score term
optimizer: explicit inner optimizer and maxiter
stop: max ADAPT iterations by default, optionally gradient/target stops
noise: off
telemetry: on, but separated from algorithm decisions
```

Important scoring rule from the user:

- The cost term should be the same mathematical object across phase 1, phase 2,
  phase 3, batching, and beam.
- The record set may change between phases.
- The scoring equation should not silently change because a local variable name
  says `phase2`, `phase3`, `legacy`, or `cheap`.

## Refactor Order

1. Finish behavior-preserving beam extraction.
   - Current module: `beam_search.py`.
   - Do not move `_evaluate_beam_branch` or `_materialize_beam_child` until the
     payload/policy helpers around them are stable.

2. Extract config normalization into dataclasses.
   - No behavior change.
   - Goal: reduce the top-of-function run-setting block into a resolved config
     object.

3. Extract optimizer dispatch.
   - Preserve exact optimizer outputs and telemetry.
   - Add unit tests for SPSA/QNSPSA/deterministic dispatch payloads.

4. Extract problem and pool routes.
   - Preserve current CLI/API inputs.
   - Add smoke tests for Hubbard, Hubbard-Holstein, and molecular fixture
     dispatch if those routes remain live.

5. Extract noise/oracle routes.
   - Keep disabled/noiseless route cheap and explicit.
   - Move Qiskit/noise/final-audit machinery out of the canonical hot path.

6. Extract controller/prune/route-C modules.
   - Keep behavior parity first.
   - Only then decide what is canonical, optional, or legacy.

7. Define the Paper-I canonical route.
   - One named config object.
   - One score equation.
   - Explicit stop policy.
   - Noise off.
   - Legacy variants behind explicit compatibility routes.

8. Legacy deletion/quarantine pass.
   - Requires user approval per feature group.
   - Requires tests or run-payload parity proving no canonical route drift.

## Deletion Gate

Before deleting or disabling a suspected archaic feature, answer:

- Is it reachable from current CLI/API defaults?
- Is it reachable from Paper-I run scripts or support artifacts?
- Does a test cover it?
- Does any output payload/source-map/current-status artifact expect the field?
- Is it only telemetry, or does it affect selected generators/theta/energy?
- Has the user explicitly chosen delete, quarantine, or keep?

Allowed outcomes:

- `keep_canonical`
- `move_optional`
- `quarantine_legacy`
- `delete_after_tests`
- `unknown_needs_user_decision`

## Immediate Next Safe Work

No behavior changes yet.

Recommended next implementation slice:

- continue extracting beam payload/policy/helper code from
  `_run_hardcoded_adapt_vqe`
- update `beam_refactor_migration.md` after each slice
- keep `_evaluate_beam_branch` and `_materialize_beam_child` in place until
  their dependencies are isolated

Recommended next planning slice:

- create an inventory table for every CLI/config option in
  `_run_hardcoded_adapt_vqe`
- classify each as canonical Paper-I, optional live feature, telemetry/resume,
  noise/hardware route, molecular route, or suspected legacy

## Open User Decisions

- Should the canonical Paper-I route allow gradient/target stop, or only max
  iteration stop by default?
- Is route-C plateau acquisition part of Paper I, or an optional exploratory
  route?
- Should molecular routes remain in this entrypoint or move behind a separate
  molecular/vibronic entrypoint?
- Which score formula is the canonical Paper-I equation?
- Should all `legacy_*` scoring modes be quarantined once the canonical score
  route is explicit?
- Should shot/noise/controller maturity fields remain in static ADAPT config
  when noise is off, or move entirely under noise/controller route specs?
