# Static ADAPT Pipeline Teaching And Refactor Map

Purpose: explain what `adapt_pipeline.py::_run_hardcoded_adapt_vqe`
currently owns, how that maps to Paper-I SNAKE notation, and how to refactor
the file without changing behavior first.

This is a documentation and planning artifact only. It does not authorize code
deletion, route-identity changes, or semantic changes by itself.

## Ground Rules

1. Route A is canonical Paper-I SNAKE.
2. Route B is legacy pairwise behavior. Route C is plateau-acquisition behavior
   and should be quarantined as a legacy/diagnostic route unless a later user
   decision promotes it. If retained, B/C should live in separate quarantined
   files. Route A must not depend on them.
3. Route A may have explicit submodes, such as greedy versus combinatorial
   reduced-plane batching, as long as those submodes are visible and route
   identity is intentionally updated when needed.
4. Append-only ADAPT and Geo-ADAPT are important visible comparators.
5. TETRIS-ADAPT, Qubit/QEB-ADAPT, HEA, and family-informed pathways are
   legacy/archaic for the current Paper-I route unless explicitly requested.
   They can be hidden or deleted later only after explicit approval.
6. Cost-term weights and toggles may remain phase-specific, including whether
   Phase 0 uses cost. This flexibility should be represented as configuration,
   not as separate hidden score equations.
7. Beam cost is additive along a branch. Beam survival can have its own
   branch-ranking policy, but it should consume the same route-level cost
   primitive.

## Paper-I Dictionary

Route A should eventually read as a small loop over a growing ordered support:

\[
|\psi_k\rangle = U_k(\theta_k)|\phi_0\rangle,
\qquad
U_k(\theta_k) = \prod_{j\in O_k} \exp(-i\theta_{k,j}G_j/2).
\]

Here `O_k` is the ordered support already admitted into the ansatz, `G_j` is an
operator/generator in that support, `theta_k` is the current parameter vector,
and `|phi_0>` is the problem reference state.

At step `k`, the route builds candidate records:

\[
r=(m,p)\in R_k,
\]

where `m` is a candidate generator, `p` is an insertion position, and `R_k` is
the stage-local candidate set. The canonical family of scores is:

\[
S_k(r;t) =
\frac{\mathrm{Gain}_k(r;t)\,\mathrm{Novelty}_k(r;t)}
     {1+K_k(r;t)}.
\]

`K_k` is the cost burden. The specific gain/novelty features can differ by
stage because the available records differ, but the denominator shape should be
the same conceptual cost primitive.

Batching selects a set:

\[
B\subset R_k,
\qquad
O_{k+1}=O_k\oplus B.
\]

Beam search keeps branch-local supports:

\[
O_b \longrightarrow O_b\oplus B.
\]

Pruning proposes a deletion from `O_k`; deletion is accepted only after the
remove/refit energy-safety check passes. Schur or surrogate scores may nominate
or rank candidates, but they are not deletion authority.

## Current Call Chain

```text
main()
  -> parse CLI
  -> cli_config._resolve_main_cli_configs(...)
  -> cli_config._build_run_hardcoded_adapt_vqe_kwargs(...)
  -> adapt_pipeline._run_hardcoded_adapt_vqe(...)
       validate route/problem/pool/optimizer/noise controls
       resolve or validate problem context
       resolve pool and pool filters
       initialize O_k, theta_k, executors, controller, scoring, prune, beam
       run ADAPT/SNAKE loop
       emit current/checkpoint/final payloads
```

The problem is not that every line is dead. The problem is that one function
currently owns too many separable responsibilities.

## What The Mega-Function Currently Owns

| Concern | Current anchor | Classification |
|---|---|---|
| Public run argument surface | `_run_hardcoded_adapt_vqe` signature and `cli_config._build_run_hardcoded_adapt_vqe_kwargs` | Runtime plumbing |
| Problem request/identity checks | `ProblemRequest`, `ResolvedProblemContext`, local validation | Shared problem boundary |
| Hamiltonian/reference construction | mostly `builders/problem_setup.py` and `builders/problem_registry.py` | Should stay outside route runner |
| Pool filters and full-meta policy | `builders/pool_resolution.py`, `hh_pool_presets.py`, `static_provenance.py` | Shared pool boundary |
| Route identity payload | `route_identity.py`, local payload assembly | Route metadata |
| Phase 0/1/2/3 score record assembly | local loop plus `hh_continuation_scoring.py` | Active Route A plus compatibility shims |
| Batch selection | `hh_continuation_scoring.py`, local admission | Active Route-A submode |
| Beam branch mutation | `_materialize_beam_child`, `engine_support.py`, `beam_search.py` | Active Route-A beam surface |
| SPSA/QNSPSA/POWELL/ROTOSOLVE dispatch | local closures plus `engine_support.py` | Optimizer layer |
| Refit windows | local active-index/window logic | Optimizer and geometry boundary |
| Prune nomination/acceptance | local helpers plus `prune_ladder.py` | Active prune, with diagnostics |
| Route C plateau acquisition | local plateau helpers and payloads | Quarantined Route C |
| Oracle gradient / value noise / final audit | `cli_config.py` dataclasses plus local closures | Noise route |
| Controller snapshots / phase shots | `hh_continuation_stage_control.py` plus local serializers | Controller/telemetry |
| Selector debug rows | local debug serializers | Telemetry only |
| Resume/current/final payloads | local payload assembly | Checkpoint/telemetry |

## Active Route A Surface

The current canonical route is anchored by:

- `pipelines/static_adapt/route_identity.py`
- `agent_guidance/static-adapt/route-a-language.md`
- `pipelines/static_adapt/adapt_pipeline_route_alignment.md`

Route A should own:

- the clean SNAKE loop over `k`;
- `O_k`, `theta_k`, and `|psi_k>`;
- candidate records `r=(m,p)`;
- staged candidate sets `R_k`;
- canonical cost-aware scoring;
- batching submodes that are explicit Route-A submodes;
- beam branch exploration when enabled;
- pruning through energy-safety acceptance;
- checkpoint hooks that do not change the selection rule.

Route A should not own:

- Route B pairwise legacy scoring;
- Route C plateau acquisition;
- noise/oracle execution;
- TETRIS/Qubit/QEB/HEA/family-informed comparator logic;
- bulky final payload formatting;
- molecular or Hubbard-Holstein Hamiltonian construction.

## Route And Comparator Classification

| Surface | Desired status | Notes |
|---|---|---|
| Route A / SNAKE | visible canonical route | Clean, readable, isolated from B/C |
| Route A reduced-plane batching | current canonical Route-A batching name | `route_identity.py` currently anchors `phase3_batch_selection_mode=reduced_plane` |
| Greedy reduced-plane batching | implemented/available ordered batching submode | Treat as Route-A submode only after route identity is intentionally extended |
| Combinatorial reduced-plane batching | implemented/available ordered batching submode | Treat as Route-A submode only after route identity is intentionally extended |
| Route A beam | visible Route-A submode | Additive branch cost |
| Append-only ADAPT | visible comparator | Important for Paper-I comparisons |
| Geo-ADAPT | visible comparator | Important for Paper-I comparisons |
| Route B legacy pairwise | hidden/quarantined route file | May reuse A/shared primitives; A must not import B |
| Route C plateau acquisition | hidden/quarantined route file | May reuse A/shared primitives; A must not import C |
| TETRIS-ADAPT | archaic comparator | Hide; deletion requires explicit approval |
| Qubit/QEB-ADAPT | archaic comparator | Hide; deletion requires explicit approval |
| HEA / family-informed pathways | archaic comparator/control pathways | Hide unless explicitly requested |
| Oracle/noise route | separate diagnostic/noise route | Must not leak into clean Route A |

## Problem And Pool Boundaries

Most problem construction is already outside `adapt_pipeline.py`.

Code anchors:

- `pipelines/contracts/problem.py`
- `pipelines/static_adapt/builders/problem_registry.py`
- `pipelines/static_adapt/builders/problem_setup.py`
- `pipelines/static_adapt/builders/pool_resolution.py`
- `pipelines/static_adapt/builders/hh_pool_presets.py`
- `pipelines/contracts/static_provenance.py`

The route runner should consume a resolved problem:

```text
ResolvedProblemContext
  -> Hamiltonian/operator representation
  -> reference state |phi_0>
  -> exact/reference metadata when available
  -> default pool key / family metadata
```

and a resolved pool:

```text
ResolvedPoolPlan
  -> candidate generators
  -> labels/classes
  -> HH full_meta/full_meta_minus_hva filter metadata
  -> child/shared Pauli expansion metadata when enabled
```

Remaining problem-specific conditionals inside the mega-function should be
classified as either:

- validation gates;
- temporary compatibility checks;
- pool-resolution leftovers;
- route-specific constraints that should move into route config.

Pool plumbing also includes:

- selected-logical matching/filter reports;
- legal-subspace filter metadata;
- HH `full_meta` cache/reinstantiation metadata;
- class/label filter provenance;
- child/shared Pauli expansion compatibility checks.

These affect the resolved candidate universe, performance, or provenance. They
should remain pool-resolution/runtime plumbing rather than Route-A selector
logic.

## Child-Set And Beam Surfaces

Child-set expansion and beam are related but distinct.

Current child-set surfaces include:

- archival Phase-III runtime split paths such as `phase3_runtime_split_mode`;
- global pre-Phase-1 pool expansion such as
  `adapt_child_pool_expansion_mode=global_pauli_child_sets_v1`;
- `shared_pauli_pool_mode`, which is a separate shared-Pauli pool path.

Where validation enforces mutual exclusion, those surfaces must stay mutually
exclusive. Beam may rank branch continuations that include selected child sets,
but beam is not itself the child-set mechanism.

## Cost And Score Boundary

The desired conceptual score family is:

\[
S_k(r;t) =
\frac{\mathrm{Gain}_k(r;t)\,\mathrm{Novelty}_k(r;t)}
     {1+K_k(r;t)}.
\]

The code may keep phase-local weights and toggles:

```text
cost_enabled_phase0
cost_enabled_phase1
cost_enabled_phase2
cost_enabled_phase3
lambda_2q[k], lambda_depth[k], lambda_theta[k], lambda_shot[k], ...
```

This is acceptable because it is configuration, not another hidden equation.
The refactor goal is to make this explicit:

```text
ScoreSpec
  -> phase index k
  -> enabled cost components
  -> phase-local weights
  -> gain feature source
  -> novelty feature source
  -> denominator shape: 1 + K_k
```

Code anchors:

- `pipelines/scaffold/hh_continuation_scoring.py`
- `FullScoreConfig`
- `SimpleScoreConfig`
- `phase0_raw_gradient_pilot_components`
- `phase2_raw_geometry_score`
- `phase3_canonical_score_components`
- `BatchSelectionProposal`

Batching should remain additive unless a documented joint component replaces a
specific additive component:

\[
K(B;t)=\sum_{r\in B} K_3(r;t),
\qquad
S(B;t)=\frac{\Delta E(B;t)}{1+K(B;t)}.
\]

Beam should carry cumulative branch burden:

\[
K(O_b\oplus B)=K(O_b)+K(B).
\]

Beam survival may use an energy-cost branch policy, but that policy should be
documented as branch survival, not as a new candidate score equation.

## Concrete Topic Index

### Molecular And Hubbard-Holstein Definitions

Most definitions belong outside `adapt_pipeline.py`.

`problem_registry.py` owns problem-family registration and default pool keys.
`problem_setup.py` owns Hamiltonian and reference-state construction.
`pool_resolution.py` and `hh_pool_presets.py` own HH pool construction and
filters.

What remains inside the mega-function is orchestration: passing request fields,
validating compatibility, and wiring the resolved problem into the route. During
refactor, the function should move toward accepting resolved problem/pool specs
instead of carrying problem-specific construction details.

### SPSA And Inner Optimizer Config

SPSA/QNSPSA/POWELL/ROTOSOLVE are optimizer choices, not route identities.

Current anchors:

- `cli_config.py` maps CLI flags into `_run_hardcoded_adapt_vqe` kwargs.
- `engine_support.py` owns valid optimizer constants and deterministic helper
  wrappers.
- The mega-function owns local stochastic/deterministic optimizer closures.

Refactor target: `optimizer_routes.py`.

That module should own:

- optimizer key validation;
- SPSA/QNSPSA parameter normalization;
- deterministic optimizer dispatch;
- reduced-active-index objective setup;
- optimizer memory reuse;
- normalized result telemetry.

### Segment Target Depth, Max New Admissions, Wallclock Cap

These are run-control/resume controls:

```text
adapt_segment_target_depth
adapt_segment_max_new_admissions
adapt_segment_wallclock_cap_s
```

They limit how much of a run segment executes. They are not Paper-I route
identity. They belong in a future `RunControlSpec` or `segment_routes.py`.

### Phase3 Motif / History Bonuses

Motif/history features can seed or annotate candidate behavior. In canonical
Route A they should be tie-break or diagnostic unless explicitly routed as an
ablation.

Classification:

- canonical tie-break: allowed;
- additive score modification: ablation/diagnostic;
- hidden primary-score modification: not allowed for clean Route A.

### Phase3 Shadow Legacy Geometry

Shadow legacy geometry should be treated as telemetry unless evidence shows it
changes selector behavior.

Classification:

- shadow payload attached for comparison/debug: telemetry;
- proxy-reduced selector mode that changes score inputs: explicit noncanonical
  selector mode;
- hidden fallback that changes the selected record: must be quarantined or
  made explicit.

### Backend Shortlist

Backend shortlist/transpile modes are hardware-cost diagnostics or hardware
route support. They are not clean Route-A identity.

Classification:

- clean Route A can emit enough metadata for later hardware diagnostics;
- backend shortlist belongs in `noise_routes.py` or `backend_cost_routes.py`;
- `transpile_shortlist_v1` should remain explicit and fail closed when the
  required backend context is absent.

### Legacy Hooks

Any field with `legacy`, Route B, Route C, old pairwise novelty, old raw score,
legacy backend, or legacy geometry naming should be classified before movement:

```text
route_b_legacy
route_c_legacy_or_diagnostic
compatibility_payload_only
hidden_archaic_comparator
delete_after_user_approval
unknown_needs_evidence
```

The default stance is quarantine first, delete later only after approval.

### Phase Shots And Controller Snapshots

Controller snapshots and phase shots come from `hh_continuation_stage_control.py`
and local serializers.

They can affect caps, liveness, maturity, and measurement-work accounting. Much
of their payload is telemetry. Refactor target: `controller_routes.py` plus
`checkpoint_telemetry.py`.

### Inactive Prune Schur Nomination Payload

Schur/inactive prune payloads are nomination and diagnostic machinery.

They may answer:

```text
Which existing direction in O_k looks removable by a local surrogate?
```

They do not answer:

```text
Is deletion accepted?
```

Deletion acceptance must remain remove/refit energy safety.

### Runtime Prune Derivative Propagation

Runtime derivative propagation computes derivative information for prune
surrogates. It belongs near prune diagnostics, not in the route runner.

Refactor target: `prune_routes.py` or `static_prune_derivatives.py`.

### Route C

Route C is plateau acquisition. It is not canonical Route A.

Desired boundary:

```text
route_c_plateau.py
  -> plateau state
  -> dormant-coordinate records
  -> plateau score payload
  -> route-C-specific trial optimizer hooks
```

Route A may expose shared primitives that Route C calls. Route A must not import
or branch on Route C internally.

### Phase3 Oracle Gradient Config

Oracle gradient config is a noise/oracle route concern. It belongs outside the
clean no-noise Route-A loop.

Current anchors:

- `Phase3OracleGradientConfig`
- oracle-gradient validation in `cli_config.py`
- local oracle scout/session handling in `adapt_pipeline.py`

Refactor target: `noise_routes.py`.

### Phase3 Inner Value Noise

Inner value noise is post-expectation value perturbation, not the canonical
Paper-I clean objective and not the same as physical shot execution.

Classification: noise diagnostic route. It must be explicit.

### Resume And Telemetry

Resume/checkpoint/final payloads are persistence, not algorithm identity.

Refactor target:

```text
checkpoint_telemetry.py
resume_routes.py
```

These should own:

- structural resume validation payloads;
- current JSON payloads;
- final route/optimizer/noise/selector payloads;
- replay telemetry;
- boundary-refit metadata.

### Selector Debug Rows

Selector debug rows explain why a record was selected. They should serialize:

- selected `r`;
- gain/novelty/cost components;
- batch/beam/child-set fields;
- shadow legacy diagnostics;
- score aliases for compatibility.

They should not change selection.

Refactor target: `checkpoint_telemetry.py` or `selector_debug.py`.

### Refit Windows

There are two distinct ideas:

1. optimizer refit windows: which active coordinates are reoptimized;
2. geometry/scoring windows: which coordinates enter local reduced geometry.

Do not merge these concepts.

Refactor target:

- optimizer windows -> `optimizer_routes.py`;
- geometry windows -> `score_routes.py` or Route-A selector config.

### `maybe_*` Helpers

Every `maybe_*` helper should be classified by authority:

| Helper type | Meaning |
|---|---|
| selector-changing | may change the selected `r`; must be explicit route/config behavior |
| telemetry-only | records comparison/debug data only |
| fail-open diagnostic | errors become diagnostic status, selection continues |
| fail-closed validation | invalid config stops the run |

This classification is more useful than asking whether the helper is "dead."

## Target Module Boundaries

| Module | Owns |
|---|---|
| `route_a_snake.py` | clean canonical SNAKE loop over `k`: state, records, score, admission, refit, prune calls |
| `route_b_legacy_pairwise.py` | legacy pairwise Route B reproduction, hidden/quarantined |
| `route_c_plateau.py` | Route C plateau acquisition, hidden/quarantined |
| `problem_routes.py` | thin wrapper over existing problem registry/setup contracts |
| `pool_routes.py` | thin wrapper over existing pool resolution/filter contracts |
| `score_routes.py` | `ScoreSpec`, phase cost toggles/weights, score aliases, legacy shims |
| `optimizer_routes.py` | SPSA/QNSPSA/deterministic dispatch, refit windows, optimizer telemetry |
| `beam_search.py` | beam policy plus eventually branch materialization; not child-set expansion itself |
| `prune_routes.py` | prune nomination, derivative diagnostics, energy-safety orchestration |
| `noise_routes.py` | oracle gradient, value noise, final audit, backend shortlist/noise sessions |
| `controller_routes.py` | StageController snapshots, phase shots, measurement-work ledgers |
| `checkpoint_telemetry.py` | current JSON, final payload, selector debug, replay telemetry |
| `comparator_routes.py` | visible append-only and Geo-ADAPT comparators |
| `archaic_comparators.py` | hidden TETRIS/Qubit/QEB/HEA/family-informed compatibility, or deletion staging |

## Staged Refactor Plan

1. Documentation inventory only.
   Classify local helpers and knobs without moving behavior.

2. Config extraction.
   Group current arguments into internal specs while preserving the public
   function signature.

3. Score/cost consolidation.
   Make `ScoreSpec` explicit, including phase-specific toggles/weights and
   optional Phase-0 cost.

4. Optimizer extraction.
   Move SPSA/QNSPSA/POWELL/ROTOSOLVE dispatch and optimizer refit windows.

5. Telemetry extraction.
   Move selector debug rows, current JSON, replay payloads, and final payload
   helpers.

6. Prune extraction.
   Move derivative propagation and Schur nomination payloads while preserving
   energy-safety acceptance.

7. Noise extraction.
   Move oracle gradient, inner value noise, backend shortlist, and final audit.

8. Route quarantine.
   Move Route B and Route C into separate hidden/quarantined modules.

9. Comparator cleanup.
   Keep append-only and Geo-ADAPT visible. Hide TETRIS, Qubit/QEB, HEA, and
   family-informed pathways or delete them after explicit approval.

10. Route-A runner extraction.
    Introduce `route_a_snake.py` only after parity tests can verify that the
    same candidates, admissions, beam children, prune decisions, and payload
    fields are preserved.

## Parity Requirements Before Code Movement

Any later extraction should preserve:

- selected operator sequence `O_k`;
- parameter vector lengths and values after refit within existing tolerances;
- selected candidate record `r`;
- selected batch `B`;
- beam child lineage and cumulative additive cost;
- prune nomination payloads and accepted deletion decisions;
- stop reasons;
- current/final JSON keys used by reports and source maps;
- clean Route-A behavior when noise/oracle routes are disabled.

## Files Referenced

- `pipelines/static_adapt/adapt_pipeline.py`
- `pipelines/static_adapt/cli_config.py`
- `pipelines/static_adapt/route_identity.py`
- `pipelines/static_adapt/beam_search.py`
- `pipelines/static_adapt/engine_support.py`
- `pipelines/static_adapt/prune_ladder.py`
- `pipelines/static_adapt/adapt_pipeline_route_alignment.md`
- `pipelines/static_adapt/adapt_pipeline_refactor_plan.md`
- `pipelines/scaffold/hh_continuation_scoring.py`
- `pipelines/scaffold/hh_continuation_stage_control.py`
- `pipelines/scaffold/hh_continuation_types.py`
- `pipelines/contracts/problem.py`
- `pipelines/contracts/static_provenance.py`
- `pipelines/static_adapt/builders/problem_registry.py`
- `pipelines/static_adapt/builders/problem_setup.py`
- `pipelines/static_adapt/builders/pool_resolution.py`
- `pipelines/static_adapt/builders/hh_pool_presets.py`
- `pipelines/static_adapt/builders/child_pool_expansion.py`
- `agent_guidance/static-adapt/AGENTS.md`
- `agent_guidance/static-adapt/route-a-language.md`

## Immediate Next Step

The next safe working step is a mechanical inventory table of local helpers and
configuration fields inside `_run_hardcoded_adapt_vqe`, using these categories:

```text
route_a_active
route_a_submode
shared_problem
shared_pool
score_cost
optimizer
run_control
beam
prune
noise_oracle
telemetry
visible_comparator
hidden_archaic_comparator
route_b_legacy
route_c_legacy
unknown_needs_evidence
```

That inventory should precede any code movement.
