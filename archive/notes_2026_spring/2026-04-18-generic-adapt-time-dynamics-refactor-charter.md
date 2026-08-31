# Generic ADAPT / Time-Dynamics Refactor Charter

## Purpose

Freeze one implementation-spec-level architecture for the current repo state so work can stop oscillating between prompt exports and start moving into controlled code changes.

This charter is intentionally pragmatic. It is not a greenfield design. It is a minimal-churn plan for repairing the current seam between:

- multi-family static ADAPT production,
- scaffold artifact persistence,
- scaffold replay/follow-on consumers,
- and time-dynamics bootstrap/controller inputs.

## Core diagnosis

The repo already has the hard producer-side generic pieces:

- `pipelines/static_adapt/builders/problem_registry.py`
- `pipelines/static_adapt/builders/problem_setup.py`
- family-local pool builders in `src/quantum/*`
- reusable controller logic in `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py`

The main defect is downstream.

The artifact-to-runtime boundary is still effectively owned by HH-named replay/bootstrap modules. That defect appears in three forms:

1. HH-owned artifact/runtime loading
2. overloaded pool/editability semantics (`candidate_pool_complete` doing too much)
3. default drift risk when loaders recompute today’s defaults instead of using the resolved settings actually used when the artifact was produced

## Hard rule

Every downstream scaffold consumer must start from:

```python
load_scaffold_runtime_input(...)
```

This is the most important rule in the refactor.

If this rule is violated, the repo will recreate hidden artifact/runtime contracts in multiple places even if the file tree looks cleaner.

Consumers that must obey this rule:

- `pipelines/scaffold/adapt_circuit_cost.py`
- replay/follow-on scaffold consumers
- fixed-manifold loaders/runners
- realtime bootstraps
- the controller itself
- downstream HH diagnostics that currently start from HH replay/fixed-manifold payload assumptions

## Canonical owners to keep

Do not move these out of their current package path in phase 1.

### Problem / Hamiltonian / pool owners

- `pipelines/static_adapt/builders/problem_registry.py`
  - canonical owner for `ProblemRequest`, `ResolvedProblemContext`, `ProblemFamilySpec`, family registry defaults
- `pipelines/static_adapt/builders/problem_setup.py`
  - canonical owner for Hamiltonian reconstruction, exact/reference-state construction, artifact-facing state helpers
- `pipelines/static_adapt/builders/pool_resolution.py`
  - canonical owner for pool-source interpretation and pool materialization
- `src/quantum/operator_pools/*`
- `src/quantum/chemistry/molecular_uccsd.py`
  - keep as family-local source builders

### ADAPT owners

- `pipelines/static_adapt/adapt_pipeline.py`
  - keep as canonical ADAPT orchestrator
- `pipelines/static_adapt/output_artifacts.py`
  - keep as artifact emission owner
- `pipelines/static_adapt/engine_support.py`
  - keep as extracted engine/state helper layer
- `pipelines/static_adapt/cli_config.py`
  - keep as config layer

### Time-dynamics owners

- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py`
  - keep as canonical controller algorithm owner
- `pipelines/time_dynamics/hh_realtime_checkpoint_types.py`
  - keep as canonical runtime/checkpoint type owner
- `pipelines/time_dynamics/hh_fixed_manifold_observables.py`
  - keep as canonical observable-planning math owner

### Keep as wrappers / compatibility surfaces

- `pipelines/hardcoded/*.py`

Do not spend phase-1 refactor budget deleting wrappers.

## New neutral seam to add

Add exactly two new canonical modules under `pipelines/scaffold/`.

### `pipelines/scaffold/runtime_contract.py`

Owns:

- `CandidatePoolSource`
- `ScaffoldRuntimeInput`
- small enums / literals for completeness and structure lock semantics
- runtime provenance helpers if needed

### `pipelines/scaffold/runtime_loader.py`

Owns:

- `load_scaffold_runtime_input(...)`
- parsing of the new neutral artifact block
- fallback normalization of legacy HH payloads during migration
- reconstruction of `ResolvedProblemContext` via the current builder-layer owners
- decoding of serialized states / layout / runtime theta

This must become the only generic artifact-to-runtime parser.

## Neutral serialized artifact contract

ADAPT should dual-write a new top-level block:

```text
scaffold_artifact:
  schema_version: scaffold_artifact_v2
  artifact_kind: adapt_result

  problem:
    request: <serialized ProblemRequest>
    resolved_family: <canonical family key>
    resolved_layout: <serialized layout/register semantics>
    runtime_data: <JSON-safe family payload>
    resolved_runtime_defaults:
      controller_profile: <resolved value actually used>
      continuation_mode: <resolved value actually used>

  states:
    reference_state: <repo-standard encoded state>
    prepared_state: <repo-standard encoded state>

  scaffold:
    parameterization: <deserialize_layout-compatible payload>
    theta_runtime: [float]
    theta_logical: [float] | null
    candidate_pool_source:
      source_kind: resolved_pool | selected_terms_only
      pool_key: str | null
      completeness: complete | partial | selected_only
      pool_build_kwargs: {...}
      filter_payload: {...}
    structure_locked: bool

  observables:
    exact_energy: float | null

  provenance:
    requested_problem_key: <original user/input alias> | null
    source_artifact_kind: <legacy/import source kind> | null
    source_artifact_ref: <path/hash/id> | null

  extensions:
    hh: {...} | null
    <other family namespaces as needed>
```

### Required rules for this block

- `resolved_runtime_defaults` must carry the resolved values actually used when the artifact was created.
- loaders must not silently recompute current defaults for new neutral artifacts.
- `resolved_layout` should be persisted if layout semantics could drift or cannot be reconstructed deterministically from the raw request.
- raw legacy JSON blobs should not be part of the semantic core.

## Neutral loaded runtime contract

```python
@dataclass(frozen=True)
class CandidatePoolSource:
    source_kind: Literal["resolved_pool", "selected_terms_only"]
    pool_key: str | None
    completeness: Literal["complete", "partial", "selected_only"]
    pool_build_kwargs: Mapping[str, Any]
    filter_payload: Mapping[str, Any]

@dataclass(frozen=True)
class ScaffoldRuntimeInput:
    resolved_problem: ResolvedProblemContext
    psi_ref: np.ndarray
    psi_initial: np.ndarray
    base_layout: AnsatzParameterLayout
    theta_runtime: np.ndarray
    theta_logical: np.ndarray | None
    candidate_pool_source: CandidatePoolSource
    structure_locked: bool
    exact_energy: float | None
    provenance: Mapping[str, Any]
    extensions: Mapping[str, Any] = field(default_factory=dict)

    @property
    def h_poly(self):
        return self.resolved_problem.hamiltonian
```

### Semantics

- `structure_locked` answers: may scaffold structure change at all?
- `completeness` answers: how much of the originating candidate pool can be faithfully re-materialized?

Phase-1 editability rule:

```text
can_structural_edit = (structure_locked == False) and (completeness == "complete")
```

Compatibility field only:

```text
candidate_pool_complete := (completeness == "complete")
```

New code should stop reading the legacy boolean directly.

## HH modules that become adapters/shims

These stop owning the generic boundary and become HH-specific adapter surfaces.

- `pipelines/scaffold/hh_vqe_from_adapt_family.py`
  - HH legacy normalizer / family alias adapter / compatibility `ReplayScaffoldContext` builder
- `pipelines/time_dynamics/hh_fixed_manifold_mclachlan.py`
  - HH bootstrap shim over `runtime_loader`
- `pipelines/time_dynamics/hh_realtime_from_adapt_artifact.py`
  - HH CLI/bootstrap surface over neutral runtime input + canonical controller
- `pipelines/time_dynamics/hh_fixed_manifold_measured.py`
  - HH measured/oracle adapter runner over neutral runtime input
- `pipelines/time_dynamics/hh_realtime_measurement.py`
  - HH services implementation; keep math intact, do not genericize in phase 1

## HH-specific modules to leave HH-specific

Do not flatten these into fake-generic abstractions in phase 1.

- `pipelines/static_adapt/builders/hh_pool_presets.py`
- `pipelines/scaffold/hh_continuation_types.py`
- `pipelines/scaffold/hh_continuation_generators.py`
- `pipelines/scaffold/hh_continuation_motifs.py`
- `pipelines/scaffold/hh_continuation_symmetry.py`
- downstream HH-only diagnostics:
  - `pipelines/time_dynamics/hh_checkpoint_local_adapt.py`
  - `pipelines/time_dynamics/hh_fixed_scaffold_exact_fit.py`

## Controller boundary decision

Do not make `ReplayScaffoldContext` the target generic type.

Change the controller boundary directly so the controller consumes:

- `ScaffoldRuntimeInput`
- a small service bundle for observable planning, measurement/oracle behavior, drive policy, and diagnostics

If HH paths still need `ReplayScaffoldContext`, build it in HH adapter layers only.

## Time-dynamics services

Do not create per-family controller subclasses.

Use one controller algorithm with injected services.

Minimal service bundle concept:

- observable planning / summarization service
- measurement/oracle service
- drive service
- metric/diagnostic service

Profiles:

- generic/default profile
  - exact-mode / simple metric support only
- HH profile
  - current HH drive augmentation
  - current HH measurement/oracle behavior
  - current HH metric shaping and diagnostics

## Non-goals for phase 1

Do not do these yet.

- move builder files out of `builders/`
- delete `pipelines/hardcoded/*`
- genericize HH measurement/oracle math
- genericize HH motif/symmetry/generator metadata
- create per-family controller subclasses
- serialize full candidate pools into every artifact by default
- promise universal legacy non-HH artifact support before new neutral artifacts exist
- perform package/path churn before the runtime seam is stable

## PR sequence

### PR0 — contract freeze

Deliver:

- this charter
- exact field map from legacy HH artifact fields to neutral fields
- exact semantics for `structure_locked` and `completeness`
- identified parity fixtures

### PR1 — add neutral runtime modules

Add:

- `pipelines/scaffold/runtime_contract.py`
- `pipelines/scaffold/runtime_loader.py`

Goal:

- normalize at least one legacy HH artifact into `ScaffoldRuntimeInput`

### PR2 — problem snapshot and state serialization helpers

Modify:

- `problem_registry.py`
- `problem_setup.py`

Add:

- request serialization/deserialization
- resolved-problem snapshot serialization/rebuild
- JSON-safe runtime-data serialization hooks
- state encode/decode helpers

### PR3 — split pool source contract from HH overlays

Modify:

- `pool_resolution.py`

Add:

- `resolve_candidate_pool_source(...)`
- `materialize_candidate_pool(...)`

Goal:

- persist a neutral pool source without persisting HH overlay baggage

### PR4 — dual-write neutral scaffold artifact from ADAPT

Modify:

- `adapt_pipeline.py`
- `output_artifacts.py`

Goal:

- successful ADAPT runs emit both legacy HH fields and `scaffold_artifact_v2`

### PR5 — move a non-controller consumer first

Target:

- `pipelines/scaffold/adapt_circuit_cost.py`

Goal:

- prove the neutral contract is sufficient for a real downstream consumer before touching the controller

### PR6 — retarget observable planning and fixed-manifold bootstrap

Modify:

- `hh_fixed_manifold_observables.py`
- `hh_fixed_manifold_mclachlan.py`

Goal:

- observables and fixed-manifold bootstrap begin from neutral runtime input

### PR7 — extend checkpoint type semantics

Modify:

- `hh_realtime_checkpoint_types.py`

Add:

- explicit completeness semantics
- explicit structure-lock semantics
- service bundle types

### PR8 — change controller input boundary

Modify:

- `hh_realtime_checkpoint_controller.py`

Goal:

- controller boots from `ScaffoldRuntimeInput` + services

### PR9 — switch HH bootstraps and diagnostics

Modify together:

- `hh_realtime_from_adapt_artifact.py`
- `hh_fixed_manifold_measured.py`
- later `hh_checkpoint_local_adapt.py`
- later `hh_fixed_scaffold_exact_fit.py`

Goal:

- HH bootstrap/diagnostic surfaces preserve behavior while no longer owning the generic boundary

## Proof obligations / tests

### Unit contract tests

Need round-trip tests for:

- `ProblemRequest`
- resolved problem snapshot
- JSON-safe `runtime_data`
- encoded states
- serialized scaffold layout
- `CandidatePoolSource`
- completeness semantics

### Legacy HH normalization tests

For fixture artifacts, old path vs new neutral loader must agree on:

- `psi_ref`
- `psi_initial`
- `base_layout`
- `theta_runtime`
- resolved family
- resolved pool key
- completeness mapping
- structure-lock mapping

### Non-controller consumer parity

First target:

- `adapt_circuit_cost.py`

Goal:

- same scaffold interpretation and cost behavior through neutral runtime input

### Observable-plan parity

Target:

- `hh_fixed_manifold_observables.py`

Goal:

- same plan structure / ordering / runtime-rotation semantics

### Controller bootstrap parity

Use current HH artifact anchors.

Goal:

- same startup acceptance/rejection
- same reconstruction behavior
- same gating for structure lock / completeness
- HH services initialize equivalently

### Historical artifact regression

Use exact compare for:

- schema/version fields
- family/pool identifiers
- structure lock
- completeness state
- deterministic discrete decisions

Use tolerance compare for:

- energies
- fidelities
- density/doublon/staggered metrics
- time-series floats

Normalize away:

- timestamps
- wall-clock durations
- temp/output paths

### Architecture guardrails

Add rules/tests that:

- only `runtime_loader.py` knows legacy artifact field names directly
- generic consumers import `load_scaffold_runtime_input(...)`
- controller no longer imports HH replay artifact loaders directly
- scaffold consumers stop parsing raw legacy HH payload fields

## First non-HH runtime milestone

Be strict.

Phase-1 “generic time dynamics” means:

- ADAPT emits neutral scaffold artifact for at least one non-HH family
- neutral loader reconstructs `ScaffoldRuntimeInput`
- controller runs under default/generic service profile
- pool conventions are still family-correct via `CandidatePoolSource` + `pool_resolution.py`

Recommended first non-HH smoke target:

1. `hubbard`
2. `spin_boson`

Do not promise HH-equivalent measured/oracle richness for non-HH in phase 1.

## End-state definition

The refactor is successful when all of these are true.

- ADAPT for any supported family can emit a neutral scaffold artifact block that freezes the runtime-relevant settings actually used.
- Downstream consumers no longer need HH-specific legacy parsing to reconstruct problem, Hamiltonian, scaffold layout, runtime theta, and pool source.
- The controller starts from `ScaffoldRuntimeInput` plus services, not an HH replay object.
- HH remains the richest path in the repo, but as a rich adapter profile layered on top of generic contracts rather than as the hidden owner of those contracts.

## Decision

This charter is the implementation spec.

No more architecture prompt churn unless the implementation reveals a real contradiction in these assumptions.
