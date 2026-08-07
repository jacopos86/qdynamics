# Paper-Lane Routing and Ownership Refactor Plan

Status: planning and decision capture only. Do not treat this file as
authorization to move production code, alter scientific defaults, or launch
runs.

## Planning authority

This plan is a replacement contract, not a summary of current Markdown.
Resolve Paper-I defaults in this order:

1. explicit decisions recorded with the user during this refactor;
2. the locked provenance at
   `MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723/paper_i_hh_macro_common_accuracy_20260723_provenance.json`
   and its hash-locked tracker;
3. route-faithful characterization tests and verified executable behavior;
4. current repository Markdown only as migration inventory.

Conflicting documentation, unqualified aliases, registry defaults, old route
names, and historical implementation notes are not default-setting authority.
The cited provenance is accepted wholesale except where this plan records an
explicit user-approved change. Do not reopen its individual numerical settings
as user questions.

## Objective

Create a minimal agent-facing repository interface in which root routing reveals
only the paper lane and workflow needed for the current request. Keep scientific
implementation, provenance, evidence, and analysis owned by the appropriate
paper lane while preserving a neutral shared scientific core.

The narrower `agent_guidance/shared/icm-gitnexus-pilot-plan.md` applies this
ownership model to cross-paper scientific flow, progressive-disclosure
campaigns, and read-only structural navigation. It does not replace this plan.

## Active scope

Only Paper I is active for detailed implementation and cleanup. The
Paper-I/Paper-IV peer-producer relationship and the allowed downstream
Paper-II/Paper-III handoffs are settled at the ownership level only. Detailed
Paper-II through Paper-V interfaces remain deferred.

## Settled decisions

1. This remains one Git repository.
2. The repository contains five separate paper-owned lanes.
3. Root `AGENTS.md` is ultimately a global if-then router, not a catalog of
   Paper-I settings or implementation details.
4. Detailed restructuring starts with Paper I; detailed Paper-II through
   Paper-V agent surfaces are deferred except where their ownership must be
   stated to prevent cross-lane drift.
5. Paper I owns Hubbard--Holstein static-construction evidence; Paper IV owns
   molecular-vibronic static-construction evidence. They are equal-footing
   producer lanes that may invoke the same paper-neutral ADAPT/SNAKE method.
6. Paper II owns physical time dynamics and may consume a neutral accepted
   ansatz export from Paper I or Paper IV.
7. Paper III owns QSE/excited-dynamics evidence and may later consume declared
   static-construction exports, Paper-II dynamics exports, or both.
8. Shared scientific code is paper-neutral and owns no paper evidence.
9. A Paper-I ADAPT **adaptive sequence** is distinct from a Paper-II physical
   **time trajectory**.
10. Geo-ADAPT is a hidden Paper-I-local benchmark and is excluded from every
   ordinary Paper-I request and every other paper-owned lane. It remains
   reachable only when explicitly named for locked benchmark/replay work.
11. The canonical Hubbard--Holstein Paper-I run size is `L=2`.
12. A Paper-I run's standard analysis is part of run finalization, not a
    separate user workflow. Every completed run, and every deliberately stopped
    run with at least one complete accepted state, reports the accepted
    same-cutoff energy-error trace over all completed controller rounds.
13. Standard post-run resource reporting includes the effective-plateau prefix
    and a common-accuracy comparison against the canonical append-ADAPT source
    for the same resolved physical problem when that source is available.
14. Qiskit compilation remains a post-run observation. It cannot enter
    candidate selection, admission, pruning, beam competition, stopping, or
    optimizer decisions.
15. The canonical automatic effective-plateau reporting policy is
    `paper_i_effective_plateau_v1`: select the earliest accepted prefix whose
    same-cutoff absolute energy error is within 10 percent of the best error
    observed over the available accepted trajectory. This is a versioned
    current policy, not a claim that every historical Paper-I result used the
    same tolerance. In particular, an earlier Paper-I candidate note described
    a different five-percent condition based on whether any later prefix
    improved the error; that historical rule must not be conflated with this
    current closeness-to-best selector.
16. A successfully accepted scientific trajectory remains successful if its
    post-run resource summary encounters an unexpected tooling defect. The
    summary is post-run observation, not part of the scientific run's success
    condition; the defect is repaired and the same summary is rerun.
17. Newly produced canonical Paper-I runs must emit the complete accepted
    history, reference state, operator definitions, parameter identities, and
    provenance needed by the standard resource compiler. Automatic Qiskit
    reporting is therefore the expected success path, not an optional best-effort
    feature surrounded by speculative failure branches.
18. The new summary interface accepts only canonical Paper-I run results and
    their canonical append-ADAPT comparison source. It does not normalize old
    schemas, search for substitute fields, accept incomplete histories, or
    expose legacy recovery modes. Existing historical-recovery scripts remain
    outside this interface.
19. The Paper-I agent interface uses pre-execution progressive disclosure.
    User intent selects a canonical configuration branch and reveals only the
    decisions required by that branch. This if-then structure is not fallback
    behavior: it does not react to failure, substitute another route, or retry
    with different scientific semantics.
20. The standard matched-accuracy observation resolves the canonical
    append-ADAPT comparison source automatically from the same resolved
    physical problem and comparison contract. The user does not select an
    artifact path as part of the normal interface.
21. The canonical pool begins with unfiltered `full_meta`, including HVA
    families. Physical macro lanes drive Phase I and Phase II; shortlisted
    macros are projected into exact-cardinality-one Pauli children for Phase
    III. The fixed-sector and binary-padding symmetry guard is mandatory.
22. `full_meta_minus_hva` is a hidden explicit ablation, not a default. Intact
    physical-macro admission is one explicit controlled comparison.
23. The provenance-locked Phase-III selector, source-metric no-overlap trust
    rule, whitened complete accepted refit, optimizer, seed, resource cost,
    shortlist, and numerical policies remain silent canonical internals.
    Unwhitened accepted refitting is a one-off source-locked experiment and
    creates no public policy, route, adapter, or architecture seam.
24. Append-only ADAPT is hidden from the ordinary interface. Existing locked
    artifacts are frozen comparator/replay sources, while an explicitly named
    comparator study may still run append-ADAPT. The method-general Qiskit
    compiler and future accepted-ansatz exporter may consume either SNAKE or
    append results from an allowed producer lane, but SNAKE is the ordinary
    Paper-I default.
25. JR-SNAKE, FM-SNAKE, historical Route-A/B/C spellings, versioned SR
    profiles, H2O derivative routes, and other experimental controllers remain
    explicit compatibility or lane-owned research paths. The canonical resolver
    never enumerates, probes, or falls back to them.

## Paper-I routing work completed

- Paper-I SR-SNAKE language now lives in
  `agent_guidance/static-adapt/CONTEXT.md`, not the root context.
- `agent_guidance/static-adapt/AGENTS.md` now provides the Paper-I conditional
  router.
- `agent_guidance/static-adapt/run-guide.md` now defines the minimal canonical
  `run_sr_snake(problem, request=None)` interface.
- Optional batching, pruning, beam, insertion, stopping, and resume details are
  separated into conditional policy files.
- `PaperIRunSummary` now owns canonical post-run trajectory, plateau,
  common-accuracy, selected-prefix Qiskit resources, and `S_alg` accounting.
- Remaining Paper-I prune/batch studies use the existing source-locked run
  workflow. A new six-regime canonical campaign launcher is not a prerequisite.
- Cross-paper handoff and ICM/GitNexus sequencing are governed by
  `agent_guidance/shared/icm-gitnexus-pilot-plan.md`.

Deferred cross-lane drift:

- The root router still names a missing Paper-II lane `AGENTS.md`.
- Paper-II code imports problem, pool, and exact-energy helpers from the
  Paper-I `pipelines.static_adapt` namespace, including a private helper.
- Current Paper-II runners require a raw `--artifact-json` path.
  `pipelines.scaffold.runtime_loader.load_scaffold_runtime_input(...)` loads the
  saved ansatz represented by that artifact but does not resolve canonical
  Paper-I provenance or select a historical accepted prefix by iteration.
- Current manuscript routing references missing
  `MATH/paper_facing/two_paper_strategy.md` and
  `MATH/paper_facing/shared/claim_source_types.md`.

## Intended ownership shape

```text
root router
  -> shared scientific core
       -> static ansatz construction
  -> Paper I: Hubbard--Holstein problem adapter and evidence
       -> accepted ansatz export
  -> Paper IV: molecular-vibronic problem adapter and evidence
       -> accepted ansatz export
  -> Paper II: accepted-ansatz import adapter
       -> AP-McLachlan time dynamics -> dynamics export
  -> Paper III: owned import of declared static and/or dynamics exports
       -> QSE/excited dynamics
  -> Paper V: high-U regularization/GKBA
```

Each paper-owned lane contains or routes to its own:

- agent contract and progressive-disclosure interface;
- executable workflows and canonical settings;
- run manifests and imported-input receipts;
- validation and analysis workflows;
- provenance, evidence, and paper-support records.

## Deferred accepted-ansatz handoff notes

The neutral export preserves producer-lane evidence identity. Paper II owns the
adapter that transforms a Paper-I or Paper-IV export into a Paper-II-native
input. This is preserved context for later refactors, not an active Paper-I
deliverable.

Its eventual external interface may be conceptually equivalent to:

```python
resolve_dynamics_seed(
    source,
    *,
    construction_method="snake",   # snake | append_adapt
    through_accepted_step,
) -> PaperIIDynamicsSeed
```

The adapter hides artifact paths, schema variants, archive members, and replay
details. It must:

- resolve the named producer-lane result and provenance for the requested
  physical problem;
- reconstruct the accepted ansatz through the requested inclusive accepted
  construction step;
- validate the physical problem, encoding, cutoff, symmetry sector, ordered
  generators, parameter identities and values, and prepared-state replay;
- emit a Paper-II-owned seed plus an import receipt that preserves the exact
  producer-lane source identity and hash;
- leave every source artifact unchanged.

SNAKE and append-ADAPT are two accepted source variants behind this one
interface. Geo-ADAPT is not an accepted source variant.

The user will commonly specify an ansatz prefix as “up to iteration 10” or
“including iteration 10.” Both mean through accepted construction step 10,
inclusive, unless the user explicitly requests another convention. The export
records the source-native index kind, such as SNAKE controller round or
append-ADAPT iteration.

## Target Paper-I agent control surface

The Paper-I lane should use the following shallow routing tree:

```text
agent_guidance/static-adapt/
  AGENTS.md
  CONTEXT.md
  run-guide.md
  policies/
    batching.md
    pruning.md
    beam.md
    insertion.md
    stopping.md
  reporting/
    run-summary.md
  handoffs/
    <implementation-only handoffs; not ordinary routing>
```

Responsibilities:

- `AGENTS.md` is an if-then router. It contains no detailed settings.
- `CONTEXT.md` is only the Paper-I domain glossary.
- `run-guide.md` defines the minimal canonical run request and silent defaults.
- `policies/*.md` are read only when the user activates or asks about that
  policy.
- `reporting/run-summary.md` defines the automatic completion output and the
  user-specified accepted-prefix query.
- `handoffs/*.md` are implementation-only transfer artifacts. The ordinary
  router never reads them.

The detailed `sr-snake-refactor-plan.md`, route registries, historical handoffs,
paper-support records, and compatibility notes remain deeper evidence. The
ordinary agent router must not require them for a canonical run.

## Canonical Paper-I run

An ordinary Paper-I run request resolves without an interview to:

```text
physical model = Hubbard--Holstein
lattice size = L=2
method family = SR-SNAKE
pool = unfiltered full_meta, HVA included
candidate flow = physical macro lanes -> symmetry-retained singleton children
hard symmetry and binary-padding guard = on
admission = singleton
batching = off
pruning = off
beam = off
insertion = plateau-triggered commutation-reduced insertion
maximum controller rounds = 50
exact-ED stop = off
phase-live hysteresis = absent
post-run summary = automatic
```

Except for the insertion promotion and typed optional policies below, the
canonical settings resolver copies the complete locked provenance baseline:
full active-plus-singleton Phase-III response, source-metric no-overlap trust,
whitened complete accepted refit, optimizer, seed, shortlist, resource-cost,
and numerical policies. These are not questions in the ordinary interface.

Current executable alignment:

- the typed facade defaults to plateau-triggered, commutation-reduced
  insertion;
- append-only insertion remains an explicit typed ablation;
- route identity and public behavior tests preserve that distinction;
- historical append-only evidence retains its original identity.

## Progressive disclosure is not fallback

The minimal Paper-I interface is a decision tree over user intent:

```text
Paper I run
  -> canonical silent defaults
  -> batching mentioned?
       no  -> reveal no batching settings
       yes -> choose greedy or combinatorial
              -> reveal only that admission policy's required settings
  -> pruning mentioned?
       no  -> keep canonical no-prune policy silent
       yes -> choose metric or trust-region pruning
              -> reveal only that pruning policy's required settings
  -> beam mentioned?
       no  -> keep beam settings silent
       yes -> reveal live-branch, children-per-parent, and S_alg-weight choices
  -> insertion mentioned?
       no  -> use canonical plateau-triggered commutation-reduced insertion
       yes -> reveal only the requested insertion ablation
  -> stopping mentioned?
       no  -> use 50 controller rounds
       yes -> reveal maximum-round and optional exact-ED target choices
  -> run completes or user stops after an accepted round
       -> emit the automatic Paper-I run summary
```

Each leaf is resolved before execution and represents one intentional canonical
request. A fallback would instead begin after a requested path failed and try a
different schema, route, setting, or scientific behavior. The new interface
must provide the former and avoid the latter.

## Conditional policy contracts

### Batching

Batching is off and silent by default. If the user enables batching:

1. ask for greedy or combinatorial admission;
2. show the subtype's suggested defaults;
3. ask only for values the user wants to change.

Suggested values:

```text
greedy:
  maximum admitted records = 3
  ranked search window = full

combinatorial:
  maximum admitted records = 3
  ranked search window = 6
```

Combinatorial batching searches generator-distinct subsets of already selected
generator-position records. It does not permute batch order or search new
insertion positions. One admitted batch is one controller round.

### Pruning

Pruning is off and silent by default. Enabling pruning reveals two peer
policies; neither is canonical until the planned comparison supplies evidence:

- **metric pruning** nominates deletions from the regularized local
  metric/response model;
- **trust-region pruning** nominates deletions from the full-logical
  trust-domain model.

Both use a measured delete-and-full-refit result as the acceptance authority.
Neither silently falls back to the other. Historical amplitude, hysteresis,
terminal, and mixed prune modes remain compatibility-only. Deeper settings are
revealed only after the user chooses one of the two peer policies.

### Beam

Beam is off and silent by default. Enabling beam opens:

```text
live branches = 3
children per parent = 2
maximum expanded children per round = 6
fork-local S_alg weight = 0.01
calibration status = uncalibrated default
```

These are suggested values, not hidden constants. The agent must tell the user
that the `0.01` estimator-work weight remains uncalibrated.

Only accepted children continue when a parent produces accepted children.
Unchanged parent archival is not an option. Survival compares cumulative
post-refit energy difference and cumulative fork-local `S_alg` since the common
fork. There is no patience rule, uncertainty margin, hysteresis, or speculative
one-candidate fallback.

### Insertion

The target canonical behavior is
`insertion_commutation_plateau_v1`: append-only position construction while
accepted progress is adequate, followed by a widened logical insertion domain
when the immediately preceding accepted transition is on the plateau. Exactly
commuting-equivalent positions collapse to one canonical representative before
ranking.

The plateau trigger uses realized accepted-state energy decrease, never exact-ED
error. Its current `1e-8` threshold is an internal uncalibrated profile value,
not an ordinary interface question. Append-only insertion remains an explicit
ablation/replay choice rather than the new silent default.

### Stopping

The default finite horizon is 50 controller rounds. If stopping is mentioned,
the user may:

- replace 50 with another positive controller-round limit;
- add a predefined same-problem, same-cutoff exact-ED energy and tolerance.

Exact-ED stopping is evaluated only after a complete accepted refit. It cannot
enter candidate scoring, trust solving, pruning, beam survival, or optimization.
The finite round limit remains active when exact stopping is enabled.

### Observation and reporting

There is no ordinary observation interview. Canonical checkpoints, provenance,
the accepted error trace, plateau resources, append-matched resources, and
algorithmic-work accounting are automatic. A later request for Qiskit costs at
a named accepted controller round reuses the same run-summary interface.

Custom diagnostic logging or artifact destinations are revealed only when the
user explicitly asks about observation mechanics.

## Compatibility and retirement boundary

The new Paper-I router excludes the following from canonical reachability:

- phase-live and retirement hysteresis;
- the historical frozen-parent or unchanged-parent beam archive;
- the legacy adaptive-insertion union of plateau, flatness, repeated-family,
  and escape triggers;
- dormant settings belonging to disabled batching, pruning, or beam policies;
- implicit JR-SNAKE, FM-SNAKE, Geo-ADAPT, append-ADAPT, H2O derivative routes,
  versioned SR profiles, or historical Route-A/B/C selection from an
  unqualified SR-SNAKE request;
- `full_meta_minus_hva`, no-guard, intact-macro, unwhitened-refit, no-Phase-III,
  and end-position-only behavior unless the exact ablation is requested.

Until the final contraction audit is complete, exclusion means quarantine, not
deletion. Historical replay paths retain explicit versioned identities.
Geo-ADAPT remains an explicitly named Paper-I-local benchmark. Existing
append-ADAPT evidence remains a frozen comparator/replay source, and new append
execution requires an explicitly named comparator study. JR-SNAKE and FM-SNAKE
remain explicit compatibility/research routes. None can define canonical
SR-SNAKE defaults.

## Paper-I run-completion interface

The agent-facing Paper-I route should not expose a second mandatory
`analyze this run` workflow. A run ends by passing its immutable accepted
trajectory to one deep post-run summary module. The conceptual interface is:

```python
summarize_paper_i_run(
    run_source,
    *,
    append_reference="canonical_for_resolved_problem",
    requested_controller_rounds=(),
) -> PaperIRunSummary
```

The same interface serves both automatic completion and later user queries:

- normal completion calls it with no requested rounds;
- a deliberate user stop calls it against the last complete accepted
  checkpoint;
- a later request such as “give me Qiskit costs at round 10” calls it with
  `requested_controller_rounds=(10,)`.

Callers should not need to know result paths, history schemas, archive members,
Qiskit transpiler settings, prefix-reconstruction rules, or whether the source
is SNAKE or append-ADAPT. Those are hidden behind the module.

### Automatic summary contents

1. **Accepted error trace**
   - one row for every completed controller round;
   - controller round, active ansatz depth, accepted energy, same-cutoff exact
     energy, and absolute energy error;
   - batches remain one controller round while exposing their resulting active
     depth, so “round” and “number of admitted generators” are never conflated.
2. **Effective-plateau resource observation**
   - apply `paper_i_effective_plateau_v1`: select the earliest accepted prefix
     whose error is within 10 percent of the best error observed over the
     available accepted trajectory;
   - compile that exact prefix with the locked Paper-I Qiskit convention;
   - report compiled two-qubit count, two-qubit depth, total circuit depth, and
     separately the algorithmic-work accounting receipt;
   - record whether the available horizon is a natural terminal run or a
     deliberate user-stopped prefix. A user-stopped summary is still computed
     but must not imply that later, unobserved rounds could not improve it.
3. **Append-matched common-accuracy observation**
   - compare only sources for the same resolved physical problem, cutoff,
     optimizer contract, candidate representation, and compile convention;
   - end the shared comparison window at the earlier effective plateau;
   - define the common target as the larger of the two best errors reached
     inside that window, which guarantees that both methods can reach it;
   - select and compile the first accepted prefix from each method that reaches
     that target.
4. **Requested-round observations**
   - validate each requested controller round against the complete accepted
     history;
   - reconstruct and compile the exact accepted prefix at that round;
   - reuse a matching existing sidecar rather than recompiling it.

Every standard summary input is part of the canonical run producer contract and
the automatic Qiskit observations are expected to succeed. The new interface
has no legacy normalization, substitute-field search, partial-history mode, or
public recovery variants. Historical artifacts remain on their existing
explicit recovery/reporting paths.

If the canonical reporting path encounters a real implementation or environment
defect, preserve the completed scientific result, repair the defect, and rerun
the same summary. Do not create another supported summary state or translate a
reporting bug into a failed SNAKE/ADAPT result.

### Current implementation

- `pipelines.static_adapt.sr_snake.SRRunResult` already exposes the accepted
  trajectory, while legacy run artifacts retain per-round history and accepted
  energies.
- `pipelines/reporting/build_paper_i_selected_prefix_qiskit_sidecar.py` already
  reconstructs and compiles a user-selected accepted history position.
- `pipelines/reporting/build_paper_i_hh_tracking_plateau_costs.py` already
  implements the current 10-percent effective-plateau selector and exact-prefix
  compilation, but it is tied to a specific tracker/campaign.
- `pipelines.reporting.paper_i_run_summary` owns the canonical plateau,
  common-accuracy, exact-prefix compilation, and four-component `S_alg`
  summary contracts.
- older campaign/report builders that duplicate these selectors remain
  compatibility consumers to migrate only when their publication obligations
  are resolved.

The controller and numerical core remain Qiskit-free.

### Governing public behavior regressions

- a completed singleton run emits every accepted error-trace row and the
  expected plateau observation;
- a batched run reports controller round separately from active ansatz depth;
- a deliberately stopped run summarizes the last complete accepted trajectory
  and labels its available-horizon scope;
- the common-accuracy selector returns the first crossing for both SNAKE and
  append-ADAPT under one target;
- a requested controller round reuses the same prefix compiler and compile
  convention as the automatic plateau/matched observations;
- the canonical interface rejects no valid canonical run result through
  speculative legacy or fallback conditions;
- an injected compiler/tooling defect does not alter the already completed
  scientific result and succeeds after the implementation defect is repaired.

## Implementation sequence

The high-level Paper-I interface, conditional policies, and run-summary seam
are implemented. Do not reopen them as one giant refactor or recreate the
superseded GPT-5.6-sol-ultra handoff.

1. Preserve compatibility code and historical evidence through Paper-I
   publication.
2. Perform only explicitly requested, source-locked Paper-I pruning and
   batching studies; do not build a general canonical campaign launcher for
   them.
3. Use ICM first as routing and receipt discipline around a real remaining
   workflow, not as empty campaign scaffolding.
4. When Paper II becomes active, implement the neutral accepted-ansatz
   export/import seam with SNAKE and append fixtures.
5. When Paper IV becomes active, move the Hubbard--Holstein-only restriction
   out of the shared method seam under explicit family-capability tests.
6. Define Paper-III static/dynamics imports only when its scientific workflow
   becomes active.
7. Use GitNexus index-only navigation for a concrete reachability question and
   confirm every graph claim against source and tests.

No scientific run, manuscript mutation, evidence promotion, compatibility
deletion, GitNexus index, commit, or push is authorized by this plan.

## Unresolved questions

None for Paper-I cleanup. Paper-IV family-specific SNAKE admissibility and
Paper-III input composition remain intentionally deferred to their owning
lanes.

Files to edit:

- `agent_guidance/paper-lane-refactor-plan.md`: settled Paper-I run-completion
  behavior, cross-paper ownership, compatibility map, and sequencing.
- `agent_guidance/static-adapt/CONTEXT.md`: Paper-I and cross-paper handoff
  language used by this lane.
- `agent_guidance/shared/icm-gitnexus-pilot-plan.md`: cross-paper flow and
  control/navigation pilot.
- Code: none in this planning step.
