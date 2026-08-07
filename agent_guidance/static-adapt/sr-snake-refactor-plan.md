# SR-SNAKE Refactor Plan

Status: Issues 7--16 and the
`insertion_commutation_plateau_v1` experiment are complete. This file now
preserves deep-module design and implementation history; it does not select the
next ticket or define canonical defaults independently.
Scope: `pipelines/static_adapt/` and the minimum tests needed to preserve the
current Paper-I SR-SNAKE execution contract.
Not authorized by this document: scientific reruns, route promotion, settings
changes, manuscript edits, artifact deletion, or removal of compatibility
routes.

## Governing target and authority

The current architectural target is
`agent_guidance/paper-lane-refactor-plan.md`. If this issue history conflicts
with that plan, the target plan wins.

Canonical Paper-I settings come from the locked provenance at
`MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723/paper_i_hh_macro_common_accuracy_20260723_provenance.json`
and its hash-locked tracker, except for explicit user-approved changes recorded
in the governing plan. Current Markdown, unqualified aliases, registry defaults,
and old ticket notes are migration evidence rather than default-setting
authority.

The provenance baseline fixes the current SR-SNAKE profile, unfiltered
`full_meta` pool with HVA included, macro-lane to hard-guarded singleton-child
funnel, full active-plus-singleton Phase-III response, source-metric
no-endpoint-overlap trust rule, whitened complete accepted refit, and the
remaining silent numerical settings. Do not interview the user about those
fields individually.

The target architecture changes only the explicitly settled surfaces:
plateau-triggered commutation-reduced insertion becomes canonical; batching,
pruning, and beam remain silent/off until requested; enabled pruning exposes
metric and trust-region policies as peers; historical routes are quarantined
from canonical reachability; and post-run summary behavior is automatic.
Unwhitened accepted refitting is a one-off experiment and creates no
architecture seam.

## Purpose

Paper I presents SR-SNAKE as a compact state machine over a growing ordered
ansatz. The implementation does not currently expose that state machine at the
same conceptual resolution. Its mathematical kernels, pool builders, refit
logic, accounting, and reporting helpers are partly modular, but
`adapt_pipeline.py::_run_hardcoded_adapt_vqe` still assembles them through one
very large interface that also exposes historical routes, comparators,
diagnostics, noise paths, and experimental controls.

The refactor goal is not to make the repository small. It is to make the
scientific SR-SNAKE path readable through one deep module with a small
interface, while keeping numerical kernels and compatibility implementations
behind internal seams.

## Current Architecture

The present execution path is:

```text
paper_i_runner / CLI
  -> cli_config
  -> adapt_pipeline._run_hardcoded_adapt_vqe
       -> problem and pool builders
       -> Phase-I/II/III candidate construction
       -> supported H/G response and trust solve
       -> singleton admission
       -> accepted Powell refit
       -> optional prune / beam / batching branches
       -> estimator ledger
       -> checkpoint and result payloads
```

Important existing modules:

| Concern | Current authority |
|---|---|
| Core Pauli, ansatz, and execution primitives | `src/quantum/` |
| Hamiltonian and pool construction | `pipelines/static_adapt/builders/` |
| SR family/profile contracts | `pipelines/static_adapt/sr_snake_route_profile.py` |
| Candidate features and historical scoring | `pipelines/scaffold/hh_continuation_scoring.py` |
| Supported metric/Hessian trust solves | `pipelines/static_adapt/joint_linear_solve.py` |
| Accepted full-ansatz Powell chart/refit | `pipelines/static_adapt/accepted_refit.py` |
| Prune planning | `pipelines/static_adapt/prune_ladder.py` and related prune modules |
| Beam execution | `pipelines/static_adapt/beam_search.py` |
| Measurement accounting | `pipelines/static_adapt/estimator_call_ledger.py` |
| Qiskit/resource reporting | Paper-I exact-bench/reporting sidecar paths |
| Checkpoint/result construction | `checkpoint_telemetry.py`, `output_artifacts.py` |
| Current orchestration | `pipelines/static_adapt/adapt_pipeline.py` |

The architecture problem is intent diffusion: callers must understand a union
of canonical, optional, experimental, and historical controls even when they
want one ordinary canonical Paper-I SR run.

## Scientific State Machine To Preserve

The extracted controller must make this path visible:

```text
initialize accepted ansatz O_k and parameters theta_k
repeat:
    construct admissible candidate-position records R_k
    Phase I: screen and shortlist records
    Phase II: acquire the retained singleton response
    Phase III: build the supported active-plus-singleton H/G model
    solve the configured trust problem
    select exactly one candidate-position record
    append it at the authorized position
    refit the complete accepted ansatz under the configured refit policy
    optionally verify pruning without mutating the surviving branch
    close estimator accounting and checkpoint the accepted state
until the configured run-control condition
```

Batch admission is supported by the future architecture but is not silently
SR singleton admission. A batching policy must carry a distinct route/profile
identity and explicit admission cardinality.

## Proposed Deep Module

Create a package only after characterization tests protect the current route.
Do not pre-create a file for every conceptual noun. The initial package is:

```text
pipelines/static_adapt/sr_snake/
  __init__.py
  contracts.py
  runner.py
  _legacy_adapter.py
```

Later extraction may add the following internal modules only when each hides a
substantial coherent decision:

```text
  _controller.py
  _selection.py
  _transition.py
```

Do not create new `candidate_domain.py`, `scoring.py`, `policies.py`,
`measurement_accounting.py`, or `hardware_costs.py` stubs merely to mirror the
planning vocabulary. Candidate-domain construction and ranking remain separate
functions and types inside the selection seam, but they need not begin in
separate files. Policy types belong with the request contract. Measurement
accounting continues to use `estimator_call_ledger.py`; accepted refitting
continues to use `accepted_refit.py`; supported trust solves continue to use
`joint_linear_solve.py`; and Qiskit costs continue through the existing
reporting paths.

### Public interface

Expose one small call:

```python
def run_sr_snake(
    problem: ResolvedProblemContext,
    request: SRRunRequest | None = None,
) -> SRRunResult:
    ...
```

`ResolvedProblemContext` is the existing resolved physical-calculation type.
Pool resolution and estimator execution are internal implementation details in
the first extraction. Do not invent public `ResolvedPoolPlan` or
`EstimatorExecutor` seams before a second concrete implementation needs them.

The new call is the deep module: the caller supplies the physical calculation
and the scientifically meaningful SR-SNAKE instructions, while the module hides
pool materialization, state preparation, candidate phases, trust solving,
accepted refitting, accounting, checkpoint events, and legacy translation.

### `contracts.py`

Own typed public data:

- `SRRunRequest`;
- `SRMethodPolicy`;
- `SRExecutionPolicy`;
- `SRObservationPolicy`;
- `SRStopPolicy` and optional `ExactEDStop`;
- conditional batching, pruning, and beam policy choices;
- `SRRunResult`.

The request distinguishes scientific method policy, run-control/resume policy,
and observation/persistence policy. Route family, profile, digest, seed,
supported trust policy, accepted-refit policy, and numerical guards are derived
internally and returned as receipts.

Disabled optional features carry no dormant subordinate settings. Legacy flag
names and compatibility payload shapes do not enter these public types.

The first stopping contract is intentionally small:

```text
maximum controller rounds:
  default 50 or explicit positive value

optional exact-ED target:
  predefined same-cutoff energy plus absolute tolerance and source identity
```

One controller round means one complete candidate-selection and accepted-refit
cycle. A batch may add more than one generator in that round; it does not
silently consume one "iteration" per generator. The run stops when the first
active configured condition is satisfied. The maximum-round condition remains
active when an exact-ED target is supplied so an unreachable target cannot make
the run unbounded.

### `runner.py`

Own request normalization and complete-run orchestration. It resolves the
profile receipt, constructs internal dependencies, invokes the controller, and
returns the primary scientific result plus provenance. It does not implement
Phase-I/II/III mathematics and does not compile post-run Qiskit circuits.

During the expand stage, `runner.py` may delegate through a private legacy
adapter to `_run_hardcoded_adapt_vqe` without changing the current execution.
After migration, the new controller becomes authoritative for the current
Paper-I default while historical routes continue through explicitly named
compatibility entry points.

### `_controller.py`

Own the accepted SR-SNAKE state machine:

```text
accepted state -> one complete controller round -> next accepted state
```

`SRControllerState` contains accepted operators and parameters, accepted
energy, candidate availability, live phase/trust memory, estimator-ledger
prefix, and controller counters. It does not contain CLI parsing, output paths,
Qiskit reporting, or manuscript/report formatting.

The controller coordinates two deep internal decisions:

1. **selection**: construct admissible candidate-position records, run
   Phase I/II/III, perform the supported trust solve, and return an immutable
   singleton or batch admission decision with receipts;
2. **transition**: admit that decision, perform the full accepted refit,
   optionally perform pruning verification and deletion refit, close the
   estimator-ledger round, and return the next accepted state with receipts.

Checkpointing consumes accepted-state events after a transition. It does not
participate in candidate selection or change the transition.

### Internal `_selection.py`, when earned

Expose one internal operation equivalent to:

```text
accepted controller state -> admission decision
```

It owns the Phase-I/II/III orchestration while delegating numerical operations
to existing modules. Candidate-set construction and ranking stay logically
separate inside this seam: domain construction answers which records exist;
ranking answers which record or batch should be admitted.

The existing Phase-III predictive cost remains here. Realized fork-local
`S_alg` does not: it exists only after a branch has executed and is consumed by
an enabled beam-survival policy.

### Internal `_transition.py`, when earned

Expose one internal operation equivalent to:

```text
accepted controller state + admission decision -> next accepted controller state
```

It owns append/batch admission, accepted full-ansatz refit, optional pruning,
non-worsening acceptance, ledger closure, and transition receipts. It does not
select candidates or format output artifacts.

### Existing deep dependencies

The refactor composes existing authorities instead of wrapping each in a new
pass-through module:

- `phase_shortlists.py` for shortlist mechanics;
- `joint_linear_solve.py` for supported metric/trust solves;
- `accepted_refit.py` for the supported-FS Powell chart and refit;
- `prune_ladder.py` and the current prune helpers for enabled pruning;
- `estimator_call_ledger.py` for immutable measurement work and `S_alg`;
- `checkpoint_telemetry.py` for checkpoint projection;
- `output_artifacts.py` and Paper-I reporting paths for observational output;
- `beam_search.py` only as compatibility evidence until the new fork-local beam
  semantics are implemented behind the new controller.

## Approved Interface

The interface was designed three ways and exercised with a throwaway state
prototype before approval:

1. **Flat configuration object**: one run call, but it recreates the
   mega-function as a dataclass containing enable flags and dormant settings.
   It is shallow and was rejected.
2. **Stateful fluent builder**: progressive disclosure is visually convenient,
   but callers must learn configuration order and many methods, while hidden
   mutable defaults complicate replay and tests. It was rejected.
3. **Nested immutable request**: one run call plus typed admission, pruning,
   beam, stopping, resume, and observation choices. Disabled variants have no
   subordinate fields. It gives the greatest depth and locality and is
   approved.

The prototype confirmed that the approved form represents singleton, greedy
batching, combinatorial batching, pruning, beam, default and explicit round
limits, optional exact-ED stopping, and simultaneous stop receipts without a
route string or historical flag.

### External seam

The entire public operation is:

```python
def run_sr_snake(
    problem: ResolvedProblemContext,
    request: SRRunRequest | None = None,
) -> SRRunResult:
    ...
```

`request=None` means the current Paper-I default:

```text
singleton admission
pruning off
beam off
50 controller rounds
exact-ED stopping absent
fresh reference-state start
default observation policy
```

The package root exports only:

- `run_sr_snake`;
- `SRRunRequest` and its intentional policy variants;
- `ExactEDStop`;
- typed resume and observation inputs;
- `SRRunResult` and its public result receipts.

`SRControllerState`, `CandidatePositionRecord`, `AdmissionDecision`, and
`AcceptedTransition` are internal implementation types. Callers and
high-interface tests do not construct them.

### Public request

`SRRunRequest` contains exactly three top-level choices:

```text
method
execution
observation
```

- `method: SRMethodPolicy` owns only currently intentional scientific
  variability:
  - `SingletonAdmission`, `GreedyBatchAdmission`, or
    `CombinatorialBatchAdmission`;
  - `PruningOff` or `RecoverabilityPruning`;
  - `BeamOff` or `ForkLocalBeam`.
- `execution: SRExecutionPolicy` owns:
  - `SRStopPolicy`;
  - a fresh reference-state start or typed accepted-state resume.
- `observation: SRObservationPolicy` owns checkpoint cadence and destination,
  event logging, diagnostics, and artifact destinations. Changing only this
  object must leave the accepted trajectory unchanged.

The caller does not choose a route profile. The runner derives the immutable
route family, profile, digest, and active-policy receipt from the request.
Full Phase-III active-plus-proposal response geometry, the supported trust
solve, the full supported-FS Powell accepted refit, the current symmetric
candidate-cost policy, seed and numerical guards, Phase liveness, and
hysteresis-off behavior remain internal to the current method profile.

If future scientific work deliberately makes one of those behaviors variable,
it receives a new typed choice and profile identity then; it is not exposed in
advance as a dormant generic setting.

The public request never exposes historical Route-A/Route-B/Route-C names,
formal-manifold toggles, optimizer selection, hysteresis, plateau escape,
noise/oracle controls, dormant optional settings, raw checkpoint dictionaries,
individual mega-function keyword names, or compatibility payload fields.

### Stop policy

`SRStopPolicy` contains:

```text
maximum_controller_rounds
exact_ed_target = absent or ExactEDStop
```

`maximum_controller_rounds` defaults to 50 and may be explicitly replaced.
Paper-specific reproduction tools may separately verify that the requested
value matches their recorded source lock; ordinary callers do not need to
understand that provenance mechanism. `ExactEDStop` contains the predefined
energy, absolute tolerance, and immutable source/comparison-space receipt. The
runner validates that the receipt names the same resolved physical problem,
sector, and cutoff. The controller does not call an exact solver to discover
or update the target during execution.

The stop receipt records:

- every active condition and whether it fired;
- completed controller rounds;
- accepted-operator count, which may differ under batching;
- primary and simultaneously satisfied stop reasons;
- accepted energy at stopping;
- exact target, tolerance, observed absolute difference, and source receipt
  only when exact stopping is enabled.

### Controller state

`SRControllerState` owns only information needed to continue the algorithm:

- ordered accepted operators and insertion positions;
- physical, logical, and runtime parameter identities and values;
- accepted energy and accepted-state fingerprint;
- available candidate identities and selection counts;
- controller-round index;
- Phase-I/II/III live state and shortlist memory required by the current
  profile;
- adaptive trust state;
- optimizer memory required for deterministic continuation;
- estimator-ledger prefix;
- optional prune metadata;
- optional beam lineage and common-fork ledger prefix.

Output paths, report payloads, Qiskit costs, exact-state fidelity, manuscript
labels, and compatibility-only fields are not controller state.

### Selection result

`AdmissionDecision` is immutable and contains:

- one selected candidate-position record or one explicitly enabled batch;
- generator, pool, lineage, insertion-position, and symmetry identities;
- Phase-I, Phase-II, and Phase-III population and shortlist receipts;
- the decision-authoritative Phase-III response and supported-rank receipt;
- trust-solve receipt and proposed initial coordinate values;
- predictive candidate-cost receipt;
- estimator events acquired while reaching the decision.

It contains no accepted post-refit energy and cannot mutate the accepted
ansatz.

### Accepted transition

`AcceptedTransition` contains:

- the preceding accepted-state identity;
- the admission decision identity;
- appended singleton or batch identities;
- complete accepted-refit chart, optimizer, and energy receipts;
- optional prune nomination, verification, deletion, and deletion-refit
  receipts;
- non-worsening acceptance receipt;
- authoritative estimator-ledger delta and closed prefix;
- the next `SRControllerState`;
- one checkpoint-ready accepted-state event.

Selection never writes a checkpoint. Checkpointing consumes the accepted event
after the transition has closed.

### Public result

`SRRunResult` contains:

- one primary final accepted ansatz, parameters, and energy;
- ordered accepted-state trajectory;
- resolved problem and route/profile receipts;
- stop receipt;
- Phase and transition receipts needed for scientific replay;
- all-work and winning-lineage estimator accounting;
- full fork-tree provenance only when beam is enabled;
- observation/artifact receipts.

Legacy payload dictionaries are produced by a compatibility serializer outside
the controller. Post-run Qiskit costs attach as an observational sidecar and
cannot change `SRRunResult`'s accepted trajectory.

## Characterization Contract

The characterization suite has three layers:

1. **Pure contract tests**
   - profile resolution and digest;
   - progressive-disclosure request shapes;
   - stop-policy composition and same-transition precedence;
   - deterministic serialization of request, state, transition, and result
     receipts.
2. **Small deterministic Hubbard--Holstein trajectory**
   - two-site open-boundary Hubbard--Holstein problem;
   - binary boson encoding;
   - the smallest cutoff and unfiltered `full_meta` pool that still exercise
     the current Phase-I/II/III, supported trust, full accepted-refit, fixed
     sector, and estimator-ledger path;
   - two or three controller rounds with Powell and a fixed seed;
   - exact equality where identities/counts are discrete and explicitly
     defined numerical tolerances for energies, parameters, response values,
     and supported solves.
3. **Read-only Paper-I provenance anchor**
   - current no-prune route family/profile/digest;
   - active source-lock requirement that the current completion study uses 50
     rounds, plus same-cutoff problem identities;
   - known accepted-trajectory, Phase-III, accepted-refit, and estimator-ledger
     values from an immutable completed artifact when locally available;
   - an explicit missing-source failure if the manuscript/support pointer
     cannot be resolved, rather than fabricated expected values.

The existing two-site accepted-refit tests are useful fixture material but are
not sufficient by themselves: some manually activate beam behavior or filter
the pool and therefore do not characterize the exact current default. The
implementation agent must time the unfiltered small fixture and select the
smallest route-faithful cutoff; test speed may determine the physical size, but
must not justify disabling a scientific stage that the characterization is
supposed to protect.

## Composition Rules

Pruning is naturally a policy inside an SR singleton route:

```python
profile = SRProfile(
    admission=SingletonAdmission(),
    pruning=VerifiedFSTrustPruning(...),
)
```

Batching changes the admission mechanism and therefore must change the
route/profile identity:

```python
profile = SRProfile(
    admission=GreedyBatchAdmission(maximum_size=3),
    pruning=PruningOff(),
)
```

FM outer information should wrap the internal SR controller rather than copy
it:

```text
FM outer-information wrapper
  -> prepare transported or exact geometry context
  -> delegate one ordinary round to SRController
  -> update, invalidate, or reanchor outer-information state
```

Shadow FM cannot change the SR trajectory. Active reuse may omit approved
measurements only when closure passes and must fall back to exact SR
acquisition when closure fails. A pure FM optimizer that owns selection,
accepted refitting, and rollback is a separate controller family.

## Failure Discipline: Success-First, Traceable Rejection

The refactor must not introduce validation gates, assertions, rejection
conditions, or negative tests merely because an invalid state is conceivable.

A hard failure is permitted only when it is traceable to at least one of:

1. an explicit requirement requiring rejection;
2. a demonstrated scientific or logical invariant;
3. a concrete security, privacy, financial, or data-integrity risk;
4. an observed regression or specifically documented edge case.

Every hard-failure branch and failure-oriented test must identify its
authorizing requirement, invariant, risk, or regression. If that justification
cannot be stated concretely, omit the failure branch or test.

For incomplete, ambiguous, legacy, or unexpected representations, prefer, in
order:

1. lossless normalization;
2. use of an authoritative equivalent field or receipt;
3. a documented reasonable default;
4. graceful degradation;
5. a warning or recoverable error;
6. hard failure only when continued execution would be unsafe or logically
   invalid.

Before introducing a hard failure, record:

- the concrete harm caused by continuing;
- why normalization or graceful degradation is insufficient;
- the requirement or invariant authorizing rejection.

Do not promote implementation preferences, cleaner schemas, missing redundant
receipts, or speculative edge cases into scientific invariants.

Negative tests must reproduce required rejection behavior or a documented
regression. Test externally meaningful behavior through the public interface;
do not test speculative internal constraints.

Source-lock, route-identity, pool-identity, estimator-accounting, symmetry,
cutoff, and scientific-policy conflicts may remain strict because silently
continuing could change the experiment or its evidence. Equivalent legacy
representations, reporting-schema variation, serialization shape loss that can
be reconstructed unambiguously, and missing redundant metadata should normally
be normalized rather than rejected.

A repaired or normalized path must emit a receipt describing what was
normalized and why, but creating that receipt must not add algorithmic
measurement work.

## Migration Strategy

Use expand--migrate--contract. Do not rewrite the mega-function in place.

### Stage 0: characterize

Before extraction:

1. Resolve the current Paper-I visible SR source and its immutable route
   contract.
2. Add interface-level characterization tests for:
   - normalized route/profile identity;
   - ordered accepted operators;
   - controller energies/checkpoints;
   - Phase-III response indices and supported ranks;
   - accepted-refit chart;
   - estimator ledger closure;
   - `S_alg`;
   - Qiskit sidecar provenance.
3. Preserve at least one exact source-value anchor.
4. Do not manufacture negative characterization tests. Preserve observed
   behavior and add rejection tests only for an existing contract or reproduced
   regression.

### Stage 1: add the facade

Introduce `SRRunRequest` and `run_sr_snake` while delegating to the existing
pipeline unchanged. The old CLI remains a compatibility adapter. This makes
new run construction smaller without changing science.

The first facade accepts only `ResolvedProblemContext` and `SRRunRequest`.
Internally it translates to the current legacy keyword surface. It must not
route through `paper_i_runner.py::run_paper_i_route_a`, because that facade owns
the separate Route-A joint/batch-oriented execution contract.

### Stage 2: extract resolved controller context

Move construction of immutable dependencies needed by the current no-prune
profile behind the new runner: resolved profile, pool, compiled Hamiltonian,
reference state, numerical kernels, optimizer configuration, estimator ledger,
and accepted initial state. This is a private execution context, not a new
public parameter bundle.

The legacy adapter and the future controller must be constructible from the
same resolved request so characterization can compare them without translating
scientific settings twice.

### Stage 3: extract selection as one deep decision

Move the current no-batching Phase-I/II/III path behind the internal selection
seam. The seam starts from an accepted state and ends with one immutable
admission decision. It includes admissible record construction, shortlist
progression, measured response acquisition, supported trust solving, and final
ranking. It does not mutate the accepted ansatz.

Historical funnels, FM controllers, noise paths, hysteresis, plateau escape,
and compatibility fallbacks stay in the old pipeline. Do not move their union
into the new selection seam.

### Stage 4: extract the accepted transition

Move append admission, full accepted refit, optional prune verification, energy
acceptance, estimator-ledger closure, and accepted-state receipt construction
behind the internal transition seam. Preserve the current no-prune behavior
first; add enabled pruning only after the no-prune transition is characterized.

### Stage 5: move the ordinary controller loop

Compose selection and transition in `SRController`. The controller owns
internally available stopping rules and emits accepted-state events. The runner
owns complete-run assembly. Checkpoint/output adapters consume events without
changing controller decisions.

At this stage the current Paper-I no-prune, no-batch, no-beam profile no longer
executes its controller loop inside `adapt_pipeline.py`. Compare the new
interface and the legacy entry point against the same characterization anchors.

### Stage 6: add optional policies deliberately

Add pruning, batching, and beam composition in separate behavior-changing
tickets after the default controller is extracted:

1. current recoverability pruning after accepted refit;
2. greedy and combinatorial batching at the admission seam;
3. the new fork-local beam controller and realized `S_alg` dominance policy.

Do not preserve an archaic optional route merely because it exists in the
mega-function. Each enabled policy implements the decisions recorded in this
plan and receives an explicit serialized identity.

### Stage 7: separate accounting and hardware reporting

Make the estimator event ledger authoritative for `S_alg`. Keep Qiskit
compilation observational. Prove deterministic post-run recomputation against
runtime totals.

### Stage 8: quarantine compatibility routes

Only after canonical SR no longer imports their control flow:

- keep historical replay adapters explicitly named;
- move Route-B/Route-C and obsolete diagnostics out of the canonical hot path;
- retain payload readers required by preserved artifacts;
- delete code only after reachability, tests, and user approval.

## One-Day Useful Slice

A one-day refactor should not promise full cleanup. The useful bounded target
is:

1. characterization tests for the current canonical no-prune route and its
   immutable provenance;
2. typed `SRRunRequest`;
3. `run_sr_snake` facade;
4. a canonical profile constructor;
5. the existing mega-function retained as the implementation adapter.

This reduces command complexity and gives future extraction a stable seam. It
does not yet reduce the internal size of `adapt_pipeline.py`.

## Non-Negotiable Invariants

- Pauli/Jordan--Wigner conventions remain unchanged.
- Regime physics and same-cutoff exact references remain outside route
  profiles and are source-locked per run.
- Route family, profile, digest, and executable settings remain separately
  serialized.
- Phase-III supported projection and accepted-refit whitening remain separate
  policies.
- Runtime, logical, and physical coordinate identities remain explicit.
- Candidate-set construction remains separate from scoring.
- The estimator ledger remains the sole authority for `S_alg`. An explicitly
  configured beam policy may consume a closed fork-local `S_alg` view;
  post-run Qiskit costs remain observational and separate.
- Optional policy enablement never silently changes route identity.
- Historical artifacts remain readable through compatibility adapters.

## Completion Criteria

The refactor is complete when:

- the canonical SR scientific loop is readable in one controller module;
- a caller can launch it without knowing legacy flags;
- canonical and experimental profiles use typed policies;
- the old CLI can translate into the new request without scientific drift;
- canonical tests exercise the new public interface rather than private
  helpers;
- `S_alg` is reproducible from immutable ledger events;
- Qiskit compilation is outside the controller hot path;
- GitNexus impact analysis shows canonical SR no longer depends on quarantined
  historical controller branches.

## Main-Agent Working Notes

This section preserves dated planning and implementation history. Later notes
may supersede earlier sequencing decisions, but must not rewrite what an earlier
ticket actually did.

### 2026-07-23 planning decisions

Planning and implementation follow the Matt Peacock engineering flow:

```text
planning:
  grill-with-docs -> to-spec -> to-tickets

implementation, one fresh context per approved ticket:
  GPT-5.6-sol (ultra for the high-coupling tickets) -> implement -> tdd -> code-review
```

The planning interview originally asked one dependent decision at a time. The
user later accepted the cited provenance wholesale and requested that any
genuine remaining decisions be grouped rather than re-confirming unchanged
fields. The user is not expected to know repository names, data types, or
historical route labels.

Confirmed decisions:

1. **Superseded by decision 9 below.** The initial planning assumption was that
   the first extraction target was conventional SR-SNAKE v3.1:
   `route_family=singleton_response_snake`,
   `route_profile=supported_whitened_adaptive_trust_full_response_full_accepted_refit_v3_1`,
   Phase-II/III batching disabled, and exactly one candidate-position record
   admitted per controller round. V4, experimental SR profiles, JR-SNAKE,
   FM-SNAKE, and historical routes remain behind compatibility paths during
   this first extraction. Current Paper-I source inspection later established
   that the reported rows instead use the no-prune symmetric-cost profile in
   decision 9.
2. The public SR-SNAKE interface keeps the physical calculation separate from
   the instructions for applying SR-SNAKE. The physical calculation includes
   the Hamiltonian, register layout, symmetry sector, reference state, cutoff,
   and exact-comparison definition. The SR-SNAKE instructions configure the
   method applied to that resolved physical calculation.
3. The SR-SNAKE instructions distinguish:
   - method policy: admission, Phase-III response geometry, trust solve,
     accepted refit, pruning, and scoring;
   - execution policy: optimizer, evaluation budget, seed, stopping condition,
     and maximum controller rounds;
   - observation policy: checkpoints, logs, and output locations, which must
     not alter the accepted ansatz trajectory.
4. Characterization at the public SR-SNAKE interface protects the complete
   scientific trajectory: resolved route family/profile/digest; ordered
   accepted operators and insertion positions; energy after every accepted
   controller round; Phase-III active-plus-candidate coordinates and supported
   ranks; trust-solve, admission, and full-ansatz-refit receipts; checkpointed
   accepted states; estimator-ledger closure; and `S_alg`. Post-run Qiskit
   compiled-resource costs are verified through a separate observational
   reporting seam because they do not control ansatz construction.
5. Characterization uses two complementary anchors:
   - a small deterministic Hubbard--Holstein calculation that is fast enough
     for routine tests and exercises the complete SR-SNAKE trajectory contract;
   - a read-only production provenance anchor that verifies the current
     Paper-I source artifact, settings hash, route receipt, and selected known
     values without rerunning the paper-scale calculation.
   The small calculation detects behavioral drift continuously; the production
   anchor proves that the test route remains connected to the Paper-I method
   identity.
6. Phase-live hysteresis is not part of the new controller interface.
   Conventional SR-SNAKE v3.1 keeps Phase II and Phase III live and records
   `phase_live_hysteresis_enabled=false` in its route receipt. Historical
   hysteresis controls remain quarantined in compatibility paths and cannot be
   enabled through the new interface. Intentional future variability, including
   an explicitly defined batching option, does not reopen unrelated historical
   switches.
7. Batching is an intentional conditional choice in the new architecture, not
   a reason to import every historical route switch. The conceptual decision
   tree is:

   ```text
   batching off
     -> conventional singleton admission

   batching on
     -> choose greedy or combinatorial batching
     -> choose the applicable batch/window size
   ```

   Batching-specific settings do not exist when batching is off. Internally,
   prefer a typed choice such as `batching=None`,
   `GreedyBatching(...)`, or `CombinatorialBatching(...)` over an independent
   Boolean plus nullable fields that could represent contradictory states.
   Conventional SR-SNAKE v3.1 remains the batching-off identity. Each enabled
   batching behavior must receive an explicit non-v3.1 serialized identity.
8. Optional scientific features use progressive disclosure. When a feature is
   disabled, its subordinate settings are silent: they are not requested from
   the caller, passed into the controller, or serialized as active settings.
   Enabling the feature reveals only the settings shared by that feature;
   choosing a subtype then reveals only that subtype's settings. Batching is
   the motivating example, not a request to settle every batching parameter
   during architecture planning. A disabled-feature receipt may record the
   single fact that the feature is off, but it must not emit dormant defaults
   that appear scientifically active.
9. The default and first extraction target is the current reported Paper-I
   no-prune SR-SNAKE configuration, not the code registry's unqualified v3.1
   profile:

   ```text
   route_family = singleton_response_snake
   profile_request = sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1
   route_profile = supported_projected_generalized_source_metric_no_overlap_trust_full_response_symmetric_cost_no_prune_v1
   route_contract_sha256 = fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2
   phase3_response_coordinate_scope = full_active_plus_singleton_v1
   batching = off
   beam = off
   pruning = off
   phase_live_hysteresis_enabled = false
   ```

   Issue 7 resolved the earlier stale identity against the current visible
   support JSON, frozen input bundle, executable profile resolver, and
   route-specific call path. The durable navigation authority is
   `paper-i-sr-snake-current-run-map.md`; the compact characterization receipt is
   `test/fixtures/paper_i_sr_snake_issue7_provenance_anchor.json`.
   The older code behavior selected when a caller asks for `sr_snake` without
   naming a version or variant currently resolves to v3.1 and has recoverability
   pruning enabled. It is therefore not the characterization baseline for the
   currently reported Paper-I rows. In the new interface, asking to run
   SR-SNAKE without selecting optional features means the current Paper-I
   no-prune configuration. Preserve the older pruning-enabled behavior under
   its explicit versioned identity for replay; do not let its defaults define
   the new interface.
10. Pruning is off by default. When pruning is enabled, the conditional
    configuration exposes two peer policies:
    - metric pruning, which uses the regularized local metric/response model to
      nominate a deletion;
    - trust-region pruning, which uses the full-logical trust-domain model to
      nominate a deletion.

    Both use the measured delete-and-complete-refit result as acceptance
    authority. Neither is the enabled default until the planned empirical
    comparison resolves that choice. Historical amplitude, hysteresis,
    terminal, and mixed prune modes remain compatibility-only.
11. Beam search is off by default. Enabling it starts a conditional interview
    whose suggested values are the established Paper-I beam shape:

    ```text
    live parent branches = 3
    admission children per parent = 2
    maximum admission children expanded per round = 6
    ```

    These are suggested canonical defaults, not hidden constants. The caller
    may deliberately choose another beam shape, and the serialized route receipt
    records the selected values. Beam-specific settings remain absent when beam
    search is off.
12. The new beam policy does not archive an unchanged copy of every parent when
    that parent has accepted children. That cumulative `legacy` terminal-archive
    path is archaic and remains compatibility-only. Appending a candidate at
    zero initial angle embeds the parent state in the enlarged ansatz, and the
    accepted refit must enforce non-worsening realized energy within the defined
    numerical tolerance. When a parent produces accepted children, only selected
    children continue. A branch with no admissible accepted child may terminate
    at its last accepted ansatz; this is ordinary stopping, not parallel parent
    archival.
13. Each live beam is an accepted child constructed from one of the top-ranked
    Phase-III admission records, subject to the configured children-per-parent
    and total-expansion caps. An admission record may contain one
    candidate-position record when batching is off or an explicitly constructed
    candidate batch when batching is on. After expansion and full refit, beam
    survival compares the accepted child branches directly using their measured
    energies and branch costs, not the Phase-III predicted score alone. For two
    competing branches `A` and `B`, the decision-authoritative energy difference
    is `delta_E_AB = E_A - E_B`; it is not exact ground-state error or error
    relative to a presumed correct route. The cost-weighted comparison has the
    pairwise form
    `delta_E_AB + lambda_beam * (K_A - K_B)`. The comparison:
    - prefers a branch that is no worse in measured energy and no worse in
      branch cost, and is strictly better in at least one;
    - resolves an energy-versus-cost tradeoff with the configured
      `lambda_beam` weighting;
    - retains up to the configured live-beam cap, with deterministic tie-breaks.
    Thus Phase III determines which records are worth expanding, while measured
    post-refit branch-to-branch `delta E` weighted by branch cost determines
    which expanded children survive. Unchanged parents do not enter this
    survival competition when they produced accepted children. This uses only
    quantities available to the algorithm and deliberately rejects the current
    alternate Route-A helper that scores cumulative gain from one global shared
    energy root.
14. Beam costs and continuation decisions are local to a branch fork. Suppose
    an accepted ansatz `P` splits into continuing child lineages `A` and `B`.
    The frozen state `P` is not retained as an additional competitor. Instead:
    - `A` and `B` may each continue through multiple controller rounds;
    - each lineage accumulates its total realized energy descent and its total
      cost from the common fork `P`, excluding history shared before the split;
    - the controller compares the branches by cost-adjusting those fork-local
      cumulative energy descents;
    - while neither branch has the better cost-adjusted cumulative descent,
      both lineages remain live;
    - once one lineage has the better cost-adjusted cumulative descent, subject
      only to deterministic floating-point comparison tolerance, the losing
      lineage terminates, the winner becomes the main continuation for that
      fork, and the released live-beam capacity becomes available for a later
      split;
    - the newly dominant continuation has first claim on that released
      capacity, so resolving a fork gives the winning lineage additional
      freedom to split again from one of its future top-ranked Phase-III
      admission records.

    Thus “accumulated branch cost” means cost incurred on the divergent portion
    of a lineage since its common fork, not cost over the entire calculation.
    Calling one continuation the “parent” is lineage shorthand only: it
    continues evolving and is not the unchanged parent archive prohibited by
    decision 12. Global run-control limits may still stop the complete
    calculation; they are separate from this pairwise fork-resolution rule.
    The target regime is noiseless. There is no measurement-uncertainty margin,
    waiting period, patience rule, or hysteresis control, and there is no
    separate scientific dominance threshold beyond the configured cost
    weighting itself.
15. Do not design a beam fallback for the speculative case in which Phase III
    retains only one admissible admission record while a split is requested.
    The intended beam route has multiple top-ranked Phase-III admission records
    available, so the dominant continuation can use released capacity for a
    further split. Do not add a flag, special validation branch, fallback
    policy, or dedicated diagnostic receipt for an unobserved singleton
    shortlist. Existing per-phase generator and candidate accounting already
    records how many records reach and leave each phase. If the case is ever
    observed, diagnose the concrete run from that accounting before deciding
    whether any behavior change is warranted.
16. **Superseded by decision 20 below.** The initial decision was that the cost
    used for fork-local beam dominance was only the same
    decision-authoritative algorithmic cost assigned to each admitted
    Phase-III singleton or batch. A lineage accumulates those admission costs
    only over the divergent portion after its common fork. Wall-clock time,
    estimator-work accounting such as `S_alg`, and post-run Qiskit compiled
    costs were to remain observational outputs. Decision 20 intentionally adds
    branch-local `S_alg` to beam cost while retaining the wall-clock and Qiskit
    exclusions.
17. A completed beam-enabled run returns one primary scientific result: the
    final dominant ansatz, its optimized parameters, and its accepted energy
    trajectory. It does not present the final live or terminated branches as
    equally authoritative answers. The complete fork tree remains attached as
    provenance, including bifurcation points, admitted singleton or batch
    records, fork-local accumulated decision costs, branch energies,
    dominance decisions, terminated lineages, discarded estimator work, and
    released-capacity reuse. The primary result stays structurally comparable
    to the ordinary non-beam SR-SNAKE result.
18. Batching, beam search, and pruning are independently composable optional
    policies. Their responsibilities remain separate:
    - batching constructs and admits either singleton or multi-record
      Phase-III proposals;
    - beam search manages competing accepted continuations and fork-local
      dominance;
    - pruning verifies and removes recoverably redundant coordinates within a
      continuation.

    The default Paper-I profile keeps all three off. Enabling any combination
    requires only that combination's progressive-disclosure settings, and the
    normalized route/profile identity and receipt explicitly record the active
    composition. The new interface must not force callers through historical
    named-route bundles merely to combine these policies.
19. When pruning and beam search are both enabled, one branch transition is
    ordered as follows: admit the selected singleton or batch, perform the
    accepted full-ansatz refit, perform any enabled recoverability-pruning
    verification and its deletion refit, and only then enter beam-dominance
    comparison. Beam survival therefore evaluates the actual accepted
    post-pruning branch that can continue, not an intermediate ansatz containing
    coordinates already selected for removal.
20. An explicitly enabled beam policy incorporates realized `S_alg` into its
    fork-local branch cost. At each bifurcation, the controller records the
    authoritative estimator-ledger prefix. Each continuing lineage then
    accumulates the unique estimator primitives it requires after that fork,
    across outer-energy, refit-energy, gradient, and metric components. Shared
    pre-fork work is excluded because it is common history. A primitive consumed
    by more than one post-fork lineage is treated symmetrically for branch
    comparison rather than charged according to execution order; run-wide
    all-work `S_alg` still deduplicates that primitive exactly once. Work spent
    on a terminated lineage remains in all-work and discarded-branch
    accounting, but it is not reassigned to the winning lineage. Wall-clock
    duration and post-run Qiskit compiled costs remain outside beam decisions.
    The existing Phase-III algorithmic cost remains a predictive cost used only
    to rank singleton or batch admission proposals before expansion. After
    expansion, realized fork-local `S_alg` is the beam-dominance cost. Beam
    dominance uses a weighted additive branch-to-branch comparison: cumulative
    energy difference since the bifurcation and cumulative fork-local `S_alg`
    difference enter on equal conceptual footing, with every unique estimator
    primitive receiving the same configured `S_alg` weight. The suggested
    beam-enabled default weight is `0.01`. That value is explicitly
    **uncalibrated** and must not be described as an established Paper-I
    numerical constant. Its configuration contract, code comment, normalized
    receipt, and any agent handoff or run interpretation must surface
    `calibration_status=uncalibrated_default` and tell the user that calibration
    remains outstanding. This planning decision does not authorize a
    calibration run. The comparison does not rank
    branches by an energy-gain-per-`S_alg` ratio. The
    predictive Phase-III cost and realized `S_alg` are not added into one beam
    term, avoiding double penalization of the same prospective burden. If
    pruning later removes an admitted coordinate, the estimator work already
    incurred remains in fork-local and all-work `S_alg`.
21. Ticket granularity, dependency edges, prefactoring order, and other
    implementation-management choices are delegated to the planning agent.
    The user is not expected to assess unfamiliar code-organization tickets.
    After the broader refactor plan is complete, the agent may approve and
    publish the tracer-bullet breakdown using engineering judgment. Continue to
    ask the user only about unresolved scientific semantics or user-visible
    request/result behavior.
22. The new controller may optionally stop against a predefined exact-ED
    energy. This is an explicit execution/run-control choice, not part of the
    default SR-SNAKE method policy. When disabled, no exact target or dormant
    exact-stop tolerance appears in the active request. When enabled, the caller
    supplies:
    - the exact-ED reference energy;
    - the accepted absolute-energy tolerance;
    - source identity proving that the ED reference uses the same physical
      problem and cutoff as the SR-SNAKE run.

    The controller checks the target only against an accepted post-refit state.
    If the target is reached, the primary result records the exact-target stop
    reason, accepted energy, observed absolute difference, configured
    tolerance, and ED source identity. Exact ED remains unavailable to
    Phase-I/II/III candidate construction, trust solving, admission ranking,
    pruning, beam dominance, and optimizer steps. Thus the explicit stop may
    use the predefined answer as a benchmark termination condition without
    allowing that answer to steer the constructed ansatz.
23. Stopping uses one typed policy rather than independent historical flags.
    The new interface defaults to 50 controller rounds. A caller may:
    - replace that default with another explicit maximum number of controller
      rounds;
    - add the optional exact-ED target from decision 22.

    Active conditions compose with first-hit semantics. The maximum-round cap
    remains active alongside an exact-ED target; exact-target configuration
    does not create an unlimited run. One controller round is one full
    selection, admission, and accepted-refit cycle, even when batching admits
    multiple generators during that cycle. If more than one stop condition is
    satisfied on the same accepted transition, the receipt records every
    satisfied condition and uses exact-ED target reached as the primary reason,
    followed by maximum controller rounds as the deterministic precedence.
    Pool exhaustion or loss of an admissible candidate is a terminal controller
    outcome rather than a caller-configured stop condition.
24. The user delegated implementation-ticket granularity, dependency edges,
    prefactoring order, and approval to the planning agent. The Matt Peacock
    tracer-bullet sequence is approved and published as GitHub issues 7 through
    20. Only the current dependency frontier receives `ready-for-agent`; a
    blocked but approved ticket is not mislabeled as executable. The parent
    specification is issue 6 and is not itself an implementation unit.
25. “Source-locked horizon” is a paper-provenance requirement, not a concept
    ordinary users should need in the simpler interface. The current completion
    study records 50 rounds, so the public stop policy adopts 50 as its ordinary
    default. A paper replay adapter may still verify that its requested value
    matches its immutable source record; that verification does not change the
    public default or add another stop-policy variant.
26. A throwaway logic prototype compared a flat settings object, a stateful
    builder, and a nested immutable request. The nested request is approved:
    - the public package has one run operation;
    - `request=None` means singleton, no pruning, no beam, 50 rounds, no exact
      target, fresh start, and default observation;
    - the request contains only method, execution, and observation choices;
    - method choices are admission, pruning, and beam;
    - execution choices are stopping and typed resume;
    - route profile, Phase-III geometry, trust solve, accepted refit, optimizer,
      seed, numerical guards, liveness, and hysteresis behavior are internal
      current-profile semantics;
    - the runner derives route/profile/digest receipts from the active policy
      composition;
    - controller state, candidate-position records, admission decisions, and
      accepted transitions are internal types, not public extension points.

    This decision supersedes decision 3 only where decision 3 proposed exposing
    optimizer, evaluation budget, and seed as ordinary public execution
    settings. The method/execution/observation conceptual separation remains.
    The prototype was rerun after approval and then deleted; it is design
    evidence, not implementation scaffolding or a second source of defaults.
27. Keep one fresh implementation context per ticket, but reserve
    GPT-5.6-sol-ultra for the high-coupling extraction and scientific-control
    tickets: selection, accepted transition, controller-loop migration,
    fork-local accounting, beam survival, combined resume/composition, and the
    final contraction audit. Characterization, facade, resolved-context
    prefactoring, caller migration, pruning, and the two batching policies may
    use ordinary GPT-5.6-sol reasoning unless concrete complexity discovered in
    the ticket justifies escalation.
28. Add an opt-in plateau-triggered, commutation-reduced insertion experiment
    before continuing the numbered refactor sequence. The ordinary Paper-I
    default remains append-only. The experimental policy has these semantics:
    - the first controller round and every round following adequate accepted
      progress use append-only candidate-position construction;
    - plateau detection uses only the realized energy decrease between the
      previous accepted state and the new accepted post-refit state; exact-ED
      energy or exact energy error cannot control the policy;
    - one below-threshold accepted transition opens the full logical insertion
      domain on the next selection round, with no patience, hysteresis,
      gradient-flatness, repeated-family, or historical escape trigger;
    - for each candidate generator, exactly certified commuting-equivalent
      insertion positions collapse to one canonical representative before
      Phase-I/II/III ranking;
    - the widened domain remains open while the immediately preceding accepted
      energy decrease is below threshold and closes automatically once adequate
      accepted progress resumes;
    - opening the domain does not admit a candidate by itself and does not
      change Phase-I/II/III scores, trust, accepted refitting, pruning,
      batching, beam behavior, or stopping;
    - the experimental route uses a provisional uncalibrated energy-decrease
      threshold of `1e-8` and records that threshold and calibration status in
      its normalized receipt.

    The immediate implementation is a one-factor runnable route layered on the
    active no-prune Paper-I profile. It may reuse the existing legacy executor
    so runs can begin before controller extraction. The new deep controller
    later owns this as an explicit insertion policy at the candidate-domain
    seam; it must not import the legacy `adaptive` union of plateau, flatness,
    repeated-family, or escape triggers. The public request retains exactly
    three top-level fields; eventual typed exposure belongs under
    `method.insertion`.
29. Defer the newly requested Paper-I evidence program until the SR-SNAKE
    cleanup is complete. No implementation agent should stage, launch,
    aggregate, or report the macro/Pauli-child comparison, operator-family
    heatmap, additional no-Phase-III ablations, whitening comparison, `L=3`
    append baseline, batching results, or pruning results while Issues 9--20
    remain incomplete. The durable deferred queue is
    `agent_guidance/static-adapt/post-refactor-paper-i-evidence-queue.md`.
    Completion of the refactor does not itself authorize a run: each queued
    study must then pass the current Paper-I run gates, resolve its visible or
    explicit reader-facing target and source-locked baseline, and receive the
    user's launch authorization.
30. After Issues 17--20 complete, promote
    `insertion_commutation_plateau_v1` from its opt-in experimental identity to
    the canonical Paper-I insertion policy. This is a future behavior-changing
    implementation slice, not part of Issues 17--20. It must update the typed
    request resolver, route identity and digest, canonical settings contract,
    and public behavior tests together. Historical append-only results retain
    their original identities, and append-only insertion remains available only
    as an explicitly requested ablation or replay policy. Until that dedicated
    slice is implemented and verified, decision 28 continues to describe
    current executable behavior while this decision records the target
    interface.

### Approved implementation tickets

This table is the completion ledger for the staged implementation inventory.
The later canonical-interface handoff superseded the one-ticket-at-a-time
execution instruction without changing the tracer-bullet outcomes.

| Issue | Tracer-bullet outcome | Blocked by |
|---:|---|---|
| 7 | Characterize the active no-prune trajectory and provenance | Complete |
| 8 | Add the typed request/result facade over the legacy executor | Complete |
| 9 | Prefactor one private resolved execution context | Complete |
| 10 | Extract current Phase-I/II/III selection | Complete |
| 11 | Extract singleton admission and accepted full refit | Complete |
| 12 | Move the default no-prune loop behind the facade | Complete |
| 13 | Migrate canonical callers and observational adapters | Complete |
| 14 | Add optional recoverability pruning | Complete |
| 15 | Add optional greedy batching | Complete |
| 16 | Add optional combinatorial batching | Complete |
| 17 | Add fork-local estimator-accounting views | Complete |
| 18 | Implement persistent fork-local beam survival | Complete |
| 19 | Compose beam, batching, pruning, and checkpoint resume | Complete |
| 20 | Quarantine legacy routes and complete contraction audit | Complete |

Issue 20 does not authorize deletion. Historical code or readers may be deleted
only after its reachability finding and separate explicit user approval.

Current code evidence affecting the design:

- `paper_i_runner.py` already exposes a typed Route-A facade, but its canonical
  path includes joint/batch-oriented mechanisms and is not the conventional
  no-prune SR-SNAKE singleton interface. Its `PaperISnakeRunConfig` also
  requires batching, beam, shortlist, child-funnel, and other values even when
  those mechanisms are inactive. Reuse its resolved-problem and typed
  run-control patterns; do not relabel or wrap that facade as the new SR-SNAKE
  entry point.
- `ResolvedProblemContext` is the existing resolved physical-calculation type.
- The proposed names `ResolvedPoolPlan` and `EstimatorExecutor` do not currently
  exist. Do not create public seams for them in the first slice. Pool
  resolution and estimator execution stay internal until concrete variation
  justifies a seam.
- The current no-prune route is materialized by
  `normalize_sr_route_profile_namespace` and then passed as a complete contract
  and a large keyword set into `_run_hardcoded_adapt_vqe`. Its ordinary
  no-beam loop begins at approximately line 40020 of `adapt_pipeline.py` and
  runs through approximately line 53240.
- One ordinary round in that loop already has the conceptual sequence needed
  by the new controller: refresh accepted state and energy; acquire the
  gradient surface; build and rank Phase-I/II/III records; check internally
  available convergence; admit the selected record; run the accepted refit;
  perform optional prune work; append history and checkpoint data; then check
  energy convergence.
- The loop's nested helpers and branches mix the current no-prune path with
  Route-A funnels, formal-manifold controllers and overlays, noise/oracle
  paths, plateau and escape modes, pruning, beam execution, and historical
  fallbacks. The extraction target is the current canonical no-prune path, not
  a move of that entire union into a new file.
- Existing modules already own substantial numerical and provenance work:
  `phase_shortlists.py`, `joint_linear_solve.py`, `accepted_refit.py`,
  `estimator_call_ledger.py`, `checkpoint_telemetry.py`, and
  `output_artifacts.py`. The architecture problem is orchestration and intent
  diffusion, not a lack of helper modules.
- `benchmark_target_abs_delta_e` currently allows the controller to compare its
  accepted energy to an exact/reference answer and terminate when the requested
  error is reached. The new interface preserves this capability only as the
  explicit exact-ED stopping policy in decision 22, with same-cutoff provenance
  and accepted-state-only evaluation.
- The implementation baseline is the current working tree, which contains
  substantial uncommitted SR-SNAKE changes. Characterization must protect that
  exact state; implementation must not assume repository `HEAD` is the current
  scientific baseline.

Planning coverage:

- Settled: current Paper-I default identity; physical-problem/request
  separation; prototype-validated nested immutable request; progressive
  disclosure; batching, pruning, hysteresis, and beam user-visible semantics;
  primary result and branch provenance; one deep two-argument public interface;
  internal selection and transition seams; output/checkpoint call direction;
  legacy adapter direction; the expand--migrate--contract extraction order;
  default 50-round stopping plus an optional same-cutoff exact-ED condition;
  typed public request/result receipts and private controller types;
  characterization layers; and implementation-ticket boundaries.
- No unresolved user-level scientific question remains in the planning
  contract. Ticket 7 established the small route-faithful characterization
  fixture and compact immutable provenance receipt. The default suite is
  repository-contained; validation against ignored or fetched production
  sources is a separate explicit opt-in audit.
- GitHub specification issue 6 is approved as the parent specification but is
  not agent-grabbable as one implementation unit. Issues 7 through 20 are the
  approved execution sequence.

Issue 12 completion evidence:

- The exact default `run_sr_snake` path constructs a direct numerical session;
  it does not call either the compatibility adapter or the legacy
  `_run_hardcoded_adapt_vqe` loop.
- `_controller.py` solely owns accepted-state iteration, selection/transition
  ordering, checkpoint-ready event projection, next-state publication, stop
  evaluation, and typed finalization.
- Checkpoint, replay, finalization, and runtime-factory boundaries use concrete
  copy-isolated private records/protocols. Runner mapping export is confined to
  observation serialization; no generator/program controller remains.
- The exact two-round fixture preserves operators, energies, Phase-III
  coordinates/ranks, refit/trust receipts, checkpoint replay and hashes,
  `S_alg=[299,709]`, `S_unique=[250,564]`, and all 709 ordered estimator
  occurrences.
- The Issue-7--12 regression aggregate passed with 288 tests and one opt-in
  provenance-source audit skipped. The resolved-context suite passed 12 tests,
  and the historical formal-manifold compatibility plus legacy-loop sentinel
  passed two tests. `py_compile`, path-limited diff hygiene, import/type and
  copy-isolation checks passed. Independent Spec and Standards reviews returned
  no findings.
- No production 21/50-round scientific rerun was performed, and the external
  provenance-source audit remains explicit opt-in work; neither is Issue-12
  implementation evidence.

Issue 13 completion evidence:

- The normalized exact active CLI route calls the public `run_sr_snake`
  operation. Profile/digest drift and all currently reachable noncanonical
  stop, noise, parity, worker, resume, pruning, batching, beam, FM/JR, and
  experimental controls retain the named legacy compatibility route with an
  explicit reason receipt.
- Accepted checkpoints are published at the configured cadence as a
  consumer-complete `adapt_vqe` envelope plus a content-addressed authenticated
  estimator-ledger sidecar and a content-addressed v2 verified-singleton resume
  sidecar. The current envelope authenticates the v2 sidecar bytes; the sidecar
  binds a canonical source projection with only its pointer omitted, avoiding a
  digest cycle, and records a resolved source path so relative public-current
  paths round-trip through the reader. Sidecar bytes are durably written before
  the atomic
  current-pointer replacement, and every atomic replacement fsyncs its parent
  directory. Fixed-name v1 resume sidecars remain an explicit legacy
  compatibility branch, but the active `fd5ec3fa...` envelope identity
  independently requires v2 and rejects pointer-removal downgrade. A private
  temporary checkpoint provides the same authenticated observation channel
  when no public current path is requested.
- The unchanged public `extract_verified_singleton_resume_checkpoint`
  entrypoint consumes both requested history-tail `1` and history-tail `0`
  envelopes. Its sidecar helpers now authenticate the v2 pointer/projection
  proof while retaining the explicit legacy-v1 branch. Tail requests retain an
  explicit retention receipt while the authoritative current envelope
  normalizes both `history` and `history_tail` to the complete serialized
  lineage required by that reader. A forced post-publication interruption
  after the exact one-round horizon preserved the round-1 operator/checkpoint
  hashes and restored `S_alg=299`, `S_unique=250`, and all 299 occurrence
  records. Tampering the resume-sidecar bytes, source-projection hash, or
  estimator-ledger sidecar bytes fails closed, as does deleting the active v2
  pointer while presenting a retained fixed-name v1 sidecar.
- The two-round projection preserves the characterized operators, parameters,
  energies, checkpoint hashes, route digest, and cumulative accounting
  `S_alg=[299,709]`, `S_unique=[250,564]`. Public observation receipts
  authenticate the final current bytes, and selected-prefix/recovery-prefix
  readers leave result and ledger bytes unchanged.
- The immutable Paper-I replay lock and reduced fresh CLI segment lock remain
  distinct validators and receipts. The reduced segment projection does not
  claim a terminal-Qiskit round or exact source-locked horizon.
- Compatibility is consumer-complete, not the historical executor's full
  diagnostic union. The unsegmented adapter projection retains 51 of 224
  observed historical top-level keys; the outer CLI postprocessor adds the
  generic `boson_subspace_diagnostics` field, so serialized `result.json`
  contains 52. A segment lock adds only `adapt_segment`, yielding 52 at the
  adapter layer and 53 after the generic postprocessor. The unsegmented
  projection retains history 35 of 264, continuation 6 of 92, and accounting
  17 of 21; segment enrichment yields history 39 of 264, continuation 7 of 92,
  and accounting 17 of 21. Reachability and retirement of omitted
  diagnostic-only fields remain explicit Issue-20 debt.
- The July-18 six-regime bundle remains a named historical
  whitened/stale-identity compatibility fixture. It was not edited, executed,
  or reclassified as the active `fd5ec3fa...` route.
- Post-repair verification passed: SR-SNAKE `168 passed, 1 skipped`; affected
  consumer/caller/resume/Qiskit `162 passed, 9 skipped`; CLI/control/output/
  checkpoint `88 passed`; route-profile `238 passed`; exact Issue-13 route
  `5 passed`; retained Powell/Pareto caller `50 passed`.
- Final post-v2 independent review closed with zero findings: Spec `PASS` and
  Standards `PASS`. Both reviews reproduced the authenticated v2
  pointer/projection path, active-route v1 downgrade rejection, relative
  public-current path round trip, tamper failures, exact accounting, layered
  compatibility counts, July-18 compatibility status, and the Issue-14
  boundary.
- No production scientific run, manuscript edit, commit, push, issue-label
  change, or Issue-14 implementation was performed.

Issue 14 completion evidence:

- `RecoverabilityPruning()` resolves an exact child of the active
  `fd5ec3fa...` singleton route with contract digest
  `44f9ef70c114e88efd4ff9c3fb1c64abc7d7a25c15a978bbe735243ac1dd27de`.
  Public context, direct runtime, and normalized CLI agree on the complete
  fixed prune setting set and fail closed on every tested single-field drift,
  parent-plus-pruning, or child-with-pruning-disabled combination.
- The accepted transition now owns recoverability nomination and at most one
  measured full-survivor sibling per round after the supported-FS Powell
  refit. Measured delete/refit energy remains the decision authority; the
  surrogate only nominates.
- The natural two-site Hubbard--Holstein fixture reports honest no-nominee
  states through round 3 and one rejected round-5 sibling. The rejection
  preserves the keep fingerprint, contracts radius `0.125 -> 0.0625`, charges
  zero overlap, keeps damping zero, and contributes exactly `103` to all-work
  but not winning-lineage `S_alg`. Round 6 starts from `0.0625`, proving
  continuation-state persistence.
- Controlled transition and public-run acceptance fixtures prove the actual
  reduced operator/parameter state reaches the accepted transition, public
  final result, and terminal checkpoint. The small untuned natural fixture did
  not produce an accepted deletion; that explicit fixture limitation did not
  justify changing production acceptance semantics.
- Public and internal typed prune receipts enforce status-dependent
  acceptance, classification, deletion/remap, trust, work, and immutable
  rejected/no-trial state invariants. Pruning-off output retains its exact
  progressive-disclosure shape.
- Final verification passed: affected SR-SNAKE/controller/route suites
  `230 passed`; estimator-ledger, `S_alg`, pruning-ladder, and trust-prune
  suites `84 passed`. Fresh Spec and Standards reviews both returned `PASS`
  after independently reproducing the repaired exact-route and receipt gates.
- No scientific run, evidence promotion, manuscript edit, commit, push,
  issue-label change, Issue-15 implementation, or other external mutation was
  performed.

Issue 15 completion evidence:

- `GreedyBatchAdmission(maximum_size=3, search_window_size=None)` is reachable
  as a distinct request-specific child of the active `fd5ec3fa...` route.
  `maximum_size` bounds actual admission at the reduced-plane kernel ceiling of
  five; the optional search window is a ranked Phase-III prefix, and `None`
  retains the full ranked population. The historical `0.9` score shell is
  inactive.
- Selection observes actual joint-pair kernel results and records exact
  physical/cache and live-ledger accounting without recomputing pair geometry.
  Transition commits the immutable ordered batch at zero amplitude and owns
  one full supported-FS Powell refit, one trust update, one ledger closure, one
  checkpoint event, and one controller-round increment.
- Public batch transition and replay receipts are plural and do not overload
  singleton scalar fields. A one-member greedy fallback retains the greedy
  route identity. The default singleton request, result shape, trajectory,
  hashes, and verified-resume sidecar remain unchanged.
- Normalized CLI dispatch reconstructs dynamic maximum/window controls from
  the exact canonical contract. Any drift carrying explicit greedy intent
  fails closed before the legacy executor; unrelated historical profiles keep
  the named compatibility path.
- Greedy accepted checkpoints publish a distinct authenticated,
  content-addressed projection sidecar and explicitly deny reconstruction
  until Issue 19. Combinatorial admission and batching/pruning/beam/resume
  composition remain unreachable.
- Final verification passed: focused Issue-15 facade/controller `70 passed`;
  route-profile `155 passed`; exact no-prune/prune guards `17 passed`; affected
  selector/order/ledger/checkpoint/no-batch `71 passed`. `py_compile` and
  path-limited diff checks passed. Fresh post-repair Spec and Standards reviews
  both returned clean.
- The affected aggregate also exposed and repaired one nonsemantic shared
  terminal-Phase-I initialization defect, recorded in
  `sr-snake-issue-15-handoff.md`.
- No scientific run, evidence promotion, manuscript edit, commit, push, issue
  mutation, Issue-16 implementation, or other external action was performed.

Issue 16 completion evidence:

- `CombinatorialBatchAdmission()` resolves the approved bounded default
  `maximum_size=3, search_window_size=6`; an omitted window follows
  `min(2 * maximum_size, 10)`, a positive integer is an explicit bounded
  window, and `FullCombinatorialSearchWindow()` is the explicit full-population
  choice. Public `0` is invalid.
- The distinct `combinatorial_batch_response_snake` child enumerates exhaustive
  generator-distinct subsets of fixed generator-plus-insertion-position
  Phase-III records, not permutations or alternative insertion placements.
  Subsets use coupled joint response/Gram/Hessian descent with symmetric cost
  and additivity gating off.
- One shared pair-geometry workspace reconciles physical cache misses and exact
  metric/Hessian ledger occurrences without multiplying work per classical
  subset. Exact considered-subset counts are validated for every cardinality.
- One selected subset remains one zero-angle atomic admission, one complete
  supported-FS Powell refit, one trust update, one ledger closure, one
  authenticated checkpoint, and one controller-round increment. A singleton
  winner retains combinatorial identity.
- Public proposal, transition, replay, CLI, and projection-only checkpoint
  contracts are combinatorial-specific. Resume and policy composition remain
  fail-closed until Issue 19; explicit combinatorial CLI drift cannot fall
  through to legacy execution.
- Recorded pre-closure validation passed core `85`, helper/accounting `8`,
  route/configuration `152`, full continuation-scoring `162`, and default-route
  guard `26` test groups. After the final diagnostic/boundary fixes, a direct
  narrow closure command passed `13` focused tests, including the frozen
  no-prune trajectory. The implementation handoff is
  `sr-snake-issue-16-handoff.md`.
- No scientific run, evidence promotion, manuscript edit, commit, push, issue
  mutation, Issue-17 implementation, or external action was performed.

2026-07-25
Observation:
Issues 7--16 and the insertion experiment are complete. The Paper-I
progressive-disclosure plan is now settled, and the user chose to continue the
SR-SNAKE cleanup before the ICM/GitNexus pilot or deferred evidence runs.
Evidence:
The ledger already owns physical primitive identity, ordered occurrences,
branch consumers, global component charging, explicit primitive-set summaries,
and closed-prefix summaries. The typed facade already exposes all-work and
winning-lineage accounting without beam behavior.
Decision or open question:
Proceed with Issue 17 only. Add fork-local post-prefix estimator-accounting
views at the existing ledger seam; do not implement beam survival or apply the
uncalibrated beam weight yet. No scientific question remains open for this
slice.
Files/symbols:
`pipelines/static_adapt/estimator_call_ledger.py`,
`pipelines/static_adapt/sr_snake/contracts.py`, the smallest necessary
controller/runner projection, and focused ledger/facade tests.
Next safe action:
Give `agent_guidance/static-adapt/sr-snake-issue-17-handoff.md` to the adjacent
implementation agent. Stop again after Issue 17 validation and two-axis review;
Issue 18 is not authorized by this handoff.

2026-07-25
Observation:
The user accepted the July-23 macro/common-accuracy provenance as the complete
canonical default baseline and rejected further field-by-field confirmation.
The high-level Paper-I interface, compatibility quarantine, automatic
run-summary behavior, and explicit deviations now form one high-coupling
refactor target.
Evidence:
The provenance resolves unfiltered `full_meta` with HVA included, physical
macro lanes followed by hard-guarded singleton Pauli children, the no-overlap
trust profile, and whitened complete accepted refits. The governing
`paper-lane-refactor-plan.md` records the explicit deviations and progressive
disclosure contract.
Decision or open question:
The preceding Issue-17-only sequencing note is superseded. Issues 17--20 remain
implementation inventory, but they do not define the public architecture or
require separate user interviews. No scientific question remains open.
Files/symbols:
Planning Markdown only in the current phase. Production scope will be enumerated
in one implementation handoff after the plan is reconciled.
Next safe action (historical; completed on 2026-07-26):
Execute the superseding canonical-interface handoff with internally staged
implementation, route-faithful tests, path-limited reachability verification,
and independent Standards and Spec review. Do not reuse this historical model
allocation as a future instruction.

2026-07-26
Observation:
The superseding canonical-interface handoff and Issues 17--20 are complete.
The ordinary `run_sr_snake(problem, request=None)` path resolves typed,
parser-free runtime inputs and enters the direct controller. Compatibility
projection remains available only through an explicit adapter; no historical
implementation was deleted.
Evidence:
The accepted interface includes plateau-triggered commutation insertion,
singleton/greedy/combinatorial admission, peer metric/trust pruning, optional
fork-local beam, authenticated resume, exact accepted-state stopping, closed
four-component `S_alg`, and post-finalization Paper-I summaries. The repaired
acceptance aggregate passed `500` tests with one opt-in provenance audit
skipped. Fresh independent Spec and Standards re-reviews returned zero
findings. The append registry digest and isolated report layout/provenance
checks also closed.
Decision or open question:
The deep-module cleanup gate is satisfied. This completion does not authorize
a scientific run, evidence replacement, manuscript edit, compatibility
deletion, commit, push, or issue mutation.
Files/symbols:
`pipelines/static_adapt/sr_snake/`,
`pipelines/static_adapt/adapt_pipeline.py`,
`pipelines/reporting/paper_i_run_summary.py`, the canonical append registry,
the Powell support-PDF builder, lane policy/reporting guidance, and focused
tests.
Next safe action:
Design the minimal Paper-I campaign launcher and use that thin workflow as the
ICM/GitNexus pilot before requesting a specific deferred evidence study.

```text
YYYY-MM-DD
Observation:
Evidence:
Decision or open question:
Files/symbols:
Next safe action:
```

Historical implementation inventory (not current authorization):

- `test/`: the smallest route-faithful complete-run characterization fixture
  selected by issue 7, followed by public-seam and policy-composition tests;
- `pipelines/static_adapt/sr_snake/__init__.py`: intentional public exports;
- `pipelines/static_adapt/sr_snake/contracts.py`: public request, policy,
  stopping, resume, observation, result, and receipt types;
- `pipelines/static_adapt/sr_snake/runner.py`: `run_sr_snake`;
- `pipelines/static_adapt/sr_snake/_legacy_adapter.py`: temporary translation
  to the characterized executor;
- `pipelines/static_adapt/sr_snake/_controller.py`,
  `pipelines/static_adapt/sr_snake/_selection.py`, and
  `pipelines/static_adapt/sr_snake/_transition.py`: only in the extraction
  tickets where each deep internal module becomes earned;
- `pipelines/static_adapt/adapt_pipeline.py`:
  `_run_hardcoded_adapt_vqe` and the current no-prune controller region, reduced
  incrementally as ownership moves behind the new seam;
- `pipelines/static_adapt/paper_i_runner.py` and other callers identified by
  the issue-13 impact audit: canonical request construction and compatibility
  translation;
- the current machine-readable Paper-I route/default contract identified
  during characterization: synchronize the approved 50-round default and typed
  optional policy identities without changing the physical problem;
- `agent_guidance/static-adapt/CONTEXT.md` and a future ADR only if
  implementation evidence changes the approved vocabulary or deep-module seam.
