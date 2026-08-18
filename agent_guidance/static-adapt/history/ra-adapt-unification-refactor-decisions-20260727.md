# RA-ADAPT Unification, Repair, and Legacy-Contraction Decisions

Status: live agent-facing decision record, 2026-07-27.

This file records the user-approved scope for the Paper-I RA-ADAPT
implementation plan. It is an input to the Claude Fable planning handoff, the
original audit agent's review, and a later GPT-5.6-ultra implementation.

This file does not itself launch scientific runs, mutate the manuscript,
replace evidence, or authorize an agent to judge scientific results. Agents
must ask the user whenever a choice changes scientific semantics, run scope, or
result interpretation. Routine implementation, repair, testing, bundle
materialization, and CHTC progression through already locked objective gates do
not require repeated questions.

## Objective

Replace the historically divergent macro and single-Pauli-word RA-ADAPT
implementations with one deep Paper-I RA-ADAPT engine. The refactor is the
mechanism for performing the scientific alignment; agents must not first patch
both historical paths and then repeat the work in a second refactor.

The target sequence is:

```text
lock scientific contracts and characterization evidence
  -> refactor onto one shared engine
  -> verify mathematical and route contracts
  -> materialize finite source-locked run bundles
  -> run staged CHTC studies
  -> present objective results
  -> user selects scientific winner
```

Paper-IV generalization is explicitly out of scope. The design must not block a
future problem-neutral static-construction seam, but it must not activate
molecular-vibronic physics, pools, or validation in this Paper-I repair.

## Agent roles

1. **GPT-5.6 xhigh preflight**: completed a read-only source/reachability audit.
   It did not run GitNexus because the required strict ignore surface is absent.
2. **Claude Fable**: read-only architecture planner. It will inspect the code
   and produce the detailed implementation specification; it will not implement
   or launch science.
3. **Original audit agent**: reviews Claude's plan for scientific and
   provenance fidelity before implementation. It does not judge whether new
   numerical results are desirable.
4. **GPT-5.6 ultra**: implements the reviewed plan, tests the refactor, and
   materializes the run bundles.
5. **GPT-5.6 high**: executes the staged CHTC campaign from the validated
   bundles. It may proceed from the small validation matrix to the full macro
   repair matrix when objective implementation/provenance gates pass.
6. **User**: judges numerical results, selects the stationarity-policy winner,
   decides manuscript/evidence replacement, and answers material questions
   throughout the workflow.

## Locked domain distinctions

### Method and representations

**RA-ADAPT** is the canonical method name. New canonical code should use an
interface such as:

```python
result = run_ra_adapt(problem, request=None)
```

SNAKE, SR-SNAKE, FM-SNAKE, and similar names are not new canonical interface
terms. Historical artifacts retain their source-native labels.

**Parent template pool** is the full `full_meta` parent source:

```text
nph = 3: 123 parents
nph = 7: 171 parents
```

It may contain parents that cannot be admitted through the current
termwise-product macro execution but can still produce individually guarded
single-Pauli-word candidates.

**Executable macro pool** is the current fixed-sector-safe subset:

```text
nph = 3: 102 macro candidates
nph = 7: 148 macro candidates
```

The replacement macro RA-ADAPT and macro Append-ADAPT comparisons use this same
102/148 pool. A preflight suggestion to execute the excluded 21/23 parents via
`grouped_exact` is not adopted by this decision record. That would be a
separate scientific change and requires an explicit later user decision.

**Macro-generator candidate** is a directly executable member of the
sector-safe macro pool.

**Single-Pauli-word candidate** is a canonical unit Pauli direction obtained by
splitting a parent expansion and then applying the required symmetry guards and
deduplication. Do not call it a projected singleton or an intact generator.

Singleton RA-ADAPT and singleton Append-ADAPT use the same initial 123/171
parent source. Their intended divergence begins only at child exposure:

- RA-ADAPT shortlists parents before constructing children.
- Append-ADAPT constructs the global eligible child pool before gradient
  selection.

### Protocol and bundle terms

Use:

- **RA-ADAPT request** for user-authorized choices;
- **resolved RA-ADAPT protocol** for immutable scientific settings plus digest;
- **execution manifest** for what actually ran;
- **run bundle** for a finite, source-locked collection of resolved protocols
  and expected artifacts for one study.

A run bundle is not a reusable campaign framework, generic six-regime
launcher, or second source of scientific defaults.

The two active-gradient policies are:

- **stationary-source response**: impose
  \(\mathbf g_\theta=0\) and do not acquire the active-gradient vector;
- **measured-residual response**: acquire and use the residual
  \(\mathbf g_\theta\).

The two cost-scope policies are:

- **late resource weighting**: Phase I energy-only; Phases II and III energy
  per cost;
- **all-phase resource weighting**: Phases I, II, and III energy per cost.

## Shared-engine contract

One deep RA-ADAPT engine owns:

- Phase-I/II/III orchestration;
- parent retention and representation-adapter transitions;
- candidate-position enumeration;
- actual-position candidate geometry;
- Fubini--Study support construction;
- Phase-III solve;
- trust calibration and update;
- accepted-proposal refit and whitening;
- estimator/resource accounting;
- typed receipts, manifests, and results.

Thin representation adapters own only representation-specific behavior:

- `MacroCandidateAdapter`: exposes the executable 102/148 macro pool.
- `SinglePauliWordCandidateAdapter`: starts from the 123/171 parent source and
  constructs guarded canonical children only after parent retention.

Append-ADAPT shares pool definitions, representation adapters, candidate
execution, compilation, accounting, and scientifically common refit
infrastructure. Its conventional selector remains separate; Append-ADAPT is not
forced through RA-ADAPT's staged P1/P2/P3 funnel.

Append-ADAPT receives its own small typed facade, conceptually:

```python
result = run_append_adapt(problem, request=None)
```

After this facade owns the retained Append comparison, contract the old
multipurpose generic comparator implementation and archive its author-retired
TETRIS/disjoint branches.

The implementation should use a strangler-style migration:

1. lock characterization and mathematical contracts;
2. extract the shared engine and neutral kernels;
3. migrate canonical macro, singleton, and Append callers;
4. verify route and accounting behavior;
5. prove obsolete branches unreachable;
6. archive only the author-retired route-specific code.

Do not perform a big-bang monolith rewrite.

## Locked scientific alignment

### Insertion geometry

For an insertion at position \(p\), the modeled circuit is the actual enlarged
ordered circuit:

\[
U_{k,p}(\alpha)
=
U_{k,>p}e^{-i\alpha A}U_{k,\le p}.
\]

Phase-II/III macro response geometry must use the position-dependent dressed
candidate, not the historical append-after-current-ansatz chart. The successful
single-Pauli-word commutation-aware insertion route is preservation-critical.

### Trust and support

- Macro and single-Pauli-word adapters use one source-Gram, no-endpoint-overlap
  trust transaction.
- Phase-III selector coordinates and accepted-refit whitening are distinct
  stages and must remain distinct.
- One neutral retained-support factorization should own support thresholds,
  eigenpairs, tolerances, and receipts while exposing the coordinate view each
  stage requires.
- Historical regularized selector whitening remains outside the canonical
  engine.
- Curvature stabilization and trust-boundary enforcement retain distinct
  quantities and receipts.

### Accepted refit

The complete enlarged ansatz is optimized in a fixed supported
Fubini--Study-whitened chart for one accepted refit. The chart is discarded
afterward. Macro and single-Pauli-word RA-ADAPT use the same accepted-refit
implementation.

## Staged scientific studies

### Study 1: stationarity comparison

Hold the corrected shared protocol fixed and use late resource weighting.
Prepare matched bundles for:

1. stationary-source response;
2. measured-residual response.

Execution begins with a small defect-sensitive validation matrix. When
objective implementation and provenance checks pass, the CHTC agent may
continue to the complete macro repair matrix without asking for another run
approval.

The complete discrepancy-driven replacement scope is primarily macro:

- macro Append-ADAPT with the common 102/148 executable macro pool;
- macro RA-ADAPT append-only under the common trust/support protocol;
- macro RA-ADAPT plateau-triggered commutation-aware insertion;
- macro RA-ADAPT always-enabled commutation-aware insertion.

Single-Pauli-word execution begins with targeted preservation runs for the
successful plateau-triggered commutation-aware insertion behavior. A full
single-Pauli-word replacement matrix is required only if the selected canonical
policy differs from the historical single-Pauli-word protocol.

Agents report complete objective outputs. They do not select the winner. The
workflow pauses after Study 1 for the user to select stationary-source or
measured-residual response.

### Study 2: Phase-I resource weighting

Using the user-selected Study-1 gradient policy, prepare and run the matched
all-phase-resource-weighting follow-up. Its comparison source is the
corresponding late-resource-weighting run from Study 1.

Do not combine the stationarity and Phase-I-cost changes into one uninterpretable
comparison.

### Result handling

Automated checks may verify route identity, source hashes, pool membership,
derivative contracts, trust receipts, accounting, deterministic compatibility,
and artifact completeness. Agents must not judge whether an energy/resource
trajectory is scientifically preferable or promote/demote paper evidence.

## Mathematical and characterization tests

Tests remain outside the algorithm interface. Do not add test-only flags,
callbacks, branches, or public knobs. A production seam extracted for testing
must also be a scientifically meaningful module boundary.

Required coverage includes:

- finite-difference actual-position gradients and Hessians for append and
  interior insertion;
- common retained-support factorization and Phase-III solve contracts;
- no-overlap trust receipts and overlap-call counts across adapters;
- accepted-refit chart construction and one-chart-per-admission behavior;
- separate curvature-shift and trust-multiplier receipts;
- common executable macro-pool hashes for RA and Append;
- unchanged singleton parent ancestry;
- staged-child RA exposure versus global-child Append exposure;
- stationary-source omission of active-gradient acquisitions and charges;
- measured-residual acquisition and use of \(\mathbf g_\theta\);
- cost-scope receipts for late and all-phase resource weighting;
- deterministic preservation of the existing single-Pauli-word
  plateau-insertion route under the historical-compatible policy;
- import/reachability checks before archival.

Trajectory regressions are behavioral evidence, not agent authority to judge
scientific quality.

## ICM and GitNexus

Use the real repair as the first useful ICM workflow:

```text
lock -> refactor -> verify -> materialize bundles -> analyze -> user review
```

ICM owns stage routing and receipts, not scientific defaults.

GitNexus is an index-only structural sidecar, not a runtime dependency,
scientific authority, or evidence selector. The 2026-07-27 preflight did not
run an index because the strict ignore contract is absent. Before indexing:

- create and verify a strict `.gitnexusignore`;
- exclude generated outputs, artifacts, PDFs, caches, archives, environments,
  and editor state;
- never run `gitnexus setup`;
- generate no agent files, hooks, skills, or context files;
- confirm consequential graph claims directly in source and focused tests.

Use GitNexus to verify caller concentration, dependency direction, duplicate
policy paths, and zero active reachability before archival.

Creating and validating the strict ignore surface and running this index-only
analysis are part of the `lock` stage. They precede caller migration and
archival.

## Legacy contraction and inert archive

Legacy contraction is Paper-I/static-ADAPT scoped. Do not absorb unrelated
Paper-II--V code.

The version-controlled inert archive should use a dated path such as:

```text
archive/paper_i_static_adapt_legacy_20260727/
```

It contains:

- a compact manifest;
- original paths and hashes;
- removal reasons and explicit user retirement decisions;
- inert `.py.txt` snapshots and/or patches;
- tests dedicated solely to retired routes, stored as inert `.py.txt`;
- obsolete route-specific guidance, handoffs, and runtime-setting documents.

It contains no importable `.py` modules, wrappers, executors, tests, or
fallbacks. Exclude it from imports, test discovery, packaging, GitNexus, and
ordinary agent navigation. Git history and immutable historical artifacts
remain additional provenance.

Shared mathematical and scientific contract tests remain active. Only tests
whose sole purpose is exercising an author-retired route move into the archive.
Active agent navigation should retain one compact archive/provenance manifest,
not the retired route-specific guidance tree.

Archival order is mandatory:

1. classify one named family;
2. obtain an explicit author-retirement decision when the path is reachable;
3. extract any neutral code still used by canonical RA-ADAPT or a retained
   comparator;
4. migrate live callers;
5. pass focused tests;
6. prove zero active reachability;
7. archive the route-specific remainder.

Non-canonical does not mean dead. Do not move whole directories based only on
their names.

### Locked legacy decisions

| Family | Decision |
|---|---|
| FM-SNAKE | Author-retired. Extract any currently imported neutral accepted-refit or exact-geometry primitives, then archive FM route/campaign/config/adapter code. Do not preserve a future FM extension seam before Paper I publication. |
| JR-SNAKE | Retain as an optional extension. Preserve its route/funnel and any optional joint-response batching behavior. It is not an ordinary RA-ADAPT default and is not an archival target. Verify the exact macro-to-child P1/P2/P3 funnel semantics before refactoring. |
| Optuna calibration | Author-retired. Archive the Optuna study, launchers, calibration configs, and callers that exist only for Optuna campaigns. Do not retain generic abstractions merely because they are currently imported from an Optuna-named module. If a current RA-ADAPT or Append run bundle demonstrably needs a small utility, extract only that minimum utility into its proper active owner before archival. |
| Phase-live hysteresis | Author-retired. Archive the old mechanism that retires and reactivates scoring phases, together with route/CLI/checkpoint controls that exist only for that mechanism. Preserve the distinct plateau detector used to trigger commutation-aware insertion. |
| Ordinary novelty scoring | Author-retired. Archive ordinary Phase-II/III novelty multipliers, gamma controls, pairwise novelty, and route/config surfaces that exist only for them. Retain the deferred-Gram all-models-infeasible fallback as a robustness mechanism and rename it so it is not presented as ordinary novelty scoring. |
| Historical amplitude pruning | Author-retired. Archive amplitude-collapse witnesses, amplitude-based acceptance and telemetry, and legacy small-angle pruning. Retain the typed metric- and trust-region-nominated measured delete-and-refit policies. |
| TETRIS/disjoint batching | Author-retired in full. Archive both the standalone TETRIS comparator and the duplicated in-monolith disjoint-batching selector after preserving their historical artifacts and proving no retained caller depends on them. |
| Legacy ADAPT executors | Author-retired. After canonical callers migrate, archive `adapt_pipeline_legacy_20260322.py`, `compare_adapt_current_vs_legacy_20260322.py`, and the explicit legacy RA/SR adapter. Do not move `pipelines/hardcoded/` wholesale; migrate and classify its live aliases separately while preserving dirty work. |
| Historical profiles and CLI controls | Author-retired. Archive old SR v2/v3 profiles, obsolete route registries, and legacy CLI controls after the canonical RA-ADAPT and retained JR-SNAKE callers migrate. |
| Current optional RA policies | Retain typed greedy/combinatorial RA batching and metric/trust-region pruning as non-default optional extensions. |
| Paper-I `pipelines/hardcoded` aliases | Author-retired after caller migration. The seven current untracked files are compatibility forwarders: `adapt_pipeline.py`, `adapt_circuit_cost.py`, `hh_continuation_generators.py`, `hh_continuation_scoring.py`, `hh_continuation_symmetry.py`, `hh_continuation_types.py`, and `imported_artifact_resolution.py`. Preserve their dirty contents, migrate imports to the canonical owners, prove zero reachability, and archive these aliases as inert source. Do not move the rest of `pipelines/hardcoded/`. |
| Old Paper-I runners | Author-retired after replacement. Replace `paper_i_runner.py` and the route-specific executable portions of `paper_i_hh_powell_pareto.py` with typed RA-ADAPT/Append run-bundle surfaces. Preserve immutable source locks, run artifacts, and provenance before archival. |
| Commutation Route A | Author-retired record partitioning by commutation and qubit support. Archive only that routing/partitioning implementation after caller migration. |
| Commutation-aware insertion | Canonical and preservation-critical. Retain actual-position commutation-reduced insertion. |
| Phase-I physical-family lane protection | Retain. |
| Post-split lane-based decisions | Remove from the canonical engine after reachability confirmation. |
| Historical artifacts and PDFs | Preserve immutably; do not maintain executable legacy parsers solely to regenerate them. |

### Retirement implementation discipline

Several author-retired families are still reachable in the pre-refactor code.
Claude must therefore treat retirement as caller migration plus archival, not
as a blind directory move. Each family still requires its own reachability
proof, focused tests, and archive manifest entry before implementation removes
it from active source.

## Preservation boundaries

- Preserve all completed Paper-I run artifacts, manifests, source maps, PDFs,
  and provenance through publication.
- Preserve unrelated dirty work, especially untracked compatibility aliases in
  `pipelines/hardcoded/`.
- Do not create public configuration seams for the stationary-source,
  measured-residual, or Phase-I-cost experiments. Keep them as typed
  source-locked bundle policies until the user selects a canonical result.
- Do not add a generic cross-paper launcher or six-regime default registry.
- Do not preserve obsolete executors merely to read old identifiers.
- Do not assume the corrected macro trajectory should reproduce the historical
  operator sequence; exact reproduction would preserve the insertion-chart
  defect.
- Update `CONTEXT.md`, routers, and ordinary agent guidance from SR-SNAKE to
  RA-ADAPT only after the new facade and characterization tests pass. Guidance
  must not point at an unfinished interface.

## Inputs for the Claude Fable handoff

Claude must consume:

- this decision record;
- `agent_guidance/shared/icm-gitnexus-pilot-plan.md`;
- `agent_guidance/static-adapt/CONTEXT.md`;
- `MATH/paper_facing/paper_I_static_scaffold/paper_i_macro_singleton_protocol_alignment_20260727.md`;
- `/Users/jakestrobel/.codex/attachments/6cb91f3b-abf9-4327-9390-e808674c2b9c/pasted-text.txt`;
- the GPT-5.6-xhigh preflight findings supplied by the parent agent.

Claude must separate:

1. confirmed current-code facts;
2. locked author decisions;
3. proposed module boundaries;
4. objective tests and migrations;
5. questions requiring the user;
6. optional future work outside Paper I.

## Code Math Bijection sequencing

Code Math Bijection is the separate semantic-engine repository:

`/Users/jakestrobel/Documents/Code math bijection.`

It owns UUID-backed mathematical identity, lexical resolution, semantic drift,
deterministic project-semantic JSON, and read-only scientific adapters.
Holstein remains the scientific implementation and provenance authority. The
two repositories must not acquire a runtime dependency or merge their semantic,
GitNexus, ICM, pedagogical, or learner-state graphs.

Use the same cached Claude Fable session in two steps:

1. Claude first completes the RA-ADAPT implementation specification.
2. Before the specification is handed to GPT-5.6 ultra, Claude consumes the
   follow-up handoff and creates a bounded Code Math Bijection readiness
   addendum. The addendum identifies stable semantic objects, receipts, hashes,
   locators, and implementation constraints; it does not implement or redesign
   the semantic engine.

GPT-5.6 ultra then implements the RA-ADAPT refactor with the readiness addendum
as a constraint. The full Code Math Bijection plan and Paper-I adapter update
occur only after the refactor and audit pass. Structural mapping may proceed
while GPT-5.6 high executes CHTC bundles, but final result/manuscript bindings
wait for the user's stationarity and cost-policy decisions.

The cached-session follow-up handoff is:

`prompt-exports/2026-07-27-paper-i-ra-adapt-code-math-bijection-claude-fable-followup.md`

Files to edit:

- `agent_guidance/static-adapt/history/ra-adapt-unification-refactor-decisions-20260727.md`
- Code: none in this decision-record step.
