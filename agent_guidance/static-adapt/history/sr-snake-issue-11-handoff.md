# Implement Issue #11: accepted singleton transition and full refit

## Objective

Implement GitHub Issue #11 on the reviewed Issue-10 selection baseline. Route
the active default no-prune singleton run through one deep private accepted
state transition:

```text
preceding accepted state
  + immutable singleton admission decision
  -> admit exactly that generator-position record at zero amplitude
  -> complete supported-FS Powell refit of the full accepted ansatz
  -> close the round estimator-ledger prefix
  -> immutable next accepted state
  + checkpoint-ready accepted-state event
```

Stop before Issue #12. The transition emits a checkpoint-ready event but does
not write a checkpoint, append a report/history row, apply stopping policy, or
format output.

## Authority and settled behavior

- Use GPT-5.6-Sol Ultra, `$codebase-design`, `$tdd`, and both axes of
  `$code-review`.
- Read root `AGENTS.md`, `MATH/AGENTS.md`, `agent_guidance/README.md`,
  `agent_guidance/static-adapt/history/sr-snake-refactor-plan.md`, the current run map,
  GitHub Issue #11, and the completed Issue-7 through Issue-10 tests before
  editing.
- The active dispatch remains the exact characterized profile/digest with beam
  disabled. Do not widen the gate.
- `_select_singleton` remains the sole ranking owner. The transition consumes
  `_SingletonAdmissionDecision`; it must not rebuild the domain, rescore,
  rerank, substitute a candidate, or mutate the preceding accepted state.
- The Issue-10 one-key runtime sidecar is a temporary live-object bridge. The
  transition must validate that its sole key equals
  `decision.selected.domain_record_id`; identities and order come from the
  immutable decision, never from sidecar reassignment.
- Preserve the current singleton admission at zero amplitude, exact insertion
  position, logical/runtime layout mapping, optimizer-memory remap, selection
  count, pool availability, and source/child identity receipts.
- Preserve the complete accepted-ansatz supported-FS whitened Powell refit and
  its same-iteration external Gram reuse. Reuse the existing
  `AcceptedRefitConfig`, `SupportedFSPowellChart`,
  `_make_accepted_refit_optimizer_chart`, and accepted-refit numerical
  authorities; do not duplicate their mathematics.
- Pruning and batching are off and carry no subordinate settings in this
  transition. Do not import beam, plateau insertion, formal-manifold, noise,
  historical fallback, Route-C, or compatibility-route behavior into the new
  seam.
- Exact stopping, checkpoint persistence, observation destinations, history
  formatting, and final result formatting remain outside the transition.

## Deep private interface

Earn `pipelines/static_adapt/sr_snake/_transition.py`. Use private immutable
types for the transition input/receipt rather than a dictionary union. The
exact names may follow the code, but the interface must make these concepts
explicit:

- preceding accepted-state identity and numerical snapshot;
- immutable singleton admission decision;
- cohesive private numerical/runtime workspace;
- next accepted-state identity and numerical snapshot;
- admission receipt;
- supported-FS refit chart and optimizer receipt;
- preceding and final accepted energies;
- observational non-worsening receipt carrying the raw energy comparison;
- authoritative round-ledger prefix/closure identity;
- one checkpoint-ready accepted-state event.

The checkpoint-ready event contains the data needed by the existing checkpoint
builder, but it is not a serialized report payload and causes no file write.
Keep live arrays/executors in the private runtime workspace; keep portable
identities and immutable receipts in the returned transition result.

## Extraction boundary

Trace the exact default route after `default_singleton_decision` in
`pipelines/static_adapt/adapt_pipeline.py`. Extract only the behavior actually
reached by the characterized profile:

1. validate the decision-bound live record and admit its `AnsatzTerm` at the
   decision insertion position with zero new angle;
2. update the selected layout, logical/runtime parameter identities,
   optimizer memory, selection count, and available-pool membership;
3. construct/reuse the characterized external Phase-III Gram receipt;
4. run the complete supported-FS Powell chart/refit;
5. construct the accepted-refit result and record whether the accepted energy
   is non-worsening; do not invent a new rollback or hard-failure policy;
6. close the exact round estimator-ledger prefix and emit the immutable next
   state plus checkpoint-ready event.

Do not copy the surrounding legacy admission/refit union into the new module.
Use one exact-route kernel/workspace if live closures are temporarily needed,
and remove unreachable duplicate default-only bodies. Other profiles continue
through their existing paths.

## Required tests

Add the smallest direct transition contract suite, normally
`test/test_static_adapt_sr_snake_transition.py`, plus focused integration
assertions where necessary:

- one transition call per successful default round;
- the transition receives the exact immutable decision and cannot rerank or
  substitute its winner;
- preceding accepted operators, parameters, state fingerprint, and energy are
  unchanged at the call boundary;
- exactly one zero-amplitude singleton is admitted at the authorized position;
- full accepted-ansatz supported-FS Powell refit, chart dimension/rank, external
  Gram reuse, optimizer receipt, and final energy match the Issue-7 anchors;
- the non-worsening receipt is explicit and true for both characterized rounds;
  the extraction does not reject an otherwise completed legacy transition
  solely because an uncharacterized future run reports a false receipt;
- the returned next-state identity replays to the returned accepted state;
- the round ledger closes once, with the exact ordered estimator prefix and
  cumulative `S_alg`/`S_unique`;
- exactly one checkpoint-ready event is emitted, with no checkpoint,
  observation, history, or report write inside the transition;
- selection remains mutation-free and the transition performs no phase
  ranking;
- the plateau route and all Issue-7 through Issue-10 focused tests remain
  unchanged.

The complete two-round characterization must retain operator order, energies,
Phase-III coordinates/ranks, trust receipts, refit charts, parameter
identities, checkpoint hashes/replay, cumulative `S_alg=[299,709]`,
`S_unique=[250,564]`, and all 709 estimator occurrence identities/reuse/order.

## Definition of done

- The exact active default route calls one private transition operation after
  one private selection operation.
- `_transition.py` owns the immutable accepted-state transition contract;
  numerical kernels remain in their existing authorities.
- Admission/refit no longer depends on legacy reassignment of `best_idx`,
  `selected_position`, or winner identity.
- The transition returns the next accepted state and one checkpoint-ready
  event but performs no persistence or formatting.
- Direct transition tests and the complete focused regression aggregate pass.
- `py_compile` and path-limited `git diff --check` pass.
- Independent Standards and Spec reviews return no unresolved findings.
- Issue #12 is not started.

## True stop conditions

Stop only if the characterized default refit is energy-worsening under its
current tolerance, the accepted-refit chart cannot be separated from
checkpoint/history writes without changing numerical execution, the ledger
cannot close at the transition boundary without changing occurrence identity,
or code/tests disagree on the route's refit mathematics. Ordinary private-type,
adapter, extraction, and focused-test failures are repairable.

## Files to edit

- `pipelines/static_adapt/sr_snake/_transition.py`: private accepted-state,
  transition, refit, ledger-closure, and checkpoint-ready event types/operation.
- `pipelines/static_adapt/adapt_pipeline.py`: exact default admission/refit
  dispatch and removal of duplicated default-only transition bodies.
- `test/test_static_adapt_sr_snake_transition.py`: direct transition contract.
- Existing Issue-7 through Issue-10 tests: only the minimum strengthened
  integration assertions needed to prove unchanged public behavior.
