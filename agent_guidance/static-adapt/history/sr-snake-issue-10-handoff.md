# Implement Issue #10: extract current Phase-I/II/III selection

## Objective

Implement GitHub Issue #10 on the completed Issue-9 resolved-context baseline.
Route the active default no-prune singleton run through one deep private
selection operation:

```text
accepted controller state
  -> admissible candidate-position domain
  -> Phase I
  -> Phase II
  -> Phase III and supported trust solve
  -> immutable singleton admission decision
```

Selection must not mutate the accepted ansatz. Stop before Issue #11; accepted
admission/refit remains in the legacy loop for this ticket.

## Decisions already made

- Use GPT-5.6-Sol Ultra, `$codebase-design`, `$tdd`, and both axes of
  `$code-review`.
- The Issue-9 `_ResolvedExecutionContext` remains the immutable run dependency
  boundary. It does not own mutable controller state, the live materialized
  pool, or the live estimator ledger.
- `SRControllerState`, candidate-position records, and admission decisions are
  private internal types, not public extension points.
- Candidate-domain construction is logically separate from ranking. A
  generator-position record is the unit that advances through the phases.
- Preserve the characterized default: current Phase-I and Phase-II policies,
  full active-plus-singleton Phase-III response, projected generalized trust
  solve, symmetric candidate cost, no pruning, no batching, no beam, full
  accepted refit later in the loop, and unchanged estimator accounting.
- Selection returns one immutable singleton decision carrying generator, pool,
  insertion, symmetry, shortlist, response-coordinate, supported-rank, trust,
  predictive-cost, and estimator-event identities.
- The immediate `insertion_commutation_plateau_v1` experiment stays on its
  registered legacy path. Do not import its plateau state machine into the
  default selection seam and do not break its tests/profile.

## Authority to implement and repair

Read root `AGENTS.md`, `MATH/AGENTS.md`, `agent_guidance/README.md`,
`agent_guidance/static-adapt/history/sr-snake-refactor-plan.md`,
`agent_guidance/static-adapt/history/paper-i-sr-snake-current-run-map.md`, and GitHub
Issue #10 before editing. Read the completed Issue-7/8/9 tests and the exact
active no-beam route call path.

Create the smallest deep internal module, normally
`pipelines/static_adapt/sr_snake/_selection.py`, plus private immutable types
earned by the extraction. Ordinary adapter, typing, callback/kernel bundling,
and focused-test failures are repairable; preserve the scientific trajectory
and continue.

## Constraints

- Do not change the public `run_sr_snake` signature, request fields, exports, or
  default route/profile/digest.
- Do not perform accepted refitting, pruning, checkpoint writing, output
  formatting, stopping, or observation writes inside selection.
- Do not import historical funnels, formal-manifold controllers, noise/oracle
  routes, plateau or escape modes, phase-live hysteresis, batching, beam
  execution, or compatibility fallbacks into the new seam.
- Do not create a second route/profile normalizer or materialize the pool
  twice.
- Avoid replacing the legacy mega-function with an equally shallow
  many-argument selection function. Group cohesive live dependencies behind
  private state/workspace types and keep the selection interface small.
- Preserve exact estimator-ledger event order, occurrence identities, reuse
  receipts, shortlist ordering, deterministic tie-breaks, Phase-III coordinate
  identity, and trust receipts.
- Keep the plateau route and Issue-9 context tests green.
- No manuscript, scientific run, CHTC action, commit, push, label, or other
  external repository action.
- Preserve unrelated dirty work and use path-limited Git inspection.

## Definition of done

- The active default run calls one private selection operation from an accepted
  state and receives one immutable singleton admission decision.
- Candidate-domain construction is independently testable and separate from
  phase ranking.
- Focused tests prove no accepted-state mutation, deterministic record/order
  identity, complete decision receipts, estimator-event identity, and
  separation from refit/checkpoint/output concerns.
- The Issue-7 trajectory, route identity, Phase-III ranks/coordinates, trust
  receipts, accepted refits, checkpoint replay, estimator ledger, `S_alg`,
  facade behavior, resolved context, and plateau route are unchanged.
- The complete focused aggregate passes.
- Independent Standards and Spec reviews return no unresolved findings.

Report changed files, extracted interfaces/types, exact tests/results, and both
review outcomes. Stop after Issue #10.

## True stop conditions

Stop only if current code and characterization disagree on selection
mathematics, if the supported trust solve cannot be separated from accepted
state mutation, or if extraction necessarily changes estimator-event identity
or trajectory. Ordinary plumbing, private-type, and test failures are
repairable.
