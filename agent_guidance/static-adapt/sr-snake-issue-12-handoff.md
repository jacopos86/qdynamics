# Implement Issue #12: authoritative default controller behind the facade

## Objective

Implement GitHub Issue #12 on the reviewed Issue-11 transition baseline.
Compose the extracted selection and transition into the authoritative
controller for the exact current Paper-I default:

```text
resolve initial accepted state
repeat:
    immutable accepted state
      -> _select_singleton(...)
      -> _transition_singleton(...)
      -> immutable next accepted state/event
    project configured checkpoint observation from the accepted event
    evaluate public stop conditions after the accepted transition
until a stop condition fires
return one typed SRRunResult
```

The active public `run_sr_snake(problem, request)` path must no longer execute
its ordinary controller loop inside
`adapt_pipeline._run_hardcoded_adapt_vqe`. Historical routes remain available
through explicit compatibility paths. Stop before Issue #13; do not migrate
other callers in this ticket.

## Authority and settled behavior

- Use GPT-5.6-Sol Ultra, `$codebase-design`, `$tdd`, and both axes of
  `$code-review`.
- Read root `AGENTS.md`, `MATH/AGENTS.md`, `agent_guidance/README.md`,
  `agent_guidance/static-adapt/sr-snake-refactor-plan.md`, the current run map,
  GitHub Issue #12, and completed Issue-7 through Issue-11 tests before editing.
- The exact profile/digest/no-beam gate remains the only new-controller route.
  Do not widen route ownership.
- `_selection.py` remains the sole ranking owner. `_transition.py` remains the
  sole accepted admission/refit/trust/round-ledger owner.
- The controller owns iteration, public stop evaluation, accepted-event order,
  checkpoint projection, and construction of the typed primary result. It does
  not reimplement Phase I/II/III mathematics, accepted refitting, or estimator
  primitives.
- The public default is exactly 50 controller rounds. A supplied positive
  `maximum_controller_rounds` replaces 50.
- One controller round is one successful selection plus one accepted
  transition/refit cycle. It is not generator count. This ticket remains
  singleton/no-batch.
- Recorded Paper-replay round counts are provenance checks, not a new public
  stop-policy variant and not a source-locked horizon.
- Optional exact-ED stopping keeps the finite maximum-round cap and is
  evaluated only after an accepted transition. Exact/reference energy never
  enters selection, trust, refit, or online controller decisions other than
  this explicit post-transition stop comparison.
- Preserve deterministic simultaneous-stop semantics already characterized:
  `exact_ed_target_reached` is primary when it and
  `maximum_controller_rounds` fire on the same accepted transition; fired
  reasons remain ordered exact target then maximum.
- Observation policy must not affect operators, parameters, energies,
  transition receipts, stop receipts, or estimator accounting.
- Accepted-state resume remains intentionally unreachable until Issue #19.
  Pruning, batching, and beam remain off with no subordinate live settings.

## Deep controller boundary

Earn `pipelines/static_adapt/sr_snake/_controller.py`. The new controller must
have a small private interface over the already resolved Issue-9 context and
cohesive numerical runtime. It should own explicit private types for:

- current accepted state and controller-round index;
- one selection result and one accepted transition result;
- ordered accepted-state/transition event stream;
- deterministic stop evaluation/receipt;
- checkpoint projection requests/results;
- final in-memory controller outcome consumed by `runner.py`.

Do not hide the controller behind an `executor(**legacy_kwargs)` callback that
still loops in `_run_hardcoded_adapt_vqe`. A default-route regression must be
able to replace `run_legacy_sr_snake` and the legacy mega-loop entry with
raising sentinels while `run_sr_snake` still completes through the new
controller.

The exact mechanical extraction can keep numerical runtime construction in
`adapt_pipeline.py` temporarily, but iteration control and accepted-state
progression must live in `_controller.py`. Group live numerical dependencies
behind substantive private runtime/kernel types; do not use a dictionary union
or a callback that implicitly captures the entire mega-function.

## Public result flow

`SRRunResult` must remain the two-argument facade's single primary result and
contain:

- final accepted state;
- accepted-state trajectory;
- one typed accepted-transition receipt per completed controller round;
- problem and route receipts;
- deterministic stop receipt;
- scientific replay/checkpoint receipts;
- estimator accounting;
- observation artifact receipts.

Add the smallest public portable transition receipt type(s) in `contracts.py`;
do not expose live private numerical objects, optimizer instances, statevectors,
or internal sidecars. Preserve existing result fields and serialization unless
the new transition field is the required additive change.

## Stop contract

Move stop evaluation out of legacy payload inference and into the controller:

1. `maximum_controller_rounds` is always active and fires after exactly that
   many accepted transitions.
2. `exact_ed_target_reached` is active only when configured and may fire only
   after at least one accepted transition.
3. Evaluate all active conditions against the same next accepted state.
4. Record each condition's `active`/`fired` status, all simultaneous fired
   reasons, and one deterministic primary reason.
5. Keep unexpected terminal numerical outcomes explicit and fail closed rather
   than relabeling them as a public configured stop.

Do not preserve `max_depth` or `benchmark_abs_delta_e_target` as hidden
controller authorities; they may remain compatibility translations only.

## Checkpoint and observation contract

- Consume the Issue-11 `_CheckpointReadyAcceptedStateEvent` only after the
  transition returns successfully.
- Checkpoint projection/serialization is outside `_transition_singleton`.
- Preserve current checkpoint hashes, strict replay, parameter identities,
  route/digest, estimator prefix receipts, cadence, and history-tail behavior.
- Changing checkpoint cadence or estimator-ledger destination must not change
  the in-memory trajectory or accounting.
- Result construction must not depend on an observation file already existing.

## Required tests

Add a direct controller suite, normally
`test/test_static_adapt_sr_snake_controller.py`, and strengthen facade
integration:

- default request resolves to 50 rounds without executing 50 scientific rounds
  in a unit test: use a deterministic fake selection/transition runtime to
  prove the loop count;
- explicit maximum 1/2/N counts accepted transitions, not operators;
- exact target is ignored at the initial state, evaluated after transition,
  and keeps the maximum cap;
- simultaneous exact+maximum firing has deterministic condition, fired-reason,
  and primary-reason order;
- one selection and one transition occur per completed round and the next state
  feeds the following selection;
- transition/checkpoint-ready events are ordered and one-to-one with the typed
  public transition receipts;
- the active public facade succeeds when both `run_legacy_sr_snake` and the
  legacy mega-loop entry are patched to raise;
- historical compatibility remains explicitly routed through the legacy
  adapter;
- observation destinations/cadence do not change the trajectory;
- exact two-round characterization preserves operators, energies, Phase-III
  ranks/coordinates, refit/trust receipts, checkpoint hashes/replay,
  `S_alg=[299,709]`, `S_unique=[250,564]`, and all 709 estimator occurrences;
- default result serialization contains the additive typed transition receipts
  and all existing primary fields.

Run the complete Issue-7 through Issue-12 focused aggregate, `py_compile`, and
path-limited `git diff --check`.

## Definition of done

- `_controller.py` visibly owns accepted state -> selection -> transition ->
  next state -> stop.
- The exact default `run_sr_snake` path does not call the legacy controller loop
  or infer its stop receipt from a legacy payload.
- Checkpoint projection consumes accepted-state events after transition.
- `SRRunResult` includes typed transition receipts and complete existing
  primary receipts.
- Historical routes remain on explicit compatibility paths.
- All focused tests and exact trajectory/accounting/checkpoint anchors pass.
- Independent Standards and Spec reviews return no unresolved findings.
- Issue #13 is not started.

## True stop conditions

Stop only if the extracted Issue-10/11 operations cannot be invoked under a
controller-owned loop without changing estimator occurrence identity, if
checkpoint projection necessarily causes numerical work before stop
evaluation, if exact-stop provenance disagrees with the resolved problem, or if
code/tests disagree on simultaneous-stop semantics. Ordinary runtime-bundle,
adapter, serialization, and test failures are repairable.

## Files to edit

- `pipelines/static_adapt/sr_snake/_controller.py`: authoritative exact-default
  loop, state/event progression, stop evaluation, and in-memory outcome.
- `pipelines/static_adapt/sr_snake/contracts.py`: minimal public typed
  transition receipt and additive `SRRunResult` field.
- `pipelines/static_adapt/sr_snake/runner.py`: route the exact default through
  the controller and assemble the primary result without legacy stop inference.
- `pipelines/static_adapt/sr_snake/_context.py`: only the minimum resolved
  numerical/controller runtime dependency changes.
- `pipelines/static_adapt/sr_snake/_legacy_adapter.py`: retain explicit
  compatibility ownership; do not use it for the exact default controller.
- `pipelines/static_adapt/adapt_pipeline.py`: expose/move the exact active
  numerical runtime and remove the ordinary default loop from legacy ownership
  without moving historical route unions.
- `test/test_static_adapt_sr_snake_controller.py`: direct controller contract.
- Existing facade/trajectory/context/transition tests: minimum strengthened
  integration assertions for route ownership and unchanged behavior.

