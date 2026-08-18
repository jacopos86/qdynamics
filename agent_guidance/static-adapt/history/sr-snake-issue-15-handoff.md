# SR-SNAKE Issue 15 implementation handoff

Status: implemented and verified on 2026-07-25. The user-visible greedy-window
decision is resolved below. This ticket covers only GitHub Issue 15;
combinatorial batching in Issue 16 and later composition work remain separate
tickets.

## Governing behavior contract

- `SRRunRequest()` remains the exact characterized fd5ec singleton/no-prune
  route. Singleton requests and results retain their existing serialized shape,
  route digest, trajectory, and receipts; no dormant batching settings appear
  while batching is off.
- `GreedyBatchAdmission(maximum_size=3, search_window_size=None)` is the only
  newly reachable admission variant. `maximum_size` bounds the number of
  records actually admitted. `search_window_size` is a positive ranked
  Phase-III candidate-window cardinality when supplied; `None` means the full
  ranked candidate population. The legacy executor's `0`-means-all encoding is
  private to its adapter boundary. The request remains a typed
  singleton-or-greedy choice, never an independent batching Boolean plus
  nullable subordinate fields.
- A greedy request resolves a distinct normalized route/profile contract whose
  only intentional parent changes are the greedy admission policy,
  `maximum_size`, and ranked `search_window_size`. The route identity stays
  greedy even when a particular round admits only one record. The historical
  `batch_near_degenerate_ratio=0.9` shell does not participate in this route.
- Phase I and Phase II retain their current scientific roles. Phase III returns
  one immutable admission decision containing an ordered nonempty tuple of
  distinct candidate-position records, bounded by the configured maximum
  cardinality. The decision records the selected cardinality, record
  identities, deterministic order, joint score/geometry receipt, predictive
  cost, and estimator-event identities.
- Selection is observationally pure: constructing a greedy proposal cannot
  mutate the accepted state. It may reuse the existing reduced-plane joint
  geometry/greedy proposal machinery only through a small adapter with the
  active route's supported-metric, response, cost, lineage, and estimator-ledger
  invariants; it must not import a historical Route-A/beam/batch bundle.
- One accepted transition consumes exactly the immutable ordered decision,
  admits all authorized records at zero initial logical amplitude with explicit
  old-to-new logical/runtime remaps, and performs one full supported-FS Powell
  refit over the entire enlarged accepted ansatz.
- One controller round remains one complete
  selection-admission-full-refit-ledger-close cycle, regardless of how many
  generators the decision admits. Stop counters advance once per accepted
  transition, not once per admitted generator.
- Transition and public receipts carry the ordered batch identities,
  cardinality, per-record admissions/remaps, full-refit coverage, accepted state,
  and exact estimator-ledger closure. Existing scalar singleton fields may not
  ambiguously stand in for a multi-record batch.
- Pruning, beam, resume, combinatorial batching, scientific runs, evidence
  promotion, manuscript edits, commits, pushes, and external issue mutation are
  not authorized by this handoff.

## Verification contract

- Start test-first at the typed request/route and immutable selection/transition
  seams.
- Prove progressive disclosure for singleton versus greedy request and result
  serialization, including route/profile/digest distinction and no inactive
  greedy settings on the singleton route.
- Use controlled selection fixtures to prove deterministic ordered greedy
  growth, cardinality bounds, distinct record identities, singleton fallback
  under the greedy route, state immutability, and estimator-event closure.
- Use controlled transition fixtures to prove one atomic multi-record
  admission, complete logical/runtime remaps, zero-angle embedding, one
  full-ansatz supported-FS refit, one trust update, one ledger close, one
  checkpoint event, and one controller-round increment.
- Add one small deterministic complete public Hubbard--Holstein run that
  naturally admits at least one multi-record greedy batch and validates the
  typed result and route receipts. This is a test fixture, not a scientific
  production run.
- Re-run the exact default singleton characterization unchanged, followed by
  affected route/profile, controller, transition, estimator-accounting, and
  CLI compatibility tests.
- The affected aggregate exposed a shared legacy-path initialization defect:
  a terminal Phase-I round could read
  `archival_projected_route_active_for_round` before either route branch
  assigned it. The common round setup now initializes that flag to `False`;
  both existing route-specific assignments remain authoritative. This is a
  nonsemantic shared-plumbing repair, not greedy-batching behavior.

## Completion evidence

- The public greedy route admits an ordered batch of one to the configured
  maximum, performs one full supported-FS Powell refit, one trust update, one
  ledger closure, one checkpoint event, and one controller-round increment.
  A one-member fallback retains the greedy route and batch receipt types.
- Candidate-pair accounting observes the actual scoring-kernel results. It
  does not recompute metric or Hessian entries while recording metadata;
  physical evaluations reconcile with cache misses and the live ledger records
  one metric and one Hessian occurrence for every required pair.
- Normalized CLI dispatch reconstructs request-specific `maximum_size` and
  `search_window_size` from the canonical contract. Explicit greedy intent
  fails closed before legacy dispatch on profile, family, contract, digest,
  maximum, or window drift. Genuinely other historical profiles retain their
  named compatibility path.
- Greedy checkpoints use a distinct content-addressed sidecar, authenticate
  their source projection and ordered batch history, omit the verified
  singleton resume pointer, and state that reconstruction is not authorized
  until Issue 19.
- Final local verification passed: focused Issue-15 facade/controller tests
  `70 passed`; route-profile tests `155 passed`; exact singleton no-prune and
  recoverability-prune guards `17 passed`; affected selector/order/ledger/
  checkpoint/no-batch tests `71 passed`. `py_compile` and path-limited
  diff-hygiene checks passed.
- Fresh post-repair Spec and Standards reviews both returned clean with no
  findings. No scientific run, evidence promotion, manuscript edit, commit,
  push, issue mutation, or other external action was performed.

Unresolved questions/problems: none at the contract boundary. The approved
window is ranked Phase-III search-window cardinality, not a near-degenerate
score-shell ratio. The public full-window default is `None`; only the private
legacy adapter maps it to `0`.

Files to edit:
- `agent_guidance/static-adapt/history/sr-snake-refactor-plan.md`
- `pipelines/static_adapt/sr_snake/contracts.py`
- `pipelines/static_adapt/sr_snake/_context.py`
- `pipelines/static_adapt/sr_snake/_selection.py`
- `pipelines/static_adapt/sr_snake/_transition.py`
- `pipelines/static_adapt/sr_snake/_controller.py`
- `pipelines/static_adapt/sr_snake/runner.py`
- `pipelines/static_adapt/adapt_pipeline.py`
- `pipelines/static_adapt/sr_snake_route_profile.py`
- `pipelines/scaffold/hh_continuation_scoring.py` only if the approved active
  adapter cannot preserve the required summary/receipt without a narrow kernel
  change
- focused `test/test_static_adapt_sr_snake_*.py`, route-profile tests, and
  existing greedy-selector tests
