# SR-SNAKE Issue 16 implementation handoff

Status: implemented and narrowly verified on 2026-07-25. This ticket covers
only optional combinatorial batching. Issue 17 and later beam/composition work
were not started.

## Governing behavior contract

- `SRRunRequest()` remains the characterized fd5ec singleton/no-prune route.
  Issue-15 greedy request/result types and serialized checkpoint schemas remain
  distinct and unchanged.
- `CombinatorialBatchAdmission()` defaults to `maximum_size=3` and a resolved
  ranked Phase-III search window of `6`. An omitted window resolves as
  `min(2 * maximum_size, 10)`. A positive integer requests an explicit bounded
  window. `FullCombinatorialSearchWindow()` explicitly requests the complete
  ranked Phase-III population. Public `0` is invalid.
- The route family is `combinatorial_batch_response_snake`, using profile
  `supported_projected_generalized_source_metric_no_overlap_trust_`
  `full_response_symmetric_cost_no_prune_combinatorial_batch_v1`.
  Maximum size and resolved window semantics participate in its
  request-specific contract and digest.
- The search population is a ranked Phase-III prefix. Each member is an
  immutable generator-plus-insertion-position record. The canonical existing
  Phase-II ordering within that fixed population defines subset proposal and
  commit order.
- Selection enumerates every generator-distinct subset of cardinality one
  through the configured cap. It enumerates subsets, not permutations, and
  never searches alternative insertion positions.
- Each subset is evaluated with the coupled full-active-plus-batch
  Gram/Hessian response and supported generalized trust solve. The score uses
  joint predicted energy descent with symmetric predictive cost. Historical
  additivity gating and the `0.9` score shell are off.
- Pair geometry is acquired once in a shared workspace. Physical evaluations
  equal cache misses; the estimator ledger records one metric and one Hessian
  occurrence per required pair. Classical subset enumeration adds no estimator
  charge.
- A selected subset is one atomic accepted round: fixed insertion positions
  receive only deterministic index shifts caused by earlier members, every
  member starts at zero angle, and the controller performs one full
  supported-FS Powell refit, trust update, ledger closure, checkpoint, and
  round increment.
- A one-member winner retains combinatorial route identity and combinatorial
  receipts.
- Combinatorial checkpoints use a distinct authenticated, content-addressed
  projection sidecar. They emit no verified-singleton or greedy resume pointer
  and explicitly deny reconstruction until Issue 19.
- Explicit combinatorial CLI intent is exact-gated. Contract, digest, profile,
  family, maximum-size, or window drift fails before the legacy executor.
  Genuinely unrelated historical profiles retain named compatibility.
- Pruning, beam, accepted-state resume, and their compositions remain gated to
  later tickets. Combinatorial batching plus fork-local beam correctly reports
  the Issue-19 composition boundary; singleton beam remains the Issue-18
  boundary.

## Completion evidence

- Controlled and deterministic HH fixtures prove exhaustive cardinality
  receipts, fixed insertion positions, coupled selection, pair-cache
  accounting, one atomic controller round, public replay/checkpoint projection,
  one-member fallback identity, and exact CLI dispatch.
- The default-window contract is consistent across the public policy, direct
  route helper, context, and CLI for caps 1, 3, and 5. Explicit bounded and
  explicit full-window requests remain distinct.
- Internal exhaustive receipts validate exact considered-subset counts for
  every cardinality; evaluated and feasible counts cannot exceed considered
  counts.
- Shared batch diagnostics and pair-accounting schemas receive an explicit
  strategy label, so combinatorial failures and sidecars do not claim greedy
  provenance.
- Pre-closure validation recorded `85 passed` for the enlarged core aggregate,
  `8 passed` for helper/cache/accounting targets, `152 passed` for route and
  Paper-I configuration regressions, `162 passed` for the complete continuation
  scoring test file, and `26 passed` for controller/no-prune/recoverability
  guards.
- After the final low-severity diagnostic and Issue-19-boundary repairs, a
  direct narrow closure command passed `13` focused combinatorial, greedy, CLI,
  checkpoint, and frozen no-prune tests in `14.93s`. Path-limited
  `git diff --check` was clean.
- No scientific run, evidence promotion, manuscript edit, commit, push, issue
  mutation, or external action was performed.

Unresolved questions/problems: none inside Issue 16. Issue 17 is a separate
fork-local estimator-accounting ticket and was intentionally not started.

Files to edit: None for Issue 16 closure. Future work begins from the separate
Issue-17 contract only after an explicit decision to continue the refactor.
