# Implement Issue #13: migrate canonical callers and observational adapters

## Objective

After Issue #12 is complete and reviewed, migrate only the exact active
singleton/no-prune/no-batch/no-beam SR-SNAKE caller path through
`run_sr_snake`. Preserve legacy payloads, replay readers, resumed executions,
historical routes, and post-run reporting through explicit compatibility
boundaries. Stop before Issue #14.

## Governing behavior

- The public controller and typed result are the authority for the active
  fresh-run route.
- The active CLI builds `SRRunRequest` only after its normalized route,
  profile digest, and inactive-policy gates exactly match the current route.
- A Paper-I replay adapter may compare a requested round limit with an
  immutable source record. This is private provenance validation, not a new
  public stop-policy variant.
- A compatibility serializer outside the controller projects `SRRunResult`
  into the legacy inner payload required by current consumers. Do not attach a
  raw legacy dictionary to `SRRunResult`.
- Qiskit compilation and cost reporting remain post-run consumers. No Qiskit
  object or compiled-cost value enters selection, transition, controller
  state, stopping, or the typed scientific result.
- `AcceptedStateResume` remains unreachable through the public facade until
  Issue #19. Issue #13 preserves fresh-run checkpoint projection and the
  explicit legacy resume bridge; it does not activate typed frontier resume.
- No historical route, reader, or implementation is deleted.

## Exact caller disposition

### Migrate

- The active exact-profile branch of the `adapt_pipeline` CLI after complete
  argument normalization. It must call `run_sr_snake`.
- The frozen six-regime Paper-I commands are covered through that CLI branch;
  do not edit the frozen command bundles.

### Retain as explicit compatibility

- `paper_i_runner.run_paper_i_route_a`: its canonical settings include
  batching/pruning and it is not the conventional facade.
- Generic Hubbard--Holstein benchmark comparators.
- Powell/FM/JR dispatch.
- Historical and staged-Optuna profiles unless their fully normalized request
  is exactly the active facade contract.
- Noise, formal-manifold, experimental, pruning, batching, beam, and resumed
  invocations.
- `resume_scaffold` readers.
- Qiskit selected-prefix and recovery-prefix readers/reporters.

Give the old executor one clearly named compatibility entry. Do not move,
delete, or contract its implementation before Issue #20.

## Required implementation seams

1. Add one exact-gated CLI adapter that converts the already resolved physical
   problem, stop policy, and observations into the typed request.
2. Add one deterministic typed-result-to-legacy-payload projection outside the
   controller. Preserve operators, parameters, accepted history, active-prefix
   checkpoints, route identity, stop data, and estimator accounting.
3. Add one private Paper-I replay round-lock validator. It must reject internal
   disagreement in the immutable record and requested/source disagreement
   before execution. The public stop contract remains only a finite round cap
   plus optional exact target.
4. Name every retained direct legacy-executor call as a compatibility call.
5. Preserve fresh checkpoint output and prove existing legacy resume readers
   can consume it. A resumed CLI request remains on the compatibility route.

## Tests

- Exact active CLI routing succeeds with both the legacy adapter and legacy
  executor patched to raise.
- Profile/digest drift, FM/JR, comparator, noise, resume, pruning, batching,
  beam, and experimental cases use only the named compatibility entry.
- The exact two-round fixture preserves legacy operators, logical/runtime
  parameters, accepted energies, checkpoint hashes, route digest,
  `S_alg=[299,709]`, and complete estimator accounting after compatibility
  projection.
- Projected final state matches the typed fingerprint and energy.
- Matching replay round succeeds; requested/source mismatch and internally
  inconsistent source records fail before execution; public serialization
  contains no source-locked-horizon variant.
- Observation cadence and destination do not change the accepted trajectory.
- Existing legacy checkpoint-plus-resume reproduces the uninterrupted
  singleton trajectory and estimator-prefix closure.
- Public `AcceptedStateResume` still raises as an Issue-19 guard.
- The active run succeeds with Qiskit compiler entry points patched to raise;
  a post-run sidecar leaves the accepted-trajectory digest unchanged.
- Run the Issue-12 controller/facade/trajectory suite plus affected CLI,
  resume-reader, Qiskit-reader, and compatibility tests.

## Stop conditions

Stop and report rather than changing scientific behavior if:

- Issue #12 is incomplete or has unresolved review findings;
- a required current consumer field cannot be reconstructed from typed
  receipts;
- compatibility projection cannot reproduce the typed final fingerprint;
- legacy resume cannot preserve estimator-prefix closure without changing the
  accepted trajectory; or
- code, tests, canonical settings, and route identity disagree.

Do not start Issue #14, launch scientific runs, edit manuscripts, commit, push,
or change issue labels.

Files to edit:

- `pipelines/static_adapt/adapt_pipeline.py`: exact-profile CLI dispatch and
  named compatibility fallback.
- `pipelines/static_adapt/sr_snake/`: private CLI and result compatibility
  adapter(s).
- `pipelines/static_adapt/paper_i_runner.py`: compatibility entry-name update
  only if its direct call remains.
- `pipelines/exact_bench/hh_static_ground_state_benchmark.py`: compatibility
  entry-name update only.
- `pipelines/exact_bench/paper_i_hh_powell_pareto.py`: compatibility
  entry-name update only.
- Focused Issue-13 routing, serialization, replay-round, checkpoint/resume, and
  Qiskit-boundary tests.

## Issue #13 completion record

Status: implementation complete; Issue #14 has not begun.

- The normalized exact active CLI route calls the public `run_sr_snake`
  operation. Profile/digest drift and every currently reachable noncanonical
  stop, noise, parity, worker, resume, pruning, batching, beam, FM/JR, and
  experimental control retain the named legacy compatibility route with an
  explicit reason receipt.
- Accepted checkpoints are published at the configured cadence as a
  consumer-complete `adapt_vqe` envelope plus a content-addressed authenticated
  estimator-ledger sidecar and a content-addressed v2 verified-singleton resume
  sidecar. The current envelope authenticates the v2 sidecar bytes; the sidecar
  binds a canonical source projection with only its pointer omitted, avoiding a
  digest cycle, and records a resolved source path so relative public-current
  paths round-trip through the reader. Both sidecars are durably written before
  the atomic
  current-pointer replacement, and every atomic replacement fsyncs its parent
  directory. Fixed-name v1 resume sidecars remain an explicit legacy
  compatibility branch, but the active `fd5ec3fa...` envelope identity
  independently requires v2 and rejects pointer-removal downgrade. A private
  temporary checkpoint supplies the same authenticated observation channel
  when the caller requests no public current path.
- The unchanged public `extract_verified_singleton_resume_checkpoint`
  entrypoint consumes both requested history-tail `1` and history-tail `0`
  current envelopes. Its sidecar helpers now authenticate the v2
  pointer/projection proof while retaining the explicit legacy-v1 branch. Tail
  requests retain an explicit retention receipt while the authoritative
  current envelope normalizes both `history` and `history_tail` to the complete
  serialized lineage required by that reader. A forced post-publication
  interruption after the exact one-round horizon preserved the round-1
  operator/checkpoint hashes and restored `S_alg=299`, `S_unique=250`, and all
  299 occurrence records. Tampering the resume-sidecar bytes,
  source-projection hash, or estimator-ledger sidecar bytes fails closed, as
  does deleting the active v2 pointer while presenting a retained fixed-name
  v1 sidecar.
- The exact two-round route preserves the characterized operators, parameters,
  energies, checkpoint hashes, route digest, and cumulative accounting
  `S_alg=[299,709]`, `S_unique=[250,564]`. Public observation receipts
  authenticate the final current bytes, and selected-prefix/recovery-prefix
  readers leave result and ledger bytes unchanged.
- The immutable Paper-I replay lock and the reduced fresh CLI segment lock
  remain distinct validators and receipts. The reduced segment projection does
  not claim a terminal-Qiskit round or an exact source-locked horizon.
- Compatibility scope is explicitly consumer-complete, not the historical
  executor's full diagnostic union. The unsegmented adapter projection retains
  51 of 224 observed historical top-level keys; the outer CLI postprocessor
  adds the generic `boson_subspace_diagnostics` field, so serialized
  `result.json` contains 52. A segment lock adds only `adapt_segment`, yielding
  52 at the adapter layer and 53 after the generic postprocessor. The
  unsegmented projection retains history 35 of 264, continuation 6 of 92, and
  accounting 17 of 21; segment enrichment yields history 39 of 264,
  continuation 7 of 92, and accounting 17 of 21. Reachability and retirement
  of the omitted diagnostic-only fields remain Issue-20 debt.
- The July-18 six-regime bundle remains a named historical
  whitened/stale-identity compatibility fixture. It was not edited, executed,
  or reclassified as the active `fd5ec3fa...` route.
- Focused verification after the checkpoint redesign: SR-SNAKE aggregate
  `168 passed, 1 skipped`; affected consumer/caller/resume/Qiskit aggregate
  `162 passed, 9 skipped`; CLI/control/output/checkpoint aggregate `88 passed`;
  route-profile aggregate `238 passed`; exact Issue-13 route file `5 passed`.
- Final post-v2 independent review closed with zero findings: Spec `PASS` and
  Standards `PASS`. Both reviews reproduced the authenticated v2
  pointer/projection path, active-route v1 downgrade rejection, relative
  public-current path round trip, tamper failures, exact accounting, layered
  compatibility counts, July-18 compatibility status, and the Issue-14
  boundary.

Unresolved questions/problems:

- None within Issue #13. The omitted diagnostic-union reachability work belongs
  to Issue #20, and typed frontier resume remains deferred to Issue #19.
- Issue #14 is outside this handoff and remains unstarted.

Files to edit: None.
