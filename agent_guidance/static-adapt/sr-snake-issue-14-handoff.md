# SR-SNAKE Issue 14 implementation handoff

Status: complete. Implementation, focused verification, and independent Spec
and Standards reviews are closed. Issue 13 is closed. This ticket implements
only GitHub Issue 14; Issues 15 and later remain out of scope.

## Governing behavior contract

- `SRRunRequest()` remains the exact fd5ec singleton/no-prune route. Pruning-off
  requests and results expose no active subordinate pruning settings and retain
  the Issue-7 characterization.
- `RecoverabilityPruning()` selects an explicit child of the active fd5ec
  route. It changes only the documented pruning fields:
  `recoverability_ladder_v1`, live mode, one full-logical affine-FS nomination,
  radius `0.125`, zero metric damping, endpoint-overlap measurements off, and
  terminal prune off.
- The accepted-transition authority order is admission, full supported-FS
  Powell refit, prune nomination, at most one measured full-survivor
  delete/refit sibling, accepted-state choice, estimator-ledger closure, and
  checkpoint-ready event.
- The measured delete/refit energy is the deletion authority. A rejected sibling
  is discarded without mutating the keep state; its estimator occurrences stay
  in all-work accounting. An accepted sibling becomes the actual post-deletion
  state returned by the transition and public result.
- The active policy, nomination, decision, trust-radius update, measured work,
  and accepted/rejected classification are serialized in typed receipts.
- Existing query-neutral, material-window, v3.1, and July-18 compatibility
  execution paths are not imported into the new controller.
- No scientific run, evidence promotion, manuscript edit, commit, push, or
  external issue mutation is authorized.

## Verification contract

- Start with public-contract and transition tests that fail while Issue 14 is
  unreachable.
- Preserve the exact no-prune characterization and affected SR-SNAKE suites.
- The deterministic public two-site Hubbard--Holstein fixture reports honest
  no-nominee receipts through round 3 and exactly one naturally nominated,
  measured, rejected sibling at round 5. Its rejected branch contributes
  exactly `103` logical scalar-estimator invocations to all-work but not the
  winning lineage; the keep-state fingerprint is unchanged.
- A six-round public continuation proves the contracted deletion trust radius
  persists (`0.125 -> 0.0625`, then the next round starts at `0.0625`).
- Controlled typed transition fixtures cover accepted and immutable rejected
  deletion. A controlled public run retains the exact measured sibling/refit
  while widening only the fixture guard, and proves that an accepted deletion
  becomes the reduced public final state and terminal checkpoint. A bounded,
  untuned search did not yield a natural accepted deletion in the small public
  fixture; do not alter physics or production acceptance semantics merely to
  manufacture one.
- The normalized CLI admits only the exact new profile, digest, and pruning
  settings into the public controller. Historical pruning profiles remain on
  their compatibility paths.
- Final affected verification passed `230` SR-SNAKE/controller/route tests and
  `84` estimator-ledger/`S_alg`/prune/trust tests. Fresh Spec and Standards
  reviews returned `PASS`.
- Do not begin Issue 15.

Unresolved questions/problems: natural accepted-deletion numerical coverage is
limited to the controlled transition fixture described above. This is a stated
fixture limitation, not authorization to tune scientific settings.

Files to edit:
- `agent_guidance/static-adapt/sr-snake-refactor-plan.md`
- `pipelines/static_adapt/sr_snake/contracts.py`
- `pipelines/static_adapt/sr_snake/_context.py`
- `pipelines/static_adapt/sr_snake/_transition.py`
- `pipelines/static_adapt/sr_snake/_controller.py`
- `pipelines/static_adapt/sr_snake/_cli_compatibility.py`
- `pipelines/static_adapt/sr_snake/runner.py`
- `pipelines/static_adapt/adapt_pipeline.py`
- `pipelines/static_adapt/sr_snake_route_profile.py`
- focused `test/test_static_adapt_sr_snake_*.py` and route-profile tests
