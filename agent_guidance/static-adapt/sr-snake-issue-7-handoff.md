# Implement issue 7: characterize the active no-prune SR-SNAKE trajectory

## Objective

Implement GitHub issue
`https://github.com/20jastrobel/Holstein_test/issues/7` in the active checkout:

`/Users/jakestrobel/local_repos/Holstein_test_fullclone_3`

Add a small, deterministic, complete-run characterization test for the current
Paper-I no-prune singleton SR-SNAKE path. Protect its route identity, accepted
trajectory, Phase-III behavior, accepted refits, checkpoints, estimator-ledger
closure, and `S_alg`. Finish the implementation and focused tests. Do not begin
issue 8.

## Decisions already made

- This ticket characterizes existing behavior; it does not redesign or change
  the algorithm.
- The intended baseline is the current reported Paper-I no-prune singleton
  route: full Phase-III active-plus-singleton response, batching off, beam off,
  pruning off, hysteresis off, and noiseless execution.
- Use the smallest route-faithful two-site Hubbard-Holstein fixture that
  exercises the real Phase-I/II/III, supported-trust, full accepted-refit,
  checkpoint, and estimator-accounting path and remains suitable for routine
  tests.
- Use exact assertions for discrete identities and justified tolerances for
  floating-point outputs.
- The current dirty working tree is the implementation baseline. Preserve
  unrelated user changes and use path-limited Git inspection.
- This is implementation and unit-test work, not a paper-scale scientific run,
  manuscript edit, evidence promotion, CHTC submission, or issue-8 facade work.

## Authority to implement and repair

Read root `AGENTS.md`, `MATH/AGENTS.md`,
`agent_guidance/README.md`,
`agent_guidance/static-adapt/sr-snake-refactor-plan.md`,
`agent_guidance/static-adapt/paper-i-sr-snake-current-run-map.md`, and
`agent_guidance/static-adapt/route-identities.md` before editing. The referenced
`agent_guidance/static-adapt/AGENTS.md` is currently absent; do not invent it.

Inspect the exact active call path and existing nearby tests. You may add
focused fixtures, test helpers, and minimal non-scientific plumbing required to
exercise and observe the current route. Repair ordinary test or serialization
plumbing, validate it narrowly, and continue. Do not alter mathematical
policies, defaults, route behavior, optimizer behavior, or evidence artifacts.

## Required provenance resolution

Do not request the unqualified `sr_snake` alias: it resolves to an older
pruning-enabled profile.

There is a stale-identity discrepancy to resolve from current local evidence:

- refactor-plan decision 9 names
  `supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_v1`
  with digest beginning `023bc7ac`;
- the newer current-run map names
  `supported_projected_generalized_source_metric_no_overlap_trust_full_response_symmetric_cost_no_prune_v1`
  with digest beginning `fd5ec3fa`, and identifies it as the route used by the
  current visible Paper-I rows.

Trace the current-run map’s named support JSON, frozen input bundle, executable
profile resolver, and complete route-specific call site. Use the unambiguous
newest visible-result provenance if those sources agree. Report the exact stale
plan fields for correction; do not fabricate expectations or silently select a
third route. This inspection is read-only provenance support for the test, not
authorization to launch or promote a run.

## Definition of done

- A focused characterization test executes the real complete route and passes.
- It protects route family/profile/digest; accepted generator identities,
  ordering, and insertion positions; energy after each accepted controller
  round; Phase-III coordinate/rank and trust receipts; accepted-refit receipts;
  checkpointed accepted states; estimator-ledger closure; and `S_alg`.
- An observation-only variation is proven not to change the accepted
  trajectory.
- The best immutable Paper-I anchor is checked read-only for identity and
  available known values; missing coverage is reported rather than invented.
- Relevant focused tests are run and their exact commands/results are reported.
- Final report lists changed files, explains any minimal plumbing repair,
  states the resolved profile/digest evidence, and confirms that issue 8 was not
  started. Do not commit, push, or change GitHub labels.

## True stop conditions

Stop and ask one precise question only if the named current support artifact,
frozen bundle, executable resolver, and route-specific call site remain
scientifically inconsistent after tracing, or if completing the test would
require changing algorithm semantics. Ordinary missing test plumbing, fixture
construction, serialization, or focused test failures are repairable and
should not end the task.
