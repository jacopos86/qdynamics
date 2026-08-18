# Implement issue 8: add the typed SR-SNAKE request and result facade

## Objective

Implement GitHub issue
`https://github.com/20jastrobel/Holstein_test/issues/8` in the active checkout:

`/Users/jakestrobel/local_repos/Holstein_test_fullclone_3`

Add the deep two-argument public seam:

```python
run_sr_snake(
    problem: ResolvedProblemContext,
    request: SRRunRequest | None = None,
) -> SRRunResult
```

The first implementation must translate the typed request through a private
legacy adapter and preserve the Issue-7 trajectory. Do not begin issue 9.

## Decisions already made

- Use `$codebase-design`, `$tdd`, and the two-axis `$code-review` workflow.
- The external seam accepts only the resolved physical problem and one optional
  nested immutable request. `SRRunRequest` has exactly `method`, `execution`,
  and `observation` top-level fields.
- `request=None` means singleton admission, pruning off, beam off, 50 controller
  rounds, no exact-ED target, fresh start, and default observation.
- Public method choices represent singleton, greedy-batch, or
  combinatorial-batch admission; pruning off or recoverability pruning; and beam
  off or fork-local beam. Disabled variants have no subordinate settings.
- The caller never supplies route/profile strings, optimizer, seed, trust/refit
  internals, numerical guards, liveness, hysteresis, or historical flags.
- The runner derives route family/profile/digest and resolved execution
  receipts. The characterized default identity is:
  `singleton_response_snake` /
  `supported_projected_generalized_source_metric_no_overlap_trust_full_response_symmetric_cost_no_prune_v1`
  / `fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2`.
- The stop policy always has a positive finite controller-round cap, defaulting
  to 50. It may also carry a predefined same-cutoff exact-ED target. Exact ED is
  checked only after an accepted full refit and cannot enter selection, trust,
  optimization, pruning, or beam decisions.
- If the exact target and round cap fire on the same accepted transition, record
  both and make exact-target reached the primary reason.
- Define the approved optional policy types now, but do not silently map them to
  incompatible historical behavior. Enabled pruning, batching, and the new
  fork-local beam execution belong to issues 14-19. Issue 8 must not make the
  historical frozen-parent beam reachable through the new facade.
- `SRControllerState`, `CandidatePositionRecord`, `AdmissionDecision`, and
  `AcceptedTransition` remain private and are not needed in this ticket.

## Authority to implement and repair

Read root `AGENTS.md`, `MATH/AGENTS.md`, `agent_guidance/README.md`,
`agent_guidance/static-adapt/history/sr-snake-refactor-plan.md`, and
`agent_guidance/static-adapt/history/paper-i-sr-snake-current-run-map.md` before
editing. The static-adapt subtree currently has no `AGENTS.md`; do not invent
one.

Create the public package at `pipelines/static_adapt/sr_snake/` with intentional
exports, immutable contracts, the runner, and `_legacy_adapter.py`. Ordinary
missing adapter, serialization, or test plumbing is repairable; implement it,
validate narrowly, and continue.

## Constraints

- Do not route through `paper_i_runner.py::run_paper_i_route_a`; that facade owns
  a different joint/batch-oriented contract.
- Reuse `pipelines.contracts.problem.ResolvedProblemContext`.
- The private adapter may translate to
  `adapt_pipeline._run_hardcoded_adapt_vqe` and existing CLI/profile
  normalization machinery, but none of that keyword surface may leak through
  the public interface.
- Do not add public pool-plan or estimator-executor abstractions.
- Preserve Issue-7 tests and fixtures. The routine trajectory test is
  `test/test_static_adapt_sr_snake_no_prune_trajectory.py`; the compact
  provenance receipt and opt-in live audit are in
  `test/test_static_adapt_sr_snake_issue7_provenance_anchor.py`.
- Preserve unrelated dirty-tree work. Use path-limited Git inspection.
- No paper-scale run, CHTC work, manuscript edit, evidence promotion, commit,
  push, or label change.

## Definition of done

- The public package exports only the run operation, intentional request policy
  variants, exact-stop/resume/observation inputs, result, and public receipts.
- Request variants are immutable, deterministically serializable, and enforce
  progressive disclosure structurally rather than with enable flags plus
  dormant fields.
- The default facade executes the real characterized route through the private
  adapter and reproduces Issue 7: route identity, operators and positions,
  energy trajectory, Phase-III ranks/coordinates, accepted refits,
  checkpoints/replay, estimator-ledger closure, `S_alg`, and observation
  invariance.
- The typed result contains the final accepted ansatz/parameters/energy,
  accepted trajectory, resolved problem and route receipts, stop receipt,
  scientific replay receipts, estimator accounting, and observation receipts.
- Tests cover the exact three-field request shape, defaults, serialization,
  disabled-feature silence, round-cap replacement, exact-ED composition and
  precedence, accepted-state-only exact checks, typed resume shape, derived
  route receipts, and unchanged Issue-7 trajectory.
- Focused tests pass, followed by independent Standards and Spec reviews. Report
  exact commands and results. Stop after Issue 8.

## True stop conditions

Stop only if current code and the approved plan disagree on scientific
semantics, exact-stop behavior cannot be kept post-refit-only, or the facade
would require changing the characterized trajectory. Ordinary adapter,
typing, serialization, or test failures are repairable and should not end the
task.
