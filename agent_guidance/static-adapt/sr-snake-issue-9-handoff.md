# Implement Issue #9: prefactor one private resolved execution context

## Objective

Implement GitHub Issue #9 after the Issue-8 facade and the immediate
`insertion_commutation_plateau_v1` slice are stable. Resolve the typed request
and physical problem exactly once into one private immutable execution context
consumed by both the legacy adapter and future controller extraction. Do not
begin Issue #10.

## Decisions already made

- The public seam remains
  `run_sr_snake(problem: ResolvedProblemContext, request: SRRunRequest | None =
  None) -> SRRunResult`.
- The public request retains exactly `method`, `execution`, and `observation`
  top-level fields.
- The characterized default remains singleton admission, pruning off, beam
  off, append-only insertion, 50 controller rounds, no exact target, and the
  current no-prune Paper-I route identity.
- Route/profile strings, pool construction, optimizer settings, numerical
  kernels, estimator execution, trust/refit internals, and historical flags
  remain private.
- The immediate plateau-insertion experiment is an optional nondefault route.
  Issue #9 must not absorb, delete, or redefine it. It remains on its registered
  experimental legacy path until a later candidate-domain policy ticket moves
  it behind the deep controller.
- The context owns immutable resolved dependencies, not mutable controller
  state, candidate records, admission decisions, or accepted transitions.

## Authority to implement and repair

Read root `AGENTS.md`, `MATH/AGENTS.md`, `agent_guidance/README.md`,
`agent_guidance/static-adapt/sr-snake-refactor-plan.md`,
`agent_guidance/static-adapt/paper-i-sr-snake-current-run-map.md`, and GitHub
Issue #9 before editing. Use `$codebase-design`, `$tdd`, and the two-axis
`$code-review` workflow. Ordinary context construction, serialization,
adapter, typing, and test plumbing are authorized repair work.

## Constraints

- Work from the completed Issue-8 facade and preserve every Issue-7/8
  trajectory and accounting anchor.
- Prefer a new private module such as
  `pipelines/static_adapt/sr_snake/_context.py`.
- The legacy adapter consumes the resolved context; it must not independently
  normalize the same request again.
- Do not create public pool-plan or estimator-executor abstractions.
- Do not move Phase-I/II/III selection or the controller loop in this ticket.
- Do not import historical funnels, FM controllers, noise, escape,
  hysteresis, or dormant route settings into the current context.
- Do not edit the plateau agent's dedicated test module while that work is in
  progress. Re-read its changed route/profile surfaces before final validation.
- No manuscript, CHTC, paper-scale run, commit, push, label, or external
  repository action.
- Preserve unrelated dirty changes and use path-limited Git inspection.

## Definition of done

- One private immutable resolved context owns the active problem, derived
  route receipt, pool/materialized scientific dependencies, numerical kernels,
  optimizer/refit policy, estimator ledger, observation destinations, stop
  policy, and initial accepted state needed by the current executor.
- Construction occurs exactly once per public run.
- The legacy adapter consumes this context without public API or scientific
  behavior changes.
- Focused tests prove single resolution, immutability, invalid-composition
  failure, observation invariance, and unchanged default route identity.
- The registered plateau-insertion route and its dedicated tests remain
  unchanged.
- The complete Issue-7/8 focused regression set passes.
- Independent Standards and Spec reviews return no unresolved findings.

Report changed files, exact tests/results, and review outcomes. Stop after
Issue #9; do not start Issue #10.

## True stop conditions

Stop only if the completed facade and the active characterized route disagree
on scientific semantics, or resolving once would necessarily change the
trajectory or estimator accounting. Ordinary adapter, typing, and test
failures are repairable.
