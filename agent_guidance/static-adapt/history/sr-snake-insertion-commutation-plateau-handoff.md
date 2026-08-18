# Implement the immediate plateau-triggered commutation-reduced insertion route

## Objective

Implement a runnable, opt-in SR-SNAKE route named
`insertion_commutation_plateau_v1` against the current legacy executor. During
ordinary progress it remains append-only. After one weak accepted post-refit
energy decrease, its next selection round opens the full logical insertion
domain and collapses exactly commuting-equivalent positions before ordinary
Phase-I/II/III ranking.

This is an immediate one-factor experiment so runs can begin before the
numbered deep-controller extraction is complete. Do not begin Issue #9 or
rewrite the new controller architecture.

## Decisions already made

- The active parent is the current Paper-I no-prune profile:
  `sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1`.
- The parent remains unchanged and append-only. The new route receives its own
  normalized profile identity and digest.
- The first round is append-only.
- Plateau means the realized energy decrease from the previous accepted state
  to the newly accepted post-full-refit state is below `1e-8`.
- `1e-8` is a provisional, uncalibrated route default and must be identified as
  such in code comments and the normalized receipt.
- The trigger is immediate: one below-threshold accepted transition. There is
  no patience, hysteresis, exact-ED error, gradient-flatness, repeated-family,
  noise, escape, or historical plateau-acquisition trigger.
- If the immediately preceding accepted decrease remains below threshold, the
  widened domain remains active. Once accepted progress is at least the
  threshold, the next round returns to append-only.
- When active, start from every logical insertion gap for each shortlisted
  generator, then reuse the existing exact termwise commutation certificate to
  retain one canonical representative per commuting-equivalence class.
- Opening positions changes only candidate-domain construction. Phase I,
  Phase II, Phase III, trust solving, admission, full refitting, stopping,
  pruning-off, batching-off, beam-off, and accounting remain those of the
  parent profile.
- Do not use exact ground-state energy or exact error in any online decision.

## Authority to implement and repair

Read root `AGENTS.md`, `MATH/AGENTS.md`, `agent_guidance/README.md`,
`agent_guidance/static-adapt/history/sr-snake-refactor-plan.md`, and
`agent_guidance/static-adapt/history/paper-i-sr-snake-current-run-map.md` before
editing. Use `$codebase-design`, `$tdd`, and the two-axis `$code-review`
workflow. Repair ordinary route registration, CLI normalization, receipt, and
focused-test plumbing as needed and continue.

The write scope for this slice is:

- `pipelines/static_adapt/adapt_pipeline.py`;
- `pipelines/static_adapt/cli_config.py`;
- `pipelines/static_adapt/sr_snake_route_profile.py`;
- a new dedicated test module for this policy.

Do not edit `pipelines/static_adapt/sr_snake/`, the Issue-8 facade test, or the
future Issue-9 context files.

## Constraints

- Preserve the characterized Issue-7/8 default trajectory and route digest.
- Do not reinterpret the existing historical `adaptive` mode as this policy.
- Do not mutate the existing full-commutation diagnostic profile.
- Do not expand the generator pool; expand only generator-position records.
- Do not add a plateau stop condition. Existing run stopping remains
  authoritative.
- No manuscript, CHTC, paper-scale run, commit, push, label, or external
  repository action.
- Preserve unrelated dirty-tree changes and use path-limited Git inspection.

## Definition of done

- A registered route/profile can be selected immediately for local runs.
- Tests prove first-round append-only behavior, immediate opening after one
  weak accepted decrease, full-domain construction while open,
  commutation-equivalence reduction, automatic closing after restored
  progress, and exact-reference invariance.
- A route-profile test proves that only the intended insertion policy differs
  from the active no-prune parent.
- Receipts record open/closed state, trigger energy decrease, threshold,
  requested positions, retained representatives, and uncalibrated status.
- Existing commutation-reduction, facade, Issue-7 characterization, route
  profile, accepted-refit, checkpoint, and estimator-ledger focused tests pass.
- Independent Standards and Spec reviews return no unresolved findings.

Report changed files, the exact resolved route/profile/digest, test commands
and results, and review outcomes. Stop after this route; do not start Issue #9.

## True stop conditions

Stop only if the active parent profile cannot remain unchanged, the policy
would require exact-reference information online, or current accepted-state
semantics cannot distinguish pre-refit from post-refit energy. Ordinary
implementation and test failures are repairable.
