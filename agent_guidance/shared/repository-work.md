# Shared Repository-Work Contract

Read this file for code changes, documentation cleanup, repository inspection,
Git operations, or agent-to-agent handoffs. It is tool-neutral; platform
entrypoints and installed skills are adapters to this contract.

## Default scope

Default coding scope:

- `src/`
- `pipelines/static_adapt/`
- `pipelines/time_dynamics/`
- `pipelines/scaffold/`
- `pipelines/hardcoded/`
- `pipelines/reporting/`
- `test/`
- `agent_guidance/`
- `AGENTS.md`
- `README.md`

Ignore by default unless the user explicitly requests them:

- `.obsidian/`
- `.vscode-extensions/`
- `.vscode-home/`
- `.vscode-userdata/`
- `archive/`
- `docs/`, except `docs/reports/` when report output is in scope
- `claude_code_adapt_wave/`

Use `docs/reports/`, `pipelines/exact_bench/`, and `pipelines/shell/` only when
the task requires them. `MATH/` is conditional scientific/paper scope governed
by `MATH/AGENTS.md`.

The user's narrower scope always wins.

## Change discipline

- Read the nearest applicable subtree `AGENTS.md` before changing files there.
- Preserve unrelated and uncommitted work. Do not overwrite changes whose
  ownership or purpose is uncertain.
- Make the smallest coherent change that satisfies the requested behavior.
- Diagnose read-only requests without implementing an unrequested fix.
- Git commits are at the agent's discretion, always (user policy,
  2026-08-17); commit deliberately and descriptively without asking.
- When other agent sessions may be active in this checkout, work in your
  own git worktree on your own branch and stage with explicit paths
  only; never `git add -A` in a shared checkout (user-relayed policy,
  2026-08-18).
- Do not infer permission for pushes, issue changes, scheduler
  actions, scientific runs, manuscript edits, or artifact promotion.
- For non-semantic refactors, proceed against code and tests. Stop for the user
  only when code, tests, or contracts disagree about numerical conventions,
  user-visible defaults, scientific semantics, artifact meaning, or requested
  scope.

## Search, Git, and artifact hygiene

- Use path-limited searches. Do not run `find .`, `ls -R`, `du .`, or
  repository-wide text scans from the root.
- Do not run broad `git diff`, `git status`, `git add -A`, or `git add -u`
  during normal work.
- Prefer explicit paths, for example:

  ```text
  git diff -- AGENTS.md agent_guidance/shared/repository-work.md
  git status --short -- pipelines/static_adapt test
  rg "pattern" pipelines/time_dynamics test
  find pipelines/time_dynamics -maxdepth 2 -type f -name "*.py"
  ```

- For a larger review, exclude generated trees explicitly rather than scanning
  them accidentally.
- Treat `artifacts/`, `raw_outputs/`, `output/`, `tmp/`, `logs/`, `plots/`,
  `prompt-exports/*`, and generated CHTC input/output bundles as noise unless
  the current task names them.
- Never stage generated runs, fetched scheduler outputs, PDFs, plots, logs, or
  scratch directories unless the user explicitly asks to version a specific
  artifact.
- If a command begins scanning generated output or becomes unexpectedly slow,
  stop it and narrow the path or repair ignore rules.
- Never use destructive Git or filesystem commands against a broad or
  unresolved target. Resolve exact targets first and prefer recoverable
  operations.

## Evidence-preserving output behavior

- Do not open generated reports, PDFs, plots, or images in external desktop
  applications automatically. Deliver them in the current agent session with
  clickable paths or previews. Open or reveal them externally only when the
  user asks.
- Preserve completed evidence while later work is queued, running, failed, or
  incomplete.

## Current-code claims

Before claiming how current code behaves:

1. inspect the active non-iCloud checkout;
2. trace the route-specific call path rather than only generic helpers;
3. verify configuration at the call site;
4. cite the relevant files and lines;
5. distinguish confirmed behavior from inference;
6. do not draft an implementation handoff until those checks are complete.

## Collaboration and user messages

- Do not start, stop, resume, redirect, or delegate work merely because the
  user asks “what next?” or requests status.
- A mid-turn status request, correction, answer, or compatible addition does
  not cancel the active objective. Incorporate it and continue.
- Replace the active objective only when the user clearly cancels it, gives an
  incompatible objective, or invalidates the current approach.
- When ambiguity is safe and reversible, state the interpretation briefly and
  continue. Ask only when the choice changes scientific semantics, external
  state, or material scope.
- Use subagents only when the user or an applicable local workflow explicitly
  requests delegation. Give each one a bounded responsibility and disjoint
  write scope; one primary agent integrates.

## Correctness and response contract

- Do not trade correctness, scientific fidelity, testing, or review quality for
  brevity, token economy, or an arbitrary tool-call limit.
- Keep routine updates and plans concise. Lead with the objective, outcome, or
  blocker.
- Treat the user's “shots” as accepted shorthand for estimator or query burden
  unless the distinction changes the requested calculation.
- Near the end of a plan, state unresolved questions or problems.
- End every plan with `Files to edit:` and the intended paths and symbols, or
  `Files to edit: None`.
