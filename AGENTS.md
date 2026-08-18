# Repository Agent Router

This is the shared entry contract for coding agents. `CLAUDE.md` imports it for
Claude Code; Codex reads it directly. Tool-specific entrypoints and installed
skills are adapters and must not duplicate or override repository or scientific
authority.

High-level Markdown is an agent control surface, not human narrative. Optimize
it for explicit contracts, stable paths, progressive disclosure,
machine-readable manifests, and regression-backed behavior.

## Active checkout

Use:

`/Users/jakestrobel/local_repos/Holstein_test_fullclone_3`

The checkout under
`/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3`
is FileProvider/iCloud-managed recovery material unless the current task
explicitly revalidates it. Before a run, confirm the active path and verify that
Python sources under `src/` and `pipelines/` are not `compressed,dataless`.

Invariant: other agent sessions may be active in this checkout at any
time. For repo-changing work, create your own git worktree (under
`Holstein_test_fullclone_3.worktrees/`) on your own branch, and stage
commits with explicit paths only — never `git add -A` here. See Hard
invariants.

## Route first

Read only the next surface triggered by the request. Do not mine manuscripts,
generated PDFs, run artifacts, route registries, or historical handoffs first.

| Request | Read next |
|---|---|
| Repository inspection, code, tests, docs cleanup, Git, or handoff | `agent_guidance/shared/repository-work.md`, then the nearest subtree `AGENTS.md` and target files |
| Paper identity or workflow routing | `agent_guidance/README.md`, then the selected lane contract when present |
| ADAPT, Hubbard--Holstein, molecular-vibronic, time dynamics, QSE, reporting, mathematical defaults, or route identity | `MATH/AGENTS.md`, `agent_guidance/shared/scientific-invariants.md`, then the selected lane contract when present |
| Benchmark planning, local/remote execution, CHTC, Optuna, manifests, artifact aggregation, evidence report, or canonical run settings | the scientific route above, then `agent_guidance/shared/run-guide.md` and the exact existing run workflow named by the lane |
| Completed-evidence table, source-map, or provenance refresh | `MATH/AGENTS.md`, then an existing matching results workflow or explicit target evidence contract |
| Manuscript prose, title, abstract, caption, citation, `.tex`, or PDF-facing edit | `MATH/AGENTS.md`, then its manuscript contract and available journal-review workflow |
| Prompt/GPT/Atlas handoff export | the matching available handoff workflow; write the requested export under `prompt-exports/` |
| Issue or triage work | `docs/agents/issue-tracker.md` and `docs/agents/triage-labels.md` |
| Storage, disk space, artifact retention, compression, or cleanup | `agent_guidance/shared/artifact-retention.md`, then the dated cleanup records under `agent_guidance/shared/` |
| Domain terminology or architectural decision | `docs/agents/domain.md` and the selected lane `CONTEXT.md` when present |

If a routed file does not exist, do not invent it. Continue read-only work under
the nearest valid contract and fail closed only when the missing contract is
required for a state-changing run, evidence transfer, or manuscript operation.

## Paper lanes

Internal shorthand:

| Paper | Lane | Scientific role |
|---|---|---|
| Paper I | `agent_guidance/static-adapt/` | Hubbard--Holstein RA-ADAPT producer |
| Paper II | `agent_guidance/time-dynamics/` | AP-McLachlan dynamics consumer/producer |
| Paper III | `agent_guidance/qse/` | QSE and excited-dynamics consumer |
| Paper IV | `agent_guidance/molecular-vibronic/` | Molecular-vibronic static-construction producer |
| Paper V | `agent_guidance/paper-v/`, `paper_5/` | high-`U` regularization/GKBA exploration |

Paper numbers express workflow and evidence ownership, not software
inheritance. Use method names rather than paper numbers in reader-facing prose.

Static construction is paper-first and method-second:

```text
Paper I -> Hubbard--Holstein problem/evidence ─┐
                                               ├-> shared static ADAPT/SNAKE
Paper IV -> molecular-vibronic problem/evidence┘
```

The shared method returns its result to the originating lane. Paper IV must not
inherit Paper-I Hamiltonian defaults, source locks, evidence, or manuscript
contracts. The current typed `run_ra_adapt` facade remains executable only for
the Paper-I Hubbard--Holstein `L=2` problem until molecular-vibronic
admissibility is explicitly implemented and tested.

## Authority

For the current task, apply:

1. this root contract;
2. the nearest applicable subtree `AGENTS.md`;
3. machine-readable configuration and typed defaults;
4. `MATH/AGENTS.md` and the selected paper lane when scientific scope applies;
5. an existing workflow or skill triggered by the requested action;
6. target code, tests, support contracts, and preserved manifests;
7. `README.md` for orientation only.

The user's current, narrower instruction wins over broader default scope.
Historical documentation never overrides current code, tests, typed settings,
or source-locked provenance.

## Workflow and skill discipline

- Trigger a workflow or skill because the requested action requires it, not
  merely because a paper or module is mentioned.
- Confirm every repo-local `SKILL.md` exists before treating it as a gate.
- A `$name` in downstream guidance identifies a capability adapter. Use it when
  the current agent environment provides it; do not fabricate it when absent.
- Missing run/evidence/manuscript capability blocks only the state-changing
  workflow that requires it, not ordinary inspection, implementation, or tests.
- Do not edit a workflow skill unless the user explicitly asks to change that
  skill.

## Hard invariants

- Preserve unrelated work, completed evidence, manifests, and source locks.
- Git commits are at the agent's discretion, always (user policy,
  2026-08-17): commit deliberately with descriptive messages, without
  asking. This deliberately overrules earlier commit gating anywhere in
  this repository's documentation.
- Multi-agent isolation (user-relayed policy, 2026-08-18): when other
  agent sessions may be active in this repository, do repo-changing work
  in your own git worktree (under `Holstein_test_fullclone_3.worktrees/`)
  on your own branch. Stage commits with explicit paths only — a broad
  `git add -A` in a shared checkout sweeps other agents' uncommitted
  work into your commit (incident: `ba7f2ac9`, 2026-08-18).
- Do not run, submit, stop, kill, resume, push, promote evidence, or
  change external state without authority from the current request or an
  already-active authorized objective.
- Never interfere with a job from another chat, agent, terminal, notebook, or
  scheduler submission unless the user identifies that job as in scope.
- A status question or “what next?” is not authorization to change state or
  delegate work.
- Scientific changes synchronize implementation, tests, and typed canonical
  settings. Named experiments and ablations do not silently change defaults.
- Exact/classical references are reporting inputs, not online controller inputs.
- Paper evidence adoption, promotion, and demotion are user decisions.
- Do not silently rewrite manuscripts when implementation or settings change.
- Do not trade correctness, scientific fidelity, testing, or review quality for
  brevity or token economy.

## Plans and responses

Keep routine responses compact and lead with the outcome, objective, or
blocker. Ask only questions whose answers materially affect scientific
semantics, external state, or scope.

Plans must state unresolved problems near the end and finish with:

`Files to edit: ...`

Use `Files to edit: None` when no edits are planned.
