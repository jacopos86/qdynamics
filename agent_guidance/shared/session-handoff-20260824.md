# Session handoff — Paper-I refactor scoping + Paper-II route audit (2026-08-24)

**For:** a fresh Claude Code session in this checkout.
**From:** Claude session that audited both routes on `paper-ii-exchange-selector` @ `7edf45e3`.

---

## Start here

1. Read this file end to end, then
   `agent_guidance/static-adapt/HANDOFF_ADAPT_PIPELINE_DECOMPOSITION_20260824.md`.
2. Read the governing contract (path in the next section). It defines
   correctness; this file only reports state.
3. **Confirm the author has done the golden-data rescue** ("Do this first"
   below). If not, offer to do it before anything else. Do not begin refactor
   work on top of unprotected evidence.
4. Verify the anchors still hold — `git rev-parse --short HEAD` should be at or
   after `eb88c7c4`, and `wc -l pipelines/static_adapt/adapt_pipeline.py` should
   read 72528. If either differs, the tree moved; re-measure before trusting any
   number in these documents.

The `mattpocock-skills` plugin is installed and should now be loadable. A
sensible sequence for this work:

| skill | use it for |
|---|---|
| `mattpocock-skills:grilling` | Pressure-test this handoff before acting. It rests on static analysis and one prior-art discovery; both deserve challenge. |
| `mattpocock-skills:domain-modeling` | The core defect is that "profile" has no first-class model — 116 settings sprawl across a 7-deep inheritance chain. The contract's `run_profile(...)` seam is the modelling answer; work out its types. |
| `mattpocock-skills:improve-codebase-architecture` | The decomposition itself. |
| `mattpocock-skills:to-tickets` | Split the Codex handoff's seven increments into discrete tickets if the author wants them tracked separately. |

Do not use the plugin's `handoff` skill for repo handoffs — use
`holstein-agent-handoff`, which encodes this repo's
`agent_guidance/shared/agent-handoff-contract.md` conventions that Codex reads.

Start by reading this file, then
`agent_guidance/static-adapt/HANDOFF_ADAPT_PIPELINE_DECOMPOSITION_20260824.md`
(the executable Codex spec, committed at `64e42db0`). Everything below is either
already committed or is an open decision for the author.

---

## Where things stand

Nothing has been refactored. This session produced **analysis and two committed
documents**, and deliberately changed no code, no runs, and no evidence.

| commit | what |
|---|---|
| `a00cf1fa` | Codex refactor handoff for `adapt_pipeline.py` |
| `64e42db0` | same handoff, bound to the Paper-I behavioral contract |

Read-only visual map of the Paper-II route (module graph, controller algorithm,
dead-code audit): `https://claude.ai/code/artifact/f29dd686-39ac-4a19-9cb2-8241fe384acd`

---

## The governing document

`/Users/jakestrobel/local_repos/ADAPT---Paper-I/PAPER_I_REFACTOR_BEHAVIORAL_CONTRACT.md`
— separate repo, Overleaf-synced, **never merge it into this checkout**.

It defines correctness for the Paper-I refactor: four evidence profiles
(`H-L3`, `HH-B3`, `HH-B5`, `HH-B9`), cross-profile invariants, five acceptance
gates, and a prescribed seam
`run_profile(profile_id, problem_id, arm, horizon) -> run_result`.
It explicitly licenses free internal restructuring. Do not design a competing
architecture; that decision is already made.

---

## What was found (evidence in the committed handoff)

**Paper I — the settings-drift mechanism.** Not caused by file size.
`normalize_sr_route_profile_namespace()` is fail-closed, so a flag disagreeing
with a profile raises. The only way to change one setting is to author a new
frozen profile — hence 31 settings dicts on a chain **7 levels deep** from a
single 116-key root, where changing the root changes 27 descendants and no
profile's effective settings are ever materialized. 467 CLI flags, three
competing default sources.

**Paper I — the mega function.** `_run_hardcoded_adapt_vqe`
(`adapt_pipeline.py:14876–56085`) is **41,210 lines, 348 parameters**, 1,661
`if`s, nesting depth 15, 3,104 locals, 200 nested defs. Its signature is used
reflectively at `adapt_pipeline.py:72076` as a **runtime kwargs filter** —
removing a parameter silently drops values rather than erroring. Largest hazard
in the refactor.

**Paper I — prior art.** Branch `codex/static-adapt-beam-refactor` holds a
182-commit extraction from 2026-07-04 where the file is **23,701 lines**.
Sixteen of its eighteen modules already landed on HEAD; `route_identity.py`
(723 lines) did not, and its absence causes **34 of the 55 test collection
errors**. Divergence is 173/182 commits — treat the branch as a reference
design and a map, not something to merge.

**Paper I — file audit.** 101 of 116 `static_adapt` modules are reachable.
~10,433 lines across 11 modules are unreachable (full table in the handoff).
The author's `sr_*` hypothesis was only half right: the four largest `sr_*`
files are load-bearing, and **every** `selector_*` and `route_*` file is
imported by live product code.

**Paper II — dead code with live documentation.** `ap_mclachlan/support_frontier.py`
(645 lines) has **zero importers**. Commit `657239a3` (2026-08-17) removed its
call site and left the module. Four documents still describe it as active,
including the paper-facing
`MATH/paper_facing/paper_II_dynamics/runtime_algorithm_settings.md`. The README's
claim that `append_macro_scout_exchange_fail_open` "is the canonical default"
refers to strings that exist only inside the dead file, so no run can emit that
telemetry. Frozen CHTC source still has it wired — runs before and after
2026-08-17 used different append frontiers.

**Paper II — `legacy/` is not one-commit deletable.** `benchmarks/legacy_native.py`
and `tables/generic_dynamics_*` import each other, and that pair is the
comparator/benchmark-table path.

---

## Do this first

**Rescue the golden regression data.** It is the only losable item here. Six
sources the contract's gates depend on are outside version control — three under
`output/` (gitignored, `.gitignore:49`) and three inside worktrees that
`git worktree prune` would destroy, including `kstar_tables.json` and the
Bundle-9 `package_manifest.json` in `ra-refactor-stage2`. All six existed on
`7edf45e3`. If any is lost, gates 1, 3 and 4 cannot run and the refactor cannot
be verified. This is Increment 0 of the Codex handoff; ~10 minutes.

Then Increment 1 (restore `route_identity.py` from the refactor branch) to clear
34 collection errors before Codex starts.

---

## Open decisions — author's, not the agent's

1. **`support_frontier.py`** — delete and correct four documents, or re-wire it?
   Affects a scientific claim. Commit `657239a3`'s message does not say which was
   intended.
2. **Canonical realtime runner** — root `README.md` calls
   `runners/hh_from_adapt_artifact.py` the "Chapter 17A realtime default" and
   "realtime anchor"; `pipelines/time_dynamics/README.md` calls it legacy awaiting
   migration. The `legacy/` migration cannot start until this is settled.
3. **H2O / Paper-IV material in the Paper-I lane** — `ra_adapt/h2o_application.py`
   plus two `sr_snake_route_profile.py` aliases. Root `AGENTS.md` forbids Paper IV
   inheriting Paper-I contracts. Relocate, or document an exception?
4. **Bundle-9 candidate-gain item** — the behavioral contract flags
   `phase3_candidate_gain_policy=joint_total_gain_v1` in the runtime checkpoint
   against the marginal joint-minus-active-only score in the current mathematics,
   and requires resolution before numerical lock.

---

## Method caveats — do not skip

Every reachability and dead-code verdict in these documents came from **AST
import-closure and word-boundary analysis**. That method does not follow
callables passed as arguments. `agent_guidance/shared/agent-handoff-contract.md`
records a prior incident where exactly this method marked three live cost symbols
dead because they were invoked through closures. **An index proposes; source
ratifies** — grep, read the call site, run the tests before deleting anything.

The `support_frontier` verdict was separately checked against dynamic-reference
patterns and is clean.

## Environment notes

- `mattpocock-skills@claude-plugins-official` v1.2.3 was installed this session
  (user scope, 25 skills) but was not loadable in the installing session. It
  should be available now. Relevant ones: `improve-codebase-architecture`,
  `codebase-design`, `domain-modeling`, `grilling`, `to-tickets`. Note it also
  ships `code-review`, `diagnosing-bugs` and `handoff`, which collide by name with
  personal skills — address the plugin versions as `mattpocock-skills:<name>`, and
  keep using `holstein-agent-handoff` for repo handoffs.
- GitNexus indexes **two** repos (this checkout and the `ra-refactor-stage2`
  worktree), so CLI calls need `--repo`. The index is dated 2026-08-19: stale for
  Paper-II files, current for `adapt_pipeline.py`.
- Thirteen worktrees exist and several Codex branches were committed to on
  2026-08-24. Do repo-changing work in your own worktree and stage explicit paths
  only.
