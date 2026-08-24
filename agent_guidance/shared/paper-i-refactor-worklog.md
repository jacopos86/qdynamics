# Paper-I refactor — shared worklog

**Purpose:** one coordination surface for two high-level agents (Claude and
Codex) working the `adapt_pipeline.py` decomposition in parallel. Claude holds
architecture and diagnosis; Codex executes with repo access. Both write here.

Conventions come from `agent_guidance/shared/agent-handoff-contract.md`. This
file does not restate it.

---

## Anchors

| field | value |
|---|---|
| Checkout | `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3` (NOT the `~/Documents/Holstein_implementation` iCloud mirror) |
| Branch / commit | `paper-ii-exchange-selector` @ `00a5f098` |
| Governing contract | `/Users/jakestrobel/local_repos/ADAPT---Paper-I/PAPER_I_REFACTOR_BEHAVIORAL_CONTRACT.md` — separate repo, Overleaf-synced, never merge into this checkout |
| Domain glossary | `agent_guidance/static-adapt/CONTEXT.md` |
| Architecture decision | `docs/adr/0001-sr-snake-deep-module-seam.md` |
| Golden data | `agent_guidance/static-adapt/golden/` on branch `golden-rescue-20260824` — see "Evidence state" |
| RAM ceiling | 10 GB aggregate, all agents. `agent_guidance/shared/memory-budget.md`. Wrap heavy work in `pipelines/shell/ram_guard.py --limit-mb 8000` |

### Test baseline — READ THIS BEFORE QUOTING A NUMBER

```bash
# Collection only. This is what everyone has been quoting.
python3 -m pytest test --collect-only -q
#   -> 5624 collected, 55 errors, "Interrupted: 55 errors during collection"

# The suite does NOT run without this flag. Plain `pytest test` executes ZERO tests.
python3 pipelines/shell/ram_guard.py --limit-mb 8000 -- \
  python3 -m pytest test -q --tb=no -rf --continue-on-collection-errors
```

The 55 collection errors abort the session before any test executes. That is
why no failure count existed. Measured with `--continue-on-collection-errors`:
**~16% of executed tests fail** (see "Live findings" for the settled number).

---

## Coordination protocol — two writers, one file

1. **Claim before you work.** Add a row to the Claims table below, with your
   agent name and the date. Do not start an item another agent holds.
2. **Never rewrite another agent's section.** Append a new dated entry under
   "Live findings" instead. Corrections to someone else's finding go in a new
   entry that names the entry it corrects.
3. **Sections have single owners.** "Verified findings" is Claude-owned.
   "Execution log" is Codex-owned. "Open decisions" is the author's.
   Anyone may append to "Live findings" and "Claims".
4. **Stage explicit paths only.** Never `git add -A` here — a broad add sweeps
   the other agent's uncommitted work into your commit (incident `ba7f2ac9`).
5. **Commit this file on its own** when you update it, so a conflict is a
   one-file conflict.
6. **An index proposes; source ratifies.** Every claim below carries the
   command that produced it. Re-run it rather than trusting the number.

### Claims

| item | agent | date | state |
|---|---|---|---|
| Increment 0 — golden data rescue | Claude | 2026-08-24 | done, `5e6fcb17` on `golden-rescue-20260824` |
| Full-suite failure census | Claude | 2026-08-24 | in progress |
| _(add yours)_ | | | |

---

## CORRECTIONS to the committed Codex handoff

`agent_guidance/static-adapt/HANDOFF_ADAPT_PIPELINE_DECOMPOSITION_20260824.md`
(commits `a00cf1fa`, `64e42db0`) is still broadly sound on the diagnosis, but
**three of its numbers are wrong and one of its increments should not be run as
written.** Measured on `00a5f098`.

| claim | handoff | measured | impact |
|---|---|---|---|
| Increment 1: restoring `route_identity.py` fixes 34 of 55 collection errors | 34, "55 → 21" | **2**, "55 → 53" | **Increment 1 is not worth running as specified** |
| Production code imports `route_identity` | implied load-bearing | **zero importers**; only two `.md` files mention it | restoring it adds an unused module |
| Profile inheritance depth | 7 levels | **5** | cosmetic |
| Descendants of the 116-key root | 27 | **6** | changes the drift story — see below |
| CLI flags | 467 | **450** (409 in `cli_config.py`) | cosmetic |
| The `run_profile(...)` seam | to be designed | **already exists** as `run_ra_adapt`, 3 params | changes the work from design to migration |

### Why Increment 1 is wrong

The prior session counted files *containing the string* `route_identity`. There
are 18 in `test/`, but 17 use it inside test names and assertion strings
(`test_summary_rejects_..._noncanonical_route_identity`), and the one real
import is `historical_route_identity`, which **exists** at
`pipelines/static_adapt/historical_route_identity.py`.

```bash
# 18 files mention it
grep -rl "route_identity" test/ | wc -l
# 0 files import the missing module
grep -rl '^\s*\(from\|import\).*[^_]route_identity' test/ | wc -l
# only 2 collection errors are actually caused by it
```

This is exactly the failure mode the handoff's own §4 warns about.

### What the 55 collection errors actually are

~30 distinct causes, dominated by a missing package surface, not by the beam
refactor:

| cause | count |
|---|---|
| `pipelines.hardcoded.*` — ~12 modules (`hubbard_pipeline`, `hh_pareto_tracking`, `hh_staged_cli_args`, `hh_time_dynamics_spectra`, …) | ~17 |
| `docs.reports.report_labels` | 5 |
| `chtc.*` and `pipelines.exact_bench.*` | ~8 |
| `plots`, `pipelines.error_protected.contracts`, `pipelines.excited_dynamics.*`, misc | ~8 |
| `pipelines.static_adapt.route_identity` | **2** |
| `FileNotFoundError` (missing fixture files) | 3 |

Reproduce:

```bash
python3 -m pytest test --collect-only --continue-on-collection-errors --tb=short
```

That pattern reads like a package that was moved or removed without updating
its tests. **Open question for the author:** was `pipelines/hardcoded/`
deliberately retired? If so these tests should be deleted or ported, not
repaired.

---

## Verified findings _(Claude-owned)_

Every number here was measured from source on `00a5f098`, not inherited.

### F1 — The prescribed seam already exists

`run_ra_adapt(problem, request, operational_controls)` —
`pipelines/static_adapt/ra_adapt/engine.py:5625–6388`, **3 parameters**.
This is the contract's `run_profile(...)` seam and ADR-0001's "one run
operation". **Do not design a competing architecture; the decision is made.**

The work is migration, not design: the interface is deep, but the
implementation still lives in the monolith.

### F2 — The 348-name reflective seam (largest hazard)

```
adapt_pipeline.py:72076   _CANONICAL_SR_SNAKE_LEGACY_EXECUTOR_PARAMETER_NAMES
                          = frozenset(inspect.signature(_run_hardcoded_adapt_vqe).parameters)
adapt_pipeline.py:72081   _canonical_sr_snake_legacy_executor_kwargs(...)
                          -> keeps only keys matching those 348 names
```

- `_run_hardcoded_adapt_vqe` — `adapt_pipeline.py:14876–56085`, **41,210
  lines, 348 keyword-only parameters, no `**kwargs`**.
- **Keys that do not match a parameter name are dropped silently.** Removing a
  parameter loses values instead of raising.
- Two adapters cross this seam: `cli_config.py:3736` (CLI) and
  `sr_snake/_context.py:249` (typed). Two adapters make the seam real; its
  placement at a 348-name signature makes it shallow.

Reproduce:

```bash
python3 - <<'PY'
import ast
t=ast.parse(open('pipelines/static_adapt/adapt_pipeline.py').read())
for n in ast.walk(t):
    if isinstance(n,ast.FunctionDef) and n.name=='_run_hardcoded_adapt_vqe':
        a=n.args
        print(n.lineno, n.end_lineno, len(a.posonlyargs+a.args+a.kwonlyargs), a.kwarg)
PY
```

### F3 — Settings drift: five disconnected roots, no shared base

31 settings dicts in `sr_snake_route_profile.py` (4,588 lines), dispatched by
`normalize_sr_route_profile_namespace:3904–4233` (330 lines, 37 branches).

| root | own keys | descendants |
|---|---|---|
| `CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1` | 18 | **20** — the active family |
| `CANONICAL_SR_SNAKE_V1` | 116 | 6 — the legacy family |
| 3 × `HISTORICAL_SR_SNAKE_*` | 1–4 | 0, isolated |

**The family carrying 20 of the profiles does not inherit the 116-key canonical
root.** So most keys were never pinned by the profile at all — they fall through
to parser defaults. That, not a deep chain, is why a minimal change moves many
things at once.

The author's model, adopted: **a profile is a run plus a diff.** One base holds
the complete effective settings; every other profile is base + a named delta.
`CONTEXT.md` already defines **Controlled ablation** this way; the code never
implemented it.

### F4 — Guard sprawl is a symptom, not the disease

**1,878 `raise` statements across 87k lines — one every 46 lines.** In
`sr_snake/_selection.py` and `sr_snake/_transition.py`, **one every 9 lines**.

Of 1,867 that carry a message:

| kind | count |
|---|---|
| agreement between two representations | **190** |
| required-but-missing | 212 |
| value range / finiteness | 88 |
| type / shape assertion | 58 |
| forbidden combination | 49 |
| unknown enum value | 37 |
| uncategorised | 1,233 |

Samples of the 190:

> "Fresh Phase-III Gram candidate position differs from the admitted singleton position."
> "Default runtime sidecar identity set disagrees with the immutable admission decision."
> "Resolved SR-SNAKE profile does not match its required legacy controller contract."

Each is only possible because the same fact is stored twice — once in the typed
decision, once in the flat runtime dict. **Do not delete these guards.** They
are catching real drift. They become unreachable once F2 and F3 remove the
duplication, and can go then.

---

## Recommended order

Numbered by dependency, not priority.

1. **Close the reflective seam (F2).** Replace the `inspect.signature` filter
   with an explicit typed payload; unknown key raises. Implementation does not
   move. Smallest change, and it makes the rest safe.
2. **Materialize the profile as base + delta (F3).** Must come *after* 1:
   today the filter would silently truncate a newly complete mapping to the 348
   names it recognises, and the result would look like it worked.
3. **Move execution behind `run_ra_adapt` (F1).** Use the `sr_snake` package
   that already holds selection, transition and resume (~528 KB). The mega
   function becomes a **Compatibility route** in the `CONTEXT.md` sense.

Guard sprawl (F4) is retired as a consequence of 1 and 2, not as its own task.

---

## Evidence state

Increment 0 is **done** — `5e6fcb17` on branch `golden-rescue-20260824`,
preserved under `agent_guidance/static-adapt/golden/` with `MANIFEST.sha256`
passing. That branch is not merged; merge it before relying on the data.

**Blocking evidence problem.** `bundle3_final_results_manifest.json` is a
pointer file recording 24 inputs with hashes. **20 are already gone** — they
lived in a `/private/tmp` scratchpad of another session under the iCloud
checkout. The 4 survivors all match their recorded hashes and are preserved.

Among the missing is `kstar_tables.json` @ `629c8c13…`, the version that
generated the Bundle-3 PDF; no surviving copy matches. **Bundle 3 cannot
currently be re-derived from its recorded inputs, so contract Gate 1 is not
satisfiable for it.** The contract forbids filling those cells from another
bundle. This is an author decision, not an agent decision.

The two run-archive trees (11 GB + 2.5 GB) are hash-recorded in place and
**still have no off-repo backup**.

---

## Open decisions — author's, not an agent's

1. **Bundle 3.** Gate 1 is unsatisfiable for it. Accept, re-run, or re-scope?
2. **`pipelines/hardcoded/`** — deliberately retired? Decides whether ~17
   collection errors are repairs or deletions.
3. **`support_frontier.py`** (Paper II) — delete and correct four documents, or
   re-wire? Affects a scientific claim. Commit `657239a3` does not say which was
   intended.
4. **Canonical realtime runner** — root `README.md` calls
   `runners/hh_from_adapt_artifact.py` the "realtime anchor";
   `pipelines/time_dynamics/README.md` calls it legacy awaiting migration.
5. **H2O / Paper-IV material in the Paper-I lane** — relocate, or document an
   exception to root `AGENTS.md`?
6. **Bundle-9 candidate-gain item** — `phase3_candidate_gain_policy=joint_total_gain_v1`
   versus the marginal joint-minus-active-only score. Contract requires
   resolution before numerical lock.

---

## Traps

- **`pytest test` runs zero tests.** The 55 collection errors abort the
  session. Always pass `--continue-on-collection-errors`.
- **Removing a parameter from `_run_hardcoded_adapt_vqe` silently drops
  values.** It does not raise. Until F2 is fixed, a green suite does not mean
  settings survived.
- **`output/` and `prompt-exports/` are gitignored** (`.gitignore:49`).
- **Thirteen worktrees exist** and several Codex branches were committed to on
  2026-08-24. Stage explicit paths only.
- **AST/word-boundary analysis has produced two wrong verdicts in this effort
  already** (three live cost symbols marked dead via closures; `route_identity`
  credited with 34 errors it does not cause). Grep, read the call site, run the
  test.
- **The contract's "Deliberately out of scope" list binds the refactor:** do
  not change optimizer tolerances, estimator accounting, forced-admission
  behavior, or cost normalization under the label of refactoring.

---

## Live findings _(append-only; anyone may add)_

Add dated entries. Name the entry you are correcting, if any.

### 2026-08-24 — Claude — full-suite failure census (SETTLED)

```
789 failed, 4789 passed, 18 skipped, 88 errors in 1233.23s (20:33)
peak RSS 1168 MB
```

88 errors = the 55 collection errors + 33 runtime errors.

**Most of those 789 failures are not defects.** Re-running the worst offenders
in isolation:

| file | in full suite | run alone | pollution |
|---|---|---|---|
| `test_ra_adapt_semantic_closure_routes.py` | 77 | **2** | 75 |
| `test_ra_adapt_facade.py` + `test_static_adapt_sr_snake_resolved_context.py` | 70 | **7** | 63 |
| `test_generic_static_benchmark.py` | 104 | **41** | 63 |

**251 reported, 50 real — ~80% is cross-test interference.** Tests share mutable
module-level state, so outcomes depend on execution order.

Corrects the earlier partial entry in this log, which reported "~16% of tests
fail". That figure is inflated and should not be quoted.

The suite is also slow: `test_ra_adapt_semantic_closure_routes.py` alone takes
8m16s for 92 tests. Slow plus order-dependent is why nobody runs it, which is
why no failure count existed.

**Consequence for the refactor:** the suite cannot serve as the acceptance
signal until isolation is fixed. Before trusting any test result, re-run the
affected files alone and compare. A file's failure count in a full run is not
evidence about that file.

### 2026-08-24 — Claude — the gates are a scaffold, not a standing rule

Correcting an overstatement I made earlier in this effort. I described the
behavioral contract as *forbidding* changes such as adopting Qiskit Nature for
operator construction. It does not. It says such a change "requires its own
named profile, rerun, and manuscript review." Those are different claims and I
collapsed them into a prohibition.

Framing to carry forward:

- "The accepted generator sequence must not move" is valuable **while code is
  being moved and no numerical change is intended** — it is the cheap way to
  distinguish "I refactored" from "I changed the science and did not notice",
  which is the author's original drift symptom.
- It has near-zero value as a **permanent** rule, and as a standing ban on
  adopting better libraries it costs more than it returns.
- Scope the gates to the refactor window. Do not treat them as an acceptance
  criterion for all future work, and do not let them block a change the author
  has decided to make deliberately.

### 2026-08-24 — Claude — one root cause behind three symptoms

The guard sprawl (F4), the settings drift (F3), and the ~80% test pollution
above are the same defect in three places: **the same fact is stored in several
places, and each copy needs a check that it still agrees with the others.**

- 190 guards asserting two representations agree
- 31 settings dicts across 5 roots, so no single place states what a profile is
- tests sharing mutable module state, so order changes outcomes

Adding checks does not fix this, and deleting checks is unsafe because they are
catching real drift. Collapsing the duplicate representations fixes it, and the
checks then have nothing left to disagree about.

---

### 2026-08-24 — Claude — CORRECTION: Bundle 3 is probably re-derivable

Corrects "Evidence state" above, which says Bundle 3 "cannot currently be
re-derived from its recorded inputs, so contract Gate 1 is not satisfiable for
it." **That was over-claimed.** What is gone are *derived intermediates* from a
dead scratchpad, not the upstream evidence.

Measured:

- The 18 Bundle-3 run archives are intact and hash-recorded. Each carries
  `run/checkpoints/current.json`, `run/result/estimator_ledger.json`, **and a
  full `source/` snapshot of the pipeline that produced it**.
- The renderer named in the manifest, `results_tooling/build_variant_pages_v7.py`,
  **exists at the exact recorded tooling commit `551a9fe0`**:

```bash
git show 551a9fe0:chtc/paper_i_ra_adapt_repair_20260727/results_tooling/build_variant_pages_v7.py
```

So the path archives → renderer → `kstar_tables.json` appears intact.
`kstar_tables.json` is an aggregate reporting table (keys: `kstar`, `aavqe`,
`b1`, `b3`, `live_star`, `spot_check`, by regime), i.e. a *derived* artifact.

**Status: plausibly recoverable, not confirmed.** One experiment settles it —
regenerate with the recorded renderer and compare against
`bundle3_final_results_manifest.json`. Do not describe Bundle 3 as lost
evidence until that experiment has been run and failed.

### 2026-08-24 — Claude — CORRECTION: `pipelines/hardcoded/` was not deleted

Corrects "What the 55 collection errors actually are" above, which reads as if
the package was removed. **The package still exists** with 8 modules
(`adapt_circuit_execution.py`, `hh_continuation_pruning.py`,
`hh_staged_noise*.py`, and others). What is missing are ~12 specific modules
the tests import (`hubbard_pipeline`, `hh_pareto_tracking`, `hh_staged_cli_args`,
`hh_time_dynamics_spectra`, …). Last touched by `6442fbb5`
"Replace Snake snapshot with RA". This reads as a partial deliberate retirement
that left its tests behind — still an author question, but a narrower one.

---

## Execution log _(Codex-owned)_

Append one entry per increment: goal, commands run, measured result, and
whether the stop condition triggered.
