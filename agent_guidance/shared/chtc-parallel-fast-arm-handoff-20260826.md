# CHTC handoff — submit parallel fast arms, do NOT stop the running ones

**For:** the agent operating the Paper-I CHTC campaign.
**From:** the refactor session on `paper-ii-exchange-selector`.
**Date:** 2026-08-26.

---

## 1. Anchors

| fact | value |
|---|---|
| branch | `paper-ii-exchange-selector` |
| commit to submit from | `20b6cedf` |
| pushed | yes — `origin/paper-ii-exchange-selector` is at `20b6cedf` |
| `adapt_pipeline.py` | 30,317 lines (was 69,723) |
| parity gate | `test/test_ra_adapt_refactor_parity.py` — green |
| physics gate | `test/test_ra_adapt_physics_invariants.py` — green |

Verify before doing anything:

```bash
git -C <checkout> fetch origin paper-ii-exchange-selector
git -C <checkout> log --oneline -1 origin/paper-ii-exchange-selector   # expect 20b6cedf
```

## 2. Scope — what NOT to do

**Do not stop, kill, hold, or remove any currently running job.** The in-flight
b3/b9 arms represent days of compute and they keep running to completion. This
handoff adds arms; it removes nothing.

**Do not attempt a resume or continuation of a running arm onto this commit.**
It will be refused, by design and correctly — see §4. Do not try to force it,
and do not "fix" the resume guard to make it pass.

**Do not promote, demote, or re-point any evidence.** Adoption is the author's
decision.

## 3. The objective

Submit **new, from-zero arms** of the same b3/b9 cells at `20b6cedf`, running
alongside the existing ones. The new arms do strictly less work per round, so
they may finish first despite starting at round 0. Whichever lands first is
usable; if the old arms land first, nothing was lost.

This is a race, not a replacement.

## 4. Why the new arms are a separate identity (read this before wiring manifests)

The new arms carry a **different route-contract sha256** from the running ones.
That is correct and expected, not a defect: the executed code differs.

Measured: the current tree reproduces **none** of the four contract digests
recorded across `chtc/` (`d14d582e…` 2264 files, `fd5ec3fa…` 705,
`eecd11fc…` 208, `8d809064…`). The resume guard at
`pipelines/static_adapt/sr_snake/_resume.py:2158` compares that digest and
raises `CanonicalResumeError` on mismatch, which is why §2 forbids continuation.

Consequence for reporting: the retrieved partial results from the old arms and
the results from the new arms are **two arms**, not one continued trajectory.
Do not splice them into a single cell. Register the new arms with their own
identity and let the author decide which to adopt.

## 5. What actually changed, and what is proven

**Proven, load-independent (counted, not timed):**

- The generator sector audit was recomputed **687 times when only 166 distinct
  audits exist** — 76% were re-derivations. Now memoized.
  (`pipelines/static_adapt/sector_invariants.py`, merged from
  `perf/sector-audit-memoization`.)
- Verdicts **bit-identical across 132 audit rows**, energy
  `5.876936059036121` in every run, regression failure set byte-identical.
- The candidate-record cache does one deep copy per path instead of two.

**NOT proven — do not quote a speedup number:**

- The originating session **retracted** its 2.03x cold / 1.19x warm claim. Every
  one of those timings was taken with `cProfile` attached, which roughly doubles
  the run (the same run unprofiled was 40.8 s, not 79 s). The clean
  re-measurement ran at load average 25 on 8 cores and could not separate the
  two states (baseline 95.1/62.1 s vs changed 56.2/64.4 s).
- So: **strictly less work per round, magnitude unquantified.** If you want a
  defensible number, that needs a quiet machine or a dedicated CHTC arm — no
  profiler, ≥5 interleaved repeats per state, medians and spread.

**Refactor (separate from the speedup):**

- `_run_hardcoded_adapt_vqe` (39,386 lines) deleted. It was the legacy
  comparator engine and was never on the RA path — the parity gate reproduces a
  **bit-identical accepted trajectory** across the deletion. This does not make
  anything faster; it is not expected to change any number.
- The RA route is no longer pinned to `full_meta`. Adapters accept an optional
  `pool_key`, validated fail-closed against `problem.admissible_pool_keys`.
  **Leave it unset for these arms** — the default is the canonical identity.

## 6. Evidence standard for the new arms

Before submitting, on the submit host:

```bash
python3 -m pytest test/test_ra_adapt_refactor_parity.py \
                  test/test_ra_adapt_physics_invariants.py \
                  test/test_static_adapt_sector_invariants.py -q
```

All must pass. If parity fails, stop and report — do not submit.

After the first new arm produces its first accepted rounds, before scaling out:

1. Confirm the accepted operator sequence and energies match the corresponding
   rounds of the running arm for the same cell. They should agree; the trajectory
   is not supposed to change.
2. If they disagree at any round, **stop submitting further arms and report the
   first disagreeing round with both energies.** That is a real finding and takes
   priority over throughput.

## 7. Traps

- **`git stash` is unsafe in this checkout.** The stash stack is shared across
  worktrees; a `stash pop` here picked up another agent's stash and conflicted
  four files (2026-08-25). Use `git worktree add --detach <rev>` for baselines.
- **Never `git add -A`.** Stage explicit paths only. Prior incident `ba7f2ac9`.
- **Memory budget is 10 GB aggregate** across all agent-launched work on the
  16 GB machine; wrap heavy processes in `pipelines/shell/ram_guard.py
  --limit-mb 8000`, one heavy job at a time locally. Big compute goes to CHTC.
  A prior session breached this by running three 27-minute suites alongside
  three production arms.
- **Disk is tight** — ~26 GB free of 460 GB, `.git` is 30 GB and the worktrees
  are ~50 GB. A `git worktree add` already failed once mid-command for space.

## 8. Report back

One block, plain text:

```
## New arms submitted
cells:            <cell ids>
commit:           20b6cedf
cluster ids:      <ids>
old arms:         still running / completed  (must not be "stopped")

## First-rounds agreement check
cell:             <cell id>
rounds compared:  <n>
agreement:        exact / first disagreement at round <k> (old <E>, new <E>)

## Anything refused or surprising
```

Files to edit: None (submission only; no repository changes are requested).
