# review-projective-face (2026-09-15)

Independent review of the round-3 implementer `projective-face`.  Nothing was
edited; every check below is a rerun, a re-solve, or a read.

Verdict: **pass**.  Every headline number reproduced, the default path is
bitwise unchanged under the final working tree, and no file outside the
implementer's ownership or the section-7 allowlist was touched.

## Scope

`git status --short` in the worktree lists twelve modified and eight untracked
files.  Ownership, read from each round-3 report's own "files owned" section:

| file | owner |
|---|---|
| `paper_5/src/paper5/stability/projected_apcm.py` | projective-face |
| `paper_5/src/paper5/stability/apcm_moment_projection.py` | projective-face |
| `paper_5/src/paper5/stability/apcm_projective_event_guard_analysis.py` | projective-face |
| `paper_5/tests/test_projected_apcm.py` | projective-face |
| `paper_5/tests/test_apcm_moment_projection.py` | projective-face |
| `paper_5/tests/test_apcm_projective_event_guard.py` | projective-face |
| `agent_guidance/paper-v/overnight/projective-face-20260915.md` | projective-face |
| `paper_5/src/paper5/stability/apcm_carried_witness.py` | bundle-rank-seed |
| `paper_5/src/paper5/stability/apcm_carried_witness_analysis.py` | bundle-rank-seed |
| `paper_5/tests/test_apcm_carried_witness.py` | bundle-rank-seed |
| `paper_5/src/paper5/stability/cone_long_horizon.py` | joint-gram-t20 |
| `paper_5/tests/test_cone_long_horizon.py` | joint-gram-t20 |
| `pipelines/open_dynamics/run_paper_v_energy_neutral_barrier_long_horizon.py` | joint-gram-t20 |
| `paper_5/src/paper5/stability/apcm_outer_selector_formulations.py` (new) | outer-selector-formulation |
| `paper_5/tests/test_apcm_outer_selector_formulations.py` (new) | outer-selector-formulation |
| the remaining untracked `agent_guidance/paper-v/overnight/*.md` | the agent each is named for |

Every path lies in the section-7 allowlist (`paper_5/**`,
`agent_guidance/paper-v/**`, `pipelines/open_dynamics/**`).  No Paper I--IV
path, no `src/`, `test/`, `pipelines/static_adapt` or `pipelines/time_dynamics`
entry.  `find /Users/lordskyler/Holstein_test/paper_5
/Users/lordskyler/Holstein_test/agent_guidance/paper-v -newermt "2026-09-14
20:00"` returns nothing, so the main checkout was not written to.

The three files the implementer was told not to edit (`apcm_carried_witness.py`,
`cone_correction.py`, `cone_long_horizon.py`, and `matrix_reference.py`) carry
no projective-face content: the added lines of `apcm_carried_witness.py` match
`bundle_seed` / `bundle_growth` / `bundle_interior` 36 times and match
`face_policy`, `stage_failure` and `completion_retry` zero times.
`apcm_projective_guard.py` and `adaptive_positive_moment.py` are theirs and are
unmodified.  No scope violation.

## Tests, rerun

| gate | reported | this review |
|---|---|---|
| the four owned test files together | 63 passed in 87.4 s | 63 passed in 86.12 s |
| test counts at `HEAD` (d189c50a/44b97966) | 44 | 9 + 4 + 20 + 11 = 44; now 15 + 11 + 26 + 11 = 63 (+6 / +7 / +6 / 0) |
| whole `paper_5` suite under `ram_guard --limit-mb 4500` | 428 passed, 9 warnings, 189.9 s, peak 1664 MB | 428 passed, 9 warnings, 177.01 s, peak 2004 MB |

The wall time and the guard peak differ with machine load; the counts are
identical.  The forced-failure test
`test_driver_saves_the_horizon_a_failed_rollout_reached` is a real gate: it
monkeypatches `SymmetryReducedAPCMGeometry.retract_stage` to raise on its
fourth call, then asserts exit status 1, one accepted step, the two-sample
partial trajectory with its exact reference, the failure block and
`stage_failure.npz`.  One line was deleted across the three test diffs (an
import that moved); no existing assertion was weakened.

## Parity, re-measured rather than read

I did not take the implementer's gate artifact on trust, because its
`summary.json` predates the retry-ladder fields (`stage_policy` there has five
keys, not seven), so their bitwise gate ran against an intermediate state of
their own code.  I reran both paths under the final working tree.

Default projective path, `t = 0` to `0.05` at `h = 0.0025`, all flags at their
defaults, phonon cutoff 16, 15 entrance keys, strong coupling, exact-correlated
preparation.  Artifact:
`output/local_runs/paper_v_apcm_projective_reviewer_default_recheck_t005_20260915_v1/`.
Every array of `autonomous_trajectory.npz` (times, states, archive
coordinates, both eigenvalue series, stage retractions, accepted substeps,
triggered stages) and of `trajectory.npz` (including the exact archive
coordinates) is equal under `np.array_equal` to round 2's
`paper_v_apcm_projective_ablation_baseline_recheck_t005_20260915_v1`, and to
the implementer's `..._facegate_default_...` run.  All-coordinate scalar RMS
against the exact cutoff-16 driven reference `7.931830738563173e-05`, 7 stage
retractions, minimum unshifted retained joint-Gram eigenvalue
`2.892080456917745e-05`, `initial_time` 0.0.

Carried-completion path under `--face-policy continuation`, `t = 0` to `0.5` at
`h = 0.00125`.  Artifact:
`output/local_runs/paper_v_apcm_projective_reviewer_facecontinuation_t05_20260915_v1/`.
Reaches `t = 0.5`, every array equal under `np.array_equal` to the
implementer's `..._carried_facecontinuation_...` run, RMS `9.547375173e-04`,
7 stage retractions, 61 faces continued, 1,176 carried lifts accepted and 36
declined.

Also confirmed by reading the artifacts: continuation and inflated agree array
for array; the committed `refuse` path's 92 stored samples agree array for
array with the continuation run's first 92.  Recomputing the score myself from
`trajectory.npz` (difference minus its value at the first sample, then the
root mean square over all 31 closed scalar coordinates) gives
`1.2875122500666604e-05` over `t = 0` to `0.05` for all four carried rows, and
`1.288e-05`, `4.504e-05`, `1.710e-04`, `3.661e-04`, `6.315e-04`, `9.547e-04`
up to `t = 0.05, 0.1, 0.2, 0.3, 0.4, 0.5`, reproducing the report's growth
table including every blockwise entry.

## The central claim, re-solved independently

The report's result is that which declared setting certifies the proximal
log-det completion is a property of the stage.  I re-solved the declared
completion offline at each of the three stages saved by (a), from each run's
own `stage_failure.npz`, with the declared warm frontier:

| stage saved by | declared | shift 1e-8 | shift 1e-7 | weight 1e-1 | weight 1e-2 |
|---|---|---|---|---|---|
| `t = 5.08626953125` (default run) | refused, 40 Newton iterations, lift `9.297881e-06` | certified, 4 | certified, 5 | certified, 11 | certified, 23 |
| `t = 5.104736328125` (shift-1e-8 run) | certified, 4 | refused (conic Newton descent) | certified, 5 | certified, 8 | certified, 20 |
| `t = 4.831923828125` (weight-1e-2 run) | certified, 10 | certified, 10 | certified, 10 | certified, 11 | refused, lift `3.386031e-10` |

The reversal the report rests on is real and reproduces exactly, including the
lift minima it quotes.

## Defaults and the named options

Read from the diff, confirmed by the reruns above.

- `APCMProjectionSettings.face_policy` defaults to `"refuse"`; the two
  tolerances are read only by the policy naming them.  Under `"refuse"`
  `_kernel_basis` adds `0.0` to the eigenvalues and takes the same branch, so
  the arithmetic is unchanged.
- `ProjectedAPCMStagePolicy.completion_retry_shifts` / `_weights` default to
  empty, and `is_default` still routes containment to the original single
  `self.extension.complete(...)` call, so the ladder cannot be consulted on a
  default rollout at all.
- `--completion-logdet-shift` / `--completion-logdet-weight` default to
  `APCMExtensionSettings.logdet_shift = 1e-10` and `logdet_weight = 1.0`.
- `initial_time` defaults to `0.0` and `times` becomes
  `0.0 + np.linspace(0.0, final_time, n)`, which is bit for bit the previous
  array.
- The driver now passes an explicit geometry; with default flags it is
  `SymmetryReducedAPCMGeometry(extension, APCMProjectionSettings())`, which is
  what the model constructed for itself before.
- The failure-path save is unconditional but additive: it writes artifacts
  where none were written and leaves the succeeding path's numbers alone,
  which the rerun above shows.

No strict or balanced default, acceptance tolerance, envelope or floor was
moved.

## Issues (none blocking)

1. **The "stage retractions" column has no stated definition.**  The report's
   own tables carry `feasibility.total_stage_retractions` from `summary.json`,
   which is the sum of the per-accepted-step counter, while the framing quotes
   round 1's "1,068 stage retractions" for the recorded `t = 0` to `4` run,
   which is the number of stage records in `outer_guard_stages.jsonl` whose
   `stage_retraction_applied` is true.  For that run the two are `1,041` and
   `1,068`.  No comparison in the report crosses the two, so no number is
   wrong; the column should say which counter it is.
2. **Three quantitative claims have no saved artifact.**  The completion sweep
   directory holds only the `t = 5.08627` stage; the sweeps at
   `t = 5.104736328125` and `t = 4.831923828125`, the 16-endpoint cold-solve
   certification over the `t = 4` restart ("certifies 14"), and the
   11-endpoint carried-lift certification over the carried rollout to
   `t = 0.5` ("certifies only 6") were not written out.  I re-solved the two
   extra stages myself and they reproduce exactly (table above); the two
   endpoint-sampling counts remain unverifiable from artifacts and should be
   re-run into a directory before any of them is quoted outside this report.
3. **The implementer's own bitwise gate ran against an intermediate state of
   their code** (the gate artifact's `stage_policy` lacks the retry-ladder
   fields).  Closed by the two reruns above under the final tree; worth
   stating because the same pattern would hide a real regression in a round
   where the last change did touch the default path.
4. Cosmetic: the report file's gate table says the four owned test files took
   `93.0 s` and the structured report says `87.4 s`.

## What I ran

```
git status --short ; git diff --stat
pytest -q tests/test_projected_apcm.py tests/test_apcm_projective_guard.py \
        tests/test_apcm_projective_event_guard.py tests/test_apcm_moment_projection.py
ram_guard --limit-mb 4500 -- pytest -q tests
ram_guard --limit-mb 4500 -- python -m paper5.stability.apcm_projective_event_guard_analysis \
        --mode rollout --output-directory .../paper_v_apcm_projective_reviewer_default_recheck_t005_20260915_v1
ram_guard --limit-mb 4500 -- python -m paper5.stability.apcm_projective_event_guard_analysis \
        --mode rollout --carried-completion --face-policy continuation \
        --final-time 0.5 --time-step 0.00125 \
        --output-directory .../paper_v_apcm_projective_reviewer_facecontinuation_t05_20260915_v1
```

plus offline array comparisons and three offline re-solves of the declared
completion at the saved failing stages.  Peak resident of the reruns: 297 MB
and 296 MB under the 4500 MB guard; the suite peaked at 2004 MB.  One heavy
process at a time.
