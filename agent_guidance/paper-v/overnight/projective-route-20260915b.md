# projective-route (2026-09-15, afternoon)

Persistent agent for the Paper V projective route, branch
`claude/paper-v-routes-20260914`, worktree
`Holstein_test.worktrees/claude-paper-v-routes-20260914`.  Task 1: wire the
repaired one-shell outer selector into the event-triggered guard (a),
diagnose the post-pulse wall at `t = 5.09` to `5.24` by origin (b), and act
on that diagnosis (c).

Configuration common to every rollout below unless stated otherwise: fixed
dictionary APCM, 15 entrance keys, SSPRK(3,3), `h = 0.0025`, phonon cutoff
16, `lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude 1, containment before
every right-hand-side call, event-triggered one-shell outer guard at its
declared floors (`event_gram_floor 1e-8`, `event_gram_hard_floor 1e-3`,
`event_coupling_threshold 1e-2`), scored against the exact cutoff-16 driven
reference regenerated in process from `t = 0`.

Status: complete for task 1 (a), (b) and (c).

## The result, first

**The post-pulse wall is accumulated error, not the frozen box.**  The exact
cutoff-16 driven state at `t = 5.0`, contracted into the projected APCM state,
extends inside the frozen box and its declared proximal log-det completion
certifies with no retry at all (14 interior Newton iterations, KKT backward
error `2.623e-09`, minimum scaled eigenvalue `-6.177e-15`, box envelope ratio
`0.246332` against the frozen envelope `1`).  Continued from that state with
the carried lift, the face carried across stages and the declared completion
retry ladder, the same integrator at the same step reached `t = 5.985` in 394
steps with **2 stage retractions**, stopped by its 1,080 s wall budget and not
by a refusal, at all-coordinate scalar RMS `6.486478e-03` against the exact
cutoff-16 reference over `t = 5.0` to `5.985`.  Every rollout that arrives in
that interval carrying the error accumulated over the driven interval stops
between `t = 4.83` and `t = 5.235` after 25 to 269 stage retractions.

Halving the step from the stored `t = 4` state with the declared retry
ladder does not move that wall: it reaches `t = 5.1075` against the
`h = 0.0025` run's `t = 5.12`, in `1,114.8 s` against `2,583.8 s`, with 6
stage retractions against 109.  The cost of the wall is a step-size artifact;
the wall itself is the accumulated error, and it was acquired before `t = 4`.

The repaired one-shell outer selector is now a named option of the
event-triggered guard (`outer_target_solver="tightened_retry"`), with the
recorded selector as the default and the injected-selector seam unchanged.

## (a) The repaired selector is a named option of the event-triggered guard

Round 3 measured that the recorded one-shell outer target program, run at the
recorded routine's own tightened numerical profile (conic tolerance `1e-10`,
`max_iter` 500, 20 iterative refinement passes at `1e-14`, equilibration
off), returns an accepted witness at relative duality gap `7.4634e-07` at the
cutoff-16 correlated preparation and `7.5469e-07` at a forced-trigger stage's
own lower moments, against the unchanged `1e-6` exploratory acceptance, and
that the recorded ladder never reaches that profile because its retry trigger
set `{AlmostSolved, InsufficientProgress, MaxIterations}`
(`apcm_projective_guard.py:1493`) omits `NumericalError`.  That callable,
`apcm_outer_selector_formulations.solve_outer_target_with_tightened_retry`,
was not reachable from a rollout.  It is now.

Two seams, both with the recorded path as the default.

1. `select_outer_feasible_projective_guard(..., target_solver=None)`
   (`apcm_projective_guard.py`).  `None`, the default, is the recorded
   `_solve_explicit_target_qp`.  A supplied callable is handed the same four
   operands the recorded solve takes --- the one-shell outer extension, the
   stage's own lower moments, the invariant target image readout embedded in
   the outer frontier, and the whitened target metric --- with `boxed` and
   `frontier_weight`, and returns the `GuardConicResult` shape with its
   acceptance already decided.  Everything downstream of the target stage is
   untouched: the minimum-norm lift, the restriction to the current frontier,
   and the invariant-image agreement test.  The four named options of the
   recorded solve that a supplied callable does not carry
   (`accept_first_provisional`, `frontier_metric`, `warm_refinement_passes`,
   `exploratory_acceptance_tolerance`) are refused rather than silently
   dropped.
2. `ProjectiveOuterEventGuardSettings.outer_target_solver`
   (`projected_apcm.py`), `"recorded"` by default, `"tightened_retry"` the
   named alternative, with `outer_target_ladder` (first pass then the
   tightened profile, default `True`) and
   `outer_target_time_limit_seconds`.  `resolve_outer_target_selector`
   returns `select_outer_feasible_projective_guard` itself at the default ---
   the same object the guard has always called --- and at
   `"tightened_retry"` returns that selector with the target stage bound to
   the tightened-retry callable.  The guard's injected-`selector` seam is
   unchanged and still wins over the setting.
3. Driver flags `--outer-target-solver`, `--outer-target-single-pass` and
   `--outer-target-time-limit-seconds`; the choice is recorded in
   `summary.json` through `guard_settings`.

## (b) Where the post-pulse wall comes from

The question the run answers: the rollouts that stop between `t = 5.085` and
`t = 5.235` all reach that interval carrying the error they accumulated over
the driven interval, so a refusal there can be a property of the post-pulse
exact state's extension inside the frozen box, or a property of the
accumulated moment error.  Starting the same integrator from the *exact*
cutoff-16 state at `t = 5.0` separates the two: that start carries no
accumulated moment error at all.

The named option that makes this possible is `--initial-exact-time`
(`apcm_projective_event_guard_analysis.py`, default `None`).  The exact
truncated wavefunction is propagated from the prepared ground state at
`t = 0` to the declared time and contracted there into the same two
representations `prepare_projected_apcm_initial_state` contracts the
preparation into at `t = 0`: the retained matrix moments, and the
relative-mode entrance moments normal ordered in the canonical CCR algebra,
which is the ordering APCM initialization requires.  The declared proximal
log-det completion at those lower moments is the initial lift; when it is not
certified the declared completion retry ladder
(`--completion-retry-shifts` / `--completion-retry-weights`) is tried in
order and the first certified result is taken, and the initialization fails
closed if none is.

### The contraction itself already answers half the question

Contracting the exact cutoff-16 driven state at `t = 5.0` (strong coupling,
15 entrance keys, `maximum_step 0.01` on the exact propagation, `1.4 s`):

| quantity | value |
|---|---:|
| declared proximal log-det completion at the exact lower moments | certified, interior Newton stationarity, 14 Newton iterations, `reduced_gradient 2.623e-09` |
| minimum scaled eigenvalue of the completed extension | `-6.177e-15` |
| box envelope ratio of the completed frontier, against the frozen box `1` | `0.246332` |
| minimum eigenvalue of the retained joint moment matrix | `1.882525e-05` |
| exact state norm | `1.0000000000000016` |

So the exact state after the drive ends *does* extend inside the frozen box,
and its declared completion certifies without any retry: the post-pulse
degeneracy is not a property of the exact state at `t = 5.0`.

### The rollout from the exact `t = 5.0` state

Continued from that contraction toward `t = 8.0` at `h = 0.0025` with
`--carried-completion --face-policy continuation --completion-retry-shifts
1e-8 1e-7 --completion-retry-weights 1e-1` under an 1,080 s wall budget
(artifact
`output/local_runs/paper_v_apcm_projective_exact_t5_carried_facecont_t8_20260915_v1/`).
Progress at the driver's own sample points, every one with **zero stage
retractions** and one accepted substep per step:

| `t` | minimum retained joint-Gram eigenvalue | minimum scaled positive-extension eigenvalue | stage retractions |
|---:|---:|---:|---:|
| 5.0025 | `1.883e-05` | `-1.557e-15` | 0 |
| 5.150 | `5.038e-05` | `2.973e-07` | 0 |
| 5.300 | `6.824e-05` | `1.095e-06` | 0 |
| 5.450 | `6.825e-05` | `2.259e-07` | 0 |
| 5.600 | `6.825e-05` | `1.998e-06` | 0 |
| 5.750 | `6.825e-05` | `1.998e-06` | 0 |
| 5.900 | `6.826e-05` | `1.998e-06` | 0 |

Against the four round-3 continuations from the *stored* `t = 4` state, which
stop at `t = 4.83` (proximal weight `1e-2`, 25 retractions), `5.085` (the
committed path, 122), `5.0975` (carried completion, 3), `5.1025` (PSD
inflation `1e-8`, 216) and `5.235` (carried completion with the face carried,
269 retractions, stopped by its budget).  The exact start passes all five
walls without a single retraction.

Reading: after the drive ends the fixed dictionary's positive extension does
not approach a face of the cone *because the drive ended*.  It approaches it
because the propagated state has drifted from the exact state
(`2.5037e-01` in Euclidean norm over the 31 closed scalar coordinates at
`t = 4`, round 3's figure), and it is that drifted state, not the post-pulse
physics, whose completion stops certifying.  The frozen box is not the
constraint that binds: the exact state at `t = 5.0` sits at envelope ratio
`0.246332` inside it.

The run was stopped by its budget, not by a refusal:
`wall_budget_exceeded after 394 of 1200 completed steps`.

| quantity | value |
|---|---:|
| horizon reached | `t = 5.985`, 394 accepted steps of the 1,200 that `t = 8` needs |
| why it stopped | the 1,080 s wall budget, tested between steps |
| all-coordinate scalar RMS against the exact cutoff-16 reference over `t = 5.0` to `5.985` (31 closed scalar coordinates) | `6.486478e-03` |
| the same with the `t = 5` offset subtracted | `6.486478e-03` (the offset is `7.339e-15` in Euclidean norm, which is what the exact start means) |
| blockwise scalar RMS, rho / B / N / A / C | `7.4733e-03` / `7.1183e-04` / `5.1005e-04` / `1.4567e-03` / `8.9481e-03` |
| site-0 occupation RMS | `4.4477e-03` |
| static-energy RMS (the driver's own quantity, not the record's interaction energy) | `1.2183e-04` |
| stage retractions over the 394 steps | 2 |
| carried lifts accepted / declined | 611 / 571 |
| declared completion retry ladder at the stage lift: certified / exhausted | 63 / 31 |
| the same ladder inside a stage retraction: certified / exhausted | 9 / 0 |
| faces identified / continued / refused | 2,363 / 1 / 0 |
| outer guard stages evaluated / triggered / selector solves | 1,182 / 0 / 0 |
| minimum unshifted retained joint-Gram eigenvalue over the run | `1.882515e-05` |
| minimum scaled positive-extension eigenvalue over the run | `-2.330e-11` |
| autonomous wall time / peak resident | `1,081.9 s` / `409 MB` |

Reading: the same integrator, the same step, the same policy and the same
post-pulse physics run 0.75 time units past the wall every propagated-state
continuation hits, with 2 stage retractions instead of 122 to 269, and stop
only because the clock ran out.  The wall at `t = 5.085` to `5.235` is a
property of the state the rollout arrives with, not of the interval it
arrives in.

## (c) The step-halved rerun from the stored `t = 4` state

Because (b) says accumulated error, the declared action is the second branch:
rerun from the stored `t = 4` state with the declared completion retry ladder
at half the step.  Configuration: `--initial-state-file` the stored `t = 4`
trajectory of
`output/local_runs/paper_v_apcm_projective_event_guard_t4_h0025_cutoff16_20260914_v1/autonomous_trajectory.npz`,
`h = 0.00125`, `--completion-retry-shifts 1e-8 1e-7 --completion-retry-weights
1e-1 --face-policy continuation`, toward `t = 8` under a 1,020 s wall budget,
which is the same pair of options round 3 ran at `h = 0.0025` (reached
`t = 5.12` in 2,583.8 s with 109 stage retractions).  Artifact:
`output/local_runs/paper_v_apcm_projective_t4restart_ladder_h000125_t8_20260915_v1/`.

| quantity | this run, `h = 0.00125` | round 3, same options at `h = 0.0025` |
|---|---:|---:|
| horizon reached | `t = 5.1075`, 886 of the 3,200 steps `t = 8` needs | `t = 5.12`, 448 steps |
| why it stopped | the 1,020 s wall budget | the 2,400 s wall budget |
| stage retractions | 6 | 109 |
| declared retry ladder at the stage lift: certified / exhausted | 108 / 78 | 115 / 149 |
| the same ladder inside a stage retraction: certified / exhausted | 19 / 12 | not printed |
| dynamic all-coordinate scalar RMS against the exact cutoff-16 reference, `t = 4` offset subtracted | `4.477829e-02` over `t = 4` to `5.1075` | `4.532038e-02` over `t = 4` to `5.12` |
| the same without subtracting the inherited offset | `4.085900e-02` | not printed |
| inherited `t = 4` error, Euclidean norm over the 31 closed scalar coordinates | `2.503676e-01` | the same state, the same number |
| minimum unshifted retained joint-Gram eigenvalue / minimum scaled positive-extension eigenvalue | `7.605570e-04` / `2.411796e-06` | `3.0567e-03` / `2.262e-07` |
| faces identified / continued / refused | 5,418 / 0 / 0 | not printed |
| outer guard stages triggered / selector solves | 0 / 0 | 0 / 0 |
| autonomous wall time / peak resident | `1,114.8 s` / `517 MB` | `2,583.8 s` / not printed |

Readings.

1. **Halving the step buys the cost, not the horizon.**  The same two options
   at half the step reach essentially the same wall (`5.1075` against `5.12`)
   in `1,114.8 s` against `2,583.8 s`, with 6 stage retractions against 109.
   The retraction storm is a step-size artifact; the wall is not.
2. **The rate collapses at exactly `t = 5.0`.**  The run covered `t = 4` to
   `5.0` --- 800 steps --- in about 120 s, and the remaining 86 steps took the
   other 995 s.  The declared ladder was exhausted 78 times, all of them past
   `t = 5`.
3. **Refining the step after the error exists does not remove the error.**
   The `t = 4` state this run starts from already carries `2.503676e-01` in
   Euclidean norm over the 31 closed scalar coordinates, acquired before
   `t = 4`; halving the step from `t = 4` onward cannot undo it, and (b)
   shows it is that drift, not the post-pulse interval, that the completion
   stops certifying at.  A rollout that arrives at `t = 5` with the exact
   state runs through the same interval at 2 retractions per 394 steps.


## Impact analysis (main checkout index, read-only; index 4 commits behind
and the worktree is not indexed at all)

| symbol | verdict | callers, resolved by text search |
|---|---|---|
| `select_outer_feasible_projective_guard` | UNKNOWN -> LOW | `projected_apcm.py:1199` (the guard), `apcm_projective_event_guard_analysis.py:435` (the selector audit), `tests/test_apcm_projective_event_guard.py` |
| `ProjectiveOuterEventGuardSettings` | UNKNOWN -> LOW | `projected_apcm.py:1275` (the guard's own default), `apcm_projective_event_guard_analysis.py:1011`, `tests/test_apcm_projective_event_guard.py` |
| `resolve_outer_target_selector` | UNKNOWN (new symbol, absent from the index) | `projected_apcm.py` (the guard) and the tests |

No HIGH or CRITICAL verdict.  Both edited functions gained a keyword-only
argument or a dataclass field with the recorded value as its default; no
existing call site changes behaviour.

## Gates

| gate | result |
|---|---|
| the recorded projective path through the driver, `t = 0` to `0.05` at `h = 0.0025`, default everything | every array of `autonomous_trajectory.npz` equal under `np.array_equal` to round 3's `paper_v_apcm_projective_facegate_default_t005_h0025_20260915_v1`: times, states, archive coordinates, both eigenvalue series, stage retractions, accepted substeps, triggered stages.  Artifact: `output/local_runs/paper_v_apcm_projective_defaultgate_t005_h0025_20260915b_v1/` |
| the five test files this round touches or wraps (`test_apcm_projective_event_guard.py`, `test_projected_apcm.py`, `test_apcm_moment_projection.py`, `test_apcm_outer_selector_formulations.py`, `test_apcm_projective_guard.py`) | 85 passed in `95.73 s` under `ram_guard --limit-mb 4500`, peak `423 MB` |
| the default event guard still calls the recorded selector object | `test_the_default_event_guard_selector_is_the_recorded_selector` |
| an injected target solver changes only the target stage | `test_an_injected_target_solver_replaces_only_the_target_stage`: the relayed selection's target image, standardized values and objective are equal under `np.array_equal` to the declared selection's |
| the named options the seam cannot carry are refused | `test_a_supplied_target_solver_refuses_the_recorded_named_options` |
| the injected-`selector` seam still wins over the setting, and a fired stage carries the witness through the seam | `test_a_forced_trigger_stage_carries_the_witness_through_the_seam` |
| the exact start at a small time is the prepared state | `test_the_exact_start_contracts_the_state_the_preparation_agrees_with`: `7.86e-07` maximum coordinate difference at `t = 1e-3`, phonon cutoff 6 |

## Comparability in a shared worktree

Three other agents were editing `paper_5/src/paper5/stability/` during this
round.  `apcm_carried_witness.py` and `apcm_carried_witness_analysis.py`
(another agent's) were modified while the rollouts below ran;
`projected_apcm.py` imports `negative_mode_coupling` from the first of them,
which the event guard calls once per stage.  No stage triggered in any
rollout below, so no number here passes through the changed selector path,
but the coupling measure does.  `apcm_positive_extension.py`,
`apcm_moment_projection.py` and `matrix_reference.py` were unchanged
throughout.

## Kept

- `target_solver` on `select_outer_feasible_projective_guard`: one keyword,
  `None` by default, which is the only place the recorded target stage is
  chosen.  It is what makes any alternative outer target solve reachable from
  a rollout without copying the restriction logic.
- `ProjectiveOuterEventGuardSettings.outer_target_solver` with
  `outer_target_ladder` and `outer_target_time_limit_seconds`, and
  `resolve_outer_target_selector`, which returns the recorded selector object
  itself at the default.
- Driver flags `--outer-target-solver`, `--outer-target-single-pass`,
  `--outer-target-time-limit-seconds`.
- `--initial-exact-time` and `_exact_contracted_initialization`: the exact
  cutoff state at a declared time contracted into the projected APCM state
  and completed by the declared proximal log-det completion, with the
  declared completion retry ladder consulted only on a refusal.  This is the
  experiment that separates an accumulated-error failure from a
  representation failure, and it costs `1.4 s` at `t = 5`.
- `--phonon-envelope`, the declared box envelope of the completion, default
  `16.0` unchanged.

## Dropped

- Nothing measured this round was dropped.  The `--phonon-envelope` widening
  of the frozen box, which was the (c) branch prepared for a box-degeneracy
  verdict, is implemented but not exercised: (b) says the box is not what
  binds, so widening it would answer a question nobody asked.

## Blockers

1. **The wired selector has not yet solved one real triggered stage inside a
   rollout.**  The wiring is tested at the seam (an injected target solver
   reproduces the declared selection bitwise; a forced-trigger stage carries
   the witness through it) and the tightened-retry callable is tested on the
   small two-key outer cone, but the measured `75 s` per solve at the
   cutoff-16 dictionary was not re-run through
   `ProjectiveOuterEventGuard.contain_stage` this round, for the compute
   budget.  The script is
   `output/local_runs/paper_v_apcm_projective_forced_trigger_selector_20260915_v1/`
   (to be run: one forced-trigger stage per named `outer_target_solver` at
   the cutoff-16 preparation, floors lowered to roundoff, about `50 s` for
   the recorded rejection and `125 s` for the accepted ladder, peak about
   `4 GB`).
2. **No trigger has ever fired on its own.**  Across every rollout measured
   in rounds 1 to 3 and here --- 1,182 stages on the exact `t = 5` run alone
   --- the endpoint unshifted 24-row Gram minimum never reached the declared
   floor, so the repaired selector is a capability the route does not yet
   exercise.  Whether the event floors should be tightened so that it does is
   a scientific-acceptance question, and the user's.
3. **Nothing measured here reaches `t = 6` from `t = 0`.**  The exact start
   reaches `t = 5.985` from `t = 5`; the step-halved restart reaches
   `t = 5.1075` from `t = 4`.  What (b) and (c) together say is that the
   remaining work is upstream of the pulse: a rollout that arrives at
   `t = 5` with materially less than `2.5e-01` of accumulated coordinate
   error should pass the wall, and that is an accuracy question over
   `t = 0` to `4`, not a conic one at `t = 5.1`.
4. **The accuracy past the pulse is `6.5e-03`, not `2.6e-02` and not
   `1.2e-05`.**  The `6.486478e-03` above is over `t = 5.0` to `5.985` from
   an exact start; it is not comparable with the `0.026049` of the `t = 0` to
   `4` run (which inherits the whole driven interval) nor with the
   `1.227728e-05` of `t = 0` to `0.05`.  A like-for-like post-pulse number
   for the propagated route needs a rollout that reaches `t = 5.985` from
   `t = 0`, which nothing does yet.

# projective-route, task 2 (2026-09-15, evening)

The full rollout from `t = 0` toward `t = 8` at `h = 0.0025` under the carried
lift, face continuation and the declared completion retry ladder, in three
20-minute segments chained through the saved state; then one 20-minute segment
at `h = 0.00125` from `t = 0`.  Configuration as declared at the top of this
file (fixed dictionary APCM, 15 entrance keys, SSPRK(3,3), phonon cutoff 16,
`lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude 1, containment before every
right-hand-side call, event-triggered one-shell outer guard at its declared
floors), plus `--carried-completion` (margin multiple 10), `--face-policy
continuation`, `--completion-retry-shifts 1e-8 1e-7`,
`--completion-retry-weights 1e-1`, scored against the exact cutoff-16 driven
reference propagated in process from `t = 0` on the chained sample times.
Every count below is the run's own total, from `summary.json`
(`feasibility.total_stage_retractions`, `stage_policy_counters`,
`retraction_completion_ladder_counters`), not a last-step value.

## The result, first

**The rollout does not reach `t = 4`.**  Three chained segments, `3,182.0 s`
of autonomous wall in total, reached `t = 3.115` --- 1,246 accepted steps of
the 3,200 that `t = 8` needs --- and the third segment spent its whole
`1,080 s` budget advancing 21 steps, `0.0525` time units.  So there is no
`t = 4` coordinate error to set beside the default policy's `2.503676e-01`,
and the step-halving of part (b) is the measurement that was run instead.

**The cost wall near `t = 3.0` is not the carried lift's.**  The recorded
default policy pays it too, in the same place: `1,589.0 s` of its `3,229.8 s`
run --- 49 percent --- falls in `t = 3.00` to `3.45`, which is exactly the
interval where its endpoint unshifted 24-row Gram minimum is negative.  The
difference is that the default policy comes out the other side at `t = 3.45`
and finishes `t = 4`, while the carried lift makes each stage inside the
episode about four times more expensive and the chain never got out.

**Over the interval both reach, the carried-lift policy is 4.6 percent more
accurate than the recorded default at the same step and about the same cost.**
All-coordinate scalar RMS against the exact cutoff-16 reference on the 31
closed scalar coordinates, `t = 0` to `3.115`, both at `h = 0.0025`, both
1,246 accepted steps: `1.936806e-02` for the chain against `2.030594e-02` for
the recorded default policy run.

## (a) The three segments

Artifacts:
`output/local_runs/paper_v_apcm_projective_t0_carried_ladder_facecont_t8_20260915c_seg1/`,
`..._seg2/`, `..._seg3/`; chain scoring
`output/local_runs/paper_v_projective_t0_chain_20260915c/chain_score.json`
(script `score_chain.py` in the same directory).

| quantity | segment 1 | segment 2 | segment 3 |
|---|---:|---:|---:|
| interval | `0` to `2.085` | `2.085` to `3.0625` | `3.0625` to `3.115` |
| accepted steps | 834 | 391 | 21 |
| why it stopped | `1,020 s` wall budget | `1,020 s` wall budget | `1,080 s` wall budget |
| autonomous wall / peak resident | `1,021.5 s` / `313 MB` | `1,057.2 s` / `272 MB` | `1,103.4 s` / `325 MB` |
| wall per accepted step | `1.225 s` | `2.703 s` | `52.54 s` |
| stage retractions, run total | 7 | 35 | 75 |
| declared retry ladder at the stage lift: certified / exhausted | 98 / 8 | 90 / 40 | 1 / 84 |
| the same ladder inside a stage retraction: certified / exhausted | 1 / 1 | 16 / 4 | 39 / 7 |
| carried lift accepted / rejected | 1,804 / 704 | 834 / 357 | 0 / 93 |
| faces identified / continued / refused | 5,008 / 8 / 0 | 2,379 / 3 / 0 | 186 / 0 / 0 |
| outer guard stages evaluated / triggered / selector solves | 2,507 / 0 / 0 | 1,187 / 0 / 0 | 86 / 0 / 0 |
| minimum scaled positive-extension eigenvalue | `-4.246e-13` | `1.069e-08` | `3.006e-07` |
| minimum eigenvalue of the retained joint moment matrix | `7.228e-06` | `7.002e-05` | `3.401e-04` |
| minimum endpoint unshifted 24-row Gram eigenvalue | `-3.137e-05` | `-1.139e-04` | `-1.240e-04` |

Chain totals over `t = 0` to `3.115`: 1,246 accepted steps, **117 stage
retractions** (`7 + 35 + 75`; the per-step series in
`autonomous_trajectory.npz` sums to the same 117 --- the stage ledger
`outer_guard_stages.jsonl` records 120 stages with a retraction applied, three
of them on stage attempts that are not part of an accepted step),
`3,182.0 s` of autonomous wall, peak resident `325 MB`, 3,780 outer guard
stages evaluated and **0 triggered**.

Accuracy, cumulative from `t = 0`, all-coordinate scalar RMS on the 31 closed
scalar coordinates against the exact cutoff-16 driven reference, blockwise in
the record's blocks:

| horizon | all-coordinate | rho | B | N | A | C | site-0 occupation | static energy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `t = 0` to `2.085` (segment 1) | `1.188372e-02` | `2.0755e-02` | `4.7508e-03` | `8.7958e-03` | `9.6950e-03` | `1.2311e-02` | `1.5711e-02` | `1.8018e-02` |
| `t = 0` to `3.0625` (segments 1--2) | `1.907967e-02` | `3.5555e-02` | `2.0140e-02` | `1.6201e-02` | `9.5747e-03` | `1.7465e-02` | `2.5082e-02` | `2.8609e-02` |
| `t = 0` to `3.115` (segments 1--3) | `1.936806e-02` | `3.6004e-02` | `2.0723e-02` | `1.6215e-02` | `9.5303e-03` | `1.7780e-02` | `2.5040e-02` | `2.8870e-02` |

`t = 0` to `4` is not reached, so that row does not exist and no number is
offered in its place.  The Euclidean norm of the 31 closed scalar coordinate
difference at the sample times the chain does reach, beside the recorded
default policy's at the same times:

| `t` | 0.25 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 2.75 | 3.0 | 3.115 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chained carried-lift rollout | `2.715e-03` | `1.112e-02` | `3.453e-02` | `8.058e-02` | `1.299e-01` | `1.604e-01` | `1.698e-01` | `1.742e-01` | `1.795e-01` |
| recorded default policy, same `h` | `3.581e-03` | `1.262e-02` | `3.921e-02` | `8.845e-02` | `1.380e-01` | `1.668e-01` | `1.749e-01` | `1.785e-01` | `1.831e-01` |

The recorded default policy's own `t = 4` value is `2.503676e-01`, reproduced
here from its stored trajectory
(`output/local_runs/paper_v_apcm_projective_event_guard_t4_h0025_cutoff16_20260914_v1/`)
to the printed digit; over `t = 0` to `3.115` that run took 701 of its 1,041
stage retractions and scored `2.030594e-02`.

## The cost wall near `t = 3.0`

What the per-step wall time does.  Summing the three stage wall times of each
step from `outer_guard_stages.jsonl`, the rolling 20-step median per-step wall
sits at about `1 s` from `t = 0.06` to `t = 3.0`, first exceeds `5 s` at
`t = 3.0419` and first exceeds `10 s` at `t = 3.0913`; the last 21 steps cost
`52.54 s` each against the first 834 steps' `1.225 s`, a 43-fold rise.  By
0.02-wide windows of the chain:

| window | stages | stage wall summed | stages with a retraction applied | minimum endpoint unshifted 24-row Gram eigenvalue |
|---|---:|---:|---:|---:|
| `2.90`--`3.00` (five windows) | 120 | `84.8 s` | 0 | `-2.388e-05` |
| `3.00`--`3.02` | 24 | `30.8 s` | 0 | `-4.977e-05` |
| `3.02`--`3.04` | 24 | `129.7 s` | 6 | `-1.139e-04` |
| `3.04`--`3.06` | 38 | `312.5 s` | 26 | `-1.099e-04` |
| `3.06`--`3.08` | 30 | `249.2 s` | 25 | `-1.135e-04` |
| `3.08`--`3.10` | 38 | `430.1 s` | 33 | `-1.178e-04` |
| `3.10`--`3.115` | 22 | `261.7 s` | 22 | `-1.240e-04` |

Why.  Three measured facts, in the order they act.

1. **The stage endpoint leaves the retained cone.**  The endpoint unshifted
   24-row Gram minimum, which is the quantity the event guard floors on, drifts
   from a mean of `+1.063e-07` over `t = 0` to `0.2` to `-6.719e-05` over
   `t = 3.0` to `3.2`, minimum `-1.240e-04`.  It never reaches the declared
   hard floor `1e-3`, and the negative-mode coupling measure peaks at
   `9.182e-04` against the declared threshold `1e-2`, so **no stage triggered
   the outer guard: 0 of 3,780**.  The containment's stage retraction, not the
   guard, is what repairs these stages.
2. **The carried lift stops being accepted, so every stage pays a cold
   completion.**  Carried lift accepted / rejected runs 1,804 / 704 over
   `t = 0` to `2.085`, 834 / 357 over `2.085` to `3.0625`, and **0 / 93** over
   `3.0625` to `3.115`: past `t ~ 3.06` the carried frontier never keeps the
   declared 10-conic-tolerance interior.  The declared completion retry ladder
   at the stage lift goes with it: 98 certified / 8 exhausted, then 90 / 40,
   then **1 / 84**.
3. **A retracted stage is two orders of magnitude more expensive than a quiet
   one.**  Over the whole chain a stage with a retraction applied costs
   `10.25 s` mean and `6.50 s` median (120 stages); a stage without one costs
   `0.459 s` mean and `0.0105 s` median (3,660 stages).  112 of the 176 stages
   past `t = 3.0` retract, against 8 of the 3,604 before it.

The same episode in the recorded default policy, which has no carried lift and
no face continuation, from its own stage ledger:

| window | stages | stage wall summed | stages with a retraction applied | mean endpoint unshifted 24-row Gram eigenvalue |
|---|---:|---:|---:|---:|
| `0.00`--`3.00` | 4,069 | `899.5 s` | 615 | `+1.042e-04` |
| `2.90`--`3.00` | 120 | `30.7 s` | 19 | `+1.430e-04` |
| `3.00`--`3.45` | 749 | `1,589.0 s` | 322 | `-2.848e-05` |
| `3.45`--`4.00` | 741 | `218.4 s` | 131 | `+1.741e-04` |

So the episode belongs to the trajectory at `t = 3.00` to `3.45`, where the
endpoint unshifted Gram minimum is negative, and not to the named policy: both
policies pay it, and the drive field difference
`2 sin(pi t / 4) exp(-t^2 / 2)` is `2.707e-01` at `t = 2`, `1.571e-02` at
`t = 3` and `1.674e-03` at `t = 3.5`, so the episode opens as the pulse
switches off.  What the carried lift changes is the price inside it: `8.034 s`
per stage over the chain's `t >= 3.0` stages, against `0.415 s` per stage over
its own `t < 3.0` and against the default policy's `2.121 s` per stage over
`t = 3.00` to `3.45` (and `0.221 s` over its own `t < 3.0`).  Both policies
slow by roughly the same factor entering the episode --- 19-fold for the
carried lift, 10-fold for the default --- but the carried lift enters it from a
cheaper regime and leaves it four times more expensive per stage, which is why
the default policy crosses the episode inside its budget and the chain does
not.

## (b) The step-halved rollout from `t = 0`

Because `t = 4` was not reached, the branch the task declares is the one that
ran: the same named policy from `t = 0` over the driven interval at
`h = 0.00125`, one 20-minute segment, `--final-time 4.0`,
`--wall-budget-seconds 1050`.  Artifact
`output/local_runs/paper_v_apcm_projective_t0_carried_ladder_facecont_h000125_20260915c_v1/`.

| quantity | `h = 0.00125` | the chain at `h = 0.0025` |
|---|---:|---:|
| horizon reached | `t = 1.7075`, 1,366 of the 3,200 steps `t = 4` needs | `t = 3.115`, 1,246 of the 3,200 steps `t = 8` needs |
| why it stopped | the `1,050 s` wall budget | three wall budgets, `3,182.0 s` in total |
| autonomous wall / peak resident | `1,052.1 s` / `360 MB` | `3,182.0 s` / `325 MB` |
| stage retractions, run total | 6 | 117 |
| declared retry ladder at the stage lift: certified / exhausted | 127 / 7 | 189 / 132 |
| the same ladder inside a stage retraction: certified / exhausted | 0 / 1 | 56 / 12 |
| carried lift accepted / rejected | 3,369 / 734 | 2,638 / 1,154 |
| faces identified / continued / refused | 8,196 / 10 / 0 | 7,573 / 11 / 0 |
| outer guard stages evaluated / triggered | 4,102 / 0 | 3,780 / 0 |
| minimum scaled positive-extension eigenvalue | `-2.657e-15` | `-4.246e-13` |
| minimum endpoint unshifted 24-row Gram eigenvalue | `-7.204e-06` | `-1.240e-04` |
| all-coordinate scalar RMS over `t = 0` to `1.7075` | `8.818433e-03` | `8.361075e-03` |

**Negative result: halving the step does not reduce the error, it raises it.**
Over the interval the refined run reaches, `t = 0` to `1.7075`, its
all-coordinate scalar RMS is `8.818433e-03` against the `h = 0.0025` chain's
`8.361075e-03` on the same 31 closed scalar coordinates --- 5.5 percent
larger.  The same at every shared sample time, as the Euclidean norm of the 31
closed scalar coordinate difference:

| `t` | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 |
|---|---:|---:|---:|---:|---:|---:|
| `h = 0.0025` | `2.7154e-03` | `1.1121e-02` | `2.1715e-02` | `3.4532e-02` | `5.7050e-02` | `8.0579e-02` |
| `h = 0.00125` | `3.1547e-03` | `1.1975e-02` | `2.2490e-02` | `3.7046e-02` | `6.1622e-02` | `8.4602e-02` |
| ratio | `1.162` | `1.077` | `1.036` | `1.073` | `1.080` | `1.050` |

The refined run is the cheaper and quieter of the two while it lasts --- 6
stage retractions against the chain's 117, `0.249 s` mean stage wall against
the chain's `0.770 s`, the carried lift accepted 3,369 times of 4,103 --- and
it is still less accurate.  Whatever the error over the driven interval is, it
is not the time discretization.

## (c) Does a single projective rollout pass `t = 6`?

**No, and no rollout from `t = 0` has passed `t = 4` except the recorded
default policy, which stops exactly there.**  Single rollouts from `t = 0`:

| rollout | horizon | wall | stage retractions, run total |
|---|---:|---:|---:|
| recorded default policy, `h = 0.0025` | `t = 4.0`, completed | `3,229.8 s` | 1,041 |
| carried lift + face continuation + retry ladder, `h = 0.0025` (this chain) | `t = 3.115` | `3,182.0 s` | 117 |
| the same at `h = 0.00125` | `t = 1.7075` | `1,052.1 s` | 6 |

and continuations from a stored `t = 4` state, every one of which stops inside
`t = 4.83` to `5.235`: `4.83` / 25, `5.085` / 122, `5.0975` / 3, `5.1025` /
216, `5.12` / 109, `5.1075` / 64 (at `h = 0.00125`), `5.235` / 269.  The only
run past `t = 5.235` is the one started from the *exact* cutoff-16 state at
`t = 5`, which reached `t = 5.985` in `1,081.9 s` with 31 stage retractions,
26 of them in its last 20 steps.

So two cost episodes stand between `t = 0` and `t = 6`: the one measured this
round at `t = 3.00` to `3.45`, which the default policy crosses in `1,589.0 s`
and the carried lift did not cross in three segments, and the one at `t = 5.09`
to `5.24`, which no propagated state has crossed at all.  More segments of the
same kind do not reach `t = 6`.

**What the error growth during the pulse implies for the smallest achievable
`t = 4` error under this dictionary.**  The Euclidean coordinate error grows
where the field is large and stops growing when it decays: `2.715e-03` at
`t = 0.25`, `3.453e-02` at `t = 1` (field difference `8.578e-01`, near the
largest value the pulse takes), `1.299e-01` at `t = 2` (field `2.707e-01`),
then only `4.42e-02`
more over the whole of `t = 2` to `3` (field `1.571e-02` at `t = 3`).  That
growth is invariant to the step --- (b) measures `1.04` to `1.16` times *more*
error at half the step at every shared time --- so it is the fixed 15-key
dictionary's closure defect accumulated while the drive pushes population out
of the retained moments, not integration error.

The consequence is that the `t = 4` error is not reachable by refining
anything inside this dictionary.  The two levers measured on this route move
it by a few percent in opposite directions: the named policy buys 4.6 percent
(`1.936806e-02` against `2.030594e-02` over `t = 0` to `3.115`), and halving
the step costs 5.5 percent.  The recorded default policy's `2.503676e-01` at
`t = 4` is therefore within a few percent of the floor for a 15-entrance-key
fixed dictionary at phonon cutoff 16, and halving it is a dictionary question
--- more entrance keys, or more retained relative-mode moments --- not a
projection question.  Nothing measured here supports the round-3 expectation
that a rollout arriving at `t = 5` with materially less than `2.5e-01` of
accumulated coordinate error is obtainable by running the same dictionary
more carefully.

## What the projective route still offers beyond the cheap hybrid

The cheap projection route's hybrid (lift re-selected every 16 steps, Dykstra
between) reaches `t = 8` at `5.77300e-02` in `195.7 s` with no post-pulse
wall; the projective route reached `t = 3.115` at `1.936806e-02` in
`3,182.0 s`, which is `1,021 s` per unit of time against the hybrid's
`24.5 s`, a factor of 42.  On accuracy over the intervals each reaches the two
are comparable (`1.94e-02` over `t = 0` to `3.115` here; the hybrid's
`2.68e-02` over `t = 0` to `4`), so the projective route is not buying
accuracy and it is certainly not buying horizon.  What it buys is a
certificate, and the certificate is the difference the artifacts show: at
`t = 8` the hybrid's state has minimum scaled 24-row extension eigenvalue
`-9.124e-06` and minimum unshifted retained joint Gram `-8.296e-07`, both
negative, so its repaired moments sit outside the cone they are meant to lie
in and the Dykstra repair is what put them close to it; the projective chain's
accepted steps carry a declared proximal log-det completion certified inside
the frozen box, minimum scaled positive-extension eigenvalue `-4.246e-13` /
`1.069e-08` / `3.006e-07` by segment and minimum retained joint moment
eigenvalue `7.228e-06` / `7.002e-05` / `3.401e-04`, all at or above roundoff.
The projective route also reports, at every stage, how far the trajectory
stands from the outer cone's face --- 3,780 stages this round, endpoint
unshifted 24-row Gram minimum never below `-1.240e-04` against the declared
hard floor `1e-3`, negative-mode coupling measure never above `9.182e-04`
against the declared threshold `1e-2`, 0 triggers --- a quantity the cheap
route does not compute.  If the deliverable is a long-horizon trajectory, the
cheap hybrid wins outright on every axis measured.  If the deliverable is the
statement that the closure stays inside both cones with a per-step
certificate, only the projective route produces it, and the price measured
this round is that it does not reach the end of the pulse.

## Summary

| quantity | value |
|---|---|
| horizon of the chained rollout from `t = 0`, `h = 0.0025`, carried lift + face continuation + declared retry ladder | `t = 3.115` in three segments, `3,182.0 s`, 1,246 of 3,200 steps |
| stage retractions, chain total (`summary.json`, per-step series agrees) | 117 (7, 35, 75 by segment) |
| declared completion retry ladder, chain total: at the stage lift / inside a retraction | 189 certified, 132 exhausted / 56 certified, 12 exhausted |
| all-coordinate scalar RMS against exact cutoff-16, 31 closed scalar coordinates, `t = 0` to `3.115` | `1.936806e-02` (rho `3.6004e-02`, B `2.0723e-02`, N `1.6215e-02`, A `9.5303e-03`, C `1.7780e-02`) |
| the same for the recorded default policy over the same interval and step | `2.030594e-02` |
| `t = 0` to `4` | not reached; no coordinate error at `t = 4` is offered beside the default policy's `2.503676e-01` |
| where the per-step wall time breaks | rolling 20-step median about `1 s` to `t = 3.0`, above `5 s` at `t = 3.0419`, above `10 s` at `t = 3.0913`; last 21 steps `52.54 s` each |
| why | the endpoint unshifted 24-row Gram minimum goes negative in `t = 3.00` to `3.45` (chain minimum `-1.240e-04`), the carried lift stops being accepted (0 of 93 past `t = 3.0625`), the declared ladder is exhausted 84 of 85 times, and a retracted stage costs `10.25 s` mean against `0.459 s` for a quiet one |
| whose wall it is | the trajectory's, not the policy's: the recorded default policy spends `1,589.0 s` of its `3,229.8 s` in the same `t = 3.00` to `3.45` window and then leaves it; the carried lift pays `8.034 s` per stage inside it against the default's `2.121 s` |
| step-halved rollout from `t = 0`, `h = 0.00125`, 1 segment | `t = 1.7075` in `1,052.1 s`, 6 stage retractions, ladder 127 / 7 at the stage lift; all-coordinate scalar RMS `8.818433e-03` over `t = 0` to `1.7075` against the chain's `8.361075e-03` --- 5.5 percent **worse** |
| does a single projective rollout pass `t = 6` | no; nothing from `t = 0` passes `t = 4` except the recorded default policy, which stops there, and no propagated state has crossed `t = 5.24` |
| smallest achievable `t = 4` error under this dictionary | the driven-interval error is step-invariant, so it is the fixed 15-key dictionary's closure defect; `2.503676e-01` is within a few percent of the floor, and halving it is a dictionary question, not a projection one |
| outer guard triggers | 0 of 3,780 stages (chain) and 0 of 4,102 (refined); endpoint unshifted 24-row Gram minimum `-1.240e-04` against the declared hard floor `1e-3`, coupling measure `9.182e-04` against the declared threshold `1e-2` |
| what the projective route offers beyond the cheap hybrid | a certified positive extension per accepted step (minimum scaled positive-extension eigenvalue `-4.246e-13` to `3.006e-07`, retained joint moment minimum `7.228e-06` to `3.401e-04`) where the hybrid's `t = 8` state carries `-9.124e-06` and `-8.296e-07`; and a measured distance to the outer cone's face at every stage.  Not accuracy, not cost (42 times the wall per unit time), not horizon |
| artifacts | `output/local_runs/paper_v_apcm_projective_t0_carried_ladder_facecont_t8_20260915c_seg{1,2,3}/`, `..._h000125_20260915c_v1/`, `output/local_runs/paper_v_projective_t0_chain_20260915c/` (`score_chain.py`, `score_common_horizon.py`, `compare_steps.py`, `chain_score.json`, `chain_score_seg12.json`, `default_policy_t4_score.json`, four logs) |
| code changed | none.  No default, acceptance rule, or module was edited this round; all four runs use the recorded driver flags |
| wall times | measured with other agent sessions possibly resident on the machine; upper bounds |

# projective-route, task 3 (2026-09-15, late evening)

Does the `2.503676e-01` coordinate error at `t = 4` sit at the fixed 15-key
entrance dictionary's closure floor?  Task 2 concluded it does, from the
step-invariance of the driven-interval error.  This round tests that by
enlarging the dictionary itself and propagating each enlargement with the
cheap route's hybrid, imported at its committed API.

Configuration for every rollout below: strong coupling (`lambda_ep = 1.5`,
`gamma = 0.5`, drive amplitude 1), phonon cutoff 16, exact-correlated
preparation, SSPRK(3,3) at `h = 0.0025`, the unguarded fixed-dictionary
predictor, Dykstra 30 on both targets at stage cadence between re-selections,
and the declared proximal log-det completion re-solved every `N` steps
warm-started on the carried frontier.  Only the active dictionary and `N`
change.  Every score is the all-coordinate scalar RMS of the 31 closed scalar
coordinates against the exact cutoff-16 driven reference
(`all_coordinate_scalar_rms_error`), on the stated horizon; the raw and the
offset-subtracted conventions agree to every digit printed, the `t = 0`
coordinate offset being `8.08e-16` in Euclidean norm.  Physicality numbers are
measured minimum eigenvalues along the trajectory; there is no certificate
anywhere on this route.

Imports are from the committed tree at `d31a0a9e`, exported to
`output/local_runs/paper_v_projective_dictionary_20260915d/head_src/` and put
on `PYTHONPATH`, because another agent was editing `cheap_projection_route.py`
throughout.  That export differs from the working tree in exactly the two
files that agent owns.  No module was edited this round.

## The result, first

**The `t = 4` error is not at the dictionary's floor; enlarging the dictionary
moves it by a factor 1.65.**  At the same cadence `N = 16` and the same
horizon `t = 0` to `4`, the 15-key entrance dictionary scores `2.684227e-02`
and the 27-key dictionary --- the entrance keys plus the twelve relative-mode
moments of two spin operators and one boson coordinate --- scores
`1.630660e-02`, below the recorded projective event-guard route's `0.026049`
on the same horizon, at `749.1 s` against `3,229.8 s`.  The Euclidean norm of
the closed scalar coordinate difference at `t = 4` falls from `3.025108e-01`
to `2.350741e-01`.

**But the improvement is not uniform over the chart, and the direction
reverses on four of the 31 coordinates.**  On the 27 closed scalar
coordinates outside the phonon-occupation block the score falls monotonically
with every key added --- `2.810405e-02`, `2.491478e-02`, `1.319865e-02` at 15,
19 and 27 keys over `t = 0` to `4` at `N = 16` --- while on the four
phonon-occupation coordinates it rises monotonically, `1.589107e-02`,
`1.716470e-02`, `2.974702e-02`.  At a loose re-selection cadence the rising
block wins and the all-coordinate score gets worse with dictionary size: at
`N = 64` over `t = 0` to `4` the 31-key dictionary scores `6.232531e-02`
against the 15-key's `4.126006e-02`.

**The residual-driven dictionary enlargement layer runs and admits nothing.**
`APCMClosureGraphAnalyzer.analyze` completes its checkpoint in `50.8` to
`53.7 s` at peak `3.29 GB` and returns
`nested_completion_failure: moment_extension_degeneracy: reference completion:
PrimalInfeasible` at both `t = 0` and `t = 1`, so no candidate is scored and
no key is admitted.  It also has no caller anywhere in `paper_5`, is absent
from `stability/__init__.py`, and there is no code that turns an eligible
candidate into an enlarged `active_keys`.

## (a) What declared dictionaries the owned modules already support

Every dictionary on this route is declared the same way: as the `active_keys`
of a `SymmetryReducedPositiveExtension`, which `FixedDictionaryProjectedAPCM`
then adopts whole (`projected_apcm.py:515`).  The positive extension's Gram
rows are the nine base half-words plus one half-word per active key, so the
row count is `9 + len(active_keys)`.

`adaptive_positive_moment.py` declares four key sets at module level.
`ARCHIVE_RELATIVE_MOMENT_KEYS` (14) are the relative-mode moments the archive
chart already fixes and are excluded from any active dictionary.
`HIDDEN_RELATIVE_MOMENT_KEYS` (31) is everything the third-order hierarchy
holds that the archive does not fix.  `ENTRANCE_RELATIVE_MOMENT_KEYS` (15) is
the subset of those of two spin operators and no boson coordinate, or one
spin operator and two boson coordinates.  `INITIAL_PROMOTION_CANDIDATE_KEYS`
(16) is the complement, and it is the whole of the declared enlargement: the
four relative-mode moments of the identity and three boson coordinates, and
the twelve of two spin operators and one boson coordinate.  All sixteen are
degree three, so `15 -> 31` is the only declared enlargement and those two
blocks are its only declared sub-steps.

| dictionary | active keys | state coordinates | positive extension rows | frontier keys (right-hand-side facing / auxiliary) | lower keys | constructs today |
|---|---:|---:|---:|---:|---:|---|
| entrance | 15 | 44 | 24 | 62 (15 / 47) | 29 | yes |
| entrance plus the four identity-and-three-boson keys | 19 | 48 | 28 | 111 (15 / 96) | 33 | yes |
| entrance plus the twelve two-spin-and-one-boson keys | 27 | 56 | 36 | 70 (16 / 54) | 41 | only through the declared completion retry ladder |
| the whole hidden set | 31 | 60 | 40 | 129 (16 / 113) | 45 | yes |

The 27-key entry is the one qualification, and it is a preparation failure,
not a construction failure.  The extension, the geometry and the model all
build.  `prepare_projected_apcm_initial_state` then asks the declared proximal
log-det completion at the contracted preparation's lower moments and it
refuses --- 40 Newton iterations, relative face support 35 of 36, directions
19 of 70, `reduced_gradient 6.209e-08` --- and that function fails closed
without consulting any retry.  The first rung of the declared completion retry
ladder (`logdet_shift 1e-8`, the same `--completion-retry-shifts 1e-8 1e-7
--completion-retry-weights 1e-1` ladder the projective rollouts declare)
certifies at the same lower moments in 12 Newton iterations at
`reduced_gradient 4.270e-08`.  That is the rule the projective driver already
applies at an exact-time start, so the 27-key rollouts below use it and record
which rung certified.  The other three dictionaries certify on the declared
completion itself, in `1.109 s`, `2.618 s` and `33.337 s`.

The record's 60-coordinate positive-`M4` route (cell A, `0.012580` over `t = 0`
to `4` at `3,399.6 s`; worklog:2608--2664) declared its dictionary as the same
module-level constant, hard-wired: `adaptive_positive_moment_propagation.py`
names `HIDDEN_RELATIVE_MOMENT_KEYS` at eight places and takes no dictionary
argument, and its 60 coordinates are the 29 raw archive coordinates plus those
31 relative-mode moments.  Its terminal rule is not the same object: it closes
the degree-four frontier with `PositiveFourthMomentCompletion` at
`phonon_envelope 16.0`, `logdet_weight 1.0`, `logdet_shift 1e-5`,
`solver_tolerance 1e-7`, `maximum_iterations 2000`, where the projected route
uses `SymmetryReducedPositiveExtension`.  So cell A's dictionary is available
to `FixedDictionaryProjectedAPCM` today --- it is the 31-key row above --- but
cell A's `0.012580` is not a line this round's rows can be set beside, because
the completion differs.

The residual-driven enlargement layer is a different construction from all of
these.  `APCMClosureGraphAnalyzer.analyze` takes its candidates from
`model.extension.rhs_frontier_keys`, which at the 15-key entrance dictionary
is nine relative-mode moments of two spin operators and one boson coordinate
and six degree-four moments of one spin operator and three boson coordinates
--- not the sixteen promotion candidates --- and conditions them by building a
nested diagnostic extension at 30 active keys and 223 frontier keys.  It is a
diagnostic: it returns `eligible` and `mandatory` flags with an impact and a
cost score per candidate, and nothing in the repository consumes them.

## (b) Every dictionary that constructs, propagated

The cheap route's hybrid accepts the enlarged model through its committed API
unchanged: `prepare_cheap_projection_route(..., active_keys=...)` and
`integrate_cheap_projection_ssprk3(..., lift_reselection_interval=N)`.  The
15-key rows reproduce the record in every printed digit ---
`0.026842268193069854` at `N = 16` against the recorded `0.0268423`, and
`0.0412600573162618` at `N = 64` against `0.0412601` --- which is what makes
the enlarged rows comparable.

### `t = 0` to `4` at `N = 16`

| quantity | 15 keys | 19 keys | 27 keys | 31 keys |
|---|---:|---:|---:|---:|
| state coordinates / extension rows / frontier keys | 44 / 24 / 62 | 48 / 28 / 111 | 56 / 36 / 70 | 60 / 40 / 129 |
| horizon reached | `t = 4`, 1,600 steps | `t = 4`, 1,600 steps | `t = 4`, 1,600 steps | `t = 2.52`, 1,008 steps |
| all-coordinate scalar RMS, `t = 0` to `4` | `2.684227e-02` | `2.405548e-02` | `1.630660e-02` | not reached |
| the same on the 27 coordinates outside the phonon-occupation block | `2.810405e-02` | `2.491478e-02` | `1.319865e-02` | not reached |
| the same on the four phonon-occupation coordinates | `1.589107e-02` | `1.716470e-02` | `2.974702e-02` | not reached |
| blockwise scalar RMS, rho / B / N / A / C | `6.014e-02` / `1.956e-02` / `1.589e-02` / `1.711e-02` / `2.266e-02` | `5.272e-02` / `1.911e-02` / `1.716e-02` / `1.630e-02` / `1.958e-02` | `1.501e-02` / `4.528e-03` / `2.975e-02` / `1.576e-02` / `1.324e-02` | not reached |
| Euclidean coordinate difference at `t = 4`, all 31 / outside the phonon-occupation block | `3.025108e-01` / `2.946822e-01` | `2.424770e-01` / `2.313614e-01` | `2.350741e-01` / `1.829419e-01` | not reached |
| site-0 occupation RMS | `4.6156e-02` | `4.1258e-02` | `1.0071e-02` | not reached |
| internal-energy sample RMS, Eq. (22) | `4.2087e-02` | `5.1218e-02` | `7.0888e-02` | not reached |
| minimum unshifted retained joint Gram | `-1.705e-08` | `-4.169e-09` | `-1.877e-08` | `-7.744e-09` to `t = 2.5` |
| minimum scaled positive-extension eigenvalue | `-1.969e-07` | `-5.824e-07` | `-6.347e-06` | `-4.473e-06` to `t = 2.5` |
| maximum standardized frontier envelope ratio, against the frozen box 1 | `0.4366` | `0.4433` | `0.3646` | `0.5407` |
| completions solved / refused | 100 / 9 | 100 / 10 | 100 / 18 | 63 / 11 |
| mean completion solve | `0.71 s` | `1.57 s` | `7.10 s` | `16.28 s` |
| build / autonomous wall / peak resident | `1.3 s` / `93.1 s` / `352 MiB` | `3.0 s` / `181.1 s` / `443 MiB` | `18.8 s` / `749.1 s` / `621 MiB` | `37.1 s` / `1,053.1 s` / `901 MiB` |

The 31-key rollout does not reach `t = 4` inside a 20-minute run: it was
stopped by its `1,050 s` wall budget at `t = 2.52`, 1,008 of the 1,600 steps
`t = 4` needs, and `1,025.9 s` of its `1,053.1 s` --- 97 percent --- is the 63
completion solves.  On the horizon it does reach it is the most accurate of
the four.

### `t = 0` to `2.5` at `N = 16`, which all four reach

| quantity | 15 keys | 19 keys | 27 keys | 31 keys |
|---|---:|---:|---:|---:|
| all-coordinate scalar RMS | `1.344608e-02` | `1.379038e-02` | `5.875667e-03` | `6.455155e-03` |
| the same on the 27 coordinates outside the phonon-occupation block | `1.412399e-02` | `1.455775e-02` | `5.161930e-03` | `5.106720e-03` |
| the same on the four phonon-occupation coordinates | `7.391782e-03` | `6.583078e-03` | `9.364804e-03` | `1.212042e-02` |
| Euclidean coordinate difference at `t = 2.5`, all 31 | `1.459697e-01` | `1.456684e-01` | `7.119462e-02` | `6.513699e-02` |

### `t = 0` to `4` at `N = 32`, which all four reach

| quantity | 15 keys | 19 keys | 27 keys | 31 keys |
|---|---:|---:|---:|---:|
| all-coordinate scalar RMS | `3.252016e-02` | `2.831108e-02` | `2.595243e-02` | `3.495362e-02` |
| the same on the 27 coordinates outside the phonon-occupation block | `3.275830e-02` | `2.632542e-02` | `1.769154e-02` | `1.719508e-02` |
| the same on the four phonon-occupation coordinates | `3.086475e-02` | `3.916403e-02` | `5.574190e-02` | `8.644553e-02` |
| Euclidean coordinate difference at `t = 4`, all 31 / outside the phonon-occupation block | `3.331475e-01` / `3.001558e-01` | `3.099550e-01` / `2.636738e-01` | `3.217649e-01` / `2.146453e-01` | `4.459023e-01` / `1.993453e-01` |
| minimum unshifted retained joint Gram / minimum scaled positive-extension eigenvalue | `-2.212e-08` / `-5.301e-07` | `-2.099e-08` / `-1.129e-06` | `-1.820e-08` / `-7.088e-06` | `-1.502e-08` / `-8.050e-06` |
| autonomous wall / completions solved / refused | `67.2 s` / 50 / 5 | `118.0 s` / 50 / 4 | `379.7 s` / 50 / 3 | `738.7 s` / 50 / 7 |

### `t = 0` to `4` at `N = 64`

| quantity | 15 keys | 19 keys | 31 keys |
|---|---:|---:|---:|
| all-coordinate scalar RMS | `4.126006e-02` | `3.564294e-02` | `6.232531e-02` |
| the same on the 27 coordinates outside the phonon-occupation block | `4.201567e-02` | `2.886996e-02` | `2.667855e-02` |
| the same on the four phonon-occupation coordinates | `3.574428e-02` | `6.495995e-02` | `1.590602e-01` |
| autonomous wall | `50.2 s` | `76.7 s` | `391.2 s` |

## (c) Does the error fall with dictionary size, and is the `t = 4` value a dictionary floor?

**It falls, on 27 of the 31 closed scalar coordinates, at every cadence and
every horizon measured, and it rises on the other four.**  Taking `N = 32` at
`t = 0` to `4`, the one grid where all four dictionaries complete the full
horizon, the scalar RMS on the 27 coordinates outside the phonon-occupation
block goes `3.275830e-02`, `2.632542e-02`, `1.769154e-02`, `1.719508e-02` at
15, 19, 27 and 31 keys, a factor `1.905` over the 16 added keys; on the four
phonon-occupation coordinates it goes `3.086475e-02`, `3.916403e-02`,
`5.574190e-02`, `8.644553e-02`, a factor `2.80` the other way.

By block rather than by key, on those 27 coordinates at `N = 32` over `t = 0`
to `4`:

| dictionary | added over the entrance keys | scalar RMS on the 27 coordinates outside the phonon-occupation block | reduction against the entrance keys |
|---|---|---:|---:|
| entrance | --- | `3.275830e-02` | --- |
| 19 keys | the four identity-and-three-boson moments | `2.632542e-02` | 19.6 percent, `4.9` percent per added key |
| 27 keys | the twelve two-spin-and-one-boson moments | `1.769154e-02` | 46.0 percent, `3.8` percent per added key |
| 31 keys | both blocks | `1.719508e-02` | 47.5 percent |

The two blocks do not compose: the twelve two-spin-and-one-boson moments carry
46.0 of the 47.5 percent on their own, and adding the four
identity-and-three-boson moments on top of them buys `1.5` further percentage
points while costing the four phonon-occupation coordinates a further factor
`1.55` (`5.574190e-02` to `8.644553e-02`).  That is why the 27-key dictionary
is the best all-coordinate row at both `N = 16` (`1.630660e-02`) and `N = 32`
(`2.595243e-02`), and why the 31-key dictionary is the worst at `N = 64`
(`6.232531e-02`).

**So the `t = 4` floor is a dictionary floor, and task 2's reading of it needs
one correction.**  Task 2 said the `2.503676e-01` at `t = 4` is within a few
percent of the floor for a 15-entrance-key fixed dictionary at phonon cutoff
16 and that halving it is a dictionary question.  The dictionary question has
now been asked: at the same propagation and cadence the Euclidean coordinate
difference at `t = 4` falls from `3.025108e-01` at 15 keys to `2.350741e-01`
at 27, and on the 27 coordinates outside the phonon-occupation block from
`2.946822e-01` to `1.829419e-01`, a factor `1.61`.  It is not halved by the
one enlargement the modules declare, and the part of the chart that gets
worse is the part that already carried this route's long-horizon error.
Whether the projective route itself inherits the same reduction is untested:
everything here is measured at the cheap hybrid's propagation, where the
15-key line at `t = 4` is `3.025108e-01` and not the projective route's
`2.503676e-01`.

**The cost of the enlargement is the completion solve and nothing else.**  The
mean declared proximal log-det completion goes `0.71 s`, `1.57 s`, `7.10 s`,
`16.28 s` at 15, 19, 27 and 31 keys, while the velocity stays at `8.7` to
`14.2 s` for the whole 1,600-step rollout and the Dykstra repair at `11` to
`45 s`.  The solve cost does not track the frontier dimension: the 27-key
dictionary has 70 frontier keys against the 19-key's 111 and costs `4.5` times
as much per solve, so it is the extension row count and the face the solve
lands on, not the number of frontier coordinates, that sets the price.
Refusals rise with the dictionary too --- 9, 10, 18 of 100 at `N = 16` --- and
none of them stops a rollout, because the hybrid keeps and repairs the carried
lift at a refusal.

**Physicality degrades one to two decades with size and stays inside the
box.**  At `N = 16` over `t = 0` to `4` the minimum scaled positive-extension
eigenvalue is `-1.969e-07`, `-5.824e-07` and `-6.347e-06` at 15, 19 and 27
keys, the minimum unshifted retained joint Gram stays at `10^-8` throughout
(`-1.705e-08`, `-4.169e-09`, `-1.877e-08`), and the maximum standardized
frontier envelope ratio is `0.4366`, `0.4433` and `0.3646`, all inside the
frozen box.

**The residual-driven enlargement layer admits nothing, at either time
tried.**  One checkpoint of `APCMClosureGraphAnalyzer.analyze` at the 15-key
entrance dictionary, at the preparation and at the `t = 1` state of the
15-key `N = 16` rollout, `local_step_error 1e-6`, declared
`checkpoint_interval 1e-2` and `impact_threshold 1.0`:

| quantity | `t = 0` | `t = 1` |
|---|---|---|
| candidates conditioned | 15 | 15 |
| which keys | the nine relative-mode moments of two spin operators and one boson coordinate (`XX`, `XY`, `XZ` twice, `YY`, `YZ` twice, `ZZ` twice) and the six degree-four moments of one spin operator and three boson coordinates (`IX` and `IY` at `x+p = 3`) | the same |
| nested diagnostic extension | 30 active keys, 223 frontier keys, 60 Gram rows | the same |
| outcome | `nested_completion_failure: moment_extension_degeneracy: reference completion: PrimalInfeasible` | the same |
| candidates scored / eligible / mandatory | 0 / 0 / 0 | 0 / 0 / 0 |
| wall / peak resident | `53.7 s` / `3.29 GB` | `50.8 s` / `3.29 GB` |

So the layer neither names an admission nor rejects one: its nested one-shell
completion is primal infeasible at both states, and every candidate diagnostic
is skipped.  Even had it returned candidates, nothing would act on them:
`APCMClosureGraphAnalyzer` has no caller anywhere under `paper_5`, is not
exported from `stability/__init__.py`, and no function rebuilds a
`FixedDictionaryProjectedAPCM` from an admitted key.  The enlargements
measured above were declared by hand, which is the only way a larger
dictionary reaches a rollout today.

## Provenance and gates

- Every rollout was wrapped in `pipelines/shell/ram_guard.py --limit-mb 4500`,
  one at a time; the largest peak was `3,293 MB`, on the adaptation checkpoint,
  and no run exceeded 20 minutes.  Wall times were measured while another
  agent's jobs shared the machine, so they are upper bounds.
- The 15-key rows reproduce the record: `0.026842268193069854` at `N = 16`
  against the recorded `0.0268423`, and `0.0412600573162618` at `N = 64`
  against `0.0412601`, both over `t = 0` to `4`.
- No module was edited and no default, acceptance rule or ladder was changed.
  The only rule applied that `prepare_projected_apcm_initial_state` does not
  apply itself is the declared completion retry ladder at the preparation, used
  for the 27-key dictionary alone, recorded in that run's `summary.json` as
  `initialization_ladder_rung 0` (`logdet_shift 1e-8`).
- Scripts, all under
  `output/local_runs/paper_v_projective_dictionary_20260915d/`:
  `inventory_dictionaries.py` (the dimension table, builds nothing else),
  `probe_preparation_completion.py` (the declared completion and its ladder at
  the preparation, per dictionary), `run_dictionary_rollout.py` (one rollout),
  `run_adaptation_checkpoint.py` (one closure-graph checkpoint),
  `score_subhorizons.py`, `block_attribution.py`, `instantaneous_at_t.py`,
  `collect_costs.py` and `tabulate.py`, the last five reading only
  `trajectory.npz` and `summary.json` and propagating nothing.
- Artifacts, under `output/local_runs/`:
  `paper_v_dictionary_entrance15_n{16,32,64}_t4_20260915d`,
  `paper_v_dictionary_entrance15_plus_boson_cubic_n{16,32,64}_t4_20260915d`,
  `paper_v_dictionary_entrance15_plus_two_spin_boson_n{16,32}_t4_20260915d`,
  `paper_v_dictionary_hidden31_n{16,32,64}_t4_20260915d`,
  `paper_v_dictionary_smoke_hidden31_20260915d`,
  `paper_v_dictionary_adaptation_20260915d`, and the scripts and logs above.

## Structured summary of task 3

| item | value |
|---|---|
| declared dictionaries the owned modules support | four, all as `SymmetryReducedPositiveExtension(active_keys=...)`: 15 entrance keys (44 state coordinates, 24 extension rows, 62 frontier keys), 19 (48 / 28 / 111), 27 (56 / 36 / 70), 31 (60 / 40 / 129); `INITIAL_PROMOTION_CANDIDATE_KEYS` is the whole declared enlargement and all 16 of its keys are degree three |
| which construct today | the 15-, 19- and 31-key dictionaries through `prepare_projected_apcm_initial_state` unchanged; the 27-key one only through the declared completion retry ladder, whose first rung (`logdet_shift 1e-8`) certifies where the declared completion refuses |
| the record's 60-coordinate positive-`M4` route | declared its dictionary as the hard-wired `HIDDEN_RELATIVE_MOMENT_KEYS`, the 31-key row above, with `PositiveFourthMomentCompletion` (`phonon_envelope 16.0`, `logdet_shift 1e-5`) rather than `SymmetryReducedPositiveExtension`, so its `0.012580` is not a comparable line |
| best `t = 0` to `4` score | 27 keys at `N = 16`: `1.630660e-02` in `749.1 s`, against 15 keys' `2.684227e-02` in `93.1 s` and the recorded projective event guard's `0.026049` in `3,229.8 s` |
| does the error fall with dictionary size | on 27 of the 31 closed scalar coordinates, yes, at every cadence: `3.275830e-02`, `2.632542e-02`, `1.769154e-02`, `1.719508e-02` at 15, 19, 27, 31 keys (`N = 32`, `t = 0` to `4`), a factor `1.905` |
| per added key or block | the twelve two-spin-and-one-boson moments carry 46.0 percent of the 47.5, at `3.8` percent per key; the four identity-and-three-boson moments carry 19.6 percent alone at `4.9` percent per key, and only `1.5` further points on top of the twelve |
| where it rises | the four phonon-occupation coordinates: `3.086475e-02`, `3.916403e-02`, `5.574190e-02`, `8.644553e-02` at 15, 19, 27, 31 keys on the same grid, which is why the 31-key dictionary is the worst all-coordinate row at `N = 64` (`6.232531e-02` against 15 keys' `4.126006e-02`) |
| is `2.503676e-01` at `t = 4` a dictionary floor | yes, and the dictionary moves it: the same propagation's `3.025108e-01` at 15 keys falls to `2.350741e-01` at 27, and to `1.829419e-01` from `2.946822e-01` on the coordinates outside the phonon-occupation block.  The one declared enlargement does not halve it |
| cost of the enlargement | the completion solve alone: `0.71`, `1.57`, `7.10`, `16.28 s` mean per solve at 15, 19, 27, 31 keys, 97 percent of the 31-key wall; the solve cost does not track the frontier dimension (27 keys has 70 frontier keys against 19 keys' 111 and costs `4.5` times as much) |
| the 31-key rollout at `N = 16` | does not reach `t = 4` in 20 minutes: `t = 2.52`, 1,008 of 1,600 steps, `1,053.1 s`, of which `1,025.9 s` is 63 completion solves |
| physicality | minimum scaled positive-extension eigenvalue `-1.969e-07`, `-5.824e-07`, `-6.347e-06` at 15, 19, 27 keys (`N = 16`, `t = 0` to `4`); retained joint Gram at `10^-8` throughout; standardized frontier inside the frozen box at `0.4366`, `0.4433`, `0.3646` |
| residual-driven enlargement layer | runs, refuses, admits nothing: `nested_completion_failure: ... PrimalInfeasible` at `t = 0` and `t = 1`, 15 candidates conditioned and 0 scored, `50.8` to `53.7 s` at peak `3.29 GB`; and it has no caller under `paper_5`, so no admitted key could reach a rollout even if one were named |
| files changed | none.  No module edited, nothing committed; the report and the worklog block only |

# projective-route, task 4 (2026-09-15, night)

Task 3 measured one lever, the dictionary, and found that the twelve
two-spin-and-one-boson relative-mode moments buy the whole chart except the
four phonon-occupation coordinates, which get worse.  The cheap route's task 4
measured the other lever, the energy-neutral Dykstra repair, and found that
holding the Eq. (22) internal energy on a declared reference is what repairs
the phonon-occupation block.  This round combines them: the 27-key dictionary
at the hybrid's `N = 16` cadence with the energy-neutral repair, in the
`drive_power` reference mode and, as the second variant, in `post_pulse`.

Configuration for every rollout below: 27 active keys (the 15 entrance keys
plus the twelve relative-mode moments of two spin operators and one boson
coordinate), 56 state coordinates, 36 positive-extension rows, 70 frontier
keys, 41 lower keys; strong coupling (`lambda_ep = 1.5`, `gamma = 0.5`, drive
amplitude 1), phonon cutoff 16, exact-correlated preparation carried through
the declared completion retry ladder at its first rung (`logdet_shift 1e-8`,
recorded as `initialization_ladder_rung 0`), SSPRK(3,3) at `h = 0.0025`, the
unguarded fixed-dictionary predictor, Dykstra 30 on both targets at stage
cadence, the declared proximal log-det completion re-solved every 16 steps
warm-started on the carried frontier, and `energy_tolerance 0`, so the
hyperplane is met at every repair the constraint is active for.  Every score
is the all-coordinate scalar RMS of the 31 closed scalar coordinates against
the exact cutoff-16 driven reference (`all_coordinate_scalar_rms_error`) on
the stated horizon; the internal energy is the sample RMS of the
time-independent energy of Eq. (22) (`static_energy_rms_error`) on the same
horizon; physicality numbers are measured minimum eigenvalues along the
trajectory, and there is no certificate anywhere on this route.  Imports are
from the committed tree at `2d8f0efb`, exported to
`output/local_runs/paper_v_projective_dictionary_energy_20260915e/head_src/`
and put on `PYTHONPATH`, because another agent was editing
`cheap_projection_route.py` throughout; that export differs from the working
tree in exactly the two files that agent owns.  No module was edited this
round.

## The result, first

**The two levers compose, and the combination is the most accurate Paper V
trajectory on record over `t = 0` to `4` and over `t = 0` to `8`.**  The
27-key dictionary at `N = 16` with the energy-neutral repair in `drive_power`
mode scores `1.142716e-02` over `t = 0` to `4` and `2.593079e-02` over `t = 0`
to `8`, against the 27-key unconstrained `1.630660e-02` at `t = 4`, the 15-key
energy-neutral `drive_power` row's `2.374400e-02` and `4.557100e-02`, the
recorded projective event guard's `0.026049` at `t = 4` and the recorded
60-coordinate positive-`M4` route's `0.012580` at `t = 4`.  It reached `t = 8`
in three chained segments, `2,058.0 s` of autonomous wall in total.

**The energy-neutral repair removes the phonon-occupation penalty the
enlargement introduced, and more than removes it.**  On the four
phonon-occupation coordinates over `t = 0` to `4` the same 27-key dictionary
scores `9.836299e-03` with the constraint against `2.974702e-02` without it, a
factor `3.02`, and the constrained 27-key value is below the *15*-key
unconstrained `1.589107e-02` as well.  The Eq. (22) internal-energy sample RMS
over that horizon falls from `7.088814e-02` to `4.597225e-03`, a factor `15.4`.

**The price is the same one the cheap route measured at 15 keys: the cone and
the conic solve.**  Over `t = 0` to `4` the minimum unshifted retained joint
Gram moves from `-1.877e-08` to `-4.535e-04` and the minimum scaled
positive-extension eigenvalue from `-6.347e-06` to `-4.852e-05`; the declared
proximal log-det completion refuses 32 of 99 solves against 18 of 100, at
`9.49 s` mean against `7.10 s`.

**`post_pulse` is the worse variant at 27 keys, and its reference is why.**
Held at the trajectory's own Eq. (22) energy at `t = 4`,
`-3.3905733452008326`, it sits `0.180825` above the exact reference's
`-3.571397982`, where the 15-key `post_pulse` reference sat `0.074667` above
it: the enlarged dictionary drifts further in energy over the driven interval,
so the quantity the post-pulse mode freezes is further from the truth.  Over
`t = 0` to `8` it scores `4.537517e-02` at internal energy `1.373215e-01`
against `drive_power`'s `2.593079e-02` and `5.751890e-03`.  The `drive_power`
reference at 27 keys, by contrast, is the closest this route has produced:
`-3.564687762684483`, `0.006710` above the exact value, against the 15-key
`drive_power` reference's `0.027219`.

## Gates

| gate | result |
|---|---|
| chaining reproduces a single rollout | 27 keys, `N = 16`, `drive_power`: the chain `t = 0` to `0.05` resumed at `t = 0.04` and continued to `t = 0.08` differs from the single rollout `t = 0` to `0.08` by `1.732e-14` at most over the 56 state coordinates and the 31 closed scalar coordinates, with the same 33 sample times and the same re-selection schedule; the drive-power reference agrees to `8.9e-16`.  Artifacts `paper_v_dictionary_energy_smoke{A,B,S}_20260915e` |
| a segment resumes on a re-selection boundary | every segment's `resume.npz` is written at the last accepted step that is a whole number of 16-step intervals from that segment's start, so the chain's completion solves fall on the steps a single rollout's would have |
| the recorded driven interval is reusable for `post_pulse` | the 27-key `N = 16` unconstrained rollout rerun at the committed tree imported here, `t = 0` to `0.2`, is equal under `np.array_equal` to the task-3 artifact on times, states, carried frontier, both eigenvalue series and archive coordinates --- 81 samples, maximum absolute difference `0.0`.  Artifact `paper_v_dictionary_energy_prefixgate_27key_n16_20260915e`, comparison `prefix_gate.json` |
| the constraint is off before its declared time | the `post_pulse` chain's `t = 0` to `4` row reproduces the unconstrained 27-key row in every printed digit: `1.630660e-02`, phonon-occupation `2.974702e-02`, internal energy `7.088814e-02` |
| the energy is held where the constraint is on | maximum deviation from the declared reference `1.78e-15` over 9,559 constrained repairs (`drive_power`, `t = 0` to `8`) and `1.78e-15` over 4,800 (`post_pulse`, `t = 4` to `8`); the hyperplane was representable at every one of them |

## (a) The chained rollouts

Every segment is a separate `ram_guard --limit-mb 4500` process, resumed from
the previous segment's saved state and carried frontier, and no two ever ran
at once.  Counts are each segment's own totals from its `summary.json`.

| quantity | `drive_power` seg 1 | seg 2 | seg 3 | `post_pulse` seg 1 |
|---|---:|---:|---:|---:|
| interval | `0` to `3.96` | `3.96` to `7.84` | `7.84` to `8.0` | `4.0` to `8.0` |
| accepted steps | 1,584 | 1,552 | 64 | 1,600 |
| why it stopped | `1,000 s` wall budget | `1,000 s` wall budget | reached `t = 8` | reached `t = 8` |
| autonomous wall / build | `1,011.7 s` / `19.9 s` | `1,010.8 s` / `20.0 s` | `35.5 s` / `19.9 s` | `679.8 s` / `19.9 s` |
| declared completions re-solved / refused | 99 / 32 | 97 / 42 | 4 / 1 | 100 / 12 |
| wall in those solves | `939.3 s` | `937.6 s` | `32.4 s` | `616.2 s` |
| repairs applied / skipped | 4,711 / 41 | 4,656 / 0 | 192 / 0 | 4,800 / 0 |
| constrained repairs / hyperplane not representable | 4,711 / 0 | 4,656 / 0 | 192 / 0 | 4,800 / 0 |
| repair wall / velocity wall | `47.6 s` / `24.4 s` | `48.7 s` / `24.1 s` | `2.0 s` / `1.0 s` | `50.2 s` / `12.9 s` |
| peak resident | `630 MB` | `626 MB` | `628 MB` | `632 MB` |

The `post_pulse` chain's driven interval is the recorded task-3 27-key
`N = 16` rollout itself (`749.1 s`, 1,600 steps), reused under the gate above
rather than repeated: in that mode the constraint is off before `t = 4`, and
the rerun is bitwise equal over the prefix tested.

### `t = 0` to `4`

| quantity | 27 keys, `drive_power` | 27 keys, `post_pulse` = unconstrained | 15 keys, `drive_power` | 15 keys, unconstrained | recorded projective event guard | recorded positive-`M4` (cell A) |
|---|---:|---:|---:|---:|---:|---:|
| all-coordinate scalar RMS | `1.142716e-02` | `1.630660e-02` | `2.374400e-02` | `2.684227e-02` | `2.604900e-02` | `1.258000e-02` |
| the same on the 27 coordinates outside the phonon-occupation block | `1.164437e-02` | `1.319865e-02` | --- | `2.810405e-02` | --- | --- |
| the same on the four phonon-occupation coordinates | `9.836299e-03` | `2.974702e-02` | `1.149900e-02` | `1.589107e-02` | --- | --- |
| blockwise scalar RMS, rho / B / N / A / C | `2.0368e-02` / `6.9088e-03` / `9.8363e-03` / `8.9418e-03` / `1.1167e-02` | `1.501e-02` / `4.528e-03` / `2.975e-02` / `1.576e-02` / `1.324e-02` | --- | `6.014e-02` / `1.956e-02` / `1.589e-02` / `1.711e-02` / `2.266e-02` | --- | --- |
| Eq. (22) internal-energy sample RMS | `4.597225e-03` | `7.088814e-02` | `2.040600e-02` | `4.208700e-02` | `3.620100e-02` | `7.963000e-04` |
| site-0 occupation RMS | `1.430879e-02` | `1.007098e-02` | `4.235300e-02` | `4.615600e-02` | --- | `6.957000e-03` |
| minimum unshifted retained joint Gram | `-4.535e-04` | `-1.877e-08` | --- | `-1.705e-08` | --- | `2.889e-05` |
| minimum scaled positive-extension eigenvalue | `-4.852e-05` | `-6.347e-06` | --- | `-1.969e-07` | --- | `-8.76e-07` (extended `M4`) |
| maximum standardized frontier envelope ratio | `0.3694` | `0.3646` | --- | `0.4366` | --- | --- |
| Euclidean coordinate difference at `t = 4`, all 31 / outside the phonon-occupation block | `1.551037e-01` / `1.439192e-01` | `2.350741e-01` / `1.829419e-01` | --- | `3.025108e-01` / `2.946822e-01` | --- | --- |
| autonomous wall to this horizon | about `1,022 s` (segment 1's `1,011.7 s` for 1,584 steps plus 16 steps of segment 2 at its own rate) | `749.1 s` | --- | `93.1 s` | `3,229.8 s` | `3,399.6 s` |

### `t = 0` to `8`

| quantity | 27 keys, `drive_power` | 27 keys, `post_pulse` | 15 keys, `drive_power` | 15 keys, `post_pulse` | 15 keys, unconstrained |
|---|---:|---:|---:|---:|---:|
| all-coordinate scalar RMS | `2.593079e-02` | `4.537517e-02` | `4.557100e-02` | `4.254200e-02` | `5.773000e-02` |
| the same on the 27 coordinates outside the phonon-occupation block | `2.474607e-02` | `4.219396e-02` | --- | --- | --- |
| the same on the four phonon-occupation coordinates | `3.282777e-02` | `6.276381e-02` | `3.548400e-02` | `4.299000e-02` | `1.070910e-01` |
| blockwise scalar RMS, rho / B / N / A / C | `4.0173e-02` / `2.7248e-02` / `3.2828e-02` / `1.9942e-02` / `2.1275e-02` | --- | --- | --- | --- |
| Eq. (22) internal-energy sample RMS | `5.751890e-03` | `1.373215e-01` | `2.405500e-02` | `6.060300e-02` | `2.269430e-01` |
| site-0 occupation RMS | `3.027929e-02` | `3.984892e-02` | `7.415600e-02` | `6.450500e-02` | `6.582300e-02` |
| minimum unshifted retained joint Gram | `-2.152e-03` | `-2.907e-04` | --- | --- | --- |
| minimum scaled positive-extension eigenvalue | `-1.922e-04` | `-2.486e-05` | --- | --- | --- |
| maximum standardized frontier envelope ratio | `0.4518` | `0.3748` | --- | --- | --- |
| Euclidean coordinate difference at `t = 8`, all 31 / outside the phonon-occupation block | `2.372364e-01` / `2.292268e-01` | --- | --- | --- | --- |
| declared completions re-solved / refused to this horizon | 200 / 75 | 200 / 30 | --- | --- | --- |
| autonomous wall to this horizon | `2,058.0 s` | `1,429.0 s` | --- | --- | --- |

The 15-key columns are the cheap route's task-4 rows, rescored there on the
same sub-horizons; its `t = 0` to `8` figures are read from that report's
sub-horizon table.

## (b) Does the energy-neutral repair remove the phonon-occupation penalty?

**Yes, and it reverses it.**  On the four phonon-occupation coordinates over
`t = 0` to `4`, at 27 keys and `N = 16`, with everything else identical:

| dictionary | constraint | scalar RMS on the four phonon-occupation coordinates, `t = 0` to `4` |
|---|---|---:|
| 15 entrance keys | none | `1.589107e-02` |
| 27 keys | none (and the `post_pulse` rollout, whose constraint is off before `t = 4`) | `2.974702e-02` |
| 27 keys | energy-neutral, `drive_power` | `9.836299e-03` |

The penalty task 3 measured is the rise from `1.589107e-02` to
`2.974702e-02`, a factor `1.87`, when the twelve two-spin-and-one-boson
moments are added.  Holding the Eq. (22) energy on the drive-power reference
takes that block to `9.836299e-03`, which is `0.331` of the unconstrained
27-key value and `0.619` of the 15-key value the enlargement started from.  So
the constraint does not merely pay back the penalty; the enlarged dictionary
with the constraint is better on those four coordinates than the small
dictionary ever was.

What the constraint costs on the rest of the chart at 27 keys over the same
horizon, by block, constrained against unconstrained:

| block | constrained | unconstrained | ratio |
|---|---:|---:|---:|
| rho (3 coordinates) | `2.0368e-02` | `1.501e-02` | `1.357` |
| B (4) | `6.9088e-03` | `4.528e-03` | `1.526` |
| N (4), the phonon-occupation block | `9.8363e-03` | `2.975e-02` | `0.331` |
| A (6) | `8.9418e-03` | `1.576e-02` | `0.567` |
| C (14) | `1.1167e-02` | `1.324e-02` | `0.843` |
| all 31 | `1.142716e-02` | `1.630660e-02` | `0.701` |
| the 27 outside the phonon-occupation block | `1.164437e-02` | `1.319865e-02` | `0.882` |
| site-0 occupation | `1.430879e-02` | `1.007098e-02` | `1.421` |
| Eq. (22) internal energy | `4.597225e-03` | `7.088814e-02` | `0.065` |

So the constraint is not free on the chart: the electronic density block rho
and the coherent-phonon block B are `1.36` and `1.53` times worse with it, and
the site-0 occupation `1.42` times worse, which is the same direction the
15-key `drive_power` row shows against its own unconstrained row.  The net is
a `1.43`-fold improvement on the whole chart, carried by the phonon-occupation
and correlation blocks, and a `15.4`-fold improvement on the quantity the
constraint actually holds.

The build-time reason the cheap route recorded for this was measured at the
24-row extension; it was re-measured here at the 36-row one and it is
unchanged.  Perturbing each of the 29 raw moment coordinates at the prepared
state and taking the Frobenius norm of the derivative of the scaled positive
extension matrix (the map is affine, so one finite difference is exact):

| raw coordinate | 24 rows (15 keys) | 36 rows (27 keys) |
|---|---:|---:|
| `bloch_x` / `bloch_y` / `bloch_z` | `4.6904` / `4.4721` / `4.6904` | `4.6915` / `4.4729` / `4.6915` |
| `raw_normal_00` / `raw_normal_11` | `0.10497` / `0.10497` | `0.14628` / `0.14628` |
| `raw_normal_01_real` | `0.20993` | `0.29257` |
| `raw_normal_01_imag` | `0` | `0` |

So one of the four raw phonon normal coordinates is invisible to the positive
extension at both dictionary sizes --- its derivative is exactly zero --- and
the other three are seen `3.1` to `6.6` percent as strongly as the electronic
Bloch coordinates at 27 keys (`2.2` to `4.5` percent at 15).  Enlarging the
extension by twelve rows multiplies the phonon normal sensitivities by `1.39`
and changes nothing about which coordinate it cannot constrain at all.  The
Eq. (22) hyperplane, by contrast, is exactly a statement about that block ---
`omega_ph` times the trace of the normal correction, plus the electronic and
coupling terms --- which is why it is the constraint that repairs it.
The probe is `probe_extension_sensitivity.py` with
`extension_sensitivity_{15,27}key.json`; every retained-Gram derivative is
`1.414` or `2.000` at both sizes, so the retained cone sees all 29 raw
coordinates equally at both.

## Provenance and gates of the runs themselves

- Every rollout was wrapped in `pipelines/shell/ram_guard.py --limit-mb 4500`,
  one at a time; the largest peak resident was `632 MB` and no segment
  exceeded twenty minutes of autonomous wall.  Wall times were measured with
  other agents' jobs possibly on the machine, so they are upper bounds.
- No module was edited, nothing was committed, and no default, acceptance rule
  or ladder was changed.  The only rule applied that
  `prepare_projected_apcm_initial_state` does not apply itself is the declared
  completion retry ladder at the preparation, which the 27-key dictionary
  needs and which every run here records as
  `initialization_ladder_rung 0` (`logdet_shift 1e-8`).
- Scripts, all under
  `output/local_runs/paper_v_projective_dictionary_energy_20260915e/`:
  `run_energy_segment.py` (one chained segment, the only script that
  propagates), `run_chain.py` (runs segments one after another, never two at
  once), `seed_resume_from_run.py` (writes the `post_pulse` chain's `t = 4`
  start from the recorded task-3 rollout), `compare_prefix.py` (the reuse
  gate), `score_energy_chain.py` (concatenates segments, regenerates the exact
  reference on the chained times and scores the horizons) and
  `probe_extension_sensitivity.py` (the build-time derivative table of
  section (b)); `head_src/` is the committed tree at `2d8f0efb`.
- Artifacts, under `output/local_runs/`:
  `paper_v_dictionary_energy_power_27key_n16_20260915e_seg{1,...}`,
  `paper_v_dictionary_energy_postpulse_27key_n16_20260915e_seg{0,1}`,
  `paper_v_dictionary_energy_prefixgate_27key_n16_20260915e`,
  `paper_v_dictionary_energy_smoke{A,B,S}_20260915e`, and the scores, logs and
  probes in the script directory (`power_chain_score_*.json`,
  `postpulse_chain_score_t8.json`, `prefix_gate.json`,
  `extension_sensitivity_{15,27}key.json`).
- The 27-key `post_pulse` chain reuses the recorded task-3 rollout
  `paper_v_dictionary_entrance15_plus_two_spin_boson_n16_t4_20260915d` as its
  driven interval under the bitwise gate above; nothing else is reused.

## (a) continued: `t = 8` to `20` in three more segments

The chain reached `t = 20` in three segments past `t = 8`, which is the budget
the task set, for `4,794.4 s` of autonomous wall over all six segments and
8,000 accepted steps.

| quantity | `drive_power` seg 4 | seg 5 | seg 6 |
|---|---:|---:|---:|
| interval | `8.0` to `11.84` | `11.84` to `16.32` | `16.32` to `20.0` |
| accepted steps | 1,536 | 1,792 | 1,472 |
| why it stopped | `1,000 s` wall budget | `1,000 s` wall budget | reached `t = 20` |
| autonomous wall | `1,000.4 s` | `1,004.5 s` | `731.5 s` |
| declared completions re-solved / refused | 96 / 32 | 112 / 29 | 92 / 8 |
| wall in those solves | `926.6 s` | `919.9 s` | `661.1 s` |
| peak resident | `611 MB` | `631 MB` | `629 MB` |

Chain totals over `t = 0` to `20`: 8,000 accepted steps, `4,794.4 s`, 500
declared completions re-solved with **144 refused**, `4,416.9 s` of that wall
in the solves (`8.83 s` mean), 23,959 repairs applied and 41 skipped, every
applied repair carrying the hyperplane, maximum deviation of the energy from
its reference `1.78e-15`, peak resident `631 MB`.

Cumulative scores, all from `t = 0`:

| horizon | all-coordinate | outside the phonon-occupation block (27) | the phonon-occupation block (4) | Eq. (22) internal energy | site-0 occupation | minimum retained joint Gram | minimum scaled extension |
|---|---:|---:|---:|---:|---:|---:|---:|
| `t = 4` | `1.142716e-02` | `1.164437e-02` | `9.836299e-03` | `4.597225e-03` | `1.430879e-02` | `-4.535e-04` | `-4.852e-05` |
| `t = 6` | `2.090571e-02` | `1.856285e-02` | `3.257634e-02` | `5.394635e-03` | `2.116592e-02` | `-1.174e-03` | `-1.442e-04` |
| `t = 8` | `2.593079e-02` | `2.474607e-02` | `3.282777e-02` | `5.751890e-03` | `3.027929e-02` | `-2.152e-03` | `-1.922e-04` |
| `t = 12` | `4.250925e-02` | `4.385377e-02` | `3.198825e-02` | `6.088286e-03` | `5.278107e-02` | `-2.463e-03` | `-2.040e-04` |
| `t = 16` | `6.377276e-02` | `6.722257e-02` | `3.188383e-02` | `6.249722e-03` | `8.393451e-02` | `-2.463e-03` | `-2.040e-04` |
| `t = 20` | `8.519313e-02` | `9.045792e-02` | `3.187001e-02` | `6.344620e-03` | `1.125899e-01` | `-2.463e-03` | `-2.040e-04` |

Beside the same horizons of the 15-key energy-neutral rows and the record:

| row | all-coordinate, `t = 0` to `20` | Eq. (22) internal energy | site-0 occupation | autonomous wall |
|---|---:|---:|---:|---:|
| 27 keys, `drive_power`, `N = 16` (this chain) | `8.519313e-02` | `6.344620e-03` | `1.125899e-01` | `4,794.4 s` |
| 15 keys, `post_pulse`, `N = 16` | `2.391720e-01` | `6.938400e-02` | `2.479700e-01` | `669.8 s` |
| 15 keys, `drive_power`, `N = 16` | `3.286740e-01` | `2.600000e-02` | `1.869090e-01` | `719.6 s` |
| 15 keys, unconstrained, `N = 16` | `5.793320e-01` | `2.593707e+00` | `1.999330e-01` | `486.2 s` |
| record, both cones | `1.286070e-01` | `9.396280e-04` | `2.046900e-01` | `5,832.34 s` |
| record, plain archive EOM | `1.932050e-01` | `1.823610e-02` | `2.861700e-01` | --- |

Three readings.

1. **The phonon-occupation block stops growing after the pulse.**  Its
   cumulative scalar RMS is `3.257634e-02` at `t = 6` and `3.187001e-02` at
   `t = 20`; every later horizon's growth is in the other 27 coordinates,
   whose figure goes `1.856285e-02` to `9.045792e-02` over the same span.
   That is the reverse of the unconstrained hybrid, whose `N`-block score at
   15 keys runs from `3.8838e-02` at `t = 6` to `1.362353e+00` at `t = 20`.
2. **The blocks that grow are rho and B.**  At `t = 20` the blockwise scalar
   RMS is rho `1.3580e-01`, B `1.8729e-01`, N `3.1870e-02`, A `4.3970e-02`, C
   `3.1283e-02`: the electronic density and the coherent phonon, which are the
   two blocks the constraint makes worse at `t = 4` as well.
3. **The wall is the completion solve and nothing else.**  `4,416.9 s` of the
   `4,794.4 s` --- 92 percent --- is 500 declared completions, and 144 of them
   refuse.  The repair itself costs `250.5 s` and the velocity `124.8 s` over
   24,000 stages.

## (c) Is it the best Paper V moment-closure trajectory on record?

**On the all-coordinate scalar RMS of the 31 closed scalar coordinates
against the exact cutoff-16 driven reference, yes at every horizon it reached
--- `1.142716e-02` at `t = 4` against the best recorded `1.258000e-02`,
`2.593079e-02` at `t = 8` against the best recorded `4.254200e-02`, and
`8.519313e-02` at `t = 20` against the best recorded `1.286070e-01` --- but
not on the Eq. (22) internal energy, where the recorded positive-`M4` route's
`7.963000e-04` at `t = 4` and the recorded both-cones route's `9.396280e-04`
at `t = 20` are `5.8` and `6.8` times smaller, and not on physicality, where
both of those keep a positive retained joint Gram and this chain's minimum is
`-2.463e-03`.**

| horizon | this chain | the best other row on record at that horizon | what that row is |
|---|---:|---:|---|
| `t = 4` | `1.142716e-02` | `1.258000e-02` | the 60-coordinate positive-`M4` route, cell A, at `3,399.6 s` |
| `t = 8` | `2.593079e-02` | `4.254200e-02` | the 15-key `post_pulse` energy-neutral hybrid |
| `t = 12` | `4.250925e-02` | `1.251390e-01` | the 15-key `drive_power` energy-neutral hybrid |
| `t = 16` | `6.377276e-02` | `1.950960e-01` | the 15-key `post_pulse` energy-neutral hybrid |
| `t = 20` | `8.519313e-02` | `1.286070e-01` | the recorded both-cones run, at `5,832.34 s` |

Two qualifications the record requires.  The `t = 4` comparator, cell A, is
not the same construction --- 60 coordinates closed with
`PositiveFourthMomentCompletion` at `logdet_shift 1e-5`, against 56
coordinates closed with `SymmetryReducedPositiveExtension` here --- and no
`t = 8`, `12` or `16` row exists for either recorded cone route, because those
runs are recorded only at `t = 4` and `t = 20` and their artifacts are no
longer on disk to rescore.  And the recorded cone routes are the only ones
that carry a physicality statement: both cones held for 20 time units with a
minimum retained joint Gram of `1.36819e-5`, where this route's repaired
moments sit `2.463e-03` outside the retained cone and `2.040e-04` outside the
scaled positive extension.  On accuracy per unit of wall, `4,794.4 s` against
`5,832.34 s` for a score `1.51` times smaller, this chain wins; on the
statement that the closure stayed inside its cones, it has nothing to say.

# projective-route, task 5 (2026-09-16)

The cheap route's task 5 found that two runs of one 15-key `N = 4`
`post_pulse` configuration, differing only in whether near-zero repairs were
skipped, separate to `1e-2` at `t = 6.93`, `0.1` at `t = 8.38` and order one
by `t = 11`, and named the completion's refusal --- a discontinuous function
of the state --- as the candidate amplifier.  This round measures the
sensitivity of the 27-key `N = 16` `drive_power` energy-neutral chain of task
4 to the one setting that decides a refusal: the completion solve's conic
tolerance, `APCMExtensionSettings.conic_tolerance`, moved one decade tighter
from the committed `1e-9` to `1e-10`, everything else identical.  Imports are
again from `head_src/` (the committed tree at `2d8f0efb`, unchanged by the
cheap route's later commits); no module was edited and nothing was
committed.  The reviewer's corrections to task 4
(`review-27key-chain-20260916.md`) are taken as the record below: the "best
other row" at `t = 12`, `16` and `20` is the route's own 27-key `post_pulse`
chain (`6.075829e-02`, `6.385168e-02`, `6.990927e-02`), so the task-4
statement was already withdrawn at `t = 16` (a tie) and `t = 20` (second).

Where the tolerance enters, read from `apcm_positive_extension.py`: it is
Clarabel's `tol_gap_abs`, `tol_gap_rel` and `tol_feas` in the completion's
conic solves, the interior Newton's warm-start, stopping and relative-interior
thresholds (`0.1`, `5`, `10` and `100` times it), the acceptance of a
`NumericalError` exit (`100` times it), and the certificate every completion
must pass: scaled minimum eigenvalue `>= -10 tol` and KKT backward error
`<= 50 tol`.  It enters nothing else on this route: the Dykstra repair, the
velocity and the integrator never read it.  The same setting on the *model's*
extension does enter the carried completion the velocity accepts at every
stage (`FixedDictionaryProjectedAPCM._carried_completion`: box margin
`1 - 10 m tol`, scaled minimum `>= m tol`), which is why the run below
tightens the re-selection solve alone.

## The result, first

**The recorded chain's `t = 4` score is not a second-digit number, and the
"best on record" statement at `t = 4` does not survive.**  With the
re-selection completion solved at `1e-10`, the all-coordinate scalar RMS of
the 31 closed scalar coordinates over `t = 0` to `4` is `1.292283e-02`
against the recorded `1.142716e-02`, a `13.1` percent difference for a
one-decade solver-tolerance change, and the recorded comparator, the
positive-`M4` cell A at `1.258000e-02`, lies between the two.  The recorded
member of the configuration is `0.908` of cell A and the tightened member
`1.027` of it; the `9` percent margin task 4 reported sits inside the
sensitivity.

**At `t = 8` and `t = 12` the statement survives with room to spare.**
Over `t = 0` to `8` the two members score `2.593079e-02` and `2.709352e-02`
(`4.5` percent apart) against the best other recorded row's `4.254200e-02`;
over `t = 0` to `12` they score `4.250925e-02` and `4.331967e-02` (`1.9`
percent) against the 27-key `post_pulse` chain's `6.075829e-02`.

**The two trajectories separate at the first re-selection whose certificate
flips, and never by more than a fifth of their own error.**  The two chains
start from the identical state and carried frontier (the preparation is at
the committed setting; the gate below is bitwise).  At `t = 0.2` the recorded
chain certified its re-selection (conic certificate `Solved`, reduced
gradient `1.247e-08`, under the committed `5e-8`) and the tightened chain
refused the same solve (`1.247e-08` is over `5e-9`) and carried its lift, a
`0.353` jump in the 70 standardized frontier coordinates; its first certified
re-selection is at `t = 0.72`.  The 31-coordinate Euclidean distance between
the chains is `1e-4` at the next sample (`t = 0.2025`), `1e-3` at
`t = 0.5275`, `1e-2` at `t = 1.7375`, and never reaches `1e-1` through
`t = 12`; its maximum is `0.0937` at `t = 11.7475`.  Against the trajectories'
own distance from the exact reference at the same horizon (`0.155` to `0.171`
at `t = 4`, `0.234` to `0.237` at `t = 8`, `0.503` to `0.533` at `t = 12`),
the two members are `21`, `16` and `16` percent of the way apart.

**The whole-extension form of the change cannot even be prepared.**  With
the model's extension itself at `1e-10` the declared completion retry ladder
is exhausted at the exact-correlated preparation: at the base setting the
interior-Newton certificate's KKT backward error is `6.209e-08` against
`5e-9`, at the `1e-8` shift rung `9.076e-08`, and at the `1e-7` shift and
`0.1` weight rungs the conic certificates report `AlmostSolved` with KKT
errors `1.459` and `2.447`.  At the committed `1e-9` the base setting refuses
by the same check (`6.209e-08` over `5e-8`) and the `1e-8` shift rung
certifies at `4.270e-08`, a `15` percent margin under the threshold: the
recorded chain's own preparation is inside a factor `1.2` of refusing.  The
scaled minimum eigenvalue is `-1e-15` at every setting and both tolerances
--- the cone check never binds; the discriminating check of this completion
is the KKT backward error against `50 tol`.  The tightened chain therefore
prepares at the committed setting, keeps the model's extension at the
committed setting (so the carried completion the velocity accepts is
unchanged) and solves only the re-selection completion at `1e-10`, through a
driver-level subclass of `CheapProjectionRoute` whose `reselect_lift` body is
the committed one with the solver swapped.  The cheap route's sixth pass met
the same wall at 15 keys (`AlmostSolved` at the preparation) and made the
same choice.

## Gates

| gate | result |
|---|---|
| the patched driver at the committed setting reproduces the recorded chain | `t = 0` to `0.4` with `--record-reselections` and no tolerance flag: equal under `np.array_equal` to the recorded segment 1 on all 161 samples of times, states, standardized frontier, both eigenvalue series, envelope ratio, archive coordinates, internal energy and its reference (maximum difference `0.0`).  Artifact `paper_v_dictionary_energy_prefixgate_tightdriver_20260916f`, `prefix_gate_tightdriver.json` |
| the re-selection subclass at the committed `1e-9` reproduces it too | the same bitwise equality on the same arrays, and the same 10-entry re-selection ledger (applied, iterations, message) as the first gate.  Artifact `paper_v_dictionary_energy_prefixgate_reselroute_20260916f`, `prefix_gate_reselroute.json` |
| the recording of re-selections changes no number | the two gates above record every re-selection and are bitwise the recorded run, which recorded none |
| a segment resumes on a re-selection boundary | segment 1 stopped at `t = 3.74` and segment 2 resumed at `t = 3.72`, the last accepted step that is a whole number of 16-step intervals from `t = 0`; segment 2 stopped at `7.52` and resumed there; segments 3 and 4 reached their horizons |
| the energy is held where the constraint is on | maximum deviation from the declared reference `1.78e-15` over 14,401 constrained repairs, hyperplane representable at every one |
| the two chains are on one time grid | 4,801 common samples, `t = 0` to `12`, times equal to `1e-12` |

## (a) The chain to `t = 8`, and (b) its continuation to `t = 12`

Every segment is a separate `ram_guard --limit-mb 4500` process, resumed
from the previous segment's saved state and carried frontier, no two ever at
once, with the `1,000 s` autonomous budget of the recorded chain.  The third
segment fit and a fourth reached `t = 12` inside one budget.

| quantity | tightened seg 1 | seg 2 | seg 3 | seg 4 | recorded seg 1 | seg 2 | seg 3 | seg 4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| interval | `0` to `3.74` | `3.72` to `7.52` | `7.52` to `8.0` | `8.0` to `12.0` | `0` to `3.96` | `3.96` to `7.84` | `7.84` to `8.0` | `8.0` to `11.84` |
| accepted steps | 1,496 | 1,520 | 192 | 1,600 | 1,584 | 1,552 | 64 | 1,536 |
| why it stopped | budget | budget | reached `t = 8` | reached `t = 12` | budget | budget | reached `t = 8` | budget |
| autonomous wall | `1,000.0 s` | `1,001.3 s` | `113.5 s` | `961.0 s` | `1,011.7 s` | `1,010.8 s` | `35.5 s` | `1,000.4 s` |
| declared completions re-solved / refused | 93 / 42 | 95 / 48 | 12 / 7 | 100 / 54 | 99 / 32 | 97 / 42 | 4 / 1 | 96 / 32 |
| wall in those solves | `929.9 s` | `930.0 s` | `104.5 s` | `887.6 s` | `939.3 s` | `937.6 s` | `32.4 s` | `926.6 s` |
| repairs applied / skipped | 4,465 / 23 | 4,560 / 0 | 576 / 0 | 4,800 / 0 | 4,711 / 41 | 4,656 / 0 | 192 / 0 | 4,608 / 0 |
| peak resident | `637 MiB` | `642 MiB` | `646 MiB` | `634 MiB` | `629 MiB` | `626 MiB` | `627 MiB` | `611 MiB` |

Chain totals: to `t = 8`, 200 declared completions re-solved with **97
refused** (recorded: 75), `1,964.4 s` of `2,114.8 s` in the solves
(recorded `1,909.4 s` of `2,058.0 s`); to `t = 12`, 300 re-solved with 151
refused, `2,852.0 s` of `3,075.9 s` (`93` percent), `9.51 s` mean solve
(recorded to `t = 11.84`: 296 re-solved, 107 refused).  The tightened chain
is `2.8` percent slower to `t = 8` and every segment stayed inside twenty
minutes.

Of the 151 refusals, by the certificate the completion reported: 69
interior-Newton stationarity certificates whose KKT backward error is over
`5e-9`; 41 `conic Newton step failed descent`; 15 conic certificates `Solved`
with the KKT error over the threshold; 15 reference completions
`PrimalInfeasible` and 3 `AlmostPrimalInfeasible`; 7 facial reductions that
removed every frontier direction; 1 `AlmostSolved`.  All 149 certified
re-selections carry the interior-Newton certificate.  27 of the 234
interior-Newton solves have their reduced gradient in `(5e-9, 5e-8]`, the
band the decade moves from certified to refused.  Refusals by window: 31 of
50 re-selections over `(0, 2]`, 11 of 50 over `(2, 4]`, 25 of 49 over
`(4, 6]`, 30 of 51 over `(6, 8]`, 29 of 50 over `(8, 10]`, 25 of 50 over
`(10, 12]`.

Cumulative scores from `t = 0`, tightened member against the recorded one;
the relative column is (tightened minus recorded) over recorded; the best
other row is the reviewer's corrected one:

| horizon | all-coordinate, recorded | tightened | relative | Eq. (22) internal energy, recorded | tightened | best other row on record | statement |
|---|---:|---:|---:|---:|---:|---:|---|
| `t = 4` | `1.142716e-02` | `1.292283e-02` | `+13.1 %` | `4.597225e-03` | `3.687212e-03` | `1.258000e-02` (cell A) | fails for the tightened member |
| `t = 6` | `2.090571e-02` | `2.314018e-02` | `+10.7 %` | `5.394635e-03` | `4.390441e-03` | --- | --- |
| `t = 8` | `2.593079e-02` | `2.709352e-02` | `+4.5 %` | `5.751890e-03` | `4.702571e-03` | `4.254200e-02` (15-key `post_pulse`) | survives, factor `1.57` on the worse member |
| `t = 10` | `3.332970e-02` | `3.356926e-02` | `+0.7 %` | `5.955999e-03` | `4.880305e-03` | --- | --- |
| `t = 12` | `4.250925e-02` | `4.331967e-02` | `+1.9 %` | `6.088286e-03` | `4.995294e-03` | `6.075829e-02` (27-key `post_pulse`) | survives, factor `1.40` |

The rest of the chart at the same horizons (site-0 occupation is
`rho[0,0]` sample RMS for both members, the chain's own convention; the
reviewer's factor-2 note applies only where a record value is quoted):

| quantity | `t = 4`, recorded / tightened | `t = 8` | `t = 12` |
|---|---:|---:|---:|
| the 27 coordinates outside the phonon-occupation block | `1.164437e-02` / `1.347560e-02` | `2.474607e-02` / `2.577152e-02` | `4.385377e-02` / `4.483890e-02` |
| the four phonon-occupation coordinates | `9.836299e-03` / `8.276556e-03` | `3.282777e-02` / `3.472464e-02` | `3.198825e-02` / `3.118564e-02` |
| blockwise, rho / B / N / A / C, recorded | `2.0368e-02` / `6.9088e-03` / `9.8363e-03` / `8.9418e-03` / `1.1167e-02` | `4.0173e-02` / `2.7248e-02` / `3.2828e-02` / `1.9942e-02` / `2.1275e-02` | `6.9808e-02` / `7.5185e-02` / `3.1988e-02` / `3.0539e-02` / `2.5493e-02` |
| blockwise, tightened | `2.2501e-02` / `7.2741e-03` / `8.2766e-03` / `1.2051e-02` / `1.2820e-02` | `4.1681e-02` / `2.5356e-02` / `3.4725e-02` / `2.2288e-02` / `2.2628e-02` | `7.2200e-02` / `7.3405e-02` / `3.1186e-02` / `3.5349e-02` / `2.6180e-02` |
| site-0 occupation | `1.430879e-02` / `1.515553e-02` | `3.027929e-02` / `3.095132e-02` | `5.278107e-02` / `5.430664e-02` |
| minimum unshifted retained joint Gram | `-4.535e-04` / `-4.511e-04` | `-2.152e-03` / `-2.201e-03` | `-2.463e-03` / `-2.374e-03` |
| minimum scaled positive-extension eigenvalue | `-4.852e-05` / `-5.846e-05` | `-1.922e-04` / `-1.980e-04` | `-2.040e-04` / `-2.098e-04` |
| maximum standardized frontier envelope ratio | `0.3694` / `0.3605` | `0.4518` / `0.4212` | `0.4518` / `0.4365` |
| Euclidean coordinate difference from the exact reference at the horizon, all 31 | `1.551037e-01` / `1.706443e-01` | `2.372364e-01` / `2.335909e-01` | `5.027756e-01` / `5.331822e-01` |

Three readings of the table.

1. **The internal energy is the one column the tightened member improves,
   by `18` to `20` percent at every horizon, and the reason is its
   reference, not its repair.**  The `drive_power` reference the tightened
   chain selects at `t = 4` is `-3.565863978942048`, `0.005534` above the
   exact `-3.571397982`, where the recorded chain's is
   `-3.564687762684483`, `0.006710` above: the two references differ by
   `1.176e-03`, and that is exactly the difference of the two chains'
   Eq. (22) energies at every sample after `t = 4` (maximum `2.168e-03`
   during the drive).  The physicality minima agree to `2` to `5` percent,
   so neither member holds a cone and the task-4 physicality statement is
   unchanged.
2. **The block that moves the score is not the phonon-occupation block.**
   At `t = 4` the tightened member is better on the four phonon-occupation
   coordinates (`8.277e-03` against `9.836e-03`) and worse on the other 27
   (`1.348e-02` against `1.164e-02`), the A and C correlation blocks most
   (`1.35` and `1.15` times).  The task-4 finding that the energy constraint
   reverses the enlargement's phonon-occupation penalty holds for both
   members (`8.277e-03` and `9.836e-03` against the unconstrained
   `2.975e-02`).
3. **The pointwise separation is still growing at `t = 12` while the
   cumulative score's sensitivity is shrinking.**  Window means of the
   31-coordinate distance between the chains: `0.0162` over `(0, 4]`,
   `0.0346` over `(4, 8]`, `0.0532` over `(8, 12]`, with the maximum
   `0.0937` at `t = 11.75` and `0.0828` at `t = 12`; the relative
   difference of the cumulative score runs `13.1`, `10.7`, `4.5`, `0.7`,
   `1.9` percent at `t = 4`, `6`, `8`, `10`, `12`.  The cumulative RMS
   averages over a horizon that is mostly common history, so its agreement
   past `t = 8` is not evidence that the trajectories agree there.

Times at which the two trajectories first separate, in the Euclidean
distance at equal sample times:

| threshold | 31 closed scalar coordinates | 56 state coordinates | 70 standardized frontier coordinates |
|---:|---:|---:|---:|
| `1e-6`, `1e-5`, `1e-4` | `0.2025` | `0.2025` | `0.2` |
| `1e-3` | `0.5275` | `0.23` | `0.2` |
| `1e-2` | `1.7375` | `0.545` | `0.2` |
| `1e-1` | never through `t = 12` (maximum `0.0937` at `11.7475`) | `2.485` | `0.2` |
| `1` | never | never (maximum `0.337` at `10.9925`) | never (maximum `0.353`) |

Beside the two sibling measurements on this branch: the cheap route's 15-key
`N = 4` `post_pulse` pair separated from a perturbation of order `1e-5` to
`1e-2` at `t = 6.93`, `0.1` at `t = 8.38` and order one by `t = 11`; its
sixth pass's 15-key `N = 16` `drive_power` rerun at the same re-selection
tolerance `1e-10` flipped at its first re-selection (`t = 0.04`), crossed
`1e-2` at `t = 0.48` and `0.1` at `t = 3.085`, and reached a maximum of
`0.72` near `t = 12` to `13`, with an `18` percent score change over
`t = 0` to `4`.  Here the perturbation is likewise a whole lift (`0.353` in
the frontier, at `t = 0.2`), the distance reaches `1e-2` at `t = 1.74` and
stays under `0.1` through `t = 12`, and the `t = 0` to `4` score changes by
`13` percent.  The 27-key `N = 16` `drive_power` chain is the least
sensitive of the three configurations by the pointwise measure and the same
as the 15-key one by the cumulative one, and the amplifier the cheap route
named is the one measured here: every difference between the two chains
descends from re-selections whose certificate flipped, `97` refusals
against `75` over the first 200, with the first flip at `t = 0.2`.

## (c) Which recorded horizons carry a second-digit meaning

| horizon | recorded | tightened member | second digit | "best on record" at that horizon, against the reviewer's corrected comparator |
|---|---:|---:|---|---|
| `t = 4` | `1.142716e-02` | `1.292283e-02` | no: `1.1e-2` against `1.3e-2`; only the leading digit is stable | does not survive: cell A's `1.258000e-02` lies between the two members, and the `9` percent margin is inside the `13` percent sensitivity.  Both members remain below the recorded projective event guard's `2.604900e-02` and every 15-key row |
| `t = 8` | `2.593079e-02` | `2.709352e-02` | one unit: `2.6e-2` against `2.7e-2` | survives: the best other row `4.254200e-02` is `1.57` times the worse member |
| `t = 12` | `4.250925e-02` | `4.331967e-02` | yes: both round to `4.3e-2` | survives: the 27-key `post_pulse` chain's `6.075829e-02` is `1.40` times the worse member |
| `t = 16` | `6.377276e-02` | not measured | unmeasured | already withdrawn by the review (a tie with the 27-key `post_pulse` chain's `6.385168e-02`, `0.12` percent apart); a tie inside any sensitivity measured here is not an ordering |
| `t = 20` | `8.519313e-02` | not measured | unmeasured | already withdrawn by the review (second to the 27-key `post_pulse` chain's `6.990927e-02`); the `18` percent gap between the two 27-key chains is the size of the sensitivities measured at `t = 4` here and at `t = 20` for the 15-key `N = 16` chain (`26` percent), so which 27-key chain is the better `t = 20` row is unresolved until both are measured there |

What the record should say from now on: the 27-key `N = 16` `drive_power`
chain scores `1.1e-2` to `1.3e-2` over `t = 0` to `4` depending on the
completion's certificate tolerance, which brackets cell A's `1.258e-02`; it
is the best moment-closure row on record over `t = 0` to `8` and `t = 0` to
`12` under either setting; and its `t = 16` and `t = 20` values are single
samples of a configuration whose sensitivity is measured only to `t = 12`.
The recorded chain remains the reported one --- nothing here says the
tightened member is the better estimate --- but a second digit at `t = 4`
is not a property of the configuration.

## Provenance and gates of the runs themselves

- Every rollout was wrapped in `pipelines/shell/ram_guard.py --limit-mb
  4500`, one at a time; the largest peak resident was `646 MiB` and the
  longest segment `1,001.3 s` of autonomous wall (plus about `20 s` of build
  and under `2 s` of exact-reference scoring).  Wall times were measured
  with other agents' jobs possibly on the machine and are upper bounds.
- No module was edited, nothing was committed, no default, acceptance rule
  or ladder changed.  The edits are to driver scripts under
  `output/local_runs/paper_v_projective_dictionary_energy_20260915e/`, which
  are not repository modules and not in the GitNexus index; no impact
  analysis applies to them.  `run_energy_segment.py` gained
  `--conic-tolerance` (the whole extension; the form that fails at the
  preparation), `--reselection-conic-tolerance` (the form run, through the
  `ReselectionToleranceRoute` subclass defined in the driver) and
  `--record-reselections` (writes the integrator's per-re-selection records
  to `reselections.json`; changes no number, gate above), and records the
  effective tolerances in `summary.json`; `run_chain.py` forwards the three;
  `compare_prefix.py` compares three more arrays when both runs carry them.
  With none of the new flags the driver is the recorded one, bitwise.
- New scripts in the same directory: `compare_chains.py` (the separation
  tables; reads only `trajectory.npz`, `summary.json` and
  `reselections.json`), `tabulate_task5.py` (the ledger, horizon and
  refusal tables), `probe_preparation_tolerance.py` with its `.sh` (the
  preparation's completion at both tolerances and every rung),
  `prefixgate_tightdriver.sh`, `prefixgate_reselroute.sh`, `queue_tight.sh`
  (the chain, one segment at a time).
- Outputs in that directory: `preparation_tolerance_probe.{json,log}`,
  `prefix_gate_tightdriver.json`, `prefix_gate_reselroute.json`,
  `compare_tightresel_{partial_seg1,partial_seg2,t8,t12}.json`,
  `tightresel_chain_score_{partial,t8,t12}.json`,
  `power_chain_score_t10.json` (the recorded chain rescored at `t = 10`),
  `tight_seg1.log` (the whole-extension attempt that exhausted the ladder),
  `tightresel_seg1.log`, `tightresel_chain.log` and the per-segment logs.
- Artifacts, under `output/local_runs/`:
  `paper_v_dictionary_energy_power_27key_n16_tightresel_20260916f_seg{1,2,3,4}`
  (each with `reselections.json`),
  `paper_v_dictionary_energy_prefixgate_tightdriver_20260916f`,
  `paper_v_dictionary_energy_prefixgate_reselroute_20260916f`.  The recorded
  chain's artifacts are untouched.

## Structured summary of task 5

| item | value |
|---|---|
| the lever | `APCMExtensionSettings.conic_tolerance` of the re-selection completion, `1e-10` against the committed `1e-9`; preparation, model extension, repair, velocity, integrator, cadence and reference mode identical; imports from `head_src` at `2d8f0efb` |
| the whole-extension form | not preparable: the declared completion retry ladder is exhausted at `t = 0` (KKT backward errors `6.209e-08`, `9.076e-08`, `1.459`, `2.447` against `5e-9`); at `1e-9` the `1e-8` shift rung certifies at `4.270e-08` against `5e-8`, a `15` percent margin; the cone check never binds (`-1e-15`) |
| (a) `t = 0` to `8`, three segments, `2,114.8 s` | `2.709352e-02` against the recorded `2.593079e-02` (`+4.5 %`); internal energy `4.702571e-03` against `5.751890e-03`; retained joint Gram `-2.201e-03` against `-2.152e-03`, extension `-1.980e-04` against `-1.922e-04`; 200 completions re-solved, 97 refused against 75 |
| (a) `t = 0` to `4` | `1.292283e-02` against `1.142716e-02` (`+13.1 %`); internal energy `3.687212e-03` against `4.597225e-03`; phonon-occupation block `8.276556e-03` against `9.836299e-03`; the other 27 `1.347560e-02` against `1.164437e-02` |
| separation, 31-coordinate Euclidean distance | identical at `t = 0`; first flip at the `t = 0.2` re-selection (recorded certified at reduced gradient `1.247e-08`, tightened refused); `1e-3` at `t = 0.5275`, `1e-2` at `t = 1.7375`, `1e-1` never through `t = 12` (maximum `0.0937` at `t = 11.7475`); 56-state distance `1e-1` at `t = 2.485` |
| (b) `t = 0` to `12`, four segments, `3,075.9 s` | `4.331967e-02` against `4.250925e-02` (`+1.9 %`); `t = 10`: `3.356926e-02` against `3.332970e-02` (`+0.7 %`); internal energy `4.995294e-03` against `6.088286e-03`; retained `-2.374e-03` against `-2.463e-03`; 300 re-solved, 151 refused against 107 by `t = 11.84` |
| (c) second-digit meaning | `t = 4`: no (`1.1e-2` to `1.3e-2`); `t = 8`: one unit (`2.6e-2` to `2.7e-2`); `t = 12`: yes (`4.3e-2`); `t = 16`, `20`: unmeasured |
| (c) "best on record" | fails at `t = 4` (cell A `1.258e-02` between the members); survives at `t = 8` (factor `1.57`) and `t = 12` (`1.40` against the 27-key `post_pulse` chain); at `t = 16` and `t = 20` already withdrawn by the review, and the `t = 20` ordering of the two 27-key chains (`18` percent apart) is inside the measured sensitivities |
| the amplifier | the completion's certificate: 27 of 234 interior-Newton solves have reduced gradients in `(5e-9, 5e-8]`; refusal fractions by window `31/50`, `11/50`, `25/49`, `30/51`, `29/50`, `25/50` over `(0,2]` to `(10,12]` |
| gates | two bitwise prefix gates (161 samples, `0.0`); resumption on 16-step boundaries; energy held to `1.78e-15` over 14,401 constrained repairs |
| files changed | `run_energy_segment.py`, `run_chain.py`, `compare_prefix.py` (drivers under `output/local_runs/paper_v_projective_dictionary_energy_20260915e/`), new `compare_chains.py`, `tabulate_task5.py`, `probe_preparation_tolerance.{py,sh}`, `prefixgate_{tightdriver,reselroute}.sh`, `queue_tight.sh`; this report; the worklog block.  No module; nothing committed |
| next | two more segments carry the tightened chain to `t = 20` and settle the ordering of the two 27-key chains there; the same lever on the 27-key `post_pulse` chain; a decade looser (`1e-8`) measures the other direction, where the recorded preparation's `15` percent margin says most refusals would certify |
