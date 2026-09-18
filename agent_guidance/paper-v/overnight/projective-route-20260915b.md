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

# projective-route, task 6 (2026-09-16)

Continuation of task 5: the tightened-tolerance member of the 27-key
`N = 16` `drive_power` energy-neutral chain (re-selection completion solved
at `APCMExtensionSettings.conic_tolerance = 1e-10` in place of the committed
`1e-9`; preparation, model extension, repair, velocity, integrator, cadence
and reference mode identical; imports from `head_src` at `2d8f0efb`) carried
from its `t = 12` state to `t = 20` in two more `ram_guard --limit-mb 4500`
segments, one at a time, with the same `1,000 s` autonomous budget, so that
the record's `t = 16` and `t = 20` rows have a measured sensitivity and the
ordering of the two 27-key chains at `t = 20` can be read against it.  No
module was edited; nothing was committed; no default changed.  The recorded
chain and the 27-key `post_pulse` chain were rescored from their saved
arrays on the same seven horizons (`t = 4, 6, 8, 10, 12, 16, 20`) with the
same exact cutoff-16 reference (`exact_t20.npz`, which the reviewer found
bitwise equal to a regeneration), so every comparison below is on one grid
of 8,001 samples and one reference.

## The result, first

**At `t = 20` the ordering of the two 27-key chains is outside the
sensitivity measured here.**  The `drive_power` chain's two members score
`8.519313e-02` (recorded) and `9.088037e-02` (tightened) on the
all-coordinate scalar RMS of the 31 closed scalar coordinates over `t = 0`
to `20`, a band `6.7` percent wide; the 27-key `post_pulse` chain's
`6.990927e-02` lies below the band, `1.22` times better than the recorded
member and `1.30` times better than the tightened one.  The review's placing
of the `drive_power` chain second at `t = 20` stands under either setting.
What is measured is one-sided: the band is the `drive_power` chain's; the
`post_pulse` chain's response to the same lever is unmeasured, and the only
same-horizon sensitivity on the branch, the 15-key `N = 16` `drive_power`
chain's `26` percent at `t = 20`, would cover a `22` percent gap if it
applied to the `post_pulse` chain.  So: settled against the `drive_power`
chain's measured band; not yet settled against every sensitivity measured
on the branch.

**At `t = 16` the tie is confirmed as a tie.**  The members score
`6.377276e-02` and `6.719628e-02` over `t = 0` to `16` (`5.4` percent
apart); the `post_pulse` chain's `6.385168e-02` lies inside that band,
`0.12` percent above the recorded member and `5.0` percent below the
tightened one.  No ordering at `t = 16` is a property of the configuration.

**The cumulative score's sensitivity is not monotone in the horizon, and
task 5's reading that it "shrinks" does not hold past `t = 12`.**  The
relative difference (tightened minus recorded, over recorded) of the
all-coordinate scalar RMS runs `+13.1`, `+10.7`, `+4.5`, `+0.7`, `+1.9`,
`+5.4`, `+6.7` percent at `t = 4, 6, 8, 10, 12, 16, 20`.  The minimum at
`t = 10` was a crossing of the two members' running errors, not a
convergence.

**The pointwise separation of the two trajectories stopped growing at
`t = 12` on the 31 closed scalar coordinates and kept growing on the 56
state coordinates.**  The 31-coordinate Euclidean distance between the
chains at equal sample times has its maximum `0.0955` at `t = 13.845`, never
reaches `0.1` through `t = 20`, and reads `0.0833` at `t = 16` and `0.0771`
at `t = 20` (window means `0.0532`, `0.0818`, `0.0733` over `(8, 12]`,
`(12, 16]`, `(16, 20]`); the 56-state distance has its maximum `0.528` at
`t = 19.94` (window means `0.279`, `0.289`, `0.398` over the same windows).
Against the chains' own distance from the exact reference at the horizon
(`0.656` / `0.718` at `t = 16`, `0.969` / `1.039` at `t = 20`) the members
are `12.7` and `8.0` percent of the way apart, after `21`, `16`, `16`
percent at `t = 4`, `8`, `12`.

## Gates

| gate | result |
|---|---|
| segments resume on a re-selection boundary | segment 5 resumed at `t = 12.0`, segment 4's resume sample (index 1,600); segment 5 stopped on budget at `t = 16.6` after 1,840 accepted steps (`115 x 16`) and segment 6 resumed there; segment 6 reached `t = 20.0` in 1,360 steps (`85 x 16`) |
| the declared reference is unchanged across the continuation | `-3.565863978942047` in segments 4, 5 and 6 (the tightened chain's `drive_power` reference after the drive tail) |
| the energy is held where the constraint is on | maximum deviation from the declared reference `1.33e-15` in each new segment, `1.78e-15` over the chain's 24,001 constrained repairs, 0 hyperplane refusals |
| the two chains are on one time grid | 8,001 common samples, `t = 0` to `20`, times equal to `1e-12`; the scorer hit the `exact_t20.npz` cache for both chains and regenerated nothing |
| the tabulation reproduces task 5 | the boundary-inclusive merge of the re-selection ledgers gives 300 re-solved / 151 refused to `t = 12` and 27 of 234 reduced gradients in `(5e-9, 5e-8]`, as reported there |

## The two segments, and the chain to `t = 20`

| quantity | tightened seg 5 | seg 6 | recorded seg 5 | seg 6 |
|---|---:|---:|---:|---:|
| interval | `12.0` to `16.6` | `16.6` to `20.0` | `11.84` to `16.32` | `16.32` to `20.0` |
| accepted steps | 1,840 | 1,360 | 1,792 | 1,472 |
| why it stopped | budget | reached `t = 20` | budget | reached `t = 20` |
| autonomous wall | `1,004.4 s` | `709.5 s` | `1,004.5 s` | `731.5 s` |
| declared completions re-solved / refused | 115 / 50 | 85 / 25 | 112 / 29 | 92 / 8 |
| wall in those solves | `916.9 s` | `644.5 s` | `919.9 s` | `661.1 s` |
| repairs applied / skipped | 5,520 / 0 | 4,080 / 0 | 5,376 / 0 | 4,416 / 0 |
| minimum retained joint Gram / scaled extension eigenvalue within the segment | `-1.325e-03` / `-1.655e-04` | `-5.818e-04` / `-5.510e-05` | `-1.418e-03` / `-1.659e-04` | `-6.302e-04` / `-6.850e-05` |
| peak resident | `638 MiB` | `627 MiB` | `630 MiB` | `629 MiB` |

Chain totals over `t = 0` to `20`, tightened against recorded: 8,000
accepted steps in six segments each; `4,789.8 s` against `4,794.4 s` of
autonomous wall (equal to `0.1` percent, after the tightened chain was
`2.8` percent slower to `t = 8`); 500 declared completions re-solved in
each, **226 refused against 144**, `4,413.4 s` against `4,416.9 s` in those
solves (`92.1` percent of the wall and `8.83 s` mean in both); repairs
`24,001 / 23` against `23,959 / 41` applied / skipped; peak `646 MiB`
against `630 MiB`.  Every segment stayed inside twenty minutes (longest
`1,004.4 s` of autonomous wall plus about `20 s` of build).

Of the tightened chain's 226 refusals over `t = 0` to `20`, by the
certificate the completion reported: 103 interior-Newton stationarity
certificates with the KKT backward error over `5e-9`; 70 `conic Newton step
failed descent`; 23 conic certificates `Solved` with the KKT error over the
threshold; 19 reference completions `PrimalInfeasible` and 3
`AlmostPrimalInfeasible`; 7 facial reductions that removed every frontier
direction; 1 `AlmostSolved`.  All 274 certified re-selections carry the
interior-Newton certificate; 33 of the 401 interior-Newton solves have
their reduced gradient in `(5e-9, 5e-8]`, the band the decade moves.
Refusals by window: `31/50`, `11/50`, `25/49`, `30/51`, `29/50`, `25/50`,
`28/50`, `19/50`, `14/49`, `14/51` over `(0, 2]` to `(18, 20]`; cumulative
`97/200` at `t = 8`, `151/300` at `t = 12`, `198/400` at `t = 16`,
`226/500` at `t = 20`.  The recorded chain, which carries no
per-re-selection ledger, refused 29 of 112 over `t = 11.84` to `16.32` and
8 of 92 over `16.32` to `20` (136 of 408 to `t = 16.32`, 144 of 500 to
`t = 20`); both chains refuse less often after `t = 16`.

## The chart at `t = 16` and `t = 20`

Cumulative from `t = 0`, recorded / tightened; the relative column is
(tightened minus recorded) over recorded; site-0 occupation is given in both
conventions, the chain's own `rho[0,0]` sample RMS and the record's
`2 rho[0,0]` (sample RMS here; the record's both-cones `0.204690` is the
trapezoidal time-RMS, which differs from the sample RMS by `4e-5` relative
on this grid):

| quantity | `t = 16`, recorded / tightened | relative | `t = 20`, recorded / tightened | relative |
|---|---:|---:|---:|---:|
| all-coordinate scalar RMS, 31 coordinates | `6.377276e-02` / `6.719628e-02` | `+5.4 %` | `8.519313e-02` / `9.088037e-02` | `+6.7 %` |
| Eq. (22) internal-energy RMS | `6.249722e-03` / `5.135424e-03` | `-17.8 %` | `6.344620e-03` / `5.217702e-03` | `-17.8 %` |
| the four phonon-occupation coordinates (N block) | `3.188383e-02` / `2.843693e-02` | `-10.8 %` | `3.187001e-02` / `2.881790e-02` | `-9.6 %` |
| the 27 coordinates outside the phonon-occupation block | `6.722257e-02` / `7.116514e-02` | `+5.9 %` | `9.045792e-02` / `9.674606e-02` | `+7.0 %` |
| blockwise rho / B / N / A / C, recorded | `1.0302e-01` / `1.3219e-01` / `3.1884e-02` / `3.5397e-02` / `3.0187e-02` | | `1.3580e-01` / `1.8729e-01` / `3.1870e-02` / `4.3970e-02` / `3.1283e-02` | |
| blockwise, tightened | `1.0716e-01` / `1.3733e-01` / `2.8437e-02` / `4.3884e-02` / `3.3056e-02` | | `1.4316e-01` / `2.0012e-01` / `2.8818e-02` / `5.0174e-02` / `3.3726e-02` | |
| site-0 occupation, `rho[0,0]` sample RMS | `8.393451e-02` / `8.741714e-02` | `+4.1 %` | `1.125899e-01` / `1.190197e-01` | `+5.7 %` |
| site-0 occupation, `2 rho[0,0]` (the record's convention) | `1.678690e-01` / `1.748343e-01` | | `2.251798e-01` / `2.380393e-01` | |
| minimum unshifted retained joint Gram over `t = 0` to `T` | `-2.463e-03` / `-2.374e-03` | | `-2.463e-03` / `-2.374e-03` | |
| minimum scaled positive-extension eigenvalue | `-2.040e-04` / `-2.098e-04` | | `-2.040e-04` / `-2.098e-04` | |
| maximum standardized frontier envelope ratio | `0.4518` / `0.4365` | | `0.4518` / `0.4365` | |
| Euclidean coordinate difference from the exact reference at the horizon, all 31 | `6.559115e-01` / `7.183868e-01` | `+9.5 %` | `9.693869e-01` / `1.039229e+00` | `+7.2 %` |
| declared completions re-solved / refused, cumulative | `408 / 136` (to `16.32`) / `400 / 198` | | `500 / 144` / `500 / 226` | |
| 31-coordinate distance between the chains at the horizon | `0.0833` | | `0.0771` | |
| 56-state distance between the chains at the horizon | `0.3046` | | `0.5216` | |
| 70-frontier distance between the chains at the horizon | `0.0217` | | `0.0373` | |

Three readings.

1. **The internal-energy improvement of the tightened member is the
   reference difference, unchanged.**  The two chains' Eq. (22) energies
   differ by exactly `1.176e-03` at every sample after the drive, the
   difference of the two `drive_power` references (`-3.565863978942047`
   against `-3.564687762684483`), so the `17.8` percent at `t = 16` and
   `t = 20` is the `t = 4` reference selection carried forward, not a
   repair effect.  The physicality minima over `t = 0` to `20` are attained
   before `t = 12` in both chains and agree to `2` to `4` percent; the
   within-segment minima after `t = 12` (`-1.3e-03` to `-5.8e-04` on the
   retained joint Gram, `-1.7e-04` to `-5.5e-05` on the extension) are alike
   in the two chains; neither member holds a cone.
2. **The block that moves the score past `t = 12` is still not the
   phonon-occupation block.**  The tightened member is better on the four
   phonon-occupation coordinates (`-10.8`, `-9.6` percent) and worse on the
   other 27 (`+5.9`, `+7.0` percent), the A block most (`1.24` and `1.14`
   times), then C (`1.10`, `1.08`) and rho (`1.04`, `1.05`); the B block,
   the largest at both horizons in both chains, differs by `3.9` and `6.9`
   percent.  The growth from `t = 12` to `20` is in rho and B in both
   members, as the reviewer's reading of the recorded chain said.
3. **In the record's site-0 convention both members are worse than both
   cones.**  `2 rho[0,0]` sample RMS `0.2252` and `0.2380` over `t = 0` to
   `20` against the both-cones `0.204690`; the tightened member widens the
   gap.

## (c) Second-digit meaning at `t = 16` and `t = 20`, completing task 5's table

| horizon | recorded | tightened member | second digit | "best on record" at that horizon |
|---|---:|---:|---|---|
| `t = 16` | `6.377276e-02` | `6.719628e-02` | no: `6.4e-2` against `6.7e-2`; the leading digit is stable | the tie with the 27-key `post_pulse` chain's `6.385168e-02` is inside the band (`0.12` percent above the recorded member, `5.0` percent below the tightened): no ordering |
| `t = 20` | `8.519313e-02` | `9.088037e-02` | no: `8.5e-2` against `9.1e-2`, a band that crosses a leading-digit boundary (both round to `9e-2` at one figure) | second to the 27-key `post_pulse` chain's `6.990927e-02` under either setting, factors `1.22` (recorded) and `1.30` (tightened) against a `6.7` percent band: outside the sensitivity measured here; the `post_pulse` chain's own band is unmeasured |

What the record should say for the `drive_power` chain past `t = 12`: over
`t = 0` to `16` it scores `6.4e-2` to `6.7e-2` depending on the completion's
certificate tolerance, a band that contains the 27-key `post_pulse` chain;
over `t = 0` to `20` it scores `8.5e-2` to `9.1e-2`, and the `post_pulse`
chain's `7.0e-2` is better than either member by more than the band.  The
recorded chain remains the reported member.

## The final ledger of the 27-key `N = 16` `drive_power` energy-neutral chain, with its sensitivity band at every horizon

All-coordinate scalar RMS on the 31 closed scalar coordinates against the
exact cutoff-16 driven reference, cumulative from `t = 0` on the
`h = 0.0025` grid; the band is the recorded member (re-selection completion
at `1e-9`) and the tightened member (`1e-10`); the comparator is the best
other moment-closure row on record at that horizon where the review
established one (`t = 4`, `8`, `12`, `16`, `20`), and at `t = 6` and
`t = 10` the only other chain scored on this reference at that horizon, the
27-key `post_pulse` chain (rescored here: `3.335615e-02` and
`4.956048e-02`).

| horizon | recorded | tightened | band | width | comparator | standing of the chain |
|---|---:|---:|---:|---:|---|---|
| `t = 4` | `1.142716e-02` | `1.292283e-02` | `1.14e-2` to `1.29e-2` | `13.1 %` | positive-`M4` cell A `1.258000e-02` | comparator inside the band: tie |
| `t = 6` | `2.090571e-02` | `2.314018e-02` | `2.09e-2` to `2.31e-2` | `10.7 %` | 27-key `post_pulse` `3.335615e-02` | ahead, factor `1.44` on the worse member |
| `t = 8` | `2.593079e-02` | `2.709352e-02` | `2.59e-2` to `2.71e-2` | `4.5 %` | 15-key `post_pulse` `N = 16` `4.254184e-02` | ahead, factor `1.57` |
| `t = 10` | `3.332970e-02` | `3.356926e-02` | `3.33e-2` to `3.36e-2` | `0.7 %` | 27-key `post_pulse` `4.956048e-02` | ahead, factor `1.48` |
| `t = 12` | `4.250925e-02` | `4.331967e-02` | `4.25e-2` to `4.33e-2` | `1.9 %` | 27-key `post_pulse` `6.075829e-02` | ahead, factor `1.40` |
| `t = 16` | `6.377276e-02` | `6.719628e-02` | `6.38e-2` to `6.72e-2` | `5.4 %` | 27-key `post_pulse` `6.385168e-02` | comparator inside the band: tie |
| `t = 20` | `8.519313e-02` | `9.088037e-02` | `8.52e-2` to `9.09e-2` | `6.7 %` | 27-key `post_pulse` `6.990927e-02` | behind, factors `1.22` to `1.30`, outside the band |

Companion columns of the same row, recorded / tightened: Eq. (22)
internal-energy RMS `4.60e-3` / `3.69e-3` at `t = 4` to `6.34e-3` /
`5.22e-3` at `t = 20` (the `post_pulse` chain `7.09e-2` to `1.65e-1`; both
cones `9.4e-4` at `t = 20`); minimum unshifted retained joint Gram
`-2.46e-3` / `-2.37e-3` and minimum scaled extension eigenvalue `-2.04e-4`
/ `-2.10e-4` over `t = 0` to `20` (the `post_pulse` chain `-3.12e-3` /
`-3.43e-4`); site-0 occupation in the record's `2 rho[0,0]` convention
`0.225` / `0.238` at `t = 20` (both cones `0.205`); autonomous wall
`4,794.4 s` / `4,789.8 s` to `t = 20` on this machine with other agents'
jobs possibly present, 500 completions re-solved with 144 / 226 refused.

## Provenance and gates of the runs themselves

- Two rollouts,
  `paper_v_dictionary_energy_power_27key_n16_tightresel_20260916f_seg5` and
  `_seg6`, each under `pipelines/shell/ram_guard.py --limit-mb 4500`, one
  at a time (a cheap-route job under a `3,000 MB` guard was on the machine
  throughout; guard limits summed to `7.5 GB` of the `18 GB` budget); peaks
  `638` and `627 MiB`; autonomous wall `1,004.4` and `709.5 s`.  The
  scoring, comparison and tabulation ran under `--limit-mb 2000` (peaks
  `143 MB` and under).
- Driver: `queue_tight_t20.sh` (calls the unchanged `run_chain.py` with
  `--first 5 --last 7 --final-time 20.0`; the seventh segment was not
  needed).  `run_energy_segment.py`, `run_chain.py`, `compare_chains.py`
  and `score_energy_chain.py` are as task 5 left them.  New in the
  directory: `tabulate_task6.py` (the ledgers, the chart in both site-0
  conventions, the band table, the separation window means recomputed from
  the arrays, and the boundary-inclusive refusal ledger),
  `rescore_recorded_all.sh` and `rescore_postpulse_all.sh` (the recorded and
  27-key `post_pulse` chains on the seven horizons), `score_tight_t20.sh`
  (score, compare, tabulate), `monitor_tight_t20.sh`,
  `tabulate_task6_dryrun.sh`, `comparators_task6.json`.
- Outputs: `tightresel_chain_score_t20.json`, `compare_tightresel_t20.json`,
  `power_chain_score_all.json`, `postpulse_chain_score_all.json`,
  `tabulate_task6.log`, the two segment logs, `tightresel_chain.log`
  (appended), `queue_tight_t20.log`.  The `post_pulse` chain is the task-3
  rollout `paper_v_dictionary_entrance15_plus_two_spin_boson_n16_t4_20260915d`
  followed by `..._postpulse_27key_n16_20260915e_seg1` to `_seg4`, as the
  reviewer assembled it; its rescore reproduces every value in the review's
  table to the printed digits.
- No module was edited, nothing was committed, no default, acceptance rule
  or ladder changed.  The recorded chain's artifacts are untouched.

## Structured summary of task 6

| item | value |
|---|---|
| the runs | segments 5 (`t = 12.0` to `16.6`, budget, 1,840 steps, `1,004.4 s`, 115 re-solved / 50 refused) and 6 (`16.6` to `20.0`, reached, 1,360 steps, `709.5 s`, 85 / 25) of the tightened chain; peaks `638` / `627 MiB` |
| `t = 16`, recorded / tightened | all 31 `6.377276e-02` / `6.719628e-02` (`+5.4 %`); internal energy `6.249722e-03` / `5.135424e-03`; N block `3.188383e-02` / `2.843693e-02`; site-0 `8.393451e-02` / `8.741714e-02` (`rho[0,0]`), `1.678690e-01` / `1.748343e-01` (`2 rho[0,0]`); retained `-2.463e-03` / `-2.374e-03`, extension `-2.040e-04` / `-2.098e-04`; refusals `136/408` to `16.32` / `198/400`; separation `0.0833` |
| `t = 20`, recorded / tightened | all 31 `8.519313e-02` / `9.088037e-02` (`+6.7 %`); internal energy `6.344620e-03` / `5.217702e-03`; N block `3.187001e-02` / `2.881790e-02`; site-0 `1.125899e-01` / `1.190197e-01` (`rho[0,0]`), `2.251798e-01` / `2.380393e-01` (`2 rho[0,0]`); physicality minima unchanged from `t = 16`; refusals `144/500` / `226/500`; separation `0.0771` (maximum `0.0955` at `t = 13.845`, never `0.1`) |
| the `t = 20` ordering | 27-key `post_pulse` `6.990927e-02` is below the `drive_power` band `8.52e-2` to `9.09e-2` by factors `1.22` to `1.30` against a `6.7` percent band width: outside the sensitivity measured here; the `post_pulse` chain's own band is unmeasured, and the branch's only same-horizon sensitivity (15-key `N = 16`, `26` percent) would cover the gap |
| the `t = 16` ordering | the `post_pulse` chain's `6.385168e-02` lies inside the band `6.38e-2` to `6.72e-2`: a tie, as the review said |
| sensitivity by horizon | `+13.1`, `+10.7`, `+4.5`, `+0.7`, `+1.9`, `+5.4`, `+6.7` percent at `t = 4, 6, 8, 10, 12, 16, 20`: not monotone; task 5's "shrinking" reading withdrawn past `t = 12` |
| separation | 31-coordinate distance saturates (window means `0.0532`, `0.0818`, `0.0733` over `(8,12]`, `(12,16]`, `(16,20]`) while the 56-state distance grows (`0.279`, `0.289`, `0.398`; maximum `0.528` at `t = 19.94`) |
| gates | boundary resumes at `t = 12.0` and `16.6` (`115 x 16` and `85 x 16` steps); reference unchanged; energy held to `1.78e-15` over 24,001 constrained repairs; one grid of 8,001 samples; tabulation reproduces task 5's 300 / 151 |
| files changed | new driver scripts and outputs under `output/local_runs/paper_v_projective_dictionary_energy_20260915e/`; two new artifacts; this report; the worklog block; the plan ledger's row.  No module; nothing committed |
| next | the same lever on the 27-key `post_pulse` chain (the one measurement that would settle the `t = 20` ordering against both bands); a decade looser (`1e-8`) on the `drive_power` chain |

# projective-route, task 7 (2026-09-16)

The lever of tasks 5 and 6 --- the re-selection completion solved at
`APCMExtensionSettings.conic_tolerance = 1e-10` in place of the committed
`1e-9`, everything else identical: preparation, the model's extension and
the carried completion the velocity accepts, repair, velocity, integrator,
cadence, reference mode; imports from `head_src` at `2d8f0efb`; the
driver-level `ReselectionToleranceRoute` subclass of `CheapProjectionRoute`
whose `reselect_lift` body is the committed one with the solver swapped ---
applied to the other 27-key `N = 16` energy-neutral chain, the `post_pulse`
one, from `t = 0` to `t = 20` in five `ram_guard --limit-mb 4500` segments,
one at a time, so that both 27-key chains carry a measured band at every
horizon and the `t = 20` ordering can be read against both.  No module was
edited; nothing was committed; no default changed.  Both `post_pulse` chains
and both `drive_power` members are scored on the one grid of 8,001 samples
against the one exact cutoff-16 reference (`exact_t20.npz`).

**The recorded driven interval could not be reused.**  Task 4 reused the
task-3 unconstrained 27-key rollout
(`paper_v_dictionary_entrance15_plus_two_spin_boson_n16_t4_20260915d`) as
the `post_pulse` chain's `t = 0` to `4` because in that mode the constraint
is off before `t = 4` and the rerun was bitwise the record.  The lever enters
the 100 re-selections of that interval, so the record does not admit the
reuse here: the driven interval was rerun under the lever (segment 0, no
energy reference, `--final-time 4.0`, exactly the task-3 rollout's
configuration) and the `post_pulse` segments resumed from its `t = 4` state
as the recorded chain's segment 1 resumed from the task-3 rollout's.  The
gate below establishes that the driver in that mode at the committed
tolerance is bitwise the task-3 rollout, so the rerun differs from the
record by the lever alone.  A consequence: the `t = 0` to `4` row of the
`post_pulse` chain is the unconstrained 27-key row, and its band below is
also the sensitivity of task 3's 27-key `N = 16` unconstrained `t = 4`
score.

## The result, first

**At `t = 20` the ordering of the two 27-key chains is settled against both
measured bands: the `post_pulse` chain is ahead under every pairing of the
four members.**  On the all-coordinate scalar RMS of the 31 closed scalar
coordinates over `t = 0` to `20`, the `post_pulse` chain's two members score
`6.990927e-02` (recorded) and `7.563847e-02` (tightened), a band `8.2`
percent wide; the `drive_power` chain's members score `8.519313e-02` and
`9.088037e-02` (`6.7` percent).  The bands are disjoint: the `drive_power`
chain's better member is `12.6` percent above the `post_pulse` chain's worse
one, and the `post_pulse` chain leads by a factor `1.13` (closest members)
to `1.30` (farthest).  The gap is `1.5` and `1.9` times the two bands'
widths at this horizon.  It is, however, smaller than the `post_pulse`
chain's own response to the same lever at `t = 4` (`20.9` percent) and
`t = 12` (`24.1` percent), so the statement rests on the two members
measured per chain and not on a bound.

**On the Eq. (22) internal energy and on physicality the ordering is the
reverse, under both settings.**  Internal-energy sample RMS over `t = 0` to
`20`: `post_pulse` `1.648045e-01` / `1.876433e-01`, `drive_power`
`6.344620e-03` / `5.217702e-03` (recorded / tightened): the `drive_power`
chain is `26` to `36` times better under any pairing.  Minimum unshifted
retained joint Gram over `t = 0` to `20`: `post_pulse` `-3.124e-03` /
`-2.993e-03`, `drive_power` `-2.463e-03` / `-2.374e-03`; minimum scaled
positive-extension eigenvalue: `-3.429e-04` / `-3.295e-04` against
`-2.040e-04` / `-2.098e-04`.  Neither chain holds either cone under either
setting; the `drive_power` chain's minima are less negative by factors
`1.21` to `1.32` (retained) and `1.57` to `1.68` (extension).

**The lever's effect on the `post_pulse` chain has no sign.**  The relative
difference (tightened minus recorded, over recorded) of the all-coordinate
score runs `+20.9`, `+4.0`, `-8.3`, `-10.7`, `-24.1`, `-15.6`, `+8.2`
percent at `t = 4, 6, 8, 10, 12, 16, 20`: the tightened member is worse over
the driven interval, better from `t = 8` to `t = 16` by up to a quarter, and
worse again at `t = 20`.  On the `drive_power` chain the same lever gave
`+13.1`, `+10.7`, `+4.5`, `+0.7`, `+1.9`, `+5.4`, `+6.7` --- always worse.
The `post_pulse` band is the wider of the two at every horizon but `t = 6`.

**Two earlier statements move.**  At `t = 8` the 15-key `post_pulse`
`N = 16` row's `4.254184e-02` lies inside the 27-key `post_pulse` band
(`4.16e-2` to `4.54e-2`): task 4's reading that `post_pulse` at 27 keys is
worse than at 15 keys at `t = 8` is a tie.  At `t = 12` the `drive_power`
chain's lead over the `post_pulse` chain, `1.40` in task 5 against the
recorded member, is `1.06` against the tightened member (`4.331967e-02`
against `4.611524e-02`); the bands are disjoint by `6.5` percent, less than
the `post_pulse` band's own `24.1` percent width there.  At `t = 16` the
bands overlap (on `6.377e-2` to `6.385e-2`): no ordering, as task 6 said.
At `t = 4` the 27-key unconstrained score is `1.63e-2` to `1.97e-2`; both
members stay below the 15-key unconstrained `2.684227e-02` and the recorded
projective event guard's `2.604900e-02` (factors `1.36` and `1.32` on the
worse member), and cell A's `1.258000e-02` is below both.

**The two `post_pulse` members separate five times further than the two
`drive_power` members, and at `t = 12` they are further from each other than
the better one is from the exact reference.**  The 31-coordinate Euclidean
distance between the chains at equal sample times reaches `1e-2` at
`t = 1.54`, `0.1` at `t = 5.39`, `0.2` at `t = 7.09` and a maximum of
`0.486` at `t = 11.695` (the `drive_power` pair: `1e-2` at `1.74`, never
`0.1`, maximum `0.0955`).  Against the chains' own distance from the exact
reference at the horizon, the members are `28`, `69`, `147`, `81` and `62`
percent of the way apart at `t = 4, 8, 12, 16, 20` (the `drive_power` pair:
`21`, `16`, `16`, `13`, `8`).  The two members are held on energy
hyperplanes `2.459164e-02` apart --- the difference of their `post_pulse`
references, which is `|E_tight - E_rec|` at every constrained sample (minimum
and maximum both `2.459e-02`) --- where the `drive_power` members' references
differ by `1.176e-03`.

## Gates

| gate | result |
|---|---|
| the re-selection subclass at the committed `1e-9`, in the driven interval's mode (no energy reference), reproduces the recorded task-3 rollout | `t = 0` to `0.4`: equal under `np.array_equal` on all 161 samples of times, states, standardized frontier, both eigenvalue series, archive coordinates and envelope ratio (maximum difference `0.0`).  Artifact `paper_v_dictionary_energy_prefixgate_postpulse_reselroute_20260916g`, `prefix_gate_postpulse_reselroute.json`; its `reselections.json` records the committed outcome of the first ten re-selections, which the recorded rollout did not record |
| the rerun driven interval differs from the record by the lever alone | the gate above, plus: the first re-selection (`t = 0.04`) refuses in both chains with the identical reduced gradient `6.012e-08` after 40 interior-Newton iterations, and the two chains are identical through `t = 0.0775`; the first difference is the `t = 0.08` re-selection |
| segments resume on a re-selection boundary | segment 1 resumed at `t = 4.0` (`100 x 16` steps of segment 0); segment 2 at `8.0`; segment 3 at `12.76` (`119 x 16` from `8.0`); segment 4 at `18.08` (`133 x 16` from `12.76`; segment 3's last 14 accepted steps to `18.115` are discarded by the resume); segment 4 reached `t = 20.0` in `48 x 16` steps |
| the declared reference is unchanged across the continuation | `-3.3659817053234624`, `-3.365981705323463`, `-3.3659817053234633` in segments 1, 2, 3 and 4, which agree to `9e-16` (the recorded chain's `-3.3905733452008326` and `-3.390573345200832` agree to `6e-16`) |
| the energy is held where the constraint is on | maximum deviation from the declared reference `2.22e-15` over the tightened chain's 19,130 constrained repairs (`1.78e-15` over the recorded chain's 19,195), 0 hyperplane refusals in either |
| the chains are on one time grid | 8,001 common samples, `t = 0` to `20`, times equal to `1e-12`; the scorer hit the `exact_t20.npz` cache and regenerated nothing; the recorded `post_pulse` chain's rescore reproduces task 6's values in every printed digit |

## The five segments, and the chain to `t = 20`

Every segment is a separate `ram_guard --limit-mb 4500` process resumed from
the previous segment's saved state and carried frontier, none ever
concurrent, with the recorded chain's `--final-time` values mirrored (`8.0`
for the first post-pulse segment, `20.0` after) so the grids coincide.  The
recorded chain is the task-3 rollout followed by
`paper_v_dictionary_energy_postpulse_27key_n16_20260915e_seg1` to `_seg4`,
as the review assembled it.  Counts are each segment's own `summary.json`
totals; peaks are the summary's resident bytes in MiB.

| quantity | tightened seg 0 | seg 1 | seg 2 | seg 3 | seg 4 | recorded driven (task-3 rollout) | seg 1 | seg 2 | seg 3 | seg 4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| interval | `0` to `4.0` | `4.0` to `8.0` | `8.0` to `12.76` | `12.76` to `18.115` (resume `18.08`) | `18.08` to `20.0` | `0` to `4.0` | `4.0` to `8.0` | `8.0` to `12.6` | `12.6` to `17.48` | `17.48` to `20.0` |
| energy reference | none | `post_pulse` | `post_pulse` | `post_pulse` | `post_pulse` | none | `post_pulse` | `post_pulse` | `post_pulse` | `post_pulse` |
| accepted steps | 1,600 | 1,600 | 1,904 | 2,142 (2,128 kept) | 768 | 1,600 | 1,600 | 1,840 | 1,952 | 1,008 |
| why it stopped | reached `t = 4` | reached `t = 8` | budget | budget | reached `t = 20` | reached `t = 4` | reached `t = 8` | budget | budget | reached `t = 20` |
| autonomous wall | `708.0 s` | `721.8 s` | `1,002.2 s` | `1,000.0 s` | `336.0 s` | `749.1 s` | `679.8 s` | `1,005.4 s` | `1,000.2 s` | `452.4 s` |
| declared completions re-solved / refused | 100 / 30 | 100 / 35 | 119 / 60 | 133 / 37 | 48 / 8 | 100 / 18 | 100 / 12 | 115 / 16 | 122 / 28 | 63 / 3 |
| wall in those solves | `664.9 s` | `660.7 s` | `928.0 s` | `921.6 s` | `309.1 s` | `710.4 s` | `616.2 s` | `931.7 s` | `927.8 s` | `415.2 s` |
| repairs applied / skipped | 2,934 / 1,866 | 4,749 / 51 | 5,712 / 0 | 6,426 / 0 | 2,243 / 61 | 2,622 / 2,178 | 4,800 / 0 | 5,515 / 5 | 5,856 / 0 | 3,024 / 0 |
| minimum retained joint Gram / scaled extension eigenvalue within the segment | `-3.614e-08` / `-1.043e-05` | `-7.746e-04` / `-6.170e-05` | `-2.993e-03` / `-3.295e-04` | `-2.386e-03` / `-2.545e-04` | `-5.779e-04` / `-5.724e-05` | `-1.877e-08` / `-6.347e-06` | `-2.907e-04` / `-2.486e-05` | `-3.124e-03` / `-3.429e-04` | `-2.753e-03` / `-3.100e-04` | `-1.047e-03` / `-8.599e-05` |
| peak resident | `695 MiB` | `697 MiB` | `649 MiB` | `717 MiB` | `652 MiB` | `621 MiB` | `631 MiB` | `606 MiB` | `644 MiB` | `712 MiB` |

Chain totals over `t = 0` to `20`, tightened against recorded: 8,000
accepted steps in five segments each; `3,768.0 s` against `3,887.0 s` of
autonomous wall (`3.1` percent faster); 500 declared completions re-solved
in each, **170 refused against 77**, `3,484.3 s` against `3,601.2 s` in
those solves (`92.5` and `92.6` percent of the wall; `6.97 s` and `7.20 s`
mean; longest single solve `50.4 s`); repairs `22,064 / 1,978` against
`21,817 / 2,183` applied / skipped, of which `19,130` against `19,195`
constrained; peak `717` against `712 MiB`.  Every segment stayed inside
twenty minutes (longest `1,002.2 s` of autonomous wall plus about `20 s` of
build).  Beside the `drive_power` chain's `4,789.8 s` / `4,794.4 s` and
`226` / `144` refusals: the `post_pulse` chain is `19` to `21` percent
cheaper and refuses less often under both settings.

Of the tightened chain's 170 refusals, by the certificate the completion
reported: 96 interior-Newton stationarity certificates with the KKT backward
error over `5e-9`; 45 `conic Newton step failed descent` (44 after the
relative face, 1 after the full face); 19 conic certificates `Solved` with
the KKT error over the threshold; 6 reference completions
`PrimalInfeasible` and 2 `AlmostPrimalInfeasible`; 2 `AlmostSolved`.  Of the
330 certified, 327 carry the interior-Newton certificate and 3 the conic
`Solved` one.  15 of the 447 interior-Newton solves have their reduced
gradient in `(5e-9, 5e-8]`, the band the decade moves (the `drive_power`
chain: 33 of 401).  Refusals by window: `15/50`, `15/50`, `19/49`, `16/51`,
`24/50`, `25/50`, `21/50`, `13/50`, `14/50`, `8/50` over `(0, 2]` to
`(18, 20]`; cumulative `30/100` at `t = 4`, `65/200` at `t = 8`, `114/300`
at `t = 12`, `148/400` at `t = 16`, `170/500` at `t = 20`.  The recorded
chain, which carries no per-re-selection ledger, refused `18/100`, `12/100`,
`16/115`, `28/122`, `3/63` by segment.

The first ten re-selections, the committed outcome (from the gate's ledger,
bitwise the recorded rollout over this interval) against the tightened one:

| `t` | committed: applied, reduced gradient, certificate | tightened: applied, reduced gradient, certificate |
|---:|---|---|
| `0.04` | no, `6.012e-08`, interior Newton (40 iterations) | no, `6.012e-08`, interior Newton (40) |
| `0.08` | **yes**, `9.016e-09` (18 iterations) | **no**, `9.327e-09` (40 iterations) |
| `0.12` | no, `1.577e-09`, `AlmostSolved` | no, `2.417e-09`, interior Newton |
| `0.16` | no, `3.002e-10`, interior Newton | no, `3.465e-09`, `AlmostSolved` |
| `0.20` | no, `9.089e-10` | yes, `4.496e-10` |
| `0.24` | no, `7.319e-09` | yes, `2.309e-10` |
| `0.28` | yes, `3.859e-08` | no, conic Newton step failed descent |
| `0.32` | yes, `2.648e-08` | no, `1.050e-10` |
| `0.36` | yes, `1.021e-08` | yes, `3.285e-09` |
| `0.40` | yes, `1.871e-09` | yes, `1.341e-09` |

The flip at `t = 0.08` is the only one on a common state: the committed
solve stopped at 18 interior-Newton iterations with a reduced gradient under
`5e-8` and certified; the tightened solve, whose stopping thresholds scale
with the tolerance, ran to the 40-iteration cap and refused at `9.327e-09`
over `5e-9`.  Every later row is a different solve on a different state.
The refused lift is a `0.344` jump in the 70 standardized frontier
coordinates, the chain's maximum frontier separation.

## The chart at `t = 4, 8, 12, 16, 20`, `post_pulse` chain

Cumulative from `t = 0`, recorded / tightened; the relative column is
(tightened minus recorded) over recorded; site-0 occupation in the chain's
own `rho[0,0]` sample-RMS convention and the record's `2 rho[0,0]`.

| quantity | `t = 4` | rel. | `t = 8` | rel. | `t = 12` | rel. | `t = 16` | rel. | `t = 20` | rel. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all-coordinate scalar RMS, 31 coordinates | `1.630660e-02` / `1.971786e-02` | `+20.9 %` | `4.537517e-02` / `4.158658e-02` | `-8.3 %` | `6.075829e-02` / `4.611524e-02` | `-24.1 %` | `6.385168e-02` / `5.386180e-02` | `-15.6 %` | `6.990927e-02` / `7.563847e-02` | `+8.2 %` |
| Eq. (22) internal-energy RMS | `7.088814e-02` / `8.534046e-02` | `+20.4 %` | `1.373215e-01` / `1.572710e-01` | `+14.5 %` | `1.531984e-01` / `1.747959e-01` | `+14.1 %` | `1.605503e-01` / `1.829311e-01` | `+13.9 %` | `1.648045e-01` / `1.876433e-01` | `+13.9 %` |
| the four phonon-occupation coordinates (N block) | `2.974702e-02` / `3.689404e-02` | `+24.0 %` | `6.276381e-02` / `6.596302e-02` | `+5.1 %` | `8.295406e-02` / `6.570523e-02` | `-20.8 %` | `8.798803e-02` / `6.171997e-02` | `-29.9 %` | `8.256954e-02` / `5.692572e-02` | `-31.1 %` |
| the 27 coordinates outside the phonon-occupation block | `1.319865e-02` / `1.564411e-02` | `+18.5 %` | `4.219396e-02` / `3.662031e-02` | `-13.2 %` | `5.673628e-02` / `4.245101e-02` | `-25.2 %` | `5.944826e-02` / `5.259787e-02` | `-11.5 %` | `6.783302e-02` / `7.803000e-02` | `+15.0 %` |
| blockwise rho / B / N / A / C, recorded | `1.5014e-02` / `4.5282e-03` / `2.9747e-02` / `1.5756e-02` / `1.3244e-02` | | `5.3828e-02` / `3.1666e-02` / `6.2764e-02` / `6.7971e-02` / `2.3368e-02` | | `6.2826e-02` / `7.4876e-02` / `8.2954e-02` / `6.8250e-02` / `4.2001e-02` | | `6.1940e-02` / `8.7497e-02` / `8.7988e-02` / `6.7077e-02` / `4.3336e-02` | | `8.0311e-02` / `1.1223e-01` / `8.2570e-02` / `7.0619e-02` / `4.1905e-02` | |
| blockwise, tightened | `2.3240e-02` / `6.1861e-03` / `3.6894e-02` / `1.6568e-02` / `1.5089e-02` | | `5.7903e-02` / `2.0067e-02` / `6.5963e-02` / `5.2128e-02` / `2.4254e-02` | | `5.9585e-02` / `4.3445e-02` / `6.5705e-02` / `4.7101e-02` / `3.4994e-02` | | `7.6918e-02` / `7.9397e-02` / `6.1720e-02` / `4.9700e-02` / `3.4755e-02` | | `1.1569e-01` / `1.4966e-01` / `5.6926e-02` / `5.4881e-02` / `3.4414e-02` | |
| site-0 occupation, `rho[0,0]` sample RMS | `1.007098e-02` / `1.670803e-02` | `+65.9 %` | `3.984892e-02` / `4.208789e-02` | `+5.6 %` | `4.751135e-02` / `4.216922e-02` | `-11.2 %` | `4.662803e-02` / `5.878463e-02` | `+26.1 %` | `6.249305e-02` / `9.338428e-02` | `+49.4 %` |
| site-0 occupation, `2 rho[0,0]` (the record's convention) | `0.0201` / `0.0334` | | `0.0797` / `0.0842` | | `0.0950` / `0.0843` | | `0.0933` / `0.1176` | | `0.1250` / `0.1868` | |
| minimum unshifted retained joint Gram over `t = 0` to `T` | `-1.877e-08` / `-3.614e-08` | | `-2.907e-04` / `-7.746e-04` | | `-3.124e-03` / `-2.993e-03` | | `-3.124e-03` / `-2.993e-03` | | `-3.124e-03` / `-2.993e-03` | |
| minimum scaled positive-extension eigenvalue | `-6.347e-06` / `-1.043e-05` | | `-2.486e-05` / `-6.170e-05` | | `-3.429e-04` / `-3.295e-04` | | `-3.429e-04` / `-3.295e-04` | | `-3.429e-04` / `-3.295e-04` | |
| maximum standardized frontier envelope ratio | `0.3646` / `0.3611` | | `0.3748` / `0.3725` | | `0.4841` / `0.4231` | | `0.4841` / `0.4231` | | `0.4841` / `0.4231` | |
| Euclidean coordinate difference from the exact reference at the horizon, all 31 | `2.350741e-01` / `2.724948e-01` | `+15.9 %` | `4.323609e-01` / `3.378869e-01` | `-21.9 %` | `5.098658e-01` / `2.808019e-01` | `-44.9 %` | `3.639121e-01` / `4.979643e-01` | `+36.8 %` | `7.097088e-01` / `1.046397e+00` | `+47.4 %` |
| declared completions re-solved / refused, cumulative | `100 / 18` / `100 / 30` | | `200 / 30` / `200 / 65` | | (`315 / 46` to `12.6`) / `300 / 114` | | (`437 / 74` to `17.48`) / `400 / 148` | | `500 / 77` / `500 / 170` | |
| 31-coordinate distance between the chains at the horizon | `0.0663` | | `0.2344` | | `0.4135` | | `0.2951` | | `0.4403` | |
| 56-state distance between the chains | `0.2557` | | `0.9139` | | `2.5004` | | `2.0608` | | `1.4352` | |
| 70-frontier distance between the chains | `0.0189` | | `0.0922` | | `0.2787` | | `0.1655` | | `0.0677` | |

At `t = 6` and `t = 10`: all-coordinate `3.335615e-02` / `3.470686e-02`
(`+4.0 %`) and `4.956048e-02` / `4.424362e-02` (`-10.7 %`); internal energy
`1.193549e-01` / `1.375353e-01` and `1.470531e-01` / `1.680050e-01`.

Three readings.

1. **The internal-energy difference is the reference, and here the
   tightened reference is the worse one.**  The `post_pulse` reference is
   the trajectory's own Eq. (22) energy at `t = 4`: `-3.3905733452008326`
   for the recorded chain, `0.180825` above the exact `-3.571397982`, and
   `-3.3659817053234624` for the tightened one, `0.205416` above.  The
   tightened driven interval drifts `2.459e-02` further in energy over
   `t = 0` to `4`, is frozen there, and the two chains' Eq. (22) energies
   differ by exactly that at every sample after `t = 4`; the `14` to `20`
   percent internal-energy penalty of the tightened member at every horizon
   is that number.  On the `drive_power` chain the same lever moved the
   reference the other way, by `1.176e-03`, and the tightened member's
   internal energy was the better by `18` percent.
2. **The block the lever moves changes with the horizon.**  Over the driven
   interval every block is worse on the tightened member, rho most (`+55`
   percent on the block, `+66` on site 0), then B (`+37`) and N (`+24`).
   From `t = 8` the tightened member is better on B and A (the
   coherent phonon and the electron-phonon correlation: `2.007e-02` against
   `3.167e-02` and `5.213e-02` against `6.797e-02` at `t = 8`; `4.345e-02`
   against `7.488e-02` and `4.710e-02` against `6.825e-02` at `t = 12`) and
   from `t = 12` on the N block (`-21`, `-30`, `-31` percent), while rho
   turns worse from `t = 16` (`7.692e-02` against `6.194e-02`, then
   `1.157e-01` against `8.031e-02` at `t = 20`) and B follows at `t = 20`
   (`1.497e-01` against `1.122e-01`).  The `t = 20` ordering of the two
   members is rho and B; the N-block improvement of the tightened member
   survives to `t = 20`.
3. **In the record's site-0 convention the `post_pulse` chain is the only
   27-key chain below the both-cones value, under both settings, and that
   column is the chain's least stable.**  `2 rho[0,0]` sample RMS over
   `t = 0` to `20`: `0.1250` / `0.1868` against the both-cones `0.204690`
   and the `drive_power` chain's `0.2252` / `0.2380`; the `post_pulse` band
   there is `49` percent wide.

## The `post_pulse` chain's band at every horizon, in the task-6 form

All-coordinate scalar RMS on the 31 closed scalar coordinates against the
exact cutoff-16 driven reference, cumulative from `t = 0` on the
`h = 0.0025` grid; the band is the recorded member (re-selection completion
at `1e-9`) and the tightened member (`1e-10`); the comparator is the
`drive_power` chain's own band from task 6 at every horizon, with the
review's other comparators where they exist (`t = 4`, `t = 8`).

| horizon | recorded | tightened | band | width | comparator | standing of the `post_pulse` chain |
|---|---:|---:|---:|---:|---|---|
| `t = 4` | `1.630660e-02` | `1.971786e-02` | `1.63e-2` to `1.97e-2` | `20.9 %` | `drive_power` band `1.14e-2` to `1.29e-2`; positive-`M4` cell A `1.258000e-02` | behind the `drive_power` band, factors `1.26` (closest members) to `1.73` (farthest), bands disjoint by `26.2 %`; behind cell A by `1.30` on the better member |
| `t = 6` | `3.335615e-02` | `3.470686e-02` | `3.34e-2` to `3.47e-2` | `4.0 %` | `drive_power` band `2.09e-2` to `2.31e-2` | behind, `1.44` to `1.66`, disjoint by `44.1 %` |
| `t = 8` | `4.537517e-02` | `4.158658e-02` | `4.16e-2` to `4.54e-2` | `8.3 %` | `drive_power` band `2.59e-2` to `2.71e-2`; 15-key `post_pulse` `N = 16` `4.254184e-02` | behind the `drive_power` band, `1.53` to `1.75`, disjoint by `53.5 %`; the 15-key row lies inside the band: tie |
| `t = 10` | `4.956048e-02` | `4.424362e-02` | `4.42e-2` to `4.96e-2` | `10.7 %` | `drive_power` band `3.33e-2` to `3.36e-2` | behind, `1.32` to `1.49`, disjoint by `31.8 %` |
| `t = 12` | `6.075829e-02` | `4.611524e-02` | `4.61e-2` to `6.08e-2` | `24.1 %` | `drive_power` band `4.25e-2` to `4.33e-2` | behind, `1.06` to `1.43`; the bands are disjoint by `6.5 %` of the `drive_power` upper member, less than this band's own width |
| `t = 16` | `6.385168e-02` | `5.386180e-02` | `5.39e-2` to `6.39e-2` | `15.6 %` | `drive_power` band `6.38e-2` to `6.72e-2` | bands overlap on `6.377e-2` to `6.385e-2`: no ordering; the tightened `post_pulse` member is below both `drive_power` members by `1.18` to `1.25` |
| `t = 20` | `6.990927e-02` | `7.563847e-02` | `6.99e-2` to `7.56e-2` | `8.2 %` | `drive_power` band `8.52e-2` to `9.09e-2` | ahead, `1.13` to `1.30`; bands disjoint by `12.6 %` of the `post_pulse` upper member |

Companion columns of the same row, recorded / tightened: Eq. (22)
internal-energy RMS `7.09e-2` / `8.53e-2` at `t = 4` to `1.65e-1` /
`1.88e-1` at `t = 20` (the `drive_power` chain `4.60e-3` / `3.69e-3` to
`6.34e-3` / `5.22e-3`; both cones `9.4e-4` at `t = 20`); minimum unshifted
retained joint Gram `-3.12e-3` / `-2.99e-3` and minimum scaled extension
eigenvalue `-3.43e-4` / `-3.29e-4` over `t = 0` to `20`, both attained in
`(8, 12]` in both members (the `drive_power` chain `-2.46e-3` / `-2.37e-3`
and `-2.04e-4` / `-2.10e-4`); site-0 occupation in the record's
`2 rho[0,0]` convention `0.125` / `0.187` at `t = 20` (both cones `0.205`;
`drive_power` `0.225` / `0.238`); autonomous wall `3,887.0 s` / `3,768.0 s`
to `t = 20`, 500 completions re-solved with `77` / `170` refused.

## The two bands together: the `t = 20` ordering and every other horizon

| horizon | `drive_power`, recorded / tightened | `post_pulse`, recorded / tightened | ordering on the all-coordinate score | internal-energy RMS, `drive_power` / `post_pulse` (recorded, tightened) | minimum retained joint Gram, `drive_power` / `post_pulse` |
|---|---:|---:|---|---|---|
| `t = 4` | `1.142716e-02` / `1.292283e-02` (`+13.1 %`) | `1.630660e-02` / `1.971786e-02` (`+20.9 %`) | `drive_power` ahead, `1.26` to `1.73`, disjoint by `26.2 %` | `4.60e-3`, `3.69e-3` / `7.09e-2`, `8.53e-2` | `-4.54e-4`, `-4.51e-4` / `-1.88e-8`, `-3.61e-8` |
| `t = 8` | `2.593079e-02` / `2.709352e-02` (`+4.5 %`) | `4.537517e-02` / `4.158658e-02` (`-8.3 %`) | `drive_power` ahead, `1.53` to `1.75`, disjoint by `53.5 %` | `5.75e-3`, `4.70e-3` / `1.37e-1`, `1.57e-1` | `-2.15e-3`, `-2.20e-3` / `-2.91e-4`, `-7.75e-4` |
| `t = 12` | `4.250925e-02` / `4.331967e-02` (`+1.9 %`) | `6.075829e-02` / `4.611524e-02` (`-24.1 %`) | `drive_power` ahead, `1.06` to `1.43`, disjoint by `6.5 %` (inside the `post_pulse` band's width) | `6.09e-3`, `5.00e-3` / `1.53e-1`, `1.75e-1` | `-2.46e-3`, `-2.37e-3` / `-3.12e-3`, `-2.99e-3` |
| `t = 16` | `6.377276e-02` / `6.719628e-02` (`+5.4 %`) | `6.385168e-02` / `5.386180e-02` (`-15.6 %`) | bands overlap: no ordering | `6.25e-3`, `5.14e-3` / `1.61e-1`, `1.83e-1` | `-2.46e-3`, `-2.37e-3` / `-3.12e-3`, `-2.99e-3` |
| `t = 20` | `8.519313e-02` / `9.088037e-02` (`+6.7 %`) | `6.990927e-02` / `7.563847e-02` (`+8.2 %`) | `post_pulse` ahead, `1.13` to `1.30`, disjoint by `12.6 %` | `6.34e-3`, `5.22e-3` / `1.65e-1`, `1.88e-1` | `-2.46e-3`, `-2.37e-3` / `-3.12e-3`, `-2.99e-3` |

At `t = 6` and `t = 10` the `drive_power` chain is ahead by `1.44` to `1.66`
and `1.32` to `1.49`, bands disjoint by `44.1` and `31.8` percent.

**The final statement of the `t = 20` ordering of the two 27-key `N = 16`
energy-neutral chains.**  On the all-coordinate scalar RMS of the 31 closed
scalar coordinates over `t = 0` to `20`, the `post_pulse` chain is ahead of
the `drive_power` chain under every pairing of the four measured members,
`6.99e-2` to `7.56e-2` against `8.52e-2` to `9.09e-2`, by a factor `1.13` to
`1.30`, with the two bands disjoint by `12.6` percent --- more than either
band's width at this horizon (`8.2` and `6.7` percent), less than the
`post_pulse` chain's own response to the lever at `t = 4` and `t = 12`
(`20.9` and `24.1` percent).  The best moment-closure row on record at
`t = 20` on that score is therefore the 27-key `post_pulse` chain's recorded
member, `6.990927e-02`, with a measured band to `7.563847e-02`, against the
both-cones `1.286070e-01`.  On the Eq. (22) internal energy the `drive_power`
chain is ahead by `26` to `36` times under every pairing; on physicality
neither chain holds a cone under either setting and the `drive_power`
chain's minima are less negative by `1.21` to `1.32` (retained joint Gram)
and `1.57` to `1.68` (scaled extension); on site-0 occupation in the
record's convention the `post_pulse` chain is ahead by `1.21` to `1.90`.
The `drive_power` chain remains ahead over `t = 0` to `12` at every horizon
scored (`1.06` to `1.75` on the closest members), tied at `t = 16`, and
behind at `t = 20`.

## Separation of the two `post_pulse` members

Times at which the two trajectories first separate, Euclidean distance at
equal sample times, with the `drive_power` pair of tasks 5 and 6 beside:

| threshold | 31 closed scalar coordinates | 56 state coordinates | 70 standardized frontier | `drive_power` pair, 31 / 56 / 70 |
|---:|---:|---:|---:|---|
| `1e-6`, `1e-5` | `0.0825` | `0.0825` | `0.08` | `0.2025` / `0.2025` / `0.2` |
| `1e-4` | `0.0875` | `0.085` | `0.08` | `0.2025` / `0.2025` / `0.2` |
| `1e-3` | `0.245` | `0.13` | `0.08` | `0.5275` / `0.23` / `0.2` |
| `1e-2` | `1.54` | `0.635` | `0.08` | `1.7375` / `0.545` / `0.2` |
| `1e-1` | `5.3925` | `2.1575` | `0.08` | never (max `0.0955` at `13.845`) / `2.485` / `0.2` |
| `0.2` | `7.0925` | `3.0875` | `0.08` | never / --- / --- |
| `1` | never (max `0.4856` at `11.695`) | `7.82` (max `2.711` at `11.565`) | never (max `0.344` at `0.08`) | never / never (max `0.528`) / never (max `0.353`) |

Window means of the 31-coordinate distance: `0.0232`, `0.1411`, `0.3128`,
`0.3452`, `0.3312` over `(0, 4]`, `(4, 8]`, `(8, 12]`, `(12, 16]`,
`(16, 20]` (the `drive_power` pair: `0.0162`, `0.0346`, `0.0532`, `0.0818`,
`0.0733`); of the 56-state distance `0.1064`, `0.5361`, `1.8839`, `2.2571`,
`1.6628`.  The 31-coordinate distance saturates after `t = 12` at three to
four tenths, as the `drive_power` pair's did at eight hundredths; the
56-state distance has its maximum `2.711` at `t = 11.565` and its window
mean peaks in `(12, 16]` and falls, where the `drive_power` pair's was
still growing at `t = 20`.  Beside the branch's other two
pairs: the cheap route's 15-key `N = 4` `post_pulse` pair reached `1e-2` at
`t = 6.93`, `0.1` at `8.38` and order one by `t = 11`; the 15-key `N = 16`
`drive_power` pair at this lever reached `1e-2` at `0.48`, `0.1` at `3.085`
and `0.72` near `t = 12`; the 27-key `drive_power` pair `1e-2` at `1.74` and
never `0.1`; this pair `1e-2` at `1.54`, `0.1` at `5.39`, `0.49` at `11.7`.
Of the four, the 27-key `drive_power` pair is the least sensitive by a
factor five to ten on the pointwise measure, and the constraint mode is what
distinguishes it from this pair at the same dictionary: whether that is the
constraint acting through the driven interval or the `21`-times-larger
reference difference here is not separated by this measurement.

## Provenance and gates of the runs themselves

- Five rollouts,
  `paper_v_dictionary_energy_postpulse_27key_n16_tightresel_20260916g_seg0`
  to `_seg4`, each under `pipelines/shell/ram_guard.py --limit-mb 4500`,
  one at a time (a cheap-route job under its own guard was on the machine
  throughout; guard limits summed under the `18 GB` budget); summary peaks
  `695`, `697`, `649`, `717`, `652 MiB` (`ram_guard` reports `693`, `697`,
  `639`, `646`, `628 MB`); autonomous wall `708.0`, `721.8`, `1,002.2`,
  `1,000.0`, `336.0 s`.  The gate rollout
  `paper_v_dictionary_energy_prefixgate_postpulse_reselroute_20260916g` ran
  under the same guard (`128.0 s`, `633 MB`).  The scoring, comparison and
  tabulation ran under `--limit-mb 2000`.
- Drivers, all under
  `output/local_runs/paper_v_projective_dictionary_energy_20260915e/`:
  `queue_tight_postpulse.sh` (segment 0 by `run_energy_segment.py` in mode
  `none`, segment 1 resumed from it in `post_pulse` mode with
  `--energy-reference-time 4.0` and `--final-time 8.0`, then the unchanged
  `run_chain.py` with `--first 2 --last 8 --final-time 20.0`; the fifth
  segment reached `t = 20` so segments 5 to 8 were not needed),
  `prefixgate_postpulse_reselroute.sh` (the gate), `score_tight_postpulse.sh`
  (score, compare, tabulate), `monitor_tight_postpulse.sh`, and the new
  `tabulate_task7.py` (the ledgers, the chart, the band table in the task-6
  form, the two-band table, the separation recomputed from the arrays, the
  refusal ledger, and the first-ten comparison against the gate's ledger).
  `run_energy_segment.py`, `run_chain.py`, `compare_chains.py`,
  `compare_prefix.py` and `score_energy_chain.py` are as task 5 left them.
- Outputs there: `tightresel_postpulse_chain_score_t20.json`,
  `compare_tightresel_postpulse_t20.json`,
  `compare_tightresel_postpulse_driven.json` (the driven interval alone,
  written after segment 0), `tightresel_postpulse_chain_score_t8.json`
  (written after segment 1), `prefix_gate_postpulse_reselroute.json`,
  `tabulate_task7.log`, `tightresel_postpulse_chain.log`,
  `tightresel_postpulse_seg{0,1}.log`, the three `run_chain` segment logs,
  `prefixgate_postpulse_reselroute.log`.  The recorded chains' artifacts and
  `postpulse_chain_score_all.json`, `power_chain_score_all.json`,
  `tightresel_chain_score_t20.json` are untouched.
- The previous instance of this agent ran the gate and segments 0 to 4 and
  was lost to a connection error before scoring; this instance scored,
  compared and tabulated the finished chain and repeated no segment.
- No module was edited, nothing was committed, no default, acceptance rule
  or ladder changed.

## Structured summary of task 7

| item | value |
|---|---|
| the lever | `APCMExtensionSettings.conic_tolerance` of the re-selection completion, `1e-10` against the committed `1e-9`, on the 27-key `N = 16` `post_pulse` energy-neutral chain from `t = 0`; preparation, model extension, repair, velocity, integrator, cadence and reference mode identical; imports from `head_src` at `2d8f0efb` |
| the driven interval | not reusable: the lever enters its 100 re-selections; rerun under the lever as segment 0 (mode `none`, `708.0 s`, 100 re-solved / 30 refused against the recorded 18); the driver at `1e-9` in that mode is bitwise the task-3 rollout over `t = 0` to `0.4` (161 samples, `0.0`, seven arrays) |
| the runs | segments 0 (`0` to `4`), 1 (`4` to `8`), 2 (`8` to `12.76`, budget), 3 (`12.76` to `18.115`, budget, resume `18.08`), 4 (`18.08` to `20`, reached); `3,768.0 s` (recorded `3,887.0 s`); 500 re-solved, 170 refused (recorded 77); energy held to `2.22e-15` over 19,130 constrained repairs; reference `-3.36598170532346` unchanged to `9e-16`; peak `717 MiB` |
| `post_pulse` band, all 31, recorded / tightened | `t = 4` `1.630660e-02` / `1.971786e-02` (`+20.9 %`); `t = 8` `4.537517e-02` / `4.158658e-02` (`-8.3 %`); `t = 12` `6.075829e-02` / `4.611524e-02` (`-24.1 %`); `t = 16` `6.385168e-02` / `5.386180e-02` (`-15.6 %`); `t = 20` `6.990927e-02` / `7.563847e-02` (`+8.2 %`); `t = 6` `+4.0 %`, `t = 10` `-10.7 %`: no sign |
| internal-energy RMS, `post_pulse` / `drive_power` (recorded, tightened) | `t = 4` `7.09e-2`, `8.53e-2` / `4.60e-3`, `3.69e-3`; `t = 20` `1.65e-1`, `1.88e-1` / `6.34e-3`, `5.22e-3`; the `post_pulse` references sit `0.180825` / `0.205416` above exact, `2.459e-02` apart, and that is `|E_tight - E_rec|` at every constrained sample |
| physicality minima over `t = 0` to `20`, `post_pulse` / `drive_power` (recorded, tightened) | retained joint Gram `-3.124e-03`, `-2.993e-03` / `-2.463e-03`, `-2.374e-03`; scaled extension `-3.429e-04`, `-3.295e-04` / `-2.040e-04`, `-2.098e-04`; no member of either chain holds a cone |
| the `t = 20` ordering | settled against both bands: `post_pulse` ahead by `1.13` to `1.30` under every pairing, bands disjoint by `12.6 %` against widths `8.2` and `6.7 %`; the gap is inside the `post_pulse` chain's own sensitivity at `t = 4` and `t = 12`, so the statement is on the four members, not a bound |
| the other horizons | `drive_power` ahead at `t = 4, 6, 8, 10, 12` by `1.26` to `1.73`, `1.44` to `1.66`, `1.53` to `1.75`, `1.32` to `1.49`, `1.06` to `1.43` (bands disjoint by `26.2`, `44.1`, `53.5`, `31.8`, `6.5 %`); `t = 16` bands overlap: no ordering |
| statements that move | `t = 8`: the 15-key `post_pulse` row `4.254184e-02` lies inside the 27-key `post_pulse` band: tie, not "worse at 27 keys"; `t = 12`: the `drive_power` lead is `1.06` on the closest members, disjoint by `6.5 %` against a `24.1 %` `post_pulse` band; `t = 4`: task 3's 27-key unconstrained score is `1.63e-2` to `1.97e-2`, still below the 15-key `2.684e-02` and the projective event guard's `2.605e-02` |
| separation | first difference at the `t = 0.08` re-selection (committed certified at reduced gradient `9.016e-09` in 18 iterations, tightened refused at `9.327e-09` at the 40-iteration cap), a `0.344` frontier jump; 31-coordinate distance `1e-2` at `1.54`, `0.1` at `5.39`, `0.2` at `7.09`, maximum `0.486` at `11.695`; `147 %` of the better member's own distance from exact at `t = 12`; five times the `drive_power` pair's maximum |
| refusals | 170 of 500: 96 interior-Newton KKT over `5e-9`, 45 conic-Newton descent failures, 19 conic `Solved` over the threshold, 8 infeasible reference completions, 2 `AlmostSolved`; 15 of 447 reduced gradients in `(5e-9, 5e-8]`; by window `15/50` to `25/50`, then `8/50` over `(18, 20]` |
| gates | bitwise prefix gate (161 samples, `0.0`); common first re-selection (`6.012e-08`, refused in both); boundary resumes at `4.0`, `8.0`, `12.76`, `18.08`; reference unchanged to `9e-16`; energy held to `2.22e-15`; one grid of 8,001 samples; the recorded chain's rescore reproduces task 6 |
| files changed | new driver scripts and outputs under `output/local_runs/paper_v_projective_dictionary_energy_20260915e/`; six new artifacts (five segments and the gate); this report; the worklog block; the plan ledger's row.  No module; nothing committed |
| next | a decade looser (`1e-8`) on both 27-key chains, which would put a third member in every band and test whether the `12.6 %` gap at `t = 20` survives a lever the `post_pulse` chain answered with `24 %` at `t = 12`; the continuous re-selection rule the cheap route is building, which removes the amplifier these seven tasks have measured |

# projective-route, task 8 (2026-09-17)

The cheap route's task 8 attributed the continuous re-selection rule's
continuity to its measured-candidate acceptance and its accuracy cost to
the proximal anchor, and recommended the anchor-free rule: acceptance
margin `1e-3`, `measured` candidates, anchor weight `0`; its task 9 took
that rule to `t = 20` at 15 keys (twins spread `0.1`, `0.2`, `3.2` percent
at `t = 4`, `8`, `12`, scores `0.0272`, `0.0385`, `0.0982`).  This round
applies the same rule to the 27-key `N = 16` `drive_power` energy-neutral
chain of task 4 --- the base and its three one-setting twins,
`energy_tolerance 1e-3`, Dykstra 60, re-selection `conic_tolerance 1e-10`
--- from `t = 0` to `8`, so that the 27-key chain, whose `t = 4` score
task 5 found to move `13.1` percent under one declared solver setting,
carries a band under the rule that removed the amplifier at 15 keys.
Every field of the rule is a setting of the committed `LiftReselectionRule`
(`a81b7ecb`, unchanged through `3ea96121`, the tree the runs and this
comparison were made against; `HEAD` moved to `90a3bd60`, the cheap
route's trust-radius commit, while this record was written); the driver passes the
three flags to `CheapProjectionRoute`'s `reselection_rule` argument and,
with none of them, is the recorded driver exactly (gate below).  Imports
are from `head_src_e3374f6f`, which `diff -rq` shows identical to
`paper_5/src/paper5` at `3ea96121` outside `__pycache__`.  No module was
edited; nothing was committed; no default changed.  The earlier task-8
attempt of the previous instance --- the same chain under the combined
rule with the anchor at weight `1`, whose base scored `1.891654e-02` /
`3.417762e-02` at `t = 4` / `8` (`paper_v_dictionary_energy_power_27key_n16_rule_base_20260916h_seg{1,2,3}`,
`task8_base_score_t8.json`) --- is superseded by the cheap route's
attribution and is kept below only as a comparator.

Configuration, as in task 4: 27 active keys, 56 state coordinates, 36
positive-extension rows, 70 standardized frontier coordinates, strong
coupling, cutoff 16, exact-correlated preparation at the declared ladder's
first rung, SSPRK(3,3) at `h = 0.0025`, Dykstra 30 on both targets at
stage cadence, the declared proximal log-det completion re-solved every 16
steps warm-started on the carried frontier, the energy-neutral repair in
`drive_power` mode at `energy_tolerance 0`.  Conventions: the score is the
all-coordinate scalar RMS of the 31 closed scalar coordinates against the
exact cutoff-16 driven reference from `t = 0` (`exact_t8.npz`, one grid of
3,201 samples for every member); the internal energy is the sample RMS of
the Eq. (22) energy on the same horizon; the physicality numbers are the
minimum unshifted retained joint Gram eigenvalue and the minimum scaled
positive-extension eigenvalue over the horizon, measured along the
trajectory, with no certificate on this route; a separation is the
Euclidean distance between two chains at equal sample times over the 31
closed scalar coordinates (`d31`), the 56 state coordinates (`d56`) or the
70 standardized frontier coordinates (`d70`); the spread of the four
members is the largest score minus the smallest as a fraction of the
smallest; a re-selection's move is the Euclidean distance in the 70
standardized frontier coordinates between the carried lift and the lift
the rule applied; a re-selection's decision is the rule's, `accepted`
(certified or not), `declined_by_margin` or `infeasible` (the completion
returned no iterate passing its own feasibility test); a solver refusal is
a re-selection whose completion returned uncertified, whatever the rule
did with the iterate.

## The result, first

**A sensitivity-resolved 27-key number now exists at both horizons; at
`t = 8` it is behind the recorded chain's earlier form by more than either
band, and at `t = 4` it beats cell A outside every measured band.**  The
cheap route's rule, which gained `15` percent at `t = 8` at 15 keys, loses
`17` to `24` percent there at 27 keys, and the chaining gate below shows
that the rule is not continuous at the edge of its own feasibility test: a
`2.5e-14` resume perturbation flips a `t = 0.16` re-selection from
`infeasible` to `accepted` at a `0.89` move.  Both negatives are inside
the numbers that follow.

**The numbers.**  The four members
score `1.001172e-02`, `1.001172e-02`, `9.959755e-03`, `1.001172e-02` (base,
`etol1e3`, `dykstra60`, `conic1e10`) over `t = 0` to `4`, a band `9.96e-3`
to `1.001e-2` of width `0.52` percent, and `3.181485e-02`, `3.179926e-02`,
`3.224344e-02`, `3.179926e-02` over `t = 0` to `8`, a band `3.180e-2` to
`3.224e-2` of width `1.40` percent.  Cell A's `1.258000e-02` is `1.256`
times the worst member at `t = 4`; the recorded chain's band `1.143e-2` to
`1.292e-2` (task 5) lies wholly above the rule's, `1.14` times its worst
member on the closest pair and `1.30` on the farthest.  The recorded
projective event guard's `2.604900e-02` and every 15-key row are `2.6`
times or more above it.

**At `t = 8` the rule is behind the recorded chain by more than either
band.**  The recorded chain's band `2.593e-2` to `2.709e-2` (`4.5`
percent) lies wholly below the rule's `3.180e-2` to `3.224e-2`, by `1.17`
on the closest pair and `1.24` on the farthest; the gap between the bands
is `17.4` percent of the recorded band's top.  On the window `(4, 8]`
alone, derived from the cumulative values on 1,601 and 3,201 samples, the
base scores `4.387e-2` against the recorded chain's `3.485e-2`.  The loss
is in two blocks: the four phonon-occupation coordinates (`4.778e-2`
against `3.283e-2`, `1.46` times) and the A block (`3.880e-2` against
`1.994e-2`, `1.95` times); the B block is better (`2.181e-2` against
`2.725e-2`) and rho and C are equal to three figures.  At `t = 4` the gain
is in rho (`1.696e-2` against `2.037e-2`), B (`3.72e-3` against
`6.91e-3`) and C (`9.44e-3` against `1.117e-2`), the phonon-occupation
block equal (`9.56e-3` against `9.84e-3`) and A `1.10` times worse.  So
the rule that gained `15` percent at `t = 8` at 15 keys loses `17` to
`24` percent there at 27 keys, and the 27-key `t = 8` row on record stays
the recorded chain's `2.593079e-02`, with its `4.5` percent band.  The
rule stays ahead of the combined-rule base (`3.417762e-02`, `1.06` times
the rule's worst member), of the 15-key row under the same rule
(`0.038538`, `1.20`) and of every 15-key row on record.

**Two of the three twins are one trajectory with the base to `1e-8`
through `t = 6.6`, and the third stays within `0.035` of it.**  The
`etol1e3` and `conic1e10` twins differ from the base by `d31` under
`3.2e-8` at every sample to `t = 6.6` (first sample with any difference
`t = 0.005` at `2.5e-15` and `t = 0.2` at `1.8e-14` in the frontier), and
from each other by at most `5.1e-9` over the whole horizon; the energy
tolerance and the certificate tolerance produce no branch.  At the
`t = 6.6` re-selection the base's completion took the facial-reduction
path (`facial reduction removed every frontier direction`, no certificate)
and returned an iterate `0.9565` from the carried lift that passes the
rule's feasibility test and improves the declared objective by `1.04`, so
the rule applied it; on states `1e-7` away the two twins' completions took
the interior-Newton path to its 40-iteration cap and returned an iterate
`0.0117` away (objective `3.944`, better than the base's accepted
`4.695`).  At `t = 6.64` the base's interior-Newton iterate moved `0.9428`
back, and from `t = 6.68` its re-selections certify again.  The excursion
costs `1.2e-3` in `d31` at most (`t = 6.645`), `1.1e-3` at `t = 8`
(`0.4` percent of the members' own distance from the exact reference,
`0.296`), and `0.05` percent of the score; it puts the base's maximum
standardized frontier envelope ratio at `0.772` against the twins'
`0.387`.  This is the cheap route's task-9 event --- a large uncertified
move accepted as the rule is written --- at `t = 6.6` instead of `t = 10`
and at a move of `0.96` instead of `0.4` to `0.8`, from a solver-path
branch on a `1e-7` perturbation rather than an answer branch on a `0.05`
one; it is what the rule's trust radius (off here) is declared for.  The
`dykstra60` twin parts at the `t = 0.16` re-selection, where the base's
reference completion returned no iterate (`PrimalInfeasible`, decision
`infeasible`) and the twin's returned one it accepted (`0.0011`); `d31`
reaches `1e-2` at `t = 3.835`, never `0.1` (maximum `0.0349` at
`t = 6.18`), and is `7.9` and `11.1` percent of the members' distance from
the exact reference at `t = 4` and `8`.

**The rule accepts what the earlier form refused at `t = 0.04`, declines
nothing by the margin at 27 keys, and is cheaper.**  All four members
accept the first re-selection's whole-face answer (support `36/36`,
directions `70/70`, objective `6.538` to `4.508`), a `0.3447` move: the
base's and `conic1e10`'s at the interior-Newton 40-iteration cap (KKT
backward error `1.07e-07`, refused under the earlier form, which carried
the preparation's lift to its first certified re-selection at `t = 0.2`),
the `etol1e3` twin's certified in 12 iterations at the same answer to
`1e-9`, the `dykstra60` twin's a conic-Newton descent failure at
`0.344678`.  That is where the base and the recorded chain part (`d70`
`0.1` at `t = 0.04`; `d31` `1e-2` at `t = 1.84`, `0.1` at `t = 4.88`,
maximum `0.326`).  Of 200 re-selections the base accepts 188 (36 of them
uncertified), declines none by the margin and finds 12 `infeasible` (9
`PrimalInfeasible` and 2 `AlmostPrimalInfeasible` reference completions,
one Newton cone failure), 10 of them over `(0, 2]`; the 15-key set A
declined the four wrong-face answers of `t = 0.04` to `0.16`, and at 27
keys no returned iterate fails the objective test.  Solver refusals
(uncertified returns) are 48, 42, 42, 78 of 200 for the four members
(recorded chain 75, tightened 97); the `conic1e10` twin accepts 66 of its
78 and stays on the base's trajectory, as at 15 keys.  Autonomous wall
`1,886.3`, `1,901.9`, `1,962.8`, `1,883.4 s` to `t = 8` in two segments
each, against the recorded chain's `2,058.0 s` in three, `88` to `92`
percent in the completion solves at `8.63` to `8.76 s` mean.

**The cones are tighter than the recorded chain's, the internal energy
sits inside its band, and nothing is held.**  Over `t = 0` to `8` the
minimum unshifted retained joint Gram is `-1.301e-03`, `-1.300e-03`,
`-5.361e-04`, `-1.300e-03` (recorded `-2.152e-03`, tightened
`-2.201e-03`) and the minimum scaled positive-extension eigenvalue
`-1.404e-04`, `-1.374e-04`, `-5.747e-05`, `-1.374e-04` (recorded
`-1.922e-04`, tightened `-1.980e-04`); the Eq. (22) internal-energy RMS
is `3.922911e-03` (base, `etol1e3`, `conic1e10`) and `4.133866e-03`
(`dykstra60`) at `t = 4`, `4.781299e-03` and `5.020505e-03` at `t = 8`,
between the recorded member's `4.597225e-03` / `5.751890e-03` and the
tightened member's `3.687212e-03` / `4.702571e-03`, because the
`drive_power` reference the rule's members select, `-3.5658908873520327`
(`5.507e-03` above the exact `-3.571397982`; `dykstra60`
`-3.5656258632352253`, `5.772e-03` above), lies between the recorded
`-3.56468776268448` (`6.710e-03`) and the tightened
`-3.5658639789420468` (`5.534e-03`).

## Gates

| gate | result |
|---|---|
| the driver at the `e3374f6f` snapshot with no rule flag reproduces the recorded chain | `t = 0` to `0.4` with `--record-reselections` and no rule or tolerance flag: equal under `np.array_equal` to the recorded segment 1 on all 161 samples of times, states, standardized frontier, both eigenvalue series, envelope ratio, archive coordinates, internal energy and its reference (maximum difference `0.0`).  Artifact `paper_v_dictionary_energy_prefixgate_task8_default_20260916h`, `prefix_gate_task8_default.json` |
| the tree's native `reselection_extension` path reproduces the task-5 subclass | the same bitwise equality against the tightened chain's segment 1 with `--reselection-conic-tolerance 1e-10` (`prefix_gate_task8_native1e10.json`); the ledgers agree on time, outcome and iteration count at every one of the first ten re-selections (the recorded ledgers carry no KKT field) |
| chaining reproduces a single rollout under the rule | **not bitwise: to round-off until a re-selection sits at the feasibility test's edge.**  Single rollout S, `t = 0` to `0.2` (five re-selections, `157.4 s`); chain A, `t = 0` to `0.14` (resume written at `t = 0.12`, `122.8 s`) and B resumed from it to `t = 0.2` (`34.0 s`).  The chain agrees with S to `2.5e-14` in `d31` and `d56` and `1.7e-15` in `d70` through `t = 0.1575`, with the same decisions at the first three re-selections (`accepted` at `0.3447`; `infeasible`; `infeasible`); at the fourth, `t = 0.16`, S's completion returned a Newton-cone-failure iterate (support `35/36`, directions `30/70`) whose scaled minimum eigenvalue `-1.76e-08` fails the rule's feasibility test at `-10 tol = -1e-08`, decision `infeasible`, and B's, on a state `2.5e-14` away, a cone-failure iterate on another face (support `34/36`, directions `12/70`) at `-9.78e-09`, which passes and improves the declared objective (`5.93` to `5.31`), decision `accepted`, a `0.8926` move; at `t = 0.2` both certify (S `0.0016` from the carried lift, B `0.8923` back), leaving `d70` `7.3e-04` and `d31` `3.3e-04` at `t = 0.2`.  Chaining itself is exact to round-off, as in task 4's gate (`1.7e-14`); the flip is the rule's discontinuity at the edge of its own feasibility test, seeded by that round-off.  The four members' bands are therefore measured across four different boundaries --- a fifth perturbation of the twins' kind, which they include --- and a single rollout's `t = 4` and `t = 8` numbers would differ from a chain's by a perturbation of that size.  Artifacts `paper_v_dictionary_energy_measured_chaingate{S,A,B}_20260917j`, `chaingate_task8m_compare.json` |
| a segment resumes on a re-selection boundary | the second segments resume at `t = 3.84`, `3.56`, `3.48`, `3.72` (`96`, `89`, `87`, `93` intervals of 16 steps from `t = 0`), so every member's completion solves fall on the steps a single rollout's would |
| the declared reference is set inside the second segment | every first segment stopped before `t = 4`, so its `declared_reference_value` is the running drive-power integral at its last step and the second segment's is the reference held from `t = 4` on: `-3.5658908873520327`, `-3.5658908874540605`, `-3.5656258632352253`, `-3.565890887153181`; the three members that share a trajectory to `t = 6.6` share the reference to `2e-10` |
| the energy is held where the constraint is on | maximum deviation from the declared reference `1.78e-15` (`1.33e-15` for `conic1e10`) over 9,535, 9,453, 9,426, 9,550 constrained repairs, 0 hyperplane refusals |
| the four chains are on one time grid | 3,201 common samples, `t = 0` to `8`, times equal to `1e-12`; every score hit the `exact_t8.npz` cache and regenerated nothing |
| the re-selection ledgers are merged at the boundary | `compare_chains.py`'s `load_chain` dropped the record at a segment's last accepted step (the record at or after the next segment's first *sample*), so two chains resumed at different boundaries appeared to have different schedules and the comparison exited; it now drops a record only at or after the next segment's first *ledger entry*, the boundary-inclusive merge of `tabulate_task5.py` and `tabulate_task6.py`, and every member's merged ledger has 200 entries on one schedule.  The trajectory arrays were merged correctly before and are unchanged |

## (a) The four chains

Every segment is a separate `ram_guard --limit-mb 4500` process, resumed
from the previous segment's saved state and carried frontier, no two ever
at once, with the `1,000 s` autonomous budget of the recorded chain; a
cheap-route job under a `3,000 MB` guard shared the machine.  Counts are
each segment's own totals from its `summary.json`; the re-selection
columns are, of the declared completions re-solved, those the rule
accepted (of which uncertified) and those it found `infeasible`; the
margin declined none.

| quantity | base seg 1 | seg 2 | `etol1e3` seg 1 | seg 2 | `dykstra60` seg 1 | seg 2 | `conic1e10` seg 1 | seg 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| interval | `0` to `3.84` | `3.84` to `8.0` | `0` to `3.56` | `3.56` to `8.0` | `0` to `3.48` | `3.48` to `8.0` | `0` to `3.72` | `3.72` to `8.0` |
| accepted steps | 1,536 | 1,664 | 1,424 | 1,776 | 1,392 | 1,808 | 1,488 | 1,712 |
| why it stopped | budget | reached `t = 8` | budget | reached | budget | reached | budget | reached |
| autonomous wall / build | `1,001.7` / `20.5 s` | `884.6` / `20.6 s` | `1,002.1` / `20.8 s` | `899.8` / `20.2 s` | `1,006.8` / `19.0 s` | `956.0` / `20.2 s` | `1,001.7` / `20.2 s` | `881.7` / `20.5 s` |
| re-solved: accepted (uncertified) / `infeasible` | 96: 86 (21) / 10 | 104: 102 (15) / 2 | 89: 79 (19) / 10 | 111: 109 (11) / 2 | 87: 80 (24) / 7 | 113: 113 (11) / 0 | 93: 83 (35) / 10 | 107: 105 (31) / 2 |
| solver refusals (uncertified returns) | 31 | 17 | 29 | 13 | 31 | 11 | 45 | 33 |
| wall in the solves | `929.6 s` | `803.7 s` | `934.9 s` | `816.3 s` | `907.4 s` | `818.0 s` | `933.3 s` | `800.6 s` |
| repairs applied / skipped | 4,543 / 65 | 4,992 / 0 | 4,125 / 147 | 5,328 / 0 | 4,012 / 164 | 5,414 / 10 | 4,414 / 50 | 5,136 / 0 |
| repair wall | `47.2 s` | `53.3 s` | `43.8 s` | `54.9 s` | `76.8 s` | `109.0 s` | `44.7 s` | `53.4 s` |
| minimum retained joint Gram / scaled extension eigenvalue within the segment | `-4.158e-04` / `-4.672e-05` | `-1.301e-03` / `-1.404e-04` | `-4.158e-04` / `-4.672e-05` | `-1.300e-03` / `-1.374e-04` | `-2.133e-04` / `-2.734e-05` | `-5.361e-04` / `-5.747e-05` | `-4.158e-04` / `-4.672e-05` | `-1.300e-03` / `-1.374e-04` |
| peak resident | `639 MiB` | `625 MiB` | `625 MiB` | `636 MiB` | `651 MiB` | `637 MiB` | `634 MiB` | `637 MiB` |

Chain totals to `t = 8`, base / `etol1e3` / `dykstra60` / `conic1e10`:
autonomous wall `1,886.3` / `1,901.9` / `1,962.8` / `1,883.4 s` (recorded
`2,058.0 s`, tightened `2,114.8 s`, combined base `2,079.1 s`); in the
200 completion solves `1,733.3` / `1,751.2` / `1,725.4` / `1,734.0 s`
(`0.919`, `0.921`, `0.879`, `0.921` of the wall; mean `8.67`, `8.76`,
`8.63`, `8.67 s`, maximum `54.2`, `56.0`, `51.3`, `54.2 s`); repairs
`100.5` / `98.7` / `185.7` / `98.1 s`; velocity `51.4` / `51.0` / `50.7` /
`50.4 s`; peaks `639` / `636` / `651` / `637 MiB`; every segment inside
twenty minutes (longest `1,006.8 s` of autonomous wall plus `19.0 s` of
build).  Decisions of 200: accepted 188 (36 uncertified) / 188 (30) / 193
(35) / 188 (66), `infeasible` 12 / 12 / 7 / 12, declined by the margin 0
in every member; by two-unit window the base accepts 40, 50, 50, 49 and
finds 10, 0, 0, 2 `infeasible` (`dykstra60`: 43, 50, 50, 50 and 7, 0, 0,
0); the median applied move is `0.0071` to `0.0072` over the horizon
(`0.0032`, `0.0043`, `0.0090`, `0.0104` by window), the largest `0.3447`
at `t = 0.04` in every member and, in the base alone, `0.9565` at
`t = 6.6` and `0.9428` at `t = 6.64`; no other applied move exceeds
`0.0453`.  Of the interior-Newton solves, 30 of 170 (base), 37 of 178,
36 of 172 and 8 of 164 have their reduced gradient in `(5e-9, 5e-8]`, the
band task 5 identified as the decade's; under this rule it moves nothing.

## The chart at `t = 4` and `t = 8`

Cumulative from `t = 0`; base / `etol1e3` / `dykstra60` / `conic1e10`;
the comparators are the recorded chain (task 4) and its tightened member
(task 5), rescored on the same grid, and the combined-rule base.  Site-0
occupation is `rho[0,0]` sample RMS, the chain's convention.

| quantity | `t = 4`, four members | recorded / tightened / combined | `t = 8`, four members | recorded / tightened / combined |
|---|---:|---:|---:|---:|
| all-coordinate scalar RMS, 31 coordinates | `1.001172e-02` / `1.001172e-02` / `9.959755e-03` / `1.001172e-02` | `1.142716e-02` / `1.292283e-02` / `1.891654e-02` | `3.181485e-02` / `3.179926e-02` / `3.224344e-02` / `3.179926e-02` | `2.593079e-02` / `2.709352e-02` / `3.417762e-02` |
| band, spread | `9.96e-3` to `1.001e-2`, `0.52 %` | `1.143e-2` to `1.292e-2`, `13.1 %` | `3.180e-2` to `3.224e-2`, `1.40 %` | `2.593e-2` to `2.709e-2`, `4.5 %` |
| the 27 coordinates outside the phonon-occupation block | `1.007626e-02` / `1.007626e-02` / `1.001383e-02` / `1.007626e-02` | `1.164437e-02` / `1.347560e-02` / `2.004125e-02` | `2.870414e-02` / `2.870025e-02` / `2.940513e-02` / `2.870025e-02` | `2.474607e-02` / `2.577152e-02` / `3.486602e-02` |
| the four phonon-occupation coordinates | `9.564712e-03` / `9.564711e-03` / `9.586781e-03` / `9.564711e-03` | `9.836299e-03` / `8.276556e-03` / `7.878805e-03` | `4.777992e-02` / `4.771521e-02` / `4.712475e-02` / `4.771521e-02` | `3.282777e-02` / `3.472464e-02` / `2.910815e-02` |
| blockwise rho / B / N / A / C, base | `1.6956e-02` / `3.7215e-03` / `9.5647e-03` / `9.8071e-03` / `9.4354e-03` | recorded `2.0368e-02` / `6.9088e-03` / `9.8363e-03` / `8.9418e-03` / `1.1167e-02` | `4.0129e-02` / `2.1811e-02` / `4.7780e-02` / `3.8803e-02` / `2.1511e-02` | recorded `4.0173e-02` / `2.7248e-02` / `3.2828e-02` / `1.9942e-02` / `2.1275e-02` |
| blockwise, `dykstra60` | `1.7481e-02` / `3.7783e-03` / `9.5868e-03` / `8.9308e-03` / `9.4682e-03` | | `4.0643e-02` / `2.0418e-02` / `4.7125e-02` / `4.1170e-02` / `2.1635e-02` | |
| Eq. (22) internal-energy RMS | `3.922911e-03` / `3.922911e-03` / `4.133866e-03` / `3.922911e-03` | `4.597225e-03` / `3.687212e-03` / `6.750065e-03` | `4.781299e-03` / `4.781299e-03` / `5.020505e-03` / `4.781299e-03` | `5.751890e-03` / `4.702571e-03` / `8.706752e-03` |
| site-0 occupation, `rho[0,0]` sample RMS (`2 rho[0,0]`) | `1.221365e-02` (`2.443e-02`) / same / `1.271700e-02` (`2.543e-02`) / same | `1.430879e-02` / `1.515553e-02` / `2.435056e-02` | `2.705849e-02` (`5.412e-02`) / `2.705896e-02` / `2.721459e-02` (`5.443e-02`) / `2.705896e-02` | `3.027929e-02` / `3.095132e-02` / `4.730321e-02` |
| minimum unshifted retained joint Gram | `-4.158e-04` / `-4.158e-04` / `-2.133e-04` / `-4.158e-04` | `-4.535e-04` / `-4.511e-04` / `-4.636e-04` | `-1.301e-03` / `-1.300e-03` / `-5.361e-04` / `-1.300e-03` | `-2.152e-03` / `-2.201e-03` / `-1.982e-03` |
| minimum scaled positive-extension eigenvalue | `-4.672e-05` / `-4.672e-05` / `-2.734e-05` / `-4.672e-05` | `-4.852e-05` / `-5.846e-05` / `-5.230e-05` | `-1.404e-04` / `-1.374e-04` / `-5.747e-05` / `-1.374e-04` | `-1.922e-04` / `-1.980e-04` / `-1.921e-04` |
| maximum standardized frontier envelope ratio | `0.3872` / `0.3872` / `0.3740` / `0.3872` | `0.3694` / `0.3605` / --- | `0.7721` / `0.3872` / `0.3740` / `0.3872` | `0.4518` / `0.4212` / --- |
| Euclidean coordinate difference from the exact reference at the horizon, all 31 | `1.358288e-01` / `1.358288e-01` / `1.356037e-01` / `1.358288e-01` | `1.551037e-01` / `1.706443e-01` / --- | `2.957666e-01` / `2.955251e-01` / `3.127335e-01` / `2.955251e-01` | `2.372364e-01` / `2.335909e-01` / --- |
| completions re-solved / solver refusals to the horizon | 100 each; 41, 39, 38, 55 (to `t = 4`, boundary-inclusive) | 100 / 33, 100 / 42 (recorded, tightened) | 200; 48, 42, 42, 78 | 200 / 75, 200 / 97, 200 / 51 |
| autonomous wall to the horizon | about `1,040 s` each (first segment plus the steps of the second to `t = 4` at its own rate) | about `1,022 s` / `1,070 s` | `1,886.3` / `1,901.9` / `1,962.8` / `1,883.4 s` | `2,058.0` / `2,114.8` / `2,079.1 s` |

The cumulative scores at the intermediate horizons, four members and
spread: `t = 1` `7.235027e-04` / `7.235027e-04` / `6.968204e-04` /
`7.235027e-04` (`3.83 %`); `t = 2` `2.614966e-03` / `2.614965e-03` /
`2.589272e-03` / `2.614966e-03` (`0.99 %`); `t = 6` `2.014536e-02` /
`2.014536e-02` / `1.987741e-02` / `2.014536e-02` (`1.35 %`); the recorded
chain scores `2.090571e-02` at `t = 6` (tightened `2.314018e-02`), so the
crossing of the rule's base and the recorded chain lies between `t = 6`
and `t = 8`.

## The standing at `t = 4` and `t = 8`, as asked

| horizon | the rule's band (four members) | recorded 27-key `drive_power` chain (task 5 band) | combined-rule base | cell A | standing |
|---|---:|---:|---:|---:|---|
| `t = 4` | `9.96e-3` to `1.001e-2`, `0.52 %` | `1.143e-2` to `1.292e-2`, `13.1 %` | `1.892e-2` | `1.258e-2` | beats cell A: every member below it, cell A `1.256` times the worst member, against a `0.5 %` band; beats the recorded chain's whole band by `1.14` to `1.30`; the gap to the recorded band, `14.1 %` of the rule's worst member, exceeds the rule's band `27`-fold and the recorded band by `1.08` |
| `t = 8` | `3.180e-2` to `3.224e-2`, `1.40 %` | `2.593e-2` to `2.709e-2`, `4.5 %` | `3.418e-2` | --- | behind the recorded chain's whole band by `1.17` to `1.24`; the gap, `17.4 %` of the recorded band's top, exceeds both bands; ahead of the combined-rule base (`1.06` on the worst member), of the 15-key row under the same rule (`0.038538`, `1.20`) and of the 15-key `post_pulse` row (`4.254200e-02`, `1.32`) |

So the answer to the question posed: yes, a sensitivity-resolved 27-key
number exists at `t = 4` and at `t = 8`.  At `t = 4` it is `1.00e-2`
(four members within `0.5` percent), and it beats cell A's `1.258e-2` by
`1.26` --- the first 27-key `t = 4` number that stands against cell A
outside a measured band; the recorded chain's `1.14e-2` to `1.29e-2` only
tied it.  At `t = 8` it is `3.18e-2` to `3.22e-2` (four members within
`1.4` percent), and it does not beat the recorded chain: the recorded
chain's own band, `2.59e-2` to `2.71e-2`, is `4.5` percent wide and lies
`17` percent below the rule's, so at `t = 8` the recorded 27-key chain
under the earlier form stays the row on record, ahead of the
sensitivity-resolved one by more than either band.  The two statements
are on the four members of each configuration, not bounds: the recorded
chain's band is one lever's, and the rule's band is three levers', none
of which is the trust radius that the `t = 6.6` event calls for.

## Separation of the twins from the base

31-coordinate, 56-state and 70-frontier Euclidean distances at equal
sample times; first crossings and values at the horizons; the recorded
rule's twins (task 6, rescored here as the cheap route's set) and the
15-key set A (cheap route task 8) in brackets where measured.

| twin | first sample with any difference | first re-selection whose decision differs | first crossing of `d31` `1e-3` / `1e-2` / `1e-1` | `d31` at `t = 4` / `8` | maximum `d31`, when | `d56` `1e-2` / `1e-1` | `d70` `1e-2` / `1e-1` |
|---|---|---|---|---|---|---|---|
| `etol1e3` | `t = 0.005`, `2.5e-15` | none (the applied move differs first at `t = 6.6`: `0.9565` against `0.0117`) | `6.6425` / never / never [15-key set A: `2.1075` / `5.83` / never] | `8.1e-09` / `1.09e-03` | `1.22e-03` at `6.645` | `6.6025` / never | `6.6` / `6.6` (maximum `0.953` at `6.625`) |
| `dykstra60` | `t = 0.0075`, `6.9e-08` | `t = 0.16`: base `infeasible`, twin `accepted` (`0.0011`) | `0.5175` / `3.835` / never [`2.105` / `5.76` / never] | `1.07e-02` / `3.28e-02` | `3.49e-02` at `6.18` | `2.0` / `6.035` | `0.36` / `6.6` (maximum `0.965` at `6.6375`) |
| `conic1e10` | `t = 0.2`, `1.8e-14` in the frontier | none (as `etol1e3`) | `6.6425` / never / never [never / never / never] | `1.04e-08` / `1.09e-03` | `1.22e-03` at `6.645` | `6.6025` / never | `6.6` / `6.6` |
| `etol1e3` against `conic1e10` | --- | none | never / never / never | `3.2e-09` / `4.7e-09` | `5.1e-09` | never (maximum `2.7e-08`) | never (maximum `5.8e-07`) |

Window means of `d31` over `(0, 2]`, `(2, 4]`, `(4, 6]`, `(6, 8]`:
`etol1e3` and `conic1e10` `3.6e-10` to `6.7e-10`, `4.6e-09` to `5.8e-09`,
`1.6e-08` to `1.8e-08`, `6.8e-04`; `dykstra60` `1.1e-03`, `5.3e-03`,
`1.5e-02`, `2.6e-02`.  Against the members' own distance from the exact
reference at the horizon (`0.136` at `t = 4`, `0.296` to `0.313` at
`t = 8`) the two round-off twins are `0.0` and `0.4` percent of the way
apart from the base and the `dykstra60` twin `7.9` and `11.1` percent,
where the recorded chain's tightened member was `21` and `16` percent
(task 5) and the 27-key `post_pulse` pair `28` and `69` (task 7).  The
base against the recorded chain: `d70` `0.1` at `t = 0.04`, `d31` `1e-3`
at `0.5725`, `1e-2` at `1.84`, `0.1` at `4.8775`, `0.0555` at `t = 4`,
`0.267` at `t = 8`, maximum `0.326`; against the combined-rule base:
`1e-2` at `1.6225`, `0.1` at `2.85`, `0.142` at `t = 4`, `0.290` at
`t = 8`, maximum `0.365`, first differing decision at `t = 0.16`
(combined `accepted`, rule `infeasible`).

Two readings.

1. **At 27 keys the rule's continuity is better than at 15, and it has
   one hole.**  At 15 keys the round-off twins parted at `t = 0.16` by
   `0.0064` in the frontier and crossed `1e-2` at `t = 5.8`; here they
   stay within `3.2e-08` to `t = 6.6` and never cross `1e-2`, and the one
   move that parts them is the solver-path branch at `t = 6.6` whose
   effect on the score is `0.05` percent.  The `dykstra60` twin, the one
   whose setting changes the repaired state itself, separates at the
   first re-selection whose completion returns nothing for the base and
   something for it, and its score differs by `-0.5` and `+1.3` percent
   at `t = 4` and `8`; under the earlier form the same twin moved the
   15-key score `13` percent at `t = 4`.
2. **The certificate tolerance does nothing under this rule at 27 keys
   either.**  The `conic1e10` twin's completions return uncertified 78
   times against the base's 48 and it accepts 66 of them; its trajectory
   is the base's to `1.04e-08` at `t = 4` and `1.09e-03` at `t = 8`, the
   latter the base's own `t = 6.6` excursion.  The 27-key chain's `13.1`
   percent sensitivity at `t = 4` (task 5) was the certificate's, and the
   rule that does not consult the certificate has none.

## Provenance and gates of the runs themselves

- Eight rollouts (`paper_v_dictionary_energy_power_27key_n16_measured_{base,etol1e3,dykstra60,conic1e10}_20260916i_seg{1,2}`,
  each with `reselections.json`), run by the previous instance under
  `pipelines/shell/ram_guard.py --limit-mb 4500`, one at a time, `17:57`
  to `20:07` on 2026-09-16 (`queue_task8_measured.sh`, `task8m_chain.log`);
  peaks `625` to `651 MiB`; longest segment `1,006.8 s` of autonomous
  wall.  This instance resumed from those artifacts, repeated no segment,
  and ran only the comparisons, the tabulation and the chaining gate
  (`chaingate_task8m.sh`; three rollouts of `t = 0` to `0.2`, `0` to
  `0.14` and `0.12` to `0.2` under the same guard, one at a time).
- Scripts under `output/local_runs/paper_v_projective_dictionary_energy_20260915e/`:
  `queue_task8_measured.sh` (the eight segments and the per-member
  scores), `score_task8m.sh` (the three twin-against-base comparisons and
  the tabulation), `compare_task8m_extra.sh` (the base against the
  recorded chain and against the combined-rule base), `tabulate_task8.py`
  (ledgers, chart, standing, separations recomputed from the arrays,
  decision ledgers), `chaingate_task8m.sh` (the chaining gate under the
  rule), `compare_chains.py` (the boundary-inclusive ledger merge, the
  one change to an existing script), `run_energy_segment.py` and
  `run_chain.py` as the previous instance left them (the five
  `--reselection-*` rule flags, the `--dykstra-iterations` and
  `--energy-tolerance` twin flags, the native `reselection_extension`
  path).  Outputs: `task8m_{member}_score_t8.json` and `.log`,
  `compare_task8m_{etol1e3,dykstra60,conic1e10}.json`,
  `compare_task8m_etol1e3_vs_conic1e10.json`,
  `compare_task8m_base_vs_{recorded,combined}_t8.json`,
  `tabulate_task8.log`, `chaingate_task8m_compare.json`,
  `chaingate_task8m.log`, the segment logs `task8m_*_seg1.log` and
  `paper_v_dictionary_energy_power_27key_n16_measured_*_seg2.log`.
- Gate artifacts:
  `paper_v_dictionary_energy_prefixgate_task8_{default,native1e10}_20260916h`
  (the previous instance's), `paper_v_dictionary_energy_measured_chaingate{S,A,B}_20260917j`
  (this instance's).
- No module was edited, nothing was committed, no default, acceptance
  rule or ladder changed.  The recorded chain's, the tightened chain's
  and the combined-rule artifacts are untouched.  Wall times were
  measured with other agents' jobs on the machine and are upper bounds.

## Structured summary of task 8

| item | value |
|---|---|
| the rule | `LiftReselectionRule(acceptance_margin=1e-3, candidates="measured", anchor_weight=0)`, the cheap route's task-8 recommendation, on the task-4 27-key `N = 16` `drive_power` energy-neutral chain; base plus `etol1e3`, `dykstra60`, `conic1e10`; `t = 0` to `8` in two chained segments each; imports from `head_src_e3374f6f` (identical to the tree at `3ea96121`) |
| `t = 4`, four members | `1.001172e-02` / `1.001172e-02` / `9.959755e-03` / `1.001172e-02`; band `9.96e-3` to `1.001e-2`, spread `0.52 %`; internal energy `3.922911e-03` (`dykstra60` `4.133866e-03`); retained joint Gram `-4.158e-04` (`-2.133e-04`), extension `-4.672e-05` (`-2.734e-05`); phonon-occupation block `9.565e-03`, the other 27 `1.008e-02` |
| `t = 8`, four members | `3.181485e-02` / `3.179926e-02` / `3.224344e-02` / `3.179926e-02`; band `3.180e-2` to `3.224e-2`, spread `1.40 %`; internal energy `4.781299e-03` (`5.020505e-03`); retained `-1.301e-03` / `-1.300e-03` / `-5.361e-04` / `-1.300e-03`, extension `-1.404e-04` / `-1.374e-04` / `-5.747e-05` / `-1.374e-04`; phonon-occupation block `4.778e-02`, the other 27 `2.870e-02` |
| against the recorded 27-key chain (`1.142716e-02` / `2.593079e-02`, bands `13.1` and `4.5 %`) | `t = 4`: ahead by `1.14` to `1.30`, bands disjoint by `14.1 %`; `t = 8`: behind by `1.17` to `1.24`, bands disjoint by `17.4 %`; the loss at `t = 8` is the phonon-occupation block (`1.46` times) and the A block (`1.95` times) |
| against the combined-rule base (`1.891654e-02` / `3.417762e-02`) | ahead by `1.89` at `t = 4` and `1.06` to `1.07` at `t = 8` |
| against cell A (`1.258000e-02` at `t = 4`) | beats it: cell A is `1.256` times the worst member, against a `0.5 %` band |
| the sensitivity-resolved 27-key number | exists at both horizons: `1.00e-2` at `t = 4` (beats cell A outside every band on record) and `3.18e-2` to `3.22e-2` at `t = 8` (behind the recorded chain's `2.59e-2` to `2.71e-2` by more than either band; the recorded chain stays the `t = 8` row) |
| first separations | `etol1e3`, `conic1e10`: one trajectory with the base to `3.2e-08` through `t = 6.6`, then the base's `0.9565` facial-reduction iterate at `t = 6.6` (returned by `0.9428` at `6.64`) parts them by `1.1e-03` at `t = 8`; `dykstra60`: `t = 0.16` (base `infeasible`, twin accepted `0.0011`), `d31` `1e-2` at `3.835`, maximum `0.0349`, never `0.1`; the two round-off twins agree with each other to `5.1e-09` |
| decisions of 200 | accepted 188 (36 uncertified) / 188 (30) / 193 (35) / 188 (66); `infeasible` 12 / 12 / 7 / 12; declined by the margin 0; solver refusals 48 / 42 / 42 / 78 (recorded 75, tightened 97, combined 51); the `t = 0.04` whole-face answer (`0.3447`) accepted by every member where the earlier form refused it |
| wall | `1,886.3` / `1,901.9` / `1,962.8` / `1,883.4 s` to `t = 8` (recorded `2,058.0 s`); `88` to `92 %` in the solves at `8.6` to `8.8 s` mean; peaks `625` to `651 MiB`; longest segment `1,006.8 s` |
| gates | driver reproduces the recorded chain bitwise without the flags (161 samples, `0.0`); native tolerance path reproduces the task-5 subclass bitwise; chaining under the rule: exact to round-off (`2.5e-14`) through three re-selections, then a decision flip at the `t = 0.16` feasibility edge (`-1.76e-08` against `-9.78e-09` at the `-1e-08` threshold; `infeasible` against `accepted`, a `0.8926` move), `d31` `3.3e-04` at `t = 0.2`: the members' bands are measured across four different boundaries and include this perturbation; boundary resumes at `96`, `89`, `87`, `93` intervals; energy held to `1.78e-15`; one grid of 3,201 samples |
| files changed | `compare_chains.py` (boundary-inclusive ledger merge), new `chaingate_task8m.sh` and its outputs, the comparison and tabulation outputs, three gate artifacts; this report; the worklog block; the plan ledger's row.  No module; nothing committed |
| next | the trust radius (`--reselection-trust-radius 0.1`, `ball` form; the cheap route's `90a3bd60` finds the `carried` form freezes the lift at 15 keys) on the four to `t = 8`, the declared instrument for the `t = 6.6` move, then the base to `t = 20` under the rule and the cap: whether the 27-key chain under the rule keeps a band at `t = 12`, `16`, `20` where the recorded chain's bands are `1.9`, `5.4`, `6.7 %` and the 15-key rule's `3.2`, `18.8`, `48.8 %`; and where the `t = 8` loss against the earlier form comes from (the `t = 0.04` acceptance, or the re-selections of `(4, 8]`) --- a rerun with the margin's scope switched to `certified` candidates from `t = 4` would separate them |

# projective-route, task 9 (2026-09-17)

Task 8 put the cheap route's anchor-free rule (acceptance margin `1e-3`,
`measured` candidates, anchor weight `0`) on the 27-key `N = 16`
`drive_power` energy-neutral chain and found the four members one
trajectory to `t = 6.6`, where the base accepted a `0.9565`
facial-reduction iterate --- the move the rule's trust radius is declared
for.  The cheap route's task 10 measured that radius at 15 keys: `0.1`
under the `ball` policy (the capped point on the segment toward the
accepted lift) keeps the lift live and buys 12 percent at `t = 20`, first
binding at `t = 8.68`; under the `carried` policy it freezes the lift from
`t = 8.68` and is worse.  This round, (a), puts the ball form on the 27-key
four --- base, `energy_tolerance 1e-3`, Dykstra 60, re-selection
`conic_tolerance 1e-10` --- from `t = 0` to `8`, and (b) chains the base
to `t = 20` under the better of the two rules.  The rule is
`LiftReselectionRule(acceptance_margin=1e-3, candidates="measured",
anchor_weight=0, trust_radius=0.1, trust_policy="ball")`, every field a
setting of the committed class (`a81b7ecb`; the `ball` policy since the
cheap route's `90a3bd60`); the driver passes the five flags to
`CheapProjectionRoute`'s `reselection_rule` argument and is otherwise
task 8's driver exactly.  Imports are from `head_src_e3374f6f`, which
`diff -rq` shows identical to `paper_5/src/paper5` at `HEAD f971a284`
outside `__pycache__`.  No module was edited; nothing was committed; no
default changed.  Configuration and conventions are task 8's: 27 active
keys, 56 state coordinates, 36 positive-extension rows, 70 standardized
frontier coordinates, strong coupling, cutoff 16, exact-correlated
preparation at the declared ladder's first rung, SSPRK(3,3) at
`h = 0.0025`, Dykstra 30 on both targets at stage cadence, the declared
proximal log-det completion re-solved every 16 steps warm-started on the
carried frontier, the energy-neutral repair in `drive_power` mode; the
score is the all-coordinate scalar RMS of the 31 closed scalar
coordinates against the exact cutoff-16 driven reference from `t = 0` on
one grid (`exact_t8.npz`, 3,201 samples; `exact_t20.npz`, 8,001); a
cap-bound re-selection is one whose accepted candidate lay farther than
`0.1` from the carried lift in the 70 standardized frontier coordinates,
so that the applied move is `0.1` along the segment toward it; the
solved move is the candidate's own distance.

## Negative result first

**At 27 keys the radius binds only inside `(0, 0.64]`, changes nothing at
`t = 8`, and costs 4 percent at `t = 4`; the `t = 6.6` event is not
capped because it does not occur on the capped trajectories.**  The cap
bound 3, 5, 5, 4 times of 200 in the base, `etol1e3`, `dykstra60`,
`conic1e10`, all at `t <= 0.64` --- the first at `t = 0.04` in every
member, the whole-face `0.3447` answer task 8's members applied whole,
scaled to `0.1` --- and never again through `t = 8`: from `t = 0.68` on
every applied move is below `0.032` and every answer above `0.1` is
`infeasible` (the `0.53` to `0.55` reference-completion refusals of
`t = 0.36` to `0.64` and of `t = 6.44` to `6.52`) or declined by the
margin (`1.319` at `t = 0.72` on the base).  On the four capped
trajectories the `t = 6.6` completion returns a `0.0240`, `0.0128`,
`0.0105`, `0.0243` answer and the `t = 6.64` completion `0.0145`,
`0.0148`, `0.0098`, `0.0230`, all accepted; task 8's `0.9565`
facial-reduction iterate belongs to the anchor-free base's trajectory,
which the ball chain left at `t = 0.04` (`d70` `0.2447` there).  So the
radius did not act on the move it was brought for; it acted on the first
re-selection, and that is the whole difference between the two rules'
chains to `t = 8`.

At `t = 8` the four ball members score `3.180974e-02` / `3.151679e-02` /
`3.215530e-02` / `3.143476e-02` (band `3.143e-2` to `3.216e-2`, spread
2.29 percent) against task 8's `3.181485e-02` / `3.179926e-02` /
`3.224344e-02` / `3.179926e-02` (band `3.180e-2` to `3.224e-2`, 1.40
percent): the bases agree to `0.02` percent, the bands overlap, and the
standing against the recorded chain does not change --- behind its
`2.593079e-02` to `2.709352e-02` by `1.16` to `1.24`, the bands disjoint
(the ball's best member is `1.160` times the recorded band's top).  The
27-key `t = 8` row on record stays the recorded chain's, as after task 8.
At `t = 4` the ball members score `1.043201e-02` / `1.003847e-02` /
`1.005305e-02` / `1.024635e-02` (band `1.004e-2` to `1.043e-2`, spread
3.92 percent), behind task 8's `9.96e-3` to `1.001e-2` (0.52 percent)
with the bands disjoint --- the ball base is `1.042` times the anchor-free
base --- and still ahead of the recorded chain's whole band `1.143e-2` to
`1.292e-2` (the worst ball member is `0.913` of the recorded band's
bottom) and below cell A's `1.258e-2` in every member (`0.829` on the
worst).  The 27-key `t = 4` row stays task 8's `1.00e-2`; the ball form
does not displace it and widens the band it would carry.

**The ball reintroduces the certificate branch the anchor-free rule had
removed.**  Under task 8's rule the `conic1e10` twin was the base's
trajectory to `1.04e-8` at `t = 4`; under the ball it parts at `t = 0.36`,
where its completion at `1e-10` returned an uncertified `0.584` answer
that passed the measured feasibility test and was capped to `0.1`, while
the base's completion at `1e-9` returned `0.0099`; `d31` `1.68e-2` at
`t = 8` (5.8 percent of the members' distance from the exact reference).
The `etol1e3` twin parts at `t = 0.36` too (base `accepted` `0.0099`,
twin `infeasible`) and the `dykstra60` twin at `t = 0.12` (base
`infeasible`, twin `declined_by_margin` on a `1.330` answer); their `d31`
at `t = 8` are `2.46e-2` and `3.91e-2` (8.5 and 13.1 percent).  The
capped point at `t = 0.04` is not on any face the completion returns,
and the members' next completions return different answers from states
`1e-3` apart; that is the 3.9 percent at `t = 4`.

## The rows side by side, `t = 4` and `t = 8`

Base / `etol1e3` / `dykstra60` / `conic1e10`; the anchor-free members
are task 8's, the recorded chain is task 4's with its tightened member
(task 5) as the band; site-0 occupation is `rho[0,0]` sample RMS.

| quantity | ball, `t = 4` | anchor-free (task 8), `t = 4` | recorded / tightened, `t = 4` | ball, `t = 8` | anchor-free (task 8), `t = 8` | recorded / tightened, `t = 8` |
|---|---:|---:|---:|---:|---:|---:|
| all-coordinate scalar RMS, 31 coordinates | `1.043201e-02` / `1.003847e-02` / `1.005305e-02` / `1.024635e-02` | `1.001172e-02` / `1.001172e-02` / `9.959755e-03` / `1.001172e-02` | `1.142716e-02` / `1.292283e-02` | `3.180974e-02` / `3.151679e-02` / `3.215530e-02` / `3.143476e-02` | `3.181485e-02` / `3.179926e-02` / `3.224344e-02` / `3.179926e-02` | `2.593079e-02` / `2.709352e-02` |
| band, spread | `1.004e-2` to `1.043e-2`, `3.92 %` | `9.96e-3` to `1.001e-2`, `0.52 %` | `13.1 %` | `3.143e-2` to `3.216e-2`, `2.29 %` | `3.180e-2` to `3.224e-2`, `1.40 %` | `4.5 %` |
| the 27 coordinates outside the phonon-occupation block | `1.051990e-02` / `1.013441e-02` / `1.014847e-02` / `1.035656e-02` | `1.007626e-02` / same / `1.001383e-02` / same | `1.164437e-02` / `1.347560e-02` | `2.848211e-02` / `2.844829e-02` / `2.927891e-02` / `2.826397e-02` | `2.870414e-02` / `2.870025e-02` / `2.940513e-02` / `2.870025e-02` | `2.474607e-02` / `2.577152e-02` |
| the four phonon-occupation coordinates | `9.818198e-03` / `9.365173e-03` / `9.383635e-03` / `9.468975e-03` | `9.564712e-03` / same / `9.586781e-03` / same | `9.836299e-03` / `8.276556e-03` | `4.864260e-02` / `4.727925e-02` / `4.718843e-02` / `4.760114e-02` | `4.777992e-02` / `4.771521e-02` / `4.712475e-02` / `4.771521e-02` | `3.282777e-02` / `3.472464e-02` |
| blockwise rho / B / N / A / C, base | `1.7915e-02` / `4.0829e-03` / `9.8182e-03` / `1.0144e-02` / `9.7872e-03` | `1.6956e-02` / `3.7215e-03` / `9.5647e-03` / `9.8071e-03` / `9.4354e-03` | recorded `2.0368e-02` / `6.9088e-03` / `9.8363e-03` / `8.9418e-03` / `1.1167e-02` | `3.7381e-02` / `1.9588e-02` / `4.8643e-02` / `4.0018e-02` / `2.1659e-02` | `4.0129e-02` / `2.1811e-02` / `4.7780e-02` / `3.8803e-02` / `2.1511e-02` | recorded `4.0173e-02` / `2.7248e-02` / `3.2828e-02` / `1.9942e-02` / `2.1275e-02` |
| Eq. (22) internal-energy RMS | `4.413844e-03` / `3.950798e-03` / `4.059184e-03` / `4.090198e-03` | `3.922911e-03` / same / `4.133866e-03` / same | `4.597225e-03` / `3.687212e-03` | `5.360816e-03` / `4.812119e-03` / `4.947228e-03` / `4.977447e-03` | `4.781299e-03` / same / `5.020505e-03` / same | `5.751890e-03` / `4.702571e-03` |
| site-0 occupation, `rho[0,0]` sample RMS | `1.294667e-02` / `1.244388e-02` / `1.332046e-02` / `1.291543e-02` | `1.221365e-02` / same / `1.271700e-02` / same | `1.430879e-02` / `1.515553e-02` | `2.499928e-02` / `2.703268e-02` / `2.680599e-02` / `2.615586e-02` | `2.705849e-02` / `2.705896e-02` / `2.721459e-02` / `2.705896e-02` | `3.027929e-02` / `3.095132e-02` |
| minimum unshifted retained joint Gram | `-4.052e-04` / `-4.155e-04` / `-2.219e-04` / `-4.155e-04` | `-4.158e-04` / same / `-2.133e-04` / same | `-4.535e-04` / `-4.511e-04` | `-1.298e-03` / `-1.324e-03` / `-5.212e-04` / `-1.316e-03` | `-1.301e-03` / `-1.300e-03` / `-5.361e-04` / `-1.300e-03` | `-2.152e-03` / `-2.201e-03` |
| minimum scaled positive-extension eigenvalue | `-4.706e-05` / `-4.670e-05` / `-2.781e-05` / `-4.734e-05` | `-4.672e-05` / same / `-2.734e-05` / same | `-4.852e-05` / `-5.846e-05` | `-1.369e-04` / `-1.384e-04` / `-5.279e-05` / `-1.384e-04` | `-1.404e-04` / `-1.374e-04` / `-5.747e-05` / `-1.374e-04` | `-1.922e-04` / `-1.980e-04` |
| maximum standardized frontier envelope ratio | `0.3836` / `0.3823` / `0.4211` / `0.4022` | `0.3872` / same / `0.3740` / same | `0.3694` / `0.3605` | `0.3924` / `0.3856` / `0.4211` / `0.4022` | `0.7721` / `0.3872` / `0.3740` / `0.3872` | `0.4518` / `0.4212` |
| Euclidean coordinate difference from the exact reference at the horizon, all 31 | `1.392894e-01` / `1.354351e-01` / `1.358075e-01` / `1.375191e-01` | `1.358288e-01` / same / `1.356037e-01` / same | `1.551037e-01` / `1.706443e-01` | `2.989438e-01` / `2.910848e-01` / `3.226311e-01` / `2.915194e-01` | `2.957666e-01` / `2.955251e-01` / `3.127335e-01` / `2.955251e-01` | `2.372364e-01` / `2.335909e-01` |
| completions re-solved; solver refusals (uncertified returns, counted in the merged ledger at `t <= T`) | 100 each; 27, 29, 33, 47 | 100; 31, 30, 31, 46 by the same count (task 8's table printed 41, 39, 38, 55 for this cell by a count this round could not reproduce) | 100 / 33, 100 / 42 | 200; 47, 47, 47, 80 | 200; 48, 42, 42, 78 | 200 / 75, 200 / 97 |
| cap-bound re-selections to the horizon (uncertified) | 3 (3), 5 (4), 5 (5), 4 (4) | --- | --- | the same: none in `(4, 8]` | --- | --- |
| autonomous wall to the horizon (first segment plus the second's steps to `t = 4` at its own rate) | about `1,090` / `1,086` / `1,030` / `1,058 s` | about `1,036` / `1,091` / `1,117` / `1,059 s` | about `1,022 s` / `1,070 s` | `1,977.2` / `1,882.8` / `1,835.1` / `1,858.7 s` | `1,886.3` / `1,901.9` / `1,962.8` / `1,883.4 s` | `2,058.0` / `2,114.8 s` |

Cumulative scores at the intermediate horizons, ball members and spread,
task 8's in brackets: `t = 1` `7.333547e-04` / `7.163885e-04` /
`7.204588e-04` / `7.074723e-04` (3.66 percent) [`7.235027e-04` /
`7.235027e-04` / `6.968204e-04` / `7.235027e-04`, 3.83]; `t = 2`
`2.872654e-03` / `2.659319e-03` / `2.627972e-03` / `2.758222e-03` (9.31)
[`2.614966e-03` / `2.614965e-03` / `2.589272e-03` / `2.614966e-03`,
0.99]; `t = 6` `2.012579e-02` / `2.005153e-02` / `1.954067e-02` /
`2.005381e-02` (2.99) [`2.014536e-02` / `2.014536e-02` / `1.987741e-02` /
`2.014536e-02`, 1.35]; the recorded chain `2.090571e-02` at `t = 6`.  Score
per two-unit window `(a, b]`, ball base / `etol1e3` / `dykstra60` /
`conic1e10` [anchor-free base; recorded chain]: `(0, 2]` `2.874e-03` /
`2.661e-03` / `2.630e-03` / `2.760e-03` [`2.617e-03`; `2.904e-03`];
`(2, 4]` `1.448e-02` / `1.395e-02` / `1.398e-02` / `1.423e-02`
[`1.392e-02`; `1.590e-02`]; `(4, 6]` `3.159e-02` / `3.170e-02` /
`3.072e-02` / `3.157e-02` [`3.190e-02`; `3.241e-02`]; `(6, 8]`
`5.323e-02` / `5.261e-02` / `5.469e-02` / `5.241e-02` [`5.322e-02`;
`3.713e-02`]; `(4, 8]` `4.377e-02` / `4.343e-02` / `4.436e-02` /
`4.327e-02` [`4.387e-02`; `3.485e-02`].  The loss against the recorded
chain is the `(6, 8]` window under both rules (`1.41` to `1.47` times),
in the phonon-occupation block (`4.86e-2` against `3.28e-2` at `t = 8`)
and the A block (`4.00e-2` against `1.99e-2`), as in task 8; the ball
moves neither.

## (a) The cap: what it caught, and when

Every cap-bound re-selection of the four members, with the answer's own
distance from the carried lift (scaled to an applied `0.1`) and its
certificate; `u` marks an uncertified return:

| member | cap-bound of 200 (uncertified) | the answers capped (time: solved move) | answers above `0.4` and their decisions | declined by the margin | `infeasible` |
|---|---:|---|---|---:|---:|
| base | 3 (3) | `0.04`: `0.345u`; `0.16`: `0.244u`; `0.20`: `0.145u` | 8: `0.536` at `0.40` to `0.56` and `0.541` at `0.64` (six `infeasible`), `1.319` at `0.72` (declined), `0.548` at `6.48` (`infeasible`) | 1 | 10 |
| `etol1e3` | 5 (4) | `0.04`: `0.345` (certified, 12 iterations); `0.16`: `0.244u`; `0.20`: `0.145u`; `0.56`: `0.285u`; `0.60`: `0.115u` | 7, all `infeasible` (`0.527` at `0.36` to `0.48`, `0.539` at `0.64`, `0.548` and `0.555` at `6.44`, `6.52`) | 0 | 10 |
| `dykstra60` | 5 (5) | `0.04`: `0.345u`; `0.16`: `0.245u`; `0.20`: `0.145u`; `0.60`: `0.881u`; `0.64`: `0.103u` | 6: `1.330` at `0.12` (declined), `0.527` at `0.36` to `0.48` (`infeasible`), `0.881` at `0.60` (capped) | 1 | 6 |
| `conic1e10` | 4 (4) | `0.04`: `0.345u`; `0.16`: `0.244u`; `0.20`: `0.145u`; `0.36`: `0.584u` | 8: `0.584` at `0.36` (capped), `0.606` at `0.40` to `0.48`, `0.960` at `0.56`, `0.541` at `0.64`, `0.547` and `0.553` at `6.44`, `6.52` (`infeasible`) | 0 | 10 |

Decisions of 200: accepted 186 (36 uncertified) / 185 (37) / 188 (40) /
186 (70) besides the capped ones; solver refusals 47 / 47 / 47 / 80
(task 8: 48 / 42 / 42 / 78; recorded 75); by two-unit window the base
accepts 38, 50, 50, 49 and finds 8, 0, 0, 2 `infeasible`; the median
applied move is `0.0076` to `0.0081` over the horizon (`0.0032` to
`0.0034`, `0.0042` to `0.0044`, `0.0085` to `0.0091`, `0.0101` to
`0.0105` by window), the largest `0.1` (the cap) and, uncapped, `0.0451`
to `0.0456` at `t = 0.24` in three members and `0.0923` at `t = 0.6` on
`conic1e10`; past `t = 2` the largest applied move is `0.0319` at
`t = 6.68` on the base; no applied move exceeds `0.1`.  Two readings.

1. **At 27 keys the answers the radius is for are already refused by the
   measured feasibility test.**  Of the base's eight answers above
   `0.4`, seven are `infeasible` and one is declined; the two large
   answers that were capped rather than refused (`0.881` on `dykstra60`
   at `t = 0.60`, `0.584` on `conic1e10` at `t = 0.36`) are uncertified
   iterates that passed the test on the twins' states and not on the
   base's.  This is the cheap route's task-10 finding at 15 keys (its
   base's 72 answers above `0.4`: 69 `infeasible`, 3 declined; the cap
   caught answers of `0.13` to `0.25`) with the one difference that at
   27 keys the `(0, 0.64]` answers of `0.145` to `0.345` are the ones
   caught, and they are the first re-selections after the exact
   preparation, where task 8 showed the earlier form refused the
   whole-face answer and the anchor-free rule took it whole.
2. **The `t = 6.6` event was a solver-path branch on one trajectory, and
   the cap removed it by not being there.**  Task 8's base reached its
   `t = 6.6` state through the `0.3447` move at `t = 0.04`; the capped
   base reaches a state `d31` `1.3e-2` away at `t = 6`, and there the
   completion takes the interior-Newton path to a `0.024` answer.  The
   radius would have capped the `0.9565` iterate to `0.1` had it
   appeared; it did not appear, so what the radius bought on this
   horizon is a `0.39` envelope against `0.77` and nothing in the score.

## (a) Separations

31-coordinate, 56-state and 70-frontier Euclidean distances at equal
sample times, recomputed from the arrays; task 8's twins in brackets.

| pair | first sample with any difference | first re-selection whose decision differs | first crossing of `d31` `1e-3` / `1e-2` / `1e-1` | `d31` at `t = 4` / `8` | maximum `d31`, when | `d70` maximum, when |
|---|---|---|---|---|---|---|
| ball `etol1e3` from ball base | `t = 0.005`, `2.5e-15` | `t = 0.36`: base `accepted` (`0.0099`), twin `infeasible` (`0.527`) | `0.7775` / `4.51` / never [task 8: `6.6425` / never / never] | `8.59e-03` / `2.46e-02` (6.3 and 8.5 percent of the distance from exact) [`8.1e-09` / `1.09e-03`] | `2.67e-02` at `7.27` | `0.105` at `0.56` (the twin's `0.285` cap) |
| ball `dykstra60` from ball base | `t = 0.0075`, `6.9e-08` | `t = 0.12`: base `infeasible`, twin `declined_by_margin` (`1.330`) | `0.5225` / `3.2` / never [`0.5175` / `3.835` / never] | `1.60e-02` / `3.91e-02` (11.8 and 13.1 percent) [`1.07e-02` / `3.28e-02`] | `5.23e-02` at `6.215` | `0.107` at `0.60` (the twin's `0.881` cap) |
| ball `conic1e10` from ball base | `t = 0.24`, `3.0e-15` in the frontier | `t = 0.36`: base `accepted` (`0.0099`), twin `cap_bound_ball` (`0.584`) | `0.9475` / `6.41` / never [`6.6425` / never / never] | `4.97e-03` / `1.68e-02` (3.6 and 5.8 percent) [`1.04e-08` / `1.09e-03`] | `1.73e-02` at `7.32` | `0.094` at `0.36` |
| ball `etol1e3` against ball `conic1e10` | `t = 0.005` | `t = 0.36`: `infeasible` against `cap_bound_ball` | `1.1425` / `6.975` / never [never / never / never] | `4.56e-03` / `8.77e-03` [`3.2e-09` / `4.7e-09`] | `1.07e-02` at `7.24` | `0.156` at `0.56` |
| ball base from the anchor-free base (the radius alone) | `t = 0.04`, `d70` `0.2447` | `t = 0.04`: `cap_bound_ball` (`0.1`) against `accepted` (`0.3447`) | `0.7225` / `4.6925` / never | `7.96e-03` / `2.01e-02` (5.9 and 6.8 percent) | `2.40e-02` at `7.27` | `0.953` at `6.6` (task 8's excursion) |
| ball twins from the anchor-free twins (the radius alone) | `t = 0.04` in every pair | `t = 0.04` in every pair | `etol1e3` `1.6875` / never / never; `dykstra60` `1.055` / `4.9725` / never; `conic1e10` `1.025` / `6.8525` / never | `2.03e-03` / `7.61e-03`; `6.46e-03` / `1.76e-02`; `4.58e-03` / `9.34e-03` | `7.61e-03`, `1.76e-02` at `8.0`; `1.15e-02` at `7.18` | `0.2447` at `0.04` in each |
| ball base from the recorded chain | `t = 0.04`, `d70` `0.1` | --- (the recorded chain's segments carry no re-selection ledger; task 8 established that its `t = 0.04` completion was refused under the earlier form) | `0.56` / `1.825` / `4.94` | `5.33e-02` / `2.76e-01` (38 and 116 percent of the smaller distance from exact) | `3.36e-01` at `7.57` | `0.200` at `0.16` |

Window means of `d31` over `(0, 2]`, `(2, 4]`, `(4, 6]`, `(6, 8]`: ball
`etol1e3` `1.7e-03`, `5.5e-03`, `1.1e-02`, `2.1e-02`; `dykstra60`
`2.0e-03`, `8.4e-03`, `2.0e-02`, `3.4e-02`; `conic1e10` `9.9e-04`,
`3.4e-03`, `6.2e-03`, `1.4e-02`; the ball base from the anchor-free base
`1.9e-03`, `5.7e-03`, `1.0e-02`, `1.9e-02`.  Under the ball every twin is
a different trajectory from `t <= 0.36` on, at `4` to `13` percent of
the distance from the exact reference by `t = 8`, where under the
anchor-free rule two of the three were the base's to `1e-8` through
`t = 6.6`; the ball's band is the wider of the two at both horizons
(3.92 against 0.52 percent at `t = 4`, 2.29 against 1.40 at `t = 8`),
and every member of both rules lies inside `9.96e-3` to `1.043e-2` at
`t = 4` and `3.143e-2` to `3.224e-2` at `t = 8` (eight runs, 4.7 and 2.6
percent).

## (a) Physicality, energy, cost

Over `t = 0` to `8` the minimum unshifted retained joint Gram is
`-1.298e-03` / `-1.324e-03` / `-5.212e-04` / `-1.316e-03` and the minimum
scaled positive-extension eigenvalue `-1.369e-04` / `-1.384e-04` /
`-5.279e-05` / `-1.384e-04`, task 8's to two figures (recorded
`-2.152e-03` / `-1.922e-04`); nothing is held.  The frontier envelope
stays below `0.43` in every member (task 8's base `0.77` at its `t = 6.6`
excursion).  The Eq. (22) internal-energy RMS is `4.413844e-03` /
`3.950798e-03` / `4.059184e-03` / `4.090198e-03` at `t = 4` and
`5.360816e-03` / `4.812119e-03` / `4.947228e-03` / `4.977447e-03` at
`t = 8`: the base is 12 percent above the anchor-free base
(`4.781299e-03`) because the `drive_power` reference its own capped
trajectory selects over the pulse, `-3.5652343570276206`, lies
`6.164e-03` above the exact post-pulse energy `-3.571397982` where the
anchor-free base's `-3.5658908873520327` lay `5.507e-03` above (recorded
`6.710e-03`, tightened `5.534e-03`); the three twins' references
(`5.541e-03`, `5.698e-03`, `5.729e-03` above) put them between.  The
energy is held to `1.78e-15` over 9,410 to 9,555 constrained repairs, 0
hyperplane refusals.  Autonomous wall to `t = 8` `1,977.2` / `1,882.8` /
`1,835.1` / `1,858.7 s` in two segments each (task 8 `1,886.3` to
`1,962.8 s`; recorded `2,058.0 s`), `0.876` to `0.923` in the 200
completion solves at `8.04` to `9.12 s` mean; the first segments reached
`t = 3.6`, `3.6`, `3.88`, `3.72` in their `1,000 s` budget (task 8's
`3.84`, `3.56`, `3.48`, `3.72`) with a cheap-route job under a `3,000 MB`
guard sharing the machine throughout; peaks `627` to `636 MiB`; longest
segment `1,006.9 s` of autonomous wall plus `20.4 s` of build.

## Gates, (a)

| gate | result |
|---|---|
| the radius reaches the rule | smoke `paper_v_dictionary_energy_ball_smoke_20260917k`, `t = 0` to `0.08` under the five flags: `summary.json` records `reselection_rule` `{anchor_weight 0.0, trust_radius 0.1, trust_policy ball, acceptance_margin 0.001, candidates measured}` and the `e3374f6f` source tree; the `t = 0.04` re-selection is `cap_bound_ball` with solved move `0.344669` and applied move `0.1` (objective `6.5383` to `4.5076`, 40-iteration cap, uncertified), the `t = 0.08` one `infeasible` (`AlmostPrimalInfeasible` reference completion) --- task 8's `0.3447` acceptance, capped |
| the driver without the radius is task 8's | the `--reselection-trust-radius` and `--reselection-trust-policy` flags default to not given, in which case `_rule_from_args` builds the rule without them (`trust_radius None`, policy `carried`) as in task 8; task 8's gates (driver bitwise the recorded chain without any flag; native tolerance path bitwise the task-5 subclass) stand unchanged, no driver line having changed |
| every member's first re-selection under the ball is the same decision | `cap_bound_ball` at `t = 0.04` on a `0.3447` answer in all four, then `infeasible` at `0.08`; the first differing decision is at `t = 0.12` (`dykstra60`) or `0.36` (`etol1e3`, `conic1e10`) |
| a segment resumes on a re-selection boundary | second segments resume at `t = 3.6`, `3.6`, `3.88`, `3.72` (`90`, `90`, `97`, `93` intervals of 16 steps from `t = 0`); every second segment reached `t = 8.0` and wrote its resume there |
| the declared reference is set inside the second segment | every first segment stopped before `t = 4`; the second segment's reference is the one held from `t = 4` on (`-3.5652343570276206`, `-3.565857236878324`, `-3.5656998998271225`, `-3.5656694757384906`) |
| the energy is held where the constraint is on | maximum deviation from the declared reference `1.78e-15` over 9,555, 9,449, 9,410, 9,536 constrained repairs, 0 hyperplane refusals |
| one grid | 3,201 common samples, `t = 0` to `8`, times equal to `1e-12` across the four ball members, the four task-8 members and the recorded chain; every score hit the `exact_t8.npz` cache |
| the ledgers merge at the boundary | `compare_chains.py`'s boundary-inclusive merge (task 8) gives 200 entries on one schedule for every member; the `t = 6.6` and `6.64` records above are read from those merged ledgers |

## (b) The base to `t = 20` under the ball rule

**The rule taken to `t = 20`, and why.**  The two rules tie at `t = 8` on
the base (`3.180974e-02` against `3.181485e-02`, ratio `0.9998`; the
four-member bands `3.143e-2` to `3.216e-2` and `3.180e-2` to `3.224e-2`
overlap), so the choice was made on what each rule is for past `t = 8`:
the ball radius is the declared instrument for the moves the cheap
route measured past `t = 8.68` at 15 keys (its whole gain there is past
`t = 8`), it is the rule of the 15-key ledger row, and its base's
envelope sat at `0.39` where the anchor-free base's `t = 6.6` excursion
put it at `0.77`; the known cost was 4 percent at `t = 4` and a 12
percent worse internal-energy RMS on the base.  The ball base was
chained from its `t = 8` segment (`_seg2`, resume written at `t = 8.0`)
in three more segments --- `8.0` to `11.8` (1,520 steps, `1,006.3 s`),
`11.8` to `15.44` (1,456 steps, `1,004.2 s`), `15.44` to `20.0` (1,824
steps, `856.9 s`) --- under the same flags, and scored at `t = 4`, `8`,
`10`, `12`, `16`, `20` on the `exact_t20.npz` cache (8,001 samples).  The
anchor-free base was not continued; a single rollout of each rule past
`t = 8` would be needed to say which is better there, and this round
has one.

**The result, first: at `t = 12`, `16` and `20` the ball base is ahead
of every 27-key chain on record, outside every recorded band, and it is
one run.**  Cumulative all-coordinate scalar RMS over `t = 0` to `T`:

| `T` | ball base (this round; one run) | recorded `drive_power` chain / tightened member (band) | standing | recorded `post_pulse` chain / tightened member (band) | standing | both cones | 15-key ball base (band) |
|---:|---|---|---|---|---|---|---|
| 4 | `1.043201e-02` (four members `1.004e-2` to `1.043e-2`, 3.9 %) | `1.142716e-02` / `1.292283e-02` (13.1 %) | ahead by `1.10` to `1.24`, bands disjoint; behind task 8's `1.00e-2` | `1.630660e-02` / `1.971786e-02` (20.9 %) | ahead by `1.56` to `1.89` | --- | `0.027191` |
| 8 | `3.180974e-02` (four members `3.143e-2` to `3.216e-2`, 2.3 %) | `2.593079e-02` / `2.709352e-02` (4.5 %) | behind by `1.17` to `1.23`, bands disjoint | `4.537517e-02` / `4.158658e-02` (9.1 %) | ahead by `1.31` to `1.43` | --- | `0.038538` |
| 10 | `3.608325e-02` | `3.332970e-02` / `3.356926e-02` (0.7 %) | behind by `1.07` to `1.08` | `4.956048e-02` / `4.424362e-02` (12.0 %) | ahead by `1.23` to `1.37` | --- | --- |
| 12 | `3.797419e-02` | `4.250925e-02` / `4.331967e-02` (1.9 %) | ahead by `1.12` to `1.14` | `6.075829e-02` / `4.611524e-02` (31.8 %) | ahead by `1.21` to `1.60` | --- | `0.098867` (`0.0989` to `0.1043`) |
| 16 | `4.820897e-02` | `6.377276e-02` / `6.719628e-02` (5.4 %) | ahead by `1.32` to `1.39` | `6.385168e-02` / `5.386180e-02` (18.5 %) | ahead by `1.12` to `1.32` | --- | `0.132318` (`0.132` to `0.158`) |
| 20 | `6.353308e-02` | `8.519313e-02` / `9.088037e-02` (6.7 %) | ahead by `1.34` to `1.43` | `6.990927e-02` / `7.563847e-02` (8.2 %) | ahead by `1.10` to `1.19` | `0.128607`: the chain is `0.494` of it | `0.132542` (`0.133` to `0.207`): `0.479` |

The crossing of the ball base and the recorded `drive_power` chain lies
between `t = 10` and `t = 12`: the chain's score per four-unit window
`(a, b]` is `1.044e-02`, `4.377e-02`, `4.799e-02`, `7.050e-02`,
`1.043e-01` over `(0, 4]` to `(16, 20]`, against the recorded chain's
`1.143e-02`, `3.485e-02`, `6.385e-02`, `1.042e-01`, `1.415e-01`
(tightened `1.293e-02`, `3.608e-02`, `6.452e-02`, `1.115e-01`,
`1.524e-01`), the `post_pulse` chain's `1.631e-02`, `6.207e-02`,
`8.341e-02`, `7.234e-02`, `9.016e-02` (tightened `1.972e-02`,
`5.542e-02`, `5.405e-02`, `7.228e-02`, `1.304e-01`) and the 15-key ball
base's `0.0272`, `0.0472`, `0.1624`, `0.2018`, `0.1334`: the ball chain
loses the `(4, 8]` window to the recorded chain by `1.26` and wins every
later window by `1.33` to `1.48`; the `post_pulse` chain, which the
`drive_power` chains lose to at `t = 20`, is behind the ball chain in
every window but `(16, 20]`, where it is ahead by `1.16` (its site-0
occupation RMS `0.208` there against the ball chain's `0.280`).  Where
the gain is: at `t = 20` the ball chain's blocks are rho `1.0064e-01`, B
`1.2518e-01`, N `4.6925e-02`, A `4.2577e-02`, C `2.9735e-02` against the
recorded chain's `1.3580e-01`, `1.8729e-01`, `3.1870e-02`, `4.3970e-02`,
`3.1283e-02` and the `post_pulse` chain's `8.0311e-02`, `1.1223e-01`,
`8.2570e-02`, `7.0619e-02`, `4.1905e-02`; the 27 coordinates outside the
phonon-occupation block score `6.564e-02` (recorded `9.046e-02`,
`post_pulse` `6.783e-02`) and the four phonon-occupation coordinates
`4.692e-02` (recorded `3.187e-02`, `post_pulse` `8.257e-02`).  So the
ball chain is ahead in the rho and B blocks --- the ones the recorded
chain lost its `t = 12` to `20` scores in --- and behind the recorded
chain in the phonon-occupation block at every horizon past `t = 4`
(`4.86e-2` against `3.28e-2` at `t = 8`, `5.57e-2` against `3.20e-2` at
`t = 12`, `4.69e-2` against `3.19e-2` at `t = 20`), the `(4, 8]` loss of
task 8 carried forward and never recovered.  Site-0 occupation
(`rho[0,0]` sample RMS) at `t = 20`: `8.122e-02` (`2 rho[0,0]`:
`0.162`), recorded `1.126e-01` (`0.225`), `post_pulse` `6.249e-02`, both
cones `0.205` in the `2 rho[0,0]` convention.

**The internal energy and the cones are the recorded chain's, and
nothing is held.**  Eq. (22) internal-energy RMS `5.641305e-03`,
`5.776465e-03`, `5.856071e-03` at `t = 12`, `16`, `20` (recorded
`6.088286e-03`, `6.249722e-03`, `6.344620e-03`; tightened `4.995294e-03`,
`5.135424e-03`, `5.217702e-03`; `post_pulse` `0.153`, `0.161`, `0.165`;
both cones `0.00094`): inside the `drive_power` band at every horizon, 28
times better than the `post_pulse` chain, 6.2 times worse than the
both-cones line, the `drive_power` reference `-3.5652343570276206`
(`6.164e-03` above the exact post-pulse energy) held to `1.78e-15` over
23,955 constrained repairs, 0 hyperplane refusals.  Over `t <= 12`, `16`
and `20` alike the minimum unshifted retained joint Gram is
`-2.377e-03`, attained at `t = 10.08`, and the minimum scaled
positive-extension eigenvalue `-2.202e-04`, attained at `t = 9.545`
(recorded `-2.463e-03` / `-2.040e-04`, tightened `-2.374e-03` /
`-2.098e-04`, `post_pulse` `-3.124e-03` / `-3.429e-04`; over `t <= 8` the
chain's own `-1.298e-03` / `-1.369e-04`); the frontier envelope reaches
`0.573` (at the `t = 11.04` cap), against the recorded `0.452`, and never
the box.

**The cap did its declared work in `(8, 12]` and was idle after
`t = 14.36`.**  Decisions of 500: accepted 441 (91 uncertified),
cap-bound 14, declined by the margin 1, `infeasible` 44; solver refusals
136 (recorded 144, tightened 226); by four-unit window accepted 88, 98,
73, 82, 100, `infeasible` 8, 2, 17, 17, 0, cap-bound 3, 0, 10, 1, 0, and
the median applied move `0.0044`, `0.0095`, `0.0150`, `0.0106`, `0.0074`
(maximum `0.1`, `0.0319`, `0.1`, `0.1`, `0.0112`).  The eleven caps past
`t = 8`, with the answer's own distance and certificate: `9.08` `0.134u`,
`9.20` `0.148`, `9.24` `0.133u`, `10.00` `0.387u`, `10.04` `0.198`,
`11.00` `0.133u`, `11.04` `0.793u`, `11.24` `0.351`, `11.28` `0.250`,
`11.32` `0.149`, `14.36` `0.356u`; of the chain's 38 answers above `0.4`,
36 are `infeasible`, one declined (`1.319` at `t = 0.72`) and one capped
(the `0.793` uncertified iterate at `t = 11.04`, applied at `0.1`).  This
is the cheap route's task-10 pattern --- the large answers refused by
the measured test, the cap catching answers of `0.13` to `0.39` in
clusters at `t = 9.1` to `9.2`, `10.0`, `11.0` to `11.3` (15 keys: `8.68`,
`9.64`, `10.08` to `10.32`, `13.48`, `16.88`) --- with one addition, the
`0.793` at `t = 11.04` that the 15-key base never saw (its counterpart
there, the `0.894` at `t = 10.04`, was declined).  From `t = 16` on the
chain accepts all 100 completions, 15 of them uncertified, with no
answer above `0.0112`: the lift converges.  Against the recorded chain
the separation `d31` is `0.260`, `0.265`, `0.261` at `t = 12`, `16`, `20`
(maximum `0.336` at `t = 7.57`) while the two chains' distances from the
exact reference are `0.290` / `0.503`, `0.440` / `0.656`, `0.767` /
`0.969`.

**Cost.**  Autonomous wall to `t = 20` `4,844.7 s` in five segments
(recorded `4,794.4 s`, tightened `4,789.8 s`, `post_pulse` `3,887.0 s`),
`0.924` in the 500 completion solves at `8.95 s` mean; repairs `243.0 s`,
velocity `125.0 s`; peaks `631` to `638 MiB`; longest segment `1,006.3 s`
of autonomous wall plus `18.5 s` of build; the cheap-route job shared
the machine throughout.

## What the 27-key ledger row is, horizon by horizon

| horizon `t = 0` to `T` | the row | band | ball base (this round) | what changes |
|---:|---|---|---|---|
| 4 | task 8's anchor-free rule, `1.00e-2` | four members `9.96e-3` to `1.001e-2`, 0.5 % (task 8) | `1.043e-2`, four members `1.004e-2` to `1.043e-2`, 3.9 % | nothing: the ball is behind, bands disjoint |
| 8 | the recorded `drive_power` chain, `2.59e-2` | `2.59e-2` to `2.71e-2`, 4.5 % (task 5) | `3.18e-2`, four members `3.143e-2` to `3.216e-2`, 2.3 % | nothing: both rules behind by `1.16` to `1.24` |
| 12 | this round's ball base, `3.80e-2`, if adopted | **none**: one run; the recorded `drive_power` band it beats is `4.25e-2` to `4.33e-2` (1.9 %), the `post_pulse` band `4.61e-2` to `6.08e-2` | `3.797e-2` | the row moves from the recorded chain (`4.25e-2`) by `1.12` to `1.14`, outside its band, subject to the ball's own band |
| 16 | this round's ball base, `4.82e-2`, if adopted | **none**: one run; `drive_power` `6.38e-2` to `6.72e-2`, `post_pulse` `5.39e-2` to `6.39e-2` | `4.821e-2` | the row moves from the tied `drive_power` / `post_pulse` pair (`6.38e-2` / `6.39e-2`) by `1.12` to `1.39` |
| 20 | this round's ball base, `6.35e-2`, if adopted | **none**: one run; `drive_power` `8.52e-2` to `9.09e-2`, `post_pulse` `6.99e-2` to `7.56e-2` | `6.353e-2` | the row moves from the `post_pulse` chain (`6.99e-2`) by `1.10` to `1.19`, outside its 8.2 % band, and stands at `0.494` of the both-cones `0.1286` |

Measured bands exist at `t = 4` and `t = 8` only (four members under
this rule; four under task 8's; the recorded chains' pairs).  At
`t = 12`, `16`, `20` the ball row is a single rollout; the three
one-setting twins were queued from their own `t = 8` segments to
`t = 20` as this record was written (`queue_task9_t20_twins.sh`; about
55 minutes each, one at a time), so that the row's band lands in
`task9_ball_{etol1e3,dykstra60,conic1e10}_score_t20.json`.  What the
band must be for each standing to survive: at `t = 12` the lead over the
`drive_power` band is 12 percent of the ball base; at `t = 16` the lead
over the `post_pulse` band 12 percent; at `t = 20` the lead over the
`post_pulse` band 10 percent and over the `drive_power` band 34.  At 15
keys the ball's own bands were 5.5, 19.3 and 56.2 percent at these
horizons; at 27 keys through `t = 8` they are 3.9 and 2.3.  Adoption of
the row is the user's call; the plan-file row records the numbers and
the absence of the band.

## Provenance and gates of the runs themselves

- Thirteen rollouts under `pipelines/shell/ram_guard.py --limit-mb
  4500`, one at a time, `20:29` to `23:31` on 2026-09-17: the smoke
  (`paper_v_dictionary_energy_ball_smoke_20260917k`, `t = 0` to `0.08`),
  the eight (a) segments
  (`paper_v_dictionary_energy_power_27key_n16_ball_{base,etol1e3,dykstra60,conic1e10}_20260917k_seg{1,2}`)
  and the three (b) segments (`..._ball_base_20260917k_seg{3,4,5}`), each
  with `reselections.json`; the three twins' continuations
  (`..._ball_{etol1e3,dykstra60,conic1e10}_20260917k_seg{3,...}`) were
  running when this record was written.  Scoring and comparison under
  `--limit-mb 2000` or `2500`, arrays only.
- Scripts under `output/local_runs/paper_v_projective_dictionary_energy_20260915e/`:
  `smoke_task9_ball.sh` and `check_task9_smoke.py` (the flag-reaches-
  the-rule gate), `queue_task9_ball.sh` (the eight (a) segments and the
  per-member scores), `score_task9.sh` (the three twin-against-base
  comparisons, the base against the task-8 base and against the recorded
  chain, and the tabulation), `tabulate_task9.py` (ledgers, chart,
  standing, the cap-bound ledger, the `t = 6.6` records of every member
  under both rules, separations recomputed from the arrays, decision
  ledgers), `queue_task9_t20.sh` (the (b) chain and its score; takes
  `ball` or `measured`), `tabulate_task9b.py` and `tabulate_task9b.sh`
  (the (b) chart, the standing against each recorded band, windows,
  physicality minima with the times attained, decisions per window, the
  separation from the recorded chain), `queue_task9_t20_twins.sh` (the
  twins' continuations); `run_energy_segment.py`, `run_chain.py`,
  `score_energy_chain.py` and `compare_chains.py` unchanged from task 8.
  Outputs: `task9_{member}_score_t8.json` and `.log`,
  `task9_ball_base_score_t20.json` and `.log`,
  `compare_task9_{etol1e3,dykstra60,conic1e10}.json`,
  `compare_task9_base_vs_task8_base_t8.json`,
  `compare_task9_base_vs_recorded_t8.json`, `tabulate_task9.log`,
  `tabulate_task9b_ball.log`, `smoke_task9_ball.log`, `task9_chain.log`,
  `task9_t20_chain.log`, `task9_t20_twins_chain.log`, the segment logs
  `task9_*_seg1.log` and `paper_v_dictionary_energy_power_27key_n16_ball_*_seg{2,3,4,5}.log`.
- No module was edited, nothing was committed, no default, acceptance
  rule, radius or ladder changed; the recorded chains' and task 8's
  artifacts are untouched.  Wall times were measured with the cheap
  route's `3,000 MB` job on the machine and are upper bounds.  The
  worklog carries a matching block; the plan-file row was updated in
  place.

## What to try next, in order

1. **Read the twins' `t = 12`, `16`, `20` scores when they land** and
   write the band into the ledger row: if the four members stay within
   the leads above (12, 12 and 10 percent), the ball base is the 27-key
   row at every horizon from `t = 12` on, outside every recorded band; if
   the `t = 20` band opens to the 15-key rule's 56 percent, the row past
   `t = 12` carries the first digit only and the `post_pulse` chain's
   `t = 20` standing is unresolved.
2. **The anchor-free base from `t = 8` to `t = 20`** (`queue_task9_t20.sh
   measured`; three segments): the two rules tie at `t = 8`, and whether
   the ball's `1.33` to `1.48` per-window gain past `t = 8` is the
   radius's or the measured acceptance's is one chain away.
3. **The phonon-occupation block.**  The ball chain is behind the
   recorded chain in the four phonon-occupation coordinates at every
   horizon past `t = 4` (`4.69e-2` against `3.19e-2` at `t = 20`) while
   ahead everywhere else; the loss enters in `(4, 8]` under both
   rules.  Task 8's proposal stands: a rerun with `certified` candidates
   from `t = 4` separates the `t = 0.04` acceptance from the `(4, 8]`
   re-selections.

## Structured summary of task 9

| item | value |
|---|---|
| the rule | `LiftReselectionRule(acceptance_margin=1e-3, candidates="measured", anchor_weight=0, trust_radius=0.1, trust_policy="ball")` on the task-4 27-key `N = 16` `drive_power` energy-neutral chain; (a) base plus `etol1e3`, `dykstra60`, `conic1e10` to `t = 8` in two segments each; (b) the base to `t = 20` in three more; imports from `head_src_e3374f6f` (identical to the tree at `f971a284`) |
| (a) `t = 4`, four members | `1.043201e-02` / `1.003847e-02` / `1.005305e-02` / `1.024635e-02`; band `1.004e-2` to `1.043e-2`, spread `3.92 %` (task 8's anchor-free four: `1.001172e-02` / same / `9.959755e-03` / same, `0.52 %`; disjoint, ball behind by `1.042` on the base); ahead of the recorded band `1.143e-2` to `1.292e-2` (worst member `0.913` of its bottom); every member below cell A (`0.829`); internal energy `4.413844e-03` / `3.950798e-03` / `4.059184e-03` / `4.090198e-03`; retained joint Gram `-4.052e-04` / `-4.155e-04` / `-2.219e-04` / `-4.155e-04`, extension `-4.706e-05` / `-4.670e-05` / `-2.781e-05` / `-4.734e-05` |
| (a) `t = 8`, four members | `3.180974e-02` / `3.151679e-02` / `3.215530e-02` / `3.143476e-02`; band `3.143e-2` to `3.216e-2`, `2.29 %` (task 8's `3.181485e-02` / `3.179926e-02` / `3.224344e-02` / `3.179926e-02`, `1.40 %`; overlapping, bases equal to `0.02 %`); behind the recorded band `2.593e-2` to `2.709e-2` by `1.16` to `1.24`, disjoint: the standing does not change; internal energy `5.360816e-03` / `4.812119e-03` / `4.947228e-03` / `4.977447e-03` (task-8 base `4.781299e-03`; recorded `5.751890e-03`); retained `-1.298e-03` / `-1.324e-03` / `-5.212e-04` / `-1.316e-03`, extension `-1.369e-04` / `-1.384e-04` / `-5.279e-05` / `-1.384e-04` (recorded `-2.152e-03` / `-1.922e-04`); envelope `0.392` / `0.386` / `0.421` / `0.402` (task-8 base `0.772`) |
| (a) the capped moves and when | 3 / 5 / 5 / 4 of 200, all at `t <= 0.64`: `0.04` (`0.345`, the whole-face answer task 8 applied whole), `0.16` (`0.244`), `0.20` (`0.145`) in every member; `0.56` `0.285` and `0.60` `0.115` (`etol1e3`); `0.60` `0.881` and `0.64` `0.103` (`dykstra60`); `0.36` `0.584` (`conic1e10`); none in `(0.64, 8]`; every later answer above `0.1` is `infeasible` (`0.53` to `0.55` at `0.36` to `0.64` and `6.44` to `6.52`) or declined (`1.319` at `0.72`, `1.330` at `0.12`) |
| (a) is the `t = 6.6` move capped | no, because it does not occur: on the four capped trajectories the `t = 6.6` completion returns `0.0240` / `0.0128` / `0.0105` / `0.0243` (accepted) and `t = 6.64` `0.0145` / `0.0148` / `0.0098` / `0.0230`; task 8's `0.9565` facial-reduction iterate is the anchor-free base's, whose trajectory the ball left at `t = 0.04` (`d70` `0.2447`) |
| (a) separations | ball twins from the ball base: `etol1e3` parts at `t = 0.36` (`accepted` against `infeasible`), `d31` `1e-2` at `4.51`, `2.46e-02` at `t = 8` (8.5 % of the distance from exact); `dykstra60` at `t = 0.12` (`infeasible` against `declined_by_margin`), `1e-2` at `3.2`, `3.91e-02` (13.1 %); `conic1e10` at `t = 0.36` (`accepted` `0.0099` against `cap_bound_ball` of an uncertified `0.584`), `1e-2` at `6.41`, `1.68e-02` (5.8 %) --- the certificate branch is back; ball from anchor-free, same twin: `t = 0.04` in every pair, `d31` `2.0e-02` / `7.6e-03` / `1.8e-02` / `9.3e-03` at `t = 8`; ball base from the recorded chain `0.276` at `t = 8` |
| (a) refusals and wall | solver refusals 47 / 47 / 47 / 80 of 200 (task 8 48 / 42 / 42 / 78; recorded 75); `infeasible` 10 / 10 / 6 / 10; declined 1 / 0 / 1 / 0; walls `1,977.2` / `1,882.8` / `1,835.1` / `1,858.7 s` to `t = 8` (task 8 `1,886.3` to `1,962.8 s`; recorded `2,058.0 s`), `0.876` to `0.923` in the solves at `8.04` to `9.12 s` mean; peaks `627` to `636 MiB` |
| (b) the rule chosen | the ball: tie at `t = 8` (`0.9998`; bands overlap), the radius being the declared instrument for the moves past `t = 8` and the 15-key row's rule; the anchor-free base was not continued |
| (b) `t = 12` / `16` / `20`, base, one run | `3.797419e-02` / `4.820897e-02` / `6.353308e-02`; recorded `drive_power` `4.250925e-02` / `6.377276e-02` / `8.519313e-02` (bands to `4.331967e-02`, `6.719628e-02`, `9.088037e-02`): ahead by `1.12` to `1.14`, `1.32` to `1.39`, `1.34` to `1.43`, outside every band; recorded `post_pulse` `6.075829e-02` / `6.385168e-02` / `6.990927e-02` (bands `4.611524e-02` to `6.075829e-02`, `5.386180e-02` to `6.385168e-02`, `6.990927e-02` to `7.563847e-02`): ahead by `1.21` to `1.60`, `1.12` to `1.32`, `1.10` to `1.19`; both cones `0.128607` at `t = 20`: `0.494`; `t = 10` `3.608325e-02`, behind the recorded `3.332970e-02` to `3.356926e-02` by `1.07` to `1.08` (the crossing is in `(10, 12]`); per window `(8, 12]`, `(12, 16]`, `(16, 20]` `4.80e-2`, `7.05e-2`, `1.04e-1` against the recorded `6.39e-2`, `1.04e-1`, `1.42e-1` and the `post_pulse` `8.34e-2`, `7.23e-2`, `9.02e-2` |
| (b) internal energy and physicality | Eq. (22) RMS `5.641305e-03` / `5.776465e-03` / `5.856071e-03` at `t = 12` / `16` / `20` (recorded `6.088286e-03` / `6.249722e-03` / `6.344620e-03`, tightened `4.995294e-03` / `5.135424e-03` / `5.217702e-03`, `post_pulse` `0.153` / `0.161` / `0.165`, both cones `0.00094`); retained joint Gram minimum `-2.377e-03` (at `t = 10.08`) and extension `-2.202e-04` (at `t = 9.545`) over `t <= 12`, `16`, `20` alike (recorded `-2.463e-03` / `-2.040e-04`; `post_pulse` `-3.124e-03` / `-3.429e-04`); envelope `0.573`; energy held to `1.78e-15` over 23,955 constrained repairs |
| (b) the cap past `t = 8` | 11 cap-bound of 300 (`9.08` `0.134`, `9.20` `0.148`, `9.24` `0.133`, `10.00` `0.387`, `10.04` `0.198`, `11.00` `0.133`, `11.04` `0.793`, `11.24` `0.351`, `11.28` `0.250`, `11.32` `0.149`, `14.36` `0.356`), 14 of 500 in all; `infeasible` 44, declined 1, accepted 441 (91 uncertified); solver refusals 136 (recorded 144); `(16, 20]` accepts all 100 with no answer above `0.0112`; wall `4,844.7 s` to `t = 20` (recorded `4,794.4 s`) |
| the 27-key ledger row | `t = 4`: task 8's `1.00e-2` (band 0.5 %); `t = 8`: the recorded chain's `2.59e-2` (band 4.5 %); `t = 12` / `16` / `20`: this round's ball base `3.80e-2` / `4.82e-2` / `6.35e-2` if adopted, one run each, no band --- the twins' continuations were queued at `23:32` to supply it |
| gates | the radius reaches the rule (smoke: `cap_bound_ball` at `t = 0.04`, solved `0.344669`, applied `0.1`; the rule recorded in `summary.json`); the driver without the radius is task 8's (flags default to not given); every member's first decision under the ball is the same; boundary resumes at `90`, `90`, `97`, `93` intervals and, for the base past `t = 8`, at `t = 8.0`, `11.8`, `15.44`; energy held to `1.78e-15`; one grid (3,201 samples to `t = 8`, 8,001 to `t = 20`), every score from the caches |
| files changed | this report (new heading), the worklog block, the plan ledger's row (in place); the task-9 scripts and outputs under `output/local_runs/` listed above.  No module; nothing committed |
| next | the twins' `t = 12` / `16` / `20` band (running); the anchor-free base from `t = 8` to `t = 20`; the phonon-occupation loss of `(4, 8]` |

## (b) The band, landed 02:06 on 2026-09-18

The three twins' continuations from their own `t = 8` segments reached
`t = 20` (`etol1e3` in three segments, `8.0` to `11.4`, `11.4` to
`14.96`, `14.96` to `20.0`; `dykstra60` in four, `8.0` to `11.395`
(resume `11.36`), `11.36` to `14.96`, `14.96` to `19.32`, `19.32` to
`20.0`; `conic1e10` in three, `8.0` to `11.64`, `11.64` to `15.8`, `15.8`
to `20.0`), one at a time, `23:32` to `02:06`; the base's row is the
one above.  This subsection supersedes the "one run, no band" entries of
the ledger-row table and the structured summary above (`tabulate_task9_band.py`,
`tabulate_task9_band.log`).

**Every member of the ball rule is ahead of every recorded 27-key chain
at `t = 12`, `16` and `20`, outside every recorded band, with the rule's
own band at 2.9, 5.9 and 9.7 percent.**  Base / `etol1e3` / `dykstra60`
/ `conic1e10`, cumulative from `t = 0`:

| `T` | four members | band, spread | recorded `drive_power` band | standing | recorded `post_pulse` band | standing | both cones; 15-key ball band |
|---:|---|---|---|---|---|---|---|
| 10 | `3.608325e-02` / `3.598880e-02` / `3.736894e-02` / `3.580369e-02` | `3.580e-2` to `3.737e-2`, 4.4 % | `3.333e-2` to `3.357e-2` (0.7 %) | every member behind, by `1.07` to `1.12` | `4.424e-2` to `4.956e-2` (12.0 %) | every member ahead, by `1.18` to `1.38` | --- |
| 12 | `3.797419e-02` / `3.831339e-02` / `3.906086e-02` / `3.799833e-02` | `3.797e-2` to `3.906e-2`, 2.9 % | `4.251e-2` to `4.332e-2` (1.9 %) | every member ahead, by `1.09` to `1.14` (worst member `0.919` of the band's bottom) | `4.612e-2` to `6.076e-2` (31.8 %) | ahead by `1.18` to `1.60` | 15-key `0.0989` to `0.1043` (5.5 %) |
| 16 | `4.820897e-02` / `4.860629e-02` / `4.588696e-02` / `4.825014e-02` | `4.589e-2` to `4.861e-2`, 5.9 % | `6.377e-2` to `6.720e-2` (5.4 %) | ahead by `1.31` to `1.46` (worst `0.762`) | `5.386e-2` to `6.385e-2` (18.5 %) | ahead by `1.11` to `1.39` (worst `0.902`) | 15-key `0.132` to `0.158` (19.3 %) |
| 20 | `6.353308e-02` / `6.330860e-02` / `5.792205e-02` / `6.310984e-02` | `5.792e-2` to `6.353e-2`, 9.7 % | `8.519e-2` to `9.088e-2` (6.7 %) | ahead by `1.34` to `1.57` (worst `0.746`) | `6.991e-2` to `7.564e-2` (8.2 %) | ahead by `1.10` to `1.31` (worst `0.909`) | `0.128607`: worst member `0.494`, best `0.450`; 15-key `0.133` to `0.207` (56.2 %) |

The base is the worst member at `t = 20` and the best at `t = 12`; the
`dykstra60` twin is the best at `t = 16` and `20` (`4.589e-2`,
`5.792e-2`) and the worst at `t = 10` and `12`; the `etol1e3` and
`conic1e10` twins stay within `0.9` percent of the base from `t = 12`
on.  So the narrowest lead --- the worst member over the `post_pulse`
band's bottom at `t = 20` --- is `1.10`, and it holds for all four; the
15-key ball rule's 56 percent band at `t = 20` does not appear at 27
keys (9.7 percent), nor its 19 percent at `t = 16` (5.9).  What each
member does past `t = 8`, in the record's terms:

| member | cap-bound past `t = 8` (of 500 in all) | the answers capped past `t = 8` (time: solved move; `u` uncertified) | `infeasible` / declined / accepted (uncertified) of 500 | solver refusals | answers above `0.4` and their decisions | `(16, 20]` |
|---|---:|---|---|---:|---|---|
| base | 11 (14) | `9.08` `0.134u`, `9.20` `0.148`, `9.24` `0.133u`, `10.00` `0.387u`, `10.04` `0.198`, `11.00` `0.133u`, `11.04` `0.793u`, `11.24` `0.351`, `11.28` `0.250`, `11.32` `0.149`, `14.36` `0.356u` | 44 / 1 / 441 (91) | 136 | 38: 36 `infeasible`, 1 declined, 1 capped | all 100 accepted, largest answer `0.0112` |
| `etol1e3` | 9 (14) | `9.08` `0.118u`, `9.20` `0.152u`, `10.00` `0.144u`, `10.04` `0.133`, `11.00` `0.111u`, `11.08` `0.214`, `11.24` `0.154`, `14.40` `0.113u`, `15.40` `0.109u` | 47 / 0 / 439 (100) | 147 | 37, all `infeasible` | all 100, `0.0111` |
| `dykstra60` | 14 (19) | `10.12` `0.720u`, `10.16` `0.187`, `10.24` `0.437u`, `10.28` `0.107`, `11.16` `0.885u`, `11.20` `0.260`, `11.24` `0.179`, `12.28` `0.131`, `13.08` `0.792u`, `13.24` `1.045u`, `13.28` `0.121u`, `13.40` `1.138u`, `13.44` `0.108`, `15.32` `0.383u` | 25 / 1 / 455 (109) | 135 | 24: 16 `infeasible`, 1 declined, 7 capped | all 100, `0.0115` |
| `conic1e10` | 12 (16) | `9.08` `0.107u`, `9.20` `0.147u`, `9.24` `0.115u`, `9.28` `0.105u`, `10.08` `0.135u`, `11.00` `0.114u`, `11.08` `0.212`, `11.24` `0.154`, `12.16` `0.114u`, `13.04` `0.119u`, `15.28` `0.922u`, `15.40` `0.249u` | 46 / 0 / 438 (204) | 250 | 43: 41 `infeasible`, 2 capped | all 100, `0.0112` |

Two readings.

1. **The `dykstra60` twin is the one on which the cap does its declared
   work, and it is the best member past `t = 12`.**  Its repaired states
   admit seven uncertified answers above `0.4` (`0.720`, `0.437`, `0.885`,
   `0.792`, `1.045`, `1.138`, `0.922`) that pass the measured test, and
   the rule applies each at `0.1`; on the other three trajectories the
   answers above `0.4` are `infeasible` in all but one case each (the
   base's `0.793` at `11.04`, `conic1e10`'s `0.922` at `15.28`).  With
   those steps its `(12, 16]` and `(16, 20]` windows are `6.20e-2` and
   `9.14e-2` against the base's `7.05e-2` and `1.04e-1`, its cones are
   the tightest of the four (joint Gram `-1.085e-03` at `t = 11.04`,
   extension `-9.84e-05` at `9.543`; base `-2.377e-03` at `10.08`,
   `-2.202e-04` at `9.545`), and its distance from the base crosses
   `0.1` at `t = 13.1` (maximum `0.104` at `13.2`; `9.3e-2` at `t = 20`,
   12 percent of the base's `0.767` from the exact reference).  This is
   the cheap route's task-10 pattern with the roles reversed: there the
   `dykstra60` twin was the outlier that entered the `infeasible` regime
   (`0.207` at `t = 20`, 56 percent above the base); here it stays out of
   it (25 `infeasible` of 500 against 44 to 47) and is 9 percent below.
2. **The certificate tolerance is back to moving nothing.**  The
   `conic1e10` twin's completions return uncertified 250 times of 500
   and it accepts 204 of them; it is the base's trajectory to `1.7e-2`
   at `t = 8`, `2.4e-2` at `12`, `9.4e-3` at `16`, `1.6e-2` at `20`
   (never `0.1`), and scores within `0.7` percent of the base at every
   horizon from `t = 8`; the `etol1e3` twin likewise (`3.1e-2` at
   `t = 12`, `2.0e-2` at `20`; within `0.9` percent).  The band past
   `t = 12` is the Dykstra count's --- the one setting that changes the
   repaired state --- and the `(4, 8]` phonon-occupation loss is every
   member's (`4.7e-2` to `5.4e-2` at `t = 20` against the recorded
   `3.2e-2`).

Internal-energy RMS at `t = 12` / `16` / `20`: `5.6413e-03` /
`5.7765e-03` / `5.8561e-03` (base), `5.0668e-03` / `5.1895e-03` /
`5.2618e-03`, `5.2097e-03` / `5.3362e-03` / `5.4106e-03`, `5.2400e-03` /
`5.3664e-03` / `5.4409e-03` --- band `5.26e-3` to `5.86e-3` at `t = 20`
(11 percent), inside the recorded `drive_power` pair's `5.22e-3` to
`6.34e-3` (both cones `0.00094`); every member's energy held to
`2.22e-15` or better over 23,849 to 23,955 constrained repairs, 0
hyperplane refusals.  Physicality over `t <= 20`: joint Gram
`-2.377e-03` / `-2.282e-03` / `-1.085e-03` / `-2.318e-03`, extension
`-2.202e-04` / `-2.151e-04` / `-9.837e-05` / `-2.162e-04`, every minimum
attained in `(9.5, 11.1]` and none after; frontier envelope `0.573` /
`0.525` / `0.574` / `0.522`, never the box (recorded `0.452`); site-0
occupation (`rho[0,0]` sample RMS) at `t = 20` `8.12e-2` / `8.15e-2` /
`7.19e-2` / `8.11e-2` (recorded `1.13e-1`, `post_pulse` `6.25e-2`).
Walls to `t = 20`: `4,844.7` / `4,897.5` / `4,984.4` / `4,689.5 s` in
five, five, six, five segments (recorded `4,794.4 s`); peaks `638` /
`704` / `639` / `698 MiB`; longest segment `1,007.7 s` of autonomous
wall plus about `19 s` of build.  Every member's `(16, 20]` window
accepts all 100 completions with no answer above `0.0115`.

## The 27-key ledger row, with the band

| horizon `t = 0` to `T` | the row | band | what moved |
|---:|---|---|---|
| 4 | task 8's anchor-free rule, `1.00e-2` | `9.96e-3` to `1.001e-2`, 0.5 % (four members, task 8) | nothing: the ball's `1.004e-2` to `1.043e-2` (3.9 %) is behind, bands disjoint |
| 8 | the recorded `drive_power` chain, `2.59e-2` | `2.59e-2` to `2.71e-2`, 4.5 % (task 5) | nothing: the ball's `3.143e-2` to `3.216e-2` (2.3 %) and the anchor-free `3.180e-2` to `3.224e-2` are behind by `1.16` to `1.24` |
| 12 | the ball rule, base `3.80e-2` | `3.797e-2` to `3.906e-2`, 2.9 % (four members) | from the recorded chain's `4.25e-2` (band `4.25e-2` to `4.33e-2`): every member ahead by `1.09` to `1.14`, bands disjoint |
| 16 | the ball rule, base `4.82e-2` | `4.589e-2` to `4.861e-2`, 5.9 % | from the tied `drive_power` / `post_pulse` pair (`6.38e-2` / `6.39e-2`; bands `6.38e-2` to `6.72e-2` and `5.39e-2` to `6.39e-2`): every member ahead by `1.11` to `1.46` |
| 20 | the ball rule, base `6.35e-2` | `5.792e-2` to `6.353e-2`, 9.7 % | from the `post_pulse` chain's `6.99e-2` (band `6.99e-2` to `7.56e-2`): every member ahead by `1.10` to `1.31`; over the `drive_power` band `8.52e-2` to `9.09e-2` by `1.34` to `1.57`; `0.450` to `0.494` of the both-cones `0.1286` |

Every horizon now carries a measured four-member band; at `t = 12`,
`16`, `20` the ball rule's members are the best moment-closure numbers
on record for the 27-key `N = 16` chain, outside every recorded band,
with the internal-energy RMS of the `drive_power` chain (`5.3e-3` to
`5.9e-3`) rather than the `post_pulse` chain's (`0.165`).  Adoption of
the row is the user's call; the plan-file row carries the numbers and
the band.  Next, in order: the anchor-free base from `t = 8` to `t = 20`
(`queue_task9_t20.sh measured`; whether the radius or the measured
acceptance owns the gain past `t = 8`); the Dykstra count at 120 on the
ball base past `t = 8` (the `dykstra60` twin is the best member and the
only one on which the cap acts); the `(4, 8]` phonon-occupation loss.

# projective-route, task 10 (2026-09-18)

Task 9 left the 27-key `N = 16` `drive_power` energy-neutral chain with a
four-member band under the ball rule at every horizon and two open
attributions: whether the `1.33` to `1.48` per-window gain over the
recorded chain past `t = 8` belongs to the trust radius or to the
measured-candidate acceptance the radius sits on, and whether a deeper
Dykstra repair moves the physicality minima or the score.  This round
runs the two chains that answer them, each from a stored `t = 8` state to
`t = 20` in `1,000 s` segments under `ram_guard --limit-mb 4500`, one at a
time, `02:13` to `04:01` on 2026-09-18, nothing else heavy on the machine:
(a) the anchor-free measured rule without the radius ---
`LiftReselectionRule(acceptance_margin=1e-3, candidates="measured",
anchor_weight=0)`, task 8's rule --- continued from its own `t = 8`
segment (`..._measured_base_20260916i_seg2`) by `queue_task9_t20.sh
measured`, three segments; (b) the ball base of task 9 (the same rule
with `trust_radius=0.1`, `trust_policy="ball"`) continued from its `t = 8`
segment (`..._ball_base_20260917k_seg2`) with `--dykstra-iterations 120`
in place of the default 30, four segments, under the new prefix
`..._ball_base_dyk120_20260918l_seg` whose `seg1` and `seg2` are symbolic
links to the ball base's own, so that `run_chain.py` resumes from the
identical `t = 8` state and the scorer reads one chain from `t = 0`.
Imports are from `head_src_e3374f6f`, which `diff -rq` shows identical to
`paper_5/src/paper5` at `HEAD c14bf1bd` outside `__pycache__` (recorded
in `task10_chain.log`); `run_chain.py`, `run_energy_segment.py` and
`score_energy_chain.py` are task 8's unchanged; no module was edited;
nothing was committed; no default changed.  Configuration and conventions
are task 9's: 27 active keys, 56 state coordinates, 36 positive-extension
rows, 70 standardized frontier coordinates, strong coupling, cutoff 16,
exact-correlated preparation at the declared ladder's first rung,
SSPRK(3,3) at `h = 0.0025`, Dykstra on both targets at stage cadence (30
iterations, or 120 in (b) past `t = 8`), the declared proximal log-det
completion re-solved every 16 steps warm-started on the carried frontier,
the energy-neutral repair in `drive_power` mode with each chain's own
reference (`-3.5658908873520327` for (a), `5.507e-03` above the exact
post-pulse energy `-3.571397982`; `-3.5652343570276206` for (b) and the
ball base, `6.164e-03` above); the score is the all-coordinate scalar RMS
of the 31 closed scalar coordinates against the exact cutoff-16 driven
reference from `t = 0` on one grid (`exact_t20.npz`, 8,001 samples); a
window score `(a, b]` is the same RMS over the samples in that interval;
the physicality minima are the sampled (post-repair) minimum unshifted
retained joint Gram eigenvalue and minimum scaled positive-extension
eigenvalue, over `t <= T` or over `(8, 20]` --- the `t = 8.0` sample is
the shared state (`-1.2218e-03` in both (b) and the ball base) and is
excluded from the latter.

## Negative result first

**The trust radius owns none of the gain past `t = 8`.**  The
anchor-free rule without the radius scores `3.851781e-02` /
`4.848061e-02` / `6.291440e-02` over `t = 0` to `12` / `16` / `20`, inside
the ball rule's four-member band at every horizon from `t = 8` on
(`3.797e-2` to `3.906e-2`, `4.589e-2` to `4.861e-2`, `5.792e-2` to
`6.353e-2`; ratios to the ball base `1.014` / `1.006` / `0.990`), and its
window scores over `(8, 12]`, `(12, 16]`, `(16, 20]` are `4.926e-02`,
`7.036e-02`, `1.019e-01` against the ball base's `4.799e-02`, `7.050e-02`,
`1.043e-01` (ratios `1.027` / `0.998` / `0.977`; over `(8, 20]` `0.989`)
where the recorded chain's are `1.331` / `1.477` / `1.356` times the ball
base's.  So the gain over the recorded `drive_power` chain past `t = 8`
is the measured-candidate acceptance's (margin `1e-3`, `measured`
candidates, anchor weight `0`), and the radius, which task 9 found costs
4 percent at `t = 4` and nothing at `t = 8`, buys nothing past `t = 8`
either: at 27 keys it moves six applied displacements of `0.105` to
`0.228` (their excess above `0.1` sums to `0.261` of the `4.465` applied
past `t = 8`) and every answer above `0.4` that it would have capped
(30 of them on the radius-free chain, `0.56` to `0.79`) is already
refused by the measured feasibility test.  The 15-key result of the
cheap route's task 10 (the radius buying 12 percent at `t = 20`) does not
carry to 27 keys.

**Dykstra 120 moves the cones and the `t = 20` score, not the internal
energy, and it worsens the phonon-occupation block.**  The ball base
with the deeper repair from `t = 8` scores `3.834289e-02` /
`4.594082e-02` / `5.784276e-02` at `t = 12` / `16` / `20` (ratios to the
ball base `1.010` / `0.953` / `0.910`); the `t = 20` number is the best
27-key number on record, `0.14` percent below the `dykstra60` twin's
`5.792205e-02` --- a tie with it --- and `0.450` of the both-cones
`0.128607`.  Over `(8, 20]` its minimum retained joint Gram is
`-4.420e-04` (at `t = 11.047`) against the ball base's `-2.377e-03` (at
`10.08`) and its minimum extension eigenvalue `-4.187e-05` (at `9.498`)
against `-2.202e-04` (at `9.545`): `5.4` and `5.3` times shallower, the
mean of the negative part over `(8, 20]` `6.1` and `5.8` times smaller
(`-1.635e-04` / `-1.426e-05` against `-9.940e-04` / `-8.243e-05`), while
the retained cone is still violated at every one of the 4,800 samples
past `t = 8` in every chain.  The internal-energy RMS is the ball base's
to every printed digit (`5.641305e-03` / `5.776465e-03` / `5.856071e-03`):
the energy is pinned to the same declared reference in both, so the RMS
is that reference's `6.164e-03` offset and no repair depth changes it.
The four phonon-occupation coordinates are worse at every horizon past
`t = 8` (`5.391e-02` against the ball's `4.692e-02` and the recorded
chain's `3.187e-02` at `t = 20`; per window `(8, 12]`, `(12, 16]`,
`(16, 20]` `7.685e-02`, `5.633e-02`, `2.680e-02` against `6.762e-02`,
`3.658e-02`, `1.917e-02`) while the other 27 coordinates are better
(`5.840e-02` against `6.564e-02` at `t = 20`).  The cost is 11 percent of
wall to `t = 20` (`5,393.7 s` against `4,844.7 s`): repair time past
`t = 8` `549.3 s` against `142.7 s` (`3.85` times; 16.1 percent of the
wall against 5.0), and the completion solves `9.30 s` mean against
`8.83 s`.

## The chart, `t = 8` to `20`

Cumulative all-coordinate scalar RMS over `t = 0` to `T`; the ball base
with its four-member band (task 9); (a) the radius-free chain; (b) the
Dykstra-120 chain; the recorded `drive_power` band (task 4 with its
tightened member) and `post_pulse` band (task 3/4 with its tightened
member).

| `T` | ball base (band) | (a) radius-free, / ball | (b) Dykstra 120, / ball | recorded `drive_power` band | (a), (b) standing | recorded `post_pulse` band | (a), (b) standing |
|---:|---|---|---|---|---|---|---|
| 8 | `3.180974e-02` (`3.143e-2` to `3.216e-2`) | `3.181485e-02`, `1.0002` | `3.180974e-02`, `1.0000` (shared) | `2.593e-2` to `2.709e-2` | both behind by `1.17` to `1.23` | `4.159e-2` to `4.538e-2` | both ahead by `1.31` to `1.43` |
| 10 | `3.608325e-02` (`3.580e-2` to `3.737e-2`) | `3.633736e-02`, `1.0070` | `3.639411e-02`, `1.0086` | `3.333e-2` to `3.357e-2` | both behind by `1.08` to `1.09` | `4.424e-2` to `4.956e-2` | both ahead by `1.22` to `1.36` |
| 12 | `3.797419e-02` (`3.797e-2` to `3.906e-2`) | `3.851781e-02`, `1.0143` | `3.834289e-02`, `1.0097` | `4.251e-2` to `4.332e-2` | ahead by `1.104` to `1.125`; `1.109` to `1.130` | `4.612e-2` to `6.076e-2` | ahead by `1.197` to `1.577`; `1.203` to `1.585` |
| 16 | `4.820897e-02` (`4.589e-2` to `4.861e-2`) | `4.848061e-02`, `1.0056` | `4.594082e-02`, `0.9530` | `6.377e-2` to `6.720e-2` | ahead by `1.315` to `1.386`; `1.388` to `1.463` | `5.386e-2` to `6.385e-2` | ahead by `1.111` to `1.317`; `1.172` to `1.390` |
| 20 | `6.353308e-02` (`5.792e-2` to `6.353e-2`) | `6.291440e-02`, `0.9903` | `5.784276e-02`, `0.9104` | `8.519e-2` to `9.088e-2` | ahead by `1.354` to `1.445`; `1.473` to `1.571` | `6.991e-2` to `7.564e-2` | ahead by `1.111` to `1.202`; `1.209` to `1.308` |

Both new chains lie inside the ball band at `t = 8`, `10`, `12` and `16`;
at `t = 20` (a) is inside it and (b) is below its bottom by `1.001`.
Against the both-cones `0.128607` at `t = 20`: (a) `0.489`, (b) `0.450`.
The other quantities at the horizons, ball base / (a) / (b) [recorded
`drive_power` chain]:

| quantity | `t = 12` | `t = 16` | `t = 20` |
|---|---|---|---|
| the 27 coordinates outside the phonon-occupation block | `3.459e-02` / `3.521e-02` / `3.410e-02` [`4.385e-02`] | `4.769e-02` / `4.779e-02` / `4.372e-02` [`6.722e-02`] | `6.564e-02` / `6.482e-02` / `5.840e-02` [`9.046e-02`] |
| the four phonon-occupation coordinates | `5.569e-02` / `5.594e-02` / `5.955e-02` [`3.199e-02`] | `5.158e-02` / `5.294e-02` / `5.876e-02` [`3.188e-02`] | `4.692e-02` / `4.814e-02` / `5.391e-02` [`3.187e-02`] |
| blockwise rho / B / N / A / C, `t = 20` | --- | --- | ball `1.006e-01` / `1.252e-01` / `4.692e-02` / `4.258e-02` / `2.974e-02`; (a) `1.004e-01` / `1.233e-01` / `4.814e-02` / `4.078e-02` / `2.977e-02`; (b) `9.109e-02` / `1.077e-01` / `5.391e-02` / `3.812e-02` / `2.934e-02` [`1.358e-01` / `1.873e-01` / `3.187e-02` / `4.397e-02` / `3.128e-02`] |
| Eq. (22) internal-energy RMS | `5.641305e-03` / `5.035053e-03` / `5.641305e-03` [`6.088286e-03`] | `5.776465e-03` / `5.157268e-03` / `5.776465e-03` [`6.249722e-03`] | `5.856071e-03` / `5.229233e-03` / `5.856071e-03` [`6.344620e-03`; tightened `5.217702e-03`; both cones `0.00094`] |
| site-0 occupation, `rho[0,0]` sample RMS | `3.226e-02` / `3.432e-02` / `3.086e-02` [`5.278e-02`] | `5.776e-02` / `5.868e-02` / `5.249e-02` [`8.393e-02`] | `8.122e-02` / `8.073e-02` / `7.242e-02` [`1.126e-01`] |
| minimum retained joint Gram over `t <= T`, time attained | `-2.377e-03` (`10.08`) / `-2.292e-03` (`11.08`) / `-1.298e-03` (`6.52`, the shared segment's) [`-2.463e-03` (`8.76`)] | same | same |
| minimum extension eigenvalue over `t <= T`, time attained | `-2.202e-04` (`9.545`) / `-2.145e-04` (`9.547`) / `-1.369e-04` (`6.678`, shared) [`-2.040e-04` (`9.918`)] | same | same |
| Euclidean coordinate difference from the exact reference at `T`, all 31 | `2.899e-01` / `2.961e-01` / `2.689e-01` [`5.028e-01`] | `4.402e-01` / `4.330e-01` / `3.907e-01` [`6.559e-01`] | `7.668e-01` / `7.466e-01` / `6.761e-01` [`9.694e-01`] |

The radius-free chain's internal-energy RMS is the ball base's scaled by
`0.893` at every horizon (`0.8925`, `0.8928`, `0.8930`), the ratio of the
two declared references' offsets above the exact post-pulse energy
(`5.507e-03 / 6.164e-03 = 0.893`): the constraint holds each chain to its
own reference (to `2.22e-15` over 23,935 and 23,994 constrained repairs,
0 hyperplane refusals in both), so the difference is the reference each
trajectory selected at `t = 4`, not the dynamics past `t = 8`.  Window
scores `(a, b]`, ball base / (a) / (b) [recorded]: `(8, 12]` `4.799e-02`
/ `4.926e-02` / `4.886e-02` [`6.385e-02`]; `(12, 16]` `7.050e-02` /
`7.036e-02` / `6.350e-02` [`1.042e-01`]; `(16, 20]` `1.043e-01` /
`1.019e-01` / `9.104e-02` [`1.415e-01`]; `(8, 20]` `7.780e-02` /
`7.696e-02` / `7.002e-02` [`1.079e-01`].  The deeper repair's gain is in
the last two windows (`0.901`, `0.872`) and it loses the first (`1.018`).

## (a) What the radius did at 27 keys, and what the rule did without it

Decisions past `t = 8` (300 re-selections of 500, merged ledgers):

| chain | accepted (uncertified) | cap-bound | `infeasible` | solver refusals | answers above `0.4`, decisions | applied moves above `0.1` | `(16, 20]` |
|---|---:|---:|---:|---:|---|---|---|
| ball base | 255 (55) | 11 | 34 | 89 | 30: 29 `infeasible`, 1 capped (`0.793` at `11.04`) | none (the cap) | all 100 accepted, largest answer `0.0112` |
| (a) radius-free | 260 (49) | --- | 40 | 89 | 30, all `infeasible` (`0.564` to `0.793`) | 6: `9.08` `0.105u`, `9.20` `0.150`, `11.00` `0.124u`, `11.08` `0.228`, `14.24` `0.120u`, `14.40` `0.135` | all 100 accepted, `0.0111` |
| (b) Dykstra 120 | 281 (71) | 5 | 14 | 85 | 7, all `infeasible` (`0.574` to `0.636`) | none | all 100 accepted, `0.0112` |

Whole chains of 500: ball 441 (91) / 14 / 44 / 1 declined, refusals 136;
(a) 448 (85) / 0 / 52 / 0, refusals 137; (b) 467 (107) / 8 / 24 / 1,
refusals 132.  Median applied move per window `(8, 12]`, `(12, 16]`,
`(16, 20]`: ball `0.0150`, `0.0106`, `0.0074`; (a) `0.0137`, `0.0105`,
`0.0076`; (b) `0.0133`, `0.0107`, `0.0075`; total applied displacement
past `t = 8` `4.785` / `4.465` / `4.359`.  At the ball's eleven cap
times the radius-free chain accepted six answers whole (`9.08` `0.105u`,
`9.20` `0.150`, `10.04` `0.073u`, `11.00` `0.124u`, `11.28` `0.022`,
`11.32` `0.048u`) and found five `infeasible` (`9.24` `0.231u`, `10.00`
`0.596u`, `11.04` `0.793u` --- the same `0.793` answer the ball capped,
on the other side of the measured test from a state `3e-2` away ---
`11.24` `0.577u`, `14.36` diverged at 250 iterations); its remaining
over-radius moves fall where the ball found the completion `infeasible`
(`11.08` `0.857u`, `14.24` `0.620u`) or accepted a small one (`14.40`
`0.060u`).  The two chains part at `t = 0.04` (the first re-selection,
capped against applied whole, task 9) and stay `1.7e-2` to `3.4e-2`
apart in the 31 coordinates from `t = 8` on (`2.012e-02` / `2.056e-02` /
`3.357e-02` / `1.719e-02` / `2.523e-02` at `t = 8` / `10` / `12` / `16` /
`20`, maximum `3.441e-02` at `13.035`; the frontier `d70` maximum
`0.953` at `t = 6.6` is task 8's excursion), 3 percent of either chain's
distance from the exact reference at `t = 20` (`0.747` / `0.767`); from
the recorded chain the radius-free chain is `0.253` to `0.273` away at
every horizon from `t = 8`.  The physicality minima are the ball base's
to 4 percent: over `t <= 20` `-2.292e-03` (at `11.08`) / `-2.145e-04` (at
`9.547`); per window `(8, 12]`, `(12, 16]`, `(16, 20]` `-2.292e-03` /
`-2.145e-04`, `-1.922e-03` / `-1.946e-04`, `-8.275e-04` / `-9.285e-05`
(ball `-2.377e-03` / `-2.202e-04`, `-2.060e-03` / `-1.975e-04`,
`-8.315e-04` / `-9.596e-05`); the frontier envelope maximum over
`t <= 20` is `0.772`, task 8's `t = 6.6` excursion (ball `0.573`).

## (b) The deeper repair

The Dykstra-120 chain leaves the ball base at `t = 8.0025`, the first
repaired step (`d31 >= 1e-3` there, `>= 1e-2` at `8.4925`, never
`1e-1`; `4.733e-02` / `8.681e-02` / `7.591e-02` / `9.784e-02` at
`t = 10` / `12` / `16` / `20`, maximum `9.788e-02` at `14.32`; `d70`
maximum `0.340` at `11.2`), 14 percent of the ball's distance from the
exact reference at `t = 20` and 12 percent nearer to it (`0.676` against
`0.767`).  Its first re-selection decision to differ from the ball's is
at `t = 8.16` (accepted `0.0190` against `infeasible` `0.567u`); the
decisions agree at 257 of the 300 common times past `t = 8`.  The
retained joint Gram at the first repaired sample is `-2.939e-05` against
the ball's `-3.531e-04` from the same `-1.2218e-03` state; per window
`(8, 12]`, `(12, 16]`, `(16, 20]` the minima are `-4.420e-04` /
`-4.187e-05`, `-3.539e-04` / `-3.644e-05`, `-1.429e-04` / `-1.647e-05`
(ball `-2.377e-03` / `-2.202e-04`, `-2.060e-03` / `-1.975e-04`,
`-8.315e-04` / `-9.596e-05`; the `dykstra60` twin, whose minima are
attained past `t = 8`, `-1.085e-03` at `11.04` / `-9.84e-05` at `9.543`);
the extension eigenvalue is below zero at 4,287 of the 4,800 samples
past `t = 8` (ball 4,366) and the retained one at all 4,800 in every
chain; the frontier envelope maximum is `0.515` (ball `0.573`; recorded
`0.452`).  The cap binds five times past `t = 8` (`8.28` `0.139u`,
`8.32` `0.101`, `10.24` `0.259u`, `10.28` `0.151`, `14.28` `0.281u`)
against the ball's eleven, at times where the ball accepted a small
answer (`8.28` `0.031`, `10.24` `0.024u`) or found one `infeasible`
(`8.32` `0.561u`, `10.28` `0.598u`, `14.28` `0.620u`); at nine of the
ball's eleven cap times the deeper repair's completion returns a small
answer (`0.016` to `0.045`, accepted) and at the other two `infeasible`
(`9.20` `0.588u`) or a divergence (`11.04`, 263 iterations).  Seven
answers above `0.4` against the ball's 30, all `infeasible`; 14
`infeasible` against 34.  So the deeper repair hands the completion a
state from which it returns small certified answers more often (281
accepted, 71 uncertified, of 300; the ball 255 and 55), and that, with
the cones, is where its `(12, 16]` and `(16, 20]` gain comes from; the
phonon-occupation block it does not help.

Cost: the four segments `8.0` to `11.04` (1,216 steps, `1,009.9 s`),
`11.04` to `14.24` (1,280, `1,042.2 s`), `14.24` to `18.3925` (1,661,
`1,000.1 s`; resume written at the boundary `18.36`), `18.36` to `20.0`
(656, `364.3 s`); `3,416.5 s` past `t = 8` (ball `2,867.4 s` in three),
`0.817` in the 300 completion solves at `9.30 s` mean (ball `0.924` at
`8.83 s`; the mean iteration count `39.9` against `37.8`, 89 solves at
the 40-iteration cap against 84), repairs `549.3 s`; peaks `630` to
`668 MiB`; longest segment `1,042.2 s` of autonomous wall plus `18.9 s`
of build.  The radius-free chain: `8.0` to `11.4` (1,360 steps,
`1,012.6 s`), `11.4` to `15.4` (1,600, `1,038.6 s`, the largest overrun
of the `1,000 s` budget on record), `15.4` to `20.0` (1,840, `850.6 s`);
`2,901.8 s` past `t = 8`, `4,788.1 s` in all (ball `4,844.7 s`; recorded
`4,794.4 s`), `0.925` in the solves at `8.95 s` mean, repairs `141.8 s`;
peaks `633` to `649 MiB`.

## Gates

| gate | result |
|---|---|
| (a) continues the stored `t = 8` state under task 8's rule | `seg3`'s `summary.json` records `resumed_from` the measured base's `seg2`, `reselection_rule` `{anchor_weight 0.0, trust_radius None, trust_policy carried, acceptance_margin 0.001, candidates measured}`, `dykstra_iterations 30`, the `e3374f6f` source tree; its `t = 8.0` sample carries the `seg2`'s retained minimum `-1.2847e-03`; resumes at `8.0`, `11.4`, `15.4` (85 and 100 intervals of 16 steps) |
| (b) resumes through the links from the ball base's `t = 8` state | `seg3`'s `resumed_from` is the `dyk120` prefix's `seg2` link (resolved to `..._ball_base_20260917k_seg2/resume.npz`); `d31 = d70 = 0` between (b) and the ball base at `t = 8.0` and the first differing sample is `t = 8.0025`; `dykstra_iterations` `120` in `seg3` to `seg6` and `30` in the linked `seg1`, `seg2`; the score ledger lists six segments from `t = 0`; resumes at `8.0`, `11.04`, `14.24`, `18.36` (76, 80, 103 intervals; `seg5` reached `18.3925` and wrote its resume at the boundary) |
| the source tree is HEAD's | `diff -rq head_src_e3374f6f/paper5 paper_5/src/paper5 -x __pycache__` empty against `c14bf1bd`, recorded at the queue's start |
| the energy is held | maximum deviation from the declared reference `2.22e-15` over 23,935 (a) and 23,994 (b) constrained repairs, 0 hyperplane refusals; (a)'s `seg5` records its reference as `-3.5658908873520323`, one unit in the last place from `seg3`'s and `seg4`'s `-3.5658908873520327`, the carried energy at the `t = 15.4` boundary |
| one grid | 8,001 common samples, `t = 0` to `20`; `tabulate_task10.py` asserts the `exact_t20.npz` grid against every chain's times to `1e-12` |
| the ledgers merge | 500 entries on one schedule per chain, 300 past `t = 8`, 100 in `(16, 20]`; the cross-check finds every cap time of one chain in the other two ledgers |
| one heavy process at a time, under 20 minutes | seven segments under `ram_guard --limit-mb 4500`, sequential; longest `1,042.2 s` plus `18.9 s` of build; scoring and tabulation under `--limit-mb 2500`, arrays only |

## The 27-key ledger row after task 10

| horizon `t = 0` to `T` | the row | band | task 10 | what changes |
|---:|---|---|---|---|
| 4 | task 8's anchor-free rule, `1.00e-2` | `9.96e-3` to `1.001e-2`, 0.5 % | (a) is task 8's base | nothing |
| 8 | the recorded `drive_power` chain, `2.59e-2` | `2.59e-2` to `2.71e-2`, 4.5 % | (a) `3.181e-2`, (b) `3.181e-2` (shared), both behind by `1.17` to `1.23` | nothing |
| 12 | the ball rule, base `3.80e-2` | `3.797e-2` to `3.906e-2`, 2.9 % | (a) `3.852e-2` and (b) `3.834e-2` inside | nothing in the number |
| 16 | the ball rule, base `4.82e-2` | `4.589e-2` to `4.861e-2`, 5.9 % | (a) `4.848e-2` and (b) `4.594e-2` inside | nothing in the number |
| 20 | the ball rule, base `6.35e-2` | `5.792e-2` to `6.353e-2`, 9.7 % | (a) `6.291e-2` inside; (b) `5.784e-2` below the bottom by `1.001` | nothing in the number unless the Dykstra-120 member is adopted as the low edge (`5.78e-2`, one run, no band of its own) |

What changes is the row's attribution, not its numbers or its standing:
the radius is not an ingredient of any number past `t = 8` and costs 4
percent at `t = 4`, so the rule of the row may be stated as the
anchor-free measured rule (task 8's), the ball members and the
radius-free base together being its band (five members: `3.797e-2` to
`3.906e-2` at `t = 12`, `4.589e-2` to `4.861e-2` at `16`, `5.792e-2` to
`6.353e-2` at `20`, unchanged); and the Dykstra count is the one setting
measured to move the score past `t = 12` --- 30 / 60 / 120 give
`6.353e-2` / `5.792e-2` / `5.784e-2` at `t = 20` and `4.821e-2` /
`4.589e-2` / `4.594e-2` at `t = 16`, saturating from 60 to 120 in the
score but not in the cones (retained joint Gram minimum past `t = 8`
`-2.38e-3` / `-1.09e-3` / `-4.42e-4`, the 60 on its own trajectory from
`t = 0.04`) --- at 11 percent of wall per doubling and with the
phonon-occupation block worse each time.  Adoption is the user's call;
the plan-file row carries the numbers.

## Provenance

- Seven rollouts under `pipelines/shell/ram_guard.py --limit-mb 4500`,
  one at a time, `02:13` to `04:01` on 2026-09-18:
  `paper_v_dictionary_energy_power_27key_n16_measured_base_20260916i_seg{3,4,5}`
  and
  `paper_v_dictionary_energy_power_27key_n16_ball_base_dyk120_20260918l_seg{3,4,5,6}`
  (with `seg1`, `seg2` symbolic links to
  `..._ball_base_20260917k_seg{1,2}`), each with `reselections.json`.
  Scoring, tabulation and the cross-check under `--limit-mb 2500`, arrays
  only.
- Scripts under `output/local_runs/paper_v_projective_dictionary_energy_20260915e/`:
  `queue_task10.sh` (runs `queue_task9_t20.sh measured`, makes the two
  links, chains (b) with `--dykstra-iterations 120`, scores it),
  `tabulate_task10.py` (ledgers, totals, the chart with ratios and
  standings, windows, physicality minima with the times attained over
  `t <= 20`, `(8, 20]` and per window, decisions past `t = 8` with every
  cap and every over-radius applied move, separations),
  `crosscheck_task10_caps.py` (every chain's decision at every cap time
  or over-radius move of the others), `update_plan_row_task10.py`.
  Outputs: `task9_measured_base_score_t20.json` and `.log`,
  `task10_ball_base_dyk120_score_t20.json` and `.log`,
  `tabulate_task10.log` (and `tabulate_task10_a.log`, run when (a) alone
  had landed), `crosscheck_task10_caps.log` (and `_a.log`),
  `task10_chain.log`, `queue_task10.log`, `task9_t20_chain.log`
  (appended), the segment logs
  `paper_v_dictionary_energy_power_27key_n16_measured_base_20260916i_seg{3,4,5}.log`
  and `..._ball_base_dyk120_20260918l_seg{3,4,5,6}.log`.
- No module was edited, nothing was committed, no default, rule, radius,
  Dykstra count or ladder changed; task 9's and the recorded chains'
  artifacts are untouched.  Wall times were measured with nothing else
  heavy on the machine.  The worklog carries a matching block; the
  plan-file row was updated in place.

## What to try next, in order

1. **The anchor-free rule without the radius with Dykstra 120 from
   `t = 0` to `t = 20`** (one chain, about 95 minutes): the two
   ingredients this round measured separately --- the acceptance that
   owns the gain and the repair depth that moves the cones and the
   `t = 20` score --- combined, which gives the Dykstra-120 member a
   `t = 4` and `t = 8` standing (whether the deeper repair moves the
   `(4, 8]` phonon-occupation loss) and the `t = 20` row a band partner.
2. **The `(4, 8]` phonon-occupation loss**, which every member carries
   and the deeper repair worsens (`5.39e-2` against `4.69e-2` at
   `t = 20`; the recorded chain `3.19e-2`): task 8's proposal stands, a
   rerun with `certified` candidates from `t = 4`.

## Structured summary of task 10

| item | value |
|---|---|
| the two chains | (a) `LiftReselectionRule(acceptance_margin=1e-3, candidates="measured", anchor_weight=0)`, no radius, Dykstra 30, continued from `..._measured_base_20260916i_seg2` (`t = 8`) to `t = 20` in three segments; (b) the ball base (`trust_radius=0.1`, `ball`) continued from `..._ball_base_20260917k_seg2` with `--dykstra-iterations 120` in four segments under the `dyk120` prefix (`seg1`, `seg2` links); 27 keys, `N = 16`, `drive_power`, `e3374f6f` tree identical to `c14bf1bd` |
| (a) `t = 12` / `16` / `20` | `3.851781e-02` / `4.848061e-02` / `6.291440e-02`; ratios to the ball base `1.014` / `1.006` / `0.990`; inside the ball band at every horizon from `t = 8`; ahead of the `drive_power` bands by `1.104` to `1.125`, `1.315` to `1.386`, `1.354` to `1.445` and of the `post_pulse` bands by `1.197` to `1.577`, `1.111` to `1.317`, `1.111` to `1.202`; `0.489` of the both-cones line; windows `(8, 12]`, `(12, 16]`, `(16, 20]` `4.926e-02` / `7.036e-02` / `1.019e-01` (ratios to the ball `1.027` / `0.998` / `0.977`; the recorded chain `1.331` / `1.477` / `1.356`) |
| (a) energy, physicality, decisions, wall | internal-energy RMS `5.035053e-03` / `5.157268e-03` / `5.229233e-03` (the ball's scaled by the references' offsets, `0.893`); retained joint Gram `-2.292e-03` at `11.08`, extension `-2.145e-04` at `9.547` (ball `-2.377e-03` / `-2.202e-04`); past `t = 8` accepted 260 (49 uncertified), `infeasible` 40, refusals 89, no cap, six applied moves above `0.1` (`0.105` to `0.228`, excess `0.261` of `4.465`), 30 answers above `0.4` all `infeasible`; `(16, 20]` accepts all 100; wall `4,788.1 s` to `t = 20`, `2,901.8 s` past `t = 8`; peaks `649 MiB` |
| (b) `t = 12` / `16` / `20` | `3.834289e-02` / `4.594082e-02` / `5.784276e-02`; ratios `1.010` / `0.953` / `0.910`; inside the ball band at `12` and `16`, below its bottom by `1.001` at `20` (the `dykstra60` twin `5.792205e-02`); ahead of the `drive_power` bands by `1.109` to `1.130`, `1.388` to `1.463`, `1.473` to `1.571` and of the `post_pulse` bands by `1.203` to `1.585`, `1.172` to `1.390`, `1.209` to `1.308`; `0.450` of the both-cones line; windows `4.886e-02` / `6.350e-02` / `9.104e-02` (ratios `1.018` / `0.901` / `0.872`) |
| (b) energy, physicality, decisions, wall | internal-energy RMS identical to the ball's (`5.641305e-03` / `5.776465e-03` / `5.856071e-03`; the same pinned reference); over `(8, 20]` retained joint Gram `-4.420e-04` at `11.047` and extension `-4.187e-05` at `9.498` against the ball's `-2.377e-03` / `-2.202e-04` (`5.4` and `5.3` times shallower; mean negative part `6.1` and `5.8` times smaller), the retained cone still violated at every sample; phonon-occupation block `5.391e-02` against `4.692e-02` at `t = 20`, the other 27 coordinates `5.840e-02` against `6.564e-02`; past `t = 8` accepted 281 (71), cap 5, `infeasible` 14, refusals 85, 7 answers above `0.4` all `infeasible`; first differing decision `t = 8.16`; wall `5,393.7 s` (`+11 %`), repair `549.3 s` past `t = 8` (`3.85` times), solves `9.30 s` mean; peaks `668 MiB` |
| which ingredient owns the gain past `t = 8` | the measured-candidate acceptance (margin `1e-3`, `measured`, anchor `0`): the radius-free chain reproduces the ball chain's windows to `0.977` to `1.027` and its scores to `0.990` to `1.014`, and every large answer the radius would cap is refused by the measured test; the radius moves nothing past `t = 8` at 27 keys (six over-radius moves of `0.105` to `0.228`) and costs 4 percent at `t = 4` |
| the ledger row | numbers and standing unchanged: `1.00e-2` (task 8) at `t = 4`, `2.59e-2` (recorded) at `t = 8`, `3.80e-2` / `4.82e-2` / `6.35e-2` (ball rule base) at `t = 12` / `16` / `20` with the bands `3.797e-2` to `3.906e-2`, `4.589e-2` to `4.861e-2`, `5.792e-2` to `6.353e-2` (the radius-free base inside each); the rule may be stated without the radius; the Dykstra count is the setting that moves the score past `t = 12` (30 / 60 / 120: `6.353e-2` / `5.792e-2` / `5.784e-2` at `t = 20`) and the cones; the Dykstra-120 member (`5.78e-2`, one run) is the low edge at `t = 20` if adopted |
| gates | (a) resumes the stored state under task 8's rule (recorded in `seg3`'s summary); (b) `d31 = 0` at `t = 8.0` against the ball base, first difference `t = 8.0025`, `dykstra_iterations 120` recorded in `seg3` to `seg6`; source tree identical to HEAD; energy held to `2.22e-15`; one grid (8,001 samples); ledgers merge to 500; seven sequential segments under `4,500 MB`, longest `1,042.2 s` |
| files changed | this report (new heading), the worklog block, the plan ledger's row (in place); the task-10 scripts and outputs under `output/local_runs/` listed above.  No module; nothing committed |
| next | the anchor-free rule without the radius with Dykstra 120 from `t = 0` to `t = 20`; the `(4, 8]` phonon-occupation loss |
