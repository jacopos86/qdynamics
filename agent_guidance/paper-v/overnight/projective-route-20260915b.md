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
