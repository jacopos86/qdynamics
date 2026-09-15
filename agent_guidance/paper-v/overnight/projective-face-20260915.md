# projective-face (2026-09-15)

Round 3, overnight loop.  Make the projective fixed-dictionary route survive
its two recorded degeneracies and always save what it reached.

Round 1 and 2 measured, for fixed-dictionary APCM with the event-triggered
one-shell outer guard, SSPRK(3,3), 15 entrance keys, phonon cutoff 16,
`lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude 1, exact-correlated
preparation:

- the default stage policy reaches `t = 4` at `h = 0.0025` with
  all-coordinate scalar RMS `0.026049` against the exact cutoff-16 driven
  reference, 1,068 stage retractions, `3,229.8 s`, and then fails between
  `t = 5` and `t = 6` with `moment_extension_failure persisted after 8 stage
  subdivisions: retracted lower moments did not admit the declared selector:
  moment_extension_degeneracy: conic Newton step failed descent after full
  cone interior`, saving no trajectory;
- the carried-completion option reaches all-coordinate scalar RMS
  `1.2875e-5` over `t = 0` to `0.05` at `h = 0.00125` and stops at
  `t = 0.11375` with `extended operator moment matrix numerical face is
  unstable under null-tolerance refinement`.

Defaults untouched: every new behaviour below is a named option, flag or
settings field whose default reproduces the committed path, with a test on
the default path.

Status: complete.

## The result, first

The projective route's stage retractions are not repairs of an infeasible
point.  They fire because the declared proximal log-det completion returns
`success = False` at a lift that is inside its cone, and containment treats
an uncertified completion as infeasible.  Which declared setting certifies is
a property of the stage, not of the trajectory: at each of the three stages
where a rollout stopped this round, the setting that failed there is
certified by another setting, and the setting that certified there fails at
one of the other two.

Retrying a refused completion under declared alternatives and taking the
first certified result --- one named, empty-by-default ladder --- removes all
seven stage retractions of the recorded `t = 0` to `0.05` rollout and takes
its all-coordinate scalar RMS against the exact cutoff-16 reference from
`7.931831e-05` to `1.227728e-05`, with the N block from `2.1728e-04` to
`3.1027e-08`.  That is round 2's attributed gap, closed at its cause.

Past the pulse, no option reaches `t = 6` inside a 40-minute budget.  The
committed path stops at `t = 5.085`; the best horizon measured here is
`t = 5.235`, reached by the carried lift with the face carried across stages
and stopped by the budget rather than by a refusal.

## Impact analysis (main checkout index, read-only; index 4 commits behind)

| symbol | verdict | callers |
|---|---|---|
| `integrate_projected_apcm_ssprk3` | UNKNOWN -> LOW after grep | `apcm_projective_event_guard_analysis.py`, `tests/test_projected_apcm.py`, `tests/test_apcm_projective_event_guard.py` |
| `_advance_projected_apcm_interval` | UNKNOWN -> LOW after grep | `projected_apcm.py` only (itself and the integrator) |
| `_kernel_basis` | LOW | `SymmetryReducedAPCMGeometry.project_velocity` |
| `project_velocity` | UNKNOWN -> LOW after grep | `projected_apcm.py:680`, `tests/test_apcm_moment_projection.py:147` |
| `_stage_cone_data` | UNKNOWN -> LOW after grep | `apcm_moment_projection.py:896` only |
| `contain_stage` | UNKNOWN -> LOW after grep | `apcm_adaptation.py:215` (a different model class), `projected_apcm.py:1185,1284,1290`, three test files |
| `evaluate` | UNKNOWN -> LOW after grep | `projected_apcm.py` interval advance; the other `.evaluate(` hits are other model classes |

No HIGH or CRITICAL verdict.  Every UNKNOWN was resolved by text search
before editing.

## What changed, and where

All five modules and four test files below are mine; every default is
unchanged and every new behaviour is a named field, keyword or flag.

- `paper_5/src/paper5/stability/projected_apcm.py`
  - `ProjectedAPCMStageFailure`, a frozen dataclass carrying the stage a
    declared failure was raised at (`"evaluate"` or `"contain"`), its time,
    the state it was raised at, that state's lower moments in the declared
    extension ordering, the frontier moments the stage was given, and the
    message.  `projected_apcm_stage_failure(error)` and
    `projected_apcm_partial_trajectory(error)` read the two payloads back.
  - `FixedDictionaryProjectedAPCM.evaluate` and `.contain_stage` attach that
    record to any exception they raise and re-raise it unchanged; the body of
    `contain_stage` moved unchanged into `_contain_stage`.  The raised type
    and message are untouched, so nothing that catches them sees a change.
  - `integrate_projected_apcm_ssprk3` builds its return value through one
    local `reached(...)`, and attaches the trajectory over the steps already
    accepted to the exception before re-raising on the failure path, on the
    nonfinite-state path, and on any other exception.  The tolerated path
    (`tolerate_stage_failure=True`) keeps its exact message.
  - `integrate_projected_apcm_ssprk3(..., initial_time=...)`, default `0.0`:
    the rollout spans `initial_time` to `final_time` and the drive is
    evaluated at the times the state belongs to.  At the default the sample
    times are bit for bit what they were.
  - `ProjectedAPCMTrajectory` gains `stage_failure`, default `None`, so the
    named `tolerate_stage_failure` path reports the same stage record the
    raising path attaches to its exception, and the driver writes
    `stage_failure.npz` there too.
  - `ProjectedAPCMStagePolicy` gains `completion_retry_shifts` and
    `completion_retry_weights`, both empty by default, and
    `build_completion_ladder` builds the extensions they name once per
    model.  When the declared completion is not certified the ladder is
    tried in order, at the stage lift and again inside the stage retraction,
    and the first certified result is taken; the counters
    `completion_retry_accepted` and `completion_retry_exhausted` say how
    often.  An empty ladder leaves the committed single-solve path, which is
    what the bitwise gate reproduces.
- `paper_5/src/paper5/stability/apcm_moment_projection.py`
  - `APCMProjectionSettings` gains `face_policy` (`"refuse"`, the committed
    path), `face_continuation_tolerance` (`1e-6`) and
    `face_inflation_multiple` (`100.0`, the declared stage margin of the
    coupled stage cone).  The two tolerances are read only by the policy
    that names them.
  - `_kernel_basis` returns the face basis, the rank, and how it was
    identified.  Under `"continuation"` an ambiguous face falls back to the
    face carried from the previous identification when the carried rank lies
    in the ambiguous range and the relative consistency residual
    `||M V||_inf / (1 + ||M||_inf)` is at or below the declared tolerance,
    which is the same shape of gate `FrozenOuterFaceSelector.solve` uses
    (`apcm_projective_guard.py:1926-1938`); otherwise fresh identification
    runs and refuses as before.  Under `"inflated"` the ranks are counted on
    the spectrum of the cone inflated by `face_inflation_multiple` conic
    tolerances.
  - `SymmetryReducedAPCMGeometry._identify_face` holds the carried faces (one
    per matrix name) and the counters `identified`, `continued`, `refused`;
    under the default policy nothing is carried.
  - The refusal raised after a stage retraction carries the retracted lower
    moments, the retraction's frontier moments and the two cone minima on
    the exception (`apcm_retracted_stage`), because that refusal is about the
    retracted point, not the trial.
- `paper_5/src/paper5/stability/apcm_projective_event_guard_analysis.py`
  - The rollout is wrapped: on any exception the driver writes the partial
    trajectory, the exact reference over the horizon reached, the accuracy
    metrics, the failing stage's state, lower moments and frontier moments
    (`stage_failure.npz`), and a `failure` block in `summary.json`, and
    returns 1.  A rollout that accepted no step writes the failure block
    alone.
  - Nine new flags, each defaulting to the committed value:
    `--face-policy`, `--face-continuation-tolerance`,
    `--face-inflation-multiple`, `--completion-logdet-shift`,
    `--completion-logdet-weight`, `--completion-retry-shifts`,
    `--completion-retry-weights`, `--initial-state-file`, `--initial-time`.
    The summary gains `projection_settings`, `extension_settings`,
    `face_policy_counters`, `retraction_completion_ladder_counters` and
    `continuation`.
  - The exact reference is built from `t = 0` through a grid that ends on the
    rollout's own sample times and scored on the tail, so a rollout continued
    from a stored state is still scored against the exact cutoff-16 driven
    reference.
- Tests: `test_projected_apcm.py` (+6), `test_apcm_moment_projection.py`
  (+7), `test_apcm_projective_event_guard.py` (+6).

## (a) A failed rollout is now scoreable over the horizon it reached

The recorded `t = 20` attempt wrote only `progress.jsonl`.  The same failure
now writes `autonomous_trajectory.npz`, `trajectory.npz` (with the exact
cutoff-16 reference over the horizon reached), `outer_guard_stages.jsonl`,
`stage_failure.npz` and a `summary.json` carrying the accuracy metrics, the
failure message, the number of accepted steps, the last accepted time, and
the failing stage's state and lower moments.

Forced-failure test: `test_driver_saves_the_horizon_a_failed_rollout_reached`
forces the fourth stage containment of a cutoff-6 rollout to raise, which is
the first stage of the second step, and asserts one accepted step, the
partial trajectory of two samples, the failure block, the stage record and
the nonzero exit status.  `test_a_failed_rollout_carries_the_steps_it_accepted`
and `test_the_failing_stage_carries_its_own_lower_moments` cover the two
payloads at the integrator and the model.

First use of it, which is also the (c) diagnosis run: the default policy
continued from the stored `t = 4` state of
`output/local_runs/paper_v_apcm_projective_event_guard_t4_h0025_cutoff16_20260914_v1/autonomous_trajectory.npz`
toward `t = 8` at `h = 0.0025`, 15 entrance keys, phonon cutoff 16, strong
coupling.  Artifact:
`output/local_runs/paper_v_apcm_projective_t4restart_default_t8_20260915_v1/`.

| quantity | value |
|---|---:|
| accepted steps, last accepted time | 434, `t = 5.085` |
| failing stage and its time | containment, `t = 5.08626953125` |
| failure | `moment_extension_failure persisted after 8 stage subdivisions: retracted lower moments did not admit the declared selector: moment_extension_degeneracy: conic Newton step failed descent after full cone interior` |
| all-coordinate scalar RMS against the exact cutoff-16 reference over `t = 4` to `5.085` | `4.073972e-02` |
| the same with the `t = 4` offset subtracted | `4.384520e-02` |
| blockwise, offset subtracted, rho / B / N / A / C | `1.1331e-1` / `4.0277e-2` / `1.6280e-3` / `1.3612e-2` / `3.1013e-2` |
| stage retractions, minimum unshifted retained joint-Gram eigenvalue | 122, `3.0568e-3` |
| minimum scaled positive-extension eigenvalue | `2.0113e-5` |
| autonomous wall time / peak resident | `250.5 s` / `352 MB` |

The outer guard evaluated 1,330 stages on the restart, triggered none, fired
no selector solve, and spent `211.3 s` of the `250.5 s`; its minimum endpoint
unshifted 24-row Gram eigenvalue was `-1.55e-04` and its maximum negative-mode
coupling measure `7.31e-04`, both inside the tolerated band.

The restart reproduces the recorded failure: the drive has ended, the rollout
passes `t = 5` (the recorded attempt's progress line at `t = 5` reads
`7.700e-4` / `2.022e-4`, this one `7.695e-4` / `2.021e-4`, the difference
being that the restart re-selects the completion at `t = 4` cold), and it
stops inside `5 < t < 6` with the same message.

The continuation is aligned with the exact reference: the error the restart
inherits at `t = 4`, blockwise in Euclidean norm, is `0.2075616308543594`,
`0.08379382331525087`, `0.013377736122679642`, `0.06403895371237167` and
`0.0911085202381703` for rho, B, N, A and C, which are the five final
blockwise errors the recorded `t = 0` to `4` run reported, to every printed
digit.

## (b) The unstable numerical face: two named options, and both carry the rollout to t=0.5

All three rows: fixed-dictionary APCM with the event-triggered one-shell
outer guard at its declared floors, SSPRK(3,3), `--carried-completion`,
`h = 0.00125`, `t = 0` to `0.5`, 15 entrance keys, phonon cutoff 16,
`lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude 1, exact-correlated
preparation, scored against the exact cutoff-16 driven reference regenerated
in process.  "face" counts are the identifications of the two numerical
faces (`retained affine moment matrix` and `extended operator moment
matrix`) over the whole rollout.

| face policy | reached `t` | all-coordinate scalar RMS over the horizon reached | over `t = 0` to `0.05` | stage retractions | carried lifts accepted / declined | face identified / continued / refused | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `refuse` (committed path) | 0.11375 | `5.824278e-05` | `1.287512e-05` | 1 | 293 / 7 | 609 / 0 / 9 | 15.1 |
| `continuation` | 0.5 | `9.547375e-04` | `1.287512e-05` | 7 | 1176 / 36 | 2363 / 61 / 0 | 70.3 |
| `inflated` | 0.5 | `9.547375e-04` | `1.287512e-05` | 7 | 1176 / 36 | 2424 / 0 / 0 | 71.0 |

Artifacts:
`output/local_runs/paper_v_apcm_projective_carried_{facerefuse,facecontinuation,faceinflated}_t05_h000125_20260915_v1/`.

The outer guard fired no selector solve in any of the three: 300 stages
evaluated on the committed path and 1,209 on each of the two that reach
`t = 0.5`, none triggered, minimum endpoint unshifted 24-row Gram eigenvalue
`-6.946e-07`, which is inside the tolerated band between the declared floor
`1e-8` and the hard floor `1e-3`, and the maximum negative-mode coupling
measure is `1.934e-05` against the threshold `1e-2`.

Readings.

1. **Both named options carry the carried-completion rollout from `t =
   0.11375` to the full `t = 0.5`**, with the minimum unshifted retained
   joint-Gram eigenvalue `2.892080e-05` and the minimum scaled
   positive-extension eigenvalue `-7.67e-16` (attained at `t = 0`), 7 stage
   retractions over 400 steps, in `70.3 s` and `71.0 s`.
2. **The two options produce the same trajectory, array for array.**  Every
   array of `autonomous_trajectory.npz` is equal under `np.array_equal`
   between the continuation and the inflated run.  The reason is that at
   every ambiguous stage here the face carried from the previous
   identification has rank zero and the inflated cone exposes no face, so
   both policies say the same thing: at this stage there is no face.  The
   continuation policy reached that answer 61 times; the inflated policy
   never had to, because the inflated rank was already zero.
3. **Neither option changes anything before the refusal.**  Over the 91
   steps the committed path survives, every array of the continuation run is
   equal under `np.array_equal` to the committed path's, and the
   all-coordinate scalar RMS over `t = 0` to `0.05` is
   `1.287512e-05` in all three runs, reproducing round 2's `1.2875e-5`.
4. **The horizon costs accuracy, and the cost is in the C block.**  Over
   `t = 0` to `0.5` the all-coordinate scalar RMS is `9.547375e-04`
   (blockwise, offset subtracted, rho / B / N / A / C
   `3.4962e-5` / `1.5703e-5` / `1.3000e-4` / `1.4459e-4` / `1.4157e-3`),
   against `5.824278e-05` over `t = 0` to `0.11375` and `1.287512e-05` over
   `t = 0` to `0.05`.  The carried-witness flow without its Gram guard is
   `3.19169e-5` over the same `t = 0` to `0.5` (worklog:3484-3493 and the
   2026-09-14 anchor), so the projective route with a carried lift is a
   factor 30 behind it at this horizon, having been a factor 3.98 behind
   the strict carried-witness figure at `t = 0.05`.

How the error grows once the rollout is allowed to continue, on the
continuation run (all-coordinate and blockwise scalar RMS over `t = 0` to the
stated time, at `h = 0.00125`):

| up to `t` | all | rho | B | N | A | C |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | `1.288e-05` | `1.524e-08` | `1.580e-06` | `1.830e-05` | `9.084e-06` | `1.534e-05` |
| 0.1 | `4.504e-05` | `7.734e-08` | `3.150e-06` | `1.864e-05` | `1.811e-05` | `6.519e-05` |
| 0.2 | `1.710e-04` | `1.181e-06` | `6.289e-06` | `2.372e-05` | `3.622e-05` | `2.530e-04` |
| 0.3 | `3.661e-04` | `5.764e-06` | `9.425e-06` | `4.219e-05` | `5.638e-05` | `5.430e-04` |
| 0.4 | `6.315e-04` | `1.613e-05` | `1.256e-05` | `6.661e-05` | `8.697e-05` | `9.372e-04` |
| 0.5 | `9.547e-04` | `3.496e-05` | `1.570e-05` | `1.300e-04` | `1.446e-04` | `1.416e-03` |

The C block is 92 times its `t = 0.05` value at `t = 0.5` while the N block,
which the retraction moved in round 2, is 7 times its own, so what limits the
projective route past `t = 0.1` is the electron--phonon correlation block of
the closure, not the stage policy this round changed.

What the refused face actually is, read from the stage record (a) saved for
the committed-path run: at the failing stage (a right-hand-side evaluation at
`t = 0.1149755859375`) the scaled extended operator moment matrix has
eigenvalues `1.9998489537e-08`, `3.8893138102e-04`, `5.6271442961e-03`,
`1.2076926845e-02`, `1.3462262431e-02` in its five smallest, against the
declared null tolerance `null_tolerance_factor * conic_tolerance = 1e-08`.
The refinement counts 0, 0 and 1 eigenvalues at or below 0.5, 1 and 2
tolerances, so the refusal is one eigenvalue sitting `0.008` percent below
twice the null tolerance, with the next eigenvalue four decades above it.
The scaled retained affine moment matrix at the same stage is clear of its
own tolerance (`3.638e-06`).  Both named options answer that stage the same
way: the previous identification exposed no face, and the cone inflated by
100 conic tolerances exposes none either.

One property of the continuation policy should be stated rather than left to
be discovered: a carried face of rank zero says that the previous
identification exposed no face, and the consistency residual of an empty
basis is zero, so what discriminates at rank zero is the rank range, which
admits zero only when the refinement is itself undecided between no face and
a face.  That is why the two policies coincide on this rollout.  Where a
carried face has positive rank the residual gate is the part that decides,
and it is the same shape of gate the frozen-face selector already uses.

## (c) The post-pulse degeneracy is a certification failure, not an infeasible point

At the failing stage saved by (a) (`stage_failure.npz` of the `t = 4`
restart, containment at `t = 5.08626953125`, 29 lower moments and the 62
frontier moments the stage was given), the declared proximal log-det
completion was re-solved offline at those lower moments, warm started from
those frontier moments, once per named option.  Artifact:
`output/local_runs/paper_v_apcm_projective_completion_sweep_t5086_20260915_v1/`
(`summary.json`, `spectra.npz`).

The extension at the failure is inside its cone.  The scaled
positive-extension matrix the declared completion returns has spectrum
`9.297881e-06`, `5.851432e-05`, `2.072768e-04`, `5.260885e-03`,
`7.345651e-03`, `1.228254e-02`, `5.550521e-02`, `2.236574e-01`,
`2.391787e-01`, `2.512733e-01` in its ten smallest values, and the lift
carried from the previous stage builds a scaled extension matrix at the same
lower moments with minimum eigenvalue `9.346049e-06` and maximum absolute
standardized coordinate `0.554376` against the frozen envelope `1`.  What
fails is the certificate: the interior Newton iteration reaches its declared
cap of 40 iterations with stationarity residual `6.39e-07` against the
acceptance `50 * conic_tolerance = 5e-08`, so `complete` returns
`success = False` with a positive definite lift, containment treats the
stage as infeasible, the retraction fires, and the re-selection at the
retracted point refuses in the same way.

| named option | certified | scaled minimum eigenvalue of the lift | KKT backward error | Newton iterations | wall s |
|---|---|---:|---:|---:|---:|
| declared (`logdet_shift = 1e-10`, `logdet_weight = 1`) | no | `9.2979e-06` | `6.39e-07` | 40 (cap) | 0.12 |
| PSD inflation `logdet_shift = 1e-9` | yes | `9.2973e-06` | `6.66e-09` | 5 | 0.00 |
| PSD inflation `logdet_shift = 1e-8` | yes | `9.2918e-06` | `8.69e-09` | 4 | 0.00 |
| PSD inflation `logdet_shift = 1e-7` | yes | `9.2362e-06` | `9.64e-09` | 5 | 0.00 |
| PSD inflation `logdet_shift = 1e-6` | no | `8.6822e-06` | `1.89e-03` | 4 | 2.91 |
| proximal weight `logdet_weight = 1e-1` | yes | `2.1241e-06` | `3.47e-09` | 11 | 0.01 |
| proximal weight `logdet_weight = 1e-2` | yes | `2.4579e-07` | `4.30e-08` | 23 | 0.02 |
| proximal weight `logdet_weight = 1e-3` | no | `7.0066e-10` | `3.08e+01` | 40 (cap) | 6.81 |
| `logdet_weight = 1e-2` with `logdet_shift = 1e-8` | no | `2.3605e-07` | `3.34e+00` | 19 | 3.61 |
| the carried lift, no solve | --- | `9.3460e-06` | --- | 0 | 0.00 |

Both failures at the far end of each sweep have the same cause, and it is
declared in the code: the interior branch is taken only while the relative
minimum eigenvalue exceeds `max(0.1 * conic_tolerance, 10 * logdet_shift)`
(`apcm_positive_extension.py:1224-1231`).  At `logdet_shift = 1e-6` that
threshold is `1e-05`, above the lift's own `8.68e-06`, so the solve falls
to the conic certificate and returns `1.89e-03`; at
`logdet_weight = 1e-3` the barrier is too weak to hold the minimizer off the
boundary (`7.0e-10`) and the same thing happens.  The three options that
certify all keep the lift in the cone's relative interior and differ in how
far they move it: the PSD inflation moves the minimum eigenvalue by at most
0.7 percent, the proximal weight moves it by a factor 4.4 at `1e-1` and 38
at `1e-2`.

Which direction of the barrier weight certifies is not symmetric.  Raising it
to 10 and to 100 at the same failing stage leaves the solve uncertified
(KKT backward error `1.33e-01` and `7.88e-01`, lift minimum `1.5254e-05` and
`1.5290e-05`, 2.18 s and 2.01 s;
`.../paper_v_apcm_projective_completion_sweep_t5086_20260915_v1/weight_up.json`),
so "a proximal step of larger declared weight" here means a smaller
`logdet_weight`: the objective is `0.5 ||z||^2` minus
`logdet_weight / dimension` times the log-determinant barrier, and it is the
proximal term's weight relative to the barrier that has to grow.

How close to the certification boundary the post-pulse trajectory runs: at 16
step endpoints sampled evenly over the `t = 4` to `5.085` restart, the
declared completion solved cold (no warm frontier) certifies 14, failing at
`t = 4.94` and at `t = 5.085` with the conic Newton descent failure (KKT
backward error reported as infinite); with `logdet_shift = 1e-8` it certifies
all 16.  The rollout survived `t = 4.94` because the stage completion there
is warm started from the previous stage, not cold.

The same sweep at the other two failing stages says the same thing three
times.  At the stage where the inflated run stopped (`t = 5.104736328125`)
the declared settings certify with KKT backward error `5.86e-09` in 4 Newton
iterations while `logdet_shift = 1e-8` returns the conic Newton descent
failure, and `1e-7`, `1e-1` and `1e-2` certify.  At the stage where the
proximal-weight run stopped (`t = 4.831923828125`) the declared settings and
the shifts `1e-9`, `1e-8`, `1e-7` and the weight `1e-1` all certify
(`4.05e-09` to `4.82e-09`), while `logdet_weight = 1e-2`, which is what that
run was using, returns the conic Newton descent failure at a lift of minimum
eigenvalue `3.39e-10`, and the lift carried from the previous stage is
outside the cone there (`-5.6630e-06`).  No declared setting certifies at all
three stages, and at each stage several of the others do.

### Two declared constants collide at the carried lift's acceptance margin

At the stage where the carried-completion continuation of the `t = 4` state
stopped (a right-hand-side evaluation at `t = 5.0990625`), the lift carried
from the previous stage builds a scaled extension matrix whose minimum
eigenvalue is `1.1189e-08`.  The carried lift is accepted when that minimum
is at or above `carried_completion_margin_multiple * conic_tolerance`, which
is `10 * 1e-09 = 1e-08`, and the numerical face is refused as unstable when
an eigenvalue lies between `0.5` and `2` times
`null_tolerance_factor * conic_tolerance`, which is the band
`5e-09` to `2e-08`.  The accepted region therefore begins inside the refused
band: a carried lift accepted at its margin has an ambiguous face by
construction, and that is exactly what happened, twice, once at
`t = 0.1149755859375` at `h = 0.00125` and once here.

Both named face policies answer that stage, and so does raising the carried
lift's acceptance margin past two null tolerances with no face policy at all,
which is the existing `--carried-completion-margin-multiple`.  Its rollouts
are the last rows below.

This is a stage-level event that the stored trajectory cannot show: over the
440 stored step endpoints of the carried-completion continuation from
`t = 4`, no endpoint minimum lies in the refinement band at all, the five
smallest being `4.829e-07`, `1.047e-06`, `2.432e-06`, `2.659e-06` and
`3.904e-06`.  The collision is visible only at the trial point of the stage
that refused, which is what (a) now saves.

The collision has a third answer that needs no new code at all: raise the
carried lift's declared acceptance margin past two null tolerances, which the
existing `--carried-completion-margin-multiple` does.  At 30 conic tolerances
(`3e-08`, above the refusal band's upper edge `2e-08`), with the committed
face policy and nothing else changed, the carried-completion rollout reaches
`t = 0.5` at `h = 0.00125` in `56.9 s` with no face refusal at all, 9 stage
retractions, 1,175 of 1,204 stage lifts carried, all-coordinate scalar RMS
`1.287512e-05` over `t = 0` to `0.05` (the same figure as every other row)
and `1.043702e-03` over `t = 0` to `0.5` against the face policies'
`9.547375e-04`.  Artifact:
`output/local_runs/paper_v_apcm_projective_carried_margin30_t05_h000125_20260915_v1/`.

### The named options over the post-pulse interval

Every row: the same fixed-dictionary APCM with the event-triggered one-shell
outer guard at its declared floors, SSPRK(3,3), `h = 0.0025`, 15 entrance
keys, phonon cutoff 16, strong coupling, continued from the same stored
`t = 4` state toward `t = 8` under a 2,400 s wall budget, scored against the
exact cutoff-16 driven reference over the horizon each reached.  The budget
is tested between steps, so a step that enters the subdivision cascade can
overrun it; none of these did.  "RMS" is the all-coordinate scalar RMS with
the inherited `t = 4` offset subtracted.

| named option | reached `t` | steps | RMS over the horizon reached | stage retractions | wall s | where it stopped |
|---|---:|---:|---:|---:|---:|---|
| declared (the committed path) | 5.085 | 434 | `4.384520e-02` | 122 | 250.5 | completion not certified at the retracted point |
| PSD inflation `logdet_shift = 1e-8` | 5.1025 | 441 | `4.458225e-02` | 216 | 1100.0 | the same refusal, seven steps later |
| carried completion | 5.0975 | 439 | `4.388984e-02` | 3 | 103.9 | the numerical face of the extension, not the completion |
| proximal weight `logdet_weight = 1e-2` | 4.83 | 332 | `3.912952e-02` | 25 | 147.4 | a conic certificate at `relative_lambda = 1.271e-07` |

The RMS column above is over each run's own horizon and is therefore not a
ranking of accuracy.  Over the common horizon `t = 4` to `4.83`, which all
four reach, the dynamic all-coordinate scalar RMS is `3.254739e-02` for the
committed path, `3.254739e-02` for the PSD inflation (the same to every
printed digit), `3.227387e-02` for the carried completion and `3.912952e-02`
for the proximal weight `1e-2`.  All four inherit the same `t = 4` error, of
Euclidean norm `2.5037e-01` in the 31 closed scalar coordinates.

Readings.

1. **None of the three carries the rollout past `t = 6`.**  Two of them move
   the wall by five and seven steps out of the 1,200 that `t = 6` would need,
   and the third moves it 102 steps earlier.
2. **Which settings certify is a property of the stage, not of the
   settings.**  At the stage where the declared completion fails
   (`t = 5.08627`), `logdet_shift = 1e-8` certifies in 4 Newton iterations;
   at the stage where the inflated run then fails (`t = 5.10474`) the
   ordering is reversed --- the declared settings certify there with KKT
   backward error `5.86e-09` while `logdet_shift = 1e-8` returns the conic
   Newton descent failure.  The `1e-7` inflation certifies at both.  So the
   refusal is a local conditioning accident of the interior Newton iteration
   at that stage, not a property the trajectory has acquired.
3. **The carried completion is the cheap one, and it changes which
   degeneracy is fatal.**  It reaches essentially the same horizon in
   `103.9 s` against `250.5 s`, with 3 stage retractions against 122 and
   1,307 of 1,351 stage lifts carried, and then stops on the (b) degeneracy:
   `extended operator moment matrix numerical face is unstable under
   null-tolerance refinement`, with 6 refusals recorded.  That is the
   degeneracy (b) has two named answers for, which is what the combination
   rows below test.
4. **The proximal step of larger declared weight is worse than the committed
   path.**  It stops 102 steps earlier, at a lift whose relative minimum
   eigenvalue is `1.271e-07`: the smaller barrier weight moves the minimizer
   toward the boundary, the interior branch is abandoned for the conic
   certificate, and that certificate is what refuses
   (`certificate=Solved; reduced_gradient=7.984e-04`).

Artifacts:
`output/local_runs/paper_v_apcm_projective_t4restart_{default,shift1em8,carried,weight1em2}_t8_20260915_v1/`.

### The combination that survives the post-pulse wall

The three single options each stop at a stage another would have certified,
and the carried completion stops on the face instead of the completion.  The
combination of the two --- the carried lift, and the face carried by
continuation when the refinement is undecided --- is the only configuration
measured here that passes the wall at all.  Same preparation, dictionary,
step and budget as the table above.

| named options | reached `t` | steps | dynamic RMS over the horizon reached | over `t = 4` to `5.085` | stage retractions | wall s | why it stopped |
|---|---:|---:|---:|---:|---:|---:|---|
| carried completion + `face_policy continuation` | 5.235 | 494 | `4.924052e-02` | `4.337435e-02` | 269 | 2400.8 | the 40-minute wall budget, not a refusal |

Readings.

1. **It passes every other option's wall and is stopped only by the
   budget.**  The committed path stops at `t = 5.085`, the PSD inflation at
   `5.1025`, the carried completion alone at `5.0975`; this reaches `5.235`
   with the rollout still healthy (minimum unshifted retained joint-Gram
   eigenvalue `3.0567e-03`, minimum scaled positive-extension eigenvalue
   `3.79e-08`, no triggered guard stage, 1,529 of 2,236 stage lifts carried).
2. **It does not reach `t = 6`, and at this rate it will not.**  It advanced
   `0.235` of time in 2,400 s, so `t = 6` would need about 2.2 hours more at
   the same cost, and the cost is rising: 269 stage retractions over 494
   steps against 3 over 439 for the carried completion alone, and 707 of the
   2,236 carried lifts were declined and fell back to the declared
   completion.
3. **The lift decays toward the refinement band as the run goes on.**  The
   scaled positive-extension minimum at the stored step endpoints is
   `4.829e-07` at worst through `t = 5.0`, `6.415e-06` at `t = 5.1`,
   `2.382e-06` at `t = 5.2` and `3.794e-08` at `t = 5.235`, and the
   retraction count over the same points is 0, 9, 208 and 269.  The run is
   walking into the band the two declared constants share.
4. **The face continuation was used once.**  The face refusal the carried
   completion alone died on is a single stage; after it the run's remaining
   cost is the retraction, not the face.

The same continuation with the declared retry ladder instead of the carried
lift (`--completion-retry-shifts 1e-8 1e-7 --completion-retry-weights 1e-1
--face-policy continuation`) reached `t = 5.12` in 448 steps, also stopped by
the budget, with 109 stage retractions, dynamic all-coordinate scalar RMS
`4.532038e-02` over `t = 4` to `5.12`, minimum unshifted retained joint-Gram
eigenvalue `3.0567e-03` and minimum scaled positive-extension eigenvalue
`2.262e-07`.  Past `t = 5` the ladder stops answering: it certified 115
refusals and was exhausted on 149, and those 149 are what the 109 retractions
repair.  Artifact:
`output/local_runs/paper_v_apcm_projective_t4restart_ladder_facecontinuation_t8_20260915_v1/`.

| named options | reached `t` | steps | dynamic RMS over the horizon reached | stage retractions | wall s | why it stopped |
|---|---:|---:|---:|---:|---:|---|
| carried completion + `face_policy continuation` | 5.235 | 494 | `4.924052e-02` | 269 | 2400.8 | the 40-minute wall budget |
| declared retry ladder + `face_policy continuation` | 5.12 | 448 | `4.532038e-02` | 109 | 2583.8 | the same budget, tested between steps |

### The declared retry ladder removes the retractions, and with them the N block

The retry ladder was built for the post-pulse failure, but it also answers
round 2's attribution of the projective route's accuracy gap.  Over the
recorded horizon `t = 0` to `0.05` at `h = 0.0025`, 15 entrance keys, phonon
cutoff 16, strong coupling, with the ladder
`--completion-retry-shifts 1e-8 1e-7 --completion-retry-weights 1e-1` and
nothing else changed:

| quantity | committed path | with the declared retry ladder |
|---|---:|---:|
| all-coordinate scalar RMS against the exact cutoff-16 reference | `7.931831e-05` | `1.227728e-05` |
| blockwise dynamic rho / B / N / A / C | `5.0153e-07` / `1.5962e-06` / `2.1728e-04` / `9.1036e-06` / `2.0152e-05` | `5.2825e-07` / `1.5896e-06` / `3.1027e-08` / `9.1311e-06` / `1.7241e-05` |
| stage retractions applied | 7 | 0 |
| stage lifts the declared settings did not certify | 7 (retracted) | 6, of which the ladder certified 5 |
| minimum unshifted retained joint-Gram eigenvalue | `2.892080e-05` | `2.892080e-05` |
| autonomous wall time | 54.4 s | 112.5 s |

Artifacts:
`output/local_runs/paper_v_apcm_projective_{facegate_default,ladder_gate}_t005_h0025_20260915_v1/`.

Reading: the seven stage retractions round 2 attributed the N-block error to
were not repairs of an infeasible point.  They fired because the declared
completion returned `success = False` at a lift that was inside the cone, and
containment treats an uncertified completion as infeasible
(`apcm_moment_projection.py:1348-1351` sets the extension minimum to minus
infinity when `completion.success` is false).  Retrying the same lower
moments under a declared alternative certifies five of the six, no retraction
is applied at all, and the N-block scalar RMS falls from `2.1728e-04` to
`3.1027e-08`, which is the value round 2 measured before the first retraction
fired.  The all-coordinate figure `1.227728e-05` at `h = 0.0025` is slightly
better than the carried lift's `1.2875e-05` at `h = 0.00125`, and unlike the
carried lift every stage here carries a certified declared completion.  The
sixth refusal is answered by one step subdivision, and the ladder inside the
retraction was exhausted once.

Step by step, the N-block error of the two runs (Euclidean norm of the four
N coordinates at the stored sample):

| t | with the ladder | committed path |
|---:|---:|---:|
| 0.0100 | `7.7043e-09` | `7.7043e-09` |
| 0.0200 | `2.8770e-08` | `2.2820e-04` |
| 0.0300 | `5.7955e-08` | `4.8932e-04` |
| 0.0400 | `8.8927e-08` | `6.0460e-04` |
| 0.0500 | `1.1680e-07` | `6.6003e-04` |

The two agree to every digit until the committed path's first retraction at
`t = 0.02`, and the ladder's N block then grows smoothly by about `3e-08` per
`0.01` of time instead of stepping.

## Gates

| gate | result |
|---|---|
| the recorded projective path through the driver, `t = 0` to `0.05` at `h = 0.0025`, default everything | every array of `autonomous_trajectory.npz` equal under `np.array_equal` to round 2's `paper_v_apcm_projective_ablation_baseline_recheck_t005_20260915_v1`: times, states, archive coordinates, both eigenvalue series, stage retractions, accepted substeps, triggered stages.  All-coordinate scalar RMS `7.931830738563173e-05`, 7 retractions, joint-Gram minimum `2.892080456917745e-05`, 54.4 s.  Artifact: `output/local_runs/paper_v_apcm_projective_facegate_default_t005_h0025_20260915_v1/` |
| the carried-completion path, `t = 0` to `0.11375` at `h = 0.00125` | the two face policies reproduce the committed path's arrays bitwise over the 91 steps it survives |
| `initial_time` at its default | `test_initial_time_default_reproduces_the_recorded_sample_times_bitwise` |
| the declared flags reach the model that runs | `test_driver_declared_settings_reach_the_model_that_runs` |
| the retry ladder is consulted only on a refusal | structurally (the ladder runs only when the declared solve returns `success = False`), plus `test_completion_retry_ladder_takes_the_first_certified_result` and `test_the_default_stage_policy_declares_no_completion_retry`; the empty-ladder default is what the bitwise row above reproduces |
| the four test files I own, run together | 63 passed in 93.0 s (44 before this round) |
| the whole `paper_5` suite under the 4500 MB guard | 428 passed, 9 warnings in 189.9 s, peak 1664 MB (376 at the end of round 2; the count includes other agents' tests added during the round) |

## Comparability in a shared worktree

Other agents were editing `paper_5/src/paper5/stability/` throughout this
round.  MD5 of the shared modules every number above was produced through, at
the time of the gate run: `apcm_positive_extension.py`
`fc799d8589fb416d665e2c048d5b6498`, `matrix_reference.py`
`e62409317773b7378b97ef5721f7e2bb`, `adaptive_positive_moment.py`
`714cd28ca96a2f9bf72a06e224c132d1`, `apcm_projective_guard.py`
`03efd75c341235afa764d9ee5c07dfd1`, `apcm_carried_witness.py`
`679473f85ff47fe70cbb51fda8696dbc`.  The bitwise gate against round 2's
recorded-path artifact was run under those, so every row here is comparable
with round 1 and round 2.  `apcm_carried_witness.py` moved again later in the
round (`679473f8...` to `ad545fab...`); `projected_apcm.py` reads one symbol
from it, `negative_mode_coupling`, which enters only the outer guard's
trigger measure, and no stage in any run reported here triggered.

Wall times in this round were measured with three to four other agent
processes running on the same machine (load average about 8 during the
`t = 4` continuations), so they are upper bounds and are not comparable with
round 2's wall times on a quieter machine.  Peak resident memory of every run
here is under 360 MB, and the aggregate across agents stayed under 3 GB.

## Kept

- The failure-path save.  A failed projective rollout now writes the partial
  trajectory, the exact reference over the horizon it reached, the accuracy
  metrics, the last accepted state, and the failing stage's state, lower
  moments, frontier moments and message.  This is not a named option: it adds
  artifacts on a path that previously wrote none and changes no number on the
  path that succeeds, which the bitwise gate shows.
- `ProjectedAPCMStageFailure`, `projected_apcm_stage_failure`,
  `projected_apcm_partial_trajectory`, `apcm_retracted_stage`: the payloads a
  diagnosis needs, attached to the exception and readable by a caller.
- `initial_time` on `integrate_projected_apcm_ssprk3` and
  `--initial-state-file` / `--initial-time` on the driver, which is what made
  the post-pulse degeneracy reachable in four minutes instead of an hour.
- `face_policy="continuation"` and `face_policy="inflated"` as named options,
  both off by default.  Either carries the carried-completion rollout from
  `t = 0.11375` to `t = 0.5`; they give the same trajectory here.
- `--completion-logdet-shift` and `--completion-logdet-weight` as named
  options whose defaults are the declared `1e-10` and `1.0`.
- The declared completion retry ladder (`completion_retry_shifts`,
  `completion_retry_weights`, empty by default), which is what the
  three-stage sweep points at: no single setting certifies everywhere, and
  each stage that one setting refuses is certified by another.  It changes
  nothing at a stage the declared settings certify.

## Dropped

- Raising the log-determinant barrier weight.  At the failing stage both `10`
  and `100` leave the solve uncertified.
- `logdet_shift = 1e-6` and `logdet_weight = 1e-3`: both push the lift below
  the declared interior threshold `max(0.1 * conic_tolerance,
  10 * logdet_shift)` and the solve falls to the conic certificate, which
  returns KKT backward errors `1.89e-03` and `3.08e+01`.
- Combining `logdet_weight = 1e-2` with `logdet_shift = 1e-8`: uncertified
  (`3.34e+00`) where each alone certifies.

## Blockers

1. **The post-pulse completion degeneracy is not removed by any single
   declared setting.**  Each of the three named options moves the wall by a
   few steps in either direction and each fails at a stage where one of the
   others would have certified.  What the measurement points at instead is a
   declared ladder --- try the declared settings, and on a refusal retry the
   same lower moments from the same warm start under declared alternatives,
   taking the first certified result --- which is implemented here as
   `completion_retry_shifts` / `completion_retry_weights`, empty by default.
   Its rollout is reported above.
2. **A carried lift carries no certificate.**  At 11 step endpoints sampled
   evenly over the carried-completion rollout to `t = 0.5`, the declared
   completion solved cold at those lower moments certifies only 6, the other
   five returning KKT backward errors `1.00e+00`, `2.03e-07`, `3.41e-07`,
   `2.63e-07` and `3.56e-07` against the acceptance `5e-08`.  The carried
   option accepts a lift that is positive semidefinite inside the frozen
   envelope, which is a weaker statement than the declared selector's, and
   the record should say so wherever a carried-lift number is quoted.
3. **The projective route's accuracy at `t = 0.5` is a factor 30 behind the
   carried-witness flow** (`9.547375e-04` against `3.19169e-5`), so the
   horizon the face options buy is not yet an accuracy result.  Whether the
   projective route is worth carrying past the pulse is a question for the
   `t = 4` line, not for `t = 0.5`.

## What a next round should do first

1. Run the declared completion ladder over the full driven interval `t = 0`
   to `4` and compare it with the recorded `0.026049`, since the ladder is
   the only option measured here that answers a refusal without changing the
   lift the run uses when the declared settings certify.
2. Decide whether the carried lift's acceptance margin and the face
   refinement band should be allowed to overlap at all.  That is a question
   about two declared constants (`carried_completion_margin_multiple = 10`
   against `null_tolerance_factor = 10` on the same `conic_tolerance`), and
   moving either is a scientific acceptance change, so it is the user's.
3. Score the projective route past the pulse against the exact reference at
   `t = 6` and `t = 20` only after one of these carries a rollout there; at
   present no option reaches `t = 6`, and the best horizon past `t = 4` is
   `t = 5.235`, reached by the carried lift with the face carried across
   stages and stopped by the 40-minute budget rather than by a refusal.
4. One run was stopped deliberately rather than by its budget: the `t = 4`
   continuation at `--carried-completion-margin-multiple 30`
   (`output/local_runs/paper_v_apcm_projective_t4restart_carried_margin30_t8_20260915_v1/`,
   `progress.jsonl` only, last printed `t = 4.8`).  It is the row that would
   say whether the margin alone carries the post-pulse rollout as far as the
   face policy does.
