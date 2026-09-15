# review-joint-gram-t20 (2026-09-15)

Independent review of the round-3 implementer `joint-gram-t20`. Nothing was
edited; every check below is a rerun, a re-solve, or a read of the stored
arrays.

Verdict: **fail**, on the report's central negative result. The code, the
scope, the tests and the default path are all clean, and the wall-clock,
step-size and reference-line numbers all reproduce. What fails is the
comparison the report leads with: the joint-Gram lane is built with
`subtract_initial_residual=False`, so the uncorrected velocity it corrects is
the **plain** archive EOM (`closed_scalar_rhs`), not the Eq. (112)
residual-subtracted archive EOM the driver carries in its `raw` lane and the
report calls "the right-hand side it corrects". Against its own base velocity
the barrier is slightly better, not two orders of magnitude worse, and the
barrier census reported for the lane was solved at a velocity the lane never
used.

## Scope

`git status --short` lists twelve modified and eight untracked files; ownership
from each round-3 report's own "files owned" section:

| file | owner |
|---|---|
| `paper_5/src/paper5/stability/cone_long_horizon.py` | joint-gram-t20 (owned) |
| `paper_5/tests/test_cone_long_horizon.py` | joint-gram-t20 (owned) |
| `pipelines/open_dynamics/run_paper_v_energy_neutral_barrier_long_horizon.py` | joint-gram-t20 (owned) |
| `agent_guidance/paper-v/overnight/joint-gram-t20-20260915.md` | joint-gram-t20 (owned, new) |
| `paper_5/src/paper5/stability/apcm_carried_witness.py`, `apcm_carried_witness_analysis.py`, `paper_5/tests/test_apcm_carried_witness.py` | bundle-rank-seed |
| `paper_5/src/paper5/stability/projected_apcm.py`, `apcm_moment_projection.py`, `apcm_projective_event_guard_analysis.py`, `paper_5/tests/test_projected_apcm.py`, `test_apcm_moment_projection.py`, `test_apcm_projective_event_guard.py` | projective-face |
| `paper_5/src/paper5/stability/apcm_outer_selector_formulations.py`, `paper_5/tests/test_apcm_outer_selector_formulations.py` (new) | outer-selector-formulation |
| the remaining untracked `agent_guidance/paper-v/overnight/*.md` | the agent each is named for |

No scope violation. The diff of the nine files owned by the other three agents
matches `joint_gram`, `corrected_controller`, `integrate_sampled_dop853`,
`cone_long_horizon` and `step_callback` zero times. `cone_correction.py`,
`matrix_reference.py` and `exact_reference.py` — the three this task must not
edit — are unmodified. Every written path is inside the section-7 allowlist,
and the main checkout carries none of the three files and nothing under
`paper_5/`, `pipelines/` or `agent_guidance/` written after 2026-09-14 12:00.
One note, not a violation: the implementer's scorer also wrote
`matched_tables_record_convention.json` into round 2's artifact directory
`output/local_runs/paper_v_boson_barrier_t100_20260915_v1/`; the path is inside
the allowlist and the write is additive.

## Tests, rerun

| gate | reported | this review |
|---|---|---|
| `paper_5/tests/test_cone_long_horizon.py` | 19 passed (11 before, 8 added) | 19 passed in 4.12 s; `HEAD` carries 11 test functions and no parametrization, the working tree 18 with one two-case parametrization |
| `paper_5/tests/test_cone_correction.py` | 29 passed, unmoved | 29 passed (48 together with the file above in 10.36 s); the file is unmodified against `HEAD` |
| whole `paper_5` suite under `ram_guard` | 428 passed in 160.91 s, peak 1619 MB | 428 passed, 9 warnings, 162.70 s, peak 1968 MB at `--limit-mb 3000` |

The byte-equality tests are real gates: both parametrized cases build the same
right-hand side twice and assert `np.array_equal` on times and states and
equality of `nfev` against `solve_ivp`.

## Default path

Bitwise. The `HEAD` driver and the working-tree driver were both run at
`--final-time 2` with every other argument at its default, the `HEAD` copy
taken from `git show HEAD:` and run from a scratch directory against this
worktree's `paper5`:

| check | result |
|---|---|
| `corrected_trajectory.npz`, `raw_trajectory.npz`, `exact_trajectory.npz` | every array `np.array_equal` (`times`, `states`, `observables`, `internal_energies`, `boson_moment_minima`, `electron_minima`, `initial_state`) |
| `summary.json` shared keys | identical except the two `wall_seconds` fields |
| `summary.json` new keys | 22, all under the additive `corrected_lane` block (controller, right-hand side, correction, settings, resolved half-space backend, `on_rhs_error`, wall budget, solver counters) |
| comparison and census window keys | identical (`t_0_to_2`, `t_0_to_20`) |

`integrate_sampled_dop853` keeps `step_callback=None` and
`on_rhs_error="raise"`, the new `try`/`except` re-raises unless the named
policy is set, and `closed_cone_projected_rhs` is called with
`halfspace_backend=None`, which resolves to `DEFAULT_HALFSPACE_BACKEND`
(`cone_correction.py:41`). One default-path behaviour is not behind a flag:
when a lane stops before the requested window's stop time, the driver now also
scores and censuses the window every lane reached. Measured on a forced early
stop (`--corrected-wall-budget-seconds 0.05`, `--final-time 2`), the effect is
one additional key (`t_0_to_0.9`) in `comparisons` for each lane; no existing
key changes. It is additive reporting, but it is new default behaviour without
a named option and it carries the duplicate census the implementer discloses.

## Numbers that reproduce

Recomputed from the stored arrays with my own scorer (`np.load`, then
`sqrt(mean(difference**2))` and `sqrt(trapezoid(l2**2, t) / duration)`), not
with the implementer's script:

| quantity, `t = 0` to 0.30, 7 samples of the 0.05 grid | reported | this review |
|---|---:|---:|
| joint-Gram `slsqp` lane, `coordinate_scalar_rms` / `coordinate_l2_time_rms` | `0.01184757` / `0.06420598` | `0.011847574707288862` / `0.06420598113399795` |
| Eq. (112) archive EOM, same two | `0.00013435` / `0.00065244` | identical |
| round-2 ten-coordinate barrier, same two | `0.00013435` / `0.00065244` | identical, from round 2's own `corrected_trajectory.npz` |
| maximum absolute coordinate difference between the two backends, and relative to the largest coordinate of the default lane | `1.667e-8` / `1.361e-8` | `1.6667620561516827e-08` / `1.3609081211055076e-08`, per-sample Euclidean distance rising monotonically to `4.081e-8` |

Round 2's three lanes over `t = 0` to 20, 401 samples, in the record's
`coordinate_scalar_rms` convention: ten-coordinate barrier `0.16586508`,
Eq. (112) `0.18406451`, plain archive EOM `0.19325182`; `coordinate_l2_time_rms`
`0.92394673`, `1.02515085`, `1.07566340`. All six match the report. The
definitional finding therefore stands: the record's `0.193205` uncorrected line
over `t = 0` to 20 is the plain archive EOM in that convention. The exact
cutoff-16 lanes of the two run directories are bitwise equal on the shared
samples, so these cross-run comparisons are exact.

The step-size table reproduces to every printed digit from
`step_trace_slsqp.jsonl` (counts 4 / 56 / 230 / 124 / 142 / 217 / 643 / 1,004 /
1,107; means `1.2425e-2` down to `9.0278e-6`; first accepted step below `1e-4`
at `t = 0.109299` step 93, below `1e-5` at `t = 0.114340` step 125, below
`3e-6` at `t = 0.211653` step 1,562). The horizon-per-wall-minute figures
reproduce from the 149 progress lines of `slsqp_driver.log`
(`0.01440` / `0.00500` / `0.00120` / `0.00070` / `0.00060`), as do
`650,243`, `26,324,792`, `46,034`, `688,322`, `28,457,232`, `48,878`, the
active-set split `688,322` / `25,413,882` / `2,355,028`, and the derived
`0.01384` and `0.01308` wall seconds per right-hand-side evaluation. The
round-2 substitute values also reproduce from round 2's artifacts: post-pulse
maximum drift from `E(4)` `2.5390e-6` and `2.4156e-6`, barrier activity
`0.551122` at maximum correction norm `0.59424` over `t = 0` to 20, and the
polarization quadruple `0.2610` / `2.5364` / `0.1916` / `0.1186` against the
exact lane's `0` / `2.3421` / `0.1264` / `0.0729` over `10 <= t <= 79.8`.

## Blocking issues

### 1. The lane is not compared against the velocity it corrects

`closed_electron_phonon_cone_projected_rhs` is called with
`subtract_initial_residual=False` (the pinned round-1 setting, faithfully
copied), so its base velocity is `closed_scalar_rhs` — the plain archive EOM
(`cone_correction.py:2432-2441`). The driver's `raw` lane is
`closed_residual_subtracted_rhs`, the Eq. (112) archive EOM. These are
different trajectories, as the report's own definitional finding says, and the
initial-residual vector they differ by is not small: l2 norm `4.0969e-1`,
largest entry `1.7299e-1`.

Propagated from the same exact cutoff-16 preparation on the same 7 samples of
the 0.05 grid over `t = 0` to 0.30 (my own integration, reproducing round 2's
stored plain lane to every printed digit):

| lane over `t = 0` to 0.30 | `coordinate_scalar_rms` | `coordinate_l2_time_rms` |
|---|---:|---:|
| joint-Gram energy-neutral barrier, `slsqp` | `0.01184757` | `0.06420598` |
| plain archive EOM, `closed_scalar_rhs` — the velocity this lane corrects | `0.01328016` | `0.07152712` |
| Eq. (112) archive EOM — the driver's `raw` lane | `0.00013435` | `0.00065244` |

So the barrier is about 11 percent closer to exact than the trajectory it
actually repairs, not 98 times further from it. Negative result 3, the
"Reading" paragraph, and the structured summary's headline all have to be
rewritten around the middle row; the Eq. (112) row stays as what it is, a
different closure that happens to be accurate this early.

### 2. The census reported for the joint-Gram lane was solved at the wrong velocity

The driver passes `raw_rhs` (Eq. 112) as the `base_rhs` of
`joint_gram_barrier_activity` for the joint-Gram lane
(`run_paper_v_energy_neutral_barrier_long_horizon.py`, the
`corrected_controller == "joint_gram_barrier"` census branch), so the
correction is re-solved from a velocity the lane never used. Re-solving the
same census at the lane's own velocity on the same 7 stored samples:

| census of the corrected lane's own samples, `t = 0` to 0.30 | at the Eq. (112) velocity (reported) | at the plain velocity (the lane's own) |
|---|---:|---:|
| active sample fraction | `0.857143` (6 of 7, first at `t = 0.05`) | `1.0` (7 of 7, first at `t = 0`) |
| maximum correction norm | `0.0646506` | `0.2501716` |
| maximum constraint count | `19` | `51` |
| minimum joint moment eigenvalue | `2.067511e-5` | `2.067511e-5` (a property of the state, unaffected) |

The same census on the plain archive EOM trajectory itself, at its own
velocity, gives active fraction `1.0`, maximum correction norm `0.3954804` and
minimum joint moment eigenvalue `-2.2845e-2` over `t = 0` to 0.30. The report's
supporting sentence "over that window the uncorrected trajectory needs no
repair: its minimum joint moment eigenvalue is `2.8921e-5` and the correction
... has norm at most `7.97e-6`" is therefore about the Eq. (112) lane, not
about the trajectory the barrier corrects, which is outside the joint cone
(`-2.2845e-2`) over exactly that window. The default ten-coordinate lane is
unaffected: it runs `closed_cone_projected_rhs` with the default
`subtract_initial_residual=True` and is censused with the matching `raw_rhs`,
so the mismatch is new to the joint-Gram branch. Either pass a base velocity
that matches the lane's own `subtract_initial_residual`, or record in the
summary which velocity the census used.

### 3. The headline ratio is wrong and mixes conventions

"the barrier is 88 times less accurate than the right-hand side it corrects"
does not follow from the two numbers in the same sentence: `0.06420598 /
0.00065244 = 98.4`, which is what the structured summary says. `88` is the
ratio of the `coordinate_l2_sample_rms` pair (`0.06596450 / 0.00074805 =
88.2`), a third convention, in a sentence that quotes the time-RMS pair. Both
values also have to change with issue 1.

## Non-blocking issues

1. Both production runs were executed with `--corrected-on-rhs-error stop`
   (`summary.json`, `corrected_lane.on_rhs_error = "stop"` in both lanes), and
   neither the run description nor the gate table says so; the reported
   commands would not reproduce the recorded configuration. Neither lane
   exercised the path — both stopped with `stop_reason = "wall_budget"`.
2. "Half-space projections per accepted step stay between 468 and 734
   throughout" is not reproducible from the artifacts: over the 148 intervals
   of the run's own 60-second progress trace the incremental ratio runs from
   `404.0` to `768.1`, and by interval of `step_trace_slsqp.jsonl` the mean
   runs from `290` to `492`. The bracket falls out only under an unstated
   block aggregation (15 to 25 progress intervals). The conclusion the number
   supports — flat per-step work against a factor of about 1,400 in the mean
   accepted step — survives either way; state the definition with the bracket.
3. `summary.json` `protocol.corrected_derivative_coordinates` reads `[7, 17]`
   in both joint-Gram runs. That is the ten-coordinate boson barrier's
   derivative slice; the joint-Gram lane corrects all 31. The `corrected_lane`
   block names the controller correctly, so the two blocks of the same file
   disagree.
4. "both lanes carry the same minimum bosonic eigenvalue `2.068516e-5` to
   seven digits" overstates the agreement: `2.0685171e-5` (`slsqp`) against
   `2.0685163e-5` (`active_set`), equal to six significant digits. The
   report's own table value `2.06852e-5` is right.
5. New default behaviour without a named option (scoring and censusing the
   matched window) — see the default-path section. Additive only, and it
   carries the duplicated census the implementer already discloses.

## What I did not have to re-derive

The measurement the task asked for stands on its own: 150 minutes of the
default backend reaches `t = 0.30` and of the named `active_set` backend
`t = 0.35`, against `t = 20` requested; neither reaches the post-pulse window
`t >= 4` or the spectral window `10 <= t`, so the maximum post-pulse drift from
`E(4)` and the polarization spectrum are unavailable on both lanes, and the
record's both-cones `t = 0` to 20 line cannot be put in a column beside them.
The step-size collapse is documented from the run's own trace. Issues 1 and 2
do not touch any of that; they touch what the barrier is compared with, and
what the barrier was doing while it ran.
