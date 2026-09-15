# Cheap physicality repair with no conic solve (2026-09-15)

Branch `claude/paper-v-routes-20260914`, worktree
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`.
New modules `paper_5/src/paper5/stability/cheap_projection_route.py` and
`cheap_projection_route_analysis.py`; no existing file edited; nothing
committed; no default changed anywhere.

Question put to this route: the record's guarded routes pay `10^3` to `10^4 s`
per time unit because every stage re-solves a cone, while the exact cutoff-16
reference propagates to `t=4` in `0.7 s`.  Propagate with the unguarded
velocity instead, and repair physicality by linear algebra alone --
eigenvalue clipping to a declared floor, Higham nearest positive
semidefinite, or a declared number of Dykstra alternating projections between
the affine Gram set and the positive semidefinite cone -- with no
certificate, and report the physicality minima honestly next to the score.

## 0. What makes the repair possible

Both moment matrices the record reports are exactly affine in the propagated
coordinates, which the module verifies at build time against the public
functions themselves:

| matrix | rows | affine in | build |
|---|---:|---|---:|
| retained joint Gram, `uncentered_joint_moment_matrix` | 8 | the 29 raw coordinates | `3 ms` |
| 24-row positive extension, `SymmetryReducedPositiveExtension.matrix` composed with `state_lower_moments` | 24 | the 44 fixed-dictionary state coordinates and the 62 standardized frontier coordinates | `4 ms` |
| 62-row unified Gram, `CarriedWitnessGeometry.scaled_unified_matrix` | 62 | all 384 carried-witness coordinates | `207 ms` |

Reconstruction error of the affine map against the public function at a
random probe: `1.1e-16`, `1.1e-16` and `2.8e-17` absolute.  So a repair is:
eigendecompose, clip, and map the corrected Gram back to coordinates by one
least-squares solve against a pseudo-inverse factored once per rollout.

Two structural facts about the 24-row map, measured:

- its design matrix has rank 91 of 106 parameters, with a clean gap
  (`1.573e-1` then `4.2e-16`).  The 15 unseen directions are all
  propagated-state directions, so the declared `state_motion_scale` and
  `frontier_motion_scale` are inert: the weighted and unweighted
  least-squares repairs of the same residual agree to the eleventh digit
  (`t=0` to `4`, `frontier_motion_scale` 100 against 1: all-coordinate
  scalar RMS `0.13133511462184333` against `0.1313351146218195`, maximum
  state displacement `5.246846339128665e-3` against
  `5.2468463391238205e-3`).
- one eigenvalue clip plus one least-squares map back does not close the
  gap, because the clipped matrix leaves the affine Gram set: on a perturbed
  preparation state the 24-row minimum moves `-3.145e-4` to `-2.113e-4`
  (33 percent) for the clip and Higham policies, `-9.850e-5` at 3 Dykstra
  iterations, `-7.317e-6` at 10, `-4.552e-9` at 30, and `-2.9e-16` at 73,
  where Dykstra stops on its own.  Cost of that repair: `0.31 ms` for the
  clip, `1.47 ms` at 10 iterations, `3.99 ms` at 30, `10.09 ms` at 73.

## 1. Fixed-dictionary state (44 coordinates, 24-row extension)

Configuration for every row: strong coupling (`lambda_ep = 1.5`,
`gamma = 0.5`, drive amplitude 1), phonon cutoff 16, exact-correlated
preparation, 15 entrance keys, SSPRK(3,3) at `h = 0.0025`, the unguarded
fixed-dictionary predictor (archive equations plus the exact K/P/D entrance
correction plus the exact active commutators), the lift carried from the
preparation's proximal log-det completion and never re-solved.  The
containment, the stage retraction and the tangent-cone projection never run.
Scores are the all-coordinate scalar RMS on the 31 closed scalar coordinates
against the exact cutoff-16 driven reference, the same helper the
carried-witness driver uses.

### `t = 0` to `0.05`, 20 steps

| policy | all-coordinate scalar RMS | N block | minimum scaled 24-row extension eigenvalue | minimum unshifted retained joint Gram | repairs applied / skipped | autonomous wall |
|---|---:|---:|---:|---:|---:|---:|
| none (control) | `1.272e-5` | `2.923e-8` | `-1.645e-11` | `2.892e-5` | 0 / 60 | `0.141 s` |
| `eigenvalue_clip`, extension, floor 0 | `1.272e-5` | `2.923e-8` | `-5.308e-12` | `2.892e-5` | 46 / 14 | `0.154 s` |
| `higham_nearest_psd`, both | `1.272e-5` | `2.923e-8` | `-5.308e-12` | `2.892e-5` | 54 / 6 | `0.161 s` |
| `dykstra` 5, both | `1.272e-5` | `2.923e-8` | `-6.938e-13` | `2.892e-5` | 46 / 14 | `0.198 s` |
| `dykstra` 30, both | `1.272e-5` | `2.923e-8` | `-4.853e-16` | `2.892e-5` | 46 / 14 | `0.296 s` |
| `dykstra` 30, both, per-step cadence | `1.272e-5` | `2.923e-8` | `-5.812e-16` | `2.892e-5` | 15 / 5 | `0.198 s` |

Record lines on the same horizon: carried-witness strict always-on guard
`3.236e-6`; the projective route with the declared completion retry ladder
`1.2277e-5` at `112.5 s`; the projective route with the stage retraction
disabled `1.9408e-6` over `t=0` to `0.015` before its per-stage completion
solve refuses.

Reading: over `t=0` to `0.05` the state never leaves either cone by more than
`1.6e-11`, so every policy scores identically to four digits, including the
control.  The number that matters is the control's: the unguarded
fixed-dictionary velocity with a frozen lift and no repair at all scores
`1.272e-5` in `0.141 s`, which is the recorded projective retry-ladder score
`1.2277e-5` at `112.5 s`.  The repair is free and, here, does nothing.

### `t = 0` to `4`, 1,600 steps

| policy | all-coordinate scalar RMS | minimum scaled 24-row extension eigenvalue | minimum unshifted retained joint Gram | maximum standardized frontier envelope ratio | autonomous wall |
|---|---:|---:|---:|---:|---:|
| none (control) | `0.13304` | `-1.5159` | `-1.5578` | `0.4216` | `11.03 s` |
| `higham_nearest_psd`, both | `0.13460` | `-2.2815e-3` | `-2.3217e-4` | `0.4458` | `13.06 s` |
| `dykstra` 30, both | `0.13134` | `-1.0431e-5` | `-7.9239e-7` | `0.4487` | `39.44 s` |
| `dykstra` 30, extension, `frontier_only` | `0.13489` | `-0.9211` | `-2.4067` | `4.874` | `31.81 s` |

Record lines on the same horizon: projective fixed-dictionary APCM with the
event-triggered outer guard `0.026049` at `3,229.8 s`, retained joint Gram
positive to `2.892e-5`; raw 31-coordinate archive EOM `0.11090`; carried
moments with no Gram guard `0.48049` at `153.53 s`.

Three readings.

- The repair buys physicality and nothing else.  Dykstra-30 takes the
  24-row extension from `-1.5159` to `-1.0431e-5` and the retained joint
  Gram from `-1.5578` to `-7.9239e-7`, five and six decades, while the score
  moves from `0.13304` to `0.13134`, that is by 1.3 percent.
- The route is 5.0 times less accurate than the projective route at 1/82 of
  its cost, and 1.18 times less accurate than the raw archive EOM.
- Repairing the lift alone is harmful, not merely weak: `frontier_only`
  cannot make the 24-row Gram positive (`-0.9211` against the control's
  `-1.5159`), leaves the retained joint Gram worse than the control
  (`-2.4067` against `-1.5578`, because the lift no longer constrains it),
  and pushes the standardized frontier coordinates to 4.87 times their
  frozen envelope.  The negativity lives in the propagated lower moments,
  not in the lift.

### Why `0.131` and not `0.026`: the frozen lift, measured

At six times along the `dykstra` 30 trajectory the carried lift was compared
with the declared proximal log-det completion solved at the same lower
moments (one solve each, offline, not in the loop):

| `t` | declared completion wall | standardized frontier distance, carried against solved | relative difference of the 15 entrance-moment commutator velocities | relative difference of the 29 retained velocities |
|---:|---:|---:|---:|---:|
| 0.1 | `1.342 s` | `0.4743` | `1.0090` | `0.0` |
| 0.5 | `0.877 s` | `0.0378` | `0.3662` | `0.0` |
| 1.0 | `1.150 s` | `0.9006` | `0.5200` | `0.0` |
| 2.0 | `0.832 s` | `0.0924` | `0.4926` | `0.0` |
| 3.0 | `1.027 s` | `0.1140` | `0.5127` | `0.0` |
| 4.0 | `0.876 s` | `0.1587` | `0.4856` | `0.0` |

The retained velocity reads only lower moments and is unaffected exactly.
The entrance moments' commutator velocity reads the lift and is 37 to 101
percent different from the one the declared completion would give, at every
sampled time from `t=0.1` on.  That is the accuracy limiter of this route,
and it is not a physicality defect: re-selecting the lift costs `0.83` to
`1.34 s` per solve, three per step, which is the projective route's
`3,230 s`.

### `t = 0` to `0.015`, the horizon the record's no-retraction ablation reached

`dykstra` 30, both, `h=0.0025`, 6 steps: all-coordinate scalar RMS
`1.7231e-6`, blockwise rho `1.698e-10`, B `4.903e-7`, N `4.282e-9`, A
`2.820e-6`, C `1.760e-6`, minimum scaled 24-row extension eigenvalue
`-4.853e-16`, minimum unshifted retained joint Gram `2.892e-5`, `0.057 s`.
The record's projective route with the stage retraction disabled scores
`1.9408e-6` with N block `4.26e-9` on exactly this horizon and then stops
after 6 of 20 steps because its per-stage completion solve refuses.  The
cheap route is 1.13 times more accurate there and does not refuse, because
it never asks a cone solver anything.

### `t = 0` to `0.5`

`dykstra` 30, both, `h=0.0025`: all-coordinate scalar RMS `2.0130e-3`,
minimum scaled 24-row extension eigenvalue `-2.977e-7`, minimum unshifted
retained joint Gram `-2.890e-9`, `4.59 s`.  Record lines: carried-witness
no-guard anchor `3.19169e-5` at `9.40 s`; the projective route with the
carried lift and face continuation `9.5474e-4` at `h=0.00125`.

## 2. Carried-witness state (384 coordinates, 62-row unified Gram)

The record's own no-guard CWRMF flow (`enforce_gram_guard=False`, which
removes the 62-row positive-semidefinite optimization and the endpoint
rejection together), stepped one step at a time through the record's
`integrate_cwrmf_ssprk2`, with the declared cheap repair applied to the
62-row unified Gram after every accepted step.  Same preparation, strong
coupling, cutoff 16, `h=0.0025`.

| quantity, `t=0` to `4`, 1,600 steps | no repair (control) | `dykstra` 30 after every step | `dykstra` 200 after every step |
|---|---:|---:|---:|
| all-coordinate scalar RMS against the exact cutoff-16 reference (31 closed scalar coordinates) | `0.48048` | `0.18430` | `0.18409` |
| site-0 occupation RMS | --- | `6.995e-3` | `7.157e-3` |
| minimum scaled 62-row unified Gram eigenvalue | `-26.4455` | `-3.6615e-5` | `-4.2461e-7` |
| minimum scaled 62-row unified Gram eigenvalue before each repair | `-26.4455` | `-6.4918e-4` | `-6.4041e-4` |
| minimum unshifted retained joint Gram eigenvalue | `-17.6151` | `-2.499e-16` | `-2.499e-16` |
| autonomous wall (propagation / repair) | `26.01 s` (`25.22` / `0`) | `69.32 s` (`26.21` / `43.06`) | `300.76 s` (`26.52` / `274.18`) |
| mean repair | --- | `26.9 ms` | `171.4 ms` |
| maximum retained / completion coordinate displacement of one repair | --- | `3.558e-3` / `6.410e-4` | `3.554e-3` / `6.339e-4` |
| peak resident | `718 MB` | `751 MB` | `751 MB` |

Raising the Dykstra budget from 30 to 200 iterations buys two decades of
physicality on the 62-row Gram (`-3.66e-5` to `-4.25e-7`) and 0.1 percent of
score, at 4.3 times the wall time, so 30 is the operating point.

The control reproduces the record's no-guard line in every printed digit
(record: `0.48049`, `-26.4455`, `-17.6151`), so the per-step call pattern is
the recorded propagation.

Reading: the cheap repair carries the no-guard carried-witness flow through
the full driven interval with the 62-row unified Gram held at `-3.7e-5` and
the retained joint Gram at `-2.5e-16`, and takes the score from `0.48048` to
`0.18430`, in `69 s`.  The record's guarded alternatives on this same flow
are `43,692 s` for `t=0` to `0.5` (strict always-on) and `14,722 s` for
`0.38` time units (balanced, event-triggered, hard floor `2e-4`), neither of
which has ever reached `t=4`.  Against the raw 31-coordinate archive EOM
`0.11090` this is still 1.66 times worse, and against the projective
event-guard route's `0.026049` it is 7.1 times worse at 1/47 of that route's
cost.

## 3. Verdict

The cheapest repair is real and it works as a *containment*: three decades to
six decades of physicality on both moment matrices, at `0.3` to `27 ms` per
repair, with no conic solve, no certificate and no refusal anywhere.  It is
not an accuracy repair on either state: on the fixed dictionary it moves the
score by 1.3 percent, and on the carried witness the 2.6-fold improvement
comes from keeping the trajectory representable rather than from the
projection being a better closure.

What the measurement settles:

- A cone repair does not have to be a cone solve.  The record's `10^3` to
  `10^4 s` per time unit buys a certificate, not the physicality number.
- On the fixed dictionary the accuracy limiter is the frozen lift, measured
  at 37 to 101 percent of the entrance-moment velocity, not the cone.
- Repairing the lift while holding the propagated state fixed is worse than
  no repair at all.

The one number Paper V did not have, plan step 6 -- a moment closure that
reaches `t=4` with its Gram at the cone -- now exists at `69 s`: the
carried-witness flow with a Dykstra repair, `0.18430` at 62-row Gram
`-3.7e-5`.  It is 1.66 times worse than the raw archive EOM, so it does not
yet close that step; the projective event-guard route does close it
(`0.026049`) at 47 times the cost.

## 4. Provenance and gates

- `paper_5` suite on the tree these runs were made on: 458 passed
  (`cd paper_5 && PYTHONPATH=src python -m pytest -q`), including the 16 new
  tests in `paper_5/tests/test_cheap_projection_route.py`.
- No existing module was edited.  During this session another agent working
  in the same worktree advanced `HEAD` from `d1f4eb83` to `5f976077` and left
  uncommitted additions in `projected_apcm.py`,
  `apcm_projective_guard.py`, `apcm_projective_event_guard_analysis.py` and
  `test_apcm_projective_event_guard.py`.  Every one of those additions to
  `projected_apcm.py` lies after line 1003, that is inside
  `ProjectiveOuterEventGuardSettings`, a new `resolve_outer_target_selector`
  and `__all__`; this route uses only `FixedDictionaryProjectedAPCM`
  (lines 487 to 1002), `prepare_projected_apcm_initial_state`,
  `unpack_projected_apcm_state` and `ProjectedAPCMSettings`, none of which
  changed, so the numbers above are unaffected.  Nothing here is committed.
- Artifacts, all under `output/local_runs/`:
  `paper_v_cheap_projection_{none,clip_ext,higham_both,dykstra5_both,dykstra30_both,dykstra30_ext,dykstra30_step}_t005_20260915_v1`,
  `paper_v_cheap_projection_dykstra30_both_t0015_20260915_v1`,
  `paper_v_cheap_projection_dykstra30_both_t05_20260915_v1`,
  `paper_v_cheap_projection_{none,higham_both,dykstra30_both,dykstra30_frontier100,dykstra30_frontieronly}_t4_20260915_v1`,
  `paper_v_cheap_projection_cw_{smoke_t001,none_t4,dykstra30_t4,dykstra200_t4}_20260915_v1`.
  Every run was wrapped in `pipelines/shell/ram_guard.py`; the largest peak
  was `751 MB`.

## 5. What to try next, in order

1. The carried-witness Dykstra repair at `h=0.00125` and to `t=20`, which is
   the horizon the both-cones record line (`0.128607` over `t=0` to `20`)
   lives on.  At `69 s` for `t=4` a `t=20` run is about `6 min`.
2. A declared refresh cadence for the fixed dictionary's lift: re-solve the
   proximal log-det completion every `K` steps and carry it in between.  One
   solve costs `0.83` to `1.34 s`; at `K=50` that is `96 s` over `t=0` to
   `4`, and the drift table says the entrance velocity error would fall from
   50 percent toward zero.  This leaves the "no conic solve at all"
   constraint, which is why it was not run here, and it is the single
   cheapest way to find out whether `0.131` moves toward `0.026`.
3. A cheap surrogate for the lift's own velocity, so the fixed dictionary
   stops freezing it without paying a solve.
