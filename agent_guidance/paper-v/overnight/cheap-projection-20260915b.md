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

# Task 2: the `t=20` horizon, and re-selecting the lift (2026-09-15, same day)

Same branch, worktree and modules as the section above; the two modules gained
a declared lift re-selection cadence and the tests that cover it, and no
file this route does not own was edited.  Every score below is the
all-coordinate scalar RMS of the 31 closed scalar coordinates against the
exact cutoff-16 driven reference, which is the record's
`coordinate_scalar_rms` (`all_coordinate_scalar_rms_error`), on the stated
horizon; every internal-energy number is the sample RMS of the
time-independent energy of Eq. (22) (`static_energy_rms_error`) on that same
horizon; every physicality number is a measured minimum eigenvalue along the
trajectory, with no certificate anywhere on this route.  Configuration
throughout: strong coupling (`lambda_ep = 1.5`, `gamma = 0.5`, drive
amplitude 1), phonon cutoff 16, exact-correlated preparation, 15 entrance
keys, `h = 0.0025`.

Answering the reviewer's correction (1b) of
`review-routes-20260915b.md` before it applies here: every figure in this
section is the same in the raw and the offset-subtracted convention to every
digit printed.  The fixed-dictionary runs carry a `t=0` coordinate offset of
`8.08e-16` in Euclidean norm, and the carried-witness `t=20` run carries
`3.24e-4`, which moves its score from `12.967652` to `12.967653`.  Both
conventions are in every `summary.json`
(`all_coordinate_scalar_rms_error` and
`dynamic_all_coordinate_scalar_rms_error`,
`static_energy_rms_error` and `dynamic_static_energy_rms_error`), and the
record's `0.128607`, `0.193205`, `0.000939628` and `0.0182361` are the raw
convention.

## Negative result first

Neither cheap repair route is a long-horizon closure.  Over `t=0` to `20` the
carried-witness no-guard flow with a Dykstra-30 repair after every step scores
`12.9677` and the fixed-dictionary Dykstra-30 flow scores `1.41587`, against
the record's both-cones `0.128607` and the record's plain archive EOM
`0.193205` on that horizon.  Their internal-energy sample RMS over `t=0` to
`20` is `47.2251` and `6.79887` against the record's `0.000939628` and
`0.0182361`.  The repair holds the Gram at the cone for about four time units
and then loses it: the carried-witness 62-row unified Gram minimum after
repair is `-9.93e-6` at `t=4`, `-2.41e-4` at `t=8`, `-1.03e-3` at `t=12` and
`-2.85e-2` at `t=20`.

The positive result is in section (b): re-selecting the declared proximal
log-det completion every 16 steps, with Dykstra 30 on both targets in
between, reaches the projective event-guard route's `t=0` to `4` accuracy
`0.026049` at `89.6 s` instead of `3,229.8 s`; and section (c) takes that
configuration through the post-pulse interval `t=5.09` to `5.24`, where every
propagated projective continuation in the record has stopped.

## (a) `t=0` to `20` at `h=0.0025`

### The carried-witness state (384 coordinates, 62-row unified Gram)

The record's own no-guard CWRMF flow (`enforce_gram_guard=False`), stepped one
step at a time through `integrate_cwrmf_ssprk2`, with the declared Dykstra-30
repair applied to the 62-row unified Gram after every accepted step.  8,000
steps, 8,000 repairs applied and none skipped, mean repair `26.7 ms`.

| quantity, `t=0` to `20` | this run | record, both cones | record, plain archive EOM |
|---|---:|---:|---:|
| all-coordinate scalar RMS, 31 closed coordinates | `12.9677` | `0.128607` | `0.193205` |
| internal-energy sample RMS, Eq. (22) | `47.2251` | `0.000939628` | `0.0182361` |
| site-0 occupation RMS | `0.210425` | `0.204690` | `0.286170` |
| minimum scaled 62-row unified Gram eigenvalue after repair | `-3.356e-2` | --- | --- |
| minimum scaled 62-row unified Gram eigenvalue before repair | `-1.124e-1` | --- | --- |
| minimum unshifted retained joint Gram eigenvalue | `-2.4989e-16` | `1.36819e-5` | --- |
| autonomous wall (propagation / repair) | `343.86 s` (`130.25` / `213.37`) | `5,832.34 s` | --- |
| peak resident | `852 MB` | --- | --- |

Blockwise scalar RMS over `t=0` to `20`: rho `0.287388`, B `0.750984`, N
`35.5660`, A `4.96770`, C `0.434290`.  Essentially the whole error is the
phonon-occupation block.

Where it goes: the score on the horizon `t=0` to `T` is `0.18430` at `T=4`
(reproducing this route's earlier `t=4` run in every printed digit),
`0.258996` at `5`, `0.353739` at `6`, `0.783342` at `8`, `1.44132` at `10`,
`2.77530` at `12`, `6.94741` at `16` and `12.9677` at `20`.  The
instantaneous 31-coordinate Euclidean error is `2.07` at `t=4`, `4.85` at
`t=6`, `10.27` at `t=8` and `183.16` at `t=20`, of which the N block carries
`2.02`, `4.69`, `10.07` and `181.59`.  The site-0 occupation RMS stays at the
record's own scale (`0.210` against the both-cones `0.205`) over the whole
horizon, so the divergence is in the phonon occupations, not the electronic
density.

Reading: the repair keeps the flow representable and finite to `t=20`, and
that is all it does.  It costs `343.9 s`, 17 times less than the record's
`5,832.34 s` both-cones run, for a score 101 times worse.

### The fixed-dictionary state (44 coordinates, 24-row extension), same horizon

Dykstra 30 on both targets at stage cadence, carried lift, no re-selection:
all-coordinate scalar RMS `1.41587` over `t=0` to `20`, internal-energy sample
RMS `6.79887`, site-0 occupation RMS `0.230607`, minimum scaled 24-row
extension eigenvalue `-1.508e-4`, minimum unshifted retained joint Gram
`-2.254e-5`, maximum standardized frontier envelope ratio `1.940`, autonomous
wall `200.00 s` (repair `142.10 s`, velocity `56.26 s`), 23,730 repairs
applied and 270 skipped, peak resident `786 MB`.  Sub-horizon scores:
`0.131335` at `T=4` (reproducing the earlier run), `0.163107` at `5`,
`0.201465` at `6`, `0.283900` at `8`, `0.532202` at `12`, `1.41587` at `20`.
The carried lift's standardized coordinates leave the frozen box
(`envelope > 1`) at about `t=13`, which is where its score turns over.

On the long horizon, then, the fixed dictionary with a frozen lift is 8.8
times more accurate than the carried witness with a repaired Gram, and both
are worse than the plain archive EOM.

## (b) The hybrid: re-select the lift every `N` steps, `t=0` to `4`

The declared proximal log-det completion,
`SymmetryReducedPositiveExtension.complete`, is solved at the accepted state
every `N` steps and its frontier carried until the next re-selection, with
Dykstra 30 on both targets at every stage in between.  The carried frontier is
handed to the solve as its warm frontier, which is the projective route's
non-cold `completion_start`.  A completion that refuses is counted and the
carried lift is kept, so the rollout still cannot stop on a conic refusal.
1,600 steps in every row.

| `N` | all-coordinate scalar RMS, `t=0` to `4` | internal-energy sample RMS | minimum scaled 24-row extension eigenvalue | minimum unshifted retained joint Gram | maximum frontier envelope ratio | completions solved / refused | completion wall | autonomous wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none, carried lift, no repair | `0.13304` | not recorded | `-1.5159` | `-1.5578` | `0.4216` | 0 / 0 | `0 s` | `11.03 s` |
| none, carried lift, Dykstra 30 | `0.131335` | `0.780101` | `-1.043e-5` | `-7.924e-7` | `0.4487` | 0 / 0 | `0 s` | `39.44 s` |
| 64 | `0.0412601` | `0.110171` | `-4.281e-7` | `-1.423e-8` | `0.4230` | 25 / 1 | `20.5 s` | `49.8 s` |
| 16 | `0.0268423` | `0.0420873` | `-1.969e-7` | `-6.219e-8` | `0.4370` | 100 / 9 | `68.3 s` | `89.6 s` |
| 4 | `0.0262899` | `0.0376080` | `-5.067e-7` | `-4.020e-8` | `0.4390` | 400 / 59 | `120.3 s` | `137.9 s` |
| 1 | `0.0273992` | `0.0474351` | `-8.211e-7` | `-6.176e-8` | `0.5560` | 1,600 / 250 | `682.5 s` | `702.2 s` |
| record: projective event guard | `0.026049` | not printed | `-9.3e-16` | `2.892e-5` | --- | 3 per stage | --- | `3,229.8 s` |

Four readings.

- **The lift was the whole gap.**  One re-selection every 16 steps takes the
  score from `0.131335` to `0.0268423`, that is onto the projective
  event-guard route's `0.026049`, at `89.6 s` against `3,229.8 s`: the same
  accuracy at 1/36 of the cost.  The first pass's drift table said the frozen
  lift cost 37 to 101 percent of the entrance-moment commutator velocity;
  this is that statement with a score attached.
- **More re-selection is not monotonically better.**  `N=4` scores
  `0.0262899` and `N=1` scores `0.0273992`, so `N=1` is 4 percent *worse*
  than `N=4` at 5.1 times the wall.  At `N=1` the completion is re-solved at
  a state the repair has just moved, 250 of the 1,600 solves refuse, and the
  carried lift is kept at those steps, so the densest cadence is also the one
  whose lift is least often the declared one.
- **A warm start makes the solve cheap, and cheaper as the cadence tightens**:
  mean completion wall `0.822 s` at `N=64`, `0.683 s` at `N=16`, `0.301 s` at
  `N=4` (median `0.006 s`, maximum `2.601 s`), because the carried frontier
  handed to the solve is closer to its answer.  The cheap repair itself goes
  nearly idle at a tight cadence: over `t=0` to `4` the Dykstra repair was
  applied at 3,015 of 4,800 stage opportunities at `N=64`, 1,911 at `N=16`,
  953 at `N=4` and 1,259 at `N=1`, against 23,730 of 24,000 over `t=0` to
  `20` with a frozen lift.
- **Refusals are ordinary here.**  1 of 25 at `N=64`, 9 of 100 at `N=16`, 59
  of 400 at `N=4`, 250 of 1,600 at `N=1`, spread over the whole interval
  (`t=0.02` to `3.62` at `N=4`), with the messages the record's completion
  emits: at `N=1`, 126 of the 250 refusals are completions that returned
  without a certificate the extension accepts, 117 are
  `moment_extension_degeneracy: conic Newton step failed descent`, and 7 are
  a Clarabel `NumericalError` or `InsufficientProgress`.  Because the
  route keeps the carried lift on refusal and repairs it, no refusal stops
  anything, which is the difference between this route and every projective
  continuation in the record.

## (c) `N=16` and `N=64` through the post-pulse wall, `t=0` to `8`

Both cadences beat `0.05` over `t=0` to `4` at under `600 s`, so both were run
to `t=8`.  Both completed all 3,200 steps.

| quantity, `t=0` to `8` | `N=16` | `N=64` |
|---|---:|---:|
| all-coordinate scalar RMS | `0.0577300` | `0.121366` |
| internal-energy sample RMS | `0.226943` | `0.870186` |
| site-0 occupation RMS | `0.0658232` | `0.0981753` |
| minimum scaled 24-row extension eigenvalue | `-9.124e-6` | `-3.217e-5` |
| minimum unshifted retained joint Gram | `-8.296e-7` | `-8.488e-7` |
| maximum standardized frontier envelope ratio | `0.471` | `0.454` |
| completions solved / refused | 200 / 20 | 50 / 2 |
| autonomous wall | `195.7 s` | `98.4 s` |

**It does not fail at `t=5.09` to `5.24`.**  The projective route's wall --
every propagated-state continuation in the record stops between `t=4.83` and
`t=5.235` -- is not a feature of this trajectory.  The `N=16` score over
`t=0` to `T` is `0.0268` at `T=4`, `0.0332` at `5`, `0.0393` at `6`, `0.0461`
at `7` and `0.0577` at `8`; the instantaneous 31-coordinate Euclidean error is
`0.3025` at `t=4`, `0.2867` at `t=5`, `0.3562` at `t=6` and `0.7041` at
`t=8`, so there is not even a local maximum where the projective route stops.
Four of the 20 `N=16` refusals fall in `5.16 <= t <= 5.92`, and each is
absorbed by carrying the repaired lift for another 16 steps.

What does degrade, smoothly and after the drive ends, is the phonon
occupation: the N block's contribution to the `N=16` score is `0.0159` over
`t=0` to `4` and `0.107` over `t=0` to `8`, while the C block moves only from
`0.0227` to `0.0341`.  The post-pulse error of this route is a
phonon-occupation error, not a representability failure and not a refusal.

## Provenance and gates

- `paper_5` suite on the tree these runs were made on: 466 passed in 251.53 s
  (`cd paper_5 && PYTHONPATH=src python -m pytest -q`), including the 21 tests
  in `paper_5/tests/test_cheap_projection_route.py` -- the 16 of the first
  pass plus 5 covering the lift re-selection cadence.
- Modules changed, both owned by this route and neither imported by any other:
  `cheap_projection_route.py` gained `CheapProjectionRoute.reselect_lift`, the
  `lift_reselection_interval` and `lift_reselection_warm_start` arguments of
  `integrate_cheap_projection_ssprk3`, and five trajectory fields recording
  the cadence; `cheap_projection_route_analysis.py` gained
  `--lift-reselection-interval`, `--lift-reselection-start`, a `lift` block in
  the summary, and the `t=0` to `20` record lines.  Nothing is committed.
- The three tables of sub-horizon scores, the hybrid table and the refusal
  statistics are produced by three scripts kept with the runs, answering the
  reviewer's correction (1d): `score_subhorizons.py`,
  `summarize_hybrid_rows.py` and `summarize_reselections.py`, all under
  `output/local_runs/paper_v_cheap_projection_hybrid_n16_t8_20260915_v1/`.
  Each takes run directories on its command line and reads only
  `trajectory.npz`, `summary.json` and `repairs.jsonl`.
- Artifacts, all under `output/local_runs/`:
  `paper_v_cheap_projection_cw_dykstra30_t20_20260915_v1`,
  `paper_v_cheap_projection_dykstra30_both_t20_20260915_v1`,
  `paper_v_cheap_projection_hybrid_n{1,4,16,64}_t4_20260915_v1`,
  `paper_v_cheap_projection_hybrid_n{16,64}_t8_20260915_v1`.  Every run was
  wrapped in `pipelines/shell/ram_guard.py --limit-mb 3000`, one at a time;
  the largest peak was `861 MB` and no run exceeded 12 minutes.  Wall times
  were measured while other agents' jobs shared the machine (load average 4.8
  to 5.6), so they are upper bounds.

## What to try next, in order

1. `N=16` from `t=0` to `20`.  At `195.7 s` for `t=8` that is about `500 s`,
   and it is the first configuration on this route with a chance of standing
   next to the record's both-cones `0.128607`; the `t=0` to `8` trend leaves
   open whether its phonon-occupation error keeps growing like the frozen
   lift's or settles.
2. A refusal policy for the hybrid: at `N=4` and `N=1` a seventh of the solves
   refuse and the carried lift is kept.  The record's completion retry ladder
   exists for exactly this and costs nothing when no solve refuses; whether it
   moves `0.0263` is one run.
3. The cadence is counted in steps.  A declared trigger -- re-solve when the
   standardized frontier has moved more than a declared distance since the
   last re-selection -- would spend the solves where the drift table says they
   are needed, rather than uniformly.

## Structured summary of task 2

| item | value |
|---|---|
| horizon reached, carried-witness Dykstra-30 per step | `t=20`, 8,000 steps, `343.86 s`, peak `852 MB` |
| its all-coordinate scalar RMS over `t=0` to `20` | `12.9677` against the record's both-cones `0.128607` and plain archive EOM `0.193205` |
| its internal-energy sample RMS over the same horizon | `47.2251` against `0.000939628` and `0.0182361` |
| its physicality | 62-row unified Gram `-3.356e-2` after repair (`-1.124e-1` before), retained joint Gram `-2.4989e-16` |
| fixed dictionary, Dykstra 30, frozen lift, `t=0` to `20` | `1.41587`, internal energy `6.79887`, extension `-1.508e-4`, retained `-2.254e-5`, envelope `1.940`, `200.00 s` |
| best hybrid over `t=0` to `4` | `N=16`: `0.0268423` at `89.6 s`, against the projective event guard's `0.026049` at `3,229.8 s` and the frozen lift's `0.131335` at `39.44 s` |
| hybrid cadence trend, `t=0` to `4` | `N=64` `0.0412601` / `49.8 s`; `N=16` `0.0268423` / `89.6 s`; `N=4` `0.0262899` / `137.9 s`; `N=1` `0.0273992` / `702.2 s` |
| post-pulse behaviour, `N=16` to `t=8` | completes 3,200 steps in `195.7 s`, `0.0577300` over `t=0` to `8`, no wall at `t=5.09` to `5.24`, 20 of 200 completions refused and absorbed |
| what fails after the drive | the phonon-occupation block: `N=16`'s N-block scalar RMS is `0.0159` over `t=0` to `4` and `0.107` over `t=0` to `8` |
| gates | `paper_5` suite 466 passed in `251.53 s`; 21 tests in `test_cheap_projection_route.py` |
| files changed | `cheap_projection_route.py`, `cheap_projection_route_analysis.py`, `test_cheap_projection_route.py`, this report.  Nothing committed |

# Task 3: the hybrid at `t=20`, the internal-energy column, and what the repair is for (2026-09-15, same day)

Same branch, worktree and modules as the two sections above; no module was
edited in this pass and no file this route does not own was touched.  The
`t=20` hybrid runs at `N=16` and `N=4` were launched by the previous instance
of this key and both completed (8,000 of 8,000 steps, `message: completed`);
this pass scores them, adds the Eq. (22) internal energy to every hybrid row by
rescoring saved artifacts, and runs the one ablation that separates the two
things the hybrid does.

Conventions, as in task 2: the score is the all-coordinate scalar RMS of the
31 closed scalar coordinates against the exact cutoff-16 driven reference
(`all_coordinate_scalar_rms_error`) on the stated horizon; the internal energy
is the sample RMS of the time-independent energy of Eq. (22)
(`static_energy_rms_error`) on that same horizon; physicality numbers are
measured minimum eigenvalues along the trajectory and there is no certificate
anywhere on this route.  The fixed-dictionary runs carry a `t=0` coordinate
offset of `8.08e-16`, so the raw and the offset-subtracted convention agree to
every digit printed here; the record's `0.128607`, `0.193205`, `0.000939628`
and `0.0182361` are the raw convention.  Configuration for every row: strong
coupling (`lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude 1), phonon cutoff
16, exact-correlated preparation, 15 entrance keys, SSPRK(3,3) at
`h = 0.0025`, the unguarded fixed-dictionary predictor, Dykstra 30 on both
targets at stage cadence between re-selections, and the declared proximal
log-det completion re-solved every `N` steps warm-started on the carried
frontier.

## Negative result first

Neither hybrid cadence reaches the record's both-cones line over `t=0` to
`20`.  `N=4` scores `0.259573` and `N=16` scores `0.579332`, against the
record's both-cones `0.128607` and the record's plain archive EOM `0.193205`
on that horizon; their Eq. (22) internal-energy sample RMS is `0.804515` and
`2.59371` against the record's `0.000939628` and `0.0182361`.  `N=4` is
therefore 2.02 times the both-cones score and 1.34 times the plain archive
EOM, and the configuration that matched the projective event guard over `t=0`
to `4` matches nothing over `t=0` to `20`.

What is new and positive is that the hybrid holds both cones for the whole 20
time units while doing it, which no other cheap-repair configuration in this
report does, and that the internal-energy penalty the reviewer measured on the
Dykstra-only repair is a repair artifact which the re-selection almost
entirely removes.

## (a) `t=0` to `20` at `h=0.0025`, both cadences

| quantity, `t=0` to `20`, 8,000 steps | `N=16` | `N=4` | record, both cones | record, plain archive EOM |
|---|---:|---:|---:|---:|
| all-coordinate scalar RMS, 31 closed coordinates | `0.579332` | `0.259573` | `0.128607` | `0.193205` |
| internal-energy sample RMS, Eq. (22) | `2.59371` | `0.804515` | `0.000939628` | `0.0182361` |
| site-0 occupation RMS | `0.199933` | `0.203657` | `0.204690` | `0.286170` |
| C-block scalar RMS | `0.132404` | `0.0917819` | --- | --- |
| N-block scalar RMS | `1.36235` | `0.532930` | --- | --- |
| minimum scaled 24-row extension eigenvalue | `-2.401e-5` | `-5.553e-6` | --- | --- |
| minimum unshifted retained joint Gram | `-8.296e-7` | `-8.975e-8` | `1.36819e-5` | --- |
| maximum standardized frontier envelope ratio | `0.9105` | `0.6356` | --- | --- |
| completions solved / refused | 500 / 22 | 2,000 / 154 | --- | --- |
| completion wall (mean per solve) | `353.3 s` (`0.707 s`) | `791.6 s` (`0.396 s`) | --- | --- |
| repairs applied / skipped of 24,000 stage opportunities | 13,024 / 10,976 | 5,643 / 18,357 | --- | --- |
| repair wall (mean per repair) | `76.7 s` (`3.20 ms`) | `36.7 s` (`1.53 ms`) | --- | --- |
| autonomous wall | `486.2 s` | `884.2 s` | `5,832.34 s` | --- |
| peak resident | `773 MiB` | `776 MiB` | --- | --- |

Four readings.

- **Both cones are held for the whole horizon.**  This is the first
  configuration on this route where neither matrix is lost.  The minimum
  24-row extension eigenvalue over `t=0` to `T` is `-5.74e-8` at `T=1`,
  `-1.97e-7` at `4`, `-9.12e-6` at `6`, `-1.35e-5` at `12` and `-2.40e-5` at
  `20` for `N=16`, and `-8.91e-9`, `-5.07e-7`, `-1.15e-6`, `-4.52e-6` and
  `-5.55e-6` at those same times for `N=4`; the standardized frontier stays
  inside its frozen box throughout (`0.911` and `0.636`, both under 1).  Both
  cheap routes of task 2 lose their cone on this horizon: the carried-witness
  Dykstra-30 flow reaches `-3.36e-2` on the 62-row unified Gram, and the fixed
  dictionary with a frozen lift reaches `-1.51e-4` on the extension with its
  frontier at `1.940` times the frozen box.  Re-selecting the lift, not the
  repair, is what keeps the frontier in the box.
- **Denser re-selection is now monotone, and the `t=0` to `4` ordering has
  reversed.**  Over `t=0` to `4` the two cadences were within 2 percent of each
  other (`N=16` `0.0268423`, `N=4` `0.0262899`); over `t=0` to `20` `N=4` is
  2.23 times more accurate than `N=16` (`0.259573` against `0.579332`) and
  3.22 times more accurate on the internal energy.  The running score of the
  `N=16` run reproduces the earlier `t=4` and `t=8` artifacts in every printed
  digit (`0.0268423`, `0.05773`) and then departs: `0.0875` at `T=10`, `0.136`
  at `12`, `0.324` at `16`, `0.579` at `20`, against `N=4`'s `0.101`, `0.146`,
  `0.202` and `0.260`.
- **The error is the phonon-occupation block, as it was at `t=8`.**  The N
  block carries `1.36235` of `N=16`'s `0.579332` and `0.532930` of `N=4`'s
  `0.259573`; the instantaneous 31-coordinate Euclidean error at `t=20` is
  `6.427` for `N=16`, of which the N block is `5.709`, and `3.008` for `N=4`,
  of which the N block is `2.319`.  The electronic density does not follow:
  the site-0 occupation RMS is `0.199933` and `0.203657` against the record's
  both-cones `0.204690`, that is at the record's own scale and better than the
  plain archive EOM's `0.286170`, on a horizon where the coordinate score is
  2.0 to 4.5 times the record's.
- **Refusals stay ordinary and never stop the rollout.**  22 of 500 at `N=16`
  and 154 of 2,000 at `N=4`, spread over the whole interval (`N=16` at
  `t=0.120` to `13.360`, `N=4` at `t=0.020` to `14.19`), with the messages the
  record's completion emits -- at `N=16`, 9 direct Clarabel proximal log-det
  failures, 8 `moment_extension_degeneracy: conic Newton step failed descent`
  on a full-support relative face and 5 on the full cone interior.  The
  carried lift is kept and repaired at a refusal, so the route reaches `t=20`
  either way.

## (b) The Eq. (22) internal energy of every hybrid row, by rescoring saved artifacts

Answering the reviewer's correction (1a), which measured a 51-fold
internal-energy regression from the Dykstra-only repair on the carried-witness
lane (`0.0138792` unrepaired to `0.7092441` repaired over `t=0` to `4`) and
asked for the column on these lanes.  Nothing was propagated for this table: it
is `score_internal_energy.py` reading `trajectory.npz` and `summary.json` of
runs that already existed, recomputing `matrix_total_energy` on the propagated
and the exact coordinates and taking the sample RMS of the difference on the
requested sub-horizon.  On the full horizon of each run the recomputed value is
that run's own `static_energy_rms_error` to every digit printed.

| fixed-dictionary row | score, `t=0` to `4` | internal energy, `t=0` to `4` | score, `t=0` to `8` | internal energy, `t=0` to `8` |
|---|---:|---:|---:|---:|
| no repair, frozen lift (control) | `0.133036` | `0.0292046` | --- | --- |
| Dykstra 30, frozen lift | `0.131335` | `0.780101` | `0.283900` | `1.90816` |
| `N=64` | `0.0412601` | `0.110171` | `0.121366` | `0.870186` |
| `N=16` | `0.0268423` | `0.0420873` | `0.0577300` | `0.226943` |
| `N=4` | `0.0262899` | `0.0376080` | `0.0495752` | `0.0915587` |
| `N=1` | `0.0273992` | `0.0474351` | --- | --- |
| `N=16`, no repair between re-selections (section (c)) | `0.0394447` | `0.0305130` | --- | --- |
| record, projective event guard | `0.026049` | `0.036201` | --- | --- |
| record, plain archive EOM | `0.11090` | `0.027587` | --- | --- |

**The hybrid does not inherit the 51-fold penalty.**  On the fixed dictionary
the Dykstra-only repair costs a factor 26.7 on the internal energy
(`0.0292046` unrepaired to `0.780101` repaired over `t=0` to `4`), which is the
same defect the reviewer measured at 51.1 on the carried witness.  Re-selecting
the lift removes 95 percent of it: `N=16` is `0.0420873` and `N=4` is
`0.0376080`, that is 18.5 and 20.7 times better than the Dykstra-only
`0.780101`, and 1.16 and 1.04 times the projective event-guard route's
`0.036201` on the same horizon.  At `N=4` the hybrid matches the projective
route on the internal energy as well as on the coordinate score.

What is left of the penalty is a factor **1.38**, and section (c) is the
measurement of it: `N=16` with the repair is `0.0420873` and the identical run
without the repair is `0.0305130`.  The repair still moves the energy the wrong
way, by 38 percent rather than by 2,600 percent, and that residual is the price
of the physicality in the row above it.

The penalty grows with the horizon at every cadence -- `N=16` goes `0.0420873`
to `0.226943` to `2.59371` over `t=0` to 4, 8 and 20, and `N=4` goes
`0.0376080` to `0.0915587` to `0.804515` -- and it grows faster than the
coordinate score does (a factor 62 against a factor 22 for `N=16` between
`T=4` and `T=20`).  Eq. (22) is the sharper diagnostic on this route.

## (c) One ablation: `N=16` with no repair between re-selections, `t=0` to `4`

Identical to the `N=16` row of task 2 (b) in every setting except
`--projection-policy none`, so the declared completion is re-solved every 16
steps and nothing is repaired at any of the 4,800 stage opportunities; the
carried lift then moves only where the completion is re-solved.  1,600 steps,
completed.

| quantity, `t=0` to `4` | `N=16`, Dykstra 30 between | `N=16`, no repair between |
|---|---:|---:|
| all-coordinate scalar RMS | `0.0268423` | `0.0394447` |
| internal-energy sample RMS, Eq. (22) | `0.0420873` | `0.0305130` |
| site-0 occupation RMS | `0.0461563` | `0.0199981` |
| C-block scalar RMS | `0.0226584` | `0.0485123` |
| N-block scalar RMS | `0.0158911` | `0.0106550` |
| minimum scaled 24-row extension eigenvalue | `-1.969e-7` | `-0.323490` |
| minimum unshifted retained joint Gram | `-6.219e-8` | `-0.759345` |
| maximum standardized frontier envelope ratio | `0.4370` | `0.8174` |
| completions solved / refused | 100 / 9 | 100 / 52 |
| mean completion wall per solve | `0.683 s` | `1.082 s` |
| repairs applied / skipped of 4,800 stage opportunities | 1,911 / 2,889 | 0 / 4,800 |
| autonomous wall | `89.6 s` | `119.7 s` |
| peak resident | `281 MiB` | `351 MiB` |

Three readings, and together they separate the hybrid's two ingredients.

- **The re-selection carries the accuracy; the repair carries the
  physicality.**  With no repair anywhere, re-selection alone takes the score
  from the frozen lift's `0.133036` to `0.0394447`, which is 88 percent of the
  hybrid's total reduction in the score, while the 24-row extension sits
  at `-0.323490` and the retained joint Gram at `-0.759345`.  Adding the
  repair moves the score a further 1.47 times, to `0.0268423`, and the two
  minima by six and seven decades, to `-1.969e-7` and `-6.219e-8`.
- **The repair pays for that physicality with the internal energy and the
  electronic density.**  Without it the internal energy is `0.0305130` and the
  site-0 occupation RMS is `0.0199981`; with it they are `0.0420873` and
  `0.0461563`, worse by 1.38 and 2.31.  The C block moves the other way, from
  `0.0485123` to `0.0226584`, and the N block from `0.0106550` to `0.0158911`.
  The repair is therefore not uniformly good on accuracy even at this cadence:
  it trades the energy, the density and the phonon occupation for the
  correlation block and for the cone.
- **The repair makes the conic solve easier.**  52 of the 100 completions
  refuse when nothing is repaired, against 9 of 100 when Dykstra 30 runs
  between re-selections, and the mean solve is `1.082 s` against `0.683 s`.
  Keeping the state near the cone is what makes the declared completion cheap
  and reliable at its warm start, so the unrepaired route is both less
  accurate and 34 percent slower in wall time despite doing strictly less work
  per step.

## Provenance and gates

- `paper_5` suite on the tree these runs were made on: 467 passed in
  `250.29 s` (`cd paper_5 && PYTHONPATH=src python -m pytest -q`), the 466 of
  task 2 plus the one new test below.
- `paper_5/tests/test_cheap_projection_route.py` now holds 22 tests, the 21 of
  task 2 plus
  `test_lift_reselection_without_a_repair_moves_the_lift_only_on_cadence`,
  which is the section (c) configuration at `policy="none"`: it asserts that no
  repair is applied at any of the 12 stage opportunities of a four-step
  rollout, that the state and frontier displacements are exactly zero, and
  that the carried lift changes only at the two re-selection steps.  That test
  was left uncommitted by the previous instance of this key and is kept.
- No module was edited in this pass.  `cheap_projection_route.py` and
  `cheap_projection_route_analysis.py` are unchanged from commit `b04ec23b`.
  Nothing is committed.
- The internal-energy table is produced by
  `output/local_runs/paper_v_cheap_projection_rescore_20260915_v1/score_internal_energy.py`,
  and the sub-horizon, physicality and refusal tables by `score_subhorizons.py`
  and `summarize_reselections.py` under
  `output/local_runs/paper_v_cheap_projection_hybrid_n16_t8_20260915_v1/`.
  This answers the reviewer's correction (1d): each reads only
  `trajectory.npz`, `summary.json` and `repairs.jsonl` of the run directories
  named on its command line, and propagates nothing.
- Artifacts, under `output/local_runs/`:
  `paper_v_cheap_projection_hybrid_n16_t20_20260915_v1` and
  `paper_v_cheap_projection_hybrid_n4_t20_20260915_v1`, both launched by the
  previous instance and both `completed`; and
  `paper_v_cheap_projection_hybrid_n16_norepair_t4_20260915_v1` from this pass.
  The ablation was wrapped in `pipelines/shell/ram_guard.py --limit-mb 3000`,
  one at a time.  Naming the clock, which the reviewer's correction (1e)
  asked for: every peak-resident figure in this section is the run's own
  `peak_resident_bytes` (`ru_maxrss` of the propagating process), so `773 MiB`
  and `776 MiB` for the two `t=20` runs and `351 MiB` for the ablation; the
  `ram_guard` reported `231 MB` for the same ablation.  No run in this pass
  exceeded `120 s`.

## What to try next, in order

1. `N=4` is the long-horizon cadence and `N=1` was never run past `t=4`.  The
   `t=0` to `4` ordering (`N=1` 4 percent worse than `N=4`) is the reverse of
   what happened between `N=16` and `N=4` at `T=20`, so whether it reverses
   again at `N=1` is one run of the `900 s` class and decides whether to
   tighten the cadence further or to fix the refusal policy instead.
2. The repair's 1.38-fold internal-energy cost is now isolated, and it is a
   property of the Dykstra projection, which minimizes a Euclidean distance in
   coordinates that Eq. (22) does not weight equally.  A repair that projects
   in the Eq. (22) metric, or that holds the energy fixed while clipping, is
   the same linear algebra and would say whether the cost is removable.
3. The phonon-occupation block carries the whole long-horizon error at every
   cadence while the site-0 occupation stays at the record's scale.  Whether
   the 24-row extension constrains the N block at all at cutoff 16 is a
   build-time question about the affine map, not a run.

## Structured summary of task 3

| item | value |
|---|---|
| `t=0` to `20`, `N=16` | `0.579332`, internal energy `2.59371`, site-0 occupation `0.199933`, C block `0.132404`, extension `-2.401e-5`, retained joint Gram `-8.296e-7`, envelope `0.9105`, 500 solved / 22 refused, `486.2 s`, `773 MiB` |
| `t=0` to `20`, `N=4` | `0.259573`, internal energy `0.804515`, site-0 occupation `0.203657`, C block `0.0917819`, extension `-5.553e-6`, retained joint Gram `-8.975e-8`, envelope `0.6356`, 2,000 solved / 154 refused, `884.2 s`, `776 MiB` |
| against the record on that horizon | both cones `0.128607` / `0.000939628` / site-0 `0.204690` at `5,832.34 s`; plain archive EOM `0.193205` / `0.0182361` / site-0 `0.286170`.  `N=4` is 2.02 times the both-cones score and 1.34 times the archive EOM |
| what is new | both cones held for the whole 20 time units (extension `-2.4e-5` and `-5.6e-6`, frontier inside its frozen box at `0.911` and `0.636`), which neither task-2 cheap route manages |
| where the error is | the phonon-occupation block: N block `1.36235` of `N=16`'s `0.579332` and `0.532930` of `N=4`'s `0.259573`, while the site-0 occupation stays at the record's scale |
| (b) internal energy, `t=0` to `4` | control `0.0292046`; Dykstra-only `0.780101`; `N=64` `0.110171`; `N=16` `0.0420873`; `N=4` `0.0376080`; `N=1` `0.0474351`; projective event guard `0.036201`; plain archive EOM `0.027587` |
| (b) internal energy, `t=0` to `8` | Dykstra-only `1.90816`; `N=64` `0.870186`; `N=16` `0.226943`; `N=4` `0.0915587` |
| does the hybrid inherit the 51-fold penalty | **No.**  The Dykstra-only repair costs 26.7 on the fixed dictionary (51.1 on the carried witness); re-selection removes 95 percent of it, leaving a factor `1.38` measured directly by section (c) |
| (c) ablation, `N=16` no repair, `t=0` to `4` | `0.0394447` at internal energy `0.0305130` and site-0 occupation `0.0199981`, but extension `-0.323490` and retained joint Gram `-0.759345`, 52 of 100 completions refused, `119.7 s` |
| (c) attribution | re-selection alone gives 88 percent of the hybrid's total reduction in the score (`0.133036` to `0.0394447` of `0.133036` to `0.0268423`); the repair adds 1.47 times more accuracy and six to seven decades of physicality, and costs 1.38 on the internal energy and 2.31 on the site-0 occupation |
| gates | `paper_5` suite 467 passed in `250.29 s`; 22 tests in `test_cheap_projection_route.py` (21 of task 2 plus the no-repair re-selection test) |
| files changed | `paper_5/tests/test_cheap_projection_route.py` (kept from the previous instance), this report, and the worklog block in `paper_5/notes/electron_phonon_closure_worklog_20260803.md`.  No module edited, nothing committed |

# Task 4: the energy-neutral repair, and what the cone costs (2026-09-15, same day)

Same branch, worktree and modules as the three sections above; the two
modules this route owns gained a named energy-neutral option of the Dykstra
repair and the tests that cover it, and no file this route does not own was
edited.  Conventions are those of tasks 2 and 3: the score is the
all-coordinate scalar RMS of the 31 closed scalar coordinates against the
exact cutoff-16 driven reference (`all_coordinate_scalar_rms_error`) on the
stated horizon; the internal energy is the sample RMS of the
time-independent energy of Eq. (22) (`static_energy_rms_error`) on that same
horizon; physicality numbers are measured minimum eigenvalues along the
trajectory and there is no certificate anywhere on this route.
Configuration for every row: strong coupling (`lambda_ep = 1.5`,
`gamma = 0.5`, drive amplitude 1), phonon cutoff 16, exact-correlated
preparation, 15 entrance keys, SSPRK(3,3) at `h = 0.0025`, the unguarded
fixed-dictionary predictor, Dykstra 30 on both targets at stage cadence
between re-selections, and the declared proximal log-det completion
re-solved every `N` steps warm-started on the carried frontier.

## Negative result first

The energy-neutral repair buys the phonon-occupation block by giving back
the cone and by making the declared completion refuse.  Over `t=0` to `20`
at `N=16` the minimum scaled 24-row extension eigenvalue moves from
`-2.401e-5` to `-3.311e-4` and the minimum unshifted retained joint Gram
from `-8.296e-7` to `-8.776e-3`, that is one and four decades the wrong way,
while the completion refuses 107 of 500 solves instead of 22 of 500.  At
`N=4` the refusals reach 503 of 1,163 and the mean solve `0.917 s` against
the unconstrained cadence's `0.396 s`, so that run did not reach `t=20`
inside its `1,150 s` budget: it stopped at `t=11.63` after 4,652 of 8,000
steps.  And the constrained `N=16` score over `t=0` to `20`, `0.239172`, is
still 1.86 times the record's both-cones `0.128607` and 1.24 times the
record's plain archive EOM `0.193205`.

What the option does deliver is the phonon-occupation block and Eq. (22)
itself.  At `N=16` over `t=0` to `20` the internal-energy sample RMS falls
from `2.59371` to `0.069384`, the N-block scalar RMS from `1.36235` to
`0.267235` and the score from `0.579332` to `0.239172`; the energy is held
on its declared reference to `1.8e-15` at every one of 18,939 constrained
repairs.  The second reference mode, which follows the power the drive
delivers from `t=0` instead of waiting for the drive to end, gives the best
internal energy this route has produced, `0.026000` over `t=0` to `20`
against the plain archive EOM's `0.0182361`, and over `t=0` to `4` it scores
`0.023744` at internal energy `0.020406` against the record's projective
event guard's `0.026049` and `0.036201`.

## 0. The conservation law, and what is not one

Eq. (22) evaluated on the coordinates the scorer uses -- the 29 raw moment
coordinates to a matrix state, that state to the 31 closed scalar
coordinates, and those back to a matrix state, which is exactly the chain
`cheap_projection_route_analysis` scores -- is an exactly linear functional
of those 29 coordinates.  The module rebuilds it by evaluating that public
chain at the coordinate basis and verifies it at random probes: offset zero,
reconstruction error `2.2e-16`.  Seven of the 106 parameters carry the whole
functional: `-2 t_hop` on the electronic Bloch `x` coordinate, `2 g` on each
coherent-phonon real part and on two of the twelve raw cross coordinates,
and `omega_ph` on each of the two diagonal raw normal coordinates.  It
vanishes on the six anomalous coordinates, on the 15 entrance coordinates
and on all 62 frontier coordinates.  That is the working notes' statement
with its numbers attached: a correction confined to the phonon normal and
anomalous moments changes the energy by `omega_ph` times the trace of the
normal correction, and re-selecting the lift cannot change the internal
energy at all.

Two post-pulse quantities, measured on the exact cutoff-16 driven reference
stored in the run artifacts, over `t=4` to `20`:

| quantity | value at `t=4` | spread over `t=4` to `20` |
|---|---:|---:|
| Eq. (22) internal energy | `-3.571397982` | `3.0734e-06` |
| phonon number, the trace of the uncentered phonon normal moment | `4.017037038` | `7.1580e-01` |

against an Eq. (22) spread of `0.286911` over the driven interval `t=0` to
`4`.  So the energy is the conservation law available after the drive ends,
to five decades below the swing the drive itself produces, and the phonon
number is not one.  This settles task (c) before it is run, and section (c)
below carries the rest of that argument.

One build-time fact, which answers the third item of task 3's list: the
24-row positive extension does not constrain the phonon normal block.  Of
the four raw normal coordinates, one lies entirely in the 15-dimensional
null space of the extension's affine map and two more lie 83 percent in it;
the retained joint Gram sees all 29 raw coordinates and none of the 15
entrance or 62 frontier ones; and the two targets together, which is what
every run in this report repairs, have full column rank 106 of 106, with the
Eq. (22) gradient direction fully visible to them.  The extension alone
cannot see 32 percent of that direction, which is why the hyperplane is an
option of the `both` target.

## (a) The named option

`CheapProjectionSettings.energy_reference_mode` is the option, and it is an
option of the `dykstra` policy alone: a single clip cannot meet a
hyperplane, and the settings refuse the combination.  The alternation now
runs over three declared sets in the one realified feature space the repair
already used:

1. the positive semidefinite cone of each target block, at its declared
   floor;
2. the affine Gram set, the image of the affine coordinate map through the
   current point;
3. the hyperplane on which the Eq. (22) internal energy of the mapped-back
   coordinates equals a declared reference.

The third set is expressible there because the map back is one linear solve:
the repaired coordinates are the current ones plus
`scales * (pseudo_inverse @ residual)`, so their internal energy is the
current energy plus `w . residual` with
`w = pseudo_inverse^T (scales * gradient)`, and `w` lies in the span of the
range basis by construction.  Sets 2 and 3 are therefore projected onto
together in closed form, which is the limit an alternation between those two
affine sets converges to, and the cone is left as the only set the iteration
has to chase.  A repair is applied whenever a cone is below its floor *or*
the internal energy is further from the reference than the declared
`energy_tolerance`; when neither holds the repair is skipped and counted as
skipped exactly as before.  When a cone is already met and the energy is
already on its reference the alternation is the identity, so the extra
repair opportunities the option opens cost the state nothing by themselves:
of the `N=16` run's 20,850 applied repairs, 18,939 fall after `t=4` and
carried the hyperplane, and the remaining 1,911 fall in the driven interval,
which is exactly the count the task-3 `N=16` run applied there.

Two declared reference modes:

`post_pulse`
    The constraint is off until a declared drive-end time and from the first
    accepted state at or after it the reference is the internal energy that
    state itself carried.  No exact information enters: the reference is the
    trajectory's own value, `-3.4967309497207015` at `N=16` and
    `-3.5141117233509993` at `N=4`, which are `0.074667` and `0.057286`
    above the exact reference's `-3.571397982`.

`drive_power`
    The constraint is on from the start of the rollout and the reference is
    advanced with the same SSPRK(3,3) weights as the state, by the power the
    drive delivers, `gradient . (velocity at this time - velocity at t=0)`.
    That quantity is available from the model without the exact state,
    because the Hamiltonian's only time dependence is
    `DimerParameters.drive_difference` and that difference is exactly zero at
    `t = 0`, so the velocity evaluated at `t = 0` is the drive-free velocity
    of the same coordinates.  It costs a second velocity evaluation per
    stage.  Its reference reaches `-3.544171128` at `t=4`, `0.027219` above
    the exact value, and then moves by `7.8e-6` over the rest of the
    interval, which is the residual drive tail and is the same size as the
    exact reference's own `3.1e-6`.

This is not the record's `energy_neutral` control restriction.  The record's
ten-coordinate and 31-coordinate barriers restrict the correction to the
null space of the exact Eq. (22) energy-gradient row, so the correction
contributes zero instantaneous energy flux and the propagated velocity's own
energy drift is left where it is.  The option here holds the *state* on a
declared reference, so it removes the velocity's drift as well as the
repair's; those are the same statement only if the closure conserves the
energy, and over `t=4` to `20` it does not -- the unconstrained `N=16`
trajectory's own internal energy moves by `5.006` on that interval and the
`N=4` one by `1.558`.

## (b) `t=0` to `20` with the post-pulse reference, both cadences

Identical to the two `t=20` rows of task 3 in every setting except
`--energy-reference-mode post_pulse --energy-reference-time 4.0`.  The
`N=16` run completed all 8,000 steps; the `N=4` run stopped at `t=11.63`
after 4,652 of 8,000 steps on its `1,150 s` wall budget, so its column is
its own horizon and the task-3 `N=4` column next to it is rescored on that
same horizon.

| quantity, `t=0` to `20`, 8,000 steps | `N=16` task 3 | `N=16` energy-neutral | record, both cones | record, plain archive EOM |
|---|---:|---:|---:|---:|
| all-coordinate scalar RMS | `0.579332` | `0.239172` | `0.128607` | `0.193205` |
| internal-energy sample RMS, Eq. (22) | `2.59371` | `0.069384` | `0.000939628` | `0.0182361` |
| site-0 occupation RMS | `0.199933` | `0.247970` | `0.204690` | `0.286170` |
| N-block scalar RMS | `1.36235` | `0.267235` | --- | --- |
| C-block scalar RMS | `0.132404` | `0.104318` | --- | --- |
| minimum scaled 24-row extension eigenvalue | `-2.401e-5` | `-3.311e-4` | --- | --- |
| minimum unshifted retained joint Gram | `-8.296e-7` | `-8.776e-3` | `1.36819e-5` | --- |
| maximum standardized frontier envelope ratio | `0.9105` | `0.9555` | --- | --- |
| completions solved / refused | 500 / 22 | 500 / 107 | --- | --- |
| repairs applied / skipped of 24,000 | 13,024 / 10,976 | 20,850 / 3,150 | --- | --- |
| constrained repairs / hyperplane not representable | --- | 18,939 / 0 | --- | --- |
| maximum deviation of the energy from its reference | --- | `1.78e-15` | --- | --- |
| repair wall / completion wall | `76.7 s` / `353.3 s` | `117.7 s` / `496.8 s` | --- | --- |
| autonomous wall | `486.2 s` | `669.8 s` | `5,832.34 s` | --- |
| peak resident | `773 MiB` | `782 MiB` | --- | --- |

| quantity, `t=0` to `11.63` | `N=4` task 3 | `N=4` energy-neutral |
|---|---:|---:|
| all-coordinate scalar RMS | `0.138634` | `0.079973` |
| internal-energy sample RMS, Eq. (22) | `0.382336` | `0.051374` |
| site-0 occupation RMS | `0.097051` | `0.068658` |
| N-block scalar RMS | `0.256683` | `0.101002` |
| C-block scalar RMS | `0.064744` | `0.055830` |
| minimum scaled 24-row extension eigenvalue | `-4.518e-6` | `-5.048e-4` |
| minimum unshifted retained joint Gram | `-4.020e-8` | `-1.183e-2` |
| maximum standardized frontier envelope ratio | `0.636` | `0.811` |
| completions solved / refused | 154 refused of 2,000 over the full `t=20` | 503 refused of 1,163 by `t=11.63` |
| repairs applied / skipped | --- | 10,109 / 3,847 |
| constrained repairs / hyperplane not representable | --- | 9,156 / 0 |
| maximum deviation of the energy from its reference | --- | `2.22e-15` |
| repair wall / completion wall | --- | `51.7 s` / `1,066.0 s` |
| autonomous wall | `884.2 s` for the full `t=20` | `1,150.8 s` for `t=11.63` |
| peak resident | `776 MiB` | `520 MiB` |

Sub-horizon scores of the three `N=16` runs, by rescoring the saved
artifacts: task 3 / post-pulse / drive-power.

| `T` | all-coordinate scalar RMS | internal energy | site-0 occupation | N block |
|---:|---|---|---|---|
| 4 | `0.026842` / `0.026842` / `0.023744` | `0.042087` / `0.042087` / `0.020406` | `0.046156` / `0.046156` / `0.042353` | `0.015891` / `0.015891` / `0.011499` |
| 6 | `0.039263` / `0.038367` / `0.036163` | `0.064181` / `0.055125` / `0.022903` | `0.063397` / `0.063171` / `0.064630` | `0.038838` / `0.033087` / `0.021020` |
| 8 | `0.057730` / `0.042542` / `0.045571` | `0.226943` / `0.060603` / `0.024055` | `0.065823` / `0.064505` / `0.074156` | `0.107091` / `0.042990` / `0.035484` |
| 12 | `0.136016` / `0.138493` / `0.125139` | `1.027849` / `0.065626` / `0.025154` | `0.074194` / `0.118314` / `0.100527` | `0.320295` / `0.191237` / `0.177614` |
| 16 | `0.323536` / `0.195096` / `0.258499` | `1.846763` / `0.067999` / `0.025686` | `0.145593` / `0.200532` / `0.165076` | `0.769278` / `0.220164` / `0.397754` |
| 20 | `0.579332` / `0.239172` / `0.328674` | `2.593707` / `0.069384` / `0.026000` | `0.199933` / `0.247970` / `0.186909` | `1.362353` / `0.267235` / `0.519769` |

Four readings.

- **The constraint does nothing until it is switched on.**  The post-pulse
  `N=16` rollout is bitwise equal to the task-3 `N=16` rollout over the whole
  driven interval `t=0` to `4` -- the propagated states, the carried frontier
  and the scored coordinates -- which is why the `T=4` row above repeats to
  every digit.  The default of the option is `"none"` and no default changed
  anywhere.
- **The energy is held exactly, and what it buys is the phonon
  occupation.**  Maximum deviation from the declared reference `1.78e-15`
  over 18,939 constrained repairs at `N=16` and `2.22e-15` over 9,156 at
  `N=4`.  The instantaneous 31-coordinate Euclidean error at `t=20` falls
  from `6.4271` to `2.0538`, of which the N block falls from `5.7088` to
  `0.8312`; the post-pulse excursion of the propagated phonon number falls
  from `7.042` to `0.813`, against the exact reference's own `0.716`.
- **The price is the cone and the conic solve.**  Both minima are worse at
  both cadences, and the declared completion refuses at 4.9 times its
  unconstrained rate at `N=16` (107 of 500 against 22 of 500) and at 5.6
  times it at `N=4` (503 of 1,163 against 154 of 2,000), at 2.3 times the
  mean solve.  That last effect is the reverse of task 3's
  section (c), where keeping the state near the cone was what made the
  warm-started completion cheap and reliable; pulling it onto a hyperplane
  undoes exactly that.  It is also why the `N=4` run did not finish: its
  completion wall alone was `1,066 s` of its `1,150 s` budget.
- **Following the drive power is the better internal-energy repair and the
  worse cone.**  Over `t=0` to `20` the drive-power reference gives internal
  energy `0.026000`, which is 2.7 times the post-pulse `0.069384` the other
  way and within 1.43 of the plain archive EOM's `0.0182361`, and a site-0
  occupation RMS of `0.186909`, better than the record's both-cones
  `0.204690`; its score is `0.328674`, worse than the post-pulse
  `0.239172`, and its minima are `-2.331e-3` on the extension and
  `-5.284e-2` on the retained joint Gram with 276 of 500 completions
  refusing.  Over `t=0` to `4` it is the best row this route has produced:
  `0.023744` at internal energy `0.020406`, against the record's projective
  event guard's `0.026049` and `0.036201` at `3,229.8 s`, inside a 20-unit
  run that cost `719.6 s` in total.

## (c) The phonon-number-neutral variant is not admissible, and was not run

Task (c) asked for a variant holding the trace of the phonon covariance at
its drive-end value *when the model implies it is conserved after the
drive*, and to check the equations of motion first.  It does not.  The
Hamiltonian's electron-phonon term is `g n_q (b_q + b_q^dagger)`, which does
not commute with `b^dagger b`; nothing in the equations of motion holds the
phonon number fixed once the drive has ended, because the coupling keeps
exchanging quanta between the electronic and the phonon sector at every
time.  Section 0 is that statement's number: over `t=4` to `20` the exact
cutoff-16 reference moves its phonon number by `0.716`, which is 233,000
times the `3.07e-6` by which it moves its Eq. (22) energy on the same
interval, in a quantity whose value at `t=4` is `4.017`.  A repair holding
that number would impose a law this model does not have and would be wrong
by about 18 percent of the quantity by `t=20`.  No such run was made.

What the admissible constraint already does to that number is the part worth
recording.  Over `t=4` to `20` the post-pulse excursion of the propagated
phonon number is `7.042` without the constraint, `0.813` with the post-pulse
reference and `1.354` with the drive-power reference, against the exact
reference's `0.716`.  Holding the one linear combination Eq. (22) does
constrain brings the phonon number's post-pulse excursion to within 14
percent of the exact one without ever constraining it.

## Provenance and gates

- `paper_5` suite on the tree these runs were made on: 473 passed in
  `239.12 s` (`cd paper_5 && PYTHONPATH=src python -m pytest -q`), the 467 of
  task 3 plus the six new tests below.
- `paper_5/tests/test_cheap_projection_route.py` now holds 28 tests, the 22
  of task 3 plus six covering the energy-neutral option: that Eq. (22) is
  affine in the propagated coordinates with a gradient that vanishes on the
  entrance and the anomalous coordinates and equals `omega_ph` on each
  diagonal raw normal coordinate; that an energy-neutral repair lands on the
  declared reference to better than `1e-10` while the same repair without the
  option does not, and while the extension minimum still rises; that the
  option changes nothing when it is off even if a reference is supplied;
  that the settings refuse it outside the `dykstra` policy, refuse it with
  `frontier_only`, and refuse an unknown mode; that `post_pulse` leaves the
  reference undeclared before the declared time and holds the energy to
  better than `1e-12` after it; and that `drive_power` moves its reference
  while holding the state's energy on it, with `drive_difference(0.0)`
  exactly zero, which is what makes the drive-free velocity available.
- Modules changed, both owned by this route and neither imported by any
  other.  `cheap_projection_route.py` gained `InternalEnergyFunctional`, the
  `energy_reference_mode` and `energy_tolerance` settings with their
  `holds_internal_energy` predicate, the `energy_feature_functional`
  accessor, the `energy_reference` argument of
  `AffineGramCoordinates.repair`, the `hyperplane` argument of the Dykstra
  alternation, the `energy_reference_time` argument of the integrator, four
  fields on the repair record and seven on the trajectory.
  `cheap_projection_route_analysis.py` gained `--energy-reference-mode`,
  `--energy-reference-time`, `--energy-tolerance`, an `internal_energy`
  block in the summary and the two energy series in `trajectory.npz`.
  Nothing is committed and no default changed; the bitwise reading in
  section (b) is the demonstration on a 1,600-step rollout, and the tests
  are the demonstration on the repair itself.
- The tables are produced by two scripts kept with the runs, each reading
  only `trajectory.npz` and `summary.json` of the run directories named on
  its command line and propagating nothing: `score_energy_rows.py`
  (sub-horizon scores, blockwise RMS, physicality minima) and
  `check_post_pulse_invariants.py` (the post-pulse spread of the Eq. (22)
  energy and of the phonon number, on the exact reference and on the
  propagated trajectory), both under
  `output/local_runs/paper_v_cheap_projection_energy_pilot_n16_t6_20260915_v1/`.
- Artifacts, under `output/local_runs/`:
  `paper_v_cheap_projection_energy_pilot_n16_t6_20260915_v1` (the `t=6`
  pilot, `124.2 s`, `420 MiB`),
  `paper_v_cheap_projection_energy_n16_t20_20260915_v1`,
  `paper_v_cheap_projection_energy_n4_t20_20260915_v1` (stopped at `t=11.63`
  on its wall budget) and
  `paper_v_cheap_projection_energy_power_n16_t20_20260915_v1`.  Every run was
  wrapped in `pipelines/shell/ram_guard.py --limit-mb 3000`, one at a time;
  every peak-resident figure quoted is the run's own
  `peak_resident_bytes` (`ru_maxrss` of the propagating process).  No run
  exceeded 20 minutes.  Wall times were measured with other agents' jobs on
  the machine (load average 4.3 to 4.7), so they are upper bounds.

## What to try next, in order

1. The constraint and the conic solve now fight each other, and that is the
   one thing standing between this route and a completed `N=4` run to
   `t=20`: 503 of 1,163 completions refused and the completion wall was
   `1,066 s` of the `1,150 s` budget.  The record's completion retry ladder
   exists for exactly this and costs nothing when no solve refuses.  Running
   `N=4` with the ladder, or with a declared `energy_tolerance` above zero so
   the hyperplane is met only where the drift has accumulated, is one run and
   decides whether the `t=0` to `11.63` numbers hold to `t=20`.
2. A third reference mode whose reference at each repair is the energy the
   state carried *before* that repair is the record's `energy_neutral`
   control restriction in projection form: it removes the repair's own
   contribution to the energy and leaves the closure's drift untouched.  It
   would split the 37-fold improvement measured here into the part that was
   the repair's fault and the part that was the velocity's, which is the
   split task 3 measured at `t=4` and found to be `1.38`.
3. The retained block is where the constraint does its damage
   (`-8.776e-3` at `N=16`, `-1.183e-2` at `N=4`).
   `retained_eigenvalue_floor` is declared on this route and has never been
   used; raising it would push that block interior before the hyperplane
   pulls it out, at no extra cost per repair.

## Structured summary of task 4

| item | value |
|---|---|
| what was added | a named energy-neutral option of the Dykstra repair: the alternation gains a third declared set, the hyperplane on which the Eq. (22) internal energy of the mapped-back coordinates equals a declared reference, with two reference modes (`post_pulse`, `drive_power`); default `"none"`, no default changed |
| why it is exact | Eq. (22) on the coordinates the scorer uses is a linear functional of the 29 raw coordinates (offset zero, reconstruction `2.2e-16`), carried by seven of the 106 parameters, `omega_ph` on each diagonal raw normal coordinate and zero on the anomalous, entrance and frontier ones |
| `t=0` to `20`, `N=16`, post-pulse reference | `0.239172` against task 3's `0.579332`; internal energy `0.069384` against `2.59371`; N block `0.267235` against `1.36235`; site-0 occupation `0.247970` against `0.199933`; extension `-3.311e-4` against `-2.401e-5`; retained joint Gram `-8.776e-3` against `-8.296e-7`; 500 solved / 107 refused against 500 / 22; `669.8 s` against `486.2 s`; `782 MiB` |
| `t=0` to `11.63`, `N=4`, post-pulse reference | `0.079973` against task 3's `0.138634` on that horizon; internal energy `0.051374` against `0.382336`; N block `0.101002` against `0.256683`; site-0 occupation `0.068658` against `0.097051`; extension `-5.048e-4` against `-4.518e-6`; retained joint Gram `-1.183e-2` against `-4.020e-8`; 1,163 solved / 503 refused; stopped on its `1,150 s` wall budget after 4,652 of 8,000 steps |
| `t=0` to `20`, `N=16`, drive-power reference | `0.328674`, internal energy `0.026000`, site-0 occupation `0.186909`, N block `0.519769`, extension `-2.331e-3`, retained joint Gram `-5.284e-2`, 500 solved / 276 refused, `719.6 s`, `768 MiB`; over `t=0` to `4` it scores `0.023744` at internal energy `0.020406` against the record's projective event guard's `0.026049` and `0.036201` at `3,229.8 s` |
| how exactly the energy is held | maximum deviation from the declared reference `1.78e-15` over 18,939 constrained repairs (`N=16`), `2.22e-15` over 9,156 (`N=4`); the hyperplane was representable at every one of them |
| against the record over `t=0` to `20` | both cones `0.128607` / `0.000939628` / site-0 `0.204690` at `5,832.34 s`; plain archive EOM `0.193205` / `0.0182361` / site-0 `0.286170`.  The best constrained score here is 1.86 times the both-cones score and 1.24 times the archive EOM; the best constrained internal energy is 27.7 times the both-cones value and 1.43 times the archive EOM's |
| what it costs | the cone and the conic solve: one to four decades on the two minima at both cadences, a completion refusal rate 4.9 times its unconstrained value at `N=16` and 5.6 times it at `N=4`, and 2.3 times the mean solve at `N=4`, which is the reverse of task 3's finding that keeping the state near the cone is what makes the warm-started completion cheap |
| (c) phonon-number-neutral variant | **not admissible and not run.**  `g n (b + b^dagger)` does not commute with `b^dagger b`; the exact reference's phonon number moves by `0.716` over `t=4` to `20` against `3.07e-6` for its energy.  The admissible constraint already brings the propagated phonon number's post-pulse excursion from `7.042` to `0.813`, against the exact `0.716` |
| build-time fact answering task 3's next step 3 | the 24-row extension does not constrain the phonon normal block -- one of the four raw normal coordinates lies entirely in the 15-dimensional null space of its affine map and two more lie 83 percent in it -- while the two targets together have full column rank 106 of 106 |
| gates | `paper_5` suite 473 passed in `239.12 s`; 28 tests in `test_cheap_projection_route.py` (22 of task 3 plus six) |
| files changed | `cheap_projection_route.py`, `cheap_projection_route_analysis.py`, `test_cheap_projection_route.py`, this report, and the worklog block in `paper_5/notes/electron_phonon_closure_worklog_20260803.md`.  Nothing committed |

# Task 5: the chained `N=4` energy-neutral runs to `t=20`, and the completion retry ladder (2026-09-15/16)

Same branch, worktree and modules as the four sections above.  The two
modules this route owns gained three things the task-4 list asked for, and
the tests that cover them: the record's declared completion retry ladder,
built by the record's own `build_completion_ladder` and walked only where a
lift re-selection's declared completion is not certified; a declared
`energy_tolerance` above zero, so the hyperplane is met only where the drift
has accumulated; and `--resume-from`, which continues a rollout that stopped
on its wall budget from its last accepted state, its carried frontier, its
time and the declared internal-energy reference the first segment selected,
so that a `t=20` run at `N=4` can be made of segments that each obey the
twenty-minute rule.  No file this route does not own was edited.  The
previous instance of this key launched the three chains, saw the last one
reach `t=20`, and was killed while scoring; this pass propagated nothing
except the exact cutoff-16 driven reference on the chained sample times,
which every score needs.

Conventions are those of tasks 2 to 4: the score is the all-coordinate
scalar RMS of the 31 closed scalar coordinates against the exact cutoff-16
driven reference (`all_coordinate_scalar_rms_error`) on the stated horizon;
the internal energy is the sample RMS of the time-independent energy of
Eq. (22) (`static_energy_rms_error`) on that horizon; physicality numbers
are measured minimum eigenvalues along the trajectory and there is no
certificate anywhere on this route; the record's `0.128607`, `0.193205`,
`0.000939628` and `0.0182361` are the raw convention, and every
fixed-dictionary run here carries a `t=0` offset of `8.08e-16`, so the raw
and offset-subtracted conventions agree to every digit printed.
Configuration for every row: strong coupling (`lambda_ep = 1.5`,
`gamma = 0.5`, drive amplitude 1), phonon cutoff 16, exact-correlated
preparation, 15 entrance keys, SSPRK(3,3) at `h = 0.0025`, the unguarded
fixed-dictionary predictor, Dykstra 30 on both targets at stage cadence
between re-selections, the declared proximal log-det completion re-solved
every 4 steps warm-started on the carried frontier, and the energy-neutral
option of task 4 in the mode each row names.

## Negative result first

Neither completed `N=4` energy-neutral chain stands next to the record's
both-cones line, and both are worse over `t=0` to `20` than the
unconstrained `N=4` hybrid they were meant to improve.  The post-pulse chain
scores `0.318110` and the drive-power chain `0.283805`, against the
unconstrained `N=4` row's `0.259573`, the record's both-cones `0.128607` and
the record's plain archive EOM `0.193205`; their Eq. (22) internal-energy
sample RMS is `0.0539267` and `0.0437311` against `0.000939628` and
`0.0182361`.  Both lose the retained cone (`-2.632e-2` and `-1.777e-2`
against the both-cones run's `1.36819e-5`) and both carry the standardized
frontier to the edge of its frozen box (`0.9966` and `0.9938`).  The
completion refuses 1,270 and 1,165 of 2,000 solves; in the last segment of
the post-pulse chain, `t=18.27` to `20`, all 173 declared completions
refuse, so its last 1.73 time units are the frozen-lift route of task 2 with
the hyperplane added.

The `t=0` to `11.63` advantage task 4 reported for `N=4` (`0.079973`
against the unconstrained `0.138634`) is also not a reproducible number.
The post-pulse chain differs from the task-4 run only in repairs skipped
while the sampled energy sat within `8.9e-6` of its reference, coincides
with it over `t=0` to `4` to every digit, separates past `1e-2` in the
31-coordinate Euclidean norm at `t=6.93` and past `0.1` at `t=8.38`, and
scores `0.132357` on that same horizon.  Past about `t=8` the post-pulse
`N=4` row is one sample of a sensitive trajectory, and its digits there
should not be quoted against another row's.

The record's completion retry ladder does not fix the constrained
completion's refusals: walked at 493 refused re-selections it certified 119,
all before `t=8.19`, and none of the 289 after, at `3.87 s` a solve; that
chain stopped at `t=11.09` and was not continued.

## (a) The three variants, and how far each got

Every segment is its own `ram_guard --limit-mb 3000` process with a
`1,150 s` wall budget, resumed from the previous segment's last accepted
state and carried frontier; no two ever ran at once.  Counts are each
segment's own `summary.json`; the score script checks that each segment
starts at the state the previous one ended on, to `1e-12`.

| variant | reference mode | `energy_tolerance` | ladder | segments (`t` reached) | autonomous wall | completions re-solved / refused | of those, refused per segment | why the chain ended |
|---|---|---:|---|---|---:|---:|---|---|
| A | `post_pulse` at `t=4` | `0` | shifts `1e-8`, `1e-7`; weight `0.1` | `0` to `8.19`, `8.19` to `11.09` | `2,322.5 s` | 1,109 / 493 (374 after the ladder) | 85 of 819; 289 of 290 | stopped at `t=11.09` on the second segment's wall budget, not continued: the ladder was exhausted at all 289 refusals of that segment and the mean solve was `3.87 s`, so `t=20` would have needed three more segments |
| B | `post_pulse` at `t=4` | `1e-3` | none | `0` to `11.14`, to `18.27`, to `20` | `2,467.4 s` | 2,000 / 1,270 | 447 of 1,114; 650 of 713; 173 of 173 | reached `t=20` |
| C | `drive_power` | `1e-3` | none | `0` to `6.73`, to `14.71`, to `20` | `3,179.8 s` | 2,000 / 1,165 | 357 of 673; 379 of 798; 429 of 529 | reached `t=20` |

The declared tolerance never bound.  In B the sampled energy sits within
`8.92e-6` of its reference at every one of 6,401 post-pulse samples, in C
within `3.91e-14` at every one of 8,001, and no sample of either falls
outside `1e-3`; the tolerance changed only the 2,043 stages of B between
`t=4` and `11.14` at which both cones were met and the repair was skipped
(against none skipped in the task-4 run), and in C the 1,213 skipped
stages.  What reached `t=20` was the chaining, not the tolerance and not
the ladder.  The ladder's cost is the other half of its verdict: 1,306
retry solves took `1,408.3 s` of A's `2,322.5 s`, and with it A reached
`t=11.09` in two segments where B reached `t=11.14` in one.

Where the completion wall goes is unchanged from task 4: `2,305.9 s` of
B's `2,467.4 s` and `2,936.9 s` of C's `3,179.8 s` are the declared
completions, 93 percent of each; the repair costs `106.7 s` and `133.9 s`,
the velocity `53.0 s` and `107.0 s`.  Peak resident (each run's own
`peak_resident_bytes`, `ru_maxrss` of the propagating process): `507 MiB`
(B), `382 MiB` (C), `400 MiB` (A).

## (b) `t=0` to `20` next to the record and the 27-key chain

| quantity, `t=0` to `20`, 8,000 steps | B: `N=4`, `post_pulse`, tol `1e-3` | C: `N=4`, `drive_power`, tol `1e-3` | task 3: `N=4`, unconstrained | task 4: `N=16`, `post_pulse` | task 4: `N=16`, `drive_power` | projective task 4: 27 keys, `N=16`, `drive_power` | record, both cones | record, plain archive EOM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all-coordinate scalar RMS | `0.318110` | `0.283805` | `0.259573` | `0.239172` | `0.328674` | `0.085193` | `0.128607` | `0.193205` |
| internal-energy sample RMS, Eq. (22) | `0.0539267` | `0.0437311` | `0.804515` | `0.069384` | `0.026000` | `0.0063446` | `0.000939628` | `0.0182361` |
| site-0 occupation RMS | `0.212252` | `0.290898` | `0.203657` | `0.247970` | `0.186909` | `0.112590` | `0.204690` | `0.286170` |
| N-block scalar RMS | `0.4805` | `0.3089` | `0.532930` | `0.267235` | `0.519769` | `0.03187` | --- | --- |
| C-block scalar RMS | `0.1715` | `0.1207` | `0.0917819` | `0.104318` | --- | `0.031283` | --- | --- |
| rho / B-block scalar RMS | `0.2546` / `0.3848` | `0.3435` / `0.5254` | `0.2466` / `0.3187` | --- | --- | `0.1358` / `0.1873` | --- | --- |
| minimum scaled 24-row extension eigenvalue | `-1.153e-3` | `-6.815e-4` | `-5.553e-6` | `-3.311e-4` | `-2.331e-3` | `-2.040e-4` (36-row) | --- | --- |
| minimum unshifted retained joint Gram | `-2.632e-2` | `-1.777e-2` | `-8.975e-8` | `-8.776e-3` | `-5.284e-2` | `-2.463e-3` | `1.36819e-5` | --- |
| maximum standardized frontier envelope ratio | `0.9966` | `0.9938` | `0.6356` | `0.9555` | --- | --- | --- | --- |
| completions solved / refused | 2,000 / 1,270 | 2,000 / 1,165 | 2,000 / 154 | 500 / 107 | 500 / 276 | 500 / 144 | --- | --- |
| repairs applied / skipped of 24,000; constrained | 18,110 / 5,890; 17,157 | 22,787 / 1,213; 22,787 | 5,643 / 18,357; 0 | 20,850 / 3,150; 18,939 | --- | 23,959 / 41; 23,959 | --- | --- |
| reference above the exact `-3.571397982` at `t=4` | `0.057286` | `0.045186` | --- | `0.074667` | `0.027219` | `0.006710` | --- | --- |
| autonomous wall | `2,467.4 s` (3 segments) | `3,179.8 s` (3) | `884.2 s` | `669.8 s` | `719.6 s` | `4,794.4 s` (6) | `5,832.34 s` | --- |

Cumulative scores from `t=0`, score / internal energy, by rescoring the
saved artifacts (`chain_score_*.json` under the task-5 score directory; the
27-key column is the projective report's table):

| `T` | B | C | task 3, `N=4` | task 4, `N=16` `post_pulse` | task 4, `N=16` `drive_power` | 27 keys, `N=16` `drive_power` |
|---:|---|---|---|---|---|---|
| 4 | `0.026290` / `0.037608` | `0.047670` / `0.037386` | `0.026290` / `0.037608` | `0.026842` / `0.042087` | `0.023744` / `0.020406` | `0.011427` / `0.004597` |
| 8 | `0.046722` / `0.048453` | `0.126944` / `0.041465` | `0.049575` / `0.091559` | `0.042542` / `0.060603` | `0.045571` / `0.024055` | `0.025931` / `0.005752` |
| 12 | `0.146385` / `0.051566` | `0.184550` / `0.042739` | `0.145974` / `0.408489` | `0.138493` / `0.065626` | `0.125139` / `0.025154` | `0.042509` / `0.006088` |
| 16 | `0.264509` / `0.053054` | `0.240903` / `0.043362` | `0.201714` / `0.578188` | `0.195096` / `0.067999` | `0.258499` / `0.025686` | `0.063773` / `0.006250` |
| 20 | `0.318110` / `0.053927` | `0.283805` / `0.043731` | `0.259573` / `0.804515` | `0.239172` / `0.069384` | `0.328674` / `0.026000` | `0.085193` / `0.006345` |

Four readings.

- **The tighter cadence is the worse one once the energy is held.**  Over
  `t=0` to `20` the `N=4` chains are 1.33 and 1.19 times the `N=16`
  post-pulse row, where without the constraint `N=4` was 2.23 times *better*
  than `N=16` (task 3).  The reason is in the refusal column: the constraint
  makes the warm-started completion refuse at 58 to 64 percent of
  re-selections at `N=4` against 21 percent at `N=16`, and a refusal keeps a
  carried lift the hyperplane then pulls on for another four steps.  More
  re-selections that mostly refuse are not more re-selections.
- **What the hyperplane holds, it holds; what it does not, it makes worse.**
  The internal energy of B and C is 15 and 18 times better than the
  unconstrained `N=4` row's `0.804515`, and C's phonon-occupation block
  (`0.3089`) is the best of the three `N=4` rows; but C's electronic density
  and coherent-phonon blocks are `0.3435` and `0.5254` against the
  unconstrained `0.2466` and `0.3187`, and its site-0 occupation `0.290898`
  is worse than the plain archive EOM's `0.286170`.  That is the same
  direction the 27-key chain reports for its rho and B blocks; at 15 keys
  and `N=4` it outweighs the phonon-occupation gain.
- **The 27-key chain is the only row in the table below the both-cones
  score**, `0.085193` against `0.128607`, at `4,794.4 s` against
  `5,832.34 s`; on the internal energy it is 6.8 times the both-cones value
  and on the retained cone it sits at `-2.463e-3`.  Every 15-key row is
  above both record lines on the score.
- **Drive power at `N=4` is worse than at `N=16` over the driven
  interval.**  C scores `0.047670` over `t=0` to `4` where the `N=16`
  drive-power row scored `0.023744`, with the completion refusing 357 of
  673 solves by `t=6.73`.  The likeliest amplifier is the one above: at
  `N=4` the constrained state reaches the completion four steps after the
  last hyperplane pull rather than sixteen.

## (c) `t=0` to `4` next to the projective event guard

| row, `t=0` to `4` | all-coordinate scalar RMS | internal energy | site-0 occupation | N block | minimum unshifted retained joint Gram | wall to `t=4` |
|---|---:|---:|---:|---:|---:|---:|
| B (`post_pulse` is off before `t=4`; bitwise the task-3 `N=4` row) | `0.0262899` | `0.037608` | `0.0487012` | `0.01479` | `-4.020e-8` | `137.9 s` (task 2) |
| A (ladder; the lift differs from B's at the 119 certified retries) | `0.0263373` | `0.0382408` | `0.0492229` | `0.01548` | `-5.183e-8` | inside segment 1 |
| C (`drive_power`, `N=4`) | `0.0476697` | `0.0373856` | `0.0755899` | `0.05205` | `-1.771e-3` | inside segment 1 (`1,151.1 s` to `t=6.73`) |
| task 4: `N=16`, `drive_power` | `0.023744` | `0.020406` | `0.042353` | `0.011499` | --- | inside a `719.6 s` run to `t=20` |
| projective task 4: 27 keys, `N=16`, `drive_power` | `0.011427` | `0.004597` | `0.014309` | `0.009836` | `-4.535e-4` | about `1,022 s` |
| record: projective event guard | `0.026049` | `0.036201` | --- | --- | `2.892e-5` | `3,229.8 s` |

A's `t=0` to `4` row differs from B's in the fourth digit because the
ladder certified retries inside the driven interval and each certified
retry is a different lift; the two trajectories are `1e-3` apart in the
31-coordinate Euclidean norm from `t=0.815` and `1e-2` apart from `t=2.90`.

## (d) The post-pulse `N=4` row past `t=8` is one sample of a sensitive trajectory

Three rollouts of the post-pulse `N=4` configuration exist: the task-4 run
(tolerance `0`, stopped at `t=11.63`), B (tolerance `1e-3`) and A (tolerance
`0` with the ladder).  Euclidean distance between the 31 closed scalar
coordinates of B and of the task-4 run, and of A and of the task-4 run, at
the same times:

| `t` | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B against task 4 | `0` | `2.3e-4` | `5.3e-3` | `1.1e-2` | `5.4e-2` | `2.3e-1` | `3.9e-1` | `1.07` |
| A against task 4 | `1.9e-2` | `2.7e-2` | `2.9e-2` | `3.9e-2` | `1.6e-1` | `5.2e-1` | `1.28` | `1.78` |

and the three scores on the common horizon `t=0` to `11.09`: task 4
`0.072875`, B `0.110625`, A `0.163635`, against the unconstrained `N=4`
row's `0.127335` there.  B and the task-4 run are the same declared
configuration up to repairs skipped while the energy drift was below
`8.9e-6`; that perturbation, of order `1e-5` in the coordinates, is `1e-2`
by `t=6.93`, `0.1` by `t=8.38` and order one by `t=11`.  The amplifier was
not measured separately; the candidate is the completion's refusal, which
is a discontinuous function of the state and flips at 40 to 100 percent of
re-selections on this interval, so a state difference below the solver's
tolerance can exchange a re-selected lift for a carried one.  What follows
for the record is that the task-4 `t=0` to `11.63` comparison
(`0.079973` against `0.138634`) is not a factor-1.7 result: on the same
horizon B scores `0.132357`, 0.95 of the unconstrained row.  The
`t=0` to `8` scores of all three, `0.046854`, `0.046722` and `0.046972`,
agree to three digits and are the reproducible part.

## (e) What stands next to the both-cones line

- **B, `N=4` `post_pulse` to `t=20`, `0.318110`**: a containment only; it
  reaches `t=20` with the energy on its reference, at 2.47 times the
  both-cones score, 1.65 times the plain archive EOM, worse than the
  unconstrained `N=4` row, with the retained cone at `-2.6e-2` and the
  frontier at the edge of its box.
- **C, `N=4` `drive_power` to `t=20`, `0.283805`**: a containment only; 2.21
  times the both-cones score, 1.47 times the archive EOM, internal energy
  2.4 times the archive EOM's, site-0 occupation worse than the archive
  EOM's, retained cone at `-1.8e-2`.
- **A, `N=4` `post_pulse` with the ladder, stopped at `t=11.09`,
  `0.163635` there**: not a result at any horizon; what it establishes is
  that the record's retry ladder certifies nothing past `t=8.19` under the
  constraint and triples the solve, so it is not the refusal fix.
- **Task 3, `N=4` unconstrained, `0.259573`**: still the best 15-key `N=4`
  row at `t=20` and the only one of the four that holds both cones
  (`-8.975e-8`); at 2.02 times the both-cones score it is a containment with
  both cones held, not a result.
- **Task 4, `N=16` `post_pulse`, `0.239172`**: the best 15-key row at `t=20`
  on this route, 1.86 times the both-cones score; a containment.
- **Projective task 4, 27 keys `N=16` `drive_power`, `0.085193`**: the one
  row that stands next to the both-cones line on the score and on the wall
  (0.66 of the score at 0.82 of the wall), and it stands there without a
  physicality statement, at 6.8 times the both-cones internal energy and
  `-2.463e-3` on the retained cone; the both-cones run remains the only
  `t=20` row that holds both cones.

## Provenance and gates

- `paper_5/tests/test_cheap_projection_route.py`: 34 passed in `15.87 s`
  (`cd paper_5 && PYTHONPATH=src python -m pytest -q
  tests/test_cheap_projection_route.py`), the 28 of task 4 plus six: that the
  ladder this route builds is the record's own `build_completion_ladder` for
  the declared shifts `1e-8`, `1e-7` and weight `1e-1`; that a refused
  re-selection walks it and takes the first certified rung; that an
  exhausted ladder keeps the carried lift and stops nothing; that a declared
  `energy_tolerance` skips a repair the cones do not need and tolerance zero
  applies it onto the reference; that a supplied reference is the one a
  resumed segment holds, and is refused without a mode; and that two chained
  segments reproduce one rollout to `1e-12` in state and frontier with the
  same re-selection count.  The full `paper_5` suite was not re-run in this
  pass; the previous instance's last full run was task 4's 473.
- Modules changed, both owned by this route and neither imported by any
  other: `cheap_projection_route.py` gained the `completion_retry_shifts`
  and `completion_retry_weights` arguments of `CheapProjectionRoute` and
  `prepare_cheap_projection_route`, the ladder walk inside `reselect_lift`
  with its counters, the `energy_reference_value` argument of the integrator
  and six trajectory fields; `cheap_projection_route_analysis.py` gained
  `--completion-retry-shifts`, `--completion-retry-weights`,
  `--resume-from`, `_resume_state` (which refuses a segment whose
  parameters, cutoff, dictionary or step differ and records changed repair
  settings), a `resume` block and ladder counters in the summary, and an
  exact reference that is propagated from `t=0` and sampled on a resumed
  segment's own times.  Every default is unchanged: an empty ladder, a zero
  tolerance and no resume are the earlier form.  Nothing is committed.
  `HEAD` moved from `2d8f0efb` to `0615282f` during this task (the
  projective agent's report); that commit does not touch these modules.
- Scoring: `score_segments.py` under
  `output/local_runs/paper_v_cheap_projection_task5_score_20260915_v1/`
  reads only `trajectory.npz` and `summary.json` of the segments named on
  its command line, checks the chain boundaries, regenerates the exact
  reference on the chained times and scores the horizons; its outputs are
  `chain_score_{power_tol_n4,postpulse_tol_n4,postpulse_ladder_n4,task4_postpulse_n4,task3_unconstrained_n4}.json`.
  The two single-directory rescorings reproduce the task-3 and task-4 rows
  in every printed digit.  The energy-drift and separation figures are from
  two scratch scripts reading the same arrays.
- Artifacts, under `output/local_runs/`:
  `paper_v_cheap_projection_energy_ladder_n4_t20_20260915_v1_seg{1,2}`,
  `paper_v_cheap_projection_energy_tol_n4_t20_20260915_v1_seg{1,2,3}`,
  `paper_v_cheap_projection_energy_power_tol_n4_t20_20260915_v1_seg{1,2,3}`,
  `paper_v_cheap_projection_resume_smoke_{a,b}_20260915_v1` (the resume
  smoke, two 2-step segments).  Every segment ran under
  `pipelines/shell/ram_guard.py --limit-mb 3000`, one at a time, with a
  `1,150 s` autonomous budget; the longest segment was `1,172.5 s`.  Wall
  times were measured with other agents' jobs on the machine and are upper
  bounds.

## What to try next, in order

1. Nothing on this route at `N=4`.  The constrained completion refuses more
   often the tighter the cadence, and the ladder does not change that; the
   long-horizon cadence for the energy-neutral repair is `N=16`, and the
   dictionary lever the projective agent measured moves the score more than
   anything measured here.
2. The sensitivity of section (d) is the first thing to settle before any
   `t>8` row on this route is compared with another at the third digit: two
   seeds of the same configuration, or the same run with the completion's
   conic tolerance moved one decade, would say whether the refusal is the
   amplifier.
3. The record's `energy_neutral` control restriction in projection form
   (task 4's item 2) is still the unrun variant that would separate the
   repair's energy drift from the velocity's.

## Structured summary of task 5

| item | value |
|---|---|
| what was added | the record's completion retry ladder at a refused lift re-selection; a declared `energy_tolerance`; `--resume-from`, which chains segments from the last accepted state, carried frontier, time and the first segment's declared reference; 34 tests (6 new); no default changed; nothing committed |
| B: `N=4`, `post_pulse`, tol `1e-3`, `t=0` to `20`, 3 segments, `2,467.4 s` | `0.318110`, internal energy `0.0539267`, site-0 `0.212252`, N block `0.4805`, C block `0.1715`, extension `-1.153e-3`, retained joint Gram `-2.632e-2`, envelope `0.9966`, 2,000 solved / 1,270 refused (173 of 173 in the last segment), energy within `8.92e-6` of its reference at every sample, `507 MiB` |
| C: `N=4`, `drive_power`, tol `1e-3`, `t=0` to `20`, 3 segments, `3,179.8 s` | `0.283805`, internal energy `0.0437311`, site-0 `0.290898`, N block `0.3089`, C block `0.1207`, rho / B `0.3435` / `0.5254`, extension `-6.815e-4`, retained joint Gram `-1.777e-2`, envelope `0.9938`, 2,000 solved / 1,165 refused, energy within `3.91e-14` at every sample, `382 MiB` |
| A: `N=4`, `post_pulse`, tol `0`, ladder | stopped at `t=11.09` after two segments (`2,322.5 s`), not continued; the ladder certified 119 of 493 refusals, all before `t=8.19`, and 0 of 289 after, at `3.87 s` a solve; `0.163635` over `t=0` to `11.09`, the worst of the three variants there |
| against the record over `t=0` to `20` | both cones `0.128607` / `0.000939628` / site-0 `0.204690` / retained `1.36819e-5` at `5,832.34 s`; plain archive EOM `0.193205` / `0.0182361` / `0.286170`.  B is 2.47 and 1.65 times the two scores, C 2.21 and 1.47; both are worse than the unconstrained `N=4` row's `0.259573` and the `N=16` post-pulse row's `0.239172` |
| against the 27-key `N=16` drive-power chain | `0.085193` / `0.0063446` / site-0 `0.112590` / retained `-2.463e-3` at `4,794.4 s`: the only row below the both-cones score, 6.8 times its internal energy, no physicality statement |
| `t=0` to `4` against the projective event guard `0.026049` / `0.036201` | B `0.0262899` / `0.037608` (bitwise task 3); A `0.0263373` / `0.0382408`; C `0.0476697` / `0.0373856`; the `N=16` drive-power row `0.023744` / `0.020406`; 27 keys `0.011427` / `0.004597` |
| the tolerance | never bound: no sample of B or C is outside `1e-3`; it changed 2,043 skipped stages of B and 1,213 of C |
| the sensitivity | B and the task-4 run, identical up to those skipped repairs, separate to `1e-2` at `t=6.93`, `0.1` at `t=8.38`, order one at `t=11`; scores on `t=0` to `11.09`: `0.072875` (task 4), `0.110625` (B), `0.163635` (A); the task-4 `t=0` to `11.63` advantage is not reproducible (B `0.132357` against the unconstrained `0.138634`) |
| what stands next to the both-cones line | none of the 15-key rows; B and C are containments, A is a negative result about the ladder, the 27-key chain stands on the score axis alone |
| gates | 34 tests in `test_cheap_projection_route.py` passed in `15.87 s`; full suite not re-run this pass |
| files changed | `cheap_projection_route.py`, `cheap_projection_route_analysis.py`, `test_cheap_projection_route.py` (all from the previous instance, kept), `score_segments.py` (four lines, tolerates the task-3 summary schema), this report, and the worklog block.  Nothing committed |

# Task 6: the sensitivity of the recorded `N=16` drive-power row to one declared setting (2026-09-16)

Same branch, worktree and modules as the five sections above.  The two
modules this route owns gained one declared setting, needed for one of the
three reruns and absent from the driver before: `--completion-conic-tolerance`,
which builds a second `SymmetryReducedPositiveExtension` on the same
dictionary whose conic tolerance the lift re-selections solve at, leaving
the preparation's completion at the declared `APCMExtensionSettings`
(`CheapProjectionRoute(reselection_extension=...)`,
`prepare_cheap_projection_route(reselection_extension_settings=...)`,
recorded in the summary's `reselection_extension_settings` block; `None`
is the earlier form, one extension for both).  No file this route does not
own was edited; no default changed; nothing is committed.  Conventions are
those of tasks 2 to 5: the score is the all-coordinate scalar RMS of the 31
closed scalar coordinates against the exact cutoff-16 driven reference
(`all_coordinate_scalar_rms_error`) on the stated horizon; the internal
energy is the sample RMS of the time-independent energy of Eq. (22)
(`static_energy_rms_error`) on that horizon; physicality numbers are
measured minimum eigenvalues along the trajectory and there is no
certificate anywhere on this route.  The configuration of every row is the
recorded run's, `paper_v_cheap_projection_energy_power_n16_t20_20260915_v1`
(task 4, section (b), the cheapest configuration with a `t=20` row,
`719.6 s`): strong coupling (`lambda_ep = 1.5`, `gamma = 0.5`, drive
amplitude 1), phonon cutoff 16, exact-correlated preparation, 15 entrance
keys, SSPRK(3,3) at `h = 0.0025`, the unguarded fixed-dictionary predictor,
Dykstra 30 on both targets at stage cadence, the declared proximal log-det
completion re-solved every 16 steps warm-started on the carried frontier,
the energy-neutral option in `drive_power` mode with `energy_tolerance` 0,
`8,000` steps to `t=20`.  Each rerun differs from that in exactly one
declared setting.  The separation between two rollouts is the Euclidean
distance between their 31 closed scalar coordinates at the same accepted
step; "first crossing" is the first sample at or above the level.

## Negative result first

The tighter decade cannot be applied to the completion as declared.  At
`conic_tolerance = 1e-10` the preparation's proximal log-det completion is
not certified (Clarabel `AlmostSolved`, `relative_lambda 2.0e-7`,
`reduced_gradient 3.7e-9`, face support 23 of 24), so
`prepare_projected_apcm_initial_state` refuses and no route can be built;
the rerun below tightens the re-selection completion alone and keeps the
preparation's solve at `1e-9`.  The same refusal is the first thing that
rerun does at `t=0.04`: where the declared `1e-9` solve certifies a lift
`0.333` from the carried one, the `1e-10` solve on the same lower moments
and the same warm start (both runs are bitwise equal to that step) refuses
and keeps the carried lift.

No horizon of this configuration is resolved at the second digit.  The
three reruns move the `t=0` to `4` score from the recorded `0.023744` to
`0.024240`, `0.026837` and `0.027953` (spread 18 percent of the smallest),
the `t=0` to `8` score from `0.045571` to `0.044530`, `0.040677` and
`0.040450` (13 percent), `t=0` to `12` from `0.125139` to `0.130946`,
`0.114863` and `0.108195` (21 percent), and `t=0` to `20` from `0.328674`
to `0.325970`, `0.260123` and `0.300340` (26 percent); the internal energy
moves by 40 percent at both `t=4` and `t=20`.  The recorded row's lead over
the projective event guard on `t=0` to `4` (`0.023744` against `0.026049`,
task 4) is inside that spread: two of its three one-setting twins score
worse than the guard.

The amplifier is the lift re-selection's completion, not the flow and not
the repair.  In every rerun the 31 coordinates agree with the recorded run
to round-off (`5e-16` to `3e-15`) or exactly until a re-selection step,
where the standardized frontier the two runs carry jumps apart by `0.33` to
`0.40` in one step; the propagated state follows one step later, from
`1e-14` to `1.2e-3`, `2.3e-3` and `1.4e-3` in the 44 coordinates (one-step
growth `4.7e10` and `4.9e10` where the difference before it was nonzero),
and every later one-step growth of the 31-coordinate separation is at most
`4.0`.  Two mechanisms were caught: a refusal flip, where one run's
completion certifies and the other's refuses on inputs that agree to
`1e-14` (`dykstra60` at `t=0.08`, `conic1e10` at `t=0.04`); and two
certified solves on inputs that agree to `1e-14` returning lifts `0.40`
apart (`etol1e3` at `t=0.04`).  Beside this the exact reference is inert:
the record's smallest physical perturbation is amplified by at most `1.16`
over `t=0` to `20`, at the record's size and at the route's.

## (a) The three reruns

Every rerun completed its 8,000 steps under `ram_guard --limit-mb 3000`,
one at a time, inside the recorded `1,140 s` budget.  Counters are each
run's own `summary.json`; the exact reference stored with each rerun equals
the recorded run's in every one of `8,001 x 31` entries.

| run | the one setting changed | autonomous wall | completions solved / refused, mean solve | repairs applied / skipped of 24,000, largest Dykstra count | minimum scaled 24-row extension eigenvalue | minimum unshifted retained joint Gram | maximum standardized frontier envelope ratio | drive-power reference above the exact `-3.571397982` | peak resident |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recorded (task 4) | --- | `719.6 s` | 500 / 276, `0.966 s` | 22,212 / 1,788, 30 | `-2.331e-3` | `-5.284e-2` | `0.8422` | `0.027219` | `768 MiB` |
| `etol1e3` | `energy_tolerance` 0 to `1e-3` (the flag of task 5's chain B) | `702.4 s` | 500 / 285, `0.959 s` | 19,925 / 4,075, 30 | `-2.042e-3` | `-4.637e-2` | `0.9238` | `0.027382` | `774 MiB` |
| `dykstra60` | `dykstra_iterations` 30 to 60 | `984.9 s` | 500 / 192, `1.280 s` | 22,568 / 1,432, 60 | `-5.860e-4` | `-1.423e-2` | `0.7224` | `0.033253` | `771 MiB` |
| `conic1e10` | re-selection completion's `conic_tolerance` `1e-9` to `1e-10` (preparation at `1e-9`) | `831.0 s` | 500 / 267, `1.204 s` | 22,583 / 1,417, 30 | `-1.566e-3` | `-3.585e-2` | `0.8927` | `0.038320` | `803 MiB` |

The energy is held on each run's own reference to `1.2e-14` or better at
every constrained repair; the reference column is why the internal-energy
column below moves: the drive-power reference is advanced along each
trajectory by the power the drive delivers, so a different trajectory
selects a different reference, by up to `1.1e-2` here.

Cumulative scores from `t=0`, by rescoring the saved artifacts
(`score_sensitivity.py`; the recorded column reproduces task 4's rows in
every printed digit):

| `T` | recorded | `etol1e3` | `dykstra60` | `conic1e10` | spread of the four, all-coordinate scalar RMS |
|---:|---|---|---|---|---:|
| 4 | `0.023744` / `0.020406` | `0.024240` / `0.020400` | `0.026837` / `0.025140` | `0.027953` / `0.028639` | 18 percent |
| 8 | `0.045571` / `0.024055` | `0.044530` / `0.024145` | `0.040677` / `0.029477` | `0.040450` / `0.033827` | 13 percent |
| 12 | `0.125139` / `0.025154` | `0.130946` / `0.025270` | `0.114863` / `0.030787` | `0.108195` / `0.035388` | 21 percent |
| 20 | `0.328674` / `0.026000` | `0.325970` / `0.026136` | `0.260123` / `0.031797` | `0.300340` / `0.036590` | 26 percent |

(each cell: all-coordinate scalar RMS / Eq. (22) internal-energy sample
RMS over `t=0` to `T`; the spread is the largest minus the smallest of the
four scores, as a fraction of the smallest.)

Where each rerun leaves the recorded run:

| run | first nonzero separation, 31 coordinates | first nonzero difference of the carried frontier | first crossing of `1e-3` / `1e-2` / `1e-1` | separation at `t=4` / `8` / `12` / `20` | maximum, and when |
|---|---|---|---|---|---|
| `etol1e3` | `5.0e-16` at `t=0.0075` | `2.5e-32` at `t=0.0075`; `0.401` at `t=0.04` | `0.1475` / `0.7525` / `7.01` | `0.0246` / `0.1627` / `0.3738` / `0.0805` | `0.4080` at `t=12.16` |
| `dykstra60` | `2.7e-15` at `t=0.0525` | `1.7e-15` at `t=0.0525`; `0.376` at `t=0.08` | `0.145` / `0.36` / `4.0025` | `0.1000` / `0.2365` / `0.5692` / `0.6955` | `0.9868` at `t=12.85` |
| `conic1e10` | `9.1e-7` at `t=0.0425` | `0.333` at `t=0.04` | `0.125` / `0.48` / `3.085` | `0.1271` / `0.1680` / `0.3177` / `0.3516` | `0.7219` at `t=13.09` |

Three readings.

- **The perturbation each setting makes is at round-off, or nothing, until
  a re-selection.**  The tolerance flag first differs from the recorded run
  by `5e-16` at `t=0.0075` (a skipped repair against an applied one that
  moved the state by round-off); the doubled Dykstra cap first differs by
  `2.7e-15` at `t=0.0525` (a repair that ran past 30 iterations moves the
  point by the residual it had at 30); the tightened re-selection tolerance
  differs by nothing at all until its first solve refuses at `t=0.04`.
  Then in each case the separation is `1e-3` within `0.125` to `0.1475`
  time units, `1e-2` within `0.36` to `0.7525`, and `1e-1` between `t=3.1`
  and `7.0`, and reaches `0.4` to `1.0` near `t=12` to `13`.
- **The recorded run is the worst of the four at every horizon past `t=4`
  and the best at `t=4`.**  Its three twins score `0.0405` to `0.0445`
  against `0.0456` over `t=0` to `8` and `0.260` to `0.326` against `0.329`
  over `t=0` to `20`; the doubled Dykstra cap also refuses least (192 of 500
  against 276) and holds both cones best (`-5.9e-4` and `-1.4e-2` against
  `-2.3e-3` and `-5.3e-2`).  One sample each, and the point of this section
  is that one sample resolves one digit, so this is a direction and not a
  number: with the Dykstra cap at 60 the `t=0` to `20` score reads `0.26`,
  the `N=16` post-pulse row's `0.24` (task 4) is then inside 9 percent of
  it, and the ordering of the 15-key `t=20` rows in task 5's table
  (`0.239`, `0.260`, `0.284`, `0.318`, `0.329`) is within one setting's
  spread.
- **The tighter tolerance buys nothing it was asked for.**  At `1e-10` the
  re-selection refuses 267 of 500 against 276, at `1.25` times the mean
  solve, and its first act is to refuse a lift the declared solve
  certifies; the certificate at `1e-9` is an `AlmostSolved` at `1e-10`.

## (b) The exact reference under the record's physical perturbations

The record's helper `exact_reference_sensitivity` (the matched
sensitivity study's two perturbations of the exact ground state, `drive`
generated by the site-imbalance operator the pulse couples to and
`relative_phonon_position` by the relative phonon displacement) was run at
its smallest recorded wavefunction amplitude, `5e-6`, and at `5e-4`, the
amplitude at which its 31-coordinate distance at `t=0` matches the size
the route's separation reaches at its first crossing (`1e-3`), on the
recorded run's own 8,001 sample times.  Its base trajectory agrees with the
exact reference stored in the run artifacts to `1.1e-11`; the state
fidelity of every shadow is constant to `1.5e-14`, so the flow is the
unitary one.

| direction | amplitude | 31-coordinate distance at `t=0` | at `t=4` / `8` / `12` / `20` | maximum, and when | maximum over initial | first crossing of `1e-3` / `1e-2` / `1e-1` |
|---|---:|---:|---|---|---:|---|
| `drive` | `5e-6` | `1.185e-5` | `8.53e-6` / `9.83e-6` / `1.18e-5` / `4.49e-6` | `1.185e-5` at `t=0` | `1.000` | never / never / never |
| `relative_phonon_position` | `5e-6` | `1.281e-5` | `1.19e-5` / `1.14e-5` / `1.49e-5` / `5.25e-6` | `1.489e-5` at `t=13.995` | `1.163` | never / never / never |
| `drive` | `5e-4` | `1.185e-3` | --- | `1.185e-3` at `t=0` | `1.000` | `0` / never / never |
| `relative_phonon_position` | `5e-4` | `1.281e-3` | --- | `1.488e-3` | `1.162` | `0` / never / never |

The two amplitudes give the same amplification to three digits, which is
the flow's linearity in the wavefunction.  So a physical perturbation of
the size the route's separation has at its first crossing, `1e-3` at
`t=0.125` to `0.1475`, is carried by the exact dynamics to at most
`1.5e-3` anywhere in `t=0` to `20`, while the route carries it to `1e-2`
by `t=0.36` to `0.75` and to `1e-1` by `t=3.1` to `7.0`.  Two decades in
three to seven time units is the closure's, and none of it is the physics'.

## The amplifier, located

`separation_jumps.py` and `refusal_flip.py` read the saved frontiers and
states of the four runs.  At a re-selection step the frontier a run
carries out of the step is the solved lift when the declared completion
certifies and the carried, repaired lift when it refuses; the two are told
apart by the frontier's own displacement over the step, at most `1.0e-3`
for a repair on the recorded run and up to `0.334` for a certified
re-selection.

| rerun | first step at which the two runs' frontiers differ by more than `1e-6` | the recorded run's frontier displacement over that step | the rerun's | distance between the two frontiers after it | 44-coordinate state distance one step later | 31-coordinate separation one step later |
|---|---|---:|---:|---:|---:|---:|
| `etol1e3` | step 16, `t=0.04`, a re-selection step | `0.3335` (certified) | `0.3021` (certified) | `0.4013` | `1.173e-3` | `6.6e-7` |
| `dykstra60` | step 32, `t=0.08`, a re-selection step (step 16 certified in both: `0.3335` and `0.3335`, frontiers then equal) | `1.6e-11` (refused, the carried lift) | `0.3763` (certified) | `0.3763` | `2.339e-3` | `1.6e-6` |
| `conic1e10` | step 16, `t=0.04`, a re-selection step | `0.3335` (certified) | `1.3e-15` (refused, the carried lift) | `0.3335` | `1.420e-3` | `9.1e-7` |

Two facts about the declared completion follow, both measured on inputs
that agree to at most `1.2e-14` in the 44 propagated coordinates and
`1e-15` in the warm frontier (`dykstra60`, `etol1e3`) or exactly
(`conic1e10`):

- **its certificate is a discontinuous function of the input and of the
  tolerance**: at step 32 the recorded run's solve refused and the
  `dykstra60` run's certified; at step 16 the `1e-9` solve certified and
  the `1e-10` solve on identical inputs refused.  Task 5's section (d)
  named this the candidate amplifier; it is one of two.
- **its certified solution is not a continuous function of the input
  either**: at step 16 the recorded run and the `etol1e3` run both
  certified, and the two certified lifts are `0.40` apart in the
  standardized frontier, that is 1.2 times the largest displacement any
  single re-selection makes on the recorded run.  The completion is
  deterministic for identical inputs (task 4's bitwise reading over 100
  re-selections), so this is a branch of the solve, not noise.

After the one jump, the 31-coordinate separation grows by at most `4.0` in
any step and by at most `1.45` at any re-selection step, and no later
one-step growth reaches a factor of ten in any of the three reruns: the
later refusal flips (the refusal counts differ by 9 to 84 of 500) do not
show as decades, because the frontiers are already `0.3` to `0.4` apart.
The two decades from `1e-3` to `1e-1` in `3` to `7` time units are the
unguarded velocity's own sensitivity to a lift that differs by `0.3`
to `0.4`, which task 1 measured as 37 to 101 percent of the entrance-moment
velocity.

## (c) Which comparisons are meaningful at the second digit

For this configuration, none, at any horizon.  Rounded to two significant
digits the four `t=0` to `4` scores read `0.024`, `0.024`, `0.027`, `0.028`;
`t=0` to `8`: `0.046`, `0.045`, `0.041`, `0.040`; `t=0` to `12`: `0.13`,
`0.13`, `0.11`, `0.11`; `t=0` to `20`: `0.33`, `0.33`, `0.26`, `0.30`;
the internal energy `0.020` to `0.029` at `t=4` and `0.026` to `0.037` at
`t=20`.  What (b) adds is that none of this spread is the physics': the
exact dynamics' own amplification is `1.16`, so every digit past the
first, past the first re-selection at `t=0.04`, belongs to the branch the
completion took and not to the closure's accuracy.

A comparison between this row and another route's is resolved at the
first digit where the other row lies outside the one-setting spread,
that is by more than about 1.3 times.  On the record:

| horizon | resolved (ratio to this row across its four samples) | not resolved |
|---|---|---|
| `t=0` to `4` | the 27-key `N=16` drive-power chain `0.011427` (2.1 to 2.4 times better); the `N=4` drive-power chain C `0.0476697` (1.7 to 2.0 times worse); the raw archive EOM `0.11090` (4 to 5 times worse) | the projective event guard `0.026049` (0.91 to 1.10 times: two twins are worse than it, two better); the `N=4` post-pulse chain B `0.0262899`; the task-3 `N=16` row `0.026842` |
| `t=0` to `8` | --- | the task-3 `N=16` `0.057730` (1.27 to 1.43 times); the `N=16` post-pulse `0.042542`; the `N=4` rows `0.046722` to `0.049575` |
| `t=0` to `20` | both cones `0.128607` (2.0 to 2.6 times better); the 27-key chain `0.085193` (3.1 to 3.9 times better); the task-3 `N=16` `0.579332` (1.8 to 2.2 times worse); the plain archive EOM `0.193205` (1.35 to 1.70 times better, at the edge) | the `N=16` post-pulse row `0.239172` (1.09 to 1.37 times); the unconstrained `N=4` `0.259573`; the `N=4` chains `0.318110` and `0.283805` |
| internal energy, `t=0` to `20` | both cones `0.000939628` (28 to 39 times better); the `N=16` post-pulse `0.069384` (1.9 to 2.7 times worse) | the plain archive EOM `0.0182361` (1.4 to 2.0 times better, at the edge) |

Two consequences for the record.  The sentence in task 4 that the
drive-power `N=16` row over `t=0` to `4` "is the best row this route has
produced, `0.023744` at internal energy `0.020406`, against the record's
projective event guard's `0.026049` and `0.036201`" stands only on the
internal energy (1.3 to 1.8 times better across the four samples), not on
the score.  And every ordering among the 15-key `t=20` rows in task 5's
table (`0.239` to `0.329`) is inside one setting's spread and should be
read as "all `0.2` to `0.3`", not as a ranking.  Whether the 27-key chain's
lead over both cones (`0.085193` against `0.128607`, 1.5 times) survives
its own spread was not measured here; that chain and the both-cones run
are other agents' rows and their sensitivity is not this section's.

## Provenance and gates

- `paper_5/tests/test_cheap_projection_route.py`: 35 passed in `21.01 s`
  (`cd paper_5 && PYTHONPATH=src python -m pytest -q
  tests/test_cheap_projection_route.py`), the 34 of task 5 plus one: that
  `--completion-conic-tolerance` is `None` by default, that
  `_extension_settings(None)` is the declared `APCMExtensionSettings` and a
  value replaces the conic tolerance and nothing else, that the value
  reaches the re-selection completion while the model's extension keeps
  the declared settings, that a refusal keeps the carried lift and a
  certified solve at `1e-8` moves it, and that a re-selection extension on
  a different dictionary is refused.  Full `paper_5` suite: 480 passed in `264.86 s` (`cd paper_5 && PYTHONPATH=src python -m pytest -q --ignore=tests/test_paper5_import_without_parent_repo.py`, under `ram_guard`, peak `2,042 MB`), the 479 the tree collects plus the one new test.  The ignored module is an untracked file another agent left in this worktree (`git status`: `??`) that imports `locate_repo_root` from `paper5.repo`, a name that module has neither here nor on `main`; it fails at collection on any tree and is not this route's.
- Modules changed, both owned by this route and neither imported by any
  other: `cheap_projection_route.py` gained the `reselection_extension`
  argument of `CheapProjectionRoute` (validated against the model's
  dictionary, lower keys and frontier keys; the ladder is built on it) and
  the `reselection_extension_settings` argument of
  `prepare_cheap_projection_route`; `cheap_projection_route_analysis.py`
  gained `--completion-conic-tolerance`, `_extension_settings`, and the
  `reselection_extension_settings` summary block.  Every default is
  unchanged: with the flag absent the route solves `model.extension`
  itself, as before.  Nothing is committed.  `HEAD` is `a3146764` (task
  5's commit, made by another agent); the recorded run predates every
  change here and was not re-run.
- Scripts, all under
  `output/local_runs/paper_v_cheap_projection_task6_sensitivity_20260916_v1/`,
  each reading only `trajectory.npz` and `summary.json` of the runs named
  on its command line and propagating nothing but, in one case, the exact
  reference: `run_chain.zsh` (the three reruns, one at a time),
  `score_sensitivity.py` (`score_sensitivity.json`: the horizon scores,
  counters and separation crossings), `separation_jumps.py`
  (`separation_jumps.json`: decade crossings, re-selection-step
  attribution, one-step growth), `refusal_flip.py` (`refusal_flip.json`:
  frontier displacements at the first differing re-selection),
  `exact_sensitivity.py` (`exact_sensitivity.json` at `5e-6`,
  `exact_sensitivity_amplitude5e-4.json` at `5e-4`, with the arrays in the
  matching `.npz`; this one propagates the exact reference and its
  shadows, `4.7 s`), and the run logs and `chain.log`.
- Artifacts, under `output/local_runs/`:
  `paper_v_cheap_projection_energy_power_n16_t20_{etol1e3,dykstra60,conic1e10}_20260916_v1`.
  Every rerun under `pipelines/shell/ram_guard.py --limit-mb 3000`, one at
  a time, `1,140 s` budget, longest `984.9 s`; peaks are each run's own
  `peak_resident_bytes` (`ru_maxrss` of the propagating process).  Wall
  times were measured with a reviewer's test process on the machine and
  are upper bounds.

## What to try next, in order

1. Record the stages (`--record-stages`) on this configuration once, so the
   certified / refused branch of every one of the 500 re-selections is on
   disk and the count of flips between two runs is a number rather than an
   inference from frontier displacements.
2. The completion's branch, not its tolerance, is what a second digit
   needs: a re-selection rule that cannot flip on a `1e-14` input change
   (a declared tie-break between the carried lift and the solved one, or a
   solve warm-started from a fixed reference lift rather than the carried
   frontier) is the one experiment that could make two runs of one
   configuration agree past the first re-selection.  The completion is
   not this route's module; the rule around it is.
3. The same three perturbations on the 27-key `N=16` drive-power chain and
   on the both-cones run, which are the two `t=20` rows whose ordering the
   record currently reads at the second digit.

## Structured summary of task 6

| item | value |
|---|---|
| what was added | `--completion-conic-tolerance` in the driver and `reselection_extension` / `reselection_extension_settings` on the route, a second extension on the same dictionary for the re-selection solves alone; one test (35 in the file); no default changed; nothing committed |
| negative result, the tolerance | the preparation's completion is not certified at `1e-10` (Clarabel `AlmostSolved`), so the tighter decade applies to re-selections only, and its first act at `t=0.04` is to refuse a lift the `1e-9` solve certifies on identical inputs |
| (a) three reruns, each one setting away from the recorded `N=16` drive-power run | `energy_tolerance` `1e-3`: `0.024240` / `0.044530` / `0.130946` / `0.325970` over `t=0` to `4` / `8` / `12` / `20`, internal energy `0.020400` to `0.026136`, `702.4 s`; Dykstra 60: `0.026837` / `0.040677` / `0.114863` / `0.260123`, energy `0.025140` to `0.031797`, `984.9 s`; re-selection `conic_tolerance` `1e-10`: `0.027953` / `0.040450` / `0.108195` / `0.300340`, energy `0.028639` to `0.036590`, `831.0 s`; recorded `0.023744` / `0.045571` / `0.125139` / `0.328674`, energy `0.020406` to `0.026000` |
| spread across the four samples | 18, 13, 21 and 26 percent of the score at `T=4`, `8`, `12`, `20`; 40 percent of the internal energy at `T=4` and `20` |
| separation times, 31-coordinate Euclidean distance from the recorded run | first nonzero `5.0e-16` at `t=0.0075`, `2.7e-15` at `t=0.0525`, `9.1e-7` at `t=0.0425`; `1e-3` at `t=0.1475`, `0.145`, `0.125`; `1e-2` at `0.7525`, `0.36`, `0.48`; `1e-1` at `7.01`, `4.0025`, `3.085`; maxima `0.41`, `0.99`, `0.72` near `t=12` to `13` |
| (b) the exact reference under the record's two physical perturbations | amplitude `5e-6`: 31-coordinate distance `1.19e-5` and `1.28e-5` at `t=0`, maximum over initial `1.000` and `1.163`, never reaches `1e-3`; amplitude `5e-4` (the route's first-crossing size): `1.19e-3` and `1.28e-3` at `t=0`, maximum `1.49e-3`, never reaches `1e-2` |
| the amplifier | the lift re-selection's completion: a refusal flip on inputs equal to `1e-14` (`dykstra60` at step 32, recorded refused / rerun certified; `conic1e10` at step 16, `1e-9` certified / `1e-10` refused), and two certified lifts `0.40` apart on inputs equal to `1e-14` (`etol1e3` at step 16); the state follows one step later at `1.2e-3` to `2.3e-3`; every later one-step growth is at most `4.0` |
| (c) meaningful at the second digit | no horizon, for this configuration; resolved at the first digit against rows more than about 1.3 times away: at `t=4` the 27-key chain, chain C and the raw archive EOM, not the projective event guard; at `t=20` both cones, the 27-key chain, the task-3 `N=16` row and (at the edge) the plain archive EOM, not the `N=16` post-pulse row or any `N=4` row; the task-4 "best row" sentence stands on the internal energy only |
| gates | 35 tests in the file; full suite 480 passed in `264.86 s` with one untracked foreign test module (`test_paper5_import_without_parent_repo.py`, collection `ImportError`, not this route's) ignored |
| files changed | `cheap_projection_route.py`, `cheap_projection_route_analysis.py`, `test_cheap_projection_route.py`, this report, the worklog block, and the task-6 scripts under `output/local_runs/`.  Nothing committed |

# Task 7: a continuous lift update, and whether it resolves the second digit (2026-09-16)

Same branch, worktree and modules as the six sections above.  The two
modules this route owns gained one declared object, the lift re-selection
rule (`LiftReselectionRule` on `CheapProjectionRoute` and
`prepare_cheap_projection_route`; driver flags
`--reselection-anchor-weight`, `--reselection-trust-radius`,
`--reselection-trust-policy`, `--reselection-acceptance-margin`,
`--reselection-candidates`), every field of which is off by default, and
the stage record of a re-selection now carries the completion's
certificate, the identified face, the KKT backward error, the declared
objective at the carried lift and at the completion's answer, and the
answer's move separately from the move the route applied.  No file this
route does not own was edited; no default changed; nothing is committed.
Conventions are those of tasks 2 to 6: the score is the all-coordinate
scalar RMS of the 31 closed scalar coordinates against the exact cutoff-16
driven reference on the stated horizon; the internal energy is the sample
RMS of the time-independent energy of Eq. (22) on that horizon;
physicality numbers are measured minimum eigenvalues and there is no
certificate anywhere on this route; the configuration is the recorded
`paper_v_cheap_projection_energy_power_n16_t20_20260915_v1` (task 4:
strong coupling, cutoff 16, exact-correlated preparation, 15 entrance
keys, SSPRK(3,3) at `h = 0.0025`, the unguarded fixed-dictionary
predictor, Dykstra 30 on both targets at stage cadence, the declared
proximal log-det completion re-solved every 16 steps warm-started on the
carried frontier, the energy-neutral option in `drive_power` mode with
`energy_tolerance` 0, 8,000 steps to `t=20`), and its three one-setting
twins are task 6's (`energy_tolerance` `1e-3`; Dykstra 60; re-selection
`conic_tolerance` `1e-10`).  The separation between two rollouts is the
Euclidean distance between their 31 closed scalar coordinates at the same
accepted step; the spread of a set of scores is the largest minus the
smallest as a fraction of the smallest.  A lift is a point of the 62
standardized frontier coordinates (each frontier moment divided by its
declared scale, so the frontier box is `[-1, 1]` per coordinate); the
carried lift is the one the route propagates between re-selections,
repaired by Dykstra; the answer is what the completion returns at a
re-selection, whatever the rule then does with it.

## (a) How the declared completion is posed, and what its 500 re-selections did

Read from `apcm_positive_extension.py` (not edited).  The declared
proximal log-det completion, `SymmetryReducedPositiveExtension.complete`,
solves, over the 62 frontier moments `f` with the 29 lower moments of the
state fixed,

    minimize    0.5 |z|^2  -  (w / n) log det ( S(lower, f) + s I )
    subject to  S(lower, f) >= 0   (the scaled 24-row positive extension),
                |z_k| <= 1         (the frontier box),
    where       z = f / frontier_scales,  n = 24,
                w = logdet_weight = 1,  s = logdet_shift = 1e-10.

Its proximal term `0.5 |z|^2` is proximal to the *zero* standardized
frontier -- the completion is the minimum-norm lift regularized by a
log-determinant barrier of weight `1/24` per eigenvalue -- and not to any
carried lift.  The warm frontier the route hands it is the starting point
of the solve and enters the objective nowhere, so the API exposes no
proximal centre, and (b) acts on the answer, outside the solve.  The
objective the result reports is `0.5 |z|^2 - (w/n) sum log(max(lambda_i,
0) + s)`, finite at an infeasible point; the barrier contributes
`-log(1e-10)/24 = +0.96` for an eigenvalue at zero and `+0.48` for one at
`1e-5`, so on this route, where the Dykstra repair (floor 0) leaves the
carried lift's scaled minimum eigenvalue near zero, the objective
difference between two lifts is set by where their smallest eigenvalue
sits and not by the `0.5 |z|^2` distance between them.

Certification (`clarabel_newton`, the declared backend): a result is
certified when the solver's own stationarity test passed, the answer's
scaled minimum eigenvalue is at or above `-10 tol` and the KKT backward
error is at most `50 tol`, `tol = conic_tolerance = 1e-9`.  The solve
takes one of two paths.  When the warm frontier is *strictly* feasible
(scaled minimum eigenvalue above `100 tol = 1e-7`, box margin above
`1e-7`) the Newton iteration starts from it on the full cone (support 24,
all 62 directions).  Otherwise a facial reduction first solves a Clarabel
reference problem at the same lower moments, identifies the face of the
cone the feasible set lies in, and restricts the Newton iteration to that
face's directions (support 23 of 24 leaves 24 of the 62 directions).  Then
at most 40 Newton steps, each a line search keeping the iterate at or
above `-0.1 tol`, with a direct conic QP when the interior step is not
accepted, and a final certificate.  A refusal is one of: the reference
problem infeasible or the reduction not stabilizing
(`moment_extension_degeneracy`), a conic Newton step that fails descent, a
Newton cone failure, or a final certificate above `50 tol`.  On a refusal
the result still carries the last iterate as its frontier.

The recorded configuration was rerun once with `--record-stages`
(`paper_v_cheap_projection_energy_power_n16_t20_recorded_stages_20260916_v1`,
`671.2 s` against the recorded `719.6 s`, peak `874 MiB`); its 8,001
states and carried frontiers equal the recorded run's in every entry and
its score is the recorded `0.328674`.  The 500 re-selections, every 16
steps over `t=0` to `20` (`reselection_stats_recorded_rule.json`):

| quantity, over the 500 re-selections of `t=0` to `20` | value |
|---|---|
| certified / refused | 224 / 276; every certified answer replaced the carried lift (the earlier form) |
| certified per quarter, `t` in `[0,5)`, `[5,10)`, `[10,15)`, `[15,20)` | 112 of 124, 98 of 125, 14 of 125, 0 of 125; the last certified re-selection is at `t=11.76` and all 206 after it refuse |
| refusals by message | 227 reference completion `PrimalInfeasible` and 7 `AlmostPrimalInfeasible` (the state's 29 lower moments admit no positive 24-row extension at all); 23 conic Newton descent failures; 19 final certificates above `50 tol` (KKT `5.3e-8` to `1.6e-5`) |
| face of the certified answers | 223 on the full face (support 24, 62 directions); one on the 23-of-24 face (`t=0.04`) |
| carried lift's scaled minimum eigenvalue at the re-selection | `-2.33e-3` to `+2.69e-3`, median `-2.0e-4`; strictly feasible (above `1e-7`, the warm-start path) at 79, of which 69 certify; not at 421, of which 155 certify |
| move of the certified answer from the carried lift (Euclidean, standardized frontier) | minimum `0.0026`, 10th percentile `0.0077`, median `0.0122`, 90th `0.0367`, maximum `0.3340`; 160 above `0.01`, 5 above `0.1` (`t=0.04` `0.3335` on the 23/24 face, `t=0.16` `0.3340`, `t=11.08` `0.1142`, `t=11.48` `0.1092`, `t=11.76` `0.2007`); median per quarter `0.0095`, `0.017`, `0.046` |
| declared objective at the answer against the carried lift, certified | improvement minimum `-0.460` (`t=0.04`, the 23/24 answer, the only certified answer worse than the carried lift), 10th percentile `0.0067`, median `0.653`, 90th `1.275`, maximum `1.426` |
| refused answers that pass the completion's own feasibility test and improve the objective | 42 (the 23 descent failures and the 19 certificate failures); moves median `0.025`, maximum `0.505`; improvement median `0.83` |
| certified KKT backward error | median `1.6e-9`, maximum `4.9e-8` |
| solve wall | mean `0.904 s`, median `0.76 s`, maximum `22.7 s` |

Three readings.

- **The recorded rule's update is the full-face minimizer where it
  certifies, and the minimizer moves little**: 223 of 224 certified
  answers are on the full face, the median move is `0.012` and the 90th
  percentile `0.037`, so between re-selections the carried lift drifts by
  a few hundredths from the completion's own answer.  The one certified
  answer on the 23-of-24 face, at `t=0.04`, is the run's largest move
  (`0.334`) and the only one whose declared objective is above the
  carried lift's: it is not the minimizer the other 223 are, and it is
  the answer task 6 found `0.40` away from the `etol1e3` twin's on inputs
  equal to `1e-14`.
- **Past `t=11.76` the recorded row is the frozen-lift route with the
  Dykstra repair.**  Nothing certifies after that, and 234 of the 276
  refusals are the reference problem reporting the state's lower moments
  primal-infeasible: the propagated state has left the set of moment
  sequences that admit any positive 24-row extension (its scaled
  extension minimum reaches `-2.3e-3` and the retained Gram `-5.3e-2`),
  and no rule on the answer can act where there is no answer.  The task-4
  `t=20` score of `0.329` is therefore a hybrid to `t=11.76` continued by
  the task-2 frozen-lift route (which scored `1.42` at `t=20` on its own).
- **42 refusals hold a usable answer**: their last iterate passes the
  completion's own feasibility test of a warm frontier (scaled minimum
  eigenvalue at or above `-1e-8`, inside the box) and improves the
  declared objective, by `0.83` at the median; 12 of them move the lift by
  more than `0.1`.  These are the refusal flips of task 6 (the `dykstra60`
  twin's step 32 certified where the recorded run's refused).

## (b) The named options, all off by default

The completion exposes no proximal centre, so every option acts on the
answer and the carried lift in the standardized frontier, in
`CheapProjectionRoute.reselect_lift_decision`; `reselect_lift` keeps its
signature and returns what the rule selects.  With every field off the
selected lift is the certified answer exactly (the record-stages run above
is bitwise the recorded run).

- **Proximal anchor at the carried lift**, `anchor_weight` `mu >= 0`
  (`--reselection-anchor-weight`): the carried lift moves the fraction
  `1/(1+mu)` of the way to the accepted answer.  That point is the
  minimizer of `0.5 |z - z*|^2 + (mu/2) |z - z_c|^2`, the anchored
  objective under the identity-Hessian model of the declared objective
  (whose Hessian is the identity plus the log-determinant term's), and it
  lies on the segment between two lifts, hence inside the positive cone
  and the box whenever both ends are.  `0` is off.
- **Trust radius on the move**, `trust_radius` `r`
  (`--reselection-trust-radius`), with `trust_policy`
  (`--reselection-trust-policy`) `carried` (the cap binds: keep the
  carried lift, the rule named in the task) or `ball` (the cap binds:
  take the point at distance `r` on the segment toward the answer).
  `None` is off.
- **Deterministic acceptance margin**, `acceptance_margin` `m >= 0`
  (`--reselection-acceptance-margin`): a candidate replaces the carried
  lift only when the declared objective at it lies below the declared
  objective at the carried lift by more than `m`, both values by one
  formula (`CheapProjectionRoute.declared_objective`, the completion's own
  `0.5 |z|^2 - (w/n) sum log(max(lambda, 0) + s)`; a test checks it equals
  the completion's reported objective).  A flip needs the improvement to
  cross `m`, and the improvement is a continuous function of the inputs
  with the barrier's Lipschitz constant, so an input change of `1e-14`
  cannot flip it unless the improvement sits within `1e-14` times that
  constant of `m`.  `None` is off.
- **Candidate scope**, `candidates` (`--reselection-candidates`):
  `certified` (the earlier form: only a certified answer is a candidate)
  or `measured`: any returned iterate is a candidate when it passes the
  completion's own feasibility test of a warm frontier (scaled minimum
  eigenvalue at or above `-10 tol`, inside the box to `10 tol`) and
  improves the declared objective by more than `max(m, 0)`; the
  certificate is recorded and not consulted.  This is the option aimed at
  the refusal flip: the rule then depends on measured, continuous
  quantities of the iterate and not on the solver's status.

Each rule was screened on the recorded configuration and its three
one-setting twins over `t=0` to `2` (`screens_all.zsh`, 20 rollouts of 800
steps, `55 s` to `68 s` each, one at a time; `screen_<rule>_t2.json`); the
recorded rule's own four `t=2` rows are the task-4 and task-6 artifacts
rescored (`baseline_recorded_rule.json`).  Every screened rule carries the
margin `1e-3`; the anchor, ball and cap rows add one option to it.

| rule (margin `1e-3` throughout) | scores over `t=0` to `2`, base / `etol1e3` / `dykstra60` / `conic1e10` | spread at `t=2` (`t=1`) | internal energy at `t=2`, smallest to largest | largest frontier difference between a twin and the base in one step | first crossing of `1e-3` / `1e-2` by any twin | accepted of 50 per run (of which uncertified) |
|---|---|---|---|---:|---|---|
| recorded rule (every field off) | `0.007852` / `0.008091` / `0.009999` / `0.010136` | 29.1 % (125.6 %) | `0.0066` to `0.0123` | `0.334` to `0.401` at `t=0.04` and `0.08` | `0.125` / `0.36` | 47, 43, 50, 45 (0) |
| margin only, certified candidates | `0.011272` / `0.010279` / `0.010327` / `0.010136` | 11.2 % (3.5 %) | `0.0123` to `0.0160` | `0.0064` to `0.0089` at `t=0.16` and `0.6` | `0.6375` / `1.1` | 40, 36, 39, 35 (0) |
| margin and trust radius `0.05`, `ball` | identical to the margin row: no accepted move exceeds `0.031`, the cap never binds | 11.2 % | --- | --- | --- | --- |
| margin and trust radius `0.05`, `carried` | identical to the margin row, for the same reason | 11.2 % | --- | --- | --- | --- |
| margin and anchor `mu = 1` | `0.008973` / `0.009117` / `0.009286` / `0.008883` | 4.5 % (9.3 %) | `0.0042` to `0.0082` | `0.0032` to `0.0088` | `0.52` / `0.9825` | 44, 40, 46, 36 (0) |
| margin and `measured` candidates | `0.011160` / `0.011215` / `0.011217` / `0.011160` | 0.5 % (0.1 %) | `0.0163` to `0.0164` | `0.0064` at `t=0.16` (`etol1e3`, `dykstra60`); none above `1e-6` for `conic1e10` | never / never | 46, 46, 46, 46 (4, 3, 4, 11) |
| margin, `measured` candidates and anchor `mu = 1` | `0.009265` / `0.009306` / `0.009286` / `0.009265` | 0.4 % (0.6 %) | `0.0082` to `0.0083` | `0.0032` at `t=0.16`; none above `1e-6` for `conic1e10` | `1.7875` (`etol1e3` only) / never | 46, 47, 46, 46 (1, 3, 0, 12) |

(each score is the all-coordinate scalar RMS over `t=0` to `2`; the
separation columns are the 31-coordinate Euclidean distance from the base
run of the same rule; `conic1e10` under the two `measured` rules is
identical to its base in every one of the 801 samples' 31 coordinates,
its certificate flipping 11 to 12 times without effect on the lift.)

Four readings.

- **Declining the wrong-face answer is what the margin does, and it costs
  early accuracy.**  Under the margin the `t=0.04` answer on the 23-of-24
  face (declared objective `0.46` above the carried lift's) is declined
  in every run, as are the 23/24 and 23/23 answers at `t=0.08` to `0.16`,
  and the first accepted lift is a full-face answer `0.006` to `0.009`
  from the preparation's lift at `t=0.16` to `0.20`.  That is the branch
  the recorded `conic1e10` twin took (its `t=2` score `0.010136` is
  reproduced to every digit), and it scores `0.0101` to `0.0113` at
  `t=2` where the recorded run's excursion to the wrong face and back
  (`0.334` out at `t=0.04`, `0.334` back at `t=0.16`) scored `0.0079`.
  The remaining 11 percent spread is the refusal flip: with only
  certified answers as candidates, one run's refusal is another's
  accepted `0.006` move.
- **The `measured` scope removes the refusal flip.**  Judged by the
  completion's own feasibility test and the declared objective rather
  than by the certificate, the four runs accept the same 46 answers and
  the base and the `conic1e10` twin (which differs only in the certificate
  tolerance) are identical to the last digit; the other two twins, whose
  Dykstra repair differs at round-off or at a skipped stage, separate by
  `8e-4` to `9e-4` at `t=2`, never reaching `1e-3`.  The uncertified
  answers accepted move the lift by `0.004` to `0.011`, the same as their
  certified neighbours.
- **The anchor buys accuracy back.**  Half-way moves score `0.0089` to
  `0.0093` against the margin's `0.0101` to `0.0113` and the internal
  energy `0.004` to `0.008` against `0.012` to `0.016`; with the
  `measured` scope the combination scores `0.0093` at spread 0.4 percent
  with internal energy `0.0082`.  Why the half-way point beats the answer
  was not measured here; one reading is that the answer is the minimum
  norm lift for the state's lower moments at the re-selection step while
  the carried lift holds information from the trajectory, and the mean of
  the two is nearer the exact frontier than either.
- **The trust radius is inert once the margin is on**: at `0.05` no
  accepted move reaches it over `t=0` to `2`.  It would have capped the
  recorded rule's `0.334` moves, but the margin declines those first.

The rule taken to `t=20` in (c) is the last row: margin `1e-3`,
`measured` candidates, anchor `mu = 1`.

## (c) The recorded configuration and its three twins to `t=20` under the chosen rule

Four rollouts of 8,000 steps
(`paper_v_cheap_projection_task7_measured_anchor1_{base,etol1e3,dykstra60,conic1e10}_t20_20260916_v1`),
each differing from the base in exactly the one setting task 6's twin
changed, all under the rule margin `1e-3`, `measured` candidates, anchor
`mu = 1`, with `--record-stages`; one at a time under `ram_guard
--limit-mb 3000` inside the recorded `1,140 s` budget (walls `843.7`,
`822.6`, `998.7`, `908.5 s`; peaks `794` to `869 MiB`); scores by
`screen_scores.py` (`screen_measured_anchor1_t20.json`).

| `T` | base | `etol1e3` | `dykstra60` | `conic1e10` | spread of the four (recorded rule's four, task 6) | recorded row (task 4), and the recorded rule's four |
|---:|---|---|---|---|---|---|
| 4 | `0.031566` / `0.02369` | `0.031582` / `0.02393` | `0.031594` / `0.02369` | `0.031566` / `0.02369` | 0.1 % (17.7 %) | `0.023744` / `0.020406`; `0.0237` to `0.0280` |
| 8 | `0.075551` / `0.02913` | `0.073518` / `0.02940` | `0.076622` / `0.02913` | `0.076060` / `0.02913` | 4.2 % (12.7 %) | `0.045571` / `0.024055`; `0.0405` to `0.0456` |
| 12 | `0.143618` / `0.03073` | `0.120502` / `0.03101` | `0.130712` / `0.03073` | `0.127896` / `0.03073` | 19.2 % (21.0 %) | `0.125139` / `0.025154`; `0.108` to `0.131` |
| 20 | `0.264357` / `0.03195` | `0.256831` / `0.03225` | `0.273564` / `0.03195` | `0.254707` / `0.03195` | 7.4 % (26.4 %) | `0.328674` / `0.026000`; `0.260` to `0.329` |

(each cell: all-coordinate scalar RMS / Eq. (22) internal-energy sample
RMS over `t=0` to `T`.  The internal energy is the same to five digits in
three of the four runs because the repair holds it on the drive-power
reference at every stage and the three references agree to `2e-10`; the
`etol1e3` twin's reference differs by `3e-4`.)

Where each twin leaves the base (31-coordinate Euclidean separation at the
same accepted step; the recorded rule's values from task 6 in brackets):

| twin | first nonzero | first carried-frontier difference above `1e-6` | first crossing of `1e-3` / `1e-2` / `1e-1` | separation at `t=4` / `8` / `12` / `20` | maximum, and when |
|---|---|---|---|---|---|
| `etol1e3` | `t=0.0075` | `0.0032` at `t=0.16` [`0.401` at `0.04`] | `1.7875` / `4.1` / `8.415` [`0.1475` / `0.7525` / `7.01`] | `0.0078` / `0.094` / `0.57` / `1.39` [`0.025` / `0.16` / `0.37` / `0.08`] | `1.68` at `t=15.3` [`0.41` at `12.2`] |
| `dykstra60` | `t=0.0625` | `0.0032` at `t=0.16` [`0.376` at `0.08`] | `2.305` / `5.65` / `8.6025` [`0.145` / `0.36` / `4.0025`] | `0.0027` / `0.067` / `0.43` / `1.65` [`0.10` / `0.24` / `0.57` / `0.70`] | `1.74` at `t=18.6` [`0.99` at `12.9`] |
| `conic1e10` | `t=0.505` | `0.1417` at `t=6.44` [`0.334` at `0.04`] | `6.4525` / `6.8` / `8.545` [`0.125` / `0.48` / `3.085`] | `3.2e-9` / `0.069` / `0.49` / `1.37` [`0.13` / `0.17` / `0.32` / `0.35`] | `1.37` at `t=20` [`0.72` at `13.1`] |

Lift re-selections per quarter of the horizon (accepted of 125; in
brackets the recorded rule's certified-and-accepted):

| `t` in | base | `etol1e3` | `dykstra60` | `conic1e10` |
|---|---|---|---|---|
| `[0, 5)` | 120, 3 uncertified, 4 declined [112] | 121, 3, 3 | 120, 2, 4 | 120, 25, 4 |
| `[5, 10)` | 55, 6 uncertified [98] | 66, 8 | 61, 10 | 64, 20 |
| `[10, 15)` | 78, 6 uncertified [14] | 119, 18 | 121, 15 | 121, 23 |
| `[15, 20)` | 94, 15 uncertified [0] | 79, 11 | 83, 14, 1 declined | 90, 32, 1 declined |

(every re-selection not accepted or declined is an iterate that fails
the measured feasibility test, decision `infeasible`: 149, 112, 110, 100
of 500, none in `t=0` to `5`; under the `measured` scope no re-selection
is a `refused` decision, the certificate being recorded and not
consulted.  Of the `infeasible` iterates 19, 13, 10, 9 per run are
reference-completion `PrimalInfeasible` refusals whose last iterate is
out of the frontier box by `1e22` to `1e113`; they set the summaries'
`maximum_reselection_solved_frontier_displacement_euclidean` field,
`4.2e113`, `2.5e113`, `1.4e111`, `5.9e112`, which is therefore a
record of diverged refused iterates and not of any move; the largest
answer the rule accepted stood `0.30`, `0.71`, `0.79`, `0.56` from the
carried lift and the anchor applied half, `0.15`, `0.35`, `0.39`,
`0.28`, the summaries' `maximum_reselection_frontier_displacement_euclidean`.)

Both cones over `t=0` to `20`: scaled 24-row extension minimum `-3.5e-4`,
`-5.5e-4`, `-3.2e-4`, `-5.8e-4` (recorded rule `-2.3e-3` to `-5.9e-4`);
unshifted retained joint Gram minimum `-9.1e-3`, `-1.3e-2`, `-7.6e-3`,
`-1.4e-2` (recorded rule `-5.3e-2` to `-1.4e-2`); the standardized
frontier envelope reaches `0.63`, `0.84`, `0.97`, `0.95` (recorded rule
`0.72` to `0.92`).

**Is the second digit resolved, and at what cost.**

- Over `t=0` to `4`, yes: the four scores are `0.03157` to `0.03159`,
  agreeing at the third significant digit (spread 0.1 percent against
  the recorded rule's 17.7 percent), and the `conic1e10` twin is within
  `3.2e-9` of the base in the 31 coordinates at every one of the 1,601
  samples (first nonzero `3.2e-15` at `t=0.505`).  The cost is 33
  percent: `0.0316` against the recorded `0.0237`, above every one of the
  recorded rule's four (`0.0237` to `0.0280`) and 21 percent above the
  projective event guard's `0.026049`; internal energy `0.0237` against
  `0.0204`.
- Over `t=0` to `8`, the first digit and the second to within three
  units: `0.0735`, `0.0756`, `0.0761`, `0.0766` (4.2 percent; the recorded
  rule 12.7 percent).  The cost is 66 percent: `0.0756` against the
  recorded `0.0456`, 1.6 to 1.9 times the recorded rule's four (`0.0405`
  to `0.0456`).  This is the horizon on which the rule pays most: in
  `t=5` to `10` it accepts 55 to 66 lifts of 125 where the recorded rule
  certified 98, and the separation of every twin crosses `1e-1` at
  `t=8.4` to `8.6`.
- Over `t=0` to `12`, no: `0.12` to `0.14` (19.2 percent, the recorded
  rule 21.0 percent); the score is 15 percent above the recorded row and
  inside the recorded rule's own spread (`0.108` to `0.131`).
- Over `t=0` to `20`, no, and the 7.4 percent spread of the cumulative
  RMS (`0.255` to `0.274`, the recorded rule's 26.4 percent) is not the
  trajectories agreeing: by `t=20` each twin is `1.37` to `1.65` from the
  base in the 31 coordinates (the recorded rule's twins `0.08` to `0.70`),
  the base's site-0 occupation RMS is `0.34` against the recorded row's
  `0.19`, and what the four runs share is the frozen-lift decay the
  recorded rule reached at `t=11.76` being replaced by re-selection to
  the end (94 accepted lifts in `t=15` to `20` against none) while the
  state itself has wandered.  The score is 20 percent below the recorded
  row (`0.264` against `0.329`) and inside 5 percent of the recorded
  rule's best twin (`0.260`); the internal energy is 23 percent worse
  (`0.0320` against `0.0260`).

What moved the branch.  Under the recorded rule the twins parted at the
first or second re-selection (`t=0.04`, `0.08`) by `0.33` to `0.40` in the
lift; under the chosen rule the `etol1e3` and `dykstra60` twins part at
`t=0.16` by `0.0032` (two full-face answers `0.0064` apart after repairs
that differ at round-off, halved by the anchor) and the `conic1e10` twin
not until `t=6.44`, by `0.14` (the `1e-9` and `1e-10` solves converge to
feasible, improving answers `0.28` apart, halved by the anchor -- an
answer branch, not a certificate flip, and the one event of its kind in
that run's first 6.4 time units).  The `1e-2` crossing moved from
`t=0.36` to `0.75` to `t=4.1` to `6.8`, and the `1e-1` crossing from
`t=3.1` to `7.0` to `t=8.4` to `8.6`.  Past `t=8.5` the closure's own
amplification (task 6 (b): at most `1.16` for the exact flow) carries
every difference to order one by `t=15` to `20` under either rule.

What the cost is made of was not separated here: the three ingredients of
the chosen rule were screened together at `t=2` only, where the margin
alone cost 30 to 40 percent against the recorded rule's base
(`0.0101`–`0.0113` against `0.0079`) and the anchor recovered part of it
(`0.0093`); no rule with the `measured` scope and no anchor was run past
`t=2`, and no rule was run at `t=20` other than the chosen one.

## Provenance and gates

- `paper_5/tests/test_cheap_projection_route.py`: 44 passed in
  `22.73 s` (`cd paper_5 && PYTHONPATH=src python -m pytest -q
  tests/test_cheap_projection_route.py`), the 35 of task 6 plus nine: that
  `LiftReselectionRule()` is the earlier form, that each field is validated,
  and which fields make the rule evaluate the objective; that
  `declared_objective` at the completion's answer equals the objective the
  completion reports, and the decision's solved displacement is the
  Euclidean move of that answer; that the default rule returns the
  certified answer exactly and the carried lift on a refusal, without
  evaluating the objective; that anchor `mu = 3` moves the carried lift one
  quarter of the way to the answer; that a binding trust radius keeps the
  carried lift under `carried`, moves it exactly the radius under `ball`,
  is inert when wider than the move, and composes with the anchor; that a
  margin below the measured improvement accepts and one above it declines,
  by one formula, the declined decision carrying both objectives; that
  under `measured` an uncertified iterate is accepted, declined or found
  infeasible by the measured test and the objective alone; that a rollout
  counts accepted, uncertified, declined and cap-bound re-selections and
  its stage record carries the decision, the certificate, the face, the
  KKT backward error and the objectives; and that the five driver flags at
  their defaults give the earlier form and otherwise reach the route's
  rule.  Full `paper_5` suite: 499 passed in `257.04 s`
  (`cd paper_5 && PYTHONPATH=src python -m pytest -q`, under `ram_guard
  --limit-mb 5000`, peak `2,033 MB`): the 480 tests of task 6, the
  nine here, and the 10 of `tests/test_paper5_import_without_parent_repo.py`,
  the module task 6 ignored as untracked and failing at collection, tracked
  since `1023ef39` and collecting now.  The previous instance's pass (`paper5_suite.log`, `ram_guard --limit-mb 3000`, peak `2,092 MB`, `256.46 s`, 489 tests with that module ignored) had one failure outside this route, `tests/test_multi_coherent.py::test_frozen_model_trajectory_runs_without_a_reference_or_gate`, whose `exact-cost-once` subprocess of `multi_coherent_holdout_cli` exited 1 with its stderr piped and unshown; that module was last changed in `6442fbb5`, the test passed in task 6's gate and passes here, and the failure was not investigated further.
- Modules changed, both owned by this route and neither imported by any
  other: `cheap_projection_route.py` gained `LiftReselectionRule`,
  `LiftReselectionDecision`, `CheapProjectionRoute.declared_objective`,
  `CheapProjectionRoute.reselect_lift_decision` (the solve and retry
  ladder of `reselect_lift`, then the rule), the `reselection_rule`
  argument of `CheapProjectionRoute` and `prepare_cheap_projection_route`,
  five fields on `CheapProjectionTrajectory` (`lift_reselections_accepted`,
  `lift_reselections_accepted_uncertified`,
  `lift_reselections_declined_by_margin`, `lift_reselections_cap_bound`,
  `maximum_lift_reselection_solved_displacement`) and the enlarged stage
  record; `reselect_lift` keeps its signature and returns the decision's
  frontier.  `cheap_projection_route_analysis.py` gained the five flags,
  `_reselection_rule`, and the `reselection_rule` block and five counters
  in the summary's `lift` section.  Every default is unchanged: with every
  flag absent the rule is `LiftReselectionRule()`, and the record-stages
  rerun of the recorded configuration under it reproduces the recorded
  run's 8,001 times, states, carried frontiers, closed coordinates,
  eigenvalue minima and internal energies with maximum absolute difference
  `0.0` in every array, and its score `0.328674`.  Nothing is committed.
  `HEAD` is `bb576659` (another agent's commit); the recorded run and
  task 6's three twins predate every change here and were not re-run.
- Scripts, all under
  `output/local_runs/paper_v_cheap_projection_task7_continuity_20260916_v1/`:
  `run_one.zsh` (one rollout of the recorded configuration under
  `ram_guard --limit-mb 3000`, every twin or rule stated as flags on its
  command line and nowhere else), `screen.zsh` (one rule on the base and
  its three twins, one at a time, then the scores), `screens_all.zsh` (the
  five `t=2` screens of (b)), `screen_scores.py` (`screen_<rule>_t2.json`,
  `screen_measured_anchor1_t20.json`, `base_only_measured_anchor1_t20.json`:
  horizon scores, internal energies, spreads, lift counters, and the
  31-coordinate separations with their crossings; it reads only
  `trajectory.npz` and `summary.json` and propagates nothing),
  `reselection_stats.py` (`reselection_stats_recorded_rule.json`,
  `reselection_stats_measured_anchor1_base_t20.json`: every re-selection of
  a record-stages run, from `repairs.jsonl`), `baseline_recorded_rule.json`
  (the task-4 and task-6 artifacts rescored at `t=1` and `2`), the run
  logs, `chain.log`, and the two suite logs (`paper5_suite.log`, the
  previous instance's pass under `--limit-mb 3000`;
  `paper5_suite_limit5000.log`, this one).  A `t=20` stage record is
  15 MB.
- Artifacts, under `output/local_runs/`:
  `paper_v_cheap_projection_energy_power_n16_t20_recorded_stages_20260916_v1`
  (`671.2 s`, peak `874 MiB`);
  `paper_v_cheap_projection_task7_{margin,measured,anchor1,ball005,cap005,measured_anchor1}_{base,etol1e3,dykstra60,conic1e10}_t2_20260916_v1`
  (24 rollouts of 800 steps, at most `79 s` each by `chain.log`);
  `paper_v_cheap_projection_task7_measured_anchor1_{base,etol1e3,dykstra60,conic1e10}_t20_20260916_v1`
  (walls `843.7`, `822.6`, `998.7`, `908.5 s`; peaks `802`, `869`, `866`,
  `794 MiB`); and `paper_v_cheap_projection_task7_smoke_t05_20260916_v1`.
  Every rollout under `pipelines/shell/ram_guard.py --limit-mb 3000`, one
  at a time, `1,140 s` budget, longest `998.7 s`; peaks are each run's own
  `peak_resident_bytes` (`ru_maxrss` of the propagating process).  The
  instance of this agent that ran the rollouts was lost to a connection
  error after the four `t=20` rollouts and its pass of the suite had
  completed; the instance that finished this section propagated nothing,
  re-ran only the suite and the file's tests, and checked every number of
  (c) against the artifacts.

## What to try next, in order

1. Separate the cost.  The three ingredients were screened together at
   `t=2` only.  Two four-twin sets to `t=8`, the horizon on which the rule
   pays most (66 percent): `measured` candidates with margin `1e-3` and no
   anchor, and anchor `mu = 1` with certified candidates and no margin, so
   the 33 percent at `t=4` and the 66 percent at `t=8` are attributed to
   the margin (which declines the wrong-face answer and pays early), the
   anchor (which recovered accuracy at `t=2`) or the `measured` scope.
   Eight rollouts of 3,200 steps, about six minutes each.
2. The anchor weight.  `mu = 0.5` and `mu = 2` on the base alone to `t=8`:
   the half-way point beat the whole answer at `t=2`, and the reading
   offered for it (the mean of the minimum-norm lift and the carried lift
   is nearer the exact frontier than either) predicts an optimum between
   `0` and infinity that one more decade would locate.
3. The chosen rule on the 27-key `N=16` drive-power chain, the row whose
   `t=4` lead over cell A sits inside a 13 percent sensitivity band
   (`review-27key-chain-20260916.md`); the chain runs through this driver,
   so the rule is five flags away.  If it resolves that row's `t=4` score
   at the third digit as it does here, the ordering is settled either way.

## Structured summary of task 7

| item | value |
|---|---|
| what was added | `LiftReselectionRule` (proximal anchor at the carried lift, trust radius with `carried` / `ball` policy, deterministic acceptance margin on the completion's own objective, `certified` / `measured` candidate scope) on `CheapProjectionRoute` and `prepare_cheap_projection_route`, `reselect_lift_decision` and `declared_objective`, five driver flags, five counters and the enlarged stage record; nine tests (44 in the file); no default changed (the record-stages rerun equals the recorded run in every array); nothing committed |
| negative result, the recorded rule | of its 500 re-selections 224 certify and 276 refuse, none after `t=11.76` (234 of the refusals are the state's lower moments admitting no positive extension), so the recorded `t=20` row is a hybrid to `t=11.76` continued by the frozen-lift route; the declared completion is proximal to the zero frontier, not the carried lift; 223 of 224 certified answers are full-face minimizers a median `0.012` from the carried lift, and the `t=0.04` answer on the 23-of-24 face (move `0.334`) is the one certified answer worse than the carried lift by the declared objective |
| (b) the screens over `t=0` to `2`, four twins each | recorded rule spread 29.1 percent; margin `1e-3` alone 11.2 percent at `0.0101` to `0.0113` against the recorded base's `0.0079` (it declines the wrong-face answer, and that costs early accuracy); trust radius `0.05` inert; anchor `mu = 1` 4.5 percent at `0.0089` to `0.0093`; `measured` candidates 0.5 percent (the refusal flip removed; `conic1e10` identical to its base); `measured` with anchor `mu = 1` 0.4 percent at `0.0093`, the rule taken to `t=20` |
| (c) the four scores under the chosen rule, base / `etol1e3` / `dykstra60` / `conic1e10` | over `t=0` to `4`: `0.031566` / `0.031582` / `0.031594` / `0.031566`; to `8`: `0.075551` / `0.073518` / `0.076622` / `0.076060`; to `12`: `0.143618` / `0.120502` / `0.130712` / `0.127896`; to `20`: `0.264357` / `0.256831` / `0.273564` / `0.254707`; internal energy `0.02369` to `0.02393`, `0.02913` to `0.02940`, `0.03073` to `0.03101`, `0.03195` to `0.03225` on the four horizons; over `t=0` to `20` the scaled extension minima `-3.5e-4`, `-5.5e-4`, `-3.2e-4`, `-5.8e-4` and the retained joint Gram minima `-9.1e-3`, `-1.3e-2`, `-7.6e-3`, `-1.4e-2` (recorded rule `-2.3e-3` to `-5.9e-4`, `-5.3e-2` to `-1.4e-2`) |
| spread of the four (the recorded rule's, task 6) | 0.1 (17.7), 4.2 (12.7), 19.2 (21.0), 7.4 (26.4) percent at `T=4`, `8`, `12`, `20` |
| separation from the base, first crossing of `1e-3` / `1e-2` / `1e-1` (the recorded rule's) | `etol1e3` `1.7875` / `4.1` / `8.415` (`0.1475` / `0.7525` / `7.01`); `dykstra60` `2.305` / `5.65` / `8.6025` (`0.145` / `0.36` / `4.0025`); `conic1e10` `6.4525` / `6.8` / `8.545` (`0.125` / `0.48` / `3.085`) |
| is the second digit resolved, and at what cost | `t=4`: yes, at the third digit, for 33 percent (`0.0316` against the recorded `0.0237`; 21 percent above the projective event guard's `0.026049`); `t=8`: to within three units of the second digit, for 66 percent (`0.0756` against `0.0456`); `t=12`: no (19.2 percent), and 15 percent above the recorded row; `t=20`: no (7.4 percent of the cumulative RMS while every twin is `1.37` to `1.65` from the base in the 31 coordinates), 20 percent below the recorded row's `0.329` with the internal energy 23 percent worse |
| what moved the branch | the twins now part at `t=0.16` by `0.0032` (two full-face answers after repairs differing at round-off, halved by the anchor) or at `t=6.44` by `0.14` (an answer branch of the `1e-9` and `1e-10` solves), not at `t=0.04` to `0.08` by `0.33` to `0.40`; past `t=8.5` the closure carries every difference to order one by `t=15` to `20` under either rule |
| re-selections accepted of 500 (uncertified; declined; infeasible) | 347 (30; 4; 149), 385 (40; 3; 112), 385 (41; 5; 110), 395 (100; 5; 100); 94, 79, 83, 90 accepted in `t=15` to `20` where the recorded rule certified none |
| gates | 44 tests in the file; full suite 499 passed in `257.04 s` under `ram_guard --limit-mb 5000`, peak `2,033 MB` |
| files changed | `cheap_projection_route.py`, `cheap_projection_route_analysis.py`, `test_cheap_projection_route.py`, this report, the worklog block, and the task-7 scripts under `output/local_runs/`.  Nothing committed |

# Task 8: the attribution -- the measured acceptance buys the continuity, the anchor pays the accuracy (2026-09-16)

Eleven rollouts of the recorded `N=16` drive-power configuration to `t=8`
(3,200 steps), one at a time under `ram_guard --limit-mb 3000`, each
differing from the recorded run only in the twin's one setting and the
rule's flags on its command line (`chain.zsh`; walls `244.7` to `384.7 s`
against the `1,140 s` budget, peaks `410` to `486 MiB`).  No
module was edited: every rule here is a setting of task 7's
`LiftReselectionRule` (`anchor_weight` is any nonnegative float and the
two scopes are the existing ones), so there is no new option and no new
test; the 44 tests of `test_cheap_projection_route.py` pass (`24.82 s`).
Nothing is committed.  Conventions and the configuration are task 7's: the
score is the all-coordinate scalar RMS of the 31 closed scalar coordinates
against the exact cutoff-16 driven reference over `t=0` to `T`; the internal
energy is the sample RMS of the Eq. (22) energy on that horizon; the
physicality numbers are the minimum over `t <= T` of the scaled 24-row
positive-extension eigenvalue and of the unshifted retained joint Gram
eigenvalue (recomputed per sample from the states by the driver's formula,
so the `t=0` to `20` rows of tasks 6 and 7 are restricted to the same
horizons); a separation is the 31-coordinate Euclidean distance between two
rollouts at the same accepted step; the spread of a set is the largest score
minus the smallest as a fraction of the smallest; a re-selection's move is
the Euclidean distance in the 62 standardized frontier coordinates.  The
sets:

| set | rule | runs |
|---|---|---|
| recorded | every field off (task 4's row and task 6's three twins, `t=0` to `20`, restricted) | base, `etol1e3`, `dykstra60`, `conic1e10` |
| combined | margin `1e-3`, `measured` candidates, anchor `mu = 1` (task 7's chosen rule, its four `t=0` to `20` runs restricted) | the four |
| A, measured | margin `1e-3`, `measured` candidates, no anchor: the measured-candidate acceptance without the anchor | the four, `t=8` |
| B, anchor | anchor `mu = 1`, `certified` candidates, no margin: the anchor without the measured acceptance (the earlier form's acceptance, the move halved) | the four, `t=8` |
| C | the combined rule with anchor `mu = 0.1` and `mu = 0.01` | base only, `t=8` |
| D | margin `1e-3`, anchor `mu = 1`, `certified` candidates (the `t=2` screen's "margin and anchor" row, taken to `t=8`) | base only, `t=8` |

Set A's four runs, set D's and the combined base reproduce the task-7 `t=2`
screen runs with the same flags in every one of the 801 common samples' 31
coordinates and 62 frontier coordinates (largest absolute difference `0.0`,
`task8_extras.json`): the driver is deterministic across processes and the
`t=2` numbers of task 7 (b) are this section's `t=2` numbers.

## Negative result first

The anchor pays the accuracy, and task 7's reading that it "buys accuracy
back" holds only to `t=2`.  Set A (the measured acceptance alone) scores
`0.027191` over `t=0` to `4` and `0.038538` over `t=0` to `8`, 15 percent
above the recorded `0.023744` at `t=4` and 15 percent *below* the recorded
`0.045571` at `t=8`, with the four twins spread 0.1 and 0.2 percent; set B
(the anchor alone) scores `0.030650` and `0.071711`, 29 and 57 percent above
the recorded row, with the twins spread 31.0 and 14.7 percent -- no
continuity at all.  The combined rule's 33 and 66 percent are therefore the
anchor's (at `t=4` both ingredients pay and the anchor the larger part; at
`t=8` the anchor is the whole cost and the measured acceptance a gain).  Every anchored rule collapses its
re-selection in `t=6` to `8`: 1 to 17 accepted of 50 (set B), 10 (combined),
16 (D), against 50 of 50 under set A and 40 under the recorded rule, and
task 7's "cones an order of magnitude better" was a `t=0` to `20` statement
against a recorded row that is frozen-lift past `t=11.76`: over `t=0` to
`8` the recorded base holds the tightest cones of every set (scaled
extension minimum `-8.6e-5`, joint Gram `-2.1e-3`), set A is within a
factor 1.6 of it, and the anchored rules are 2 to 4 times looser.

## The rows side by side

Scores (all-coordinate scalar RMS) and internal-energy RMS over `t=0` to
`4` / `t=0` to `8`, base / `etol1e3` / `dykstra60` / `conic1e10`:

| set | score `T=4` | score `T=8` | spread `T=4` / `8` | internal energy `T=4` | `T=8` |
|---|---|---|---|---|---|
| recorded | `0.023744` / `0.024240` / `0.026837` / `0.027954` | `0.045571` / `0.044530` / `0.040677` / `0.040450` | 17.7 % / 12.7 % | `0.02041` / `0.02040` / `0.02514` / `0.02864` | `0.02406` / `0.02415` / `0.02948` / `0.03383` |
| combined | `0.031566` / `0.031582` / `0.031594` / `0.031566` | `0.075551` / `0.073518` / `0.076622` / `0.076060` | 0.1 % / 4.2 % | `0.02369` / `0.02393` / `0.02369` / `0.02369` | `0.02913` / `0.02940` / `0.02913` / `0.02913` |
| A, measured | `0.027191` / `0.027215` / `0.027214` / `0.027191` | `0.038538` / `0.038617` / `0.038605` / `0.038538` | 0.1 % / 0.2 % | `0.03314` / `0.03332` / `0.03332` / `0.03314` | `0.03816` / `0.03835` / `0.03836` / `0.03816` |
| B, anchor | `0.030650` / `0.027274` / `0.031232` / `0.035728` | `0.071711` / `0.062545` / `0.067015` / `0.070631` | 31.0 % / 14.7 % | `0.01760` / `0.01668` / `0.01322` / `0.01621` | `0.02206` / `0.02098` / `0.01705` / `0.02071` |
| C, anchor `0.1` (base) | `0.027426` | `0.040099` | --- | `0.03270` | `0.03784` |
| C, anchor `0.01` (base) | `0.027221` | `0.038647` | --- | `0.03311` | `0.03814` |
| D, margin and anchor (base) | `0.032761` | `0.064289` | --- | `0.01981` | `0.02489` |

Physicality over `t <= 4` / `t <= 8`, same order of runs (scaled 24-row
extension minimum; unshifted retained joint Gram minimum), and the
re-selections over `t=0` to `8` (accepted, of which uncertified; declined by
the margin; refused or `infeasible`) with the accepted count in `t=6` to `8`
of 50:

| set | extension minimum, `t <= 4` / `<= 8` | joint Gram minimum, `t <= 4` / `<= 8` | re-selections of 200 over `t=0` to `8` | accepted in `[6, 8)` |
|---|---|---|---|---|
| recorded | `-7.5e-6` / `-8.6e-5`; `-4.6e-6` / `-1.7e-4`; `-1.1e-6` / `-3.4e-5`; `-5.5e-6` / `-1.6e-4` | `-2.4e-4` / `-2.1e-3`; `-1.5e-4` / `-4.7e-3`; `-4.2e-5` / `-8.0e-4`; `-1.8e-4` / `-4.8e-3` | base 173 (0); 0; 27 (the twins have no stage record) | 40 |
| combined | `-1.0e-5` / `-2.9e-4`; `-1.6e-5` / `-3.6e-4`; `-3.9e-6` / `-1.7e-4`; `-1.0e-5` / `-3.5e-4` | `-6.7e-4` / `-6.8e-3`; `-7.7e-4` / `-8.7e-3`; `-2.1e-4` / `-4.1e-3`; `-6.7e-4` / `-8.6e-3` | 151 (9); 4; 45 -- 150 (6); 3; 47 -- 147 (4); 4; 49 -- 148 (30); 4; 48 | 10 |
| A, measured | `-5.0e-6` / `-1.1e-4`; `-4.9e-6` / `-1.1e-4`; `-1.0e-6` / `-4.8e-5`; `-5.0e-6` / `-1.1e-4` | `-1.9e-4` / `-3.3e-3`; `-1.8e-4` / `-3.2e-3`; `-4.0e-5` / `-1.4e-3`; `-1.9e-4` / `-3.3e-3` | 196 (13); 4; 0 -- 196 (17); 4; 0 -- 196 (13); 4; 0 -- 196 (44); 4; 0 | 50, 50, 50, 50 |
| B, anchor | `-1.5e-5` / `-2.1e-4`; `-1.7e-5` / `-3.6e-4`; `-4.6e-6` / `-1.8e-4`; `-2.8e-5` / `-3.0e-4` | `-4.1e-4` / `-5.3e-3`; `-4.4e-4` / `-8.3e-3`; `-1.3e-4` / `-4.4e-3`; `-7.3e-4` / `-6.9e-3` | 139 (0); 0; 61 -- 157; 0; 43 -- 139; 0; 61 -- 112; 0; 88 | 7, 17, 1, 4 |
| C, anchor `0.1` / `0.01` | `-4.5e-6` / `-1.6e-4`; `-4.9e-6` / `-1.2e-4` | `-1.8e-4` / `-4.3e-3`; `-1.9e-4` / `-3.6e-3` | 196 (22); 4; 0 -- 196 (19); 4; 0 | 50, 50 |
| D | `-1.4e-5` / `-3.1e-4` | `-4.2e-4` / `-7.5e-3` | 149 (0); 1; 50 | 16 |

Where each twin leaves its set's base (first nonzero separation; first
carried-frontier difference above `1e-6`; first crossing of `1e-3` / `1e-2`
/ `1e-1`; separation at `t=4` / `8`; maximum over `t <= 8`):

| set | twin | first nonzero | first frontier difference | crossings | at `t=4` / `8` | maximum, when |
|---|---|---|---|---|---|---|
| recorded | `etol1e3` | `0.0075` | `0.401` at `t=0.04` | `0.1475` / `0.7525` / `7.01` | `0.025` / `0.163` | `0.198` at `7.465` |
| recorded | `dykstra60` | `0.0525` | `0.376` at `0.08` | `0.145` / `0.36` / `4.0025` | `0.100` / `0.236` | `0.283` at `7.52` |
| recorded | `conic1e10` | `0.0425` | `0.333` at `0.04` | `0.125` / `0.48` / `3.085` | `0.127` / `0.168` | `0.263` at `6.8875` |
| combined | `etol1e3` | `0.0075` | `0.0032` at `0.16` | `1.7875` / `4.1` / never by `t=8` | `0.0078` / `0.094` | `0.097` at `7.895` |
| combined | `dykstra60` | `0.0625` | `0.0032` at `0.16` | `2.305` / `5.65` / never | `0.0027` / `0.067` | `0.071` at `7.8925` |
| combined | `conic1e10` | `0.505` | `0.142` at `6.44` | `6.4525` / `6.8` / never | `3.2e-9` / `0.069` | `0.072` at `7.895` |
| A, measured | `etol1e3` | `0.0075` | `0.0064` at `0.16` | `2.1075` / `5.83` / never | `0.0035` / `0.0175` | `0.0194` at `7.9275` |
| A, measured | `dykstra60` | `0.0625` | `0.0064` at `0.16` | `2.105` / `5.76` / never | `0.0035` / `0.0232` | `0.0236` at `7.9875` |
| A, measured | `conic1e10` | `0.245` (`8.7e-19`) | never | never / never / never | `5.1e-9` / `1.7e-8` | `1.9e-8` at `6.49` |
| B, anchor | `etol1e3` | `0.0075` | `0.201` at `0.04` | `0.215` / `0.85` / `4.4275` | `0.079` / `0.259` | `0.297` at `5.9175` |
| B, anchor | `dykstra60` | `0.06` | `0.244` at `0.08` | `0.285` / `0.745` / `5.0475` | `0.056` / `0.231` | `0.250` at `7.7575` |
| B, anchor | `conic1e10` | `0.0425` | `0.167` at `0.04` | `0.1725` / `0.7125` / `2.985` | `0.114` / `0.476` | `0.480` at `7.94` |

## (a) The measured acceptance without the anchor

The four runs accept the same 196 re-selections of 200 (the four declined
by the margin are the wrong-face answers of `t=0.04` to `0.16`, as at
`t=2`), no iterate fails the measured feasibility test before `t=8`, the
first accepted lift is a full-face answer `0.0064` to `0.0094` from the
preparation's at `t=0.16` to `0.20`, and no accepted move exceeds `0.039`
(medians `0.008`, `0.010`, `0.017`, `0.023` in the four two-unit intervals;
the largest at `t=6.2` to `6.88`).  The `conic1e10` twin is identical to
the base to `1.9e-8` in the 31 coordinates at every one of the 3,201
samples (first nonzero `8.7e-19` at `t=0.245`; its 44 uncertified
acceptances against the base's 13 are certificate flips without effect on
the lift); the `etol1e3` and `dykstra60` twins part at `t=0.16` by `0.0064`
in the frontier (two full-face answers after repairs differing at
round-off, now taken whole where task 7 halved them) and stay within `0.024`
of the base to `t=8`, crossing `1e-2` only at `t=5.8`.  The spread is 0.1
percent at `t=4` and 0.2 percent at `t=8` (the combined rule's 0.1 and 4.2,
the recorded rule's 17.7 and 12.7), so the second digit is resolved on both
horizons and the third at `t=4`.  The cost: 15 percent at `t=4` (`0.0272`
against the recorded `0.0237`, inside the recorded rule's own band `0.0237`
to `0.0280` and 4 percent above the projective event guard's `0.026049`)
and a gain of 15 percent at `t=8` (`0.0385` against `0.0456`, below the
recorded rule's band `0.0404` to `0.0456`).  The cones over `t <= 8`
(`-1.1e-4` / `-3.3e-3`) sit between the recorded base's (`-8.6e-5` /
`-2.1e-3`) and the combined rule's (`-2.9e-4` / `-6.8e-3`).  The internal
energy is the worst of every set (`0.0331` / `0.0382` against the recorded
`0.0204` / `0.0241`); what that column measures is below.

## (b) The anchor without the measured acceptance

With certified candidates and no objective test the first re-selection at
`t=0.04` accepts the 23-of-24-face answer and moves half-way, `0.167`
(`0.151` for `etol1e3`), and the twins part there by `0.17` to `0.24` in
the frontier, as under the recorded rule (`0.33` to `0.40`); the
`conic1e10` twin, whose `1e-10` solve refuses that answer (task 6), first
accepts at `t=0.2` (`0.0047`), so its branch is the certificate's.  The
spread is 31.0 percent at `t=4` and 14.7 at `t=8`, wider than the recorded
rule's, and the crossings of `1e-2` (`0.71` to `0.85`) and `1e-1` (`2.99`
to `5.05`) are of the recorded rule's order (`0.36` to `0.75`; `3.1` to
`7.0`).  The accuracy: `0.0307` / `0.0717` on the base (29 and 57 percent
above the recorded row; the twins `0.0273` to `0.0357` and `0.0625` to
`0.0717`), and the re-selection collapses in `t=6` to `8`: 33 to 49 of 50
refused (the base: 4, 2, 11, 43 refusals in the four intervals), so over
most of `t=6` to `8` the anchored run is the frozen-lift route with the
Dykstra repair.  The internal energy is the best
of every set (`0.0132` to `0.0176` / `0.0170` to `0.0221`) and the cones
the loosest with the combined rule's (`-1.8e-4` to `-3.6e-4` / `-4.4e-3`
to `-8.3e-3`).

## (c) Which ingredient buys what, and the weaker anchor

- **The measured acceptance buys the continuity.**  Set A alone gives
  0.1 / 0.2 percent; the anchor alone gives 31.0 / 14.7.  The ingredient
  is the objective test with `measured` scope: it declines the wrong-face
  answer at `t=0.04` to `0.16` (any margin `m >= 0` does, the answer's
  declared objective being `0.46` above the carried lift's) and judges
  every later iterate by measured quantities, so the `conic1e10` twin
  cannot branch on its certificate.  The anchor contributes nothing to the
  continuity at `t=4` and worsens it at `t=8`: with it (combined) the
  twins are `0.067` to `0.094` from the base at `t=8`, without it (set A)
  `0.018` to `0.023`, because the anchored lift lags the answer and the
  runs enter the `infeasible` regime (45 to 49 of 200 by `t=8`) on
  slightly different steps.
- **The anchor pays the accuracy.**  At `t=8` the anchor alone costs 57
  percent, the combined rule 66, the measured acceptance alone gains 15;
  at `t=4` the anchor alone costs 29 percent, the measured acceptance 15,
  the combined rule 33.  The `t=1` and `t=2` scores show where task 7's
  reading came from: at `t=2` the anchored rules lead (`0.0080` B,
  `0.0090` D, `0.0093` combined, against `0.0112` for set A and `0.0079`
  recorded), and the order is reversed by `t=4` (`0.0307`, `0.0328`,
  `0.0316` against `0.0272`).  The reading offered in task 7 (the mean of
  the minimum-norm lift and the carried lift is nearer the exact frontier
  than either) is not supported past `t=2`: the half-way lift makes the
  state leave the set of moment sequences that admit a positive extension
  four units earlier than the whole answer does.
- **A weaker anchor recovers the accuracy in proportion to how little
  anchor is left, and never improves on none.**  On the base, over `t=0`
  to `4` / `8`: anchor `0` `0.027191` / `0.038538`; `0.01` `0.027221` /
  `0.038647` (0.1 / 0.3 percent above); `0.1` `0.027426` / `0.040099`
  (0.9 / 4.1 percent above); `1` `0.031566` / `0.075551` (16 / 96
  percent above).  Monotone in the weight at both horizons, and the
  internal energy moves the other way by at most 1 percent (`0.0378` at
  `0.1` against `0.0382` at `0`).  The twins' spread under a weak anchor
  was not measured (base only, as asked); what was measured is that the
  anchor `0.01` base sits `0.0197` from the no-anchor base at `t=8`, the
  size of one set-A twin's separation (`0.0175` to `0.0232`), while the
  anchor `0.1` base sits `0.209` away (crossing `1e-1` at `t=5.6`), ten
  times a twin, so at `0.1` the anchor is itself a larger perturbation
  than any twin and the two measured endpoints (0.2 percent at weight
  `0`, 4.2 at weight `1`) are the only spreads on record.  The
  recommendation is to drop the anchor: the rule margin `1e-3`, `measured`
  candidates, anchor `0` is the one that resolves the second digit at
  `t=4` and `t=8`, beats the recorded row at `t=8`, and keeps every
  re-selection to `t=8` accepted.
- **Set D locates the remaining piece.**  Margin `1e-3` with the anchor on
  certified candidates scores `0.0328` / `0.0643` (38 and 41 percent
  above the recorded row) with 50 refusals: the margin removes the
  wrong-face branch (first acceptance `0.0047` at `t=0.2`, no move above
  `0.073`) and the anchor still pays, so the cost is not the margin's.

## What the internal-energy column measures on a drive-power run

The exact Eq. (22) energy on the exact reference is `-3.571398` at `t=4`
and changes by `3.1e-6` over `t=4` to `8` (the pulse envelope `exp(-t^2 /
2)` is below `4e-4` there).  Each run's repair holds its internal energy on
its declared drive-power reference from `t=4` on to `1e-14`, so its
internal-energy RMS over `t=4` to `8` equals the reference minus the exact
energy: `0.0272` (recorded), `0.0337` (combined), `0.0426` (set A; `0.0424`
and `0.0426` at anchor `0.1` and `0.01`), `0.0258` (set B base; `0.0245`,
`0.0202`, `0.0244` its twins), `0.0291` (D); the reference is the
drive-power integral along the run's own trajectory over the pulse, and the
`t=0` to `4` RMS is that running integral's error.  The two ingredients
therefore trade the 31-coordinate score against the Eq. (22) energy in
opposite directions: the measured acceptance without the anchor has the best
coordinates at `t=8` and the worst energy (59 percent above the recorded
`0.0241`), the anchor alone the best energy (8 percent below) and the worst
coordinates.  Which of the two the route is scored on is the user's call;
the recorded scoreboard of tasks 2 to 7 is the coordinate score.

## Provenance and gates

- Gates: `paper_5/tests/test_cheap_projection_route.py` 44 passed in
  `24.82 s` (`test_cheap_projection_route.log`); the full `paper_5` suite
  499 passed in `270.42 s` under `ram_guard --limit-mb 5000`, peak
  `1,984 MB` (`paper5_suite.log`).  No module, test or default changed;
  the modules committed at `a81b7ecb` are what ran (`HEAD` moved to
  `e3374f6f`, another agent's commit, during the runs; it touches nothing
  this route imports).
- Scripts, all under
  `output/local_runs/paper_v_cheap_projection_task8_attribution_20260916_v1/`:
  `run_one.zsh` (task 7's, output directory changed), `chain.zsh` (the
  eleven rollouts in order, every twin and rule as flags), `task8_scores.py`
  (`task8_scores.json` and `.md`: the horizon scores, internal energies,
  per-horizon physicality minima with the joint Gram recomputed from the
  states, spreads, separations, decisions from the stage records; it
  reproduces task 7's `t=4` and `t=8` numbers to every digit),
  `task8_extras.py` (`task8_extras.json`: `t=1` and `t=2` scores, the
  bitwise comparison against the task-7 `t=2` screens, decisions per
  two-unit interval, largest moves), `energy_reference_check.py`
  (`energy_reference_check.json`), `weak_anchor_separation.py`
  (`weak_anchor_separation.json`), the run logs, `chain.log`, the smoke
  `paper_v_cheap_projection_task8_smoke_anchor1_t005_20260916_v1`.  The
  recorded set's base is its task-7 record-stages rerun (equal to the
  recorded run in every array), which is what carries the stage record;
  the recorded set's twins carry none.
- Artifacts, under `output/local_runs/`:
  `paper_v_cheap_projection_task8_{measured,anchor1}_{base,etol1e3,dykstra60,conic1e10}_t8_20260916_v1`,
  `paper_v_cheap_projection_task8_measured_anchor0p1_base_t8_20260916_v1`,
  `paper_v_cheap_projection_task8_measured_anchor0p01_base_t8_20260916_v1`,
  `paper_v_cheap_projection_task8_margin_anchor1_base_t8_20260916_v1`, each
  with `repairs.jsonl` (`5.8 MB` at `t=8`); walls `245.4`, `244.7`, `282.9`,
  `258.4 s` (A), `325.9`, `325.2`, `384.7`, `351.7 s` (B), `276.0`, `255.5 s`
  (C), `335.7 s` (D); peaks `410` to `486 MiB` (`ru_maxrss` of the
  propagating process).  Set A's rollouts are the cheapest of any rule run
  on this configuration; why was not measured.

## What to try next, in order

1. The rule margin `1e-3`, `measured` candidates, no anchor to `t=20` on the
   four (four rollouts of 8,000 steps, about 11 minutes each by the `t=8`
   walls): whether the continuity holds past `t=8` without the anchor, and
   where the score stands against the recorded `0.329` and the both-cones
   `0.129` when every re-selection stays accepted.
2. The same rule on the 27-key `N=16` drive-power chain (task 7's item 3,
   now with the anchor dropped): the row whose `t=4` lead over cell A sits
   inside a 13 percent band.
3. If the Eq. (22) energy is to be scored, the drive-power reference's
   error (`0.043` under set A against `0.027` recorded) is the quantity to
   move, and it is set over the pulse, `t=0` to `4`, not by the lift rule
   after it.

## Structured summary of task 8

| item | value |
|---|---|
| what was run | eleven `t=8` rollouts of the recorded `N=16` drive-power configuration, one at a time under `ram_guard --limit-mb 3000`: set A (margin `1e-3`, `measured`, no anchor) on the base and three twins; set B (anchor `1`, certified, no margin) on the four; anchor `0.1` and `0.01` under the combined rule, base only; margin `1e-3` with anchor `1` on certified candidates, base only; no module edited, no new option, 44 tests pass, nothing committed |
| negative result | the anchor pays the accuracy: alone it costs 29 percent at `t=4` and 57 at `t=8` (`0.0307` / `0.0717` against the recorded `0.0237` / `0.0456`), the combined rule 33 and 66; the measured acceptance alone costs 15 at `t=4` and gains 15 at `t=8` (`0.0272` / `0.0385`); task 7's "the anchor buys accuracy back" holds only to `t=2`, and its "cones an order of magnitude better" only over `t=0` to `20` -- over `t <= 8` the recorded base holds the tightest cones and the anchored rules the loosest |
| (a) set A, base / `etol1e3` / `dykstra60` / `conic1e10` | `0.027191` / `0.027215` / `0.027214` / `0.027191` at `t=4`, `0.038538` / `0.038617` / `0.038605` / `0.038538` at `t=8`; spread 0.1 / 0.2 percent; `conic1e10` within `1.9e-8` of the base throughout; twins part at `t=0.16` by `0.0064`, cross `1e-3` at `2.1`, `1e-2` at `5.8`, never `1e-1`; internal energy `0.0331` to `0.0333` / `0.0382` to `0.0384`; extension minimum `-4.8e-5` to `-1.1e-4`, joint Gram `-1.4e-3` to `-3.3e-3` over `t <= 8`; 196 of 200 accepted (4 declined, none infeasible), largest move `0.039` |
| (b) set B, same order | `0.030650` / `0.027274` / `0.031232` / `0.035728` at `t=4`, `0.071711` / `0.062545` / `0.067015` / `0.070631` at `t=8`; spread 31.0 / 14.7 percent; twins part at `t=0.04` to `0.08` by `0.17` to `0.24` (the halved wrong-face answer; `conic1e10` on its certificate), cross `1e-3` at `0.17` to `0.29`, `1e-2` at `0.71` to `0.85`, `1e-1` at `3.0` to `5.0`; internal energy `0.0132` to `0.0176` / `0.0170` to `0.0221`; extension minimum `-1.8e-4` to `-3.6e-4`, joint Gram `-4.4e-3` to `-8.3e-3`; 112 to 157 accepted, 43 to 88 refused, 1 to 17 accepted in `t=6` to `8` |
| (c) attribution | continuity: the measured acceptance (objective test with `measured` scope); accuracy cost: the anchor, whose half-way lift drives the state into the no-positive-extension regime by `t=6` (combined 45 `infeasible` of 200, set A none); the margin is not the cost (set D `0.0328` / `0.0643` with 50 refusals) |
| (c) the weaker anchor, base only | anchor `0.01` `0.027221` / `0.038647`, anchor `0.1` `0.027426` / `0.040099`, against `0` `0.027191` / `0.038538` and `1` `0.031566` / `0.075551`: monotone, none better than none; the `0.01` base is `0.020` from the no-anchor base at `t=8` (one twin's size), the `0.1` base `0.21` (ten twins); the twins' spread under a weak anchor was not measured; recommendation: drop the anchor |
| the internal-energy column | past `t=4` it is the declared drive-power reference minus the exact energy (`0.0272` recorded, `0.0337` combined, `0.0426` set A, `0.0258` set B), set by the drive-power integral along the run's own trajectory over the pulse; the two ingredients trade the coordinate score against the Eq. (22) energy in opposite directions |
| reproducibility | set A's four runs, set D and the combined base equal the task-7 `t=2` screens with the same flags in every coordinate over the 801 common samples |
| gates | 44 tests pass in `24.82 s`; full `paper_5` suite 499 passed in `270.42 s` under `ram_guard --limit-mb 5000`, peak `1,984 MB` |
| files changed | this report, the worklog block, and the task-8 scripts and artifacts under `output/local_runs/`; no module, test or default; nothing committed |

# Task 9: the measured acceptance without the anchor to `t=20` -- the continuity holds to `t=12`, the gain grows to `t=20`, and past `t=12` the large uncertified moves branch the twins (2026-09-16)

Four rollouts of the recorded `N=16` drive-power configuration to `t=20`
(8,000 steps) under task 8's set-A rule, margin `1e-3`, `measured`
candidates, no anchor, on the base and the three one-setting twins, one at
a time under `ram_guard --limit-mb 3000` (`chain.zsh`; walls `900.2`,
`851.5`, `841.1`, `848.7 s` against the `1,140 s` budget, all reaching
`t=20`; peaks `782`, `771`, `774`, `784 MiB`, each run's own
`peak_resident_bytes`).  No module was edited: the rule is a setting of
task 7's `LiftReselectionRule` and every run is `run_one.zsh` (task 7's
script, output directory changed) plus the twin's one setting plus the
rule's three flags on its command line; the 44 tests of
`test_cheap_projection_route.py` pass (`24.03 s`); nothing is committed.
Conventions and the configuration are tasks 7 and 8's: the score is the
all-coordinate scalar RMS of the 31 closed scalar coordinates against the
exact cutoff-16 driven reference over `t=0` to `T`; the internal energy is
the sample RMS of the Eq. (22) energy on that horizon; the physicality
numbers are the minimum over `t <= T` of the scaled 24-row
positive-extension eigenvalue and of the unshifted retained joint Gram
eigenvalue (recomputed per sample from the states by the driver's formula);
a separation is the 31-coordinate Euclidean distance between two rollouts
at the same accepted step; the spread of a set is the largest score minus
the smallest as a fraction of the smallest; a re-selection's move is the
Euclidean distance in the 62 standardized frontier coordinates between the
carried lift and the lift the rule applied (under this rule, the
completion's answer whole); a solver refusal is a re-selection whose
completion returned uncertified, whatever the rule did with the iterate.
The comparison rows are the recorded rule's four runs (task 4's row as its
task-7 record-stages rerun, equal to it in every array, and task 6's three
twins) and the combined rule's four (task 7: margin `1e-3`, `measured`
candidates, anchor `mu = 1`), all `t=0` to `20`, rescored here on the five
horizons by one script (`task9_scores.py`; the `t=4`, `8`, `12`, `20`
numbers of tasks 7 and 8 are reproduced to every digit).  The four runs
here equal task 8's four set-A `t=8` runs in every one of the 3,201
common samples' times, states, 31 closed coordinates and 62 standardized
frontier coordinates (largest absolute difference `0.0` in each array), so
the `t=4` and `t=8` rows below are task 8's set-A rows and the driver is
deterministic across processes and horizons.

## Negative result first

The continuity does not survive to `t=16` and `t=20` in one of the three
twins, and what breaks it is the rule's own acceptance of large
uncertified moves once the state reaches the edge of the set admitting a
positive extension.  The four scores over `t=0` to `20` are `0.150102`
(base), `0.223420` (`etol1e3`), `0.160803` (`dykstra60`), `0.150285`
(`conic1e10`): spread 48.8 percent, the `etol1e3` twin alone; the other
three lie within 7.1 percent.  At `t=16` the spread is 18.8 percent (7.4
without `etol1e3`), at `t=12` 3.2 percent (2.7), at `t=8` 0.2 and at `t=4`
0.1.  The separation of the `etol1e3` twin from the base crosses `1e-1`
at `t=11.23` and reaches `2.27` at `t=20`; the `dykstra60` twin crosses
`1e-1` at `t=12.9` and reaches `0.53`; the `conic1e10` twin never crosses
`1e-1` (`0.089` at most, `t=19.9`).  From `t=10` on every run accepts, as
the rule is written, iterates that move the lift by `0.4` to `0.8` in the
standardized frontier (the base `0.716` at `t=10.04`, an uncertified
conic-Newton descent failure's last iterate that passes the measured
feasibility test and improves the declared objective by `0.175`; then
`0.686` certified at `t=10.2`), the size of the wrong-face jump the
margin was introduced to decline at `t=0.04` (`0.334`), and the frontier
envelope reaches `1.0` (a frontier coordinate at its declared scale) at
`t=10.04` in the base.  Two twins whose repairs differ at round-off reach
`t=10` with states `0.05` to `0.07` apart and their completions there
return different large iterates; that, and not the certificate (the
`conic1e10` twin accepts the base's `0.716` iterate to `0.0022`), is the
branch.

The gain, stated once: at every horizon from `t=8` on, all four runs are
below every run of the recorded rule and of the combined rule, and the
band of the four is disjoint from both other bands.  Over `t=0` to `20`
the base's `0.150` is 54 percent below the recorded row's `0.329`, 42
percent below the recorded rule's best twin (`0.260`), 41 percent below
the combined rule's best (`0.255`), and 1.17 times the record's
both-cones line (`0.128607`, the `5,832 s` conic run); the worst twin,
`0.223`, is still below the best of the other eight (`0.255`).  The cost
is unchanged from task 8: 15 percent at `t=4` (`0.0272` against the
recorded `0.0237`, inside the recorded rule's band), and the Eq. (22)
internal energy, `0.0409` over `t=0` to `20` against the recorded `0.0260`
(57 percent worse; the drive-power reference's error over the pulse,
`0.0426` against `0.0272`, as task 8 measured).

## The rows side by side

Scores (all-coordinate scalar RMS over `t=0` to `T`), base / `etol1e3` /
`dykstra60` / `conic1e10`, and the spread of the four:

| `T` | recorded rule (task 4 row, task 6 twins) | spread | combined rule (task 7) | spread | measured, no anchor (this task) | spread (without `etol1e3`) |
|---:|---|---|---|---|---|---|
| 4 | `0.023744` / `0.024240` / `0.026837` / `0.027954` | 17.7 % | `0.031566` / `0.031582` / `0.031594` / `0.031566` | 0.1 % | `0.027191` / `0.027215` / `0.027214` / `0.027191` | 0.1 % (0.1) |
| 8 | `0.045571` / `0.044530` / `0.040677` / `0.040450` | 12.7 % | `0.075551` / `0.073518` / `0.076622` / `0.076060` | 4.2 % | `0.038538` / `0.038617` / `0.038605` / `0.038538` | 0.2 % (0.2) |
| 12 | `0.125139` / `0.130946` / `0.114863` / `0.108195` | 21.0 % | `0.143618` / `0.120502` / `0.130712` / `0.127896` | 19.2 % | `0.098187` / `0.101367` / `0.100824` / `0.098186` | 3.2 % (2.7) |
| 16 | `0.258499` / `0.256609` / `0.201866` / `0.228471` | 28.1 % | `0.223586` / `0.183926` / `0.196958` / `0.188325` | 21.6 % | `0.136367` / `0.161977` / `0.146397` / `0.136288` | 18.8 % (7.4) |
| 20 | `0.328674` / `0.325970` / `0.260123` / `0.300340` | 26.4 % | `0.264357` / `0.256831` / `0.273564` / `0.254707` | 7.4 % | `0.150102` / `0.223420` / `0.160803` / `0.150285` | 48.8 % (7.1) |

Internal-energy sample RMS (Eq. (22)) over `t=0` to `T`, same order:

| `T` | recorded rule | combined rule | measured, no anchor |
|---:|---|---|---|
| 4 | `0.02041` / `0.02040` / `0.02514` / `0.02864` | `0.02369` / `0.02393` / `0.02369` / `0.02369` | `0.03314` / `0.03332` / `0.03332` / `0.03314` |
| 8 | `0.02406` / `0.02415` / `0.02948` / `0.03383` | `0.02913` / `0.02940` / `0.02913` / `0.02913` | `0.03816` / `0.03835` / `0.03836` / `0.03816` |
| 12 | `0.02515` / `0.02527` / `0.03079` / `0.03539` | `0.03073` / `0.03101` / `0.03073` / `0.03073` | `0.03969` / `0.03989` / `0.03990` / `0.03969` |
| 16 | `0.02569` / `0.02582` / `0.03142` / `0.03614` | `0.03150` / `0.03179` / `0.03150` / `0.03150` | `0.04043` / `0.04064` / `0.04064` / `0.04043` |
| 20 | `0.02600` / `0.02614` / `0.03180` / `0.03659` | `0.03195` / `0.03225` / `0.03196` / `0.03195` | `0.04087` / `0.04108` / `0.04109` / `0.04087` |

(The internal-energy spread of the four is 0.5 percent at every horizon
under this rule; on every window past the pulse the RMS is the run's
drive-power reference minus the exact energy, `0.04259` (base,
`conic1e10`) and `0.0428` (`etol1e3`, `dykstra60`), against `0.02722`
recorded and `0.0337` combined.)

Score per four-unit window `(a, b]` of the horizon, the same RMS on that
window's samples alone (`windows.py`), base runs and the `etol1e3` twin of
this rule:

| window | recorded base | combined base | measured base | measured `etol1e3` | measured `dykstra60` |
|---|---|---|---|---|---|
| `(0, 4]` | `0.0237` | `0.0316` | `0.0272` | `0.0272` | `0.0272` |
| `(4, 8]` | `0.0599` | `0.1021` | `0.0472` | `0.0474` | `0.0473` |
| `(8, 12]` | `0.2070` | `0.2247` | `0.1611` | `0.1669` | `0.1659` |
| `(12, 16]` | `0.4694` | `0.3716` | `0.2132` | `0.2723` | `0.2350` |
| `(16, 20]` | `0.5224` | `0.3866` | `0.1956` | `0.3803` | `0.2087` |

Physicality over `t <= T` (scaled 24-row extension minimum; unshifted
retained joint Gram minimum), base / `etol1e3` / `dykstra60` /
`conic1e10`; the standardized frontier envelope's first arrival at `1.0`
and the number of the 8,001 samples at or above it:

| set | extension minimum, `t <= 8` / `<= 12` / `<= 20` | joint Gram minimum, `t <= 8` / `<= 12` / `<= 20` | envelope first at `1.0` (samples at `1.0`) |
|---|---|---|---|
| recorded | `-8.6e-5` / `-6.0e-4` / `-2.3e-3`; `-1.7e-4` / `-6.8e-4` / `-2.0e-3`; `-3.4e-5` / `-1.4e-4` / `-5.9e-4`; `-1.6e-4` / `-3.8e-4` / `-1.6e-3` | `-2.1e-3` / `-1.4e-2` / `-5.3e-2`; `-4.7e-3` / `-1.6e-2` / `-4.6e-2`; `-8.0e-4` / `-3.4e-3` / `-1.4e-2`; `-4.8e-3` / `-9.2e-3` / `-3.6e-2` | never (maxima `0.84`, `0.92`, `0.72`, `0.89`) |
| combined | `-2.9e-4` / `-3.1e-4` / `-3.5e-4`; `-3.6e-4` / `-3.6e-4` / `-5.5e-4`; `-1.7e-4` / `-1.7e-4` / `-3.2e-4`; `-3.5e-4` / `-3.5e-4` / `-5.8e-4` | `-6.8e-3` / `-9.1e-3` / `-9.1e-3`; `-8.7e-3` / `-1.1e-2` / `-1.3e-2`; `-4.1e-3` / `-4.9e-3` / `-7.6e-3`; `-8.6e-3` / `-9.2e-3` / `-1.4e-2` | never (`0.63`, `0.84`, `0.97`, `0.95`) |
| measured, no anchor | `-1.1e-4` / `-2.6e-4` / `-2.6e-4`; `-1.1e-4` / `-2.5e-4` / `-6.6e-4`; `-4.8e-5` / `-1.2e-4` / `-1.2e-4`; `-1.1e-4` / `-2.6e-4` / `-2.6e-4` | `-3.3e-3` / `-7.9e-3` / `-7.9e-3`; `-3.2e-3` / `-7.8e-3` / `-1.6e-2`; `-1.4e-3` / `-3.7e-3` / `-3.7e-3`; `-3.3e-3` / `-7.9e-3` / `-7.9e-3` | `t=10.04` (199), `18.96` (79), `10.20` (15), `10.04` (108); maxima `1.00012`, `1.00016`, `1.00001`, `1.00012` |

Site-0 occupation RMS over `t=0` to `20`: `0.158` / `0.201` / `0.159` /
`0.159` (recorded rule `0.187` / `0.191` / `0.192` / `0.183`; combined
`0.337` / `0.238` / `0.252` / `0.264`; the both-cones record `0.205`).

## (a) Where each twin leaves the base, and what moves it

31-coordinate Euclidean separation from the base at the same accepted step
(the combined rule's crossings from task 7 and the recorded rule's from
task 6 in brackets):

| twin | first nonzero | first carried-frontier difference above `1e-6` | first crossing of `1e-3` / `1e-2` / `1e-1` [combined] [recorded] | at `t=4` / `8` / `12` / `16` / `20` | window mean `(8,12]` / `(12,16]` / `(16,20]` | maximum, when |
|---|---|---|---|---|---|---|
| `etol1e3` | `t=0.0075` | `0.0064` at `t=0.16` | `2.1075` / `5.83` / `11.2275` [`1.7875` / `4.1` / `8.415`] [`0.1475` / `0.7525` / `7.01`] | `0.0035` / `0.0175` / `0.214` / `0.755` / `2.27` | `0.066` / `0.49` / `1.49` | `2.27` at `t=20` |
| `dykstra60` | `t=0.0625` | `0.0064` at `t=0.16` | `2.105` / `5.76` / `12.905` [`2.305` / `5.65` / `8.6025`] [`0.145` / `0.36` / `4.0025`] | `0.0035` / `0.0232` / `0.066` / `0.347` / `0.534` | `0.046` / `0.23` / `0.43` | `0.534` at `t=20` |
| `conic1e10` | `t=0.245` (`8.7e-19`) | `0.0022` at `t=10.04` | `14.205` / `18.6425` / never [`6.4525` / `6.8` / `8.545`] [`0.125` / `0.48` / `3.085`] | `5.1e-9` / `1.7e-8` / `1.2e-4` / `0.0030` / `0.0875` | `2.4e-5` / `0.0014` / `0.023` | `0.0886` at `t=19.9` |

Under the combined rule every twin crossed `1e-1` at `t=8.4` to `8.6`;
here the crossings are at `11.2`, `12.9` and never, so the continuity
extends by three to four time units and, for the twin that differs only
in the certificate tolerance, to the end of the horizon.  What then parts
the other two is measured in the stage records (`reselection_moves.py`):

| run | accepted moves per quarter `[0,5)` / `[5,10)` / `[10,15)` / `[15,20)`: median, 90th percentile, maximum (count above `0.1`) | largest applied moves (time; certificate; declared-objective improvement) |
|---|---|---|
| base | `0.010`, `0.013`, `0.019` (0) / `0.023`, `0.035`, `0.152` (2) / `0.026`, `0.048`, `0.716` (5) / `0.044`, `0.246`, `0.624` (13) | `0.716` at `t=10.04` (uncertified, conic-Newton descent failure; `+0.175`); `0.686` at `10.2` (certified; `+1.588`); `0.624` at `17.12` (uncertified, Newton cone failure on the 23/24 face; `+0.657`); `0.610` at `17.16` (certified; `+1.726`); `0.513` at `20` (uncertified; `+1.588`) |
| `etol1e3` | `0.010`, `0.014`, `0.018` (0) / `0.023`, `0.034`, `0.164` (2) / `0.023`, `0.051`, `0.522` (5) / `0.045`, `0.190`, `0.805` (15) | `0.805` at `t=18.96` (uncertified, descent failure; `+0.050`); `0.533` at `19.16` (certified; `+1.525`); `0.522` at `10.28` (uncertified; `+0.440`); `0.434` at `10.32` (uncertified, 40-iteration cap, KKT `1.0e-7`; `+1.269`); `0.394` at `10.04` (uncertified, 40-iteration cap; `+0.641`) |
| `dykstra60` | `0.010`, `0.013`, `0.018` (0) / `0.022`, `0.033`, `0.155` (2) / `0.023`, `0.039`, `0.510` (5) / `0.017`, `0.032`, `0.471` (5) | `0.510` at `t=10.2` (uncertified, descent failure; `+0.308`); `0.471` at `17.32` (certified; `+1.501`); `0.358` at `10.24` (uncertified; `+1.218`); `0.278` at `17.24` (uncertified, 40-iteration cap; `+1.202`); `0.265` at `17.28` (uncertified, `AlmostSolved`, KKT `0.24`; `+0.736`) |
| `conic1e10` | `0.010`, `0.013`, `0.019` (0) / `0.023`, `0.035`, `0.152` (2) / `0.026`, `0.048`, `0.717` (5) / `0.045`, `0.202`, `0.510` (14) | `0.717` at `t=10.04` (uncertified, `Solved` with KKT `0.998`; `+0.175`: the base's iterate to `0.0022`); `0.687` at `10.2` (uncertified, 40-iteration cap, KKT `3.5e-8`; `+1.590`); `0.510` at `17.28` (uncertified; `+0.382`) |

Three readings.

- **To `t=10` the rule moves the lift by hundredths and the twins stay
  together.**  No accepted move exceeds `0.019` in `[0, 5)` and two exceed
  `0.1` in `[5, 10)` (`0.13` at `t=8.68` and `0.15` to `0.16` at `t=9.56`
  to `9.64`, certified in every run but the `conic1e10` twin's second);
  the twins' separations stay below `0.024` to `t=8`, are `0.051`
  (`etol1e3`, `dykstra60`) at `t=10`, and stay below `0.07` (`dykstra60`)
  and `0.22` (`etol1e3`) to `t=12`.  This is task 8's continuity carried
  four time units further.
- **From `t=10` the rule accepts moves of `0.4` to `0.8`, most of them
  uncertified.**  They are the last iterates of conic-Newton descent
  failures, Newton cone failures and 40-iteration caps that pass the
  completion's own feasibility test of a warm frontier and improve the
  declared objective by `0.05` to `1.7`, so the rule as written accepts
  them; the certified answers that follow within `0.04` to `0.2` time
  units are of the same size (`0.686`, `0.610`, `0.533`, `0.471`).  The
  frontier envelope reaches its declared scale at the first of them
  (`t=10.04` in the base, `10.2` in `dykstra60`, `18.96` in `etol1e3`) and
  stays there for 15 to 199 samples.  The `etol1e3` twin's departure is
  its `0.394` / `0.522` / `0.434` moves at `t=10.04` to `10.32` where the
  base took `0.716` / `0.686`: two states `0.05` apart at `t=10` whose
  completions return different large iterates, an answer branch of the
  kind task 7 saw once at `t=6.44` under the combined rule.  The
  `conic1e10` twin, whose state equals the base's to `1.7e-8` at `t=8`
  and `1.2e-4` at `t=12`, accepts the same large iterates and stays
  within `0.09` to the end: the certificate tolerance does not branch
  this rule anywhere on the horizon.
- **The `infeasible` regime begins at `t=8` to `10` and the `dykstra60`
  twin largely avoids it.**  Re-selections whose iterate fails the
  measured feasibility test: base 0 / 34 / 29 / 56 per quarter, `etol1e3`
  0 / 32 / 24 / 53, `conic1e10` 0 / 34 / 29 / 52, `dykstra60` 0 / 32 / 1
  / 1, so the 60-iteration repair keeps the state's lower moments
  admitting a positive extension through `t=10` to `20` where the
  30-iteration repair loses it at 24 to 56 of 125 re-selections per
  quarter; that twin accepts 124 of 125 in each of the last two quarters
  and scores `0.161` at `t=20`, 7 percent above the base, with the
  tightest cones of the four (`-1.2e-4` / `-3.7e-3`).

## (b) The gain past `t=8`

- Over `t=0` to `12` the four runs (`0.0982` to `0.1014`) lie below the
  recorded rule's best twin (`0.1082`) and the combined rule's best
  (`0.1205`); over `t=0` to `16` (`0.136` to `0.162`) below `0.202` and
  `0.184`; over `t=0` to `20` (`0.150` to `0.223`) below `0.260` and
  `0.255`.  On the windows: `(8, 12]` `0.161` against the recorded base's
  `0.207`, `(12, 16]` `0.213` against `0.469`, `(16, 20]` `0.196` against
  `0.522` -- the recorded row is the frozen-lift route past `t=11.76`
  (task 7 (a)) and its last two windows are that route's decay; this rule
  keeps re-selecting to the end (69 to 124 accepted of 125 in `[15, 20)`
  against the recorded rule's none) and its last window is no worse than
  its `(8, 12]` window.
- Against the record over `t=0` to `20`: the base's `0.150` stands at 1.17
  times the both-cones line (`0.128607`) and below the 15-key
  `post_pulse` row (`0.239`, task 4's best 15-key row at `t=20` until now)
  by 37 percent, with the site-0 occupation RMS `0.158` against the
  both-cones `0.205`; the 27-key chains remain ahead (`0.0852`
  `drive_power`, `0.0699` `post_pulse`).  The cones are not held: joint
  Gram minimum `-7.9e-3` (both cones `+1.4e-5`), 6.7 times better than
  the recorded row's `-5.3e-2` and within 15 percent of the combined
  rule's `-9.1e-3`; scaled extension minimum `-2.6e-4` against the
  recorded `-2.3e-3`; the frontier box is exceeded by at most `1.6e-4`.
- The internal energy is the worst of the three sets at every horizon
  (`0.0409` at `t=20` against `0.0260` recorded and `0.0320` combined),
  and it is set over the pulse: the drive-power reference is `0.0426`
  above the exact energy from `t=4` on, task 8's number, and no
  re-selection after `t=4` changes it.

## (c) Re-selections of 500 over `t=0` to `20`

| run | accepted (of which uncertified) / declined by the margin / `infeasible` | solver refusals (uncertified returns), of which accepted | accepted per quarter `[0,5)` / `[5,10)` / `[10,15)` / `[15,20)` |
|---|---|---|---|
| base | 377 (39) / 4 / 119 | 161, 39 | 120 / 91 / 96 / 69 |
| `etol1e3` | 385 (51) / 5 / 110 | 163, 51 | 120 / 93 / 101 / 71 |
| `dykstra60` | 462 (35) / 4 / 34 | 72, 35 | 120 / 93 / 124 / 124 |
| `conic1e10` | 380 (88) / 4 / 116 | 208, 88 | 120 / 91 / 96 / 73 |
| recorded rule, base (certified and accepted / refused) | 224 (0) / 0 / 0; 276 refused | 276, 0 | 112 / 98 / 14 / 0 |
| combined rule, base | 347 (30) / 4 / 149 | 182, 30 | 120 / 55 / 78 / 93 |

(the four declined are the wrong-face answers of `t=0.04` to `0.16`, as
in tasks 7 and 8; no re-selection is a `refused` decision under the
`measured` scope; the recorded rule's twins carry no stage record and
their whole-run solver refusals are 285, 192, 267.)

## The 15-key ledger row

For the plan file's cheap-projection row (15 entrance keys, `N=16`,
strong coupling, cutoff 16, exact-correlated preparation, `h=0.0025`,
Dykstra 30 on both targets at stage cadence, the declared proximal
log-det completion re-solved every 16 steps warm-started on the carried
frontier, the energy-neutral repair in `drive_power` mode, and the lift
re-selection rule margin `1e-3`, `measured` candidates, no anchor; every
band the base and its three one-setting twins, `energy_tolerance` `1e-3`,
Dykstra 60, re-selection `conic_tolerance` `1e-10`):

| horizon `t=0` to `T` | score, base (band of the four; spread) | Eq. (22) internal-energy RMS (band) | recorded rule's band (task 6) | record, both cones |
|---:|---|---|---|---|
| 4 | `0.0272` (`0.0272` to `0.0272`; 0.1 %) | `0.0331` (`0.0331` to `0.0333`) | `0.0237` to `0.0280` | --- |
| 8 | `0.0385` (`0.0385` to `0.0386`; 0.2 %) | `0.0382` (`0.0382` to `0.0384`) | `0.0404` to `0.0456` | --- |
| 12 | `0.0982` (`0.0982` to `0.1014`; 3.2 %) | `0.0397` (`0.0397` to `0.0399`) | `0.108` to `0.131` | --- |
| 16 | `0.136` (`0.136` to `0.162`; 18.8 %, three of four within 7.4 %) | `0.0404` (`0.0404` to `0.0406`) | `0.202` to `0.258` | --- |
| 20 | `0.150` (`0.150` to `0.223`; 48.8 %, three of four within 7.1 %) | `0.0409` (`0.0409` to `0.0411`) | `0.260` to `0.329` | `0.1286` (energy `0.00094`; joint Gram `+1.4e-5`) |

with the joint Gram minimum over `t=0` to `20` `-7.9e-3` (band `-3.7e-3`
to `-1.6e-2`), the scaled extension minimum `-2.6e-4` (`-1.2e-4` to
`-6.6e-4`), the site-0 occupation RMS `0.158` (`0.158` to `0.201`), walls
`841` to `900 s` for 8,000 steps under `ram_guard --limit-mb 3000`, and
the caveat that the `t=16` and `t=20` bands are set by the accepted
`0.4` to `0.8` frontier moves from `t=10` on, which no option of the rule
caps in these runs (the trust radius was off).  The plan file
(`agent_guidance/paper-v/paper-v-state-and-plan-20260914.md`, the
cheap-projection row) was not edited here: it is shared with the
projective-route agent, which commits to it, and this route does not
commit.

## Provenance and gates

- Gates: `paper_5/tests/test_cheap_projection_route.py` 44 passed in
  `24.03 s` (`cd paper_5 && PYTHONPATH=src python -m pytest -q
  tests/test_cheap_projection_route.py`).  The full `paper_5` suite was
  not re-run: no module, test or default changed since task 8's pass
  (499 passed) of the same module contents, committed at `4de52486`
  (`HEAD`), and the projective-route agent's chain was running on the
  machine.
- Scripts, all under
  `output/local_runs/paper_v_cheap_projection_task9_measured_t20_20260916_v1/`:
  `run_one.zsh` (task 7's, output directory changed), `chain.zsh` (the
  four rollouts in order, every twin and the rule as flags),
  `task9_scores.py` (`task9_scores.json` and `.md`: the five-horizon
  scores, internal energies, site-0 occupation, per-horizon physicality
  minima with the joint Gram recomputed from the states, spreads,
  separations with crossings and horizon values, decisions over `t <= T`
  and per quarter, from `trajectory.npz`, `summary.json` and
  `repairs.jsonl`; it propagates nothing and reproduces every `t=4`, `8`,
  `12`, `20` number of tasks 7 and 8), `windows.py` (`windows.json` and
  `.md`: the score, internal energy and site-0 RMS per four-unit window
  and the twins' mean and maximum separation per window),
  `reselection_moves.py` (`reselection_moves.json`: the ten largest
  applied moves per run with certificate, face, KKT backward error and
  objective improvement; the accepted moves' median, 90th percentile and
  maximum per quarter; the envelope's first arrival at `0.99` and `1.0`),
  the run logs, `chain.log`, `chain_stdout.log`, `task9_scores.log`, the
  partial score `task9_scores_partial.json` (taken after the base run),
  and the smoke
  `paper_v_cheap_projection_task9_smoke_measured_t005_20260916_v1`
  (`1.7 s`, peak `228 MiB`).
- Artifacts, under `output/local_runs/`:
  `paper_v_cheap_projection_task9_measured_{base,etol1e3,dykstra60,conic1e10}_t20_20260916_v1`,
  each with `repairs.jsonl` (`15 MB`); walls `900.2`, `851.5`, `841.1`,
  `848.7 s` (`autonomous_wall_seconds`; `1,140 s` budget, none reached
  it), peaks `782`, `771`, `774`, `784 MiB`; every rollout under
  `pipelines/shell/ram_guard.py --limit-mb 3000`, one at a time,
  `17:55` to `18:53`.  `HEAD` is `4de52486` throughout.

## What to try next, in order

1. Cap the move.  The rule's trust radius (`--reselection-trust-radius`,
   task 7's option, inert at `0.05` over `t=0` to `2` and off here) is the
   declared instrument for the `0.4` to `0.8` moves that set the `t=16`
   and `t=20` bands: radius `0.1` under `ball` and under `carried`, on
   the four to `t=20` (eight rollouts of about 14 minutes), to learn
   whether the `t=12` continuity (3.2 percent) extends to `t=20` and what
   the cap costs against the base's `0.150`.  A radius binds 5 to 15
   times per quarter from `t=10` on by the counts above.
2. The same rule (and, if 1 helps, the cap) on the 27-key `N=16`
   drive-power chain: task 8's item 2, unchanged.
3. The Eq. (22) energy: if it is to be scored, the drive-power
   reference's `0.0426` over the pulse is the quantity, set before any
   re-selection rule acts; task 8's item 3, unchanged.

## Structured summary of task 9

| item | value |
|---|---|
| what was run | four `t=0` to `20` rollouts (8,000 steps) of the recorded `N=16` drive-power configuration under the rule margin `1e-3`, `measured` candidates, no anchor, on the base and the three one-setting twins, one at a time under `ram_guard --limit-mb 3000`; walls `900.2` / `851.5` / `841.1` / `848.7 s`, peaks `782` to `784 MiB`; no module edited, 44 tests pass, nothing committed |
| scores, base / `etol1e3` / `dykstra60` / `conic1e10` | `t=4` `0.027191` / `0.027215` / `0.027214` / `0.027191`; `t=8` `0.038538` / `0.038617` / `0.038605` / `0.038538`; `t=12` `0.098187` / `0.101367` / `0.100824` / `0.098186`; `t=16` `0.136367` / `0.161977` / `0.146397` / `0.136288`; `t=20` `0.150102` / `0.223420` / `0.160803` / `0.150285` |
| spread (recorded rule; combined rule) | 0.1 (17.7; 0.1), 0.2 (12.7; 4.2), 3.2 (21.0; 19.2), 18.8 (28.1; 21.6), 48.8 (26.4; 7.4) percent at `T=4`, `8`, `12`, `16`, `20`; without `etol1e3` 0.1, 0.2, 2.7, 7.4, 7.1 |
| does the continuity survive past `t=8` without the anchor | to `t=12` yes (3.2 percent, the second digit); to `t=16` and `20` no in the `etol1e3` twin (`1e-1` crossed at `t=11.23`, separation `2.27` at `t=20`) and marginally in `dykstra60` (`12.9`; `0.53`); the `conic1e10` twin never crosses `1e-1` (`0.089`); the combined rule's crossings were `8.4` to `8.6`, the recorded rule's `3.1` to `7.0` |
| does the `t=8` gain survive | yes, and grows: every run below every recorded-rule and combined-rule run from `t=8` on, bands disjoint at `8`, `12`, `16`, `20`; the base 22 / 47 / 54 percent below the recorded row at `t=12` / `16` / `20` and 42 percent below the recorded rule's best twin at `t=20`; `0.150` is 1.17 times the both-cones `0.1286` and 37 percent below the 15-key `post_pulse` row `0.239`; window RMS `(16, 20]` `0.196` against the recorded `0.522` |
| what breaks the continuity past `t=12` | accepted moves of `0.4` to `0.8` from `t=10` on (base `0.716` at `t=10.04`, uncertified, objective improvement `0.175`; 13 to 15 moves above `0.1` in `[15, 20)`), the envelope at `1.0` from `t=10.04`; an answer branch between states `0.05` apart at `t=10`, not the certificate (`conic1e10` takes the base's iterate to `0.0022`) |
| internal energy | `0.0331` / `0.0382` / `0.0397` / `0.0404` / `0.0409` on the five horizons (band 0.5 percent), against the recorded `0.0204` / `0.0241` / `0.0252` / `0.0257` / `0.0260`; the drive-power reference `0.0426` above exact from `t=4` on |
| physicality over `t <= 20` | scaled extension minimum `-2.6e-4` / `-6.6e-4` / `-1.2e-4` / `-2.6e-4` (recorded `-2.3e-3` to `-5.9e-4`); joint Gram `-7.9e-3` / `-1.6e-2` / `-3.7e-3` / `-7.9e-3` (recorded `-5.3e-2` to `-1.4e-2`; combined `-7.6e-3` to `-1.4e-2`); frontier box exceeded by at most `1.6e-4` |
| re-selections of 500: accepted (uncertified) / declined / `infeasible`; solver refusals | 377 (39) / 4 / 119; 161 -- 385 (51) / 5 / 110; 163 -- 462 (35) / 4 / 34; 72 -- 380 (88) / 4 / 116; 208; accepted in `[15, 20)` 69, 71, 124, 73 (recorded rule none, combined 93) |
| ledger row | the table under "The 15-key ledger row": `0.0272` (0.1 %), `0.0385` (0.2 %), `0.0982` (3.2 %), `0.136` (18.8 %; three within 7.4), `0.150` (48.8 %; three within 7.1) at `t=4`, `8`, `12`, `16`, `20`; plan file not edited |
| next | trust radius `0.1` (`ball`, `carried`) on the four to `t=20`; then the 27-key chain |
| files changed | this report, the worklog block, and the task-9 scripts and artifacts under `output/local_runs/`; no module, test or default; nothing committed |

# Task 10: the trust radius `0.1` on the frontier move under `ball` and `carried`, and the certified screen, to `t=20` -- the cap improves the base to `1.03` times the both-cones line and closes no band; the `carried` policy is a freeze from `t=8.68` (2026-09-16/17)

Nine rollouts of the recorded `N=16` drive-power configuration to `t=20`
(8,000 steps), one at a time under `ram_guard --limit-mb 3000`
(`chain.zsh`, `19:07` to `21:12` on 2026-09-16): task 9's rule (margin
`1e-3`, `measured` candidates, no anchor) with the declared trust radius
`0.1` on the Euclidean move of the standardized frontier in one
re-selection, under the `ball` policy (the move scaled to the radius on the
segment toward the accepted lift) and under the `carried` policy (the
carried lift kept when the cap binds), each on the base and the three
one-setting twins (`energy_tolerance` `1e-3`; Dykstra 60; re-selection
`conic_tolerance` `1e-10`); and, base only, the certified screen (margin
`1e-3`, `certified` candidates, no anchor, no trust radius: an uncertified
return is `refused` outright and never tested).  Every run is `run_one.zsh`
(task 7's script, output directory changed) plus the twin's one setting
plus the rule's flags on its command line; no module was edited (the radius
and both policies are task 7's `LiftReselectionRule` fields, committed at
`a81b7ecb`, the last commit to the owned modules; `HEAD` is `3ea96121`
throughout); the 44 tests of `test_cheap_projection_route.py` pass before
the chain (`23.95 s`) and after the scoring (`25.20 s`); nothing is
committed.  Walls `817.8`, `825.5`, `847.1`, `761.6 s` (`ball` base /
`etol1e3` / `dykstra60` / `conic1e10`), `879.5`, `838.1`, `860.1`,
`832.3 s` (`carried`, same order), `743.4 s` (the certified screen), all
against the `1,140 s` budget and all at `t=20` with 8,001 samples; peaks
`778` to `872 MiB` (`ram_guard` `745` to `794 MB`).  The agent instance
that launched the chain was lost to a network outage after it completed
and before the scoring; this section, the scoring (`analyse.zsh`) and the
worklog block were written on 2026-09-17 from the artifacts, and no
rollout was repeated.  Conventions are tasks 7 to 9's (score, internal
energy, site-0 occupation RMS, physicality minima over `t <= T`,
separation, spread, move, answer, solver refusal); a cap-bound
re-selection is one whose candidate's move exceeded the radius, decision
`cap_bound_ball` (counted among the accepted, since the lift moves) or
`cap_bound_carried` (counted apart, since it does not).  The comparison
rows are task 9's four (the same rule without the radius), the recorded
rule's four and the combined rule's four, rescored by one script
(`task10_scores.py`, task 9's with the three new sets; every task-9 number
is reproduced to every digit).  The smoke run under the `ball` flags
(`t=0.05`, `1.7 s`, peak `232 MiB`) equals task 9's smoke in every array
and records the rule with `trust_radius` `0.1`, `trust_policy` `ball`.

## Negative result first

The trust radius `0.1` closes neither the `t=16` nor the `t=20` band below
10 percent under either policy.  Under `ball` the four scores over `t=0`
to `20` are `0.132542` / `0.163027` / `0.206990` / `0.132542` (base /
`etol1e3` / `dykstra60` / `conic1e10`): spread 56.2 percent (task 9's rule
48.8), 19.3 percent at `t=16` (`0.132318` to `0.157814`; task 9's 18.8)
and 5.5 at `t=12` (`0.098867` to `0.104307`; task 9's 3.2); three of the
four lie within 14.3 percent at `t=16` and 23.0 at `t=20`, the `dykstra60`
twin apart.  Under `carried` the spread is 6.9, 10.1 and 16.3 percent at
`t=12`, `16`, `20`, the narrowest bands past `t=12` of any rule on record,
at scores `0.135734` to `0.145165`, `0.187509` to `0.206528` and
`0.216450` to `0.251668`, 37 to 63 percent above the `ball` base at the
same horizons, because from `t=8.68` the `carried` policy keeps the
carried lift at all but 7 of the remaining 283 re-selections (78 accepted
in `[5, 10)`, then 5 and 2 in `[10, 15)` and `[15, 20)`; 175 cap-bound,
116 `infeasible`): it is the frozen-lift route with the Dykstra repair
from `t=8.68` on, and its band is that route's.  The radius binds first at
`t=8.68` in every one of the eight runs (the certified answer `0.125` to
`0.133`, task 9's first move above `0.1`), so the eight equal task 9's four
in every array to `t=8.68`, and the `t=4` and `t=8` rows below are
task 9's.

What the cap buys, stated once: the `ball` base scores `0.132542` over
`t=0` to `20`, 12 percent below task 9's base (`0.150102`), 60 percent
below the recorded row (`0.328674`) and 1.03 times the both-cones line
(`0.128607`), with the site-0 occupation RMS `0.1245` against the
both-cones `0.205`, the last window `(16, 20]` at `0.133` against task 9's
`0.196`, the `conic1e10` twin identical to it within `6.0e-5` in the 31
coordinates at every sample, the frontier envelope never reaching the box
(maximum `0.748`; task 9's base sat at `1.0` for 199 samples), and the
same cones as task 9's base (`-2.6e-4` / `-7.9e-3`, attained at `t=8.37`,
before the first cap).  The internal energy is unchanged (`0.0409`: the
drive-power reference is set over the pulse).  On the base's own
trajectory the cap never touches a `0.4` to `0.8` answer: of its 72
answers above `0.4`, 69 fail the measured feasibility test (`infeasible`)
and 3 are declined by the margin (the wrong-face answers of `t=0.08` and
`0.16`, and the `0.894` uncertified iterate at `t=10.04`, where task 9's
base accepted `0.716`); the eight cap-bound re-selections are answers of
`0.13` to `0.25`, seven certified, one uncertified (`0.139` at `t=16.88`).
The two caps at `t=8.68` and `9.64` (answers `0.133` and `0.152` scaled to
`0.1`) change the state by enough that the `t=10.04` descent-failure
iterate no longer improves the declared objective; that is the gain, and
also the branch: on the `etol1e3` twin's trajectory the `0.670` and
`0.511` uncertified iterates of `t=10.08` and `10.12` pass the test and are
capped to `0.1`, on `dykstra60`'s the `0.429` at `10.28` and the `0.567` at
`16.44`, after which that twin enters the `infeasible` regime it avoided in
task 9 (38 of 125 in `[15, 20)` against 1) and scores `0.339` on
`(16, 20]`, the worst window of the set.

## The rows side by side

Scores (all-coordinate scalar RMS over `t=0` to `T`), base / `etol1e3` /
`dykstra60` / `conic1e10`, with the spread of the four; the certified
screen is one run:

| `T` | task 9 (measured, no anchor) | spread | `ball`, radius `0.1` | spread (three of four) | `carried`, radius `0.1` | spread | certified screen, base |
|---:|---|---|---|---|---|---|---|
| 4 | `0.027191` / `0.027215` / `0.027214` / `0.027191` | 0.1 % | the same four | 0.1 % | the same four | 0.1 % | `0.027635` |
| 8 | `0.038538` / `0.038617` / `0.038605` / `0.038538` | 0.2 % | the same four | 0.2 % | the same four | 0.2 % | `0.046342` |
| 12 | `0.098187` / `0.101367` / `0.100824` / `0.098186` | 3.2 % | `0.098867` / `0.104307` / `0.103216` / `0.098867` | 5.5 % (4.4) | `0.135734` / `0.145165` / `0.141612` / `0.135734` | 6.9 % | `0.097956` |
| 16 | `0.136367` / `0.161977` / `0.146397` / `0.136288` | 18.8 % | `0.132318` / `0.151238` / `0.157814` / `0.132318` | 19.3 % (14.3) | `0.187509` / `0.206528` / `0.196694` / `0.187509` | 10.1 % | `0.123294` |
| 20 | `0.150102` / `0.223420` / `0.160803` / `0.150285` | 48.8 % | `0.132542` / `0.163027` / `0.206990` / `0.132542` | 56.2 % (23.0) | `0.216450` / `0.251668` / `0.236386` / `0.216450` | 16.3 % | `0.149326` |

(the recorded rule's and the combined rule's rows are task 9's table,
unchanged: spreads 17.7, 12.7, 21.0, 28.1, 26.4 and 0.1, 4.2, 19.2, 21.6,
7.4 percent.)  Across the eight runs of the two rules that keep
re-selecting to the end (task 9's and `ball`) the scores span `0.0982` to
`0.1043` at `t=12` (6.2 percent), `0.132` to `0.162` at `t=16` (22.4) and
`0.133` to `0.223` at `t=20` (68.6).

Internal-energy sample RMS (Eq. (22)) over `t=0` to `T`: the `ball` and
`carried` sets equal task 9's set at every horizon and in every run
(`0.03314` / `0.03816` / `0.03969` / `0.04043` / `0.04087` on the base and
`conic1e10`, `3e-4` above on `etol1e3` and `dykstra60`; spread 0.5
percent), since the runs coincide to `t=8.68` and the drive-power
reference is fixed at `t=4`; the certified screen `0.03323` / `0.03845` /
`0.04004` / `0.04081` / `0.04127` (its reference `0.04304` above the exact
energy from `t=4` on; task 9's `0.04259`).  Site-0 occupation RMS over
`t=0` to `20`: `ball` `0.1245` / `0.1705` / `0.1695` / `0.1245`, `carried`
`0.2279` / `0.2752` / `0.2566` / `0.2279`, certified `0.1464` (task 9
`0.158` / `0.201` / `0.159` / `0.159`; recorded `0.183` to `0.192`; both
cones `0.205`).

Score per four-unit window `(a, b]` (`windows.py`):

| window | task 9 base | `ball` base | `ball` `etol1e3` | `ball` `dykstra60` | `carried` base | `carried` `etol1e3` | `carried` `dykstra60` | certified | recorded base |
|---|---|---|---|---|---|---|---|---|---|
| `(0, 4]` | `0.0272` | `0.0272` | `0.0272` | `0.0272` | `0.0272` | `0.0272` | `0.0272` | `0.0276` | `0.0237` |
| `(4, 8]` | `0.0472` | `0.0472` | `0.0474` | `0.0473` | `0.0472` | `0.0474` | `0.0473` | `0.0594` | `0.0599` |
| `(8, 12]` | `0.1611` | `0.1624` | `0.1722` | `0.1703` | `0.2287` | `0.2455` | `0.2392` | `0.1565` | `0.2070` |
| `(12, 16]` | `0.2132` | `0.2018` | `0.2426` | `0.2601` | `0.2922` | `0.3277` | `0.3076` | `0.1790` | `0.4694` |
| `(16, 20]` | `0.1956` | `0.1334` | `0.2035` | `0.3386` | `0.3060` | `0.3822` | `0.3531` | `0.2251` | `0.5224` |

(the `conic1e10` twins' windows equal their bases' to four digits.)

Physicality over `t <= T` (scaled 24-row extension minimum; unshifted
retained joint Gram minimum), base / `etol1e3` / `dykstra60` /
`conic1e10`, the time at which the `t <= 20` extension minimum is
attained, and the largest standardized frontier envelope ratio over
`t <= 20`:

| set | extension minimum, `t <= 8` / `<= 12` / `<= 20` (attained at) | joint Gram minimum, `t <= 8` / `<= 12` / `<= 20` | envelope maximum |
|---|---|---|---|
| task 9 | `-1.1e-4` / `-2.6e-4` / `-2.6e-4` (`8.37`); `-1.1e-4` / `-2.5e-4` / `-6.6e-4` (`20.0`); `-4.8e-5` / `-1.2e-4` / `-1.2e-4` (`8.375`); as base | `-3.3e-3` / `-7.9e-3` / `-7.9e-3`; `-3.2e-3` / `-7.8e-3` / `-1.6e-2`; `-1.4e-3` / `-3.7e-3` / `-3.7e-3`; as base | `1.0` in every run (from `t=10.04`, `18.96`, `10.20`, `10.04`) |
| `ball` | `-1.1e-4` / `-2.6e-4` / `-2.6e-4` (`8.37`); `-1.1e-4` / `-2.5e-4` / `-2.5e-4` (`8.3775`); `-4.8e-5` / `-1.4e-4` / `-2.2e-4` (`18.98`); as base | `-3.3e-3` / `-7.9e-3` / `-7.9e-3` (at `8.37`); `-3.2e-3` / `-7.8e-3` / `-7.8e-3`; `-1.4e-3` / `-3.7e-3` / `-5.4e-3`; as base | `0.748` / `0.819` / `0.910` / `0.748` |
| `carried` | `-1.1e-4` / `-5.3e-4` / `-5.3e-4` (`10.26`); `-1.1e-4` / `-5.7e-4` / `-5.7e-4` (`10.25`); `-4.8e-5` / `-2.7e-4` / `-2.7e-4` (`10.24`); as base | `-3.3e-3` / `-1.2e-2` / `-1.2e-2` (at `10.295`); `-3.2e-3` / `-1.3e-2` / `-1.3e-2`; `-1.4e-3` / `-6.2e-3` / `-6.2e-3`; as base | `0.593` / `0.600` / `0.603` / `0.593` |
| certified screen | `-1.7e-4` / `-1.7e-4` / `-1.9e-4` (`20.0`) | `-4.4e-3` / `-4.4e-3` / `-4.4e-3` (in `t <= 8`) | `0.710` |

The `ball` base's cones are task 9's, attained at `t=8.37` before the cap
first binds; the `carried` runs' are 1.6 to 2.1 times looser and are
attained at `t=10.24` to `10.30`, in the frozen stretch; the `ball`
`dykstra60` twin's extension minimum moves from `t=8.375` to `18.98` and
loosens by 1.8, with the late `infeasible` regime.  Under both policies
the envelope stays below the box, and under `carried` below `0.61`.

## (a) `ball`: what the cap catches, and where the gain and the branch come from

Cap-bound re-selections per run, with the answers they scaled to `0.1`
(`cap_events.py`, `cap_summary.py`):

| run | cap-bound (uncertified) | first, last | per quarter `[0,5)` / `[5,10)` / `[10,15)` / `[15,20)` | the answers capped (time; certificate) |
|---|---|---|---|---|
| base | 8 (1) | `8.68`, `16.88` | 0 / 2 / 5 / 1 | `0.133` (`8.68`), `0.152` (`9.64`), `0.228` (`10.08`), `0.142` (`10.16`), `0.254` (`10.28`), `0.167` (`10.32`), `0.141` (`13.48`), all certified; `0.139` (`16.88`, uncertified descent failure) |
| `etol1e3` | 28 (8) | `8.68`, `19.76` | 0 / 2 / 10 / 16 | `0.131`, `0.139` (`8.68`, `9.6`); `0.183`, `0.354` (`10.0`, `10.04`, certified), `0.670`, `0.511` (`10.08`, `10.12`, uncertified descent failures), `0.341` to `0.110` (`10.2` to `10.32`), `0.181` (`10.4`, 40-iteration cap); `0.111` (`14.16`); `0.121`, `0.106` (`15.44`, `16.24`); `0.176`, `0.116` (`16.84`, `16.92`, uncertified); `0.146`, `0.142` (`17.28`, `17.32`); nine in a row `0.137` to `0.332` at `18.48` to `18.8`; `0.214` (`19.76`) |
| `dykstra60` | 20 (3) | `8.68`, `18.36` | 0 / 2 / 10 / 8 | `0.125`, `0.135` (`8.68`, `9.6`); `0.139` to `0.264` (`10.04` to `10.16`), `0.139` (`10.2`, uncertified), `0.429` (`10.28`, uncertified descent failure), `0.387`, `0.314`, `0.229`, `0.137` (`10.32` to `10.44`); `0.139` (`16.4`), `0.567` (`16.44`, uncertified descent failure), `0.156` to `0.115` (`16.48` to `16.68`); `0.131`, `0.110` (`17.92`, `18.36`) |
| `conic1e10` | 8 (3) | `8.68`, `16.88` | 0 / 2 / 5 / 1 | the base's eight answers to three digits (three of them uncertified at `1e-10`: the `10.08` and `10.16` answers at the 40-iteration cap, the `10.28` answer a descent failure) |

Three readings.

- **On the base's trajectory the cap and the `0.4` to `0.8` moves never
  meet.**  The base's 72 answers above `0.4` are 69 `infeasible` (34 of 34
  in `[5, 10)`, 24 of 25 in `[10, 15)`, 11 of 11 in `[15, 20)`) and 3
  declined by the margin (`0.569` at `t=0.08`, `0.507` at `0.16`, and
  `0.894` at `t=10.04`, the uncertified descent-failure iterate whose
  counterpart task 9's base accepted at `0.716` with improvement
  `0.175`).  So the rule as written already refuses every large answer on
  this trajectory; what the two caps at `t=8.68` and `9.64` did was move
  the state by `0.0325` and `0.052` in the frontier, and on the moved
  state the `t=10.04` iterate does not improve the declared objective.
  The gain over task 9's base (`0.1325` against `0.1501`; `(16, 20]`
  `0.133` against `0.196`) is that one declined move and the eight
  moderate caps after it; the `conic1e10` twin, accepting the same eight
  answers under a different certificate, is identical to `6.0e-5` (task 9:
  `0.089` at `t=19.9`), so the cap removes the certificate from the rule
  entirely.
- **The twins branch at `t=10.0` to `10.3`, as in task 9, on different
  answers of states `0.05` apart.**  The `etol1e3` twin's trajectory
  admits the `0.670` and `0.511` iterates at `t=10.08` and `10.12` that the
  base's declines or fails, and the cap applies them at `0.1` each; the
  `dykstra60` twin's admits `0.429` at `10.28` and, late, `0.567` at
  `16.44`.  The cap turns the `0.4` to `0.8` jumps into `0.1` steps, and
  the separation from the base at `t=12` is `0.116` (`etol1e3`; task 9
  `0.214`) and `0.136` (`dykstra60`; task 9 `0.066`), crossing `1e-1` at
  `11.78` and `10.30` (task 9: `11.23`, `12.91`); by `t=20` `0.626` and
  `1.41` (task 9: `2.27`, `0.534`).  The band is the same band with the
  twins' roles exchanged: under task 9's rule the `dykstra60` twin was the
  one that stayed out of the `infeasible` regime (1 of 125 in each of the
  last two quarters) and scored within 7 percent of the base; under `ball`
  it enters it in `[15, 20)` (38 of 125; base 11) and is the outlier
  (`0.207`, 56 percent above), while `etol1e3` improves from `0.223` to
  `0.163`.  Which twin enters the regime is set by the trajectory at
  `t=10`, not by the twin's setting, and no radius acts on it.
- **The cap binds 8 to 28 times in 500, in clusters.**  The `etol1e3`
  twin's 16 caps in `[15, 20)` include nine consecutive re-selections at
  `t=18.48` to `18.8` whose answers stay `0.14` to `0.33` from the lift
  while the lift walks toward them at `0.1` per re-selection; the median
  accepted move stays `0.020` to `0.027` per quarter (task 9: `0.026`,
  `0.044` in the last two), the 90th percentile `0.03` to `0.09` (task 9:
  `0.048`, `0.246`), and no applied move exceeds `0.1`.

## (b) `carried`: a freeze from `t=8.68`

Under the `carried` policy the first cap-bound answer, `0.133` at `t=8.68`,
is kept out and the lift stays; the next answer is `0.135`, then `0.135`,
`0.134`, `0.132`, `0.129` at `t=8.72` to `8.88`, each above the radius,
each kept out.  Of the 175 cap-bound re-selections of the base, 128 have
answers below `0.2` (minimum `0.105`, median `0.183`): past `t=8.7` the
completion's answer sits `0.1` to `0.2` from a lift that no longer follows
it, so the radius `0.1` is below the answer's typical distance and the
policy is a freeze, not a cap on outliers.  The base accepts 7 moves after
`t=8.68` (all below `0.1`; 5 in `[10, 15)`, 2 in `[15, 20)`), the
`etol1e3` twin none after `t=10`, `dykstra60` 4; `infeasible` decisions
116, 132, 114, 116.  The four runs score `0.216` / `0.252` / `0.236` /
`0.216` at `t=20`, 44 percent above task 9's base and 63 above the `ball`
base, 34 percent below the recorded row (`0.329`, frozen from `t=11.76`)
and 18 below the combined rule's (`0.264`); windows `(8, 12]` `0.229`,
`(12, 16]` `0.292`, `(16, 20]` `0.306` on the base (task 9: `0.161`,
`0.213`, `0.196`).  The band is the narrowest past `t=12` (6.9, 10.1, 16.3
percent) because a frozen lift has no answer branch to take; the
`conic1e10` twin never differs from the base above `6.6e-7` and the
`etol1e3` and `dykstra60` twins are `0.71` and `0.41` from it at `t=20`
through the repair alone.  The cones are the loosest of the three
measured-candidate sets (`-5.3e-4` / `-1.2e-2` on the base, attained at
`t=10.26` to `10.30`) and the envelope the lowest (`0.59` to `0.60`).  The
policy is not a candidate for the ledger row.

## (c) The certified screen

With `certified` candidates and the margin, base only: `0.027635` /
`0.046342` / `0.097956` / `0.123294` / `0.149326` over `t=0` to `4`, `8`,
`12`, `16`, `20`.  It refuses 65 of 500 (12 / 19 / 14 / 19 per quarter,
every uncertified return; the first at `t=0.08`, the `0.569` answer at the
40-iteration cap), declines 1 (the `t=0.04` wrong-face answer), has no
`infeasible` decision (the scope has no measured test) and accepts 434,
all certified, with two moves above `0.1` in the whole run (`0.290` at
`t=11.12` and `0.316` at `11.2`; task 9's base: 21, to `0.716`); its 14
answers above `0.4` are all uncertified and all refused, in runs of
consecutive re-selections (five at `0.755` to `0.758` over `t=9.36` to
`9.52`, three at `0.705` over `13.68` to `13.76`, `17.64`, `18.8`, `18.88`,
`20`).  It leaves task 9's base at `t=0.24` (a refused iterate the measured
rule accepted; frontier difference `0.0036`), crosses `1e-3` / `1e-2` /
`1e-1` at `0.59` / `2.1` / `4.75`, and is `0.49` from it at `t=8`.  The
cost is at `t=8`: `0.0463`, 20 percent above task 9's `0.0385` and 2
percent above the recorded row's `0.0456` (18 refusals over `t=0` to `8`,
10 of them by `t=4`; task 9's base accepts every uncertified return on
that horizon but the four it declines), and the `(4, 8]` window `0.0594`
against `0.0472`.  Past `t=12` it is the best single run of the
task at `t=16` (`0.1233`: 7 percent below the `ball` base's `0.1323`, 10
below task 9's) and ties task 9's base at `t=20` (`0.1493` against
`0.1501`; 13 percent above the `ball` base), with the tightest cones of
any run past `t=8` (`-1.9e-4` / `-4.4e-3`, the joint Gram minimum in
`t <= 8`), the envelope at most `0.710`, and the site-0 RMS `0.146`.  Its
band is not measured: the recorded rule's 17 to 28 percent was the
certificate's (task 6), and the margin removes only the `t=0.04` branch,
so the three twins are the measurement (item 1 below).

## Where each run leaves task 9's, and each twin its base

31-coordinate separation at the same accepted step (`task10_scores.py`):

| pair | first nonzero | first frontier difference above `1e-6` | crossing `1e-3` / `1e-2` / `1e-1` | at `t=12` / `16` / `20` | maximum, when |
|---|---|---|---|---|---|
| `ball` base from task 9 base | `8.6825` | `0.0325` at `8.68` (the first cap) | `8.695` / `10.52` / `13.65` | `0.034` / `0.300` / `0.858` | `0.858` at `20` |
| `ball` `etol1e3` from task 9 `etol1e3` | `8.6825` | `0.0308` at `8.68` | `8.695` / `9.255` / `11.695` | `0.129` / `0.531` / `1.93` | `1.93` at `20` |
| `ball` `dykstra60` from task 9 `dykstra60` | `8.6825` | `0.025` at `8.68` | `8.6975` / `9.3625` / `11.635` | `0.137` / `0.568` / `1.25` | `1.26` at `19.78` |
| `carried` base from task 9 base | `8.6825` (`0.0019`) | `0.133` at `8.68` (the answer kept out) | `8.6825` / `8.8125` / `9.11` | `0.721` / `0.49` / `0.907` | `0.935` at `17.31` |
| `carried` base from `ball` base | `8.6825` (`0.0019`) | `0.1` at `8.68` | `8.6825` / `8.82` / `9.115` | `0.705` / `0.588` / `1.45` | `1.45` at `20` |
| certified screen from task 9 base | `0.2425` | `0.0036` at `0.24` | `0.59` / `2.105` / `4.75` | `0.417` / `0.385` / `1.32` (`0.036` at `4`, `0.489` at `8`) | `1.32` at `20` |
| `ball` `etol1e3` from `ball` base | `0.0075` | `0.0064` at `0.16` | `2.1075` / `5.83` / `11.7825` [task 9 `11.2275`] | `0.116` / `0.531` / `0.626` [`0.214` / `0.755` / `2.27`] | `0.636` at `19.82` |
| `ball` `dykstra60` from `ball` base | `0.0625` | `0.0064` at `0.16` | `2.105` / `5.76` / `10.295` [`12.905`] | `0.136` / `0.826` / `1.41` [`0.066` / `0.347` / `0.534`] | `1.54` at `19.05` |
| `ball` `conic1e10` from `ball` base | `0.245` (`8.7e-19`) | `2.1e-6` at `10.36` | never / never / never [`14.2` / `18.6` / never] | `6.3e-7` / `2.0e-5` / `6.0e-5` [`1.2e-4` / `0.0030` / `0.0875`] | `6.0e-5` at `19.99` |
| `carried` `etol1e3` from `carried` base | `0.0075` | `0.0064` at `0.16` | `2.1075` / `5.83` / `10.0075` | `0.142` / `0.401` / `0.71` | `0.809` at `19.61` |
| `carried` `dykstra60` from `carried` base | `0.0625` | `0.0064` at `0.16` | `2.105` / `5.76` / `14.2225` | `0.082` / `0.152` / `0.411` | `0.44` at `19.07` |
| `carried` `conic1e10` from `carried` base | `0.245` (`8.7e-19`) | never | never / never / never | `7.0e-8` / `1.2e-7` / `6.6e-7` | `6.6e-7` at `20` |

Window means of the `ball` twins' separation from the `ball` base, `(8,
12]` / `(12, 16]` / `(16, 20]`: `etol1e3` `0.058` / `0.345` / `0.534`
(task 9 `0.066` / `0.49` / `1.49`), `dykstra60` `0.080` / `0.488` / `1.37`
(`0.046` / `0.23` / `0.43`).

## Re-selections of 500 over `t=0` to `20`

| run | accepted (of which uncertified; of which cap-bound) / declined by the margin / `infeasible` / `refused` / cap-bound kept out (uncertified) | solver refusals | accepted per quarter `[0,5)` / `[5,10)` / `[10,15)` / `[15,20)` (`infeasible`; cap-bound) | largest applied move |
|---|---|---|---|---|
| `ball` base | 426 (44; 8) / 5 / 69 / 0 / --- | 117 | 120 / 91 / 100 / 114 (0, 34, 24, 11; 0, 2, 5, 1) | `0.1` |
| `ball` `etol1e3` | 427 (52; 28) / 5 / 68 / 0 / --- | 122 | 120 / 92 / 110 / 104 (0, 33, 14, 21; 0, 2, 10, 16) | `0.1` |
| `ball` `dykstra60` | 413 (44; 20) / 4 / 83 / 0 / --- | 130 | 120 / 92 / 114 / 87 (0, 33, 11, 38; 0, 2, 10, 8) | `0.1` |
| `ball` `conic1e10` | 426 (109; 8) / 5 / 69 / 0 / --- | 183 | 120 / 91 / 100 / 114 (0, 34, 24, 11; 0, 2, 5, 1) | `0.1` |
| `carried` base | 205 (15) / 4 / 116 / 0 / 175 (14) | 148 | 120 / 78 / 5 / 2 (0, 40, 47, 29; 0, 7, 73, 94) | `0.0998` |
| `carried` `etol1e3` | 198 (18) / 4 / 132 / 0 / 166 (17) | 168 | 120 / 78 / 0 / 0 (0, 41, 51, 40; 0, 6, 74, 85) | `0.0387` |
| `carried` `dykstra60` | 202 (13) / 4 / 114 / 0 / 180 (16) | 146 | 120 / 78 / 4 / 0 (0, 40, 51, 23; 0, 7, 70, 102) | `0.0969` |
| `carried` `conic1e10` | 205 (46) / 4 / 116 / 0 / 175 (30) | 196 | as base | `0.0998` |
| certified screen | 434 (0) / 1 / 0 / 65 / --- | 65 | 111 / 106 / 111 / 106 (refused 12, 19, 14, 19) | `0.316` |
| task 9 base | 377 (39) / 4 / 119 / 0 / --- | 161 | 120 / 91 / 96 / 69 (0, 34, 29, 56) | `0.716` |

The `infeasible` regime begins in `[5, 10)` at 33 to 34 of 125 in every
measured-candidate run whatever the rule, and its later course is the
trajectory's: 24 / 11 (`ball` base), 14 / 21 (`ball` `etol1e3`), 11 / 38
(`ball` `dykstra60`), 47 / 29 (`carried` base), 29 / 56 (task 9 base).

## The 15-key ledger row

For the plan file's cheap-projection row (15 entrance keys, `N=16`, strong
coupling, cutoff 16, exact-correlated preparation, `h=0.0025`, Dykstra 30
on both targets at stage cadence, the declared proximal log-det completion
re-solved every 16 steps warm-started on the carried frontier, the
energy-neutral repair in `drive_power` mode, and the lift re-selection
rule margin `1e-3`, `measured` candidates, no anchor, trust radius `0.1`
under the `ball` policy; every band the base and its three one-setting
twins), next to task 9's row (the same rule without the radius):

| horizon `t=0` to `T` | score, base (band of the four; spread; three of four) | task 9's row (band; spread) | Eq. (22) internal-energy RMS (band) | recorded rule's band (task 6) | record, both cones |
|---:|---|---|---|---|---|
| 4 | `0.0272` (`0.0272` to `0.0272`; 0.1 %) | `0.0272` (0.1 %) | `0.0331` (`0.0331` to `0.0333`) | `0.0237` to `0.0280` | --- |
| 8 | `0.0385` (`0.0385` to `0.0386`; 0.2 %) | `0.0385` (0.2 %) | `0.0382` (`0.0382` to `0.0384`) | `0.0404` to `0.0456` | --- |
| 12 | `0.0989` (`0.0989` to `0.1043`; 5.5 %; 4.4 %) | `0.0982` (`0.0982` to `0.1014`; 3.2 %) | `0.0397` (`0.0397` to `0.0399`) | `0.108` to `0.131` | --- |
| 16 | `0.132` (`0.132` to `0.158`; 19.3 %; 14.3 %) | `0.136` (`0.136` to `0.162`; 18.8 %) | `0.0404` (`0.0404` to `0.0406`) | `0.202` to `0.258` | --- |
| 20 | `0.133` (`0.133` to `0.207`; 56.2 %; 23.0 %) | `0.150` (`0.150` to `0.223`; 48.8 %) | `0.0409` (`0.0409` to `0.0411`) | `0.260` to `0.329` | `0.1286` (energy `0.00094`; joint Gram `+1.4e-5`) |

with the joint Gram minimum over `t=0` to `20` `-7.9e-3` (band `-5.4e-3`
to `-7.9e-3`), the scaled extension minimum `-2.6e-4` (`-2.2e-4` to
`-2.6e-4`), the frontier envelope at most `0.748` (`0.748` to `0.910`; the
box never reached), the site-0 occupation RMS `0.1245` (`0.1245` to
`0.1705`), walls `762` to `847 s` for 8,000 steps under `ram_guard
--limit-mb 3000`.  What the row can claim: the second digit to `t=12`
under either rule (all eight runs of the two rules within `0.0982` to
`0.1043`); at `t=16` and `t=20` the first digit only, `0.13` to `0.16` and
`0.13` to `0.21` under this rule (`0.13` to `0.22` across the eight), with
the base at `1.03` times the both-cones line and the band set by which
twin's trajectory enters the `infeasible` regime after `t=10`, which no
setting of the rule controls.  Whether the plan-file row moves from
task 9's rule to this one is an adoption call; the plan file
(`agent_guidance/paper-v/paper-v-state-and-plan-20260914.md`) was not
edited here, as in task 9 (shared with the projective-route agent; this
route does not commit).

## Provenance and gates

- Gates: `paper_5/tests/test_cheap_projection_route.py` 44 passed in
  `23.95 s` before the chain (`test_cheap_projection_route.log`) and
  `25.20 s` after the scoring (`test_cheap_projection_route_after.log`;
  `cd paper_5 && PYTHONPATH=src python -m pytest -q
  tests/test_cheap_projection_route.py`).  The full `paper_5` suite was
  not re-run: no module, test or default changed since task 8's pass (499
  passed) of the same module contents (`a81b7ecb`); `HEAD` is `3ea96121`
  (task 9's commit, `19:01` on 2026-09-16) throughout, the chain having
  started at `19:06`.
- Scripts, all under
  `output/local_runs/paper_v_cheap_projection_task10_trust_t20_20260916_v1/`:
  `run_one.zsh` (task 7's, output directory changed), `chain.zsh` (the
  nine rollouts in order, every twin and rule as flags), `smoke_check.py`,
  `analyse.zsh` (runs the three scoring scripts; `analyse.log`),
  `task10_scores.py` (`task10_scores.json`, `.md`, `.log`: task 9's
  scorer with the `ball`, `carried` and certified sets, the cap-bound and
  `refused` decisions counted and timed per horizon and per quarter, and
  each new run's separation from task 9's same twin), `windows.py`
  (`windows.json`, `.md`, `.log`), `cap_events.py` (`cap_events.json`,
  `.log`: every cap-bound or refused re-selection with its answer, applied
  move, certificate, face and objective improvement; the accepted moves'
  quartiles per quarter; the envelope's arrivals), `cap_summary.py`
  (`cap_summary.json`, `.log`: the compact reading above and the times of
  the physicality minima), `large_answers.py` (`large_answers.log`: the
  decision each answer above `0.4` received), `test_gate.zsh`, the run
  logs, `chain.log`, `chain_stdout.log`, the two test logs.
- Artifacts, under `output/local_runs/`:
  `paper_v_cheap_projection_task10_{ball,carried}_{base,etol1e3,dykstra60,conic1e10}_t20_20260916_v1`,
  `paper_v_cheap_projection_task10_certified_base_t20_20260916_v1`, each
  with `repairs.jsonl` (`15 MB`); the smoke
  `paper_v_cheap_projection_task10_smoke_ball_t005_20260916_v1`.  Walls
  and peaks as in the opening paragraph (`autonomous_wall_seconds`,
  `peak_resident_bytes`); every rollout under `pipelines/shell/ram_guard.py
  --limit-mb 3000`, one at a time, `19:07` to `21:12` on 2026-09-16.

## What to try next, in order

1. The certified screen's three one-setting twins to `t=20` (three
   rollouts of about 12.5 minutes): its base is the best run of the task
   at `t=16` and ties task 9's at `t=20` with two moves above `0.1` and no
   `infeasible` decision, and its band is unmeasured.  If the band past
   `t=12` is below task 9's, the screen replaces the measured acceptance
   in the ledger row; if it is the recorded rule's 17 to 28 percent, the
   certificate is the branch and the measured rule stands.
2. The 27-key `N=16` drive-power chain under the rule of the ledger row
   (task 8's item 2, unchanged; `ball` if 1 does not displace it).
3. The band past `t=12` is the `infeasible` regime's (33 to 34 of 125
   re-selections in `[5, 10)` in every run, then 11 to 56 per quarter
   depending on the trajectory), and no field of the lift rule acts on
   the state's lower moments; the repair does (Dykstra 60 kept task 9's
   twin out of it and did not keep the `ball` twin out).  A measurement of
   that regime against the Dykstra iteration count on the base (60, 120)
   is the next instrument after 1 and 2, and a default question if it
   moves the band.

## Structured summary of task 10

| item | value |
|---|---|
| what was run | nine `t=0` to `20` rollouts (8,000 steps) of the recorded `N=16` drive-power configuration, one at a time under `ram_guard --limit-mb 3000`: task 9's rule (margin `1e-3`, `measured`, no anchor) with trust radius `0.1` under `ball` and under `carried`, each on the base and the three one-setting twins; the certified screen (margin `1e-3`, `certified`, no anchor, no radius), base only; walls `761.6` to `879.5 s`, peaks `778` to `872 MiB`; all reached `t=20`; the launching instance was lost to a network outage after the chain, and the scoring and this section were written from the artifacts with no rollout repeated; no module edited, 44 tests pass, nothing committed |
| scores, `ball`, base / `etol1e3` / `dykstra60` / `conic1e10` | `t=4`, `8` task 9's four (the cap first binds at `t=8.68` in every run); `t=12` `0.098867` / `0.104307` / `0.103216` / `0.098867`; `t=16` `0.132318` / `0.151238` / `0.157814` / `0.132318`; `t=20` `0.132542` / `0.163027` / `0.206990` / `0.132542` |
| scores, `carried` | `t=12` `0.135734` / `0.145165` / `0.141612` / `0.135734`; `t=16` `0.187509` / `0.206528` / `0.196694` / `0.187509`; `t=20` `0.216450` / `0.251668` / `0.236386` / `0.216450` |
| scores, certified screen (base) | `0.027635` / `0.046342` / `0.097956` / `0.123294` / `0.149326` at `T=4`, `8`, `12`, `16`, `20` |
| spreads (task 9's rule) | `ball` 0.1, 0.2, 5.5, 19.3, 56.2 percent (three of four 4.4, 14.3, 23.0); `carried` 0.1, 0.2, 6.9, 10.1, 16.3; task 9 0.1, 0.2, 3.2, 18.8, 48.8; across the eight runs of both re-selecting rules 6.2, 22.4, 68.6 at `t=12`, `16`, `20` |
| do the `t=16` and `t=20` bands close below 10 percent | no, under either policy: `ball` 19.3 and 56.2 percent at `0.132` to `0.158` and `0.133` to `0.207`; `carried` 10.1 and 16.3 percent but at `0.188` to `0.207` and `0.216` to `0.252` (37 to 63 percent above the `ball` base), a frozen lift from `t=8.68` (7 accepted of 283 re-selections after) |
| what the cap buys | the `ball` base `0.1325` at `t=20`: 12 percent below task 9's `0.1501`, 60 below the recorded `0.3287`, `1.03` times the both-cones `0.1286`; `(16, 20]` window `0.133` against `0.196`; site-0 RMS `0.1245` against `0.205`; `conic1e10` identical within `6.0e-5`; envelope never at the box (`0.748`); cones unchanged (`-2.6e-4` / `-7.9e-3`, at `t=8.37`) |
| first-separation times | `ball` from task 9, every twin: `t=8.6825`, the `t=8.68` cap (answer `0.125` to `0.133` scaled to `0.1`); `1e-1` crossed at `13.65` (base), `11.7` (`etol1e3`), `11.6` (`dykstra60`); `carried` from task 9: `8.6825`, `1e-1` at `9.11`; certified screen from task 9: `t=0.2425` (a refused iterate at `0.24`), `1e-1` at `4.75`; `ball` twins from the `ball` base: `1e-1` at `11.78` (`etol1e3`; task 9 `11.23`), `10.30` (`dykstra60`; `12.91`), never (`conic1e10`, `6.0e-5` at most) |
| capped, kept-out and refused moves, and when | `ball`: 8 / 28 / 20 / 8 cap-bound (1 / 8 / 3 / 3 uncertified), first `t=8.68`, last `16.88` / `19.76` / `18.36` / `16.88`, per quarter 0, 2, 5, 1 (base), 0, 2, 10, 16, 0, 2, 10, 8; the base's capped answers `0.13` to `0.25`, its 72 answers above `0.4` all `infeasible` (69) or declined (3, one the `0.894` at `t=10.04`); the twins cap `0.670`, `0.511` (`etol1e3`, `10.08`, `10.12`) and `0.429`, `0.567` (`dykstra60`, `10.28`, `16.44`); `carried`: 175 / 166 / 180 / 175 kept out, from `t=8.68` at every re-selection but 7 / 0 / 4 / 7, 128 of the base's on answers below `0.2`; certified screen: 65 refused (12 / 19 / 14 / 19 per quarter; 14 answers above `0.4`, all refused), 2 accepted moves above `0.1` (`0.290`, `0.316` at `t=11.12`, `11.2`) |
| internal energy | `ball` and `carried` equal task 9's at every horizon (`0.0331` / `0.0382` / `0.0397` / `0.0404` / `0.0409`, band 0.5 percent); certified `0.0332` / `0.0384` / `0.0400` / `0.0408` / `0.0413` (reference `0.0430` above exact) |
| physicality over `t <= 20` and wall | `ball` extension `-2.6e-4` / `-2.5e-4` / `-2.2e-4` / `-2.6e-4`, joint Gram `-7.9e-3` / `-7.8e-3` / `-5.4e-3` / `-7.9e-3`, envelope `0.748` / `0.819` / `0.910` / `0.748`; `carried` `-5.3e-4` / `-5.7e-4` / `-2.7e-4` / `-5.3e-4`, `-1.2e-2` / `-1.3e-2` / `-6.2e-3` / `-1.2e-2`, envelope `0.59` to `0.60`; certified `-1.9e-4`, `-4.4e-3`, `0.710`; walls `ball` `817.8` / `825.5` / `847.1` / `761.6 s`, `carried` `879.5` / `838.1` / `860.1` / `832.3 s`, certified `743.4 s` |
| ledger row (`ball` rule) | `0.0272` (0.1 %), `0.0385` (0.2 %), `0.0989` (`0.0989` to `0.1043`; 5.5 %), `0.132` (`0.132` to `0.158`; 19.3 %; three within 14.3), `0.133` (`0.133` to `0.207`; 56.2 %; three within 23.0) at `t=4`, `8`, `12`, `16`, `20`; second digit resolved to `t=12`, first digit past it; plan file not edited |
| next | the certified screen's three twins to `t=20`; the 27-key chain under the ledger row's rule; the `infeasible` regime against the Dykstra count |
| files changed | this report, the worklog block, and the task-10 scripts and outputs under `output/local_runs/`; no module, test or default; nothing committed |

# Task 11: the certified-only rule's band, and the rule combined with the ball radius -- the certificate is the branch: the four certified-only runs spread 132 percent at `t=20`, two twins freeze from `t=11.96` and `t=13.68`, the `conic1e10` twin is the recorded rule's twin bitwise, and the radius on the certified base changes `0.149` to `0.152` (2026-09-17)

Four rollouts of the recorded `N=16` drive-power configuration to `t=20`
(8,000 steps), one at a time under `ram_guard --limit-mb 3000`
(`chain.zsh`, `20:27` to `21:25` on 2026-09-17): the certified-only rule
(margin `1e-3`, `certified` candidates, no anchor, no trust radius: an
uncertified return is `refused` outright and the carried lift kept, a
certified answer is accepted when its declared-objective improvement
exceeds the margin) on the three one-setting twins (`energy_tolerance`
`1e-3`; Dykstra 60; re-selection `conic_tolerance` `1e-10`), completing
the band whose base is task 10's certified screen; and the same rule with
the trust radius `0.1` under the `ball` policy (a certified answer whose
standardized-frontier move exceeds the radius is scaled to it on the
segment toward the answer, decision `cap_bound_ball`; an uncertified
return is refused before the cap is consulted), base only.  Every run is
`run_one.zsh` (task 7's script, output directory changed) plus the twin's
one setting plus the rule's flags on its command line; no module was
edited (the fields are task 7's `LiftReselectionRule`, committed at
`a81b7ecb`; `HEAD` is `f971a284` throughout, the projective route's task-8 commit of `20:26`, which touches no owned module); the 44 tests of
`test_cheap_projection_route.py` pass before the chain (`25.08 s`) and
after it (`24.66 s`); nothing is committed.  Walls `914.8`, `869.8`,
`840.7 s` (certified-only `etol1e3` / `dykstra60` / `conic1e10`) and
`802.4 s` (certified plus ball, base), all against the `1,140 s` budget
and all at `t=20` with 8,001 samples; peaks `756`, `783`, `780`, `748 MiB`.
The smoke run under the certified-plus-ball flags (`t=0.05`, `1.8 s`, peak
`228 MiB`) equals task 10's certified screen in every array over the 21
common samples and records the rule with `candidates` `certified`,
`trust_radius` `0.1`, `trust_policy` `ball`.  Conventions are tasks 7 to
10's (score, internal energy, site-0 occupation RMS, physicality minima
over `t <= T`, separation, spread, move, answer, solver refusal, cap-bound);
"three of four" is the tightest band of three members.  The comparison
rows are task 10's certified base (the fourth member of this band), task
9's four, task 10's eight, the recorded rule's four and the combined rule's
four, rescored by one script (`task11_scores.py`, task 10's with the two
new sets; every task-9 and task-10 number is reproduced to every digit).

## Negative result first

The certified-only rule has the widest band of any rule on record, and the
certificate is what sets it.  The four scores over `t=0` to `20` are
`0.149326` (base) / `0.203554` (`etol1e3`) / `0.345743` (`dykstra60`) /
`0.300340` (`conic1e10`): spread 131.5 percent (task 9's measured rule
48.8, the `ball` form 56.2, the recorded rule 26.4); 135.3 percent at
`t=16` (`0.123294` to `0.290131`), 90.4 at `t=12` (`0.095070` to
`0.181008`), 43.7 at `t=8` (`0.040450` to `0.058107`) and 1.7 at `t=4`.
No three of the four are within 70 percent at `t=20` or 85 at `t=16`.  The
twins leave the base at the first refusal flip, `t=0.6` to `0.8` (a
re-selection the base accepts at a move of `0.008` and the twin refuses,
on states `5e-16` to `3e-5` apart), cross `1e-1` at `t=3.9` to `6.1` (the
recorded rule's `3.1` to `7.0`; task 9's `11.2`, `12.9`, never) and end
`0.77` to `1.63` from it.  Two twins freeze: the `dykstra60` twin's
completion refuses 202 consecutive re-selections from `t=11.96` (its last
accepted lift at `t=11.92`) and the `conic1e10` twin's 159 from `t=13.68`,
so both are the frozen-lift route with the Dykstra repair over their last
6 to 8 time units, as the recorded base was from `t=11.76`; the
`dykstra60` twin's frontier envelope reaches `1.018` at `t=18.78`, above
the box.  The rule is the recorded rule plus the margin: the `conic1e10`
twin, whose margin never declines, is the recorded rule's `conic1e10` twin
(task 6) bitwise in every sample, and the other three leave the recorded
rule's same twin at `t=0.0425`, the sample after the `t=0.04`
re-selection, where the margin declines the certified wrong-face answer
(frontier move `0.30` to `0.33`) that the recorded rule accepted.  That one
decline moves the `t=20` score from `0.329` to `0.149` on the base, from
`0.326` to `0.204` on `etol1e3`, and from `0.260` to `0.346` on
`dykstra60`.

The radius on the certified base changes nothing that matters: certified
plus `ball` scores `0.027635` / `0.046342` / `0.097938` / `0.124542` /
`0.151644` at `T=4`, `8`, `12`, `16`, `20`, equal to the certified screen
to `t=11.12` (the first cap, the certified `0.290` answer scaled to `0.1`)
and 1.0 and 1.6 percent above it at `t=16` and `t=20`, 14 percent above
the `ball` base (`0.132542`); the cap binds seven times, all on certified
answers of `0.10` to `0.29`, and the refusals in `[15, 20)` rise from 19
to 35.  Its band is not measured and would be the certificate's.  What
task 10's item 1 asked is answered: the band past `t=12` is not below
task 9's, it is the recorded rule's certificate branch made wider by the
margin's `t=0.04` decline, and the measured acceptance stands in the
ledger row.

## The rows side by side

Scores (all-coordinate scalar RMS over `t=0` to `T`), base / `etol1e3` /
`dykstra60` / `conic1e10`, with the spread of the four (three of four in
brackets); the certified-plus-ball run is base only:

| `T` | measured, no anchor (task 9) | spread | `ball`, radius `0.1` (task 10) | spread | certified-only (task 10 base; this task's twins) | spread [three of four] | certified plus `ball`, base | recorded rule (task 6) | spread |
|---:|---|---|---|---|---|---|---|---|---|
| 4 | `0.027191` / `0.027215` / `0.027214` / `0.027191` | 0.1 % | the same four | 0.1 % | `0.027635` / `0.028102` / `0.027735` / `0.027953` | 1.7 % [1.2] | `0.027635` | `0.023744` / `0.024240` / `0.026837` / `0.027954` | 17.7 % |
| 8 | `0.038538` / `0.038617` / `0.038605` / `0.038538` | 0.2 % | the same four | 0.2 % | `0.046342` / `0.051402` / `0.058107` / `0.040450` | 43.7 % [25.4] | `0.046342` | `0.045571` / `0.044530` / `0.040677` / `0.040450` | 12.7 % |
| 12 | `0.098187` / `0.101367` / `0.100824` / `0.098186` | 3.2 % | `0.098867` / `0.104307` / `0.103216` / `0.098867` | 5.5 % | `0.097956` / `0.095070` / `0.181008` / `0.108195` | 90.4 % [13.8] | `0.097938` | `0.125139` / `0.130946` / `0.114863` / `0.108195` | 21.0 % |
| 16 | `0.136367` / `0.161977` / `0.146397` / `0.136288` | 18.8 % | `0.132318` / `0.151238` / `0.157814` / `0.132318` | 19.3 % | `0.123294` / `0.144537` / `0.290131` / `0.228471` | 135.3 % [85.3] | `0.124542` | `0.258499` / `0.256609` / `0.201866` / `0.228471` | 28.1 % |
| 20 | `0.150102` / `0.223420` / `0.160803` / `0.150285` | 48.8 % | `0.132542` / `0.163027` / `0.206990` / `0.132542` | 56.2 % | `0.149326` / `0.203554` / `0.345743` / `0.300340` | 131.5 % [69.9] | `0.151644` | `0.328674` / `0.325970` / `0.260123` / `0.300340` | 26.4 % |

(the combined rule's row and the `carried` policy's row are tasks 9 and
10's, unchanged: spreads 0.1, 4.2, 19.2, 21.6, 7.4 and 0.1, 0.2, 6.9,
10.1, 16.3 percent.)  The certified-only `conic1e10` column equals the
recorded rule's `conic1e10` column at every horizon because the runs are
one trajectory.  Across the thirteen runs that keep the rule's margin
(task 9's four, task 10's `ball` four, the certified-only four and the
certified-plus-ball base) the scores span `0.0951` to `0.1810` at `t=12`,
`0.123` to `0.290` at `t=16` and `0.133` to `0.346` at `t=20`; across the
eight measured-candidate runs alone `0.0982` to `0.1043`, `0.132` to
`0.162` and `0.133` to `0.223`.

Internal-energy sample RMS (Eq. (22)) over `t=0` to `T`, certified-only
base / `etol1e3` / `dykstra60` / `conic1e10`, and the declared drive-power
reference minus the exact post-pulse energy (the internal-energy error on
every window past the pulse, since the repair holds the energy at the
reference to `2e-15`):

| `T` | certified-only | measured, no anchor (task 9) |
|---:|---|---|
| 4 | `0.03323` / `0.02904` / `0.02948` / `0.02864` | `0.03314` / `0.03332` / `0.03332` / `0.03314` |
| 8 | `0.03845` / `0.03404` / `0.03427` / `0.03383` | `0.03816` / `0.03835` / `0.03836` / `0.03816` |
| 12 | `0.04004` / `0.03555` / `0.03572` / `0.03539` | `0.03969` / `0.03989` / `0.03990` / `0.03969` |
| 16 | `0.04081` / `0.03628` / `0.03643` / `0.03614` | `0.04043` / `0.04064` / `0.04064` / `0.04043` |
| 20 | `0.04127` / `0.03672` / `0.03684` / `0.03659` | `0.04087` / `0.04108` / `0.04109` / `0.04087` |
| reference minus exact | `0.04304` / `0.03840` / `0.03847` / `0.03832` | `0.04259` / `0.04280` / `0.04281` / `0.04259` |

The certified-only twins' references lie `0.0384` above the exact energy
where the base's lies `0.0430`, because their trajectories have left the
base's by `0.05` to `0.10` in the 31 coordinates at `t=4`, when the
reference is set; the internal-energy spread of this set is 12.8 percent
at `t=20` against 0.5 under the measured rule.  The certified-plus-ball
base's column equals the certified base's (reference `0.04304`).  Site-0
occupation RMS over `t=0` to `20`: certified-only `0.1464` / `0.2073` /
`0.1951` / `0.1832`, certified plus ball `0.1495` (task 9 `0.158` to
`0.201`; `ball` `0.1245` to `0.1705`; the both-cones record `0.205`).

Score per four-unit window `(a, b]` of the horizon (`windows.py`):

| window | certified base | certified `etol1e3` | certified `dykstra60` | certified `conic1e10` | certified plus `ball` | `ball` base (task 10) | task 9 base | recorded base |
|---|---|---|---|---|---|---|---|---|
| `(0, 4]` | `0.0276` | `0.0281` | `0.0277` | `0.0280` | `0.0276` | `0.0272` | `0.0272` | `0.0237` |
| `(4, 8]` | `0.0594` | `0.0671` | `0.0774` | `0.0499` | `0.0594` | `0.0472` | `0.0472` | `0.0599` |
| `(8, 12]` | `0.1565` | `0.1478` | `0.3026` | `0.1785` | `0.1565` | `0.1624` | `0.1611` | `0.2070` |
| `(12, 16]` | `0.1790` | `0.2376` | `0.4883` | `0.4168` | `0.1824` | `0.2018` | `0.2132` | `0.4694` |
| `(16, 20]` | `0.2251` | `0.3516` | `0.5109` | `0.4922` | `0.2301` | `0.1334` | `0.1956` | `0.5224` |

The frozen twins' last two windows (`0.42` to `0.51`) are the recorded
base's (`0.47`, `0.52`): a frozen lift decays at the same rate whatever
rule froze it.

Physicality over `t <= T` (scaled 24-row extension minimum; unshifted
retained joint Gram minimum), certified-only base / `etol1e3` /
`dykstra60` / `conic1e10`, the time at which the `t <= 20` minimum is
attained, and the largest standardized frontier envelope ratio:

| set | extension minimum, `t <= 8` / `<= 12` / `<= 20` (attained at) | joint Gram minimum, `t <= 8` / `<= 12` / `<= 20` (attained at) | envelope maximum |
|---|---|---|---|
| certified-only | `-1.7e-4` / `-1.7e-4` / `-1.9e-4` (`20.0`); `-1.6e-4` / `-3.1e-4` / `-6.3e-4` (`18.27`); `-9.9e-5` / `-2.1e-4` / `-1.0e-3` (`20.0`); `-1.6e-4` / `-3.8e-4` / `-1.6e-3` (`19.80`) | `-4.4e-3` / `-4.4e-3` / `-4.4e-3` (`6.92`); `-4.2e-3` / `-9.3e-3` / `-1.5e-2` (`18.27`); `-2.4e-3` / `-5.0e-3` / `-2.3e-2` (`20.0`); `-4.8e-3` / `-9.2e-3` / `-3.6e-2` (`19.82`) | `0.710` / `0.851` / `1.018` (first at `1.0` `t=18.78`) / `0.893` |
| certified plus `ball`, base | `-1.7e-4` / `-1.7e-4` / `-2.1e-4` (`18.89`) | `-4.4e-3` / `-4.4e-3` / `-5.0e-3` (`18.89`) | `0.616` |
| task 9 (for reference) | `-1.1e-4` / `-2.6e-4` / `-2.6e-4`; `-1.1e-4` / `-2.5e-4` / `-6.6e-4`; `-4.8e-5` / `-1.2e-4` / `-1.2e-4`; as base | `-3.3e-3` / `-7.9e-3` / `-7.9e-3`; `-3.2e-3` / `-7.8e-3` / `-1.6e-2`; `-1.4e-3` / `-3.7e-3` / `-3.7e-3`; as base | `1.0` in every run |
| recorded rule (task 6) | `-8.6e-5` / `-6.0e-4` / `-2.3e-3`; `-1.7e-4` / `-6.8e-4` / `-2.0e-3`; `-3.4e-5` / `-1.4e-4` / `-5.9e-4`; `-1.6e-4` / `-3.8e-4` / `-1.6e-3` | `-2.1e-3` / `-1.4e-2` / `-5.3e-2`; `-4.7e-3` / `-1.6e-2` / `-4.6e-2`; `-8.0e-4` / `-3.4e-3` / `-1.4e-2`; `-4.8e-3` / `-9.2e-3` / `-3.6e-2` | `0.84` / `0.92` / `0.72` / `0.89` |

The certified base's cones are the tightest of any run past `t=8` (task
10), and its twins' are the recorded rule's: the frozen twins' joint Gram
minima (`-2.3e-2`, `-3.6e-2`) are attained in the frozen stretch, at
`t=19.8` to `20`, as the recorded base's `-5.3e-2` was.

## (a) Where each certified-only twin leaves the base, and what moves it

31-coordinate Euclidean separation from the certified base at the same
accepted step (`task11_scores.py`, `refusals.py`), with the crossings under
the measured rule (task 9) and the recorded rule (task 6) in brackets:

| twin | first nonzero | first carried-frontier difference above `1e-6` | first stage-record difference | first crossing of `1e-3` / `1e-2` / `1e-1` [task 9] [recorded] | at `t=4` / `8` / `12` / `16` / `20` | window mean `(8, 12]` / `(12, 16]` / `(16, 20]` | maximum, when |
|---|---|---|---|---|---|---|---|
| `etol1e3` | `t=0.0075` (`5e-16`) | `0.0064` at `t=0.16` | `t=0.08`: the base refuses its uncertified `0.569` answer, the twin's certified `0.437` answer is declined by the margin | `0.6375` / `1.1525` / `5.175` [`2.1075` / `5.83` / `11.2275`] [`0.1475` / `0.7525` / `7.01`] | `0.060` / `0.272` / `0.463` / `0.706` / `0.772` | `0.43` / `0.49` / `0.79` | `0.973` at `t=19.2` |
| `dykstra60` | `t=0.0625` (`4.8e-15`) | `0.0064` at `t=0.16` | `t=0.16`: the base refuses its uncertified `0.507` answer, the twin accepts a certified `0.0064` move | `0.81` / `1.305` / `6.1375` [`2.105` / `5.76` / `12.905`] [`0.145` / `0.36` / `4.0025`] | `0.052` / `0.260` / `1.35` / `1.92` / `1.63` | `0.84` / `1.83` / `1.83` | `2.00` at `t=15.37` |
| `conic1e10` | `t=0.6025` (`3.1e-5`) | `0.0089` at `t=0.6` | `t=0.04`: the base declines the certified `0.334` wrong-face answer, the twin's `0.564` answer is uncertified at `1e-10` and refused | `0.6375` / `1.1` / `3.9225` [`14.205` / `18.6425` / never] [`0.125` / `0.48` / `3.085`] | `0.101` / `0.382` / `0.71` / `1.82` / `1.60` | `0.41` / `1.41` / `1.74` | `1.84` at `t=16.22` |

Three readings.

- **The first separation is a refusal flip at `t=0.6` to `0.8` in every
  twin, on a state difference of round-off size.**  The `etol1e3` and
  `conic1e10` twins cross `1e-3` at `t=0.6375`, the sample after the
  `t=0.6` re-selection, where the base's completion certifies a `0.0089`
  move and the twins' refuses on states `5e-16` (`etol1e3`) and `3.1e-5`
  (`conic1e10`) apart; the `dykstra60` twin crosses at `t=0.81` after the
  `t=0.8` re-selection (base `0.0084` accepted, twin refused; states
  `4.8e-15` apart at first difference).  Under the measured rule the same
  twins stayed within `1e-3` to `t=2.1` and, for `conic1e10`, to
  `t=14.2`, because that rule takes the iterate whatever the certificate
  says; under this rule every uncertified return is a lost re-selection
  and, since the certificate flips on round-off (task 6), each twin's set
  of lost re-selections is its own from `t=0.6` on.  The `1e-1` crossings
  (`5.2`, `6.1`, `3.9`) lie inside the recorded rule's (`3.1` to `7.0`)
  and four to eight time units before task 9's.
- **The band past `t=8` is the refusal count's.**  Refusals per quarter
  `[0,5)` / `[5,10)` / `[10,15)` / `[15,20)`: base 12 / 19 / 14 / 19,
  `etol1e3` 17 / 25 / 23 / 60, `dykstra60` 13 / 29 / 106 / 125,
  `conic1e10` 26 / 34 / 81 / 125.  The `dykstra60` twin accepts 19
  re-selections in `[10, 15)` (ten of them moves of `0.11` to `0.37` at
  `t=11.28` to `11.92`, certified, improvements `0.66` to `1.25`) and then
  none: 202 consecutive refusals from `t=11.96` to the end, the
  completion reporting answers `0.79` to `1.31` from the carried lift
  and no certificate.  The `conic1e10` twin accepts 44 in `[10, 15)`
  (fifteen moves of `0.11` to `0.47` at `t=10.96` to `13.64`) and then
  none: 159 consecutive refusals from `t=13.68`.  The `etol1e3` twin
  keeps re-selecting (65 accepted in `[15, 20)`, thirteen moves of `0.10`
  to `0.44` there, a run of 22 refusals from `t=17.68`) and scores
  `0.204`; the base (19 refused in `[15, 20)`, two moves above `0.1` in
  the whole run) `0.149`.  The rule has no candidate but the certificate,
  so nothing in it acts on this count; the `infeasible` decision does not
  exist under `certified` scope, and the 111 to 125 answers above `0.4`
  that the frozen twins' completions return in `[15, 20)` are the same
  answers the measured rule declares `infeasible` (60 of 125 on task 9's
  base) or, when they pass the feasibility test, applies.
- **The frozen twins are the recorded base's late course.**  Their windows
  `(12, 16]` and `(16, 20]` (`0.49`, `0.51` and `0.42`, `0.49`) are the
  recorded base's (`0.47`, `0.52`, frozen from `t=11.76`), their joint
  Gram minima (`-2.3e-2`, `-3.6e-2`) are attained at `t=19.8` to `20` as
  the recorded base's `-5.3e-2` was, and the `dykstra60` twin's envelope
  leaves the frontier box at `t=18.78` (`1.018` at most), the only run of
  tasks 9 to 11 to exceed it by more than `1.6e-4`.

## (b) The certified-only rule is the recorded rule plus the margin's decline at `t=0.04`

Separation of each certified-only run from the recorded rule's same twin
(task 6's runs; the record-stages rerun of the task-4 row for the base),
and what the margin declined:

| run | declined by the margin (time; answer move; certified) | separation from the recorded twin: first nonzero; first frontier difference; crossing `1e-3` / `1e-2` / `1e-1`; at `t=20` | recorded twin's score at `t=8` / `20` | certified-only score at `t=8` / `20` |
|---|---|---|---|---|
| base | one: `t=0.04`; `0.334`; certified (the wrong-face answer) | `t=0.0425`; `0.333` at `t=0.04`; `0.125` / `0.48` / `4.405`; `1.66` | `0.045571` / `0.328674` | `0.046342` / `0.149326` |
| `etol1e3` | three: `t=0.04`, `0.08`, `0.12`; `0.302`, `0.437`, `0.342`; all certified | `t=0.0425`; `0.302` at `t=0.04`; `0.12` / `0.4225` / `4.4425`; `1.47` | `0.044530` / `0.325970` | `0.051402` / `0.203554` |
| `dykstra60` | one: `t=0.04`; `0.334`; certified | `t=0.0425`; `0.333` at `t=0.04`; `0.13` / `0.3925` / `3.4`; `0.693` | `0.040677` / `0.260123` | `0.058107` / `0.345743` |
| `conic1e10` | none | never; zero in every array over 8,001 samples | `0.040450` / `0.300340` | `0.040450` / `0.300340` |

Two readings.

- **The rule differs from the recorded rule in one to three decisions,
  all before `t=0.13`.**  With `certified` candidates the rule accepts
  exactly the certified answers the recorded rule accepted, less those
  whose declared-objective improvement is at most `1e-3`; over `t=0` to
  `20` that is one decline on the base and on `dykstra60` (the `t=0.04`
  wrong-face answer, improvement `-0.46`: the certified answer is worse
  than the carried lift by the completion's own functional), three on
  `etol1e3` (`t=0.04`, `0.08`, `0.12`, improvements `-0.40`, `-0.47`,
  `-0.24`) and none on `conic1e10`, whose `t=0.04` answer is uncertified
  at tolerance `1e-10` and refused by both rules.  Every other decision
  over 500 re-selections is the recorded rule's on the run's own
  trajectory.  So the certified-only `conic1e10` run *is* task 6's
  `conic1e10` twin, and the band measured here contains the recorded
  rule's band (`0.260` to `0.329` at `t=20` inside `0.149` to `0.346`)
  as it must.
- **The `t=0.04` decline is worth a factor 2.2 on the base at `t=20` and
  costs a factor 1.3 on the `dykstra60` twin.**  Declining one answer
  that moves the frontier by `0.33` with a negative improvement takes the
  base from `0.329` to `0.149` and the `etol1e3` twin from `0.326` to
  `0.204`, and takes the `dykstra60` twin from `0.260` (task 6's best
  recorded-rule twin, frozen late) to `0.346` (frozen from `t=11.96`).
  The recorded rule's `t=8` numbers (`0.0404` to `0.0456`, band 12.7
  percent) become `0.0404` to `0.0581` (43.7): on `dykstra60` the decline
  costs 43 percent at `t=8` already (`(4, 8]` window `0.0774` against
  `0.0509`).  A rule whose single early decision moves the `t=20` score
  by factors of 1.3 to 2.2 in either direction has no band narrower than
  the certificate's, which is what the measured acceptance was
  introduced to remove (tasks 7 and 8).

## (c) The certified rule with the ball radius, base only

Separation of the certified-plus-ball base from the certified base and the
cap-bound re-selections (`refusals.py`, `cap_events.py`):

| pair | first nonzero | first stage-record difference | crossing `1e-3` / `1e-2` / `1e-1` | window mean / maximum `(8, 12]` / `(12, 16]` / `(16, 20]` |
|---|---|---|---|---|
| certified plus `ball` from certified base | `t=11.1225` | `t=11.12`: the certified `0.290` answer (improvement `0.741`), accepted whole by the screen, scaled to `0.1` | `11.18` / `12.13` / `15.84` | `8.7e-4` / `6.7e-3`; `0.042` / `0.107`; `0.152` / `0.218` |
| certified plus `ball` from `ball` base (task 10) | `t=0.2425` | `t=0.08`: the `0.569` uncertified answer, declined by the measured rule and refused here (no move either way); the first move difference at `t=0.24` (task 10 (c)) | `0.59` / `2.105` / `4.75` | `0.375` / `0.489`; `0.412` / `0.479`; `0.723` / `1.13` |

| run | cap-bound (time; answer scaled to `0.1`; certified; improvement) | refused per quarter | accepted per quarter | largest applied move | score `(16, 20]` |
|---|---|---|---|---|---|
| certified plus `ball`, base | 7: `11.12` (`0.290`; yes; `0.741`), `16.96` (`0.209`; yes; `0.875`), `17.16` (`0.123`; yes; `1.26`), `19.28` (`0.102`), `19.4` (`0.105`), `19.72` (`0.124`), `19.8` (`0.114`), all certified | 12 / 19 / 13 / 35 | 111 / 106 / 112 / 90 | `0.1` | `0.2301` |
| certified screen, base (task 10) | --- (moves above `0.1`: `0.290` at `11.12`, `0.316` at `11.2`) | 12 / 19 / 14 / 19 | 111 / 106 / 111 / 106 | `0.316` | `0.2251` |

The cap first binds at `t=11.12`, on the first of the screen's two moves
above `0.1`; on the capped trajectory the `t=11.2` answer is below the
radius and the runs stay within `6.7e-3` to `t=12` and `0.107` to
`t=16`.  From `t=17` the capped run's completion refuses more often (35
against 19 in `[15, 20)`, in runs of four and five consecutive refusals
at `t=17.48`, `18.76`, `19.84`) and its last window is 2 percent worse;
over `t=0` to `20` `0.151644` against `0.149326`.  The seven caps are on
certified answers of `0.10` to `0.29`, none on the `0.4` to `0.8` answers
(the screen's 14 answers above `0.4` are uncertified and refused, as in
task 10; here 20 of 33 such answers are in `[15, 20)`, all refused).  The
radius has nothing to catch on this rule: the certificate already refuses
every large answer, and what it lets through is the moderate certified
move the radius halves to no effect.  Against the `ball` base the run is
14 percent worse at `t=20` (`0.1516` against `0.1325`), with the
separation of the two set at `t=0.24` by the first refused iterate, as
task 10 (c) found for the screen.  Its band is not measured; by (a) and
(b) it would be the certificate's.

## Re-selections of 500 over `t=0` to `20`

| run | accepted (of which uncertified; of which cap-bound) / declined by the margin / `infeasible` / `refused` | solver refusals (uncertified returns) | accepted per quarter `[0,5)` / `[5,10)` / `[10,15)` / `[15,20)` (refused) | last accepted re-selection; longest run of consecutive refusals (from) | largest applied move (moves above `0.1`) | mean re-selection wall |
|---|---|---|---|---|---|---|
| certified-only, base (task 10) | 434 (0; 0) / 1 / 0 / 65 | 65 | 111 / 106 / 111 / 106 (12, 19, 14, 19) | `t=19.96`; 6 (`t=9.36`) | `0.316` (2) | `1.01 s` |
| certified-only, `etol1e3` | 372 (0; 0) / 3 / 0 / 125 | 125 | 104 / 100 / 102 / 65 (17, 25, 23, 60) | `t=20.0`; 22 (`t=17.68`) | `0.440` (13, all in `[15, 20)`) | `1.32 s` |
| certified-only, `dykstra60` | 225 (0; 0) / 1 / 0 / 274 | 274 | 110 / 96 / 19 / 0 (13, 29, 106, 125) | `t=11.92`; 202 (`t=11.96`) | `0.369` (10, at `t=9.68` to `11.92`) | `1.02 s` |
| certified-only, `conic1e10` | 233 (0; 0) / 0 / 0 / 267 | 267 | 98 / 91 / 44 / 0 (26, 34, 81, 125) | `t=13.64`; 159 (`t=13.68`) | `0.465` (15, at `t=10.96` to `13.64`) | `1.21 s` |
| certified plus `ball`, base | 419 (0; 7) / 1 / 0 / 80 | 80 | 111 / 106 / 112 / 90 (12, 19, 13, 35) | `t=19.8`; 6 (`t=9.36`) | `0.1` (0) | `1.12 s` |
| measured, base (task 9) | 377 (39; 0) / 4 / 119 / 0 | 161 | 120 / 91 / 96 / 69 | `t=20.0`; 0 | `0.716` (21) | `1.32 s` |
| `ball`, base (task 10) | 426 (44; 8) / 5 / 69 / 0 | 117 | 120 / 91 / 100 / 114 | `t=20.0`; 0 | `0.1` (0) | `1.17 s` |
| recorded rule, base | 224 (0; 0) / 0 / 0 / 276 | 276 | 112 / 98 / 14 / 0 (12, 27, 111, 125) | `t=11.76`; --- | `0.334` (5) | --- |

Under `certified` scope the solver's refusals and the rule's are one
count; under `measured` scope 44 to 109 of the solver's 117 to 208
uncertified returns are applied and the rest are `infeasible`.  The
frozen twins' refusal counts (274, 267) are the recorded rule's (276 on
the base; 285, 192, 267 on its twins, task 6).

## Which rule gives the tightest `t=16` and `t=20` bands, and at what accuracy

Every rule on record at 15 keys, `N=16`, drive-power, base and three
one-setting twins to `t=20`; band = smallest to largest of the four,
spread as a fraction of the smallest, three of four in brackets:

| rule | `t=16` band (spread) [three of four] | `t=20` band (spread) [three of four] | base at `t=20` | what sets the band |
|---|---|---|---|---|
| recorded rule (task 6; every field off) | `0.202` to `0.258` (28.1 %) [13.1] | `0.260` to `0.329` (26.4 %) [9.4] | `0.329` | the certificate; frozen lift from `t=11.76` on the base |
| combined rule (task 7; margin, measured, anchor 1) | `0.184` to `0.224` (21.6 %) [7.1] | `0.255` to `0.274` (7.4 %) [3.8] | `0.264` | the anchor's accuracy cost; an answer branch at `t=6.44` |
| measured, no anchor (task 9) | `0.136` to `0.162` (18.8 %) [7.4] | `0.150` to `0.223` (48.8 %) [7.1] | `0.150` | `0.4` to `0.8` accepted moves from `t=10`; one twin |
| `ball`, radius `0.1` (task 10) | `0.132` to `0.158` (19.3 %) [14.3] | `0.133` to `0.207` (56.2 %) [23.0] | `0.133` | which twin's trajectory enters the `infeasible` regime after `t=10` |
| `carried`, radius `0.1` (task 10) | `0.188` to `0.207` (10.1 %) [4.9] | `0.216` to `0.252` (16.3 %) [9.2] | `0.216` | a frozen lift from `t=8.68` (7 accepted of 283) |
| certified-only (task 10 base, this task's twins) | `0.123` to `0.290` (135.3 %) [85.3] | `0.149` to `0.346` (131.5 %) [69.9] | `0.149` | the certificate: refusal flips from `t=0.6`, two twins frozen from `t=11.96` and `13.68` |
| certified plus `ball` (base only) | `0.125` | `0.152` | `0.152` | unmeasured; would be the certificate's |

The tightest bands at an accuracy below `0.21` at `t=16` and below `0.25`
at `t=20` are the measured rule's without the radius (task 9): 18.8
percent at `0.136` to `0.162` and 48.8 percent at `0.150` to `0.223`, with
three of the four within 7.4 and 7.1 percent (`0.136` to `0.146`; `0.150`
to `0.161`).  Tighter bands exist only at worse accuracy: the `carried`
policy's 10.1 and 16.3 percent at `0.188` to `0.207` and `0.216` to
`0.252`, and the combined rule's 7.4 percent at `0.255` to `0.274`, both
frozen-lift bands.  The `ball` form has the best base (`0.132` and
`0.133`, `1.03` times the both-cones line) at a band of the same kind as
task 9's (19.3 and 56.2 percent; three of four 14.3 and 23.0).  The
certified-only rule's band is the widest of the seven at both horizons,
and its base's `0.149` at `t=20` (task 10's "ties task 9's base") is one
member of a `0.149` to `0.346` band, so the tie is not a number the row can
carry.

## The 15-key ledger row

The row the plan file should carry is the `ball` row it has carried since
`90a3bd60` (the rule margin `1e-3`, `measured` candidates, no anchor,
trust radius `0.1` under the `ball` policy; 15 entrance keys, `N=16`,
strong coupling, cutoff 16, exact-correlated preparation, `h=0.0025`,
Dykstra 30 on both targets at stage cadence, the declared proximal log-det
completion re-solved every 16 steps warm-started on the carried frontier,
the energy-neutral repair in `drive_power` mode; every band the base and
its three one-setting twins), unchanged by this task:

| horizon `t=0` to `T` | score, base (band of the four; spread; three of four) | across the eight measured-candidate runs (tasks 9 and 10) | Eq. (22) internal-energy RMS (band) | certified-only band (excluded) | record, both cones |
|---:|---|---|---|---|---|
| 4 | `0.0272` (`0.0272` to `0.0272`; 0.1 %) | `0.0272` to `0.0272` | `0.0331` (`0.0331` to `0.0333`) | `0.0276` to `0.0281` (1.7 %) | --- |
| 8 | `0.0385` (`0.0385` to `0.0386`; 0.2 %) | `0.0385` to `0.0386` | `0.0382` (`0.0382` to `0.0384`) | `0.0404` to `0.0581` (43.7 %) | --- |
| 12 | `0.0989` (`0.0989` to `0.1043`; 5.5 %; 4.4 %) | `0.0982` to `0.1043` (6.2 %) | `0.0397` (`0.0397` to `0.0399`) | `0.0951` to `0.181` (90.4 %) | --- |
| 16 | `0.132` (`0.132` to `0.158`; 19.3 %; 14.3 %) | `0.132` to `0.162` (22.4 %) | `0.0404` (`0.0404` to `0.0406`) | `0.123` to `0.290` (135.3 %) | --- |
| 20 | `0.133` (`0.133` to `0.207`; 56.2 %; 23.0 %) | `0.133` to `0.223` (68.6 %) | `0.0409` (`0.0409` to `0.0411`) | `0.149` to `0.346` (131.5 %) | `0.1286` (energy `0.00094`; joint Gram `+1.4e-5`) |

with the joint Gram minimum over `t=0` to `20` `-7.9e-3` (band `-5.4e-3`
to `-7.9e-3`), the scaled extension minimum `-2.6e-4` (`-2.2e-4` to
`-2.6e-4`), the frontier envelope at most `0.748` (`0.748` to `0.910`), the
site-0 occupation RMS `0.1245` (`0.1245` to `0.1705`), walls `762` to
`847 s`.  The sentence the plan file's cheap-projection row should gain
(the plan file, `agent_guidance/paper-v/paper-v-state-and-plan-20260914.md`,
was not edited here: it is shared with the projective-route agent, which
commits to it, and this route does not commit):

> the certified-only rule (margin `1e-3`, `certified` candidates, no
> anchor; 2026-09-17, cheap task 11) is excluded from the row on its band:
> its four members spread 43.7, 90.4, 135.3 and 131.5 percent at `t=8`,
> `12`, `16`, `20` (`0.149` to `0.346` at `t=20`; two members frozen from
> `t=11.96` and `t=13.68`; the `conic1e10` member is the recorded rule's
> member bitwise, the rule being the recorded rule plus the margin's one
> decline at `t=0.04`), and the `ball` radius on its base moves `0.149`
> to `0.152` at `t=20`; the row stays the `ball` row, `0.0272` (0.1 %),
> `0.0385` (0.2 %), `0.0989` (`0.0989` to `0.1043`), `0.132` (`0.132` to
> `0.158`), `0.133` (`0.133` to `0.207`; three of four within 23 percent;
> `0.133` to `0.223` across the eight measured-candidate runs) at `t=4`,
> `8`, `12`, `16`, `20`, with the measured rule without the radius as the
> alternative with the tighter `t=20` band (`0.150` to `0.223`, three of
> four within 7.1 percent) at the worse base.

## Provenance and gates

- Gates: `paper_5/tests/test_cheap_projection_route.py` 44 passed in
  `25.08 s` before the chain (`test_cheap_projection_route_before.log`)
  and `24.66 s` after it (`test_cheap_projection_route_after.log`;
  `test_gate.zsh`).  The full `paper_5` suite was not re-run: no module,
  test or default changed since task 8's pass (499 passed) of the same
  module contents (`a81b7ecb`); `HEAD` is `f971a284` (the projective route's task-8 commit of `20:26`, before the chain; the owned modules unchanged since `a81b7ecb`)
  throughout.
- Scripts, all under
  `output/local_runs/paper_v_cheap_projection_task11_certified_t20_20260917_v1/`:
  `run_one.zsh` (task 7's, output directory changed), `chain.zsh` (the
  four rollouts in order, every twin and rule as flags), `smoke_check.py`
  and `smoke_check.zsh` (`smoke_check.log`), `test_gate.zsh`, `dry_run.zsh`
  (the scoring scripts on the runs available before the chain finished;
  `*_dry.*`), `analyse.zsh` (`analyse.log`), `task11_scores.py`
  (`task11_scores.json`, `.md`, `.log`: task 10's scorer with the
  certified-only set of four and the certified-plus-ball set, the
  refusals timed per horizon and per quarter, the reference-minus-exact
  energy, the times of the physicality minima, the three-of-four and
  pooled spreads, and each certified run's separation from the recorded
  rule's, task 9's and the `ball` set's same twin), `windows.py`
  (`windows.json`, `.md`, `.log`), `refusals.py` (`refusals.json`, `.log`:
  per run the refusals per quarter, the runs of consecutive refusals, the
  answers above `0.4` with their decisions, the accepted moves above
  `0.1`, the caps; per pair the first stage-record difference and the
  decisions on either side of each crossing), task 10's `cap_events.py`
  and `large_answers.py` on the five certified-candidate runs
  (`cap_events.json`, `.log`, `large_answers.log`), the run logs,
  `chain.log`, `chain_stdout.log`.
- Artifacts, under `output/local_runs/`:
  `paper_v_cheap_projection_task11_certified_{etol1e3,dykstra60,conic1e10}_t20_20260917_v1`,
  `paper_v_cheap_projection_task11_certball_base_t20_20260917_v1`, each
  with `repairs.jsonl`; the smoke
  `paper_v_cheap_projection_task11_smoke_certball_t005_20260917_v1`.
  Walls `914.8`, `869.8`, `840.7`, `802.4 s` (`autonomous_wall_seconds`),
  peaks `756`, `783`, `780`, `748 MiB` (`peak_resident_bytes`); every
  rollout under `pipelines/shell/ram_guard.py --limit-mb 3000`, one at a
  time, `20:27` to `21:25` on 2026-09-17.  The certified-only base is
  task 10's `paper_v_cheap_projection_task10_certified_base_t20_20260916_v1`.

## What to try next, in order

1. The 27-key `N=16` drive-power chain under the ledger row's rule (the
   `ball` form; task 8's item 2, task 10's item 2, unchanged).  The
   projective route's task 8 ran the measured rule without the radius on
   that chain to `t=8` and found the `t=6.6` facial-reduction iterate
   (`0.96` move), the radius's job; the cheap route's contribution is the
   base to `t=20` under the cap, a `2,000 s` chain in two segments.
2. The `infeasible` regime against the Dykstra iteration count on the
   base (60, 120; task 10's item 3): the only instrument left that acts
   on the band's cause, since this task closes the certificate as an
   alternative.
3. Nothing further on the certified screen: its base's `t=16` number
   (`0.123`) is one member of a `0.123` to `0.290` band and is not a row.

## Structured summary of task 11

| item | value |
|---|---|
| what was run | four `t=0` to `20` rollouts (8,000 steps) of the recorded `N=16` drive-power configuration, one at a time under `ram_guard --limit-mb 3000`: the certified-only rule (margin `1e-3`, `certified`, no anchor, no radius) on the three one-setting twins, completing the band whose base is task 10's certified screen, and the same rule with trust radius `0.1` under `ball` on the base; walls `914.8` / `869.8` / `840.7` / `802.4 s`, peaks `748` to `783 MiB`; all at `t=20`; no module edited, 44 tests pass before and after, nothing committed |
| certified-only scores, base / `etol1e3` / `dykstra60` / `conic1e10` | `t=4` `0.027635` / `0.028102` / `0.027735` / `0.027953`; `t=8` `0.046342` / `0.051402` / `0.058107` / `0.040450`; `t=12` `0.097956` / `0.095070` / `0.181008` / `0.108195`; `t=16` `0.123294` / `0.144537` / `0.290131` / `0.228471`; `t=20` `0.149326` / `0.203554` / `0.345743` / `0.300340` |
| spread (three of four) | 1.7 (1.2), 43.7 (25.4), 90.4 (13.8), 135.3 (85.3), 131.5 (69.9) percent at `T=4`, `8`, `12`, `16`, `20`; task 9's 0.1, 0.2, 3.2, 18.8, 48.8; `ball` 0.1, 0.2, 5.5, 19.3, 56.2; recorded 17.7, 12.7, 21.0, 28.1, 26.4 |
| certified plus `ball`, base | `0.027635` / `0.046342` / `0.097938` / `0.124542` / `0.151644`: equal to the certified screen to `t=11.12` (first cap), 1.0 and 1.6 percent above it at `t=16` and `20`, 14 percent above the `ball` base; 7 caps (all certified, answers `0.10` to `0.29`), 80 refused (12 / 19 / 13 / 35), envelope `0.616`, cones `-2.1e-4` / `-5.0e-3` |
| first-separation times (certified twins from the certified base) | `1e-3` at `t=0.6375` (`etol1e3`; the twin refuses at `t=0.6` where the base accepts `0.0089`), `0.81` (`dykstra60`; refusal at `0.8`), `0.6375` (`conic1e10`); `1e-1` at `5.175`, `6.1375`, `3.9225` (task 9 `11.2`, `12.9`, never; recorded `7.0`, `4.0`, `3.1`); at `t=20` `0.772`, `1.63`, `1.60`; the certified-plus-ball base leaves the certified base at `t=11.1225` and crosses `1e-1` at `15.84` |
| refusals (of 500) | 65 / 125 / 274 / 267 (certified-only base / `etol1e3` / `dykstra60` / `conic1e10`), 80 (certified plus ball); per quarter `dykstra60` 13 / 29 / 106 / 125 (202 consecutive from `t=11.96`), `conic1e10` 26 / 34 / 81 / 125 (159 from `t=13.68`), `etol1e3` 17 / 25 / 23 / 60; no `infeasible`, no uncertified acceptance; declined by the margin 1 / 3 / 1 / 0 / 1 |
| internal energy (Eq. (22)) at `t=4` / `8` / `12` / `16` / `20` | base `0.0332` / `0.0384` / `0.0400` / `0.0408` / `0.0413` (reference `0.0430` above exact); twins `0.0290` to `0.0295` / `0.0338` to `0.0343` / `0.0354` to `0.0357` / `0.0361` to `0.0364` / `0.0366` to `0.0368` (references `0.0383` to `0.0385`); certified plus ball as the base |
| physicality over `t <= 20` and wall | extension `-1.9e-4` / `-6.3e-4` / `-1.0e-3` / `-1.6e-3`; joint Gram `-4.4e-3` / `-1.5e-2` / `-2.3e-2` / `-3.6e-2`; envelope `0.710` / `0.851` / `1.018` (above the box from `t=18.78`) / `0.893`; certified plus ball `-2.1e-4`, `-5.0e-3`, `0.616`; walls `743.4` (task 10) / `914.8` / `869.8` / `840.7 s`, `802.4 s` |
| the rule against the recorded rule | the certified-only `conic1e10` twin is task 6's `conic1e10` twin bitwise (zero separation over 8,001 samples); base, `etol1e3`, `dykstra60` leave the recorded twin at `t=0.0425` through the margin's decline of the `t=0.04` wrong-face answer (improvement `-0.40` to `-0.46`), which takes `t=20` from `0.329` to `0.149`, `0.326` to `0.204`, and `0.260` to `0.346` |
| tightest `t=16` and `t=20` bands | below `0.21` / `0.25`: the measured rule without the radius (task 9), 18.8 percent at `0.136` to `0.162` and 48.8 percent at `0.150` to `0.223` (three of four 7.4, 7.1); tighter only at worse accuracy (`carried` 10.1 / 16.3 at `0.188` to `0.207` / `0.216` to `0.252`; combined 7.4 at `0.255` to `0.274`); best base the `ball` form (`0.132`, `0.133`; bands 19.3, 56.2); the certified-only rule the widest (135.3, 131.5) |
| ledger row | unchanged: the `ball` row, `0.0272` (0.1 %), `0.0385` (0.2 %), `0.0989` (`0.0989` to `0.1043`; 5.5 %), `0.132` (`0.132` to `0.158`; 19.3 %), `0.133` (`0.133` to `0.207`; 56.2 %; three of four within 23.0; `0.133` to `0.223` across the eight measured-candidate runs) at `t=4`, `8`, `12`, `16`, `20`; the certified-only rule excluded on its band (`0.149` to `0.346` at `t=20`); the sentence for the plan row is under "The 15-key ledger row"; plan file not edited |
| next | the 27-key chain under the `ball` rule (base to `t=20`); the `infeasible` regime against the Dykstra count |
| files changed | this report, the worklog block, and the task-11 scripts and artifacts under `output/local_runs/`; no module, test or default; nothing committed |


# The declared per-coordinate metric of the map back: implemented, swept, and refused at the running configuration (2026-09-18)

The repair-displacement measurement ended on one instruction: *the
per-coordinate projection scale, in that order --- module change, free replay
on the stored `(6, 8]` samples, then the one 15-minute `t = 4` to `8`
segment*.  This round is the module change and the free replay.  The replay
refuses the fix, so the segment is not spent.

Configuration and conventions are tasks 11's and 12's: 27 active keys
(`entrance15_plus_two_spin_boson`), 56 propagated state coordinates, 70
standardized frontier coordinates, 36 positive-extension rows, 8 retained
rows, `N = 16`, strong coupling, cutoff 16, the exact-correlated preparation
at the declared ladder's first rung, time step `0.0025`, `policy="dykstra"`,
`cadence="stage"`, `target="both"`, `eigenvalue_floor` and `repair_tolerance`
and `energy_tolerance` all `0`, `dykstra_iterations` 120 on the task-11 base
and 30 on the recorded `drive_power` chain, 800 stored samples in `(6, 8]`
per chain after duplicate resume times are dropped, one replayed repair at
each of them.  The quantity throughout is the relative-mode population
`(phonon[0,0] + phonon[1,1]) / 2 - Re phonon[0,1]` of the centred phonon
density, read through the chart the scorer uses.  Stored trajectories and
affine algebra only: nothing was propagated, no conic solve was run, nothing
was committed, and no default, rule, radius, Dykstra count, reference mode or
ladder changed.

## Negative results first

**The recommended fix, taken literally, is exactly inert here --- and for a
stronger reason than the one the record gave for `state_motion_scale`.**  At
`target="both"` on the 27-key dictionary the realified affine Gram map has
126 columns and rank 126: it is injective, so the least-squares map back has
one exact answer and every positive weight returns it.  The earlier round
explained `state_motion_scale`'s inertness by the null space lying inside the
propagated-state group; at this target there is no null space at all.  The
replay confirms it to every printed digit: with the Eq. (22) hyperplane off,
a declared metric of `0.001` on the four raw phonon normal coordinates leaves
the base's displacement at `4.681371e-05` per repair, its correction length at
`2.096e-04` and its both-cone fraction at `0.134`, all identical to the empty
declaration; at `target="retained"` (126 columns, rank 29, the 97 unseen
directions being exactly the 27 entrance and 70 frontier coordinates the
retained Gram does not read, none of them a raw phonon normal coordinate) the
displacement stays `8.310995e-08`; and on the recorded chain every one of the
five declared values `10`, `1`, `0.1`, `0.01`, `0.001` returns `9.368395e-04`
with correction length `3.790e-03`.

**Where the declared metric *can* act, it moves the coordinate three times
further, not less.**  The one place a coordinate metric survives an injective
map back is the direction inside the affine Gram set along which the Eq. (22)
energy hyperplane is met, which the committed repair takes along the
hyperplane's own normal in the realified feature space.  Met instead along
the smallest correction in the declared metric, the base's displacement of
the relative-mode population rises from `1.502784e-04` to `4.486596e-04` per
repair in mean absolute value over `(6, 8]` (`2.99` times), and the recorded
chain's from `5.264001e-04` to `8.756030e-03` (`16.6` times).  The target was
below the closure's own per-step displacement of `8.421646e-05`.  No declared
value reaches it; the committed behaviour is the best member of the family,
and the family is monotone away from it.

**Weighting the coherent coordinates as well does not recover it.**  The
relative-mode population is a quantity of the centred phonon density
`N = raw_normal - alpha alpha^dagger`, so it also reads the four coherent
coordinates, through `-(1/2) |alpha_0 - alpha_1|^2`.  Declaring the metric
over all eight at `0.01` gives `4.580947e-04` per repair, `1.02` times the
four-coordinate figure and `3.05` times the committed one.  The coordinate
set is not what fails.

**A strided subsample of the window is not a uniform sample of it.**  The
chain carries `lift_reselection_interval` 16, so a stride that divides 16
lands on one phase of the re-selection cadence.  At stride 16 the base's
committed setting reads `3.145754e-06` per repair instead of the `1.502784e-04`
it reads at stride 1 --- a factor `47.8`, and the state correction `3.668e-06`
instead of `2.131e-04`.  Every figure in this report is at stride 1, all 800
samples.  The earlier round's Dykstra-count and motion-scale sweeps were taken
at stride 8, which divides 16; their *comparisons* stand (all settings share
the phase) but their absolute per-repair figures do not transfer to the
window.

## (a) What the option is, what it changes, and what it leaves alone

`CheapProjectionSettings.raw_coordinate_motion_scales`, default `()`, an
empty declaration.  Each entry names one of the 29 raw moment coordinates of
`RAW_MOMENT_COORDINATE_NAMES` and a positive factor; the factor multiplies
`state_motion_scale` on that one coordinate.  A mapping is accepted and
stored as the tuple of pairs, so the declaration is hashable and keys the
factorization caches.  `RAW_PHONON_NORMAL_COORDINATE_NAMES` names the four
raw phonon normal coordinates (`raw_normal_00`, `raw_normal_11`,
`raw_normal_01_real`, `raw_normal_01_imag`) --- the block whose diagonal the
Eq. (22) gradient reads at `0.5`, `0.5` and whose diagonal the relative-mode
population reads at the same `0.5`, `0.5`.

It changes exactly two quantities.

1. **The metric of the least-squares map back.**
   `AffineGramCoordinates.state_coordinate_scales` builds the per-coordinate
   freedom vector the factorization already carried as two group constants,
   and the map back stays `increment = scales * (pseudo_inverse @ residual)`
   with `pseudo_inverse = pinv(design * scales)`, that is the minimum-norm
   correction in the declared metric.  Measured inert at both targets of this
   configuration, for the reason above.
2. **The direction the Eq. (22) energy hyperplane is met along.**
   `AffineGramCoordinates.energy_correction_direction` returns
   `scales**2 * gradient` carried through the same affine Gram map and reduced
   onto the range basis, which is the smallest correction in the declared
   metric that changes the internal energy; `_dykstra` takes the offset up
   along it instead of along the hyperplane's own normal, dividing by the
   normal's pairing with it so that the constraint is still met exactly.  An
   empty declaration returns `None` and the committed branch runs.

It leaves alone: the propagated velocity, the two cones and their declared
floors, the eigenvalue clip that projects onto them, the affine Gram set and
its orthogonal projection (the range basis is built from the unscaled design
and is the same subspace for every declared metric), the number of Dykstra
iterations and the Dykstra correction bookkeeping, the declared Eq. (22)
reference and the tolerance at which a repair fires, the lift re-selection,
and the internal energy the repaired coordinates carry --- measured on the
reference to `6.084e-16` to `6.378e-16` per repair on the base and `6.334e-16`
to `6.783e-16` on the recorded chain at every declared value, against
`6.3782e-16` and `6.3949e-16` for the committed repair.  Because the cone is
still projected onto in the Frobenius metric while the affine sets are met in
the declared one, the alternation is no longer a single-metric Dykstra
iteration: how often it reaches the cones is a measured quantity below, not a
guaranteed one.  The carried-witness lane refuses a declared metric outright,
its 384 coordinates not being the raw moment ones.

Two consequences worth stating.  `dataclasses.asdict` now carries
`"raw_coordinate_motion_scales": []` into every `summary.json`, so a segment
resumed from a summary written before this change lists that one field in its
recorded `projection_settings_changed`; nothing reads or raises on that field.
And the option is deliberately not exposed on
`cheap_projection_route_analysis.py`'s command line: it is reachable from
Python only, because the replay below refuses it.

## (b) The sweep on the stored `(6, 8]` samples

One replayed repair at each of the 800 stored samples of `(6, 8]`, at the
chain's own stored `frontier_standardized` and stored
`internal_energy_reference`.  `mean absolute`, `signed mean` and `integrated`
are of the per-repair displacement of the relative-mode population;
`correction` is the mean Euclidean length of the repair's correction to the 56
propagated coordinates and `share` the mean fraction of that length landing on
this one direction; the three cone columns are the fraction of the 800 repairs
after which that cone is reached; `dE` is the mean absolute change of the
Eq. (22) internal energy.  `--` is the empty declaration, that is the
committed repair.

Task-11 base (`..._measured_dyk120_base_20260918m`, `dykstra_iterations` 120,
`energy_reference_mode="drive_power"`):

| declared metric | mean absolute | signed mean | integrated | correction | share | both cones | retained | extension | `dE` |
|---|---|---|---|---|---:|---:|---:|---:|---|
| `--` (committed) | `1.502784e-04` | `-1.502784e-04` | `-1.202227e-01` | `2.131e-04` | `0.771` | `0.010` | `0.010` | `0.161` | `6.378e-16` |
| `10` | `2.532811e-04` | `-2.532811e-04` | `-2.026249e-01` | `2.964e-04` | `0.933` | `0.086` | `0.089` | `0.159` | `6.251e-16` |
| `1` | `4.021687e-04` | `+3.999627e-04` | `+3.199701e-01` | `7.997e-04` | `0.348` | `0.095` | `0.149` | `0.163` | `6.212e-16` |
| `0.1` | `4.481513e-04` | `+4.481325e-04` | `+3.585060e-01` | `8.622e-04` | `0.362` | `0.085` | `0.134` | `0.165` | `6.317e-16` |
| `0.01` | `4.486544e-04` | `+4.486543e-04` | `+3.589234e-01` | `8.628e-04` | `0.362` | `0.091` | `0.140` | `0.165` | `6.378e-16` |
| `0.001` | `4.486596e-04` | `+4.486595e-04` | `+3.589276e-01` | `8.628e-04` | `0.362` | `0.080` | `0.126` | `0.166` | `6.084e-16` |
| `0.01` over eight names | `4.580947e-04` | `+4.580945e-04` | `+3.664756e-01` | `6.813e-04` | `0.480` | `0.100` | `0.547` | `0.171` | `6.112e-16` |

Recorded `drive_power` chain (`..._27key_n16_20260915e`, `dykstra_iterations`
30):

| declared metric | mean absolute | signed mean | integrated | correction | share | both cones | retained | extension | `dE` |
|---|---|---|---|---|---:|---:|---:|---:|---|
| `--` (committed) | `5.264001e-04` | `-5.264001e-04` | `-4.211201e-01` | `1.099e-03` | `0.495` | `0.000` | `0.000` | `0.007` | `6.395e-16` |
| `10` | `4.078098e-03` | `-4.078098e-03` | `-3.262478e+00` | `4.744e-03` | `0.886` | `0.000` | `0.000` | `0.004` | `6.783e-16` |
| `1` | `7.554921e-03` | `+7.554903e-03` | `+6.043922e+00` | `1.618e-02` | `0.461` | `0.000` | `0.000` | `0.000` | `6.467e-16` |
| `0.1` | `8.742676e-03` | `+8.742676e-03` | `+6.994141e+00` | `1.784e-02` | `0.485` | `0.000` | `0.000` | `0.000` | `6.334e-16` |
| `0.01` | `8.755897e-03` | `+8.755897e-03` | `+7.004718e+00` | `1.785e-02` | `0.485` | `0.000` | `0.000` | `0.000` | `6.711e-16` |
| `0.001` | `8.756030e-03` | `+8.756030e-03` | `+7.004824e+00` | `1.785e-02` | `0.485` | `0.000` | `0.000` | `0.000` | `6.467e-16` |

With the hyperplane off (`energy_reference_mode="none"`), where only the map
back can act, every declared value reproduces the empty declaration exactly:
base `target="both"` `4.681371e-05` per repair, correction `2.096e-04`, both
cones `0.134`; base `target="retained"` `8.310995e-08`, correction
`1.898e-04`, both cones `0.087`; recorded chain `target="both"`
`9.368395e-04`, correction `3.790e-03`, both cones `0.000`.

Read plainly.  The declared metric does three things and only one of them is
wanted.  It cuts the share of the correction landing on this one direction
from `0.771` to `0.362` on the base --- so it does make the coordinate
expensive, as designed.  It raises the both-cone fraction from `0.010` to
`0.080` to `0.100` and the retained-cone fraction from `0.010` to `0.126` to
`0.149`, which is the alternation no longer spending its budget fighting the
hyperplane.  And it lengthens the correction from `2.131e-04` to `8.628e-04`,
`4.05` times, because meeting the energy along a coordinate-cheap direction is
far in the Frobenius metric the cones are projected in, so the cone chase has
more to undo.  The product of a smaller share and a much longer correction is
`2.99` times the displacement the fix was meant to cut by `1.78`.  The
saturation is complete by `0.1` (`4.481513e-04`, `4.486544e-04`,
`4.486596e-04` at `0.1`, `0.01`, `0.001`), so no smaller value exists to try;
and above `1` the family walks back towards the committed behaviour
(`2.532811e-04` at `10`), which is the Frobenius-metric limit.

## (c) The recommendation, and the test that would decide it

**Adopt no value: keep the empty declaration.**  The replay's pass condition
was stated in advance --- the per-repair displacement below the closure's own
`8.421646e-05`, the both-cone fraction not below `0.010`, the energy still on
the reference to `1e-15`.  Two of the three hold at every declared value (the
cone fraction improves by `8` to `10` times, the energy lands at `6.1e-16`);
the one the fix exists for fails by `2.99` on the base and `16.6` on the
recorded chain.  If a value had to be named for its cone attainment alone it
would be `0.01` on the four raw phonon normal coordinates --- the saturated
member, both cones at `0.091` against `0.010` --- and it is not worth
`4.486544e-04` against `1.502784e-04` on the coordinate that carries the
`(6, 8]` loss.

**The decisive rollout test, not run.**  Task 12's `t = 4` to `8` rerun from
the task-11 base's stored `t = 3.92` state
(`..._measured_dyk120_base_20260918m_seg1`), one segment to `t = 8.0` under
`ram_guard --limit-mb 4500`, about 15 minutes, scored on the `exact_t8.npz`
grid; improving means the relative-mode population's RMS over `(6, 8]` below
`1.907462e-01` and the four phonon-occupation coordinates' block RMS below
`8.278358e-02`, and closing the gap to the recorded chain means
`7.513319e-02`.  It would also need one line in
`cheap_projection_route_analysis.py` to reach the declared metric from the
driver, which was deliberately not added.  By the record's own rule that
segment is not spent: the replay it was conditioned on does not pass.

**What the measurement points at instead: a slab rather than a hyperplane,
which is a module change and not a setting.**  Both ends are now measured over
the whole horizon.  With the Eq. (22) hyperplane in the alternation at every
repair the base moves this coordinate `1.502784e-04` per repair at both-cone
attainment `0.010`; with it off, `4.681371e-05` --- below the `8.421646e-05`
target --- at `0.134`, at the price of an energy change of `1.8006e-04` per
repair, and the projective route's task 13 is what a whole chain run that way
costs: both cones held from `t = 0` to `20` (retained `-1.549433e-12`,
extension `-1.349691e-09`) at `6.360` times the constrained base's score at
`t = 20` and `256.9` times its Eq. (22) internal-energy RMS.  What sits
between them is a slab: the repair corrects the internal energy only when it
is further than a declared band from the reference, and then only to the
band's boundary, so a cone correction that costs less energy than the band is
let through whole and the coordinate pays only the excess.  `energy_tolerance`
is *not* that setting --- it decides whether a repair runs at all, and since a
cone is violated at `0.990` of the base's stored samples the repair runs and
the hyperplane lands the energy exactly on the reference anyway, which is the
same reading task 13 records.  The change is one place, the hyperplane step of
`AffineGramCoordinates._dykstra` in the file this round already touches: the
offset `reference - energy_before` soft-thresholded by the declared band,
zero inside it.  The pass condition to state in advance, at a band of `1e-3`
--- `5.6` times the energy one hyperplane-free repair changes, against an
Eq. (22) internal energy of `0.0332` to `0.0413` over `t = 4` to `20` on the
task-11 base --- is the same as this round's: the per-repair displacement of
the relative-mode population below `8.421646e-05` with the both-cone fraction
not below `0.010`, on the same free replay of the stored `(6, 8]` samples
(about `100 s` per chain), and only then the same 15-minute `t = 4` to `8`
segment.  Unlike the metric, this one the replay *can* score: the band acts on
the offset the stored sample carries (`3.469447e-16`), so every repair in the
replay falls inside any band worth declaring and the replay reads the
hyperplane-free rows --- which is the answer, at the cone attainment and
energy those rows also carry.

## Gates

| gate | result |
|---|---|
| the default path is unchanged bitwise | a 4-step rollout (`t = 0` to `0.01`, step `0.0025`, 15 entrance keys, cutoff 16, `policy="dykstra"`, `target="both"`, `cadence="stage"`, `dykstra_iterations` 30, `energy_reference_mode="drive_power"`, 8 repairs applied, all 8 energy-constrained) run under the changed module and under the module at `94cf93b5` --- which is the module at `e990cbb0`, the branch head after the projective route's task 13, unchanged --- agrees in all seven stored arrays --- times, states, standardized frontier, both minimum-eigenvalue series, internal energy and its reference --- at maximum difference `0.000000e+00` |
| the replay reproduces the record | base `target="both"` at the chain's own mode: `1.502784e-04` mean absolute, `-1.502784e-04` signed mean, `-1.202227e-01` integrated, `5.846746e-04` maximum, `4.916299e-04` p99, negative at `0.990`, both cones after `0.010`; `energy_reference_mode="none"` `4.681371e-05`; `target="retained"` with the hyperplane off `8.310995e-08`; recorded chain `5.264001e-04` and `9.368395e-04` --- every digit of the repair-displacement measurement's tables |
| the replay reads the record's coordinate | the relative-mode population read from the stored 56 propagated coordinates through the scorer's chart equals the one read from the stored `archive_coordinates` to `0.000000e+00`, and the affine Eq. (22) energy equals the stored `internal_energy` to `0.000000e+00`, at every sample of both chains |
| the constraint is still met | the replayed repair changes the Eq. (22) internal energy by `6.084e-16` to `6.378e-16` (base) and `6.334e-16` to `6.783e-16` (recorded) in mean absolute value at every declared value, against `6.3782e-16` and `6.3949e-16` committed |
| the map back has no freedom to weight | the realified design over the 126 movable coordinates has rank 126 at `target="both"`, 29 at `target="retained"` (the 97 unseen directions are the 27 entrance and 70 frontier coordinates) and 111 at `target="extension"` (15 unseen, three of them wholly inside the four raw phonon normal coordinates), so at the chain's own target the correction is unique |
| the trajectories are the recorded ones | each chain read from its own stored `trajectory.npz` segments, six per chain; sample spacing `0.0025` exactly throughout; 800 samples per chain in `(6, 8]` after duplicate resume times are dropped; stride 1 |
| the tests | `paper_5/tests/test_cheap_projection_route.py` 48 passed in `22.9 s` (44 before, 4 added), under `ram_guard --limit-mb 2500`, peak `465 MiB` |
| nothing else changed | no default, rule, radius, Dykstra count, reference mode, tolerance or ladder changed; `projected_apcm.py` and `apcm_projective_event_guard_analysis.py` untouched; tasks 8 to 12's and the recorded chains' artifacts untouched; nothing committed |
| light compute only | eight processes under `pipelines/shell/ram_guard.py --limit-mb 2500`, one at a time, peaks `1 MiB` to `465 MiB`, each under two minutes |

## Provenance

- Module: `paper_5/src/paper5/stability/cheap_projection_route.py`
  (`CheapProjectionSettings.raw_coordinate_motion_scales`,
  `RAW_PHONON_NORMAL_COORDINATE_NAMES`,
  `AffineGramCoordinates.state_coordinate_scales`,
  `AffineGramCoordinates.energy_correction_direction`, the
  `hyperplane_direction` argument of `AffineGramCoordinates._dykstra`, the
  factorization and energy-functional cache signatures, and the
  carried-witness refusal).  Tests:
  `paper_5/tests/test_cheap_projection_route.py`
  (`test_raw_coordinate_motion_scales_are_off_and_change_no_number`,
  `test_raw_coordinate_motion_scales_reject_what_they_cannot_name`,
  `test_raw_coordinate_motion_scales_move_the_phonon_normal_block_less`,
  `test_the_carried_witness_lane_refuses_a_declared_metric`).
- Scripts under
  `output/local_runs/paper_v_projective_dictionary_energy_20260915e/`:
  `probe_repair_displacement_20260918.py` extended with
  `--phonon-normal-scale` and `--metric-coordinate` (additive; without them
  every earlier invocation takes the same path, which the first row of every
  table above checks), and `tabulate_phonon_scale_20260918.py`.
- Outputs in the same directory:
  `repair_displacement_base_phononscale_a_20260918.json` and `.log`,
  `repair_displacement_base_phononscale_b_20260918.json` and `.log`,
  `repair_displacement_base_phononscale_mapback_20260918.json` and `.log`,
  `repair_displacement_base_metricset8_20260918.json` and `.log`,
  `repair_displacement_recorded_phononscale_20260918.json` and `.log`,
  `repair_displacement_phonon_scale_tables_20260918.log`.
- Read but not modified: `adaptive_positive_moment.py`
  (`RAW_MOMENT_COORDINATE_NAMES`, `raw_moment_coordinates_to_matrix_state`),
  `apcm_positive_extension.py`, `matrix_reference.py`,
  `cheap_projection_route_analysis.py`, `run_energy_segment.py`.

## Structured summary of the declared-metric round

| item | value |
|---|---|
| (a) what was implemented | `CheapProjectionSettings.raw_coordinate_motion_scales`, default `()`, naming raw moment coordinates and a positive factor each, multiplying `state_motion_scale` on that coordinate.  It sets the metric of the least-squares map back and the direction inside the affine Gram set along which the Eq. (22) hyperplane is met.  It leaves the velocity, the cones and their floors, the clip, the affine Gram set and its range basis, the Dykstra count and bookkeeping, the declared reference and the firing tolerance, and the lift re-selection untouched; the energy still lands on the reference (`6.1e-16`).  The alternation stops being single-metric, so cone attainment becomes a measured quantity.  The default path is bitwise unchanged against the module at `94cf93b5` on a 4-step `drive_power` rollout, maximum difference `0.000000e+00` |
| (b) the map-back channel | exactly inert at this configuration: the affine Gram map is injective at `target="both"` (126 columns, rank 126), so the weighted minimum-norm correction is the only correction.  With the hyperplane off every declared value reproduces the empty declaration to every printed digit --- base `4.681371e-05` per repair at `target="both"` and `8.310995e-08` at `target="retained"`, recorded chain `9.368395e-04`, correction lengths and cone fractions included |
| (b) the hyperplane channel, base over `(6, 8]` | `1.502784e-04` per repair committed; `2.532811e-04` at `10`, `4.021687e-04` at `1`, `4.481513e-04` at `0.1`, `4.486544e-04` at `0.01`, `4.486596e-04` at `0.001` (signed means `+3.999627e-04` to `+4.486595e-04` from `1` down, integrated `+3.199701e-01` to `+3.589276e-01`); the share of the correction on this direction falls `0.771` to `0.362`, the correction lengthens `2.131e-04` to `8.628e-04` (`4.05`), both cones rise `0.010` to `0.080`--`0.100` and the retained cone `0.010` to `0.126`--`0.149`; eight names at `0.01` gives `4.580947e-04` |
| (b) the hyperplane channel, recorded chain | `5.264001e-04` committed; `4.078098e-03`, `7.554921e-03`, `8.742676e-03`, `8.755897e-03`, `8.756030e-03` at `10`, `1`, `0.1`, `0.01`, `0.001`; correction `1.099e-03` to `1.785e-02`; both cones `0.000` throughout, the extension cone `0.007` falling to `0.000` |
| (c) the recommendation | adopt no value; keep the empty declaration.  The pre-stated pass condition --- below the closure's own `8.421646e-05` per step --- fails at every value by `2.99` (base) and `16.6` (recorded), while the other two conditions hold.  The saturation is complete by `0.1`, so no smaller value exists to try, and above `1` the family returns towards the committed behaviour |
| (c) the decisive rollout, not run | task 12's `t = 4` to `8` rerun from the base's stored `t = 3.92` state, one segment under `ram_guard --limit-mb 4500`, about 15 minutes on the `exact_t8.npz` grid, improving only below `1.907462e-01` for the relative-mode population's `(6, 8]` RMS and `8.278358e-02` for the four phonon-occupation coordinates' block; plus one line in the analysis driver to reach the option.  Not spent, because the replay it was conditioned on does not pass |
| what to try next | a slab rather than a hyperplane: correct the Eq. (22) energy only when it is further than a declared band from the reference, and then only to the boundary.  `energy_tolerance` cannot do it --- it gates whether a repair runs, and a cone is violated at `0.990` of the base's samples, so the repair runs and the hyperplane lands the energy exactly on the reference.  One place, the hyperplane step of `_dykstra`: the offset soft-thresholded by the band.  The two ends are measured --- the hyperplane at every repair (`1.502784e-04` per repair, both cones `0.010`) and off (`4.681371e-05`, `0.134`, and task 13's whole-horizon price of `6.360` at `t = 20` and `256.9` in internal-energy RMS).  Free replay first at a band of `1e-3`, about `100 s` per chain, then the same 15-minute `t = 4` to `8` segment |
| method note | a stride that divides the chain's `lift_reselection_interval` 16 samples one phase of the re-selection cadence: at stride 16 the committed setting reads `3.145754e-06` per repair against `1.502784e-04` at stride 1, a factor `47.8`.  Every figure here is stride 1 over all 800 samples |
| the ledger row | unchanged; this round produces no horizon score |
| files changed | the module and its tests (the named option, four tests), the replay script's two additive flags and the tabulation script, the outputs listed above, this report and the worklog block.  `projected_apcm.py` and `apcm_projective_event_guard_analysis.py` untouched; nothing committed |
