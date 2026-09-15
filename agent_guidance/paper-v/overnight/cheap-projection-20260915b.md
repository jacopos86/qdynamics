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
