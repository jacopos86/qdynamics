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
