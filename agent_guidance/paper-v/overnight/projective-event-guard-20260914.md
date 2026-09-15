# projective-event-guard (2026-09-14)

Plan steps 10 and 13 under the exploratory reading of D1 (event-triggered
outer guard) and D2 (strictly convex selector as a named option).  Defaults
untouched: every new behaviour is a keyword whose default reproduces the
recorded path.

Status: in progress.

## Context read

- `agent_guidance/paper-v/paper-v-state-and-plan-20260914.md` sections 3.4,
  3.5, 4, 7.
- `paper_5/notes/electron_phonon_closure_worklog_20260803.md:3013-3340`
  (full APCM implementation, RAM-safe fixed-dictionary APCM, URPG amendment
  pre-rollout, exploratory URPG continuation, per-stage acceleration audit).
- `MATH/paper_details/paper_V_high_u_gkba.tex:172-326` (McLachlan pathway).

## Dimensions of the fixed-dictionary entrance problem (measured)

| quantity | value |
|---|---:|
| raw archive coordinates (`RAW_MOMENT_COORDINATE_NAMES`) | 29 |
| entrance relative moment keys (`ENTRANCE_RELATIVE_MOMENT_KEYS`) | 15 |
| propagated projected-APCM state | 44 |
| current positive extension Gram rows | 24 |
| current frontier moments | 62 |
| invariant target readout rank | 15 |
| one-shell outer extension Gram rows / frontier moments | 60 / 238 |

## Impact analysis (main checkout index, read-only)

| symbol | callers found | risk |
|---|---|---|
| `FixedDictionaryProjectedAPCM.contain_stage` | `apcm_adaptation.py:215`, `projected_apcm.py:632,649,666`; `test_projected_apcm.py` duck-types it in `FakeModel` | LOW |
| `integrate_projected_apcm_ssprk3` | index UNKNOWN; grep: `test_projected_apcm.py:379,455` only | LOW after grep |
| `select_outer_feasible_projective_guard` | index UNKNOWN; grep: `test_apcm_projective_guard.py:324` only | LOW after grep |
| `_solve_explicit_target_qp` | `apcm_projective_guard.py:1331` (self-retry), `2352`, `2450` | LOW |
| `SymmetryReducedAPCMGeometry.retract_stage` | `projected_apcm.py:542`, `test_apcm_resource_safety.py:202` | LOW |

No HIGH or CRITICAL verdict. Every UNKNOWN was resolved by text search
before editing.

## Negative result first

The recorded gate reorder does not discriminate at the cutoff-16 correlated
preparation on this machine, because the first-pass Clarabel status is no
longer the status the reorder is keyed on.  The outer-aware invariant target
solve now ends in `explicit_target:NumericalError` after about 46 s, and
`NumericalError` is not in the retry ladder's trigger set
(`apcm_projective_guard.py:1351`, `{"AlmostSolved", "InsufficientProgress",
"MaxIterations"}`), so no retry ladder runs with either setting and there is
nothing for `accept_first_provisional` to skip.  The record's 500 s then
171 s pair was measured on a first-pass `AlmostSolved`
(worklog:3282-3288); that status does not reproduce here.

## (a) Gate reorder at the cutoff-16 correlated preparation

Artifact:
`output/local_runs/paper_v_apcm_projective_gate_reorder_cutoff16_20260914_v1/`
(`summary.json`, `gate_reorder.npz`, `progress.jsonl`).
One outer-aware invariant target solve on the one-shell outer cone
(60 Gram rows, 238 frontier moments), minimum-norm lift skipped
(`canonicalize=False`), boxed, at the 15-key entrance dictionary and phonon
cutoff 16, lambda_ep = 1.5, gamma = 0.5, drive amplitude 1, t = 0, one
evaluation.

| `accept_first_provisional` | wall seconds | status | acceptance | invariant target image quadratic | minimum scaled eigenvalue | stationarity residual | relative duality gap |
|---|---:|---|---|---:|---:|---:|---:|
| `False` (default) | 46.36 | `explicit_target:NumericalError` | rejected | 0.011286373848805029 | -2.5903862151141617e-09 | 6.411437065053453e-05 | 3.5739542823707714e-04 |
| `True` (kept gate reorder) | 45.96 | `explicit_target:NumericalError` | rejected | 0.011286373848805029 | -2.5903862151141617e-09 | 6.411437065053453e-05 | 3.5739542823707714e-04 |

Speedup 1.0087; the two solves are the same solve. Peak resident 3.80 GB
under `ram_guard --limit-mb 4000`. The record's objective for the accepted
outer-aware witness is 0.01126712315; the rejected point above differs from
it by 0.17 percent, so the solver still lands near the same minimizer but
returns no usable certificate, and `select_outer_feasible_projective_guard`
returns `witness_source="none"` with a NaN target image.

## (c) Single strictly convex solve, one-shell outer cone, cutoff-16 preparation

Artifact:
`output/local_runs/paper_v_apcm_projective_strictly_convex_cutoff16_20260914_v1/`.
The named option replaces the two-solve selector (invariant target image,
then the unique minimum-norm lift under `R_D y = q*`) by the single solve of
`0.5 q^T M q + 0.5 w ||z||^2` on the same one-shell outer cone, with `q` an
explicit decision variable, `M` the whitened target metric, `z` the
standardized outer frontier coordinates and `w` the declared weight.  The
lift equality is never formed.  Same preparation, dictionary, cutoff and
horizon as (a): 15 entrance keys, cutoff 16, lambda_ep = 1.5, gamma = 0.5,
drive amplitude 1, t = 0, one evaluation, boxed.

| `w` | wall s | Clarabel status | acceptance | invariant target image quadratic | minimum scaled eigenvalue | independent feasibility | stationarity | relative duality gap | frontier term `0.5 w ||z||^2` |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 (declared selector) | 46.0 | `NumericalError` | rejected | 0.0112863738488 | -2.5904e-09 | 2.038e-08 | 6.411e-05 | 3.574e-04 | 0 |
| 1e-4 | 125.2 | `AlmostSolved` | rejected | 0.0112761561709 | -5.2018e-14 | 4.194e-10 | 5.641e-08 | 1.242e-06 | 4.9288e-06 |
| 1e-2 | 108.9 | `AlmostSolved` | rejected | 0.0112909204656 | -7.0563e-14 | 1.185e-09 | 1.509e-08 | 1.124e-06 | 2.08528e-04 |
| 1 | 45.5 | `NumericalError` | rejected | 0.0113092408936 | -5.0680e-09 | 2.824e-08 | 7.082e-05 | 8.722e-04 | 2.05062e-02 |

Peak resident 3.84 GB under `ram_guard --limit-mb 4000`.

Reading.  At `w` = 1e-4 the frontier term moves the outer-aware target stage
from `NumericalError` with a -2.59e-09 minimum scaled eigenvalue of the
reconstructed outer Gram to `AlmostSolved` with -5.20e-14, and the
stationarity residual from 6.41e-05 to 5.64e-08, at 2.7 times the wall time.
It is still classified `rejected`, and by one residual only: the exploratory
acceptance rule (`apcm_projective_guard.py:690-700`) admits a noncertified
point when the largest of the independently recomputed feasibility,
stationarity, complementarity and relative duality-gap residuals is at most
1e-6, and the relative duality gap is 1.242e-06, a factor 1.24 above it.  At
`w` = 1 the frontier term is large enough to dominate and the solve returns
to `NumericalError`.

Objectives: the record's accepted outer-aware witness objective is
0.01126712315; all four points above lie within 0.17 percent of it, so the
selector reaches the same neighbourhood of the minimizer under every weight
and the open question is certification, not location.  No local
outer-aware witness target image exists for a displacement comparison,
because every solve at this preparation was rejected and
`select_outer_feasible_projective_guard` then returns a NaN target image; the
record's 0.5193246 displacement of the outer-aware witness from the
current-only target therefore has no local counterpart in this run.

## (b) Event-triggered outer guard for the fixed-dictionary APCM integrator

Changed files and lines.

- `paper_5/src/paper5/stability/projected_apcm.py:577-610`
  `ProjectiveOuterEventGuardSettings`, a frozen dataclass whose three floors
  carry the same names and the same default values as the carried-witness
  flow's `CWRMFSettings` fields (`apcm_carried_witness.py:81-83`):
  `event_gram_floor` 1e-8, `event_gram_hard_floor` 1e-3,
  `event_coupling_threshold` 1e-2, `event_coupling_horizon` 0.05; plus the
  two selector options `accept_first_provisional` (default `False`) and
  `frontier_weight` (default 0.0).
- `paper_5/src/paper5/stability/projected_apcm.py:612-719`
  `_CouplingGeometryAdapter` and `_CouplingModelAdapter`, the duck-typed
  surface on which the shared `negative_mode_coupling` measure, imported
  unchanged from `apcm_carried_witness.py`, is evaluated for the
  fixed-dictionary state.  The retained coordinates are the same 29 raw plus
  entrance layout the measure already assumes (its `17:29` correlation block
  and `29:` entrance block); the completion coordinates are the 62
  standardized frontier moments; the retained velocity is
  `FixedDictionaryProjectedAPCM.unprojected_velocity_with_frontier`, which
  evaluates the archive equations and the exact active commutators while
  holding the lift fixed.  The fixed dictionary propagates no completion
  velocity, so `desired_completion_velocity` is zero; that value enters only
  the reported `weakest_eigenvalue_velocity` diagnostic, never the trigger.
- `paper_5/src/paper5/stability/projected_apcm.py:721-966`
  `ProjectiveOuterEventGuard`.  Per stage: form the unconstrained
  fixed-dictionary predictor from
  `targets.augmented_retained_velocity` and `targets.auxiliary_velocity`
  (before the retained-first tangent-cone projection), carry the stage's own
  frontier witness forward unchanged to the endpoint, eigendecompose the
  endpoint scaled Gram, and compare its minimum.  Above `-event_gram_floor`
  the stage takes the declared proximal log-det completion; between the two
  floors the shared coupling measure decides against
  `event_coupling_threshold`; below `-event_gram_hard_floor` the stage fires
  unconditionally.  A fired stage calls
  `select_outer_feasible_projective_guard` at the contained trial state's own
  lower moments and propagates that witness as the stage lift.  Declared
  fallbacks, each recorded: `selector_error`, `selector_rejected`, and
  `envelope_fallback` when the witness reaches the frozen moment envelope
  `validate_envelopes` enforces.  The selector is an injected seam
  (`selector=`) so the policy can be tested without paying the conic cost.
- `paper_5/src/paper5/stability/projected_apcm.py:532-560`
  `FixedDictionaryProjectedAPCM.contain_stage` gains a keyword-only
  `completion` (default `None`, the recorded path) and validates that a
  supplied lift belongs to the same lower moments.
- `paper_5/src/paper5/stability/projected_apcm.py:1042-1070`, `1183-1196`, `1236-1262`, and `153`
  `_advance_projected_apcm_interval` and `integrate_projected_apcm_ssprk3`
  gain `outer_guard` (default `None`) and `deadline` (default `None`), and
  `ProjectedAPCMTrajectory` gains `outer_guard_triggered_stages` (default
  `None`).  With `outer_guard=None` the containment call is the original call
  with the original arguments.

Bitwise gate: `test_quiet_event_guard_reproduces_the_per_stage_default_bitwise`
integrates the declared 15-key entrance dictionary at phonon cutoff 6 for one
step at h = 0.0025 with and without the guard and asserts
`np.testing.assert_array_equal` on the state and extension-eigenvalue arrays.

## (a) continued: the option reaches the driver

`--accept-first-provisional` on
`paper_5/src/paper5/stability/apcm_projective_event_guard_analysis.py`
sets it both for `--mode gate-reorder` (which measures both settings) and
inside every triggered outer solve of `--mode rollout`.

## (d) First projective rollout

Artifact:
`output/local_runs/paper_v_apcm_projective_event_guard_t005_h0025_cutoff16_20260914_v1/`
(`summary.json`, `trajectory.npz`, `autonomous_trajectory.npz`,
`outer_guard_stages.jsonl`, `progress.jsonl`).

Fixed-dictionary APCM, SSPRK(3,3), containment before every RHS call, the
event-triggered one-shell outer guard at its declared floors, 15 entrance
keys, phonon cutoff 16, lambda_ep = 1.5, gamma = 0.5, drive amplitude 1,
t = 0 to 0.05 at h = 0.0025, scored offline against the exact cutoff-16
driven reference with `_accuracy_metrics` imported from
`apcm_carried_witness_analysis.py`.

| quantity | value | horizon and configuration |
|---|---:|---|
| autonomous wall time | 50.58 s | 20 completed steps, 66 right-hand-side evaluations, t = 0 to 0.05 |
| peak resident memory | 300 MB | whole process, under `ram_guard --limit-mb 4000` |
| stages at which the event test ran | 65 | every SSPRK stage plus subdivided stages |
| stages that triggered | 0 | every stage classified `quiet` |
| time spent in the event test itself | 0.0138 s total, 2.12e-4 s mean, 5.59e-4 s maximum per stage | 65 stages; 0.027 percent of autonomous wall time |
| minimum endpoint unshifted 24-row Gram eigenvalue of the unconstrained predictor against the carried frontier witness | -1.3847e-13 | t = 0 to 0.05, over all 65 stages |
| minimum unshifted retained joint Gram eigenvalue | 2.8921e-05 | t = 0 to 0.05, 21 stored states |
| minimum scaled positive-extension eigenvalue along the trajectory | -9.2959e-16 | t = 0 to 0.05 |
| stage retractions applied | 7 | t = 0 to 0.05 |
| all-coordinate scalar RMS error against the exact cutoff-16 reference | 7.9318e-05 | t = 0 to 0.05, 31 closed scalar coordinates, 21 sample times |
| the same quantity with the fixed canonical-embedding offset subtracted | 7.9318e-05 | the offset itself is 8.08e-16 in Euclidean norm, so the two agree |
| maximum absolute coordinate error | 3.8197e-04 | same |
| blockwise scalar RMS error, rho / B / N / A / C | 5.0153e-07 / 1.5962e-06 / 2.1728e-04 / 9.1036e-06 / 2.0152e-05 | same |
| site-0 occupation RMS error | 1.6280e-10 | same |
| static energy RMS error | 2.5751e-04 | same |

Next to the record's carried-witness strict figure for the same horizon,
all-coordinate scalar RMS 3.24e-06 over t = 0 to 0.05 (worklog:3796-3816):
the fixed-dictionary projective rollout is a factor 24.5 less accurate, and
the error is dominated by the N block (phonon-number sector) at 2.1728e-04
against 5.0153e-07 in rho.

Reading.  The event-triggered outer guard is executable: the trigger test
costs 2.12e-4 s per stage against a stage cost of about 0.77 s, and over this
horizon it never fires, because the endpoint unshifted Gram minimum of the
unconstrained predictor stays five orders of magnitude above the declared
floor `-event_gram_floor` = -1e-8.  The record's objection to the projective
route — that recomputing the outer selector at every stage is about 513 s per
step and "not an executable controller" (worklog:3267-3269) — does not apply
to the event-triggered policy on this horizon, because the selector is never
reached.  What the rollout does not yet establish is the guard's behaviour
when it does fire; see the blocker below.

## (c) continued: declared frontier weight sweep

Artifact:
`output/local_runs/paper_v_apcm_projective_strictly_convex_sweep_cutoff16_20260914_v1/`
(weights 1e-6, 1e-5, 1e-3, 1e-1), combined below with the four weights of the
run above.  Same preparation, dictionary, cutoff and horizon; the exploratory
acceptance rule admits a noncertified point when the largest of the
independently recomputed feasibility, stationarity, complementarity and
relative duality-gap residuals is at most 1e-6.

| `w` | wall s | Clarabel status | minimum scaled eigenvalue | stationarity | relative duality gap | invariant target image quadratic |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 46.0 | `NumericalError` | -2.5904e-09 | 6.411e-05 | 3.574e-04 | 0.0112863738488 |
| 1e-6 | 39.9 | `NumericalError` | -5.3974e-09 | 1.584e-04 | 8.833e-04 | 0.0113061376960 |
| 1e-5 | 125.8 | `AlmostSolved` | -4.2422e-14 | 5.495e-08 | 1.119e-06 | 0.0112710461302 |
| 1e-4 | 125.2 | `AlmostSolved` | -5.2018e-14 | 5.641e-08 | 1.242e-06 | 0.0112761561709 |
| 1e-3 | 116.0 | `AlmostSolved` | -5.2623e-13 | 8.998e-08 | 3.150e-06 | 0.0112826713025 |
| 1e-2 | 108.9 | `AlmostSolved` | -7.0563e-14 | 1.509e-08 | 1.124e-06 | 0.0112909204656 |
| 1e-1 | 125.5 | `AlmostSolved` | -1.2376e-12 | 2.912e-08 | 2.252e-06 | 0.0113005157431 |
| 1 | 45.5 | `NumericalError` | -5.0680e-09 | 7.082e-05 | 8.722e-04 | 0.0113092408936 |

For every weight between 1e-5 and 1e-1 the outer-aware target stage reaches
`AlmostSolved` with a reconstructed outer Gram whose minimum scaled
eigenvalue is of order 1e-14 to 1e-12, against -2.59e-09 for the declared
lexicographic selector, and the only residual still above the exploratory
acceptance threshold is the relative duality gap, which lands between
1.119e-06 and 3.150e-06.  Its minimum over the sweep is 1.119e-06 at
`w` = 1e-5, a factor 1.12 above 1e-6.  Outside that band, at `w` = 1e-6 and
`w` = 1, the solve returns to `NumericalError`.  Every objective lies within
0.35 percent of the record's 0.01126712315.

## (d) continued: what a triggered stage actually costs

The declared-floor rollout never fired, so the cost and outcome of a fired
stage were measured separately by lowering the two floors to
`event_gram_floor` 1e-17 and `event_gram_hard_floor` 1e-16, which makes the
roundoff-scale endpoint depth fire the hard-floor branch.  Everything else is
the declared configuration: 15 entrance keys, cutoff 16, lambda_ep = 1.5,
gamma = 0.5, drive amplitude 1, one step from t = 0 to 0.0025 at h = 0.0025,
lexicographic selector (`frontier_weight` 0).

Artifacts:
`output/local_runs/paper_v_apcm_projective_forced_trigger_cutoff16_20260914_v1/`
(killed) and
`output/local_runs/paper_v_apcm_projective_forced_trigger_cutoff16_20260914_v2/`
(completed).

| quantity | value | horizon and configuration |
|---|---:|---|
| autonomous wall time | 460.06 s | 1 step, 3 right-hand-side evaluations, t = 0 to 0.0025 |
| stages that triggered | 3 of 3 | reason `hard_floor` at every stage |
| outer selector wall time per triggered stage | 45.21 s, 45.04 s, 364.81 s | the three SSPRK stages of that one step |
| outcome of each triggered outer solve | `selector_rejected:explicit_target:NumericalError`, the same, and `selector_rejected:explicit_target:AlmostSolved` | the third stage's `AlmostSolved` first pass is what admits it to the retry ladder and costs 364.81 s |
| event test itself | 1.73e-4 to 1.75e-4 s per stage | unchanged by firing |
| peak resident memory | 5.29 GB | whole process |
| stage retractions applied | 0 | every triggered stage fell back to the declared proximal log-det completion and was already feasible |
| all-coordinate scalar RMS error against the exact cutoff-16 reference | 2.5232e-07 | t = 0 to 0.0025, one step, 2 sample times; not comparable with the 20-step figure above |

The first attempt of this measurement was killed by
`ram_guard --limit-mb 4000` at 5201 MB.  Deviation declared: the completed
run used `--limit-mb 6000`, which is the limit the plan's own step 4.1 uses
for the carried-witness anchors and is well inside the 18 GB aggregate
contract; no other heavy process of mine was running.  Nothing else about the
run changed.

Reading.  Every triggered stage was rejected, so on this machine the
event-triggered outer guard currently degrades to the declared per-stage
proximal log-det completion whenever it fires, at a cost of 45 to 365 s and
about 5.3 GB for the attempt.  The record's verdict that the one-shell
selector is too expensive and too ill-conditioned for stagewise use
(worklog:3267-3269, 3290-3316) is reproduced and sharpened: event triggering
removes the cost from the quiet path entirely but does not repair the solve
on the path where it matters.

## Gates

| gate | command | result |
|---|---|---|
| the two existing files I own plus the new one | `cd paper_5 && PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_apcm_projective_guard.py tests/test_projected_apcm.py tests/test_apcm_projective_event_guard.py` | 30 passed in 24.67 s |
| full Paper V suite under `ram_guard --limit-mb 5000` | `cd paper_5 && PYTHONPATH=src python ../pipelines/shell/ram_guard.py --limit-mb 5000 -- python -m pytest -q -p no:cacheprovider tests` | 319 passed, 9 warnings, 106.60 s, peak 1739 MB |

The two bitwise gates inside the new file:

- `test_frontier_weight_default_reproduces_the_declared_selector_bitwise`
  asserts `np.testing.assert_array_equal` between
  `select_outer_feasible_projective_guard(..., boxed=True, canonicalize=False)`
  and the same call with `accept_first_provisional=False, frontier_weight=0.0`
  spelled out, on the target image, the restricted standardized values, and
  the objective.
- `test_quiet_event_guard_reproduces_the_per_stage_default_bitwise` asserts
  `np.testing.assert_array_equal` on the state and extension-eigenvalue arrays
  of one h = 0.0025 step of the declared 15-key dictionary at cutoff 6, run
  with and without the guard.

## Artifacts

| path | what it holds |
|---|---|
| `output/local_runs/paper_v_apcm_projective_gate_reorder_cutoff16_20260914_v1/` | (a) both gate settings at the cutoff-16 preparation |
| `output/local_runs/paper_v_apcm_projective_strictly_convex_cutoff16_20260914_v1/` | (c) weights 0, 1e-4, 1e-2, 1 |
| `output/local_runs/paper_v_apcm_projective_strictly_convex_sweep_cutoff16_20260914_v1/` | (c) weights 1e-6, 1e-5, 1e-3, 1e-1 |
| `output/local_runs/paper_v_apcm_projective_event_guard_t005_h0025_cutoff16_20260914_v1/` | (d) the first projective rollout, t = 0 to 0.05 |
| `output/local_runs/paper_v_apcm_projective_forced_trigger_cutoff16_20260914_v2/` | (d) the cost and outcome of a triggered stage |
| `output/local_runs/paper_v_apcm_projective_forced_trigger_cutoff16_20260914_v1/` | the same run killed by `ram_guard --limit-mb 4000` at 5201 MB |
| `output/local_runs/paper_v_apcm_projective_gate_reorder_smoke_20260914_v2/`, `..._selector_smoke_20260914_v1/`, `..._rollout_smoke_20260914_v1/` | small-fixture smoke runs of the three driver modes |

## Kept

- The event-triggered outer guard as a named policy on the fixed-dictionary
  APCM integrator, default untouched (`outer_guard=None` is the recorded
  per-stage policy, reproduced bitwise).
- The shared negative-mode coupling measure reused verbatim from
  `apcm_carried_witness.py` through a declared adapter, so both routes fire on
  the same quantity.  `apcm_carried_witness.py` and `cone_correction.py` were
  not edited.
- `frontier_weight` on `_solve_explicit_target_qp` and
  `select_outer_feasible_projective_guard`, default 0.0, which is the
  recorded lexicographic two-solve selector.
- `accept_first_provisional` plumbed to the new driver as
  `--accept-first-provisional`.
- The `deadline` keyword on `integrate_projected_apcm_ssprk3` and
  `--wall-budget-seconds` on the driver, so an exploratory rollout reports how
  far it reached instead of being killed.
- The injected `selector=` seam on the guard, which is what lets the fired
  path be tested without a 45 s conic solve.

## Dropped

- Changing the retry ladder's trigger set to include `NumericalError`.  That
  would make `accept_first_provisional` reachable at this preparation, but it
  is a default, and the record's acceptance semantics are not mine to move.
- Raising the exploratory acceptance tolerance from 1e-6 to admit the
  strictly convex selector's 1.119e-06 relative duality gap.  Same reason:
  it is a default and a scientific acceptance rule.
- Carrying the outer-feasible witness through a stage whose retraction is
  applied.  `SymmetryReducedAPCMGeometry.retract_stage` re-solves the
  proximal log-det completion on the retracted lower moments
  (`apcm_moment_projection.py:1312-1321`), so on a retracted stage the guard's
  witness only decides whether the retraction runs.  Not changed; stated.
- A `t = 0` displacement of the strictly convex target image against a local
  outer-aware witness.  Every lexicographic solve at this preparation was
  rejected, so no local witness image exists to displace against.

## Blockers

1. The one-shell outer selector returns no accepted witness at the cutoff-16
   correlated preparation on this machine, under either gate setting and
   under every frontier weight tried.  The nearest miss is the strictly
   convex selector at `w` = 1e-5, whose relative duality gap is 1.119e-06
   against the exploratory acceptance threshold 1e-6.  Closing that factor of
   1.12 needs either a decision on the acceptance rule (the user's, not
   mine) or a different conic formulation; it is not reachable by a named
   option on the current one.
2. A triggered stage costs 45 to 365 s and about 5.3 GB.  It therefore
   exceeds the `--limit-mb 4000` guard I was given; the completed measurement
   used 6000, declared above.
3. The rollout's accuracy question is open: 7.9318e-05 all-coordinate scalar
   RMS over t = 0 to 0.05 against the carried-witness strict 3.24e-06 for the
   same horizon, with the N block carrying 2.1728e-04 of it.  Whether that
   gap is the fixed dictionary, the SSPRK(3,3) containment, or the absence of
   a carried witness is a separate measurement, not this one.

## Files I changed, and files I did not

Mine, in this worktree only:

- `paper_5/src/paper5/stability/apcm_projective_guard.py` (lines listed under
  (c) and (a); the diff is 53 lines)
- `paper_5/src/paper5/stability/projected_apcm.py` (lines listed under (b))
- `paper_5/src/paper5/stability/apcm_projective_event_guard_analysis.py` (new)
- `paper_5/tests/test_apcm_projective_event_guard.py` (new)
- `agent_guidance/paper-v/overnight/projective-event-guard-20260914.md` (this
  file)

`paper_5/src/paper5/stability/apcm_moment_projection.py` is mine to edit but
needed no change.

Not mine: `apcm_carried_witness.py`, `apcm_carried_witness_analysis.py`,
`apcm_positive_extension.py`, `cone_correction.py` and
`tests/test_apcm_carried_witness.py` are also modified in this shared
worktree by other agents.  Every number above was produced against whatever
state those files were in when the process started; the two selector runs and
both rollouts ran inside the same window, so they are mutually comparable.
Nothing outside the writable list was touched, and nothing was committed.

No `detect_changes` run: this worktree is not in the GitNexus index (only the
main checkout is), and I am not committing.
