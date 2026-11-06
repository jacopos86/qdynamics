# projective-gap (2026-09-15)

Round 2, overnight loop.  Attribute and reduce the accuracy gap of the
projective route's first rollout.  Round 1 measured, for fixed-dictionary
APCM with the event-triggered one-shell outer guard, SSPRK(3,3),
`h = 0.0025`, 15 entrance keys, phonon cutoff 16, `lambda_ep = 1.5`,
`gamma = 0.5`, drive amplitude 1, over `t = 0` to `0.05`: all-coordinate
scalar RMS `7.9318e-5` against the exact cutoff-16 driven reference (N block
`2.1728e-4`), 7 stage retractions applied, no stage triggered, against
`3.236e-6` for the strict carried-witness flow on the same horizon.

Defaults untouched: every new behaviour below is a named field of a frozen
options dataclass or a keyword whose default reproduces the committed path,
with a bitwise gate.

Status: complete.

## Worktree state at the start of this round

`git rev-parse HEAD` = `0e7bb66b6f2160a342e4716a180a512fe3dabd66` (round 1).
Working tree at start: `paper_5/notes/electron_phonon_closure_worklog_20260803.md`
modified and `paper_5/src/paper5/stability/cone_long_horizon.py` untracked,
both another agent's.  Shared modules I do not own, MD5 at start:
`apcm_positive_extension.py` `fc799d8589fb416d665e2c048d5b6498`,
`apcm_carried_witness.py` `7ba3167731ff71c743de7307d2b3af41`,
`matrix_reference.py` `e45c4d14eb2a2f1bb39eda9cfddca6b4`.

## Impact analysis (main checkout index, read-only; index 8 commits behind)

| symbol | verdict | callers |
|---|---|---|
| `_solve_explicit_target_qp` | LOW | `select_outer_feasible_projective_guard`, `select_projective_guard`, `audit_projective_preparation` |
| `_guard_acceptance` | MEDIUM | `_solve_guard_qp`, `_solve_guard_qp_on_relative_face`, `_solve_explicit_target_qp`, `_solve_explicit_target_on_relative_face`, two `solve` methods |
| `select_outer_feasible_projective_guard` | UNKNOWN -> LOW after grep | `projected_apcm.py:756`, `apcm_projective_event_guard_analysis.py:241`, `tests/test_apcm_projective_event_guard.py` |
| `contain_stage` | UNKNOWN -> LOW after grep | `apcm_adaptation.py:215`, `projected_apcm.py:964,1062`, two duck-typed fakes in `tests/test_projected_apcm.py` |
| `retract_stage` | LOW | `projected_apcm.py:568`, `tests/test_apcm_resource_safety.py:202` |
| `integrate_projected_apcm_ssprk3` | UNKNOWN -> LOW after grep | `apcm_projective_event_guard_analysis.py:545`, `tests/test_projected_apcm.py`, `tests/test_apcm_projective_event_guard.py` |

`_guard_acceptance` is MEDIUM, so its exploratory tolerance is threaded as a
keyword whose default `None` selects the module constant
`_EXPLORATORY_ACCEPTANCE_TOLERANCE = 1e-6`, and only
`_solve_explicit_target_qp` and its relative-face retry pass it; the other
five callers are untouched.

## What changed, and where

All four files are mine; the defaults are unchanged and each new behaviour is
a named field or keyword with a gate.

- `paper_5/src/paper5/stability/projected_apcm.py`
  - `ProjectedAPCMStagePolicy`, a frozen dataclass with four fields whose
    defaults are the recorded path: `stage_retraction` (True),
    `quiet_containment` (True), `completion_start` (`"warm"`) and
    `carried_completion` (False).  `FixedDictionaryProjectedAPCM.__init__`
    takes it as `stage_policy=` (default `None`, the recorded path) and keeps
    four counters in `stage_policy_counters`.
  - `FixedDictionaryProjectedAPCM._carried_completion` builds an
    `APCMExtensionResult` from the carried frontier coordinates at the trial
    point's lower moments with no conic solve, and declines by returning
    `None` unless the scaled extension matrix is positive semidefinite to ten
    times the declared conic tolerance and every standardized coordinate
    stays 100 tolerances inside the frozen box envelope.
  - `FixedDictionaryProjectedAPCM.contain_stage` gains the keyword-only
    `stage_triggered` (default `False`) and branches on the stage policy.
    With the default policy the original two statements run unchanged, which
    is what the bitwise gate checks.
  - `integrate_projected_apcm_ssprk3` gains `tolerate_stage_failure`
    (default `False`, which raises as before); at `True` a declared stage
    failure stops the rollout at the last accepted step and is reported in
    `message` with `success` false, the same shape the `deadline` path
    already used.
  - `ProjectiveOuterEventGuardSettings` gains `frontier_metric`
    (`"identity"`), `warm_refinement_passes` (`0`) and
    `exploratory_acceptance_tolerance` (`None`), validated in `__post_init__`
    and forwarded to the selector.
- `paper_5/src/paper5/stability/apcm_projective_guard.py`
  - `_exploratory_acceptance_tolerance` resolves `None` to the module
    constant `_EXPLORATORY_ACCEPTANCE_TOLERANCE = 1e-6`, so the recorded
    acceptance rule is what every caller that passes nothing still gets.
  - `_guard_acceptance` gains `exploratory_acceptance_tolerance=None`.  Only
    `_solve_explicit_target_qp` and `_solve_explicit_target_on_relative_face`
    pass it; the other five call sites are untouched.
  - `_solve_explicit_target_qp` gains `frontier_metric`,
    `warm_refinement_passes` and `exploratory_acceptance_tolerance`.  The
    frontier block is the recorded `w I` at `"identity"`; at
    `"readout_diagonal"` it is `w diag(d)` with `d` the diagonal of
    `R^T M R` in the standardized frontier coordinates, divided by its mean
    and floored at one thousandth of that mean.  A positive
    `warm_refinement_passes` runs further Clarabel passes on the displacement
    variable, each with linear term `P x` and cone right-hand side `b - A x`
    of the accumulated point, and the accumulated point is what is reported.
  - `_original_relative_duality_gap` recomputes the relative duality gap of
    `min 0.5 x^T P x` subject to `A x + s = b` directly from `(x, z)`, since
    after a refinement pass the solver's own `obj_val` and `obj_val_dual`
    belong to the displaced problem.  It replaces the reported gap only when
    a refinement pass ran.
  - `select_outer_feasible_projective_guard` forwards the same three
    keywords.
- `paper_5/src/paper5/stability/apcm_projective_event_guard_analysis.py`:
  nine new flags, each off by default: `--disable-stage-retraction`,
  `--disable-quiet-containment`, `--carried-completion`,
  `--cold-completion`, `--disable-kpd-correction`,
  `--tolerate-stage-failure`, `--frontier-metric`,
  `--warm-refinement-passes`, `--exploratory-acceptance-tolerance`.  The
  rollout summary gains `stage_policy`, `stage_policy_counters` and
  `model_settings`.
- `paper_5/tests/test_apcm_projective_event_guard.py`: eleven tests added.
  The default-path gates are
  `test_stage_policy_default_reproduces_the_recorded_rollout_bitwise`
  (`np.testing.assert_array_equal` on states, extension eigenvalues and stage
  retractions of a two-step `h = 0.0025` rollout at phonon cutoff 6, with and
  without the policy spelled out) and
  `test_frontier_metric_default_reproduces_the_identity_term_bitwise`.

`paper_5/src/paper5/stability/apcm_moment_projection.py` and
`adaptive_positive_moment.py` are mine to edit and needed no change: the
retraction ablation is a branch in `contain_stage` that never calls
`SymmetryReducedAPCMGeometry.retract_stage`, and the K/P/D switches already
exist as `ProjectedAPCMSettings`.  `apcm_carried_witness.py`,
`cone_correction.py` and `matrix_reference.py` were not edited.

## The result, first

The N-block error of the projective rollout is not a discretization error and
not a dictionary error.  It is the conic stage retraction: over `t = 0` to
`0.05` at `h = 0.0025` the N-block error against the exact cutoff-16 reference
is flat between retractions and steps up at every step in which a retraction
is applied, and it never moves otherwise.

Baseline rerun in this tree of the round-1 rollout (same command, no new
option): all-coordinate scalar RMS `7.931830738563173e-05`, blockwise
`5.015e-7 / 1.596e-6 / 2.173e-4 / 9.104e-6 / 2.015e-5` for rho / B / N / A / C,
7 stage retractions, minimum unshifted retained joint-Gram eigenvalue
`2.892080456917745e-05`, 54.04 s.  Every printed digit matches round 1's
`7.9318e-5`, so the concurrent edits to `apcm_positive_extension.py`,
`apcm_carried_witness.py` and `matrix_reference.py` did not move the recorded
path.  Artifact:
`output/local_runs/paper_v_apcm_projective_ablation_baseline_t005_20260915_v1/`.

Euclidean norm of the error of each closed scalar block at each stored sample
time, next to the number of stage retractions applied in the step that
produced it, baseline run, 15 entrance keys, phonon cutoff 16,
`lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude 1, `h = 0.0025`:

| t | retractions | min scaled extension eigenvalue | rho | B | N | A | C |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0150 | 0 | -4.619e-16 | 5.128e-8 | 1.632e-6 | 1.685e-8 | 1.149e-5 | 1.671e-5 |
| 0.0175 | 0 | -4.958e-16 | 8.798e-8 | 1.904e-6 | 2.250e-8 | 1.341e-5 | 2.198e-5 |
| 0.0200 | 1 | 1.244e-7 | 1.387e-7 | 2.176e-6 | 2.282e-4 | 1.532e-5 | 2.976e-5 |
| 0.0225 | 2 | 1.365e-5 | 2.267e-7 | 2.466e-6 | 4.324e-4 | 1.723e-5 | 3.759e-5 |
| 0.0250 | 0 | 1.365e-5 | 3.091e-7 | 2.736e-6 | 4.324e-4 | 1.914e-5 | 4.524e-5 |
| 0.0275 | 1 | 1.590e-5 | 4.192e-7 | 3.012e-6 | 4.893e-4 | 2.104e-5 | 5.377e-5 |
| 0.0300 | 0 | 1.590e-5 | 5.312e-7 | 3.282e-6 | 4.893e-4 | 2.294e-5 | 6.259e-5 |
| 0.0325 | 1 | 2.185e-5 | 6.772e-7 | 3.559e-6 | 5.635e-4 | 2.484e-5 | 7.243e-5 |
| 0.0400 | 1 | 2.459e-5 | 1.166e-6 | 4.370e-6 | 6.046e-4 | 3.051e-5 | 1.047e-4 |
| 0.0425 | 1 | 2.875e-5 | 1.379e-6 | 4.642e-6 | 6.600e-4 | 3.240e-5 | 1.170e-4 |
| 0.0500 | 0 | 2.875e-5 | 2.033e-6 | 5.454e-6 | 6.600e-4 | 3.805e-5 | 1.570e-4 |

Before the first retraction the N-block error is `2.250e-8`; the first
retraction raises it to `2.282e-4` in one step, and each later retraction adds
another step.  The rows at `t = 0.025`, `0.030`, `0.035`, `0.0375`, `0.045`,
`0.0475` and `0.050` carry no retraction and their N-block error equals the
previous row's to every printed digit.  The rho, B, A and C blocks grow
smoothly across the same rows, so those are consistency errors of the closure
and the N block is not.

## (a) Attribution by named ablation

All rows: fixed-dictionary APCM, SSPRK(3,3), event-triggered one-shell outer
guard at its declared floors, phonon cutoff 16, `lambda_ep = 1.5`,
`gamma = 0.5`, drive amplitude 1, `t = 0` to `0.05`, 15 entrance keys and
`h = 0.0025` unless the row says otherwise, scored against the exact
cutoff-16 driven reference regenerated in process.  "joint Gram" is the
minimum unshifted retained joint-Gram eigenvalue over the stored states.
"carried" is the number of stage lifts taken from the previous stage without
a solve, out of the number of stage lifts selected.

| run | all-coordinate scalar RMS | rho | B | N | A | C | joint Gram | retractions | s | reached t | carried |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (recorded path) | 7.9318e-05 | 5.0153e-07 | 1.5962e-06 | 2.1728e-04 | 9.1036e-06 | 2.0152e-05 | 2.892080e-05 | 7 | 54.0 | 0.0500 | 0/0 |
| stage retraction disabled | 1.9408e-06 | 1.2971e-08 | 4.9028e-07 | 4.2577e-09 | 2.8199e-06 | 2.2054e-06 | 2.892080e-05 | 0 | 23.4 | 0.0150 | 0/0 |
| containment disabled on quiet stages | 1.9408e-06 | 1.2971e-08 | 4.9028e-07 | 4.2577e-09 | 2.8199e-06 | 2.2054e-06 | 2.892080e-05 | 0 | 23.3 | 0.0150 | 0/0 |
| K/P/D entrance correction disabled | 2.1281e-03 | 1.3235e-04 | 1.5900e-06 | 1.1436e-04 | 8.6939e-05 | 3.1650e-03 | 2.892080e-05 | 5 | 44.1 | 0.0500 | 0/0 |
| 13 entrance keys | 6.2000e-05 | 6.3606e-08 | 1.5948e-06 | 1.7073e-04 | 9.1285e-06 | 1.2117e-05 | 2.892080e-05 | 7 | 38.6 | 0.0500 | 0/0 |
| 14 entrance keys | 9.2993e-05 | 2.0275e-07 | 1.6024e-06 | 2.5812e-04 | 9.1153e-06 | 8.7551e-06 | 2.892080e-05 | 9 | 60.6 | 0.0500 | 0/0 |
| `h = 0.00125` | 1.5805e-04 | 2.3796e-07 | 1.6121e-06 | 4.3878e-04 | 8.9742e-06 | 1.6487e-05 | 2.892080e-05 | 46 | 132.2 | 0.0500 | 0/0 |

Readings, in order of how much each moves the N block.

1. **The single change that moves the N block is the stage retraction.**  With
   it disabled the N-block scalar RMS over the horizon the ablation survives
   is `4.2577e-09`; the recorded path over the same seven sample times gives
   `4.2577e-09` as well, because no retraction has fired yet by `t = 0.015`.
   The whole of the recorded path's `2.1728e-04` is produced after `t = 0.015`
   and only in steps that apply a retraction, as the table in the previous
   section shows step by step.
2. **Halving the step makes the N block worse, by the ratio of the retraction
   counts.**  At `h = 0.00125` the retraction count rises from 7 to 46 and the
   N block from `2.1728e-04` to `4.3878e-04`, while rho falls from
   `5.0153e-07` to `2.3796e-07` and C from `2.0152e-05` to `1.6487e-05`.  A
   third-order SSPRK truncation error would have fallen by eight.  So rho, B,
   A and C are consistency errors of the closure and N is not a
   discretization error at all.
3. **Containment on a quiet stage is exactly its retraction.**  The two
   ablation rows agree in every reported digit, including the failure message
   and the horizon reached.  The difference between them is the frozen
   moment-envelope validation and the base/extension feasibility test, and
   neither fired over this horizon, so neither changes a number.  What is left
   of containment, the declared proximal log-det completion at the trial
   point, is not removable: the next right-hand-side evaluation needs a lift
   there.
4. **The retraction is what keeps the rollout alive.**  Both ablations stop
   after 6 of 20 steps with `moment_extension_failure persisted after 8 stage
   subdivisions`: the trial point's lower moments stop admitting the declared
   proximal log-det completion (`relative face support 23/24, directions
   24/62`, certificate `AlmostSolved`, `relative_lambda = 1.114e-05`).  On the
   recorded path the retraction moves the point back inside and the completion
   succeeds again.  So the retraction cannot simply be switched off; it has to
   stop being reached.
5. **The K/P/D entrance correction is not the gap; it is what makes rho and C
   accurate.**  Switching all three of `include_k`, `include_pauli` and
   `include_opposite_spin` off raises the all-coordinate scalar RMS by a
   factor 26.8 to `2.1281e-03`, with C up by a factor 157 to `3.1650e-03` and
   rho up by a factor 264 to `1.3235e-04`.  The N block falls, from
   `2.1728e-04` to `1.1436e-04`, and the retraction count falls from 7 to 5,
   which is the same N-versus-retraction relation and not an improvement of
   the closure.
6. **The dictionary cannot be truncated to the first 5 or 10 entrance keys.**
   At 5 keys the cutoff-16 correlated preparation is refused before the first
   step: `initial moment_extension_failure: moment_extension_degeneracy: conic
   Newton step failed descent after relative face support=14/14,
   directions=11/11`.  At 10 keys the first right-hand-side evaluation raises
   `required relative-mode moment is missing: MomentKey(spin_up='I',
   spin_down='Y', x_power=1, p_power=1)`, which is entrance key 11, because
   the exact K/P/D entrance correction resolves it.  At 12 keys the
   preparation is refused again (`relative face support=20/21,
   directions=13/48`).  The two admissible reductions are 13 and 14 keys, and
   their all-coordinate scalar RMS is `6.2000e-05` and `9.2993e-05` against
   the 15-key `7.9318e-05`, with retraction counts 7, 9 and 7 and N blocks
   `1.7073e-04`, `2.5812e-04` and `2.1728e-04`.  The dictionary therefore
   changes the accuracy only through how often the retraction fires; it is not
   the gap.

Artifacts: `output/local_runs/paper_v_apcm_projective_ablation_*_20260915_v1/`
(`baseline_t005`, `no_retraction_tolerant_t005`,
`no_quiet_containment_tolerant_t005`, `no_kpd_t005`, `keys13_t005`,
`keys14_t005`, `h000125_t005`, and the three refused runs `keys5_t005`,
`keys10_t005`, `keys12_t005`, plus `no_retraction_t005` and
`no_quiet_containment_t005` which raised before `--tolerate-stage-failure`
existed).

## (b) The cheapest hybrid: carry the completion coordinates as the lift

Negative result first: the recorded path already warm starts every stage
completion from the previous stage's frontier moments
(`contain_stage(..., warm_frontier=evaluation.targets.completion.frontier_moments)`),
so "warm completion" in the sense of a starting point is not a change.  What
is available without editing `apcm_carried_witness.py` is to carry the
previous stage's completion coordinates as the lift itself: accept them at the
trial point's lower moments without a solve whenever the scaled extension
matrix they build there is positive semidefinite inside the frozen box
envelope, and fall back to the declared proximal log-det completion otherwise.
That is `ProjectedAPCMStagePolicy(carried_completion=True)`,
`--carried-completion`.

Same configuration and scoring as the table above.

| run | all-coordinate scalar RMS | rho | B | N | A | C | retractions | s | reached t | carried |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 7.9318e-05 | 5.0153e-07 | 1.5962e-06 | 2.1728e-04 | 9.1036e-06 | 2.0152e-05 | 7 | 54.0 | 0.0500 | 0/0 |
| carried, `h = 0.0025` | 3.5137e-05 | 4.7368e-07 | 1.5897e-06 | 8.9775e-05 | 9.1227e-06 | 1.9864e-05 | 1 | 42.0 | 0.0500 | 36/66 |
| carried, margin multiple 0 | 3.5137e-05 | 4.7368e-07 | 1.5897e-06 | 8.9775e-05 | 9.1227e-06 | 1.9864e-05 | 1 | 44.8 | 0.0500 | 36/66 |
| carried, `h = 0.00125` | 1.2875e-05 | 1.5239e-08 | 1.5799e-06 | 1.8296e-05 | 9.0844e-06 | 1.5341e-05 | 1 | 14.2 | 0.0500 | 113/120 |
| carried, `h = 0.000625` | 1.4635e-05 | 1.2924e-08 | 1.5750e-06 | 1.8768e-05 | 9.0562e-06 | 1.8378e-05 | 1 | 10.1 | 0.0500 | 236/240 |
| cold completion (control) | 5.1100e-05 | 4.7630e-07 | 1.5916e-06 | 1.3683e-04 | 9.1165e-06 | 1.9904e-05 | 4 | 87.0 | 0.0500 | 0/0 |

Readings.

1. **Carrying the lift reduces the all-coordinate scalar RMS from
   `7.9318e-05` to `1.2875e-05`, a factor 6.16, and it is cheaper: 14.2 s
   against 54.0 s.**  At `h = 0.00125` the carried lift is accepted at 113 of
   120 stages, so only 7 proximal log-det completions are solved over the
   whole rollout instead of 120.  Against the carried-witness strict figure
   `3.2359e-06` on the same horizon, the projective route is now a factor
   3.98 away instead of 24.5.
2. **The mechanism is the retraction count, not the lift's own accuracy.**
   Carrying reduces the retraction count from 7 to 1 at `h = 0.0025` and the N
   block from `2.1728e-04` to `8.9775e-05`.  The single remaining retraction
   fires near `t = 0.02`, the N-block error steps once from `2.250e-08` to
   `2.282e-04` in that step and is then exactly flat through `t = 0.05`.
3. **With the lift carried, refining the step now converges instead of
   diverging.**  On the recorded path halving the step raised the
   all-coordinate scalar RMS from `7.9318e-05` to `1.5805e-04`; with the lift
   carried it falls from `3.5137e-05` to `1.2875e-05`.  Quartering it again
   gives `1.4635e-05`, so the closure's own consistency error floor over this
   horizon is about `1.3e-05`, carried between the N, C and A blocks
   (`1.83e-05`, `1.53e-05`, `9.08e-06`).
4. **The declared completion is not a function of its lower moments at the
   accuracy being measured.**  Solving it from the origin instead of from the
   carried frontier moments (`--cold-completion`, the same strictly convex
   proximal log-det problem to the same declared conic tolerance) changes the
   all-coordinate scalar RMS from `7.9318e-05` to `5.1100e-05` and the
   retraction count from 7 to 4, at 87.0 s against 54.0 s.  A selector whose
   solution moves the score by 35 percent when only its starting point
   changes is not delivering a determined lift; this is the same instability
   the retraction then has to repair.
5. **The acceptance margin does not discriminate.**  Accepting any positive
   semidefinite carried lift inside the envelope
   (`--carried-completion-margin-multiple 0.0`) accepts the same 36 of 66
   stages and reproduces every reported digit of the declared margin.
6. **Over a longer horizon the carried lift runs out.**  `h = 0.00125` to
   `t = 0.5` reproduces the `t = 0.05` figure `1.2875122500666576e-05`
   exactly and then stops after 91 of 400 steps at `t = 0.11375` with
   `extended operator moment matrix numerical face is unstable under
   null-tolerance refinement`, 293 of 300 lifts carried, 1 retraction, 14.8 s.
   Over `t = 0` to `0.11375` its all-coordinate scalar RMS is `5.8243e-05`
   with C at `8.5008e-05` and N at `1.8664e-05`.

Artifacts:
`output/local_runs/paper_v_apcm_projective_hybrid_carried_completion_t005_20260915_v1/`,
`..._hybrid_carried_margin0_t005_...`, `..._hybrid_carried_h000125_t005_...`,
`..._hybrid_carried_h00625em4_t005_...`, `..._hybrid_carried_h000125_t05_...`,
`..._control_cold_completion_t005_...`.

## (c) The strictly convex selector's relative duality gap

All measurements are on the one-shell outer cone (60 Gram rows, 238 frontier
moments) at the cutoff-16 correlated preparation, 15 entrance keys,
`lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude 1, `t = 0`, boxed,
`canonicalize=False`, frontier weight `w = 1e-5`, one evaluation.  The
exploratory acceptance rule admits a noncertified point when the largest of
the independently recomputed feasibility, stationarity, complementarity and
relative duality-gap residuals is at most the acceptance tolerance.

| named option | wall s | status | acceptance | objective (invariant target image quadratic) | min scaled eigenvalue | stationarity | relative duality gap | peak GB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| acceptance tolerance 2e-6 | 138.7 | `AlmostSolved` | provisional_independent_kkt | 0.01127104613019021 | -4.2422e-14 | 5.4955e-08 | 1.1194e-06 | 3.50 |
| `frontier_metric="readout_diagonal"` | 50.2 | `NumericalError` | rejected | 0.01129774686986394 | -3.3612e-09 | 1.1869e-04 | 6.6179e-04 | 3.06 |
| `warm_refinement_passes=1` | 271.7 | `AlmostSolved+warmAlmostSolved` | rejected | 0.01127107028205352 | -3.9375e-13 | 1.7207e-07 | 3.4755e-06 | 3.46 |
| round 1, no new option | 125.8 | `AlmostSolved` | rejected | 0.0112710461302 | -4.2422e-14 | 5.4955e-08 | 1.1194e-06 | 3.84 |

1. **With the acceptance tolerance set to 2e-6, an accepted witness exists at
   the preparation.**  The same solve round 1 classified rejected is now
   `provisional_independent_kkt`, and
   `select_outer_feasible_projective_guard` returns
   `witness_source="strictly_convex_outer_target_stage"` with the
   restriction to the current cone feasible to roundoff: minimum scaled
   eigenvalue of the restricted 24-row Gram `-9.7955e-16`.  The solve itself
   is unchanged; only the classification moved.  Its objective
   0.01127104613019021 is 0.035 percent from the record's 0.01126712315.
2. **At a triggered stage's own lower moments no witness is accepted, and the
   acceptance tolerance cannot change that.**  With the floors lowered as in
   round 1 (`event_gram_floor` 1e-17, `event_gram_hard_floor` 1e-16),
   `w = 1e-5` and the acceptance tolerance 2e-6, one step from `t = 0` to
   `0.0025` fired all three stages and all three returned
   `selector_rejected:explicit_target:NumericalError`.  Provisional
   acceptance requires the status `AlmostSolved`, so a `NumericalError` is
   outside the rule at any tolerance.  What the strictly convex selector does
   buy is cost: 46.42, 43.14 and 58.75 s per triggered stage and peak
   resident 3.90 GB, against round 1's 45.21, 45.04 and 364.81 s and 5.29 GB
   at `w = 0`, because a positive frontier weight skips the relative-face
   retry.  This run fits inside the 5000 MB guard; round 1's did not.  Every
   triggered stage fell back to the declared proximal log-det completion, no
   retraction was applied, and the one-step all-coordinate scalar RMS was
   2.5232307679793967e-07, the value round 1 measured.
3. **Neither way of improving the gap without touching acceptance works.**
   Scaling the frontier term by the Jacobi diagonal of the invariant target
   image quadratic takes the solve from `AlmostSolved` with gap 1.1194e-06 to
   `NumericalError` with gap 6.6179e-04 and a reconstructed outer Gram
   minimum scaled eigenvalue of -3.3612e-09.  A second Clarabel pass warm
   started from the first solution raises the gap from 1.1194e-06 to
   3.4755e-06, the stationarity residual from 5.4955e-08 to 1.7207e-07, and
   the wall time from 138.7 s to 271.7 s.  On the two-key cutoff-6 fixture
   the same refinement improves the gap slightly (8.5448e-08 to 8.0885e-08 at
   `w = 1`), so the second pass is not wrong; at this preparation the
   displaced problem is simply no better conditioned than the original.

Artifacts:
`output/local_runs/paper_v_apcm_projective_selector_acceptance2em6_w1em5_20260915_v1/`,
`..._selector_readout_diagonal_w1em5_...`,
`..._selector_warm_refinement_w1em5_v2_...`,
`..._forced_trigger_acceptance2em6_w1em5_...`.  The first
`selector_warm_refinement_w1em5` run raised `TypeError: float() argument must
be a string or a real number, not 'list'` in the driver after a 306.8 s
solve: the refinement branch shadowed the local name holding
`float(solution.r_dual)` with the dual vector `solution.z`.  Fixed, and the
refinement test now asserts both reported residuals are finite floats.

## (d) Which explanation the measurements support

Of the three candidates the measurements support the SSPRK(3,3) stage
retraction, and through it the per-stage re-selection of the completion.
Neither of the other two survives on its own.

- *The fixed dictionary.*  Not supported.  The two admissible reductions of
  the entrance dictionary, 13 and 14 keys, give all-coordinate scalar RMS
  6.2000e-05 and 9.2993e-05 against the 15-key 7.9318e-05, and their N blocks
  track their retraction counts (7, 9, 7).  Reductions to 5, 10 and 12 keys
  are refused by the preparation or by the exact K/P/D entrance correction,
  so the dictionary is not a free parameter here at all.
- *The SSPRK(3,3) containment or retraction.*  Supported, and it is the
  retraction specifically.  The N-block error is 2.250e-08 before the first
  retraction, steps to 2.282e-04 in the step that applies it, is exactly flat
  in every step that applies none, and steps again at every step that does.
  Halving the step raises the retraction count from 7 to 46 and the N block
  from 2.1728e-04 to 4.3878e-04, the opposite of a third-order truncation
  error.  Containment on a quiet stage contributes nothing else: with the
  retraction disabled, disabling containment as well reproduces every digit.
- *The absence of a carried witness.*  Supported as the cause, not as the
  proximate error.  The retraction fires because the per-stage proximal
  log-det completion, re-solved at each trial point, leaves the trial outside
  the extension cone; the cold-start control shows the same completion
  problem solved from a different starting point moves the score by 35
  percent, so the lift is not determined at the accuracy being measured.
  Carrying the previous stage's completion coordinates as the lift cuts the
  retraction count from 7 to 1 and the all-coordinate scalar RMS from
  7.9318e-05 to 3.5137e-05, and at `h = 0.00125` to 1.2875e-05 in 14.2 s
  against 54.0 s.

For the record, in one sentence: the projective route's accuracy gap over
`t = 0` to `0.05` is the conic stage retraction repairing a per-stage
re-solved positive completion, and carrying that completion instead of
re-solving it removes six of the seven retractions, reduces the
all-coordinate scalar RMS against the exact cutoff-16 reference from
7.9318e-05 to 1.2875e-05 at `h = 0.00125`, and is 3.8 times faster.

## Gates

| gate | result |
|---|---|
| the four test files I own, run together | 44 passed in 53.67 s (34 before this round) |
| the whole `paper_5` suite under the 5000 MB guard | 376 passed, 9 warnings in 139.14 s, peak 1505 MB |
| recorded-path reproduction | the baseline rerun reproduces round 1's 7.931830738563173e-05, its five blockwise values, 7 retractions and joint-Gram minimum 2.892080456917745e-05 to every printed digit |
| bitwise default gates | `test_stage_policy_default_reproduces_the_recorded_rollout_bitwise` and `test_frontier_metric_default_reproduces_the_identity_term_bitwise`, plus round 1's two gates, all pass |

## Kept

- `ProjectedAPCMStagePolicy` with `stage_retraction`, `quiet_containment`,
  `completion_start`, `carried_completion` and
  `carried_completion_margin_multiple`; the default instance is the recorded
  path bitwise.
- The carried completion as a named option.  It is the one change that
  reduces both the error and the cost, and it is off by default.
- `tolerate_stage_failure` on `integrate_projected_apcm_ssprk3` and the
  matching driver flag, which is what let the two failing ablations be scored
  over the horizon they survive instead of only raising.
- `exploratory_acceptance_tolerance` threaded to the outer target solve only;
  default `None`, so the recorded 1e-6 is what every other caller gets.
- `frontier_metric` and `warm_refinement_passes`, kept because their
  measurements are negative results the record should carry, not because they
  help.

## Dropped

- Raising the exploratory acceptance tolerance as a default.  It is a
  scientific acceptance rule and the user's to move.  What is now measured is
  that at 2e-6 the preparation's witness is accepted and a triggered stage's
  is still not, because the triggered stage returns `NumericalError` rather
  than `AlmostSolved`.
- `frontier_metric="readout_diagonal"` as a recommendation: it makes the gap
  worse by a factor 591.
- `warm_refinement_passes` as a recommendation: it makes the gap worse by a
  factor 3.1 at twice the wall time.
- Reducing the entrance dictionary: 5, 10 and 12 keys are inadmissible, and
  13 and 14 change the score only through the retraction count.

## Blockers

1. The carried-lift rollout stops at `t = 0.11375` of `t = 0.5` at
   `h = 0.00125` with `extended operator moment matrix numerical face is
   unstable under null-tolerance refinement`.  Carrying the lift removes the
   retraction but not the eventual degeneracy of the completion problem, so a
   long-horizon projective rollout still needs the completion coordinates to
   be propagated by their own velocity and repaired conically, which is the
   carried-witness flow's construction and lives in a file this round was
   told not to edit.
2. No accepted outer witness exists at a triggered stage's lower moments
   under any option tried, at either acceptance tolerance.  The obstruction is
   the Clarabel status `NumericalError`, not the size of any residual.

## Comparability in a shared worktree

`apcm_carried_witness.py` and `matrix_reference.py` were changed by other
agents during my measurement window (MD5 `7ba31677...` to `9335b8f9...` and
`e45c4d14...` to `e6240931...`).  `matrix_reference.py` supplies
`matrix_state_to_closed_scalar_coordinates`, which every score above is
computed through, so the baseline was rerun at the end of the round under the
tree as it then stood:
`output/local_runs/paper_v_apcm_projective_ablation_baseline_recheck_t005_20260915_v1/`
gives all-coordinate scalar RMS 7.9318e-05, blockwise
5.0153e-07 / 1.5962e-06 / 2.1728e-04 / 9.1036e-06 / 2.0152e-05, 7
retractions, joint-Gram minimum 2.892080e-05, 52.5 s.  Every reported digit
matches both round 1 and my opening baseline, so all rows in this report are
mutually comparable and comparable with round 1.

## Files I changed, and files I did not

Mine, in this worktree only, nothing committed:

- `paper_5/src/paper5/stability/projected_apcm.py`
- `paper_5/src/paper5/stability/apcm_projective_guard.py`
- `paper_5/src/paper5/stability/apcm_projective_event_guard_analysis.py`
- `paper_5/tests/test_apcm_projective_event_guard.py`
- `agent_guidance/paper-v/overnight/projective-gap-20260915.md` (this file)
- run directories `output/local_runs/paper_v_apcm_projective_*_20260915_v1/`

`paper_5/src/paper5/stability/apcm_moment_projection.py` and
`adaptive_positive_moment.py` are mine to edit and needed no change.
`apcm_carried_witness.py`, `cone_correction.py` and `matrix_reference.py`
were not edited by me; their modifications in this worktree are other agents'.
Nothing outside the section 7 writable list was touched.

No `detect_changes` run: this worktree is not in the GitNexus index (only the
main checkout is), and nothing was committed.
