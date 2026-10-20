# boson-barrier-t100 (2026-09-15)

Plan step 9, `agent_guidance/paper-v/paper-v-state-and-plan-20260914.md:276-282`:
the calculation the working notes declared next (notes:1868-1872) and never
ran. Branch `claude/paper-v-routes-20260914`, worktree
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`.
Not committed; the parent session commits.

## What was run

Three lanes on one sample grid, `t = 0` to 100 sampled every 0.05, strong
coupling `lambda_ep = 1.5`, `gamma = 0.5`, drive amplitude 1, pulse width 1,
single pulse, exact ground-state contractions at phonon cutoff 16, DOP853 with
`rtol = 1e-9`, `atol = 1e-11`, maximum step 0.05:

| lane | right-hand side |
|---|---|
| `corrected` | `closed_cone_projected_rhs(..., energy_neutral=True)`: the ten-coordinate energy-neutral structured boson matrix barrier on derivative coordinates 7--16, `activation_margin = 1e-5`, `target_flux = 0`, `barrier_rate = 5` |
| `raw` | `closed_residual_subtracted_rhs`: the uncorrected Eq. (112) archive EOM, the same right-hand side the corrected lane corrects and the one `report._cone_correction_audit` calls `correction_mode="none"` |
| `exact` | `exact_holstein_driven_trajectory` at phonon cutoff 16, contracted to the same 31 coordinates |

A fourth lane was added after the polarization comparison identified a
definitional difference (finding 3 below): `plain_raw`, the plain archive EOM
`closed_scalar_rhs` with no Eq. (112) initial-residual subtraction.

Artifacts:
`output/local_runs/paper_v_boson_barrier_t100_20260915_v1/`
(`summary.json`, `{exact,raw,corrected,plain_raw}_trajectory.npz`,
`polarization_spectrum.json`, `plain_archive_lane.json`, `driver.log`, and the
three post-processing scripts `report_tables.py`, `polarization_spectrum.py`,
`plain_archive_lane.py`).

## Definitions used for every number below

- **complete 31-coordinate l2 time-RMS distance from the exact reference**:
  `sqrt(trapezoid(||lane(t) - exact(t)||_2^2, t) / (t_stop - t_start))` on the
  unscaled closed scalar coordinates, the definition of
  `score_coupling_controller_exact_reference._time_rms` applied to
  `coordinate_l2`.
- **all-31 scaled RMS**: `sqrt(mean over samples and over the 31 coordinates of
  ((lane - exact) / scale)^2)`, the convention of the adaptive-packet line
  (`analyze_adaptive_packet_event_ablation.py:239`). The scales are
  `multi_coherent_scores.development_coordinate_scales` recomputed here from the
  exact cutoff-16 path over `t = 0` to 100 at cutoff 16; the packet route's
  `0.063649` used scales frozen from its own development path, so the two
  numbers share a formula and not a scale vector.
- **observable RMS** (site-0 occupation, electronic, phonon, electron--phonon
  and total internal energy): the six columns of
  `plot_high_coupling_controller_observables.observable_trajectories`. Both the
  unweighted sample RMS (the packet line's convention) and the trapezoidal
  time-RMS are reported; the packet line is a sample RMS.
- **minimum bosonic eigenvalue** is `min_t lambda_min(M_B(t))`
  (`closed_boson_moment_eigenvalues`), **minimum electronic density
  eigenvalue** is `min_t lambda_min(rho(t))` (`closed_electron_eigenvalues`).
- **maximum post-pulse drift from E(4)**: `max_{t >= 4} |E(t) - E(4)|` on the
  stored samples, `E` the matrix internal energy, exactly
  `report._cone_correction_audit`'s `post_pulse_max_drift_from_t4`.
- **fraction of sampled times with the barrier active**: the correction is
  re-solved at every stored sample from the uncorrected velocity there and a
  sample counts as active when its correction norm exceeds `1e-10`, exactly
  `report._cone_correction_audit`'s `active_sample_fraction`. The working
  notes' `55.11` percent is 221 of the 401 samples of the `0.05` grid on
  `t = 0` to 20.


## Negative results first

1. **The ten-coordinate energy-neutral barrier does not repair the accuracy of
   the 31-coordinate archive closure.** On the matched window `t = 0` to
   79.95, the all-31 scaled RMS against the exact cutoff-16 reference is
   `0.700816` corrected against `0.713667` for the Eq. (112) archive EOM it
   corrects: a 1.8 percent reduction. The complete 31-coordinate l2 time-RMS
   distance is `1.016703` against `1.549827`. The adaptive packet route's line
   on `t = 0` to 98 is `0.063649` in the same scaled convention, 11 times
   smaller than the corrected barrier's. What the barrier does deliver is
   containment: the minimum bosonic eigenvalue over `t = 0` to 79.95 is
   `2.893142825017192e-5` corrected against `-4.707881e-1` raw, and the
   maximum post-pulse drift from `E(4)` is `2.53904e-6` against `2.41557e-6`,
   so direct energy neutrality holds at the same order as the uncorrected run.

2. **The corrected lane did not reach `t = 100`.** It was stopped by the task's
   25-minute rule at `t = 79.95` after `1500.07 s`, having taken 20,359 DOP853
   steps and 328,358 right-hand-side evaluations. Cost per unit time grows
   through the post-pulse interval: 375 evaluations per unit time over `t = 0`
   to 2 (749 evaluations, `0.23 s`, round 1) against 4,107 per unit time over
   `t = 0` to 79.95. The same interval costs the uncorrected Eq. (112) lane
   `4.37 s` and 30,041 evaluations: the barrier is 343 times the wall time at
   11 times the evaluation count, so about 31 times the cost per evaluation.

3. **The record's polarization "raw" lane is not the Eq. (112) lane.** The
   working notes' pinned table and `report._cone_correction_audit` use
   `closed_residual_subtracted_rhs` for "Eq. (112), no cone correction", and my
   `raw` lane reproduces all four of that table's columns exactly (below). The
   polarization comparison at `10 <= t <= 100` (worklog:2934-2941) instead uses
   the plain archive EOM: propagating `closed_scalar_rhs` with no initial-
   residual subtraction on the same grid and integrator reproduces the record's
   Hellinger distance `0.6381` (measured `0.63807363`) and its electronic-band
   weight `0.5225` (measured `0.52250499`), while the Eq. (112) lane gives
   `0.495458` and `0.044160` on the same window. Both are named separately
   below; `closed_cone_projected_rhs` carries `subtract_initial_residual=True`
   by default, so the corrected lane sits on the Eq. (112) base.

## The working notes' pinned `t = 0` to 20 table, reproduced

`MATH/paper_facing/paper_V_high_u_gkba/stability_quantum_workflow_working_notes.md:1858-1866`,
401 samples of the `0.05` grid.

| protocol | min lambda(M_B) | min lambda(rho) | max abs coordinate | max post-pulse drift from E(4) | active sample fraction | max correction norm |
|---|---:|---:|---:|---:|---:|---:|
| notes, Eq. (112) no correction | `-3.05033e-1` | `5.51945e-2` | `1.97874` | `2.41557e-6` | not printed | not printed |
| this run, `raw` | `-3.0503297e-1` | `5.5194495e-2` | `1.9787388` | `2.4155657e-6` | `0.733167` | `1.685048` |
| notes, energy-neutral barrier | `2.89314e-5` | `4.88418e-2` | `1.97408` | `2.53904e-6` | `55.11` percent | `0.59424` |
| this run, `corrected` | `2.893142825017192e-5` | `4.8841811e-2` | `1.9740781` | `2.5390378e-6` | `0.5511221945137157` | `0.5942358` |

`0.5511221945137157` is 221 of 401 samples, the notes' `55.11` percent to every
printed digit. The `raw` active sample fraction is a new quantity, not one of
the notes' numbers: it is the fraction of the *uncorrected* trajectory's
samples at which the energy-neutral correction would be active if it were
applied there, and 7 of its 401 subproblems did not converge (168 of 2001 over
`t = 0` to 100). The corrected lane's own census has 0 nonconverged
subproblems over `t = 0` to 20 and 16 of 1600 over `t = 0` to 79.95.

## Scores against exact driven-dimer propagation at phonon cutoff 16

Window `t = 0` to 20, 401 samples. Observable entries are the unweighted
sample RMS of the difference from the exact lane.

| quantity | `raw` (Eq. 112) | `corrected` | `plain_raw` |
|---|---:|---:|---:|
| complete 31-coordinate l2 time-RMS distance | `1.025151` | `0.923947` | `1.075663` |
| all-31 scaled RMS | `0.602005` | `0.585856` | `0.969447` |
| site-0 occupation RMS | `0.154372` | `0.138356` | `0.287318` |
| electronic energy RMS | `0.173935` | `0.170802` | `0.385465` |
| phonon energy RMS | `0.363118` | `0.376097` | `0.150325` |
| electron--phonon energy RMS | `0.477545` | `0.469412` | `0.366308` |
| total internal energy RMS | `0.043401` | `0.043398` | `0.018221` |
| minimum bosonic eigenvalue | `-3.050330e-1` | `2.893143e-5` | `-7.081170e-1` |
| minimum electronic density eigenvalue | `5.519450e-2` | `4.884181e-2` | `-5.482215e-2` |
| maximum post-pulse drift from E(4) | `2.415566e-6` | `2.539038e-6` | `7.081733e-6` |
| fraction of sampled times with the barrier active | `0.733167` | `0.551122` | not measured |

The exact lane's own minimum electronic density eigenvalue on this window is
`3.282317e-2`.

Window `t = 0` to 79.95, the horizon the corrected lane reached, 1600 samples,
all three lanes on the same window:

| quantity | `raw` (Eq. 112) | `corrected` | `plain_raw` |
|---|---:|---:|---:|
| complete 31-coordinate l2 time-RMS distance | `1.549827` | `1.016703` | `2.050020` |
| all-31 scaled RMS | `0.713667` | `0.700816` | `1.585195` |
| site-0 occupation RMS | `0.610960` | `0.330411` | `0.587281` |
| electronic energy RMS | `0.221581` | `0.230908` | `0.413684` |
| phonon energy RMS | `0.410832` | `0.302944` | `0.283611` |
| electron--phonon energy RMS | `0.562932` | `0.422100` | `0.467906` |
| total internal energy RMS | `0.043976` | `0.043971` | `0.015877` |
| minimum bosonic eigenvalue | `-4.707881e-1` | `2.893143e-5` | `-2.480841` |
| minimum electronic density eigenvalue | `2.297326e-2` | `8.511798e-3` | `-9.259019e-2` |
| maximum abs coordinate | `2.083504` | `1.974078` | `2.107902` |
| fraction of sampled times with the barrier active | `0.764118` (over `t = 0` to 100) | `0.583750` | not measured |
| wall seconds to this horizon | `4.37` (to `t = 100`) | `1500.07` | `4.18` (to `t = 100`) |

The two uncorrected lanes reached `t = 100`; over the full `t = 0` to 100 the
Eq. (112) lane scores `1.511724` l2 time-RMS, `0.705135` all-31 scaled RMS,
minimum bosonic eigenvalue `-6.185545e-1` and minimum electronic density
eigenvalue `2.297326e-2`, and the plain lane `2.379686`, `2.012503`,
`-3.113141` and `-9.290891e-2`.

## Next to the adaptive packet route's line

The packet line is `t = 0` to 98 (worklog:2509-2516); the barrier line is
`t = 0` to 79.95, the horizon it reached in 25 minutes. Both RMS columns use
the same formula; the scale vector differs, as stated above.

| quantity | adaptive packet, `t = 0` to 98 | energy-neutral barrier, `t = 0` to 79.95 | ratio |
|---|---:|---:|---:|
| all-31 scaled RMS | `0.063649` | `0.700816` | 11.0 |
| site-0 occupation RMS | `0.013600` | `0.330411` | 24.3 |
| electronic energy RMS | `0.023320` | `0.230908` | 9.9 |
| phonon energy RMS | `0.028350` | `0.302944` | 10.7 |
| electron--phonon energy RMS | `0.033785` | `0.422100` | 12.5 |
| total internal energy RMS | `0.000579` | `0.043971` | 75.9 |

The packet route's minimum electronic density eigenvalue is `0.058149` with a
joint-Gram minimum of `-1.72e-15` and no online correction; the barrier's
minimum electronic density eigenvalue over its horizon is `8.511798e-3`, which
is a quarter of the exact lane's `3.273866e-2` on the same window, so the
barrier holds the bosonic cone at the cost of taking the electronic density
closer to its boundary than the exact state ever goes.

## Polarization spectrum

`P = -Delta n / 2`, mean subtracted, Hann window, sample step `0.2`, positive
frequencies, electronic band `1.5 <= omega <= 3.5`, observed finite-window
FWHM with no lifetime deconvolution. Two windows are reported because the
corrected lane stopped at `t = 79.95`.

`10 <= t <= 100`, exact and the two uncorrected lanes only:

| lane | Hellinger distance to exact | electronic-band peak | observed FWHM | electronic-band weight |
|---|---:|---:|---:|---:|
| exact cutoff 16 | `0` | `2.356108` | `0.114463` | `0.0569111` |
| `plain_raw` | `0.6380736` | `2.939889` | `0.155661` | `0.5225050` |
| `raw` (Eq. 112) | `0.4954585` | `2.594804` | `0.143885` | `0.0441596` |
| record, exact / raw (worklog:2934-2941) | `0` / `0.6381` | not printed | not printed | `0.0569` / `0.5225` |

The record's exact and raw entries are reproduced to every printed digit by the
exact lane and by `plain_raw`.

`10 <= t <= 79.8`, all three lanes:

| lane | Hellinger distance to exact | electronic-band peak | observed FWHM | electronic-band weight |
|---|---:|---:|---:|---:|
| exact cutoff 16 | `0` | `2.342119` | `0.126434` | `0.0729181` |
| `raw` (Eq. 112) | `0.2812745` | `2.600526` | `0.119563` | `0.0380705` |
| `corrected` | `0.2610264` | `2.536359` | `0.191591` | `0.1185834` |
| `plain_raw` | `0.5008225` | `2.678542` | `0.164077` | `0.3972976` |

## The question plan step 9 asked

Step 9 asks whether the barrier's electronic-band suppression (`0.0026`
against exact `0.0569` over `10 <= t <= 100`) is intrinsic to the
31-coordinate closure or a controller artifact. On this evidence it is a
property of the controller, not of the closure: the ten-coordinate
energy-neutral barrier, acting on the same 31-coordinate archive closure and
the same problem, puts `0.1185834` of its normalized polarization power in the
electronic band against the exact lane's `0.0729181` on the same window, 1.63
times the exact weight, the opposite sign of departure from the factor `0.046`
the record's 31-coordinate joint-Gram lane produced. The two controllers differ
in their control space (ten `(dN, dA)` velocity coordinates against all 31
closed-state coordinates) and the record's corrected lane also carries
`preserve_correlation_trace`; this run did not re-derive that lane, so the
comparison is between two named controllers and not a decomposition of the
difference.

## Reproducibility notes

- The current worktree reproduces round 1's stored pinned run
  (`output/local_runs/paper_v_cone_ref_t2_20260914_v1/pinned_energy_neutral_barrier.npz`,
  `t = 0` to 2, 749 evaluations) with `np.array_equal` True and maximum
  absolute difference `0.0`, although
  `paper_5/src/paper5/stability/matrix_reference.py` carries a sibling
  overnight agent's uncommitted change (+126/-71). That change is therefore
  bitwise-neutral on this path.
- The `t = 0` to 100 corrected run's first 41 samples differ from that stored
  `t = 0` to 2 run by `2.220446e-16`, one ulp, because `DOP853`'s initial step
  selection clips the first step by the distance to `t_bound` and the two runs
  have different `t_bound`. The sample times are equal.
- `integrate_sampled_dop853` with no wall budget reproduces
  `scipy.integrate.solve_ivp(..., method="DOP853", t_eval=...)` exactly on both
  the corrected and the uncorrected right-hand side; this is the first two
  tests of `paper_5/tests/test_cone_long_horizon.py`.

## Gates

| gate | result |
|---|---|
| `paper_5/tests/test_cone_long_horizon.py` | 11 passed |
| `paper_5/tests/test_cone_correction.py`, the file this task must not move | 29 passed |
| whole `paper_5` suite, `cd paper_5 && PYTHONPATH=src pytest -q -p no:cacheprovider` | 347 passed in 141.25 s, `ram_guard` peak `1274 MB` |
| notes' pinned `t = 0` to 20 table | all printed columns reproduced (table above) |
| record's `10 <= t <= 100` polarization entries for exact and raw | reproduced to every printed digit |
| defaults | no existing file was edited; the three source files are new |

## Kept

- `paper_5/src/paper5/stability/cone_long_horizon.py`: the budgeted DOP853
  sampling loop, the six observables, the two RMS conventions, the barrier
  census and the post-pulse drift, all new functions that call
  `cone_correction` and `matrix_reference` without changing them.
- `pipelines/open_dynamics/run_paper_v_energy_neutral_barrier_long_horizon.py`:
  the step-9 driver, with `--corrected-wall-budget-seconds` (default 1500) as
  the named stop rule and a repeatable `--score-window`.
- `paper_5/tests/test_cone_long_horizon.py`: 11 tests, the first two of which
  are the `solve_ivp` parity that makes the budgeted loop a reporting-safe
  substitute for it.
- The three post-processing scripts under the artifact directory, because the
  spectrum window and the second uncorrected lane were both decided after the
  main run.

## Dropped

- **Reaching `t = 100` on the corrected lane.** At the measured 4,107
  right-hand-side evaluations per unit time the remaining `t = 79.95` to 100
  would cost about another 6 minutes of the same wall clock; the task's stop
  rule is 25 minutes, so the horizon reached is reported instead. Nothing was
  changed to make it fit.
- **A single `10 <= t <= 100` spectrum table for all three lanes.** The
  corrected lane has no samples past `t = 79.95`, so the three-lane comparison
  is at `10 <= t <= 79.8` and the record's own window carries only the two
  lanes that reached it.

## Blockers

1. The corrected lane's cost is in the correction, not the integrator: 328,358
   corrections over `t = 0` to 79.95, `4.57 ms` per right-hand-side evaluation
   against `0.145 ms` for the uncorrected Eq. (112) evaluation. Round 1
   measured that 47 percent of the joint-Gram route's profiled time is in
   `paper_5/src/paper5/stability/matrix_reference.py`; the same profile
   question applies to this controller, and this task did not own that file.
   A sibling overnight agent is working that file in this worktree this round
   (`agent_guidance/paper-v/overnight/matrix-reference-speedup-20260915.md`),
   so the `4.57 ms` per evaluation measured here is the cost before whatever
   that work lands.
2. Whether the record's `0.0026` electronic-band weight is attributable to the
   control space, to `preserve_correlation_trace`, or to the base right-hand
   side needs the 31-coordinate joint-Gram lane propagated over
   `10 <= t <= 100`. Round 1 measured that controller at 16,700 evaluations for
   `t = 0` to 0.2, so that horizon is not reachable on this machine without the
   step-8 acceleration continuing.
## Impact analysis (read-only, main checkout index)

No existing function was edited: the three files of this task are new
(`paper_5/src/paper5/stability/cone_long_horizon.py`,
`paper_5/tests/test_cone_long_horizon.py`,
`pipelines/open_dynamics/run_paper_v_energy_neutral_barrier_long_horizon.py`).
`cone_correction.py`, `exact_reference.py` and `matrix_reference.py` were not
touched. The symbols this driver calls were checked read-only against the main
checkout's index, which the tool reports is 8 commits behind this worktree:

| symbol | risk | impacted | direct callers |
|---|---|---:|---|
| `closed_cone_projected_rhs` | LOW | 3 | `report._cone_correction_audit` |
| `structured_boson_barrier_correction` | LOW | 4 | `report._cone_correction_audit`, `closed_cone_projected_rhs.rhs` |
| `closed_residual_subtracted_rhs` | LOW | 4 | `report._cone_correction_audit`, `reproduce._build_manifest` |
| `exact_holstein_driven_trajectory` | MEDIUM | 19 | `compare_exact_and_closed_protocols`, `pauli_repair_analysis.analyze_pauli_repair_ablation`, `matched_exact_controller_sensitivity.analyze_matched_case`, and three more |

`detect_changes` was not run: this worktree is not in the GitNexus index, so a
zero from it would mean unseen, not clean.
