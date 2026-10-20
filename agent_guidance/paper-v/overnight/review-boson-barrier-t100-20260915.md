# Review: boson-barrier-t100 (2026-09-15)

Independent reviewer for the plan step 9 implementer. Verdict: **pass**. Every
number in `agent_guidance/paper-v/overnight/boson-barrier-t100-20260915.md` that
could be recomputed was recomputed from the stored arrays without reusing the
implementer's post-processing scripts, and every one reproduced to all printed
digits. No source file was edited by this reviewer.

## Scope

The worktree's working tree lists nine modified tracked files and fourteen
untracked paths. Attribution:

| path | owner |
|---|---|
| `paper_5/src/paper5/stability/matrix_reference.py` (M) | matrix-reference-speedup |
| `paper_5/tests/test_matrix_scalar_parity.py` (M) | matrix-reference-speedup |
| `paper_5/tests/test_matrix_reference_moment_assembly.py` (new) | matrix-reference-speedup |
| `paper_5/src/paper5/stability/projected_apcm.py` (M) | projective-gap |
| `paper_5/src/paper5/stability/apcm_projective_guard.py` (M) | projective-gap |
| `paper_5/src/paper5/stability/apcm_projective_event_guard_analysis.py` (M) | projective-gap |
| `paper_5/tests/test_apcm_projective_event_guard.py` (M) | projective-gap |
| `paper_5/src/paper5/stability/apcm_carried_witness.py` (M) | cone-solve-instances |
| `paper_5/src/paper5/stability/apcm_carried_witness_analysis.py` (M) | cone-solve-instances lane |
| `paper_5/tests/test_apcm_carried_witness.py` (M) | cone-solve-instances |
| `paper_5/src/paper5/stability/cwrmf_solver_instances.py`, `cwrmf_solver_replay.py`, `cwrmf_solver_replay_bench.py`, `paper_5/tests/cwrmf_solver_*_test.py` (new) | cone-solve-instances |
| `paper_5/src/paper5/stability/cone_long_horizon.py` (new) | **boson-barrier-t100** |
| `paper_5/tests/test_cone_long_horizon.py` (new) | **boson-barrier-t100** |
| `pipelines/open_dynamics/run_paper_v_energy_neutral_barrier_long_horizon.py` (new) | **boson-barrier-t100** |
| `agent_guidance/paper-v/overnight/boson-barrier-t100-20260915.md` (new) | **boson-barrier-t100** |

No scope violation. All four of this implementer's paths are inside the
section 7 allowlist (`paper_5/**`, `pipelines/open_dynamics/**`,
`agent_guidance/paper-v/**`); the artifact directory
`output/local_runs/paper_v_boson_barrier_t100_20260915_v1/` is matched by
`.gitignore:50`. The three files the task forbade -- `cone_correction.py`,
`exact_reference.py`, `matrix_reference.py` -- carry no change from this
implementer; `matrix_reference.py` is modified, but by matrix-reference-speedup,
which names it as its own in `matrix-reference-speedup-20260915.md:11-13`.
The main checkout `/Users/lordskyler/Holstein_test` is untouched: neither new
source file exists there, `agent_guidance/paper-v/overnight/` does not exist
there, and the only file under it modified since 2026-09-14 20:00 is
`.claude/scheduled_tasks.lock`.

## Tests, rerun by the reviewer

| command | result |
|---|---|
| `pytest -q tests/test_cone_long_horizon.py tests/test_cone_correction.py` under `ram_guard --limit-mb 2000` | 40 passed in 8.03 s, peak 121 MB (11 + 29, the counts claimed) |
| whole `paper_5` suite under `ram_guard --limit-mb 3000` | 376 passed in 143.29 s, peak 1236 MB |

The implementer reported 347; the suite now stands at 376 because three sibling
agents added tests to this worktree after their run. Green either way.

## Parity evidence, re-checked independently

The load-bearing claim is that the new budgeted stepping loop is
`scipy.integrate.solve_ivp(..., method="DOP853", t_eval=...)`. It was checked on
the run's own protocol, not only on the unit test's `t = 0.3` horizon:

- `integrate_sampled_dop853` at `final_time = 2.0` on
  `closed_cone_projected_rhs(..., energy_neutral=True)` returns states
  `np.array_equal` to round 1's stored `solve_ivp` run
  (`output/local_runs/paper_v_cone_ref_t2_20260914_v1/pinned_energy_neutral_barrier.npz`),
  maximum absolute difference `0.0`, same `nfev = 749`.
- Re-running round 1's `solve_ivp` harness protocol directly under the current
  worktree state also reproduces that stored file bitwise
  (`array_equal` True, maximum absolute difference `0.0`, `nfev = 749`), so
  matrix-reference-speedup's uncommitted `matrix_reference.py` change is
  bitwise-neutral on this path, as the implementer states.
- The step-9 corrected run's first 41 stored samples differ from that
  `t_bound = 2.0` run by `2.220446049250313e-16`, the one ulp the report
  attributes to DOP853's initial-step selection under a different `t_bound`.
  Because the loop is proven bitwise-identical at equal `t_bound`, that
  attribution is the only remaining explanation and it holds.

## Numbers recomputed from the stored arrays

`{exact,raw,corrected,plain_raw}_trajectory.npz` were loaded, the scales
recomputed with `multi_coherent_scores.development_coordinate_scales` from the
exact cutoff-16 path over `t = 0` to 100, the sample grids asserted to coincide
index for index to `1e-12`, and the report's headline matched-window table
recomputed. Definitions: trapezoidal time-RMS of the unscaled 31-coordinate
Euclidean distance; unweighted RMS over samples and the 31 scaled coordinates;
unweighted sample RMS of the observable difference. Horizon `t = 0` to 79.95,
1600 samples of the 0.05 grid, `lambda_ep = 1.5`, `gamma = 0.5`, drive
amplitude 1, cutoff 16, DOP853 rtol `1e-9` atol `1e-11` maximum step 0.05.

| quantity | raw (Eq. 112) | corrected | plain_raw | matches report |
|---|---:|---:|---:|---|
| coordinate l2 time-RMS | `1.549827` | `1.016703` | `2.050020` | yes |
| all-31 scaled RMS | `0.713667` | `0.700816` | `1.585195` | yes |
| site-0 occupation RMS | `0.610960` | `0.330411` | `0.587281` | yes |
| electronic energy RMS | `0.221581` | `0.230908` | `0.413684` | yes |
| phonon energy RMS | `0.410832` | `0.302944` | `0.283611` | yes |
| electron--phonon energy RMS | `0.562932` | `0.422100` | `0.467906` | yes |
| total internal energy RMS | `0.043976` | `0.043971` | `0.015877` | yes |
| minimum bosonic eigenvalue | `-4.7078810e-1` | `2.8931428e-5` | `-2.4808413` | yes |
| minimum electronic density eigenvalue | `2.2973260e-2` | `8.5117982e-3` | `-9.2590192e-2` | yes |
| maximum abs coordinate | `2.083504` | `1.974078` | `2.107902` | yes |

Exact lane minimum electronic density eigenvalue on the same window:
`0.032738658963314925`, the report's `3.273866e-2`.

Over `t = 0` to 20, 401 samples, the maximum absolute coordinate recomputed
directly is `1.9787388` for the Eq. (112) lane and `1.9740781` for the barrier;
the working notes' pinned table
(`MATH/paper_facing/paper_V_high_u_gkba/stability_quantum_workflow_working_notes.md:1856-1858`)
prints `1.97874` and `1.97408`, with `-3.05033e-1`, `5.51945e-2`, `2.41557e-6`
and `2.89314e-5`, `4.88418e-2`, `2.53904e-6` -- every column reproduced, and
`0.5511221945137157` is the notes' `55.11` percent as 221 of 401 samples.

The record's adaptive packet line (worklog:2509-2516) reads exactly `0.063649`,
`0.013600`, `0.023320`, `0.028350`, `0.033785`, `0.000579` on `0 <= t <= 98`, as
quoted. The record's polarization table (worklog:2934-2941) reads exact
`2.3561`, `0.1145`, `0.0569` and uncorrected archive EOM `0.6381`, `2.9399`,
`0.1557`, `0.5225`; the implementer's exact lane gives `2.3561081`, `0.1144630`,
`0.0569111` and `plain_raw` gives `0.6380736`, `2.9398888`, `0.1556609`,
`0.5225050`. The definitional finding -- that the record's "raw" polarization
lane is `closed_scalar_rhs` and not `closed_residual_subtracted_rhs` -- is
therefore established, not asserted: the Eq. (112) lane gives `0.4954585` and
`0.0441596` on the same window, which are not the record's numbers.

The polarization definition matches the record's producer exactly:
`analyze_archive_polarization_peaks._weak_polarization:112` comments that the
first coordinate is `Delta n = rho_00 - rho_11` and forms `-0.5 * delta_n`, and
its `_interpolate` is `np.interp` with a bounds check, which is what the new
script uses.

Derived ratios spot-checked: `1500.0695/328358 = 4.568 ms` and
`4.3715/30041 = 0.1455 ms` per right-hand-side evaluation; `749/2 = 374.5` and
`328358/79.95 = 4107` evaluations per unit time; `0.1185834/0.0729181 = 1.626`;
`0.700816/0.063649 = 11.0`; `(0.713667-0.700816)/0.713667 = 1.80` percent.

## Default path

No default, profile, threshold or acceptance rule changed. All three source
files are new. `integrate_sampled_dop853`'s `wall_budget_seconds` defaults to
`None`, which is the unbudgeted `solve_ivp`-identical path, and the budget is
reached only through the driver's named `--corrected-wall-budget-seconds`
(default 1500) and `--raw-wall-budget-seconds`. The new module defines only
`OBSERVABLE_NAMES`, `POST_PULSE_START = 4.0` and `BARRIER_ACTIVITY_NORM = 1e-10`,
each copied with attribution from `report._cone_correction_audit` and
`plot_high_coupling_controller_observables`; neither file mutates a module
global, monkeypatches, or writes to a settings object.
`paper_5/src/paper5/stability/__init__.py` is unmodified, so nothing enters the
package's exported surface.

## Impact analysis, re-run read-only against the main checkout index

`closed_cone_projected_rhs`: risk LOW, 3 impacted, 1 direct caller, module
Stability, no affected process. `exact_holstein_driven_trajectory`: risk MEDIUM,
19 impacted, 6 direct callers, one affected process
(`apcm_carried_witness_analysis.main`). Both match the report's table. No
existing function was edited, so these are informational. `detect_changes` is
correctly reported as not run: this worktree is not in the index, so a zero from
it would mean unseen, not clean.

## Vocabulary

No quantity name is invented and presented as established. The two RMS
conventions are named for their producers and the difference in scale vector is
stated. Where the implementer needed a quantity the record does not carry -- the
fraction of the *uncorrected* trajectory's samples at which the correction would
be active, and the plain archive EOM lane -- the report says so in a full
sentence before using a short label. Every number in the report carries its
window, sample count and configuration.

## Issue found (non-blocking)

The report's central matched-window table (`t = 0` to 79.95 for all three
lanes) is not regenerated by any saved script in the artifact directory:
`report_tables.py` prints only the `t_0_to_20` and `t_0_to_100` entries of
`summary.json`, and `plain_archive_lane.py` scores `plain_raw` only on
`[0, 20]` and `[0, 100]`. The matched-window numbers are reproducible from the
stored `.npz` files -- all thirty were reproduced above -- but nothing in the
directory regenerates them on its own. A third post-processing script, or a
`--score-window 0 79.95` invocation recorded in `driver.log`, would close it.
