# Investigation: HH L=2 Onsite Spectra Improvement Hypothesis

## Summary
The spectra problem is upstream of FFT/post-processing. The strongest code-level mismatch is that the `exact_v1` controller forecast objective in `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py` optimizes a mixed scalar of fidelity, energy, staggered, doublon, and site-max error, while the repo’s spectra-facing path for `L=2` explicitly treats `d(t)=n_0(t)-n_1(t)` as the primary target. The best first method to improve spectra is therefore to add an explicit primary density target to the controller forecast (`pair_difference` for `L=2`, `staggered` for `L>2`) plus density-shape horizon terms; simple weight retuning should be used only as a short validation probe.

## Symptoms
- `artifacts/agent_runs/20260405_controller_drive_heavier_secant_lead_scan_v1/json/secant_lead100.json` finishes with good-looking final energy/fidelity but poor density accuracy:
  - final `abs_energy_total_error = 0.0027005689438111546`
  - final `fidelity_exact = 0.9886697218412795`
  - summary `max_abs_site_occupations_error = 0.11217073769642516`
  - summary `max_abs_staggered_error = 0.11217073769640451`
  - mean site MAE `0.04285511748539235`
  - pair/staggered correlation to exact `0.31360331165754773`
- `artifacts/agent_runs/20260405_controller_drive_heavier_secant_lead_scan_v1/json/secant_lead100_spectra_report.json` confirms the spectral target is already the density difference and that the dominant line is wrong:
  - `spectral_target.kind = pair_difference`
  - `epsilon_osc = 1.0831889466276459`
  - `drive_line_amplitude_ratio_ctrl_over_exact = 0.3961337925744032`
  - controller dominant peak `omega = 0.7853981633974481`, amplitude `0.06072618696721901`
  - exact dominant peak `omega = 2.356194490192344`, amplitude `0.10279758403440209`
- Append helps only modestly in the comparison artifact `artifacts/agent_runs/20260407_hh_l2_u4_g05_driven_adapt_then_controller_analytic_noise_ladder_v1/lanes/weak/controller/json/weak_controller.json`:
  - final `abs_energy_total_error = 0.010059055278185763`
  - final `fidelity_exact = 0.9929873505400618`
  - `append_count = 3`
  - summary `max_abs_site_occupations_error = 0.1062077335846805`

## Investigation Log

### Phase 1 - Initial assessment from the audit note
**Hypothesis:** The bad spectra are inherited from bad density trajectories rather than created by FFT/windowing.
**Findings:** The audit note already laid out the controller -> trajectory rows -> spectra path and pointed to controller scoring as the likely issue.
**Evidence:** `notes/2026-04-11-hh-onsite-spectra-audit.md`
**Conclusion:** Needed direct code and artifact verification.

### Phase 2 - Broad context gathering via context builder/oracle
**Hypothesis:** The controller objective underweights or mis-specifies the density observable that spectra/reporting later care about.
**Findings:** Context builder selected the controller/config/spectra/staged-workflow/test files and the oracle pointed to an objective mismatch: density is not first-class in the controller forecast, while `L=2` spectra/reporting are density-difference-first.
**Evidence:** Oracle chat `hh-density-score-3B0720`
**Conclusion:** Promising; verify with line-number evidence.

### Phase 3 - Causal path: controller writes the raw density traces, spectra only reads them
**Hypothesis:** The FFT layer is downstream and non-causal for trajectory quality.
**Findings:** `_observable_snapshot()` and `_exact_step_forecast()` in the controller compute the observables; `run()` serializes `site_occupations`, `site_occupations_exact`, `staggered`, `doublon`, and their errors into trajectory rows; `hh_time_dynamics_spectra.py` only loads those rows and derives `pair_difference` downstream.
**Evidence:**
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1200-1251`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:6008-6033`
- `pipelines/time_dynamics/hh_time_dynamics_spectra.py:171-214`
- `pipelines/time_dynamics/hh_time_dynamics_spectra.py:272-276`
- `pipelines/time_dynamics/hh_time_dynamics_spectra.py:437-506`
- `pipelines/time_dynamics/hh_time_dynamics_spectra.py:568-598`
**Conclusion:** Confirmed. FFT/post-processing is downstream of the bad signal.

### Phase 4 - Raw observable extraction bug check
**Hypothesis:** The controller’s site-occupation extractor itself is wrong.
**Findings:** The controller and the hardcoded pipeline use the same spin-orbital bit indexing, the same per-basis-state accumulation loop for `n_up`, `n_dn`, and doublon, and the same alternating-sign staggered formula. The secant artifact also preserves the expected `L=2` identities `staggered = (n_0-n_1)/2` and `n_0+n_1 = 2` at the final checkpoint.
**Evidence:**
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:159-196`
- `pipelines/hardcoded/hubbard_pipeline.py:487-529`
- direct artifact check on `secant_lead100.json`:
  - final controller `n_0+n_1 = 2.000000000000004`
  - final exact `n_0+n_1 = 2.000000000000058`
  - final controller `staggered = -0.05604207551767565`
  - final controller `(n_0-n_1)/2 = -0.05604207551767565`
  - final exact `staggered_exact = 0.0007714497046912849`
  - final exact `(n_0^{exact}-n_1^{exact})/2 = 0.0007714497046912849`
**Conclusion:** Eliminated as the primary cause. The extractor math is internally consistent.

### Phase 5 - Objective mismatch inside `exact_v1`
**Hypothesis:** The controller forecast objective is not density-first and can score trajectories well while density phase/shape are wrong.
**Findings:** `_exact_step_forecast()` returns only fidelity, total-energy error, staggered error, doublon error, and site-max occupation error. `_forecast_tracking_score()` scalarizes those terms over the horizon. `_energy_shape_tracking_terms()` and `_energy_excursion_tracking_terms()` add shape penalties only for energy. The config surface exposes weights only for those terms.
**Evidence:**
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1214-1251`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1253-1273`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1324-1500`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1614-1689`
- `pipelines/time_dynamics/hh_realtime_checkpoint_types.py:24-46`
- `pipelines/time_dynamics/hh_realtime_from_adapt_artifact.py:167-236`
- tests that lock in this behavior:
  - `test/test_hh_realtime_checkpoint_controller.py:3388-3450`
  - `test/test_hh_realtime_checkpoint_controller.py:3453-3567`
**Conclusion:** Confirmed. Density is present only through proxies (`staggered`, `site_occupations_abs_error_max_next`) and has no dedicated shape-aware objective.

### Phase 6 - Repo target mismatch: spectra/reporting are `density_difference`-first for `L=2`
**Hypothesis:** The repo’s own downstream trust logic wants a different target than the controller is directly optimizing.
**Findings:** The staged workflow defaults `L=2` to `density_difference`; it computes the spectral-trust trace from `d(t)=n_0-n_1`; the spectra note says to optimize the same observable that will be Fourier transformed, and for `L=2` that should be `d(t)`; the spectra tests also encode the two-site relation between fluctuation and pair difference.
**Evidence:**
- `pipelines/hardcoded/hh_staged_workflow.py:1946-1979`
- `pipelines/hardcoded/hh_staged_workflow.py:2000-2060`
- `notes/2026-04-05-qpu-spectra-accuracy-checklist.md:5-23`
- `notes/2026-04-05-qpu-spectra-accuracy-checklist.md:158-203`
- `test/test_hh_time_dynamics_spectra.py:51-61`
**Conclusion:** Confirmed. The controller and the spectra-facing trust logic are optimizing/reporting against different primary notions of “good.”

### Phase 7 - Existing hooks already point toward density-first handling, but do not solve it
**Hypothesis:** The missing fix might simply be “add aligned density.”
**Findings:** Driven `exact_v1` already auto-augments with `drive_aligned_density`, and tests prove it. Separate fixed-manifold code and sweep scripts also already support `aligned_density`. However, the controller still does not expose or score a first-class `pair_difference` / primary-density error term, so aligned-density augmentation by itself is not the missing root-cause repair.
**Evidence:**
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:208-268`
- `test/test_hh_realtime_checkpoint_controller.py:1598-1642`
- `pipelines/time_dynamics/hh_fixed_manifold_measured.py:197-256`
- `pipelines/time_dynamics/hh_l2_driven_realtime_pareto_sweep.py:224-258`
**Conclusion:** Eliminated as the first fix target. The generator hook exists; the objective still does not reward the right density target enough.

### Phase 8 - Direct artifact verification and nearby-case comparison
**Hypothesis:** The observed failure could be peculiar to `secant_lead100` rather than structural.
**Findings:** I verified the audit numbers directly from local artifacts. I also re-ran the existing spectra analysis code on `secant_lead150` and `secant_lead200`; both remain spectrally poor (`epsilon_osc > 1.03`) and still put the dominant controller peak below the exact value, so nearby secant-lead tuning does not repair the density spectrum.
**Evidence:**
- direct inspection of
  - `artifacts/agent_runs/20260405_controller_drive_heavier_secant_lead_scan_v1/json/secant_lead100.json`
  - `artifacts/agent_runs/20260405_controller_drive_heavier_secant_lead_scan_v1/json/secant_lead100_spectra_report.json`
  - `artifacts/agent_runs/20260407_hh_l2_u4_g05_driven_adapt_then_controller_analytic_noise_ladder_v1/lanes/weak/controller/json/weak_controller.json`
- offline spectra analysis using the current `pipelines/time_dynamics/hh_time_dynamics_spectra.py` code:
  - `secant_lead150`: `epsilon_osc = 1.036722872207117`, controller top peak `omega ≈ 1.5610398278707045`, exact top peak `omega ≈ 2.3415597418060563`
  - `secant_lead200`: `epsilon_osc = 1.0320646180775908`, controller top peak `omega ≈ 1.5610398278707045`, exact top peak `omega ≈ 2.3415597418060563`
**Conclusion:** Confirmed. This is not just a single bad lead choice; the nearby controller family remains density/spectra-poor.

### Phase 9 - Secondary reporting/data-contract mismatch
**Hypothesis:** Downstream reporting/calibration wants pair-difference fields the controller never emits.
**Findings:** The controller rows emit `staggered` and `site_occupations`, but not `pair_difference` or `abs_pair_difference_error`. Meanwhile, the calibration/reporting code already accepts pair-difference fields if present.
**Evidence:**
- direct row-key check on `secant_lead100.json`: `pair_difference` absent, `abs_pair_difference_error` absent, `staggered` present, `site_occupations` present
- `pipelines/hardcoded/hh_analytic_noise_calibration.py:82-140`
- `test/test_hh_analytic_noise_calibration.py:12-89`
**Conclusion:** Secondary issue only. This omission does not create the bad spectrum, but it shows the controller/reporting contract is behind the repo’s density-first reporting logic.

### Phase 10 - Provenance drift
**Hypothesis:** The mismatch may reflect recent untracked controller work drifting away from tracked spectra/reporting logic.
**Findings:** `git status` shows the entire `pipelines/time_dynamics/` directory is currently untracked in this checkout, so `git blame` cannot establish history there. By contrast, `git blame` on the staged-workflow spectra logic shows the `L=2` density-difference target landed in commit `01b0252` on `2026-04-05`.
**Evidence:**
- `git status` in the repo root: `pipelines/time_dynamics/` listed as untracked
- `git blame pipelines/hardcoded/hh_staged_workflow.py` for lines `1946-2050`
**Conclusion:** Likely contributing factor. The tracked downstream truth and the current untracked controller objective appear to have drifted apart.

## Root Cause
The root cause is an upstream objective mismatch in the `exact_v1` controller law.

For `L=2`, the repo’s spectra-facing truth is the pair-density difference

`d(t) = n_0(t) - n_1(t)`.

But the controller forecast currently scores a horizon-weighted scalar blend of:
- fidelity defect
- total-energy error
- staggered error
- doublon error
- site-occupation max error

and then adds shape/excursion penalties only for energy. In `L=2`, the controller therefore has only proxy access to the real spectral target through `staggered = d/2` and site-max error, while energy remains structurally privileged through the extra slope/curvature/excursion terms. That asymmetry allows runs with decent final energy/fidelity to still have the wrong density amplitude, phase, and dominant spectral line placement.

The spectra layer is not causing the problem: it simply Fourier-transforms the site-density traces the controller already wrote. The observable extractor math is also not the main problem: the controller and hardcoded pipeline use the same bit-indexing/occupation logic and preserve the expected two-site identities. The issue is that the controller is not directly optimizing the same density observable the repo later interprets spectrally.

## Recommendations
1. **Best first method:** Add an explicit primary-density target to the controller forecast and score it directly.
   - `L=2`: use `pair_difference = n_0 - n_1`
   - `L>2`: use `staggered`
   - Main edit sites when implementation begins:
     - `pipelines/time_dynamics/hh_realtime_checkpoint_types.py`
     - `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py`
     - `pipelines/time_dynamics/hh_realtime_from_adapt_artifact.py`
2. **Add density-shape horizon terms**, parallel to the current energy-only shape terms.
   - Add at least density slope/curvature mismatch over the forecast horizon.
   - Keep `site_occupations_abs_error_max_next` as a guardrail/veto rather than the main spectral objective.
3. **Use weight-retuning only as a quick validation experiment.**
   - Upweight current staggered/site terms and downweight energy.
   - Check whether `epsilon_osc`, drive-line amplitude ratio, and dominant density peak placement improve.
   - If they improve, treat that as confirmation of the diagnosis, not the final fix.
4. **Secondary contract cleanup:** emit `pair_difference`, `pair_difference_exact`, and `abs_pair_difference_error` in controller rows.
   - This aligns the controller output with `hh_time_dynamics_spectra.py` and `hh_analytic_noise_calibration.py`.
   - Useful implementation site: `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py` around the row-serialization block at `6008-6033`.

## Preventive Measures
- Keep controller objectives aligned with the observable that downstream spectra/reporting will actually use.
- When adding new spectral-trust/reporting logic, add corresponding controller-side objective terms or explicitly document the gap.
- Add regression tests that fail if a new controller objective can improve energy while leaving `pair_difference` / `staggered` spectral-trust metrics badly degraded.
- Track the `pipelines/time_dynamics/` controller stack in git before further tuning; right now the untracked status makes historical intent and drift analysis hard.
