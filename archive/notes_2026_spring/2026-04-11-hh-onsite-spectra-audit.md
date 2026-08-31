# HH L=2 Onsite-Density / Spectra Audit

- Date: 2026-04-11
- Scope: current checkout, driven HH controller artifacts for `L=2`
- Objective: explain why the current spectra are inaccurate, show that the problem is upstream of FFT/post-processing, and isolate the first fix target

## Headline finding

The current spectral inaccuracy is driven primarily by inaccurate onsite/site-density trajectories, not by the FFT layer itself. The cleanest evidence anchor is `artifacts/agent_runs/20260405_controller_drive_heavier_secant_lead_scan_v1/json/secant_lead100.json`: this run has small final total-energy error and high final fidelity, yet materially wrong site occupations and staggered density, and the spectra report simply reflects those bad density traces.

## Repo-truth path

The repo path is straightforward:

1. `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:159-196` defines `_site_resolved_number_observables(...)`, which computes `n_up_site`, `n_dn_site`, `n_site`, `doublon`, and `staggered` directly from the controller statevector probabilities.
2. `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1200-1220` calls that helper inside `_observable_snapshot(...)`, and `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1221-1256` uses the resulting observables in `_exact_step_forecast(...)`.
3. `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:6008-6030` serializes those values directly into trajectory rows as `site_occupations`, `site_occupations_exact`, `staggered`, `staggered_exact`, `doublon`, `doublon_exact`, and `site_occupations_abs_error_max`.
4. `pipelines/time_dynamics/hh_time_dynamics_spectra.py:104-163` resolves controller rows from JSON, and `pipelines/time_dynamics/hh_time_dynamics_spectra.py:164-185` merely loads `site_occupations`, `site_occupations_exact`, `staggered`, and `doublon` for post-processing.

So the spectra layer is downstream: it reads the already-produced density traces and computes FFT-derived summaries from them. If those traces are wrong, the spectra will faithfully be wrong too.

## Observable extractor consistency check

There is currently no evidence that the raw observable extractor is internally inconsistent.

- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:159-196` and
- `pipelines/hardcoded/hubbard_pipeline.py:498-529`

use the same bit-indexing logic and the same site-resolved occupation / doublon construction, with `staggered` then built from the site occupations in `pipelines/hardcoded/hubbard_pipeline.py:520-524`. The present audit therefore does **not** accuse `_site_resolved_number_observables(...)` of being wrong; the stronger evidence points instead to the controller selecting trajectories whose density dynamics are bad.

## Primary evidence: `secant_lead100`

Primary artifact:

- `artifacts/agent_runs/20260405_controller_drive_heavier_secant_lead_scan_v1/json/secant_lead100.json`

Key summary values from that artifact:

- final `abs_energy_total_error = 2.7005689438111546e-03`
- final `fidelity_exact = 0.9886697218412795`
- `max_abs_site_occupations_error = 1.1217073769642516e-01`
- `max_abs_staggered_error = 1.1217073769640451e-01`
- `max_abs_doublon_error = 5.537413295929808e-02`

Full-trajectory onsite statistics computed directly from the stored row fields:

- site-occupation MAE, mean over both sites and all checkpoints: `4.285511748539235e-02`
- site-occupation RMS, mean over both sites and all checkpoints: `5.125191173836178e-02`
- per-site MAE: `[4.285511748539598e-02, 4.285511748538869e-02]`
- per-site RMS: `[5.125191173836330e-02, 5.125191173836024e-02]`

Trajectory-shape agreement is also poor despite the good final energy:

- pair-density correlation to exact: `0.31360331165754773`
- staggered correlation to exact: `0.31360331165754773`

For `L=2` at fixed `N_e=2`, the rows still satisfy the expected identities, which is important because it shows the spectra layer is not inventing the problem:

- final controller `site_occupations[0] + site_occupations[1] = 2.000000000000004`
- final exact `site_occupations_exact[0] + site_occupations_exact[1] = 2.000000000000058`
- final controller `staggered = -5.604207551767565e-02`
- final controller `(n_0 - n_1)/2 = -5.604207551767565e-02`
- final exact `staggered_exact = 7.714497046912849e-04`
- final exact `(n_0^{\mathrm{exact}} - n_1^{\mathrm{exact}})/2 = 7.714497046912849e-04`

So the controller preserves the obvious density identities while still getting the density dynamics wrong. That is exactly the pattern we would expect if the FFT is faithfully reflecting a bad raw target.

## Spectral evidence

Primary spectra artifact:

- `artifacts/agent_runs/20260405_controller_drive_heavier_secant_lead_scan_v1/json/secant_lead100_spectra_report.json`

That report says the analyzed spectral target is the pair density difference

\[
d_{01}(t)=n_0(t)-n_1(t),
\]

with

- `spectral_target.kind = pair_difference`
- `spectral_target.display_label = d_01(t) = n_0(t) - n_1(t)`
- `epsilon_osc = 1.0831889466276459`
- `drive_line_amplitude_ratio_ctrl_over_exact = 0.3961337925744032`

Most importantly, the dominant density peaks are already wrong before any manuscript-facing interpretation:

- controller dominant pair-density peak: `omega = 0.7853981633974481`, amplitude `0.06072618696721901`
- exact dominant pair-density peak: `omega = 2.356194490192344`, amplitude `0.10279758403440209`

So the line-placement and line-amplitude problem is already present in the raw density target supplied to the spectra layer. The FFT is not the first thing to fix.

## Comparison: append can help, but is not the first root-cause fix

Best append-accepting comparison artifact examined here:

- `artifacts/agent_runs/20260407_hh_l2_u4_g05_driven_adapt_then_controller_analytic_noise_ladder_v1/lanes/weak/controller/json/weak_controller.json`

Important values:

- final `abs_energy_total_error = 1.0059055278185763e-02`
- final `fidelity_exact = 0.9929873505400618`
- `append_count = 3`
- `max_abs_site_occupations_error = 1.062077335846805e-01`

This confirms that append is real and can help in some runs, but it does **not** yet solve the onsite-density problem: the max site-occupation error is still about `0.106`, which is only modestly better than `secant_lead100` at about `0.112`. So append-family expansion is not the first root-cause fix for the spectra issue.

## Diagnosis: likely cause in the current controller objective

The current controller objective is not density-first.

- `pipelines/time_dynamics/hh_realtime_from_adapt_artifact.py:496-524` sets the default exact-forecast base weights to
  - fidelity defect = `1.0`
  - staggered error = `1.0`
  - doublon error = `1.0`
  - site-occupations error = `1.0`
  - total-energy error = `1.0`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1614-1646` shows that `_forecast_tracking_score(...)` combines those terms with equal base weight, and the onsite term enters only as `site_occupations_abs_error_max_next`.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1638-1659` adds optional shape terms only for energy (`energy_slope_abs_error_mean`, `energy_curvature_abs_error_mean`, excursion terms), not for density.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1661-1689` likewise enforces the bounded-defect override with separate thresholds, but the forecast objective itself still does not make density-shape tracking first-class.

The present mismatch therefore looks like this:

1. energy can remain good,
2. final fidelity can remain high,
3. yet the shape, phase, and dominant line placement of the density signal can still be wrong,
4. because onsite/site-density dynamics are not treated as the dominant forecast target over the horizon.

That diagnosis is consistent with the secant artifact: the run is objectively decent by energy and fidelity, but the density signal is poor enough that the spectra are wrong.

## First-fix target

The audit supports **controller-law / forecast-objective repair before any spectra-layer change**.

Recommended first fix target:

- make primary density tracking first-class in the controller,
- use `density_difference` as the primary compact density target for `L=2`,
- use `staggered` as the compact density target for `L>2`,
- add density-shape terms over the forecast horizon,
- keep site max error as a veto / guardrail.

What not to do first:

- do **not** retune FFT windowing, detrending, or harmonic fitting first,
- do **not** treat append-family expansion as the first fix.

In short: **the first repair target is the controller’s density-tracking objective, not the spectra post-processing layer.**
