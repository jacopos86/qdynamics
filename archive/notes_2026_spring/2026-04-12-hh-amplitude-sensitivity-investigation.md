# Investigation: HH Low-Amplitude vs High-Amplitude Spectra Sensitivity

## Summary
The apparent advantage of the high-amplitude U=1 run is mostly a large-signal / observability effect, not evidence that the controller law is intrinsically more accurate in that regime. The low-amplitude density-patched probe already outperforms the high-amplitude Phase A run on a normalized pair-tracking measure, and the remaining weak-drive error is best explained by residual non-scale-free controller logic: the raw bounded-defect stay gate first, then the energy-coupled secant lead cap, with horizon length a secondary contributor.

## Symptoms
- In the low-amplitude L=2 patched probe, pair-density spectrum improved materially, but the dominant pair line is still one FFT bin low and site occupations remain inaccurate.
- A high-amplitude U=1 summary PDF suggested stronger-looking pair / site dynamics under a much larger drive and different physics.
- The concern was that the weak-drive regime might simply be too low-amplitude to track well, and that the high-amplitude regime might therefore be inherently easier.

## Investigation Log

### Phase 1 - Initial assessment
**Hypothesis:** The high-amplitude run may look better because larger observable span makes density/site tracking easier to see and easier to score.
**Findings:** The weak-drive amplitude note explicitly records exact/controller pair spans of about `0.37` and `0.24`, motivating amplitude sensitivity as a live concern. The high-amplitude summary PDF and associated scripts confirm a very different physical regime: `U=1.0`, `drive_A=1.5`, `drive_omega=1.2`, versus the weak-drive probe at `U=4.0`, `drive_A=0.55`, `drive_omega=2.0`.
**Evidence:**
- `notes/HH_ENHANCE_OSCILLATION_AMPLITUDE_SPEC.md:4-10`
- `artifacts/agent_runs/high_amp_spectra_run/pdf/high_amp_three_run_summary.pdf` page 1 text extract
- `artifacts/agent_runs/high_amp_spectra_run/run_step2_dynamics.py:62-88`
**Conclusion:** Confirmed as a plausible hypothesis, but insufficient by itself.

### Phase 2 - Broad context gathering
**Hypothesis:** Remaining amplitude sensitivity may live outside the normalized density score itself.
**Findings:** `context_builder` and Oracle both pointed to an interaction story: the score has already been made largely scale-aware, so the remaining weak-drive fragility is more likely to come from raw absolute gates and energy-coupled proposal suppression than from the normalized score alone.
**Evidence:**
- `context_builder` chat `hh-amplitude-cause-0C7AEF`
- Oracle initial assessment in chat `hh-amplitude-cause-0C7AEF`
**Conclusion:** Strong lead for direct verification.

### Phase 3 - Controller-law verification
**Hypothesis:** The weak-drive failure is caused less by the normalized density score and more by raw absolute gates / energy-coupled proposal suppression.
**Findings:** The forecast score itself normalizes primary-density, site, doublon, and energy errors by exact horizon excursion and adds normalized density-slope mismatch. But the bounded-defect stay override still uses fixed raw thresholds (`2e-2`, `2e-3`, etc.), and the tangent-secant proposal can be tapered or suppressed by a cap tied to the next exact energy delta plus a fixed trust radius.
**Evidence:**
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1359-1474`
  - `primary_density_scale`, `site_occupations_scale`, `energy_total_scale`, and `primary_density_slope_scale` are built from exact-horizon excursion plus floors.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1910-1961`
  - `_forecast_tracking_score(...)` consumes `normalized_primary_density_error_next`, `normalized_site_occupations_abs_error_max_next`, `normalized_energy_total_error_next`, and adds normalized density-slope mismatch.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1988-2004`
  - `_stay_forecast_within_exact_v1_bounded_defect(...)` still gates stay on raw absolute thresholds:
    - `primary_density_error <= 2.0e-2`
    - `primary_density_slope_error <= 2.0e-2`
    - `abs_doublon_error_next <= 2.0e-3`
    - `site_occupations_abs_error_max_next <= 2.0e-2`
    - `abs_energy_total_error_next <= 2.0e-3`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:2005-2024`
  - `_exact_v1_forecast_override_reason(...)` can force `exact_forecast_stay_within_bounded_defect` before append can win.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:2908-2961`
  - `_exact_tangent_secant_proposal(...)` computes `lead_cap = signed_energy_lead_limit * abs(next_exact_energy_delta)` and can return `None` if the signed-energy taper collapses, then separately clips by a fixed `trust_radius`.
- `pipelines/time_dynamics/hh_realtime_from_adapt_artifact.py:486-503`
  - current standalone defaults keep `trust_radius = 0.75`, `signed_energy_lead_limit = 1.0`, horizon steps `2`, horizon weights `(2.0, 1.0)`.
**Conclusion:** Confirmed. The normalized score is not the main remaining amplitude-sensitive piece; the raw gate and energy-coupled secant suppression are.

### Phase 3 - High-amplitude artifact verification
**Hypothesis:** The high-amplitude run is genuinely much more accurate than the low-amplitude patched probe.
**Findings:** The original high-amplitude controller run is actually very poor despite using the same core controller regime. The Phase A `lead200_base` / `lead250_base` runs increase pair span, but they still have large pair/site errors. Relative to exact pair span, the low-amplitude patched probe (`pair_mae/span_exact ~ 0.138`) is better than the high-amplitude Phase A `lead250_base` run (`~0.251`).
**Evidence:**
- `artifacts/agent_runs/high_amp_spectra_run/run_step2_dynamics.py:62-88`
  - high-amplitude original run uses `drive_A=1.5`, `drive_omega=1.2`, `trust_radius=0.75`, `signed_energy_lead_limit=1.0`, horizon `(2.0, 1.0)`.
- `artifacts/agent_runs/high_amp_spectra_run/step2_dynamics.json:35-36,126,135-154,338-346`
  - original high-amplitude controller metrics: `append_count=0`, `final_fidelity_exact=0.0352988`, `final_abs_energy_total_error=0.339059`, `max_abs_doublon_error=0.603276`, `max_abs_site_occupations_error=1.025984`, drive profile `A=1.5`, `omega=1.2`.
- `artifacts/agent_runs/20260407_high_amp_u1_phaseA_balanced_observable_scan_v1/controller_phaseA_balanced_scan.py:28-58`
  - Phase A keeps the same high-amplitude physics and adds heavy energy-shape weights.
- `artifacts/agent_runs/20260407_high_amp_u1_phaseA_balanced_observable_scan_v1/controller_phaseA_balanced_scan.py:61-87`
  - `lead200_base` / `lead250_base` mainly relax `signed_energy_lead_limit` to `2.0` / `2.5` while keeping `trust_radius=0.75`.
- `artifacts/agent_runs/20260407_high_amp_u1_phaseA_balanced_observable_scan_v1/json/analysis.json:7,12-13`
  - confirms high-amplitude physics `U=1.0`, `drive_A=1.5`, `drive_omega=1.2`.
- `artifacts/agent_runs/20260407_high_amp_u1_phaseA_balanced_observable_scan_v1/json/analysis.json:120-130,192-202`
  - `lead200_base` / `lead250_base` metrics:
    - `pair_span_controller ~ 2.372553`
    - `mean_abs_pair_error ~ 0.722817`
    - `epsilon_osc_pair ~ 1.088336`
    - `max_abs_site_occ_error ~ 0.818516`
- `artifacts/agent_runs/20260412_hh_l2_secant_lead100_density_patch_probe_v1/json/secant_lead100_vs_density_patch_compare_auto.json:17-22,68-107`
  - low-amplitude patched metrics:
    - `max_abs_site_occupations_error = 0.08296695508472007`
    - `pair_corr = 0.7306776850552963`
    - `pair_mae = 0.05125566123734739`
    - `pair_span_exact = 0.37098163941149154`
    - `dominant_peak_abs_omega_error = 0.7805199139353518`
    - `drive_line_ratio_ctrl_over_exact = 0.766096694394865`
**Conclusion:** Eliminated. The high-amplitude run is not actually more accurate in normalized pair tracking; it mostly looks better because the signal is larger.

### Phase 4 - Refocused Oracle check
**Hypothesis:** After accounting for normalized accuracy, the main remaining low-amplitude problem should be controller-law residuals rather than weak drive alone.
**Findings:** Oracle’s updated judgment matched the verified evidence: the stronger signal is mainly an observability illusion, while the next controller priority should be the raw bounded-defect stay gate first, then the signed-energy-lead cap, then horizon length, with site-max formulation lower priority.
**Evidence:**
- Oracle follow-up in chat `hh-amplitude-cause-0C7AEF`
**Conclusion:** Confirmed and ranked.

## Root Cause
The remaining low-amplitude inaccuracy is best explained by an interaction of weak-drive physics with residual non-scale-free controller logic.

1. **Weak drive creates a squeezed observable span**
   - The weak-drive note records exact/controller pair spans of only about `0.37` and `0.24` (`notes/HH_ENHANCE_OSCILLATION_AMPLITUDE_SPEC.md:4-10`).
   - This makes any fixed absolute defect look large relative to the signal.

2. **But the normalized score itself is already mostly scale-aware**
   - `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1359-1474` and `1910-1961` show that primary density, site, doublon, energy, and density-slope terms are normalized by exact-horizon excursion before entering the forecast score.

3. **The real low-amplitude bottleneck is the raw bounded-defect stay gate**
   - `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1988-2004` still uses fixed raw thresholds such as `2e-2` for primary density / site and `2e-3` for doublon / energy.
   - `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:2005-2024` lets that raw gate force `stay` before append can win.
   - In the weak-drive regime, those raw thresholds can be spectrally too loose even when the normalized density objective is still unhappy.

4. **The second bottleneck is energy-coupled secant suppression**
   - `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:2908-2961` scales the secant proposal by a cap tied to `abs(next_exact_energy_delta)` and then clips again by a fixed trust radius.
   - When the exact energy swing is small, this can taper or delete the secant correction precisely where low-amplitude density tracking needs help most.

5. **High-amplitude Phase A mainly changed visibility and the secant cap, not intrinsic accuracy**
   - `artifacts/agent_runs/20260407_high_amp_u1_phaseA_balanced_observable_scan_v1/controller_phaseA_balanced_scan.py:61-87` shows Phase A primarily relaxed `signed_energy_lead_limit` to `1.5/2.0/2.5` and varied energy-shape weights.
   - `artifacts/agent_runs/20260407_high_amp_u1_phaseA_balanced_observable_scan_v1/json/analysis.json:120-130,192-202` shows that even then the high-amplitude pair/site errors remain large.
   - So the high-amplitude regime is easier to *see*, but not actually better in normalized tracking.

## Eliminated Hypotheses
- **Eliminated:** “The high-amplitude run proves the controller law is more accurate.”
  - The original high-amplitude same-default controller run is very poor (`step2_dynamics.json:126,135-154`).
- **Eliminated:** “Amplitude alone is enough to fix tracking.”
  - Higher amplitude without controller reparameterization still fails badly.
- **Weakened:** “Phase A lead250_base is simply the same controller succeeding because the signal is larger.”
  - Phase A also relaxed `signed_energy_lead_limit` and used heavy energy-shape weights (`controller_phaseA_balanced_scan.py:36-87`).
  - Even then, normalized errors remain poor.

## Recommendations
1. **Patch the raw bounded-defect stay override before further physics retuning**
   - File: `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1988-2024`
   - Make the stay gate scale-aware / normalized, or gate it with density-compatible criteria rather than fixed `2e-2` / `2e-3` absolute values.
2. **Patch the signed-energy-lead cap next**
   - File: `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:2908-2961`
   - Replace the cap’s dependence on raw `abs(next_exact_energy_delta)` with a more density-compatible or normalized rule, or expose a selector mode that relaxes it in density-first runs.
3. **Only after 1 and 2, test a longer low-amplitude horizon**
   - Files: `pipelines/time_dynamics/hh_realtime_from_adapt_artifact.py:486-507`, `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py`
   - Try a narrow sweep such as horizon weights `(3,1)` or `(2,2)` to address the residual one-bin phase / frequency lag.
4. **Keep comparing against normalized metrics**
   - Use `pair_mae / pair_span_exact`, pair correlation, and dominant-peak error; do not use visible oscillation size alone as evidence of better control.

## Preventive Measures
- Compare candidate runs relative to exact observable span, not only on raw trace plots.
- Keep physics changes (`U`, `drive_A`, `drive_omega`) separate from controller-law changes when interpreting improvement.
- When reporting spectra quality, always include a normalized time-domain metric alongside the spectral picture.
- Treat visually larger oscillations as an observability change until normalized error metrics also improve.
