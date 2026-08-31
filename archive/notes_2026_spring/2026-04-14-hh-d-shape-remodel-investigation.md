# Investigation: HH d-shape forecast remodel

## Summary
GPT Pro’s main diagnosis holds up against the current controller code: the cleanest next seam is the exact-v1 forecast/checkpoint objective, not the continuous McLachlan law. The strongest practical next move is **shadow-only d-shape diagnostics first** — specifically curvature and excursion-under/over on the existing L=2 primary density coordinate `d = n0 - n1`, plus a light total-occupation `N` guardrail diagnostic — before any live scoring or gate changes.

## Symptoms
- Energy tracking is comparatively good while the site/pair-density reversal remains late in the target high-amplitude site3 lane.
- The recent gate-patch family appears exhausted: broad relaxations leak, tighter ones become inert.
- Shadow per-site telemetry showed that late-window runner-up candidates can have better turn geometry, and sometimes even better total tracking score, yet still be screened upstream.

## Investigation Log

### Phase 1 - Initial assessment
**Hypothesis:** GPT Pro is pointing at the correct seam: the live problem is in the forecast/checkpoint decision geometry, not in the continuous projective law.
**Findings:** The present controller already forecasts all relevant observables from the same candidate future state, and the strongest recent evidence is about decision-surface behavior: good alternates exist but are not being selected.
**Evidence:** `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1330-1361` stores exact/controller next-step values for primary density, site occupations, doublon, and energy in the step forecast payload. `notes/2026-04-13-hh-postcross-ranking-investigation.md` records that broad turn escapes leaked, while tighter ones became inert.
**Conclusion:** Confirmed directionally. The next seam to investigate is score/gate structure, not the continuous McLachlan ODE.

### Phase 2 - Verify the live L=2 coordinate
**Hypothesis:** GPT Pro’s recommendation to work in `(N,d)` rather than raw per-site coordinates matches the code’s existing live density channel.
**Findings:** For `L=2`, the controller already uses pair difference as the primary density target in auto mode.
**Evidence:** `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1291-1304` shows `_exact_forecast_primary_density_target_mode()` returning `"pair_difference"` for `L=2`, and `_primary_density_value_from_snapshot()` computing `site_occ[0] - site_occ[1]`.
**Conclusion:** Confirmed. The current controller is already d-centric for `L=2`, which makes a d-shape extension cleaner than promoting raw per-site terms directly.

### Phase 3 - Verify score asymmetry
**Hypothesis:** The score is materially richer for energy than for site occupation.
**Findings:** `_forecast_tracking_score()` is an additive weighted sum of base pointwise terms plus density extras and energy-shape extras. Energy contributes not only pointwise error, but also slope, curvature, and excursion-under/over penalties. Site occupation contributes only the pointwise max-site channel in the live score.
**Evidence:** `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:2624-2812`. Base live channels: fidelity, primary density, doublon, max-site occupation, and energy. Live extras: density slope/sign-lag/postcross/harmonic plus energy slope/curvature/under/over. No `tracking_site_*` term is consumed there.
**Conclusion:** Confirmed. GPT Pro’s “shared forecast, asymmetric decision surface” diagnosis is consistent with the actual scoring code.

### Phase 4 - Verify existing site-turn telemetry
**Hypothesis:** The repo may already have the mathematics needed for site-turn-aware diagnostics, but not as live score terms.
**Findings:** The code already computes per-site slope, curvature, and excursion-under/over telemetry and stores it on forecast payloads, but only for diagnostics/compare logic.
**Evidence:** `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:2230-2395` defines `_site_shape_tracking_terms()` and `_site_excursion_tracking_terms()`. `...controller.py:2491-2588` stores `tracking_site_slope_abs_error_mean_by_site`, `tracking_site_curvature_abs_error_mean_by_site`, `tracking_site_excursion_under_response_mean_by_site`, and `tracking_site_excursion_over_response_mean_by_site`. `...controller.py:2624-2812` does not use those fields in the live score.
**Conclusion:** Confirmed. The repo already has a shadow site-turn geometry surface; the main question is how to introduce a cleaner live d-shape analogue, not whether the code can represent slope/curvature/excursion ideas at all.

### Phase 5 - Verify stay/admission screening risk
**Hypothesis:** GPT Pro’s warning is correct: enriching ranking alone could still be screened upstream by older next-step gate logic.
**Findings:** The stay bounded-defect gate and append admission logic are still keyed to next-step scalar channels: primary density, density slope, sign lag, max-site error, fidelity, and energy. They do not know about any horizon-aggregated d-shape quantity.
**Evidence:** `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:2922-3013` (`_exact_v1_componentwise_aspiration_result()`), `...controller.py:3015-3094` (`_exact_v1_nonimproving_score_allows_density_first_append()`), and `...controller.py:3096-3118` (`_stay_forecast_within_exact_v1_bounded_defect()`).
**Conclusion:** Confirmed. A future live d-shape score patch would need gate alignment after ranking, or it risks another “live in score, screened in gates” outcome.

### Phase 6 - Verify energy’s current role
**Hypothesis:** GPT Pro’s recommendation to treat energy more like a guardrail than a co-equal scorer is not how the current code behaves end-to-end.
**Findings:** Energy plays two roles today: it is an ordinary additive weighted score term inside `_forecast_tracking_score()`, but it also becomes a harder admission/gating constraint in the below-floor energy-safe window, where energy shape caps can block candidates outright.
**Evidence:** Additive score role: `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:2624-2812`. Gating role: `...controller.py:6405-6458`, where `outside_energy_safe_window` and energy slope/curvature/excursion caps can veto candidates.
**Conclusion:** Confirmed. This mixed role explains why energy can dominate decisions more strongly than a casual look at the weighted sum alone would suggest.

### Phase 7 - Verify whether a total-occupation guardrail already exists
**Hypothesis:** If GPT Pro wants `(N,d)` with `N` as a guardrail, the current forecast seam may already expose `N` explicitly.
**Findings:** There is no explicit `N`/total-occupation scalar currently persisted at the forecast decision seam. The payload stores per-site occupations, from which `N` can be computed cheaply, but no dedicated total-occupation diagnostic is currently recorded.
**Evidence:** `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1330-1361` stores `site_occupations_controller_next` and `site_occupations_exact_next`, but no separate total-occupation field.
**Conclusion:** Confirmed. A small `N` guardrail diagnostic would need to be added explicitly.

### Phase 8 - Check the best diagnostic insertion seam
**Hypothesis:** The existing compare-diagnostic seam is the safest place to add shadow-only d-shape metrics first.
**Findings:** The compare seam already collects stay, selected-pre-override, and runner-up summaries, persists deltas vs stay, and is opt-in under the existing `exact_v1_postcross_compare_diag` flag. Anchor widening also already happens when this compare diagnostic is enabled.
**Evidence:** Compare summary helpers: `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1510-1580`. Compare assembly and persistence: `...controller.py:6757-6929` and `...controller.py:7696-7705`. Anchor widening when compare diag is enabled: `...controller.py:2419-2427`.
**Conclusion:** Confirmed. This is the lowest-risk seam for a shadow-only d-shape probe.

### Phase 9 - Oracle synthesis against verified code facts
**Hypothesis:** The cleanest next step is not to promote existing per-site telemetry directly, but to add a dedicated shadow-only d-shape diagnostic first.
**Findings:** Oracle agreed with GPT Pro’s high-level diagnosis but recommended a safer staged order for this codebase: **C first, then B, not A**. Specifically: add dedicated d-only helpers computed directly from `primary_density_controller_next/exact`, add a tiny `N` helper from summed site occupations, surface both only through the existing compare seam, and only if the shadow signal lights up should we consider promoting a d-shape block into live ranking/gates.
**Evidence:** Oracle chat `hh-postcross-rank-5454DA`, first response: “C first. Then, if it lights up, do B. Not A.” Follow-up response: use dedicated helpers `_d_shape_tracking_terms(...)`, `_d_excursion_tracking_terms(...)`, and `_total_occupation_tracking_terms(...)`; persist atomic shadow fields first, derive totals only in compare summaries.
**Conclusion:** Strongly supported. This is the best next move given the current code structure and recent failed gate-patch family.

### Phase 10 - Live below-floor d-shape gate-alignment probe
**Hypothesis:** A very narrow, default-off below-floor `d`-shape escape inside the existing energy-safe-window gate might admit the late good site3 runner-ups without reopening the earlier leakage problem.
**Findings:** The patch was mechanically correct and test-clean, but behaviorally too broad. It widened shadow `d`/`N` metric availability only when compare-diag or the new flag was enabled, then allowed a below-floor energy-safe-window escape whenever the candidate improved full forecast score, improved shadow-only `d`-shape total, and did not worsen raw next-step energy or total occupation `N`. Focused tests passed and the full controller file passed (`119 passed`), but the real replay at `artifacts/agent_runs/20260414_hh_l2_high_amp_d_shape_escape_compare_diag_site3_v1/` changed decisions early and unfavorably: append count dropped `8 -> 3`, first new divergence happened at `t=0.56`, and the controller admitted an early below-floor candidate that baseline had rejected as `outside_energy_safe_window`.
**Evidence:** New run artifact: `artifacts/agent_runs/20260414_hh_l2_high_amp_d_shape_escape_compare_diag_site3_v1/json/site3_d_shape_escape_compare_diag_w1p00.json`. Summary files: `.../json/d_shape_escape_compare_summary.json` and `.../json/d_shape_escape_compare_events.json`. Controller diff result versus the prior compare-diag lane: `row_diff_count_on_interesting_keys = 373`, first divergence at trajectory index `28` (`time=0.56`) where baseline stayed but the new run appended via `below_floor_componentwise_aspiration`. Late-window compare summary from `d_shape_escape_compare_summary.json`: `late_window_runner_better_d_shadow_only_count = 45`, but `late_window_runner_admitted_count = 0`; the late good runner-ups were no longer blocked primarily by `outside_energy_safe_window`, because the earlier divergence changed the trajectory and moved the late decision surface into mostly `no_target_win_vs_stay`. Metric comparison from the two JSON artifacts: baseline mean site error `0.19205`, late-window mean site error `0.27594`, final site error `3.1e-5`; new run mean site error `0.22244`, late-window mean site error `0.24330`, but final site error `0.58324`.
**Conclusion:** Eliminated as a production patch. The `d`-shape signal is real, but this local gate-alignment form leaks too early: it improves the target window somewhat while catastrophically damaging the final landing. The next move should not be to stack more escape clauses onto this same gate.

## Root Cause
The immediate issue is not that the controller cannot forecast site occupation. It can. The deeper issue is that the live exact-v1 decision surface is still organized around a relatively thin site signal and a richer energy signal. In code terms:
- the live score already operates on the L=2 antisymmetric density coordinate `d = n0 - n1`,
- the code already contains shadow per-site turn geometry,
- but the live ranking and stay/admission gates still privilege energy shape and older next-step scalar channels,
- and there is no live horizon-aggregated d-shape scalar that could express “the candidate is turning correctly even before the old sign-bucket logic activates.”

That makes GPT Pro’s proposed seam plausible, but the safest conclusion from this investigation is: **do not promote per-site telemetry directly and do not jump straight to a live behavioral patch.** First add a dedicated shadow-only d-shape diagnostic on the existing L=2 primary density coordinate, plus a light `N` guardrail diagnostic, and use the existing compare seam to determine whether the proposed d-shape signal is actually live on the target late window.

## Recommendations
1. **Add a shadow-only d-shape diagnostic block first.**
   - New dedicated helpers in `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py`:
     - `_d_shape_tracking_terms(...)`
     - `_d_excursion_tracking_terms(...)`
     - `_total_occupation_tracking_terms(...)`
   - Compute directly from `primary_density_controller_next/exact` and from summed site occupations for `N`.
2. **Persist only atomic shadow fields at first**, and only through the existing compare seam:
   - `tracking_d_curvature_abs_error_mean`
   - `tracking_d_excursion_under_response_mean`
   - `tracking_d_excursion_over_response_mean`
   - `tracking_total_occupation_abs_error_next`
   - `tracking_total_occupation_abs_error_mean`
   - Reuse existing live/diagnostic density-slope field instead of duplicating slope.
3. **Derive aggregate d-shape totals only in compare summaries**, not in the rollout payload yet.
   - Extend `_forecast_postcross_compare_summary()` / `_exact_v1_postcross_candidate_compare_entry()` to emit `d_shape_shadow_total` and deltas vs stay.
4. **If the shadow d-shape signal lights up on rejected runner-ups only**, the next live seam should be gate alignment, not just another ranking weight.
5. **If the shadow d-shape signal is absent/common-mode**, stop and do not promote it live; that would be another inert patch family.

## Preventive Measures
- When adding new exact-v1 geometry terms, attach them first to an opt-in compare seam that reports stay / selected / runner-up deltas rather than jumping straight into live behavior.
- Avoid promoting raw per-site terms directly in `L=2` when the controller is already organized around `d = n0 - n1`; otherwise weights become difficult to interpret and double-counting risk goes up.
- Keep ranking and gate logic aligned: a future live d-shape promotion should not be introduced into `_forecast_tracking_score()` alone without corresponding stay/admission updates.
