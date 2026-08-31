# Investigation: HH postcross ranking weight patch

## Summary
We confirmed the original ranking asymmetry, implemented the ranking-side postcross term, and then falsified it on the target site3 lane. The highest-value new finding is that late-window runner-up candidates can carry better per-site turn telemetry — and sometimes even better total tracking score — yet still get screened out upstream, but the first two escape-hatch patches were too broad and did not produce a clean objective win.

## Symptoms
- Floor-open sign-lag controls improved the bad local window but left the density/site reversal late and the dominant pair-density spectral line one bin low.
- A post-cross wrong-sign admission-floor reducer was inert on the target lane, suggesting the missing lever may be ranking rather than admission.

## Investigation Log

### Phase 1 - Initial assessment
**Hypothesis:** The controller already computes the right post-cross density object, but it is not yet used by ranking.
**Findings:** RepoPrompt context-builder + oracle and direct code reads agreed that the controller already computes and stores a post-cross wrong-sign debt term, but `_forecast_tracking_score()` did not consume it. The term was only lowering the local aspiration gain floor.
**Evidence:** `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:1586-1643` computes `_primary_density_postcross_wrong_sign_terms()`. `...controller.py:1777-1833` returns the normalized post-cross fields. `...controller.py:2166-2173` stores them into rollout payload as `tracking_primary_density_postcross_wrong_sign_*`. `...controller.py:2294-2383` scored density-slope/sign-lag, drive-harmonic, and energy extras, but had no post-cross ranking contribution before this patch. `...controller.py:2418-2463` used the post-cross term only for gain-floor reduction. For `L=2`, primary density auto-mode is already `pair_difference = n0 - n1` (`...controller.py:1289-1303`).
**Conclusion:** Confirmed: the missing first lever was ranking-side, not observability-side.

### Phase 2 - Ranking implementation seam
**Hypothesis:** A ranking-only post-cross patch can be made safely by adding a new forecast weight and leaving aspiration/override logic unchanged.
**Findings:** Added a new config/CLI/controller weight seam and ranking contribution for post-cross wrong-sign debt. Also widened rollout anchor creation when the new weight is enabled so the term cannot silently stay inactive when density-slope weight is zero.
**Evidence:** `pipelines/time_dynamics/hh_realtime_checkpoint_types.py` includes `exact_forecast_density_postcross_wrong_sign_weight`. `pipelines/time_dynamics/hh_realtime_from_adapt_artifact.py` exposes `--checkpoint-controller-exact-forecast-density-postcross-wrong-sign-weight` and threads it into `RealtimeCheckpointConfig`. `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py` validates the new weight, exposes `_exact_forecast_density_postcross_wrong_sign_weight()`, includes the new weight in `_forecast_tracking_score()`, and widens `_exact_forecast_rollout()` anchor creation when the new weight is active.
**Conclusion:** Implemented safely; next step was regression verification and bounded sweep.

### Phase 3 - Focused regression verification
**Hypothesis:** The patch should be regression-safe and should activate scoring without changing admission/override logic.
**Findings:** Focused regressions passed, and the canonical CLI help showed the new flag.
**Evidence:** Focused `pytest` passed after the ranking patch, and `python -m pipelines.time_dynamics.hh_realtime_from_adapt_artifact --help` included `--checkpoint-controller-exact-forecast-density-postcross-wrong-sign-weight`.
**Conclusion:** Safe enough to run the bounded site3 sweep.

### Phase 4 - Bounded site3 sweep
**Hypothesis:** A bounded sweep over the new ranking weight on the existing floor-open site3 lane will identify the smallest useful weight, if any.
**Findings:** The sweep falsified the ranking-weight hypothesis on this lane. We varied only `--checkpoint-controller-exact-forecast-density-postcross-wrong-sign-weight` over `0.0, 0.25, 0.5, 0.75, 1.0`, and every headline metric remained exactly unchanged. Stronger still: `w0` and `w1` had zero differing keys across all 401 trajectory rows; only metadata/config changed.
**Evidence:** Sweep bundle: `artifacts/agent_runs/20260413_hh_l2_high_amp_postcross_rank_sweep_site3_v1/`. The generated `json/summary.json` reports identical metrics for all five weights and `selection.best_weight = None`.
**Conclusion:** Eliminated: on this lane, the new ranking term was not a live discriminator in the executed decision path.

### Phase 5 - Why the artifacts looked silent
**Hypothesis:** The controller may be computing the new term internally while omitting it from persisted trajectory rows, which would hide whether it is zero or merely non-binding.
**Findings:** The run loop serialized a narrow subset of forecast fields into trajectory rows and did not include the new post-cross tracking fields. The controller computed and stored `tracking_primary_density_postcross_wrong_sign_*` at rollout time, but trajectory persistence kept only selected headline forecast values like fidelity and abs-error next-step terms.
**Evidence:** The tracking fields are attached in `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:2177-2189`. The trajectory serialization block at `...controller.py:7168-7237` records `forecast_stay_*` / `forecast_selected_*` headline error fields but not the post-cross tracking fields.
**Conclusion:** Confirmed: the sweep proved the term was inert, but the artifacts did not yet tell us whether it was numerically inactive or merely non-binding.

### Phase 6 - Compare-time postcross diagnostic replay
**Hypothesis:** The sweep left one ambiguity: the post-cross term might be computed but hidden/common-mode. An opt-in stay-vs-selected-vs-runner-up compare snapshot should tell us whether the term is inactive, common-mode, or filtered away.
**Findings:** The diagnostic closed that ambiguity. On the full instrumented `w=1.0` replay, the post-cross term was almost entirely absent from compared forecasts, and completely absent in the actual bad late window. Across 400 diagnostic rows, `selected_postcross_active_count = 0`, `late_window_selected_active_count = 0`, `late_window_runner_active_count = 0`, and the only nontrivial event occurred earlier at `t = 2.6`, where a single runner-up candidate had active post-cross debt but was rejected with `fails_energy_cap` and was worse than stay on both post-cross contribution and total score.
**Evidence:** Instrumented replay bundle: `artifacts/agent_runs/20260413_hh_l2_high_amp_postcross_compare_diag_site3_v1/`. Diagnostic summary: `.../json/postcross_compare_summary.json`. Diagnostic events: `.../json/postcross_compare_events.json`.
**Conclusion:** Confirmed: the post-cross ranking term was not merely too weak; on the target late window it was not a live compared quantity at all.

### Phase 7 - Horizon-geometry falsifier
**Hypothesis:** The compared forecast might simply be too short; extending the horizon should wake up post-cross activation in the late bad window if that is the missing seam.
**Findings:** Extending the horizon to 6 with weights `8,4,2,1,1,1` did make late-window runner-up postcross activation appear, but only on runner-up candidates, never on the selected candidate, and never with a beneficial margin vs stay. It also made the whole lane much worse.
**Evidence:** Bundle: `artifacts/agent_runs/20260413_hh_l2_high_amp_horizon6_postcross_compare_diag_site3_v1/`. Summary showed `late_window_runner_active_count = 11` and `late_window_selected_active_count = 0`, while the replay degraded headline metrics to `mean_site_error = 0.2534`, `window_mean_site_error = 0.4282`, `mean_pair_error = 0.5067`, `window_mean_pair_error = 0.8564`.
**Conclusion:** Eliminated as a fix. The 4-step horizon was not the whole reason postcross was inert.

### Phase 8 - Shadow-only per-site turn telemetry
**Hypothesis:** The scalar density mode `d = n0 - n1` may be too compressed to explain the failure; per-site slope/curvature/excursion telemetry may expose a hidden site-turn pattern.
**Findings:** Added shadow-only per-site compare telemetry to the existing compare seam and reran the original 4-step lane. The replay was trajectory-identical to the floor-open baseline after stripping the diagnostic payload, so the telemetry itself was truly noninvasive. The new telemetry showed that late-window runner-up candidates sometimes had better per-site turn geometry than stay, and in 7 rows even had better total tracking score than stay, yet were still rejected upstream with `outside_energy_safe_window`.
**Evidence:** Telemetry replay bundle: `artifacts/agent_runs/20260413_hh_l2_high_amp_site_turn_compare_diag_site3_v1/`. It is identical to baseline on all non-diag fields. In the late window, 20 runner-up rows showed better combined per-site turn telemetry, and 7 of those also had `tracking_score_delta_vs_stay < 0`; e.g. at `t = 6.66`, runner-up `drive_aligned_density(pattern=staggered)` had better combined site-turn telemetry and `tracking_score_delta_vs_stay = -4.1669`, yet was rejected by `outside_energy_safe_window`.
**Conclusion:** This is the sharpest remaining diagnosis. The next live seam moved from ranking to the blanket below-floor energy-safe-window rejection.

### Phase 9 - First below-floor turn-escape patch
**Hypothesis:** A narrow, default-off escape hatch on the blanket `outside_energy_safe_window` rejection could admit candidates that are better on total tracking score, better on combined per-site turn telemetry, and no worse on raw next-step energy.
**Findings:** The patch was live and regression-safe in tests, but too broad in practice. On the real site3 replay it fired too early (starting around `t = 0.56` / `0.64`), pulled the trajectory onto a different branch, improved the target window, but made the overall lane worse.
**Evidence:** Escape bundle: `artifacts/agent_runs/20260413_hh_l2_high_amp_turn_escape_site3_v1/`. Compared with baseline, it improved the target window (`window_mean_site_error = 0.2433`, `window_mean_pair_error = 0.4866`) but regressed globally (`mean_site_error = 0.2224`, `mean_pair_error = 0.4449`) and changed 373 trajectory rows. First divergence came at `t = 0.56`, where a below-floor candidate was newly admitted.
**Conclusion:** Directionally real, but the activation was too global.

### Phase 10 - Sign-lag-localized turn-escape patch
**Hypothesis:** If the escape is gated by the existing sign-lag window, it should only fire in the delayed-reversal regime and avoid the early harmful branch changes.
**Findings:** Gating by the existing sign-lag window did suppress the very early `t ≈ 0.56` firings, but it still started changing the lane around `t ≈ 2.7` and did not produce an objective win. The resulting replay was better than the first broad escape, but still worse than the baseline both globally and in the target window.
**Evidence:** Bundle: `artifacts/agent_runs/20260413_hh_l2_high_amp_turn_escape_signlag_site3_v1/`. Compared with baseline, it yielded `mean_site_error = 0.1998`, `mean_pair_error = 0.3996`, `window_mean_site_error = 0.2954`, `window_mean_pair_error = 0.5909`, with 266 changed trajectory rows. First divergence moved to `t = 2.7`, where a below-floor candidate was newly admitted under the localized escape.
**Conclusion:** Improved localization, but still not the right corrective lever.

### Phase 11 - Stay-side density/sign-lag-localized turn-escape patch
**Hypothesis:** Requiring the stay forecast itself to show existing density/sign-lag failure, plus a material primary-density win, would eliminate the remaining leakage while preserving the useful late-window admits.
**Findings:** This tighter gate eliminated the remaining leakage, but it also eliminated all behavioral effect. The replay was trajectory-identical to baseline after stripping the diagnostic payload.
**Evidence:** Bundle: `artifacts/agent_runs/20260413_hh_l2_high_amp_turn_escape_signlag_densityfail_site3_v1/`. Compared with baseline, every headline metric matched exactly (`mean_site_error = 0.1920546920`, `window_mean_site_error = 0.2759405407`, `mean_pair_error = 0.3841093840`, `window_mean_pair_error = 0.5518810813`, `append_count = 8`), and direct row diff showed `0` non-diagnostic changes.
**Conclusion:** Clean stop signal. This gate family is now exhausted: broad versions leak, tighter versions become inert.

## Root Cause
The original code asymmetry was real: the controller computed a primary-density post-cross wrong-sign debt but did not include it in ranking. That gap is now fixed in code. But the deeper issue on this site3 lane is not just a missing ranking weight. The most diagnostic result is Phase 8: late-window runner-up candidates can have better per-site turn telemetry and even better total tracking score than stay, yet still be screened out by the blanket below-floor energy-safe window. However, the first two escape-hatch patches show that directly relaxing that gate is delicate: broad relaxation leaks too early, and sign-lag-localized relaxation still alters the lane too early and does not improve the true objective.

## Recommendations
1. **Do not keep either turn-escape patch enabled by default.** Leave `exact_v1_below_floor_energy_safe_turn_escape = False`.
2. Treat the new per-site compare telemetry as the durable win from this investigation. It gave us the first concrete evidence that the late-window issue is not just scalar-density ranking, but also an upstream gate/filtering geometry problem.
3. The next safest move is a tighter, stay-side density-failure localization built from existing density/sign-lag telemetry rather than more blanket gate relaxation. If we continue on the gate path, the next patch should be narrower than the sign-lag-localized escape, not broader.
4. If we want to stop patching and bank evidence, the strongest current statement is: *the controller now has a useful site-turn diagnostic, but simple below-floor energy-safe-window escapes do not yet convert that insight into a stable objective improvement.*

## Preventive Measures
- Add focused score/anchor tests whenever a new exact_v1 ranking term depends on rollout anchor construction.
- When adding a new internal ranking or gating term, also expose a minimal persisted score-decomposition field at the forecast comparison seam so sweeps can distinguish “inactive”, “non-binding”, and “screened upstream”.
- Prefer reusing existing localized controller seams (like sign-lag windows) before introducing new broad policy relaxations.