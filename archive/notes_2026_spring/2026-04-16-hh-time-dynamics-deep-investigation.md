# Investigation: HH site3 time-dynamics deep investigation

## Summary
The strongest current read is now split cleanly into two levels: the **best immediate local seam** is a real two-surface commit-law mismatch, while the **best global-cause seam** remains upstream stay/baseline basin construction. The late append family is no longer the default first suspect, but a shortlist/pool divergence question is back on the table because the `v18` late rescue family is absent from the recorded `v19` confirmed top-k at the late window. Current Optuna is still acting more like a feasibility filter than a ranking tool.

## Symptoms
- Repaired guarded d-shape run `v18` still stopped early at checkpoint 120 / `t=2.4`.
- New fidelity-first run `v19` also stopped early at checkpoint 120 / `t=2.4`.
- Plain fixed-manifold McLachlan control is substantially worse than both controller lanes.
- Current fidelity-first Optuna study has zero valid completed trials.

## Investigation Log

### Phase 1 - Initial assessment
**Hypothesis:** The remaining problem is upstream stay/baseline geometry or shared veto logic, not lack of a late append family.
**Findings:** Context-builder assessment ranked baseline/stay forecast construction as the highest-confidence seam, with guardrail/screening second and Optuna feasibility gating third.
**Evidence:** `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py`; `pipelines/time_dynamics/optimization/hh_realtime_optuna.py`; notes from 2026-04-11, 2026-04-13, 2026-04-14.
**Conclusion:** Confirmed as current working hypothesis; requires direct file/artifact verification.

### Phase 2 - Run artifact comparison: v18 vs v19
**Hypothesis:** Fidelity-first changed the core score but may not have fixed late-window behavior.
**Findings:**
- `v18` stopped early with `append_count=6`, `stay_count=115`, final site error `0.5748`, final fidelity `0.2815`, final energy error `0.2101`.
- `v19` stopped early with `append_count=3`, `stay_count=118`, final site error `0.6828`, final fidelity `0.2379`, final energy error `0.1341`.
- Earliest decision divergence is checkpoint 3 / `t=0.06`: `v18` stayed with `outside_exact_turn_window`, while `v19` appended via `below_floor_componentwise_aspiration`.
- By the late failure window, `v18` repeatedly reaches `d_shape_barrier_protected_horizon`, but `v19` is still stuck on `no_tracking_win_vs_stay`.
**Evidence:**
- `artifacts/agent_runs/20260415_hh_l2_high_amp_d_shape_guardrail_rank_site3_v18_prefsite_autobaseline_v1/json/site3_d_shape_guardrail_rank_prefsite_autobaseline_v1.json`
- `artifacts/agent_runs/20260415_hh_l2_high_amp_fidelity_first_guardrail_rank_site3_v19_fidcore_v1/json/site3_fidelity_first_guardrail_rank_fidcore_v1.json`
**Conclusion:** Fidelity-first is behaviorally different, but not in the way needed; it creates some early aspiration behavior and loses the late protected-horizon rescue pattern.

### Phase 3 - Controller code seams
**Hypothesis:** Baseline selection and shared guardrails can dominate the final policy even after a score change.
**Findings:**
- `_select_exact_v1_baseline_step_scale()` constructs the stay forecast by searching `norm_locked_blend_v1` candidates plus optional tangent-secant proposals, then refining step scale and gain scale against rollout score.
- `_forecast_tracking_score()` dispatches to either live d-shape score, fidelity-first score, or generic score.
- `fidelity_first_barrier_v1` uses generic forecast score with `include_energy_core_terms=False`, but still adds the existing live d-barrier penalty.
- Protected-horizon / target-win checks still require guarded turn-window score improvement, shadow-turn win, and `_exact_forecast_override_reason()` to pass.
- `_exact_v1_componentwise_aspiration_result()` and `_exact_v1_below_floor_energy_safe_window()` still impose fidelity / energy / site / total-occupation style restrictions.
**Evidence:**
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:3438-3466`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:3818-3898`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:3969-4051`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:4055-4098`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:4236-4308`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:4477-4504`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:5728-6157`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:8083-8128`
**Conclusion:** Confirmed: the score swap does not bypass the shared barrier/veto structure, and the stay baseline search remains a major upstream decision point.

### Phase 4 - Optuna feasibility
**Hypothesis:** The current study is dominated by validity gates and may be hiding useful ranking signal.
**Findings:**
- `_default_validity_gates()` for density-first feasible profiles require `max_mean_abs_energy_total_error<=0.25`, `max_final_abs_energy_total_error<=0.25`, `min_fidelity_exact>=0.90`, and `max_mean_total_occupation_abs_error<=0.05`.
- Current study command uses profile `high_amp_guarded_fidelity_first_site3_prefsite_autobaseline_v1`.
- The study currently has `0` valid completed trials and `15` invalid trials.
- Representative invalid trials still show meaningful controller structure (`fidelity_first_barrier_protected_horizon` selections, nontrivial site error), but are rejected by the same energy/fidelity gates.
- The current fidelity-first Optuna profile is also still a heavily locked surface: `_search_space()` and `_baseline_trial_params()` use the same site3 prefsite/autobaseline helper for both guarded d-shape and fidelity-first profiles, with the profile-specific change coming primarily from `guardrail_mode` rather than a broad baseline-surface redesign.
**Evidence:**
- `pipelines/time_dynamics/optimization/hh_realtime_optuna.py:212-229`
- `pipelines/time_dynamics/optimization/hh_realtime_optuna.py:230-251`
- `pipelines/time_dynamics/optimization/hh_realtime_optuna.py:360-419`
- `pipelines/time_dynamics/optimization/hh_realtime_optuna.py:588-655`
- `artifacts/agent_runs/20260415_hh_l2_high_amp_fidelity_first_site3_prefsite_autobaseline_optuna_v1/logs/command.sh`
- `artifacts/agent_runs/20260415_hh_l2_high_amp_fidelity_first_site3_prefsite_autobaseline_optuna_v1/progress.json`
- `.../trials/trial_0000/trial_summary.json`
- `.../trials/trial_0010/trial_summary.json`
- `.../trials/trial_0014/trial_summary.json`
**Conclusion:** Confirmed: Optuna is currently functioning more as a hard feasibility alarm than as a useful ranking study, and it is not yet exploring a very broad fidelity-first surface.

### Phase 5 - First subagent synthesis
**Hypothesis:** Parallel focused investigations will sharpen whether the dominant near-term seam is baseline construction or shared policy mismatch.
**Findings:**
- Divergence-timeline agent: checkpoint `102` is the first late point of no return; in the late window `v19` fails on guarded core win (`no_tracking_win_vs_stay`) before shadow-turn or override checks become decisive.
- Guardrail/override agent: the important local mismatch is that below-floor aspiration admits on guarded-core logic, but later full-score override can force the action back to stay as `exact_forecast_nonimproving_tracking_score`.
- Baseline-basin agent: the stay baseline is upstream and likely already different between `v18` and `v19`; secant untapering looks secondary compared with the score/guardrail swap.
- Fidelity-first mechanics agent: `fidelity_first_barrier_v1` is a hybrid policy, not a clean rewrite — it keeps the live d-barrier, uses the shared guarded machinery, and loses the d-shape-only outside-turn probe delay, which explains the early-aspiration / late-nonrescue pattern.
- Optuna-signal agent: current Optuna is a feasibility alarm first and ranking study second; it also under-counts fidelity-first protected-horizon events because the aggregate metric only counts the d-shape string.
**Evidence:**
- Subagent memos returned on 2026-04-16 from divergence timeline, guardrail override audit, baseline basin audit, fidelity-first mechanics, and Optuna signal audit.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:4071-4109`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:4236-4267`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:4477-4505`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:5272-5283`
- `pipelines/time_dynamics/optimization/hh_realtime_optuna.py:2288-2305`
**Conclusion:** Initial ranking update: best global cause = upstream stay/baseline geometry; best immediate diagnostic seam = shared guarded-policy / override mismatch.

### Phase 6 - Checkpoint packet + probe replay
**Hypothesis:** A checkpoint-only replay will tell us whether the next seam is a real policy contradiction, a path-dependent early-append artifact, or a deeper stay/pool problem.
**Findings:**
- Built shared packet: `notes/2026-04-16-hh-site3-checkpoint-packet.json` covering checkpoints `3,7,11,15,31,35,61,67,102,105,115,117,118,119,120` plus benchmark lanes `site3_signlag_flooropen` and `ub401_site3`.
- Ran checkpoint probes in `artifacts/agent_runs/20260416_hh_l2_high_amp_site3_checkpoint_packet_probe_v1/`:
  - `site3_v18_checkpoint_probe.json`
  - `site3_v19_checkpoint_probe.json`
- `v18` parity is exact at late rescue rows `102/115/117/118/119/120`.
- `v19` actual-prefix has explicit contradiction rows at `11/15/31/35/67`: candidate admitted as `below_floor_componentwise_aspiration`, final action `stay`, override `exact_forecast_nonimproving_tracking_score`.
- `v19 force_stay(3,7,61)` changes the mid-window path, but by `102+` both paths reconverge into `no_tracking_win_vs_stay -> stay`.
- `v18` late rescue labels are absent from the recorded `v19` confirmed top-k at `102/115/117/118/119/120` in both actual-prefix and force-stay probes.
- Probe code confirms the telemetry boundary: `debug_probe_exact_v1()` only emits `confirmed[:candidate_rank_limit]`, so current evidence proves “absent from recorded confirmed top-k,” not full-pool absence.
**Evidence:**
- `notes/2026-04-16-hh-site3-checkpoint-packet.json`
- `notes/2026-04-16-hh-site3-checkpoint-probe-summary.md`
- `artifacts/agent_runs/20260416_hh_l2_high_amp_site3_checkpoint_packet_probe_v1/json/site3_v18_checkpoint_probe.json`
- `artifacts/agent_runs/20260416_hh_l2_high_amp_site3_checkpoint_packet_probe_v1/json/site3_v19_checkpoint_probe.json`
- `pipelines/time_dynamics/hh_realtime_checkpoint_probe.py:24-95`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:8710-9179`
**Conclusion:** The probe elevates two seams simultaneously: (1) a real local commit-law mismatch, and (2) a deeper late shortlist/stay-basin divergence that survives removal of the early `v19` appends.

### Phase 7 - Refined six-agent synthesis after probe
**Hypothesis:** Once the checkpoint probe exists, the strongest remaining uncertainty is whether the next real move should be commit-law unification, stay-basin redesign, or pool/shortlist instrumentation.
**Findings:**
- Refined alignment agent: first branch split = checkpoint `3`; last shared-recoverable checkpoint = `67`; first shared benchmark-basin exit = `102`; `v18` peels off the floor-open basin earlier at `61`.
- Commit-law contradiction agent: verified a true two-surface mismatch in code. `_exact_v1_componentwise_aspiration_result()` compares guarded **core-only** score, but `_exact_v1_forecast_override_reason()` later compares full `_forecast_tracking_score(...)`; in fidelity-first mode this means core-only admission can still be vetoed by core-plus-barrier total score.
- Rescue-family presence agent: current probe does **not** prove full-pool absence; it proves absence from recorded confirmed top-k. So late rescue-family failure is at least upstream of final late override, but may still be shortlist/pool demotion rather than absolute pool absence.
- Baseline hardening agent: found a second hidden branch fork. Because `_exact_v1_preferred_site_index_at_time()` returns `None` unless `d_shape_barrier_v1` is active, fidelity-first loses the d-shape-only preferred-site baseline scale extension, candidate scale extension, and shortlist injection hooks. That creates a hidden search-surface fork in addition to the score fork.
- Path-dependence agent: forcing away early `v19` appends only changes the mid-window gate path; both paths reconverge to the same late `no_tracking_win_vs_stay -> stay` basin by checkpoint `102`.
**Evidence:**
- Agent outputs from 2026-04-16: lane alignment, commit-law contradiction, rescue-family presence, baseline hardening, path dependence.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:3818-3850`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:4236-4267`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:4477-4505`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:5161-5183`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:5200-5211`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:6038-6059`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:7549-7579`
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:7767-7779`
**Conclusion:** Current strongest diagnosis is now three-part:
1. **Immediate local seam:** unify admission and final commit on one score surface.
2. **Next global seam:** explain why fidelity-first loses the d-shape-only preferred-site search machinery and hardens onto the late anti-drive stay basin.
3. **Then:** instrument the full pool/shortlist/confirm chain if the late rescue family still does not reappear.

### Phase 8 - Final Oracle synthesis
**Hypothesis:** The finished packet + probe + six-agent pass should let us separate the best local fix from the best benchmark-moving fix.
**Findings:** Oracle’s final synthesis cleanly separated four levels:
1. **Immediate local seam:** single-surface commit-law alignment.
2. **Benchmark-moving seam:** remove the hidden preferred-site search-surface fork in fidelity-first.
3. **Global-cause seam:** upstream late stay/baseline basin hardening.
4. **Later diagnostic seam:** full `pool -> scout -> shortlist -> confirm` rescue-family audit at `102+` if the family still does not reappear.
**Evidence:** Oracle chat `hh-site3-failure-map-7A489D`, final synthesis after the checkpoint packet, probe replay, refined subagents, and benchmark translation pass.
**Conclusion:** Final ranking is now stable enough to carry forward into the report and the next intervention choice.

## Root Cause
Current leading diagnosis is now layered:

1. **Immediate local defect:** a real two-surface commit-law mismatch. In guardrail mode, `_exact_v1_componentwise_aspiration_result()` can admit on guarded **core-only** score, but `_exact_v1_forecast_override_reason()` later judges the same append against full `_forecast_tracking_score(...)` and can cancel it as `exact_forecast_nonimproving_tracking_score`.
2. **Likely global cause:** the controller is still selecting the wrong upstream stay/baseline basin. Removing the early `v19` appends does not recover the late `v18` rescue regime; both `v19` paths reconverge to the same late `no_tracking_win_vs_stay -> stay` basin by checkpoint `102`.
3. **Important hidden branch fork:** `fidelity_first_barrier_v1` is not just a score swap. Because `_exact_v1_preferred_site_index_at_time()` is gated on `d_shape_barrier_v1`, fidelity-first also loses the d-shape-only preferred-site baseline scale extension, candidate scale extension, and shortlist injection. So the branch comparison is confounded by both a **score-law fork** and a **search-surface fork**.
4. **Open but narrowed question:** the probe only records confirmed top-k candidates, so the late `v18` rescue-family absence in `v19` is currently evidence of shortlist/ranking divergence, not yet proof of full-pool absence.

## Recommendations
1. Test the **single-surface commit law** first: the same score surface that admits a below-floor/protected-horizon candidate should decide final commit-vs-stay.
2. Immediately after that, remove or parity-check the **hidden preferred-site search-surface fork** in fidelity-first: give fidelity-first the same preferred-site baseline/candidate-scale/shortlist hooks before judging the branch against the site3 benchmarks.
3. If the late rescue family still does not reappear, instrument the full `candidate_pool_terms -> scout_candidates -> shortlist_records -> confirm_candidates` chain at `102/115/117/118/119/120` to settle whether the family is absent from the pool or merely demoted before the recorded top-k.
4. Treat current Optuna outputs as feasibility evidence until at least one valid trial exists, and do not over-interpret the current fidelity-first profile as a broad search surface.

## Preventive Measures
- Preserve artifact-level selection/override telemetry in all proof runs.
- Keep run-to-run controller config diffs explicit in reports.
