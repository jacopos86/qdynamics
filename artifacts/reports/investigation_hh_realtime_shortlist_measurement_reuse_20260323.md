# Investigation: HH realtime shortlist/probe and adaptive measurement reuse

## Summary
Current live HH execution does not have an active realtime McLachlan/projected-real-time controller. The repo does contain shortlist/probe/batching primitives, but they are wired only into ADAPT candidate selection, while realtime-specific shortlist/probe/beam logic survives only in legacy artifacts. Measurement policy in the active oracle path is fixed-shot/fixed-repeat, and the only live reuse primitive is an accounting-only measurement-cache audit rather than reuse of measured geometry data.

## Symptoms
- Historical measured McLachlan smoke artifacts show large per-estimate and trajectory-level transition/ancilla workload even for `time_end = 0.01`.
- The user suspects current workflow is not capitalizing on past measurements, lacks realtime shortlisting/probing, and lacks change-aware measurement intensity.
- The user wants to defer hybrid classical/QPU, shadows, and stochastic metric surrogates for now.

## Investigation Log

### Phase 1 - Current live HH dynamics path
**Hypothesis:** The current live stack may still contain a realtime McLachlan/projected-real-time engine.
**Findings:** I found no active realtime source path. The staged CLI exposes direct propagation controls and static noise controls, while the live workflow routes into replay and direct exact/Suzuki/CFQM propagation.
**Evidence:**
- `pipelines/` and `src/` search for `realtime_vqs|projected_real_time|phase_b_enabled` returned 0 source matches.
- `pipelines/hardcoded/hh_staged_cli_args.py:161-177` defines only `--noiseless-methods`, `--t-final`, `--num-times`, `--trotter-steps`, `--exact-steps-multiplier`, and CFQM options in the dynamics section.
- `pipelines/hardcoded/hh_staged_cli_args.py:228-234` defines static noise controls: `--shots`, `--oracle-repeats`, `--oracle-aggregate`.
- `pipelines/hardcoded/hubbard_pipeline.py:1425-1431` accepts only `suzuki2`, `piecewise_exact`, `cfqm4`, `cfqm6` as propagators.
- `artifacts/json/20260320_hh_l2_g0p5_nph2_ptw_live_beam3_window3_replay_fgv2.json:542-548` shows `"realtime_vqs": {"mode": "off", "phase_b_enabled": false, ...}`.
**Conclusion:** Confirmed. The current live path is replay + direct propagation, not live realtime McLachlan.

### Phase 2 - Existing shortlist/probe mechanisms
**Hypothesis:** The repo may already have shortlist/probe logic that could reduce cost, but it may live upstream of propagation.
**Findings:** The repo does have shortlist/probe/batching logic, but it is implemented in ADAPT continuation/candidate selection rather than realtime propagation.
**Evidence:**
- `pipelines/hardcoded/hh_continuation_stage_control.py:13-72` defines `allowed_positions(...)`, `detect_trough(...)`, and `should_probe_positions(...)`.
- `pipelines/hardcoded/adapt_pipeline.py:3470-3503` invokes `should_probe_positions(...)`, expands `positions_considered` via `allowed_positions(...)`, and switches between append-only and probe evaluation.
- `pipelines/hardcoded/hh_continuation_scoring.py:456-485` implements `shortlist_records(...)` for cheap shortlist trimming.
- `pipelines/hardcoded/adapt_pipeline.py:4018-4048` applies `greedy_batch_select(...)` to full records when phase-2 batching is enabled.
- `pipelines/hardcoded/hh_continuation_scoring.py:928-978` shows `greedy_batch_select(...)` batching shortlist records with compatibility penalties.
**Conclusion:** Confirmed. We have shortlist/probe/batching primitives only upstream in ADAPT selection.

### Phase 3 - Measurement reuse and batching already present
**Hypothesis:** There may already be some measurement-reuse plumbing that could be leveraged.
**Findings:** There is partial reuse logic, but it is accounting/scoring-oriented, not actual reuse of measured geometry data across realtime checkpoints.
**Evidence:**
- `pipelines/hardcoded/hh_continuation_scoring.py:141-193` defines `MeasurementCacheAudit` and its docstring says `"Phase 1 accounting-only grouped reuse tracker."`
- `pipelines/hardcoded/hh_continuation_scoring.py:159-178` computes reused/new groups and nominal shot counts only from group-key coverage, not from stored measured expectation data.
- `pipelines/hardcoded/adapt_pipeline.py:2773` constructs `phase1_measure_cache = MeasurementCacheAudit(nominal_shots_per_group=1)`.
- `pipelines/hardcoded/adapt_pipeline.py:3361-3362` and `3615-3616` call `phase1_measure_cache.estimate(measurement_group_keys_for_term(...))` during candidate scoring.
- `pipelines/hardcoded/adapt_pipeline.py:5097-5100` only `commit(...)`s measurement keys after selection.
**Conclusion:** Confirmed. Partial grouped-reuse/accounting exists, but not persistent checkpoint-level reuse of measured metric/force/cross-term data.

### Phase 4 - Non-smoothness / directional-change penalties
**Hypothesis:** The repo may already penalize trajectory non-smoothness or directional changes somewhere in the stack.
**Findings:** I found no propagation-side smoothness/direction penalty in the live path. The closest live source mechanisms are candidate-selection proxies: curvature, measurement mismatch, schedule overlap, trough detection, and novelty.
**Evidence:**
- `pipelines/hardcoded/hh_continuation_scoring.py:286-305` defines `_measurement_group_overlap_score(...)` with an internal `_directional(...)`, but it measures basis-cover overlap between measurement-group keys, not path smoothness.
- `pipelines/hardcoded/hh_continuation_scoring.py:824-909` computes compatibility penalties from `support_overlap`, `noncommutation`, `cross_curvature`, `schedule`, and `measurement_mismatch` between candidate terms.
- `pipelines/hardcoded/hh_continuation_scoring.py:624-680` defines `Phase2CurvatureOracle`, described as a shortlist-only metric/curvature proxy.
- `pipelines/hardcoded/hh_continuation_stage_control.py:57-72` uses plateau/flat-family conditions to trigger probing, but this is still ADAPT-selection control.
**Conclusion:** Confirmed. No live propagation-side non-smoothness/directional-change penalty is implemented; current mechanisms are candidate-selection proxies only.

### Phase 5 - Historical realtime artifacts
**Hypothesis:** Historical realtime artifacts may show the exact kind of shortlist/probe controller the user is asking about.
**Findings:** Legacy realtime artifacts do show realtime-side shortlist/probe/beam policy and branch actions, but they are not live in current source, and the winning branch still ended in mixed failure.
**Evidence:**
- `artifacts/json/hh_staged_adapt_vqs_l2_g0p5_nph2_phaseb_only_realtime_vqs.json:45723-45742` contains realtime policy fields:
  - `"max_shortlist": 4`
  - `"probe": {"horizon_steps": 1}`
  - `"beam": {"live_branches": 1, ...}`
- `artifacts/json/hh_staged_adapt_vqs_l2_g0p5_nph2_phaseb_only_realtime_vqs.json:53355-53417` shows branch ledger entries with `"action_kind": "stay"` and `"append_candidate"`, with some branches ending in `"terminal_reason": "beam_pruned"`.
- `artifacts/json/hh_staged_adapt_vqs_l2_g0p5_nph2_phaseb_only_realtime_vqs.json:56552-56567` shows the winner had `"terminal_reason": "final_time"` but `"endpoint_failure_classification": "mixed_failure"`.
- `artifacts/json/20260320_hh_l2_g0p5_nph2_ptw_live_beam3_window3_replay_fgv2.json:112-123` marks derivative capability as `"phase_b_supported": false`, `"runtime_submission_blocked": true`, and `"static_hamiltonian_only": true`.
**Conclusion:** Confirmed. Historical realtime shortlist/probe/beam exists in artifact lineage, not in the current live code path.

### Phase 6 - Adaptive measurement intensity
**Hypothesis:** Current measurement policy may already support escalating/de-escalating shots or repeats when changes are detected.
**Findings:** The active oracle path uses static `shots`, `oracle_repeats`, and `oracle_aggregate`. Historical artifacts record defect/stabilization telemetry, but I found no live mechanism that changes measurement intensity in response.
**Evidence:**
- `pipelines/exact_bench/noise_oracle_runtime.py:31-46` defines `OracleConfig` with fixed `shots`, `oracle_repeats`, and `oracle_aggregate`.
- `pipelines/exact_bench/noise_oracle_runtime.py:1150-1178` normalizes those fields into `ExpectationOracle.config` and only validates aggregate/fallback mode; no adaptive policy is introduced there.
- `artifacts/json/phase_a_local_rehearsal_20260317_134642/light_shots_fixed_patch_smoke.bundle.json:274-333` projects a large fixed workload per estimate and over the nominal trajectory.
- `artifacts/json/phase_a_local_rehearsal_20260317_134642/active_window_shots_fixed_patch_smoke.bundle.json:750-770` records `epsilon_proj_sq`, `epsilon_step_sq`, `stabilization_gap_sq`, `damping_gap_sq`, and `active_window_applied: false`, but these are telemetry fields, not adaptive shot-control decisions.
**Conclusion:** Confirmed. Change-aware measurement escalation/de-escalation is absent from the live path.

## Root Cause
The main issue is structural rather than a single bug. The current live HH stack no longer runs a realtime McLachlan/projected-real-time controller at all; instead it runs ADAPT selection and then switches to fixed-family replay plus exact/Suzuki/CFQM propagation. Because of that architecture:

1. Existing shortlist/probe/batching mechanisms live only in ADAPT operator selection, not inside a realtime checkpoint loop.
2. Existing measurement-reuse support is only an accounting/compatibility aid (`MeasurementCacheAudit`), not a persistent cache of measured geometry data.
3. Existing oracle execution keeps measurement intensity fixed via `shots` and `oracle_repeats`, with no change-triggered escalation or calm-segment de-escalation.
4. Historical realtime artifacts demonstrate that the repo once carried the right family of concepts (`stay`, `append_candidate`, shortlist, horizon-1 probe, beam pruning, defect telemetry), but those concepts are not presently wired into the active live code path.

## Recommendations
1. Treat realtime shortlist/probe as **not currently implemented in live execution**; when discussing future work, refer to current ADAPT shortlist/probe as the nearest reusable precursor, not as an existing realtime feature.
2. Prioritize a future realtime checkpoint controller that explicitly owns:
   - checkpoint-level observable pooling and data reuse,
   - `stay` vs `append_candidate` shortlist/probe decisions,
   - defect/change-triggered shot and repeat scheduling.
3. Reuse historical realtime artifact concepts (`max_shortlist`, `probe.horizon_steps`, beam/stay/append branch actions) as design inspiration, but do not assume they are live or validated because the winning legacy branch still ended in `mixed_failure`.
4. Keep `MeasurementCacheAudit` conceptually separate from any future measurement-data cache; its current role is accounting-only and should not be mistaken for persistent estimator reuse.

## Preventive Measures
- When evaluating future HH realtime ideas, first distinguish **ADAPT selection-time logic** from **propagation-time logic** so we do not over-credit existing machinery.
- Require every future realtime artifact/report to state whether shortlist/probe/beam and adaptive measurement are merely logged, actually executed, or disabled.
- Keep telemetry (`epsilon_proj_sq`, `stabilization_gap_sq`, `damping_gap_sq`, active-window changes) separate from policy decisions, so reports make it obvious when measurements were adaptively increased versus merely observed.

_No code changes were made during this investigation._
