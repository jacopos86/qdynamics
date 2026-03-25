# Investigation: Kingston next shot for best |ΔE|

## Summary
For the strict goal of minimizing the next real-QPU `|ΔE|`, the strongest evidence still favors a **7-term Kingston fixed-theta energy-only baseline** over a fresh 6-term Kingston rerun. For the different goal of learning whether the lean circuit is salvageable, the best experiment is instead a **fresh 6-term Kingston SPSA rerun**.

## Symptoms
- The 6-term fixed-theta runtime baseline was poor on Marrakesh.
- The interrupted 6-term Kingston SPSA run improved over its early evaluations but only left job-level recovery, not a faithful optimizer trace.
- Existing attribution evidence indicates the dominant residual is gate/state-prep, not readout.
- The 7-term anchor remains much better than the 6-term candidate in local exact `|ΔE|`.

## Investigation Log

### Phase 1 - Current rerun-planner logic
**Hypothesis:** The repo is currently recommending the 7-term anchor first because the evidence base is mismatched, not because the 6-term lean line is impossible.
**Findings:** `build_fixed_scaffold_rerun_plan()` recommends `select_backend_then_anchor_7term_energy_only_fixedtheta_baseline` when the anchor lacks a matched energy-only runtime baseline and the recovered backend differs from the candidate fixed-theta backend.
**Evidence:**
- `pipelines/exact_bench/fixed_scaffold_runtime_followups.py:482-505`
- `test/test_fixed_scaffold_runtime_followups.py:258-276`
**Conclusion:** Confirmed. The current planner recommendation is driven by evidence mismatch and missing anchor baseline, not by a claim that the 6-term line cannot work.

### Phase 2 - What real-runtime routes are actually supported
**Hypothesis:** A real-QPU fixed-scaffold optimizer replay is not an exposed staged route today, while a narrow real-runtime energy-only baseline is.
**Findings:** The staged workflow allows a real-runtime fixed-scaffold energy-only baseline, but explicitly rejects fixed-scaffold noisy replay unless `--use-fake-backend` is enabled. Runtime DD/ZNE are exposed as follow-on fixed-theta phases on the real-runtime energy-only route.
**Evidence:**
- `pipelines/hardcoded/hh_staged_noise_workflow.py:571-616`
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py:468-547`
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py:639-711`
- `pipelines/hardcoded/hh_staged_cli_args.py:367-410`
**Conclusion:** Confirmed. Repo-native real-QPU support today is strongest for the narrow fixed-theta energy-only baseline, with DD/ZNE as add-on probes, not optimizer rescues.

### Phase 3 - 6-term versus 7-term starting quality
**Hypothesis:** The 7-term anchor remains the better immediate minimization bet because it starts from much lower exact `|ΔE|` than the 6-term lean candidate.
**Findings:** The 7-term anchor has exact `abs_delta_e = 0.0004227991086915017`, while the 6-term candidate has exact `abs_delta_e = 0.004613056000159821` after omitting `eyezee`.
**Evidence:**
- `artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json:18-47`
- `artifacts/json/hh_marrakesh_fixed_scaffold_6term_drop_eyezee_20260323T171528Z.json:18-47`
- Structural omission recorded in the rerun plan: `artifacts/json/hh_6term_vs_7term_runtime_rerun_plan_20260323.json:147-181`
**Conclusion:** Confirmed. The 7-term line starts about 10x better locally in `|ΔE|`.

### Phase 4 - Existing real-hardware evidence on the lean line
**Hypothesis:** The 6-term line should still be treated seriously because the same-backend Kingston SPSA prefix is the most relevant direct evidence for lean-circuit viability.
**Findings:** The 6-term fixed-theta Marrakesh baseline was poor (`delta_mean = 0.28530393275267446`), but the interrupted Kingston SPSA run improved to a best recovered job energy of `0.3935622931602998`. However, the recovery is job-level only and contains no theta journal or objective-call boundaries.
**Evidence:**
- `artifacts/json/hh_gatepruned6_fixedtheta_runtime_ibm_marrakesh_20260323T171528Z.json:1-40`
- `artifacts/json/hh_kingston_6term_spsa48_runtime_partial_recovery_20260323.json:1-40`
- `artifacts/json/hh_kingston_6term_spsa48_runtime_partial_recovery_20260323.json:80-119`
- `artifacts/reports/investigation_kingston_partial_trace_and_rerun_plan_20260323.md:4-22`
**Conclusion:** Confirmed. The 6-term Kingston prefix is the best direct lean-circuit hardware evidence, but it is not strong enough by itself to displace the 7-term anchor as the safest next minimization bet.

### Phase 5 - Noise model interpretation
**Hypothesis:** Because readout is not the dominant residual, reducing gates alone is not enough; retained state-prep / gate robustness matters more for best `|ΔE|`.
**Findings:** The attribution artifact shows `full = 0.17849474598433385`, `gate_stateprep_only = 0.11921435047652135`, and `readout_only = 0.04855089832808385`. The saved-parameter M3 residual (`0.11868750301318737`) aligns with the gate/state-prep slice, not the readout slice.
**Evidence:**
- `artifacts/json/hh_gate_pruned_fixed_scaffold_attr_20260323T013814Z.json:1138-1141`
- `artifacts/json/hh_6term_vs_7term_runtime_rerun_plan_20260323.json:118-137`
**Conclusion:** Confirmed. The dominant residual floor is gate/state-prep, so the extra expressive direction in the 7-term anchor remains valuable for best-`|ΔE|` bets.

### Phase 6 - Nearby DD/ZNE options
**Hypothesis:** Runtime DD-probe or final-ZNE are follow-on probes, not the best very next shot for minimizing `|ΔE|`.
**Findings:** The real-runtime path supports `dd_probe_twirled_readout_v1` and `final_audit_zne_twirled_readout_v1` only as add-on phase evaluations on the same fixed-theta circuit. Local FakeMarrakesh ablation for the 6-term candidate shows readout helps (`delta_mean 0.1516 -> 0.0928`), but the nearby local twirling variant worsens to `0.1136`.
**Evidence:**
- `pipelines/exact_bench/noise_oracle_runtime.py:306-349`
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py:639-711`
- `artifacts/json/hh_marrakesh_6term_local_mitigation_ablation_20260324T001622Z.json:31-34`
- `artifacts/json/hh_marrakesh_6term_local_mitigation_ablation_20260324T001622Z.json:174-176`
- `artifacts/json/hh_marrakesh_6term_local_mitigation_ablation_20260324T001622Z.json:332-335`
**Conclusion:** Confirmed. DD/ZNE are sensible only after a clean baseline; if forced, DD-probe is a more natural first add-on than final-ZNE.

### Phase 7 - Local optimizer robustness
**Hypothesis:** The 7-term anchor is at least as robust as the 6-term candidate under perturbed-start SPSA, so there is no local robustness argument favoring the lean line for best `|ΔE|`.
**Findings:** From perturbed starts, the 6-term final `abs_delta_e` lands around `0.0102-0.0161`, while the 7-term lands around `0.0087-0.0124` under the same local exact SPSA(128) setup.
**Evidence:**
- `artifacts/json/hh_marrakesh_6term_local_exact_spsa128_perturbed_20260323T182700Z.json:14-211`
- `artifacts/json/hh_marrakesh_7term_local_exact_spsa128_perturbed_20260323T182846Z.json:14-212`
**Conclusion:** Confirmed. The local perturbation evidence slightly favors the 7-term anchor on robustness as well.

## Root Cause
The current tension is not a code bug; it is an evidence mismatch plus a physics tradeoff. The 6-term candidate is the lean executable circuit, but it starts from a materially worse exact `|ΔE|` than the 7-term anchor and lives in a noise regime where gate/state-prep dominates readout (`artifacts/json/hh_gate_pruned_fixed_scaffold_attr_20260323T013814Z.json:1138-1141`). That makes the extra retained direction in the 7-term anchor more valuable than pure gate-count reduction if the goal is the next lowest likely real-QPU `|ΔE|`. The interrupted Kingston 6-term SPSA trace is still the best direct evidence for lean-circuit viability, but because it is only job-level recovery (`artifacts/json/hh_kingston_6term_spsa48_runtime_partial_recovery_20260323.json:1-20`) it is not strong enough to overrule the anchor as the safer minimization bet.

## Recommendations
1. **If the goal is the lowest likely next real-QPU `|ΔE|`, run the 7-term Kingston fixed-theta energy-only baseline first.**
   - Route: fixed-scaffold runtime energy-only baseline with `--fixed-final-state-json artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json`
   - Why: best local starting point, slightly better local robustness, and directly fills the missing matched Kingston anchor datum.
2. **If the goal is instead “is the lean candidate salvageable on Kingston?”, rerun the 6-term Kingston SPSA screen as a fresh rerun, not a resume.**
   - Why: the partial Kingston trace is the best direct lean-circuit evidence, but it does not preserve a faithful best-theta restart point.
3. **Do not spend the very next shot on final-ZNE.**
   - If an add-on probe is needed after a clean baseline, prefer runtime DD-probe before final-ZNE.

## Preventive Measures
- Keep a theta journal or objective-trace sidecar for any future real-runtime optimizer-style experiment so interrupted runs are restartable.
- Separate “best likely next `|ΔE|` shot” from “best learning experiment for the lean candidate” in future run plans; they are not the same decision.
- When comparing 6-term and 7-term on hardware, keep the backend and route contract matched (same backend, same energy-only surface) before drawing Pareto conclusions.
