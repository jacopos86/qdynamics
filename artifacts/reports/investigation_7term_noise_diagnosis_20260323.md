# Investigation: 7-term FakeNighthawk noise diagnosis

## Summary
In progress. The correct gate-pruned 7-term scaffold is near-exact noiselessly, but its saved-parameter FakeNighthawk backend-scheduled energy is badly biased even with mthree. We are gathering code-grounded evidence to separate readout effects from gate/stateprep effects and to decide whether noisy continuous replay is warranted.

## Symptoms
- `artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json` is near exact in ideal form (`|ΔE| ≈ 4.23e-04`).
- Saved-parameter FakeNighthawk evaluation on the same artifact is much worse:
  - unmitigated `|ΔE| ≈ 0.17891754509302488`
  - `mthree` `|ΔE| ≈ 0.1191103021218784`
- The `circuit_optimized_7term` export is a different logical subject and cannot be used for physics conclusions.

## Investigation Log

### Context builder - architecture pass
**Hypothesis:** The large bias is more likely a backend-scheduled execution-path issue than a broken fixed-scaffold import path.
**Findings:** The builder selected the fixed-scaffold export artifacts, imported-artifact evaluation code, backend-scheduled oracle code, and staged wrapper defaults. It highlighted that `backend_scheduled` hardcodes transpile optimization level 1, while report compile tables/circuit-optimized compile claims use optimization level 2, and that fixed-scaffold workflow defaults point to the wrong `circuit_optimized` subject.
**Evidence:** `pipelines/exact_bench/noise_oracle_runtime.py`, `pipelines/exact_bench/hh_noise_robustness_seq_report.py`, `pipelines/hardcoded/hh_staged_noise_workflow.py`, `artifacts/json/hh_prune_nighthawk_fixed_scaffold_export_20260322T232852Z.json`.
**Conclusion:** Confirmed as a strong working hypothesis. Need direct compile and attribution evidence on the correct gate-pruned subject.

### Code verification - measurement semantics mismatch
**Hypothesis:** The report’s compile-cost / shot-budget story may not match the execution semantics of the saved-parameter noisy evaluator.
**Findings:** The report presents QWC-group measurement cost, but the `backend_scheduled` evaluator loops over `observable.to_list()` term-by-term, copies the already compiled base circuit, appends basis rotations and measurements for each Pauli term, and runs that separately. This means the noisy saved-parameter route repeatedly incurs the full stateprep circuit for each Hamiltonian term rather than using the report’s grouped picture directly.
**Evidence:** `artifacts/reports/hh_prune_nighthawk_pareto_menu_20260322.md` (QWC groups = 4); `pipelines/exact_bench/noise_oracle_runtime.py:1531-1610`.
**Conclusion:** Confirmed as an apples-to-oranges trap. It does not by itself prove the large bias, but it means the report’s grouped hardware picture should not be read as the literal execution schedule of the noisy audit.

### Evidence gathering - Hamiltonian measurement burden
**Hypothesis:** A small 7-term scaffold can still incur a larger repeated measurement burden because the evaluator measures the full HH Hamiltonian, not the ansatz term set.
**Findings:** The resolved fixed-scaffold context builds a static HH observable with 17 Pauli terms (`1` identity, `6` weight-1, `10` weight-2). Combined with the term-by-term `backend_scheduled` evaluator, each attribution slice repeats the full stateprep across these Hamiltonian terms rather than across 4 grouped measurement batches.
**Evidence:** direct runtime inspection via `_resolve_locked_imported_fixed_scaffold_context(...)` and `build_time_dependent_sparse_qop(...)` on `artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json`; result: `hamiltonian_terms=17`, `weight_hist={0:1,1:6,2:10}`.
**Conclusion:** Confirmed. “7 runtime terms” does not imply a tiny measurement execution surface under the current noisy evaluator.

### Code verification - backend_scheduled and fixed-scaffold routes
**Hypothesis:** The current noisy evaluation path uses a distinct compile/noise contract that could explain the mismatch between report compile tables and saved-parameter noisy results.
**Findings:** `backend_scheduled` requires fake backends, only supports mitigation `none` or `readout`, forces transpile optimization level 1, and disables active symmetry mitigation in this path. Fixed-scaffold replay changes only continuous parameters, while fixed-scaffold attribution keeps parameters fixed and decomposes one shared compiled circuit into `readout_only`, `gate_stateprep_only`, and `full` slices.
**Evidence:** `pipelines/exact_bench/noise_oracle_runtime.py:1190-1284, 1430-1469`; `pipelines/exact_bench/hh_noise_robustness_seq_report.py:2679-3158`.
**Conclusion:** Confirmed. Saved-parameter noisy numbers should not be compared directly to report compile tables without accounting for the opt-level and route differences.

### Evidence gathering - compile burden on the correct subject
**Hypothesis:** The large noisy bias may come from the `backend_scheduled` path compiling the correct gate-pruned artifact into a much worse physical circuit than the report implies.
**Findings:** Direct compile-scout runs on `artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json` for `FakeNighthawk` with seed 7 gave:
- optimization level 1: `2Q=19`, `depth=63`, `size=93`
- optimization level 2: `2Q=19`, `depth=56`, `size=99`
This is somewhat worse than the report’s grouped/historical cost presentation, but not remotely large enough to explain the full noisy energy shift by routing alone.
**Evidence:** direct `adapt_circuit_cost.py` runs on the current gate-pruned artifact.
**Conclusion:** Eliminates the simplest “routing exploded under backend_scheduled” explanation.

### Code verification - subject default trap
**Hypothesis:** The staged workflow defaults can silently steer fixed-scaffold routes to the wrong physical subject.
**Findings:** Fixed-scaffold routes default to `hh_nighthawk_circuit_optimized_7term_v1` and `FakeNighthawk` when no explicit import JSON is supplied.
**Evidence:** `pipelines/hardcoded/hh_staged_noise_workflow.py:186-197, 223-231, 328-360`.
**Conclusion:** Confirmed. All further diagnosis must explicitly pass `artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json`.

### Evidence gathering - direct noise attribution on the correct subject
**Hypothesis:** The large bias may be mostly gate/stateprep noise rather than mostly readout noise.
**Findings:** The explicit fixed-scaffold attribution run on `artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json` finished successfully.
- workflow: `artifacts/json/hh_gate_pruned_fixed_scaffold_attr_20260323T013814Z.json`
- shared compile: FakeNighthawk, transpile opt-level 1, seed 7, layout `[47,58,38,48,37,57]`
- ideal mean: `0.15909070323441615`
- `readout_only`: noisy mean `0.2076416015625`, delta `+0.04855089832808385 ± 0.010513345210922244`
- `gate_stateprep_only`: noisy mean `0.2783050537109375`, delta `+0.11921435047652135 ± 0.009210342815004496`
- `full`: noisy mean `0.33758544921875`, delta `+0.17849474598433385 ± 0.011407697664128816`
- component additivity residual: `+0.010729497179728653`
Comparing against the separate saved-parameter `mthree` run (`artifacts/json/hh_gate_pruned_savedparam_eval_20260323T011240Z.json`), the mitigated full-noise delta `+0.11868750301318737` almost exactly matches the attribution `gate_stateprep_only` delta.
**Evidence:** `artifacts/json/hh_gate_pruned_fixed_scaffold_attr_20260323T013814Z.json`; `artifacts/json/hh_gate_pruned_savedparam_eval_20260323T011240Z.json`.
**Conclusion:** Confirmed. Readout is a meaningful but minority contributor; `mthree` removes roughly the readout slice, and the remaining floor is dominated by gate/stateprep noise.

## Root Cause
The correct gate-pruned 7-term scaffold is **not** failing because of a broken import path or because its logical variational structure is poor. The evidence instead points to an **execution-contract limitation on FakeNighthawk under the repo’s `backend_scheduled` path**:
- the artifact is near exact ideally,
- the imported evaluation reproduces that ideal energy faithfully,
- the compiled circuit remains modest (`19` 2Q gates, depth `63` at the actual opt-level-1 noisy path),
- but the noisy evaluator repeatedly executes the compiled stateprep across the full 17-term HH Hamiltonian in a term-by-term measurement contract,
- and the attribution split shows the dominant residual is gate/stateprep noise, not readout.

## Recommendations
1. **Next experiment: fixed-scaffold continuous noisy replay on the explicit good gate-pruned artifact.**
   - Goal: test whether the saved noiseless-optimal angles are simply noise-misaligned under the same FakeNighthawk `backend_scheduled` contract.
   - Keep structure fixed; move only continuous parameters.
2. **Do not jump to operator-level reoptimization under noise yet.**
   - The current evidence does not indict the operator set.
   - Structural changes would confound operator choice with the already-dominant execution-path noise.
3. **Treat the current diagnosis as execution-contract limited on FakeNighthawk unless noisy replay materially improves the energy.**
   - If replay barely beats the present `mthree` floor (~`|ΔE| ≈ 0.119`), then the scaffold is best understood as hardware/execution limited on this backend model.
4. **If structural revisiting becomes necessary later, use the viable multi-term 5-op family rather than the known-negative single-term gate-pruned ADAPT pool.**

## Preventive Measures
- Always override fixed-scaffold routes with the explicit artifact path when diagnosing physics questions; the workflow default points to `hh_nighthawk_circuit_optimized_7term_v1`, which is a different logical subject.
- Keep compile-only/report comparisons separate from noisy execution claims; the report’s grouped cost picture is not the same contract as the term-by-term `backend_scheduled` energy estimator.
- Preserve logs for every long run so route-level timeouts or worker failures cannot erase the diagnostic trail.
