# Investigation: Marrakesh Pareto Path and Multi-Job Runtime Audit

## Summary
The fixed-theta Marrakesh run used multiple Runtime jobs because it went through the broad full-circuit audit surface, which explicitly evaluates six observables rather than energy alone. More importantly, the current trustworthy executable Marrakesh candidate set is narrower than the FakeNighthawk report suggests: the 5-op family is not a faithful executable replay surface in the current repo import path, so the only physics-credible imported circuit I can honestly recommend today is `artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json`.

## Symptoms
- Fixed-theta Marrakesh validation used multiple Runtime jobs instead of a single energy-only job.
- The user only cares about `|ΔE|` on the real QPU and does not want auxiliary observables in this path.
- The FakeNighthawk report recommends a 5-op hardware artifact, but Marrakesh/Heron-r2 is the real target backend.
- The fixed-theta `gate_pruned_7term` live Marrakesh result was poor.
- The 5-op imported artifacts looked attractive on saved ideal energy, but their executable replay status was unclear.

## Investigation Log

### Phase 1 - Runtime fan-out cause
**Hypothesis:** Multiple jobs were caused by the imported full-circuit audit measuring extra observables beyond energy.
**Findings:** Confirmed. `_run_static_observable_audit_core()` explicitly builds an `obs_map` with six observables: `energy_static`, `energy_total`, `n_up_site0`, `n_dn_site0`, `doublon`, and `staggered`. In runtime mode, each `ExpectationOracle.evaluate()` call routes to `_evaluate_runtime()`, which calls `_run_estimator_job()` once per observable per repeat. With `oracle_repeats=1`, that means six backend-mode Runtime submissions. The successful fixed-theta artifact records six IBM job IDs.
**Evidence:**
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py:1855-1944`
- `pipelines/exact_bench/noise_oracle_runtime.py:806-838`
- `pipelines/exact_bench/noise_oracle_runtime.py:1836-1883`
- `pipelines/exact_bench/noise_oracle_runtime.py:2013-2027`
- `artifacts/json/hh_gatepruned7_fixedtheta_runtime_ibm_marrakesh_20260323T145219Z_reconstructed.json:1-94`
**Conclusion:** Confirmed.

### Phase 2 - Energy-only surface existence
**Hypothesis:** The repo already has a narrow `|ΔE|`-only path and the multi-job behavior came from choosing the wrong surface, not from a fundamental Runtime requirement.
**Findings:** Confirmed. `_evaluate_locked_imported_circuit_energy()` evaluates only the Hamiltonian `qop` once noisy and once ideal. It does not request `n_up`, `n_dn`, `doublon`, or `staggered`. Separately, the imported staged wrapper dispatches multiple imported-audit routes, which can do more work than the user intended if we invoke the broad wrapper.
**Evidence:**
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py:468-533`
- `pipelines/hardcoded/hh_staged_noise_workflow.py:1172-1281`
**Conclusion:** Confirmed.

### Phase 3 - Marrakesh-conditioned compile ranking
**Hypothesis:** The FakeNighthawk hardware recommendation does not transfer directly to Marrakesh; compile burden on the real target must be checked separately.
**Findings:** Confirmed. On real `ibm_marrakesh` target compilation (optimization level 1, transpiler seeds 0-9), the cost ranking is stable:
- `circuit_optimized_7term`: cheapest compile (`22` 2Q, depth `52`, size `107`) but physically poor (`saved_abs_delta ≈ 0.338`)
- `gate_pruned_7term`: next cheapest and stable (`25` 2Q, depth `63-75`, size `110-120`)
- `aggressive_5op` and `readapt_5op`: much heavier and effectively identical on Marrakesh (`37-40` 2Q, depth `110-135`, size `161-189`)
This means the 5-op family is not Marrakesh-Pareto on hardware burden.
**Evidence:**
- `artifacts/json/investigation_marrakesh_compile_compare_20260323.json:1-340`
- `artifacts/reports/hh_prune_nighthawk_pareto_menu_20260322.md:195`
**Conclusion:** Confirmed.

### Phase 4 - Executable replay fidelity audit
**Hypothesis:** The 5-op exports may not be trustworthy executable circuit imports, even if their saved logical energies are good.
**Findings:** Confirmed. The rebuilt fixed-scaffold 7-term circuits match their saved `initial_state` almost perfectly (fidelity ~1.0), but both 5-op artifacts only reach fidelity ~`0.5717` when rebuilt through the current import/replay seam. That mismatch appears in energy too: the `readapt_5op` energy-only Marrakesh replay produced `ideal_mean = 0.6007677586325756`, even though the saved artifact energy is `0.15872408236129765`. That means the replay was already wrong before hardware noise, so it is not a valid live-QPU point for the saved 5-op artifact.

Only the two 7-term variants are exported as fixed-scaffold executable artifacts in the repo, and the fixed-scaffold export bundle contains only those two variants. The compile/reconstruction boundary code itself also describes this path as a transpilation-oriented boundary conversion, not a universal guarantee that every logical ADAPT artifact can be replayed as an equivalent executable circuit.
**Evidence:**
- `artifacts/json/investigation_import_replay_fidelity_20260323.json:1-38`
- `artifacts/json/hh_readapt5_energyonly_runtime_ibm_marrakesh_20260323T153244Z.json:1-94`
- `artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json:155-229`
- `artifacts/json/hh_prune_nighthawk_fixed_scaffold_export_20260322T232852Z.json:1-21`
- `pipelines/hardcoded/adapt_circuit_execution.py:1-6`
- `pipelines/hardcoded/adapt_circuit_execution.py:69-91`
- `pipelines/hardcoded/adapt_circuit_cost.py:317-339`
**Conclusion:** Confirmed.

### Phase 5 - Marrakesh Pareto conclusion
**Hypothesis:** There is a clean current answer to “what is the minimal circuit / Pareto-optimal circuit I can do on Marrakesh?”
**Findings:** Partially confirmed. There is a clean answer for the **current trustworthy executable import set**, but not yet for the **true Marrakesh optimum**. Today:
- Executable imported set: `gate_pruned_7term` and `circuit_optimized_7term`
- Physics-credible imported set: `gate_pruned_7term` only
- 5-op family: keep in logical design discussion and compile discussion, but exclude from current real-QPU executable Pareto claims until it has a faithful executable export
The fixed-theta Marrakesh result for `gate_pruned_7term` is still poor, so the true Marrakesh Pareto optimum remains unresolved.
**Evidence:**
- `artifacts/json/hh_gatepruned7_fixedtheta_runtime_ibm_marrakesh_20260323T145219Z_reconstructed.json:1-94`
- `artifacts/json/investigation_marrakesh_compile_compare_20260323.json:1-340`
- `artifacts/json/investigation_import_replay_fidelity_20260323.json:1-38`
**Conclusion:** Current minimal executable candidate identified; true Marrakesh optimum still unresolved.

## Root Cause
There were two separate issues:

1. **Why multiple Runtime jobs happened:** the run went through the broad full-circuit imported audit surface, which explicitly evaluates six observables. That is why multiple backend-mode Runtime jobs were created. This was not required for an energy-only `|ΔE|` check.

2. **Why the Marrakesh Pareto answer diverged from the FakeNighthawk report:** the report mixes logical Nighthawk-derived artifacts and fixed-scaffold executable descendants. On Marrakesh, the compile ranking changes materially, and the 5-op logical artifacts are not faithfully reconstructable through the current executable import seam. So they cannot honestly be used as real-QPU executable Pareto points today.

## Recommendations
1. For real-QPU `|ΔE|` checks, use the narrow energy-only helper path (`_evaluate_locked_imported_circuit_energy()`) rather than the full-circuit imported audit surface.
2. Treat `artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json` as the **minimal existing executable Marrakesh candidate** today.
3. Do **not** treat `artifacts/json/hh_prune_nighthawk_aggressive_5op.json` or `artifacts/json/hh_prune_nighthawk_readapt_5op.json` as executable Marrakesh Pareto points until they are exported as faithful fixed-scaffold descendants and pass a replay-fidelity check.
4. If the goal is the true Marrakesh Pareto frontier rather than the current executable frontier, generate Marrakesh-conditioned fixed-scaffold descendants first, then rank them with energy-only runtime evaluations.

## Preventive Measures
- Add an automatic replay-fidelity gate before any imported-artifact Runtime submission.
- Keep reports separate for:
  - logical/design Pareto artifacts
  - executable fixed-scaffold artifacts
- Default to the energy-only runtime surface when the user only asks about `|ΔE|`.
