# Oracle Review

## Summary

The plan is directionally right in centralizing mechanism-resolved accounting in the strict SNAKE normalizer and keeping candidate counters out of `S_alg`, but its Phase-II metric split is not yet semantically safe. Gram/novelty and Schur/second-order work should be reconstructed from phase-local candidate records, window indices, feature telemetry, and runtime settings wherever those are present in raw JSONs. A ratio allocation of coarse Phase-II operator probes by “formula units” should be avoided unless the formula units are proven to be exactly the same charge basis as the controller’s typed operator-probe count.

## Findings

### P1

1. **`pipelines/exact_bench/snake_table_i_measurement_work.py` — Phase-II metric ratio allocation is not legitimate as proposed**

   **What’s wrong:**  
   The plan proposes:

   ```py
   allocated_bin = actual_operator_probe_count * bin_units / total_known_units
   ```

   for splitting Phase-II metric work between Gram/novelty and Schur/second-order bins. This is only valid if those “formula units” are already in the exact same dimensional charge basis as `actual_operator_probe_count`, with no sharing, caching, overlap, or phase-local guard behavior. Otherwise it creates a plausible but artificial split and can hide real mismatches.

   **Suggestion:**  
   Reconstruct Phase-II sub-bins by exact event/formula rules from candidate records and runtime settings:

   - Use `phase2_geometry_window_indices`, `schur_window_indices`, `schur_window_solve`, curvature/Hessian flags, and the actual scored/retained record lists.
   - Compute exact per-mechanism operator-probe counts using the same formula semantics as the scoring path.
   - Reconcile:

     ```text
     reconstructed_gram_novelty
     + reconstructed_schur_second_order
     + reconstructed_shared_or_unclassified
     == phase2 actual_operator_probe_count
     ```

   - If reconciliation fails, do **not** ratio-scale. Mark the excess/unknown as `metric_unclassified` or `invalid_reconciliation`.

2. **`pipelines/static_adapt/adapt_pipeline.py` / `pipelines/exact_bench/snake_table_i_measurement_work.py` — plan overstates need for new instrumentation**

   **What’s wrong:**  
   The proposed `mechanism_work_trace` is useful for future runs, but the plan makes it sound like full mechanism attribution may require reruns. For Paper-I HH SNAKE, much of the mechanism split should be reconstructable from existing raw result JSONs if the phase-local candidate records and feature fields are present.

   **Suggestion:**  
   Make existing raw JSON reconstruction the primary path:

   - First reconstruct from history rows, controller work ledgers, scored/shortlisted/retained records, and `CandidateFeatures` fields.
   - Add `mechanism_work_trace` only as a future validation/provenance aid.
   - Treat instrumentation as required only for genuinely missing labels, hidden cache/reuse behavior, or Hamiltonian sub-evaluation categories that were not emitted.

3. **`pipelines/exact_bench/snake_table_i_measurement_work.py` — Hamiltonian sub-bins should not be inferred beyond explicit nfev fields**

   **What’s wrong:**  
   The plan lists detailed H outer bins such as batch-order finite-step proxy, prune recoverability trial, and Schur prune warm-start guard. These are only exact if corresponding `nfev_*` fields exist in the raw payload. Otherwise assigning work to those categories risks the same problem as candidate-count promotion.

   **Suggestion:**  
   Keep this hierarchy:

   - Exact total:
     - `N_H_refit_eval`
     - `N_H_outer_eval`
   - Exact sub-bins only when explicit fields exist:
     - `initial_energy_nfev`
     - `nfev_seed_probe`
     - `nfev_schur_warm_start_guard`
     - `nfev_opt`
   - Everything else remains:

     ```text
     H_outer.unclassified_outer
     ```

   Do not infer H-eval mechanism bins from candidate counts, shortlist counts, or prune ladder sizes.

4. **`pipelines/exact_bench/snake_table_i_measurement_work.py` — `by_phase` fallback is too weak for mechanism classification**

   **What’s wrong:**  
   Falling back from event rows or `by_scope` to `by_phase` can reconcile coarse gradient/metric totals, but it usually loses event identity. That makes it unsafe to classify Phase-II work into Gram/novelty versus Schur/second-order, or Phase-III versus batch scoring.

   **Suggestion:**  
   Use fallback levels explicitly:

   - `events` or parseable `by_scope`: mechanism classification allowed.
   - phase-local candidate records with runtime settings: formula reconstruction allowed.
   - `by_phase` only: coarse reconciliation allowed, mechanism bins should be `unclassified`.

5. **`pipelines/static_adapt/selector_measurement_proxy.py` — exposure counters must remain diagnostic, even when used as formula operands**

   **What’s wrong:**  
   The plan says candidate counts are diagnostic-only, which is correct, but the proposed formula-unit examples risk blurring the line. For example, `sum(1 + len(window_indices))` may be a formula operand, but it is not automatically `S_alg`.

   **Suggestion:**  
   Distinguish three layers in the output:

   ```text
   candidate_exposure: diagnostic counts only
   formula_operands: reconstructed record/window dimensions
   measurement_work: chargeable typed operator-probe/H-eval bins
   ```

   Candidate counts and window sizes may feed an exact phase formula, but only the resulting validated operator-probe count belongs in `S_alg`.

### P2

1. **`pipelines/exact_bench/snake_table_i_measurement_work.py` — add per-bin provenance/status**

   **Suggestion:**  
   Each mechanism bin should say whether it is:

   - `ledger_exact`
   - `formula_reconstructed`
   - `coarse_unclassified`
   - `requires_instrumentation`
   - `invalid_reconciliation`

   Also include `operator_probe_charge_basis`, especially since current ledgers use logical estimator requests pre-grouping.

2. **`pipelines/scaffold/hh_continuation_scoring.py` — avoid duplicated formula semantics**

   **Suggestion:**  
   If feasible, extract small pure counting helpers from the scoring logic, or mirror them in tests with explicit expected values. The reconstructor should not drift from the scoring formulas.

3. **`pipelines/reporting/build_paper_i_hh_weak_weak_snake_mechanism_ablation_pdf.py` — defer reporting changes**

   **Suggestion:**  
   Do not touch PDF/report generation yet. First stabilize the reconstructor and tests against raw JSON-style synthetic payloads.

## Revised Minimal Path

1. Keep current coarse `snake_algorithmic_work_from_payload()` authoritative.
2. Add a mechanism reconstructor that consumes the same scoped history rows.
3. For gradient bins, use exact typed controller events.
4. For Phase-II/III metric bins, reconstruct exact formula counts from candidate records/window fields/runtime settings.
5. Reconcile reconstructed metric sub-bins against existing `S_alg_N_metric_probe`.
6. Put unresolved work into explicit unclassified bins; never ratio-allocate.
7. Add tests for:
   - candidate counters not becoming `S_alg`
   - exact Phase-II Gram/Schur reconstruction
   - missing feature fields producing unclassified/partial status
   - beam winner-lineage scope
   - display-prefix scope
   - aggregate `outer_nfev` not double-counted
   - by-phase-only payloads staying coarse/unclassified