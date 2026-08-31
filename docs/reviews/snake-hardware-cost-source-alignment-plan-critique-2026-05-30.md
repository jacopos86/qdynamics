# Critique: SNAKE Hardware-Cost Source Alignment Plan

## Context / Scope
Reviewed `docs/plans/snake-hardware-cost-source-alignment-2026-05-30.md` against the original context-builder export. Scope is code only: no TeX/PDF/table/run-data/generated-artifact edits are recommended here.

## 1. Top 3 under-specified seams
1. **Layout-aware depth fallback is not implementable enough.** Item 2 says `C_hat_d=sum_nu 2 span_G(S_nu)` with a graph/layout context, otherwise a logical-ladder fallback (`docs/plans/snake-hardware-cost-source-alignment-2026-05-30.md:55`). It does not define how graph context enters `Phase1CompileCostOracle`, how fallback source markers are propagated, or whether backend and proxy depth share normalization.
2. **Shot primitive provenance is vague.** Item 3 asks for a measurement-group spec/helper, coefficient norms, `sigma_star`, and cache-new shot proxy (`docs/plans/snake-hardware-cost-source-alignment-2026-05-30.md:70-74`) but leaves implementers to guess coefficient source, merged-QWC coefficient aggregation, default `sigma_star`, and how string-key compatibility coexists with richer specs.
3. **Family rescoring insertion point is too abstract.** Item 5 says add `score_phase_candidate_records(...)` and call it after records are assembled but before shortlisting/batching (`docs/plans/snake-hardware-cost-source-alignment-2026-05-30.md:103-104`). It should identify the exact record shapes/fields to resynchronize; otherwise scoring can diverge between `CandidateFeatures`, top-level records, and Phase-II/III batch selectors.

## 2. Specificity balance
- **Over-specified:** The helper name `score_phase_candidate_records(...)` and schema name are tactical choices the implementation agent could own if the contract is precise.
- **Over-specified / scope-sensitive:** Item 7 requires a new `hardware_cost_config` output diagnostics section (`docs/plans/snake-hardware-cost-source-alignment-2026-05-30.md:136`). That is code-only, but it risks changing future run-output schema; make it conditional on existing diagnostic-summary compatibility.
- **Dropped useful export framing:** The export gave concrete `MeasurementGroupSpec(group_key, coeff_l2, term_count)` and explicit legacy-to-lambda mapping. The plan now only says aliases resolve old fields (`:90`) and specs/helpers exist (`:70-74`), leaving high-impact compatibility decisions underspecified.

## 3. Contradictions / missing dependencies
- `Open Questions: None blocking` (`:190-191`) conflicts with unresolved implementation-order questions around depth fallback, shot coefficient provenance, and alias mapping.
- Item 9 depends on Item 4 (`:160-170`), but the stale formula-string cleanup is separable and could be done earlier once tests confirm confidence remains telemetry-only.
- Item 6 backend work depends on the same record-rescoring semantics as Item 5, not just Items 1, 2, and 4 (`:114-125`).

## 4. Risk of over-planning
- Items 7-8 may be deferred until the core denominator/telemetry path is testable; otherwise CLI/Optuna plumbing can amplify churn before the scoring contract is stable.
- The long background/reference material is useful provenance, but implementers mainly need a smaller contract: primitives, normalization, denominator exclusions, insertion point, and compatibility aliases.

## 5. Questions that would change implementation order
1. Must CLI/Optuna alias compatibility ship in the same change as the core denominator, or can it follow after core scoring tests pass?
2. Is adding future run-output diagnostics acceptable, or should the first pass keep diagnostics inside candidate telemetry/tests only?
3. What exact legacy weight-to-lambda mapping should be canonical?
4. What graph/layout source, if any, should proxy `C_hat_d` use before backend compile data exists?
