# Paper I Shot-Proxy Cost Repair Plan Critique

## Context/Scope
Reviewed only `docs/plans/paper-i-shot-proxy-cost-repair-2026-06-11.md` against the original context-builder export at `prompt-exports/oracle-plan-2026-06-11-160106-paper-i-cost-repair-17db.md`. RepoPrompt could not read the export, so it was read via shell fallback as requested. One named-seam spot-check confirmed `pipelines/reporting/build_paper_i_ablation_results_tables_latex_pdf.py` exists.

## Findings

### 1. Top under-specified seams
1. **SNAKE prefix/plateau/terminal reconstruction contract.** The plan says to compute scoped SNAKE `S_alg` and avoid double-counting (`docs/plans/paper-i-shot-proxy-cost-repair-2026-06-11.md:89-94`), but an implementer still has to guess the helper interface, phase-to-component mapping, and exact prefix slicing rules. The export had useful detail here: illustrative interface, component mapping, and terminal-summary rejection rules (`prompt-exports/oracle-plan-2026-06-11-160106-paper-i-cost-repair-17db.md:277-306`).
2. **Cross-surface audit source boundaries.** Item 5 names broad surfaces and wildcard-like inputs (`docs/plans/paper-i-shot-proxy-cost-repair-2026-06-11.md:122-128`) but not the concrete input/output/status taxonomy. The export specified audit inputs, outputs, statuses, blocker codes, and exit behavior (`prompt-exports/oracle-plan-2026-06-11-160106-paper-i-cost-repair-17db.md:333-383`).
3. **Tables I-II blocked-row/display behavior.** Item 6 correctly identifies SNAKE, external TETRIS, and retained fixed rows (`docs/plans/paper-i-shot-proxy-cost-repair-2026-06-11.md:133-138`), but omits how `Series`, row-source fields, and blocked work cells should behave. The export gave concrete fields and display policy (`prompt-exports/oracle-plan-2026-06-11-160106-paper-i-cost-repair-17db.md:387-412`).

### 2. Specificity balance
- **Over-specified:** The plan pre-locks payload/status names before implementation (`docs/plans/paper-i-shot-proxy-cost-repair-2026-06-11.md:36-59`, `:78-83`). That is useful direction, but exact constants should be reconciled with existing readers/status language rather than treated as final prose-invented schema.
- **Dropped useful framing:** The export explicitly framed this as a targeted repair because existing metric enrichment and sidecar validation already do much of the work (`prompt-exports/oracle-plan-2026-06-11-160106-paper-i-cost-repair-17db.md:201-209`). The plan implies this but should make the anti-broad-refactor constraint more visible.
- **Dropped named surface:** The export names the ablation builder and Table IV/reporting surface handling (`prompt-exports/oracle-plan-2026-06-11-160106-paper-i-cost-repair-17db.md:434-439`, `:521-526`); the plan only says “ablation/reporting tables” (`docs/plans/paper-i-shot-proxy-cost-repair-2026-06-11.md:23`, `:125`).

### 3. Contradictions or missing dependencies
- The export says the two SNAKE reconstruction surfaces should be audited/standardized before choosing one canonical path (`prompt-exports/oracle-plan-2026-06-11-160106-paper-i-cost-repair-17db.md:50-53`), but the plan orders helper/bridge work before the cross-surface audit (`docs/plans/paper-i-shot-proxy-cost-repair-2026-06-11.md:67-74`, `:122-129`). This may force implementation before inventory.
- Item 8 includes optional terminal diagnostics (`docs/plans/paper-i-shot-proxy-cost-repair-2026-06-11.md:155-160`) but does not depend on Item 9, which defines terminal/physical reporting (`:166-173`).
- Status/source-hierarchy dependencies should include `agent_guidance/skills/shared/run-results-common.md`; the export listed it as policy authority (`prompt-exports/oracle-plan-2026-06-11-160106-paper-i-cost-repair-17db.md:10-11`, `:25`).

### 4. Risk of over-planning
- Defer Item 7 unless the audit confirms the legacy support map still feeds a visible/reporting surface.
- Fold most of Item 9 into audit/report serialization unless the user has already requested a concrete terminal-shot display surface.
- Avoid turning Item 5 into a broad `MATH/*`/`output/pdf/*` crawler; require source-map-driven inputs only.

### 5. Order-changing questions
1. Should a minimal inventory audit run before implementing the SNAKE helper, or is the helper first acceptable?
2. Is the first repair target Tables I-II only, or must HH/Table V diagnostics land in the same implementation pass?
3. Are terminal-shot quantities required only in JSON/audit outputs, or also in a PDF/report appendix?
4. Is `build_paper_i_three_model_support_maps.py` still active for current visible/reporting surfaces, or can Item 7 be deferred?
