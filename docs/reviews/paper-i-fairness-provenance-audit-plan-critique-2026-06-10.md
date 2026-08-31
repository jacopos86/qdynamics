# Paper I Fairness/Provenance Audit Plan Critique

## Context / Scope
Reviewed `docs/plans/paper-i-fairness-provenance-audit-2026-06-10.md` against the original context-builder export `prompt-exports/oracle-plan-2026-06-10-132152-paper-i-audit-d2dac6-0f42.md`. This critique is limited to implementability seams, plan/export deltas, contradictions/dependencies, over-planning risk, and ordering questions.

## Findings

### 1. Top 3 under-specified seams
1. **PDF-vs-TeX authority remains ambiguous.** The plan says the audit targets the condensed PDF but reads `static_adapt_paper_I_condensed.tex` as the source proxy (`docs/plans/paper-i-fairness-provenance-audit-2026-06-10.md:3-4`, `docs/plans/paper-i-fairness-provenance-audit-2026-06-10.md:19-20`). It also includes `pdf_source_sync_unknown` (`docs/plans/paper-i-fairness-provenance-audit-2026-06-10.md:38-39`) but never specifies whether to check PDF metadata/hash, parse the PDF, or treat TeX as sufficient.
2. **Source-hash expectations lack a precedence rule.** Item 3 asks for expected/actual SHA256 and `active_or_legacy` (`docs/plans/paper-i-fairness-provenance-audit-2026-06-10.md:81-83`), but does not say where expected hashes come from when comments, source maps, and support JSON disagree or omit hashes. Implementers may guess between “missing hash is okay,” “blocked,” or “not_checked.”
3. **Work-proxy comparability is not operationalized.** Item 6 asks for `S_alg`/legacy/proxy classification and “comparability status” (`docs/plans/paper-i-fairness-provenance-audit-2026-06-10.md:102-104`), while Item 7 adds optimizer/settings/pool fairness (`docs/plans/paper-i-fairness-provenance-audit-2026-06-10.md:109-111`). The plan lacks a decision table for mixed evidence, e.g. both `S_alg` and `Snorm`, partial components, or visible-prefix custom work. The export’s explicit “fixed-structure energy-eval proxy” case was dropped (`prompt-exports/oracle-plan-2026-06-10-132152-paper-i-audit-d2dac6-0f42.md:379`).

### 2. Specificity balance
- **Over-specified:** Fixed dated output filenames and seven companion CSVs (`docs/plans/paper-i-fairness-provenance-audit-2026-06-10.md:26-35`) plus exact orchestration split/final writer rule (`docs/plans/paper-i-fairness-provenance-audit-2026-06-10.md:173-177`) are tactical. The implementation agent can own file decomposition once the JSON/report contract is fixed.
- **Useful export framing dropped:** The export included an illustrative CLI and exit-code behavior (`prompt-exports/oracle-plan-2026-06-10-132152-paper-i-audit-d2dac6-0f42.md:217-231`), which would reduce ambiguity more than the detailed CSV inventory. It also named a test guard against `safe_for_manuscript_transfer`/promotion flags (`prompt-exports/oracle-plan-2026-06-10-132152-paper-i-audit-d2dac6-0f42.md:425`), while the plan only says “absence of promotion/table-transfer flags” (`docs/plans/paper-i-fairness-provenance-audit-2026-06-10.md:123-125`).

### 3. Contradictions or missing dependencies
- The plan’s in-scope boundary includes reading local “TeX/PDF metadata” (`docs/plans/paper-i-fairness-provenance-audit-2026-06-10.md:156-160`), but the Approach says to use TeX as the source proxy (`docs/plans/paper-i-fairness-provenance-audit-2026-06-10.md:19-20`) and no work item owns PDF sync/metadata checking.
- Item 10 requires an exact local audit command (`docs/plans/paper-i-fairness-provenance-audit-2026-06-10.md:130-132`), but the plan never preserves the export’s CLI shape or exit semantics. That makes the handoff dependent on a later design decision.
- `run-results-common.md` appears in the export’s policy-contract list but is absent from the plan’s key-file/dependency surfaces; if implementation follows policy gates strictly, it may need that file before schema/status wording is finalized.

### 4. Risk of over-planning
The plan is close to an implementation spec rather than an executable plan. Consider cutting or compressing: the long artifact list, Item 10 handoff details, and the orchestration split. Items 5-7 could be one “classification matrices” phase with subchecks, reducing false sequencing between cost, work, replay, and fairness logic.

### 5. Questions that would change implementation order
1. Is TeX-only auditing acceptable, or must the first pass establish PDF/TeX sync before parsing tables?
2. Should missing expected hashes be `not_checked`, `qualified`, or `blocked`?
3. Are output filenames/schema frozen, or may the implementation choose parameterized/date-stamped paths?
4. Should legacy/proxy work values be classified before fairness matrices, or should all mixed-proxy rows be blocked early?
