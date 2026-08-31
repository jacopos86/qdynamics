# Critique: Paper-I Main Tables SPSA Profile Plan

## 1. Top 3 under-specified seams

1. **Exact visible case IDs.** The plan says the profile covers Hubbard weak/strong, spin-boson weak/strong, and four HH regimes (`docs/plans/paper-i-main-tables-spsa-profile-2026-05-31.md:39-41`), but it drops the export’s explicit eight case IDs (`prompt-exports/oracle-plan-2026-05-31-115207-spsa-profile-plan-1f-5d70.md:17-21`). Implementers may guess current helper names or accidentally inherit a broader profile.
2. **SPSA budget contract.** The plan records the user decision to use method-specific SPSA budgets (`docs/plans/paper-i-main-tables-spsa-profile-2026-05-31.md:12`) and asks for HEA budget/schedule fields (`docs/plans/paper-i-main-tables-spsa-profile-2026-05-31.md:53`), but does not name budget fields, preset values, calibration policy, or whether ADAPT should expose new schedule knobs. The export usefully warned not to add ADAPT schedule knobs unless existing APIs expose them (`prompt-exports/oracle-plan-2026-05-31-115207-spsa-profile-plan-1f-5d70.md:108-113`).
3. **HEA Qiskit Algorithms integration boundary.** The plan correctly requires `qiskit_algorithms.optimizers.SPSA` and fail-closed behavior (`docs/plans/paper-i-main-tables-spsa-profile-2026-05-31.md:33`, `docs/plans/paper-i-main-tables-spsa-profile-2026-05-31.md:49-53`), but does not specify whether to add a package/dependency check, local import wrapper, version guard, or test fixture for “package missing vs API incompatible.”

## 2. Specificity balance

- **Good correction vs export:** keep the final plan’s Qiskit Algorithms requirement for HEA. The Oracle plan suggested native HEA SPSA (`prompt-exports/oracle-plan-2026-05-31-115207-spsa-profile-plan-1f-5d70.md:41-54`), but the user handoff explicitly requires Qiskit Algorithms SPSA and fail-closed behavior; the final plan fixes that (`docs/plans/paper-i-main-tables-spsa-profile-2026-05-31.md:51-53`).
- **Over-specified tactic:** `dataclasses.replace` is named as an implementation mechanism for ADAPT overrides (`docs/plans/paper-i-main-tables-spsa-profile-2026-05-31.md:77`). The implementation agent should own that choice; the requirement is an immutable/effective config with defaults preserved.
- **Dropped useful framing:** the export proposed a shared constants module for profile ID, target, visible algorithm IDs, SNAKE ID, smoke IDs, and TSV defaults (`prompt-exports/oracle-plan-2026-05-31-115207-spsa-profile-plan-1f-5d70.md:21`, `prompt-exports/oracle-plan-2026-05-31-115207-spsa-profile-plan-1f-5d70.md:28-30`). The plan only says “possibly” a constants module (`docs/plans/paper-i-main-tables-spsa-profile-2026-05-31.md:43`), leaving duplication risk across generator/runner/tests.

## 3. Contradictions or missing dependencies

- “Open Questions — None blocking” (`docs/plans/paper-i-main-tables-spsa-profile-2026-05-31.md:191-193`) is too strong: Qiskit Algorithms availability, exact SPSA budget presets, and shared-constants ownership are unresolved enough to affect ordering.
- Item 11’s smokes include HEA Qiskit-SPSA (`docs/plans/paper-i-main-tables-spsa-profile-2026-05-31.md:157-163`) but should depend on the HEA dependency/import gate from Item 2, not merely the code path.
- Item 6’s record generation (`docs/plans/paper-i-main-tables-spsa-profile-2026-05-31.md:97-107`) depends practically on resolving the exact eight case IDs and budget-field schema, not just Items 1 and 5.

## 4. Risk of over-planning

- Items 8 and 9 can likely be one SNAKE/Phase3 validation item; both are profile-record validation for fixed SPSA Route-A (`docs/plans/paper-i-main-tables-spsa-profile-2026-05-31.md:121-143`).
- Item 12 mostly duplicates acceptance criteria and earlier fail-closed requirements (`docs/plans/paper-i-main-tables-spsa-profile-2026-05-31.md:169-189`); it could be a short guardrail checklist rather than a work item.
- The long Background is useful for handoff, but implementation sequencing should start at profile constants/cases, then optimizer seams, then record/env propagation.

## 5. Questions that would change implementation order

1. Is `qiskit-algorithms` already a required/available dependency for test and CHTC environments? If not, add the HEA dependency/fail-closed shim before threading the full profile.
2. What are the canonical method-specific SPSA budgets/presets, or should implementation add schema-only fields first and leave values to a later run-settings decision?
3. Should `paper_i_main_tables_spsa_v1` constants live in a new shared module or inside `table_i_canonical_cases.py`? This choice should be made before generator/runner/test edits to avoid duplicate literals.
