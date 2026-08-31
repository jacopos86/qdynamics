# Critique: SNAKE Plateau Acquisition Plan

## Context / Scope

Reviewed only `docs/plans/snake-plateau-acquisition-2026-06-08.md` and the original context-builder export at `prompt-exports/oracle-plan-2026-06-07-213254-plateau-acquisition-b9aa.md`. No broad codebase exploration was performed.

## 1. Top 3 under-specified seams

1. **Plateau-entry ownership boundary.** The plan says to reuse the drop gate and enter before residual/terminal handling (`docs/plans/snake-plateau-acquisition-2026-06-08.md:71`, `:236-237`), but does not say whether the implementation belongs in `adapt_pipeline.py` only or in the stage controller. The export explicitly names `hh_continuation_stage_control.py` as owner of `drop_plateau_hits`, `should_probe_positions`, `resolve_stage_transition`, and `record_admission` (`prompt-exports/oracle-plan-2026-06-07-213254-plateau-acquisition-b9aa.md:26`, `:58`).
2. **Candidate universe after bypassing gain filters.** The plan requires Phase-0/gain filters not to eliminate plateau candidates and allows same generator/different position (`docs/plans/snake-plateau-acquisition-2026-06-08.md:239`, `:241`), but it does not pin where to bypass `available_indices`/repeat filtering. The export’s useful framing—use the Route-A pool/core-stage universe rather than only `available_indices`—was dropped (`prompt-exports/oracle-plan-2026-06-07-213254-plateau-acquisition-b9aa.md:612`).
3. **Active+dormant index-space contract.** The plan identifies the missing active-plus-dormant tangent span (`docs/plans/snake-plateau-acquisition-2026-06-08.md:46`) and says the same context feeds novelty, geometry, and optimizer indices (`:111-113`), but it leaves the old-pre/logical/runtime conversion boundary between `plateau_acquisition.py`, `nested_windows.py`, and `adapt_pipeline.py` for the implementer to infer.

## 2. Specificity balance

- **Possibly over-specified:** Mandating a new helper module to own state normalization/duplicates/index remapping (`docs/plans/snake-plateau-acquisition-2026-06-08.md:87`) and a broad `CandidateFeatures` field bundle (`:215`) may constrain tactical implementation more than needed; the semantic contract matters more than exact storage placement.
- **Useful export framing dropped:** The export gave a concrete unlock-margin default (`1e-8`) and tie-break sort order (`prompt-exports/oracle-plan-2026-06-07-213254-plateau-acquisition-b9aa.md:326`, `:517`), while the plan leaves the margin open (`docs/plans/snake-plateau-acquisition-2026-06-08.md:361`). If intentional, mark these as implementer-owned.
- **Tactical choice embedded:** “One plateau record per depth in v1” (`docs/plans/snake-plateau-acquisition-2026-06-08.md:240`) is reasonable, but should be framed as first-slice scope, not core semantics.

## 3. Contradictions or missing dependencies

- The plan says “No behavior-level blocker remains” (`docs/plans/snake-plateau-acquisition-2026-06-08.md:357`), but beam fail-fast vs full parity is explicitly still open (`:359`) and changes implementation order.
- Item 8 telemetry records fields added in Item 4 (`docs/plans/snake-plateau-acquisition-2026-06-08.md:301`), but its dependencies list only Items 5–7 (`:312`); Item 4 should be included.
- Stage-controller files/tests appear in the export (`prompt-exports/oracle-plan-2026-06-07-213254-plateau-acquisition-b9aa.md:26`, `:51`, `:865`) but not the plan’s work items. If no stage-controller edits are intended, the plan should say why.

## 4. Risk of over-planning

- The background/reference block is useful but too large for the executable path; move most references to an appendix.
- Item 10 can be a post-implementation documentation cleanup, not part of the critical implementation sequence.
- Beam “production-ready” parity can be simplified to a first-slice decision: fail fast now or split full parity into a separate follow-up.

## 5. Questions whose answers change implementation order

1. Should plateau entry be implemented through `StageController` or intercepted solely in `adapt_pipeline.py`?
2. Is first slice required to support beam parity, or should it fail fast for beam width > 1?
3. Should the default unlock margin be fixed now, e.g. export’s `1e-8`, or deliberately left unset beyond finite/nonnegative validation?
4. What is the authoritative plateau candidate universe when bypassing gain/repeat filters?
