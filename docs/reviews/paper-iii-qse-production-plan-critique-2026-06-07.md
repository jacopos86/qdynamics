# Paper III QSE production plan critique

Scope: critique only of `docs/plans/paper-iii-qse-production-2026-06-07.md` against the original context-builder export. No implementation, artifact, manuscript, promotion, or plan rewrite is proposed here.

## 1. Top 3 under-specified seams

1. **HH `full_meta` anchor resolution.** Item 1 requires resolved pool/source/settings/state provenance and fail-closed behavior, but the canonical Paper III HH source artifact is still only an open question (`docs/plans/paper-iii-qse-production-2026-06-07.md:81-83`, `:307-309`). An implementer would have to guess the resolver inputs, accepted artifact classes, and manifest schema.
2. **Response/observable schema conventions.** Items 3-4 name `S_{AB}(\omega)`, `C_{AB}(t)`, moments, HH `n/X/P/nX` builders, and layout reuse, but not serialization shape, complex-value convention, frequency/time-grid convention, sum-rule source, channel naming, or the exact HH layout authority (`docs/plans/paper-iii-qse-production-2026-06-07.md:112-114`, `:126-130`).
3. **Rolling report/source-map contract.** Item 8 says to consume source maps/evidence manifests and write TeX/PDF/audit outputs, but does not define the stable rolling-report path, update policy, input lock criteria, source-map role taxonomy, or whether candidate evidence appears separately from paper-facing evidence (`docs/plans/paper-iii-qse-production-2026-06-07.md:187-195`).

## 2. Specificity balance

- **Over-specified:** Items 9-10 are detailed L/XL feature designs for conductivity and Green functions before the immediate neutral-HH + rolling-PDF infrastructure is proven (`docs/plans/paper-iii-qse-production-2026-06-07.md:203-233`). These should likely be backlog milestones, not part of the first runnable-infrastructure pass.
- **Dropped useful export framing:** the export’s current-state flow, hard constraints, and migration/risk notes were compressed. Keeping those as a short implementer preface would reduce accidental refactors and preserve the “additive manifest, old diagnostics still valid” framing.

## 3. Contradictions or missing dependencies

- **Skill-edit authorization conflict:** Item 13 directly lists edits under `agent_guidance/skills/` (`docs/plans/paper-iii-qse-production-2026-06-07.md:265-277`), while `MATH/AGENTS.md` forbids skill-file edits unless the current user request explicitly asks to change a skill (`MATH/AGENTS.md:81-83`, `:102-104`). This needs an explicit authorization gate before implementation.
- **Compatibility gate arrives too late.** Item 6 and the report builder depend on fail-closed compatibility metadata, but the validator that supplies this is Item 12, ordered after source maps/reporting (`docs/plans/paper-iii-qse-production-2026-06-07.md:157-161`, `:250-257`, `:296-305`).
- **Plan hygiene defect:** the References heading contains stray tool-note text, which could confuse future agents (`docs/plans/paper-iii-qse-production-2026-06-07.md:307-320`).

## 4. Risk of over-planning

The plan risks becoming a full Paper III research roadmap rather than a runnable infrastructure milestone. Conductivity, Green functions, and excited-dynamics integration (Items 9-11) can be deferred until Items 1-8 plus the compatibility gate produce one neutral-HH, source-mapped, TeX-compiled rolling report path (`docs/plans/paper-iii-qse-production-2026-06-07.md:203-248`). Item 14 should be incremental per milestone, not a final all-items sweep.

## 5. Questions that would change implementation order

- What exact HH `full_meta` artifact/source-map/run manifest is canonical for Item 1?
- Should compatibility validation move before Optuna/reporting so production modes can fail closed from the start?
- Where should the rolling TeX/PDF report live, and is it append/update-in-place or rebuilt from locked inputs each time?
- Are Paper III skill-file edits explicitly authorized for this implementation pass, or should skill parity wait for a separate user approval?
- Are conductivity, Green functions, and excited-dynamics integration in the first milestone, or backlog after neutral response + rolling PDF works?
