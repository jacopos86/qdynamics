# HH SNAKE Noise Route Faithfulness Plan Critique

**Scope:** Critique of `docs/plans/hh-snake-noise-route-faithfulness-2026-06-05.md` against the context-builder export `prompt-exports/oracle-plan-2026-06-05-194434-hh-noise-plan-2865ed-28a3.md`. No code implementation or broad exploration performed.

## Top 3 under-specified seams

1. **Original-route baseline comparison is not executable enough.** Item 4 says to compare “route-critical fields” and preserve rollback/batching/pruning/beam/optimizer/target/route settings, but does not define the source artifact, canonical field list, or tolerated drift (`docs/plans/...:74-79`). This matters because the plan’s target is the local depth-13 replay, while Table III trial0011 remains provenance context (`docs/plans/...:7-10`).
2. **Zero-noise “noise-capable” mode is ambiguous.** Item 3 says to request noise-capable oracle/inner-objective settings with zero noise (`docs/plans/...:63-68`), but does not specify whether candidate gradients, selected-energy evaluations, or both should route through oracle surfaces. The export was more explicit that the seam is `phase3_oracle_gradient_mode` plus inner objective settings (`prompt-exports/...:142-145`).
3. **Rung pass-gate ownership is unclear.** Item 11 blocks combined-noise rows unless earlier rungs are “explicitly marked passed or user-selected CLI options request them” (`docs/plans/...:151-158`), but the plan does not say where that pass state lives: generator manifest, validation report, CLI flag, or external operator decision.

## Specificity balance

- **Over-specified tactical names:** Item 2 proposes exact metadata names such as `noise_ladder_rung`, `oracle_surface_class`, `target_hit_required`, and `diagnostic_non_hit_allowed` (`docs/plans/...:52-57`). Useful semantics, but naming/storage should probably be owned by the implementation agent once command/manifest constraints are inspected.
- **Dropped useful export framing:** The export’s Item 4 included `pipelines/static_adapt/route_identity.py` as a key file (`prompt-exports/...:149-153`); the checked-in plan omits it (`docs/plans/...:79`). If route identity fields exist there, dropping it weakens the “original-route knobs” contract.
- **Useful framing retained:** The plan keeps the export’s major guardrail that existing hooks mostly exist and this is mainly ordering/schema/validation, not a new noiseless pipeline (`prompt-exports/...:12-17`; `docs/plans/...:32-37`).

## Contradictions or missing dependencies

- **Shot-noise formula conflict:** Background says shot-equivalent value noise is `sigma0_abs/sqrt(N_eff)` (`docs/plans/...:27`), but Item 9 says `sigma0_abs/N_eff` (`docs/plans/...:129-134`). That must be resolved before implementation.
- **Ladder-order tension with Paper-I run guidance:** The run skill says same-noise target-hit first, then same-noise noise-floor stop, then confidence scoring, then escalate noise (`agent_guidance/skills/paper-i-run/SKILL.md:426-437`). The plan delays noise-floor/confidence until after shot-only, gate-only, and combined-noise rungs (`docs/plans/...:32-33`, `162-180`). If intentional, it needs an explicit user-override note.
- **Logical dependency inversion:** Item 3 can create the zero-noise control before Item 4 preserves original-route knobs (`docs/plans/...:63-81`). It may be fine to code in that order, but no zero-noise control should be run/trusted before Item 4’s comparison gate exists.

## Risk of over-planning

Items 9-13 are likely too detailed for the first repair pass; they should probably remain future acceptance gates until Items 2-8 and Item 14 prove the zero-noise route-faithfulness failure is fixed. Items 5-7 could also be one “non-hit/current semantics” workstream rather than three medium-sized work items.

## Questions that change implementation order

1. Is the authoritative knob baseline the local depth-13 replay, the Table III trial0011 source, or a merged source-lock record?
2. Should zero-noise noise-capable control split into separate checkpoint/replay-only, oracle-gradient-only, selected-energy-oracle-only, and both-oracle controls?
3. Is the Paper-I run-skill ladder order authoritative at each noise level, or did the June 5 user decision intentionally move noise-floor/confidence after combined noise?
4. Where should rung pass/fail state be persisted so generators can fail closed without editing generated CHTC inputs?
