# Agent Navigation Spine Refresh Plan — Focused Critique

## Context / Scope

Reviewed `docs/plans/agent-navigation-spine-refresh-2026-05-30.md` against the context-builder export at `prompt-exports/oracle-plan-2026-05-30-165235-navigation-spine-pla-30c5.md`. Scope is limited to implementation-readiness seams, specificity balance, contradictions/dependencies, over-planning risk, and order-changing questions.

## 1. Top 3 under-specified seams

1. **Root router shape is not concrete enough.** Item 2 says the beginning of `AGENTS.md` should answer “what should an agent read next?” while preserving all gates (`docs/plans/agent-navigation-spine-refresh-2026-05-30.md:61-67`), but it does not say whether to add a compact top router, move existing sections, duplicate mandatory gates, or leave deep policy text in place. That choice affects risk of semantic weakening.
2. **Paper bridge load order is underspecified.** Item 3 points to `MATH/paper_facing/README.md`, “shared claim/source/style guides,” and support folders (`docs/plans/agent-navigation-spine-refresh-2026-05-30.md:73-79`), but the export named more specific bridge surfaces: `journal_math_skill_supplement.md`, `repo_to_journal_translation.md`, and `ai_manuscript_style_guardrails.md` in addition to claim-source docs (`prompt-exports/...navigation-spine-pla-30c5.md:34-38`). The plan leaves implementers guessing which shared docs should be first-class links.
3. **Authority-order reconciliation is vague.** The verification item requires root and run-guide authority orders to agree (`docs/plans/agent-navigation-spine-refresh-2026-05-30.md:130-136`), but the export explicitly flagged ambiguity between current authority order and routing-map placement of mandatory gates (`prompt-exports/...navigation-spine-pla-30c5.md:62-64`). The plan should identify the intended winner rather than leaving reconciliation to taste.

## 2. Specificity balance

- **Over-specified:** The inactive future-skill ledger is prescribed as a concrete table with candidate IDs and row text (`docs/plans/agent-navigation-spine-refresh-2026-05-30.md:109-123`). That is more tactical than needed; especially `artifact-retention-cleanup` (`:121`) expands beyond the core navigation-spine problem.
- **Useful framing dropped:** The export’s bridge framing included repo-to-journal translation, style guardrails, and the local manuscript-skill supplement (`prompt-exports/...navigation-spine-pla-30c5.md:34-38`). The plan mostly collapses this to `shared/*.md` and only references `claim_source_types.md` explicitly (`docs/plans/agent-navigation-spine-refresh-2026-05-30.md:79,161-162`).

## 3. Contradictions or missing dependencies

- Item 1 treats the plan file as an intended edit surface (`docs/plans/agent-navigation-spine-refresh-2026-05-30.md:49-55`), but implementation order starts with root `AGENTS.md` (`:142-149`). Since the plan already contains the scope freeze, Item 1 is really a preflight check, not an implementation step.
- The reference to the global `skill-creator` skill (`docs/plans/agent-navigation-spine-refresh-2026-05-30.md:165`) sits awkwardly beside “No new skills” / “No edits to skills” (`:37-40`). If retained, it should be context-only; otherwise cut it.
- Item 3 depends only on Item 1 (`:81`) even though its success depends on the paper-facing shared docs being the intended bridge surface.

## 4. Risk of over-planning

Cut or simplify Item 1 into an acceptance/preflight note. Compress Item 6 to a short inactive-candidates note, or defer it until after core router docs are aligned. Do not add a file-by-file impact table from the export; the current work-item structure is already enough.

## 5. Questions whose answers would change implementation order

1. Should root `AGENTS.md` be physically reorganized, or should a new top router summarize and point into existing sections?
2. Which `MATH/paper_facing/shared/` docs must be explicit first-class bridge links?
3. Is the inactive future-skill ledger required in this pass, and must it live in `MATH/AGENTS.md` despite the export’s location ambiguity (`prompt-exports/...navigation-spine-pla-30c5.md:62-63`)?
4. Should the `skill-creator` reference be removed before implementation to avoid mixed signals?
