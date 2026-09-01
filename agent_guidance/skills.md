# Agent Skill Index

This index lists only skills that actually exist in the current checkout or are
installed global skills. A paper or lane name alone never triggers a skill.
Ordinary implementation, mathematics, tests, conceptual review, and agent
handoffs follow `AGENTS.md`, the nearest subtree contract, code, and tests.

## Active skills

| Skill | Surface | Trigger |
|---|---|---|
| CHTC operations | `agent_guidance/skills/chtc-direct/SKILL.md` (recreated repo-local 2026-08-31; the lost Global `$chtc-direct` is superseded) | Direct login, submit, status, fetch, cleanup, or storage work. |
| Paper-I runs | `agent_guidance/skills/paper-i-run/SKILL.md` | Actual Paper-I run planning, execution, monitoring, repair, aggregation, or run-evidence reporting. |
| Paper-I noise model | `agent_guidance/skills/paper-i-noise-model-primer/SKILL.md` | Explicit Paper-I noise equations, appendix support, noise plots, or noise-model handoffs. |
| Paper-II runs | `agent_guidance/skills/paper-ii-run/SKILL.md` | Actual checkpoint-McLachlan run or evidence-production work. |
| Paper-II results | `agent_guidance/skills/paper-ii-results/SKILL.md` | Explicit transfer of locked Paper-II evidence into a table or source map. |
| Source-locked sensitivity | `agent_guidance/skills/source-locked-sensitivity/SKILL.md` | A real one-variable run/report claiming identical source settings except one named variable. |
| Journal manuscripts | Global `$journal-math-manuscript-refiner` at `/Users/lordskyler/.codex/skills/journal-math-manuscript-refiner/SKILL.md`; local author overlay/fallback: `agent_guidance/skills/manuscript-editor/SKILL.md`; repository supplement: `MATH/paper_facing/shared/journal_math_skill_supplement.md` | Manuscript review, wording, claims, citations, `.tex`, captions, tables, or PDF-facing edits. Resolve the manuscript contract before selecting a source. |
| Citation evidence | `agent_guidance/skills/citation-evidence/SKILL.md` (restored from `517eca4c` 2026-09-01) | Finding, verifying, or quote-grounding a citation for a scientific claim; checking whether a source supports the intended claim strength. |
| Pedagogical math | Global `$pedagogical-math-primer` | Teaching notes, derivation companions, and explicitly pedagogical artifacts. |
| Paper-II calibration | `agent_guidance/skills/time-dynamics-benchmark-calibration/SKILL.md` (recreated repo-local 2026-09-01; the lost Global `$time-dynamics-benchmark-calibration` is superseded) plus its runner `pipelines/time_dynamics/diagnostics/paper_ii_benchmark_calibration.py` | Paper-II calibration before CHTC submission, aggregation, evidence-PDF generation, or a visible table update. |
| GPT-Pro export | Global `$gpt-pro-handoff` | A requested standalone GPT-Pro dossier or export. |

Load only the smallest matching set. For example, a Paper-I algorithm change
with no run, manuscript, plot, or noise artifact triggers none of these skills.

## Deliberately unavailable repo-local skills

This checkout has no `SKILL.md` for Paper-I results, Paper-III run/results,
Paper-I GPT-Pro review, replay-overlay, visible-prefix-cost, cost-convergence,
or Paper-I CHTC-Optuna skills. Some same-named directories retain useful
scripts, but scripts do not constitute discoverable skills. Do not cite,
simulate, or auto-compose the missing skills. Follow root/MATH/lane contracts
and explicit target scripts; fail closed only when an actual state-changing run,
evidence transfer, or manuscript operation lacks a required contract.

## Run-setting rule

For paper-facing runs, use the best-visible-settings baseline gate in
`agent_guidance/shared/run-guide.md`. The reusable resolver is
`python3 agent_guidance/skills/shared/scripts/resolve_visible_settings.py ...`.
Never substitute current defaults for source-backed settings.
