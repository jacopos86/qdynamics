---
name: paper-ii-results
description: Transfer locked Paper-II checkpoint-McLachlan evidence into explicitly requested manuscript or report table cells and source maps. Use only for a concrete Paper-II evidence-to-table audit or edit after paper-ii-run validation; do not use for run execution, ordinary dynamics implementation, conceptual analysis, agent handoffs, or unrelated manuscript prose.
---

# Paper II Results: dynamics evidence-to-table transfer

Use this skill after `paper-ii-run` has produced or identified validated
Paper-II dynamics evidence. Root `AGENTS.md`, `MATH/AGENTS.md`, and this file
contain the applicable preservation and table-boundary rules.

## Boundary with `paper-ii-run`

- Use `agent_guidance/skills/paper-ii-run/SKILL.md` to plan, launch, monitor, repair, aggregate, report, or promote Paper-II runs.
- Use this skill to consume completed Paper-II artifacts and update existing table data cells/source maps only by default.
- Do not launch CHTC jobs, change controller settings, promote artifacts, or treat missing evidence as table-ready data.
- If evidence is missing, stop with a blocker and return a Paper-II run repair/rerun target.

## Required context

1. Root `AGENTS.md` and `MATH/AGENTS.md`.
2. `agent_guidance/skills/paper-ii-run/SKILL.md` for run/evidence semantics.
3. `$time-dynamics-benchmark-calibration` output when available.
4. `$journal-math-manuscript-refiner` and `MATH/paper_facing/shared/journal_math_skill_supplement.md` before any manuscript-facing advice, proposed wording, prose/caption/table-language plan, `.tex` edit, or PDF-facing work.
5. Target table block for `tab:dyn_claims` and nearby provenance comments for source mapping only.
6. `MATH/paper_facing/paper_II_dynamics/` support docs and source maps named by the handoff.
7. `chtc/generic_time_dynamics_table/input/class_settings/paper_ii_class_settings_lock_v1.json` when class settings are relevant.

## Fail-closed blockers

Stop instead of updating a visible Paper-II table cell when any of these are true:

- calibration audit has `ERROR`s or required checks were not run;
- aggregate is diagnostic-only, exact-assisted in decision logic, or lacks QPU-faithful decision-data separation for a strict row;
- methods in one benchmark point do not share the same static seed hash, drive, grid, cutoff, observable set, diagnostic reference, and compile target;
- seed artifact is staged, fallback, pending, recovery, or missing locally when paper-facing evidence requires it;
- a trajectory-freeze invalid row would be used in an automated table/evidence update without explicit user approval;
- required displayed metrics are missing without an explicit missing-evidence status;
- an existing completed visible cell would be erased, downgraded, or replaced by incomplete/running evidence without explicit user approval.

## Default edit scope

Edit existing Paper-II data cells/source maps only. Do not edit or propose replacement prose, captions, table structure, labels, narrative comments, literature-positioning text, or paper-edit plans unless the user explicitly asks and `$journal-math-manuscript-refiner` is active.

## Handoff/report

Report changed cells, source artifacts, calibration status, QPU-faithful status, missing metrics, PDF rebuild status when applicable, and blockers.
