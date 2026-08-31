# Critique: Paper-I Comparator SPSA Optuna Calibration Plan

## Context/Scope

Reviewed only the saved plan, the original context-builder export, and mandatory run-routing guidance; no broader codebase exploration was performed.

## Findings

### 1. Top 3 under-specified seams

1. **Approved config contract.** The plan repeatedly gates on an “approved config JSON” (`docs/plans/paper-i-comparator-spsa-optuna-calibration-2026-05-31.md:83-96,202,280`) but does not define its path, schema, approval metadata, smoke-vs-full distinction, hash semantics, or search-space grammar. Item 1 says “config validation helpers” (`:104-106`), but the rest of the plan can still force each runner/generator/preflight piece to guess.
2. **Multi-case target row + env isolation.** Targets group two Hubbard cases and two spin-boson cases into single calibration records (`:40-47`), and each trial calls `run_single()` for every case (`:81-85`) while the runner depends on process-global env (`:187`). The plan does not specify row encoding for `case keys`, per-case output names, env save/restore boundaries, or stale-field clearing beyond a late test bullet (`:251`).
3. **Objective/progress telemetry.** The loss names `abs_delta_e`, `nfev_or_shots`, clipping, and resource precedence (`:86-92`) without pinning normalized row fields. The runner writes `progress/current.json` (`:187`), but the run skill requires a live current-best ledger with enough content to answer best-trial status (`agent_guidance/skills/run/SKILL.md:583-599`). “Current” vs “current best” is a seam.

### 2. Specificity balance

- The plan may over-specify tactical Optuna mechanics: `TPESampler(seed=...)`, sqlite, `load_if_exists=True`, and exact progress filenames (`docs/plans/...:187`). The export framed Optuna/heartbeat code as patterns to emulate, not required choices (`prompt-exports/oracle-plan-2026-05-31-140457-comparator-spsa-cali-3e40.md:17,56-58`).
- The saved plan drops useful export framing: the generated export included a final “Update planning docs only” item and explicit risk/migration notes (`prompt-exports/...:506-545`). The plan keeps the evidence gate (`docs/plans/...:98`) but loses the closeout reminder that no persistence/manuscript/artifact migration is part of this pass.

### 3. Contradictions or missing dependencies

- Item 6 loads “one calibration row” (`docs/plans/...:187`), but Item 7 creates the row generator afterward and depends on Item 6 (`:198-210`). Either Item 1 must define a complete row schema, or Items 6/7 are circular.
- Smoke/full configs are required (`:96,202,251`), but no config schema/template/smoke fixture appears in `Files to edit` (`:306-322`).
- CHTC templates transfer code directories (`:217`), but the plan omits an explicit dependency/build-image check even though the export selected existing CHTC dependency surfaces (`prompt-exports/...:37,78-89`).

### 4. Risk of over-planning

Items 8-9 (wrappers, submit templates, preflight, fetched-output checker) look premature until the config schema, local runner, and progress/current-best contract are closed. Item 10 as one large “tests at end” bucket also hides which seams must be proven before CHTC-facing files are worth adding.

### 5. Questions whose answers would change implementation order

1. Is the approved config schema/template/smoke fixture versioned in this implementation pass, or supplied later by the user?
2. Should a calibration record be method-target with multiple cases, or method-case with aggregation outside the runner?
3. Must the lane emit `current_best.json`-compatible live progress, or is `progress/current.json` accepted as equivalent?
4. Should CHTC dependency/image validation be implemented before submit templates, or only as preflight metadata?
