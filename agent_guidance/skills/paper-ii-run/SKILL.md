---
name: paper-ii-run
description: Manage actual Paper-II checkpoint-McLachlan dynamics benchmark execution, including launch, monitoring, repair, aggregation, evidence reports, class-tuned settings studies, same-seed comparator batches, and CHTC jobs. Use only when the current request plans or operates a dynamics run or consumes its evidence; do not use for ordinary dynamics implementation, mathematical explanation, unit tests, agent handoffs, or manuscript prose.
---

# Paper II Run: checkpoint-McLachlan dynamics manager

Use this skill for Paper-II time-dynamics evidence production and repair. Root
`AGENTS.md`, `MATH/AGENTS.md`, and this file contain the applicable run gates.

## Response guidance

Keep routine run/status updates concise and evidence-based. A progress update
does not end an authorized monitoring, repair, aggregation, or report workflow.
Use a compact table when the user asks for current run status and additional
detail only when a blocker, failure, or final handoff requires it. Do not force
a fixed response template unrelated to the user's question.

## Active Paper-II paths

- Manuscript: `MATH/paper_details/time_dynamics_paper_II.tex`
- Primary visible table: `tab:dyn_claims`
- Paper support: `MATH/paper_facing/paper_II_dynamics/`
- Current class setting lock: `chtc/generic_time_dynamics_table/input/class_settings/paper_ii_class_settings_lock_v1.json`
- Calibration skill: `$time-dynamics-benchmark-calibration` when available

## Read order

1. Root `AGENTS.md` for global invariants and QPU-faithful data-flow rules.
2. `MATH/AGENTS.md` for paper-program policy.
3. `agent_guidance/time-dynamics/AGENTS.md` and `agent_guidance/time-dynamics/run-guide.md`.
4. The visible `tab:dyn_claims` row/claim being targeted.
5. Paper-II support/source-map/audit docs named by that visible target.
6. Machine-readable class settings, seed manifests, source maps, aggregates, and run manifests referenced by the target.

Do not mine hidden comments or generated artifacts as a run queue. Visible table rows, figure panels, and reader-facing claims define paper-facing work.

If a Paper-II run/report discussion turns into manuscript wording, caption/table-prose advice, literature positioning, or a paper-edit plan, pause run-skill mode and use `$journal-math-manuscript-refiner` before proposing text in chat or applying edits.

## Paper-II run gates

- Primary method first in run reports: checkpoint McLachlan.
- Use class-level checkpoint-McLachlan settings from the current promoted lock unless the user approves a newer promoted lock.
- Controller settings are class-level, not per-Hamiltonian tuning for table claims.
- Before CHTC submission, aggregation, evidence-PDF generation, or table update, run the `$time-dynamics-benchmark-calibration` workflow when available and fail closed on calibration `ERROR`s.
- Every method inside one benchmark point must share the same static seed hash, drive, time grid, phonon cutoff, reference/diagnostic policy, observable set, diagnostic exact reference, and compile target.
- Do not use staged, fallback, pending, or recovery seed artifacts in paper-facing aggregates.
- A dynamics row is invalid for the paper table if the exact diagnostic trajectory moves while the algorithm trajectory freezes, e.g. `theta_dot_l2=0`, `rho_num=0`, `rho_miss≈1`.
- Exact/reference data is valid for diagnostics and report columns, never for controller decisions, append/prune admission, integrator policy, candidate scoring, or strict Optuna online feedback.
- Cover all Hamiltonians wired through the relevant static-seed/dynamics path unless the user explicitly narrows to smoke/diagnostic scope. If a wired dynamics path is broken, repair it first, prove the repair narrowly, then rerun/resume the same scientific target.

## QPU-faithful boundary

For strict/QPU-faithful rows, controller decisions must use measurement-compatible data for the prepared ansatz/circuit state. Exact ED/reference trajectories are diagnostic side channels only. Follow the detailed QPU-faithful contract in root `AGENTS.md` and fail closed on exact-target/reference leakage into decision telemetry.

## Evidence handoff to results

When evidence is complete, hand off to `agent_guidance/skills/paper-ii-results/SKILL.md` with:

- run class and visible target;
- class settings lock/source settings record;
- static seed hash and seed source artifact for every method;
- same-seed/same-grid/same-drive/same-observable audit;
- calibration audit path/status;
- QPU-faithful decision-data-flow status;
- aggregate/source JSON paths and hashes;
- PDF/report path if generated;
- missing metrics or diagnostic-only blockers.

The Paper-II results skill consumes locked evidence; it does not launch or repair runs.
