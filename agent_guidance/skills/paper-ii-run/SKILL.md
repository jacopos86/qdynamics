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
- Calibration skill: `agent_guidance/skills/time-dynamics-benchmark-calibration/SKILL.md`

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
- Before CHTC submission, aggregation, evidence-PDF generation, or table update, run the `agent_guidance/skills/time-dynamics-benchmark-calibration/SKILL.md` workflow and fail closed on calibration `ERROR`s.
- For the staged matched-accuracy matrix, use the canonical response-normalized
  policy from `agent_guidance/time-dynamics/paper-ii-comparison-protocol.md`.
  Normalize energy, doublon, and maximum site-occupation errors by the bound RMS
  exact response for the same drive and horizon. The individual target is five
  percent with mean infidelity at most `1e-4`. A completed APM/AVQDS pair may
  also be admitted when both arms remain inside the fifteen-percent response
  guard and absolute safety ceilings and their aggregate accuracy burdens agree
  within a factor of two. This pair test is mandatory before a cost discrepancy
  becomes a paper-facing comparison. An unmatched miss tightens the method-owned
  activation cut. There is no lower fidelity bound and no coarsening branch.
  The former infidelity-window study is preserved historical evidence only,
  and the fixed componentwise-`1e-4` policy likewise must not classify new
  campaign work.
- Every method inside one benchmark point must share the same static seed hash, drive, time grid, phonon cutoff, reference/diagnostic policy, observable set, diagnostic exact reference, and compile target.
- Do not use staged, fallback, pending, or recovery seed artifacts in paper-facing aggregates.
- A dynamics row is invalid for the paper table if the exact diagnostic trajectory moves while the algorithm trajectory freezes, e.g. `theta_dot_l2=0`, `rho_num=0`, `rho_miss≈1`.
- Exact/reference data is valid for diagnostics and report columns, never for controller decisions, append/prune admission, integrator policy, candidate scoring, or strict Optuna online feedback.
- Cover all Hamiltonians wired through the relevant static-seed/dynamics path unless the user explicitly narrows to smoke/diagnostic scope. If a wired dynamics path is broken, repair it first, prove the repair narrowly, then rerun/resume the same scientific target.

## QPU-faithful boundary

For strict/QPU-faithful rows, controller decisions must use measurement-compatible data for the prepared ansatz/circuit state. Exact ED/reference trajectories are diagnostic side channels only. Follow the detailed QPU-faithful contract in root `AGENTS.md` and fail closed on exact-target/reference leakage into decision telemetry.

## Evidence handoff to results

For an already-authorized active campaign, a scheduler snapshot that discovers
one or more newly completed process IDs is not a terminal status-only event.
Before the run/status turn may finish, hand that exact process delta through
packaging, verified transfer, artifact validation, exact evaluation, ledger
ingestion, results-PDF refresh, and the results skill's table refresh. If any
step fails, write a durable process-scoped blocker receipt and refresh the
lightweight results-PDF status view before reporting the blocker. It is never
acceptable to report that completed rows merely “await ingestion” while
leaving the visible report stale.

Ingestion must replay the campaign-construction inputs recorded by the
submission package exactly, including whether a prior-threshold root was
absent. Do not invent, add, remove, or substitute a `prior_root` during
record-package validation; doing so changes `StagedCampaign.next_actions()` and
can falsely classify a valid submitted action as stale.

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
