# Paper II exact-estimator rerun campaign

Date: 2026-09-04

Status: submitted; results pending.

## Scope

This is the repaired estimator-accounting campaign. It is distinct from the
older selector-result fetch and ignores the unrelated cross-task delegation
message that appeared during preparation.

The campaign runs the fifteen locked matched-accuracy cells. Each cell has:

- repaired proxy (`exchange_legacy_proxy_fixed`);
- realized `Gamma(v*)` with compiled cost (`exchange_q_compiled`, retained as
  a legacy internal identifier);
- ideal `Gamma(v)` ranking with compiled cost (`exchange_gamma_compiled`,
  retained as a legacy internal identifier); and
- a fresh AVQDS comparator at the cell's locked AVQDS cut.

Thus the submitted matrix has 45 APM actions plus 15 AVQDS actions. “Final,
not precursors” means every action is a fresh 501-point trajectory from
`t=0` through `t=20` at its already selected locked cut. There are no `t=5`
screening runs, threshold-search rungs, continuation legs, or resume actions.
The historical old-proxy route remains an archival stitched baseline and is
not a fourth active APM algorithm.

## Why AVQDS is rerun

Historical AVQDS artifacts retain aggregate candidate counts but not every
candidate geometry round. Exact candidate-acquisition work cannot be recovered
retroactively. The fresh AVQDS runs therefore use the same locked physics,
controller, cut, time grid, and candidate pool while persisting the new exact
round ledger.

## Telemetry contract

Runtime commit `426807d9ccf599f16bed0d2c7679fd32bdbfca8d` contains:

- exact AVQDS per-round candidate-geometry accounting;
- exact generalized-exchange cache, selector-call, finalist-geometry, and
  local-refit accounting; and
- the reconciled eight-worker compiled-cost execution path, with decision
  parity tested at one, four, and eight workers.

The metric is exact logical dense-statevector estimator work. It is not a
physical shot count; physical shots remain unavailable because no measurement
grouping, target precision, variance, or shot-allocation protocol is defined.

Every successful cluster action must archive a trajectory, an estimator-work
receipt whose total status is `exact`, and a fresh terminal FakeMarrakesh cost.
An incomplete telemetry receipt or failed terminal compilation makes the
action fail for targeted retry rather than silently entering the comparison.

## Calibration and package

The calibration self-test passed. The submission calibration has zero errors
and status `INCOMPLETE`. The only unrun checks are
`benchmark_point_sameness` and `response_normalized_matched_accuracy`; both
require the new results and must be rerun after fetch. Degraded response-scale
mode was not used. The report-refresh resolution has no active submission
block and explicitly authorizes a new campaign.

Package:
`chtc/paper_ii_final_estimator_telemetry_v1_426807d9_20260904`

- source archive SHA-256:
  `09b9d837bae8981a53ca1dd7df206c4b1b7aac322bf0dcf1c35dbe2b4c8fffe5`;
- actions SHA-256:
  `9134012a2f2cc7fcabd0e2277a8d20578c0c40f0f7bffa03a2abcfa91eff6860`;
- locked-cell SHA-256:
  `0d7d25d51504aacba64f8824da2b49241460c9cf70e02f5a4c9d3800f4148030`;
- remote package and container validation: `OK`;
- Condor dry run: 45 APM plus 15 AVQDS actions.

## Submission

Remote workspace:
`/home/jsstrobel/Holstein_paper_ii_final_estimator_telemetry_v1_426807d9_20260904`

| Cluster | Scope | Jobs | CPUs/job | Memory/job | Max runtime |
|---|---|---:|---:|---:|---:|
| 9719355 | three APM routes | 45 | 9 | 6 GB | 432000 s (5 d) |
| 9719356 | fresh AVQDS | 15 | 1 | 6 GB | 432000 s (5 d) |

The first scheduler verification found all 60 jobs idle with the requested
resources and runtime budgets. The older selector clusters were not modified.

## Post-run requirements

After successful archives land:

1. SHA-verify every archive and its per-action receipt.
2. Require `SUCCESS_EXACT`, the bound action/source/lock digests, a 501-point
   `t=0..20` grid, exact logical estimator status, and terminal-cost parity.
3. Rerun the calibration gate over all 60 trajectories; no required check may
   remain unrun before a visible results refresh.
4. Assert same-cell comparability and recompute fidelity plus mean absolute
   errors for energy, doublon, and charge imbalance.
5. Refresh the single established human-facing LaTeX comparison PDF; do not
   adopt results into Paper II until the author decides.
