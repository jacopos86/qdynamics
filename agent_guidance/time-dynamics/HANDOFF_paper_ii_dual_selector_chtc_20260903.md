# Handoff — Paper II dual-selector campaign: clear the submission block, package and submit the 45 actions

**Author:** Fable (Claude Code), 2026-09-03 · **Executor:** Codex with repo and CHTC access
**Contract:** read `agent_guidance/shared/agent-handoff-contract.md`, then
`agent_guidance/skills/paper-ii-run/SKILL.md` and
`agent_guidance/skills/chtc-direct/SKILL.md` before any scheduler action.
This file supplies only what is specific to this task. Implementation report:
`docs/reports/PAPER_II_DUAL_SELECTOR_20260902.md` (read §1, §4.2c, §6, §7).

## 1. Anchors

| field | value |
|---|---|
| Checkout | `/Users/lordskyler/Holstein_test` (active M4 checkout); work in your own worktree under `Holstein_test.worktrees/` |
| Branch | `fable/paper-ii-dual-selector-20260902` (pushed) — run from it directly; the user has waived a separate review. Merging into `main` is the user's call, not this task's |
| Base | `origin/main` `557f20ae`; the branch cherry-picks the parallelization commits and adds the three selector bundles |
| Test baseline | `PYTHONPATH=. .venv/bin/python -m pytest -q -p no:randomly test/test_ap_mclachlan_*.py test/test_compiled_ansatz.py test/test_paper_ii_runs.py test/test_paper_ii_final_selector_comparison.py test/test_paper_ii_staged_accuracy_campaign.py test/test_time_dynamics_campaign.py` → **330 passed** at `5ee42540`; the four pre-existing `test_siam_runner.py` reds are on `main` and unrelated |
| Registry | `PYTHONPATH=. .venv/bin/python -m pipelines.time_dynamics.paper_ii_runs list` shows arms `exchange_legacy_proxy_fixed`, `exchange_q_compiled`, `exchange_gamma_compiled` |
| Action generator | `PYTHONPATH=. .venv/bin/python -m pipelines.time_dynamics.campaigns.paper_ii_final_selector_comparison_v1 --output-root raw_outputs --actions-jsonl <path> --compile-workers 8` → 45 APM actions, 0 AVQDS, `PREPARED_NOT_SUBMITTED` |
| Memory contract | `agent_guidance/shared/memory-budget.md` — nothing heavy runs locally in this task |

## 2. Scope

**In scope:** resolving `report_refresh_required.json`; the calibration
audit; building one CHTC package for the 45 prepared actions; submission
in one authenticated session; quiet timed monitoring; fetch, SHA, ledger
ingest, `assert_comparable` aggregation; one comprehensive refreshed PDF.

**Out of scope, with reasons:** any change to selector code, route bundles,
`τ_del`, placement rule, guards, cuts, seeds, or numerics — those are
scientific settings and the report's §6 lists the author-owned choices;
manuscript or Overleaf edits; evidence adoption or promotion (user-only);
the 15 AVQDS comparators — they are reused from the locked historical
cells, not rerun. If any step appears to require touching these, stop and
report.

## 3. Increments

Proceed through I1–I5 autonomously; I0 is optional and cheap. Commit at
each boundary with explicit paths; never `git add -A`.

### I0 (optional) — one diagnostic pilot on a CHTC slot
Goal: confirm compile time per checkpoint on CHTC CPUs before 45 full
trajectories. Run ONE `exchange_q_compiled` cell at `--horizon t5` with the
same flags as the campaign actions on a `request_cpus = 9` slot. It is a
diagnostic, not a campaign cell: keep it outside the campaign directory and
never ingest it as evidence. Expected: per large-miss checkpoint compile
≈ 5–10 s at 8 workers (M4 measured 26.3 s per 4-point smoke at 8 workers,
3.2× over serial). Stop condition: > 60 s per checkpoint — report before
submitting the 45.

### I1 — resolve the submission block
Goal: `chtc/paper_ii_staged_accuracy_euler_v1_t20_resume_repair_d3301581_20260826/report_refresh_required.json`
no longer blocks `further_campaign_submission`. Follow its own
`required_resolution.steps`: ingest the two attempts in
`FETCHED_VERIFIED_GATE_PENDING` (clusters 9698247, 9698264), compute the
exact-reference gates, rebuild
`output/pdf/paper_ii_staged_accuracy_campaign/paper_ii_staged_report.pdf`,
record the refreshed ledger/report digests, row count, and qualified pair
count, then remove or update the block. Ingestion must replay the
campaign-construction inputs recorded in the submission package exactly
(do not add, remove, or substitute a `prior_root`).
Expected: ledger record count > 47; block cleared. Stop: any gate ERROR.

### I2 — calibration
```
PYTHONPATH=. .venv/bin/python pipelines/time_dynamics/diagnostics/exact_reference_store.py      # ~22 s, six references
PYTHONPATH=. .venv/bin/python -m pipelines.time_dynamics.diagnostics.paper_ii_benchmark_calibration --self-test
PYTHONPATH=. .venv/bin/python -m pipelines.time_dynamics.diagnostics.paper_ii_benchmark_calibration \
  --artifact MATH/paper_facing/paper_II_dynamics/<receipt>.json --run-root output/<campaign>/ \
  --receipt-json output/diagnostics/paper_ii_calibration/<stamp>/calibration_receipt.json
```
`ERROR` stops. `INCOMPLETE` is not a pass: carry the unrun-check list into
the package manifest. `--allow-unverified-response-scales` stays a degraded
mode; do not report it as `OK`.

### I3 — package
Goal: one CHTC package for exactly the 45 actions.
1. `python -m pipelines.time_dynamics.campaigns.paper_ii_final_selector_comparison_v1 --output-root raw_outputs --actions-jsonl chtc/<campaign>/input/actions.jsonl --compile-workers 8` — every argv must carry `--compile-workers 8`, `--record-statevector`, `--t-final 20.0 --num-times 501`, and its route's `--exchange-route-bundle`; none carries `--resume-from-run-json`. Re-parse each argv with the live runner parser (the generator does; assert it did).
2. Build the package on the `staged_campaign_chtc.py` shape: `manifest.json` with `status: PREPARED_NOT_SUBMITTED`, 40-hex clean `code_revision`, seed sha `7a36a2da…`, the lock JSON's sha, `compile_profile`, and the 45 action rows.
3. `submit.sub`: `request_cpus = 9`, `request_memory = 6GB` (measured 1.1 GB process tree at 4 workers on M4; 8 workers ≈ 2 GB; headroom for support growth), `request_disk` as the staged campaign uses, `+MaxRuntime` sized for a t = 20 trajectory (compile ≈ 5–10 s × in-debt checkpoints at 8 workers, plus propagation).
4. **Wrapper — mandatory:** `run_action_apptainer.sh` must export
   `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`
   before invoking Python. The compile pool pins its own workers (BLAS and
   `RAYON_NUM_THREADS`), but the parent's OpenBLAS binds at `numpy` import,
   which the runner cannot undo. Today the wrapper exports nothing.
5. Do not pass `--compile-overlap-solve-warm`: measured neutral on this
   seed's solve share; it only pays where solves dominate.
Expected: `prepare_audit.json` `PASS_NOT_SUBMITTED`; 45 rows; 0 AVQDS rows.
Stop: any argv fails the live parser, any lock/seed sha mismatch.

### I4 — submit and monitor
One authenticated 12-hour session; record exact cluster IDs in the
package; confirm scheduler records once; then the run-guide intervals
(2–8 h remaining → 30–60 min checks). Do not touch any other cluster,
including HH factorial cluster 9717614.

### I5 — fetch, verify, aggregate, report
Fetch with SHA verification; ingest into the ledger; `assert_comparable`
on every APM/AVQDS pair before any table; rerun the exact-reference and
pair gates (changed methods do not inherit the historical pass); one
comprehensive PDF pairing each of the three routes with its locked AVQDS
comparator, with: gate/source status, final and mean infidelity, mean
absolute errors of E / doublon / Δn, accepted/attempted/rejected/subdivided
time steps, graph costs, terminal N2q/D2q/Dc (live-θ convention, distinct
from the selector's structural-θ K), online K, compile counts, Q/Γ
disagreement and `realized_l2_not_improved` rejections, wall / estimator /
compiler / cache resources. Failed runs stay failed; targeted cutoff
recovery is a later task.

## 4. Traps

- The lock's `source_artifacts`, `run_lock` fingerprints, `metrics`, and
  `paper_status` are `null` on purpose: the promotion-preview and ledger
  JSONs named by the plan were absent locally. Fill them only from artifacts
  you actually fetch and hash; never from the plan's prose.
- `report_refresh_required.json` also blocks *claiming the visible PDF is
  current*; the lock records `currency_claim: not_asserted`. Keep it so
  until I5 completes.
- Historical route IDs are never reinterpreted; the K ≡ 1 historical rows
  remain valid post-run reporting (only online selection was cost-blind).
- Terminal compiled resources bind live θ; selector K binds structural
  θ = 1. Do not present one as the other.
- `output/` and `prompt-exports/` are gitignored; the package and receipts
  live under `chtc/<campaign>/`.
- Another agent's broad `git add` can sweep your work; commit at each
  increment.

## 5. Report back

One block per increment in the contract's format (Status / Commands /
Result / Verification / Deviations), with commit SHAs, cluster IDs, and the
receipt paths. A blocked increment reports the blocker, not a workaround.
