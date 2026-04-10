# Change Log

## 2026-03-20

### Added

- [HH_MEASUREMENT_BRIEF.md](notes/HH_MEASUREMENT_BRIEF.md)
  Brief HH-only investigation note. Reframed the optimization order around the
  upstream noiseless staged ADAPT core rather than the downstream noisy wrapper.

- [HH_MEASUREMENT_IMPL_GUIDE.md](notes/HH_MEASUREMENT_IMPL_GUIDE.md)
  Detailed implementation guide for improving the staged HH `phase3_v1`
  `|Delta E| / (cost, measurements)` frontier.

- [pipelines/hardcoded/hh_pareto_tracking.py](/home/moh/Holstein_test/pipelines/hardcoded/hh_pareto_tracking.py)
  Added persistent staged-HH Pareto tracking utilities that extract per-depth
  ADAPT tradeoff rows, compute non-dominated frontiers on
  `delta_E_abs` / cumulative measurement groups / cumulative gate proxy, and
  maintain per-run plus rolling sidecars.

- [test/test_hh_pareto_tracking.py](/home/moh/Holstein_test/test/test_hh_pareto_tracking.py)
  Added focused regression coverage for staged-HH Pareto row extraction,
  frontier filtering, and rolling-ledger replacement by `run_tag`.

### Changed

- [pipelines/hardcoded/hh_continuation_types.py](/home/moh/Holstein_test/pipelines/hardcoded/hh_continuation_types.py)
  Extended `CompileCostEstimate` with backward-compatible richer gate-burden
  fields:
  `cx_proxy_total`, `sq_proxy_total`, `gate_proxy_total`, `max_pauli_weight`.

- [pipelines/hardcoded/hh_continuation_scoring.py](/home/moh/Holstein_test/pipelines/hardcoded/hh_continuation_scoring.py)
  Added QWC-style measurement group derivation and coverage-aware measurement
  reuse in `MeasurementCacheAudit`.
  Added `measurement_group_keys_for_term(...)`.
  Upgraded `Phase1CompileCostOracle` from term-count-style proxying to a richer
  gate-like proxy while preserving the existing interface.
  Added measurement-affinity handling to `compatibility_penalty(...)` and
  `greedy_batch_select(...)`.
  Made `depth_cost` follow the stronger gate proxy when available.

- [pipelines/hardcoded/adapt_pipeline.py](/home/moh/Holstein_test/pipelines/hardcoded/adapt_pipeline.py)
  Wired staged ADAPT phase-1 and phase-2 measurement scoring to the new
  measurement group keys instead of label-only reuse.
  Wired compile-cost estimation to the actual candidate term.
  Changed batch measurement-cache commit logic to record measurement keys from
  the admitted term, including runtime-split children.
  Extended compile-cost proxy summary metadata to mention the richer proxy
  components.

- [test/test_hh_continuation_scoring.py](/home/moh/Holstein_test/test/test_hh_continuation_scoring.py)
  Added focused regression coverage for:
  measurement-group merging,
  coverage-aware measurement reuse,
  heavier Pauli structures producing larger gate proxies,
  and measurement-aware compatibility penalties.

- [pipelines/hardcoded/hh_staged_workflow.py](/home/moh/Holstein_test/pipelines/hardcoded/hh_staged_workflow.py)
  Wired the staged HH noiseless workflow to emit persistent Pareto artifacts.
  The workflow JSON now records Pareto artifact paths and summary counts, and
  the stage summary now preserves ADAPT measurement-cache, compile-proxy, and
  runtime-split summaries needed for cost/frontier analysis.

- [pipelines/hardcoded/hh_staged_noiseless.py](/home/moh/Holstein_test/pipelines/hardcoded/hh_staged_noiseless.py)
  Extended the CLI contract to print the per-run and rolling Pareto artifact
  paths alongside the existing workflow and replay outputs.

- [test/test_hh_staged_noiseless_workflow.py](/home/moh/Holstein_test/test/test_hh_staged_noiseless_workflow.py)
  Extended workflow integration coverage to assert Pareto artifact emission and
  preservation of ADAPT cost summaries in the staged HH payload.

### Verification

- `py_compile` passed for:
  - [pipelines/hardcoded/hh_continuation_types.py](/home/moh/Holstein_test/pipelines/hardcoded/hh_continuation_types.py)
  - [pipelines/hardcoded/hh_continuation_scoring.py](/home/moh/Holstein_test/pipelines/hardcoded/hh_continuation_scoring.py)
  - [pipelines/hardcoded/adapt_pipeline.py](/home/moh/Holstein_test/pipelines/hardcoded/adapt_pipeline.py)
  - [test/test_hh_continuation_scoring.py](/home/moh/Holstein_test/test/test_hh_continuation_scoring.py)
  - [pipelines/hardcoded/hh_pareto_tracking.py](/home/moh/Holstein_test/pipelines/hardcoded/hh_pareto_tracking.py)
  - [pipelines/hardcoded/hh_staged_workflow.py](/home/moh/Holstein_test/pipelines/hardcoded/hh_staged_workflow.py)
  - [pipelines/hardcoded/hh_staged_noiseless.py](/home/moh/Holstein_test/pipelines/hardcoded/hh_staged_noiseless.py)
  - [test/test_hh_pareto_tracking.py](/home/moh/Holstein_test/test/test_hh_pareto_tracking.py)
  - [test/test_hh_staged_noiseless_workflow.py](/home/moh/Holstein_test/test/test_hh_staged_noiseless_workflow.py)

- Runtime smoke passed for:
  - [pipelines/hardcoded/hh_pareto_tracking.py](/home/moh/Holstein_test/pipelines/hardcoded/hh_pareto_tracking.py)
    Verified row extraction plus sidecar emission using `python3` and a
    temporary output directory.

- `pytest` could not be run in the available environments because `pytest` was
  not installed, and the repo venv also lacked runtime deps such as `numpy`.

### Notes

- This log entry covers only the changes made in this session.
- The worktree already contained unrelated local edits, including
  [pipelines/run_guide.md](/home/moh/Holstein_test/pipelines/run_guide.md),
  which are not described here.
