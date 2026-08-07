## Final Prompt
<taskname="SNAKE work accounting"/>
<task>
Audit and plan a robust Paper-I Hubbard-Holstein SNAKE mechanism-resolved algorithmic work accounting path. Do not edit files. Produce a plan for a stable function/reconstructor that can split branch-local S-Alg/S-ELG into: candidate exposure, winner-lineage/refit work, and true quantum measurement bins for gradient, Gram/novelty, Schur/second-order/Hessian-like scoring, Phase-3 scoring, and Hamiltonian/refit evaluations. Treat `candidate_count_total`, `records_evaluated`, shortlist/retained counts, and similar fields as exposure/diagnostic counters only; they must feed phase/window-specific formulas and must not be promoted directly to `S_alg`.
</task>

<architecture>
- `pipelines/static_adapt/adapt_pipeline.py` emits per-step controller events, branch-local history rows, `outer_nfev`/`nfev_opt`/Schur guard fields, shortlisted/scored/retained records, and final `controller_measurement_work_summary`.
- `pipelines/static_adapt/selector_measurement_proxy.py` defines the controller work ledger schema, typed actual/common exposure operator-probe counts, candidate exposure counters, and aggregation by phase/scope.
- `pipelines/exact_bench/snake_table_i_measurement_work.py` is the strict canonical SNAKE work normalizer. Current coarse `S_alg` is reconstructed as `N_grad = phase0 + phase1`, `N_metric = phase2 + phase3`, `N_H_refit_eval = history[*].nfev_opt + resume/final refit nfev`, `N_H_outer_eval = nfev_total - N_H_refit_eval`, plus `N_other_quantum = 0` in the runtime event ledger.
- `pipelines/scaffold/hh_continuation_scoring.py` contains the source formulas for Phase-II novelty/Gram/projection/trust-region scoring, Phase-III reduced-geometry scoring, curvature/Hessian-like oracles, and batch reduced-plane Gram/Schur telemetry.
- `pipelines/scaffold/hh_continuation_pruning.py`, `pipelines/static_adapt/schur_warm_start.py`, and `pipelines/static_adapt/batch_ordering.py` are formula/evaluation sources for Schur surrogate scoring, guarded warm-start objective evaluations, recoverability prune refits, and finite-step batch ordering evaluations.
- `pipelines/reporting/build_paper_i_hh_weak_weak_snake_mechanism_ablation_pdf.py` is the current PDF/sidecar builder for `paper_i_hh_weak_weak_snake_mechanism_ablation_20260708_v2`; it currently flattens `S_alg_N_grad_probe`, `S_alg_N_metric_probe`, and H-eval bins into report rows.
</architecture>

<selected_context>
`pipelines/exact_bench/snake_table_i_measurement_work.py`: strict SNAKE normalization slices: raw proxy rejection, candidate-ledger audit, typed operator probe extraction, phase0 inference, terminal/prefix/beam winner-lineage reconstruction, fair common exposure, deterministic shot proxy, runtime event ledger, and row aggregation.
`pipelines/static_adapt/selector_measurement_proxy.py`: controller work schema constants, candidate ledger aggregation, actual/common exposure operator-probe ledgers, `_record_controller_work_for_records()`, and run-level `controller_proxy_from_history_rows()`.
`pipelines/static_adapt/adapt_pipeline.py`: focused slices where beam/non-beam phase0/phase1/phase2/phase3/batch controller events are recorded, branch/non-branch history nfev fields are emitted, and final payload summary is built.
`pipelines/scaffold/hh_continuation_scoring.py`: Phase-II novelty/projection/trust-region formulas, Phase-III score formulas, raw Gram/tangent/Hessian helpers, novelty/curvature oracles, candidate feature telemetry, and batch reduced-plane scoring slices.
`pipelines/scaffold/hh_continuation_pruning.py`: Schur and metric-regularized prune surrogate ladders, score builders, and recoverability prune remove/refit loop.
`pipelines/static_adapt/schur_warm_start.py`: scalar Schur seed formula, append/batch seed proposals, and guarded seed objective-evaluation accounting.
`pipelines/static_adapt/batch_ordering.py`: finite-step `_order_batch_records_for_admission()` and Schur batch-context extraction.
`pipelines/static_adapt/phase_shortlists.py`: shortlist helpers that define candidate/shortlist/retained counts for Phase1/Phase2/Phase3.
`pipelines/static_adapt/engine_support.py`: beam branch/scratch dataclasses, exact gradient-surface telemetry, active refit window helpers.
`pipelines/static_adapt/runtime_split.py`, `controller_phase_state.py`: runtime split policy and phase liveness/cap/shot helpers.
`pipelines/reporting/build_paper_i_hh_weak_weak_snake_mechanism_ablation_pdf.py`: current report row compilation and artifact paths.
`test/test_snake_table_i_measurement_work.py`: selected strict tests for candidate-counter blocking, explicit candidate ledger requirements, common exposure/fair work, beam winner lineage, display-prefix scope, outer_nfev, and runtime event ledger totals.
`test/test_static_adapt_controller_measurement_work.py`: full tests for controller work recording and common exposure probe payloads.
`chtc/phase3_optuna/input/paper_i_hh_weak_weak_snake_mechanism_ablation_20260708_v2/paper_i_hh_spsa_budget_ladder_manifest.json` and `chtc/phase3_optuna/submit_paper_i_hh_weak_weak_snake_mechanism_ablation_20260708_v2.sub`: run/provenance roots and expected output paths.
`pipelines/scaffold/hh_continuation_types.py`: auto codemap only; use it for `CandidateFeatures` field names such as `phase2_geometry_window_indices`, `schur_window_indices`, `phase3_geometry_refit_window_indices`, `nested_window_accounting`, and `schur_window_solve`.
</selected_context>

<relationships>
- `adapt_pipeline` phase events -> `_record_controller_work_for_records()` -> `ControllerMeasurementWorkAccumulator` -> `controller_measurement_work_summary` -> `snake_algorithmic_work_from_payload()`.
- Current coarse work: controller `phase0/phase1` actual operator probes -> gradient bin; controller `phase2/phase3` actual operator probes -> metric bin; history/refit nfev fields -> H outer/refit bins.
- Beam runs: branch-local controller proxies live in history rows; terminal row accounting uses winner-lineage rows, while aggregate beam search totals are provenance (`S_beam_search_total`) and not row `S_alg`.
- Scoring formulas attach telemetry to candidate records through `build_full_candidate_features()`; shortlist and retained counts describe exposure/winnowing, while actual operator probe counts remain the quantum measurement basis for coarse `S_alg` reconciliation.
- Schur/second-order sources are split across scoring (`Phase2CurvatureOracle`, Hessian entries, `schur_window_solve`), pruning Schur ladders, Schur warm-start guard trials, and batch reduced-plane Schur context.
</relationships>

<validation_targets>
- Any proposed sub-bin reconstructor should reconcile exactly to current strict coarse bins where possible: gradient subbins sum to `S_alg_N_grad_probe`; Gram/novelty + Schur/second-order + Phase3 scoring subbins must explain or partition `S_alg_N_metric_probe`; H/refit bins must reconcile to `S_alg_N_H_outer_eval + S_alg_N_H_refit_eval`.
- Preserve strict blockers: no raw shot proxies, group counts, `records_evaluated`, `candidate_count_total`, or missing candidate ledgers as direct `S_alg` sources.
- Validate terminal, `display_prefix`, and beam winner-lineage scopes; test missing/ambiguous `outer_nfev` and candidate ledger cases.
- Add/extend tests beside `test/test_snake_table_i_measurement_work.py` and `test/test_static_adapt_controller_measurement_work.py`; use selected tests as the expected style.
</validation_targets>

<ambiguities>
- Raw fetched result JSONs and generated PDF sidecars are referenced by paths but not selected here; RepoPrompt did not expose the output/fetch roots in the available project tree.
- Current `snake_algorithmic_work.json`/runtime event ledger has coarse bins only; named `S_Gram`, `S_novelty`, `S_Schur`, and Phase-3 subbins are not yet explicit.
- The selected CHTC manifest/submit files capture v2 run roots and source transfers, but the larger mechanism-ablation TSV/generator was omitted to stay under budget.
</ambiguities>

## Selection
- Files: 17 total (5 full, 11 slice, 1 codemap)
- Total tokens: 110744 (Auto view)
- Token breakdown: full 6985, slice 99406, codemap 4353

### Files
### Selected Files
/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/
├── chtc/
│   └── phase3_optuna/
│       ├── input/
│       │   └── paper_i_hh_weak_weak_snake_mechanism_ablation_20260708_v2/
│       │       └── paper_i_hh_spsa_budget_ladder_manifest.json — 1,985 tokens (full)
│       └── submit_paper_i_hh_weak_weak_snake_mechanism_ablation_20260708_v2.sub — 721 tokens (full)
├── pipelines/
│   ├── exact_bench/
│   │   └── snake_table_i_measurement_work.py — 19,189 tokens (lines 1-180 (Constants, strict component aliases, and raw-proxy rejection helpers for S_norm/S_alg normalization.), 190-455 (Payload/controller summary helpers, candidate-work ledger audit, and strict typed operator-probe extraction for actual/common exposure phase counts.), 529-860 (Phase0 gradient inference, history/refit nfev reconstruction, and terminal runtime component formulas for coarse S_alg bins.), 1022-1185 (Beam aggregate detection, beam-search provenance, winner-lineage terminal reconstruction, and public shot-proxy entry point.), 1299-1605 (Explicit/runtime normalization dispatch, prefix scope row slicing, scoped runtime reconstruction, and snake_algorithmic_work_from_payload entry point.), 1682-1865 (Fair expanded common-exposure work and deterministic shot proxy public helpers.), 1959-2145 (Runtime reconstruction event ledger, candidate-work ledger validation, embedded ledger handling, and normalize_snake_measurement_work_row start.), 2355-2438 (S_alg component extraction and aggregate event-ledger helper for enriched rows.))
│   ├── reporting/
│   │   └── build_paper_i_hh_weak_weak_snake_mechanism_ablation_pdf.py — 4,011 tokens (lines 1-420 (Current weak-weak mechanism-ablation PDF builder: paths, ReportRow, runtime split label lookup, prefix/terminal S_alg compile helpers, and row assembly.))
│   ├── scaffold/
│   │   ├── hh_continuation_pruning.py — 8,037 tokens (lines 1-120 (Prune configuration and policy/tolerance normalization context.), 680-1005 (Static and metric-regularized Schur surrogate ladder formulas with effective Hessian/metric solves.), 1194-1325 (Build Schur/metric-regularized prune surrogate scores, rank windows, and score telemetry.), 1679-1815 (Recoverability prune ladder remove/refit trial loop and acceptance source context.))
│   │   └── hh_continuation_scoring.py — 16,057 tokens (lines 1-150 (Scoring module imports and canonical Phase-II/Phase-III formula constants.), 741-805 (Phase-II novelty mode normalization and novelty factor helper.), 1817-2035 (Canonical Phase-II collective/pairwise novelty, projection, and trust-region raw score formulas.), 2058-2135 (phase_shortlist_records rank/shortlist metadata emission.), 2330-2465 (Phase-III plateau novelty/cost score components and cost burden structure.), 2545-2645 (Canonical Phase-III reduced-geometry score formula and emitted component fields.), 2990-3075 (Raw F metric, tangent overlap Gram, horizontal tangent, and Hessian-entry helpers.), 3220-3400 (Phase2 novelty/curvature oracle implementations that produce Gram/projection/Hessian-like telemetry.), 3535-3675 (build_full_candidate_features core output fields for novelty/curvature/window/Schur telemetry.), 3973-4055 (Batch geometry summary: reduced plane Gram, alpha, Schur solves, additivity telemetry.), 4849-5055 (Reduced-plane batch selection core scoring machinery and proposal evaluation loop.))
│   └── static_adapt/
│       ├── adapt_pipeline.py — 14,355 tokens (lines 300-335 (Imports tying adapt_pipeline to controller measurement proxy and summary reconstruction helpers.), 4880-5055 (Runtime setup for controller measurement work and scoring/candidate-cache state needed by emitted accounting fields.), 10845-10875 (Beam branch phase0 controller work recording with typed probe and common-exposure payloads.), 11300-11345 (Beam branch phase1 shortlist and gradient controller work event counters.), 12730-12815 (Beam branch Phase-II/Phase-III shortlist/rerank controller events with candidate/evaluated/retained counters.), 12890-12925 (Beam branch batch scoring controller event and common-exposure payload.), 14390-14415 (Beam branch history nfev fields including Schur guard and outer_nfev.), 14670-14715 (Beam branch history controller proxy and shortlisted/scored/retained record payloads.), 15890-15915 (Non-beam phase0 controller work event with actual/common exposure probe payloads.), 16270-16312 (Non-beam phase1 shortlist and gradient controller work event counters.), 17500-17585 (Non-beam Phase-II/Phase-III shortlist/rerank controller work events with typed probe and common exposure counts.), 17930-17960 (Non-beam batch scoring controller event and common-exposure payload.), 20080-20105 (Non-beam history nfev decomposition fields including outer_nfev.), 20535-20590 (Non-beam history controller proxy, shortlisted/scored records, retained/admitted payload fields.), 23490-23580 (Final payload summary setup and controller_measurement_work_summary construction for beam/non-beam runs.), 23780-23890 (Final adapt_vqe payload fields including history, refit metadata, nfev_total, and controller work summary.))
│       ├── batch_ordering.py — 2,680 tokens (lines 1-220 (Batch ordering config/runtime and finite-step _order_batch_records_for_admission Hamiltonian/proxy evaluation accounting.), 318-360 (Schur batch context extraction from batch geometry summary into records.))
│       ├── controller_phase_state.py — 1,005 tokens (full)
│       ├── engine_support.py — 4,321 tokens (lines 50-270 (Adapt result, beam branch state, branch scratch, and beam parent scratch dataclasses carrying local nfev/controller work state.), 727-785 (evaluate_exact_gradient_surface exact gradient scan telemetry and evaluated-count source.), 1004-1120 (Active refit window resolution and reduced objective helpers for optimizer/refit Hamiltonian evaluation accounting.))
│       ├── phase_shortlists.py — 4,539 tokens (lines 1-190 (PhaseShortlistRuntime and generic shortlist helper that records shortlist sizes and legacy hook behavior.), 231-395 (Phase1 and Phase2 lane shortlist helpers that define retained/shortlist counts feeding controller events.), 441-585 (Phase3 tie/beam selection pool helpers and positive selector filtering for Phase3 retained-count semantics.))
│       ├── runtime_split.py — 1,101 tokens (full)
│       ├── schur_warm_start.py — 4,787 tokens (lines 1-120 (Schur warm-start telemetry/config helpers and scalar Schur trust-step formula.), 190-390 (Append/batch Schur seed proposal formulas and lifted seed construction.), 500-627 (Guarded seed selection and objective-evaluation accounting for Schur warm-start guard trials.))
│       └── selector_measurement_proxy.py — 12,486 tokens (lines 1-115 (Controller work schema constants, charge basis/common-exposure identifiers, runtime dataclass, and numeric validation entry.), 373-650 (Empty summary shape, event merge logic, candidate-work ledger aggregation, and operator-probe ledger aggregation.), 697-1065 (Logical operator/common exposure probe counting and _record_controller_work_for_records event construction.), 1361-1698 (Controller summary finalization, history-row proxy extraction, and run-level controller_proxy_from_history_rows/adapt_payload helpers.))
└── test/
    ├── test_snake_table_i_measurement_work.py — 8,944 tokens (lines 1-120 (Test imports/helpers and baseline payload construction for SNAKE work tests.), 300-455 (Tests blocking candidate/group-only counters and requiring explicit candidate-work ledger semantics.), 650-860 (Tests common exposure/fair work and beam aggregate-vs-winner lineage scope.), 1000-1195 (Tests display-prefix scoped reconstruction and explicit outer_nfev accounting.), 1350-1501 (Tests runtime event ledger promotion and canonical component totals.))
    └── test_static_adapt_controller_measurement_work.py — 2,173 tokens (full)

### Codemaps
/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/
└── pipelines/
    └── scaffold/
        └── hh_continuation_types.py — 4,353 tokens (auto)


---

## Generated Plan

## Chat Send ✅
- **Chat**: `snake-work-accounting-C3E75B` | **Mode**: plan

### Response
## 1. **Summary**

Add an additive, strict mechanism-resolved SNAKE work reconstructor centered in `pipelines/exact_bench/snake_table_i_measurement_work.py`. It should reuse the existing canonical `snake_algorithmic_work_from_payload()` coarse reconstruction, then partition only typed operator-probe and Hamiltonian-evaluation work into mechanism sub-bins. Candidate counts, shortlist sizes, retained counts, and `records_evaluated` remain exposure/diagnostic fields and are never promoted directly to `S_alg`; they only provide formula/context weights for attributing already-validated phase work. This is best solved as a targeted extension, not a broad refactor, because the current strict normalizer and controller ledger already provide the correct ownership boundaries and blockers.

---

## 2. **Current-state analysis**

### Existing data/control flow

1. `pipelines/static_adapt/adapt_pipeline.py`
   - Emits controller measurement events via `_record_controller_work_for_records()`.
   - Events include:
     - `phase0_pilot_screen`
     - `phase1_append_probe` / `phase1_insertion_probe`
     - `phase2_rerank_records`
     - `phase3_reduced_geometry_rerank`
     - `batch_union_scoring`
   - Per-step history rows include:
     - `controller_measurement_work_proxy`
     - `nfev_opt`
     - `nfev_seed_probe`
     - `initial_energy_nfev`
     - `nfev_schur_warm_start_guard`
     - `outer_nfev`
     - shortlisted/scored/retained/admitted record payloads
   - Final `controller_measurement_work_summary` is reconstructed from history rows.
   - Beam runs mark aggregate summaries with `beam_run_scope = "all_expanded_scored_branches"` and preserve winner-lineage history separately.

2. `pipelines/static_adapt/selector_measurement_proxy.py`
   - Defines the controller work schema and aggregation.
   - Tracks typed actual operator-probe counts:
     - `actual_operator_probe_count`
     - `operator_probe_charge_basis = logical_estimator_request_pre_grouping_v1`
   - Tracks common-exposure counts separately.
   - Tracks exposure counters:
     - `candidate_count_total`
     - `evaluated_count_total`
     - `pre_shortlist_count_total`
     - `shortlist_size_total`
     - `retained_count_total`
     - `rejected_count_total`
   - These exposure counters are already ledgered separately and must stay non-chargeable.

3. `pipelines/exact_bench/snake_table_i_measurement_work.py`
   - Is the strict canonical normalizer.
   - Current coarse reconstruction:
     - `N_grad = phase0 + phase1 actual_operator_probe_count`
     - `N_metric = phase2 + phase3 actual_operator_probe_count`
     - `N_H_refit_eval = sum(history[*].nfev_opt) + resume/final refits`
     - `N_H_outer_eval = nfev_total - N_H_refit_eval`
   - Blocks invalid sources:
     - raw shot proxies
     - group counts
     - missing typed operator-probe fields
     - missing explicit candidate-work ledger
   - Beam terminal scope uses winner-lineage history for row `S_alg`; aggregate beam search work remains provenance as `S_beam_search_total`.

4. Formula sources for mechanism attribution:
   - `pipelines/scaffold/hh_continuation_scoring.py`
     - Phase-II novelty/Gram/projection formulas.
     - Phase-II curvature/Schur/Hessian-like telemetry.
     - Phase-III reduced-geometry scoring.
     - Batch reduced-plane Gram/Schur summaries.
   - `pipelines/scaffold/hh_continuation_pruning.py`
     - Schur prune surrogate ladders and remove/refit trials.
   - `pipelines/static_adapt/schur_warm_start.py`
     - Schur warm-start proposals and guarded objective evaluations.
   - `pipelines/static_adapt/batch_ordering.py`
     - Finite-step batch-order proxy objective evaluations.

### Blocking gaps

- Current `S_alg` only exposes coarse bins:
  - gradient
  - metric
  - Hamiltonian outer/refit
- There is no stable mechanism-resolved ledger for:
  - Gram/novelty
  - Schur/second-order/Hessian-like scoring
  - Phase-3 scoring
  - batch Schur scoring
  - warm-start/prune/batch-order Hamiltonian evals
- Existing exposure counters are correctly present but can be misread as work unless the reconstructor keeps them isolated.
- Prefix `outer_nfev` handling needs an aggregate-aware audit so aggregate `outer_nfev` is not double-counted with named subfields.

---

## 3. **Design**

### A. New public reconstructor

Add a new additive public function in:

`pipelines/exact_bench/snake_table_i_measurement_work.py`

Illustrative signature:

```py
def snake_mechanism_resolved_work_from_payload(
    source_payload: Mapping[str, Any],
    *,
    scope: str = "terminal",
    history_position: int | None = None,
    source_label: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ...
```

It must call `snake_algorithmic_work_from_payload()` first and treat that result as authoritative for coarse `S_alg`.

#### Output schema

Use schema:

```text
paper_i_hh_snake_mechanism_resolved_work_v1
```

Core shape:

```py
{
  "schema": "paper_i_hh_snake_mechanism_resolved_work_v1",
  "status": "ok | partial | blocked | invalid_reconciliation",
  "scope": "terminal | display_prefix | ...",
  "source_label": "...",

  "coarse_reconciliation": {
    "S_alg": float | None,
    "N_grad_probe": float | None,
    "N_metric_probe": float | None,
    "N_H_outer_eval": float | None,
    "N_H_refit_eval": float | None,
    "gradient_reconciled": bool,
    "metric_reconciled": bool,
    "hamiltonian_reconciled": bool,
  },

  "candidate_exposure": {
    "policy": "diagnostic_only_not_promoted_to_S_alg",
    "candidate_count_total": int,
    "evaluated_count_total": int,
    "pre_shortlist_count_total": int,
    "shortlist_size_total": int,
    "retained_count_total": int,
    "rejected_count_total": int,
  },

  "mechanism_bins": {
    "gradient": {...},
    "metric_attribution": {...},
    "hamiltonian": {...},
  },

  "event_rows": [...]
}
```

### B. Mechanism bin policy

#### Gradient bins

Only typed actual operator-probe counts may contribute.

| Source phase/event | Bin |
|---|---|
| `phase0_pilot_screen` | `gradient.phase0_pilot_screen` |
| `phase1_append_probe` | `gradient.phase1_append_probe` |
| `phase1_insertion_probe` | `gradient.phase1_insertion_probe` |
| unknown phase0/phase1 typed count | `gradient.unclassified_gradient` |

The sum must equal current `S_alg_N_grad_probe`.

#### Metric attribution bins

Metric events are already true chargeable operator-probe work, but Phase-II probes can serve multiple mechanisms. Therefore sub-bins are **mutually exclusive attribution partitions** of typed metric work, not additional work.

| Source event | Target bin |
|---|---|
| `phase2_rerank_records` | Split between `phase2_gram_novelty` and `phase2_schur_second_order` by formula units |
| `phase3_reduced_geometry_rerank` | `phase3_reduced_geometry_scoring` |
| `batch_union_scoring` | `batch_schur_union_scoring` |
| unknown phase2/phase3 typed count | `metric_unclassified` |

Formula-unit attribution for `phase2_rerank_records`:

```py
phase2_gram_novelty_units =
    sum(1 + len(record.phase2_geometry_window_indices))

phase2_schur_second_order_units =
    sum(1 + len(record.schur_window_indices))
    for records with second-order telemetry present
```

Allocation:

```py
allocated_bin = actual_operator_probe_count * bin_units / total_known_units
```

If units are unavailable, place typed count into `metric_unclassified` and mark `status = "partial"`.

The sum must equal current `S_alg_N_metric_probe`.

#### Hamiltonian/refit bins

Use explicit nfev fields only.

Outer bins:

- `H_outer.initial_energy`
- `H_outer.seed_probe`
- `H_outer.schur_warm_start_guard`
- `H_outer.schur_prune_warm_start_guard`
- `H_outer.batch_order_finite_step_proxy`
- `H_outer.prune_recoverability_trial`
- `H_outer.unclassified_outer`

Refit bins:

- `H_refit.optimizer_history_nfev`
- `H_refit.resume_boundary_refit`
- `H_refit.final_full_refit`

Add a new aggregate-aware row helper:

```py
_history_outer_eval_breakdown(row) -> {
  "outer_total": float | None,
  "named_components": {...},
  "unclassified_outer": float,
  "status": "ok | missing | ambiguous_outer_nfev | invalid_outer_nfev"
}
```

Rule:

- If `outer_nfev` exists, treat it as the aggregate total.
- Named fields are a breakdown of that aggregate, not additive with it.
- If named sum exceeds `outer_nfev` beyond tolerance, mark ambiguous/invalid.
- If `outer_nfev` is absent, use sum of named fields.
- Never infer Hamiltonian work from candidate counts.

### C. Event extraction

For each scoped history row:

1. Prefer explicit `events` from `controller_measurement_work_proxy`.
2. Else parse `by_scope` keys, because `_default_controller_work_scope()` encodes:

```text
static_adapt|phase=phase2|event=phase2_rerank_records|depth=...
```

3. Else fall back to `by_phase`, but mark the mechanism split partial.

The reconstructor must only charge an event if it contains valid typed operator-probe fields. Candidate counters remain exposure.

### D. Scoped history behavior

Use the same scope semantics as current `snake_algorithmic_work_from_payload()`:

- `terminal`, non-beam: all history rows.
- `display_prefix`: first `history_position` rows only.
- Beam terminal: winner-lineage history rows only.
- Beam aggregate summary:
  - keep `S_beam_search_total` as provenance
  - do not promote aggregate beam search total into row `S_alg`

### E. Optional future trace emitted by `adapt_pipeline`

Additive history-row field:

```py
"mechanism_work_trace": {
  "schema": "paper_i_hh_snake_mechanism_work_trace_v1",
  "events": {
    "phase2:phase2_rerank_records": {
      "formula_units": {
        "phase2_gram_novelty": float,
        "phase2_schur_second_order": float
      },
      "record_count": int,
      "window_summary": {...}
    },
    "phase3:phase3_reduced_geometry_rerank": {
      "formula_units": {
        "phase3_reduced_geometry_scoring": float
      }
    },
    "phase3:batch_union_scoring": {
      "formula_units": {
        "batch_schur_union_scoring": float
      }
    }
  }
}
```

The reconstructor should prefer this trace when present, then fall back to deriving units from existing record payloads.

---

## 4. **File-by-file impact**

### `pipelines/exact_bench/snake_table_i_measurement_work.py`

Add:

- Schema constants:
  - `SNAKE_MECHANISM_WORK_SCHEMA_VERSION`
  - `SNAKE_MECHANISM_TRACE_SCHEMA_VERSION`
- Public function:
  - `snake_mechanism_resolved_work_from_payload()`
- Internal helpers:
  - scoped history row resolver
  - controller event extractor
  - work-scope parser
  - candidate exposure extractor
  - formula-unit extractor
  - outer nfev breakdown helper
  - reconciliation helper

Modify:

- `_explicit_prefix_outer_nfev()` should delegate to aggregate-aware outer breakdown logic to prevent aggregate/component double counting.

Why:

- Keeps the strict canonical accounting path centralized.
- Reuses existing blockers and coarse reconstruction.

### `pipelines/static_adapt/selector_measurement_proxy.py`

Modify aggregation only if validation shows `by_scope` summaries do not preserve enough metadata.

Required preserved fields in by-scope summaries:

- `phase`
- `event_kind`
- `probe_role`
- `operator_probe_charge_basis`
- `actual_operator_probe_count`
- `common_exposure_operator_probe_count`
- candidate exposure counters

Do **not** change candidate counters into work counters.

### `pipelines/static_adapt/adapt_pipeline.py`

Add optional per-history-row `mechanism_work_trace`.

Also ensure future history rows include explicit nfev fields when available:

- `nfev_batch_order_finite_step_proxy`
- `nfev_prune_recoverability_trial`
- `nfev_schur_prune_warm_start_guard`

Only include these as chargeable if they are already included in `nfev_total` / `outer_nfev`.

### `pipelines/reporting/build_paper_i_hh_weak_weak_snake_mechanism_ablation_pdf.py`

Modify:

- Import `snake_mechanism_resolved_work_from_payload`.
- For plateau and terminal rows, build mechanism work sidecar data.
- Add compact status/subbin fields to `ReportRow` only if needed for the PDF.
- Prefer a JSON sidecar for full event rows to avoid widening the PDF table.

Suggested sidecar path:

```text
output/pdf/paper_i_hh_weak_weak_snake_mechanism_ablation_20260708/
paper_i_hh_weak_weak_snake_mechanism_ablation_20260708_mechanism_work.json
```

### `test/test_snake_table_i_measurement_work.py`

Add tests for:

- Candidate counters not promoted to `S_alg`.
- Mechanism bins reconcile to current coarse bins.
- Phase-II metric split using formula units.
- Missing formula units falling back to `metric_unclassified`.
- Prefix scope uses only prefix history rows.
- Beam terminal uses winner lineage and keeps aggregate beam search provenance separate.
- Aggregate/component `outer_nfev` is not double-counted.
- Invalid typed operator-probe fields block mechanism work.

### `test/test_static_adapt_controller_measurement_work.py`

Add/extend tests for:

- `by_scope` summaries preserve event identity.
- Candidate ledger counters remain diagnostic.
- Common-exposure counts remain separate from actual chargeable operator probes.

---

## 5. **Risks and migration**

- Historical payloads may not have enough event-level detail for a full mechanism split. Return `status = "partial"` with unclassified bins rather than inferring from candidate counts.
- Updating `_explicit_prefix_outer_nfev()` may change prefix work for rows that currently contain both aggregate `outer_nfev` and named subfields. Treat this as a bug fix and cover it with explicit tests.
- New payload fields are additive. Old code can ignore them. Old artifacts require no migration.
- The v2 CHTC mechanism-ablation outputs can be reprocessed with partial mechanism ledgers; reruns are only needed if full formula-unit traces are required.

---

## 6. **Implementation order**

1. Add tests for aggregate-aware `outer_nfev` behavior.
2. Implement `_history_outer_eval_breakdown()` and route `_explicit_prefix_outer_nfev()` through it.
3. Add mechanism schema constants and public `snake_mechanism_resolved_work_from_payload()`.
4. Implement scoped history/event extraction.
5. Implement candidate exposure aggregation as diagnostic-only.
6. Implement gradient, metric, and Hamiltonian mechanism bin attribution.
7. Add reconciliation checks against `snake_algorithmic_work_from_payload()`.
8. Extend `selector_measurement_proxy.py` only if by-scope event metadata is not preserved.
9. Add optional `mechanism_work_trace` emission in `adapt_pipeline.py`.
10. Update reporting builder to emit the mechanism sidecar.
11. Run/extend the selected test suites:
    - `test/test_snake_table_i_measurement_work.py`
    - `test/test_static_adapt_controller_measurement_work.py`


> 💡 Continue this plan conversation with ask_oracle(chat_id: "snake-work-accounting-C3E75B", new_chat: false)