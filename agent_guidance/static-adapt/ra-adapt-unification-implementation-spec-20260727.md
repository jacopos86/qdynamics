# RA-ADAPT Unification Implementation Specification

> Correction, 2026-07-29: raw unreduced `full` insertion is retired. The
> executable always-open policy is `AlwaysCommutationReducedInsertion`
> (`always_commutation_reduced`) and the runtime mode is
> `full_commutation_reduced`; every open domain uses the shared exact
> commutation reducer. The former capped-domain CLI spelling `always` is also
> retired because it does not denote the complete logical position domain.
> Historical raw-mode statements below are superseded.

Date: 2026-07-27
Author role: Claude Fable, source-read-only architecture planner.
Decision authority: `agent_guidance/static-adapt/ra-adapt-unification-refactor-decisions-20260727.md` (the "decision record").
Scientific alignment source: `MATH/paper_facing/paper_I_static_scaffold/paper_i_macro_singleton_protocol_alignment_20260727.md` (the "alignment note").

This document is the implementation specification for the Paper-I RA-ADAPT
unification, repair, and legacy contraction. It authorizes no runs, no
manuscript edits, no evidence replacement, no deletions, and no commits by
itself. The implementing agent (GPT-5.6 ultra) executes it only after the
original audit agent's review.

Every factual claim below is labeled by class:

- **[FACT]** confirmed in current source at the cited `file:line`;
- **[LOCKED]** a user decision recorded in the decision record or alignment
  note — do not reopen;
- **[PROPOSED]** an architecture/implementation choice this spec makes —
  reviewable, but routine;
- **[VERIFY]** an objective check the implementer must run before relying on
  the statement.

Line numbers were read on 2026-07-27 in the working tree of
`/Users/jakestrobel/local_repos/Holstein_test_fullclone_3` (branch
`codex/sr-snake-v3-foundation-20260716`, dirty). Treat them as anchors, not
guarantees; re-resolve by symbol name if the file has moved.

---

## 1. Current topology and defects

### 1.1 Shared numerical substrate (all five routes touch it)

**[FACT]** The compatibility monolith is
`pipelines/static_adapt/adapt_pipeline.py` (~79,354 lines). Its core
`_run_hardcoded_adapt_vqe` is one function from line 13126 to ~69026. The
`pipelines/hardcoded/adapt_pipeline.py` file is a module-forwarding alias to it
(untracked, dirty; see §5.11).

**[FACT]** Candidate scoring, Phase-1/2/3 score payloads, joint-geometry
receipts, novelty oracles, and batch selection live in
`pipelines/scaffold/hh_continuation_scoring.py` (~18,808 lines).

**[FACT]** Shared numerical modules already extracted from the monolith:

| Concern | Module | Key symbols |
|---|---|---|
| Phase-III solves + support factorization | `pipelines/static_adapt/joint_linear_solve.py` | `factor_supported_metric` (l.389), `_solve_supported_metric_whitened` (l.1427), `_solve_supported_metric_projected_generalized_trust` (l.1102), `solve_joint_linear_model` (l.3314); policy ids `supported_metric_whitened_eigh_v1` (l.15), `supported_metric_projected_generalized_trust_v1` (l.21) |
| Trust transactions | `pipelines/static_adapt/route_a_trust_region.py` | `_sr_projected_source_metric_trust_transaction` (l.955, source-Gram, no endpoint overlap), `update_trust_region_state` (l.1192, overlap-calibrated `displacement_calibrated_unbounded_v2`), `exact_fubini_study_distance` (l.80) |
| Accepted refit | `pipelines/static_adapt/accepted_refit.py` | scope `full_ansatz_v1`, chart `supported_fs_whitened_fixed_v1`; one chart per invocation (module docstring); imports `build_compiled_exact_manifold_adapter` from `formal_manifold_exact_backend` (l.21-23) |
| Exact manifold geometry | `pipelines/static_adapt/formal_manifold_exact_backend.py` | `CompiledExactManifoldAdapter` (l.535), `build_compiled_exact_manifold_adapter` (l.953) — FM-named but consumed by active neutral code |
| Estimator ledger | `pipelines/static_adapt/estimator_call_ledger.py` | `S_alg` components `N_H_outer`, `N_H_refit`, `N_grad`, `N_metric` (l.40-43) |
| Pools and problems | `pipelines/static_adapt/builders/{problem_registry,problem_setup,pool_resolution,hh_pool_presets,child_pool_expansion,shared_pauli_pool_contract}.py`, `pipelines/contracts/problem.py` | `ResolvedProblemContext`, `full_meta` resolution, `build_shared_pauli_child_pool`, `expand_snake_pool_with_global_child_sets` (child_pool_expansion l.182) |
| Lanes/shortlists | `pipelines/static_adapt/lane_routes.py`, `phase_shortlists.py` | `STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE` (retained), `STATIC_LANE_ROUTE_ALGEBRAIC` (retired family, §5.13) |
| Commutation math + algebraic lanes | `pipelines/static_adapt/algebraic_metadata.py` | retained: `pauli_words_commute` (l.207), `exact_expansions_commute` (l.402), `support_qubits_from_pauli_word` (l.193); retired: `LANES_PHASE1 = (LANE_FLAT, LANE_CURV, LANE_DISJ, LANE_MIX)` (l.19-20) |
| Child symmetry/padding | `pipelines/static_adapt/route_a_child_padding.py`; monolith `_historical_singleton_child_padding_contract` (adapt_pipeline l.10519) | hard fixed-sector + binary-padding guard |
| Route profiles | `pipelines/static_adapt/sr_snake_route_profile.py` (4,266 l.) | canonical contract builder `canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract` (l.1809) and `_sha256` (l.1861) |
| Run summary | `pipelines/reporting/paper_i_run_summary.py` | `summarize_paper_i_run`, plateau/common-accuracy/`S_alg`/Qiskit contracts |

### 1.2 Singleton RA-ADAPT (current canonical SR-SNAKE) — call map

**[FACT]** Public seam: `pipelines/static_adapt/sr_snake/__init__.py` exports
`run_sr_snake` plus the typed receipt vocabulary from
`sr_snake/contracts.py` (1,674 l.).

```text
run_sr_snake(problem, request)                sr_snake/runner.py:1810
  -> _execute_sr_snake                        runner.py (same file)
  -> _resolve_execution_context               sr_snake/_context.py:613
       HH L=2 lock                            _context.py:623-631
       route contract + digest                _context.py:648-653 -> sr_snake_route_profile.py
       runtime kwargs                         adapt_pipeline._build_canonical_sr_snake_runtime_kwargs (l.78205)
       runtime factory                        adapt_pipeline._build_default_sr_controller_numerical_runtime (l.78366)
       legacy fallback executor               adapt_pipeline._run_hardcoded_adapt_vqe_compatibility (l.69026)
  -> _run_default_singleton_controller        sr_snake/_controller.py
       selection kernel                       sr_snake/_selection.py  (typed wrappers)
       transition kernel                      sr_snake/_transition.py ("numerical kernel remains in adapt_pipeline", docstring l.3)
       numerical kernels                      adapt_pipeline._DefaultNoPrune* classes (l.2432-7275, 69087-77504)
  -> SRRunResult + summarize_paper_i_run      runner.py:1764-1803
```

Route-scientific behavior of this path (all **[FACT]**):

- **Pool/eligibility**: unfiltered `full_meta` parents (123 at `nph=3`, 171 at
  `nph=7` — counts are runtime facts recorded in the audit; verify via the
  pool-hash fixture in §4). Termwise-sector-violating parents are *deferred*,
  not removed, when the projected-child route is active:
  `_resolve_parent_sector_filter_policy` (adapt_pipeline l.10474) returns
  `deferred_execution_indices` under
  `runtime_split_selection_mode in {global_child_only_v1, archival_child_set_forward_v1}`
  with `hard_guard`, keeping those parents as child templates.
- **Phase I/II**: parent records in physical operator lanes
  (`lane_routes.py`); Phase-1 score `phase1_score_payload`
  (hh_continuation_scoring l.2274) in `trust_region_v1` mode with the
  hardware-cost factor applied (see §1.7 D5).
- **Child exposure (staged)**: children are constructed only from retained
  parents via `phase3_runtime_split_mode=shortlist_pauli_children_v1`,
  `archival_child_set_forward_v1`, exact-cardinality-one canonical unit-Pauli
  children, hard symmetry guard and padding projection
  (`route_a_child_padding.py`, `_historical_singleton_child_padding_contract`
  adapt_pipeline l.10519).
- **Phase III**: full active-plus-singleton response; fresh joint-geometry
  receipts carry `exact_ordered_insertion_zero_angle_v1`
  (`_promote_fresh_phase3_joint_geometry_receipt`, hh_continuation_scoring
  l.6149, chart stamped l.6056/6249); solve is the projected generalized
  raw-Gram trust solve `_solve_supported_metric_projected_generalized_trust`
  (joint_linear_solve l.1102) — no Gram inverse square root, no metric ridge,
  whitening recorded false.
- **Trust**: `_sr_projected_source_metric_trust_transaction`
  (route_a_trust_region l.955): reuses the certified raw-Gram support, no
  endpoint-overlap acquisition.
- **Accepted refit**: complete enlarged ansatz, fixed supported-FS-whitened
  chart per invocation (`accepted_refit.py`), Powell over
  `expanded_runtime_projected_logical_v1` base chart.
- **Accounting**: closed occurrence receipt
  `S_alg = N_H_outer + N_H_refit + N_grad + N_metric`
  (estimator_call_ledger l.40-43).
- **Route identity**: unqualified canonical Paper-I baseline is
  `sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1` →
  `supported_projected_generalized_source_metric_no_overlap_trust_full_response_symmetric_cost_no_prune_v1`,
  digest `fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2`
  (pinned in `test/test_static_adapt_sr_snake_facade.py` and six other tests).

Preflight finding 1 is **confirmed**.

### 1.3 Macro RA-ADAPT — call map

**[FACT]** There is no typed macro facade. Macro RA-ADAPT executes only inside
the monolith through the CLI (`cli_config.py`, 4,854 l.) or through
`pipelines/static_adapt/paper_i_runner.py` (1,087 l., "translates … into the
compatibility runner", docstring l.1-7).

```text
main()/paper_i_runner
  -> cli_config._resolve_main_cli_configs / _build_run_hardcoded_adapt_vqe_kwargs
  -> adapt_pipeline._run_hardcoded_adapt_vqe (l.13126)
       pool: full_meta -> _resolve_parent_sector_filter_policy (l.10474)
             macro admission (no projected-child guard) => removed_indices
             = the 21/23 termwise-sector-violating parents  [102/148 remain]
       Phase 0/1/2: macro records; RouteAJointResponseEvaluator scope="macro_phase2"
             (adapt_pipeline l.38241-38294; route_a_schur_selector.py:404)
       insertion positions: _phase1_position_probe_plan (l.11394),
             _candidate_insertion_position_plans (l.11551),
             _commutation_reduced_insertion_position_plan (l.11491),
             plateau trigger _insertion_commutation_plateau_round_policy (l.11327)
       Phase 2/3 joint response: build_full_candidate_features
             (hh_continuation_scoring l.6399) emits phase2_joint_geometry_reuse
             with coordinate_chart = append_candidate_after_current_ansatz_v1
             (hh_continuation_scoring l.6500)                      <-- DEFECT D1
       selector solve: supported_metric_whitened_eigh_v1 with 1e-9 metric
             ridge (whitened-eigh path in joint_linear_solve)      <-- DEFECT D3
       trust: update_trust_region_state (route_a_trust_region l.1192),
             displacement_calibrated_unbounded_v2, endpoint FS overlap
             (exact_fubini_study_distance l.80)                    <-- DEFECT D2
       splice/refit: _splice_candidate_at_position (l.11670),
             _splice_logical_candidate_at_position (l.11753),
             _predict_phase3_geometry_window_for_position (l.11821)
       accepted refit / accounting / payloads: same shared modules as §1.2
```

**[FACT]** First-order position probes are position-aware (interior positions
are enumerated when the insertion domain is open: modes `append_only`,
`insertion_commutation_plateau_v1`, `full`/`full_commutation_reduced`,
`always` — `_phase1_position_probe_plan` l.11394-11447). The *reused*
Phase-II/III joint-response geometry is what carries the append chart.
Preflight finding 2 is **confirmed**: position-aware first-order geometry,
append-chart joint response, overlap-calibrated trust.

**[FACT]** `build_full_candidate_features` accepts
`emit_fresh_phase3_joint_geometry_receipt=True` which promotes the payload to
a fresh receipt stamped `exact_ordered_insertion_zero_angle_v1`
(hh_continuation_scoring l.6544-6560, l.6149). The exact-chart machinery
therefore already exists and is exercised by the singleton route; the repair
routes the macro representation through it at every recorded position.

### 1.4 Macro Append-ADAPT — call map

**[FACT]** Lives in the generic comparator monolith
`pipelines/exact_bench/generic_static_adapt_variants.py` (11,171 l.), driven
by `pipelines/exact_bench/generic_static_benchmark.py` (3,115 l.) and
`table_i_static_benchmark.py`.

```text
generic_static_benchmark -> _get_config(STATIC_FULL_META_APPEND_ADAPT_VQE)
  algorithm id "static_full_meta_append_adapt_vqe"   (generic_static_adapt_variants l.199)
  pool: unfiltered full_meta parents (123/171) — no termwise-sector removal
  selection: conventional append gradient ranking; append position only
  optimizer/estimator/compile: file-local closures + _PaperIComparatorEstimatorLedger (l.422)
```

**[FACT]** Historical macro Append therefore evaluated 123/171 parents while
historical macro RA evaluated 102/148 — preflight finding 3 **confirmed**
(mechanism: the `_resolve_parent_sector_filter_policy` branch, §1.3).
**[LOCKED]** The replacement macro comparison uses the same 102/148 executable
macro pool for both methods. The preflight's `grouped_exact` alternative is
**not adopted** (`_enforce_hh_component_risk_grouped_execution`, adapt_pipeline
l.12107, stays unused for this purpose).

### 1.5 Singleton Append-ADAPT — call map

**[FACT]** Same comparator monolith. The single-Pauli-word condition uses the
shared-Pauli child pool: `_expand_pool_with_shared_pauli_children`
(generic_static_adapt_variants l.2917) →
`builders/shared_pauli_pool_contract.build_shared_pauli_child_pool`, with
`projected_singleton_children_only_v1` validation (l.7266-7275) and the
canonical unit-Pauli representative convention (coefficient absorbed into the
variational coordinate — alignment note "Single-Pauli-word representative",
**[LOCKED]** for all replacement runs). Ancestry is the same 123/171 parent
source; the global eligible child pool is scanned each iteration. Preflight
finding 4 **confirmed**: divergence from singleton RA begins at child
exposure (global pool vs staged from retained parents).

### 1.6 Retained optional JR-SNAKE — call map

**[FACT]** JR-SNAKE (`joint_response_snake`) anchors:

```text
route_a_funnel.py
  ROUTE_A_FUNNEL_CHILD_12_JOINT_RESPONSE_V2 = "child_12_joint_response_v2" (l.34)
  run_route_a_child_funnel (l.353): "direct or hierarchical child stages
    over one global population"; dedup -> padding filter -> sector filter
    -> staged P1/P2/P3 evaluators
route_a_schur_selector.py
  RouteAJointResponseEvaluator (l.404), select_route_a_schur_proposals (l.455),
  TrustRegionUpdateConfig (l.100)
joint_step_warm_start.py, batch_ordering.py — JR batch admission support
```

**[LOCKED]** JR-SNAKE is retained as an optional extension with its funnel and
optional joint-response batching; it is not an ordinary default and not an
archival target. **[VERIFY]** Before refactoring anything the funnel imports,
run `test/test_static_adapt_canonical_beam_and_subsets.py`,
`test/test_static_adapt_route_a_shortlists.py`, and
`test/test_build_paper_i_hh_joint_response_six_regime_overlay.py`, and record
the observed macro-to-child P1/P2/P3 funnel semantics in the lock-stage
receipt. Note the JR funnel operates on a global child population handed to
it; macro shortlisting occurs in the caller. Do not restate its semantics
from memory.

### 1.7 Confirmed defects versus deliberate differences

Defects being repaired (all cross-checked against the alignment note):

| id | Defect | Current-code anchor | Repair |
|---|---|---|---|
| D1 | Macro Phase-II/III joint-response geometry reuses the append-coordinate chart `append_candidate_after_current_ansatz_v1` while candidate-position records may be interior | hh_continuation_scoring l.6500 | All replacement insertion trajectories use `exact_ordered_insertion_zero_angle_v1` at every recorded position; append-only records arise as the `p = n_k` special case (**[LOCKED]**) |
| D2 | Macro trust update is endpoint-overlap-calibrated (`displacement_calibrated_unbounded_v2`) while singleton uses source-Gram no-overlap | route_a_trust_region l.1192 vs l.955 | One source-Gram no-endpoint-overlap trust transaction for both adapters (**[LOCKED]**) |
| D3 | Macro selector solves in the whitened-eigh chart with a 1e-9 metric ridge; singleton solves the projected generalized raw-Gram model | joint_linear_solve l.1427 vs l.1102 | One retained-support factorization + projected generalized Phase-III solve for both (**[LOCKED]**); historical regularized selector whitening stays outside the canonical engine |
| D4 | Macro RA pool 102/148 vs macro Append pool 123/171 | §1.3/§1.4 | Common 102/148 executable macro pool for the replacement macro comparison (**[LOCKED]**) |
| D5 | Historical displayed results used Phase-I energy-only cost scope (`S1 = ΔE1`); current code applies the hardware-cost factor in Phase I too (`phase1_score_payload` multiplies `cost_factor / burden`, hh_continuation_scoring l.2286-2297) | hh_continuation_scoring l.2274-2319 | Two typed source-locked bundle policies: `late_resource_weighting_v1` (Phase I energy-only) and `all_phase_resource_weighting_v1`; Study 2 compares them (**[LOCKED]**) |

Deliberate representation differences (not defects, must be preserved):

- staged child exposure (singleton RA) versus global child pool (singleton
  Append) — intended method difference beginning at child exposure;
- macro (undecomposed) versus single-Pauli-word representation itself;
- canonical unit-Pauli representative for every single-Pauli-word candidate;
- RA's staged P1/P2/P3 funnel versus Append's conventional selector.

Lock-stage provenance checks (objective, no user judgment):

- **[VERIFY]** From macro candidate telemetry of the displayed macro rows
  (`insertion_position_commutation_reduced`, position ids in
  `phase2_joint_geometry_reuse`), resolve whether any accepted or scored
  record had `p < n_k`. Interior `p` + append chart is the material
  mismatch. Record the answer in the lock receipt. The repair does not depend
  on the answer; the manuscript framing does (user-owned, out of scope here).
- **[VERIFY]** Identify the excluded 21/23 macro parents by label from
  `_resolve_parent_sector_filter_policy` output on the resolved `nph=3` and
  `nph=7` pools, and serialize them into the pool-hash fixture (§4).
- **[VERIFY]** Confirm from displayed receipts that macro and singleton used
  the same complete accepted-refit whitening policy (alignment note question
  3). Anchor: `adapt_accepted_refit_*` fields in preserved manifests.

---

## 2. Target deep-module design

### 2.1 Package layout **[PROPOSED]**

One new package owns the canonical engine; adapters and facades are thin.

```text
pipelines/static_adapt/ra_adapt/
  __init__.py          # exports: run_ra_adapt, run_append_adapt, typed contracts
  contracts.py         # RAAdaptRequest, resolved-protocol + receipt types
  engine.py            # the deep engine (Phase I/II/III orchestration)
  adapters.py          # MacroCandidateAdapter, SinglePauliWordCandidateAdapter
  pools.py             # parent-template inventory vs executable-macro inventory
  insertion_geometry.py# actual-position candidate geometry (exact ordered chart)
  support.py           # neutral retained-support factorization (one owner)
  trust.py             # source-Gram no-overlap trust transaction
  append.py            # run_append_adapt facade + conventional append selector
  bundles.py           # typed run-bundle schema + materialization (§7)
```

Facades:

```python
result = run_ra_adapt(problem, request=None)      # ra_adapt/__init__.py
result = run_append_adapt(problem, request=None)  # ra_adapt/append.py
```

Rules **[LOCKED]** enforced by this layout:

- no flag union, callback framework, paper-number switch, or second defaults
  registry — representation is selected by a typed adapter object inside the
  resolved request, never by a string route soup;
- Append shares pools, adapters, candidate execution, compilation, accounting,
  and the accepted-refit implementation, but its conventional selector lives
  in `append.py` and never enters the RA P1/P2/P3 funnel;
- Paper IV stays out of scope: `run_ra_adapt` keeps the Hubbard–Holstein
  `L=2` lock exactly where `run_sr_snake` has it today (moved verbatim from
  `sr_snake/_context.py:623-631`); the problem-neutral seam is not blocked
  (the lock is one validation at the facade boundary) but nothing
  molecular-vibronic is activated;
- the stationary-source / measured-residual and cost-scope policies are typed
  fields of the resolved protocol reachable **only** through
  `bundles.py`-materialized source-locked protocols — no public CLI seam
  (**[LOCKED]** preservation boundary).

### 2.2 Ownership map **[PROPOSED]**

| Responsibility | Owner | Built from (current source) |
|---|---|---|
| Phase-I/II/III orchestration, controller rounds, parent retention, representation-adapter transitions | `ra_adapt/engine.py` | typed controller skeleton from `sr_snake/_controller.py`/`_selection.py`/`_transition.py`; numerical kernels extracted from `adapt_pipeline._DefaultNoPrune*` |
| Candidate-position enumeration + actual-position geometry | `ra_adapt/insertion_geometry.py` | `_phase1_position_probe_plan`, `_commutation_reduced_insertion_position_plan`, `_candidate_insertion_position_plans`, `_splice_*`, `_predict_*_window_for_position` (adapt_pipeline l.11327-12049); fresh-receipt promotion path of `hh_continuation_scoring` (l.6149) |
| Parent template pool (123/171) vs executable macro pool (102/148); pool hashes | `ra_adapt/pools.py` | `builders/pool_resolution.py`, `hh_pool_presets.py`, `_resolve_parent_sector_filter_policy`, `_hh_termwise_component_risk_gate` |
| Fubini–Study retained-support factorization (thresholds, eigenpairs, tolerances, receipts; exposes the coordinate view each stage requires) | `ra_adapt/support.py` (single owner) | `joint_linear_solve.factor_supported_metric` + supported-projection receipt code (l.364-546); dedupes the support/ridge/tolerance logic currently scattered per preflight finding 5 |
| Phase-III solve | `joint_linear_solve.py` (retained in place), called through `engine.py` with policy fixed to `supported_metric_projected_generalized_trust_v1` | l.1102 |
| Source-metric trust transaction (both adapters, no endpoint overlap) | `ra_adapt/trust.py` | `_sr_projected_source_metric_trust_transaction` (route_a_trust_region l.955); the overlap-calibrated updaters stay behind in `route_a_trust_region.py` as compatibility identities |
| Curvature stabilization vs trust boundary | Phase-III receipt contract in `contracts.py`: separate `kappa_stabilization_shift` (κ) and `trust_boundary_multiplier_lambda` (λ) fields for the solved system `(H_s + (κ+λ) Λ_s) q = g_s`, where κ is the curvature-stabilization floor (`curvature_shift`, joint_linear_solve l.1250) and λ = μ − κ ≥ 0 is the trust-boundary increment above it (μ is the total multiplier currently reported as `trust_lambda`, l.1383). Trust-boundary activity is derived from λ > 0, never from μ (μ > 0 alone can mean curvature-only stabilization); single combined field forbidden — alignment note §3 | joint_linear_solve solver internals |
| Accepted refit + whitening | `pipelines/static_adapt/accepted_refit.py` retained as-is (already a deep module); exact-geometry dependency re-homed (§5.1) | — |
| Estimator/resource accounting | `estimator_call_ledger.py` retained; engine emits occurrence receipts; stationary-source protocol suppresses active-gradient occurrences (§4 T10) | l.40-79 |
| Typed protocol / execution manifest / result / bundle | `ra_adapt/contracts.py` + `ra_adapt/bundles.py` | modeled on `sr_snake/contracts.py` receipt style |
| Append conventional selector | `ra_adapt/append.py` | conventional gradient-ranking logic extracted from `generic_static_adapt_variants.py` for the retained full-meta append comparison only |

### 2.3 Adapter contract **[PROPOSED]**

```python
class CandidateRepresentationAdapter(Protocol):
    def parent_inventory(self, problem) -> ParentTemplateInventory      # 123/171, hashed
    def executable_pool(self, problem) -> ExecutableCandidatePool       # macro: 102/148; single-Pauli-word: guarded children factory
    def expose_children(self, retained_parents) -> CandidateRecords     # macro: identity (parents advance intact)
                                                                        # single-Pauli-word: split -> symmetry guard -> padding -> canonicalize -> dedupe
    def candidate_geometry(self, record, position) -> InsertionGeometryRequest  # always exact ordered chart
```

`MacroCandidateAdapter.expose_children` is the identity on retained parents
(macro condition of alignment note §4.3). Both adapters consume the same
123/171 parent inventory; the macro adapter's *executable* pool is the
102/148 subset, and the RA-vs-Append macro comparison draws both methods from
that same subset (**[LOCKED]**; note the upstream parent supply control from
alignment note §4.3 is thereby satisfied for the macro–macro comparison,
while the representation comparison controls the parent supply at 123/171).

### 2.4 What is intentionally not built

- no six-regime campaign launcher, no generic cross-paper runner;
- no public config seam for gradient policy or cost scope;
- no Paper-IV pools/physics/validation;
- no replacement of the JR-SNAKE funnel or its batching semantics;
- no new whitened selector path in the engine (historical
  `supported_metric_whitened_eigh_v1` remains a compatibility identity
  outside the canonical engine);
- no reuse of `run_sr_snake` internals by new callers: `run_sr_snake` becomes
  a compatibility alias delegating to the engine (§3 step 5) and is retired
  from guidance only at step 7.

---

## 3. Migration sequence (strangler; no big-bang rewrite)

Each step lists files, tests/evidence, rollback boundary, and likely failure
modes. A step is complete only when its evidence exists as a committed test
or receipt. Steps 1-2 are the `lock` ICM stage; 3-6 `refactor`; 7-8
`verify`; 9 closes contraction; 10 is `materialize bundles`.

### Step 1 — Lock characterization and mathematical contracts

- Add characterization fixtures (no production-code changes):
  - `test/test_ra_adapt_lock_pool_inventory.py`: serialize parent labels +
    sha256 of the resolved `full_meta` parent inventory at `nph=3` and
    `nph=7`; assert counts 123/171; serialize the
    `_resolve_parent_sector_filter_policy` removed set; assert 21/23 and
    resulting 102/148; write `test/fixtures/ra_adapt_pool_inventory_{3,7}.json`.
  - `test/test_ra_adapt_lock_singleton_trajectory.py`: short-horizon
    (3-4 round) deterministic `run_sr_snake` runs at `nph=3` under the
    canonical no-overlap-trust profile; freeze operator sequence, insertion
    positions, energies, trust receipts, `S_alg` components into fixtures.
  - `test/test_ra_adapt_lock_macro_receipts.py`: drive
    `build_full_candidate_features` both with reuse and with
    `emit_fresh_phase3_joint_geometry_receipt=True`; assert the chart ids of
    each path (`append_candidate_after_current_ansatz_v1` vs
    `exact_ordered_insertion_zero_angle_v1`) so the defect and its repair
    lever are pinned before movement.
  - Run the three provenance **[VERIFY]** checks of §1.7 against the
    displayed-row artifacts; write the results into
    `agent_guidance/static-adapt/ra-adapt-lock-receipt-20260727.json`
    (new file, receipt only).
- Existing tests that must pass untouched as part of the lock baseline:
  `test_static_adapt_sr_snake_facade.py`,
  `test_static_adapt_sr_snake_no_prune_trajectory.py`,
  `test_static_adapt_commutation_reduced_insertion.py`,
  `test_static_adapt_insertion_commutation_plateau.py`,
  `test_static_adapt_projected_generalized_trust_solve.py`,
  `test_static_adapt_joint_linear_solve.py`,
  `test_static_adapt_accepted_refit.py`,
  `test_static_adapt_route_a_trust_region.py`,
  `test_paper_i_run_summary.py`, `test_paper_i_s_alg_accounting.py`,
  `test_generic_static_adapt_variants.py`.
- Rollback: fixtures are additive; delete them.
- Failure modes: nondeterminism in fixture runs (pin seeds/threads via
  `adapt_worker_limits.py`); dirty-tree drift between fixture creation and
  later parity checks (record the tree state in the receipt).

### Step 2 — GitNexus strict ignore + index-only pilot

See §6. Deliverables: `.gitnexusignore` (new), index run, and the answers to
the five reachability questions written into the lock receipt. No `gitnexus
setup`, no generated agent files. Rollback: delete the ignore file and index
artifacts. Failure mode: index accidentally ingesting artifacts — the ignore
file is validated first by listing the index candidate set (§6).

### Step 3 — Extract neutral shared kernels

- `pipelines/static_adapt/exact_geometry_backend.py` (new): move
  `CompiledExactManifoldAdapter` + `build_compiled_exact_manifold_adapter`
  out of `formal_manifold_exact_backend.py`; the FM module becomes a thin
  re-export shim (kept until step 9 archival of the FM route). Update
  importers: `accepted_refit.py:21`, `generic_static_adapt_variants.py:112`,
  `adapt_pipeline.py:595`. Tests:
  `test_static_adapt_accepted_refit.py`,
  `test_static_adapt_accepted_refit_external_gram.py`.
- `pipelines/static_adapt/ra_adapt/support.py` (new): host the retained
  raw-Gram support factorization by delegation to
  `joint_linear_solve.factor_supported_metric`; add the single receipt type
  with threshold/eigenpair/tolerance fields; migrate duplicated support/ridge
  logic call-sites discovered in step 2's index (preflight finding 5) to it
  one at a time. Tests: `test_static_adapt_joint_linear_solve.py`,
  `test_static_adapt_projected_generalized_trust_solve.py`, plus a new
  `test_ra_adapt_support_factorization.py` asserting identical eigenpairs and
  retained masks between old and new call paths on random SPD fixtures.
- `pipelines/static_adapt/ra_adapt/trust.py`,
  `ra_adapt/insertion_geometry.py`, `ra_adapt/pools.py` (new): thin owners
  delegating to the monolith module-level helpers named in §2.2 (these
  helpers are already top-level functions, so extraction is import-motion,
  not logic rewriting). Tests: `test_static_adapt_route_a_trust_region.py`,
  `test_static_adapt_commutation_reduced_insertion.py`, new
  `test_ra_adapt_pools.py` against the step-1 fixtures.
- Rollback boundary: each new module is additive with delegation; reverting
  is deleting the module and restoring the direct imports.
- Failure modes: import cycles (monolith imports scoring which imports
  static_adapt modules — keep new modules dependency-light and import the
  monolith lazily as `_context.py` already does, adapt_pipeline import at
  `sr_snake/_context.py:637`).

### Step 4 — Build the RA and Append facades

- `ra_adapt/contracts.py`: typed request/protocol/receipt surface. Reuse the
  `sr_snake/contracts.py` receipt types where semantics are identical
  (import, do not fork). New fields: `candidate_representation`
  (`macro_generator_v1` | `single_pauli_word_v1`), `active_gradient_policy`
  (`stationary_source_response_v1` | `measured_residual_response_v1`),
  `resource_weighting_scope` (`late_resource_weighting_v1` |
  `all_phase_resource_weighting_v1`), separate `kappa`/`lambda` Phase-III
  receipt fields, parent-inventory and executable-pool hashes.
- `ra_adapt/engine.py`: start as the generalization of the typed singleton
  controller: the engine runs the existing singleton path unchanged when
  given `SinglePauliWordCandidateAdapter` (delegating to the same numerical
  runtime the sr_snake facade uses today), and runs the repaired macro path
  when given `MacroCandidateAdapter` (exact ordered chart at every position,
  projected generalized solve, no-overlap trust, common accepted refit).
- `ra_adapt/append.py`: `run_append_adapt` executes the conventional append
  selector over `adapter.executable_pool()` (macro) or the global guarded
  child pool (single-Pauli-word), sharing execution/compile/accounting/refit
  with the engine.
- `run_sr_snake` is untouched in this step.
- Tests (new): `test_ra_adapt_facade.py` (request resolution, digests,
  fail-closed validation), `test_ra_adapt_macro_exact_insertion.py` (§4 T1),
  `test_ra_adapt_append_facade.py` (§4 T7/T8).
- Rollback: new package only; nothing else references it yet.
- Failure modes: silently diverging singleton behavior — guarded by step 6
  parity; accidental public seam for the study policies — guarded by a test
  asserting the facade rejects requests carrying gradient/cost-scope fields
  unless bundle-materialized (`test_ra_adapt_facade.py`).

### Step 5 — Migrate callers

Order: singleton, macro, Append, JR.

1. Rewire `sr_snake/runner.run_sr_snake` to delegate to
   `run_ra_adapt(problem, request-with-SinglePauliWordCandidateAdapter)`;
   keep the exported receipt types intact. All existing sr_snake facade tests
   must pass unmodified — they are the parity harness.
2. Macro callers: there is no retained ordinary macro caller. The historical
   entry points (`paper_i_runner.py`, monolith CLI macro configurations) are
   *not* migrated — they are superseded by bundle-materialized protocols
   (§7). Mark both as retirement candidates pending step 8 reachability.
3. Append callers: point the retained Paper-I append comparison at
   `run_append_adapt`. `generic_static_adapt_variants.py` keeps serving
   Geo-ADAPT and frozen replay identities; its append/TETRIS execution roles
   shrink per §5.7/§5.8.
4. JR: no migration; add the funnel-semantics regression noted in §1.6 if
   step 1 did not already capture it.
- Tests: full sr_snake suite, `test_paper_i_canonical_interface.py`,
  `test_paper_i_append_registry.py`, new append-facade tests.
- Rollback: step-5.1 is a single-commit rewire; revert restores the old
  runner body.
- Failure modes: resume/checkpoint compatibility (`sr_snake/_resume.py`
  authenticates the route contract digest — the delegation must preserve the
  emitted profile id and digest exactly; test with
  `test_paper_i_sr_snake_resume_adapter.py` and
  `test_static_adapt_resume_scaffold.py`).

### Step 6 — Prove parity and contract behavior

- Singleton parity: step-1 trajectory fixtures replayed through the new path
  byte-identically (operator sequence, positions, energies within recorded
  tolerances, trust receipts, `S_alg` components, route digest).
- Macro contract behavior (not parity — the repair intentionally changes
  trajectories, **[LOCKED]** "do not assume the corrected macro trajectory
  should reproduce the historical operator sequence"): assert receipts, not
  trajectories — §4 matrix.
- Append: identical selected sequence vs the frozen comparator on a fixed
  short-horizon fixture with the common 102/148 pool for the macro condition.
- Evidence: green runs of the §4 matrix committed to the lock receipt.
- Rollback: previous steps remain individually revertible.

### Step 7 — Migrate active guidance (only now)

- Update `agent_guidance/static-adapt/AGENTS.md` ordinary-interface lines,
  `agent_guidance/static-adapt/CONTEXT.md`, `agent_guidance/shared/run-guide.md`,
  root `AGENTS.md` routing rows, and `MATH/AGENTS.md` route-identity text from
  SR-SNAKE/`run_sr_snake` to RA-ADAPT/`run_ra_adapt`, preserving historical
  labels for preserved artifacts (**[LOCKED]** naming rules). Do not touch
  manuscripts. Guidance must not point at an unfinished interface — this step
  strictly follows a green step 6.
- Failure mode: stale-router references to skills/files that do not exist —
  follow the root AGENTS.md skill-trigger discipline when editing.

### Step 8 — Prove retired paths unreachable

For each family in §5: path-limited `rg` caller sweeps + the GitNexus index
answers + a focused import test
(`test/test_ra_adapt_retired_reachability.py`, asserting the retired modules
are not imported by any active package under `pipelines/` and `src/` — allowed
importers are only the archive manifest and the family's own files). Record
per-family proofs in the archive manifest (§5.15).

### Step 9 — Create the inert archive

`archive/paper_i_static_adapt_legacy_20260727/` per §5.15. Git-mv nothing
blindly; every file entry follows its family's row and the mandatory order
(classify → author-retirement confirmed → extract neutral code → migrate
callers → focused tests → reachability proof → archive). `.py` files land as
`.py.txt` snapshots plus the manifest; no importable modules, no test
discovery, excluded from GitNexus and packaging.

### Step 10 — Materialize the source-locked run bundles

Per §7. No CHTC submission in this step; materialization + validation only.

---

## 4. Scientific and characterization test matrix

Tests live under `test/`; none may add production flags, callbacks, or
test-only branches (**[LOCKED]**). Where a seam is needed it must be one of
the scientifically meaningful module boundaries of §2.

| id | Contract | New/existing test |
|---|---|---|
| T1 | Finite-difference actual-position gradients and Hessians match the engine's insertion geometry for append (`p = n_k`) and interior (`p < n_k`) positions, both adapters | new `test_ra_adapt_insertion_derivatives_fd.py`; extends `test_static_adapt_commutation_reduced_insertion.py` |
| T2 | One neutral support-factorization *implementation* (single owner, `ra_adapt/support.py`): the Phase-III solve and the trust transaction reuse a single selector-support receipt with identical thresholds/eigenpairs/retained masks (same raw selector Gram); accepted refit independently constructs and factorizes its own post-admission full-ansatz Gram (accepted_refit l.744-749) through the same factorization implementation and conventions — generally a different dimension and support, so no eigenpair/mask identity is asserted against the selector window; all receipts carry provenance ids | new `test_ra_adapt_support_factorization.py`; existing `test_static_adapt_projected_generalized_trust_solve.py`, `test_static_adapt_accepted_refit.py` (refit scope independent of the selector window, l.129) |
| T3 | Phase-III solve contract: projected generalized raw-Gram solve, no inverse square root, no metric ridge, whitening recorded false; separate `kappa` and `lambda` receipt fields with λ = μ − κ ≥ 0, exercised in both regimes — curvature-only interior (κ>0, λ=0; trust boundary reported inactive) and trust-bounded (κ>0, λ>0; boundary active) | existing solver tests + new receipt assertions in `test_ra_adapt_facade.py` |
| T4 | No-overlap trust: zero endpoint-overlap acquisitions in both adapters' trust receipts; overlap-call count assertions via the estimator ledger | new `test_ra_adapt_trust_no_overlap.py`; existing `test_static_adapt_route_a_trust_region.py` |
| T5 | Accepted refit: one fixed supported-FS-whitened chart per admission, complete enlarged ansatz, chart discarded after; macro and singleton use the same implementation | existing `test_static_adapt_accepted_refit.py` + new cross-adapter identity assertion |
| T6 | Common executable macro-pool hash for macro RA and macro Append (102/148); parent-inventory hash (123/171) unchanged for both singleton methods | new `test_ra_adapt_pools.py` against step-1 fixtures |
| T7 | Staged-child RA exposure vs global-child Append exposure: RA children only descend from retained parents; Append scans the full guarded child pool; both use canonical unit-Pauli representatives | new `test_ra_adapt_child_exposure.py`; existing `test_generic_static_projected_singleton_pool.py` |
| T8 | Append facade shares execution/compile/accounting/refit but never invokes the P1/P2/P3 funnel | new `test_ra_adapt_append_facade.py` |
| T9 | Singleton parent ancestry unchanged (123/171 source, lanes, guard, dedup) | step-1 fixtures + existing `test_static_adapt_guarded_singleton_pool_route.py` |
| T10 | Stationary-source protocol: no active-gradient acquisitions (`active_gradient_indices_acquired` empty), no `N_grad` charges for the active block, coupling-only Schur response; measured-residual protocol acquires and uses `g_theta` and charges it | new `test_ra_adapt_gradient_policy.py` |
| T11 | Cost-scope receipts: `late_resource_weighting_v1` produces Phase-I score with unit cost factor and resource-weighted Phases II/III; `all_phase_resource_weighting_v1` applies it in all three; receipts record the scope id | new `test_ra_adapt_cost_scope.py` |
| T12 | Estimator accounting closure: `S_alg = N_H_outer + N_H_refit + N_grad + N_metric` closed-occurrence reconciliation for every protocol | existing `test_paper_i_s_alg_accounting.py`, `test_paper_i_hh_runtime_postrun_s_alg_audit.py` + new bundle-level assertions |
| T13 | Deterministic preservation of the existing single-Pauli-word plateau-triggered commutation-aware insertion route under the historical-compatible policy: replay equals step-1 fixture trajectory | new `test_ra_adapt_singleton_plateau_preservation.py`; existing `test_static_adapt_insertion_commutation_plateau.py` |
| T14 | Import/reachability before archival | `test_ra_adapt_retired_reachability.py` (§3 step 8) |
| T15 | Facade fail-closed behavior: unknown fields, missing horizon, non-HH problems, study policies outside bundles | `test_ra_adapt_facade.py` |

Trajectory regressions (T13, step-6 parity) are behavioral evidence only;
no test asserts that a repaired trajectory is scientifically better
(**[LOCKED]**).

---

## 5. Legacy-contraction ledger

Scope guard **[LOCKED]**: Paper-I/static-ADAPT only; nothing from
`pipelines/time_dynamics/`, QSE, Paper IV/V. Non-canonical does not mean
dead; every row follows the seven-step archival order. "Archive" always means
inert `.py.txt` + manifest entry in
`archive/paper_i_static_adapt_legacy_20260727/`, never deletion of git
history or preserved run artifacts.

### 5.1 FM-SNAKE — author-retired route; extract neutral primitives first

- Files: `formal_manifold_exact_backend.py`, `formal_manifold_local_campaign.py`,
  `formal_manifold_pareto_campaign.py`, `formal_manifold_sr_source_locked_campaign.py`,
  `formal_manifold_outer_information.py`, `formal_manifold_sr_v3_outer_bridge.py`,
  `formal_manifold_warm_start.py`, `formal_manifold_route_profile.py`,
  `reclose_formal_manifold_query_accounting.py`, monolith `_formal_*` helpers
  (adapt_pipeline l.1160-1401, 10254-10430, 15783, 17650).
- Current active importers **[FACT]**: `accepted_refit.py:21`
  (exact backend); `hh_continuation_scoring.py:28` (route profile constants);
  `adapt_pipeline.py:534-595`; `cli_config.py:94-109`;
  `output_artifacts.py:27`; `selector_exact_query_geometry.py:34`;
  `resume_scaffold.py:34-43`; `generic_static_adapt_variants.py:112-115`;
  `paper_i_runner.py:84`; `paper_i_hh_powell_pareto.py:48-58`.
- Neutral extraction: exact backend → `exact_geometry_backend.py` (step 3);
  audit `formal_manifold_warm_start` imports — anything consumed by retained
  Append/Geo replay or accepted-refit paths moves to its proper active owner
  first; FM route-profile constants used by scoring get re-homed with the
  scoring config they parameterize.
- Then: migrate callers, T14 proof, archive FM route/campaign/config/adapter
  code and FM-only tests (`test_static_adapt_formal_manifold_*.py`,
  `test_paper_i_hh_fm_vs_append_fm_first_hit.py`,
  `test_paper_i_hh_append_fm_first_hit_campaign.py` — the latter two only if
  solely FM-route-specific after classification). No future FM seam.

### 5.2 JR-SNAKE — retained optional extension

- Files kept live: `route_a_funnel.py`, `route_a_schur_selector.py`,
  `joint_step_warm_start.py`, `batch_ordering.py`, JR reporting builders.
- Action: none beyond the §1.6 semantics verification and keeping imports
  working through the kernel extraction. Not an archival target.

### 5.3 Optuna calibration — author-retired

- Files: `pipelines/static_adapt/optimization/{staged_adapt_optuna,phase3_policy_optuna,hh_optuna_evidence_ledger,phase3_robustness_gate,hh_snake_interpretable_ml_analysis,hh_snake_shallow_feature_extract}.py`,
  `pipelines/exact_bench/paper_i_hh_route_a_optuna.py`,
  `paper_i_hh_optuna_artifact_offload.py`, `paper_i_hh_live_optuna_overlay_refresh.py`,
  Optuna-only tests (`test_paper_i_hh_route_a_optuna.py`, …).
- **[FACT]** Broad import tentacles into `pipelines/exact_bench/*` (table_i,
  generic benchmark/enrichment) — classify each import: if a current RA/Append
  bundle demonstrably needs a small utility, extract only that minimum into
  its active owner; do not retain a generic abstraction because it is
  imported from an Optuna-named module (**[LOCKED]**).
- `chtc/phase3_optuna/` submit scripts and generated records: historical run
  artifacts — preserve immutably, do not archive or edit.

### 5.4 Phase-live hysteresis — author-retired mechanism, retained plateau detector

- Files: `controller_phase_state.py` (`_controller_phase_live`, l.41),
  `phase_live_hysteresis` fields in `cli_config.py` (4 sites) and
  `adapt_pipeline.py` (6 sites), checkpoint/resume controls that exist only
  for phase retirement/reactivation.
- Retain and clearly separate: the plateau detector used for
  commutation-aware insertion (`_insertion_commutation_plateau_round_policy`,
  adapt_pipeline l.11327 — note l.11389 `hysteresis_active: False`; the two
  mechanisms are already distinct **[FACT]**).
- Archive the hysteresis mechanism + controls after reachability proof;
  canonical profiles already pin `phase_live_hysteresis_enabled=false`.

### 5.5 Ordinary novelty scoring — author-retired; rename the fallback

- Files/symbols: `NoveltyOracle`, pairwise novelty
  (`PHASE2_NOVELTY_LEGACY_PAIRWISE_V1`, hh_continuation_scoring l.271-277,
  3030-3223), `novelty_gamma_schedule_context` (l.2862), gamma controls,
  Phase-II/III novelty multiplier policies and their route/config surfaces.
- Retain: the deferred-Gram all-models-infeasible fallback
  (`_all_energy_models_infeasible_novelty_fallback_telemetry`, adapt_pipeline
  l.10017; `_selected_admission_novelty_fallback_receipt` l.10164).
  **[LOCKED 2026-07-27]** renamed to
  `deferred_gram_all_models_infeasible_fallback_v1` for new receipts. Do not
  add active compatibility machinery solely to reinterpret old telemetry
  keys; historical artifacts and checkpoints retain their original fields
  unchanged (user decision resolving Q3).
- Note: the fallback's telemetry serialization contract is load-bearing for
  canonical checkpoints (route-identities registry requires explicit
  `enabled/fired/rounds/charge`) — the new-name receipts must keep those
  fields.

### 5.6 Historical amplitude pruning — author-retired; retain measured delete-and-refit

- Files/symbols: `PRUNE_POLICY_LEGACY_SMALL_ANGLE_V1` +
  `amplitude_collapse_witness` (imported into adapt_pipeline l.268-280, used
  l.27123), amplitude-collapse witness knobs (adapt_pipeline l.14630-14647),
  `prune_risk_dataset.py` amplitude fields, amplitude acceptance/telemetry.
- **[FACT]** `cli_config.py:2095-2096` still defaults
  `phase1_prune_policy` to `legacy_small_angle_v1`. The replacement surfaces
  must not inherit that default; canonical RA protocols carry pruning off or
  the typed metric/trust policies only.
- Retain: `recoverability_ladder_v1` machinery and the typed metric- and
  trust-region-nominated measured delete-and-refit policies
  (`hh_continuation_pruning.py`, `prune_ladder.py` retained parts,
  `query_neutral_full_geometry_prune.py`, `phase3_material_window.py` as
  currently used by canonical profiles).

### 5.7 TETRIS/disjoint batching — author-retired in full

- Standalone comparator: `static_tetris_qubit_adapt_vqe` config + branches in
  `generic_static_adapt_variants.py` (l.194, 1367-1375, 5508-5542, rule
  l.1105).
- Duplicated selector: `tetris_disjoint_batch_select`
  (hh_continuation_scoring l.18116, mode registration l.1471, dispatch
  l.18258). Correction to the decision record's wording **[FACT]**: this
  duplicate lives in the scoring module, not in `adapt_pipeline.py`; ledger
  targets it there. (The `isdisjoint` use at adapt_pipeline l.24704 is
  phase-0 screen context assembly, not a batch selector — do not archive it
  under this family without its own classification.)
- Preserve historical TETRIS artifacts; prove no retained caller; archive
  both surfaces.

### 5.8 Legacy ADAPT executors — author-retired after caller migration

- `pipelines/static_adapt/adapt_pipeline_legacy_20260322.py`,
  `compare_adapt_current_vs_legacy_20260322.py`, and the explicit legacy
  adapter `sr_snake/_legacy_adapter.py` (42 l., `run_legacy_sr_snake`).
- The monolith itself contracts progressively via steps 3-6; whatever remains
  of `_run_hardcoded_adapt_vqe` after the engine owns the canonical paths is
  classified at step 8 (retained compatibility for preserved replay vs
  archived). Do not archive the monolith wholesale in this refactor.
- `pipelines/hardcoded/` is **not** moved wholesale; see §5.11.

### 5.9 Historical profiles and CLI controls — author-retired after migration

- Old SR v1/v2/v3/v3.1/v4 profile registrations in
  `sr_snake_route_profile.py` remain **readable identities** for preserved
  evidence and resume authentication; what is archived are the retired
  route registries and legacy CLI controls: `route_identity.py` (767 l.),
  retired portions of `cli_config.py`, `sr_snake/_cli_compatibility.py`
  (2,442 l.) after the bundle surface replaces CLI launches, `noise_routes`/
  `optimizer_routes` CLI plumbing that only legacy launches used.
- Order matters: canonical RA-ADAPT and retained JR callers migrate first;
  resume-digest authentication for old checkpoints must keep working
  (`sr_snake/_resume.py` retains its profile table).

### 5.10 Current optional RA policies — retained

- Typed greedy/combinatorial batching (`sr_snake` contracts + controllers),
  metric/trust-region pruning, beam policy, escape profiles: retained as
  non-default optional extensions; the engine must keep their typed seams
  working (facade tests already cover greedy/combinatorial admission).

### 5.11 Paper-I `pipelines/hardcoded` aliases — author-retired after migration

- **[FACT]** Exactly seven untracked forwarders confirmed by `git status`:
  `adapt_pipeline.py`, `adapt_circuit_cost.py`, `hh_continuation_generators.py`,
  `hh_continuation_scoring.py`, `hh_continuation_symmetry.py`,
  `hh_continuation_types.py`, `imported_artifact_resolution.py` — each a
  module-forwarding alias to `pipelines/static_adapt/adapt_pipeline` or
  `pipelines/scaffold/*`.
- Preserve their dirty contents (snapshot into the archive as `.py.txt`
  *copies*; the untracked working files themselves are user dirty work — do
  not delete without the migration + reachability proof). Migrate any
  remaining imports of `pipelines.hardcoded.<alias>` to the canonical owners
  (**[VERIFY]** current importers with a path-limited sweep; note stale
  `__pycache__` entries exist for retired names like
  `hh_realtime_checkpoint_controller` — pycache is not evidence of
  reachability).
- The rest of `pipelines/hardcoded/` (noise/staged-noise/realtime tools,
  `hh_continuation_pruning.py`, `hh_continuation_rescue.py`,
  `adapt_circuit_execution.py`) is out of scope — untouched.

### 5.12 Old Paper-I runners — author-retired after replacement

- `pipelines/static_adapt/paper_i_runner.py` (facade over the compatibility
  runner) and the route-specific executable portions of
  `pipelines/exact_bench/paper_i_hh_powell_pareto.py` (2,896 l.).
- Replacement: the typed RA/Append run-bundle surfaces (§7). Preserve
  immutable source locks, run artifacts, and provenance before archival.
  **[LOCKED 2026-07-27]** (user decision resolving Q2): replacement of the
  route-specific executable portions of `paper_i_hh_powell_pareto.py` is
  authorized in this refactor, preconditioned on an objective verification
  that no active, queued, or outstanding source-locked run still invokes it
  (scheduler queues, chtc submit records, and run-skill queue sources checked
  and recorded in the lock receipt). All historical artifacts and provenance
  are preserved.

### 5.13 Commutation Route A record partitioning — author-retired

- The algebraic-lane partitioning: `STATIC_LANE_ROUTE_ALGEBRAIC`
  (lane_routes.py l.16), `LANES_PHASE1`/`LANE_MIX`/`LANE_FLAT`/`LANE_CURV`/
  `LANE_DISJ` and the lane-assignment logic in `algebraic_metadata.py`
  (l.19-20, 478-546), plus monolith call-sites routing shortlists by
  algebraic lane.
- Retain in place: `pauli_words_commute`, `exact_expansions_commute`,
  `support_qubits_from_pauli_word`, `multiply_pauli_words`, expansion
  serialization — required by commutation-aware insertion (canonical,
  preservation-critical) and the symmetry gates. Extraction boundary: split
  `algebraic_metadata.py` into retained commutation/expansion math and the
  archived lane partitioning before archival.
- Retain: Phase-I physical-family lane protection
  (`STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE` and
  `pipelines/contracts/static_provenance.physical_operator_lanes_for_problem`).
- Remove from the canonical engine after reachability confirmation:
  post-split lane-based decisions (lane fields consulted after child
  projection; **[VERIFY]** enumerate these sites via the GitNexus index +
  `rg -n "lane" pipelines/static_adapt/adapt_pipeline.py` classification
  during step 3).

### 5.14 Historical artifacts and PDFs

Preserved immutably wherever they live (`MATH/paper_details/figures/…`,
`docs/reports/`, chtc records, provenance JSON). No executable legacy parser
is preserved solely to regenerate them.

### 5.15 Archive layout

```text
archive/paper_i_static_adapt_legacy_20260727/
  MANIFEST.json        # schema paper_i_legacy_archive_manifest_v1:
                       #   per entry: original_path, sha256, family,
                       #   retirement_decision (pointer into decision record),
                       #   reachability_proof (test id + index answer),
                       #   neutral_extractions (what moved where first)
  code/<family>/<original_name>.py.txt
  tests/<family>/<original_test>.py.txt
  guidance/<family>/…  # obsolete route-specific handoffs/runtime docs
```

Excluded from imports, pytest discovery (`norecursedirs`/collect-ignore
entry), packaging, GitNexus (§6 ignore), and ordinary agent navigation
(root/lane AGENTS ignore lists already exclude `archive/`). Active
navigation retains only the compact manifest pointer.

---

## 6. ICM and GitNexus

ICM stages for this repair: `lock -> refactor -> verify -> materialize
bundles -> analyze -> user review`. ICM owns stage routing and receipts, not
scientific defaults. Stage receipts are JSON files under
`agent_guidance/static-adapt/icm/ra-adapt-repair-20260727/<stage>.json`
**[PROPOSED]**, each recording authoritative inputs (paths + sha256),
outputs, checks run, state, and supersession lineage — never duplicating
algorithm settings (they point at the resolved protocol digests instead).

GitNexus is index-only, part of the `lock` stage, and never a runtime
dependency or evidence selector.

`.gitnexusignore` (new, repo root) **[PROPOSED]** — strict surface:

```text
# generated outputs and evidence
artifacts/
raw_outputs/
output/
tmp/
logs/
plots/
docs/
prompt-exports/
archive/
MATH/**/*.pdf
MATH/paper_details/figures/
chtc/**/input/
chtc/**/output/
*.pdf
*.png
*.svg
*.log
# environments, caches, editor state
.venv*/
__pycache__/
*.pyc
.obsidian/
.vscode*/
.claude/
claude_code_adapt_wave/
paper_5/
```

Included (implicitly): `src/`, `pipelines/`, `test/`, consumed
schemas/configs, and importable compatibility code needed for reachability
audits. Validation before indexing: list the candidate file set and assert it
contains no PDFs, artifacts, or cache files. Never run `gitnexus setup`;
generate no agent files, hooks, skills, or context files.

Questions the index must answer (each confirmed in source + focused tests
before use):

1. Which callers reach `run_sr_snake` / will reach `run_ra_adapt`?
2. Which modules import the monolith (`pipelines.static_adapt.adapt_pipeline`)
   and which of its top-level helpers they use?
3. Caller concentration and dependency direction for each §5 family
   (especially FM neutral primitives and Optuna tentacles into
   `exact_bench`).
4. Duplicate policy paths for support/ridge/tolerance logic (preflight
   finding 5) — enumerate call-sites for `factor_supported_metric`,
   whitened-eigh solves, and ad-hoc `eigh` support selection.
5. Zero active reachability for every archival candidate at step 8.

---

## 7. Run bundles and execution handoff

### 7.1 Bundle contract **[PROPOSED]**

A run bundle is a finite, source-locked collection of resolved RA-ADAPT
protocols and expected artifacts for one study — not a campaign framework,
launcher, or defaults source (**[LOCKED]**).

```text
chtc/paper_i_ra_adapt_repair_20260727/bundles/<bundle_id>/
  bundle_manifest.json      # schema ra_adapt_run_bundle_v1
  protocols/<cell_id>.json  # one resolved RA-ADAPT protocol per cell + digest
  source_locks.json         # provenance pointers (visible-row source maps,
                            #  settings JSON, sha256) resolved via
                            #  agent_guidance/skills/shared/scripts/resolve_visible_settings.py
  expected_artifacts.json   # per cell: manifest/checkpoint/result/summary paths
  validation_report.json    # written by the materializer's local checks
```

Each resolved protocol records explicitly (alignment-note checklist items
1-11): derivative chart id (`exact_ordered_insertion_zero_angle_v1`),
trust-update policy id (source-metric no-overlap), Phase-III solver identity
(projected generalized) with separate `kappa`/`lambda` fields,
parent-inventory hash (123/171) and executable-pool hash (102/148 for macro
cells), `candidate_representation`, `active_gradient_policy`,
`resource_weighting_scope`, accepted-refit scope/chart/base-chart, optimizer
and budget, stopping rule and horizon, seeds, estimator-accounting
convention, and the Qiskit compile identity (optimization level 0, transpiler
seed 7, common basis set, reference prep included, no coupling map — same as
the displayed rows, **[FACT]** per alignment note "Compiled resources").

Baseline settings come from the best visible table/figure result for the same
method and regime (Paper-I run-skill gate); the only intended changes are the
locked repair items D1-D5 plus output paths/labels. Missing provenance fails
closed.

### 7.2 Study 1 — stationarity comparison (both bundles, late weighting)

Two matched bundles, identical except `active_gradient_policy`:

1. `ra_repair_stationary_late_v1` — `stationary_source_response_v1`;
2. `ra_repair_measured_late_v1` — `measured_residual_response_v1`.

Cell matrix per bundle:

- **Validation matrix (small, defect-sensitive), runs first**: macro cells at
  `nph=3` only, reduced horizon (enough controller rounds to exercise ≥1
  interior insertion and ≥1 trust contraction), all four macro rows: macro
  Append (102/148), macro RA append-only, macro RA plateau-triggered
  insertion, macro RA always-enabled insertion; plus one singleton
  plateau-route preservation cell (historical-compatible policy, T13
  contract). Regime choice **[LOCKED 2026-07-27]** (user decision resolving
  Q1): `strong_weak_u8` and `strong_strong_u8` at `nph=3`, chosen because
  these regimes strongly exercise interior insertion.
- **Objective progression gates** (CHTC agent may continue without new run
  approval when all pass, **[LOCKED]**): route/profile digest match per cell;
  every Phase-II/III geometry receipt carries the exact ordered chart at its
  recorded position; zero endpoint-overlap acquisitions in trust receipts;
  pool-hash equality across compared cells; `S_alg` closure; gradient-policy
  receipts match the bundle (T10 semantics); deterministic short replay of
  one cell reproduces its trajectory; artifact completeness per
  `expected_artifacts.json`.
- **Complete macro repair matrix**: the four macro rows × both cutoffs
  (`nph=3`, `nph=7`) × every visible displayed macro regime resolved from the
  macro provenance tracker, at full source-locked horizons.
- **Singleton scope**: targeted preservation runs for the successful
  plateau-triggered commutation-aware insertion route only. A full singleton
  replacement matrix is required only if the user-selected canonical policy
  differs from the historical single-Pauli-word protocol (**[LOCKED]**).

The workflow pauses after Study 1. Agents report complete objective outputs
(trajectories, receipts, `S_alg`, compiled resources, gate outcomes); they do
not select the winner and do not characterize results as promotable.

### 7.3 Study 2 — Phase-I resource weighting

After the user selects the Study-1 gradient policy: one bundle
`ra_repair_<winner>_allphase_v1` with `all_phase_resource_weighting_v1`,
otherwise identical to the winning Study-1 bundle; its comparison source is
the corresponding late-weighting run. Never combined with the stationarity
change in one comparison (**[LOCKED]**).

### 7.4 Responsibilities

- GPT-5.6 ultra: implement, test, materialize bundles, run
  `validation_report.json` checks locally.
- GPT-5.6 high: CHTC submission/fetch/monitoring per
  `agent_guidance/skills/paper-i-run/SKILL.md` and root AGENTS.md run-safety
  cadence; advances validation → full matrix on green gates; pauses at the
  Study-1/Study-2 user decision gates.
- User: selects the stationarity winner, decides evidence/manuscript use,
  answers §8.3.

### 7.5 Serialization and provenance contract **[LOCKED — Code Math Bijection readiness addendum §6, reqs 1–12; environment/dependency rule resolved 2026-07-27]**

This is ordinary scientific provenance realized as serialized manifests and
movement receipts. It adds no TypeScript/semantic-engine import, no runtime
callback, no semantic-graph generation, no UUID resolver, and no CMB-specific
public knob to the scientific algorithm (§2.4). GPT-5.6 ultra must satisfy
every item below in the serialized artifacts, not only in memory.

1. **Append-only typed ids.** Policy/chart/solver/representation/profile id
   strings are never renamed or reused after emission; new semantics require a
   new versioned id. Historical ids on preserved artifacts are never relabeled
   (§5.9 naming rules).
2. **Canonical digested JSON.** Every resolved RA/Append protocol, route
   contract, pool inventory, and bundle manifest serializes as canonical JSON
   (sorted keys, no NaN) with a `sha256` digest field, following the existing
   route-contract pattern.
3. **Pool hashes.** The parent-template inventory (123/171) and executable
   macro pool (102/148) serialize as ordered labels + sha256; the identical
   hash values appear in every RA and Append manifest that shares the pool
   (T6).
4. **Protocol vs execution separation.** The resolved protocol (immutable,
   digested) is kept distinct from the execution manifest (observed run). Each
   execution manifest records: command argv + cwd, seeds, `git_commit`,
   `dirty_working_tree`, input source-lock hashes, output artifact path +
   sha256, timestamps, and exit status. It additionally records deterministic,
   non-secret **environment fingerprints and dependency-lock hashes**; a
   missing lock hash is recorded explicitly (e.g. `dependency_lock_sha256:
   null` with a `dependency_lock_status` reason) — never inferred or silently
   omitted (resolved 2026-07-27: environment/dependency fingerprints are
   included, not deferred).
5. **Candidate lineage.** Candidate records serialize representation id,
   generator identity, parent identity for children, and the explicit
   insertion position `p` in geometry receipts.
6. **Policy echo.** `active_gradient_policy` and `resource_weighting_scope`
   appear in every protocol and are echoed in result receipts; stationary-
   source results show empty active-gradient acquisition telemetry and zero
   corresponding ledger charges (T10/T11).
7. **Typed scientific receipts, serialized.** Support factorization
   (threshold/eigenvalues/mask/provenance id), trust transaction (policy id,
   radii, zero overlap acquisitions), Phase-III with separate κ and λ fields
   (§2.2, T3), accepted-refit chart (scope/chart/base-chart ids + chart hash),
   and `S_alg` component closure are written into the serialized results, not
   held only in memory (T2/T3/T4/T5/T12).
8. **Stable qualified names + movement receipts.** The public seams
   (`run_ra_adapt`, `run_append_adapt`, the adapters, the support
   factorization, the trust transaction, the accepted-refit builder, the
   bundle materializer) keep stable qualified names. Every code move is
   recorded in the archive `MANIFEST.json` (§5.15) or an ICM stage receipt
   (§6) as old path/qualified name → new path/qualified name + content sha at
   move time.
9. **Route supersession lineage.** New RA route contracts keep the
   `lineage_authority` pattern (`parent_route_profile` +
   `parent_contract_sha256`) pointing at the historical profile they supersede
   (§5.9 resume-digest preservation).
10. **Selector identity.** Result manifests carry a selector-identity field
    distinguishing the RA staged funnel from the Append conventional selector,
    alongside the shared pool hashes (T8).
11. **No CMB runtime coupling.** No TypeScript semantic engine import, runtime
    callback, semantic-graph build, UUID resolution, or CMB-specific public
    knob is added to the scientific algorithm — everything above is ordinary
    provenance.
12. **Fail-closed drift.** Historical semantic outputs, telemetry keys, and
    promotion receipts are never overwritten to appear current; drift is
    handled by new revisions and supersession links, fail-closed (§5.5, §8.3
    Q3).

---

## 8. Review packet

### 8.1 Checklist for the original audit agent

- [ ] D1-D5 defect statements match the alignment note and cite real code
      (§1.7 anchors spot-checkable in <10 min).
- [ ] The engine/adapters/append split preserves: staged RA exposure vs
      global Append exposure; conventional Append selector outside the
      funnel; 102/148 macro comparison; 123/171 singleton ancestry.
- [ ] No public seam for gradient policy or cost scope; bundle-only (§2.1,
      T15).
- [ ] Insertion repair uses the existing exact-chart machinery
      (`_promote_fresh_phase3_joint_geometry_receipt`) rather than new
      geometry code.
- [ ] Every §5 family has: files, callers, neutral extraction, migration,
      reachability proof, archive destination; JR and the plateau detector
      and measured delete-and-refit and the deferred-Gram fallback and
      commutation math are on retain lists, not archive lists.
- [ ] Migration steps each carry tests, evidence, rollback; guidance
      migration strictly after parity (step 7 after step 6).
- [ ] Bundles are finite and source-locked; progression gates are objective;
      user decision gates present after Study 1 and before Study 2.
- [ ] No step launches science, edits manuscripts, or replaces evidence.

### 8.2 Deviations from the governing decision record

1. **Location correction**: the duplicated disjoint/TETRIS batching selector
   is in `pipelines/scaffold/hh_continuation_scoring.py:18116`
   (`tetris_disjoint_batch_select`), not inside `adapt_pipeline.py`. Same
   retirement decision, corrected target (§5.7).
2. **Clarification, not deviation**: `algebraic_metadata.py` hosts both the
   author-retired commutation/qubit-support lane partitioning and the
   retained commutation mathematics; §5.13 defines the split so the retained
   math is not accidentally archived.
3. No other deviations. The `grouped_exact` execution proposal remains not
   adopted; the repair uses the 102/148 executable pool as locked.

### 8.3 User questions — resolved 2026-07-27

All three questions raised by the initial version of this specification were
answered by the user on 2026-07-27; the answers are recorded here and folded
into the affected sections:

- **Q1 (resolved)**: Study-1 validation matrix uses `strong_weak_u8` and
  `strong_strong_u8` at `nph=3` with reduced horizon, because these regimes
  strongly exercise interior insertion. The singleton plateau-route
  preservation cell is retained. (§7.2 updated.)
- **Q2 (resolved)**: the route-specific executable portions of
  `paper_i_hh_powell_pareto.py` are replaced in this refactor, after an
  objective verification that no active, queued, or outstanding
  source-locked run still invokes it; historical artifacts and provenance
  preserved. (§5.12 updated.)
- **Q3 (resolved)**: `deferred_gram_all_models_infeasible_fallback_v1` is
  confirmed for new receipts. No active compatibility machinery is added
  solely to reinterpret old telemetry keys; historical artifacts retain
  their original fields unchanged. (§5.5 updated.)

No user decisions remain open to begin implementation; all remaining choices
are either locked or routine implementation shape.

### 8.4 Risks, ranked by scientific and migration impact

1. **Monolith extraction parity** — `_run_hardcoded_adapt_vqe` is ~56k lines
   with pervasive closure state; any silent behavior change contaminates the
   singleton preservation contract. Mitigation: step-1 fixtures, delegation-
   first extraction (import motion before logic motion), byte-level parity
   gate at step 6, resume-digest authentication tests.
2. **Wrong-lever repair** — repairing the macro chart by patching the reuse
   path instead of routing macro P2/P3 through the fresh-receipt exact-chart
   path could leave a second hidden append-chart consumer. Mitigation: T1 +
   receipt assertions that *every* macro Phase-II/III geometry receipt
   carries the exact chart and a position id.
3. **FM extraction breakage** — `accepted_refit.py` imports the FM exact
   backend; premature FM archival breaks the accepted refit for every route.
   Mitigation: step-3 re-homing precedes any FM movement; T5.
4. **Accounting drift between gradient policies** — stationary-source must
   remove active-gradient occurrences and charges symmetrically in both
   adapters or Study 1 is uninterpretable. Mitigation: T10 + T12 closure per
   protocol.
5. **Optuna minimum-utility creep** — wide exact_bench imports invite either
   over-retention or accidental breakage of retained comparators.
   Mitigation: per-import classification with the index; extract minimum
   utilities only.
6. **Resume/checkpoint compatibility** — old checkpoints authenticate
   against profile digests; the delegating facade must emit identical
   contracts. Mitigation: resume tests in step 5.
7. **Guidance pointing at an unfinished interface** — step 7 strictly after
   step 6; router edits reviewed against the naming rules.
8. **JR semantics misread** — funnel refactor collateral. Mitigation: §1.6
   verification before touching shared imports; JR tests in the step-1
   baseline.
9. **Dirty-tree hazards** — the seven untracked aliases and stale
   `__pycache__` files can mask or fake reachability. Mitigation: snapshot
   aliases first; never treat pycache as evidence; path-limited sweeps only.

### 8.5 Optional future work outside Paper I (not authorized here)

Problem-neutral facade generalization for Paper IV (move the HH `L=2` lock
into a Paper-I problem adapter); accepted-ansatz export seam for Paper II;
ICM workspaces beyond this repair. Listed for boundary clarity only.

---

Files to edit:

- This planning step edited only:
  `agent_guidance/static-adapt/ra-adapt-unification-implementation-spec-20260727.md`.

Proposed implementation surface for GPT-5.6 ultra (create/change; nothing is
deleted before its §5 row completes):

- Create: `pipelines/static_adapt/ra_adapt/{__init__,contracts,engine,adapters,pools,insertion_geometry,support,trust,append,bundles}.py`
- Create: `pipelines/static_adapt/exact_geometry_backend.py`
- Create: `.gitnexusignore`
- Create: `agent_guidance/static-adapt/ra-adapt-lock-receipt-20260727.json`,
  `agent_guidance/static-adapt/icm/ra-adapt-repair-20260727/<stage>.json`
- Create: `archive/paper_i_static_adapt_legacy_20260727/` (manifest + inert
  snapshots, step 9 only)
- Create tests: `test/test_ra_adapt_lock_pool_inventory.py`,
  `test_ra_adapt_lock_singleton_trajectory.py`,
  `test_ra_adapt_lock_macro_receipts.py`, `test_ra_adapt_facade.py`,
  `test_ra_adapt_insertion_derivatives_fd.py`,
  `test_ra_adapt_support_factorization.py`, `test_ra_adapt_pools.py`,
  `test_ra_adapt_trust_no_overlap.py`, `test_ra_adapt_child_exposure.py`,
  `test_ra_adapt_append_facade.py`, `test_ra_adapt_gradient_policy.py`,
  `test_ra_adapt_cost_scope.py`,
  `test_ra_adapt_singleton_plateau_preservation.py`,
  `test_ra_adapt_macro_exact_insertion.py`,
  `test_ra_adapt_retired_reachability.py`
- Change: `pipelines/static_adapt/sr_snake/runner.py` (delegate),
  `pipelines/static_adapt/accepted_refit.py` (import re-home),
  `pipelines/exact_bench/generic_static_adapt_variants.py` (append/TETRIS
  contraction), `pipelines/static_adapt/formal_manifold_exact_backend.py`
  (shim), `pipelines/static_adapt/algebraic_metadata.py` (retained/retired
  split), monolith `pipelines/static_adapt/adapt_pipeline.py` (progressive
  kernel extraction), `pipelines/static_adapt/cli_config.py` (retired-control
  contraction, late), guidance files listed in step 7 (only after step 6).
- Archive (step 9, per §5): FM route/campaign files, Optuna calibration
  files, phase-live hysteresis mechanism, ordinary novelty surfaces,
  amplitude-pruning surfaces, TETRIS/disjoint surfaces,
  `adapt_pipeline_legacy_20260322.py`,
  `compare_adapt_current_vs_legacy_20260322.py`, `sr_snake/_legacy_adapter.py`,
  `route_identity.py` and retired CLI controls, the seven `pipelines/hardcoded`
  aliases, `paper_i_runner.py`, `paper_i_hh_powell_pareto.py` executable
  portions (Q2 resolved: authorized after the no-outstanding-run
  verification), algebraic-lane partitioning, retired-route tests and
  guidance.
