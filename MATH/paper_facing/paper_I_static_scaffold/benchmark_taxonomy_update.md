# Paper I Benchmark Taxonomy Update

Created: 2026-05-11  
Target manuscript: `MATH/paper_details/static_adapt_paper_I.tex`  
Role: implementation-and-editing handoff for Table I and related prose. This is an advisory support document, not manuscript prose.

## 1. Reason for the update

The current Table I draft mixes three different comparison questions:

1. fixed-ansatz VQE baselines;
2. adaptive-controller comparisons under a matched candidate universe;
3. operator-class comparisons such as qubit-excitation/QEB pools.

This makes several rows ambiguous. In particular, Qubit/QEB-ADAPT, TETRIS-ADAPT, Geo-ADAPT, append-only ADAPT, and SNAKE should not be interpreted as fair controller comparisons unless their candidate pools are either matched or deliberately labeled as different operator-class baselines.

The revised protocol should make the comparison object explicit: fixed ansatz, same-pool ADAPT controller, or operator-pool comparator.

## 2. Core decision

Use two fixed-ansatz VQE baselines and separate them from ADAPT rows:

1. **Hardware-efficient VQE.** Generic HEA baseline. This tests an unstructured hardware-facing ansatz.
2. **Family-informed fixed VQE.** A physics-informed fixed ansatz chosen by Hamiltonian class before running benchmarks.

Use `full_meta` for same-pool ADAPT rows whenever the algorithm can naturally operate on a problem-local pool:

- append-only ADAPT: `full_meta`;
- TETRIS-ADAPT: `full_meta` with batching/disjoint-support admission;
- Pos-Geo-ADAPT-VQE: `full_meta` with projected tangent-metric/natural-gradient selection, position search, and QNGD/SPSA refit;
- SNAKE: problem-local `full_meta` or the declared SNAKE pool for that row.

Keep Qubit/QEB-ADAPT and the faithful-enough Geo-ADAPT-VQE row separate as operator-class comparators:

- Qubit/QEB-ADAPT uses a qubit-excitation/QEB singles-doubles pool with raw-gradient ADAPT selection.
- Geo-ADAPT-VQE now refers to `static_geo_qeb_adapt_vqe`: the QEB singles/doubles pool, projected Fubini--Study/natural-gradient selection and stopping, with-replacement selection except immediate repeats, and a benchmark-local QNGD-style inner optimizer.
- QEB rows are not inherently fermion-only: qubit excitations can be applied to any mapped qubit register.
- Scientifically, QEB rows are most interpretable for fermionic/electronic and qubit-excitation-style ansatz construction. In bosonic or mixed registers they should be labeled as mapped-qubit excitation-pool diagnostics, not as family-native bosonic methods.

## 3. Recommended Table I method taxonomy

| Manuscript row | Benchmark question | Candidate/ansatz class | Include in class averages? | Notes |
|---|---|---|---:|---|
| HEA VQE | unstructured fixed ansatz | hardware-efficient layers | yes | Keep as generic hardware-facing baseline. |
| Family-informed VQE | conventional physics-informed fixed ansatz | class-specific fixed ansatz | yes | New fixed-ansatz row. See Sec. 4. |
| append-only ADAPT | same-pool ADAPT baseline | `full_meta`, tail append | yes | Canonical append-only adaptive growth under the same problem-local pool. |
| Qubit/QEB-ADAPT-VQE | operator-class comparator | QEB singles/doubles | yes if clearly labeled; optionally separate from same-pool average | Do not describe as pool-matched to SNAKE. |
| TETRIS-ADAPT-VQE | same-pool batching comparator | `full_meta` plus TETRIS batching | yes | Fair batching comparison requires same candidate universe. |
| Pos-Geo-ADAPT-VQE | same-pool geometry comparator | `full_meta` plus projected Fubini--Study natural-gradient selection/stopping, position search, and QNGD/SPSA refit | yes | Main Geo-family literature comparator. |
| Geo-ADAPT-VQE | operator-class geometry diagnostic | QEB singles/doubles plus projected Fubini--Study natural-gradient selection/stopping and QNGD-style refit | optional diagnostic only | Not a default Table-I row unless explicitly requested. |
| SNAKE | full method | problem-local acquisition pool with candidate-position scoring | yes | Main method. |

Do not merge TETRIS with CEO-style ADAPT. They are distinct algorithms.

## 4. Family-informed fixed VQE definition

The family-informed fixed VQE row is class-specific by design. It should be defined before running the sweep, not selected post hoc per instance.

Recommended policy:

| Hamiltonian family/class | Family-informed fixed ansatz | Rationale |
|---|---|---|
| Spinful fermionic lattice models | UCCSD-style fixed VQE | Standard chemistry/electronic-correlation baseline. Applies naturally to Hubbard, ionic Hubbard, extended Hubbard, and t--t' Hubbard when spinful sectors are present. |
| Molecular restricted closed shell | UCCSD VQE | Canonical molecular fixed-ansatz baseline. |
| Spinless fermion chains | family HVA or Hamiltonian-quadrature fixed VQE | Spinful UCCSD is not the natural object; use a number-conserving spinless family ansatz. |
| Boson-only chains | bosonic quadrature/HVA fixed VQE | There is no universal bosonic UCCSD; use displacement/squeezing/current/quadrature or Hamiltonian-block layers from the family pool. |
| Spin-boson / generalized Rabi | spin-boson family quadrature/dressing fixed VQE | Use the family-native spin/boson coupling blocks rather than QEB as the physics-informed row. |
| Hubbard--Holstein | Lang-Firsov plus lifted electronic UCCSD plus bosonic quadrature/dressing blocks, if available; otherwise report component baselines separately | The fair HH fixed baseline should include electronic correlation and phonon dressing. Lang-Firsov alone is meaningful but too narrow to represent the whole physics-informed fixed row. |

If implementation time forces a narrower first pass, use this fallback hierarchy:

1. fermionic/molecular: UCCSD;
2. spinless/bosonic: family HVA or Hamiltonian quadrature/block ansatz;
3. HH/mixed: existing Lang-Firsov and lifted-UCCSD rows, reported as separate component baselines until a composite family-informed row is implemented.

## 5. Paper-editing implications

The Table I caption and source map should say which rows are pool-matched and which are not.

Suggested caption language:

> Ground-state Pareto benchmark by Hamiltonian class. HEA and family-informed VQE are fixed-ansatz baselines. Append-only ADAPT, TETRIS-ADAPT, Pos-Geo-ADAPT-VQE, and SNAKE use the same problem-local candidate pool where compatible. Qubit/QEB-ADAPT-VQE is retained as a mapped-qubit excitation-pool comparator.

Suggested prose near the results table:

> The benchmark separates fixed-ansatz baselines from adaptive-selection rules and operator-class comparators. Append-only ADAPT, TETRIS-ADAPT, Pos-Geo-ADAPT-VQE, and SNAKE are evaluated over the problem-local pool whenever the method admits such a pool. Qubit/QEB-ADAPT-VQE is reported separately as a qubit-excitation-pool baseline because its defining feature is the operator class rather than the problem-local full-meta candidate universe.

Avoid saying:

- "all ADAPT competitors use the same pool" if QEB remains in the table;
- "QEB is a bosonic physics baseline";
- "TETRIS/CEO-style";
- "fixed ansatz VQE" as a single row if both HEA and family-informed VQE are present.

## 6. Code implementation checklist

Historical/completed checklist for the current comparator suite:

1. Rewire TETRIS-ADAPT benchmark-local row to use `full_meta`, not QEB singles/doubles.
2. Remove `static_geo_qubit_adapt_vqe` from the default Table-I suite; it is an invented diagnostic row, not a literature benchmark.
3. Keep `static_pos_geo_adapt_vqe` as the Geo-family Table-I comparator.
4. Rewire append-only ADAPT to use a Qiskit-compatible `full_meta` operator pool where feasible, or add a benchmark-local append-only full-meta statevector runner if Qiskit cannot accept the pool cleanly.
5. Keep Qubit/QEB-ADAPT on QEB singles/doubles and label it as an operator-class comparator.
6. Add a `static_family_informed_vqe` row for the physics-informed fixed ansatz policy in Sec. 4.
7. Keep HEA VQE as a separate fixed-ansatz row.
8. Update table-generation labels so the manuscript sees both fixed VQE rows: `HEA VQE` and `family-informed VQE`, with `Pos-Geo-ADAPT-VQE` as the sole Geo-family default row.
9. Smoke locally before CHTC. Do not submit a full CHTC batch until all row schemas, pool labels, and guardrails are validated.

## 7. Interpretation rule for future paper agents

When updating Table I, average rows only after confirming that the source records identify:

- method id;
- Hamiltonian class;
- pool/ansatz class;
- whether the row is fixed-ansatz, pool-matched ADAPT, or operator-class comparator;
- exact-reference usage as reporting-only;
- compiled depth/two-qubit proxy and shot proxy.

If old records come from `static_geo_qubit_adapt_vqe`, treat them as legacy diagnostic artifacts only. Do not copy them into Table I or into the `Pos-Geo-ADAPT-VQE` row.

## 8. Current bosonic SNAKE evidence note

This note is for the separate SNAKE/full-method row update, not for the current non-SNAKE comparator aggregate already inserted into Table I. Use the new Bose-Hubbard runs as the primary future bosonic SNAKE evidence source rather than the older harmonic-current placeholder.

Current CHTC cluster/status source:

- Cluster: `6335209`.
- Bosonic A: finished successfully; best trial `13`.
- Bosonic B: finished successfully; best trial `2`.
- Bose-Hubbard L2 and Bose-Hubbard L2 `u2` both report:
  - `ΔE ≈ 4.82e-13` for Bosonic A and `ΔE ≈ 4.18e-12` for Bosonic B;
  - two-qubit count `4`;
  - circuit depth `20`;
  - parameter count `1`;
  - shot proxy `9`.

Paper-editing implication:

- For the bosonic class row, the paper-facing table should cite/average the Bose-Hubbard L2 family records once the final aggregation script has recovered the artifacts.
- Do not treat harmonic-current results as the primary bosonic evidence if the Bose-Hubbard artifacts are available and schema-compatible.
- HH/mixed remains pending until the live `HH L2 nph1` SNAKE jobs finish or are intentionally stopped and recovered.
- Do not update SNAKE manuscript averages until the recovered artifacts are parsed by method id and Hamiltonian class.

## 9. Current comparator Table-I execution status

Current non-SNAKE comparator source for Table I before the QEB Geo rerun:

- CHTC smoke cluster: `6335354`; status: 13/13 benchmarked, no contract violations. These smoke records predate `static_geo_qeb_adapt_vqe`.
- CHTC full comparator cluster: `6335355`; status: 70/72 benchmarked, 0 unusable, 0 quality-nonpassing, 0 contract violations. These outputs predate the current default taxonomy; old `static_geo_qubit_adapt_vqe` rows are legacy geometry diagnostics and must not be promoted into Table I.
- Updated exact-bench catalog target after removing the invented `static_geo_qubit_adapt_vqe` default row: 72 comparator records and 14 smoke records, plus SNAKE/full-method run inputs managed by the SNAKE execution workflow.
- Intentionally missing/stopped rows: `static_table__molecular_restricted_closed_shell__molecular_restricted_closed_shell_h2o_sto3g_L7__static_hea_qiskit_vqe` and `static_table__molecular_restricted_closed_shell__molecular_restricted_closed_shell_lih_sto3g_L6__static_hea_qiskit_vqe`. These were the two large-molecule HEA rows and were stopped after roughly one hour because they were not converging on the table-building timescale.
- Records file: `chtc/phase3_optuna/input/generic_static_table_records.tsv`.
- Raw output root: `raw_outputs/chtc_phase3_optuna/generic_static_table/`.
- Validation summary: `raw_outputs/chtc_phase3_optuna/generic_static_table_full_output_check_final.json`.
- Manuscript-row aggregate: `raw_outputs/table_i_static_paper/table_i_static_results_summary.json`.
- CSV aggregate: `raw_outputs/table_i_static_paper/table_i_static_results_rows.csv`.
- Generated TeX row block: `raw_outputs/table_i_static_paper/table_i_static_claim_rows.tex`.

Paper I Table I (`tab:static_claims`) has been updated with the non-SNAKE comparator aggregate above while leaving the current SNAKE cells unchanged. Future SNAKE/full-method updates should touch only the SNAKE cells unless the comparator suite is rerun.

## 10. Normalized measurement-work status

The comparator rows now report measurement work through the normalized
`S_norm` schema:

```text
S_norm = N_H_eval + N_grad + N_metric + N_refit_eval
```

This is a reporting currency for estimator/probe work, not a physical shot
count. It regularizes the old raw `shots_total`, `shot_cost_proxy`, and
`measurement_shots_proxy` fields by separating energy evaluations, adaptive
gradient probes, metric probes, and refit evaluations.

SNAKE support artifacts now carry the same
`normalized_measurement_work_v1` metadata. Current SNAKE rows are marked
`S_norm_status = missing_component_breakdown` because the available support
artifact exposes raw scalar proxies but not the four explicit components. Their
`measurement_work_proxy` is therefore a visible raw fallback, not promoted
`S_norm`; future SNAKE rows may be promoted only when all four components are
present, finite, and nonnegative.

Paper-editing implication:

- Use `S_norm` / normalized measurement work for comparator rows with
  `S_norm_status = ok`.
- Do not compare raw SNAKE fallback values as normalized `S_norm`.
- If a SNAKE row is updated from a new run, require explicit
  `N_H_eval`, `N_grad`, `N_metric`, and `N_refit_eval` telemetry before filling
  the normalized measurement-work cell as apples-to-apples.

### 2026-07-09 HH three-method `S` policy

For the current Hubbard--Holstein three-method comparison, the visible table
column `S` is sourced from scoped `S_alg`, i.e. actual estimator-query work in
the displayed row scope, not from `S_fair`, `S_common_exposure`,
`S_beam_search_total`, `shots_total`, or any legacy scalar fallback.  The
intended SNAKE scope is the displayed accepted prefix on the winning branch
(`display_prefix` / `winner_lineage`): count all actual candidate
gradient/metric probes requested by the scoring phases along that prefix,
including probed candidates that were rejected within those phases, and count
the associated outer/refit Hamiltonian evaluations.  Do not charge losing
sibling beam branches or all-expanded search totals to the visible row `S`.

`S_fair` remains a diagnostic/common-exposure currency for explicit fairness
reports.  It may be reported alongside `S_alg` only when a support artifact or
user-approved table explicitly asks for that diagnostic, and it must not replace
the HH three-method visible `S` column by default.

## 11. 2026-06-13 first-pass comparator-fairness guardrails

This section is a support note for future agents/scripts, not manuscript prose.
It records the current audit interpretation so historical source comments are not
mistaken for the active table policy.

### Error-display policy

Current rendered Paper-I comparison tables should be read as raw/same-cutoff
absolute-error tables:

- Tables I--II: raw absolute or same-cutoff absolute energy error for the
  displayed ansatz, as stated by the current table captions and 2026-06-10
  repeat-enabled comparator support artifact.
- HH plateau Table III: same-cutoff ED error at the algorithmic working cutoff,
  with higher-cutoff ED values retained only as cutoff-resolution diagnostics.

Older machine-readable comments and older support JSONs may contain fields such
as `display_delta_e_source: legacy_*_minus_threshold`,
`display_delta_e_policy: max(0, abs_delta_e - target_abs_delta_e)`, or
`target_abs_delta_e`. Treat those as historical provenance unless a current
source map explicitly reactivates them. Do not copy target-subtracted values into
current rendered tables or current table-refresh scripts.

### SPSA implementation taxonomy

Use precise language when auditing SPSA fairness. Separate three questions:

1. **SPSA family:** two-point random-perturbation gradient estimate with
   decaying `a_k`/`c_k` schedules.
2. **SPSA engine/function:** the concrete optimizer function or library class
   that executes the update loop.
3. **SPSA parameterization/wrapper policy:** `a`, `c`, `alpha`, `gamma`, `A`,
   `maxiter`, seed, evaluation repeats, averaging, bounds/projection, restarts,
   step caps, and accept/reject or return policy.

Current evidence and future reruns should be described as SPSA-family matched,
with method-specific engines/wrappers and recorded parameterizations. As of the
2026-06-13 native-SPSA rerun patch, the primary exact-bench ADAPT
`optimizer_kind=spsa` refit path uses `src.quantum.spsa_optimizer.spsa_minimize`.
Locked artifacts generated before that patch may still carry the legacy
`exact_bench_spsa:energy_only_descent` / `_spsa_polish` label; treat those as
pre-native-rerun evidence and do not mix them with native-SPSA rerun rows.

| Row family | Current optimizer engine/wrapper | SPSA-specific audit interpretation |
|---|---|---|
| HEA VQE | `qiskit_algorithms.optimizers.SPSA` | External library SPSA engine; audit learning-rate/perturbation/maxiter/seed metadata. |
| family-informed VQE | `src.quantum.spsa_optimizer.spsa_minimize` (`repo_native_spsa`) | Repo-native SPSA engine over a fixed, class-chosen ansatz. |
| SNAKE/Route A when `adapt_inner_optimizer=SPSA` | `src.quantum.spsa_optimizer.spsa_minimize` through `pipelines/static_adapt/adapt_pipeline.py` | Same repo-native SPSA engine as other Phase3/SNAKE SPSA paths; audit route-specific schedule values. |
| append/TETRIS/Geo/Qubit-QEB exact-bench ADAPT variants | `src.quantum.spsa_optimizer.spsa_minimize` for new native-SPSA rerun rows; legacy artifacts may show `generic_static_adapt_variants.py::_spsa_polish` | Primary rerun path is the repo-native SPSA engine; legacy polish artifacts require explicit rerun/legacy labeling. |

When writing audit summaries, avoid the undefined phrase “identical SPSA
implementation.” Say which axis is matched. If only the family and budget/profile
are matched, say that. If the exact engine is matched, name the shared function.
If the concern is schedule fairness, compare the parameterization fields rather
than the surrounding ADAPT/VQE method logic.

### Repeat policy wording

Current Tables I--II adaptive comparator rows come from the repeat-enabled
Suite-B evidence path (`phase3_adapt_allow_repeats=true`) with fixed-horizon,
no-target stopping. The generic ADAPT variant code implements this as
with-replacement selection except the immediately previous label is blocked.
Avoid paraphrasing this as “immediate repeats allowed” unless a future source
actually changes that rule.
