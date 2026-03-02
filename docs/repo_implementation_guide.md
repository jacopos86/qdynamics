# Repo Implementation Guide v3 (HH-First)

Implementation-first guide for `Holstein_test` with Hubbard-Holstein (HH) as the primary production model and pure Hubbard as secondary validation context.

Audience assumption: physics and math are already understood; this document focuses on how physics is encoded in code, what invariants are enforced, and how to read artifacts/plots correctly.

## 1. Reader Contract And Review Goal (HH-First)

This guide answers one question: is the current implementation faithful and auditable?

What is in scope:

- exact codepath mapping for HH and Hubbard runs,
- invariants that must not drift,
- how VQE/ADAPT/Trotter/drive are actually implemented,
- what each JSON key and plot channel means,
- how to detect implementation bugs from artifact signatures.

What is intentionally out of scope:

- re-deriving Hubbard or HH theory from first principles,
- CLI catalogs beyond a tiny appendix,
- workflow/process material not tied to implementation correctness.

Canonical audit artifacts in this guide:

- HH primary static: `artifacts/json/hc_hh_L2_static_t1.0_U2.0_g1.0_nph1_strong.json`
- HH primary drive: `artifacts/json/hc_hh_L2_drive_J1_U2_g1_nph1_from_adapt.json`
- HH primary ADAPT: `artifacts/json/adapt_hh_L2_static_t1.0_U2.0_g1.0_nph1_paop_deep.json`
- Hubbard context static: `artifacts/json/hc_hubbard_L3_static_t1.0_U4.0_S128_heavy.json`
- Hubbard context drive: `artifacts/json/hc_hubbard_L4_drive_t1.0_U4.0_S256.json`

Review method used throughout:

1. State quantity/contract.
2. Point to implementation function family.
3. Point to JSON/plot channel.
4. State failure signature.

![Repository structure](repo_guide_assets/01_repo_structure_map.png)

![Import DAG](repo_guide_assets/02_internal_import_dag.png)

## 2. Non-Negotiable Invariants

### 2.1 Pauli alphabet invariant

Internal operator algebra uses `e/x/y/z`. Any `I/X/Y/Z` conversions are boundary-only formatting concerns.

Why this matters: labels are dictionary keys and graph nodes in several stages; mixed conventions silently break term-matching.

### 2.2 Pauli word ordering + bit indexing invariant

String convention: left-to-right is `q_(n-1)...q_0`; qubit 0 is rightmost in strings.

Statevector bit arithmetic still uses qubit 0 as least significant bit in integer basis index operations.

### 2.3 JW source-of-truth invariant

JW ladders should come from the established helper layer (`pauli_polynomial_class` path), not fresh ad-hoc rewrites.

### 2.4 Canonical `PauliTerm` source

Canonical class is `src/quantum/qubitization_module.py` with compatibility aliases only; no parallel class implementations.

### 2.5 Drive pass-through invariant

Compare pipeline forwards drive parameters; it should not reinterpret drive physics.

### 2.6 Safe-test invariant

The A=0 drive path should match no-drive trajectory behavior within threshold (`1e-10` in compare checks).

![Qubit ordering](repo_guide_assets/04_qubit_ordering_convention.png)

![Invariant evidence](repo_guide_assets/43_invariant_evidence_snippets.png)

![Safe-test logic](repo_guide_assets/11_safe_test_invariant_logic.png)

## 3. Current Repo Topology And Active Codepaths

Primary active code paths in this repo shape:

- hardcoded production pipelines:
  - `pipelines/hardcoded/hubbard_pipeline.py`
  - `pipelines/hardcoded/adapt_pipeline.py`
- qiskit archive validation/orchestration:
  - `pipelines/qiskit_archive/qiskit_baseline.py`
  - `pipelines/qiskit_archive/compare_hc_vs_qk.py`
- physics core:
  - `src/quantum/hubbard_latex_python_pairs.py`
  - `src/quantum/vqe_latex_python_pairs.py`
  - `src/quantum/hartree_fock_reference_state.py`
  - `src/quantum/drives_time_potential.py`
  - `src/quantum/operator_pools/polaron_paop.py`

Implementation split used in practice:

- HH production depth is in hardcoded path (+ ADAPT/pool variants).
- Qiskit path remains a comparator for supported Hubbard cases and shared drive reference semantics.
- Tests in `test/` are executable specification for many invariants, not just smoke checks.

![Hardcoded pipeline flow](repo_guide_assets/06_hardcoded_pipeline_flow.png)

![Qiskit pipeline flow](repo_guide_assets/07_qiskit_pipeline_static_vs_drive.png)

![Shared drive backend](repo_guide_assets/08_shared_drive_backend.png)

![Compare fanout](repo_guide_assets/09_compare_pipeline_fanout_fanin.png)

## 4. HH Hamiltonian Assembly And JW Contracts

### 4.1 Core decomposition implemented

The implemented HH model logic is assembled into PauliPolynomial terms and then traversed as label-coefficient maps:

$$
H = H_t + H_U + H_{\text{phonon}} + H_{e\text{-}ph} + H_v.
$$

For Hubbard-only runs, the reduced decomposition is used.

### 4.2 Number operator and label placement contract

Number operator convention remains:

$$
n_p = \frac{I - Z_p}{2},
$$

with Pauli-string placement consistent with `q_(n-1)...q_0` convention.

### 4.3 Term collection vs ordering

Collection stage accumulates duplicate labels into one coefficient map; ordering stage chooses traversal order (`native` vs `sorted`) for product-form exponentials.

Reordering changes finite-step Trotter error profile; it does not change Hamiltonian definition.

### 4.4 HH register expansion

HH introduces phonon register qubits in addition to fermion spin-orbital qubits.

![JW flow](repo_guide_assets/05_jw_mapping_flow.png)

![Operator model](repo_guide_assets/03_operator_layer_object_model.png)

![Term traversal/order](repo_guide_assets/24_term_traversal_vs_ordering.png)

![HH register layout](repo_guide_assets/47_hh_register_layout.png)

## 5. HF And Reference-State Construction (Hubbard + HH)

### 5.1 Hubbard HF construction

For `L` sites, fermion qubits are `N_q = 2L` with half-filling split:

$$
N_\uparrow = \lceil L/2 \rceil, \qquad N_\downarrow = \lfloor L/2 \rfloor.
$$

Placement is indexing-dependent (`blocked` vs `interleaved`) and implemented via bit-index helper + reference-state constructor.

### 5.2 HH reference construction

HH reference state includes fermionic half-filling pattern plus phonon vacuum encoding block; this is not the same as the pure Hubbard HF vector on `2L` qubits.

### 5.3 Basis index interpretation

Bitstring printed order and integer-basis bit tests are intentionally dual-view; both are required for correct occupancy extraction.

![HF blocked L2](repo_guide_assets/15_hf_blocked_L2_table.png)

![HF blocked L3](repo_guide_assets/16_hf_blocked_L3_table.png)

![HF blocked L4](repo_guide_assets/17_hf_blocked_L4_table.png)

![HF interleaved L2](repo_guide_assets/18_hf_interleaved_L2_table.png)

![HF interleaved L3](repo_guide_assets/19_hf_interleaved_L3_table.png)

![HF interleaved L4](repo_guide_assets/20_hf_interleaved_L4_table.png)

![Bit-index examples](repo_guide_assets/21_bit_index_place_value_examples.png)

![HH sector filtering](repo_guide_assets/48_hh_sector_filtering.png)

## 6. VQE Internals: Ansatz Families, Theta, Restarts

### 6.1 Objective implemented

VQE minimizes:

$$
E(\theta)=\langle \psi(\theta)|H|\psi(\theta)\rangle,
$$

with statevector backend in hardcoded path.

### 6.2 Theta semantics

`theta` is always a real vector, but dimension semantics differ by ansatz family.

- layerwise families: grouped parameterization,
- termwise families: one scalar per traversed generator term per repetition.

### 6.3 Inner optimizer and restarts

Hardcoded VQE path uses SciPy when available (default configurations commonly COBYLA in pipeline settings) and fallback search when absent.

Restart semantics:

1. sample independent initialization,
2. optimize objective,
3. keep lowest terminal energy.

### 6.4 Practical correctness checks

For meaningful comparisons, keep fixed:

- optimizer method,
- restart count,
- `maxiter`,
- seed,
- ansatz family/depth.

![Theta layout](repo_guide_assets/22_theta_vector_layout.png)

![Optimizer flow](repo_guide_assets/23_optimizer_restart_flow.png)

![Theta VQE vs ADAPT](repo_guide_assets/51_theta_vqe_vs_adapt.png)

![Function line spans](repo_guide_assets/41_function_line_span_map.png)

![Function call graph](repo_guide_assets/42_function_call_graph_focus.png)

## 7. ADAPT Internals And Pool Semantics (PAOP/LF)

### 7.1 ADAPT loop in code terms

Implemented ADAPT flow:

1. build pool,
2. prepare current state,
3. compute commutator-gradient score per candidate,
4. select best candidate (repeat-bias logic when repeats enabled),
5. finite-angle fallback scan when gradients are below threshold (if enabled),
6. re-optimize all parameters (COBYLA),
7. stop by gradient/energy/depth conditions.

### 7.2 Pool family semantics

HH pool families are materially distinct and should not be treated as aliases except where explicit alias rules are defined.

PAOP-LF additions introduce new channel structure (e.g., current-like odd channel, second-order even channel, full extensions).

### 7.3 Merge/dedup path

For HH with nonzero coupling in several branches, merged pool construction and polynomial-signature deduplication are explicit implementation steps.

### 7.4 Artifact interpretation impact

`adapt_vqe` payload fields (depth, selected operators, stop reason, pool type) must be read with the same weight as final energy when validating implementation behavior.

![ADAPT loop](repo_guide_assets/49_adapt_loop_gradient_fallback.png)

![Pool family map](repo_guide_assets/50_pool_family_map_hh_paop_lf.png)

## 8. Trotter Ordering Semantics And Non-Commutation

For finite step size, product ordering matters whenever terms do not commute.

Implemented symmetric Suzuki-2 style traversal uses forward/reverse passes over ordered labels with per-step sampling semantics.

Why step refinement helps:

- local BCH error terms scale with higher powers of $\Delta t$,
- reducing $\Delta t$ shrinks order-sensitive error terms,
- ordering effect approaches zero in the exact-limit regime.

This does not mean ordering is irrelevant at finite discretization; it means its induced error shrinks predictably with refinement.

![Suzuki-2 anatomy](repo_guide_assets/25_suzuki2_step_anatomy.png)

## 9. Drive Implementation, Reference Split, Safe-Test

### 9.1 Drive waveform implementation

Drive path is Gaussian-envelope sinusoid with spatial weighting patterns.

### 9.2 Static vs drive-enabled reference methods

- no-drive: static eigendecomposition reference branch,
- drive: piecewise exact numerical propagator path with `reference_method` metadata and step refinement via `exact_steps_multiplier`.

### 9.3 Safe-test contract

Drive enabled with `A=0` should match no-drive channels within threshold; this guards routing and numerical branch consistency.

### 9.4 Architecture boundaries

Compare path routes drive args and evaluates consistency metrics; hardcoded/qiskit sub-pipelines perform actual physics propagation.

![Drive waveform](repo_guide_assets/26_drive_waveform_decomposition.png)

![Drive spatial patterns](repo_guide_assets/27_drive_spatial_patterns.png)

![Reference split](repo_guide_assets/28_reference_propagator_split.png)

![Amplitude 6-run flow](repo_guide_assets/10_amplitude_comparison_six_runs.png)

## 10. Observable Computations And Trajectory Contracts

### 10.1 Site-0 occupations

`n_up_site0_*`, `n_dn_site0_*` channels come from basis-index occupancy extraction over probability amplitudes.

### 10.2 Doublon and staggered observables

Doublon is summed sitewise joint occupancy channel; staggered order is alternating-sign site density aggregate.

### 10.3 Energy channels

- static energy channels use static Hamiltonian expectation,
- total energy channels include instantaneous drive contribution when drive active.

### 10.4 Fidelity channel

Fidelity is projector-overlap against filtered ground-manifold subspace basis, not generic full-space overlap to a single eigenvector.

### 10.5 Schema differences

HH ADAPT payloads may use compact energy key names (`energy_exact`, `energy_trotter`) while HC VQE payloads include expanded static/total key families. Audit tooling must map both.

![Trajectory schema map](repo_guide_assets/29_trajectory_row_schema_map.png)

![Formula energy](repo_guide_assets/30_formula_legend_energy.png)

![Formula fidelity](repo_guide_assets/31_formula_legend_fidelity.png)

![Formula occupancy/doublon](repo_guide_assets/32_formula_legend_occupancy_doublon.png)

![HH trajectory key expansion](repo_guide_assets/53_hh_trajectory_key_expansion.png)

## 11. Plot Generation Internals And Interpretation

### 11.1 Pipeline plot assembly behavior

Pipeline PDFs combine trajectory arrays into page families (summary, energy overlays, occupancy channels, drive pages, error pages, provenance/manifest sections).

### 11.2 Correct interpretation rules

- compare like-for-like branches,
- distinguish static from total energies in drive runs,
- treat branch provenance (`exact_gs`, `paop`, `hva`) explicitly,
- check fidelity with its subspace definition in mind.

### 11.3 Common misreads to avoid

- mistaking ordering effects for Hamiltonian-definition changes,
- mixing branch labels across paop/hva/legacy tracks,
- interpreting `A=0` drive equivalence failures as physical instead of implementation routing issues.

![Plot meaning map](repo_guide_assets/46_plot_meaning_map.png)

![Branch provenance map](repo_guide_assets/52_branch_provenance_map.png)

![Plot/formula audit panel](repo_guide_assets/54_plot_formula_audit_panel.png)

## 12. Artifact-Grounded Audits (HH Primary, Hubbard Context)

### 12.1 HH primary case A (static)

Audit source: `hc_hh_L2_static_t1.0_U2.0_g1.0_nph1_strong.json`

Key checks:

- consistency of static and total channels in no-drive mode,
- physically bounded occupancy channels,
- sensible fidelity and energy error envelopes.

![HH static energy audit](repo_guide_assets/33_artifact_L2_energy_audit.png)

![HH static site0 audit](repo_guide_assets/34_artifact_L2_site0_audit.png)

### 12.2 HH primary case B (drive)

Audit source: `hc_hh_L2_drive_J1_U2_g1_nph1_from_adapt.json`

Key checks:

- static vs total energy divergence when drive active,
- branch-expanded paop/hva channel coherence,
- drive metadata consistency with behavior.

![HH drive energy audit](repo_guide_assets/35_artifact_L3_energy_audit.png)

![HH drive site0 audit](repo_guide_assets/36_artifact_L3_site0_audit.png)

### 12.3 HH primary case C (ADAPT)

Audit source: `adapt_hh_L2_static_t1.0_U2.0_g1.0_nph1_paop_deep.json`

Key checks:

- compact schema decoding correctness,
- ADAPT depth/selection behavior consistency with trajectory quality,
- no contract break in mapped energy channels.

![HH ADAPT energy audit](repo_guide_assets/37_artifact_L4_total_energy_drive_audit.png)

![HH ADAPT site0/fidelity audit](repo_guide_assets/38_artifact_L4_drive_waveform_audit.png)

![HH ADAPT error audit](repo_guide_assets/39_artifact_L4_reference_method_audit.png)

### 12.4 Hubbard context cases

Context sources:

- `hc_hubbard_L3_static_t1.0_U4.0_S128_heavy.json`
- `hc_hubbard_L4_drive_t1.0_U4.0_S256.json`

Purpose:

- maintain continuity with prior validation baseline,
- sanity-check shared kernels and interpretation rules outside HH-primary trio.

![Canonical metrics table](repo_guide_assets/44_canonical_artifact_metrics_table.png)

![Case summary](repo_guide_assets/40_case_comparison_summary.png)

## 13. Extension Safety: Safe And High-Risk Edits

Safe edits usually preserve:

- Pauli/order/JW invariants,
- canonical operator-core semantics,
- drive pass-through and safe-test contracts,
- trajectory key meaning and manifest structure.

High-risk edits include:

- changing indexing/bit mapping without synchronized observable updates,
- rewriting number/JW logic ad hoc,
- adding drive knobs without full pipeline surfacing,
- changing existing JSON key semantics under old names.

Minimal pre-merge checks for implementation changes:

1. regenerate canonical guide assets,
2. verify evidence anchors still resolve,
3. run `pytest -q`,
4. inspect safe-test behavior in drive-related changes,
5. inspect metrics deltas on HH primary cases.

![Extension playbook](repo_guide_assets/14_extension_playbook_decision_tree.png)

![Quality gate summary](repo_guide_assets/45_quality_gate_summary.png)

![Test contract coverage](repo_guide_assets/13_test_contract_coverage.png)

## 14. Ultra-Brief Run Appendix

Build from repo root:

```bash
pipelines/shell/build_guide.sh
```

It regenerates diagrams/summary JSON, runs deterministic checks, snapshots `pytest -q`, and rebuilds:

- `docs/Repo implementation guide.PDF`
- `docs/repo_guide_assets/repo_guide_summary.json`
- `docs/repo_guide_assets/repo_guide_artifact_metrics.json`

For implementation review, start with sections 5, 7, 9, 10, and 12.

## Appendix A. Figure Atlas

![01](repo_guide_assets/01_repo_structure_map.png)
![02](repo_guide_assets/02_internal_import_dag.png)
![03](repo_guide_assets/03_operator_layer_object_model.png)
![04](repo_guide_assets/04_qubit_ordering_convention.png)
![05](repo_guide_assets/05_jw_mapping_flow.png)
![06](repo_guide_assets/06_hardcoded_pipeline_flow.png)
![07](repo_guide_assets/07_qiskit_pipeline_static_vs_drive.png)
![08](repo_guide_assets/08_shared_drive_backend.png)
![09](repo_guide_assets/09_compare_pipeline_fanout_fanin.png)
![10](repo_guide_assets/10_amplitude_comparison_six_runs.png)
![11](repo_guide_assets/11_safe_test_invariant_logic.png)
![12](repo_guide_assets/12_artifact_json_contract_map.png)
![13](repo_guide_assets/13_test_contract_coverage.png)
![14](repo_guide_assets/14_extension_playbook_decision_tree.png)
![15](repo_guide_assets/15_hf_blocked_L2_table.png)
![16](repo_guide_assets/16_hf_blocked_L3_table.png)
![17](repo_guide_assets/17_hf_blocked_L4_table.png)
![18](repo_guide_assets/18_hf_interleaved_L2_table.png)
![19](repo_guide_assets/19_hf_interleaved_L3_table.png)
![20](repo_guide_assets/20_hf_interleaved_L4_table.png)
![21](repo_guide_assets/21_bit_index_place_value_examples.png)
![22](repo_guide_assets/22_theta_vector_layout.png)
![23](repo_guide_assets/23_optimizer_restart_flow.png)
![24](repo_guide_assets/24_term_traversal_vs_ordering.png)
![25](repo_guide_assets/25_suzuki2_step_anatomy.png)
![26](repo_guide_assets/26_drive_waveform_decomposition.png)
![27](repo_guide_assets/27_drive_spatial_patterns.png)
![28](repo_guide_assets/28_reference_propagator_split.png)
![29](repo_guide_assets/29_trajectory_row_schema_map.png)
![30](repo_guide_assets/30_formula_legend_energy.png)
![31](repo_guide_assets/31_formula_legend_fidelity.png)
![32](repo_guide_assets/32_formula_legend_occupancy_doublon.png)
![33](repo_guide_assets/33_artifact_L2_energy_audit.png)
![34](repo_guide_assets/34_artifact_L2_site0_audit.png)
![35](repo_guide_assets/35_artifact_L3_energy_audit.png)
![36](repo_guide_assets/36_artifact_L3_site0_audit.png)
![37](repo_guide_assets/37_artifact_L4_total_energy_drive_audit.png)
![38](repo_guide_assets/38_artifact_L4_drive_waveform_audit.png)
![39](repo_guide_assets/39_artifact_L4_reference_method_audit.png)
![40](repo_guide_assets/40_case_comparison_summary.png)
![41](repo_guide_assets/41_function_line_span_map.png)
![42](repo_guide_assets/42_function_call_graph_focus.png)
![43](repo_guide_assets/43_invariant_evidence_snippets.png)
![44](repo_guide_assets/44_canonical_artifact_metrics_table.png)
![45](repo_guide_assets/45_quality_gate_summary.png)
![46](repo_guide_assets/46_plot_meaning_map.png)
![47](repo_guide_assets/47_hh_register_layout.png)
![48](repo_guide_assets/48_hh_sector_filtering.png)
![49](repo_guide_assets/49_adapt_loop_gradient_fallback.png)
![50](repo_guide_assets/50_pool_family_map_hh_paop_lf.png)
![51](repo_guide_assets/51_theta_vqe_vs_adapt.png)
![52](repo_guide_assets/52_branch_provenance_map.png)
![53](repo_guide_assets/53_hh_trajectory_key_expansion.png)
![54](repo_guide_assets/54_plot_formula_audit_panel.png)

## Appendix B. Evidence Anchors

Authoritative generated anchors are emitted to:

- `docs/repo_guide_assets/repo_guide_summary.json`

Key target function families tracked include:

- hardcoded dynamics trajectory kernels,
- VQE minimization and HF constructors,
- ADAPT core and pool constructors,
- drive waveform/build helpers,
- PAOP/LF pool internals.

## Appendix C. Metrics Snapshot Contract

Authoritative generated metrics are emitted to:

- `docs/repo_guide_assets/repo_guide_artifact_metrics.json`

Interpret this JSON as the canonical computed view of:

- HH primary-case validity,
- Hubbard context-case validity,
- trajectory schema mapping used during auditing,
- cross-case summary values rendered into guide figures.
