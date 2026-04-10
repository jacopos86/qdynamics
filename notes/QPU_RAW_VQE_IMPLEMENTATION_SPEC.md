# QPU Raw VQE / ADAPT-VQE Implementation Spec

Date: 2026-03-26
Status: planning spec, repo-specific
Scope: HH real-QPU VQE + ADAPT-VQE only
Non-scope: time-dynamics, realtime controller migration, driven propagation, full inner-loop QPU reoptimization

## 0. Executive decision

We should **not** continue centering the repo’s real-QPU architecture on `EstimatorV2`.

We also should **not** treat literal `backend.run()` as the invariant IBM production contract anymore.

### Verified external platform constraint
- IBM announced on **December 17, 2024** that `backend.run()` support in Qiskit Runtime was being dropped server-side around **January 7, 2025**.
- As of **March 26, 2026**, the current `IBMBackend` docs say `run()` is **no longer supported**.
- IBM’s Runtime migration docs and V2 primitive docs indicate that **`SamplerV2` is the supported measured-circuit raw-shot bridge** on current IBM QPUs.

### Repo decision from that constraint
For this repo, the canonical target is:
- **raw-shot-first execution**, not estimator-first execution
- **`SamplerV2` on current IBM Runtime** as the production IBM transport
- **`backend.run()` only as an optional transport** when a non-IBM or legacy backend genuinely supports it

That preserves the spirit of `notes/HH_QPU_RAW_BRIEF.md`:
- canonical artifact = raw backend-level data
- superior post-processing happens downstream of raw acquisition
- `EstimatorV2` becomes explicit legacy compatibility only

---

## 1. Current repo surface to migrate

### 1.1 Shared execution choke point
Primary file:
- `pipelines/exact_bench/noise_oracle_runtime.py`

Current key symbols:
- `OracleConfig`
- `NoiseBackendInfo`
- `RuntimeJobRecord`
- `build_runtime_layout_circuit(...)`
- `ExpectationOracle`
- `_run_sampler_job(...)`
- `ExpectationOracle._get_runtime_sampler(...)`
- `ExpectationOracle.collect_backend_scheduled_group_sample(...)`
- `ExpectationOracle.collect_runtime_group_sample(...)`

Current issue:
- the public runtime surface is still scalar-expectation-first via `ExpectationOracle.evaluate(...)`
- raw grouped sampling exists, but it is secondary and not the canonical artifact path

### 1.2 Conventional scaffold-based VQE runtime surface
Primary file:
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py`

Current key symbols:
- `_evaluate_locked_imported_circuit_energy(...)`
- `_run_locked_imported_runtime_phase_evals(...)`
- `_run_imported_fixed_scaffold_runtime_energy_only(...)`
- `_run_imported_fixed_scaffold_runtime_energy_only_mode_isolated(...)`

Current issue:
- this is the active real-runtime fixed-scaffold energy-only route
- it is explicitly `EstimatorV2`-centered

### 1.3 ADAPT-VQE runtime scout surface
Primary file:
- `pipelines/hardcoded/adapt_pipeline.py`

Current key symbols:
- `Phase3OracleGradientConfig`
- `_phase3_oracle_runtime_bindings()`
- `_validate_phase3_oracle_gradient_config(...)`
- `_phase3_oracle_gradient_config_payload(...)`
- `_run_hardcoded_adapt_vqe(...)`
- nested `_phase3_oracle_gradient_scout(...)`
- `parse_args(...)`
- `main(...)`

Current issue:
- phase3 oracle mode is a scouting layer only
- runtime/raw acquisition is not yet the canonical surface
- reoptimization intentionally remains exact/local and should stay that way in this slice

### 1.4 Shared scaffold circuit-build surface
Primary file:
- `pipelines/hardcoded/adapt_circuit_execution.py`

Current key symbols:
- `append_reference_state(...)`
- `append_pauli_rotation_exyz(...)`
- `build_ansatz_circuit(...)`
- `build_structural_ansatz_circuit(...)`

Current issue:
- only concrete-angle circuits are exposed
- there is no structure-level parameterized plan for compile reuse across repeated theta binds

### 1.5 Workflow / CLI surface
Primary files:
- `pipelines/hardcoded/hh_staged_cli_args.py`
- `pipelines/hardcoded/hh_staged_noise_workflow.py`

Current key symbols:
- `add_staged_hh_noise_args(...)`
- `FixedScaffoldRuntimeEnergyOnlyConfig`
- `resolve_staged_hh_noise_config(...)`
- `run_staged_hh_noise(...)`

Current issue:
- the existing imported fixed-scaffold runtime baseline is named and wired as an Estimator runtime route

### 1.6 Conventional replay/orchestration surface
Primary file:
- `pipelines/hardcoded/hh_vqe_from_adapt_family.py`

Current key symbols:
- `build_replay_scaffold_context(...)`
- `run(...)`

Repo-specific note:
- this file should remain replay/scaffold orchestration
- **do not** move core Qiskit circuit construction ownership here in phase 1
- use it as a source of scaffold metadata / layout / theta provenance only

---

## 2. Architecture to implement

## 2.1 Core rule
Make the canonical runtime artifact:
- **raw measured grouped-circuit records**
- persisted durably as sidecars
- reduced later to energies / gradients / uncertainties / mitigated variants

Do **not** make the canonical runtime artifact a compressed expectation scalar.

## 2.2 Public execution shape
Implement a raw-first three-stage surface inside `noise_oracle_runtime.py`:

1. **plan**
   - decide basis groups
   - produce parameterized measured templates
   - stamp semantic metadata

2. **execute**
   - run via transport
   - current IBM route: `SamplerV2`
   - optional transport: direct `backend.run()` only if actually supported

3. **reduce**
   - raw counts -> mean / stderr / uncertainty-aware scout values
   - produce optional derived mitigated variants

## 2.3 Repo-specific class/function naming
Use repo-native names rather than a brand-new subsystem package.

### Add to `pipelines/exact_bench/noise_oracle_runtime.py`
Proposed new symbols:
- `RawObservableEstimate`
- `RawMeasurementRecord`
- `RawExecutionBundle`
- `RawMeasurementOracle`
- `_resolve_runtime_execution_target(...)`
- `_runtime_backend_supports_direct_run(...)`
- `_run_raw_transport_job(...)`
- `_write_raw_measurement_record(...)`
- `_group_observable_terms_by_qwc_basis(...)`
- `_reduce_grouped_counts_to_observable(...)`

### Keep and quarantine
Keep `ExpectationOracle`, but mark runtime estimator use as legacy only.

## 2.4 Transport policy

### IBM Runtime policy
- Preferred IBM runtime transport in this repo: `sampler_v2`
- `backend.run()` is **not** the default IBM transport because current IBM Runtime does not support it

### General transport policy
Expose transport choice explicitly:
- `auto`
- `sampler_v2`
- `backend_run`

Resolution rules:
- if backend is current IBM Runtime backend -> choose `sampler_v2`
- if non-IBM backend advertises real direct run support -> allow `backend_run`
- if explicit unsupported transport is requested -> fail fast

## 2.5 Compile reuse rule
Compile reuse must be keyed by:
- scaffold structure
- reference state digest
- measurement basis label
- backend / target snapshot
- transpile seed / optimization level

Counts reuse must **not** cross changed parameter points.

Therefore:
- conventional fixed-scaffold VQE can reuse compiled templates across many theta evaluations
- ADAPT phase3 can reuse compiled templates across `plus` / `minus` for the **same candidate structure**
- no sample reuse across different theta values

---

## 3. Exact file-by-file implementation plan

## 3.1 `pipelines/hardcoded/adapt_circuit_execution.py`

### Add
- `@dataclass(frozen=True) class ParameterizedAnsatzCircuit`
  - fields:
    - `layout`
    - `nq`
    - `circuit`
    - `parameters`
    - `structural_key`
    - `reference_state_key`
- `build_parameterized_ansatz_circuit(...)`
- `bind_parameterized_ansatz_circuit(...)`

### Modify
- `build_ansatz_circuit(...)`
  - reimplement on top of `build_parameterized_ansatz_circuit(...)` + bind helper
- `build_structural_ansatz_circuit(...)`
  - keep contract stable, but route through the parameterized builder internally

### Do not change
- Pauli ordering conventions
- reference-state semantics
- operator-layer imports

### Why this file owns the parameterized builder
This file is already the shared Qiskit boundary for logical ADAPT scaffolds. It is the right place for reusable parameterized scaffold execution plans.

---

## 3.2 `pipelines/exact_bench/noise_oracle_runtime.py`

### Modify `OracleConfig`
Add fields:
- `execution_surface: str = "expectation_v1"`
- `raw_transport: str = "auto"`
- `raw_store_memory: bool = False`
- `raw_grouping_mode: str = "qwc_basis_cover_reuse"`
- `raw_artifact_path: str | None = None`

### Add dataclasses
- `RawObservableEstimate`
- `RawMeasurementRecord`
- `RawExecutionBundle`

### Add class
- `RawMeasurementOracle`

### `RawMeasurementOracle` responsibilities
- own backend/session/sampler handles
- own compile cache for measured templates
- submit grouped measurement jobs
- emit normalized raw records
- reduce grouped raw data into observable estimates

### Add helpers
- `_resolve_runtime_execution_target(...)`
- `_runtime_backend_supports_direct_run(...)`
- `_run_raw_transport_job(...)`
- `_write_raw_measurement_record(...)`
- `_group_observable_terms_by_qwc_basis(...)`
- `_reduce_grouped_counts_to_observable(...)`

### Modify legacy paths
Keep:
- `ExpectationOracle`
- `_run_sampler_job(...)`
- `ExpectationOracle._get_runtime_sampler(...)`
- `ExpectationOracle.collect_backend_scheduled_group_sample(...)`
- `ExpectationOracle.collect_runtime_group_sample(...)`

But:
- mark runtime estimator calls with explicit legacy metadata
- never silently route raw-first users back through estimator mode

### Explicit runtime metadata additions
Every raw record emitted from this file must include:
- transport actually used
- backend name
- compile signature
- job/session identifiers
- logical/physical qubit maps
- requested mitigation policy
- resolved runtime option policy
- stable semantic tags

---

## 3.3 `pipelines/exact_bench/hh_noise_robustness_seq_report.py`

### Keep as legacy
- `_evaluate_locked_imported_circuit_energy(...)`
- `_run_locked_imported_runtime_phase_evals(...)`
- `_run_imported_fixed_scaffold_runtime_energy_only(...)`

These remain the explicit Estimator legacy baseline.

### Add
- `_evaluate_locked_imported_circuit_raw_energy(...)`
- `_run_imported_fixed_scaffold_runtime_raw_baseline(...)`
- `_run_imported_fixed_scaffold_runtime_raw_baseline_mode_isolated(...)`

### Behavior of new raw helper
`_evaluate_locked_imported_circuit_raw_energy(...)` should:
- build `SparsePauliOp` Hamiltonian as today
- call `RawMeasurementOracle`
- compute ideal comparator separately using existing ideal path
- return summary fields compatible with current reporting shape:
  - `noisy_mean`
  - `noisy_stderr`
  - `ideal_mean`
  - `ideal_stderr`
  - `delta_mean`
  - `delta_stderr`
- also return raw route metadata:
  - `execution_surface`
  - `raw_transport`
  - `raw_sidecar_path`
  - `sample_record_count`
  - `compile_signature`

### Behavior of new route wrapper
`_run_imported_fixed_scaffold_runtime_raw_baseline(...)` should:
- mirror current imported fixed-scaffold route resolution
- use raw route only
- disable runtime DD-probe / runtime-ZNE follow-up in phase 1
- emit sidecar references in the top-level payload

---

## 3.4 `pipelines/hardcoded/adapt_pipeline.py`

### Modify dataclass
Extend `Phase3OracleGradientConfig` with:
- `runtime_raw_transport: str = "auto"`
- `runtime_session_policy: str = "require_session"`
- `seed_transpiler: int | None = 0`
- `transpile_optimization_level: int = 1`
- `raw_store_memory: bool = False`

### Modify functions
- `_phase3_oracle_runtime_bindings()`
  - add `RawMeasurementOracle`
  - add parameterized circuit builder symbols
- `_validate_phase3_oracle_gradient_config(...)`
  - allow runtime raw route
  - reject unsupported mitigation combinations for phase 1
- `_phase3_oracle_gradient_config_payload(...)`
  - include execution surface + raw transport metadata
- `_run_hardcoded_adapt_vqe(...)`
  - thread raw-sidecar writer callback / path plumbing
- nested `_phase3_oracle_gradient_scout(...)`
  - use parameterized candidate scaffold plans
  - execute plus/minus probes through `RawMeasurementOracle`
  - compute gradient and propagated stderr from raw reductions
- `parse_args(...)`
  - add runtime raw flags
- `main(...)`
  - plumb new flags into `Phase3OracleGradientConfig`

### Phase-1 ADAPT rule
Keep reoptimization backend unchanged.
- selection/scouting uses raw runtime when requested
- inner optimization remains exact/local

That preserves current repo semantics.

---

## 3.5 `pipelines/hardcoded/hh_staged_cli_args.py`

### Modify `add_staged_hh_noise_args(...)`

### Keep existing legacy flag
Keep:
- `--include-fixed-scaffold-runtime-energy-only-baseline`

But change help text to say:
- legacy `EstimatorV2` compatibility baseline

### Add raw runtime flags
Add:
- `--include-fixed-scaffold-runtime-raw-baseline`
- `--fixed-scaffold-runtime-raw-transport {auto,sampler_v2,backend_run}`
- `--fixed-scaffold-runtime-raw-store-memory`
- `--fixed-scaffold-runtime-raw-grouping-mode qwc_basis_cover_reuse`
- `--fixed-scaffold-runtime-artifact-dir PATH` (optional; defaults near output json)

### Add phase3 raw flags
Add:
- `--phase3-oracle-runtime-raw-transport {auto,sampler_v2,backend_run}`
- `--phase3-oracle-runtime-session-policy {prefer_session,require_session,backend_only}`
- `--phase3-oracle-seed-transpiler INT`
- `--phase3-oracle-transpile-optimization-level INT`
- `--phase3-oracle-raw-store-memory`

### Keep but narrow semantics
Keep existing phase3 oracle mitigation flags, but phase-1 raw runtime validation should force:
- `mitigation=none`
- symmetry mitigation only as offline reduction / derived artifact, not live runtime primitive resilience

---

## 3.6 `pipelines/hardcoded/hh_staged_noise_workflow.py`

### Add dataclass
- `FixedScaffoldRuntimeRawConfig`

Fields:
- `enabled`
- `subject_kind`
- `noise_mode`
- `execution_surface`
- `raw_transport`
- `raw_store_memory`
- `raw_grouping_mode`
- `mitigation_config`
- `symmetry_mitigation_config`
- `runtime_session_config`
- `transpile_optimization_level`
- `seed_transpiler`
- `artifact_dir`

### Modify
- `resolve_staged_hh_noise_config(...)`
  - resolve the new raw route
  - keep `FixedScaffoldRuntimeEnergyOnlyConfig` as legacy route
  - reject enabling both raw and legacy imported runtime baselines simultaneously
- `run_staged_hh_noise(...)`
  - dispatch raw baseline vs legacy baseline explicitly

### Explicit phase-1 policy
For imported fixed-scaffold raw runtime baseline:
- real backend required
- fake backend not allowed
- runtime DD probe disabled
- runtime final ZNE audit disabled
- mitigation on acquisition path = none
- symmetry mitigation stored as requested post-processing recipe, not inline primitive resilience

---

## 3.7 `pipelines/hardcoded/hh_vqe_from_adapt_family.py`

### Phase-1 edit policy
No required phase-1 implementation move here.

### Allowable small edit only if needed
If later needed, add a tiny helper for stable scaffold provenance extraction, but do **not** move core Qiskit circuit build logic out of `adapt_circuit_execution.py`.

### Why
This file is currently replay/scaffold orchestration and should remain so in this migration slice.

---

## 3.8 `pipelines/qiskit_backend_tools.py`

### Add helper utilities
- `backend_supports_direct_run(...)`
- `snapshot_backend_target(...)`
- `build_compile_signature(...)`
- `extract_layout_maps(...)`
- `normalize_runtime_job_metadata(...)`

### Keep existing utilities
- `compile_circuit_for_backend(...)`
- `safe_circuit_depth(...)`
- `compiled_gate_stats(...)`

### Role of this file
This file should become the normalized backend/provenance helper layer, not the raw execution owner.

---

## 3.9 `pipelines/hardcoded/hh_realtime_measurement.py`

### Phase-1 rule
Do not migrate the realtime controller in this slice.

### Small compatibility edit allowed
If needed, align raw grouped sample normalization with the new canonical record fields so future controller migration is easier.

Possible symbols to modify only if required:
- `BackendScheduledRawGroupPool`
- `controller_oracle_supports_raw_group_sampling(...)`

---

## 3.10 `pipelines/run_guide.md`

### Update docs to reflect reality
Add a section stating:
- canonical real-QPU HH VQE route is now raw-shot-first
- current IBM production transport is `SamplerV2`
- direct `backend.run()` is optional only where supported
- `--include-fixed-scaffold-runtime-energy-only-baseline` is legacy estimator compatibility
- new raw baseline route is preferred

Also add example commands for:
- imported fixed-scaffold raw runtime baseline
- ADAPT phase3 raw runtime scout

---

## 4. Canonical raw artifact contract

## 4.1 Storage model
Use both:
- top-level run manifest JSON
- NDJSON raw sidecar as canonical append-only measurement log
- optional compressed shot-memory sidecars only when requested

## 4.2 Path convention
For a run with `output_json=.../foo.json`:
- manifest remains `foo.json`
- raw sidecar:
  - `foo.raw_groups.ndjson.gz` for fixed-scaffold raw baseline
  - `foo.phase3_raw_groups.ndjson.gz` for ADAPT phase3 raw scouting

## 4.3 Required per-record fields
Each raw record must include:
- schema version
- run id / evaluation id
- surface kind (`fixed_scaffold_vqe_energy`, `adapt_phase3_gradient`)
- semantic tags:
  - candidate id
  - probe sign
  - group id
  - basis label
- circuit digests:
  - structural scaffold key
  - compiled circuit key
- parameter values
- measured logical qubits
- measured physical qubits
- logical->physical and physical->logical maps
- transpile seed / optimization level
- compile signature
- backend snapshot
- call path used
- job id / session id / timestamps / status
- shots requested / shots completed
- counts
- optional memory ref
- requested mitigation recipe
- raw runtime metadata blob
- lineage / top-up ancestry

## 4.4 Derived artifacts
Mitigated outputs must not overwrite raw records.

Derived artifacts should live separately and point back to:
- raw record ids
- calibration artifact ids
- mitigation recipe digest
- code revision

---

## 5. Mitigation stack for phase 1

## 5.1 Acquisition-time defaults
For phase 1 raw runtime VQE / ADAPT on current IBM QPU:
- acquisition path remains raw
- no estimator resilience stack
- default acquisition mitigation = none
- optional DD / gate twirling may be added later, but are not blocking for the first landing

## 5.2 First offline mitigation stack
Implement this order in reduction/post-processing, not in raw record mutation:

1. symmetry verification diagnostics
2. symmetry postselection / projector renormalization derived result
3. offline readout mitigation derived result
4. readout + symmetry combined derived result
5. uncertainty propagation for gradient ranking

## 5.3 Why this order
It provides the highest immediate value while preserving raw provenance.

It also fits the repo’s existing symmetry mitigation vocabulary:
- `off`
- `verify_only`
- `postselect_diag_v1`
- `projector_renorm_v1`

---

## 6. Rollout order

## Phase 1
Implement raw infrastructure only:
- `adapt_circuit_execution.py`
- `noise_oracle_runtime.py`
- tests for raw planner/executor/reducer

## Phase 2
Migrate imported fixed-scaffold runtime baseline:
- `hh_noise_robustness_seq_report.py`
- `hh_staged_cli_args.py`
- `hh_staged_noise_workflow.py`
- `run_guide.md`

## Phase 3
Migrate ADAPT phase3 scout:
- `adapt_pipeline.py`
- sidecar plumbing
- uncertainty-aware reduction

## Phase 4
Add offline mitigation tooling and optional acquisition suppressors:
- readout mitigation artifacts
- symmetry-derived artifacts
- optional DD / gate twirling policies

## Explicitly deferred
- time-dynamics raw-QPU migration
- realtime controller migration
- full inner-loop QPU optimization
- broad ZNE/PEC/PEA integration
- heavy methods like virtual distillation / CDR

---

## 7. Validation / regression plan

## 7.1 Tests to edit first
- `test/test_hh_noise_oracle_runtime.py`
- `test/test_adapt_vqe_integration.py`
- `test/test_hh_realtime_measurement.py`
- `test/test_hh_staged_noise_workflow.py`

## 7.2 Tests to add

### Raw runtime layer
- transport selection: IBM runtime -> `sampler_v2`
- unsupported `backend_run` on IBM runtime -> fail fast
- compile signature determinism for repeated binds
- raw record schema completeness
- grouped reduction parity on synthetic counts

### Fixed-scaffold runtime route
- raw baseline flag dispatches to raw helper
- legacy estimator flag dispatches to legacy helper
- payload contains raw sidecar path and transport metadata

### ADAPT phase3 route
- plus/minus raw reductions produce correct finite-difference sign
- raw scout does not alter exact/local reoptimization backend
- sidecar records contain candidate/probe semantic tags

## 7.3 Recommended test command
```bash
pytest -q \
  test/test_hh_noise_oracle_runtime.py \
  test/test_adapt_vqe_integration.py \
  test/test_hh_realtime_measurement.py \
  test/test_hh_staged_noise_workflow.py
```

---

## 8. Immediate edit list

### Files to edit now
- `pipelines/hardcoded/adapt_circuit_execution.py`
- `pipelines/exact_bench/noise_oracle_runtime.py`
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py`
- `pipelines/hardcoded/adapt_pipeline.py`
- `pipelines/hardcoded/hh_staged_cli_args.py`
- `pipelines/hardcoded/hh_staged_noise_workflow.py`
- `pipelines/qiskit_backend_tools.py`
- `pipelines/run_guide.md`
- `test/test_hh_noise_oracle_runtime.py`
- `test/test_adapt_vqe_integration.py`
- `test/test_hh_realtime_measurement.py`
- `test/test_hh_staged_noise_workflow.py`

### Files not to edit in phase 1 unless forced
- `pipelines/hardcoded/hh_vqe_from_adapt_family.py`
- `pipelines/hardcoded/hh_realtime_measurement.py`

---

## 9. Concise implementation plan

1. add parameterized scaffold plan in `adapt_circuit_execution.py`
2. add raw planner/executor/reducer in `noise_oracle_runtime.py`
3. wire imported fixed-scaffold raw baseline
4. wire ADAPT phase3 raw scout
5. add raw sidecars + provenance
6. update tests
7. update `run_guide.md`

Unresolved questions:
- none blocking for code start
- only external constraint already resolved: on current IBM Runtime, use `SamplerV2`, not literal `backend.run()`

Files to edit:
- `pipelines/hardcoded/adapt_circuit_execution.py`
- `pipelines/exact_bench/noise_oracle_runtime.py`
- `pipelines/exact_bench/hh_noise_robustness_seq_report.py`
- `pipelines/hardcoded/adapt_pipeline.py`
- `pipelines/hardcoded/hh_staged_cli_args.py`
- `pipelines/hardcoded/hh_staged_noise_workflow.py`
- `pipelines/qiskit_backend_tools.py`
- `pipelines/run_guide.md`
- `test/test_hh_noise_oracle_runtime.py`
- `test/test_adapt_vqe_integration.py`
- `test/test_hh_realtime_measurement.py`
- `test/test_hh_staged_noise_workflow.py`

Functions/classes to alter:
- `OracleConfig`
- `ExpectationOracle`
- `RawMeasurementOracle` (new)
- `RawObservableEstimate` (new)
- `RawMeasurementRecord` (new)
- `RawExecutionBundle` (new)
- `build_parameterized_ansatz_circuit` (new)
- `bind_parameterized_ansatz_circuit` (new)
- `build_ansatz_circuit`
- `build_structural_ansatz_circuit`
- `_evaluate_locked_imported_circuit_raw_energy` (new)
- `_run_imported_fixed_scaffold_runtime_raw_baseline` (new)
- `_run_imported_fixed_scaffold_runtime_raw_baseline_mode_isolated` (new)
- `Phase3OracleGradientConfig`
- `_phase3_oracle_runtime_bindings`
- `_validate_phase3_oracle_gradient_config`
- `_phase3_oracle_gradient_config_payload`
- `_run_hardcoded_adapt_vqe`
- nested `_phase3_oracle_gradient_scout`
- `parse_args`
- `main`
- `add_staged_hh_noise_args`
- `FixedScaffoldRuntimeRawConfig` (new)
- `resolve_staged_hh_noise_config`
- `run_staged_hh_noise`
- `backend_supports_direct_run` (new)
- `snapshot_backend_target` (new)
- `build_compile_signature` (new)
- `extract_layout_maps` (new)
- `normalize_runtime_job_metadata` (new)

---

## 10. External references
- IBM Runtime `backend.run()` deprecation announcement (2024-12-17): https://quantum.cloud.ibm.com/announcements/en/product-updates/2024-12-17-backend-run
- IBMBackend API (`run()` no longer supported): https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/0.39/ibm-backend
- Qiskit Runtime migration guide: https://docs.quantum.ibm.com/api/migration-guides/qiskit-runtime
- V2 primitives / Sampler shot output: https://quantum.cloud.ibm.com/docs/en/guides/v2-primitives
- Sampler readout-mitigation tutorial: https://quantum.cloud.ibm.com/docs/en/tutorials/readout-error-mitigation-sampler
