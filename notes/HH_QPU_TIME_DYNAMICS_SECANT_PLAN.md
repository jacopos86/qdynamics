# Time Dynamics QPU Execution Plan: Secant Controller Path (Heron)

## Objective
Implement compiled circuit execution on IBM backends (fake and real QPU) for the driven time dynamics of our best controller configurations: **`secant_lead100`** and **`secant_lead200`**. The goal is to evaluate observables over discrete time steps on actual quantum hardware using `EstimatorV2`/`SamplerV2` and a full error mitigation stack identical to the ADAPT pipeline.

**Crucial Context:** This task is explicitly focused on **time dynamics**, *not* VQE. We are tracking operators / observables across the trajectory dictated by the `secant_lead` controllers. 

## Architectural Context / Prior Art
To minimize planning overhead:
- **Configurations to target:** 
  1. `secant_lead100` (Defined in `MATH/Math.md`, page 1-4).
  2. `secant_lead200` (CLI arguments inside `/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/artifacts/agent_runs/high_amp_spectra_run/pdf/high_amp_three_run_summary.pdf`).
- **Relevant Core Modules:** `hh_staged_workflow.py`, `hh_realtime_checkpoint_controller.py`, `hh_realtime_measurement.py`.
- **Backend Infrastructure:** Leverage and expand `pipelines/qiskit_backend_tools.py` just like we did for the ADAPT path. 
- **Target Hardware Architecture:** IBM Heron (must match the exact QPU model used during the ADAPT route compilation).

## Implementation Phases
Implement this in safe, verifiable stages. Ensure each phase produces actionable metrics/reports before moving to the next.

### Phase 1: Assess Existing Fake Backend Paths
- The `secant` time dynamics loops (checkpoint controller `oracle_v1`) *already implement Phase 1* via the `noise_mode="backend_scheduled"` mode.
- In `hh_realtime_measurement.py` and `hh_realtime_checkpoint_controller.py`, there is a hard guard currently checking `not bool(base_config.use_fake_backend)`.
- Your primary task here is merely ensuring this existing fake backend pipeline is stable and acts as your testing ground before taking the guardrails off.

### Phase 2: Real QPU Integration & Raw Execution
- Remove the `use_fake_backend=True` limitation inside `validate_controller_oracle_base_config` within `pipelines/hardcoded/hh_realtime_measurement.py` (and any related validations) so it accepts real `QiskitRuntimeService` backends.
- Route the existing Qiskit observables structure directly to the physical IBM Heron `EstimatorV2`.
- Ensure the QPU integration script handles job batching/session management efficiently to avoid queue timeouts over many timesteps.

### Phase 3: The Full ADAPT-Parity Noise Stack
Layer and validate the exact same mitigation pipeline implemented for the ADAPT path:
1. **Readout Error Mitigation (TREX)**
2. **Pauli Twirling**
3. **Dynamical Decoupling (DD)**
4. **Zero Noise Extrapolation (ZNE)** (crucial for time dynamics depth scaling).
5. **Symmetry Projection** (if required for observable accuracy).

## Instructions for Repo Agent
1. Determine the minimal-diff integration point. (e.g. extending `hh_staged_noise_workflow.py` or creating `hh_staged_qpu_workflow.py`).
2. Provide a short implementation strategy acknowledging the fake backend compilation plan.
3. Check the `secant_lead` JSONs / PDF CLI for the exact benchmark targets.
4. Begin executing **Phase 1**.

## Concrete Code Hooks for the Downstream Agent
To save your context window, here are the exact files and integration points discovered regarding how noise and QPU execution is currently hooked up for ADAPT and how they should be hooked up for time dynamics:

- **`pipelines/qiskit_backend_tools.py`**: 
  - Use `resolve_backend_targets` and `compile_circuit_for_backend`. This file already handles caching and FakeBackend/Runtime integrations (e.g. `snapshot_backend_target`).
- **`pipelines/hardcoded/adapt_pipeline.py`**:
  - *Do not run this file directly*, but use it as the template for the **noise CLI arguments**. It establishes the contract for:
    - `--final-noise-audit-local-readout-strategy`
    - `--final-noise-audit-local-gate-twirling`
    - `--final-noise-audit-dd-sequence`
    - `--final-noise-audit-zne-extrapolator`
  - You must replicate (or directly reuse) these exact CLI arguments to trigger the Phase 3 features for the time dynamics script.
- **`pipelines/hardcoded/hh_realtime_measurement.py`**:
  - Look at `estimate_grouped_raw_mclachlan_geometry` and `estimate_grouped_raw_mclachlan_incremental_block`. 
  - This file currently handles `raw_group_pool` and measurement loops. Notice how it passes an `oracle` object containing `noise_mode`, `estimator_kind`, and configures `min_total_shots`.
  - Your time dynamics QPU evaluations should route through or mimic this estimation architecture rather than building a new Qiskit `EstimatorV2` loop from scratch, preserving the `RawGroupKey` caching if applicable.

**Unresolved questions to answer/problems:**
- Do we need to batch all timestep circuits into a single Qiskit Runtime primitive job, or evaluate them step-by-step incrementally? (Hint: Batching is significantly more queue-efficient but limits adaptive real-time feedback).
- Will the circuit depth at late time dynamics (`t_final=8.0`) exceed the ISA capabilities or coherence limit without aggressive ZNE?

**Files to edit:** 
- `pipelines/qiskit_backend_tools.py` (Likely extension)
- `pipelines/hardcoded/hh_staged_workflow.py` OR a new execution script.
- Optionally `pipelines/hardcoded/hh_realtime_measurement.py`.

## Mandatory Repository Rules (AGENTS.md Compliance)
To ensure the implementation stays perfectly aligned with the repository's strict operational invariants, the downstream agent must adhere to the following when building this time dynamics execution plan:
1. **PDF Parameter Manifests**: Any PDF generated by this new pipeline feature **must** include a clear, list-style parameter manifest at the start.
2. **Qiskit Scoping**: Qiskit is explicitly allowed here because this is QPU hardware execution/validation, but you must ensure it remains isolated to these executor/backend integration paths and does not leak back into the core non-QPU numpy math paths.
3. **Docs Hierarchy**: For execution contracts, refer back to `run_guide.md` if any conflicts arise concerning run scaling, but `MATH/Math.md` holds the specific `secant` equations.
