# LLM Optimization Routine Spec (Core + Wrappers)

## 1) Purpose, Scope, Non-Goals

### Purpose
This document is an implementation-grade specification for an external LLM that advises Codex agents on how to improve convergence and runtime in this repository's optimization workflow. It is intended to be decision-complete: an implementing agent should not need to guess architecture or guardrails.

### Scope (in-scope modules)
- `pipelines/hardcoded/adapt_pipeline.py`
- `pipelines/hardcoded/hubbard_pipeline.py`
- `src/quantum/vqe_latex_python_pairs.py`
- `src/quantum/pauli_actions.py`
- `pipelines/exact_bench/cross_check_suite.py`
- `pipelines/qiskit_archive/compare_hc_vs_qk.py`
- `pipelines/run_guide.md`
- `test/test_adapt_vqe_integration.py`

### Non-goals
- Do not change Pauli ordering conventions.
- Do not modify operator-core base files (`pauli_letters_module.py`, `pauli_words.py`, `qubitization_module.py`).
- Do not introduce Qiskit into the hardcoded core optimization path.
- Do not change physics behavior by default when new optimizations are introduced.


## 2) Repo Invariants To Preserve

All optimization changes must preserve the following repository rules:

- ADAPT compiled-cache invariant:
  - Keep compiled Pauli-action caching enabled in production ADAPT gradient flow.
  - Do not replace cached production gradient flow with uncached `apply_pauli_string` loops.
  - Preserve cached-vs-uncached numerical parity tests.
- Pauli symbol convention:
  - Internal symbols are `e/x/y/z` (`e` identity).
- Pauli ordering convention:
  - Pauli string left-to-right is `q_(n-1) ... q_0`.
  - Qubit 0 is the rightmost character.
- Wrapper/shim policy:
  - Integration changes should be wrappers/shims around operator-core files; avoid redefining operator primitives.
- Core VQE path:
  - Hardcoded production VQE remains numpy statevector based and Qiskit-free.
- Drive safety invariant:
  - A=0 drive must remain numerically equivalent to no-drive within safe-test threshold (`<= 1e-10`) in compare flow.


## 2.5) Implementation Status Update (2026-03-04)

This spec has now been largely implemented in the hardcoded path. The sections below describing "as-is" behavior are historical context.

Implemented:
- Prompt 1: shared compiled polynomial utility landed in `src/quantum/compiled_polynomial.py` with:
  - `compile_polynomial_action`
  - `apply_compiled_polynomial`
  - `energy_via_one_apply`
  - `adapt_commutator_grad_from_hpsi`
- Prompt 2: compiled ansatz executor landed in `src/quantum/compiled_ansatz.py` (`CompiledAnsatzExecutor`).
- Prompt 3: compiled one-apply energy backend landed in `src/quantum/vqe_latex_python_pairs.py`:
  - `expval_pauli_polynomial_one_apply(...)`
  - `vqe_minimize(..., energy_backend=\"legacy\"|\"one_apply_compiled\")` (default legacy).
- Prompt 4: ADAPT gradient reuse landed in `pipelines/hardcoded/adapt_pipeline.py`:
  - one `Hpsi` per depth
  - pool gradients via `2*Im(vdot(Hpsi, Apsi))`
  - optional parity guard retained.
- Prompt 5: ADAPT inner optimizer now uses compiled ansatz execution and compiled one-apply energy objective, with shared Pauli-action cache reuse.
- Prompt 6: benchmark + timing instrumentation landed:
  - `pipelines/hardcoded/bench_compiled_energy_and_grad.py`
  - ADAPT `AI_LOG` timing events and `optimizer_elapsed_s` history field.

Validation coverage added:
- `test/test_compiled_polynomial.py`
- `test/test_compiled_ansatz.py`
- `test/test_vqe_energy_backend.py`
- `test/test_adapt_vqe_integration.py` parity/regression coverage remains active.


## 3) Current Optimization Routine (Historical As-Is Snapshot)

### 3.1 ADAPT flow map

| Topic | Current behavior | Location |
|---|---|---|
| ADAPT gradient backend | Uses compiled polynomial actions when provided. | `adapt_pipeline.py::_commutator_gradient` |
| Redundant Hamiltonian action | `H|psi>` is recomputed inside `_commutator_gradient` for each pool operator during one depth sweep. | `adapt_pipeline.py::_commutator_gradient` and caller loop in `_run_hardcoded_adapt_vqe` |
| ADAPT ansatz state prep | Uses `apply_exp_pauli_polynomial(...)` per selected op; this path applies Pauli strings through slower per-index/per-qubit logic. | `adapt_pipeline.py::_prepare_adapt_state`, `vqe_latex_python_pairs.py::apply_exp_pauli_polynomial` |
| ADAPT energy eval | Uses `expval_pauli_polynomial(...)`, which computes termwise expectations repeatedly. | `adapt_pipeline.py::_adapt_energy_fn`, `vqe_latex_python_pairs.py::expval_pauli_polynomial` |
| ADAPT inner reopt method | Fixed SciPy COBYLA in production ADAPT pipeline (`method="COBYLA"`). | `adapt_pipeline.py::_run_hardcoded_adapt_vqe` |
| Conventional VQE optimizer | Method is configurable (e.g., COBYLA, SLSQP, L-BFGS-B, Powell, Nelder-Mead) in hardcoded pipeline. | `hubbard_pipeline.py` (`--vqe-method`) and `vqe_latex_python_pairs.py::vqe_minimize` |

### 3.2 Existing fast primitives already present

- Compiled Pauli action via permutation + phase exists in:
  - `src/quantum/pauli_actions.py`
  - local compiled helpers inside `adapt_pipeline.py`
- Compiled polynomial apply exists in `adapt_pipeline.py::_apply_compiled_polynomial`.
- ADAPT telemetry already reports compiled-cache metadata in payload (`compiled_pauli_cache` block).

### 3.3 Wrapper dependency notes

- `hubbard_pipeline.py` calls ADAPT internals via `_run_internal_adapt_paop(...)` delegating to `adapt_pipeline._run_hardcoded_adapt_vqe(...)`.
- `cross_check_suite.py` includes a simplified ADAPT mini-loop with its own slower polynomial apply and COBYLA reopt.
- `compare_hc_vs_qk.py` orchestrates runs and enforces A=0 safe-test behavior for drive paths.


## 4) Bottlenecks And Why (Engineering View)

Kernel-level costs dominate wall-clock before optimizer sophistication does.

### B1) Redundant `H|psi>` in ADAPT gradient sweep
- Current ADAPT depth loop evaluates gradients for many pool operators.
- `H|psi_current>` is identical for all operators at a fixed depth.
- Recomputing it per operator multiplies Hamiltonian-apply cost by `|pool|`.

### B2) Slow ansatz state prep path
- ADAPT state prep repeatedly calls `apply_exp_pauli_polynomial`.
- That path delegates to string-based Pauli application (`apply_pauli_string`) with per-index/per-qubit logic.
- For repeated optimizer calls, this dominates objective evaluation time.

### B3) Energy evaluation overhead
- Termwise expectation accumulation (`expval_pauli_polynomial`) performs repeated Pauli applications and dot products.
- For objective-heavy loops, a single compiled Hamiltonian apply followed by one `vdot` reduces overhead and can reuse compiled caches.

### Expected impact ordering
1. Gradient reuse (`H|psi>` once per depth).
2. Compiled-action ansatz execution.
3. One-apply energy evaluation.

These changes typically unlock deeper/longer optimization within the same runtime budget.


## 5) Implementation Plan For LLM To Instruct Codex

## Phase A: Kernel speedups first

### A1) ADAPT gradient reuse (`H_psi` once per depth)

#### Change intent
Compute `H_psi_current` once in each ADAPT depth iteration and reuse it for all pool operators.

#### Proposed API delta
- Extend gradient function to accept optional precomputed Hamiltonian action:
  - `_commutator_gradient(..., h_psi: np.ndarray | None = None, ...)`

#### Implementation notes
- In `_run_hardcoded_adapt_vqe`, after `psi_current` is built, compute:
  - `H_psi_current = _apply_pauli_polynomial(psi_current, h_poly, compiled=h_compiled)`
- Pass `h_psi=H_psi_current` into every per-operator `_commutator_gradient` call.
- Keep backward-compatible behavior:
  - If `h_psi is None`, `_commutator_gradient` computes `H_psi` internally (parity fallback).

#### Parity requirement
- Exact numerical parity with old path (within floating tolerance used by existing tests).

### A2) Compiled-action ansatz execution

#### Change intent
Replace repeated slow ansatz application in ADAPT with compiled per-term actions while preserving existing operator/term order semantics.

#### Proposed new helper types
- `CompiledAnsatzRotationTerm`:
  - `coeff: complex`
  - `action: CompiledPauliAction | None` (`None` for identity)
- `CompiledAnsatzOperator`:
  - ordered tuple of `CompiledAnsatzRotationTerm`
  - metadata needed to preserve current order policy

#### Proposed helper functions
- `_compile_ansatz_operator(polynomial, *, sort_terms=True, tol=...) -> CompiledAnsatzOperator`
- `_apply_compiled_exp_pauli_polynomial(psi, compiled_op, theta, *, ignore_identity=True, tol=...) -> np.ndarray`
- `_prepare_adapt_state_compiled(psi_ref, compiled_selected_ops, theta) -> np.ndarray`

#### Ordering semantics
- Preserve current behavior of `apply_exp_pauli_polynomial(..., sort_terms=True)` in ADAPT path.
- Preserve identity handling and coefficient tolerance behavior.

#### Caching model
- Compile each selected operator once when appended to ADAPT ansatz.
- Reuse compiled selected-operator list across all objective calls in re-optimization.

### A3) Energy via one Hamiltonian apply

#### Change intent
Use one compiled Hamiltonian apply and one `vdot` for energy:
- `H_psi = _apply_pauli_polynomial(psi, h_poly, compiled=h_compiled)`
- `E = Re(vdot(psi, H_psi))`

#### Proposed helper
- `_adapt_energy_from_h_apply(psi, h_poly, *, h_compiled=None) -> float`

#### Integration points
- Update `_adapt_energy_fn` to use compiled state prep (A2) and one-apply energy (A3).
- Keep fallback path for parity/debug safety.


## Phase B: Inner optimizer policy (after kernel wins)

### B1) Deterministic/ideal regime policy
- Recommend `L-BFGS-B` or `BFGS` for heavier parameter regimes.
- Preserve backward-compatible defaults unless user opts in.
- Keep run-guide escalation logic coherent (`fallback_A` currently escalates optimizer effort and may switch method for heavier L).

### B2) Noisy regime policy
- Recommend SPSA-style optimization for noisy/stochastic objectives.
- Treat as a policy/staging recommendation unless and until a hardcoded implementation path is added.
- Keep existing objective-aggregation/repeat knobs as primary variance control.

### B3) Optional plumbing proposal
- Add explicit ADAPT inner optimizer options only if adopted:
  - `--adapt-optimizer-method` (default legacy behavior).
  - Method-specific options guarded by compatibility defaults.
- If not adopted now, document as planned follow-up and keep ADAPT COBYLA-fixed.


## Phase C: Gradient step-change roadmap (staged follow-up)

### C1) Objective
Implement adjoint-style gradients for Pauli-rotation sequence ansatz execution so gradient cost scales as a small multiple of one full state-prep/energy pass, instead of finite-difference-like `O(P)` objective calls.

### C2) Prerequisite
A2 compiled ansatz representation must exist and expose ordered elementary rotations.

### C3) Integration target
- Extend `vqe_minimize(...)` flows to optionally pass `jac=` into SciPy `minimize`.
- Use gradient-aware methods (`L-BFGS-B`, `BFGS`) where stable.

### C4) Staging rule
Do not begin C-phase until A/B parity and profiling gates pass.


## 6) Public Interfaces / Type Changes To Propose

These are proposed deltas for implementation planning:

- `adapt_pipeline.py`
  - `_commutator_gradient(..., h_psi: np.ndarray | None = None, ...)`
  - compiled ansatz operator/term dataclasses for reusable exp application.
  - energy helper that computes from one Hamiltonian apply.
- Optional CLI/interface (only if optimizer policy is implemented now):
  - ADAPT inner optimizer method plumbing with legacy default preserved.

No changes to operator-core base files are required.


## 7) Validation And Acceptance Criteria

### 7.1 Numerical parity tests (required)
- Cached vs uncached polynomial apply parity.
- Gradient parity:
  - with precomputed `H_psi`
  - vs legacy no-precompute path.
- Compiled ansatz prep parity:
  - vs existing `_prepare_adapt_state` slow path.
- Energy parity:
  - one-apply energy vs `expval_pauli_polynomial`.

### 7.2 Behavioral parity (required)
- ADAPT default stopping behavior remains consistent (`eps_grad`, `eps_energy`, `pool_exhausted`, `max_depth` semantics).
- Default ADAPT run contract remains backward compatible for `hubbard_pipeline` wrapper path.
- Cross-check wrapper remains operational if ADAPT shared helpers are reused or mirrored.

### 7.3 Performance checks (required)
- Microbench targets:
  - Gradient sweep throughput improves materially after A1.
  - Objective evaluation throughput improves after A2+A3.
- Keep acceptance thresholds explicit in tests/bench harness comments.

### 7.4 Regression guardrails
- Existing expectations in `test/test_adapt_vqe_integration.py` must remain valid.
- Preserve ADAPT compiled-cache telemetry fields.
- Preserve drive A=0 safe-test behavior in compare flow.


## 8) Rollout Strategy

1. Land A1 with focused parity tests.
2. Land A2 with compiled-ansatz parity tests.
3. Land A3 with energy parity tests and microbench updates.
4. Land optimizer policy changes (B-phase) separately from kernel changes.
5. Keep defaults stable; expose new behavior behind explicit flags when needed.

This sequencing isolates risk and simplifies root-cause analysis for numerical drifts.


## 9) Codex Execution Guidance Appendix

Use this appendix to prompt Codex agents with minimal ambiguity.

### 9.1 Required files touched by phase

- A1:
  - `pipelines/hardcoded/adapt_pipeline.py`
  - `test/test_adapt_vqe_integration.py`
- A2:
  - `pipelines/hardcoded/adapt_pipeline.py`
  - `test/test_adapt_vqe_integration.py`
  - optional: `src/quantum/pauli_actions.py` (only if shared helper extraction is justified)
- A3:
  - `pipelines/hardcoded/adapt_pipeline.py`
  - `test/test_adapt_vqe_integration.py`
- Optional wrapper alignment:
  - `pipelines/exact_bench/cross_check_suite.py`
  - `pipelines/run_guide.md` (if user-facing behavior/flags are changed)

### 9.2 Do-not-touch list

- `src/quantum/pauli_letters_module.py`
- `src/quantum/pauli_words.py`
- `src/quantum/qubitization_module.py`

### 9.3 Must-pass checks

- ADAPT integration parity tests in `test/test_adapt_vqe_integration.py`.
- Any newly added parity tests for precomputed `H_psi`, compiled ansatz state prep, and one-apply energy.

### 9.4 Prompt templates for Codex

#### Template A1 (gradient reuse)
Implement ADAPT gradient reuse in `pipelines/hardcoded/adapt_pipeline.py` by computing `H_psi` once per ADAPT depth and passing it into `_commutator_gradient`. Add backward-compatible `_commutator_gradient(..., h_psi=None, ...)` behavior and parity tests in `test/test_adapt_vqe_integration.py`.

#### Template A2 (compiled ansatz path)
Add compiled ansatz execution in `pipelines/hardcoded/adapt_pipeline.py`: compile selected operators once, then apply exp rotations with compiled perm+phase actions while preserving current term ordering and identity/tolerance semantics. Add strict parity tests against the existing `_prepare_adapt_state` path.

#### Template A3 (one-apply energy)
Replace ADAPT energy evaluation internals with one Hamiltonian apply plus one dot product (`Re(vdot(psi, H_psi))`) using compiled Hamiltonian actions where available. Keep parity with `expval_pauli_polynomial` and add tests.

#### Template B (optimizer policy, optional)
Propose a backward-compatible ADAPT inner-optimizer policy that keeps default behavior stable and allows deterministic heavy runs to use `L-BFGS-B`/`BFGS` under explicit configuration. Do not break existing run defaults or run-guide contracts.


## 10) Assumptions Locked For This Spec

- This document is design-spec style, not prompt-pack only.
- Scope is core + wrappers.
- This spec itself is doc-only and does not change runtime code.
