# Analytic Noise Injection Implementation Spec
Do not assume the instructions are 100% correct; They are meant as strong guidance, but it is up to your discretion what spec to implement.
## 1. Executive Summary & Objectives
This document provides the theoretical foundation and implementation guidelines for adding an **Analytic Noise Injection** layer to the Hubbard-Holstein workflows in this repository. 

Our goal is to accurately predict how our active decision loops (both VQE state preparation and the time-dynamics secant controller) will behave on real superconducting QPU hardware. Evaluating the metric landscapes (gradients, energies, geometric tensors) step-by-step using thousands of noisy `AerSimulator` circuits is computationally unfeasible. 

To bypass this bottleneck, the implementation agent will **inject calibrated Gaussian white noise directly into the exact statevector evaluations**. Because the statevector algebra completes instantly, this approach allows us to rapidly simulate the *statistical noise, navigational failures, and erroneous tree pruning* the controllers will experience on a real QPU, without paying the cost of full circuit simulation inside the optimization loops.

---

## 2. Theory: Replicating Measurement Shot Noise
When measuring an observable $\hat{O}$ exactly, we compute $E = \langle \psi | \hat{O} | \psi \rangle$.
On a QPU with $N$ shots, the result is drawn from a distribution with mean $E$ and variance $\sigma^2 = \frac{\langle \hat{O}^2 \rangle - \langle \hat{O} \rangle^2}{N}$.

To mock this using an exact statevector backend, the agent must intercept the exact calculation of gradients and energies and alter the outcome before returning it to the optimizer/controller.

The injected noise model is:
$$ E_{noisy} = E_{exact} + \mathcal{N}(0, \sigma^2_{eff}) $$

Where $\sigma^2_{eff}$ is a tunable variance parameter that acts as a proxy for $1/N$ scaling and backend depolarizing noise.

---

## 3. Pipeline 1: ADAPT L=2 State Preparation

**Primary script:** `pipelines/hardcoded/adapt_pipeline.py`
**SPSA optimizer:** `src/quantum/spsa_optimizer.py`
**Gradient helper:** `src/quantum/compiled_polynomial.py`

### Objective
We must answer: *Will the Phase 3 tree search and pruning mechanics clip the correct entanglement branches if the energy scoring and SPSA gradients are blurred by noise?*

### Implementation Reasoning
In ADAPT-VQE, the operator pool is scored based on energy gradients, and terms are pruned or appended based on these gradients surpassing certain thresholds (`--adapt-eps-grad`, etc.). 
- If the agent drops a term because a noisy gradient calculation randomly falls below `eps-grad`, the trajectory fails.
- If the SPSA optimizer gets caught in false local minima due to noisy energy evaluations, it aborts prematurely.

### 3.1 — Add CLI argument `--adapt-analytic-noise-std`

**File:** `pipelines/hardcoded/adapt_pipeline.py`

**Insert at line 16106** (immediately after `--adapt-spsa-progress-every-s` on line 16105):

```python
p.add_argument("--adapt-analytic-noise-std", type=float, default=0.0,
               help="Std-dev of Gaussian noise injected into exact energy and gradient evaluations (0 = disabled).")
```

### 3.2 — Inject noise into the energy objective function

**Target function:** `_evaluate_selected_energy_objective` — defined at `adapt_pipeline.py:6987`.

This inner function is the single chokepoint for **all** energy evaluations (SPSA objective, final refit, prune refit). It has two return paths for exact evaluation:

1. **Compiled path** — line 7152: `energy_obj, _ = energy_via_one_apply(psi_obj, h_compiled)` then `return float(energy_obj)` on line 7153.
2. **Fallback path** — lines 7154-7163: `return float(_adapt_energy_fn(...))`.

**What to do:** After both exact return paths, add Gaussian noise before returning. Concretely, replace the return statements at lines 7152–7163 with logic that computes the exact energy and then returns `float(energy_exact) + np.random.normal(0.0, adapt_analytic_noise_std)` when `adapt_analytic_noise_std > 0.0`. The `adapt_analytic_noise_std` value (from `args.adapt_analytic_noise_std`) must be captured in the enclosing scope of `_evaluate_selected_energy_objective` (it is defined inside `run_adapt_vqe`). The noisy oracle path (lines 7005–7144) should NOT be modified — it already has its own noise model.

**Specifically, replace lines 7145–7163:**

```python
# BEFORE (exact paths):
            if executor_now is not None:
                psi_obj = _prepare_selected_state(...)
                energy_obj, _ = energy_via_one_apply(psi_obj, h_compiled)
                return float(energy_obj)
            return float(_adapt_energy_fn(...))

# AFTER:
            if executor_now is not None:
                psi_obj = _prepare_selected_state(
                    ops_now=list(ops_now),
                    theta_now=theta_eval,
                    executor_now=executor_now,
                    parameter_layout_now=layout_eval,
                )
                energy_obj, _ = energy_via_one_apply(psi_obj, h_compiled)
                energy_exact = float(energy_obj)
            else:
                energy_exact = float(
                    _adapt_energy_fn(
                        h_poly,
                        psi_ref,
                        list(ops_now),
                        theta_eval,
                        h_compiled=h_compiled,
                        parameter_layout=parameter_layout_now,
                    )
                )
            if adapt_analytic_noise_std > 0.0:
                energy_exact += float(np.random.normal(0.0, adapt_analytic_noise_std))
            return energy_exact
```

**All six `spsa_minimize(...)` call sites** (lines 7292, 5823, 9849, 12451, 13321, 13680) pass objective wrappers (`_seed_obj`, `_obj_reduced`, `_obj_opt_local`, `_obj_opt`, `_obj_final`, `_obj_prune_reduced`) that all funnel through `_evaluate_selected_energy_objective`. Therefore, modifying `_evaluate_selected_energy_objective` **once** is sufficient — no changes are needed at any of the six `spsa_minimize` call sites.

### 3.3 — Inject noise into the pool gradient scorer

There are **two** gradient paths (selected at `adapt_pipeline.py:10758` by the `phase3_oracle_gradient_enabled` flag):

#### Path A: Exact commutator gradients (non-oracle, line 10776–10779)

```python
for i in available_indices:
    apsi = _apply_compiled_polynomial(psi_current, pool_compiled[i])
    gradients[i] = adapt_commutator_grad_from_hpsi(hpsi_current, apsi)
    grad_magnitudes[i] = abs(float(gradients[i]))
```

**What to do:** After line 10778, add noise to `gradients[i]` before computing the magnitude on line 10779:

```python
for i in available_indices:
    apsi = _apply_compiled_polynomial(psi_current, pool_compiled[i])
    gradients[i] = adapt_commutator_grad_from_hpsi(hpsi_current, apsi)
    if adapt_analytic_noise_std > 0.0:
        gradients[i] += float(np.random.normal(0.0, adapt_analytic_noise_std))
    grad_magnitudes[i] = abs(float(gradients[i]))
```

#### Path B: Phase 3 oracle gradient scout (line 10758–10767)

This path calls `_phase3_oracle_gradient_scout` (defined at line 6503) which already uses a noisy measurement surface. **Do NOT modify this path** — it has its own noise model.

### 3.4 — Log noise std in JSON output

**File:** `pipelines/hardcoded/adapt_pipeline.py`

**Location:** The settings payload block at lines 17089–17101 where `adapt_spsa` settings are serialized.

**What to do:** After line 17101 (after the closing `}` of the `adapt_spsa` dict), add:

```python
    payload["settings"]["adapt_analytic_noise_std"] = float(args.adapt_analytic_noise_std)
```

Also emit it in the `_ai_log` calls for SPSA heartbeats if desired (e.g., at lines 7280–7290).

### 3.5 — Pruning robustness (no extra code needed)

The `eps_grad` termination check at line 9413 (`if max_grad_local < float(eps_grad)`) already reads from `grad_magnitudes`, which will now contain noisy values from step 3.3. No separate modification is needed — the pruning mechanics will automatically consume the noisy gradients.

---

## 4. Pipeline 2: Time-Dynamics Secant Controller

**Controller:** `pipelines/hardcoded/hh_realtime_checkpoint_controller.py`
**Config dataclass:** `pipelines/hardcoded/hh_realtime_checkpoint_types.py` — `RealtimeCheckpointConfig` (line 24)
**Ledger dataclass:** `pipelines/hardcoded/hh_realtime_checkpoint_types.py` — `CheckpointLedgerEntry` (line 230)
**CLI args:** `pipelines/hardcoded/hh_staged_cli_args.py` — `add_staged_hh_base_args` (line 10)
**Config construction:** `pipelines/hardcoded/hh_staged_workflow.py` — `RealtimeCheckpointConfig(...)` at line 1033

### Objective
We must answer: *Does the Secant lead-tapering controller over-correct or prune critical terms under noise when extrapolating the time-evolution trajectory?*

### Implementation Reasoning
The dynamic controller builds a moving trajectory using exact metrics:
- The projection metric tensor (geometric tensor $K$).
- The energy gradient error (vector $V$).
- Matrix pseudo-inverses ($\text{pinv}(K)$).

Matrix inversion is notoriously unstable under noise. If $K$ and $V$ are perfectly exact, `pinv_reg` securely finds the best differential step $\dot{\theta}$. If $K$ is noisy, the pseudo-inverse might explode or select completely incorrect directions for the Secant baseline.

### 4.1 — Add config field

**File:** `pipelines/hardcoded/hh_realtime_checkpoint_types.py`

**Insert at line 55** (after `pinv_rcond: float = 1e-10` on line 54):

```python
    analytic_noise_std: float = 0.0
```

### 4.2 — Add CLI argument

**File:** `pipelines/hardcoded/hh_staged_cli_args.py`

**Insert at line 514** (after `--checkpoint-controller-commit-repeats` on line 513, before the `# Dynamics` comment on line 515):

```python
    p.add_argument("--checkpoint-controller-analytic-noise-std", type=float, default=0.0,
                   help="Std-dev of Gaussian noise injected into geometric tensor G, gradient vector f, and rho_miss (0 = disabled).")
```

### 4.3 — Wire CLI arg into config construction

**File:** `pipelines/hardcoded/hh_staged_workflow.py`

**Insert inside the `RealtimeCheckpointConfig(...)` constructor call.** Add after the `pinv_rcond=...` line (line 1173):

```python
        analytic_noise_std=float(getattr(args, "checkpoint_controller_analytic_noise_std", 0.0)),
```

### 4.4 — Inject noise into geometric tensor G and gradient vector f

**File:** `pipelines/hardcoded/hh_realtime_checkpoint_controller.py`

**Target function:** `_compute_baseline_geometry_for_runtime_state` (line 3923).

The critical computation block is lines 3981–3987:

```python
G = np.asarray(np.real(tangents_matrix.conj().T @ tangents_matrix), dtype=float)            # line 3981
f = np.asarray(np.real(tangents_matrix.conj().T @ b_bar), dtype=float).reshape(-1)           # line 3982
G_pinv = np.linalg.pinv(G, rcond=float(self.cfg.pinv_rcond)) if G.size else ...             # line 3983
K = np.asarray(G + float(self.cfg.regularization_lambda) * np.eye(...), dtype=float)         # line 3984
K_pinv = np.linalg.pinv(K, rcond=float(self.cfg.pinv_rcond)) if K.size else ...             # line 3985
theta_dot_proj = np.asarray(G_pinv @ f, ...).reshape(-1) if G.size else ...                  # line 3986
theta_dot_step = np.asarray(K_pinv @ f, ...).reshape(-1) if K.size else ...                  # line 3987
```

**What to do:** Insert noise injection between lines 3982 and 3983, after `G` and `f` are computed but before `pinv` is called:

```python
        # --- analytic noise injection ---
        if float(self.cfg.analytic_noise_std) > 0.0 and G.size:
            _noise_std = float(self.cfg.analytic_noise_std)
            G = G + np.random.normal(0.0, _noise_std, size=G.shape).astype(float)
            # Re-symmetrize G after noise (it must remain symmetric for pinv stability)
            G = 0.5 * (G + G.T)
            f = f + np.random.normal(0.0, _noise_std, size=f.shape).astype(float)
        # --- end analytic noise injection ---
```

Lines 3983–3987 then compute `G_pinv`, `K`, `K_pinv`, `theta_dot_proj`, `theta_dot_step` from the now-noisy `G` and `f`. The downstream `rho_miss` (line 3991) will automatically reflect the noisy geometry.

### 4.5 — Inject noise into `rho_miss` at the decision boundary

**File:** `pipelines/hardcoded/hh_realtime_checkpoint_controller.py`

**Target function:** `_controller_lane` (line 1990).

The critical check is at line 2002:

```python
if float(rho_miss) > float(self.cfg.miss_threshold):
    return "append", "exact_rho_miss_above_threshold"
```

**What to do:** Add noise to `rho_miss` before the threshold comparison. Insert before line 2002:

```python
        if float(self.cfg.analytic_noise_std) > 0.0:
            rho_miss = float(rho_miss) + float(np.random.normal(0.0, float(self.cfg.analytic_noise_std)))
            rho_miss = max(0.0, rho_miss)  # rho_miss cannot be negative
```

Similarly, add noise before the prune threshold check in `_prune_permitted` (line 1613):

```python
if float(rho_miss) > float(getattr(self.cfg, "prune_miss_threshold", 0.0)):
    return False, "rho_miss_above_prune_threshold"
```

Insert before line 1613:

```python
        if float(self.cfg.analytic_noise_std) > 0.0:
            rho_miss = float(rho_miss) + float(np.random.normal(0.0, float(self.cfg.analytic_noise_std)))
            rho_miss = max(0.0, rho_miss)
```

**Note:** `numpy` is already imported in this file. Add `import numpy as np` only if it is not present (it is — verify at the top of the file).

### 4.6 — Log noise std in the ledger

**File:** `pipelines/hardcoded/hh_realtime_checkpoint_types.py`

**Add to `CheckpointLedgerEntry`** (after `degraded_reason` on line 311):

```python
    analytic_noise_std: float = 0.0
```

**File:** `pipelines/hardcoded/hh_realtime_checkpoint_controller.py`

**At the ledger entry construction** (line 5745, `CheckpointLedgerEntry(...)`), add:

```python
                    analytic_noise_std=float(self.cfg.analytic_noise_std),
```

This value propagates automatically to JSON via the existing `dataclass_to_payload(ledger_entry)` call on line 5869 and into the progress/partial payloads written at lines 872 and 911.

---

## 5. Implementation Guardrails for the Agent
- **No Heavy Simulations:** The agent must **NOT** import `qiskit_aer` or build quantum circuits to do this. All injections must happen via fast `numpy.random.normal()` additions to the existing dense matrix math (statevectors).
- **Control Flags:** By default, all analytic noise should be disabled (`std = 0.0`) so that existing exact baseline tests still pass effortlessly.
- **Log the Noise:** The JSON output ledgers must log `analytic_noise_std` so we can track which trajectories used which noise thresholds in later analysis.
- **Do NOT modify** any `spsa_minimize` call sites — the noise is injected upstream in `_evaluate_selected_energy_objective`.
- **Do NOT modify** the Phase 3 oracle gradient scout path (`_phase3_oracle_gradient_scout`) — it has its own noise model.
- **Preserve symmetry** of G after noise injection by re-symmetrizing: `G = 0.5 * (G + G.T)`.

---

## 6. File Change Summary

| # | File | Lines | Change |
|---|------|-------|--------|
| 1 | `pipelines/hardcoded/adapt_pipeline.py` | after 16105 | Add `--adapt-analytic-noise-std` CLI arg |
| 2 | `pipelines/hardcoded/adapt_pipeline.py` | 7145–7163 | Noise injection in `_evaluate_selected_energy_objective` exact return paths |
| 3 | `pipelines/hardcoded/adapt_pipeline.py` | 10776–10779 | Noise injection in exact commutator gradient loop |
| 4 | `pipelines/hardcoded/adapt_pipeline.py` | after 17101 | Log `adapt_analytic_noise_std` in JSON payload |
| 5 | `pipelines/hardcoded/hh_realtime_checkpoint_types.py` | after line 54 | Add `analytic_noise_std: float = 0.0` to `RealtimeCheckpointConfig` |
| 6 | `pipelines/hardcoded/hh_realtime_checkpoint_types.py` | after line 311 | Add `analytic_noise_std: float = 0.0` to `CheckpointLedgerEntry` |
| 7 | `pipelines/hardcoded/hh_staged_cli_args.py` | after line 513 | Add `--checkpoint-controller-analytic-noise-std` CLI arg |
| 8 | `pipelines/hardcoded/hh_staged_workflow.py` | after line 1173 | Wire `analytic_noise_std` into `RealtimeCheckpointConfig(...)` |
| 9 | `pipelines/hardcoded/hh_realtime_checkpoint_controller.py` | between 3982–3983 | Noise injection into G and f before pinv |
| 10 | `pipelines/hardcoded/hh_realtime_checkpoint_controller.py` | before line 2002 | Noise injection into rho_miss in `_controller_lane` |
| 11 | `pipelines/hardcoded/hh_realtime_checkpoint_controller.py` | before line 1613 | Noise injection into rho_miss in `_prune_permitted` |
| 12 | `pipelines/hardcoded/hh_realtime_checkpoint_controller.py` | at line 5745 | Pass `analytic_noise_std` to `CheckpointLedgerEntry(...)` |
