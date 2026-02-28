# Design Note: Qiskit Baseline and Time-Dependent Hamiltonians

**Date:** 2026-02-20  
**Status:** Accepted — Option A (keep bypass), with Option B sketched as an optional future integration  
**Scope:** `qiskit_hubbard_baseline_pipeline.py`, `hardcoded_hubbard_pipeline.py`

---

## 1. The Question

When the drive is enabled (`--enable-drive`), `qiskit_hubbard_baseline_pipeline.py` already
bypasses `PauliEvolutionGate` for the Trotter propagation path.  Instead, it calls the same
`_evolve_trotter_suzuki2_absolute` kernel used by the hardcoded pipeline — a pure numpy/scipy
loop that evaluates `H(t_k)` explicitly at each slice.

The same bypass applies to the reference propagator: `_evolve_piecewise_exact` runs our own
piecewise-constant matrix-exponential loop (now using `scipy.sparse.linalg.expm_multiply`).

**Should we:**  
**(A)** Keep the current bypass permanently — both pipelines share identical drive physics, no
Qiskit-specific path for time-dependent evolution.  
**(B)** Introduce an *optional* `qiskit-dynamics` integration that uses `HamiltonianModel` /
`Solver` for time-dependent simulation, without touching the production core.

---

## 2. Current Architecture (as of this note)

### 2.1 Time-independent path (no drive)

```
PauliEvolutionGate(H_static, t, synthesis=SuzukiTrotter)
  └── Qiskit circuit ─► Statevector.evolve()   ← Qiskit-native, O(d²) per step
```

The Qiskit `PauliEvolutionGate` + `SuzukiTrotter` synthesis is the production path for the
no-drive case.  It benefits from Qiskit's transpiler/optimiser and serves as the *validation
target* against the hardcoded Suzuki-2 loop.

### 2.2 Time-dependent path (drive enabled)

```
_evolve_trotter_suzuki2_absolute(psi, H_static_terms, H_drive(t_k), ...)
  ├── H(t_k) = H_static + H_drive(t_k)
  └── compiled CompiledPauliAction — O(d) per Pauli per step   ← numpy, no Qiskit

_evolve_piecewise_exact(psi, H_static, drive_provider, ...)
  ├── dim ≥ 64: expm_multiply( (-iΔt)*(H_static_csc + diags(d_drive)), psi )
  └── dim < 64: dense expm(-iΔt*H_total) @ psi                ← scipy, no Qiskit
```

`PauliEvolutionGate` is *not used* when the drive is active.  The Qiskit pipeline therefore
acts as a pure numpy/scipy code path that happens to obtain its Hamiltonian coefficients from
Qiskit Nature / JW mapper — the circuit layer is silent.

### 2.3 Why the bypass was introduced

`PauliEvolutionGate` accepts a *time-independent* `SparsePauliOp`.  For a driven Hamiltonian
`H(t) = H_static + H_drive(t)` there is no standard way in Qiskit ≤ 2.3 to schedule a
time-varying coefficient without rebuilding and re-transpiling a circuit at every time step,
which is prohibitively expensive.  The bypass is not a workaround — it is the correct
architectural boundary: Qiskit's circuit layer is appropriate for static or parameterised
circuits; per-slice coefficient evaluation belongs in classical simulation code.

---

## 3. Option A: Keep the Bypass (Recommended)

### Decision: **Adopt Option A permanently.**

The bypass is kept as the **production** path for all time-dependent evolution in both
pipelines.  The Qiskit baseline's purpose is to validate that our JW mapping and Hamiltonian
coefficients agree with Qiskit Nature's derivation; the dynamics core is shared with the
hardcoded pipeline so that fidelity comparisons are strictly apples-to-apples.

### Rationale

| Criterion | Analysis |
|---|---|
| **Correctness parity** | Both pipelines call the same `_evolve_trotter_suzuki2_absolute` and `_evolve_piecewise_exact` kernels.  Any algorithmic difference (ordering, time-sampling rule, etc.) would obscure whether a fidelity gap is physical or numerical. |
| **Dependency footprint** | `qiskit-dynamics` is **not installed** in the current environment.  `requirements.txt` pins `qiskit==0.45.1` (though the live environment has 2.3.0); `qiskit-dynamics` latest is 0.6.0 and has its own JAX/array-backend dependency chain that would be a significant addition to CI. |
| **AGENTS.md contract** | §3 mandates "No Qiskit in the core algorithm path" and §6 prohibits adding heavy dependencies without strong reason.  The drive physics is core algorithm, not validation. |
| **Performance** | The sparse `expm_multiply` path (Prompt 6) already eliminates the O(d³) bottleneck.  For the Hubbard model at L ≤ 6 the current implementation is fast enough; `qiskit-dynamics` would add solver setup overhead with no asymptotic gain for batch-free trajectories. |
| **Maintainability** | One kernel, two pipelines.  If the drive model changes, one edit propagates everywhere.  A Qiskit-specific ODE path would diverge over time. |

### What "Option A" means concretely

- `PauliEvolutionGate` is used **only** for the time-independent Trotter path in
  `qiskit_hubbard_baseline_pipeline.py`.  This boundary is intentional.
- When `has_drive` is `True`, both pipelines unconditionally use the shared
  `_evolve_trotter_suzuki2_absolute` / `_evolve_piecewise_exact` kernels.
- This fact **must** be documented in any output JSON (`settings.drive.propagator_backend:
  "scipy_sparse_expm_multiply"`) so that downstream readers know no Qiskit circuit was used.
- Future extensions to the drive model (multi-tone, phonon coupling, etc.) should extend
  the shared kernels, not introduce a separate Qiskit path.

---

## 4. Option B: Optional qiskit-dynamics Integration (Future Sketch)

Option B is **not adopted** today but is recorded here so that a future developer who genuinely
needs a high-accuracy ODE integrator (e.g., for the Holstein–polaron version of this pipeline)
can integrate it without redesigning the production core.

### What qiskit-dynamics provides

`qiskit_dynamics.HamiltonianModel` accepts a list of static operators and a corresponding list
of time-dependent coefficient functions.  `qiskit_dynamics.Solver` wraps a JAX/numpy ODE
solver (Runge-Kutta or Magnus) around `HamiltonianModel` and returns the propagated state
vector at requested time points.

For `H(t) = H_static + f(t) · H_drive` with a Gaussian-sinusoid `f(t)`:

```python
from qiskit_dynamics import HamiltonianModel, Solver

model = HamiltonianModel(
    operators=[H_static_array, H_drive_array],      # two d×d arrays
    signals=[1.0, drive_signal],                    # 1.0 for static; Signal for drive
    rotating_frame=None,                            # or a rotating frame if needed
)
solver = Solver(
    static_hamiltonian=H_static_array,
    hamiltonian_operators=[H_drive_array],
    hamiltonian_signals=[drive_signal],
    dt=dt,                                          # fixed time step
    method="RK45",                                  # or "magnus4" etc.
    atol=1e-9,
    rtol=1e-9,
)
result = solver.solve(t_span=[0, T], y0=psi0, t_eval=time_points)
```

### Minimal integration design

If Option B were adopted, the guiding principle would be: **optional, non-breaking, isolated**.

```
qiskit_hubbard_baseline_pipeline.py
  └── _evolve_drive_qiskit_dynamics(...)   ← NEW, optional helper
        ├── try: import qiskit_dynamics
        ├──   if available: use HamiltonianModel + Solver
        └──   except ImportError: fall through to _evolve_piecewise_exact (unchanged)

hardcoded_hubbard_pipeline.py
  └── NOT modified — Qiskit-dynamics is Qiskit-specific validation only
```

Key constraints:
1. `qiskit-dynamics` import is **always guarded** by `try/except ImportError`.
2. The function signature is identical to `_evolve_piecewise_exact` — a drop-in.
3. A CLI flag `--use-qiskit-dynamics` (default `False`) activates it; without the flag the
   code path is unreachable.
4. It appears **only** in `qiskit_hubbard_baseline_pipeline.py`, never in the hardcoded
   pipeline or any core module.
5. The JSON output records `propagator_backend: "qiskit_dynamics_rk45"` vs
   `"scipy_sparse_expm_multiply"` so output files are distinguishable.
6. `requirements.txt` does **not** add `qiskit-dynamics` — it remains a *soft* extra-require.

### Pros of Option B

- Higher-order adaptive ODE integrators (RK45, Magnus 4th-order) have global-error
  O(Δt⁴) or better vs O(Δt²) for the midpoint method — relevant for long-time or
  rapidly-oscillating drives.
- Rotating-frame support in `HamiltonianModel` is essential for realistic resonance physics
  (carrier frequency, RWA validation).
- JAX-backend option enables GPU acceleration for large L (≥ 7) without rewriting kernels.
- Natural integration with Qiskit pulse-level description once Holstein–phonon mode-coupling
  is added.

### Cons of Option B

- `qiskit-dynamics` 0.6.0 requires JAX ≥ 0.4 (optional but recommended for performance).
  JAX is a heavy dependency with its own CUDA/XLA version matrix.
- API stability: `qiskit-dynamics` is pre-1.0; `HamiltonianModel` and `Solver` signatures
  changed between 0.4 and 0.5.  Pin risk is real.
- The two pipelines would then use *different* dynamics backends by default, making direct
  fidelity comparison ambiguous unless explicitly controlled.
- The current `expm_multiply` path achieves **26–80× speedup** over dense expm already;
  the marginal gain from adaptive ODE solvers is only relevant at very high accuracy
  requirements (ε ≪ 10⁻⁶) which the current physics studies do not yet need.
- JAX's JIT compilation on first call adds 1–5 s cold-start latency — unacceptable for
  short CI runs.

### Trigger for adopting Option B

Adopt if **any** of the following become true:
1. Drive frequency is comparable to the Hubbard bandwidth (strong-driving / Floquet regime),
   making the Magnus-2 O(Δt²) error visible in observables at accessible step counts.
2. Holstein phonon modes are added (continuous-variable coupling to bosonic modes), which
   requires a genuine ODE solver rather than a piecewise-constant approximation.
3. L ≥ 7 is targeted and GPU acceleration is needed.
4. A rotating-frame transformation is needed to suppress fast oscillations.

---

## 5. Summary

| | Option A (adopted) | Option B (future) |
|---|---|---|
| Drive propagation | shared `expm_multiply` kernel | `qiskit_dynamics.Solver` |
| Qiskit circuit used for drive | ✗ never | ✗ never (dynamics is ODE not circuit) |
| New hard dependency | none | `qiskit-dynamics` (+ optionally JAX) |
| AGENTS.md compliance | ✓ | ✓ if confined to qiskit baseline only |
| Global error order | O(Δt²) midpoint | O(Δt⁴) or adaptive |
| Both pipelines consistent | ✓ identical kernel | ✗ qiskit baseline diverges |
| Recommended | **yes** | when strong-driving / Holstein phonons needed |

**Bottom line:** `PauliEvolutionGate` is the right tool for time-independent Trotter validation.
It is not the right tool for time-dependent Hamiltonian simulation.  The current bypass is the
correct architectural choice.  The shared scipy/sparse kernel should remain the production
dynamics core until the physics demands a higher-order adaptive solver — at which point
`qiskit-dynamics` should be introduced as an *optional* validator in the Qiskit baseline only,
never touching the hardcoded pipeline or any `src/` module.

---

## 6. Required Code Annotation

The following comment block **must** appear in `_simulate_trajectory` in
`qiskit_hubbard_baseline_pipeline.py` at the branch point where `PauliEvolutionGate` is
bypassed, to make the architectural intent clear to future readers:

```python
# PauliEvolutionGate bypass (intentional, not a workaround)
# ─────────────────────────────────────────────────────────
# PauliEvolutionGate accepts only a time-INDEPENDENT SparsePauliOp.
# For a driven Hamiltonian H(t) = H_static + H_drive(t) there is no
# efficient way to schedule time-varying coefficients through the Qiskit
# circuit layer without rebuilding the circuit at every time slice.
#
# Decision (see DESIGN_NOTE_QISKIT_BASELINE_TIMEDEP.md §3):
#   Both pipelines share _evolve_trotter_suzuki2_absolute and
#   _evolve_piecewise_exact — identical kernels, identical results.
#   qiskit-dynamics (Option B) is deferred until strong-driving or
#   Holstein phonon modes make adaptive ODE accuracy necessary.
```
