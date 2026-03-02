
```markdown
<!-- agents.md -->

# Agent Instructions (Repo Rules of Engagement)

This file defines how automated coding agents (and humans) should modify this repository.

The priority is **correctness and consistency of operator conventions**, not “cleverness”.

---

## 1) Non-negotiable conventions

### Runbook authority for operational workflows
- For HH staging, run presets, and the new `ecut_1`/`ecut_2` interpretation, agents must consult [`pipelines/run_guide.md`](/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test/pipelines/run_guide.md) before editing pipeline invocation defaults, scaling tables, or manual run plans.
- Treat this as the canonical source for execution contracts that are not operator-level invariants (e.g., thresholds, gating policy, and recommended run ladders).

### Pauli symbols
- Use `e/x/y/z` internally (`e` = identity)
- If you need I/X/Y/Z output for reports, convert at the boundaries only.

### Pauli-string qubit ordering
- Pauli word string is ordered:
  - **left-to-right = q_(n-1) ... q_0**
  - **qubit 0 is rightmost character**
- All statevector bit indexing must match this.

### JW mapping source of truth
Do not re-derive JW mapping ad-hoc in new files. Use:
- `fermion_plus_operator(repr_mode="JW", nq, j)`
- `fermion_minus_operator(repr_mode="JW", nq, j)`
from `pauli_polynomial_class.py`

If you need number operators, implement:
- `n_p = (I - Z_p)/2`
in a way consistent with the Pauli-string convention above.

### PDF artifact parameter manifest (mandatory)
Every generated PDF artifact must include a **clear, list-style parameter manifest at the start of the document** (first page or first text-summary page), not just a raw command dump.

Required fields:
- Model family/name (for this repo: `Hubbard` unless/ until additional models are added)
- Ansatz type(s) used
- Whether drive is enabled (`--enable-drive` true/false)
- Core physical parameters: `t`, `U`, `dv`
- Any other run-defining parameters needed to reproduce the physics for that PDF

This rule applies to all PDF outputs (single-pipeline PDFs, compare PDFs, bundle PDFs, amplitude-comparison PDFs, and future report PDFs).

---

## 2) Keep the operator layer clean

### Operator layer responsibilities
The following modules are “operator algebra core”:
- `pauli_letters_module.py` (Symbol Product Map + PauliLetter)
- `qubitization_module.py` (PauliTerm)
- `pauli_polynomial_class.py` (PauliPolynomial + JW ladder operators)

### PauliTerm canonical source (mandatory)
Canonical `PauliTerm` source:
- `src.quantum.qubitization_module.PauliTerm`

Compatibility aliases (same class, not separate definitions):
- `src.quantum.pauli_words.PauliTerm`
- `pydephasing.quantum.pauli_words.PauliTerm`

Rules:
- Core package code **must** import `PauliTerm` from `qubitization_module.py`.
- Compatibility scripts may import `pauli_words.PauliTerm` only when required by existing interfaces; they must not introduce a new `PauliTerm` implementation.
- Base operator files **must remain unchanged**: `pauli_letters_module.py`, `pauli_words.py`, `qubitization_module.py`.
- Repo integration changes **must** be implemented with wrappers/shims around base files.

---

## 3) VQE implementation rules

### No Qiskit in the core VQE path
Qiskit is allowed only for:
- validation scripts/notebooks
- reference data generation/comparison

The production VQE path must be:
- numpy statevector backend
- minimal optimizer dependencies (SciPy optional; provide fallback if absent)

### VQE structure (required)
Implement VQE using the notation:

- Hamiltonian:
  `H = Σ_j h_j P_j`
- Energy:
  `E(θ) = Σ_j h_j ⟨ψ(θ)|P_j|ψ(θ)⟩`
- Ansatz:
  `|ψ(θ)⟩ = U_p(θ_p)…U_1(θ_1)|ψ_ref⟩`

### Ansatz selection
Default ansatz should be compatible with future time evolution:
- prefer “term-wise” or “Hamiltonian-variational” style layers
- each unitary should be representable as exp(-i θ * (PauliPolynomial))

Do not hardcode an ansatz that cannot be decomposed into Pauli exponentials.

---

## 4) Time-dynamics readiness (Suzuki–Trotter / QPE)

When implementing primitives, favor ones reusable for time evolution:
- Implement "apply exp(-i θ * PauliTerm)" and "apply exp(-i θ * PauliPolynomial)" as first-class utilities.
- Keep functions that return **term lists** (coeff, pauli_string) available for later grouping/ordering.
- Avoid architectures that require opaque circuit objects.

If adding higher-order Suzuki–Trotter later:
- do it by composition on top of the same primitive exp(PauliTerm) backend.

## 4a) Time-dependent drive implementation rules

The repo supports a **time-dependent onsite density drive** with a Gaussian-envelope sinusoidal waveform:

```
v(t) = A · sin(ω t + φ) · exp(-(t - t₀)² / (2 t̄²))
```

### Drive architecture
- The drive waveform and spatial patterns are defined in `src/quantum/drives_time_potential.py` (if present) or inline in the pipeline files.
- The compare pipeline forwards drive flags verbatim to both sub-pipelines via `_build_drive_args()` and `_build_drive_args_with_amplitude()`.
- All drive parameters are pass-through CLI flags; the compare pipeline does **not** interpret drive physics, only routes them.

### Drive reference propagator
- When drive is enabled, the **reference (exact) propagator** uses `scipy.sparse.linalg.expm_multiply` with piecewise-constant H(t) at each time step.
- The `--exact-steps-multiplier` flag controls refinement: `N_ref = multiplier × trotter_steps`.
- The `reference_method_name` in JSON output records which method was used (`expm_multiply_sparse_timedep` vs `eigendecomposition`).
- When drive is disabled, the static reference uses exact eigendecomposition — no changes.

### Spatial patterns
Three built-in patterns (`--drive-pattern`):
| Pattern | Weights s_j per site j |
|---------|------------------------|
| `staggered` | `(-1)^j` alternating sign |
| `dimer_bias` | `[+1, -1, +1, -1, ...]` (same as staggered for even L) |
| `custom` | User-supplied JSON array via `--drive-custom-s` |

### Rules for agents modifying drive code
- Do **not** add new drive parameters without also updating: (1) both pipeline `parse_args()`, (2) the compare pipeline's `_build_drive_args()` and `_build_drive_args_with_amplitude()`, (3) `PIPELINE_RUN_GUIDE.md`.
- Drive must be **opt-in** (`--enable-drive`). Default behaviour (no flag) must be bit-for-bit identical to the static case.
- All drive-related CLI args must have the `--drive-` prefix (except `--enable-drive` and `--exact-steps-multiplier`).
- The safe-test (`_safe_test_check`) must remain: A=0 drive must produce trajectories identical to the no-drive case within `_SAFE_TEST_THRESHOLD = 1e-10`.

## 4b) Drive amplitude comparison PDF

The compare pipeline supports `--with-drive-amplitude-comparison-pdf` which:
1. Runs both pipelines 3× per L: drive-disabled, A0-enabled, A1-enabled (6 sub-runs total per L).
2. Generates a multi-page physics-facing PDF per L with scoreboard tables, drive waveform, response deltas, and a combined HC/QK overlay.
3. Writes `json/amp_cmp_hubbard_{tag}_metrics.json` with `safe_test`, `delta_vqe_hc_minus_qk_at_A0`, `delta_vqe_hc_minus_qk_at_A1`.

### Rules for agents modifying amplitude comparison
- All artifacts go to `json/` or `pdf/` subdirectories. Filenames use the tag convention `L{L}_{drive|static}_t{t}_U{u}_S{steps}`.
- Intermediate JSON files use the `amp_hc_hubbard_` / `amp_qk_hubbard_` prefix: `json/amp_hc_hubbard_L2_static_t1.0_U4.0_S32_disabled.json`, `json/amp_qk_hubbard_L2_static_t1.0_U4.0_S32_A0.json`, etc.
- Safe-test scalar metrics must always be reported on the scoreboard table. The full safe-test timeseries page is conditional (fail, near-threshold, or `--report-verbose`).
- VQE delta is defined as `ΔE = VQE_hardcoded − VQE_qiskit` (the sector-filtered energy, not full-Hilbert).
- New amplitude comparison CLI args: `--drive-amplitudes A0,A1`, `--with-drive-amplitude-comparison-pdf`, `--report-verbose`, and `--safe-test-near-threshold-factor`.

## 4c) User shorthand run convention (`run L`)

When the user requests a shorthand run like:
- "run L=4"
- "run a number L"
- "run L 5"

interpret it with the following **default contract**:

1. The run is **drive-enabled, never static**.
2. The run is **accuracy-gated** and must target:
   `abs(vqe.energy - ground_state.exact_energy_filtered) < 1e-4`.
3. Use **L-scaled heaviness** (stronger settings for larger L), not one-size-fits-all settings.

Implementation rule:
- Prefer `pipelines/run_L_drive_accurate.sh --L <L>` when available.
- If this script is unavailable, emulate its semantics manually:
  - drive enabled with scaling profile defaults,
  - per-L parameter table,
  - fallback escalation until the `1e-7` gate passes or budget is exhausted.

## 4d) Mandatory minimum VQE / Trotter parameters per L

**Agents must never run a pipeline with settings weaker than the table below.**
Under-parameterised runs waste wall-clock time and produce unconverged results
that are useless for diagnostics — you cannot tell whether a failure is a code
bug or just insufficient optimiser effort.

If in doubt, **round up** to the next row.

### Hubbard (pure) — minimum settings

| L | `--trotter-steps` | `--exact-steps-multiplier` | `--num-times` | `--vqe-reps` | `--vqe-restarts` | `--vqe-maxiter` | optimizer | `--t-final` |
|---|---|---|---|---|---|---|---|---|
| 2 | 64 | 2 | 201 | 2 | 2 | 600 | COBYLA | 10.0 |
| 3 | 128 | 2 | 201 | 2 | 3 | 1200 | COBYLA | 15.0 |
| 4 | 256 | 3 | 241 | 3 | 4 | 6000 | SLSQP | 20.0 |
| 5 | 384 | 3 | 301 | 3 | 5 | 8000 | SLSQP | 20.0 |
| 6 | 512 | 4 | 361 | 4 | 6 | 10000 | SLSQP | 20.0 |

### Hubbard-Holstein (HH) — minimum settings

HH requires heavier settings than pure Hubbard at the same L due to
the enlarged Hilbert space (phonon modes).

| L | `--n-ph-max` | `--trotter-steps` | `--vqe-reps` | `--vqe-restarts` | `--vqe-maxiter` | optimizer |
|---|---|---|---|---|---|---|
| 2 | 1 | 64 | 2 | 3 | 800 | COBYLA |
| 2 | 2 | 128 | 3 | 4 | 1500 | COBYLA |
| 3 | 1 | 192 | 2 | 4 | 2400 | COBYLA |

### Rules

1. **Never use L=2 defaults for L≥3.** The Hilbert space grows as $2^{2L}$
   (Hubbard) or $2^{2L} \cdot (n_{ph}+1)^L$ (HH). Parameters that converge
   at L=2 are catastrophically insufficient at L=3+.
2. If the user says "run L=3" without specifying parameters, use this table
   (or `run_L_drive_accurate.sh`) — do not invent lighter settings.
3. For validation / smoke-test runs that intentionally use weak settings,
   add an explicit comment: `# SMOKE TEST — intentionally weak settings`.
4. When writing tests, light settings (e.g., `maxiter=40`) are acceptable
   because tests verify implementation correctness, not convergence quality.
   But pipeline runs and demo artifacts must meet the table above.

## 4e) Cross-check suite (`pipelines/exact_bench/cross_check_suite.py`)

The cross-check suite compares **all available ansätze × VQE modes** against
exact ED for a given L, with Trotter dynamics and multi-page PDF output.

### Usage

```bash
# Pure Hubbard, auto-scaled parameters from §4d:
python pipelines/exact_bench/cross_check_suite.py --L 2 --problem hubbard

# Hubbard-Holstein:
python pipelines/exact_bench/cross_check_suite.py --L 2 --problem hh --omega0 1.0 --g-ep 0.5

# Override auto-scaled params (e.g. for smoke tests):
python pipelines/exact_bench/cross_check_suite.py --L 2 --problem hubbard \
  --vqe-reps 1 --vqe-restarts 1 --vqe-maxiter 40 --trotter-steps 8
```

### Shorthand convention

When the user says "run cross-check L=3" or "cross-check L 4":
1. Run `cross_check_suite.py --L <L> --problem hubbard` (auto-scaled from §4d).
2. If the user says "cross-check HH L=2", add `--problem hh`.
3. Do **not** override `--vqe-maxiter` or `--trotter-steps` below §4d minimums.

### Trial matrix

| Problem | Ansätze |
|---------|---------|
| `hubbard` | HVA-Layerwise, UCCSD-Layerwise, ADAPT(UCCSD), ADAPT(full_H) |
| `hh` | HH-Termwise, HH-Layerwise, ADAPT(full_H) |

### Output

- JSON: `<output-dir>/xchk_L{L}_{problem}_t{t}_U{U}.json`
- PDF: same path with `.pdf` — parameter manifest, scoreboard table, per-ansatz 3-panel trajectory plots (fidelity, energy, occupation), fidelity/energy/doublon overlay pages, command page.

---

## 5) Style and maintainability

### Clean/simple code
- Prefer pure functions where possible (no hidden global state).
- Keep modules small, with a single responsibility.
- Use explicit types for public function signatures.
- Prefer explicit errors (`log.error(...)` or raising) over silent coercions.

### Built Math-Symbols/Description above Python pairing
When adding new modules :
- include the Built-in math symbolic expression in a string right above the function that implements it
- keep the math and code aligned 1:1

### Regression/validation
Whenever you modify:
- Hubbard Hamiltonian construction
- indexing conventions
- JW mapping / number operator

You must update or re-run reference checks against:
- `hubbard_jw_*.json`

Qiskit baseline scripts may be used to sanity check, but they are not the core test oracle.

---

## 6) What an agent should NOT do
- Do not change Pauli-string ordering conventions.
- Do not introduce Qiskit into core/'hardcoded' algorithm modules.
- Do not add heavy dependencies without a strong reason.
- Do not "optimize" by rewriting algebra rules unless correctness is proven with regression tests.
- Do not add new drive parameters without updating all three pipelines' `parse_args()`, `_build_drive_args()`, `_build_drive_args_with_amplitude()`, and `PIPELINE_RUN_GUIDE.md`.
- Do not break the safe-test invariant (A=0 drive must equal no-drive to machine precision).
- Do not stop a run because you think it is taking up too much run-time. The only acceptable reason to stop/interrupt an already active run/script is for debugging.
- **Do not run a pipeline with parameters below the §4d minimum table.** If the user does not specify parameters, look up the table — never guess or use L=2 defaults for larger L.



--Note -- Take your time coding! Be safe, and do not rush. The user has a lot of time and does not need things quickly.

## Plans

- Make the plan extremely consise. Sacrifice grammar for the sake of concision.
- Near the end of each plan, give me a list of unresolved questions to answer/problems, if any, and the files you will edit.
- At the end of each plan, state all files intended to alter, and functions and classes to be altered. If none, write 'Files to edit: None'.

