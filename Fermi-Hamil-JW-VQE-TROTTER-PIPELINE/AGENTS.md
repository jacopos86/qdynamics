
```markdown
<!-- agents.md -->

# Agent Instructions (Repo Rules of Engagement)

This file defines how automated coding agents (and humans) should modify this repository.

The priority is **correctness and consistency of operator conventions**, not “cleverness”.

---

## 1) Non-negotiable conventions

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

---

## 2) Keep the operator layer clean

### Operator layer responsibilities
The following modules are “operator algebra core”:
- `pauli_letters_module.py` (Symbol Product Map + PauliLetter)
- `qubitization_module.py` (PauliTerm)
- `pauli_polynomial_class.py` (PauliPolynomial + JW ladder operators)

Agents should avoid adding algorithm logic here (VQE, QPE, optimizers, simulators).

### Avoid mixing PauliTerm classes
There are two `PauliTerm` definitions:
- `pydephasing.quantum.qubitization_module.PauliTerm`
- `pydephasing.quantum.pauli_words.PauliTerm`

Mixing them can create subtle issues (type checks, multiplication dispatch, shared references).
Rule:
- Inside the package, prefer **one** canonical PauliTerm class and use it everywhere.
- If a standalone script needs fallback logging, it can import `pauli_words.PauliTerm`, but then it should not intermix with `PauliPolynomial` that expects the other type.

If a refactor is needed, consolidate to a single PauliTerm source.

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
- Implement “apply exp(-i θ * PauliTerm)” and “apply exp(-i θ * PauliPolynomial)” as first-class utilities.
- Keep functions that return **term lists** (coeff, pauli_string) available for later grouping/ordering.
- Avoid architectures that require opaque circuit objects.

If adding higher-order Suzuki–Trotter later:
- do it by composition on top of the same primitive exp(PauliTerm) backend.

---

## 5) Style and maintainability

### Clean/simple code
- Prefer pure functions where possible (no hidden global state).
- Keep modules small, with a single responsibility.
- Use explicit types for public function signatures.
- Prefer explicit errors (`log.error(...)` or raising) over silent coercions.

### “LaTeX above Python” pairing
When adding new literate modules (like `hubbard_latex_python_pairs.py`):
- include the LaTeX expression in a string right above the function that implements it
- keep LaTeX and code aligned 1:1

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
- Do not introduce Qiskit into core algorithm modules.
- Do not add heavy dependencies without a strong reason.
- Do not “optimize” by rewriting algebra rules unless correctness is proven with regression tests.

---

## 7) Suggested next implementation steps (for agents)
1. Replace `quantum_eigensolver.py` stub with a hardcoded VQE driver calling into a dedicated VQE module.
2. Add a VQE literate module (LaTeX+Python pairs) mirroring the Hubbard pairs pattern.
3. Add a small regression runner that:
   - builds H for L=2,3 (blocked & interleaved)
   - compares canonical Pauli dictionaries vs existing JSON references
   - runs hardcoded VQE and compares energy to exact filtered diagonalization for small sizes.
