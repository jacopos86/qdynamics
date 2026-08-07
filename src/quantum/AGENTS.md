# src/quantum AGENTS.md

This subtree owns operator algebra, Hamiltonian builders, ansatz/statevector
math, and reusable quantum primitives. Correctness of conventions outranks
cleverness.

## Operator Conventions

- Use `e/x/y/z` internally; convert to `I/X/Y/Z` only at output/report boundaries.
- Pauli words are ordered left-to-right as `q_(n-1) ... q_0`; qubit 0 is the rightmost character.
- All statevector bit indexing must match that Pauli-word convention.
- Do not re-derive JW mapping ad hoc. Use `fermion_plus_operator(repr_mode="JW", nq, j)` and `fermion_minus_operator(repr_mode="JW", nq, j)` from `pauli_polynomial_class.py`.
- Number operators use `n_p = (I - Z_p)/2` in the repo ordering convention.

## PauliTerm Source

Canonical `PauliTerm` source:
- `src.quantum.qubitization_module.PauliTerm`

Compatibility aliases, same class and not separate definitions:
- `src.quantum.pauli_words.PauliTerm`
- `pydephasing.quantum.pauli_words.PauliTerm`

Rules:
- Core package code imports `PauliTerm` from `qubitization_module.py`.
- Compatibility scripts may import `pauli_words.PauliTerm` only for existing interfaces.
- Do not introduce another `PauliTerm` implementation.

## Operator Layer Boundaries

Operator algebra core:
- `pauli_letters_module.py`
- `qubitization_module.py`
- `pauli_polynomial_class.py`

Base operator files should remain unchanged unless the user explicitly asks for
operator-core work. Prefer wrappers/shims for repo integration changes around
these files.

## Math And Validation

- For new physics/model code, keep the implemented equation or symbolic contract adjacent to the function, method, or class that implements it.
- Use explicit types for public function signatures and explicit errors over silent coercions.
- When modifying Hamiltonian construction, indexing conventions, JW mapping, or number operators, update or re-run the relevant reference checks, for example `hubbard_jw_*.json` for Hubbard/JW changes and HH/model-specific fixtures for expanded Hamiltonian families.
- Qiskit may be used for validation/reference scripts, not production/core algorithm paths.
