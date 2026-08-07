#!/usr/bin/env python3
"""Benchmark-local Qiskit Algorithms AdaptVQE adapter.

Qiskit and qiskit-algorithms are intentionally optional and isolated to
``pipelines.exact_bench``.  This module provides lazy imports plus small
conversion helpers for the exact-bench-only ``static_qiskit_adapt_vqe`` row; it
is not a production/static_adapt dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

_DEPENDENCY_MESSAGE = (
    "Qiskit AdaptVQE benchmark support requires qiskit.circuit.QuantumCircuit, "
    "qiskit.quantum_info.SparsePauliOp/Statevector, qiskit.primitives.StatevectorEstimator, "
    "qiskit_algorithms.minimum_eigensolvers.AdaptVQE/VQE, and "
    "qiskit_algorithms.optimizers.COBYLA."
)


class QiskitAdaptVQEUnavailable(ImportError):
    """Raised when optional benchmark-only Qiskit AdaptVQE support is unavailable."""


@dataclass(frozen=True)
class QiskitAdaptVQEComponents:
    QuantumCircuit: Any
    SparsePauliOp: Any
    Statevector: Any
    StatevectorEstimator: Any
    AdaptVQE: Any
    VQE: Any
    COBYLA: Any


def import_qiskit_adaptvqe_components() -> QiskitAdaptVQEComponents:
    """Import optional Qiskit AdaptVQE components lazily."""
    try:
        from qiskit.circuit import QuantumCircuit
        from qiskit.primitives import StatevectorEstimator
        from qiskit.quantum_info import SparsePauliOp, Statevector
        from qiskit_algorithms.minimum_eigensolvers import AdaptVQE, VQE
        from qiskit_algorithms.optimizers import COBYLA
    except Exception as exc:  # pragma: no cover - optional-dep failure varies by install
        raise QiskitAdaptVQEUnavailable(_DEPENDENCY_MESSAGE) from exc
    return QiskitAdaptVQEComponents(
        QuantumCircuit=QuantumCircuit,
        SparsePauliOp=SparsePauliOp,
        Statevector=Statevector,
        StatevectorEstimator=StatevectorEstimator,
        AdaptVQE=AdaptVQE,
        VQE=VQE,
        COBYLA=COBYLA,
    )


def has_qiskit_adaptvqe_support() -> bool:
    """Return whether optional benchmark-local AdaptVQE support is importable."""
    try:
        import_qiskit_adaptvqe_components()
    except Exception:
        return False
    return True


def _to_ixyz(label_exyz: str) -> str:
    return (
        str(label_exyz)
        .replace("e", "I")
        .replace("x", "X")
        .replace("y", "Y")
        .replace("z", "Z")
    )


def _polynomial_terms(poly: Any) -> list[Any]:
    try:
        return list(poly.return_polynomial())
    except Exception as exc:
        raise ValueError("Expected a repo PauliPolynomial-like object with return_polynomial().") from exc


def _term_label(term: Any) -> str:
    try:
        return str(term.pw2strng())
    except Exception as exc:
        raise ValueError("Expected Pauli terms to expose pw2strng().") from exc


def _term_coeff(term: Any) -> complex:
    try:
        return complex(term.p_coeff)
    except Exception as exc:
        raise ValueError("Expected Pauli terms to expose p_coeff.") from exc


def _term_nqubit(term: Any) -> int:
    try:
        return int(term.nqubit())
    except Exception as exc:
        raise ValueError("Expected Pauli terms to expose nqubit().") from exc


def hamiltonian_term_pool_labels(
    poly: Any,
    *,
    max_terms: int | None = 128,
    tol: float = 1e-12,
) -> tuple[str, ...]:
    """Return unique non-identity repo ``e/x/y/z`` Hamiltonian Pauli labels.

    Labels are preserved in repo order and converted to Qiskit only at the
    SparsePauliOp boundary.  Coefficients are intentionally ignored for the pool:
    each returned label becomes one unit-coefficient append-only AdaptVQE
    operator.
    """
    labels: list[str] = []
    seen: set[str] = set()
    for term in _polynomial_terms(poly):
        coeff = _term_coeff(term)
        if abs(coeff) <= float(tol):
            continue
        label = _term_label(term).lower()
        if not label:
            continue
        if label == "e" * len(label):
            continue
        if label in seen:
            continue
        labels.append(label)
        seen.add(label)
        if max_terms is not None and len(labels) > int(max_terms):
            raise ValueError(
                f"Hamiltonian Pauli-term pool exceeds cap: {len(labels)} > {int(max_terms)}"
            )
    return tuple(labels)


def pauli_poly_to_sparse_pauli_op(
    poly: Any,
    *,
    sparse_pauli_op_cls: Any | None = None,
    tol: float = 1e-12,
) -> Any:
    """Convert a repo PauliPolynomial (``e/x/y/z`` labels) into SparsePauliOp."""
    SparsePauliOp = sparse_pauli_op_cls
    if SparsePauliOp is None:
        SparsePauliOp = import_qiskit_adaptvqe_components().SparsePauliOp

    terms = _polynomial_terms(poly)
    if not terms:
        nq = int(getattr(poly, "get_nq", lambda: 1)())
        return SparsePauliOp.from_list([("I" * max(1, nq), 0.0 + 0.0j)])

    nq = _term_nqubit(terms[0])
    coeff_map: dict[str, complex] = {}
    for term in terms:
        coeff = _term_coeff(term)
        if abs(coeff) <= float(tol):
            continue
        label = _to_ixyz(_term_label(term).lower())
        if len(label) != nq:
            raise ValueError(f"Pauli label length mismatch: got {len(label)}, expected {nq}")
        coeff_map[label] = coeff_map.get(label, 0.0 + 0.0j) + coeff

    cleaned = [(label, coeff) for label, coeff in coeff_map.items() if abs(coeff) > float(tol)]
    if not cleaned:
        cleaned = [("I" * nq, 0.0 + 0.0j)]
    return SparsePauliOp.from_list(cleaned).simplify(atol=float(tol))


def hamiltonian_term_pool_to_sparse_pauli_ops(
    poly: Any,
    *,
    sparse_pauli_op_cls: Any | None = None,
    max_terms: int = 128,
    tol: float = 1e-12,
) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    """Build one unit-coefficient SparsePauliOp per Hamiltonian Pauli label."""
    SparsePauliOp = sparse_pauli_op_cls
    if SparsePauliOp is None:
        SparsePauliOp = import_qiskit_adaptvqe_components().SparsePauliOp
    labels_exyz = hamiltonian_term_pool_labels(poly, max_terms=max_terms, tol=tol)
    ops = tuple(SparsePauliOp.from_list([(_to_ixyz(label), 1.0 + 0.0j)]) for label in labels_exyz)
    return ops, labels_exyz


def _basis_index_if_one_hot(state: np.ndarray, *, tol: float = 1e-12) -> int | None:
    ref = np.asarray(state, dtype=complex).reshape(-1)
    nz = np.where(np.abs(ref) > float(tol))[0]
    if nz.size != 1:
        return None
    idx = int(nz[0])
    amp = complex(ref[idx])
    if abs(abs(amp) - 1.0) > 1e-10:
        return None
    return idx


def build_reference_state_circuit(
    reference_state: np.ndarray | Sequence[complex],
    *,
    num_qubits: int,
    quantum_circuit_cls: Any | None = None,
) -> Any:
    """Build a Qiskit circuit preparing the repo reference state.

    One-hot basis states are prepared with X gates using the repo convention that
    qubit 0 is the rightmost/least-significant bit.  General states fall back to
    Qiskit's ``initialize`` instruction.
    """
    QuantumCircuit = quantum_circuit_cls
    if QuantumCircuit is None:
        QuantumCircuit = import_qiskit_adaptvqe_components().QuantumCircuit
    nq = int(num_qubits)
    if nq <= 0:
        raise ValueError("num_qubits must be positive for reference-state preparation.")
    ref = np.asarray(reference_state, dtype=complex).reshape(-1)
    dim = int(1 << nq)
    if int(ref.size) != dim:
        raise ValueError(f"reference_state dimension {ref.size} does not match num_qubits={nq}")
    norm = float(np.linalg.norm(ref))
    if norm <= 0.0:
        raise ValueError("reference_state has zero norm")
    ref = ref / norm

    circuit = QuantumCircuit(nq)
    idx = _basis_index_if_one_hot(ref)
    if idx is not None:
        bit_string = format(idx, f"0{nq}b")
        for qubit in range(nq):
            if bit_string[nq - 1 - qubit] == "1":
                circuit.x(qubit)
        return circuit

    circuit.initialize(ref, list(range(nq)))
    return circuit


__all__ = [
    "QiskitAdaptVQEComponents",
    "QiskitAdaptVQEUnavailable",
    "build_reference_state_circuit",
    "hamiltonian_term_pool_labels",
    "hamiltonian_term_pool_to_sparse_pauli_ops",
    "has_qiskit_adaptvqe_support",
    "import_qiskit_adaptvqe_components",
    "pauli_poly_to_sparse_pauli_op",
]
