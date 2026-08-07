#!/usr/bin/env python3
"""Benchmark-local Qiskit hardware-efficient ansatz adapter.

Qiskit is intentionally isolated to ``pipelines.exact_bench``.  The adapter only
prepares statevectors for the repo-native VQE objective; it is not a production
ADAPT/VQE dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

_QISKIT_HEA_DEPENDENCY_MESSAGE = (
    "Qiskit benchmark-only HEA support requires qiskit.circuit.QuantumCircuit, "
    "qiskit.circuit.ParameterVector, and qiskit.quantum_info.Statevector."
)


class QiskitHeaUnavailable(ImportError):
    """Raised when optional benchmark-only Qiskit HEA support is unavailable."""


def import_qiskit_hea_components() -> tuple[Any, Any, Any]:
    """Import optional Qiskit HEA components lazily."""
    try:
        from qiskit.circuit import ParameterVector, QuantumCircuit
        from qiskit.quantum_info import Statevector
    except Exception as exc:  # pragma: no cover - exact exception varies by install
        raise QiskitHeaUnavailable(_QISKIT_HEA_DEPENDENCY_MESSAGE) from exc
    return QuantumCircuit, ParameterVector, Statevector


def has_qiskit_hea_support() -> bool:
    """Return whether optional benchmark-only Qiskit HEA support is importable."""
    try:
        import_qiskit_hea_components()
    except Exception:
        return False
    return True


def num_qubits_from_state_vector(psi_ref: np.ndarray) -> int:
    size = int(np.asarray(psi_ref).size)
    if size <= 0:
        raise ValueError("psi_ref must be a non-empty state vector.")
    if size & (size - 1):
        raise ValueError("psi_ref length must be a power of two for Qiskit HEA evolution.")
    return int(size.bit_length() - 1)


@dataclass(frozen=True)
class QiskitHeaCircuitStats:
    depth: int | None
    count_2q: int | None
    op_counts: dict[str, int]


class QiskitStatevectorAnsatzAdapter:
    """Minimal Qiskit circuit adapter for the repo-native VQE minimizer."""

    ansatz_name = "qiskit_hea_linear_ryrz_cx"

    def __init__(self, *, circuit: Any, parameters: Sequence[Any], statevector_cls: Any) -> None:
        self._circuit = circuit
        self._parameters = tuple(parameters)
        self._statevector_cls = statevector_cls
        self.num_qubits = int(getattr(circuit, "num_qubits"))
        self.num_parameters = int(len(self._parameters))

    @property
    def circuit(self) -> Any:
        return self._circuit

    @property
    def parameters(self) -> tuple[Any, ...]:
        return self._parameters

    def circuit_stats(self) -> QiskitHeaCircuitStats:
        depth: int | None
        try:
            depth = int(self._circuit.depth())
        except Exception:
            depth = None
        op_counts_raw: dict[str, int] = {}
        count_2q = 0
        try:
            op_counts_raw = {str(k): int(v) for k, v in dict(self._circuit.count_ops()).items()}
        except Exception:
            op_counts_raw = {}
        try:
            for item in self._circuit.data:
                operation = getattr(item, "operation", None)
                if operation is None and isinstance(item, (tuple, list)) and item:
                    operation = item[0]
                if int(getattr(operation, "num_qubits", 0)) == 2:
                    count_2q += 1
        except Exception:
            count_2q = int(op_counts_raw.get("cx", 0) + op_counts_raw.get("cz", 0))
        return QiskitHeaCircuitStats(depth=depth, count_2q=int(count_2q), op_counts=op_counts_raw)

    def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=float).ravel()
        if int(theta_arr.size) != int(self.num_parameters):
            raise ValueError(
                "theta size mismatch for Qiskit HEA ansatz: "
                f"expected {int(self.num_parameters)}, got {int(theta_arr.size)}"
            )
        psi_ref_arr = np.asarray(psi_ref, dtype=complex).ravel()
        num_qubits = num_qubits_from_state_vector(psi_ref_arr)
        if num_qubits != int(self.num_qubits):
            raise ValueError(
                "psi_ref qubit count does not match Qiskit HEA circuit: "
                f"state has {num_qubits}, circuit has {int(self.num_qubits)}"
            )
        assignments = {param: float(theta_arr[idx]) for idx, param in enumerate(self._parameters)}
        bound_circuit = self._circuit.assign_parameters(assignments, inplace=False)
        evolved = self._statevector_cls(psi_ref_arr).evolve(bound_circuit)
        return np.asarray(getattr(evolved, "data", evolved), dtype=complex).ravel()


def build_qiskit_hea_ansatz(*, num_qubits: int, reps: int) -> QiskitStatevectorAnsatzAdapter:
    """Build a simple benchmark-only linear HEA circuit adapter."""
    n_qubits = int(num_qubits)
    n_reps = int(reps)
    if n_qubits <= 0:
        raise ValueError("num_qubits must be positive for Qiskit HEA ansatz.")
    if n_reps <= 0:
        raise ValueError("reps must be positive for Qiskit HEA ansatz.")
    QuantumCircuit, ParameterVector, Statevector = import_qiskit_hea_components()
    circuit = QuantumCircuit(n_qubits)
    parameter_count = int(2 * n_qubits * (n_reps + 1))
    theta = ParameterVector("theta", parameter_count)
    ordered_parameters = [theta[idx] for idx in range(parameter_count)]
    cursor = 0
    for _layer in range(n_reps):
        for qubit in range(n_qubits):
            circuit.ry(theta[cursor], qubit)
            cursor += 1
            circuit.rz(theta[cursor], qubit)
            cursor += 1
        for qubit in range(n_qubits - 1):
            circuit.cx(qubit, qubit + 1)
    for qubit in range(n_qubits):
        circuit.ry(theta[cursor], qubit)
        cursor += 1
        circuit.rz(theta[cursor], qubit)
        cursor += 1
    if cursor != parameter_count:
        raise RuntimeError("internal Qiskit HEA parameter-count mismatch")
    return QiskitStatevectorAnsatzAdapter(
        circuit=circuit,
        parameters=ordered_parameters,
        statevector_cls=Statevector,
    )


__all__ = [
    "QiskitHeaCircuitStats",
    "QiskitHeaUnavailable",
    "QiskitStatevectorAnsatzAdapter",
    "build_qiskit_hea_ansatz",
    "has_qiskit_hea_support",
    "import_qiskit_hea_components",
    "num_qubits_from_state_vector",
]
