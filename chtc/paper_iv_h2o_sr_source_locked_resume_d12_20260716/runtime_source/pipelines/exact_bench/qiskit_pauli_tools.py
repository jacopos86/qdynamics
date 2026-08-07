#!/usr/bin/env python3
"""Lazy benchmark-local Qiskit Pauli/circuit helpers.

The repo keeps Pauli labels internally in lower-case ``e/x/y/z`` with the
left-to-right convention ``q_(n-1) ... q_0``.  Qiskit labels are upper-case
``I/X/Y/Z`` at the SparsePauliOp boundary; Qiskit circuit qubit index 0 remains
repo qubit 0, the rightmost label character.  This module deliberately imports
Qiskit lazily and is isolated under ``pipelines.exact_bench``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

_QISKIT_PAULI_DEPENDENCY_MESSAGE = (
    "Qiskit dynamics parity support requires qiskit.circuit.QuantumCircuit and "
    "qiskit.quantum_info.Statevector/SparsePauliOp."
)


class QiskitPauliUnavailable(ImportError):
    """Raised when optional benchmark-local Qiskit Pauli support is unavailable."""


@dataclass(frozen=True)
class QiskitPauliComponents:
    QuantumCircuit: Any
    Statevector: Any
    SparsePauliOp: Any


@dataclass(frozen=True)
class ParameterizedRuntimeCircuit:
    """Qiskit ansatz circuit with parameters in repo runtime-vector order."""

    circuit: Any
    parameters: tuple[Any, ...]
    parameter_order: str = "repo_runtime_parameter_order"


def import_qiskit_pauli_components() -> QiskitPauliComponents:
    """Import optional Qiskit components lazily."""

    try:
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp, Statevector
    except Exception as exc:  # pragma: no cover - depends on optional install state
        raise QiskitPauliUnavailable(_QISKIT_PAULI_DEPENDENCY_MESSAGE) from exc
    return QiskitPauliComponents(
        QuantumCircuit=QuantumCircuit,
        Statevector=Statevector,
        SparsePauliOp=SparsePauliOp,
    )


def has_qiskit_pauli_support() -> bool:
    """Return whether optional benchmark-local Qiskit Pauli support is importable."""

    try:
        import_qiskit_pauli_components()
    except Exception:
        return False
    return True


def to_ixyz_label(label_exyz: str) -> str:
    """Convert a repo exyz Pauli label to a Qiskit IXYZ label without reversal."""

    label = str(label_exyz).strip().lower()
    allowed = {"e", "x", "y", "z"}
    bad = sorted(set(label) - allowed)
    if bad:
        raise ValueError(f"unsupported exyz Pauli symbols {bad!r} in {label_exyz!r}")
    return label.replace("e", "I").replace("x", "X").replace("y", "Y").replace("z", "Z")


def _basis_index_if_one_hot(state: np.ndarray, *, tol: float = 1e-12) -> int | None:
    arr = np.asarray(state, dtype=complex).reshape(-1)
    nz = np.where(np.abs(arr) > float(tol))[0]
    if nz.size != 1:
        return None
    idx = int(nz[0])
    amp = complex(arr[idx])
    if abs(abs(amp) - 1.0) > 1e-10:
        return None
    return idx


def append_reference_state(circuit: Any, reference_state: np.ndarray | Sequence[complex]) -> None:
    """Append gates that prepare ``reference_state`` on an existing Qiskit circuit."""

    ref = np.asarray(reference_state, dtype=complex).reshape(-1)
    nq = int(getattr(circuit, "num_qubits"))
    dim = int(1 << nq)
    if int(ref.size) != dim:
        raise ValueError(f"reference_state dimension {ref.size} does not match num_qubits={nq}")
    norm = float(np.linalg.norm(ref))
    if norm <= 0.0:
        raise ValueError("reference_state has zero norm")
    ref = ref / norm

    idx = _basis_index_if_one_hot(ref)
    if idx is not None:
        bit_string = format(idx, f"0{nq}b")
        for qubit in range(nq):
            if bit_string[nq - 1 - qubit] == "1":
                circuit.x(qubit)
        return

    circuit.initialize(ref, list(range(nq)))


def build_reference_state_circuit(
    reference_state: np.ndarray | Sequence[complex],
    *,
    num_qubits: int,
    quantum_circuit_cls: Any | None = None,
) -> Any:
    """Return a Qiskit circuit preparing the repo reference state."""

    QuantumCircuit = quantum_circuit_cls
    if QuantumCircuit is None:
        QuantumCircuit = import_qiskit_pauli_components().QuantumCircuit
    circuit = QuantumCircuit(int(num_qubits))
    append_reference_state(circuit, reference_state)
    return circuit


def append_pauli_rotation_exyz(circuit: Any, *, label_exyz: str, angle: Any) -> None:
    """Append ``exp(-i angle/2 P)`` for a repo ``e/x/y/z`` Pauli label.

    The repo label index ``0`` denotes the leftmost ``q_(n-1)`` character, so the
    corresponding Qiskit circuit qubit is ``nq - 1 - index``.
    """

    label = str(label_exyz).strip().lower()
    nq = int(getattr(circuit, "num_qubits"))
    if len(label) != nq:
        raise ValueError(f"Pauli label length mismatch: got {len(label)}, expected {nq}")
    active: list[tuple[int, str]] = []
    for idx, ch in enumerate(label):
        if ch == "e":
            continue
        if ch not in {"x", "y", "z"}:
            raise ValueError(f"unsupported Pauli symbol {ch!r} in {label_exyz!r}")
        active.append((int(nq - 1 - idx), ch))
    if not active:
        return
    active.sort(key=lambda item: item[0])
    for qubit, ch in active:
        if ch == "x":
            circuit.h(qubit)
        elif ch == "y":
            circuit.sdg(qubit)
            circuit.h(qubit)
    active_qubits = [qubit for qubit, _ch in active]
    if len(active_qubits) == 1:
        circuit.rz(angle, active_qubits[0])
    else:
        for control, target in zip(active_qubits[:-1], active_qubits[1:]):
            circuit.cx(control, target)
        circuit.rz(angle, active_qubits[-1])
        for control, target in reversed(list(zip(active_qubits[:-1], active_qubits[1:]))):
            circuit.cx(control, target)
    for qubit, ch in reversed(active):
        if ch == "x":
            circuit.h(qubit)
        elif ch == "y":
            circuit.h(qubit)
            circuit.s(qubit)


def build_runtime_layout_circuit(
    layout: Any,
    theta_runtime: np.ndarray | Sequence[float],
    num_qubits: int,
    *,
    reference_state: np.ndarray | Sequence[complex] | None = None,
    quantum_circuit_cls: Any | None = None,
) -> Any:
    """Convert a repo runtime ansatz layout into a benchmark-local Qiskit circuit."""

    QuantumCircuit = quantum_circuit_cls
    if QuantumCircuit is None:
        QuantumCircuit = import_qiskit_pauli_components().QuantumCircuit
    circuit = QuantumCircuit(int(num_qubits))
    if reference_state is not None:
        append_reference_state(circuit, reference_state)
    theta_arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
    expected = int(getattr(layout, "runtime_parameter_count"))
    if int(theta_arr.size) != expected:
        raise ValueError(f"theta_runtime length mismatch: got {theta_arr.size}, expected {expected}")
    for block in getattr(layout, "blocks", ()):  # AnsatzParameterLayout-compatible
        runtime_count = int(getattr(block, "runtime_count", 0))
        if runtime_count <= 0:
            continue
        start = int(getattr(block, "runtime_start"))
        stop = int(getattr(block, "runtime_stop"))
        block_theta = theta_arr[start:stop]
        for local_idx, spec in enumerate(getattr(block, "terms", ())):
            angle = 2.0 * float(block_theta[int(local_idx)]) * float(getattr(spec, "coeff_real"))
            append_pauli_rotation_exyz(circuit, label_exyz=str(getattr(spec, "pauli_exyz")), angle=angle)
    return circuit


def build_parameterized_runtime_layout_circuit(
    layout: Any,
    *,
    num_qubits: int,
    reference_state: np.ndarray | Sequence[complex] | None = None,
    parameter_prefix: str = "theta",
    quantum_circuit_cls: Any | None = None,
    parameter_vector_cls: Any | None = None,
) -> ParameterizedRuntimeCircuit:
    """Convert a repo runtime ansatz layout into a parameterized Qiskit circuit."""

    QuantumCircuit = quantum_circuit_cls
    ParameterVector = parameter_vector_cls
    if QuantumCircuit is None or ParameterVector is None:
        try:
            from qiskit.circuit import ParameterVector as _ParameterVector
        except Exception as exc:  # pragma: no cover - depends on optional install state
            raise QiskitPauliUnavailable(_QISKIT_PAULI_DEPENDENCY_MESSAGE) from exc
        if QuantumCircuit is None:
            QuantumCircuit = import_qiskit_pauli_components().QuantumCircuit
        if ParameterVector is None:
            ParameterVector = _ParameterVector
    circuit = QuantumCircuit(int(num_qubits))
    if reference_state is not None:
        append_reference_state(circuit, reference_state)
    expected = int(getattr(layout, "runtime_parameter_count"))
    params = ParameterVector(str(parameter_prefix), expected)
    for block in getattr(layout, "blocks", ()):  # AnsatzParameterLayout-compatible
        runtime_count = int(getattr(block, "runtime_count", 0))
        if runtime_count <= 0:
            continue
        start = int(getattr(block, "runtime_start"))
        for local_idx, spec in enumerate(getattr(block, "terms", ())):
            runtime_idx = int(start + int(local_idx))
            if runtime_idx >= expected:
                raise ValueError(
                    f"layout term runtime index {runtime_idx} exceeds parameter count {expected}"
                )
            angle = 2.0 * params[runtime_idx] * float(getattr(spec, "coeff_real"))
            append_pauli_rotation_exyz(circuit, label_exyz=str(getattr(spec, "pauli_exyz")), angle=angle)
    return ParameterizedRuntimeCircuit(
        circuit=circuit,
        parameters=tuple(params),
        parameter_order="repo_runtime_parameter_order",
    )


def pauli_poly_to_sparse_pauli_op(
    poly: Any,
    *,
    sparse_pauli_op_cls: Any | None = None,
    tol: float = 1e-12,
) -> Any:
    """Convert a repo PauliPolynomial-like object into a Qiskit SparsePauliOp."""

    SparsePauliOp = sparse_pauli_op_cls
    if SparsePauliOp is None:
        SparsePauliOp = import_qiskit_pauli_components().SparsePauliOp
    terms = list(poly.return_polynomial())
    if not terms:
        nq = int(getattr(poly, "get_nq", lambda: 1)())
        return SparsePauliOp.from_list([("I" * max(1, nq), 0.0 + 0.0j)])

    nq = int(terms[0].nqubit())
    coeff_map: dict[str, complex] = {}
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        label = to_ixyz_label(str(term.pw2strng()))
        if len(label) != nq:
            raise ValueError(f"Pauli label length mismatch: got {len(label)}, expected {nq}")
        coeff_map[label] = coeff_map.get(label, 0.0 + 0.0j) + coeff
    cleaned = [(label, coeff) for label, coeff in coeff_map.items() if abs(coeff) > float(tol)]
    if not cleaned:
        cleaned = [("I" * nq, 0.0 + 0.0j)]
    return SparsePauliOp.from_list(cleaned).simplify(atol=float(tol))


def sparse_pauli_label_coefficients(op: Any) -> dict[str, complex]:
    """Return a label->coefficient mapping from a SparsePauliOp-like object."""

    labels = list(op.paulis.to_labels())
    coeffs = np.asarray(op.coeffs, dtype=complex).reshape(-1)
    return {str(label): complex(coeffs[idx]) for idx, label in enumerate(labels)}


def circuit_stats(circuit: Any) -> dict[str, Any]:
    """Return small Qiskit circuit stats without requiring a backend."""

    try:
        op_counts = {str(k): int(v) for k, v in dict(circuit.count_ops()).items()}
    except Exception:
        op_counts = {}
    twoq = 0
    try:
        for item in circuit.data:
            operation = getattr(item, "operation", None)
            if operation is None and isinstance(item, (tuple, list)) and item:
                operation = item[0]
            if int(getattr(operation, "num_qubits", 0)) == 2:
                twoq += 1
    except Exception:
        twoq = int(op_counts.get("cx", 0) + op_counts.get("cz", 0))
    try:
        depth = int(circuit.depth())
    except Exception:
        depth = None
    try:
        size = int(circuit.size())
    except Exception:
        size = None
    return {"depth": depth, "size": size, "count_2q": int(twoq), "op_counts": op_counts}


__all__ = [
    "QiskitPauliComponents",
    "QiskitPauliUnavailable",
    "ParameterizedRuntimeCircuit",
    "append_pauli_rotation_exyz",
    "append_reference_state",
    "build_parameterized_runtime_layout_circuit",
    "build_reference_state_circuit",
    "build_runtime_layout_circuit",
    "circuit_stats",
    "has_qiskit_pauli_support",
    "import_qiskit_pauli_components",
    "pauli_poly_to_sparse_pauli_op",
    "sparse_pauli_label_coefficients",
    "to_ixyz_label",
]
