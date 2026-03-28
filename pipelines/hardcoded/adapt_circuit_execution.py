#!/usr/bin/env python3
"""Shared Qiskit circuit construction for logical ADAPT scaffolds.

This module keeps the search manifold at the logical operator level while
providing a deterministic boundary conversion into Qiskit circuits for
transpilation-only executable-burden estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from pipelines.exact_bench.noise_oracle_runtime import _append_reference_state as _append_reference_state_runtime
from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    build_parameter_layout,
    serialize_layout,
)
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


@dataclass(frozen=True)
class ParameterizedAnsatzPlan:
    layout: AnsatzParameterLayout
    nq: int
    circuit: QuantumCircuit
    parameters: tuple[Any, ...]
    structure_digest: str
    reference_state_digest: str | None
    plan_digest: str



def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))



def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()



def _normalize_reference_state(ref_state: np.ndarray | None, *, nq: int) -> np.ndarray | None:
    if ref_state is None:
        return None
    arr = np.asarray(ref_state, dtype=complex).reshape(-1)
    expected_dim = 1 << int(nq)
    if int(arr.size) != int(expected_dim):
        raise ValueError(
            f"reference_state dimension {arr.size} does not match num_qubits={int(nq)}"
        )
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError("reference_state has zero norm")
    return np.asarray(arr / norm, dtype=complex)



def _reference_state_digest(ref_state: np.ndarray | None, *, nq: int) -> str | None:
    normalized = _normalize_reference_state(ref_state, nq=int(nq))
    if normalized is None:
        return None
    payload = {
        "schema_version": "parameterized_ansatz_ref_state_v1",
        "nq": int(nq),
        "amplitudes": [
            [float(np.real(val)), float(np.imag(val))]
            for val in np.asarray(normalized, dtype=complex).reshape(-1)
        ],
    }
    return _hash_payload(payload)



def _structure_digest(layout: AnsatzParameterLayout, *, nq: int) -> str:
    payload = {
        "schema_version": "parameterized_ansatz_plan_v1",
        "nq": int(nq),
        "layout": serialize_layout(layout),
        "rotation_convention": "exp(-i angle/2 P_exyz)",
        "angle_rule": "angle = 2 * theta_runtime[i] * coeff_real",
        "qubit_map": "exyz index i -> qiskit qubit nq-1-i",
    }
    return _hash_payload(payload)



def append_reference_state(qc: QuantumCircuit, ref_state: np.ndarray | None) -> None:
    """Built-in math expression: |psi> = U(theta) |psi_ref>."""
    if ref_state is None:
        return
    normalized = _normalize_reference_state(ref_state, nq=int(qc.num_qubits))
    if normalized is None:
        return
    _append_reference_state_runtime(qc, normalized)



def append_pauli_rotation_exyz(qc: QuantumCircuit, *, label_exyz: str, angle: Any) -> None:
    """Built-in math expression: exp(-i * angle/2 * P_exyz)."""
    label = str(label_exyz).strip().lower()
    nq = int(qc.num_qubits)
    if len(label) != nq:
        raise ValueError(f"Pauli label length mismatch: got {len(label)}, expected {nq}.")

    active: list[tuple[int, str]] = []
    for idx, ch in enumerate(label):
        if ch == "e":
            continue
        qubit = int(nq - 1 - idx)
        active.append((qubit, ch))
    if not active:
        return
    active.sort(key=lambda item: item[0])

    for qubit, ch in active:
        if ch == "x":
            qc.h(qubit)
        elif ch == "y":
            qc.sdg(qubit)
            qc.h(qubit)
        elif ch == "z":
            pass
        else:
            raise ValueError(f"Unsupported Pauli letter '{ch}' in {label_exyz!r}.")

    active_qubits = [q for q, _ in active]
    if len(active_qubits) == 1:
        qc.rz(angle, active_qubits[0])
    else:
        for control, target in zip(active_qubits[:-1], active_qubits[1:]):
            qc.cx(control, target)
        qc.rz(angle, active_qubits[-1])
        for control, target in reversed(list(zip(active_qubits[:-1], active_qubits[1:]))):
            qc.cx(control, target)

    for qubit, ch in reversed(active):
        if ch == "x":
            qc.h(qubit)
        elif ch == "y":
            qc.h(qubit)
            qc.s(qubit)



def build_parameterized_ansatz_plan(
    layout: AnsatzParameterLayout,
    *,
    nq: int,
    ref_state: np.ndarray | None = None,
) -> ParameterizedAnsatzPlan:
    """Built-in math expression: U(theta) = Π_b Π_j exp(-i θ_bj c_bj P_bj)."""
    nq_i = int(nq)
    qc = QuantumCircuit(nq_i)
    append_reference_state(qc, ref_state)
    theta_params = ParameterVector("theta", int(layout.runtime_parameter_count))
    for block in layout.blocks:
        if int(block.runtime_count) <= 0:
            continue
        for local_idx, spec in enumerate(block.terms):
            theta_param = theta_params[int(block.runtime_start) + int(local_idx)]
            angle = 2.0 * float(spec.coeff_real) * theta_param
            append_pauli_rotation_exyz(qc, label_exyz=str(spec.pauli_exyz), angle=angle)
    structure_digest = _structure_digest(layout, nq=nq_i)
    ref_digest = _reference_state_digest(ref_state, nq=nq_i)
    plan_digest = _hash_payload(
        {
            "schema_version": "parameterized_ansatz_plan_identity_v1",
            "structure_digest": str(structure_digest),
            "reference_state_digest": ref_digest,
        }
    )
    return ParameterizedAnsatzPlan(
        layout=layout,
        nq=nq_i,
        circuit=qc,
        parameters=tuple(theta_params),
        structure_digest=str(structure_digest),
        reference_state_digest=ref_digest,
        plan_digest=str(plan_digest),
    )



def bind_parameterized_ansatz_circuit(
    plan: ParameterizedAnsatzPlan,
    theta_runtime: np.ndarray | Sequence[float],
) -> QuantumCircuit:
    theta_arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
    if int(theta_arr.size) != int(plan.layout.runtime_parameter_count):
        raise ValueError(
            f"theta_runtime length mismatch: got {theta_arr.size}, expected {plan.layout.runtime_parameter_count}."
        )
    assignments = {
        param: float(theta_arr[idx])
        for idx, param in enumerate(tuple(plan.parameters))
    }
    return plan.circuit.assign_parameters(assignments, inplace=False)



def build_ansatz_circuit(
    layout: AnsatzParameterLayout,
    theta_runtime: np.ndarray,
    nq: int,
    ref_state: np.ndarray | None = None,
) -> QuantumCircuit:
    """Built-in math expression: U(theta) = Π_b Π_j exp(-i θ_bj c_bj P_bj)."""
    plan = build_parameterized_ansatz_plan(layout, nq=int(nq), ref_state=ref_state)
    return bind_parameterized_ansatz_circuit(plan, theta_runtime)



def build_structure_theta(layout: AnsatzParameterLayout, value: float = 1.0) -> np.ndarray:
    return np.full(int(layout.runtime_parameter_count), float(value), dtype=float)



def build_structural_ansatz_circuit(
    scaffold_ops: Sequence[AnsatzTerm],
    *,
    nq: int,
    ref_state: np.ndarray | None,
    structure_theta_value: float = 1.0,
) -> tuple[AnsatzParameterLayout, QuantumCircuit]:
    layout = build_parameter_layout(scaffold_ops, ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    theta_runtime = build_structure_theta(layout, value=float(structure_theta_value))
    plan = build_parameterized_ansatz_plan(layout, nq=int(nq), ref_state=ref_state)
    qc = bind_parameterized_ansatz_circuit(plan, theta_runtime)
    return layout, qc
