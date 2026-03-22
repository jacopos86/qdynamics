#!/usr/bin/env python3
"""Shared Qiskit circuit construction for logical ADAPT scaffolds.

This module keeps the search manifold at the logical operator level while
providing a deterministic boundary conversion into Qiskit circuits for
transpilation-only executable-burden estimation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit

from pipelines.exact_bench.noise_oracle_runtime import _append_reference_state as _append_reference_state_runtime
from src.quantum.ansatz_parameterization import AnsatzParameterLayout, build_parameter_layout
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def append_reference_state(qc: QuantumCircuit, ref_state: np.ndarray | None) -> None:
    """Built-in math expression: |psi> = U(theta) |psi_ref>."""
    if ref_state is None:
        return
    _append_reference_state_runtime(qc, np.asarray(ref_state, dtype=complex).reshape(-1))


def append_pauli_rotation_exyz(qc: QuantumCircuit, *, label_exyz: str, angle: float) -> None:
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
        qc.rz(float(angle), active_qubits[0])
    else:
        for control, target in zip(active_qubits[:-1], active_qubits[1:]):
            qc.cx(control, target)
        qc.rz(float(angle), active_qubits[-1])
        for control, target in reversed(list(zip(active_qubits[:-1], active_qubits[1:]))):
            qc.cx(control, target)

    for qubit, ch in reversed(active):
        if ch == "x":
            qc.h(qubit)
        elif ch == "y":
            qc.h(qubit)
            qc.s(qubit)


def build_ansatz_circuit(
    layout: AnsatzParameterLayout,
    theta_runtime: np.ndarray,
    nq: int,
    ref_state: np.ndarray | None = None,
) -> QuantumCircuit:
    """Built-in math expression: U(theta) = Π_b Π_j exp(-i θ_bj c_bj P_bj)."""
    qc = QuantumCircuit(int(nq))
    append_reference_state(qc, ref_state)
    theta_arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
    if int(theta_arr.size) != int(layout.runtime_parameter_count):
        raise ValueError(
            f"theta_runtime length mismatch: got {theta_arr.size}, expected {layout.runtime_parameter_count}."
        )
    for block in layout.blocks:
        if int(block.runtime_count) <= 0:
            continue
        block_theta = theta_arr[block.runtime_start:block.runtime_stop]
        for local_idx, spec in enumerate(block.terms):
            angle = 2.0 * float(block_theta[local_idx]) * float(spec.coeff_real)
            append_pauli_rotation_exyz(qc, label_exyz=str(spec.pauli_exyz), angle=float(angle))
    return qc


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
    qc = build_ansatz_circuit(layout, theta_runtime, int(nq), ref_state=ref_state)
    return layout, qc
