#!/usr/bin/env python3
"""Compute circuit cost metrics for a completed ADAPT-VQE scaffold.

Loads an ADAPT output JSON, reconstructs the ansatz circuit, transpiles to
an IBM fake backend, and reports gate counts, depth, and measurement cost.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2

from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


def _load_adapt_result(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def _rebuild_pool_term(label: str, h_poly, all_pool: list[AnsatzTerm]) -> AnsatzTerm | None:
    """Find pool term by label."""
    for term in all_pool:
        if term.label == label:
            return term
    return None


def _pauli_poly_to_sparse_pauli_op(poly, nq: int, tol: float = 1e-12) -> SparsePauliOp:
    """Convert PauliPolynomial to Qiskit SparsePauliOp."""
    labels = []
    coeffs = []
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= tol:
            continue
        ps = str(term.pw2strng())
        # Convert exyz -> IXYZ Qiskit convention (reverse qubit order)
        qiskit_label = ps.upper().replace("E", "I")[::-1]
        labels.append(qiskit_label)
        coeffs.append(coeff)
    if not labels:
        labels = ["I" * nq]
        coeffs = [0.0]
    return SparsePauliOp(labels, coeffs)


def _build_ansatz_circuit(
    operators: list[AnsatzTerm],
    theta: np.ndarray,
    nq: int,
    ref_state: np.ndarray | None = None,
) -> QuantumCircuit:
    """Build ansatz circuit: prod_k exp(-i * theta_k * G_k) |ref>."""
    qc = QuantumCircuit(nq)

    # Reference state: HF bitstring
    if ref_state is not None:
        # Find the computational basis state with largest amplitude
        idx = int(np.argmax(np.abs(ref_state)))
        bitstring = format(idx, f"0{nq}b")
        for q in range(nq):
            if bitstring[nq - 1 - q] == "1":
                qc.x(q)

    synthesis = SuzukiTrotter(order=2, reps=1)
    for k, (op, angle) in enumerate(zip(operators, theta)):
        qop = _pauli_poly_to_sparse_pauli_op(op.polynomial, nq)
        coeffs = np.asarray(qop.coeffs, dtype=complex)
        if coeffs.size == 0 or np.max(np.abs(coeffs)) <= 1e-12:
            continue
        gate = PauliEvolutionGate(qop, time=float(angle), synthesis=synthesis)
        qc.append(gate, list(range(nq)))
    return qc


def _hamiltonian_measurement_groups(h_poly, nq: int) -> list[SparsePauliOp]:
    """Group Hamiltonian terms by QWC (qubit-wise commuting) sets."""
    terms = h_poly.return_polynomial()
    pauli_list = []
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= 1e-12:
            continue
        ps = str(term.pw2strng())
        qiskit_label = ps.upper().replace("E", "I")[::-1]
        if all(c == "I" for c in qiskit_label):
            continue  # identity term
        pauli_list.append((qiskit_label, coeff))

    if not pauli_list:
        return []

    # QWC grouping: two Paulis are QWC if on every qubit they either
    # match or at least one is I
    groups: list[list[tuple[str, complex]]] = []
    for label, coeff in pauli_list:
        placed = False
        for group in groups:
            compatible = True
            for glabel, _ in group:
                for a, b in zip(label, glabel):
                    if a != "I" and b != "I" and a != b:
                        compatible = False
                        break
                if not compatible:
                    break
            if compatible:
                group.append((label, coeff))
                placed = True
                break
        if not placed:
            groups.append([(label, coeff)])

    return [
        SparsePauliOp([lbl for lbl, _ in g], [c for _, c in g])
        for g in groups
    ]


def _estimate_shots_per_group(
    observable: SparsePauliOp,
    target_precision: float = 1.6e-3,  # mHa precision
) -> int:
    """Estimate shots needed per QWC group for given precision.

    For a group of Pauli operators P_i with coefficients c_i,
    Var(E) = sum_i |c_i|^2 * Var(<P_i>).
    Since Var(<P>) <= 1 for any Pauli, Var(E) <= sum_i |c_i|^2.
    To achieve precision eps: shots >= Var(E) / eps^2.
    """
    coeffs = np.asarray(observable.coeffs, dtype=complex)
    variance_bound = float(np.sum(np.abs(coeffs) ** 2))
    shots = int(np.ceil(variance_bound / (target_precision ** 2)))
    return max(shots, 1)


def main(json_path: str, backend_name: str = "FakeGuadalupeV2", opt_level: int = 1) -> None:
    path = Path(json_path)
    data = _load_adapt_result(path)
    adapt_vqe = data.get("adapt_vqe", {})
    settings = data.get("settings", {})

    operator_labels = adapt_vqe.get("operators", [])
    optimal_point = np.array(adapt_vqe.get("optimal_point", []), dtype=float)
    energy = adapt_vqe.get("energy")
    exact_gs = adapt_vqe.get("exact_gs_energy")
    depth = len(operator_labels)

    L = int(settings.get("L", 3))
    n_ph_max = int(settings.get("n_ph_max", 1))
    boson_encoding = str(settings.get("boson_encoding", "binary"))

    ferm_nq = 2 * L
    boson_bits_per_site = int(np.ceil(np.log2(n_ph_max + 1)))
    boson_nq = L * boson_bits_per_site
    nq = ferm_nq + boson_nq

    print(f"{'='*60}")
    print(f"ADAPT-VQE Circuit Cost Analysis")
    print(f"{'='*60}")
    print(f"Source: {path.name}")
    print(f"L={L}, n_ph_max={n_ph_max}, encoding={boson_encoding}")
    print(f"Total qubits: {nq} (fermion={ferm_nq}, boson={boson_nq})")
    print(f"Scaffold depth: {depth}")
    print(f"Energy: {energy}")
    print(f"Exact GS: {exact_gs}")
    if energy is not None and exact_gs is not None:
        print(f"abs_delta_e: {abs(energy - exact_gs):.6e}")
    print()

    # Rebuild Hamiltonian
    from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian
    h_poly = build_hubbard_holstein_hamiltonian(
        dims=L,
        J=float(settings.get("t", 1.0)),
        U=float(settings.get("u", 4.0)),
        omega0=float(settings.get("omega0", 1.0)),
        g=float(settings.get("g_ep", 0.5)),
        n_ph_max=n_ph_max,
        boson_encoding=boson_encoding,
        v_t=None,
        v0=float(settings.get("dv", 0.0)),
        t_eval=None,
        repr_mode="JW",
        indexing=str(settings.get("ordering", "blocked")),
        pbc=(str(settings.get("boundary", "open")) == "periodic"),
        include_zero_point=True,
    )

    # Rebuild pool to get operator polynomials
    from pipelines.hardcoded.adapt_pipeline import (
        _build_hh_pareto_lean_pool,
        _build_hh_full_meta_pool,
    )
    from src.quantum.vqe_latex_python_pairs import half_filled_num_particles

    num_particles = tuple(half_filled_num_particles(L))
    pool_key = str(settings.get("adapt_pool", "pareto_lean"))

    builder = _build_hh_pareto_lean_pool if pool_key == "pareto_lean" else _build_hh_full_meta_pool
    pool, pool_meta = builder(
        h_poly=h_poly,
        num_sites=L,
        t=float(settings.get("t", 1.0)),
        u=float(settings.get("u", 4.0)),
        omega0=float(settings.get("omega0", 1.0)),
        g_ep=float(settings.get("g_ep", 0.5)),
        dv=float(settings.get("dv", 0.0)),
        n_ph_max=n_ph_max,
        boson_encoding=boson_encoding,
        ordering=str(settings.get("ordering", "blocked")),
        boundary=str(settings.get("boundary", "open")),
        paop_r=int(settings.get("paop_r", 1)),
        paop_split_paulis=bool(settings.get("paop_split_paulis", False)),
        paop_prune_eps=float(settings.get("paop_prune_eps", 0.0)),
        paop_normalization=str(settings.get("paop_normalization", "none")),
        num_particles=num_particles,
    )

    # Map labels to pool terms — handle child_set labels by finding parent
    pool_by_label = {t.label: t for t in pool}
    scaffold_ops: list[AnsatzTerm] = []
    missing = []
    for label in operator_labels:
        if label in pool_by_label:
            scaffold_ops.append(pool_by_label[label])
        else:
            # Child set or repeat — try parent label
            parent = label.split("::child_set")[0] if "::child_set" in label else label
            if parent in pool_by_label:
                scaffold_ops.append(AnsatzTerm(label=label, polynomial=pool_by_label[parent].polynomial))
            else:
                missing.append(label)

    if missing:
        print(f"WARNING: {len(missing)} operators not found in pool (child sets from runtime split):")
        for m in missing:
            print(f"  - {m}")
        print("Using parent polynomial as approximation for circuit cost.")
        print()

    # Build abstract circuit
    print(f"Building ansatz circuit ({depth} layers)...")
    qc = _build_ansatz_circuit(scaffold_ops, optimal_point, nq)
    print(f"  Abstract gates: {qc.size()}")
    print(f"  Abstract depth: {qc.depth()}")
    print()

    # Decompose evolution gates
    print("Decomposing PauliEvolutionGates...")
    qc_decomposed = qc.decompose(reps=3)
    ops_count = qc_decomposed.count_ops()
    print(f"  Decomposed gate counts: {dict(ops_count)}")
    print(f"  Decomposed depth: {qc_decomposed.depth()}")
    print()

    # Transpile to fake backend
    from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
    backend = FakeGuadalupeV2()
    print(f"Transpiling to {backend_name} ({backend.num_qubits} qubits), optimization_level={opt_level}...")

    pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend)
    qc_transpiled = pm.run(qc)

    ops_transpiled = qc_transpiled.count_ops()
    cx_count = ops_transpiled.get("cx", 0) + ops_transpiled.get("ecr", 0)
    rz_count = ops_transpiled.get("rz", 0)
    sx_count = ops_transpiled.get("sx", 0)
    x_count = ops_transpiled.get("x", 0)
    sq_total = rz_count + sx_count + x_count
    total_gates = sum(ops_transpiled.values())

    print(f"  Transpiled gate counts: {dict(ops_transpiled)}")
    print(f"  Transpiled depth: {qc_transpiled.depth()}")
    print(f"  Two-qubit gates (CX/ECR): {cx_count}")
    print(f"  Single-qubit gates: {sq_total} (rz={rz_count}, sx={sx_count}, x={x_count})")
    print(f"  Total gates: {total_gates}")
    print(f"  Qubits used: {qc_transpiled.num_qubits}")
    print()

    # Also try optimization_level=3 for comparison
    if opt_level != 3:
        print(f"Transpiling at optimization_level=3 for comparison...")
        pm3 = generate_preset_pass_manager(optimization_level=3, backend=backend)
        qc_t3 = pm3.run(qc)
        ops_t3 = qc_t3.count_ops()
        cx3 = ops_t3.get("cx", 0) + ops_t3.get("ecr", 0)
        print(f"  opt_level=3: depth={qc_t3.depth()}, CX/ECR={cx3}, total={sum(ops_t3.values())}")
        print()

    # Measurement cost
    print(f"{'='*60}")
    print(f"Measurement Cost Analysis")
    print(f"{'='*60}")

    h_terms = h_poly.return_polynomial()
    h_pauli_count = sum(1 for t in h_terms if abs(complex(t.p_coeff)) > 1e-12
                        and not all(c == "e" for c in str(t.pw2strng())))
    print(f"Hamiltonian Pauli terms (non-identity): {h_pauli_count}")

    groups = _hamiltonian_measurement_groups(h_poly, nq)
    print(f"QWC measurement groups: {len(groups)}")
    print()

    precisions = [1.6e-3, 1e-3, 1e-4]
    print(f"Shot estimates per precision target:")
    print(f"  {'Precision':<15} {'Shots/group':<15} {'Total shots':<15} {'Groups':<10}")
    for eps in precisions:
        total_shots = 0
        for g in groups:
            total_shots += _estimate_shots_per_group(g, eps)
        avg_per_group = total_shots // max(len(groups), 1)
        print(f"  {eps:<15.1e} {avg_per_group:<15,} {total_shots:<15,} {len(groups)}")

    print()
    print(f"Per-group detail (eps=1.6e-3):")
    for i, g in enumerate(groups):
        n_terms = len(g)
        shots = _estimate_shots_per_group(g, 1.6e-3)
        paulis = [str(g.paulis[j]) for j in range(min(3, n_terms))]
        suffix = "..." if n_terms > 3 else ""
        print(f"  Group {i+1:2d}: {n_terms:2d} terms, {shots:>8,} shots  [{', '.join(paulis)}{suffix}]")

    # Summary block
    print()
    print(f"{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Qubits:              {nq}")
    print(f"Scaffold depth:      {depth}")
    print(f"CX/ECR gates:        {cx_count}  (opt_level={opt_level})")
    print(f"Circuit depth:       {qc_transpiled.depth()}  (opt_level={opt_level})")
    print(f"Total gates:         {total_gates}")
    print(f"Measurement groups:  {len(groups)}")
    print(f"Total shots (mHa):   {sum(_estimate_shots_per_group(g, 1.6e-3) for g in groups):,}")
    print(f"Total shots (0.1mHa):{sum(_estimate_shots_per_group(g, 1e-4) for g in groups):,}")
    print(f"Energy:              {energy}")
    print(f"abs_delta_e:         {abs(energy - exact_gs):.6e}" if energy and exact_gs else "")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="ADAPT-VQE circuit cost analysis")
    p.add_argument("json_path", type=str, help="Path to ADAPT output JSON")
    p.add_argument("--backend", type=str, default="FakeGuadalupeV2")
    p.add_argument("--opt-level", type=int, default=1)
    args = p.parse_args()
    main(args.json_path, args.backend, args.opt_level)
