#!/usr/bin/env python3
"""Qiskit baseline end-to-end Hubbard pipeline.

Flow:
1) Build Hubbard Hamiltonian with Qiskit Nature + JW mapping.
2) Run Qiskit VQE.
3) Run Qiskit QPE (with robust fallback).
4) Run Suzuki-2 Trotter dynamics + exact dynamics (statevector math using Qiskit-derived terms).
5) Emit JSON + compact PDF artifact in schema aligned with hardcoded pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter
from qiskit_algorithms import PhaseEstimation
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import BoundaryCondition, LineLattice
from qiskit_nature.second_q.mappers import JordanWignerMapper

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.hartree_fock_reference_state import hartree_fock_statevector
from src.quantum.drives_time_potential import (
    build_gaussian_sinusoid_density_drive,
    reference_method_name,
)

EXACT_LABEL = "Exact_Qiskit"
EXACT_METHOD = "python_matrix_eigendecomposition"


def _ai_log(event: str, **fields: Any) -> None:
    payload = {
        "event": str(event),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


@dataclass(frozen=True)
class CompiledPauliAction:
    label_exyz: str
    perm: np.ndarray
    phase: np.ndarray


def _to_exyz(label_ixyz: str) -> str:
    return str(label_ixyz).lower().replace("i", "e")


def _to_ixyz(label_exyz: str) -> str:
    return str(label_exyz).replace("e", "I").upper()


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(psi))
    if nrm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    return psi / nrm


def _half_filled_particles(num_sites: int) -> tuple[int, int]:
    return ((num_sites + 1) // 2, num_sites // 2)


def _sector_basis_indices(
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
) -> np.ndarray:
    nq = 2 * int(num_sites)
    dim = 1 << nq
    n_up_want, n_dn_want = int(num_particles[0]), int(num_particles[1])
    norm_ordering = str(ordering).strip().lower()

    idx_all = np.arange(dim, dtype=np.int64)

    if norm_ordering == "blocked":
        up_mask = (1 << num_sites) - 1
        dn_mask = up_mask << num_sites
        n_up_arr = np.array([bin(int(i) & int(up_mask)).count("1") for i in idx_all], dtype=np.int32)
        n_dn_arr = np.array([bin(int(i) & int(dn_mask)).count("1") for i in idx_all], dtype=np.int32)
    else:
        even_mask = int(sum(1 << (2 * q) for q in range(num_sites)))
        odd_mask = int(sum(1 << (2 * q + 1) for q in range(num_sites)))
        n_up_arr = np.array([bin(int(i) & even_mask).count("1") for i in idx_all], dtype=np.int32)
        n_dn_arr = np.array([bin(int(i) & odd_mask).count("1") for i in idx_all], dtype=np.int32)

    sector_indices = np.where((n_up_arr == n_up_want) & (n_dn_arr == n_dn_want))[0]
    if sector_indices.size == 0:
        raise ValueError(
            f"No basis states found for sector (n_up={n_up_want}, n_dn={n_dn_want}) "
            f"with ordering='{ordering}', num_sites={num_sites}."
        )
    return sector_indices


def _ground_manifold_basis_sector_filtered(
    hmat: np.ndarray,
    *,
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
    energy_tol: float,
) -> tuple[float, np.ndarray]:
    """Return (ground_energy, embedded_ground_manifold_basis) in a fixed sector."""
    tol = float(energy_tol)
    if tol < 0.0:
        raise ValueError(f"fidelity_subspace_energy_tol must be >= 0, got {tol}.")

    sector_indices = _sector_basis_indices(num_sites, num_particles, ordering)
    h_sector = hmat[np.ix_(sector_indices, sector_indices)]
    evals_sector, evecs_sector = np.linalg.eigh(h_sector)
    evals_real = np.real(evals_sector)
    gs_energy = float(np.min(evals_real))
    mask = evals_real <= (gs_energy + tol)
    if not bool(np.any(mask)):
        mask[int(np.argmin(evals_real))] = True

    basis_sector = np.asarray(evecs_sector[:, mask], dtype=complex)
    basis_full = np.zeros((hmat.shape[0], basis_sector.shape[1]), dtype=complex)
    basis_full[sector_indices, :] = basis_sector
    basis_full, _ = np.linalg.qr(basis_full)
    if basis_full.shape[1] == 0:
        raise RuntimeError("Filtered ground manifold basis is empty.")
    return gs_energy, basis_full


def _orthonormalize_basis_columns(
    basis: np.ndarray,
    *,
    rank_tol: float = 1e-12,
) -> np.ndarray:
    if basis.ndim != 2 or basis.shape[1] == 0:
        raise ValueError("Basis matrix must have shape (dim, k) with k>=1.")
    qmat, rmat = np.linalg.qr(basis)
    diag = np.abs(np.diag(rmat))
    rank = int(np.sum(diag > float(rank_tol)))
    if rank <= 0:
        raise RuntimeError("Ground-manifold basis lost rank during orthonormalization.")
    return qmat[:, :rank]


def _projector_fidelity_from_basis(
    basis_orthonormal: np.ndarray,
    psi: np.ndarray,
) -> float:
    amps = np.conjugate(basis_orthonormal).T @ psi
    raw = np.vdot(amps, amps)
    val = float(np.real(raw))
    if val < 0.0 and val > -1e-12:
        val = 0.0
    if val > 1.0 and val < 1.0 + 1e-12:
        val = 1.0
    return float(min(1.0, max(0.0, val)))


def _interleaved_to_blocked_permutation(n_sites: int) -> list[int]:
    return [idx for site in range(n_sites) for idx in (site, n_sites + site)]


def _spin_orbital_index_sets(num_sites: int, ordering: str) -> tuple[list[int], list[int]]:
    normalized_ordering = ordering.strip().lower()
    if normalized_ordering == "blocked":
        return list(range(num_sites)), list(range(num_sites, 2 * num_sites))
    if normalized_ordering == "interleaved":
        return list(range(0, 2 * num_sites, 2)), list(range(1, 2 * num_sites, 2))
    raise ValueError(f"Unsupported ordering '{ordering}'.")


def _number_operator_qop(num_qubits: int, indices: list[int]) -> SparsePauliOp:
    coeffs: dict[str, complex] = {}
    id_label = "I" * num_qubits
    coeffs[id_label] = coeffs.get(id_label, 0.0 + 0.0j) + complex(0.5 * len(indices))
    for q in indices:
        chars = ["I"] * num_qubits
        chars[num_qubits - 1 - q] = "Z"
        lbl = "".join(chars)
        coeffs[lbl] = coeffs.get(lbl, 0.0 + 0.0j) + complex(-0.5)
    return SparsePauliOp.from_list([(k, v) for k, v in coeffs.items()]).simplify(atol=1e-12)


def _filtered_exact_energy(
    qop: SparsePauliOp,
    *,
    num_sites: int,
    ordering: str,
    num_particles: tuple[int, int],
) -> float:
    alpha_idx, beta_idx = _spin_orbital_index_sets(int(num_sites), ordering)
    n_alpha_op = _number_operator_qop(qop.num_qubits, alpha_idx)
    n_beta_op = _number_operator_qop(qop.num_qubits, beta_idx)

    def _filter(_state, _energy, aux_values):
        n_alpha = float(np.real(aux_values["N_alpha"][0]))
        n_beta = float(np.real(aux_values["N_beta"][0]))
        return np.isclose(n_alpha, num_particles[0]) and np.isclose(n_beta, num_particles[1])

    solver = NumPyMinimumEigensolver(filter_criterion=_filter)
    res = solver.compute_minimum_eigenvalue(
        qop,
        aux_operators={"N_alpha": n_alpha_op, "N_beta": n_beta_op},
    )
    return float(np.real(res.eigenvalue))


def _uniform_potential_qubit_op(num_sites: int, dv: float) -> SparsePauliOp:
    nq = 2 * int(num_sites)
    coeffs: dict[str, complex] = {}

    id_label = "I" * nq
    coeffs[id_label] = coeffs.get(id_label, 0.0 + 0.0j) + complex(-0.5 * dv * nq)

    for p in range(nq):
        chars = ["I"] * nq
        chars[nq - 1 - p] = "Z"
        z_label = "".join(chars)
        coeffs[z_label] = coeffs.get(z_label, 0.0 + 0.0j) + complex(0.5 * dv)

    return SparsePauliOp.from_list([(lbl, coeff) for lbl, coeff in coeffs.items()]).simplify(atol=1e-12)


def _build_qiskit_qubit_hamiltonian(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    ordering: str,
) -> tuple[SparsePauliOp, Any]:
    boundary_enum = BoundaryCondition.PERIODIC if boundary.strip().lower() == "periodic" else BoundaryCondition.OPEN
    lattice = LineLattice(
        num_nodes=int(num_sites),
        edge_parameter=-float(t),
        onsite_parameter=0.0,
        boundary_condition=boundary_enum,
    )
    ferm_op = FermiHubbardModel(lattice=lattice, onsite_interaction=float(u)).second_q_op()

    if ordering.strip().lower() == "blocked":
        ferm_op = ferm_op.permute_indices(_interleaved_to_blocked_permutation(int(num_sites)))

    mapper = JordanWignerMapper()
    qop = mapper.map(ferm_op).simplify(atol=1e-12)

    if abs(float(dv)) > 1e-15:
        qop = (qop + _uniform_potential_qubit_op(int(num_sites), float(dv))).simplify(atol=1e-12)

    return qop, mapper


def _qiskit_terms_exyz(qop: SparsePauliOp, tol: float = 1e-12) -> tuple[list[str], dict[str, complex]]:
    order: list[str] = []
    coeff_map: dict[str, complex] = {}
    for label_ixyz, coeff in qop.to_list():
        coeff_c = complex(coeff)
        if abs(coeff_c) <= tol:
            continue
        lbl = _to_exyz(str(label_ixyz))
        if lbl not in coeff_map:
            order.append(lbl)
            coeff_map[lbl] = 0.0 + 0.0j
        coeff_map[lbl] += coeff_c

    cleaned_order = [lbl for lbl in order if abs(coeff_map[lbl]) > tol]
    cleaned_map = {lbl: coeff_map[lbl] for lbl in cleaned_order}
    return cleaned_order, cleaned_map


def _ordered_qop_from_exyz(
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    *,
    tol: float = 1e-12,
) -> SparsePauliOp:
    terms: list[tuple[str, complex]] = []
    for lbl in ordered_labels_exyz:
        coeff = complex(coeff_map_exyz[lbl])
        if abs(coeff) <= tol:
            continue
        terms.append((_to_ixyz(lbl), coeff))
    if not terms:
        nq = len(ordered_labels_exyz[0]) if ordered_labels_exyz else 1
        terms = [("I" * nq, 0.0 + 0.0j)]
    return SparsePauliOp.from_list(terms)


def _compile_pauli_action(label_exyz: str, nq: int) -> CompiledPauliAction:
    dim = 1 << nq
    idx = np.arange(dim, dtype=np.int64)
    perm = idx.copy()
    phase = np.ones(dim, dtype=complex)

    for q in range(nq):
        op = label_exyz[nq - 1 - q]
        bits = ((idx >> q) & 1).astype(np.int8)
        sign = (1 - 2 * bits).astype(np.int8)

        if op == "e":
            continue
        if op == "x":
            perm ^= (1 << q)
            continue
        if op == "y":
            perm ^= (1 << q)
            phase *= 1j * sign
            continue
        if op == "z":
            phase *= sign
            continue
        raise ValueError(f"Unsupported Pauli symbol '{op}' in '{label_exyz}'.")

    return CompiledPauliAction(label_exyz=label_exyz, perm=perm, phase=phase)


def _apply_compiled_pauli(psi: np.ndarray, action: CompiledPauliAction) -> np.ndarray:
    out = np.empty_like(psi)
    out[action.perm] = action.phase * psi
    return out


def _apply_exp_term(
    psi: np.ndarray,
    action: CompiledPauliAction,
    coeff: complex,
    alpha: float,
    tol: float = 1e-12,
) -> np.ndarray:
    if abs(coeff.imag) > tol:
        raise ValueError(f"Imaginary coefficient encountered for {action.label_exyz}: {coeff}")
    theta = float(alpha) * float(coeff.real)
    ppsi = _apply_compiled_pauli(psi, action)
    return math.cos(theta) * psi - 1j * math.sin(theta) * ppsi


def _evolve_trotter_suzuki2_absolute(
    psi0: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    compiled_actions: dict[str, CompiledPauliAction],
    time_value: float,
    trotter_steps: int,
    *,
    drive_coeff_provider_exyz: Any | None = None,
    t0: float = 0.0,
    time_sampling: str = "midpoint",
    coeff_tol: float = 1e-12,
) -> np.ndarray:
    """Suzuki-Trotter order-2 evolution, with optional time-dependent drive."""
    # --- time-independent fast path (bit-for-bit identical to original) ---
    if drive_coeff_provider_exyz is None:
        psi = np.array(psi0, copy=True)
        if abs(time_value) <= 1e-15:
            return psi
        dt = float(time_value) / float(trotter_steps)
        half = 0.5 * dt
        for _ in range(trotter_steps):
            for label in ordered_labels_exyz:
                psi = _apply_exp_term(psi, compiled_actions[label], coeff_map_exyz[label], half)
            for label in reversed(ordered_labels_exyz):
                psi = _apply_exp_term(psi, compiled_actions[label], coeff_map_exyz[label], half)
        return _normalize_state(psi)

    # --- time-dependent path ---
    psi = np.array(psi0, copy=True)
    if abs(time_value) <= 1e-15:
        return psi
    dt = float(time_value) / float(trotter_steps)
    half = 0.5 * dt

    sampling = str(time_sampling).strip().lower()
    if sampling not in {"midpoint", "left", "right"}:
        raise ValueError("time_sampling must be one of {'midpoint','left','right'}")

    t0_f = float(t0)
    tol = float(coeff_tol)

    for k in range(int(trotter_steps)):
        if sampling == "midpoint":
            t_sample = t0_f + (float(k) + 0.5) * dt
        elif sampling == "left":
            t_sample = t0_f + float(k) * dt
        else:  # right
            t_sample = t0_f + (float(k) + 1.0) * dt

        drive_map = dict(drive_coeff_provider_exyz(float(t_sample)))

        for label in ordered_labels_exyz:
            c_total = coeff_map_exyz.get(label, 0.0 + 0.0j) + complex(drive_map.get(label, 0.0))
            if abs(c_total) <= tol:
                continue
            psi = _apply_exp_term(psi, compiled_actions[label], c_total, half)
        for label in reversed(ordered_labels_exyz):
            c_total = coeff_map_exyz.get(label, 0.0 + 0.0j) + complex(drive_map.get(label, 0.0))
            if abs(c_total) <= tol:
                continue
            psi = _apply_exp_term(psi, compiled_actions[label], c_total, half)

    return _normalize_state(psi)


def _expectation_hamiltonian(psi: np.ndarray, hmat: np.ndarray) -> float:
    return float(np.real(np.vdot(psi, hmat @ psi)))


def _build_drive_matrix_at_time(
    drive_coeff_provider_exyz: Any,
    t_physical: float,
    nq: int,
) -> np.ndarray:
    """Build the full drive Hamiltonian matrix at a given physical time.

    Returns a ``(2**nq, 2**nq)`` complex matrix representing H_drive(t).
    If the drive map is empty at this time (e.g., A=0 or envelope → 0), the
    returned matrix is the zero matrix.
    """
    dim = 1 << nq
    drive_map = dict(drive_coeff_provider_exyz(float(t_physical)))
    if not drive_map:
        return np.zeros((dim, dim), dtype=complex)
    # Fast path: check if all labels are Z-type (diagonal).
    if all(_is_all_z_type(lbl) for lbl in drive_map if abs(drive_map[lbl]) > 1e-15):
        diag = _build_drive_diagonal(
            {lbl: complex(c) for lbl, c in drive_map.items() if abs(c) > 1e-15},
            dim,
            nq,
        )
        return np.diag(diag)
    # General path: build from Pauli matrices.
    hmat_drive = np.zeros((dim, dim), dtype=complex)
    for lbl, c in drive_map.items():
        if abs(c) <= 1e-15:
            continue
        hmat_drive += complex(c) * _pauli_matrix_exyz(lbl)
    return hmat_drive


def _spin_orbital_bit_index(site: int, spin: int, num_sites: int, ordering: str) -> int:
    ord_norm = str(ordering).strip().lower()
    if ord_norm == "blocked":
        return int(site) if int(spin) == 0 else int(num_sites) + int(site)
    if ord_norm == "interleaved":
        return (2 * int(site)) + int(spin)
    raise ValueError(f"Unsupported ordering {ordering!r}")


def _site_resolved_number_observables(
    psi: np.ndarray,
    num_sites: int,
    ordering: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    probs = np.abs(psi) ** 2
    n_up = np.zeros(int(num_sites), dtype=float)
    n_dn = np.zeros(int(num_sites), dtype=float)
    doublon_total = 0.0
    up_bits = [_spin_orbital_bit_index(site, 0, num_sites, ordering) for site in range(int(num_sites))]
    dn_bits = [_spin_orbital_bit_index(site, 1, num_sites, ordering) for site in range(int(num_sites))]

    for idx, prob in enumerate(probs):
        p = float(prob)
        if p <= 0.0:
            continue
        for site in range(int(num_sites)):
            up = int((idx >> up_bits[site]) & 1)
            dn = int((idx >> dn_bits[site]) & 1)
            n_up[site] += float(up) * p
            n_dn[site] += float(dn) * p
            doublon_total += float(up * dn) * p
    return n_up, n_dn, float(doublon_total)


def _staggered_order(n_total_site: np.ndarray) -> float:
    if n_total_site.size == 0:
        return float("nan")
    signs = np.array([1.0 if (i % 2 == 0) else -1.0 for i in range(int(n_total_site.size))], dtype=float)
    return float(np.sum(signs * n_total_site) / float(n_total_site.size))


def _state_to_amplitudes_qn_to_q0(psi: np.ndarray, cutoff: float = 1e-12) -> dict[str, dict[str, float]]:
    nq = int(round(math.log2(psi.size)))
    out: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(psi):
        if abs(amp) < cutoff:
            continue
        bit = format(idx, f"0{nq}b")
        out[bit] = {"re": float(np.real(amp)), "im": float(np.imag(amp))}
    return out


def _build_qiskit_optimizer(name: str, maxiter: int):
    opt = str(name).strip().upper()
    effective_maxiter = max(1, int(maxiter))
    if opt == "SLSQP":
        return SLSQP(maxiter=effective_maxiter), "SLSQP"
    if opt == "COBYLA":
        return COBYLA(maxiter=effective_maxiter), "COBYLA"
    if opt in {"L_BFGS_B", "L-BFGS-B", "LBFGSB"}:
        return L_BFGS_B(maxiter=effective_maxiter), "L_BFGS_B"
    raise ValueError(f"Unsupported --vqe-optimizer '{name}'.")


def _run_qiskit_vqe(
    *,
    num_sites: int,
    qop: SparsePauliOp,
    mapper: Any,
    hopping_t: float,
    onsite_u: float,
    dv: float,
    boundary: str,
    ordering: str,
    reps: int,
    restarts: int,
    seed: int,
    maxiter: int,
    optimizer_name: str,
) -> tuple[dict[str, Any], np.ndarray | None]:
    t0 = time.perf_counter()
    _ai_log(
        "qiskit_vqe_start",
        L=int(num_sites),
        reps=int(reps),
        restarts=int(restarts),
        maxiter=int(maxiter),
        seed=int(seed),
        optimizer=str(optimizer_name),
    )

    def _finish(payload: dict[str, Any], psi: np.ndarray | None) -> tuple[dict[str, Any], np.ndarray | None]:
        _ai_log(
            "qiskit_vqe_done",
            L=int(num_sites),
            method=str(payload.get("method", "")),
            success=bool(payload.get("success", False)),
            energy=payload.get("energy"),
            best_restart=payload.get("best_restart"),
            elapsed_sec=round(time.perf_counter() - t0, 6),
        )
        return payload, psi

    num_particles = _half_filled_particles(int(num_sites))
    estimator = None

    try:
        from qiskit.primitives import StatevectorEstimator

        estimator = StatevectorEstimator()
    except Exception:
        estimator = None

    if estimator is None:
        np_solver = NumPyMinimumEigensolver()
        eig = np_solver.compute_minimum_eigenvalue(qop)
        exact_filtered = None
        try:
            exact_filtered = _filtered_exact_energy(
                qop,
                num_sites=int(num_sites),
                ordering=str(ordering),
                num_particles=num_particles,
            )
        except Exception:
            exact_filtered = None
        return _finish(
            {
                "success": True,
                "method": "qiskit_numpy_minimum_eigensolver_fallback",
                "energy": float(np.real(eig.eigenvalue)),
                "exact_filtered_energy": exact_filtered,
                "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
            },
            None,
        )

    try:
        hf = HartreeFock(
            num_spatial_orbitals=int(num_sites),
            num_particles=tuple(num_particles),
            qubit_mapper=mapper,
        )
        ansatz = UCCSD(
            num_spatial_orbitals=int(num_sites),
            num_particles=tuple(num_particles),
            qubit_mapper=mapper,
            initial_state=hf,
            reps=int(reps),
        )
        effective_maxiter = max(1, int(maxiter))
        _, optimizer_label = _build_qiskit_optimizer(str(optimizer_name), effective_maxiter)
        rng = np.random.default_rng(int(seed))
        best_energy = float("inf")
        best_restart = -1
        best_point = np.zeros(ansatz.num_parameters, dtype=float)

        for restart in range(max(1, int(restarts))):
            initial_point = 0.3 * rng.normal(size=ansatz.num_parameters)
            optimizer, _ = _build_qiskit_optimizer(optimizer_label, effective_maxiter)
            vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer, initial_point=initial_point)
            res = vqe.compute_minimum_eigenvalue(qop)
            energy = float(np.real(res.eigenvalue))
            if energy < best_energy:
                best_energy = energy
                best_restart = restart
                best_point = np.asarray(res.optimal_point, dtype=float)
            _ai_log(
                "qiskit_vqe_restart_done",
                L=int(num_sites),
                restart=int(restart),
                total_restarts=max(1, int(restarts)),
                energy=float(energy),
                best_energy=float(best_energy),
            )

        psi_vqe = None
        try:
            bound_ansatz = ansatz.assign_parameters(best_point)
            psi_vqe = _normalize_state(np.asarray(Statevector(bound_ansatz).data, dtype=complex))
        except Exception:
            psi_vqe = None

        exact_filtered = None
        try:
            exact_filtered = _filtered_exact_energy(
                qop,
                num_sites=int(num_sites),
                ordering=str(ordering),
                num_particles=num_particles,
            )
        except Exception:
            exact_filtered = None

        return _finish(
            {
                "success": True,
                "method": "qiskit_vqe_uccsd",
                "energy": float(best_energy),
                "exact_filtered_energy": exact_filtered,
                "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
                "num_parameters": int(ansatz.num_parameters),
                "effective_maxiter": int(effective_maxiter),
                "optimizer": str(optimizer_label),
                "best_restart": int(best_restart),
                "optimal_point": [float(x) for x in best_point.tolist()],
            },
            psi_vqe,
        )
    except Exception as exc:
        np_solver = NumPyMinimumEigensolver()
        eig = np_solver.compute_minimum_eigenvalue(qop)
        exact_filtered = None
        try:
            exact_filtered = _filtered_exact_energy(
                qop,
                num_sites=int(num_sites),
                ordering=str(ordering),
                num_particles=num_particles,
            )
        except Exception:
            exact_filtered = None
        return _finish(
            {
                "success": True,
                "method": "qiskit_numpy_minimum_eigensolver_fallback",
                "energy": float(np.real(eig.eigenvalue)),
                "exact_filtered_energy": exact_filtered,
                "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
                "warning": str(exc),
            },
            None,
        )


def _run_qiskit_qpe(
    *,
    qop: SparsePauliOp,
    psi_init: np.ndarray,
    eval_qubits: int,
    shots: int,
    seed: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    _ai_log(
        "qiskit_qpe_start",
        eval_qubits=int(eval_qubits),
        shots=int(shots),
        seed=int(seed),
        num_qubits=int(qop.num_qubits),
    )

    def _finish(payload: dict[str, Any]) -> dict[str, Any]:
        _ai_log(
            "qiskit_qpe_done",
            success=bool(payload.get("success", False)),
            method=str(payload.get("method", "")),
            energy_estimate=payload.get("energy_estimate"),
            elapsed_sec=round(time.perf_counter() - t0, 6),
        )
        return payload

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.primitives import StatevectorSampler
    from qiskit.synthesis import SuzukiTrotter

    bound = float(sum(abs(float(np.real(coeff))) for _lbl, coeff in qop.to_list()))
    bound = max(bound, 1e-9)
    evo_time = float(np.pi / bound)

    # Large qubit counts can make canonical QPE prohibitively expensive in CI-like runs.
    # Use a deterministic Qiskit eigensolver fallback while keeping output schema aligned.
    if qop.num_qubits >= 8:
        np_solver = NumPyMinimumEigensolver()
        eig = np_solver.compute_minimum_eigenvalue(qop)
        return _finish({
            "success": True,
            "method": "qiskit_numpy_minimum_eigensolver_fastpath_large_n",
            "energy_estimate": float(np.real(eig.eigenvalue)),
            "phase": None,
            "bound": bound,
            "evolution_time": evo_time,
            "num_evaluation_qubits": int(eval_qubits),
            "shots": int(shots),
        })

    try:
        prep = QuantumCircuit(qop.num_qubits)
        prep.initialize(np.asarray(psi_init, dtype=complex), list(range(qop.num_qubits)))

        evo = PauliEvolutionGate(
            qop,
            time=evo_time,
            synthesis=SuzukiTrotter(order=2, reps=1, preserve_order=True),
        )
        unitary = QuantumCircuit(qop.num_qubits)
        unitary.append(evo, range(qop.num_qubits))

        try:
            sampler = StatevectorSampler(default_shots=int(shots), seed=int(seed))
        except TypeError:
            sampler = StatevectorSampler()

        qpe = PhaseEstimation(num_evaluation_qubits=int(eval_qubits), sampler=sampler)
        qpe_res = qpe.estimate(unitary=unitary, state_preparation=prep)
        phase = float(qpe_res.phase)
        phase_shift = phase if phase <= 0.5 else (phase - 1.0)
        energy = float(-2.0 * bound * phase_shift)

        return _finish({
            "success": True,
            "method": "qiskit_phase_estimation",
            "energy_estimate": energy,
            "phase": phase,
            "bound": bound,
            "evolution_time": evo_time,
            "num_evaluation_qubits": int(eval_qubits),
            "shots": int(shots),
        })
    except Exception as exc:
        np_solver = NumPyMinimumEigensolver()
        eig = np_solver.compute_minimum_eigenvalue(qop)
        return _finish({
            "success": True,
            "method": "qiskit_numpy_minimum_eigensolver_fallback",
            "energy_estimate": float(np.real(eig.eigenvalue)),
            "phase": None,
            "bound": bound,
            "evolution_time": evo_time,
            "num_evaluation_qubits": int(eval_qubits),
            "shots": int(shots),
            "warning": str(exc),
        })


def _reference_terms_for_case(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    ordering: str,
) -> dict[str, float] | None:
    norm_boundary = boundary.strip().lower()
    norm_ordering = ordering.strip().lower()
    if norm_boundary != "periodic" or norm_ordering != "blocked":
        return None
    if abs(float(t) - 1.0) > 1e-12 or abs(float(u) - 4.0) > 1e-12 or abs(float(dv)) > 1e-12:
        return None

    candidate_files = [
        REPO_ROOT / "src" / "quantum" / "exports" / "hubbard_jw_L2_L3_periodic_blocked.json",
        REPO_ROOT / "src" / "quantum" / "exports" / "hubbard_jw_L4_L5_periodic_blocked.json",
        ROOT / "Tests" / "hubbard_jw_L4_L5_periodic_blocked_qiskit.json",
    ]
    case_key = f"L={int(num_sites)}"

    for path in candidate_files:
        if not path.exists():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        cases = obj.get("cases", {}) if isinstance(obj, dict) else {}
        if case_key in cases:
            terms = cases[case_key].get("pauli_terms", {})
            if isinstance(terms, dict):
                return {str(k): float(v) for k, v in terms.items()}
    return None


def _reference_sanity(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    ordering: str,
    coeff_map_exyz: dict[str, complex],
) -> dict[str, Any]:
    ref_terms = _reference_terms_for_case(
        num_sites=num_sites,
        t=t,
        u=u,
        dv=dv,
        boundary=boundary,
        ordering=ordering,
    )
    if ref_terms is None:
        return {
            "checked": False,
            "reason": "no matching bundled reference for these settings",
        }

    cand = {_to_ixyz(lbl): float(np.real(coeff)) for lbl, coeff in coeff_map_exyz.items()}
    all_keys = sorted(set(ref_terms) | set(cand))
    max_abs_delta = 0.0
    missing_from_reference: list[str] = []
    missing_from_candidate: list[str] = []
    for key in all_keys:
        rv = float(ref_terms.get(key, 0.0))
        cv = float(cand.get(key, 0.0))
        max_abs_delta = max(max_abs_delta, abs(cv - rv))
        if key not in ref_terms:
            missing_from_reference.append(key)
        if key not in cand:
            missing_from_candidate.append(key)

    return {
        "checked": True,
        "max_abs_delta": float(max_abs_delta),
        "matches_within_1e-12": bool(max_abs_delta <= 1e-12),
        "missing_from_reference": missing_from_reference,
        "missing_from_candidate": missing_from_candidate,
    }


_PAULI_MATS = {
    "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


# ---------------------------------------------------------------------------
# Sparse / expm_multiply helpers
# ---------------------------------------------------------------------------

# Minimum Hilbert-space dimension at which expm_multiply is preferred over
# dense scipy.linalg.expm.  Below this threshold (dim < 64 ↔ L ≤ 2, nq ≤ 5)
# the Al-Mohy–Higham norm-estimation overhead outweighs the O(d³) dense cost.
# At dim = 64 (L = 3) the crossover begins; at dim ≥ 256 (L ≥ 4) sparse wins
# by an order of magnitude and the advantage grows exponentially with L.
_EXPM_SPARSE_MIN_DIM: int = 64


def _is_all_z_type(label: str) -> bool:
    """Return True if every character in an exyz label is ``'z'`` or ``'e'``.

    A Pauli string composed only of Z and I (identity, here 'e') operators is
    diagonal in the computational basis.  Its full tensor product is therefore
    also diagonal, meaning H_drive can be stored as a 1-D vector rather than a
    d × d matrix.

    The density drive (``TimeDependentOnsiteDensityDrive``) always returns
    labels of this form, so this check confirms the fast diagonal pathway is
    available.
    """
    return all(ch in ("z", "e") for ch in label)


def _build_drive_diagonal(
    drive_map: dict[str, complex],
    dim: int,
    nq: int,
) -> np.ndarray:
    """Build the diagonal of H_drive as a 1-D complex numpy array.

    Only valid when every label in *drive_map* is Z-type (caller must ensure
    this via :func:`_is_all_z_type`).

    For a Z-type label ``l``, the diagonal entry for computational-basis state
    ``|idx⟩`` is the product of eigenvalues:

    .. math::
        d[\\text{idx}] = \\prod_{q:\\, l[n_q-1-q]=\\text{'z'}} (-1)^{\\bigl(\\text{idx} >> q\\bigr) \\& 1}

    This is computed in O(|drive_map| × d) with fully vectorised numpy
    operations — no d × d matrix is ever allocated.

    Parameters
    ----------
    drive_map:
        ``{label: coeff}`` mapping, all labels Z-type.
    dim:
        Hilbert-space dimension (must equal ``1 << nq``).
    nq:
        Number of qubits.

    Returns
    -------
    np.ndarray of shape ``(dim,)`` and dtype ``complex``.
    """
    idx = np.arange(dim, dtype=np.int64)
    diag = np.zeros(dim, dtype=complex)
    for label, coeff in drive_map.items():
        if abs(coeff) <= 1e-15:
            continue
        eig = np.ones(dim, dtype=np.float64)
        for q in range(nq):
            if label[nq - 1 - q] == "z":
                eig *= 1.0 - 2.0 * ((idx >> q) & 1).astype(np.float64)
        diag += coeff * eig
    return diag


def _pauli_matrix_exyz(label: str) -> np.ndarray:
    mats = [_PAULI_MATS[ch] for ch in label]
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out


def _build_hamiltonian_matrix_from_exyz(coeff_map_exyz: dict[str, complex]) -> np.ndarray:
    if not coeff_map_exyz:
        return np.zeros((1, 1), dtype=complex)
    nq = len(next(iter(coeff_map_exyz)))
    dim = 1 << nq
    hmat = np.zeros((dim, dim), dtype=complex)
    for label, coeff in coeff_map_exyz.items():
        hmat += coeff * _pauli_matrix_exyz(label)
    return hmat


def _evolve_piecewise_exact(
    *,
    psi0: np.ndarray,
    hmat_static: np.ndarray,
    drive_coeff_provider_exyz: Any,
    time_value: float,
    trotter_steps: int,
    t0: float = 0.0,
    time_sampling: str = "midpoint",
) -> np.ndarray:
    """Piecewise-constant matrix-exponential reference propagator.

    Approximation order
    -------------------
    This function is **not** a true time-ordered exponential.  It is a
    piecewise-constant approximation: each sub-interval of width
    Δt = time_value / trotter_steps is replaced by the exact exponential
    of H evaluated at a single representative time t_k.

    The order depends on how t_k is chosen (``time_sampling``):

    * ``"midpoint"`` (default): t_k = t₀ + (k + ½)Δt.
      **Exponential midpoint / Magnus-2 integrator — second order O(Δt²).**
    * ``"left"``: t_k = t₀ + k Δt.  First order O(Δt).
    * ``"right"``: t_k = t₀ + (k+1)Δt.  First order O(Δt).

    See ``src/quantum/drives_time_potential.py`` module docstring for a full
    discussion.  The JSON metadata records ``reference_method`` and
    ``reference_steps_multiplier`` to document which approximation was used.

    Sparse / expm_multiply optimisation
    ------------------------------------
    For Hilbert-space dimension ``dim >= _EXPM_SPARSE_MIN_DIM`` (currently 64,
    i.e., L ≥ 3) this function uses ``scipy.sparse.linalg.expm_multiply``
    instead of ``scipy.linalg.expm``.

    * **H_static** is pre-converted to a ``scipy.sparse.csc_matrix`` *once*
      before the time-step loop.
    * **H_drive(t)** — which for the density drive is always a sum of Z-type
      (diagonal) Paulis — is stored as a 1-D diagonal vector and converted to
      a ``scipy.sparse.diags`` matrix per step.  No d × d dense allocation is
      needed.
    * ``expm_multiply`` uses the Al-Mohy–Higham algorithm: cost O(d · nnz · p)
      vs O(d³) for dense expm.

    Trade-offs
    ~~~~~~~~~~
    +-------------------+-----------------+-------------------------------------+
    | Method            | Cost / step     | When preferred                      |
    +===================+=================+=====================================+
    | Dense ``expm``    | O(d³)           | d < 64 (L ≤ 2); SciPy sparse absent |
    +-------------------+-----------------+-------------------------------------+
    | ``expm_multiply`` | O(d · nnz · p)  | d ≥ 64 (L ≥ 3); sparse available    |
    +-------------------+-----------------+-------------------------------------+

    The dense fallback is retained automatically whenever ``dim <
    _EXPM_SPARSE_MIN_DIM`` or ``scipy.sparse`` is unavailable.
    """
    psi = np.array(psi0, copy=True)
    if abs(time_value) <= 1e-15:
        return _normalize_state(psi)

    dt = float(time_value) / float(trotter_steps)
    t0_f = float(t0)
    sampling = str(time_sampling).strip().lower()
    dim = int(hmat_static.shape[0])
    nq = dim.bit_length() - 1  # dim == 1 << nq

    # ------------------------------------------------------------------
    # Decide once which propagation path to use.
    # ------------------------------------------------------------------
    use_sparse = False
    H_static_sparse = None
    drive_is_diagonal = False

    if dim >= _EXPM_SPARSE_MIN_DIM:
        try:
            from scipy.sparse import csc_matrix as _csc_matrix, diags as _diags
            from scipy.sparse.linalg import expm_multiply as _expm_multiply

            H_static_sparse = _csc_matrix(hmat_static)

            # Probe the drive at t=0 to inspect the label structure.
            _probe_map = dict(drive_coeff_provider_exyz(float(t0_f)))
            drive_is_diagonal = bool(_probe_map) and all(
                _is_all_z_type(lbl) for lbl in _probe_map
            )
            use_sparse = True
        except ImportError:
            pass  # scipy.sparse unavailable — fall through to dense path

    # Keep dense expm import available as fallback.
    from scipy.linalg import expm as _expm_dense

    for k in range(int(trotter_steps)):
        if sampling == "midpoint":
            t_sample = t0_f + (float(k) + 0.5) * dt
        elif sampling == "left":
            t_sample = t0_f + float(k) * dt
        else:  # right
            t_sample = t0_f + (float(k) + 1.0) * dt

        drive_map = dict(drive_coeff_provider_exyz(float(t_sample)))
        filtered_drive = {lbl: complex(c) for lbl, c in drive_map.items() if abs(c) > 1e-15}

        if use_sparse and drive_is_diagonal:
            # Fast path: drive is diagonal → build a 1-D vector and use
            # expm_multiply on H_static_sparse + diags(diag_drive).
            if filtered_drive:
                diag_drive = _build_drive_diagonal(filtered_drive, dim, nq)
                H_drive_sparse = _diags(diag_drive, format="csc")
                H_total_sparse = H_static_sparse + H_drive_sparse
            else:
                H_total_sparse = H_static_sparse
            psi = _expm_multiply((-1j * dt) * H_total_sparse, psi)

        elif use_sparse:
            # Sparse path but drive has off-diagonal terms (non-Z labels).
            if filtered_drive:
                h_drive_dense = _build_hamiltonian_matrix_from_exyz(filtered_drive)
                if h_drive_dense.shape != hmat_static.shape:
                    h_drive_dense = np.zeros_like(hmat_static)
                    for lbl, c in filtered_drive.items():
                        h_drive_dense += complex(c) * _pauli_matrix_exyz(lbl)
                H_total_sparse = H_static_sparse + _csc_matrix(h_drive_dense)
            else:
                H_total_sparse = H_static_sparse
            psi = _expm_multiply((-1j * dt) * H_total_sparse, psi)

        else:
            # Dense fallback (small systems or scipy.sparse absent).
            if filtered_drive:
                h_drive = _build_hamiltonian_matrix_from_exyz(filtered_drive)
                if h_drive.shape != hmat_static.shape:
                    h_drive = np.zeros_like(hmat_static)
                    for lbl, c in filtered_drive.items():
                        h_drive += complex(c) * _pauli_matrix_exyz(lbl)
            else:
                h_drive = np.zeros_like(hmat_static)
            h_total = hmat_static + h_drive
            psi = _expm_dense(-1j * dt * h_total) @ psi

    return _normalize_state(psi)


def _simulate_trajectory(
    *,
    num_sites: int,
    ordering: str,
    psi0: np.ndarray,
    fidelity_subspace_basis_v0: np.ndarray,
    fidelity_subspace_energy_tol: float,
    hmat: np.ndarray,
    trotter_hamiltonian_qop: SparsePauliOp,
    ordered_labels_exyz: list[str] | None = None,
    coeff_map_exyz: dict[str, complex] | None = None,
    trotter_steps: int,
    t_final: float,
    num_times: int,
    suzuki_order: int,
    drive_coeff_provider_exyz: Any | None = None,
    drive_t0: float = 0.0,
    drive_time_sampling: str = "midpoint",
    exact_steps_multiplier: int = 1,
) -> tuple[list[dict[str, float]], list[np.ndarray]]:
    if int(suzuki_order) != 2:
        raise ValueError("This script currently supports suzuki_order=2 only.")

    nq = int(trotter_hamiltonian_qop.num_qubits)
    if nq != 2 * int(num_sites):
        raise ValueError("Qubit-count mismatch between qiskit trotter operator and lattice size.")
    evals, evecs = np.linalg.eigh(hmat)
    evecs_dag = np.conjugate(evecs).T

    synthesis = SuzukiTrotter(order=int(suzuki_order), reps=int(trotter_steps), preserve_order=True)

    has_drive = drive_coeff_provider_exyz is not None

    # When drive is enabled the reference propagator may use a finer step count
    # to improve its quality independently of the Trotter discretization.
    reference_steps = int(trotter_steps) * max(1, int(exact_steps_multiplier))

    # Pre-compile Pauli actions when drive is enabled.
    compiled = None
    if has_drive:
        if ordered_labels_exyz is None or coeff_map_exyz is None:
            raise ValueError("ordered_labels_exyz and coeff_map_exyz required for time-dependent evolution")
        compiled = {lbl: _compile_pauli_action(lbl, nq) for lbl in ordered_labels_exyz}

    times = np.linspace(0.0, float(t_final), int(num_times))
    n_times = int(times.size)
    stride = max(1, n_times // 20)
    t0 = time.perf_counter()
    basis_v0 = np.asarray(fidelity_subspace_basis_v0, dtype=complex)
    if basis_v0.ndim != 2 or basis_v0.shape[0] != psi0.size:
        raise ValueError("fidelity_subspace_basis_v0 must have shape (dim, k) with matching dim.")
    if basis_v0.shape[1] <= 0:
        raise ValueError("fidelity_subspace_basis_v0 must contain at least one basis vector.")
    static_basis_eig = evecs_dag @ basis_v0
    _ai_log(
        "qiskit_trajectory_start",
        L=int(num_sites),
        num_times=n_times,
        t_final=float(t_final),
        trotter_steps=int(trotter_steps),
        reference_steps=reference_steps,
        exact_steps_multiplier=int(exact_steps_multiplier),
        suzuki_order=int(suzuki_order),
        drive_enabled=has_drive,
        ground_subspace_dimension=int(basis_v0.shape[1]),
        fidelity_subspace_energy_tol=float(fidelity_subspace_energy_tol),
        fidelity_selection_rule="E <= E0 + tol",
    )

    rows: list[dict[str, float]] = []
    exact_states: list[np.ndarray] = []

    for idx, time_val in enumerate(times):
        t = float(time_val)

        # --- exact / reference propagation ---
        if not has_drive:
            # Time-independent: eigendecomposition shortcut (exact to machine
            # precision — does not depend on exact_steps_multiplier).
            psi_exact = evecs @ (np.exp(-1j * evals * t) * (evecs_dag @ psi0))
            psi_exact = _normalize_state(psi_exact)
        else:
            # Time-dependent: piecewise-constant matrix-exponential reference
            # (exponential midpoint / Magnus-2 when time_sampling="midpoint").
            psi_exact = _evolve_piecewise_exact(
                psi0=psi0,
                hmat_static=hmat,
                drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                time_value=t,
                trotter_steps=reference_steps,
                t0=float(drive_t0),
                time_sampling=str(drive_time_sampling),
            )

        # --- Trotter propagation ---
        if not has_drive:
            if abs(t) <= 1e-15:
                psi_trot = np.array(psi0, copy=True)
            else:
                # PauliEvolutionGate bypass (intentional, not a workaround)
                # ─────────────────────────────────────────────────────────
                # PauliEvolutionGate accepts only a time-INDEPENDENT SparsePauliOp.
                # For H(t) = H_static + H_drive(t) there is no efficient way to
                # schedule time-varying coefficients through the Qiskit circuit
                # layer without rebuilding and re-transpiling at every time slice.
                #
                # Decision (see DESIGN_NOTE_QISKIT_BASELINE_TIMEDEP.md §3):
                #   This no-drive branch uses PauliEvolutionGate as intended:
                #   static Hamiltonian → validate Trotter vs hardcoded kernel.
                #   The drive branch below uses the shared scipy/sparse kernel.
                #   qiskit-dynamics (Option B) is deferred until strong-driving
                #   or Holstein phonon modes make adaptive ODE accuracy necessary.
                evo_gate = PauliEvolutionGate(
                    trotter_hamiltonian_qop,
                    time=t,
                    synthesis=synthesis,
                )
                evo_circuit = synthesis.synthesize(evo_gate)
                psi_trot = np.asarray(
                    Statevector(np.asarray(psi0, dtype=complex)).evolve(evo_circuit).data,
                    dtype=complex,
                )
                psi_trot = _normalize_state(psi_trot)
        else:
            # Drive is active: PauliEvolutionGate is NOT used here (see comment
            # above). Both pipelines share _evolve_trotter_suzuki2_absolute so
            # fidelity comparisons are apples-to-apples.
            psi_trot = _evolve_trotter_suzuki2_absolute(
                psi0,
                list(ordered_labels_exyz or []),
                dict(coeff_map_exyz or {}),
                dict(compiled or {}),
                t,
                int(trotter_steps),
                drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                t0=float(drive_t0),
                time_sampling=str(drive_time_sampling),
            )

        # --- norm-drift diagnostic ---
        norm_before = float(np.linalg.norm(psi_trot))
        norm_drift = abs(norm_before - 1.0)
        if norm_drift > 1e-6:
            _ai_log(
                "trotter_norm_drift",
                time=t,
                norm_before_renorm=norm_before,
                norm_drift=norm_drift,
            )

        if not has_drive:
            phases = np.exp(-1j * evals * t).reshape(-1, 1)
            basis_t = evecs @ (phases * static_basis_eig)
        else:
            basis_t = np.zeros((psi_trot.size, basis_v0.shape[1]), dtype=complex)
            for col in range(basis_v0.shape[1]):
                basis_t[:, col] = _evolve_piecewise_exact(
                    psi0=basis_v0[:, col],
                    hmat_static=hmat,
                    drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                    time_value=t,
                    trotter_steps=reference_steps,
                    t0=float(drive_t0),
                    time_sampling=str(drive_time_sampling),
                )

        basis_t_orth = _orthonormalize_basis_columns(basis_t)
        if basis_t_orth.shape[1] < basis_v0.shape[1]:
            _ai_log(
                "qiskit_fidelity_subspace_rank_reduced",
                time=float(t),
                original_dimension=int(basis_v0.shape[1]),
                effective_dimension=int(basis_t_orth.shape[1]),
            )
        fidelity = _projector_fidelity_from_basis(basis_t_orth, psi_trot)
        n_up_exact_site, n_dn_exact_site, doublon_exact = _site_resolved_number_observables(
            psi_exact,
            num_sites,
            ordering,
        )
        n_up_trot_site, n_dn_trot_site, doublon_trot = _site_resolved_number_observables(
            psi_trot,
            num_sites,
            ordering,
        )
        n_exact_site = n_up_exact_site + n_dn_exact_site
        n_trot_site = n_up_trot_site + n_dn_trot_site
        n_up_exact = float(n_up_exact_site[0]) if n_up_exact_site.size > 0 else float("nan")
        n_dn_exact = float(n_dn_exact_site[0]) if n_dn_exact_site.size > 0 else float("nan")
        n_up_trot = float(n_up_trot_site[0]) if n_up_trot_site.size > 0 else float("nan")
        n_dn_trot = float(n_dn_trot_site[0]) if n_dn_trot_site.size > 0 else float("nan")
        energy_static_exact = _expectation_hamiltonian(psi_exact, hmat)
        energy_static_trotter = _expectation_hamiltonian(psi_trot, hmat)

        # --- total (instantaneous) energy: H_static + H_drive(t) ---
        # The physical time for the drive at observation time t is
        # drive_t0 + t, matching the propagator convention.
        if has_drive:
            t_physical = float(drive_t0) + t
            hmat_drive_t = _build_drive_matrix_at_time(
                drive_coeff_provider_exyz, t_physical, nq,
            )
            hmat_total_t = hmat + hmat_drive_t
            energy_total_exact = _expectation_hamiltonian(psi_exact, hmat_total_t)
            energy_total_trotter = _expectation_hamiltonian(psi_trot, hmat_total_t)
        else:
            energy_total_exact = energy_static_exact
            energy_total_trotter = energy_static_trotter

        rows.append(
            {
                "time": t,
                "fidelity": fidelity,
                "energy_static_exact": energy_static_exact,
                "energy_static_trotter": energy_static_trotter,
                "energy_total_exact": energy_total_exact,
                "energy_total_trotter": energy_total_trotter,
                "n_up_site0_exact": n_up_exact,
                "n_up_site0_trotter": n_up_trot,
                "n_dn_site0_exact": n_dn_exact,
                "n_dn_site0_trotter": n_dn_trot,
                "n_site_exact": [float(x) for x in n_exact_site.tolist()],
                "n_site_trotter": [float(x) for x in n_trot_site.tolist()],
                "staggered_exact": _staggered_order(n_exact_site),
                "staggered_trotter": _staggered_order(n_trot_site),
                "doublon_exact": doublon_exact,
                "doublon_trotter": doublon_trot,
                "doublon_avg_exact": float(doublon_exact / float(num_sites)),
                "doublon_avg_trotter": float(doublon_trot / float(num_sites)),
                "norm_before_renorm": norm_before,
            }
        )
        exact_states.append(psi_exact)
        if idx == 0 or idx == n_times - 1 or ((idx + 1) % stride == 0):
            _ai_log(
                "qiskit_trajectory_progress",
                step=int(idx + 1),
                total_steps=n_times,
                frac=round(float((idx + 1) / n_times), 6),
                time=float(t),
                subspace_fidelity=float(fidelity),
                elapsed_sec=round(time.perf_counter() - t0, 6),
            )

    _ai_log(
        "qiskit_trajectory_done",
        total_steps=n_times,
        elapsed_sec=round(time.perf_counter() - t0, 6),
        final_subspace_fidelity=float(rows[-1]["fidelity"]) if rows else None,
        final_energy_static_trotter=float(rows[-1]["energy_static_trotter"]) if rows else None,
    )

    return rows, exact_states


def _current_command_string() -> str:
    return " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])


def _render_command_page(pdf: PdfPages, command: str) -> None:
    wrapped = textwrap.wrap(command, width=112, subsequent_indent="  ")
    lines = [
        "Executed Command",
        "",
        "Reference: pipelines/PIPELINE_RUN_GUIDE.md",
        "Script: pipelines/qiskit_hubbard_baseline_pipeline.py",
        "",
        *wrapped,
    ]
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


def _render_text_page(pdf: PdfPages, lines: list[str], *, fontsize: int = 10) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=fontsize)
    pdf.savefig(fig)
    plt.close(fig)


def _write_pipeline_pdf(pdf_path: Path, payload: dict[str, Any], run_command: str) -> None:
    traj = payload["trajectory"]
    times = np.array([float(r["time"]) for r in traj], dtype=float)
    markevery = max(1, times.size // 25)

    def arr(key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in traj], dtype=float)

    fid = arr("fidelity")
    e_exact = arr("energy_static_exact")
    e_trot = arr("energy_static_trotter")
    nu_exact = arr("n_up_site0_exact")
    nu_trot = arr("n_up_site0_trotter")
    nd_exact = arr("n_dn_site0_exact")
    nd_trot = arr("n_dn_site0_trotter")
    d_exact = arr("doublon_exact")
    d_trot = arr("doublon_trotter")
    gs_exact = float(payload["ground_state"]["exact_energy"])
    gs_exact_filtered_raw = payload["ground_state"].get("exact_energy_filtered")
    gs_exact_filtered = float(gs_exact_filtered_raw) if gs_exact_filtered_raw is not None else gs_exact
    vqe_e = payload.get("vqe", {}).get("energy")
    vqe_val = float(vqe_e) if vqe_e is not None else np.nan
    vqe_sector = payload.get("vqe", {}).get("num_particles", {})
    sector_label = (
        f"N_up={vqe_sector.get('n_up','?')}, N_dn={vqe_sector.get('n_dn','?')}"
        if vqe_sector else "half-filled"
    )
    settings = payload.get("settings", {})
    qiskit_ansatz = str(payload.get("vqe", {}).get("method", "unknown")).strip().lower()
    run_mode = "drive-enabled" if isinstance(settings.get("drive"), dict) else "static"
    _ = run_command  # command text is preserved in CLI logs; PDF starts with summary

    with PdfPages(str(pdf_path)) as pdf:
        summary_lines = [
            f"Qiskit Hubbard Summary (L={settings.get('L')})",
            "",
            "Ansatz Used:",
            f"  - qiskit ansatz/method: {qiskit_ansatz}",
            "",
            "Run Settings:",
            f"  - mode: {run_mode}",
            f"  - t={settings.get('t')}  u={settings.get('u')}  dv={settings.get('dv')}",
            f"  - boundary={settings.get('boundary')}  ordering={settings.get('ordering')}",
            f"  - trotter_steps={settings.get('trotter_steps')}  suzuki_order={settings.get('suzuki_order')}",
            f"  - t_final={settings.get('t_final')}  num_times={settings.get('num_times')}",
            f"  - initial_state_source={settings.get('initial_state_source')}",
            "",
            "Topline:",
            f"  - subspace_fidelity_at_t0: {float(fid[0]) if fid.size > 0 else None}",
            f"  - vqe_energy: {payload.get('vqe', {}).get('energy')}",
            f"  - exact_filtered_energy: {payload['ground_state'].get('exact_energy_filtered')}",
            f"  - qpe_energy_estimate: {payload.get('qpe', {}).get('energy_estimate')}",
        ]
        _render_text_page(pdf, summary_lines, fontsize=10)

        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
        ax00, ax01 = axes[0, 0], axes[0, 1]
        ax10, ax11 = axes[1, 0], axes[1, 1]

        ax00.plot(times, fid, color="#0b3d91", marker="o", markersize=3, markevery=markevery)
        fid_title = payload.get("settings", {}).get(
            "fidelity_definition_short",
            "Subspace Fidelity(t) = <psi_ansatz_trot(t)|P_exact_gs_subspace(t)|psi_ansatz_trot(t)>",
        )
        ax00.set_title(str(fid_title))
        ax00.grid(alpha=0.25)

        ax01.plot(times, e_exact, label=f"{EXACT_LABEL} (static)", color="#111111", linewidth=2.0, marker="s", markersize=3, markevery=markevery)
        ax01.plot(times, e_trot, label="Trotter (static)", color="#d62728", linestyle="--", linewidth=1.4, marker="^", markersize=3, markevery=markevery)

        # --- optional total-energy overlay (when drive active and differs) ---
        e_total_exact = arr("energy_total_exact")
        e_total_trot = arr("energy_total_trotter")
        if not (np.allclose(e_total_exact, e_exact, atol=1e-14) and
                np.allclose(e_total_trot, e_trot, atol=1e-14)):
            ax01.plot(times, e_total_exact, label=f"{EXACT_LABEL} (total)", color="#17becf",
                      linewidth=1.6, marker="D", markersize=2.5, markevery=markevery, alpha=0.8)
            ax01.plot(times, e_total_trot, label="Trotter (total)", color="#ff7f0e",
                      linestyle="--", linewidth=1.2, marker="<", markersize=2.5, markevery=markevery, alpha=0.8)

        ax01.set_title("Energy")
        ax01.grid(alpha=0.25)
        ax01.legend(fontsize=7)

        ax10.plot(times, nu_exact, label=f"n_up0 {EXACT_LABEL}", color="#17becf", linewidth=1.8, marker="o", markersize=3, markevery=markevery)
        ax10.plot(times, nu_trot, label="n_up0 trotter", color="#0f7f8b", linestyle="--", linewidth=1.2, marker="s", markersize=3, markevery=markevery)
        ax10.plot(times, nd_exact, label=f"n_dn0 {EXACT_LABEL}", color="#9467bd", linewidth=1.8, marker="^", markersize=3, markevery=markevery)
        ax10.plot(times, nd_trot, label="n_dn0 trotter", color="#6f4d8f", linestyle="--", linewidth=1.2, marker="v", markersize=3, markevery=markevery)
        ax10.set_title("Site-0 Occupations")
        ax10.set_xlabel("Time")
        ax10.grid(alpha=0.25)
        ax10.legend(fontsize=8)

        ax11.plot(times, d_exact, label=f"doublon {EXACT_LABEL}", color="#8c564b", linewidth=1.8, marker="o", markersize=3, markevery=markevery)
        ax11.plot(times, d_trot, label="doublon trotter", color="#c251a1", linestyle="--", linewidth=1.2, marker="s", markersize=3, markevery=markevery)
        ax11.set_title("Total Doublon")
        ax11.set_xlabel("Time")
        ax11.grid(alpha=0.25)
        ax11.legend(fontsize=8)

        fig.suptitle(f"Qiskit Hubbard Baseline Pipeline: L={payload['settings']['L']}", fontsize=14)
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        filt_label = f"Exact (sector {sector_label})"
        figv, axesv = plt.subplots(1, 2, figsize=(11.0, 8.5))
        vx0, vx1 = axesv[0], axesv[1]
        vx0.bar([0, 1], [gs_exact_filtered, vqe_val], color=["#2166ac", "#ff7f0e"], edgecolor="black", linewidth=0.4)
        vx0.set_xticks([0, 1])
        vx0.set_xticklabels([filt_label, "VQE"], fontsize=8)
        vx0.set_ylabel("Energy")
        vx0.set_title("VQE Energy vs Exact (filtered sector)")
        vx0.grid(axis="y", alpha=0.25)

        err_vqe = abs(vqe_val - gs_exact_filtered) if (np.isfinite(vqe_val) and np.isfinite(gs_exact_filtered)) else np.nan
        vx1.bar([0], [err_vqe], color="#ff7f0e", edgecolor="black", linewidth=0.4)
        vx1.set_xticks([0])
        vx1.set_xticklabels([f"|VQE \u2212 Exact (filtered)|"])
        vx1.set_ylabel("Absolute Error")
        vx1.set_title("VQE Absolute Error vs Exact (filtered sector)")
        vx1.grid(axis="y", alpha=0.25)

        figv.suptitle(
            "VQE optimises within the half-filled sector; exact (filtered) is the true sector ground state.\n"
            "Full-Hilbert exact energy is in the JSON text summary only.",
            fontsize=10,
        )
        figv.tight_layout(rect=(0.0, 0.03, 1.0, 0.91))
        pdf.savefig(figv)
        plt.close(figv)

        lines = [
            "Qiskit Hubbard baseline pipeline summary",
            "",
            "Ansatz:",
            f"  - qiskit ansatz/method: {qiskit_ansatz}",
            "",
            "Energy + Fidelity:",
            f"  - subspace_fidelity_at_t0: {float(fid[0]) if fid.size > 0 else None}",
            f"  - ground_state_exact_energy_full_hilbert: {payload['ground_state']['exact_energy']:.12f}",
            f"  - ground_state_exact_energy_filtered: {payload['ground_state'].get('exact_energy_filtered')}",
            f"  - filtered_sector: {payload['ground_state'].get('filtered_sector')}",
            f"  - vqe_energy: {payload['vqe'].get('energy')}",
            f"  - qpe_energy_estimate: {payload['qpe'].get('energy_estimate')}",
            "",
            "Config:",
            f"  - initial_state_source: {payload['initial_state']['source']}",
            f"  - exact_trajectory_label: {EXACT_LABEL}",
            f"  - exact_trajectory_method: {EXACT_METHOD}",
            f"  - fidelity_definition: {payload['settings'].get('fidelity_definition')}",
            f"  - hamiltonian_terms: {payload['hamiltonian']['num_terms']}",
            f"  - reference_sanity: {payload['sanity']['jw_reference']}",
        ]
        _render_text_page(pdf, lines, fontsize=9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qiskit baseline Hubbard pipeline runner.")
    parser.add_argument("--L", type=int, required=True, help="Number of lattice sites.")
    parser.add_argument("--t", type=float, default=1.0, help="Hopping coefficient.")
    parser.add_argument("--u", type=float, default=4.0, help="Onsite interaction U.")
    parser.add_argument("--dv", type=float, default=0.0, help="Uniform local potential term v (Hv = -v n).")
    parser.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    parser.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    parser.add_argument("--t-final", type=float, default=20.0)
    parser.add_argument("--num-times", type=int, default=201)
    parser.add_argument("--suzuki-order", type=int, default=2)
    parser.add_argument("--trotter-steps", type=int, default=64)
    parser.add_argument(
        "--fidelity-subspace-energy-tol",
        type=float,
        default=1e-8,
        help=(
            "Energy tolerance for the filtered-sector ground manifold used in "
            "subspace fidelity: include states with E <= E0 + tol."
        ),
    )
    parser.add_argument("--term-order", choices=["qiskit", "sorted"], default="sorted")

    # --- time-dependent drive arguments ---
    parser.add_argument("--enable-drive", action="store_true", help="Enable time-dependent onsite density drive.")
    parser.add_argument("--drive-A", type=float, default=0.0, help="Drive amplitude A in v(t)=A*sin(wt+phi)*exp(-t^2/(2 tbar^2)).")
    parser.add_argument("--drive-omega", type=float, default=1.0, help="Drive carrier angular frequency w.")
    parser.add_argument("--drive-tbar", type=float, default=1.0, help="Drive Gaussian envelope width tbar (must be > 0).")
    parser.add_argument("--drive-phi", type=float, default=0.0, help="Drive phase phi.")
    parser.add_argument(
        "--drive-pattern",
        choices=["dimer_bias", "staggered", "custom"],
        default="staggered",
        help="Spatial pattern mode for v_i(t)=s_i*v(t).",
    )
    parser.add_argument(
        "--drive-custom-s",
        type=str,
        default=None,
        help="Custom spatial weights s_i (comma-separated or JSON list), length L, used when --drive-pattern custom.",
    )
    parser.add_argument("--drive-include-identity", action="store_true", help="Include identity term from n=(I-Z)/2 (global phase).")
    parser.add_argument(
        "--drive-time-sampling",
        choices=["midpoint", "left", "right"],
        default="midpoint",
        help="Time sampling rule per Trotter slice (midpoint recommended; left/right for diagnostics).",
    )
    parser.add_argument("--drive-t0", type=float, default=0.0, help="Drive start time t0 for evolution (default 0.0).")
    parser.add_argument(
        "--exact-steps-multiplier",
        type=int,
        default=1,
        help=(
            "Reference-propagator refinement factor (default 1). "
            "When drive is enabled the reference runs at "
            "N_ref = exact_steps_multiplier * trotter_steps steps while the "
            "Trotter circuit runs at trotter_steps. "
            "With midpoint sampling (Magnus-2, O(Δt²)) a larger multiplier "
            "strictly improves reference quality. "
            "Has no effect when drive is disabled (the static reference uses "
            "exact eigendecomposition)."
        ),
    )

    parser.add_argument("--vqe-reps", type=int, default=2)
    parser.add_argument("--vqe-restarts", type=int, default=3)
    parser.add_argument("--vqe-seed", type=int, default=7)
    parser.add_argument("--vqe-maxiter", type=int, default=120)
    parser.add_argument(
        "--vqe-optimizer",
        type=str,
        default="COBYLA",
        choices=["SLSQP", "COBYLA", "L_BFGS_B"],
        help="Qiskit optimizer used by VQE.",
    )
    parser.add_argument("--qpe-eval-qubits", type=int, default=6)
    parser.add_argument("--qpe-shots", type=int, default=1024)
    parser.add_argument("--qpe-seed", type=int, default=11)
    parser.add_argument("--skip-qpe", action="store_true", help="Skip QPE execution and mark qpe payload as skipped.")

    parser.add_argument("--initial-state-source", choices=["exact", "vqe", "hf"], default="vqe")

    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument("--skip-pdf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ai_log("qiskit_main_start", settings=vars(args))
    run_command = _current_command_string()
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_json = args.output_json or (artifacts_dir / f"qiskit_pipeline_L{args.L}.json")
    output_pdf = args.output_pdf or (artifacts_dir / f"qiskit_pipeline_L{args.L}.pdf")

    qop, mapper = _build_qiskit_qubit_hamiltonian(
        num_sites=int(args.L),
        t=float(args.t),
        u=float(args.u),
        dv=float(args.dv),
        boundary=str(args.boundary),
        ordering=str(args.ordering),
    )

    native_order, coeff_map_exyz = _qiskit_terms_exyz(qop)
    _ai_log("qiskit_hamiltonian_built", L=int(args.L), num_terms=int(len(coeff_map_exyz)))
    if args.term_order == "qiskit":
        ordered_labels_exyz = list(native_order)
    else:
        ordered_labels_exyz = sorted(coeff_map_exyz)
    trotter_qop_ordered = _ordered_qop_from_exyz(ordered_labels_exyz, coeff_map_exyz)

    # --- build time-dependent drive (if enabled) ---
    drive = None
    drive_coeff_provider_exyz = None
    if bool(args.enable_drive):
        custom_weights = None
        if str(args.drive_pattern) == "custom":
            if args.drive_custom_s is None:
                raise ValueError("--drive-custom-s is required when --drive-pattern custom")
            raw = str(args.drive_custom_s).strip()
            if raw.startswith("["):
                custom_weights = json.loads(raw)
            else:
                custom_weights = [float(x) for x in raw.split(",") if x.strip()]
        drive = build_gaussian_sinusoid_density_drive(
            n_sites=int(args.L),
            nq_total=int(2 * args.L),
            indexing=str(args.ordering),
            A=float(args.drive_A),
            omega=float(args.drive_omega),
            tbar=float(args.drive_tbar),
            phi=float(args.drive_phi),
            pattern_mode=str(args.drive_pattern),
            custom_weights=custom_weights,
            include_identity=bool(args.drive_include_identity),
            coeff_tol=0.0,
        )
        drive_coeff_provider_exyz = drive.coeff_map_exyz
        drive_labels = set(drive.template.labels_exyz(include_identity=bool(drive.include_identity)))
        missing = sorted(drive_labels.difference(ordered_labels_exyz))
        ordered_labels_exyz = list(ordered_labels_exyz) + list(missing)
        _ai_log(
            "qiskit_drive_built",
            L=int(args.L),
            drive_labels=len(drive_labels),
            new_labels=len(missing),
        )

    hmat = np.asarray(qop.to_matrix(sparse=False), dtype=complex)
    evals, evecs = np.linalg.eigh(hmat)
    gs_idx = int(np.argmin(evals))
    gs_energy_exact = float(np.real(evals[gs_idx]))
    psi_exact_ground = _normalize_state(np.asarray(evecs[:, gs_idx], dtype=complex).reshape(-1))
    num_particles = _half_filled_particles(int(args.L))
    _fidelity_subspace_tol = float(args.fidelity_subspace_energy_tol)
    if _fidelity_subspace_tol < 0.0:
        raise ValueError("--fidelity-subspace-energy-tol must be >= 0.")
    try:
        gs_energy_exact_filtered, fidelity_subspace_basis_v0 = _ground_manifold_basis_sector_filtered(
            hmat=hmat,
            num_sites=int(args.L),
            num_particles=num_particles,
            ordering=str(args.ordering),
            energy_tol=_fidelity_subspace_tol,
        )
        fidelity_subspace_dimension = int(fidelity_subspace_basis_v0.shape[1])
    except Exception as _exc_filt:
        _ai_log("qiskit_filtered_exact_failed", error=str(_exc_filt))
        gs_energy_exact_filtered = None
        fidelity_subspace_basis_v0 = psi_exact_ground.reshape(-1, 1)
        fidelity_subspace_dimension = 1

    vqe_payload, psi_vqe = _run_qiskit_vqe(
        num_sites=int(args.L),
        qop=qop,
        mapper=mapper,
        hopping_t=float(args.t),
        onsite_u=float(args.u),
        dv=float(args.dv),
        boundary=str(args.boundary),
        ordering=str(args.ordering),
        reps=int(args.vqe_reps),
        restarts=int(args.vqe_restarts),
        seed=int(args.vqe_seed),
        maxiter=int(args.vqe_maxiter),
        optimizer_name=str(args.vqe_optimizer),
    )

    psi_hf = _normalize_state(
        np.asarray(
            hartree_fock_statevector(int(args.L), num_particles, indexing=str(args.ordering)),
            dtype=complex,
        ).reshape(-1)
    )

    if args.initial_state_source == "vqe" and psi_vqe is not None:
        psi0 = psi_vqe
        init_source = "vqe"
        _ai_log("qiskit_initial_state_selected", source="vqe")
    elif args.initial_state_source == "vqe":
        raise RuntimeError("Requested --initial-state-source vqe but Qiskit VQE statevector is unavailable.")
    elif args.initial_state_source == "hf":
        psi0 = psi_hf
        init_source = "hf"
        _ai_log("qiskit_initial_state_selected", source="hf")
    else:
        psi0 = psi_exact_ground
        init_source = "exact"
        _ai_log("qiskit_initial_state_selected", source="exact")

    if args.skip_qpe:
        qpe_payload = {
            "success": False,
            "method": "qpe_skipped",
            "energy_estimate": None,
            "phase": None,
            "skipped": True,
            "reason": "--skip-qpe enabled",
            "num_evaluation_qubits": int(args.qpe_eval_qubits),
            "shots": int(args.qpe_shots),
        }
        _ai_log("qiskit_qpe_skipped", eval_qubits=int(args.qpe_eval_qubits), shots=int(args.qpe_shots))
    else:
        qpe_payload = _run_qiskit_qpe(
            qop=qop,
            psi_init=psi0,
            eval_qubits=int(args.qpe_eval_qubits),
            shots=int(args.qpe_shots),
            seed=int(args.qpe_seed),
        )

    trajectory, _exact_states = _simulate_trajectory(
        num_sites=int(args.L),
        ordering=str(args.ordering),
        psi0=psi0,
        fidelity_subspace_basis_v0=fidelity_subspace_basis_v0,
        fidelity_subspace_energy_tol=_fidelity_subspace_tol,
        hmat=hmat,
        trotter_hamiltonian_qop=trotter_qop_ordered,
        ordered_labels_exyz=ordered_labels_exyz,
        coeff_map_exyz=coeff_map_exyz,
        trotter_steps=int(args.trotter_steps),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        suzuki_order=int(args.suzuki_order),
        drive_coeff_provider_exyz=drive_coeff_provider_exyz,
        drive_t0=float(args.drive_t0),
        drive_time_sampling=str(args.drive_time_sampling),
        exact_steps_multiplier=int(args.exact_steps_multiplier),
    )

    sanity = {
        "jw_reference": _reference_sanity(
            num_sites=int(args.L),
            t=float(args.t),
            u=float(args.u),
            dv=float(args.dv),
            boundary=str(args.boundary),
            ordering=str(args.ordering),
            coeff_map_exyz=coeff_map_exyz,
        )
    }

    settings: dict[str, Any] = {
        "L": int(args.L),
        "t": float(args.t),
        "u": float(args.u),
        "dv": float(args.dv),
        "boundary": str(args.boundary),
        "ordering": str(args.ordering),
        "t_final": float(args.t_final),
        "num_times": int(args.num_times),
        "suzuki_order": int(args.suzuki_order),
        "trotter_steps": int(args.trotter_steps),
        "term_order": str(args.term_order),
        "initial_state_source": str(init_source),
        "skip_qpe": bool(args.skip_qpe),
        "fidelity_definition_short": (
            "Subspace Fidelity(t) = <psi_ansatz_trot(t)|P_exact_gs_subspace(t)|psi_ansatz_trot(t)>"
        ),
        "fidelity_definition": (
            "fidelity(t) = <psi_ansatz_trot(t)|P_exact_gs_subspace(t)|psi_ansatz_trot(t)>, "
            "where P_exact_gs_subspace(t) projects onto the time-evolved filtered-sector "
            "ground manifold selected by E <= E0 + tol."
        ),
        "fidelity_subspace_energy_tol": float(_fidelity_subspace_tol),
        "fidelity_reference_subspace": {
            "sector": {
                "n_up": int(num_particles[0]),
                "n_dn": int(num_particles[1]),
            },
            "ground_subspace_dimension": int(fidelity_subspace_dimension),
            "selection_rule": "E <= E0 + tol",
        },
        "fidelity_reference_initial_state": "exact_static_ground_manifold_filtered_sector",
        "fidelity_reference_sector": {
            "n_up": int(num_particles[0]),
            "n_dn": int(num_particles[1]),
        },
        "fidelity_ansatz_initial_state": str(init_source),
        "energy_observable_definition": (
            "energy_static_* measures <psi|H_static|psi>. "
            "energy_total_* measures <psi|H_static + H_drive(drive_t0 + t)|psi>. "
            "When drive is disabled, energy_total_* == energy_static_*. "
            "Drive sampling uses the same drive_t0 convention as propagation."
        ),
    }
    if bool(args.enable_drive):
        settings["drive"] = {
            "enabled": True,
            "A": float(args.drive_A),
            "omega": float(args.drive_omega),
            "tbar": float(args.drive_tbar),
            "phi": float(args.drive_phi),
            "pattern": str(args.drive_pattern),
            "custom_s": (str(args.drive_custom_s) if args.drive_custom_s is not None else None),
            "include_identity": bool(args.drive_include_identity),
            "time_sampling": str(args.drive_time_sampling),
            "t0": float(args.drive_t0),
            # Reference-propagator metadata (Prompt 4/5).
            "reference_steps_multiplier": int(args.exact_steps_multiplier),
            "reference_steps": int(args.trotter_steps) * int(args.exact_steps_multiplier),
            "reference_method": reference_method_name(str(args.drive_time_sampling)),
            # Architecture metadata (see DESIGN_NOTE_QISKIT_BASELINE_TIMEDEP.md §5).
            # PauliEvolutionGate is NOT used for time-dependent propagation.
            # Both drive paths use the shared scipy/sparse expm_multiply kernel.
            "propagator_backend": "scipy_sparse_expm_multiply",
        }

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "qiskit",
        "settings": settings,
        "hamiltonian": {
            "num_qubits": int(2 * args.L),
            "num_terms": int(len(ordered_labels_exyz) if bool(args.enable_drive) else len(coeff_map_exyz)),
            "coefficients_exyz": [
                {
                    "label_exyz": lbl,
                    "coeff": {
                        "re": float(np.real(coeff_map_exyz.get(lbl, 0.0 + 0.0j))),
                        "im": float(np.imag(coeff_map_exyz.get(lbl, 0.0 + 0.0j))),
                    },
                }
                for lbl in ordered_labels_exyz
            ],
        },
        "ground_state": {
            "exact_energy": float(gs_energy_exact),
            "exact_energy_filtered": (
                float(gs_energy_exact_filtered)
                if gs_energy_exact_filtered is not None
                else vqe_payload.get("exact_filtered_energy")
            ),
            "filtered_sector": {
                "n_up": int(num_particles[0]),
                "n_dn": int(num_particles[1]),
            },
            "ground_subspace_dimension": int(fidelity_subspace_dimension),
            "method": "matrix_diagonalization",
        },
        "vqe": vqe_payload,
        "qpe": qpe_payload,
        "initial_state": {
            "source": str(init_source),
            "amplitudes_qn_to_q0": _state_to_amplitudes_qn_to_q0(psi0),
        },
        "trajectory": trajectory,
        "sanity": sanity,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.skip_pdf:
        _write_pipeline_pdf(output_pdf, payload, run_command)

    _ai_log(
        "qiskit_main_done",
        L=int(args.L),
        output_json=str(output_json),
        output_pdf=(str(output_pdf) if not args.skip_pdf else None),
        vqe_energy=vqe_payload.get("energy"),
        qpe_energy=qpe_payload.get("energy_estimate"),
    )
    print(f"Wrote JSON: {output_json}")
    if not args.skip_pdf:
        print(f"Wrote PDF:  {output_pdf}")


if __name__ == "__main__":
    main()
