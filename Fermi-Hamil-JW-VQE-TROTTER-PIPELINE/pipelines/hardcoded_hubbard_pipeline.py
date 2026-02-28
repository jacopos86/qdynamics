#!/usr/bin/env python3
"""Hardcoded-first end-to-end Hubbard pipeline.

Flow:
1) Build hardcoded Hubbard Hamiltonian (JW) from repo source-of-truth helpers.
2) Run hardcoded VQE on numpy-statevector backend (SciPy optional fallback comes from
   the notebook implementation).
3) Run temporary QPE adapter (Qiskit-only, isolated in one function).
4) Run hardcoded Suzuki-2 Trotter dynamics and exact dynamics.
5) Emit JSON + compact PDF artifact.
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

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.hartree_fock_reference_state import hartree_fock_statevector
from src.quantum.hubbard_latex_python_pairs import build_hubbard_hamiltonian
from src.quantum.drives_time_potential import (
    build_gaussian_sinusoid_density_drive,
    evaluate_drive_waveform,
    reference_method_name,
)


def _ai_log(event: str, **fields: Any) -> None:
    payload = {
        "event": str(event),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


PAULI_MATS = {
    "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}

EXACT_LABEL = "Exact_Hardcode"
EXACT_METHOD = "python_matrix_eigendecomposition"


@dataclass(frozen=True)
class CompiledPauliAction:
    label_exyz: str
    perm: np.ndarray
    phase: np.ndarray


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
        # Qubits 0 … L-1 carry spin-up; L … 2L-1 carry spin-down.
        up_mask = (1 << num_sites) - 1          # bits 0..L-1
        dn_mask = up_mask << num_sites           # bits L..2L-1
        n_up_arr = np.array([bin(int(i) & int(up_mask)).count("1") for i in idx_all], dtype=np.int32)
        n_dn_arr = np.array([bin(int(i) & int(dn_mask)).count("1") for i in idx_all], dtype=np.int32)
    else:
        # Interleaved: even bits → spin-up, odd bits → spin-down.
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
    """Return (ground_energy, embedded_ground_manifold_basis) in a fixed sector.

    The returned basis matrix has orthonormal columns spanning the filtered
    sector states with energies satisfying E <= E0 + tol.
    """
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
    """QR-orthonormalize columns and drop near-null columns by rank threshold."""
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
    """Return <psi|P|psi> for P projecting onto span(basis_orthonormal)."""
    amps = np.conjugate(basis_orthonormal).T @ psi
    raw = np.vdot(amps, amps)
    val = float(np.real(raw))
    if val < 0.0 and val > -1e-12:
        val = 0.0
    if val > 1.0 and val < 1.0 + 1e-12:
        val = 1.0
    return float(min(1.0, max(0.0, val)))


def _exact_ground_state_sector_filtered(
    hmat: np.ndarray,
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
) -> tuple[float, np.ndarray]:
    """Return (ground_energy, embedded_ground_statevector) in a fixed sector."""
    gs_energy, basis_full = _ground_manifold_basis_sector_filtered(
        hmat=hmat,
        num_sites=num_sites,
        num_particles=num_particles,
        ordering=ordering,
        energy_tol=0.0,
    )
    psi_full = _normalize_state(np.asarray(basis_full[:, 0], dtype=complex).reshape(-1))
    return float(gs_energy), psi_full


def _exact_energy_sector_filtered(
    hmat: np.ndarray,
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
) -> float:
    """Backward-compatible helper: return only the filtered-sector energy."""
    gs_energy, _psi = _exact_ground_state_sector_filtered(
        hmat=hmat,
        num_sites=num_sites,
        num_particles=num_particles,
        ordering=ordering,
    )
    return gs_energy


def _pauli_matrix_exyz(label: str) -> np.ndarray:
    mats = [PAULI_MATS[ch] for ch in label]
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out


def _collect_hardcoded_terms_exyz(
    h_poly,
) -> tuple[list[str], dict[str, complex]]:
    """Walk a PauliPolynomial and return (native_order, coeff_map_exyz).

    Duplicate labels are accumulated (summed).  The native_order list
    preserves insertion order of *first occurrence*, matching the order
    the Hamiltonian builder emitted the terms.
    """
    coeff_map: dict[str, complex] = {}
    native_order: list[str] = []
    for term in h_poly.return_polynomial():
        label: str = term.pw2strng()  # lowercase exyz string
        coeff: complex = complex(term.p_coeff)
        if label not in coeff_map:
            native_order.append(label)
            coeff_map[label] = coeff
        else:
            coeff_map[label] = coeff_map[label] + coeff
    return native_order, coeff_map



def _build_hamiltonian_matrix(coeff_map_exyz: dict[str, complex]) -> np.ndarray:
    if not coeff_map_exyz:
        return np.zeros((1, 1), dtype=complex)
    nq = len(next(iter(coeff_map_exyz)))
    dim = 1 << nq
    hmat = np.zeros((dim, dim), dtype=complex)
    for label, coeff in coeff_map_exyz.items():
        hmat += coeff * _pauli_matrix_exyz(label)
    return hmat


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
    """Suzuki-Trotter order-2 evolution, with optional time-dependent drive.

    When *drive_coeff_provider_exyz* is ``None`` the original bit-for-bit
    time-independent path is taken (no behavioural change).

    When provided, drive coefficients are sampled once per Trotter slice and
    additively merged with the static coefficients.
    """
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

    # --- time-dependent path: coefficients sampled once per slice ---
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


def _load_hardcoded_vqe_namespace() -> dict[str, Any]:
    from src.quantum import vqe_latex_python_pairs as vqe_mod

    ns: dict[str, Any] = {name: getattr(vqe_mod, name) for name in dir(vqe_mod)}

    required = [
        "half_filled_num_particles",
        "hartree_fock_bitstring",
        "basis_state",
        "HardcodedUCCSDAnsatz",
        "HardcodedUCCSDLayerwiseAnsatz",
        "HubbardLayerwiseAnsatz",
        "exact_ground_energy_sector",
        "vqe_minimize",
    ]
    missing = [name for name in required if name not in ns]
    if missing:
        raise RuntimeError(f"Missing required VQE notebook symbols: {missing}")
    return ns


def _run_hardcoded_vqe(
    *,
    num_sites: int,
    ordering: str,
    boundary: str,
    hopping_t: float,
    onsite_u: float,
    potential_dv: float,
    h_poly: Any,
    reps: int,
    restarts: int,
    seed: int,
    maxiter: int,
    method: str,
    ansatz_name: str,
) -> tuple[dict[str, Any], np.ndarray]:
    t0 = time.perf_counter()
    _ai_log(
        "hardcoded_vqe_start",
        L=int(num_sites),
        ordering=str(ordering),
        reps=int(reps),
        restarts=int(restarts),
        maxiter=int(maxiter),
        seed=int(seed),
        method=str(method),
        ansatz=str(ansatz_name),
    )
    ns = _load_hardcoded_vqe_namespace()
    ansatz_name_s = str(ansatz_name).strip().lower()
    if ansatz_name_s not in {"uccsd", "hva"}:
        raise ValueError(f"Unsupported --vqe-ansatz '{ansatz_name}'. Expected one of: uccsd, hva.")

    num_particles = tuple(ns["half_filled_num_particles"](int(num_sites)))
    hf_bits = str(ns["hartree_fock_bitstring"](n_sites=int(num_sites), num_particles=num_particles, indexing=ordering))
    nq = 2 * int(num_sites)
    psi_ref = np.asarray(ns["basis_state"](nq, hf_bits), dtype=complex)

    if ansatz_name_s == "uccsd":
        ansatz = ns["HardcodedUCCSDLayerwiseAnsatz"](
            dims=int(num_sites),
            num_particles=num_particles,
            reps=int(reps),
            repr_mode="JW",
            indexing=ordering,
            include_singles=True,
            include_doubles=True,
        )
        method_name = "hardcoded_uccsd_layerwise_statevector"
    else:
        ansatz = ns["HubbardLayerwiseAnsatz"](
            dims=int(num_sites),
            t=float(hopping_t),
            U=float(onsite_u),
            v=float(potential_dv),
            reps=int(reps),
            repr_mode="JW",
            indexing=ordering,
            pbc=(str(boundary).strip().lower() == "periodic"),
            include_potential_terms=True,
        )
        method_name = "hardcoded_hva_layerwise_statevector"

    result = ns["vqe_minimize"](
        h_poly,
        ansatz,
        psi_ref,
        restarts=int(restarts),
        seed=int(seed),
        maxiter=int(maxiter),
        method=str(method),
    )

    theta = np.asarray(result.theta, dtype=float)
    psi_vqe = np.asarray(ansatz.prepare_state(theta, psi_ref), dtype=complex).reshape(-1)
    psi_vqe = _normalize_state(psi_vqe)
    exact_filtered_energy = float(
        ns["exact_ground_energy_sector"](
            h_poly,
            num_sites=int(num_sites),
            num_particles=num_particles,
            indexing=ordering,
        )
    )

    payload = {
        "success": True,
        "method": method_name,
        "energy": float(result.energy),
        "ansatz": str(ansatz_name_s),
        "parameterization": "layerwise",
        "exact_filtered_energy": float(exact_filtered_energy),
        "best_restart": int(getattr(result, "best_restart", 0)),
        "nfev": int(getattr(result, "nfev", 0)),
        "nit": int(getattr(result, "nit", 0)),
        "message": str(getattr(result, "message", "")),
        "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
        "num_parameters": int(ansatz.num_parameters),
        "reps": int(reps),
        "optimizer_method": str(method),
        "optimal_point": [float(x) for x in theta.tolist()],
        "hf_bitstring_qn_to_q0": hf_bits,
    }
    if ansatz_name_s == "hva" and int(num_sites) == 2:
        delta_vs_exact = float(payload["energy"]) - float(exact_filtered_energy)
        if np.isfinite(delta_vs_exact) and delta_vs_exact > 1e-3:
            payload["warning"] = (
                "L=2 strict layer-wise HVA may be expressivity-limited under shared-parameter tying; "
                "large positive gap vs filtered-sector exact energy can be expected."
            )
            payload["warning_delta_vs_exact_filtered"] = float(delta_vs_exact)
            _ai_log(
                "hardcoded_vqe_layerwise_hva_l2_warning",
                L=int(num_sites),
                energy=float(payload["energy"]),
                exact_filtered_energy=float(exact_filtered_energy),
                delta_vs_exact_filtered=float(delta_vs_exact),
            )
    _ai_log(
        "hardcoded_vqe_done",
        L=int(num_sites),
        ansatz=str(ansatz_name_s),
        success=True,
        energy=float(result.energy),
        exact_filtered_energy=float(exact_filtered_energy),
        best_restart=int(getattr(result, "best_restart", 0)),
        nfev=int(getattr(result, "nfev", 0)),
        nit=int(getattr(result, "nit", 0)),
        elapsed_sec=round(time.perf_counter() - t0, 6),
    )
    return payload, psi_vqe


def _run_qpe_adapter_qiskit(
    *,
    coeff_map_exyz: dict[str, complex],
    psi_init: np.ndarray,
    eval_qubits: int,
    shots: int,
    seed: int,
) -> dict[str, Any]:
    """Temporary QPE adapter using minimal Qiskit calls.

    TODO: replace this adapter with a fully hardcoded QPE implementation.
    """

    t0 = time.perf_counter()
    _ai_log(
        "hardcoded_qpe_start",
        eval_qubits=int(eval_qubits),
        shots=int(shots),
        seed=int(seed),
        num_qubits=int(round(math.log2(psi_init.size))),
    )

    def _finish(payload: dict[str, Any]) -> dict[str, Any]:
        _ai_log(
            "hardcoded_qpe_done",
            success=bool(payload.get("success", False)),
            method=str(payload.get("method", "")),
            energy_estimate=payload.get("energy_estimate"),
            elapsed_sec=round(time.perf_counter() - t0, 6),
        )
        return payload

    # Isolated Qiskit-only section for temporary QPE support.
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.primitives import StatevectorSampler
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.synthesis import SuzukiTrotter
        from qiskit_algorithms import PhaseEstimation
        from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
    except Exception as exc:
        return _finish({
            "success": False,
            "method": "qiskit_import_failed",
            "energy_estimate": None,
            "phase": None,
            "error": str(exc),
        })

    terms_ixyz = [(_to_ixyz(lbl), complex(coeff)) for lbl, coeff in coeff_map_exyz.items() if abs(coeff) > 1e-12]
    if not terms_ixyz:
        terms_ixyz = [("I" * int(round(math.log2(psi_init.size))), 0.0)]

    h_op = SparsePauliOp.from_list(terms_ixyz).simplify(atol=1e-12)
    bound = float(sum(abs(float(np.real(coeff))) for _lbl, coeff in terms_ixyz))
    bound = max(bound, 1e-9)
    evo_time = float(np.pi / bound)

    # Large qubit counts can make canonical QPE prohibitively expensive in CI-like runs.
    # Use a deterministic Qiskit eigensolver fallback while keeping output schema aligned.
    if h_op.num_qubits >= 8:
        np_solver = NumPyMinimumEigensolver()
        res = np_solver.compute_minimum_eigenvalue(h_op)
        return _finish({
            "success": True,
            "method": "qiskit_numpy_minimum_eigensolver_fastpath_large_n",
            "energy_estimate": float(np.real(res.eigenvalue)),
            "phase": None,
            "bound": bound,
            "evolution_time": evo_time,
            "num_evaluation_qubits": int(eval_qubits),
            "shots": int(shots),
        })

    try:
        prep = QuantumCircuit(h_op.num_qubits)
        prep.initialize(np.asarray(psi_init, dtype=complex), list(range(h_op.num_qubits)))

        evo = PauliEvolutionGate(
            h_op,
            time=evo_time,
            synthesis=SuzukiTrotter(order=2, reps=1, preserve_order=True),
        )
        unitary = QuantumCircuit(h_op.num_qubits)
        unitary.append(evo, range(h_op.num_qubits))

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
        try:
            np_solver = NumPyMinimumEigensolver()
            res = np_solver.compute_minimum_eigenvalue(h_op)
            energy = float(np.real(res.eigenvalue))
            return _finish({
                "success": True,
                "method": "qiskit_numpy_minimum_eigensolver_fallback",
                "energy_estimate": energy,
                "phase": None,
                "bound": bound,
                "evolution_time": evo_time,
                "num_evaluation_qubits": int(eval_qubits),
                "shots": int(shots),
                "warning": str(exc),
            })
        except Exception as fallback_exc:
            return _finish({
                "success": False,
                "method": "qpe_failed",
                "energy_estimate": None,
                "phase": None,
                "bound": bound,
                "evolution_time": evo_time,
                "num_evaluation_qubits": int(eval_qubits),
                "shots": int(shots),
                "error": str(exc),
                "fallback_error": str(fallback_exc),
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
        # Accumulate the eigenvalue product for each basis state.
        eig = np.ones(dim, dtype=np.float64)
        for q in range(nq):
            if label[nq - 1 - q] == "z":
                # Z_q eigenvalue: +1 if bit q is 0, −1 if bit q is 1
                eig *= 1.0 - 2.0 * ((idx >> q) & 1).astype(np.float64)
        diag += coeff * eig
    return diag


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
    piecewise-constant approximation: each sub-interval [t_k, t_{k+1}] of
    width Δt = time_value / trotter_steps is replaced by the exact
    exponential of H evaluated at a single representative time t_k.

    The order depends on how t_k is chosen (``time_sampling``):

    * ``"midpoint"`` (default): t_k = t₀ + (k + ½)Δt.
      **Exponential midpoint / Magnus-2 integrator — second order O(Δt²).**
      For a non-autonomous linear ODE, this equals the one-term Magnus
      expansion with the midpoint quadrature and is equivalent to the
      classical exponential midpoint rule.  The global error scales as
      O(Δt²) = O(1/N²).
    * ``"left"``: t_k = t₀ + k Δt.  First order O(Δt).  Use only for
      diagnostics or explicit order-convergence studies.
    * ``"right"``: t_k = t₀ + (k+1)Δt.  First order O(Δt).

    The JSON metadata records ``reference_method`` and
    ``reference_steps_multiplier`` so downstream readers know which
    approximation was used.

    The time-sampling rule is applied consistently to both this reference
    propagator and the Trotter integrator so that fidelity comparisons are
    apples-to-apples at the *same* discretization.  When
    ``--exact-steps-multiplier M > 1``, this function is called with
    M × trotter_steps to provide a finer (higher-quality) reference while the
    Trotter circuit runs at the original trotter_steps.

    Sparse / expm_multiply optimisation
    ------------------------------------
    For Hilbert-space dimension ``dim >= _EXPM_SPARSE_MIN_DIM`` (currently 64,
    i.e., L ≥ 3) this function uses ``scipy.sparse.linalg.expm_multiply``
    instead of ``scipy.linalg.expm``.

    * **H_static** is pre-converted to a ``scipy.sparse.csc_matrix`` *once*
      before the time-step loop, exploiting its O(L) non-zero structure.
    * **H_drive(t)** — which for the density drive is always a sum of Z-type
      (diagonal) Paulis — is stored as a 1-D diagonal vector and converted to
      a ``scipy.sparse.diags`` matrix per step.  No d × d dense allocation is
      needed.
    * ``expm_multiply`` applies the Al-Mohy–Higham algorithm: it approximates
      exp(A) b via a Krylov subspace sequence without ever forming the full
      matrix exponential.  Time cost is O(d · nnz(H) · p) where p is the
      polynomial degree (typically 3–55 for physics-scale problems), compared
      with O(d³) for dense expm.

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
            # Build the drive as a dense matrix then convert to sparse.
            if filtered_drive:
                h_drive_dense = _build_hamiltonian_matrix(filtered_drive)
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
                h_drive = _build_hamiltonian_matrix(filtered_drive)
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
    psi0_ansatz_trot: np.ndarray,
    psi0_exact_ref: np.ndarray,
    fidelity_subspace_basis_v0: np.ndarray,
    fidelity_subspace_energy_tol: float,
    hmat: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
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

    nq = 2 * int(num_sites)
    evals, evecs = np.linalg.eigh(hmat)
    evecs_dag = np.conjugate(evecs).T

    compiled = {lbl: _compile_pauli_action(lbl, nq) for lbl in ordered_labels_exyz}
    times = np.linspace(0.0, float(t_final), int(num_times))
    n_times = int(times.size)
    stride = max(1, n_times // 20)
    t0 = time.perf_counter()
    basis_v0 = np.asarray(fidelity_subspace_basis_v0, dtype=complex)
    if basis_v0.ndim != 2 or basis_v0.shape[0] != psi0_ansatz_trot.size:
        raise ValueError("fidelity_subspace_basis_v0 must have shape (dim, k) with matching dim.")
    if basis_v0.shape[1] <= 0:
        raise ValueError("fidelity_subspace_basis_v0 must contain at least one basis vector.")

    has_drive = drive_coeff_provider_exyz is not None

    # When drive is enabled the reference propagator may use a finer step count
    # to improve its quality independently of the Trotter discretization.
    reference_steps = int(trotter_steps) * max(1, int(exact_steps_multiplier))
    static_basis_eig = evecs_dag @ basis_v0

    _ai_log(
        "hardcoded_trajectory_start",
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

        # --- exact / reference propagation (filtered-sector GS branch) ---
        if not has_drive:
            # Time-independent: eigendecomposition shortcut (exact to machine
            # precision — does not depend on exact_steps_multiplier).
            psi_exact = evecs @ (np.exp(-1j * evals * t) * (evecs_dag @ psi0_exact_ref))
            psi_exact = _normalize_state(psi_exact)
        else:
            # Time-dependent: piecewise-constant matrix-exponential reference
            # (exponential midpoint / Magnus-2 when time_sampling="midpoint").
            # reference_steps = exact_steps_multiplier * trotter_steps so that
            # this can be made arbitrarily accurate independently of the
            # Trotter circuit discretization.
            psi_exact = _evolve_piecewise_exact(
                psi0=psi0_exact_ref,
                hmat_static=hmat,
                drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                time_value=t,
                trotter_steps=reference_steps,
                t0=float(drive_t0),
                time_sampling=str(drive_time_sampling),
            )

        # --- exact propagation from the same ansatz initial state ---
        if not has_drive:
            psi_exact_ansatz = evecs @ (np.exp(-1j * evals * t) * (evecs_dag @ psi0_ansatz_trot))
            psi_exact_ansatz = _normalize_state(psi_exact_ansatz)
        else:
            psi_exact_ansatz = _evolve_piecewise_exact(
                psi0=psi0_ansatz_trot,
                hmat_static=hmat,
                drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                time_value=t,
                trotter_steps=reference_steps,
                t0=float(drive_t0),
                time_sampling=str(drive_time_sampling),
            )

        psi_trot = _evolve_trotter_suzuki2_absolute(
            psi0_ansatz_trot,
            ordered_labels_exyz,
            coeff_map_exyz,
            compiled,
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
                "hardcoded_fidelity_subspace_rank_reduced",
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
        n_up_exact_ansatz_site, n_dn_exact_ansatz_site, doublon_exact_ansatz = _site_resolved_number_observables(
            psi_exact_ansatz,
            num_sites,
            ordering,
        )
        n_up_trot_site, n_dn_trot_site, doublon_trot = _site_resolved_number_observables(
            psi_trot,
            num_sites,
            ordering,
        )
        n_exact_site = n_up_exact_site + n_dn_exact_site
        n_exact_ansatz_site = n_up_exact_ansatz_site + n_dn_exact_ansatz_site
        n_trot_site = n_up_trot_site + n_dn_trot_site
        n_up_exact = float(n_up_exact_site[0]) if n_up_exact_site.size > 0 else float("nan")
        n_dn_exact = float(n_dn_exact_site[0]) if n_dn_exact_site.size > 0 else float("nan")
        n_up_exact_ansatz = float(n_up_exact_ansatz_site[0]) if n_up_exact_ansatz_site.size > 0 else float("nan")
        n_dn_exact_ansatz = float(n_dn_exact_ansatz_site[0]) if n_dn_exact_ansatz_site.size > 0 else float("nan")
        n_up_trot = float(n_up_trot_site[0]) if n_up_trot_site.size > 0 else float("nan")
        n_dn_trot = float(n_dn_trot_site[0]) if n_dn_trot_site.size > 0 else float("nan")
        energy_static_exact = _expectation_hamiltonian(psi_exact, hmat)
        energy_static_exact_ansatz = _expectation_hamiltonian(psi_exact_ansatz, hmat)
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
            energy_total_exact_ansatz = _expectation_hamiltonian(psi_exact_ansatz, hmat_total_t)
            energy_total_trotter = _expectation_hamiltonian(psi_trot, hmat_total_t)
        else:
            energy_total_exact = energy_static_exact
            energy_total_exact_ansatz = energy_static_exact_ansatz
            energy_total_trotter = energy_static_trotter

        rows.append(
            {
                "time": t,
                "fidelity": fidelity,
                "energy_static_exact": energy_static_exact,
                "energy_static_exact_ansatz": energy_static_exact_ansatz,
                "energy_static_trotter": energy_static_trotter,
                "energy_total_exact": energy_total_exact,
                "energy_total_exact_ansatz": energy_total_exact_ansatz,
                "energy_total_trotter": energy_total_trotter,
                "n_up_site0_exact": n_up_exact,
                "n_up_site0_exact_ansatz": n_up_exact_ansatz,
                "n_up_site0_trotter": n_up_trot,
                "n_dn_site0_exact": n_dn_exact,
                "n_dn_site0_exact_ansatz": n_dn_exact_ansatz,
                "n_dn_site0_trotter": n_dn_trot,
                "n_site_exact": [float(x) for x in n_exact_site.tolist()],
                "n_site_exact_ansatz": [float(x) for x in n_exact_ansatz_site.tolist()],
                "n_site_trotter": [float(x) for x in n_trot_site.tolist()],
                "staggered_exact": _staggered_order(n_exact_site),
                "staggered_exact_ansatz": _staggered_order(n_exact_ansatz_site),
                "staggered_trotter": _staggered_order(n_trot_site),
                "doublon_exact": doublon_exact,
                "doublon_exact_ansatz": doublon_exact_ansatz,
                "doublon_trotter": doublon_trot,
                "doublon_avg_exact": float(doublon_exact / float(num_sites)),
                "doublon_avg_exact_ansatz": float(doublon_exact_ansatz / float(num_sites)),
                "doublon_avg_trotter": float(doublon_trot / float(num_sites)),
                "n_up_site_exact": [float(x) for x in n_up_exact_site.tolist()],
                "n_up_site_exact_ansatz": [float(x) for x in n_up_exact_ansatz_site.tolist()],
                "n_up_site_trotter": [float(x) for x in n_up_trot_site.tolist()],
                "n_dn_site_exact": [float(x) for x in n_dn_exact_site.tolist()],
                "n_dn_site_exact_ansatz": [float(x) for x in n_dn_exact_ansatz_site.tolist()],
                "n_dn_site_trotter": [float(x) for x in n_dn_trot_site.tolist()],
                "norm_before_renorm": norm_before,
            }
        )
        exact_states.append(psi_exact)
        if idx == 0 or idx == n_times - 1 or ((idx + 1) % stride == 0):
            _ai_log(
                "hardcoded_trajectory_progress",
                step=int(idx + 1),
                total_steps=n_times,
                frac=round(float((idx + 1) / n_times), 6),
                time=float(t),
                subspace_fidelity=float(fidelity),
                elapsed_sec=round(time.perf_counter() - t0, 6),
            )

    _ai_log(
        "hardcoded_trajectory_done",
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
        "Script: pipelines/hardcoded_hubbard_pipeline.py",
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
    if len(traj) == 0:
        raise ValueError("Cannot render pipeline PDF: trajectory is empty.")
    times = np.array([float(r["time"]) for r in traj], dtype=float)
    markevery = max(1, times.size // 25)

    def arr(key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in traj], dtype=float)

    def arr_optional(key: str, fallback: np.ndarray | None = None) -> np.ndarray:
        vals: list[float] = []
        for row in traj:
            if key in row:
                vals.append(float(row[key]))
            elif fallback is not None:
                vals.append(float(fallback[len(vals)]))
            else:
                vals.append(float("nan"))
        return np.array(vals, dtype=float)

    def mat(key: str) -> np.ndarray:
        out: list[list[float]] = []
        for i, row in enumerate(traj):
            if key not in row:
                raise KeyError(f"Missing key '{key}' at trajectory row {i}.")
            raw = row[key]
            if not isinstance(raw, list):
                raise TypeError(f"Expected list-valued key '{key}' at row {i}.")
            out.append([float(x) for x in raw])
        return np.array(out, dtype=float)

    def mat_optional(key: str, fallback: np.ndarray) -> np.ndarray:
        if key not in traj[0]:
            return np.array(fallback, copy=True)
        out: list[list[float]] = []
        for i, row in enumerate(traj):
            raw = row.get(key)
            if raw is None:
                out.append([float(x) for x in fallback[i].tolist()])
                continue
            if not isinstance(raw, list):
                raise TypeError(f"Expected list-valued key '{key}' at row {i}.")
            out.append([float(x) for x in raw])
        return np.array(out, dtype=float)

    def _plot_density_surface(
        ax: Any,
        data: np.ndarray,
        *,
        title: str,
        zlim: tuple[float, float],
        cmap: str,
    ) -> None:
        sites = np.arange(data.shape[1], dtype=float)
        t_grid, s_grid = np.meshgrid(times, sites, indexing="xy")
        ax.plot_surface(
            t_grid,
            s_grid,
            data.T,
            cmap=cmap,
            linewidth=0.0,
            antialiased=True,
            alpha=0.95,
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time")
        ax.set_ylabel("Site")
        ax.set_zlabel("Density")
        ax.set_zlim(float(zlim[0]), float(zlim[1]))
        ax.view_init(elev=25, azim=-60)

    def _plot_lane_3d(
        ax: Any,
        *,
        series: list[np.ndarray],
        labels: list[str],
        colors: list[str],
        title: str,
        zlabel: str,
    ) -> None:
        for lane, (vals, lbl, col) in enumerate(zip(series, labels, colors)):
            lane_vals = np.full_like(times, float(lane), dtype=float)
            ax.plot(times, lane_vals, vals, color=col, linewidth=1.8, label=lbl)
            ax.scatter(
                times[::max(1, len(times) // 25)],
                lane_vals[::max(1, len(times) // 25)],
                vals[::max(1, len(times) // 25)],
                color=col,
                s=8,
            )
        ax.set_yticks([0.0, 1.0, 2.0])
        ax.set_yticklabels(["Exact GS", "Exact Ansatz", "Trotter"])
        ax.set_xlabel("Time")
        ax.set_ylabel("Branch")
        ax.set_zlabel(zlabel)
        ax.set_title(title, fontsize=9)
        ax.view_init(elev=22, azim=-58)
        ax.legend(fontsize=7, loc="upper left")

    fid = arr("fidelity")
    e_exact = arr("energy_static_exact")
    e_exact_ans = arr_optional("energy_static_exact_ansatz", fallback=e_exact)
    e_trot = arr("energy_static_trotter")
    nu_exact = arr("n_up_site0_exact")
    nu_exact_ans = arr_optional("n_up_site0_exact_ansatz", fallback=nu_exact)
    nu_trot = arr("n_up_site0_trotter")
    nd_exact = arr("n_dn_site0_exact")
    nd_exact_ans = arr_optional("n_dn_site0_exact_ansatz", fallback=nd_exact)
    nd_trot = arr("n_dn_site0_trotter")
    d_exact = arr("doublon_exact")
    d_exact_ans = arr_optional("doublon_exact_ansatz", fallback=d_exact)
    d_trot = arr("doublon_trotter")
    stg_exact = arr("staggered_exact")
    stg_exact_ans = arr_optional("staggered_exact_ansatz", fallback=stg_exact)
    stg_trot = arr("staggered_trotter")

    n_site_exact = mat("n_site_exact")
    n_site_exact_ans = mat_optional("n_site_exact_ansatz", n_site_exact)
    n_site_trot = mat("n_site_trotter")
    n_up_site_exact = mat_optional("n_up_site_exact", n_site_exact * 0.5)
    n_up_site_exact_ans = mat_optional("n_up_site_exact_ansatz", n_site_exact_ans * 0.5)
    n_up_site_trot = mat_optional("n_up_site_trotter", n_site_trot * 0.5)
    n_dn_site_exact = mat_optional("n_dn_site_exact", n_site_exact * 0.5)
    n_dn_site_exact_ans = mat_optional("n_dn_site_exact_ansatz", n_site_exact_ans * 0.5)
    n_dn_site_trot = mat_optional("n_dn_site_trotter", n_site_trot * 0.5)

    err_n_trot_vs_exact_ans = np.abs(n_site_trot - n_site_exact_ans)
    err_n_exact_ans_vs_exact_gs = np.abs(n_site_exact_ans - n_site_exact)
    err_n_trot_vs_exact_gs = np.abs(n_site_trot - n_site_exact)

    err_scalar_rows = np.vstack(
        [
            np.abs(e_trot - e_exact_ans),
            np.abs(e_exact_ans - e_exact),
            np.abs(d_trot - d_exact_ans),
            np.abs(d_exact_ans - d_exact),
            np.abs(stg_trot - stg_exact_ans),
            np.abs(stg_exact_ans - stg_exact),
        ]
    )
    err_scalar_labels = [
        "|E_trot - E_exact_ans|",
        "|E_exact_ans - E_exact_gs|",
        "|D_trot - D_exact_ans|",
        "|D_exact_ans - D_exact_gs|",
        "|S_trot - S_exact_ans|",
        "|S_exact_ans - S_exact_gs|",
    ]

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
    ansatz_label = str(payload.get("vqe", {}).get("ansatz", settings.get("vqe_ansatz", "unknown"))).strip().lower()
    vqe_method = payload.get("vqe", {}).get("method", "unknown")
    run_mode = "drive-enabled" if isinstance(settings.get("drive"), dict) else "static"
    _ = run_command  # command text is preserved in CLI logs; PDF starts with summary

    with PdfPages(str(pdf_path)) as pdf:
        summary_lines = [
            f"Hardcoded Hubbard Summary (L={settings.get('L')})",
            "",
            "Ansatz Used:",
            f"  - hardcoded ansatz: {ansatz_label}",
            f"  - vqe method: {vqe_method}",
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

        # ------------------------------------------------------------------
        # Front matter: 3D pages first (non-redundant)
        # ------------------------------------------------------------------
        dens_all = np.concatenate([n_site_exact.reshape(-1), n_site_exact_ans.reshape(-1), n_site_trot.reshape(-1)])
        dens_zlim = (float(np.min(dens_all)), float(np.max(dens_all)))
        if abs(dens_zlim[1] - dens_zlim[0]) < 1e-12:
            dens_zlim = (dens_zlim[0] - 1e-6, dens_zlim[1] + 1e-6)

        fig3d_n = plt.figure(figsize=(14.0, 8.5))
        axn0 = fig3d_n.add_subplot(1, 3, 1, projection="3d")
        axn1 = fig3d_n.add_subplot(1, 3, 2, projection="3d")
        axn2 = fig3d_n.add_subplot(1, 3, 3, projection="3d")
        _plot_density_surface(axn0, n_site_exact, title="Exact GS Filtered: n(site,t)", zlim=dens_zlim, cmap="Blues")
        _plot_density_surface(axn1, n_site_exact_ans, title="Exact Ansatz: n(site,t)", zlim=dens_zlim, cmap="Greens")
        _plot_density_surface(axn2, n_site_trot, title="Trotter Ansatz: n(site,t)", zlim=dens_zlim, cmap="Oranges")
        fig3d_n.suptitle(f"L={payload['settings']['L']} 3D Densities (Total n)", fontsize=13)
        fig3d_n.tight_layout(rect=(0.0, 0.02, 1.0, 0.93))
        pdf.savefig(fig3d_n)
        plt.close(fig3d_n)

        up_all = np.concatenate([n_up_site_exact.reshape(-1), n_up_site_exact_ans.reshape(-1), n_up_site_trot.reshape(-1)])
        up_zlim = (float(np.min(up_all)), float(np.max(up_all)))
        if abs(up_zlim[1] - up_zlim[0]) < 1e-12:
            up_zlim = (up_zlim[0] - 1e-6, up_zlim[1] + 1e-6)
        fig3d_up = plt.figure(figsize=(14.0, 8.5))
        axu0 = fig3d_up.add_subplot(1, 3, 1, projection="3d")
        axu1 = fig3d_up.add_subplot(1, 3, 2, projection="3d")
        axu2 = fig3d_up.add_subplot(1, 3, 3, projection="3d")
        _plot_density_surface(axu0, n_up_site_exact, title="Exact GS Filtered: n_up(site,t)", zlim=up_zlim, cmap="PuBu")
        _plot_density_surface(axu1, n_up_site_exact_ans, title="Exact Ansatz: n_up(site,t)", zlim=up_zlim, cmap="YlGn")
        _plot_density_surface(axu2, n_up_site_trot, title="Trotter Ansatz: n_up(site,t)", zlim=up_zlim, cmap="YlOrBr")
        fig3d_up.suptitle(f"L={payload['settings']['L']} 3D Densities (Spin-Up)", fontsize=13)
        fig3d_up.tight_layout(rect=(0.0, 0.02, 1.0, 0.93))
        pdf.savefig(fig3d_up)
        plt.close(fig3d_up)

        dn_all = np.concatenate([n_dn_site_exact.reshape(-1), n_dn_site_exact_ans.reshape(-1), n_dn_site_trot.reshape(-1)])
        dn_zlim = (float(np.min(dn_all)), float(np.max(dn_all)))
        if abs(dn_zlim[1] - dn_zlim[0]) < 1e-12:
            dn_zlim = (dn_zlim[0] - 1e-6, dn_zlim[1] + 1e-6)
        fig3d_dn = plt.figure(figsize=(14.0, 8.5))
        axd0 = fig3d_dn.add_subplot(1, 3, 1, projection="3d")
        axd1 = fig3d_dn.add_subplot(1, 3, 2, projection="3d")
        axd2 = fig3d_dn.add_subplot(1, 3, 3, projection="3d")
        _plot_density_surface(axd0, n_dn_site_exact, title="Exact GS Filtered: n_dn(site,t)", zlim=dn_zlim, cmap="PuBu")
        _plot_density_surface(axd1, n_dn_site_exact_ans, title="Exact Ansatz: n_dn(site,t)", zlim=dn_zlim, cmap="YlGn")
        _plot_density_surface(axd2, n_dn_site_trot, title="Trotter Ansatz: n_dn(site,t)", zlim=dn_zlim, cmap="YlOrBr")
        fig3d_dn.suptitle(f"L={payload['settings']['L']} 3D Densities (Spin-Down)", fontsize=13)
        fig3d_dn.tight_layout(rect=(0.0, 0.02, 1.0, 0.93))
        pdf.savefig(fig3d_dn)
        plt.close(fig3d_dn)

        fig3d_scalars = plt.figure(figsize=(14.0, 8.5))
        axe0 = fig3d_scalars.add_subplot(2, 2, 1, projection="3d")
        axe1 = fig3d_scalars.add_subplot(2, 2, 2, projection="3d")
        axe2 = fig3d_scalars.add_subplot(2, 2, 3, projection="3d")
        axe3 = fig3d_scalars.add_subplot(2, 2, 4, projection="3d")
        labels_3 = ["Exact GS Filtered", "Exact Ansatz", "Trotter Ansatz"]
        colors_3 = ["#1f77b4", "#2ca02c", "#d62728"]
        _plot_lane_3d(
            axe0,
            series=[arr("energy_total_exact"), arr_optional("energy_total_exact_ansatz", arr("energy_total_exact")), arr("energy_total_trotter")],
            labels=labels_3,
            colors=colors_3,
            title="3D Lanes: Total Energy",
            zlabel="Energy",
        )
        _plot_lane_3d(
            axe1,
            series=[arr("energy_static_exact"), arr_optional("energy_static_exact_ansatz", arr("energy_static_exact")), arr("energy_static_trotter")],
            labels=labels_3,
            colors=colors_3,
            title="3D Lanes: Static Energy",
            zlabel="Energy",
        )
        _plot_lane_3d(
            axe2,
            series=[arr("doublon_exact"), arr_optional("doublon_exact_ansatz", arr("doublon_exact")), arr("doublon_trotter")],
            labels=labels_3,
            colors=colors_3,
            title="3D Lanes: Doublon",
            zlabel="Doublon",
        )
        _plot_lane_3d(
            axe3,
            series=[arr("staggered_exact"), arr_optional("staggered_exact_ansatz", arr("staggered_exact")), arr("staggered_trotter")],
            labels=labels_3,
            colors=colors_3,
            title="3D Lanes: Staggered Order",
            zlabel="Order",
        )
        fig3d_scalars.suptitle(f"L={payload['settings']['L']} 3D Scalar Observables (Three Evolutions)", fontsize=13)
        fig3d_scalars.tight_layout(rect=(0.0, 0.02, 1.0, 0.94))
        pdf.savefig(fig3d_scalars)
        plt.close(fig3d_scalars)

        fig_err = plt.figure(figsize=(14.0, 8.5))
        axh0 = fig_err.add_subplot(1, 3, 1)
        axh1 = fig_err.add_subplot(1, 3, 2)
        axh2 = fig_err.add_subplot(1, 3, 3)
        heatmaps = [
            (err_n_trot_vs_exact_ans, "|n_trot - n_exact_ansatz|"),
            (err_n_exact_ans_vs_exact_gs, "|n_exact_ansatz - n_exact_gs|"),
            (err_n_trot_vs_exact_gs, "|n_trot - n_exact_gs|"),
        ]
        for axh, (hmat_err, title) in zip((axh0, axh1, axh2), heatmaps):
            im = axh.imshow(
                hmat_err.T,
                origin="lower",
                aspect="auto",
                extent=(float(times[0]), float(times[-1]), -0.5, float(hmat_err.shape[1] - 0.5)),
                cmap="magma",
            )
            axh.set_title(title)
            axh.set_xlabel("Time")
            axh.set_ylabel("Site")
            plt.colorbar(im, ax=axh, fraction=0.046, pad=0.04, label="Absolute Error")
        fig_err.suptitle(f"L={payload['settings']['L']} Error Heatmaps (Absolute Errors)", fontsize=13)
        fig_err.tight_layout(rect=(0.0, 0.02, 1.0, 0.94))
        pdf.savefig(fig_err)
        plt.close(fig_err)

        fig_err_scalar, axes_scalar = plt.subplots(1, 1, figsize=(14.0, 8.5))
        im_scalar = axes_scalar.imshow(
            err_scalar_rows,
            origin="lower",
            aspect="auto",
            extent=(float(times[0]), float(times[-1]), -0.5, float(err_scalar_rows.shape[0] - 0.5)),
            cmap="inferno",
        )
        axes_scalar.set_yticks(np.arange(len(err_scalar_labels)))
        axes_scalar.set_yticklabels(err_scalar_labels, fontsize=9)
        axes_scalar.set_xlabel("Time")
        axes_scalar.set_title("Absolute Error Heatmap (Scalar Observables)")
        plt.colorbar(im_scalar, ax=axes_scalar, fraction=0.03, pad=0.02, label="Absolute Error")
        fig_err_scalar.suptitle(f"L={payload['settings']['L']} Scalar Error Heatmap", fontsize=13)
        fig_err_scalar.tight_layout(rect=(0.0, 0.02, 1.0, 0.94))
        pdf.savefig(fig_err_scalar)
        plt.close(fig_err_scalar)

        # ------------------------------------------------------------------
        # Main body: drive diagnostics + compact 2D diagnostics
        # ------------------------------------------------------------------
        drive_cfg = payload.get("settings", {}).get("drive")
        if isinstance(drive_cfg, dict) and times.size >= 2:
            A = float(drive_cfg.get("A", 0.0))
            omega = float(drive_cfg.get("omega", 1.0))
            tbar = float(drive_cfg.get("tbar", 1.0))
            phi = float(drive_cfg.get("phi", 0.0))
            t0 = float(drive_cfg.get("t0", 0.0))

            waveform = evaluate_drive_waveform(
                times,
                {"omega": omega, "tbar": tbar, "phi": phi, "t0": t0},
                amplitude=A,
            )
            t_phys = times + t0
            if tbar > 0.0:
                envelope = abs(A) * np.exp(-(t_phys * t_phys) / (2.0 * tbar * tbar))
            else:
                envelope = np.zeros_like(t_phys, dtype=float)

            figd, (axw, axf) = plt.subplots(1, 2, figsize=(11.0, 8.5))
            axw.plot(times, waveform, color="#1f77b4", linewidth=1.5, label="f(t)")
            axw.plot(times, envelope, color="#d62728", linestyle="--", linewidth=1.1, label="+envelope")
            axw.plot(times, -envelope, color="#d62728", linestyle="--", linewidth=1.1, label="-envelope")
            axw.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.8)
            axw.set_title("Drive Waveform and Gaussian Envelope")
            axw.set_xlabel("Time")
            axw.set_ylabel("f(t)")
            axw.grid(alpha=0.25)
            axw.legend(fontsize=8, loc="best")

            dt = np.diff(times)
            dt_mean = float(np.mean(dt)) if dt.size > 0 else 0.0
            if dt_mean > 0.0 and waveform.size > 1:
                centered = waveform - float(np.mean(waveform))
                windowed = centered * np.hanning(centered.size)
                fft_vals = np.fft.rfft(windowed)
                freqs = np.fft.rfftfreq(windowed.size, d=dt_mean)
                omega_axis = 2.0 * np.pi * freqs
                mag = np.abs(fft_vals)
                if mag.size > 0:
                    mag = mag / (float(np.max(mag)) + 1e-15)
                axf.plot(omega_axis, mag, color="#2ca02c", linewidth=1.4, label="|FFT(windowed f(t))|")
                axf.axvline(abs(omega), color="#d62728", linestyle="--", linewidth=1.0, label=f"drive omega={omega:.3f}")
                axf.set_title("Drive Spectrum (Normalized Magnitude)")
                axf.set_xlabel("Angular frequency")
                axf.set_ylabel("Normalized magnitude")
                axf.grid(alpha=0.25)
                axf.legend(fontsize=8, loc="best")
            else:
                axf.text(0.5, 0.5, "Insufficient time grid for FFT", ha="center", va="center", transform=axf.transAxes)
                axf.set_title("Drive Spectrum")
                axf.set_axis_off()

            figd.suptitle(
                f"Drive Diagnostics: A={A}, omega={omega}, tbar={tbar}, phi={phi}, t0={t0}",
                fontsize=11,
            )
            figd.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
            pdf.savefig(figd)
            plt.close(figd)

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

        ax01.plot(times, e_exact, label="Exact GS filtered (static)", color="#111111", linewidth=2.0, marker="s", markersize=3, markevery=markevery)
        ax01.plot(times, e_exact_ans, label="Exact ansatz init (static)", color="#2ca02c", linewidth=1.4, marker="D", markersize=3, markevery=markevery)
        ax01.plot(times, e_trot, label="Trotter ansatz init (static)", color="#d62728", linestyle="--", linewidth=1.4, marker="^", markersize=3, markevery=markevery)

        # --- optional total-energy overlay (when drive active and differs) ---
        e_total_exact = arr("energy_total_exact")
        e_total_exact_ans = arr_optional("energy_total_exact_ansatz", fallback=e_total_exact)
        e_total_trot = arr("energy_total_trotter")
        if not (np.allclose(e_total_exact, e_exact, atol=1e-14) and
                np.allclose(e_total_trot, e_trot, atol=1e-14)):
            ax01.plot(times, e_total_exact, label="Exact GS filtered (total)", color="#17becf",
                      linewidth=1.6, marker="D", markersize=2.5, markevery=markevery, alpha=0.8)
            ax01.plot(times, e_total_exact_ans, label="Exact ansatz init (total)", color="#1f77b4",
                      linewidth=1.3, marker="o", markersize=2.5, markevery=markevery, alpha=0.8)
            ax01.plot(times, e_total_trot, label="Trotter ansatz init (total)", color="#ff7f0e",
                      linestyle="--", linewidth=1.2, marker="<", markersize=2.5, markevery=markevery, alpha=0.8)

        ax01.set_title("Energy")
        ax01.grid(alpha=0.25)
        ax01.legend(fontsize=7)

        ax10.plot(times, nu_exact, label="n_up0 exact GS", color="#17becf", linewidth=1.8, marker="o", markersize=3, markevery=markevery)
        ax10.plot(times, nu_exact_ans, label="n_up0 exact ansatz", color="#2ca02c", linewidth=1.2, marker="D", markersize=3, markevery=markevery)
        ax10.plot(times, nu_trot, label="n_up0 trotter", color="#0f7f8b", linestyle="--", linewidth=1.2, marker="s", markersize=3, markevery=markevery)
        ax10.plot(times, nd_exact, label="n_dn0 exact GS", color="#9467bd", linewidth=1.8, marker="^", markersize=3, markevery=markevery)
        ax10.plot(times, nd_exact_ans, label="n_dn0 exact ansatz", color="#8c564b", linewidth=1.2, marker="X", markersize=3, markevery=markevery)
        ax10.plot(times, nd_trot, label="n_dn0 trotter", color="#6f4d8f", linestyle="--", linewidth=1.2, marker="v", markersize=3, markevery=markevery)
        ax10.set_title("Site-0 Occupations")
        ax10.set_xlabel("Time")
        ax10.grid(alpha=0.25)
        ax10.legend(fontsize=8)

        ax11.plot(times, d_exact, label="doublon exact GS", color="#8c564b", linewidth=1.8, marker="o", markersize=3, markevery=markevery)
        ax11.plot(times, d_exact_ans, label="doublon exact ansatz", color="#2ca02c", linewidth=1.2, marker="D", markersize=3, markevery=markevery)
        ax11.plot(times, d_trot, label="doublon trotter", color="#c251a1", linestyle="--", linewidth=1.2, marker="s", markersize=3, markevery=markevery)
        ax11.set_title("Total Doublon")
        ax11.set_xlabel("Time")
        ax11.grid(alpha=0.25)
        ax11.legend(fontsize=8)

        fig.suptitle(f"Hardcoded Hubbard Pipeline: L={payload['settings']['L']}", fontsize=14)
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        # Additional focused energy pages (requested): static-only and total-only.
        fig_energy_static, ax_energy_static = plt.subplots(1, 1, figsize=(11.0, 8.5))
        ax_energy_static.plot(times, e_exact, label="Exact GS filtered (static)", color="#111111", linewidth=2.0, marker="s", markersize=3, markevery=markevery)
        ax_energy_static.plot(times, e_exact_ans, label="Exact ansatz init (static)", color="#2ca02c", linewidth=1.4, marker="D", markersize=3, markevery=markevery)
        ax_energy_static.plot(times, e_trot, label="Trotter ansatz init (static)", color="#d62728", linestyle="--", linewidth=1.4, marker="^", markersize=3, markevery=markevery)
        ax_energy_static.set_title("Energy (Static Hamiltonian Only)")
        ax_energy_static.set_xlabel("Time")
        ax_energy_static.set_ylabel("Energy")
        ax_energy_static.grid(alpha=0.25)
        ax_energy_static.legend(fontsize=8)
        fig_energy_static.tight_layout(rect=(0.0, 0.02, 1.0, 0.98))
        pdf.savefig(fig_energy_static)
        plt.close(fig_energy_static)

        fig_energy_total, ax_energy_total = plt.subplots(1, 1, figsize=(11.0, 8.5))
        ax_energy_total.plot(times, e_total_exact, label="Exact GS filtered (total)", color="#17becf", linewidth=1.6, marker="D", markersize=2.5, markevery=markevery, alpha=0.8)
        ax_energy_total.plot(times, e_total_exact_ans, label="Exact ansatz init (total)", color="#1f77b4", linewidth=1.3, marker="o", markersize=2.5, markevery=markevery, alpha=0.8)
        ax_energy_total.plot(times, e_total_trot, label="Trotter ansatz init (total)", color="#ff7f0e", linestyle="--", linewidth=1.2, marker="<", markersize=2.5, markevery=markevery, alpha=0.8)
        ax_energy_total.set_title("Energy (Total Hamiltonian H(t) Only)")
        ax_energy_total.set_xlabel("Time")
        ax_energy_total.set_ylabel("Energy")
        ax_energy_total.grid(alpha=0.25)
        ax_energy_total.legend(fontsize=8)
        fig_energy_total.tight_layout(rect=(0.0, 0.02, 1.0, 0.98))
        pdf.savefig(fig_energy_total)
        plt.close(fig_energy_total)

        # ------------------------------------------------------------------
        # Appendix: redundant/supporting material
        # ------------------------------------------------------------------
        fig_appendix = plt.figure(figsize=(11.0, 8.5))
        ax_appendix = fig_appendix.add_subplot(111)
        ax_appendix.axis("off")
        appendix_lines = [
            f"Appendix — L={payload['settings']['L']}",
            "",
            "This section contains supporting/redundant views and metadata.",
            "Core non-redundant analysis appears earlier in the PDF.",
        ]
        ax_appendix.text(0.03, 0.95, "\n".join(appendix_lines), va="top", ha="left", family="monospace", fontsize=12)
        pdf.savefig(fig_appendix)
        plt.close(fig_appendix)

        filt_label = f"Exact (sector {sector_label})"
        figv, vx0 = plt.subplots(1, 1, figsize=(11.0, 8.5))
        vx0.bar([0, 1], [gs_exact_filtered, vqe_val], color=["#2166ac", "#2ca02c"], edgecolor="black", linewidth=0.4)
        vx0.set_xticks([0, 1])
        vx0.set_xticklabels([filt_label, "VQE"], fontsize=8)
        vx0.set_ylabel("Energy")
        err_vqe = abs(vqe_val - gs_exact_filtered) if (np.isfinite(vqe_val) and np.isfinite(gs_exact_filtered)) else np.nan
        err_text = (f"|VQE - Exact(filtered)| = {err_vqe:.3e}"
                    if np.isfinite(err_vqe) else "|VQE - Exact(filtered)| = N/A")
        vx0.set_title(f"VQE Energy vs Exact (filtered sector)\n{err_text}")
        vx0.grid(axis="y", alpha=0.25)

        figv.suptitle(
            "VQE optimises within the half-filled sector; exact (filtered) is the true sector ground state.\n"
            "Full-Hilbert exact energy is in the JSON text summary only.",
            fontsize=10,
        )
        figv.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))
        pdf.savefig(figv)
        plt.close(figv)

        lines = [
            "Hardcoded Hubbard pipeline summary",
            "",
            "Ansatz:",
            f"  - hardcoded ansatz: {ansatz_label}",
            f"  - vqe method: {vqe_method}",
            "",
            "Energy + Fidelity:",
            f"  - subspace_fidelity_at_t0: {float(fid[0]) if fid.size > 0 else None}",
            f"  - energy_t0_exact_gs: {float(e_exact[0]) if e_exact.size > 0 else None}",
            f"  - energy_t0_exact_ansatz: {float(e_exact_ans[0]) if e_exact_ans.size > 0 else None}",
            f"  - energy_t0_trotter: {float(e_trot[0]) if e_trot.size > 0 else None}",
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
    parser = argparse.ArgumentParser(description="Hardcoded-first Hubbard pipeline runner.")
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
    parser.add_argument("--term-order", choices=["native", "sorted"], default="sorted")

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
    parser.add_argument(
        "--vqe-ansatz",
        type=str,
        default="uccsd",
        choices=["uccsd", "hva"],
        help="Hardcoded VQE ansatz family (mapped to layer-wise implementations).",
    )
    parser.add_argument("--vqe-reps", type=int, default=2, help="Number of ansatz repetitions (layer depth).")
    parser.add_argument("--vqe-restarts", type=int, default=1)
    parser.add_argument("--vqe-seed", type=int, default=7)
    parser.add_argument("--vqe-maxiter", type=int, default=120)
    parser.add_argument(
        "--vqe-method",
        type=str,
        default="COBYLA",
        choices=["SLSQP", "COBYLA", "L-BFGS-B", "Powell", "Nelder-Mead"],
        help="SciPy optimizer used by hardcoded VQE.",
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
    _ai_log("hardcoded_main_start", settings=vars(args))
    run_command = _current_command_string()
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_json = args.output_json or (artifacts_dir / f"hardcoded_pipeline_L{args.L}.json")
    output_pdf = args.output_pdf or (artifacts_dir / f"hardcoded_pipeline_L{args.L}.pdf")

    h_poly = build_hubbard_hamiltonian(
        dims=int(args.L),
        t=float(args.t),
        U=float(args.u),
        v=float(args.dv),
        repr_mode="JW",
        indexing=str(args.ordering),
        pbc=(str(args.boundary) == "periodic"),
    )

    native_order, coeff_map_exyz = _collect_hardcoded_terms_exyz(h_poly)
    _ai_log("hardcoded_hamiltonian_built", L=int(args.L), num_terms=int(len(coeff_map_exyz)))
    if args.term_order == "native":
        ordered_labels_exyz = list(native_order)
    else:
        ordered_labels_exyz = sorted(coeff_map_exyz)

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
            "hardcoded_drive_built",
            L=int(args.L),
            drive_labels=len(drive_labels),
            new_labels=len(missing),
        )

    hmat = _build_hamiltonian_matrix(coeff_map_exyz)
    evals, evecs = np.linalg.eigh(hmat)
    gs_idx = int(np.argmin(evals))
    gs_energy_exact = float(np.real(evals[gs_idx]))
    psi_exact_ground = _normalize_state(np.asarray(evecs[:, gs_idx], dtype=complex).reshape(-1))

    # Sector-filtered exact ground manifold used by subspace-fidelity:
    # select all filtered-sector eigenstates with E <= E0 + tol.
    _vqe_num_particles = _half_filled_particles(int(args.L))
    _fidelity_subspace_tol = float(args.fidelity_subspace_energy_tol)
    if _fidelity_subspace_tol < 0.0:
        raise ValueError("--fidelity-subspace-energy-tol must be >= 0.")
    try:
        gs_energy_exact_filtered, fidelity_subspace_basis_v0 = _ground_manifold_basis_sector_filtered(
            hmat=hmat,
            num_sites=int(args.L),
            num_particles=_vqe_num_particles,
            ordering=str(args.ordering),
            energy_tol=_fidelity_subspace_tol,
        )
        psi_exact_ground_filtered = _normalize_state(
            np.asarray(fidelity_subspace_basis_v0[:, 0], dtype=complex).reshape(-1)
        )
        fidelity_subspace_dimension = int(fidelity_subspace_basis_v0.shape[1])
    except Exception as _exc_filt:
        _ai_log("hardcoded_filtered_exact_failed", error=str(_exc_filt))
        gs_energy_exact_filtered = None
        psi_exact_ground_filtered = psi_exact_ground
        fidelity_subspace_basis_v0 = psi_exact_ground.reshape(-1, 1)
        fidelity_subspace_dimension = 1

    try:
        vqe_payload, psi_vqe = _run_hardcoded_vqe(
            num_sites=int(args.L),
            ordering=str(args.ordering),
            boundary=str(args.boundary),
            hopping_t=float(args.t),
            onsite_u=float(args.u),
            potential_dv=float(args.dv),
            h_poly=h_poly,
            reps=int(args.vqe_reps),
            restarts=int(args.vqe_restarts),
            seed=int(args.vqe_seed),
            maxiter=int(args.vqe_maxiter),
            method=str(args.vqe_method),
            ansatz_name=str(args.vqe_ansatz),
        )
    except Exception as exc:
        _ai_log("hardcoded_vqe_failed", L=int(args.L), error=str(exc))
        vqe_payload = {
            "success": False,
            "method": "hardcoded_layerwise_statevector",
            "ansatz": str(args.vqe_ansatz),
            "parameterization": "layerwise",
            "exact_filtered_energy": None,
            "energy": None,
            "error": str(exc),
        }
        psi_vqe = psi_exact_ground

    num_particles = _half_filled_particles(int(args.L))
    psi_hf = _normalize_state(
        np.asarray(
            hartree_fock_statevector(int(args.L), num_particles, indexing=str(args.ordering)),
            dtype=complex,
        ).reshape(-1)
    )

    if args.initial_state_source == "vqe" and bool(vqe_payload.get("success", False)):
        psi0 = psi_vqe
        selected_initial_source = "vqe"
        _ai_log("hardcoded_initial_state_selected", source=selected_initial_source)
    elif args.initial_state_source == "vqe":
        raise RuntimeError("Requested --initial-state-source vqe but hardcoded VQE statevector is unavailable.")
    elif args.initial_state_source == "hf":
        psi0 = psi_hf
        selected_initial_source = "hf"
        _ai_log("hardcoded_initial_state_selected", source=selected_initial_source)
    else:
        psi0 = psi_exact_ground
        selected_initial_source = "exact"
        _ai_log("hardcoded_initial_state_selected", source=selected_initial_source)

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
        _ai_log("hardcoded_qpe_skipped", eval_qubits=int(args.qpe_eval_qubits), shots=int(args.qpe_shots))
    else:
        qpe_payload = _run_qpe_adapter_qiskit(
            coeff_map_exyz=coeff_map_exyz,
            psi_init=psi0,
            eval_qubits=int(args.qpe_eval_qubits),
            shots=int(args.qpe_shots),
            seed=int(args.qpe_seed),
        )

    trajectory, _exact_states = _simulate_trajectory(
        num_sites=int(args.L),
        ordering=str(args.ordering),
        psi0_ansatz_trot=psi0,
        psi0_exact_ref=psi_exact_ground_filtered,
        fidelity_subspace_basis_v0=fidelity_subspace_basis_v0,
        fidelity_subspace_energy_tol=_fidelity_subspace_tol,
        hmat=hmat,
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
        "vqe_ansatz": str(args.vqe_ansatz),
        "initial_state_source": str(args.initial_state_source),
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
                "n_up": int(_vqe_num_particles[0]),
                "n_dn": int(_vqe_num_particles[1]),
            },
            "ground_subspace_dimension": int(fidelity_subspace_dimension),
            "selection_rule": "E <= E0 + tol",
        },
        "fidelity_reference_initial_state": "exact_static_ground_manifold_filtered_sector",
        "fidelity_reference_sector": {
            "n_up": int(_vqe_num_particles[0]),
            "n_dn": int(_vqe_num_particles[1]),
        },
        "fidelity_ansatz_initial_state": str(selected_initial_source),
        "trajectory_branches_definition": (
            "exact_gs: exact/reference propagation from filtered-sector ground-state init; "
            "exact_ansatz: exact/reference propagation from ansatz initial state; "
            "trotter: Suzuki-Trotter propagation from ansatz initial state."
        ),
        "energy_observable_definition": (
            "energy_static_exact is <psi_exact_gs_ref(t)|H_static|psi_exact_gs_ref(t)>. "
            "energy_static_exact_ansatz is <psi_exact_ansatz_ref(t)|H_static|psi_exact_ansatz_ref(t)>. "
            "energy_static_trotter is <psi_ansatz_trot(t)|H_static|psi_ansatz_trot(t)>. "
            "energy_total_exact is <psi_exact_gs_ref(t)|H_static + H_drive(drive_t0 + t)|psi_exact_gs_ref(t)>. "
            "energy_total_exact_ansatz is <psi_exact_ansatz_ref(t)|H_static + H_drive(drive_t0 + t)|psi_exact_ansatz_ref(t)>. "
            "energy_total_trotter is <psi_ansatz_trot(t)|H_static + H_drive(drive_t0 + t)|psi_ansatz_trot(t)>. "
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
            "propagator_backend": "scipy_sparse_expm_multiply",
        }

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "hardcoded",
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
            "exact_energy_filtered": float(gs_energy_exact_filtered) if gs_energy_exact_filtered is not None else None,
            "filtered_sector": {
                "n_up": int(_vqe_num_particles[0]),
                "n_dn": int(_vqe_num_particles[1]),
            },
            "ground_subspace_dimension": int(fidelity_subspace_dimension),
            "method": "matrix_diagonalization",
        },
        "vqe": vqe_payload,
        "qpe": qpe_payload,
        "initial_state": {
            "source": str(selected_initial_source),
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
        "hardcoded_main_done",
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
