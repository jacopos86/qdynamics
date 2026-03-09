#!/usr/bin/env python3
"""Cross-check suite: compare multiple ansätze × VQE modes against exact ED.

For a given (L, t, U) setup, runs every applicable ansatz through VQE,
propagates the VQE ground state under Trotter dynamics, compares against
exact eigendecomposition, and produces a multi-page PDF.

Supported modes:
  • Pure Hubbard: HVA-layerwise, UCCSD-layerwise, ADAPT(uccsd), ADAPT(full_hamiltonian)
  • Hubbard-Holstein: HH-termwise, HH-layerwise, ADAPT(hva), ADAPT(full_hamiltonian)

Auto-scales parameters by L per AGENTS.md §4d.
No Qiskit in the core path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.reports.pdf_utils import (
    HAS_MATPLOTLIB,
    require_matplotlib,
    get_plt,
    get_PdfPages,
    render_command_page,
    render_compact_table,
    current_command_string,
)
from docs.reports.report_pages import (
    render_executive_summary_page,
    render_manifest_overview_page,
    render_section_divider_page,
)

# plt and PdfPages are fetched inside _write_pdf after require_matplotlib() guard.

from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
    boson_qubits_per_site,
)
from src.quantum.hartree_fock_reference_state import (
    hartree_fock_statevector,
    hubbard_holstein_reference_state,
)
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    HardcodedUCCSDLayerwiseAnsatz,
    HubbardLayerwiseAnsatz,
    HubbardHolsteinTermwiseAnsatz,
    HubbardHolsteinLayerwiseAnsatz,
    apply_exp_pauli_polynomial,
    apply_pauli_string,
    basis_state,
    exact_ground_energy_sector,
    exact_ground_energy_sector_hh,
    expval_pauli_polynomial,
    half_filled_num_particles,
    hartree_fock_bitstring,
    vqe_minimize,
)
from pipelines.exact_bench.benchmark_metrics_proxy import write_proxy_sidecars

# ---------------------------------------------------------------------------
# Structured-log helper
# ---------------------------------------------------------------------------

def _ai_log(event: str, **fields: Any) -> None:
    payload = {"event": str(event), "ts_utc": datetime.now(timezone.utc).isoformat(), **fields}
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


# ---------------------------------------------------------------------------
# §4d auto-scaling tables
# ---------------------------------------------------------------------------

# Pure Hubbard minimum parameters per L
_HUBBARD_PARAMS: dict[int, dict[str, Any]] = {
    2: {"trotter_steps": 64,  "exact_mult": 2, "num_times": 201, "reps": 2, "restarts": 2, "maxiter": 600,   "method": "COBYLA", "t_final": 10.0},
    3: {"trotter_steps": 128, "exact_mult": 2, "num_times": 201, "reps": 2, "restarts": 3, "maxiter": 1200,  "method": "COBYLA", "t_final": 15.0},
    4: {"trotter_steps": 256, "exact_mult": 3, "num_times": 241, "reps": 3, "restarts": 4, "maxiter": 6000,  "method": "SLSQP",  "t_final": 20.0},
    5: {"trotter_steps": 384, "exact_mult": 3, "num_times": 301, "reps": 3, "restarts": 5, "maxiter": 8000,  "method": "SLSQP",  "t_final": 20.0},
    6: {"trotter_steps": 512, "exact_mult": 4, "num_times": 361, "reps": 4, "restarts": 6, "maxiter": 10000, "method": "SLSQP",  "t_final": 20.0},
}

# Hubbard-Holstein minimum parameters per (L, n_ph_max)
_HH_PARAMS: dict[tuple[int, int], dict[str, Any]] = {
    (2, 1): {"trotter_steps": 64,  "reps": 2, "restarts": 3, "maxiter": 800,  "method": "COBYLA"},
    (2, 2): {"trotter_steps": 128, "reps": 3, "restarts": 4, "maxiter": 1500, "method": "COBYLA"},
    (3, 1): {"trotter_steps": 192, "reps": 2, "restarts": 4, "maxiter": 2400, "method": "COBYLA"},
}

# ADAPT-VQE defaults per L
_ADAPT_PARAMS: dict[int, dict[str, Any]] = {
    2: {"max_depth": 40,  "eps_grad": 1e-6, "eps_energy": 1e-8, "maxiter": 600,  "seed": 42},
    3: {"max_depth": 60,  "eps_grad": 1e-6, "eps_energy": 1e-8, "maxiter": 1200, "seed": 42},
    4: {"max_depth": 100, "eps_grad": 1e-7, "eps_energy": 1e-9, "maxiter": 4000, "seed": 42},
    5: {"max_depth": 140, "eps_grad": 1e-7, "eps_energy": 1e-9, "maxiter": 6000, "seed": 42},
    6: {"max_depth": 200, "eps_grad": 1e-7, "eps_energy": 1e-9, "maxiter": 8000, "seed": 42},
}


def _get_hubbard_params(L: int) -> dict[str, Any]:
    if L in _HUBBARD_PARAMS:
        return dict(_HUBBARD_PARAMS[L])
    # Extrapolate from largest known
    base = dict(_HUBBARD_PARAMS[max(_HUBBARD_PARAMS)])
    scale = L / max(_HUBBARD_PARAMS)
    base["trotter_steps"] = int(base["trotter_steps"] * scale)
    base["maxiter"] = int(base["maxiter"] * scale)
    return base


def _get_hh_params(L: int, n_ph_max: int) -> dict[str, Any]:
    key = (L, n_ph_max)
    if key in _HH_PARAMS:
        return dict(_HH_PARAMS[key])
    # Fallback: take closest known or escalate from Hubbard
    hub = _get_hubbard_params(L)
    hub["trotter_steps"] = int(hub["trotter_steps"] * 1.5)
    hub["maxiter"] = int(hub["maxiter"] * 1.5)
    hub["restarts"] = hub.get("restarts", 3) + 1
    return hub


def _get_adapt_params(L: int) -> dict[str, Any]:
    if L in _ADAPT_PARAMS:
        return dict(_ADAPT_PARAMS[L])
    base = dict(_ADAPT_PARAMS[max(_ADAPT_PARAMS)])
    scale = L / max(_ADAPT_PARAMS)
    base["max_depth"] = int(base["max_depth"] * scale)
    base["maxiter"] = int(base["maxiter"] * scale)
    return base


# ---------------------------------------------------------------------------
# Pauli algebra helpers  (copied from hubbard_pipeline.py — same repo)
# ---------------------------------------------------------------------------

PAULI_MATS = {
    "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _pauli_matrix_exyz(label: str) -> np.ndarray:
    mats = [PAULI_MATS[ch] for ch in label]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(psi))
    if nrm <= 0.0:
        raise ValueError("Zero-norm state.")
    return psi / nrm


def _collect_hardcoded_terms_exyz(h_poly: Any) -> tuple[list[str], dict[str, complex]]:
    coeff_map: dict[str, complex] = {}
    native_order: list[str] = []
    for term in h_poly.return_polynomial():
        label: str = term.pw2strng()
        coeff: complex = complex(term.p_coeff)
        if label not in coeff_map:
            native_order.append(label)
            coeff_map[label] = coeff
        else:
            coeff_map[label] += coeff
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


@dataclass(frozen=True)
class _CompiledPauliAction:
    label_exyz: str
    perm: np.ndarray
    phase: np.ndarray


def _compile_pauli_action(label_exyz: str, nq: int) -> _CompiledPauliAction:
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
        elif op == "x":
            perm ^= (1 << q)
        elif op == "y":
            perm ^= (1 << q)
            phase *= 1j * sign
        elif op == "z":
            phase *= sign
        else:
            raise ValueError(f"Unsupported Pauli symbol '{op}'.")
    return _CompiledPauliAction(label_exyz=label_exyz, perm=perm, phase=phase)


def _apply_compiled_pauli(psi: np.ndarray, action: _CompiledPauliAction) -> np.ndarray:
    out = np.empty_like(psi)
    out[action.perm] = action.phase * psi
    return out


def _apply_exp_term(psi: np.ndarray, action: _CompiledPauliAction, coeff: complex, alpha: float) -> np.ndarray:
    if abs(coeff.imag) > 1e-12:
        raise ValueError(f"Imaginary coeff for {action.label_exyz}: {coeff}")
    theta = float(alpha) * float(coeff.real)
    ppsi = _apply_compiled_pauli(psi, action)
    return math.cos(theta) * psi - 1j * math.sin(theta) * ppsi


def _evolve_trotter_suzuki2(
    psi0: np.ndarray,
    ordered_labels: list[str],
    coeff_map: dict[str, complex],
    compiled: dict[str, _CompiledPauliAction],
    time_value: float,
    trotter_steps: int,
) -> np.ndarray:
    """Time-independent Suzuki-2 Trotter propagation."""
    psi = np.array(psi0, copy=True)
    if abs(time_value) <= 1e-15:
        return psi
    dt = float(time_value) / float(trotter_steps)
    half = 0.5 * dt
    for _ in range(trotter_steps):
        for lbl in ordered_labels:
            psi = _apply_exp_term(psi, compiled[lbl], coeff_map[lbl], half)
        for lbl in reversed(ordered_labels):
            psi = _apply_exp_term(psi, compiled[lbl], coeff_map[lbl], half)
    return _normalize_state(psi)


def _expectation_hamiltonian(psi: np.ndarray, hmat: np.ndarray) -> float:
    return float(np.real(np.vdot(psi, hmat @ psi)))


def _spin_orbital_bit_index(site: int, spin: int, num_sites: int, ordering: str) -> int:
    ord_norm = str(ordering).strip().lower()
    if ord_norm == "blocked":
        return int(site) if int(spin) == 0 else int(num_sites) + int(site)
    if ord_norm == "interleaved":
        return (2 * int(site)) + int(spin)
    raise ValueError(f"Unsupported ordering {ordering!r}")


def _site_resolved_observables(
    psi: np.ndarray,
    num_sites: int,
    ordering: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    probs = np.abs(psi) ** 2
    n_up = np.zeros(int(num_sites), dtype=float)
    n_dn = np.zeros(int(num_sites), dtype=float)
    doublon_total = 0.0
    up_bits = [_spin_orbital_bit_index(s, 0, num_sites, ordering) for s in range(int(num_sites))]
    dn_bits = [_spin_orbital_bit_index(s, 1, num_sites, ordering) for s in range(int(num_sites))]
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


# ---------------------------------------------------------------------------
# Sector filtering (exact ground state)
# ---------------------------------------------------------------------------

def _sector_basis_indices(
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
    nq_total: int | None = None,
) -> np.ndarray:
    nq = nq_total if nq_total is not None else 2 * int(num_sites)
    dim = 1 << nq
    n_up_want, n_dn_want = int(num_particles[0]), int(num_particles[1])
    norm = str(ordering).strip().lower()
    idx_all = np.arange(dim, dtype=np.int64)
    if norm == "blocked":
        up_mask = (1 << num_sites) - 1
        dn_mask = up_mask << num_sites
    else:
        up_mask = int(sum(1 << (2 * q) for q in range(num_sites)))
        dn_mask = int(sum(1 << (2 * q + 1) for q in range(num_sites)))
    n_up_arr = np.array([bin(int(i) & int(up_mask)).count("1") for i in idx_all], dtype=np.int32)
    n_dn_arr = np.array([bin(int(i) & int(dn_mask)).count("1") for i in idx_all], dtype=np.int32)
    sector = np.where((n_up_arr == n_up_want) & (n_dn_arr == n_dn_want))[0]
    if sector.size == 0:
        raise ValueError(f"No basis for sector ({n_up_want},{n_dn_want})")
    return sector


def _exact_ground_state_sector_filtered(
    hmat: np.ndarray,
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
    nq_total: int | None = None,
) -> tuple[float, np.ndarray]:
    sector_idx = _sector_basis_indices(num_sites, num_particles, ordering, nq_total)
    h_sector = hmat[np.ix_(sector_idx, sector_idx)]
    evals, evecs = np.linalg.eigh(h_sector)
    gs_energy = float(np.real(evals[0]))
    psi_sector = np.asarray(evecs[:, 0], dtype=complex)
    psi_full = np.zeros(hmat.shape[0], dtype=complex)
    psi_full[sector_idx] = psi_sector
    return gs_energy, _normalize_state(psi_full)


# ---------------------------------------------------------------------------
# ADAPT-VQE mini-runner  (simplified from adapt_pipeline.py)
# ---------------------------------------------------------------------------

def _build_adapt_pool_hubbard(
    h_poly: Any,
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
    pool_name: str,
) -> list[AnsatzTerm]:
    """Build an ADAPT operator pool for pure Hubbard."""
    from src.quantum.vqe_latex_python_pairs import (
        HardcodedUCCSDAnsatz,
    )

    pool_name = pool_name.strip().lower()

    if pool_name == "uccsd":
        # Reuse the exact same excitation generators the VQE ansatz uses
        # (HardcodedUCCSDAnsatz produces generators with real Pauli coefficients)
        dummy = HardcodedUCCSDAnsatz(
            dims=int(num_sites),
            num_particles=num_particles,
            reps=1,
            repr_mode="JW",
            indexing=str(ordering),
            include_singles=True,
            include_doubles=True,
        )
        return list(dummy.base_terms)

    elif pool_name == "full_hamiltonian":
        # Each Hamiltonian term as a pool operator (unit coefficient)
        pool: list[AnsatzTerm] = []
        for term in h_poly.return_polynomial():
            label = term.pw2strng()
            coeff = complex(term.p_coeff)
            if abs(coeff) < 1e-15:
                continue
            nq = int(term.nqubit())
            id_label = "e" * nq
            if label == id_label:
                continue
            from src.quantum.pauli_polynomial_class import PauliPolynomial
            from src.quantum.pauli_words import PauliTerm as PW
            p = PauliPolynomial(nq)
            p.add_term(PW(nq, ps=label, pc=1.0))
            pool.append(AnsatzTerm(label=f"ham_term({label})", polynomial=p))
        return pool

    else:
        raise ValueError(f"Unsupported Hubbard ADAPT pool: {pool_name}")


def _build_adapt_pool_hh(
    h_poly: Any,
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
    pool_name: str,
    n_ph_max: int,
    boson_encoding: str,
) -> list[AnsatzTerm]:
    """Build an ADAPT operator pool for Hubbard-Holstein."""
    pool_name = pool_name.strip().lower()

    if pool_name == "full_hamiltonian":
        pool: list[AnsatzTerm] = []
        for term in h_poly.return_polynomial():
            label = term.pw2strng()
            coeff = complex(term.p_coeff)
            if abs(coeff) < 1e-15:
                continue
            from src.quantum.pauli_polynomial_class import PauliPolynomial
            from src.quantum.pauli_words import PauliTerm as PW
            nq = int(term.nqubit())
            p = PauliPolynomial(nq)
            p.add_term(PW(nq, ps=label, pc=1.0))
            pool.append(AnsatzTerm(label=f"ham_term({label})", polynomial=p))
        return pool

    elif pool_name == "hva":
        # Delegate to the adapt_pipeline's builder via direct import
        sys_path_orig = list(sys.path)
        try:
            adapt_dir = str(REPO_ROOT / "pipelines" / "hardcoded")
            if adapt_dir not in sys.path:
                sys.path.insert(0, adapt_dir)
            from adapt_pipeline import _build_hva_pool  # type: ignore[import]
        finally:
            sys.path[:] = sys_path_orig

        # HVA pool needs physical parameters — we use defaults
        # (the pool structure doesn't depend on exact coupling values,
        # just the topology)
        pool = _build_hva_pool(
            num_sites, 1.0, 1.0,  # t, U placeholder (pool ops come from H structure)
            1.0, 0.5,  # omega0, g_ep placeholder
            0.0,  # dv
            n_ph_max, boson_encoding,
            ordering, "periodic",
        )
        return pool

    else:
        raise ValueError(f"Unsupported HH ADAPT pool: {pool_name}")


# ---------------------------------------------------------------------------
# ADAPT-VQE mini loop
# ---------------------------------------------------------------------------

def _run_adapt_vqe(
    h_poly: Any,
    pool: list[AnsatzTerm],
    psi_ref: np.ndarray,
    *,
    max_depth: int,
    eps_grad: float,
    eps_energy: float,
    maxiter: int,
    seed: int,
) -> tuple[float, np.ndarray, int, int, str]:
    """Minimal ADAPT-VQE loop. Returns (energy, psi, nfev, depth, stop_reason)."""
    from scipy.optimize import minimize as scipy_minimize

    np.random.seed(seed)
    selected_ops: list[AnsatzTerm] = []
    theta = np.zeros(0, dtype=float)
    nfev_total = 0
    stop_reason = "max_depth"
    energy_current = float(expval_pauli_polynomial(psi_ref, h_poly))
    nfev_total += 1
    available = set(range(len(pool)))

    for depth in range(int(max_depth)):
        psi_current = _prepare_adapt_state(psi_ref, selected_ops, theta)

        # Compute gradients
        gradients = np.zeros(len(pool), dtype=float)
        for i in available:
            gradients[i] = _commutator_gradient(h_poly, pool[i], psi_current)

        grad_mag = np.abs(gradients)
        mask = np.zeros(len(pool), dtype=bool)
        for i in available:
            mask[i] = True
        grad_mag[~mask] = 0.0
        best_idx = int(np.argmax(grad_mag))
        max_grad = float(grad_mag[best_idx])

        if max_grad < float(eps_grad):
            stop_reason = "eps_grad"
            break

        selected_ops.append(pool[best_idx])
        theta = np.append(theta, 0.0)
        available.discard(best_idx)

        energy_prev = energy_current

        def _obj(x: np.ndarray) -> float:
            return _adapt_energy_fn(h_poly, psi_ref, selected_ops, x)

        result = scipy_minimize(
            _obj, theta, method="COBYLA",
            options={"maxiter": int(maxiter), "rhobeg": 0.3},
        )
        theta = np.asarray(result.x, dtype=float)
        energy_current = float(result.fun)
        nfev_total += int(getattr(result, "nfev", 0))

        _ai_log("adapt_iter", depth=depth + 1, energy=energy_current,
                max_grad=max_grad, op=str(pool[best_idx].label))

        if abs(energy_current - energy_prev) < float(eps_energy):
            stop_reason = "eps_energy"
            break

        if not available:
            stop_reason = "pool_exhausted"
            break

    psi_final = _prepare_adapt_state(psi_ref, selected_ops, theta)
    psi_final = _normalize_state(psi_final)
    return energy_current, psi_final, nfev_total, len(selected_ops), stop_reason


def _apply_pauli_polynomial(state: np.ndarray, poly: Any) -> np.ndarray:
    terms = poly.return_polynomial()
    nq = int(terms[0].nqubit())
    result = np.zeros_like(state)
    for term in terms:
        ps = term.pw2strng()
        coeff = complex(term.p_coeff)
        if abs(coeff) < 1e-15:
            continue
        if ps == "e" * nq:
            result += coeff * state
        else:
            result += coeff * apply_pauli_string(state, ps)
    return result


def _commutator_gradient(h_poly: Any, pool_op: AnsatzTerm, psi: np.ndarray) -> float:
    G_psi = _apply_pauli_polynomial(psi, pool_op.polynomial)
    H_psi = _apply_pauli_polynomial(psi, h_poly)
    return float(2.0 * np.vdot(H_psi, G_psi).imag)


def _prepare_adapt_state(psi_ref: np.ndarray, ops: list[AnsatzTerm], theta: np.ndarray) -> np.ndarray:
    psi = np.array(psi_ref, copy=True)
    for k, op in enumerate(ops):
        psi = apply_exp_pauli_polynomial(psi, op.polynomial, float(theta[k]))
    return psi


def _adapt_energy_fn(h_poly: Any, psi_ref: np.ndarray, ops: list[AnsatzTerm], theta: np.ndarray) -> float:
    psi = _prepare_adapt_state(psi_ref, ops, theta)
    return float(expval_pauli_polynomial(psi, h_poly))


# ---------------------------------------------------------------------------
# Trial data container
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    name: str
    category: str  # "conventional_vqe" or "adapt_vqe"
    ansatz_label: str
    energy: float
    exact_energy: float
    delta_e: float
    num_params: int
    nfev: int
    psi_vqe: np.ndarray
    elapsed_s: float
    # ADAPT-specific
    adapt_depth: int = 0
    adapt_stop_reason: str = ""
    # Trajectory timeseries (filled after Trotter propagation)
    trajectory: list[dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trajectory simulation
# ---------------------------------------------------------------------------

def _simulate_trajectory(
    *,
    num_sites: int,
    ordering: str,
    psi0: np.ndarray,
    hmat: np.ndarray,
    ordered_labels: list[str],
    coeff_map: dict[str, complex],
    trotter_steps: int,
    t_final: float,
    num_times: int,
) -> list[dict[str, float]]:
    """Trotter + exact trajectory from the given initial state."""
    nq = int(round(math.log2(hmat.shape[0])))
    evals, evecs = np.linalg.eigh(hmat)
    evecs_dag = np.conjugate(evecs).T
    compiled = {lbl: _compile_pauli_action(lbl, nq) for lbl in ordered_labels}
    times = np.linspace(0.0, float(t_final), int(num_times))
    rows: list[dict[str, float]] = []

    for idx, tv in enumerate(times):
        t = float(tv)
        psi_exact = evecs @ (np.exp(-1j * evals * t) * (evecs_dag @ psi0))
        psi_exact = _normalize_state(psi_exact)
        psi_trot = _evolve_trotter_suzuki2(psi0, ordered_labels, coeff_map, compiled, t, trotter_steps)
        fidelity = float(abs(np.vdot(psi_exact, psi_trot)) ** 2)
        n_up_ex, n_dn_ex, dbl_ex = _site_resolved_observables(psi_exact, num_sites, ordering)
        n_up_tr, n_dn_tr, dbl_tr = _site_resolved_observables(psi_trot, num_sites, ordering)
        rows.append({
            "time": t,
            "fidelity": fidelity,
            "energy_exact": _expectation_hamiltonian(psi_exact, hmat),
            "energy_trotter": _expectation_hamiltonian(psi_trot, hmat),
            "n_up_site0_exact": float(n_up_ex[0]) if n_up_ex.size > 0 else float("nan"),
            "n_up_site0_trotter": float(n_up_tr[0]) if n_up_tr.size > 0 else float("nan"),
            "n_dn_site0_exact": float(n_dn_ex[0]) if n_dn_ex.size > 0 else float("nan"),
            "n_dn_site0_trotter": float(n_dn_tr[0]) if n_dn_tr.size > 0 else float("nan"),
            "doublon_exact": dbl_ex,
            "doublon_trotter": dbl_tr,
        })
        if idx % max(1, len(times) // 10) == 0:
            _ai_log("trajectory_progress", ansatz="...", step=idx + 1, total=len(times), fidelity=fidelity)

    return rows


# ---------------------------------------------------------------------------
# Trial runners
# ---------------------------------------------------------------------------

def _run_conventional_vqe_trial(
    *,
    name: str,
    ansatz: Any,
    h_poly: Any,
    psi_ref: np.ndarray,
    exact_energy: float,
    restarts: int,
    seed: int,
    maxiter: int,
    method: str,
) -> TrialResult:
    _ai_log("trial_start", name=name, category="conventional_vqe", num_params=int(ansatz.num_parameters))
    t0 = time.perf_counter()
    result = vqe_minimize(
        h_poly, ansatz, psi_ref,
        restarts=restarts, seed=seed, maxiter=maxiter, method=method,
    )
    theta = np.asarray(result.theta, dtype=float)
    psi_vqe = np.asarray(ansatz.prepare_state(theta, psi_ref), dtype=complex).ravel()
    psi_vqe = _normalize_state(psi_vqe)
    elapsed = time.perf_counter() - t0
    _ai_log("trial_done", name=name, energy=float(result.energy),
            delta_e=float(result.energy - exact_energy), elapsed_s=round(elapsed, 2))
    return TrialResult(
        name=name,
        category="conventional_vqe",
        ansatz_label=name,
        energy=float(result.energy),
        exact_energy=exact_energy,
        delta_e=float(result.energy - exact_energy),
        num_params=int(ansatz.num_parameters),
        nfev=int(getattr(result, "nfev", 0)),
        psi_vqe=psi_vqe,
        elapsed_s=elapsed,
    )


def _run_adapt_vqe_trial(
    *,
    name: str,
    h_poly: Any,
    pool: list[AnsatzTerm],
    psi_ref: np.ndarray,
    exact_energy: float,
    max_depth: int,
    eps_grad: float,
    eps_energy: float,
    maxiter: int,
    seed: int,
) -> TrialResult:
    _ai_log("trial_start", name=name, category="adapt_vqe", pool_size=len(pool))
    t0 = time.perf_counter()
    energy, psi, nfev, depth, stop = _run_adapt_vqe(
        h_poly, pool, psi_ref,
        max_depth=max_depth,
        eps_grad=eps_grad,
        eps_energy=eps_energy,
        maxiter=maxiter,
        seed=seed,
    )
    elapsed = time.perf_counter() - t0
    _ai_log("trial_done", name=name, energy=energy,
            delta_e=float(energy - exact_energy), depth=depth, stop=stop, elapsed_s=round(elapsed, 2))
    return TrialResult(
        name=name,
        category="adapt_vqe",
        ansatz_label=name,
        energy=energy,
        exact_energy=exact_energy,
        delta_e=float(energy - exact_energy),
        num_params=depth,
        nfev=nfev,
        psi_vqe=psi,
        elapsed_s=elapsed,
        adapt_depth=depth,
        adapt_stop_reason=stop,
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_cross_check(args: argparse.Namespace) -> dict[str, Any]:
    """Run all applicable ansatz trials for the given problem and L."""
    L = int(args.L)
    t_hop = float(args.t)
    U = float(args.U)
    dv = float(args.dv)
    ordering = str(args.ordering)
    boundary = str(args.boundary)
    problem = str(args.problem).strip().lower()
    is_hh = (problem == "hh")

    _ai_log("cross_check_start", L=L, problem=problem, t=t_hop, U=U, dv=dv)

    # --- 1. Build Hamiltonian ---
    if is_hh:
        omega0 = float(args.omega0)
        g_ep = float(args.g_ep)
        n_ph_max = int(args.n_ph_max)
        boson_enc = str(args.boson_encoding)
        h_poly = build_hubbard_holstein_hamiltonian(
            L,
            J=t_hop, U=U,
            omega0=omega0, g=g_ep,
            n_ph_max=n_ph_max,
            boson_encoding=boson_enc,
            repr_mode="JW", indexing=ordering,
            pbc=(boundary == "periodic"),
        )
        hh_params = _get_hh_params(L, n_ph_max)
    else:
        h_poly = build_hubbard_hamiltonian(
            L, t_hop, U,
            v=dv if abs(dv) > 1e-15 else None,
            repr_mode="JW", indexing=ordering,
            pbc=(boundary == "periodic"),
        )

    # Auto-scale parameters
    hub_params = _get_hubbard_params(L)
    adapt_params = _get_adapt_params(L)
    if is_hh:
        hub_params.update(hh_params)

    # CLI overrides
    reps = int(args.vqe_reps) if args.vqe_reps is not None else hub_params["reps"]
    restarts = int(args.vqe_restarts) if args.vqe_restarts is not None else hub_params["restarts"]
    maxiter = int(args.vqe_maxiter) if args.vqe_maxiter is not None else hub_params["maxiter"]
    method = str(args.vqe_method) if args.vqe_method is not None else hub_params["method"]
    trotter_steps = int(args.trotter_steps) if args.trotter_steps is not None else hub_params["trotter_steps"]
    num_times = int(args.num_times) if args.num_times is not None else hub_params.get("num_times", 201)
    t_final = float(args.t_final) if args.t_final is not None else hub_params.get("t_final", 10.0)
    seed = int(args.seed)

    # --- 2. Collect terms and build matrix ---
    ordered_labels, coeff_map = _collect_hardcoded_terms_exyz(h_poly)
    hmat = _build_hamiltonian_matrix(coeff_map)
    num_particles = half_filled_num_particles(L)
    nq = int(round(math.log2(hmat.shape[0])))

    # --- 3. Exact ground state ---
    gs_energy, psi_gs = _exact_ground_state_sector_filtered(
        hmat, L, num_particles, ordering,
        nq_total=nq if is_hh else None,
    )
    _ai_log("exact_ground_state", energy=gs_energy)

    # --- 4. Reference state ---
    if is_hh:
        psi_ref = np.asarray(
            hubbard_holstein_reference_state(
                dims=L,
                num_particles=num_particles,
                n_ph_max=n_ph_max,
                boson_encoding=boson_enc,
                indexing=ordering,
            ), dtype=complex,
        )
    else:
        hf_bits = hartree_fock_bitstring(n_sites=L, num_particles=num_particles, indexing=ordering)
        psi_ref = np.asarray(basis_state(2 * L, hf_bits), dtype=complex)

    # --- 5. Build and run all trials ---
    trials: list[TrialResult] = []
    pbc_flag = (boundary == "periodic")

    if is_hh:
        # Trial 1: HH Termwise
        _ai_log("building_ansatz", name="HH-Termwise")
        ans_tw = HubbardHolsteinTermwiseAnsatz(
            dims=L, J=t_hop, U=U, omega0=omega0, g=g_ep,
            n_ph_max=n_ph_max, boson_encoding=boson_enc,
            reps=reps, repr_mode="JW", indexing=ordering, pbc=pbc_flag,
        )
        trials.append(_run_conventional_vqe_trial(
            name="HH-Termwise", ansatz=ans_tw, h_poly=h_poly, psi_ref=psi_ref,
            exact_energy=gs_energy, restarts=restarts, seed=seed, maxiter=maxiter, method=method,
        ))

        # Trial 2: HH Layerwise
        _ai_log("building_ansatz", name="HH-Layerwise")
        ans_lw = HubbardHolsteinLayerwiseAnsatz(
            dims=L, J=t_hop, U=U, omega0=omega0, g=g_ep,
            n_ph_max=n_ph_max, boson_encoding=boson_enc,
            reps=reps, repr_mode="JW", indexing=ordering, pbc=pbc_flag,
        )
        trials.append(_run_conventional_vqe_trial(
            name="HH-Layerwise", ansatz=ans_lw, h_poly=h_poly, psi_ref=psi_ref,
            exact_energy=gs_energy, restarts=restarts, seed=seed, maxiter=maxiter, method=method,
        ))

        # Trial 3: ADAPT(full_hamiltonian) for HH
        pool_fh = _build_adapt_pool_hh(h_poly, L, num_particles, ordering, "full_hamiltonian", n_ph_max, boson_enc)
        if pool_fh:
            trials.append(_run_adapt_vqe_trial(
                name="ADAPT(full_H)", h_poly=h_poly, pool=pool_fh, psi_ref=psi_ref,
                exact_energy=gs_energy, max_depth=adapt_params["max_depth"],
                eps_grad=adapt_params["eps_grad"], eps_energy=adapt_params["eps_energy"],
                maxiter=adapt_params["maxiter"], seed=adapt_params["seed"],
            ))

    else:
        # Trial 1: HVA Layerwise (Hubbard)
        _ai_log("building_ansatz", name="HVA-Layerwise")
        ans_hva = HubbardLayerwiseAnsatz(
            dims=L, t=t_hop, U=U,
            v=dv if abs(dv) > 1e-15 else 0.0,
            reps=reps, repr_mode="JW", indexing=ordering, pbc=pbc_flag,
            include_potential_terms=True,
        )
        trials.append(_run_conventional_vqe_trial(
            name="HVA-Layerwise", ansatz=ans_hva, h_poly=h_poly, psi_ref=psi_ref,
            exact_energy=gs_energy, restarts=restarts, seed=seed, maxiter=maxiter, method=method,
        ))

        # Trial 2: UCCSD Layerwise
        _ai_log("building_ansatz", name="UCCSD-Layerwise")
        ans_uccsd = HardcodedUCCSDLayerwiseAnsatz(
            dims=L, num_particles=num_particles,
            reps=reps, repr_mode="JW", indexing=ordering,
            include_singles=True, include_doubles=True,
        )
        trials.append(_run_conventional_vqe_trial(
            name="UCCSD-Layerwise", ansatz=ans_uccsd, h_poly=h_poly, psi_ref=psi_ref,
            exact_energy=gs_energy, restarts=restarts, seed=seed, maxiter=maxiter, method=method,
        ))

        # Trial 3: ADAPT(uccsd)
        pool_uccsd = _build_adapt_pool_hubbard(h_poly, L, num_particles, ordering, "uccsd")
        if pool_uccsd:
            trials.append(_run_adapt_vqe_trial(
                name="ADAPT(UCCSD)", h_poly=h_poly, pool=pool_uccsd, psi_ref=psi_ref,
                exact_energy=gs_energy, max_depth=adapt_params["max_depth"],
                eps_grad=adapt_params["eps_grad"], eps_energy=adapt_params["eps_energy"],
                maxiter=adapt_params["maxiter"], seed=adapt_params["seed"],
            ))

        # Trial 4: ADAPT(full_hamiltonian)
        pool_fh = _build_adapt_pool_hubbard(h_poly, L, num_particles, ordering, "full_hamiltonian")
        if pool_fh:
            trials.append(_run_adapt_vqe_trial(
                name="ADAPT(full_H)", h_poly=h_poly, pool=pool_fh, psi_ref=psi_ref,
                exact_energy=gs_energy, max_depth=adapt_params["max_depth"],
                eps_grad=adapt_params["eps_grad"], eps_energy=adapt_params["eps_energy"],
                maxiter=adapt_params["maxiter"], seed=adapt_params["seed"],
            ))

    # --- 6. Run Trotter trajectories for each trial ---
    _ai_log("trajectory_phase_start", n_trials=len(trials), trotter_steps=trotter_steps,
            num_times=num_times, t_final=t_final)
    for trial in trials:
        _ai_log("trajectory_start", ansatz=trial.name)
        trial.trajectory = _simulate_trajectory(
            num_sites=L,
            ordering=ordering,
            psi0=trial.psi_vqe,
            hmat=hmat,
            ordered_labels=ordered_labels,
            coeff_map=coeff_map,
            trotter_steps=trotter_steps,
            t_final=t_final,
            num_times=num_times,
        )
        _ai_log("trajectory_done", ansatz=trial.name,
                final_fidelity=trial.trajectory[-1]["fidelity"] if trial.trajectory else None)

    # --- 7. Build output payload ---
    payload = _build_payload(args, trials, gs_energy, hub_params, adapt_params)

    # --- 8. Write artifacts ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"L{L}_{problem}_t{t_hop}_U{U}"
    json_path = out_dir / f"xchk_{tag}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    _ai_log("json_written", path=str(json_path))

    if HAS_MATPLOTLIB:
        pdf_path = out_dir / f"xchk_{tag}.pdf"
        _write_pdf(pdf_path, trials, args, gs_energy, payload)
        _ai_log("pdf_written", path=str(pdf_path))
    else:
        _ai_log("pdf_skipped", reason="matplotlib_unavailable")

    summary_dir = out_dir / "summary"
    sidecars = write_proxy_sidecars(
        payload.get("trials", []),
        summary_dir,
        defaults={
            "problem": str(problem),
            "L": int(L),
            "vqe_reps": int(reps),
            "vqe_restarts": int(restarts),
            "vqe_maxiter": int(maxiter),
        },
    )
    _ai_log(
        "metrics_proxy_written",
        csv=str(sidecars["csv"]),
        jsonl=str(sidecars["jsonl"]),
        summary_json=str(sidecars["summary_json"]),
    )

    _ai_log("cross_check_done", n_trials=len(trials))
    return payload


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def _build_payload(
    args: argparse.Namespace,
    trials: list[TrialResult],
    exact_energy: float,
    hub_params: dict,
    adapt_params: dict,
) -> dict[str, Any]:
    trial_summaries = []
    for tr in trials:
        d: dict[str, Any] = {
            "run_id": tr.name,
            "method_id": tr.name,
            "method_kind": tr.category,
            "ansatz_name": tr.ansatz_label,
            "name": tr.name,
            "category": tr.category,
            "energy": tr.energy,
            "exact_energy": tr.exact_energy,
            "delta_e": tr.delta_e,
            "abs_delta_e": abs(tr.delta_e),
            "delta_E_abs": abs(tr.delta_e),
            "num_parameters": tr.num_params,
            "num_params": tr.num_params,
            "nfev": tr.nfev,
            "runtime_s": round(tr.elapsed_s, 3),
            "elapsed_s": round(tr.elapsed_s, 3),
        }
        if tr.category == "adapt_vqe":
            d["adapt_depth_reached"] = tr.adapt_depth
            d["adapt_depth"] = tr.adapt_depth
            d["adapt_stop_reason"] = tr.adapt_stop_reason
        if tr.trajectory:
            d["final_fidelity"] = tr.trajectory[-1]["fidelity"]
            d["min_fidelity"] = min(r["fidelity"] for r in tr.trajectory)
        trial_summaries.append(d)

    return {
        "cross_check_suite": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "L": int(args.L),
            "problem": str(args.problem),
            "t": float(args.t),
            "U": float(args.U),
            "dv": float(args.dv),
            "ordering": str(args.ordering),
            "boundary": str(args.boundary),
            "auto_scaled_hub": hub_params,
            "auto_scaled_adapt": adapt_params,
        },
        "exact_ground_energy": exact_energy,
        "trials": trial_summaries,
    }


# ---------------------------------------------------------------------------
# PDF writer
# ---------------------------------------------------------------------------

def _write_pdf(
    pdf_path: Path,
    trials: list[TrialResult],
    args: argparse.Namespace,
    exact_energy: float,
    payload: dict[str, Any],
) -> None:
    require_matplotlib()
    plt = get_plt()
    PdfP = get_PdfPages()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    model_name = "Hubbard-Holstein" if args.problem == "hh" else "Hubbard"
    categories = sorted({str(tr.category).replace("_", " ") for tr in trials})
    ranked_trials = sorted(trials, key=lambda tr: abs(float(tr.delta_e)))
    best_trial = ranked_trials[0] if ranked_trials else None
    adapt_trials = [tr for tr in ranked_trials if "adapt" in str(tr.category).lower() or "adapt" in str(tr.name).lower()]
    conventional_trials = [tr for tr in ranked_trials if tr not in adapt_trials]

    def _trial_metric(trial: TrialResult | None, attr: str) -> Any:
        if trial is None:
            return "n/a"
        value = getattr(trial, attr)
        if attr == "delta_e":
            return abs(float(value))
        return value

    manifest_sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "Model and regime",
            [
                ("Model family", model_name),
                ("Problem", args.problem),
                ("L", args.L),
                ("Ordering", args.ordering),
                ("Boundary", args.boundary),
            ],
        ),
        (
            "Trial matrix",
            [
                ("Categories", categories),
                ("# ansatz trials", len(trials)),
                ("Exact filtered ground energy", exact_energy),
            ],
        ),
        (
            "Core physical parameters",
            [
                ("t", args.t),
                ("U", args.U),
                ("dv", args.dv),
            ],
        ),
        (
            "Dynamics grid",
            [
                ("t_final", args.t_final),
                ("num_times", args.num_times),
                ("trotter_steps", args.trotter_steps),
                ("Seed", args.seed),
            ],
        ),
    ]
    if args.problem == "hh":
        manifest_sections.append(
            (
                "Hubbard-Holstein parameters",
                [
                    ("omega0", args.omega0),
                    ("g_ep", args.g_ep),
                    ("n_ph_max", args.n_ph_max),
                    ("Boson encoding", args.boson_encoding),
                ],
            )
        )

    summary_sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "Best performers",
            [
                ("Best overall", _trial_metric(best_trial, "name")),
                ("Best overall |ΔE|", _trial_metric(best_trial, "delta_e")),
                ("Best conventional", _trial_metric(conventional_trials[0] if conventional_trials else None, "name")),
                ("Best conventional |ΔE|", _trial_metric(conventional_trials[0] if conventional_trials else None, "delta_e")),
                ("Best ADAPT", _trial_metric(adapt_trials[0] if adapt_trials else None, "name")),
                ("Best ADAPT |ΔE|", _trial_metric(adapt_trials[0] if adapt_trials else None, "delta_e")),
            ],
        ),
        (
            "Coverage",
            [
                ("Categories", categories),
                ("Trials with trajectories", sum(1 for tr in trials if tr.trajectory)),
                ("Max parameter count", max((int(tr.num_params) for tr in trials), default=0)),
            ],
        ),
        (
            "Reference and grid",
            [
                ("Exact filtered ground energy", exact_energy),
                ("t_final", args.t_final),
                ("num_times", args.num_times),
                ("trotter_steps", args.trotter_steps),
            ],
        ),
    ]

    with PdfP(str(pdf_path)) as pdf:
        render_manifest_overview_page(
            pdf,
            title=f"{model_name} cross-check — L={args.L}",
            experiment_statement=(
                f"Cross-check suite comparing {len(trials)} ansatz / VQE-mode trials against exact ED "
                "with a common propagation grid."
            ),
            sections=manifest_sections,
            notes=[
                "Machine-readable benchmark detail remains in sidecars.",
                "The full executed command appears at the end of the PDF.",
            ],
        )
        render_executive_summary_page(
            pdf,
            title="Executive summary",
            experiment_statement="Best-by-family overview before dense tables or per-trial pages.",
            sections=summary_sections,
            notes=[
                "Scoreboard page comes next, followed by per-trial trajectories and then cross-trial overlays.",
            ],
        )
        render_section_divider_page(
            pdf,
            title="Scoreboard and per-trial trajectories",
            summary="Start with the compact scoreboard, then inspect each ansatz trajectory on its own page.",
            bullets=[
                "Exact filtered energy is the common reference.",
                "Per-trial pages show fidelity, energy, and site-0 occupations.",
            ],
        )

        headers = ["Ansatz", "Category", "E_VQE", "|ΔE|", "#Params", "NFev", "Final Fid.", "Time(s)"]
        rows = []
        for tr in trials:
            fid_str = f"{tr.trajectory[-1]['fidelity']:.6f}" if tr.trajectory else "N/A"
            rows.append([
                tr.name,
                tr.category.replace("_", " "),
                f"{tr.energy:.8f}",
                f"{abs(tr.delta_e):.2e}",
                str(tr.num_params),
                str(tr.nfev),
                fid_str,
                f"{tr.elapsed_s:.1f}",
            ])
        fig_tbl, ax_tbl = plt.subplots(figsize=(12, max(3, 1 + 0.4 * len(rows))))
        render_compact_table(ax_tbl, title="VQE Scoreboard", col_labels=headers, rows=rows)
        plt.tight_layout()
        pdf.savefig(fig_tbl)
        plt.close(fig_tbl)

        for i, tr in enumerate(trials):
            if not tr.trajectory:
                continue
            times_arr = [r["time"] for r in tr.trajectory]
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            fig.suptitle(f"{tr.name}  |  E = {tr.energy:.8f}  |  |ΔE| = {abs(tr.delta_e):.2e}", fontsize=13)

            # Fidelity
            ax = axes[0]
            ax.plot(times_arr, [r["fidelity"] for r in tr.trajectory], color=colors[i % len(colors)], lw=1.5)
            ax.set_ylabel("Fidelity (Trotter vs Exact)")
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(1.0, color="gray", ls="--", lw=0.5)
            ax.grid(True, alpha=0.3)

            # Energy
            ax = axes[1]
            ax.plot(times_arr, [r["energy_exact"] for r in tr.trajectory], label="Exact", color="black", lw=1.0)
            ax.plot(times_arr, [r["energy_trotter"] for r in tr.trajectory], label="Trotter",
                    color=colors[i % len(colors)], lw=1.5, ls="--")
            ax.set_ylabel("Energy")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            # Occupation site 0
            ax = axes[2]
            ax.plot(times_arr, [r["n_up_site0_exact"] for r in tr.trajectory], label="n↑ exact",
                    color="blue", lw=1.0)
            ax.plot(times_arr, [r["n_up_site0_trotter"] for r in tr.trajectory], label="n↑ Trotter",
                    color="blue", lw=1.5, ls="--")
            ax.plot(times_arr, [r["n_dn_site0_exact"] for r in tr.trajectory], label="n↓ exact",
                    color="red", lw=1.0)
            ax.plot(times_arr, [r["n_dn_site0_trotter"] for r in tr.trajectory], label="n↓ Trotter",
                    color="red", lw=1.5, ls="--")
            ax.set_xlabel("Time")
            ax.set_ylabel("Occupation (site 0)")
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        if any(tr.trajectory for tr in trials):
            render_section_divider_page(
                pdf,
                title="Overlay appendix",
                summary="Cross-trial overlays collect the main trajectory comparisons once the per-trial pages are complete.",
                bullets=[
                    "Fidelity overlay.",
                    "Energy overlay.",
                    "Doublon overlay.",
                ],
            )

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.suptitle("Fidelity Overlay — All Ansätze", fontsize=13)
            for i, tr in enumerate(trials):
                if not tr.trajectory:
                    continue
                times_arr = [r["time"] for r in tr.trajectory]
                ax.plot(times_arr, [r["fidelity"] for r in tr.trajectory],
                        color=colors[i % len(colors)], lw=1.5, label=tr.name)
            ax.set_xlabel("Time")
            ax.set_ylabel("Fidelity")
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(1.0, color="gray", ls="--", lw=0.5)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.suptitle("Energy Overlay — All Ansätze (Trotter)", fontsize=13)
            for i, tr in enumerate(trials):
                if not tr.trajectory:
                    continue
                times_arr = [r["time"] for r in tr.trajectory]
                ax.plot(times_arr, [r["energy_trotter"] for r in tr.trajectory],
                        color=colors[i % len(colors)], lw=1.5, label=tr.name)
            # Exact reference from first trial
            ref_tr = next((t for t in trials if t.trajectory), None)
            if ref_tr:
                ax.plot([r["time"] for r in ref_tr.trajectory],
                        [r["energy_exact"] for r in ref_tr.trajectory],
                        color="black", lw=1.0, ls="-", label="Exact (ref)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Energy")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.suptitle("Total Doublon Overlay — All Ansätze", fontsize=13)
            for i, tr in enumerate(trials):
                if not tr.trajectory:
                    continue
                times_arr = [r["time"] for r in tr.trajectory]
                ax.plot(times_arr, [r["doublon_trotter"] for r in tr.trajectory],
                        color=colors[i % len(colors)], lw=1.5, ls="--", label=f"{tr.name} (Trotter)")
                ax.plot(times_arr, [r["doublon_exact"] for r in tr.trajectory],
                        color=colors[i % len(colors)], lw=0.8, ls="-", alpha=0.5, label=f"{tr.name} (Exact)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Doublon Occupation")
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        render_command_page(
            pdf,
            current_command_string(),
            script_name="pipelines/exact_bench/cross_check_suite.py",
        )

    _ai_log("pdf_complete", path=str(pdf_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-check suite: compare multiple ansätze vs exact ED.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--L", type=int, required=True, help="Number of sites")
    p.add_argument("--problem", choices=["hubbard", "hh"], default="hubbard",
                   help="Problem type (default: hubbard)")
    p.add_argument("--t", type=float, default=1.0, help="Hopping parameter")
    p.add_argument("--U", type=float, default=4.0, help="Onsite repulsion")
    p.add_argument("--dv", type=float, default=0.0, help="Staggered potential")
    p.add_argument("--ordering", default="interleaved", choices=["interleaved", "blocked"])
    p.add_argument("--boundary", default="periodic", choices=["periodic", "open"])
    p.add_argument("--seed", type=int, default=42, help="RNG seed")

    # HH-specific
    p.add_argument("--omega0", type=float, default=1.0, help="Phonon frequency (HH)")
    p.add_argument("--g-ep", type=float, default=0.5, help="Electron-phonon coupling (HH)")
    p.add_argument("--n-ph-max", type=int, default=1, help="Max phonon occupancy (HH)")
    p.add_argument("--boson-encoding", default="binary", choices=["binary", "unary"])

    # Override auto-scaled params (optional — None means use §4d table)
    p.add_argument("--vqe-reps", type=int, default=None)
    p.add_argument("--vqe-restarts", type=int, default=None)
    p.add_argument("--vqe-maxiter", type=int, default=None)
    p.add_argument("--vqe-method", type=str, default=None, choices=["COBYLA", "SLSQP", "L-BFGS-B"])
    p.add_argument("--trotter-steps", type=int, default=None)
    p.add_argument("--num-times", type=int, default=None)
    p.add_argument("--t-final", type=float, default=None)

    # Output
    p.add_argument("--output-dir", default=str(REPO_ROOT / "artifacts" / "cross_check"),
                   help="Output directory for JSON and PDF")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_cross_check(args)


if __name__ == "__main__":
    main()
