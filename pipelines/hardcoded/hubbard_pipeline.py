#!/usr/bin/env python3
"""Hardcoded-first end-to-end Hubbard pipeline.

Flow:
1) Build hardcoded Hubbard Hamiltonian (JW) from repo source-of-truth helpers.
2) Run hardcoded VQE on numpy-statevector backend (SciPy optional fallback comes from
   the notebook implementation).
3) Run temporary QPE adapter (Qiskit-only, isolated in one function).
4) Run hardcoded Suzuki-2 Trotter dynamics and exact dynamics.
5) Emit JSON + compact PDF artifact.

CFQM migration notes:
- CFQM (`--propagator cfqm4/cfqm6`) uses fixed scheme nodes `c_j`; it does not
  use midpoint/left/right sampling semantics.
- `--exact-steps-multiplier` refines only the reference propagation path and does
  not change CFQM macro-step count.
- Hubbard and Hubbard-Holstein share the same propagator interface (no HH special
  casing needed for CFQM selection).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — this file lives at pipelines/hardcoded/hubbard_pipeline.py
# REPO_ROOT is the top-level Holstein_test directory.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.reports.pdf_utils import (
    HAS_MATPLOTLIB,
    require_matplotlib,
    get_plt,
    get_PdfPages,
    render_text_page,
    render_command_page,
    current_command_string,
)
from docs.reports.report_labels import report_branch_label, report_method_label
from docs.reports.report_pages import (
    render_executive_summary_page,
    render_manifest_overview_page,
    render_section_divider_page,
)

# Module-level aliases used by the plotting body
plt = get_plt() if HAS_MATPLOTLIB else None  # type: ignore[assignment]
PdfPages = get_PdfPages() if HAS_MATPLOTLIB else type("PdfPages", (), {})  # type: ignore[misc]

from src.quantum.hartree_fock_reference_state import (
    hartree_fock_statevector,
    hubbard_holstein_reference_state,
)
from src.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
)
from src.quantum.drives_time_potential import (
    build_gaussian_sinusoid_density_drive,
    evaluate_drive_waveform,
    reference_method_name,
)
from src.quantum.pauli_actions import (
    CompiledPauliAction,
    apply_compiled_pauli as _apply_compiled_pauli_shared,
    apply_exp_term as _apply_exp_term_shared,
    compile_pauli_action_exyz as _compile_pauli_action_exyz_shared,
)
from src.quantum.time_propagation import (
    cfqm_step,
    get_cfqm_scheme,
)
from src.quantum.time_propagation.cfqm_schemes import validate_scheme

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


def _sector_basis_indices_hh(
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
    nq_total: int,
) -> np.ndarray:
    """Select basis states in the full HH Hilbert space by fermion particle-number.

    Phonon qubits are unconstrained.  Only the first 2*L (fermion) qubits
    are inspected for spin-up/spin-down counts.
    """
    dim = 1 << int(nq_total)
    n_up_want, n_dn_want = int(num_particles[0]), int(num_particles[1])
    norm_ordering = str(ordering).strip().lower()
    L = int(num_sites)

    idx_all = np.arange(dim, dtype=np.int64)

    if norm_ordering == "blocked":
        up_mask = (1 << L) - 1
        dn_mask = up_mask << L
    else:
        up_mask = int(sum(1 << (2 * q) for q in range(L)))
        dn_mask = int(sum(1 << (2 * q + 1) for q in range(L)))

    n_up_arr = np.array([bin(int(i) & int(up_mask)).count("1") for i in idx_all], dtype=np.int32)
    n_dn_arr = np.array([bin(int(i) & int(dn_mask)).count("1") for i in idx_all], dtype=np.int32)

    sector_indices = np.where((n_up_arr == n_up_want) & (n_dn_arr == n_dn_want))[0]
    if sector_indices.size == 0:
        raise ValueError(
            f"No HH basis states for sector (n_up={n_up_want}, n_dn={n_dn_want}) "
            f"with ordering='{ordering}', num_sites={num_sites}, nq_total={nq_total}."
        )
    return sector_indices


def _ground_manifold_basis_sector_filtered_hh(
    hmat: np.ndarray,
    *,
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
    nq_total: int,
    energy_tol: float,
) -> tuple[float, np.ndarray]:
    """HH-aware ground manifold: filters on fermion sector, phonon qubits free."""
    tol = float(energy_tol)
    if tol < 0.0:
        raise ValueError(f"fidelity_subspace_energy_tol must be >= 0, got {tol}.")

    sector_indices = _sector_basis_indices_hh(num_sites, num_particles, ordering, nq_total)
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
        raise RuntimeError("HH filtered ground manifold basis is empty.")
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
    return _compile_pauli_action_exyz_shared(label_exyz=label_exyz, nq=nq)


def _apply_compiled_pauli(psi: np.ndarray, action: CompiledPauliAction) -> np.ndarray:
    return _apply_compiled_pauli_shared(psi=psi, action=action)


def _apply_exp_term(
    psi: np.ndarray,
    action: CompiledPauliAction,
    coeff: complex,
    alpha: float,
    tol: float = 1e-12,
) -> np.ndarray:
    return _apply_exp_term_shared(
        psi=psi,
        action=action,
        coeff=complex(coeff),
        dt=float(alpha),
        tol=float(tol),
    )


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


def _state_from_amplitudes_qn_to_q0(
    amplitudes_qn_to_q0: dict[str, Any],
    nq_total: int,
) -> np.ndarray:
    if not isinstance(amplitudes_qn_to_q0, dict) or len(amplitudes_qn_to_q0) == 0:
        raise ValueError("Missing or empty initial_state.amplitudes_qn_to_q0 in ADAPT JSON.")
    dim = 1 << int(nq_total)
    psi = np.zeros(dim, dtype=complex)
    for bitstr, comp in amplitudes_qn_to_q0.items():
        if not isinstance(bitstr, str) or len(bitstr) != int(nq_total) or any(ch not in "01" for ch in bitstr):
            raise ValueError(f"Invalid bitstring key in ADAPT amplitudes: {bitstr!r}")
        if not isinstance(comp, dict):
            raise ValueError(f"Amplitude payload for bitstring {bitstr!r} must be a dict.")
        re_val = float(comp.get("re", 0.0))
        im_val = float(comp.get("im", 0.0))
        idx = int(bitstr, 2)
        psi[idx] = complex(re_val, im_val)
    return _normalize_state(psi)


def _load_adapt_initial_state(
    adapt_json_path: Path,
    nq_total: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not adapt_json_path.exists():
        raise FileNotFoundError(f"ADAPT input JSON not found: {adapt_json_path}")
    raw = json.loads(adapt_json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("ADAPT input JSON must be a top-level object.")
    initial_state = raw.get("initial_state")
    if not isinstance(initial_state, dict):
        raise ValueError("ADAPT input JSON missing object key: initial_state")
    amplitudes = initial_state.get("amplitudes_qn_to_q0")
    psi = _state_from_amplitudes_qn_to_q0(amplitudes, int(nq_total))
    meta = {
        "settings": raw.get("settings", {}),
        "adapt_vqe": raw.get("adapt_vqe", {}),
        "initial_state_source": initial_state.get("source"),
    }
    return psi, meta


def _validate_adapt_metadata(
    *,
    adapt_settings: dict[str, Any],
    args: argparse.Namespace,
    is_hh: bool,
    float_tol: float = 1e-10,
) -> list[str]:
    if not isinstance(adapt_settings, dict):
        return ["settings missing from ADAPT input JSON"]

    mismatches: list[str] = []

    def _cmp_scalar(field: str, expected: Any, actual: Any) -> None:
        if actual != expected:
            mismatches.append(f"{field}: expected={expected!r} adapt_json={actual!r}")

    def _cmp_float(field: str, expected: float, actual_raw: Any) -> None:
        try:
            actual = float(actual_raw)
        except Exception:
            mismatches.append(f"{field}: expected={expected!r} adapt_json={actual_raw!r}")
            return
        if abs(float(expected) - actual) > float(float_tol):
            mismatches.append(f"{field}: expected={float(expected)!r} adapt_json={actual!r}")

    _cmp_scalar("L", int(args.L), adapt_settings.get("L"))
    _cmp_scalar("problem", str(args.problem).strip().lower(), str(adapt_settings.get("problem", "")).strip().lower())
    _cmp_scalar("ordering", str(args.ordering), adapt_settings.get("ordering"))
    _cmp_scalar("boundary", str(args.boundary), adapt_settings.get("boundary"))
    _cmp_float("t", float(args.t), adapt_settings.get("t"))
    _cmp_float("u", float(args.u), adapt_settings.get("u"))
    _cmp_float("dv", float(args.dv), adapt_settings.get("dv"))

    if bool(is_hh):
        _cmp_float("omega0", float(args.omega0), adapt_settings.get("omega0"))
        _cmp_float("g_ep", float(args.g_ep), adapt_settings.get("g_ep"))
        _cmp_scalar("n_ph_max", int(args.n_ph_max), adapt_settings.get("n_ph_max"))
        _cmp_scalar("boson_encoding", str(args.boson_encoding), adapt_settings.get("boson_encoding"))

    return mismatches


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
        "HubbardHolsteinLayerwiseAnsatz",
        "HubbardHolsteinTermwiseAnsatz",
        "HubbardHolsteinPhysicalTermwiseAnsatz",
        "exact_ground_energy_sector",
        "exact_ground_energy_sector_hh",
        "hubbard_holstein_reference_state",
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
    energy_backend: str,
    vqe_progress_every_s: float = 60.0,
    ansatz_name: str,
    spsa_a: float = 0.2,
    spsa_c: float = 0.1,
    spsa_alpha: float = 0.602,
    spsa_gamma: float = 0.101,
    spsa_A: float = 10.0,
    spsa_avg_last: int = 0,
    spsa_eval_repeats: int = 1,
    spsa_eval_agg: str = "mean",
    # --- HH-specific (ignored when problem=hubbard) ---
    problem: str = "hubbard",
    omega0: float = 0.0,
    g_ep: float = 0.0,
    n_ph_max: int = 1,
    boson_encoding: str = "binary",
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
        energy_backend=str(energy_backend),
        vqe_progress_every_s=float(vqe_progress_every_s),
        ansatz=str(ansatz_name),
        problem=str(problem),
    )
    ns = _load_hardcoded_vqe_namespace()
    ansatz_name_s = str(ansatz_name).strip().lower()
    problem_s = str(problem).strip().lower()

    valid_ansatzes = {"uccsd", "hva", "hh_hva", "hh_hva_tw", "hh_hva_ptw"}
    if ansatz_name_s not in valid_ansatzes:
        raise ValueError(f"Unsupported --vqe-ansatz '{ansatz_name}'. Expected one of: {sorted(valid_ansatzes)}.")

    if ansatz_name_s in ("hh_hva", "hh_hva_tw", "hh_hva_ptw") and problem_s != "hh":
        raise ValueError(f"--vqe-ansatz {ansatz_name_s} requires --problem hh.")

    is_hh = problem_s == "hh"

    num_particles = tuple(ns["half_filled_num_particles"](int(num_sites)))

    if is_hh:
        psi_ref = np.asarray(
            ns["hubbard_holstein_reference_state"](
                dims=int(num_sites),
                num_particles=num_particles,
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                indexing=ordering,
            ),
            dtype=complex,
        )
        nq = int(round(math.log2(psi_ref.size)))
    else:
        hf_bits = str(ns["hartree_fock_bitstring"](
            n_sites=int(num_sites), num_particles=num_particles, indexing=ordering,
        ))
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
    elif ansatz_name_s == "hva":
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
    elif ansatz_name_s == "hh_hva":
        ansatz = ns["HubbardHolsteinLayerwiseAnsatz"](
            dims=int(num_sites),
            J=float(hopping_t),
            U=float(onsite_u),
            omega0=float(omega0),
            g=float(g_ep),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            reps=int(reps),
            repr_mode="JW",
            indexing=ordering,
            pbc=(str(boundary).strip().lower() == "periodic"),
        )
        method_name = "hardcoded_hh_hva_layerwise_statevector"
    elif ansatz_name_s == "hh_hva_tw":
        ansatz = ns["HubbardHolsteinTermwiseAnsatz"](
            dims=int(num_sites),
            J=float(hopping_t),
            U=float(onsite_u),
            omega0=float(omega0),
            g=float(g_ep),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            reps=int(reps),
            repr_mode="JW",
            indexing=ordering,
            pbc=(str(boundary).strip().lower() == "periodic"),
        )
        method_name = "hardcoded_hh_hva_termwise_statevector"
    elif ansatz_name_s == "hh_hva_ptw":
        ansatz = ns["HubbardHolsteinPhysicalTermwiseAnsatz"](
            dims=int(num_sites),
            J=float(hopping_t),
            U=float(onsite_u),
            omega0=float(omega0),
            g=float(g_ep),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            reps=int(reps),
            repr_mode="JW",
            indexing=ordering,
            pbc=(str(boundary).strip().lower() == "periodic"),
        )
        method_name = "hardcoded_hh_hva_physical_termwise_statevector"
    else:
        raise ValueError(f"Unsupported ansatz: {ansatz_name_s}")

    progress_event_map = {
        "run_start": "hardcoded_vqe_run_start",
        "restart_start": "hardcoded_vqe_restart_start",
        "heartbeat": "hardcoded_vqe_heartbeat",
        "restart_end": "hardcoded_vqe_restart_end",
        "run_end": "hardcoded_vqe_run_end",
    }

    def _vqe_progress_logger(payload: dict[str, Any]) -> None:
        raw_event = str(payload.get("event", ""))
        mapped_event = progress_event_map.get(raw_event)
        if mapped_event is None:
            return
        fields = {k: v for k, v in payload.items() if k != "event"}
        _ai_log(mapped_event, **fields)

    result = ns["vqe_minimize"](
        h_poly,
        ansatz,
        psi_ref,
        restarts=int(restarts),
        seed=int(seed),
        maxiter=int(maxiter),
        method=str(method),
        energy_backend=str(energy_backend),
        spsa_a=float(spsa_a),
        spsa_c=float(spsa_c),
        spsa_alpha=float(spsa_alpha),
        spsa_gamma=float(spsa_gamma),
        spsa_A=float(spsa_A),
        spsa_avg_last=int(spsa_avg_last),
        spsa_eval_repeats=int(spsa_eval_repeats),
        spsa_eval_agg=str(spsa_eval_agg),
        progress_logger=_vqe_progress_logger,
        progress_every_s=float(vqe_progress_every_s),
        progress_label="hardcoded_vqe",
    )

    theta = np.asarray(result.theta, dtype=float)
    psi_vqe = np.asarray(ansatz.prepare_state(theta, psi_ref), dtype=complex).reshape(-1)
    psi_vqe = _normalize_state(psi_vqe)

    if is_hh:
        exact_filtered_energy = float(
            ns["exact_ground_energy_sector_hh"](
                h_poly,
                num_sites=int(num_sites),
                num_particles=num_particles,
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                indexing=ordering,
            )
        )
    else:
        exact_filtered_energy = float(
            ns["exact_ground_energy_sector"](
                h_poly,
                num_sites=int(num_sites),
                num_particles=num_particles,
                indexing=ordering,
            )
        )

    hf_bits_display = "N/A (HH ref state)"
    if not is_hh:
        hf_bits_display = str(ns["hartree_fock_bitstring"](
            n_sites=int(num_sites), num_particles=num_particles, indexing=ordering,
        ))

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
        "energy_backend": str(energy_backend),
        "optimal_point": [float(x) for x in theta.tolist()],
        "hf_bitstring_qn_to_q0": hf_bits_display,
    }
    if str(method).strip().lower() == "spsa":
        payload["spsa"] = {
            "a": float(spsa_a),
            "c": float(spsa_c),
            "alpha": float(spsa_alpha),
            "gamma": float(spsa_gamma),
            "A": float(spsa_A),
            "avg_last": int(spsa_avg_last),
            "eval_repeats": int(spsa_eval_repeats),
            "eval_agg": str(spsa_eval_agg),
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
        energy_backend=str(energy_backend),
        success=True,
        energy=float(result.energy),
        exact_filtered_energy=float(exact_filtered_energy),
        best_restart=int(getattr(result, "best_restart", 0)),
        nfev=int(getattr(result, "nfev", 0)),
        nit=int(getattr(result, "nit", 0)),
        elapsed_sec=round(time.perf_counter() - t0, 6),
    )
    return payload, psi_vqe


def _run_internal_adapt_paop(
    *,
    h_poly: Any,
    num_sites: int,
    ordering: str,
    problem: str,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    boson_encoding: str,
    adapt_pool: str | None,
    adapt_max_depth: int,
    adapt_eps_grad: float,
    adapt_eps_energy: float,
    adapt_maxiter: int,
    adapt_seed: int,
    adapt_allow_repeats: bool,
    adapt_finite_angle_fallback: bool,
    adapt_finite_angle: float,
    adapt_finite_angle_min_improvement: float,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    adapt_disable_hh_seed: bool,
    psi_ref_override: np.ndarray | None = None,
    adapt_reopt_policy: str = "append_only",
    adapt_window_size: int = 3,
    adapt_window_topk: int = 0,
    adapt_full_refit_every: int = 0,
    adapt_final_full_refit: bool = True,
    adapt_continuation_mode: str = "legacy",
    phase1_lambda_F: float = 1.0,
    phase1_lambda_compile: float = 0.05,
    phase1_lambda_measure: float = 0.02,
    phase1_lambda_leak: float = 0.0,
    phase1_score_z_alpha: float = 0.0,
    phase1_probe_max_positions: int = 6,
    phase1_plateau_patience: int = 2,
    phase1_trough_margin_ratio: float = 1.0,
    phase1_prune_enabled: bool = True,
    phase1_prune_fraction: float = 0.25,
    phase1_prune_max_candidates: int = 6,
    phase1_prune_max_regression: float = 1e-8,
    phase3_motif_source_json: Path | None = None,
    phase3_symmetry_mitigation_mode: str = "off",
    phase3_enable_rescue: bool = False,
    phase3_lifetime_cost_mode: str = "phase3_v1",
    phase3_runtime_split_mode: str = "off",
) -> tuple[dict[str, Any], np.ndarray]:
    from pipelines.hardcoded import adapt_pipeline as adapt_mod

    return adapt_mod._run_hardcoded_adapt_vqe(
        h_poly=h_poly,
        num_sites=int(num_sites),
        ordering=str(ordering),
        problem=str(problem),
        adapt_pool=(str(adapt_pool) if adapt_pool is not None else None),
        t=float(t),
        u=float(u),
        dv=float(dv),
        boundary=str(boundary),
        omega0=float(omega0),
        g_ep=float(g_ep),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        max_depth=int(adapt_max_depth),
        eps_grad=float(adapt_eps_grad),
        eps_energy=float(adapt_eps_energy),
        maxiter=int(adapt_maxiter),
        seed=int(adapt_seed),
        allow_repeats=bool(adapt_allow_repeats),
        finite_angle_fallback=bool(adapt_finite_angle_fallback),
        finite_angle=float(adapt_finite_angle),
        finite_angle_min_improvement=float(adapt_finite_angle_min_improvement),
        paop_r=int(paop_r),
        paop_split_paulis=bool(paop_split_paulis),
        paop_prune_eps=float(paop_prune_eps),
        paop_normalization=str(paop_normalization),
        disable_hh_seed=bool(adapt_disable_hh_seed),
        psi_ref_override=psi_ref_override,
        adapt_reopt_policy=str(adapt_reopt_policy),
        adapt_window_size=int(adapt_window_size),
        adapt_window_topk=int(adapt_window_topk),
        adapt_full_refit_every=int(adapt_full_refit_every),
        adapt_final_full_refit=bool(adapt_final_full_refit),
        adapt_continuation_mode=str(adapt_continuation_mode),
        phase1_lambda_F=float(phase1_lambda_F),
        phase1_lambda_compile=float(phase1_lambda_compile),
        phase1_lambda_measure=float(phase1_lambda_measure),
        phase1_lambda_leak=float(phase1_lambda_leak),
        phase1_score_z_alpha=float(phase1_score_z_alpha),
        phase1_probe_max_positions=int(phase1_probe_max_positions),
        phase1_plateau_patience=int(phase1_plateau_patience),
        phase1_trough_margin_ratio=float(phase1_trough_margin_ratio),
        phase1_prune_enabled=bool(phase1_prune_enabled),
        phase1_prune_fraction=float(phase1_prune_fraction),
        phase1_prune_max_candidates=int(phase1_prune_max_candidates),
        phase1_prune_max_regression=float(phase1_prune_max_regression),
        phase3_motif_source_json=(Path(phase3_motif_source_json) if phase3_motif_source_json is not None else None),
        phase3_symmetry_mitigation_mode=str(phase3_symmetry_mitigation_mode),
        phase3_enable_rescue=bool(phase3_enable_rescue),
        phase3_lifetime_cost_mode=str(phase3_lifetime_cost_mode),
        phase3_runtime_split_mode=str(phase3_runtime_split_mode),
    )


def _run_qpe_adapter_qiskit(
    *,
    coeff_map_exyz: dict[str, complex],
    psi_init: np.ndarray,
    eval_qubits: int,
    shots: int,
    seed: int,
) -> dict[str, Any]:
    """Delegate to the temporary Qiskit-backed hardcoded QPE shim."""
    from pipelines.hardcoded.qpe_qiskit_shim import run_qpe_adapter_qiskit
    return run_qpe_adapter_qiskit(
        coeff_map_exyz=coeff_map_exyz,
        psi_init=psi_init,
        eval_qubits=eval_qubits,
        shots=shots,
        seed=seed,
    )


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
        REPO_ROOT / "Tests" / "hubbard_jw_L4_L5_periodic_blocked_qiskit.json",
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
    psi0_legacy_trot: np.ndarray | None = None,
    psi0_paop_trot: np.ndarray | None = None,
    psi0_hva_trot: np.ndarray | None = None,
    legacy_branch_label: str = "vqe",
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
    propagator: str = "cfqm4",
    cfqm_stage_exp: str = "expm_multiply_sparse",
    cfqm_coeff_drop_abs_tol: float = 0.0,
    cfqm_normalize: bool = False,
    psi0_ansatz_trot: np.ndarray | None = None,
) -> tuple[list[dict[str, float]], list[np.ndarray]]:
    if int(suzuki_order) != 2:
        raise ValueError("This script currently supports suzuki_order=2 only.")
    if int(trotter_steps) <= 0:
        raise ValueError("trotter_steps must be >= 1.")

    # Backward-compatibility: legacy callers may provide only psi0_ansatz_trot.
    if psi0_legacy_trot is None:
        if psi0_ansatz_trot is None:
            raise ValueError("Provide psi0_legacy_trot (or compatibility alias psi0_ansatz_trot).")
        psi0_legacy_trot = np.asarray(psi0_ansatz_trot, dtype=complex).reshape(-1)
    if psi0_paop_trot is None:
        psi0_paop_trot = np.array(psi0_legacy_trot, copy=True)
    if psi0_hva_trot is None:
        psi0_hva_trot = np.array(psi0_legacy_trot, copy=True)

    nq = int(round(math.log2(hmat.shape[0])))
    evals, evecs = np.linalg.eigh(hmat)
    evecs_dag = np.conjugate(evecs).T

    propagator_key = str(propagator).strip().lower()
    if propagator_key not in {"suzuki2", "piecewise_exact", "cfqm4", "cfqm6"}:
        raise ValueError(
            "propagator must be one of {'suzuki2','piecewise_exact','cfqm4','cfqm6'}."
        )

    compiled: dict[str, CompiledPauliAction] | None = None
    if propagator_key == "suzuki2":
        compiled = {lbl: _compile_pauli_action(lbl, nq) for lbl in ordered_labels_exyz}

    cfqm_scheme: dict[str, Any] | None = None
    if propagator_key in {"cfqm4", "cfqm6"}:
        cfqm_scheme = get_cfqm_scheme(propagator_key)
        validate_scheme(cfqm_scheme)

    times = np.linspace(0.0, float(t_final), int(num_times))
    n_times = int(times.size)
    stride = max(1, n_times // 20)
    t0 = time.perf_counter()
    basis_v0 = np.asarray(fidelity_subspace_basis_v0, dtype=complex)
    if basis_v0.ndim != 2 or basis_v0.shape[0] != psi0_legacy_trot.size:
        raise ValueError("fidelity_subspace_basis_v0 must have shape (dim, k) with matching dim.")
    if basis_v0.shape[1] <= 0:
        raise ValueError("fidelity_subspace_basis_v0 must contain at least one basis vector.")

    has_drive = drive_coeff_provider_exyz is not None
    cfqm_unknown_label_warned_labels: set[str] = set()

    if (
        propagator_key in {"cfqm4", "cfqm6"}
        and str(drive_time_sampling).strip().lower() != "midpoint"
    ):
        warnings.warn(
            "CFQM ignores midpoint/left/right sampling; uses fixed scheme nodes c_j.",
            RuntimeWarning,
            stacklevel=2,
        )

    if (
        propagator_key in {"cfqm4", "cfqm6"}
        and str(cfqm_stage_exp).strip().lower() == "pauli_suzuki2"
    ):
        warnings.warn(
            "Inner Suzuki-2 makes overall method 2nd order; use expm_multiply_sparse/dense_expm for true CFQM order.",
            RuntimeWarning,
            stacklevel=2,
        )

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
        propagator=str(propagator_key),
        drive_enabled=has_drive,
        ground_subspace_dimension=int(basis_v0.shape[1]),
        fidelity_subspace_energy_tol=float(fidelity_subspace_energy_tol),
        fidelity_selection_rule="E <= E0 + tol",
    )

    rows: list[dict[str, float]] = []
    exact_states: list[np.ndarray] = []

    for idx, time_val in enumerate(times):
        t = float(time_val)

        def _exact_from_initial(psi0_branch: np.ndarray) -> np.ndarray:
            if not has_drive:
                psi_out = evecs @ (np.exp(-1j * evals * t) * (evecs_dag @ psi0_branch))
                return _normalize_state(psi_out)
            return _evolve_piecewise_exact(
                psi0=psi0_branch,
                hmat_static=hmat,
                drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                time_value=t,
                trotter_steps=reference_steps,
                t0=float(drive_t0),
                time_sampling=str(drive_time_sampling),
            )

        def _trotter_from_initial(psi0_branch: np.ndarray) -> np.ndarray:
            if propagator_key == "suzuki2":
                assert compiled is not None
                return _evolve_trotter_suzuki2_absolute(
                    psi0_branch,
                    ordered_labels_exyz,
                    coeff_map_exyz,
                    compiled,
                    t,
                    int(trotter_steps),
                    drive_coeff_provider_exyz=drive_coeff_provider_exyz,
                    t0=float(drive_t0),
                    time_sampling=str(drive_time_sampling),
                )

            if propagator_key == "piecewise_exact":
                provider = drive_coeff_provider_exyz
                if provider is None:
                    provider = lambda _t: {}
                return _evolve_piecewise_exact(
                    psi0=psi0_branch,
                    hmat_static=hmat,
                    drive_coeff_provider_exyz=provider,
                    time_value=t,
                    trotter_steps=int(trotter_steps),
                    t0=float(drive_t0),
                    time_sampling=str(drive_time_sampling),
                )

            assert cfqm_scheme is not None
            n_steps = int(trotter_steps)
            if n_steps < 1:
                raise ValueError(f"CFQM requires n_steps >= 1; got n_steps={n_steps}.")
            if abs(t) <= 1e-15:
                return np.array(psi0_branch, copy=True)

            dt_macro = float(t) / float(n_steps)
            if not np.isfinite(dt_macro) or dt_macro <= 0.0:
                raise ValueError(
                    f"CFQM requires dt > 0; got dt={dt_macro} from t={t} and n_steps={n_steps}."
                )
            psi_out = np.array(psi0_branch, copy=True)
            cfqm_cfg: dict[str, Any] = {
                "backend": str(cfqm_stage_exp),
                "coeff_drop_abs_tol": float(cfqm_coeff_drop_abs_tol),
                "normalize": bool(cfqm_normalize),
                "sparse_min_dim": int(_EXPM_SPARSE_MIN_DIM),
                # Warning emitted once at trajectory setup.
                "emit_inner_order_warning": False,
                # Unknown drive labels: warn once per label then ignore.
                "unknown_label_policy": "warn_ignore",
                "unknown_label_warn_abs_tol": 1e-14,
                "unknown_label_warned_labels": cfqm_unknown_label_warned_labels,
            }
            for step_idx in range(n_steps):
                t_abs_step = float(drive_t0) + float(step_idx) * dt_macro
                psi_out = cfqm_step(
                    psi=psi_out,
                    t_abs=t_abs_step,
                    dt=dt_macro,
                    static_coeff_map=coeff_map_exyz,
                    drive_coeff_provider=drive_coeff_provider_exyz,
                    ordered_labels=ordered_labels_exyz,
                    scheme=cfqm_scheme,
                    config=cfqm_cfg,
                )
            return np.asarray(psi_out, dtype=complex)

        # --- exact / reference propagation (filtered-sector GS branch) ---
        psi_exact = _exact_from_initial(psi0_exact_ref)
        psi_exact_legacy = _exact_from_initial(psi0_legacy_trot)
        psi_exact_paop = _exact_from_initial(psi0_paop_trot)
        psi_exact_hva = _exact_from_initial(psi0_hva_trot)

        psi_trot_legacy = _trotter_from_initial(psi0_legacy_trot)
        psi_trot_paop = _trotter_from_initial(psi0_paop_trot)
        psi_trot_hva = _trotter_from_initial(psi0_hva_trot)

        # --- norm-drift diagnostic ---
        norm_before = float(np.linalg.norm(psi_trot_legacy))
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
            basis_t = np.zeros((psi_trot_legacy.size, basis_v0.shape[1]), dtype=complex)
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
        fidelity = _projector_fidelity_from_basis(basis_t_orth, psi_trot_legacy)
        fidelity_paop = _projector_fidelity_from_basis(basis_t_orth, psi_trot_paop)
        fidelity_hva = _projector_fidelity_from_basis(basis_t_orth, psi_trot_hva)

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
        else:
            hmat_total_t = hmat
            energy_total_exact = _expectation_hamiltonian(psi_exact, hmat_total_t)

        def _branch_observables(psi_branch: np.ndarray) -> dict[str, Any]:
            n_up_site, n_dn_site, doublon = _site_resolved_number_observables(
                psi_branch,
                num_sites,
                ordering,
            )
            n_site = n_up_site + n_dn_site
            return {
                "n_up_site": n_up_site,
                "n_dn_site": n_dn_site,
                "n_site": n_site,
                "n_up_site0": float(n_up_site[0]) if n_up_site.size > 0 else float("nan"),
                "n_dn_site0": float(n_dn_site[0]) if n_dn_site.size > 0 else float("nan"),
                "doublon": float(doublon),
                "staggered": _staggered_order(n_site),
                "energy_static": _expectation_hamiltonian(psi_branch, hmat),
                "energy_total": _expectation_hamiltonian(psi_branch, hmat_total_t),
            }

        obs_exact_gs = _branch_observables(psi_exact)
        obs_exact_legacy = _branch_observables(psi_exact_legacy)
        obs_trot_legacy = _branch_observables(psi_trot_legacy)
        obs_exact_paop = _branch_observables(psi_exact_paop)
        obs_trot_paop = _branch_observables(psi_trot_paop)
        obs_exact_hva = _branch_observables(psi_exact_hva)
        obs_trot_hva = _branch_observables(psi_trot_hva)

        rows.append(
            {
                "time": t,
                "fidelity": fidelity,
                "fidelity_paop_trotter": fidelity_paop,
                "fidelity_hva_trotter": fidelity_hva,
                "legacy_branch_label": str(legacy_branch_label),
                "energy_static_exact": obs_exact_gs["energy_static"],
                "energy_static_exact_ansatz": obs_exact_legacy["energy_static"],
                "energy_static_trotter": obs_trot_legacy["energy_static"],
                "energy_static_exact_paop": obs_exact_paop["energy_static"],
                "energy_static_trotter_paop": obs_trot_paop["energy_static"],
                "energy_static_exact_hva": obs_exact_hva["energy_static"],
                "energy_static_trotter_hva": obs_trot_hva["energy_static"],
                "energy_total_exact": energy_total_exact,
                "energy_total_exact_ansatz": obs_exact_legacy["energy_total"],
                "energy_total_trotter": obs_trot_legacy["energy_total"],
                "energy_total_exact_paop": obs_exact_paop["energy_total"],
                "energy_total_trotter_paop": obs_trot_paop["energy_total"],
                "energy_total_exact_hva": obs_exact_hva["energy_total"],
                "energy_total_trotter_hva": obs_trot_hva["energy_total"],
                "n_up_site0_exact": obs_exact_gs["n_up_site0"],
                "n_up_site0_exact_ansatz": obs_exact_legacy["n_up_site0"],
                "n_up_site0_trotter": obs_trot_legacy["n_up_site0"],
                "n_dn_site0_exact": obs_exact_gs["n_dn_site0"],
                "n_dn_site0_exact_ansatz": obs_exact_legacy["n_dn_site0"],
                "n_dn_site0_trotter": obs_trot_legacy["n_dn_site0"],
                "n_up_site0_exact_paop": obs_exact_paop["n_up_site0"],
                "n_up_site0_trotter_paop": obs_trot_paop["n_up_site0"],
                "n_up_site0_exact_hva": obs_exact_hva["n_up_site0"],
                "n_up_site0_trotter_hva": obs_trot_hva["n_up_site0"],
                "n_dn_site0_exact_paop": obs_exact_paop["n_dn_site0"],
                "n_dn_site0_trotter_paop": obs_trot_paop["n_dn_site0"],
                "n_dn_site0_exact_hva": obs_exact_hva["n_dn_site0"],
                "n_dn_site0_trotter_hva": obs_trot_hva["n_dn_site0"],
                "n_site_exact": [float(x) for x in obs_exact_gs["n_site"].tolist()],
                "n_site_exact_ansatz": [float(x) for x in obs_exact_legacy["n_site"].tolist()],
                "n_site_trotter": [float(x) for x in obs_trot_legacy["n_site"].tolist()],
                "n_site_exact_paop": [float(x) for x in obs_exact_paop["n_site"].tolist()],
                "n_site_trotter_paop": [float(x) for x in obs_trot_paop["n_site"].tolist()],
                "n_site_exact_hva": [float(x) for x in obs_exact_hva["n_site"].tolist()],
                "n_site_trotter_hva": [float(x) for x in obs_trot_hva["n_site"].tolist()],
                "staggered_exact": obs_exact_gs["staggered"],
                "staggered_exact_ansatz": obs_exact_legacy["staggered"],
                "staggered_trotter": obs_trot_legacy["staggered"],
                "staggered_exact_paop": obs_exact_paop["staggered"],
                "staggered_trotter_paop": obs_trot_paop["staggered"],
                "staggered_exact_hva": obs_exact_hva["staggered"],
                "staggered_trotter_hva": obs_trot_hva["staggered"],
                "doublon_exact": obs_exact_gs["doublon"],
                "doublon_exact_ansatz": obs_exact_legacy["doublon"],
                "doublon_trotter": obs_trot_legacy["doublon"],
                "doublon_exact_paop": obs_exact_paop["doublon"],
                "doublon_trotter_paop": obs_trot_paop["doublon"],
                "doublon_exact_hva": obs_exact_hva["doublon"],
                "doublon_trotter_hva": obs_trot_hva["doublon"],
                "doublon_avg_exact": float(obs_exact_gs["doublon"] / float(num_sites)),
                "doublon_avg_exact_ansatz": float(obs_exact_legacy["doublon"] / float(num_sites)),
                "doublon_avg_trotter": float(obs_trot_legacy["doublon"] / float(num_sites)),
                "doublon_avg_exact_paop": float(obs_exact_paop["doublon"] / float(num_sites)),
                "doublon_avg_trotter_paop": float(obs_trot_paop["doublon"] / float(num_sites)),
                "doublon_avg_exact_hva": float(obs_exact_hva["doublon"] / float(num_sites)),
                "doublon_avg_trotter_hva": float(obs_trot_hva["doublon"] / float(num_sites)),
                "n_up_site_exact": [float(x) for x in obs_exact_gs["n_up_site"].tolist()],
                "n_up_site_exact_ansatz": [float(x) for x in obs_exact_legacy["n_up_site"].tolist()],
                "n_up_site_trotter": [float(x) for x in obs_trot_legacy["n_up_site"].tolist()],
                "n_up_site_exact_paop": [float(x) for x in obs_exact_paop["n_up_site"].tolist()],
                "n_up_site_trotter_paop": [float(x) for x in obs_trot_paop["n_up_site"].tolist()],
                "n_up_site_exact_hva": [float(x) for x in obs_exact_hva["n_up_site"].tolist()],
                "n_up_site_trotter_hva": [float(x) for x in obs_trot_hva["n_up_site"].tolist()],
                "n_dn_site_exact": [float(x) for x in obs_exact_gs["n_dn_site"].tolist()],
                "n_dn_site_exact_ansatz": [float(x) for x in obs_exact_legacy["n_dn_site"].tolist()],
                "n_dn_site_trotter": [float(x) for x in obs_trot_legacy["n_dn_site"].tolist()],
                "n_dn_site_exact_paop": [float(x) for x in obs_exact_paop["n_dn_site"].tolist()],
                "n_dn_site_trotter_paop": [float(x) for x in obs_trot_paop["n_dn_site"].tolist()],
                "n_dn_site_exact_hva": [float(x) for x in obs_exact_hva["n_dn_site"].tolist()],
                "n_dn_site_trotter_hva": [float(x) for x in obs_trot_hva["n_dn_site"].tolist()],
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


def _write_pipeline_pdf(pdf_path: Path, payload: dict[str, Any], run_command: str) -> None:
    require_matplotlib()
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
        ax.set_yticks([float(i) for i in range(len(labels))])
        ax.set_yticklabels(labels, fontsize=7)
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
    fid_paop = arr_optional("fidelity_paop_trotter", fallback=fid)
    fid_hva = arr_optional("fidelity_hva_trotter", fallback=fid)

    e_exact_paop = arr_optional("energy_static_exact_paop", fallback=e_exact_ans)
    e_trot_paop = arr_optional("energy_static_trotter_paop", fallback=e_trot)
    e_exact_hva = arr_optional("energy_static_exact_hva", fallback=e_exact_ans)
    e_trot_hva = arr_optional("energy_static_trotter_hva", fallback=e_trot)
    e_total_exact = arr("energy_total_exact")
    e_total_exact_paop = arr_optional("energy_total_exact_paop", fallback=arr_optional("energy_total_exact_ansatz", fallback=e_total_exact))
    e_total_trot_paop = arr_optional("energy_total_trotter_paop", fallback=arr("energy_total_trotter"))
    e_total_exact_hva = arr_optional("energy_total_exact_hva", fallback=arr_optional("energy_total_exact_ansatz", fallback=e_total_exact))
    e_total_trot_hva = arr_optional("energy_total_trotter_hva", fallback=arr("energy_total_trotter"))

    d_exact_paop = arr_optional("doublon_exact_paop", fallback=d_exact_ans)
    d_trot_paop = arr_optional("doublon_trotter_paop", fallback=d_trot)
    d_exact_hva = arr_optional("doublon_exact_hva", fallback=d_exact_ans)
    d_trot_hva = arr_optional("doublon_trotter_hva", fallback=d_trot)

    stg_exact_paop = arr_optional("staggered_exact_paop", fallback=stg_exact_ans)
    stg_trot_paop = arr_optional("staggered_trotter_paop", fallback=stg_trot)
    stg_exact_hva = arr_optional("staggered_exact_hva", fallback=stg_exact_ans)
    stg_trot_hva = arr_optional("staggered_trotter_hva", fallback=stg_trot)

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
            np.abs(e_trot_paop - e_exact_paop),
            np.abs(e_exact_paop - e_exact),
            np.abs(e_trot_hva - e_exact_hva),
            np.abs(e_exact_hva - e_exact),
            np.abs(d_trot_paop - d_exact_paop),
            np.abs(d_exact_paop - d_exact),
            np.abs(d_trot_hva - d_exact_hva),
            np.abs(d_exact_hva - d_exact),
            np.abs(stg_trot_paop - stg_exact_paop),
            np.abs(stg_exact_paop - stg_exact),
            np.abs(stg_trot_hva - stg_exact_hva),
            np.abs(stg_exact_hva - stg_exact),
        ]
    )
    err_scalar_labels = [
        "|E_trot_paop - E_exact_paop|",
        "|E_exact_paop - E_exact_gs|",
        "|E_trot_hva - E_exact_hva|",
        "|E_exact_hva - E_exact_gs|",
        "|D_trot_paop - D_exact_paop|",
        "|D_exact_paop - D_exact_gs|",
        "|D_trot_hva - D_exact_hva|",
        "|D_exact_hva - D_exact_gs|",
        "|S_trot_paop - S_exact_paop|",
        "|S_exact_paop - S_exact_gs|",
        "|S_trot_hva - S_exact_hva|",
        "|S_exact_hva - S_exact_gs|",
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
    vqe_optimizer_method = payload.get("vqe", {}).get("optimizer_method", settings.get("vqe_method", "unknown"))
    vqe_spsa_cfg = payload.get("vqe", {}).get("spsa", settings.get("vqe_spsa", {}))
    adapt_import = payload.get("adapt_import", {})
    has_adapt_import = isinstance(adapt_import, dict) and bool(adapt_import)
    branch_meta = payload.get("ansatz_branches", {}) if isinstance(payload.get("ansatz_branches", {}), dict) else {}
    legacy_plot_label = str(branch_meta.get("legacy_selected_branch", "selected")).strip().lower()
    problem_label = str(settings.get("problem", "hubbard")).strip().lower()
    model_name = "Hubbard-Holstein" if problem_label == "hh" else "Hubbard"
    hh_block = settings.get("holstein", {})
    drive_block = settings.get("drive")
    drive_enabled = isinstance(drive_block, dict) and bool(drive_block.get("enabled", False))
    run_mode = "drive-enabled" if drive_enabled else "static"
    legacy_branch_display = report_branch_label(legacy_plot_label)
    propagator_label = report_method_label(str(settings.get("propagator", "suzuki2")))
    vqe_abs_delta = abs(vqe_val - gs_exact_filtered) if (np.isfinite(vqe_val) and np.isfinite(gs_exact_filtered)) else np.nan

    manifest_sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "Model and regime",
            [
                ("Model family", model_name),
                ("Problem", problem_label),
                ("L", settings.get("L")),
                ("Boundary", settings.get("boundary")),
                ("Ordering", settings.get("ordering")),
                ("Drive enabled", drive_enabled),
            ],
        ),
        (
            "Method chain",
            [
                ("Ansatz type", ansatz_label),
                ("Optimizer", vqe_optimizer_method),
                ("Internal VQE method", vqe_method),
                ("Propagator", propagator_label),
                ("Initial state source", settings.get("initial_state_source")),
                ("Selected propagated branch", legacy_branch_display),
                ("Branch order", "Exact filtered GS, Exact PAOP, Trotter PAOP, Exact HVA, Trotter HVA"),
            ],
        ),
        (
            "Core physical parameters",
            [
                ("t", settings.get("t")),
                ("U", settings.get("u")),
                ("dv", settings.get("dv")),
            ],
        ),
        (
            "Trajectory settings",
            [
                ("t_final", settings.get("t_final")),
                ("num_times", settings.get("num_times")),
                ("trotter_steps", settings.get("trotter_steps")),
                ("Suzuki order", settings.get("suzuki_order")),
                ("Term order", settings.get("term_order")),
            ],
        ),
    ]
    if str(vqe_optimizer_method).strip().lower() == "spsa" and isinstance(vqe_spsa_cfg, dict):
        manifest_sections.append(
            (
                "SPSA settings",
                [
                    ("a", vqe_spsa_cfg.get("a")),
                    ("c", vqe_spsa_cfg.get("c")),
                    ("A", vqe_spsa_cfg.get("A")),
                    ("alpha", vqe_spsa_cfg.get("alpha")),
                    ("gamma", vqe_spsa_cfg.get("gamma")),
                    ("eval repeats", vqe_spsa_cfg.get("eval_repeats")),
                    ("eval aggregate", vqe_spsa_cfg.get("eval_agg")),
                    ("average last", vqe_spsa_cfg.get("avg_last")),
                ],
            )
        )
    if drive_enabled and isinstance(drive_block, dict):
        manifest_sections.append(
            (
                "Drive settings",
                [
                    ("A", drive_block.get("A")),
                    ("omega", drive_block.get("omega")),
                    ("tbar", drive_block.get("tbar")),
                    ("phi", drive_block.get("phi")),
                    ("t0", drive_block.get("t0")),
                    ("Pattern", drive_block.get("pattern")),
                    ("Time sampling", drive_block.get("time_sampling")),
                    ("Exact-steps multiplier", drive_block.get("reference_steps_multiplier")),
                ],
            )
        )
    if problem_label == "hh":
        manifest_sections.append(
            (
                "Hubbard-Holstein parameters",
                [
                    ("omega0", hh_block.get("omega0")),
                    ("g_ep", hh_block.get("g_ep")),
                    ("n_ph_max", hh_block.get("n_ph_max")),
                    ("Boson encoding", hh_block.get("boson_encoding")),
                    ("nq_fermion", hh_block.get("nq_fermion")),
                    ("nq_phonon", hh_block.get("nq_phonon")),
                    ("nq_total", hh_block.get("nq_total")),
                ],
            )
        )
    if branch_meta:
        paop_meta = branch_meta.get("paop", {})
        hva_meta = branch_meta.get("hva", {})
        branch_rows: list[tuple[str, Any]] = [
            ("PAOP source", paop_meta.get("source")),
            ("PAOP pool", paop_meta.get("pool_type")),
            ("PAOP depth", paop_meta.get("ansatz_depth")),
            ("HVA source", hva_meta.get("source")),
            ("HVA ansatz", hva_meta.get("ansatz")),
            ("HVA VQE success", hva_meta.get("vqe_success")),
        ]
        if settings.get("adapt_ref_source") is not None:
            branch_rows.append(("ADAPT reference source", settings.get("adapt_ref_source")))
        manifest_sections.append(("Branch provenance", branch_rows))
    if has_adapt_import:
        manifest_sections.append(
            (
                "Imported ADAPT state",
                [
                    ("Source JSON", adapt_import.get("input_json_path")),
                    ("Metadata match passed", adapt_import.get("metadata_match_passed")),
                    ("Strict match", adapt_import.get("strict_match")),
                    ("Pool", adapt_import.get("pool_type")),
                    ("Ansatz depth", adapt_import.get("ansatz_depth")),
                    ("ADAPT energy", adapt_import.get("energy")),
                    ("ADAPT |ΔE|", adapt_import.get("abs_delta_e")),
                ],
            )
        )

    summary_sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "Headline results",
            [
                ("Run mode", run_mode),
                ("Subspace fidelity at t=0", float(fid[0]) if fid.size > 0 else None),
                ("VQE energy", payload.get("vqe", {}).get("energy")),
                ("Exact filtered energy", payload["ground_state"].get("exact_energy_filtered")),
                ("|VQE - exact(filtered)|", vqe_abs_delta),
                ("QPE energy estimate", payload.get("qpe", {}).get("energy_estimate")),
            ],
        ),
        (
            "Prepared state and propagation",
            [
                ("Ansatz", ansatz_label),
                ("Propagator", propagator_label),
                ("Initial state source", settings.get("initial_state_source")),
                ("Selected propagated branch", legacy_branch_display),
                ("Filtered sector", payload["ground_state"].get("filtered_sector")),
            ],
        ),
        (
            "Trajectory coverage",
            [
                ("t_final", settings.get("t_final")),
                ("num_times", settings.get("num_times")),
                ("trotter_steps", settings.get("trotter_steps")),
                ("Boundary", settings.get("boundary")),
                ("Ordering", settings.get("ordering")),
            ],
        ),
    ]

    adapt_summary_lines: list[str] | None = None
    if has_adapt_import and bool(settings.get("adapt_summary_in_pdf", True)):
        mismatch_lines = adapt_import.get("metadata_mismatches") or []
        adapt_summary_lines = [
            "ADAPT Initial-State Provenance",
            "",
            f"import_json: {adapt_import.get('input_json_path')}",
            f"import_source: {adapt_import.get('initial_state_source')}",
            f"strict_match: {adapt_import.get('strict_match')}",
            f"metadata_match_passed: {adapt_import.get('metadata_match_passed')}",
            "",
            f"pool_type: {adapt_import.get('pool_type')}",
            f"ansatz_depth: {adapt_import.get('ansatz_depth')}",
            f"num_parameters: {adapt_import.get('num_parameters')}",
            f"operator_count: {adapt_import.get('operator_count')}",
            f"adapt_energy: {adapt_import.get('energy')}",
            f"adapt_abs_delta_e: {adapt_import.get('abs_delta_e')}",
            "",
            "Dynamics used imported ADAPT state as t=0 initial condition.",
        ]
        if isinstance(mismatch_lines, list) and len(mismatch_lines) > 0:
            adapt_summary_lines += ["", "Metadata mismatches:"]
            adapt_summary_lines.extend([f"  - {str(item)}" for item in mismatch_lines])

    with PdfPages(str(pdf_path)) as pdf:
        render_manifest_overview_page(
            pdf,
            title=f"{model_name} report — L={settings.get('L')}",
            experiment_statement=(
                f"Prepared-state quality and exact-vs-propagated dynamics for {model_name} "
                f"using {ansatz_label} state preparation and {propagator_label} evolution."
            ),
            sections=manifest_sections,
            notes=[
                "The parameter manifest is intentionally separate from the executed command; full provenance appears in the appendix.",
            ],
        )
        render_executive_summary_page(
            pdf,
            title="Executive summary",
            experiment_statement=(
                f"Scientist-facing overview of the selected {legacy_branch_display.lower()} versus exact reference."
            ),
            sections=summary_sections,
            notes=[
                "Core result pages focus on prepared-state quality and exact-versus-propagated observables.",
                "3D surfaces, branch-complete diagnostics, imported-state metadata, and the full command are appendix material.",
            ],
        )
        render_section_divider_page(
            pdf,
            title="Core scientific results",
            summary="Prepared-state quality is shown first; supporting diagnostic surfaces are deferred to the appendix.",
            bullets=[
                "Reference / prepared / propagated observables only.",
                "Drive diagnostics stay near the main dynamics pages when drive is enabled.",
            ],
        )

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

        ax00.plot(times, fid_paop, color="#d62728", marker="o", markersize=3, markevery=markevery, label="Fidelity (PAOP trotter)")
        ax00.plot(times, fid_hva, color="#1f77b4", marker="s", markersize=2.5, markevery=markevery, label="Fidelity (HVA trotter)")
        fid_title = payload.get("settings", {}).get(
            "fidelity_definition_short",
            "Subspace Fidelity(t) = <psi_ansatz_trot(t)|P_exact_gs_subspace(t)|psi_ansatz_trot(t)>",
        )
        ax00.set_title(str(fid_title))
        ax00.grid(alpha=0.25)
        ax00.legend(fontsize=7)

        ax01.plot(times, e_exact, label="Exact GS filtered (static)", color="#111111", linewidth=2.0, marker="s", markersize=3, markevery=markevery)
        ax01.plot(times, e_exact_paop, label="Exact PAOP init (static)", color="#2ca02c", linewidth=1.4, marker="D", markersize=3, markevery=markevery)
        ax01.plot(times, e_trot_paop, label="Trotter PAOP init (static)", color="#d62728", linestyle="--", linewidth=1.4, marker="^", markersize=3, markevery=markevery)
        ax01.plot(times, e_exact_hva, label="Exact HVA init (static)", color="#1f77b4", linewidth=1.3, marker="o", markersize=2.5, markevery=markevery)
        ax01.plot(times, e_trot_hva, label="Trotter HVA init (static)", color="#ff7f0e", linestyle="--", linewidth=1.3, marker="<", markersize=2.5, markevery=markevery)

        # --- optional total-energy overlay (when drive active and differs) ---
        if not (np.allclose(e_total_exact, e_exact, atol=1e-14) and
                np.allclose(e_total_trot_paop, e_trot_paop, atol=1e-14) and
                np.allclose(e_total_trot_hva, e_trot_hva, atol=1e-14)):
            ax01.plot(times, e_total_exact, label="Exact GS filtered (total)", color="#17becf",
                      linewidth=1.6, marker="D", markersize=2.5, markevery=markevery, alpha=0.8)
            ax01.plot(times, e_total_exact_paop, label="Exact PAOP init (total)", color="#2ca02c",
                      linewidth=1.3, marker="o", markersize=2.5, markevery=markevery, alpha=0.8)
            ax01.plot(times, e_total_trot_paop, label="Trotter PAOP init (total)", color="#d62728",
                      linestyle=":", linewidth=1.1, marker="x", markersize=2.5, markevery=markevery, alpha=0.7)
            ax01.plot(times, e_total_exact_hva, label="Exact HVA init (total)", color="#1f77b4",
                      linewidth=1.2, marker="v", markersize=2.3, markevery=markevery, alpha=0.75)
            ax01.plot(times, e_total_trot_hva, label="Trotter HVA init (total)", color="#ff7f0e",
                      linestyle="--", linewidth=1.2, marker="<", markersize=2.5, markevery=markevery, alpha=0.8)

        ax01.set_title("Energy")
        ax01.grid(alpha=0.25)
        ax01.legend(fontsize=7)

        ax10.plot(times, nu_exact, label="n_up0 exact GS", color="#17becf", linewidth=1.8, marker="o", markersize=3, markevery=markevery)
        ax10.plot(times, nu_exact_ans, label=f"n_up0 exact selected ({legacy_plot_label})", color="#2ca02c", linewidth=1.2, marker="D", markersize=3, markevery=markevery)
        ax10.plot(times, nu_trot, label="n_up0 trotter", color="#0f7f8b", linestyle="--", linewidth=1.2, marker="s", markersize=3, markevery=markevery)
        ax10.plot(times, nd_exact, label="n_dn0 exact GS", color="#9467bd", linewidth=1.8, marker="^", markersize=3, markevery=markevery)
        ax10.plot(times, nd_exact_ans, label=f"n_dn0 exact selected ({legacy_plot_label})", color="#8c564b", linewidth=1.2, marker="X", markersize=3, markevery=markevery)
        ax10.plot(times, nd_trot, label="n_dn0 trotter", color="#6f4d8f", linestyle="--", linewidth=1.2, marker="v", markersize=3, markevery=markevery)
        ax10.set_title("Site-0 Occupations")
        ax10.set_xlabel("Time")
        ax10.grid(alpha=0.25)
        ax10.legend(fontsize=8)

        ax11.plot(times, d_exact, label="doublon exact GS", color="#8c564b", linewidth=1.8, marker="o", markersize=3, markevery=markevery)
        ax11.plot(times, d_exact_paop, label="doublon exact PAOP", color="#2ca02c", linewidth=1.2, marker="D", markersize=3, markevery=markevery)
        ax11.plot(times, d_trot_paop, label="doublon trotter PAOP", color="#d62728", linestyle="--", linewidth=1.2, marker="s", markersize=3, markevery=markevery)
        ax11.plot(times, d_exact_hva, label="doublon exact HVA", color="#1f77b4", linewidth=1.2, marker="v", markersize=2.5, markevery=markevery)
        ax11.plot(times, d_trot_hva, label="doublon trotter HVA", color="#ff7f0e", linestyle="--", linewidth=1.1, marker="<", markersize=2.5, markevery=markevery)
        ax11.set_title("Total Doublon")
        ax11.set_xlabel("Time")
        ax11.grid(alpha=0.25)
        ax11.legend(fontsize=7)

        fig.suptitle(f"Hardcoded Hubbard Pipeline: L={payload['settings']['L']}", fontsize=14)
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        # Additional focused energy pages (requested): static-only and total-only.
        fig_energy_static, ax_energy_static = plt.subplots(1, 1, figsize=(11.0, 8.5))
        ax_energy_static.plot(times, e_exact, label="Exact GS filtered (static)", color="#111111", linewidth=2.0, marker="s", markersize=3, markevery=markevery)
        ax_energy_static.plot(times, e_exact_paop, label="Exact PAOP init (static)", color="#2ca02c", linewidth=1.4, marker="D", markersize=3, markevery=markevery)
        ax_energy_static.plot(times, e_trot_paop, label="Trotter PAOP init (static)", color="#d62728", linestyle="--", linewidth=1.4, marker="^", markersize=3, markevery=markevery)
        ax_energy_static.plot(times, e_exact_hva, label="Exact HVA init (static)", color="#1f77b4", linewidth=1.3, marker="o", markersize=2.5, markevery=markevery)
        ax_energy_static.plot(times, e_trot_hva, label="Trotter HVA init (static)", color="#ff7f0e", linestyle="--", linewidth=1.2, marker="<", markersize=2.5, markevery=markevery)
        ax_energy_static.set_title("Energy (Static Hamiltonian Only)")
        ax_energy_static.set_xlabel("Time")
        ax_energy_static.set_ylabel("Energy")
        ax_energy_static.grid(alpha=0.25)
        ax_energy_static.legend(fontsize=7)
        fig_energy_static.tight_layout(rect=(0.0, 0.02, 1.0, 0.98))
        pdf.savefig(fig_energy_static)
        plt.close(fig_energy_static)

        fig_energy_total, ax_energy_total = plt.subplots(1, 1, figsize=(11.0, 8.5))
        ax_energy_total.plot(times, e_total_exact, label="Exact GS filtered (total)", color="#17becf", linewidth=1.6, marker="D", markersize=2.5, markevery=markevery, alpha=0.8)
        ax_energy_total.plot(times, e_total_exact_paop, label="Exact PAOP init (total)", color="#2ca02c", linewidth=1.3, marker="o", markersize=2.5, markevery=markevery, alpha=0.8)
        ax_energy_total.plot(times, e_total_trot_paop, label="Trotter PAOP init (total)", color="#d62728", linestyle="--", linewidth=1.2, marker="<", markersize=2.5, markevery=markevery, alpha=0.8)
        ax_energy_total.plot(times, e_total_exact_hva, label="Exact HVA init (total)", color="#1f77b4", linewidth=1.2, marker="v", markersize=2.5, markevery=markevery, alpha=0.75)
        ax_energy_total.plot(times, e_total_trot_hva, label="Trotter HVA init (total)", color="#ff7f0e", linestyle="--", linewidth=1.1, marker="x", markersize=2.5, markevery=markevery, alpha=0.75)
        ax_energy_total.set_title("Energy (Total Hamiltonian H(t) Only)")
        ax_energy_total.set_xlabel("Time")
        ax_energy_total.set_ylabel("Energy")
        ax_energy_total.grid(alpha=0.25)
        ax_energy_total.legend(fontsize=7)
        fig_energy_total.tight_layout(rect=(0.0, 0.02, 1.0, 0.98))
        pdf.savefig(fig_energy_total)
        plt.close(fig_energy_total)

        # ------------------------------------------------------------------
        # Appendix: redundant/supporting material
        # ------------------------------------------------------------------
        render_section_divider_page(
            pdf,
            title="Technical appendix",
            summary="Supporting diagnostics, imported-state provenance, branch-complete views, and reproducibility material.",
            bullets=[
                "3D density surfaces and scalar lane plots.",
                "Absolute-error heatmaps.",
                "Imported ADAPT state metadata when present.",
                "Full executed command for reproducibility.",
            ],
        )
        if adapt_summary_lines is not None:
            render_text_page(pdf, adapt_summary_lines, fontsize=10, line_spacing=0.03)

        dens_all = np.concatenate([n_site_exact.reshape(-1), n_site_exact_ans.reshape(-1), n_site_trot.reshape(-1)])
        dens_zlim = (float(np.min(dens_all)), float(np.max(dens_all)))
        if abs(dens_zlim[1] - dens_zlim[0]) < 1e-12:
            dens_zlim = (dens_zlim[0] - 1e-6, dens_zlim[1] + 1e-6)

        fig3d_n = plt.figure(figsize=(14.0, 8.5))
        axn0 = fig3d_n.add_subplot(1, 3, 1, projection="3d")
        axn1 = fig3d_n.add_subplot(1, 3, 2, projection="3d")
        axn2 = fig3d_n.add_subplot(1, 3, 3, projection="3d")
        _plot_density_surface(axn0, n_site_exact, title="Exact GS filtered: n(site,t)", zlim=dens_zlim, cmap="Blues")
        _plot_density_surface(axn1, n_site_exact_ans, title=f"Exact {legacy_branch_display}: n(site,t)", zlim=dens_zlim, cmap="Greens")
        _plot_density_surface(axn2, n_site_trot, title=f"Trotter {legacy_branch_display}: n(site,t)", zlim=dens_zlim, cmap="Oranges")
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
        _plot_density_surface(axu0, n_up_site_exact, title="Exact GS filtered: n_up(site,t)", zlim=up_zlim, cmap="PuBu")
        _plot_density_surface(axu1, n_up_site_exact_ans, title=f"Exact {legacy_branch_display}: n_up(site,t)", zlim=up_zlim, cmap="YlGn")
        _plot_density_surface(axu2, n_up_site_trot, title=f"Trotter {legacy_branch_display}: n_up(site,t)", zlim=up_zlim, cmap="YlOrBr")
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
        _plot_density_surface(axd0, n_dn_site_exact, title="Exact GS filtered: n_dn(site,t)", zlim=dn_zlim, cmap="PuBu")
        _plot_density_surface(axd1, n_dn_site_exact_ans, title=f"Exact {legacy_branch_display}: n_dn(site,t)", zlim=dn_zlim, cmap="YlGn")
        _plot_density_surface(axd2, n_dn_site_trot, title=f"Trotter {legacy_branch_display}: n_dn(site,t)", zlim=dn_zlim, cmap="YlOrBr")
        fig3d_dn.suptitle(f"L={payload['settings']['L']} 3D Densities (Spin-Down)", fontsize=13)
        fig3d_dn.tight_layout(rect=(0.0, 0.02, 1.0, 0.93))
        pdf.savefig(fig3d_dn)
        plt.close(fig3d_dn)

        fig3d_scalars = plt.figure(figsize=(14.0, 8.5))
        axe0 = fig3d_scalars.add_subplot(2, 2, 1, projection="3d")
        axe1 = fig3d_scalars.add_subplot(2, 2, 2, projection="3d")
        axe2 = fig3d_scalars.add_subplot(2, 2, 3, projection="3d")
        axe3 = fig3d_scalars.add_subplot(2, 2, 4, projection="3d")
        labels_5 = ["Exact GS", "Exact PAOP", "Trotter PAOP", "Exact HVA", "Trotter HVA"]
        colors_5 = ["#111111", "#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"]
        _plot_lane_3d(
            axe0,
            series=[e_total_exact, e_total_exact_paop, e_total_trot_paop, e_total_exact_hva, e_total_trot_hva],
            labels=labels_5,
            colors=colors_5,
            title="3D lanes: total energy",
            zlabel="Energy",
        )
        _plot_lane_3d(
            axe1,
            series=[e_exact, e_exact_paop, e_trot_paop, e_exact_hva, e_trot_hva],
            labels=labels_5,
            colors=colors_5,
            title="3D lanes: static energy",
            zlabel="Energy",
        )
        _plot_lane_3d(
            axe2,
            series=[d_exact, d_exact_paop, d_trot_paop, d_exact_hva, d_trot_hva],
            labels=labels_5,
            colors=colors_5,
            title="3D lanes: doublon",
            zlabel="Doublon",
        )
        _plot_lane_3d(
            axe3,
            series=[stg_exact, stg_exact_paop, stg_trot_paop, stg_exact_hva, stg_trot_hva],
            labels=labels_5,
            colors=colors_5,
            title="3D lanes: staggered order",
            zlabel="Order",
        )
        fig3d_scalars.suptitle(f"L={payload['settings']['L']} 3D Scalar Observables (Five Branches)", fontsize=13)
        fig3d_scalars.tight_layout(rect=(0.0, 0.02, 1.0, 0.94))
        pdf.savefig(fig3d_scalars)
        plt.close(fig3d_scalars)

        fig_err = plt.figure(figsize=(14.0, 8.5))
        axh0 = fig_err.add_subplot(1, 3, 1)
        axh1 = fig_err.add_subplot(1, 3, 2)
        axh2 = fig_err.add_subplot(1, 3, 3)
        heatmaps = [
            (err_n_trot_vs_exact_ans, f"|n_trot({legacy_branch_display}) - n_exact({legacy_branch_display})|"),
            (err_n_exact_ans_vs_exact_gs, f"|n_exact({legacy_branch_display}) - n_exact_gs|"),
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
            f"  - subspace_fidelity_paop_t0: {float(fid_paop[0]) if fid_paop.size > 0 else None}",
            f"  - subspace_fidelity_hva_t0: {float(fid_hva[0]) if fid_hva.size > 0 else None}",
            f"  - energy_t0_exact_gs: {float(e_exact[0]) if e_exact.size > 0 else None}",
            f"  - energy_t0_exact_paop: {float(e_exact_paop[0]) if e_exact_paop.size > 0 else None}",
            f"  - energy_t0_trotter_paop: {float(e_trot_paop[0]) if e_trot_paop.size > 0 else None}",
            f"  - energy_t0_exact_hva: {float(e_exact_hva[0]) if e_exact_hva.size > 0 else None}",
            f"  - energy_t0_trotter_hva: {float(e_trot_hva[0]) if e_trot_hva.size > 0 else None}",
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
        render_text_page(pdf, lines, fontsize=9)
        render_command_page(
            pdf,
            run_command,
            script_name="pipelines/hardcoded/hubbard_pipeline.py",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hardcoded-first Hubbard / Hubbard-Holstein pipeline runner.")
    parser.add_argument("--L", type=int, required=True, help="Number of lattice sites.")
    parser.add_argument("--t", type=float, default=1.0, help="Hopping coefficient.")
    parser.add_argument("--u", type=float, default=4.0, help="Onsite interaction U.")
    parser.add_argument("--dv", type=float, default=0.0, help="Uniform local potential term v (Hv = -v n).")
    parser.add_argument("--boundary", choices=["periodic", "open"], default="periodic")

    # --- problem type: Hubbard vs Hubbard-Holstein ---
    parser.add_argument(
        "--problem",
        type=str,
        choices=["hubbard", "hh"],
        default="hubbard",
        help="Problem type: pure Hubbard (default) or Hubbard-Holstein.",
    )
    parser.add_argument("--omega0", type=float, default=1.0, help="(HH) Phonon frequency omega_0.")
    parser.add_argument("--g-ep", type=float, default=0.5, help="(HH) Electron-phonon coupling g.")
    parser.add_argument("--n-ph-max", type=int, default=1, help="(HH) Max phonon occupancy per site (truncation).")
    parser.add_argument(
        "--boson-encoding",
        type=str,
        default="binary",
        help="(HH) Boson-to-qubit encoding (default: binary).",
    )
    parser.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    parser.add_argument("--t-final", type=float, default=20.0)
    parser.add_argument("--num-times", type=int, default=201)
    parser.add_argument("--suzuki-order", type=int, default=2)
    parser.add_argument("--trotter-steps", type=int, default=64)
    parser.add_argument(
        "--propagator",
        choices=["suzuki2", "piecewise_exact", "cfqm4", "cfqm6"],
        default="cfqm4",
        help=(
            "Trajectory propagator: cfqm4 (default), "
            "piecewise_exact, or CFQM variants cfqm4/cfqm6."
        ),
    )
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
        help=(
            "Time sampling rule per Trotter slice (used by suzuki2/piecewise_exact; "
            "CFQM ignores this and uses fixed scheme nodes c_j)."
        ),
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
            "exact eigendecomposition). "
            "Reference-only control: does not apply to cfqm4/cfqm6 macro-step counts."
        ),
    )
    parser.add_argument(
        "--cfqm-stage-exp",
        choices=["expm_multiply_sparse", "dense_expm", "pauli_suzuki2"],
        default="expm_multiply_sparse",
        help="CFQM stage exponential backend (used only when --propagator cfqm4/cfqm6).",
    )
    parser.add_argument(
        "--cfqm-coeff-drop-abs-tol",
        type=float,
        default=0.0,
        help="Drop |coeff|<tol after CFQM stage accumulation (used only for cfqm4/cfqm6).",
    )
    parser.add_argument(
        "--cfqm-normalize",
        action="store_true",
        help="Renormalize state after each CFQM macro-step (default off).",
    )
    parser.add_argument(
        "--vqe-ansatz",
        type=str,
        default="uccsd",
        choices=["uccsd", "hva", "hh_hva", "hh_hva_tw", "hh_hva_ptw"],
        help=(
            "Hardcoded VQE ansatz family: uccsd, hva (Hubbard-only), "
            "hh_hva (Hubbard-Holstein layerwise — shared θ per group), "
            "hh_hva_tw (Hubbard-Holstein termwise — one θ per Pauli term, "
            "does not preserve sector), "
            "hh_hva_ptw (Hubbard-Holstein physical-termwise — one θ per physical term, "
            "sector-preserving)."
        ),
    )
    parser.add_argument("--vqe-reps", type=int, default=2, help="Number of ansatz repetitions (layer depth).")
    parser.add_argument("--vqe-restarts", type=int, default=1)
    parser.add_argument("--vqe-seed", type=int, default=7)
    parser.add_argument("--vqe-maxiter", type=int, default=120)
    parser.add_argument(
        "--vqe-method",
        type=str,
        default="SPSA",
        choices=["SLSQP", "COBYLA", "L-BFGS-B", "Powell", "Nelder-Mead", "SPSA"],
        help="SciPy optimizer (or SPSA) used by hardcoded VQE.",
    )
    parser.add_argument("--vqe-spsa-a", type=float, default=0.2)
    parser.add_argument("--vqe-spsa-c", type=float, default=0.1)
    parser.add_argument("--vqe-spsa-alpha", type=float, default=0.602)
    parser.add_argument("--vqe-spsa-gamma", type=float, default=0.101)
    parser.add_argument("--vqe-spsa-A", type=float, default=10.0)
    parser.add_argument("--vqe-spsa-avg-last", type=int, default=0)
    parser.add_argument("--vqe-spsa-eval-repeats", type=int, default=1)
    parser.add_argument(
        "--vqe-spsa-eval-agg",
        choices=["mean", "median"],
        default="mean",
    )
    parser.add_argument(
        "--vqe-energy-backend",
        type=str,
        default="one_apply_compiled",
        choices=["legacy", "one_apply_compiled"],
        help=(
            "Hardcoded VQE energy objective backend: "
            "'legacy' (termwise expectation sum) or "
            "'one_apply_compiled' (compiled one-apply energy)."
        ),
    )
    parser.add_argument(
        "--vqe-progress-every-s",
        type=float,
        default=60.0,
        help="Emit hardcoded VQE heartbeat AI_LOG events every N seconds.",
    )

    parser.add_argument("--qpe-eval-qubits", type=int, default=6)
    parser.add_argument("--qpe-shots", type=int, default=1024)
    parser.add_argument("--qpe-seed", type=int, default=11)
    parser.add_argument("--skip-qpe", action="store_true", help="Skip QPE execution and mark qpe payload as skipped.")

    parser.add_argument("--initial-state-source", choices=["exact", "vqe", "hf", "adapt_json"], default="vqe")
    parser.add_argument(
        "--adapt-input-json",
        type=Path,
        default=None,
        help="Path to ADAPT pipeline JSON used when --initial-state-source adapt_json.",
    )
    parser.set_defaults(adapt_strict_match=True)
    parser.add_argument(
        "--adapt-strict-match",
        dest="adapt_strict_match",
        action="store_true",
        help="Require ADAPT JSON physics settings to match this run (default: enabled).",
    )
    parser.add_argument(
        "--no-adapt-strict-match",
        dest="adapt_strict_match",
        action="store_false",
        help="Allow ADAPT JSON import with physics-setting mismatches (logged in payload/PDF).",
    )
    parser.set_defaults(adapt_summary_in_pdf=True)
    parser.add_argument(
        "--adapt-summary-in-pdf",
        dest="adapt_summary_in_pdf",
        action="store_true",
        help="Include ADAPT provenance page in comprehensive PDF (default: enabled).",
    )
    parser.add_argument(
        "--no-adapt-summary-in-pdf",
        dest="adapt_summary_in_pdf",
        action="store_false",
        help="Skip ADAPT provenance page even when using --initial-state-source adapt_json.",
    )
    parser.add_argument(
        "--adapt-pool",
        choices=[
            "uccsd",
            "cse",
            "full_hamiltonian",
            "hva",
            "full_meta",
            "uccsd_paop_lf_full",
            "paop",
            "paop_min",
            "paop_std",
            "paop_full",
            "paop_lf",
            "paop_lf_std",
            "paop_lf2_std",
            "paop_lf_full",
        ],
        default=None,
        help=(
            "PAOP/ADAPT branch pool. If omitted, runtime resolves by mode/problem "
            "(hh phase1_v1/phase2_v1/phase3_v1 core defaults to paop_lf_std; hubbard legacy defaults to uccsd)."
        ),
    )
    parser.add_argument(
        "--adapt-continuation-mode",
        choices=["legacy", "phase1_v1", "phase2_v1", "phase3_v1"],
        default="legacy",
        help="Continuation mode for internal ADAPT branch (default: legacy). phase1_v1 is staged HH continuation; phase2_v1 adds shortlist/full scoring; phase3_v1 adds generator/motif/symmetry/rescue metadata.",
    )
    parser.add_argument(
        "--adapt-ref-source",
        choices=["hf", "vqe"],
        default="hf",
        help="Reference source for internal ADAPT branch optimization (HF default, or warm-start from hardcoded VQE state).",
    )
    parser.add_argument("--adapt-max-depth", type=int, default=30)
    parser.add_argument("--adapt-eps-grad", type=float, default=1e-5)
    parser.add_argument(
        "--adapt-eps-energy",
        type=float,
        default=1e-8,
        help=(
            "Internal ADAPT energy convergence threshold. Acts as a terminating guard for Hubbard and HH legacy "
            "runs; in HH phase1_v1/phase2_v1/phase3_v1 it is telemetry-only."
        ),
    )
    parser.add_argument("--adapt-maxiter", type=int, default=800)
    parser.add_argument("--adapt-seed", type=int, default=7)
    parser.set_defaults(adapt_allow_repeats=True)
    parser.add_argument("--adapt-allow-repeats", dest="adapt_allow_repeats", action="store_true")
    parser.add_argument("--adapt-no-repeats", dest="adapt_allow_repeats", action="store_false")
    parser.set_defaults(adapt_finite_angle_fallback=True)
    parser.add_argument("--adapt-finite-angle-fallback", dest="adapt_finite_angle_fallback", action="store_true")
    parser.add_argument("--adapt-no-finite-angle-fallback", dest="adapt_finite_angle_fallback", action="store_false")
    parser.add_argument("--adapt-finite-angle", type=float, default=0.1)
    parser.add_argument("--adapt-finite-angle-min-improvement", type=float, default=1e-12)
    parser.add_argument("--adapt-disable-hh-seed", action="store_true")
    parser.add_argument(
        "--adapt-reopt-policy",
        choices=["append_only", "full", "windowed"],
        default="append_only",
        help=(
            "Per-depth ADAPT re-optimization policy. "
            "'append_only' (default): newest parameter only. "
            "'full': all parameters. "
            "'windowed': sliding window + optional top-k carry."
        ),
    )
    parser.add_argument(
        "--adapt-window-size", type=int, default=3,
        help="Window size for 'windowed' reopt policy.",
    )
    parser.add_argument(
        "--adapt-window-topk", type=int, default=0,
        help="Number of older high-magnitude parameters in windowed active set.",
    )
    parser.add_argument(
        "--adapt-full-refit-every", type=int, default=0,
        help="Periodic full-prefix cadence for 'windowed' (0=disabled).",
    )
    parser.add_argument(
        "--adapt-final-full-refit",
        choices=["true", "false"],
        default="true",
        help="Final full-prefix refit after ADAPT loop for 'windowed'.",
    )
    parser.add_argument("--phase1-lambda-F", type=float, default=1.0)
    parser.add_argument("--phase1-lambda-compile", type=float, default=0.05)
    parser.add_argument("--phase1-lambda-measure", type=float, default=0.02)
    parser.add_argument("--phase1-lambda-leak", type=float, default=0.0)
    parser.add_argument("--phase1-score-z-alpha", type=float, default=0.0)
    parser.add_argument("--phase1-probe-max-positions", type=int, default=6)
    parser.add_argument("--phase1-plateau-patience", type=int, default=2)
    parser.add_argument("--phase1-trough-margin-ratio", type=float, default=1.0)
    parser.set_defaults(phase1_prune_enabled=True)
    parser.add_argument("--phase1-prune-enabled", dest="phase1_prune_enabled", action="store_true")
    parser.add_argument("--phase1-no-prune", dest="phase1_prune_enabled", action="store_false")
    parser.add_argument("--phase1-prune-fraction", type=float, default=0.25)
    parser.add_argument("--phase1-prune-max-candidates", type=int, default=6)
    parser.add_argument("--phase1-prune-max-regression", type=float, default=1e-8)
    parser.add_argument("--phase3-motif-source-json", type=Path, default=None)
    parser.add_argument(
        "--phase3-symmetry-mitigation-mode",
        choices=["off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"],
        default="off",
    )
    parser.set_defaults(phase3_enable_rescue=False)
    parser.add_argument("--phase3-enable-rescue", dest="phase3_enable_rescue", action="store_true")
    parser.add_argument("--phase3-no-rescue", dest="phase3_enable_rescue", action="store_false")
    parser.add_argument(
        "--phase3-lifetime-cost-mode",
        choices=["off", "phase3_v1"],
        default="phase3_v1",
    )
    parser.add_argument(
        "--phase3-runtime-split-mode",
        choices=["off", "shortlist_pauli_children_v1"],
        default="off",
        help="Opt-in shortlist-only macro splitting via serialized Pauli child atoms with symmetry-safe child-set admission.",
    )
    parser.add_argument("--paop-r", type=int, default=1)
    parser.add_argument("--paop-split-paulis", action="store_true")
    parser.add_argument("--paop-prune-eps", type=float, default=0.0)
    parser.add_argument(
        "--paop-normalization",
        choices=["none", "fro", "maxcoeff"],
        default="none",
    )

    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument("--skip-pdf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ai_log("hardcoded_main_start", settings=vars(args))
    run_command = current_command_string()
    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    json_dir = artifacts_dir / "json"
    pdf_dir = artifacts_dir / "pdf"
    json_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    prob = "hh" if str(args.problem).strip().lower() == "hh" else "hubbard"
    output_json = args.output_json or (json_dir / f"hc_{prob}_L{args.L}.json")
    output_pdf = args.output_pdf or (pdf_dir / f"hc_{prob}_L{args.L}.pdf")

    is_hh = str(args.problem).strip().lower() == "hh"

    # --- build Hamiltonian (branched on problem type) ---
    if is_hh:
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=int(args.L),
            J=float(args.t),
            U=float(args.u),
            omega0=float(args.omega0),
            g=float(args.g_ep),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            repr_mode="JW",
            indexing=str(args.ordering),
            pbc=(str(args.boundary) == "periodic"),
        )
        _qpb = int(boson_qubits_per_site(int(args.n_ph_max), str(args.boson_encoding)))
        nq_total = 2 * int(args.L) + int(args.L) * _qpb
    else:
        h_poly = build_hubbard_hamiltonian(
            dims=int(args.L),
            t=float(args.t),
            U=float(args.u),
            v=float(args.dv),
            repr_mode="JW",
            indexing=str(args.ordering),
            pbc=(str(args.boundary) == "periodic"),
        )
        nq_total = 2 * int(args.L)

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
            nq_total=int(nq_total),
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
        if is_hh:
            gs_energy_exact_filtered, fidelity_subspace_basis_v0 = _ground_manifold_basis_sector_filtered_hh(
                hmat=hmat,
                num_sites=int(args.L),
                num_particles=_vqe_num_particles,
                ordering=str(args.ordering),
                nq_total=int(nq_total),
                energy_tol=_fidelity_subspace_tol,
            )
        else:
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
            energy_backend=str(args.vqe_energy_backend),
            vqe_progress_every_s=float(args.vqe_progress_every_s),
            spsa_a=float(args.vqe_spsa_a),
            spsa_c=float(args.vqe_spsa_c),
            spsa_alpha=float(args.vqe_spsa_alpha),
            spsa_gamma=float(args.vqe_spsa_gamma),
            spsa_A=float(args.vqe_spsa_A),
            spsa_avg_last=int(args.vqe_spsa_avg_last),
            spsa_eval_repeats=int(args.vqe_spsa_eval_repeats),
            spsa_eval_agg=str(args.vqe_spsa_eval_agg),
            ansatz_name=str(args.vqe_ansatz),
            problem=str(args.problem),
            omega0=float(args.omega0),
            g_ep=float(args.g_ep),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
        )
    except Exception as exc:
        _ai_log("hardcoded_vqe_failed", L=int(args.L), error=str(exc))
        vqe_payload = {
            "success": False,
            "method": "hardcoded_layerwise_statevector",
            "ansatz": str(args.vqe_ansatz),
            "optimizer_method": str(args.vqe_method),
            "energy_backend": str(args.vqe_energy_backend),
            "parameterization": "layerwise",
            "exact_filtered_energy": None,
            "energy": None,
            "error": str(exc),
        }
        if str(args.vqe_method).strip().lower() == "spsa":
            vqe_payload["spsa"] = {
                "a": float(args.vqe_spsa_a),
                "c": float(args.vqe_spsa_c),
                "alpha": float(args.vqe_spsa_alpha),
                "gamma": float(args.vqe_spsa_gamma),
                "A": float(args.vqe_spsa_A),
                "avg_last": int(args.vqe_spsa_avg_last),
                "eval_repeats": int(args.vqe_spsa_eval_repeats),
                "eval_agg": str(args.vqe_spsa_eval_agg),
            }
        psi_vqe = psi_exact_ground

    num_particles = _half_filled_particles(int(args.L))
    if is_hh:
        psi_hf = _normalize_state(
            np.asarray(
                hubbard_holstein_reference_state(
                    dims=int(args.L),
                    num_particles=num_particles,
                    n_ph_max=int(args.n_ph_max),
                    boson_encoding=str(args.boson_encoding),
                    indexing=str(args.ordering),
                ),
                dtype=complex,
            ).reshape(-1)
        )
    else:
        psi_hf = _normalize_state(
            np.asarray(
                hartree_fock_statevector(int(args.L), num_particles, indexing=str(args.ordering)),
                dtype=complex,
            ).reshape(-1)
        )

    adapt_import_payload: dict[str, Any] | None = None
    adapt_internal_payload: dict[str, Any] | None = None

    if args.initial_state_source == "adapt_json":
        if args.adapt_input_json is None:
            raise ValueError("--adapt-input-json is required when --initial-state-source adapt_json.")
        psi_adapt_import, adapt_meta = _load_adapt_initial_state(Path(args.adapt_input_json), int(nq_total))
        adapt_settings = adapt_meta.get("settings", {})
        mismatches = _validate_adapt_metadata(
            adapt_settings=adapt_settings if isinstance(adapt_settings, dict) else {},
            args=args,
            is_hh=bool(is_hh),
        )
        if len(mismatches) > 0 and bool(args.adapt_strict_match):
            mismatch_text = "; ".join(mismatches)
            raise ValueError(
                "ADAPT JSON metadata mismatch under strict mode. "
                f"Use --no-adapt-strict-match to override. Details: {mismatch_text}"
            )
        if len(mismatches) > 0:
            _ai_log(
                "hardcoded_adapt_import_mismatch",
                strict=False,
                mismatch_count=int(len(mismatches)),
                mismatches=mismatches,
            )
        psi_paop = psi_adapt_import
        adapt_vqe_meta = adapt_meta.get("adapt_vqe", {}) if isinstance(adapt_meta.get("adapt_vqe"), dict) else {}
        ops = adapt_vqe_meta.get("operators")
        op_count = len(ops) if isinstance(ops, list) else None
        adapt_import_payload = {
            "input_json_path": str(Path(args.adapt_input_json)),
            "strict_match": bool(args.adapt_strict_match),
            "metadata_match_passed": bool(len(mismatches) == 0),
            "metadata_mismatches": mismatches,
            "initial_state_source": adapt_meta.get("initial_state_source"),
            "pool_type": adapt_vqe_meta.get("pool_type"),
            "ansatz_depth": adapt_vqe_meta.get("ansatz_depth"),
            "num_parameters": adapt_vqe_meta.get("num_parameters"),
            "operator_count": op_count,
            "energy": adapt_vqe_meta.get("energy"),
            "abs_delta_e": adapt_vqe_meta.get("abs_delta_e"),
            "source": "adapt_json",
        }
    else:
        adapt_ref_source_key = str(args.adapt_ref_source).strip().lower()
        psi_ref_override_for_adapt: np.ndarray | None = None
        if adapt_ref_source_key == "vqe":
            if not bool(vqe_payload.get("success", False)):
                raise RuntimeError(
                    "Requested --adapt-ref-source vqe but hardcoded VQE failed; internal ADAPT reference state is unavailable."
                )
            psi_ref_override_for_adapt = np.asarray(psi_vqe, dtype=complex).reshape(-1)
        try:
            adapt_internal_payload_raw, psi_paop = _run_internal_adapt_paop(
                h_poly=h_poly,
                num_sites=int(args.L),
                ordering=str(args.ordering),
                problem=str(args.problem),
                t=float(args.t),
                u=float(args.u),
                dv=float(args.dv),
                boundary=str(args.boundary),
                omega0=float(args.omega0),
                g_ep=float(args.g_ep),
                n_ph_max=int(args.n_ph_max),
                boson_encoding=str(args.boson_encoding),
                adapt_pool=(str(args.adapt_pool) if args.adapt_pool is not None else None),
                adapt_max_depth=int(args.adapt_max_depth),
                adapt_eps_grad=float(args.adapt_eps_grad),
                adapt_eps_energy=float(args.adapt_eps_energy),
                adapt_maxiter=int(args.adapt_maxiter),
                adapt_seed=int(args.adapt_seed),
                adapt_allow_repeats=bool(args.adapt_allow_repeats),
                adapt_finite_angle_fallback=bool(args.adapt_finite_angle_fallback),
                adapt_finite_angle=float(args.adapt_finite_angle),
                adapt_finite_angle_min_improvement=float(args.adapt_finite_angle_min_improvement),
                paop_r=int(args.paop_r),
                paop_split_paulis=bool(args.paop_split_paulis),
                paop_prune_eps=float(args.paop_prune_eps),
                paop_normalization=str(args.paop_normalization),
                adapt_disable_hh_seed=bool(args.adapt_disable_hh_seed),
                psi_ref_override=psi_ref_override_for_adapt,
                adapt_reopt_policy=str(args.adapt_reopt_policy),
                adapt_window_size=int(args.adapt_window_size),
                adapt_window_topk=int(args.adapt_window_topk),
                adapt_full_refit_every=int(args.adapt_full_refit_every),
                adapt_final_full_refit=bool(str(args.adapt_final_full_refit).strip().lower() == "true"),
                adapt_continuation_mode=str(args.adapt_continuation_mode),
                phase1_lambda_F=float(args.phase1_lambda_F),
                phase1_lambda_compile=float(args.phase1_lambda_compile),
                phase1_lambda_measure=float(args.phase1_lambda_measure),
                phase1_lambda_leak=float(args.phase1_lambda_leak),
                phase1_score_z_alpha=float(args.phase1_score_z_alpha),
                phase1_probe_max_positions=int(args.phase1_probe_max_positions),
                phase1_plateau_patience=int(args.phase1_plateau_patience),
                phase1_trough_margin_ratio=float(args.phase1_trough_margin_ratio),
                phase1_prune_enabled=bool(args.phase1_prune_enabled),
                phase1_prune_fraction=float(args.phase1_prune_fraction),
                phase1_prune_max_candidates=int(args.phase1_prune_max_candidates),
                phase1_prune_max_regression=float(args.phase1_prune_max_regression),
                phase3_motif_source_json=(Path(args.phase3_motif_source_json) if args.phase3_motif_source_json is not None else None),
                phase3_symmetry_mitigation_mode=str(args.phase3_symmetry_mitigation_mode),
                phase3_enable_rescue=bool(args.phase3_enable_rescue),
                phase3_lifetime_cost_mode=str(args.phase3_lifetime_cost_mode),
                phase3_runtime_split_mode=str(args.phase3_runtime_split_mode),
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to build required internal PAOP branch via ADAPT: {exc}") from exc
        adapt_internal_payload = {
            "source": "internal_adapt",
            "pool_type": adapt_internal_payload_raw.get("pool_type"),
            "ansatz_depth": adapt_internal_payload_raw.get("ansatz_depth"),
            "num_parameters": adapt_internal_payload_raw.get("num_parameters"),
            "energy": adapt_internal_payload_raw.get("energy"),
            "abs_delta_e": adapt_internal_payload_raw.get("abs_delta_e"),
            "success": bool(adapt_internal_payload_raw.get("success", False)),
            "stop_reason": adapt_internal_payload_raw.get("stop_reason"),
            "eps_energy_termination_enabled": adapt_internal_payload_raw.get("eps_energy_termination_enabled"),
            "eps_grad_termination_enabled": adapt_internal_payload_raw.get("eps_grad_termination_enabled"),
            "adapt_drop_policy_enabled": adapt_internal_payload_raw.get("adapt_drop_policy_enabled"),
            "adapt_drop_floor_resolved": adapt_internal_payload_raw.get("adapt_drop_floor_resolved"),
            "adapt_drop_patience_resolved": adapt_internal_payload_raw.get("adapt_drop_patience_resolved"),
            "adapt_drop_min_depth_resolved": adapt_internal_payload_raw.get("adapt_drop_min_depth_resolved"),
            "adapt_grad_floor_resolved": adapt_internal_payload_raw.get("adapt_grad_floor_resolved"),
            "adapt_drop_floor_source": adapt_internal_payload_raw.get("adapt_drop_floor_source"),
            "adapt_drop_patience_source": adapt_internal_payload_raw.get("adapt_drop_patience_source"),
            "adapt_drop_min_depth_source": adapt_internal_payload_raw.get("adapt_drop_min_depth_source"),
            "adapt_grad_floor_source": adapt_internal_payload_raw.get("adapt_grad_floor_source"),
            "adapt_drop_policy_source": adapt_internal_payload_raw.get("adapt_drop_policy_source"),
            "elapsed_s": adapt_internal_payload_raw.get("elapsed_s"),
            "allow_repeats": adapt_internal_payload_raw.get("allow_repeats"),
        }
        if adapt_ref_source_key != "hf" or str(args.adapt_pool).strip().lower() in {"uccsd_paop_lf_full", "full_meta"}:
            adapt_internal_payload["adapt_ref_source"] = str(adapt_ref_source_key)
            adapt_internal_payload["nfev_total"] = adapt_internal_payload_raw.get("nfev_total")
        _ai_log(
            "hardcoded_paop_branch_built",
            source="internal_adapt",
            pool=str(adapt_internal_payload.get("pool_type")),
            depth=adapt_internal_payload.get("ansatz_depth"),
            energy=adapt_internal_payload.get("energy"),
        )

    if bool(vqe_payload.get("success", False)):
        psi_hva = psi_vqe
        hva_vqe_success = True
    else:
        psi_hva = psi_exact_ground
        hva_vqe_success = False
        _ai_log("hardcoded_hva_branch_fallback_to_exact", reason="vqe_failed")

    if args.initial_state_source == "adapt_json":
        psi0 = psi_paop
        selected_initial_source = "adapt_json"
        _ai_log("hardcoded_initial_state_selected", source=selected_initial_source, adapt_json=str(args.adapt_input_json))
    elif args.initial_state_source == "vqe" and bool(vqe_payload.get("success", False)):
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

    legacy_branch_label = (
        "paop" if selected_initial_source == "adapt_json"
        else ("hva" if selected_initial_source == "vqe" else str(selected_initial_source))
    )

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
        psi0_legacy_trot=psi0,
        psi0_paop_trot=psi_paop,
        psi0_hva_trot=psi_hva,
        legacy_branch_label=str(legacy_branch_label),
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
        propagator=str(args.propagator),
        cfqm_stage_exp=str(args.cfqm_stage_exp),
        cfqm_coeff_drop_abs_tol=float(args.cfqm_coeff_drop_abs_tol),
        cfqm_normalize=bool(args.cfqm_normalize),
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
        "problem": str(args.problem),
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
        "vqe_method": str(args.vqe_method),
        "vqe_energy_backend": str(args.vqe_energy_backend),
        "vqe_progress_every_s": float(args.vqe_progress_every_s),
        "initial_state_source": str(args.initial_state_source),
        "adapt_input_json": (str(args.adapt_input_json) if args.adapt_input_json is not None else None),
        "adapt_strict_match": bool(args.adapt_strict_match),
        "adapt_summary_in_pdf": bool(args.adapt_summary_in_pdf),
        "adapt_pool": (str(args.adapt_pool) if args.adapt_pool is not None else None),
        "adapt_continuation_mode": str(args.adapt_continuation_mode),
        "adapt_max_depth": int(args.adapt_max_depth),
        "adapt_eps_grad": float(args.adapt_eps_grad),
        "adapt_eps_energy": float(args.adapt_eps_energy),
        "adapt_maxiter": int(args.adapt_maxiter),
        "adapt_seed": int(args.adapt_seed),
        "adapt_allow_repeats": bool(args.adapt_allow_repeats),
        "adapt_finite_angle_fallback": bool(args.adapt_finite_angle_fallback),
        "adapt_finite_angle": float(args.adapt_finite_angle),
        "adapt_finite_angle_min_improvement": float(args.adapt_finite_angle_min_improvement),
        "adapt_disable_hh_seed": bool(args.adapt_disable_hh_seed),
        "paop_r": int(args.paop_r),
        "paop_split_paulis": bool(args.paop_split_paulis),
        "paop_prune_eps": float(args.paop_prune_eps),
        "paop_normalization": str(args.paop_normalization),
        "phase1_lambda_F": float(args.phase1_lambda_F),
        "phase1_lambda_compile": float(args.phase1_lambda_compile),
        "phase1_lambda_measure": float(args.phase1_lambda_measure),
        "phase1_lambda_leak": float(args.phase1_lambda_leak),
        "phase1_score_z_alpha": float(args.phase1_score_z_alpha),
        "phase1_probe_max_positions": int(args.phase1_probe_max_positions),
        "phase1_plateau_patience": int(args.phase1_plateau_patience),
        "phase1_trough_margin_ratio": float(args.phase1_trough_margin_ratio),
        "phase1_prune_enabled": bool(args.phase1_prune_enabled),
        "phase1_prune_fraction": float(args.phase1_prune_fraction),
        "phase1_prune_max_candidates": int(args.phase1_prune_max_candidates),
        "phase1_prune_max_regression": float(args.phase1_prune_max_regression),
        "phase3_motif_source_json": (str(args.phase3_motif_source_json) if args.phase3_motif_source_json is not None else None),
        "phase3_symmetry_mitigation_mode": str(args.phase3_symmetry_mitigation_mode),
        "phase3_enable_rescue": bool(args.phase3_enable_rescue),
        "phase3_lifetime_cost_mode": str(args.phase3_lifetime_cost_mode),
        "phase3_runtime_split_mode": str(args.phase3_runtime_split_mode),
        "skip_qpe": bool(args.skip_qpe),
        "fidelity_definition_short": (
            "Subspace Fidelity(t): projected fidelity of trotterized PAOP/HVA branches vs filtered exact GS manifold."
        ),
        "fidelity_definition": (
            "fidelity_paop_trotter(t) = <psi_paop_trot(t)|P_exact_gs_subspace(t)|psi_paop_trot(t)> and "
            "fidelity_hva_trotter(t) = <psi_hva_trot(t)|P_exact_gs_subspace(t)|psi_hva_trot(t)>, "
            "where P_exact_gs_subspace(t) projects onto the time-evolved filtered-sector ground manifold "
            "selected by E <= E0 + tol. Legacy key fidelity(t) follows the selected initial-state branch."
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
            "exact_gs_filtered: exact/reference propagation from filtered-sector GS init; "
            "exact_paop/trotter_paop: exact/Trotter propagation from PAOP-ADAPT initial state; "
            "exact_hva/trotter_hva: exact/Trotter propagation from regular hardcoded VQE (HVA-family) initial state; "
            "legacy exact_ansatz/trotter/fidelity map to the selected initial-state branch for compatibility."
        ),
        "energy_observable_definition": (
            "energy_static_exact is <psi_exact_gs_ref(t)|H_static|psi_exact_gs_ref(t)>. "
            "energy_static_exact_paop and energy_static_trotter_paop are from PAOP-ADAPT initial state; "
            "energy_static_exact_hva and energy_static_trotter_hva are from hardcoded VQE initial state. "
            "energy_total_exact is <psi_exact_gs_ref(t)|H_static + H_drive(drive_t0 + t)|psi_exact_gs_ref(t)>. "
            "energy_total_exact_paop/energy_total_trotter_paop and "
            "energy_total_exact_hva/energy_total_trotter_hva are defined analogously. "
            "Legacy energy_*_exact_ansatz and energy_*_trotter keys map to the selected initial-state branch. "
            "When drive is disabled, energy_total_* == energy_static_*. "
            "Drive sampling uses the same drive_t0 convention as propagation."
        ),
        "plot_branch_order": [
            "exact_gs_filtered",
            "exact_paop",
            "trotter_paop",
            "exact_hva",
            "trotter_hva",
        ],
    }
    if str(args.vqe_method).strip().lower() == "spsa":
        settings["vqe_spsa"] = {
            "a": float(args.vqe_spsa_a),
            "c": float(args.vqe_spsa_c),
            "alpha": float(args.vqe_spsa_alpha),
            "gamma": float(args.vqe_spsa_gamma),
            "A": float(args.vqe_spsa_A),
            "avg_last": int(args.vqe_spsa_avg_last),
            "eval_repeats": int(args.vqe_spsa_eval_repeats),
            "eval_agg": str(args.vqe_spsa_eval_agg),
        }
    _propagator_key = str(args.propagator).strip().lower()
    if _propagator_key != "suzuki2":
        settings["propagator"] = str(_propagator_key)
    if _propagator_key in {"cfqm4", "cfqm6"}:
        settings["cfqm"] = {
            "stage_exp_backend": str(args.cfqm_stage_exp),
            "coeff_drop_abs_tol": float(args.cfqm_coeff_drop_abs_tol),
            "normalize": bool(args.cfqm_normalize),
        }
    _use_internal_adapt = str(args.initial_state_source) != "adapt_json"
    _adapt_ref_source_key = str(args.adapt_ref_source).strip().lower()
    if _use_internal_adapt and (
        _adapt_ref_source_key != "hf" or str(args.adapt_pool).strip().lower() in {"uccsd_paop_lf_full", "full_meta"}
    ):
        settings["adapt_ref_source"] = str(_adapt_ref_source_key)
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

    if is_hh:
        settings["holstein"] = {
            "omega0": float(args.omega0),
            "g_ep": float(args.g_ep),
            "n_ph_max": int(args.n_ph_max),
            "boson_encoding": str(args.boson_encoding),
            "nq_fermion": 2 * int(args.L),
            "nq_phonon": int(args.L) * _qpb,
            "nq_total": int(nq_total),
        }

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "hardcoded",
        "settings": settings,
        "hamiltonian": {
            "num_qubits": int(nq_total),
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
        "ansatz_branches": {
            "branch_order": [
                "exact_gs_filtered",
                "exact_paop",
                "trotter_paop",
                "exact_hva",
                "trotter_hva",
            ],
            "paop": {
                "source": (
                    "adapt_json"
                    if adapt_import_payload is not None
                    else "internal_adapt"
                ),
                "pool_type": (
                    adapt_import_payload.get("pool_type")
                    if adapt_import_payload is not None
                    else (adapt_internal_payload.get("pool_type") if adapt_internal_payload is not None else str(args.adapt_pool))
                ),
                "ansatz_depth": (
                    adapt_import_payload.get("ansatz_depth")
                    if adapt_import_payload is not None
                    else (adapt_internal_payload.get("ansatz_depth") if adapt_internal_payload is not None else None)
                ),
                "num_parameters": (
                    adapt_import_payload.get("num_parameters")
                    if adapt_import_payload is not None
                    else (adapt_internal_payload.get("num_parameters") if adapt_internal_payload is not None else None)
                ),
                "energy": (
                    adapt_import_payload.get("energy")
                    if adapt_import_payload is not None
                    else (adapt_internal_payload.get("energy") if adapt_internal_payload is not None else None)
                ),
                "abs_delta_e": (
                    adapt_import_payload.get("abs_delta_e")
                    if adapt_import_payload is not None
                    else (adapt_internal_payload.get("abs_delta_e") if adapt_internal_payload is not None else None)
                ),
            },
            "hva": {
                "source": "regular_vqe",
                "ansatz": str(args.vqe_ansatz),
                "vqe_success": bool(vqe_payload.get("success", False)),
                "energy": vqe_payload.get("energy"),
                "exact_filtered_energy": vqe_payload.get("exact_filtered_energy"),
            },
            "legacy_selected_branch": str(legacy_branch_label),
            "legacy_selected_source": str(selected_initial_source),
        },
        "trajectory": trajectory,
        "sanity": sanity,
    }
    if adapt_import_payload is not None:
        payload["adapt_import"] = adapt_import_payload
    if adapt_internal_payload is not None:
        payload["adapt_internal"] = adapt_internal_payload

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
