"""Problem setup helpers for the static ADAPT pipeline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from src.quantum.hartree_fock_reference_state import hartree_fock_statevector
from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
)
from src.quantum.vqe_latex_python_pairs import (
    exact_ground_energy_sector,
    exact_ground_energy_sector_hh,
    half_filled_num_particles,
    hubbard_holstein_reference_state,
)

_HH_STAGED_CONTINUATION_MODES = frozenset({"phase1_v1", "phase2_v1", "phase3_v1"})

PAULI_MATS = {
    "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(psi))
    if nrm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    return psi / nrm


def _collect_hardcoded_terms_exyz(poly: Any, tol: float = 1e-12) -> tuple[list[str], dict[str, complex]]:
    coeff_map: dict[str, complex] = {}
    order: list[str] = []
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= tol:
            continue
        if label not in coeff_map:
            coeff_map[label] = 0.0 + 0.0j
            order.append(label)
        coeff_map[label] += coeff
    cleaned_order = [lbl for lbl in order if abs(coeff_map[lbl]) > tol]
    cleaned_map = {lbl: coeff_map[lbl] for lbl in cleaned_order}
    return cleaned_order, cleaned_map


def _pauli_matrix_exyz(label: str) -> np.ndarray:
    mats = [PAULI_MATS[ch] for ch in label]
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out


def _build_hamiltonian_matrix(coeff_map_exyz: dict[str, complex]) -> np.ndarray:
    if not coeff_map_exyz:
        return np.zeros((1, 1), dtype=complex)
    nq = len(next(iter(coeff_map_exyz)))
    dim = 1 << nq
    hmat = np.zeros((dim, dim), dtype=complex)
    for label, coeff in coeff_map_exyz.items():
        hmat += coeff * _pauli_matrix_exyz(label)
    return hmat


def build_problem_hamiltonian(
    *,
    problem_key: str,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
) -> Any:
    if str(problem_key).strip().lower() == "hh":
        return build_hubbard_holstein_hamiltonian(
            dims=int(num_sites),
            J=float(t),
            U=float(u),
            omega0=float(omega0),
            g=float(g_ep),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            v_t=float(dv),
            v0=None,
            t_eval=None,
            repr_mode="JW",
            indexing=str(ordering),
            pbc=(str(boundary) == "periodic"),
            include_zero_point=True,
        )
    return build_hubbard_hamiltonian(
        dims=int(num_sites),
        t=float(t),
        U=float(u),
        v=float(dv),
        repr_mode="JW",
        indexing=str(ordering),
        pbc=(str(boundary) == "periodic"),
    )


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
    stored_nq_total_raw = initial_state.get("nq_total", None)
    if stored_nq_total_raw is not None and int(stored_nq_total_raw) != int(nq_total):
        raise ValueError(
            f"ADAPT input JSON initial_state.nq_total={int(stored_nq_total_raw)} does not match expected nq_total={int(nq_total)}."
        )
    amplitudes = initial_state.get("amplitudes_qn_to_q0")
    psi = _state_from_amplitudes_qn_to_q0(amplitudes, int(nq_total))
    meta = {
        "settings": raw.get("settings", {}),
        "adapt_vqe": raw.get("adapt_vqe", {}),
        "ground_state": raw.get("ground_state", {}),
        "vqe": raw.get("vqe", {}),
        "initial_state_source": initial_state.get("source"),
        "initial_state_handoff_state_kind": initial_state.get("handoff_state_kind"),
    }
    return psi, meta


def _default_adapt_input_state(
    *,
    problem: str,
    num_sites: int,
    ordering: str,
    n_ph_max: int,
    boson_encoding: str,
) -> tuple[np.ndarray, str, str]:
    problem_key = str(problem).strip().lower()
    num_particles = half_filled_num_particles(int(num_sites))
    if problem_key == "hh":
        psi = _normalize_state(
            np.asarray(
                hubbard_holstein_reference_state(
                    dims=int(num_sites),
                    num_particles=num_particles,
                    n_ph_max=int(n_ph_max),
                    boson_encoding=str(boson_encoding),
                    indexing=str(ordering),
                ),
                dtype=complex,
            ).reshape(-1)
        )
    else:
        psi = _normalize_state(
            np.asarray(
                hartree_fock_statevector(
                    int(num_sites),
                    num_particles,
                    indexing=str(ordering),
                ),
                dtype=complex,
            ).reshape(-1)
        )
    return psi, "hf", "reference_state"


def _extract_nested(payload: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _resolve_exact_energy_from_payload(payload: Mapping[str, Any]) -> float | None:
    candidates = (
        _extract_nested(payload, "ground_state", "exact_energy_filtered"),
        _extract_nested(payload, "ground_state", "exact_energy"),
        _extract_nested(payload, "adapt_vqe", "exact_gs_energy"),
        _extract_nested(payload, "vqe", "exact_energy"),
    )
    for raw in candidates:
        if raw is None:
            continue
        try:
            value = float(raw)
        except Exception:
            continue
        if np.isfinite(value):
            return float(value)
    return None


def _validate_adapt_ref_metadata_for_exact_reuse(
    *,
    adapt_settings: Mapping[str, Any],
    args: argparse.Namespace,
    is_hh: bool,
    float_tol: float = 1e-10,
) -> list[str]:
    if not isinstance(adapt_settings, Mapping):
        return ["settings missing from adapt_ref_json"]

    mismatches: list[str] = []

    def _cmp_scalar(field: str, expected: Any, actual: Any) -> None:
        if actual != expected:
            mismatches.append(f"{field}: expected={expected!r} adapt_ref_json={actual!r}")

    def _cmp_float(field: str, expected: float, actual_raw: Any) -> None:
        try:
            actual = float(actual_raw)
        except Exception:
            mismatches.append(f"{field}: expected={expected!r} adapt_ref_json={actual_raw!r}")
            return
        if abs(float(expected) - actual) > float(float_tol):
            mismatches.append(f"{field}: expected={float(expected)!r} adapt_ref_json={actual!r}")

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


def _resolve_exact_energy_override_from_adapt_ref(
    *,
    adapt_ref_meta: Mapping[str, Any] | None,
    args: argparse.Namespace,
    problem: str,
    continuation_mode: str | None,
) -> tuple[float | None, str, list[str]]:
    if not isinstance(adapt_ref_meta, Mapping):
        return None, "computed", []
    if str(problem).strip().lower() != "hh":
        return None, "computed", []
    mode_key = str(continuation_mode if continuation_mode is not None else "legacy").strip().lower()
    if mode_key not in _HH_STAGED_CONTINUATION_MODES:
        return None, "computed", []

    mismatches = _validate_adapt_ref_metadata_for_exact_reuse(
        adapt_settings=adapt_ref_meta.get("settings", {}),
        args=args,
        is_hh=True,
    )
    if len(mismatches) > 0:
        return None, "computed", mismatches

    exact_energy = _resolve_exact_energy_from_payload(adapt_ref_meta)
    if exact_energy is None:
        return None, "computed", []
    return float(exact_energy), "adapt_ref_json", []


def _exact_gs_energy_for_problem(
    h_poly: Any,
    *,
    problem: str,
    num_sites: int,
    num_particles: tuple[int, int],
    indexing: str,
    n_ph_max: int = 1,
    boson_encoding: str = "binary",
    t: float | None = None,
    u: float | None = None,
    dv: float | None = None,
    omega0: float | None = None,
    g_ep: float | None = None,
    boundary: str = "open",
    ai_log: Callable[..., None] | None = None,
) -> float:
    """Dispatch to the correct sector-filtered exact ground energy."""
    if str(problem).strip().lower() == "hh":
        if (
            t is not None
            and u is not None
            and dv is not None
            and omega0 is not None
            and g_ep is not None
        ):
            try:
                from src.quantum.ed_hubbard_holstein import build_hh_sector_hamiltonian_ed

                h_sector = build_hh_sector_hamiltonian_ed(
                    dims=int(num_sites),
                    J=float(t),
                    U=float(u),
                    omega0=float(omega0),
                    g=float(g_ep),
                    n_ph_max=int(n_ph_max),
                    num_particles=tuple(num_particles),
                    indexing=str(indexing),
                    boson_encoding=str(boson_encoding),
                    pbc=(str(boundary).strip().lower() == "periodic"),
                    delta_v=float(dv),
                    include_zero_point=True,
                    sparse=True,
                    return_basis=False,
                )
                try:
                    from scipy.sparse import spmatrix as _spmatrix
                    from scipy.sparse.linalg import eigsh as _eigsh

                    if isinstance(h_sector, _spmatrix):
                        eval0 = _eigsh(
                            h_sector,
                            k=1,
                            which="SA",
                            return_eigenvectors=False,
                            tol=1e-10,
                            maxiter=max(1000, 10 * int(h_sector.shape[0])),
                        )
                        return float(np.real(eval0[0]))
                except Exception:
                    pass

                h_dense = np.asarray(
                    h_sector.toarray() if hasattr(h_sector, "toarray") else h_sector,
                    dtype=complex,
                )
                evals = np.linalg.eigvalsh(h_dense)
                return float(np.min(np.real(evals)))
            except Exception as exc:
                if callable(ai_log):
                    ai_log(
                        "hardcoded_adapt_hh_exact_sparse_fallback",
                        status="failed",
                        error=str(exc),
                    )
        return exact_ground_energy_sector_hh(
            h_poly,
            num_sites=int(num_sites),
            num_particles=num_particles,
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            indexing=str(indexing),
        )
    return exact_ground_energy_sector(
        h_poly,
        num_sites=int(num_sites),
        num_particles=num_particles,
        indexing=str(indexing),
    )


def _exact_reference_state_for_hh(
    *,
    num_sites: int,
    num_particles: tuple[int, int],
    indexing: str,
    n_ph_max: int,
    boson_encoding: str,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    g_ep: float,
    boundary: str,
    ai_log: Callable[..., None] | None = None,
) -> np.ndarray | None:
    try:
        from src.quantum.ed_hubbard_holstein import build_hh_sector_hamiltonian_ed
        from scipy.sparse import spmatrix as _spmatrix
        from scipy.sparse.linalg import eigsh as _eigsh

        h_sector, basis = build_hh_sector_hamiltonian_ed(
            dims=int(num_sites),
            J=float(t),
            U=float(u),
            omega0=float(omega0),
            g=float(g_ep),
            n_ph_max=int(n_ph_max),
            num_particles=tuple(num_particles),
            indexing=str(indexing),
            boson_encoding=str(boson_encoding),
            pbc=(str(boundary).strip().lower() == "periodic"),
            delta_v=float(dv),
            include_zero_point=True,
            sparse=True,
            return_basis=True,
        )
        if isinstance(h_sector, _spmatrix):
            _evals, evecs = _eigsh(
                h_sector,
                k=1,
                which="SA",
                return_eigenvectors=True,
                tol=1e-10,
                maxiter=max(1000, 10 * int(h_sector.shape[0])),
            )
            vec_sector = np.asarray(evecs[:, 0], dtype=complex).reshape(-1)
        else:
            dense = np.asarray(h_sector, dtype=complex)
            evals, evecs = np.linalg.eigh(dense)
            vec_sector = np.asarray(evecs[:, int(np.argmin(np.real(evals)))], dtype=complex).reshape(-1)
        psi_full = np.zeros(1 << int(basis.total_qubits), dtype=complex)
        for local_idx, basis_idx in enumerate(basis.basis_indices):
            psi_full[int(basis_idx)] = complex(vec_sector[int(local_idx)])
        return _normalize_state(psi_full)
    except Exception as exc:
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_exact_reference_state_unavailable",
                error=str(exc),
            )
        return None


__all__ = [
    "_HH_STAGED_CONTINUATION_MODES",
    "_build_hamiltonian_matrix",
    "_collect_hardcoded_terms_exyz",
    "_default_adapt_input_state",
    "_exact_gs_energy_for_problem",
    "_exact_reference_state_for_hh",
    "_load_adapt_initial_state",
    "_normalize_state",
    "_resolve_exact_energy_from_payload",
    "_resolve_exact_energy_override_from_adapt_ref",
    "_validate_adapt_ref_metadata_for_exact_reuse",
    "build_problem_hamiltonian",
]
