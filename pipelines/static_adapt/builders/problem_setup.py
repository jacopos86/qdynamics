"""Problem setup helpers for the static ADAPT pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.quantum.hartree_fock_reference_state import (
    hartree_fock_statevector,
    mode_index as fermion_mode_index,
)
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
from src.quantum.operator_pools.boson_chains import (
    build_bose_hubbard_hamiltonian,
    build_harmonic_kerr_chain_hamiltonian,
    exact_ground_energy_boson_chain,
)
from src.quantum.operator_pools.spin_boson import (
    build_spin_boson_hamiltonian,
    exact_ground_energy_spin_boson,
)
from src.quantum.chemistry.vibronic_h2 import (
    build_cached_vibronic_h2_model,
    exact_ground_energy_physical_sector as exact_ground_energy_vibronic_physical_sector,
)
from src.quantum.chemistry.vibronic_h2o import build_cached_vibronic_h2o_model
from src.quantum.chemistry.vibronic_h2o_linear_fd import (
    load_cached_production_vibronic_h2o_linear_fd_fixture,
)
from .lattice_hamiltonians import (
    build_extended_hubbard_hamiltonian,
    build_ionic_hubbard_hamiltonian,
    build_spinless_tv_hamiltonian,
    build_ttprime_hubbard_hamiltonian,
)

_HH_STAGED_CONTINUATION_MODES = frozenset({"phase1_v1", "phase2_v1", "phase3_v1"})

PAULI_MATS = {
    "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


@dataclass(frozen=True)
class ExactReferenceStateResolution:
    state: np.ndarray | None
    available: bool
    source: str
    comparison_space_label: str
    skip_reason: str | None = None
    state_dimension: int | None = None


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
    include_zero_point: bool = True,
    molecular_vibronic_h2_fixture_json: str | Path | None = None,
    molecular_vibronic_h2o_fixture_json: str | Path | None = None,
    molecular_vibronic_h2o_linear_fd_fixture_json: str | Path | None = None,
    v_nn: float = 0.0,
    t_prime: float = 0.0,
) -> Any:
    problem_key_norm = str(problem_key).strip().lower()
    if problem_key_norm == "hh":
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
            include_zero_point=bool(include_zero_point),
        )
    if problem_key_norm == "hubbard":
        return build_hubbard_hamiltonian(
            dims=int(num_sites),
            t=float(t),
            U=float(u),
            v=float(dv),
            repr_mode="JW",
            indexing=str(ordering),
            pbc=(str(boundary) == "periodic"),
        )
    if problem_key_norm == "ionic_hubbard":
        return build_ionic_hubbard_hamiltonian(
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            dv=float(dv),
            ordering=str(ordering),
            boundary=str(boundary),
        )
    if problem_key_norm == "extended_hubbard":
        return build_extended_hubbard_hamiltonian(
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            dv=float(dv),
            v_nn=float(v_nn),
            ordering=str(ordering),
            boundary=str(boundary),
        )
    if problem_key_norm == "ttprime_hubbard":
        return build_ttprime_hubbard_hamiltonian(
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            dv=float(dv),
            t_prime=float(t_prime),
            ordering=str(ordering),
            boundary=str(boundary),
        )
    if problem_key_norm == "spinless_tv":
        return build_spinless_tv_hamiltonian(
            num_sites=int(num_sites),
            t=float(t),
            v_nn=float(v_nn),
            dv=float(dv),
            boundary=str(boundary),
        )
    if problem_key_norm == "spin_boson":
        return build_spin_boson_hamiltonian(
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            dv=float(dv),
            omega0=float(omega0),
            g_ep=float(g_ep),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            ordering=str(ordering),
            include_zero_point=bool(include_zero_point),
        )
    if problem_key_norm == "molecular_vibronic_h2":
        if int(num_sites) != 2:
            raise ValueError(f"molecular_vibronic_h2 supports L=2 only; got L={num_sites}.")
        if str(ordering).strip().lower() != "blocked":
            raise ValueError("molecular_vibronic_h2 supports ordering='blocked' only.")
        if int(n_ph_max) < 1:
            raise ValueError(f"molecular_vibronic_h2 supports n_ph_max>=1 only; got {n_ph_max}.")
        if str(boson_encoding).strip().lower() != "binary":
            raise ValueError("molecular_vibronic_h2 supports boson_encoding='binary' only.")
        if str(boundary).strip().lower() != "open":
            raise ValueError("molecular_vibronic_h2 supports boundary='open' only.")
        if not bool(include_zero_point):
            raise ValueError("molecular_vibronic_h2 fixture includes zero-point energy; include_zero_point must be true.")
        fixture_path = None if molecular_vibronic_h2_fixture_json in {None, ""} else Path(str(molecular_vibronic_h2_fixture_json))
        return build_cached_vibronic_h2_model(
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            coupling_scale=float(g_ep),
            ordering=str(ordering),
            fixture_path=fixture_path,
        ).h_vibronic
    if problem_key_norm == "molecular_vibronic_h2o":
        if int(num_sites) != 2:
            raise ValueError(f"molecular_vibronic_h2o supports active-space L=2 only; got L={num_sites}.")
        if str(ordering).strip().lower() != "blocked":
            raise ValueError("molecular_vibronic_h2o supports ordering='blocked' only.")
        if int(n_ph_max) < 1:
            raise ValueError(f"molecular_vibronic_h2o supports n_ph_max>=1 only; got {n_ph_max}.")
        if str(boson_encoding).strip().lower() != "binary":
            raise ValueError("molecular_vibronic_h2o supports boson_encoding='binary' only.")
        if str(boundary).strip().lower() != "open":
            raise ValueError("molecular_vibronic_h2o supports boundary='open' only.")
        if not bool(include_zero_point):
            raise ValueError("molecular_vibronic_h2o fixture includes zero-point energy; include_zero_point must be true.")
        fixture_path = None if molecular_vibronic_h2o_fixture_json in {None, ""} else Path(str(molecular_vibronic_h2o_fixture_json))
        return build_cached_vibronic_h2o_model(
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            coupling_scale=float(g_ep),
            ordering=str(ordering),
            fixture_path=fixture_path,
        ).h_vibronic
    if problem_key_norm == "molecular_vibronic_h2o_linear_fd":
        if molecular_vibronic_h2o_linear_fd_fixture_json in {None, ""}:
            raise ValueError(
                "molecular_vibronic_h2o_linear_fd requires "
                "molecular_vibronic_h2o_linear_fd_fixture_json."
            )
        if str(ordering).strip().lower() != "blocked":
            raise ValueError("molecular_vibronic_h2o_linear_fd supports ordering='blocked' only.")
        if str(boson_encoding).strip().lower() != "binary":
            raise ValueError("molecular_vibronic_h2o_linear_fd supports boson_encoding='binary' only.")
        if str(boundary).strip().lower() != "open":
            raise ValueError("molecular_vibronic_h2o_linear_fd supports boundary='open' only.")
        if not bool(include_zero_point):
            raise ValueError(
                "molecular_vibronic_h2o_linear_fd fixture includes zero-point energy; "
                "include_zero_point must be true."
            )
        cached = load_cached_production_vibronic_h2o_linear_fd_fixture(
            Path(str(molecular_vibronic_h2o_linear_fd_fixture_json))
        )
        model = cached.model
        if int(num_sites) != int(model.n_spatial_orbitals):
            raise ValueError(
                "molecular_vibronic_h2o_linear_fd active-space L mismatch: "
                f"got L={num_sites}, fixture has {model.n_spatial_orbitals}."
            )
        scalar_cutoff = max(int(cutoff) for cutoff in model.mode_cutoffs)
        if int(n_ph_max) != int(scalar_cutoff):
            raise ValueError(
                "molecular_vibronic_h2o_linear_fd scalar n_ph_max must match the fixture "
                f"cutoff summary {scalar_cutoff}; got {n_ph_max}."
            )
        return model.h_vibronic
    if problem_key_norm == "bose_hubbard":
        return build_bose_hubbard_hamiltonian(
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            dv=float(dv),
            omega0=float(omega0),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            boundary=str(boundary),
            include_zero_point=bool(include_zero_point),
        )
    if problem_key_norm == "harmonic_kerr_chain":
        return build_harmonic_kerr_chain_hamiltonian(
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            dv=float(dv),
            omega0=float(omega0),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            boundary=str(boundary),
            include_zero_point=bool(include_zero_point),
        )
    raise ValueError(f"Unsupported problem family for build_problem_hamiltonian: {problem_key!r}")


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
    elif problem_key == "molecular_vibronic_h2":
        if int(num_sites) != 2:
            raise ValueError(f"molecular_vibronic_h2 supports L=2 only; got L={num_sites}.")
        if str(ordering).strip().lower() != "blocked":
            raise ValueError("molecular_vibronic_h2 supports ordering='blocked' only.")
        if int(n_ph_max) < 1:
            raise ValueError(f"molecular_vibronic_h2 supports n_ph_max>=1 only; got {n_ph_max}.")
        if str(boson_encoding).strip().lower() != "binary":
            raise ValueError("molecular_vibronic_h2 supports boson_encoding='binary' only.")
        model = build_cached_vibronic_h2_model(
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            coupling_scale=1.0,
            ordering=str(ordering),
        )
        psi = _normalize_state(np.asarray(model.psi_ref, dtype=complex).reshape(-1))
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


def _exact_ground_energy_spinless_fixed_count(
    h_poly: Any,
    *,
    num_sites: int,
    n_fermions: int,
    tol: float = 1e-12,
) -> float:
    _, coeff_map = _collect_hardcoded_terms_exyz(h_poly, tol=float(tol))
    hmat = _build_hamiltonian_matrix(coeff_map)
    nq = int(round(math.log2(hmat.shape[0])))
    if nq != int(num_sites):
        raise ValueError(
            f"spinless_tv exact-energy helper expected nq={int(num_sites)}, got nq={nq}."
        )
    basis = [idx for idx in range(1 << nq) if int(idx).bit_count() == int(n_fermions)]
    if len(basis) == 0:
        raise ValueError(
            f"No spinless basis states found for num_sites={num_sites}, n_fermions={n_fermions}."
        )
    sub = hmat[np.ix_(basis, basis)]
    evals = np.linalg.eigvalsh(sub)
    return float(np.min(np.real(evals)))


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
    v_nn: float | None = None,
    t_prime: float | None = None,
    omega0: float | None = None,
    g_ep: float | None = None,
    boundary: str = "open",
    include_zero_point: bool = True,
    molecular_vibronic_h2o_linear_fd_fixture_json: str | Path | None = None,
    ai_log: Callable[..., None] | None = None,
) -> float:
    """Dispatch to the correct sector-filtered exact ground energy."""
    problem_key = str(problem).strip().lower()
    if problem_key == "hh":
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
                    include_zero_point=bool(include_zero_point),
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
    if problem_key == "molecular_restricted_closed_shell":
        return exact_ground_energy_sector(
            h_poly,
            num_sites=int(num_sites),
            num_particles=num_particles,
            indexing=str(indexing),
        )
    if problem_key == "molecular_vibronic_h2":
        return exact_ground_energy_vibronic_physical_sector(
            h_poly,
            n_spatial_orbitals=2,
            num_particles=(1, 1),
            n_ph_max=int(n_ph_max),
            boson_encoding="binary",
        )
    if problem_key == "molecular_vibronic_h2o":
        return exact_ground_energy_vibronic_physical_sector(
            h_poly,
            n_spatial_orbitals=2,
            num_particles=(1, 1),
            n_ph_max=int(n_ph_max),
            boson_encoding="binary",
        )
    if problem_key == "molecular_vibronic_h2o_linear_fd":
        if molecular_vibronic_h2o_linear_fd_fixture_json in {None, ""}:
            raise ValueError(
                "molecular_vibronic_h2o_linear_fd exact energy requires "
                "molecular_vibronic_h2o_linear_fd_fixture_json."
            )
        if str(indexing).strip().lower() != "blocked":
            raise ValueError("molecular_vibronic_h2o_linear_fd exact energy supports indexing='blocked' only.")
        if str(boson_encoding).strip().lower() != "binary":
            raise ValueError("molecular_vibronic_h2o_linear_fd exact energy supports boson_encoding='binary' only.")
        if str(boundary).strip().lower() != "open":
            raise ValueError("molecular_vibronic_h2o_linear_fd exact energy supports boundary='open' only.")
        if not bool(include_zero_point):
            raise ValueError("molecular_vibronic_h2o_linear_fd exact energy fixture includes zero-point energy.")
        cached = load_cached_production_vibronic_h2o_linear_fd_fixture(
            Path(str(molecular_vibronic_h2o_linear_fd_fixture_json))
        )
        model = cached.model
        if int(num_sites) != int(model.n_spatial_orbitals):
            raise ValueError(
                "molecular_vibronic_h2o_linear_fd exact energy active-space L mismatch: "
                f"got L={num_sites}, fixture has {model.n_spatial_orbitals}."
            )
        scalar_cutoff = max(int(cutoff) for cutoff in model.mode_cutoffs)
        if int(n_ph_max) != int(scalar_cutoff):
            raise ValueError(
                "molecular_vibronic_h2o_linear_fd exact energy scalar n_ph_max must match "
                f"the fixture cutoff summary {scalar_cutoff}; got {n_ph_max}."
            )
        if tuple(int(v) for v in num_particles) != tuple(int(v) for v in model.num_particles):
            raise ValueError(
                "molecular_vibronic_h2o_linear_fd exact energy particle-sector mismatch: "
                f"got {tuple(num_particles)}, fixture has {tuple(model.num_particles)}."
            )
        energy = cached.fixture.exact_reference.ground_energy_hartree
        if energy is None or not np.isfinite(float(energy)):
            raise ValueError("molecular_vibronic_h2o_linear_fd fixture exact ground energy is unavailable.")
        return float(energy)
    if problem_key == "spinless_tv":
        return _exact_ground_energy_spinless_fixed_count(
            h_poly,
            num_sites=int(num_sites),
            n_fermions=int(num_particles[0]),
        )
    if problem_key == "spin_boson":
        return exact_ground_energy_spin_boson(
            h_poly,
            num_sites=int(num_sites),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
        )
    if problem_key in {"bose_hubbard", "harmonic_kerr_chain"}:
        return exact_ground_energy_boson_chain(
            h_poly,
            num_sites=int(num_sites),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
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
    include_zero_point: bool = True,
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
            include_zero_point=bool(include_zero_point),
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


def _spinful_sector_basis_indices(
    *,
    n_qubits: int,
    num_sites: int,
    indexing: str,
    n_alpha: int,
    n_beta: int,
) -> list[int]:
    alpha_indices = [
        int(fermion_mode_index(site, 0, n_sites=int(num_sites), indexing=str(indexing)))
        for site in range(int(num_sites))
    ]
    beta_indices = [
        int(fermion_mode_index(site, 1, n_sites=int(num_sites), indexing=str(indexing)))
        for site in range(int(num_sites))
    ]
    basis: list[int] = []
    for idx in range(1 << int(n_qubits)):
        alpha_count = sum((int(idx) >> int(q)) & 1 for q in alpha_indices)
        beta_count = sum((int(idx) >> int(q)) & 1 for q in beta_indices)
        if alpha_count == int(n_alpha) and beta_count == int(n_beta):
            basis.append(int(idx))
    return basis


def _molecular_vibronic_h2_physical_basis_indices(
    *,
    n_qubits: int,
    n_ph_max: int,
    boson_encoding: str,
) -> list[int]:
    n_fermion_qubits = 4
    if int(n_qubits) < n_fermion_qubits:
        raise ValueError("molecular_vibronic_h2 exact basis requires at least four fermion qubits.")
    fermion_basis = _spinful_sector_basis_indices(
        n_qubits=n_fermion_qubits,
        num_sites=2,
        indexing="blocked",
        n_alpha=1,
        n_beta=1,
    )
    d = int(n_ph_max) + 1
    encoding = str(boson_encoding).strip().lower()
    if encoding == "binary":
        boson_codes = [int(level) for level in range(d)]
    elif encoding == "unary":
        boson_codes = [int(1 << level) for level in range(d)]
    else:
        raise ValueError(f"Unsupported molecular_vibronic_h2 boson encoding: {boson_encoding!r}")
    full_dim = 1 << int(n_qubits)
    basis = [
        int(f_bits + (b_bits << n_fermion_qubits))
        for b_bits in boson_codes
        for f_bits in fermion_basis
    ]
    if basis and max(basis) >= full_dim:
        raise ValueError("molecular_vibronic_h2 physical-sector basis exceeds layout dimension.")
    return basis


def _sector_ground_state_full_register(
    hmat: np.ndarray,
    *,
    basis_indices: Sequence[int],
) -> np.ndarray:
    if len(basis_indices) == 0:
        raise ValueError("Cannot build a sector ground state with an empty basis.")
    sub = hmat[np.ix_(list(basis_indices), list(basis_indices))]
    evals, evecs = np.linalg.eigh(np.asarray(sub, dtype=complex))
    vec_sector = np.asarray(
        evecs[:, int(np.argmin(np.real(evals)))],
        dtype=complex,
    ).reshape(-1)
    psi_full = np.zeros(int(hmat.shape[0]), dtype=complex)
    for local_idx, basis_idx in enumerate(basis_indices):
        psi_full[int(basis_idx)] = complex(vec_sector[int(local_idx)])
    return _normalize_state(psi_full)


def _exact_reference_state_from_h2o_linear_fd_fixture(
    resolved_problem: Any,
    *,
    comparison_space_label: str,
    total_qubits: int,
) -> ExactReferenceStateResolution:
    runtime_data = getattr(resolved_problem, "runtime_data", None)
    if not isinstance(runtime_data, Mapping):
        return ExactReferenceStateResolution(
            state=None,
            available=False,
            source="fixture_molecular_vibronic_h2o_linear_fd_exact_state",
            comparison_space_label=comparison_space_label,
            skip_reason="missing_runtime_data",
            state_dimension=None,
        )
    fixture = runtime_data.get("vibronic_h2o_linear_fd_fixture")
    exact_reference = getattr(fixture, "exact_reference", None)
    ground_state = getattr(exact_reference, "ground_state", None)
    if ground_state is None or not bool(getattr(ground_state, "available", False)):
        return ExactReferenceStateResolution(
            state=None,
            available=False,
            source="fixture_molecular_vibronic_h2o_linear_fd_exact_state",
            comparison_space_label=comparison_space_label,
            skip_reason="fixture_exact_state_unavailable",
            state_dimension=None,
        )
    representation = str(getattr(ground_state, "representation", ""))
    if representation == "external_sidecar":
        return ExactReferenceStateResolution(
            state=None,
            available=False,
            source="fixture_molecular_vibronic_h2o_linear_fd_exact_state",
            comparison_space_label=comparison_space_label,
            skip_reason="external_sidecar_exact_state_not_loaded",
            state_dimension=None,
        )
    if representation != "sparse_full_register_qn_to_q0":
        return ExactReferenceStateResolution(
            state=None,
            available=False,
            source="fixture_molecular_vibronic_h2o_linear_fd_exact_state",
            comparison_space_label=comparison_space_label,
            skip_reason="fixture_exact_state_invalid",
            state_dimension=None,
        )
    try:
        if int(getattr(ground_state, "n_qubits")) != int(total_qubits):
            raise ValueError("exact-state qubit count mismatch")
        state = np.zeros(1 << int(total_qubits), dtype=complex)
        amps = getattr(ground_state, "amplitudes_qn_to_q0", {})
        if not isinstance(amps, Mapping) or not amps:
            raise ValueError("missing exact-state amplitudes")
        for bitstr, coeff_payload in amps.items():
            if not isinstance(bitstr, str) or len(bitstr) != int(total_qubits) or any(ch not in "01" for ch in bitstr):
                raise ValueError(f"invalid exact-state bitstring {bitstr!r}")
            if not isinstance(coeff_payload, Mapping):
                raise ValueError(f"invalid exact-state amplitude payload for {bitstr!r}")
            state[int(bitstr, 2)] = complex(
                float(coeff_payload.get("re", 0.0)),
                float(coeff_payload.get("im", 0.0)),
            )
        raw_norm = float(np.vdot(state, state).real)
        declared_norm = float(getattr(ground_state, "norm", raw_norm))
        if not np.isfinite(raw_norm) or raw_norm <= 0.0:
            raise ValueError("embedded exact state has non-positive norm")
        if abs(raw_norm - declared_norm) > 1.0e-8 or abs(raw_norm - 1.0) > 1.0e-8:
            raise ValueError("embedded exact-state amplitudes are not normalized")
        state = _normalize_state(state)
        return ExactReferenceStateResolution(
            state=state,
            available=True,
            source="fixture_molecular_vibronic_h2o_linear_fd_exact_state",
            comparison_space_label=comparison_space_label,
            skip_reason=None,
            state_dimension=int(state.size),
        )
    except Exception:
        return ExactReferenceStateResolution(
            state=None,
            available=False,
            source="fixture_molecular_vibronic_h2o_linear_fd_exact_state",
            comparison_space_label=comparison_space_label,
            skip_reason="fixture_exact_state_invalid",
            state_dimension=None,
        )


def resolve_exact_reference_state_for_problem(
    h_poly: Any,
    *,
    resolved_problem: Any,
    ai_log: Callable[..., None] | None = None,
    max_dense_dim: int = 8192,
) -> ExactReferenceStateResolution:
    problem_key = str(getattr(resolved_problem, "family_key", "")).strip().lower()
    comparison_space_label = str(
        getattr(
            getattr(resolved_problem, "exact_target", None),
            "comparison_space_label",
            getattr(resolved_problem, "exact_comparison_space_label", "unknown"),
        )
    )
    total_qubits = int(getattr(getattr(resolved_problem, "layout", None), "total_qubits", 0))
    if total_qubits <= 0:
        return ExactReferenceStateResolution(
            state=None,
            available=False,
            source="unavailable",
            comparison_space_label=comparison_space_label,
            skip_reason="missing_layout",
            state_dimension=None,
        )

    dim = 1 << int(total_qubits)
    if problem_key == "molecular_vibronic_h2o_linear_fd":
        return _exact_reference_state_from_h2o_linear_fd_fixture(
            resolved_problem,
            comparison_space_label=comparison_space_label,
            total_qubits=int(total_qubits),
        )
    if int(dim) > int(max_dense_dim):
        return ExactReferenceStateResolution(
            state=None,
            available=False,
            source="dense_solver_guard",
            comparison_space_label=comparison_space_label,
            skip_reason="dense_dimension_cap_exceeded",
            state_dimension=int(dim),
        )

    if problem_key == "hh":
        request = getattr(resolved_problem, "request", None)
        sector = getattr(resolved_problem, "sector", None)
        state = _exact_reference_state_for_hh(
            num_sites=int(getattr(request, "num_sites", 0)),
            num_particles=tuple(getattr(sector, "num_particles", (0, 0))),
            indexing=str(getattr(request, "ordering", "blocked")),
            n_ph_max=int(getattr(request, "n_ph_max", 1)),
            boson_encoding=str(getattr(request, "boson_encoding", "binary")),
            t=float(getattr(request, "t", 0.0)),
            u=float(getattr(request, "u", 0.0)),
            dv=float(getattr(request, "dv", 0.0)),
            omega0=float(getattr(request, "omega0", 0.0)),
            g_ep=float(getattr(request, "g_ep", 0.0)),
            boundary=str(getattr(request, "boundary", "open")),
            include_zero_point=bool(getattr(request, "include_zero_point", True)),
            ai_log=ai_log,
        )
        return ExactReferenceStateResolution(
            state=state,
            available=state is not None,
            source=("hh_sector_sparse" if state is not None else "hh_sector_sparse"),
            comparison_space_label=comparison_space_label,
            skip_reason=(None if state is not None else "hh_exact_state_unavailable"),
            state_dimension=(None if state is None else int(state.size)),
        )

    try:
        _, coeff_map = _collect_hardcoded_terms_exyz(h_poly)
        hmat = _build_hamiltonian_matrix(coeff_map)
        if int(hmat.shape[0]) != int(dim):
            return ExactReferenceStateResolution(
                state=None,
                available=False,
                source="dense_full_register",
                comparison_space_label=comparison_space_label,
                skip_reason="layout_dimension_mismatch",
                state_dimension=int(hmat.shape[0]),
            )
        request = getattr(resolved_problem, "request", None)
        sector = getattr(resolved_problem, "sector", None)
        num_particles = tuple(getattr(sector, "num_particles", ()) or getattr(resolved_problem, "default_num_particles", (0, 0)))

        if problem_key in {
            "hubbard",
            "ionic_hubbard",
            "extended_hubbard",
            "ttprime_hubbard",
            "molecular_restricted_closed_shell",
        }:
            basis = _spinful_sector_basis_indices(
                n_qubits=int(total_qubits),
                num_sites=int(getattr(request, "num_sites", 0)),
                indexing=str(getattr(request, "ordering", "blocked")),
                n_alpha=int(num_particles[0]),
                n_beta=int(num_particles[1]),
            )
            if len(basis) == 0:
                raise ValueError("empty_sector_basis")
            state = _sector_ground_state_full_register(hmat, basis_indices=basis)
            source = "dense_spin_sector"
        elif problem_key in {"molecular_vibronic_h2", "molecular_vibronic_h2o"}:
            basis = _molecular_vibronic_h2_physical_basis_indices(
                n_qubits=int(total_qubits),
                n_ph_max=int(getattr(request, "n_ph_max", 1)),
                boson_encoding=str(getattr(request, "boson_encoding", "binary")),
            )
            if len(basis) == 0:
                raise ValueError("empty_physical_basis")
            state = _sector_ground_state_full_register(hmat, basis_indices=basis)
            source = f"dense_{problem_key}_physical_sector"
        elif problem_key == "spinless_tv":
            n_fermions = int(num_particles[0]) if len(num_particles) > 0 else 0
            basis = [idx for idx in range(int(dim)) if int(idx).bit_count() == int(n_fermions)]
            if len(basis) == 0:
                raise ValueError("empty_sector_basis")
            state = _sector_ground_state_full_register(hmat, basis_indices=basis)
            source = "dense_spinless_sector"
        elif problem_key in {"spin_boson", "bose_hubbard", "harmonic_kerr_chain"}:
            evals, evecs = np.linalg.eigh(np.asarray(hmat, dtype=complex))
            state = _normalize_state(
                np.asarray(
                    evecs[:, int(np.argmin(np.real(evals)))],
                    dtype=complex,
                ).reshape(-1)
            )
            source = "dense_full_register"
        else:
            return ExactReferenceStateResolution(
                state=None,
                available=False,
                source="unavailable",
                comparison_space_label=comparison_space_label,
                skip_reason="unsupported_family",
                state_dimension=int(dim),
            )
        return ExactReferenceStateResolution(
            state=state,
            available=True,
            source=source,
            comparison_space_label=comparison_space_label,
            skip_reason=None,
            state_dimension=int(state.size),
        )
    except ValueError as exc:
        skip_reason = str(exc)
        if skip_reason not in {"empty_sector_basis"}:
            skip_reason = "diagonalization_failed"
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        skip_reason = "diagonalization_failed"
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_exact_reference_state_unavailable",
                problem=str(problem_key),
                error=str(exc),
            )
    return ExactReferenceStateResolution(
        state=None,
        available=False,
        source="dense_full_register",
        comparison_space_label=comparison_space_label,
        skip_reason=skip_reason,
        state_dimension=int(dim),
    )


__all__ = [
    "ExactReferenceStateResolution",
    "_HH_STAGED_CONTINUATION_MODES",
    "_build_hamiltonian_matrix",
    "_collect_hardcoded_terms_exyz",
    "_default_adapt_input_state",
    "_exact_gs_energy_for_problem",
    "_exact_reference_state_for_hh",
    "resolve_exact_reference_state_for_problem",
    "_load_adapt_initial_state",
    "_normalize_state",
    "_resolve_exact_energy_from_payload",
    "_resolve_exact_energy_override_from_adapt_ref",
    "_validate_adapt_ref_metadata_for_exact_reuse",
    "build_problem_hamiltonian",
]
