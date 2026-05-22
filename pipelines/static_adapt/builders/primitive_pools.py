"""Primitive ADAPT pool builders extracted from the static ADAPT pipeline."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from src.quantum.hubbard_latex_python_pairs import (
    bravais_nearest_neighbor_edges,
    boson_operator,
    boson_qubits_per_site,
    phonon_qubit_indices_for_site,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    HardcodedUCCSDAnsatz,
    HubbardHolsteinLayerwiseAnsatz,
    HubbardHolsteinPhysicalTermwiseAnsatz,
    HubbardTermwiseAnsatz,
    half_filled_num_particles,
)
from src.quantum.chemistry.molecular_hamiltonian import (
    build_one_body_jw_polynomial,
    build_two_body_jw_polynomial,
)
from src.quantum.chemistry.molecular_uccsd import build_molecular_uccsd_pool
from src.quantum.operator_pools.boson_chains import (
    build_bose_hubbard_blocks,
    build_bose_hubbard_full_meta_terms,
    build_bose_hubbard_hva_terms,
    build_bose_hubbard_quadratures,
    build_harmonic_kerr_chain_blocks,
    build_harmonic_kerr_chain_full_meta_terms,
    build_harmonic_kerr_chain_hva_terms,
    build_harmonic_kerr_chain_quadratures,
)
from src.quantum.operator_pools.hh_paop import make_pool as make_paop_pool
from src.quantum.operator_pools.spin_boson import (
    build_spin_boson_blocks,
    build_spin_boson_full_meta_terms,
    build_spin_boson_hva_terms,
    build_spin_boson_quadratures,
)
from .legal_subspace_filter import sanitize_pool_for_binary_boson_legal_subspace
from .lattice_hamiltonians import (
    build_extended_hubbard_blocks,
    build_extended_hubbard_hva_terms,
    build_extended_hubbard_quadratures,
    build_ionic_hubbard_blocks,
    build_ionic_hubbard_hva_terms,
    build_ionic_hubbard_quadratures,
    build_spinful_bond_charge_current_primitive,
    build_spinful_bond_charge_hopping_primitive,
    build_spinful_edge_exchange_current_primitive,
    build_spinful_edge_exchange_primitive,
    build_spinful_edge_pair_current_primitive,
    build_spinful_edge_pair_hop_primitive,
    build_spinful_opposite_spin_assisted_current_primitive,
    build_spinful_opposite_spin_assisted_hopping_primitive,
    build_spinful_three_site_bridge_current_primitive,
    build_spinful_three_site_bridge_hopping_primitive,
    build_spinless_tv_blocks,
    build_spinless_tv_hva_terms,
    build_spinless_tv_quadratures,
    build_ttprime_hubbard_blocks,
    build_ttprime_hubbard_hva_terms,
    build_ttprime_hubbard_quadratures,
)

_PAOP_IMPORT_ERROR = ""
make_phonon_motifs = None
build_vlf_sq_family = None

_HH_UCCSD_PAOP_PRODUCT_SPECS: dict[str, dict[str, Any]] = {
    "uccsd_otimes_paop_lf_std": {
        "motif_family": "paop_lf_std",
        "parameterization": "single_product",
        "adapt_visible": True,
    },
    "uccsd_otimes_paop_lf2_std": {
        "motif_family": "paop_lf2_std",
        "parameterization": "single_product",
        "adapt_visible": True,
    },
    "uccsd_otimes_paop_bond_disp_std": {
        "motif_family": "paop_bond_disp_std",
        "parameterization": "single_product",
        "adapt_visible": True,
    },
    "uccsd_otimes_paop_lf_std_seq2p": {
        "motif_family": "paop_lf_std",
        "parameterization": "double_sequential",
        "adapt_visible": True,
    },
    "uccsd_otimes_paop_lf2_std_seq2p": {
        "motif_family": "paop_lf2_std",
        "parameterization": "double_sequential",
        "adapt_visible": True,
    },
    "uccsd_otimes_paop_bond_disp_std_seq2p": {
        "motif_family": "paop_bond_disp_std",
        "parameterization": "double_sequential",
        "adapt_visible": True,
    },
}

_UCCSD_SINGLE_LABEL_RE = re.compile(r"^uccsd_sing\((alpha|beta):(\d+)->(\d+)\)$")
_UCCSD_DOUBLE_LABEL_RE = re.compile(r"^uccsd_dbl\((aa|bb|ab):(\d+),(\d+)->(\d+),(\d+)\)$")
@dataclass(frozen=True)
class _HHPhononMotif:
    family: str
    label: str
    poly: PauliPolynomial
    sites: tuple[int, ...]
    bonds: tuple[tuple[int, int], ...] = ()

_STATIC_SPINFUL_LATTICE_HAMILTONIAN_FLOW_FAMILIES = frozenset(
    {"hubbard", "ionic_hubbard", "extended_hubbard", "ttprime_hubbard"}
)


def _build_uccsd_pool(
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
) -> list[AnsatzTerm]:
    """Build the UCCSD operator pool using HardcodedUCCSDAnsatz.base_terms."""
    dummy_ansatz = HardcodedUCCSDAnsatz(
        dims=int(num_sites),
        num_particles=num_particles,
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        include_singles=True,
        include_doubles=True,
    )
    return list(dummy_ansatz.base_terms)


def _build_molecular_uccsd_pool(
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
) -> list[AnsatzTerm]:
    """Build the molecular UCCSD operator pool for the closed-shell pilot family."""
    return list(
        build_molecular_uccsd_pool(
            n_spatial_orbitals=int(num_sites),
            num_particles=tuple(int(x) for x in num_particles),
            ordering=str(ordering),
        )
    )


def _build_cse_pool(
    num_sites: int,
    ordering: str,
    t: float,
    u: float,
    dv: float,
    boundary: str,
) -> list[AnsatzTerm]:
    """Build a CSE-style pool from the term-wise Hubbard ansatz base terms."""
    dummy_ansatz = HubbardTermwiseAnsatz(
        dims=int(num_sites),
        t=float(t),
        U=float(u),
        v=float(dv),
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        pbc=(str(boundary).strip().lower() == "periodic"),
        include_potential_terms=True,
    )
    return list(dummy_ansatz.base_terms)


def _build_full_hamiltonian_pool(
    h_poly: Any,
    tol: float = 1e-12,
    normalize_coeff: bool = False,
) -> list[AnsatzTerm]:
    """Build a pool with one generator per non-identity Hamiltonian Pauli term."""
    pool: list[AnsatzTerm] = []
    terms = h_poly.return_polynomial()
    if not terms:
        return pool
    nq = int(terms[0].nqubit())
    id_label = "e" * nq

    for term in terms:
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if label == id_label:
            continue
        if abs(coeff) <= tol:
            continue
        if abs(coeff.imag) > tol:
            raise ValueError(
                f"Non-negligible imaginary Hamiltonian coefficient for term {label}: {coeff}"
            )
        generator = PauliPolynomial("JW")
        term_coeff = 1.0 if bool(normalize_coeff) else float(coeff.real)
        label_prefix = "ham_unit_term" if bool(normalize_coeff) else "ham_term"
        generator.add_term(PauliTerm(nq, ps=label, pc=float(term_coeff)))
        pool.append(AnsatzTerm(label=f"{label_prefix}({label})", polynomial=generator))
    return pool


"Built Math: H_flow := H - h_I I; U_flow(θ) = exp(-i θ H_flow), with identity removed as global phase."
def _build_full_hamiltonian_flow_pool(
    h_poly: Any,
    *,
    label: str = "ham_full",
    tol: float = 1e-12,
) -> list[AnsatzTerm]:
    """Build a single grouped Hamiltonian-flow generator for static realtime TDVP."""
    terms = h_poly.return_polynomial()
    if not terms:
        return []
    nq = int(terms[0].nqubit())
    id_label = "e" * nq
    generator = PauliPolynomial("JW")
    for term in terms:
        pauli_label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if pauli_label == id_label or abs(coeff) <= float(tol):
            continue
        if abs(coeff.imag) > float(tol):
            raise ValueError(
                f"Non-negligible imaginary Hamiltonian coefficient for term {pauli_label}: {coeff}"
            )
        generator.add_term(PauliTerm(nq, ps=pauli_label, pc=float(coeff.real)))
    if len(generator.return_polynomial()) == 0:
        return []
    return [AnsatzTerm(label=str(label), polynomial=generator)]


def _build_hamiltonian_blocks_pool(
    *,
    problem_key: str,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    v_nn: float = 0.0,
    t_prime: float = 0.0,
    omega0: float = 0.0,
    g_ep: float = 0.0,
    n_ph_max: int = 0,
    boson_encoding: str = "binary",
    ordering: str,
    boundary: str,
    include_zero_point: bool = True,
    molecular_problem: Any | None = None,
    vibronic_h2_model: Any | None = None,
) -> list[AnsatzTerm]:
    """Build a symmetry-preserving Hamiltonian-derived pool from physical grouped terms."""
    key = str(problem_key).strip().lower()
    if key == "hubbard":
        base_terms = HubbardTermwiseAnsatz(
            dims=int(num_sites),
            t=float(t),
            U=float(u),
            v=float(dv),
            reps=1,
            repr_mode="JW",
            indexing=str(ordering),
            pbc=(str(boundary).strip().lower() == "periodic"),
            include_potential_terms=True,
        ).base_terms
    elif key == "hh":
        base_terms = HubbardHolsteinPhysicalTermwiseAnsatz(
            dims=int(num_sites),
            J=float(t),
            U=float(u),
            omega0=float(omega0),
            g=float(g_ep),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            v=None,
            v_t=float(dv),
            v0=None,
            t_eval=None,
            reps=1,
            repr_mode="JW",
            indexing=str(ordering),
            pbc=(str(boundary).strip().lower() == "periodic"),
            include_zero_point=bool(include_zero_point),
        ).base_terms
    elif key == "molecular_restricted_closed_shell":
        if molecular_problem is None:
            raise ValueError(
                "molecular_problem is required for molecular_restricted_closed_shell hamiltonian_blocks."
            )
        pool: list[AnsatzTerm] = []
        one_body = _clean_real_pool_polynomial(
            build_one_body_jw_polynomial(
                molecular_problem,
                ordering=str(ordering),
            )
        )
        two_body = _clean_real_pool_polynomial(
            build_two_body_jw_polynomial(
                molecular_problem,
                ordering=str(ordering),
            )
        )
        if len(one_body.return_polynomial()) > 0:
            pool.append(
                AnsatzTerm(
                    label="ham_block::molecular_one_body",
                    polynomial=one_body,
                )
            )
        if len(two_body.return_polynomial()) > 0:
            pool.append(
                AnsatzTerm(
                    label="ham_block::molecular_two_body",
                    polynomial=two_body,
                )
            )
        return pool
    elif key == "ionic_hubbard":
        pool: list[AnsatzTerm] = []
        for label, poly in build_ionic_hubbard_blocks(
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            dv=float(dv),
            ordering=str(ordering),
            boundary=str(boundary),
        ):
            cleaned = _clean_real_pool_polynomial(poly)
            if len(cleaned.return_polynomial()) == 0:
                continue
            pool.append(AnsatzTerm(label=f"ham_block::{label}", polynomial=cleaned))
        return pool
    elif key == "extended_hubbard":
        pool = []
        for label, poly in build_extended_hubbard_blocks(
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            dv=float(dv),
            v_nn=float(v_nn),
            ordering=str(ordering),
            boundary=str(boundary),
        ):
            cleaned = _clean_real_pool_polynomial(poly)
            if len(cleaned.return_polynomial()) == 0:
                continue
            pool.append(AnsatzTerm(label=f"ham_block::{label}", polynomial=cleaned))
        return pool
    elif key == "ttprime_hubbard":
        pool = []
        for label, poly in build_ttprime_hubbard_blocks(
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            dv=float(dv),
            t_prime=float(t_prime),
            ordering=str(ordering),
            boundary=str(boundary),
        ):
            cleaned = _clean_real_pool_polynomial(poly)
            if len(cleaned.return_polynomial()) == 0:
                continue
            pool.append(AnsatzTerm(label=f"ham_block::{label}", polynomial=cleaned))
        return pool
    elif key == "spinless_tv":
        pool = []
        for label, poly in build_spinless_tv_blocks(
            num_sites=int(num_sites),
            t=float(t),
            v_nn=float(v_nn),
            dv=float(dv),
            boundary=str(boundary),
        ):
            cleaned = _clean_real_pool_polynomial(poly)
            if len(cleaned.return_polynomial()) == 0:
                continue
            pool.append(AnsatzTerm(label=f"ham_block::{label}", polynomial=cleaned))
        return pool
    elif key == "spin_boson":
        pool = []
        for label, poly in build_spin_boson_blocks(
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
        ):
            cleaned = _clean_real_pool_polynomial(poly)
            if len(cleaned.return_polynomial()) == 0:
                continue
            pool.append(AnsatzTerm(label=f"ham_block::{label}", polynomial=cleaned))
        return pool
    elif key == "bose_hubbard":
        pool = []
        for label, poly in build_bose_hubbard_blocks(
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            dv=float(dv),
            omega0=float(omega0),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            boundary=str(boundary),
            include_zero_point=bool(include_zero_point),
        ):
            cleaned = _clean_real_pool_polynomial(poly)
            if len(cleaned.return_polynomial()) == 0:
                continue
            pool.append(AnsatzTerm(label=f"ham_block::{label}", polynomial=cleaned))
        return pool
    elif key == "harmonic_kerr_chain":
        pool = []
        for label, poly in build_harmonic_kerr_chain_blocks(
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            dv=float(dv),
            omega0=float(omega0),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            boundary=str(boundary),
            include_zero_point=bool(include_zero_point),
        ):
            cleaned = _clean_real_pool_polynomial(poly)
            if len(cleaned.return_polynomial()) == 0:
                continue
            pool.append(AnsatzTerm(label=f"ham_block::{label}", polynomial=cleaned))
        return pool
    else:
        raise ValueError(f"Unsupported problem for hamiltonian_blocks pool: {problem_key!r}")
    return [
        AnsatzTerm(
            label=f"ham_block::{term.label}",
            polynomial=_clean_real_pool_polynomial(term.polynomial),
        )
        for term in base_terms
    ]


def _build_family_hva_pool(
    *,
    problem_key: str,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    v_nn: float = 0.0,
    t_prime: float = 0.0,
    omega0: float = 0.0,
    g_ep: float = 0.0,
    n_ph_max: int = 0,
    boson_encoding: str = "binary",
    ordering: str,
    boundary: str,
    include_zero_point: bool = True,
    molecular_problem: Any | None = None,
    vibronic_h2_model: Any | None = None,
) -> list[AnsatzTerm]:
    """Build family-specific HVA primitives from physical Hamiltonian terms."""
    key = str(problem_key).strip().lower()
    if key == "molecular_vibronic_h2":
        if vibronic_h2_model is None or not hasattr(vibronic_h2_model, "pool"):
            raise ValueError("molecular_vibronic_h2 HVA pool requires vibronic_h2_model.")
        return _deduplicate_pool_terms(list(getattr(vibronic_h2_model, "pool")))
    if key == "molecular_restricted_closed_shell":
        return _build_hamiltonian_blocks_pool(
            problem_key=str(problem_key),
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            dv=float(dv),
            v_nn=float(v_nn),
            t_prime=float(t_prime),
            ordering=str(ordering),
            boundary=str(boundary),
            molecular_problem=molecular_problem,
        )
    builders: dict[str, Any] = {
        "ionic_hubbard": build_ionic_hubbard_hva_terms,
        "extended_hubbard": build_extended_hubbard_hva_terms,
        "ttprime_hubbard": build_ttprime_hubbard_hva_terms,
        "spinless_tv": build_spinless_tv_hva_terms,
        "spin_boson": build_spin_boson_hva_terms,
        "bose_hubbard": build_bose_hubbard_hva_terms,
        "harmonic_kerr_chain": build_harmonic_kerr_chain_hva_terms,
    }
    builder = builders.get(key)
    if builder is None:
        raise ValueError(f"Unsupported problem for family-specific hva pool: {problem_key!r}")
    kwargs: dict[str, Any] = {
        "num_sites": int(num_sites),
        "t": float(t),
        "dv": float(dv),
    }
    if key not in {"spin_boson"}:
        kwargs["boundary"] = str(boundary)
    if key not in {"spinless_tv", "bose_hubbard", "harmonic_kerr_chain"}:
        kwargs["u"] = float(u)
        kwargs["ordering"] = str(ordering)
    elif key in {"bose_hubbard", "harmonic_kerr_chain"}:
        kwargs["u"] = float(u)
        kwargs["omega0"] = float(omega0)
        kwargs["n_ph_max"] = int(n_ph_max)
        kwargs["boson_encoding"] = str(boson_encoding)
        kwargs["include_zero_point"] = bool(include_zero_point)
    if key in {"extended_hubbard", "spinless_tv"}:
        kwargs["v_nn"] = float(v_nn)
    if key == "ttprime_hubbard":
        kwargs["t_prime"] = float(t_prime)
    if key == "spin_boson":
        kwargs.update(
            {
                "u": float(u),
                "omega0": float(omega0),
                "g_ep": float(g_ep),
                "n_ph_max": int(n_ph_max),
                "boson_encoding": str(boson_encoding),
                "ordering": str(ordering),
                "include_zero_point": bool(include_zero_point),
            }
        )
    pool: list[AnsatzTerm] = []
    for label, poly in builder(**kwargs):
        cleaned = _clean_real_pool_polynomial(poly)
        if len(cleaned.return_polynomial()) == 0:
            continue
        pool.append(AnsatzTerm(label=f"hva_term::{label}", polynomial=cleaned))
    return pool


def _build_hamiltonian_quadratures_pool(
    *,
    problem_key: str,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    v_nn: float = 0.0,
    t_prime: float = 0.0,
    omega0: float = 0.0,
    g_ep: float = 0.0,
    n_ph_max: int = 0,
    boson_encoding: str = "binary",
    ordering: str,
    boundary: str,
    include_zero_point: bool = True,
) -> list[AnsatzTerm]:
    """Build family-specific physical primitives plus quadrature partners."""
    key = str(problem_key).strip().lower()
    builders: dict[str, Any] = {
        "ionic_hubbard": build_ionic_hubbard_quadratures,
        "extended_hubbard": build_extended_hubbard_quadratures,
        "ttprime_hubbard": build_ttprime_hubbard_quadratures,
        "spinless_tv": build_spinless_tv_quadratures,
        "spin_boson": build_spin_boson_quadratures,
        "bose_hubbard": build_bose_hubbard_quadratures,
        "harmonic_kerr_chain": build_harmonic_kerr_chain_quadratures,
    }
    builder = builders.get(key)
    if builder is None:
        raise ValueError(f"Unsupported problem for hamiltonian_quadratures pool: {problem_key!r}")
    kwargs: dict[str, Any] = {
        "num_sites": int(num_sites),
        "t": float(t),
        "dv": float(dv),
    }
    if key not in {"spin_boson"}:
        kwargs["boundary"] = str(boundary)
    if key not in {"spinless_tv", "bose_hubbard", "harmonic_kerr_chain"}:
        kwargs["u"] = float(u)
        kwargs["ordering"] = str(ordering)
    elif key in {"bose_hubbard", "harmonic_kerr_chain"}:
        kwargs["u"] = float(u)
        kwargs["omega0"] = float(omega0)
        kwargs["n_ph_max"] = int(n_ph_max)
        kwargs["boson_encoding"] = str(boson_encoding)
        kwargs["include_zero_point"] = bool(include_zero_point)
    if key in {"extended_hubbard", "spinless_tv"}:
        kwargs["v_nn"] = float(v_nn)
    if key == "ttprime_hubbard":
        kwargs["t_prime"] = float(t_prime)
    if key == "spin_boson":
        kwargs.update(
            {
                "u": float(u),
                "omega0": float(omega0),
                "g_ep": float(g_ep),
                "n_ph_max": int(n_ph_max),
                "boson_encoding": str(boson_encoding),
                "ordering": str(ordering),
                "include_zero_point": bool(include_zero_point),
            }
        )
    pool: list[AnsatzTerm] = []
    for label, poly in builder(**kwargs):
        cleaned = _clean_real_pool_polynomial(poly)
        if len(cleaned.return_polynomial()) == 0:
            continue
        pool.append(AnsatzTerm(label=f"ham_quad::{label}", polynomial=cleaned))
    return pool


def _deduplicate_pool_terms(pool: list[AnsatzTerm]) -> list[AnsatzTerm]:
    """Deduplicate a pool by PauliPolynomial signature while preserving first occurrence order."""
    dedup_pool: list[AnsatzTerm] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for term in pool:
        sig = _polynomial_signature(term.polynomial)
        if sig in seen:
            continue
        seen.add(sig)
        dedup_pool.append(term)
    return dedup_pool


def _build_full_meta_pool(
    *,
    problem_key: str,
    h_poly: Any,
    num_sites: int,
    num_particles: tuple[int, int],
    t: float,
    u: float,
    dv: float,
    v_nn: float = 0.0,
    t_prime: float = 0.0,
    omega0: float = 0.0,
    g_ep: float = 0.0,
    n_ph_max: int = 0,
    boson_encoding: str = "binary",
    ordering: str,
    boundary: str,
    include_zero_point: bool = True,
    molecular_problem: Any | None = None,
    vibronic_h2_model: Any | None = None,
    return_legal_subspace_filter_meta: bool = False,
) -> list[AnsatzTerm] | tuple[list[AnsatzTerm], dict[str, Any]]:
    """Build the problem-local mega pool: union of all currently available operator families."""
    key = str(problem_key).strip().lower()

    def _finalize(pool_terms: list[AnsatzTerm]) -> list[AnsatzTerm] | tuple[list[AnsatzTerm], dict[str, Any]]:
        dedup_pool = _deduplicate_pool_terms(pool_terms)
        filtered_pool, legal_filter_meta = sanitize_pool_for_binary_boson_legal_subspace(
            dedup_pool,
            problem_key=key,
            num_sites=int(num_sites),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
        )
        if bool(return_legal_subspace_filter_meta):
            return filtered_pool, legal_filter_meta
        return filtered_pool

    if key == "hh":
        raise ValueError("HH full_meta is resolved through the HH pool preset path.")
    pool: list[AnsatzTerm] = []
    if key == "hubbard":
        pool.extend(_build_full_hamiltonian_flow_pool(h_poly=h_poly))
        pool.extend(
            _build_uccsd_pool(
                int(num_sites),
                tuple(int(x) for x in num_particles),
                str(ordering),
            )
        )
        pool.extend(
            _build_cse_pool(
                int(num_sites),
                str(ordering),
                float(t),
                float(u),
                float(dv),
                str(boundary),
            )
        )
        pool.extend(
            _build_hamiltonian_blocks_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                omega0=float(omega0),
                g_ep=float(g_ep),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                include_zero_point=bool(include_zero_point),
            )
        )
        pool.extend(_build_full_hamiltonian_pool(h_poly=h_poly))
        return _finalize(pool)
    if key == "molecular_vibronic_h2":
        if vibronic_h2_model is None or not hasattr(vibronic_h2_model, "pool"):
            raise ValueError("molecular_vibronic_h2 full_meta pool requires vibronic_h2_model.")
        pool.extend(list(getattr(vibronic_h2_model, "pool")))
        pool.extend(_build_full_hamiltonian_pool(h_poly=h_poly))
        return _finalize(pool)
    if key == "molecular_restricted_closed_shell":
        pool.extend(
            _build_molecular_uccsd_pool(
                int(num_sites),
                tuple(int(x) for x in num_particles),
                str(ordering),
            )
        )
        pool.extend(
            _build_family_hva_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                v_nn=float(v_nn),
                t_prime=float(t_prime),
                omega0=float(omega0),
                g_ep=float(g_ep),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                include_zero_point=bool(include_zero_point),
                molecular_problem=molecular_problem,
            )
        )
        pool.extend(
            _build_hamiltonian_blocks_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                v_nn=float(v_nn),
                t_prime=float(t_prime),
                omega0=float(omega0),
                g_ep=float(g_ep),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                include_zero_point=bool(include_zero_point),
                molecular_problem=molecular_problem,
            )
        )
        pool.extend(_build_full_hamiltonian_pool(h_poly=h_poly))
        return _finalize(pool)
    if key in {"ionic_hubbard", "extended_hubbard", "ttprime_hubbard"}:
        if key in _STATIC_SPINFUL_LATTICE_HAMILTONIAN_FLOW_FAMILIES:
            pool.extend(_build_full_hamiltonian_flow_pool(h_poly=h_poly))
        pool.extend(
            _build_uccsd_pool(
                int(num_sites),
                tuple(int(x) for x in num_particles),
                str(ordering),
            )
        )
        pool.extend(
            _build_hamiltonian_quadratures_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                v_nn=float(v_nn),
                t_prime=float(t_prime),
                omega0=float(omega0),
                g_ep=float(g_ep),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                include_zero_point=bool(include_zero_point),
            )
        )
        pool.extend(
            _build_family_hva_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                v_nn=float(v_nn),
                t_prime=float(t_prime),
                omega0=float(omega0),
                g_ep=float(g_ep),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                include_zero_point=bool(include_zero_point),
            )
        )
        pool.extend(
            _build_hamiltonian_blocks_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                v_nn=float(v_nn),
                t_prime=float(t_prime),
                omega0=float(omega0),
                g_ep=float(g_ep),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                include_zero_point=bool(include_zero_point),
            )
        )
        pool.extend(_build_full_hamiltonian_pool(h_poly=h_poly))
        return _finalize(pool)
    if key == "spinless_tv":
        pool.extend(_build_full_hamiltonian_flow_pool(h_poly=h_poly))
        pool.extend(
            _build_hamiltonian_quadratures_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                v_nn=float(v_nn),
                t_prime=float(t_prime),
                omega0=float(omega0),
                g_ep=float(g_ep),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                include_zero_point=bool(include_zero_point),
            )
        )
        pool.extend(
            _build_family_hva_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                v_nn=float(v_nn),
                t_prime=float(t_prime),
                omega0=float(omega0),
                g_ep=float(g_ep),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                include_zero_point=bool(include_zero_point),
            )
        )
        pool.extend(
            _build_hamiltonian_blocks_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                v_nn=float(v_nn),
                t_prime=float(t_prime),
                omega0=float(omega0),
                g_ep=float(g_ep),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                include_zero_point=bool(include_zero_point),
            )
        )
        pool.extend(_build_full_hamiltonian_pool(h_poly=h_poly))
        return _finalize(pool)
    if key == "spin_boson":
        pool.extend(_build_full_hamiltonian_flow_pool(h_poly=h_poly))
        for label, poly in build_spin_boson_full_meta_terms(
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
        ):
            cleaned = _clean_real_pool_polynomial(poly)
            if len(cleaned.return_polynomial()) == 0:
                continue
            pool.append(AnsatzTerm(label=f"full_meta::{label}", polynomial=cleaned))
        pool.extend(_build_full_hamiltonian_pool(h_poly=h_poly))
        return _finalize(pool)
    if key == "bose_hubbard":
        for label, poly in build_bose_hubbard_full_meta_terms(
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            dv=float(dv),
            omega0=float(omega0),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            boundary=str(boundary),
            include_zero_point=bool(include_zero_point),
        ):
            cleaned = _clean_real_pool_polynomial(poly)
            if len(cleaned.return_polynomial()) == 0:
                continue
            pool.append(AnsatzTerm(label=f"full_meta::{label}", polynomial=cleaned))
        pool.extend(_build_full_hamiltonian_pool(h_poly=h_poly))
        return _finalize(pool)
    if key == "harmonic_kerr_chain":
        for label, poly in build_harmonic_kerr_chain_full_meta_terms(
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            dv=float(dv),
            omega0=float(omega0),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            boundary=str(boundary),
            include_zero_point=bool(include_zero_point),
        ):
            cleaned = _clean_real_pool_polynomial(poly)
            if len(cleaned.return_polynomial()) == 0:
                continue
            pool.append(AnsatzTerm(label=f"full_meta::{label}", polynomial=cleaned))
        pool.extend(_build_full_hamiltonian_pool(h_poly=h_poly))
        return _finalize(pool)
    raise ValueError(f"Unsupported problem for full_meta pool: {problem_key!r}")


def _build_family_max_pool(
    *,
    problem_key: str,
    num_sites: int,
    num_particles: tuple[int, int],
    t: float,
    u: float,
    dv: float,
    v_nn: float = 0.0,
    t_prime: float = 0.0,
    omega0: float = 0.0,
    g_ep: float = 0.0,
    n_ph_max: int = 0,
    boson_encoding: str = "binary",
    ordering: str,
    boundary: str,
    include_zero_point: bool = True,
    molecular_problem: Any | None = None,
    vibronic_h2_model: Any | None = None,
) -> list[AnsatzTerm]:
    """Build a large structured family pool without raw flattened Hamiltonian terms."""
    key = str(problem_key).strip().lower()
    pool: list[AnsatzTerm] = []
    if key == "molecular_vibronic_h2":
        if vibronic_h2_model is None or not hasattr(vibronic_h2_model, "pool"):
            raise ValueError("molecular_vibronic_h2 full_meta pool requires vibronic_h2_model.")
        pool.extend(list(getattr(vibronic_h2_model, "pool")))
        pool.extend(_build_full_hamiltonian_pool(h_poly=h_poly))
        return _deduplicate_pool_terms(pool)
    if key == "molecular_restricted_closed_shell":
        pool.extend(
            _build_molecular_uccsd_pool(
                int(num_sites),
                tuple(int(x) for x in num_particles),
                str(ordering),
            )
        )
        pool.extend(
            _build_hamiltonian_blocks_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                v_nn=float(v_nn),
                t_prime=float(t_prime),
                ordering=str(ordering),
                boundary=str(boundary),
                molecular_problem=molecular_problem,
            )
        )
        pool.extend(
            _build_family_hva_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                v_nn=float(v_nn),
                t_prime=float(t_prime),
                ordering=str(ordering),
                boundary=str(boundary),
                molecular_problem=molecular_problem,
            )
        )
        return _deduplicate_pool_terms(pool)
    if key == "spinless_tv":
        pool.extend(
            _build_hamiltonian_quadratures_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                v_nn=float(v_nn),
                t_prime=float(t_prime),
                ordering=str(ordering),
                boundary=str(boundary),
            )
        )
        pool.extend(
            _build_family_hva_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                v_nn=float(v_nn),
                t_prime=float(t_prime),
                ordering=str(ordering),
                boundary=str(boundary),
            )
        )
        pool.extend(
            _build_hamiltonian_blocks_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                v_nn=float(v_nn),
                t_prime=float(t_prime),
                ordering=str(ordering),
                boundary=str(boundary),
            )
        )
        return _deduplicate_pool_terms(pool)
    elif key == "spin_boson":
        pool.extend(
            _build_hamiltonian_quadratures_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                omega0=float(omega0),
                g_ep=float(g_ep),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                include_zero_point=bool(include_zero_point),
            )
        )
        pool.extend(
            _build_hamiltonian_blocks_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                omega0=float(omega0),
                g_ep=float(g_ep),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                include_zero_point=bool(include_zero_point),
            )
        )
        pool.extend(
            _build_family_hva_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                omega0=float(omega0),
                g_ep=float(g_ep),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                include_zero_point=bool(include_zero_point),
            )
        )
        return _deduplicate_pool_terms(pool)
    elif key in {"bose_hubbard", "harmonic_kerr_chain"}:
        pool.extend(
            _build_hamiltonian_quadratures_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                omega0=float(omega0),
                g_ep=float(g_ep),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                include_zero_point=bool(include_zero_point),
            )
        )
        pool.extend(
            _build_hamiltonian_blocks_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                omega0=float(omega0),
                g_ep=float(g_ep),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                include_zero_point=bool(include_zero_point),
            )
        )
        pool.extend(
            _build_family_hva_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                omega0=float(omega0),
                g_ep=float(g_ep),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                include_zero_point=bool(include_zero_point),
            )
        )
        return _deduplicate_pool_terms(pool)
    elif key in {"ionic_hubbard", "extended_hubbard", "ttprime_hubbard"}:
        pool.extend(
            _build_uccsd_pool(
                int(num_sites),
                tuple(int(x) for x in num_particles),
                str(ordering),
            )
        )
        pool.extend(
            _build_hamiltonian_quadratures_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                v_nn=float(v_nn),
                t_prime=float(t_prime),
                ordering=str(ordering),
                boundary=str(boundary),
            )
        )
        pool.extend(
            _build_family_hva_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                v_nn=float(v_nn),
                t_prime=float(t_prime),
                ordering=str(ordering),
                boundary=str(boundary),
            )
        )
        pool.extend(
            _build_hamiltonian_blocks_pool(
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                v_nn=float(v_nn),
                t_prime=float(t_prime),
                ordering=str(ordering),
                boundary=str(boundary),
            )
        )
        return _deduplicate_pool_terms(pool)
    raise ValueError(f"Unsupported problem for family_max pool: {problem_key!r}")


def _polynomial_signature(poly: Any, tol: float = 1e-12) -> tuple[tuple[str, float], ...]:
    """Canonical real-valued signature for deduplicating PauliPolynomial generators."""
    items: list[tuple[str, float]] = []
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= tol:
            continue
        if abs(coeff.imag) > tol:
            raise ValueError(f"Non-negligible imaginary coefficient in pool polynomial: {coeff} ({label})")
        items.append((label, round(float(coeff.real), 12)))
    items.sort()
    return tuple(items)


def _build_hh_termwise_augmented_pool(h_poly: Any, tol: float = 1e-12) -> list[AnsatzTerm]:
    """HH-only termwise pool: unit-normalized Hamiltonian terms + x->y quadrature partners."""
    base_pool = _build_full_hamiltonian_pool(h_poly, tol=tol, normalize_coeff=True)
    if not base_pool:
        return []

    terms = h_poly.return_polynomial()
    nq = int(terms[0].nqubit())
    id_label = "e" * nq

    seen_labels: set[str] = set()
    for op in base_pool:
        op_terms = op.polynomial.return_polynomial()
        if not op_terms:
            continue
        seen_labels.add(str(op_terms[0].pw2strng()))

    aug_pool = list(base_pool)
    for term in terms:
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if label == id_label or abs(coeff) <= tol:
            continue
        if "x" not in label:
            continue
        y_label = label.replace("x", "y")
        if y_label in seen_labels:
            continue
        gen = PauliPolynomial("JW")
        y_coeff = abs(float(coeff.real))
        if y_coeff <= tol:
            y_coeff = 1.0
        gen.add_term(PauliTerm(nq, ps=y_label, pc=y_coeff))
        aug_pool.append(AnsatzTerm(label=f"ham_quadrature_term({y_label})", polynomial=gen))
        seen_labels.add(y_label)
    return aug_pool


def _chain_triples_local(*, num_sites: int, boundary: str) -> tuple[tuple[int, int, int], ...]:
    if int(num_sites) < 3:
        return tuple()
    periodic = str(boundary).strip().lower() == "periodic"
    triples: set[tuple[int, int, int]] = set()
    for site in range(int(num_sites)):
        j = int(site) + 1
        k = int(site) + 2
        if k < int(num_sites):
            triples.add((int(site), int(j), int(k)))
            continue
        if not periodic:
            continue
        triples.add((int(site), int(j % int(num_sites)), int(k % int(num_sites))))
    return tuple(sorted(triples))


def _lift_fermion_polynomial_to_hh_register(
    poly: Any,
    *,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    prune_eps: float = 0.0,
) -> PauliPolynomial:
    """Lift a fermionic-only polynomial into the HH register with a boson-identity prefix."""
    terms = poly.return_polynomial()
    if not terms:
        return PauliPolynomial("JW")
    n_sites = int(num_sites)
    ferm_nq = 2 * n_sites
    boson_bits = n_sites * int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    nq_total = ferm_nq + boson_bits
    boson_prefix = "e" * boson_bits
    lifted = PauliPolynomial("JW")
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(prune_eps):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Non-negligible imaginary coefficient in HH lifted polynomial: {coeff}")
        ferm_ps = str(term.pw2strng())
        if len(ferm_ps) != ferm_nq:
            raise ValueError(
                f"Unexpected fermion Pauli length {len(ferm_ps)} != {ferm_nq} while lifting into HH register."
            )
        lifted.add_term(PauliTerm(nq_total, ps=boson_prefix + ferm_ps, pc=float(coeff.real)))
    lifted._reduce()
    return _clean_real_pool_polynomial(lifted, prune_eps=float(prune_eps))


def _build_hva_pool(
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    include_zero_point: bool = True,
    include_lifted_uccsd: bool = True,
) -> list[AnsatzTerm]:
    layerwise = HubbardHolsteinLayerwiseAnsatz(
        dims=int(num_sites),
        J=float(t),
        U=float(u),
        omega0=float(omega0),
        g=float(g_ep),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        v=None,
        v_t=float(dv),
        v0=None,
        t_eval=None,
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        pbc=(str(boundary).strip().lower() == "periodic"),
        include_zero_point=bool(include_zero_point),
    )
    pool: list[AnsatzTerm] = list(layerwise.base_terms)
    if bool(include_lifted_uccsd):
        n_sites = int(num_sites)
        pool.extend(
            _build_hh_uccsd_fermion_lifted_pool(
                int(num_sites),
                int(n_ph_max),
                str(boson_encoding),
                str(ordering),
                str(boundary),
                num_particles=tuple(half_filled_num_particles(n_sites)),
            )
        )
    return pool


def _build_hh_uccsd_fermion_lifted_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    num_particles: tuple[int, int] | None = None,
) -> list[AnsatzTerm]:
    """HH-only UCCSD pool lifted into full HH register with boson identity prefix."""
    n_sites = int(num_sites)
    num_particles_eff = tuple(num_particles) if num_particles is not None else tuple(half_filled_num_particles(n_sites))

    uccsd_kwargs = {
        "dims": n_sites,
        "num_particles": num_particles_eff,
        "include_singles": True,
        "include_doubles": True,
        "repr_mode": "JW",
        "indexing": str(ordering),
    }
    if str(boundary).strip().lower() == "periodic":
        try:
            uccsd_kwargs["pbc"] = True
            uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)
        except TypeError as exc:
            if "pbc" not in str(exc):
                raise
            uccsd_kwargs.pop("pbc", None)
            uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)
    else:
        uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)

    lifted_pool: list[AnsatzTerm] = []
    for op in uccsd.base_terms:
        lifted = _lift_fermion_polynomial_to_hh_register(
            op.polynomial,
            num_sites=int(num_sites),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            prune_eps=0.0,
        )
        if len(lifted.return_polynomial()) == 0:
            continue
        lifted_pool.append(AnsatzTerm(label=f"uccsd_ferm_lifted::{op.label}", polynomial=lifted))
    return lifted_pool


def _build_hh_fermionic_reusable_pool(
    *,
    num_sites: int,
    t: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    prune_eps: float = 0.0,
) -> list[AnsatzTerm]:
    """Build reusable HH-valid fermionic generators lifted into the HH register."""
    n_sites = int(num_sites)
    ordering_key = str(ordering)
    boundary_key = str(boundary)
    pool: list[AnsatzTerm] = []

    def _append(label: str, poly: PauliPolynomial) -> None:
        lifted = _lift_fermion_polynomial_to_hh_register(
            poly,
            num_sites=n_sites,
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            prune_eps=float(prune_eps),
        )
        if not lifted.return_polynomial():
            return
        pool.append(AnsatzTerm(label=f"hh_fermionic_reusable::{label}", polynomial=lifted))

    nn_edges = bravais_nearest_neighbor_edges(
        int(n_sites),
        pbc=(str(boundary_key).strip().lower() == "periodic"),
    )
    for site_i, site_j in nn_edges:
        for spin_label in ("up", "dn"):
            _append(
                f"bond_charge_hop_nn_{spin_label}({int(site_i)},{int(site_j)})",
                build_spinful_bond_charge_hopping_primitive(
                    num_sites=int(n_sites),
                    site_i=int(site_i),
                    site_j=int(site_j),
                    spin=str(spin_label),
                    amplitude=float(t),
                    ordering=str(ordering_key),
                ),
            )
            _append(
                f"bond_charge_current_nn_{spin_label}({int(site_i)},{int(site_j)})",
                build_spinful_bond_charge_current_primitive(
                    num_sites=int(n_sites),
                    site_i=int(site_i),
                    site_j=int(site_j),
                    spin=str(spin_label),
                    amplitude=float(t),
                    ordering=str(ordering_key),
                ),
            )
            _append(
                f"opp_spin_assist_hop_nn_{spin_label}({int(site_i)},{int(site_j)})",
                build_spinful_opposite_spin_assisted_hopping_primitive(
                    num_sites=int(n_sites),
                    site_i=int(site_i),
                    site_j=int(site_j),
                    spin=str(spin_label),
                    amplitude=float(t),
                    ordering=str(ordering_key),
                ),
            )
            _append(
                f"opp_spin_assist_current_nn_{spin_label}({int(site_i)},{int(site_j)})",
                build_spinful_opposite_spin_assisted_current_primitive(
                    num_sites=int(n_sites),
                    site_i=int(site_i),
                    site_j=int(site_j),
                    spin=str(spin_label),
                    amplitude=float(t),
                    ordering=str(ordering_key),
                ),
            )
        _append(
            f"pair_hop_nn({int(site_i)},{int(site_j)})",
            build_spinful_edge_pair_hop_primitive(
                num_sites=int(n_sites),
                site_i=int(site_i),
                site_j=int(site_j),
                amplitude=1.0,
                ordering=str(ordering_key),
            ),
        )
        _append(
            f"pair_current_nn({int(site_i)},{int(site_j)})",
            build_spinful_edge_pair_current_primitive(
                num_sites=int(n_sites),
                site_i=int(site_i),
                site_j=int(site_j),
                amplitude=1.0,
                ordering=str(ordering_key),
            ),
        )
        _append(
            f"exchange_nn({int(site_i)},{int(site_j)})",
            build_spinful_edge_exchange_primitive(
                num_sites=int(n_sites),
                site_i=int(site_i),
                site_j=int(site_j),
                amplitude=1.0,
                ordering=str(ordering_key),
            ),
        )
        _append(
            f"exchange_current_nn({int(site_i)},{int(site_j)})",
            build_spinful_edge_exchange_current_primitive(
                num_sites=int(n_sites),
                site_i=int(site_i),
                site_j=int(site_j),
                amplitude=1.0,
                ordering=str(ordering_key),
            ),
        )

    for site_i, site_j, site_k in _chain_triples_local(num_sites=int(n_sites), boundary=str(boundary_key)):
        for spin_label in ("up", "dn"):
            _append(
                f"three_site_bridge_hop_{spin_label}({int(site_i)},{int(site_j)},{int(site_k)})",
                build_spinful_three_site_bridge_hopping_primitive(
                    num_sites=int(n_sites),
                    site_i=int(site_i),
                    site_j=int(site_j),
                    site_k=int(site_k),
                    spin=str(spin_label),
                    amplitude=float(t),
                    ordering=str(ordering_key),
                ),
            )
            _append(
                f"three_site_bridge_current_{spin_label}({int(site_i)},{int(site_j)},{int(site_k)})",
                build_spinful_three_site_bridge_current_primitive(
                    num_sites=int(n_sites),
                    site_i=int(site_i),
                    site_j=int(site_j),
                    site_k=int(site_k),
                    spin=str(spin_label),
                    amplitude=float(t),
                    ordering=str(ordering_key),
                ),
            )

    if int(n_ph_max) >= 2:
        return _deduplicate_pool_terms_lightweight(pool)
    return _deduplicate_pool_terms(pool)


def _build_paop_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    pool_key: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> list[AnsatzTerm]:
    if make_paop_pool is None:
        raise RuntimeError(f"PAOP pool requested but HH operator-pool builder unavailable: {_PAOP_IMPORT_ERROR}")

    pool_specs = make_paop_pool(
        pool_key,
        num_sites=int(num_sites),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        paop_r=int(paop_r),
        paop_split_paulis=bool(paop_split_paulis),
        paop_prune_eps=float(paop_prune_eps),
        paop_normalization=str(paop_normalization),
        num_particles=tuple(num_particles),
    )
    return [AnsatzTerm(label=label, polynomial=poly) for label, poly in pool_specs]


def _build_hh_sq_lf_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    pool_key: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
    pool_key_norm = str(pool_key).strip().lower()
    if pool_key_norm != "sq_lf_std":
        raise ValueError("HH SQ/LF primitive builder only supports pool_key='sq_lf_std'.")
    if make_paop_pool is None:
        raise RuntimeError(f"SQ/LF pool requested but HH operator-pool builder unavailable: {_PAOP_IMPORT_ERROR}")

    del paop_r  # retained for the shared HH pool-builder call signature
    pool_specs = make_paop_pool(
        pool_key_norm,
        num_sites=int(num_sites),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        paop_r=0,
        paop_split_paulis=bool(paop_split_paulis),
        paop_prune_eps=float(paop_prune_eps),
        paop_normalization=str(paop_normalization),
        num_particles=tuple(num_particles),
    )
    pool_terms = [AnsatzTerm(label=label, polynomial=poly) for label, poly in pool_specs]
    labels = [str(term.label) for term in pool_terms]
    squeeze_count = sum(
        1
        for label in labels
        if label.startswith("sq_lf_std:sq(") or label.startswith("sq_lf_std:dens_sq(")
    )
    meta = {
        "family": "sq_lf_std",
        "pool_key": pool_key_norm,
        "label_prefix": "sq_lf_std:",
        "operator_count": int(len(pool_terms)),
        "squeeze_operator_count": int(squeeze_count),
    }
    return pool_terms, meta


def _build_vlf_sq_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    pool_key: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
    if build_vlf_sq_family is None:
        raise RuntimeError(f"VLF/SQ pool requested but HH operator-pool builder unavailable: {_PAOP_IMPORT_ERROR}")
    if bool(paop_split_paulis):
        raise ValueError("VLF/SQ macro families do not support --paop-split-paulis; keep grouped macro generators intact.")
    pool_specs, meta = build_vlf_sq_family(
        pool_key,
        num_sites=int(num_sites),
        num_particles=tuple(num_particles),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        shell_radius=None,
        prune_eps=float(paop_prune_eps),
        normalization=str(paop_normalization),
    )
    return [AnsatzTerm(label=label, polynomial=poly) for label, poly in pool_specs], dict(meta)


def _clean_real_pool_polynomial(poly: Any, prune_eps: float = 0.0) -> PauliPolynomial:
    terms = poly.return_polynomial()
    if not terms:
        return PauliPolynomial("JW")
    nq = int(terms[0].nqubit())
    cleaned = PauliPolynomial("JW")
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(prune_eps):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Non-negligible imaginary coefficient in product-family pool term: {coeff}")
        cleaned.add_term(PauliTerm(nq, ps=str(term.pw2strng()), pc=float(coeff.real)))
    cleaned._reduce()
    return cleaned


def _fermion_mode_to_site(mode: int, *, num_sites: int, ordering: str) -> int:
    mode_i = int(mode)
    n_sites = int(num_sites)
    ordering_key = str(ordering).strip().lower()
    if mode_i < 0 or mode_i >= 2 * n_sites:
        raise ValueError(f"Fermion mode {mode_i} out of range for num_sites={n_sites}")
    if ordering_key == "interleaved":
        return mode_i // 2
    if ordering_key == "blocked":
        if mode_i < n_sites:
            return mode_i
        return mode_i - n_sites
    raise ValueError(f"Unsupported fermion ordering '{ordering}'.")


def _parse_lifted_uccsd_support(
    label: str,
    *,
    num_sites: int,
    ordering: str,
) -> tuple[str, tuple[int, ...]]:
    raw = str(label).strip()
    prefix = "uccsd_ferm_lifted::"
    if not raw.startswith(prefix):
        raise ValueError(f"Unsupported lifted UCCSD label '{raw}'.")
    body = raw[len(prefix):]

    m_single = _UCCSD_SINGLE_LABEL_RE.match(body)
    if m_single is not None:
        modes = [int(m_single.group(2)), int(m_single.group(3))]
        kind = "single"
    else:
        m_double = _UCCSD_DOUBLE_LABEL_RE.match(body)
        if m_double is None:
            raise ValueError(f"Could not parse lifted UCCSD label '{raw}'.")
        modes = [
            int(m_double.group(2)),
            int(m_double.group(3)),
            int(m_double.group(4)),
            int(m_double.group(5)),
        ]
        kind = "double"

    sites = tuple(
        sorted(
            {
                _fermion_mode_to_site(mode, num_sites=int(num_sites), ordering=str(ordering))
                for mode in modes
            }
        )
    )
    return kind, sites


def _motif_matches_excitation_support(
    *,
    motif: Any,
    motif_family: str,
    support_sites: tuple[int, ...],
    nearest_neighbor_bonds: set[tuple[int, int]],
) -> bool:
    support_set = {int(site) for site in support_sites}
    motif_sites = {int(site) for site in getattr(motif, "sites", ())}
    motif_bonds = tuple(tuple(sorted((int(i), int(j)))) for i, j in getattr(motif, "bonds", ()))
    if not motif_sites:
        return False
    if not motif_bonds:
        return bool(motif_sites & support_set)

    if str(motif_family).strip().lower() == "paop_bond_disp_std":
        for bond in motif_bonds:
            if set(bond).issubset(support_set):
                return True
            if bond in nearest_neighbor_bonds and bond[0] in support_set and bond[1] in support_set:
                return True
        return False

    return bool(motif_sites & support_set)


def _normalize_phonon_motif_poly(poly: PauliPolynomial, mode: str) -> PauliPolynomial:
    mode_key = str(mode).strip().lower()
    if mode_key == "none":
        return poly
    terms = poly.return_polynomial()
    if not terms:
        return poly
    if mode_key == "maxcoeff":
        max_coeff = max(abs(complex(term.p_coeff)) for term in terms)
        return poly if max_coeff <= 0.0 else (1.0 / max_coeff) * poly
    if mode_key == "fro":
        norm = sum(abs(complex(term.p_coeff)) ** 2 for term in terms) ** 0.5
        return poly if norm <= 0.0 else (1.0 / norm) * poly
    raise ValueError(f"Unknown PAOP normalization '{mode_key}'. Use none|fro|maxcoeff.")


def _make_phonon_motifs_from_paop_pool(
    motif_family: str,
    *,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    prune_eps: float,
    normalization: str,
    num_particles: tuple[int, int],
) -> list[_HHPhononMotif]:
    del ordering, num_particles
    family_key = str(motif_family).strip().lower()
    if family_key not in {"paop_lf_std", "paop_lf2_std"}:
        raise ValueError(f"PAOP motif adapter does not expose {family_key} motifs.")

    n_sites = int(num_sites)
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    nq = int(2 * n_sites + n_sites * qpb)
    repr_mode = "JW"

    def p_i(site: int) -> PauliPolynomial:
        qubits = phonon_qubit_indices_for_site(
            int(site),
            n_sites=n_sites,
            qpb=qpb,
            fermion_qubits=2 * n_sites,
        )
        b_op = boson_operator(
            repr_mode,
            nq,
            qubits,
            which="b",
            n_ph_max=int(n_ph_max),
            encoding=str(boson_encoding),
        )
        bdag_op = boson_operator(
            repr_mode,
            nq,
            qubits,
            which="bdag",
            n_ph_max=int(n_ph_max),
            encoding=str(boson_encoding),
        )
        return (1j * bdag_op) + (-1j * b_op)

    def motif(label: str, poly: PauliPolynomial, sites: tuple[int, ...], bonds: tuple[tuple[int, int], ...] = ()) -> _HHPhononMotif | None:
        cleaned = _clean_real_pool_polynomial(
            _normalize_phonon_motif_poly(poly, str(normalization)),
            float(prune_eps),
        )
        if not cleaned.return_polynomial():
            return None
        return _HHPhononMotif(
            family=family_key,
            label=str(label),
            poly=cleaned,
            sites=tuple(sorted({int(site) for site in sites})),
            bonds=tuple(tuple(sorted((int(i), int(j)))) for i, j in bonds),
        )

    motifs: list[_HHPhononMotif] = []
    p_cache = {int(site): p_i(int(site)) for site in range(n_sites)}
    for site in range(n_sites):
        item = motif(f"phonon_p(site={site})", p_cache[int(site)], (int(site),))
        if item is not None:
            motifs.append(item)

    for edge in bravais_nearest_neighbor_edges(
        n_sites,
        pbc=(str(boundary).strip().lower() == "periodic"),
    ):
        i_site = int(edge[0])
        j_site = int(edge[1])
        delta_p = p_cache[i_site] + ((-1.0) * p_cache[j_site])
        item = motif(
            f"phonon_delta_p({i_site},{j_site})",
            delta_p,
            (i_site, j_site),
            ((i_site, j_site),),
        )
        if item is not None:
            motifs.append(item)
        if family_key == "paop_lf2_std":
            item2 = motif(
                f"phonon_delta_p2({i_site},{j_site})",
                delta_p * delta_p,
                (i_site, j_site),
                ((i_site, j_site),),
            )
            if item2 is not None:
                motifs.append(item2)
    return motifs


def _build_hh_uccsd_paop_product_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    family_key: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
    *,
    allow_paop_pool_motif_adapter: bool = False,
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
    del paop_r
    motif_builder = make_phonon_motifs
    if motif_builder is None and not bool(allow_paop_pool_motif_adapter):
        raise RuntimeError(f"PAOP product pool requested but HH operator-pool builder unavailable: {_PAOP_IMPORT_ERROR}")

    family_key_norm = str(family_key).strip().lower()
    spec = _HH_UCCSD_PAOP_PRODUCT_SPECS.get(family_key_norm)
    if spec is None:
        raise ValueError(f"Unsupported HH UCCSD⊗PAOP product family '{family_key}'.")
    if bool(paop_split_paulis):
        raise ValueError("UCCSD⊗PAOP product families do not support --paop-split-paulis; keep grouped logical generators intact.")

    motif_family = str(spec["motif_family"])
    parameterization = str(spec["parameterization"])
    seq2p = parameterization == "double_sequential"
    family_label_prefix = "uccsd_otimes_paop_seq2p" if seq2p else "uccsd_otimes_paop"

    uccsd_lifted_pool = _build_hh_uccsd_fermion_lifted_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        num_particles=tuple(num_particles),
    )
    if motif_builder is None:
        motifs = _make_phonon_motifs_from_paop_pool(
            motif_family,
            num_sites=int(num_sites),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            ordering=str(ordering),
            boundary=str(boundary),
            prune_eps=float(paop_prune_eps),
            normalization=str(paop_normalization),
            num_particles=tuple(num_particles),
        )
    else:
        motifs = motif_builder(
            motif_family,
            num_sites=int(num_sites),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            boundary=str(boundary),
            prune_eps=float(paop_prune_eps),
            normalization=str(paop_normalization),
        )
    nearest_neighbor_bonds = {
        tuple(sorted((int(i), int(j))))
        for i, j in bravais_nearest_neighbor_edges(
            int(num_sites),
            pbc=(str(boundary).strip().lower() == "periodic"),
        )
    }

    sorted_uccsd = sorted(
        list(uccsd_lifted_pool),
        key=lambda op: (
            0 if _parse_lifted_uccsd_support(str(op.label), num_sites=int(num_sites), ordering=str(ordering))[0] == "single" else 1,
            str(op.label),
        ),
    )
    ordered_motifs = sorted(list(motifs), key=lambda motif: (str(motif.family), str(motif.label)))

    raw_pool: list[AnsatzTerm] = []
    raw_pair_count = 0
    for op in sorted_uccsd:
        _kind, support_sites = _parse_lifted_uccsd_support(
            str(op.label),
            num_sites=int(num_sites),
            ordering=str(ordering),
        )
        for motif in ordered_motifs:
            if not _motif_matches_excitation_support(
                motif=motif,
                motif_family=motif_family,
                support_sites=support_sites,
                nearest_neighbor_bonds=nearest_neighbor_bonds,
            ):
                continue
            raw_pair_count += 1
            base_label = f"{family_label_prefix}::{op.label}::{motif.family}::{motif.label}"
            if seq2p:
                raw_pool.append(AnsatzTerm(label=f"{base_label}::step=ferm", polynomial=op.polynomial))
                raw_pool.append(AnsatzTerm(label=f"{base_label}::step=motif", polynomial=motif.poly))
                continue
            product_poly = _clean_real_pool_polynomial(op.polynomial * motif.poly, float(paop_prune_eps))
            if not product_poly.return_polynomial():
                continue
            raw_pool.append(AnsatzTerm(label=base_label, polynomial=product_poly))

    if seq2p:
        pool = list(raw_pool)
        dedup_strategy = "disabled_pair_label_preserving"
    elif int(n_ph_max) >= 2:
        pool = _deduplicate_pool_terms_lightweight(raw_pool)
        dedup_strategy = "signature_digest"
    else:
        pool = _deduplicate_pool_terms(raw_pool)
        dedup_strategy = "signature"

    return list(pool), {
        "family": family_key_norm,
        "family_kind": "uccsd_paop_product",
        "parameterization": parameterization,
        "motif_family": motif_family,
        "locality_rule": (
            "lf_overlap"
            if motif_family in {"paop_lf_std", "paop_lf2_std"}
            else "bond_disp_local_compatible"
        ),
        "raw_sizes": {
            "raw_uccsd_lifted": int(len(uccsd_lifted_pool)),
            "raw_phonon_motifs": int(len(motifs)),
            "raw_logical_pairs": int(raw_pair_count),
            "raw_emitted_terms": int(len(raw_pool)),
        },
        "logical_element_count": int(raw_pair_count),
        "expanded_term_count": int(len(pool)),
        "dedup_strategy": dedup_strategy,
        "dedup_total": int(len(pool)),
    }


def _deduplicate_pool_terms(pool: list[AnsatzTerm]) -> list[AnsatzTerm]:
    seen: set[tuple[tuple[str, float], ...]] = set()
    dedup_pool: list[AnsatzTerm] = []
    for term in pool:
        sig = _polynomial_signature(term.polynomial)
        if sig in seen:
            continue
        seen.add(sig)
        dedup_pool.append(term)
    return dedup_pool


def _polynomial_signature_digest(poly: Any, tol: float = 1e-12) -> str:
    h = hashlib.sha1()
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Non-negligible imaginary coefficient in pool term: {coeff}")
        label = str(term.pw2strng())
        coeff_real = round(float(coeff.real), 12)
        h.update(label.encode("ascii", errors="ignore"))
        h.update(b":")
        h.update(f"{coeff_real:+.12e}".encode("ascii"))
        h.update(b";")
    return h.hexdigest()


def _deduplicate_pool_terms_lightweight(pool: list[AnsatzTerm]) -> list[AnsatzTerm]:
    seen: set[str] = set()
    dedup_pool: list[AnsatzTerm] = []
    for term in pool:
        sig = _polynomial_signature_digest(term.polynomial)
        if sig in seen:
            continue
        seen.add(sig)
        dedup_pool.append(term)
    return dedup_pool


def build_runtime_pool_terms(
    *,
    pool_key: str,
    problem_key: str,
    h_poly: Any,
    num_sites: int,
    num_particles: tuple[int, int],
    t: float,
    u: float,
    dv: float,
    v_nn: float = 0.0,
    t_prime: float = 0.0,
    omega0: float = 0.0,
    g_ep: float = 0.0,
    n_ph_max: int = 0,
    boson_encoding: str = "binary",
    ordering: str = "blocked",
    boundary: str = "open",
    include_zero_point: bool = True,
    molecular_problem: Any | None = None,
    vibronic_h2_model: Any | None = None,
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
    """Build a complete runtime candidate pool for non-HH problem families."""
    key = str(pool_key).strip().lower()
    family = str(problem_key).strip().lower()
    if family == "hh":
        raise ValueError(
            "HH runtime pool materialization stays on the legacy HH replay/preset path."
        )

    base_meta: dict[str, Any] = {
        "family": key,
        "problem_key": family,
        "candidate_pool_complete": True,
    }
    kwargs = {
        "problem_key": str(family),
        "num_sites": int(num_sites),
        "t": float(t),
        "u": float(u),
        "dv": float(dv),
        "v_nn": float(v_nn),
        "t_prime": float(t_prime),
        "omega0": float(omega0),
        "g_ep": float(g_ep),
        "n_ph_max": int(n_ph_max),
        "boson_encoding": str(boson_encoding),
        "ordering": str(ordering),
        "boundary": str(boundary),
        "include_zero_point": bool(include_zero_point),
        "molecular_problem": molecular_problem,
        "vibronic_h2_model": vibronic_h2_model,
    }
    if key == "full_hamiltonian":
        pool = _build_full_hamiltonian_pool(h_poly)
        meta = dict(base_meta)
        meta["dedup_total"] = int(len(pool))
        return list(pool), meta
    if key == "hamiltonian_blocks":
        pool = _build_hamiltonian_blocks_pool(**kwargs)
        meta = dict(base_meta)
        meta["dedup_total"] = int(len(pool))
        return list(pool), meta
    if key == "hamiltonian_quadratures":
        pool = _build_hamiltonian_quadratures_pool(**kwargs)
        meta = dict(base_meta)
        meta["dedup_total"] = int(len(pool))
        return list(pool), meta
    if key == "hva":
        pool = _build_family_hva_pool(**kwargs)
        meta = dict(base_meta)
        meta["dedup_total"] = int(len(pool))
        return list(pool), meta
    if key == "family_max":
        pool = _build_family_max_pool(
            **kwargs,
            num_particles=tuple(int(x) for x in num_particles),
        )
        meta = dict(base_meta)
        meta["dedup_total"] = int(len(pool))
        return list(pool), meta
    if key == "full_meta":
        pool, legal_filter_meta = _build_full_meta_pool(
            **kwargs,
            h_poly=h_poly,
            num_particles=tuple(int(x) for x in num_particles),
            return_legal_subspace_filter_meta=True,
        )
        meta = dict(base_meta)
        meta["dedup_total"] = int(legal_filter_meta.get("original_pool_size", len(pool)))
        meta["dedup_total_before_legal_filter"] = int(
            legal_filter_meta.get("original_pool_size", len(pool))
        )
        meta["dedup_total_after_legal_filter"] = int(len(pool))
        meta["pool_legal_subspace_filter"] = dict(legal_filter_meta)
        return list(pool), meta
    if key == "uccsd":
        pool = _build_uccsd_pool(
            int(num_sites),
            tuple(int(x) for x in num_particles),
            str(ordering),
        )
        meta = dict(base_meta)
        meta["dedup_total"] = int(len(pool))
        return list(pool), meta
    if key == "cse":
        pool = _build_cse_pool(
            int(num_sites),
            str(ordering),
            float(t),
            float(u),
            float(dv),
            str(boundary),
        )
        meta = dict(base_meta)
        meta["dedup_total"] = int(len(pool))
        return list(pool), meta
    raise ValueError(
        f"Unsupported runtime pool key {pool_key!r} for non-HH problem family {problem_key!r}."
    )


__all__ = [
    "_HH_UCCSD_PAOP_PRODUCT_SPECS",
    "_build_cse_pool",
    "_build_full_meta_pool",
    "_build_family_max_pool",
    "_build_family_hva_pool",
    "_build_full_hamiltonian_pool",
    "_build_full_hamiltonian_flow_pool",
    "_build_hamiltonian_blocks_pool",
    "_build_hamiltonian_quadratures_pool",
    "_build_molecular_uccsd_pool",
    "_build_hh_termwise_augmented_pool",
    "_build_hh_sq_lf_pool",
    "_build_hh_uccsd_fermion_lifted_pool",
    "_build_hh_uccsd_paop_product_pool",
    "_build_hva_pool",
    "_build_paop_pool",
    "_build_uccsd_pool",
    "_build_vlf_sq_pool",
    "_clean_real_pool_polynomial",
    "build_runtime_pool_terms",
    "_deduplicate_pool_terms",
    "_deduplicate_pool_terms_lightweight",
    "_fermion_mode_to_site",
    "_motif_matches_excitation_support",
    "_parse_lifted_uccsd_support",
    "_polynomial_signature",
    "_polynomial_signature_digest",
]
