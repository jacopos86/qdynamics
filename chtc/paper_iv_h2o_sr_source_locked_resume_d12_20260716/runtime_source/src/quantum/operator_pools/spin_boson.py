"""Spin-boson / generalized Rabi Hamiltonian and operator-pool helpers.

This module keeps the spin-boson family operators in one place so the static
ADAPT pipeline can import a large, Hamiltonian-specific generator list from a
central operator-pool package.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.quantum.hubbard_latex_python_pairs import (
    SPIN_DN,
    SPIN_UP,
    boson_displacement_operator,
    boson_operator,
    boson_qubits_per_site,
    jw_number_operator,
    mode_index,
    phonon_qubit_indices_for_site,
)
from src.quantum.pauli_polynomial_class import (
    PauliPolynomial,
    fermion_minus_operator,
    fermion_plus_operator,
)
from src.quantum.qubitization_module import PauliTerm


_SPIN_BOSON_HAMILTONIAN_FORMULA = (
    "H_sb = -t T + dv Z + sum_i omega0 (b_i^dag b_i + include_zero_point/2) "
    "+ sum_i u X_i T + sum_i g_ep X_i Z"
)


_SPIN_BOSON_REFERENCE_FORMULA = (
    "|phi_ref> = |0_b> ⊗ |gs(-t T + dv Z)> on the one-electron emitter sector"
)


def _require_single_emitter(num_sites: int) -> int:
    n_sites = int(num_sites)
    if n_sites < 1:
        raise ValueError(f"spin_boson requires num_sites >= 1; got num_sites={n_sites}.")
    return int(n_sites)


def _identity_poly(nq: int) -> PauliPolynomial:
    return PauliPolynomial("JW", [PauliTerm(int(nq), ps="e" * int(nq), pc=1.0)])


def _single_qubit_poly(nq: int, qubit: int, letter: str) -> PauliPolynomial:
    word = ["e"] * int(nq)
    word[int(nq) - 1 - int(qubit)] = str(letter)
    return PauliPolynomial("JW", [PauliTerm(int(nq), ps="".join(word), pc=1.0)])


def _emitter_mode_indices(ordering: str) -> tuple[int, int]:
    return (
        int(mode_index(0, SPIN_UP, indexing=str(ordering), n_sites=1)),
        int(mode_index(0, SPIN_DN, indexing=str(ordering), n_sites=1)),
    )


def _boson_register(
    n_ph_max: int,
    boson_encoding: str,
    *,
    site: int = 0,
    num_sites: int = 1,
) -> tuple[int, list[int]]:
    n_sites = _require_single_emitter(int(num_sites))
    site_index = int(site)
    if site_index < 0 or site_index >= n_sites:
        raise ValueError(f"spin_boson boson site out of range: site={site_index}, num_sites={n_sites}")
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    qubits = phonon_qubit_indices_for_site(
        site_index,
        n_sites=n_sites,
        qpb=int(qpb),
        fermion_qubits=2,
    )
    return int(qpb), list(qubits)


def _boson_code_bits(*, n_ph_max: int, boson_encoding: str) -> list[int]:
    d = int(n_ph_max) + 1
    encoding_key = str(boson_encoding).strip().lower()
    if encoding_key == "binary":
        return [int(level) for level in range(d)]
    if encoding_key == "unary":
        return [int(1 << level) for level in range(d)]
    raise ValueError(f"Unsupported boson encoding {boson_encoding!r} for spin_boson.")


def _boson_vacuum_statevector(n_ph_max: int, boson_encoding: str, *, num_sites: int = 1) -> np.ndarray:
    n_sites = _require_single_emitter(int(num_sites))
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    dim = 1 << int(qpb * n_sites)
    psi = np.zeros(dim, dtype=complex)
    code0 = int(_boson_code_bits(n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))[0])
    basis_index = 0
    for site in range(n_sites):
        basis_index |= int(code0) << int(site * qpb)
    psi[int(basis_index)] = 1.0 + 0.0j
    return psi


def _emitter_flip_operator(*, ordering: str, nq_total: int) -> PauliPolynomial:
    g_mode, e_mode = _emitter_mode_indices(str(ordering))
    return (
        fermion_plus_operator("JW", int(nq_total), int(g_mode))
        * fermion_minus_operator("JW", int(nq_total), int(e_mode))
        + fermion_plus_operator("JW", int(nq_total), int(e_mode))
        * fermion_minus_operator("JW", int(nq_total), int(g_mode))
    )


def _emitter_imbalance_operator(*, ordering: str, nq_total: int) -> PauliPolynomial:
    g_mode, e_mode = _emitter_mode_indices(str(ordering))
    n_g = jw_number_operator("JW", int(nq_total), int(g_mode))
    n_e = jw_number_operator("JW", int(nq_total), int(e_mode))
    return n_e + ((-1.0) * n_g)


def _boson_number_operator(
    *,
    nq_total: int,
    n_ph_max: int,
    boson_encoding: str,
    site: int = 0,
    num_sites: int = 1,
) -> PauliPolynomial:
    _qpb, qubits = _boson_register(
        int(n_ph_max),
        str(boson_encoding),
        site=int(site),
        num_sites=int(num_sites),
    )
    b = boson_operator(
        "JW",
        int(nq_total),
        qubits,
        which="b",
        n_ph_max=int(n_ph_max),
        encoding=str(boson_encoding),
    )
    bdag = boson_operator(
        "JW",
        int(nq_total),
        qubits,
        which="bdag",
        n_ph_max=int(n_ph_max),
        encoding=str(boson_encoding),
    )
    return bdag * b


def _boson_displacement(
    *,
    nq_total: int,
    n_ph_max: int,
    boson_encoding: str,
    site: int = 0,
    num_sites: int = 1,
) -> PauliPolynomial:
    _qpb, qubits = _boson_register(
        int(n_ph_max),
        str(boson_encoding),
        site=int(site),
        num_sites=int(num_sites),
    )
    return boson_displacement_operator(
        "JW",
        int(nq_total),
        qubits,
        n_ph_max=int(n_ph_max),
        encoding=str(boson_encoding),
    )


def _boson_momentum(
    *,
    nq_total: int,
    n_ph_max: int,
    boson_encoding: str,
    site: int = 0,
    num_sites: int = 1,
) -> PauliPolynomial:
    _qpb, qubits = _boson_register(
        int(n_ph_max),
        str(boson_encoding),
        site=int(site),
        num_sites=int(num_sites),
    )
    b = boson_operator(
        "JW",
        int(nq_total),
        qubits,
        which="b",
        n_ph_max=int(n_ph_max),
        encoding=str(boson_encoding),
    )
    bdag = boson_operator(
        "JW",
        int(nq_total),
        qubits,
        which="bdag",
        n_ph_max=int(n_ph_max),
        encoding=str(boson_encoding),
    )
    return (1j * bdag) + ((-1j) * b)


def _spin_y_polynomial(*, nq_total: int, ordering: str) -> PauliPolynomial:
    g_mode, e_mode = _emitter_mode_indices(str(ordering))
    current = (
        fermion_plus_operator("JW", int(nq_total), int(g_mode))
        * fermion_minus_operator("JW", int(nq_total), int(e_mode))
        + ((-1.0) * (
            fermion_plus_operator("JW", int(nq_total), int(e_mode))
            * fermion_minus_operator("JW", int(nq_total), int(g_mode))
        ))
    )
    return (-1j) * current


def _clean_real_poly(poly: PauliPolynomial, tol: float = 1e-12) -> PauliPolynomial:
    cleaned = PauliPolynomial("JW")
    terms = poly.return_polynomial()
    if not terms:
        return cleaned
    nq = int(terms[0].nqubit())
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"spin_boson operator has non-negligible imaginary coefficient: {coeff}")
        cleaned.add_term(PauliTerm(int(nq), ps=str(term.pw2strng()), pc=float(coeff.real)))
    cleaned._reduce()
    return cleaned


def _symmetrized_product(lhs: PauliPolynomial, rhs: PauliPolynomial) -> PauliPolynomial:
    return 0.5 * ((lhs * rhs) + (rhs * lhs))


def build_spin_boson_hamiltonian(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str = "blocked",
    include_zero_point: bool = True,
) -> PauliPolynomial:
    """Math: H_sb = -t T + dv Z + omega0 (n_b + 1/2) + u X_b T + g_ep X_b Z."""
    n_sites = _require_single_emitter(int(num_sites))
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    nq_total = 2 + int(n_sites * qpb)
    flip = _emitter_flip_operator(ordering=str(ordering), nq_total=int(nq_total))
    imbalance = _emitter_imbalance_operator(ordering=str(ordering), nq_total=int(nq_total))
    h_poly = ((-float(t)) * flip) + (float(dv) * imbalance)
    for site in range(n_sites):
        x_b = _boson_displacement(
            nq_total=int(nq_total),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            site=int(site),
            num_sites=int(n_sites),
        )
        n_b = _boson_number_operator(
            nq_total=int(nq_total),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            site=int(site),
            num_sites=int(n_sites),
        )
        h_poly = h_poly + (float(omega0) * n_b)
        if abs(float(u)) > 0.0:
            h_poly = h_poly + (float(u) * (x_b * flip))
        if abs(float(g_ep)) > 0.0:
            h_poly = h_poly + (float(g_ep) * (x_b * imbalance))
    if bool(include_zero_point):
        h_poly = h_poly + (0.5 * float(omega0) * int(n_sites) * _identity_poly(int(nq_total)))
    h_poly._reduce()
    return h_poly


def build_spin_boson_reference_statevector(
    *,
    num_sites: int,
    t: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
) -> np.ndarray:
    """Math: |phi_ref> = |0_b> ⊗ |gs(-t T + dv Z)> on the one-electron emitter sector."""
    n_sites = _require_single_emitter(int(num_sites))
    matter_h = np.array(
        [[-float(dv), -float(t)], [-float(t), float(dv)]],
        dtype=complex,
    )
    evals, evecs = np.linalg.eigh(matter_h)
    gs_vec = np.asarray(evecs[:, int(np.argmin(np.real(evals)))], dtype=complex).reshape(2)
    emitter_state = np.zeros(4, dtype=complex)
    emitter_state[1] = complex(gs_vec[0])
    emitter_state[2] = complex(gs_vec[1])
    boson_vac = _boson_vacuum_statevector(
        int(n_ph_max),
        str(boson_encoding),
        num_sites=int(n_sites),
    )
    psi = np.kron(np.asarray(boson_vac, dtype=complex), emitter_state)
    norm = float(np.linalg.norm(psi))
    if norm <= 0.0:
        raise ValueError("spin_boson reference state has zero norm.")
    return np.asarray(psi / norm, dtype=complex)


def exact_ground_energy_spin_boson(
    h_poly: Any,
    *,
    n_ph_max: int,
    boson_encoding: str,
    num_sites: int | None = None,
    tol: float = 1e-12,
) -> float:
    """Math: E0 = min eig(H restricted to the one-emitter physical truncated-boson sector)."""
    coeff_map: dict[str, complex] = {}
    order: list[str] = []
    for term in h_poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        if label not in coeff_map:
            coeff_map[label] = 0.0 + 0.0j
            order.append(label)
        coeff_map[label] += coeff
    if not order:
        return 0.0
    nq = len(order[0])
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    if num_sites is None:
        boson_width = int(nq) - 2
        if boson_width < 0 or boson_width % int(qpb) != 0:
            raise ValueError(
                f"spin_boson exact-energy helper cannot infer num_sites from nq={nq}, qpb={qpb}."
            )
        n_sites = int(boson_width // int(qpb))
    else:
        n_sites = _require_single_emitter(int(num_sites))
        expected_nq = 2 + int(n_sites * qpb)
        if int(nq) != int(expected_nq):
            raise ValueError(f"spin_boson exact-energy helper expected nq={expected_nq}, got nq={nq}.")
    dim = 1 << int(nq)
    mats = {
        "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
        "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    }
    hmat = np.zeros((dim, dim), dtype=complex)
    for label in order:
        op = mats[label[0]]
        for ch in label[1:]:
            op = np.kron(op, mats[ch])
        hmat += coeff_map[label] * op
    physical_boson_bits = _boson_code_bits(
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
    )
    basis: list[int] = []
    for levels in np.ndindex(*([len(physical_boson_bits)] * n_sites)):
        boson_bits = 0
        for site, level_index in enumerate(levels):
            boson_bits |= int(physical_boson_bits[int(level_index)]) << int(site * qpb)
        for emitter_bits in (1, 2):
            basis.append(int(emitter_bits + (boson_bits << 2)))
    if len(basis) == 0:
        raise ValueError("spin_boson exact-energy helper found no physical basis states.")
    sub = hmat[np.ix_(basis, basis)]
    evals = np.linalg.eigvalsh(sub)
    return float(np.min(np.real(evals)))


def build_spin_boson_blocks(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str = "blocked",
    include_zero_point: bool = True,
) -> list[tuple[str, PauliPolynomial]]:
    """Math: grouped physical blocks {T, Z, n_b, X_b T, X_b Z}."""
    n_sites = _require_single_emitter(int(num_sites))
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    nq_total = 2 + int(n_sites * qpb)
    flip = _clean_real_poly(_emitter_flip_operator(ordering=str(ordering), nq_total=int(nq_total)))
    imbalance = _clean_real_poly(_emitter_imbalance_operator(ordering=str(ordering), nq_total=int(nq_total)))
    blocks: list[tuple[str, PauliPolynomial]] = [
        ("emitter_flip", (-float(t)) * flip),
        ("emitter_imbalance", float(dv) * imbalance),
    ]
    for site in range(n_sites):
        suffix = "" if n_sites == 1 else f"_{site}"
        n_b = _clean_real_poly(
            _boson_number_operator(
                nq_total=int(nq_total),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                site=int(site),
                num_sites=int(n_sites),
            )
        )
        x_b = _clean_real_poly(
            _boson_displacement(
                nq_total=int(nq_total),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                site=int(site),
                num_sites=int(n_sites),
            )
        )
        blocks.append((f"boson_number{suffix}", float(omega0) * n_b))
        blocks.append((f"boson_displacement{suffix}", x_b))
        if abs(float(u)) > 0.0:
            blocks.append((f"transverse_coupling{suffix}", _clean_real_poly(float(u) * (x_b * flip))))
        if abs(float(g_ep)) > 0.0:
            blocks.append((f"longitudinal_coupling{suffix}", _clean_real_poly(float(g_ep) * (x_b * imbalance))))
    return blocks


def build_spin_boson_hva_terms(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str = "blocked",
    include_zero_point: bool = True,
) -> list[tuple[str, PauliPolynomial]]:
    """Math: HVA terms reuse the grouped physical Hamiltonian blocks."""
    return list(
        build_spin_boson_blocks(
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
    )


def build_spin_boson_quadratures(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str = "blocked",
    include_zero_point: bool = True,
) -> list[tuple[str, PauliPolynomial]]:
    """Math: enlarged structured pool using X/P boson channels and emitter flip/imbalance partners."""
    n_sites = _require_single_emitter(int(num_sites))
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    nq_total = 2 + int(n_sites * qpb)
    flip = _clean_real_poly(_emitter_flip_operator(ordering=str(ordering), nq_total=int(nq_total)))
    imbalance = _clean_real_poly(_emitter_imbalance_operator(ordering=str(ordering), nq_total=int(nq_total)))
    spin_y = _clean_real_poly(_spin_y_polynomial(ordering=str(ordering), nq_total=int(nq_total)))
    pool: list[tuple[str, PauliPolynomial]] = [
        ("emitter_flip", flip),
        ("emitter_imbalance", imbalance),
        ("emitter_y", spin_y),
    ]
    for site in range(n_sites):
        suffix = "" if n_sites == 1 else f"_{site}"
        x_b = _clean_real_poly(
            _boson_displacement(
                nq_total=int(nq_total),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                site=int(site),
                num_sites=int(n_sites),
            )
        )
        p_b = _clean_real_poly(
            _boson_momentum(
                nq_total=int(nq_total),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                site=int(site),
                num_sites=int(n_sites),
            )
        )
        n_b = _clean_real_poly(
            _boson_number_operator(
                nq_total=int(nq_total),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                site=int(site),
                num_sites=int(n_sites),
            )
        )
        pool.extend(
            [
                (f"boson_number{suffix}", n_b),
                (f"boson_displacement{suffix}", x_b),
                (f"boson_momentum{suffix}", p_b),
                (f"longitudinal_x{suffix}", _clean_real_poly(x_b * imbalance)),
                (f"longitudinal_p{suffix}", _clean_real_poly(p_b * imbalance)),
                (f"transverse_x{suffix}", _clean_real_poly(x_b * flip)),
                (f"transverse_p{suffix}", _clean_real_poly(p_b * flip)),
                (f"number_weighted_imbalance{suffix}", _clean_real_poly(n_b * imbalance)),
                (f"number_weighted_flip{suffix}", _clean_real_poly(n_b * flip)),
            ]
        )
    return pool


def build_spin_boson_full_meta_terms(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str = "blocked",
    include_zero_point: bool = True,
) -> list[tuple[str, PauliPolynomial]]:
    """Math: full_meta = all current spin-boson structured generators for this family."""
    n_sites = _require_single_emitter(int(num_sites))
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    nq_total = 2 + int(n_sites * qpb)
    flip = _clean_real_poly(_emitter_flip_operator(ordering=str(ordering), nq_total=int(nq_total)))
    imbalance = _clean_real_poly(_emitter_imbalance_operator(ordering=str(ordering), nq_total=int(nq_total)))
    spin_y = _clean_real_poly(_spin_y_polynomial(ordering=str(ordering), nq_total=int(nq_total)))
    pool = list(
        build_spin_boson_quadratures(
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
    )
    for site in range(n_sites):
        suffix = "" if n_sites == 1 else f"_{site}"
        x_b = _clean_real_poly(
            _boson_displacement(
                nq_total=int(nq_total),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                site=int(site),
                num_sites=int(n_sites),
            )
        )
        p_b = _clean_real_poly(
            _boson_momentum(
                nq_total=int(nq_total),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                site=int(site),
                num_sites=int(n_sites),
            )
        )
        n_b = _clean_real_poly(
            _boson_number_operator(
                nq_total=int(nq_total),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                site=int(site),
                num_sites=int(n_sites),
            )
        )
        x_sq = _clean_real_poly(x_b * x_b)
        p_sq = _clean_real_poly(p_b * p_b)
        n_sq = _clean_real_poly(n_b * n_b)
        squeeze_x = _clean_real_poly(_clean_real_poly((x_sq + ((-1.0) * p_sq))))
        squeeze_p = _clean_real_poly(_clean_real_poly((x_b * p_b) + (p_b * x_b)))
        pool.extend(
            [
                (f"boson_x_sq{suffix}", x_sq),
                (f"boson_p_sq{suffix}", p_sq),
                (f"boson_n_sq{suffix}", n_sq),
                (f"boson_squeeze_x{suffix}", squeeze_x),
                (f"boson_xp_sym{suffix}", squeeze_p),
                (f"x_sq_imbalance{suffix}", _clean_real_poly(x_sq * imbalance)),
                (f"p_sq_imbalance{suffix}", _clean_real_poly(p_sq * imbalance)),
                (f"n_sq_imbalance{suffix}", _clean_real_poly(n_sq * imbalance)),
                (f"x_sq_flip{suffix}", _clean_real_poly(x_sq * flip)),
                (f"p_sq_flip{suffix}", _clean_real_poly(p_sq * flip)),
                (f"n_sq_flip{suffix}", _clean_real_poly(n_sq * flip)),
                (f"x_sq_emitter_y{suffix}", _clean_real_poly(x_sq * spin_y)),
                (f"p_sq_emitter_y{suffix}", _clean_real_poly(p_sq * spin_y)),
                (f"n_x{suffix}", _clean_real_poly(_symmetrized_product(n_b, x_b))),
                (f"n_p{suffix}", _clean_real_poly(_symmetrized_product(n_b, p_b))),
                (f"n_x_imbalance{suffix}", _clean_real_poly(_symmetrized_product(n_b, x_b) * imbalance)),
                (f"n_p_imbalance{suffix}", _clean_real_poly(_symmetrized_product(n_b, p_b) * imbalance)),
                (f"n_x_flip{suffix}", _clean_real_poly(_symmetrized_product(n_b, x_b) * flip)),
                (f"n_p_flip{suffix}", _clean_real_poly(_symmetrized_product(n_b, p_b) * flip)),
            ]
        )
    pool.extend(
        build_spin_boson_blocks(
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
    )
    return [(label, poly) for label, poly in pool if len(poly.return_polynomial()) > 0]


def make_spin_boson_observables(
    *,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str = "blocked",
    num_sites: int = 1,
) -> dict[str, PauliPolynomial]:
    """Return emitter/boson observables used by trajectory reporting."""
    n_sites = _require_single_emitter(int(num_sites))
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    nq_total = 2 + int(n_sites * qpb)
    g_mode, e_mode = _emitter_mode_indices(str(ordering))
    observables = {
        "n_g": _clean_real_poly(jw_number_operator("JW", int(nq_total), int(g_mode))),
        "n_e": _clean_real_poly(jw_number_operator("JW", int(nq_total), int(e_mode))),
        "imbalance": _clean_real_poly(_emitter_imbalance_operator(ordering=str(ordering), nq_total=int(nq_total))),
        "spin_x": _clean_real_poly(_emitter_flip_operator(ordering=str(ordering), nq_total=int(nq_total))),
    }
    for site in range(n_sites):
        key = "n_b" if n_sites == 1 else f"n_b_{site}"
        observables[key] = _clean_real_poly(
            _boson_number_operator(
                nq_total=int(nq_total),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                site=int(site),
                num_sites=int(n_sites),
            )
        )
    return observables


__all__ = [
    "build_spin_boson_blocks",
    "build_spin_boson_full_meta_terms",
    "build_spin_boson_hamiltonian",
    "build_spin_boson_hva_terms",
    "build_spin_boson_quadratures",
    "build_spin_boson_reference_statevector",
    "exact_ground_energy_spin_boson",
    "make_spin_boson_observables",
]
