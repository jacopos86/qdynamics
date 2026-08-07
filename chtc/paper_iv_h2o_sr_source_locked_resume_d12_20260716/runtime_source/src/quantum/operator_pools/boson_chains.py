"""Boson-only chain Hamiltonians and operator-pool helpers.

This module centralizes boson-only Hamiltonian families and their large,
problem-local operator lists for the static ADAPT pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.quantum.hubbard_latex_python_pairs import (
    boson_displacement_operator,
    boson_operator,
    boson_qubits_per_site,
    phonon_qubit_indices_for_site,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm

_BH_FORMULA = (
    "H_BH = -t Σ_<ij> (b_i^dag b_j + b_j^dag b_i) + omega0 Σ_i n_i + (u/2) Σ_i n_i(n_i-e) + dv Σ_i (-1)^i n_i"
)

_HKC_FORMULA = (
    "H_HKC = omega0 Σ_i n_i + (u/2) Σ_i n_i(n_i-e) - t Σ_<ij> X_i X_j + dv Σ_i X_i"
)

_BOSON_VACUUM_FORMULA = "|phi_ref> = |0_b>^{⊗L}"


def _identity_poly(nq: int) -> PauliPolynomial:
    return PauliPolynomial("JW", [PauliTerm(int(nq), ps="e" * int(nq), pc=1.0)])


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
            raise ValueError(f"Boson-chain operator has non-negligible imaginary coefficient: {coeff}")
        cleaned.add_term(PauliTerm(int(nq), ps=str(term.pw2strng()), pc=float(coeff.real)))
    cleaned._reduce()
    return cleaned


def _symmetrized_product(lhs: PauliPolynomial, rhs: PauliPolynomial) -> PauliPolynomial:
    return 0.5 * ((lhs * rhs) + (rhs * lhs))


def _neighbor_edges(num_sites: int, boundary: str) -> list[tuple[int, int]]:
    n_sites = int(num_sites)
    if n_sites <= 1:
        return []
    periodic = str(boundary).strip().lower() == "periodic"
    edges = [(site, site + 1) for site in range(n_sites - 1)]
    if periodic and n_sites > 2:
        edges.append((n_sites - 1, 0))
    return edges


def _boson_code_bits(*, n_ph_max: int, boson_encoding: str) -> list[int]:
    d = int(n_ph_max) + 1
    encoding_key = str(boson_encoding).strip().lower()
    if encoding_key == "binary":
        return [int(level) for level in range(d)]
    if encoding_key == "unary":
        return [int(1 << level) for level in range(d)]
    raise ValueError(f"Unknown boson encoding '{boson_encoding}'")


def _boson_register_size(num_sites: int, n_ph_max: int, boson_encoding: str) -> tuple[int, int]:
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    return int(qpb), int(num_sites) * int(qpb)


def _site_qubits(site: int, *, num_sites: int, n_ph_max: int, boson_encoding: str) -> list[int]:
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    return phonon_qubit_indices_for_site(
        int(site),
        n_sites=int(num_sites),
        qpb=int(qpb),
        fermion_qubits=0,
    )


def _boson_annihilation(site: int, *, num_sites: int, n_ph_max: int, boson_encoding: str) -> PauliPolynomial:
    _qpb, nq_total = _boson_register_size(int(num_sites), int(n_ph_max), str(boson_encoding))
    return boson_operator(
        "JW",
        int(nq_total),
        _site_qubits(int(site), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)),
        which="b",
        n_ph_max=int(n_ph_max),
        encoding=str(boson_encoding),
    )


def _boson_creation(site: int, *, num_sites: int, n_ph_max: int, boson_encoding: str) -> PauliPolynomial:
    _qpb, nq_total = _boson_register_size(int(num_sites), int(n_ph_max), str(boson_encoding))
    return boson_operator(
        "JW",
        int(nq_total),
        _site_qubits(int(site), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)),
        which="bdag",
        n_ph_max=int(n_ph_max),
        encoding=str(boson_encoding),
    )


def _boson_number(site: int, *, num_sites: int, n_ph_max: int, boson_encoding: str) -> PauliPolynomial:
    return _boson_creation(int(site), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)) * _boson_annihilation(
        int(site), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)
    )


def _boson_displacement(site: int, *, num_sites: int, n_ph_max: int, boson_encoding: str) -> PauliPolynomial:
    _qpb, nq_total = _boson_register_size(int(num_sites), int(n_ph_max), str(boson_encoding))
    return boson_displacement_operator(
        "JW",
        int(nq_total),
        _site_qubits(int(site), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)),
        n_ph_max=int(n_ph_max),
        encoding=str(boson_encoding),
    )


def _boson_momentum(site: int, *, num_sites: int, n_ph_max: int, boson_encoding: str) -> PauliPolynomial:
    b = _boson_annihilation(int(site), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    bdag = _boson_creation(int(site), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    return (1j * bdag) + ((-1j) * b)


def _boson_hopping(i: int, j: int, *, num_sites: int, n_ph_max: int, boson_encoding: str) -> PauliPolynomial:
    return (
        _boson_creation(int(i), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
        * _boson_annihilation(int(j), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
        + _boson_creation(int(j), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
        * _boson_annihilation(int(i), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    )


def _boson_current(i: int, j: int, *, num_sites: int, n_ph_max: int, boson_encoding: str) -> PauliPolynomial:
    return (1j * (
        _boson_creation(int(i), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
        * _boson_annihilation(int(j), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
        + ((-1.0) * (
            _boson_creation(int(j), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
            * _boson_annihilation(int(i), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
        ))
    ))


def _boson_pair_hopping(i: int, j: int, *, num_sites: int, n_ph_max: int, boson_encoding: str) -> PauliPolynomial:
    b_i = _boson_annihilation(int(i), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    b_j = _boson_annihilation(int(j), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    bdag_i = _boson_creation(int(i), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    bdag_j = _boson_creation(int(j), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    return ((bdag_i * bdag_i) * (b_j * b_j)) + ((bdag_j * bdag_j) * (b_i * b_i))


def _boson_pair_current(i: int, j: int, *, num_sites: int, n_ph_max: int, boson_encoding: str) -> PauliPolynomial:
    b_i = _boson_annihilation(int(i), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    b_j = _boson_annihilation(int(j), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    bdag_i = _boson_creation(int(i), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    bdag_j = _boson_creation(int(j), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    pair_ij = (bdag_i * bdag_i) * (b_j * b_j)
    pair_ji = (bdag_j * bdag_j) * (b_i * b_i)
    return (1j * (pair_ij + ((-1.0) * pair_ji)))


def _boson_squeeze_x(site: int, *, num_sites: int, n_ph_max: int, boson_encoding: str) -> PauliPolynomial:
    b = _boson_annihilation(int(site), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    bdag = _boson_creation(int(site), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    return (bdag * bdag) + (b * b)


def _boson_squeeze_p(site: int, *, num_sites: int, n_ph_max: int, boson_encoding: str) -> PauliPolynomial:
    b = _boson_annihilation(int(site), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    bdag = _boson_creation(int(site), num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    return (1j * ((bdag * bdag) + ((-1.0) * (b * b))))


_BH_HAMILTONIAN_SYMBOLIC = _BH_FORMULA

def build_bose_hubbard_hamiltonian(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    n_ph_max: int,
    boson_encoding: str,
    boundary: str,
    include_zero_point: bool = True,
) -> PauliPolynomial:
    """Math: H_BH = -t Σ_<ij> (b_i^dag b_j + b_j^dag b_i) + omega0 Σ_i n_i + (u/2) Σ_i n_i(n_i-e) + dv Σ_i (-1)^i n_i."""
    qpb, nq_total = _boson_register_size(int(num_sites), int(n_ph_max), str(boson_encoding))
    del qpb
    ident = _identity_poly(int(nq_total))
    h_poly = PauliPolynomial("JW")
    for i, j in _neighbor_edges(int(num_sites), str(boundary)):
        h_poly = h_poly + ((-float(t)) * _boson_hopping(i, j, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
    for site in range(int(num_sites)):
        n_i = _boson_number(site, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
        h_poly = h_poly + (float(omega0) * n_i)
        if bool(include_zero_point):
            h_poly = h_poly + (0.5 * float(omega0) * ident)
        if abs(float(u)) > 0.0:
            h_poly = h_poly + (0.5 * float(u) * (n_i * (n_i + ((-1.0) * ident))))
        if abs(float(dv)) > 0.0:
            h_poly = h_poly + (((-1.0) ** int(site)) * float(dv) * n_i)
    h_poly._reduce()
    return h_poly


_HKC_HAMILTONIAN_SYMBOLIC = _HKC_FORMULA

def build_harmonic_kerr_chain_hamiltonian(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    n_ph_max: int,
    boson_encoding: str,
    boundary: str,
    include_zero_point: bool = True,
) -> PauliPolynomial:
    """Math: H_HKC = omega0 Σ_i n_i + (u/2) Σ_i n_i(n_i-e) - t Σ_<ij> X_i X_j + dv Σ_i X_i."""
    qpb, nq_total = _boson_register_size(int(num_sites), int(n_ph_max), str(boson_encoding))
    del qpb
    ident = _identity_poly(int(nq_total))
    h_poly = PauliPolynomial("JW")
    for site in range(int(num_sites)):
        n_i = _boson_number(site, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
        x_i = _boson_displacement(site, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
        h_poly = h_poly + (float(omega0) * n_i)
        if bool(include_zero_point):
            h_poly = h_poly + (0.5 * float(omega0) * ident)
        if abs(float(u)) > 0.0:
            h_poly = h_poly + (0.5 * float(u) * (n_i * (n_i + ((-1.0) * ident))))
        if abs(float(dv)) > 0.0:
            h_poly = h_poly + (float(dv) * x_i)
    for i, j in _neighbor_edges(int(num_sites), str(boundary)):
        x_i = _boson_displacement(i, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
        x_j = _boson_displacement(j, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
        h_poly = h_poly + ((-float(t)) * (x_i * x_j))
    h_poly._reduce()
    return h_poly


_BOSON_VACUUM_SYMBOLIC = _BOSON_VACUUM_FORMULA

def build_boson_chain_vacuum_statevector(
    *,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
) -> np.ndarray:
    """Math: |phi_ref> = |0_b>^{⊗L}."""
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    site_dim = 1 << int(qpb)
    vac = np.zeros(site_dim, dtype=complex)
    vac[int(_boson_code_bits(n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))[0])] = 1.0 + 0.0j
    psi = np.array([1.0 + 0.0j], dtype=complex)
    for _ in range(int(num_sites)):
        psi = np.kron(vac, psi)
    norm = float(np.linalg.norm(psi))
    if norm <= 0.0:
        raise ValueError("boson-chain vacuum has zero norm")
    return np.asarray(psi / norm, dtype=complex)


def build_boson_chain_fock_statevector(
    *,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    occupations: list[int] | tuple[int, ...],
) -> np.ndarray:
    """Return a product Fock state for a truncated boson chain."""

    n_sites = int(num_sites)
    occs = tuple(int(x) for x in occupations)
    if len(occs) != n_sites:
        raise ValueError(f"Expected {n_sites} boson occupations, got {len(occs)}.")
    code_bits = _boson_code_bits(n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    basis_index = 0
    for site, occ in enumerate(occs):
        if occ < 0 or occ > int(n_ph_max):
            raise ValueError(f"Boson occupation {occ} is outside [0, {int(n_ph_max)}].")
        basis_index |= int(code_bits[int(occ)]) << int(site * qpb)
    psi = np.zeros(1 << int(n_sites * qpb), dtype=complex)
    psi[int(basis_index)] = 1.0 + 0.0j
    return psi


def boson_chain_legal_basis_indices(
    *,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
) -> np.ndarray:
    """Return computational basis indices that encode valid local boson levels.

    Binary encodings generally leave unused local codewords when ``n_ph_max+1``
    is not a power of two. ADAPT statevectors may occupy those states unless a
    route explicitly prevents or penalizes leakage, so this helper defines the
    physical subspace used by diagnostics and exact-subspace comparisons.
    """

    n_sites = int(num_sites)
    if n_sites < 1:
        raise ValueError("num_sites must be positive")
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    code_bits = tuple(_boson_code_bits(n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
    indices: list[int] = []
    for levels in np.ndindex(*([len(code_bits)] * n_sites)):
        basis_index = 0
        for site, level in enumerate(levels):
            basis_index |= int(code_bits[int(level)]) << int(site * qpb)
        indices.append(int(basis_index))
    return np.asarray(sorted(indices), dtype=int)


def boson_chain_legal_probability(
    statevector: np.ndarray,
    *,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
) -> float:
    """Return total probability mass inside the legal truncated boson subspace."""

    psi = np.asarray(statevector, dtype=complex).reshape(-1)
    expected_dim = 1 << int(int(num_sites) * int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding))))
    if psi.shape[0] != expected_dim:
        raise ValueError(f"Expected statevector dimension {expected_dim}, got {psi.shape[0]}.")
    indices = boson_chain_legal_basis_indices(num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))
    prob = float(np.sum(np.abs(psi[indices]) ** 2).real)
    return float(min(max(prob, 0.0), 1.0))


def boson_chain_illegal_probability(
    statevector: np.ndarray,
    *,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
) -> float:
    """Return probability mass outside the legal truncated boson subspace."""

    return float(1.0 - boson_chain_legal_probability(
        statevector,
        num_sites=int(num_sites),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
    ))


def exact_ground_energy_boson_chain(
    h_poly: Any,
    *,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    tol: float = 1e-12,
) -> float:
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
    mats = {
        "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
        "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    }
    nq = len(order[0])
    dim = 1 << int(nq)
    hmat = np.zeros((dim, dim), dtype=complex)
    for label in order:
        op = mats[label[0]]
        for ch in label[1:]:
            op = np.kron(op, mats[ch])
        hmat += coeff_map[label] * op
    boson_code_bits = _boson_code_bits(
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
    )
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    basis_indices: list[int] = []
    for levels in np.ndindex(*([len(boson_code_bits)] * int(num_sites))):
        idx = 0
        for site, code_idx in enumerate(levels):
            idx |= int(boson_code_bits[int(code_idx)]) << (int(site) * int(qpb))
        basis_indices.append(int(idx))
    sub = hmat[np.ix_(basis_indices, basis_indices)]
    evals = np.linalg.eigvalsh(sub)
    return float(np.min(np.real(evals)))


def build_bose_hubbard_blocks(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    n_ph_max: int,
    boson_encoding: str,
    boundary: str,
    include_zero_point: bool = True,
) -> list[tuple[str, PauliPolynomial]]:
    ident = _identity_poly(_boson_register_size(int(num_sites), int(n_ph_max), str(boson_encoding))[1])
    blocks: list[tuple[str, PauliPolynomial]] = []
    for i, j in _neighbor_edges(int(num_sites), str(boundary)):
        blocks.append((f"hop_{i}_{j}", _clean_real_poly(((-float(t)) * _boson_hopping(i, j, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding))))))
    for site in range(int(num_sites)):
        n_i = _clean_real_poly(_boson_number(site, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        blocks.append((f"number_{site}", _clean_real_poly(float(omega0) * n_i)))
        if abs(float(u)) > 0.0:
            blocks.append((f"interaction_{site}", _clean_real_poly(0.5 * float(u) * (n_i * (n_i + ((-1.0) * ident))))))
        if abs(float(dv)) > 0.0:
            blocks.append((f"staggered_number_{site}", _clean_real_poly((((-1.0) ** int(site)) * float(dv)) * n_i)))
    return [(label, poly) for label, poly in blocks if len(poly.return_polynomial()) > 0]


def build_harmonic_kerr_chain_blocks(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    n_ph_max: int,
    boson_encoding: str,
    boundary: str,
    include_zero_point: bool = True,
) -> list[tuple[str, PauliPolynomial]]:
    ident = _identity_poly(_boson_register_size(int(num_sites), int(n_ph_max), str(boson_encoding))[1])
    blocks: list[tuple[str, PauliPolynomial]] = []
    for site in range(int(num_sites)):
        n_i = _clean_real_poly(_boson_number(site, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        x_i = _clean_real_poly(_boson_displacement(site, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        blocks.append((f"number_{site}", _clean_real_poly(float(omega0) * n_i)))
        if abs(float(u)) > 0.0:
            blocks.append((f"kerr_{site}", _clean_real_poly(0.5 * float(u) * (n_i * (n_i + ((-1.0) * ident))))))
        if abs(float(dv)) > 0.0:
            blocks.append((f"drive_x_{site}", _clean_real_poly(float(dv) * x_i)))
    for i, j in _neighbor_edges(int(num_sites), str(boundary)):
        x_i = _clean_real_poly(_boson_displacement(i, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        x_j = _clean_real_poly(_boson_displacement(j, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        blocks.append((f"xx_{i}_{j}", _clean_real_poly(((-float(t)) * (x_i * x_j)))))
    return [(label, poly) for label, poly in blocks if len(poly.return_polynomial()) > 0]


def build_bose_hubbard_hva_terms(**kwargs: Any) -> list[tuple[str, PauliPolynomial]]:
    return list(build_bose_hubbard_blocks(**kwargs))


def build_harmonic_kerr_chain_hva_terms(**kwargs: Any) -> list[tuple[str, PauliPolynomial]]:
    return list(build_harmonic_kerr_chain_blocks(**kwargs))


def build_bose_hubbard_quadratures(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    n_ph_max: int,
    boson_encoding: str,
    boundary: str,
    include_zero_point: bool = True,
) -> list[tuple[str, PauliPolynomial]]:
    del t, u, dv, omega0, include_zero_point
    pool: list[tuple[str, PauliPolynomial]] = []
    for site in range(int(num_sites)):
        x_i = _clean_real_poly(_boson_displacement(site, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        p_i = _clean_real_poly(_boson_momentum(site, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        n_i = _clean_real_poly(_boson_number(site, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        pool.extend([
            (f"x_{site}", x_i),
            (f"p_{site}", p_i),
            (f"n_{site}", n_i),
            (f"x_sq_{site}", _clean_real_poly(x_i * x_i)),
            (f"p_sq_{site}", _clean_real_poly(p_i * p_i)),
            (f"n_sq_{site}", _clean_real_poly(n_i * n_i)),
            (f"squeeze_x_{site}", _clean_real_poly(_boson_squeeze_x(site, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))),
            (f"squeeze_p_{site}", _clean_real_poly(_boson_squeeze_p(site, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))),
        ])
    for i, j in _neighbor_edges(int(num_sites), str(boundary)):
        x_i = _clean_real_poly(_boson_displacement(i, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        x_j = _clean_real_poly(_boson_displacement(j, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        p_i = _clean_real_poly(_boson_momentum(i, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        p_j = _clean_real_poly(_boson_momentum(j, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        pool.extend([
            (f"hop_{i}_{j}", _clean_real_poly(_boson_hopping(i, j, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))),
            (f"current_{i}_{j}", _clean_real_poly(_boson_current(i, j, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))),
            (f"xx_{i}_{j}", _clean_real_poly(x_i * x_j)),
            (f"pp_{i}_{j}", _clean_real_poly(p_i * p_j)),
        ])
    return [(label, poly) for label, poly in pool if len(poly.return_polynomial()) > 0]


def build_harmonic_kerr_chain_quadratures(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    n_ph_max: int,
    boson_encoding: str,
    boundary: str,
    include_zero_point: bool = True,
) -> list[tuple[str, PauliPolynomial]]:
    del t, u, dv, omega0, include_zero_point
    return build_bose_hubbard_quadratures(
        num_sites=int(num_sites),
        t=0.0,
        u=0.0,
        dv=0.0,
        omega0=0.0,
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        boundary=str(boundary),
        include_zero_point=False,
    )


def build_bose_hubbard_full_meta_terms(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    n_ph_max: int,
    boson_encoding: str,
    boundary: str,
    include_zero_point: bool = True,
) -> list[tuple[str, PauliPolynomial]]:
    pool = list(build_bose_hubbard_quadratures(
        num_sites=int(num_sites), t=float(t), u=float(u), dv=float(dv), omega0=float(omega0),
        n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding), boundary=str(boundary), include_zero_point=bool(include_zero_point)
    ))
    site_number_ops: dict[int, PauliPolynomial] = {}
    for site in range(int(num_sites)):
        x_i = _clean_real_poly(_boson_displacement(site, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        p_i = _clean_real_poly(_boson_momentum(site, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        n_i = _clean_real_poly(_boson_number(site, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        site_number_ops[int(site)] = n_i
        pool.extend([
            (f"n_x_{site}", _clean_real_poly(_symmetrized_product(n_i, x_i))),
            (f"n_p_{site}", _clean_real_poly(_symmetrized_product(n_i, p_i))),
            (f"n_x_sq_{site}", _clean_real_poly(_symmetrized_product(n_i, x_i * x_i))),
            (f"n_p_sq_{site}", _clean_real_poly(_symmetrized_product(n_i, p_i * p_i))),
        ])
    for i, j in _neighbor_edges(int(num_sites), str(boundary)):
        n_i = site_number_ops[int(i)]
        n_j = site_number_ops[int(j)]
        hop_ij = _clean_real_poly(_boson_hopping(i, j, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        current_ij = _clean_real_poly(_boson_current(i, j, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        pair_hop_ij = _clean_real_poly(_boson_pair_hopping(i, j, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        pair_current_ij = _clean_real_poly(_boson_pair_current(i, j, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        pool.extend([
            (f"nn_{i}_{j}", _clean_real_poly(n_i * n_j)),
            (f"density_hop_{i}_{j}_left", _clean_real_poly(_symmetrized_product(n_i, hop_ij))),
            (f"density_hop_{i}_{j}_right", _clean_real_poly(_symmetrized_product(n_j, hop_ij))),
            (f"density_current_{i}_{j}_left", _clean_real_poly(_symmetrized_product(n_i, current_ij))),
            (f"density_current_{i}_{j}_right", _clean_real_poly(_symmetrized_product(n_j, current_ij))),
            (f"pair_hop_{i}_{j}", pair_hop_ij),
            (f"pair_current_{i}_{j}", pair_current_ij),
        ])
    pool.extend(build_bose_hubbard_blocks(
        num_sites=int(num_sites), t=float(t), u=float(u), dv=float(dv), omega0=float(omega0),
        n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding), boundary=str(boundary), include_zero_point=bool(include_zero_point)
    ))
    return [(label, poly) for label, poly in pool if len(poly.return_polynomial()) > 0]


def build_harmonic_kerr_chain_full_meta_terms(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    n_ph_max: int,
    boson_encoding: str,
    boundary: str,
    include_zero_point: bool = True,
) -> list[tuple[str, PauliPolynomial]]:
    pool = list(build_harmonic_kerr_chain_quadratures(
        num_sites=int(num_sites), t=float(t), u=float(u), dv=float(dv), omega0=float(omega0),
        n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding), boundary=str(boundary), include_zero_point=bool(include_zero_point)
    ))
    for site in range(int(num_sites)):
        x_i = _clean_real_poly(_boson_displacement(site, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        p_i = _clean_real_poly(_boson_momentum(site, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        n_i = _clean_real_poly(_boson_number(site, num_sites=int(num_sites), n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding)))
        pool.extend([
            (f"n_x_{site}", _clean_real_poly(_symmetrized_product(n_i, x_i))),
            (f"n_p_{site}", _clean_real_poly(_symmetrized_product(n_i, p_i))),
            (f"x_p_sym_{site}", _clean_real_poly((x_i * p_i) + (p_i * x_i))),
        ])
    pool.extend(build_harmonic_kerr_chain_blocks(
        num_sites=int(num_sites), t=float(t), u=float(u), dv=float(dv), omega0=float(omega0),
        n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding), boundary=str(boundary), include_zero_point=bool(include_zero_point)
    ))
    return [(label, poly) for label, poly in pool if len(poly.return_polynomial()) > 0]


def make_harmonic_kerr_chain_drive_operator(
    *,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    spatial_weights: tuple[float, ...] | list[float],
) -> PauliPolynomial:
    """Return the harmonic-Kerr displacement drive Σ_i s_i X_i."""

    n_sites = int(num_sites)
    weights = tuple(float(x) for x in spatial_weights)
    if len(weights) != n_sites:
        raise ValueError(f"Expected {n_sites} harmonic-Kerr drive weights, got {len(weights)}.")
    drive = PauliPolynomial("JW")
    for site, weight in enumerate(weights):
        drive = drive + (
            float(weight)
            * _boson_displacement(
                site,
                num_sites=int(n_sites),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
            )
        )
    return _clean_real_poly(drive)


def make_boson_chain_observables(
    *,
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
) -> dict[str, PauliPolynomial]:
    observables: dict[str, PauliPolynomial] = {}
    total = PauliPolynomial("JW")
    for site in range(int(num_sites)):
        n_site = _clean_real_poly(
            _boson_number(
                site,
                num_sites=int(num_sites),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
            )
        )
        observables[f"n_site_{site}"] = n_site
        total = total + n_site
    total._reduce()
    observables["n_site0"] = observables["n_site_0"]
    observables["n_total"] = _clean_real_poly(total)
    return observables


__all__ = [
    "build_bose_hubbard_blocks",
    "build_bose_hubbard_full_meta_terms",
    "build_bose_hubbard_hamiltonian",
    "build_bose_hubbard_hva_terms",
    "build_bose_hubbard_quadratures",
    "boson_chain_illegal_probability",
    "boson_chain_legal_basis_indices",
    "boson_chain_legal_probability",
    "build_boson_chain_fock_statevector",
    "build_boson_chain_vacuum_statevector",
    "build_harmonic_kerr_chain_blocks",
    "build_harmonic_kerr_chain_full_meta_terms",
    "build_harmonic_kerr_chain_hamiltonian",
    "build_harmonic_kerr_chain_hva_terms",
    "build_harmonic_kerr_chain_quadratures",
    "exact_ground_energy_boson_chain",
    "make_boson_chain_observables",
    "make_harmonic_kerr_chain_drive_operator",
]
