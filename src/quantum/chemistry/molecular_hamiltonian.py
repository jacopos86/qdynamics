from __future__ import annotations

from typing import Any

from src.quantum.chemistry.psi4_adapter import RestrictedClosedShellMolecularProblem
from src.quantum.pauli_polynomial_class import (
    PauliPolynomial,
    fermion_minus_operator,
    fermion_plus_operator,
)
from src.quantum.qubitization_module import PauliTerm


def _blocked_spin(spin_orbital: int, n_spatial_orbitals: int) -> int:
    return 0 if int(spin_orbital) < int(n_spatial_orbitals) else 1


def _blocked_spatial_index(spin_orbital: int, n_spatial_orbitals: int) -> int:
    idx = int(spin_orbital)
    n_spatial = int(n_spatial_orbitals)
    return idx if idx < n_spatial else idx - n_spatial


def _real_scalar(value: Any, *, tol: float = 1e-12, label: str = "coefficient") -> float:
    coeff = complex(value)
    if abs(coeff.imag) > float(tol):
        raise ValueError(f"{label} has non-negligible imaginary part: {coeff}")
    return float(coeff.real)


def _scaled_terms(poly: PauliPolynomial, coeff: complex) -> list[PauliTerm]:
    return [
        PauliTerm(term.nqubit(), ps=term.pw2strng(), pc=complex(coeff) * term.p_coeff)
        for term in poly.return_polynomial()
    ]


def _clean_reduced_polynomial(
    polynomial: PauliPolynomial,
    *,
    repr_mode: str,
    tol: float,
) -> PauliPolynomial:
    """Apply the builder tolerance after JW cancellation and aggregation."""

    terms: list[PauliTerm] = []
    for term in polynomial.return_polynomial():
        coefficient = complex(term.p_coeff)
        if abs(coefficient) <= float(tol):
            continue
        if abs(coefficient.imag) <= float(tol):
            coefficient = complex(float(coefficient.real), 0.0)
        terms.append(
            PauliTerm(
                term.nqubit(),
                ps=term.pw2strng(),
                pc=coefficient,
            )
        )
    return PauliPolynomial(str(repr_mode), terms)


_MATH_ONE_BODY_JW = "H_1 = \\sum_{pq} h_{pq} a_p^\\dagger a_q"
def build_one_body_jw_polynomial(
    problem: RestrictedClosedShellMolecularProblem,
    *,
    repr_mode: str = "JW",
    ordering: str = "blocked",
    tol: float = 1e-12,
) -> PauliPolynomial:
    if str(ordering).strip().lower() != "blocked":
        raise ValueError("Chemistry prototype currently supports ordering='blocked' only.")
    n_spatial = int(problem.n_spatial_orbitals)
    nq = 2 * n_spatial
    terms: list[PauliTerm] = []
    creators = [fermion_plus_operator(repr_mode, nq, p) for p in range(nq)]
    annihilators = [fermion_minus_operator(repr_mode, nq, q) for q in range(nq)]

    for p in range(nq):
        spin_p = _blocked_spin(p, n_spatial)
        p_spatial = _blocked_spatial_index(p, n_spatial)
        for q in range(nq):
            if spin_p != _blocked_spin(q, n_spatial):
                continue
            q_spatial = _blocked_spatial_index(q, n_spatial)
            coeff = _real_scalar(
                problem.one_body_integrals_mo[p_spatial, q_spatial],
                tol=float(tol),
                label="one-body integral",
            )
            if abs(coeff) <= float(tol):
                continue
            terms.extend(_scaled_terms(creators[p] * annihilators[q], coeff))
    return _clean_reduced_polynomial(
        PauliPolynomial(repr_mode, terms),
        repr_mode=str(repr_mode),
        tol=float(tol),
    )


_MATH_TWO_BODY_JW = "H_2 = \\tfrac{1}{2} \\sum_{pqrs} (pr|qs) a_p^\\dagger a_q^\\dagger a_s a_r"
def build_two_body_jw_polynomial(
    problem: RestrictedClosedShellMolecularProblem,
    *,
    repr_mode: str = "JW",
    ordering: str = "blocked",
    tol: float = 1e-12,
) -> PauliPolynomial:
    if str(ordering).strip().lower() != "blocked":
        raise ValueError("Chemistry prototype currently supports ordering='blocked' only.")
    n_spatial = int(problem.n_spatial_orbitals)
    nq = 2 * n_spatial
    terms: list[PauliTerm] = []
    creators = [fermion_plus_operator(repr_mode, nq, p) for p in range(nq)]
    annihilators = [fermion_minus_operator(repr_mode, nq, p) for p in range(nq)]

    for p in range(nq):
        p_spin = _blocked_spin(p, n_spatial)
        p_spatial = _blocked_spatial_index(p, n_spatial)
        for q in range(nq):
            q_spin = _blocked_spin(q, n_spatial)
            q_spatial = _blocked_spatial_index(q, n_spatial)
            for r in range(nq):
                if p_spin != _blocked_spin(r, n_spatial):
                    continue
                r_spatial = _blocked_spatial_index(r, n_spatial)
                for s in range(nq):
                    if q_spin != _blocked_spin(s, n_spatial):
                        continue
                    s_spatial = _blocked_spatial_index(s, n_spatial)
                    # Psi4 MO ERIs are in chemists' notation (pq|rs). The second-quantized
                    # Hamiltonian used here is 1/2 * sum_{pqrs} (pr|qs) a^†_p a^†_q a_s a_r,
                    # so the lookup is eri[p, r, q, s].
                    coeff = 0.5 * _real_scalar(
                        problem.two_body_integrals_mo[p_spatial, r_spatial, q_spatial, s_spatial],
                        tol=float(tol),
                        label="two-body integral",
                    )
                    if abs(coeff) <= float(tol):
                        continue
                    term = (((creators[p] * creators[q]) * annihilators[s]) * annihilators[r])
                    terms.extend(_scaled_terms(term, coeff))
    return _clean_reduced_polynomial(
        PauliPolynomial(repr_mode, terms),
        repr_mode=str(repr_mode),
        tol=float(tol),
    )


_MATH_MOLECULAR_H_JW = "H = E_{nuc} I + H_1 + H_2"
def build_restricted_closed_shell_molecular_hamiltonian(
    problem: RestrictedClosedShellMolecularProblem,
    *,
    repr_mode: str = "JW",
    ordering: str = "blocked",
    tol: float = 1e-12,
) -> PauliPolynomial:
    n_spatial = int(problem.n_spatial_orbitals)
    if n_spatial <= 0:
        raise ValueError("n_spatial_orbitals must be positive.")
    nq = 2 * n_spatial
    terms: list[PauliTerm] = []
    enuc = _real_scalar(problem.nuclear_repulsion_energy, tol=float(tol), label="nuclear repulsion energy")
    if abs(enuc) > float(tol):
        terms.append(PauliTerm(nq, ps="e" * nq, pc=float(enuc)))
    terms.extend(
        _scaled_terms(
            build_one_body_jw_polynomial(
                problem,
                repr_mode=str(repr_mode),
                ordering=str(ordering),
                tol=float(tol),
            ),
            1.0,
        )
    )
    terms.extend(
        _scaled_terms(
            build_two_body_jw_polynomial(
                problem,
                repr_mode=str(repr_mode),
                ordering=str(ordering),
                tol=float(tol),
            ),
            1.0,
        )
    )
    return _clean_reduced_polynomial(
        PauliPolynomial(repr_mode, terms),
        repr_mode=str(repr_mode),
        tol=float(tol),
    )
