"""Compiled PauliPolynomial helpers (exyz convention).

These utilities compile and apply PauliPolynomial operators using the shared
compiled-Pauli backend in ``src.quantum.pauli_actions``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.quantum.pauli_actions import (
    CompiledPauliAction,
    apply_compiled_pauli,
    compile_pauli_action_exyz,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial


@dataclass(frozen=True)
class CompiledPolynomialTerm:
    coeff: complex
    action: CompiledPauliAction | None


@dataclass(frozen=True)
class CompiledPolynomialAction:
    nq: int
    terms: tuple[CompiledPolynomialTerm, ...]


_MATH_COMPILE_POLYNOMIAL_ACTION = (
    r"C(H)=\{(c_\ell,A_\ell)\}_\ell,\ "
    r"A_\ell=\mathrm{compile\_pauli\_action\_exyz}(\ell,n_q),\ "
    r"H=\sum_\ell c_\ell P_\ell"
)


def compile_polynomial_action(
    poly: PauliPolynomial,
    *,
    tol: float = 1e-15,
    pauli_action_cache: dict[str, CompiledPauliAction] | None = None,
) -> CompiledPolynomialAction:
    """Compile a PauliPolynomial into reusable term actions."""
    terms = poly.return_polynomial()
    if not terms:
        raise ValueError("Cannot compile empty PauliPolynomial: unable to infer qubit count.")

    nq = int(terms[0].nqubit())
    id_label = "e" * nq
    coeff_by_label: dict[str, complex] = {}

    for term in terms:
        term_nq = int(term.nqubit())
        if term_nq != nq:
            raise ValueError(f"Inconsistent term qubit count: expected {nq}, got {term_nq}.")
        label = str(term.pw2strng())
        if len(label) != nq:
            raise ValueError(f"Invalid Pauli label length for '{label}': expected {nq}.")
        coeff_by_label[label] = coeff_by_label.get(label, 0.0 + 0.0j) + complex(term.p_coeff)

    cache = pauli_action_cache if pauli_action_cache is not None else {}
    compiled_terms: list[CompiledPolynomialTerm] = []
    for label, coeff in coeff_by_label.items():
        coeff_c = complex(coeff)
        if abs(coeff_c) <= float(tol):
            continue
        if label == id_label:
            compiled_terms.append(CompiledPolynomialTerm(coeff=coeff_c, action=None))
            continue
        action = cache.get(label)
        if action is None:
            action = compile_pauli_action_exyz(label, nq)
            cache[label] = action
        compiled_terms.append(CompiledPolynomialTerm(coeff=coeff_c, action=action))

    return CompiledPolynomialAction(nq=nq, terms=tuple(compiled_terms))


_MATH_APPLY_COMPILED_POLYNOMIAL = r"H|\psi\rangle=\sum_j c_j P_j|\psi\rangle"


def apply_compiled_polynomial(psi: np.ndarray, compiled: CompiledPolynomialAction) -> np.ndarray:
    """Apply a compiled PauliPolynomial action to a statevector."""
    psi_vec = np.asarray(psi, dtype=complex).reshape(-1)
    expected_dim = 1 << int(compiled.nq)
    if psi_vec.size != expected_dim:
        raise ValueError(
            f"Statevector length mismatch: got {psi_vec.size}, expected {expected_dim} for nq={compiled.nq}."
        )

    out = np.zeros_like(psi_vec, dtype=complex)
    for term in compiled.terms:
        if term.action is None:
            out += term.coeff * psi_vec
        else:
            out += term.coeff * apply_compiled_pauli(psi_vec, term.action)
    return out


_MATH_ENERGY_VIA_ONE_APPLY = r"E=\operatorname{Re}\langle\psi|H|\psi\rangle,\ H|\psi\rangle\text{ computed once}"


def energy_via_one_apply(
    psi: np.ndarray, compiled_h: CompiledPolynomialAction,
) -> tuple[float, np.ndarray]:
    """Compute energy via one compiled Hamiltonian apply and return (E, Hpsi)."""
    psi_vec = np.asarray(psi, dtype=complex).reshape(-1)
    hpsi = apply_compiled_polynomial(psi_vec, compiled_h)
    energy = float(np.real(np.vdot(psi_vec, hpsi)))
    return energy, hpsi


_MATH_ADAPT_COMMUTATOR_GRAD = r"g=\frac{dE}{d\theta}\big|_{\theta=0}=2\,\operatorname{Im}\langle H\psi|A\psi\rangle"


def adapt_commutator_grad_from_hpsi(Hpsi: np.ndarray, Apsi: np.ndarray) -> float:
    """Return the signed ADAPT commutator gradient from precomputed vectors."""
    hpsi_vec = np.asarray(Hpsi, dtype=complex).reshape(-1)
    apsi_vec = np.asarray(Apsi, dtype=complex).reshape(-1)
    if hpsi_vec.size != apsi_vec.size:
        raise ValueError(f"Vector size mismatch: {hpsi_vec.size} != {apsi_vec.size}.")
    return float(2.0 * np.imag(np.vdot(hpsi_vec, apsi_vec)))


__all__ = [
    "CompiledPolynomialTerm",
    "CompiledPolynomialAction",
    "compile_polynomial_action",
    "apply_compiled_polynomial",
    "energy_via_one_apply",
    "adapt_commutator_grad_from_hpsi",
]
