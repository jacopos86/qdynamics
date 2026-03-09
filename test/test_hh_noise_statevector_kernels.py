from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.statevector_kernels import (
    apply_h_poly,
    commutator_gradient,
    compile_ansatz_executor,
    compile_h_poly,
    energy_one_apply,
    prepare_state_for_ansatz,
)
from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import (
    HubbardHolsteinPhysicalTermwiseAnsatz,
    apply_exp_pauli_polynomial,
    apply_pauli_string,
)


def _random_state(nq: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    vec = rng.normal(size=1 << int(nq)) + 1j * rng.normal(size=1 << int(nq))
    vec = np.asarray(vec, dtype=complex)
    return vec / np.linalg.norm(vec)


def _legacy_apply_h_poly(psi: np.ndarray, poly: Any) -> np.ndarray:
    terms = poly.return_polynomial()
    if not terms:
        return np.zeros_like(np.asarray(psi, dtype=complex))
    nq = int(terms[0].nqubit())
    id_label = "e" * nq
    psi_vec = np.asarray(psi, dtype=complex).reshape(-1)
    out = np.zeros_like(psi_vec)
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= 1e-15:
            continue
        label = str(term.pw2strng())
        if label == id_label:
            out += coeff * psi_vec
        else:
            out += coeff * apply_pauli_string(psi_vec, label)
    return out


def _legacy_prepare_state(psi_ref: np.ndarray, selected_ops: list[Any], theta: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi_ref, dtype=complex).reshape(-1).copy()
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    for idx, op in enumerate(selected_ops):
        psi = apply_exp_pauli_polynomial(psi, op.polynomial, float(theta_vec[idx]))
    return psi


def _build_small_hubbard_poly() -> Any:
    return build_hubbard_hamiltonian(
        dims=2,
        t=1.0,
        U=2.0,
        v=0.2,
        repr_mode="JW",
        indexing="blocked",
        pbc=True,
    )


def _build_small_hh_poly() -> Any:
    return build_hubbard_holstein_hamiltonian(
        dims=2,
        J=1.0,
        U=2.0,
        omega0=1.0,
        g=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        v_t=None,
        v0=0.0,
        t_eval=None,
        repr_mode="JW",
        indexing="blocked",
        pbc=True,
        include_zero_point=True,
    )


def _nq(poly: Any) -> int:
    terms = poly.return_polynomial()
    if not terms:
        raise ValueError("Expected non-empty PauliPolynomial for parity test.")
    return int(terms[0].nqubit())


def _single_term_operator_from_poly(poly: Any) -> PauliPolynomial:
    terms = poly.return_polynomial()
    if not terms:
        raise ValueError("Expected non-empty PauliPolynomial for parity test.")
    nq = int(terms[0].nqubit())
    id_label = "e" * nq

    out = PauliPolynomial("JW")
    for term in terms:
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if label == id_label or abs(coeff) <= 1e-12:
            continue
        coeff_real = float(coeff.real) if abs(coeff.real) > 1e-12 else 1.0
        out.add_term(PauliTerm(nq, ps=label, pc=coeff_real))
        break

    if not out.return_polynomial():
        out.add_term(PauliTerm(nq, ps=id_label, pc=1.0))
    return out


@pytest.mark.parametrize(
    ("poly_builder", "seed"),
    [
        (_build_small_hubbard_poly, 123),
        (_build_small_hh_poly, 321),
    ],
)
def test_apply_and_energy_parity_against_legacy(
    poly_builder: Callable[[], Any],
    seed: int,
) -> None:
    h_poly = poly_builder()
    psi = _random_state(_nq(h_poly), seed=seed)

    hpsi_legacy = _legacy_apply_h_poly(psi, h_poly)
    hpsi_fallback = apply_h_poly(psi, h_poly, compiled=None)
    h_compiled = compile_h_poly(h_poly)
    hpsi_compiled = apply_h_poly(psi, h_poly, compiled=h_compiled)

    assert np.max(np.abs(hpsi_fallback - hpsi_legacy)) < 1e-12
    assert np.max(np.abs(hpsi_compiled - hpsi_legacy)) < 1e-12

    e_legacy = float(np.real(np.vdot(psi, hpsi_legacy)))
    e_fallback = energy_one_apply(psi, h_poly, compiled=None)
    e_compiled = energy_one_apply(psi, h_poly, compiled=h_compiled)
    assert abs(e_fallback - e_legacy) < 1e-12
    assert abs(e_compiled - e_legacy) < 1e-12


def test_prepare_state_compiled_ansatz_matches_legacy() -> None:
    ansatz = HubbardHolsteinPhysicalTermwiseAnsatz(
        dims=2,
        J=1.0,
        U=2.0,
        omega0=1.0,
        g=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        reps=1,
        repr_mode="JW",
        indexing="blocked",
        pbc=True,
    )
    selected_ops = list(ansatz.base_terms[: min(6, len(ansatz.base_terms))])
    assert selected_ops

    rng = np.random.default_rng(17)
    theta = rng.normal(scale=0.2, size=len(selected_ops))
    psi_ref = _random_state(int(ansatz.nq), seed=18)

    psi_legacy = _legacy_prepare_state(psi_ref, selected_ops, theta)
    psi_fallback = prepare_state_for_ansatz(
        psi_ref,
        selected_ops,
        theta,
        compiled_cache=None,
        normalize=False,
    )
    compiled_ansatz = compile_ansatz_executor(selected_ops)
    assert compiled_ansatz is not None
    psi_compiled = prepare_state_for_ansatz(
        psi_ref,
        selected_ops,
        theta,
        compiled_cache=compiled_ansatz,
        normalize=False,
    )

    assert np.linalg.norm(psi_fallback - psi_legacy) < 1e-12
    assert np.linalg.norm(psi_compiled - psi_legacy) < 1e-10


@pytest.mark.parametrize(
    ("poly_builder", "seed"),
    [
        (_build_small_hubbard_poly, 222),
        (_build_small_hh_poly, 444),
    ],
)
def test_commutator_gradient_parity(
    poly_builder: Callable[[], Any],
    seed: int,
) -> None:
    h_poly = poly_builder()
    op_poly = _single_term_operator_from_poly(h_poly)
    psi = _random_state(_nq(h_poly), seed=seed)

    hpsi_legacy = _legacy_apply_h_poly(psi, h_poly)
    apsi_legacy = _legacy_apply_h_poly(psi, op_poly)
    grad_legacy = float(2.0 * np.imag(np.vdot(hpsi_legacy, apsi_legacy)))

    h_compiled = compile_h_poly(h_poly)
    op_compiled = compile_h_poly(op_poly)
    op_term = type("OpTerm", (), {"polynomial": op_poly})()
    hpsi_compiled = apply_h_poly(psi, h_poly, compiled=h_compiled)
    grad_compiled = commutator_gradient(
        op_term,
        psi,
        h_poly,
        compiled={"h_compiled": h_compiled, "pool_compiled": op_compiled},
        h_psi_precomputed=hpsi_compiled,
    )

    assert abs(grad_compiled - grad_legacy) < 1e-12
