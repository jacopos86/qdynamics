from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on path (same pattern as other integration tests).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.compiled_polynomial import (
    adapt_commutator_grad_from_hpsi,
    apply_compiled_polynomial,
    compile_polynomial_action,
    energy_via_one_apply,
)
from src.quantum.pauli_actions import compile_pauli_action_exyz
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import (
    apply_pauli_string,
    expval_pauli_polynomial,
    hamiltonian_matrix,
)


def _random_state(nq: int, rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=1 << int(nq)) + 1j * rng.normal(size=1 << int(nq))
    vec = np.asarray(vec, dtype=complex)
    return vec / np.linalg.norm(vec)


def _random_label_exyz(nq: int, rng: np.random.Generator) -> str:
    symbols = np.array(list("exyz"))
    return "".join(symbols[rng.integers(0, len(symbols), size=int(nq))])


def _random_real_pauli_polynomial(
    nq: int,
    rng: np.random.Generator,
    *,
    num_terms: int,
) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    poly.add_term(PauliTerm(int(nq), ps="e" * int(nq), pc=float(rng.normal())))
    for _ in range(int(num_terms)):
        label = _random_label_exyz(int(nq), rng)
        coeff = float(rng.normal())
        poly.add_term(PauliTerm(int(nq), ps=label, pc=coeff))
    return poly


def _term_labels(compiled) -> list[str]:
    out: list[str] = []
    for term in compiled.terms:
        if term.action is None:
            out.append("e" * int(compiled.nq))
        else:
            out.append(str(term.action.label_exyz))
    return out


def test_compile_polynomial_action_accumulates_duplicates_and_uses_cache():
    poly = PauliPolynomial("JW")
    poly.add_term(PauliTerm(4, ps="xzee", pc=0.5))
    poly.add_term(PauliTerm(4, ps="eeee", pc=1.0))
    poly.add_term(PauliTerm(4, ps="yyzz", pc=0.10))
    poly.add_term(PauliTerm(4, ps="xzee", pc=-0.25))
    poly.add_term(PauliTerm(4, ps="zzzz", pc=5e-16))  # pruned by tol
    poly.add_term(PauliTerm(4, ps="yyzz", pc=0.025))

    cache = {"xzee": compile_pauli_action_exyz("xzee", 4)}
    compiled = compile_polynomial_action(poly, tol=1e-15, pauli_action_cache=cache)

    labels = _term_labels(compiled)
    assert labels == ["xzee", "eeee", "yyzz"]

    coeff_by_label = {label: term.coeff for label, term in zip(labels, compiled.terms)}
    assert coeff_by_label["xzee"] == pytest.approx(0.25)
    assert coeff_by_label["eeee"] == pytest.approx(1.0)
    assert coeff_by_label["yyzz"] == pytest.approx(0.125)

    id_idx = labels.index("eeee")
    assert compiled.terms[id_idx].action is None
    assert compiled.terms[0].action is cache["xzee"]
    assert "yyzz" in cache
    assert compiled.terms[2].action is cache["yyzz"]
    assert "eeee" not in cache


@pytest.mark.parametrize("nq", [4, 6, 8])
def test_energy_via_one_apply_matches_expval_pauli_polynomial(nq: int):
    rng = np.random.default_rng(100 + int(nq))
    psi = _random_state(nq, rng)
    h_poly = _random_real_pauli_polynomial(nq, rng, num_terms=20)

    compiled_h = compile_polynomial_action(h_poly, tol=1e-15)
    energy_compiled, hpsi = energy_via_one_apply(psi, compiled_h)
    energy_ref = float(expval_pauli_polynomial(psi, h_poly, tol=1e-15))

    assert hpsi.shape == psi.shape
    assert abs(energy_compiled - energy_ref) < 1e-12


@pytest.mark.parametrize("nq", [4, 6])
def test_apply_compiled_polynomial_matches_dense_action(nq: int):
    rng = np.random.default_rng(200 + int(nq))
    psi = _random_state(nq, rng)
    poly = _random_real_pauli_polynomial(nq, rng, num_terms=16)

    compiled = compile_polynomial_action(poly, tol=1e-15)
    hpsi_compiled = apply_compiled_polynomial(psi, compiled)
    h_dense = hamiltonian_matrix(poly, tol=1e-15)
    hpsi_dense = h_dense @ psi

    assert np.max(np.abs(hpsi_compiled - hpsi_dense)) < 1e-12


def test_adapt_commutator_grad_from_hpsi_sign_and_value():
    psi = np.array([1.0, 1.0j], dtype=complex) / math.sqrt(2.0)
    hpsi = apply_pauli_string(psi, "z")
    apsi = apply_pauli_string(psi, "x")

    grad = adapt_commutator_grad_from_hpsi(hpsi, apsi)
    grad_flipped = adapt_commutator_grad_from_hpsi(hpsi, -apsi)

    assert grad == pytest.approx(2.0, abs=1e-12)
    assert grad_flipped == pytest.approx(-2.0, abs=1e-12)

    direct = float(2.0 * np.imag(np.vdot(hpsi, apsi)))
    assert grad == pytest.approx(direct, abs=1e-15)
