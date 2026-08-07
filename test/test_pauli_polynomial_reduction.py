from __future__ import annotations

import pytest

from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _coefficients(poly: PauliPolynomial) -> dict[str, complex]:
    return {
        str(term.pw2strng()): complex(term.p_coeff)
        for term in poly.return_polynomial()
    }


def test_reduce_preserves_small_nonzero_aggregate_coefficients() -> None:
    poly = PauliPolynomial(
        "JW",
        [
            PauliTerm(1, ps="x", pc=1.0),
            PauliTerm(1, ps="x", pc=-1.0 + 5.0e-8),
        ],
    )

    assert _coefficients(poly)["x"] == pytest.approx(5.0e-8)


def test_reduce_removes_only_exactly_cancelled_words() -> None:
    poly = PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="xy", pc=0.25j),
            PauliTerm(2, ps="xy", pc=-0.25j),
            PauliTerm(2, ps="zz", pc=2.0e-12),
        ],
    )

    assert _coefficients(poly) == {"zz": pytest.approx(2.0e-12)}


def test_reduce_does_not_mutate_input_terms() -> None:
    first = PauliTerm(1, ps="z", pc=1.0)
    second = PauliTerm(1, ps="z", pc=2.0)

    poly = PauliPolynomial("JW", [first, second])

    assert complex(first.p_coeff) == pytest.approx(1.0)
    assert complex(second.p_coeff) == pytest.approx(2.0)
    assert _coefficients(poly)["z"] == pytest.approx(3.0)
