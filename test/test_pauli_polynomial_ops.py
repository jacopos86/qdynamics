#!/usr/bin/env python3
"""Focused regression tests for PauliPolynomial add/subtract semantics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, REPO_ROOT_STR)

from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _coeff_map(pp: PauliPolynomial) -> dict[str, complex]:
    return {pt.pw2strng(): complex(pt.p_coeff) for pt in pp.return_polynomial()}


class TestPauliPolynomialOps(unittest.TestCase):
    def test_add_does_not_mutate_inputs(self) -> None:
        p1 = PauliPolynomial("JW", [PauliTerm(2, ps="xe", pc=1.0)])
        p2 = PauliPolynomial("JW", [PauliTerm(2, ps="xe", pc=2.0)])
        p1_before = _coeff_map(p1)
        p2_before = _coeff_map(p2)

        p3 = p1 + p2

        self.assertEqual(_coeff_map(p1), p1_before)
        self.assertEqual(_coeff_map(p2), p2_before)
        self.assertAlmostEqual(_coeff_map(p3)["xe"].real, 3.0, places=12)
        self.assertAlmostEqual(_coeff_map(p3)["xe"].imag, 0.0, places=12)

    def test_subtract_polynomial_returns_correct_coefficients(self) -> None:
        p1 = PauliPolynomial(
            "JW",
            [
                PauliTerm(2, ps="xe", pc=2.0),
                PauliTerm(2, ps="ze", pc=1.0),
            ],
        )
        p2 = PauliPolynomial(
            "JW",
            [
                PauliTerm(2, ps="xe", pc=0.5),
                PauliTerm(2, ps="yy", pc=3.0),
            ],
        )

        out = p1 - p2
        cmap = _coeff_map(out)

        self.assertEqual(set(cmap.keys()), {"xe", "ze", "yy"})
        self.assertAlmostEqual(cmap["xe"].real, 1.5, places=12)
        self.assertAlmostEqual(cmap["ze"].real, 1.0, places=12)
        self.assertAlmostEqual(cmap["yy"].real, -3.0, places=12)
        self.assertAlmostEqual(cmap["xe"].imag, 0.0, places=12)
        self.assertAlmostEqual(cmap["ze"].imag, 0.0, places=12)
        self.assertAlmostEqual(cmap["yy"].imag, 0.0, places=12)

    def test_subtract_cancels_identical_strings(self) -> None:
        p1 = PauliPolynomial("JW", [PauliTerm(3, ps="xyz", pc=1.25)])
        p2 = PauliPolynomial("JW", [PauliTerm(3, ps="xyz", pc=1.25)])

        out = p1 - p2

        self.assertEqual(out.count_number_terms(), 0)


if __name__ == "__main__":
    unittest.main()
