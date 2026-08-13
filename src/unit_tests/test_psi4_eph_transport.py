"""Algebraic tests for the Psi4 finite-difference EPH potential."""

import numpy as np
import pytest

pytest.importorskip("psi4")

from src.chemistry.psi4.electron_phonon_fd_solver import (
    FiniteDifferenceElectronPhononSolver,
)


def test_mean_field_potential_uses_rhf_spin_factors():
    nuclear = np.array([[-1.0, 0.2], [0.2, -0.7]])
    eri = np.arange(16, dtype=float).reshape(2, 2, 2, 2) / 20.0
    density_a = np.array([[0.8, 0.1], [0.1, 0.2]])
    density_b = density_a.copy()

    actual = FiniteDifferenceElectronPhononSolver._mean_field_potential(
        nuclear, eri, density_a, density_b)
    expected = (
        nuclear
        + np.einsum("mnkl,kl->mn", eri, 2.0 * density_a)
        - np.einsum("mknl,kl->mn", eri, density_a)
    )
    np.testing.assert_allclose(actual, expected, atol=1.0e-14)
