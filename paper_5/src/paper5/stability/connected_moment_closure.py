r"""Autonomous regression closure for the connected mixed moment in Eq. (14d).

The retained joint Gram matrix supplies a state-weighted least-squares map
from centered phonon directions to centered electronic Pauli directions.  The
map is applied sequentially from the right to the product
``delta_X_r delta_b_q |psi>``.  This reconstructs an approximation to the
connected electron--two-phonon moment using only the current 31-coordinate
matrix state.  Exact trajectories are not inputs to this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .hubbard_dimer import DimerParameters
from .matrix_reference import (
    MatrixDimerState,
    closed_scalar_to_matrix_state,
    electron_phonon_moment_matrix,
    matrix_derivative_to_closed_scalar,
    matrix_dimer_rhs,
    pauli_repaired_matrix_dimer_rhs,
)

ComplexArray = NDArray[np.complex128]

_PAULI_BASIS = np.asarray(
    [
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, -1.0j], [1.0j, 0.0]],
        [[1.0, 0.0], [0.0, -1.0]],
    ],
    dtype=complex,
)


@dataclass(frozen=True)
class ConditionalPauliRegression:
    """One state-local reconstruction and its numerical diagnostics."""

    mixed_moment: ComplexArray
    phonon_to_pauli_coefficients: ComplexArray
    electronic_support_rank: int
    electronic_gram_minimum_eigenvalue: float
    maximum_normal_equation_relative_residual: float


def _state_weighted_regression_coefficients(
    state: MatrixDimerState,
    *,
    support_tolerance: float,
    physicality_tolerance: float,
) -> tuple[ComplexArray, int, float, float]:
    if support_tolerance <= 0.0:
        raise ValueError("support_tolerance must be positive")
    if physicality_tolerance <= 0.0:
        raise ValueError("physicality_tolerance must be positive")

    joint_gram = electron_phonon_moment_matrix(state)
    cross_gram = joint_gram[:4, 4:]
    electronic_gram = joint_gram[4:, 4:]
    eigenvalues, eigenvectors = np.linalg.eigh(electronic_gram)
    minimum_eigenvalue = float(eigenvalues[0])
    if minimum_eigenvalue < -physicality_tolerance:
        raise ValueError(
            "conditional Pauli regression requires a positive electronic "
            f"Gram block; minimum eigenvalue is {minimum_eigenvalue:.6e}"
        )
    threshold = support_tolerance * max(
        1.0,
        float(np.max(np.abs(eigenvalues))),
    )
    retained = eigenvalues > threshold
    support_rank = int(np.count_nonzero(retained))
    if support_rank:
        inverse = (
            eigenvectors[:, retained] / eigenvalues[retained]
        ) @ eigenvectors[:, retained].conjugate().T
    else:
        inverse = np.zeros_like(electronic_gram)

    # For y in (db_0, db_1, db_0^dagger, db_1^dagger), solve
    # E l_y = <s y>.  The stored cross block is <y^dagger s>, so the
    # row-form coefficients are conjugate(Z E^+).
    coefficients = np.conjugate(cross_gram @ inverse)
    residuals = []
    for direction in range(4):
        target = np.conjugate(cross_gram[direction])
        residual = electronic_gram @ coefficients[direction] - target
        residuals.append(
            float(
                np.linalg.norm(residual)
                / max(np.linalg.norm(target), np.finfo(float).tiny)
            )
        )
    return (
        coefficients,
        support_rank,
        minimum_eigenvalue,
        max(residuals, default=0.0),
    )


def conditional_pauli_regression_mixed_moment(
    state: MatrixDimerState,
    *,
    support_tolerance: float = 1e-10,
    physicality_tolerance: float = 1e-9,
) -> ConditionalPauliRegression:
    r"""Approximate the connected moment ``K[q,r,i,j]`` from retained blocks.

    The rightmost phonon direction first maps to a centered Pauli operator
    ``s_b``.  The remaining phonon direction commutes past it and maps to
    ``s_a``, giving the ordered electronic product ``s_b s_a``.  The centered
    commutator is evaluated exactly in the one-electron site space.
    """

    (
        coefficients,
        support_rank,
        minimum_eigenvalue,
        maximum_residual,
    ) = _state_weighted_regression_coefficients(
        state,
        support_tolerance=support_tolerance,
        physicality_tolerance=physicality_tolerance,
    )
    rho = np.asarray(state.electron_density, dtype=complex)
    identity = np.eye(2, dtype=complex)
    pauli_means = np.asarray(
        [np.trace(rho @ operator) for operator in _PAULI_BASIS],
        dtype=complex,
    )
    centered_pauli = (
        _PAULI_BASIS - pauli_means[:, None, None] * identity
    )

    mixed_moment = np.zeros((2, 2, 2, 2), dtype=complex)
    for q in range(2):
        annihilation_coefficients = coefficients[q]
        for site in range(2):
            displacement_coefficients = (
                coefficients[site] + coefficients[site + 2]
            )
            occupation = np.zeros((2, 2), dtype=complex)
            occupation[site, site] = 1.0
            for i in range(2):
                for j in range(2):
                    one_body_operator = np.zeros((2, 2), dtype=complex)
                    one_body_operator[j, i] = 1.0
                    commutator = (
                        one_body_operator @ occupation
                        - occupation @ one_body_operator
                    )
                    centered_commutator = (
                        commutator
                        - np.trace(rho @ commutator) * identity
                    )
                    electronic_third_moment = np.empty(
                        (3, 3),
                        dtype=complex,
                    )
                    for first in range(3):
                        for second in range(3):
                            electronic_third_moment[first, second] = np.trace(
                                rho
                                @ centered_commutator
                                @ centered_pauli[second]
                                @ centered_pauli[first]
                            )
                    mixed_moment[q, site, i, j] = np.einsum(
                        "a,b,ab->",
                        displacement_coefficients,
                        annihilation_coefficients,
                        electronic_third_moment,
                    )

    return ConditionalPauliRegression(
        mixed_moment=mixed_moment,
        phonon_to_pauli_coefficients=coefficients,
        electronic_support_rank=support_rank,
        electronic_gram_minimum_eigenvalue=minimum_eigenvalue,
        maximum_normal_equation_relative_residual=maximum_residual,
    )


def conditional_pauli_regression_velocity_correction(
    state: MatrixDimerState,
    parameters: DimerParameters,
    *,
    support_tolerance: float = 1e-10,
    physicality_tolerance: float = 1e-9,
) -> ComplexArray:
    r"""Return ``-i g sum_r K[q,r]`` for the reconstructed mixed moment."""

    result = conditional_pauli_regression_mixed_moment(
        state,
        support_tolerance=support_tolerance,
        physicality_tolerance=physicality_tolerance,
    )
    return -1j * parameters.coupling * np.sum(result.mixed_moment, axis=1)


def _conditional_k_matrix_rhs(
    time: float,
    state: MatrixDimerState,
    parameters: DimerParameters,
    *,
    include_pauli_repair: bool,
) -> MatrixDimerState:
    base_rhs = (
        pauli_repaired_matrix_dimer_rhs
        if include_pauli_repair
        else matrix_dimer_rhs
    )
    derivative = base_rhs(time, state, parameters)
    return MatrixDimerState(
        electron_density=derivative.electron_density,
        coherent_phonon=derivative.coherent_phonon,
        phonon_density=derivative.phonon_density,
        anomalous_phonon_density=derivative.anomalous_phonon_density,
        electron_phonon_correlation=(
            derivative.electron_phonon_correlation
            + conditional_pauli_regression_velocity_correction(
                state,
                parameters,
            )
        ),
    )


def conditional_k_matrix_dimer_rhs(
    time: float,
    state: MatrixDimerState,
    parameters: DimerParameters,
) -> MatrixDimerState:
    """Evaluate the archive matrix EOM with the autonomous ``K`` source."""

    return _conditional_k_matrix_rhs(
        time,
        state,
        parameters,
        include_pauli_repair=False,
    )


def conditional_k_pauli_repaired_matrix_dimer_rhs(
    time: float,
    state: MatrixDimerState,
    parameters: DimerParameters,
) -> MatrixDimerState:
    """Evaluate the archive matrix EOM with autonomous ``K`` and Pauli terms."""

    return _conditional_k_matrix_rhs(
        time,
        state,
        parameters,
        include_pauli_repair=True,
    )


def conditional_k_closed_scalar_rhs(
    time: float,
    state: np.ndarray,
    parameters: DimerParameters,
) -> np.ndarray:
    """Return the 31-coordinate archive velocity plus autonomous ``K``."""

    return matrix_derivative_to_closed_scalar(
        conditional_k_matrix_dimer_rhs(
            time,
            closed_scalar_to_matrix_state(state),
            parameters,
        )
    )


def conditional_k_pauli_repaired_closed_scalar_rhs(
    time: float,
    state: np.ndarray,
    parameters: DimerParameters,
) -> np.ndarray:
    """Return the 31-coordinate velocity plus autonomous ``K`` and Pauli terms."""

    return matrix_derivative_to_closed_scalar(
        conditional_k_pauli_repaired_matrix_dimer_rhs(
            time,
            closed_scalar_to_matrix_state(state),
            parameters,
        )
    )


__all__ = [
    "ConditionalPauliRegression",
    "conditional_k_closed_scalar_rhs",
    "conditional_k_matrix_dimer_rhs",
    "conditional_k_pauli_repaired_closed_scalar_rhs",
    "conditional_k_pauli_repaired_matrix_dimer_rhs",
    "conditional_pauli_regression_mixed_moment",
    "conditional_pauli_regression_velocity_correction",
]
