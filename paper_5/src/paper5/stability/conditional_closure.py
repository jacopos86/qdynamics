r"""Electron-conditioned Gaussian closure for the Holstein moment hierarchy.

For every relative-mode Weyl monomial, the retained Pauli moments reconstruct
an electronic moment matrix

``Gamma[a,b] = Tr_ph(rho_ep W(x**a p**b))``.

The closure models the relative phonon as an operator-valued displaced and
squeezed Gaussian conditioned on the electronic state.  Hermitian electronic
operators ``D_x``, ``D_p`` and ``V_xx``, ``V_xp``, ``V_pp`` are fixed by the
retained moment matrices through second phonon order.  A Jordan-product Wick
recurrence then predicts the first omitted moments.  The construction uses
only the retained autonomous coordinates; exact states enter only later when
the prediction is audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .moment_hierarchy import (
    IDENTITY,
    PAULI_LABELS,
    MomentKey,
    PreparedTerminalMomentClosure,
    _canonical_key,
)

ComplexMatrix = np.ndarray

_PAULI_MATRICES: dict[str, ComplexMatrix] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}
_TWO_SPIN_PAULI = {
    (left, right): np.kron(
        _PAULI_MATRICES[left],
        _PAULI_MATRICES[right],
    )
    for left in PAULI_LABELS
    for right in PAULI_LABELS
}


def _jordan(left: ComplexMatrix, right: ComplexMatrix) -> ComplexMatrix:
    return 0.5 * (left @ right + right @ left)


def _hermitian(matrix: ComplexMatrix) -> ComplexMatrix:
    return 0.5 * (matrix + matrix.conj().T)


@dataclass(frozen=True)
class ElectronicConditionedGaussianClosure:
    """Prepare an operator-valued conditional Gaussian fifth-moment rule."""

    support_tolerance: float = 1e-10
    physicality_tolerance: float = 1e-9
    imaginary_tolerance: float = 1e-8
    name: str = "electronic_conditioned_gaussian"

    def __post_init__(self) -> None:
        if self.support_tolerance <= 0.0:
            raise ValueError("support_tolerance must be positive")
        if self.physicality_tolerance <= 0.0:
            raise ValueError("physicality_tolerance must be positive")
        if self.imaginary_tolerance <= 0.0:
            raise ValueError("imaginary_tolerance must be positive")

    def prepare(
        self,
        moments: Mapping[MomentKey, float],
        maximum_degree: int,
    ) -> PreparedTerminalMomentClosure:
        if maximum_degree != 4:
            raise ValueError(
                "the electronic-conditioned Gaussian closure currently "
                "requires the complete fourth-order hierarchy"
            )
        return ElectronicConditionedGaussianResolver(
            moments=moments,
            maximum_degree=maximum_degree,
            support_tolerance=self.support_tolerance,
            physicality_tolerance=self.physicality_tolerance,
            imaginary_tolerance=self.imaginary_tolerance,
        )


@dataclass
class ElectronicConditionedGaussianResolver:
    """State-local operator-valued Gaussian reconstruction."""

    moments: Mapping[MomentKey, float]
    maximum_degree: int
    support_tolerance: float
    physicality_tolerance: float
    imaginary_tolerance: float
    support_rank: int = field(init=False)
    density_minimum_eigenvalue: float = field(init=False)
    maximum_jordan_relative_residual: float = field(init=False)
    _matrix_cache: dict[tuple[int, int], ComplexMatrix] = field(
        init=False,
        repr=False,
    )
    _displacement_x: ComplexMatrix = field(init=False, repr=False)
    _displacement_p: ComplexMatrix = field(init=False, repr=False)
    _covariance_xx: ComplexMatrix = field(init=False, repr=False)
    _covariance_xp: ComplexMatrix = field(init=False, repr=False)
    _covariance_pp: ComplexMatrix = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._matrix_cache = {
            (x_power, degree - x_power): self._electronic_moment_matrix(
                x_power,
                degree - x_power,
            )
            for degree in range(3)
            for x_power in range(degree, -1, -1)
        }
        density = self._matrix_cache[(0, 0)]
        eigenvalues = np.linalg.eigvalsh(density)
        self.density_minimum_eigenvalue = float(np.min(eigenvalues))
        if self.density_minimum_eigenvalue < -self.physicality_tolerance:
            raise ValueError(
                "electronic-conditioned closure requires a positive "
                "electronic density matrix; minimum eigenvalue is "
                f"{self.density_minimum_eigenvalue:.6e}"
            )
        threshold = self.support_tolerance * max(
            1.0,
            float(np.max(np.abs(eigenvalues))),
        )
        self.support_rank = int(np.count_nonzero(eigenvalues > threshold))

        residuals: list[float] = []
        self._displacement_x = self._solve_jordan(
            density,
            self._matrix_cache[(1, 0)],
            threshold,
            residuals,
        )
        self._displacement_p = self._solve_jordan(
            density,
            self._matrix_cache[(0, 1)],
            threshold,
            residuals,
        )
        self._covariance_xx = self._solve_jordan(
            density,
            self._matrix_cache[(2, 0)]
            - _jordan(
                self._displacement_x,
                self._matrix_cache[(1, 0)],
            ),
            threshold,
            residuals,
        )
        self._covariance_pp = self._solve_jordan(
            density,
            self._matrix_cache[(0, 2)]
            - _jordan(
                self._displacement_p,
                self._matrix_cache[(0, 1)],
            ),
            threshold,
            residuals,
        )
        cross_displacement = 0.5 * (
            _jordan(
                self._displacement_x,
                self._matrix_cache[(0, 1)],
            )
            + _jordan(
                self._displacement_p,
                self._matrix_cache[(1, 0)],
            )
        )
        self._covariance_xp = self._solve_jordan(
            density,
            self._matrix_cache[(1, 1)] - cross_displacement,
            threshold,
            residuals,
        )
        self.maximum_jordan_relative_residual = max(residuals, default=0.0)

    @property
    def diagnostics(self) -> dict[str, float | int]:
        return {
            "support_rank": self.support_rank,
            "density_minimum_eigenvalue": self.density_minimum_eigenvalue,
            "maximum_jordan_relative_residual": (
                self.maximum_jordan_relative_residual
            ),
        }

    def _raw_moment(self, key: MomentKey) -> float:
        if key.degree == 0:
            return 1.0
        try:
            return float(self.moments[key])
        except KeyError as error:
            raise ValueError(
                f"conditional closure requires retained moment {key}"
            ) from error

    def _electronic_moment_matrix(
        self,
        x_power: int,
        p_power: int,
    ) -> ComplexMatrix:
        matrix = np.zeros((4, 4), dtype=complex)
        for left in PAULI_LABELS:
            for right in PAULI_LABELS:
                key = _canonical_key(left, right, x_power, p_power)
                matrix += (
                    0.25
                    * self._raw_moment(key)
                    * _TWO_SPIN_PAULI[(left, right)]
                )
        return _hermitian(matrix)

    @staticmethod
    def _solve_jordan(
        density: ComplexMatrix,
        target: ComplexMatrix,
        threshold: float,
        residuals: list[float],
    ) -> ComplexMatrix:
        eigenvalues, eigenvectors = np.linalg.eigh(density)
        transformed = eigenvectors.conj().T @ target @ eigenvectors
        denominators = eigenvalues[:, None] + eigenvalues[None, :]
        solution = np.zeros_like(transformed)
        supported = np.abs(denominators) > threshold
        solution[supported] = (
            2.0 * transformed[supported] / denominators[supported]
        )
        result = _hermitian(eigenvectors @ solution @ eigenvectors.conj().T)
        residual = _jordan(result, density) - target
        residuals.append(
            float(
                np.linalg.norm(residual)
                / max(np.linalg.norm(target), np.finfo(float).tiny)
            )
        )
        return result

    def _predicted_matrix(
        self,
        x_power: int,
        p_power: int,
    ) -> ComplexMatrix:
        powers = (x_power, p_power)
        if powers in self._matrix_cache:
            return self._matrix_cache[powers]
        paths: list[ComplexMatrix] = []
        if x_power > 0:
            prediction = _jordan(
                self._displacement_x,
                self._predicted_matrix(x_power - 1, p_power),
            )
            if x_power >= 2:
                prediction += (x_power - 1) * _jordan(
                    self._covariance_xx,
                    self._predicted_matrix(x_power - 2, p_power),
                )
            if p_power > 0:
                prediction += p_power * _jordan(
                    self._covariance_xp,
                    self._predicted_matrix(x_power - 1, p_power - 1),
                )
            paths.append(prediction)
        if p_power > 0:
            prediction = _jordan(
                self._displacement_p,
                self._predicted_matrix(x_power, p_power - 1),
            )
            if p_power >= 2:
                prediction += (p_power - 1) * _jordan(
                    self._covariance_pp,
                    self._predicted_matrix(x_power, p_power - 2),
                )
            if x_power > 0:
                prediction += x_power * _jordan(
                    self._covariance_xp,
                    self._predicted_matrix(x_power - 1, p_power - 1),
                )
            paths.append(prediction)
        if not paths:  # pragma: no cover - the identity is cached
            raise ValueError("cannot predict the zero-degree matrix")
        matrix = _hermitian(sum(paths) / len(paths))
        self._matrix_cache[powers] = matrix
        return matrix

    def moment(self, key: MomentKey) -> float:
        if key.degree <= self.maximum_degree:
            return self._raw_moment(key)
        if key.degree != self.maximum_degree + 1:
            raise ValueError(
                "conditional closure generated unsupported degree "
                f"{key.degree} for order {self.maximum_degree}"
            )
        matrix = self._predicted_matrix(key.x_power, key.p_power)
        if key.spin_up == key.spin_down:
            value = np.trace(
                _TWO_SPIN_PAULI[(key.spin_up, key.spin_down)] @ matrix
            )
        else:
            value = 0.5 * (
                np.trace(
                    _TWO_SPIN_PAULI[(key.spin_up, key.spin_down)]
                    @ matrix
                )
                + np.trace(
                    _TWO_SPIN_PAULI[(key.spin_down, key.spin_up)]
                    @ matrix
                )
            )
        if abs(value.imag) > self.imaginary_tolerance:
            raise FloatingPointError(
                "conditional Gaussian reconstructed a complex Hermitian "
                f"moment for {key}: {value}"
            )
        return float(value.real)


ELECTRONIC_CONDITIONED_GAUSSIAN_CLOSURE = (
    ElectronicConditionedGaussianClosure()
)


__all__ = [
    "ELECTRONIC_CONDITIONED_GAUSSIAN_CLOSURE",
    "ElectronicConditionedGaussianClosure",
    "ElectronicConditionedGaussianResolver",
]
