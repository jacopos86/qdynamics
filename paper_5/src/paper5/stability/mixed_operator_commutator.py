"""Exact first Liouvillian layer of the relative-mode mixed operators."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray

from .hubbard_dimer import DimerParameters, GaussianSineDrive
from .multi_coherent import _oscillator_operators, relative_holstein_hamiltonian

ComplexArray = NDArray[np.complex128]

MIXED_OPERATOR_LABELS = (
    "a_sigma_x",
    "a_sigma_y",
    "a_sigma_z",
    "a_dagger_sigma_x",
    "a_dagger_sigma_y",
    "a_dagger_sigma_z",
)

FIRST_AUXILIARY_OPERATOR_LABELS = (
    "a_squared_sigma_x",
    "a_squared_sigma_y",
    "number_sigma_x",
    "number_sigma_y",
    "a_dagger_squared_sigma_x",
    "a_dagger_squared_sigma_y",
    "opposite_z_sigma_x",
    "opposite_z_sigma_y",
    "opposite_z_sigma_z",
)


@dataclass(frozen=True)
class MixedOperatorCommutatorAudit:
    """Matrix residuals for the six analytic mixed-operator identities."""

    labels: tuple[str, ...]
    infinite_boson_relative_residual: NDArray[np.float64]
    cutoff_corrected_relative_residual: NDArray[np.float64]
    cutoff_boundary_relative_norm: NDArray[np.float64]


def _liouvillian(hamiltonian: ComplexArray, operator: ComplexArray) -> ComplexArray:
    return 1j * (hamiltonian @ operator - operator @ hamiltonian)


def mixed_operator_commutator_audit(
    parameters: DimerParameters,
    *,
    time: float,
    relative_dimension: int,
    drive_protocol: GaussianSineDrive | None = None,
) -> MixedOperatorCommutatorAudit:
    """Compare the analytic first-layer formulas with finite-cutoff matrices."""

    if relative_dimension < 2:
        raise ValueError("relative_dimension must be at least two")
    annihilation, creation = _oscillator_operators(relative_dimension)
    number = creation @ annihilation
    boson_identity = np.eye(relative_dimension, dtype=complex)
    top_projector = np.zeros_like(boson_identity)
    top_projector[-1, -1] = 1.0

    identity = np.eye(2, dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    electron_identity = np.eye(4, dtype=complex)
    up = tuple(np.kron(operator, identity) for operator in (sigma_x, sigma_y, sigma_z))
    down_z = np.kron(identity, sigma_z)
    electron_total_z = np.kron(sigma_z, identity) + down_z

    def electron(operator: ComplexArray) -> ComplexArray:
        return np.kron(operator, boson_identity)

    def boson(operator: ComplexArray) -> ComplexArray:
        return np.kron(electron_identity, operator)

    def product(
        electronic: ComplexArray,
        phononic: ComplexArray,
    ) -> ComplexArray:
        return np.kron(electronic, phononic)

    x_up, y_up, z_up = up
    drive = (
        parameters.drive_difference(time)
        if drive_protocol is None
        else drive_protocol.difference(time)
    )
    kappa = parameters.coupling / sqrt(2.0)
    hopping = parameters.hopping
    omega = parameters.omega_ph
    dimension = float(relative_dimension)
    hamiltonian = relative_holstein_hamiltonian(
        float(time),
        parameters,
        relative_dimension=relative_dimension,
        drive_protocol=drive_protocol,
    )

    entrance = (
        product(x_up, annihilation),
        product(y_up, annihilation),
        product(z_up, annihilation),
        product(x_up, creation),
        product(y_up, creation),
        product(z_up, creation),
    )
    identity_full = np.eye(4 * relative_dimension, dtype=complex)
    infinite = (
        -1j * omega * entrance[0]
        - drive * entrance[1]
        - 2.0 * kappa * product(y_up, annihilation @ annihilation + number)
        - kappa * electron(y_up)
        - 1j * kappa * electron(down_z @ x_up),
        -1j * omega * entrance[1]
        + 2.0 * hopping * entrance[2]
        + drive * entrance[0]
        + 2.0 * kappa * product(x_up, annihilation @ annihilation + number)
        + kappa * electron(x_up)
        - 1j * kappa * electron(down_z @ y_up),
        -1j * omega * entrance[2]
        - 2.0 * hopping * entrance[1]
        - 1j * kappa * identity_full
        - 1j * kappa * electron(down_z @ z_up),
        1j * omega * entrance[3]
        - drive * entrance[4]
        - 2.0 * kappa * product(y_up, creation @ creation + number)
        - kappa * electron(y_up)
        + 1j * kappa * electron(down_z @ x_up),
        1j * omega * entrance[4]
        + 2.0 * hopping * entrance[5]
        + drive * entrance[3]
        + 2.0 * kappa * product(x_up, creation @ creation + number)
        + kappa * electron(x_up)
        + 1j * kappa * electron(down_z @ y_up),
        1j * omega * entrance[5]
        - 2.0 * hopping * entrance[4]
        + 1j * kappa * identity_full
        + 1j * kappa * electron(down_z @ z_up),
    )
    boundary = (
        kappa
        * dimension
        * product(y_up + 1j * down_z @ x_up, top_projector),
        kappa
        * dimension
        * product(-x_up + 1j * down_z @ y_up, top_projector),
        1j
        * kappa
        * dimension
        * product(np.eye(4) + down_z @ z_up, top_projector),
        kappa
        * dimension
        * product(y_up - 1j * down_z @ x_up, top_projector),
        kappa
        * dimension
        * product(-x_up - 1j * down_z @ y_up, top_projector),
        -1j
        * kappa
        * dimension
        * product(np.eye(4) + down_z @ z_up, top_projector),
    )
    exact = tuple(_liouvillian(hamiltonian, operator) for operator in entrance)
    denominator = np.asarray(
        [max(float(np.linalg.norm(value)), np.finfo(float).tiny) for value in exact]
    )
    infinite_residual = np.asarray(
        [
            np.linalg.norm(actual - candidate) / scale
            for actual, candidate, scale in zip(exact, infinite, denominator, strict=True)
        ],
        dtype=float,
    )
    corrected_residual = np.asarray(
        [
            np.linalg.norm(actual - candidate - correction) / scale
            for actual, candidate, correction, scale in zip(
                exact,
                infinite,
                boundary,
                denominator,
                strict=True,
            )
        ],
        dtype=float,
    )
    boundary_relative = np.asarray(
        [
            np.linalg.norm(correction) / scale
            for correction, scale in zip(boundary, denominator, strict=True)
        ],
        dtype=float,
    )
    return MixedOperatorCommutatorAudit(
        labels=MIXED_OPERATOR_LABELS,
        infinite_boson_relative_residual=infinite_residual,
        cutoff_corrected_relative_residual=corrected_residual,
        cutoff_boundary_relative_norm=boundary_relative,
    )


__all__ = [
    "FIRST_AUXILIARY_OPERATOR_LABELS",
    "MIXED_OPERATOR_LABELS",
    "MixedOperatorCommutatorAudit",
    "mixed_operator_commutator_audit",
]
