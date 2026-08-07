"""Correlated initial-state construction for the scalar stability harness."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares

from .exact_reference import exact_holstein_ground_state
from .hubbard_dimer import (
    DimerParameters,
    EXTENDED_FAN_MIGDAL_STATE_NAMES,
    FAN_MIGDAL_STATE_NAMES,
    FloatArray,
    fan_migdal_rhs,
    fan_migdal_with_anomalous_rhs,
)
from .matrix_reference import (
    MatrixDimerState,
    boson_moment_matrix,
    closed_scalar_rhs,
    closed_scalar_to_matrix_state,
    matrix_state_to_extended_scalar_coordinates,
    matrix_state_to_closed_scalar_coordinates,
    matrix_state_to_scalar_coordinates,
    matrix_total_energy,
    scalar_to_matrix_state,
)


@dataclass(frozen=True)
class ScalarStationaryState:
    """A source-connected scalar fixed point and its basic diagnostics."""

    state: FloatArray
    residual_norm: float
    energy: float
    exact_seed_energy: float
    exact_seed_residual_norm: float
    electron_eigenvalues: FloatArray
    phonon_eigenvalues: FloatArray
    phonon_cutoff: int


def electron_density_eigenvalues(state: FloatArray) -> FloatArray:
    """Return the two eigenvalues of the scalar electronic 1-RDM."""

    delta_n, rho_real, rho_imag = np.asarray(state, dtype=float)[:3]
    bloch_length = np.sqrt(
        delta_n**2 + 4.0 * (rho_real**2 + rho_imag**2)
    )
    return np.array(
        [0.5 * (1.0 - bloch_length), 0.5 * (1.0 + bloch_length)],
        dtype=float,
    )


def phonon_density_eigenvalues(state: FloatArray) -> FloatArray:
    """Return eigenvalues of the retained two-site phonon fluctuation matrix."""

    array = np.asarray(state, dtype=float)
    population = array[5]
    coherence = array[6]
    return np.array(
        [population - abs(coherence), population + abs(coherence)],
        dtype=float,
    )


def exact_ground_scalar_coordinates(
    parameters: DimerParameters,
    *,
    phonon_cutoff: int = 16,
) -> FloatArray:
    """Contract the exact ground state into the thirteen scalar coordinates."""

    undriven = replace(parameters, drive_amplitude=0.0)
    exact = exact_holstein_ground_state(
        undriven,
        phonon_cutoff=phonon_cutoff,
    )
    return matrix_state_to_scalar_coordinates(exact.matrix_state)


def exact_ground_extended_scalar_coordinates(
    parameters: DimerParameters,
    *,
    phonon_cutoff: int = 16,
) -> FloatArray:
    """Contract the exact ground state into the fifteen-coordinate model."""

    undriven = replace(parameters, drive_amplitude=0.0)
    exact = exact_holstein_ground_state(
        undriven,
        phonon_cutoff=phonon_cutoff,
    )
    return matrix_state_to_extended_scalar_coordinates(exact.matrix_state)


def exact_ground_closed_scalar_coordinates(
    parameters: DimerParameters,
    *,
    phonon_cutoff: int = 16,
) -> FloatArray:
    """Contract the exact ground state into the invariant 31D closure."""

    undriven = replace(parameters, drive_amplitude=0.0)
    exact = exact_holstein_ground_state(
        undriven,
        phonon_cutoff=phonon_cutoff,
    )
    return matrix_state_to_closed_scalar_coordinates(exact.matrix_state)


def hartree_fock_closed_scalar_coordinates(
    parameters: DimerParameters,
) -> FloatArray:
    """Return HF/zero correlations with the decoupled center mode equilibrated."""

    scalar_state = np.zeros(13, dtype=float)
    scalar_state[1] = 0.5
    matrix_state = scalar_to_matrix_state(scalar_state)
    equilibrium_center = np.full(
        2,
        -parameters.coupling / parameters.omega_ph,
        dtype=complex,
    )
    matrix_state = MatrixDimerState(
        electron_density=matrix_state.electron_density,
        coherent_phonon=(
            matrix_state.coherent_phonon + equilibrium_center
        ),
        phonon_density=matrix_state.phonon_density,
        anomalous_phonon_density=matrix_state.anomalous_phonon_density,
        electron_phonon_correlation=(
            matrix_state.electron_phonon_correlation
        ),
    )
    return matrix_state_to_closed_scalar_coordinates(matrix_state)


def closed_electron_eigenvalues(state: FloatArray) -> FloatArray:
    """Return electronic 1-RDM eigenvalues for a 31D closed scalar state."""

    matrix_state = closed_scalar_to_matrix_state(state)
    electron = 0.5 * (
        matrix_state.electron_density
        + matrix_state.electron_density.conjugate().T
    )
    return np.linalg.eigvalsh(electron)


def closed_phonon_eigenvalues(state: FloatArray) -> FloatArray:
    """Return normal two-mode phonon-density eigenvalues."""

    matrix_state = closed_scalar_to_matrix_state(state)
    phonon = 0.5 * (
        matrix_state.phonon_density
        + matrix_state.phonon_density.conjugate().T
    )
    return np.linalg.eigvalsh(phonon)


def closed_boson_moment_eigenvalues(state: FloatArray) -> FloatArray:
    """Return eigenvalues of the full normal/anomalous boson moment matrix."""

    moment = boson_moment_matrix(closed_scalar_to_matrix_state(state))
    moment = 0.5 * (moment + moment.conjugate().T)
    return np.linalg.eigvalsh(moment)


def relative_boson_uncertainty_margin(state: FloatArray) -> float:
    """Return ``n_rel(n_rel + 1) - |m_rel|^2`` for the relative mode.

    Nonnegative values are the one-mode second-moment uncertainty condition.
    The anomalous amplitude is zero for the thirteen-coordinate model.
    """

    array = np.asarray(state, dtype=float)
    if array.shape not in (
        (len(FAN_MIGDAL_STATE_NAMES),),
        (len(EXTENDED_FAN_MIGDAL_STATE_NAMES),),
    ):
        raise ValueError(
            "state must use the thirteen- or fifteen-coordinate model"
        )
    relative_population = array[5] - array[6]
    anomalous_squared = (
        float(array[-2] ** 2 + array[-1] ** 2)
        if array.size == len(EXTENDED_FAN_MIGDAL_STATE_NAMES)
        else 0.0
    )
    return float(
        relative_population * (relative_population + 1.0)
        - anomalous_squared
    )


def source_connected_stationary_state(
    parameters: DimerParameters,
    *,
    phonon_cutoff: int = 16,
    residual_tolerance: float = 1e-10,
) -> ScalarStationaryState:
    """Find the scalar fixed point continuously connected to the exact seed.

    The nonlinear scalar equations have many algebraic roots, including roots
    that violate elementary density-matrix positivity.  This routine therefore
    starts from exact ground-state contractions and reports, rather than hides,
    the electronic and phonon eigenvalue checks.  It does not claim a global
    minimum over the unresolved higher-moment representability constraints.
    """

    if residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be positive")
    undriven = replace(parameters, drive_amplitude=0.0)
    exact = exact_holstein_ground_state(
        undriven,
        phonon_cutoff=phonon_cutoff,
    )
    seed = matrix_state_to_scalar_coordinates(exact.matrix_state)
    seed_residual = fan_migdal_rhs(0.0, seed, undriven)
    solution = least_squares(
        lambda state: fan_migdal_rhs(0.0, state, undriven),
        seed,
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
        max_nfev=100_000,
    )
    state = np.asarray(solution.x, dtype=float)
    residual_norm = float(
        np.linalg.norm(fan_migdal_rhs(0.0, state, undriven))
    )
    if not solution.success or residual_norm > residual_tolerance:
        raise RuntimeError(
            "source-connected stationary solve failed: "
            f"success={solution.success}, residual={residual_norm:.3e}"
        )

    matrix_state = scalar_to_matrix_state(state)
    equilibrium_center = np.full(
        2,
        -undriven.coupling / undriven.omega_ph,
        dtype=complex,
    )
    matrix_state = MatrixDimerState(
        electron_density=matrix_state.electron_density,
        coherent_phonon=(
            matrix_state.coherent_phonon + equilibrium_center
        ),
        phonon_density=matrix_state.phonon_density,
        anomalous_phonon_density=matrix_state.anomalous_phonon_density,
        electron_phonon_correlation=(
            matrix_state.electron_phonon_correlation
        ),
    )

    return ScalarStationaryState(
        state=state,
        residual_norm=residual_norm,
        energy=matrix_total_energy(matrix_state, undriven),
        exact_seed_energy=exact.energy,
        exact_seed_residual_norm=float(np.linalg.norm(seed_residual)),
        electron_eigenvalues=electron_density_eigenvalues(state),
        phonon_eigenvalues=phonon_density_eigenvalues(state),
        phonon_cutoff=phonon_cutoff,
    )


def residual_subtracted_rhs(
    parameters: DimerParameters,
    initial_state: FloatArray,
):
    """Return the Eq. (112) diagnostic regularization of the scalar RHS."""

    undriven = replace(parameters, drive_amplitude=0.0)
    initial = np.asarray(initial_state, dtype=float).copy()
    initial_residual = fan_migdal_rhs(0.0, initial, undriven)

    def rhs(time: float, state: FloatArray) -> FloatArray:
        return fan_migdal_rhs(time, state, parameters) - initial_residual

    return rhs


def extended_residual_subtracted_rhs(
    parameters: DimerParameters,
    initial_state: FloatArray,
):
    """Return Eq. (112) for the fifteen-coordinate Eq. (14c) projection."""

    undriven = replace(parameters, drive_amplitude=0.0)
    initial = np.asarray(initial_state, dtype=float).copy()
    initial = _require_extended_state(initial)
    initial_residual = fan_migdal_with_anomalous_rhs(
        0.0,
        initial,
        undriven,
    )

    def rhs(time: float, state: FloatArray) -> FloatArray:
        return (
            fan_migdal_with_anomalous_rhs(time, state, parameters)
            - initial_residual
        )

    return rhs


def closed_residual_subtracted_rhs(
    parameters: DimerParameters,
    initial_state: FloatArray,
):
    """Return Eq. (112) for the invariant 31D scalar closure."""

    undriven = replace(parameters, drive_amplitude=0.0)
    initial = np.asarray(initial_state, dtype=float).copy()
    initial_residual = closed_scalar_rhs(0.0, initial, undriven)

    def rhs(time: float, state: FloatArray) -> FloatArray:
        return closed_scalar_rhs(time, state, parameters) - initial_residual

    return rhs


def _require_extended_state(state: FloatArray) -> FloatArray:
    array = np.asarray(state, dtype=float)
    expected = (len(EXTENDED_FAN_MIGDAL_STATE_NAMES),)
    if array.shape != expected:
        raise ValueError(
            f"expected extended state shape {expected}, got {array.shape}"
        )
    return array
