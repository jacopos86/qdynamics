"""Matrix reference for the two-site non-Markovian electron-phonon EOMs.

This module transcribes Eqs. (14a)--(14e) of Riva, Simoni, and Ping,
arXiv:2606.22233v1, and applies the spin-degeneracy factors documented for the
Holstein dimer in ``Dynamics_on_the_Hubbard_DIMER.pdf``, Eqs. (56), (58)--(61).

The existing thirteen-scalar model omits the anomalous two-phonon field of
Eq. (14c).  ``scalar_to_matrix_state`` therefore embeds that model on the
``anomalous_phonon_density == 0`` slice.  This makes it possible to distinguish
agreement within the retained coordinates from failure of that slice to be an
invariant manifold of the complete matrix equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .hubbard_dimer import (
    EXTENDED_FAN_MIGDAL_STATE_NAMES,
    FAN_MIGDAL_STATE_NAMES,
    DimerParameters,
    FloatArray,
    _require_state,
)

ComplexArray = NDArray[np.complex128]

CLOSED_SCALAR_STATE_NAMES = (
    "electron_delta_n",
    "electron_coherence_real",
    "electron_coherence_imag",
    "coherent_0_real",
    "coherent_0_imag",
    "coherent_1_real",
    "coherent_1_imag",
    "phonon_00",
    "phonon_11",
    "phonon_01_real",
    "phonon_01_imag",
    "anomalous_00_real",
    "anomalous_00_imag",
    "anomalous_11_real",
    "anomalous_11_imag",
    "anomalous_01_real",
    "anomalous_01_imag",
    "correlation_shared_trace_real",
    "correlation_shared_trace_imag",
    "correlation_0_diag_difference_real",
    "correlation_0_diag_difference_imag",
    "correlation_0_01_real",
    "correlation_0_01_imag",
    "correlation_0_10_real",
    "correlation_0_10_imag",
    "correlation_1_diag_difference_real",
    "correlation_1_diag_difference_imag",
    "correlation_1_01_real",
    "correlation_1_01_imag",
    "correlation_1_10_real",
    "correlation_1_10_imag",
)

_MATRIX_STATE_FIELDS = (
    "electron_density",
    "coherent_phonon",
    "phonon_density",
    "anomalous_phonon_density",
    "electron_phonon_correlation",
)
_MATRIX_STATE_SHAPES = ((2, 2), (2,), (2, 2), (2, 2), (2, 2, 2))
_MATRIX_STATE_COMPLEX_SIZE = sum(
    int(np.prod(shape)) for shape in _MATRIX_STATE_SHAPES
)


@dataclass(frozen=True)
class MatrixDimerState:
    """Two-electronic-site, two-local-phonon representation of Eqs. (14)."""

    electron_density: ComplexArray
    coherent_phonon: ComplexArray
    phonon_density: ComplexArray
    anomalous_phonon_density: ComplexArray
    electron_phonon_correlation: ComplexArray


def _validate_matrix_state(state: MatrixDimerState) -> None:
    for name, shape in zip(
        _MATRIX_STATE_FIELDS, _MATRIX_STATE_SHAPES, strict=True
    ):
        value = np.asarray(getattr(state, name))
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")


def local_holstein_couplings(parameters: DimerParameters) -> ComplexArray:
    """Return the two real, diagonal local Holstein coupling matrices."""

    coupling = parameters.coupling
    matrices = np.zeros((2, 2, 2), dtype=complex)
    matrices[0, 0, 0] = coupling
    matrices[1, 1, 1] = coupling
    return matrices


def pack_matrix_state(state: MatrixDimerState) -> FloatArray:
    """Pack complex matrix fields into a real vector for ODE solvers."""

    _validate_matrix_state(state)
    complex_vector = np.concatenate(
        [
            np.asarray(getattr(state, name), dtype=complex).reshape(-1)
            for name in _MATRIX_STATE_FIELDS
        ]
    )
    return np.concatenate([complex_vector.real, complex_vector.imag])


def unpack_matrix_state(vector: FloatArray) -> MatrixDimerState:
    """Inverse of :func:`pack_matrix_state`."""

    real_vector = np.asarray(vector, dtype=float)
    expected_size = 2 * _MATRIX_STATE_COMPLEX_SIZE
    if real_vector.shape != (expected_size,):
        raise ValueError(
            f"matrix state vector must have shape {(expected_size,)}, "
            f"got {real_vector.shape}"
        )
    complex_vector = (
        real_vector[:_MATRIX_STATE_COMPLEX_SIZE]
        + 1j * real_vector[_MATRIX_STATE_COMPLEX_SIZE:]
    )
    fields: list[ComplexArray] = []
    offset = 0
    for shape in _MATRIX_STATE_SHAPES:
        size = int(np.prod(shape))
        fields.append(complex_vector[offset : offset + size].reshape(shape))
        offset += size
    return MatrixDimerState(*fields)


def scalar_to_matrix_state(state: FloatArray) -> MatrixDimerState:
    """Embed Eqs. (87)--(99) into their two-site matrix coordinates.

    The mapping uses Eqs. (100)--(107) of the Hubbard-dimer working document,
    zero center-of-mass coherent displacement, and the relative-mode
    correlation identities ``C[1] = -C[0]`` and ``trace(C[0]) = 0``.
    """

    (
        delta_n,
        rho_real,
        rho_imag,
        delta_b_real,
        delta_b_imag,
        phonon_population,
        phonon_coherence,
        delta_corr_real,
        delta_corr_imag,
        delta_corr_imag_plus,
        delta_corr_imag_minus,
        delta_corr_real_plus,
        delta_corr_real_minus,
    ) = _require_state(state, FAN_MIGDAL_STATE_NAMES)

    electron_density = np.array(
        [
            [0.5 * (1.0 + delta_n), rho_real + 1j * rho_imag],
            [rho_real - 1j * rho_imag, 0.5 * (1.0 - delta_n)],
        ],
        dtype=complex,
    )
    delta_b = delta_b_real + 1j * delta_b_imag
    coherent_phonon = np.array([0.5 * delta_b, -0.5 * delta_b], dtype=complex)
    phonon_density = np.array(
        [
            [phonon_population, phonon_coherence],
            [phonon_coherence, phonon_population],
        ],
        dtype=complex,
    )

    corr_11 = delta_corr_real + 1j * delta_corr_imag
    corr_12 = 0.5 * (
        delta_corr_real_plus
        + delta_corr_real_minus
        + 1j * (delta_corr_imag_plus + delta_corr_imag_minus)
    )
    corr_21 = 0.5 * (
        delta_corr_real_plus
        - delta_corr_real_minus
        + 1j * (delta_corr_imag_plus - delta_corr_imag_minus)
    )
    first_correlation = np.array(
        [[corr_11, corr_12], [corr_21, -corr_11]],
        dtype=complex,
    )
    electron_phonon_correlation = np.stack(
        [first_correlation, -first_correlation]
    )

    return MatrixDimerState(
        electron_density=electron_density,
        coherent_phonon=coherent_phonon,
        phonon_density=phonon_density,
        anomalous_phonon_density=np.zeros((2, 2), dtype=complex),
        electron_phonon_correlation=electron_phonon_correlation,
    )


def extended_scalar_to_matrix_state(state: FloatArray) -> MatrixDimerState:
    """Embed the fifteen-coordinate scalar model including matrix Eq. (14c)."""

    array = _require_state(state, EXTENDED_FAN_MIGDAL_STATE_NAMES)
    retained = scalar_to_matrix_state(
        array[: len(FAN_MIGDAL_STATE_NAMES)]
    )
    anomalous_relative = array[-2] + 1j * array[-1]
    anomalous_phonon_density = 0.5 * anomalous_relative * np.array(
        [[1.0, -1.0], [-1.0, 1.0]],
        dtype=complex,
    )
    return MatrixDimerState(
        electron_density=retained.electron_density,
        coherent_phonon=retained.coherent_phonon,
        phonon_density=retained.phonon_density,
        anomalous_phonon_density=anomalous_phonon_density,
        electron_phonon_correlation=retained.electron_phonon_correlation,
    )


def closed_scalar_to_matrix_state(state: FloatArray) -> MatrixDimerState:
    """Map the 31-real-coordinate invariant closure into matrix fields."""

    array = _require_state(state, CLOSED_SCALAR_STATE_NAMES)
    electron_density = np.array(
        [
            [
                0.5 * (1.0 + array[0]),
                array[1] + 1j * array[2],
            ],
            [
                array[1] - 1j * array[2],
                0.5 * (1.0 - array[0]),
            ],
        ],
        dtype=complex,
    )
    coherent_phonon = np.array(
        [
            array[3] + 1j * array[4],
            array[5] + 1j * array[6],
        ],
        dtype=complex,
    )
    phonon_01 = array[9] + 1j * array[10]
    phonon_density = np.array(
        [
            [array[7], phonon_01],
            [phonon_01.conjugate(), array[8]],
        ],
        dtype=complex,
    )
    anomalous_phonon_density = np.array(
        [
            [
                array[11] + 1j * array[12],
                array[15] + 1j * array[16],
            ],
            [
                array[15] + 1j * array[16],
                array[13] + 1j * array[14],
            ],
        ],
        dtype=complex,
    )

    shared_trace = array[17] + 1j * array[18]
    correlations = []
    for offset in (19, 25):
        diagonal_difference = array[offset] + 1j * array[offset + 1]
        correlation_01 = array[offset + 2] + 1j * array[offset + 3]
        correlation_10 = array[offset + 4] + 1j * array[offset + 5]
        correlations.append(
            np.array(
                [
                    [
                        0.5 * (shared_trace + diagonal_difference),
                        correlation_01,
                    ],
                    [
                        correlation_10,
                        0.5 * (shared_trace - diagonal_difference),
                    ],
                ],
                dtype=complex,
            )
        )

    return MatrixDimerState(
        electron_density=electron_density,
        coherent_phonon=coherent_phonon,
        phonon_density=phonon_density,
        anomalous_phonon_density=anomalous_phonon_density,
        electron_phonon_correlation=np.stack(correlations),
    )


def _phonon_rhs_terms(
    state: MatrixDimerState,
    parameters: DimerParameters,
    *,
    spin_degeneracy: float,
) -> dict[str, ComplexArray]:
    """Return the free and correlation-source pieces of Eqs. (14b)--(14c)."""

    phonon = np.asarray(state.phonon_density, dtype=complex)
    anomalous = np.asarray(state.anomalous_phonon_density, dtype=complex)
    correlation = np.asarray(state.electron_phonon_correlation, dtype=complex)
    coupling = local_holstein_couplings(parameters)
    omega = np.full(2, parameters.omega_ph, dtype=float)

    normal_free = np.zeros((2, 2), dtype=complex)
    normal_minus_correlation = np.zeros((2, 2), dtype=complex)
    normal_plus_conjugate_correlation = np.zeros(
        (2, 2),
        dtype=complex,
    )
    for q in range(2):
        for q_prime in range(2):
            normal_free[q, q_prime] = (
                -1j * (omega[q] - omega[q_prime]) * phonon[q, q_prime]
            )
            for one in range(2):
                for two in range(2):
                    normal_minus_correlation[q, q_prime] += (
                        -1j
                        * spin_degeneracy
                        * -coupling[q_prime, one, two]
                        * correlation[q, two, one]
                    )
                    normal_plus_conjugate_correlation[q, q_prime] += (
                        -1j
                        * spin_degeneracy
                        * coupling[q, one, two]
                        * correlation[q_prime, two, one].conjugate()
                    )

    anomalous_free = np.zeros((2, 2), dtype=complex)
    anomalous_first_correlation = np.zeros((2, 2), dtype=complex)
    anomalous_second_correlation = np.zeros((2, 2), dtype=complex)
    for q_prime in range(2):
        for q in range(2):
            anomalous_free[q_prime, q] = (
                -1j
                * (omega[q_prime] + omega[q])
                * anomalous[q_prime, q]
            )
            for one in range(2):
                for two in range(2):
                    anomalous_first_correlation[q_prime, q] += (
                        -1j
                        * spin_degeneracy
                        * coupling[q, one, two]
                        * correlation[q_prime, one, two]
                    )
                    anomalous_second_correlation[q_prime, q] += (
                        -1j
                        * spin_degeneracy
                        * coupling[q_prime, one, two]
                        * correlation[q, one, two]
                    )

    return {
        "eq14b_free_rotation": normal_free,
        "eq14b_minus_correlation": normal_minus_correlation,
        "eq14b_plus_conjugate_correlation": (
            normal_plus_conjugate_correlation
        ),
        "eq14b_correlation_source": (
            normal_minus_correlation
            + normal_plus_conjugate_correlation
        ),
        "eq14c_free_rotation": anomalous_free,
        "eq14c_first_correlation": anomalous_first_correlation,
        "eq14c_second_correlation": anomalous_second_correlation,
        "eq14c_correlation_source": (
            anomalous_first_correlation + anomalous_second_correlation
        ),
    }


def _correlation_homogeneous_rhs(
    time: float,
    state: MatrixDimerState,
    correlation: ComplexArray,
    parameters: DimerParameters,
) -> ComplexArray:
    """Propagate a correlation component through the linear Eq. (14d) part."""

    coherent = np.asarray(state.coherent_phonon, dtype=complex)
    coupling = local_holstein_couplings(parameters)
    drive = parameters.drive_difference(time)
    hamiltonian = np.array(
        [
            [0.5 * drive, -parameters.hopping],
            [-parameters.hopping, -0.5 * drive],
        ],
        dtype=complex,
    )
    for phonon_index in range(2):
        hamiltonian += coupling[phonon_index] * (
            coherent[phonon_index] + coherent[phonon_index].conjugate()
        )

    result = np.zeros((2, 2, 2), dtype=complex)
    for q in range(2):
        result[q] = -1j * (
            hamiltonian @ correlation[q]
            - correlation[q] @ hamiltonian
            + parameters.omega_ph * correlation[q]
        )
    return result


def _correlation_rhs_terms(
    time: float,
    state: MatrixDimerState,
    parameters: DimerParameters,
) -> dict[str, ComplexArray]:
    """Return the transport and atomic source terms of matrix Eq. (14d)."""

    source_terms_by_mode = _correlation_source_terms_by_mode(
        state,
        parameters,
    )
    return {
        "eq14d_transport": _correlation_homogeneous_rhs(
            time,
            state,
            state.electron_phonon_correlation,
            parameters,
        ),
        **{
            name: np.sum(value, axis=1)
            for name, value in source_terms_by_mode.items()
        },
    }


def _correlation_source_terms_by_mode(
    state: MatrixDimerState,
    parameters: DimerParameters,
) -> dict[str, ComplexArray]:
    """Partition each nontransport Eq. (14d) source by coupling mode.

    Every returned array has indices ``[q, q_prime, i, j]``.  Summing over
    ``q_prime`` reproduces the corresponding source returned by
    :func:`_correlation_rhs_terms`.  Keeping that otherwise contracted index
    is the reporting seam needed to compare the archive factorization with the
    exact mixed moment at the same hierarchy level.
    """

    rho = np.asarray(state.electron_density, dtype=complex)
    phonon = np.asarray(state.phonon_density, dtype=complex)
    anomalous = np.asarray(state.anomalous_phonon_density, dtype=complex)
    coupling = local_holstein_couplings(parameters)
    identity = np.eye(2, dtype=complex)

    source_shape = (2, 2, 2, 2)
    bare_pauli_source = np.zeros(source_shape, dtype=complex)
    normal_particle_source = np.zeros(source_shape, dtype=complex)
    normal_hole_source = np.zeros(source_shape, dtype=complex)
    anomalous_first_source = np.zeros(source_shape, dtype=complex)
    anomalous_second_source = np.zeros(source_shape, dtype=complex)

    for q in range(2):
        for q_prime in range(2):
            first_blocking_product = (
                (identity - rho) @ coupling[q_prime].T @ rho
            )
            second_blocking_product = (
                rho @ coupling[q_prime].T @ (identity - rho)
            )
            if q == q_prime:
                bare_pauli_source[q, q_prime] = (
                    -1j * first_blocking_product
                )
            normal_particle_source[q, q_prime] = (
                -1j
                * phonon[q, q_prime]
                * first_blocking_product
            )
            normal_hole_source[q, q_prime] = (
                1j
                * phonon[q, q_prime]
                * second_blocking_product
            )
            for one in range(2):
                for two in range(2):
                    for three in range(2):
                        anomalous_first_source[
                            q, q_prime, one, two
                        ] += (
                            -1j
                            * coupling[q_prime, two, three]
                            * anomalous[q_prime, q]
                            * rho[three, one]
                        )
                        anomalous_second_source[
                            q, q_prime, one, two
                        ] += (
                            1j
                            * coupling[q_prime, three, two]
                            * anomalous[q_prime, q]
                            * rho[one, three]
                        )

    return {
        "eq14d_bare_pauli_source": bare_pauli_source,
        "eq14d_normal_particle_source": normal_particle_source,
        "eq14d_normal_hole_source": normal_hole_source,
        "eq14d_anomalous_first_source": anomalous_first_source,
        "eq14d_anomalous_second_source": anomalous_second_source,
    }


def correlation_source_by_phonon_mode(
    state: MatrixDimerState,
    parameters: DimerParameters,
) -> ComplexArray:
    """Return the archive Eq. (14d) source with its summed mode restored.

    The result has indices ``[q, q_prime, i, j]``.  The first index labels the
    fluctuation operator in ``C[q]``; the second labels the phonon mode carried
    by the electron--phonon interaction.  No new approximation is introduced:
    summing axis one gives the nontransport source already used by
    :func:`matrix_dimer_rhs`.
    """

    terms = _correlation_source_terms_by_mode(state, parameters)
    return sum(
        terms.values(),
        start=np.zeros((2, 2, 2, 2), dtype=complex),
    )


def same_spin_density_covariance(
    electron_density: ComplexArray,
) -> ComplexArray:
    """Return the exact one-up-electron covariance ``Cov(O_ij, n_q)``.

    In the fixed one-spin-up-electron sector, fermionic multiplication closes
    this covariance on the retained one-body density:

    ``P[q,i,j] = delta[i,q] * rho[q,j] - rho[i,j] * rho[q,q]``.
    """

    rho = np.asarray(electron_density, dtype=complex)
    if rho.shape != (2, 2):
        raise ValueError(f"electron_density must have shape (2, 2), got {rho.shape}")
    covariance = np.empty((2, 2, 2), dtype=complex)
    for q in range(2):
        for i in range(2):
            for j in range(2):
                covariance[q, i, j] = (
                    (rho[q, j] if i == q else 0.0)
                    - rho[i, j] * rho[q, q]
                )
    return covariance


def factorized_mixed_correlation_moment(
    state: MatrixDimerState,
) -> ComplexArray:
    """Return the archive-level product for the mixed Eq. (14d) moment.

    The result has indices ``[q, r, i, j]`` and equals
    ``<[O_ij,n_r]> <delta_X_r delta_b_q>``.  It uses only the retained
    ``rho``, ``N``, and ``A`` blocks.
    """

    _validate_matrix_state(state)
    rho = np.asarray(state.electron_density, dtype=complex)
    phonon = np.asarray(state.phonon_density, dtype=complex)
    anomalous = np.asarray(state.anomalous_phonon_density, dtype=complex)
    factorized = np.empty((2, 2, 2, 2), dtype=complex)
    for q in range(2):
        for site in range(2):
            phonon_pair = anomalous[site, q] + phonon[q, site]
            for i in range(2):
                for j in range(2):
                    commutator_mean = (
                        (rho[site, j] if i == site else 0.0)
                        - (rho[i, site] if j == site else 0.0)
                    )
                    factorized[q, site, i, j] = (
                        commutator_mean * phonon_pair
                    )
    return factorized


def same_spin_pauli_velocity_correction(
    state: MatrixDimerState,
    parameters: DimerParameters,
) -> ComplexArray:
    """Replace the effective Eq. (14d) Pauli source by its exact sector form.

    This correction is autonomous: it is reconstructed from the current
    retained ``rho``, ``N``, and ``A`` fields.  The archive source already
    contains an effective same-spin contribution, so that contribution is
    subtracted before the exact fixed-sector covariance is inserted.
    """

    _validate_matrix_state(state)
    archive_source = np.sum(
        correlation_source_by_phonon_mode(state, parameters),
        axis=1,
    )
    factorized_mixed_velocity = -1j * parameters.coupling * np.sum(
        factorized_mixed_correlation_moment(state),
        axis=1,
    )
    archive_pauli_velocity = archive_source - factorized_mixed_velocity
    exact_pauli_velocity = (
        -1j
        * parameters.coupling
        * same_spin_density_covariance(state.electron_density)
    )
    return exact_pauli_velocity - archive_pauli_velocity


def matrix_dimer_rhs(
    time: float,
    state: MatrixDimerState,
    parameters: DimerParameters,
    *,
    spin_degeneracy: float = 2.0,
) -> MatrixDimerState:
    """Evaluate the matrix Eqs. (14a)--(14e) for the Holstein dimer.

    The electronic density and electron-phonon correlation describe one spin
    channel.  ``spin_degeneracy`` accounts for the identical up/down-spin
    contributions to the phonon and coherent-field equations.
    """

    _validate_matrix_state(state)
    rho = np.asarray(state.electron_density, dtype=complex)
    coherent = np.asarray(state.coherent_phonon, dtype=complex)
    phonon = np.asarray(state.phonon_density, dtype=complex)
    anomalous = np.asarray(state.anomalous_phonon_density, dtype=complex)
    correlation = np.asarray(state.electron_phonon_correlation, dtype=complex)
    coupling = local_holstein_couplings(parameters)
    omega = np.full(2, parameters.omega_ph, dtype=float)

    drive = parameters.drive_difference(time)
    hamiltonian = np.array(
        [[0.5 * drive, -parameters.hopping], [-parameters.hopping, -0.5 * drive]],
        dtype=complex,
    )
    for phonon_index in range(2):
        hamiltonian += coupling[phonon_index] * (
            coherent[phonon_index] + coherent[phonon_index].conjugate()
        )

    electron_derivative = -1j * (
        hamiltonian @ rho - rho @ hamiltonian
    )
    correlation_feedback = np.zeros((2, 2), dtype=complex)
    for one in range(2):
        for two in range(2):
            for phonon_index in range(2):
                for three in range(2):
                    correlation_feedback[one, two] += -1j * (
                        coupling[phonon_index, one, three]
                        * correlation[phonon_index, three, two]
                        + coupling[phonon_index, three, one]
                        * correlation[phonon_index, two, three].conjugate()
                    )
    electron_derivative += correlation_feedback + correlation_feedback.conjugate().T

    phonon_terms = _phonon_rhs_terms(
        state,
        parameters,
        spin_degeneracy=spin_degeneracy,
    )
    phonon_derivative = (
        phonon_terms["eq14b_free_rotation"]
        + phonon_terms["eq14b_correlation_source"]
    )
    anomalous_derivative = (
        phonon_terms["eq14c_free_rotation"]
        + phonon_terms["eq14c_correlation_source"]
    )

    correlation_terms = _correlation_rhs_terms(time, state, parameters)
    correlation_derivative = np.zeros((2, 2, 2), dtype=complex)
    for term in correlation_terms.values():
        correlation_derivative += term

    coherent_derivative = np.empty(2, dtype=complex)
    for q in range(2):
        coherent_derivative[q] = -1j * (
            omega[q] * coherent[q]
            + spin_degeneracy * np.sum(coupling[q] * rho)
        )

    return MatrixDimerState(
        electron_density=electron_derivative,
        coherent_phonon=coherent_derivative,
        phonon_density=phonon_derivative,
        anomalous_phonon_density=anomalous_derivative,
        electron_phonon_correlation=correlation_derivative,
    )


def pauli_repaired_matrix_dimer_rhs(
    time: float,
    state: MatrixDimerState,
    parameters: DimerParameters,
    *,
    spin_degeneracy: float = 2.0,
) -> MatrixDimerState:
    """Evaluate Eqs. (14) with the autonomous same-spin Pauli repair in ``dC``."""

    derivative = matrix_dimer_rhs(
        time,
        state,
        parameters,
        spin_degeneracy=spin_degeneracy,
    )
    return MatrixDimerState(
        electron_density=derivative.electron_density,
        coherent_phonon=derivative.coherent_phonon,
        phonon_density=derivative.phonon_density,
        anomalous_phonon_density=derivative.anomalous_phonon_density,
        electron_phonon_correlation=(
            derivative.electron_phonon_correlation
            + same_spin_pauli_velocity_correction(state, parameters)
        ),
    )


def matrix_total_energy(
    state: MatrixDimerState,
    parameters: DimerParameters,
    *,
    spin_degeneracy: float = 2.0,
) -> float:
    """Evaluate the time-independent energy in Eq. (22)."""

    _validate_matrix_state(state)
    rho = np.asarray(state.electron_density, dtype=complex)
    coherent = np.asarray(state.coherent_phonon, dtype=complex)
    phonon = np.asarray(state.phonon_density, dtype=complex)
    correlation = np.asarray(state.electron_phonon_correlation, dtype=complex)
    coupling = local_holstein_couplings(parameters)
    bare_electron = np.array(
        [[0.0, -parameters.hopping], [-parameters.hopping, 0.0]],
        dtype=complex,
    )

    electron_energy = spin_degeneracy * np.trace(bare_electron @ rho).real
    phonon_energy = parameters.omega_ph * (
        np.vdot(coherent, coherent).real + np.trace(phonon).real
    )
    electron_phonon_amplitude = 0.0j
    for q in range(2):
        for one in range(2):
            for two in range(2):
                electron_phonon_amplitude += coupling[q, one, two] * (
                    coherent[q] * rho[two, one]
                    + correlation[q, two, one]
                )
    electron_phonon_energy = (
        2.0 * spin_degeneracy * electron_phonon_amplitude.real
    )
    return float(electron_energy + phonon_energy + electron_phonon_energy)


def matrix_state_to_scalar_coordinates(state: MatrixDimerState) -> FloatArray:
    """Project a matrix state or derivative onto the retained coordinates."""

    _validate_matrix_state(state)
    electron = state.electron_density
    delta_b = state.coherent_phonon[0] - state.coherent_phonon[1]
    phonon = state.phonon_density
    correlation = state.electron_phonon_correlation[0]

    return np.array(
        [
            (electron[0, 0] - electron[1, 1]).real,
            electron[0, 1].real,
            electron[0, 1].imag,
            delta_b.real,
            delta_b.imag,
            phonon[0, 0].real,
            phonon[0, 1].real,
            correlation[0, 0].real,
            correlation[0, 0].imag,
            (correlation[0, 1] + correlation[1, 0]).imag,
            (correlation[0, 1] - correlation[1, 0]).imag,
            (correlation[0, 1] + correlation[1, 0]).real,
            (correlation[0, 1] - correlation[1, 0]).real,
        ],
        dtype=float,
    )


def matrix_derivative_to_scalar(derivative: MatrixDimerState) -> FloatArray:
    """Backward-compatible derivative-specific projection name."""

    return matrix_state_to_scalar_coordinates(derivative)


def matrix_state_to_extended_scalar_coordinates(
    state: MatrixDimerState,
) -> FloatArray:
    """Project a matrix state onto the minimal Eq. (14c)-complete coordinates."""

    retained = matrix_state_to_scalar_coordinates(state)
    anomalous_relative = (
        state.anomalous_phonon_density[0, 0]
        - state.anomalous_phonon_density[0, 1]
    )
    return np.concatenate(
        [
            retained,
            np.array(
                [anomalous_relative.real, anomalous_relative.imag],
                dtype=float,
            ),
        ]
    )


def matrix_derivative_to_extended_scalar(
    derivative: MatrixDimerState,
) -> FloatArray:
    """Derivative-specific alias for the fifteen-coordinate projection."""

    return matrix_state_to_extended_scalar_coordinates(derivative)


def matrix_state_to_closed_scalar_coordinates(
    state: MatrixDimerState,
) -> FloatArray:
    """Project a matrix state onto the explicit 31-coordinate closure."""

    _validate_matrix_state(state)
    electron = state.electron_density
    coherent = state.coherent_phonon
    phonon = state.phonon_density
    anomalous = state.anomalous_phonon_density
    correlation = state.electron_phonon_correlation
    shared_trace = 0.5 * (
        np.trace(correlation[0]) + np.trace(correlation[1])
    )
    values = [
        (electron[0, 0] - electron[1, 1]).real,
        electron[0, 1].real,
        electron[0, 1].imag,
        coherent[0].real,
        coherent[0].imag,
        coherent[1].real,
        coherent[1].imag,
        phonon[0, 0].real,
        phonon[1, 1].real,
        phonon[0, 1].real,
        phonon[0, 1].imag,
        anomalous[0, 0].real,
        anomalous[0, 0].imag,
        anomalous[1, 1].real,
        anomalous[1, 1].imag,
        anomalous[0, 1].real,
        anomalous[0, 1].imag,
        shared_trace.real,
        shared_trace.imag,
    ]
    for phonon_index in range(2):
        diagonal_difference = (
            correlation[phonon_index, 0, 0]
            - correlation[phonon_index, 1, 1]
        )
        values.extend(
            [
                diagonal_difference.real,
                diagonal_difference.imag,
                correlation[phonon_index, 0, 1].real,
                correlation[phonon_index, 0, 1].imag,
                correlation[phonon_index, 1, 0].real,
                correlation[phonon_index, 1, 0].imag,
            ]
        )
    return np.asarray(values, dtype=float)


def matrix_derivative_to_closed_scalar(
    derivative: MatrixDimerState,
) -> FloatArray:
    """Derivative-specific alias for the 31-coordinate projection."""

    return matrix_state_to_closed_scalar_coordinates(derivative)


def closed_scalar_rhs(
    time: float,
    state: FloatArray,
    parameters: DimerParameters,
) -> FloatArray:
    """Evaluate the matrix-exact RHS in 31 explicit real scalar coordinates."""

    return matrix_derivative_to_closed_scalar(
        matrix_dimer_rhs(
            time,
            closed_scalar_to_matrix_state(state),
            parameters,
        )
    )


def pauli_repaired_closed_scalar_rhs(
    time: float,
    state: FloatArray,
    parameters: DimerParameters,
) -> FloatArray:
    """Evaluate the 31-coordinate RHS with the autonomous Pauli repair in ``dC``."""

    return matrix_derivative_to_closed_scalar(
        pauli_repaired_matrix_dimer_rhs(
            time,
            closed_scalar_to_matrix_state(state),
            parameters,
        )
    )


def scalar_embedding_normal_residual(derivative: MatrixDimerState) -> dict[str, float]:
    """Measure derivative components normal to the thirteen-scalar embedding."""

    _validate_matrix_state(derivative)
    correlation = derivative.electron_phonon_correlation
    phonon = derivative.phonon_density
    coherent = derivative.coherent_phonon
    return {
        "anomalous_phonon_rhs_norm": float(
            np.linalg.norm(derivative.anomalous_phonon_density)
        ),
        "correlation_mode_sum_rhs_norm": float(
            np.linalg.norm(correlation[0] + correlation[1])
        ),
        "correlation_trace_rhs_abs": float(abs(np.trace(correlation[0]))),
        "coherent_center_rhs_abs": float(abs(coherent[0] + coherent[1])),
        "phonon_diagonal_difference_rhs_abs": float(
            abs(phonon[0, 0] - phonon[1, 1])
        ),
        "phonon_offdiagonal_asymmetry_rhs_abs": float(
            abs(phonon[0, 1] - phonon[1, 0])
        ),
        "phonon_offdiagonal_imag_rhs_abs": float(abs(phonon[0, 1].imag)),
    }


def extended_scalar_embedding_normal_residual(
    derivative: MatrixDimerState,
) -> dict[str, float]:
    """Measure normal components after adding Eq. (14c).

    The uniform coherent displacement is excluded because fixed electron
    number drives it independently and its Hamiltonian contribution is
    proportional to the identity.  The returned residuals therefore test the
    closed, dynamically relevant quotient used by the scalar equations.
    """

    _validate_matrix_state(derivative)
    correlation = derivative.electron_phonon_correlation
    phonon = derivative.phonon_density
    anomalous = derivative.anomalous_phonon_density
    return {
        "correlation_mode_sum_rhs_norm": float(
            np.linalg.norm(correlation[0] + correlation[1])
        ),
        "correlation_trace_rhs_abs": float(abs(np.trace(correlation[0]))),
        "phonon_diagonal_difference_rhs_abs": float(
            abs(phonon[0, 0] - phonon[1, 1])
        ),
        "phonon_offdiagonal_asymmetry_rhs_abs": float(
            abs(phonon[0, 1] - phonon[1, 0])
        ),
        "phonon_offdiagonal_imag_rhs_abs": float(abs(phonon[0, 1].imag)),
        "anomalous_diagonal_difference_rhs_abs": float(
            abs(anomalous[0, 0] - anomalous[1, 1])
        ),
        "anomalous_offdiagonal_asymmetry_rhs_abs": float(
            abs(anomalous[0, 1] - anomalous[1, 0])
        ),
        "anomalous_relative_structure_rhs_abs": float(
            abs(anomalous[0, 0] + anomalous[0, 1])
        ),
    }


def closed_scalar_embedding_normal_residual(
    derivative: MatrixDimerState,
) -> dict[str, float]:
    """Measure components normal to the 31-coordinate invariant closure."""

    _validate_matrix_state(derivative)
    electron = derivative.electron_density
    phonon = derivative.phonon_density
    anomalous = derivative.anomalous_phonon_density
    correlation = derivative.electron_phonon_correlation
    return {
        "electron_antihermitian_rhs_norm": float(
            np.linalg.norm(electron - electron.conjugate().T)
        ),
        "electron_trace_rhs_abs": float(abs(np.trace(electron))),
        "phonon_antihermitian_rhs_norm": float(
            np.linalg.norm(phonon - phonon.conjugate().T)
        ),
        "anomalous_asymmetry_rhs_norm": float(
            np.linalg.norm(anomalous - anomalous.T)
        ),
        "correlation_trace_difference_rhs_abs": float(
            abs(np.trace(correlation[0]) - np.trace(correlation[1]))
        ),
    }


def boson_moment_matrix(state: MatrixDimerState) -> ComplexArray:
    """Return the normally ordered two-mode moment matrix.

    Positive semidefiniteness is the full second-moment bosonic uncertainty
    condition.  Its one-mode determinant reduces to
    ``n * (n + 1) - |m|**2 >= 0``.
    """

    _validate_matrix_state(state)
    phonon = np.asarray(state.phonon_density, dtype=complex)
    anomalous = np.asarray(state.anomalous_phonon_density, dtype=complex)
    return np.block(
        [
            [phonon.T, anomalous.conjugate()],
            [anomalous, np.eye(2, dtype=complex) + phonon],
        ]
    )


_ELECTRON_FLUCTUATION_BASIS = np.asarray(
    [
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, -1.0j], [1.0j, 0.0]],
        [[1.0, 0.0], [0.0, -1.0]],
    ],
    dtype=complex,
)


def _electron_fluctuation_moment_matrix(
    electron_density: ComplexArray,
) -> ComplexArray:
    """Return ``<delta sigma_a delta sigma_b>`` for the Pauli basis."""

    electron = np.asarray(electron_density, dtype=complex)
    means = np.asarray(
        [
            np.trace(electron @ operator).real
            for operator in _ELECTRON_FLUCTUATION_BASIS
        ],
        dtype=float,
    )
    moment = np.empty((3, 3), dtype=complex)
    for row, first in enumerate(_ELECTRON_FLUCTUATION_BASIS):
        for column, second in enumerate(_ELECTRON_FLUCTUATION_BASIS):
            moment[row, column] = (
                np.trace(electron @ first @ second)
                - means[row] * means[column]
            )
    return 0.5 * (moment + moment.conjugate().T)


def _electron_fluctuation_moment_derivative(
    electron_density: ComplexArray,
    electron_derivative: ComplexArray,
) -> ComplexArray:
    """Differentiate ``<delta sigma_a delta sigma_b>``."""

    electron = np.asarray(electron_density, dtype=complex)
    derivative = np.asarray(electron_derivative, dtype=complex)
    means = np.asarray(
        [
            np.trace(electron @ operator).real
            for operator in _ELECTRON_FLUCTUATION_BASIS
        ],
        dtype=float,
    )
    mean_derivatives = np.asarray(
        [
            np.trace(derivative @ operator).real
            for operator in _ELECTRON_FLUCTUATION_BASIS
        ],
        dtype=float,
    )
    moment_derivative = np.empty((3, 3), dtype=complex)
    for row, first in enumerate(_ELECTRON_FLUCTUATION_BASIS):
        for column, second in enumerate(_ELECTRON_FLUCTUATION_BASIS):
            moment_derivative[row, column] = (
                np.trace(derivative @ first @ second)
                - mean_derivatives[row] * means[column]
                - means[row] * mean_derivatives[column]
            )
    return 0.5 * (
        moment_derivative + moment_derivative.conjugate().T
    )


def _correlation_pauli_block(correlation: ComplexArray) -> ComplexArray:
    """Contract the two correlation matrices against the Pauli basis."""

    values = np.empty((2, 3), dtype=complex)
    for phonon_index in range(2):
        for operator_index, operator in enumerate(
            _ELECTRON_FLUCTUATION_BASIS
        ):
            values[phonon_index, operator_index] = np.trace(
                correlation[phonon_index] @ operator
            )
    return values


def electron_phonon_moment_matrix(state: MatrixDimerState) -> ComplexArray:
    """Return the joint boson--electron fluctuation Gram matrix.

    The operator list is
    ``(delta b_0, delta b_1, delta b_0^dagger, delta b_1^dagger,
    delta sigma_x, delta sigma_y, delta sigma_z)``.  Its upper-left block is
    :func:`boson_moment_matrix`; its off-diagonal block contains the retained
    connected correlation ``C``.  Every quantum state therefore requires
    this seven-by-seven Hermitian matrix to be positive semidefinite.
    """

    _validate_matrix_state(state)
    boson = boson_moment_matrix(state)
    electron = _electron_fluctuation_moment_matrix(
        state.electron_density
    )
    correlation = _correlation_pauli_block(
        state.electron_phonon_correlation
    )
    cross = np.vstack([correlation.conjugate(), correlation])
    moment = np.block(
        [
            [boson, cross],
            [cross.conjugate().T, electron],
        ]
    )
    return 0.5 * (moment + moment.conjugate().T)


def electron_phonon_moment_derivative(
    state: MatrixDimerState,
    derivative: MatrixDimerState,
) -> ComplexArray:
    """Differentiate :func:`electron_phonon_moment_matrix` exactly."""

    _validate_matrix_state(state)
    _validate_matrix_state(derivative)
    boson_derivative = _boson_moment_derivative_from_blocks(
        derivative.phonon_density,
        derivative.anomalous_phonon_density,
    )
    electron_derivative = _electron_fluctuation_moment_derivative(
        state.electron_density,
        derivative.electron_density,
    )
    correlation_derivative = _correlation_pauli_block(
        derivative.electron_phonon_correlation
    )
    cross_derivative = np.vstack(
        [correlation_derivative.conjugate(), correlation_derivative]
    )
    moment_derivative = np.block(
        [
            [boson_derivative, cross_derivative],
            [cross_derivative.conjugate().T, electron_derivative],
        ]
    )
    return 0.5 * (
        moment_derivative + moment_derivative.conjugate().T
    )


def _boson_moment_derivative_from_blocks(
    normal_derivative: ComplexArray,
    anomalous_derivative: ComplexArray,
) -> ComplexArray:
    """Lift normal/anomalous phonon derivatives to the moment matrix."""

    return np.block(
        [
            [normal_derivative.T, anomalous_derivative.conjugate()],
            [anomalous_derivative, normal_derivative],
        ]
    )


def _closed_scalar_boson_derivative_blocks(
    derivative: FloatArray,
) -> tuple[ComplexArray, ComplexArray]:
    """Extract ``dN`` and ``dA`` from a 31-coordinate derivative vector."""

    array = _require_state(derivative, CLOSED_SCALAR_STATE_NAMES)
    normal_01 = array[9] + 1j * array[10]
    normal = np.array(
        [
            [array[7], normal_01],
            [normal_01.conjugate(), array[8]],
        ],
        dtype=complex,
    )
    anomalous = np.array(
        [
            [
                array[11] + 1j * array[12],
                array[15] + 1j * array[16],
            ],
            [
                array[15] + 1j * array[16],
                array[13] + 1j * array[14],
            ],
        ],
        dtype=complex,
    )
    return normal, anomalous


def _closed_scalar_correlation_blocks(
    derivative: FloatArray,
) -> ComplexArray:
    """Extract the two correlation matrices from a 31-coordinate vector."""

    array = _require_state(derivative, CLOSED_SCALAR_STATE_NAMES)
    shared_trace = array[17] + 1j * array[18]
    correlations = []
    for offset in (19, 25):
        diagonal_difference = array[offset] + 1j * array[offset + 1]
        correlations.append(
            np.array(
                [
                    [
                        0.5 * (shared_trace + diagonal_difference),
                        array[offset + 2] + 1j * array[offset + 3],
                    ],
                    [
                        array[offset + 4] + 1j * array[offset + 5],
                        0.5 * (shared_trace - diagonal_difference),
                    ],
                ],
                dtype=complex,
            )
        )
    return np.stack(correlations)


def boson_moment_derivative_terms(
    state: MatrixDimerState,
    parameters: DimerParameters,
    *,
    spin_degeneracy: float = 2.0,
    residual_subtraction: FloatArray | None = None,
) -> dict[str, ComplexArray]:
    """Decompose ``dM_B/dt`` into Eqs. (14b)--(14d) contributions.

    The bosonic moment matrix depends only on the normal and anomalous phonon
    moments.  Equation (14d) therefore has zero direct instantaneous
    contribution; it acts indirectly through the correlation field appearing
    in the source terms of Eqs. (14b) and (14c).  When supplied,
    ``residual_subtraction`` is the 31-coordinate constant removed by
    Eq. (112).
    """

    _validate_matrix_state(state)
    phonon_terms = _phonon_rhs_terms(
        state,
        parameters,
        spin_degeneracy=spin_degeneracy,
    )
    zero = np.zeros((2, 2), dtype=complex)
    terms = {
        "eq14b_free_rotation": _boson_moment_derivative_from_blocks(
            phonon_terms["eq14b_free_rotation"],
            zero,
        ),
        "eq14b_minus_correlation": _boson_moment_derivative_from_blocks(
            phonon_terms["eq14b_minus_correlation"],
            zero,
        ),
        "eq14b_plus_conjugate_correlation": (
            _boson_moment_derivative_from_blocks(
                phonon_terms["eq14b_plus_conjugate_correlation"],
                zero,
            )
        ),
        "eq14c_free_rotation": _boson_moment_derivative_from_blocks(
            zero,
            phonon_terms["eq14c_free_rotation"],
        ),
        "eq14c_first_correlation": _boson_moment_derivative_from_blocks(
            zero,
            phonon_terms["eq14c_first_correlation"],
        ),
        "eq14c_second_correlation": _boson_moment_derivative_from_blocks(
            zero,
            phonon_terms["eq14c_second_correlation"],
        ),
        "eq14d_direct": np.zeros((4, 4), dtype=complex),
    }
    if residual_subtraction is not None:
        normal_residual, anomalous_residual = (
            _closed_scalar_boson_derivative_blocks(residual_subtraction)
        )
        terms["eq112_residual_subtraction"] = (
            -_boson_moment_derivative_from_blocks(
                normal_residual,
                anomalous_residual,
            )
        )
    return terms


def boson_boundary_flux_decomposition(
    time: float,
    state: MatrixDimerState,
    parameters: DimerParameters,
    *,
    spin_degeneracy: float = 2.0,
    residual_subtraction: FloatArray | None = None,
) -> dict[str, object]:
    """Project every moment derivative term on the minimum-eigenvalue mode."""

    moment = boson_moment_matrix(state)
    moment = 0.5 * (moment + moment.conjugate().T)
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    null_mode = eigenvectors[:, 0]
    derivative_terms = boson_moment_derivative_terms(
        state,
        parameters,
        spin_degeneracy=spin_degeneracy,
        residual_subtraction=residual_subtraction,
    )
    term_fluxes = {}
    reconstructed_derivative = np.zeros((4, 4), dtype=complex)
    for name, derivative in derivative_terms.items():
        hermitian_derivative = 0.5 * (
            derivative + derivative.conjugate().T
        )
        reconstructed_derivative += hermitian_derivative
        term_fluxes[name] = float(
            np.vdot(
                null_mode,
                hermitian_derivative @ null_mode,
            ).real
        )

    direct_derivative_state = matrix_dimer_rhs(
        time,
        state,
        parameters,
        spin_degeneracy=spin_degeneracy,
    )
    direct_derivative = _boson_moment_derivative_from_blocks(
        direct_derivative_state.phonon_density,
        direct_derivative_state.anomalous_phonon_density,
    )
    if residual_subtraction is not None:
        normal_residual, anomalous_residual = (
            _closed_scalar_boson_derivative_blocks(residual_subtraction)
        )
        direct_derivative -= _boson_moment_derivative_from_blocks(
            normal_residual,
            anomalous_residual,
        )
    direct_derivative = 0.5 * (
        direct_derivative + direct_derivative.conjugate().T
    )
    direct_flux = float(
        np.vdot(null_mode, direct_derivative @ null_mode).real
    )
    total_flux = float(sum(term_fluxes.values()))
    finite_difference_step = 1e-6
    finite_difference_flux = float(
        (
            np.linalg.eigvalsh(
                moment + finite_difference_step * direct_derivative
            )[0]
            - np.linalg.eigvalsh(
                moment - finite_difference_step * direct_derivative
            )[0]
        )
        / (2.0 * finite_difference_step)
    )
    result: dict[str, object] = {
        "minimum_eigenvalue": float(eigenvalues[0]),
        "next_eigenvalue": float(eigenvalues[1]),
        "spectral_gap": float(eigenvalues[1] - eigenvalues[0]),
        "null_eigenvector": null_mode,
        "term_fluxes": term_fluxes,
        "total_flux": total_flux,
        "direct_rhs_flux": direct_flux,
        "reconstruction_error": float(
            np.linalg.norm(reconstructed_derivative - direct_derivative)
        ),
        "finite_difference_flux": finite_difference_flux,
        "finite_difference_error": float(
            abs(finite_difference_flux - direct_flux)
        ),
    }
    result.update(
        {
            f"{name}_flux": value
            for name, value in term_fluxes.items()
        }
    )
    result["eq14b_correlation_source_flux"] = float(
        term_fluxes["eq14b_minus_correlation"]
        + term_fluxes["eq14b_plus_conjugate_correlation"]
    )
    result["eq14c_correlation_source_flux"] = float(
        term_fluxes["eq14c_first_correlation"]
        + term_fluxes["eq14c_second_correlation"]
    )
    return result


def closed_eq14d_history_flux_decomposition(
    parameters: DimerParameters,
    initial_state: FloatArray,
    *,
    residual_subtraction: FloatArray | None = None,
    maximum_time: float = 2.0,
    spin_degeneracy: float = 2.0,
) -> dict[str, object]:
    """Trace Eq. (14d) sources into the first bosonic-boundary flux.

    The realized 31-coordinate trajectory supplies the time-dependent
    electronic, coherent-phonon, normal-phonon, and anomalous-phonon fields.
    Along that trajectory Eq. (14d) is linear in its correlation field.  A
    variation-of-constants system therefore propagates the initial
    correlation and every independent source through the same homogeneous
    transport.  Their sum exactly reconstructs the realized boundary
    correlation, and the linear Eq. (14b)/(14c) source maps then assign its
    boundary flux by causal history.
    """

    from scipy.integrate import solve_ivp

    initial = _require_state(initial_state, CLOSED_SCALAR_STATE_NAMES).copy()
    if maximum_time <= 0.0:
        raise ValueError("maximum_time must be positive")
    if residual_subtraction is None:
        residual = np.zeros_like(initial)
        include_residual_history = False
    else:
        residual = _require_state(
            residual_subtraction,
            CLOSED_SCALAR_STATE_NAMES,
        ).copy()
        include_residual_history = True

    source_names = [
        "eq14d_bare_pauli_source",
        "eq14d_normal_particle_source",
        "eq14d_normal_hole_source",
        "eq14d_anomalous_first_source",
        "eq14d_anomalous_second_source",
    ]
    if include_residual_history:
        source_names.append("eq112_correlation_subtraction")
    history_names = ["initial_correlation", *source_names]
    correlation_shape = (2, 2, 2)
    correlation_size = int(np.prod(correlation_shape))
    packed_correlation_size = 2 * correlation_size

    def pack_correlation(correlation: ComplexArray) -> FloatArray:
        flat = np.asarray(correlation, dtype=complex).reshape(-1)
        return np.concatenate([flat.real, flat.imag])

    def unpack_correlation(vector: FloatArray) -> ComplexArray:
        flat = np.asarray(vector, dtype=float)
        if flat.shape != (packed_correlation_size,):
            raise ValueError(
                "packed correlation must have shape "
                f"{(packed_correlation_size,)}, got {flat.shape}"
            )
        complex_flat = flat[:correlation_size] + 1j * flat[correlation_size:]
        return complex_flat.reshape(correlation_shape)

    initial_matrix = closed_scalar_to_matrix_state(initial)
    zero_correlation = np.zeros(correlation_shape, dtype=complex)
    augmented_initial = [initial]
    for name in history_names:
        correlation = (
            initial_matrix.electron_phonon_correlation
            if name == "initial_correlation"
            else zero_correlation
        )
        augmented_initial.append(pack_correlation(correlation))
    augmented_initial_vector = np.concatenate(augmented_initial)
    residual_correlation = _closed_scalar_correlation_blocks(residual)

    def augmented_rhs(time: float, vector: FloatArray) -> FloatArray:
        scalar_state = vector[: len(CLOSED_SCALAR_STATE_NAMES)]
        matrix_state = closed_scalar_to_matrix_state(scalar_state)
        scalar_derivative = (
            closed_scalar_rhs(time, scalar_state, parameters) - residual
        )
        source_terms = _correlation_rhs_terms(
            time,
            matrix_state,
            parameters,
        )
        source_terms["eq112_correlation_subtraction"] = -residual_correlation

        derivatives = [scalar_derivative]
        offset = len(CLOSED_SCALAR_STATE_NAMES)
        for name in history_names:
            correlation = unpack_correlation(
                vector[offset : offset + packed_correlation_size]
            )
            derivative = _correlation_homogeneous_rhs(
                time,
                matrix_state,
                correlation,
                parameters,
            )
            if name != "initial_correlation":
                derivative += source_terms[name]
            derivatives.append(pack_correlation(derivative))
            offset += packed_correlation_size
        return np.concatenate(derivatives)

    def boundary_event(_time: float, vector: FloatArray) -> float:
        matrix_state = closed_scalar_to_matrix_state(
            vector[: len(CLOSED_SCALAR_STATE_NAMES)]
        )
        moment = boson_moment_matrix(matrix_state)
        moment = 0.5 * (moment + moment.conjugate().T)
        return float(np.linalg.eigvalsh(moment)[0])

    boundary_event.terminal = True
    boundary_event.direction = -1
    solution = solve_ivp(
        augmented_rhs,
        (0.0, maximum_time),
        augmented_initial_vector,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        max_step=0.02,
        events=boundary_event,
    )
    if not solution.success:
        raise RuntimeError(
            f"Eq. (14d) history integration failed: {solution.message}"
        )
    if len(solution.t_events[0]) != 1:
        raise RuntimeError(
            "no bosonic moment boundary crossing found before "
            f"t={maximum_time}"
        )

    crossing_time = float(solution.t_events[0][0])
    crossing_vector = np.asarray(solution.y_events[0][0], dtype=float)
    crossing_scalar_state = crossing_vector[: len(CLOSED_SCALAR_STATE_NAMES)]
    crossing_matrix_state = closed_scalar_to_matrix_state(
        crossing_scalar_state
    )
    histories: dict[str, ComplexArray] = {}
    offset = len(CLOSED_SCALAR_STATE_NAMES)
    for name in history_names:
        histories[name] = unpack_correlation(
            crossing_vector[offset : offset + packed_correlation_size]
        )
        offset += packed_correlation_size
    reconstructed_correlation = sum(
        histories.values(),
        start=np.zeros(correlation_shape, dtype=complex),
    )

    moment = boson_moment_matrix(crossing_matrix_state)
    moment = 0.5 * (moment + moment.conjugate().T)
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    null_mode = eigenvectors[:, 0]
    zero_phonon = np.zeros((2, 2), dtype=complex)

    def correlation_fluxes(correlation: ComplexArray) -> tuple[float, float]:
        component_state = MatrixDimerState(
            electron_density=crossing_matrix_state.electron_density,
            coherent_phonon=crossing_matrix_state.coherent_phonon,
            phonon_density=crossing_matrix_state.phonon_density,
            anomalous_phonon_density=(
                crossing_matrix_state.anomalous_phonon_density
            ),
            electron_phonon_correlation=correlation,
        )
        phonon_terms = _phonon_rhs_terms(
            component_state,
            parameters,
            spin_degeneracy=spin_degeneracy,
        )
        eq14b_derivative = _boson_moment_derivative_from_blocks(
            phonon_terms["eq14b_correlation_source"],
            zero_phonon,
        )
        eq14c_derivative = _boson_moment_derivative_from_blocks(
            zero_phonon,
            phonon_terms["eq14c_correlation_source"],
        )
        return (
            float(np.vdot(null_mode, eq14b_derivative @ null_mode).real),
            float(np.vdot(null_mode, eq14c_derivative @ null_mode).real),
        )

    eq14b_flux_by_history = {}
    eq14c_flux_by_history = {}
    correlation_norm_by_history = {}
    for name, correlation in histories.items():
        eq14b_flux, eq14c_flux = correlation_fluxes(correlation)
        eq14b_flux_by_history[name] = eq14b_flux
        eq14c_flux_by_history[name] = eq14c_flux
        correlation_norm_by_history[name] = float(np.linalg.norm(correlation))

    realized_eq14b_flux, realized_eq14c_flux = correlation_fluxes(
        crossing_matrix_state.electron_phonon_correlation
    )
    crossing_rhs_terms = _correlation_rhs_terms(
        crossing_time,
        crossing_matrix_state,
        parameters,
    )
    if include_residual_history:
        crossing_rhs_terms["eq112_correlation_subtraction"] = (
            -residual_correlation
        )
    instantaneous_eq14b_flux_rate_by_term = {
        name: correlation_fluxes(term)[0]
        for name, term in crossing_rhs_terms.items()
    }
    instantaneous_eq14c_flux_rate_by_term = {
        name: correlation_fluxes(term)[1]
        for name, term in crossing_rhs_terms.items()
    }
    source_histories = {
        name: value
        for name, value in eq14b_flux_by_history.items()
        if name != "initial_correlation"
    }
    dominant_outward_history = min(
        source_histories,
        key=source_histories.__getitem__,
    )

    return {
        "crossing_time": crossing_time,
        "minimum_eigenvalue": float(eigenvalues[0]),
        "spectral_gap": float(eigenvalues[1] - eigenvalues[0]),
        "correlation_by_history": histories,
        "correlation_norm_by_history": correlation_norm_by_history,
        "correlation_reconstruction_error": float(
            np.linalg.norm(
                reconstructed_correlation
                - crossing_matrix_state.electron_phonon_correlation
            )
        ),
        "eq14b_flux_by_history": eq14b_flux_by_history,
        "eq14c_flux_by_history": eq14c_flux_by_history,
        "realized_eq14b_flux": realized_eq14b_flux,
        "realized_eq14c_flux": realized_eq14c_flux,
        "eq14b_flux_reconstruction_error": float(
            abs(sum(eq14b_flux_by_history.values()) - realized_eq14b_flux)
        ),
        "eq14c_flux_reconstruction_error": float(
            abs(sum(eq14c_flux_by_history.values()) - realized_eq14c_flux)
        ),
        "instantaneous_eq14b_flux_rate_by_term": (
            instantaneous_eq14b_flux_rate_by_term
        ),
        "instantaneous_eq14c_flux_rate_by_term": (
            instantaneous_eq14c_flux_rate_by_term
        ),
        "dominant_outward_history": dominant_outward_history,
    }


def discover_invariant_closure(
    parameters: DimerParameters,
    *,
    samples_per_iteration: int = 300,
    validation_samples: int = 500,
    random_seed: int = 310014,
) -> dict[str, object]:
    """Discover the real linear invariant hull containing the 15D projection.

    Random polynomial probes supply candidate normal directions, while an SVD
    adds every numerically independent direction.  The returned dimension is
    then checked on a disjoint validation sample.  The explicit 31-coordinate
    map is implemented separately and tested for tangency.
    """

    if samples_per_iteration < 10:
        raise ValueError("samples_per_iteration must be at least 10")
    if validation_samples < 10:
        raise ValueError("validation_samples must be at least 10")

    scalar_zero = np.zeros(len(EXTENDED_FAN_MIGDAL_STATE_NAMES), dtype=float)
    origin = pack_matrix_state(extended_scalar_to_matrix_state(scalar_zero))
    directions = []
    for index in range(len(EXTENDED_FAN_MIGDAL_STATE_NAMES)):
        scalar_direction = scalar_zero.copy()
        scalar_direction[index] = 1.0
        directions.append(
            pack_matrix_state(
                extended_scalar_to_matrix_state(scalar_direction)
            )
            - origin
        )
    basis = np.linalg.qr(np.stack(directions, axis=1))[0]
    rng = np.random.default_rng(random_seed)
    added_ranks = []

    for _iteration in range(origin.size):
        residuals = []
        for _ in range(samples_per_iteration):
            coefficients = rng.normal(scale=0.25, size=basis.shape[1])
            time = float(rng.uniform(0.0, 3.0))
            vector = origin + basis @ coefficients
            derivative = pack_matrix_state(
                matrix_dimer_rhs(
                    time,
                    unpack_matrix_state(vector),
                    parameters,
                )
            )
            residuals.append(derivative - basis @ (basis.T @ derivative))
        residual_matrix = np.stack(residuals, axis=1)
        left_vectors, singular_values, _ = np.linalg.svd(
            residual_matrix,
            full_matrices=False,
        )
        scale = max(1.0, float(singular_values[0]))
        rank = int(np.sum(singular_values > 1e-10 * scale))
        if rank == 0:
            break
        additions = left_vectors[:, :rank]
        additions -= basis @ (basis.T @ additions)
        orthogonal_additions, triangular = np.linalg.qr(additions)
        keep = np.abs(np.diag(triangular)) > 1e-10
        accepted = int(np.sum(keep))
        if accepted == 0:
            break
        added_ranks.append(accepted)
        basis = np.column_stack(
            [basis, orthogonal_additions[:, keep]]
        )
    else:
        raise RuntimeError("invariant-closure discovery did not converge")

    maximum_validation_residual = 0.0
    for _ in range(validation_samples):
        coefficients = rng.normal(scale=0.3, size=basis.shape[1])
        time = float(rng.uniform(0.0, 5.0))
        vector = origin + basis @ coefficients
        derivative = pack_matrix_state(
            matrix_dimer_rhs(
                time,
                unpack_matrix_state(vector),
                parameters,
            )
        )
        residual = derivative - basis @ (basis.T @ derivative)
        maximum_validation_residual = max(
            maximum_validation_residual,
            float(np.linalg.norm(residual)),
        )

    return {
        "ambient_real_dimension": int(origin.size),
        "initial_dimension": len(EXTENDED_FAN_MIGDAL_STATE_NAMES),
        "closure_dimension": int(basis.shape[1]),
        "added_ranks_by_iteration": added_ranks,
        "random_seed": random_seed,
        "samples_per_iteration": samples_per_iteration,
        "validation_samples": validation_samples,
        "maximum_validation_residual": maximum_validation_residual,
    }
