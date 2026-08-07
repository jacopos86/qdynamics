"""Archive-backed positive commutator-moment closure foundations.

This module contains the pieces of the adaptive positive commutator-moment
design that can be verified independently of a terminal-moment closure:

* the 29-real-coordinate raw-moment chart whose affine moment matrix has the
  centered joint Gram matrix as its Schur complement; and
* the exact dimer reduction of the connected electron--two-phonon and
  opposite-spin entrance terms from spin-symmetric Pauli--Weyl moments.

The archive equations remain the ambient vector field.  The helpers here only
construct the missing ``C``-block source that is added to that field.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from .hubbard_dimer import DimerParameters, FloatArray
from .matrix_reference import (
    MatrixDimerState,
    same_spin_pauli_velocity_correction,
)
from .moment_hierarchy import (
    IDENTITY,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    MomentKey,
    THIRD_ORDER_HIERARCHY,
    build_moment_keys,
)

ComplexArray = NDArray[np.complex128]

RAW_MOMENT_COORDINATE_NAMES = (
    "bloch_x",
    "bloch_y",
    "bloch_z",
    "coherent_0_real",
    "coherent_0_imag",
    "coherent_1_real",
    "coherent_1_imag",
    "raw_normal_00",
    "raw_normal_11",
    "raw_normal_01_real",
    "raw_normal_01_imag",
    "raw_anomalous_00_real",
    "raw_anomalous_00_imag",
    "raw_anomalous_11_real",
    "raw_anomalous_11_imag",
    "raw_anomalous_01_real",
    "raw_anomalous_01_imag",
    *tuple(
        f"raw_b_sigma_{mode}_{axis}_{part}"
        for mode in range(2)
        for axis in ("x", "y", "z")
        for part in ("real", "imag")
    ),
)

_PAULI = {
    IDENTITY: np.eye(2, dtype=complex),
    PAULI_X: np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    PAULI_Y: np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    PAULI_Z: np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}
_PAULI_LABELS = (PAULI_X, PAULI_Y, PAULI_Z)
_PAULI_ARRAY = np.asarray([_PAULI[label] for label in _PAULI_LABELS])
_SITE_SIGNS = np.asarray([1.0, -1.0])
_RELATIVE_MODE = np.asarray([1.0, -1.0], dtype=complex) / np.sqrt(2.0)


def _spin_degree(key: MomentKey) -> int:
    return int(key.spin_up != IDENTITY) + int(key.spin_down != IDENTITY)


def _is_archive_resolved_relative_moment(key: MomentKey) -> bool:
    """Whether one relative-mode moment is fixed by ``(rho,B,N,A,C)``."""

    spin_degree = _spin_degree(key)
    boson_degree = key.x_power + key.p_power
    return (spin_degree == 0 and boson_degree in (1, 2)) or (
        spin_degree == 1 and boson_degree in (0, 1)
    )


ARCHIVE_RELATIVE_MOMENT_KEYS = tuple(
    key
    for key in build_moment_keys(2)
    if _is_archive_resolved_relative_moment(key)
)

HIDDEN_RELATIVE_MOMENT_KEYS = tuple(
    key
    for key in THIRD_ORDER_HIERARCHY.moment_keys
    if not _is_archive_resolved_relative_moment(key)
)

ENTRANCE_RELATIVE_MOMENT_KEYS = tuple(
    key
    for key in HIDDEN_RELATIVE_MOMENT_KEYS
    if (
        key.degree == 2
        and _spin_degree(key) == 2
        and key.x_power + key.p_power == 0
    )
    or (
        key.degree == 3
        and _spin_degree(key) == 1
        and key.x_power + key.p_power == 2
    )
)

INITIAL_PROMOTION_CANDIDATE_KEYS = tuple(
    key
    for key in HIDDEN_RELATIVE_MOMENT_KEYS
    if key not in set(ENTRANCE_RELATIVE_MOMENT_KEYS)
)


def _require_real(value: complex, *, name: str, tolerance: float = 2e-9) -> float:
    scalar = complex(value)
    if abs(scalar.imag) > tolerance:
        raise ValueError(f"{name} must be real, got {scalar}")
    return float(scalar.real)


def _pack_hermitian(matrix: ComplexArray) -> FloatArray:
    value = np.asarray(matrix, dtype=complex)
    return np.asarray(
        [value[0, 0].real, value[1, 1].real, value[0, 1].real, value[0, 1].imag],
        dtype=float,
    )


def _unpack_hermitian(coordinates: FloatArray) -> ComplexArray:
    value = np.asarray(coordinates, dtype=float)
    off_diagonal = value[2] + 1j * value[3]
    return np.asarray(
        [[value[0], off_diagonal], [off_diagonal.conjugate(), value[1]]],
        dtype=complex,
    )


def _pack_symmetric(matrix: ComplexArray) -> FloatArray:
    value = np.asarray(matrix, dtype=complex)
    return np.asarray(
        [
            value[0, 0].real,
            value[0, 0].imag,
            value[1, 1].real,
            value[1, 1].imag,
            value[0, 1].real,
            value[0, 1].imag,
        ],
        dtype=float,
    )


def _unpack_symmetric(coordinates: FloatArray) -> ComplexArray:
    value = np.asarray(coordinates, dtype=float)
    diagonal_0 = value[0] + 1j * value[1]
    diagonal_1 = value[2] + 1j * value[3]
    off_diagonal = value[4] + 1j * value[5]
    return np.asarray(
        [[diagonal_0, off_diagonal], [off_diagonal, diagonal_1]],
        dtype=complex,
    )


def matrix_state_to_raw_moment_coordinates(
    state: MatrixDimerState,
    *,
    structural_tolerance: float = 2e-9,
) -> FloatArray:
    """Extract the 29 independent raw moments from a physical matrix tuple.

    The correlation blocks must be traceless.  The map is otherwise the exact
    change of variables

    ``Nbar=N+B B^dagger``, ``Abar=A+B B^T``, and
    ``d[q,a]=Tr(C[q] sigma_a)+B[q] r[a]``.
    """

    rho = np.asarray(state.electron_density, dtype=complex)
    coherent = np.asarray(state.coherent_phonon, dtype=complex)
    normal = np.asarray(state.phonon_density, dtype=complex)
    anomalous = np.asarray(state.anomalous_phonon_density, dtype=complex)
    correlation = np.asarray(state.electron_phonon_correlation, dtype=complex)
    if rho.shape != (2, 2) or coherent.shape != (2,):
        raise ValueError("state does not have dimer electronic/phonon shapes")
    if normal.shape != (2, 2) or anomalous.shape != (2, 2):
        raise ValueError("state does not have two-mode second moments")
    if correlation.shape != (2, 2, 2):
        raise ValueError("state does not have two correlation matrices")
    if np.linalg.norm(rho - rho.conjugate().T) > structural_tolerance:
        raise ValueError("electron density must be Hermitian")
    if abs(np.trace(rho) - 1.0) > structural_tolerance:
        raise ValueError("electron density must have unit trace")
    if np.linalg.norm(normal - normal.conjugate().T) > structural_tolerance:
        raise ValueError("normal phonon moment must be Hermitian")
    if np.linalg.norm(anomalous - anomalous.T) > structural_tolerance:
        raise ValueError("anomalous phonon moment must be symmetric")
    if np.max(np.abs(np.trace(correlation, axis1=1, axis2=2))) > structural_tolerance:
        raise ValueError("correlation matrices must be traceless")

    bloch = np.asarray(
        [np.trace(rho @ operator).real for operator in _PAULI_ARRAY],
        dtype=float,
    )
    raw_normal = normal + np.outer(coherent, coherent.conjugate())
    raw_anomalous = anomalous + np.outer(coherent, coherent)
    raw_cross = np.empty((2, 3), dtype=complex)
    for mode in range(2):
        for axis, operator in enumerate(_PAULI_ARRAY):
            raw_cross[mode, axis] = (
                np.trace(correlation[mode] @ operator)
                + coherent[mode] * bloch[axis]
            )

    coordinates = np.concatenate(
        [
            bloch,
            np.asarray(
                [
                    coherent[0].real,
                    coherent[0].imag,
                    coherent[1].real,
                    coherent[1].imag,
                ]
            ),
            _pack_hermitian(raw_normal),
            _pack_symmetric(raw_anomalous),
            np.asarray(
                [part for value in raw_cross.reshape(-1) for part in (value.real, value.imag)]
            ),
        ]
    )
    if coordinates.shape != (len(RAW_MOMENT_COORDINATE_NAMES),):
        raise RuntimeError("raw-moment coordinate packing has the wrong size")
    return np.asarray(coordinates, dtype=float)


def raw_moment_coordinates_to_matrix_state(
    coordinates: FloatArray,
) -> MatrixDimerState:
    """Reconstruct ``(rho,B,N,A,C)`` from the 29 raw coordinates."""

    values = np.asarray(coordinates, dtype=float)
    expected = (len(RAW_MOMENT_COORDINATE_NAMES),)
    if values.shape != expected:
        raise ValueError(f"expected raw coordinate shape {expected}, got {values.shape}")
    bloch = values[:3]
    coherent = np.asarray(
        [values[3] + 1j * values[4], values[5] + 1j * values[6]],
        dtype=complex,
    )
    raw_normal = _unpack_hermitian(values[7:11])
    raw_anomalous = _unpack_symmetric(values[11:17])
    raw_cross_values = values[17:].reshape(2, 3, 2)
    raw_cross = raw_cross_values[..., 0] + 1j * raw_cross_values[..., 1]

    rho = 0.5 * (
        np.eye(2, dtype=complex)
        + sum(
            bloch[axis] * _PAULI_ARRAY[axis]
            for axis in range(3)
        )
    )
    normal = raw_normal - np.outer(coherent, coherent.conjugate())
    anomalous = raw_anomalous - np.outer(coherent, coherent)
    correlation = np.empty((2, 2, 2), dtype=complex)
    for mode in range(2):
        centered_coefficients = raw_cross[mode] - coherent[mode] * bloch
        correlation[mode] = 0.5 * sum(
            centered_coefficients[axis] * _PAULI_ARRAY[axis]
            for axis in range(3)
        )
    return MatrixDimerState(
        electron_density=np.asarray(rho, dtype=complex),
        coherent_phonon=coherent,
        phonon_density=np.asarray(normal, dtype=complex),
        anomalous_phonon_density=np.asarray(anomalous, dtype=complex),
        electron_phonon_correlation=correlation,
    )


def matrix_derivative_to_raw_moment_velocity(
    state: MatrixDimerState,
    derivative: MatrixDimerState,
) -> FloatArray:
    """Differentiate the nonlinear raw-moment chart at ``state``.

    ``Nbar``, ``Abar``, and the raw electron--phonon cross moments contain
    products of lower moments.  Their velocities therefore require both the
    current matrix tuple and its matrix-form velocity.
    """

    rho = np.asarray(state.electron_density, dtype=complex)
    coherent = np.asarray(state.coherent_phonon, dtype=complex)
    correlation = np.asarray(state.electron_phonon_correlation, dtype=complex)
    rho_velocity = np.asarray(derivative.electron_density, dtype=complex)
    coherent_velocity = np.asarray(
        derivative.coherent_phonon,
        dtype=complex,
    )
    normal_velocity = np.asarray(derivative.phonon_density, dtype=complex)
    anomalous_velocity = np.asarray(
        derivative.anomalous_phonon_density,
        dtype=complex,
    )
    correlation_velocity = np.asarray(
        derivative.electron_phonon_correlation,
        dtype=complex,
    )

    bloch = np.asarray(
        [np.trace(rho @ operator).real for operator in _PAULI_ARRAY],
        dtype=float,
    )
    bloch_velocity = np.asarray(
        [
            _require_real(
                np.trace(rho_velocity @ operator),
                name=f"Bloch velocity {axis}",
            )
            for axis, operator in enumerate(_PAULI_ARRAY)
        ],
        dtype=float,
    )
    raw_normal_velocity = (
        normal_velocity
        + np.outer(coherent_velocity, coherent.conjugate())
        + np.outer(coherent, coherent_velocity.conjugate())
    )
    raw_anomalous_velocity = (
        anomalous_velocity
        + np.outer(coherent_velocity, coherent)
        + np.outer(coherent, coherent_velocity)
    )
    raw_cross_velocity = np.empty((2, 3), dtype=complex)
    for mode in range(2):
        for axis, operator in enumerate(_PAULI_ARRAY):
            raw_cross_velocity[mode, axis] = (
                np.trace(correlation_velocity[mode] @ operator)
                + coherent_velocity[mode] * bloch[axis]
                + coherent[mode] * bloch_velocity[axis]
            )

    velocity = np.concatenate(
        [
            bloch_velocity,
            np.asarray(
                [
                    coherent_velocity[0].real,
                    coherent_velocity[0].imag,
                    coherent_velocity[1].real,
                    coherent_velocity[1].imag,
                ],
                dtype=float,
            ),
            _pack_hermitian(raw_normal_velocity),
            _pack_symmetric(raw_anomalous_velocity),
            np.asarray(
                [
                    part
                    for value in raw_cross_velocity.reshape(-1)
                    for part in (value.real, value.imag)
                ],
                dtype=float,
            ),
        ]
    )
    if velocity.shape != (len(RAW_MOMENT_COORDINATE_NAMES),):
        raise RuntimeError("raw-moment velocity packing has the wrong size")
    return np.asarray(velocity, dtype=float)


def raw_moment_velocity_to_matrix_derivative(
    state: MatrixDimerState,
    velocity: FloatArray,
) -> MatrixDimerState:
    """Apply the reconstruction differential to one raw velocity.

    This is the exact inverse of :func:`matrix_derivative_to_raw_moment_velocity`
    on the trace/Hermiticity/symmetry-consistent tangent space.  It realizes
    the Jacobian ``J(u)=D chi(u)`` used by the APCM moment metric without a
    finite-difference approximation.
    """

    values = np.asarray(velocity, dtype=float)
    expected = (len(RAW_MOMENT_COORDINATE_NAMES),)
    if values.shape != expected:
        raise ValueError(f"expected raw velocity shape {expected}, got {values.shape}")

    coherent = np.asarray(state.coherent_phonon, dtype=complex)
    bloch = np.asarray(
        [
            np.trace(state.electron_density @ operator).real
            for operator in _PAULI_ARRAY
        ],
        dtype=float,
    )
    bloch_velocity = values[:3]
    coherent_velocity = np.asarray(
        [values[3] + 1j * values[4], values[5] + 1j * values[6]],
        dtype=complex,
    )
    raw_normal_velocity = _unpack_hermitian(values[7:11])
    raw_anomalous_velocity = _unpack_symmetric(values[11:17])
    raw_cross_parts = values[17:].reshape(2, 3, 2)
    raw_cross_velocity = (
        raw_cross_parts[..., 0] + 1j * raw_cross_parts[..., 1]
    )

    electron_velocity = 0.5 * sum(
        bloch_velocity[axis] * _PAULI_ARRAY[axis]
        for axis in range(3)
    )
    normal_velocity = (
        raw_normal_velocity
        - np.outer(coherent_velocity, coherent.conjugate())
        - np.outer(coherent, coherent_velocity.conjugate())
    )
    anomalous_velocity = (
        raw_anomalous_velocity
        - np.outer(coherent_velocity, coherent)
        - np.outer(coherent, coherent_velocity)
    )
    centered_cross_velocity = (
        raw_cross_velocity
        - np.outer(coherent_velocity, bloch)
        - np.outer(coherent, bloch_velocity)
    )
    correlation_velocity = np.empty((2, 2, 2), dtype=complex)
    for mode in range(2):
        correlation_velocity[mode] = 0.5 * sum(
            centered_cross_velocity[mode, axis] * _PAULI_ARRAY[axis]
            for axis in range(3)
        )

    return MatrixDimerState(
        electron_density=np.asarray(electron_velocity, dtype=complex),
        coherent_phonon=coherent_velocity,
        phonon_density=np.asarray(normal_velocity, dtype=complex),
        anomalous_phonon_density=np.asarray(
            anomalous_velocity,
            dtype=complex,
        ),
        electron_phonon_correlation=correlation_velocity,
    )


def uncentered_joint_moment_matrix(coordinates: FloatArray) -> ComplexArray:
    """Return the affine 8-by-8 Gram matrix of ``(1,b,b^dagger,sigma)``."""

    values = np.asarray(coordinates, dtype=float)
    state = raw_moment_coordinates_to_matrix_state(values)
    bloch = values[:3]
    coherent = np.asarray(state.coherent_phonon, dtype=complex)
    raw_normal = _unpack_hermitian(values[7:11])
    raw_anomalous = _unpack_symmetric(values[11:17])
    raw_cross_values = values[17:].reshape(2, 3, 2)
    raw_cross = raw_cross_values[..., 0] + 1j * raw_cross_values[..., 1]

    boson = np.block(
        [
            [raw_normal.T, raw_anomalous.conjugate()],
            [raw_anomalous, np.eye(2, dtype=complex) + raw_normal],
        ]
    )
    cross = np.vstack([raw_cross.conjugate(), raw_cross])
    electronic = np.empty((3, 3), dtype=complex)
    for row, first in enumerate(_PAULI_ARRAY):
        for column, second in enumerate(_PAULI_ARRAY):
            electronic[row, column] = np.trace(
                state.electron_density @ first @ second
            )
    means = np.concatenate(
        [coherent, coherent.conjugate(), bloch.astype(complex)]
    )
    lower = np.block([[boson, cross], [cross.conjugate().T, electronic]])
    result = np.block(
        [
            [np.ones((1, 1), dtype=complex), means[None, :]],
            [means.conjugate()[:, None], lower],
        ]
    )
    return 0.5 * (result + result.conjugate().T)


def raw_moment_schur_complement(coordinates: FloatArray) -> ComplexArray:
    """Return the centered seven-by-seven Gram Schur complement."""

    matrix = uncentered_joint_moment_matrix(coordinates)
    return matrix[1:, 1:] - np.outer(matrix[1:, 0], matrix[0, 1:])


def _pauli_coefficients(operator: ComplexArray) -> dict[str, complex]:
    value = np.asarray(operator, dtype=complex)
    if value.shape != (2, 2):
        raise ValueError("electronic operator must be 2 by 2")
    return {
        label: 0.5 * complex(np.trace(value @ basis))
        for label, basis in _PAULI.items()
    }


def _moment_value(
    moments: Mapping[MomentKey, float],
    spin_up: str,
    spin_down: str,
    x_power: int,
    p_power: int,
) -> float:
    if (
        spin_up == IDENTITY
        and spin_down == IDENTITY
        and x_power == 0
        and p_power == 0
    ):
        return 1.0
    labels = (spin_up, spin_down)
    if (IDENTITY, PAULI_X, PAULI_Y, PAULI_Z).index(labels[0]) > (
        IDENTITY,
        PAULI_X,
        PAULI_Y,
        PAULI_Z,
    ).index(labels[1]):
        labels = (labels[1], labels[0])
    key = MomentKey(labels[0], labels[1], x_power, p_power)
    try:
        return float(moments[key])
    except KeyError as error:
        raise ValueError(f"required relative-mode moment is missing: {key}") from error


def spin_boson_expectation(
    moments: Mapping[MomentKey, float],
    up_operator: ComplexArray,
    down_operator: ComplexArray | None = None,
    *,
    x_power: int = 0,
    p_power: int = 0,
) -> complex:
    """Evaluate one Pauli--Weyl product from the real moment dictionary."""

    up_coefficients = _pauli_coefficients(up_operator)
    down_coefficients = _pauli_coefficients(
        np.eye(2, dtype=complex) if down_operator is None else down_operator
    )
    value = 0.0j
    for up_label, up_coefficient in up_coefficients.items():
        if abs(up_coefficient) <= 1e-15:
            continue
        for down_label, down_coefficient in down_coefficients.items():
            if abs(down_coefficient) <= 1e-15:
                continue
            value += (
                up_coefficient
                * down_coefficient
                * _moment_value(
                    moments,
                    up_label,
                    down_label,
                    x_power,
                    p_power,
                )
            )
    return complex(value)


def _one_body_operator(i: int, j: int) -> ComplexArray:
    operator = np.zeros((2, 2), dtype=complex)
    operator[j, i] = 1.0
    return operator


def _site_occupation(site: int) -> ComplexArray:
    operator = np.zeros((2, 2), dtype=complex)
    operator[site, site] = 1.0
    return operator


def connected_k_from_relative_moments(
    moments: Mapping[MomentKey, float],
) -> ComplexArray:
    """Decode ``K[q,r,i,j]`` from spin-symmetric moments through degree three.

    The decoupled center mode is coherent and factorizes from the interacting
    relative mode.  Its centered contribution cancels in the connected
    covariance, leaving the relative-mode factor ``s_q s_r / 2``.
    """

    mean_x = _moment_value(moments, IDENTITY, IDENTITY, 1, 0)
    mean_p = _moment_value(moments, IDENTITY, IDENTITY, 0, 1)
    mean_x2 = _moment_value(moments, IDENTITY, IDENTITY, 2, 0)
    mean_xp_weyl = _moment_value(moments, IDENTITY, IDENTITY, 1, 1)
    centered_x2 = mean_x2 - mean_x**2
    centered_xp_ordered = (
        mean_xp_weyl - mean_x * mean_p + 0.5j
    )
    centered_pair = centered_x2 + 1j * centered_xp_ordered

    result = np.empty((2, 2, 2, 2), dtype=complex)
    for q in range(2):
        for site in range(2):
            occupation = _site_occupation(site)
            factor = 0.5 * _SITE_SIGNS[q] * _SITE_SIGNS[site]
            for i in range(2):
                for j in range(2):
                    one_body = _one_body_operator(i, j)
                    commutator = one_body @ occupation - occupation @ one_body
                    mean_j = spin_boson_expectation(moments, commutator)
                    mean_jx = spin_boson_expectation(
                        moments, commutator, x_power=1
                    )
                    mean_jp = spin_boson_expectation(
                        moments, commutator, p_power=1
                    )
                    mean_jx2 = spin_boson_expectation(
                        moments, commutator, x_power=2
                    )
                    mean_jxp_ordered = (
                        spin_boson_expectation(
                            moments,
                            commutator,
                            x_power=1,
                            p_power=1,
                        )
                        + 0.5j * mean_j
                    )
                    centered_jx2 = (
                        mean_jx2 - 2.0 * mean_x * mean_jx + mean_x**2 * mean_j
                    )
                    centered_jxp = (
                        mean_jxp_ordered
                        - mean_p * mean_jx
                        - mean_x * mean_jp
                        + mean_x * mean_p * mean_j
                    )
                    result[q, site, i, j] = factor * (
                        centered_jx2
                        + 1j * centered_jxp
                        - mean_j * centered_pair
                    )
    return result


def opposite_spin_covariance_from_relative_moments(
    moments: Mapping[MomentKey, float],
) -> ComplexArray:
    """Decode ``D[q,i,j]=Cov(O_ij,n_{q down})`` from electronic moments."""

    result = np.empty((2, 2, 2), dtype=complex)
    for q in range(2):
        occupation = _site_occupation(q)
        occupation_mean = spin_boson_expectation(
            moments,
            np.eye(2, dtype=complex),
            occupation,
        )
        for i in range(2):
            for j in range(2):
                one_body = _one_body_operator(i, j)
                one_body_mean = spin_boson_expectation(moments, one_body)
                joint = spin_boson_expectation(moments, one_body, occupation)
                result[q, i, j] = joint - one_body_mean * occupation_mean
    return result


def relative_moments_from_matrix_state(
    state: MatrixDimerState,
    hidden_moments: Mapping[MomentKey, float],
    *,
    required_keys: tuple[MomentKey, ...] | None = None,
) -> tuple[complex, dict[MomentKey, float]]:
    """Reconstruct selected relative moments from ``X`` and raw auxiliaries.

    The archive tuple determines the one-spin, pure-boson, and one-spin--boson
    moments through degree two.  Every other requested key must be supplied by
    ``hidden_moments``.  Omitting ``required_keys`` preserves the complete
    degree-three reconstruction used by the earlier fixed-dictionary audit.
    """

    keys = (
        tuple(THIRD_ORDER_HIERARCHY.moment_keys)
        if required_keys is None
        else tuple(required_keys)
    )
    hierarchy_keys = set(THIRD_ORDER_HIERARCHY.moment_keys)
    unknown = set(keys).difference(hierarchy_keys)
    if unknown:
        raise ValueError(f"requested moments are outside the degree-three chart: {unknown}")
    expected_hidden = {
        key for key in keys if not _is_archive_resolved_relative_moment(key)
    }
    missing_hidden = expected_hidden.difference(hidden_moments)
    extra_hidden = set(hidden_moments).difference(expected_hidden)
    if missing_hidden or extra_hidden:
        raise ValueError(
            "hidden moment mapping mismatch: "
            f"{len(missing_hidden)} missing, {len(extra_hidden)} extra"
        )

    rho = np.asarray(state.electron_density, dtype=complex)
    coherent = np.asarray(state.coherent_phonon, dtype=complex)
    normal = np.asarray(state.phonon_density, dtype=complex)
    anomalous = np.asarray(state.anomalous_phonon_density, dtype=complex)
    correlation = np.asarray(state.electron_phonon_correlation, dtype=complex)

    center = complex(np.sum(coherent) / np.sqrt(2.0))
    relative_amplitude = complex(np.vdot(_RELATIVE_MODE, coherent))
    mean_x = np.sqrt(2.0) * relative_amplitude.real
    mean_p = np.sqrt(2.0) * relative_amplitude.imag
    relative_population = complex(
        np.vdot(_RELATIVE_MODE, normal @ _RELATIVE_MODE)
    )
    relative_anomalous = complex(
        _RELATIVE_MODE.T @ anomalous @ _RELATIVE_MODE
    )
    population = _require_real(relative_population, name="relative population")
    covariance_xx = population + 0.5 + relative_anomalous.real
    covariance_pp = population + 0.5 - relative_anomalous.real
    covariance_xp = relative_anomalous.imag
    raw_xx = covariance_xx + mean_x**2
    raw_pp = covariance_pp + mean_p**2
    raw_xp = covariance_xp + mean_x * mean_p

    bloch = {
        label: _require_real(
            np.trace(rho @ _PAULI[label]),
            name=f"Bloch {label}",
        )
        for label in _PAULI_LABELS
    }
    relative_correlation = 0.5 * (correlation[0] - correlation[1])
    covariances: dict[str, tuple[float, float]] = {}
    c00 = relative_correlation[0, 0]
    c01 = relative_correlation[0, 1]
    c10 = relative_correlation[1, 0]
    covariances[PAULI_Z] = (4.0 * c00.real, 4.0 * c00.imag)
    covariances[PAULI_X] = (
        2.0 * (c01 + c10).real,
        2.0 * (c01 + c10).imag,
    )
    covariances[PAULI_Y] = (
        2.0 * (c10 - c01).imag,
        2.0 * (c01 - c10).real,
    )

    moments: dict[MomentKey, float] = {}
    for key in keys:
        if key in hidden_moments:
            moments[key] = float(hidden_moments[key])
            continue
        spin_labels = tuple(
            label
            for label in (key.spin_up, key.spin_down)
            if label != IDENTITY
        )
        boson_degree = key.x_power + key.p_power
        if not spin_labels:
            if (key.x_power, key.p_power) == (1, 0):
                value = mean_x
            elif (key.x_power, key.p_power) == (0, 1):
                value = mean_p
            elif (key.x_power, key.p_power) == (2, 0):
                value = raw_xx
            elif (key.x_power, key.p_power) == (1, 1):
                value = raw_xp
            elif (key.x_power, key.p_power) == (0, 2):
                value = raw_pp
            else:  # pragma: no cover - guarded by the hidden-key partition
                raise ValueError(f"unsupported retained boson key {key}")
        elif len(spin_labels) == 1 and boson_degree == 0:
            value = bloch[spin_labels[0]]
        elif len(spin_labels) == 1 and boson_degree == 1:
            label = spin_labels[0]
            if key.x_power == 1:
                value = covariances[label][0] + bloch[label] * mean_x
            else:
                value = covariances[label][1] + bloch[label] * mean_p
        else:  # pragma: no cover - guarded by the hidden-key partition
            raise ValueError(f"moment {key} is neither archive-resolved nor hidden")
        moments[key] = float(value)
    return center, moments


def kpd_correlation_velocity_correction(
    state: MatrixDimerState,
    parameters: DimerParameters,
    moments: Mapping[MomentKey, float],
    *,
    include_k: bool = True,
    include_pauli: bool = True,
    include_opposite_spin: bool = True,
) -> ComplexArray:
    """Return the verified additive ``K/P/D`` correction to ``dot C``."""

    correction = np.zeros((2, 2, 2), dtype=complex)
    if include_k:
        correction += -1j * parameters.coupling * np.sum(
            connected_k_from_relative_moments(moments),
            axis=1,
        )
    if include_pauli:
        correction += same_spin_pauli_velocity_correction(state, parameters)
    if include_opposite_spin:
        correction += (
            -1j
            * parameters.coupling
            * opposite_spin_covariance_from_relative_moments(moments)
        )
    return correction


__all__ = [
    "ARCHIVE_RELATIVE_MOMENT_KEYS",
    "ENTRANCE_RELATIVE_MOMENT_KEYS",
    "HIDDEN_RELATIVE_MOMENT_KEYS",
    "INITIAL_PROMOTION_CANDIDATE_KEYS",
    "RAW_MOMENT_COORDINATE_NAMES",
    "connected_k_from_relative_moments",
    "kpd_correlation_velocity_correction",
    "matrix_state_to_raw_moment_coordinates",
    "matrix_derivative_to_raw_moment_velocity",
    "raw_moment_velocity_to_matrix_derivative",
    "opposite_spin_covariance_from_relative_moments",
    "relative_moments_from_matrix_state",
    "raw_moment_coordinates_to_matrix_state",
    "raw_moment_schur_complement",
    "spin_boson_expectation",
    "uncentered_joint_moment_matrix",
]
