"""Offline archive-Gram and mixed-tangent diagnostics for packet states.

The stored multi-coherent trajectory contains the interacting relative phonon
mode.  This module restores a centered center-mode vacuum, enforces the dimer's
spin-exchange symmetry, and compares several local real tangent spaces against
the same-state Schrodinger velocity.  It never supplies exact-reference data
to an online right-hand side and does not propagate a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import sqrt

import numpy as np
from numpy.typing import NDArray

from .hubbard_dimer import DimerParameters, GaussianSineDrive
from .matrix_reference import (
    MatrixDimerState,
    electron_phonon_moment_matrix,
    matrix_derivative_to_closed_scalar,
)
from .multi_coherent import (
    _oscillator_operators,
    multi_coherent_state_and_tangent,
    relative_holstein_hamiltonian,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

ARCHIVE_LABELS = (
    "delta_b0",
    "delta_b1",
    "delta_b0_dagger",
    "delta_b1_dagger",
    "delta_sigma_x",
    "delta_sigma_y",
    "delta_sigma_z",
)
LOCAL_MIXED_LABELS = tuple(
    f"delta_b{phonon}{suffix}_delta_sigma_{pauli}"
    for phonon in range(2)
    for suffix in ("", "_dagger")
    for pauli in ("x", "y", "z")
)
MIXED_LABELS = tuple(
    f"delta_a_relative{suffix}_delta_sigma_{pauli}"
    for suffix in ("", "_dagger")
    for pauli in ("x", "y", "z")
)
SPACE_NAMES = (
    "archive",
    "archive_relative_mixed",
    "archive_mixed",
    "packet_geometric",
    "packet_tikhonov",
    "packet_archive",
    "packet_relative_mixed",
    "packet_mixed",
)


@dataclass(frozen=True)
class TangentProjection:
    """Projection of one complex target onto a complex-by-real frame."""

    projected_velocity: ComplexArray
    absolute_residual: float
    relative_residual: float
    retained_rank: int
    geometric_rank: int
    largest_singular_value: float
    smallest_retained_singular_value: float
    coefficient_norm: float
    real_range_basis: FloatArray


@dataclass(frozen=True)
class ArchiveMixedFrames:
    """Packet, archive, and mixed tangents at one symmetrized packet state."""

    relative_state: ComplexArray
    full_state: ComplexArray
    packet_tangent: ComplexArray
    archive_tangent: ComplexArray
    relative_mixed_tangent: ComplexArray
    mixed_tangent: ComplexArray
    archive_vectors: ComplexArray
    relative_mixed_vectors: ComplexArray
    mixed_vectors: ComplexArray
    matrix_state: MatrixDimerState
    spin_swap_fidelity: float
    symmetry_projection_fidelity: float
    relative_top_population: float
    archive_gram_max_error: float


@dataclass(frozen=True)
class ArchiveMixedPilotPoint:
    """All tangent-space scores at one stored packet state and time."""

    space_names: tuple[str, ...]
    hilbert_relative_residual: FloatArray
    closed_coordinate_scaled_rms: FloatArray
    correlation_scaled_rms: FloatArray
    retained_rank: NDArray[np.int64]
    geometric_rank: NDArray[np.int64]
    target_velocity_norm: float
    spin_swap_fidelity: float
    symmetry_projection_fidelity: float
    relative_top_population: float
    archive_gram_max_error: float
    archive_novelty_fraction: float
    mixed_novelty_fraction: float
    full_mixed_novelty_fraction: float
    archive_novelty_eigenvalues: FloatArray
    mixed_novelty_eigenvalues: FloatArray
    full_mixed_novelty_eigenvalues: FloatArray
    mixed_candidate_residual_reduction: FloatArray


@dataclass(frozen=True)
class _FullOperators:
    identity: ComplexArray
    local_boson: tuple[ComplexArray, ComplexArray]
    one_body: ComplexArray
    pauli: tuple[ComplexArray, ComplexArray, ComplexArray]
    relative_swap: ComplexArray


def _realify_vector(vector: ComplexArray) -> FloatArray:
    return np.concatenate((vector.real, vector.imag))


def _complexify_vector(vector: FloatArray) -> ComplexArray:
    size = vector.size // 2
    return vector[:size] + 1j * vector[size:]


def _realify_frame(frame: ComplexArray) -> FloatArray:
    return np.vstack((frame.real, frame.imag))


def _complex_coefficient_frame(vectors: ComplexArray) -> ComplexArray:
    return np.column_stack((vectors, 1j * vectors))


def project_real_tangent(
    target_velocity: ComplexArray,
    tangent: ComplexArray,
    *,
    geometric_gram_relative_threshold: float = 1e-10,
    relative_damping: float | None = None,
) -> TangentProjection:
    """Project with either a geometric pseudoinverse or Tikhonov filter."""

    target = np.asarray(target_velocity, dtype=complex)
    frame = np.asarray(tangent, dtype=complex)
    if target.ndim != 1 or frame.ndim != 2 or frame.shape[0] != target.size:
        raise ValueError("target and tangent shapes are incompatible")
    if frame.shape[1] < 1:
        raise ValueError("tangent must contain at least one column")
    if not 0.0 < geometric_gram_relative_threshold < 1.0:
        raise ValueError("geometric threshold must lie between zero and one")
    if relative_damping is not None and relative_damping <= 0.0:
        raise ValueError("relative_damping must be positive")

    real_frame = _realify_frame(frame)
    real_target = _realify_vector(target)
    left, singular_values, right_adjoint = np.linalg.svd(
        real_frame,
        full_matrices=False,
    )
    largest = float(singular_values[0])
    geometric_threshold = (
        sqrt(geometric_gram_relative_threshold) * largest
    )
    geometric_mask = singular_values > geometric_threshold
    geometric_rank = int(np.count_nonzero(geometric_mask))
    range_basis = left[:, geometric_mask]

    if relative_damping is None:
        inverse = np.zeros_like(singular_values)
        inverse[geometric_mask] = 1.0 / singular_values[geometric_mask]
        retained_mask = geometric_mask
    else:
        damping = relative_damping * largest
        inverse = singular_values / (singular_values**2 + damping**2)
        retained_mask = singular_values > damping

    coefficients = right_adjoint.T @ (
        inverse * (left.T @ real_target)
    )
    projected = frame @ coefficients
    residual = projected - target
    retained = singular_values[retained_mask]
    smallest = float(retained[-1]) if retained.size else 0.0
    target_norm = float(np.linalg.norm(target))
    return TangentProjection(
        projected_velocity=np.asarray(projected, dtype=complex),
        absolute_residual=float(np.linalg.norm(residual)),
        relative_residual=float(
            np.linalg.norm(residual)
            / max(target_norm, np.finfo(float).tiny)
        ),
        retained_rank=int(np.count_nonzero(retained_mask)),
        geometric_rank=geometric_rank,
        largest_singular_value=largest,
        smallest_retained_singular_value=smallest,
        coefficient_norm=float(np.linalg.norm(coefficients)),
        real_range_basis=np.asarray(range_basis, dtype=float),
    )


@lru_cache(maxsize=None)
def _full_operators(relative_dimension: int) -> _FullOperators:
    if relative_dimension < 2:
        raise ValueError("relative_dimension must be at least two")
    center_dimension = 2
    center_annihilation, _ = _oscillator_operators(center_dimension)
    relative_annihilation, _ = _oscillator_operators(relative_dimension)
    center_identity = np.eye(center_dimension, dtype=complex)
    electron_identity = np.eye(4, dtype=complex)
    relative_identity = np.eye(relative_dimension, dtype=complex)
    full_identity = np.eye(
        center_dimension * 4 * relative_dimension,
        dtype=complex,
    )

    center_operator = np.kron(
        np.kron(center_annihilation, electron_identity),
        relative_identity,
    )
    relative_operator = np.kron(
        center_identity,
        np.kron(electron_identity, relative_annihilation),
    )
    local_boson = (
        (center_operator + relative_operator) / sqrt(2.0),
        (center_operator - relative_operator) / sqrt(2.0),
    )

    site_identity = np.eye(2, dtype=complex)
    pauli_small = (
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    )
    pauli = tuple(
        np.kron(
            center_identity,
            np.kron(np.kron(operator, site_identity), relative_identity),
        )
        for operator in pauli_small
    )
    one_body = np.empty(
        (2, 2, full_identity.shape[0], full_identity.shape[1]),
        dtype=complex,
    )
    for row in range(2):
        for column in range(2):
            operator = np.zeros((2, 2), dtype=complex)
            operator[column, row] = 1.0
            one_body[row, column] = np.kron(
                center_identity,
                np.kron(
                    np.kron(operator, site_identity),
                    relative_identity,
                ),
            )

    electron_swap = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=complex,
    )
    relative_swap = np.kron(electron_swap, relative_identity)
    return _FullOperators(
        identity=full_identity,
        local_boson=local_boson,
        one_body=one_body,
        pauli=pauli,
        relative_swap=relative_swap,
    )


def _expectation(state: ComplexArray, operator: ComplexArray) -> complex:
    return complex(np.vdot(state, operator @ state))


def _expectation_product(
    state: ComplexArray,
    first: ComplexArray,
    second: ComplexArray,
) -> complex:
    return complex(np.vdot(state, first @ (second @ state)))


def _expectation_derivative(
    state: ComplexArray,
    derivative: ComplexArray,
    operator: ComplexArray,
) -> complex:
    return complex(
        np.vdot(derivative, operator @ state)
        + np.vdot(state, operator @ derivative)
    )


def _expectation_product_derivative(
    state: ComplexArray,
    derivative: ComplexArray,
    first: ComplexArray,
    second: ComplexArray,
) -> complex:
    return complex(
        np.vdot(derivative, first @ (second @ state))
        + np.vdot(state, first @ (second @ derivative))
    )


def _centered_blocks(
    state: ComplexArray,
    operators: _FullOperators,
) -> tuple[
    MatrixDimerState,
    tuple[ComplexArray, ComplexArray],
    tuple[ComplexArray, ComplexArray, ComplexArray],
    ComplexArray,
]:
    identity = operators.identity
    coherent = np.asarray(
        [_expectation(state, operator) for operator in operators.local_boson],
        dtype=complex,
    )
    delta_b = tuple(
        operator - coherent[index] * identity
        for index, operator in enumerate(operators.local_boson)
    )
    electron = np.empty((2, 2), dtype=complex)
    for row in range(2):
        for column in range(2):
            electron[row, column] = _expectation(
                state,
                operators.one_body[row, column],
            )
    delta_one_body = np.empty_like(operators.one_body)
    for row in range(2):
        for column in range(2):
            delta_one_body[row, column] = (
                operators.one_body[row, column]
                - electron[row, column] * identity
            )

    pauli_means = np.asarray(
        [_expectation(state, operator).real for operator in operators.pauli]
    )
    delta_pauli = tuple(
        operator - pauli_means[index] * identity
        for index, operator in enumerate(operators.pauli)
    )
    normal = np.empty((2, 2), dtype=complex)
    anomalous = np.empty((2, 2), dtype=complex)
    correlation = np.empty((2, 2, 2), dtype=complex)
    for phonon in range(2):
        for other in range(2):
            normal[phonon, other] = _expectation_product(
                state,
                delta_b[other].conjugate().T,
                delta_b[phonon],
            )
            anomalous[phonon, other] = _expectation_product(
                state,
                delta_b[phonon],
                delta_b[other],
            )
        for row in range(2):
            for column in range(2):
                correlation[phonon, row, column] = _expectation_product(
                    state,
                    delta_b[phonon],
                    delta_one_body[row, column],
                )
    matrix_state = MatrixDimerState(
        electron_density=electron,
        coherent_phonon=coherent,
        phonon_density=normal,
        anomalous_phonon_density=anomalous,
        electron_phonon_correlation=correlation,
    )
    return matrix_state, delta_b, delta_pauli, delta_one_body


def full_state_matrix_derivative(
    state: ComplexArray,
    state_derivative: ComplexArray,
    *,
    relative_dimension: int,
) -> MatrixDimerState:
    """Contract one full center--electron--relative tangent into matrix rates."""

    vector = np.asarray(state, dtype=complex)
    derivative = np.asarray(state_derivative, dtype=complex)
    operators = _full_operators(relative_dimension)
    if vector.shape != (operators.identity.shape[0],):
        raise ValueError("state has incompatible dimension")
    if derivative.shape != vector.shape:
        raise ValueError("state_derivative must match state")
    norm = float(np.linalg.norm(vector))
    vector = vector / norm
    derivative = derivative / norm
    derivative = derivative - vector * np.vdot(vector, derivative)
    _, delta_b, _, delta_one_body = _centered_blocks(vector, operators)

    electron = np.empty((2, 2), dtype=complex)
    coherent = np.empty(2, dtype=complex)
    normal = np.empty((2, 2), dtype=complex)
    anomalous = np.empty((2, 2), dtype=complex)
    correlation = np.empty((2, 2, 2), dtype=complex)
    for row in range(2):
        for column in range(2):
            electron[row, column] = _expectation_derivative(
                vector,
                derivative,
                operators.one_body[row, column],
            )
    for phonon in range(2):
        coherent[phonon] = _expectation_derivative(
            vector,
            derivative,
            operators.local_boson[phonon],
        )
        for other in range(2):
            normal[phonon, other] = _expectation_product_derivative(
                vector,
                derivative,
                delta_b[other].conjugate().T,
                delta_b[phonon],
            )
            anomalous[phonon, other] = _expectation_product_derivative(
                vector,
                derivative,
                delta_b[phonon],
                delta_b[other],
            )
        for row in range(2):
            for column in range(2):
                correlation[phonon, row, column] = (
                    _expectation_product_derivative(
                        vector,
                        derivative,
                        delta_b[phonon],
                        delta_one_body[row, column],
                    )
                )
    return MatrixDimerState(
        electron_density=electron,
        coherent_phonon=coherent,
        phonon_density=normal,
        anomalous_phonon_density=anomalous,
        electron_phonon_correlation=correlation,
    )


def packet_archive_mixed_frames(
    packet_parameters: FloatArray,
    *,
    relative_dimension: int,
) -> ArchiveMixedFrames:
    """Construct the three tangent frames at one packet parameter vector."""

    raw_state, raw_tangent = multi_coherent_state_and_tangent(
        np.asarray(packet_parameters, dtype=float),
        relative_dimension=relative_dimension,
    )
    raw_state = raw_state / np.linalg.norm(raw_state)
    operators = _full_operators(relative_dimension)
    swapped = operators.relative_swap @ raw_state
    spin_swap_fidelity = float(abs(np.vdot(raw_state, swapped)) ** 2)

    unnormalized_state, unnormalized_tangent = (
        multi_coherent_state_and_tangent(
            np.asarray(packet_parameters, dtype=float),
            relative_dimension=relative_dimension,
        )
    )
    projected_state = 0.5 * (
        unnormalized_state + operators.relative_swap @ unnormalized_state
    )
    projected_tangent = 0.5 * (
        unnormalized_tangent + operators.relative_swap @ unnormalized_tangent
    )
    projected_norm = float(np.linalg.norm(projected_state))
    if projected_norm <= np.finfo(float).tiny:
        raise ValueError("spin-symmetric projection has zero norm")
    relative_state = projected_state / projected_norm
    overlaps = relative_state.conjugate() @ projected_tangent
    relative_tangent = (
        projected_tangent
        - relative_state[:, np.newaxis] * overlaps[np.newaxis, :]
    ) / projected_norm
    symmetry_projection_fidelity = float(
        abs(np.vdot(raw_state, relative_state)) ** 2
    )

    full_state = np.concatenate(
        (relative_state, np.zeros_like(relative_state))
    )
    packet_tangent = np.vstack(
        (relative_tangent, np.zeros_like(relative_tangent))
    )
    matrix_state, delta_b, delta_pauli, _ = _centered_blocks(
        full_state,
        operators,
    )
    archive_vectors = np.column_stack(
        (
            delta_b[0] @ full_state,
            delta_b[1] @ full_state,
            delta_b[0].conjugate().T @ full_state,
            delta_b[1].conjugate().T @ full_state,
            *(operator @ full_state for operator in delta_pauli),
        )
    )
    mixed_vectors: list[ComplexArray] = []
    for phonon in range(2):
        for boson_operator in (
            delta_b[phonon],
            delta_b[phonon].conjugate().T,
        ):
            for pauli_operator in delta_pauli:
                value = boson_operator @ (pauli_operator @ full_state)
                value -= full_state * np.vdot(full_state, value)
                mixed_vectors.append(value)
    mixed_array = np.column_stack(mixed_vectors)
    relative_mixed_array = np.column_stack(
        (
            *(
                (mixed_array[:, index] - mixed_array[:, 6 + index])
                / sqrt(2.0)
                for index in range(3)
            ),
            *(
                (mixed_array[:, 3 + index] - mixed_array[:, 9 + index])
                / sqrt(2.0)
                for index in range(3)
            ),
        )
    )
    archive_tangent = _complex_coefficient_frame(archive_vectors)
    relative_mixed_tangent = _complex_coefficient_frame(
        relative_mixed_array
    )
    mixed_tangent = _complex_coefficient_frame(mixed_array)
    operator_gram = archive_vectors.conjugate().T @ archive_vectors
    moment_gram = electron_phonon_moment_matrix(matrix_state)
    return ArchiveMixedFrames(
        relative_state=relative_state,
        full_state=full_state,
        packet_tangent=packet_tangent,
        archive_tangent=archive_tangent,
        relative_mixed_tangent=relative_mixed_tangent,
        mixed_tangent=mixed_tangent,
        archive_vectors=archive_vectors,
        relative_mixed_vectors=relative_mixed_array,
        mixed_vectors=mixed_array,
        matrix_state=matrix_state,
        spin_swap_fidelity=spin_swap_fidelity,
        symmetry_projection_fidelity=symmetry_projection_fidelity,
        relative_top_population=float(
            np.sum(
                np.abs(
                    relative_state.reshape(4, relative_dimension)[:, -1]
                )
                ** 2
            )
        ),
        archive_gram_max_error=float(
            np.max(np.abs(operator_gram - moment_gram))
        ),
    )


def _novelty_spectrum(
    observer_tangent: ComplexArray,
    packet_range_basis: FloatArray,
) -> tuple[float, FloatArray]:
    frame = _realify_frame(observer_tangent)
    residual = frame - packet_range_basis @ (
        packet_range_basis.T @ frame
    )
    denominator = float(np.sum(frame * frame))
    fraction = float(
        np.sum(residual * residual)
        / max(denominator, np.finfo(float).tiny)
    )
    singular_values = np.linalg.svd(residual, compute_uv=False)
    return fraction, singular_values**2


def _candidate_scores(
    target_velocity: ComplexArray,
    archive_projection: TangentProjection,
    mixed_vectors: ComplexArray,
    *,
    geometric_gram_relative_threshold: float,
) -> FloatArray:
    target = _realify_vector(target_velocity)
    archive_basis = archive_projection.real_range_basis
    residual = target - archive_basis @ (archive_basis.T @ target)
    denominator = float(np.dot(residual, residual))
    scores = np.empty(mixed_vectors.shape[1], dtype=float)
    for index in range(mixed_vectors.shape[1]):
        candidate = _realify_frame(
            _complex_coefficient_frame(mixed_vectors[:, index : index + 1])
        )
        candidate -= archive_basis @ (archive_basis.T @ candidate)
        left, singular_values, _ = np.linalg.svd(
            candidate,
            full_matrices=False,
        )
        threshold = (
            sqrt(geometric_gram_relative_threshold) * singular_values[0]
        )
        basis = left[:, singular_values > threshold]
        captured = basis.T @ residual
        scores[index] = float(
            np.dot(captured, captured)
            / max(denominator, np.finfo(float).tiny)
        )
    return scores


def archive_mixed_tangent_pilot_point(
    packet_parameters: FloatArray,
    *,
    time: float,
    parameters: DimerParameters,
    drive_protocol: GaussianSineDrive,
    relative_dimension: int,
    coordinate_scales: FloatArray,
    geometric_gram_relative_threshold: float = 1e-10,
    relative_damping: float = 3e-4,
) -> ArchiveMixedPilotPoint:
    """Compare archive, mixed, packet, and augmented tangent projections."""

    scales = np.asarray(coordinate_scales, dtype=float)
    if scales.shape != (31,) or np.any(scales <= 0.0):
        raise ValueError("coordinate_scales must be positive with shape (31,)")
    frames = packet_archive_mixed_frames(
        packet_parameters,
        relative_dimension=relative_dimension,
    )
    hamiltonian = relative_holstein_hamiltonian(
        float(time),
        parameters,
        relative_dimension=relative_dimension,
        drive_protocol=drive_protocol,
    )
    energy = float(
        np.vdot(
            frames.relative_state,
            hamiltonian @ frames.relative_state,
        ).real
    )
    relative_target = -1j * (
        hamiltonian @ frames.relative_state
        - energy * frames.relative_state
    )
    full_target = np.concatenate(
        (relative_target, np.zeros_like(relative_target))
    )
    archive_mixed = np.column_stack(
        (frames.archive_tangent, frames.mixed_tangent)
    )
    archive_relative_mixed = np.column_stack(
        (frames.archive_tangent, frames.relative_mixed_tangent)
    )
    packet_archive = np.column_stack(
        (frames.packet_tangent, frames.archive_tangent)
    )
    packet_mixed = np.column_stack(
        (
            frames.packet_tangent,
            frames.archive_tangent,
            frames.mixed_tangent,
        )
    )
    packet_relative_mixed = np.column_stack(
        (
            frames.packet_tangent,
            frames.archive_tangent,
            frames.relative_mixed_tangent,
        )
    )
    projections = (
        project_real_tangent(
            full_target,
            frames.archive_tangent,
            geometric_gram_relative_threshold=(
                geometric_gram_relative_threshold
            ),
        ),
        project_real_tangent(
            full_target,
            archive_relative_mixed,
            geometric_gram_relative_threshold=(
                geometric_gram_relative_threshold
            ),
        ),
        project_real_tangent(
            full_target,
            archive_mixed,
            geometric_gram_relative_threshold=(
                geometric_gram_relative_threshold
            ),
        ),
        project_real_tangent(
            full_target,
            frames.packet_tangent,
            geometric_gram_relative_threshold=(
                geometric_gram_relative_threshold
            ),
        ),
        project_real_tangent(
            full_target,
            frames.packet_tangent,
            geometric_gram_relative_threshold=(
                geometric_gram_relative_threshold
            ),
            relative_damping=relative_damping,
        ),
        project_real_tangent(
            full_target,
            packet_archive,
            geometric_gram_relative_threshold=(
                geometric_gram_relative_threshold
            ),
        ),
        project_real_tangent(
            full_target,
            packet_relative_mixed,
            geometric_gram_relative_threshold=(
                geometric_gram_relative_threshold
            ),
        ),
        project_real_tangent(
            full_target,
            packet_mixed,
            geometric_gram_relative_threshold=(
                geometric_gram_relative_threshold
            ),
        ),
    )
    exact_closed_velocity = matrix_derivative_to_closed_scalar(
        full_state_matrix_derivative(
            frames.full_state,
            full_target,
            relative_dimension=relative_dimension,
        )
    )
    closed_errors = np.empty(len(projections), dtype=float)
    correlation_errors = np.empty(len(projections), dtype=float)
    for index, projection in enumerate(projections):
        candidate = matrix_derivative_to_closed_scalar(
            full_state_matrix_derivative(
                frames.full_state,
                projection.projected_velocity,
                relative_dimension=relative_dimension,
            )
        )
        difference = (candidate - exact_closed_velocity) / scales
        closed_errors[index] = float(np.sqrt(np.mean(difference**2)))
        correlation_errors[index] = float(
            np.sqrt(np.mean(difference[17:31] ** 2))
        )

    packet_geometric = projections[3]
    archive_novelty, archive_spectrum = _novelty_spectrum(
        frames.archive_tangent,
        packet_geometric.real_range_basis,
    )
    mixed_novelty, mixed_spectrum = _novelty_spectrum(
        frames.relative_mixed_tangent,
        packet_geometric.real_range_basis,
    )
    full_mixed_novelty, full_mixed_spectrum = _novelty_spectrum(
        frames.mixed_tangent,
        packet_geometric.real_range_basis,
    )
    candidate_scores = _candidate_scores(
        full_target,
        projections[0],
        frames.relative_mixed_vectors,
        geometric_gram_relative_threshold=geometric_gram_relative_threshold,
    )
    return ArchiveMixedPilotPoint(
        space_names=SPACE_NAMES,
        hilbert_relative_residual=np.asarray(
            [projection.relative_residual for projection in projections]
        ),
        closed_coordinate_scaled_rms=closed_errors,
        correlation_scaled_rms=correlation_errors,
        retained_rank=np.asarray(
            [projection.retained_rank for projection in projections],
            dtype=np.int64,
        ),
        geometric_rank=np.asarray(
            [projection.geometric_rank for projection in projections],
            dtype=np.int64,
        ),
        target_velocity_norm=float(np.linalg.norm(full_target)),
        spin_swap_fidelity=frames.spin_swap_fidelity,
        symmetry_projection_fidelity=frames.symmetry_projection_fidelity,
        relative_top_population=frames.relative_top_population,
        archive_gram_max_error=frames.archive_gram_max_error,
        archive_novelty_fraction=archive_novelty,
        mixed_novelty_fraction=mixed_novelty,
        full_mixed_novelty_fraction=full_mixed_novelty,
        archive_novelty_eigenvalues=archive_spectrum,
        mixed_novelty_eigenvalues=mixed_spectrum,
        full_mixed_novelty_eigenvalues=full_mixed_spectrum,
        mixed_candidate_residual_reduction=candidate_scores,
    )


__all__ = [
    "ARCHIVE_LABELS",
    "LOCAL_MIXED_LABELS",
    "MIXED_LABELS",
    "SPACE_NAMES",
    "ArchiveMixedFrames",
    "ArchiveMixedPilotPoint",
    "TangentProjection",
    "archive_mixed_tangent_pilot_point",
    "full_state_matrix_derivative",
    "packet_archive_mixed_frames",
    "project_real_tangent",
]
