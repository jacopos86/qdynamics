"""Factor the packet-state correlation source through mixed tangents.

This module is an offline identifiability diagnostic.  At one stored
multi-coherent state it residualizes the six complex relative-phonon--Pauli
directions against the archive fluctuation frame, projects the exact
same-state Schrodinger velocity into that mixed complement, and contracts the
result into the fourteen real ``C``-velocity coordinates.  The resulting
factorization distinguishes a candidate hidden coefficient vector from the
state-dependent map that makes those coefficients observable in ``dot C``.

No learned quantity or exact-reference value enters an online propagator.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .archive_gram_tangent_pilot import (
    MIXED_LABELS,
    full_state_matrix_derivative,
    packet_archive_mixed_frames,
)
from .hubbard_dimer import DimerParameters, GaussianSineDrive
from .matrix_reference import (
    MatrixDimerState,
    matrix_derivative_to_closed_scalar,
    matrix_state_to_closed_scalar_coordinates,
    pauli_repaired_closed_scalar_rhs,
)
from .multi_coherent import relative_holstein_hamiltonian

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

C_BLOCK = slice(17, 31)


@dataclass(frozen=True)
class MixedTangentClosurePoint:
    """One same-state factorization of the missing correlation velocity."""

    closed_coordinates: FloatArray
    mixed_coefficients: FloatArray
    mixed_response: FloatArray
    target_source: FloatArray
    archive_frame_source: FloatArray
    mixed_source: FloatArray
    unresolved_source: FloatArray
    target_correlation_velocity: FloatArray
    enriched_correlation_velocity: FloatArray
    archive_eom_correlation_velocity: FloatArray
    archive_rank: int
    mixed_complement_rank: int
    mixed_largest_singular_value: float
    mixed_smallest_retained_singular_value: float
    archive_gram_max_error: float
    relative_top_population: float


class _ProtocolParameters:
    """Delegate physical parameters while replacing the drive protocol."""

    def __init__(
        self,
        parameters: DimerParameters,
        drive_protocol: GaussianSineDrive,
    ) -> None:
        self._parameters = parameters
        self._drive_protocol = drive_protocol

    def __getattr__(self, name: str) -> Any:
        return getattr(self._parameters, name)

    def drive_difference(self, time: float) -> float:
        return self._drive_protocol.difference(time)


def _realify_vector(vector: ComplexArray) -> FloatArray:
    values = np.asarray(vector, dtype=complex)
    return np.concatenate((values.real, values.imag))


def _complexify_vector(vector: FloatArray) -> ComplexArray:
    values = np.asarray(vector, dtype=float)
    if values.ndim != 1 or values.size % 2:
        raise ValueError("realified vector must have even one-dimensional size")
    size = values.size // 2
    return values[:size] + 1j * values[size:]


def _realify_frame(frame: ComplexArray) -> FloatArray:
    values = np.asarray(frame, dtype=complex)
    if values.ndim != 2:
        raise ValueError("frame must be two-dimensional")
    return np.vstack((values.real, values.imag))


def _range_basis(
    frame: FloatArray,
    *,
    relative_gram_threshold: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    values = np.asarray(frame, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("frame must have at least one column")
    left, singular_values, right_adjoint = np.linalg.svd(
        values,
        full_matrices=False,
    )
    threshold = sqrt(relative_gram_threshold) * singular_values[0]
    retained = singular_values > threshold
    return (
        left[:, retained],
        singular_values[retained],
        right_adjoint[retained],
        retained,
    )


def _physical_matrix_state(
    centered_state: MatrixDimerState,
    *,
    center_amplitude: complex,
) -> MatrixDimerState:
    """Restore the factored center-mode displacement to the local amplitudes."""

    local_shift = center_amplitude / sqrt(2.0)
    return MatrixDimerState(
        electron_density=np.asarray(
            centered_state.electron_density,
            dtype=complex,
        ),
        coherent_phonon=(
            np.asarray(centered_state.coherent_phonon, dtype=complex)
            + local_shift
        ),
        phonon_density=np.asarray(
            centered_state.phonon_density,
            dtype=complex,
        ),
        anomalous_phonon_density=np.asarray(
            centered_state.anomalous_phonon_density,
            dtype=complex,
        ),
        electron_phonon_correlation=np.asarray(
            centered_state.electron_phonon_correlation,
            dtype=complex,
        ),
    )


def _closed_velocity(
    state: ComplexArray,
    velocity: ComplexArray,
    *,
    relative_dimension: int,
) -> FloatArray:
    return matrix_derivative_to_closed_scalar(
        full_state_matrix_derivative(
            state,
            velocity,
            relative_dimension=relative_dimension,
        )
    )


def mixed_tangent_closure_point(
    packet_parameters: FloatArray,
    *,
    time: float,
    parameters: DimerParameters,
    drive_protocol: GaussianSineDrive,
    relative_dimension: int,
    geometric_gram_relative_threshold: float = 1e-10,
) -> MixedTangentClosurePoint:
    """Return the mixed-tangent factorization at one packet state."""

    if not 0.0 < geometric_gram_relative_threshold < 1.0:
        raise ValueError(
            "geometric_gram_relative_threshold must lie between zero and one"
        )
    frames = packet_archive_mixed_frames(
        np.asarray(packet_parameters, dtype=float),
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
    target_real = _realify_vector(full_target)

    archive_frame = _realify_frame(frames.archive_tangent)
    archive_basis, _, _, _ = _range_basis(
        archive_frame,
        relative_gram_threshold=geometric_gram_relative_threshold,
    )
    archive_real_velocity = archive_basis @ (
        archive_basis.T @ target_real
    )

    mixed_frame = _realify_frame(frames.relative_mixed_tangent)
    mixed_complement = mixed_frame - archive_basis @ (
        archive_basis.T @ mixed_frame
    )
    mixed_basis, mixed_singular, mixed_right, _ = _range_basis(
        mixed_complement,
        relative_gram_threshold=geometric_gram_relative_threshold,
    )
    mixed_coordinates = mixed_basis.T @ (
        target_real - archive_real_velocity
    )
    mixed_coefficients = mixed_right.T @ (
        mixed_coordinates / mixed_singular
    )
    mixed_real_velocity = mixed_complement @ mixed_coefficients
    enriched_real_velocity = archive_real_velocity + mixed_real_velocity

    target_closed = _closed_velocity(
        frames.full_state,
        full_target,
        relative_dimension=relative_dimension,
    )
    archive_frame_closed = _closed_velocity(
        frames.full_state,
        _complexify_vector(archive_real_velocity),
        relative_dimension=relative_dimension,
    )
    enriched_closed = _closed_velocity(
        frames.full_state,
        _complexify_vector(enriched_real_velocity),
        relative_dimension=relative_dimension,
    )

    mixed_response = np.empty((14, mixed_complement.shape[1]), dtype=float)
    for column in range(mixed_complement.shape[1]):
        mixed_response[:, column] = _closed_velocity(
            frames.full_state,
            _complexify_vector(mixed_complement[:, column]),
            relative_dimension=relative_dimension,
        )[C_BLOCK]

    center_amplitude = (
        -sqrt(2.0) * parameters.coupling / parameters.omega_ph
    )
    matrix_state = _physical_matrix_state(
        frames.matrix_state,
        center_amplitude=center_amplitude,
    )
    closed_coordinates = matrix_state_to_closed_scalar_coordinates(
        matrix_state
    )
    protocol_parameters = _ProtocolParameters(parameters, drive_protocol)
    archive_eom = pauli_repaired_closed_scalar_rhs(
        float(time),
        closed_coordinates,
        protocol_parameters,  # type: ignore[arg-type]
    )
    mixed_source = mixed_response @ mixed_coefficients
    target_source = target_closed[C_BLOCK] - archive_eom[C_BLOCK]
    archive_frame_source = (
        archive_frame_closed[C_BLOCK] - archive_eom[C_BLOCK]
    )
    unresolved_source = (
        target_source - archive_frame_source - mixed_source
    )
    return MixedTangentClosurePoint(
        closed_coordinates=np.asarray(closed_coordinates, dtype=float),
        mixed_coefficients=np.asarray(mixed_coefficients, dtype=float),
        mixed_response=np.asarray(mixed_response, dtype=float),
        target_source=np.asarray(target_source, dtype=float),
        archive_frame_source=np.asarray(archive_frame_source, dtype=float),
        mixed_source=np.asarray(mixed_source, dtype=float),
        unresolved_source=np.asarray(unresolved_source, dtype=float),
        target_correlation_velocity=np.asarray(
            target_closed[C_BLOCK],
            dtype=float,
        ),
        enriched_correlation_velocity=np.asarray(
            enriched_closed[C_BLOCK],
            dtype=float,
        ),
        archive_eom_correlation_velocity=np.asarray(
            archive_eom[C_BLOCK],
            dtype=float,
        ),
        archive_rank=archive_basis.shape[1],
        mixed_complement_rank=mixed_basis.shape[1],
        mixed_largest_singular_value=float(mixed_singular[0]),
        mixed_smallest_retained_singular_value=float(mixed_singular[-1]),
        archive_gram_max_error=frames.archive_gram_max_error,
        relative_top_population=frames.relative_top_population,
    )


__all__ = [
    "C_BLOCK",
    "MIXED_LABELS",
    "MixedTangentClosurePoint",
    "mixed_tangent_closure_point",
]
