"""Autonomous first-order rollout of the integrable mixed packet layer.

The archive-observer directions are not used here.  At each accepted state the
Schrodinger velocity is projected onto the spin-symmetric native packet
tangent plus the twelve real analytic mixed-layer tangents.  One local
retraction step updates the native parameters, applies the mixed exponential,
and compresses its exact packet union to the requested state tolerance without
an externally imposed packet-count cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .archive_gram_tangent_pilot import (
    full_state_matrix_derivative,
    packet_archive_mixed_frames,
    project_real_tangent,
)
from .hubbard_dimer import DimerParameters, GaussianSineDrive
from .matrix_reference import (
    electron_phonon_moment_derivative,
    electron_phonon_moment_matrix,
    matrix_derivative_to_closed_scalar,
)
from .mixed_exponential_layer import (
    mixed_exponential_origin_tangent,
    mixed_layer_centers,
    retract_mixed_exponential_layer,
)
from .multi_coherent import (
    multi_coherent_capacity,
    multi_coherent_state,
    pack_multi_coherent_parameters,
    relative_holstein_hamiltonian,
    retract_multi_coherent_parameters,
    spawn_residual_coherent_packets,
    unpack_multi_coherent_parameters,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class MixedEnrichedProjection:
    """Real-coordinate McLachlan solve in the native-plus-mixed frame."""

    state: ComplexArray
    native_tangent: ComplexArray
    mixed_tangent: ComplexArray
    native_parameter_velocity: FloatArray
    mixed_coordinate_velocity: FloatArray
    native_projected_velocity: ComplexArray
    projected_velocity: ComplexArray
    target_velocity: ComplexArray
    native_relative_residual: float
    enriched_relative_residual: float
    native_rank: int
    enriched_rank: int
    largest_singular_value: float
    smallest_retained_singular_value: float


@dataclass(frozen=True)
class MixedEnrichedStep:
    """One accepted local retraction step."""

    parameters: FloatArray
    packet_count: int
    candidate_counts: tuple[int, int, int, int]
    retained_counts: tuple[int, int, int, int]
    retraction_state_error: float
    retraction_fidelity: float
    native_relative_residual: float
    enriched_relative_residual: float
    native_rank: int
    enriched_rank: int
    native_parameter_speed: float
    mixed_coordinate_speed: float


@dataclass(frozen=True)
class MixedGuidedPacketAdmission:
    """State-continuous packet admission fitted to the mixed-only gain."""

    parameters: FloatArray
    previous_packet_count: int
    packet_count: int
    fitted_centers: tuple[complex, complex, complex, complex]
    state_discontinuity: float
    native_relative_residual_before: float
    native_relative_residual_after: float
    mixed_gain_norm_before: float
    mixed_gain_norm_after: float
    function_evaluations: int


@dataclass(frozen=True)
class ArchiveGramAdmissionSignals:
    """Three current-state Route-2 packet-admission objectives."""

    native_hilbert_residual_squared: float
    native_hilbert_relative_residual: float
    target_velocity_norm: float
    joint_gram_rate_defect_squared: float
    mixed_observable_impact_squared: float
    native_geometric_rank: int
    native_condition_number: float
    joint_gram_support_rank: int
    mixed_novel_rank: int
    minimum_joint_gram_eigenvalue: float


def _real_projection(
    target: ComplexArray,
    frame: ComplexArray,
    *,
    relative_damping: float | None,
    geometric_relative_threshold: float,
) -> tuple[FloatArray, ComplexArray, float, int, float, float]:
    real_frame = np.vstack((frame.real, frame.imag))
    real_target = np.concatenate((target.real, target.imag))
    left, singular_values, right_adjoint = np.linalg.svd(
        real_frame,
        full_matrices=False,
    )
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise ValueError("tangent frame has no nonzero direction")
    geometric_threshold = (
        np.sqrt(geometric_relative_threshold) * singular_values[0]
    )
    if relative_damping is None:
        retained_mask = singular_values > geometric_threshold
        inverse = np.zeros_like(singular_values)
        inverse[retained_mask] = 1.0 / singular_values[retained_mask]
    else:
        damping = relative_damping * singular_values[0]
        inverse = singular_values / (singular_values**2 + damping**2)
        retained_mask = singular_values > damping
    coefficients = right_adjoint.T @ (
        inverse * (left.T @ real_target)
    )
    projected = frame @ coefficients
    relative_residual = float(
        np.linalg.norm(projected - target)
        / max(np.linalg.norm(target), np.finfo(float).tiny)
    )
    rank = int(np.count_nonzero(singular_values > geometric_threshold))
    retained = singular_values[retained_mask]
    smallest = float(retained[-1] if retained.size else singular_values[0])
    return (
        np.asarray(coefficients, dtype=float),
        np.asarray(projected, dtype=complex),
        relative_residual,
        rank,
        float(singular_values[0]),
        smallest,
    )


def archive_gram_admission_signals(
    time: float,
    packet_parameters: FloatArray,
    parameters: DimerParameters,
    *,
    relative_dimension: int,
    coordinate_scales: FloatArray,
    drive_protocol: GaussianSineDrive | None = None,
    geometric_relative_threshold: float = 1e-10,
    gram_support_relative_threshold: float = 1e-10,
) -> ArchiveGramAdmissionSignals:
    """Evaluate the full memorandum admission signals from the current ket.

    No exact or future trajectory value is used.  ``coordinate_scales`` is a
    frozen construction metric for the 31 retained coordinates.
    """

    scales = np.asarray(coordinate_scales, dtype=float)
    if scales.shape != (31,) or np.any(~np.isfinite(scales)) or np.any(
        scales <= 0.0
    ):
        raise ValueError("coordinate_scales must be finite and positive")
    if not 0.0 < gram_support_relative_threshold < 1.0:
        raise ValueError("gram support threshold must lie in (0, 1)")

    frames = packet_archive_mixed_frames(
        packet_parameters,
        relative_dimension=relative_dimension,
    )
    centers = mixed_layer_centers(
        packet_parameters,
        relative_dimension=relative_dimension,
    )
    chart_state, mixed_tangent = mixed_exponential_origin_tangent(
        packet_parameters,
        relative_dimension=relative_dimension,
        centers=centers,
    )
    if np.linalg.norm(chart_state - frames.relative_state) > 1e-10:
        raise RuntimeError("native and mixed charts reconstruct different states")
    native_tangent = frames.packet_tangent[: 4 * relative_dimension]
    hamiltonian = relative_holstein_hamiltonian(
        time,
        parameters,
        relative_dimension=relative_dimension,
        drive_protocol=drive_protocol,
    )
    energy = float(np.vdot(chart_state, hamiltonian @ chart_state).real)
    target = -1j * (hamiltonian @ chart_state - energy * chart_state)
    native = project_real_tangent(
        target,
        native_tangent,
        geometric_gram_relative_threshold=geometric_relative_threshold,
    )
    residual = target - native.projected_velocity
    native_residual_squared = float(np.vdot(residual, residual).real)

    full_residual = np.concatenate((residual, np.zeros_like(residual)))
    residual_matrix_derivative = full_state_matrix_derivative(
        frames.full_state,
        full_residual,
        relative_dimension=relative_dimension,
    )
    joint_gram = electron_phonon_moment_matrix(frames.matrix_state)
    gram_rate_defect = electron_phonon_moment_derivative(
        frames.matrix_state,
        residual_matrix_derivative,
    )
    gram_eigenvalues, gram_eigenvectors = np.linalg.eigh(joint_gram)
    gram_threshold = gram_support_relative_threshold * max(
        1.0,
        float(np.max(np.abs(gram_eigenvalues))),
    )
    gram_support = gram_eigenvalues > gram_threshold
    if np.any(gram_support):
        inverse_square_root = (
            gram_eigenvectors[:, gram_support]
            / np.sqrt(gram_eigenvalues[gram_support])
        ) @ gram_eigenvectors[:, gram_support].conjugate().T
        whitened_rate = (
            inverse_square_root
            @ gram_rate_defect
            @ inverse_square_root
        )
        joint_gram_rate_squared = float(
            np.linalg.norm(whitened_rate, ord="fro") ** 2
        )
    else:
        joint_gram_rate_squared = 0.0

    full_native_tangent = np.vstack(
        (native_tangent, np.zeros_like(native_tangent))
    )
    native_observer_tangent = np.column_stack(
        (full_native_tangent, frames.archive_tangent)
    )
    full_target = np.concatenate((target, np.zeros_like(target)))
    native_observer = project_real_tangent(
        full_target,
        native_observer_tangent,
        geometric_gram_relative_threshold=geometric_relative_threshold,
    )
    full_mixed_tangent = np.vstack(
        (mixed_tangent, np.zeros_like(mixed_tangent))
    )
    real_mixed = np.vstack(
        (full_mixed_tangent.real, full_mixed_tangent.imag)
    )
    observer_basis = native_observer.real_range_basis
    novel_mixed = real_mixed - observer_basis @ (
        observer_basis.T @ real_mixed
    )
    left, singular_values, _ = np.linalg.svd(
        novel_mixed,
        full_matrices=False,
    )
    if singular_values.size and singular_values[0] > 0.0:
        mixed_threshold = (
            np.sqrt(geometric_relative_threshold) * singular_values[0]
        )
        mixed_support = singular_values > mixed_threshold
        mixed_basis = left[:, mixed_support]
        real_residual = np.concatenate(
            (full_residual.real, full_residual.imag)
        )
        mixed_projected_real = mixed_basis @ (
            mixed_basis.T @ real_residual
        )
        full_size = full_residual.size
        mixed_projected = (
            mixed_projected_real[:full_size]
            + 1j * mixed_projected_real[full_size:]
        )
        mixed_coordinate_velocity = matrix_derivative_to_closed_scalar(
            full_state_matrix_derivative(
                frames.full_state,
                mixed_projected,
                relative_dimension=relative_dimension,
            )
        )
        mixed_impact_squared = float(
            np.dot(
                mixed_coordinate_velocity / scales,
                mixed_coordinate_velocity / scales,
            )
        )
        mixed_rank = int(np.count_nonzero(mixed_support))
    else:
        mixed_impact_squared = 0.0
        mixed_rank = 0

    native_condition = float(
        native.largest_singular_value
        / max(native.smallest_retained_singular_value, np.finfo(float).tiny)
    )
    return ArchiveGramAdmissionSignals(
        native_hilbert_residual_squared=native_residual_squared,
        native_hilbert_relative_residual=float(native.relative_residual),
        target_velocity_norm=float(np.linalg.norm(target)),
        joint_gram_rate_defect_squared=joint_gram_rate_squared,
        mixed_observable_impact_squared=mixed_impact_squared,
        native_geometric_rank=native.geometric_rank,
        native_condition_number=native_condition,
        joint_gram_support_rank=int(np.count_nonzero(gram_support)),
        mixed_novel_rank=mixed_rank,
        minimum_joint_gram_eigenvalue=float(gram_eigenvalues[0]),
    )


def project_mixed_enriched_velocity(
    time: float,
    packet_parameters: FloatArray,
    parameters: DimerParameters,
    *,
    relative_dimension: int,
    drive_protocol: GaussianSineDrive | None = None,
    relative_damping: float | None = None,
    geometric_relative_threshold: float = 1e-10,
) -> MixedEnrichedProjection:
    """Project the autonomous Hamiltonian velocity onto native plus mixed."""

    if relative_damping is not None and relative_damping <= 0.0:
        raise ValueError("relative_damping must be positive")
    if not 0.0 < geometric_relative_threshold < 1.0:
        raise ValueError("geometric_relative_threshold must lie in (0, 1)")
    frames = packet_archive_mixed_frames(
        packet_parameters,
        relative_dimension=relative_dimension,
    )
    centers = mixed_layer_centers(
        packet_parameters,
        relative_dimension=relative_dimension,
    )
    chart_state, mixed_tangent = mixed_exponential_origin_tangent(
        packet_parameters,
        relative_dimension=relative_dimension,
        centers=centers,
    )
    if np.linalg.norm(chart_state - frames.relative_state) > 1e-10:
        raise RuntimeError("native and mixed charts reconstruct different states")
    native_tangent = frames.packet_tangent[: 4 * relative_dimension]
    hamiltonian = relative_holstein_hamiltonian(
        time,
        parameters,
        relative_dimension=relative_dimension,
        drive_protocol=drive_protocol,
    )
    energy = float(np.vdot(chart_state, hamiltonian @ chart_state).real)
    target = -1j * (hamiltonian @ chart_state - energy * chart_state)
    native_solution = _real_projection(
        target,
        native_tangent,
        relative_damping=relative_damping,
        geometric_relative_threshold=geometric_relative_threshold,
    )
    enriched_tangent = np.column_stack((native_tangent, mixed_tangent))
    enriched_solution = _real_projection(
        target,
        enriched_tangent,
        relative_damping=relative_damping,
        geometric_relative_threshold=geometric_relative_threshold,
    )
    coefficients, projected, residual, rank, largest, smallest = (
        enriched_solution
    )
    native_size = native_tangent.shape[1]
    return MixedEnrichedProjection(
        state=chart_state,
        native_tangent=native_tangent,
        mixed_tangent=mixed_tangent,
        native_parameter_velocity=coefficients[:native_size],
        mixed_coordinate_velocity=coefficients[native_size:],
        native_projected_velocity=native_solution[1],
        projected_velocity=projected,
        target_velocity=target,
        native_relative_residual=native_solution[2],
        enriched_relative_residual=residual,
        native_rank=native_solution[3],
        enriched_rank=rank,
        largest_singular_value=largest,
        smallest_retained_singular_value=smallest,
    )


def admit_mixed_guided_packets(
    time: float,
    packet_parameters: FloatArray,
    parameters: DimerParameters,
    *,
    relative_dimension: int,
    drive_protocol: GaussianSineDrive | None = None,
    relative_damping: float | None = None,
    geometric_relative_threshold: float = 1e-10,
    fit_maximum_iterations: int = 40,
    fit_population_size: int = 6,
    fit_seed: int = 0,
) -> MixedGuidedPacketAdmission:
    """Append zero-weight packets fitted only to the mixed-layer gain."""

    before = project_mixed_enriched_velocity(
        time,
        packet_parameters,
        parameters,
        relative_dimension=relative_dimension,
        drive_protocol=drive_protocol,
        relative_damping=relative_damping,
        geometric_relative_threshold=geometric_relative_threshold,
    )
    mixed_gain = before.projected_velocity - before.native_projected_velocity
    gain_norm_before = float(np.linalg.norm(mixed_gain))
    if gain_norm_before <= np.finfo(float).tiny:
        raise ValueError("mixed layer supplies no unresolved velocity gain")
    spawn = spawn_residual_coherent_packets(
        packet_parameters,
        mixed_gain,
        relative_dimension=relative_dimension,
        maximum_iterations=fit_maximum_iterations,
        population_size=fit_population_size,
        seed=fit_seed,
    )
    spawned_coefficients, spawned_displacements = (
        unpack_multi_coherent_parameters(spawn.parameters)
    )
    shared_center = 0.5 * (
        spawned_displacements[1, -1]
        + spawned_displacements[2, -1]
    )
    spawned_displacements[1, -1] = shared_center
    spawned_displacements[2, -1] = shared_center
    symmetric_parameters = pack_multi_coherent_parameters(
        spawned_coefficients,
        spawned_displacements,
    )
    after = project_mixed_enriched_velocity(
        time,
        symmetric_parameters,
        parameters,
        relative_dimension=relative_dimension,
        drive_protocol=drive_protocol,
        relative_damping=relative_damping,
        geometric_relative_threshold=geometric_relative_threshold,
    )
    gain_after = after.projected_velocity - after.native_projected_velocity
    previous_count = multi_coherent_capacity(
        packet_parameters
    ).packets_per_electronic_branch
    return MixedGuidedPacketAdmission(
        parameters=np.asarray(symmetric_parameters, dtype=float),
        previous_packet_count=previous_count,
        packet_count=spawn.packet_count,
        fitted_centers=tuple(
            complex(value) for value in spawned_displacements[:, -1]
        ),
        state_discontinuity=spawn.state_discontinuity,
        native_relative_residual_before=before.native_relative_residual,
        native_relative_residual_after=after.native_relative_residual,
        mixed_gain_norm_before=gain_norm_before,
        mixed_gain_norm_after=float(np.linalg.norm(gain_after)),
        function_evaluations=spawn.function_evaluations,
    )


def _apply_mixed_enriched_increment(
    packet_parameters: FloatArray,
    native_parameter_increment: FloatArray,
    mixed_coordinate_increment: FloatArray,
    *,
    relative_dimension: int,
    centers,
    retraction_relative_tolerance: float,
):
    native_endpoint = retract_multi_coherent_parameters(
        np.asarray(packet_parameters, dtype=float)
        + np.asarray(native_parameter_increment, dtype=float),
        relative_dimension=relative_dimension,
    )
    return retract_mixed_exponential_layer(
        native_endpoint,
        np.asarray(mixed_coordinate_increment, dtype=float),
        relative_dimension=relative_dimension,
        centers=centers,
        relative_tolerance=retraction_relative_tolerance,
    )


def mixed_enriched_euler_step(
    time: float,
    packet_parameters: FloatArray,
    time_step: float,
    parameters: DimerParameters,
    *,
    relative_dimension: int,
    drive_protocol: GaussianSineDrive | None = None,
    relative_damping: float | None = None,
    geometric_relative_threshold: float = 1e-10,
    retraction_relative_tolerance: float = 1e-10,
) -> MixedEnrichedStep:
    """Advance one autonomous first-order native-plus-mixed manifold step."""

    if time_step <= 0.0:
        raise ValueError("time_step must be positive")
    projection = project_mixed_enriched_velocity(
        time,
        packet_parameters,
        parameters,
        relative_dimension=relative_dimension,
        drive_protocol=drive_protocol,
        relative_damping=relative_damping,
        geometric_relative_threshold=geometric_relative_threshold,
    )
    centers = mixed_layer_centers(
        packet_parameters,
        relative_dimension=relative_dimension,
    )
    retraction = _apply_mixed_enriched_increment(
        packet_parameters,
        time_step * projection.native_parameter_velocity,
        time_step * projection.mixed_coordinate_velocity,
        relative_dimension=relative_dimension,
        centers=centers,
        retraction_relative_tolerance=retraction_relative_tolerance,
    )
    return MixedEnrichedStep(
        parameters=retraction.parameters,
        packet_count=retraction.packet_count,
        candidate_counts=retraction.candidate_counts,
        retained_counts=retraction.retained_counts,
        retraction_state_error=retraction.state_error,
        retraction_fidelity=retraction.fidelity,
        native_relative_residual=projection.native_relative_residual,
        enriched_relative_residual=projection.enriched_relative_residual,
        native_rank=projection.native_rank,
        enriched_rank=projection.enriched_rank,
        native_parameter_speed=float(
            np.linalg.norm(projection.native_parameter_velocity)
        ),
        mixed_coordinate_speed=float(
            np.linalg.norm(projection.mixed_coordinate_velocity)
        ),
    )


def mixed_enriched_midpoint_step(
    time: float,
    packet_parameters: FloatArray,
    time_step: float,
    parameters: DimerParameters,
    *,
    relative_dimension: int,
    drive_protocol: GaussianSineDrive | None = None,
    relative_damping: float | None = None,
    geometric_relative_threshold: float = 1e-10,
    retraction_relative_tolerance: float = 1e-10,
) -> MixedEnrichedStep:
    """Advance with an ambiently transported explicit manifold midpoint."""

    if time_step <= 0.0:
        raise ValueError("time_step must be positive")
    origin = project_mixed_enriched_velocity(
        time,
        packet_parameters,
        parameters,
        relative_dimension=relative_dimension,
        drive_protocol=drive_protocol,
        relative_damping=relative_damping,
        geometric_relative_threshold=geometric_relative_threshold,
    )
    origin_centers = mixed_layer_centers(
        packet_parameters,
        relative_dimension=relative_dimension,
    )
    midpoint_retraction = _apply_mixed_enriched_increment(
        packet_parameters,
        0.5 * time_step * origin.native_parameter_velocity,
        0.5 * time_step * origin.mixed_coordinate_velocity,
        relative_dimension=relative_dimension,
        centers=origin_centers,
        retraction_relative_tolerance=retraction_relative_tolerance,
    )
    midpoint = project_mixed_enriched_velocity(
        time + 0.5 * time_step,
        midpoint_retraction.parameters,
        parameters,
        relative_dimension=relative_dimension,
        drive_protocol=drive_protocol,
        relative_damping=relative_damping,
        geometric_relative_threshold=geometric_relative_threshold,
    )
    midpoint_overlap = complex(np.vdot(origin.state, midpoint.state))
    phase = np.exp(-1j * np.angle(midpoint_overlap))
    transported_velocity = phase * midpoint.projected_velocity
    transported_velocity -= origin.state * np.vdot(
        origin.state,
        transported_velocity,
    )
    origin_frame = np.column_stack(
        (origin.native_tangent, origin.mixed_tangent)
    )
    transported_solution = _real_projection(
        transported_velocity,
        origin_frame,
        relative_damping=relative_damping,
        geometric_relative_threshold=geometric_relative_threshold,
    )
    transported_coordinates = transported_solution[0]
    native_size = origin.native_tangent.shape[1]
    endpoint_retraction = _apply_mixed_enriched_increment(
        packet_parameters,
        time_step * transported_coordinates[:native_size],
        time_step * transported_coordinates[native_size:],
        relative_dimension=relative_dimension,
        centers=origin_centers,
        retraction_relative_tolerance=retraction_relative_tolerance,
    )
    return MixedEnrichedStep(
        parameters=endpoint_retraction.parameters,
        packet_count=endpoint_retraction.packet_count,
        candidate_counts=endpoint_retraction.candidate_counts,
        retained_counts=endpoint_retraction.retained_counts,
        retraction_state_error=max(
            midpoint_retraction.state_error,
            endpoint_retraction.state_error,
        ),
        retraction_fidelity=min(
            midpoint_retraction.fidelity,
            endpoint_retraction.fidelity,
        ),
        native_relative_residual=midpoint.native_relative_residual,
        enriched_relative_residual=midpoint.enriched_relative_residual,
        native_rank=midpoint.native_rank,
        enriched_rank=midpoint.enriched_rank,
        native_parameter_speed=float(
            np.linalg.norm(transported_coordinates[:native_size])
        ),
        mixed_coordinate_speed=float(
            np.linalg.norm(transported_coordinates[native_size:])
        ),
    )


def normalized_packet_state(
    packet_parameters: FloatArray,
    *,
    relative_dimension: int,
) -> ComplexArray:
    """Return the normalized ket represented by one packet parameter vector."""

    state = multi_coherent_state(
        packet_parameters,
        relative_dimension=relative_dimension,
    )
    return np.asarray(state / np.linalg.norm(state), dtype=complex)


__all__ = [
    "ArchiveGramAdmissionSignals",
    "MixedEnrichedProjection",
    "MixedEnrichedStep",
    "MixedGuidedPacketAdmission",
    "admit_mixed_guided_packets",
    "archive_gram_admission_signals",
    "mixed_enriched_euler_step",
    "mixed_enriched_midpoint_step",
    "normalized_packet_state",
    "project_mixed_enriched_velocity",
]
