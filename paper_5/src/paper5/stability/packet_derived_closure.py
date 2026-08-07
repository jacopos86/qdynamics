"""Analytic packet-state velocities for an offline closure-source audit.

The routines in this module evaluate two velocities at the same normalized
multi-coherent packet state: the McLachlan tangent projection and the exact
Schrodinger tangent before projection.  Contracting both into the established
31-coordinate moment chart separates packet-manifold error from the missing
source of the archive moment equations.  No trajectory is propagated here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .hubbard_dimer import DimerParameters, GaussianSineDrive
from .matrix_reference import (
    matrix_derivative_to_closed_scalar,
    matrix_state_to_closed_scalar_coordinates,
)
from .moment_hierarchy import MomentHierarchy
from .multi_coherent import (
    multi_coherent_state,
    project_schrodinger_velocity,
    relative_holstein_hamiltonian,
    relative_state_moment_coordinates,
    relative_state_moment_derivative,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PacketClosedVelocityPair:
    """McLachlan and exact-Hamiltonian velocities at one packet state."""

    closed_coordinates: FloatArray
    projected_closed_velocity: FloatArray
    schrodinger_closed_velocity: FloatArray
    hierarchy_coordinate_max_error: float
    tangent_relative_residual: float
    tangent_absolute_residual: float
    tangent_rank: int
    geometric_tangent_rank: int
    parameter_velocity_norm: float


def _closed_velocity_from_state_tangent(
    state: np.ndarray,
    state_tangent: np.ndarray,
    hierarchy: MomentHierarchy,
    hierarchy_coordinates: FloatArray,
    *,
    center_derivative: complex,
) -> FloatArray:
    hierarchy_derivative = relative_state_moment_derivative(
        state,
        state_tangent,
        hierarchy,
        center_derivative=center_derivative,
    )
    return matrix_derivative_to_closed_scalar(
        hierarchy.matrix_derivative(
            hierarchy_coordinates,
            hierarchy_derivative,
        )
    )


def packet_closed_velocity_pair(
    packet_parameters: FloatArray,
    *,
    time: float,
    parameters: DimerParameters,
    drive_protocol: GaussianSineDrive,
    relative_dimension: int,
    hierarchy: MomentHierarchy,
    center_amplitude: complex,
    center_derivative: complex = 0.0j,
    hierarchy_coordinates: FloatArray | None = None,
    tangent_singular_value_cutoff: float = 1e-2,
    tangent_regularization: str = "tikhonov",
    relative_damping: float = 3e-4,
) -> PacketClosedVelocityPair:
    """Contract matched projected and Schrodinger packet-state velocities."""

    packed = np.asarray(packet_parameters, dtype=float)
    if packed.ndim != 1 or packed.size % 16 != 0:
        raise ValueError("packet_parameters must be a finite 16*K vector")
    if not np.all(np.isfinite(packed)):
        raise ValueError("packet_parameters must be finite")
    state = multi_coherent_state(
        packed,
        relative_dimension=relative_dimension,
    )
    normalized_state = state / np.linalg.norm(state)
    hamiltonian = relative_holstein_hamiltonian(
        float(time),
        parameters,
        relative_dimension=relative_dimension,
        drive_protocol=drive_protocol,
    )
    projection = project_schrodinger_velocity(
        packed,
        hamiltonian,
        relative_dimension=relative_dimension,
        relative_singular_value_cutoff=tangent_singular_value_cutoff,
        regularization=tangent_regularization,  # type: ignore[arg-type]
        relative_damping=relative_damping,
    )
    reconstructed_coordinates = relative_state_moment_coordinates(
        state,
        hierarchy,
        center_amplitude=center_amplitude,
    )
    if hierarchy_coordinates is None:
        hierarchy_coordinate_max_error = 0.0
    else:
        stored_coordinates = np.asarray(hierarchy_coordinates, dtype=float)
        if stored_coordinates.shape != (hierarchy.coordinate_count,):
            raise ValueError("hierarchy_coordinates have incompatible shape")
        hierarchy_coordinate_max_error = float(
            np.max(np.abs(reconstructed_coordinates - stored_coordinates))
        )
    coordinates = reconstructed_coordinates
    closed = matrix_state_to_closed_scalar_coordinates(
        hierarchy.to_matrix_state(coordinates)
    )
    projected = _closed_velocity_from_state_tangent(
        normalized_state,
        projection.projected_velocity,
        hierarchy,
        coordinates,
        center_derivative=center_derivative,
    )
    schrodinger = _closed_velocity_from_state_tangent(
        normalized_state,
        projection.target_velocity,
        hierarchy,
        coordinates,
        center_derivative=center_derivative,
    )
    return PacketClosedVelocityPair(
        closed_coordinates=closed,
        projected_closed_velocity=projected,
        schrodinger_closed_velocity=schrodinger,
        hierarchy_coordinate_max_error=hierarchy_coordinate_max_error,
        tangent_relative_residual=projection.relative_residual,
        tangent_absolute_residual=projection.absolute_residual,
        tangent_rank=projection.tangent_rank,
        geometric_tangent_rank=projection.geometric_tangent_rank,
        parameter_velocity_norm=projection.parameter_velocity_norm,
    )


def scaled_source_fluctuation_rms(
    source: np.ndarray,
    coordinate_scales: FloatArray,
) -> float:
    """Return RMS Euclidean fluctuation in fixed scaled coordinates."""

    values = np.asarray(source, dtype=float)
    scales = np.asarray(coordinate_scales, dtype=float)
    if values.ndim < 2 or values.shape[-1] != scales.size:
        raise ValueError("source and coordinate_scales have incompatible shapes")
    if np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
        raise ValueError("coordinate_scales must be finite and positive")
    flattened = values.reshape(-1, scales.size) / scales
    centered = flattened - np.mean(flattened, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def normalized_scaled_source_error(
    candidate: np.ndarray,
    reference: np.ndarray,
    coordinate_scales: FloatArray,
    *,
    reference_fluctuation_scale: float | None = None,
) -> float:
    """Return scaled vector RMS error normalized by reference fluctuation."""

    candidate_values = np.asarray(candidate, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    scales = np.asarray(coordinate_scales, dtype=float)
    if candidate_values.shape != reference_values.shape:
        raise ValueError("candidate and reference shapes must match")
    if candidate_values.shape[-1] != scales.size:
        raise ValueError("coordinate_scales have incompatible shape")
    scale = (
        scaled_source_fluctuation_rms(reference_values, scales)
        if reference_fluctuation_scale is None
        else float(reference_fluctuation_scale)
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("reference fluctuation scale must be positive")
    difference = (candidate_values - reference_values) / scales
    return float(
        np.sqrt(np.mean(np.sum(difference * difference, axis=-1))) / scale
    )


def reconstruct_frozen_source_subspace(
    source: np.ndarray,
    coordinate_scales: FloatArray,
    center: FloatArray,
    basis_rows: FloatArray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a source through a frozen row-orthonormal scaled basis."""

    values = np.asarray(source, dtype=float)
    scales = np.asarray(coordinate_scales, dtype=float)
    source_center = np.asarray(center, dtype=float)
    basis = np.asarray(basis_rows, dtype=float)
    if values.shape[-1] != scales.size:
        raise ValueError("source and scales have incompatible shapes")
    if source_center.shape != scales.shape:
        raise ValueError("center and scales must have matching shapes")
    if basis.ndim != 2 or basis.shape[1] != scales.size:
        raise ValueError("basis_rows have incompatible shape")
    scaled = (values - source_center) / scales
    coefficients = np.einsum("...i,ri->...r", scaled, basis)
    reconstructed_scaled = np.einsum(
        "...r,ri->...i",
        coefficients,
        basis,
    )
    reconstructed = source_center + scales * reconstructed_scaled
    return coefficients, reconstructed
