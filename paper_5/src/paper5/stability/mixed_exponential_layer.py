"""Integrable mixed electron--phonon exponential chart for packet states.

The chart freezes the relative-phonon and electronic centers at one packet
state, applies three ordered mixed exponential factors analytically in the
coherent-state representation, projects the resulting finite packet union to
the declared relative-mode cutoff, and fixes norm and phase.  Its twelve real
coordinate derivatives at the chart origin are the six complex mixed
operator tangents used by the stored-state pilot.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr

from .multi_coherent import (
    multi_coherent_state,
    pack_multi_coherent_parameters,
    unpack_multi_coherent_parameters,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

AXIS_LABELS = ("x", "y", "z")
MIXED_COORDINATE_LABELS = tuple(
    f"z_{sign}_{axis}_{part}"
    for part in ("real", "imag")
    for sign in ("minus", "plus")
    for axis in AXIS_LABELS
)


@dataclass(frozen=True)
class FrozenMixedCenters:
    """Centers held fixed over one local mixed-exponential chart."""

    relative_amplitude: complex
    pauli_means: FloatArray


@dataclass(frozen=True)
class MixedLayerEvaluation:
    """Normalized cutoff image of one analytic finite packet union."""

    state: ComplexArray
    unnormalized_state: ComplexArray
    branch_count: int
    unnormalized_norm: float
    phase_overlap: complex


@dataclass(frozen=True)
class MixedLayerRetraction:
    """Branch-local packet representation of one mixed-layer endpoint."""

    parameters: FloatArray
    packet_count: int
    candidate_counts: tuple[int, int, int, int]
    retained_counts: tuple[int, int, int, int]
    state_error: float
    fidelity: float


def _pauli_matrices() -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    return (
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    )


def _spin_swap() -> ComplexArray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=complex,
    )


def _relative_operators(relative_dimension: int) -> tuple[ComplexArray, ComplexArray]:
    if relative_dimension < 2:
        raise ValueError("relative_dimension must be at least two")
    annihilation = np.zeros(
        (relative_dimension, relative_dimension),
        dtype=complex,
    )
    for occupation in range(1, relative_dimension):
        annihilation[occupation - 1, occupation] = sqrt(float(occupation))
    return annihilation, annihilation.conjugate().T


def _projected_infinite_coherent(
    alpha: complex,
    relative_dimension: int,
) -> ComplexArray:
    values = np.empty(relative_dimension, dtype=complex)
    values[0] = np.exp(-0.5 * abs(alpha) ** 2)
    for occupation in range(1, relative_dimension):
        values[occupation] = (
            values[occupation - 1]
            * alpha
            / sqrt(float(occupation))
        )
    return values


def pack_mixed_coordinates(values: ComplexArray) -> FloatArray:
    """Pack ``[minus/plus, x/y/z]`` complex values into twelve reals."""

    array = np.asarray(values, dtype=complex)
    if array.shape != (2, 3):
        raise ValueError("mixed values must have shape (2, 3)")
    flattened = array.reshape(6)
    return np.concatenate((flattened.real, flattened.imag))


def unpack_mixed_coordinates(coordinates: FloatArray) -> ComplexArray:
    """Return complex ``[minus/plus, x/y/z]`` mixed coordinates."""

    vector = np.asarray(coordinates, dtype=float)
    if vector.shape != (12,) or not np.all(np.isfinite(vector)):
        raise ValueError("mixed coordinates must be finite with shape (12,)")
    return (vector[:6] + 1j * vector[6:]).reshape(2, 3)


def _initial_generalized_packets(
    packet_parameters: FloatArray,
    *,
    relative_dimension: int,
    spin_symmetrize: bool,
) -> list[tuple[ComplexArray, complex]]:
    coefficients, displacements = unpack_multi_coherent_parameters(
        packet_parameters
    )
    swap = _spin_swap()
    packets: list[tuple[ComplexArray, complex]] = []
    for electronic_index in range(4):
        basis = np.zeros(4, dtype=complex)
        basis[electronic_index] = 1.0
        if spin_symmetrize:
            basis = 0.5 * (basis + swap @ basis)
        for packet_index in range(coefficients.shape[1]):
            alpha = complex(displacements[electronic_index, packet_index])
            retained = float(
                np.linalg.norm(
                    _projected_infinite_coherent(alpha, relative_dimension)
                )
            )
            if retained <= np.finfo(float).tiny:
                raise ValueError("coherent packet has no retained cutoff support")
            electronic = (
                coefficients[electronic_index, packet_index]
                * basis
                / retained
            )
            if np.linalg.norm(electronic) > np.finfo(float).tiny:
                packets.append((electronic, alpha))
    return packets


def _packets_to_state(
    packets: list[tuple[ComplexArray, complex]],
    *,
    relative_dimension: int,
) -> ComplexArray:
    state = np.zeros((4, relative_dimension), dtype=complex)
    for electronic, alpha in packets:
        state += electronic[:, None] * _projected_infinite_coherent(
            alpha,
            relative_dimension,
        )[None, :]
    return state.reshape(-1)


def mixed_layer_centers(
    packet_parameters: FloatArray,
    *,
    relative_dimension: int,
    spin_symmetrize: bool = True,
) -> FrozenMixedCenters:
    """Contract the centers frozen by one local chart."""

    packets = _initial_generalized_packets(
        packet_parameters,
        relative_dimension=relative_dimension,
        spin_symmetrize=spin_symmetrize,
    )
    state = _packets_to_state(packets, relative_dimension=relative_dimension)
    state /= np.linalg.norm(state)
    annihilation, _ = _relative_operators(relative_dimension)
    reshaped = state.reshape(4, relative_dimension)
    relative_amplitude = complex(
        sum(
            np.vdot(block, annihilation @ block)
            for block in reshaped
        )
    )
    electronic_density = reshaped @ reshaped.conjugate().T
    pauli_means = np.asarray(
        [
            np.trace(
                electronic_density
                @ np.kron(pauli, np.eye(2, dtype=complex))
            ).real
            for pauli in _pauli_matrices()
        ],
        dtype=float,
    )
    return FrozenMixedCenters(
        relative_amplitude=relative_amplitude,
        pauli_means=pauli_means,
    )


def _apply_axis_factor(
    packets: list[tuple[ComplexArray, complex]],
    *,
    pauli: ComplexArray,
    pauli_mean: float,
    relative_amplitude: complex,
    z_minus: complex,
    z_plus: complex,
) -> list[tuple[ComplexArray, complex]]:
    electronic_identity = np.eye(4, dtype=complex)
    sigma = np.kron(pauli, np.eye(2, dtype=complex))
    projectors = (
        0.5 * (electronic_identity + sigma),
        0.5 * (electronic_identity - sigma),
    )
    eigenvalues = (1.0 - pauli_mean, -1.0 - pauli_mean)
    result: list[tuple[ComplexArray, complex]] = []
    for electronic, alpha in packets:
        for projector, eigenvalue in zip(
            projectors,
            eigenvalues,
            strict=True,
        ):
            projected = projector @ electronic
            if np.linalg.norm(projected) <= 1e-15:
                continue
            creation_coefficient = eigenvalue * z_plus
            annihilation_coefficient = eigenvalue * z_minus
            constant = -eigenvalue * (
                z_minus * relative_amplitude
                + z_plus * relative_amplitude.conjugate()
            )
            shifted = alpha + creation_coefficient
            logarithm = (
                constant
                + 0.5
                * creation_coefficient
                * annihilation_coefficient
                + annihilation_coefficient * alpha
                + 0.5 * (abs(shifted) ** 2 - abs(alpha) ** 2)
            )
            result.append((np.exp(logarithm) * projected, shifted))
    return result


def _transformed_generalized_packets(
    packet_parameters: FloatArray,
    mixed_coordinates: FloatArray,
    *,
    relative_dimension: int,
    centers: FrozenMixedCenters,
    axis_order: tuple[str, str, str],
    spin_symmetrize: bool,
) -> list[tuple[ComplexArray, complex]]:
    values = unpack_mixed_coordinates(mixed_coordinates)
    packets = _initial_generalized_packets(
        packet_parameters,
        relative_dimension=relative_dimension,
        spin_symmetrize=spin_symmetrize,
    )
    pauli = _pauli_matrices()
    label_to_index = {label: index for index, label in enumerate(AXIS_LABELS)}
    for label in axis_order:
        index = label_to_index[label]
        packets = _apply_axis_factor(
            packets,
            pauli=pauli[index],
            pauli_mean=float(centers.pauli_means[index]),
            relative_amplitude=centers.relative_amplitude,
            z_minus=complex(values[0, index]),
            z_plus=complex(values[1, index]),
        )
    return packets


def mixed_exponential_layer_state(
    packet_parameters: FloatArray,
    mixed_coordinates: FloatArray,
    *,
    relative_dimension: int,
    centers: FrozenMixedCenters | None = None,
    axis_order: tuple[str, str, str] = AXIS_LABELS,
    spin_symmetrize: bool = True,
) -> MixedLayerEvaluation:
    """Evaluate the normalized analytic mixed-exponential packet union.

    ``axis_order`` is the order in which factors act on the state.  The
    centers remain fixed throughout this call and therefore throughout any
    local macrostep that reuses the supplied ``centers``.
    """

    if tuple(sorted(axis_order)) != tuple(sorted(AXIS_LABELS)):
        raise ValueError("axis_order must contain x, y, and z exactly once")
    frozen = (
        mixed_layer_centers(
            packet_parameters,
            relative_dimension=relative_dimension,
            spin_symmetrize=spin_symmetrize,
        )
        if centers is None
        else centers
    )
    if np.asarray(frozen.pauli_means).shape != (3,):
        raise ValueError("frozen pauli means must have shape (3,)")
    packets = _transformed_generalized_packets(
        packet_parameters,
        mixed_coordinates,
        relative_dimension=relative_dimension,
        centers=frozen,
        axis_order=axis_order,
        spin_symmetrize=spin_symmetrize,
    )
    unnormalized = _packets_to_state(
        packets,
        relative_dimension=relative_dimension,
    )
    norm = float(np.linalg.norm(unnormalized))
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError("mixed layer produced a zero or non-finite state")
    normalized = unnormalized / norm
    reference = _packets_to_state(
        _initial_generalized_packets(
            packet_parameters,
            relative_dimension=relative_dimension,
            spin_symmetrize=spin_symmetrize,
        ),
        relative_dimension=relative_dimension,
    )
    reference /= np.linalg.norm(reference)
    overlap = complex(np.vdot(reference, normalized))
    if abs(overlap) <= np.finfo(float).tiny:
        raise ValueError("mixed layer lost its local phase anchor")
    normalized *= np.exp(-1j * np.angle(overlap))
    return MixedLayerEvaluation(
        state=np.asarray(normalized, dtype=complex),
        unnormalized_state=np.asarray(unnormalized, dtype=complex),
        branch_count=len(packets),
        unnormalized_norm=norm,
        phase_overlap=overlap,
    )


def _unique_centers(
    values: list[complex],
    *,
    relative_tolerance: float = 1e-12,
) -> list[complex]:
    result: list[complex] = []
    for value in values:
        if not any(
            abs(value - current)
            <= relative_tolerance * max(1.0, abs(value), abs(current))
            for current in result
        ):
            result.append(value)
    return result


def _compress_branch_to_candidate_centers(
    target: ComplexArray,
    centers: list[complex],
    *,
    relative_dimension: int,
    relative_tolerance: float,
) -> tuple[ComplexArray, ComplexArray]:
    target_norm = float(np.linalg.norm(target))
    if target_norm <= np.finfo(float).tiny:
        return (
            np.zeros(1, dtype=complex),
            np.zeros(1, dtype=complex),
        )
    unique = _unique_centers(centers)
    if not unique:
        raise ValueError("nonzero electronic branch has no packet centers")
    packets = np.column_stack(
        tuple(
            _projected_infinite_coherent(alpha, relative_dimension)
            / np.linalg.norm(
                _projected_infinite_coherent(alpha, relative_dimension)
            )
            for alpha in unique
        )
    )
    _, _, pivots = qr(packets, mode="economic", pivoting=True)
    maximum_rank = min(relative_dimension, packets.shape[1])
    selected_coefficients: ComplexArray | None = None
    selected_centers: ComplexArray | None = None
    for rank in range(1, maximum_rank + 1):
        indices = np.asarray(pivots[:rank], dtype=int)
        selected = packets[:, indices]
        coefficients = np.linalg.lstsq(
            selected,
            target,
            rcond=1e-13,
        )[0]
        residual = float(np.linalg.norm(selected @ coefficients - target))
        selected_coefficients = np.asarray(coefficients, dtype=complex)
        selected_centers = np.asarray(
            [unique[index] for index in indices],
            dtype=complex,
        )
        if residual <= relative_tolerance * target_norm:
            break
    if selected_coefficients is None or selected_centers is None:
        raise RuntimeError("packet compression selected no coherent centers")
    return selected_coefficients, selected_centers


def retract_mixed_exponential_layer(
    packet_parameters: FloatArray,
    mixed_coordinates: FloatArray,
    *,
    relative_dimension: int,
    centers: FrozenMixedCenters | None = None,
    relative_tolerance: float = 1e-10,
    axis_order: tuple[str, str, str] = AXIS_LABELS,
    spin_symmetrize: bool = True,
) -> MixedLayerRetraction:
    """Compress one exact packet-union endpoint without a packet-count cap.

    Candidate coherent centers come from the analytic mixed layer itself.
    A target-specific rank-revealing selection retains as many centers as are
    needed to meet ``relative_tolerance``, up to the cutoff-space dimension.
    """

    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    if tuple(sorted(axis_order)) != tuple(sorted(AXIS_LABELS)):
        raise ValueError("axis_order must contain x, y, and z exactly once")
    frozen = (
        mixed_layer_centers(
            packet_parameters,
            relative_dimension=relative_dimension,
            spin_symmetrize=spin_symmetrize,
        )
        if centers is None
        else centers
    )
    evaluation = mixed_exponential_layer_state(
        packet_parameters,
        mixed_coordinates,
        relative_dimension=relative_dimension,
        centers=frozen,
        axis_order=axis_order,
        spin_symmetrize=spin_symmetrize,
    )
    generalized = _transformed_generalized_packets(
        packet_parameters,
        mixed_coordinates,
        relative_dimension=relative_dimension,
        centers=frozen,
        axis_order=axis_order,
        spin_symmetrize=spin_symmetrize,
    )
    candidates: list[list[complex]] = [[] for _ in range(4)]
    for electronic, alpha in generalized:
        for electronic_index, coefficient in enumerate(electronic):
            if abs(coefficient) > 1e-14:
                candidates[electronic_index].append(alpha)
    unique_candidates = [_unique_centers(values) for values in candidates]
    target_blocks = evaluation.state.reshape(4, relative_dimension)
    branch_coefficients: list[ComplexArray] = []
    branch_centers: list[ComplexArray] = []
    for electronic_index in range(4):
        coefficients, selected_centers = (
            _compress_branch_to_candidate_centers(
                target_blocks[electronic_index],
                unique_candidates[electronic_index],
                relative_dimension=relative_dimension,
                relative_tolerance=relative_tolerance,
            )
        )
        branch_coefficients.append(coefficients)
        branch_centers.append(selected_centers)
    packet_count = max(values.size for values in branch_coefficients)
    coefficients = np.zeros((4, packet_count), dtype=complex)
    displacements = np.zeros_like(coefficients)
    for electronic_index in range(4):
        count = branch_coefficients[electronic_index].size
        coefficients[electronic_index, :count] = branch_coefficients[
            electronic_index
        ]
        displacements[electronic_index, :count] = branch_centers[
            electronic_index
        ]
        if count < packet_count:
            displacements[electronic_index, count:] = branch_centers[
                electronic_index
            ][0]
    packed = pack_multi_coherent_parameters(coefficients, displacements)
    reconstructed = multi_coherent_state(
        packed,
        relative_dimension=relative_dimension,
    )
    reconstructed_norm = float(np.linalg.norm(reconstructed))
    if reconstructed_norm <= np.finfo(float).tiny:
        raise RuntimeError("mixed-layer retraction produced a zero state")
    coefficients /= reconstructed_norm
    reconstructed /= reconstructed_norm
    overlap = complex(np.vdot(evaluation.state, reconstructed))
    phase = np.exp(-1j * np.angle(overlap))
    coefficients *= phase
    reconstructed *= phase
    packed = pack_multi_coherent_parameters(coefficients, displacements)
    state_error = float(np.linalg.norm(reconstructed - evaluation.state))
    return MixedLayerRetraction(
        parameters=np.asarray(packed, dtype=float),
        packet_count=packet_count,
        candidate_counts=tuple(len(values) for values in unique_candidates),
        retained_counts=tuple(
            int(values.size) for values in branch_coefficients
        ),
        state_error=state_error,
        fidelity=float(abs(np.vdot(evaluation.state, reconstructed)) ** 2),
    )


def mixed_exponential_origin_tangent(
    packet_parameters: FloatArray,
    *,
    relative_dimension: int,
    centers: FrozenMixedCenters | None = None,
    spin_symmetrize: bool = True,
) -> tuple[ComplexArray, ComplexArray]:
    """Return the chart-origin state and its twelve real mixed tangents.

    The layer acts on analytic coherent states before cutoff projection.  In
    particular, the annihilation derivative of one packet is ``alpha`` times
    that packet.  Near the cutoff boundary this is not the same operation as
    applying the finite truncated annihilation matrix to the already projected
    packet.
    """

    zero = np.zeros(12, dtype=float)
    state = mixed_exponential_layer_state(
        packet_parameters,
        zero,
        relative_dimension=relative_dimension,
        centers=centers,
        spin_symmetrize=spin_symmetrize,
    ).state
    frozen = (
        mixed_layer_centers(
            packet_parameters,
            relative_dimension=relative_dimension,
            spin_symmetrize=spin_symmetrize,
        )
        if centers is None
        else centers
    )
    packets = _initial_generalized_packets(
        packet_parameters,
        relative_dimension=relative_dimension,
        spin_symmetrize=spin_symmetrize,
    )
    packet_state = _packets_to_state(
        packets,
        relative_dimension=relative_dimension,
    )
    packet_norm = float(np.linalg.norm(packet_state))
    _, creation = _relative_operators(relative_dimension)
    complex_vectors = []
    for sign in ("minus", "plus"):
        for index, pauli in enumerate(_pauli_matrices()):
            electronic_operator = np.kron(
                pauli,
                np.eye(2, dtype=complex),
            )
            electronic_operator -= float(frozen.pauli_means[index]) * np.eye(
                4,
                dtype=complex,
            )
            derivative = np.zeros((4, relative_dimension), dtype=complex)
            for electronic, alpha in packets:
                coherent = _projected_infinite_coherent(
                    alpha,
                    relative_dimension,
                )
                if sign == "minus":
                    boson_vector = (
                        alpha - frozen.relative_amplitude
                    ) * coherent
                else:
                    boson_vector = (
                        creation @ coherent
                        - frozen.relative_amplitude.conjugate() * coherent
                    )
                derivative += (
                    electronic_operator @ electronic
                )[:, None] * boson_vector[None, :]
            vector = derivative.reshape(-1) / packet_norm
            vector -= state * np.vdot(state, vector)
            complex_vectors.append(vector)
    complex_frame = np.column_stack(complex_vectors)
    tangent = np.column_stack((complex_frame, 1j * complex_frame))
    return state, np.asarray(tangent, dtype=complex)


__all__ = [
    "AXIS_LABELS",
    "MIXED_COORDINATE_LABELS",
    "FrozenMixedCenters",
    "MixedLayerEvaluation",
    "MixedLayerRetraction",
    "mixed_exponential_layer_state",
    "mixed_exponential_origin_tangent",
    "mixed_layer_centers",
    "pack_mixed_coordinates",
    "retract_mixed_exponential_layer",
    "unpack_mixed_coordinates",
]
