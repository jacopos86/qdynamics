r"""Autonomous electron-conditioned multi-coherent relative-mode ansatz.

The two-site Holstein Hamiltonian separates into a driven electron--relative
phonon problem and an electron-independent center oscillator.  This module
represents the joint electron--relative ket as

.. math::

   |\psi(\theta)\rangle =
   \sum_{s=0}^{3}\sum_{k=0}^{K-1}
   c_{sk}|s\rangle|\alpha_{sk}\rangle,

and obtains an autonomous real parameter velocity by projecting the
Schrodinger velocity onto the ansatz tangent space.  Exact trajectories are
not inputs to the right-hand side; they are used only by separate validation
adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from typing import Literal

import numpy as np

from .conditional_packets import fit_coherent_packet_span
from .hubbard_dimer import DimerParameters, GaussianSineDrive
from .moment_hierarchy import MomentHierarchy, PAULI_LABELS


@dataclass(frozen=True)
class MultiCoherentFit:
    """Best coherent-span fit for all four electronic branches."""

    parameters: np.ndarray
    packet_count: int
    fidelity: float
    block_fidelities: tuple[float, ...]
    function_evaluations: int


@dataclass(frozen=True)
class TangentProjection:
    """McLachlan projection of one Schrodinger velocity."""

    parameter_velocity: np.ndarray
    projected_velocity: np.ndarray
    target_velocity: np.ndarray
    absolute_residual: float
    relative_residual: float
    tangent_rank: int
    geometric_tangent_rank: int
    geometric_gram_relative_threshold: float
    largest_singular_value: float
    smallest_retained_singular_value: float
    parameter_velocity_norm: float
    regularization: str
    relative_regularization: float


@dataclass(frozen=True)
class SymmetricGeneratorKick:
    """A normalized symmetric pair obtained inside the model chart."""

    plus_parameters: np.ndarray
    minus_parameters: np.ndarray
    projected_direction_norm: float
    parameter_direction_norm: float
    projection_relative_residual: float


@dataclass(frozen=True)
class PacketSpawn:
    """State-continuous addition of one packet per electronic branch."""

    parameters: np.ndarray
    previous_packet_count: int
    packet_count: int
    parent_electronic_index: int
    centers: tuple[complex, ...]
    residual_block_norms: tuple[float, ...]
    fit_fidelities: tuple[float, ...]
    fit_successes: tuple[bool, ...]
    function_evaluations: int
    state_discontinuity: float
    norm_change: float


@dataclass(frozen=True)
class MultiCoherentObservables:
    """Compact physical observables reconstructed from the ansatz ket."""

    norm: float
    electron_density: np.ndarray
    relative_position: float
    relative_momentum: float
    relative_population: float
    energy: float


@dataclass(frozen=True)
class MultiCoherentCapacity:
    """Unambiguous capacity counts for the rectangular branch ansatz."""

    packets_per_electronic_branch: int
    total_branch_packets: int
    raw_coordinate_count: int


def _oscillator_operators(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    if dimension < 2:
        raise ValueError("relative_dimension must be at least two")
    annihilation = np.diag(
        np.sqrt(np.arange(1, dimension, dtype=float)),
        1,
    ).astype(complex)
    return annihilation, annihilation.conj().T


@lru_cache(maxsize=None)
def _relative_moment_operators(
    relative_dimension: int,
    maximum_degree: int,
) -> tuple[np.ndarray, ...]:
    """Build the hierarchy's symmetric Pauli--Weyl operators."""

    hierarchy = MomentHierarchy(maximum_degree)
    annihilation, creation = _oscillator_operators(relative_dimension)
    position = (annihilation + creation) / sqrt(2.0)
    momentum = (annihilation - creation) / (1j * sqrt(2.0))
    identity = np.eye(relative_dimension, dtype=complex)
    weyl: dict[tuple[int, int], np.ndarray] = {(0, 0): identity}
    required = sorted(
        {(key.x_power, key.p_power) for key in hierarchy.moment_keys},
        key=lambda powers: (sum(powers), powers),
    )
    for x_power, p_power in required:
        degree = x_power + p_power
        if degree == 0:
            continue
        value = np.zeros_like(identity)
        if x_power:
            value += (
                x_power
                / degree
                * position
                @ weyl[(x_power - 1, p_power)]
            )
        if p_power:
            value += (
                p_power
                / degree
                * momentum
                @ weyl[(x_power, p_power - 1)]
            )
        weyl[(x_power, p_power)] = 0.5 * (value + value.conj().T)

    pauli = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    }
    if tuple(pauli) != PAULI_LABELS:
        raise RuntimeError("Pauli ordering mismatch")
    operators: list[np.ndarray] = []
    for key in hierarchy.moment_keys:
        electronic = np.kron(pauli[key.spin_up], pauli[key.spin_down])
        if key.spin_up != key.spin_down:
            electronic = 0.5 * (
                electronic
                + np.kron(pauli[key.spin_down], pauli[key.spin_up])
            )
        operators.append(
            np.kron(electronic, weyl[(key.x_power, key.p_power)])
        )
    return tuple(operators)


def relative_holstein_hamiltonian(
    time: float,
    parameters: DimerParameters,
    *,
    relative_dimension: int,
    drive_protocol: GaussianSineDrive | None = None,
) -> np.ndarray:
    """Return the exact electron--relative-mode Hamiltonian matrix.

    The omitted center mode is electron independent because the calculation
    fixes the total electron number to two.  Starting in its displaced ground
    state, it contributes only a common phase and never affects retained
    electron or relative-mode observables.
    """

    annihilation, creation = _oscillator_operators(relative_dimension)
    oscillator_identity = np.eye(relative_dimension, dtype=complex)
    site_identity = np.eye(2, dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    electron_hamiltonian = -parameters.hopping * (
        np.kron(sigma_x, site_identity)
        + np.kron(site_identity, sigma_x)
    )
    occupation_imbalance = np.diag([2.0, 0.0, 0.0, -2.0]).astype(
        complex
    )
    relative_hamiltonian = parameters.omega_ph * (creation @ annihilation)
    drive = (
        parameters.drive_difference(time)
        if drive_protocol is None
        else drive_protocol.difference(time)
    )
    return (
        np.kron(electron_hamiltonian, oscillator_identity)
        + np.kron(np.eye(4), relative_hamiltonian)
        + parameters.coupling
        / sqrt(2.0)
        * np.kron(occupation_imbalance, annihilation + creation)
        + 0.5
        * drive
        * np.kron(occupation_imbalance, oscillator_identity)
    )


def pack_multi_coherent_parameters(
    coefficients: np.ndarray,
    displacements: np.ndarray,
) -> np.ndarray:
    """Pack complex ``[electron, packet]`` arrays into real coordinates."""

    coefficient_array = np.asarray(coefficients, dtype=complex)
    displacement_array = np.asarray(displacements, dtype=complex)
    if coefficient_array.ndim != 2 or coefficient_array.shape[0] != 4:
        raise ValueError("coefficients must have shape (4, packet_count)")
    if displacement_array.shape != coefficient_array.shape:
        raise ValueError("displacements must match coefficients")
    packed = np.empty((*coefficient_array.shape, 4), dtype=float)
    packed[..., 0] = coefficient_array.real
    packed[..., 1] = coefficient_array.imag
    packed[..., 2] = displacement_array.real
    packed[..., 3] = displacement_array.imag
    return packed.reshape(-1)


def unpack_multi_coherent_parameters(
    parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return complex coefficients and displacements from real coordinates."""

    vector = np.asarray(parameters, dtype=float)
    if vector.ndim != 1 or vector.size % 16 != 0:
        raise ValueError("parameter vector must contain 16*K real values")
    packet_count = vector.size // 16
    packed = vector.reshape(4, packet_count, 4)
    coefficients = packed[..., 0] + 1j * packed[..., 1]
    displacements = packed[..., 2] + 1j * packed[..., 3]
    return coefficients, displacements


def multi_coherent_capacity(parameters: np.ndarray) -> MultiCoherentCapacity:
    """Return packet and coordinate counts without conflating conventions."""

    coefficients, _ = unpack_multi_coherent_parameters(parameters)
    packets_per_branch = int(coefficients.shape[1])
    return MultiCoherentCapacity(
        packets_per_electronic_branch=packets_per_branch,
        total_branch_packets=4 * packets_per_branch,
        raw_coordinate_count=int(np.asarray(parameters).size),
    )


def retract_multi_coherent_parameters(
    parameters: np.ndarray,
    *,
    relative_dimension: int,
) -> np.ndarray:
    """Normalize the represented ket and fix one deterministic global phase."""

    coefficients, displacements = unpack_multi_coherent_parameters(parameters)
    coefficients = coefficients.copy()
    raw_state = multi_coherent_state(
        parameters,
        relative_dimension=relative_dimension,
    )
    norm = float(np.linalg.norm(raw_state))
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError("cannot retract a zero or non-finite state")
    if abs(norm - 1.0) > 1e-14:
        coefficients /= norm
    flat = coefficients.reshape(-1)
    anchor_index = int(np.argmax(np.abs(flat)))
    anchor = flat[anchor_index]
    if abs(anchor.imag) > 1e-14 or anchor.real < 0.0:
        coefficients *= np.exp(-1j * np.angle(anchor))
        flat = coefficients.reshape(-1)
        flat[anchor_index] = complex(abs(flat[anchor_index]), 0.0)
    return pack_multi_coherent_parameters(coefficients, displacements)


def _coherent_packet_and_derivatives(
    alpha: complex,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.empty(dimension, dtype=complex)
    # The Gaussian prefactor is common to every retained Fock amplitude and
    # cancels under cutoff-space normalization.  Omitting it avoids numerical
    # underflow for displaced packets without changing the normalized ket or
    # its horizontal derivatives.
    raw[0] = 1.0
    for occupation in range(1, dimension):
        raw[occupation] = (
            raw[occupation - 1] * alpha / np.sqrt(float(occupation))
        )
    _, creation = _oscillator_operators(dimension)
    derivative_real_raw = creation @ raw - alpha.real * raw
    derivative_imag_raw = 1j * (creation @ raw) - alpha.imag * raw
    norm = float(np.linalg.norm(raw))
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError("coherent packet has zero or non-finite norm")
    packet = raw / norm

    def normalized_derivative(raw_derivative: np.ndarray) -> np.ndarray:
        derivative = raw_derivative / norm
        return derivative - packet * float(np.vdot(packet, derivative).real)

    return (
        packet,
        normalized_derivative(derivative_real_raw),
        normalized_derivative(derivative_imag_raw),
    )


def multi_coherent_state_and_tangent(
    parameters: np.ndarray,
    *,
    relative_dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the ansatz ket and its complex-by-real tangent matrix."""

    coefficients, displacements = unpack_multi_coherent_parameters(parameters)
    packet_count = coefficients.shape[1]
    state = np.zeros(4 * relative_dimension, dtype=complex)
    tangent = np.zeros((state.size, parameters.size), dtype=complex)
    for electronic_index in range(4):
        block = slice(
            electronic_index * relative_dimension,
            (electronic_index + 1) * relative_dimension,
        )
        for packet_index in range(packet_count):
            coefficient = coefficients[electronic_index, packet_index]
            packet, derivative_real, derivative_imag = (
                _coherent_packet_and_derivatives(
                    displacements[electronic_index, packet_index],
                    relative_dimension,
                )
            )
            state[block] += coefficient * packet
            column = 4 * (
                electronic_index * packet_count + packet_index
            )
            tangent[block, column] = packet
            tangent[block, column + 1] = 1j * packet
            tangent[block, column + 2] = coefficient * derivative_real
            tangent[block, column + 3] = coefficient * derivative_imag
    return state, tangent


def multi_coherent_state(
    parameters: np.ndarray,
    *,
    relative_dimension: int,
) -> np.ndarray:
    """Reconstruct only the joint electron--relative ket."""

    state, _ = multi_coherent_state_and_tangent(
        parameters,
        relative_dimension=relative_dimension,
    )
    return state


def normalized_multi_coherent_state_and_horizontal_tangent(
    parameters: np.ndarray,
    *,
    relative_dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the normalized ket and its projective real-coordinate tangent.

    For the unnormalized reconstruction ``phi(theta)``, the returned columns
    are

    .. math::

       T_j = \frac{(I-|\psi\rangle\langle\psi|)
                    \partial_j|\phi\rangle}{\|\phi\|},
       \qquad |\psi\rangle=|\phi\rangle/\|\phi\|.

    Thus every column is orthogonal to ``psi`` and changes neither norm nor
    global phase to first order.  The coordinates remain the existing real
    coefficient/displacement coordinates; this function only supplies their
    physical projective tangent.
    """

    raw_state, raw_tangent = multi_coherent_state_and_tangent(
        parameters,
        relative_dimension=relative_dimension,
    )
    norm = float(np.linalg.norm(raw_state))
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError("multi-coherent state has zero or non-finite norm")
    state = raw_state / norm
    overlaps = state.conj() @ raw_tangent
    horizontal = (
        raw_tangent - state[:, np.newaxis] * overlaps[np.newaxis, :]
    ) / norm
    return state, horizontal


def fit_coherent_electron_relative_state(
    state: np.ndarray,
    *,
    packet_count: int,
    maximum_iterations: int = 100,
    population_size: int = 9,
    seed: int = 0,
) -> MultiCoherentFit:
    """Fit a coherent packet span independently in each electronic branch."""

    target = np.asarray(state, dtype=complex)
    if target.ndim != 1 or target.size % 4 != 0:
        raise ValueError("state size must be divisible by four")
    target = target / np.linalg.norm(target)
    if packet_count < 1:
        raise ValueError("packet_count must be positive")
    relative_dimension = target.size // 4
    blocks = target.reshape(4, relative_dimension)
    coefficients = np.zeros((4, packet_count), dtype=complex)
    displacements = np.zeros((4, packet_count), dtype=complex)
    block_fidelities: list[float] = []
    function_evaluations = 0
    for electronic_index, block in enumerate(blocks):
        probability = float(np.vdot(block, block).real)
        if probability <= np.finfo(float).tiny:
            displacements[electronic_index] = (
                1e-3 * np.arange(packet_count)
            )
            block_fidelities.append(1.0)
            continue
        normalized_block = block / np.sqrt(probability)
        fit = fit_coherent_packet_span(
            normalized_block,
            packet_count=packet_count,
            maximum_iterations=maximum_iterations,
            population_size=population_size,
            seed=seed + 17 * electronic_index,
        )
        function_evaluations += fit.function_evaluations
        centers = np.asarray(
            [
                complex(
                    fit.parameters[2 * packet_index],
                    fit.parameters[2 * packet_index + 1],
                )
                for packet_index in range(packet_count)
            ]
        )
        packets = np.column_stack(
            [
                _coherent_packet_and_derivatives(
                    center,
                    relative_dimension,
                )[0]
                for center in centers
            ]
        )
        coefficients[electronic_index] = np.linalg.lstsq(
            packets,
            block,
            rcond=1e-11,
        )[0]
        displacements[electronic_index] = centers
        block_fidelities.append(fit.fidelity)
    packed = pack_multi_coherent_parameters(coefficients, displacements)
    fitted = multi_coherent_state(
        packed,
        relative_dimension=relative_dimension,
    )
    fitted_norm = float(np.linalg.norm(fitted))
    coefficients /= fitted_norm
    packed = pack_multi_coherent_parameters(coefficients, displacements)
    fitted = multi_coherent_state(
        packed,
        relative_dimension=relative_dimension,
    )
    fidelity = float(abs(np.vdot(target, fitted)) ** 2)
    return MultiCoherentFit(
        parameters=packed,
        packet_count=packet_count,
        fidelity=fidelity,
        block_fidelities=tuple(block_fidelities),
        function_evaluations=function_evaluations,
    )


def fit_two_coherent_electron_relative_state(
    state: np.ndarray,
    *,
    maximum_iterations: int = 100,
    population_size: int = 9,
    seed: int = 0,
) -> MultiCoherentFit:
    """Fit two coherent packets independently in each electronic branch."""

    return fit_coherent_electron_relative_state(
        state,
        packet_count=2,
        maximum_iterations=maximum_iterations,
        population_size=population_size,
        seed=seed,
    )


def project_schrodinger_velocity(
    parameters: np.ndarray,
    hamiltonian: np.ndarray,
    *,
    relative_dimension: int,
    target_state: np.ndarray | None = None,
    relative_singular_value_cutoff: float = 1e-9,
    regularization: Literal["truncated_svd", "tikhonov"] = "truncated_svd",
    relative_damping: float = 1e-2,
    geometric_gram_relative_threshold: float = 1e-10,
) -> TangentProjection:
    """Project a phase-fixed Schrodinger velocity onto the ansatz tangent."""

    if relative_singular_value_cutoff <= 0.0:
        raise ValueError("relative_singular_value_cutoff must be positive")
    if relative_damping <= 0.0:
        raise ValueError("relative_damping must be positive")
    if not 0.0 < geometric_gram_relative_threshold < 1.0:
        raise ValueError(
            "geometric_gram_relative_threshold must lie between zero and one"
        )
    if regularization not in ("truncated_svd", "tikhonov"):
        raise ValueError("unknown tangent regularization")
    ansatz_state, tangent = normalized_multi_coherent_state_and_horizontal_tangent(
        parameters,
        relative_dimension=relative_dimension,
    )
    state = (
        ansatz_state
        if target_state is None
        else np.asarray(target_state, dtype=complex)
    )
    if state.shape != ansatz_state.shape:
        raise ValueError("target_state shape must match the ansatz state")
    state = state / np.linalg.norm(state)
    energy = float(np.vdot(state, hamiltonian @ state).real)
    target_velocity = -1j * (hamiltonian @ state - energy * state)
    real_tangent = np.vstack((tangent.real, tangent.imag))
    real_target = np.concatenate((target_velocity.real, target_velocity.imag))
    left, singular_values, right_adjoint = np.linalg.svd(
        real_tangent,
        full_matrices=False,
    )
    geometric_threshold = (
        np.sqrt(geometric_gram_relative_threshold) * singular_values[0]
    )
    geometric_tangent_rank = int(
        np.count_nonzero(singular_values > geometric_threshold)
    )
    if regularization == "truncated_svd":
        relative_regularization = relative_singular_value_cutoff
        threshold = relative_singular_value_cutoff * singular_values[0]
        inverse = np.where(singular_values > threshold, 1.0 / singular_values, 0.0)
        retained = singular_values[singular_values > threshold]
    else:
        relative_regularization = relative_damping
        damping = relative_damping * singular_values[0]
        inverse = singular_values / (singular_values**2 + damping**2)
        retained = singular_values[singular_values > damping]
    parameter_velocity = right_adjoint.T @ (
        inverse * (left.T @ real_target)
    )
    projected = tangent @ parameter_velocity
    residual = projected - target_velocity
    absolute_residual = float(np.linalg.norm(residual))
    target_norm = float(np.linalg.norm(target_velocity))
    if retained.size == 0:
        retained = singular_values[:1]
    return TangentProjection(
        parameter_velocity=np.asarray(parameter_velocity, dtype=float),
        projected_velocity=projected,
        target_velocity=target_velocity,
        absolute_residual=absolute_residual,
        relative_residual=float(
            absolute_residual / max(target_norm, np.finfo(float).tiny)
        ),
        tangent_rank=int(retained.size),
        geometric_tangent_rank=geometric_tangent_rank,
        geometric_gram_relative_threshold=float(
            geometric_gram_relative_threshold
        ),
        largest_singular_value=float(singular_values[0]),
        smallest_retained_singular_value=float(retained[-1]),
        parameter_velocity_norm=float(np.linalg.norm(parameter_velocity)),
        regularization=regularization,
        relative_regularization=float(relative_regularization),
    )


def multi_coherent_rhs(
    time: float,
    state: np.ndarray,
    parameters: DimerParameters,
    *,
    relative_dimension: int,
    drive_protocol: GaussianSineDrive | None = None,
    relative_singular_value_cutoff: float = 1e-9,
    regularization: Literal["truncated_svd", "tikhonov"] = "truncated_svd",
    relative_damping: float = 1e-2,
) -> np.ndarray:
    """Return the autonomous McLachlan parameter velocity."""

    hamiltonian = relative_holstein_hamiltonian(
        time,
        parameters,
        relative_dimension=relative_dimension,
        drive_protocol=drive_protocol,
    )
    return project_schrodinger_velocity(
        state,
        hamiltonian,
        relative_dimension=relative_dimension,
        relative_singular_value_cutoff=relative_singular_value_cutoff,
        regularization=regularization,
        relative_damping=relative_damping,
    ).parameter_velocity


def symmetric_projected_generator_kick(
    parameters: np.ndarray,
    generator: np.ndarray,
    *,
    relative_dimension: int,
    step: float = 1e-4,
    relative_singular_value_cutoff: float = 1e-9,
    regularization: Literal["truncated_svd", "tikhonov"] = "truncated_svd",
    relative_damping: float = 1e-2,
) -> SymmetricGeneratorKick:
    """Project ``-i(G-<G>)psi`` and retract a symmetric chart pair."""

    if step <= 0.0:
        raise ValueError("kick step must be positive")
    matrix = np.asarray(generator, dtype=complex)
    expected = (4 * relative_dimension, 4 * relative_dimension)
    if matrix.shape != expected:
        raise ValueError(f"generator must have shape {expected}")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12, rtol=0.0):
        raise ValueError("generator must be Hermitian")
    projection = project_schrodinger_velocity(
        parameters,
        matrix,
        relative_dimension=relative_dimension,
        relative_singular_value_cutoff=relative_singular_value_cutoff,
        regularization=regularization,
        relative_damping=relative_damping,
    )
    projected_norm = float(np.linalg.norm(projection.projected_velocity))
    if not np.isfinite(projected_norm) or projected_norm <= 1e-12:
        raise ValueError("projected generator direction is unresolved")
    parameter_direction = projection.parameter_velocity / projected_norm
    parameter_array = np.asarray(parameters, dtype=float)
    plus = retract_multi_coherent_parameters(
        parameter_array + step * parameter_direction,
        relative_dimension=relative_dimension,
    )
    minus = retract_multi_coherent_parameters(
        parameter_array - step * parameter_direction,
        relative_dimension=relative_dimension,
    )
    return SymmetricGeneratorKick(
        plus_parameters=plus,
        minus_parameters=minus,
        projected_direction_norm=projected_norm,
        parameter_direction_norm=float(np.linalg.norm(parameter_direction)),
        projection_relative_residual=projection.relative_residual,
    )


def normalized_diagonal_kick_generator(
    parameters: np.ndarray,
    *,
    relative_dimension: int,
) -> np.ndarray:
    """Return the centered unit-variance electronic/relative diagonal kick."""

    state, _ = normalized_multi_coherent_state_and_horizontal_tangent(
        parameters,
        relative_dimension=relative_dimension,
    )
    identity = np.eye(state.size, dtype=complex)

    def centered_unit_variance(operator: np.ndarray) -> np.ndarray:
        mean = float(np.vdot(state, operator @ state).real)
        centered = operator - mean * identity
        variance = float(np.vdot(state, centered @ centered @ state).real)
        if not np.isfinite(variance) or variance <= 1e-14:
            raise ValueError("kick generator has unresolved variance")
        return centered / np.sqrt(variance)

    oscillator_identity = np.eye(relative_dimension, dtype=complex)
    electronic_imbalance = 0.5 * np.kron(
        np.diag([2.0, 0.0, 0.0, -2.0]),
        oscillator_identity,
    )
    annihilation, creation = _oscillator_operators(relative_dimension)
    relative_position = np.kron(
        np.eye(4, dtype=complex),
        annihilation + creation,
    )
    combined = centered_unit_variance(
        electronic_imbalance
    ) + centered_unit_variance(relative_position)
    return centered_unit_variance(combined)


def spawn_residual_coherent_packets(
    parameters: np.ndarray,
    residual_velocity: np.ndarray,
    *,
    relative_dimension: int,
    maximum_iterations: int = 40,
    population_size: int = 6,
    seed: int = 0,
) -> PacketSpawn:
    """Append zero-weight packets fitted to the current tangent residual.

    One coherent center is selected independently for each electronic branch
    using only the supplied McLachlan residual.  Every new coefficient is zero,
    so the represented state is unchanged at the spawn while the coefficient
    tangent directions enlarge the variational space.
    """

    if relative_dimension < 2:
        raise ValueError("relative_dimension must be at least two")
    if maximum_iterations < 1 or population_size < 1:
        raise ValueError("spawn fit controls must be positive")
    coefficients, displacements = unpack_multi_coherent_parameters(parameters)
    residual = np.asarray(residual_velocity, dtype=complex)
    expected_shape = (4 * relative_dimension,)
    if residual.shape != expected_shape:
        raise ValueError(
            f"expected residual shape {expected_shape}, got {residual.shape}"
        )

    residual_blocks = residual.reshape(4, relative_dimension)
    residual_norms = np.linalg.norm(residual_blocks, axis=1)
    centers: list[complex] = []
    fidelities: list[float] = []
    successes: list[bool] = []
    function_evaluations = 0
    for electronic_index, block in enumerate(residual_blocks):
        if residual_norms[electronic_index] <= np.finfo(float).eps:
            center = complex(
                np.mean(displacements[electronic_index])
                + (1.0 + 1.0j)
                * (1e-3 * (electronic_index + 1))
            )
            fidelity = 0.0
            success = True
        else:
            fit = fit_coherent_packet_span(
                block,
                packet_count=1,
                maximum_iterations=maximum_iterations,
                population_size=population_size,
                seed=seed + electronic_index,
            )
            center = complex(fit.parameters[0], fit.parameters[1])
            fidelity = fit.fidelity
            success = fit.success
            function_evaluations += fit.function_evaluations
        centers.append(center)
        fidelities.append(float(fidelity))
        successes.append(bool(success))

    previous_packet_count = coefficients.shape[1]
    expanded_coefficients = np.pad(coefficients, ((0, 0), (0, 1)))
    expanded_displacements = np.pad(displacements, ((0, 0), (0, 1)))
    expanded_displacements[:, -1] = np.asarray(centers)
    expanded_parameters = pack_multi_coherent_parameters(
        expanded_coefficients,
        expanded_displacements,
    )
    state_before = multi_coherent_state(
        parameters,
        relative_dimension=relative_dimension,
    )
    state_after = multi_coherent_state(
        expanded_parameters,
        relative_dimension=relative_dimension,
    )
    norm_before = float(np.vdot(state_before, state_before).real)
    norm_after = float(np.vdot(state_after, state_after).real)
    return PacketSpawn(
        parameters=expanded_parameters,
        previous_packet_count=previous_packet_count,
        packet_count=previous_packet_count + 1,
        parent_electronic_index=int(np.argmax(residual_norms)),
        centers=tuple(centers),
        residual_block_norms=tuple(float(value) for value in residual_norms),
        fit_fidelities=tuple(fidelities),
        fit_successes=tuple(successes),
        function_evaluations=function_evaluations,
        state_discontinuity=float(np.linalg.norm(state_after - state_before)),
        norm_change=float(abs(norm_after - norm_before)),
    )


def multi_coherent_observables(
    time: float,
    state: np.ndarray,
    parameters: DimerParameters,
    *,
    relative_dimension: int,
    drive_protocol: GaussianSineDrive | None = None,
) -> MultiCoherentObservables:
    """Contract norm, one-body electron density, relative moments, and energy."""

    vector = multi_coherent_state(
        state,
        relative_dimension=relative_dimension,
    )
    norm = float(np.vdot(vector, vector).real)
    normalized = vector / np.sqrt(norm)
    blocks = normalized.reshape(2, 2, relative_dimension)
    electron_density = np.einsum(
        "adr,bdr->ab",
        blocks,
        blocks.conj(),
        optimize=True,
    )
    annihilation, creation = _oscillator_operators(relative_dimension)
    relative_density = np.einsum(
        "adr,ads->rs",
        blocks,
        blocks.conj(),
        optimize=True,
    )
    position = (annihilation + creation) / sqrt(2.0)
    momentum = (annihilation - creation) / (1j * sqrt(2.0))
    hamiltonian = relative_holstein_hamiltonian(
        time,
        parameters,
        relative_dimension=relative_dimension,
        drive_protocol=drive_protocol,
    )
    return MultiCoherentObservables(
        norm=norm,
        electron_density=electron_density,
        relative_position=float(np.trace(relative_density @ position).real),
        relative_momentum=float(np.trace(relative_density @ momentum).real),
        relative_population=float(
            np.trace(relative_density @ creation @ annihilation).real
        ),
        energy=float(np.vdot(normalized, hamiltonian @ normalized).real),
    )


def relative_state_moment_coordinates(
    state: np.ndarray,
    hierarchy: MomentHierarchy,
    *,
    center_amplitude: complex,
) -> np.ndarray:
    """Contract one electron--relative ket into hierarchy coordinates."""

    vector = np.asarray(state, dtype=complex)
    if vector.ndim != 1 or vector.size % 4 != 0:
        raise ValueError("state size must be divisible by four")
    vector = vector / np.linalg.norm(vector)
    operators = _relative_moment_operators(
        vector.size // 4,
        hierarchy.maximum_degree,
    )
    moments = {
        key: float(np.vdot(vector, operator @ vector).real)
        for key, operator in zip(
            hierarchy.moment_keys,
            operators,
            strict=True,
        )
    }
    return hierarchy.pack(center_amplitude, moments)


def relative_state_closed_coordinates(
    state: np.ndarray,
    hierarchy: MomentHierarchy,
    *,
    center_amplitude: complex,
) -> np.ndarray:
    """Contract an electron--relative ket into the established 31 slots."""

    from .matrix_reference import matrix_state_to_closed_scalar_coordinates

    hierarchy_coordinates = relative_state_moment_coordinates(
        state,
        hierarchy,
        center_amplitude=center_amplitude,
    )
    return matrix_state_to_closed_scalar_coordinates(
        hierarchy.to_matrix_state(hierarchy_coordinates)
    )


def relative_state_moment_derivative(
    state: np.ndarray,
    state_derivative: np.ndarray,
    hierarchy: MomentHierarchy,
    *,
    center_derivative: complex = 0.0j,
) -> np.ndarray:
    """Contract a ket and tangent velocity into hierarchy moment velocities."""

    vector = np.asarray(state, dtype=complex)
    derivative = np.asarray(state_derivative, dtype=complex)
    if vector.shape != derivative.shape or vector.size % 4 != 0:
        raise ValueError("state and derivative shapes must match and divide by four")
    norm = float(np.linalg.norm(vector))
    vector = vector / norm
    derivative = derivative / norm
    derivative -= vector * float(np.vdot(vector, derivative).real)
    operators = _relative_moment_operators(
        vector.size // 4,
        hierarchy.maximum_degree,
    )
    derivatives = {
        key: float(
            2.0 * np.vdot(derivative, operator @ vector).real
        )
        for key, operator in zip(
            hierarchy.moment_keys,
            operators,
            strict=True,
        )
    }
    return hierarchy.pack(center_derivative, derivatives)


__all__ = [
    "MultiCoherentFit",
    "MultiCoherentCapacity",
    "MultiCoherentObservables",
    "PacketSpawn",
    "SymmetricGeneratorKick",
    "TangentProjection",
    "fit_coherent_electron_relative_state",
    "fit_two_coherent_electron_relative_state",
    "multi_coherent_observables",
    "multi_coherent_capacity",
    "multi_coherent_rhs",
    "multi_coherent_state",
    "multi_coherent_state_and_tangent",
    "normalized_diagonal_kick_generator",
    "normalized_multi_coherent_state_and_horizontal_tangent",
    "pack_multi_coherent_parameters",
    "project_schrodinger_velocity",
    "retract_multi_coherent_parameters",
    "relative_holstein_hamiltonian",
    "relative_state_moment_coordinates",
    "relative_state_closed_coordinates",
    "relative_state_moment_derivative",
    "spawn_residual_coherent_packets",
    "symmetric_projected_generator_kick",
    "unpack_multi_coherent_parameters",
]
