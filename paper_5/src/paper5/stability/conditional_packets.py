r"""Electron-conditioned relative-phonon packet diagnostics.

The exact two-local-mode state is transformed to center/relative normal modes.
Projecting one electronic site configuration and tracing the center mode gives
the corresponding relative-phonon block.  The module then compares its
dominant pure state with one displaced-squeezed Gaussian packet and with the
optimal span of two coherent packets.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb, factorial, pi

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import differential_evolution, minimize
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply


@dataclass(frozen=True)
class ConditionalRelativeState:
    """One normalized relative-mode state conditioned on an electron basis ket."""

    electronic_index: int
    probability: float
    density_matrix: np.ndarray
    dominant_state: np.ndarray
    center_relative_factorization: float


@dataclass(frozen=True)
class ElectronRelativeState:
    """Global electron--relative-mode ket after removing the center mode."""

    state: np.ndarray
    center_state: np.ndarray
    center_factorization: float


@dataclass(frozen=True)
class LocalProductEmbedding:
    """Local-cutoff image of one center/electron-relative product ket."""

    state: np.ndarray
    retained_norm: float


@dataclass(frozen=True)
class PacketFit:
    """Maximum fidelity with one declared packet family."""

    fidelity: float
    parameters: tuple[float, ...]
    function_evaluations: int
    success: bool


@dataclass(frozen=True)
class ConditionalPacketMetrics:
    """Compression and phase-space metrics for one conditional block."""

    electronic_index: int
    probability: float
    center_relative_factorization: float
    relative_purity: float
    mean_x: float
    mean_p: float
    covariance_xx: float
    covariance_xp: float
    covariance_pp: float
    gaussian_non_gaussianity: float
    husimi_peak_count: int
    husimi_second_peak_ratio: float
    single_gaussian_fit: PacketFit
    two_coherent_fit: PacketFit


@lru_cache(maxsize=None)
def local_to_normal_mode_transform(phonon_cutoff: int) -> csc_matrix:
    """Return the isometry from local Fock modes to center/relative modes."""

    if phonon_cutoff < 1:
        raise ValueError("phonon_cutoff must be at least one")
    local_dimension = phonon_cutoff + 1
    normal_dimension = 2 * phonon_cutoff + 1
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for occupation_0 in range(local_dimension):
        for occupation_1 in range(local_dimension):
            total = occupation_0 + occupation_1
            column = occupation_0 * local_dimension + occupation_1
            for relative_occupation in range(total + 1):
                polynomial = 0
                lower = max(0, relative_occupation - occupation_1)
                upper = min(occupation_0, relative_occupation)
                for from_mode_0 in range(lower, upper + 1):
                    from_mode_1 = relative_occupation - from_mode_0
                    polynomial += (
                        comb(occupation_0, from_mode_0)
                        * comb(occupation_1, from_mode_1)
                        * (-1) ** from_mode_1
                    )
                center_occupation = total - relative_occupation
                coefficient = (
                    2.0 ** (-0.5 * total)
                    * polynomial
                    * np.sqrt(
                        factorial(center_occupation)
                        * factorial(relative_occupation)
                        / (
                            factorial(occupation_0)
                            * factorial(occupation_1)
                        )
                    )
                )
                if abs(coefficient) > 1e-15:
                    rows.append(
                        center_occupation * normal_dimension
                        + relative_occupation
                    )
                    columns.append(column)
                    values.append(float(coefficient))
    shape = (
        normal_dimension * normal_dimension,
        local_dimension * local_dimension,
    )
    return csc_matrix((values, (rows, columns)), shape=shape, dtype=complex)


def electron_relative_product_to_local_state(
    electron_relative: np.ndarray,
    center_state: np.ndarray,
    *,
    phonon_cutoff: int,
) -> LocalProductEmbedding:
    """Project a normalized center/relative product into the local cutoff."""

    normal_dimension = 2 * phonon_cutoff + 1
    relative = np.asarray(electron_relative, dtype=complex)
    center = np.asarray(center_state, dtype=complex)
    if relative.shape != (4 * normal_dimension,):
        raise ValueError("electron_relative has the wrong dimension")
    if center.shape != (normal_dimension,):
        raise ValueError("center_state has the wrong dimension")
    relative_norm = float(np.linalg.norm(relative))
    center_norm = float(np.linalg.norm(center))
    if relative_norm <= np.finfo(float).tiny or center_norm <= np.finfo(float).tiny:
        raise ValueError("product factors must have nonzero norm")
    relative = relative / relative_norm
    center = center / center_norm
    transform_adjoint = local_to_normal_mode_transform(phonon_cutoff).getH()
    relative_blocks = relative.reshape(4, normal_dimension)
    local_blocks = np.asarray(
        [
            transform_adjoint @ np.outer(center, block).reshape(-1)
            for block in relative_blocks
        ]
    )
    local = local_blocks.reshape(-1)
    retained_norm = float(np.linalg.norm(local))
    if retained_norm <= np.finfo(float).tiny:
        raise ValueError("center/relative product has no local-cutoff support")
    return LocalProductEmbedding(
        state=local / retained_norm,
        retained_norm=retained_norm,
    )


def conditional_relative_state(
    state_vector: np.ndarray,
    *,
    electronic_index: int,
    phonon_cutoff: int,
) -> ConditionalRelativeState:
    """Project one electronic ket and trace the decoupled center normal mode."""

    if electronic_index not in range(4):
        raise ValueError("electronic_index must lie in range(4)")
    local_dimension = phonon_cutoff + 1
    expected_size = 4 * local_dimension * local_dimension
    vector = np.asarray(state_vector, dtype=complex)
    if vector.shape != (expected_size,):
        raise ValueError(
            f"expected state shape {(expected_size,)}, got {vector.shape}"
        )
    local_amplitudes = vector.reshape(
        4,
        local_dimension,
        local_dimension,
    )[electronic_index]
    normal_dimension = 2 * phonon_cutoff + 1
    normal_amplitudes = (
        local_to_normal_mode_transform(phonon_cutoff)
        @ local_amplitudes.reshape(-1)
    ).reshape(normal_dimension, normal_dimension)
    probability = float(np.vdot(normal_amplitudes, normal_amplitudes).real)
    if probability <= np.finfo(float).tiny:
        raise ValueError(
            f"electronic configuration {electronic_index} has zero probability"
        )
    density = (
        normal_amplitudes.T @ normal_amplitudes.conj() / probability
    )
    density = 0.5 * (density + density.conj().T)
    _, singular_values, right_adjoint = np.linalg.svd(
        normal_amplitudes,
        full_matrices=False,
    )
    dominant = np.asarray(right_adjoint[0], dtype=complex)
    dominant /= np.linalg.norm(dominant)
    phase_anchor = int(np.argmax(np.abs(dominant)))
    dominant *= np.exp(-1j * np.angle(dominant[phase_anchor]))
    factorization = float(
        singular_values[0] ** 2 / np.sum(singular_values**2)
    )
    return ConditionalRelativeState(
        electronic_index=electronic_index,
        probability=probability,
        density_matrix=density,
        dominant_state=dominant,
        center_relative_factorization=factorization,
    )


def electron_relative_state(
    state_vector: np.ndarray,
    *,
    phonon_cutoff: int,
) -> ElectronRelativeState:
    """Extract the common-center and joint electron--relative factors.

    The local-to-normal-mode transform is applied exactly within the supplied
    local cutoff.  A Schmidt decomposition across ``center | (electron,
    relative)`` then returns the dominant product factors and the fraction of
    the exact norm carried by that product.
    """

    local_dimension = phonon_cutoff + 1
    expected_size = 4 * local_dimension * local_dimension
    vector = np.asarray(state_vector, dtype=complex)
    if vector.shape != (expected_size,):
        raise ValueError(
            f"expected state shape {(expected_size,)}, got {vector.shape}"
        )
    normal_dimension = 2 * phonon_cutoff + 1
    local_blocks = vector.reshape(4, local_dimension, local_dimension)
    normal_blocks = np.asarray(
        [
            (
                local_to_normal_mode_transform(phonon_cutoff)
                @ block.reshape(-1)
            ).reshape(normal_dimension, normal_dimension)
            for block in local_blocks
        ]
    )
    center_by_electron_relative = np.transpose(
        normal_blocks,
        (1, 0, 2),
    ).reshape(normal_dimension, 4 * normal_dimension)
    left, singular_values, right_adjoint = np.linalg.svd(
        center_by_electron_relative,
        full_matrices=False,
    )
    center = np.asarray(left[:, 0], dtype=complex)
    electron_relative = np.asarray(right_adjoint[0], dtype=complex)
    center /= np.linalg.norm(center)
    electron_relative /= np.linalg.norm(electron_relative)
    phase_anchor = int(np.argmax(np.abs(center)))
    phase = np.exp(-1j * np.angle(center[phase_anchor]))
    center *= phase
    electron_relative /= phase
    factorization = float(
        singular_values[0] ** 2 / np.sum(singular_values**2)
    )
    return ElectronRelativeState(
        state=electron_relative,
        center_state=center,
        center_factorization=factorization,
    )


@lru_cache(maxsize=None)
def _mode_operators(dimension: int) -> tuple[csc_matrix, csc_matrix]:
    annihilation = csc_matrix(
        np.diag(np.sqrt(np.arange(1, dimension, dtype=float)), 1)
    )
    return annihilation, annihilation.getH()


def _quadrature_statistics(
    density: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    dimension = density.shape[0]
    annihilation, creation = _mode_operators(dimension)
    position = (annihilation + creation) / np.sqrt(2.0)
    momentum = (annihilation - creation) / (1j * np.sqrt(2.0))
    mean_x = float(np.trace(density @ position).real)
    mean_p = float(np.trace(density @ momentum).real)
    identity = np.eye(dimension, dtype=complex)
    centered_x = position - mean_x * identity
    centered_p = momentum - mean_p * identity
    covariance = np.array(
        [
            [
                np.trace(density @ centered_x @ centered_x).real,
                np.trace(
                    density
                    @ (centered_x @ centered_p + centered_p @ centered_x)
                    / 2.0
                ).real,
            ],
            [
                0.0,
                np.trace(density @ centered_p @ centered_p).real,
            ],
        ],
        dtype=float,
    )
    covariance[1, 0] = covariance[0, 1]
    return mean_x, mean_p, covariance


def _entropy(eigenvalues: np.ndarray) -> float:
    positive = np.asarray(eigenvalues, dtype=float)
    positive = positive[positive > 1e-14]
    return float(-np.sum(positive * np.log(positive)))


def gaussian_non_gaussianity(density: np.ndarray) -> float:
    """Return relative-entropy non-Gaussianity from first and second moments."""

    _, _, covariance = _quadrature_statistics(density)
    symplectic_eigenvalue = np.sqrt(max(float(np.linalg.det(covariance)), 0.25))
    thermal_population = max(symplectic_eigenvalue - 0.5, 0.0)
    gaussian_entropy = (thermal_population + 1.0) * np.log(
        thermal_population + 1.0
    )
    if thermal_population > 1e-15:
        gaussian_entropy -= thermal_population * np.log(thermal_population)
    state_entropy = _entropy(np.linalg.eigvalsh(density))
    return float(max(gaussian_entropy - state_entropy, 0.0))


def _coherent_state(alpha: complex, dimension: int) -> np.ndarray:
    factorials = np.sqrt(
        np.asarray([factorial(index) for index in range(dimension)], dtype=float)
    )
    state = (
        np.exp(-0.5 * abs(alpha) ** 2)
        * alpha ** np.arange(dimension)
        / factorials
    )
    return state / np.linalg.norm(state)


def _gaussian_state(parameters: np.ndarray, dimension: int) -> np.ndarray:
    alpha = complex(parameters[0], parameters[1])
    squeezing = parameters[2] * np.exp(1j * parameters[3])
    annihilation, creation = _mode_operators(dimension)
    vacuum = np.zeros(dimension, dtype=complex)
    vacuum[0] = 1.0
    squeezed = expm_multiply(
        0.5
        * (
            np.conj(squeezing) * (annihilation @ annihilation)
            - squeezing * (creation @ creation)
        ),
        vacuum,
    )
    displaced = expm_multiply(
        alpha * creation - np.conj(alpha) * annihilation,
        squeezed,
    )
    return displaced / np.linalg.norm(displaced)


def fit_single_gaussian_packet(
    state: np.ndarray,
    *,
    random_starts: int = 2,
    maximum_iterations: int = 350,
    seed: int = 0,
) -> PacketFit:
    """Maximize fidelity with ``D(alpha) S(zeta)|0>``."""

    vector = np.asarray(state, dtype=complex)
    vector = vector / np.linalg.norm(vector)
    density = np.outer(vector, vector.conj())
    mean_x, mean_p, covariance = _quadrature_statistics(density)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    squeezing_magnitude = float(
        np.clip(
            0.25
            * np.log(
                max(float(eigenvalues[-1]), 1e-12)
                / max(float(eigenvalues[0]), 1e-12)
            ),
            0.0,
            1.5,
        )
    )
    squeezed_axis = eigenvectors[:, 0]
    squeezing_angle = float(
        2.0 * np.arctan2(squeezed_axis[1], squeezed_axis[0])
    )
    displacement = np.array(
        [mean_x / np.sqrt(2.0), mean_p / np.sqrt(2.0)]
    )
    starts = [
        np.array([*displacement, 0.0, 0.0]),
        np.array(
            [
                *displacement,
                squeezing_magnitude,
                squeezing_angle,
            ]
        ),
        np.array(
            [
                *displacement,
                -squeezing_magnitude,
                squeezing_angle,
            ]
        ),
    ]
    generator = np.random.default_rng(seed)
    for _ in range(random_starts):
        starts.append(
            np.array(
                [
                    displacement[0] + 0.25 * generator.normal(),
                    displacement[1] + 0.25 * generator.normal(),
                    0.4 * generator.normal(),
                    pi * generator.uniform(-1.0, 1.0),
                ]
            )
        )

    def objective(parameters: np.ndarray) -> float:
        if (
            abs(parameters[0]) > 5.0
            or abs(parameters[1]) > 5.0
            or abs(parameters[2]) > 2.0
        ):
            return 1.0
        packet = _gaussian_state(parameters, vector.size)
        return float(1.0 - abs(np.vdot(packet, vector)) ** 2)

    best = None
    function_evaluations = 0
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={
                "maxiter": maximum_iterations,
                "xatol": 1e-7,
                "fatol": 1e-10,
            },
        )
        function_evaluations += int(result.nfev)
        if best is None or result.fun < best.fun:
            best = result
    assert best is not None  # starts is nonempty
    return PacketFit(
        fidelity=float(np.clip(1.0 - best.fun, 0.0, 1.0)),
        parameters=tuple(float(value) for value in best.x),
        function_evaluations=function_evaluations,
        success=bool(best.success or best.fun < 1e-8),
    )


def fit_coherent_packet_span(
    state: np.ndarray,
    *,
    packet_count: int,
    maximum_iterations: int = 100,
    population_size: int = 9,
    seed: int = 0,
    displacement_bound: float = 4.0,
) -> PacketFit:
    """Maximize fidelity with the span of ``packet_count`` coherent states."""

    vector = np.asarray(state, dtype=complex)
    vector = vector / np.linalg.norm(vector)
    if packet_count < 1:
        raise ValueError("packet_count must be positive")

    def objective(parameters: np.ndarray) -> float:
        packets = np.column_stack(
            [
                _coherent_state(
                    complex(
                        parameters[2 * packet_index],
                        parameters[2 * packet_index + 1],
                    ),
                    vector.size,
                )
                for packet_index in range(packet_count)
            ]
        )
        gram = packets.conj().T @ packets
        overlaps = packets.conj().T @ vector
        fidelity = float(
            np.real(
                np.vdot(
                    overlaps,
                    np.linalg.pinv(gram, rcond=1e-11) @ overlaps,
                )
            )
        )
        return float(1.0 - np.clip(fidelity, 0.0, 1.0))

    result = differential_evolution(
        objective,
        [(-displacement_bound, displacement_bound)] * (2 * packet_count),
        seed=seed,
        maxiter=maximum_iterations,
        popsize=population_size,
        tol=1e-8,
        polish=True,
        workers=1,
        updating="immediate",
    )
    return PacketFit(
        fidelity=float(np.clip(1.0 - result.fun, 0.0, 1.0)),
        parameters=tuple(float(value) for value in result.x),
        function_evaluations=int(result.nfev),
        success=bool(result.success or result.fun < 1e-8),
    )


def fit_two_coherent_packets(
    state: np.ndarray,
    *,
    maximum_iterations: int = 100,
    population_size: int = 9,
    seed: int = 0,
    displacement_bound: float = 4.0,
) -> PacketFit:
    """Maximize projection fidelity onto the span of two coherent states."""

    return fit_coherent_packet_span(
        state,
        packet_count=2,
        maximum_iterations=maximum_iterations,
        population_size=population_size,
        seed=seed,
        displacement_bound=displacement_bound,
    )


def husimi_peaks(
    state: np.ndarray,
    *,
    extent: float = 4.0,
    grid_points: int = 81,
    relative_threshold: float = 0.12,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Return resolved Husimi-Q peak count, second-peak ratio, grid, and Q."""

    if grid_points < 9:
        raise ValueError("grid_points must be at least nine")
    if extent <= 0.0:
        raise ValueError("extent must be positive")
    if not 0.0 < relative_threshold < 1.0:
        raise ValueError("relative_threshold must lie in (0, 1)")
    vector = np.asarray(state, dtype=complex)
    vector = vector / np.linalg.norm(vector)
    grid = np.linspace(-extent, extent, grid_points)
    real, imaginary = np.meshgrid(grid, grid)
    alpha = (real + 1j * imaginary).reshape(-1)
    factorials = np.sqrt(
        np.asarray(
            [factorial(index) for index in range(vector.size)],
            dtype=float,
        )
    )
    coherent = (
        np.exp(-0.5 * np.abs(alpha) ** 2)[:, None]
        * alpha[:, None] ** np.arange(vector.size)[None, :]
        / factorials[None, :]
    )
    q_values = (
        np.abs(coherent.conj() @ vector) ** 2 / pi
    ).reshape(grid_points, grid_points)
    local_maximum = maximum_filter(q_values, size=5, mode="constant")
    mask = (q_values == local_maximum) & (
        q_values > relative_threshold * np.max(q_values)
    )
    peak_values = np.sort(q_values[mask])[::-1]
    second_ratio = (
        float(peak_values[1] / peak_values[0])
        if peak_values.size >= 2
        else 0.0
    )
    return int(peak_values.size), second_ratio, grid, q_values


def analyze_conditional_packet(
    conditional: ConditionalRelativeState,
    *,
    single_gaussian_random_starts: int = 2,
    single_gaussian_maximum_iterations: int = 350,
    two_packet_maximum_iterations: int = 100,
    two_packet_population_size: int = 9,
    seed: int = 0,
    husimi_grid_points: int = 81,
) -> ConditionalPacketMetrics:
    """Return all packet-compression diagnostics for one conditional state."""

    density = conditional.density_matrix
    mean_x, mean_p, covariance = _quadrature_statistics(density)
    peak_count, second_peak_ratio, _, _ = husimi_peaks(
        conditional.dominant_state,
        grid_points=husimi_grid_points,
    )
    return ConditionalPacketMetrics(
        electronic_index=conditional.electronic_index,
        probability=conditional.probability,
        center_relative_factorization=(
            conditional.center_relative_factorization
        ),
        relative_purity=float(np.trace(density @ density).real),
        mean_x=mean_x,
        mean_p=mean_p,
        covariance_xx=float(covariance[0, 0]),
        covariance_xp=float(covariance[0, 1]),
        covariance_pp=float(covariance[1, 1]),
        gaussian_non_gaussianity=gaussian_non_gaussianity(density),
        husimi_peak_count=peak_count,
        husimi_second_peak_ratio=second_peak_ratio,
        single_gaussian_fit=fit_single_gaussian_packet(
            conditional.dominant_state,
            random_starts=single_gaussian_random_starts,
            maximum_iterations=single_gaussian_maximum_iterations,
            seed=seed,
        ),
        two_coherent_fit=fit_two_coherent_packets(
            conditional.dominant_state,
            maximum_iterations=two_packet_maximum_iterations,
            population_size=two_packet_population_size,
            seed=seed,
        ),
    )


__all__ = [
    "ConditionalPacketMetrics",
    "ConditionalRelativeState",
    "ElectronRelativeState",
    "LocalProductEmbedding",
    "PacketFit",
    "analyze_conditional_packet",
    "conditional_relative_state",
    "electron_relative_state",
    "electron_relative_product_to_local_state",
    "fit_single_gaussian_packet",
    "fit_coherent_packet_span",
    "fit_two_coherent_packets",
    "gaussian_non_gaussianity",
    "husimi_peaks",
    "local_to_normal_mode_transform",
]
