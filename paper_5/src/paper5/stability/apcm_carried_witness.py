"""Carried-witness radial moment flow for archive-backed APCM dynamics.

The memoryless projective selector repeatedly solved for a positive outer
completion.  This module instead propagates one completion as causal state.
The retained velocity is the archive matrix EOM augmented by the explicit
K/P/D correlation source reconstructed from the carried moments; it and the
entrance-channel rates are evaluated first and held fixed by the Gram guard.
Completion velocities are obtained from every commutator rate readable
inside the carried registry; unresolved rates receive the minimum-motion
tie-break.  Each forward-Euler atom is checked against the literal 62-row
finite Gram and, when needed, corrected by a small spectral-bundle QP.

The operational cone is the coefficient-scaled Gram shifted by the explicitly
declared ``psd_inflation``.  It is a numerical finite-Gram guard, not a claim of
positivity on the full CCR algebra.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Callable, Mapping

import clarabel
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.optimize import minimize

from .adaptive_positive_moment import (
    ENTRANCE_RELATIVE_MOMENT_KEYS,
    RAW_MOMENT_COORDINATE_NAMES,
    kpd_correlation_velocity_correction,
    matrix_derivative_to_raw_moment_velocity,
    matrix_state_to_raw_moment_coordinates,
    raw_moment_coordinates_to_matrix_state,
)
from .apcm_moment_projection import state_lower_moments
from .apcm_positive_extension import (
    SymmetryReducedPositiveExtension,
    _clarabel_svec_upper,
    _realify_hermitian,
)
from .apcm_projective_guard import (
    _HAMILTONIAN_OPERATOR_BASIS,
    center_core_null_directions,
    projective_guard_outer_extension,
    relative_hermitian_core_restriction,
    unified_core_moment_matrix,
    unified_glued_moment_matrix,
    unified_to_relative_restriction,
)
from .exact_reference import exact_holstein_joint_moment_initial_state
from .hubbard_dimer import DimerParameters, FloatArray
from .matrix_reference import MatrixDimerState, matrix_dimer_rhs
from .moment_hierarchy import (
    MomentHierarchy,
    MomentKey,
    _canonical_key,
    _commutator,
    _hamiltonian_terms,
)

ComplexArray = NDArray[np.complex128]

_RAW_COUNT = len(RAW_MOMENT_COORDINATE_NAMES)
_ENTRANCE_COUNT = len(ENTRANCE_RELATIVE_MOMENT_KEYS)


class CarriedWitnessError(RuntimeError):
    """The declared carried-witness numerical contract could not be met."""


@dataclass(frozen=True)
class CWRMFSettings:
    """Frozen settings for the short carried-witness pilot."""

    psd_inflation: float = 1e-11
    affine_tolerance: float = 1e-10
    spectral_entry_threshold: float = 1e-8
    spectral_exit_threshold: float = 1e-7
    maximum_critical_modes: int | None = None
    maximum_local_corrections: int = 3
    solver_tolerance: float = 1e-10
    readable_rate_tolerance: float = 1e-5
    archive_velocity_tolerance: float = 1e-10
    velocity_ceiling_factor: float = 10.0
    velocity_ceiling_margin: float = 1e-3
    schur_safety_margin: float = 2e-12
    schur_maximum_iterations: int = 600
    maximum_cutting_plane_iterations: int = 256

    def __post_init__(self) -> None:
        if self.psd_inflation < 0.0:
            raise ValueError("psd_inflation must be nonnegative")
        if self.affine_tolerance <= 0.0 or self.solver_tolerance <= 0.0:
            raise ValueError("numerical tolerances must be positive")
        if not (
            0.0 < self.spectral_entry_threshold < self.spectral_exit_threshold
        ):
            raise ValueError("spectral thresholds must be positive and ordered")
        if (
            self.maximum_critical_modes is not None
            and self.maximum_critical_modes <= 0
        ):
            raise ValueError("maximum_critical_modes must be positive")
        if self.maximum_local_corrections <= 0:
            raise ValueError("maximum_local_corrections must be positive")
        if self.velocity_ceiling_factor <= 1.0:
            raise ValueError("velocity_ceiling_factor must exceed one")
        if not 0.0 < self.velocity_ceiling_margin < 1.0:
            raise ValueError("velocity_ceiling_margin must lie in (0,1)")
        if self.schur_safety_margin <= 0.0:
            raise ValueError("schur_safety_margin must be positive")
        if self.schur_maximum_iterations <= 0:
            raise ValueError("schur_maximum_iterations must be positive")
        if self.maximum_cutting_plane_iterations <= 0:
            raise ValueError(
                "maximum_cutting_plane_iterations must be positive"
            )

    @property
    def critical_mode_limit(self) -> int:
        """Return the explicit cap or the natural Schur-complement limit."""

        if self.maximum_critical_modes is None:
            return 61
        return min(self.maximum_critical_modes, 61)


@dataclass(frozen=True)
class SpectralEnclosure:
    """Backward-error enclosure for one ordered Hermitian spectrum."""

    eigenvalues: FloatArray
    lower_bounds: FloatArray
    upper_bounds: FloatArray
    eigenvectors: ComplexArray
    backward_error: float
    orthogonality_error: float

    @property
    def minimum_lower_bound(self) -> float:
        return float(self.lower_bounds[0])


@dataclass(frozen=True)
class CWRMFPreparation:
    """Canonical correlated preparation and its carried outer witness."""

    state: FloatArray
    ground_energy: float
    hierarchy_degree: int
    minimum_unshifted_eigenvalue: float
    minimum_shifted_lower_bound: float
    restriction_residual: float
    factorization_residual: float


@dataclass(frozen=True)
class RadialAtomResult:
    """One accepted or rejected finite-step radial atom."""

    success: bool
    endpoint: FloatArray
    archive_velocity: FloatArray
    completion_velocity: FloatArray
    desired_completion_velocity: FloatArray
    minimum_unshifted_eigenvalue: float
    minimum_shifted_lower_bound: float
    readable_rate_residual: float
    archive_intervention: float
    completion_correction_norm: float
    velocity_margin: float
    critical_modes: int
    correction_iterations: int
    elapsed_seconds: float
    message: str


@dataclass(frozen=True)
class CWRMFTrajectory:
    """Fixed-step SSPRK2 carried-witness rollout."""

    times: FloatArray
    states: FloatArray
    minimum_unshifted_eigenvalues: FloatArray
    minimum_shifted_lower_bounds: FloatArray
    maximum_atom_seconds: FloatArray
    correction_iterations: NDArray[np.int64]
    readable_rate_residuals: FloatArray
    completion_correction_norms: FloatArray
    velocity_margins: FloatArray
    critical_modes: NDArray[np.int64]
    completed_steps: int
    atom_evaluations: int
    success: bool
    message: str


def _project_correlation_trace_velocity(
    derivative: MatrixDimerState,
) -> MatrixDimerState:
    correlation = np.asarray(
        derivative.electron_phonon_correlation, dtype=complex
    ).copy()
    identity = np.eye(2, dtype=complex)
    for mode in range(correlation.shape[0]):
        correlation[mode] -= 0.5 * np.trace(correlation[mode]) * identity
    return MatrixDimerState(
        electron_density=derivative.electron_density,
        coherent_phonon=derivative.coherent_phonon,
        phonon_density=derivative.phonon_density,
        anomalous_phonon_density=derivative.anomalous_phonon_density,
        electron_phonon_correlation=correlation,
    )


def _moment_derivative(
    key: MomentKey,
    time: float,
    parameters: DimerParameters,
    moments: Mapping[MomentKey, float],
) -> float:
    derivative = 0.0j
    for coefficient, hamiltonian_word in _hamiltonian_terms(time, parameters):
        for generated, commutator_coefficient in _commutator(
            hamiltonian_word, key
        ).items():
            value = 1.0 if generated.degree == 0 else moments[generated]
            derivative += (
                1j * coefficient * commutator_coefficient * float(value)
            )
    if abs(derivative.imag) > 5e-9:
        raise FloatingPointError(
            f"Hermitian moment {key} acquired complex rate {derivative}"
        )
    return float(derivative.real)


class CarriedWitnessGeometry:
    """Immutable quotient, Gram, scaling, and readable-rate compiler."""

    def __init__(self, *, phonon_envelope: float = 16.0) -> None:
        current = SymmetryReducedPositiveExtension(
            active_keys=ENTRANCE_RELATIVE_MOMENT_KEYS
        )
        if not np.isclose(
            current.settings.phonon_envelope,
            phonon_envelope,
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError("phonon envelope differs from extension settings")
        self.current_extension = current
        self.outer_extension = projective_guard_outer_extension(current)
        self.frontier_count = len(self.outer_extension.frontier_keys)
        self.extra_count = self.outer_extension.dimension - 9
        self.center_cross_count = 2 * self.extra_count
        self.completion_count = self.frontier_count + self.center_cross_count
        self.retained_count = _RAW_COUNT + _ENTRANCE_COUNT
        self.state_count = self.retained_count + self.completion_count
        self.hierarchy_degree = max(
            key.degree
            for key in (
                *self.outer_extension.lower_keys,
                *self.outer_extension.frontier_keys,
            )
        )
        self.preparation_hierarchy = MomentHierarchy(self.hierarchy_degree)
        self._relative_restriction = relative_hermitian_core_restriction()
        self._center_directions = center_core_null_directions()
        self._unified_restriction = unified_to_relative_restriction(
            self.outer_extension.dimension
        )
        root_scale = np.sqrt(phonon_envelope + 1.0)
        self.unified_word_scales = np.concatenate(
            (
                [1.0],
                np.full(4, root_scale),
                np.ones(6),
                self.outer_extension.word_scales[9:],
            )
        )
        self.center_cross_scales = (
            root_scale * self.outer_extension.word_scales[9:][None, :]
        ).repeat(2, axis=0)
        self._scaled_completion_coefficients = (
            self._compile_scaled_completion_coefficients()
        )
        self.readable_frontier_indices = self._compile_readable_frontier()
        (
            self.readable_center_indices,
            self._center_generated_rows,
        ) = self._compile_readable_center_cross()
        self.readable_completion_indices = np.asarray(
            [
                *self.readable_frontier_indices,
                *(
                    self.frontier_count + self.readable_center_indices
                ),
            ],
            dtype=int,
        )

    @property
    def completion_scales(self) -> FloatArray:
        return np.concatenate(
            (
                self.outer_extension.frontier_scales,
                self.center_cross_scales.reshape(-1),
            )
        )

    def pack_state(
        self,
        retained: FloatArray,
        completion: FloatArray,
    ) -> FloatArray:
        retained_array = np.asarray(retained, dtype=float)
        completion_array = np.asarray(completion, dtype=float)
        if retained_array.shape != (self.retained_count,):
            raise ValueError("retained state has the wrong dimension")
        if completion_array.shape != (self.completion_count,):
            raise ValueError("completion state has the wrong dimension")
        return np.concatenate((retained_array, completion_array))

    def unpack_state(self, state: FloatArray) -> tuple[FloatArray, FloatArray]:
        values = np.asarray(state, dtype=float)
        if values.shape != (self.state_count,):
            raise ValueError(
                f"carried state must have shape {(self.state_count,)}"
            )
        return (
            values[: self.retained_count].copy(),
            values[self.retained_count :].copy(),
        )

    def split_retained(self, retained: FloatArray) -> tuple[FloatArray, FloatArray]:
        values = np.asarray(retained, dtype=float)
        if values.shape != (self.retained_count,):
            raise ValueError("retained state has the wrong dimension")
        return values[:_RAW_COUNT], values[_RAW_COUNT:]

    def split_completion(
        self, completion: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        values = np.asarray(completion, dtype=float)
        if values.shape != (self.completion_count,):
            raise ValueError("completion state has the wrong dimension")
        frontier = values[: self.frontier_count]
        center = values[self.frontier_count :].reshape(2, self.extra_count)
        return frontier, center

    def lower_moments(self, retained: FloatArray) -> Mapping[MomentKey, float]:
        raw, entrance = self.split_retained(retained)
        return state_lower_moments(
            raw, entrance, ENTRANCE_RELATIVE_MOMENT_KEYS
        )

    def moment_mapping(
        self, retained: FloatArray, completion: FloatArray
    ) -> Mapping[MomentKey, float]:
        frontier, _ = self.split_completion(completion)
        values = dict(self.lower_moments(retained))
        values.update(
            {
                key: float(
                    frontier[index]
                    * self.outer_extension.frontier_scales[index]
                )
                for index, key in enumerate(self.outer_extension.frontier_keys)
            }
        )
        return MappingProxyType(values)

    def relative_matrix(
        self, retained: FloatArray, completion: FloatArray
    ) -> ComplexArray:
        frontier, _ = self.split_completion(completion)
        frontier_mapping = {
            key: float(frontier[index] * self.outer_extension.frontier_scales[index])
            for index, key in enumerate(self.outer_extension.frontier_keys)
        }
        return self.outer_extension.matrix(
            self.lower_moments(retained), frontier_mapping
        )

    def core_matrix(self, retained: FloatArray) -> ComplexArray:
        raw, _ = self.split_retained(retained)
        return unified_core_moment_matrix(raw, self.lower_moments(retained))

    def unified_matrix(
        self, retained: FloatArray, completion: FloatArray
    ) -> ComplexArray:
        relative = self.relative_matrix(retained, completion)
        _, center = self.split_completion(completion)
        physical_center = center * self.center_cross_scales
        return unified_glued_moment_matrix(
            self.core_matrix(retained),
            relative,
            center_cross=physical_center,
        )

    def scaled_unified_matrix(
        self, retained: FloatArray, completion: FloatArray
    ) -> ComplexArray:
        matrix = self.unified_matrix(retained, completion)
        inverse = 1.0 / self.unified_word_scales
        scaled = inverse[:, None] * matrix * inverse[None, :]
        return np.asarray(0.5 * (scaled + scaled.conjugate().T), dtype=complex)

    def scaled_completion_coefficients(self) -> ComplexArray:
        return self._scaled_completion_coefficients

    def restriction_residual(
        self, retained: FloatArray, completion: FloatArray
    ) -> float:
        unified = self.unified_matrix(retained, completion)
        restricted = (
            self._unified_restriction.conjugate()
            @ unified
            @ self._unified_restriction.T
        )
        return float(
            np.linalg.norm(
                restricted - self.relative_matrix(retained, completion),
                ord=np.inf,
            )
        )

    def _compile_scaled_completion_coefficients(self) -> ComplexArray:
        outer = self.outer_extension
        physical_relative = (
            outer.frontier_coefficients
            * outer.frontier_scales[:, None, None]
        )
        coefficients = np.zeros(
            (self.completion_count, 62, 62), dtype=complex
        )
        for index, relative in enumerate(physical_relative):
            if np.linalg.norm(relative[:9, :9]) > 3e-12:
                raise RuntimeError("frontier moment changed the fixed relative core")
            cross = self._relative_restriction.T @ relative[:9, 9:]
            coefficients[index, :11, 11:] = cross
            coefficients[index, 11:, :11] = cross.conjugate().T
            coefficients[index, 11:, 11:] = relative[9:, 9:]
        offset = self.frontier_count
        for center_index in range(2):
            for extra_index in range(self.extra_count):
                index = offset + center_index * self.extra_count + extra_index
                scale = self.center_cross_scales[center_index, extra_index]
                column = self._center_directions[:, center_index] * scale
                coefficients[index, :11, 11 + extra_index] = column
                coefficients[index, 11 + extra_index, :11] = column.conjugate()
        inverse = 1.0 / self.unified_word_scales
        scaled = (
            inverse[None, :, None]
            * coefficients
            * inverse[None, None, :]
        )
        return np.asarray(
            0.5 * (scaled + scaled.conjugate().transpose(0, 2, 1)),
            dtype=complex,
        )

    def _compile_readable_frontier(self) -> NDArray[np.int64]:
        supported = {
            *self.outer_extension.lower_keys,
            *self.outer_extension.frontier_keys,
        }
        readable: list[int] = []
        for index, key in enumerate(self.outer_extension.frontier_keys):
            generated = {
                moment
                for _, hamiltonian_word in _HAMILTONIAN_OPERATOR_BASIS
                for moment in _commutator(hamiltonian_word, key)
                if moment.degree > 0
            }
            if generated.issubset(supported):
                readable.append(index)
        return np.asarray(readable, dtype=np.int64)

    def _relative_row_key(self, row: int) -> MomentKey:
        word = self.outer_extension.words[row]
        return _canonical_key(
            word.spin_up, word.spin_down, word.x_power, word.p_power
        )

    def _center_row_lookup(self) -> Mapping[MomentKey, int]:
        result: dict[MomentKey, int] = {}
        for row in range(9):
            key = self._relative_row_key(row)
            result.setdefault(key, row)
        return MappingProxyType(result)

    def _compile_readable_center_cross(
        self,
    ) -> tuple[NDArray[np.int64], tuple[tuple[tuple[float, MomentKey], ...], ...]]:
        core_lookup = self._center_row_lookup()
        extra_lookup = {
            self._relative_row_key(9 + index): index
            for index in range(self.extra_count)
        }
        available = set(core_lookup).union(extra_lookup)
        readable: list[int] = []
        generated_rows: list[tuple[tuple[float, MomentKey], ...]] = []
        for extra_index in range(self.extra_count):
            key = self._relative_row_key(9 + extra_index)
            generated: set[MomentKey] = set()
            for _, hamiltonian_word in _HAMILTONIAN_OPERATOR_BASIS:
                generated.update(
                    moment
                    for moment in _commutator(hamiltonian_word, key)
                    if moment.degree > 0
                )
            if not generated.issubset(available):
                continue
            readable.append(extra_index)
            # The numerical coefficients remain time dependent; storing the
            # structural keys here records the immutable readability decision.
            generated_rows.append(tuple((1.0, moment) for moment in sorted(generated)))
        expanded = np.asarray(
            [
                center_index * self.extra_count + extra_index
                for center_index in range(2)
                for extra_index in readable
            ],
            dtype=np.int64,
        )
        return expanded, tuple(generated_rows)

    def center_cross_value(
        self,
        center_index: int,
        key: MomentKey,
        retained: FloatArray,
        completion: FloatArray,
    ) -> float:
        core_lookup = self._center_row_lookup()
        if key in core_lookup:
            core = self.core_matrix(retained)
            overlap = (
                self._center_directions.conjugate().T
                @ core
                @ self._relative_restriction.T
            )
            value = overlap[center_index, core_lookup[key]]
        else:
            extra_lookup = {
                self._relative_row_key(9 + index): index
                for index in range(self.extra_count)
            }
            try:
                extra_index = extra_lookup[key]
            except KeyError as error:
                raise KeyError(f"center cross moment {key} is not carried") from error
            _, center = self.split_completion(completion)
            value = (
                center[center_index, extra_index]
                * self.center_cross_scales[center_index, extra_index]
            )
        if abs(complex(value).imag) > 5e-9:
            raise FloatingPointError(
                f"Hermitian center-cross moment became complex: {value}"
            )
        return float(complex(value).real)


class CarriedWitnessModel:
    """Archive-first carried-witness radial model for one frozen dictionary."""

    def __init__(
        self,
        parameters: DimerParameters,
        *,
        settings: CWRMFSettings | None = None,
        geometry: CarriedWitnessGeometry | None = None,
    ) -> None:
        self.parameters = parameters
        self.settings = CWRMFSettings() if settings is None else settings
        self.geometry = (
            CarriedWitnessGeometry() if geometry is None else geometry
        )
        self._retained_scales: FloatArray | None = None
        self._bundle_rank_hint = 0

    @property
    def bundle_rank_hint(self) -> int:
        """Return the largest critical bundle required so far."""

        return self._bundle_rank_hint

    def restore_bundle_rank_hint(self, value: int) -> None:
        """Restore continuation metadata without changing the physical state."""

        if value < 0 or value > 61:
            raise ValueError("bundle rank hint must lie between zero and 61")
        self._bundle_rank_hint = int(value)

    def prepare(self, *, phonon_cutoff: int = 16) -> CWRMFPreparation:
        geometry = self.geometry
        exact = exact_holstein_joint_moment_initial_state(
            self.parameters,
            hierarchy=geometry.preparation_hierarchy,
            phonon_cutoff=phonon_cutoff,
            canonical_embedding=True,
        )
        center, moments = geometry.preparation_hierarchy.unpack(
            exact.hierarchy_coordinates
        )
        del center
        # Use the same canonically embedded hierarchy for the retained matrix
        # tuple.  Mixing it with direct cutoff contractions recreates the
        # finite-boundary CCR mismatch that the preparation audit removed.
        matrix_state = geometry.preparation_hierarchy.to_matrix_state(
            exact.hierarchy_coordinates
        )
        raw = matrix_state_to_raw_moment_coordinates(matrix_state)
        entrance = np.asarray(
            [moments[key] for key in ENTRANCE_RELATIVE_MOMENT_KEYS], dtype=float
        )
        retained = np.concatenate((raw, entrance))
        frontier = np.asarray(
            [
                moments[key] / geometry.outer_extension.frontier_scales[index]
                for index, key in enumerate(geometry.outer_extension.frontier_keys)
            ],
            dtype=float,
        )
        relative = geometry.outer_extension.matrix(
            geometry.lower_moments(retained),
            {
                key: moments[key]
                for key in geometry.outer_extension.frontier_keys
            },
        )
        core = geometry.core_matrix(retained)
        center_overlap = (
            geometry._center_directions.conjugate().T
            @ core
            @ geometry._relative_restriction.T
        )
        factorization_residual = float(
            np.linalg.norm(
                center_overlap
                - center_overlap[:, [0]] @ relative[[0], :9],
                ord=np.inf,
            )
        )
        physical_center = np.asarray(
            center_overlap[:, [0]] @ relative[[0], 9:], dtype=complex
        )
        if np.max(np.abs(physical_center.imag)) > self.settings.affine_tolerance:
            raise CarriedWitnessError(
                "canonical preparation has complex Hermitian center crosses"
            )
        center_standardized = (
            physical_center.real / geometry.center_cross_scales
        ).reshape(-1)
        completion = np.concatenate((frontier, center_standardized))
        state = geometry.pack_state(retained, completion)
        self._retained_scales = np.maximum(1.0, np.abs(retained))
        enclosure = self.spectral_enclosure(
            geometry.scaled_unified_matrix(retained, completion),
            shifted=True,
        )
        minimum_unshifted = float(
            np.linalg.eigvalsh(
                geometry.scaled_unified_matrix(retained, completion)
            )[0]
        )
        restriction = geometry.restriction_residual(retained, completion)
        if factorization_residual > self.settings.affine_tolerance:
            raise CarriedWitnessError(
                "canonical preparation violates center/relative factorization"
            )
        if restriction > self.settings.affine_tolerance:
            raise CarriedWitnessError("unified preparation restriction failed")
        if enclosure.minimum_lower_bound < 0.0:
            raise CarriedWitnessError(
                "canonical preparation is unresolved in the shifted Gram guard"
            )
        return CWRMFPreparation(
            state=state,
            ground_energy=exact.energy,
            hierarchy_degree=geometry.hierarchy_degree,
            minimum_unshifted_eigenvalue=minimum_unshifted,
            minimum_shifted_lower_bound=enclosure.minimum_lower_bound,
            restriction_residual=restriction,
            factorization_residual=factorization_residual,
        )

    def spectral_enclosure(
        self, matrix: ComplexArray, *, shifted: bool
    ) -> SpectralEnclosure:
        value = np.asarray(matrix, dtype=complex)
        if shifted:
            value = value + self.settings.psd_inflation * np.eye(
                value.shape[0], dtype=complex
            )
        eigenvalues, eigenvectors = np.linalg.eigh(value)
        reconstructed = (
            eigenvectors
            @ np.diag(eigenvalues)
            @ eigenvectors.conjugate().T
        )
        residual = float(np.linalg.norm(value - reconstructed, ord=2))
        orthogonality = float(
            np.linalg.norm(
                eigenvectors.conjugate().T @ eigenvectors
                - np.eye(value.shape[0]),
                ord=2,
            )
        )
        rounding = (
            20.0
            * np.finfo(float).eps
            * max(1.0, float(np.linalg.norm(value, ord=2)))
            * value.shape[0]
        )
        error = residual + orthogonality * max(
            1.0, float(np.max(np.abs(eigenvalues)))
        ) + rounding
        return SpectralEnclosure(
            eigenvalues=np.asarray(eigenvalues, dtype=float),
            lower_bounds=np.asarray(eigenvalues - error, dtype=float),
            upper_bounds=np.asarray(eigenvalues + error, dtype=float),
            eigenvectors=np.asarray(eigenvectors, dtype=complex),
            backward_error=error,
            orthogonality_error=orthogonality,
        )

    def retained_velocity(
        self, time: float, retained: FloatArray, completion: FloatArray
    ) -> FloatArray:
        """Return archive rates plus the carried-moment K/P/D C-rate repair."""

        geometry = self.geometry
        raw, _ = geometry.split_retained(retained)
        matrix_state = raw_moment_coordinates_to_matrix_state(raw)
        moments = geometry.moment_mapping(retained, completion)
        archive = _project_correlation_trace_velocity(
            matrix_dimer_rhs(time, matrix_state, self.parameters)
        )
        augmented = MatrixDimerState(
            electron_density=archive.electron_density,
            coherent_phonon=archive.coherent_phonon,
            phonon_density=archive.phonon_density,
            anomalous_phonon_density=archive.anomalous_phonon_density,
            electron_phonon_correlation=(
                archive.electron_phonon_correlation
                + kpd_correlation_velocity_correction(
                    matrix_state, self.parameters, moments
                )
            ),
        )
        raw_velocity = matrix_derivative_to_raw_moment_velocity(
            matrix_state, augmented
        )
        entrance_velocity = np.asarray(
            [
                _moment_derivative(
                    key, time, self.parameters, moments
                )
                for key in ENTRANCE_RELATIVE_MOMENT_KEYS
            ],
            dtype=float,
        )
        return np.concatenate((raw_velocity, entrance_velocity))

    def desired_completion_velocity(
        self, time: float, retained: FloatArray, completion: FloatArray
    ) -> FloatArray:
        geometry = self.geometry
        moments = geometry.moment_mapping(retained, completion)
        result = np.zeros(geometry.completion_count, dtype=float)
        for index in geometry.readable_frontier_indices:
            key = geometry.outer_extension.frontier_keys[int(index)]
            result[index] = _moment_derivative(
                key, time, self.parameters, moments
            ) / geometry.outer_extension.frontier_scales[index]

        _, center_standardized = geometry.split_completion(completion)
        physical_center = center_standardized * geometry.center_cross_scales
        center_rates = (
            self.parameters.omega_ph * physical_center[1],
            -self.parameters.omega_ph * physical_center[0]
            - 2.0
            * self.parameters.coupling
            * np.asarray(
                [
                    moments[geometry._relative_row_key(9 + index)]
                    for index in range(geometry.extra_count)
                ],
                dtype=float,
            ),
        )
        readable_extra = sorted(
            {
                int(index % geometry.extra_count)
                for index in geometry.readable_center_indices
            }
        )
        for extra_index in readable_extra:
            key = geometry._relative_row_key(9 + extra_index)
            relative_terms = _hamiltonian_terms(time, self.parameters)
            for center_index in range(2):
                derivative = float(center_rates[center_index][extra_index])
                for coefficient, hamiltonian_word in relative_terms:
                    for generated, commutator_coefficient in _commutator(
                        hamiltonian_word, key
                    ).items():
                        cross = geometry.center_cross_value(
                            center_index,
                            generated,
                            retained,
                            completion,
                        )
                        derivative += float(
                            (
                                1j
                                * coefficient
                                * commutator_coefficient
                                * cross
                            ).real
                        )
                local_index = center_index * geometry.extra_count + extra_index
                result[geometry.frontier_count + local_index] = (
                    derivative
                    / geometry.center_cross_scales[center_index, extra_index]
                )
        return result

    def _retained_scaled_norm(self, velocity: FloatArray) -> float:
        if self._retained_scales is None:
            raise CarriedWitnessError("prepare must be called before propagation")
        return float(np.linalg.norm(velocity / self._retained_scales))

    def _velocity_ceiling(
        self, retained_velocity: FloatArray, desired: FloatArray
    ) -> float:
        readable = desired[self.geometry.readable_completion_indices]
        return self.settings.velocity_ceiling_factor * (
            1.0
            + self._retained_scaled_norm(retained_velocity)
            + float(np.linalg.norm(readable))
        )

    def _solve_cut_qp(
        self,
        retained_velocity: FloatArray,
        desired: FloatArray,
        cuts: list[tuple[FloatArray, float]],
    ) -> tuple[bool, FloatArray, str, float, float]:
        geometry = self.geometry
        count = geometry.completion_count
        readable = geometry.readable_completion_indices
        blocks: list[sparse.csc_matrix] = []
        rhs: list[FloatArray] = []
        cones: list[object] = []
        if cuts:
            normals = np.asarray([normal for normal, _ in cuts], dtype=float)
            offsets = np.asarray([offset for _, offset in cuts], dtype=float)
            blocks.append(sparse.csc_matrix(-normals))
            rhs.append(offsets)
            cones.append(clarabel.NonnegativeConeT(len(cuts)))
        ceiling = self._velocity_ceiling(retained_velocity, desired)
        retained_norm = self._retained_scaled_norm(retained_velocity)
        completion_radius_squared = ceiling**2 - retained_norm**2
        if completion_radius_squared <= 0.0:
            return False, np.zeros(count), "velocity_prior_active", np.inf, -np.inf
        completion_radius = np.sqrt(completion_radius_squared)
        soc = sparse.vstack(
            (
                sparse.csc_matrix((1, count)),
                -sparse.eye(count, format="csc"),
            ),
            format="csc",
        )
        blocks.append(soc)
        rhs.append(np.concatenate(([completion_radius], np.zeros(count))))
        cones.append(clarabel.SecondOrderConeT(count + 1))
        equality = sparse.coo_matrix(
            (
                np.ones(readable.size),
                (np.arange(readable.size), readable),
            ),
            shape=(readable.size, count),
        ).tocsc()
        blocks.append(equality)
        rhs.append(desired[readable])
        cones.append(clarabel.ZeroConeT(readable.size))
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_threads = 1
        settings.max_iter = 200
        settings.tol_gap_abs = self.settings.solver_tolerance
        settings.tol_gap_rel = self.settings.solver_tolerance
        settings.tol_feas = self.settings.solver_tolerance
        settings.iterative_refinement_enable = True
        settings.iterative_refinement_max_iter = 10
        settings.iterative_refinement_abstol = 1e-12
        settings.iterative_refinement_reltol = 1e-12
        matrix = sparse.vstack(blocks, format="csc")
        vector = np.concatenate(rhs)
        solution = clarabel.DefaultSolver(
            sparse.eye(count, format="csc"),
            np.zeros(count),
            matrix,
            vector,
            cones,
            settings,
        ).solve()
        status = str(solution.status)
        if solution.x is None:
            return False, np.zeros(count), status, np.inf, -np.inf
        velocity = np.asarray(solution.x, dtype=float)
        residual = max(float(solution.r_prim), float(solution.r_dual))
        margin = 1.0 - np.sqrt(
            retained_norm**2 + float(np.dot(velocity, velocity))
        ) / ceiling
        success = bool(
            status == "Solved" and residual <= 10.0 * self.settings.solver_tolerance
        )
        return success, velocity, status, residual, float(margin)

    def _solve_critical_schur(
        self,
        endpoint_retained: FloatArray,
        completion: FloatArray,
        step_size: float,
        retained_velocity: FloatArray,
        desired: FloatArray,
        *,
        minimum_critical_modes: int = 0,
        initial_velocity: FloatArray | None = None,
    ) -> tuple[bool, FloatArray, str, float, float, int]:
        """Solve the fixed-archive completion tier in the critical subspace.

        Readable commutator rates are eliminated exactly.  The optimization
        variables are only the terminal completion-rate directions.  The
        small Schur complement enforces the dangerous part of the literal
        62-row Gram; the caller still performs the authoritative full-Gram
        eigendecomposition before accepting the atom.
        """

        geometry = self.geometry
        readable = geometry.readable_completion_indices
        free = np.setdiff1d(
            np.arange(geometry.completion_count, dtype=np.int64),
            readable,
            assume_unique=True,
        )
        shifted_predictor = (
            geometry.scaled_unified_matrix(
                endpoint_retained,
                completion + step_size * desired,
            )
            + self.settings.psd_inflation * np.eye(62, dtype=complex)
        )
        eigenvalues, eigenvectors = np.linalg.eigh(shifted_predictor)
        critical_count = max(
            int(minimum_critical_modes),
            int(
                np.count_nonzero(
                    eigenvalues < self.settings.spectral_entry_threshold
                )
            ),
        )
        if critical_count <= 0:
            return True, desired.copy(), "no_critical_modes", 0.0, 1.0, 0
        if critical_count > self.settings.critical_mode_limit:
            return (
                False,
                desired.copy(),
                "critical_mode_limit",
                np.inf,
                -np.inf,
                critical_count,
            )

        critical_basis = eigenvectors[:, :critical_count]
        positive_basis = eigenvectors[:, critical_count:]
        positive_eigenvalues = eigenvalues[critical_count:]
        if positive_eigenvalues.size == 0 or positive_eigenvalues[0] <= 0.0:
            return (
                False,
                desired.copy(),
                "positive_block_unresolved",
                np.inf,
                -np.inf,
                critical_count,
            )

        free_coefficients = (
            step_size
            * geometry.scaled_completion_coefficients()[free]
        )
        positive_base = (
            positive_basis.conjugate().T
            @ shifted_predictor
            @ positive_basis
        )
        cross_base = (
            positive_basis.conjugate().T
            @ shifted_predictor
            @ critical_basis
        )
        critical_base = (
            critical_basis.conjugate().T
            @ shifted_predictor
            @ critical_basis
        )
        positive_coefficients = np.einsum(
            "pa,jpq,qb->jab",
            positive_basis.conjugate(),
            free_coefficients,
            positive_basis,
            optimize=True,
        )
        cross_coefficients = np.einsum(
            "pa,jpq,qb->jab",
            positive_basis.conjugate(),
            free_coefficients,
            critical_basis,
            optimize=True,
        )
        critical_coefficients = np.einsum(
            "pa,jpq,qb->jab",
            critical_basis.conjugate(),
            free_coefficients,
            critical_basis,
            optimize=True,
        )
        # Schur equivalence requires a safely positive complement, not that
        # every positive eigenvalue remain within a fixed fraction of its
        # predictor value.  The frozen entry gap is the directly relevant
        # inversion bound and avoids rejecting a fully certified Gram merely
        # because its harmless positive spectrum redistributes.
        trust_base = self.settings.spectral_entry_threshold * np.eye(
            positive_eigenvalues.size, dtype=complex
        )

        def schur_linearization(
            free_velocity: FloatArray,
        ) -> tuple[ComplexArray, ComplexArray, float]:
            positive = positive_base + np.einsum(
                "j,jab->ab",
                free_velocity,
                positive_coefficients,
                optimize=True,
            )
            cross = cross_base + np.einsum(
                "j,jab->ab",
                free_velocity,
                cross_coefficients,
                optimize=True,
            )
            critical = critical_base + np.einsum(
                "j,jab->ab",
                free_velocity,
                critical_coefficients,
                optimize=True,
            )
            positive = 0.5 * (positive + positive.conjugate().T)
            critical = 0.5 * (critical + critical.conjugate().T)
            solved_cross = np.linalg.solve(positive, cross)
            schur = critical - cross.conjugate().T @ solved_cross
            schur = 0.5 * (schur + schur.conjugate().T)
            schur_derivatives = (
                critical_coefficients
                - np.einsum(
                    "jpr,ps->jrs",
                    cross_coefficients.conjugate(),
                    solved_cross,
                    optimize=True,
                )
                - np.einsum(
                    "rp,jps->jrs",
                    solved_cross.conjugate().T,
                    cross_coefficients,
                    optimize=True,
                )
                + np.einsum(
                    "rp,jpq,qs->jrs",
                    solved_cross.conjugate().T,
                    positive_coefficients,
                    solved_cross,
                    optimize=True,
                )
            )
            trust = positive - trust_base
            trust = 0.5 * (trust + trust.conjugate().T)
            return (
                np.asarray(schur, dtype=complex),
                np.asarray(schur_derivatives, dtype=complex),
                float(np.linalg.eigvalsh(trust)[0]),
            )

        ceiling = self._velocity_ceiling(retained_velocity, desired)
        retained_norm = self._retained_scaled_norm(retained_velocity)
        fixed_completion_norm = float(np.dot(desired, desired))
        free_radius_squared = (
            ceiling**2 - retained_norm**2 - fixed_completion_norm
        )
        if free_radius_squared <= 0.0:
            return (
                False,
                desired.copy(),
                "velocity_prior_active",
                np.inf,
                -np.inf,
                critical_count,
            )

        current = np.zeros(free.size, dtype=float)
        if initial_velocity is not None:
            supplied = np.asarray(initial_velocity, dtype=float)
            if supplied.shape != (geometry.completion_count,):
                raise ValueError("initial completion velocity has wrong shape")
            current = supplied[free] - desired[free]
            radius = np.sqrt(free_radius_squared)
            norm = float(np.linalg.norm(current))
            if norm >= radius:
                current *= 0.95 * radius / norm

        safety = self.settings.schur_safety_margin
        status = "critical_schur_not_started"
        normalized_residual = np.inf
        solver_converged = False
        maximum_sqp_iterations = min(
            self.settings.maximum_local_corrections,
            self.settings.schur_maximum_iterations,
        )
        for sqp_iteration in range(maximum_sqp_iterations):
            schur, derivatives, trust_value = schur_linearization(current)
            linear_base = (
                schur
                - np.einsum(
                    "j,jab->ab", current, derivatives, optimize=True
                )
                - safety * np.eye(critical_count, dtype=complex)
            )
            # Critical Schur entries are O(1e-9) in the frozen scaled chart.
            # Homogeneous cone scaling keeps Clarabel's absolute feasibility
            # tolerance from treating that entire block as numerical zero.
            schur_cone_scale = 1e5
            linear_base *= schur_cone_scale
            derivatives = derivatives * schur_cone_scale
            real_base = _realify_hermitian(linear_base)
            coefficient_vectors = np.column_stack(
                [
                    _clarabel_svec_upper(_realify_hermitian(coefficient))
                    for coefficient in derivatives
                ]
            )
            cone_blocks = [sparse.csc_matrix(-coefficient_vectors)]
            cone_rhs = [_clarabel_svec_upper(real_base)]
            cones: list[object] = [
                clarabel.PSDTriangleConeT(real_base.shape[0])
            ]
            soc = sparse.vstack(
                (
                    sparse.csc_matrix((1, free.size)),
                    -sparse.eye(free.size, format="csc"),
                ),
                format="csc",
            )
            cone_blocks.append(soc)
            cone_rhs.append(
                np.concatenate(
                    ([np.sqrt(free_radius_squared)], np.zeros(free.size))
                )
            )
            cones.append(clarabel.SecondOrderConeT(free.size + 1))
            solver_settings = clarabel.DefaultSettings()
            solver_settings.verbose = False
            solver_settings.max_threads = 1
            solver_settings.max_iter = self.settings.schur_maximum_iterations
            solver_settings.tol_gap_abs = self.settings.solver_tolerance
            solver_settings.tol_gap_rel = self.settings.solver_tolerance
            solver_settings.tol_feas = self.settings.solver_tolerance
            solver_settings.iterative_refinement_enable = True
            solver_settings.iterative_refinement_max_iter = 10
            solver_settings.iterative_refinement_abstol = 1e-12
            solver_settings.iterative_refinement_reltol = 1e-12
            solution = clarabel.DefaultSolver(
                sparse.eye(free.size, format="csc"),
                np.zeros(free.size),
                sparse.vstack(cone_blocks, format="csc"),
                np.concatenate(cone_rhs),
                cones,
                solver_settings,
            ).solve()
            status = f"SQP{1 + sqp_iteration}:{solution.status}"
            if solution.x is None or str(solution.status) not in {
                "Solved",
                "AlmostSolved",
            }:
                break
            current = np.asarray(solution.x, dtype=float)
            schur, _, trust_value = schur_linearization(current)
            schur_minimum = float(np.linalg.eigvalsh(schur)[0])
            normalized_residual = max(
                0.0,
                safety - schur_minimum,
                -trust_value,
                float(np.dot(current, current)) - free_radius_squared,
                float(solution.r_prim),
                float(solution.r_dual),
            ) / (1.0 + abs(schur_minimum) + abs(trust_value))
            if (
                schur_minimum >= safety
                and trust_value >= 0.0
                and normalized_residual
                <= 10.0 * self.settings.solver_tolerance
            ):
                solver_converged = True
                break

        if not solver_converged:
            cache_key: bytes | None = None
            cache_value: tuple[FloatArray, FloatArray] | None = None

            def schur_spectrum(
                free_velocity: FloatArray,
            ) -> tuple[FloatArray, FloatArray]:
                nonlocal cache_key, cache_value
                key = np.asarray(free_velocity, dtype=float).tobytes()
                if cache_key == key and cache_value is not None:
                    return cache_value
                schur, derivatives, _ = schur_linearization(free_velocity)
                values, vectors = np.linalg.eigh(schur)
                gradients = np.einsum(
                    "rk,jrs,sk->kj",
                    vectors.conjugate(),
                    derivatives,
                    vectors,
                    optimize=True,
                ).real
                cache_key = key
                cache_value = (
                    np.asarray(values, dtype=float),
                    np.asarray(gradients, dtype=float),
                )
                return cache_value

            nonlinear = minimize(
                lambda value: 0.5 * float(np.dot(value, value)),
                current,
                jac=lambda value: np.asarray(value, dtype=float),
                method="SLSQP",
                constraints=(
                    {
                        "type": "ineq",
                        "fun": lambda value: schur_spectrum(value)[0]
                        - safety,
                        "jac": lambda value: schur_spectrum(value)[1],
                    },
                    {
                        "type": "ineq",
                        "fun": lambda value: free_radius_squared
                        - float(np.dot(value, value)),
                        "jac": lambda value: -2.0
                        * np.asarray(value, dtype=float),
                    },
                ),
                options={
                    "ftol": max(1e-14, self.settings.solver_tolerance**2),
                    "maxiter": self.settings.schur_maximum_iterations,
                    "disp": False,
                },
            )
            current = np.asarray(nonlinear.x, dtype=float)
            schur, _, trust_value = schur_linearization(current)
            schur_minimum = float(np.linalg.eigvalsh(schur)[0])
            normalized_residual = max(
                0.0,
                safety - schur_minimum,
                -trust_value,
                float(np.dot(current, current)) - free_radius_squared,
            ) / (1.0 + abs(schur_minimum) + abs(trust_value))
            solver_converged = bool(
                nonlinear.success
                and normalized_residual
                <= 10.0 * self.settings.solver_tolerance
            )
            status += (
                f"->SLSQP:{nonlinear.status}:{nonlinear.message}:"
                f"nit={nonlinear.nit}"
            )

        candidate = desired.copy()
        candidate[free] += current
        schur, _, trust_value = schur_linearization(current)
        schur_value = float(np.linalg.eigvalsh(schur)[0])
        total_norm = np.sqrt(
            retained_norm**2 + float(np.dot(candidate, candidate))
        )
        velocity_margin = 1.0 - total_norm / ceiling
        if (
            schur_value >= safety - 1e-13
            and trust_value < -1e-13
            and (solver_converged or "SLSQP:0:" in status)
            and critical_count < self.settings.critical_mode_limit
        ):
            return self._solve_critical_schur(
                endpoint_retained,
                completion,
                step_size,
                retained_velocity,
                desired,
                minimum_critical_modes=critical_count + 1,
                initial_velocity=candidate,
            )
        success = bool(
            solver_converged
            and schur_value >= safety - 1e-13
            and trust_value >= -1e-13
            and normalized_residual <= 10.0 * self.settings.solver_tolerance
        )
        return (
            success,
            candidate,
            status,
            float(normalized_residual),
            float(velocity_margin),
            critical_count,
        )

    def _solve_relaxed_readable_schur(
        self,
        endpoint_retained: FloatArray,
        completion: FloatArray,
        step_size: float,
        retained_velocity: FloatArray,
        desired: FloatArray,
    ) -> tuple[bool, FloatArray, str, float, float, int]:
        """Apply the readable-rate residual tier when exact locking conflicts.

        The archive retained velocity remains fixed.  Tier one minimizes the
        commutator-readable completion-rate residual over the finite radial
        cone.  A small frozen physical-rate regularizer selects a stable point
        on a numerically flat readable-rate face.  Every candidate is checked
        against the original 62-row Gram by the caller.
        """

        geometry = self.geometry
        readable = geometry.readable_completion_indices
        shifted_predictor = (
            geometry.scaled_unified_matrix(
                endpoint_retained,
                completion + step_size * desired,
            )
            + self.settings.psd_inflation * np.eye(62, dtype=complex)
        )
        predictor_values, predictor_vectors = np.linalg.eigh(
            shifted_predictor
        )
        critical_count = max(
            1,
            int(
                np.count_nonzero(
                    predictor_values < self.settings.spectral_entry_threshold
                )
            ),
        )
        if predictor_values[0] < -self.settings.spectral_entry_threshold:
            critical_count = min(
                self.settings.critical_mode_limit,
                critical_count + 4,
            )
        ceiling = self._velocity_ceiling(retained_velocity, desired)
        retained_norm = self._retained_scaled_norm(retained_velocity)
        completion_radius_squared = ceiling**2 - retained_norm**2
        if completion_radius_squared <= 0.0:
            return (
                False,
                desired.copy(),
                "velocity_prior_active",
                np.inf,
                -np.inf,
                critical_count,
            )

        safety = self.settings.schur_safety_margin
        all_coefficients = (
            step_size * geometry.scaled_completion_coefficients()
        )
        last_status = "relaxed_readable_not_started"
        last_residual = np.inf
        last_candidate = desired.copy()
        while critical_count <= self.settings.critical_mode_limit:
            critical_basis = predictor_vectors[:, :critical_count]
            positive_basis = predictor_vectors[:, critical_count:]
            positive_base = (
                positive_basis.conjugate().T
                @ shifted_predictor
                @ positive_basis
            )
            cross_base = (
                positive_basis.conjugate().T
                @ shifted_predictor
                @ critical_basis
            )
            critical_base = (
                critical_basis.conjugate().T
                @ shifted_predictor
                @ critical_basis
            )
            positive_coefficients = np.einsum(
                "pa,jpq,qb->jab",
                positive_basis.conjugate(),
                all_coefficients,
                positive_basis,
                optimize=True,
            )
            cross_coefficients = np.einsum(
                "pa,jpq,qb->jab",
                positive_basis.conjugate(),
                all_coefficients,
                critical_basis,
                optimize=True,
            )
            critical_coefficients = np.einsum(
                "pa,jpq,qb->jab",
                critical_basis.conjugate(),
                all_coefficients,
                critical_basis,
                optimize=True,
            )
            cache_key: bytes | None = None
            cache_value: tuple[FloatArray, FloatArray, float] | None = None

            def spectrum(
                correction: FloatArray,
            ) -> tuple[FloatArray, FloatArray, float]:
                nonlocal cache_key, cache_value
                key = np.asarray(correction, dtype=float).tobytes()
                if cache_key == key and cache_value is not None:
                    return cache_value
                positive = positive_base + np.einsum(
                    "j,jab->ab",
                    correction,
                    positive_coefficients,
                    optimize=True,
                )
                cross = cross_base + np.einsum(
                    "j,jab->ab",
                    correction,
                    cross_coefficients,
                    optimize=True,
                )
                critical = critical_base + np.einsum(
                    "j,jab->ab",
                    correction,
                    critical_coefficients,
                    optimize=True,
                )
                positive = 0.5 * (positive + positive.conjugate().T)
                critical = 0.5 * (critical + critical.conjugate().T)
                solved_cross = np.linalg.solve(positive, cross)
                schur = critical - cross.conjugate().T @ solved_cross
                schur = 0.5 * (schur + schur.conjugate().T)
                values, vectors = np.linalg.eigh(schur)
                derivatives = (
                    critical_coefficients
                    - np.einsum(
                        "jpr,ps->jrs",
                        cross_coefficients.conjugate(),
                        solved_cross,
                        optimize=True,
                    )
                    - np.einsum(
                        "rp,jps->jrs",
                        solved_cross.conjugate().T,
                        cross_coefficients,
                        optimize=True,
                    )
                    + np.einsum(
                        "rp,jpq,qs->jrs",
                        solved_cross.conjugate().T,
                        positive_coefficients,
                        solved_cross,
                        optimize=True,
                    )
                )
                gradients = np.einsum(
                    "rk,jrs,sk->kj",
                    vectors.conjugate(),
                    derivatives,
                    vectors,
                    optimize=True,
                ).real
                cache_key = key
                cache_value = (
                    np.asarray(values, dtype=float),
                    np.asarray(gradients, dtype=float),
                    float(np.linalg.eigvalsh(positive)[0]),
                )
                return cache_value

            def radial_value(correction: FloatArray) -> float:
                candidate = desired + correction
                return completion_radius_squared - float(
                    np.dot(candidate, candidate)
                )

            def radial_gradient(correction: FloatArray) -> FloatArray:
                return -2.0 * (desired + correction)

            regularization = 1e-8
            tier_one = minimize(
                lambda value: 0.5
                * float(np.dot(value[readable], value[readable]))
                + 0.5
                * regularization
                * float(np.dot(desired + value, desired + value)),
                np.zeros(geometry.completion_count, dtype=float),
                jac=lambda value: np.bincount(
                    readable,
                    weights=value[readable],
                    minlength=geometry.completion_count,
                )
                + regularization * (desired + value),
                method="SLSQP",
                constraints=(
                    {
                        "type": "ineq",
                        "fun": lambda value: spectrum(value)[0] - safety,
                        "jac": lambda value: spectrum(value)[1],
                    },
                    {
                        "type": "ineq",
                        "fun": radial_value,
                        "jac": radial_gradient,
                    },
                ),
                options={
                    "ftol": max(1e-14, self.settings.solver_tolerance**2),
                    "maxiter": self.settings.schur_maximum_iterations,
                    "disp": False,
                },
            )
            correction = np.asarray(tier_one.x, dtype=float)
            candidate = desired + correction
            schur_values, _, positive_minimum = spectrum(correction)
            endpoint_matrix = (
                geometry.scaled_unified_matrix(
                    endpoint_retained,
                    completion + step_size * candidate,
                )
                + self.settings.psd_inflation * np.eye(62, dtype=complex)
            )
            full_minimum = float(np.linalg.eigvalsh(endpoint_matrix)[0])
            last_residual = max(
                0.0,
                safety - float(schur_values[0]),
                self.settings.spectral_entry_threshold - positive_minimum,
                -radial_value(correction),
                -full_minimum,
            ) / (1.0 + abs(float(schur_values[0])) + abs(positive_minimum))
            last_status = (
                f"relaxed:J={critical_count}:"
                f"tier1={tier_one.status}/{tier_one.nit}"
            )
            last_candidate = candidate
            if (
                tier_one.success
                and last_residual
                <= 10.0 * self.settings.solver_tolerance
            ):
                total_norm = np.sqrt(
                    retained_norm**2 + float(np.dot(candidate, candidate))
                )
                return (
                    True,
                    candidate,
                    last_status,
                    float(last_residual),
                    float(1.0 - total_norm / ceiling),
                    critical_count,
                )
            critical_count += 2

        total_norm = np.sqrt(
            retained_norm**2 + float(np.dot(last_candidate, last_candidate))
        )
        return (
            False,
            last_candidate,
            last_status,
            float(last_residual),
            float(1.0 - total_norm / ceiling),
            critical_count,
        )

    def _solve_relaxed_readable_cutting_plane(
        self,
        endpoint_retained: FloatArray,
        completion: FloatArray,
        step_size: float,
        retained_velocity: FloatArray,
        desired: FloatArray,
    ) -> tuple[bool, FloatArray, str, float, float, int]:
        """Solve the convex relaxed-rate tier by certified spectral cuts.

        Each eigenvector of a violated full-Gram candidate supplies a linear
        half-space that contains the PSD cone.  The QP is solved over the
        accumulated outer approximation.  Once its minimizer also passes the
        authoritative full-Gram eigendecomposition, it is the minimizer over
        the complete PSD constraint, not merely over the sampled cuts.
        """

        geometry = self.geometry
        readable = geometry.readable_completion_indices
        shifted_predictor = (
            geometry.scaled_unified_matrix(
                endpoint_retained,
                completion + step_size * desired,
            )
            + self.settings.psd_inflation * np.eye(62, dtype=complex)
        )
        coefficients = step_size * geometry.scaled_completion_coefficients()
        ceiling = self._velocity_ceiling(retained_velocity, desired)
        retained_norm = self._retained_scaled_norm(retained_velocity)
        completion_radius_squared = ceiling**2 - retained_norm**2
        if completion_radius_squared <= 0.0:
            return (
                False,
                desired.copy(),
                "velocity_prior_active",
                np.inf,
                -np.inf,
                0,
            )
        completion_radius = np.sqrt(completion_radius_squared)
        regularization = 1e-6
        weights = np.full(geometry.completion_count, regularization)
        weights[readable] += 1.0
        objective_matrix = sparse.diags(weights, format="csc")
        objective_vector = regularization * desired
        cuts: list[tuple[FloatArray, float]] = []
        correction = np.zeros(geometry.completion_count, dtype=float)
        maximum_bundle = 0
        status = "cutting_plane_not_started"
        normalized_residual = np.inf

        for iteration in range(self.settings.maximum_cutting_plane_iterations):
            candidate_matrix = shifted_predictor + np.einsum(
                "j,jab->ab", correction, coefficients, optimize=True
            )
            candidate_matrix = 0.5 * (
                candidate_matrix + candidate_matrix.conjugate().T
            )
            eigenvalues, eigenvectors = np.linalg.eigh(candidate_matrix)
            violated = np.flatnonzero(
                eigenvalues < self.settings.schur_safety_margin
            )
            bundle = np.flatnonzero(
                eigenvalues < self.settings.spectral_entry_threshold
            )
            maximum_bundle = max(maximum_bundle, int(bundle.size))
            readable_error = float(
                np.linalg.norm(correction[readable])
                / (1.0 + np.linalg.norm(desired[readable]))
            )
            radial_residual = max(
                0.0,
                float(np.dot(desired + correction, desired + correction))
                - completion_radius_squared,
            )
            normalized_residual = max(
                0.0,
                self.settings.schur_safety_margin - float(eigenvalues[0]),
                radial_residual,
            ) / (1.0 + abs(float(eigenvalues[0])))
            if (
                violated.size == 0
                and normalized_residual
                <= 10.0 * self.settings.solver_tolerance
            ):
                total_norm = np.sqrt(
                    retained_norm**2
                    + float(np.dot(desired + correction, desired + correction))
                )
                return (
                    True,
                    desired + correction,
                    f"cutting_plane:Solved:iterations={iteration}",
                    float(normalized_residual),
                    float(1.0 - total_norm / ceiling),
                    maximum_bundle,
                )

            if violated.size:
                vectors = eigenvectors[:, bundle]
                normals = np.einsum(
                    "ak,jab,bk->kj",
                    vectors.conjugate(),
                    coefficients,
                    vectors,
                    optimize=True,
                ).real
                offsets = np.einsum(
                    "ak,ab,bk->k",
                    vectors.conjugate(),
                    shifted_predictor
                    - self.settings.schur_safety_margin
                    * np.eye(62, dtype=complex),
                    vectors,
                    optimize=True,
                ).real
                for normal, offset in zip(normals, offsets, strict=True):
                    scale = 1.0 / max(
                        1e-12,
                        abs(float(offset)),
                        float(np.linalg.norm(normal)),
                    )
                    cuts.append((scale * normal, scale * float(offset)))

            blocks: list[sparse.csc_matrix] = []
            rhs: list[FloatArray] = []
            cones: list[object] = []
            if cuts:
                blocks.append(
                    sparse.csc_matrix(
                        -np.asarray([normal for normal, _ in cuts])
                    )
                )
                rhs.append(np.asarray([offset for _, offset in cuts]))
                cones.append(clarabel.NonnegativeConeT(len(cuts)))
            radial_block = sparse.vstack(
                (
                    sparse.csc_matrix((1, geometry.completion_count)),
                    -sparse.eye(geometry.completion_count, format="csc"),
                ),
                format="csc",
            )
            blocks.append(radial_block)
            rhs.append(
                np.concatenate(([completion_radius], desired.copy()))
            )
            cones.append(
                clarabel.SecondOrderConeT(geometry.completion_count + 1)
            )
            solver_settings = clarabel.DefaultSettings()
            solver_settings.verbose = False
            solver_settings.max_threads = 1
            solver_settings.max_iter = 500
            solver_settings.tol_gap_abs = self.settings.solver_tolerance
            solver_settings.tol_gap_rel = self.settings.solver_tolerance
            solver_settings.tol_feas = self.settings.solver_tolerance
            solver_settings.iterative_refinement_enable = True
            solver_settings.iterative_refinement_max_iter = 10
            solver_settings.iterative_refinement_abstol = 1e-12
            solver_settings.iterative_refinement_reltol = 1e-12
            solution = clarabel.DefaultSolver(
                objective_matrix,
                objective_vector,
                sparse.vstack(blocks, format="csc"),
                np.concatenate(rhs),
                cones,
                solver_settings,
            ).solve()
            status = (
                f"cutting_plane:{solution.status}:iteration={iteration + 1}:"
                f"cuts={len(cuts)}"
            )
            if solution.x is None or str(solution.status) not in {
                "Solved",
                "AlmostSolved",
            }:
                break
            correction = np.asarray(solution.x, dtype=float)

        candidate_matrix = shifted_predictor + np.einsum(
            "j,jab->ab", correction, coefficients, optimize=True
        )
        candidate_matrix = 0.5 * (
            candidate_matrix + candidate_matrix.conjugate().T
        )
        final_minimum = float(np.linalg.eigvalsh(candidate_matrix)[0])
        readable_error = float(
            np.linalg.norm(correction[readable])
            / (1.0 + np.linalg.norm(desired[readable]))
        )
        radial_residual = max(
            0.0,
            float(np.dot(desired + correction, desired + correction))
            - completion_radius_squared,
        )
        normalized_residual = max(
            0.0,
            self.settings.schur_safety_margin - final_minimum,
            radial_residual,
        ) / (1.0 + abs(final_minimum))
        total_norm = np.sqrt(
            retained_norm**2
            + float(np.dot(desired + correction, desired + correction))
        )
        if (
            final_minimum >= self.settings.schur_safety_margin - 1e-13
            and normalized_residual
            <= 10.0 * self.settings.solver_tolerance
        ):
            return (
                True,
                desired + correction,
                f"{status}:limit_feasible",
                float(normalized_residual),
                float(1.0 - total_norm / ceiling),
                maximum_bundle,
            )
        return (
            False,
            desired + correction,
            status,
            float(normalized_residual),
            float(1.0 - total_norm / ceiling),
            maximum_bundle,
        )

    def _solve_relaxed_readable_bundle_cone(
        self,
        endpoint_retained: FloatArray,
        completion: FloatArray,
        step_size: float,
        retained_velocity: FloatArray,
        desired: FloatArray,
    ) -> tuple[bool, FloatArray, str, float, float, int]:
        """Solve the relaxed tier with an adaptively enlarged PSD bundle."""

        geometry = self.geometry
        readable = geometry.readable_completion_indices
        shifted_predictor = (
            geometry.scaled_unified_matrix(
                endpoint_retained,
                completion + step_size * desired,
            )
            + self.settings.psd_inflation * np.eye(62, dtype=complex)
        )
        coefficients = step_size * geometry.scaled_completion_coefficients()
        predictor_values, predictor_vectors = np.linalg.eigh(
            shifted_predictor
        )
        selected = np.flatnonzero(
            predictor_values < self.settings.spectral_entry_threshold
        )
        if selected.size == 0:
            selected = np.asarray([0], dtype=np.int64)
        spectral_rank = int(selected.size)
        required_rank = min(
            self.settings.critical_mode_limit,
            max(spectral_rank, self._bundle_rank_hint),
        )
        selected = np.arange(required_rank, dtype=np.int64)
        basis = predictor_vectors[:, selected]
        maximum_bundle = basis.shape[1]
        backed_off_hint = False
        correction = np.zeros(geometry.completion_count, dtype=float)
        ceiling = self._velocity_ceiling(retained_velocity, desired)
        retained_norm = self._retained_scaled_norm(retained_velocity)
        completion_radius_squared = ceiling**2 - retained_norm**2
        if completion_radius_squared <= 0.0:
            return (
                False,
                desired.copy(),
                "velocity_prior_active",
                np.inf,
                -np.inf,
                maximum_bundle,
            )
        regularization = 1e-6
        weights = np.full(geometry.completion_count, regularization)
        weights[readable] += 1.0
        status = "bundle_cone_not_started"
        normalized_residual = np.inf

        for iteration in range(self.settings.maximum_cutting_plane_iterations):
            compressed_base = (
                basis.conjugate().T @ shifted_predictor @ basis
                - self.settings.schur_safety_margin
                * np.eye(basis.shape[1], dtype=complex)
            )
            compressed_coefficients = np.einsum(
                "pa,jpq,qb->jab",
                basis.conjugate(),
                coefficients,
                basis,
                optimize=True,
            )
            cone_scale = 1e4
            real_base = _realify_hermitian(compressed_base) * cone_scale
            base_vector = _clarabel_svec_upper(real_base)
            coefficient_vectors = np.column_stack(
                [
                    _clarabel_svec_upper(
                        _realify_hermitian(coefficient) * cone_scale
                    )
                    for coefficient in compressed_coefficients
                ]
            )
            radial_block = sparse.vstack(
                (
                    sparse.csc_matrix((1, geometry.completion_count)),
                    -sparse.eye(geometry.completion_count, format="csc"),
                ),
                format="csc",
            )
            cone_matrix = sparse.vstack(
                (
                    sparse.csc_matrix(-coefficient_vectors),
                    radial_block,
                ),
                format="csc",
            )
            cone_rhs = np.concatenate(
                (
                    base_vector,
                    np.asarray([np.sqrt(completion_radius_squared)]),
                    desired,
                )
            )
            cones: list[object] = [
                clarabel.PSDTriangleConeT(real_base.shape[0]),
                clarabel.SecondOrderConeT(geometry.completion_count + 1),
            ]
            solver_settings = clarabel.DefaultSettings()
            solver_settings.verbose = False
            solver_settings.max_threads = 1
            solver_settings.max_iter = 500
            bundle_tolerance = min(self.settings.solver_tolerance, 1e-12)
            solver_settings.tol_gap_abs = bundle_tolerance
            solver_settings.tol_gap_rel = bundle_tolerance
            solver_settings.tol_feas = bundle_tolerance
            solver_settings.iterative_refinement_enable = True
            solver_settings.iterative_refinement_max_iter = 10
            solver_settings.iterative_refinement_abstol = 1e-13
            solver_settings.iterative_refinement_reltol = 1e-13
            solution = clarabel.DefaultSolver(
                sparse.diags(weights, format="csc"),
                regularization * desired,
                cone_matrix,
                cone_rhs,
                cones,
                solver_settings,
            ).solve()
            status = (
                f"bundle_cone:{solution.status}:iteration={iteration + 1}:"
                f"rank={basis.shape[1]}"
            )
            if solution.x is None or str(solution.status) not in {
                "Solved",
                "AlmostSolved",
            }:
                if not backed_off_hint and basis.shape[1] > spectral_rank:
                    backed_off_hint = True
                    retry_rank = max(spectral_rank, basis.shape[1] // 2)
                    self._bundle_rank_hint = retry_rank
                    basis = predictor_vectors[:, :retry_rank]
                    maximum_bundle = basis.shape[1]
                    status += f":retry_with_rank={retry_rank}"
                    continue
                break
            correction = np.asarray(solution.x, dtype=float)
            candidate_matrix = shifted_predictor + np.einsum(
                "j,jab->ab", correction, coefficients, optimize=True
            )
            candidate_matrix = 0.5 * (
                candidate_matrix + candidate_matrix.conjugate().T
            )
            eigenvalues, eigenvectors = np.linalg.eigh(candidate_matrix)
            readable_error = float(
                np.linalg.norm(correction[readable])
                / (1.0 + np.linalg.norm(desired[readable]))
            )
            radial_residual = max(
                0.0,
                float(np.dot(desired + correction, desired + correction))
                - completion_radius_squared,
            )
            normalized_residual = max(
                0.0,
                self.settings.schur_safety_margin - float(eigenvalues[0]),
                radial_residual,
                float(solution.r_prim),
                float(solution.r_dual),
            ) / (1.0 + abs(float(eigenvalues[0])))
            if (
                eigenvalues[0]
                >= self.settings.schur_safety_margin - 1e-13
                and normalized_residual
                <= 10.0 * self.settings.solver_tolerance
            ):
                candidate = desired + correction
                total_norm = np.sqrt(
                    retained_norm**2 + float(np.dot(candidate, candidate))
                )
                if not backed_off_hint:
                    self._bundle_rank_hint = max(
                        self._bundle_rank_hint,
                        maximum_bundle,
                    )
                return (
                    True,
                    candidate,
                    status,
                    float(normalized_residual),
                    float(1.0 - total_norm / ceiling),
                    maximum_bundle,
                )

            new_indices = np.flatnonzero(
                eigenvalues < self.settings.schur_safety_margin
            )
            augmented = np.column_stack(
                (basis, eigenvectors[:, new_indices])
            )
            left, singular_values, _ = np.linalg.svd(
                augmented, full_matrices=False
            )
            rank = int(np.count_nonzero(singular_values > 1e-10))
            if rank <= basis.shape[1]:
                status += ":no_rank_gain"
                polished = self._polish_terminal_completion_velocity(
                    endpoint_retained,
                    completion,
                    step_size,
                    retained_velocity,
                    desired,
                    desired + correction,
                )
                status = f"{status}->{polished[2]}"
                if polished[0]:
                    if not backed_off_hint:
                        self._bundle_rank_hint = max(
                            self._bundle_rank_hint,
                            maximum_bundle,
                            polished[5],
                        )
                    return (
                        polished[0],
                        polished[1],
                        status,
                        polished[3],
                        polished[4],
                        max(maximum_bundle, polished[5]),
                    )
                break
            basis = left[:, :rank]
            maximum_bundle = max(maximum_bundle, rank)

        candidate = desired + correction
        total_norm = np.sqrt(
            retained_norm**2 + float(np.dot(candidate, candidate))
        )
        return (
            False,
            candidate,
            status,
            float(normalized_residual),
            float(1.0 - total_norm / ceiling),
            maximum_bundle,
        )

    def _polish_terminal_completion_velocity(
        self,
        endpoint_retained: FloatArray,
        completion: FloatArray,
        step_size: float,
        retained_velocity: FloatArray,
        desired: FloatArray,
        candidate_velocity: FloatArray,
    ) -> tuple[bool, FloatArray, str, float, float, int]:
        """Repair cone defects while retaining the readable-rate tolerance."""

        geometry = self.geometry
        readable = geometry.readable_completion_indices
        coefficients = step_size * geometry.scaled_completion_coefficients()
        candidate = np.asarray(candidate_velocity, dtype=float).copy()
        ceiling = self._velocity_ceiling(retained_velocity, desired)
        retained_norm = self._retained_scaled_norm(retained_velocity)
        completion_radius_squared = ceiling**2 - retained_norm**2
        maximum_bundle = 0
        status = "terminal_polish_not_started"
        accumulated_normals: list[FloatArray] = []
        accumulated_gaps: list[float] = []

        for iteration in range(self.settings.maximum_cutting_plane_iterations):
            matrix = (
                geometry.scaled_unified_matrix(
                    endpoint_retained,
                    completion + step_size * candidate,
                )
                + self.settings.psd_inflation * np.eye(62, dtype=complex)
            )
            enclosure = self.spectral_enclosure(matrix, shifted=False)
            readable_error = float(
                np.linalg.norm(
                    candidate[readable] - desired[readable]
                )
                / (1.0 + np.linalg.norm(desired[readable]))
            )
            total_norm = np.sqrt(
                retained_norm**2 + float(np.dot(candidate, candidate))
            )
            velocity_margin = 1.0 - total_norm / ceiling
            if (
                enclosure.minimum_lower_bound >= 0.0
                and velocity_margin >= self.settings.velocity_ceiling_margin
            ):
                return (
                    True,
                    candidate,
                    f"terminal_polish:Solved:iterations={iteration}",
                    0.0,
                    float(velocity_margin),
                    maximum_bundle,
                )
            eigenvalues = enclosure.eigenvalues
            eigenvectors = enclosure.eigenvectors
            violated = np.flatnonzero(
                enclosure.lower_bounds < self.settings.schur_safety_margin
            )
            if violated.size == 0:
                return (
                    False,
                    candidate,
                    "terminal_polish:no_spectral_violation:"
                    f"readable={readable_error:.3e}:"
                    f"velocity_margin={velocity_margin:.3e}",
                    float("inf"),
                    float(velocity_margin),
                    maximum_bundle,
                )
            maximum_bundle = max(maximum_bundle, int(violated.size))
            vectors = eigenvectors[:, violated]
            normals = np.einsum(
                "ak,jab,bk->kj",
                vectors.conjugate(),
                coefficients,
                vectors,
                optimize=True,
            ).real
            gaps = self.settings.schur_safety_margin - eigenvalues[violated]
            scales = np.asarray(
                [
                    1.0
                    / max(
                        1e-14,
                        abs(float(gap)),
                        float(np.linalg.norm(normal)),
                    )
                    for normal, gap in zip(normals, gaps, strict=True)
                ]
            )
            for normal, gap, scale in zip(
                normals, gaps, scales, strict=True
            ):
                accumulated_normals.append(scale * normal)
                accumulated_gaps.append(scale * float(gap))
            cut_matrix = sparse.csc_matrix(
                -np.asarray(accumulated_normals)
            )
            cut_rhs = -np.asarray(accumulated_gaps)
            readable_selection = sparse.coo_matrix(
                (
                    np.ones(readable.size),
                    (np.arange(readable.size), readable),
                ),
                shape=(readable.size, geometry.completion_count),
            ).tocsc()
            radial_block = sparse.vstack(
                (
                    sparse.csc_matrix((1, geometry.completion_count)),
                    -sparse.eye(geometry.completion_count, format="csc"),
                ),
                format="csc",
            )
            readable_block = sparse.vstack(
                (
                    sparse.csc_matrix((1, geometry.completion_count)),
                    -readable_selection,
                ),
                format="csc",
            )
            readable_radius = max(
                self.settings.readable_rate_tolerance
                * (1.0 + np.linalg.norm(desired[readable])),
                float(
                    np.linalg.norm(
                        candidate[readable] - desired[readable]
                    )
                ),
            )
            readable_radius *= 1.0 + 1e-8
            cone_matrix = sparse.vstack(
                (cut_matrix, radial_block, readable_block), format="csc"
            )
            cone_rhs = np.concatenate(
                (
                    cut_rhs,
                    np.asarray([np.sqrt(completion_radius_squared)]),
                    candidate,
                    np.asarray([readable_radius]),
                    candidate[readable] - desired[readable],
                )
            )
            cones: list[object] = [
                clarabel.NonnegativeConeT(len(accumulated_gaps)),
                clarabel.SecondOrderConeT(geometry.completion_count + 1),
                clarabel.SecondOrderConeT(readable.size + 1),
            ]
            solver_settings = clarabel.DefaultSettings()
            solver_settings.verbose = False
            solver_settings.max_threads = 1
            solver_settings.max_iter = 200
            solver_settings.tol_gap_abs = 1e-12
            solver_settings.tol_gap_rel = 1e-12
            solver_settings.tol_feas = 1e-12
            polish_weights = np.full(geometry.completion_count, 1e-6)
            polish_weights[readable] = 1.0
            solution = clarabel.DefaultSolver(
                sparse.diags(polish_weights, format="csc"),
                np.zeros(geometry.completion_count),
                cone_matrix,
                cone_rhs,
                cones,
                solver_settings,
            ).solve()
            status = (
                f"terminal_polish:{solution.status}:iteration={iteration + 1}"
            )
            if solution.x is None or str(solution.status) not in {
                "Solved",
                "AlmostSolved",
            }:
                break
            candidate += np.asarray(solution.x, dtype=float)

        total_norm = np.sqrt(
            retained_norm**2 + float(np.dot(candidate, candidate))
        )
        return (
            False,
            candidate,
            status,
            float("inf"),
            float(1.0 - total_norm / ceiling),
            maximum_bundle,
        )

    def _solve_relaxed_readable_full_cone(
        self,
        endpoint_retained: FloatArray,
        completion: FloatArray,
        step_size: float,
        retained_velocity: FloatArray,
        desired: FloatArray,
    ) -> tuple[bool, FloatArray, str, float, float, int]:
        """Solve the relaxed-rate tier over the literal full Gram PSD cone."""

        geometry = self.geometry
        readable = geometry.readable_completion_indices
        shifted_predictor = (
            geometry.scaled_unified_matrix(
                endpoint_retained,
                completion + step_size * desired,
            )
            + self.settings.psd_inflation * np.eye(62, dtype=complex)
            - self.settings.schur_safety_margin * np.eye(62, dtype=complex)
        )
        coefficients = step_size * geometry.scaled_completion_coefficients()
        cone_scale = 1e5
        real_base = _realify_hermitian(shifted_predictor) * cone_scale
        base_vector = _clarabel_svec_upper(real_base)
        coefficient_vectors = np.column_stack(
            [
                _clarabel_svec_upper(
                    _realify_hermitian(coefficient) * cone_scale
                )
                for coefficient in coefficients
            ]
        )
        ceiling = self._velocity_ceiling(retained_velocity, desired)
        retained_norm = self._retained_scaled_norm(retained_velocity)
        completion_radius_squared = ceiling**2 - retained_norm**2
        if completion_radius_squared <= 0.0:
            return (
                False,
                desired.copy(),
                "velocity_prior_active",
                np.inf,
                -np.inf,
                0,
            )
        regularization = 1e-8
        weights = np.full(geometry.completion_count, regularization)
        weights[readable] += 1.0
        radial_block = sparse.vstack(
            (
                sparse.csc_matrix((1, geometry.completion_count)),
                -sparse.eye(geometry.completion_count, format="csc"),
            ),
            format="csc",
        )
        cone_matrix = sparse.vstack(
            (
                sparse.csc_matrix(-coefficient_vectors),
                radial_block,
            ),
            format="csc",
        )
        cone_rhs = np.concatenate(
            (
                base_vector,
                np.asarray([np.sqrt(completion_radius_squared)]),
                desired,
            )
        )
        cones: list[object] = [
            clarabel.PSDTriangleConeT(real_base.shape[0]),
            clarabel.SecondOrderConeT(geometry.completion_count + 1),
        ]
        solver_settings = clarabel.DefaultSettings()
        solver_settings.verbose = False
        solver_settings.max_threads = 1
        solver_settings.max_iter = 500
        solver_settings.tol_gap_abs = self.settings.solver_tolerance
        solver_settings.tol_gap_rel = self.settings.solver_tolerance
        solver_settings.tol_feas = self.settings.solver_tolerance
        solver_settings.iterative_refinement_enable = True
        solver_settings.iterative_refinement_max_iter = 10
        solver_settings.iterative_refinement_abstol = 1e-12
        solver_settings.iterative_refinement_reltol = 1e-12
        solution = clarabel.DefaultSolver(
            sparse.diags(weights, format="csc"),
            regularization * desired,
            cone_matrix,
            cone_rhs,
            cones,
            solver_settings,
        ).solve()
        status = f"full_cone:{solution.status}:iterations={solution.iterations}"
        if solution.x is None:
            return False, desired.copy(), status, np.inf, -np.inf, 62
        correction = np.asarray(solution.x, dtype=float)
        candidate = desired + correction
        endpoint_matrix = (
            geometry.scaled_unified_matrix(
                endpoint_retained,
                completion + step_size * candidate,
            )
            + self.settings.psd_inflation * np.eye(62, dtype=complex)
        )
        minimum = float(np.linalg.eigvalsh(endpoint_matrix)[0])
        readable_error = float(
            np.linalg.norm(correction[readable])
            / (1.0 + np.linalg.norm(desired[readable]))
        )
        radial_residual = max(
            0.0,
            float(np.dot(candidate, candidate))
            - completion_radius_squared,
        )
        normalized_residual = max(
            0.0,
            self.settings.schur_safety_margin - minimum,
            radial_residual,
            float(solution.r_prim),
            float(solution.r_dual),
        ) / (1.0 + abs(minimum))
        total_norm = np.sqrt(
            retained_norm**2 + float(np.dot(candidate, candidate))
        )
        success = bool(
            str(solution.status) in {"Solved", "AlmostSolved"}
            and minimum >= self.settings.schur_safety_margin - 1e-13
            and normalized_residual
            <= 10.0 * self.settings.solver_tolerance
        )
        return (
            success,
            candidate,
            status,
            float(normalized_residual),
            float(1.0 - total_norm / ceiling),
            62,
        )

    def radial_atom(
        self,
        time: float,
        state: FloatArray,
        step_size: float,
        *,
        completion_velocity_guess: FloatArray | None = None,
    ) -> RadialAtomResult:
        started = perf_counter()
        geometry = self.geometry
        retained, completion = geometry.unpack_state(state)
        retained_velocity = self.retained_velocity(time, retained, completion)
        desired = self.desired_completion_velocity(time, retained, completion)
        endpoint_retained = retained + step_size * retained_velocity
        candidate_velocity = desired.copy()
        correction_iterations = 0
        message = "unconstrained predictor"
        velocity_margin = 1.0
        maximum_critical_used = 0

        while True:
            endpoint_completion = completion + step_size * candidate_velocity
            endpoint_matrix = geometry.scaled_unified_matrix(
                endpoint_retained, endpoint_completion
            )
            enclosure = self.spectral_enclosure(endpoint_matrix, shifted=True)
            critical = int(
                np.count_nonzero(
                    enclosure.lower_bounds < self.settings.spectral_entry_threshold
                )
            )
            maximum_critical_used = max(maximum_critical_used, critical)
            if critical > self.settings.critical_mode_limit:
                message = "critical_mode_limit"
                break
            if enclosure.minimum_lower_bound >= 0.0:
                readable_error = float(
                    np.linalg.norm(
                        candidate_velocity[geometry.readable_completion_indices]
                        - desired[geometry.readable_completion_indices]
                    )
                    /
                    (
                        1.0
                        + np.linalg.norm(
                            desired[geometry.readable_completion_indices]
                        )
                    )
                )
                archive_intervention = 0.0
                ceiling = self._velocity_ceiling(retained_velocity, desired)
                total_norm = np.sqrt(
                    self._retained_scaled_norm(retained_velocity) ** 2
                    + float(np.dot(candidate_velocity, candidate_velocity))
                )
                velocity_margin = 1.0 - total_norm / ceiling
                if velocity_margin < self.settings.velocity_ceiling_margin:
                    message = "velocity_prior_active"
                    break
                endpoint = geometry.pack_state(
                    endpoint_retained, endpoint_completion
                )
                unshifted_minimum = float(np.linalg.eigvalsh(endpoint_matrix)[0])
                restriction = geometry.restriction_residual(
                    endpoint_retained, endpoint_completion
                )
                if restriction > self.settings.affine_tolerance:
                    message = "restriction_residual"
                    break
                return RadialAtomResult(
                    success=True,
                    endpoint=endpoint,
                    archive_velocity=retained_velocity,
                    completion_velocity=candidate_velocity,
                    desired_completion_velocity=desired,
                    minimum_unshifted_eigenvalue=unshifted_minimum,
                    minimum_shifted_lower_bound=enclosure.minimum_lower_bound,
                    readable_rate_residual=readable_error,
                    archive_intervention=archive_intervention,
                    completion_correction_norm=(
                        step_size
                        * float(np.linalg.norm(candidate_velocity - desired))
                    ),
                    velocity_margin=velocity_margin,
                    critical_modes=maximum_critical_used,
                    correction_iterations=correction_iterations,
                    elapsed_seconds=perf_counter() - started,
                    message=message,
                )
            if correction_iterations >= self.settings.maximum_local_corrections:
                message = "local_correction_limit"
                break
            if (
                enclosure.minimum_lower_bound
                < -self.settings.spectral_entry_threshold
            ):
                (
                    solved,
                    candidate_velocity,
                    status,
                    residual,
                    velocity_margin,
                    critical,
                ) = self._solve_relaxed_readable_bundle_cone(
                    endpoint_retained,
                    completion,
                    step_size,
                    retained_velocity,
                    desired,
                )
                message = (
                    f"readable_relaxation:{status}:residual={residual:.3e}"
                )
            else:
                (
                    solved,
                    candidate_velocity,
                    status,
                    residual,
                    velocity_margin,
                    critical,
                ) = self._solve_critical_schur(
                    endpoint_retained,
                    completion,
                    step_size,
                    retained_velocity,
                    desired,
                    minimum_critical_modes=critical + correction_iterations,
                    initial_velocity=completion_velocity_guess,
                )
                message = f"critical_schur:{status}:residual={residual:.3e}"
                if not solved:
                    (
                        solved,
                        candidate_velocity,
                        status,
                        residual,
                        velocity_margin,
                        critical,
                    ) = self._solve_relaxed_readable_bundle_cone(
                        endpoint_retained,
                        completion,
                        step_size,
                        retained_velocity,
                        desired,
                    )
                    message = (
                        f"readable_relaxation:{status}:"
                        f"residual={residual:.3e}"
                    )
            correction_iterations += 1
            maximum_critical_used = max(maximum_critical_used, critical)
            if not solved:
                break

        return RadialAtomResult(
            success=False,
            endpoint=np.asarray(state, dtype=float).copy(),
            archive_velocity=retained_velocity,
            completion_velocity=candidate_velocity,
            desired_completion_velocity=desired,
            minimum_unshifted_eigenvalue=float("nan"),
            minimum_shifted_lower_bound=float("nan"),
            readable_rate_residual=float("inf"),
            archive_intervention=0.0,
            completion_correction_norm=float("inf"),
            velocity_margin=velocity_margin,
            critical_modes=maximum_critical_used,
            correction_iterations=correction_iterations,
            elapsed_seconds=perf_counter() - started,
            message=message,
        )


def integrate_cwrmf_ssprk2(
    model: CarriedWitnessModel,
    initial_state: FloatArray,
    *,
    initial_time: float = 0.0,
    final_time: float,
    time_step: float,
    progress: Callable[[str], None] | None = None,
    checkpoint: Callable[[int, float, FloatArray], None] | None = None,
) -> CWRMFTrajectory:
    """Integrate the direct fixed-archive CWRMF branch with SSPRK2."""

    if initial_time < 0.0 or final_time <= initial_time or time_step <= 0.0:
        raise ValueError("times must be ordered and time_step must be positive")
    duration = final_time - initial_time
    intervals = int(round(duration / time_step))
    if not np.isclose(intervals * time_step, duration, atol=1e-12):
        raise ValueError("time interval must be an integer multiple of time_step")
    geometry = model.geometry
    state = np.asarray(initial_state, dtype=float).copy()
    geometry.unpack_state(state)
    times = np.linspace(initial_time, final_time, intervals + 1)
    states = np.empty((intervals + 1, geometry.state_count), dtype=float)
    minima = np.empty(intervals + 1, dtype=float)
    lower_bounds = np.empty(intervals + 1, dtype=float)
    maximum_atom_seconds = np.zeros(intervals + 1, dtype=float)
    iterations = np.zeros(intervals + 1, dtype=np.int64)
    readable = np.zeros(intervals + 1, dtype=float)
    corrections = np.zeros(intervals + 1, dtype=float)
    velocity_margins = np.ones(intervals + 1, dtype=float)
    critical_modes = np.zeros(intervals + 1, dtype=np.int64)
    states[0] = state
    retained, completion = geometry.unpack_state(state)
    initial_matrix = geometry.scaled_unified_matrix(retained, completion)
    initial_enclosure = model.spectral_enclosure(initial_matrix, shifted=True)
    minima[0] = float(np.linalg.eigvalsh(initial_matrix)[0])
    lower_bounds[0] = initial_enclosure.minimum_lower_bound
    atom_evaluations = 0

    for step in range(intervals):
        time = float(times[step])
        first = model.radial_atom(
            time,
            state,
            time_step,
        )
        atom_evaluations += 1
        if not first.success:
            return CWRMFTrajectory(
                times=times[: step + 1],
                states=states[: step + 1],
                minimum_unshifted_eigenvalues=minima[: step + 1],
                minimum_shifted_lower_bounds=lower_bounds[: step + 1],
                maximum_atom_seconds=maximum_atom_seconds[: step + 1],
                correction_iterations=iterations[: step + 1],
                readable_rate_residuals=readable[: step + 1],
                completion_correction_norms=corrections[: step + 1],
                velocity_margins=velocity_margins[: step + 1],
                critical_modes=critical_modes[: step + 1],
                completed_steps=step,
                atom_evaluations=atom_evaluations,
                success=False,
                message=f"first atom at t={time:.6f}: {first.message}",
            )
        second = model.radial_atom(
            time + time_step,
            first.endpoint,
            time_step,
        )
        atom_evaluations += 1
        if not second.success:
            return CWRMFTrajectory(
                times=times[: step + 1],
                states=states[: step + 1],
                minimum_unshifted_eigenvalues=minima[: step + 1],
                minimum_shifted_lower_bounds=lower_bounds[: step + 1],
                maximum_atom_seconds=maximum_atom_seconds[: step + 1],
                correction_iterations=iterations[: step + 1],
                readable_rate_residuals=readable[: step + 1],
                completion_correction_norms=corrections[: step + 1],
                velocity_margins=velocity_margins[: step + 1],
                critical_modes=critical_modes[: step + 1],
                completed_steps=step,
                atom_evaluations=atom_evaluations,
                success=False,
                message=f"second atom at t={time + time_step:.6f}: {second.message}",
            )
        state = 0.5 * state + 0.5 * second.endpoint
        retained, completion = geometry.unpack_state(state)
        matrix = geometry.scaled_unified_matrix(retained, completion)
        enclosure = model.spectral_enclosure(matrix, shifted=True)
        if enclosure.minimum_lower_bound < 0.0:
            return CWRMFTrajectory(
                times=times[: step + 1],
                states=states[: step + 1],
                minimum_unshifted_eigenvalues=minima[: step + 1],
                minimum_shifted_lower_bounds=lower_bounds[: step + 1],
                maximum_atom_seconds=maximum_atom_seconds[: step + 1],
                correction_iterations=iterations[: step + 1],
                readable_rate_residuals=readable[: step + 1],
                completion_correction_norms=corrections[: step + 1],
                velocity_margins=velocity_margins[: step + 1],
                critical_modes=critical_modes[: step + 1],
                completed_steps=step,
                atom_evaluations=atom_evaluations,
                success=False,
                message=f"SSPRK convex node unresolved at t={times[step + 1]:.6f}",
            )
        states[step + 1] = state
        minima[step + 1] = float(np.linalg.eigvalsh(matrix)[0])
        lower_bounds[step + 1] = enclosure.minimum_lower_bound
        maximum_atom_seconds[step + 1] = max(
            first.elapsed_seconds, second.elapsed_seconds
        )
        iterations[step + 1] = max(
            first.correction_iterations, second.correction_iterations
        )
        readable[step + 1] = max(
            first.readable_rate_residual, second.readable_rate_residual
        )
        corrections[step + 1] = max(
            first.completion_correction_norm,
            second.completion_correction_norm,
        )
        velocity_margins[step + 1] = min(
            first.velocity_margin, second.velocity_margin
        )
        critical_modes[step + 1] = max(
            first.critical_modes, second.critical_modes
        )
        if checkpoint is not None:
            checkpoint(step + 1, float(times[step + 1]), state.copy())
        if progress is not None and (
            step == 0
            or step + 1 == intervals
            or (step + 1) % max(1, intervals // 10) == 0
        ):
            progress(
                f"t={times[step + 1]:.5f}/{final_time:.5f} "
                f"lambda_shifted_lo={lower_bounds[step + 1]:.3e} "
                f"corr={corrections[step + 1]:.3e} "
                f"local_it={iterations[step + 1]} "
                f"critical={critical_modes[step + 1]} "
                f"atom_s={maximum_atom_seconds[step + 1]:.3f} "
                f"solver_1={first.message} solver_2={second.message}"
            )

    return CWRMFTrajectory(
        times=times,
        states=states,
        minimum_unshifted_eigenvalues=minima,
        minimum_shifted_lower_bounds=lower_bounds,
        maximum_atom_seconds=maximum_atom_seconds,
        correction_iterations=iterations,
        readable_rate_residuals=readable,
        completion_correction_norms=corrections,
        velocity_margins=velocity_margins,
        critical_modes=critical_modes,
        completed_steps=intervals,
        atom_evaluations=atom_evaluations,
        success=True,
        message="completed direct fixed-archive CWRMF rollout",
    )


__all__ = [
    "CWRMFPreparation",
    "CWRMFSettings",
    "CWRMFTrajectory",
    "CarriedWitnessError",
    "CarriedWitnessGeometry",
    "CarriedWitnessModel",
    "RadialAtomResult",
    "SpectralEnclosure",
    "integrate_cwrmf_ssprk2",
]
