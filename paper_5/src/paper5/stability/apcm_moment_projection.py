"""Moment-metric tangent projection and coupled APCM stage retraction.

This module implements the physical layer prescribed by the APCM design.  It
does not call the legacy finite-rate barrier controller.  At a cone boundary
it first chooses the retained 29-coordinate velocity nearest the augmented
archive target in the lifted block/Frobenius metric, then chooses the hidden
moment velocity in the covariance-dual metric, and finally chooses the
smallest frontier slack.  Trial stages are repaired in the same retained-first
order over the coupled retained and extended cones.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping

import clarabel
import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .adaptive_positive_moment import (
    ARCHIVE_RELATIVE_MOMENT_KEYS,
    HIDDEN_RELATIVE_MOMENT_KEYS,
    RAW_MOMENT_COORDINATE_NAMES,
    raw_moment_coordinates_to_matrix_state,
    raw_moment_velocity_to_matrix_derivative,
    relative_moments_from_matrix_state,
    uncentered_joint_moment_matrix,
)
from .apcm_positive_extension import (
    APCMExtensionResult,
    SymmetryReducedPositiveExtension,
    _clarabel_svec_upper,
    _realify_hermitian,
)
from .hubbard_dimer import DimerParameters, FloatArray
from .matrix_reference import electron_phonon_moment_derivative
from .moment_hierarchy import MomentKey

ComplexArray = NDArray[np.complex128]

_RAW_DIMENSION = len(RAW_MOMENT_COORDINATE_NAMES)


class APCMProjectionError(RuntimeError):
    """The declared lexicographic cone problem could not be certified."""


class APCMMomentEnvelopeError(APCMProjectionError):
    """An active or terminal moment reached the frozen model envelope."""


@dataclass(frozen=True)
class APCMProjectionSettings:
    """Numerical tolerances for the faithful APCM physical layer."""

    phonon_envelope: float = 16.0
    conic_tolerance: float = 1e-9
    null_tolerance_factor: float = 10.0
    maximum_iterations: int = 1_000
    maximum_cvxpy_psd_dimension: int = 16
    maximum_direct_workspace_bytes: int = 512 * 1024**2
    clarabel_max_threads: int = 1

    def __post_init__(self) -> None:
        if self.phonon_envelope <= 0.0:
            raise ValueError("phonon_envelope must be positive")
        if self.conic_tolerance <= 0.0:
            raise ValueError("conic_tolerance must be positive")
        if self.null_tolerance_factor <= 1.0:
            raise ValueError("null_tolerance_factor must exceed one")
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive")
        if self.maximum_cvxpy_psd_dimension <= 0:
            raise ValueError("maximum_cvxpy_psd_dimension must be positive")
        if self.maximum_direct_workspace_bytes <= 0:
            raise ValueError("maximum_direct_workspace_bytes must be positive")
        if self.clarabel_max_threads <= 0:
            raise ValueError("clarabel_max_threads must be positive")


@dataclass(frozen=True)
class APCMVelocityProjection:
    """One retained-first tangent-cone projection result."""

    retained_velocity: FloatArray
    auxiliary_velocity: FloatArray
    frontier_velocity: FloatArray
    retained_metric: FloatArray
    auxiliary_metric: FloatArray
    retained_correction_norm: float
    auxiliary_correction_norm: float
    projection_energy_flux: float
    work_equality_feasible: bool
    base_kernel_dimension: int
    extension_kernel_dimension: int
    iterations: int
    status: str


@dataclass(frozen=True)
class APCMStageRetraction:
    """Retained-first finite-stage correction over both moment cones."""

    raw_coordinates: FloatArray
    hidden_values: FloatArray
    completion: APCMExtensionResult
    retained_correction_norm: float
    auxiliary_correction_norm: float
    iterations: int
    applied: bool
    status: str


@dataclass(frozen=True)
class _DirectConicSolution:
    values: FloatArray
    iterations: int
    success: bool
    status: str
    primal_residual: float
    dual_residual: float


def _real_feature(array: ComplexArray | FloatArray, scale: float) -> FloatArray:
    value = np.asarray(array)
    if np.iscomplexobj(value):
        return np.concatenate((value.real.reshape(-1), value.imag.reshape(-1))) / scale
    return np.asarray(value, dtype=float).reshape(-1) / scale


def _hidden_mapping(
    values: FloatArray,
    active_keys: tuple[MomentKey, ...],
) -> dict[MomentKey, float]:
    array = np.asarray(values, dtype=float)
    if array.shape != (len(active_keys),):
        raise ValueError("hidden moment vector has the wrong dimension")
    return {
        key: float(value)
        for key, value in zip(
            active_keys,
            array,
            strict=True,
        )
    }


def state_lower_moments(
    raw_coordinates: FloatArray,
    hidden_values: FloatArray,
    active_keys: tuple[MomentKey, ...] = HIDDEN_RELATIVE_MOMENT_KEYS,
) -> Mapping[MomentKey, float]:
    """Reconstruct archive-resolved moments plus one active dictionary."""

    matrix_state = raw_moment_coordinates_to_matrix_state(raw_coordinates)
    active = tuple(active_keys)
    required_set = set(ARCHIVE_RELATIVE_MOMENT_KEYS).union(active)
    from .moment_hierarchy import build_moment_keys

    required = tuple(
        key
        for key in build_moment_keys(max(key.degree for key in required_set))
        if key in required_set
    )
    _, archive_moments = relative_moments_from_matrix_state(
        matrix_state,
        {},
        required_keys=ARCHIVE_RELATIVE_MOMENT_KEYS,
    )
    hidden = _hidden_mapping(hidden_values, active)
    moments = {
        key: (
            archive_moments[key]
            if key in archive_moments
            else hidden[key]
        )
        for key in required
    }
    return MappingProxyType(moments)


@lru_cache(maxsize=None)
def _lower_moment_affine_map(
    active_keys: tuple[MomentKey, ...] = HIDDEN_RELATIVE_MOMENT_KEYS,
) -> tuple[FloatArray, FloatArray]:
    """Return ``lower_values = offset + map @ (u,eta)`` exactly."""

    zero_raw = np.zeros(_RAW_DIMENSION, dtype=float)
    hidden_dimension = len(active_keys)
    zero_hidden = np.zeros(hidden_dimension, dtype=float)
    lower_keys = tuple(
        state_lower_moments(zero_raw, zero_hidden, active_keys).keys()
    )
    offset_mapping = state_lower_moments(zero_raw, zero_hidden, active_keys)
    offset = np.asarray([offset_mapping[key] for key in lower_keys])
    affine = np.empty(
        (len(lower_keys), _RAW_DIMENSION + hidden_dimension),
        dtype=float,
    )
    for column in range(affine.shape[1]):
        raw = zero_raw.copy()
        hidden = zero_hidden.copy()
        if column < _RAW_DIMENSION:
            raw[column] = 1.0
        else:
            hidden[column - _RAW_DIMENSION] = 1.0
        mapping = state_lower_moments(raw, hidden, active_keys)
        affine[:, column] = (
            np.asarray([mapping[key] for key in lower_keys]) - offset
        )
    rng = np.random.default_rng(38291)
    probe = rng.normal(scale=0.1, size=affine.shape[1])
    direct = state_lower_moments(
        probe[:_RAW_DIMENSION],
        probe[_RAW_DIMENSION:],
        active_keys,
    )
    predicted = offset + affine @ probe
    if not np.allclose(
        predicted,
        [direct[key] for key in lower_keys],
        atol=3e-13,
        rtol=0.0,
    ):
        raise RuntimeError("relative lower moments are not affine in APCM state")
    return offset, affine


@lru_cache(maxsize=1)
def _base_affine_map() -> tuple[ComplexArray, ComplexArray]:
    zero = np.zeros(_RAW_DIMENSION, dtype=float)
    constant = uncentered_joint_moment_matrix(zero)
    coefficients = np.empty((_RAW_DIMENSION, 8, 8), dtype=complex)
    for column in range(_RAW_DIMENSION):
        basis = zero.copy()
        basis[column] = 1.0
        coefficients[column] = (
            uncentered_joint_moment_matrix(basis) - constant
        )
    return constant, coefficients


def _kernel_basis(
    matrix: ComplexArray,
    tolerance: float,
    *,
    name: str,
) -> ComplexArray:
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (matrix + matrix.conjugate().T)
    )
    if eigenvalues[0] < -10.0 * tolerance:
        raise APCMProjectionError(
            f"{name} is outside its cone: lambda_min={eigenvalues[0]:.3e}"
        )
    ranks = tuple(
        int(np.count_nonzero(eigenvalues <= factor * tolerance))
        for factor in (0.5, 1.0, 2.0)
    )
    if len(set(ranks)) != 1:
        raise APCMProjectionError(
            f"{name} numerical face is unstable under null-tolerance refinement"
        )
    return np.asarray(eigenvectors[:, : ranks[1]], dtype=complex)


def _hermitian_linear_expression(
    coefficients: ComplexArray,
    variable: cp.Expression,
) -> cp.Expression:
    if coefficients.shape[0] != variable.shape[0]:
        raise ValueError("coefficient/variable dimensions do not match")
    expression: cp.Expression = cp.Constant(
        np.zeros(coefficients.shape[1:], dtype=complex)
    )
    for index, coefficient in enumerate(coefficients):
        expression += variable[index] * cp.Constant(coefficient)
    return cp.hermitian_wrap(expression)


def _solver_scaled_metric(metric: FloatArray) -> FloatArray:
    """Rescale one positive metric without changing its minimizer."""

    value = np.asarray(metric, dtype=float)
    eigenvalues = np.linalg.eigvalsh(0.5 * (value + value.T))
    if eigenvalues[0] <= 0.0:
        raise APCMProjectionError("projection metric is not positive definite")
    return np.asarray(value / eigenvalues[0], dtype=float)


class SymmetryReducedAPCMGeometry:
    """Metrics, tangent cones, work row, and coupled stage containment."""

    def __init__(
        self,
        extension: SymmetryReducedPositiveExtension,
        settings: APCMProjectionSettings | None = None,
    ) -> None:
        self.extension = extension
        self.settings = (
            APCMProjectionSettings() if settings is None else settings
        )
        if not np.isclose(
            self.settings.phonon_envelope,
            self.extension.settings.phonon_envelope,
        ):
            raise ValueError("geometry and extension envelopes must agree")
        self.active_keys = tuple(self.extension.active_keys)
        self._hidden_dimension = len(self.active_keys)
        offset, lower_map = _lower_moment_affine_map(self.active_keys)
        lower_probe = state_lower_moments(
            np.zeros(_RAW_DIMENSION, dtype=float),
            np.zeros(self._hidden_dimension, dtype=float),
            self.active_keys,
        )
        if tuple(lower_probe) != self.extension.lower_keys:
            raise RuntimeError(
                "geometry and positive extension use different lower-moment orderings"
            )
        self._lower_offset = offset
        self._lower_map = lower_map
        self._extension_constant = (
            self.extension._constant
            + np.tensordot(
                offset,
                self.extension.lower_coefficients,
                axes=(0, 0),
            )
        )
        self._extension_state_coefficients = np.einsum(
            "ki,kab->iab",
            lower_map,
            self.extension.lower_coefficients,
            optimize=True,
        )
        base_constant, base_coefficients = _base_affine_map()
        self._base_constant = base_constant
        self._base_coefficients = base_coefficients
        base_word_scales = np.asarray(
            [
                1.0,
                *(
                    np.sqrt(self.settings.phonon_envelope + 1.0)
                    for _ in range(4)
                ),
                1.0,
                1.0,
                1.0,
            ],
            dtype=float,
        )
        self._base_inverse_scale = np.diag(1.0 / base_word_scales)
        self._base_scaled_coefficients = np.einsum(
            "ab,kbc,cd->kad",
            self._base_inverse_scale,
            self._base_coefficients,
            self._base_inverse_scale,
            optimize=True,
        )
        inverse_extension = np.diag(1.0 / self.extension.word_scales)
        self._extension_inverse_scale = inverse_extension
        self._extension_scaled_state_coefficients = np.einsum(
            "ab,kbc,cd->kad",
            inverse_extension,
            self._extension_state_coefficients,
            inverse_extension,
            optimize=True,
        )
        self._extension_scaled_frontier_coefficients = np.einsum(
            "ab,kbc,cd->kad",
            inverse_extension,
            self.extension.frontier_coefficients,
            inverse_extension,
            optimize=True,
        )
        self._stage_common_cone_data: (
            tuple[sparse.csc_matrix, FloatArray, tuple[object, ...]] | None
        ) = None

    @property
    def direct_retraction_estimate_bytes(self) -> int:
        """Conservative storage estimate for the directly assembled stage SDP."""

        frontier_dimension = len(self.extension.frontier_keys)
        variable_dimension = (
            _RAW_DIMENSION + self._hidden_dimension + frontier_dimension
        )
        base_real_dimension = 2 * self._base_constant.shape[0]
        extension_real_dimension = 2 * self._extension_constant.shape[0]
        base_rows = base_real_dimension * (base_real_dimension + 1) // 2
        extension_rows = (
            extension_real_dimension * (extension_real_dimension + 1) // 2
        )
        # Eight bytes per numerical value plus a deliberately conservative
        # factor for sparse indices, solver copies, factorization, and iterates.
        affine_entries = (
            base_rows * _RAW_DIMENSION
            + extension_rows * variable_dimension
            + 4 * (self._hidden_dimension + frontier_dimension)
        )
        quadratic_entries = _RAW_DIMENSION**2 + self._hidden_dimension**2
        return int(64 * (affine_entries + quadratic_entries))

    def retained_metric(self, raw_coordinates: FloatArray) -> FloatArray:
        """Assemble ``G_u=J^T W_X J`` from exact reconstruction tangents."""

        raw = np.asarray(raw_coordinates, dtype=float)
        if raw.shape != (_RAW_DIMENSION,):
            raise ValueError("raw_coordinates has the wrong dimension")
        state = raw_moment_coordinates_to_matrix_state(raw)
        envelope = self.settings.phonon_envelope
        s_rho = np.sqrt(2.0)
        s_b = 2.0 * np.sqrt(2.0 * envelope)
        s_n = 4.0 * envelope
        s_a = 4.0 * np.sqrt(envelope * (envelope + 1.0))
        s_c = 8.0 * np.sqrt(2.0 * envelope)
        s_g = np.sqrt(
            2.0 * s_n**2
            + 2.0 * s_a**2
            + 8.0 * s_c**2
            + 25.0 * s_rho**2
        )
        columns: list[FloatArray] = []
        for coordinate in range(_RAW_DIMENSION):
            basis = np.zeros(_RAW_DIMENSION, dtype=float)
            basis[coordinate] = 1.0
            derivative = raw_moment_velocity_to_matrix_derivative(
                state,
                basis,
            )
            joint_derivative = electron_phonon_moment_derivative(
                state,
                derivative,
            )
            columns.append(
                np.concatenate(
                    (
                        _real_feature(derivative.electron_density, s_rho),
                        _real_feature(derivative.coherent_phonon, s_b),
                        _real_feature(derivative.phonon_density, s_n),
                        _real_feature(
                            derivative.anomalous_phonon_density,
                            s_a,
                        ),
                        _real_feature(
                            derivative.electron_phonon_correlation,
                            s_c,
                        ),
                        _real_feature(joint_derivative, s_g),
                    )
                )
            )
        feature = np.column_stack(columns)
        metric = feature.T @ feature
        metric = 0.5 * (metric + metric.T)
        if np.linalg.eigvalsh(metric)[0] <= 0.0:
            raise APCMProjectionError("retained moment metric is not positive definite")
        return np.asarray(metric, dtype=float)

    def frozen_hamiltonian_gradient(
        self,
        time: float,
        parameters: DimerParameters,
    ) -> FloatArray:
        """Return the affine raw-coordinate gradient of ``E_H(u,t)``."""

        gradient = np.zeros(_RAW_DIMENSION, dtype=float)
        gradient[0] = -2.0 * parameters.hopping
        gradient[2] = parameters.drive_difference(float(time))
        gradient[3] = 2.0 * parameters.coupling
        gradient[5] = 2.0 * parameters.coupling
        gradient[7] = parameters.omega_ph
        gradient[8] = parameters.omega_ph
        gradient[21] = 2.0 * parameters.coupling
        gradient[27] = -2.0 * parameters.coupling
        return gradient

    def validate_envelopes(
        self,
        hidden_values: FloatArray,
        completion: APCMExtensionResult,
    ) -> None:
        tolerance = 10.0 * self.settings.conic_tolerance
        active_ratio = np.abs(hidden_values) / self.extension.active_scales
        frontier_values = np.asarray(
            [
                completion.frontier_moments[key]
                for key in self.extension.frontier_keys
            ],
            dtype=float,
        )
        frontier_ratio = np.abs(frontier_values) / self.extension.frontier_scales
        if max(float(np.max(active_ratio)), float(np.max(frontier_ratio))) >= 1.0 - tolerance:
            raise APCMMomentEnvelopeError(
                "active positive-extension moment reached its frozen envelope"
            )

    def project_velocity(
        self,
        time: float,
        raw_coordinates: FloatArray,
        hidden_values: FloatArray,
        target_retained_velocity: FloatArray,
        target_auxiliary_velocity: FloatArray,
        completion: APCMExtensionResult,
        parameters: DimerParameters,
    ) -> APCMVelocityProjection:
        """Solve the retained-first lexicographic tangent-cone problem."""

        raw = np.asarray(raw_coordinates, dtype=float)
        hidden = np.asarray(hidden_values, dtype=float)
        f_u = np.asarray(target_retained_velocity, dtype=float)
        g_eta = np.asarray(target_auxiliary_velocity, dtype=float)
        if raw.shape != (_RAW_DIMENSION,) or f_u.shape != (_RAW_DIMENSION,):
            raise ValueError("retained vectors have the wrong dimension")
        if hidden.shape != (self._hidden_dimension,) or g_eta.shape != (
            self._hidden_dimension,
        ):
            raise ValueError("auxiliary vectors have the wrong dimension")
        if not completion.success:
            raise APCMProjectionError("cannot project from an uncertified completion")
        self.validate_envelopes(hidden, completion)

        retained_metric = self.retained_metric(raw)
        auxiliary_metric = self.extension.auxiliary_metric(
            completion,
            hidden,
        )
        base = uncentered_joint_moment_matrix(raw)
        scaled_base = self._base_inverse_scale @ base @ self._base_inverse_scale
        tolerance = (
            self.settings.null_tolerance_factor
            * self.settings.conic_tolerance
        )
        base_kernel = _kernel_basis(
            scaled_base,
            tolerance,
            name="retained affine moment matrix",
        )
        scaled_extension = self.extension.scaled_matrix(
            completion.moment_matrix
        )
        extension_kernel = _kernel_basis(
            scaled_extension,
            tolerance,
            name="extended operator moment matrix",
        )
        if (
            extension_kernel.shape[1] > 0
            and not completion.facial_reduction_certified
        ):
            raise APCMProjectionError(
                "moment_extension_degeneracy: the selected lifted face lacks "
                "a relative Slater/facial-reduction certificate"
            )
        if base_kernel.shape[1] == 0 and extension_kernel.shape[1] == 0:
            return APCMVelocityProjection(
                retained_velocity=f_u.copy(),
                auxiliary_velocity=g_eta.copy(),
                frontier_velocity=np.zeros(len(self.extension.frontier_keys)),
                retained_metric=retained_metric,
                auxiliary_metric=auxiliary_metric,
                retained_correction_norm=0.0,
                auxiliary_correction_norm=0.0,
                projection_energy_flux=0.0,
                work_equality_feasible=True,
                base_kernel_dimension=0,
                extension_kernel_dimension=0,
                iterations=0,
                status="interior_identity",
            )

        zero_frontier = np.zeros(len(self.extension.frontier_keys))
        if self._fixed_tangent_is_viable(
            f_u,
            g_eta,
            zero_frontier,
            base_kernel,
            extension_kernel,
        ):
            return APCMVelocityProjection(
                retained_velocity=f_u.copy(),
                auxiliary_velocity=g_eta.copy(),
                frontier_velocity=zero_frontier,
                retained_metric=retained_metric,
                auxiliary_metric=auxiliary_metric,
                retained_correction_norm=0.0,
                auxiliary_correction_norm=0.0,
                projection_energy_flux=0.0,
                work_equality_feasible=True,
                base_kernel_dimension=base_kernel.shape[1],
                extension_kernel_dimension=extension_kernel.shape[1],
                iterations=0,
                status="relative_face_identity",
            )

        work_row = self.frozen_hamiltonian_gradient(time, parameters)
        stage_one = self._solve_retained_tier(
            f_u,
            retained_metric,
            base_kernel,
            extension_kernel,
            work_row,
        )
        v_star, work_feasible, stage_one_iterations, status = stage_one
        w_star, stage_two_iterations = self._solve_auxiliary_tier(
            v_star,
            g_eta,
            auxiliary_metric,
            base_kernel,
            extension_kernel,
        )
        s_star, stage_three_iterations = self._solve_frontier_tier(
            v_star,
            w_star,
            extension_kernel,
        )
        retained_delta = v_star - f_u
        auxiliary_delta = w_star - g_eta
        return APCMVelocityProjection(
            retained_velocity=v_star,
            auxiliary_velocity=w_star,
            frontier_velocity=s_star,
            retained_metric=retained_metric,
            auxiliary_metric=auxiliary_metric,
            retained_correction_norm=float(
                np.sqrt(max(0.0, retained_delta @ retained_metric @ retained_delta))
            ),
            auxiliary_correction_norm=float(
                np.sqrt(max(0.0, auxiliary_delta @ auxiliary_metric @ auxiliary_delta))
            ),
            projection_energy_flux=float(work_row @ retained_delta),
            work_equality_feasible=work_feasible,
            base_kernel_dimension=base_kernel.shape[1],
            extension_kernel_dimension=extension_kernel.shape[1],
            iterations=(
                stage_one_iterations
                + stage_two_iterations
                + stage_three_iterations
            ),
            status=status,
        )

    def _compressed_coefficients(
        self,
        coefficients: ComplexArray,
        kernel: ComplexArray,
    ) -> ComplexArray:
        if kernel.shape[1] == 0:
            return np.empty((coefficients.shape[0], 0, 0), dtype=complex)
        return np.einsum(
            "ai,kab,bj->kij",
            kernel.conjugate(),
            coefficients,
            kernel,
            optimize=True,
        )

    def _fixed_tangent_is_viable(
        self,
        retained_velocity: FloatArray,
        auxiliary_velocity: FloatArray,
        frontier_velocity: FloatArray,
        base_kernel: ComplexArray,
        extension_kernel: ComplexArray,
    ) -> bool:
        tolerance = 10.0 * self.settings.conic_tolerance
        if base_kernel.shape[1] > 0:
            coefficients = self._compressed_coefficients(
                self._base_scaled_coefficients,
                base_kernel,
            )
            derivative = np.tensordot(
                retained_velocity,
                coefficients,
                axes=(0, 0),
            )
            if np.linalg.eigvalsh(derivative)[0] < -tolerance:
                return False
        if extension_kernel.shape[1] > 0:
            state_coefficients = self._compressed_coefficients(
                self._extension_scaled_state_coefficients,
                extension_kernel,
            )
            frontier_coefficients = self._compressed_coefficients(
                self._extension_scaled_frontier_coefficients,
                extension_kernel,
            )
            derivative = np.tensordot(
                retained_velocity,
                state_coefficients[:_RAW_DIMENSION],
                axes=(0, 0),
            )
            derivative += np.tensordot(
                auxiliary_velocity,
                state_coefficients[_RAW_DIMENSION:],
                axes=(0, 0),
            )
            derivative += np.tensordot(
                frontier_velocity,
                frontier_coefficients,
                axes=(0, 0),
            )
            if np.linalg.eigvalsh(derivative)[0] < -tolerance:
                return False
        return True

    def _tangent_constraints(
        self,
        v: cp.Variable | FloatArray,
        w: cp.Variable | FloatArray,
        s: cp.Variable | FloatArray,
        base_kernel: ComplexArray,
        extension_kernel: ComplexArray,
    ) -> list[cp.Constraint]:
        constraints: list[cp.Constraint] = []
        if base_kernel.shape[1] > 0:
            base_coefficients = self._compressed_coefficients(
                self._base_scaled_coefficients,
                base_kernel,
            )
            if isinstance(v, cp.Variable):
                constraints.append(
                    _hermitian_linear_expression(base_coefficients, v) >> 0.0
                )
            else:
                matrix = np.tensordot(v, base_coefficients, axes=(0, 0))
                if np.linalg.eigvalsh(matrix)[0] < -10.0 * self.settings.conic_tolerance:
                    raise APCMProjectionError("fixed retained velocity is not base viable")
        if extension_kernel.shape[1] > 0:
            state_coefficients = self._compressed_coefficients(
                self._extension_scaled_state_coefficients,
                extension_kernel,
            )
            frontier_coefficients = self._compressed_coefficients(
                self._extension_scaled_frontier_coefficients,
                extension_kernel,
            )
            expression: cp.Expression = cp.Constant(
                np.zeros((extension_kernel.shape[1],) * 2, dtype=complex)
            )
            retained_coefficients = state_coefficients[:_RAW_DIMENSION]
            auxiliary_coefficients = state_coefficients[_RAW_DIMENSION:]
            if isinstance(v, cp.Variable):
                expression += _hermitian_linear_expression(
                    retained_coefficients,
                    v,
                )
            else:
                expression += cp.Constant(
                    np.tensordot(v, retained_coefficients, axes=(0, 0))
                )
            if isinstance(w, cp.Variable):
                expression += _hermitian_linear_expression(
                    auxiliary_coefficients,
                    w,
                )
            else:
                expression += cp.Constant(
                    np.tensordot(w, auxiliary_coefficients, axes=(0, 0))
                )
            if isinstance(s, cp.Variable):
                expression += _hermitian_linear_expression(
                    frontier_coefficients,
                    s,
                )
            else:
                expression += cp.Constant(
                    np.tensordot(s, frontier_coefficients, axes=(0, 0))
                )
            constraints.append(cp.hermitian_wrap(expression) >> 0.0)
        return constraints

    def _stage_cone_data(
        self,
    ) -> tuple[sparse.csc_matrix, FloatArray, tuple[object, ...]]:
        """Return one bounded direct representation of the coupled stage cone."""

        if self._stage_common_cone_data is not None:
            return self._stage_common_cone_data
        estimate = self.direct_retraction_estimate_bytes
        budget = self.settings.maximum_direct_workspace_bytes
        if estimate > budget:
            raise APCMProjectionError(
                "resource preflight rejected direct stage retraction: "
                f"estimated {estimate / 1024**2:.1f} MiB exceeds "
                f"the {budget / 1024**2:.1f} MiB budget"
            )

        frontier_dimension = len(self.extension.frontier_keys)
        variable_dimension = (
            _RAW_DIMENSION + self._hidden_dimension + frontier_dimension
        )
        stage_margin = 100.0 * max(
            self.settings.conic_tolerance,
            self.extension.settings.conic_tolerance,
        )
        scaled_base_constant = (
            self._base_inverse_scale
            @ self._base_constant
            @ self._base_inverse_scale
        ) - stage_margin * np.eye(self._base_constant.shape[0], dtype=complex)
        real_base_constant = _realify_hermitian(scaled_base_constant)
        base_columns = np.column_stack(
            [
                _clarabel_svec_upper(_realify_hermitian(coefficient))
                for coefficient in self._base_scaled_coefficients
            ]
        )
        base_operator = sparse.hstack(
            (
                sparse.csc_matrix(-base_columns),
                sparse.csc_matrix(
                    (
                        base_columns.shape[0],
                        variable_dimension - _RAW_DIMENSION,
                    )
                ),
            ),
            format="csc",
        )
        base_rhs = _clarabel_svec_upper(real_base_constant)

        scaled_extension_constant = (
            self._extension_inverse_scale
            @ self._extension_constant
            @ self._extension_inverse_scale
        )
        standardized_extension_coefficients = np.concatenate(
            (
                self._extension_scaled_state_coefficients[:_RAW_DIMENSION],
                self._extension_scaled_state_coefficients[_RAW_DIMENSION:]
                * self.extension.active_scales[:, None, None],
                self._extension_scaled_frontier_coefficients
                * self.extension.frontier_scales[:, None, None],
            ),
            axis=0,
        )
        if standardized_extension_coefficients.shape[0] != variable_dimension:
            raise RuntimeError("stage extension coefficient count is inconsistent")
        extension_columns = np.column_stack(
            [
                _clarabel_svec_upper(_realify_hermitian(coefficient))
                for coefficient in standardized_extension_coefficients
            ]
        )
        extension_operator = sparse.csc_matrix(-extension_columns)
        extension_rhs = _clarabel_svec_upper(
            _realify_hermitian(scaled_extension_constant)
        )

        standardized_dimension = self._hidden_dimension + frontier_dimension
        standardized_selector = sparse.hstack(
            (
                sparse.csc_matrix((standardized_dimension, _RAW_DIMENSION)),
                sparse.eye(standardized_dimension, format="csc"),
            ),
            format="csc",
        )
        bound_operator = sparse.vstack(
            (standardized_selector, -standardized_selector),
            format="csc",
        )
        bound_rhs = np.ones(2 * standardized_dimension, dtype=float)
        operator = sparse.vstack(
            (base_operator, extension_operator, bound_operator),
            format="csc",
        )
        rhs = np.concatenate((base_rhs, extension_rhs, bound_rhs))
        cones: tuple[object, ...] = (
            clarabel.PSDTriangleConeT(real_base_constant.shape[0]),
            clarabel.PSDTriangleConeT(
                2 * scaled_extension_constant.shape[0]
            ),
            clarabel.NonnegativeConeT(2 * standardized_dimension),
        )
        self._stage_common_cone_data = (operator, rhs, cones)
        return self._stage_common_cone_data

    def _solve_direct_stage_tier(
        self,
        metric: FloatArray,
        target: FloatArray,
        *,
        block_offset: int,
        fixed_raw: FloatArray | None = None,
        fixed_raw_tolerance: float | None = None,
    ) -> _DirectConicSolution:
        """Solve one lexicographic stage tier without a CVXPY graph."""

        cone_operator, cone_rhs, cones = self._stage_cone_data()
        solver_tolerance = max(
            self.settings.conic_tolerance,
            self.extension.settings.conic_tolerance,
        )
        frontier_dimension = len(self.extension.frontier_keys)
        variable_dimension = (
            _RAW_DIMENSION + self._hidden_dimension + frontier_dimension
        )
        block_metric = _solver_scaled_metric(metric)
        target_value = np.asarray(target, dtype=float)
        block_dimension = block_metric.shape[0]
        if block_metric.shape != (block_dimension, block_dimension):
            raise ValueError("stage metric must be square")
        if target_value.shape != (block_dimension,):
            raise ValueError("stage target has the wrong dimension")
        if block_offset < 0 or block_offset + block_dimension > variable_dimension:
            raise ValueError("stage objective block lies outside the variables")

        rows, columns = np.triu_indices(block_dimension)
        quadratic = sparse.coo_matrix(
            (
                block_metric[rows, columns],
                (rows + block_offset, columns + block_offset),
            ),
            shape=(variable_dimension, variable_dimension),
        ).tocsc()
        linear = np.zeros(variable_dimension, dtype=float)
        linear[block_offset : block_offset + block_dimension] = (
            -block_metric @ target_value
        )

        operator = cone_operator
        rhs = cone_rhs
        selected_cones = cones
        if fixed_raw is not None:
            raw_value = np.asarray(fixed_raw, dtype=float)
            if raw_value.shape != (_RAW_DIMENSION,):
                raise ValueError("fixed retained stage has the wrong dimension")
            equality_margin = (
                100.0 * solver_tolerance
                if fixed_raw_tolerance is None
                else float(fixed_raw_tolerance)
            )
            if equality_margin <= 0.0:
                raise ValueError("fixed_raw_tolerance must be positive")
            selector = sparse.hstack(
                (
                    sparse.eye(_RAW_DIMENSION, format="csc"),
                    sparse.csc_matrix(
                        (
                            _RAW_DIMENSION,
                            variable_dimension - _RAW_DIMENSION,
                        )
                    ),
                ),
                format="csc",
            )
            equality_operator = sparse.vstack(
                (selector, -selector),
                format="csc",
            )
            operator = sparse.vstack(
                (operator, equality_operator),
                format="csc",
            )
            rhs = np.concatenate(
                (
                    rhs,
                    raw_value + equality_margin,
                    -raw_value + equality_margin,
                )
            )
            selected_cones = (
                *selected_cones,
                clarabel.NonnegativeConeT(2 * _RAW_DIMENSION),
            )

        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = self.settings.maximum_iterations
        settings.max_threads = self.settings.clarabel_max_threads
        settings.tol_gap_abs = solver_tolerance
        settings.tol_gap_rel = solver_tolerance
        settings.tol_feas = solver_tolerance
        try:
            solver = clarabel.DefaultSolver(
                quadratic,
                linear,
                operator,
                rhs,
                list(selected_cones),
                settings,
            )
            solution = solver.solve()
        except (RuntimeError, ValueError) as error:
            raise APCMProjectionError(
                f"direct Clarabel stage failure: {error}"
            ) from error
        status = str(solution.status)
        primal_residual = float(solution.r_prim)
        dual_residual = float(solution.r_dual)
        residual = max(primal_residual, dual_residual)
        success = bool(
            solution.x is not None
            and (
                status in ("Solved", "AlmostSolved")
                or (
                    status == "NumericalError"
                    and residual <= 100.0 * solver_tolerance
                )
            )
        )
        values = (
            np.asarray(solution.x, dtype=float)
            if solution.x is not None
            else np.zeros(variable_dimension, dtype=float)
        )
        return _DirectConicSolution(
            values=values,
            iterations=int(solution.iterations),
            success=success,
            status=status,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
        )

    def _solve_problem(self, problem: cp.Problem) -> tuple[int, str]:
        for constraint in problem.constraints:
            if isinstance(constraint, cp.constraints.psd.PSD):
                dimension = int(constraint.shape[0])
                if dimension > self.settings.maximum_cvxpy_psd_dimension:
                    raise APCMProjectionError(
                        "resource preflight rejected CVXPY PSD canonicalization: "
                        f"dimension {dimension} exceeds the bounded limit "
                        f"{self.settings.maximum_cvxpy_psd_dimension}"
                    )
        try:
            problem.solve(
                solver=cp.CLARABEL,
                warm_start=True,
                verbose=False,
                max_iter=self.settings.maximum_iterations,
                tol_gap_abs=self.settings.conic_tolerance,
                tol_gap_rel=self.settings.conic_tolerance,
                tol_feas=self.settings.conic_tolerance,
                max_threads=self.settings.clarabel_max_threads,
            )
        except cp.error.SolverError as error:
            raise APCMProjectionError(f"CLARABEL projection failure: {error}") from error
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise APCMProjectionError(f"conic projection status {problem.status}")
        return int(problem.solver_stats.num_iters or 0), str(problem.status)

    def _solve_retained_tier(
        self,
        f_u: FloatArray,
        metric: FloatArray,
        base_kernel: ComplexArray,
        extension_kernel: ComplexArray,
        work_row: FloatArray,
    ) -> tuple[FloatArray, bool, int, str]:
        solver_metric = _solver_scaled_metric(metric)
        v = cp.Variable(_RAW_DIMENSION)
        w = cp.Variable(self._hidden_dimension)
        s = cp.Variable(len(self.extension.frontier_keys))
        constraints = self._tangent_constraints(
            v,
            w,
            s,
            base_kernel,
            extension_kernel,
        )
        work = work_row @ (v - f_u)
        strict_problem = cp.Problem(
            cp.Minimize(
                0.5
                * cp.quad_form(v - f_u, cp.psd_wrap(solver_metric))
            ),
            [*constraints, work == 0.0],
        )
        try:
            iterations, status = self._solve_problem(strict_problem)
            if v.value is None:
                raise APCMProjectionError("retained tier returned no velocity")
            return np.asarray(v.value, dtype=float), True, iterations, status
        except APCMProjectionError:
            pass

        residual = cp.Variable(nonneg=True)
        residual_problem = cp.Problem(
            cp.Minimize(residual),
            [*constraints, work <= residual, -work <= residual],
        )
        iterations_one, _ = self._solve_problem(residual_problem)
        if residual.value is None:
            raise APCMProjectionError("work fallback returned no residual")
        residual_star = float(residual.value)
        fallback_problem = cp.Problem(
            cp.Minimize(
                0.5
                * cp.quad_form(v - f_u, cp.psd_wrap(solver_metric))
            ),
            [
                *constraints,
                work <= residual_star + self.settings.conic_tolerance,
                -work <= residual_star + self.settings.conic_tolerance,
            ],
        )
        iterations_two, status = self._solve_problem(fallback_problem)
        if v.value is None:
            raise APCMProjectionError("retained fallback returned no velocity")
        return (
            np.asarray(v.value, dtype=float),
            False,
            iterations_one + iterations_two,
            f"{status}:work_fallback",
        )

    def _solve_auxiliary_tier(
        self,
        v_star: FloatArray,
        g_eta: FloatArray,
        metric: FloatArray,
        base_kernel: ComplexArray,
        extension_kernel: ComplexArray,
    ) -> tuple[FloatArray, int]:
        if extension_kernel.shape[1] == 0:
            return g_eta.copy(), 0
        solver_metric = _solver_scaled_metric(metric)
        w = cp.Variable(self._hidden_dimension)
        s = cp.Variable(len(self.extension.frontier_keys))
        constraints = self._tangent_constraints(
            v_star,
            w,
            s,
            base_kernel,
            extension_kernel,
        )
        problem = cp.Problem(
            cp.Minimize(
                0.5
                * cp.quad_form(w - g_eta, cp.psd_wrap(solver_metric))
            ),
            constraints,
        )
        iterations, _ = self._solve_problem(problem)
        if w.value is None:
            raise APCMProjectionError("auxiliary tier returned no velocity")
        return np.asarray(w.value, dtype=float), iterations

    def _solve_frontier_tier(
        self,
        v_star: FloatArray,
        w_star: FloatArray,
        extension_kernel: ComplexArray,
    ) -> tuple[FloatArray, int]:
        if extension_kernel.shape[1] == 0:
            return np.zeros(len(self.extension.frontier_keys)), 0
        s = cp.Variable(len(self.extension.frontier_keys))
        constraints = self._tangent_constraints(
            v_star,
            w_star,
            s,
            np.empty((8, 0), dtype=complex),
            extension_kernel,
        )
        problem = cp.Problem(
            cp.Minimize(
                0.5
                * cp.sum_squares(
                    cp.multiply(1.0 / self.extension.frontier_scales, s)
                )
            ),
            constraints,
        )
        iterations, _ = self._solve_problem(problem)
        if s.value is None:
            raise APCMProjectionError("frontier tier returned no velocity")
        return np.asarray(s.value, dtype=float), iterations

    def retract_stage(
        self,
        raw_trial: FloatArray,
        hidden_trial: FloatArray,
        completion: APCMExtensionResult,
        *,
        retained_metric: FloatArray,
        auxiliary_metric: FloatArray,
    ) -> APCMStageRetraction:
        """Retract an infeasible trial over the coupled base/extension fiber."""

        raw = np.asarray(raw_trial, dtype=float)
        hidden = np.asarray(hidden_trial, dtype=float)
        base = uncentered_joint_moment_matrix(raw)
        base_minimum = float(
            np.linalg.eigvalsh(
                self._base_inverse_scale @ base @ self._base_inverse_scale
            )[0]
        )
        if completion.success:
            extension_minimum = completion.scaled_minimum_eigenvalue
        else:
            extension_minimum = float("-inf")
        if (
            base_minimum >= -self.settings.conic_tolerance
            and extension_minimum >= -self.settings.conic_tolerance
        ):
            self.validate_envelopes(hidden, completion)
            return APCMStageRetraction(
                raw_coordinates=raw.copy(),
                hidden_values=hidden.copy(),
                completion=completion,
                retained_correction_norm=0.0,
                auxiliary_correction_norm=0.0,
                iterations=0,
                applied=False,
                status="already_feasible",
            )

        first_solution = self._solve_direct_stage_tier(
            retained_metric,
            raw,
            block_offset=0,
        )
        if not first_solution.success:
            raise APCMProjectionError(
                "retained stage retraction failed: "
                f"{first_solution.status}; primal="
                f"{first_solution.primal_residual:.3e}; dual="
                f"{first_solution.dual_residual:.3e}"
            )
        raw_star = first_solution.values[:_RAW_DIMENSION]

        active_scale = np.diag(self.extension.active_scales)
        standardized_auxiliary_metric = (
            active_scale @ auxiliary_metric @ active_scale
        )
        standardized_hidden = hidden / self.extension.active_scales
        second_solution = self._solve_direct_stage_tier(
            standardized_auxiliary_metric,
            standardized_hidden,
            block_offset=_RAW_DIMENSION,
            fixed_raw=raw_star,
            fixed_raw_tolerance=max(
                100.0
                * max(
                    self.settings.conic_tolerance,
                    self.extension.settings.conic_tolerance,
                ),
                10.0 * first_solution.primal_residual,
            ),
        )
        if not second_solution.success:
            raise APCMProjectionError(
                "auxiliary stage retraction failed: "
                f"{second_solution.status}; primal="
                f"{second_solution.primal_residual:.3e}; dual="
                f"{second_solution.dual_residual:.3e}"
            )
        raw_star = second_solution.values[:_RAW_DIMENSION]
        hidden_star = (
            second_solution.values[
                _RAW_DIMENSION : _RAW_DIMENSION + self._hidden_dimension
            ]
            * self.extension.active_scales
        )
        frontier_values = (
            second_solution.values[_RAW_DIMENSION + self._hidden_dimension :]
            * self.extension.frontier_scales
        )
        frontier_warm = {
            key: float(value)
            for key, value in zip(
                self.extension.frontier_keys,
                frontier_values,
                strict=True,
            )
        }
        lower = state_lower_moments(
            raw_star,
            hidden_star,
            self.active_keys,
        )
        direct_base_minimum = float(
            np.linalg.eigvalsh(
                self._base_inverse_scale
                @ uncentered_joint_moment_matrix(raw_star)
                @ self._base_inverse_scale
            )[0]
        )
        direct_extension_matrix = self.extension.matrix(lower, frontier_warm)
        direct_extension_minimum = float(
            np.linalg.eigvalsh(
                self.extension.scaled_matrix(direct_extension_matrix)
            )[0]
        )
        direct_tolerance = 100.0 * max(
            self.settings.conic_tolerance,
            self.extension.settings.conic_tolerance,
        )
        if min(direct_base_minimum, direct_extension_minimum) < -direct_tolerance:
            raise APCMProjectionError(
                "direct stage certificate violated a PSD cone: "
                f"base={direct_base_minimum:.3e}, "
                f"extension={direct_extension_minimum:.3e}"
            )
        if (
            np.max(np.abs(hidden_star) / self.extension.active_scales)
            > 1.0 + direct_tolerance
            or np.max(np.abs(frontier_values) / self.extension.frontier_scales)
            > 1.0 + direct_tolerance
        ):
            raise APCMProjectionError(
                "direct stage certificate violated a moment envelope"
            )
        selected = self.extension.complete(
            lower,
            warm_frontier=frontier_warm,
        )
        if not selected.success:
            raise APCMProjectionError(
                "retracted lower moments did not admit the declared selector: "
                f"{selected.message}"
            )
        self.validate_envelopes(hidden_star, selected)
        retained_delta = raw_star - raw
        auxiliary_delta = hidden_star - hidden
        return APCMStageRetraction(
            raw_coordinates=raw_star,
            hidden_values=hidden_star,
            completion=selected,
            retained_correction_norm=float(
                np.sqrt(max(0.0, retained_delta @ retained_metric @ retained_delta))
            ),
            auxiliary_correction_norm=float(
                np.sqrt(max(0.0, auxiliary_delta @ auxiliary_metric @ auxiliary_delta))
            ),
            iterations=(
                first_solution.iterations
                + second_solution.iterations
                + selected.iterations
            ),
            applied=True,
            status=(
                "direct_clarabel:"
                f"{first_solution.status}/{second_solution.status}"
            ),
        )


__all__ = [
    "APCMMomentEnvelopeError",
    "APCMProjectionError",
    "APCMProjectionSettings",
    "APCMStageRetraction",
    "APCMVelocityProjection",
    "SymmetryReducedAPCMGeometry",
    "state_lower_moments",
]
