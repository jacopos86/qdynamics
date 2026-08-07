"""Positive operator-moment extension for the symmetry-reduced APCM state.

The earlier ``PositiveFourthMomentCompletion`` constrains a degree-two word
Gram matrix but does not include propagated auxiliary operators as Gram rows.
It therefore cannot supply their covariance metric and is not the extended
cone specified by the APCM construction.  This module builds the extension
for one explicit active dictionary: every propagated auxiliary operator is a
Gram row, while products not already present in the retained or active state
form that dictionary's terminal frontier.

The dimer's center phonon is exactly decoupled in the fixed-particle sector,
so the online operator algebra below is the exact spin-symmetric relative-mode
reduction of the two-mode construction.  The retained two-local-mode cone is
still imposed separately by the 29-coordinate affine moment matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Literal, Mapping

import clarabel
import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .adaptive_positive_moment import (
    ARCHIVE_RELATIVE_MOMENT_KEYS,
    HIDDEN_RELATIVE_MOMENT_KEYS,
)
from .moment_hierarchy import (
    IDENTITY,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    MomentKey,
    _OperatorKey,
    _commutator,
    _operator_product,
    build_moment_keys,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

_PAULI_LABELS = (PAULI_X, PAULI_Y, PAULI_Z)


@dataclass(frozen=True)
class APCMExtensionSettings:
    """Frozen positive-completion settings from the APCM memorandum."""

    phonon_envelope: float = 16.0
    logdet_weight: float = 1.0
    logdet_shift: float = 1e-10
    conic_tolerance: float = 1e-9
    maximum_iterations: int = 2_000
    backend: Literal["clarabel_newton", "cvxpy_dense"] = (
        "clarabel_newton"
    )
    maximum_dense_canonicalization_bytes: int = 512 * 1024**2
    maximum_newton_iterations: int = 40
    clarabel_max_threads: int = 1

    def __post_init__(self) -> None:
        if self.phonon_envelope <= 0.0:
            raise ValueError("phonon_envelope must be positive")
        if self.logdet_weight < 0.0:
            raise ValueError("logdet_weight must be nonnegative")
        if self.logdet_shift <= 0.0:
            raise ValueError("logdet_shift must be positive")
        if self.conic_tolerance <= 0.0:
            raise ValueError("conic_tolerance must be positive")
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive")
        if self.backend not in ("clarabel_newton", "cvxpy_dense"):
            raise ValueError("unknown positive-extension backend")
        if self.maximum_dense_canonicalization_bytes <= 0:
            raise ValueError(
                "maximum_dense_canonicalization_bytes must be positive"
            )
        if self.maximum_newton_iterations <= 0:
            raise ValueError("maximum_newton_iterations must be positive")
        if self.clarabel_max_threads <= 0:
            raise ValueError("clarabel_max_threads must be positive")


@dataclass(frozen=True)
class APCMExtensionResult:
    """Unique proximal log-det completion and its numerical certificate."""

    lower_moments: Mapping[MomentKey, float]
    frontier_moments: Mapping[MomentKey, float]
    moment_matrix: ComplexArray
    scaled_minimum_eigenvalue: float
    minimum_eigenvalue: float
    objective: float
    kkt_backward_error: float
    iterations: int
    success: bool
    message: str
    relative_face_support_dimension: int = 0
    relative_face_direction_dimension: int = 0
    facial_qualification_error: float = float("inf")
    facial_reduction_certified: bool = False

    @property
    def moments(self) -> Mapping[MomentKey, float]:
        return MappingProxyType(
            {**dict(self.lower_moments), **dict(self.frontier_moments)}
        )

    def moment(self, key: MomentKey) -> float:
        if key.degree == 0:
            return 1.0
        if key in self.lower_moments:
            return float(self.lower_moments[key])
        try:
            return float(self.frontier_moments[key])
        except KeyError as error:
            raise ValueError(f"completion does not contain {key}") from error


@dataclass(frozen=True)
class APCMLinearExtrema:
    """Certified-outward bounds for one affine frontier functional."""

    primal_minimum: float
    primal_maximum: float
    outward_lower_bound: float
    outward_upper_bound: float
    minimum_gap: float
    maximum_gap: float
    iterations: int
    success: bool
    message: str


@dataclass(frozen=True)
class _ConicQPSolution:
    values: FloatArray
    dual: FloatArray
    slack: FloatArray
    iterations: int
    success: bool
    message: str
    primal_residual: float
    dual_residual: float
    primal_objective: float
    dual_objective: float


@dataclass(frozen=True)
class _FacialReduction:
    values: FloatArray
    support: ComplexArray
    directions: FloatArray
    iterations: int
    success: bool
    message: str
    qualification_error: float


def _realify_hermitian(matrix: ComplexArray) -> FloatArray:
    """Embed one complex Hermitian matrix in a real symmetric PSD cone."""

    value = np.asarray(matrix, dtype=complex)
    realified = np.block(
        [[value.real, -value.imag], [value.imag, value.real]]
    )
    return np.asarray(0.5 * (realified + realified.T), dtype=float)


def _clarabel_svec_upper(matrix: FloatArray) -> FloatArray:
    """Pack a symmetric matrix in Clarabel's scaled upper-triangle order."""

    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("svec requires one square matrix")
    packed: list[float] = []
    root_two = np.sqrt(2.0)
    for column in range(value.shape[1]):
        for row in range(column + 1):
            scale = 1.0 if row == column else root_two
            packed.append(scale * float(value[row, column]))
    return np.asarray(packed, dtype=float)


def _word_scale(word: _OperatorKey, envelope: float) -> float:
    degree = word.x_power + word.p_power
    return float((envelope + degree) ** (0.5 * degree))


def _moment_scale(key: MomentKey, envelope: float) -> float:
    degree = key.x_power + key.p_power
    return float((envelope + degree) ** (0.5 * degree))


@lru_cache(maxsize=1)
def _base_half_words() -> tuple[_OperatorKey, ...]:
    """Return the minimal factors preceding the active operators.

    The retained two-local-mode Gram is imposed independently.  This list
    therefore needs only the identity and the degree-one relative-mode/two-spin
    factors required to expose the active moments and their commutator
    frontier.  Appending every active operator below then supplies its full
    covariance as a principal block without the prototype's unused complete
    degree-two shell.
    """

    words: list[_OperatorKey] = [
        _OperatorKey(IDENTITY, IDENTITY, 0, 0)
    ]
    for spin in (0, 1):
        for label in _PAULI_LABELS:
            labels = [IDENTITY, IDENTITY]
            labels[spin] = label
            words.append(_OperatorKey(labels[0], labels[1], 0, 0))
    words.extend(
        (
            _OperatorKey(IDENTITY, IDENTITY, 1, 0),
            _OperatorKey(IDENTITY, IDENTITY, 0, 1),
        )
    )
    return tuple(words)


@lru_cache(maxsize=None)
def apcm_extension_words(
    active_keys: tuple[MomentKey, ...] = HIDDEN_RELATIVE_MOMENT_KEYS,
    additional_halfword_keys: tuple[MomentKey, ...] = (),
) -> tuple[_OperatorKey, ...]:
    """Return base, active, and explicitly requested diagnostic half-words."""

    words = list(_base_half_words())
    present = set(words)
    for key in (*active_keys, *additional_halfword_keys):
        word = _OperatorKey(
            key.spin_up,
            key.spin_down,
            key.x_power,
            key.p_power,
        )
        if word not in present:
            words.append(word)
            present.add(word)
    return tuple(words)


@lru_cache(maxsize=None)
def apcm_extension_coefficients(
    active_keys: tuple[MomentKey, ...] = HIDDEN_RELATIVE_MOMENT_KEYS,
    additional_halfword_keys: tuple[MomentKey, ...] = (),
) -> Mapping[MomentKey, ComplexArray]:
    """Compile the affine coefficients of the active operator Gram matrix."""

    words = apcm_extension_words(active_keys, additional_halfword_keys)
    maximum_degree = 2 * max(
        int(word.spin_up != IDENTITY)
        + int(word.spin_down != IDENTITY)
        + word.x_power
        + word.p_power
        for word in words
    )
    coefficients: dict[MomentKey, ComplexArray] = {}
    for row, left in enumerate(words):
        for column, right in enumerate(words):
            for key, coefficient in _operator_product(left, right).items():
                if key.degree > maximum_degree:
                    raise RuntimeError(
                        "active operator Gram generated a moment above twice "
                        "its maximum word degree"
                    )
                matrix = coefficients.setdefault(
                    key,
                    np.zeros((len(words), len(words)), dtype=complex),
                )
                matrix[row, column] += coefficient
    for key, matrix in coefficients.items():
        hermitian = 0.5 * (matrix + matrix.conjugate().T)
        if np.linalg.norm(matrix - hermitian) > 5e-12:
            raise RuntimeError(
                f"extension coefficient for {key} is not Hermitian"
            )
        coefficients[key] = hermitian
    return MappingProxyType(coefficients)


class SymmetryReducedPositiveExtension:
    """Proximal positive completion for one explicit active dictionary."""

    def __init__(
        self,
        settings: APCMExtensionSettings | None = None,
        *,
        active_keys: tuple[MomentKey, ...] | None = None,
        additional_halfword_keys: tuple[MomentKey, ...] = (),
    ) -> None:
        self.settings = (
            APCMExtensionSettings() if settings is None else settings
        )
        self.active_keys = (
            tuple(HIDDEN_RELATIVE_MOMENT_KEYS)
            if active_keys is None
            else tuple(active_keys)
        )
        if not self.active_keys:
            raise ValueError("the positive extension requires an active dictionary")
        if len(set(self.active_keys)) != len(self.active_keys):
            raise ValueError("active moment keys must be unique")
        self.additional_halfword_keys = tuple(additional_halfword_keys)
        if len(set(self.additional_halfword_keys)) != len(
            self.additional_halfword_keys
        ):
            raise ValueError("additional half-word moment keys must be unique")
        archive_set = set(ARCHIVE_RELATIVE_MOMENT_KEYS)
        overlap = archive_set.intersection(self.active_keys)
        if overlap:
            raise ValueError(f"active keys duplicate archive-resolved moments: {overlap}")
        maximum_active_degree = max(key.degree for key in self.active_keys)
        lower_set = archive_set.union(self.active_keys)
        self.lower_keys = tuple(
            key
            for key in build_moment_keys(maximum_active_degree)
            if key in lower_set
        )
        if set(self.lower_keys) != lower_set:
            raise ValueError("active dictionary contains unsupported canonical moments")
        self.words = apcm_extension_words(
            self.active_keys,
            self.additional_halfword_keys,
        )
        coefficients = apcm_extension_coefficients(
            self.active_keys,
            self.additional_halfword_keys,
        )
        lower_set = set(self.lower_keys)
        self.frontier_keys = tuple(
            sorted(
                (
                    key
                    for key in coefficients
                    if key.degree > 0 and key not in lower_set
                ),
                key=lambda key: (
                    key.degree,
                    key.spin_up,
                    key.spin_down,
                    key.x_power,
                    key.p_power,
                ),
            )
        )
        h0_words = (
            _OperatorKey(PAULI_X, IDENTITY, 0, 0),
            _OperatorKey(IDENTITY, PAULI_X, 0, 0),
            _OperatorKey(IDENTITY, IDENTITY, 2, 0),
            _OperatorKey(IDENTITY, IDENTITY, 0, 2),
            _OperatorKey(PAULI_Z, IDENTITY, 1, 0),
            _OperatorKey(IDENTITY, PAULI_Z, 1, 0),
        )
        hv_words = (
            _OperatorKey(PAULI_Z, IDENTITY, 0, 0),
            _OperatorKey(IDENTITY, PAULI_Z, 0, 0),
        )
        frontier_set = set(self.frontier_keys)

        def generated_frontier(words: tuple[_OperatorKey, ...]) -> set[MomentKey]:
            return {
                generated
                for observable in self.active_keys
                for hamiltonian_word in words
                for generated in _commutator(hamiltonian_word, observable)
                if generated in frontier_set
            }

        h0_support = generated_frontier(h0_words)
        hv_support = generated_frontier(hv_words)
        rhs_support = h0_support.union(hv_support)
        self.rhs_frontier_h0_keys = tuple(
            key for key in self.frontier_keys if key in h0_support
        )
        self.rhs_frontier_hv_keys = tuple(
            key for key in self.frontier_keys if key in hv_support
        )
        self.rhs_frontier_keys = tuple(
            key for key in self.frontier_keys if key in rhs_support
        )
        self.auxiliary_frontier_keys = tuple(
            key for key in self.frontier_keys if key not in rhs_support
        )
        self._identity_key = MomentKey(IDENTITY, IDENTITY, 0, 0)
        self._constant = coefficients[self._identity_key]
        self.lower_coefficients = np.asarray(
            [coefficients[key] for key in self.lower_keys],
            dtype=complex,
        )
        self.frontier_coefficients = np.asarray(
            [coefficients[key] for key in self.frontier_keys],
            dtype=complex,
        )
        self.frontier_scales = np.asarray(
            [
                _moment_scale(key, self.settings.phonon_envelope)
                for key in self.frontier_keys
            ],
            dtype=float,
        )
        self.active_scales = np.asarray(
            [
                _moment_scale(key, self.settings.phonon_envelope)
                for key in self.active_keys
            ],
            dtype=float,
        )
        words = self.words
        self.word_scales = np.asarray(
            [
                _word_scale(word, self.settings.phonon_envelope)
                for word in words
            ],
            dtype=float,
        )
        lookup = {word: index for index, word in enumerate(words)}
        self.active_word_indices = np.asarray(
            [
                lookup[
                    _OperatorKey(
                        key.spin_up,
                        key.spin_down,
                        key.x_power,
                        key.p_power,
                    )
                ]
                for key in self.active_keys
            ],
            dtype=int,
        )

        inverse_word_scales = np.diag(1.0 / self.word_scales)
        self.scaled_frontier_coefficients = np.einsum(
            "ab,kbc,cd->kad",
            inverse_word_scales,
            self.frontier_coefficients,
            inverse_word_scales,
            optimize=True,
        )
        self._lower_parameter: cp.Parameter | None = None
        self._frontier_variable: cp.Variable | None = None
        self._scaled_expression: cp.Expression | None = None
        self._psd_constraint: cp.Constraint | None = None
        self._lower_bound_constraint: cp.Constraint | None = None
        self._upper_bound_constraint: cp.Constraint | None = None
        self._problem: cp.Problem | None = None
        self._cached_lower_values: FloatArray | None = None
        self._cached_result: APCMExtensionResult | None = None
        if self.settings.backend == "cvxpy_dense":
            self._build_dense_cvxpy_problem()

    @property
    def dimension(self) -> int:
        return len(self.words)

    @property
    def dense_canonicalization_estimate_bytes(self) -> int:
        """Conservative storage estimate for CVXPY's dense log-det lift.

        A complex Hermitian log-det canonicalization introduces realified
        matrix-cone operators and several simultaneous sparse/dense work
        copies.  The leading storage scales as ``n^4 m`` for Gram dimension
        ``n`` and ``m`` affine frontier variables.  The factor 64 deliberately
        includes realification and canonicalization work copies; it matched
        the order of the observed tens-of-GB failure and is a rejection bound,
        not a runtime prediction.
        """

        return int(
            64
            * self.dimension**4
            * max(1, len(self.frontier_keys))
        )

    def _build_dense_cvxpy_problem(self) -> None:
        self._lower_parameter = cp.Parameter(len(self.lower_keys))
        self._frontier_variable = cp.Variable(len(self.frontier_keys))
        expression: cp.Expression = cp.Constant(self._constant)
        for index, coefficient in enumerate(self.lower_coefficients):
            expression += self._lower_parameter[index] * cp.Constant(
                coefficient
            )
        for index, coefficient in enumerate(self.frontier_coefficients):
            expression += self._frontier_variable[index] * cp.Constant(
                coefficient
            )
        inverse_word_scales = np.diag(1.0 / self.word_scales)
        scaled_expression = (
            cp.Constant(inverse_word_scales)
            @ expression
            @ cp.Constant(inverse_word_scales)
        )
        self._scaled_expression = cp.hermitian_wrap(scaled_expression)
        standardized = cp.multiply(
            1.0 / self.frontier_scales,
            self._frontier_variable,
        )
        objective = 0.5 * cp.sum_squares(standardized)
        if self.settings.logdet_weight > 0.0:
            shifted = self._scaled_expression + (
                self.settings.logdet_shift
                * np.eye(self.dimension, dtype=complex)
            )
            objective -= (
                self.settings.logdet_weight
                / self.dimension
                * cp.log_det(shifted)
            )
        self._psd_constraint = self._scaled_expression >> 0.0
        self._lower_bound_constraint = (
            self._frontier_variable >= -self.frontier_scales
        )
        self._upper_bound_constraint = (
            self._frontier_variable <= self.frontier_scales
        )
        self._problem = cp.Problem(
            cp.Minimize(objective),
            [
                self._psd_constraint,
                self._lower_bound_constraint,
                self._upper_bound_constraint,
            ],
        )

    def matrix(
        self,
        lower_moments: Mapping[MomentKey, float],
        frontier_moments: Mapping[MomentKey, float],
    ) -> ComplexArray:
        lower = np.asarray(
            [float(lower_moments[key]) for key in self.lower_keys]
        )
        frontier = np.asarray(
            [float(frontier_moments[key]) for key in self.frontier_keys]
        )
        matrix = (
            self._constant
            + np.tensordot(lower, self.lower_coefficients, axes=(0, 0))
            + np.tensordot(
                frontier,
                self.frontier_coefficients,
                axes=(0, 0),
            )
        )
        return 0.5 * (matrix + matrix.conjugate().T)

    def scaled_matrix(self, matrix: ComplexArray) -> ComplexArray:
        inverse = np.diag(1.0 / self.word_scales)
        result = inverse @ np.asarray(matrix, dtype=complex) @ inverse
        return 0.5 * (result + result.conjugate().T)

    def linear_functional_extrema(
        self,
        lower_moments: Mapping[MomentKey, float],
        coefficients: Mapping[MomentKey, float],
        *,
        constant: float = 0.0,
    ) -> APCMLinearExtrema:
        """Range one affine moment functional over the positive prior fiber.

        The frontier variables are standardized and boxed, so a normalized
        stationarity residual gives a finite outward correction to Clarabel's
        dual objectives.  The returned interval is deliberately outward-facing;
        it is the quantity used by admission rather than the primal extrema
        alone.
        """

        missing = set(self.lower_keys).difference(lower_moments)
        if missing:
            raise ValueError(
                f"positive extension is missing {len(missing)} lower moments"
            )
        identity = MomentKey(IDENTITY, IDENTITY, 0, 0)
        supported = {identity, *self.lower_keys, *self.frontier_keys}
        unsupported = {
            key
            for key, value in coefficients.items()
            if abs(float(value)) > 1e-15 and key not in supported
        }
        if unsupported:
            raise ValueError(
                "linear functional contains moments outside this extension: "
                f"{unsupported}"
            )

        fixed = float(constant) + float(coefficients.get(identity, 0.0))
        fixed += sum(
            float(coefficients.get(key, 0.0)) * float(lower_moments[key])
            for key in self.lower_keys
        )
        frontier_linear = np.asarray(
            [float(coefficients.get(key, 0.0)) for key in self.frontier_keys],
            dtype=float,
        )
        standardized_linear = frontier_linear * self.frontier_scales
        if not self.frontier_keys:
            return APCMLinearExtrema(
                primal_minimum=fixed,
                primal_maximum=fixed,
                outward_lower_bound=fixed,
                outward_upper_bound=fixed,
                minimum_gap=0.0,
                maximum_gap=0.0,
                iterations=0,
                success=True,
                message="algebraic interval collapse",
            )

        lower_values = np.asarray(
            [float(lower_moments[key]) for key in self.lower_keys],
            dtype=float,
        )
        unscaled_base = self._constant + np.tensordot(
            lower_values,
            self.lower_coefficients,
            axes=(0, 0),
        )
        scaled_base = self.scaled_matrix(unscaled_base)
        standardized_coefficients = (
            self.scaled_frontier_coefficients
            * self.frontier_scales[:, np.newaxis, np.newaxis]
        )
        cone_matrix, cone_rhs, cones = self._completion_cone_data(
            scaled_base,
            standardized_coefficients,
        )
        zero_quadratic = np.zeros(
            (len(self.frontier_keys), len(self.frontier_keys)),
            dtype=float,
        )
        minimum = self._solve_conic_qp(
            zero_quadratic,
            standardized_linear,
            cone_matrix,
            cone_rhs,
            cones,
        )
        maximum = self._solve_conic_qp(
            zero_quadratic,
            -standardized_linear,
            cone_matrix,
            cone_rhs,
            cones,
        )
        if not minimum.success or not maximum.success:
            return APCMLinearExtrema(
                primal_minimum=float("nan"),
                primal_maximum=float("nan"),
                outward_lower_bound=float("-inf"),
                outward_upper_bound=float("inf"),
                minimum_gap=float("inf"),
                maximum_gap=float("inf"),
                iterations=minimum.iterations + maximum.iterations,
                success=False,
                message=(
                    f"minimum={minimum.message}; maximum={maximum.message}"
                ),
            )

        primal_minimum = fixed + float(
            standardized_linear @ minimum.values
        )
        primal_maximum = fixed + float(
            standardized_linear @ maximum.values
        )

        def dual_residual_correction(
            linear: FloatArray,
            solution: _ConicQPSolution,
        ) -> float:
            stationarity = np.asarray(
                linear + cone_matrix.T @ solution.dual,
                dtype=float,
            ).reshape(-1)
            residual = max(solution.primal_residual, solution.dual_residual)
            scale = (
                1.0
                + float(np.linalg.norm(cone_rhs, ord=1))
                + float(np.linalg.norm(solution.dual, ord=1))
            )
            return float(
                np.linalg.norm(stationarity, ord=1)
                + 10.0 * residual * scale
                + 100.0
                * np.finfo(float).eps
                * max(1.0, abs(solution.dual_objective))
            )

        minimum_correction = dual_residual_correction(
            standardized_linear,
            minimum,
        )
        maximum_correction = dual_residual_correction(
            -standardized_linear,
            maximum,
        )
        outward_lower = (
            fixed + minimum.dual_objective - minimum_correction
        )
        outward_upper = (
            fixed - maximum.dual_objective + maximum_correction
        )
        return APCMLinearExtrema(
            primal_minimum=primal_minimum,
            primal_maximum=primal_maximum,
            outward_lower_bound=float(outward_lower),
            outward_upper_bound=float(outward_upper),
            minimum_gap=float(max(0.0, primal_minimum - outward_lower)),
            maximum_gap=float(max(0.0, outward_upper - primal_maximum)),
            iterations=minimum.iterations + maximum.iterations,
            success=bool(
                outward_lower <= primal_minimum
                and primal_minimum <= primal_maximum
                and primal_maximum <= outward_upper
            ),
            message=(
                f"minimum={minimum.message}; maximum={maximum.message}"
            ),
        )

    def complete(
        self,
        lower_moments: Mapping[MomentKey, float],
        *,
        warm_frontier: Mapping[MomentKey, float] | None = None,
    ) -> APCMExtensionResult:
        """Select the unique scaled minimum-norm/log-det frontier."""

        missing = set(self.lower_keys).difference(lower_moments)
        if missing:
            raise ValueError(
                f"positive extension is missing {len(missing)} lower moments"
            )
        lower = MappingProxyType(
            {key: float(lower_moments[key]) for key in self.lower_keys}
        )
        lower_values = np.asarray(
            [lower[key] for key in self.lower_keys],
            dtype=float,
        )
        if (
            self._cached_lower_values is not None
            and self._cached_result is not None
            and np.array_equal(lower_values, self._cached_lower_values)
        ):
            return self._cached_result
        if warm_frontier is not None:
            missing_frontier = set(self.frontier_keys).difference(
                warm_frontier
            )
            if missing_frontier:
                raise ValueError(
                    "warm frontier is missing "
                    f"{len(missing_frontier)} moments"
                )
        if self.settings.backend == "clarabel_newton":
            result = self._complete_clarabel_newton(lower, warm_frontier)
            if result.success:
                self._cached_lower_values = lower_values.copy()
                self._cached_result = result
            return result

        estimate = self.dense_canonicalization_estimate_bytes
        budget = self.settings.maximum_dense_canonicalization_bytes
        if estimate > budget:
            raise RuntimeError(
                "resource preflight rejected dense CVXPY canonicalization: "
                f"estimated {estimate / 1024**3:.2f} GiB exceeds "
                f"the {budget / 1024**3:.2f} GiB budget"
            )
        if (
            self._lower_parameter is None
            or self._frontier_variable is None
            or self._problem is None
        ):
            raise RuntimeError("dense CVXPY problem was not constructed")
        self._lower_parameter.value = np.asarray(
            [lower[key] for key in self.lower_keys],
            dtype=float,
        )
        if warm_frontier is None:
            self._frontier_variable.value = np.zeros(len(self.frontier_keys))
        else:
            self._frontier_variable.value = np.asarray(
                [float(warm_frontier[key]) for key in self.frontier_keys],
                dtype=float,
            )
        message = ""
        iterations = 0
        try:
            self._problem.solve(
                solver=cp.CLARABEL,
                warm_start=True,
                verbose=False,
                max_iter=self.settings.maximum_iterations,
                tol_gap_abs=self.settings.conic_tolerance,
                tol_gap_rel=self.settings.conic_tolerance,
                tol_feas=self.settings.conic_tolerance,
            )
            message = str(self._problem.status)
            iterations = int(self._problem.solver_stats.num_iters or 0)
        except cp.error.SolverError as error:
            message = f"CLARABEL failure: {error}"
        if self._frontier_variable.value is None:
            return APCMExtensionResult(
                lower_moments=lower,
                frontier_moments=MappingProxyType({}),
                moment_matrix=np.empty((0, 0), dtype=complex),
                scaled_minimum_eigenvalue=float("-inf"),
                minimum_eigenvalue=float("-inf"),
                objective=float("inf"),
                kkt_backward_error=float("inf"),
                iterations=iterations,
                success=False,
                message=message,
            )

        values = np.asarray(
            self._frontier_variable.value,
            dtype=float,
        ).reshape(-1)
        frontier = MappingProxyType(
            {
                key: float(value)
                for key, value in zip(
                    self.frontier_keys,
                    values,
                    strict=True,
                )
            }
        )
        matrix = self.matrix(lower, frontier)
        scaled = self.scaled_matrix(matrix)
        minimum = float(np.linalg.eigvalsh(matrix)[0])
        scaled_minimum = float(np.linalg.eigvalsh(scaled)[0])
        standardized = values / self.frontier_scales
        shifted = scaled + self.settings.logdet_shift * np.eye(
            scaled.shape[0], dtype=complex
        )
        sign, logdet = np.linalg.slogdet(shifted)
        objective = (
            0.5 * float(standardized @ standardized)
            - self.settings.logdet_weight / self.dimension * float(logdet)
            if sign > 0.0
            else float("inf")
        )
        kkt_error = self._kkt_backward_error(values, scaled)
        success = bool(
            self._problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            and scaled_minimum >= -10.0 * self.settings.conic_tolerance
            and kkt_error <= 50.0 * self.settings.conic_tolerance
        )
        return APCMExtensionResult(
            lower_moments=lower,
            frontier_moments=frontier,
            moment_matrix=matrix,
            scaled_minimum_eigenvalue=scaled_minimum,
            minimum_eigenvalue=minimum,
            objective=objective,
            kkt_backward_error=kkt_error,
            iterations=iterations,
            success=success,
            message=message,
        )

    def _complete_clarabel_newton(
        self,
        lower: Mapping[MomentKey, float],
        warm_frontier: Mapping[MomentKey, float] | None,
    ) -> APCMExtensionResult:
        """Solve the proximal log-det selector with direct conic Newton steps.

        Each Newton model is a frontier-sized quadratic SDP passed directly
        to Clarabel.  This avoids CVXPY's dense log-det canonicalization and
        explicitly restricts Clarabel to one thread.
        """

        lower_values = np.asarray(
            [lower[key] for key in self.lower_keys],
            dtype=float,
        )
        unscaled_base = self._constant + np.tensordot(
            lower_values,
            self.lower_coefficients,
            axes=(0, 0),
        )
        scaled_base = self.scaled_matrix(unscaled_base)
        scales = self.frontier_scales
        coefficients = (
            self.scaled_frontier_coefficients
            * scales[:, np.newaxis, np.newaxis]
        )

        def scaled_matrix(standardized: FloatArray) -> ComplexArray:
            result = scaled_base + np.tensordot(
                np.asarray(standardized, dtype=float),
                coefficients,
                axes=(0, 0),
            )
            return 0.5 * (result + result.conjugate().T)

        count = len(self.frontier_keys)
        support = np.eye(self.dimension, dtype=complex)
        directions = np.eye(count, dtype=float)
        qualification_error = 0.0
        facial_reduction_certified = False

        def objective(values: FloatArray) -> float:
            matrix = scaled_matrix(values)
            reduced_matrix = support.conjugate().T @ matrix @ support
            eigenvalues = np.linalg.eigvalsh(reduced_matrix)
            shifted = eigenvalues + self.settings.logdet_shift
            safe = np.maximum(
                shifted,
                0.1 * self.settings.logdet_shift,
            )
            barrier = -(
                self.settings.logdet_weight
                / self.dimension
                * float(np.sum(np.log(safe)))
            )
            violation = np.minimum(shifted, 0.0)
            penalty = (
                1e6 * float(violation @ violation)
                if np.any(violation < 0.0)
                else 0.0
            )
            return (
                0.5 * float(np.asarray(values) @ np.asarray(values))
                + barrier
                + penalty
            )

        def derivatives(
            values: FloatArray,
        ) -> tuple[FloatArray, FloatArray]:
            matrix = scaled_matrix(values)
            reduced_matrix = support.conjugate().T @ matrix @ support
            reduced_coefficients = np.einsum(
                "ai,kab,bj->kij",
                support.conjugate(),
                coefficients,
                support,
                optimize=True,
            )
            eigenvalues, eigenvectors = np.linalg.eigh(reduced_matrix)
            shifted = eigenvalues + self.settings.logdet_shift
            safe = np.maximum(
                shifted,
                0.1 * self.settings.logdet_shift,
            )
            inverse = (
                eigenvectors * (1.0 / safe)
            ) @ eigenvectors.conjugate().T
            transformed = np.einsum(
                "ab,kbc->kac",
                inverse,
                reduced_coefficients,
                optimize=True,
            )
            barrier_gradient = np.asarray(
                [
                    np.trace(value).real for value in transformed
                ],
                dtype=float,
            )
            gradient = np.asarray(values, dtype=float).copy()
            gradient -= (
                self.settings.logdet_weight
                / self.dimension
                * barrier_gradient
            )
            hessian = np.eye(count, dtype=float)
            hessian += (
                self.settings.logdet_weight
                / self.dimension
                * np.einsum(
                    "kab,lba->kl",
                    transformed,
                    transformed,
                    optimize=True,
                ).real
            )
            for index in np.flatnonzero(shifted < 0.0):
                mode = eigenvectors[:, index]
                mode_gradient = np.asarray(
                    [
                        np.vdot(mode, coefficient @ mode).real
                        for coefficient in reduced_coefficients
                    ],
                    dtype=float,
                )
                gradient += 2e6 * shifted[index] * mode_gradient
                hessian += 2e6 * np.outer(mode_gradient, mode_gradient)
            hessian = 0.5 * (hessian + hessian.T)
            return gradient, hessian

        tolerance = self.settings.conic_tolerance
        warm_is_feasible = False
        if warm_frontier is not None:
            warm = np.asarray(
                [float(warm_frontier[key]) for key in self.frontier_keys],
                dtype=float,
            ) / scales
            warm_matrix = scaled_matrix(warm)
            warm_is_feasible = bool(
                np.max(np.abs(warm)) <= 1.0
                + 10.0 * tolerance
                and np.linalg.eigvalsh(warm_matrix)[0]
                >= -10.0 * tolerance
            )

        cone_matrix: sparse.csc_matrix | None = None
        cone_rhs: FloatArray | None = None
        cones: tuple[object, ...] | None = None
        total_iterations = 0
        warm_is_strict = bool(
            warm_is_feasible
            and np.linalg.eigvalsh(warm_matrix)[0] > 100.0 * tolerance
            and 1.0 - np.max(np.abs(warm)) > 100.0 * tolerance
        )
        facial_message = "full cone interior"
        if warm_is_strict:
            standardized = np.clip(warm, -1.0, 1.0)
            facial_reduction_certified = True
        else:
            cone_matrix, cone_rhs, cones = self._completion_cone_data(
                scaled_base,
                coefficients,
            )
            facial = self._facially_reduced_reference(
                scaled_base,
                coefficients,
                cone_matrix,
                cone_rhs,
                cones,
            )
            total_iterations = facial.iterations
            if not facial.success:
                return self._result_from_values(
                    lower,
                    facial.values * scales,
                    iterations=facial.iterations,
                    solver_success=False,
                    message=(
                        "moment_extension_degeneracy: "
                        f"{facial.message}"
                    ),
                    kkt_error=float("inf"),
                )
            standardized = facial.values.copy()
            support = facial.support
            directions = facial.directions
            qualification_error = facial.qualification_error
            facial_message = facial.message
            facial_reduction_certified = True

        outer_iterations = 0
        final_qp: _ConicQPSolution | None = None
        used_conic_newton = False
        for outer_iterations in range(
            1,
            self.settings.maximum_newton_iterations + 1,
        ):
            gradient, hessian = derivatives(standardized)
            matrix = scaled_matrix(standardized)
            relative_eigenvalue = float(
                np.linalg.eigvalsh(
                    support.conjugate().T @ matrix @ support
                )[0]
            )
            box_margin = float(1.0 - np.max(np.abs(standardized)))
            relative_interior = bool(
                relative_eigenvalue > max(
                    0.1 * tolerance,
                    10.0 * self.settings.logdet_shift,
                )
                and box_margin > 100.0 * tolerance
            )
            reduced_gradient = directions.T @ gradient
            gradient_error = float(
                np.linalg.norm(reduced_gradient, ord=np.inf)
            )
            if relative_interior and gradient_error <= 50.0 * tolerance:
                break

            accepted_unconstrained = False
            if relative_interior:
                try:
                    reduced_hessian = directions.T @ hessian @ directions
                    reduced_direction = -np.linalg.solve(
                        reduced_hessian,
                        reduced_gradient,
                    )
                    direction = directions @ reduced_direction
                except np.linalg.LinAlgError:
                    direction = np.zeros_like(gradient)
                directional_derivative = float(gradient @ direction)
                step = 1.0
                for value, delta in zip(
                    standardized,
                    direction,
                    strict=True,
                ):
                    if delta > 0.0:
                        step = min(step, 0.995 * (1.0 - value) / delta)
                    elif delta < 0.0:
                        step = min(step, 0.995 * (-1.0 - value) / delta)
                current_objective = objective(standardized)
                while step >= 2.0**-30:
                    candidate = standardized + step * direction
                    candidate_minimum = float(
                        np.linalg.eigvalsh(scaled_matrix(candidate))[0]
                    )
                    if (
                        candidate_minimum >= -0.1 * tolerance
                        and objective(candidate)
                        <= current_objective
                        + 1e-4 * step * directional_derivative
                    ):
                        standardized = candidate
                        accepted_unconstrained = True
                        break
                    step *= 0.5
            if accepted_unconstrained:
                continue

            if cone_matrix is None or cone_rhs is None or cones is None:
                cone_matrix, cone_rhs, cones = self._completion_cone_data(
                    scaled_base,
                    coefficients,
                )
            used_conic_newton = True
            final_qp = self._solve_conic_qp(
                hessian,
                gradient - hessian @ standardized,
                cone_matrix,
                cone_rhs,
                cones,
            )
            total_iterations += final_qp.iterations
            if not final_qp.success:
                return self._result_from_values(
                    lower,
                    standardized * scales,
                    iterations=total_iterations,
                    solver_success=False,
                    message=(
                        f"Newton cone failure after {facial_message}: "
                        f"{final_qp.message}; relative_lambda="
                        f"{relative_eigenvalue:.3e}"
                    ),
                    kkt_error=float("inf"),
                )
            direction = final_qp.values - standardized
            directional_derivative = float(gradient @ direction)
            step = 1.0
            current_objective = objective(standardized)
            accepted_conic = False
            while step >= 2.0**-30:
                candidate = standardized + step * direction
                candidate_minimum = float(
                    np.linalg.eigvalsh(scaled_matrix(candidate))[0]
                )
                if (
                    candidate_minimum >= -0.1 * tolerance
                    and objective(candidate)
                    <= current_objective
                    + 1e-4 * step * directional_derivative
                ):
                    standardized = candidate
                    accepted_conic = True
                    break
                step *= 0.5
            if not accepted_conic:
                return self._result_from_values(
                    lower,
                    standardized * scales,
                    iterations=total_iterations,
                    solver_success=False,
                    message=(
                        "moment_extension_degeneracy: conic Newton step "
                        f"failed descent after {facial_message}"
                    ),
                    kkt_error=float("inf"),
                )
            if np.linalg.norm(step * direction, ord=np.inf) <= 5.0 * tolerance:
                break

        gradient, hessian = derivatives(standardized)
        matrix = scaled_matrix(standardized)
        relative_eigenvalue = float(
            np.linalg.eigvalsh(
                support.conjugate().T @ matrix @ support
            )[0]
        )
        box_margin = float(1.0 - np.max(np.abs(standardized)))
        relative_interior = bool(
            relative_eigenvalue > max(
                0.1 * tolerance,
                10.0 * self.settings.logdet_shift,
            )
            and box_margin > 100.0 * tolerance
        )
        if relative_interior and not used_conic_newton:
            kkt_error = max(
                qualification_error,
                float(np.linalg.norm(directions.T @ gradient, ord=np.inf)),
            )
            solver_success = kkt_error <= 50.0 * tolerance
            certificate_message = "interior Newton stationarity"
        else:
            if cone_matrix is None or cone_rhs is None or cones is None:
                cone_matrix, cone_rhs, cones = self._completion_cone_data(
                    scaled_base,
                    coefficients,
                )
            certificate = self._solve_conic_qp(
                hessian,
                gradient - hessian @ standardized,
                cone_matrix,
                cone_rhs,
                cones,
            )
            total_iterations += certificate.iterations
            kkt_error = self._conic_kkt_backward_error(
                standardized,
                gradient,
                cone_matrix,
                cone_rhs,
                certificate,
                scaled_matrix(standardized),
            )
            kkt_error = max(kkt_error, qualification_error)
            solver_success = bool(certificate.success)
            certificate_message = certificate.message
        return self._result_from_values(
            lower,
            standardized * scales,
            iterations=total_iterations,
            solver_success=solver_success,
            message=(
                "direct Clarabel proximal log-det; "
                f"Newton iterations={outer_iterations}; "
                f"face={facial_message}; certificate={certificate_message}; "
                f"relative_lambda={relative_eigenvalue:.3e}; "
                f"reduced_gradient="
                f"{np.linalg.norm(directions.T @ gradient, ord=np.inf):.3e}; "
                f"conic_newton={used_conic_newton}"
            ),
            kkt_error=kkt_error,
            relative_face_support_dimension=support.shape[1],
            relative_face_direction_dimension=directions.shape[1],
            facial_qualification_error=qualification_error,
            facial_reduction_certified=facial_reduction_certified,
        )

    def _completion_cone_data(
        self,
        scaled_base: ComplexArray,
        standardized_coefficients: ComplexArray,
    ) -> tuple[sparse.csc_matrix, FloatArray, tuple[object, ...]]:
        real_base = _realify_hermitian(scaled_base)
        real_coefficients = np.asarray(
            [
                _realify_hermitian(coefficient)
                for coefficient in standardized_coefficients
            ],
            dtype=float,
        )
        base_vector = _clarabel_svec_upper(real_base)
        coefficient_vectors = np.column_stack(
            [
                _clarabel_svec_upper(coefficient)
                for coefficient in real_coefficients
            ]
        )
        count = len(self.frontier_keys)
        cone_matrix = sparse.vstack(
            (
                sparse.csc_matrix(-coefficient_vectors),
                sparse.eye(count, format="csc"),
                -sparse.eye(count, format="csc"),
            ),
            format="csc",
        )
        cone_rhs = np.concatenate(
            (base_vector, np.ones(count), np.ones(count))
        )
        cones: tuple[object, ...] = (
            clarabel.PSDTriangleConeT(real_base.shape[0]),
            clarabel.NonnegativeConeT(2 * count),
        )
        return cone_matrix, cone_rhs, cones

    def _facially_reduced_reference(
        self,
        scaled_base: ComplexArray,
        standardized_coefficients: ComplexArray,
        cone_matrix: sparse.csc_matrix,
        cone_rhs: FloatArray,
        cones: tuple[object, ...],
    ) -> _FacialReduction:
        """Find and certify the relative face of the completion fiber."""

        count = len(self.frontier_keys)
        reference = self._solve_conic_qp(
            np.eye(count, dtype=float),
            np.zeros(count, dtype=float),
            cone_matrix,
            cone_rhs,
            cones,
        )
        if not reference.success:
            return _FacialReduction(
                values=reference.values,
                support=np.empty((self.dimension, 0), dtype=complex),
                directions=np.empty((count, 0), dtype=float),
                iterations=reference.iterations,
                success=False,
                message=f"reference completion: {reference.message}",
                qualification_error=float("inf"),
            )

        values = reference.values.copy()
        total_iterations = reference.iterations
        worst_solver_residual = max(
            reference.primal_residual,
            reference.dual_residual,
        )
        qualification_error = float("inf")
        null_basis = np.empty((self.dimension, 0), dtype=complex)
        support = np.eye(self.dimension, dtype=complex)
        for _ in range(self.dimension + 1):
            matrix = scaled_base + np.tensordot(
                values,
                standardized_coefficients,
                axes=(0, 0),
            )
            eigenvalues, eigenvectors = np.linalg.eigh(
                0.5 * (matrix + matrix.conjugate().T)
            )
            candidate_tolerance = max(
                self.settings.conic_tolerance,
                100.0 * worst_solver_residual,
                100.0 * np.finfo(float).eps * max(1.0, eigenvalues[-1]),
            )
            null_count = int(
                np.count_nonzero(eigenvalues <= candidate_tolerance)
            )
            if null_count == 0:
                null_basis = np.empty(
                    (self.dimension, 0),
                    dtype=complex,
                )
                support = eigenvectors
                qualification_error = worst_solver_residual
                break

            candidate_null = eigenvectors[:, :null_count]
            exposed_trace = np.asarray(
                [
                    np.trace(
                        candidate_null.conjugate().T
                        @ coefficient
                        @ candidate_null
                    ).real
                    for coefficient in standardized_coefficients
                ],
                dtype=float,
            )
            exposure = self._solve_conic_qp(
                1e-8 * np.eye(count, dtype=float),
                -exposed_trace,
                cone_matrix,
                cone_rhs,
                cones,
            )
            total_iterations += exposure.iterations
            if not exposure.success:
                return _FacialReduction(
                    values=values,
                    support=np.empty((self.dimension, 0), dtype=complex),
                    directions=np.empty((count, 0), dtype=float),
                    iterations=total_iterations,
                    success=False,
                    message=f"facial exposure: {exposure.message}",
                    qualification_error=float("inf"),
                )
            worst_solver_residual = max(
                worst_solver_residual,
                exposure.primal_residual,
                exposure.dual_residual,
            )
            exposed_matrix = scaled_base + np.tensordot(
                exposure.values,
                standardized_coefficients,
                axes=(0, 0),
            )
            exposed_value = float(
                np.trace(
                    candidate_null.conjugate().T
                    @ exposed_matrix
                    @ candidate_null
                ).real
            )
            qualification_error = max(
                worst_solver_residual,
                max(0.0, exposed_value) / max(1, null_count),
            )
            qualification_tolerance = max(
                10.0 * self.settings.conic_tolerance,
                1000.0 * worst_solver_residual,
            )
            if exposed_value / max(1, null_count) <= qualification_tolerance:
                null_basis = candidate_null
                support = eigenvectors[:, null_count:]
                break
            values = 0.5 * (values + exposure.values)
        else:
            return _FacialReduction(
                values=values,
                support=np.empty((self.dimension, 0), dtype=complex),
                directions=np.empty((count, 0), dtype=float),
                iterations=total_iterations,
                success=False,
                message="facial reduction did not stabilize",
                qualification_error=float("inf"),
            )

        if null_basis.shape[1] == 0:
            directions = np.eye(count, dtype=float)
        else:
            equality_map = np.column_stack(
                [
                    (coefficient @ null_basis).reshape(-1)
                    for coefficient in standardized_coefficients
                ]
            )
            real_equality_map = np.vstack(
                (equality_map.real, equality_map.imag)
            )
            _, singular_values, right = np.linalg.svd(
                real_equality_map,
                full_matrices=True,
            )
            rank_tolerance = max(
                1e-12,
                100.0 * np.finfo(float).eps
                * max(real_equality_map.shape)
                * (singular_values[0] if singular_values.size else 1.0),
            )
            rank = int(np.count_nonzero(singular_values > rank_tolerance))
            directions = np.asarray(right[rank:].T, dtype=float)
            face_residual = float(
                np.linalg.norm(real_equality_map @ directions, ord=np.inf)
            )
            qualification_error = max(qualification_error, face_residual)
            if directions.shape[1] == 0:
                return _FacialReduction(
                    values=values,
                    support=support,
                    directions=directions,
                    iterations=total_iterations,
                    success=False,
                    message="facial reduction removed every frontier direction",
                    qualification_error=qualification_error,
                )
        return _FacialReduction(
            values=values,
            support=support,
            directions=directions,
            iterations=total_iterations,
            success=True,
            message=(
                f"relative face support={support.shape[1]}/"
                f"{self.dimension}, directions={directions.shape[1]}/"
                f"{count}"
            ),
            qualification_error=qualification_error,
        )

    def _solve_conic_qp(
        self,
        quadratic: FloatArray,
        linear: FloatArray,
        cone_matrix: sparse.csc_matrix,
        cone_rhs: FloatArray,
        cones: tuple[object, ...],
    ) -> _ConicQPSolution:
        count = len(self.frontier_keys)
        quadratic = np.asarray(quadratic, dtype=float)
        linear = np.asarray(linear, dtype=float)
        if quadratic.shape != (count, count) or linear.shape != (count,):
            raise ValueError("conic QP data have inconsistent dimensions")
        quadratic = 0.5 * (quadratic + quadratic.T)
        rows, columns = np.triu_indices(count)
        packed_quadratic = sparse.coo_matrix(
            (quadratic[rows, columns], (rows, columns)),
            shape=(count, count),
        ).tocsc()
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = self.settings.maximum_iterations
        settings.max_threads = self.settings.clarabel_max_threads
        settings.tol_gap_abs = self.settings.conic_tolerance
        settings.tol_gap_rel = self.settings.conic_tolerance
        settings.tol_feas = self.settings.conic_tolerance
        solver = clarabel.DefaultSolver(
            packed_quadratic,
            linear,
            cone_matrix,
            cone_rhs,
            list(cones),
            settings,
        )
        solution = solver.solve()
        status = str(solution.status)
        primal_residual = float(solution.r_prim)
        dual_residual = float(solution.r_dual)
        success = bool(
            solution.x is not None
            and (
                status in ("Solved", "AlmostSolved")
                or (
                    status == "NumericalError"
                    and max(primal_residual, dual_residual)
                    <= 100.0 * self.settings.conic_tolerance
                )
            )
        )
        values = (
            np.asarray(solution.x, dtype=float)
            if solution.x is not None
            else np.zeros(count, dtype=float)
        )
        dual = (
            np.asarray(solution.z, dtype=float)
            if solution.z is not None
            else np.empty(0, dtype=float)
        )
        slack = (
            np.asarray(solution.s, dtype=float)
            if solution.s is not None
            else np.empty(0, dtype=float)
        )
        return _ConicQPSolution(
            values=values,
            dual=dual,
            slack=slack,
            iterations=int(solution.iterations),
            success=success,
            message=status,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
            primal_objective=float(solution.obj_val),
            dual_objective=float(solution.obj_val_dual),
        )

    def _conic_kkt_backward_error(
        self,
        values: FloatArray,
        gradient: FloatArray,
        cone_matrix: sparse.csc_matrix,
        cone_rhs: FloatArray,
        certificate: _ConicQPSolution,
        scaled_matrix: ComplexArray,
    ) -> float:
        primal_psd = max(
            0.0,
            -float(np.linalg.eigvalsh(scaled_matrix)[0]),
        )
        primal_box = float(
            np.max(np.maximum(np.abs(values) - 1.0, 0.0))
        )
        stationarity = np.asarray(gradient, dtype=float) + np.asarray(
            cone_matrix.T @ certificate.dual,
            dtype=float,
        ).reshape(-1)
        stationarity_error = float(
            np.linalg.norm(stationarity, ord=np.inf)
        ) / max(1.0, float(np.linalg.norm(gradient, ord=np.inf)))
        implied_slack = cone_rhs - cone_matrix @ np.asarray(values)
        complementarity = abs(
            float(np.dot(implied_slack, certificate.dual))
        ) / max(
            1.0,
            float(
                np.linalg.norm(implied_slack)
                * np.linalg.norm(certificate.dual)
            ),
        )
        return max(
            primal_psd,
            primal_box,
            stationarity_error,
            complementarity,
            certificate.primal_residual,
            certificate.dual_residual,
        )

    def _result_from_values(
        self,
        lower: Mapping[MomentKey, float],
        values: FloatArray,
        *,
        iterations: int,
        solver_success: bool,
        message: str,
        kkt_error: float,
        relative_face_support_dimension: int = 0,
        relative_face_direction_dimension: int = 0,
        facial_qualification_error: float = float("inf"),
        facial_reduction_certified: bool = False,
    ) -> APCMExtensionResult:
        values = np.asarray(values, dtype=float)
        frontier = MappingProxyType(
            {
                key: float(value)
                for key, value in zip(
                    self.frontier_keys,
                    values,
                    strict=True,
                )
            }
        )
        matrix = self.matrix(lower, frontier)
        scaled = self.scaled_matrix(matrix)
        minimum = float(np.linalg.eigvalsh(matrix)[0])
        scaled_eigenvalues = np.linalg.eigvalsh(scaled)
        scaled_minimum = float(scaled_eigenvalues[0])
        standardized = values / self.frontier_scales
        shifted_eigenvalues = (
            np.maximum(scaled_eigenvalues, 0.0)
            + self.settings.logdet_shift
        )
        objective = (
            0.5 * float(standardized @ standardized)
            - self.settings.logdet_weight
            / self.dimension
            * float(np.sum(np.log(shifted_eigenvalues)))
        )
        success = bool(
            solver_success
            and scaled_minimum >= -10.0 * self.settings.conic_tolerance
            and kkt_error <= 50.0 * self.settings.conic_tolerance
        )
        return APCMExtensionResult(
            lower_moments=MappingProxyType(dict(lower)),
            frontier_moments=frontier,
            moment_matrix=matrix,
            scaled_minimum_eigenvalue=scaled_minimum,
            minimum_eigenvalue=minimum,
            objective=objective,
            kkt_backward_error=kkt_error,
            iterations=iterations,
            success=success,
            message=message,
            relative_face_support_dimension=relative_face_support_dimension,
            relative_face_direction_dimension=relative_face_direction_dimension,
            facial_qualification_error=facial_qualification_error,
            facial_reduction_certified=facial_reduction_certified,
        )

    def auxiliary_covariance(
        self,
        result: APCMExtensionResult,
        active_values: FloatArray,
    ) -> FloatArray:
        """Return ``Re Cov(R_A,R_A)`` from the selected positive lift."""

        values = np.asarray(active_values, dtype=float)
        if values.shape != (len(self.active_keys),):
            raise ValueError("active_values has the wrong dimension")
        block = result.moment_matrix[
            np.ix_(self.active_word_indices, self.active_word_indices)
        ]
        covariance = np.real(block) - np.outer(values, values)
        return np.asarray(0.5 * (covariance + covariance.T), dtype=float)

    def auxiliary_metric(
        self,
        result: APCMExtensionResult,
        active_values: FloatArray,
    ) -> FloatArray:
        """Return the regularized covariance-dual auxiliary metric."""

        covariance = self.auxiliary_covariance(result, active_values)
        inverse_scale = np.diag(1.0 / self.active_scales)
        standardized = inverse_scale @ covariance @ inverse_scale
        standardized = 0.5 * (standardized + standardized.T)
        regularization = max(
            1e-10,
            10.0 * result.kkt_backward_error,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(standardized)
        inverse = eigenvectors @ np.diag(
            1.0 / np.maximum(eigenvalues + regularization, regularization)
        ) @ eigenvectors.T
        metric = inverse_scale @ inverse @ inverse_scale
        return np.asarray(0.5 * (metric + metric.T), dtype=float)

    def _kkt_backward_error(
        self,
        values: FloatArray,
        scaled_matrix: ComplexArray,
    ) -> float:
        """Compute a dimensionless primal/dual/complementarity audit."""

        tolerance_floor = np.finfo(float).eps
        primal_psd = max(0.0, -float(np.linalg.eigvalsh(scaled_matrix)[0]))
        primal_box = float(
            np.max(
                np.maximum(
                    np.abs(values) / self.frontier_scales - 1.0,
                    0.0,
                )
            )
        )
        if (
            self._psd_constraint is None
            or self._lower_bound_constraint is None
            or self._upper_bound_constraint is None
        ):
            return float("inf")
        psd_dual = self._psd_constraint.dual_value
        lower_dual = self._lower_bound_constraint.dual_value
        upper_dual = self._upper_bound_constraint.dual_value
        if psd_dual is None or lower_dual is None or upper_dual is None:
            return float("inf")
        psd_dual = np.asarray(psd_dual, dtype=complex)
        lower_dual = np.asarray(lower_dual, dtype=float).reshape(-1)
        upper_dual = np.asarray(upper_dual, dtype=float).reshape(-1)
        dual_psd = max(0.0, -float(np.linalg.eigvalsh(psd_dual)[0]))
        complementarity = abs(
            float(np.trace(psd_dual @ scaled_matrix).real)
        ) / max(1.0, float(np.linalg.norm(psd_dual) * np.linalg.norm(scaled_matrix)))

        inverse_word = np.diag(1.0 / self.word_scales)
        scaled_coefficients = np.einsum(
            "ab,kbc,cd->kad",
            inverse_word,
            self.frontier_coefficients,
            inverse_word,
            optimize=True,
        )
        shifted_inverse = np.linalg.inv(
            scaled_matrix
            + self.settings.logdet_shift
            * np.eye(self.dimension, dtype=complex)
        )
        objective_gradient = values / self.frontier_scales**2
        objective_gradient -= (
            self.settings.logdet_weight
            / self.dimension
            * np.asarray(
                [
                    np.trace(shifted_inverse @ coefficient).real
                    for coefficient in scaled_coefficients
                ],
                dtype=float,
            )
        )
        cone_gradient = np.asarray(
            [
                np.trace(psd_dual @ coefficient).real
                for coefficient in scaled_coefficients
            ],
            dtype=float,
        )
        stationarity = (
            objective_gradient - cone_gradient - lower_dual + upper_dual
        )
        stationarity_error = float(np.linalg.norm(stationarity, ord=np.inf)) / max(
            1.0,
            float(np.linalg.norm(objective_gradient, ord=np.inf)),
            tolerance_floor,
        )
        return max(
            primal_psd,
            primal_box,
            dual_psd,
            complementarity,
            stationarity_error,
        )

__all__ = [
    "APCMLinearExtrema",
    "APCMExtensionResult",
    "APCMExtensionSettings",
    "SymmetryReducedPositiveExtension",
    "apcm_extension_coefficients",
    "apcm_extension_words",
]
