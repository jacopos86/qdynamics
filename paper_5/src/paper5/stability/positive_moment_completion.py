"""Positive terminal completion for the spin-symmetric Pauli--Weyl hierarchy.

The module exposes one deep seam: :class:`PositiveFourthMomentCompletion`
maps all moments through degree three to a deterministic degree-four
completion and a numerical certificate.  Callers do not assemble the
noncommutative Gram matrix or solve its constrained log-determinant problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Literal, Mapping

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .moment_hierarchy import (
    IDENTITY,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    MomentKey,
    PreparedTerminalMomentClosure,
    _OperatorKey,
    _operator_product,
    build_moment_keys,
    ZERO_CUMULANT_CLOSURE,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PositiveMomentCompletionSettings:
    """Frozen numerical and prior settings for one completion solve."""

    phonon_envelope: float = 16.0
    logdet_weight: float = 1.0
    logdet_shift: float = 1e-10
    minimum_eigenvalue: float = 0.0
    solver_tolerance: float = 1e-9
    maximum_iterations: int = 800
    envelope_multiplier: float = 4.0
    solver: Literal["slsqp", "cvxpy"] = "slsqp"
    cone_representation: Literal[
        "full", "spin_exchange_blocks"
    ] = "full"
    clarabel_max_threads: int = 0

    def __post_init__(self) -> None:
        if self.phonon_envelope <= 0.0:
            raise ValueError("phonon_envelope must be positive")
        if self.logdet_weight < 0.0:
            raise ValueError("logdet_weight must be nonnegative")
        if self.logdet_shift <= 0.0:
            raise ValueError("logdet_shift must be positive")
        if self.minimum_eigenvalue < 0.0:
            raise ValueError("minimum_eigenvalue must be nonnegative")
        if self.solver_tolerance <= 0.0:
            raise ValueError("solver_tolerance must be positive")
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive")
        if self.envelope_multiplier <= 1.0:
            raise ValueError("envelope_multiplier must exceed one")
        if self.solver not in ("slsqp", "cvxpy"):
            raise ValueError("solver must be 'slsqp' or 'cvxpy'")
        if self.cone_representation not in (
            "full",
            "spin_exchange_blocks",
        ):
            raise ValueError(
                "cone_representation must be 'full' or "
                "a spin-exchange representation"
            )
        if self.clarabel_max_threads < 0:
            raise ValueError("clarabel_max_threads must be nonnegative")


@dataclass(frozen=True)
class PositiveMomentCompletionResult(PreparedTerminalMomentClosure):
    """Completed degree-four moments and the conic numerical certificate."""

    moments: Mapping[MomentKey, float]
    frontier_moments: Mapping[MomentKey, float]
    prior_frontier_moments: Mapping[MomentKey, float]
    minimum_moment_matrix_eigenvalue: float
    scaled_prior_distance: float
    objective: float
    iterations: int
    success: bool
    message: str

    def moment(self, key: MomentKey) -> float:
        if key.degree == 0:
            return 1.0
        try:
            return float(self.moments[key])
        except KeyError as error:
            raise ValueError(f"completion does not contain moment {key}") from error


@dataclass(frozen=True)
class PositiveMomentRetractionResult:
    """Minimum scaled adjustment of selected lower moments into the cone."""

    lower_moments: Mapping[MomentKey, float]
    frontier_moments: Mapping[MomentKey, float]
    minimum_moment_matrix_eigenvalue: float
    scaled_lower_correction_norm: float
    iterations: int
    success: bool
    message: str


_PAULI_LABELS = (PAULI_X, PAULI_Y, PAULI_Z)


@lru_cache(maxsize=1)
def _moment_matrix_words() -> tuple[_OperatorKey, ...]:
    words: list[_OperatorKey] = [
        _OperatorKey(IDENTITY, IDENTITY, 0, 0)
    ]
    for spin in (0, 1):
        for label in _PAULI_LABELS:
            labels = [IDENTITY, IDENTITY]
            labels[spin] = label
            words.append(_OperatorKey(labels[0], labels[1], 0, 0))
    words.extend(
        [
            _OperatorKey(IDENTITY, IDENTITY, 1, 0),
            _OperatorKey(IDENTITY, IDENTITY, 0, 1),
        ]
    )
    for up_label in _PAULI_LABELS:
        for down_label in _PAULI_LABELS:
            words.append(_OperatorKey(up_label, down_label, 0, 0))
    for spin in (0, 1):
        for label in _PAULI_LABELS:
            for x_power, p_power in ((1, 0), (0, 1)):
                labels = [IDENTITY, IDENTITY]
                labels[spin] = label
                words.append(
                    _OperatorKey(
                        labels[0],
                        labels[1],
                        x_power,
                        p_power,
                    )
                )
    words.extend(
        [
            _OperatorKey(IDENTITY, IDENTITY, 2, 0),
            _OperatorKey(IDENTITY, IDENTITY, 1, 1),
            _OperatorKey(IDENTITY, IDENTITY, 0, 2),
        ]
    )
    return tuple(words)


@lru_cache(maxsize=1)
def _moment_matrix_coefficients() -> Mapping[MomentKey, ComplexArray]:
    words = _moment_matrix_words()
    coefficients: dict[MomentKey, ComplexArray] = {}
    for row, left in enumerate(words):
        for column, right in enumerate(words):
            for key, coefficient in _operator_product(left, right).items():
                if key.degree > 4:
                    raise RuntimeError(
                        "degree-two word Gram generated a moment above degree four"
                    )
                matrix = coefficients.setdefault(
                    key,
                    np.zeros((len(words), len(words)), dtype=complex),
                )
                matrix[row, column] += coefficient
    for key, matrix in coefficients.items():
        hermitian = 0.5 * (matrix + matrix.conjugate().T)
        if np.linalg.norm(matrix - hermitian) > 2e-12:
            raise RuntimeError(
                f"moment coefficient for {key} is not Hermitian"
            )
        coefficients[key] = hermitian
    return MappingProxyType(coefficients)


@lru_cache(maxsize=1)
def _spin_exchange_transform() -> tuple[ComplexArray, int]:
    """Return the exact symmetric/antisymmetric word-basis transform."""

    words = _moment_matrix_words()
    lookup = {word: index for index, word in enumerate(words)}
    permutation = tuple(
        lookup[
            _OperatorKey(
                word.spin_down,
                word.spin_up,
                word.x_power,
                word.p_power,
            )
        ]
        for word in words
    )
    symmetric: list[ComplexArray] = []
    antisymmetric: list[ComplexArray] = []
    used: set[int] = set()
    inverse_sqrt_two = 1.0 / np.sqrt(2.0)
    for left, right in enumerate(permutation):
        if left in used:
            continue
        if left == right:
            vector = np.zeros(len(words), dtype=complex)
            vector[left] = 1.0
            symmetric.append(vector)
            used.add(left)
            continue
        plus = np.zeros(len(words), dtype=complex)
        minus = np.zeros(len(words), dtype=complex)
        plus[left] = inverse_sqrt_two
        plus[right] = inverse_sqrt_two
        minus[left] = inverse_sqrt_two
        minus[right] = -inverse_sqrt_two
        symmetric.append(plus)
        antisymmetric.append(minus)
        used.update((left, right))
    transform = np.column_stack((*symmetric, *antisymmetric))
    if not np.allclose(
        transform.conjugate().T @ transform,
        np.eye(len(words)),
        atol=2e-14,
        rtol=0.0,
    ):
        raise RuntimeError("spin-exchange word transform is not unitary")
    return transform, len(symmetric)


@lru_cache(maxsize=1)
def _spin_exchange_coefficient_blocks() -> Mapping[
    MomentKey, tuple[ComplexArray, ComplexArray]
]:
    """Batch-transform coefficient matrices into the two exact PSD blocks."""

    coefficients = _moment_matrix_coefficients()
    keys = tuple(coefficients)
    stacked = np.asarray([coefficients[key] for key in keys], dtype=complex)
    transform, split = _spin_exchange_transform()
    transformed = np.einsum(
        "ai,kab,bj->kij",
        transform.conjugate(),
        stacked,
        transform,
        optimize=True,
    )
    cross_norm = np.linalg.norm(transformed[:, :split, split:])
    if cross_norm > 2e-12:
        raise RuntimeError(
            "spin-exchange transform did not block-diagonalize the cone"
        )
    return MappingProxyType(
        {
            key: (
                transformed[index, :split, :split],
                transformed[index, split:, split:],
            )
            for index, key in enumerate(keys)
        }
    )


def pauli_weyl_moment_matrix(
    moments: Mapping[MomentKey, float],
) -> ComplexArray:
    """Assemble the degree-two-word noncommutative moment matrix."""

    size = len(_moment_matrix_words())
    matrix = np.zeros((size, size), dtype=complex)
    for key, coefficient in _moment_matrix_coefficients().items():
        if key.degree == 0:
            value = 1.0
        else:
            try:
                value = float(moments[key])
            except KeyError as error:
                raise ValueError(f"moment matrix requires {key}") from error
        matrix += value * coefficient
    return 0.5 * (matrix + matrix.conjugate().T)


def _frontier_scale(key: MomentKey, phonon_envelope: float) -> float:
    boson_degree = key.x_power + key.p_power
    return float((2.0 * phonon_envelope + 1.0) ** (0.5 * boson_degree))


def _clarabel_options(
    settings: PositiveMomentCompletionSettings,
) -> dict[str, float | int | str]:
    """Return the frozen tolerances and optional bounded-thread backend."""

    options: dict[str, float | int | str] = {
        "max_iter": settings.maximum_iterations,
        "tol_gap_abs": settings.solver_tolerance,
        "tol_gap_rel": settings.solver_tolerance,
        "tol_feas": settings.solver_tolerance,
    }
    if settings.clarabel_max_threads > 0:
        options.update(
            direct_solve_method="faer",
            max_threads=settings.clarabel_max_threads,
        )
    return options


class PositiveFourthMomentCompletion:
    """Deterministically complete degree-four moments inside a PSD Gram cone."""

    name = "positive_fourth_moment"

    def __init__(
        self,
        settings: PositiveMomentCompletionSettings | None = None,
    ) -> None:
        self.settings = (
            PositiveMomentCompletionSettings()
            if settings is None
            else settings
        )
        self._lower_keys = tuple(build_moment_keys(3))
        self._frontier_keys = tuple(
            key for key in build_moment_keys(4) if key.degree == 4
        )
        coefficients = _moment_matrix_coefficients()
        self._frontier_matrices = np.asarray(
            [coefficients[key] for key in self._frontier_keys],
            dtype=complex,
        )
        if self.settings.cone_representation == "full":
            self._coefficient_blocks = (coefficients,)
        else:
            blocked = _spin_exchange_coefficient_blocks()
            self._coefficient_blocks = tuple(
                MappingProxyType(
                    {
                        key: blocked[key][block]
                        for key in blocked
                    }
                )
                for block in range(2)
            )
        self._frontier_matrix_blocks = tuple(
            np.asarray(
                [block[key] for key in self._frontier_keys],
                dtype=complex,
            )
            for block in self._coefficient_blocks
        )
        self._scales = np.asarray(
            [
                _frontier_scale(key, self.settings.phonon_envelope)
                for key in self._frontier_keys
            ],
            dtype=float,
        )
        self._cvxpy_problem = None
        self._lower_parameter = None
        self._prior_parameter = None
        self._frontier_variable = None
        self._matrix_expressions: tuple[cp.Expression, ...] = ()
        if self.settings.solver == "cvxpy":
            identity_key = MomentKey(IDENTITY, IDENTITY, 0, 0)
            self._lower_parameter = cp.Parameter(len(self._lower_keys))
            self._prior_parameter = cp.Parameter(len(self._frontier_keys))
            self._frontier_variable = cp.Variable(len(self._frontier_keys))
            matrix_expressions: list[cp.Expression] = []
            for coefficient_block, frontier_block in zip(
                self._coefficient_blocks,
                self._frontier_matrix_blocks,
                strict=True,
            ):
                matrix_expression = cp.Constant(
                    coefficient_block[identity_key]
                )
                for index, key in enumerate(self._lower_keys):
                    matrix_expression += (
                        self._lower_parameter[index]
                        * cp.Constant(coefficient_block[key])
                    )
                for index, basis in enumerate(frontier_block):
                    matrix_expression += (
                        self._frontier_variable[index] * cp.Constant(basis)
                    )
                matrix_expressions.append(cp.hermitian_wrap(matrix_expression))
            self._matrix_expressions = tuple(matrix_expressions)
            standardized = cp.multiply(
                1.0 / self._scales,
                self._frontier_variable - self._prior_parameter,
            )
            objective = 0.5 * cp.sum_squares(standardized)
            if self.settings.logdet_weight > 0.0:
                for expression in self._matrix_expressions:
                    shifted = expression + (
                        self.settings.logdet_shift
                        * np.eye(expression.shape[0], dtype=complex)
                    )
                    objective -= (
                        self.settings.logdet_weight * cp.log_det(shifted)
                    )
            lower_bounds = -self.settings.envelope_multiplier * self._scales
            upper_bounds = self.settings.envelope_multiplier * self._scales
            cone_constraints = [
                expression
                >> self.settings.minimum_eigenvalue
                * np.eye(expression.shape[0])
                for expression in self._matrix_expressions
            ]
            self._cvxpy_problem = cp.Problem(
                cp.Minimize(objective),
                [
                    *cone_constraints,
                    self._frontier_variable >= lower_bounds,
                    self._frontier_variable <= upper_bounds,
                ],
            )
        self._retraction_cache: dict[tuple[MomentKey, ...], tuple] = {}

    @property
    def frontier_keys(self) -> tuple[MomentKey, ...]:
        return self._frontier_keys

    def prepare(
        self,
        moments: Mapping[MomentKey, float],
        maximum_degree: int,
    ) -> PositiveMomentCompletionResult:
        if maximum_degree != 3:
            raise ValueError(
                "positive fourth-moment completion requires maximum_degree=3"
            )
        return self.complete(moments)

    def prior_result(
        self,
        lower_moments: Mapping[MomentKey, float],
    ) -> PositiveMomentCompletionResult:
        """Evaluate the frozen zero-cumulant frontier without cone repair."""

        missing = set(self._lower_keys).difference(lower_moments)
        if missing:
            raise ValueError(
                f"terminal prior is missing {len(missing)} lower moments"
            )
        lower = {key: float(lower_moments[key]) for key in self._lower_keys}
        prior_resolver = ZERO_CUMULANT_CLOSURE.prepare(lower, 3)
        frontier = {
            key: float(prior_resolver.moment(key))
            for key in self._frontier_keys
        }
        matrix = pauli_weyl_moment_matrix({**lower, **frontier})
        minimum = float(np.linalg.eigvalsh(matrix)[0])
        return PositiveMomentCompletionResult(
            moments=MappingProxyType({**lower, **frontier}),
            frontier_moments=MappingProxyType(frontier),
            prior_frontier_moments=MappingProxyType(frontier),
            minimum_moment_matrix_eigenvalue=minimum,
            scaled_prior_distance=0.0,
            objective=0.0,
            iterations=0,
            success=True,
            message="unconstrained zero-cumulant terminal prior",
        )

    def complete(
        self,
        lower_moments: Mapping[MomentKey, float],
        *,
        initial_frontier: Mapping[MomentKey, float] | None = None,
    ) -> PositiveMomentCompletionResult:
        missing = set(self._lower_keys).difference(lower_moments)
        if missing:
            raise ValueError(f"completion is missing {len(missing)} lower moments")
        lower = {key: float(lower_moments[key]) for key in self._lower_keys}
        prior_resolver = ZERO_CUMULANT_CLOSURE.prepare(lower, 3)
        prior = np.asarray(
            [prior_resolver.moment(key) for key in self._frontier_keys],
            dtype=float,
        )
        if initial_frontier is None:
            initial = prior.copy()
        else:
            initial = np.asarray(
                [float(initial_frontier[key]) for key in self._frontier_keys],
                dtype=float,
            )

        if self._cvxpy_problem is not None:
            assert self._lower_parameter is not None
            assert self._prior_parameter is not None
            assert self._frontier_variable is not None
            self._lower_parameter.value = np.asarray(
                [lower[key] for key in self._lower_keys],
                dtype=float,
            )
            self._prior_parameter.value = prior
            self._frontier_variable.value = initial
            cvxpy_message = ""
            cvxpy_iterations = 0
            try:
                self._cvxpy_problem.solve(
                    solver=cp.CLARABEL,
                    warm_start=True,
                    verbose=False,
                    **_clarabel_options(self.settings),
                )
                cvxpy_message = str(self._cvxpy_problem.status)
                solver_stats = self._cvxpy_problem.solver_stats
                cvxpy_iterations = int(solver_stats.num_iters or 0)
            except cp.error.SolverError as error:
                cvxpy_message = f"CLARABEL failure: {error}"
            cvxpy_values = self._frontier_variable.value
        else:
            cvxpy_values = None
            cvxpy_message = ""
            cvxpy_iterations = 0
        if cvxpy_values is not None:
            values = np.asarray(cvxpy_values, dtype=float).reshape(-1)
            cvxpy_matrix = pauli_weyl_moment_matrix(
                {
                    **lower,
                    **{
                        key: float(value)
                        for key, value in zip(
                            self._frontier_keys,
                            values,
                            strict=True,
                        )
                    },
                }
            )
            cvxpy_minimum = float(np.linalg.eigvalsh(cvxpy_matrix)[0])
            cvxpy_success = bool(
                self._cvxpy_problem.status
                in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                and cvxpy_minimum
                >= self.settings.minimum_eigenvalue
                - 10.0 * self.settings.solver_tolerance
            )
            if cvxpy_success:
                frontier = {
                    key: float(value)
                    for key, value in zip(
                        self._frontier_keys,
                        values,
                        strict=True,
                    )
                }
                prior_frontier = {
                    key: float(value)
                    for key, value in zip(
                        self._frontier_keys,
                        prior,
                        strict=True,
                    )
                }
                all_moments = {**lower, **frontier}
                shifted_matrix = cvxpy_matrix + (
                    self.settings.logdet_shift
                    * np.eye(cvxpy_matrix.shape[0], dtype=complex)
                )
                sign, logdet = np.linalg.slogdet(shifted_matrix)
                standardized_values = (values - prior) / self._scales
                objective_value = (
                    0.5 * float(standardized_values @ standardized_values)
                    - self.settings.logdet_weight * float(logdet)
                    if sign > 0.0
                    else float("inf")
                )
                return PositiveMomentCompletionResult(
                    moments=MappingProxyType(all_moments),
                    frontier_moments=MappingProxyType(frontier),
                    prior_frontier_moments=MappingProxyType(prior_frontier),
                    minimum_moment_matrix_eigenvalue=cvxpy_minimum,
                    scaled_prior_distance=float(
                        np.linalg.norm(standardized_values)
                    ),
                    objective=objective_value,
                    iterations=cvxpy_iterations,
                    success=True,
                    message=cvxpy_message,
                )

        coefficients = _moment_matrix_coefficients()
        base = np.zeros_like(self._frontier_matrices[0])
        for key, value in lower.items():
            base += value * coefficients[key]
        identity_key = MomentKey(IDENTITY, IDENTITY, 0, 0)
        base += coefficients[identity_key]

        def matrix(values: FloatArray) -> ComplexArray:
            result = base + np.tensordot(
                np.asarray(values, dtype=float),
                self._frontier_matrices,
                axes=(0, 0),
            )
            return 0.5 * (result + result.conjugate().T)

        def minimum_eigenvalue(values: FloatArray) -> float:
            return float(np.linalg.eigvalsh(matrix(values))[0])

        def minimum_eigenvalue_jacobian(values: FloatArray) -> FloatArray:
            _, eigenvectors = np.linalg.eigh(matrix(values))
            mode = eigenvectors[:, 0]
            return np.asarray(
                [
                    np.vdot(mode, basis @ mode).real
                    for basis in self._frontier_matrices
                ],
                dtype=float,
            )

        settings = self.settings
        bounds = [
            (
                -settings.envelope_multiplier * scale,
                settings.envelope_multiplier * scale,
            )
            for scale in self._scales
        ]
        constraint = {
            "type": "ineq",
            "fun": lambda values: (
                minimum_eigenvalue(values) - settings.minimum_eigenvalue
            ),
            "jac": minimum_eigenvalue_jacobian,
        }

        if minimum_eigenvalue(initial) < (
            settings.minimum_eigenvalue - 10.0 * settings.solver_tolerance
        ):
            feasibility = minimize(
                lambda values: 0.5
                * float(
                    np.dot(
                        (values - prior) / self._scales,
                        (values - prior) / self._scales,
                    )
                ),
                initial,
                jac=lambda values: (values - prior) / self._scales**2,
                bounds=bounds,
                constraints=(constraint,),
                method="SLSQP",
                options={
                    "ftol": settings.solver_tolerance,
                    "maxiter": settings.maximum_iterations,
                    "disp": False,
                },
            )
            initial = np.asarray(feasibility.x, dtype=float)

        def objective(values: FloatArray) -> float:
            standardized = (values - prior) / self._scales
            shifted = matrix(values) + settings.logdet_shift * np.eye(
                base.shape[0], dtype=complex
            )
            sign, logdet = np.linalg.slogdet(shifted)
            if sign <= 0.0 or not np.isfinite(logdet):
                return 1e30 + 1e12 * float(np.dot(standardized, standardized))
            return float(
                0.5 * np.dot(standardized, standardized)
                - settings.logdet_weight * logdet
            )

        def gradient(values: FloatArray) -> FloatArray:
            shifted = matrix(values) + settings.logdet_shift * np.eye(
                base.shape[0], dtype=complex
            )
            inverse = np.linalg.inv(shifted)
            logdet_gradient = np.asarray(
                [
                    np.trace(inverse @ basis).real
                    for basis in self._frontier_matrices
                ],
                dtype=float,
            )
            return (
                (values - prior) / self._scales**2
                - settings.logdet_weight * logdet_gradient
            )

        solution = minimize(
            objective,
            initial,
            jac=gradient,
            bounds=bounds,
            constraints=(constraint,),
            method="SLSQP",
            options={
                "ftol": settings.solver_tolerance,
                "maxiter": settings.maximum_iterations,
                "disp": False,
            },
        )
        values = np.asarray(solution.x, dtype=float)
        minimum = minimum_eigenvalue(values)
        conic_success = minimum >= (
            settings.minimum_eigenvalue - 10.0 * settings.solver_tolerance
        )
        success = bool(solution.success and conic_success)
        frontier = {
            key: float(value)
            for key, value in zip(self._frontier_keys, values, strict=True)
        }
        prior_frontier = {
            key: float(value)
            for key, value in zip(self._frontier_keys, prior, strict=True)
        }
        all_moments = {**lower, **frontier}
        return PositiveMomentCompletionResult(
            moments=MappingProxyType(all_moments),
            frontier_moments=MappingProxyType(frontier),
            prior_frontier_moments=MappingProxyType(prior_frontier),
            minimum_moment_matrix_eigenvalue=minimum,
            scaled_prior_distance=float(
                np.linalg.norm((values - prior) / self._scales)
            ),
            objective=objective(values),
            iterations=int(getattr(solution, "nit", 0)),
            success=success,
            message=str(solution.message),
        )

    def result_from_frontier(
        self,
        lower_moments: Mapping[MomentKey, float],
        frontier_moments: Mapping[MomentKey, float],
    ) -> PositiveMomentCompletionResult:
        """Certify an already selected frontier without resolving the prior."""

        lower = {key: float(lower_moments[key]) for key in self._lower_keys}
        missing = set(self._frontier_keys).difference(frontier_moments)
        if missing:
            raise ValueError(f"provided frontier is missing {len(missing)} moments")
        frontier = {
            key: float(frontier_moments[key]) for key in self._frontier_keys
        }
        prior_resolver = ZERO_CUMULANT_CLOSURE.prepare(lower, 3)
        prior = {
            key: float(prior_resolver.moment(key))
            for key in self._frontier_keys
        }
        matrix = pauli_weyl_moment_matrix({**lower, **frontier})
        minimum = float(np.linalg.eigvalsh(matrix)[0])
        values = np.asarray(
            [frontier[key] for key in self._frontier_keys], dtype=float
        )
        prior_values = np.asarray(
            [prior[key] for key in self._frontier_keys], dtype=float
        )
        standardized = (values - prior_values) / self._scales
        shifted = matrix + self.settings.logdet_shift * np.eye(
            matrix.shape[0], dtype=complex
        )
        sign, logdet = np.linalg.slogdet(shifted)
        objective = (
            0.5 * float(standardized @ standardized)
            - self.settings.logdet_weight * float(logdet)
            if sign > 0.0
            else float("inf")
        )
        success = minimum >= (
            self.settings.minimum_eigenvalue
            - 10.0 * self.settings.solver_tolerance
        )
        return PositiveMomentCompletionResult(
            moments=MappingProxyType({**lower, **frontier}),
            frontier_moments=MappingProxyType(frontier),
            prior_frontier_moments=MappingProxyType(prior),
            minimum_moment_matrix_eigenvalue=minimum,
            scaled_prior_distance=float(np.linalg.norm(standardized)),
            objective=objective,
            iterations=0,
            success=bool(success),
            message=(
                "provided feasible frontier"
                if success
                else "provided frontier violates the moment cone"
            ),
        )

    def retract_lower_moments(
        self,
        lower_moments: Mapping[MomentKey, float],
        *,
        adjustable_keys: tuple[MomentKey, ...],
        warm_frontier: Mapping[MomentKey, float] | None = None,
    ) -> PositiveMomentRetractionResult:
        """Retract selected lower moments while holding the resolved ones fixed."""

        if not adjustable_keys:
            raise ValueError("adjustable_keys must not be empty")
        if not set(adjustable_keys).issubset(self._lower_keys):
            raise ValueError("every adjustable key must be a retained lower moment")
        lower = {key: float(lower_moments[key]) for key in self._lower_keys}
        cache_key = tuple(adjustable_keys)
        cached = self._retraction_cache.get(cache_key)
        if cached is None:
            fixed_keys = tuple(
                key for key in self._lower_keys if key not in cache_key
            )
            fixed_parameter = cp.Parameter(len(fixed_keys))
            target_parameter = cp.Parameter(len(cache_key))
            prior_parameter = cp.Parameter(len(self._frontier_keys))
            adjustable = cp.Variable(len(cache_key))
            frontier = cp.Variable(len(self._frontier_keys))
            identity_key = MomentKey(IDENTITY, IDENTITY, 0, 0)
            expressions: list[cp.Expression] = []
            for coefficient_block, frontier_block in zip(
                self._coefficient_blocks,
                self._frontier_matrix_blocks,
                strict=True,
            ):
                expression = cp.Constant(coefficient_block[identity_key])
                for index, key in enumerate(fixed_keys):
                    expression += fixed_parameter[index] * cp.Constant(
                        coefficient_block[key]
                    )
                for index, key in enumerate(cache_key):
                    expression += adjustable[index] * cp.Constant(
                        coefficient_block[key]
                    )
                for index, basis in enumerate(frontier_block):
                    expression += frontier[index] * cp.Constant(basis)
                expressions.append(cp.hermitian_wrap(expression))
            adjustable_scales = np.asarray(
                [
                    _frontier_scale(key, self.settings.phonon_envelope)
                    for key in cache_key
                ],
                dtype=float,
            )
            lower_bounds = -self.settings.envelope_multiplier * self._scales
            upper_bounds = self.settings.envelope_multiplier * self._scales
            objective = 0.5 * cp.sum_squares(
                cp.multiply(
                    1.0 / adjustable_scales,
                    adjustable - target_parameter,
                )
            ) + 1e-8 * cp.sum_squares(
                cp.multiply(
                    1.0 / self._scales,
                    frontier - prior_parameter,
                )
            )
            problem = cp.Problem(
                cp.Minimize(objective),
                [
                    *(expression >> 0.0 for expression in expressions),
                    frontier >= lower_bounds,
                    frontier <= upper_bounds,
                ],
            )
            cached = (
                fixed_keys,
                fixed_parameter,
                target_parameter,
                prior_parameter,
                adjustable,
                frontier,
                problem,
                adjustable_scales,
            )
            self._retraction_cache[cache_key] = cached
        (
            fixed_keys,
            fixed_parameter,
            target_parameter,
            prior_parameter,
            adjustable,
            frontier,
            problem,
            adjustable_scales,
        ) = cached
        prior_resolver = ZERO_CUMULANT_CLOSURE.prepare(lower, 3)
        prior = np.asarray(
            [prior_resolver.moment(key) for key in self._frontier_keys],
            dtype=float,
        )
        fixed_parameter.value = np.asarray(
            [lower[key] for key in fixed_keys], dtype=float
        )
        target = np.asarray([lower[key] for key in cache_key], dtype=float)
        target_parameter.value = target
        prior_parameter.value = prior
        adjustable.value = target
        if warm_frontier is not None:
            frontier.value = np.asarray(
                [warm_frontier[key] for key in self._frontier_keys], dtype=float
            )
        else:
            frontier.value = prior
        message = ""
        iterations = 0
        try:
            problem.solve(
                solver=cp.CLARABEL,
                warm_start=True,
                verbose=False,
                **_clarabel_options(self.settings),
            )
            message = str(problem.status)
            iterations = int(problem.solver_stats.num_iters or 0)
        except cp.error.SolverError as error:
            message = f"CLARABEL failure: {error}"
        if adjustable.value is None or frontier.value is None:
            return PositiveMomentRetractionResult(
                lower_moments=MappingProxyType(lower),
                frontier_moments=MappingProxyType({}),
                minimum_moment_matrix_eigenvalue=float("-inf"),
                scaled_lower_correction_norm=float("inf"),
                iterations=iterations,
                success=False,
                message=message,
            )
        corrected_lower = dict(lower)
        adjusted_values = np.asarray(adjustable.value, dtype=float).reshape(-1)
        for key, value in zip(cache_key, adjusted_values, strict=True):
            corrected_lower[key] = float(value)
        frontier_values = np.asarray(frontier.value, dtype=float).reshape(-1)
        corrected_frontier = {
            key: float(value)
            for key, value in zip(
                self._frontier_keys,
                frontier_values,
                strict=True,
            )
        }
        minimum = float(
            np.linalg.eigvalsh(
                pauli_weyl_moment_matrix(
                    {**corrected_lower, **corrected_frontier}
                )
            )[0]
        )
        success = bool(
            problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            and minimum >= -10.0 * self.settings.solver_tolerance
        )
        return PositiveMomentRetractionResult(
            lower_moments=MappingProxyType(corrected_lower),
            frontier_moments=MappingProxyType(corrected_frontier),
            minimum_moment_matrix_eigenvalue=minimum,
            scaled_lower_correction_norm=float(
                np.linalg.norm(
                    (adjusted_values - target) / adjustable_scales
                )
            ),
            iterations=iterations,
            success=success,
            message=message,
        )


__all__ = [
    "PositiveFourthMomentCompletion",
    "PositiveMomentCompletionResult",
    "PositiveMomentCompletionSettings",
    "PositiveMomentRetractionResult",
    "pauli_weyl_moment_matrix",
]
