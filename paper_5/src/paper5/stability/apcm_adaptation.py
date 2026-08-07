"""Closure-graph diagnostics for adaptive projected APCM dictionaries.

The diagnostic compares the directional derivative of the current positive
completion graph with the exact symbolic commutator of each RHS-facing
frontier operator.  A one-additional-shell positive extension conditions the
current RHS-facing frontier, reopens the auxiliary frontier, and supplies both
a unique point value and outward linear-functional extrema.  No exact or
packet trajectory enters this calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm

from .adaptive_positive_moment import RAW_MOMENT_COORDINATE_NAMES
from .apcm_positive_extension import (
    APCMExtensionResult,
    APCMLinearExtrema,
    SymmetryReducedPositiveExtension,
)
from .hubbard_dimer import FloatArray
from .moment_hierarchy import (
    IDENTITY,
    MomentKey,
    _OperatorKey,
    _commutator,
    _hamiltonian_terms,
    _operator_product,
)
from .projected_apcm import (
    FixedDictionaryProjectedAPCM,
    ProjectedAPCMEvaluation,
    ProjectedAPCMFailure,
    unpack_projected_apcm_state,
)
from .adaptive_positive_moment import uncentered_joint_moment_matrix
from .apcm_moment_projection import APCMProjectionError, state_lower_moments


@dataclass(frozen=True)
class APCMAdaptationSettings:
    """Frozen local rules for one closure-graph checkpoint."""

    checkpoint_interval: float = 1e-2
    derivative_step_scale: float = 1e-5
    derivative_refinement_tolerance: float = 0.05
    jacobian_step_scale: float = 1e-7
    impact_threshold: float = 1.0

    def __post_init__(self) -> None:
        if self.checkpoint_interval <= 0.0:
            raise ValueError("checkpoint_interval must be positive")
        if self.derivative_step_scale <= 0.0:
            raise ValueError("derivative_step_scale must be positive")
        if self.derivative_refinement_tolerance <= 0.0:
            raise ValueError("derivative_refinement_tolerance must be positive")
        if self.jacobian_step_scale <= 0.0:
            raise ValueError("jacobian_step_scale must be positive")
        if self.impact_threshold <= 0.0:
            raise ValueError("impact_threshold must be positive")


@dataclass(frozen=True)
class APCMCandidateDiagnostic:
    """One bounded or unresolved closure-graph admission candidate."""

    key: MomentKey
    graph_derivative: float
    commutator_point: float
    point_residual: float
    outward_residual_lower: float
    outward_residual_upper: float
    residual_radius: float
    graph_derivative_resolved: bool
    residual_bounds_resolved: bool
    impact: float
    cost_score: float
    support_column: FloatArray
    eligible: bool
    mandatory: bool
    message: str


@dataclass(frozen=True)
class APCMCheckpointDiagnostic:
    """The complete autonomous evidence used at one adaptation checkpoint."""

    time: float
    active_keys: tuple[MomentKey, ...]
    conditioned_rhs_keys: tuple[MomentKey, ...]
    diagnostic_dimension: int
    diagnostic_frontier_dimension: int
    diagnostic_completion: APCMExtensionResult
    candidates: tuple[APCMCandidateDiagnostic, ...]
    local_error_scale: float
    jacobian_quadrature_error: float
    success: bool
    message: str


def _raw_coordinate_scales(envelope: float) -> FloatArray:
    return np.concatenate(
        (
            np.ones(3),
            np.full(4, np.sqrt(envelope)),
            np.full(4, envelope),
            np.full(6, np.sqrt(envelope * (envelope + 1.0))),
            np.full(12, np.sqrt(envelope)),
        )
    )


def _commutator_functional(
    observable: MomentKey,
    time: float,
    model: FixedDictionaryProjectedAPCM,
) -> Mapping[MomentKey, float]:
    coefficients: dict[MomentKey, complex] = {}
    for hamiltonian_coefficient, hamiltonian_word in _hamiltonian_terms(
        float(time),
        model.parameters,
    ):
        for key, commutator_coefficient in _commutator(
            hamiltonian_word,
            observable,
        ).items():
            coefficients[key] = coefficients.get(key, 0.0j) + (
                1j * hamiltonian_coefficient * commutator_coefficient
            )
    result: dict[MomentKey, float] = {}
    for key, value in coefficients.items():
        if abs(value.imag) > 5e-11:
            raise FloatingPointError(
                "Hermitian commutator functional has a complex coefficient: "
                f"{observable}, {key}, {value}"
            )
        if abs(value.real) > 1e-15:
            result[key] = float(value.real)
    return result


def _completion_at_state(
    model: FixedDictionaryProjectedAPCM,
    state: FloatArray,
    warm: Mapping[MomentKey, float],
) -> APCMExtensionResult | None:
    raw, hidden = unpack_projected_apcm_state(
        state,
        active_keys=model.active_keys,
    )
    tolerance = 10.0 * model.geometry.settings.conic_tolerance
    if np.linalg.eigvalsh(uncentered_joint_moment_matrix(raw))[0] < -tolerance:
        return None
    lower = state_lower_moments(raw, hidden, model.active_keys)
    completion = model.extension.complete(lower, warm_frontier=warm)
    return completion if completion.success else None


def _graph_derivative_samples(
    model: FixedDictionaryProjectedAPCM,
    state: FloatArray,
    derivative: FloatArray,
    evaluation: ProjectedAPCMEvaluation,
    candidates: tuple[MomentKey, ...],
    settings: APCMAdaptationSettings,
) -> tuple[dict[MomentKey, tuple[float, float, float]], str]:
    envelope = model.extension.settings.phonon_envelope
    scales = np.concatenate(
        (_raw_coordinate_scales(envelope), model.extension.active_scales)
    )
    scaled_state = np.asarray(state, dtype=float) / scales
    scaled_derivative = np.asarray(derivative, dtype=float) / scales
    base_step = settings.derivative_step_scale * (
        max(1.0, float(np.linalg.norm(scaled_state)))
        / max(1.0, float(np.linalg.norm(scaled_derivative)))
    )
    warm = evaluation.targets.completion.frontier_moments
    base_values = {
        key: float(evaluation.targets.completion.frontier_moments[key])
        for key in candidates
    }
    samples: dict[MomentKey, list[float]] = {
        key: [] for key in candidates
    }
    mode = "central"
    for step in (base_step, 0.5 * base_step, 0.25 * base_step):
        plus = _completion_at_state(
            model,
            np.asarray(state, dtype=float) + step * derivative,
            warm,
        )
        minus = _completion_at_state(
            model,
            np.asarray(state, dtype=float) - step * derivative,
            warm,
        )
        if plus is not None and minus is not None:
            for key in candidates:
                samples[key].append(
                    (
                        plus.frontier_moments[key]
                        - minus.frontier_moments[key]
                    )
                    / (2.0 * step)
                )
            continue

        mode = "forward_retracted"
        try:
            retracted = model.contain_stage(
                np.asarray(state, dtype=float) + step * derivative,
                warm_frontier=warm,
                retained_metric=evaluation.projection.retained_metric,
                auxiliary_metric=evaluation.projection.auxiliary_metric,
            )
        except (ProjectedAPCMFailure, APCMProjectionError):
            return {}, "no feasible graph-differentiation path"
        forward_state = np.concatenate(
            (retracted.raw_coordinates, retracted.hidden_values)
        )
        realized_direction = (forward_state - state) / step
        direction_error = float(
            np.linalg.norm((realized_direction - derivative) / scales)
        )
        if direction_error > settings.derivative_refinement_tolerance * max(
            1.0,
            float(np.linalg.norm(scaled_derivative)),
        ):
            return {}, "retracted path does not approach the requested direction"
        for key in candidates:
            samples[key].append(
                (
                    retracted.completion.frontier_moments[key]
                    - base_values[key]
                )
                / step
            )
    return {
        key: tuple(float(value) for value in values)
        for key, values in samples.items()
    }, mode


def _fixed_frontier_jacobian(
    model: FixedDictionaryProjectedAPCM,
    time: float,
    state: FloatArray,
    completion: APCMExtensionResult,
    settings: APCMAdaptationSettings,
) -> FloatArray:
    values = np.asarray(state, dtype=float)
    envelope = model.extension.settings.phonon_envelope
    scales = np.concatenate(
        (_raw_coordinate_scales(envelope), model.extension.active_scales)
    )
    jacobian = np.empty((values.size, values.size), dtype=float)
    for column in range(values.size):
        step = settings.jacobian_step_scale * max(
            scales[column],
            abs(float(values[column])),
        )
        offset = np.zeros_like(values)
        offset[column] = step
        plus = model.unprojected_velocity_with_frontier(
            time,
            values + offset,
            completion.frontier_moments,
        )
        minus = model.unprojected_velocity_with_frontier(
            time,
            values - offset,
            completion.frontier_moments,
        )
        jacobian[:, column] = (plus - minus) / (2.0 * step)
    return jacobian


def _impact_responses(
    jacobian: FloatArray,
    sources: FloatArray,
    interval: float,
) -> tuple[FloatArray, FloatArray]:
    def quadrature(order: int) -> FloatArray:
        nodes, weights = leggauss(order)
        times = 0.5 * interval * (nodes + 1.0)
        result = np.zeros_like(sources, dtype=float)
        for weight, value in zip(weights, times, strict=True):
            result += (
                weight
                * (expm(jacobian * (interval - value)) @ sources)
                * value
            )
        return 0.5 * interval * result

    coarse = quadrature(16)
    fine = quadrature(32)
    return fine, np.linalg.norm(fine - coarse, axis=0)


def _source_column(
    model: FixedDictionaryProjectedAPCM,
    time: float,
    candidate: MomentKey,
) -> FloatArray:
    source = np.zeros(len(model.state_names), dtype=float)
    offset = len(RAW_MOMENT_COORDINATE_NAMES)
    for row, observable in enumerate(model.active_keys):
        source[offset + row] = _commutator_functional(
            observable,
            time,
            model,
        ).get(candidate, 0.0)
    return source


def _promoted_cost_proxy(
    model: FixedDictionaryProjectedAPCM,
    candidate: MomentKey,
) -> float:
    extension = model.extension
    current_words = tuple(extension.words)
    candidate_word = _OperatorKey(
        candidate.spin_up,
        candidate.spin_down,
        candidate.x_power,
        candidate.p_power,
    )
    generated: set[MomentKey] = set()
    for word in current_words:
        generated.update(_operator_product(word, candidate_word))
        generated.update(_operator_product(candidate_word, word))
    generated.update(_operator_product(candidate_word, candidate_word))
    new_lower = set(extension.lower_keys).union({candidate})
    new_frontier = set(extension.frontier_keys).union(generated).difference(
        new_lower
    )
    old_free = len(extension.frontier_keys)
    old_size = extension.dimension
    new_free = len(new_frontier)
    new_size = old_size + int(candidate_word not in current_words)

    def cost(free: int, size: int) -> float:
        return float(free * size**3 + free**2 * size**2 + free**3)

    old_cost = max(1.0, cost(old_free, old_size))
    return 1.0 + max(0.0, cost(new_free, new_size) - old_cost) / old_cost


class APCMClosureGraphAnalyzer:
    """Build the one-shell closure diagnostic behind APCM admission."""

    def __init__(
        self,
        settings: APCMAdaptationSettings | None = None,
    ) -> None:
        self.settings = (
            APCMAdaptationSettings() if settings is None else settings
        )

    def analyze(
        self,
        model: FixedDictionaryProjectedAPCM,
        time: float,
        state: FloatArray,
        evaluation: ProjectedAPCMEvaluation,
        *,
        local_step_error: float,
    ) -> APCMCheckpointDiagnostic:
        """Evaluate graph invariance, robust radius, and local impact."""

        candidates = tuple(model.extension.rhs_frontier_keys)
        if not candidates:
            return APCMCheckpointDiagnostic(
                time=float(time),
                active_keys=model.active_keys,
                conditioned_rhs_keys=(),
                diagnostic_dimension=model.extension.dimension,
                diagnostic_frontier_dimension=len(model.extension.frontier_keys),
                diagnostic_completion=evaluation.targets.completion,
                candidates=(),
                local_error_scale=max(
                    float(local_step_error),
                    10.0 * model.extension.settings.conic_tolerance,
                ),
                jacobian_quadrature_error=0.0,
                success=True,
                message="algebraic interval collapse",
            )

        diagnostic_active = tuple((*model.active_keys, *candidates))
        diagnostic_descendants = tuple(
            sorted(
                {
                    generated
                    for candidate in candidates
                    for _, hamiltonian_word in _hamiltonian_terms(
                        float(time),
                        model.parameters,
                    )
                    for generated in _commutator(
                        hamiltonian_word,
                        candidate,
                    )
                    if generated.degree > 0
                },
                key=lambda key: (
                    key.degree,
                    key.spin_up,
                    key.spin_down,
                    key.x_power,
                    key.p_power,
                ),
            )
        )
        diagnostic_extension = SymmetryReducedPositiveExtension(
            model.extension.settings,
            active_keys=diagnostic_active,
            additional_halfword_keys=diagnostic_descendants,
        )
        current = evaluation.targets.completion
        diagnostic_lower = {
            key: current.moment(key) for key in diagnostic_extension.lower_keys
        }
        diagnostic_warm = {
            key: (
                current.moment(key)
                if key in current.moments
                else 0.0
            )
            for key in diagnostic_extension.frontier_keys
        }
        diagnostic_completion = diagnostic_extension.complete(
            diagnostic_lower,
            warm_frontier=diagnostic_warm,
        )
        if not diagnostic_completion.success:
            return APCMCheckpointDiagnostic(
                time=float(time),
                active_keys=model.active_keys,
                conditioned_rhs_keys=candidates,
                diagnostic_dimension=diagnostic_extension.dimension,
                diagnostic_frontier_dimension=len(
                    diagnostic_extension.frontier_keys
                ),
                diagnostic_completion=diagnostic_completion,
                candidates=(),
                local_error_scale=float("inf"),
                jacobian_quadrature_error=float("inf"),
                success=False,
                message="nested_completion_failure: "
                + diagnostic_completion.message,
            )

        graph_samples, graph_mode = _graph_derivative_samples(
            model,
            state,
            evaluation.derivative,
            evaluation,
            candidates,
            self.settings,
        )
        jacobian = _fixed_frontier_jacobian(
            model,
            float(time),
            state,
            current,
            self.settings,
        )
        retained_metric = evaluation.projection.retained_metric
        metric_eigenvalues, metric_eigenvectors = np.linalg.eigh(
            retained_metric
        )
        metric_square_root = (
            metric_eigenvectors
            * np.sqrt(np.maximum(metric_eigenvalues, 0.0))
        ) @ metric_eigenvectors.T
        epsilon_local = max(
            float(local_step_error),
            10.0 * model.extension.settings.conic_tolerance,
        )
        source_matrix = np.column_stack(
            [
                _source_column(model, float(time), candidate)
                for candidate in candidates
            ]
        )
        impact_responses, quadrature_errors = _impact_responses(
            jacobian,
            source_matrix,
            self.settings.checkpoint_interval,
        )
        results: list[APCMCandidateDiagnostic] = []
        worst_quadrature_error = float(np.max(quadrature_errors))
        for candidate_index, candidate in enumerate(candidates):
            functional = _commutator_functional(candidate, float(time), model)
            unsupported = set(functional).difference(
                {MomentKey(IDENTITY, IDENTITY, 0, 0)}
                | set(diagnostic_extension.lower_keys)
                | set(diagnostic_extension.frontier_keys)
            )
            if unsupported:
                results.append(
                    APCMCandidateDiagnostic(
                        key=candidate,
                        graph_derivative=float("nan"),
                        commutator_point=float("nan"),
                        point_residual=float("nan"),
                        outward_residual_lower=float("-inf"),
                        outward_residual_upper=float("inf"),
                        residual_radius=float("inf"),
                        graph_derivative_resolved=False,
                        residual_bounds_resolved=False,
                        impact=float("inf"),
                        cost_score=float("inf"),
                        support_column=np.full(29, np.nan),
                        eligible=True,
                        mandatory=True,
                        message=f"residual shell misses {len(unsupported)} moments",
                    )
                )
                continue

            extrema: APCMLinearExtrema = (
                diagnostic_extension.linear_functional_extrema(
                    diagnostic_lower,
                    functional,
                )
            )
            commutator_point = sum(
                coefficient * diagnostic_completion.moment(key)
                for key, coefficient in functional.items()
            )
            samples = graph_samples.get(candidate)
            graph_resolved = samples is not None and len(samples) == 3
            graph_derivative = float(samples[-1]) if graph_resolved else float("nan")
            point_residual = (
                float(commutator_point - graph_derivative)
                if graph_resolved
                else float("nan")
            )
            if graph_resolved:
                scale = model.extension.frontier_scales[
                    model.extension.frontier_keys.index(candidate)
                ]
                stabilization_floor = (
                    10.0
                    * model.extension.settings.conic_tolerance
                    * scale
                    / self.settings.checkpoint_interval
                )
                graph_resolved = abs(samples[-1] - samples[-2]) <= (
                    self.settings.derivative_refinement_tolerance
                    * max(abs(point_residual), stabilization_floor)
                )

            bounds_resolved = bool(extrema.success and graph_resolved)
            impact_quadrature_resolved = bool(
                quadrature_errors[candidate_index] <= 1e-15
            )
            if bounds_resolved and impact_quadrature_resolved:
                lower = extrema.outward_lower_bound - graph_derivative
                upper = extrema.outward_upper_bound - graph_derivative
                radius = max(abs(lower), abs(upper))
                response = impact_responses[:, candidate_index]
                support = (
                    radius
                    / epsilon_local
                    * (metric_square_root @ response[:29])
                )
                impact = float(np.linalg.norm(support))
                cost_score = impact / np.sqrt(
                    _promoted_cost_proxy(model, candidate)
                )
                eligible = impact > self.settings.impact_threshold
                mandatory = False
                message = (
                    f"{graph_mode}; extrema={extrema.message}; "
                    f"gaps=({extrema.minimum_gap:.3e},{extrema.maximum_gap:.3e})"
                )
            else:
                lower = float("-inf")
                upper = float("inf")
                radius = float("inf")
                support = np.full(29, np.nan)
                impact = float("inf")
                cost_score = float("inf")
                eligible = True
                mandatory = True
                if not graph_resolved:
                    message = "graph derivative unresolved"
                elif not extrema.success:
                    message = "residual_bound_unresolved: " + extrema.message
                else:
                    message = (
                        "impact_quadrature_unresolved: error="
                        f"{quadrature_errors[candidate_index]:.3e}"
                    )
            results.append(
                APCMCandidateDiagnostic(
                    key=candidate,
                    graph_derivative=graph_derivative,
                    commutator_point=float(commutator_point),
                    point_residual=point_residual,
                    outward_residual_lower=float(lower),
                    outward_residual_upper=float(upper),
                    residual_radius=float(radius),
                    graph_derivative_resolved=graph_resolved,
                    residual_bounds_resolved=bounds_resolved,
                    impact=impact,
                    cost_score=float(cost_score),
                    support_column=np.asarray(support, dtype=float),
                    eligible=eligible,
                    mandatory=mandatory,
                    message=message,
                )
            )
        return APCMCheckpointDiagnostic(
            time=float(time),
            active_keys=model.active_keys,
            conditioned_rhs_keys=candidates,
            diagnostic_dimension=diagnostic_extension.dimension,
            diagnostic_frontier_dimension=len(diagnostic_extension.frontier_keys),
            diagnostic_completion=diagnostic_completion,
            candidates=tuple(results),
            local_error_scale=epsilon_local,
            jacobian_quadrature_error=worst_quadrature_error,
            success=True,
            message="completed",
        )


__all__ = [
    "APCMAdaptationSettings",
    "APCMCandidateDiagnostic",
    "APCMCheckpointDiagnostic",
    "APCMClosureGraphAnalyzer",
]
