"""Offline falsification gates for the Hilbert--Schmidt Krylov closure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicHermiteSpline

from .exact_reference import (
    ExactDiagnosticWavefunctionTrajectory,
    exact_holstein_wavefunction_trajectory_for_diagnostics,
)
from .hubbard_dimer import DimerParameters
from .krylov_memory_closure import (
    KrylovClosureCoefficients,
    KrylovClosureConstruction,
    build_krylov_closure_construction,
    centered_jacobian_from_orthonormal,
    orthonormal_to_closed_coordinates,
)
from .matrix_reference import closed_scalar_rhs

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

DERIVATIVE_BLOCKS: tuple[tuple[str, slice], ...] = (
    ("rho", slice(0, 3)),
    ("B", slice(3, 7)),
    ("N", slice(7, 11)),
    ("A", slice(11, 17)),
    ("C", slice(17, 31)),
)


@dataclass(frozen=True)
class TeacherForcedOrderResult:
    """One fixed Krylov order evaluated along the exact retained path."""

    order: int
    coefficients: KrylovClosureCoefficients
    auxiliary_coordinates: FloatArray
    modeled_derivatives: FloatArray
    modeled_missing_c_source: FloatArray
    static_residual_norms: FloatArray
    drive_residual_norms: FloatArray
    total_residual_norms: FloatArray
    metrics: dict[str, object]


@dataclass(frozen=True)
class TeacherForcedGateResult:
    """Exact contractions and all requested teacher-forced order results."""

    phonon_cutoff: int
    times: FloatArray
    exact_orthonormal_coordinates: FloatArray
    exact_orthonormal_derivatives: FloatArray
    exact_closed_coordinates: FloatArray
    exact_closed_derivatives: FloatArray
    archive_derivatives: FloatArray
    archive_metrics: dict[str, object]
    construction: KrylovClosureConstruction
    orders: dict[int, TeacherForcedOrderResult]
    order_3_to_4_source_difference: float | None
    exact_function_evaluations: int


def _operator_expectations(
    states: ComplexArray,
    operators: tuple,
) -> FloatArray:
    sample_count = states.shape[1]
    values = np.empty((sample_count, len(operators)), dtype=float)
    for sample_index in range(sample_count):
        state = states[:, sample_index]
        values[sample_index] = [
            np.vdot(state, operator @ state).real for operator in operators
        ]
    return values


def _operator_expectation_derivatives(
    states: ComplexArray,
    state_derivatives: ComplexArray,
    operators: tuple,
) -> FloatArray:
    sample_count = states.shape[1]
    values = np.empty((sample_count, len(operators)), dtype=float)
    for sample_index in range(sample_count):
        state = states[:, sample_index]
        state_derivative = state_derivatives[:, sample_index]
        values[sample_index] = [
            (
                np.vdot(state_derivative, operator @ state)
                + np.vdot(state, operator @ state_derivative)
            ).real
            for operator in operators
        ]
    return values


def _teacher_forced_rk4(
    parameters: DimerParameters,
    coefficients: KrylovClosureCoefficients,
    times: FloatArray,
    retained_coordinates: FloatArray,
    retained_derivatives: FloatArray,
    initial_auxiliary: FloatArray,
) -> FloatArray:
    spline = CubicHermiteSpline(
        times,
        retained_coordinates,
        retained_derivatives,
        axis=0,
    )
    auxiliary = np.empty(
        (times.size, coefficients.auxiliary_dimension),
        dtype=float,
    )
    auxiliary[0] = initial_auxiliary

    def rhs(time: float, state: FloatArray) -> FloatArray:
        retained = np.asarray(spline(time), dtype=float)
        _, velocity = coefficients.orthonormal_velocity(
            retained,
            state,
            drive_value=parameters.drive_difference(time),
        )
        return velocity

    for index in range(times.size - 1):
        initial_time = float(times[index])
        step = float(times[index + 1] - times[index])
        state = auxiliary[index]
        k1 = rhs(initial_time, state)
        k2 = rhs(initial_time + 0.5 * step, state + 0.5 * step * k1)
        k3 = rhs(initial_time + 0.5 * step, state + 0.5 * step * k2)
        k4 = rhs(initial_time + step, state + step * k3)
        auxiliary[index + 1] = state + (step / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
    return auxiliary


def _rms_vector_norm(values: FloatArray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.asarray(values) ** 2, axis=1))))


def _maximum_vector_norm(values: FloatArray) -> float:
    return float(np.max(np.linalg.norm(np.asarray(values), axis=1)))


def _derivative_error_metrics(
    modeled: FloatArray,
    exact: FloatArray,
) -> dict[str, object]:
    error = modeled - exact
    residual_subtracted = error - error[0]
    return {
        "raw_rms_l2": _rms_vector_norm(error),
        "raw_max_l2": _maximum_vector_norm(error),
        "residual_subtracted_rms_l2": _rms_vector_norm(residual_subtracted),
        "residual_subtracted_max_l2": _maximum_vector_norm(
            residual_subtracted
        ),
        "block_raw_rms_l2": {
            name: _rms_vector_norm(error[:, block_slice])
            for name, block_slice in DERIVATIVE_BLOCKS
        },
        "block_residual_subtracted_rms_l2": {
            name: _rms_vector_norm(residual_subtracted[:, block_slice])
            for name, block_slice in DERIVATIVE_BLOCKS
        },
    }


def _source_metrics(
    modeled_source: FloatArray,
    exact_source: FloatArray,
) -> dict[str, float]:
    modeled_norms = np.linalg.norm(modeled_source, axis=1)
    exact_norms = np.linalg.norm(exact_source, axis=1)
    threshold = 1e-12 * float(np.max(exact_norms))
    active = (modeled_norms > threshold) & (exact_norms > threshold)
    if np.any(active):
        cosine = float(
            np.mean(
                np.sum(
                    modeled_source[active] * exact_source[active],
                    axis=1,
                )
                / (modeled_norms[active] * exact_norms[active])
            )
        )
    else:
        cosine = float("nan")
    norm_ratio = _rms_vector_norm(modeled_source) / max(
        _rms_vector_norm(exact_source),
        1e-14,
    )
    return {
        "mean_active_cosine": cosine,
        "rms_norm_ratio": float(norm_ratio),
        "active_sample_fraction": float(np.mean(active)),
    }


def _terminal_relative_rms(
    coefficients: KrylovClosureCoefficients,
    retained_coordinates: FloatArray,
    auxiliary_coordinates: FloatArray,
    exact_terminal_derivatives: FloatArray,
    parameters: DimerParameters,
    times: FloatArray,
) -> float:
    shell_offset = sum(coefficients.shell_dimensions[:-1])
    modeled = np.empty_like(exact_terminal_derivatives)
    for index, time in enumerate(times):
        _, auxiliary_velocity = coefficients.orthonormal_velocity(
            retained_coordinates[index],
            auxiliary_coordinates[index],
            drive_value=parameters.drive_difference(float(time)),
        )
        modeled[index] = auxiliary_velocity[shell_offset:]
    numerator = float(np.sum((modeled - exact_terminal_derivatives) ** 2))
    denominator = float(np.sum(exact_terminal_derivatives**2))
    return float(np.sqrt(numerator / max(denominator, 1e-30)))


def teacher_forced_krylov_gate(
    parameters: DimerParameters,
    *,
    phonon_cutoff: int,
    final_time: float = 4.0,
    sample_step: float = 0.01,
    orders: tuple[int, ...] = (2, 3, 4),
    rank_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-11,
    absolute_tolerance: float = 1e-13,
    maximum_step: float = 0.01,
    construction: KrylovClosureConstruction | None = None,
    exact_trajectory: ExactDiagnosticWavefunctionTrajectory | None = None,
) -> TeacherForcedGateResult:
    """Run Gate B without exposing exact data to an online closure RHS."""

    if final_time <= 0.0 or sample_step <= 0.0:
        raise ValueError("final_time and sample_step must be positive")
    step_count = int(round(final_time / sample_step))
    if abs(step_count * sample_step - final_time) > 1e-12:
        raise ValueError("final_time must be an integer multiple of sample_step")
    if not orders or min(orders) <= 0:
        raise ValueError("orders must contain positive integers")
    times = np.linspace(0.0, final_time, step_count + 1)
    required_shells = max(orders) + 1
    if construction is None:
        construction = build_krylov_closure_construction(
            parameters,
            phonon_cutoff=phonon_cutoff,
            shell_count=required_shells,
            rank_tolerance=rank_tolerance,
        )
    if len(construction.shells) < max(orders):
        raise RuntimeError("the Krylov construction deflated before a requested order")
    if exact_trajectory is None:
        exact_trajectory = exact_holstein_wavefunction_trajectory_for_diagnostics(
            parameters,
            sample_times=times,
            phonon_cutoff=phonon_cutoff,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            maximum_step=maximum_step,
        )
    if not np.allclose(exact_trajectory.times, times, atol=1e-14, rtol=0.0):
        raise ValueError("exact trajectory does not use the requested sample grid")

    raw_basis = construction.raw_basis
    retained_operators = raw_basis.orthonormal_observables
    exact_retained = _operator_expectations(
        exact_trajectory.state_vectors,
        retained_operators,
    )
    exact_retained_derivatives = _operator_expectation_derivatives(
        exact_trajectory.state_vectors,
        exact_trajectory.state_derivatives,
        retained_operators,
    )
    exact_closed = np.asarray(
        [
            orthonormal_to_closed_coordinates(raw_basis, coordinates)
            for coordinates in exact_retained
        ]
    )
    exact_closed_derivatives = np.asarray(
        [
            centered_jacobian_from_orthonormal(raw_basis, coordinates)
            @ derivative
            for coordinates, derivative in zip(
                exact_retained,
                exact_retained_derivatives,
                strict=True,
            )
        ]
    )
    archive_derivatives = np.asarray(
        [
            closed_scalar_rhs(float(time), state, parameters)
            for time, state in zip(times, exact_closed, strict=True)
        ]
    )
    archive_metrics = _derivative_error_metrics(
        archive_derivatives,
        exact_closed_derivatives,
    )
    exact_missing_c_source = (
        exact_closed_derivatives[:, 17:] - archive_derivatives[:, 17:]
    )

    order_results: dict[int, TeacherForcedOrderResult] = {}
    for order in orders:
        coefficients = construction.coefficients(order)
        initial_auxiliary = coefficients.contract_auxiliary_state(
            exact_trajectory.state_vectors[:, 0]
        )
        auxiliary = _teacher_forced_rk4(
            parameters,
            coefficients,
            times,
            exact_retained,
            exact_retained_derivatives,
            initial_auxiliary,
        )
        modeled_derivatives = np.empty_like(exact_closed_derivatives)
        static_residual = np.empty(times.size, dtype=float)
        drive_residual = np.empty(times.size, dtype=float)
        total_residual = np.empty(times.size, dtype=float)
        projected_action = np.empty(times.size, dtype=float)
        for index, time in enumerate(times):
            drive_value = parameters.drive_difference(float(time))
            retained_velocity, auxiliary_velocity = (
                coefficients.orthonormal_velocity(
                    exact_retained[index],
                    auxiliary[index],
                    drive_value=drive_value,
                )
            )
            modeled_derivatives[index] = (
                centered_jacobian_from_orthonormal(
                    raw_basis,
                    exact_retained[index],
                )
                @ retained_velocity
            )
            (
                static_residual[index],
                drive_residual[index],
                total_residual[index],
            ) = coefficients.residual_norms(
                auxiliary[index],
                drive_value=drive_value,
            )
            projected_action[index] = np.sqrt(
                np.dot(retained_velocity, retained_velocity)
                + np.dot(auxiliary_velocity, auxiliary_velocity)
                + total_residual[index] ** 2
            )

        modeled_missing_source = (
            modeled_derivatives[:, 17:] - archive_derivatives[:, 17:]
        )
        derivative_metrics = _derivative_error_metrics(
            modeled_derivatives,
            exact_closed_derivatives,
        )
        source_metrics = _source_metrics(
            modeled_missing_source,
            exact_missing_c_source,
        )
        shell_start = sum(coefficients.shell_dimensions[:-1])
        terminal_operators = coefficients.auxiliary_observables[shell_start:]
        exact_terminal_derivatives = _operator_expectation_derivatives(
            exact_trajectory.state_vectors,
            exact_trajectory.state_derivatives,
            terminal_operators,
        )
        terminal_relative_rms = _terminal_relative_rms(
            coefficients,
            exact_retained,
            auxiliary,
            exact_terminal_derivatives,
            parameters,
            times,
        )
        integrated_residual = float(np.trapezoid(total_residual, times))
        integrated_action = float(np.trapezoid(projected_action, times))
        integrated_static = float(np.trapezoid(static_residual, times))
        integrated_drive = float(np.trapezoid(drive_residual, times))
        metrics: dict[str, object] = {
            **derivative_metrics,
            "source": source_metrics,
            "terminal_relative_rms": terminal_relative_rms,
            "integrated_residual_ratio_eta": integrated_residual
            / max(integrated_action, 1e-14),
            "drive_residual_fraction_phi": integrated_drive
            / max(integrated_static + integrated_drive, 1e-14),
            "maximum_total_residual_norm": float(np.max(total_residual)),
        }
        order_results[order] = TeacherForcedOrderResult(
            order=order,
            coefficients=coefficients,
            auxiliary_coordinates=auxiliary,
            modeled_derivatives=modeled_derivatives,
            modeled_missing_c_source=modeled_missing_source,
            static_residual_norms=static_residual,
            drive_residual_norms=drive_residual,
            total_residual_norms=total_residual,
            metrics=metrics,
        )

    order_difference: float | None = None
    if 3 in order_results and 4 in order_results:
        difference = (
            order_results[3].modeled_missing_c_source
            - order_results[4].modeled_missing_c_source
        )
        order_difference = _rms_vector_norm(difference) / (
            _rms_vector_norm(order_results[4].modeled_missing_c_source) + 1e-12
        )

    return TeacherForcedGateResult(
        phonon_cutoff=phonon_cutoff,
        times=times,
        exact_orthonormal_coordinates=exact_retained,
        exact_orthonormal_derivatives=exact_retained_derivatives,
        exact_closed_coordinates=exact_closed,
        exact_closed_derivatives=exact_closed_derivatives,
        archive_derivatives=archive_derivatives,
        archive_metrics=archive_metrics,
        construction=construction,
        orders=order_results,
        order_3_to_4_source_difference=order_difference,
        exact_function_evaluations=exact_trajectory.function_evaluations,
    )
