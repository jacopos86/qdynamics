"""Goal-oriented numerical certificates for PURG trajectories.

This module implements the preparation-conditioned primal correction proposed
after the original global Gate-B certificate failed.  It deliberately keeps
the deployable :class:`~paper5.stability.purg.PurgReducedModel` unchanged.
Everything here is offline certification state.

The current implementation is a *numerical a posteriori estimate*: residual
integrals are adaptively evaluated in floating point.  It must not be labeled
as a formal outward-rounded certificate until validated quadrature and
validated floating-point linear algebra are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.sparse import csc_matrix

from .hubbard_dimer import DimerParameters
from .krylov_memory_closure import raw_to_closed_jacobian
from .purg import (
    PurgOperatorBounds,
    PurgProjection,
    _CENTERED_BLOCKS,
    _CENTERING_HESSIAN,
    _adaptive_gk15,
    _time_grid,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
SparseOperator = csc_matrix


@dataclass(frozen=True)
class PurgCorrectionTrajectory:
    """Primal path and reduced approximation to its full-state error."""

    times: FloatArray
    primal_states: ComplexArray
    correction_states: ComplexArray
    unresolved_error_bound: FloatArray
    correction_residuals: ComplexArray
    midpoint_correction_residuals: ComplexArray
    correction_residual_norms: FloatArray
    initial_unresolved_error: float
    cumulative_correction_residual: FloatArray
    quadrature_error_estimate: float
    maximum_residual_identity_error: float


@dataclass(frozen=True)
class HermitianGoalInterval:
    """Numerical interval for one Hermitian expectation-value error."""

    lower: float
    upper: float
    center: float
    radius: float
    spectral_center: float
    spectral_half_width: float


@dataclass(frozen=True)
class PurgForwardRemainderTrajectory:
    """Forward reduced representation of the unresolved state remainder."""

    times: FloatArray
    states: ComplexArray
    numerical_residual_norms: FloatArray
    cumulative_numerical_residual: FloatArray
    quadrature_error_estimate: float


@dataclass(frozen=True)
class DwrGoalInterval:
    """Intersection of the cheap and reduced-adjoint numerical intervals."""

    lower: float
    upper: float
    cheap: HermitianGoalInterval
    dwr_center: float
    dwr_radius: float
    dual_terminal_defect: float


@dataclass(frozen=True)
class PurgDualEnvelope:
    """Numerical cumulative leakage envelope for one enriched dual space."""

    times: FloatArray
    cumulative_leakage: FloatArray
    static_leakage_norm: float
    drive_leakage_norm: float
    reduced_drive_norm: float
    quadrature_error_estimate: float


@dataclass(frozen=True)
class PurgCenteredDerivativeEstimate:
    """Numerical amended intervals for all 31 centered derivatives."""

    times: FloatArray
    lower: FloatArray
    upper: FloatArray
    centers: FloatArray
    absolute_bounds: FloatArray
    direct_goal_terminal_defects: FloatArray
    bilinear_radii: FloatArray
    block_metrics: dict[str, dict[str, float]]


@dataclass(frozen=True)
class PurgExplicitDualEnvelope:
    """Columnwise backward-adjoint residual envelope for one terminal goal."""

    times: FloatArray
    states: ComplexArray
    residual_norms: FloatArray
    backward_error_bound: FloatArray
    terminal_projection_defect: float
    quadrature_error_estimate: float


def _require_orthonormal_basis(
    basis: ComplexArray,
    *,
    row_dimension: int,
    name: str,
    tolerance: float = 1e-10,
) -> ComplexArray:
    array = np.asarray(basis, dtype=complex)
    if array.ndim != 2 or array.shape[0] != row_dimension:
        raise ValueError(
            f"{name} must have shape ({row_dimension}, r), got {array.shape}"
        )
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one column")
    residual = np.linalg.norm(
        array.conj().T @ array - np.eye(array.shape[1], dtype=complex)
    )
    if residual > tolerance:
        raise ValueError(f"{name} is not orthonormal: residual={residual:.3e}")
    return array


def projected_correction_residual(
    hamiltonian: ComplexArray | SparseOperator,
    source: ComplexArray,
    basis: ComplexArray,
    correction_state: ComplexArray,
    correction_velocity: ComplexArray,
) -> ComplexArray:
    """Return ``dot(e_hat) + i H e_hat + q`` in the full Hilbert space."""

    z_basis = np.asarray(basis, dtype=complex)
    coordinates = np.asarray(correction_state, dtype=complex)
    velocity = np.asarray(correction_velocity, dtype=complex)
    source_vector = np.asarray(source, dtype=complex)
    lifted = z_basis @ coordinates
    return z_basis @ velocity + 1j * (hamiltonian @ lifted) + source_vector


def projected_correction_velocity(
    hamiltonian: ComplexArray | SparseOperator,
    source: ComplexArray,
    basis: ComplexArray,
    correction_state: ComplexArray,
) -> ComplexArray:
    """Evaluate the reduced correction equation in coefficient coordinates."""

    z_basis = np.asarray(basis, dtype=complex)
    coordinates = np.asarray(correction_state, dtype=complex)
    reduced_hamiltonian = z_basis.conj().T @ (hamiltonian @ z_basis)
    return -1j * (reduced_hamiltonian @ coordinates) - z_basis.conj().T @ source


def correction_residual_projection_form(
    hamiltonian: ComplexArray | SparseOperator,
    source: ComplexArray,
    basis: ComplexArray,
    correction_state: ComplexArray,
) -> ComplexArray:
    """Return ``(I-P_Z)(q+i H e_hat)`` for the exact reduced equation."""

    z_basis = np.asarray(basis, dtype=complex)
    coordinates = np.asarray(correction_state, dtype=complex)
    vector = np.asarray(source, dtype=complex) + 1j * (
        hamiltonian @ (z_basis @ coordinates)
    )
    return vector - z_basis @ (z_basis.conj().T @ vector)


def _constant_source_exponential_step(
    eigenvalues: FloatArray,
    eigenvectors: ComplexArray,
    initial_state: ComplexArray,
    source: ComplexArray,
    elapsed: float,
) -> tuple[ComplexArray, ComplexArray]:
    """Evaluate an exponential-midpoint inhomogeneous continuous extension."""

    initial_coordinates = eigenvectors.conj().T @ initial_state
    source_coordinates = eigenvectors.conj().T @ source
    phases = np.exp(-1j * elapsed * eigenvalues)
    arguments = -1j * elapsed * eigenvalues
    coefficients = np.empty_like(arguments, dtype=complex)
    small = np.abs(arguments) <= 1e-12
    coefficients[small] = elapsed
    coefficients[~small] = np.expm1(arguments[~small]) / (
        -1j * eigenvalues[~small]
    )
    state = eigenvectors @ (
        phases * initial_coordinates + coefficients * source_coordinates
    )
    velocity = eigenvectors @ (
        -1j
        * eigenvalues
        * (phases * initial_coordinates + coefficients * source_coordinates)
        + source_coordinates
    )
    return state, velocity


def propagate_purg_error_correction(
    projection: PurgProjection,
    parameters: DimerParameters,
    certificate_basis: ComplexArray,
    *,
    final_time: float,
    step: float,
    quadrature_absolute_tolerance: float = 1e-10,
) -> PurgCorrectionTrajectory:
    """Propagate the PURG path and its reduced full-state error approximation.

    The primal uses the same exponential-midpoint continuous extension as the
    registered PURG construction.  The correction uses an exponential
    midpoint step with the midpoint source.  Its *actual* continuous-extension
    residual is integrated, so midpoint and source defects remain visible.
    """

    if quadrature_absolute_tolerance <= 0.0:
        raise ValueError("quadrature_absolute_tolerance must be positive")
    model = projection.model
    z_basis = _require_orthonormal_basis(
        certificate_basis,
        row_dimension=projection.full_dimension,
        name="certificate_basis",
    )
    times = _time_grid(final_time, step)
    primal_states = np.empty((times.size, model.dimension), dtype=complex)
    correction_states = np.empty((times.size, z_basis.shape[1]), dtype=complex)
    residual_vectors = np.empty(
        (times.size, projection.full_dimension),
        dtype=complex,
    )
    midpoint_residual_vectors = np.empty(
        (times.size - 1, projection.full_dimension),
        dtype=complex,
    )
    residual_norms = np.empty(times.size, dtype=float)
    cumulative = np.zeros(times.size, dtype=float)
    primal_states[0] = model.initial_state

    lifted_initial = projection.lift(primal_states[0])
    initial_error = projection.reference_initial_state - lifted_initial
    correction_states[0] = z_basis.conj().T @ initial_error
    initial_unresolved = float(
        np.linalg.norm(initial_error - z_basis @ correction_states[0])
    )
    quadrature_error = 0.0
    maximum_identity_error = 0.0
    local_tolerance = quadrature_absolute_tolerance / (times.size - 1)

    for index in range(times.size - 1):
        left = float(times[index])
        right = float(times[index + 1])
        midpoint = 0.5 * (left + right)
        midpoint_drive = parameters.drive_difference(midpoint)
        reduced_midpoint = model.hamiltonian(midpoint_drive)
        primal_eigenvalues, primal_eigenvectors = eigh(
            reduced_midpoint,
            check_finite=True,
        )
        primal_coordinates = primal_eigenvectors.conj().T @ primal_states[index]

        def primal_extension(time: float) -> tuple[ComplexArray, ComplexArray]:
            elapsed = time - left
            phases = np.exp(-1j * elapsed * primal_eigenvalues)
            state = primal_eigenvectors @ (phases * primal_coordinates)
            velocity = primal_eigenvectors @ (
                -1j * primal_eigenvalues * phases * primal_coordinates
            )
            return state, velocity

        def full_hamiltonian(time: float) -> SparseOperator:
            drive = parameters.drive_difference(time)
            return (
                projection.static_hamiltonian
                + drive * projection.drive_hamiltonian
            ).tocsc()

        def primal_defect(time: float) -> ComplexArray:
            state, velocity = primal_extension(time)
            lifted = projection.basis @ state
            return projection.basis @ velocity + 1j * (
                full_hamiltonian(time) @ lifted
            )

        midpoint_source = primal_defect(midpoint)
        full_midpoint = full_hamiltonian(midpoint)
        correction_hamiltonian = np.asarray(
            z_basis.conj().T @ (full_midpoint @ z_basis),
            dtype=complex,
        )
        correction_hamiltonian = 0.5 * (
            correction_hamiltonian + correction_hamiltonian.conj().T
        )
        correction_eigenvalues, correction_eigenvectors = eigh(
            correction_hamiltonian,
            check_finite=True,
        )
        midpoint_reduced_source = -z_basis.conj().T @ midpoint_source

        def correction_extension(
            time: float,
        ) -> tuple[ComplexArray, ComplexArray]:
            return _constant_source_exponential_step(
                correction_eigenvalues,
                correction_eigenvectors,
                correction_states[index],
                midpoint_reduced_source,
                time - left,
            )

        def correction_residual(time: float) -> ComplexArray:
            coordinates, velocity = correction_extension(time)
            source = primal_defect(time)
            hamiltonian = full_hamiltonian(time)
            residual = projected_correction_residual(
                hamiltonian,
                source,
                z_basis,
                coordinates,
                velocity,
            )

            projected_velocity = projected_correction_velocity(
                hamiltonian,
                source,
                z_basis,
                coordinates,
            )
            projected_form = correction_residual_projection_form(
                hamiltonian,
                source,
                z_basis,
                coordinates,
            )
            integrator_defect = z_basis @ (velocity - projected_velocity)
            identity_error = np.linalg.norm(
                residual - projected_form - integrator_defect
            )
            nonlocal maximum_identity_error
            maximum_identity_error = max(
                maximum_identity_error,
                float(identity_error),
            )
            return residual

        interval_value, interval_error = _adaptive_gk15(
            lambda time: float(np.linalg.norm(correction_residual(time))),
            left,
            right,
            absolute_tolerance=local_tolerance,
        )
        cumulative[index + 1] = cumulative[index] + interval_value
        quadrature_error += interval_error
        midpoint_residual_vectors[index] = correction_residual(midpoint)
        primal_states[index + 1] = primal_extension(right)[0]
        correction_states[index + 1] = correction_extension(right)[0]
        residual_vectors[index] = correction_residual(left)
        residual_norms[index] = float(np.linalg.norm(residual_vectors[index]))

    residual_vectors[-1] = correction_residual(float(times[-1]))
    residual_norms[-1] = float(np.linalg.norm(residual_vectors[-1]))
    return PurgCorrectionTrajectory(
        times=times,
        primal_states=primal_states,
        correction_states=correction_states,
        unresolved_error_bound=initial_unresolved + cumulative,
        correction_residuals=residual_vectors,
        midpoint_correction_residuals=midpoint_residual_vectors,
        correction_residual_norms=residual_norms,
        initial_unresolved_error=initial_unresolved,
        cumulative_correction_residual=cumulative,
        quadrature_error_estimate=quadrature_error,
        maximum_residual_identity_error=maximum_identity_error,
    )


def propagate_forward_remainder(
    projection: PurgProjection,
    parameters: DimerParameters,
    correction_basis: ComplexArray,
    correction: PurgCorrectionTrajectory,
    dual_basis: ComplexArray,
    *,
    quadrature_absolute_tolerance: float = 1e-10,
) -> PurgForwardRemainderTrajectory:
    """Propagate the amended forward reciprocity state in an enriched space.

    The equation is ``dot(xi)=-i H_Y xi-Y^dagger s``.  The source is the
    actual midpoint residual of the implemented primal-correction extension.
    Numerical defects of this second midpoint extension are retained
    separately from the mathematical unresolved-state radius.
    """

    if quadrature_absolute_tolerance <= 0.0:
        raise ValueError("quadrature_absolute_tolerance must be positive")
    z_basis = _require_orthonormal_basis(
        correction_basis,
        row_dimension=projection.full_dimension,
        name="correction_basis",
    )
    y_basis = _require_orthonormal_basis(
        dual_basis,
        row_dimension=projection.full_dimension,
        name="dual_basis",
    )
    containment = np.linalg.norm(
        z_basis - y_basis @ (y_basis.conj().T @ z_basis)
    )
    if containment > 1e-10:
        raise ValueError(
            "dual_basis must contain correction_basis: "
            f"residual={containment:.3e}"
        )
    times = np.asarray(correction.times, dtype=float)
    if times.size < 2:
        raise ValueError("correction trajectory must contain at least two times")
    steps = np.diff(times)
    if not np.allclose(steps, steps[0], atol=1e-13, rtol=0.0):
        raise ValueError("correction trajectory must use a uniform grid")
    if correction.midpoint_correction_residuals.shape != (
        times.size - 1,
        projection.full_dimension,
    ):
        raise ValueError("midpoint correction residual shape is inconsistent")

    represented_initial = projection.lift(correction.primal_states[0])
    full_error_initial = projection.reference_initial_state - represented_initial
    correction_initial = z_basis @ correction.correction_states[0]
    unresolved_initial = full_error_initial - correction_initial
    states = np.empty((times.size, y_basis.shape[1]), dtype=complex)
    states[0] = y_basis.conj().T @ unresolved_initial
    numerical_residual_norms = np.empty(times.size, dtype=float)
    cumulative = np.zeros(times.size, dtype=float)
    quadrature_error = 0.0
    local_tolerance = quadrature_absolute_tolerance / (times.size - 1)

    for index in range(times.size - 1):
        left = float(times[index])
        right = float(times[index + 1])
        midpoint = 0.5 * (left + right)
        midpoint_drive = parameters.drive_difference(midpoint)
        full_midpoint = (
            projection.static_hamiltonian
            + midpoint_drive * projection.drive_hamiltonian
        ).tocsc()
        reduced_midpoint = np.asarray(
            y_basis.conj().T @ (full_midpoint @ y_basis),
            dtype=complex,
        )
        reduced_midpoint = 0.5 * (
            reduced_midpoint + reduced_midpoint.conj().T
        )
        eigenvalues, eigenvectors = eigh(reduced_midpoint, check_finite=True)
        midpoint_source = -y_basis.conj().T @ (
            correction.midpoint_correction_residuals[index]
        )

        def extension(time: float) -> tuple[ComplexArray, ComplexArray]:
            return _constant_source_exponential_step(
                eigenvalues,
                eigenvectors,
                states[index],
                midpoint_source,
                time - left,
            )

        def interpolated_source(time: float) -> ComplexArray:
            fraction = (time - left) / (right - left)
            left_weight = 2.0 * (fraction - 0.5) * (fraction - 1.0)
            midpoint_weight = 4.0 * fraction * (1.0 - fraction)
            right_weight = 2.0 * fraction * (fraction - 0.5)
            return (
                left_weight * correction.correction_residuals[index]
                + midpoint_weight
                * correction.midpoint_correction_residuals[index]
                + right_weight * correction.correction_residuals[index + 1]
            )

        def numerical_residual(time: float) -> ComplexArray:
            state, velocity = extension(time)
            drive = parameters.drive_difference(time)
            hamiltonian = (
                projection.static_hamiltonian
                + drive * projection.drive_hamiltonian
            ).tocsc()
            return (
                velocity
                + 1j * (y_basis.conj().T @ (hamiltonian @ (y_basis @ state)))
                + y_basis.conj().T @ interpolated_source(time)
            )

        value, error = _adaptive_gk15(
            lambda time: float(np.linalg.norm(numerical_residual(time))),
            left,
            right,
            absolute_tolerance=local_tolerance,
        )
        cumulative[index + 1] = cumulative[index] + value
        quadrature_error += error
        states[index + 1] = extension(right)[0]
        numerical_residual_norms[index] = float(
            np.linalg.norm(numerical_residual(left))
        )

    numerical_residual_norms[-1] = float(
        np.linalg.norm(numerical_residual(float(times[-1])))
    )
    return PurgForwardRemainderTrajectory(
        times=times,
        states=states,
        numerical_residual_norms=numerical_residual_norms,
        cumulative_numerical_residual=cumulative,
        quadrature_error_estimate=quadrature_error,
    )


def build_dual_leakage_envelope(
    projection: PurgProjection,
    parameters: DimerParameters,
    dual_basis: ComplexArray,
    times: FloatArray,
    *,
    quadrature_absolute_tolerance: float = 1e-10,
) -> PurgDualEnvelope:
    """Build a floating-point upper envelope for reduced-adjoint leakage."""

    if quadrature_absolute_tolerance <= 0.0:
        raise ValueError("quadrature_absolute_tolerance must be positive")
    y_basis = _require_orthonormal_basis(
        dual_basis,
        row_dimension=projection.full_dimension,
        name="dual_basis",
    )
    grid = np.asarray(times, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("times must be a strictly increasing one-dimensional grid")
    static_reduced = np.asarray(
        y_basis.conj().T @ (projection.static_hamiltonian @ y_basis),
        dtype=complex,
    )
    drive_reduced = np.asarray(
        y_basis.conj().T @ (projection.drive_hamiltonian @ y_basis),
        dtype=complex,
    )
    static_residual = (
        projection.static_hamiltonian @ y_basis - y_basis @ static_reduced
    )
    drive_residual = (
        projection.drive_hamiltonian @ y_basis - y_basis @ drive_reduced
    )
    static_norm = float(np.linalg.norm(static_residual, ord=2))
    drive_norm = float(np.linalg.norm(drive_residual, ord=2))
    reduced_drive_norm = float(np.linalg.norm(drive_reduced, ord=2))
    cumulative = np.zeros(grid.size, dtype=float)
    quadrature_error = 0.0
    local_tolerance = quadrature_absolute_tolerance / (grid.size - 1)
    for index in range(grid.size - 1):
        left = float(grid[index])
        right = float(grid[index + 1])
        midpoint = 0.5 * (left + right)
        midpoint_drive = parameters.drive_difference(midpoint)

        def density(time: float) -> float:
            drive = parameters.drive_difference(time)
            projection_leakage = static_norm + abs(drive) * drive_norm
            midpoint_defect = (
                abs(drive - midpoint_drive) * reduced_drive_norm
            )
            return projection_leakage + midpoint_defect

        value, error = _adaptive_gk15(
            density,
            left,
            right,
            absolute_tolerance=local_tolerance,
        )
        cumulative[index + 1] = cumulative[index] + value
        quadrature_error += error
    return PurgDualEnvelope(
        times=grid,
        cumulative_leakage=cumulative,
        static_leakage_norm=static_norm,
        drive_leakage_norm=drive_norm,
        reduced_drive_norm=reduced_drive_norm,
        quadrature_error_estimate=quadrature_error,
    )


def propagate_explicit_reduced_dual(
    projection: PurgProjection,
    parameters: DimerParameters,
    dual_basis: ComplexArray,
    goal_action: ComplexArray,
    times: FloatArray,
    *,
    terminal_index: int,
    quadrature_absolute_tolerance: float = 1e-10,
) -> PurgExplicitDualEnvelope:
    """Propagate one threatened reduced adjoint and integrate its full defect."""

    if quadrature_absolute_tolerance <= 0.0:
        raise ValueError("quadrature_absolute_tolerance must be positive")
    y_basis = _require_orthonormal_basis(
        dual_basis,
        row_dimension=projection.full_dimension,
        name="dual_basis",
    )
    grid = np.asarray(times, dtype=float)
    if terminal_index < 0 or terminal_index >= grid.size:
        raise IndexError("terminal_index is outside the time grid")
    action = np.asarray(goal_action, dtype=complex)
    if action.shape != (projection.full_dimension,):
        raise ValueError("goal_action has the wrong shape")
    terminal = y_basis.conj().T @ action
    terminal_defect = float(np.linalg.norm(action - y_basis @ terminal))
    states = np.empty((terminal_index + 1, y_basis.shape[1]), dtype=complex)
    residual_norms = np.empty(terminal_index + 1, dtype=float)
    backward_bound = np.empty(terminal_index + 1, dtype=float)
    states[terminal_index] = terminal
    backward_bound[terminal_index] = terminal_defect
    quadrature_error = 0.0
    local_tolerance = quadrature_absolute_tolerance / max(1, terminal_index)

    for index in range(terminal_index - 1, -1, -1):
        left = float(grid[index])
        right = float(grid[index + 1])
        midpoint = 0.5 * (left + right)
        midpoint_drive = parameters.drive_difference(midpoint)
        full_midpoint = (
            projection.static_hamiltonian
            + midpoint_drive * projection.drive_hamiltonian
        ).tocsc()
        reduced_midpoint = np.asarray(
            y_basis.conj().T @ (full_midpoint @ y_basis),
            dtype=complex,
        )
        reduced_midpoint = 0.5 * (
            reduced_midpoint + reduced_midpoint.conj().T
        )
        eigenvalues, eigenvectors = eigh(reduced_midpoint, check_finite=True)
        terminal_coordinates = eigenvectors.conj().T @ states[index + 1]

        def extension(time: float) -> tuple[ComplexArray, ComplexArray]:
            elapsed_from_right = time - right
            phases = np.exp(-1j * elapsed_from_right * eigenvalues)
            state = eigenvectors @ (phases * terminal_coordinates)
            velocity = eigenvectors @ (
                -1j * eigenvalues * phases * terminal_coordinates
            )
            return state, velocity

        def full_residual(time: float) -> ComplexArray:
            state, velocity = extension(time)
            drive = parameters.drive_difference(time)
            hamiltonian = (
                projection.static_hamiltonian
                + drive * projection.drive_hamiltonian
            ).tocsc()
            lifted = y_basis @ state
            return y_basis @ velocity + 1j * (hamiltonian @ lifted)

        value, error = _adaptive_gk15(
            lambda time: float(np.linalg.norm(full_residual(time))),
            left,
            right,
            absolute_tolerance=local_tolerance,
        )
        states[index] = extension(left)[0]
        backward_bound[index] = backward_bound[index + 1] + value
        quadrature_error += error
        residual_norms[index] = float(np.linalg.norm(full_residual(left)))

    if terminal_index == 0:
        drive = parameters.drive_difference(float(grid[0]))
        hamiltonian = (
            projection.static_hamiltonian
            + drive * projection.drive_hamiltonian
        ).tocsc()
        reduced = y_basis.conj().T @ (hamiltonian @ y_basis)
        velocity = -1j * (reduced @ states[0])
        residual_norms[0] = float(
            np.linalg.norm(
                y_basis @ velocity + 1j * (hamiltonian @ (y_basis @ states[0]))
            )
        )
    else:
        residual_norms[terminal_index] = residual_norms[terminal_index - 1]
    return PurgExplicitDualEnvelope(
        times=grid[: terminal_index + 1].copy(),
        states=states,
        residual_norms=residual_norms,
        backward_error_bound=backward_bound,
        terminal_projection_defect=terminal_defect,
        quadrature_error_estimate=quadrature_error,
    )


def explicit_dual_remainder_radius(
    explicit_dual: PurgExplicitDualEnvelope,
    correction: PurgCorrectionTrajectory,
    forward_remainder: PurgForwardRemainderTrajectory,
) -> float:
    """Return the numerical DWR remainder using a columnwise dual residual."""

    count = explicit_dual.times.size
    if not np.array_equal(explicit_dual.times, correction.times[:count]):
        raise ValueError("explicit dual and correction grids differ")
    beta = explicit_dual.backward_error_bound
    residual = correction.correction_residual_norms[:count]
    if count == 1:
        pairing = 0.0
    else:
        pairing = float(
            2.0 * np.trapezoid(beta * residual, explicit_dual.times)
        )
    initial = float(2.0 * beta[0] * correction.initial_unresolved_error)
    projected_norm = float(np.linalg.norm(explicit_dual.states[-1]))
    forward_numerical = float(
        2.0
        * projected_norm
        * forward_remainder.cumulative_numerical_residual[count - 1]
    )
    return initial + pairing + forward_numerical


def numerical_dual_remainder_radius(
    *,
    goal_action: ComplexArray,
    terminal_index: int,
    dual_basis: ComplexArray,
    correction: PurgCorrectionTrajectory,
    forward_remainder: PurgForwardRemainderTrajectory,
    envelope: PurgDualEnvelope,
) -> tuple[float, float]:
    """Estimate the DWR linear remainder from the dual leakage envelope."""

    y_basis = np.asarray(dual_basis, dtype=complex)
    action = np.asarray(goal_action, dtype=complex)
    if terminal_index < 0 or terminal_index >= correction.times.size:
        raise IndexError("terminal_index is outside the trajectory")
    if not np.array_equal(correction.times, forward_remainder.times) or not np.array_equal(
        correction.times,
        envelope.times,
    ):
        raise ValueError("correction, forward remainder, and envelope grids differ")
    projected = y_basis.conj().T @ action
    terminal_defect = float(np.linalg.norm(action - y_basis @ projected))
    projected_norm = float(np.linalg.norm(projected))
    cumulative = envelope.cumulative_leakage
    beta_z = terminal_defect + projected_norm * (
        cumulative[terminal_index] - cumulative[: terminal_index + 1]
    )
    times = correction.times[: terminal_index + 1]
    residual_norms = correction.correction_residual_norms[
        : terminal_index + 1
    ]
    if terminal_index == 0:
        pairing_remainder = 0.0
    else:
        pairing_remainder = float(
            2.0 * np.trapezoid(beta_z * residual_norms, times)
        )
    initial_remainder = float(
        2.0 * beta_z[0] * correction.initial_unresolved_error
    )
    forward_numerical = float(
        2.0
        * projected_norm
        * forward_remainder.cumulative_numerical_residual[terminal_index]
    )
    return (
        initial_remainder + pairing_remainder + forward_numerical,
        terminal_defect,
    )


def _intersect_goal_intervals(
    *,
    cheap_center: float,
    cheap_radius: float,
    dwr_center: float,
    dwr_radius: float,
    tolerance: float = 1e-10,
) -> tuple[float, float]:
    lower = max(cheap_center - cheap_radius, dwr_center - dwr_radius)
    upper = min(cheap_center + cheap_radius, dwr_center + dwr_radius)
    slack = tolerance * max(1.0, abs(lower), abs(upper))
    if lower > upper + slack:
        raise ValueError("cheap and DWR numerical intervals do not intersect")
    if lower > upper:
        midpoint = 0.5 * (lower + upper)
        return midpoint, midpoint
    return float(lower), float(upper)


def _numerical_goal_interval_from_actions(
    *,
    represented_state: ComplexArray,
    lifted_correction: ComplexArray,
    action_on_represented: ComplexArray,
    action_on_correction: ComplexArray,
    operator_norm_bound: float,
    unresolved_radius: float,
    terminal_index: int,
    dual_basis: ComplexArray,
    forward_remainder: PurgForwardRemainderTrajectory,
    correction: PurgCorrectionTrajectory,
    envelope: PurgDualEnvelope,
) -> tuple[float, float, float, float]:
    if operator_norm_bound < 0.0:
        raise ValueError("operator_norm_bound must be nonnegative")
    phi = np.asarray(represented_state, dtype=complex)
    lifted = np.asarray(lifted_correction, dtype=complex)
    g_phi = np.asarray(action_on_represented, dtype=complex)
    g_correction = np.asarray(action_on_correction, dtype=complex)
    goal_action = g_phi + g_correction
    correction_center = float(
        2.0 * np.vdot(g_phi, lifted).real
        + np.vdot(lifted, g_correction).real
    )
    cheap_radius = float(
        2.0 * np.linalg.norm(goal_action) * unresolved_radius
        + operator_norm_bound * unresolved_radius**2
    )
    dual_radius, terminal_defect = numerical_dual_remainder_radius(
        goal_action=goal_action,
        terminal_index=terminal_index,
        dual_basis=dual_basis,
        correction=correction,
        forward_remainder=forward_remainder,
        envelope=envelope,
    )
    projected_goal = dual_basis.conj().T @ goal_action
    linear_center = float(
        2.0
        * np.vdot(
            projected_goal,
            forward_remainder.states[terminal_index],
        ).real
    )
    dwr_center = correction_center + linear_center
    dwr_radius = float(
        dual_radius + operator_norm_bound * unresolved_radius**2
    )
    lower, upper = _intersect_goal_intervals(
        cheap_center=correction_center,
        cheap_radius=cheap_radius,
        dwr_center=dwr_center,
        dwr_radius=dwr_radius,
    )
    return lower, upper, dwr_center, terminal_defect


def _full_goal_actions(
    projection: PurgProjection,
    drive_value: float,
    vectors: ComplexArray,
) -> tuple[ComplexArray, ComplexArray]:
    """Apply all raw observables and physical derivative operators."""

    targets = np.asarray(vectors, dtype=complex)
    if targets.ndim == 1:
        targets = targets[:, None]
    if targets.ndim != 2 or targets.shape[0] != projection.full_dimension:
        raise ValueError("vectors have the wrong full-space shape")
    hamiltonian = (
        projection.static_hamiltonian
        + drive_value * projection.drive_hamiltonian
    ).tocsc()
    h_targets = hamiltonian @ targets
    observable_actions = np.empty(
        (
            len(projection.full_raw_observables),
            projection.full_dimension,
            targets.shape[1],
        ),
        dtype=complex,
    )
    derivative_actions = np.empty_like(observable_actions)
    for index, observable in enumerate(projection.full_raw_observables):
        action = observable @ targets
        observable_actions[index] = action
        derivative_actions[index] = 1j * (
            hamiltonian @ action - observable @ h_targets
        )
    return observable_actions, derivative_actions


def estimate_centered_derivative_intervals(
    projection: PurgProjection,
    parameters: DimerParameters,
    correction_basis: ComplexArray,
    correction: PurgCorrectionTrajectory,
    dual_basis: ComplexArray,
    forward_remainder: PurgForwardRemainderTrajectory,
    envelope: PurgDualEnvelope,
    operator_bounds: PurgOperatorBounds,
    *,
    sample_stride: int = 1,
) -> PurgCenteredDerivativeEstimate:
    """Evaluate equations (4.4)--(4.6) as a numerical a posteriori estimate."""

    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    z_basis = _require_orthonormal_basis(
        correction_basis,
        row_dimension=projection.full_dimension,
        name="correction_basis",
    )
    y_basis = _require_orthonormal_basis(
        dual_basis,
        row_dimension=projection.full_dimension,
        name="dual_basis",
    )
    indices = np.arange(0, correction.times.size, sample_stride, dtype=int)
    if indices[-1] != correction.times.size - 1:
        indices = np.append(indices, correction.times.size - 1)
    times = correction.times[indices]
    lower = np.empty((indices.size, 31), dtype=float)
    upper = np.empty_like(lower)
    centers = np.empty_like(lower)
    terminal_defects = np.empty_like(lower)
    bilinear_radii = np.empty_like(lower)

    for output_index, trajectory_index in enumerate(indices):
        time = float(correction.times[trajectory_index])
        drive = parameters.drive_difference(time)
        reduced_state = correction.primal_states[trajectory_index]
        represented = projection.lift(reduced_state)
        lifted = z_basis @ correction.correction_states[trajectory_index]
        observable_actions, derivative_actions = _full_goal_actions(
            projection,
            drive,
            np.column_stack((represented, lifted)),
        )
        f_phi = observable_actions[:, :, 0]
        f_correction = observable_actions[:, :, 1]
        k_phi = derivative_actions[:, :, 0]
        k_correction = derivative_actions[:, :, 1]
        raw = projection.model.raw_coordinates(reduced_state)
        modeled_velocity = projection.model.raw_velocity(
            reduced_state,
            drive_value=drive,
        )
        physical_velocity = np.asarray(
            [np.vdot(represented, action).real for action in k_phi],
            dtype=float,
        )
        gamma = physical_velocity - modeled_velocity
        jacobian = raw_to_closed_jacobian(raw)
        value_coefficients = np.einsum(
            "iab,b->ia",
            _CENTERING_HESSIAN,
            physical_velocity,
        )
        unresolved = float(
            correction.unresolved_error_bound[trajectory_index]
        )
        raw_value_bounds = np.empty(len(projection.full_raw_observables))
        raw_derivative_bounds = np.empty_like(raw_value_bounds)
        derivative_norms = (
            operator_bounds.static_derivative
            + abs(drive) * operator_bounds.drive_derivative
        )
        for raw_index in range(raw_value_bounds.size):
            value_interval = _numerical_goal_interval_from_actions(
                represented_state=represented,
                lifted_correction=lifted,
                action_on_represented=f_phi[raw_index],
                action_on_correction=f_correction[raw_index],
                operator_norm_bound=float(operator_bounds.raw[raw_index]),
                unresolved_radius=unresolved,
                terminal_index=int(trajectory_index),
                dual_basis=y_basis,
                forward_remainder=forward_remainder,
                correction=correction,
                envelope=envelope,
            )
            derivative_interval = _numerical_goal_interval_from_actions(
                represented_state=represented,
                lifted_correction=lifted,
                action_on_represented=k_phi[raw_index],
                action_on_correction=k_correction[raw_index],
                operator_norm_bound=float(derivative_norms[raw_index]),
                unresolved_radius=unresolved,
                terminal_index=int(trajectory_index),
                dual_basis=y_basis,
                forward_remainder=forward_remainder,
                correction=correction,
                envelope=envelope,
            )
            raw_value_bounds[raw_index] = max(
                abs(value_interval[0]),
                abs(value_interval[1]),
            )
            raw_derivative_bounds[raw_index] = max(
                abs(derivative_interval[0]),
                abs(derivative_interval[1]),
            )

        for component in range(31):
            derivative_coefficients = jacobian[component]
            value_coefficients_component = value_coefficients[component]
            direct_phi = (
                derivative_coefficients @ k_phi
                + value_coefficients_component @ f_phi
            )
            direct_correction = (
                derivative_coefficients @ k_correction
                + value_coefficients_component @ f_correction
            )
            direct_norm = float(
                np.abs(derivative_coefficients) @ derivative_norms
                + np.abs(value_coefficients_component) @ operator_bounds.raw
            )
            direct_interval = _numerical_goal_interval_from_actions(
                represented_state=represented,
                lifted_correction=lifted,
                action_on_represented=direct_phi,
                action_on_correction=direct_correction,
                operator_norm_bound=direct_norm,
                unresolved_radius=unresolved,
                terminal_index=int(trajectory_index),
                dual_basis=y_basis,
                forward_remainder=forward_remainder,
                correction=correction,
                envelope=envelope,
            )
            hessian = _CENTERING_HESSIAN[component]
            elementwise_radius = float(
                raw_value_bounds @ (np.abs(hessian) @ raw_derivative_bounds)
            )
            spectral_radius = float(
                np.linalg.norm(hessian, ord=2)
                * np.linalg.norm(raw_value_bounds)
                * np.linalg.norm(raw_derivative_bounds)
            )
            bilinear = min(elementwise_radius, spectral_radius)
            offset = float(jacobian[component] @ gamma)
            lower[output_index, component] = (
                offset + direct_interval[0] - bilinear
            )
            upper[output_index, component] = (
                offset + direct_interval[1] + bilinear
            )
            centers[output_index, component] = (
                offset + direct_interval[2]
            )
            terminal_defects[output_index, component] = direct_interval[3]
            bilinear_radii[output_index, component] = bilinear

    absolute = np.maximum(np.abs(lower), np.abs(upper))
    block_metrics: dict[str, dict[str, float]] = {}
    for name, block_slice in _CENTERED_BLOCKS.items():
        norms = np.linalg.norm(absolute[:, block_slice], axis=1)
        block_metrics[name] = {
            "rms_l2_bound": float(np.sqrt(np.mean(norms**2))),
            "max_l2_bound": float(np.max(norms)),
        }
    return PurgCenteredDerivativeEstimate(
        times=times,
        lower=lower,
        upper=upper,
        centers=centers,
        absolute_bounds=absolute,
        direct_goal_terminal_defects=terminal_defects,
        bilinear_radii=bilinear_radii,
        block_metrics=block_metrics,
    )


def gershgorin_hermitian_enclosure(
    operator: ComplexArray | SparseOperator,
) -> tuple[float, float]:
    """Return a conservative floating-point Gershgorin spectral enclosure."""

    if hasattr(operator, "tocsr"):
        matrix = operator.tocsr()
        diagonal = np.asarray(matrix.diagonal().real, dtype=float)
        row_sums = np.asarray(np.abs(matrix).sum(axis=1)).ravel()
        radii = row_sums - np.abs(matrix.diagonal())
    else:
        matrix = np.asarray(operator, dtype=complex)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("operator must be square")
        diagonal = np.diag(matrix).real
        radii = np.sum(np.abs(matrix), axis=1) - np.abs(np.diag(matrix))
    return float(np.min(diagonal - radii)), float(np.max(diagonal + radii))


def quadratic_goal_interval(
    *,
    represented_state: ComplexArray,
    lifted_correction: ComplexArray,
    unresolved_radius: float,
    operator: ComplexArray | SparseOperator,
    spectral_lower: float,
    spectral_upper: float,
) -> HermitianGoalInterval:
    """Evaluate the amended cheap interval for one Hermitian goal.

    The unknown exact state is assumed normalized.  A measured norm defect of
    the represented state is retained through the spectral-shift term.
    """

    if unresolved_radius < 0.0:
        raise ValueError("unresolved_radius must be nonnegative")
    if spectral_lower > spectral_upper:
        raise ValueError("spectral_lower cannot exceed spectral_upper")
    phi = np.asarray(represented_state, dtype=complex)
    correction = np.asarray(lifted_correction, dtype=complex)
    if phi.shape != correction.shape or phi.ndim != 1:
        raise ValueError("represented_state and lifted_correction must match")
    spectral_center = 0.5 * (spectral_upper + spectral_lower)
    spectral_half_width = 0.5 * (spectral_upper - spectral_lower)

    def shifted_action(vector: ComplexArray) -> ComplexArray:
        return operator @ vector - spectral_center * vector

    center = float(
        2.0 * np.vdot(shifted_action(phi), correction).real
        + np.vdot(correction, shifted_action(correction)).real
        + spectral_center * (1.0 - np.vdot(phi, phi).real)
    )
    goal_action = shifted_action(phi + correction)
    radius = float(
        2.0 * np.linalg.norm(goal_action) * unresolved_radius
        + spectral_half_width * unresolved_radius**2
    )
    return HermitianGoalInterval(
        lower=center - radius,
        upper=center + radius,
        center=center,
        radius=radius,
        spectral_center=spectral_center,
        spectral_half_width=spectral_half_width,
    )


def reduced_adjoint_goal_interval(
    *,
    represented_state: ComplexArray,
    lifted_correction: ComplexArray,
    unresolved_radius: float,
    operator: ComplexArray | SparseOperator,
    spectral_lower: float,
    spectral_upper: float,
    dual_basis: ComplexArray,
    forward_remainder_state: ComplexArray,
    dual_remainder_radius: float,
    intersection_tolerance: float = 1e-12,
) -> DwrGoalInterval:
    """Intersect the cheap interval with the amended reduced-adjoint interval.

    ``dual_remainder_radius`` is the caller's bound or numerical estimate for
    the unresolved linear DWR pairing, including any forward-remainder
    integration defect.  The quadratic term remains bounded by the global
    spectral half-width times ``unresolved_radius**2``.
    """

    if dual_remainder_radius < 0.0:
        raise ValueError("dual_remainder_radius must be nonnegative")
    cheap = quadratic_goal_interval(
        represented_state=represented_state,
        lifted_correction=lifted_correction,
        unresolved_radius=unresolved_radius,
        operator=operator,
        spectral_lower=spectral_lower,
        spectral_upper=spectral_upper,
    )
    phi = np.asarray(represented_state, dtype=complex)
    correction = np.asarray(lifted_correction, dtype=complex)
    y_basis = _require_orthonormal_basis(
        dual_basis,
        row_dimension=phi.size,
        name="dual_basis",
    )
    xi = np.asarray(forward_remainder_state, dtype=complex)
    if xi.shape != (y_basis.shape[1],):
        raise ValueError("forward_remainder_state has the wrong shape")
    mu = cheap.spectral_center
    goal_action = operator @ (phi + correction) - mu * (phi + correction)
    terminal_projection = y_basis.conj().T @ goal_action
    terminal_defect = float(
        np.linalg.norm(goal_action - y_basis @ terminal_projection)
    )
    linear_center = float(2.0 * np.vdot(terminal_projection, xi).real)
    dwr_center = cheap.center + linear_center
    dwr_radius = float(
        dual_remainder_radius
        + cheap.spectral_half_width * unresolved_radius**2
    )
    dwr_lower = dwr_center - dwr_radius
    dwr_upper = dwr_center + dwr_radius
    lower = max(cheap.lower, dwr_lower)
    upper = min(cheap.upper, dwr_upper)
    slack = intersection_tolerance * max(
        1.0,
        abs(lower),
        abs(upper),
    )
    if lower > upper + slack:
        raise ValueError(
            "cheap and reduced-adjoint intervals have an empty intersection"
        )
    if lower > upper:
        midpoint = 0.5 * (lower + upper)
        lower = midpoint
        upper = midpoint
    return DwrGoalInterval(
        lower=float(lower),
        upper=float(upper),
        cheap=cheap,
        dwr_center=dwr_center,
        dwr_radius=dwr_radius,
        dual_terminal_defect=terminal_defect,
    )
