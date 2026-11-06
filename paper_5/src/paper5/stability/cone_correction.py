"""Structured physicality corrections for the 31D scalar closure.

The bosonic correction acts on the ten real coordinates representing the
Hermitian normal moment ``N`` and complex-symmetric anomalous moment ``A``.
The joint correction additionally acts on the three traceless-Hermitian
electronic-density coordinates.  It can therefore enforce the bosonic moment
cone and both electronic one-body density bounds in one constrained solve.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Callable, Literal

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from .hubbard_dimer import DimerParameters, FloatArray, RhsFunction
from .matrix_reference import (
    CLOSED_SCALAR_STATE_NAMES,
    MatrixDimerState,
    boson_moment_matrix,
    closed_scalar_rhs,
    closed_scalar_to_matrix_state,
    electron_phonon_moment_derivative,
    electron_phonon_moment_matrix,
)

CorrectionMetric = Literal["euclidean", "frobenius"]

HalfspaceBackend = Literal["slsqp", "active_set"]
CutHarvest = Literal["single_mode", "all_violated"]

#: Backend used by :func:`_project_origin_onto_halfspaces` when a caller does
#: not name one.  ``"slsqp"`` is the backend every recorded Paper V number was
#: produced with and stays the default; ``"active_set"`` is the warm-started
#: dual active-set minimum-norm-point solve added on 2026-09-14.  The minimum-
#: norm point of a nonempty polytope is unique, so the two backends solve the
#: same problem and differ only in floating-point path.
DEFAULT_HALFSPACE_BACKEND: HalfspaceBackend = "slsqp"


@dataclass
class SolverCounters:
    """Call counts for the correction solvers.

    Counting is unconditional and changes no number.  ``reset`` is the only
    supported way to start an accounting window.
    """

    halfspace_projections: int = 0
    slsqp_halfspace_projections: int = 0
    active_set_halfspace_projections: int = 0
    closed_form_halfspace_projections: int = 0
    active_set_fallbacks_to_slsqp: int = 0
    slsqp_halfspace_not_converged: int = 0
    active_set_halfspace_not_converged: int = 0
    direct_eigenvalue_slsqp_solves: int = 0
    cutting_plane_fallback_slsqp_solves: int = 0
    correction_memo_hits: int = 0
    correction_memo_misses: int = 0

    def reset(self) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, 0)

    def as_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }


#: Process-wide counters; see :class:`SolverCounters`.
SOLVER_COUNTERS = SolverCounters()

#: Number of (state, derivative) pairs a memoizing right-hand side keeps.
CORRECTION_MEMO_CAPACITY = 8


class _CorrectionMemo:
    """Exact-key cache for one right-hand side's correction coordinates.

    The key is the byte image of the closed state and of the derivative the
    correction is asked to repair; the settings are fixed by the enclosing
    right-hand side.  A hit returns a copy of the stored coordinates, so the
    corrected velocity is the same floating-point vector the solver produced.
    """

    def __init__(self, capacity: int = CORRECTION_MEMO_CAPACITY) -> None:
        self._capacity = max(1, int(capacity))
        self._entries: OrderedDict[bytes, FloatArray] = OrderedDict()

    def lookup(
        self,
        state: FloatArray,
        derivative: FloatArray,
    ) -> FloatArray | None:
        key = (
            np.ascontiguousarray(state, dtype=float).tobytes()
            + np.ascontiguousarray(derivative, dtype=float).tobytes()
        )
        stored = self._entries.get(key)
        if stored is None:
            SOLVER_COUNTERS.correction_memo_misses += 1
            return None
        SOLVER_COUNTERS.correction_memo_hits += 1
        self._entries.move_to_end(key)
        return stored.copy()

    def store(
        self,
        state: FloatArray,
        derivative: FloatArray,
        coordinates: FloatArray,
    ) -> None:
        key = (
            np.ascontiguousarray(state, dtype=float).tobytes()
            + np.ascontiguousarray(derivative, dtype=float).tobytes()
        )
        self._entries[key] = np.asarray(coordinates, dtype=float).copy()
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)


@dataclass(frozen=True)
class StructuredConeCorrection:
    """Diagnostics for one minimum-eigenmode velocity correction."""

    minimum_eigenvalue: float
    next_eigenvalue: float
    raw_flux: float
    corrected_flux: float
    target_flux: float
    response_vector: FloatArray
    correction_coordinates: FloatArray
    active: bool

    @property
    def correction_norm(self) -> float:
        """Return the Euclidean norm in the ten real moment coordinates."""

        return float(np.linalg.norm(self.correction_coordinates))


@dataclass(frozen=True)
class StructuredConeBarrierCorrection:
    """Diagnostics for a full matrix control-barrier correction."""

    moment_minimum_eigenvalue: float
    raw_barrier_minimum_eigenvalue: float
    corrected_barrier_minimum_eigenvalue: float
    correction_coordinates: FloatArray
    constraint_count: int
    converged: bool

    @property
    def correction_norm(self) -> float:
        """Return the Euclidean norm in the ten real moment coordinates."""

        return float(np.linalg.norm(self.correction_coordinates))


@dataclass(frozen=True)
class StructuredJointBarrierCorrection:
    """Diagnostics for one joint electronic--bosonic barrier solve."""

    electron_minimum_eigenvalue: float
    electron_maximum_eigenvalue: float
    boson_moment_minimum_eigenvalue: float
    raw_electron_lower_barrier_minimum_eigenvalue: float
    raw_electron_upper_barrier_minimum_eigenvalue: float
    raw_boson_barrier_minimum_eigenvalue: float
    corrected_electron_lower_barrier_minimum_eigenvalue: float
    corrected_electron_upper_barrier_minimum_eigenvalue: float
    corrected_boson_barrier_minimum_eigenvalue: float
    correction_coordinates: FloatArray
    correction_energy_flux: float
    constraint_count: int
    converged: bool

    @property
    def correction_norm(self) -> float:
        """Return the Euclidean norm in the thirteen control coordinates."""

        return float(np.linalg.norm(self.correction_coordinates))


@dataclass(frozen=True)
class StructuredFullStateBarrierCorrection(StructuredJointBarrierCorrection):
    """Joint barrier diagnostics with a 31-coordinate correction velocity."""

    @property
    def correction_norm(self) -> float:
        """Return the Euclidean norm in all 31 correction coordinates."""

        return float(np.linalg.norm(self.correction_coordinates))


@dataclass(frozen=True)
class StructuredElectronPhononBarrierCorrection:
    """Diagnostics for the joint electron--phonon Gram-matrix barrier."""

    electron_minimum_eigenvalue: float
    electron_maximum_eigenvalue: float
    joint_moment_minimum_eigenvalue: float
    raw_electron_lower_barrier_minimum_eigenvalue: float
    raw_electron_upper_barrier_minimum_eigenvalue: float
    raw_joint_barrier_minimum_eigenvalue: float
    corrected_electron_lower_barrier_minimum_eigenvalue: float
    corrected_electron_upper_barrier_minimum_eigenvalue: float
    corrected_joint_barrier_minimum_eigenvalue: float
    correction_coordinates: FloatArray
    correction_energy_flux: float
    corrected_correlation_trace_velocity: complex
    constraint_count: int
    converged: bool

    @property
    def correction_norm(self) -> float:
        """Return the Euclidean norm in all 31 correction coordinates."""

        return float(np.linalg.norm(self.correction_coordinates))

    @property
    def lifted_frobenius_norm(self) -> float:
        """Return the blockwise Frobenius norm of the lifted correction."""

        return closed_state_lifted_frobenius_norm(
            self.correction_coordinates
        )


def structured_electron_velocity_lift(coordinates: FloatArray) -> np.ndarray:
    """Lift three real density coordinates to a traceless Hermitian matrix.

    The order matches closed-state indices 0--2:
    ``delta_n, Re rho01, Im rho01``.  Tracelessness preserves the fixed
    one-electron trace built into the affine density parameterization.
    """

    array = np.asarray(coordinates, dtype=float)
    if array.shape != (3,):
        raise ValueError(
            f"expected three correction coordinates, got shape {array.shape}"
        )
    return np.array(
        [
            [0.5 * array[0], array[1] + 1j * array[2]],
            [array[1] - 1j * array[2], -0.5 * array[0]],
        ],
        dtype=complex,
    )


_STRUCTURED_ELECTRON_VELOCITY_BASIS = tuple(
    structured_electron_velocity_lift(np.eye(3, dtype=float)[index])
    for index in range(3)
)


def structured_boson_velocity_lift(coordinates: FloatArray) -> np.ndarray:
    """Lift ten real ``(dN, dA)`` coordinates to a Hermitian 4-by-4 matrix.

    The coordinate order is the same as indices 7--16 of the closed state:
    ``N00, N11, Re N01, Im N01, Re A00, Im A00, Re A11, Im A11,
    Re A01, Im A01``.
    """

    array = np.asarray(coordinates, dtype=float)
    if array.shape != (10,):
        raise ValueError(
            f"expected ten correction coordinates, got shape {array.shape}"
        )

    normal_01 = array[2] + 1j * array[3]
    normal = np.array(
        [
            [array[0], normal_01],
            [normal_01.conjugate(), array[1]],
        ],
        dtype=complex,
    )
    anomalous = np.array(
        [
            [array[4] + 1j * array[5], array[8] + 1j * array[9]],
            [array[8] + 1j * array[9], array[6] + 1j * array[7]],
        ],
        dtype=complex,
    )
    # Block assembly by slice copy.  ``np.block`` produced the same four
    # copies through a recursive shape check whose Python cost dominated this
    # function; every stored value is identical.
    lifted = np.empty((4, 4), dtype=complex)
    lifted[:2, :2] = normal.T
    lifted[:2, 2:] = anomalous.conjugate()
    lifted[2:, :2] = anomalous
    lifted[2:, 2:] = normal
    return lifted


_STRUCTURED_VELOCITY_BASIS = tuple(
    structured_boson_velocity_lift(np.eye(10, dtype=float)[index])
    for index in range(10)
)


def structured_correlation_velocity_lift(
    coordinates: FloatArray,
) -> np.ndarray:
    """Lift the fourteen real correlation velocities to two 2-by-2 blocks."""

    array = np.asarray(coordinates, dtype=float)
    if array.shape != (14,):
        raise ValueError(
            f"expected fourteen correction coordinates, got shape {array.shape}"
        )

    shared_trace = array[0] + 1j * array[1]
    correlations: list[np.ndarray] = []
    for offset in (2, 8):
        diagonal_difference = array[offset] + 1j * array[offset + 1]
        correlation_01 = array[offset + 2] + 1j * array[offset + 3]
        correlation_10 = array[offset + 4] + 1j * array[offset + 5]
        correlations.append(
            np.array(
                [
                    [
                        0.5 * (shared_trace + diagonal_difference),
                        correlation_01,
                    ],
                    [
                        correlation_10,
                        0.5 * (shared_trace - diagonal_difference),
                    ],
                ],
                dtype=complex,
            )
        )
    return np.stack(correlations)


def structured_closed_state_velocity_lift(
    coordinates: FloatArray,
) -> MatrixDimerState:
    """Lift 31 real correction coordinates to the five matrix blocks."""

    array = np.asarray(coordinates, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if array.shape != expected:
        raise ValueError(
            f"expected correction shape {expected}, got {array.shape}"
        )

    boson_velocity = structured_boson_velocity_lift(array[7:17])
    return MatrixDimerState(
        electron_density=structured_electron_velocity_lift(array[:3]),
        coherent_phonon=np.array(
            [
                array[3] + 1j * array[4],
                array[5] + 1j * array[6],
            ],
            dtype=complex,
        ),
        phonon_density=boson_velocity[2:, 2:],
        anomalous_phonon_density=boson_velocity[2:, :2],
        electron_phonon_correlation=(
            structured_correlation_velocity_lift(array[17:])
        ),
    )


def _matrix_state_frobenius_inner(
    left: MatrixDimerState,
    right: MatrixDimerState,
) -> float:
    return float(
        sum(
            np.vdot(getattr(left, field), getattr(right, field)).real
            for field in (
                "electron_density",
                "coherent_phonon",
                "phonon_density",
                "anomalous_phonon_density",
                "electron_phonon_correlation",
            )
        )
    )


_CLOSED_STATE_VELOCITY_BASIS = tuple(
    structured_closed_state_velocity_lift(
        np.eye(len(CLOSED_SCALAR_STATE_NAMES), dtype=float)[index]
    )
    for index in range(len(CLOSED_SCALAR_STATE_NAMES))
)
_CLOSED_STATE_LIFTED_FROBENIUS_METRIC = np.asarray(
    [
        [
            _matrix_state_frobenius_inner(left, right)
            for right in _CLOSED_STATE_VELOCITY_BASIS
        ]
        for left in _CLOSED_STATE_VELOCITY_BASIS
    ],
    dtype=float,
)


def _metric_coordinate_map(metric_matrix: FloatArray) -> FloatArray:
    """Return ``chol(metric)^-T`` as the map from metric to raw coordinates."""

    metric_factor = np.linalg.cholesky(metric_matrix).T
    return np.linalg.solve(
        metric_factor,
        np.eye(metric_matrix.shape[0], dtype=float),
    )


#: The two correction metrics are fixed matrices, so their Cholesky factor and
#: the triangular solve against the identity are evaluated once at import
#: instead of on every correction solve.  Same operands, same stored values.
_METRIC_COORDINATE_MAPS: dict[str, FloatArray] = {
    "euclidean": _metric_coordinate_map(
        np.eye(len(CLOSED_SCALAR_STATE_NAMES), dtype=float)
    ),
    "frobenius": _metric_coordinate_map(
        _CLOSED_STATE_LIFTED_FROBENIUS_METRIC
    ),
}


def closed_state_lifted_frobenius_metric() -> FloatArray:
    """Return the fixed 31-coordinate metric induced by the matrix lift."""

    return _CLOSED_STATE_LIFTED_FROBENIUS_METRIC.copy()


def closed_state_lifted_frobenius_norm(coordinates: FloatArray) -> float:
    """Return the combined Frobenius norm of all five lifted blocks."""

    array = np.asarray(coordinates, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if array.shape != expected:
        raise ValueError(
            f"expected correction shape {expected}, got {array.shape}"
        )
    squared = float(
        array @ _CLOSED_STATE_LIFTED_FROBENIUS_METRIC @ array
    )
    return float(np.sqrt(max(0.0, squared)))


def structured_electron_phonon_moment_velocity_lift(
    state: FloatArray,
    coordinates: FloatArray,
) -> np.ndarray:
    """Lift a 31-coordinate correction to the joint Gram-matrix velocity."""

    state_array = np.asarray(state, dtype=float)
    coordinate_array = np.asarray(coordinates, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if state_array.shape != expected:
        raise ValueError(
            f"expected closed state shape {expected}, got {state_array.shape}"
        )
    if coordinate_array.shape != expected:
        raise ValueError(
            "expected correction shape "
            f"{expected}, got {coordinate_array.shape}"
        )

    matrix_derivative = structured_closed_state_velocity_lift(
        coordinate_array
    )
    return electron_phonon_moment_derivative(
        closed_scalar_to_matrix_state(state_array),
        matrix_derivative,
    )


def joint_correction_energy_gradient(
    state: FloatArray,
    parameters: DimerParameters,
) -> FloatArray:
    """Return the total-energy gradient in the thirteen joint controls.

    The control order is the three electronic-density coordinates followed by
    the ten ``(N, A)`` coordinates.  For the time-independent Eq. (22) energy,
    ``A`` contributes no direct term.  This analytical row is the derivative
    of :func:`matrix_total_energy` with all uncorrected state blocks held fixed.
    """

    array = np.asarray(state, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if array.shape != expected:
        raise ValueError(
            f"expected closed state shape {expected}, got {array.shape}"
        )

    gradient = np.zeros(13, dtype=float)
    gradient[0] = 2.0 * parameters.coupling * (array[3] - array[5])
    gradient[1] = -4.0 * parameters.hopping
    gradient[3] = parameters.omega_ph
    gradient[4] = parameters.omega_ph
    return gradient


def closed_state_correction_energy_gradient(
    state: FloatArray,
    parameters: DimerParameters,
) -> FloatArray:
    """Return the Eq. (22) energy gradient in all 31 closed coordinates."""

    array = np.asarray(state, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if array.shape != expected:
        raise ValueError(
            f"expected closed state shape {expected}, got {array.shape}"
        )

    coupling = parameters.coupling
    omega = parameters.omega_ph
    rho_00 = 0.5 * (1.0 + array[0])
    rho_11 = 0.5 * (1.0 - array[0])
    gradient = np.zeros(len(CLOSED_SCALAR_STATE_NAMES), dtype=float)
    gradient[0] = 2.0 * coupling * (array[3] - array[5])
    gradient[1] = -4.0 * parameters.hopping
    gradient[3] = 2.0 * omega * array[3] + 4.0 * coupling * rho_00
    gradient[4] = 2.0 * omega * array[4]
    gradient[5] = 2.0 * omega * array[5] + 4.0 * coupling * rho_11
    gradient[6] = 2.0 * omega * array[6]
    gradient[7] = omega
    gradient[8] = omega
    gradient[17] = 4.0 * coupling
    gradient[19] = 2.0 * coupling
    gradient[25] = -2.0 * coupling
    return gradient


def _boson_velocity_from_closed_derivative(
    derivative: FloatArray,
) -> np.ndarray:
    array = np.asarray(derivative, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if array.shape != expected:
        raise ValueError(
            f"expected closed derivative shape {expected}, got {array.shape}"
        )
    return structured_boson_velocity_lift(array[7:17])


def _mode_response_vector(mode: np.ndarray) -> FloatArray:
    """Return ``v†S(e_j)v`` for the ten structured velocity directions."""

    return np.array(
        [
            np.real(np.vdot(mode, basis @ mode))
            for basis in _STRUCTURED_VELOCITY_BASIS
        ],
        dtype=float,
    )


@dataclass
class HalfspaceWarmState:
    """Carried active set for the warm-started minimum-norm-point backend.

    The cut set only grows inside one constraint-generation loop, so the
    previous pass's active set is a valid starting working set for the next.
    """

    active: tuple[int, ...] = ()
    iterations: int = 0


def _slsqp_minimum_norm_point(
    matrix: FloatArray,
    bounds: FloatArray,
    *,
    tolerance: float,
    maximum_cycles: int,
) -> tuple[FloatArray, bool]:
    """Return the SLSQP minimum-norm point of ``matrix @ y >= bounds``."""

    constraint = LinearConstraint(matrix, bounds, np.inf)
    dimension = matrix.shape[1]
    SOLVER_COUNTERS.slsqp_halfspace_projections += 1
    solution = minimize(
        lambda point: 0.5 * float(point @ point),
        np.zeros(dimension, dtype=float),
        jac=lambda point: point,
        constraints=(constraint,),
        method="SLSQP",
        options={
            "ftol": tolerance,
            "maxiter": maximum_cycles,
            "disp": False,
        },
    )
    point = np.asarray(solution.x, dtype=float)
    maximum_violation = float(np.max(bounds - matrix @ point))
    converged = bool(
        solution.success and maximum_violation <= 10.0 * tolerance
    )
    if not converged:
        SOLVER_COUNTERS.slsqp_halfspace_not_converged += 1
    return point, converged


def _active_set_minimum_norm_point(
    matrix: FloatArray,
    bounds: FloatArray,
    *,
    tolerance: float,
    maximum_cycles: int,
    warm_state: HalfspaceWarmState | None,
) -> tuple[FloatArray, bool]:
    """Return the minimum-norm point of ``matrix @ y >= bounds`` directly.

    The problem is ``min 1/2 ||y||^2`` over a polytope, whose objective is
    strictly convex, so the minimizer is the unique minimum-norm point of that
    polytope and does not depend on how it is found.  This is the dual
    active-set construction: on a working set ``W`` the Karush--Kuhn--Tucker
    point is ``y = A_W^T mu`` with ``A_W A_W^T mu = b_W``; a negative
    multiplier drops the constraint that carries it and the most violated
    remaining constraint enters.  ``warm_state`` carries the previous pass's
    working set, which is feasible to start from because the cut set only
    grows.

    One accumulated half-space has the closed-form solution
    ``y = max(0, b) n / (n . n)``, the same expression the boson barrier uses
    for its single-mode flux correction.
    """

    rows, dimension = matrix.shape
    if rows == 1:
        SOLVER_COUNTERS.closed_form_halfspace_projections += 1
        normal = matrix[0]
        compliance = float(normal @ normal)
        if compliance <= 0.0:
            return (
                np.zeros(dimension, dtype=float),
                bool(float(bounds[0]) <= 10.0 * tolerance),
            )
        scale = max(0.0, float(bounds[0])) / compliance
        point = scale * normal
        maximum_violation = float(np.max(bounds - matrix @ point))
        if warm_state is not None:
            warm_state.active = (0,) if scale > 0.0 else ()
            warm_state.iterations = 1
        return point, bool(maximum_violation <= 10.0 * tolerance)

    SOLVER_COUNTERS.active_set_halfspace_projections += 1
    gram = matrix @ matrix.T
    working: list[int] = []
    if warm_state is not None:
        working = [index for index in warm_state.active if index < rows]

    entry_tolerance = 10.0 * tolerance
    iterations = 0
    limit = max(4 * rows + 8, 32)
    point = np.zeros(dimension, dtype=float)
    while iterations < limit:
        iterations += 1
        if working:
            indices = np.asarray(working, dtype=int)
            block = gram[np.ix_(indices, indices)]
            values = bounds[indices]
            try:
                multipliers = np.linalg.solve(block, values)
            except np.linalg.LinAlgError:
                multipliers = np.linalg.lstsq(block, values, rcond=None)[0]
            if not np.all(np.isfinite(multipliers)):
                multipliers = np.linalg.lstsq(block, values, rcond=None)[0]
            negative = int(np.argmin(multipliers))
            if float(multipliers[negative]) < -entry_tolerance:
                working.pop(negative)
                continue
            point = matrix[indices].T @ multipliers
        else:
            point = np.zeros(dimension, dtype=float)
        residual = bounds - matrix @ point
        entering = int(np.argmax(residual))
        if float(residual[entering]) <= entry_tolerance:
            break
        if entering in working:
            break
        working.append(entering)
        working.sort()

    if warm_state is not None:
        warm_state.active = tuple(working)
        warm_state.iterations = iterations

    maximum_violation = float(np.max(bounds - matrix @ point))
    if maximum_violation <= 10.0 * tolerance:
        return point, True

    # Declared fallback: the active-set walk did not certify feasibility, so
    # the recorded SLSQP backend answers instead.
    SOLVER_COUNTERS.active_set_halfspace_not_converged += 1
    SOLVER_COUNTERS.active_set_fallbacks_to_slsqp += 1
    return _slsqp_minimum_norm_point(
        matrix,
        bounds,
        tolerance=tolerance,
        maximum_cycles=maximum_cycles,
    )


def _project_origin_onto_halfspaces(
    normals: list[FloatArray],
    offsets: list[float],
    *,
    tolerance: float,
    maximum_cycles: int,
    backend: HalfspaceBackend | None = None,
    warm_state: HalfspaceWarmState | None = None,
) -> tuple[FloatArray, bool]:
    """Project the origin onto ``normal @ y >= offset``.

    ``backend`` selects the solver.  ``"slsqp"`` is the recorded default and
    every existing Paper V number was produced with it; ``"active_set"`` is the
    warm-started minimum-norm-point solve.  Both return the unique minimum-norm
    point of the same polytope.
    """

    if not normals:
        return np.zeros(0, dtype=float), True

    resolved = DEFAULT_HALFSPACE_BACKEND if backend is None else backend
    if resolved not in ("slsqp", "active_set"):
        raise ValueError(
            "halfspace backend must be 'slsqp' or 'active_set'"
        )

    matrix = np.vstack(normals)
    bounds = np.asarray(offsets, dtype=float)
    SOLVER_COUNTERS.halfspace_projections += 1
    if resolved == "active_set":
        return _active_set_minimum_norm_point(
            matrix,
            bounds,
            tolerance=tolerance,
            maximum_cycles=maximum_cycles,
            warm_state=warm_state,
        )
    return _slsqp_minimum_norm_point(
        matrix,
        bounds,
        tolerance=tolerance,
        maximum_cycles=maximum_cycles,
    )


def _orthonormal_null_space(row: FloatArray) -> FloatArray:
    """Return an orthonormal basis for the null space of one real row."""

    array = np.asarray(row, dtype=float)
    if array.ndim != 1:
        raise ValueError("row must be one-dimensional")
    if np.linalg.norm(array) <= 1e-14:
        return np.eye(array.size, dtype=float)
    _, _, right_vectors = np.linalg.svd(
        array.reshape(1, -1),
        full_matrices=True,
    )
    return right_vectors[1:].T


def _affine_equality_parameterization(
    matrix: FloatArray,
    values: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return the minimum-norm point and null basis for ``matrix @ x=values``."""

    equality_matrix = np.asarray(matrix, dtype=float)
    equality_values = np.asarray(values, dtype=float)
    if equality_matrix.ndim != 2:
        raise ValueError("equality matrix must be two-dimensional")
    if equality_values.shape != (equality_matrix.shape[0],):
        raise ValueError("equality values do not match the matrix rows")

    if equality_matrix.shape[0] == 0:
        dimension = equality_matrix.shape[1]
        return np.zeros(dimension, dtype=float), np.eye(dimension, dtype=float)

    left_vectors, singular_values, right_vectors = np.linalg.svd(
        equality_matrix,
        full_matrices=True,
    )
    threshold = (
        np.finfo(float).eps
        * max(equality_matrix.shape)
        * singular_values[0]
    )
    rank = int(np.count_nonzero(singular_values > threshold))
    particular = np.linalg.lstsq(
        equality_matrix,
        equality_values,
        rcond=None,
    )[0]
    if np.linalg.norm(equality_matrix @ particular - equality_values) > 1e-10:
        raise ValueError("correction equalities are inconsistent")
    del left_vectors
    null_basis = right_vectors[rank:].T
    return np.asarray(particular, dtype=float), null_basis


def _joint_mode_response_vector(
    sector: str,
    mode: np.ndarray,
) -> FloatArray:
    """Return the thirteen-control response for one barrier eigenmode."""

    response = np.zeros(13, dtype=float)
    if sector in ("electron_lower", "electron_upper"):
        sign = 1.0 if sector == "electron_lower" else -1.0
        response[:3] = sign * np.array(
            [
                np.real(np.vdot(mode, basis @ mode))
                for basis in _STRUCTURED_ELECTRON_VELOCITY_BASIS
            ],
            dtype=float,
        )
        return response
    if sector == "boson":
        response[3:] = _mode_response_vector(mode)
        return response
    raise ValueError(f"unknown joint barrier sector {sector!r}")


def _full_state_joint_mode_response_vector(
    sector: str,
    mode: np.ndarray,
) -> FloatArray:
    """Return the 31-coordinate response for one barrier eigenmode."""

    restricted = _joint_mode_response_vector(sector, mode)
    response = np.zeros(len(CLOSED_SCALAR_STATE_NAMES), dtype=float)
    response[:3] = restricted[:3]
    response[7:17] = restricted[3:]
    return response


def _direct_joint_barrier_solve(
    raw_barriers: dict[str, np.ndarray],
    coordinate_basis: FloatArray,
    *,
    electron_coordinates: slice,
    boson_coordinates: slice,
    mode_response: Callable[[str, np.ndarray], FloatArray],
    cone_tolerance: float,
    projection_tolerance: float,
    maximum_projection_cycles: int,
) -> tuple[FloatArray, dict[str, float], bool, int]:
    """Solve the small joint semidefinite problem through all eigenvalues."""

    sectors = ("electron_lower", "electron_upper", "boson")
    # SLSQP evaluates the constraint function and its Jacobian at the same
    # reduced point, so the three corrected matrices are assembled once per
    # point and reused.  The stored matrices are the same objects the previous
    # code rebuilt.
    barrier_cache: OrderedDict[bytes, dict[str, np.ndarray]] = OrderedDict()

    def _build_corrected_barriers(
        reduced: FloatArray,
    ) -> dict[str, np.ndarray]:
        correction = coordinate_basis @ reduced
        electron_correction = structured_electron_velocity_lift(
            correction[electron_coordinates]
        )
        boson_correction = structured_boson_velocity_lift(
            correction[boson_coordinates]
        )
        return {
            "electron_lower": (
                raw_barriers["electron_lower"] + electron_correction
            ),
            "electron_upper": (
                raw_barriers["electron_upper"] - electron_correction
            ),
            "boson": raw_barriers["boson"] + boson_correction,
        }

    def corrected_barriers(reduced: FloatArray) -> dict[str, np.ndarray]:
        key = np.ascontiguousarray(reduced, dtype=float).tobytes()
        cached = barrier_cache.get(key)
        if cached is not None:
            return cached
        built = _build_corrected_barriers(reduced)
        barrier_cache[key] = built
        if len(barrier_cache) > 4:
            barrier_cache.popitem(last=False)
        return built

    def eigenvalue_constraints(reduced: FloatArray) -> FloatArray:
        return np.concatenate(
            [
                np.linalg.eigvalsh(matrix)
                for matrix in corrected_barriers(reduced).values()
            ]
        )

    def eigenvalue_jacobian(reduced: FloatArray) -> FloatArray:
        rows: list[FloatArray] = []
        matrices = corrected_barriers(reduced)
        for sector in sectors:
            _, eigenvectors = np.linalg.eigh(matrices[sector])
            for column in range(eigenvectors.shape[1]):
                rows.append(
                    coordinate_basis.T
                    @ mode_response(
                        sector,
                        eigenvectors[:, column],
                    )
                )
        return np.asarray(rows, dtype=float)

    reduced_dimension = coordinate_basis.shape[1]
    SOLVER_COUNTERS.direct_eigenvalue_slsqp_solves += 1
    solution = minimize(
        lambda reduced: 0.5 * float(reduced @ reduced),
        np.zeros(reduced_dimension, dtype=float),
        jac=lambda reduced: reduced,
        constraints=(
            {
                "type": "ineq",
                "fun": eigenvalue_constraints,
                "jac": eigenvalue_jacobian,
            },
        ),
        method="SLSQP",
        options={
            "ftol": projection_tolerance,
            "maxiter": maximum_projection_cycles,
            "disp": False,
        },
    )
    correction = coordinate_basis @ np.asarray(solution.x, dtype=float)
    corrected = corrected_barriers(np.asarray(solution.x, dtype=float))
    corrected_minima = {
        sector: float(np.linalg.eigvalsh(corrected[sector])[0])
        for sector in sectors
    }
    # SLSQP can report a line-search status at an eigenvalue degeneracy even
    # when the returned point satisfies every matrix inequality.  Feasibility
    # of the complete matrices is the convergence contract used by the RHS;
    # an optimizer failure away from feasibility still fails this check.
    converged = bool(min(corrected_minima.values()) >= -cone_tolerance)
    constraint_count = int(sum(raw_barriers[name].shape[0] for name in sectors))
    return correction, corrected_minima, converged, constraint_count


def structured_joint_barrier_correction(
    state: FloatArray,
    derivative: FloatArray,
    parameters: DimerParameters,
    *,
    activation_margin: float = 0.0,
    target_flux: float = 0.0,
    barrier_rate: float = 5.0,
    energy_neutral: bool = True,
    cone_tolerance: float = 1e-11,
    projection_tolerance: float = 1e-12,
    maximum_constraints: int = 128,
    maximum_projection_cycles: int = 10_000,
    solver: Literal["cutting_plane", "direct_eigenvalue"] = "cutting_plane",
    halfspace_backend: HalfspaceBackend | None = None,
    cut_harvest: CutHarvest = "single_mode",
) -> StructuredJointBarrierCorrection:
    """Enforce electronic and bosonic matrix barriers in one least-norm solve.

    The thirteen real controls are

    ``(d delta_n, d Re rho01, d Im rho01, dN, dA)``.

    The first three lift to a traceless Hermitian ``d rho`` and therefore
    preserve ``Tr(rho)=1``.  Constraint generation enforces the three complete
    matrix inequalities

    ``d rho + beta * (rho - margin I) - target I >= 0``,

    ``-d rho + beta * (I - rho - margin I) - target I >= 0``, and

    ``d M_B + beta * (M_B - margin I) - target I >= 0``.

    With ``energy_neutral=True``, the controls are restricted to the null
    space of the exact Eq. (22) energy-gradient row, so the correction itself
    contributes zero instantaneous energy flux.
    """

    if activation_margin < 0.0:
        raise ValueError("activation_margin must be nonnegative")
    if target_flux < 0.0:
        raise ValueError("target_flux must be nonnegative")
    if barrier_rate <= 0.0:
        raise ValueError("barrier_rate must be positive")
    if cone_tolerance <= 0.0 or projection_tolerance <= 0.0:
        raise ValueError("solver tolerances must be positive")
    if maximum_constraints <= 0 or maximum_projection_cycles <= 0:
        raise ValueError("solver iteration limits must be positive")
    if solver not in ("cutting_plane", "direct_eigenvalue"):
        raise ValueError(
            "solver must be 'cutting_plane' or 'direct_eigenvalue'"
        )
    if cut_harvest not in ("single_mode", "all_violated"):
        raise ValueError(
            "cut_harvest must be 'single_mode' or 'all_violated'"
        )

    state_array = np.asarray(state, dtype=float)
    derivative_array = np.asarray(derivative, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if state_array.shape != expected:
        raise ValueError(
            f"expected closed state shape {expected}, got {state_array.shape}"
        )
    if derivative_array.shape != expected:
        raise ValueError(
            "expected closed derivative shape "
            f"{expected}, got {derivative_array.shape}"
        )

    matrix_state = closed_scalar_to_matrix_state(state_array)
    electron = np.asarray(matrix_state.electron_density, dtype=complex)
    electron = 0.5 * (electron + electron.conjugate().T)
    electron_velocity = structured_electron_velocity_lift(
        derivative_array[:3]
    )
    electron_identity = np.eye(2, dtype=complex)

    moment = boson_moment_matrix(matrix_state)
    moment = 0.5 * (moment + moment.conjugate().T)
    boson_velocity = _boson_velocity_from_closed_derivative(derivative_array)
    boson_velocity = 0.5 * (
        boson_velocity + boson_velocity.conjugate().T
    )
    boson_identity = np.eye(4, dtype=complex)

    raw_barriers = {
        "electron_lower": electron_velocity
        + barrier_rate * (
            electron - activation_margin * electron_identity
        )
        - target_flux * electron_identity,
        "electron_upper": -electron_velocity
        + barrier_rate * (
            electron_identity
            - electron
            - activation_margin * electron_identity
        )
        - target_flux * electron_identity,
        "boson": boson_velocity
        + barrier_rate * (moment - activation_margin * boson_identity)
        - target_flux * boson_identity,
    }
    raw_barriers = {
        name: 0.5 * (value + value.conjugate().T)
        for name, value in raw_barriers.items()
    }
    raw_minima = {
        name: float(np.linalg.eigvalsh(value)[0])
        for name, value in raw_barriers.items()
    }
    electron_eigenvalues = np.linalg.eigvalsh(electron)
    moment_minimum = float(np.linalg.eigvalsh(moment)[0])

    energy_gradient = joint_correction_energy_gradient(
        state_array,
        parameters,
    )
    coordinate_basis = (
        _orthonormal_null_space(energy_gradient)
        if energy_neutral
        else np.eye(13, dtype=float)
    )

    if solver == "direct_eigenvalue":
        (
            correction,
            corrected_minima,
            converged,
            constraint_count,
        ) = _direct_joint_barrier_solve(
            raw_barriers,
            coordinate_basis,
            electron_coordinates=slice(0, 3),
            boson_coordinates=slice(3, 13),
            mode_response=_joint_mode_response_vector,
            cone_tolerance=cone_tolerance,
            projection_tolerance=projection_tolerance,
            maximum_projection_cycles=maximum_projection_cycles,
        )
        return StructuredJointBarrierCorrection(
            electron_minimum_eigenvalue=float(electron_eigenvalues[0]),
            electron_maximum_eigenvalue=float(electron_eigenvalues[-1]),
            boson_moment_minimum_eigenvalue=moment_minimum,
            raw_electron_lower_barrier_minimum_eigenvalue=raw_minima[
                "electron_lower"
            ],
            raw_electron_upper_barrier_minimum_eigenvalue=raw_minima[
                "electron_upper"
            ],
            raw_boson_barrier_minimum_eigenvalue=raw_minima["boson"],
            corrected_electron_lower_barrier_minimum_eigenvalue=(
                corrected_minima["electron_lower"]
            ),
            corrected_electron_upper_barrier_minimum_eigenvalue=(
                corrected_minima["electron_upper"]
            ),
            corrected_boson_barrier_minimum_eigenvalue=corrected_minima[
                "boson"
            ],
            correction_coordinates=correction,
            correction_energy_flux=float(energy_gradient @ correction),
            constraint_count=constraint_count,
            converged=converged,
        )

    correction = np.zeros(13, dtype=float)
    corrected_minima = dict(raw_minima)

    if min(raw_minima.values()) >= -cone_tolerance:
        return StructuredJointBarrierCorrection(
            electron_minimum_eigenvalue=float(electron_eigenvalues[0]),
            electron_maximum_eigenvalue=float(electron_eigenvalues[-1]),
            boson_moment_minimum_eigenvalue=moment_minimum,
            raw_electron_lower_barrier_minimum_eigenvalue=raw_minima[
                "electron_lower"
            ],
            raw_electron_upper_barrier_minimum_eigenvalue=raw_minima[
                "electron_upper"
            ],
            raw_boson_barrier_minimum_eigenvalue=raw_minima["boson"],
            corrected_electron_lower_barrier_minimum_eigenvalue=raw_minima[
                "electron_lower"
            ],
            corrected_electron_upper_barrier_minimum_eigenvalue=raw_minima[
                "electron_upper"
            ],
            corrected_boson_barrier_minimum_eigenvalue=raw_minima["boson"],
            correction_coordinates=correction,
            correction_energy_flux=0.0,
            constraint_count=0,
            converged=True,
        )

    normals: list[FloatArray] = []
    offsets: list[float] = []
    projection_converged = True
    halfspace_warm_state = HalfspaceWarmState()
    for _ in range(maximum_constraints):
        electron_correction = structured_electron_velocity_lift(
            correction[:3]
        )
        boson_correction = structured_boson_velocity_lift(correction[3:])
        corrected_barriers = {
            "electron_lower": raw_barriers["electron_lower"]
            + electron_correction,
            "electron_upper": raw_barriers["electron_upper"]
            - electron_correction,
            "boson": raw_barriers["boson"] + boson_correction,
        }
        eigensystems = {
            name: np.linalg.eigh(value)
            for name, value in corrected_barriers.items()
        }
        corrected_minima = {
            name: float(eigensystem[0][0])
            for name, eigensystem in eigensystems.items()
        }
        sector = min(corrected_minima, key=corrected_minima.get)
        if corrected_minima[sector] >= -cone_tolerance:
            return StructuredJointBarrierCorrection(
                electron_minimum_eigenvalue=float(electron_eigenvalues[0]),
                electron_maximum_eigenvalue=float(electron_eigenvalues[-1]),
                boson_moment_minimum_eigenvalue=moment_minimum,
                raw_electron_lower_barrier_minimum_eigenvalue=raw_minima[
                    "electron_lower"
                ],
                raw_electron_upper_barrier_minimum_eigenvalue=raw_minima[
                    "electron_upper"
                ],
                raw_boson_barrier_minimum_eigenvalue=raw_minima["boson"],
                corrected_electron_lower_barrier_minimum_eigenvalue=(
                    corrected_minima["electron_lower"]
                ),
                corrected_electron_upper_barrier_minimum_eigenvalue=(
                    corrected_minima["electron_upper"]
                ),
                corrected_boson_barrier_minimum_eigenvalue=corrected_minima[
                    "boson"
                ],
                correction_coordinates=correction,
                correction_energy_flux=float(energy_gradient @ correction),
                constraint_count=len(normals),
                converged=projection_converged,
            )

        # ``single_mode`` (the recorded default) cuts on the lowest eigenmode
        # of the most violated sector alone.  ``all_violated`` harvests every
        # eigenmode below the cone tolerance in every sector on the same pass,
        # which is what the electron--phonon routine already does; it reaches
        # feasibility in fewer passes but accumulates a different cut set, so
        # it is a named option and not the default.
        harvested: list[tuple[str, np.ndarray]] = []
        if cut_harvest == "single_mode":
            harvested.append((sector, eigensystems[sector][1][:, 0]))
        else:
            for name in ("electron_lower", "electron_upper", "boson"):
                eigenvalues, eigenvectors = eigensystems[name]
                for column, eigenvalue in enumerate(eigenvalues):
                    if eigenvalue >= -cone_tolerance:
                        continue
                    harvested.append((name, eigenvectors[:, column]))

        added = False
        degenerate = False
        for cut_sector, violated_mode in harvested:
            full_response = _joint_mode_response_vector(
                cut_sector,
                violated_mode,
            )
            reduced_response = coordinate_basis.T @ full_response
            if float(reduced_response @ reduced_response) <= 1e-14:
                degenerate = True
                continue
            normals.append(reduced_response)
            offsets.append(
                -float(
                    np.real(
                        np.vdot(
                            violated_mode,
                            raw_barriers[cut_sector] @ violated_mode,
                        )
                    )
                )
            )
            added = True
        if not added:
            if degenerate:
                break
            break
        reduced_correction, current_projection_converged = (
            _project_origin_onto_halfspaces(
                normals,
                offsets,
                tolerance=projection_tolerance,
                maximum_cycles=maximum_projection_cycles,
                backend=halfspace_backend,
                warm_state=halfspace_warm_state,
            )
        )
        projection_converged = bool(
            projection_converged and current_projection_converged
        )
        correction = coordinate_basis @ reduced_correction

    electron_correction = structured_electron_velocity_lift(correction[:3])
    corrected_barriers = {
        "electron_lower": raw_barriers["electron_lower"]
        + electron_correction,
        "electron_upper": raw_barriers["electron_upper"]
        - electron_correction,
        "boson": raw_barriers["boson"]
        + structured_boson_velocity_lift(correction[3:]),
    }
    corrected_minima = {
        name: float(np.linalg.eigvalsh(value)[0])
        for name, value in corrected_barriers.items()
    }
    return StructuredJointBarrierCorrection(
        electron_minimum_eigenvalue=float(electron_eigenvalues[0]),
        electron_maximum_eigenvalue=float(electron_eigenvalues[-1]),
        boson_moment_minimum_eigenvalue=moment_minimum,
        raw_electron_lower_barrier_minimum_eigenvalue=raw_minima[
            "electron_lower"
        ],
        raw_electron_upper_barrier_minimum_eigenvalue=raw_minima[
            "electron_upper"
        ],
        raw_boson_barrier_minimum_eigenvalue=raw_minima["boson"],
        corrected_electron_lower_barrier_minimum_eigenvalue=corrected_minima[
            "electron_lower"
        ],
        corrected_electron_upper_barrier_minimum_eigenvalue=corrected_minima[
            "electron_upper"
        ],
        corrected_boson_barrier_minimum_eigenvalue=corrected_minima["boson"],
        correction_coordinates=correction,
        correction_energy_flux=float(energy_gradient @ correction),
        constraint_count=len(normals),
        converged=False,
    )


def structured_full_state_joint_barrier_correction(
    state: FloatArray,
    derivative: FloatArray,
    parameters: DimerParameters,
    *,
    activation_margin: float = 0.0,
    target_flux: float = 0.0,
    barrier_rate: float = 5.0,
    energy_neutral: bool = True,
    cone_tolerance: float = 1e-9,
    projection_tolerance: float = 1e-12,
    maximum_projection_cycles: int = 10_000,
) -> StructuredFullStateBarrierCorrection:
    """Enforce the joint barriers with all 31 velocities available.

    Only the ``rho`` and ``(N, A)`` coordinates act on the three matrix
    barriers.  The remaining coordinates provide spectator directions that
    can cancel the correction's Eq. (22) energy flux.  This removes the
    artificial infeasibility that can occur when energy neutrality is imposed
    on the thirteen barrier-active controls alone.
    """

    if activation_margin < 0.0:
        raise ValueError("activation_margin must be nonnegative")
    if target_flux < 0.0:
        raise ValueError("target_flux must be nonnegative")
    if barrier_rate <= 0.0:
        raise ValueError("barrier_rate must be positive")
    if cone_tolerance <= 0.0 or projection_tolerance <= 0.0:
        raise ValueError("solver tolerances must be positive")
    if maximum_projection_cycles <= 0:
        raise ValueError("maximum_projection_cycles must be positive")

    state_array = np.asarray(state, dtype=float)
    derivative_array = np.asarray(derivative, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if state_array.shape != expected:
        raise ValueError(
            f"expected closed state shape {expected}, got {state_array.shape}"
        )
    if derivative_array.shape != expected:
        raise ValueError(
            "expected closed derivative shape "
            f"{expected}, got {derivative_array.shape}"
        )

    matrix_state = closed_scalar_to_matrix_state(state_array)
    electron = np.asarray(matrix_state.electron_density, dtype=complex)
    electron = 0.5 * (electron + electron.conjugate().T)
    electron_velocity = structured_electron_velocity_lift(
        derivative_array[:3]
    )
    electron_identity = np.eye(2, dtype=complex)

    moment = boson_moment_matrix(matrix_state)
    moment = 0.5 * (moment + moment.conjugate().T)
    boson_velocity = _boson_velocity_from_closed_derivative(derivative_array)
    boson_velocity = 0.5 * (
        boson_velocity + boson_velocity.conjugate().T
    )
    boson_identity = np.eye(4, dtype=complex)

    raw_barriers = {
        "electron_lower": electron_velocity
        + barrier_rate
        * (electron - activation_margin * electron_identity)
        - target_flux * electron_identity,
        "electron_upper": -electron_velocity
        + barrier_rate
        * (
            electron_identity
            - electron
            - activation_margin * electron_identity
        )
        - target_flux * electron_identity,
        "boson": boson_velocity
        + barrier_rate * (moment - activation_margin * boson_identity)
        - target_flux * boson_identity,
    }
    raw_barriers = {
        name: 0.5 * (value + value.conjugate().T)
        for name, value in raw_barriers.items()
    }
    raw_minima = {
        name: float(np.linalg.eigvalsh(value)[0])
        for name, value in raw_barriers.items()
    }
    electron_eigenvalues = np.linalg.eigvalsh(electron)
    moment_minimum = float(np.linalg.eigvalsh(moment)[0])
    energy_gradient = closed_state_correction_energy_gradient(
        state_array,
        parameters,
    )

    if min(raw_minima.values()) >= -cone_tolerance:
        return StructuredFullStateBarrierCorrection(
            electron_minimum_eigenvalue=float(electron_eigenvalues[0]),
            electron_maximum_eigenvalue=float(electron_eigenvalues[-1]),
            boson_moment_minimum_eigenvalue=moment_minimum,
            raw_electron_lower_barrier_minimum_eigenvalue=raw_minima[
                "electron_lower"
            ],
            raw_electron_upper_barrier_minimum_eigenvalue=raw_minima[
                "electron_upper"
            ],
            raw_boson_barrier_minimum_eigenvalue=raw_minima["boson"],
            corrected_electron_lower_barrier_minimum_eigenvalue=raw_minima[
                "electron_lower"
            ],
            corrected_electron_upper_barrier_minimum_eigenvalue=raw_minima[
                "electron_upper"
            ],
            corrected_boson_barrier_minimum_eigenvalue=raw_minima["boson"],
            correction_coordinates=np.zeros_like(state_array),
            correction_energy_flux=0.0,
            constraint_count=0,
            converged=True,
        )

    coordinate_basis = (
        _orthonormal_null_space(energy_gradient)
        if energy_neutral
        else np.eye(len(CLOSED_SCALAR_STATE_NAMES), dtype=float)
    )
    (
        correction,
        corrected_minima,
        converged,
        constraint_count,
    ) = _direct_joint_barrier_solve(
        raw_barriers,
        coordinate_basis,
        electron_coordinates=slice(0, 3),
        boson_coordinates=slice(7, 17),
        mode_response=_full_state_joint_mode_response_vector,
        cone_tolerance=cone_tolerance,
        projection_tolerance=projection_tolerance,
        maximum_projection_cycles=maximum_projection_cycles,
    )
    return StructuredFullStateBarrierCorrection(
        electron_minimum_eigenvalue=float(electron_eigenvalues[0]),
        electron_maximum_eigenvalue=float(electron_eigenvalues[-1]),
        boson_moment_minimum_eigenvalue=moment_minimum,
        raw_electron_lower_barrier_minimum_eigenvalue=raw_minima[
            "electron_lower"
        ],
        raw_electron_upper_barrier_minimum_eigenvalue=raw_minima[
            "electron_upper"
        ],
        raw_boson_barrier_minimum_eigenvalue=raw_minima["boson"],
        corrected_electron_lower_barrier_minimum_eigenvalue=(
            corrected_minima["electron_lower"]
        ),
        corrected_electron_upper_barrier_minimum_eigenvalue=(
            corrected_minima["electron_upper"]
        ),
        corrected_boson_barrier_minimum_eigenvalue=corrected_minima["boson"],
        correction_coordinates=correction,
        correction_energy_flux=float(energy_gradient @ correction),
        constraint_count=constraint_count,
        converged=converged,
    )


def structured_electron_phonon_barrier_correction(
    state: FloatArray,
    derivative: FloatArray,
    parameters: DimerParameters,
    *,
    activation_margin: float = 0.0,
    target_flux: float = 0.0,
    barrier_rate: float = 5.0,
    energy_neutral: bool = True,
    preserve_correlation_trace: bool = True,
    cone_tolerance: float = 1e-9,
    projection_tolerance: float = 1e-12,
    maximum_constraints: int = 256,
    maximum_projection_cycles: int = 10_000,
    solver: Literal["cutting_plane", "direct_eigenvalue"] = "cutting_plane",
    correction_metric: CorrectionMetric = "euclidean",
    halfspace_backend: HalfspaceBackend | None = None,
) -> StructuredElectronPhononBarrierCorrection:
    """Enforce the joint Gram cone containing the retained correlation ``C``.

    In addition to the lower and upper electronic-density barriers, this
    controller constrains the seven-by-seven fluctuation Gram matrix returned
    by :func:`electron_phonon_moment_matrix`.  Its off-diagonal block contains
    ``C``, so this condition detects correlation values that are incompatible
    with the simultaneously retained electronic and bosonic moments.

    The exact fixed-particle-number identity ``trace(C[q])=0`` is imposed on
    the corrected velocity by default.  The correction also has zero
    instantaneous Eq. (22) energy flux when ``energy_neutral=True``.
    """

    if activation_margin < 0.0:
        raise ValueError("activation_margin must be nonnegative")
    if target_flux < 0.0:
        raise ValueError("target_flux must be nonnegative")
    if barrier_rate <= 0.0:
        raise ValueError("barrier_rate must be positive")
    if cone_tolerance <= 0.0 or projection_tolerance <= 0.0:
        raise ValueError("solver tolerances must be positive")
    if maximum_constraints <= 0 or maximum_projection_cycles <= 0:
        raise ValueError("solver iteration limits must be positive")
    if solver not in ("cutting_plane", "direct_eigenvalue"):
        raise ValueError(
            "solver must be 'cutting_plane' or 'direct_eigenvalue'"
        )
    if correction_metric not in ("euclidean", "frobenius"):
        raise ValueError(
            "correction_metric must be 'euclidean' or 'frobenius'"
        )

    state_array = np.asarray(state, dtype=float)
    derivative_array = np.asarray(derivative, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if state_array.shape != expected:
        raise ValueError(
            f"expected closed state shape {expected}, got {state_array.shape}"
        )
    if derivative_array.shape != expected:
        raise ValueError(
            "expected closed derivative shape "
            f"{expected}, got {derivative_array.shape}"
        )

    matrix_state = closed_scalar_to_matrix_state(state_array)
    electron = np.asarray(matrix_state.electron_density, dtype=complex)
    electron = 0.5 * (electron + electron.conjugate().T)
    electron_velocity = structured_electron_velocity_lift(
        derivative_array[:3]
    )
    electron_identity = np.eye(2, dtype=complex)

    joint_moment = electron_phonon_moment_matrix(matrix_state)
    # ``structured_electron_phonon_moment_velocity_lift`` rebuilds the matrix
    # state from ``state_array`` on every call; ``matrix_state`` above is that
    # same object, so the derivative map is applied to it directly.
    joint_velocity = electron_phonon_moment_derivative(
        matrix_state,
        structured_closed_state_velocity_lift(derivative_array),
    )
    joint_identity = np.eye(joint_moment.shape[0], dtype=complex)

    raw_barriers = {
        "electron_lower": electron_velocity
        + barrier_rate
        * (electron - activation_margin * electron_identity)
        - target_flux * electron_identity,
        "electron_upper": -electron_velocity
        + barrier_rate
        * (
            electron_identity
            - electron
            - activation_margin * electron_identity
        )
        - target_flux * electron_identity,
        "electron_phonon": joint_velocity
        + barrier_rate
        * (joint_moment - activation_margin * joint_identity)
        - target_flux * joint_identity,
    }
    raw_barriers = {
        name: 0.5 * (value + value.conjugate().T)
        for name, value in raw_barriers.items()
    }
    raw_minima = {
        name: float(np.linalg.eigvalsh(value)[0])
        for name, value in raw_barriers.items()
    }

    equality_rows: list[FloatArray] = []
    equality_values: list[float] = []
    energy_gradient = closed_state_correction_energy_gradient(
        state_array,
        parameters,
    )
    if energy_neutral:
        equality_rows.append(energy_gradient)
        equality_values.append(0.0)
    if preserve_correlation_trace:
        for coordinate in (17, 18):
            row = np.zeros(len(CLOSED_SCALAR_STATE_NAMES), dtype=float)
            row[coordinate] = 1.0
            equality_rows.append(row)
            equality_values.append(
                -derivative_array[coordinate]
                - barrier_rate * state_array[coordinate]
            )

    equality_matrix = np.asarray(equality_rows, dtype=float).reshape(
        len(equality_rows),
        len(CLOSED_SCALAR_STATE_NAMES),
    )
    # Both correction metrics are fixed matrices, so their Cholesky factor and
    # the triangular solve against the identity are module constants evaluated
    # once at import instead of on every call.
    coordinates_from_metric = _METRIC_COORDINATE_MAPS[correction_metric]
    metric_equality_matrix = equality_matrix @ coordinates_from_metric
    particular, coordinate_basis = _affine_equality_parameterization(
        metric_equality_matrix,
        np.asarray(equality_values, dtype=float),
    )

    def correction_from_reduced(reduced: FloatArray) -> FloatArray:
        metric_coordinates = particular + coordinate_basis @ reduced
        return coordinates_from_metric @ metric_coordinates

    # The 31 lifted unit-coordinate matrix states are fixed, and the matrix
    # state built from ``state_array`` is already available, so the joint-Gram
    # basis is built by applying the derivative map 31 times to the two
    # existing objects instead of rebuilding both inside each of the 31 calls.
    joint_velocity_basis = np.asarray(
        [
            electron_phonon_moment_derivative(matrix_state, basis)
            for basis in _CLOSED_STATE_VELOCITY_BASIS
        ],
        dtype=complex,
    )
    sectors = ("electron_lower", "electron_upper", "electron_phonon")
    # The constraint function and the Jacobian are evaluated by SLSQP at the
    # same reduced point, and the cutting-plane loop's exit test re-reads the
    # matrices it just built.  The assembled matrices are cached on the exact
    # reduced point so each assembly happens once.
    barrier_cache: OrderedDict[bytes, dict[str, np.ndarray]] = OrderedDict()

    def _build_corrected_barriers(
        reduced: FloatArray,
    ) -> dict[str, np.ndarray]:
        correction = correction_from_reduced(reduced)
        electron_correction = structured_electron_velocity_lift(
            correction[:3]
        )
        joint_correction = np.tensordot(
            correction,
            joint_velocity_basis,
            axes=(0, 0),
        )
        return {
            "electron_lower": (
                raw_barriers["electron_lower"] + electron_correction
            ),
            "electron_upper": (
                raw_barriers["electron_upper"] - electron_correction
            ),
            "electron_phonon": (
                raw_barriers["electron_phonon"] + joint_correction
            ),
        }

    def corrected_barriers(reduced: FloatArray) -> dict[str, np.ndarray]:
        key = np.ascontiguousarray(reduced, dtype=float).tobytes()
        cached = barrier_cache.get(key)
        if cached is not None:
            return cached
        built = _build_corrected_barriers(reduced)
        barrier_cache[key] = built
        if len(barrier_cache) > 4:
            barrier_cache.popitem(last=False)
        return built

    def eigenvalue_constraints(reduced: FloatArray) -> FloatArray:
        return np.concatenate(
            [
                np.linalg.eigvalsh(matrix)
                for matrix in corrected_barriers(reduced).values()
            ]
        )

    def eigenvalue_jacobian(reduced: FloatArray) -> FloatArray:
        rows: list[FloatArray] = []
        matrices = corrected_barriers(reduced)
        for sector in sectors:
            _, eigenvectors = np.linalg.eigh(matrices[sector])
            for column in range(eigenvectors.shape[1]):
                mode = eigenvectors[:, column]
                response = np.zeros(
                    len(CLOSED_SCALAR_STATE_NAMES),
                    dtype=float,
                )
                if sector in ("electron_lower", "electron_upper"):
                    sign = 1.0 if sector == "electron_lower" else -1.0
                    response[:3] = sign * np.asarray(
                        [
                            np.real(np.vdot(mode, basis @ mode))
                            for basis in _STRUCTURED_ELECTRON_VELOCITY_BASIS
                        ],
                        dtype=float,
                    )
                else:
                    response = np.asarray(
                        [
                            np.real(np.vdot(mode, basis @ mode))
                            for basis in joint_velocity_basis
                        ],
                        dtype=float,
                    )
                rows.append(
                    coordinate_basis.T
                    @ coordinates_from_metric.T
                    @ response
                )
        return np.asarray(rows, dtype=float)

    reduced_dimension = coordinate_basis.shape[1]
    if solver == "direct_eigenvalue":
        SOLVER_COUNTERS.direct_eigenvalue_slsqp_solves += 1
        solution = minimize(
            lambda reduced: 0.5 * float(reduced @ reduced),
            np.zeros(reduced_dimension, dtype=float),
            jac=lambda reduced: reduced,
            constraints=(
                {
                    "type": "ineq",
                    "fun": eigenvalue_constraints,
                    "jac": eigenvalue_jacobian,
                },
            ),
            method="SLSQP",
            options={
                "ftol": projection_tolerance,
                "maxiter": maximum_projection_cycles,
                "disp": False,
            },
        )
        reduced = np.asarray(solution.x, dtype=float)
        constraint_count = int(
            sum(matrix.shape[0] for matrix in raw_barriers.values())
        )
    else:
        reduced = np.zeros(reduced_dimension, dtype=float)
        normals: list[FloatArray] = []
        offsets: list[float] = []
        projection_converged = True
        halfspace_warm_state = HalfspaceWarmState()
        while len(normals) < maximum_constraints:
            matrices = corrected_barriers(reduced)
            violations: list[tuple[float, FloatArray]] = []
            for sector in sectors:
                eigenvalues, eigenvectors = np.linalg.eigh(matrices[sector])
                for column, eigenvalue in enumerate(eigenvalues):
                    if eigenvalue >= -cone_tolerance:
                        continue
                    mode = eigenvectors[:, column]
                    response = np.zeros(
                        len(CLOSED_SCALAR_STATE_NAMES),
                        dtype=float,
                    )
                    if sector in ("electron_lower", "electron_upper"):
                        sign = 1.0 if sector == "electron_lower" else -1.0
                        response[:3] = sign * np.asarray(
                            [
                                np.real(np.vdot(mode, basis @ mode))
                                for basis in (
                                    _STRUCTURED_ELECTRON_VELOCITY_BASIS
                                )
                            ],
                            dtype=float,
                        )
                    else:
                        response = np.asarray(
                            [
                                np.real(np.vdot(mode, basis @ mode))
                                for basis in joint_velocity_basis
                            ],
                            dtype=float,
                        )
                    normal = (
                        coordinate_basis.T
                        @ coordinates_from_metric.T
                        @ response
                    )
                    if np.linalg.norm(normal) <= 1e-14:
                        projection_converged = False
                        continue
                    offset = float(normal @ reduced - eigenvalue)
                    violations.append((float(eigenvalue), normal))
                    normals.append(normal)
                    offsets.append(offset)
                    if len(normals) >= maximum_constraints:
                        break
                if len(normals) >= maximum_constraints:
                    break

            if not violations:
                break
            reduced, projection_converged = _project_origin_onto_halfspaces(
                normals,
                offsets,
                tolerance=projection_tolerance,
                maximum_cycles=maximum_projection_cycles,
                backend=halfspace_backend,
                warm_state=halfspace_warm_state,
            )
            if not projection_converged:
                break
        constraint_count = len(normals)

    correction = correction_from_reduced(reduced)
    corrected = corrected_barriers(reduced)
    corrected_minima = {
        sector: float(np.linalg.eigvalsh(corrected[sector])[0])
        for sector in sectors
    }
    if (
        solver == "cutting_plane"
        and min(corrected_minima.values()) < -cone_tolerance
    ):
        # A nearly degenerate PSD boundary can stall the accumulated linear
        # projection.  The complete eigenvalue solve is slower but gives a
        # robust fallback from that warm start.
        SOLVER_COUNTERS.cutting_plane_fallback_slsqp_solves += 1
        fallback = minimize(
            lambda point: 0.5 * float(point @ point),
            reduced,
            jac=lambda point: point,
            constraints=(
                {
                    "type": "ineq",
                    "fun": eigenvalue_constraints,
                    "jac": eigenvalue_jacobian,
                },
            ),
            method="SLSQP",
            options={
                "ftol": projection_tolerance,
                "maxiter": maximum_projection_cycles,
                "disp": False,
            },
        )
        reduced = np.asarray(fallback.x, dtype=float)
        correction = correction_from_reduced(reduced)
        corrected = corrected_barriers(reduced)
        corrected_minima = {
            sector: float(np.linalg.eigvalsh(corrected[sector])[0])
            for sector in sectors
        }
        constraint_count += int(
            sum(matrix.shape[0] for matrix in raw_barriers.values())
        )
    equality_error = (
        float(
            np.linalg.norm(
                equality_matrix @ correction
                - np.asarray(equality_values, dtype=float)
            )
        )
        if equality_rows
        else 0.0
    )
    converged = bool(
        min(corrected_minima.values()) >= -cone_tolerance
        and equality_error <= 10.0 * projection_tolerance
    )
    electron_eigenvalues = np.linalg.eigvalsh(electron)
    corrected_trace_velocity = complex(
        derivative_array[17] + correction[17],
        derivative_array[18] + correction[18],
    )
    return StructuredElectronPhononBarrierCorrection(
        electron_minimum_eigenvalue=float(electron_eigenvalues[0]),
        electron_maximum_eigenvalue=float(electron_eigenvalues[-1]),
        joint_moment_minimum_eigenvalue=float(
            np.linalg.eigvalsh(joint_moment)[0]
        ),
        raw_electron_lower_barrier_minimum_eigenvalue=raw_minima[
            "electron_lower"
        ],
        raw_electron_upper_barrier_minimum_eigenvalue=raw_minima[
            "electron_upper"
        ],
        raw_joint_barrier_minimum_eigenvalue=raw_minima[
            "electron_phonon"
        ],
        corrected_electron_lower_barrier_minimum_eigenvalue=(
            corrected_minima["electron_lower"]
        ),
        corrected_electron_upper_barrier_minimum_eigenvalue=(
            corrected_minima["electron_upper"]
        ),
        corrected_joint_barrier_minimum_eigenvalue=corrected_minima[
            "electron_phonon"
        ],
        correction_coordinates=correction,
        correction_energy_flux=float(energy_gradient @ correction),
        corrected_correlation_trace_velocity=corrected_trace_velocity,
        constraint_count=constraint_count,
        converged=converged,
    )


def structured_boson_barrier_correction(
    state: FloatArray,
    derivative: FloatArray,
    *,
    activation_margin: float = 0.0,
    target_flux: float = 0.0,
    barrier_rate: float = 5.0,
    energy_neutral: bool = False,
    cone_tolerance: float = 1e-11,
    projection_tolerance: float = 1e-12,
    maximum_constraints: int = 32,
    maximum_projection_cycles: int = 10_000,
    halfspace_backend: HalfspaceBackend | None = None,
) -> StructuredConeBarrierCorrection:
    """Enforce the full structured matrix control-barrier inequality.

    The returned least-norm correction is computed by constraint generation
    for the semidefinite condition

    ``D + S(y) + barrier_rate * (M - activation_margin * I)
       - target_flux * I >= 0``.

    Each violated matrix eigenvector supplies a linear half-space in the ten
    real correction coordinates.  A small convex quadratic program gives the
    Euclidean minimum-norm point for the accumulated half-spaces.  Iteration stops only
    when the complete 4-by-4 matrix, rather than one selected eigenmode, passes
    the cone tolerance.  This remains well-defined at eigenvalue degeneracies.
    With ``energy_neutral=True``, the optimization is restricted to
    ``dN00 + dN11 = 0``.  Because the direct correction changes no other
    sector, that condition makes its instantaneous contribution to the
    time-independent Eq. (22) energy exactly zero.
    """

    if activation_margin < 0.0:
        raise ValueError("activation_margin must be nonnegative")
    if target_flux < 0.0:
        raise ValueError("target_flux must be nonnegative")
    if barrier_rate <= 0.0:
        raise ValueError("barrier_rate must be positive")
    if cone_tolerance <= 0.0 or projection_tolerance <= 0.0:
        raise ValueError("solver tolerances must be positive")
    if maximum_constraints <= 0 or maximum_projection_cycles <= 0:
        raise ValueError("solver iteration limits must be positive")

    state_array = np.asarray(state, dtype=float)
    derivative_array = np.asarray(derivative, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if state_array.shape != expected:
        raise ValueError(
            f"expected closed state shape {expected}, got {state_array.shape}"
        )
    if derivative_array.shape != expected:
        raise ValueError(
            "expected closed derivative shape "
            f"{expected}, got {derivative_array.shape}"
        )

    moment = boson_moment_matrix(
        closed_scalar_to_matrix_state(state_array)
    )
    moment = 0.5 * (moment + moment.conjugate().T)
    velocity = _boson_velocity_from_closed_derivative(derivative_array)
    velocity = 0.5 * (velocity + velocity.conjugate().T)
    identity = np.eye(moment.shape[0], dtype=complex)
    raw_barrier = (
        velocity
        + barrier_rate * (moment - activation_margin * identity)
        - target_flux * identity
    )
    raw_barrier = 0.5 * (
        raw_barrier + raw_barrier.conjugate().T
    )
    raw_minimum = float(np.linalg.eigvalsh(raw_barrier)[0])
    moment_minimum = float(np.linalg.eigvalsh(moment)[0])
    if raw_minimum >= -cone_tolerance:
        return StructuredConeBarrierCorrection(
            moment_minimum_eigenvalue=moment_minimum,
            raw_barrier_minimum_eigenvalue=raw_minimum,
            corrected_barrier_minimum_eigenvalue=raw_minimum,
            correction_coordinates=np.zeros(10, dtype=float),
            constraint_count=0,
            converged=True,
        )

    if energy_neutral:
        coordinate_basis = np.zeros((10, 9), dtype=float)
        coordinate_basis[0, 0] = 1.0 / np.sqrt(2.0)
        coordinate_basis[1, 0] = -1.0 / np.sqrt(2.0)
        coordinate_basis[2:, 1:] = np.eye(8, dtype=float)
    else:
        coordinate_basis = np.eye(10, dtype=float)

    normals: list[FloatArray] = []
    offsets: list[float] = []
    reduced_correction = np.zeros(coordinate_basis.shape[1], dtype=float)
    correction = coordinate_basis @ reduced_correction
    projection_converged = True
    corrected_minimum = raw_minimum
    halfspace_warm_state = HalfspaceWarmState()
    for _ in range(maximum_constraints):
        corrected_barrier = (
            raw_barrier
            + structured_boson_velocity_lift(correction)
        )
        eigenvalues, eigenvectors = np.linalg.eigh(corrected_barrier)
        corrected_minimum = float(eigenvalues[0])
        if corrected_minimum >= -cone_tolerance:
            return StructuredConeBarrierCorrection(
                moment_minimum_eigenvalue=moment_minimum,
                raw_barrier_minimum_eigenvalue=raw_minimum,
                corrected_barrier_minimum_eigenvalue=corrected_minimum,
                correction_coordinates=correction,
                constraint_count=len(normals),
                converged=projection_converged,
            )

        violated_mode = eigenvectors[:, 0]
        response = (
            coordinate_basis.T @ _mode_response_vector(violated_mode)
        )
        response_norm_squared = float(response @ response)
        if response_norm_squared <= 1e-14:
            break
        normals.append(response)
        offsets.append(
            -float(
                np.real(
                    np.vdot(
                        violated_mode,
                        raw_barrier @ violated_mode,
                    )
                )
            )
        )
        reduced_correction, projection_converged = (
            _project_origin_onto_halfspaces(
                normals,
                offsets,
                tolerance=projection_tolerance,
                maximum_cycles=maximum_projection_cycles,
                backend=halfspace_backend,
                warm_state=halfspace_warm_state,
            )
        )
        correction = coordinate_basis @ reduced_correction

    corrected_barrier = (
        raw_barrier + structured_boson_velocity_lift(correction)
    )
    corrected_minimum = float(np.linalg.eigvalsh(corrected_barrier)[0])
    if corrected_minimum < 0.0 and not energy_neutral:
        # ``dN = shift * I`` lifts to ``shift * I_4`` and is therefore a
        # structure-preserving feasibility fallback.  Reaching it means the
        # cutting-plane tolerance was not met, so ``converged`` remains false.
        identity_shift = -corrected_minimum + cone_tolerance
        correction = correction.copy()
        correction[0] += identity_shift
        correction[1] += identity_shift
        corrected_barrier = (
            raw_barrier + structured_boson_velocity_lift(correction)
        )
        corrected_minimum = float(np.linalg.eigvalsh(corrected_barrier)[0])

    return StructuredConeBarrierCorrection(
        moment_minimum_eigenvalue=moment_minimum,
        raw_barrier_minimum_eigenvalue=raw_minimum,
        corrected_barrier_minimum_eigenvalue=corrected_minimum,
        correction_coordinates=correction,
        constraint_count=len(normals),
        converged=False,
    )


def structured_boson_cone_correction(
    state: FloatArray,
    derivative: FloatArray,
    *,
    activation_margin: float = 0.0,
    target_flux: float = 0.0,
    barrier_rate: float | None = None,
    compliance_tolerance: float = 1e-14,
) -> StructuredConeCorrection:
    """Return the least-norm structured velocity correction at the cone edge.

    Let ``v`` be the normalized eigenvector of the smallest eigenvalue of the
    bosonic moment matrix.  If that eigenvalue is within ``activation_margin``
    and its uncorrected directional flux is below ``target_flux``, this solves

    ``min ||y||_2  subject to  v†(D + S(y))v >= target_flux``.

    For the identity coordinate metric, the closed-form solution is parallel
    to the response vector ``a_j = v†S(e_j)v``.  This is a tangent-velocity
    correction; it does not project an already unphysical state back into the
    cone.  Supplying ``barrier_rate`` replaces the discontinuous edge switch
    by the control-barrier condition
    ``dot(lambda_min) >= target_flux + barrier_rate *
    (activation_margin - lambda_min)``.
    """

    if activation_margin < 0.0:
        raise ValueError("activation_margin must be nonnegative")
    if target_flux < 0.0:
        raise ValueError("target_flux must be nonnegative")
    if barrier_rate is not None and barrier_rate <= 0.0:
        raise ValueError("barrier_rate must be positive when supplied")
    if compliance_tolerance <= 0.0:
        raise ValueError("compliance_tolerance must be positive")

    state_array = np.asarray(state, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if state_array.shape != expected:
        raise ValueError(
            f"expected closed state shape {expected}, got {state_array.shape}"
        )
    derivative_array = np.asarray(derivative, dtype=float)
    if derivative_array.shape != expected:
        raise ValueError(
            "expected closed derivative shape "
            f"{expected}, got {derivative_array.shape}"
        )

    moment = boson_moment_matrix(
        closed_scalar_to_matrix_state(state_array)
    )
    moment = 0.5 * (moment + moment.conjugate().T)
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    minimum_mode = eigenvectors[:, 0]

    velocity = _boson_velocity_from_closed_derivative(derivative_array)
    velocity = 0.5 * (velocity + velocity.conjugate().T)
    raw_flux = float(
        np.real(np.vdot(minimum_mode, velocity @ minimum_mode))
    )
    response = np.array(
        [
            np.real(
                np.vdot(minimum_mode, basis @ minimum_mode)
            )
            for basis in _STRUCTURED_VELOCITY_BASIS
        ],
        dtype=float,
    )

    required_flux = float(target_flux)
    if barrier_rate is None:
        active = bool(
            eigenvalues[0] <= activation_margin and raw_flux < required_flux
        )
    else:
        required_flux += float(
            barrier_rate * (activation_margin - eigenvalues[0])
        )
        active = bool(raw_flux < required_flux)
    correction = np.zeros(10, dtype=float)
    if active:
        compliance = float(response @ response)
        if compliance <= compliance_tolerance:
            raise RuntimeError(
                "the active bosonic mode has no resolvable structured "
                "velocity response"
            )
        correction = (
            (required_flux - raw_flux) / compliance
        ) * response

    corrected_flux = float(raw_flux + response @ correction)
    return StructuredConeCorrection(
        minimum_eigenvalue=float(eigenvalues[0]),
        next_eigenvalue=float(eigenvalues[1]),
        raw_flux=raw_flux,
        corrected_flux=corrected_flux,
        target_flux=required_flux,
        response_vector=response,
        correction_coordinates=correction,
        active=active,
    )


def closed_cone_projected_rhs(
    parameters: DimerParameters,
    initial_state: FloatArray,
    *,
    activation_margin: float = 0.0,
    target_flux: float = 0.0,
    barrier_rate: float | None = 5.0,
    energy_neutral: bool = False,
    subtract_initial_residual: bool = True,
    halfspace_backend: HalfspaceBackend | None = None,
    memoize_corrections: bool = False,
) -> RhsFunction:
    """Return the 31D RHS with a structured bosonic tangent correction.

    By default this wraps the Eq. (112) residual-subtracted closure used by the
    pinned strong-coupling diagnostic.  Only derivative indices 7--16 can be
    changed.  The default control-barrier rate avoids the numerical chattering
    caused by a discontinuous correction activated only at the boundary.

    ``memoize_corrections`` caches the correction on the exact
    ``(state, derivative)`` byte key so a repeated evaluation at an identical
    argument reuses the solved coordinates instead of resolving.  It is off by
    default: on the pinned strong-coupling protocol DOP853 evaluates this
    right-hand side at a distinct state every time, so the cache only adds
    bookkeeping.
    """

    initial = np.asarray(initial_state, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if initial.shape != expected:
        raise ValueError(
            f"expected closed initial state shape {expected}, got {initial.shape}"
        )
    initial = initial.copy()

    if subtract_initial_residual:
        undriven = replace(parameters, drive_amplitude=0.0)
        residual = closed_scalar_rhs(0.0, initial, undriven)
    else:
        residual = np.zeros_like(initial)

    memo = _CorrectionMemo() if memoize_corrections else None

    def rhs(time: float, state: FloatArray) -> FloatArray:
        derivative = closed_scalar_rhs(time, state, parameters) - residual
        correction_coordinates = (
            memo.lookup(state, derivative) if memo is not None else None
        )
        if correction_coordinates is None:
            if barrier_rate is None:
                correction_coordinates = (
                    structured_boson_cone_correction(
                        state,
                        derivative,
                        activation_margin=activation_margin,
                        target_flux=target_flux,
                    ).correction_coordinates
                )
            else:
                correction_coordinates = (
                    structured_boson_barrier_correction(
                        state,
                        derivative,
                        activation_margin=activation_margin,
                        target_flux=target_flux,
                        barrier_rate=barrier_rate,
                        energy_neutral=energy_neutral,
                        halfspace_backend=halfspace_backend,
                    ).correction_coordinates
                )
            if memo is not None:
                memo.store(state, derivative, correction_coordinates)
        corrected = derivative.copy()
        corrected[7:17] += correction_coordinates
        return corrected

    return rhs


def closed_joint_cone_projected_rhs(
    parameters: DimerParameters,
    initial_state: FloatArray,
    *,
    activation_margin: float = 0.0,
    target_flux: float = 0.0,
    barrier_rate: float = 5.0,
    energy_neutral: bool = True,
    subtract_initial_residual: bool = True,
    require_convergence: bool = True,
    cone_tolerance: float = 1e-11,
    projection_tolerance: float = 1e-12,
    barrier_solver: Literal[
        "cutting_plane", "direct_eigenvalue"
    ] = "cutting_plane",
    halfspace_backend: HalfspaceBackend | None = None,
    cut_harvest: CutHarvest = "single_mode",
    memoize_corrections: bool = False,
) -> RhsFunction:
    """Return the 31D RHS with one joint electronic--bosonic barrier.

    The correction changes only closed-state derivative indices 0--2 and
    7--16.  The first block is a traceless Hermitian electronic velocity; the
    second is the structure-preserving normal/anomalous boson velocity.  The
    remaining eighteen equations are untouched.
    """

    initial = np.asarray(initial_state, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if initial.shape != expected:
        raise ValueError(
            f"expected closed initial state shape {expected}, got {initial.shape}"
        )
    initial = initial.copy()

    if subtract_initial_residual:
        undriven = replace(parameters, drive_amplitude=0.0)
        residual = closed_scalar_rhs(0.0, initial, undriven)
    else:
        residual = np.zeros_like(initial)

    memo = _CorrectionMemo() if memoize_corrections else None

    def rhs(time: float, state: FloatArray) -> FloatArray:
        derivative = closed_scalar_rhs(time, state, parameters) - residual
        coordinates = (
            memo.lookup(state, derivative) if memo is not None else None
        )
        if coordinates is None:
            result = structured_joint_barrier_correction(
                state,
                derivative,
                parameters,
                activation_margin=activation_margin,
                target_flux=target_flux,
                barrier_rate=barrier_rate,
                energy_neutral=energy_neutral,
                cone_tolerance=cone_tolerance,
                projection_tolerance=projection_tolerance,
                solver=barrier_solver,
                halfspace_backend=halfspace_backend,
                cut_harvest=cut_harvest,
            )
            if require_convergence and not result.converged:
                raise RuntimeError(
                    "joint electronic--bosonic barrier solve did not converge: "
                    f"lower={result.corrected_electron_lower_barrier_minimum_eigenvalue}, "
                    f"upper={result.corrected_electron_upper_barrier_minimum_eigenvalue}, "
                    f"boson={result.corrected_boson_barrier_minimum_eigenvalue}"
                )
            coordinates = np.asarray(
                result.correction_coordinates,
                dtype=float,
            )
            if memo is not None:
                memo.store(state, derivative, coordinates)
        corrected = derivative.copy()
        corrected[:3] += coordinates[:3]
        corrected[7:17] += coordinates[3:]
        return corrected

    return rhs


def closed_full_state_joint_cone_projected_rhs(
    parameters: DimerParameters,
    initial_state: FloatArray,
    *,
    activation_margin: float = 0.0,
    target_flux: float = 0.0,
    barrier_rate: float = 5.0,
    energy_neutral: bool = True,
    subtract_initial_residual: bool = False,
    require_convergence: bool = True,
    cone_tolerance: float = 1e-9,
    projection_tolerance: float = 1e-12,
    memoize_corrections: bool = False,
) -> RhsFunction:
    """Return the 31D RHS with the full-state joint matrix barrier."""

    initial = np.asarray(initial_state, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if initial.shape != expected:
        raise ValueError(
            f"expected closed initial state shape {expected}, got {initial.shape}"
        )
    initial = initial.copy()

    if subtract_initial_residual:
        undriven = replace(parameters, drive_amplitude=0.0)
        residual = closed_scalar_rhs(0.0, initial, undriven)
    else:
        residual = np.zeros_like(initial)

    memo = _CorrectionMemo() if memoize_corrections else None

    def rhs(time: float, state: FloatArray) -> FloatArray:
        derivative = closed_scalar_rhs(time, state, parameters) - residual
        coordinates = (
            memo.lookup(state, derivative) if memo is not None else None
        )
        if coordinates is None:
            result = structured_full_state_joint_barrier_correction(
                state,
                derivative,
                parameters,
                activation_margin=activation_margin,
                target_flux=target_flux,
                barrier_rate=barrier_rate,
                energy_neutral=energy_neutral,
                cone_tolerance=cone_tolerance,
                projection_tolerance=projection_tolerance,
            )
            if require_convergence and not result.converged:
                raise RuntimeError(
                    "full-state joint barrier solve did not converge: "
                    f"lower={result.corrected_electron_lower_barrier_minimum_eigenvalue}, "
                    f"upper={result.corrected_electron_upper_barrier_minimum_eigenvalue}, "
                    f"boson={result.corrected_boson_barrier_minimum_eigenvalue}"
                )
            coordinates = np.asarray(
                result.correction_coordinates,
                dtype=float,
            )
            if memo is not None:
                memo.store(state, derivative, coordinates)
        return derivative + coordinates

    return rhs


def closed_electron_phonon_cone_projected_rhs(
    parameters: DimerParameters,
    initial_state: FloatArray,
    *,
    activation_margin: float = 0.0,
    target_flux: float = 0.0,
    barrier_rate: float = 5.0,
    energy_neutral: bool = True,
    preserve_correlation_trace: bool = True,
    subtract_initial_residual: bool = False,
    require_convergence: bool = True,
    cone_tolerance: float = 1e-9,
    projection_tolerance: float = 1e-12,
    correction_metric: CorrectionMetric = "euclidean",
    halfspace_backend: HalfspaceBackend | None = None,
    memoize_corrections: bool = False,
) -> RhsFunction:
    """Return the 31D RHS constrained by the joint ``(rho,N,A,C)`` cone.

    ``halfspace_backend`` names the minimum-norm-point solver used inside the
    constraint-generation loop; ``memoize_corrections`` caches the solved
    correction on the exact ``(state, derivative)`` byte key.  Both leave the
    declared correction unchanged and both default to the recorded behaviour.
    """

    initial = np.asarray(initial_state, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if initial.shape != expected:
        raise ValueError(
            f"expected closed initial state shape {expected}, got {initial.shape}"
        )
    initial = initial.copy()

    if subtract_initial_residual:
        undriven = replace(parameters, drive_amplitude=0.0)
        residual = closed_scalar_rhs(0.0, initial, undriven)
    else:
        residual = np.zeros_like(initial)

    memo = _CorrectionMemo() if memoize_corrections else None

    def rhs(time: float, state: FloatArray) -> FloatArray:
        derivative = closed_scalar_rhs(time, state, parameters) - residual
        coordinates = (
            memo.lookup(state, derivative) if memo is not None else None
        )
        if coordinates is None:
            result = structured_electron_phonon_barrier_correction(
                state,
                derivative,
                parameters,
                activation_margin=activation_margin,
                target_flux=target_flux,
                barrier_rate=barrier_rate,
                energy_neutral=energy_neutral,
                preserve_correlation_trace=preserve_correlation_trace,
                cone_tolerance=cone_tolerance,
                projection_tolerance=projection_tolerance,
                correction_metric=correction_metric,
                halfspace_backend=halfspace_backend,
            )
            if require_convergence and not result.converged:
                raise RuntimeError(
                    "electron--phonon Gram barrier solve did not converge: "
                    f"lower={result.corrected_electron_lower_barrier_minimum_eigenvalue}, "
                    f"upper={result.corrected_electron_upper_barrier_minimum_eigenvalue}, "
                    f"joint={result.corrected_joint_barrier_minimum_eigenvalue}"
                )
            coordinates = np.asarray(
                result.correction_coordinates,
                dtype=float,
            )
            if memo is not None:
                memo.store(state, derivative, coordinates)
        return derivative + coordinates

    return rhs
