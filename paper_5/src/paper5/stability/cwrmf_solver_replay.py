"""Offline replay bench for the recorded carried-witness conic instances.

Each instance recorded by ``cwrmf_solver_instances`` is solved again by a
named candidate -- a declared Clarabel setting set, a warm-started SCS solve,
or a reduced bundle -- and scored on the same quantities the online guard
uses.  Per instance the bench reports the replay wall time, the solver
status, the objective ``0.5 x^T P x + q^T x``, the maximum cone violation of
``b - A x`` over the declared cone blocks, the minimum eigenvalue of the
reconstructed 62-row Gram of the candidate completion velocity, and the
maximum absolute difference of that candidate velocity from the one the
recorded solution implies.

The acceptance test is the one the bundle tier applies to the full 62-row
Gram: the Gram minimum must reach ``schur_safety_margin - 1e-13`` and the
normalized residual must not exceed ``10 x solver_tolerance``.  It is
evaluated twice, once with the strict profile's constants
(``schur_safety_margin = 2e-12``, ``solver_tolerance = 1e-10``) and once with
the balanced profile's (``1e-10``, ``1e-8``).  The recorded operands are the
strict profile's, so the balanced evaluation reads the balanced acceptance
constants applied to a strict-profile instance, not a balanced-profile
rollout.

Nothing in this module runs inside a rollout.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import clarabel
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .cwrmf_solver_instances import (
    ConicInstance,
    build_cones,
    build_solver_settings,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

#: Acceptance constants of the two named numerical profiles, read from
#: ``cwrmf_settings_for_profile``.  ``(schur_safety_margin, solver_tolerance)``.
PROFILE_ACCEPTANCE = {
    "strict": (2e-12, 1e-10),
    "balanced": (1e-10, 1e-8),
}


def cone_blocks(
    instance: ConicInstance,
) -> list[tuple[str, int, slice]]:
    """Return ``(cone name, dimension, row slice)`` for each declared cone."""

    blocks: list[tuple[str, int, slice]] = []
    offset = 0
    for name, dimension in instance.cone_specification:
        if name == "PSDTriangleConeT":
            rows = dimension * (dimension + 1) // 2
        else:
            rows = dimension
        blocks.append((name, dimension, slice(offset, offset + rows)))
        offset += rows
    if offset != instance.constraint_count:
        raise ValueError(
            "cone specification does not cover the recorded cone rows"
        )
    return blocks


def unpack_svec_upper(packed: FloatArray, size: int) -> FloatArray:
    """Invert the column-major upper-triangle svec packing with sqrt(2)."""

    matrix = np.zeros((size, size), dtype=float)
    index = 0
    scale = np.sqrt(2.0)
    for column in range(size):
        for row in range(column + 1):
            value = packed[index]
            if row == column:
                matrix[row, column] = value
            else:
                matrix[row, column] = value / scale
                matrix[column, row] = value / scale
            index += 1
    return matrix


def pack_svec_upper(matrix: FloatArray) -> FloatArray:
    """Pack a symmetric matrix the way the carried-witness call sites do."""

    size = matrix.shape[0]
    packed = np.empty(size * (size + 1) // 2, dtype=float)
    index = 0
    scale = np.sqrt(2.0)
    for column in range(size):
        for row in range(column + 1):
            value = matrix[row, column]
            packed[index] = value if row == column else scale * value
            index += 1
    return packed


def maximum_cone_violation(
    instance: ConicInstance, solution: FloatArray
) -> float:
    """Return the largest violation of ``b - A x`` over the cone blocks.

    Zero and nonnegative blocks are read entrywise, a second-order block by
    ``||s_rest|| - s_0``, and a positive-semidefinite triangle block by the
    negated minimum eigenvalue of its unpacked matrix.  A nonpositive value
    means every block lies inside its cone.
    """

    slack = instance.cone_rhs - instance.cone_matrix @ np.asarray(
        solution, dtype=float
    )
    violation = -np.inf
    for name, dimension, rows in cone_blocks(instance):
        block = slack[rows]
        if name == "ZeroConeT":
            violation = max(violation, float(np.max(np.abs(block))))
        elif name == "NonnegativeConeT":
            violation = max(violation, float(-np.min(block)))
        elif name == "SecondOrderConeT":
            violation = max(
                violation,
                float(np.linalg.norm(block[1:]) - block[0]),
            )
        elif name == "PSDTriangleConeT":
            matrix = unpack_svec_upper(block, dimension)
            violation = max(
                violation, float(-np.linalg.eigvalsh(matrix)[0])
            )
        else:
            raise ValueError(f"unhandled cone: {name}")
    return float(violation)


def objective_value(
    instance: ConicInstance, solution: FloatArray
) -> float:
    """Return ``0.5 x^T P x + q^T x`` for a candidate ``x``."""

    vector = np.asarray(solution, dtype=float)
    quadratic = float(vector @ (instance.objective_matrix @ vector))
    return 0.5 * quadratic + float(instance.objective_vector @ vector)


@dataclass(frozen=True)
class InstanceScore:
    """Every quantity the bench reports for one candidate on one instance."""

    instance_index: int
    tier: str
    atom_index: int
    candidate: str
    wall_seconds: float
    status: str
    iterations: int
    objective: float
    maximum_cone_violation: float
    minimum_gram_eigenvalue: float
    maximum_velocity_difference: float
    bitwise_equal_to_recorded: bool
    strict_accepted: bool
    balanced_accepted: bool
    strict_normalized_residual: float
    balanced_normalized_residual: float
    #: The same two acceptance tests applied to the solution Clarabel
    #: actually returned online, so a candidate can be compared against the
    #: verdict the rollout took, not against acceptance in the abstract.
    recorded_strict_accepted: bool
    recorded_balanced_accepted: bool
    recorded_minimum_gram_eigenvalue: float
    #: The balanced acceptance test with the balanced profile's
    #: ``psd_inflation`` substituted for the strict value the operands carry.
    balanced_shifted_accepted: bool
    recorded_balanced_shifted_accepted: bool


#: ``psd_inflation`` of the two named numerical profiles.  The recorded
#: operands carry the strict value, so evaluating the balanced acceptance
#: test on them compares a ``1e-10`` safety margin against a Gram inflated by
#: only ``1e-11``.  ``BALANCED_INFLATION_SHIFT`` is the difference, and the
#: bench reports the balanced verdict both without it and with it.
PROFILE_PSD_INFLATION = {"strict": 1e-11, "balanced": 1e-8}
BALANCED_INFLATION_SHIFT = (
    PROFILE_PSD_INFLATION["balanced"] - PROFILE_PSD_INFLATION["strict"]
)


def acceptance(
    instance: ConicInstance,
    coefficients: ComplexArray,
    solution: FloatArray,
    *,
    profile: str,
    primal_residual: float,
    dual_residual: float,
    inflation_shift: float = 0.0,
) -> tuple[bool, float, float]:
    """Apply a profile's full-Gram acceptance test to a candidate solution.

    ``inflation_shift`` is added to the diagonal of the reconstructed Gram
    before its minimum eigenvalue is read, which is how the balanced
    profile's larger ``psd_inflation`` is substituted for the strict value
    the recorded operands carry.

    Returns ``(accepted, normalized residual, minimum Gram eigenvalue)``.
    """

    safety, tolerance = PROFILE_ACCEPTANCE[profile]
    gram = instance.reconstructed_gram(coefficients, solution)
    if inflation_shift:
        gram = gram + inflation_shift * np.eye(
            gram.shape[0], dtype=gram.dtype
        )
    minimum = float(np.linalg.eigvalsh(gram)[0])
    candidate = instance.candidate_velocity(solution)
    radial_residual = max(
        0.0,
        float(np.dot(candidate, candidate))
        - instance.completion_radius_squared,
    )
    normalized_residual = max(
        0.0,
        safety - minimum,
        radial_residual,
        float(primal_residual),
        float(dual_residual),
    ) / (1.0 + abs(minimum))
    accepted = bool(
        minimum >= safety - 1e-13
        and normalized_residual <= 10.0 * tolerance
    )
    return accepted, float(normalized_residual), minimum


@dataclass(frozen=True)
class CandidateSolution:
    """What a candidate returns for one instance."""

    solution: FloatArray
    status: str
    iterations: int
    wall_seconds: float
    primal_residual: float
    dual_residual: float


@dataclass(frozen=True)
class ClarabelCandidate:
    """A declared Clarabel setting set, replayed on the recorded operands."""

    name: str
    description: str
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def solve(
        self,
        instance: ConicInstance,
        previous: FloatArray | None = None,
    ) -> CandidateSolution:
        settings = build_solver_settings(
            instance.solver_settings, self.overrides
        )
        cones = build_cones(instance.cone_specification)
        started = time.perf_counter()
        solution = clarabel.DefaultSolver(
            instance.objective_matrix,
            instance.objective_vector,
            instance.cone_matrix,
            instance.cone_rhs,
            cones,
            settings,
        ).solve()
        elapsed = time.perf_counter() - started
        returned = (
            np.zeros(instance.variable_count)
            if solution.x is None
            else np.asarray(solution.x, dtype=float)
        )
        return CandidateSolution(
            solution=returned,
            status=str(solution.status),
            iterations=int(solution.iterations),
            wall_seconds=elapsed,
            primal_residual=float(solution.r_prim),
            dual_residual=float(solution.r_dual),
        )


def tolerance_overrides(decades: int) -> dict[str, float]:
    """Return the three Clarabel tolerances loosened by ``decades``."""

    factor = 10.0**decades
    return {
        "tol_gap_abs_factor": factor,
        "tol_gap_rel_factor": factor,
        "tol_feas_factor": factor,
    }


@dataclass(frozen=True)
class ToleranceCandidate:
    """Loosen the recorded tolerances by a declared number of decades."""

    name: str
    description: str
    decades: int

    def solve(
        self,
        instance: ConicInstance,
        previous: FloatArray | None = None,
    ) -> CandidateSolution:
        factor = 10.0**self.decades
        overrides = {
            "tol_gap_abs": instance.solver_settings["tol_gap_abs"] * factor,
            "tol_gap_rel": instance.solver_settings["tol_gap_rel"] * factor,
            "tol_feas": instance.solver_settings["tol_feas"] * factor,
        }
        return ClarabelCandidate(
            self.name, self.description, overrides
        ).solve(instance, previous)


@dataclass(frozen=True)
class ReducedBundleCandidate:
    """Solve a smaller bundle, then score on the full 62-row Gram.

    The recorded positive-semidefinite block carries the compressed bundle
    base ``B`` and its coefficient matrices, both scaled by the call site's
    homogeneous cone scale.  This candidate eigendecomposes ``B``, keeps only
    the directions whose eigenvalue is at or below ``eigenvalue_threshold``
    times the largest eigenvalue of ``B``, and re-expresses the whole
    positive-semidefinite block in that smaller working set.  The
    second-order block, the objective and the acceptance test are untouched,
    so the reduction is only in how many bundle directions the cone
    constrains.
    """

    name: str
    description: str
    eigenvalue_threshold: float

    def solve(
        self,
        instance: ConicInstance,
        previous: FloatArray | None = None,
    ) -> CandidateSolution:
        blocks = cone_blocks(instance)
        psd = [entry for entry in blocks if entry[0] == "PSDTriangleConeT"]
        if len(psd) != 1:
            return ClarabelCandidate(self.name, self.description).solve(
                instance, previous
            )
        _, size, rows = psd[0]
        started = time.perf_counter()
        base = unpack_svec_upper(instance.cone_rhs[rows], size)
        eigenvalues, eigenvectors = np.linalg.eigh(base)
        cutoff = self.eigenvalue_threshold * float(
            np.max(np.abs(eigenvalues))
        )
        kept = np.flatnonzero(eigenvalues <= cutoff)
        if kept.size == 0:
            kept = np.asarray([0], dtype=np.int64)
        working = eigenvectors[:, kept]
        reduced_size = int(kept.size)
        dense_block = np.asarray(
            instance.cone_matrix[rows, :].todense(), dtype=float
        )
        reduced_rows = np.empty(
            (reduced_size * (reduced_size + 1) // 2, instance.variable_count),
            dtype=float,
        )
        for column in range(instance.variable_count):
            coefficient = unpack_svec_upper(dense_block[:, column], size)
            reduced_rows[:, column] = pack_svec_upper(
                working.T @ coefficient @ working
            )
        reduced_rhs = pack_svec_upper(working.T @ base @ working)
        other = [entry for entry in blocks if entry[0] != "PSDTriangleConeT"]
        stack = [sparse.csc_matrix(reduced_rows)]
        rhs = [reduced_rhs]
        specification: list[tuple[str, int]] = [
            ("PSDTriangleConeT", reduced_size)
        ]
        for name, dimension, block_rows in other:
            stack.append(
                sparse.csc_matrix(instance.cone_matrix[block_rows, :])
            )
            rhs.append(instance.cone_rhs[block_rows])
            specification.append((name, dimension))
        cone_matrix = sparse.vstack(stack, format="csc")
        cone_rhs = np.concatenate(rhs)
        settings = build_solver_settings(instance.solver_settings)
        solution = clarabel.DefaultSolver(
            instance.objective_matrix,
            instance.objective_vector,
            cone_matrix,
            cone_rhs,
            build_cones(specification),
            settings,
        ).solve()
        elapsed = time.perf_counter() - started
        returned = (
            np.zeros(instance.variable_count)
            if solution.x is None
            else np.asarray(solution.x, dtype=float)
        )
        return CandidateSolution(
            solution=returned,
            status=(
                f"{solution.status}:bundle={reduced_size}/{size}"
            ),
            iterations=int(solution.iterations),
            wall_seconds=elapsed,
            primal_residual=float(solution.r_prim),
            dual_residual=float(solution.r_dual),
        )


def _scs_cone_and_permutation(
    instance: ConicInstance,
) -> tuple[dict[str, Any], NDArray[np.int64]]:
    """Return the SCS cone dictionary and the row permutation into it.

    SCS orders its cone blocks zero, nonnegative, second-order, then
    positive-semidefinite, and packs a semidefinite block as the lower
    triangle in column-major order with the off-diagonal entries scaled by
    ``sqrt(2)``.  The recorded blocks are Clarabel's: the same ``sqrt(2)``
    scaling on the upper triangle in column-major order, in the call site's
    own block order.  For a symmetric matrix the two triangles carry the same
    values, so the map between them is a permutation of rows.
    """

    blocks = cone_blocks(instance)
    zero: list[NDArray[np.int64]] = []
    nonneg: list[NDArray[np.int64]] = []
    second_order: list[tuple[int, NDArray[np.int64]]] = []
    semidefinite: list[tuple[int, NDArray[np.int64]]] = []
    for name, dimension, rows in blocks:
        indices = np.arange(rows.start, rows.stop, dtype=np.int64)
        if name == "ZeroConeT":
            zero.append(indices)
        elif name == "NonnegativeConeT":
            nonneg.append(indices)
        elif name == "SecondOrderConeT":
            second_order.append((dimension, indices))
        elif name == "PSDTriangleConeT":
            upper_position = np.zeros((dimension, dimension), dtype=np.int64)
            index = 0
            for column in range(dimension):
                for row in range(column + 1):
                    upper_position[row, column] = index
                    index += 1
            lower_order: list[int] = []
            for column in range(dimension):
                for row in range(column, dimension):
                    lower_order.append(int(upper_position[column, row]))
            semidefinite.append(
                (dimension, indices[np.asarray(lower_order, dtype=np.int64)])
            )
        else:
            raise ValueError(f"unhandled cone: {name}")
    order = [*zero, *nonneg]
    order.extend(indices for _, indices in second_order)
    order.extend(indices for _, indices in semidefinite)
    permutation = (
        np.concatenate(order)
        if order
        else np.zeros(0, dtype=np.int64)
    )
    cone: dict[str, Any] = {}
    if zero:
        cone["z"] = int(sum(int(block.size) for block in zero))
    if nonneg:
        cone["l"] = int(sum(int(block.size) for block in nonneg))
    if second_order:
        cone["q"] = [int(dimension) for dimension, _ in second_order]
    if semidefinite:
        cone["s"] = [int(dimension) for dimension, _ in semidefinite]
    return cone, permutation


@dataclass(frozen=True)
class WarmStartedScsCandidate:
    """Warm-started SCS on the recorded operands, with an optional polish.

    SCS is called directly, not through cvxpy: the recorded instance is
    already in SCS's standard form ``min 0.5 x^T P x + q^T x`` subject to
    ``A x + s = b``, ``s`` in the cone, so a cvxpy layer would only
    re-canonicalize operands that are already canonical, and it offers no way
    to warm start one problem from another problem's solution.  The warm
    start is the previous instance's returned ``x`` when it has the same
    length, with ``s = b - A x`` and ``y = 0``.  When ``polish`` is set and
    the declared profile's acceptance test fails on the SCS answer, one
    Clarabel solve at the recorded settings follows and its wall time is
    added.

    ``polish_inflation_shift`` is the ``inflation_shift`` the polish gate
    passes to :func:`acceptance`, so the gate can be the same test the bench
    reports.  On strict-profile operands the balanced acceptance test without
    that substitution accepts no solution at all -- not even the one Clarabel
    returned online -- so a balanced polish gate at the default ``0.0`` fires
    on every instance by construction and separates nothing.
    """

    name: str
    description: str
    epsilon: float
    polish_profile: str | None = None
    coefficients: ComplexArray | None = None
    polish_inflation_shift: float = 0.0
    #: Declared bounds so one instance cannot run away with the bench.
    maximum_iterations: int = 100000
    time_limit_seconds: float = 30.0

    def solve(
        self,
        instance: ConicInstance,
        previous: FloatArray | None = None,
    ) -> CandidateSolution:
        import scs

        cone, permutation = _scs_cone_and_permutation(instance)
        started = time.perf_counter()
        cone_matrix = sparse.csc_matrix(
            instance.cone_matrix[permutation, :]
        )
        data = {
            "P": sparse.triu(
                sparse.csc_matrix(instance.objective_matrix), format="csc"
            ),
            "A": cone_matrix,
            "b": instance.cone_rhs[permutation],
            "c": instance.objective_vector,
        }
        if previous is not None and previous.size == instance.variable_count:
            data["x"] = np.asarray(previous, dtype=float)
            data["s"] = data["b"] - cone_matrix @ data["x"]
            data["y"] = np.zeros(data["b"].size)
        solver = scs.SCS(
            data,
            cone,
            eps_abs=self.epsilon,
            eps_rel=self.epsilon,
            max_iters=self.maximum_iterations,
            time_limit_secs=self.time_limit_seconds,
            verbose=False,
        )
        result = solver.solve(
            warm_start="x" in data,
            x=data.get("x"),
            y=data.get("y"),
            s=data.get("s"),
        )
        elapsed = time.perf_counter() - started
        returned = np.asarray(result["x"], dtype=float)
        info = result["info"]
        status = str(info["status"])
        iterations = int(info["iter"])
        primal = float(info["res_pri"])
        dual = float(info["res_dual"])
        if self.polish_profile is not None and self.coefficients is not None:
            accepted, _, _ = acceptance(
                instance,
                self.coefficients,
                returned,
                profile=self.polish_profile,
                primal_residual=primal,
                dual_residual=dual,
                inflation_shift=self.polish_inflation_shift,
            )
            if not accepted:
                polished = ClarabelCandidate(
                    self.name, self.description
                ).solve(instance, previous)
                return CandidateSolution(
                    solution=polished.solution,
                    status=f"{status}->clarabel:{polished.status}",
                    iterations=iterations + polished.iterations,
                    wall_seconds=elapsed + polished.wall_seconds,
                    primal_residual=polished.primal_residual,
                    dual_residual=polished.dual_residual,
                )
        return CandidateSolution(
            solution=returned,
            status=status,
            iterations=iterations,
            wall_seconds=elapsed,
            primal_residual=primal,
            dual_residual=dual,
        )


def score_candidate(
    candidate: Any,
    instances: Sequence[ConicInstance],
    coefficients: ComplexArray,
) -> list[InstanceScore]:
    """Replay one candidate over every recorded instance, in solve order."""

    scores: list[InstanceScore] = []
    previous: FloatArray | None = None
    for instance in instances:
        answer = candidate.solve(instance, previous)
        previous = answer.solution
        recorded = (
            np.zeros(instance.variable_count)
            if instance.recorded_solution is None
            else instance.recorded_solution
        )
        strict_accepted, strict_residual, minimum = acceptance(
            instance,
            coefficients,
            answer.solution,
            profile="strict",
            primal_residual=answer.primal_residual,
            dual_residual=answer.dual_residual,
        )
        balanced_accepted, balanced_residual, _ = acceptance(
            instance,
            coefficients,
            answer.solution,
            profile="balanced",
            primal_residual=answer.primal_residual,
            dual_residual=answer.dual_residual,
        )
        (
            recorded_strict,
            _,
            recorded_minimum,
        ) = acceptance(
            instance,
            coefficients,
            recorded,
            profile="strict",
            primal_residual=instance.recorded_primal_residual,
            dual_residual=instance.recorded_dual_residual,
        )
        recorded_balanced, _, _ = acceptance(
            instance,
            coefficients,
            recorded,
            profile="balanced",
            primal_residual=instance.recorded_primal_residual,
            dual_residual=instance.recorded_dual_residual,
        )
        balanced_shifted, _, _ = acceptance(
            instance,
            coefficients,
            answer.solution,
            profile="balanced",
            primal_residual=answer.primal_residual,
            dual_residual=answer.dual_residual,
            inflation_shift=BALANCED_INFLATION_SHIFT,
        )
        recorded_balanced_shifted, _, _ = acceptance(
            instance,
            coefficients,
            recorded,
            profile="balanced",
            primal_residual=instance.recorded_primal_residual,
            dual_residual=instance.recorded_dual_residual,
            inflation_shift=BALANCED_INFLATION_SHIFT,
        )
        scores.append(
            InstanceScore(
                instance_index=instance.index,
                tier=instance.tier,
                atom_index=instance.atom_index,
                candidate=candidate.name,
                wall_seconds=answer.wall_seconds,
                status=answer.status,
                iterations=answer.iterations,
                objective=objective_value(instance, answer.solution),
                maximum_cone_violation=maximum_cone_violation(
                    instance, answer.solution
                ),
                minimum_gram_eigenvalue=minimum,
                maximum_velocity_difference=float(
                    np.max(np.abs(answer.solution - recorded))
                ),
                bitwise_equal_to_recorded=bool(
                    np.array_equal(answer.solution, recorded)
                ),
                strict_accepted=strict_accepted,
                balanced_accepted=balanced_accepted,
                strict_normalized_residual=strict_residual,
                balanced_normalized_residual=balanced_residual,
                recorded_strict_accepted=recorded_strict,
                recorded_balanced_accepted=recorded_balanced,
                recorded_minimum_gram_eigenvalue=recorded_minimum,
                balanced_shifted_accepted=balanced_shifted,
                recorded_balanced_shifted_accepted=recorded_balanced_shifted,
            )
        )
    return scores


def summarize(scores: Sequence[InstanceScore]) -> dict[str, Any]:
    """Return the means and maxima the bench table reports."""

    seconds = np.asarray([score.wall_seconds for score in scores])
    violation = np.asarray(
        [score.maximum_cone_violation for score in scores]
    )
    difference = np.asarray(
        [score.maximum_velocity_difference for score in scores]
    )
    minimum = np.asarray(
        [score.minimum_gram_eigenvalue for score in scores]
    )
    return {
        "instances": len(scores),
        "total_wall_seconds": float(np.sum(seconds)),
        "mean_wall_seconds": float(np.mean(seconds)),
        "maximum_wall_seconds": float(np.max(seconds)),
        "mean_maximum_cone_violation": float(np.mean(violation)),
        "maximum_maximum_cone_violation": float(np.max(violation)),
        "mean_maximum_velocity_difference": float(np.mean(difference)),
        "maximum_maximum_velocity_difference": float(np.max(difference)),
        "minimum_gram_eigenvalue": float(np.min(minimum)),
        "strict_accepted": int(
            sum(1 for score in scores if score.strict_accepted)
        ),
        "balanced_accepted": int(
            sum(1 for score in scores if score.balanced_accepted)
        ),
        "recorded_strict_accepted": int(
            sum(1 for score in scores if score.recorded_strict_accepted)
        ),
        "recorded_balanced_accepted": int(
            sum(1 for score in scores if score.recorded_balanced_accepted)
        ),
        "strict_verdict_agreement": int(
            sum(
                1
                for score in scores
                if score.strict_accepted == score.recorded_strict_accepted
            )
        ),
        "balanced_verdict_agreement": int(
            sum(
                1
                for score in scores
                if score.balanced_accepted
                == score.recorded_balanced_accepted
            )
        ),
        "strict_accepted_where_recorded_accepted": int(
            sum(
                1
                for score in scores
                if score.recorded_strict_accepted and score.strict_accepted
            )
        ),
        "balanced_accepted_where_recorded_accepted": int(
            sum(
                1
                for score in scores
                if score.recorded_balanced_accepted
                and score.balanced_accepted
            )
        ),
        "balanced_shifted_accepted": int(
            sum(1 for score in scores if score.balanced_shifted_accepted)
        ),
        "recorded_balanced_shifted_accepted": int(
            sum(
                1
                for score in scores
                if score.recorded_balanced_shifted_accepted
            )
        ),
        "balanced_shifted_verdict_agreement": int(
            sum(
                1
                for score in scores
                if score.balanced_shifted_accepted
                == score.recorded_balanced_shifted_accepted
            )
        ),
        "bitwise_equal_to_recorded": int(
            sum(1 for score in scores if score.bitwise_equal_to_recorded)
        ),
        "statuses": sorted({score.status for score in scores}),
    }
