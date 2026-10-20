"""Alternative conic formulations of the one-shell outer target stage.

The one-shell outer target stage of the projective guard is the solve

``minimize 0.5 q^T M q  over the standardized outer frontier coordinates z``
``subject to  the one-shell outer Gram  G(z) = base(lower) + sum_i z_i C_i >= 0,``
``            the box |z_i| <= 1, and the invariant target image q = R z,``

where ``R`` is the invariant target image readout embedded in the literal
outer frontier (``_outer_current_image_matrix``), ``M`` is the whitened target
metric (the inverse of the readout covariance), and ``base`` and ``C_i`` are
the affine pieces of the scaled one-shell outer Gram at the stage's own lower
moments.  ``apcm_projective_guard._solve_explicit_target_qp`` hands exactly
this program to Clarabel with ``q`` carried as an explicit decision variable
and the image equality carried as a zero cone.

This module does not change that program, that solver, or the acceptance rule.
It extracts the program's operands through the existing builders, states each
alternative formulation of the *same* mathematical problem as a named option
whose default reproduces the recorded program, and evaluates every candidate
point on the recorded program's own operands, so the acceptance decision is
the one ``apcm_projective_guard`` already makes.

Every solve returns the ``GuardConicResult`` shape the current selector
produces, with the invariant target image quadratic ``0.5 q^T M q`` as the
reported ``objective`` in all formulations.

Nothing here is on the recorded path.  ``apcm_projective_guard`` and
``projected_apcm`` are imported from and never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from time import perf_counter
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping

import clarabel
import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse

from .apcm_positive_extension import (
    SymmetryReducedPositiveExtension,
    _clarabel_svec_upper,
    _clarabel_svec_upper_pattern,
    _realify_hermitian,
)
from .apcm_projective_guard import (
    GuardConicResult,
    InvariantTargetReadout,
    _guard_acceptance,
    _independent_kkt_diagnostics,
    _outer_current_image_matrix,
    _solve_explicit_target_qp,
    compile_invariant_target_readout,
    projective_guard_outer_extension,
)
from .moment_hierarchy import MomentKey

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# operands
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OuterTargetOperands:
    """The exact operands of one one-shell outer target stage solve.

    ``extension`` is the one-shell outer positive extension built by
    ``projective_guard_outer_extension``; ``lower_moments`` are the stage's
    own lower moments; ``scaled_readout`` is ``R``; ``target_metric`` is
    ``M``.  ``real_base`` and ``coefficient_matrices`` are the realified
    scaled Gram pieces, and ``coefficient_vectors`` their Clarabel svec
    packing, assembled exactly as ``_solve_explicit_target_qp`` assembles
    them.
    """

    name: str
    extension: SymmetryReducedPositiveExtension
    lower_moments: Mapping[MomentKey, float]
    scaled_readout: FloatArray
    target_metric: FloatArray
    boxed: bool
    real_base: FloatArray
    coefficient_matrices: FloatArray
    coefficient_vectors: FloatArray
    base_vector: FloatArray

    @property
    def frontier_count(self) -> int:
        return int(self.scaled_readout.shape[1])

    @property
    def target_count(self) -> int:
        return int(self.scaled_readout.shape[0])

    @property
    def gram_side(self) -> int:
        return int(self.real_base.shape[0])


def outer_target_operands_from_program(
    outer: SymmetryReducedPositiveExtension,
    lower_moments: Mapping[MomentKey, float],
    scaled_readout: FloatArray,
    target_metric: FloatArray,
    *,
    name: str = "outer_target_stage",
    boxed: bool = True,
) -> OuterTargetOperands:
    """Assemble the operands of one target stage already handed its readout.

    This is the entry point for a drop-in replacement of
    ``_solve_explicit_target_qp``: it takes the same four positional operands
    that routine takes and assembles the affine Gram pieces exactly as it
    assembles them.
    """

    lower_values = np.asarray(
        [float(lower_moments[key]) for key in outer.lower_keys], dtype=float
    )
    unscaled_base = outer._constant + np.tensordot(
        lower_values, outer.lower_coefficients, axes=(0, 0)
    )
    scaled_base = outer.scaled_matrix(unscaled_base)
    scaled_coefficients = (
        outer.scaled_frontier_coefficients
        * outer.frontier_scales[:, None, None]
    )
    real_base = _realify_hermitian(scaled_base)
    coefficient_matrices = np.asarray(
        [_realify_hermitian(value) for value in scaled_coefficients],
        dtype=float,
    )
    coefficient_vectors = np.column_stack(
        [_clarabel_svec_upper(value) for value in coefficient_matrices]
    )
    return OuterTargetOperands(
        name=name,
        extension=outer,
        lower_moments=MappingProxyType(
            {key: float(lower_moments[key]) for key in outer.lower_keys}
        ),
        scaled_readout=np.asarray(scaled_readout, dtype=float),
        target_metric=np.asarray(target_metric, dtype=float),
        boxed=bool(boxed),
        real_base=real_base,
        coefficient_matrices=coefficient_matrices,
        coefficient_vectors=coefficient_vectors,
        base_vector=_clarabel_svec_upper(real_base),
    )


def build_outer_target_operands(
    current: SymmetryReducedPositiveExtension,
    lower_moments: Mapping[MomentKey, float],
    readout: InvariantTargetReadout | None = None,
    *,
    name: str = "outer_target_stage",
    boxed: bool = True,
) -> OuterTargetOperands:
    """Extract the outer target stage operands through the existing builders.

    The one-shell outer extension, the readout embedding and the target
    metric are formed exactly as ``select_outer_feasible_projective_guard``
    forms them.
    """

    target_readout = (
        compile_invariant_target_readout(current)
        if readout is None
        else readout
    )
    if target_readout.frontier_keys != current.frontier_keys:
        raise ValueError("target readout belongs to another extension")
    outer = projective_guard_outer_extension(current)
    scaled_outer_image = _outer_current_image_matrix(
        current, outer, target_readout
    )
    covariance = target_readout.matrix * current.frontier_scales[None, :]
    covariance = covariance @ covariance.T
    target_metric = np.linalg.inv(covariance)
    return outer_target_operands_from_program(
        outer,
        lower_moments,
        scaled_outer_image,
        target_metric,
        name=name,
        boxed=boxed,
    )


def save_outer_target_operands(
    directory: Path | str, operands: OuterTargetOperands
) -> Path:
    """Save one operand set as arrays plus a declared description."""

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / "operands.npz",
        scaled_readout=operands.scaled_readout,
        target_metric=operands.target_metric,
        real_base=operands.real_base,
        coefficient_matrices=operands.coefficient_matrices,
        base_vector=operands.base_vector,
        lower_values=np.asarray(
            [
                float(operands.lower_moments[key])
                for key in operands.extension.lower_keys
            ],
            dtype=float,
        ),
        frontier_scales=operands.extension.frontier_scales,
        word_scales=operands.extension.word_scales,
    )
    description = {
        "name": operands.name,
        "boxed": operands.boxed,
        "one_shell_outer_gram_rows": int(operands.extension.dimension),
        "realified_gram_side": operands.gram_side,
        "outer_frontier_moments": operands.frontier_count,
        "invariant_target_readout_rank": operands.target_count,
        "outer_lower_moments": len(operands.extension.lower_keys),
        "lower_keys": [str(key) for key in operands.extension.lower_keys],
        "frontier_keys": [
            str(key) for key in operands.extension.frontier_keys
        ],
        "quantities": {
            "scaled_readout": "R, the invariant target image readout embedded in the literal outer frontier",
            "target_metric": "M, the whitened target metric (inverse readout covariance)",
            "real_base": "the realified scaled one-shell outer Gram at the stage's lower moments with every frontier coordinate zero",
            "coefficient_matrices": "the realified scaled Gram coefficient of each standardized outer frontier coordinate",
            "base_vector": "the Clarabel svec packing of real_base",
        },
    }
    (path / "operands.json").write_text(
        json.dumps(description, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# the recorded program, and the candidate evaluation on it
# ---------------------------------------------------------------------------


def _recorded_settings(
    extension: SymmetryReducedPositiveExtension,
    retry_profile: int = 0,
    *,
    time_limit_seconds: float | None = None,
) -> Any:
    """Clarabel settings copied from ``_solve_explicit_target_qp``.

    These literals are the recorded target-stage settings.  They are copied
    rather than imported because the recorded routine builds them inline;
    ``test_apcm_outer_selector_formulations`` pins the copy by checking that
    the recorded formulation of this module reproduces
    ``_solve_explicit_target_qp`` exactly on the strict fixture.
    """

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = extension.settings.clarabel_max_threads
    settings.max_iter = (
        extension.settings.maximum_iterations if retry_profile == 0 else 500
    )
    solve_tolerance = (
        extension.settings.conic_tolerance if retry_profile == 0 else 1e-10
    )
    settings.tol_gap_abs = solve_tolerance
    settings.tol_gap_rel = solve_tolerance
    settings.tol_feas = solve_tolerance
    settings.tol_infeas_abs = solve_tolerance
    settings.tol_infeas_rel = solve_tolerance
    settings.equilibrate_enable = retry_profile == 0
    settings.equilibrate_max_iter = 10
    settings.equilibrate_min_scaling = 1e-4
    settings.equilibrate_max_scaling = 1e4
    settings.iterative_refinement_enable = True
    settings.iterative_refinement_max_iter = 10 if retry_profile == 0 else 20
    settings.iterative_refinement_abstol = 1e-12 if retry_profile == 0 else 1e-14
    settings.iterative_refinement_reltol = 1e-12 if retry_profile == 0 else 1e-14
    settings.static_regularization_enable = True
    settings.static_regularization_constant = 1e-12
    settings.dynamic_regularization_enable = True
    settings.dynamic_regularization_eps = 1e-13
    settings.dynamic_regularization_delta = 2e-7
    settings.chordal_decomposition_enable = False
    if time_limit_seconds is not None:
        settings.time_limit = float(time_limit_seconds)
    return settings


def recorded_program(
    operands: OuterTargetOperands, *, frontier_weight: float = 0.0
) -> tuple[sparse.csc_matrix, FloatArray, sparse.csc_matrix, FloatArray, list]:
    """Assemble the recorded target-stage program ``(P, q, A, b, cones)``.

    The variable order is ``(z, q)``, the row order is the positive
    semidefinite block, then the box, then the image equality, exactly as
    ``_solve_explicit_target_qp`` assembles them.
    """

    count = operands.frontier_count
    target_count = operands.target_count
    psd_block = sparse.hstack(
        (
            sparse.csc_matrix(-operands.coefficient_vectors),
            sparse.csc_matrix(
                (operands.coefficient_vectors.shape[0], target_count)
            ),
        ),
        format="csc",
    )
    blocks: list[sparse.csc_matrix] = [psd_block]
    rhs: list[FloatArray] = [operands.base_vector]
    cones: list[Any] = [
        clarabel.PSDTriangleConeT(operands.gram_side)
    ]
    if operands.boxed:
        box_z = sparse.vstack(
            (sparse.eye(count, format="csc"), -sparse.eye(count, format="csc")),
            format="csc",
        )
        blocks.append(
            sparse.hstack(
                (box_z, sparse.csc_matrix((2 * count, target_count))),
                format="csc",
            )
        )
        rhs.append(np.ones(2 * count))
        cones.append(clarabel.NonnegativeConeT(2 * count))
    image_equality = sparse.hstack(
        (
            -sparse.csc_matrix(operands.scaled_readout),
            sparse.eye(target_count, format="csc"),
        ),
        format="csc",
    )
    blocks.append(image_equality)
    rhs.append(np.zeros(target_count))
    cones.append(clarabel.ZeroConeT(target_count))
    if frontier_weight < 0.0:
        raise ValueError("frontier_weight must be nonnegative")
    frontier_block = (
        sparse.csc_matrix((count, count))
        if frontier_weight == 0.0
        else sparse.csc_matrix(float(frontier_weight) * np.eye(count))
    )
    quadratic = sparse.block_diag(
        (frontier_block, sparse.csc_matrix(operands.target_metric)),
        format="csc",
    )
    return (
        quadratic,
        np.zeros(count + target_count),
        sparse.vstack(blocks, format="csc"),
        np.concatenate(rhs),
        cones,
    )


@dataclass(frozen=True)
class FormulationOutcome:
    """One candidate formulation, evaluated on the recorded program."""

    formulation: str
    options: Mapping[str, Any]
    result: GuardConicResult
    solver_status: str
    wall_seconds: float
    solver_iterations: int
    solver_primal_residual: float
    solver_dual_residual: float
    frontier_weight: float
    notes: Mapping[str, Any]

    def record(self) -> dict[str, Any]:
        result = self.result
        return {
            "formulation": self.formulation,
            "options": dict(self.options),
            "solver_status": self.solver_status,
            "wall_seconds": float(self.wall_seconds),
            "solver_iterations": int(self.solver_iterations),
            "solver_primal_residual": float(self.solver_primal_residual),
            "solver_dual_residual": float(self.solver_dual_residual),
            "frontier_weight": float(self.frontier_weight),
            "status": result.status,
            "success": bool(result.success),
            "certified": bool(result.certified),
            "provisional": bool(result.provisional),
            "acceptance": result.acceptance,
            "objective_invariant_target_image_quadratic": float(
                result.objective
            ),
            "minimum_scaled_eigenvalue": float(
                result.minimum_scaled_eigenvalue
            ),
            "independent_feasibility_residual": float(
                result.independent_feasibility_residual
            ),
            "stationarity_residual": float(result.stationarity_residual),
            "complementarity_residual": float(result.complementarity_residual),
            "relative_duality_gap": float(result.relative_duality_gap),
            "notes": dict(self.notes),
        }


def evaluate_on_recorded_program(
    operands: OuterTargetOperands,
    standardized_values: FloatArray,
    *,
    dual_psd: FloatArray | None,
    dual_box: FloatArray | None,
    status: str,
    iterations: int,
    solver_primal_residual: float,
    solver_dual_residual: float,
    frontier_weight: float = 0.0,
    status_prefix: str = "explicit_target",
) -> GuardConicResult:
    """Score one candidate point by the recorded program's own rule.

    The candidate's outer frontier coordinates are mapped back to the
    recorded variable order ``(z, q = R z)`` and the recorded program's
    ``(P, q, A, b)`` are used for every residual, so the returned
    ``GuardConicResult`` carries the acceptance
    ``apcm_projective_guard._guard_acceptance`` gives at its unchanged
    exploratory tolerance.  The relative duality gap is the one recomputed
    from ``(x, z)`` on the recorded program, the quantity
    ``_original_relative_duality_gap`` defines, because a reformulated
    solver's own objective pair belongs to its own program.
    """

    extension = operands.extension
    count = operands.frontier_count
    target_count = operands.target_count
    values = np.asarray(standardized_values, dtype=float)
    target = operands.scaled_readout @ values
    quadratic, linear, cone_matrix, cone_rhs, _ = recorded_program(
        operands, frontier_weight=frontier_weight
    )
    frontier_values = values * extension.frontier_scales
    frontier = MappingProxyType(
        {
            key: float(value)
            for key, value in zip(
                extension.frontier_keys, frontier_values, strict=True
            )
        }
    )
    matrix = extension.matrix(operands.lower_moments, frontier)
    scaled = extension.scaled_matrix(matrix)
    minimum = float(np.linalg.eigvalsh(scaled)[0])
    image_error = float(
        np.linalg.norm(target - operands.scaled_readout @ values, ord=np.inf)
    )
    box_error = (
        0.0
        if not operands.boxed
        else max(0.0, float(np.max(np.abs(values))) - 1.0)
    )
    tolerance = extension.settings.conic_tolerance
    strict_success = bool(
        status == "Solved"
        and max(
            float(solver_primal_residual),
            float(solver_dual_residual),
            image_error,
        )
        <= 10.0 * tolerance
        and minimum >= -10.0 * tolerance
        and (
            not operands.boxed
            or float(np.max(np.abs(values))) <= 1.0 + 10.0 * tolerance
        )
    )
    variables = np.concatenate((values, target))
    if dual_psd is None or dual_box is None:
        diagnostics = _independent_kkt_diagnostics(
            SimpleNamespace(z=None, s=None),
            variables,
            quadratic,
            linear,
            cone_matrix,
            cone_rhs,
            direct_feasibility=max(max(0.0, -minimum), box_error, image_error),
        )
    else:
        image_dual = -np.asarray(operands.target_metric, dtype=float) @ target
        dual_vector = np.concatenate(
            (
                np.asarray(dual_psd, dtype=float),
                np.asarray(dual_box, dtype=float),
                image_dual,
            )
        )
        slack = cone_rhs - np.asarray(cone_matrix @ variables, dtype=float)
        slack[-target_count:] = 0.0
        quadratic_form = 0.5 * float(
            np.dot(variables, np.asarray(quadratic @ variables, dtype=float))
        )
        diagnostics = _independent_kkt_diagnostics(
            SimpleNamespace(
                z=dual_vector,
                s=slack,
                obj_val=quadratic_form,
                obj_val_dual=(
                    -quadratic_form - float(np.dot(cone_rhs, dual_vector))
                ),
            ),
            variables,
            quadratic,
            linear,
            cone_matrix,
            cone_rhs,
            direct_feasibility=max(max(0.0, -minimum), box_error, image_error),
        )
    success, certified, provisional, acceptance = _guard_acceptance(
        status, strict_success=strict_success, diagnostics=diagnostics
    )
    del count
    return GuardConicResult(
        success=success,
        status=f"{status_prefix}:{status}",
        boxed=operands.boxed,
        conditioned=False,
        standardized_values=values,
        frontier_moments=frontier,
        moment_matrix=matrix,
        minimum_scaled_eigenvalue=minimum,
        primal_residual=max(float(solver_primal_residual), image_error),
        dual_residual=float(solver_dual_residual),
        objective=float(0.5 * target @ operands.target_metric @ target),
        iterations=int(iterations),
        certified=certified,
        provisional=provisional,
        acceptance=acceptance,
        independent_feasibility_residual=diagnostics.feasibility,
        stationarity_residual=diagnostics.stationarity,
        complementarity_residual=diagnostics.complementarity,
        relative_duality_gap=diagnostics.relative_gap,
    )


# ---------------------------------------------------------------------------
# svec helpers, for mapping a positive semidefinite dual back
# ---------------------------------------------------------------------------


def _unsvec_upper(vector: FloatArray, size: int) -> FloatArray:
    """Inverse of ``_clarabel_svec_upper`` on symmetric matrices."""

    rows, columns, scales = _clarabel_svec_upper_pattern(size)
    matrix = np.zeros((size, size), dtype=float)
    values = np.asarray(vector, dtype=float) / scales
    matrix[rows, columns] = values
    matrix[columns, rows] = values
    return matrix


# ---------------------------------------------------------------------------
# the reduced formulation kernel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OuterTargetFormulation:
    """One named conic formulation of the same outer target stage.

    Every field's default reproduces the recorded program assembled by
    ``_solve_explicit_target_qp``: the image equality is carried explicitly as
    a zero cone, the target metric stays in the objective, no diagonal scaling
    is applied, the positive semidefinite cone is not inflated, and the
    frontier variables are the standardized ones.
    """

    #: eliminate the image equality by the substitution ``q = R z``.
    eliminate_image_equality: bool = False
    #: carry ``t = W z`` with ``W^T W = R^T M R`` instead of ``q`` and ``M``.
    absorb_whitening: bool = False
    #: diagonal congruence ``G -> E G E`` of the one-shell outer Gram.
    gram_congruence: str = "none"
    #: diagonal scaling of the frontier variables.
    frontier_column_scaling: str = "none"
    #: declared inflation ``G >= eps I`` of the outer cone.
    psd_inflation: float = 0.0
    #: one common factor on every frontier variable.
    homogeneous_frontier_scale: float = 1.0
    #: the declared strictly convex frontier term ``0.5 w ||z||^2``.
    frontier_weight: float = 0.0
    #: Clarabel retry profile, 0 being the recorded first pass.
    retry_profile: int = 0
    #: Ruiz sweeps used by the two scaling options.
    equilibration_sweeps: int = 20
    #: declared wall budget handed to Clarabel; ``None`` is the recorded path.
    time_limit_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.gram_congruence not in ("none", "ruiz"):
            raise ValueError("gram_congruence must be 'none' or 'ruiz'")
        if self.frontier_column_scaling not in ("none", "ruiz"):
            raise ValueError(
                "frontier_column_scaling must be 'none' or 'ruiz'"
            )
        if self.psd_inflation < 0.0:
            raise ValueError("psd_inflation must be nonnegative")
        if not self.homogeneous_frontier_scale > 0.0:
            raise ValueError("homogeneous_frontier_scale must be positive")
        if self.frontier_weight < 0.0:
            raise ValueError("frontier_weight must be nonnegative")
        if self.equilibration_sweeps < 0:
            raise ValueError("equilibration_sweeps must be nonnegative")
        if self.absorb_whitening and self.eliminate_image_equality:
            # ``q`` is eliminated, so there is no image variable to whiten.
            raise ValueError(
                "absorb_whitening applies only when the image equality is "
                "carried explicitly"
            )

    def as_record(self) -> dict[str, Any]:
        return {
            "eliminate_image_equality": bool(self.eliminate_image_equality),
            "absorb_whitening": bool(self.absorb_whitening),
            "gram_congruence": self.gram_congruence,
            "frontier_column_scaling": self.frontier_column_scaling,
            "psd_inflation": float(self.psd_inflation),
            "homogeneous_frontier_scale": float(
                self.homogeneous_frontier_scale
            ),
            "frontier_weight": float(self.frontier_weight),
            "retry_profile": int(self.retry_profile),
            "equilibration_sweeps": int(self.equilibration_sweeps),
            "time_limit_seconds": (
                None
                if self.time_limit_seconds is None
                else float(self.time_limit_seconds)
            ),
        }


def gram_magnitude(operands: OuterTargetOperands) -> FloatArray:
    """Entrywise magnitude of the affine Gram map, base and coefficients."""

    magnitude = np.abs(operands.real_base)
    coefficient_magnitude = np.max(
        np.abs(operands.coefficient_matrices), axis=0
    )
    return np.maximum(magnitude, coefficient_magnitude)


def ruiz_congruence(
    magnitude: FloatArray, *, sweeps: int = 20
) -> FloatArray:
    """Symmetric Ruiz equilibration factors of one symmetric magnitude map.

    Returns the diagonal ``E`` with ``E T E`` having unit infinity row norms
    to the sweep count.  The congruence ``G -> E G E`` maps the positive
    semidefinite cone onto itself, so it is an exact reformulation of the
    cone constraint rather than a row scaling of its svec rows.
    """

    size = int(magnitude.shape[0])
    factors = np.ones(size, dtype=float)
    working = np.array(magnitude, dtype=float)
    for _ in range(int(sweeps)):
        norms = np.max(working, axis=1)
        norms = np.where(norms > 0.0, norms, 1.0)
        step = 1.0 / np.sqrt(norms)
        factors = factors * step
        working = working * step[:, None] * step[None, :]
    return factors


def _column_scaling(
    coefficient_matrices: FloatArray, *, sweeps: int = 20
) -> FloatArray:
    """Diagonal frontier scaling from the coefficient Frobenius norms."""

    del sweeps
    norms = np.linalg.norm(
        coefficient_matrices.reshape(coefficient_matrices.shape[0], -1),
        axis=1,
    )
    norms = np.where(norms > 0.0, norms, 1.0)
    return 1.0 / np.sqrt(norms)


def solve_outer_target_formulation(
    operands: OuterTargetOperands,
    formulation: OuterTargetFormulation | None = None,
) -> FormulationOutcome:
    """Solve one named formulation and score it on the recorded program."""

    spec = OuterTargetFormulation() if formulation is None else formulation
    count = operands.frontier_count
    target_count = operands.target_count
    side = operands.gram_side
    extension = operands.extension

    congruence = (
        np.ones(side, dtype=float)
        if spec.gram_congruence == "none"
        else ruiz_congruence(
            gram_magnitude(operands), sweeps=spec.equilibration_sweeps
        )
    )
    column = (
        np.ones(count, dtype=float)
        if spec.frontier_column_scaling == "none"
        else _column_scaling(
            operands.coefficient_matrices, sweeps=spec.equilibration_sweeps
        )
    )
    scale = float(spec.homogeneous_frontier_scale) * column

    if spec.gram_congruence == "none" and np.all(scale == 1.0):
        base_matrix = operands.real_base
        coefficient_vectors = operands.coefficient_vectors
    else:
        base_matrix = (
            congruence[:, None] * operands.real_base * congruence[None, :]
        )
        scaled_coefficients = (
            operands.coefficient_matrices
            * congruence[None, :, None]
            * congruence[None, None, :]
        ) * scale[:, None, None]
        coefficient_vectors = np.column_stack(
            [_clarabel_svec_upper(value) for value in scaled_coefficients]
        )
    if spec.psd_inflation > 0.0:
        base_matrix = base_matrix - float(spec.psd_inflation) * np.eye(side)
    base_vector = _clarabel_svec_upper(base_matrix)

    readout = operands.scaled_readout * scale[None, :]
    metric = np.asarray(operands.target_metric, dtype=float)
    whitening = np.asarray(
        linalg.cholesky(metric, lower=False), dtype=float
    )  # metric = whitening^T whitening

    blocks: list[sparse.csc_matrix] = []
    rhs: list[FloatArray] = []
    cones: list[Any] = []
    extra = 0 if spec.eliminate_image_equality else target_count
    variable_count = count + extra

    psd_block = sparse.hstack(
        (
            sparse.csc_matrix(-coefficient_vectors),
            sparse.csc_matrix((coefficient_vectors.shape[0], extra)),
        ),
        format="csc",
    )
    blocks.append(psd_block)
    rhs.append(base_vector)
    cones.append(clarabel.PSDTriangleConeT(side))
    if operands.boxed:
        box = sparse.diags(scale, format="csc")
        box_block = sparse.vstack((box, -box), format="csc")
        blocks.append(
            sparse.hstack(
                (box_block, sparse.csc_matrix((2 * count, extra))),
                format="csc",
            )
        )
        rhs.append(np.ones(2 * count))
        cones.append(clarabel.NonnegativeConeT(2 * count))
    if not spec.eliminate_image_equality:
        image_map = whitening @ readout if spec.absorb_whitening else readout
        blocks.append(
            sparse.hstack(
                (
                    -sparse.csc_matrix(image_map),
                    sparse.eye(target_count, format="csc"),
                ),
                format="csc",
            )
        )
        rhs.append(np.zeros(target_count))
        cones.append(clarabel.ZeroConeT(target_count))

    frontier_block = (
        sparse.csc_matrix((count, count))
        if spec.frontier_weight == 0.0
        else sparse.csc_matrix(
            float(spec.frontier_weight) * np.diag(scale * scale)
        )
    )
    if spec.eliminate_image_equality:
        # A pure nullspace parameterization: the image equality is removed by
        # the substitution ``q = R z`` and the target metric stays where the
        # recorded program puts it, inside the objective.
        quadratic = sparse.csc_matrix(
            np.asarray(frontier_block.todense())
            + readout.T @ metric @ readout
        )
    elif spec.absorb_whitening:
        quadratic = sparse.block_diag(
            (frontier_block, sparse.eye(target_count, format="csc")),
            format="csc",
        )
    else:
        quadratic = sparse.block_diag(
            (frontier_block, sparse.csc_matrix(metric)), format="csc"
        )
    cone_matrix = sparse.vstack(blocks, format="csc")
    cone_rhs = np.concatenate(rhs)
    settings = _recorded_settings(
        extension,
        spec.retry_profile,
        time_limit_seconds=spec.time_limit_seconds,
    )
    started = perf_counter()
    solution = clarabel.DefaultSolver(
        quadratic,
        np.zeros(variable_count),
        cone_matrix,
        cone_rhs,
        cones,
        settings,
    ).solve()
    elapsed = perf_counter() - started
    status = str(solution.status)
    raw = (
        np.asarray(solution.x, dtype=float)
        if solution.x is not None
        else np.zeros(variable_count)
    )
    values = raw[:count] * scale
    dual = (
        np.asarray(solution.z, dtype=float)
        if solution.z is not None
        else None
    )
    dual_psd: FloatArray | None = None
    dual_box: FloatArray | None = None
    if dual is not None:
        psd_rows = coefficient_vectors.shape[0]
        matrix_dual = _unsvec_upper(dual[:psd_rows], side)
        matrix_dual = (
            congruence[:, None] * matrix_dual * congruence[None, :]
        )
        dual_psd = _clarabel_svec_upper(matrix_dual)
        dual_box = (
            dual[psd_rows : psd_rows + 2 * count]
            if operands.boxed
            else np.zeros(0)
        )
    result = evaluate_on_recorded_program(
        operands,
        values,
        dual_psd=dual_psd,
        dual_box=dual_box,
        status=status,
        iterations=int(solution.iterations),
        solver_primal_residual=float(solution.r_prim),
        solver_dual_residual=float(solution.r_dual),
        frontier_weight=spec.frontier_weight,
    )
    return FormulationOutcome(
        formulation="reduced_kernel",
        options=spec.as_record(),
        result=result,
        solver_status=status,
        wall_seconds=elapsed,
        solver_iterations=int(solution.iterations),
        solver_primal_residual=float(solution.r_prim),
        solver_dual_residual=float(solution.r_dual),
        frontier_weight=spec.frontier_weight,
        notes={
            "variable_count": int(variable_count),
            "cone_rows": int(cone_matrix.shape[0]),
            "gram_congruence_condition": float(
                np.max(congruence) / np.min(congruence)
            ),
            "frontier_scaling_condition": float(
                np.max(scale) / np.min(scale)
            ),
        },
    )


def solve_recorded_outer_target(
    operands: OuterTargetOperands,
    *,
    frontier_weight: float = 0.0,
    accept_first_provisional: bool = False,
) -> FormulationOutcome:
    """Run the recorded target stage itself, for the comparison row."""

    started = perf_counter()
    result = _solve_explicit_target_qp(
        operands.extension,
        operands.lower_moments,
        operands.scaled_readout,
        operands.target_metric,
        boxed=operands.boxed,
        accept_first_provisional=accept_first_provisional,
        frontier_weight=frontier_weight,
    )
    elapsed = perf_counter() - started
    return FormulationOutcome(
        formulation="recorded",
        options={
            "frontier_weight": float(frontier_weight),
            "accept_first_provisional": bool(accept_first_provisional),
        },
        result=result,
        solver_status=result.status,
        wall_seconds=elapsed,
        solver_iterations=int(result.iterations),
        solver_primal_residual=float(result.primal_residual),
        solver_dual_residual=float(result.dual_residual),
        frontier_weight=float(frontier_weight),
        notes={},
    )


# ---------------------------------------------------------------------------
# the dual problem
# ---------------------------------------------------------------------------


def solve_outer_target_dual(
    operands: OuterTargetOperands,
    *,
    frontier_weight: float = 0.0,
    retry_profile: int = 0,
    time_limit_seconds: float | None = None,
) -> FormulationOutcome:
    """Solve the Lagrangian dual of the eliminated outer target stage.

    The eliminated program is ``min 0.5 z^T P z`` subject to ``A z + s = b``
    with ``s`` in the product of the positive semidefinite cone and the box's
    nonnegative cone.  Its dual is ``max -0.5 z^T P z - b^T y`` subject to
    ``P z + A^T y = 0`` and ``y`` in the same cone, which is handed to
    Clarabel as a primal program in ``(z, y)``.  The returned ``z`` is scored
    on the recorded program exactly as every other formulation is.
    """

    count = operands.frontier_count
    side = operands.gram_side
    metric = np.asarray(operands.target_metric, dtype=float)
    whitened = (
        np.asarray(linalg.cholesky(metric, lower=False), dtype=float)
        @ operands.scaled_readout
    )
    quadratic_dense = whitened.T @ whitened
    if frontier_weight > 0.0:
        quadratic_dense = quadratic_dense + float(frontier_weight) * np.eye(
            count
        )
    affine = [sparse.csc_matrix(-operands.coefficient_vectors)]
    rhs = [operands.base_vector]
    cones: list[Any] = [clarabel.PSDTriangleConeT(side)]
    if operands.boxed:
        affine.append(
            sparse.vstack(
                (
                    sparse.eye(count, format="csc"),
                    -sparse.eye(count, format="csc"),
                ),
                format="csc",
            )
        )
        rhs.append(np.ones(2 * count))
        cones.append(clarabel.NonnegativeConeT(2 * count))
    primal_matrix = sparse.vstack(affine, format="csc")
    primal_rhs = np.concatenate(rhs)
    rows = int(primal_matrix.shape[0])

    stationarity = sparse.hstack(
        (sparse.csc_matrix(quadratic_dense), primal_matrix.T), format="csc"
    )
    membership = sparse.hstack(
        (
            sparse.csc_matrix((rows, count)),
            -sparse.eye(rows, format="csc"),
        ),
        format="csc",
    )
    dual_matrix = sparse.vstack((stationarity, membership), format="csc")
    dual_rhs = np.concatenate((np.zeros(count), np.zeros(rows)))
    dual_cones: list[Any] = [clarabel.ZeroConeT(count)] + cones
    dual_quadratic = sparse.block_diag(
        (sparse.csc_matrix(quadratic_dense), sparse.csc_matrix((rows, rows))),
        format="csc",
    )
    dual_linear = np.concatenate((np.zeros(count), primal_rhs))
    settings = _recorded_settings(
        operands.extension,
        retry_profile,
        time_limit_seconds=time_limit_seconds,
    )
    started = perf_counter()
    solution = clarabel.DefaultSolver(
        dual_quadratic,
        dual_linear,
        dual_matrix,
        dual_rhs,
        dual_cones,
        settings,
    ).solve()
    elapsed = perf_counter() - started
    status = str(solution.status)
    raw = (
        np.asarray(solution.x, dtype=float)
        if solution.x is not None
        else np.zeros(count + rows)
    )
    dual_variables = raw[count:]
    dual_psd = dual_variables[: operands.coefficient_vectors.shape[0]]
    dual_box = dual_variables[operands.coefficient_vectors.shape[0] :]
    multipliers = (
        np.asarray(solution.z, dtype=float)
        if solution.z is not None
        else np.zeros(count + rows)
    )
    # ``P z + A^T y = 0`` is the dual program's zero-cone block, so its own
    # multiplier is the original program's frontier point, up to the sign
    # convention of the cone; the dual program's primal ``z`` is determined
    # only modulo the nullspace of ``P`` and is reported next to it.
    candidates = {
        "zero_cone_multiplier": multipliers[:count],
        "negated_zero_cone_multiplier": -multipliers[:count],
        "dual_program_primal_block": raw[:count],
    }
    scored = {
        label: evaluate_on_recorded_program(
            operands,
            candidate,
            dual_psd=dual_psd,
            dual_box=dual_box,
            status=status,
            iterations=int(solution.iterations),
            solver_primal_residual=float(solution.r_prim),
            solver_dual_residual=float(solution.r_dual),
            frontier_weight=frontier_weight,
            status_prefix="explicit_target_dual",
        )
        for label, candidate in candidates.items()
    }
    chosen = min(
        scored,
        key=lambda label: (
            scored[label].independent_feasibility_residual,
            scored[label].objective,
        ),
    )
    result = scored[chosen]
    return FormulationOutcome(
        formulation="dual",
        options={
            "frontier_weight": float(frontier_weight),
            "retry_profile": int(retry_profile),
        },
        result=result,
        solver_status=status,
        wall_seconds=elapsed,
        solver_iterations=int(solution.iterations),
        solver_primal_residual=float(solution.r_prim),
        solver_dual_residual=float(solution.r_dual),
        frontier_weight=float(frontier_weight),
        notes={
            "variable_count": int(count + rows),
            "cone_rows": int(dual_matrix.shape[0]),
            "frontier_point_taken_from": chosen,
            "candidate_feasibility_residuals": {
                label: float(value.independent_feasibility_residual)
                for label, value in scored.items()
            },
            "candidate_objectives": {
                label: float(value.objective)
                for label, value in scored.items()
            },
        },
    )


# ---------------------------------------------------------------------------
# a first-order pass, and a Clarabel polish shifted to its point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FirstOrderPass:
    """One alternating-direction pass on the positive-plus-affine split."""

    standardized_values: FloatArray
    iterations: int
    wall_seconds: float
    primal_residual: float
    dual_residual: float
    minimum_scaled_eigenvalue: float


def psd_affine_admm(
    operands: OuterTargetOperands,
    *,
    frontier_weight: float = 0.0,
    penalty: float = 1.0,
    iterations: int = 400,
    tolerance: float = 1e-8,
) -> FirstOrderPass:
    """Alternating directions on the eliminated outer target stage.

    Splits ``min 0.5 z^T P z`` subject to ``G(z) >= 0`` and ``|z| <= 1`` into
    one linear solve in ``z``, one eigenvalue projection of the 120-row
    realified Gram, and one box clip, all in numpy.  It is a first-order
    starting point only; no acceptance claim is made from it.
    """

    count = operands.frontier_count
    side = operands.gram_side
    metric = np.asarray(operands.target_metric, dtype=float)
    whitened = (
        np.asarray(linalg.cholesky(metric, lower=False), dtype=float)
        @ operands.scaled_readout
    )
    quadratic = whitened.T @ whitened
    if frontier_weight > 0.0:
        quadratic = quadratic + float(frontier_weight) * np.eye(count)
    coefficients = operands.coefficient_vectors
    gram_normal = coefficients.T @ coefficients
    system = quadratic + penalty * (gram_normal + np.eye(count))
    factor = linalg.cho_factor(system, lower=False)

    values = np.zeros(count, dtype=float)
    gram = np.array(operands.base_vector, dtype=float)
    gram_multiplier = np.zeros_like(gram)
    box = np.zeros(count, dtype=float)
    box_multiplier = np.zeros(count, dtype=float)
    started = perf_counter()
    primal_residual = float("inf")
    dual_residual = float("inf")
    completed = 0
    for index in range(int(iterations)):
        right = penalty * (
            coefficients.T @ (gram - operands.base_vector - gram_multiplier)
            + (box - box_multiplier)
        )
        values = linalg.cho_solve(factor, right)
        affine = operands.base_vector + coefficients @ values
        candidate = _unsvec_upper(affine + gram_multiplier, side)
        eigenvalues, vectors = np.linalg.eigh(candidate)
        projected = (
            vectors * np.maximum(eigenvalues, 0.0)[None, :]
        ) @ vectors.T
        previous = gram
        gram = _clarabel_svec_upper(projected)
        previous_box = box
        box = np.clip(values + box_multiplier, -1.0, 1.0)
        gram_multiplier = gram_multiplier + affine - gram
        box_multiplier = box_multiplier + values - box
        primal_residual = float(
            max(
                np.linalg.norm(affine - gram, ord=np.inf),
                np.linalg.norm(values - box, ord=np.inf),
            )
        )
        dual_residual = float(
            penalty
            * max(
                np.linalg.norm(
                    coefficients.T @ (gram - previous), ord=np.inf
                ),
                np.linalg.norm(box - previous_box, ord=np.inf),
            )
        )
        completed = index + 1
        if max(primal_residual, dual_residual) <= tolerance:
            break
    elapsed = perf_counter() - started
    frontier_values = values * operands.extension.frontier_scales
    frontier = {
        key: float(value)
        for key, value in zip(
            operands.extension.frontier_keys, frontier_values, strict=True
        )
    }
    matrix = operands.extension.matrix(operands.lower_moments, frontier)
    minimum = float(
        np.linalg.eigvalsh(operands.extension.scaled_matrix(matrix))[0]
    )
    return FirstOrderPass(
        standardized_values=values,
        iterations=completed,
        wall_seconds=elapsed,
        primal_residual=primal_residual,
        dual_residual=dual_residual,
        minimum_scaled_eigenvalue=minimum,
    )


def solve_outer_target_first_order_polish(
    operands: OuterTargetOperands,
    *,
    frontier_weight: float = 0.0,
    penalty: float = 1.0,
    admm_iterations: int = 400,
    retry_profile: int = 0,
    time_limit_seconds: float | None = None,
) -> FormulationOutcome:
    """First-order pass, then a Clarabel polish shifted to that point.

    The polish is the displacement program of the eliminated stage: the same
    cone, the linear term ``P z0`` and the cone right-hand side ``b - A z0``
    of the first-order point ``z0``.  The reported point is ``z0 + dz``.
    This is the shift pattern ``_solve_explicit_target_qp`` already documents
    for its named warm refinement pass, started from a first-order point
    instead of from Clarabel's own.
    """

    first = psd_affine_admm(
        operands,
        frontier_weight=frontier_weight,
        penalty=penalty,
        iterations=admm_iterations,
    )
    count = operands.frontier_count
    side = operands.gram_side
    metric = np.asarray(operands.target_metric, dtype=float)
    whitened = (
        np.asarray(linalg.cholesky(metric, lower=False), dtype=float)
        @ operands.scaled_readout
    )
    quadratic_dense = whitened.T @ whitened
    if frontier_weight > 0.0:
        quadratic_dense = quadratic_dense + float(frontier_weight) * np.eye(
            count
        )
    affine = [sparse.csc_matrix(-operands.coefficient_vectors)]
    rhs = [operands.base_vector]
    cones: list[Any] = [clarabel.PSDTriangleConeT(side)]
    if operands.boxed:
        affine.append(
            sparse.vstack(
                (
                    sparse.eye(count, format="csc"),
                    -sparse.eye(count, format="csc"),
                ),
                format="csc",
            )
        )
        rhs.append(np.ones(2 * count))
        cones.append(clarabel.NonnegativeConeT(2 * count))
    cone_matrix = sparse.vstack(affine, format="csc")
    cone_rhs = np.concatenate(rhs)
    start = np.asarray(first.standardized_values, dtype=float)
    shifted_rhs = cone_rhs - np.asarray(cone_matrix @ start, dtype=float)
    linear = np.asarray(quadratic_dense @ start, dtype=float)
    settings = _recorded_settings(
        operands.extension,
        retry_profile,
        time_limit_seconds=time_limit_seconds,
    )
    started = perf_counter()
    solution = clarabel.DefaultSolver(
        sparse.csc_matrix(quadratic_dense),
        linear,
        cone_matrix,
        shifted_rhs,
        cones,
        settings,
    ).solve()
    elapsed = perf_counter() - started
    status = str(solution.status)
    displacement = (
        np.asarray(solution.x, dtype=float)
        if solution.x is not None
        else np.zeros(count)
    )
    values = start + displacement
    dual = (
        np.asarray(solution.z, dtype=float)
        if solution.z is not None
        else None
    )
    psd_rows = operands.coefficient_vectors.shape[0]
    dual_psd = None if dual is None else dual[:psd_rows]
    dual_box = None if dual is None else dual[psd_rows:]
    result = evaluate_on_recorded_program(
        operands,
        values,
        dual_psd=dual_psd,
        dual_box=dual_box,
        status=status,
        iterations=int(solution.iterations),
        solver_primal_residual=float(solution.r_prim),
        solver_dual_residual=float(solution.r_dual),
        frontier_weight=frontier_weight,
        status_prefix="explicit_target_first_order_polish",
    )
    return FormulationOutcome(
        formulation="first_order_polish",
        options={
            "frontier_weight": float(frontier_weight),
            "penalty": float(penalty),
            "admm_iterations": int(admm_iterations),
            "retry_profile": int(retry_profile),
        },
        result=result,
        solver_status=status,
        wall_seconds=elapsed + first.wall_seconds,
        solver_iterations=int(solution.iterations),
        solver_primal_residual=float(solution.r_prim),
        solver_dual_residual=float(solution.r_dual),
        frontier_weight=float(frontier_weight),
        notes={
            "first_order_iterations": int(first.iterations),
            "first_order_wall_seconds": float(first.wall_seconds),
            "first_order_primal_residual": float(first.primal_residual),
            "first_order_dual_residual": float(first.dual_residual),
            "first_order_minimum_scaled_eigenvalue": float(
                first.minimum_scaled_eigenvalue
            ),
            "polish_wall_seconds": float(elapsed),
        },
    )


# ---------------------------------------------------------------------------
# named formulations
# ---------------------------------------------------------------------------


#: The six alternatives measured in round 3, each a named option whose
#: operands are the recorded ones.  ``recorded`` is the current formulation.
NAMED_FORMULATIONS: Mapping[str, OuterTargetFormulation] = MappingProxyType(
    {
        "recorded_kernel": OuterTargetFormulation(),
        "equilibrated_whitened": OuterTargetFormulation(
            absorb_whitening=True,
            gram_congruence="ruiz",
            frontier_column_scaling="ruiz",
        ),
        "recorded_kernel_retry2": OuterTargetFormulation(retry_profile=2),
        "nullspace_eliminated_retry2": OuterTargetFormulation(
            eliminate_image_equality=True, retry_profile=2
        ),
        "whitened": OuterTargetFormulation(absorb_whitening=True),
        "gram_congruence": OuterTargetFormulation(gram_congruence="ruiz"),
        "frontier_column_scaling": OuterTargetFormulation(
            frontier_column_scaling="ruiz"
        ),
        "whitened_retry2": OuterTargetFormulation(
            absorb_whitening=True, retry_profile=2
        ),
        "equilibrated_whitened_retry2": OuterTargetFormulation(
            absorb_whitening=True,
            gram_congruence="ruiz",
            frontier_column_scaling="ruiz",
            retry_profile=2,
        ),
        "equilibrated_whitened_inflation_1em9": OuterTargetFormulation(
            absorb_whitening=True,
            gram_congruence="ruiz",
            frontier_column_scaling="ruiz",
            psd_inflation=1e-9,
        ),
        "equilibrated_whitened_inflation_1em8": OuterTargetFormulation(
            absorb_whitening=True,
            gram_congruence="ruiz",
            frontier_column_scaling="ruiz",
            psd_inflation=1e-8,
        ),
        "psd_inflation_1em11": OuterTargetFormulation(psd_inflation=1e-11),
        "psd_inflation_1em10": OuterTargetFormulation(psd_inflation=1e-10),
        "psd_inflation_1em9": OuterTargetFormulation(psd_inflation=1e-9),
        "psd_inflation_1em8": OuterTargetFormulation(psd_inflation=1e-8),
        "nullspace_eliminated": OuterTargetFormulation(
            eliminate_image_equality=True
        ),
        "box_rescaled_1em1": OuterTargetFormulation(
            homogeneous_frontier_scale=1e-1
        ),
        "box_rescaled_1ep1": OuterTargetFormulation(
            homogeneous_frontier_scale=1e1
        ),
        "box_rescaled_1ep1_retry2": OuterTargetFormulation(
            homogeneous_frontier_scale=1e1, retry_profile=2
        ),
    }
)


def solve_outer_target_with_tightened_retry(
    extension: SymmetryReducedPositiveExtension,
    lower_moments: Mapping[MomentKey, float],
    scaled_readout: FloatArray,
    target_metric: FloatArray,
    *,
    boxed: bool = True,
    ladder: bool = True,
    retry_profile: int = 2,
    whiten_image: bool = False,
    frontier_weight: float = 0.0,
    time_limit_seconds: float | None = None,
) -> GuardConicResult:
    """The outer target stage with the tightened retry actually run.

    Same signature and return shape as ``_solve_explicit_target_qp``, so it
    can be wired as a named selector option.  The conic program is the
    recorded one; what changes is that the recorded routine's own tightened
    numerical profile is reached.  ``retry_profile`` names that profile: ``0``
    is the recorded first pass, ``2`` the recorded retry (conic tolerance
    ``1e-10``, ``max_iter`` 500, 20 iterative refinement passes at ``1e-14``,
    equilibration off).  With ``ladder=True``, the default, the first pass
    runs at profile ``0`` and the tightened pass runs whenever that pass is
    not accepted.  The recorded routine retries only on the statuses
    ``{"AlmostSolved", "InsufficientProgress", "MaxIterations"}``
    (``apcm_projective_guard.py:1493``), so a first pass ending in
    ``NumericalError`` is returned as it stands and the tightened profile is
    never run; that is the measured reason the recorded stage fails at the
    cutoff-16 correlated preparation and at a forced-trigger stage's own lower
    moments.  ``ladder=False`` runs the tightened profile alone, at half the
    wall time and the same accepted point.

    ``whiten_image`` is a named option, off by default, that carries the
    invariant target image in its whitened coordinates ``t = U R z``, with
    ``U`` the upper Cholesky factor of ``M = U^T U``, so the objective is the
    identity quadratic ``0.5 ||t||^2``.  Because ``M`` is the inverse of the
    readout covariance ``R R^T``, ``U R`` has orthonormal rows and
    ``R^T M R`` is the orthogonal projector onto the row space of ``R``; the
    recorded program hands Clarabel the same projector split into a dense
    metric and an unnormalized readout.  Whitening improves the first pass by
    two decades on the relative duality gap and leaves the tightened pass
    where it is, so it is measured, named and left off.

    Nothing in the acceptance rule is changed: the returned
    ``GuardConicResult`` is scored on the recorded program's own operands by
    ``_guard_acceptance`` at its unchanged exploratory tolerance.
    """

    operands = outer_target_operands_from_program(
        extension,
        lower_moments,
        scaled_readout,
        target_metric,
        name="tightened_retry_target_stage",
        boxed=boxed,
    )

    def _pass(profile: int) -> GuardConicResult:
        return solve_outer_target_formulation(
            operands,
            OuterTargetFormulation(
                absorb_whitening=bool(whiten_image),
                retry_profile=int(profile),
                frontier_weight=float(frontier_weight),
                time_limit_seconds=time_limit_seconds,
            ),
        ).result

    if not ladder:
        return _pass(retry_profile)
    first = _pass(0)
    if first.success:
        return first
    return _pass(retry_profile)


def solve_named_formulation(
    operands: OuterTargetOperands,
    name: str,
    *,
    frontier_weight: float = 0.0,
) -> FormulationOutcome:
    """Solve one entry of ``NAMED_FORMULATIONS`` at a declared weight."""

    if name not in NAMED_FORMULATIONS:
        raise KeyError(f"unknown outer target formulation: {name}")
    spec = replace(
        NAMED_FORMULATIONS[name], frontier_weight=float(frontier_weight)
    )
    outcome = solve_outer_target_formulation(operands, spec)
    return replace(outcome, formulation=name)


__all__ = [
    "FirstOrderPass",
    "FormulationOutcome",
    "NAMED_FORMULATIONS",
    "OuterTargetFormulation",
    "OuterTargetOperands",
    "build_outer_target_operands",
    "evaluate_on_recorded_program",
    "gram_magnitude",
    "outer_target_operands_from_program",
    "psd_affine_admm",
    "recorded_program",
    "ruiz_congruence",
    "save_outer_target_operands",
    "solve_named_formulation",
    "solve_outer_target_dual",
    "solve_outer_target_first_order_polish",
    "solve_outer_target_formulation",
    "solve_recorded_outer_target",
    "solve_outer_target_with_tightened_retry",
]
