"""Projective-guard foundations for archive-backed APCM dynamics.

This module implements the pre-rollout portion of the URPG amendment:

* a Hamiltonian-family readout of the invariant terminal image used by the
  active commutator equations;
* a literal retained-prefix electronic/phonon Gram with distinct spin rows;
* exact index restriction maps and the relative-mode congruence; and
* boxed or unboxed minimum-norm positive-completion solves conditioned on the
  invariant image rather than individual frontier coordinates.

It intentionally contains no time integrator or promotion policy.  The
projective preparation audit must pass before either is scientifically
meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping, Sequence

import clarabel
import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse

from .adaptive_positive_moment import (
    ENTRANCE_RELATIVE_MOMENT_KEYS,
    HIDDEN_RELATIVE_MOMENT_KEYS,
    uncentered_joint_moment_matrix,
)
from .apcm_positive_extension import (
    SymmetryReducedPositiveExtension,
    _clarabel_svec_upper,
    _realify_hermitian,
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
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


_HAMILTONIAN_OPERATOR_BASIS: tuple[tuple[str, _OperatorKey], ...] = (
    ("hop_up", _OperatorKey(PAULI_X, IDENTITY, 0, 0)),
    ("hop_down", _OperatorKey(IDENTITY, PAULI_X, 0, 0)),
    ("bias_up", _OperatorKey(PAULI_Z, IDENTITY, 0, 0)),
    ("bias_down", _OperatorKey(IDENTITY, PAULI_Z, 0, 0)),
    ("phonon_x2", _OperatorKey(IDENTITY, IDENTITY, 2, 0)),
    ("phonon_p2", _OperatorKey(IDENTITY, IDENTITY, 0, 2)),
    ("coupling_up", _OperatorKey(PAULI_Z, IDENTITY, 1, 0)),
    ("coupling_down", _OperatorKey(IDENTITY, PAULI_Z, 1, 0)),
)

_SPIN_LABELS = (PAULI_X, PAULI_Y, PAULI_Z)


def _fraction(value: float, *, denominator: int = 2**20) -> Fraction:
    """Return the exact dyadic/rational coefficient used by this compiler."""

    result = Fraction(float(value)).limit_denominator(denominator)
    if abs(float(result) - float(value)) > 2e-13:
        raise ValueError(f"coefficient {value!r} is not a supported rational")
    return result


def _rref(
    matrix: Sequence[Sequence[Fraction]],
) -> tuple[list[list[Fraction]], tuple[int, ...]]:
    """Compute deterministic exact reduced row echelon form."""

    if not matrix:
        return [], ()
    work = [list(row) for row in matrix]
    column_count = len(work[0])
    if any(len(row) != column_count for row in work):
        raise ValueError("RREF matrix rows have inconsistent lengths")
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(column_count):
        selected = next(
            (
                row
                for row in range(pivot_row, len(work))
                if work[row][column] != 0
            ),
            None,
        )
        if selected is None:
            continue
        work[pivot_row], work[selected] = work[selected], work[pivot_row]
        pivot = work[pivot_row][column]
        work[pivot_row] = [value / pivot for value in work[pivot_row]]
        for row in range(len(work)):
            if row == pivot_row:
                continue
            multiplier = work[row][column]
            if multiplier == 0:
                continue
            work[row] = [
                value - multiplier * pivot_value
                for value, pivot_value in zip(
                    work[row], work[pivot_row], strict=True
                )
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(work):
            break
    return work, tuple(pivot_columns)


def _independent_row_indices(
    rows: Sequence[Sequence[Fraction]],
) -> tuple[int, ...]:
    if not rows:
        return ()
    transposed = [list(column) for column in zip(*rows, strict=True)]
    _, pivots = _rref(transposed)
    return pivots


def _nullspace_basis(
    matrix: Sequence[Sequence[Fraction]],
) -> tuple[tuple[Fraction, ...], ...]:
    if not matrix:
        return ()
    reduced, pivots = _rref(matrix)
    column_count = len(matrix[0])
    free_columns = [column for column in range(column_count) if column not in pivots]
    basis: list[tuple[Fraction, ...]] = []
    for free in free_columns:
        vector = [Fraction(0) for _ in range(column_count)]
        vector[free] = Fraction(1)
        for row, pivot in enumerate(pivots):
            vector[pivot] = -reduced[row][free]
        basis.append(tuple(vector))
    return tuple(basis)


def _moment_sort_key(key: MomentKey) -> tuple[int, str, str, int, int]:
    return (
        key.degree,
        key.spin_up,
        key.spin_down,
        key.x_power,
        key.p_power,
    )


def _registry_hash(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class InvariantTargetReadout:
    """Basis-independent terminal image of the active commutator targets."""

    active_keys: tuple[MomentKey, ...]
    frontier_keys: tuple[MomentKey, ...]
    raw_rows: tuple[tuple[Fraction, ...], ...]
    raw_row_labels: tuple[str, ...]
    independent_row_indices: tuple[int, ...]
    rational_matrix: tuple[tuple[Fraction, ...], ...]
    matrix: FloatArray
    nullspace: tuple[tuple[Fraction, ...], ...]
    registry_hash: str

    @property
    def rank(self) -> int:
        return len(self.rational_matrix)

    def image(self, frontier: Mapping[MomentKey, float]) -> FloatArray:
        missing = set(self.frontier_keys).difference(frontier)
        if missing:
            raise ValueError(
                f"target image is missing {len(missing)} frontier moments"
            )
        values = np.asarray(
            [float(frontier[key]) for key in self.frontier_keys], dtype=float
        )
        return self.matrix @ values


def compile_invariant_target_readout(
    extension: SymmetryReducedPositiveExtension,
) -> InvariantTargetReadout:
    """Compile the structural Hamiltonian-family image ``q_D=R_D y``.

    The eight Hamiltonian words are kept separate.  Consequently the row
    space cannot lose directions because a drive coefficient vanishes at one
    time or because two protocol coefficients cancel numerically.
    """

    frontier_index = {
        key: index for index, key in enumerate(extension.frontier_keys)
    }
    rows: list[tuple[Fraction, ...]] = []
    labels: list[str] = []
    for hamiltonian_label, hamiltonian_word in _HAMILTONIAN_OPERATOR_BASIS:
        for active_index, observable in enumerate(extension.active_keys):
            row = [Fraction(0) for _ in extension.frontier_keys]
            for generated, coefficient in _commutator(
                hamiltonian_word, observable
            ).items():
                index = frontier_index.get(generated)
                if index is None:
                    continue
                derivative_coefficient = 1j * coefficient
                if abs(derivative_coefficient.imag) > 2e-13:
                    raise ValueError(
                        "Hermitian commutator generated a complex target "
                        f"coefficient for {generated}: {derivative_coefficient}"
                    )
                row[index] += _fraction(derivative_coefficient.real)
            if any(value != 0 for value in row):
                rows.append(tuple(row))
                labels.append(f"{hamiltonian_label}:{active_index}")

    independent = _independent_row_indices(rows)
    reduced_rows = tuple(rows[index] for index in independent)
    matrix = np.asarray(
        [[float(value) for value in row] for row in reduced_rows],
        dtype=float,
    )
    nullspace = _nullspace_basis(reduced_rows)
    payload = {
        "active": [key.__dict__ for key in extension.active_keys],
        "frontier": [key.__dict__ for key in extension.frontier_keys],
        "labels": labels,
        "pivots": independent,
        "matrix": [
            [[value.numerator, value.denominator] for value in row]
            for row in reduced_rows
        ],
    }
    return InvariantTargetReadout(
        active_keys=extension.active_keys,
        frontier_keys=extension.frontier_keys,
        raw_rows=tuple(rows),
        raw_row_labels=tuple(labels),
        independent_row_indices=independent,
        rational_matrix=reduced_rows,
        matrix=matrix,
        nullspace=nullspace,
        registry_hash=_registry_hash(payload),
    )


@dataclass(frozen=True)
class EntranceSourceAudit:
    """Static audit of the reduced entrance chart used by the K/P/D source."""

    entrance_keys: tuple[MomentKey, ...]
    omitted_hidden_keys: tuple[MomentKey, ...]
    entrance_rank: int
    registry_hash: str


def compile_entrance_source_audit() -> EntranceSourceAudit:
    """Certify the fixed independent entrance coordinates in canonical keys.

    The 15 coordinates are literal distinct Hermitian Pauli--Weyl basis
    elements: six two-spin moments and nine one-spin/two-phonon moments.  The
    K/P/D decoder accesses no other hidden degree-three key; this dependency is
    regression-tested directly against the full chart.
    """

    entrance = tuple(ENTRANCE_RELATIVE_MOMENT_KEYS)
    omitted = tuple(
        key for key in HIDDEN_RELATIVE_MOMENT_KEYS if key not in set(entrance)
    )
    if len(set(entrance)) != len(entrance):
        raise RuntimeError("entrance chart contains duplicate canonical keys")
    payload = {
        "entrance": [key.__dict__ for key in entrance],
        "omitted": [key.__dict__ for key in omitted],
    }
    return EntranceSourceAudit(
        entrance_keys=entrance,
        omitted_hidden_keys=omitted,
        entrance_rank=len(entrance),
        registry_hash=_registry_hash(payload),
    )


def unified_core_moment_matrix(
    raw_coordinates: FloatArray,
    moments: Mapping[MomentKey, float],
) -> ComplexArray:
    """Return the literal 11-row core Gram with the retained 8-row prefix.

    Row order is ``(I,b0,b1,b0^dagger,b1^dagger,up XYZ,down XYZ)``.
    Spin exchange fixes each down one-body/boson entry to its up counterpart;
    the six independent up--down Pauli entries are supplied by the entrance
    moments.
    """

    retained = uncentered_joint_moment_matrix(raw_coordinates)
    result = np.empty((11, 11), dtype=complex)
    result[:8, :8] = retained
    result[:5, 8:11] = retained[:5, 5:8]
    result[8:11, :5] = result[:5, 8:11].conjugate().T
    result[8:11, 8:11] = retained[5:8, 5:8]
    for row, left in enumerate(_SPIN_LABELS):
        for column, right in enumerate(_SPIN_LABELS):
            labels = tuple(
                sorted(
                    (left, right),
                    key=(IDENTITY, PAULI_X, PAULI_Y, PAULI_Z).index,
                )
            )
            key = MomentKey(labels[0], labels[1], 0, 0)
            try:
                result[5 + row, 8 + column] = float(moments[key])
            except KeyError as error:
                raise ValueError(
                    f"unified core is missing two-spin moment {key}"
                ) from error
    result[8:11, 5:8] = result[5:8, 8:11].conjugate().T
    return np.asarray(0.5 * (result + result.conjugate().T), dtype=complex)


def retained_prefix_restriction() -> FloatArray:
    """Restrict the unified core to its literal retained 8-row prefix."""

    result = np.zeros((8, 11), dtype=float)
    result[:, :8] = np.eye(8)
    return result


def relative_core_restriction() -> ComplexArray:
    """Map the unified core to ``(I,a,a^dagger,up XYZ,down XYZ)``."""

    result = np.zeros((9, 11), dtype=complex)
    result[0, 0] = 1.0
    root_two_inverse = 1.0 / np.sqrt(2.0)
    result[1, 1] = root_two_inverse
    result[1, 2] = -root_two_inverse
    result[2, 3] = root_two_inverse
    result[2, 4] = -root_two_inverse
    result[3:6, 5:8] = np.eye(3)
    result[6:9, 8:11] = np.eye(3)
    return result


def relative_hermitian_core_restriction() -> ComplexArray:
    """Map the core to the extension order ``(I,up,down,x_rel,p_rel)``."""

    result = np.zeros((9, 11), dtype=complex)
    result[0, 0] = 1.0
    result[1:4, 5:8] = np.eye(3)
    result[4:7, 8:11] = np.eye(3)
    result[7, 1:5] = np.asarray([0.5, -0.5, 0.5, -0.5])
    result[8, 1:5] = np.asarray([-0.5j, 0.5j, 0.5j, -0.5j])
    return result


def center_core_null_directions() -> ComplexArray:
    """Return the two center-mode core directions absent from the relative list."""

    result = np.zeros((11, 2), dtype=complex)
    result[1:5, 0] = np.asarray([0.5, 0.5, 0.5, 0.5])
    result[1:5, 1] = np.asarray([-0.5j, -0.5j, 0.5j, 0.5j])
    restriction = relative_hermitian_core_restriction()
    if np.linalg.norm(restriction.conjugate() @ result) > 2e-15:
        raise RuntimeError("center directions are not in the relative-map kernel")
    if np.linalg.norm(result.conjugate().T @ result - np.eye(2)) > 2e-15:
        raise RuntimeError("center directions are not orthonormal")
    return result


def unified_glued_moment_matrix(
    core_matrix: ComplexArray,
    relative_matrix: ComplexArray,
    *,
    center_cross: ComplexArray | None = None,
) -> ComplexArray:
    """Glue a relative extension to the literal core in one positive Gram.

    The relative matrix begins with the nine Hermitian rows
    ``(I,up,down,x_rel,p_rel)``.  Its remaining rows are retained literally.
    The two complex center-mode cross rows are free completion data; zero is a
    deterministic fixture, not a physical factorization assertion.
    """

    core = np.asarray(core_matrix, dtype=complex)
    relative = np.asarray(relative_matrix, dtype=complex)
    if core.shape != (11, 11):
        raise ValueError("literal core matrix must have shape (11,11)")
    if relative.ndim != 2 or relative.shape[0] != relative.shape[1]:
        raise ValueError("relative extension matrix must be square")
    if relative.shape[0] < 9:
        raise ValueError("relative extension does not contain its nine-row core")
    restriction = relative_hermitian_core_restriction()
    expected_base = restriction.conjugate() @ core @ restriction.T
    if not np.allclose(relative[:9, :9], expected_base, atol=2e-10, rtol=0.0):
        raise ValueError("relative and literal core blocks are inconsistent")
    extra = relative.shape[0] - 9
    free = (
        np.zeros((2, extra), dtype=complex)
        if center_cross is None
        else np.asarray(center_cross, dtype=complex)
    )
    if free.shape != (2, extra):
        raise ValueError(f"center_cross must have shape {(2, extra)}")
    relative_cross = relative[:9, 9:]
    particular = restriction.T @ relative_cross
    full_cross = particular + center_core_null_directions() @ free
    result = np.block(
        [
            [core, full_cross],
            [full_cross.conjugate().T, relative[9:, 9:]],
        ]
    )
    return np.asarray(0.5 * (result + result.conjugate().T), dtype=complex)


def canonical_psd_center_cross(
    core_matrix: ComplexArray,
    relative_matrix: ComplexArray,
    *,
    relative_tolerance: float = 1e-7,
) -> ComplexArray:
    """Return the canonical PSD completion between center and outer rows.

    The literal core and relative extension are two positive cliques sharing
    the nine relative-core rows.  Their chordal Gram completion is obtained by
    matching both cliques through the Moore--Penrose inverse of that shared
    block.  No additional physical moment or fitted parameter is introduced.
    """

    core = np.asarray(core_matrix, dtype=complex)
    relative = np.asarray(relative_matrix, dtype=complex)
    if core.shape != (11, 11):
        raise ValueError("literal core matrix must have shape (11,11)")
    if relative.ndim != 2 or relative.shape[0] != relative.shape[1]:
        raise ValueError("relative extension matrix must be square")
    if relative.shape[0] < 9:
        raise ValueError("relative extension does not contain its nine-row core")
    restriction = relative_hermitian_core_restriction()
    center = center_core_null_directions()
    overlap = restriction.conjugate() @ core @ restriction.T
    if not np.allclose(
        relative[:9, :9], overlap, atol=2e-10, rtol=0.0
    ):
        raise ValueError("relative and literal core blocks are inconsistent")
    inverse = np.linalg.pinv(overlap, rcond=1e-12, hermitian=True)
    relative_cross = relative[:9, 9:]
    center_overlap = center.conjugate().T @ core @ restriction.T
    left_range_error = float(
        np.linalg.norm(
            center_overlap - center_overlap @ inverse @ overlap,
            ord=np.inf,
        )
    )
    right_range_error = float(
        np.linalg.norm(
            relative_cross - overlap @ inverse @ relative_cross,
            ord=np.inf,
        )
    )
    if max(left_range_error, right_range_error) > relative_tolerance:
        raise ValueError(
            "shared relative block does not support a stable PSD clique completion"
        )
    return np.asarray(center_overlap @ inverse @ relative_cross, dtype=complex)


def unified_to_relative_restriction(relative_dimension: int) -> ComplexArray:
    """Restrict a glued ``11+(p-9)`` Gram back to its relative ``p`` rows."""

    if relative_dimension < 9:
        raise ValueError("relative_dimension must be at least nine")
    extra = relative_dimension - 9
    result = np.zeros((relative_dimension, 11 + extra), dtype=complex)
    result[:9, :11] = relative_hermitian_core_restriction()
    result[9:, 11:] = np.eye(extra)
    return result


def unified_guard_dimension(
    extension: SymmetryReducedPositiveExtension,
) -> int:
    """Return the literal-core guard dimension corresponding to an extension."""

    if extension.dimension < 9:
        raise ValueError("relative extension is missing its nine-row base")
    return 11 + (extension.dimension - 9)


def relative_core_moment_matrix(
    moments: Mapping[MomentKey, float],
) -> ComplexArray:
    """Assemble the same 9-row relative Gram directly from Pauli--Weyl data."""

    hermitian_words = (
        _OperatorKey(IDENTITY, IDENTITY, 0, 0),
        *tuple(
            _OperatorKey(label, IDENTITY, 0, 0) for label in _SPIN_LABELS
        ),
        *tuple(
            _OperatorKey(IDENTITY, label, 0, 0) for label in _SPIN_LABELS
        ),
        _OperatorKey(IDENTITY, IDENTITY, 1, 0),
        _OperatorKey(IDENTITY, IDENTITY, 0, 1),
    )
    base = np.zeros((9, 9), dtype=complex)
    for row, left in enumerate(hermitian_words):
        for column, right in enumerate(hermitian_words):
            value = 0.0j
            for key, coefficient in _operator_product(left, right).items():
                value += coefficient * (
                    1.0 if key.degree == 0 else float(moments[key])
                )
            base[row, column] = value
    transform = np.zeros((9, 9), dtype=complex)
    transform[0, 0] = 1.0
    transform[1, 7] = 1.0 / np.sqrt(2.0)
    transform[1, 8] = 1.0j / np.sqrt(2.0)
    transform[2, 7] = 1.0 / np.sqrt(2.0)
    transform[2, 8] = -1.0j / np.sqrt(2.0)
    transform[3:9, 1:7] = np.eye(6)
    # Gram entries use M[i,j]=<v_i^dagger v_j>.  For w=T v the induced
    # transformation is therefore T.conjugate() M T.T.
    result = transform.conjugate() @ base @ transform.T
    return np.asarray(0.5 * (result + result.conjugate().T), dtype=complex)


def prefix_restriction(
    child: Sequence[object], parent: Sequence[object]
) -> FloatArray:
    """Return the exact index restriction for one literal nested registry."""

    parent_index = {value: index for index, value in enumerate(parent)}
    if len(parent_index) != len(parent):
        raise ValueError("parent registry contains duplicate keys")
    result = np.zeros((len(child), len(parent)), dtype=float)
    for row, value in enumerate(child):
        try:
            result[row, parent_index[value]] = 1.0
        except KeyError as error:
            raise ValueError("child registry is not contained in parent") from error
    return result


def prefix_union(
    old: Sequence[MomentKey], new: Sequence[MomentKey]
) -> tuple[MomentKey, ...]:
    """Preserve every old key and append one deterministically sorted batch."""

    result = list(old)
    present = set(result)
    for key in sorted(set(new).difference(present), key=_moment_sort_key):
        result.append(key)
        present.add(key)
    return tuple(result)


@dataclass(frozen=True)
class GuardConicResult:
    """One direct QP+PSD solve on a fixed positive extension."""

    success: bool
    status: str
    boxed: bool
    conditioned: bool
    standardized_values: FloatArray
    frontier_moments: Mapping[MomentKey, float]
    moment_matrix: ComplexArray
    minimum_scaled_eigenvalue: float
    primal_residual: float
    dual_residual: float
    objective: float
    iterations: int
    certified: bool = False
    provisional: bool = False
    acceptance: str = "rejected"
    independent_feasibility_residual: float = float("inf")
    stationarity_residual: float = float("inf")
    complementarity_residual: float = float("inf")
    relative_duality_gap: float = float("inf")


@dataclass(frozen=True)
class _IndependentKKTDiagnostics:
    feasibility: float
    stationarity: float
    complementarity: float
    relative_gap: float


# This gate permits exploratory continuation without claiming a conic
# certificate.  It is deliberately much tighter than the model discrepancies
# being investigated, while allowing high-quality ``AlmostSolved`` iterates.
_EXPLORATORY_ACCEPTANCE_TOLERANCE = 1e-6


def _independent_kkt_diagnostics(
    solution: object,
    variables: FloatArray,
    quadratic: sparse.spmatrix | FloatArray,
    linear: FloatArray,
    cone_matrix: sparse.spmatrix,
    cone_rhs: FloatArray,
    *,
    direct_feasibility: float,
) -> _IndependentKKTDiagnostics:
    """Recompute KKT defects without trusting Clarabel's status label."""

    dual = getattr(solution, "z", None)
    slack = getattr(solution, "s", None)
    if dual is None or slack is None:
        return _IndependentKKTDiagnostics(
            feasibility=float("inf"),
            stationarity=float("inf"),
            complementarity=float("inf"),
            relative_gap=float("inf"),
        )
    x = np.asarray(variables, dtype=float)
    z = np.asarray(dual, dtype=float)
    s = np.asarray(slack, dtype=float)
    q = np.asarray(linear, dtype=float)
    b = np.asarray(cone_rhs, dtype=float)
    primal_equation = np.asarray(cone_matrix @ x, dtype=float) + s - b
    conic_feasibility = float(
        np.linalg.norm(primal_equation, ord=np.inf)
        / (1.0 + np.linalg.norm(b, ord=np.inf))
    )
    px = np.asarray(quadratic @ x, dtype=float)
    dual_term = np.asarray(cone_matrix.T @ z, dtype=float)
    stationarity_vector = px + q + dual_term
    stationarity_scale = (
        1.0
        + np.linalg.norm(px, ord=np.inf)
        + np.linalg.norm(q, ord=np.inf)
        + np.linalg.norm(dual_term, ord=np.inf)
    )
    stationarity = float(
        np.linalg.norm(stationarity_vector, ord=np.inf) / stationarity_scale
    )
    complementarity = float(
        abs(float(np.dot(s, z)))
        / (1.0 + np.linalg.norm(s) * np.linalg.norm(z))
    )
    primal_objective = getattr(solution, "obj_val", float("nan"))
    dual_objective = getattr(solution, "obj_val_dual", float("nan"))
    if np.isfinite(primal_objective) and np.isfinite(dual_objective):
        relative_gap = float(
            abs(float(primal_objective) - float(dual_objective))
            / (
                1.0
                + abs(float(primal_objective))
                + abs(float(dual_objective))
            )
        )
    else:
        relative_gap = float("inf")
    return _IndependentKKTDiagnostics(
        feasibility=max(float(direct_feasibility), conic_feasibility),
        stationarity=stationarity,
        complementarity=complementarity,
        relative_gap=relative_gap,
    )


def _guard_acceptance(
    status: str,
    *,
    strict_success: bool,
    diagnostics: _IndependentKKTDiagnostics,
) -> tuple[bool, bool, bool, str]:
    """Separate certified solves from exploratory numerical acceptance."""

    certified = bool(strict_success)
    provisional = bool(
        not certified
        and status == "AlmostSolved"
        and max(
            diagnostics.feasibility,
            diagnostics.stationarity,
            diagnostics.complementarity,
            diagnostics.relative_gap,
        )
        <= _EXPLORATORY_ACCEPTANCE_TOLERANCE
    )
    if certified:
        acceptance = "certified"
    elif provisional:
        acceptance = "provisional_independent_kkt"
    else:
        acceptance = "rejected"
    return certified or provisional, certified, provisional, acceptance


def _empty_guard_result(
    extension: SymmetryReducedPositiveExtension,
    lower_moments: Mapping[MomentKey, float],
    *,
    status: str,
    boxed: bool,
    conditioned: bool,
    independent_feasibility: float,
) -> GuardConicResult:
    """Return one explicit rejected result without fabricating a witness."""

    del lower_moments
    return GuardConicResult(
        success=False,
        status=status,
        boxed=boxed,
        conditioned=conditioned,
        standardized_values=np.full(
            len(extension.frontier_keys), np.nan, dtype=float
        ),
        frontier_moments=MappingProxyType({}),
        moment_matrix=np.empty((0, 0), dtype=complex),
        minimum_scaled_eigenvalue=float("nan"),
        primal_residual=float("inf"),
        dual_residual=float("inf"),
        objective=float("inf"),
        iterations=0,
        independent_feasibility_residual=float(independent_feasibility),
    )


def _solve_guard_qp_on_relative_face(
    extension: SymmetryReducedPositiveExtension,
    lower_moments: Mapping[MomentKey, float],
    quadratic: FloatArray,
    linear: FloatArray,
    *,
    scaled_base: ComplexArray,
    scaled_coefficients: ComplexArray,
    boxed: bool,
    equality_matrix: FloatArray | None,
    equality_target: FloatArray | None,
) -> GuardConicResult | None:
    """Retry one boxed problem on the extension's certified relative face."""

    if not boxed:
        return None
    cone_matrix, cone_rhs, cones = extension._completion_cone_data(
        scaled_base, scaled_coefficients
    )
    facial = extension._facially_reduced_reference(
        scaled_base,
        scaled_coefficients,
        cone_matrix,
        cone_rhs,
        cones,
    )
    if not facial.success or facial.directions.shape[1] == 0:
        return None
    anchor = np.asarray(facial.values, dtype=float)
    directions = np.asarray(facial.directions, dtype=float)
    support = np.asarray(facial.support, dtype=complex)
    reduced_count = directions.shape[1]
    affine_base = scaled_base + np.tensordot(
        anchor, scaled_coefficients, axes=(0, 0)
    )
    reduced_base = support.conjugate().T @ affine_base @ support
    directional_coefficients = np.einsum(
        "ik,iab->kab", directions, scaled_coefficients, optimize=True
    )
    reduced_coefficients = np.einsum(
        "ai,kab,bj->kij",
        support.conjugate(),
        directional_coefficients,
        support,
        optimize=True,
    )
    real_base = _realify_hermitian(reduced_base)
    coefficient_vectors = np.column_stack(
        [
            _clarabel_svec_upper(_realify_hermitian(coefficient))
            for coefficient in reduced_coefficients
        ]
    )
    blocks: list[sparse.csc_matrix] = [sparse.csc_matrix(-coefficient_vectors)]
    rhs: list[FloatArray] = [_clarabel_svec_upper(real_base)]
    reduced_cones: list[object] = [clarabel.PSDTriangleConeT(real_base.shape[0])]
    blocks.extend(
        [sparse.csc_matrix(directions), sparse.csc_matrix(-directions)]
    )
    rhs.extend([np.ones(anchor.size) - anchor, np.ones(anchor.size) + anchor])
    reduced_cones.append(clarabel.NonnegativeConeT(2 * anchor.size))
    conditioned = equality_matrix is not None
    if conditioned:
        equality = np.asarray(equality_matrix, dtype=float)
        target = np.asarray(equality_target, dtype=float)
        reduced_equality = equality @ directions
        reduced_target = target - equality @ anchor
        blocks.append(sparse.csc_matrix(reduced_equality))
        rhs.append(reduced_target)
        reduced_cones.append(clarabel.ZeroConeT(reduced_equality.shape[0]))
    reduced_quadratic = directions.T @ quadratic @ directions
    reduced_linear = directions.T @ (quadratic @ anchor + linear)
    reduced_quadratic = 0.5 * (reduced_quadratic + reduced_quadratic.T)
    rows, columns = np.triu_indices(reduced_count)
    packed_quadratic = sparse.coo_matrix(
        (
            reduced_quadratic[rows, columns],
            (rows, columns),
        ),
        shape=(reduced_count, reduced_count),
    ).tocsc()
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = extension.settings.clarabel_max_threads
    settings.max_iter = 1000
    settings.tol_gap_abs = 1e-10
    settings.tol_gap_rel = 1e-10
    settings.tol_feas = 1e-10
    settings.tol_infeas_abs = 1e-10
    settings.tol_infeas_rel = 1e-10
    settings.equilibrate_enable = True
    settings.equilibrate_max_iter = 10
    settings.equilibrate_min_scaling = 1e-4
    settings.equilibrate_max_scaling = 1e4
    settings.iterative_refinement_enable = True
    settings.iterative_refinement_max_iter = 20
    settings.iterative_refinement_abstol = 1e-14
    settings.iterative_refinement_reltol = 1e-14
    settings.static_regularization_enable = True
    settings.static_regularization_constant = 1e-12
    settings.dynamic_regularization_enable = True
    settings.dynamic_regularization_eps = 1e-13
    settings.dynamic_regularization_delta = 2e-7
    settings.chordal_decomposition_enable = False
    solution = clarabel.DefaultSolver(
        packed_quadratic,
        reduced_linear,
        sparse.vstack(blocks, format="csc"),
        np.concatenate(rhs),
        reduced_cones,
        settings,
    ).solve()
    status = str(solution.status)
    if solution.x is None:
        return None
    theta = np.asarray(solution.x, dtype=float)
    values = anchor + directions @ theta
    frontier_values = values * extension.frontier_scales
    frontier = MappingProxyType(
        {
            key: float(value)
            for key, value in zip(
                extension.frontier_keys, frontier_values, strict=True
            )
        }
    )
    matrix = extension.matrix(lower_moments, frontier)
    scaled = extension.scaled_matrix(matrix)
    minimum = float(np.linalg.eigvalsh(scaled)[0])
    primal = float(solution.r_prim)
    dual = float(solution.r_dual)
    tolerance = extension.settings.conic_tolerance
    equality_error = 0.0
    if conditioned:
        equality_error = float(
            np.linalg.norm(equality @ values - target, ord=np.inf)
        )
    strict_success = bool(
        status == "Solved"
        and max(primal, dual, equality_error) <= 10.0 * tolerance
        and minimum >= -10.0 * tolerance
        and float(np.max(np.abs(values))) <= 1.0 + 10.0 * tolerance
        and facial.qualification_error <= 10.0 * tolerance
    )
    direct_feasibility = max(
        max(0.0, -minimum),
        max(0.0, float(np.max(np.abs(values))) - 1.0),
        equality_error,
        facial.qualification_error,
    )
    variables = theta
    full_quadratic = reduced_quadratic
    diagnostics = _independent_kkt_diagnostics(
        solution,
        variables,
        full_quadratic,
        reduced_linear,
        sparse.vstack(blocks, format="csc"),
        np.concatenate(rhs),
        direct_feasibility=direct_feasibility,
    )
    success, certified, provisional, acceptance = _guard_acceptance(
        status,
        strict_success=strict_success,
        diagnostics=diagnostics,
    )
    return GuardConicResult(
        success=success,
        status=f"relative_face:{status}",
        boxed=True,
        conditioned=conditioned,
        standardized_values=values,
        frontier_moments=frontier,
        moment_matrix=matrix,
        minimum_scaled_eigenvalue=minimum,
        primal_residual=max(primal, facial.qualification_error),
        dual_residual=dual,
        objective=float(
            0.5 * values @ quadratic @ values + linear @ values
        ),
        iterations=int(solution.iterations + facial.iterations),
        certified=certified,
        provisional=provisional,
        acceptance=acceptance,
        independent_feasibility_residual=diagnostics.feasibility,
        stationarity_residual=diagnostics.stationarity,
        complementarity_residual=diagnostics.complementarity,
        relative_duality_gap=diagnostics.relative_gap,
    )


def _solve_guard_qp(
    extension: SymmetryReducedPositiveExtension,
    lower_moments: Mapping[MomentKey, float],
    quadratic: FloatArray,
    linear: FloatArray,
    *,
    boxed: bool,
    equality_matrix: FloatArray | None = None,
    equality_target: FloatArray | None = None,
    _retry_profile: int = 0,
) -> GuardConicResult:
    """Solve one native Clarabel QP on standardized frontier coordinates."""

    count = len(extension.frontier_keys)
    quadratic = np.asarray(quadratic, dtype=float)
    linear = np.asarray(linear, dtype=float)
    if quadratic.shape != (count, count) or linear.shape != (count,):
        raise ValueError("guard objective has the wrong dimension")
    missing = set(extension.lower_keys).difference(lower_moments)
    if missing:
        raise ValueError(f"guard solve is missing {len(missing)} lower moments")
    lower_values = np.asarray(
        [float(lower_moments[key]) for key in extension.lower_keys], dtype=float
    )
    unscaled_base = extension._constant + np.tensordot(
        lower_values, extension.lower_coefficients, axes=(0, 0)
    )
    scaled_base = extension.scaled_matrix(unscaled_base)
    scaled_coefficients = (
        extension.scaled_frontier_coefficients
        * extension.frontier_scales[:, None, None]
    )
    real_base = _realify_hermitian(scaled_base)
    coefficient_vectors = np.column_stack(
        [
            _clarabel_svec_upper(_realify_hermitian(coefficient))
            for coefficient in scaled_coefficients
        ]
    )
    blocks: list[sparse.csc_matrix] = [sparse.csc_matrix(-coefficient_vectors)]
    rhs: list[FloatArray] = [_clarabel_svec_upper(real_base)]
    cones: list[object] = [clarabel.PSDTriangleConeT(real_base.shape[0])]
    if boxed:
        blocks.extend(
            [sparse.eye(count, format="csc"), -sparse.eye(count, format="csc")]
        )
        rhs.extend([np.ones(count), np.ones(count)])
        cones.append(clarabel.NonnegativeConeT(2 * count))
    conditioned = equality_matrix is not None
    if conditioned:
        equality = np.asarray(equality_matrix, dtype=float)
        target = np.asarray(equality_target, dtype=float)
        if equality.ndim != 2 or equality.shape[1] != count:
            raise ValueError("guard equality matrix has the wrong dimension")
        if target.shape != (equality.shape[0],):
            raise ValueError("guard equality target has the wrong dimension")
        blocks.append(sparse.csc_matrix(equality))
        rhs.append(target)
        cones.append(clarabel.ZeroConeT(equality.shape[0]))
    cone_matrix = sparse.vstack(blocks, format="csc")
    cone_rhs = np.concatenate(rhs)

    quadratic = 0.5 * (quadratic + quadratic.T)
    row, column = np.triu_indices(count)
    packed_quadratic = sparse.coo_matrix(
        (quadratic[row, column], (row, column)), shape=(count, count)
    ).tocsc()
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = (
        extension.settings.maximum_iterations if _retry_profile == 0 else 500
    )
    settings.max_threads = extension.settings.clarabel_max_threads
    solve_tolerance = (
        extension.settings.conic_tolerance if _retry_profile == 0 else 1e-10
    )
    settings.tol_gap_abs = solve_tolerance
    settings.tol_gap_rel = solve_tolerance
    settings.tol_feas = solve_tolerance
    settings.tol_infeas_abs = solve_tolerance
    settings.tol_infeas_rel = solve_tolerance
    settings.equilibrate_enable = _retry_profile == 0
    settings.equilibrate_max_iter = 10
    settings.equilibrate_min_scaling = 1e-4
    settings.equilibrate_max_scaling = 1e4
    settings.iterative_refinement_enable = True
    settings.iterative_refinement_max_iter = 10 if _retry_profile == 0 else 20
    settings.iterative_refinement_abstol = 1e-12 if _retry_profile == 0 else 1e-14
    settings.iterative_refinement_reltol = 1e-12 if _retry_profile == 0 else 1e-14
    settings.static_regularization_enable = True
    settings.static_regularization_constant = 1e-12
    settings.dynamic_regularization_enable = True
    settings.dynamic_regularization_eps = 1e-13
    settings.dynamic_regularization_delta = 2e-7
    settings.chordal_decomposition_enable = False
    solution = clarabel.DefaultSolver(
        packed_quadratic,
        linear,
        cone_matrix,
        cone_rhs,
        cones,
        settings,
    ).solve()
    status = str(solution.status)
    if status in {"AlmostSolved", "InsufficientProgress", "MaxIterations"} and _retry_profile == 0:
        # R2 of the frozen retry ladder: reconstruct and refactor with
        # equilibration disabled and tighter tolerances.  AlmostSolved is not
        # silently promoted to a scientific certificate.
        return _solve_guard_qp(
            extension,
            lower_moments,
            quadratic,
            linear,
            boxed=boxed,
            equality_matrix=equality_matrix,
            equality_target=equality_target,
            _retry_profile=2,
        )
    if status == "AlmostSolved" and _retry_profile == 2:
        reduced = _solve_guard_qp_on_relative_face(
            extension,
            lower_moments,
            quadratic,
            linear,
            scaled_base=scaled_base,
            scaled_coefficients=scaled_coefficients,
            boxed=boxed,
            equality_matrix=equality_matrix,
            equality_target=equality_target,
        )
        if reduced is not None:
            return reduced
    values = (
        np.asarray(solution.x, dtype=float)
        if solution.x is not None
        else np.zeros(count, dtype=float)
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
    matrix = extension.matrix(lower_moments, frontier)
    scaled = extension.scaled_matrix(matrix)
    minimum = float(np.linalg.eigvalsh(scaled)[0])
    primal = float(solution.r_prim)
    dual = float(solution.r_dual)
    tolerance = extension.settings.conic_tolerance
    equality_error = (
        0.0
        if not conditioned
        else float(np.linalg.norm(equality @ values - target, ord=np.inf))
    )
    box_error = (
        0.0
        if not boxed
        else max(0.0, float(np.max(np.abs(values))) - 1.0)
    )
    strict_success = bool(
        status == "Solved"
        and max(primal, dual) <= 10.0 * tolerance
        and minimum >= -10.0 * tolerance
        and (
            not boxed
            or float(np.max(np.abs(values))) <= 1.0 + 10.0 * tolerance
        )
        and (
            not conditioned
            or equality_error <= 10.0 * tolerance
        )
    )
    diagnostics = _independent_kkt_diagnostics(
        solution,
        values,
        quadratic,
        linear,
        cone_matrix,
        cone_rhs,
        direct_feasibility=max(max(0.0, -minimum), box_error, equality_error),
    )
    success, certified, provisional, acceptance = _guard_acceptance(
        status,
        strict_success=strict_success,
        diagnostics=diagnostics,
    )
    return GuardConicResult(
        success=success,
        status=status,
        boxed=boxed,
        conditioned=conditioned,
        standardized_values=values,
        frontier_moments=frontier,
        moment_matrix=matrix,
        minimum_scaled_eigenvalue=minimum,
        primal_residual=primal,
        dual_residual=dual,
        objective=float(solution.obj_val),
        iterations=int(solution.iterations),
        certified=certified,
        provisional=provisional,
        acceptance=acceptance,
        independent_feasibility_residual=diagnostics.feasibility,
        stationarity_residual=diagnostics.stationarity,
        complementarity_residual=diagnostics.complementarity,
        relative_duality_gap=diagnostics.relative_gap,
    )


def _solve_explicit_target_qp(
    extension: SymmetryReducedPositiveExtension,
    lower_moments: Mapping[MomentKey, float],
    scaled_readout: FloatArray,
    target_metric: FloatArray,
    *,
    boxed: bool,
    retry_profile: int = 0,
    accept_first_provisional: bool = False,
) -> GuardConicResult:
    """Solve the target stage with ``q`` as an explicit decision variable."""

    count = len(extension.frontier_keys)
    target_count = scaled_readout.shape[0]
    lower_values = np.asarray(
        [float(lower_moments[key]) for key in extension.lower_keys], dtype=float
    )
    unscaled_base = extension._constant + np.tensordot(
        lower_values, extension.lower_coefficients, axes=(0, 0)
    )
    scaled_base = extension.scaled_matrix(unscaled_base)
    scaled_coefficients = (
        extension.scaled_frontier_coefficients
        * extension.frontier_scales[:, None, None]
    )
    real_base = _realify_hermitian(scaled_base)
    coefficient_vectors = np.column_stack(
        [
            _clarabel_svec_upper(_realify_hermitian(coefficient))
            for coefficient in scaled_coefficients
        ]
    )
    psd_block = sparse.hstack(
        (
            sparse.csc_matrix(-coefficient_vectors),
            sparse.csc_matrix((coefficient_vectors.shape[0], target_count)),
        ),
        format="csc",
    )
    blocks: list[sparse.csc_matrix] = [psd_block]
    rhs: list[FloatArray] = [_clarabel_svec_upper(real_base)]
    cones: list[object] = [clarabel.PSDTriangleConeT(real_base.shape[0])]
    if boxed:
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
        (-sparse.csc_matrix(scaled_readout), sparse.eye(target_count, format="csc")),
        format="csc",
    )
    blocks.append(image_equality)
    rhs.append(np.zeros(target_count))
    cones.append(clarabel.ZeroConeT(target_count))

    variable_count = count + target_count
    quadratic = sparse.block_diag(
        (sparse.csc_matrix((count, count)), sparse.csc_matrix(target_metric)),
        format="csc",
    )
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = extension.settings.clarabel_max_threads
    settings.max_iter = extension.settings.maximum_iterations if retry_profile == 0 else 500
    solve_tolerance = extension.settings.conic_tolerance if retry_profile == 0 else 1e-10
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
    cone_matrix = sparse.vstack(blocks, format="csc")
    cone_rhs = np.concatenate(rhs)
    solution = clarabel.DefaultSolver(
        quadratic,
        np.zeros(variable_count),
        cone_matrix,
        cone_rhs,
        cones,
        settings,
    ).solve()
    status = str(solution.status)
    values = (
        np.asarray(solution.x[:count], dtype=float)
        if solution.x is not None
        else np.zeros(count, dtype=float)
    )
    target = (
        np.asarray(solution.x[count:], dtype=float)
        if solution.x is not None
        else np.full(target_count, np.nan)
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
    matrix = extension.matrix(lower_moments, frontier)
    scaled = extension.scaled_matrix(matrix)
    minimum = float(np.linalg.eigvalsh(scaled)[0])
    primal = float(solution.r_prim)
    dual = float(solution.r_dual)
    tolerance = extension.settings.conic_tolerance
    image_error = float(
        np.linalg.norm(target - scaled_readout @ values, ord=np.inf)
    )
    box_error = (
        0.0
        if not boxed
        else max(0.0, float(np.max(np.abs(values))) - 1.0)
    )
    strict_success = bool(
        status == "Solved"
        and max(primal, dual, image_error) <= 10.0 * tolerance
        and minimum >= -10.0 * tolerance
        and (
            not boxed
            or float(np.max(np.abs(values))) <= 1.0 + 10.0 * tolerance
        )
    )
    variables = np.concatenate((values, target))
    diagnostics = _independent_kkt_diagnostics(
        solution,
        variables,
        quadratic,
        np.zeros(variable_count),
        cone_matrix,
        cone_rhs,
        direct_feasibility=max(
            max(0.0, -minimum), box_error, image_error
        ),
    )
    success, certified, provisional, acceptance = _guard_acceptance(
        status,
        strict_success=strict_success,
        diagnostics=diagnostics,
    )
    result = GuardConicResult(
        success=success,
        status=f"explicit_target:{status}",
        boxed=boxed,
        conditioned=False,
        standardized_values=values,
        frontier_moments=frontier,
        moment_matrix=matrix,
        minimum_scaled_eigenvalue=minimum,
        primal_residual=max(primal, image_error),
        dual_residual=dual,
        objective=float(0.5 * target @ target_metric @ target),
        iterations=int(solution.iterations),
        certified=certified,
        provisional=provisional,
        acceptance=acceptance,
        independent_feasibility_residual=diagnostics.feasibility,
        stationarity_residual=diagnostics.stationarity,
        complementarity_residual=diagnostics.complementarity,
        relative_duality_gap=diagnostics.relative_gap,
    )
    retry_statuses = {"AlmostSolved", "InsufficientProgress", "MaxIterations"}
    if retry_profile == 0 and status in retry_statuses:
        if accept_first_provisional and result.provisional:
            return result
        return _solve_explicit_target_qp(
            extension,
            lower_moments,
            scaled_readout,
            target_metric,
            boxed=boxed,
            retry_profile=2,
            accept_first_provisional=accept_first_provisional,
        )
    if (
        retry_profile == 2
        and status == "AlmostSolved"
        and boxed
        and not (accept_first_provisional and result.provisional)
    ):
        reduced = _solve_explicit_target_on_relative_face(
            extension,
            lower_moments,
            scaled_readout,
            target_metric,
            scaled_base=scaled_base,
            scaled_coefficients=scaled_coefficients,
        )
        if reduced is not None:
            return reduced
    return result


def _solve_explicit_target_on_relative_face(
    extension: SymmetryReducedPositiveExtension,
    lower_moments: Mapping[MomentKey, float],
    scaled_readout: FloatArray,
    target_metric: FloatArray,
    *,
    scaled_base: ComplexArray,
    scaled_coefficients: ComplexArray,
) -> GuardConicResult | None:
    """R3 explicit-target solve after relative-face qualification."""

    cone_matrix, cone_rhs, cones = extension._completion_cone_data(
        scaled_base, scaled_coefficients
    )
    facial = extension._facially_reduced_reference(
        scaled_base,
        scaled_coefficients,
        cone_matrix,
        cone_rhs,
        cones,
    )
    if not facial.success or facial.directions.shape[1] == 0:
        return None
    anchor = np.asarray(facial.values, dtype=float)
    directions = np.asarray(facial.directions, dtype=float)
    support = np.asarray(facial.support, dtype=complex)
    reduced_count = directions.shape[1]
    target_count = scaled_readout.shape[0]
    affine_base = scaled_base + np.tensordot(
        anchor, scaled_coefficients, axes=(0, 0)
    )
    reduced_base = support.conjugate().T @ affine_base @ support
    directional_coefficients = np.einsum(
        "ik,iab->kab", directions, scaled_coefficients, optimize=True
    )
    reduced_coefficients = np.einsum(
        "ai,kab,bj->kij",
        support.conjugate(),
        directional_coefficients,
        support,
        optimize=True,
    )
    real_base = _realify_hermitian(reduced_base)
    coefficient_vectors = np.column_stack(
        [
            _clarabel_svec_upper(_realify_hermitian(coefficient))
            for coefficient in reduced_coefficients
        ]
    )
    psd_block = sparse.hstack(
        (
            sparse.csc_matrix(-coefficient_vectors),
            sparse.csc_matrix((coefficient_vectors.shape[0], target_count)),
        ),
        format="csc",
    )
    box_theta = sparse.vstack(
        (sparse.csc_matrix(directions), sparse.csc_matrix(-directions)),
        format="csc",
    )
    box_block = sparse.hstack(
        (box_theta, sparse.csc_matrix((2 * anchor.size, target_count))),
        format="csc",
    )
    image_equality = sparse.hstack(
        (
            -sparse.csc_matrix(scaled_readout @ directions),
            sparse.eye(target_count, format="csc"),
        ),
        format="csc",
    )
    image_rhs = scaled_readout @ anchor
    variable_count = reduced_count + target_count
    quadratic = sparse.block_diag(
        (
            sparse.csc_matrix((reduced_count, reduced_count)),
            sparse.csc_matrix(target_metric),
        ),
        format="csc",
    )
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = extension.settings.clarabel_max_threads
    settings.max_iter = 1000
    settings.tol_gap_abs = 1e-10
    settings.tol_gap_rel = 1e-10
    settings.tol_feas = 1e-10
    settings.tol_infeas_abs = 1e-10
    settings.tol_infeas_rel = 1e-10
    settings.equilibrate_enable = True
    settings.equilibrate_max_iter = 10
    settings.equilibrate_min_scaling = 1e-4
    settings.equilibrate_max_scaling = 1e4
    settings.iterative_refinement_enable = True
    settings.iterative_refinement_max_iter = 20
    settings.iterative_refinement_abstol = 1e-14
    settings.iterative_refinement_reltol = 1e-14
    settings.static_regularization_enable = True
    settings.static_regularization_constant = 1e-12
    settings.dynamic_regularization_enable = True
    settings.dynamic_regularization_eps = 1e-13
    settings.dynamic_regularization_delta = 2e-7
    settings.chordal_decomposition_enable = False
    solution = clarabel.DefaultSolver(
        quadratic,
        np.zeros(variable_count),
        sparse.vstack((psd_block, box_block, image_equality), format="csc"),
        np.concatenate(
            (
                _clarabel_svec_upper(real_base),
                np.ones(anchor.size) - anchor,
                np.ones(anchor.size) + anchor,
                image_rhs,
            )
        ),
        [
            clarabel.PSDTriangleConeT(real_base.shape[0]),
            clarabel.NonnegativeConeT(2 * anchor.size),
            clarabel.ZeroConeT(target_count),
        ],
        settings,
    ).solve()
    if solution.x is None:
        return None
    status = str(solution.status)
    if status != "Solved":
        # Preserve the better R2 diagnostic when the facial retry itself is
        # unresolved; neither status is promoted to a certificate.
        return None
    theta = np.asarray(solution.x[:reduced_count], dtype=float)
    target = np.asarray(solution.x[reduced_count:], dtype=float)
    values = anchor + directions @ theta
    frontier_values = values * extension.frontier_scales
    frontier = MappingProxyType(
        {
            key: float(value)
            for key, value in zip(
                extension.frontier_keys, frontier_values, strict=True
            )
        }
    )
    matrix = extension.matrix(lower_moments, frontier)
    scaled = extension.scaled_matrix(matrix)
    minimum = float(np.linalg.eigvalsh(scaled)[0])
    image_error = float(
        np.linalg.norm(target - scaled_readout @ values, ord=np.inf)
    )
    primal = max(float(solution.r_prim), facial.qualification_error, image_error)
    dual = float(solution.r_dual)
    tolerance = extension.settings.conic_tolerance
    strict_success = bool(
        status == "Solved"
        and max(primal, dual) <= 10.0 * tolerance
        and minimum >= -10.0 * tolerance
        and float(np.max(np.abs(values))) <= 1.0 + 10.0 * tolerance
    )
    diagnostics = _IndependentKKTDiagnostics(
        feasibility=max(
            primal,
            max(0.0, -minimum),
            max(0.0, float(np.max(np.abs(values))) - 1.0),
        ),
        stationarity=dual,
        complementarity=0.0,
        relative_gap=0.0,
    )
    success, certified, provisional, acceptance = _guard_acceptance(
        status,
        strict_success=strict_success,
        diagnostics=diagnostics,
    )
    return GuardConicResult(
        success=success,
        status=f"explicit_target_relative_face:{status}",
        boxed=True,
        conditioned=False,
        standardized_values=values,
        frontier_moments=frontier,
        moment_matrix=matrix,
        minimum_scaled_eigenvalue=minimum,
        primal_residual=primal,
        dual_residual=dual,
        objective=float(0.5 * target @ target_metric @ target),
        iterations=int(solution.iterations + facial.iterations),
        certified=certified,
        provisional=provisional,
        acceptance=acceptance,
        independent_feasibility_residual=diagnostics.feasibility,
        stationarity_residual=diagnostics.stationarity,
        complementarity_residual=diagnostics.complementarity,
        relative_duality_gap=diagnostics.relative_gap,
    )


@dataclass(frozen=True)
class ProjectiveGuardSelection:
    """Unique target-image selector followed by its minimum-norm lift."""

    target_image: FloatArray
    target_metric: FloatArray
    target_stage: GuardConicResult
    lift_stage: GuardConicResult


@dataclass(frozen=True)
class OuterFeasibleProjectiveGuardSelection:
    """Target image and lift selected on one common positive outer shell."""

    target_image: FloatArray
    target_metric: FloatArray
    outer_extension: SymmetryReducedPositiveExtension
    outer_target_stage: GuardConicResult
    outer_lift_stage: GuardConicResult
    current_standardized_values: FloatArray
    current_frontier_moments: Mapping[MomentKey, float]
    current_moment_matrix: ComplexArray
    current_minimum_scaled_eigenvalue: float
    witness_source: str

    @property
    def success(self) -> bool:
        return bool(
            self.outer_target_stage.success
            and np.all(np.isfinite(self.current_standardized_values))
        )

    @property
    def canonical_lift(self) -> bool:
        return self.witness_source == "minimum_norm_outer_lift"


@dataclass(frozen=True)
class FrozenOuterFaceSelector:
    """Fast local selector on the PSD face exposed at preparation."""

    extension: SymmetryReducedPositiveExtension
    scaled_readout: FloatArray
    target_metric: FloatArray
    support: ComplexArray
    null_basis: ComplexArray
    face_left_basis: FloatArray
    face_singular_values: FloatArray
    face_right_basis: FloatArray
    face_directions: FloatArray
    scaled_coefficients: ComplexArray
    boxed: bool
    consistency_tolerance: float = 1e-6

    @classmethod
    def from_selection(
        cls,
        current: SymmetryReducedPositiveExtension,
        readout: InvariantTargetReadout,
        selection: OuterFeasibleProjectiveGuardSelection,
        *,
        boxed: bool = True,
        eigenvalue_tolerance: float = 1e-10,
        consistency_tolerance: float = 1e-6,
    ) -> "FrozenOuterFaceSelector":
        """Compile one fixed face from an independently accepted witness."""

        if not selection.success:
            raise ValueError("outer-feasible selection did not produce a witness")
        outer = selection.outer_extension
        scaled_readout = _outer_current_image_matrix(current, outer, readout)
        return cls.from_witness(
            outer,
            scaled_readout,
            selection.target_metric,
            selection.outer_target_stage,
            boxed=boxed,
            eigenvalue_tolerance=eigenvalue_tolerance,
            consistency_tolerance=consistency_tolerance,
        )

    @classmethod
    def from_witness(
        cls,
        extension: SymmetryReducedPositiveExtension,
        scaled_readout: FloatArray,
        target_metric: FloatArray,
        witness_result: GuardConicResult,
        *,
        boxed: bool = True,
        eigenvalue_tolerance: float = 1e-10,
        consistency_tolerance: float = 1e-6,
    ) -> "FrozenOuterFaceSelector":
        """Compile a fixed face from one accepted result on that cone."""

        if not witness_result.success:
            raise ValueError("face witness was not independently accepted")
        witness = extension.scaled_matrix(witness_result.moment_matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(witness)
        threshold = max(
            float(eigenvalue_tolerance),
            float(eigenvalue_tolerance) * max(1.0, float(eigenvalues[-1])),
        )
        null_count = int(np.count_nonzero(eigenvalues <= threshold))
        if null_count == 0 or null_count == extension.dimension:
            raise ValueError("preparation witness does not expose a usable face")
        null_basis = np.asarray(eigenvectors[:, :null_count], dtype=complex)
        support = np.asarray(eigenvectors[:, null_count:], dtype=complex)
        scaled_coefficients = (
            extension.scaled_frontier_coefficients
            * extension.frontier_scales[:, None, None]
        )
        equality_map = np.column_stack(
            [
                (coefficient @ null_basis).reshape(-1)
                for coefficient in scaled_coefficients
            ]
        )
        real_equality_map = np.vstack(
            (equality_map.real, equality_map.imag)
        )
        left, singular_values, right = np.linalg.svd(
            real_equality_map, full_matrices=False
        )
        rank_tolerance = max(
            1e-12,
            100.0
            * np.finfo(float).eps
            * max(real_equality_map.shape)
            * (singular_values[0] if singular_values.size else 1.0),
        )
        rank = int(np.count_nonzero(singular_values > rank_tolerance))
        if rank == 0:
            raise ValueError("exposed face has no affine frontier constraints")
        if right.shape[0] < real_equality_map.shape[1]:
            directions = linalg.null_space(
                real_equality_map,
                rcond=rank_tolerance
                / max(singular_values[0], np.finfo(float).tiny),
            )
        else:
            directions = right[rank:, :].T
        return cls(
            extension=extension,
            scaled_readout=np.asarray(scaled_readout, dtype=float),
            target_metric=np.asarray(target_metric, dtype=float),
            support=support,
            null_basis=null_basis,
            face_left_basis=np.asarray(left[:, :rank], dtype=float),
            face_singular_values=np.asarray(
                singular_values[:rank], dtype=float
            ),
            face_right_basis=np.asarray(right[:rank, :], dtype=float),
            face_directions=np.asarray(directions, dtype=float),
            scaled_coefficients=np.asarray(scaled_coefficients, dtype=complex),
            boxed=bool(boxed),
            consistency_tolerance=float(consistency_tolerance),
        )

    @property
    def face_rank(self) -> int:
        return int(self.face_singular_values.size)

    @property
    def direction_count(self) -> int:
        return int(self.face_directions.shape[1])

    def solve(
        self,
        lower_moments: Mapping[MomentKey, float],
    ) -> GuardConicResult:
        """Solve the invariant target on the frozen face at one new state."""

        missing = set(self.extension.lower_keys).difference(lower_moments)
        if missing:
            raise ValueError(
                f"frozen-face solve is missing {len(missing)} lower moments"
            )
        lower_values = np.asarray(
            [
                float(lower_moments[key])
                for key in self.extension.lower_keys
            ],
            dtype=float,
        )
        unscaled_base = self.extension._constant + np.tensordot(
            lower_values,
            self.extension.lower_coefficients,
            axes=(0, 0),
        )
        scaled_base = self.extension.scaled_matrix(unscaled_base)
        base_null = (scaled_base @ self.null_basis).reshape(-1)
        real_base_null = np.concatenate((base_null.real, base_null.imag))
        projected = self.face_left_basis.T @ real_base_null
        reconstructed = self.face_left_basis @ projected
        consistency = float(
            np.linalg.norm(real_base_null - reconstructed, ord=np.inf)
            / (1.0 + np.linalg.norm(real_base_null, ord=np.inf))
        )
        count = len(self.extension.frontier_keys)
        if consistency > self.consistency_tolerance:
            return _empty_guard_result(
                self.extension,
                lower_moments,
                status="frozen_face:inconsistent",
                boxed=self.boxed,
                conditioned=False,
                independent_feasibility=consistency,
            )
        particular = -self.face_right_basis.T @ (
            projected / self.face_singular_values
        )
        directions = self.face_directions
        direction_count = directions.shape[1]
        affine_base = scaled_base + np.tensordot(
            particular, self.scaled_coefficients, axes=(0, 0)
        )
        reduced_base = (
            self.support.conjugate().T @ affine_base @ self.support
        )
        directional_coefficients = np.einsum(
            "ik,iab->kab",
            directions,
            self.scaled_coefficients,
            optimize=True,
        )
        reduced_coefficients = np.einsum(
            "ai,kab,bj->kij",
            self.support.conjugate(),
            directional_coefficients,
            self.support,
            optimize=True,
        )
        real_base = _realify_hermitian(reduced_base)
        if direction_count:
            coefficient_vectors = np.column_stack(
                [
                    _clarabel_svec_upper(_realify_hermitian(coefficient))
                    for coefficient in reduced_coefficients
                ]
            )
        else:
            coefficient_vectors = np.empty(
                (_clarabel_svec_upper(real_base).size, 0), dtype=float
            )
        target_count = self.scaled_readout.shape[0]
        psd_block = sparse.hstack(
            (
                sparse.csc_matrix(-coefficient_vectors),
                sparse.csc_matrix((coefficient_vectors.shape[0], target_count)),
            ),
            format="csc",
        )
        blocks: list[sparse.csc_matrix] = [psd_block]
        rhs: list[FloatArray] = [_clarabel_svec_upper(real_base)]
        cones: list[object] = [clarabel.PSDTriangleConeT(real_base.shape[0])]
        if self.boxed:
            box_block = sparse.hstack(
                (
                    sparse.vstack(
                        (
                            sparse.csc_matrix(directions),
                            sparse.csc_matrix(-directions),
                        ),
                        format="csc",
                    ),
                    sparse.csc_matrix((2 * count, target_count)),
                ),
                format="csc",
            )
            blocks.append(box_block)
            rhs.append(
                np.concatenate((1.0 - particular, 1.0 + particular))
            )
            cones.append(clarabel.NonnegativeConeT(2 * count))
        image_offset = self.scaled_readout @ particular
        image_directions = self.scaled_readout @ directions
        target_whitener = np.linalg.cholesky(self.target_metric).T
        whitened_offset = target_whitener @ image_offset
        whitened_directions = target_whitener @ image_directions
        image_equality = sparse.hstack(
            (
                -sparse.csc_matrix(whitened_directions),
                sparse.eye(target_count, format="csc"),
            ),
            format="csc",
        )
        blocks.append(image_equality)
        rhs.append(whitened_offset)
        cones.append(clarabel.ZeroConeT(target_count))
        variable_count = direction_count + target_count
        quadratic = sparse.block_diag(
            (
                sparse.csc_matrix((direction_count, direction_count)),
                sparse.eye(target_count, format="csc"),
            ),
            format="csc",
        )
        linear = np.zeros(variable_count)
        cone_matrix = sparse.vstack(blocks, format="csc")
        cone_rhs = np.concatenate(rhs)
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_threads = self.extension.settings.clarabel_max_threads
        settings.max_iter = 200
        settings.tol_gap_abs = 1e-9
        settings.tol_gap_rel = 1e-9
        settings.tol_feas = 1e-9
        settings.tol_infeas_abs = 1e-9
        settings.tol_infeas_rel = 1e-9
        settings.equilibrate_enable = True
        settings.iterative_refinement_enable = True
        settings.iterative_refinement_max_iter = 10
        settings.chordal_decomposition_enable = False
        solution = clarabel.DefaultSolver(
            quadratic,
            linear,
            cone_matrix,
            cone_rhs,
            cones,
            settings,
        ).solve()
        status = str(solution.status)
        if solution.x is None:
            return _empty_guard_result(
                self.extension,
                lower_moments,
                status=f"frozen_face:{status}",
                boxed=self.boxed,
                conditioned=False,
                independent_feasibility=float("inf"),
            )
        solution_values = np.asarray(solution.x, dtype=float)
        theta = solution_values[:direction_count]
        whitened_target = solution_values[direction_count:]
        target = np.linalg.solve(target_whitener, whitened_target)
        values = particular + directions @ theta
        frontier_values = values * self.extension.frontier_scales
        frontier = MappingProxyType(
            {
                key: float(value)
                for key, value in zip(
                    self.extension.frontier_keys,
                    frontier_values,
                    strict=True,
                )
            }
        )
        matrix = self.extension.matrix(lower_moments, frontier)
        scaled = self.extension.scaled_matrix(matrix)
        minimum = float(np.linalg.eigvalsh(scaled)[0])
        image_error = float(
            np.linalg.norm(target - self.scaled_readout @ values, ord=np.inf)
        )
        box_error = (
            0.0
            if not self.boxed
            else max(0.0, float(np.max(np.abs(values))) - 1.0)
        )
        face_error = float(
            np.linalg.norm(scaled @ self.null_basis, ord=np.inf)
        )
        diagnostics = _independent_kkt_diagnostics(
            solution,
            solution_values,
            quadratic,
            linear,
            cone_matrix,
            cone_rhs,
            direct_feasibility=max(
                max(0.0, -minimum),
                image_error,
                box_error,
                face_error,
                consistency,
            ),
        )
        strict_success = bool(
            status == "Solved"
            and max(
                diagnostics.feasibility,
                diagnostics.stationarity,
                diagnostics.complementarity,
                diagnostics.relative_gap,
            )
            <= 1e-8
        )
        success, certified, provisional, acceptance = _guard_acceptance(
            status,
            strict_success=strict_success,
            diagnostics=diagnostics,
        )
        if (
            not success
            and status == "AlmostSolved"
            and max(
                diagnostics.feasibility,
                diagnostics.stationarity,
                diagnostics.complementarity,
                diagnostics.relative_gap,
            )
            <= 1e-5
        ):
            success = True
            provisional = True
            acceptance = "provisional_frozen_face_kkt"
        return GuardConicResult(
            success=success,
            status=f"frozen_face:{status}",
            boxed=self.boxed,
            conditioned=False,
            standardized_values=values,
            frontier_moments=frontier,
            moment_matrix=matrix,
            minimum_scaled_eigenvalue=minimum,
            primal_residual=float(solution.r_prim),
            dual_residual=float(solution.r_dual),
            objective=float(0.5 * target @ self.target_metric @ target),
            iterations=int(solution.iterations),
            certified=certified,
            provisional=provisional,
            acceptance=acceptance,
            independent_feasibility_residual=diagnostics.feasibility,
            stationarity_residual=diagnostics.stationarity,
            complementarity_residual=diagnostics.complementarity,
            relative_duality_gap=diagnostics.relative_gap,
        )


@dataclass(frozen=True)
class CuttingPlaneGuardResult:
    """One PSD-certified selector result and its reusable eigenvector cuts."""

    guard: GuardConicResult
    cut_eigenvectors: ComplexArray
    cut_count: int
    quadratic_lower_bound: float

    @property
    def success(self) -> bool:
        return self.guard.success


@dataclass(frozen=True)
class ProjectiveGuardCuttingPlaneSelector:
    """Exact PSD selector via successively violated eigenvector inequalities."""

    extension: SymmetryReducedPositiveExtension
    scaled_readout: FloatArray
    target_metric: FloatArray
    boxed: bool = True
    psd_tolerance: float = 1e-8
    maximum_cuts: int = 512
    frontier_regularization: float = 1e-8

    @classmethod
    def for_outer_guard(
        cls,
        current: SymmetryReducedPositiveExtension,
        readout: InvariantTargetReadout,
        *,
        boxed: bool = True,
        psd_tolerance: float = 1e-8,
        maximum_cuts: int = 512,
        frontier_regularization: float = 1e-8,
    ) -> "ProjectiveGuardCuttingPlaneSelector":
        outer = projective_guard_outer_extension(current)
        current_scaled_readout = (
            readout.matrix * current.frontier_scales[None, :]
        )
        covariance = current_scaled_readout @ current_scaled_readout.T
        return cls(
            extension=outer,
            scaled_readout=_outer_current_image_matrix(
                current, outer, readout
            ),
            target_metric=np.linalg.inv(covariance),
            boxed=boxed,
            psd_tolerance=float(psd_tolerance),
            maximum_cuts=int(maximum_cuts),
            frontier_regularization=float(frontier_regularization),
        )

    def solve(
        self,
        lower_moments: Mapping[MomentKey, float],
        *,
        seed_eigenvectors: ComplexArray | None = None,
    ) -> CuttingPlaneGuardResult:
        """Add violated Rayleigh-quotient cuts until the full Gram is PSD."""

        missing = set(self.extension.lower_keys).difference(lower_moments)
        if missing:
            raise ValueError(
                f"cutting-plane solve is missing {len(missing)} lower moments"
            )
        lower_values = np.asarray(
            [float(lower_moments[key]) for key in self.extension.lower_keys],
            dtype=float,
        )
        unscaled_base = self.extension._constant + np.tensordot(
            lower_values,
            self.extension.lower_coefficients,
            axes=(0, 0),
        )
        scaled_base = self.extension.scaled_matrix(unscaled_base)
        scaled_coefficients = (
            self.extension.scaled_frontier_coefficients
            * self.extension.frontier_scales[:, None, None]
        )
        count = len(self.extension.frontier_keys)
        target_count = self.scaled_readout.shape[0]
        target_whitener = np.linalg.cholesky(self.target_metric).T
        whitened_readout = target_whitener @ self.scaled_readout
        cuts: list[ComplexArray] = []
        if seed_eigenvectors is not None:
            seeds = np.asarray(seed_eigenvectors, dtype=complex)
            if seeds.ndim == 1:
                seeds = seeds[:, None]
            if seeds.shape[0] != self.extension.dimension:
                raise ValueError("seed eigenvectors have the wrong dimension")
            for column in range(seeds.shape[1]):
                vector = seeds[:, column]
                norm = float(np.linalg.norm(vector))
                if norm > 0.0:
                    cuts.append(np.asarray(vector / norm, dtype=complex))
        total_iterations = 0
        last_result: GuardConicResult | None = None
        last_lower_bound = float("-inf")
        while len(cuts) <= self.maximum_cuts:
            variable_count = count + target_count
            blocks: list[sparse.csc_matrix] = []
            rhs: list[FloatArray] = []
            cones: list[object] = []
            if cuts:
                cut_coefficients = np.asarray(
                    [
                        [
                            np.vdot(vector, coefficient @ vector).real
                            for coefficient in scaled_coefficients
                        ]
                        for vector in cuts
                    ],
                    dtype=float,
                )
                cut_base = np.asarray(
                    [
                        np.vdot(vector, scaled_base @ vector).real
                        for vector in cuts
                    ],
                    dtype=float,
                )
                blocks.append(
                    sparse.hstack(
                        (
                            -sparse.csc_matrix(cut_coefficients),
                            sparse.csc_matrix((len(cuts), target_count)),
                        ),
                        format="csc",
                    )
                )
                rhs.append(cut_base)
                cones.append(clarabel.NonnegativeConeT(len(cuts)))
            if self.boxed:
                box = sparse.vstack(
                    (
                        sparse.eye(count, format="csc"),
                        -sparse.eye(count, format="csc"),
                    ),
                    format="csc",
                )
                blocks.append(
                    sparse.hstack(
                        (
                            box,
                            sparse.csc_matrix((2 * count, target_count)),
                        ),
                        format="csc",
                    )
                )
                rhs.append(np.ones(2 * count))
                cones.append(clarabel.NonnegativeConeT(2 * count))
            image_equality = sparse.hstack(
                (
                    -sparse.csc_matrix(whitened_readout),
                    sparse.eye(target_count, format="csc"),
                ),
                format="csc",
            )
            blocks.append(image_equality)
            rhs.append(np.zeros(target_count))
            cones.append(clarabel.ZeroConeT(target_count))
            quadratic = sparse.block_diag(
                (
                    self.frontier_regularization
                    * sparse.eye(count, format="csc"),
                    sparse.eye(target_count, format="csc"),
                ),
                format="csc",
            )
            linear = np.zeros(variable_count)
            cone_matrix = sparse.vstack(blocks, format="csc")
            cone_rhs = np.concatenate(rhs)
            settings = clarabel.DefaultSettings()
            settings.verbose = False
            settings.max_threads = self.extension.settings.clarabel_max_threads
            settings.max_iter = 200
            settings.tol_gap_abs = 1e-9
            settings.tol_gap_rel = 1e-9
            settings.tol_feas = 1e-9
            settings.tol_infeas_abs = 1e-9
            settings.tol_infeas_rel = 1e-9
            settings.equilibrate_enable = True
            settings.iterative_refinement_enable = True
            settings.iterative_refinement_max_iter = 10
            solution = clarabel.DefaultSolver(
                quadratic,
                linear,
                cone_matrix,
                cone_rhs,
                cones,
                settings,
            ).solve()
            status = str(solution.status)
            total_iterations += int(solution.iterations)
            if solution.x is None:
                empty = _empty_guard_result(
                    self.extension,
                    lower_moments,
                    status=f"cutting_plane:{status}",
                    boxed=self.boxed,
                    conditioned=False,
                    independent_feasibility=float("inf"),
                )
                return CuttingPlaneGuardResult(
                    guard=empty,
                    cut_eigenvectors=np.column_stack(cuts)
                    if cuts
                    else np.empty((self.extension.dimension, 0), dtype=complex),
                    cut_count=len(cuts),
                    quadratic_lower_bound=float("-inf"),
                )
            variables = np.asarray(solution.x, dtype=float)
            values = variables[:count]
            whitened_target = variables[count:]
            target = np.linalg.solve(target_whitener, whitened_target)
            frontier_values = values * self.extension.frontier_scales
            frontier = MappingProxyType(
                {
                    key: float(value)
                    for key, value in zip(
                        self.extension.frontier_keys,
                        frontier_values,
                        strict=True,
                    )
                }
            )
            matrix = self.extension.matrix(lower_moments, frontier)
            scaled = self.extension.scaled_matrix(matrix)
            eigenvalues, eigenvectors = np.linalg.eigh(scaled)
            minimum = float(eigenvalues[0])
            image_error = float(
                np.linalg.norm(
                    target - self.scaled_readout @ values, ord=np.inf
                )
            )
            box_error = (
                0.0
                if not self.boxed
                else max(0.0, float(np.max(np.abs(values))) - 1.0)
            )
            diagnostics = _independent_kkt_diagnostics(
                solution,
                variables,
                quadratic,
                linear,
                cone_matrix,
                cone_rhs,
                direct_feasibility=max(
                    max(0.0, -minimum), image_error, box_error
                ),
            )
            strict_success = bool(
                status == "Solved"
                and max(
                    diagnostics.feasibility,
                    diagnostics.stationarity,
                    diagnostics.complementarity,
                    diagnostics.relative_gap,
                )
                <= 1e-8
                and minimum >= -self.psd_tolerance
            )
            success, certified, provisional, acceptance = _guard_acceptance(
                status,
                strict_success=strict_success,
                diagnostics=diagnostics,
            )
            if minimum < -self.psd_tolerance:
                success = False
                certified = False
                provisional = False
                acceptance = "violated_psd_cut_added"
            last_lower_bound = float(solution.obj_val)
            last_result = GuardConicResult(
                success=success,
                status=f"cutting_plane:{status}",
                boxed=self.boxed,
                conditioned=False,
                standardized_values=values,
                frontier_moments=frontier,
                moment_matrix=matrix,
                minimum_scaled_eigenvalue=minimum,
                primal_residual=float(solution.r_prim),
                dual_residual=float(solution.r_dual),
                objective=float(0.5 * target @ self.target_metric @ target),
                iterations=total_iterations,
                certified=certified,
                provisional=provisional,
                acceptance=acceptance,
                independent_feasibility_residual=diagnostics.feasibility,
                stationarity_residual=diagnostics.stationarity,
                complementarity_residual=diagnostics.complementarity,
                relative_duality_gap=diagnostics.relative_gap,
            )
            if success:
                return CuttingPlaneGuardResult(
                    guard=last_result,
                    cut_eigenvectors=np.column_stack(cuts)
                    if cuts
                    else np.empty((self.extension.dimension, 0), dtype=complex),
                    cut_count=len(cuts),
                    quadratic_lower_bound=last_lower_bound,
                )
            if minimum >= -self.psd_tolerance:
                return CuttingPlaneGuardResult(
                    guard=last_result,
                    cut_eigenvectors=np.column_stack(cuts)
                    if cuts
                    else np.empty((self.extension.dimension, 0), dtype=complex),
                    cut_count=len(cuts),
                    quadratic_lower_bound=last_lower_bound,
                )
            new_cuts = 0
            for index in np.flatnonzero(
                eigenvalues < -self.psd_tolerance
            ):
                candidate = np.asarray(eigenvectors[:, index], dtype=complex)
                if any(
                    abs(np.vdot(existing, candidate)) >= 1.0 - 1e-10
                    for existing in cuts
                ):
                    continue
                if len(cuts) >= self.maximum_cuts:
                    break
                cuts.append(candidate)
                new_cuts += 1
            if new_cuts == 0:
                return CuttingPlaneGuardResult(
                    guard=last_result,
                    cut_eigenvectors=np.column_stack(cuts)
                    if cuts
                    else np.empty((self.extension.dimension, 0), dtype=complex),
                    cut_count=len(cuts),
                    quadratic_lower_bound=last_lower_bound,
                )
        if last_result is None:
            last_result = _empty_guard_result(
                self.extension,
                lower_moments,
                status="cutting_plane:no_iteration",
                boxed=self.boxed,
                conditioned=False,
                independent_feasibility=float("inf"),
            )
        return CuttingPlaneGuardResult(
            guard=last_result,
            cut_eigenvectors=np.column_stack(cuts)
            if cuts
            else np.empty((self.extension.dimension, 0), dtype=complex),
            cut_count=len(cuts),
            quadratic_lower_bound=last_lower_bound,
        )


def select_projective_guard(
    extension: SymmetryReducedPositiveExtension,
    lower_moments: Mapping[MomentKey, float],
    readout: InvariantTargetReadout,
    *,
    boxed: bool = True,
) -> ProjectiveGuardSelection:
    """Select the minimum invariant target image and then its unique lift."""

    if readout.frontier_keys != extension.frontier_keys:
        raise ValueError("target readout belongs to another extension")
    scaled_readout = readout.matrix * extension.frontier_scales[None, :]
    covariance = scaled_readout @ scaled_readout.T
    target_metric = np.linalg.inv(covariance)
    first = _solve_explicit_target_qp(
        extension,
        lower_moments,
        scaled_readout,
        target_metric,
        boxed=boxed,
    )
    if not first.success:
        return ProjectiveGuardSelection(
            target_image=np.full(readout.rank, np.nan),
            target_metric=target_metric,
            target_stage=first,
            lift_stage=first,
        )
    target = scaled_readout @ first.standardized_values
    lift = _solve_guard_qp(
        extension,
        lower_moments,
        np.eye(len(extension.frontier_keys)),
        np.zeros(len(extension.frontier_keys)),
        boxed=boxed,
        equality_matrix=scaled_readout,
        equality_target=target,
    )
    return ProjectiveGuardSelection(
        target_image=target,
        target_metric=target_metric,
        target_stage=first,
        lift_stage=lift,
    )


def _outer_current_image_matrix(
    current: SymmetryReducedPositiveExtension,
    outer: SymmetryReducedPositiveExtension,
    readout: InvariantTargetReadout,
) -> FloatArray:
    """Embed the current invariant readout in a literal outer frontier."""

    outer_index = {key: index for index, key in enumerate(outer.frontier_keys)}
    old_index = {key: index for index, key in enumerate(current.frontier_keys)}
    missing = set(current.frontier_keys).difference(outer_index)
    if missing:
        raise RuntimeError(
            f"outer shell lost {len(missing)} literal current frontier keys"
        )
    result = np.zeros((readout.rank, len(outer.frontier_keys)))
    for key, old_column in old_index.items():
        outer_column = outer_index[key]
        if not np.isclose(
            current.frontier_scales[old_column],
            outer.frontier_scales[outer_column],
            atol=0.0,
            rtol=0.0,
        ):
            raise RuntimeError("a literal old moment changed its frozen scale")
        result[:, outer_column] = (
            readout.matrix[:, old_column]
            * outer.frontier_scales[outer_column]
        )
    return result


def select_outer_feasible_projective_guard(
    current: SymmetryReducedPositiveExtension,
    lower_moments: Mapping[MomentKey, float],
    readout: InvariantTargetReadout | None = None,
    *,
    boxed: bool = True,
    canonicalize: bool = True,
    accept_first_provisional: bool = False,
) -> OuterFeasibleProjectiveGuardSelection:
    """Select the invariant image only from witnesses feasible on one shell.

    The older two-stage selector minimized the image on the current cone and
    only afterward asked whether that image extended.  This routine reverses
    that failure mode: both image selection and the minimum-norm lift occur on
    the same outer cone, and the current values are a literal restriction of
    the resulting feasible witness.
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
    covariance = (
        target_readout.matrix
        * current.frontier_scales[None, :]
    )
    covariance = covariance @ covariance.T
    target_metric = np.linalg.inv(covariance)
    target_stage = _solve_explicit_target_qp(
        outer,
        lower_moments,
        scaled_outer_image,
        target_metric,
        boxed=boxed,
        accept_first_provisional=accept_first_provisional,
    )
    current_count = len(current.frontier_keys)
    empty_values = np.full(current_count, np.nan)
    empty_matrix = np.full((current.dimension, current.dimension), np.nan + 0j)
    if not target_stage.success:
        return OuterFeasibleProjectiveGuardSelection(
            target_image=np.full(target_readout.rank, np.nan),
            target_metric=target_metric,
            outer_extension=outer,
            outer_target_stage=target_stage,
            outer_lift_stage=target_stage,
            current_standardized_values=empty_values,
            current_frontier_moments=MappingProxyType({}),
            current_moment_matrix=empty_matrix,
            current_minimum_scaled_eigenvalue=float("nan"),
            witness_source="none",
        )
    target = scaled_outer_image @ target_stage.standardized_values
    outer_count = len(outer.frontier_keys)
    lift_stage = target_stage
    witness_stage = target_stage
    witness_source = "outer_target_stage"
    if canonicalize:
        lift_stage = _solve_guard_qp(
            outer,
            lower_moments,
            np.eye(outer_count),
            np.zeros(outer_count),
            boxed=boxed,
            equality_matrix=scaled_outer_image,
            equality_target=target,
        )
        if lift_stage.success:
            witness_stage = lift_stage
            witness_source = "minimum_norm_outer_lift"
    outer_index = {key: index for index, key in enumerate(outer.frontier_keys)}
    current_values = np.asarray(
        [
            witness_stage.standardized_values[outer_index[key]]
            for key in current.frontier_keys
        ],
        dtype=float,
    )
    frontier_values = current_values * current.frontier_scales
    current_frontier = MappingProxyType(
        {
            key: float(value)
            for key, value in zip(
                current.frontier_keys, frontier_values, strict=True
            )
        }
    )
    current_matrix = current.matrix(lower_moments, current_frontier)
    current_minimum = float(
        np.linalg.eigvalsh(current.scaled_matrix(current_matrix))[0]
    )
    current_image = (
        target_readout.matrix * current.frontier_scales[None, :]
    ) @ current_values
    if not np.allclose(current_image, target, atol=1e-7, rtol=0.0):
        raise RuntimeError("outer witness changed its current invariant image")
    return OuterFeasibleProjectiveGuardSelection(
        target_image=target,
        target_metric=target_metric,
        outer_extension=outer,
        outer_target_stage=target_stage,
        outer_lift_stage=lift_stage,
        current_standardized_values=current_values,
        current_frontier_moments=current_frontier,
        current_moment_matrix=current_matrix,
        current_minimum_scaled_eigenvalue=current_minimum,
        witness_source=witness_source,
    )


@dataclass(frozen=True)
class ProjectivePreparationAudit:
    """Four-way current/outer feasibility evidence at one fixed state."""

    current_selection: ProjectiveGuardSelection
    outer_extension: SymmetryReducedPositiveExtension
    boxed_conditioned: GuardConicResult
    unboxed_conditioned: GuardConicResult
    boxed_reopened: GuardConicResult
    unboxed_reopened: GuardConicResult
    outer_current_image_matrix: FloatArray
    classification: str


def projective_guard_outer_extension(
    current: SymmetryReducedPositiveExtension,
) -> SymmetryReducedPositiveExtension:
    """Build one joint outer shell without reclassifying current terminals."""

    candidates = tuple(current.rhs_frontier_keys)
    descendants = tuple(
        sorted(
            {
                generated
                for candidate in candidates
                for _, hamiltonian_word in _HAMILTONIAN_OPERATOR_BASIS
                for generated in _commutator(hamiltonian_word, candidate)
                if generated.degree > 0
            },
            key=_moment_sort_key,
        )
    )
    return SymmetryReducedPositiveExtension(
        current.settings,
        active_keys=current.active_keys,
        additional_halfword_keys=prefix_union(candidates, descendants),
    )


def audit_projective_preparation(
    current: SymmetryReducedPositiveExtension,
    lower_moments: Mapping[MomentKey, float],
) -> ProjectivePreparationAudit:
    """Run boxed/unboxed conditioned and reopened one-shell diagnostics."""

    readout = compile_invariant_target_readout(current)
    current_selection = select_projective_guard(
        current, lower_moments, readout, boxed=True
    )
    outer = projective_guard_outer_extension(current)
    scaled_outer_image = _outer_current_image_matrix(current, outer, readout)
    target = current_selection.target_image
    outer_count = len(outer.frontier_keys)
    identity = np.eye(outer_count)
    zero = np.zeros(outer_count)
    boxed_conditioned = _solve_guard_qp(
        outer,
        lower_moments,
        identity,
        zero,
        boxed=True,
        equality_matrix=scaled_outer_image,
        equality_target=target,
    )
    unboxed_conditioned = _solve_guard_qp(
        outer,
        lower_moments,
        identity,
        zero,
        boxed=False,
        equality_matrix=scaled_outer_image,
        equality_target=target,
    )
    boxed_reopened = _solve_guard_qp(
        outer, lower_moments, identity, zero, boxed=True
    )
    unboxed_reopened = _solve_guard_qp(
        outer, lower_moments, identity, zero, boxed=False
    )
    if boxed_conditioned.success:
        classification = "boxed_conditioned_feasible"
    elif unboxed_conditioned.success:
        classification = "nested_moment_envelope_failure"
    elif boxed_reopened.success:
        classification = "nonextendible_target_image"
    elif unboxed_reopened.success:
        classification = "moment_envelope_failure"
    else:
        classification = "guard_extension_failure_or_numerically_unresolved"
    return ProjectivePreparationAudit(
        current_selection=current_selection,
        outer_extension=outer,
        boxed_conditioned=boxed_conditioned,
        unboxed_conditioned=unboxed_conditioned,
        boxed_reopened=boxed_reopened,
        unboxed_reopened=unboxed_reopened,
        outer_current_image_matrix=scaled_outer_image,
        classification=classification,
    )


__all__ = [
    "EntranceSourceAudit",
    "FrozenOuterFaceSelector",
    "CuttingPlaneGuardResult",
    "GuardConicResult",
    "InvariantTargetReadout",
    "OuterFeasibleProjectiveGuardSelection",
    "ProjectiveGuardCuttingPlaneSelector",
    "ProjectiveGuardSelection",
    "ProjectivePreparationAudit",
    "audit_projective_preparation",
    "canonical_psd_center_cross",
    "compile_entrance_source_audit",
    "compile_invariant_target_readout",
    "center_core_null_directions",
    "prefix_restriction",
    "prefix_union",
    "projective_guard_outer_extension",
    "relative_core_moment_matrix",
    "relative_core_restriction",
    "relative_hermitian_core_restriction",
    "retained_prefix_restriction",
    "select_projective_guard",
    "select_outer_feasible_projective_guard",
    "unified_glued_moment_matrix",
    "unified_guard_dimension",
    "unified_core_moment_matrix",
    "unified_to_relative_restriction",
]
