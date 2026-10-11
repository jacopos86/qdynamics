"""Explicit polynomial form of the committed 31-coordinate archive right-hand side.

The committed archive right-hand side :func:`closed_scalar_rhs` (Eqs.
(14a)--(14e) of Riva, Simoni and Ping in the 31 closed scalar coordinates of
``CLOSED_SCALAR_STATE_NAMES``) is a polynomial of degree two in the
coordinates whose only time dependence is the declared Gaussian drive
``f(t) = DimerParameters.drive_difference(t)``:

``dx/dt = F_0 + (A_1 + f(t) D) x + A_2 (x (x) x)``,

with the cubic slot ``A_3`` kept as a generic interface and measured to be
zero for this target (the Eq. (14d) normal particle and hole sources combine
into ``-i N [g^T, rho]``).  ``F_0`` is nonzero only because the closed chart
is affine (``rho = I/2 + ...``); in the draft's real-doubled vectorization the
constant term vanishes.

This module extracts every coefficient exactly, by polynomial interpolation of
the committed right-hand side on integer points (one evaluation per monomial
of degree at most three), and builds the order-``K`` Carleman truncation of
that polynomial.  It does not edit or replace the committed right-hand side;
it is a reporting and planning seam for the quantum-integration question.

Conventions.  ``quadratic[m, i, j]`` is symmetric in ``(i, j)`` and
``cubic[m, i, j, k]`` is symmetric in ``(i, j, k)``, so the ``m``-th
component is ``sum_ij quadratic[m,i,j] x_i x_j`` and
``sum_ijk cubic[m,i,j,k] x_i x_j x_k``.  A Carleman block of order ``k`` is the
dense tensor ``x^{(x) k}`` of shape ``(n,) * k``.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, field, replace
from itertools import combinations, combinations_with_replacement
from math import comb, factorial
from typing import Callable

import numpy as np

from .hubbard_dimer import DimerParameters, FloatArray
from .matrix_reference import (
    CLOSED_SCALAR_STATE_NAMES,
    closed_scalar_rhs,
    closed_scalar_to_matrix_state,
    matrix_total_energy,
)

CLOSED_SCALAR_DIMENSION = len(CLOSED_SCALAR_STATE_NAMES)

# The record's five blocks of the 31 closed scalar coordinates.
CLOSED_SCALAR_BLOCKS: "OrderedDict[str, slice]" = OrderedDict(
    (
        ("rho", slice(0, 3)),
        ("B", slice(3, 7)),
        ("N", slice(7, 11)),
        ("A", slice(11, 17)),
        ("C", slice(17, 31)),
    )
)

PolynomialMap = Callable[[FloatArray], FloatArray]


def _block_of_index(index: int) -> str:
    for name, block in CLOSED_SCALAR_BLOCKS.items():
        if block.start <= index < block.stop:
            return name
    raise IndexError(index)


@dataclass(frozen=True)
class CubicPolynomialCoefficients:
    """Coefficients of a polynomial map of degree at most three.

    ``evaluations`` is the number of calls made to the interpolated map.
    """

    constant: FloatArray
    linear: FloatArray
    quadratic: FloatArray
    cubic: FloatArray
    evaluations: int

    def evaluate(self, state: FloatArray) -> FloatArray:
        x = np.asarray(state, dtype=float)
        return (
            self.constant
            + self.linear @ x
            + np.einsum("mij,i,j->m", self.quadratic, x, x, optimize=True)
            + np.einsum(
                "mijk,i,j,k->m", self.cubic, x, x, x, optimize=True
            )
        )


def extract_cubic_polynomial(
    function: PolynomialMap,
    dimension: int,
) -> CubicPolynomialCoefficients:
    """Interpolate a polynomial map of degree at most three exactly.

    The map is evaluated at ``0``, ``s e_i`` for ``s in (-1, 1, 2)``,
    ``e_i + e_j``, ``-e_i + e_j``, ``e_i - e_j`` and ``e_i + e_j + e_k``.
    These ``1 + 3n + 3 C(n,2) + C(n,3)`` points equal the number of monomials
    of degree at most three, and every coefficient is a fixed integer
    combination of them, so the result is exact up to floating-point rounding
    when ``function`` is a cubic polynomial.  Whether it is cubic is a separate
    check: reconstruct at fresh random points.
    """

    n = int(dimension)
    identity = np.eye(n)
    evaluations = 0

    def call(point: FloatArray) -> FloatArray:
        nonlocal evaluations
        evaluations += 1
        return np.asarray(function(point), dtype=float)

    constant = call(np.zeros(n))
    m = constant.size
    plus = np.stack([call(identity[i]) for i in range(n)], axis=1)
    minus = np.stack([call(-identity[i]) for i in range(n)], axis=1)
    double = np.stack([call(2.0 * identity[i]) for i in range(n)], axis=1)
    base = constant[:, None]
    diagonal_quadratic = 0.5 * (plus + minus) - base
    odd = 0.5 * (plus - minus)
    diagonal_cubic = (
        double - base - 4.0 * diagonal_quadratic - 2.0 * odd
    ) / 6.0
    linear = odd - diagonal_cubic

    def single(i: int, s: float) -> FloatArray:
        return (
            s * linear[:, i]
            + s * s * diagonal_quadratic[:, i]
            + s * s * s * diagonal_cubic[:, i]
        )

    quadratic = np.zeros((m, n, n))
    cubic = np.zeros((m, n, n, n))
    for i in range(n):
        quadratic[:, i, i] = diagonal_quadratic[:, i]
        cubic[:, i, i, i] = diagonal_cubic[:, i]

    pair_value: dict[tuple[int, int], FloatArray] = {}
    for i, j in combinations(range(n), 2):
        g_pp = call(identity[i] + identity[j]) - constant - single(i, 1.0) - single(j, 1.0)
        g_mp = call(-identity[i] + identity[j]) - constant - single(i, -1.0) - single(j, 1.0)
        g_pm = call(identity[i] - identity[j]) - constant - single(i, 1.0) - single(j, -1.0)
        iij = 0.5 * (g_pp + g_mp)  # coefficient of x_i^2 x_j
        ijj = 0.5 * (g_pp + g_pm)  # coefficient of x_i x_j^2
        ij = g_pp - iij - ijj  # coefficient of x_i x_j
        pair_value[(i, j)] = g_pp
        quadratic[:, i, j] = quadratic[:, j, i] = 0.5 * ij
        for a, b, c in ((i, i, j), (i, j, i), (j, i, i)):
            cubic[:, a, b, c] = iij / 3.0
        for a, b, c in ((i, j, j), (j, i, j), (j, j, i)):
            cubic[:, a, b, c] = ijj / 3.0

    for i, j, k in combinations(range(n), 3):
        value = (
            call(identity[i] + identity[j] + identity[k])
            - constant
            - single(i, 1.0)
            - single(j, 1.0)
            - single(k, 1.0)
            - pair_value[(i, j)]
            - pair_value[(i, k)]
            - pair_value[(j, k)]
        )
        for a, b, c in (
            (i, j, k),
            (i, k, j),
            (j, i, k),
            (j, k, i),
            (k, i, j),
            (k, j, i),
        ):
            cubic[:, a, b, c] = value / 6.0

    return CubicPolynomialCoefficients(
        constant=constant,
        linear=linear,
        quadratic=quadratic,
        cubic=cubic,
        evaluations=evaluations,
    )


@dataclass(frozen=True)
class ClosedPolynomialForm:
    """``F(t, x) = F_0 + (A_1 + f(t) D) x + A_2 x(x)x + A_3 x(x)x(x)x``.

    ``linear_free`` is ``A_1`` and ``linear_drive`` is ``D``; ``f`` is the
    declared drive ``parameters.drive_difference``.  ``drive_audit`` records,
    at each probe time ``t`` with ``f(t) != 0``, the largest absolute
    coefficient of the drive part ``F(t, .) - F(0, .)`` in degrees zero, two
    and three (zero when the drive enters only linearly) and the largest
    absolute disagreement of ``D`` re-extracted at that time.
    """

    parameters: DimerParameters
    constant: FloatArray
    linear_free: FloatArray
    linear_drive: FloatArray
    quadratic: FloatArray
    cubic: FloatArray
    drive_audit: dict = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        return int(self.constant.size)

    def drive(self, time: float) -> float:
        return float(self.parameters.drive_difference(time))

    def linear(self, time: float) -> FloatArray:
        return self.linear_free + self.drive(time) * self.linear_drive

    def evaluate(self, time: float, state: FloatArray) -> FloatArray:
        x = np.asarray(state, dtype=float)
        quadratic_matrix = self.quadratic.reshape(self.dimension, -1)
        cubic_matrix = self.cubic.reshape(self.dimension, -1)
        xx = np.outer(x, x).reshape(-1)
        xxx = np.multiply.outer(xx, x).reshape(-1)
        return (
            self.constant
            + self.linear(time) @ x
            + quadratic_matrix @ xx
            + cubic_matrix @ xxx
        )

    def jacobian(self, time: float, state: FloatArray) -> FloatArray:
        x = np.asarray(state, dtype=float)
        return (
            self.linear(time)
            + 2.0 * np.einsum("mij,j->mi", self.quadratic, x)
            + 3.0 * np.einsum("mijk,j,k->mi", self.cubic, x, x)
        )

    def quadratic_matrix(self) -> FloatArray:
        """``A_2`` as the ``n x n^2`` matrix acting on ``x (x) x``."""

        return self.quadratic.reshape(self.dimension, -1)

    def cubic_matrix(self) -> FloatArray:
        """``A_3`` as the ``n x n^3`` matrix acting on ``x (x) x (x) x``."""

        return self.cubic.reshape(self.dimension, -1)


def extract_closed_polynomial_form(
    parameters: DimerParameters,
    *,
    drive_probe_times: tuple[float, ...] = (0.7, 1.9),
    rhs: Callable[[float, FloatArray, DimerParameters], FloatArray] = closed_scalar_rhs,
) -> ClosedPolynomialForm:
    """Extract the drive-free polynomial and the drive part of ``rhs``.

    The drive-free part is the polynomial at ``drive_amplitude = 0``.  The
    drive part is extracted in full at each probe time; its linear part
    divided by ``f(t)`` is ``D`` (taken from the first probe) and its other
    degrees are recorded in ``drive_audit``.
    """

    n = CLOSED_SCALAR_DIMENSION
    undriven = replace(parameters, drive_amplitude=0.0)
    free = extract_cubic_polynomial(lambda x: rhs(0.0, x, undriven), n)
    linear_drive = None
    audit: dict[str, dict[str, float]] = {}
    for time in drive_probe_times:
        drive_value = float(parameters.drive_difference(time))
        if drive_value == 0.0:
            raise ValueError(f"drive vanishes at probe time {time}")
        driven = extract_cubic_polynomial(
            lambda x, time=time: rhs(time, x, parameters), n
        )
        candidate = (driven.linear - free.linear) / drive_value
        if linear_drive is None:
            linear_drive = candidate
        audit[f"{time:g}"] = {
            "drive_value": drive_value,
            "constant_drive_max_abs": float(
                np.max(np.abs(driven.constant - free.constant))
            ),
            "quadratic_drive_max_abs": float(
                np.max(np.abs(driven.quadratic - free.quadratic))
            ),
            "cubic_drive_max_abs": float(
                np.max(np.abs(driven.cubic - free.cubic))
            ),
            "linear_drive_disagreement_max_abs": float(
                np.max(np.abs(candidate - linear_drive))
            ),
        }
    assert linear_drive is not None
    return ClosedPolynomialForm(
        parameters=parameters,
        constant=free.constant,
        linear_free=free.linear,
        linear_drive=linear_drive,
        quadratic=free.quadratic,
        cubic=free.cubic,
        drive_audit=audit,
    )


# ---------------------------------------------------------------------------
# Structure reports
# ---------------------------------------------------------------------------


def _distinct_monomial_entries(tensor: FloatArray, degree: int):
    """Yield ``(output, indices, coefficient)`` over sorted index tuples.

    The coefficient is that of the monomial ``x_i x_j ...`` in the polynomial,
    i.e. the symmetric entry times the number of distinct permutations.
    """

    m = tensor.shape[0]
    n = tensor.shape[1]
    for indices in combinations_with_replacement(range(n), degree):
        counts = Counter(indices)
        multiplicity = factorial(degree)
        for count in counts.values():
            multiplicity //= factorial(count)
        column = tensor[(slice(None),) + indices] * multiplicity
        for output in range(m):
            yield output, indices, float(column[output])


def monomial_coefficients(
    form: ClosedPolynomialForm,
    *,
    tolerance: float = 1e-10,
) -> dict[int, list[tuple[int, tuple[int, ...], float]]]:
    """Return the nonzero monomial coefficients by degree (drive-free part)."""

    result: dict[int, list[tuple[int, tuple[int, ...], float]]] = {
        0: [],
        1: [],
        2: [],
        3: [],
    }
    for output in range(form.dimension):
        if abs(form.constant[output]) > tolerance:
            result[0].append((output, (), float(form.constant[output])))
        for column in range(form.dimension):
            value = float(form.linear_free[output, column])
            if abs(value) > tolerance:
                result[1].append((output, (column,), value))
    for degree, tensor in ((2, form.quadratic), (3, form.cubic)):
        for output, indices, value in _distinct_monomial_entries(tensor, degree):
            if abs(value) > tolerance:
                result[degree].append((output, indices, value))
    return result


def rounding_gap(
    form: ClosedPolynomialForm,
    *,
    tolerance: float = 1e-10,
) -> dict[str, float]:
    """Largest coefficient magnitude at or below ``tolerance`` and smallest above."""

    values = np.concatenate(
        [
            np.abs(form.constant).ravel(),
            np.abs(form.linear_free).ravel(),
            np.abs(form.linear_drive).ravel(),
            np.abs(form.quadratic).ravel(),
            np.abs(form.cubic).ravel(),
        ]
    )
    below = values[values <= tolerance]
    above = values[values > tolerance]
    return {
        "tolerance": tolerance,
        "largest_at_or_below": float(below.max()) if below.size else 0.0,
        "smallest_above": float(above.min()) if above.size else float("nan"),
    }


def block_coupling_table(
    form: ClosedPolynomialForm,
    *,
    tolerance: float = 1e-10,
) -> dict[str, dict]:
    """Per output block: degree, and input-block combinations per degree.

    Degree-one entries list the input blocks of ``A_1`` and of ``D``
    separately.  Degree-two and degree-three entries list the sorted
    input-block pairs and triples carrying a nonzero monomial, with the count
    of nonzero monomial coefficients.
    """

    monomials = monomial_coefficients(form, tolerance=tolerance)
    table: dict[str, dict] = {}
    for name, block in CLOSED_SCALAR_BLOCKS.items():
        outputs = set(range(block.start, block.stop))
        entry: dict[str, object] = {"coordinates": block.stop - block.start}
        degree = -1
        for d in (0, 1, 2, 3):
            rows = [row for row in monomials[d] if row[0] in outputs]
            combos: dict[str, int] = {}
            for _, indices, _ in rows:
                key = "".join(
                    "(" + _block_of_index(i) + ")" for i in sorted(indices)
                ) if indices else "const"
                combos[key] = combos.get(key, 0) + 1
            entry[f"degree_{d}"] = dict(sorted(combos.items()))
            if rows:
                degree = d
        drive_inputs: dict[str, int] = {}
        for output in outputs:
            for column in range(form.dimension):
                if abs(form.linear_drive[output, column]) > tolerance:
                    key = "(" + _block_of_index(column) + ")"
                    drive_inputs[key] = drive_inputs.get(key, 0) + 1
        entry["drive_linear"] = dict(sorted(drive_inputs.items()))
        entry["degree"] = degree
        table[name] = entry
    return table


def tensor_statistics(
    form: ClosedPolynomialForm,
    *,
    tolerance: float = 1e-10,
) -> dict[str, dict[str, float]]:
    """Nonzero counts and norms for ``F_0``, ``A_1``, ``D``, ``A_2``, ``A_3``.

    ``nonzero_entries`` counts entries of the symmetric tensor (all
    permutations), ``nonzero_monomials`` counts distinct monomial
    coefficients.  ``spectral_norm`` is the largest singular value of the
    ``n x n^p`` matricization acting on ``x^{(x) p}``; for ``F_0`` it is the
    Euclidean norm.
    """

    monomials = monomial_coefficients(form, tolerance=tolerance)

    def drive_monomials() -> int:
        return int(np.count_nonzero(np.abs(form.linear_drive) > tolerance))

    stats = {
        "F_0": {
            "nonzero_entries": int(np.count_nonzero(np.abs(form.constant) > tolerance)),
            "nonzero_monomials": len(monomials[0]),
            "spectral_norm": float(np.linalg.norm(form.constant)),
            "frobenius_norm": float(np.linalg.norm(form.constant)),
        },
        "A_1": {
            "nonzero_entries": int(np.count_nonzero(np.abs(form.linear_free) > tolerance)),
            "nonzero_monomials": len(monomials[1]),
            "spectral_norm": float(np.linalg.norm(form.linear_free, 2)),
            "frobenius_norm": float(np.linalg.norm(form.linear_free)),
        },
        "D": {
            "nonzero_entries": drive_monomials(),
            "nonzero_monomials": drive_monomials(),
            "spectral_norm": float(np.linalg.norm(form.linear_drive, 2)),
            "frobenius_norm": float(np.linalg.norm(form.linear_drive)),
        },
        "A_2": {
            "nonzero_entries": int(np.count_nonzero(np.abs(form.quadratic) > tolerance)),
            "nonzero_monomials": len(monomials[2]),
            "spectral_norm": float(np.linalg.norm(form.quadratic_matrix(), 2)),
            "frobenius_norm": float(np.linalg.norm(form.quadratic)),
        },
        "A_3": {
            "nonzero_entries": int(np.count_nonzero(np.abs(form.cubic) > tolerance)),
            "nonzero_monomials": len(monomials[3]),
            "spectral_norm": float(np.linalg.norm(form.cubic_matrix(), 2)),
            "frobenius_norm": float(np.linalg.norm(form.cubic)),
        },
    }
    return stats


def linear_part_spectrum(matrix: FloatArray) -> dict[str, object]:
    """Eigenvalues, symmetric-part spectrum and eigenvector conditioning."""

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    symmetric = 0.5 * (matrix + matrix.T)
    symmetric_eigenvalues = np.linalg.eigvalsh(symmetric)
    order = np.argsort(-eigenvalues.real)
    return {
        "eigenvalues": eigenvalues[order],
        "max_real_part": float(np.max(eigenvalues.real)),
        "min_real_part": float(np.min(eigenvalues.real)),
        "max_abs_real_part": float(np.max(np.abs(eigenvalues.real))),
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
        "eigenvector_condition_number": float(np.linalg.cond(eigenvectors)),
        "rank": int(np.linalg.matrix_rank(matrix)),
        "symmetric_part_eigenvalues": symmetric_eigenvalues,
        "symmetric_part_max": float(symmetric_eigenvalues[-1]),
        "symmetric_part_min": float(symmetric_eigenvalues[0]),
        "skew_part_spectral_norm": float(
            np.linalg.norm(0.5 * (matrix - matrix.T), 2)
        ),
        "symmetric_part_spectral_norm": float(np.linalg.norm(symmetric, 2)),
    }


# ---------------------------------------------------------------------------
# Carleman truncation
# ---------------------------------------------------------------------------


def carleman_dimensions(dimension: int, order: int) -> dict[str, int]:
    """Lifted dimension at truncation order ``K`` (constant block excluded).

    ``tensor``: ``sum_{k<=K} n^k``, the Kronecker-power lift.
    ``symmetric``: ``sum_{k<=K} C(n+k-1, k)``, distinct monomials.
    """

    n = int(dimension)
    return {
        "tensor": int(sum(n**k for k in range(1, order + 1))),
        "symmetric": int(sum(comb(n + k - 1, k) for k in range(1, order + 1))),
    }


def carleman_lift(state: FloatArray, order: int) -> list[FloatArray]:
    """Return ``[x, x(x)x, ..., x^{(x)K}]`` as dense tensors."""

    x = np.asarray(state, dtype=float)
    blocks = [x.copy()]
    for _ in range(order - 1):
        blocks.append(np.multiply.outer(blocks[-1], x))
    return blocks


def _apply_at_position(
    operator: FloatArray,
    source: FloatArray,
    position: int,
    target_order: int,
    n: int,
) -> FloatArray:
    """Apply an ``n x n^p`` operator at ``position`` of a source tensor.

    The source has order ``target_order + p - 1``; the ``p`` consecutive axes
    starting at ``position`` are contracted and replaced by one output axis.
    """

    left = n**position
    right = n ** (target_order - 1 - position)
    width = source.size // (left * right)
    reshaped = source.reshape(left, width, right)
    if right == 1:
        product = reshaped[:, :, 0] @ operator.T  # (left, n)
        return product.reshape((n,) * target_order)
    product = np.matmul(operator, reshaped)  # (left, n, right)
    return product.reshape((n,) * target_order)


def carleman_block_derivative(
    form: ClosedPolynomialForm,
    time: float,
    blocks: list[FloatArray],
    order: int,
) -> list[FloatArray]:
    """Truncated Carleman velocity of blocks ``1..K`` (``blocks[k-1]``).

    Block ``k`` receives ``F_0`` from block ``k-1`` (the constant ``1`` for
    ``k = 1``), ``A_1(t)`` from block ``k``, ``A_2`` from block ``k+1`` and
    ``A_3`` from block ``k+2``, each applied at every one of the ``k``
    positions; contributions whose source order exceeds ``K`` are dropped.
    """

    n = form.dimension
    linear = form.linear(time)
    operators = (
        form.constant.reshape(n, 1),
        linear,
        form.quadratic_matrix(),
        form.cubic_matrix(),
    )
    derivative: list[FloatArray] = []
    for k in range(1, order + 1):
        total = np.zeros((n,) * k)
        for p, operator in enumerate(operators):
            source_order = k + p - 1
            if source_order > order:
                continue
            if source_order == 0:
                total = total + operator[:, 0]
                continue
            source = blocks[source_order - 1]
            for position in range(k):
                total = total + _apply_at_position(
                    operator, source, position, k, n
                )
        derivative.append(total)
    return derivative


def _tensor_sum_norm_squared(
    k: int,
    vector: FloatArray,
    state: FloatArray,
) -> float:
    """``|| sum_m x^{(m-1)} (x) v (x) x^{(k-m)} ||^2`` in closed form."""

    xx = float(state @ state)
    vv = float(vector @ vector)
    xv = float(state @ vector)
    if k == 1:
        return vv
    return k * vv * xx ** (k - 1) + k * (k - 1) * xv * xv * xx ** (k - 2)


def carleman_truncation_remainder_norms(
    form: ClosedPolynomialForm,
    time: float,
    state: FloatArray,
    order: int,
) -> dict[str, object]:
    """Norms of what the order-``K`` truncation drops at the lift of ``x``.

    The lifted velocity of block ``k`` under the closure's own field is
    ``sum_m x^{(m-1)} (x) F(t,x) (x) x^{(k-m)}``.  The truncation keeps the
    terms of degree at most ``K``; it drops ``A_3 x^{(x)3}`` from block
    ``K-1`` and ``A_2 x^{(x)2} + A_3 x^{(x)3}`` from block ``K``.  Returned
    per block: the dropped norm, the full lifted velocity norm, and their
    ratio; and the same over all blocks together.
    """

    x = np.asarray(state, dtype=float)
    xx = np.outer(x, x).reshape(-1)
    xxx = np.multiply.outer(xx, x).reshape(-1)
    quadratic_term = form.quadratic_matrix() @ xx
    cubic_term = form.cubic_matrix() @ xxx
    velocity = form.evaluate(time, x)
    dropped_sq = []
    full_sq = []
    for k in range(1, order + 1):
        if k == order:
            dropped_vector = quadratic_term + cubic_term
        elif k == order - 1:
            dropped_vector = cubic_term
        else:
            dropped_vector = np.zeros_like(x)
        dropped_sq.append(_tensor_sum_norm_squared(k, dropped_vector, x))
        full_sq.append(_tensor_sum_norm_squared(k, velocity, x))
    dropped = np.sqrt(np.asarray(dropped_sq))
    full = np.sqrt(np.asarray(full_sq))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(full > 0.0, dropped / full, np.nan)
    return {
        "dropped_by_block": dropped,
        "lifted_velocity_by_block": full,
        "relative_by_block": ratio,
        "dropped_total": float(np.sqrt(np.sum(dropped**2))),
        "lifted_velocity_total": float(np.sqrt(np.sum(full**2))),
        "relative_total": float(
            np.sqrt(np.sum(dropped**2)) / np.sqrt(np.sum(full**2))
        ),
    }


def carleman_sparse_generator(form: ClosedPolynomialForm, order: int):
    """Sparse order-``K`` lifted generator ``dX/dt = (L + f(t) M) X + b``.

    Returns ``(L, M, b)`` with ``L`` the drive-free lifted matrix, ``M`` the
    lifted drive matrix and ``b`` the constant forcing (``F_0`` in block 1),
    all in the flattened Kronecker-power ordering of :func:`flatten_blocks`.
    Block ``k`` receives ``I_{n^m} (x) Op (x) I_{n^{k-1-m}}`` from block
    ``k + p - 1`` for each position ``m`` and each operator of degree ``p``;
    sources above ``K`` are dropped (the truncation).
    """

    import scipy.sparse as sp

    n = form.dimension
    sizes = [n**k for k in range(1, order + 1)]
    operators_free = {
        0: sp.csr_matrix(form.constant.reshape(n, 1)),
        1: sp.csr_matrix(form.linear_free),
        2: sp.csr_matrix(form.quadratic_matrix()),
        3: sp.csr_matrix(form.cubic_matrix()),
    }
    operators_drive = {1: sp.csr_matrix(form.linear_drive)}

    def assemble(operators):
        rows = []
        for k in range(1, order + 1):
            row = [None] * order
            for p, op in operators.items():
                source = k + p - 1
                if source < 1 or source > order or op.nnz == 0:
                    continue
                total = None
                for m in range(k):
                    left = sp.identity(n**m, format="csr")
                    right = sp.identity(n ** (k - 1 - m), format="csr")
                    term = sp.kron(left, sp.kron(op, right, format="csr"), format="csr")
                    total = term if total is None else total + term
                row[source - 1] = total if row[source - 1] is None else row[source - 1] + total
            for j in range(order):
                if row[j] is None and j == k - 1:
                    row[j] = sp.csr_matrix((sizes[k - 1], sizes[j]))
            rows.append(row)
        return sp.bmat(rows, format="csr")

    forcing = np.zeros(sum(sizes))
    forcing[:n] = form.constant
    return assemble(operators_free), assemble(operators_drive), forcing


def flatten_blocks(blocks: list[FloatArray]) -> FloatArray:
    return np.concatenate([block.reshape(-1) for block in blocks])


def unflatten_blocks(vector: FloatArray, n: int, order: int) -> list[FloatArray]:
    blocks = []
    offset = 0
    for k in range(1, order + 1):
        size = n**k
        blocks.append(vector[offset : offset + size].reshape((n,) * k))
        offset += size
    return blocks


@dataclass(frozen=True)
class CarlemanSolution:
    """First-block readout of a truncated Carleman integration."""

    order: int
    times: FloatArray
    first_block: FloatArray
    lifted_norm: FloatArray
    steps: int
    stopped_at: float | None


def integrate_carleman_rk4(
    form: ClosedPolynomialForm,
    initial_state: FloatArray,
    order: int,
    *,
    sample_times: FloatArray,
    steps_per_sample: int = 1,
    stop_magnitude: float = 1e4,
) -> CarlemanSolution:
    """Classical RK4 on the order-``K`` truncation from the lift of ``x(0)``.

    Integration stops after the first sample whose first block has a
    coordinate above ``stop_magnitude`` in absolute value (the record's
    ``1e4`` magnitude event) or is not finite.
    """

    n = form.dimension
    times = np.asarray(sample_times, dtype=float)
    vector = flatten_blocks(carleman_lift(initial_state, order))

    def velocity(time: float, flat: FloatArray) -> FloatArray:
        return flatten_blocks(
            carleman_block_derivative(
                form, time, unflatten_blocks(flat, n, order), order
            )
        )

    first = np.full((times.size, n), np.nan)
    norms = np.full(times.size, np.nan)
    first[0] = vector[:n]
    norms[0] = float(np.linalg.norm(vector))
    steps = 0
    stopped_at = None
    for index in range(1, times.size):
        t0 = float(times[index - 1])
        h = (float(times[index]) - t0) / steps_per_sample
        for sub in range(steps_per_sample):
            t = t0 + sub * h
            k1 = velocity(t, vector)
            k2 = velocity(t + 0.5 * h, vector + 0.5 * h * k1)
            k3 = velocity(t + 0.5 * h, vector + 0.5 * h * k2)
            k4 = velocity(t + h, vector + h * k3)
            vector = vector + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            steps += 1
        first[index] = vector[:n]
        norms[index] = float(np.linalg.norm(vector))
        if (not np.all(np.isfinite(vector[:n]))) or np.max(
            np.abs(vector[:n])
        ) > stop_magnitude:
            stopped_at = float(times[index])
            break
    return CarlemanSolution(
        order=order,
        times=times,
        first_block=first,
        lifted_norm=norms,
        steps=steps,
        stopped_at=stopped_at,
    )


# ---------------------------------------------------------------------------
# Eq. (22) internal energy and the trajectory-linearized flow
# ---------------------------------------------------------------------------


def closed_internal_energy(parameters: DimerParameters) -> CubicPolynomialCoefficients:
    """Eq. (22) internal energy as a polynomial in the 31 closed coordinates.

    Interpolated exactly from :func:`matrix_total_energy`; the output has one
    component, so ``linear[0]`` is the gradient at the origin and
    ``2 * quadratic[0]`` the (constant) Hessian.
    """

    return extract_cubic_polynomial(
        lambda x: np.array(
            [matrix_total_energy(closed_scalar_to_matrix_state(x), parameters)]
        ),
        CLOSED_SCALAR_DIMENSION,
    )


def logarithmic_norm(matrix: FloatArray, metric_root: FloatArray | None = None) -> float:
    """Largest eigenvalue of the symmetric part of ``R M R^{-1}``.

    With ``metric_root = None`` this is the Euclidean logarithmic norm; with
    ``R`` such that ``G = R^T R`` it is the logarithmic norm in ``<x, G y>``.
    """

    if metric_root is not None:
        matrix = metric_root @ matrix @ np.linalg.inv(metric_root)
    return float(np.linalg.eigvalsh(0.5 * (matrix + matrix.T))[-1])


def integrate_linearized_on_samples(
    form: ClosedPolynomialForm,
    times: FloatArray,
    states: FloatArray,
    *,
    initial: FloatArray,
    source: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """RK4 for ``d(xi)/dt = J_F(t, x(t)) xi + s(t)`` on stored samples.

    ``times`` must be uniform; the RK4 step is two sample spacings so the
    midpoint stage uses the stored middle sample and no interpolation enters.
    ``initial`` is a vector (``n``) or a matrix (``n x m``, e.g. the identity
    for the propagator).  ``source`` (same length as ``times``) is added to a
    vector integration only.  Returns ``(times[::2], solution)``.
    """

    times = np.asarray(times, dtype=float)
    spacing = np.diff(times)
    if not np.allclose(spacing, spacing[0], rtol=0.0, atol=1e-9):
        raise ValueError("times must be uniform")
    h = 2.0 * float(spacing[0])
    current = np.asarray(initial, dtype=float).copy()
    if source is not None and current.ndim != 1:
        raise ValueError("a source applies to a vector integration only")

    def velocity(index: int, value: FloatArray) -> FloatArray:
        out = form.jacobian(float(times[index]), states[index]) @ value
        if source is not None:
            out = out + source[index]
        return out

    steps = (times.size - 1) // 2
    solution = np.empty((steps + 1,) + current.shape)
    solution[0] = current
    for step in range(steps):
        i0, i1, i2 = 2 * step, 2 * step + 1, 2 * step + 2
        k1 = velocity(i0, current)
        k2 = velocity(i1, current + 0.5 * h * k1)
        k3 = velocity(i1, current + 0.5 * h * k2)
        k4 = velocity(i2, current + h * k3)
        current = current + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        solution[step + 1] = current
    return times[: 2 * steps + 1 : 2], solution
