"""Cheapest physicality repair of the moment hierarchy: no conic solve.

This module is a named exploratory route for Paper V and changes no default
anywhere else.  It propagates the fixed-dictionary APCM state with the
unguarded velocity -- the archive-plus-entrance target velocity and the exact
active commutator velocity that
:meth:`FixedDictionaryProjectedAPCM.targets` already builds -- and repairs
physicality by linear algebra alone.  Nothing here calls Clarabel, CVXPY,
SLSQP, the proximal log-det completion, the tangent-cone projection, or the
stage retraction.

The construction rests on one fact this module verifies at build time: both
moment matrices the record reports are *affine* in the propagated
coordinates.

``retained joint Gram``
    :func:`uncentered_joint_moment_matrix` is affine in the 29 raw moment
    coordinates (``apcm_moment_projection._base_affine_map`` states the same
    fact for the committed path; this module rebuilds it through the public
    function).

``24-row positive extension``
    ``SymmetryReducedPositiveExtension.matrix`` is affine in its 29 lower
    moments and 62 frontier moments, and the lower moments are affine in the
    44 propagated state coordinates (29 raw plus 15 entrance).

So a repair has the shape: build the Gram, project it onto the positive
semidefinite cone (or alternate between that cone and the affine Gram set),
and map the corrected Gram back to coordinates by one least-squares solve
with a pseudo-inverse factored once per rollout.  Three named policies are
provided:

``eigenvalue_clip``
    One eigendecomposition, eigenvalues raised to a declared floor, one
    least-squares map back.

``higham_nearest_psd``
    The nearest positive semidefinite matrix in the Frobenius norm (Higham
    1988), that is the same clip at floor zero, stated as its own policy
    because the floor is the declared quantity that separates the two.

``dykstra``
    A declared number of Dykstra alternating projections between the affine
    Gram set and the positive semidefinite cone, with the Dykstra correction
    carried between iterations, then one least-squares map back.

No policy produces a certificate.  Every reported physicality number is the
measured minimum eigenvalue of the stated matrix along the trajectory, and
the honest reading of a run is those minima next to the accuracy score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from .adaptive_positive_moment import (
    ENTRANCE_RELATIVE_MOMENT_KEYS,
    uncentered_joint_moment_matrix,
)
from .apcm_moment_projection import state_lower_moments
from .apcm_positive_extension import (
    APCMExtensionResult,
    SymmetryReducedPositiveExtension,
)
from .moment_hierarchy import MomentKey
from .projected_apcm import (
    FixedDictionaryProjectedAPCM,
    ProjectedAPCMTargets,
    unpack_projected_apcm_state,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_RAW_DIMENSION = 29

CHEAP_PROJECTION_POLICIES = (
    "none",
    "eigenvalue_clip",
    "higham_nearest_psd",
    "dykstra",
)
CHEAP_PROJECTION_TARGETS = ("extension", "retained", "both")
CHEAP_PROJECTION_CADENCES = ("stage", "step")
CHEAP_PROJECTION_COORDINATES = ("state_and_frontier", "frontier_only")


@dataclass(frozen=True)
class CheapProjectionSettings:
    """Declared parameters of one cheap physicality repair.

    ``policy``
        Which repair runs; ``"none"`` propagates the unguarded velocity with
        no repair at all and is the control every other policy is read
        against.
    ``target``
        Which moment matrix is repaired: the 24-row positive extension, the
        retained joint Gram, or both at once in one least-squares solve.
    ``cadence``
        ``"stage"`` repairs after every SSPRK(3,3) stage; ``"step"`` repairs
        only the accepted state at the end of a step.
    ``eigenvalue_floor``
        The declared floor eigenvalues are raised to, in the scaled metric
        each matrix is reported in.  ``higham_nearest_psd`` ignores it and
        uses zero, which is what makes it the nearest positive semidefinite
        matrix.
    ``dykstra_iterations``
        How many alternating projections the ``dykstra`` policy runs.
    ``repair_tolerance``
        A repair is skipped, and counted as skipped, when the target
        matrix's minimum eigenvalue is already at or above
        ``eigenvalue_floor - repair_tolerance``.  Zero repairs at every
        opportunity.
    ``retained_eigenvalue_floor``
        A separate floor for the retained joint Gram block, for the targets
        that carry it.  ``None`` uses ``eigenvalue_floor`` for both blocks.
        The two floors are separate because the retained joint Gram of a
        physical state is genuinely near the boundary of its cone, while the
        lift the entrance commutators read may be pushed interior at no
        cost to the propagated coordinates.
    ``repair_coordinates``
        Which coordinates the least-squares map back is allowed to move.
        ``"state_and_frontier"`` moves both; ``"frontier_only"`` holds the
        44 propagated coordinates fixed and repairs the lift alone, which is
        the cheapest surrogate for re-selecting the completion.
    ``state_motion_scale`` and ``frontier_motion_scale``
        Relative freedom of the 44 state coordinates and the 62 standardized
        frontier coordinates in the least-squares map back.  The repair takes
        the minimum-norm solution in the rescaled parameters, so a larger
        scale lets that group absorb more of the correction.  ``1.0`` and
        ``1.0`` is plain minimum-norm least squares.  These are inert
        whenever the null space of the affine Gram map lies entirely inside
        one group, which is the case for this dictionary: the 15 directions
        the 24-row Gram cannot see are all propagated-state directions.
    """

    policy: str = "eigenvalue_clip"
    target: str = "extension"
    cadence: str = "stage"
    eigenvalue_floor: float = 0.0
    retained_eigenvalue_floor: float | None = None
    repair_coordinates: str = "state_and_frontier"
    dykstra_iterations: int = 5
    repair_tolerance: float = 0.0
    state_motion_scale: float = 1.0
    frontier_motion_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.policy not in CHEAP_PROJECTION_POLICIES:
            raise ValueError(
                "policy must be one of " + ", ".join(CHEAP_PROJECTION_POLICIES)
            )
        if self.target not in CHEAP_PROJECTION_TARGETS:
            raise ValueError(
                "target must be one of " + ", ".join(CHEAP_PROJECTION_TARGETS)
            )
        if self.cadence not in CHEAP_PROJECTION_CADENCES:
            raise ValueError(
                "cadence must be one of " + ", ".join(CHEAP_PROJECTION_CADENCES)
            )
        if self.repair_coordinates not in CHEAP_PROJECTION_COORDINATES:
            raise ValueError(
                "repair_coordinates must be one of "
                + ", ".join(CHEAP_PROJECTION_COORDINATES)
            )
        if (
            self.repair_coordinates == "frontier_only"
            and self.target == "retained"
        ):
            raise ValueError(
                "the retained joint Gram does not depend on the frontier "
                "coordinates, so frontier_only cannot repair it"
            )
        if self.eigenvalue_floor < 0.0:
            raise ValueError("eigenvalue_floor must be nonnegative")
        if (
            self.retained_eigenvalue_floor is not None
            and self.retained_eigenvalue_floor < 0.0
        ):
            raise ValueError(
                "retained_eigenvalue_floor must be nonnegative"
            )
        if self.dykstra_iterations < 1:
            raise ValueError("dykstra_iterations must be positive")
        if self.repair_tolerance < 0.0:
            raise ValueError("repair_tolerance must be nonnegative")
        if self.state_motion_scale <= 0.0:
            raise ValueError("state_motion_scale must be positive")
        if self.frontier_motion_scale <= 0.0:
            raise ValueError("frontier_motion_scale must be positive")

    @property
    def applied_floor(self) -> float:
        """The extension-block floor the selected policy actually uses."""

        if self.policy == "higham_nearest_psd":
            return 0.0
        return float(self.eigenvalue_floor)

    @property
    def applied_retained_floor(self) -> float:
        """The retained-block floor the selected policy actually uses."""

        if self.policy == "higham_nearest_psd":
            return 0.0
        if self.retained_eigenvalue_floor is None:
            return float(self.eigenvalue_floor)
        return float(self.retained_eigenvalue_floor)

    def block_floors(self, blocks: tuple[str, ...]) -> tuple[float, ...]:
        """The declared floor of each block of the selected target."""

        return tuple(
            self.applied_retained_floor
            if name == "retained"
            else self.applied_floor
            for name in blocks
        )


@dataclass(frozen=True)
class CheapProjectionRepair:
    """One repair, its cost, and what it moved."""

    state: FloatArray
    frontier_standardized: FloatArray
    applied: bool
    iterations: int
    extension_minimum_before: float
    extension_minimum_after: float
    retained_minimum_before: float
    retained_minimum_after: float
    state_displacement: float
    frontier_displacement: float
    target_residual: float
    seconds: float


def _hermitian_features(matrix: ComplexArray) -> FloatArray:
    """Real coordinates of one complex Hermitian matrix, Frobenius-isometric."""

    value = np.asarray(matrix, dtype=complex)
    rows = value.shape[0]
    upper = np.triu_indices(rows, 1)
    return np.concatenate(
        (
            value.diagonal().real,
            np.sqrt(2.0) * value[upper].real,
            np.sqrt(2.0) * value[upper].imag,
        )
    )


def _hermitian_from_features(values: FloatArray, rows: int) -> ComplexArray:
    """Invert :func:`_hermitian_features`."""

    upper = np.triu_indices(rows, 1)
    count = upper[0].size
    matrix = np.zeros((rows, rows), dtype=complex)
    matrix[np.diag_indices(rows)] = values[:rows]
    offdiagonal = (
        values[rows : rows + count]
        + 1j * values[rows + count : rows + 2 * count]
    ) / np.sqrt(2.0)
    matrix[upper] = offdiagonal
    return matrix + matrix.conjugate().T - np.diag(matrix.diagonal().real)


def _clip_to_floor(matrix: ComplexArray, floor: float) -> ComplexArray:
    """Raise every eigenvalue of one Hermitian matrix to ``floor``."""

    hermitian = 0.5 * (matrix + matrix.conjugate().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    if eigenvalues[0] >= floor:
        return hermitian
    clipped = np.maximum(eigenvalues, floor)
    return (eigenvectors * clipped) @ eigenvectors.conjugate().T


class AffineGramCoordinates:
    """Affine maps of both moment matrices in the propagated coordinates.

    Built entirely from the public surface of
    :class:`SymmetryReducedPositiveExtension`,
    :func:`state_lower_moments` and :func:`uncentered_joint_moment_matrix`,
    by evaluating each at the coordinate basis.  The build verifies the
    affinity it assumes: a random probe must reproduce both matrices to
    ``1e-12`` absolute, or construction fails.

    Vocabulary follows the record.  The ``retained joint Gram`` is the affine
    eight-row Gram of ``(1, b, b^dagger, sigma)``; the ``24-row positive
    extension`` is the scaled extension matrix whose minimum eigenvalue the
    drivers report as the positive-extension eigenvalue.  The 62 frontier
    moments are carried in the standardized coordinates the extension's own
    frontier scales define.
    """

    def __init__(
        self,
        extension: SymmetryReducedPositiveExtension,
    ) -> None:
        self.extension = extension
        self.active_keys = tuple(extension.active_keys)
        self.lower_keys = tuple(extension.lower_keys)
        self.frontier_keys = tuple(extension.frontier_keys)
        self.state_count = _RAW_DIMENSION + len(self.active_keys)
        self.frontier_count = len(self.frontier_keys)
        self.extension_rows = int(extension.dimension)
        self.retained_rows = 8

        zero_state = np.zeros(self.state_count, dtype=float)
        probe = state_lower_moments(
            zero_state[:_RAW_DIMENSION],
            zero_state[_RAW_DIMENSION:],
            self.active_keys,
        )
        if tuple(probe) != self.lower_keys:
            raise RuntimeError(
                "state lower moments and the positive extension disagree on "
                "the lower-moment ordering"
            )
        lower_offset = np.asarray(
            [probe[key] for key in self.lower_keys], dtype=float
        )
        lower_map = np.empty(
            (len(self.lower_keys), self.state_count), dtype=float
        )
        for column in range(self.state_count):
            basis = zero_state.copy()
            basis[column] = 1.0
            mapping = state_lower_moments(
                basis[:_RAW_DIMENSION],
                basis[_RAW_DIMENSION:],
                self.active_keys,
            )
            lower_map[:, column] = (
                np.asarray([mapping[key] for key in self.lower_keys])
                - lower_offset
            )

        zero_lower = {key: 0.0 for key in self.lower_keys}
        zero_frontier = {key: 0.0 for key in self.frontier_keys}
        constant = extension.scaled_matrix(
            extension.matrix(zero_lower, zero_frontier)
        )
        lower_coefficients = np.empty(
            (len(self.lower_keys), self.extension_rows, self.extension_rows),
            dtype=complex,
        )
        for index, key in enumerate(self.lower_keys):
            values = dict(zero_lower)
            values[key] = 1.0
            lower_coefficients[index] = (
                extension.scaled_matrix(extension.matrix(values, zero_frontier))
                - constant
            )
        frontier_coefficients = np.empty(
            (self.frontier_count, self.extension_rows, self.extension_rows),
            dtype=complex,
        )
        for index, key in enumerate(self.frontier_keys):
            values = dict(zero_frontier)
            values[key] = 1.0
            frontier_coefficients[index] = (
                extension.scaled_matrix(extension.matrix(zero_lower, values))
                - constant
            )

        self._extension_constant = constant + np.tensordot(
            lower_offset, lower_coefficients, axes=(0, 0)
        )
        self._extension_state = np.einsum(
            "ki,kab->iab", lower_map, lower_coefficients, optimize=True
        )
        self._extension_frontier = (
            frontier_coefficients * extension.frontier_scales[:, None, None]
        )

        base_scales = np.asarray(
            [
                1.0,
                *(
                    np.sqrt(extension.settings.phonon_envelope + 1.0)
                    for _ in range(4)
                ),
                1.0,
                1.0,
                1.0,
            ],
            dtype=float,
        )
        inverse_base = np.diag(1.0 / base_scales)
        raw_zero = np.zeros(_RAW_DIMENSION, dtype=float)
        retained_constant = (
            inverse_base @ uncentered_joint_moment_matrix(raw_zero) @ inverse_base
        )
        retained_state = np.zeros(
            (self.state_count, self.retained_rows, self.retained_rows),
            dtype=complex,
        )
        for column in range(_RAW_DIMENSION):
            basis = raw_zero.copy()
            basis[column] = 1.0
            retained_state[column] = (
                inverse_base
                @ uncentered_joint_moment_matrix(basis)
                @ inverse_base
                - retained_constant
            )
        self._retained_constant = retained_constant
        self._retained_state = retained_state

        self._verify_affinity()
        self._factorizations: dict[
            tuple[str, float, float], tuple[FloatArray, FloatArray, FloatArray]
        ] = {}

    # -- affine evaluation -------------------------------------------------

    def scaled_extension_matrix(
        self, state: FloatArray, frontier_standardized: FloatArray
    ) -> ComplexArray:
        """The scaled 24-row positive extension at these coordinates."""

        return (
            self._extension_constant
            + np.tensordot(
                np.asarray(state, dtype=float), self._extension_state, axes=(0, 0)
            )
            + np.tensordot(
                np.asarray(frontier_standardized, dtype=float),
                self._extension_frontier,
                axes=(0, 0),
            )
        )

    def scaled_retained_matrix(self, state: FloatArray) -> ComplexArray:
        """The scaled retained joint Gram at these coordinates."""

        return self._retained_constant + np.tensordot(
            np.asarray(state, dtype=float), self._retained_state, axes=(0, 0)
        )

    def unshifted_retained_matrix(self, state: FloatArray) -> ComplexArray:
        """The retained joint Gram the record reports, without scaling."""

        raw = np.asarray(state, dtype=float)[:_RAW_DIMENSION]
        return uncentered_joint_moment_matrix(raw)

    def frontier_mapping(
        self, frontier_standardized: FloatArray
    ) -> Mapping[MomentKey, float]:
        values = (
            np.asarray(frontier_standardized, dtype=float)
            * self.extension.frontier_scales
        )
        return MappingProxyType(
            {
                key: float(value)
                for key, value in zip(self.frontier_keys, values, strict=True)
            }
        )

    def standardize_frontier(
        self, frontier_moments: Mapping[MomentKey, float]
    ) -> FloatArray:
        return (
            np.asarray(
                [float(frontier_moments[key]) for key in self.frontier_keys],
                dtype=float,
            )
            / self.extension.frontier_scales
        )

    def _verify_affinity(self) -> None:
        generator = np.random.default_rng(20260915)
        state = generator.normal(scale=0.05, size=self.state_count)
        frontier = generator.normal(scale=0.05, size=self.frontier_count)
        lower = state_lower_moments(
            state[:_RAW_DIMENSION], state[_RAW_DIMENSION:], self.active_keys
        )
        direct = self.extension.scaled_matrix(
            self.extension.matrix(lower, self.frontier_mapping(frontier))
        )
        predicted = self.scaled_extension_matrix(state, frontier)
        if not np.allclose(predicted, direct, atol=1e-12, rtol=0.0):
            raise RuntimeError(
                "the 24-row positive extension is not affine in the "
                "propagated coordinates"
            )
        base_scales = np.asarray(
            [
                1.0,
                *(
                    np.sqrt(self.extension.settings.phonon_envelope + 1.0)
                    for _ in range(4)
                ),
                1.0,
                1.0,
                1.0,
            ],
            dtype=float,
        )
        inverse_base = np.diag(1.0 / base_scales)
        direct_retained = (
            inverse_base
            @ uncentered_joint_moment_matrix(state[:_RAW_DIMENSION])
            @ inverse_base
        )
        if not np.allclose(
            self.scaled_retained_matrix(state),
            direct_retained,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                "the retained joint Gram is not affine in the propagated "
                "coordinates"
            )

    # -- least-squares machinery ------------------------------------------

    def _design(self, target: str) -> FloatArray:
        """Realified design matrix of the selected target in the parameters."""

        columns: list[FloatArray] = []
        for index in range(self.state_count):
            pieces: list[FloatArray] = []
            if target in ("extension", "both"):
                pieces.append(_hermitian_features(self._extension_state[index]))
            if target in ("retained", "both"):
                pieces.append(_hermitian_features(self._retained_state[index]))
            columns.append(np.concatenate(pieces))
        for index in range(self.frontier_count):
            pieces = []
            if target in ("extension", "both"):
                pieces.append(
                    _hermitian_features(self._extension_frontier[index])
                )
            if target in ("retained", "both"):
                pieces.append(
                    np.zeros(self.retained_rows**2, dtype=float)
                )
            columns.append(np.concatenate(pieces))
        return np.column_stack(columns)

    def factorization(
        self, settings: CheapProjectionSettings
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Return the factorization of one declared repair, built once.

        ``pseudo_inverse`` maps a realified target residual to the weighted
        minimum-norm increment of the movable coordinates; ``range_basis``
        is an orthonormal basis of the affine Gram set's tangent space under
        those same movable coordinates, which the Dykstra policy projects
        onto; ``scales`` is the declared per-group freedom; ``movable`` is
        the index set of coordinates the map back may move.
        """

        signature = (
            settings.target,
            settings.repair_coordinates,
            float(settings.state_motion_scale),
            float(settings.frontier_motion_scale),
        )
        cached = self._factorizations.get(signature)
        if cached is not None:
            return cached
        design = self._design(settings.target)
        if settings.repair_coordinates == "frontier_only":
            movable = np.arange(
                self.state_count, self.state_count + self.frontier_count
            )
        else:
            movable = np.arange(self.state_count + self.frontier_count)
        design = design[:, movable]
        scales = np.concatenate(
            (
                np.full(self.state_count, float(settings.state_motion_scale)),
                np.full(
                    self.frontier_count, float(settings.frontier_motion_scale)
                ),
            )
        )[movable]
        pseudo_inverse = np.linalg.pinv(design * scales[None, :])
        left, singular, _ = np.linalg.svd(design, full_matrices=False)
        rank = int(np.sum(singular > singular[0] * 1e-10))
        range_basis = np.ascontiguousarray(left[:, :rank])
        value = (pseudo_inverse, range_basis, scales, movable)
        self._factorizations[signature] = value
        return value

    def _target_features(
        self,
        target: str,
        state: FloatArray,
        frontier_standardized: FloatArray,
    ) -> FloatArray:
        pieces: list[FloatArray] = []
        if target in ("extension", "both"):
            pieces.append(
                _hermitian_features(
                    self.scaled_extension_matrix(state, frontier_standardized)
                )
            )
        if target in ("retained", "both"):
            pieces.append(
                _hermitian_features(self.scaled_retained_matrix(state))
            )
        return np.concatenate(pieces)

    def _cone_projection(
        self,
        features: FloatArray,
        target: str,
        floors: tuple[float, ...],
    ) -> FloatArray:
        """Clip every block of one realified point to its declared floor."""

        pieces: list[FloatArray] = []
        offset = 0
        for rows, floor in zip(self._block_rows(target), floors, strict=True):
            width = rows**2
            block = _hermitian_from_features(
                features[offset : offset + width], rows
            )
            pieces.append(_hermitian_features(_clip_to_floor(block, floor)))
            offset += width
        return np.concatenate(pieces)

    def _block_rows(self, target: str) -> tuple[int, ...]:
        if target == "extension":
            return (self.extension_rows,)
        if target == "retained":
            return (self.retained_rows,)
        return (self.extension_rows, self.retained_rows)

    def block_names(self, target: str) -> tuple[str, ...]:
        """The record's name of each block of the selected target."""

        if target == "extension":
            return ("extension",)
        if target == "retained":
            return ("retained",)
        return ("extension", "retained")

    def _block_minima(
        self, features: FloatArray, target: str
    ) -> tuple[float, ...]:
        minima: list[float] = []
        offset = 0
        for rows in self._block_rows(target):
            width = rows**2
            block = _hermitian_from_features(
                features[offset : offset + width], rows
            )
            minima.append(float(np.linalg.eigvalsh(block)[0]))
            offset += width
        return tuple(minima)

    # -- the repair itself -------------------------------------------------

    def repair(
        self,
        state: FloatArray,
        frontier_standardized: FloatArray,
        settings: CheapProjectionSettings,
    ) -> CheapProjectionRepair:
        """Apply the declared cheap repair to one pair of coordinates."""

        started = perf_counter()
        current_state = np.asarray(state, dtype=float)
        current_frontier = np.asarray(frontier_standardized, dtype=float)
        extension_before = float(
            np.linalg.eigvalsh(
                self.scaled_extension_matrix(current_state, current_frontier)
            )[0]
        )
        retained_before = float(
            np.linalg.eigvalsh(self.scaled_retained_matrix(current_state))[0]
        )
        names = self.block_names(settings.target)
        floors = settings.block_floors(names)
        if settings.policy == "none":
            return CheapProjectionRepair(
                state=current_state.copy(),
                frontier_standardized=current_frontier.copy(),
                applied=False,
                iterations=0,
                extension_minimum_before=extension_before,
                extension_minimum_after=extension_before,
                retained_minimum_before=retained_before,
                retained_minimum_after=retained_before,
                state_displacement=0.0,
                frontier_displacement=0.0,
                target_residual=0.0,
                seconds=perf_counter() - started,
            )
        features = self._target_features(
            settings.target, current_state, current_frontier
        )
        minima = self._block_minima(features, settings.target)
        if all(
            value >= floor - settings.repair_tolerance
            for value, floor in zip(minima, floors, strict=True)
        ):
            return CheapProjectionRepair(
                state=current_state.copy(),
                frontier_standardized=current_frontier.copy(),
                applied=False,
                iterations=0,
                extension_minimum_before=extension_before,
                extension_minimum_after=extension_before,
                retained_minimum_before=retained_before,
                retained_minimum_after=retained_before,
                state_displacement=0.0,
                frontier_displacement=0.0,
                target_residual=0.0,
                seconds=perf_counter() - started,
            )
        pseudo_inverse, range_basis, scales, movable = self.factorization(
            settings
        )
        if settings.policy == "dykstra":
            corrected, iterations = self._dykstra(
                features,
                range_basis,
                settings.target,
                floors,
                settings.dykstra_iterations,
            )
        else:
            corrected = self._cone_projection(
                features, settings.target, floors
            )
            iterations = 1
        residual = corrected - features
        increment = np.zeros(self.state_count + self.frontier_count)
        increment[movable] = scales * (pseudo_inverse @ residual)
        new_state = current_state + increment[: self.state_count]
        new_frontier = current_frontier + increment[self.state_count :]
        achieved = self._target_features(
            settings.target, new_state, new_frontier
        )
        extension_after = float(
            np.linalg.eigvalsh(
                self.scaled_extension_matrix(new_state, new_frontier)
            )[0]
        )
        retained_after = float(
            np.linalg.eigvalsh(self.scaled_retained_matrix(new_state))[0]
        )
        return CheapProjectionRepair(
            state=new_state,
            frontier_standardized=new_frontier,
            applied=True,
            iterations=iterations,
            extension_minimum_before=extension_before,
            extension_minimum_after=extension_after,
            retained_minimum_before=retained_before,
            retained_minimum_after=retained_after,
            state_displacement=float(
                np.linalg.norm(increment[: self.state_count])
            ),
            frontier_displacement=float(
                np.linalg.norm(increment[self.state_count :])
            ),
            target_residual=float(np.linalg.norm(achieved - corrected)),
            seconds=perf_counter() - started,
        )

    def _dykstra(
        self,
        features: FloatArray,
        range_basis: FloatArray,
        target: str,
        floors: tuple[float, ...],
        iterations: int,
    ) -> tuple[FloatArray, int]:
        """Alternate between the affine Gram set and the cone, with memory.

        The affine Gram set is the image of the affine coordinate map; its
        orthogonal projection in the realified metric is the projection of
        the displacement from the current point onto the range basis, which
        is the same subspace for every declared motion scale.
        """

        point = features.copy()
        cone_correction = np.zeros_like(features)
        affine_correction = np.zeros_like(features)
        for index in range(iterations):
            shifted = point + cone_correction
            projected = self._cone_projection(shifted, target, floors)
            cone_correction = shifted - projected
            shifted = projected + affine_correction
            displacement = shifted - features
            affine_point = features + range_basis @ (
                range_basis.T @ displacement
            )
            affine_correction = shifted - affine_point
            point = affine_point
            if all(
                value >= floor
                for value, floor in zip(
                    self._block_minima(point, target), floors, strict=True
                )
            ):
                return point, index + 1
        return point, iterations


@dataclass(frozen=True)
class CheapProjectionTrajectory:
    """A fixed-step SSPRK(3,3) rollout under one cheap repair policy."""

    times: FloatArray
    states: FloatArray
    frontier_standardized: FloatArray
    retained_minimum_eigenvalues: FloatArray
    scaled_extension_minimum_eigenvalues: FloatArray
    maximum_frontier_envelope_ratio: FloatArray
    completed_steps: int
    velocity_evaluations: int
    repairs_applied: int
    repairs_skipped: int
    repair_seconds: float
    velocity_seconds: float
    maximum_state_displacement: float
    maximum_frontier_displacement: float
    maximum_dykstra_iterations: int
    success: bool
    message: str
    records: tuple[dict[str, Any], ...] = field(default=())
    lift_reselections: int = 0
    lift_reselection_failures: int = 0
    lift_reselection_seconds: float = 0.0
    maximum_lift_reselection_displacement: float = 0.0
    minimum_reselected_scaled_extension_eigenvalue: float = float("nan")


class CheapProjectionRoute:
    """The unguarded fixed-dictionary velocity plus one cheap repair.

    The velocity is exactly the fixed dictionary's unconstrained predictor:
    the archive-plus-entrance target velocity
    ``augmented_retained_velocity`` and the exact active commutator velocity
    ``auxiliary_velocity`` that ``FixedDictionaryProjectedAPCM.targets``
    builds, including the exact K/P/D entrance correction.  The
    retained-first tangent-cone projection, the stage retraction and the
    per-stage proximal log-det completion are never called: the lift the
    commutator velocity reads is the carried frontier, repaired by the
    declared cheap projection.
    """

    def __init__(
        self,
        model: FixedDictionaryProjectedAPCM,
        settings: CheapProjectionSettings | None = None,
        *,
        coordinates: AffineGramCoordinates | None = None,
    ) -> None:
        self.model = model
        self.settings = (
            CheapProjectionSettings() if settings is None else settings
        )
        self.coordinates = (
            AffineGramCoordinates(model.extension)
            if coordinates is None
            else coordinates
        )
        if self.coordinates.extension is not model.extension:
            raise ValueError(
                "affine coordinates must be built on the model's extension"
            )

    def carried_completion(
        self, state: FloatArray, frontier_standardized: FloatArray
    ) -> APCMExtensionResult:
        """Assemble the carried lift as an extension result, with no solve.

        ``FixedDictionaryProjectedAPCM.targets`` accepts a supplied
        completion whose lower moments equal the state's own; this builds
        exactly that from the carried frontier coordinates.  Its
        ``success`` flag says the lift is usable as a moment source, not
        that any cone was certified: no certificate exists on this route.
        """

        raw, hidden = unpack_projected_apcm_state(
            state, active_keys=self.coordinates.active_keys
        )
        lower = state_lower_moments(raw, hidden, self.coordinates.active_keys)
        frontier = self.coordinates.frontier_mapping(frontier_standardized)
        matrix = self.model.extension.matrix(lower, frontier)
        scaled = self.model.extension.scaled_matrix(matrix)
        return APCMExtensionResult(
            lower_moments=MappingProxyType(
                {
                    key: float(lower[key])
                    for key in self.model.extension.lower_keys
                }
            ),
            frontier_moments=frontier,
            moment_matrix=matrix,
            scaled_minimum_eigenvalue=float(np.linalg.eigvalsh(scaled)[0]),
            minimum_eigenvalue=float(np.linalg.eigvalsh(matrix)[0]),
            objective=float("nan"),
            kkt_backward_error=float("nan"),
            iterations=0,
            success=True,
            message="cheap_projection_carried_lift",
        )

    def reselect_lift(
        self,
        state: FloatArray,
        frontier_standardized: FloatArray,
        *,
        warm_start: bool = True,
    ) -> tuple[FloatArray, APCMExtensionResult, float]:
        """Solve the declared proximal log-det completion at this state.

        This is the one place on the route where a conic solve runs after the
        preparation, and it runs only on the declared re-selection cadence:
        the hybrid re-selects the lift every ``N`` accepted steps and carries
        it, repaired by the cheap projection, in between.  The solve is
        ``SymmetryReducedPositiveExtension.complete``, that is the same
        proximal log-det completion the projective route calls at every
        stage; ``warm_start`` hands it the carried frontier as its warm
        frontier, which is the projective route's non-cold
        ``completion_start``.

        A refusal is reported, never raised: the returned frontier is then
        the carried one, unchanged, and the caller counts the refusal.
        """

        raw, hidden = unpack_projected_apcm_state(
            state, active_keys=self.coordinates.active_keys
        )
        lower = state_lower_moments(raw, hidden, self.coordinates.active_keys)
        warm = (
            self.coordinates.frontier_mapping(frontier_standardized)
            if warm_start
            else None
        )
        started = perf_counter()
        result = self.model.extension.complete(lower, warm_frontier=warm)
        seconds = perf_counter() - started
        if not result.success:
            return (
                np.asarray(frontier_standardized, dtype=float).copy(),
                result,
                seconds,
            )
        return (
            self.coordinates.standardize_frontier(result.frontier_moments),
            result,
            seconds,
        )

    def unguarded_velocity(
        self,
        time: float,
        state: FloatArray,
        frontier_standardized: FloatArray,
    ) -> tuple[FloatArray, ProjectedAPCMTargets]:
        """The unconstrained fixed-dictionary predictor at this lift."""

        completion = self.carried_completion(state, frontier_standardized)
        targets = self.model.targets(
            float(time), np.asarray(state, dtype=float), completion=completion
        )
        velocity = np.concatenate(
            (targets.augmented_retained_velocity, targets.auxiliary_velocity)
        )
        return velocity, targets


def integrate_cheap_projection_ssprk3(
    route: CheapProjectionRoute,
    initial_state: FloatArray,
    initial_frontier_standardized: FloatArray,
    *,
    final_time: float,
    time_step: float,
    initial_time: float = 0.0,
    progress: Callable[[str], None] | None = None,
    deadline: float | None = None,
    record_stages: bool = False,
    lift_reselection_interval: int = 0,
    lift_reselection_warm_start: bool = True,
) -> CheapProjectionTrajectory:
    """SSPRK(3,3) on the unguarded velocity with the declared cheap repair.

    The stage weights are the same SSPRK(3,3) weights the committed
    fixed-dictionary integrator uses.  ``cadence="stage"`` repairs the trial
    coordinates of every stage; ``cadence="step"`` repairs only the accepted
    state at the end of a step.  Nothing here can refuse: the rollout stops
    only on a nonfinite state, the wall budget, or an exception raised by the
    velocity evaluation itself.

    ``lift_reselection_interval`` is the hybrid: ``0`` carries the
    preparation's lift for the whole rollout, which is the route's original
    form; a positive ``N`` re-solves the declared proximal log-det completion
    at the accepted state every ``N`` steps, after that step's repair, and
    carries the solved lift until the next re-selection.  A completion that
    refuses is counted and the carried lift is kept, so the rollout still
    cannot stop on a conic refusal.
    """

    if final_time <= 0.0 or time_step <= 0.0:
        raise ValueError("final_time and time_step must be positive")
    if lift_reselection_interval < 0:
        raise ValueError("lift_reselection_interval must be nonnegative")
    span = float(final_time) - float(initial_time)
    if span <= 0.0:
        raise ValueError("final_time must exceed initial_time")
    intervals = int(round(span / time_step))
    if not np.isclose(intervals * time_step, span, atol=1e-12):
        raise ValueError("final_time must be an integer multiple of time_step")

    coordinates = route.coordinates
    settings = route.settings
    state = np.asarray(initial_state, dtype=float).copy()
    frontier = np.asarray(initial_frontier_standardized, dtype=float).copy()
    if state.shape != (coordinates.state_count,):
        raise ValueError("initial_state has the wrong dimension")
    if frontier.shape != (coordinates.frontier_count,):
        raise ValueError("initial frontier has the wrong dimension")

    times = float(initial_time) + np.linspace(0.0, span, intervals + 1)
    states = np.empty((intervals + 1, state.size), dtype=float)
    frontiers = np.empty((intervals + 1, frontier.size), dtype=float)
    retained_minima = np.empty(intervals + 1, dtype=float)
    extension_minima = np.empty(intervals + 1, dtype=float)
    envelope = np.empty(intervals + 1, dtype=float)
    records: list[dict[str, Any]] = []

    counters = {
        "applied": 0,
        "skipped": 0,
        "reselections": 0,
        "reselection_failures": 0,
    }
    totals = {
        "repair_seconds": 0.0,
        "velocity_seconds": 0.0,
        "state_displacement": 0.0,
        "frontier_displacement": 0.0,
        "dykstra_iterations": 0,
        "reselection_seconds": 0.0,
        "reselection_displacement": 0.0,
        "reselected_minimum": float("inf"),
    }

    def sample(index: int, current: FloatArray, lift: FloatArray) -> None:
        states[index] = current
        frontiers[index] = lift
        retained_minima[index] = float(
            np.linalg.eigvalsh(
                coordinates.unshifted_retained_matrix(current)
            )[0]
        )
        extension_minima[index] = float(
            np.linalg.eigvalsh(
                coordinates.scaled_extension_matrix(current, lift)
            )[0]
        )
        envelope[index] = float(np.max(np.abs(lift)))

    def repair(
        current: FloatArray, lift: FloatArray, stage: str, when: float
    ) -> tuple[FloatArray, FloatArray]:
        result = coordinates.repair(current, lift, settings)
        totals["repair_seconds"] += result.seconds
        if result.applied:
            counters["applied"] += 1
            totals["state_displacement"] = max(
                totals["state_displacement"], result.state_displacement
            )
            totals["frontier_displacement"] = max(
                totals["frontier_displacement"], result.frontier_displacement
            )
            totals["dykstra_iterations"] = max(
                totals["dykstra_iterations"], result.iterations
            )
        else:
            counters["skipped"] += 1
        if record_stages:
            records.append(
                {
                    "stage": stage,
                    "time": float(when),
                    "applied": bool(result.applied),
                    "iterations": int(result.iterations),
                    "extension_minimum_before": (
                        result.extension_minimum_before
                    ),
                    "extension_minimum_after": result.extension_minimum_after,
                    "retained_minimum_before": result.retained_minimum_before,
                    "retained_minimum_after": result.retained_minimum_after,
                    "state_displacement": result.state_displacement,
                    "frontier_displacement": result.frontier_displacement,
                    "target_residual": result.target_residual,
                    "seconds": result.seconds,
                }
            )
        return result.state, result.frontier_standardized

    def velocity(when: float, current: FloatArray, lift: FloatArray) -> FloatArray:
        started = perf_counter()
        value, _ = route.unguarded_velocity(when, current, lift)
        totals["velocity_seconds"] += perf_counter() - started
        return value

    sample(0, state, frontier)
    completed = intervals
    success = True
    message = "completed"
    evaluations = 0
    stage_cadence = settings.cadence == "stage"

    for step in range(intervals):
        if deadline is not None and perf_counter() >= deadline:
            completed = step
            success = False
            message = (
                f"wall_budget_exceeded after {step} of {intervals} "
                "completed steps"
            )
            break
        when = float(times[step])
        try:
            first_velocity = velocity(when, state, frontier)
            evaluations += 1
            first_state = state + time_step * first_velocity
            first_frontier = frontier
            if stage_cadence:
                first_state, first_frontier = repair(
                    first_state, first_frontier, "stage_1", when + time_step
                )

            second_velocity = velocity(
                when + time_step, first_state, first_frontier
            )
            evaluations += 1
            second_state = 0.75 * state + 0.25 * (
                first_state + time_step * second_velocity
            )
            second_frontier = first_frontier
            if stage_cadence:
                second_state, second_frontier = repair(
                    second_state,
                    second_frontier,
                    "stage_2",
                    when + 0.5 * time_step,
                )

            third_velocity = velocity(
                when + 0.5 * time_step, second_state, second_frontier
            )
            evaluations += 1
            final_state = (1.0 / 3.0) * state + (2.0 / 3.0) * (
                second_state + time_step * third_velocity
            )
            final_frontier = second_frontier
            final_state, final_frontier = repair(
                final_state,
                final_frontier,
                "stage_3" if stage_cadence else "step",
                when + time_step,
            )
            if (
                lift_reselection_interval > 0
                and (step + 1) % lift_reselection_interval == 0
            ):
                carried = final_frontier
                final_frontier, completion, seconds = route.reselect_lift(
                    final_state,
                    carried,
                    warm_start=lift_reselection_warm_start,
                )
                totals["reselection_seconds"] += seconds
                counters["reselections"] += 1
                if not completion.success:
                    counters["reselection_failures"] += 1
                else:
                    totals["reselected_minimum"] = min(
                        totals["reselected_minimum"],
                        float(completion.scaled_minimum_eigenvalue),
                    )
                totals["reselection_displacement"] = max(
                    totals["reselection_displacement"],
                    float(np.linalg.norm(final_frontier - carried)),
                )
                if record_stages:
                    records.append(
                        {
                            "stage": "lift_reselection",
                            "time": float(when + time_step),
                            "applied": bool(completion.success),
                            "iterations": int(completion.iterations),
                            "scaled_minimum_eigenvalue": float(
                                completion.scaled_minimum_eigenvalue
                            ),
                            "frontier_displacement": float(
                                np.linalg.norm(final_frontier - carried)
                            ),
                            "seconds": float(seconds),
                            "message": completion.message,
                        }
                    )
        except Exception as error:  # noqa: BLE001 - every failure is reported
            completed = step
            success = False
            message = (
                f"velocity_failure after {step} of {intervals} completed "
                f"steps: {type(error).__name__}: {error}"
            )
            break
        if not np.all(np.isfinite(final_state)) or not np.all(
            np.isfinite(final_frontier)
        ):
            completed = step
            success = False
            message = (
                f"nonfinite_state after {step} of {intervals} completed steps"
            )
            break
        state = final_state
        frontier = final_frontier
        sample(step + 1, state, frontier)
        if progress is not None and (
            step == 0
            or step + 1 == intervals
            or (step + 1) % max(1, intervals // 20) == 0
        ):
            progress(
                f"t={times[step + 1]:.6f}/{final_time:.6f} "
                f"lambda_retained={retained_minima[step + 1]:.3e} "
                f"lambda_extension={extension_minima[step + 1]:.3e} "
                f"envelope={envelope[step + 1]:.3e} "
                f"repairs={counters['applied']} "
                f"reselections={counters['reselections']}"
            )

    kept = completed + 1
    return CheapProjectionTrajectory(
        times=times[:kept],
        states=states[:kept],
        frontier_standardized=frontiers[:kept],
        retained_minimum_eigenvalues=retained_minima[:kept],
        scaled_extension_minimum_eigenvalues=extension_minima[:kept],
        maximum_frontier_envelope_ratio=envelope[:kept],
        completed_steps=completed,
        velocity_evaluations=evaluations,
        repairs_applied=counters["applied"],
        repairs_skipped=counters["skipped"],
        repair_seconds=totals["repair_seconds"],
        velocity_seconds=totals["velocity_seconds"],
        maximum_state_displacement=totals["state_displacement"],
        maximum_frontier_displacement=totals["frontier_displacement"],
        maximum_dykstra_iterations=int(totals["dykstra_iterations"]),
        success=success,
        message=message,
        records=tuple(records),
        lift_reselections=int(counters["reselections"]),
        lift_reselection_failures=int(counters["reselection_failures"]),
        lift_reselection_seconds=float(totals["reselection_seconds"]),
        maximum_lift_reselection_displacement=float(
            totals["reselection_displacement"]
        ),
        minimum_reselected_scaled_extension_eigenvalue=float(
            totals["reselected_minimum"]
        ),
    )


def prepare_cheap_projection_route(
    parameters: Any,
    *,
    phonon_cutoff: int = 16,
    active_keys: tuple[MomentKey, ...] = ENTRANCE_RELATIVE_MOMENT_KEYS,
    settings: CheapProjectionSettings | None = None,
    model_settings: Any = None,
    projection_settings: Any = None,
    extension_settings: Any = None,
) -> tuple[CheapProjectionRoute, FloatArray, FloatArray, float]:
    """Build the route at the canonical correlated preparation.

    Returns the route, the initial 44-coordinate state, the initial
    standardized frontier coordinates, and the exact ground energy.  The
    initial lift is the declared proximal log-det completion at the
    preparation, which is the one conic solve on this route and is paid once
    at ``t = 0``; every later lift is carried and repaired.
    """

    from .apcm_moment_projection import (
        APCMProjectionSettings,
        SymmetryReducedAPCMGeometry,
    )
    from .apcm_positive_extension import APCMExtensionSettings
    from .projected_apcm import (
        ProjectedAPCMSettings,
        prepare_projected_apcm_initial_state,
    )

    extension = SymmetryReducedPositiveExtension(
        settings=(
            APCMExtensionSettings()
            if extension_settings is None
            else extension_settings
        ),
        active_keys=tuple(active_keys),
    )
    geometry = SymmetryReducedAPCMGeometry(
        extension,
        settings=(
            APCMProjectionSettings()
            if projection_settings is None
            else projection_settings
        ),
    )
    model = FixedDictionaryProjectedAPCM(
        parameters,
        extension=extension,
        geometry=geometry,
        settings=(
            ProjectedAPCMSettings()
            if model_settings is None
            else model_settings
        ),
    )
    initialization = prepare_projected_apcm_initial_state(
        parameters, phonon_cutoff=phonon_cutoff, extension=extension
    )
    route = CheapProjectionRoute(model, settings)
    frontier = route.coordinates.standardize_frontier(
        initialization.completion.frontier_moments
    )
    return (
        route,
        np.asarray(initialization.state, dtype=float),
        frontier,
        float(initialization.exact_ground_energy),
    )




class CarriedWitnessAffineGram:
    """Affine map of the 62-row unified Gram in the 384 carried coordinates.

    The carried-witness flow propagates 44 retained archive-plus-entrance
    coordinates and 340 completion coordinates carrying one affine 62-row
    unified Gram.  ``CarriedWitnessGeometry.scaled_unified_matrix`` is affine
    in all 384 of them, which this class verifies at build time by probing
    that public method at the coordinate basis.  The repair is then the same
    construction as :class:`AffineGramCoordinates`: project the Gram onto the
    positive semidefinite cone, or alternate between that cone and the affine
    Gram set, and map the corrected Gram back to the 384 coordinates by one
    least-squares solve.  No conic solve runs.
    """

    def __init__(self, geometry: Any) -> None:
        self.geometry = geometry
        self.state_count = int(geometry.state_count)
        self.retained_count = int(geometry.retained_count)
        self.completion_count = int(geometry.completion_count)
        zero = np.zeros(self.state_count, dtype=float)
        constant = geometry.scaled_unified_matrix(*geometry.unpack_state(zero))
        self.rows = int(constant.shape[0])
        coefficients = np.empty(
            (self.state_count, self.rows, self.rows), dtype=complex
        )
        for column in range(self.state_count):
            basis = zero.copy()
            basis[column] = 1.0
            coefficients[column] = (
                geometry.scaled_unified_matrix(*geometry.unpack_state(basis))
                - constant
            )
        self._constant = constant
        self._coefficients = coefficients
        generator = np.random.default_rng(20260915)
        probe = generator.normal(scale=0.05, size=self.state_count)
        direct = geometry.scaled_unified_matrix(*geometry.unpack_state(probe))
        if not np.allclose(
            self.scaled_unified_matrix(probe), direct, atol=1e-12, rtol=0.0
        ):
            raise RuntimeError(
                "the 62-row unified Gram is not affine in the carried "
                "coordinates"
            )
        self._factorizations: dict[
            tuple[str, float, float], tuple[FloatArray, FloatArray, FloatArray]
        ] = {}

    def scaled_unified_matrix(self, state: FloatArray) -> ComplexArray:
        return self._constant + np.tensordot(
            np.asarray(state, dtype=float), self._coefficients, axes=(0, 0)
        )

    def _design(self) -> FloatArray:
        return np.column_stack(
            [
                _hermitian_features(self._coefficients[index])
                for index in range(self.state_count)
            ]
        )

    def factorization(
        self, settings: CheapProjectionSettings
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return ``(pseudo_inverse, range_basis, movable)``, built once."""

        signature = (
            settings.repair_coordinates,
            float(settings.state_motion_scale),
            float(settings.frontier_motion_scale),
        )
        cached = self._factorizations.get(signature)
        if cached is not None:
            return cached
        design = self._design()
        if settings.repair_coordinates == "frontier_only":
            movable = np.arange(self.retained_count, self.state_count)
        else:
            movable = np.arange(self.state_count)
        design = design[:, movable]
        scales = np.concatenate(
            (
                np.full(
                    self.retained_count, float(settings.state_motion_scale)
                ),
                np.full(
                    self.completion_count,
                    float(settings.frontier_motion_scale),
                ),
            )
        )[movable]
        pseudo_inverse = np.linalg.pinv(design * scales[None, :])
        left, singular, _ = np.linalg.svd(design, full_matrices=False)
        rank = int(np.sum(singular > singular[0] * 1e-10))
        value = (
            pseudo_inverse * scales[:, None],
            np.ascontiguousarray(left[:, :rank]),
            movable,
        )
        self._factorizations[signature] = value
        return value

    def repair(
        self, state: FloatArray, settings: CheapProjectionSettings
    ) -> CheapProjectionRepair:
        """Apply the declared cheap repair to one 384-coordinate state."""

        started = perf_counter()
        current = np.asarray(state, dtype=float)
        matrix = self.scaled_unified_matrix(current)
        before = float(np.linalg.eigvalsh(matrix)[0])
        floor = settings.applied_floor
        unchanged = CheapProjectionRepair(
            state=current.copy(),
            frontier_standardized=np.zeros(0, dtype=float),
            applied=False,
            iterations=0,
            extension_minimum_before=before,
            extension_minimum_after=before,
            retained_minimum_before=float("nan"),
            retained_minimum_after=float("nan"),
            state_displacement=0.0,
            frontier_displacement=0.0,
            target_residual=0.0,
            seconds=perf_counter() - started,
        )
        if settings.policy == "none":
            return unchanged
        if before >= floor - settings.repair_tolerance:
            return unchanged
        features = _hermitian_features(matrix)
        pseudo_inverse, range_basis, movable = self.factorization(settings)
        if settings.policy == "dykstra":
            corrected, iterations = _dykstra_single_block(
                features,
                range_basis,
                self.rows,
                floor,
                settings.dykstra_iterations,
            )
        else:
            corrected = _hermitian_features(_clip_to_floor(matrix, floor))
            iterations = 1
        increment = np.zeros(self.state_count, dtype=float)
        increment[movable] = pseudo_inverse @ (corrected - features)
        repaired = current + increment
        achieved = _hermitian_features(self.scaled_unified_matrix(repaired))
        return CheapProjectionRepair(
            state=repaired,
            frontier_standardized=np.zeros(0, dtype=float),
            applied=True,
            iterations=iterations,
            extension_minimum_before=before,
            extension_minimum_after=float(
                np.linalg.eigvalsh(self.scaled_unified_matrix(repaired))[0]
            ),
            retained_minimum_before=float("nan"),
            retained_minimum_after=float("nan"),
            state_displacement=float(
                np.linalg.norm(increment[: self.retained_count])
            ),
            frontier_displacement=float(
                np.linalg.norm(increment[self.retained_count :])
            ),
            target_residual=float(np.linalg.norm(achieved - corrected)),
            seconds=perf_counter() - started,
        )


def _dykstra_single_block(
    features: FloatArray,
    range_basis: FloatArray,
    rows: int,
    floor: float,
    iterations: int,
) -> tuple[FloatArray, int]:
    """Dykstra between one affine Gram set and one positive-definite cone."""

    point = features.copy()
    cone_correction = np.zeros_like(features)
    affine_correction = np.zeros_like(features)
    for index in range(iterations):
        shifted = point + cone_correction
        projected = _hermitian_features(
            _clip_to_floor(_hermitian_from_features(shifted, rows), floor)
        )
        cone_correction = shifted - projected
        shifted = projected + affine_correction
        displacement = shifted - features
        affine_point = features + range_basis @ (range_basis.T @ displacement)
        affine_correction = shifted - affine_point
        point = affine_point
        minimum = float(
            np.linalg.eigvalsh(_hermitian_from_features(point, rows))[0]
        )
        if minimum >= floor:
            return point, index + 1
    return point, iterations


@dataclass(frozen=True)
class CarriedWitnessCheapProjectionTrajectory:
    """A no-guard CWRMF rollout with a cheap repair after every step."""

    times: FloatArray
    states: FloatArray
    minimum_unshifted_eigenvalues: FloatArray
    completed_steps: int
    repairs_applied: int
    repairs_skipped: int
    repair_seconds: float
    rollout_seconds: float
    maximum_retained_displacement: float
    maximum_completion_displacement: float
    maximum_dykstra_iterations: int
    minimum_before_repair: float
    success: bool
    message: str


def integrate_carried_witness_cheap_projection(
    model: Any,
    gram: CarriedWitnessAffineGram,
    initial_state: FloatArray,
    *,
    settings: CheapProjectionSettings,
    final_time: float,
    time_step: float,
    initial_time: float = 0.0,
    progress: Callable[[str], None] | None = None,
    deadline: float | None = None,
) -> CarriedWitnessCheapProjectionTrajectory:
    """Step the carried-witness flow and repair the 62-row Gram each step.

    The propagation is the record's own ``integrate_cwrmf_ssprk2``, called
    one step at a time in its no-guard mode (``enforce_gram_guard=False``,
    which removes the 62-row positive-semidefinite optimization and the
    endpoint rejection together).  Between steps the declared cheap repair is
    applied to the accepted 384-coordinate state.  Nothing else changes: the
    grid, the scheme and the velocity are the no-guard ones.
    """

    from .apcm_carried_witness import integrate_cwrmf_ssprk2

    span = float(final_time) - float(initial_time)
    if span <= 0.0 or time_step <= 0.0:
        raise ValueError("times must be ordered and time_step positive")
    intervals = int(round(span / time_step))
    if not np.isclose(intervals * time_step, span, atol=1e-12):
        raise ValueError("final_time must be an integer multiple of time_step")

    times = float(initial_time) + np.linspace(0.0, span, intervals + 1)
    states = np.empty((intervals + 1, gram.state_count), dtype=float)
    minima = np.empty(intervals + 1, dtype=float)
    state = np.asarray(initial_state, dtype=float).copy()
    states[0] = state
    minima[0] = float(np.linalg.eigvalsh(gram.scaled_unified_matrix(state))[0])

    applied = 0
    skipped = 0
    repair_seconds = 0.0
    rollout_seconds = 0.0
    retained_displacement = 0.0
    completion_displacement = 0.0
    dykstra_iterations = 0
    minimum_before = minima[0]
    completed = intervals
    success = True
    message = "completed"

    for step in range(intervals):
        if deadline is not None and perf_counter() >= deadline:
            completed = step
            success = False
            message = (
                f"wall_budget_exceeded after {step} of {intervals} "
                "completed steps"
            )
            break
        started = perf_counter()
        segment = integrate_cwrmf_ssprk2(
            model,
            state,
            initial_time=float(times[step]),
            final_time=float(times[step + 1]),
            time_step=time_step,
        )
        rollout_seconds += perf_counter() - started
        if not segment.success:
            completed = step
            success = False
            message = (
                f"cwrmf_step_failure after {step} of {intervals} completed "
                f"steps: {segment.message}"
            )
            break
        result = gram.repair(segment.states[-1], settings)
        repair_seconds += result.seconds
        minimum_before = min(minimum_before, result.extension_minimum_before)
        if result.applied:
            applied += 1
            retained_displacement = max(
                retained_displacement, result.state_displacement
            )
            completion_displacement = max(
                completion_displacement, result.frontier_displacement
            )
            dykstra_iterations = max(dykstra_iterations, result.iterations)
        else:
            skipped += 1
        state = result.state
        if not np.all(np.isfinite(state)):
            completed = step
            success = False
            message = (
                f"nonfinite_state after {step} of {intervals} completed steps"
            )
            break
        states[step + 1] = state
        minima[step + 1] = result.extension_minimum_after
        if progress is not None and (
            step == 0
            or step + 1 == intervals
            or (step + 1) % max(1, intervals // 20) == 0
        ):
            progress(
                f"t={times[step + 1]:.6f}/{final_time:.6f} "
                f"lambda_unified={minima[step + 1]:.3e} "
                f"repairs={applied}"
            )

    kept = completed + 1
    return CarriedWitnessCheapProjectionTrajectory(
        times=times[:kept],
        states=states[:kept],
        minimum_unshifted_eigenvalues=minima[:kept],
        completed_steps=completed,
        repairs_applied=applied,
        repairs_skipped=skipped,
        repair_seconds=repair_seconds,
        rollout_seconds=rollout_seconds,
        maximum_retained_displacement=retained_displacement,
        maximum_completion_displacement=completion_displacement,
        maximum_dykstra_iterations=dykstra_iterations,
        minimum_before_repair=minimum_before,
        success=success,
        message=message,
    )


__all__ = [
    "AffineGramCoordinates",
    "CarriedWitnessCheapProjectionTrajectory",
    "integrate_carried_witness_cheap_projection",
    "CarriedWitnessAffineGram",
    "CHEAP_PROJECTION_CADENCES",
    "CHEAP_PROJECTION_COORDINATES",
    "CHEAP_PROJECTION_POLICIES",
    "CHEAP_PROJECTION_TARGETS",
    "CheapProjectionRepair",
    "CheapProjectionRoute",
    "CheapProjectionSettings",
    "CheapProjectionTrajectory",
    "integrate_cheap_projection_ssprk3",
    "prepare_cheap_projection_route",
]
