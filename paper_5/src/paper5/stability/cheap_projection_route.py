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
from math import copysign
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from .adaptive_positive_moment import (
    ENTRANCE_RELATIVE_MOMENT_KEYS,
    RAW_MOMENT_COORDINATE_NAMES,
    raw_moment_coordinates_to_matrix_state,
    uncentered_joint_moment_matrix,
)
from .apcm_moment_projection import state_lower_moments
from .apcm_positive_extension import (
    APCMExtensionResult,
    SymmetryReducedPositiveExtension,
)
from .hubbard_dimer import DimerParameters
from .matrix_reference import (
    closed_scalar_to_matrix_state,
    matrix_state_to_closed_scalar_coordinates,
    matrix_total_energy,
)
from .moment_hierarchy import MomentKey
from .projected_apcm import (
    FixedDictionaryProjectedAPCM,
    ProjectedAPCMTargets,
    build_completion_ladder,
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
CHEAP_PROJECTION_ENERGY_REFERENCES = ("none", "post_pulse", "drive_power")

RAW_PHONON_NORMAL_COORDINATE_NAMES = (
    "raw_normal_00",
    "raw_normal_11",
    "raw_normal_01_real",
    "raw_normal_01_imag",
)
"""The four raw phonon normal coordinates among the 29 raw moment ones.

Named here because they are the block the Eq. (22) internal energy reads
(gradient ``0.5`` on ``raw_normal_00`` and ``0.5`` on ``raw_normal_11``, zero
on the two off-diagonal ones) and the block the relative-mode population
``(phonon[0,0] + phonon[1,1]) / 2 - Re phonon[0,1]`` is a quantity of.
"""

LIFT_RESELECTION_TRUST_POLICIES = ("carried", "ball")
LIFT_RESELECTION_CANDIDATES = ("certified", "measured")


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
    ``energy_reference_mode``
        Whether the ``dykstra`` repair also holds the Eq. (22) internal
        energy, evaluated from the same coordinates the scorer uses, on a
        declared reference.  ``"none"`` is the repair of the earlier passes.
        ``"post_pulse"`` turns the constraint on at a declared time and holds
        the energy at the value the trajectory itself had there, which is the
        exact conservation law of this model after the drive ends;
        ``"drive_power"`` turns it on from the start and advances the
        reference by the power the drive delivers, which the model supplies
        without the exact state because the Hamiltonian's only time
        dependence is ``DimerParameters.drive_difference`` and that function
        vanishes at ``t = 0``.
    ``energy_tolerance``
        With an active energy reference, a repair is applied whenever the
        internal energy is further than this from the reference, even when
        both cones are already satisfied.  Zero constrains the energy at
        every repair opportunity.
    ``energy_reference_band``
        The declared half-width of a band around that same reference, which
        turns the set the alternation meets from the hyperplane
        ``E = reference`` into the slab ``|E - reference| <= band``.  Zero,
        the default, is the declared equality of every earlier pass.  With a
        positive band the alternation's energy step corrects only the excess:
        a candidate whose Eq. (22) internal energy is already inside the band
        is left where the cone projection put it, and one outside it is moved
        to the nearer boundary and no further, so a cone correction that
        costs less energy than the band is let through whole and the
        coordinates Eq. (22) reads pay for the excess alone.  This is not
        ``energy_tolerance``: that one decides whether a repair runs at all
        and leaves the set it then meets an equality, while the band changes
        the set.  It leaves alone the propagated velocity, the two cones and
        their declared floors, the eigenvalue clip, the affine Gram set and
        its range basis, the least-squares map back and its metric, the
        number of Dykstra iterations and their correction bookkeeping, the
        declared reference itself and the tolerance at which a repair fires,
        and the lift re-selection.  The slab is the intersection of the
        hyperplane's own two half-spaces, so it is met along the same
        direction the equality is --- the hyperplane's normal in the
        realified feature space, or the declared direction when
        ``raw_coordinate_motion_scales`` names one --- and the internal
        energy the repaired coordinates carry is then on the declared
        reference to the band rather than to the arithmetic.
    ``state_motion_scale`` and ``frontier_motion_scale``
        Relative freedom of the 44 state coordinates and the 62 standardized
        frontier coordinates in the least-squares map back.  The repair takes
        the minimum-norm solution in the rescaled parameters, so a larger
        scale lets that group absorb more of the correction.  ``1.0`` and
        ``1.0`` is plain minimum-norm least squares.  These are inert
        whenever the null space of the affine Gram map lies entirely inside
        one group, which is the case for this dictionary: the 15 directions
        the 24-row Gram cannot see are all propagated-state directions.
    ``raw_coordinate_motion_scales``
        A declared per-coordinate scale in the same least-squares map back,
        as ``((name, scale), ...)`` naming raw moment coordinates of
        :data:`RAW_MOMENT_COORDINATE_NAMES`.  Each declared factor
        multiplies ``state_motion_scale`` on that one coordinate, so a
        factor below one makes that coordinate expensive to move and the
        weighted minimum-norm map back places the same corrected Gram on the
        other coordinates instead.  The empty default is the plain per-group
        scaling of every earlier pass and changes no number.  Unlike the two
        group scales this is not inert: the directions the affine Gram map
        cannot see lie inside the propagated-state group, so a weight that
        distinguishes coordinates *within* that group selects a different
        member of the same affine preimage.  Where the map is injective
        -- 126 columns of rank 126 at ``target="both"`` on the 27-key
        dictionary -- there is no other member to choose and the map back
        returns the same correction for every positive weight.

        The declared metric acts in one further place, which is the one that
        reaches the injective case: the direction inside the affine Gram set
        along which the Eq. (22) energy hyperplane is met.  The committed
        repair meets it along the hyperplane's own normal in the realified
        feature space, which is the smallest correction measured in the
        Frobenius metric of the two moment matrices; with a declared metric
        it is met along the smallest correction measured in that metric on
        the propagated coordinates instead.  For an empty declaration that
        would be the plain coordinate norm, which is not the committed
        behaviour either, so this -- like the weight itself -- is off until
        the declaration is non-empty.

        What it leaves alone: the propagated velocity, the cones and their
        declared floors, the eigenvalue clip that projects onto them, the
        affine Gram set and its orthogonal projection, the number of Dykstra
        iterations and their bookkeeping, the declared Eq. (22) reference and
        the tolerance at which the repair fires.  The energy of the repaired
        coordinates still lands on that same reference; only the direction it
        is put there along, and hence which coordinates pay for it, changes.
        Because the cone is still projected onto in the Frobenius metric
        while the affine sets are met in the declared one, the alternation is
        no longer a single-metric Dykstra iteration, and how often it reaches
        the cones is a measured quantity rather than a guaranteed one.
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
    raw_coordinate_motion_scales: tuple[tuple[str, float], ...] = ()
    energy_reference_mode: str = "none"
    energy_tolerance: float = 0.0
    energy_reference_band: float = 0.0

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
        declared = self.raw_coordinate_motion_scales
        object.__setattr__(
            self,
            "raw_coordinate_motion_scales",
            tuple(
                (str(name), float(value))
                for name, value in (
                    declared.items()
                    if isinstance(declared, Mapping)
                    else declared
                )
            ),
        )
        named: set[str] = set()
        for name, value in self.raw_coordinate_motion_scales:
            if name not in RAW_MOMENT_COORDINATE_NAMES:
                raise ValueError(
                    "raw_coordinate_motion_scales must name raw moment "
                    f"coordinates; {name} is not one of them"
                )
            if name in named:
                raise ValueError(
                    f"raw_coordinate_motion_scales names {name} twice"
                )
            named.add(name)
            if value <= 0.0:
                raise ValueError(
                    "every raw coordinate motion scale must be positive"
                )
        if self.energy_reference_mode not in CHEAP_PROJECTION_ENERGY_REFERENCES:
            raise ValueError(
                "energy_reference_mode must be one of "
                + ", ".join(CHEAP_PROJECTION_ENERGY_REFERENCES)
            )
        if self.energy_reference_mode != "none":
            if self.policy != "dykstra":
                raise ValueError(
                    "energy_reference_mode is an option of the dykstra "
                    "repair, which alternates between the cones and the "
                    "affine sets; no single clip can meet a hyperplane"
                )
            if self.repair_coordinates != "state_and_frontier":
                raise ValueError(
                    "energy_reference_mode needs the propagated coordinates: "
                    "Eq. (22) does not read the frontier, so frontier_only "
                    "cannot move the internal energy"
                )
        if self.energy_tolerance < 0.0:
            raise ValueError("energy_tolerance must be nonnegative")
        if self.energy_reference_band < 0.0:
            raise ValueError("energy_reference_band must be nonnegative")
        if (
            self.energy_reference_band > 0.0
            and self.energy_reference_mode == "none"
        ):
            raise ValueError(
                "energy_reference_band is the half-width of a band around "
                "the Eq. (22) reference; without an energy_reference_mode "
                "there is no reference to band"
            )

    @property
    def holds_internal_energy(self) -> bool:
        """Whether the repair carries the Eq. (22) energy hyperplane."""

        return self.energy_reference_mode != "none"

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
class LiftReselectionRule:
    """Declared rule that turns one completion solve into the carried lift.

    Every default is the route's earlier form: a certified lift replaces the
    carried one whole and a refusal keeps the carried lift.  The declared
    completion exposes no proximal centre -- its proximal term is
    ``0.5 |z|^2`` about the zero standardized frontier, and its warm
    frontier is a starting point of the solve, not a term of the objective
    -- so every field here acts on the completion's answer and on the
    carried lift, both in the standardized frontier coordinates.

    ``anchor_weight``
        Proximal anchor at the carried lift, ``mu >= 0``: the carried lift
        moves the fraction ``1 / (1 + mu)`` of the way to the accepted lift.
        That point minimizes ``0.5 |z - z*|^2 + (mu / 2) |z - z_c|^2``,
        which is the anchored objective under the identity-Hessian model of
        the declared objective (whose Hessian is the identity plus the
        log-determinant term's), and it lies on the segment between two
        lifts, so it is inside the positive cone and the frontier box
        whenever both ends are.  ``0`` is off.
    ``trust_radius``
        Declared cap on the Euclidean move of the standardized frontier in
        one re-selection.  ``None`` is off.
    ``trust_policy``
        What happens when the cap binds: ``"carried"`` keeps the carried
        lift; ``"ball"`` takes the point at the cap on the segment toward
        the accepted lift.
    ``acceptance_margin``
        Deterministic acceptance: a candidate is accepted only when the
        declared objective at it lies below the declared objective at the
        carried lift by more than this margin.  Both values are the
        completion's own functional,
        ``0.5 |z|^2 - (w / n) sum log(max(lambda, 0) + s)`` on the scaled
        24-row matrix, evaluated here by the same formula on both lifts, so
        the rule cannot flip on a change of its inputs smaller than the
        margin divided by the functional's Lipschitz constant.  ``None`` is
        off.
    ``candidates``
        Which solves are candidates: ``"certified"`` accepts the
        completion's own certificate, the earlier form; ``"measured"``
        takes any returned iterate whose scaled minimum eigenvalue is at or
        above ``-10`` times the conic tolerance and whose standardized
        coordinates lie in the frontier box to the same tolerance (the
        completion's own feasibility test of a warm frontier) and whose
        declared objective improves on the carried lift's by more than
        ``max(acceptance_margin, 0)``; the certificate is recorded and not
        consulted.
    """

    anchor_weight: float = 0.0
    trust_radius: float | None = None
    trust_policy: str = "carried"
    acceptance_margin: float | None = None
    candidates: str = "certified"

    def __post_init__(self) -> None:
        if not np.isfinite(self.anchor_weight) or self.anchor_weight < 0.0:
            raise ValueError("anchor_weight must be finite and nonnegative")
        if self.trust_radius is not None and not (
            np.isfinite(self.trust_radius) and self.trust_radius > 0.0
        ):
            raise ValueError("trust_radius must be positive or None")
        if self.trust_policy not in LIFT_RESELECTION_TRUST_POLICIES:
            raise ValueError(
                "trust_policy must be one of "
                + ", ".join(LIFT_RESELECTION_TRUST_POLICIES)
            )
        if self.acceptance_margin is not None and not (
            np.isfinite(self.acceptance_margin)
            and self.acceptance_margin >= 0.0
        ):
            raise ValueError("acceptance_margin must be nonnegative or None")
        if self.candidates not in LIFT_RESELECTION_CANDIDATES:
            raise ValueError(
                "candidates must be one of "
                + ", ".join(LIFT_RESELECTION_CANDIDATES)
            )

    @property
    def is_earlier_form(self) -> bool:
        """True when every field is off: the certified lift, taken whole."""

        return (
            self.anchor_weight == 0.0
            and self.trust_radius is None
            and self.acceptance_margin is None
            and self.candidates == "certified"
        )

    @property
    def evaluates_objective(self) -> bool:
        return (
            self.acceptance_margin is not None or self.candidates == "measured"
        )


@dataclass(frozen=True)
class LiftReselectionDecision:
    """What one lift re-selection did, in the standardized frontier.

    ``decision`` is one of ``"accepted"`` (the lift moved to the candidate,
    anchored when the anchor is on), ``"refused"`` (no certificate, the
    carried lift kept), ``"infeasible"`` (``candidates="measured"`` and the
    returned iterate fails the measured feasibility test),
    ``"declined_by_margin"`` (a candidate whose objective improvement does
    not exceed the margin), ``"cap_bound_carried"`` (the move exceeded the
    trust radius and the carried lift was kept), ``"cap_bound_ball"`` (the
    move exceeded the trust radius and was scaled to it, then anchored) or
    ``"no_frontier"`` (the completion returned no finite frontier).
    ``solved_displacement`` is the Euclidean distance from the carried lift
    to the completion's own answer, whatever was done with it;
    ``applied_displacement`` is the distance the carried lift actually
    moved.  The objectives are the declared completion's functional at the
    carried lift and at the completion's answer, ``nan`` when the rule did
    not evaluate them and evaluation was not requested.
    """

    frontier_standardized: FloatArray
    completion: APCMExtensionResult
    seconds: float
    certified: bool
    accepted: bool
    decision: str
    solved_displacement: float
    applied_displacement: float
    objective_carried: float
    objective_solved: float

    @property
    def improvement(self) -> float:
        return float(self.objective_carried - self.objective_solved)


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
    internal_energy_before: float = float("nan")
    internal_energy_after: float = float("nan")
    internal_energy_reference: float = float("nan")
    energy_hyperplane_applied: bool = False


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


class InternalEnergyFunctional:
    """The Eq. (22) internal energy as an affine functional of the state.

    The value is ``matrix_total_energy`` evaluated on exactly the chain the
    driver scores: the 29 raw moment coordinates to a matrix state, that
    state to the 31 closed scalar coordinates, and those back to a matrix
    state.  That chain is affine in the 29 raw coordinates, which this class
    rebuilds by evaluating it at the coordinate basis and verifies at a
    random probe, or construction fails.

    Neither the 15 entrance coordinates nor the 62 frontier coordinates
    appear in Eq. (22), so the gradient vanishes on them.  Within the raw
    coordinates the gradient vanishes on the six anomalous ones and equals
    ``omega_ph`` on each diagonal raw normal coordinate, which is the
    statement that a correction confined to the phonon normal and anomalous
    moments changes the energy by ``omega_ph`` times the trace of the normal
    correction.
    """

    def __init__(
        self, parameters: DimerParameters, *, state_count: int
    ) -> None:
        if state_count < _RAW_DIMENSION:
            raise ValueError(
                "state_count must cover the 29 raw moment coordinates"
            )
        self.parameters = parameters
        self.state_count = int(state_count)
        zero = np.zeros(_RAW_DIMENSION, dtype=float)
        offset = self._raw_energy(zero)
        gradient = np.zeros(self.state_count, dtype=float)
        for column in range(_RAW_DIMENSION):
            basis = zero.copy()
            basis[column] = 1.0
            gradient[column] = self._raw_energy(basis) - offset
        self.offset = float(offset)
        self.gradient = gradient
        generator = np.random.default_rng(20260915)
        for _ in range(4):
            probe = generator.normal(scale=0.3, size=self.state_count)
            predicted = self.value(probe)
            direct = self._raw_energy(probe[:_RAW_DIMENSION])
            if not np.isclose(predicted, direct, atol=1e-12, rtol=0.0):
                raise RuntimeError(
                    "the Eq. (22) internal energy is not affine in the "
                    "propagated coordinates"
                )

    def _raw_energy(self, raw: FloatArray) -> float:
        state = raw_moment_coordinates_to_matrix_state(
            np.asarray(raw, dtype=float)
        )
        closed = matrix_state_to_closed_scalar_coordinates(state)
        return float(
            matrix_total_energy(
                closed_scalar_to_matrix_state(closed), self.parameters
            )
        )

    def value(self, state: FloatArray) -> float:
        """The Eq. (22) internal energy at these propagated coordinates."""

        return float(
            self.offset
            + self.gradient @ np.asarray(state, dtype=float)[: self.state_count]
        )


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
        *,
        parameters: DimerParameters | None = None,
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
            tuple[Any, ...], tuple[FloatArray, FloatArray, FloatArray]
        ] = {}
        self.energy = (
            None
            if parameters is None
            else InternalEnergyFunctional(
                parameters, state_count=self.state_count
            )
        )
        self._energy_functionals: dict[tuple[Any, ...], FloatArray] = {}
        self._energy_directions: dict[tuple[Any, ...], FloatArray] = {}

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

    def state_coordinate_scales(
        self, settings: CheapProjectionSettings
    ) -> FloatArray:
        """The declared freedom of each propagated coordinate, one entry each.

        ``state_motion_scale`` on all of them, multiplied on every raw moment
        coordinate the settings name by its declared factor.  The empty
        declaration returns the plain group vector of every earlier pass.
        """

        scales = np.full(self.state_count, float(settings.state_motion_scale))
        for name, value in settings.raw_coordinate_motion_scales:
            scales[RAW_MOMENT_COORDINATE_NAMES.index(name)] *= float(value)
        return scales

    def factorization(
        self, settings: CheapProjectionSettings
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Return the factorization of one declared repair, built once.

        ``pseudo_inverse`` maps a realified target residual to the weighted
        minimum-norm increment of the movable coordinates; ``range_basis``
        is an orthonormal basis of the affine Gram set's tangent space under
        those same movable coordinates, which the Dykstra policy projects
        onto; ``scales`` is the declared freedom of each movable
        coordinate, the two group constants with every declared
        per-coordinate factor already applied; ``movable`` is the index set
        of coordinates the map back may move.
        """

        signature = (
            settings.target,
            settings.repair_coordinates,
            float(settings.state_motion_scale),
            float(settings.frontier_motion_scale),
            settings.raw_coordinate_motion_scales,
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
                self.state_coordinate_scales(settings),
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

    # -- the energy hyperplane ---------------------------------------------

    def energy_feature_functional(
        self, settings: CheapProjectionSettings
    ) -> FloatArray:
        """Eq. (22) as a linear functional of the realified target features.

        The map back adds ``scales * (pseudo_inverse @ residual)`` to the
        movable coordinates, so the internal energy the repaired coordinates
        carry is the energy before the repair plus ``w . residual`` with
        ``w = pseudo_inverse^T (scales * gradient)``.  That ``w`` is the
        normal of the hyperplane on which the internal energy equals a
        declared reference, expressed in the same realified feature space as
        the positive semidefinite cones and the affine Gram set, so all three
        sets live in one space and one Dykstra alternation can meet them.
        """

        if self.energy is None:
            raise ValueError(
                "the internal-energy hyperplane needs the model parameters: "
                "build AffineGramCoordinates with parameters="
            )
        signature = (
            settings.target,
            settings.repair_coordinates,
            float(settings.state_motion_scale),
            float(settings.frontier_motion_scale),
            settings.raw_coordinate_motion_scales,
        )
        cached = self._energy_functionals.get(signature)
        if cached is not None:
            return cached
        pseudo_inverse, _, scales, movable = self.factorization(settings)
        gradient = np.concatenate(
            (self.energy.gradient, np.zeros(self.frontier_count, dtype=float))
        )
        value = pseudo_inverse.T @ (scales * gradient[movable])
        self._energy_functionals[signature] = value
        return value

    def energy_correction_direction(
        self, settings: CheapProjectionSettings
    ) -> FloatArray | None:
        """The declared direction the Eq. (22) hyperplane is met along.

        ``None`` when no per-coordinate metric is declared, which is the
        committed repair: the hyperplane is then met along its own normal in
        the realified feature space.  With a declared metric the correction
        is the one of smallest weighted norm on the propagated coordinates
        that changes the internal energy, ``scales ** 2 * gradient``, carried
        into the realified feature space by the same affine Gram map so that
        it lies in the affine Gram set like every other step of the
        alternation, and reduced onto the range basis the alternation works
        in.
        """

        if not settings.raw_coordinate_motion_scales:
            return None
        if self.energy is None:
            raise ValueError(
                "the internal-energy hyperplane needs the model parameters: "
                "build AffineGramCoordinates with parameters="
            )
        signature = (
            settings.target,
            settings.repair_coordinates,
            float(settings.state_motion_scale),
            float(settings.frontier_motion_scale),
            settings.raw_coordinate_motion_scales,
        )
        cached = self._energy_directions.get(signature)
        if cached is not None:
            return cached
        _, range_basis, scales, movable = self.factorization(settings)
        gradient = np.concatenate(
            (self.energy.gradient, np.zeros(self.frontier_count, dtype=float))
        )
        design = self._design(settings.target)[:, movable]
        value = range_basis.T @ (design @ (scales**2 * gradient[movable]))
        self._energy_directions[signature] = value
        return value

    # -- the repair itself -------------------------------------------------

    def repair(
        self,
        state: FloatArray,
        frontier_standardized: FloatArray,
        settings: CheapProjectionSettings,
        *,
        energy_reference: float | None = None,
    ) -> CheapProjectionRepair:
        """Apply the declared cheap repair to one pair of coordinates.

        ``energy_reference`` is the declared value of the Eq. (22) internal
        energy the repair must land on; it is honoured only when the settings
        name an energy reference mode, and ``None`` means the constraint is
        off at this opportunity, which is what the post-pulse mode does while
        the drive is still on.
        """

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
        energy_before = (
            float("nan")
            if self.energy is None
            else self.energy.value(current_state)
        )
        holds_energy = (
            settings.holds_internal_energy and energy_reference is not None
        )
        reference = (
            float(energy_reference) if holds_energy else float("nan")
        )
        names = self.block_names(settings.target)
        floors = settings.block_floors(names)

        def unchanged(applied_hyperplane: bool = False) -> CheapProjectionRepair:
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
                internal_energy_before=energy_before,
                internal_energy_after=energy_before,
                internal_energy_reference=reference,
                energy_hyperplane_applied=applied_hyperplane,
            )

        if settings.policy == "none":
            return unchanged()
        features = self._target_features(
            settings.target, current_state, current_frontier
        )
        minima = self._block_minima(features, settings.target)
        cones_met = all(
            value >= floor - settings.repair_tolerance
            for value, floor in zip(minima, floors, strict=True)
        )
        energy_met = not holds_energy or (
            abs(energy_before - reference) <= settings.energy_tolerance
        )
        if cones_met and energy_met:
            return unchanged()
        pseudo_inverse, range_basis, scales, movable = self.factorization(
            settings
        )
        hyperplane: tuple[FloatArray, float] | None = None
        if holds_energy:
            normal = self.energy_feature_functional(settings)
            projected_normal = range_basis.T @ normal
            if float(projected_normal @ projected_normal) > 1e-24:
                hyperplane = (normal, reference - energy_before)
        if settings.policy == "dykstra":
            corrected, iterations = self._dykstra(
                features,
                range_basis,
                settings.target,
                floors,
                settings.dykstra_iterations,
                hyperplane=hyperplane,
                hyperplane_direction=(
                    None
                    if hyperplane is None
                    else self.energy_correction_direction(settings)
                ),
                hyperplane_band=float(settings.energy_reference_band),
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
            internal_energy_before=energy_before,
            internal_energy_after=(
                float("nan")
                if self.energy is None
                else self.energy.value(new_state)
            ),
            internal_energy_reference=reference,
            energy_hyperplane_applied=hyperplane is not None,
        )

    def _dykstra(
        self,
        features: FloatArray,
        range_basis: FloatArray,
        target: str,
        floors: tuple[float, ...],
        iterations: int,
        *,
        hyperplane: tuple[FloatArray, float] | None = None,
        hyperplane_direction: FloatArray | None = None,
        hyperplane_band: float = 0.0,
    ) -> tuple[FloatArray, int]:
        """Alternate between the affine Gram set and the cone, with memory.

        The affine Gram set is the image of the affine coordinate map; its
        orthogonal projection in the realified metric is the projection of
        the displacement from the current point onto the range basis, which
        is the same subspace for every declared motion scale.

        ``hyperplane`` adds the third set of the energy-neutral option: the
        points whose mapped-back coordinates carry a declared Eq. (22)
        internal energy, as ``(normal, offset)`` with the constraint
        ``normal . (point - features) = offset``.  Because that normal lies
        in the span of the range basis, the affine Gram set and the
        hyperplane are projected onto together in closed form, which is the
        limit the alternation between those two affine sets converges to and
        leaves the cone as the only set the iteration has to chase.

        ``hyperplane_direction`` is the declared direction, in the reduced
        coordinates of the range basis, along which that offset is taken up.
        ``None`` is the committed repair, which takes it up along the
        hyperplane's own normal and is therefore the smallest correction in
        the realified feature metric.  A declared direction still meets the
        constraint exactly, because the energy a correction carries is the
        normal's pairing with it and the step divides by exactly that
        pairing; a direction the normal is orthogonal to carries no energy
        at all and is refused in favour of the normal itself.

        ``hyperplane_band`` is the declared half-width of a band around the
        reference, which makes the third set the slab
        ``|E - reference| <= band`` instead of the hyperplane
        ``E = reference``.  Zero is the declared equality and runs the
        committed arithmetic unchanged.  A positive band soft-thresholds the
        offset the step takes up: an iterate already inside the band is left
        exactly where the cone projection put it, and one outside is moved to
        the nearer boundary, so the alternation spends nothing on an energy
        change it is allowed to keep.  The slab is the intersection of the
        hyperplane's own two half-spaces, so the projection onto it is still
        along the same direction and the affine Gram set is still met in the
        same closed form.
        """

        projected_normal = None
        correction_direction = None
        if hyperplane is not None:
            projected_normal = range_basis.T @ hyperplane[0]
            normal_square = float(projected_normal @ projected_normal)
            correction_direction = projected_normal
            if hyperplane_direction is not None:
                pairing = float(projected_normal @ hyperplane_direction)
                reference_scale = float(
                    np.linalg.norm(projected_normal)
                    * np.linalg.norm(hyperplane_direction)
                )
                if abs(pairing) > 1e-10 * reference_scale:
                    correction_direction = hyperplane_direction
                    normal_square = pairing
        point = features.copy()
        cone_correction = np.zeros_like(features)
        affine_correction = np.zeros_like(features)
        for index in range(iterations):
            shifted = point + cone_correction
            projected = self._cone_projection(shifted, target, floors)
            cone_correction = shifted - projected
            shifted = projected + affine_correction
            displacement = shifted - features
            reduced = range_basis.T @ displacement
            if projected_normal is not None:
                offset = hyperplane[1] - float(projected_normal @ reduced)
                if hyperplane_band > 0.0:
                    offset = (
                        0.0
                        if abs(offset) <= hyperplane_band
                        else offset - copysign(hyperplane_band, offset)
                    )
                reduced = reduced + correction_direction * (
                    offset / normal_square
                )
            affine_point = features + range_basis @ reduced
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
    internal_energy: FloatArray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    internal_energy_reference: FloatArray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    energy_reference_value: float = float("nan")
    energy_reference_time: float = float("nan")
    energy_constrained_repairs: int = 0
    energy_hyperplane_refusals: int = 0
    maximum_absolute_energy_deviation: float = 0.0
    energy_reference_supplied: bool = False
    lift_reselection_retries_attempted: int = 0
    lift_reselection_retries_accepted: int = 0
    lift_reselection_retries_exhausted: int = 0
    lift_reselection_retry_solves: int = 0
    lift_reselection_retry_seconds: float = 0.0
    lift_reselections_accepted: int = 0
    lift_reselections_accepted_uncertified: int = 0
    lift_reselections_declined_by_margin: int = 0
    lift_reselections_cap_bound: int = 0
    maximum_lift_reselection_solved_displacement: float = 0.0


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

    ``completion_retry_shifts`` and ``completion_retry_weights`` are the
    record's declared completion retry ladder, built here by the record's
    own ``build_completion_ladder`` and walked only where a re-selection's
    declared completion is not certified.  Both empty is the route's earlier
    form and changes nothing.

    ``reselection_extension`` is the completion a lift re-selection solves,
    when it is to differ from the model's own extension in a declared
    numerical setting (the sensitivity study moves its conic tolerance one
    decade); it must carry the same dictionary, lower keys and frontier keys,
    so only the solver's settings can differ.  ``None`` is the route's
    earlier form: re-selections solve ``model.extension`` itself.

    ``reselection_rule`` is the declared :class:`LiftReselectionRule` that
    turns a completion's answer into the carried lift (proximal anchor,
    trust radius, acceptance margin, candidate scope); ``None`` is the
    earlier form, every field off.
    """

    def __init__(
        self,
        model: FixedDictionaryProjectedAPCM,
        settings: CheapProjectionSettings | None = None,
        *,
        coordinates: AffineGramCoordinates | None = None,
        completion_retry_shifts: tuple[float, ...] = (),
        completion_retry_weights: tuple[float, ...] = (),
        reselection_extension: SymmetryReducedPositiveExtension | None = None,
        reselection_rule: LiftReselectionRule | None = None,
    ) -> None:
        self.model = model
        self.reselection_rule = (
            LiftReselectionRule()
            if reselection_rule is None
            else reselection_rule
        )
        self.settings = (
            CheapProjectionSettings() if settings is None else settings
        )
        self.coordinates = (
            AffineGramCoordinates(
                model.extension, parameters=model.parameters
            )
            if coordinates is None
            else coordinates
        )
        self.reselection_extension = (
            model.extension
            if reselection_extension is None
            else reselection_extension
        )
        if (
            tuple(self.reselection_extension.active_keys)
            != tuple(model.extension.active_keys)
            or tuple(self.reselection_extension.lower_keys)
            != tuple(model.extension.lower_keys)
            or tuple(self.reselection_extension.frontier_keys)
            != tuple(model.extension.frontier_keys)
        ):
            raise ValueError(
                "the re-selection completion must carry the model's "
                "dictionary, lower keys and frontier keys; only its solver "
                "settings may differ"
            )
        self.completion_retry_shifts = tuple(
            float(value) for value in completion_retry_shifts
        )
        self.completion_retry_weights = tuple(
            float(value) for value in completion_retry_weights
        )
        self.completion_ladder = build_completion_ladder(
            self.reselection_extension,
            shifts=self.completion_retry_shifts,
            weights=self.completion_retry_weights,
        )
        self.completion_retry_counters: dict[str, int] = {
            "attempted": 0,
            "accepted": 0,
            "exhausted": 0,
            "solves": 0,
        }
        self.completion_retry_seconds = 0.0
        if self.coordinates.extension is not model.extension:
            raise ValueError(
                "affine coordinates must be built on the model's extension"
            )
        if (
            self.settings.holds_internal_energy
            and self.coordinates.energy is None
        ):
            raise ValueError(
                "an energy reference mode needs affine coordinates built "
                "with the model parameters"
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

    def declared_objective(
        self,
        lower_moments: Mapping[MomentKey, float],
        frontier_standardized: FloatArray,
    ) -> float:
        """The completion's own objective at one lift.

        ``0.5 |z|^2 - (w / n) sum log(max(lambda, 0) + s)`` in the
        standardized frontier ``z``, with ``lambda`` the eigenvalues of the
        scaled 24-row matrix the re-selection completion builds from these
        lower moments and this lift, ``w`` its log-determinant weight, ``s``
        its shift and ``n`` its dimension: the same formula
        ``APCMExtensionResult.objective`` reports for a solve, evaluated here
        so the carried lift and the solved lift are compared by one
        function.
        """

        extension = self.reselection_extension
        values = np.asarray(frontier_standardized, dtype=float)
        matrix = extension.scaled_matrix(
            extension.matrix(
                lower_moments, self.coordinates.frontier_mapping(values)
            )
        )
        shifted = (
            np.maximum(np.linalg.eigvalsh(matrix), 0.0)
            + extension.settings.logdet_shift
        )
        return float(
            0.5 * float(values @ values)
            - extension.settings.logdet_weight
            / extension.dimension
            * float(np.sum(np.log(shifted)))
        )

    def _solved_frontier(
        self, result: APCMExtensionResult
    ) -> FloatArray | None:
        """The completion's answer in standardized coordinates, or None."""

        keys = self.coordinates.frontier_keys
        if any(key not in result.frontier_moments for key in keys):
            return None
        values = self.coordinates.standardize_frontier(result.frontier_moments)
        if not np.all(np.isfinite(values)):
            return None
        return values

    def reselect_lift_decision(
        self,
        state: FloatArray,
        frontier_standardized: FloatArray,
        *,
        warm_start: bool = True,
        retries: bool = True,
        evaluate_objective: bool = False,
    ) -> LiftReselectionDecision:
        """Solve the declared completion and apply the declared rule.

        The solve is :meth:`reselect_lift`'s, ladder included; the rule is
        the route's :class:`LiftReselectionRule`.  With every field of the
        rule off the returned frontier is the certified solve's answer
        exactly, or the carried lift on a refusal, which is the earlier
        form.  ``evaluate_objective`` computes the declared objective at the
        carried and solved lifts even when the rule does not need them, for
        the stage record.
        """

        rule = self.reselection_rule
        carried = np.asarray(frontier_standardized, dtype=float)
        raw, hidden = unpack_projected_apcm_state(
            state, active_keys=self.coordinates.active_keys
        )
        lower = state_lower_moments(raw, hidden, self.coordinates.active_keys)
        warm = (
            self.coordinates.frontier_mapping(carried) if warm_start else None
        )
        started = perf_counter()
        result = self.reselection_extension.complete(
            lower, warm_frontier=warm
        )
        if not result.success and retries and self.completion_ladder:
            ladder_started = perf_counter()
            self.completion_retry_counters["attempted"] += 1
            for extension in self.completion_ladder:
                self.completion_retry_counters["solves"] += 1
                retried = extension.complete(lower, warm_frontier=warm)
                if retried.success:
                    self.completion_retry_counters["accepted"] += 1
                    result = retried
                    break
            else:
                self.completion_retry_counters["exhausted"] += 1
            self.completion_retry_seconds += perf_counter() - ladder_started
        seconds = perf_counter() - started

        certified = bool(result.success)
        solved = self._solved_frontier(result)
        tolerance = float(self.reselection_extension.settings.conic_tolerance)
        candidate: FloatArray | None
        if solved is None:
            candidate, decision = None, "no_frontier"
        elif rule.candidates == "certified":
            candidate = solved if certified else None
            decision = "accepted" if certified else "refused"
        else:
            feasible = bool(
                float(result.scaled_minimum_eigenvalue) >= -10.0 * tolerance
                and float(np.max(np.abs(solved))) <= 1.0 + 10.0 * tolerance
            )
            candidate = solved if feasible else None
            decision = "accepted" if feasible else "infeasible"

        objective_carried = objective_solved = float("nan")
        if evaluate_objective or (
            rule.evaluates_objective and candidate is not None
        ):
            objective_carried = self.declared_objective(lower, carried)
            if solved is not None:
                objective_solved = self.declared_objective(lower, solved)
        if candidate is not None and rule.evaluates_objective:
            margin = (
                0.0
                if rule.acceptance_margin is None
                else rule.acceptance_margin
            )
            improvement = objective_carried - objective_solved
            if not improvement > margin:
                candidate, decision = None, "declined_by_margin"

        solved_displacement = (
            0.0 if solved is None else float(np.linalg.norm(solved - carried))
        )
        if candidate is None:
            selected = carried.copy()
        else:
            selected = candidate
            move = candidate - carried
            distance = float(np.linalg.norm(move))
            if rule.trust_radius is not None and distance > rule.trust_radius:
                if rule.trust_policy == "carried":
                    candidate, decision = None, "cap_bound_carried"
                    selected = carried.copy()
                else:
                    decision = "cap_bound_ball"
                    selected = carried + (rule.trust_radius / distance) * move
            if candidate is not None and rule.anchor_weight > 0.0:
                selected = carried + (selected - carried) / (
                    1.0 + rule.anchor_weight
                )
        return LiftReselectionDecision(
            frontier_standardized=selected,
            completion=result,
            seconds=seconds,
            certified=certified,
            accepted=candidate is not None,
            decision=decision,
            solved_displacement=solved_displacement,
            applied_displacement=float(np.linalg.norm(selected - carried)),
            objective_carried=objective_carried,
            objective_solved=objective_solved,
        )

    def reselect_lift(
        self,
        state: FloatArray,
        frontier_standardized: FloatArray,
        *,
        warm_start: bool = True,
        retries: bool = True,
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

        ``retries`` walks the declared completion retry ladder the route was
        built with -- the record's own ``build_completion_ladder``, one
        log-determinant shift or one log-determinant weight per entry,
        solved on the same lower moments from the same warm start, the first
        certified result taken -- when the declared completion is not
        certified.  An empty ladder is the route's earlier form: one solve,
        and the carried lift kept on a refusal.  The returned ``seconds`` is
        the whole wall of the re-selection, retries included.

        The returned frontier is what the route's declared
        :class:`LiftReselectionRule` makes of the solve; with the rule's
        every field off it is the certified answer exactly.
        """

        decision = self.reselect_lift_decision(
            state,
            frontier_standardized,
            warm_start=warm_start,
            retries=retries,
        )
        return (
            decision.frontier_standardized,
            decision.completion,
            decision.seconds,
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
    energy_reference_time: float = 4.0,
    energy_reference_value: float | None = None,
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

    ``energy_reference_value`` is the declared internal-energy reference a
    segment is resumed on.  A rollout that stops on its wall budget is
    continued by handing its last accepted state, its last carried frontier,
    its last time and this value to a second call, so the declared reference
    is the one the first segment selected rather than one re-declared at the
    resumed state; ``None`` selects the reference the mode declares below.

    ``energy_reference_time`` is the declared time the drive ends, which the
    settings' ``"post_pulse"`` energy reference mode uses: the constraint is
    off before it, and from the first accepted state at or after it the
    repair holds the Eq. (22) internal energy at the value that state
    carried.  The ``"drive_power"`` mode ignores this time and instead
    advances the reference from the start of the rollout with the same
    SSPRK(3,3) weights, using the power the drive delivers,
    ``gradient . (velocity at this time - velocity at t=0)``, which is exact
    because the Hamiltonian's only time dependence is the drive difference
    and that difference vanishes at ``t = 0``.
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

    energy = coordinates.energy
    mode = settings.energy_reference_mode
    if mode != "none" and energy is None:
        raise ValueError(
            "an energy reference mode needs affine coordinates built with "
            "the model parameters"
        )
    supplied_reference = energy_reference_value is not None
    if supplied_reference and mode == "none":
        raise ValueError(
            "energy_reference_value is the reference a resumed segment "
            "carries; it needs an energy_reference_mode"
        )

    times = float(initial_time) + np.linspace(0.0, span, intervals + 1)
    states = np.empty((intervals + 1, state.size), dtype=float)
    frontiers = np.empty((intervals + 1, frontier.size), dtype=float)
    retained_minima = np.empty(intervals + 1, dtype=float)
    extension_minima = np.empty(intervals + 1, dtype=float)
    envelope = np.empty(intervals + 1, dtype=float)
    energies = np.full(intervals + 1, float("nan"), dtype=float)
    references = np.full(intervals + 1, float("nan"), dtype=float)
    records: list[dict[str, Any]] = []
    reference: float | None = None

    retries_before = dict(route.completion_retry_counters)
    retry_seconds_before = float(route.completion_retry_seconds)
    counters = {
        "applied": 0,
        "skipped": 0,
        "reselections": 0,
        "reselection_failures": 0,
        "reselection_accepted": 0,
        "reselection_accepted_uncertified": 0,
        "reselection_declined": 0,
        "reselection_cap_bound": 0,
        "energy_constrained": 0,
        "energy_refusals": 0,
    }
    totals = {
        "repair_seconds": 0.0,
        "velocity_seconds": 0.0,
        "state_displacement": 0.0,
        "frontier_displacement": 0.0,
        "dykstra_iterations": 0,
        "reselection_seconds": 0.0,
        "reselection_displacement": 0.0,
        "reselection_solved_displacement": 0.0,
        "reselected_minimum": float("inf"),
        "energy_deviation": 0.0,
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
        if energy is not None:
            energies[index] = energy.value(current)

    def capture(index: int) -> None:
        """Declare the post-pulse reference at the first accepted state."""

        nonlocal reference
        if (
            mode == "post_pulse"
            and reference is None
            and times[index] >= energy_reference_time - 1e-9
        ):
            reference = float(energies[index])
        if reference is not None:
            references[index] = reference
            totals["energy_deviation"] = max(
                totals["energy_deviation"],
                abs(float(energies[index]) - reference),
            )

    def repair(
        current: FloatArray,
        lift: FloatArray,
        stage: str,
        when: float,
        stage_reference: float | None,
    ) -> tuple[FloatArray, FloatArray]:
        result = coordinates.repair(
            current, lift, settings, energy_reference=stage_reference
        )
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
            if stage_reference is not None:
                if result.energy_hyperplane_applied:
                    counters["energy_constrained"] += 1
                else:
                    counters["energy_refusals"] += 1
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
                    "internal_energy_before": result.internal_energy_before,
                    "internal_energy_after": result.internal_energy_after,
                    "internal_energy_reference": (
                        result.internal_energy_reference
                    ),
                    "energy_hyperplane_applied": bool(
                        result.energy_hyperplane_applied
                    ),
                }
            )
        return result.state, result.frontier_standardized

    def velocity(
        when: float, current: FloatArray, lift: FloatArray
    ) -> tuple[FloatArray, float]:
        started = perf_counter()
        value, _ = route.unguarded_velocity(when, current, lift)
        power = 0.0
        if mode == "drive_power":
            drive_free, _ = route.unguarded_velocity(0.0, current, lift)
            power = float(energy.gradient @ (value - drive_free))
        totals["velocity_seconds"] += perf_counter() - started
        return value, power

    sample(0, state, frontier)
    if supplied_reference:
        reference = float(energy_reference_value)
    elif mode == "drive_power":
        reference = float(energies[0])
    capture(0)
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
            first_velocity, first_power = velocity(when, state, frontier)
            evaluations += 1
            first_state = state + time_step * first_velocity
            first_frontier = frontier
            first_reference = (
                None
                if reference is None
                else (
                    reference + time_step * first_power
                    if mode == "drive_power"
                    else reference
                )
            )
            if stage_cadence:
                first_state, first_frontier = repair(
                    first_state,
                    first_frontier,
                    "stage_1",
                    when + time_step,
                    first_reference,
                )

            second_velocity, second_power = velocity(
                when + time_step, first_state, first_frontier
            )
            evaluations += 1
            second_state = 0.75 * state + 0.25 * (
                first_state + time_step * second_velocity
            )
            second_frontier = first_frontier
            second_reference = (
                None
                if reference is None
                else (
                    0.75 * reference
                    + 0.25 * (first_reference + time_step * second_power)
                    if mode == "drive_power"
                    else reference
                )
            )
            if stage_cadence:
                second_state, second_frontier = repair(
                    second_state,
                    second_frontier,
                    "stage_2",
                    when + 0.5 * time_step,
                    second_reference,
                )

            third_velocity, third_power = velocity(
                when + 0.5 * time_step, second_state, second_frontier
            )
            evaluations += 1
            final_state = (1.0 / 3.0) * state + (2.0 / 3.0) * (
                second_state + time_step * third_velocity
            )
            final_frontier = second_frontier
            final_reference = (
                None
                if reference is None
                else (
                    (1.0 / 3.0) * reference
                    + (2.0 / 3.0)
                    * (second_reference + time_step * third_power)
                    if mode == "drive_power"
                    else reference
                )
            )
            final_state, final_frontier = repair(
                final_state,
                final_frontier,
                "stage_3" if stage_cadence else "step",
                when + time_step,
                final_reference,
            )
            if (
                lift_reselection_interval > 0
                and (step + 1) % lift_reselection_interval == 0
            ):
                carried = final_frontier
                chosen = route.reselect_lift_decision(
                    final_state,
                    carried,
                    warm_start=lift_reselection_warm_start,
                    evaluate_objective=record_stages,
                )
                final_frontier = chosen.frontier_standardized
                completion = chosen.completion
                seconds = chosen.seconds
                totals["reselection_seconds"] += seconds
                counters["reselections"] += 1
                if not completion.success:
                    counters["reselection_failures"] += 1
                else:
                    totals["reselected_minimum"] = min(
                        totals["reselected_minimum"],
                        float(completion.scaled_minimum_eigenvalue),
                    )
                if chosen.accepted:
                    counters["reselection_accepted"] += 1
                    if not chosen.certified:
                        counters["reselection_accepted_uncertified"] += 1
                if chosen.decision == "declined_by_margin":
                    counters["reselection_declined"] += 1
                if chosen.decision in ("cap_bound_carried", "cap_bound_ball"):
                    counters["reselection_cap_bound"] += 1
                totals["reselection_displacement"] = max(
                    totals["reselection_displacement"],
                    float(np.linalg.norm(final_frontier - carried)),
                )
                totals["reselection_solved_displacement"] = max(
                    totals["reselection_solved_displacement"],
                    float(chosen.solved_displacement),
                )
                if record_stages:
                    records.append(
                        {
                            "stage": "lift_reselection",
                            "time": float(when + time_step),
                            "applied": bool(chosen.accepted),
                            "certified": bool(chosen.certified),
                            "decision": chosen.decision,
                            "iterations": int(completion.iterations),
                            "scaled_minimum_eigenvalue": float(
                                completion.scaled_minimum_eigenvalue
                            ),
                            "kkt_backward_error": float(
                                completion.kkt_backward_error
                            ),
                            "relative_face_support_dimension": int(
                                completion.relative_face_support_dimension
                            ),
                            "relative_face_direction_dimension": int(
                                completion.relative_face_direction_dimension
                            ),
                            "frontier_displacement": float(
                                np.linalg.norm(final_frontier - carried)
                            ),
                            "solved_displacement": float(
                                chosen.solved_displacement
                            ),
                            "objective_carried": float(
                                chosen.objective_carried
                            ),
                            "objective_solved": float(chosen.objective_solved),
                            "objective_reported": float(completion.objective),
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
        if mode == "drive_power" and final_reference is not None:
            reference = float(final_reference)
        sample(step + 1, state, frontier)
        capture(step + 1)
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
                f"energy={energies[step + 1]:.6f} "
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
        internal_energy=energies[:kept],
        internal_energy_reference=references[:kept],
        energy_reference_value=(
            float("nan") if reference is None else float(reference)
        ),
        energy_reference_time=(
            float("nan") if mode == "none" else float(energy_reference_time)
        ),
        energy_constrained_repairs=int(counters["energy_constrained"]),
        energy_hyperplane_refusals=int(counters["energy_refusals"]),
        maximum_absolute_energy_deviation=float(totals["energy_deviation"]),
        energy_reference_supplied=bool(supplied_reference),
        lift_reselection_retries_attempted=int(
            route.completion_retry_counters["attempted"]
            - retries_before["attempted"]
        ),
        lift_reselection_retries_accepted=int(
            route.completion_retry_counters["accepted"]
            - retries_before["accepted"]
        ),
        lift_reselection_retries_exhausted=int(
            route.completion_retry_counters["exhausted"]
            - retries_before["exhausted"]
        ),
        lift_reselection_retry_solves=int(
            route.completion_retry_counters["solves"]
            - retries_before["solves"]
        ),
        lift_reselection_retry_seconds=float(
            route.completion_retry_seconds - retry_seconds_before
        ),
        lift_reselections_accepted=int(counters["reselection_accepted"]),
        lift_reselections_accepted_uncertified=int(
            counters["reselection_accepted_uncertified"]
        ),
        lift_reselections_declined_by_margin=int(
            counters["reselection_declined"]
        ),
        lift_reselections_cap_bound=int(counters["reselection_cap_bound"]),
        maximum_lift_reselection_solved_displacement=float(
            totals["reselection_solved_displacement"]
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
    completion_retry_shifts: tuple[float, ...] = (),
    completion_retry_weights: tuple[float, ...] = (),
    reselection_extension_settings: Any = None,
    reselection_rule: LiftReselectionRule | None = None,
) -> tuple[CheapProjectionRoute, FloatArray, FloatArray, float]:
    """Build the route at the canonical correlated preparation.

    Returns the route, the initial 44-coordinate state, the initial
    standardized frontier coordinates, and the exact ground energy.  The
    initial lift is the declared proximal log-det completion at the
    preparation, which is the one conic solve on this route and is paid once
    at ``t = 0``; every later lift is carried and repaired.

    ``completion_retry_shifts`` and ``completion_retry_weights`` are handed
    to the route as its declared completion retry ladder, which is walked
    only where a lift re-selection is not certified.

    ``reselection_extension_settings`` builds a second extension on the same
    dictionary whose solver settings the lift re-selections use, leaving the
    preparation's completion at ``extension_settings``; ``None`` is the
    earlier form, one extension for both.

    ``reselection_rule`` is handed to the route as its declared
    :class:`LiftReselectionRule`; ``None`` is the earlier form.
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
    reselection_extension = (
        None
        if reselection_extension_settings is None
        else SymmetryReducedPositiveExtension(
            settings=reselection_extension_settings,
            active_keys=tuple(active_keys),
        )
    )
    route = CheapProjectionRoute(
        model,
        settings,
        completion_retry_shifts=tuple(completion_retry_shifts),
        completion_retry_weights=tuple(completion_retry_weights),
        reselection_extension=reselection_extension,
        reselection_rule=reselection_rule,
    )
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

        if settings.raw_coordinate_motion_scales:
            raise ValueError(
                "raw_coordinate_motion_scales names raw moment coordinates "
                "of the fixed-dictionary state; the carried-witness lane "
                "propagates its own 384 coordinates and carries no such "
                "declaration"
            )
        if settings.energy_reference_band > 0.0:
            raise ValueError(
                "energy_reference_band bands the Eq. (22) internal energy "
                "of the fixed-dictionary coordinates; the carried-witness "
                "lane's repair carries no such hyperplane to band"
            )
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
    "CHEAP_PROJECTION_ENERGY_REFERENCES",
    "CHEAP_PROJECTION_POLICIES",
    "CHEAP_PROJECTION_TARGETS",
    "CheapProjectionRepair",
    "CheapProjectionRoute",
    "CheapProjectionSettings",
    "CheapProjectionTrajectory",
    "InternalEnergyFunctional",
    "integrate_cheap_projection_ssprk3",
    "prepare_cheap_projection_route",
]
