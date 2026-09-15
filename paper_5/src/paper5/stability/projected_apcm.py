"""Faithful fixed-dictionary core of projected APCM dynamics.

The archive matrix equations remain the retained target at every call.  The
exact ``K/P/D`` entrance augments only their electron--phonon correlation
velocity, every active auxiliary moment evolves under its exact canonical
commutator using the selected positive frontier, and the common retained and
extended tangent cones select the propagated velocity.  No legacy finite-rate
barrier controller is used here.

This module intentionally implements the fixed active dictionary between
admission events.  Residual-driven dictionary enlargement is a separate layer;
the state, target, projection, and stage-containment semantics defined here are
the core that that layer must preserve.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from .adaptive_positive_moment import (
    ENTRANCE_RELATIVE_MOMENT_KEYS,
    HIDDEN_RELATIVE_MOMENT_KEYS,
    RAW_MOMENT_COORDINATE_NAMES,
    kpd_correlation_velocity_correction,
    matrix_derivative_to_raw_moment_velocity,
    matrix_state_to_raw_moment_coordinates,
    raw_moment_coordinates_to_matrix_state,
    uncentered_joint_moment_matrix,
)
from .apcm_moment_projection import (
    APCMProjectionError,
    APCMStageRetraction,
    APCMVelocityProjection,
    SymmetryReducedAPCMGeometry,
    state_lower_moments,
)
from .apcm_carried_witness import negative_mode_coupling
from .apcm_positive_extension import (
    APCMExtensionResult,
    SymmetryReducedPositiveExtension,
)
from .apcm_projective_guard import (
    InvariantTargetReadout,
    compile_invariant_target_readout,
    select_outer_feasible_projective_guard,
)
from .exact_reference import exact_holstein_joint_moment_initial_state
from .hubbard_dimer import DimerParameters, FloatArray
from .matrix_reference import MatrixDimerState, matrix_dimer_rhs
from .moment_hierarchy import (
    MomentKey,
    THIRD_ORDER_HIERARCHY,
    _commutator,
    _hamiltonian_terms,
)

_RAW_DIMENSION = len(RAW_MOMENT_COORDINATE_NAMES)

def build_completion_ladder(
    extension: SymmetryReducedPositiveExtension,
    *,
    shifts: tuple[float, ...] = (),
    weights: tuple[float, ...] = (),
) -> tuple[SymmetryReducedPositiveExtension, ...]:
    """Declared retries of the completion, in the order they are tried.

    Each entry is the same active dictionary with one declared setting
    changed: a log-determinant shift, which inflates the completion cone, or
    a log-determinant weight, which changes the balance between the proximal
    term and the barrier.  An empty ladder is the committed path.
    """

    ladder = []
    for shift in shifts:
        ladder.append(
            SymmetryReducedPositiveExtension(
                settings=replace(extension.settings, logdet_shift=float(shift)),
                active_keys=extension.active_keys,
                additional_halfword_keys=extension.additional_halfword_keys,
            )
        )
    for weight in weights:
        ladder.append(
            SymmetryReducedPositiveExtension(
                settings=replace(
                    extension.settings, logdet_weight=float(weight)
                ),
                active_keys=extension.active_keys,
                additional_halfword_keys=extension.additional_halfword_keys,
            )
        )
    return tuple(ladder)


def projected_apcm_state_names(
    active_keys: tuple[MomentKey, ...],
) -> tuple[str, ...]:
    """Return deterministic raw and active-moment coordinate names."""

    return (
        *RAW_MOMENT_COORDINATE_NAMES,
        *tuple(
        "hidden_"
        + key.spin_up.lower()
        + key.spin_down.lower()
        + f"_x{key.x_power}_p{key.p_power}"
            for key in active_keys
        ),
    )


PROJECTED_APCM_STATE_NAMES = projected_apcm_state_names(
    HIDDEN_RELATIVE_MOMENT_KEYS
)
ENTRANCE_PROJECTED_APCM_STATE_NAMES = projected_apcm_state_names(
    ENTRANCE_RELATIVE_MOMENT_KEYS
)


class ProjectedAPCMFailure(RuntimeError):
    """The declared autonomous APCM model could not advance physically."""


_STAGE_FAILURE_ATTRIBUTE = "projected_apcm_stage_failure"
_PARTIAL_TRAJECTORY_ATTRIBUTE = "projected_apcm_partial_trajectory"


@dataclass(frozen=True)
class ProjectedAPCMStageFailure:
    """Where a declared stage failure was raised, and what it was given.

    ``stage`` is the model operation that raised: ``"evaluate"`` for the
    right-hand-side evaluation at an accepted stage state, ``"contain"`` for
    the containment of a trial stage.  ``lower_moments`` are the lower
    moments of ``state`` in the declared extension ordering, which is what a
    diagnosis of the completion problem needs.
    """

    stage: str
    time: float
    state: FloatArray
    lower_moments: Mapping[MomentKey, float]
    warm_frontier_moments: Mapping[MomentKey, float] | None
    message: str


def _attach_stage_failure(
    error: BaseException,
    failure: ProjectedAPCMStageFailure,
) -> None:
    """Record the failing stage on the exception, innermost stage winning."""

    if getattr(error, _STAGE_FAILURE_ATTRIBUTE, None) is None:
        try:
            setattr(error, _STAGE_FAILURE_ATTRIBUTE, failure)
        except AttributeError:  # pragma: no cover - builtin exceptions allow it
            pass


def projected_apcm_stage_failure(
    error: BaseException,
) -> ProjectedAPCMStageFailure | None:
    """Return the failing stage recorded on ``error``, when there is one."""

    failure = getattr(error, _STAGE_FAILURE_ATTRIBUTE, None)
    return failure if isinstance(failure, ProjectedAPCMStageFailure) else None


def projected_apcm_partial_trajectory(
    error: BaseException,
) -> "ProjectedAPCMTrajectory | None":
    """Return the steps a failed rollout had accepted, when there are any."""

    partial = getattr(error, _PARTIAL_TRAJECTORY_ATTRIBUTE, None)
    return partial if isinstance(partial, ProjectedAPCMTrajectory) else None


def _attach_partial_trajectory(
    error: BaseException,
    trajectory: "ProjectedAPCMTrajectory",
) -> None:
    """Record the steps already accepted when a rollout stops on a failure."""

    if getattr(error, _PARTIAL_TRAJECTORY_ATTRIBUTE, None) is None:
        try:
            setattr(error, _PARTIAL_TRAJECTORY_ATTRIBUTE, trajectory)
        except AttributeError:  # pragma: no cover - builtin exceptions allow it
            pass


@dataclass(frozen=True)
class ProjectedAPCMSettings:
    """Fixed entrance switches for one APCM model configuration."""

    include_k: bool = True
    include_pauli: bool = True
    include_opposite_spin: bool = True


@dataclass(frozen=True)
class ProjectedAPCMStagePolicy:
    """Named, default-off ablations of the fixed-dictionary stage policy.

    Every field's default is the committed path, so the default instance
    reproduces it bitwise.  The fields name the three stage operations the
    accuracy attribution separates:

    ``stage_retraction``
        ``False`` keeps the declared proximal log-det completion at the trial
        point and the frozen moment-envelope validation, but never applies
        the conic retraction: the stage state is the trial state and the
        retraction is recorded as not applied.
    ``quiet_containment``
        ``False`` removes containment from a stage the outer guard did not
        trigger: neither the feasibility test, nor the envelope validation,
        nor the retraction runs, and the stage state is the trial state.
        The declared completion is still selected at the trial point,
        because the next right-hand-side evaluation needs a lift there.  A
        triggered stage is contained as declared.
    ``completion_start``
        ``"warm"`` is the committed path, in which the previous stage's
        frontier moments start the completion solve; ``"cold"`` starts it
        from the origin, which measures how much of the selected lift is
        determined by the start rather than by the lower moments.
    ``carried_completion``
        ``True`` carries the previous stage's completion coordinates to the
        trial point as the lift itself, without re-solving, whenever they
        remain feasible there; otherwise the declared completion runs and
        the fallback is counted.  This is the cheapest hybrid between the
        fixed dictionary's per-stage completion and the carried-witness
        flow's propagated completion coordinates.
    ``completion_retry_shifts`` and ``completion_retry_weights``
        A declared ladder of retries for a completion the declared settings
        do not certify: each entry is one log-determinant shift, or one
        log-determinant weight, solved on the same lower moments from the
        same warm start, and the first certified result is taken.  Both
        default to empty, which is the committed path: one solve, and a
        refusal if it is not certified.
    ``carried_completion_margin_multiple``
        How much interior the carried lift must keep before it is accepted,
        in units of the declared conic tolerance: the scaled extension
        minimum eigenvalue must be at or above ``multiple`` tolerances and
        every standardized coordinate must stay ``10 * multiple`` tolerances
        inside the frozen box envelope.  ``10.0`` is the declared margin;
        ``0.0`` accepts any carried lift that is positive semidefinite and
        inside the envelope.
    """

    stage_retraction: bool = True
    quiet_containment: bool = True
    completion_start: str = "warm"
    carried_completion: bool = False
    carried_completion_margin_multiple: float = 10.0
    completion_retry_shifts: tuple[float, ...] = ()
    completion_retry_weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.completion_start not in {"warm", "cold"}:
            raise ValueError("completion_start must be 'warm' or 'cold'")
        if self.carried_completion_margin_multiple < 0.0:
            raise ValueError(
                "carried_completion_margin_multiple must be nonnegative"
            )
        if any(value <= 0.0 for value in self.completion_retry_shifts):
            raise ValueError("completion_retry_shifts must be positive")
        if any(value <= 0.0 for value in self.completion_retry_weights):
            raise ValueError("completion_retry_weights must be positive")

    @property
    def is_default(self) -> bool:
        return (
            self.stage_retraction
            and self.quiet_containment
            and self.completion_start == "warm"
            and not self.carried_completion
            and not self.completion_retry_shifts
            and not self.completion_retry_weights
        )


@dataclass(frozen=True)
class ProjectedAPCMInitialization:
    """Preparation-conditioned reduced state and selected positive lift."""

    state: FloatArray
    active_keys: tuple[MomentKey, ...]
    completion: APCMExtensionResult
    exact_ground_energy: float
    phonon_cutoff: int


@dataclass(frozen=True)
class ProjectedAPCMTargets:
    """Archive, entrance-augmented, and hidden commutator targets."""

    archive_matrix_velocity: MatrixDimerState
    augmented_matrix_velocity: MatrixDimerState
    archive_retained_velocity: FloatArray
    augmented_retained_velocity: FloatArray
    auxiliary_velocity: FloatArray
    raw_kpd_correlation_velocity: np.ndarray
    applied_correlation_velocity_increment: np.ndarray
    completion: APCMExtensionResult


@dataclass(frozen=True)
class ProjectedAPCMEvaluation:
    """One projected autonomous right-hand-side evaluation."""

    derivative: FloatArray
    targets: ProjectedAPCMTargets
    projection: APCMVelocityProjection


@dataclass(frozen=True)
class ProjectedAPCMTrajectory:
    """A fixed-step SSPRK(3,3) trajectory of the fixed active dictionary."""

    times: FloatArray
    states: FloatArray
    active_keys: tuple[MomentKey, ...]
    base_minimum_eigenvalues: FloatArray
    extension_minimum_eigenvalues: FloatArray
    retained_correction_norms: FloatArray
    auxiliary_correction_norms: FloatArray
    stage_retractions: np.ndarray
    accepted_substeps: np.ndarray
    rhs_evaluations: int
    completed_steps: int
    success: bool
    message: str
    outer_guard_triggered_stages: np.ndarray | None = None
    stage_failure: "ProjectedAPCMStageFailure | None" = None


@dataclass(frozen=True)
class _AcceptedInterval:
    state: FloatArray
    completion: APCMExtensionResult
    retained_correction_norm: float
    auxiliary_correction_norm: float
    stage_retractions: int
    accepted_substeps: int
    rhs_evaluations: int


class _MomentMappingResolver:
    def __init__(self, moments: Mapping[MomentKey, float]) -> None:
        self._moments = moments

    def moment(self, key: MomentKey) -> float:
        if key.degree == 0:
            return 1.0
        try:
            return float(self._moments[key])
        except KeyError as error:
            raise ValueError(f"moment resolver does not contain {key}") from error


def pack_projected_apcm_state(
    raw_coordinates: FloatArray,
    hidden_moments: Mapping[MomentKey, float],
    *,
    active_keys: tuple[MomentKey, ...] = HIDDEN_RELATIVE_MOMENT_KEYS,
) -> FloatArray:
    """Pack the 29 retained coordinates and one active dictionary."""

    raw = np.asarray(raw_coordinates, dtype=float)
    if raw.shape != (_RAW_DIMENSION,):
        raise ValueError("raw coordinates must have shape (29,)")
    active = tuple(active_keys)
    missing = set(active).difference(hidden_moments)
    extra = set(hidden_moments).difference(active)
    if missing or extra:
        raise ValueError(
            f"hidden mapping mismatch: {len(missing)} missing, {len(extra)} extra"
        )
    return np.concatenate(
        (
            raw,
            np.asarray(
                [hidden_moments[key] for key in active],
                dtype=float,
            ),
        )
    )


def unpack_projected_apcm_state(
    state: FloatArray,
    *,
    active_keys: tuple[MomentKey, ...] = HIDDEN_RELATIVE_MOMENT_KEYS,
) -> tuple[FloatArray, FloatArray]:
    """Return retained coordinates and hidden values in frozen key order."""

    values = np.asarray(state, dtype=float)
    state_names = projected_apcm_state_names(active_keys)
    if values.shape != (len(state_names),):
        raise ValueError(
            "projected APCM state must have shape "
            f"{(len(state_names),)}, got {values.shape}"
        )
    return values[:_RAW_DIMENSION].copy(), values[_RAW_DIMENSION:].copy()


def _trace_project_correlation_velocity(
    derivative: MatrixDimerState,
) -> MatrixDimerState:
    correlation = np.asarray(
        derivative.electron_phonon_correlation,
        dtype=complex,
    ).copy()
    identity = np.eye(2, dtype=complex)
    for mode in range(correlation.shape[0]):
        correlation[mode] -= 0.5 * np.trace(correlation[mode]) * identity
    return MatrixDimerState(
        electron_density=np.asarray(derivative.electron_density, dtype=complex),
        coherent_phonon=np.asarray(derivative.coherent_phonon, dtype=complex),
        phonon_density=np.asarray(derivative.phonon_density, dtype=complex),
        anomalous_phonon_density=np.asarray(
            derivative.anomalous_phonon_density,
            dtype=complex,
        ),
        electron_phonon_correlation=correlation,
    )


def _with_correlation_increment(
    derivative: MatrixDimerState,
    increment: np.ndarray,
) -> MatrixDimerState:
    return _trace_project_correlation_velocity(
        MatrixDimerState(
            electron_density=derivative.electron_density,
            coherent_phonon=derivative.coherent_phonon,
            phonon_density=derivative.phonon_density,
            anomalous_phonon_density=derivative.anomalous_phonon_density,
            electron_phonon_correlation=(
                derivative.electron_phonon_correlation + increment
            ),
        )
    )


def _active_commutator_velocity(
    active_keys: tuple[MomentKey, ...],
    time: float,
    parameters: DimerParameters,
    completion: APCMExtensionResult | _MomentMappingResolver,
) -> FloatArray:
    """Evaluate exact canonical-CCR commutators for the active raw moments."""

    hamiltonian = _hamiltonian_terms(float(time), parameters)
    generated_cache: dict[MomentKey, float] = {}
    derivatives = np.empty(len(active_keys), dtype=float)
    for index, observable in enumerate(active_keys):
        derivative = 0.0j
        for hamiltonian_coefficient, hamiltonian_word in hamiltonian:
            for generated_key, commutator_coefficient in _commutator(
                hamiltonian_word,
                observable,
            ).items():
                if generated_key not in generated_cache:
                    generated_cache[generated_key] = completion.moment(
                        generated_key
                    )
                derivative += (
                    1j
                    * hamiltonian_coefficient
                    * commutator_coefficient
                    * generated_cache[generated_key]
                )
        if abs(derivative.imag) > 5e-11:
            raise FloatingPointError(
                "Hermitian active moment acquired a complex velocity: "
                f"{observable} -> {derivative}"
            )
        derivatives[index] = derivative.real
    return derivatives


class FixedDictionaryProjectedAPCM:
    """Archive-backed APCM target and physical moment-space projection."""

    def __init__(
        self,
        parameters: DimerParameters,
        *,
        extension: SymmetryReducedPositiveExtension | None = None,
        geometry: SymmetryReducedAPCMGeometry | None = None,
        settings: ProjectedAPCMSettings | None = None,
        stage_policy: ProjectedAPCMStagePolicy | None = None,
    ) -> None:
        self.parameters = parameters
        self.extension = (
            SymmetryReducedPositiveExtension(
                active_keys=ENTRANCE_RELATIVE_MOMENT_KEYS
            )
            if extension is None
            else extension
        )
        self.geometry = (
            SymmetryReducedAPCMGeometry(self.extension)
            if geometry is None
            else geometry
        )
        if self.geometry.extension is not self.extension:
            raise ValueError("geometry must use the selected positive extension")
        self.active_keys = tuple(self.extension.active_keys)
        self.state_names = projected_apcm_state_names(self.active_keys)
        self.settings = (
            ProjectedAPCMSettings() if settings is None else settings
        )
        self.stage_policy = (
            ProjectedAPCMStagePolicy()
            if stage_policy is None
            else stage_policy
        )
        self.stage_policy_counters: dict[str, int] = {
            "carried_completion_accepted": 0,
            "carried_completion_rejected": 0,
            "retraction_suppressed": 0,
            "quiet_containment_skipped": 0,
            "completion_retry_accepted": 0,
            "completion_retry_exhausted": 0,
        }
        self.completion_ladder = build_completion_ladder(
            self.extension,
            shifts=self.stage_policy.completion_retry_shifts,
            weights=self.stage_policy.completion_retry_weights,
        )
        self.geometry.completion_ladder = self.completion_ladder

    def _stage_failure_record(
        self,
        stage: str,
        time: float,
        state: FloatArray,
        warm_frontier: Mapping[MomentKey, float] | None,
        error: BaseException,
    ) -> ProjectedAPCMStageFailure:
        """Describe the stage a failure was raised at, for a failed rollout.

        The lower moments are recomputed here so that a rollout that stops
        can still be diagnosed at the point it stopped.  A failure of that
        recomputation must not replace the failure being reported, so the
        lower moments are left empty in that case.
        """

        try:
            raw, hidden = unpack_projected_apcm_state(
                state,
                active_keys=self.active_keys,
            )
            lower: Mapping[MomentKey, float] = MappingProxyType(
                {
                    key: float(value)
                    for key, value in state_lower_moments(
                        raw, hidden, self.active_keys
                    ).items()
                }
            )
        except Exception:  # pragma: no cover - diagnosis must not mask
            lower = MappingProxyType({})
        return ProjectedAPCMStageFailure(
            stage=stage,
            time=float(time),
            state=np.asarray(state, dtype=float).copy(),
            lower_moments=lower,
            warm_frontier_moments=(
                None
                if warm_frontier is None
                else MappingProxyType(
                    {key: float(value) for key, value in warm_frontier.items()}
                )
            ),
            message=f"{type(error).__name__}: {error}",
        )

    def _carried_completion(
        self,
        lower: Mapping[MomentKey, float],
        warm_frontier: Mapping[MomentKey, float],
    ) -> APCMExtensionResult | None:
        """Accept the carried frontier at these lower moments, or decline.

        No conic solve is run.  The carried coordinates are accepted only if
        the scaled extension matrix they build at these lower moments is
        positive semidefinite to the declared conic tolerance and every
        standardized coordinate stays inside the frozen box envelope.
        """

        extension = self.extension
        tolerance = extension.settings.conic_tolerance
        multiple = float(self.stage_policy.carried_completion_margin_multiple)
        values = np.asarray(
            [float(warm_frontier[key]) for key in extension.frontier_keys],
            dtype=float,
        )
        standardized = values / extension.frontier_scales
        if (
            float(np.max(np.abs(standardized)))
            > 1.0 - 10.0 * multiple * tolerance
        ):
            return None
        frontier = MappingProxyType(
            {
                key: float(value)
                for key, value in zip(
                    extension.frontier_keys, values, strict=True
                )
            }
        )
        matrix = extension.matrix(lower, frontier)
        scaled = extension.scaled_matrix(matrix)
        scaled_minimum = float(np.linalg.eigvalsh(scaled)[0])
        if scaled_minimum < multiple * tolerance:
            return None
        shifted = scaled + extension.settings.logdet_shift * np.eye(
            scaled.shape[0], dtype=complex
        )
        sign, logdet = np.linalg.slogdet(shifted)
        objective = (
            0.5 * float(standardized @ standardized)
            - extension.settings.logdet_weight
            / extension.dimension
            * float(logdet)
            if sign > 0.0
            else float("inf")
        )
        return APCMExtensionResult(
            lower_moments=MappingProxyType(
                {key: float(lower[key]) for key in extension.lower_keys}
            ),
            frontier_moments=frontier,
            moment_matrix=matrix,
            scaled_minimum_eigenvalue=scaled_minimum,
            minimum_eigenvalue=float(np.linalg.eigvalsh(matrix)[0]),
            objective=objective,
            kkt_backward_error=float("nan"),
            iterations=0,
            success=True,
            message="carried_completion",
        )

    def _select_stage_completion(
        self,
        lower: Mapping[MomentKey, float],
        warm_frontier: Mapping[MomentKey, float],
    ) -> APCMExtensionResult:
        """Select the stage lift under the named stage policy."""

        policy = self.stage_policy
        if policy.carried_completion:
            carried = self._carried_completion(lower, warm_frontier)
            if carried is not None:
                self.stage_policy_counters["carried_completion_accepted"] += 1
                return carried
            self.stage_policy_counters["carried_completion_rejected"] += 1
        warm = None if policy.completion_start == "cold" else warm_frontier
        result = self.extension.complete(lower, warm_frontier=warm)
        if result.success or not self.completion_ladder:
            return result
        for extension in self.completion_ladder:
            retried = extension.complete(lower, warm_frontier=warm)
            if retried.success:
                self.stage_policy_counters["completion_retry_accepted"] += 1
                return retried
        self.stage_policy_counters["completion_retry_exhausted"] += 1
        return result

    def select_completion(
        self,
        state: FloatArray,
        *,
        warm_frontier: Mapping[MomentKey, float] | None = None,
    ) -> APCMExtensionResult:
        raw, hidden = unpack_projected_apcm_state(
            state,
            active_keys=self.active_keys,
        )
        result = self.extension.complete(
            state_lower_moments(raw, hidden, self.active_keys),
            warm_frontier=warm_frontier,
        )
        if not result.success:
            raise ProjectedAPCMFailure(
                "moment_extension_failure: " + result.message
            )
        return result

    def targets(
        self,
        time: float,
        state: FloatArray,
        *,
        completion: APCMExtensionResult | None = None,
        warm_frontier: Mapping[MomentKey, float] | None = None,
    ) -> ProjectedAPCMTargets:
        """Build the archive, exact entrance, and hidden commutator targets."""

        raw, hidden = unpack_projected_apcm_state(
            state,
            active_keys=self.active_keys,
        )
        matrix_state = raw_moment_coordinates_to_matrix_state(raw)
        lower_moments = state_lower_moments(
            raw,
            hidden,
            self.active_keys,
        )
        selected = completion
        if selected is None:
            selected = self.extension.complete(
                lower_moments,
                warm_frontier=warm_frontier,
            )
        else:
            selected_lower = np.asarray(
                [selected.lower_moments[key] for key in self.extension.lower_keys]
            )
            current_lower = np.asarray(
                [lower_moments[key] for key in self.extension.lower_keys]
            )
            if not np.array_equal(selected_lower, current_lower):
                raise ValueError("supplied completion belongs to another state")
        if not selected.success:
            raise ProjectedAPCMFailure(
                "moment_extension_failure: " + selected.message
            )

        archive_matrix = _trace_project_correlation_velocity(
            matrix_dimer_rhs(float(time), matrix_state, self.parameters)
        )
        kpd = kpd_correlation_velocity_correction(
            matrix_state,
            self.parameters,
            lower_moments,
            include_k=self.settings.include_k,
            include_pauli=self.settings.include_pauli,
            include_opposite_spin=self.settings.include_opposite_spin,
        )
        augmented_matrix = _with_correlation_increment(archive_matrix, kpd)
        applied_increment = (
            augmented_matrix.electron_phonon_correlation
            - archive_matrix.electron_phonon_correlation
        )
        archive_retained = matrix_derivative_to_raw_moment_velocity(
            matrix_state,
            archive_matrix,
        )
        augmented_retained = matrix_derivative_to_raw_moment_velocity(
            matrix_state,
            augmented_matrix,
        )

        auxiliary_velocity = _active_commutator_velocity(
            self.active_keys,
            float(time),
            self.parameters,
            selected,
        )
        return ProjectedAPCMTargets(
            archive_matrix_velocity=archive_matrix,
            augmented_matrix_velocity=augmented_matrix,
            archive_retained_velocity=archive_retained,
            augmented_retained_velocity=augmented_retained,
            auxiliary_velocity=auxiliary_velocity,
            raw_kpd_correlation_velocity=np.asarray(kpd, dtype=complex),
            applied_correlation_velocity_increment=np.asarray(
                applied_increment,
                dtype=complex,
            ),
            completion=selected,
        )

    def unprojected_velocity_with_frontier(
        self,
        time: float,
        state: FloatArray,
        frontier_moments: Mapping[MomentKey, float],
    ) -> FloatArray:
        """Evaluate the augmented target while holding a frontier lift fixed.

        This is used only for the local admission Jacobian.  It evaluates the
        archive equations and exact active commutators without differentiating
        the nonsmooth conic selector or invoking the physical projection.
        """

        raw, hidden = unpack_projected_apcm_state(
            state,
            active_keys=self.active_keys,
        )
        matrix_state = raw_moment_coordinates_to_matrix_state(raw)
        lower = state_lower_moments(raw, hidden, self.active_keys)
        all_moments = {**dict(frontier_moments), **dict(lower)}
        missing = {
            key
            for observable in self.active_keys
            for _, hamiltonian_word in _hamiltonian_terms(
                float(time),
                self.parameters,
            )
            for key in _commutator(hamiltonian_word, observable)
            if key.degree > 0 and key not in all_moments
        }
        if missing:
            raise ValueError(
                f"fixed frontier is missing {len(missing)} commutator moments"
            )
        archive_matrix = _trace_project_correlation_velocity(
            matrix_dimer_rhs(float(time), matrix_state, self.parameters)
        )
        kpd = kpd_correlation_velocity_correction(
            matrix_state,
            self.parameters,
            all_moments,
            include_k=self.settings.include_k,
            include_pauli=self.settings.include_pauli,
            include_opposite_spin=self.settings.include_opposite_spin,
        )
        augmented_matrix = _with_correlation_increment(archive_matrix, kpd)
        retained = matrix_derivative_to_raw_moment_velocity(
            matrix_state,
            augmented_matrix,
        )
        auxiliary = _active_commutator_velocity(
            self.active_keys,
            float(time),
            self.parameters,
            _MomentMappingResolver(all_moments),
        )
        return np.concatenate((retained, auxiliary))

    def evaluate(
        self,
        time: float,
        state: FloatArray,
        *,
        completion: APCMExtensionResult | None = None,
        warm_frontier: Mapping[MomentKey, float] | None = None,
    ) -> ProjectedAPCMEvaluation:
        try:
            target = self.targets(
                time,
                state,
                completion=completion,
                warm_frontier=warm_frontier,
            )
            raw, hidden = unpack_projected_apcm_state(
                state,
                active_keys=self.active_keys,
            )
            projected = self.geometry.project_velocity(
                float(time),
                raw,
                hidden,
                target.augmented_retained_velocity,
                target.auxiliary_velocity,
                target.completion,
                self.parameters,
            )
        except Exception as error:
            _attach_stage_failure(
                error,
                self._stage_failure_record(
                    "evaluate",
                    time,
                    state,
                    (
                        warm_frontier
                        if warm_frontier is not None
                        else (
                            None
                            if completion is None
                            else completion.frontier_moments
                        )
                    ),
                    error,
                ),
            )
            raise
        return ProjectedAPCMEvaluation(
            derivative=np.concatenate(
                (projected.retained_velocity, projected.auxiliary_velocity)
            ),
            targets=target,
            projection=projected,
        )

    def contain_stage(
        self,
        trial_state: FloatArray,
        *,
        warm_frontier: Mapping[MomentKey, float],
        retained_metric: FloatArray,
        auxiliary_metric: FloatArray,
        completion: APCMExtensionResult | None = None,
        stage_triggered: bool = False,
    ) -> APCMStageRetraction:
        """Select a lift for a trial stage and retract only if infeasible.

        ``completion`` is a named option used by the event-triggered outer
        guard: it supplies an already-selected lift for exactly these lower
        moments instead of the declared proximal log-det completion.  Its
        default ``None`` is the recorded path.  ``stage_triggered`` tells the
        named ``quiet_containment`` ablation whether this stage fired; its
        default ``False`` matters only when that ablation is enabled.
        """

        try:
            return self._contain_stage(
                trial_state,
                warm_frontier=warm_frontier,
                retained_metric=retained_metric,
                auxiliary_metric=auxiliary_metric,
                completion=completion,
                stage_triggered=stage_triggered,
            )
        except Exception as error:
            _attach_stage_failure(
                error,
                self._stage_failure_record(
                    "contain", float("nan"), trial_state, warm_frontier, error
                ),
            )
            raise

    def _contain_stage(
        self,
        trial_state: FloatArray,
        *,
        warm_frontier: Mapping[MomentKey, float],
        retained_metric: FloatArray,
        auxiliary_metric: FloatArray,
        completion: APCMExtensionResult | None = None,
        stage_triggered: bool = False,
    ) -> APCMStageRetraction:
        """Select a lift and retract, without the failure-record wrapper."""

        raw, hidden = unpack_projected_apcm_state(
            trial_state,
            active_keys=self.active_keys,
        )
        lower = state_lower_moments(raw, hidden, self.active_keys)
        if completion is None:
            completion = (
                self.extension.complete(
                    lower,
                    warm_frontier=warm_frontier,
                )
                if self.stage_policy.is_default
                else self._select_stage_completion(lower, warm_frontier)
            )
        else:
            supplied = np.asarray(
                [completion.lower_moments[key] for key in self.extension.lower_keys]
            )
            current = np.asarray(
                [lower[key] for key in self.extension.lower_keys]
            )
            if not np.array_equal(supplied, current):
                raise ValueError("supplied completion belongs to another state")
        policy = self.stage_policy
        if not policy.quiet_containment and not stage_triggered:
            self.stage_policy_counters["quiet_containment_skipped"] += 1
            return APCMStageRetraction(
                raw_coordinates=raw.copy(),
                hidden_values=hidden.copy(),
                completion=completion,
                retained_correction_norm=0.0,
                auxiliary_correction_norm=0.0,
                iterations=0,
                applied=False,
                status="quiet_containment_disabled",
            )
        if not policy.stage_retraction:
            self.stage_policy_counters["retraction_suppressed"] += 1
            self.geometry.validate_envelopes(hidden, completion)
            return APCMStageRetraction(
                raw_coordinates=raw.copy(),
                hidden_values=hidden.copy(),
                completion=completion,
                retained_correction_norm=0.0,
                auxiliary_correction_norm=0.0,
                iterations=0,
                applied=False,
                status="stage_retraction_disabled",
            )
        return self.geometry.retract_stage(
            raw,
            hidden,
            completion,
            retained_metric=retained_metric,
            auxiliary_metric=auxiliary_metric,
        )


#: The named one-shell outer target solves of the event-triggered guard;
#: ``"recorded"`` is the recorded explicit target stage.  See
#: ``ProjectiveOuterEventGuardSettings.outer_target_solver``.
OUTER_TARGET_SOLVERS: tuple[str, ...] = ("recorded", "tightened_retry")


@dataclass(frozen=True)
class ProjectiveOuterEventGuardSettings:
    """Declared thresholds for the event-triggered one-shell outer guard.

    The three floors mirror the carried-witness flow's ``CWRMFSettings``
    fields of the same names (``apcm_carried_witness.py:81-83``) so that the
    projective route and the carried-witness route fire on the same declared
    quantities: the endpoint unshifted Gram minimum against
    ``event_gram_floor`` and ``event_gram_hard_floor``, and the shared
    negative-mode coupling measure against ``event_coupling_threshold`` in
    between.  The selector keywords are the two named options of
    ``select_outer_feasible_projective_guard``; their defaults are the
    recorded lexicographic two-solve selector without the gate reorder.

    ``outer_target_solver`` names which solve produces the one-shell outer
    target stage of a triggered stage.  ``"recorded"``, the default, is
    ``apcm_projective_guard._solve_explicit_target_qp`` with its recorded
    retry ladder, which retries at the tightened numerical profile only for
    the statuses ``{"AlmostSolved", "InsufficientProgress",
    "MaxIterations"}``, so a first pass ending in ``NumericalError`` is
    returned as it stands.  ``"tightened_retry"`` is
    ``apcm_outer_selector_formulations.solve_outer_target_with_tightened_retry``:
    the same conic program, run at that same recorded tightened numerical
    profile (conic tolerance ``1e-10``, ``max_iter`` 500, 20 iterative
    refinement passes at ``1e-14``, equilibration off) whenever the first
    pass is not accepted, whatever its status.  The acceptance rule and its
    tolerance are unchanged.  ``outer_target_ladder`` ``False`` runs the
    tightened profile alone, and ``outer_target_time_limit_seconds`` is the
    declared per-solve time limit handed to the conic solver.
    """

    event_gram_floor: float = 1e-8
    event_gram_hard_floor: float = 1e-3
    event_coupling_threshold: float = 1e-2
    event_coupling_horizon: float = 0.05
    boxed: bool = True
    canonicalize: bool = False
    accept_first_provisional: bool = False
    frontier_weight: float = 0.0
    frontier_metric: str = "identity"
    warm_refinement_passes: int = 0
    exploratory_acceptance_tolerance: float | None = None
    outer_target_solver: str = "recorded"
    outer_target_ladder: bool = True
    outer_target_time_limit_seconds: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.event_gram_floor < self.event_gram_hard_floor:
            raise ValueError("event Gram floors must be positive and ordered")
        if self.event_coupling_threshold <= 0.0:
            raise ValueError("event_coupling_threshold must be positive")
        if self.event_coupling_horizon <= 0.0:
            raise ValueError("event_coupling_horizon must be positive")
        if self.frontier_weight < 0.0:
            raise ValueError("frontier_weight must be nonnegative")
        if self.frontier_metric not in {"identity", "readout_diagonal"}:
            raise ValueError(
                "frontier_metric must be 'identity' or 'readout_diagonal'"
            )
        if self.warm_refinement_passes < 0:
            raise ValueError("warm_refinement_passes must be nonnegative")
        if (
            self.exploratory_acceptance_tolerance is not None
            and not self.exploratory_acceptance_tolerance > 0.0
        ):
            raise ValueError(
                "exploratory_acceptance_tolerance must be positive"
            )
        if self.outer_target_solver not in OUTER_TARGET_SOLVERS:
            raise ValueError(
                "outer_target_solver must be one of "
                f"{list(OUTER_TARGET_SOLVERS)}"
            )
        if (
            self.outer_target_time_limit_seconds is not None
            and not self.outer_target_time_limit_seconds > 0.0
        ):
            raise ValueError(
                "outer_target_time_limit_seconds must be positive"
            )


def resolve_outer_target_selector(
    settings: ProjectiveOuterEventGuardSettings,
) -> Callable[..., Any]:
    """Return the selector the named ``outer_target_solver`` declares.

    At the default ``"recorded"`` this is
    ``select_outer_feasible_projective_guard`` itself, the same object the
    guard has always called, so the recorded path is unchanged.  At
    ``"tightened_retry"`` it is that same selector with the one-shell outer
    target stage handed to
    ``solve_outer_target_with_tightened_retry``; nothing downstream of the
    target stage changes.
    """

    if settings.outer_target_solver == "recorded":
        return select_outer_feasible_projective_guard
    from .apcm_outer_selector_formulations import (
        solve_outer_target_with_tightened_retry,
    )

    target_solver = partial(
        solve_outer_target_with_tightened_retry,
        ladder=bool(settings.outer_target_ladder),
        time_limit_seconds=settings.outer_target_time_limit_seconds,
    )

    def selector(
        current: Any,
        lower_moments: Mapping[MomentKey, float],
        readout: Any = None,
        **keywords: Any,
    ) -> Any:
        return select_outer_feasible_projective_guard(
            current,
            lower_moments,
            readout,
            target_solver=target_solver,
            **keywords,
        )

    selector.__name__ = (
        "select_outer_feasible_projective_guard_tightened_retry"
    )
    return selector


class _CouplingGeometryAdapter:
    """Carried-witness geometry surface over the fixed-dictionary state."""

    def __init__(self, model: FixedDictionaryProjectedAPCM) -> None:
        self._model = model
        self._extension = model.extension
        self.retained_count = _RAW_DIMENSION + len(model.active_keys)
        self.completion_count = len(self._extension.frontier_keys)
        self.state_count = self.retained_count + self.completion_count

    def unpack_state(
        self, state: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        values = np.asarray(state, dtype=float)
        if values.shape != (self.state_count,):
            raise ValueError("coupling state has the wrong dimension")
        return (
            values[: self.retained_count],
            values[self.retained_count :],
        )

    def pack_state(
        self, retained: FloatArray, completion: FloatArray
    ) -> FloatArray:
        return np.concatenate(
            (
                np.asarray(retained, dtype=float),
                np.asarray(completion, dtype=float),
            )
        )

    def split_retained(
        self, retained: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        values = np.asarray(retained, dtype=float)
        return values[:_RAW_DIMENSION], values[_RAW_DIMENSION:]

    def frontier_mapping(
        self, completion: FloatArray
    ) -> Mapping[MomentKey, float]:
        values = (
            np.asarray(completion, dtype=float)
            * self._extension.frontier_scales
        )
        return MappingProxyType(
            {
                key: float(value)
                for key, value in zip(
                    self._extension.frontier_keys, values, strict=True
                )
            }
        )

    def scaled_unified_matrix(
        self, retained: FloatArray, completion: FloatArray
    ) -> np.ndarray:
        raw, hidden = self.split_retained(retained)
        lower = state_lower_moments(raw, hidden, self._model.active_keys)
        matrix = self._extension.matrix(
            lower, self.frontier_mapping(completion)
        )
        return self._extension.scaled_matrix(matrix)

    def scaled_completion_coefficients(self) -> np.ndarray:
        return (
            self._extension.scaled_frontier_coefficients
            * self._extension.frontier_scales[:, None, None]
        )


class _CouplingModelAdapter:
    """Duck-typed model the shared negative-mode measure is defined on.

    ``negative_mode_coupling`` needs only an unpacking geometry, the scaled
    Gram and its frontier coefficients, and the retained velocity as a
    function of the frontier lift.  The fixed-dictionary APCM supplies the
    last one exactly, through ``unprojected_velocity_with_frontier``, which
    evaluates the archive equations and the exact active commutators while
    holding the lift fixed and without differentiating the conic selector.
    The fixed dictionary carries no propagated completion velocity, so
    ``desired_completion_velocity`` is zero here; that value enters only the
    reported ``weakest_eigenvalue_velocity`` diagnostic and never the trigger.
    """

    def __init__(self, model: FixedDictionaryProjectedAPCM) -> None:
        self._model = model
        self.geometry = _CouplingGeometryAdapter(model)

    def retained_velocity(
        self,
        time: float,
        retained: FloatArray,
        completion: FloatArray,
    ) -> FloatArray:
        return self._model.unprojected_velocity_with_frontier(
            float(time),
            np.asarray(retained, dtype=float),
            self.geometry.frontier_mapping(completion),
        )

    def desired_completion_velocity(
        self,
        time: float,
        retained: FloatArray,
        completion: FloatArray,
    ) -> FloatArray:
        return np.zeros(self.geometry.completion_count, dtype=float)


class ProjectiveOuterEventGuard:
    """Event-triggered one-shell outer guard for fixed-dictionary APCM.

    At every stage the unconstrained fixed-dictionary predictor is formed
    from the archive-plus-entrance target velocity and the exact active
    commutator velocity, before the retained-first tangent-cone projection.
    Its endpoint carries the stage's own frontier witness forward unchanged,
    and the endpoint Gram is eigendecomposed.  Above ``-event_gram_floor``
    the stage takes the declared proximal log-det completion.  Below it the
    shared negative-mode coupling measure decides, and below
    ``-event_gram_hard_floor`` the stage fires unconditionally.  A fired
    stage selects its lift with
    ``select_outer_feasible_projective_guard`` at the contained trial state's
    own lower moments, so that the propagated lift is a literal restriction
    of one witness feasible on the one-shell outer cone.
    """

    def __init__(
        self,
        model: FixedDictionaryProjectedAPCM,
        *,
        settings: ProjectiveOuterEventGuardSettings | None = None,
        readout: InvariantTargetReadout | None = None,
        selector: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.settings = (
            ProjectiveOuterEventGuardSettings() if settings is None else settings
        )
        self.readout = (
            compile_invariant_target_readout(model.extension)
            if readout is None
            else readout
        )
        self.selector = (
            resolve_outer_target_selector(self.settings)
            if selector is None
            else selector
        )
        self._coupling = _CouplingModelAdapter(model)
        self._records: list[dict[str, Any]] = []

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._records)

    def reset(self) -> None:
        self._records.clear()

    def _endpoint_gram_minimum(
        self,
        stage_state: FloatArray,
        unconstrained_velocity: FloatArray,
        step: float,
        carried: Mapping[MomentKey, float],
    ) -> tuple[float, FloatArray]:
        endpoint = np.asarray(stage_state, dtype=float) + step * np.asarray(
            unconstrained_velocity, dtype=float
        )
        raw, hidden = unpack_projected_apcm_state(
            endpoint, active_keys=self.model.active_keys
        )
        lower = state_lower_moments(raw, hidden, self.model.active_keys)
        matrix = self.model.extension.matrix(lower, carried)
        scaled = self.model.extension.scaled_matrix(matrix)
        return float(np.linalg.eigvalsh(scaled)[0]), endpoint

    def _coupling_measure(
        self,
        time: float,
        endpoint: FloatArray,
        carried: Mapping[MomentKey, float],
    ) -> tuple[float, dict[str, float | int]]:
        standardized = np.asarray(
            [
                float(carried[key]) / scale
                for key, scale in zip(
                    self.model.extension.frontier_keys,
                    self.model.extension.frontier_scales,
                    strict=True,
                )
            ],
            dtype=float,
        )
        diagnostics = negative_mode_coupling(
            self._coupling,
            float(time),
            self._coupling.geometry.pack_state(endpoint, standardized),
            negative_threshold=self.settings.event_gram_floor,
            coupling_horizon=self.settings.event_coupling_horizon,
        )
        measure = max(
            float(
                diagnostics[
                    "maximum_negative_mode_predicted_retained_velocity_relative_change"
                ]
            ),
            float(
                diagnostics[
                    "maximum_negative_mode_two_stage_c_velocity_relative_change"
                ]
            ),
        )
        return measure, diagnostics

    def _guarded_completion(
        self,
        lower: Mapping[MomentKey, float],
    ) -> tuple[APCMExtensionResult | None, dict[str, Any]]:
        started = perf_counter()
        try:
            selection = self.selector(
                self.model.extension,
                lower,
                self.readout,
                boxed=self.settings.boxed,
                canonicalize=self.settings.canonicalize,
                accept_first_provisional=(
                    self.settings.accept_first_provisional
                ),
                frontier_weight=self.settings.frontier_weight,
                frontier_metric=self.settings.frontier_metric,
                warm_refinement_passes=(
                    self.settings.warm_refinement_passes
                ),
                exploratory_acceptance_tolerance=(
                    self.settings.exploratory_acceptance_tolerance
                ),
            )
        except (RuntimeError, ValueError) as error:
            return None, {
                "selector_wall_seconds": perf_counter() - started,
                "selector_outcome": f"selector_error:{error}",
            }
        elapsed = perf_counter() - started
        if not selection.success:
            return None, {
                "selector_wall_seconds": elapsed,
                "selector_outcome": (
                    f"selector_rejected:{selection.outer_target_stage.status}"
                ),
            }
        frontier_values = np.asarray(
            [
                selection.current_frontier_moments[key]
                for key in self.model.extension.frontier_keys
            ],
            dtype=float,
        )
        envelope = float(
            np.max(
                np.abs(frontier_values) / self.model.extension.frontier_scales
            )
        )
        tolerance = 10.0 * self.model.extension.settings.conic_tolerance
        if envelope >= 1.0 - tolerance:
            return None, {
                "selector_wall_seconds": elapsed,
                "selector_outcome": "envelope_fallback",
                "selector_envelope_ratio": envelope,
            }
        completion = APCMExtensionResult(
            lower_moments=MappingProxyType(
                {
                    key: float(lower[key])
                    for key in self.model.extension.lower_keys
                }
            ),
            frontier_moments=selection.current_frontier_moments,
            moment_matrix=selection.current_moment_matrix,
            scaled_minimum_eigenvalue=(
                selection.current_minimum_scaled_eigenvalue
            ),
            minimum_eigenvalue=float(
                np.linalg.eigvalsh(selection.current_moment_matrix)[0]
            ),
            objective=float(selection.outer_target_stage.objective),
            kkt_backward_error=float(
                selection.outer_target_stage.independent_feasibility_residual
            ),
            iterations=int(selection.outer_target_stage.iterations),
            success=True,
            message=(
                "projective_outer_event_guard:" + selection.witness_source
            ),
        )
        return completion, {
            "selector_wall_seconds": elapsed,
            "selector_outcome": "outer_feasible_witness",
            "selector_witness_source": selection.witness_source,
            "selector_envelope_ratio": envelope,
            "selector_objective_invariant_target_image_quadratic": float(
                selection.outer_target_stage.objective
            ),
            "selector_current_minimum_scaled_eigenvalue": float(
                selection.current_minimum_scaled_eigenvalue
            ),
            "selector_iterations": int(
                selection.outer_target_stage.iterations
            ),
        }

    def contain_stage(
        self,
        time: float,
        stage_state: FloatArray,
        evaluation: ProjectedAPCMEvaluation,
        trial_state: FloatArray,
        step: float,
    ) -> APCMStageRetraction:
        """Run the event test, then contain the trial stage."""

        started = perf_counter()
        carried = evaluation.targets.completion.frontier_moments
        unconstrained = np.concatenate(
            (
                evaluation.targets.augmented_retained_velocity,
                evaluation.targets.auxiliary_velocity,
            )
        )
        endpoint_minimum, endpoint = self._endpoint_gram_minimum(
            stage_state, unconstrained, step, carried
        )
        settings = self.settings
        triggered = endpoint_minimum < -settings.event_gram_hard_floor
        reason = "hard_floor" if triggered else "quiet"
        measure = 0.0
        if not triggered and endpoint_minimum < -settings.event_gram_floor:
            measure, _ = self._coupling_measure(
                float(time) + step, endpoint, carried
            )
            triggered = measure > settings.event_coupling_threshold
            reason = "coupling" if triggered else "tolerated"
        record: dict[str, Any] = {
            "time": float(time),
            "step": float(step),
            "endpoint_unshifted_minimum": endpoint_minimum,
            "coupling_measure": measure,
            "triggered": bool(triggered),
            "reason": reason,
            "trigger_wall_seconds": perf_counter() - started,
        }
        completion: APCMExtensionResult | None = None
        if triggered:
            raw, hidden = unpack_projected_apcm_state(
                trial_state, active_keys=self.model.active_keys
            )
            lower = state_lower_moments(raw, hidden, self.model.active_keys)
            completion, selector_record = self._guarded_completion(lower)
            record.update(selector_record)
        retraction = self.model.contain_stage(
            trial_state,
            warm_frontier=carried,
            retained_metric=evaluation.projection.retained_metric,
            auxiliary_metric=evaluation.projection.auxiliary_metric,
            completion=completion,
            stage_triggered=bool(triggered),
        )
        record["stage_retraction_applied"] = bool(retraction.applied)
        record["stage_retraction_status"] = retraction.status
        record["wall_seconds"] = perf_counter() - started
        self._records.append(record)
        return retraction


def prepare_projected_apcm_initial_state(
    parameters: DimerParameters,
    *,
    phonon_cutoff: int = 16,
    extension: SymmetryReducedPositiveExtension | None = None,
) -> ProjectedAPCMInitialization:
    """Contract one correlated preparation into the autonomous reduced state."""

    selector = (
        SymmetryReducedPositiveExtension(
            active_keys=ENTRANCE_RELATIVE_MOMENT_KEYS
        )
        if extension is None
        else extension
    )
    exact = exact_holstein_joint_moment_initial_state(
        parameters,
        hierarchy=THIRD_ORDER_HIERARCHY,
        phonon_cutoff=phonon_cutoff,
        canonical_embedding=True,
    )
    raw = matrix_state_to_raw_moment_coordinates(exact.matrix_state)
    _, moments = THIRD_ORDER_HIERARCHY.unpack(exact.hierarchy_coordinates)
    hidden = {
        key: moments[key] for key in selector.active_keys
    }
    hidden_values = np.asarray(
        [hidden[key] for key in selector.active_keys],
        dtype=float,
    )
    state = pack_projected_apcm_state(
        raw,
        hidden,
        active_keys=selector.active_keys,
    )
    completion = selector.complete(
        state_lower_moments(raw, hidden_values, selector.active_keys)
    )
    if not completion.success:
        raise ProjectedAPCMFailure(
            "initial moment_extension_failure: " + completion.message
        )
    base_minimum = float(
        np.linalg.eigvalsh(uncentered_joint_moment_matrix(raw))[0]
    )
    if base_minimum < -10.0 * selector.settings.conic_tolerance:
        raise ProjectedAPCMFailure(
            "initial retained moment matrix is not positive semidefinite"
        )
    return ProjectedAPCMInitialization(
        state=state,
        active_keys=selector.active_keys,
        completion=completion,
        exact_ground_energy=exact.energy,
        phonon_cutoff=phonon_cutoff,
    )


def _stage_state(retraction: APCMStageRetraction) -> FloatArray:
    return np.concatenate(
        (retraction.raw_coordinates, retraction.hidden_values)
    )


def _advance_projected_apcm_interval(
    model: FixedDictionaryProjectedAPCM,
    time: float,
    state: FloatArray,
    completion: APCMExtensionResult,
    step: float,
    *,
    subdivision_depth: int,
    maximum_subdivisions: int,
    outer_guard: "ProjectiveOuterEventGuard | None" = None,
) -> _AcceptedInterval:
    """Advance one interval, halving only after a declared stage failure."""

    def contain(
        stage_time: float,
        stage_state: FloatArray,
        evaluation: ProjectedAPCMEvaluation,
        trial: FloatArray,
    ) -> APCMStageRetraction:
        try:
            if outer_guard is None:
                return model.contain_stage(
                    trial,
                    warm_frontier=(
                        evaluation.targets.completion.frontier_moments
                    ),
                    retained_metric=evaluation.projection.retained_metric,
                    auxiliary_metric=evaluation.projection.auxiliary_metric,
                )
            return outer_guard.contain_stage(
                stage_time,
                stage_state,
                evaluation,
                trial,
                step,
            )
        except Exception as error:
            failure = projected_apcm_stage_failure(error)
            if failure is not None and not np.isfinite(failure.time):
                setattr(
                    error,
                    _STAGE_FAILURE_ATTRIBUTE,
                    replace(failure, time=float(stage_time)),
                )
            raise

    attempted_rhs_evaluations = 0
    try:
        attempted_rhs_evaluations += 1
        first = model.evaluate(time, state, completion=completion)
        first_trial = state + step * first.derivative
        first_stage = contain(time, state, first, first_trial)
        first_state = _stage_state(first_stage)

        attempted_rhs_evaluations += 1
        second = model.evaluate(
            time + step,
            first_state,
            completion=first_stage.completion,
        )
        second_trial = 0.75 * state + 0.25 * (
            first_state + step * second.derivative
        )
        second_stage = contain(time + step, first_state, second, second_trial)
        second_state = _stage_state(second_stage)

        attempted_rhs_evaluations += 1
        third = model.evaluate(
            time + 0.5 * step,
            second_state,
            completion=second_stage.completion,
        )
        final_trial = (1.0 / 3.0) * state + (2.0 / 3.0) * (
            second_state + step * third.derivative
        )
        final_stage = contain(
            time + 0.5 * step, second_state, third, final_trial
        )
        final_state = _stage_state(final_stage)
        if not np.all(np.isfinite(final_state)):
            raise FloatingPointError("nonfinite projected APCM stage")
        return _AcceptedInterval(
            state=final_state,
            completion=final_stage.completion,
            retained_correction_norm=max(
                first.projection.retained_correction_norm,
                second.projection.retained_correction_norm,
                third.projection.retained_correction_norm,
            ),
            auxiliary_correction_norm=max(
                first.projection.auxiliary_correction_norm,
                second.projection.auxiliary_correction_norm,
                third.projection.auxiliary_correction_norm,
            ),
            stage_retractions=sum(
                int(value.applied)
                for value in (first_stage, second_stage, final_stage)
            ),
            accepted_substeps=1,
            rhs_evaluations=attempted_rhs_evaluations,
        )
    except (ProjectedAPCMFailure, APCMProjectionError) as error:
        if subdivision_depth >= maximum_subdivisions:
            exhausted = ProjectedAPCMFailure(
                "moment_extension_failure persisted after "
                f"{maximum_subdivisions} stage subdivisions: {error}"
            )
            failure = projected_apcm_stage_failure(error)
            if failure is not None:
                _attach_stage_failure(exhausted, failure)
            raise exhausted from error
        half = 0.5 * step
        left = _advance_projected_apcm_interval(
            model,
            time,
            state,
            completion,
            half,
            subdivision_depth=subdivision_depth + 1,
            maximum_subdivisions=maximum_subdivisions,
            outer_guard=outer_guard,
        )
        right = _advance_projected_apcm_interval(
            model,
            time + half,
            left.state,
            left.completion,
            half,
            subdivision_depth=subdivision_depth + 1,
            maximum_subdivisions=maximum_subdivisions,
            outer_guard=outer_guard,
        )
        return _AcceptedInterval(
            state=right.state,
            completion=right.completion,
            retained_correction_norm=max(
                left.retained_correction_norm,
                right.retained_correction_norm,
            ),
            auxiliary_correction_norm=max(
                left.auxiliary_correction_norm,
                right.auxiliary_correction_norm,
            ),
            stage_retractions=(
                left.stage_retractions + right.stage_retractions
            ),
            accepted_substeps=(
                left.accepted_substeps + right.accepted_substeps
            ),
            rhs_evaluations=(
                attempted_rhs_evaluations
                + left.rhs_evaluations
                + right.rhs_evaluations
            ),
        )


def integrate_projected_apcm_ssprk3(
    model: FixedDictionaryProjectedAPCM,
    initial_state: FloatArray,
    *,
    initial_completion: APCMExtensionResult | None = None,
    final_time: float,
    time_step: float,
    initial_time: float = 0.0,
    maximum_subdivisions: int = 8,
    progress: Callable[[str], None] | None = None,
    outer_guard: "ProjectiveOuterEventGuard | None" = None,
    deadline: float | None = None,
    tolerate_stage_failure: bool = False,
) -> ProjectedAPCMTrajectory:
    """Advance fixed-dictionary APCM with containment before every RHS call.

    ``outer_guard`` is the named event-triggered one-shell outer guard
    policy; ``None``, the default, is the recorded per-stage policy in which
    every stage takes the declared proximal log-det completion.  ``deadline``
    is a ``perf_counter`` instant after which the rollout stops and returns
    the steps it completed, so that an exploratory run reports how far it
    reached instead of being killed.  ``tolerate_stage_failure`` is a named
    option for an ablation that is expected to fail: at its default
    ``False`` a declared stage failure propagates as it always has, and at
    ``True`` the rollout stops at the last accepted step and reports the
    failure in ``message`` with ``success`` false, exactly as the deadline
    path does.  Whether it raises or not, a failed rollout carries the steps
    it did accept on the exception, readable with
    ``projected_apcm_partial_trajectory``, and the failing stage's own state
    and lower moments, readable with ``projected_apcm_stage_failure``.

    ``initial_time`` is the declared start of the interval; its default
    ``0.0`` is the recorded path, for which ``times`` is unchanged bit for
    bit.  A positive value continues a rollout from a stored state at that
    time, so the drive is evaluated at the times that state belongs to.
    """

    if final_time <= 0.0 or time_step <= 0.0:
        raise ValueError("final_time and time_step must be positive")
    if maximum_subdivisions < 0:
        raise ValueError("maximum_subdivisions must be nonnegative")
    if initial_time < 0.0:
        raise ValueError("initial_time must be nonnegative")
    span = final_time - float(initial_time)
    if span <= 0.0:
        raise ValueError("final_time must exceed initial_time")
    intervals = int(round(span / time_step))
    if not np.isclose(intervals * time_step, span, atol=1e-12):
        raise ValueError("final_time must be an integer multiple of time_step")
    state = np.asarray(initial_state, dtype=float).copy()
    if state.shape != (len(model.state_names),):
        raise ValueError("initial_state has the wrong projected APCM dimension")
    completion = (
        model.select_completion(state)
        if initial_completion is None
        else initial_completion
    )

    times = float(initial_time) + np.linspace(0.0, span, intervals + 1)
    states = np.empty((intervals + 1, state.size), dtype=float)
    base_minima = np.empty(intervals + 1, dtype=float)
    extension_minima = np.empty(intervals + 1, dtype=float)
    retained_norms = np.zeros(intervals + 1, dtype=float)
    auxiliary_norms = np.zeros(intervals + 1, dtype=float)
    stage_retractions = np.zeros(intervals + 1, dtype=np.int64)
    accepted_substeps = np.zeros(intervals + 1, dtype=np.int64)
    states[0] = state
    raw, _ = unpack_projected_apcm_state(
        state,
        active_keys=model.active_keys,
    )
    base_minima[0] = float(
        np.linalg.eigvalsh(uncentered_joint_moment_matrix(raw))[0]
    )
    extension_minima[0] = completion.scaled_minimum_eigenvalue
    accepted_substeps[0] = 0
    rhs_evaluations = 0
    guard_triggered = np.zeros(intervals + 1, dtype=np.int64)
    completed = intervals
    message = "completed"
    success = True
    tolerated_failure: ProjectedAPCMStageFailure | None = None

    def reached(
        accepted_steps: int,
        succeeded: bool,
        reason: str,
        failure: ProjectedAPCMStageFailure | None = None,
    ) -> ProjectedAPCMTrajectory:
        """The trajectory over the steps accepted so far."""

        kept = accepted_steps + 1
        return ProjectedAPCMTrajectory(
            times=times[:kept],
            states=states[:kept],
            active_keys=model.active_keys,
            base_minimum_eigenvalues=base_minima[:kept],
            extension_minimum_eigenvalues=extension_minima[:kept],
            retained_correction_norms=retained_norms[:kept],
            auxiliary_correction_norms=auxiliary_norms[:kept],
            stage_retractions=stage_retractions[:kept],
            accepted_substeps=accepted_substeps[:kept],
            rhs_evaluations=rhs_evaluations,
            completed_steps=accepted_steps,
            success=succeeded,
            message=reason,
            outer_guard_triggered_stages=(
                None if outer_guard is None else guard_triggered[:kept]
            ),
            stage_failure=failure,
        )

    for step in range(intervals):
        if deadline is not None and perf_counter() >= deadline:
            completed = step
            success = False
            message = (
                "wall_budget_exceeded after "
                f"{step} of {intervals} completed steps"
            )
            break
        time = float(times[step])
        guard_before = 0 if outer_guard is None else len(outer_guard.records)
        try:
            accepted = _advance_projected_apcm_interval(
                model,
                time,
                state,
                completion,
                time_step,
                subdivision_depth=0,
                maximum_subdivisions=maximum_subdivisions,
                outer_guard=outer_guard,
            )
        except (ProjectedAPCMFailure, APCMProjectionError) as error:
            completed = step
            success = False
            message = (
                f"stage_failure after {step} of {intervals} completed "
                f"steps: {error}"
            )
            if not tolerate_stage_failure:
                _attach_partial_trajectory(
                    error, reached(step, False, message)
                )
                raise
            tolerated_failure = projected_apcm_stage_failure(error)
            break
        except Exception as error:
            _attach_partial_trajectory(
                error,
                reached(
                    step,
                    False,
                    f"unexpected_failure after {step} of {intervals} "
                    f"completed steps: {type(error).__name__}: {error}",
                ),
            )
            raise
        if outer_guard is not None:
            guard_triggered[step + 1] = sum(
                int(bool(record["triggered"]))
                for record in outer_guard.records[guard_before:]
            )
        state = accepted.state
        completion = accepted.completion
        rhs_evaluations += accepted.rhs_evaluations
        if not np.all(np.isfinite(state)):
            nonfinite = FloatingPointError(
                f"nonfinite projected APCM state after step {step + 1}"
            )
            _attach_partial_trajectory(
                nonfinite,
                reached(
                    step,
                    False,
                    f"nonfinite_state after {step} of {intervals} "
                    "completed steps",
                ),
            )
            raise nonfinite

        states[step + 1] = state
        raw, _ = unpack_projected_apcm_state(
            state,
            active_keys=model.active_keys,
        )
        base_minima[step + 1] = float(
            np.linalg.eigvalsh(uncentered_joint_moment_matrix(raw))[0]
        )
        extension_minima[step + 1] = completion.scaled_minimum_eigenvalue
        retained_norms[step + 1] = accepted.retained_correction_norm
        auxiliary_norms[step + 1] = accepted.auxiliary_correction_norm
        stage_retractions[step + 1] = accepted.stage_retractions
        accepted_substeps[step + 1] = accepted.accepted_substeps
        if progress is not None and (
            step == 0
            or step + 1 == intervals
            or (step + 1) % max(1, intervals // 20) == 0
        ):
            progress(
                f"t={times[step + 1]:.6f}/{final_time:.6f} "
                f"lambda_base={base_minima[step + 1]:.3e} "
                f"lambda_ext={extension_minima[step + 1]:.3e} "
                f"stage_retractions={stage_retractions[step + 1]} "
                f"substeps={accepted_substeps[step + 1]}"
            )

    return reached(completed, success, message, tolerated_failure)


__all__ = [
    "ENTRANCE_PROJECTED_APCM_STATE_NAMES",
    "PROJECTED_APCM_STATE_NAMES",
    "FixedDictionaryProjectedAPCM",
    "ProjectedAPCMEvaluation",
    "ProjectedAPCMStagePolicy",
    "ProjectedAPCMFailure",
    "ProjectedAPCMInitialization",
    "ProjectedAPCMSettings",
    "ProjectedAPCMStageFailure",
    "build_completion_ladder",
    "ProjectedAPCMTargets",
    "ProjectedAPCMTrajectory",
    "OUTER_TARGET_SOLVERS",
    "ProjectiveOuterEventGuard",
    "ProjectiveOuterEventGuardSettings",
    "resolve_outer_target_selector",
    "integrate_projected_apcm_ssprk3",
    "pack_projected_apcm_state",
    "prepare_projected_apcm_initial_state",
    "projected_apcm_partial_trajectory",
    "projected_apcm_stage_failure",
    "projected_apcm_state_names",
    "unpack_projected_apcm_state",
]
