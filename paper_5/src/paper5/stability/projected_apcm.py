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

from dataclasses import dataclass
from typing import Callable, Mapping

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
from .apcm_positive_extension import (
    APCMExtensionResult,
    SymmetryReducedPositiveExtension,
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


@dataclass(frozen=True)
class ProjectedAPCMSettings:
    """Fixed entrance switches for one APCM model configuration."""

    include_k: bool = True
    include_pauli: bool = True
    include_opposite_spin: bool = True


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
    ) -> APCMStageRetraction:
        """Select a lift for a trial stage and retract only if infeasible."""

        raw, hidden = unpack_projected_apcm_state(
            trial_state,
            active_keys=self.active_keys,
        )
        lower = state_lower_moments(raw, hidden, self.active_keys)
        completion = self.extension.complete(
            lower,
            warm_frontier=warm_frontier,
        )
        return self.geometry.retract_stage(
            raw,
            hidden,
            completion,
            retained_metric=retained_metric,
            auxiliary_metric=auxiliary_metric,
        )


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
) -> _AcceptedInterval:
    """Advance one interval, halving only after a declared stage failure."""

    attempted_rhs_evaluations = 0
    try:
        attempted_rhs_evaluations += 1
        first = model.evaluate(time, state, completion=completion)
        first_trial = state + step * first.derivative
        first_stage = model.contain_stage(
            first_trial,
            warm_frontier=first.targets.completion.frontier_moments,
            retained_metric=first.projection.retained_metric,
            auxiliary_metric=first.projection.auxiliary_metric,
        )
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
        second_stage = model.contain_stage(
            second_trial,
            warm_frontier=second.targets.completion.frontier_moments,
            retained_metric=second.projection.retained_metric,
            auxiliary_metric=second.projection.auxiliary_metric,
        )
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
        final_stage = model.contain_stage(
            final_trial,
            warm_frontier=third.targets.completion.frontier_moments,
            retained_metric=third.projection.retained_metric,
            auxiliary_metric=third.projection.auxiliary_metric,
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
            raise ProjectedAPCMFailure(
                "moment_extension_failure persisted after "
                f"{maximum_subdivisions} stage subdivisions: {error}"
            ) from error
        half = 0.5 * step
        left = _advance_projected_apcm_interval(
            model,
            time,
            state,
            completion,
            half,
            subdivision_depth=subdivision_depth + 1,
            maximum_subdivisions=maximum_subdivisions,
        )
        right = _advance_projected_apcm_interval(
            model,
            time + half,
            left.state,
            left.completion,
            half,
            subdivision_depth=subdivision_depth + 1,
            maximum_subdivisions=maximum_subdivisions,
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
    maximum_subdivisions: int = 8,
    progress: Callable[[str], None] | None = None,
) -> ProjectedAPCMTrajectory:
    """Advance fixed-dictionary APCM with containment before every RHS call."""

    if final_time <= 0.0 or time_step <= 0.0:
        raise ValueError("final_time and time_step must be positive")
    if maximum_subdivisions < 0:
        raise ValueError("maximum_subdivisions must be nonnegative")
    intervals = int(round(final_time / time_step))
    if not np.isclose(intervals * time_step, final_time, atol=1e-12):
        raise ValueError("final_time must be an integer multiple of time_step")
    state = np.asarray(initial_state, dtype=float).copy()
    if state.shape != (len(model.state_names),):
        raise ValueError("initial_state has the wrong projected APCM dimension")
    completion = (
        model.select_completion(state)
        if initial_completion is None
        else initial_completion
    )

    times = np.linspace(0.0, final_time, intervals + 1)
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

    for step in range(intervals):
        time = float(times[step])
        accepted = _advance_projected_apcm_interval(
            model,
            time,
            state,
            completion,
            time_step,
            subdivision_depth=0,
            maximum_subdivisions=maximum_subdivisions,
        )
        state = accepted.state
        completion = accepted.completion
        rhs_evaluations += accepted.rhs_evaluations
        if not np.all(np.isfinite(state)):
            raise FloatingPointError(
                f"nonfinite projected APCM state after step {step + 1}"
            )

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

    return ProjectedAPCMTrajectory(
        times=times,
        states=states,
        active_keys=model.active_keys,
        base_minimum_eigenvalues=base_minima,
        extension_minimum_eigenvalues=extension_minima,
        retained_correction_norms=retained_norms,
        auxiliary_correction_norms=auxiliary_norms,
        stage_retractions=stage_retractions,
        accepted_substeps=accepted_substeps,
        rhs_evaluations=rhs_evaluations,
        completed_steps=intervals,
        success=True,
        message="completed",
    )


__all__ = [
    "ENTRANCE_PROJECTED_APCM_STATE_NAMES",
    "PROJECTED_APCM_STATE_NAMES",
    "FixedDictionaryProjectedAPCM",
    "ProjectedAPCMEvaluation",
    "ProjectedAPCMFailure",
    "ProjectedAPCMInitialization",
    "ProjectedAPCMSettings",
    "ProjectedAPCMTargets",
    "ProjectedAPCMTrajectory",
    "integrate_projected_apcm_ssprk3",
    "pack_projected_apcm_state",
    "prepare_projected_apcm_initial_state",
    "projected_apcm_state_names",
    "unpack_projected_apcm_state",
]
