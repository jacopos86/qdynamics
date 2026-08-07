"""Autonomous archive-backed positive-moment entrance-layer propagation.

This is the first executable APCM slice.  It propagates the complete archive
31-coordinate tuple, augments only its ``C`` velocity with dynamically evolved
``K/P/D`` information, and uses a positive fourth-moment completion for the
hidden commutator frontier.  The established joint-Gram controller remains the
retained physicality layer while the larger adaptive conic lift is developed.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from .adaptive_positive_moment import (
    HIDDEN_RELATIVE_MOMENT_KEYS,
    RAW_MOMENT_COORDINATE_NAMES,
    kpd_correlation_velocity_correction,
    matrix_derivative_to_raw_moment_velocity,
    matrix_state_to_raw_moment_coordinates,
    raw_moment_coordinates_to_matrix_state,
    relative_moments_from_matrix_state,
)
from .cone_correction import (
    CorrectionMetric,
    StructuredElectronPhononBarrierCorrection,
    structured_electron_phonon_barrier_correction,
)
from .exact_reference import exact_holstein_joint_moment_initial_state
from .hubbard_dimer import DimerParameters, FloatArray
from .matrix_reference import (
    MatrixDimerState,
    closed_scalar_to_matrix_state,
    electron_phonon_moment_matrix,
    matrix_derivative_to_closed_scalar,
    matrix_dimer_rhs,
    matrix_state_to_closed_scalar_coordinates,
)
from .moment_hierarchy import (
    MomentKey,
    PreparedTerminalMomentClosure,
    THIRD_ORDER_HIERARCHY,
)
from .positive_moment_completion import (
    PositiveFourthMomentCompletion,
    PositiveMomentCompletionResult,
    PositiveMomentCompletionSettings,
    PositiveMomentRetractionResult,
    pauli_weyl_moment_matrix,
)

ComplexArray = NDArray[np.complex128]

APCM_STATE_NAMES = (
    *RAW_MOMENT_COORDINATE_NAMES,
    *tuple(
        "hidden_"
        + key.spin_up.lower()
        + key.spin_down.lower()
        + f"_x{key.x_power}_p{key.p_power}"
        for key in HIDDEN_RELATIVE_MOMENT_KEYS
    ),
)


@dataclass(frozen=True)
class APCMSettings:
    """Frozen settings for the entrance-layer autonomous model."""

    include_k: bool = True
    include_pauli: bool = True
    include_opposite_spin: bool = True
    apply_physicality_controller: bool = True
    terminal_completion: Literal["positive", "zero_cumulant_prior"] = (
        "positive"
    )
    activation_margin: float = 1e-5
    barrier_rate: float = 5.0
    correction_metric: CorrectionMetric = "frobenius"
    cone_tolerance: float = 1e-8
    projection_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.activation_margin < 0.0:
            raise ValueError("activation_margin must be nonnegative")
        if self.barrier_rate <= 0.0:
            raise ValueError("barrier_rate must be positive")
        if self.cone_tolerance <= 0.0 or self.projection_tolerance <= 0.0:
            raise ValueError("controller tolerances must be positive")
        if self.terminal_completion not in (
            "positive",
            "zero_cumulant_prior",
        ):
            raise ValueError("unknown terminal completion mode")


@dataclass(frozen=True)
class APCMRhsEvaluation:
    """One autonomous RHS value with completion and controller diagnostics."""

    derivative: FloatArray
    archive_derivative: FloatArray
    augmented_derivative: FloatArray
    kpd_correction: ComplexArray
    completion: PositiveMomentCompletionResult
    controller: StructuredElectronPhononBarrierCorrection | None


@dataclass(frozen=True)
class APCMTrajectory:
    """Fixed-step SSPRK(3,3) rollout and its essential diagnostics."""

    times: FloatArray
    states: FloatArray
    completion_minimum_eigenvalues: FloatArray
    joint_gram_minimum_eigenvalues: FloatArray
    correction_norms: FloatArray
    hidden_retraction_norms: FloatArray
    completion_iterations: NDArray[np.int64]
    rhs_evaluations: int
    completed_steps: int
    success: bool
    message: str

    @property
    def archive_coordinates(self) -> FloatArray:
        return np.asarray(
            [
                matrix_state_to_closed_scalar_coordinates(
                    raw_moment_coordinates_to_matrix_state(
                        state[: len(RAW_MOMENT_COORDINATE_NAMES)]
                    )
                )
                for state in self.states
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class APCMInitializationResult:
    """Correlated preparation after canonical-CCR hidden-state retraction."""

    state: FloatArray
    frontier_moments: Mapping[MomentKey, float]
    hidden_retraction: PositiveMomentRetractionResult


class _PreparedCompletionAdapter:
    name = "prepared_positive_fourth_moment"

    def __init__(self, result: PositiveMomentCompletionResult) -> None:
        self._result = result

    def prepare(
        self,
        moments: Mapping[MomentKey, float],
        maximum_degree: int,
    ) -> PreparedTerminalMomentClosure:
        if maximum_degree != 3:
            raise ValueError("prepared completion requires hierarchy degree three")
        return self._result


class MomentCompletionError(RuntimeError):
    """The propagated lower moments do not admit the required completion."""


def pack_apcm_state(
    raw_coordinates: FloatArray,
    hidden_moments: Mapping[MomentKey, float],
) -> FloatArray:
    raw = np.asarray(raw_coordinates, dtype=float)
    if raw.shape != (len(RAW_MOMENT_COORDINATE_NAMES),):
        raise ValueError("raw coordinates must have shape (29,)")
    missing = set(HIDDEN_RELATIVE_MOMENT_KEYS).difference(hidden_moments)
    extra = set(hidden_moments).difference(HIDDEN_RELATIVE_MOMENT_KEYS)
    if missing or extra:
        raise ValueError(
            f"hidden mapping mismatch: {len(missing)} missing, {len(extra)} extra"
        )
    return np.concatenate(
        [raw, [float(hidden_moments[key]) for key in HIDDEN_RELATIVE_MOMENT_KEYS]]
    )


def unpack_apcm_state(
    state: FloatArray,
) -> tuple[FloatArray, dict[MomentKey, float]]:
    values = np.asarray(state, dtype=float)
    if values.shape != (len(APCM_STATE_NAMES),):
        raise ValueError(
            f"APCM state must have shape {(len(APCM_STATE_NAMES),)}, got {values.shape}"
        )
    hidden_values = values[len(RAW_MOMENT_COORDINATE_NAMES) :]
    hidden = {
        key: float(value)
        for key, value in zip(
            HIDDEN_RELATIVE_MOMENT_KEYS,
            hidden_values,
            strict=True,
        )
    }
    return values[: len(RAW_MOMENT_COORDINATE_NAMES)].copy(), hidden


def _default_positive_completion() -> PositiveFourthMomentCompletion:
    return PositiveFourthMomentCompletion(
        PositiveMomentCompletionSettings(
            phonon_envelope=16.0,
            logdet_weight=1.0,
            logdet_shift=1e-5,
            solver_tolerance=1e-7,
            maximum_iterations=2_000,
        )
    )


def prepare_apcm_initial_state(
    parameters: DimerParameters,
    *,
    phonon_cutoff: int = 16,
    completion: PositiveFourthMomentCompletion | None = None,
) -> APCMInitializationResult:
    """Build one consistent correlated preparation in the canonical cone."""

    exact = exact_holstein_joint_moment_initial_state(
        parameters,
        hierarchy=THIRD_ORDER_HIERARCHY,
        phonon_cutoff=phonon_cutoff,
    )
    raw = matrix_state_to_raw_moment_coordinates(
        exact.matrix_state
    )
    _, moments = THIRD_ORDER_HIERARCHY.unpack(
        exact.hierarchy_coordinates
    )
    hidden = {key: moments[key] for key in HIDDEN_RELATIVE_MOMENT_KEYS}
    _, reconstructed = relative_moments_from_matrix_state(
        exact.matrix_state,
        hidden,
    )
    selector = (
        _default_positive_completion() if completion is None else completion
    )
    retraction = selector.retract_lower_moments(
        reconstructed,
        adjustable_keys=HIDDEN_RELATIVE_MOMENT_KEYS,
    )
    if not retraction.success:
        raise MomentCompletionError(
            "initial canonical-CCR retraction failed: "
            f"{retraction.message}"
        )
    corrected_hidden = {
        key: retraction.lower_moments[key]
        for key in HIDDEN_RELATIVE_MOMENT_KEYS
    }
    return APCMInitializationResult(
        state=pack_apcm_state(raw, corrected_hidden),
        frontier_moments=retraction.frontier_moments,
        hidden_retraction=retraction,
    )


def initialize_apcm_state(
    parameters: DimerParameters,
    *,
    phonon_cutoff: int = 16,
) -> FloatArray:
    """Return the canonically feasible correlated APCM preparation."""

    return prepare_apcm_initial_state(
        parameters,
        phonon_cutoff=phonon_cutoff,
    ).state


def _closed_velocity_to_matrix_derivative(
    velocity: FloatArray,
) -> MatrixDimerState:
    """Invert the linear 31-coordinate derivative packing."""

    packed = closed_scalar_to_matrix_state(velocity)
    return MatrixDimerState(
        electron_density=(
            packed.electron_density - 0.5 * np.eye(2, dtype=complex)
        ),
        coherent_phonon=packed.coherent_phonon,
        phonon_density=packed.phonon_density,
        anomalous_phonon_density=packed.anomalous_phonon_density,
        electron_phonon_correlation=packed.electron_phonon_correlation,
    )


def _project_correlation_trace_velocity(
    derivative: MatrixDimerState,
) -> MatrixDimerState:
    """Apply the exact equality projection required by the 29-real chart."""

    correlation = np.asarray(
        derivative.electron_phonon_correlation,
        dtype=complex,
    ).copy()
    identity = np.eye(2, dtype=complex)
    for mode in range(correlation.shape[0]):
        correlation[mode] -= 0.5 * np.trace(correlation[mode]) * identity
    return MatrixDimerState(
        electron_density=derivative.electron_density,
        coherent_phonon=derivative.coherent_phonon,
        phonon_density=derivative.phonon_density,
        anomalous_phonon_density=derivative.anomalous_phonon_density,
        electron_phonon_correlation=correlation,
    )


class ArchiveBackedAPCM:
    """Evaluate the archive field plus an autonomous positive-moment source."""

    def __init__(
        self,
        parameters: DimerParameters,
        *,
        settings: APCMSettings | None = None,
        completion: PositiveFourthMomentCompletion | None = None,
    ) -> None:
        self.parameters = parameters
        self.settings = APCMSettings() if settings is None else settings
        self.completion = (
            _default_positive_completion()
            if completion is None
            else completion
        )

    def evaluate(
        self,
        time: float,
        state: FloatArray,
        *,
        warm_frontier: Mapping[MomentKey, float] | None = None,
    ) -> APCMRhsEvaluation:
        raw_coordinates, hidden = unpack_apcm_state(state)
        matrix_state = raw_moment_coordinates_to_matrix_state(raw_coordinates)
        archive_coordinates = matrix_state_to_closed_scalar_coordinates(
            matrix_state
        )
        center, moments = relative_moments_from_matrix_state(
            matrix_state,
            hidden,
        )
        if self.settings.terminal_completion == "zero_cumulant_prior":
            completion = self.completion.prior_result(moments)
        else:
            completion = (
                self.completion.complete(moments)
                if warm_frontier is None
                else self.completion.result_from_frontier(
                    moments,
                    warm_frontier,
                )
            )
            if not completion.success:
                # A stale witness can cross a singular face.  One deterministic
                # state-local completion retry is the ordinary local repair.
                completion = self.completion.complete(moments)
            if not completion.success:
                raise MomentCompletionError(
                    "positive moment completion failed: "
                    f"{completion.message}; lambda_min="
                    f"{completion.minimum_moment_matrix_eigenvalue:.3e}"
                )

        archive_matrix_derivative = _project_correlation_trace_velocity(
            matrix_dimer_rhs(
                float(time),
                matrix_state,
                self.parameters,
            )
        )
        archive_derivative = matrix_derivative_to_closed_scalar(
            archive_matrix_derivative
        )
        kpd = kpd_correlation_velocity_correction(
            matrix_state,
            self.parameters,
            moments,
            include_k=self.settings.include_k,
            include_pauli=self.settings.include_pauli,
            include_opposite_spin=self.settings.include_opposite_spin,
        )
        augmented_matrix_derivative = MatrixDimerState(
            electron_density=archive_matrix_derivative.electron_density,
            coherent_phonon=archive_matrix_derivative.coherent_phonon,
            phonon_density=archive_matrix_derivative.phonon_density,
            anomalous_phonon_density=(
                archive_matrix_derivative.anomalous_phonon_density
            ),
            electron_phonon_correlation=(
                archive_matrix_derivative.electron_phonon_correlation + kpd
            ),
        )
        augmented_derivative = matrix_derivative_to_closed_scalar(
            augmented_matrix_derivative
        )
        controller: StructuredElectronPhononBarrierCorrection | None = None
        archive_velocity = augmented_derivative
        if self.settings.apply_physicality_controller:
            controller = structured_electron_phonon_barrier_correction(
                archive_coordinates,
                augmented_derivative,
                self.parameters,
                activation_margin=self.settings.activation_margin,
                barrier_rate=self.settings.barrier_rate,
                energy_neutral=True,
                preserve_correlation_trace=True,
                cone_tolerance=self.settings.cone_tolerance,
                projection_tolerance=self.settings.projection_tolerance,
                correction_metric=self.settings.correction_metric,
            )
            if not controller.converged:
                raise RuntimeError("joint-Gram controller failed to converge")
            archive_velocity = (
                augmented_derivative + controller.correction_coordinates
            )

        retained_matrix_velocity = _closed_velocity_to_matrix_derivative(
            archive_velocity
        )
        raw_velocity = matrix_derivative_to_raw_moment_velocity(
            matrix_state,
            retained_matrix_velocity,
        )

        hierarchy_state = THIRD_ORDER_HIERARCHY.pack(center, moments)
        hierarchy_derivative = THIRD_ORDER_HIERARCHY.rhs(
            float(time),
            hierarchy_state,
            self.parameters,
            closure=_PreparedCompletionAdapter(completion),
        )
        _, moment_derivatives = THIRD_ORDER_HIERARCHY.unpack(
            hierarchy_derivative
        )
        hidden_velocity = np.asarray(
            [moment_derivatives[key] for key in HIDDEN_RELATIVE_MOMENT_KEYS],
            dtype=float,
        )
        return APCMRhsEvaluation(
            derivative=np.concatenate([raw_velocity, hidden_velocity]),
            archive_derivative=archive_derivative,
            augmented_derivative=augmented_derivative,
            kpd_correction=kpd,
            completion=completion,
            controller=controller,
        )

    def retract_hidden_state(
        self,
        state: FloatArray,
        *,
        warm_frontier: Mapping[MomentKey, float] | None = None,
    ) -> tuple[FloatArray, PositiveMomentRetractionResult]:
        """Make the smallest scaled hidden-moment stage adjustment for PSD."""

        raw_coordinates, hidden = unpack_apcm_state(state)
        matrix_state = raw_moment_coordinates_to_matrix_state(raw_coordinates)
        _, moments = relative_moments_from_matrix_state(matrix_state, hidden)
        retraction = self.completion.retract_lower_moments(
            moments,
            adjustable_keys=HIDDEN_RELATIVE_MOMENT_KEYS,
            warm_frontier=warm_frontier,
        )
        if not retraction.success:
            raise MomentCompletionError(
                "extended moment retraction failed: "
                f"{retraction.message}; lambda_min="
                f"{retraction.minimum_moment_matrix_eigenvalue:.3e}"
            )
        corrected_hidden = {
            key: retraction.lower_moments[key]
            for key in HIDDEN_RELATIVE_MOMENT_KEYS
        }
        return pack_apcm_state(raw_coordinates, corrected_hidden), retraction

    def ensure_extended_feasible(
        self,
        state: FloatArray,
        *,
        warm_frontier: Mapping[MomentKey, float],
    ) -> tuple[FloatArray, Mapping[MomentKey, float], float, float]:
        """Retain a stage when its warm witness is PSD, otherwise retract it."""

        raw_coordinates, hidden = unpack_apcm_state(state)
        matrix_state = raw_moment_coordinates_to_matrix_state(raw_coordinates)
        _, moments = relative_moments_from_matrix_state(matrix_state, hidden)
        warm_matrix = pauli_weyl_moment_matrix(
            {**moments, **dict(warm_frontier)}
        )
        minimum = float(np.linalg.eigvalsh(warm_matrix)[0])
        if self.settings.terminal_completion == "zero_cumulant_prior":
            return state, warm_frontier, 0.0, minimum
        if minimum >= -10.0 * self.completion.settings.solver_tolerance:
            return state, warm_frontier, 0.0, minimum
        corrected, retraction = self.retract_hidden_state(
            state,
            warm_frontier=warm_frontier,
        )
        return (
            corrected,
            retraction.frontier_moments,
            retraction.scaled_lower_correction_norm,
            retraction.minimum_moment_matrix_eigenvalue,
        )


def integrate_apcm_ssprk3(
    model: ArchiveBackedAPCM,
    initial_state: FloatArray,
    *,
    initial_time: float = 0.0,
    final_time: float,
    time_step: float,
    initial_frontier: Mapping[MomentKey, float] | None = None,
    progress: Callable[[str], None] | None = None,
    checkpoint: (
        Callable[[int, float, FloatArray, Mapping[MomentKey, float]], None]
        | None
    ) = None,
) -> APCMTrajectory:
    """Propagate the autonomous APCM state with fixed-step SSPRK(3,3)."""

    if initial_time < 0.0 or final_time <= initial_time or time_step <= 0.0:
        raise ValueError(
            "require 0 <= initial_time < final_time and positive time_step"
        )
    duration = final_time - initial_time
    intervals = int(round(duration / time_step))
    if not np.isclose(intervals * time_step, duration, atol=1e-12):
        raise ValueError(
            "final_time - initial_time must be an integer multiple of time_step"
        )
    state = np.asarray(initial_state, dtype=float).copy()
    if state.shape != (len(APCM_STATE_NAMES),):
        raise ValueError("initial_state has the wrong APCM dimension")

    times = np.linspace(initial_time, final_time, intervals + 1)
    states = np.empty((intervals + 1, state.size), dtype=float)
    completion_minima = np.empty(intervals + 1, dtype=float)
    joint_minima = np.empty(intervals + 1, dtype=float)
    correction_norms = np.empty(intervals + 1, dtype=float)
    hidden_retraction_norms = np.empty(intervals + 1, dtype=float)
    completion_iterations = np.empty(intervals + 1, dtype=np.int64)
    states[0] = state
    initial_matrix = raw_moment_coordinates_to_matrix_state(
        state[: len(RAW_MOMENT_COORDINATE_NAMES)]
    )
    joint_minima[0] = float(
        np.linalg.eigvalsh(electron_phonon_moment_matrix(initial_matrix))[0]
    )
    correction_norms[0] = 0.0
    hidden_retraction_norms[0] = 0.0
    warm_frontier: Mapping[MomentKey, float] | None = (
        None
        if initial_frontier is None
        else MappingProxyType(dict(initial_frontier))
    )
    _, initial_hidden = unpack_apcm_state(state)
    _, initial_moments = relative_moments_from_matrix_state(
        initial_matrix,
        initial_hidden,
    )
    if model.settings.terminal_completion == "zero_cumulant_prior":
        initial_completion = model.completion.prior_result(initial_moments)
    elif warm_frontier is None:
        initial_completion = model.completion.complete(initial_moments)
    else:
        initial_completion = model.completion.result_from_frontier(
            initial_moments,
            warm_frontier,
        )
    if not initial_completion.success:
        raise MomentCompletionError(
            "initial terminal completion failed: "
            f"{initial_completion.message}"
        )
    completion_minima[0] = (
        initial_completion.minimum_moment_matrix_eigenvalue
    )
    completion_iterations[0] = initial_completion.iterations
    warm_frontier = initial_completion.frontier_moments
    rhs_evaluations = 0

    for step in range(intervals):
        time = times[step]
        first = model.evaluate(time, state, warm_frontier=warm_frontier)
        rhs_evaluations += 1
        first_state = state + time_step * first.derivative
        first_state, first_frontier, first_retraction, _ = (
            model.ensure_extended_feasible(
                first_state,
                warm_frontier=first.completion.frontier_moments,
            )
        )
        second = model.evaluate(
            time + time_step,
            first_state,
            warm_frontier=first_frontier,
        )
        rhs_evaluations += 1
        second_state = 0.75 * state + 0.25 * (
            first_state + time_step * second.derivative
        )
        second_state, second_frontier, second_retraction, _ = (
            model.ensure_extended_feasible(
                second_state,
                warm_frontier=second.completion.frontier_moments,
            )
        )
        third = model.evaluate(
            time + 0.5 * time_step,
            second_state,
            warm_frontier=second_frontier,
        )
        rhs_evaluations += 1
        state = (1.0 / 3.0) * state + (2.0 / 3.0) * (
            second_state + time_step * third.derivative
        )
        if not np.all(np.isfinite(state)):
            raise FloatingPointError(f"nonfinite APCM state after step {step + 1}")
        state, final_frontier, final_retraction, final_minimum = (
            model.ensure_extended_feasible(
                state,
                warm_frontier=third.completion.frontier_moments,
            )
        )
        warm_frontier = MappingProxyType(dict(final_frontier))
        states[step + 1] = state
        completion_minima[step + 1] = final_minimum
        completion_iterations[step + 1] = third.completion.iterations
        matrix_state = raw_moment_coordinates_to_matrix_state(
            state[: len(RAW_MOMENT_COORDINATE_NAMES)]
        )
        joint_minima[step + 1] = float(
            np.linalg.eigvalsh(electron_phonon_moment_matrix(matrix_state))[0]
        )
        correction_norms[step + 1] = (
            0.0
            if third.controller is None
            else third.controller.lifted_frobenius_norm
        )
        hidden_retraction_norms[step + 1] = max(
            first_retraction,
            second_retraction,
            final_retraction,
        )
        if checkpoint is not None:
            checkpoint(
                step + 1,
                float(times[step + 1]),
                state.copy(),
                warm_frontier,
            )
        if progress is not None and (
            step == 0
            or step + 1 == intervals
            or (step + 1) % max(1, intervals // 20) == 0
        ):
            progress(
                f"t={times[step + 1]:.4f}/{final_time:.4f} "
                f"lambda_joint={joint_minima[step + 1]:.3e} "
                f"lambda_ext={completion_minima[step + 1]:.3e} "
                f"completion_it={completion_iterations[step + 1]} "
                f"hidden_retract={hidden_retraction_norms[step + 1]:.3e}"
            )

    return APCMTrajectory(
        times=times,
        states=states,
        completion_minimum_eigenvalues=completion_minima,
        joint_gram_minimum_eigenvalues=joint_minima,
        correction_norms=correction_norms,
        hidden_retraction_norms=hidden_retraction_norms,
        completion_iterations=completion_iterations,
        rhs_evaluations=rhs_evaluations,
        completed_steps=intervals,
        success=True,
        message="completed",
    )


__all__ = [
    "APCM_STATE_NAMES",
    "APCMRhsEvaluation",
    "APCMSettings",
    "APCMInitializationResult",
    "APCMTrajectory",
    "ArchiveBackedAPCM",
    "MomentCompletionError",
    "initialize_apcm_state",
    "integrate_apcm_ssprk3",
    "pack_apcm_state",
    "prepare_apcm_initial_state",
    "unpack_apcm_state",
]
