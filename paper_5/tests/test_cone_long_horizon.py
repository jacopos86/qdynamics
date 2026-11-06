"""Tests for the long-horizon propagation and scoring of plan step 9.

The load-bearing claim is that ``integrate_sampled_dop853`` with no wall
budget is ``scipy.integrate.solve_ivp(..., method="DOP853", t_eval=...)`` on
the pinned energy-neutral barrier, so nothing about the declared default path
changes when a budget is available; and that a budget truncates the sample
list without moving any sample it already produced.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from paper5.stability import (
    DimerParameters,
    closed_cone_projected_rhs,
    closed_electron_phonon_cone_projected_rhs,
    closed_residual_subtracted_rhs,
    exact_ground_closed_scalar_coordinates,
)
from paper5.stability.cone_long_horizon import (
    OBSERVABLE_NAMES,
    barrier_activity,
    barrier_census_base_rhs,
    closed_observable_trajectories,
    integrate_sampled_dop853,
    joint_gram_barrier_activity,
    post_pulse_energy_drift,
    sample_rms,
    time_rms,
    window_indices,
    window_scores,
)
from paper5.stability.matrix_reference import (
    closed_scalar_rhs,
    closed_scalar_to_matrix_state,
    matrix_total_energy,
)

PINNED_FINAL_TIME = 0.3
PINNED_SAMPLE_STEP = 0.05

#: The joint-Gram barrier costs 218 right-hand-side evaluations over this
#: horizon at the pinned protocol, against 16,700 over t = 0 to 0.2, so the
#: byte-equality tests against ``solve_ivp`` for that lane run here.
JOINT_GRAM_FINAL_TIME = 0.05
JOINT_GRAM_SAMPLE_STEP = 0.01

#: The settings of the joint-Gram lane of
#: ``pipelines/open_dynamics/run_paper_v_energy_neutral_barrier_long_horizon.py``,
#: which are round 1's pinned joint-Gram settings.
JOINT_GRAM_SETTINGS = {
    "activation_margin": 1e-5,
    "target_flux": 0.0,
    "barrier_rate": 5.0,
    "energy_neutral": True,
    "preserve_correlation_trace": True,
    "subtract_initial_residual": False,
    "require_convergence": True,
    "cone_tolerance": 1e-8,
    "projection_tolerance": 1e-12,
    "correction_metric": "euclidean",
}


@pytest.fixture(scope="module")
def pinned_setup() -> tuple[DimerParameters, np.ndarray, np.ndarray]:
    parameters = DimerParameters(gamma=0.5, lambda_ep=1.5, drive_amplitude=1.0)
    initial_state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=16,
    )
    sample_times = np.arange(
        0.0,
        PINNED_FINAL_TIME + 0.5 * PINNED_SAMPLE_STEP,
        PINNED_SAMPLE_STEP,
    )
    # ``np.arange`` can overshoot the final time by one ulp; a sample outside
    # the integration span is not a sample of this protocol.
    sample_times = sample_times[sample_times <= PINNED_FINAL_TIME]
    return parameters, initial_state, sample_times


def _pinned_rhs(parameters: DimerParameters, initial_state: np.ndarray):
    return closed_cone_projected_rhs(
        parameters,
        initial_state,
        activation_margin=1e-5,
        target_flux=0.0,
        barrier_rate=5.0,
        energy_neutral=True,
    )


def test_unbudgeted_loop_reproduces_solve_ivp_on_the_pinned_barrier(
    pinned_setup,
) -> None:
    parameters, initial_state, sample_times = pinned_setup
    reference = solve_ivp(
        _pinned_rhs(parameters, initial_state),
        (0.0, PINNED_FINAL_TIME),
        initial_state,
        method="DOP853",
        t_eval=sample_times,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.05,
    )
    assert reference.success
    result = integrate_sampled_dop853(
        _pinned_rhs(parameters, initial_state),
        initial_state,
        final_time=PINNED_FINAL_TIME,
        sample_times=sample_times,
        wall_budget_seconds=None,
    )
    assert result.reached_final_time
    assert result.stop_reason == "final_time"
    assert result.function_evaluations == int(reference.nfev)
    assert np.array_equal(result.times, np.asarray(reference.t, dtype=float))
    assert np.array_equal(result.states, np.asarray(reference.y, dtype=float).T)


def test_uncorrected_archive_lane_also_reproduces_solve_ivp(pinned_setup) -> None:
    parameters, initial_state, sample_times = pinned_setup
    reference = solve_ivp(
        closed_residual_subtracted_rhs(parameters, initial_state),
        (0.0, PINNED_FINAL_TIME),
        initial_state,
        method="DOP853",
        t_eval=sample_times,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.05,
    )
    result = integrate_sampled_dop853(
        closed_residual_subtracted_rhs(parameters, initial_state),
        initial_state,
        final_time=PINNED_FINAL_TIME,
        sample_times=sample_times,
    )
    assert np.array_equal(result.states, np.asarray(reference.y, dtype=float).T)


def test_wall_budget_truncates_without_moving_a_produced_sample(
    pinned_setup,
) -> None:
    parameters, initial_state, sample_times = pinned_setup
    full = integrate_sampled_dop853(
        _pinned_rhs(parameters, initial_state),
        initial_state,
        final_time=PINNED_FINAL_TIME,
        sample_times=sample_times,
    )
    truncated = integrate_sampled_dop853(
        _pinned_rhs(parameters, initial_state),
        initial_state,
        final_time=PINNED_FINAL_TIME,
        sample_times=sample_times,
        wall_budget_seconds=1e-9,
    )
    assert not truncated.reached_final_time
    assert truncated.stop_reason == "wall_budget"
    assert truncated.times.size >= 1
    assert truncated.times.size <= full.times.size
    count = truncated.times.size
    assert np.array_equal(truncated.times, full.times[:count])
    assert np.array_equal(truncated.states, full.states[:count])


def test_a_nonpositive_wall_budget_is_refused(pinned_setup) -> None:
    parameters, initial_state, sample_times = pinned_setup
    with pytest.raises(ValueError):
        integrate_sampled_dop853(
            closed_residual_subtracted_rhs(parameters, initial_state),
            initial_state,
            final_time=PINNED_FINAL_TIME,
            sample_times=sample_times,
            wall_budget_seconds=0.0,
        )


def test_internal_energy_column_is_the_matrix_total_energy(pinned_setup) -> None:
    parameters, initial_state, sample_times = pinned_setup
    result = integrate_sampled_dop853(
        closed_residual_subtracted_rhs(parameters, initial_state),
        initial_state,
        final_time=PINNED_FINAL_TIME,
        sample_times=sample_times,
    )
    observables = closed_observable_trajectories(result.states, parameters)
    assert observables.shape == (result.times.size, len(OBSERVABLE_NAMES))
    expected = np.asarray(
        [
            matrix_total_energy(closed_scalar_to_matrix_state(row), parameters)
            for row in result.states
        ]
    )
    index = OBSERVABLE_NAMES.index("internal_energy")
    assert np.allclose(observables[:, index], expected, rtol=0.0, atol=1e-12)


def test_scores_vanish_against_the_same_trajectory(pinned_setup) -> None:
    parameters, initial_state, sample_times = pinned_setup
    result = integrate_sampled_dop853(
        closed_residual_subtracted_rhs(parameters, initial_state),
        initial_state,
        final_time=PINNED_FINAL_TIME,
        sample_times=sample_times,
    )
    scores = window_scores(
        result.times,
        result.states,
        result.states,
        parameters,
        window=(0.0, PINNED_FINAL_TIME),
        coordinate_scales=np.ones(31),
    )
    assert scores["coordinate_l2_time_rms"] == 0.0
    assert scores["all31_scaled_rms"] == 0.0
    for name in OBSERVABLE_NAMES:
        assert scores["observables"][name]["time_rms"] == 0.0
        assert scores["observables"][name]["sample_rms"] == 0.0


def test_the_two_rms_conventions_have_their_declared_values(pinned_setup) -> None:
    parameters, initial_state, sample_times = pinned_setup
    result = integrate_sampled_dop853(
        closed_residual_subtracted_rhs(parameters, initial_state),
        initial_state,
        final_time=PINNED_FINAL_TIME,
        sample_times=sample_times,
    )
    offset = np.zeros(31)
    offset[0] = 0.25
    shifted = result.states + offset
    scales = np.full(31, 2.0)
    scores = window_scores(
        result.times,
        shifted,
        result.states,
        parameters,
        window=(0.0, PINNED_FINAL_TIME),
        coordinate_scales=scales,
    )
    # A constant offset makes the unscaled Euclidean distance constant, so its
    # trapezoidal time-RMS is that distance.
    assert scores["coordinate_l2_time_rms"] == pytest.approx(0.25, rel=1e-12)
    # The scaled convention averages over all 31 coordinates, 30 of them zero.
    assert scores["all31_scaled_rms"] == pytest.approx(
        np.sqrt((0.25 / 2.0) ** 2 / 31.0), rel=1e-12
    )


def test_time_rms_and_sample_rms_are_the_declared_reductions() -> None:
    times = np.linspace(0.0, 2.0, 5)
    values = np.full(times.size, 3.0)
    assert time_rms(times, values) == pytest.approx(3.0, rel=1e-14)
    assert sample_rms(values) == pytest.approx(3.0, rel=1e-14)
    ramp = times.copy()
    assert time_rms(times, ramp) == pytest.approx(
        np.sqrt(np.trapezoid(ramp**2, times) / 2.0), rel=1e-14
    )


def test_a_window_must_be_sampled_and_have_positive_duration() -> None:
    times = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError):
        window_indices(times, (1.0, 0.0))
    with pytest.raises(ValueError):
        window_indices(times, (0.999, 1.0))


def test_post_pulse_drift_is_measured_from_the_first_sample_at_or_after_four() -> None:
    times = np.linspace(0.0, 6.0, 13)
    energies = np.where(times >= 4.0, 1.0 + 0.5 * (times - 4.0), 0.0)
    drift = post_pulse_energy_drift(times, energies)
    assert drift["post_pulse_start"] == pytest.approx(4.0)
    assert drift["post_pulse_max_drift_from_t4"] == pytest.approx(1.0)
    assert drift["post_pulse_energy_range"] == pytest.approx(1.0)


def test_barrier_activity_counts_the_sampled_corrections(pinned_setup) -> None:
    parameters, initial_state, sample_times = pinned_setup
    base_rhs = closed_residual_subtracted_rhs(parameters, initial_state)
    result = integrate_sampled_dop853(
        _pinned_rhs(parameters, initial_state),
        initial_state,
        final_time=PINNED_FINAL_TIME,
        sample_times=sample_times,
    )
    census = barrier_activity(
        result.times,
        result.states,
        base_rhs,
        window=(0.0, PINNED_FINAL_TIME),
    )
    assert census["sample_count"] == result.times.size
    assert 0.0 <= census["active_sample_fraction"] <= 1.0
    assert census["nonconverged_sample_count"] == 0
    assert census["maximum_correction_norm"] >= 0.0


@pytest.fixture(scope="module")
def joint_gram_setup() -> tuple[DimerParameters, np.ndarray, np.ndarray]:
    parameters = DimerParameters(gamma=0.5, lambda_ep=1.5, drive_amplitude=1.0)
    initial_state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=16,
    )
    sample_times = np.arange(
        0.0,
        JOINT_GRAM_FINAL_TIME + 0.5 * JOINT_GRAM_SAMPLE_STEP,
        JOINT_GRAM_SAMPLE_STEP,
    )
    sample_times = sample_times[sample_times <= JOINT_GRAM_FINAL_TIME]
    return parameters, initial_state, sample_times


def _joint_gram_rhs(
    parameters: DimerParameters,
    initial_state: np.ndarray,
    halfspace_backend: str | None = None,
):
    return closed_electron_phonon_cone_projected_rhs(
        parameters,
        initial_state,
        halfspace_backend=halfspace_backend,
        **JOINT_GRAM_SETTINGS,
    )


@pytest.mark.parametrize("halfspace_backend", [None, "active_set"])
def test_joint_gram_lane_reproduces_solve_ivp_on_a_short_horizon(
    joint_gram_setup,
    halfspace_backend,
) -> None:
    """The byte-equality gate for the 31-coordinate joint-Gram lane.

    Both the default backend (``None``, resolving to ``"slsqp"``) and the
    named ``"active_set"`` backend are compared against ``solve_ivp`` driving
    the same right-hand side, so the stepping loop is shown to add nothing to
    either of the two lanes the driver can run.
    """

    parameters, initial_state, sample_times = joint_gram_setup
    reference = solve_ivp(
        _joint_gram_rhs(parameters, initial_state, halfspace_backend),
        (0.0, JOINT_GRAM_FINAL_TIME),
        initial_state,
        method="DOP853",
        t_eval=sample_times,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.05,
    )
    assert reference.success
    result = integrate_sampled_dop853(
        _joint_gram_rhs(parameters, initial_state, halfspace_backend),
        initial_state,
        final_time=JOINT_GRAM_FINAL_TIME,
        sample_times=sample_times,
        wall_budget_seconds=None,
    )
    assert result.reached_final_time
    assert result.function_evaluations == int(reference.nfev)
    assert np.array_equal(result.times, np.asarray(reference.t, dtype=float))
    assert np.array_equal(result.states, np.asarray(reference.y, dtype=float).T)


def test_the_named_backend_moves_the_joint_gram_lane_only_at_solver_tolerance(
    joint_gram_setup,
) -> None:
    """The two backends solve the same minimum-norm problem.

    The recorded departure of the ``active_set`` backend from ``slsqp`` on the
    integrated joint-Gram barrier is 6.286e-9 relative to trajectory scale
    over t = 0 to 0.2 (round 1); over this short horizon the two lanes are
    required only to agree far more tightly than they differ from the
    uncorrected velocity.
    """

    parameters, initial_state, sample_times = joint_gram_setup
    default = integrate_sampled_dop853(
        _joint_gram_rhs(parameters, initial_state, None),
        initial_state,
        final_time=JOINT_GRAM_FINAL_TIME,
        sample_times=sample_times,
    )
    named = integrate_sampled_dop853(
        _joint_gram_rhs(parameters, initial_state, "active_set"),
        initial_state,
        final_time=JOINT_GRAM_FINAL_TIME,
        sample_times=sample_times,
    )
    assert np.array_equal(default.times, named.times)
    difference = float(np.max(np.abs(default.states - named.states)))
    scale = float(np.max(np.abs(default.states)))
    assert difference / scale < 1e-6


def test_the_reporting_hooks_leave_the_default_trajectory_alone(
    pinned_setup,
) -> None:
    """``step_callback`` and ``on_rhs_error`` do not move a produced sample."""

    parameters, initial_state, sample_times = pinned_setup
    recorded = integrate_sampled_dop853(
        _pinned_rhs(parameters, initial_state),
        initial_state,
        final_time=PINNED_FINAL_TIME,
        sample_times=sample_times,
    )
    seen: list[tuple[float, int]] = []
    hooked = integrate_sampled_dop853(
        _pinned_rhs(parameters, initial_state),
        initial_state,
        final_time=PINNED_FINAL_TIME,
        sample_times=sample_times,
        step_callback=lambda t, steps, elapsed: seen.append((t, steps)),
        on_rhs_error="stop",
    )
    assert np.array_equal(recorded.times, hooked.times)
    assert np.array_equal(recorded.states, hooked.states)
    assert recorded.function_evaluations == hooked.function_evaluations
    assert len(seen) == recorded.step_count


def test_a_raising_right_hand_side_propagates_by_default() -> None:
    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        if time > 0.1:
            raise RuntimeError("the barrier solve did not converge")
        return -state

    times = np.linspace(0.0, 0.3, 7)
    with pytest.raises(RuntimeError):
        integrate_sampled_dop853(
            rhs,
            np.ones(3),
            final_time=0.3,
            sample_times=times,
            maximum_step=0.02,
        )


def test_the_named_stop_keeps_the_samples_already_produced() -> None:
    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        if time > 0.1:
            raise RuntimeError("the barrier solve did not converge")
        return -state

    times = np.linspace(0.0, 0.3, 7)
    result = integrate_sampled_dop853(
        rhs,
        np.ones(3),
        final_time=0.3,
        sample_times=times,
        maximum_step=0.02,
        on_rhs_error="stop",
    )
    assert not result.reached_final_time
    assert result.stop_reason.startswith("rhs_error:RuntimeError:")
    assert result.times.size >= 1
    assert result.final_time_reached <= 0.1 + 1e-12


def test_an_unknown_rhs_error_policy_is_refused() -> None:
    times = np.linspace(0.0, 0.3, 7)
    with pytest.raises(ValueError):
        integrate_sampled_dop853(
            lambda time, state: -state,
            np.ones(3),
            final_time=0.3,
            sample_times=times,
            on_rhs_error="ignore",
        )


def test_joint_gram_census_counts_the_sampled_corrections(joint_gram_setup) -> None:
    parameters, initial_state, sample_times = joint_gram_setup
    base_rhs, _ = barrier_census_base_rhs(
        parameters,
        initial_state,
        subtract_initial_residual=JOINT_GRAM_SETTINGS["subtract_initial_residual"],
    )
    result = integrate_sampled_dop853(
        _joint_gram_rhs(parameters, initial_state),
        initial_state,
        final_time=JOINT_GRAM_FINAL_TIME,
        sample_times=sample_times,
    )
    census = joint_gram_barrier_activity(
        result.times,
        result.states,
        base_rhs,
        parameters,
        cone_tolerance=JOINT_GRAM_SETTINGS["cone_tolerance"],
        projection_tolerance=JOINT_GRAM_SETTINGS["projection_tolerance"],
        window=(0.0, JOINT_GRAM_FINAL_TIME),
    )
    assert census["sample_count"] == result.times.size
    assert 0.0 <= census["active_sample_fraction"] <= 1.0
    assert census["nonconverged_sample_count"] == 0
    assert census["maximum_correction_norm"] >= 0.0
    assert np.isfinite(census["minimum_joint_moment_eigenvalue"])


def test_the_census_base_velocity_is_the_one_the_controller_corrects(
    pinned_setup,
) -> None:
    """The two named base velocities, each reproduced bitwise.

    ``subtract_initial_residual`` is the setting of
    ``closed_cone_projected_rhs`` and
    ``closed_electron_phonon_cone_projected_rhs`` that decides which velocity
    the controller corrects, so it is the setting the census has to be solved
    at.  The default ten-coordinate lane carries ``True`` (Eq. (112)); the
    named joint-Gram lane carries ``False`` (the plain archive EOM).
    """

    parameters, initial_state, _ = pinned_setup
    subtracted, subtracted_name = barrier_census_base_rhs(
        parameters,
        initial_state,
        subtract_initial_residual=True,
    )
    plain, plain_name = barrier_census_base_rhs(
        parameters,
        initial_state,
        subtract_initial_residual=False,
    )
    assert subtracted_name.startswith("closed_residual_subtracted_rhs")
    assert plain_name.startswith("closed_scalar_rhs")
    recorded = closed_residual_subtracted_rhs(parameters, initial_state)
    for time in (0.0, 0.17, 0.3):
        assert np.array_equal(subtracted(time, initial_state), recorded(time, initial_state))
        assert np.array_equal(
            plain(time, initial_state),
            closed_scalar_rhs(time, initial_state, parameters),
        )
    # The vector the two differ by is the initial residual of Eq. (112), which
    # is time independent and is not small at this preparation.
    difference = plain(0.0, initial_state) - subtracted(0.0, initial_state)
    assert np.allclose(
        plain(0.3, initial_state) - subtracted(0.3, initial_state),
        difference,
        rtol=0.0,
        atol=1e-15,
    )
    assert float(np.linalg.norm(difference)) > 1e-2


def test_the_census_moves_when_it_is_solved_at_the_other_velocity(
    joint_gram_setup,
) -> None:
    """The census of the joint-Gram lane is not the same at both velocities.

    This is the defect the driver's ``census_base_velocity`` removes: solved
    at Eq. (112), the census of a lane whose own base velocity is the plain
    archive EOM reports a correction that lane never applied.
    """

    parameters, initial_state, sample_times = joint_gram_setup
    result = integrate_sampled_dop853(
        _joint_gram_rhs(parameters, initial_state),
        initial_state,
        final_time=JOINT_GRAM_FINAL_TIME,
        sample_times=sample_times,
    )
    own_rhs, _ = barrier_census_base_rhs(
        parameters,
        initial_state,
        subtract_initial_residual=JOINT_GRAM_SETTINGS["subtract_initial_residual"],
    )
    other_rhs, _ = barrier_census_base_rhs(
        parameters,
        initial_state,
        subtract_initial_residual=not JOINT_GRAM_SETTINGS[
            "subtract_initial_residual"
        ],
    )
    censuses = [
        joint_gram_barrier_activity(
            result.times,
            result.states,
            base,
            parameters,
            cone_tolerance=JOINT_GRAM_SETTINGS["cone_tolerance"],
            projection_tolerance=JOINT_GRAM_SETTINGS["projection_tolerance"],
            window=(0.0, JOINT_GRAM_FINAL_TIME),
        )
        for base in (own_rhs, other_rhs)
    ]
    own, other = censuses
    # The joint moment eigenvalue is a property of the stored state and does
    # not move with the velocity; the correction does.
    assert own["minimum_joint_moment_eigenvalue"] == pytest.approx(
        other["minimum_joint_moment_eigenvalue"],
        rel=1e-9,
    )
    assert own["maximum_correction_norm"] > other["maximum_correction_norm"]
