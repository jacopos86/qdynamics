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
    closed_residual_subtracted_rhs,
    exact_ground_closed_scalar_coordinates,
)
from paper5.stability.cone_long_horizon import (
    OBSERVABLE_NAMES,
    barrier_activity,
    closed_observable_trajectories,
    integrate_sampled_dop853,
    post_pulse_energy_drift,
    sample_rms,
    time_rms,
    window_indices,
    window_scores,
)
from paper5.stability.matrix_reference import (
    closed_scalar_to_matrix_state,
    matrix_total_energy,
)

PINNED_FINAL_TIME = 0.3
PINNED_SAMPLE_STEP = 0.05


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
