from __future__ import annotations

import numpy as np

from paper5.stability import (
    FAN_MIGDAL_STATE_NAMES,
    DimerParameters,
    ehrenfest_fixed_point,
    ehrenfest_invariant,
    ehrenfest_rhs,
    fan_migdal_rhs,
    hartree_fock_zero_correlation_state,
    integrate_rk4,
)


def test_ehrenfest_fixed_points_are_stationary_and_normalized() -> None:
    for lambda_ep in (0.5, 1.5):
        parameters = DimerParameters(lambda_ep=lambda_ep, drive_amplitude=0.0)
        state = ehrenfest_fixed_point(parameters)

        np.testing.assert_allclose(
            ehrenfest_rhs(0.0, state, parameters),
            0.0,
            atol=1e-14,
        )
        assert abs(ehrenfest_invariant(state) - 1.0) < 1e-14


def test_hf_zero_correlation_state_has_two_initial_correlation_sources() -> None:
    parameters = DimerParameters(lambda_ep=1.5, drive_amplitude=1.0)
    state = hartree_fock_zero_correlation_state()
    residual = fan_migdal_rhs(0.0, state, parameters)
    nonzero = {
        name: value
        for name, value in zip(FAN_MIGDAL_STATE_NAMES, residual, strict=True)
        if abs(value) > 1e-14
    }

    assert set(nonzero) == {"delta_corr_imag", "delta_corr_imag_minus"}
    assert abs(nonzero["delta_corr_imag"] + parameters.coupling / 4.0) < 1e-14
    assert (
        abs(nonzero["delta_corr_imag_minus"] + parameters.coupling / 2.0)
        < 1e-14
    )


def test_strong_coupling_divergence_time_is_rk4_step_converged() -> None:
    parameters = DimerParameters(lambda_ep=1.5, drive_amplitude=1.0)
    initial_state = hartree_fock_zero_correlation_state()
    rhs = lambda time, state: fan_migdal_rhs(time, state, parameters)

    coarse = integrate_rk4(
        rhs,
        initial_state,
        final_time=140.0,
        time_step=0.02,
        failure_threshold=1e4,
        state_names=FAN_MIGDAL_STATE_NAMES,
    )
    fine = integrate_rk4(
        rhs,
        initial_state,
        final_time=140.0,
        time_step=0.01,
        failure_threshold=1e4,
        state_names=FAN_MIGDAL_STATE_NAMES,
    )

    assert coarse.diverged
    assert fine.diverged
    assert coarse.failure_time is not None
    assert fine.failure_time is not None
    assert abs(coarse.failure_time - fine.failure_time) < 0.1
    assert 130.0 < fine.failure_time < 131.0


def test_weak_coupling_control_remains_bounded() -> None:
    parameters = DimerParameters(lambda_ep=0.5, drive_amplitude=1.0)
    initial_state = hartree_fock_zero_correlation_state()
    rhs = lambda time, state: fan_migdal_rhs(time, state, parameters)
    result = integrate_rk4(
        rhs,
        initial_state,
        final_time=140.0,
        time_step=0.02,
        failure_threshold=10.0,
        state_names=FAN_MIGDAL_STATE_NAMES,
    )

    assert not result.diverged
    assert result.max_abs_state < 1.0
