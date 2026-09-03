from __future__ import annotations

import numpy as np
import pytest

from pipelines.time_dynamics.ap_mclachlan.avqds_stepping import (
    AVQDS_STEPPING_V1,
    advance_interval_delta_theta_controlled,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import (
    McLachlanInversePolicy,
    supported_inverse,
)
from pipelines.time_dynamics.ap_mclachlan.timestep_control import (
    RETAINED_METRIC_EIGENMODE_STEPPING_V1,
    ControlledVelocity,
    advance_interval_component_controlled,
    retained_metric_eigenmode_velocity,
)


def test_retained_metric_eigenmode_rate_uses_raw_gram_eigenvalues() -> None:
    gram = np.diag([4.0, 1.0])
    inverse = supported_inverse(
        gram,
        policy=McLachlanInversePolicy(pinv_rcond=0.0, ridge_lambda=0.5),
    )

    controlled = retained_metric_eigenmode_velocity(
        theta_dot=np.array([2.0, 3.0]),
        supported_inverse=inverse,
        epsilon=1.0e-14,
    )

    # |sqrt(4) * 2| = 4 and |sqrt(1) * 3| = 3.  Ridge must not
    # inflate these physical metric components.
    assert controlled.control_rate == pytest.approx(4.0)
    assert controlled.retained_metric_rank == 2


def test_retained_metric_eigenmode_rate_handles_a_rotated_metric() -> None:
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    gram = rotation @ np.diag([1.0, 9.0]) @ rotation.T
    theta_dot = rotation @ np.array([2.0, -0.5])
    inverse = supported_inverse(
        gram,
        policy=McLachlanInversePolicy(pinv_rcond=0.0, ridge_lambda=0.0),
    )

    controlled = retained_metric_eigenmode_velocity(
        theta_dot=theta_dot,
        supported_inverse=inverse,
        epsilon=1.0e-14,
    )

    assert controlled.control_rate == pytest.approx(2.0)


def test_retained_metric_eigenmode_rate_excludes_ridge_only_null_modes() -> None:
    gram = np.diag([0.0, 4.0])
    inverse = supported_inverse(
        gram,
        policy=McLachlanInversePolicy(pinv_rcond=0.0, ridge_lambda=1.0e-3),
    )

    controlled = retained_metric_eigenmode_velocity(
        theta_dot=np.array([1000.0, 1.0]),
        supported_inverse=inverse,
        epsilon=1.0e-14,
    )

    assert controlled.control_rate == pytest.approx(2.0)
    assert controlled.retained_metric_rank == 1


def test_component_controller_bounds_control_rate_but_updates_raw_theta() -> None:
    def rhs(_theta: np.ndarray, _time: float) -> ControlledVelocity:
        return ControlledVelocity(
            theta_dot=np.array([2.0, -1.0]),
            control_rate=4.0,
            retained_metric_rank=2,
            raw_metric_threshold=1.0e-12,
        )

    result = advance_interval_component_controlled(
        theta=np.zeros(2),
        time=0.0,
        dt_interval=0.12,
        controlled_rhs=rhs,
        step_max=0.2,
        policy=RETAINED_METRIC_EIGENMODE_STEPPING_V1,
    )

    np.testing.assert_allclose(result.theta_next, np.array([0.24, -0.12]))
    assert result.substep_count == 3
    assert result.min_dt == pytest.approx(0.02)
    assert result.max_dt == pytest.approx(0.05)
    assert result.max_control_rate == pytest.approx(4.0)
    assert result.min_retained_metric_rank == 2
    assert result.to_json_dict()["rhs_evaluation_count"] == 3


def test_component_controller_zero_rate_completes_interval_once() -> None:
    result = advance_interval_component_controlled(
        theta=np.array([1.0]),
        time=2.0,
        dt_interval=0.3,
        controlled_rhs=lambda _theta, _time: ControlledVelocity(
            theta_dot=np.zeros(1), control_rate=0.0
        ),
        step_max=0.1,
    )

    np.testing.assert_allclose(result.theta_next, np.array([1.0]))
    assert result.substep_count == 1
    assert result.min_dt == pytest.approx(0.3)


def test_raw_avqds_wrapper_preserves_policy_and_control_law() -> None:
    result = advance_interval_delta_theta_controlled(
        theta=np.zeros(2),
        time=0.0,
        dt_interval=0.25,
        theta_dot_rhs=lambda _theta, _time: np.array([2.0, -1.0]),
        delta_theta_max=0.2,
    )

    np.testing.assert_allclose(result.theta_next, np.array([0.5, -0.25]))
    assert result.substep_count == 3
    assert result.policy == AVQDS_STEPPING_V1
    assert set(result.to_json_dict()) == {
        "policy",
        "substep_count",
        "min_dt",
        "max_dt",
        "max_theta_dot",
        "delta_theta_max",
    }


@pytest.mark.parametrize("bad_bound", [0.0, -1.0, np.nan, np.inf])
def test_component_controller_rejects_invalid_bound(bad_bound: float) -> None:
    with pytest.raises(ValueError, match="step_max"):
        advance_interval_component_controlled(
            theta=np.zeros(1),
            time=0.0,
            dt_interval=0.1,
            controlled_rhs=lambda _theta, _time: ControlledVelocity(
                theta_dot=np.zeros(1), control_rate=0.0
            ),
            step_max=bad_bound,
        )
