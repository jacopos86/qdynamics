"""Pure effective-angle and ray-bound kernels for complete deletion sets."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipelines.time_dynamics.ap_mclachlan.deletion_geometry import (
    deletion_ray_bound,
    effective_deletion_angle,
)


def test_effective_deletion_angle_uses_absolute_signed_rotation_products() -> None:
    theta = np.asarray([-0.02, 0.03, 0.9])
    coefficients = np.asarray([-2.0, -3.0, 10.0])

    assert effective_deletion_angle(theta, coefficients, (0, 1)) == pytest.approx(0.13)


def test_effective_deletion_angle_includes_every_selected_coordinate() -> None:
    theta = np.asarray([0.1, -0.2, 0.3])
    coefficients = np.asarray([2.0, 0.5, 0.0])

    assert effective_deletion_angle(theta, coefficients, (2, 1, 0)) == pytest.approx(0.3)


def test_empty_deletion_has_zero_effective_angle_and_ray_bound() -> None:
    angle = effective_deletion_angle(np.empty(0), np.empty(0), ())

    assert angle == 0.0
    assert deletion_ray_bound(angle) == 0.0


@pytest.mark.parametrize("angle, expected", [(0.0, 0.0), (math.pi / 6.0, 0.5)])
def test_deletion_ray_bound_matches_known_angles(angle, expected) -> None:
    assert deletion_ray_bound(angle) == pytest.approx(expected)


@pytest.mark.parametrize("angle", [math.pi / 2.0, math.pi, 100.0])
def test_deletion_ray_bound_saturates_at_unit_distance(angle) -> None:
    assert deletion_ray_bound(angle) == 1.0
