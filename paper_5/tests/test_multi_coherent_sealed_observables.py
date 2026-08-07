from __future__ import annotations

import numpy as np

from paper5.stability.hubbard_dimer import DimerParameters
from pipelines.open_dynamics.analyze_multi_coherent_sealed_observables import (
    _observable_trajectories,
    _reference_postmortem,
)


def test_observable_trajectory_reproduces_simple_internal_energy_sum() -> None:
    times = np.array([0.0, 1.0])
    coordinates = np.zeros((2, 31), dtype=float)
    coordinates[:, 1] = 0.5

    values = _observable_trajectories(
        times,
        coordinates,
        DimerParameters(lambda_ep=0.0),
    )

    np.testing.assert_allclose(values[:, 0], 0.5, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(values[:, 1], -2.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(values[:, 2:4], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(values[:, 4], -2.0, atol=0.0, rtol=0.0)


def test_reference_postmortem_locates_member_time_and_block() -> None:
    times = np.array([0.0, 0.5, 1.0])
    dop853 = np.zeros((3, times.size, 31), dtype=float)
    midpoint = dop853.copy()
    midpoint[2, 1, 20] = 3e-6

    result = _reference_postmortem(
        times,
        dop853,
        midpoint,
        np.ones(31),
    )

    assert result["member"] == "minus"
    assert result["time"] == 0.5
    assert result["dominant_block_at_maximum"] == "C"
    coordinate = result["largest_scaled_coordinate_disagreement"]
    assert coordinate["coordinate_index"] == 20
    assert coordinate["coordinate_name"] == (
        "correlation_0_diag_difference_imag"
    )
