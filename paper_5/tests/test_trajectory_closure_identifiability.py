from __future__ import annotations

import numpy as np

from paper5.stability.multi_coherent_scores import closed_coordinate_distance
from paper5.stability.trajectory_closure_identifiability import (
    causal_history_metric_embedding,
    closed_state_metric_embedding,
    trajectory_closure_predictability,
    trajectory_source_predictability,
)


def test_metric_embedding_matches_equal_five_block_distance() -> None:
    rng = np.random.default_rng(260804)
    left = rng.normal(size=31)
    right = rng.normal(size=31)
    scales = np.exp(rng.normal(size=31))

    embedding = closed_state_metric_embedding(
        np.stack((left, right)),
        scales,
    )

    assert abs(
        np.linalg.norm(embedding[0] - embedding[1])
        - closed_coordinate_distance(left, right, scales)
    ) < 2e-15


def test_cross_time_predictability_detects_single_valued_source() -> None:
    times = np.arange(12, dtype=float)
    coordinates = np.zeros((2, times.size, 31), dtype=float)
    for member in range(2):
        coordinates[member, :, 0] = (
            np.sin(0.5 * np.pi * times) + 0.002 * member
        )
        coordinates[member, :, 3] = (
            np.cos(0.5 * np.pi * times) + 0.001 * member
        )
    target = np.empty((2, times.size, 2), dtype=float)
    target[..., 0] = 2.0 * coordinates[..., 0]
    target[..., 1] = -0.5 * coordinates[..., 3]

    result = trajectory_closure_predictability(
        times,
        coordinates,
        target,
        target + 1e-10,
        np.ones(31),
        minimum_time_separation=2.0,
        neighbor_count=2,
        query_neighbor_count=12,
    )

    assert result.normalized_prediction_rms < 1e-12
    assert result.normalized_reference_rms < 1e-8
    flat_times = np.tile(times, 2)
    assert np.all(
        np.abs(flat_times - flat_times[result.nearest_indices]) >= 2.0
    )


def test_coincident_retained_states_with_distinct_sources_are_not_predictable() -> None:
    times = np.arange(8, dtype=float)
    coordinates = np.zeros((2, times.size, 31), dtype=float)
    coordinates[:, :, 0] = np.tile([0.0, 1.0], 4)
    target = np.zeros((2, times.size, 1), dtype=float)
    target[0, :, 0] = 1.0
    target[1, :, 0] = -1.0

    result = trajectory_closure_predictability(
        times,
        coordinates,
        target,
        target + 1e-12,
        np.ones(31),
        minimum_time_separation=2.0,
        neighbor_count=3,
        query_neighbor_count=16,
    )

    assert result.close_state_threshold == 0.0
    assert result.normalized_prediction_rms > 0.9
    assert result.summary()["tension_target_to_reference_ratio"] > 1e10


def test_causal_history_disambiguates_projected_oscillator_phase() -> None:
    times = np.arange(16, dtype=float)
    coordinates = np.zeros((1, times.size, 31), dtype=float)
    coordinates[0, :, 0] = np.sin(0.5 * np.pi * times)
    target = np.cos(0.5 * np.pi * times)[None, :, None]
    scales = np.ones(31)
    lag_steps = 1

    current = closed_state_metric_embedding(coordinates, scales)[:, lag_steps:]
    history = causal_history_metric_embedding(
        coordinates,
        scales,
        lag_steps=lag_steps,
    )
    current_result = trajectory_source_predictability(
        times[lag_steps:],
        current,
        target[:, lag_steps:],
        target[:, lag_steps:] + 1e-12,
        minimum_time_separation=2.0,
        neighbor_count=2,
        query_neighbor_count=64,
    )
    history_result = trajectory_source_predictability(
        times[lag_steps:],
        history,
        target[:, lag_steps:],
        target[:, lag_steps:] + 1e-12,
        minimum_time_separation=2.0,
        neighbor_count=2,
        query_neighbor_count=64,
    )

    assert current_result.normalized_prediction_rms > 0.4
    assert history_result.normalized_prediction_rms < 1e-12
