"""Trajectory-local predictability tests for a 31-moment closure source.

The exact finite-cutoff witness in :mod:`closure_identifiability` rules out a
globally exact instantaneous 31-coordinate closure.  This module asks the
narrower empirical question: on a declared family of exact trajectories, is a
candidate missing source approximately predictable from the retained state?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .multi_coherent_scores import CLOSED_COORDINATE_BLOCKS


@dataclass(frozen=True)
class TrajectoryClosurePredictability:
    """Nearest-recurrence and cross-time kNN closure diagnostics."""

    nearest_indices: np.ndarray
    nearest_state_distances: np.ndarray
    nearest_target_distances: np.ndarray
    nearest_reference_uncertainty_bounds: np.ndarray
    prediction_errors: np.ndarray
    reference_uncertainties: np.ndarray
    target_fluctuation_scale: float
    normalized_prediction_rms: float
    normalized_reference_rms: float
    minimum_time_separation: float
    neighbor_count: int
    close_state_quantile: float
    close_state_threshold: float
    tension_sample_index: int
    tension_neighbor_index: int

    def summary(self) -> dict[str, float | int]:
        """Return scalar evidence without assigning a scientific verdict."""

        sample = self.tension_sample_index
        neighbor = self.tension_neighbor_index
        uncertainty = self.nearest_reference_uncertainty_bounds[sample]
        uncertainty_floor = max(
            float(uncertainty),
            np.finfo(float).tiny,
        )
        return {
            "sample_count": int(self.nearest_indices.size),
            "minimum_time_separation": self.minimum_time_separation,
            "neighbor_count": self.neighbor_count,
            "target_fluctuation_scale": self.target_fluctuation_scale,
            "normalized_knn_prediction_rms": (
                self.normalized_prediction_rms
            ),
            "normalized_two_reference_rms": self.normalized_reference_rms,
            "prediction_to_reference_rms_ratio": float(
                self.normalized_prediction_rms
                / max(
                    self.normalized_reference_rms,
                    np.finfo(float).tiny,
                )
            ),
            "nearest_state_distance_minimum": float(
                np.min(self.nearest_state_distances)
            ),
            "nearest_state_distance_median": float(
                np.median(self.nearest_state_distances)
            ),
            "nearest_target_distance_median": float(
                np.median(self.nearest_target_distances)
            ),
            "close_state_quantile": self.close_state_quantile,
            "close_state_threshold": self.close_state_threshold,
            "tension_sample_index": sample,
            "tension_neighbor_index": neighbor,
            "tension_state_distance": float(
                self.nearest_state_distances[sample]
            ),
            "tension_target_distance": float(
                self.nearest_target_distances[sample]
            ),
            "tension_reference_uncertainty_bound": float(uncertainty),
            "tension_target_to_reference_ratio": float(
                self.nearest_target_distances[sample] / uncertainty_floor
            ),
        }


def closed_state_metric_embedding(
    coordinates: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Embed the equal-five-block state metric into Euclidean coordinates."""

    values = np.asarray(coordinates, dtype=float)
    scale_array = np.asarray(scales, dtype=float)
    if values.shape[-1] != 31:
        raise ValueError("coordinates must have trailing dimension 31")
    if scale_array.shape != (31,) or np.any(scale_array <= 0.0):
        raise ValueError("scales must be positive with shape (31,)")
    if not np.all(np.isfinite(values)):
        raise ValueError("coordinates must be finite")

    embedded = np.empty_like(values)
    block_count = 5.0
    embedded[..., 0] = values[..., 0] / (2.0 * np.sqrt(block_count))
    embedded[..., 1:3] = values[..., 1:3] / np.sqrt(block_count)
    for name in ("B", "N", "A", "C"):
        block = CLOSED_COORDINATE_BLOCKS[name]
        block_size = block.stop - block.start
        embedded[..., block] = values[..., block] / (
            scale_array[block] * np.sqrt(block_count * block_size)
        )
    return embedded


def _valid_neighbors(
    tree: cKDTree,
    embedding: np.ndarray,
    times: np.ndarray,
    *,
    minimum_time_separation: float,
    neighbor_count: int,
    query_neighbor_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = embedding.shape[0]
    query_count = min(sample_count, max(query_neighbor_count, neighbor_count + 1))
    queried_distances, queried_indices = tree.query(
        embedding,
        k=query_count,
    )
    if query_count == 1:
        queried_distances = queried_distances[:, None]
        queried_indices = queried_indices[:, None]
    selected_indices = np.empty((sample_count, neighbor_count), dtype=int)
    selected_distances = np.empty((sample_count, neighbor_count), dtype=float)
    for sample_index in range(sample_count):
        valid = (
            np.abs(times[queried_indices[sample_index]] - times[sample_index])
            >= minimum_time_separation - 1e-12
        )
        candidate_indices = queried_indices[sample_index][valid]
        candidate_distances = queried_distances[sample_index][valid]
        if candidate_indices.size < neighbor_count and query_count < sample_count:
            all_distances, all_indices = tree.query(
                embedding[sample_index],
                k=sample_count,
            )
            valid = (
                np.abs(times[all_indices] - times[sample_index])
                >= minimum_time_separation - 1e-12
            )
            candidate_indices = all_indices[valid]
            candidate_distances = all_distances[valid]
        if candidate_indices.size < neighbor_count:
            raise ValueError(
                "not enough cross-time neighbors for the declared exclusion"
            )
        selected_indices[sample_index] = candidate_indices[:neighbor_count]
        selected_distances[sample_index] = candidate_distances[:neighbor_count]
    return selected_indices, selected_distances


def _weighted_neighbor_prediction(
    target: np.ndarray,
    indices: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    predictions = np.empty_like(target)
    epsilon = 32.0 * np.finfo(float).eps
    for sample_index, (neighbors, neighbor_distances) in enumerate(
        zip(indices, distances, strict=True)
    ):
        coincident = neighbor_distances <= epsilon
        if np.any(coincident):
            predictions[sample_index] = np.mean(
                target[neighbors[coincident]],
                axis=0,
            )
            continue
        weights = 1.0 / neighbor_distances
        weights /= np.sum(weights)
        predictions[sample_index] = np.sum(
            weights[:, None] * target[neighbors],
            axis=0,
        )
    return predictions


def trajectory_source_predictability(
    times: np.ndarray,
    embedding: np.ndarray,
    target_source: np.ndarray,
    independent_target_source: np.ndarray,
    *,
    minimum_time_separation: float = 4.0,
    neighbor_count: int = 8,
    query_neighbor_count: int = 256,
    close_state_quantile: float = 0.1,
) -> TrajectoryClosurePredictability:
    """Measure cross-time source predictability from a declared embedding."""

    time_array = np.asarray(times, dtype=float)
    embedded = np.asarray(embedding, dtype=float)
    target = np.asarray(target_source, dtype=float)
    independent = np.asarray(independent_target_source, dtype=float)
    if time_array.ndim != 1 or time_array.size < 2:
        raise ValueError("times must be one-dimensional with at least two samples")
    if embedded.ndim != 3 or embedded.shape[1] != time_array.size:
        raise ValueError("embedding must have shape (members, times, features)")
    if target.ndim != 3 or target.shape[:2] != embedded.shape[:2]:
        raise ValueError("target_source must share member and time axes")
    if independent.shape != target.shape:
        raise ValueError("independent_target_source must match target_source")
    if not np.all(np.isfinite(embedded)):
        raise ValueError("embedding must be finite")
    if minimum_time_separation <= 0.0:
        raise ValueError("minimum_time_separation must be positive")
    if neighbor_count < 1:
        raise ValueError("neighbor_count must be positive")
    if not 0.0 < close_state_quantile <= 0.5:
        raise ValueError("close_state_quantile must lie in (0, 0.5]")

    member_count = embedded.shape[0]
    flat_times = np.tile(time_array, member_count)
    flat_embedding = embedded.reshape(-1, embedded.shape[-1])
    flat_target = target.reshape(-1, target.shape[-1])
    flat_independent = independent.reshape(-1, independent.shape[-1])
    neighbor_indices, neighbor_distances = _valid_neighbors(
        cKDTree(flat_embedding),
        flat_embedding,
        flat_times,
        minimum_time_separation=minimum_time_separation,
        neighbor_count=neighbor_count,
        query_neighbor_count=query_neighbor_count,
    )
    predictions = _weighted_neighbor_prediction(
        flat_target,
        neighbor_indices,
        neighbor_distances,
    )
    prediction_errors = np.linalg.norm(predictions - flat_target, axis=1)
    reference_uncertainties = np.linalg.norm(
        flat_target - flat_independent,
        axis=1,
    )
    centered_target = flat_target - np.mean(flat_target, axis=0)
    target_scale = float(
        np.sqrt(np.mean(np.sum(centered_target**2, axis=1)))
    )
    if target_scale <= np.finfo(float).tiny:
        raise ValueError("target_source has no resolved variation")

    nearest = neighbor_indices[:, 0]
    nearest_state = neighbor_distances[:, 0]
    nearest_target = np.linalg.norm(flat_target - flat_target[nearest], axis=1)
    nearest_uncertainty = (
        reference_uncertainties + reference_uncertainties[nearest]
    )
    close_threshold = float(np.quantile(nearest_state, close_state_quantile))
    close_samples = np.flatnonzero(nearest_state <= close_threshold + 1e-15)
    uncertainty_floor = max(
        float(np.sqrt(np.mean(reference_uncertainties**2))),
        np.finfo(float).tiny,
    )
    tension_scores = nearest_target[close_samples] / np.maximum(
        nearest_uncertainty[close_samples],
        uncertainty_floor,
    )
    tension_sample = int(close_samples[int(np.argmax(tension_scores))])
    return TrajectoryClosurePredictability(
        nearest_indices=nearest,
        nearest_state_distances=nearest_state,
        nearest_target_distances=nearest_target,
        nearest_reference_uncertainty_bounds=nearest_uncertainty,
        prediction_errors=prediction_errors,
        reference_uncertainties=reference_uncertainties,
        target_fluctuation_scale=target_scale,
        normalized_prediction_rms=float(
            np.sqrt(np.mean(prediction_errors**2)) / target_scale
        ),
        normalized_reference_rms=float(
            np.sqrt(np.mean(reference_uncertainties**2)) / target_scale
        ),
        minimum_time_separation=float(minimum_time_separation),
        neighbor_count=int(neighbor_count),
        close_state_quantile=float(close_state_quantile),
        close_state_threshold=close_threshold,
        tension_sample_index=tension_sample,
        tension_neighbor_index=int(nearest[tension_sample]),
    )


def causal_history_metric_embedding(
    coordinates: np.ndarray,
    scales: np.ndarray,
    *,
    lag_steps: int,
) -> np.ndarray:
    """Append a causal retained-state increment at one declared lag."""

    if lag_steps < 1:
        raise ValueError("lag_steps must be positive")
    current = closed_state_metric_embedding(coordinates, scales)
    if current.ndim != 3 or current.shape[1] <= lag_steps:
        raise ValueError("coordinates must have enough trajectory samples")
    retained = current[:, lag_steps:]
    increment = retained - current[:, :-lag_steps]
    return np.concatenate((retained, increment), axis=2) / np.sqrt(2.0)


def trajectory_closure_predictability(
    times: np.ndarray,
    coordinates: np.ndarray,
    target_source: np.ndarray,
    independent_target_source: np.ndarray,
    scales: np.ndarray,
    *,
    minimum_time_separation: float = 4.0,
    neighbor_count: int = 8,
    query_neighbor_count: int = 256,
    close_state_quantile: float = 0.1,
) -> TrajectoryClosurePredictability:
    """Measure cross-time predictability of a missing source from 31 moments.

    ``coordinates`` and both source arrays have leading shape
    ``(member, time, ...)``.  Neighbors inside the declared time exclusion are
    forbidden, preventing ordinary interpolation along one smooth trajectory
    from masquerading as closure identifiability.
    """

    time_array = np.asarray(times, dtype=float)
    states = np.asarray(coordinates, dtype=float)
    if states.ndim != 3 or states.shape[1:] != (time_array.size, 31):
        raise ValueError("coordinates must have shape (members, times, 31)")
    embedding = closed_state_metric_embedding(states, scales)
    return trajectory_source_predictability(
        time_array,
        embedding,
        target_source,
        independent_target_source,
        minimum_time_separation=minimum_time_separation,
        neighbor_count=neighbor_count,
        query_neighbor_count=query_neighbor_count,
        close_state_quantile=float(close_state_quantile),
    )


__all__ = [
    "TrajectoryClosurePredictability",
    "causal_history_metric_embedding",
    "closed_state_metric_embedding",
    "trajectory_closure_predictability",
    "trajectory_source_predictability",
]
