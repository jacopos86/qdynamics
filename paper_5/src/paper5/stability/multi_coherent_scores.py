"""Reference-only scores for frozen multi-coherent trajectories.

The functions in this module consume stored trajectories.  They do not expose
an integrator callback and therefore cannot affect packet spawning, tangent
regularization, or the model velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cone_correction import closed_state_lifted_frobenius_metric
from .matrix_reference import closed_scalar_to_matrix_state


CLOSED_COORDINATE_BLOCKS = {
    "rho": slice(0, 3),
    "B": slice(3, 7),
    "N": slice(7, 11),
    "A": slice(11, 17),
    "C": slice(17, 31),
}


@dataclass(frozen=True)
class ClosedCoordinateErrorScores:
    """Physical block errors over one declared score interval."""

    electron_trace_distance_maximum: float
    block_rms: dict[str, float]
    block_maximum: dict[str, float]


@dataclass(frozen=True)
class SensitivityAmplification:
    """Matched model and exact separation relative to one initial distance."""

    initial_distance: float
    model: np.ndarray
    exact: np.ndarray


@dataclass(frozen=True)
class BoundedScoreCertificate:
    """Numerical-resolution certificate for one score with a fixed ceiling."""

    authoritative: float
    uncertainty: float
    ceiling: float
    resolution_limit: float
    robust_upper_bound: float
    numerically_resolved: bool
    passes: bool


def _closed_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (31,):
        raise ValueError(f"{name} must have shape (31,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _coordinate_scales(scales: np.ndarray) -> np.ndarray:
    array = _closed_vector(scales, name="scales")
    if np.any(array <= 0.0):
        raise ValueError("all coordinate scales must be positive")
    return array


def _electron_trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_density = closed_scalar_to_matrix_state(left).electron_density
    right_density = closed_scalar_to_matrix_state(right).electron_density
    left_density = left_density / np.trace(left_density)
    right_density = right_density / np.trace(right_density)
    singular_values = np.linalg.svd(
        left_density - right_density,
        compute_uv=False,
    )
    return float(0.5 * np.sum(singular_values))


def closed_coordinate_distance(
    left: np.ndarray,
    right: np.ndarray,
    scales: np.ndarray,
) -> float:
    """Return the equal-five-block distance used for sensitivity scoring."""

    left_vector = _closed_vector(left, name="left")
    right_vector = _closed_vector(right, name="right")
    scale_vector = _coordinate_scales(scales)
    contributions = [
        _electron_trace_distance(left_vector, right_vector) ** 2
    ]
    for name in ("B", "N", "A", "C"):
        block = CLOSED_COORDINATE_BLOCKS[name]
        normalized = (
            left_vector[block] - right_vector[block]
        ) / scale_vector[block]
        contributions.append(float(np.mean(normalized**2)))
    return float(np.sqrt(np.mean(contributions)))


def closed_coordinate_error_scores(
    times: np.ndarray,
    model: np.ndarray,
    reference: np.ndarray,
    scales: np.ndarray,
    *,
    interval: tuple[float, float],
) -> ClosedCoordinateErrorScores:
    """Return trace-distance and scaled block errors on a fixed interval."""

    time_array = np.asarray(times, dtype=float)
    model_array = np.asarray(model, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    scale_vector = _coordinate_scales(scales)
    if time_array.ndim != 1 or time_array.size < 2:
        raise ValueError("times must contain at least two samples")
    if model_array.shape != (time_array.size, 31):
        raise ValueError("model must have shape (times, 31)")
    if reference_array.shape != model_array.shape:
        raise ValueError("reference must match model")
    if not np.all(np.diff(time_array) > 0.0):
        raise ValueError("times must be strictly increasing")
    start, stop = interval
    if not start < stop:
        raise ValueError("score interval must have positive duration")
    selected = (time_array >= start) & (time_array <= stop)
    selected_times = time_array[selected]
    if (
        selected_times.size < 2
        or not np.isclose(selected_times[0], start, atol=1e-12)
        or not np.isclose(selected_times[-1], stop, atol=1e-12)
    ):
        raise ValueError("score interval endpoints must be sampled")
    selected_model = model_array[selected]
    selected_reference = reference_array[selected]
    if not (
        np.all(np.isfinite(selected_model))
        and np.all(np.isfinite(selected_reference))
    ):
        raise ValueError("score trajectories must be finite")

    electron_distances = np.asarray(
        [
            _electron_trace_distance(left, right)
            for left, right in zip(
                selected_model,
                selected_reference,
                strict=True,
            )
        ]
    )
    block_rms: dict[str, float] = {}
    block_maximum: dict[str, float] = {}
    duration = stop - start
    for name in ("B", "N", "A", "C"):
        block = CLOSED_COORDINATE_BLOCKS[name]
        normalized = (
            selected_model[:, block] - selected_reference[:, block]
        ) / scale_vector[block]
        mean_square_by_time = np.mean(normalized**2, axis=1)
        block_rms[name] = float(
            np.sqrt(np.trapezoid(mean_square_by_time, selected_times) / duration)
        )
        block_maximum[name] = float(np.max(np.abs(normalized)))
    return ClosedCoordinateErrorScores(
        electron_trace_distance_maximum=float(np.max(electron_distances)),
        block_rms=block_rms,
        block_maximum=block_maximum,
    )


def development_coordinate_scales(
    reference: np.ndarray,
    *,
    phonon_cutoff: int,
) -> np.ndarray:
    """Freeze Eq. (7.5) scales from an already-open development path."""

    path = np.asarray(reference, dtype=float)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 31:
        raise ValueError("reference must have shape (at least two times, 31)")
    if not np.all(np.isfinite(path)):
        raise ValueError("reference must be finite")
    if phonon_cutoff < 1:
        raise ValueError("phonon_cutoff must be positive")
    weights = np.sqrt(np.diag(closed_state_lifted_frobenius_metric()))
    scales = np.ones(31, dtype=float)
    kinematic_bounds = {
        "B": 2.0 * np.sqrt(float(phonon_cutoff)),
        "N": 4.0 * float(phonon_cutoff),
        "A": 4.0 * float(phonon_cutoff),
        "C": 4.0 * np.sqrt(float(phonon_cutoff)),
    }
    for name, bound in kinematic_bounds.items():
        block = CLOSED_COORDINATE_BLOCKS[name]
        block_path = path[:, block]
        floor = 1e-3 * weights[block] * bound
        maximum_magnitude = np.max(np.abs(block_path), axis=0)
        excursion = np.ptp(block_path, axis=0)
        scales[block] = np.maximum.reduce(
            (floor, maximum_magnitude, excursion)
        )
    return scales


def energy_work_residual(
    times: np.ndarray,
    energies: np.ndarray,
    external_power: np.ndarray,
) -> np.ndarray:
    """Return ``E(t)-E(0)-integral(power dt)`` at every stored sample."""

    time_array = np.asarray(times, dtype=float)
    energy_array = np.asarray(energies, dtype=float)
    power_array = np.asarray(external_power, dtype=float)
    if time_array.ndim != 1 or time_array.size < 2:
        raise ValueError("times must contain at least two samples")
    if energy_array.shape != time_array.shape:
        raise ValueError("energies must match times")
    if power_array.shape != time_array.shape:
        raise ValueError("external_power must match times")
    if not np.all(np.diff(time_array) > 0.0):
        raise ValueError("times must be strictly increasing")
    if not (
        np.all(np.isfinite(time_array))
        and np.all(np.isfinite(energy_array))
        and np.all(np.isfinite(power_array))
    ):
        raise ValueError("work-balance inputs must be finite")
    increments = (
        0.5
        * (power_array[:-1] + power_array[1:])
        * np.diff(time_array)
    )
    integrated_work = np.concatenate(([0.0], np.cumsum(increments)))
    return energy_array - energy_array[0] - integrated_work


def sensitivity_amplification(
    model_plus: np.ndarray,
    model_minus: np.ndarray,
    exact_plus: np.ndarray,
    exact_minus: np.ndarray,
    scales: np.ndarray,
) -> SensitivityAmplification:
    """Return matched pair amplifications in the fixed five-block metric."""

    arrays = [
        np.asarray(value, dtype=float)
        for value in (model_plus, model_minus, exact_plus, exact_minus)
    ]
    if arrays[0].ndim != 2 or arrays[0].shape[1:] != (31,):
        raise ValueError("sensitivity trajectories must have shape (times, 31)")
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("all sensitivity trajectories must share one shape")
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("sensitivity trajectories must be finite")
    scale_vector = _coordinate_scales(scales)
    initial_model = closed_coordinate_distance(
        arrays[0][0],
        arrays[1][0],
        scale_vector,
    )
    initial_exact = closed_coordinate_distance(
        arrays[2][0],
        arrays[3][0],
        scale_vector,
    )
    if initial_model <= np.finfo(float).tiny:
        raise ValueError("initial sensitivity distance is unresolved")
    if not np.isclose(initial_model, initial_exact, rtol=1e-2, atol=0.0):
        raise ValueError("model and exact initial distances do not agree")
    model_distance = np.asarray(
        [
            closed_coordinate_distance(plus, minus, scale_vector)
            for plus, minus in zip(arrays[0], arrays[1], strict=True)
        ]
    )
    exact_distance = np.asarray(
        [
            closed_coordinate_distance(plus, minus, scale_vector)
            for plus, minus in zip(arrays[2], arrays[3], strict=True)
        ]
    )
    return SensitivityAmplification(
        initial_distance=initial_model,
        model=model_distance / initial_model,
        exact=exact_distance / initial_model,
    )


def score_resolution_uncertainty(scores: np.ndarray) -> float:
    """Return the full spread over frozen model/reference combinations."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("scores must contain at least two scalar values")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    return float(np.max(values) - np.min(values))


def pointwise_resolution_uncertainty(values: np.ndarray) -> np.ndarray:
    """Return the pointwise spread over frozen trajectory combinations."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(
            "values must have shape (at least two combinations, times)"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    return np.max(array, axis=0) - np.min(array, axis=0)


def certify_bounded_score(
    *,
    authoritative: float,
    cross_combination_scores: np.ndarray,
    ceiling: float,
    resolution_fraction: float = 0.1,
) -> BoundedScoreCertificate:
    """Apply ``u <= fraction * ceiling`` and ``Q + u <= ceiling``."""

    if not np.isfinite(authoritative) or authoritative < 0.0:
        raise ValueError("authoritative score must be finite and nonnegative")
    if not np.isfinite(ceiling) or ceiling <= 0.0:
        raise ValueError("ceiling must be finite and positive")
    if (
        not np.isfinite(resolution_fraction)
        or not 0.0 < resolution_fraction < 1.0
    ):
        raise ValueError(
            "resolution_fraction must lie strictly between zero and one"
        )
    uncertainty = score_resolution_uncertainty(cross_combination_scores)
    resolution_limit = resolution_fraction * ceiling
    robust_upper_bound = authoritative + uncertainty
    numerically_resolved = uncertainty <= resolution_limit
    return BoundedScoreCertificate(
        authoritative=float(authoritative),
        uncertainty=uncertainty,
        ceiling=float(ceiling),
        resolution_limit=resolution_limit,
        robust_upper_bound=robust_upper_bound,
        numerically_resolved=numerically_resolved,
        passes=numerically_resolved and robust_upper_bound <= ceiling,
    )


__all__ = [
    "BoundedScoreCertificate",
    "CLOSED_COORDINATE_BLOCKS",
    "ClosedCoordinateErrorScores",
    "SensitivityAmplification",
    "closed_coordinate_distance",
    "closed_coordinate_error_scores",
    "certify_bounded_score",
    "development_coordinate_scales",
    "energy_work_residual",
    "pointwise_resolution_uncertainty",
    "score_resolution_uncertainty",
    "sensitivity_amplification",
]
