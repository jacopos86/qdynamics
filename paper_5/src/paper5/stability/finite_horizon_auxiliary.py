"""Finite-horizon reachable--observable audit for reciprocal memory frames.

The module implements the grid-consistent square-root Hankel construction for
one frozen :class:`ArchiveAuxiliaryFrame`.  Exact or packet trajectories are
offline development inputs only.  The returned orthogonal frames contain no
trajectory values and can be frozen for a later autonomous rollout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh, expm, svd

from .archive_auxiliary_memory import ArchiveAuxiliaryFrame
from .krylov_memory_closure import (
    centered_jacobian_from_orthonormal,
    closed_coordinates_to_orthonormal,
)

FloatArray = NDArray[np.float64]


def _orthogonal_union(
    hidden_dimension: int,
    mandatory_dimension: int,
    primal: FloatArray,
    dual: FloatArray,
    *,
    relative_tolerance: float,
) -> FloatArray:
    mandatory = np.eye(hidden_dimension, mandatory_dimension)
    candidates = np.column_stack((mandatory, primal, dual))
    left, singular_values, _ = svd(candidates, full_matrices=False)
    if singular_values.size == 0:
        return np.empty((hidden_dimension, 0), dtype=float)
    keep = singular_values >= relative_tolerance * singular_values[0]
    return np.asarray(left[:, keep], dtype=float)


@dataclass(frozen=True)
class FiniteHorizonScenario:
    """One offline path used to construct causal hidden-state directions."""

    label: str
    times: FloatArray
    closed_coordinates: FloatArray
    drive_values: FloatArray
    initial_memory_coordinates: FloatArray


@dataclass(frozen=True)
class HankelSplitAudit:
    """Reachability, observability, and balanced directions at one split."""

    scenario_label: str
    split_index: int
    split_time: float
    reachability_rank: int
    observability_rank: int
    reachability_gramian: FloatArray
    observability_gramian: FloatArray
    hankel_singular_values: FloatArray
    primal_directions: FloatArray
    dual_directions: FloatArray

    def optimal_relative_defect(self, rank: int) -> float:
        """Return the unconstrained rank-``rank`` Hankel tail fraction."""

        if rank < 0:
            raise ValueError("rank must be nonnegative")
        values = self.hankel_singular_values
        denominator = float(np.sum(values * values))
        if denominator == 0.0 or rank >= values.size:
            return 0.0
        return float(np.sqrt(np.sum(values[rank:] ** 2) / denominator))

    def orthogonal_frame(
        self,
        *,
        hidden_dimension: int,
        mandatory_dimension: int,
        pair_count: int,
        relative_tolerance: float,
    ) -> FloatArray:
        """Return the local reciprocal union at this split."""

        if not 0 <= pair_count <= self.hankel_singular_values.size:
            raise ValueError("pair_count is outside this split's supported range")
        return _orthogonal_union(
            hidden_dimension,
            mandatory_dimension,
            self.primal_directions[:, :pair_count],
            self.dual_directions[:, :pair_count],
            relative_tolerance=relative_tolerance,
        )

    def projection_residuals(self, frame: FloatArray) -> tuple[float, float]:
        """Return root trace residuals for the actual orthogonal frame."""

        basis = np.asarray(frame, dtype=float)
        projector = basis @ basis.T
        identity = np.eye(projector.shape[0])

        def residual(gramian: FloatArray) -> float:
            total = float(np.trace(gramian))
            if total <= np.finfo(float).tiny:
                return 0.0
            omitted = float(np.trace((identity - projector) @ gramian))
            return float(np.sqrt(max(0.0, omitted / total)))

        return residual(self.reachability_gramian), residual(
            self.observability_gramian
        )


@dataclass(frozen=True)
class FiniteHorizonAuxiliaryAudit:
    """Complete split audit and aggregate orthogonal-frame constructor."""

    hidden_dimension: int
    mandatory_dimension: int
    relative_tolerance: float
    split_audits: tuple[HankelSplitAudit, ...]
    aggregate_hankel_singular_values: FloatArray
    aggregate_primal_directions: FloatArray
    aggregate_dual_directions: FloatArray

    @property
    def supported_pair_count(self) -> int:
        return int(self.aggregate_hankel_singular_values.size)

    def worst_optimal_relative_defect(self, rank: int) -> float:
        """Return the largest ideal Hankel tail over every audited split."""

        if not self.split_audits:
            return 0.0
        return max(audit.optimal_relative_defect(rank) for audit in self.split_audits)

    def orthogonal_frame(self, pair_count: int) -> FloatArray:
        """Return the reciprocal trial/test frame for one balanced pair count.

        The mandatory entrance directions are retained first.  Primal and dual
        balanced proposals are then combined and Hilbert--Schmidt orthogonalized.
        The actual returned order can be smaller than
        ``mandatory_dimension + 2 * pair_count`` because overlapping proposals
        are removed numerically.
        """

        if not 0 <= pair_count <= self.supported_pair_count:
            raise ValueError(
                "pair_count must lie between zero and the supported pair count"
            )
        return _orthogonal_union(
            self.hidden_dimension,
            self.mandatory_dimension,
            self.aggregate_primal_directions[:, :pair_count],
            self.aggregate_dual_directions[:, :pair_count],
            relative_tolerance=self.relative_tolerance,
        )

    def actual_order_curve(self) -> FloatArray:
        """Return actual orthogonal orders for all supported pair counts."""

        return np.asarray(
            [
                self.orthogonal_frame(pair_count).shape[1]
                for pair_count in range(self.supported_pair_count + 1)
            ],
            dtype=int,
        )


@dataclass(frozen=True)
class _ScenarioIntervals:
    steps: FloatArray
    half_step_maps: tuple[FloatArray, ...]
    input_factors: tuple[FloatArray, ...]
    output_matrices: tuple[FloatArray, ...]


def _require_scenario(
    scenario: FiniteHorizonScenario,
    *,
    hidden_dimension: int,
) -> None:
    times = np.asarray(scenario.times, dtype=float)
    closed = np.asarray(scenario.closed_coordinates, dtype=float)
    drives = np.asarray(scenario.drive_values, dtype=float)
    memory = np.asarray(scenario.initial_memory_coordinates, dtype=float)
    if times.ndim != 1 or times.size < 3:
        raise ValueError("scenario times must contain at least three samples")
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0.0):
        raise ValueError("scenario times must be finite and strictly increasing")
    if closed.shape != (times.size, 31):
        raise ValueError("scenario closed coordinates must have shape (time, 31)")
    if drives.shape != (times.size,):
        raise ValueError("scenario drive values must have shape (time,)")
    if memory.shape != (hidden_dimension,):
        raise ValueError("scenario initial memory has incompatible dimension")
    if not all(
        np.all(np.isfinite(values)) for values in (closed, drives, memory)
    ):
        raise ValueError("scenario arrays must be finite")


def _psd_square_root_factor(
    matrix: FloatArray,
    *,
    relative_tolerance: float,
) -> tuple[FloatArray, int]:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = eigh(symmetric)
    largest = max(float(eigenvalues[-1]), 0.0)
    if largest == 0.0:
        return np.empty((matrix.shape[0], 0), dtype=float), 0
    negative_tolerance = 100.0 * np.finfo(float).eps * largest
    if float(eigenvalues[0]) < -negative_tolerance:
        raise RuntimeError(
            "finite-horizon Gramian lost positive semidefiniteness: "
            f"minimum eigenvalue {eigenvalues[0]:.3e}"
        )
    keep = eigenvalues >= relative_tolerance * largest
    values = np.maximum(eigenvalues[keep], 0.0)
    factor = eigenvectors[:, keep] * np.sqrt(values)[None, :]
    return np.asarray(factor, dtype=float), int(np.count_nonzero(keep))


def _balanced_directions(
    reachability: FloatArray,
    observability: FloatArray,
    *,
    relative_tolerance: float,
) -> tuple[FloatArray, FloatArray, FloatArray, int, int]:
    reach_factor, reach_rank = _psd_square_root_factor(
        reachability,
        relative_tolerance=relative_tolerance**2,
    )
    observe_factor, observe_rank = _psd_square_root_factor(
        observability,
        relative_tolerance=relative_tolerance**2,
    )
    dimension = reachability.shape[0]
    if reach_rank == 0 or observe_rank == 0:
        empty = np.empty((dimension, 0), dtype=float)
        return np.empty(0, dtype=float), empty, empty, reach_rank, observe_rank
    left, singular_values, right_adjoint = svd(
        observe_factor.T @ reach_factor,
        full_matrices=False,
    )
    if singular_values.size == 0 or singular_values[0] == 0.0:
        empty = np.empty((dimension, 0), dtype=float)
        return np.empty(0, dtype=float), empty, empty, reach_rank, observe_rank
    keep = singular_values >= relative_tolerance * singular_values[0]
    values = singular_values[keep]
    inverse_root = 1.0 / np.sqrt(values)
    primal = (
        reach_factor @ right_adjoint.T[:, keep]
    ) * inverse_root[None, :]
    dual = (observe_factor @ left[:, keep]) * inverse_root[None, :]
    return (
        np.asarray(values, dtype=float),
        np.asarray(primal, dtype=float),
        np.asarray(dual, dtype=float),
        reach_rank,
        observe_rank,
    )


def hidden_output_matrix(
    frame: ArchiveAuxiliaryFrame,
    closed_coordinates: FloatArray,
    drive_value: float,
    coordinate_scales: FloatArray,
) -> FloatArray:
    """Return the scaled linear map from hidden coordinates to ``dot x``."""

    retained = closed_coordinates_to_orthonormal(
        frame.raw_basis,
        closed_coordinates,
    )
    jacobian = centered_jacobian_from_orthonormal(
        frame.raw_basis,
        retained,
    )
    coupling = frame.blocks(drive_value=drive_value).resolved_hidden
    return (jacobian @ coupling) / coordinate_scales[:, None]


def _scenario_intervals(
    frame: ArchiveAuxiliaryFrame,
    scenario: FiniteHorizonScenario,
    coordinate_scales: FloatArray,
) -> _ScenarioIntervals:
    times = np.asarray(scenario.times, dtype=float)
    closed = np.asarray(scenario.closed_coordinates, dtype=float)
    drives = np.asarray(scenario.drive_values, dtype=float)
    steps = np.diff(times)
    half_step_maps: list[FloatArray] = []
    input_factors: list[FloatArray] = []
    output_matrices: list[FloatArray] = []
    for interval, step in enumerate(steps):
        midpoint_drive = 0.5 * (drives[interval] + drives[interval + 1])
        midpoint_closed = 0.5 * (closed[interval] + closed[interval + 1])
        blocks = frame.blocks(drive_value=float(midpoint_drive))
        half_step_maps.append(expm(0.5 * float(step) * blocks.hidden_hidden))
        input_factors.append(blocks.hidden_resolved)
        output_matrices.append(
            hidden_output_matrix(
                frame,
                midpoint_closed,
                float(midpoint_drive),
                coordinate_scales,
            )
        )
    return _ScenarioIntervals(
        steps=np.asarray(steps, dtype=float),
        half_step_maps=tuple(half_step_maps),
        input_factors=tuple(input_factors),
        output_matrices=tuple(output_matrices),
    )


def _split_gramians(
    scenario: FiniteHorizonScenario,
    intervals: _ScenarioIntervals,
    split_index: int,
    *,
    preparation_weight: float,
) -> tuple[FloatArray, FloatArray]:
    hidden_dimension = scenario.initial_memory_coordinates.size
    reachability = np.zeros((hidden_dimension, hidden_dimension), dtype=float)
    observability = np.zeros_like(reachability)

    # Past-to-split reachability.  ``transition`` maps the right endpoint of
    # the current interval to the split time.
    transition = np.eye(hidden_dimension)
    for interval in range(split_index - 1, -1, -1):
        step = float(intervals.steps[interval])
        half_step = intervals.half_step_maps[interval]
        midpoint_transition = transition @ half_step
        input_factor = intervals.input_factors[interval]
        reached = midpoint_transition @ input_factor
        reachability += step * (reached @ reached.T)
        transition = transition @ half_step @ half_step
    preparation = (
        preparation_weight
        * transition
        @ np.asarray(scenario.initial_memory_coordinates, dtype=float)
    )
    reachability += np.outer(preparation, preparation)

    # Split-to-future observability.  ``transition`` maps the split state to
    # the left endpoint of the current interval.
    transition = np.eye(hidden_dimension)
    for interval in range(split_index, intervals.steps.size):
        step = float(intervals.steps[interval])
        half_step = intervals.half_step_maps[interval]
        midpoint_transition = half_step @ transition
        output = intervals.output_matrices[interval]
        observed = output @ midpoint_transition
        observability += step * (observed.T @ observed)
        transition = half_step @ half_step @ transition
    return reachability, observability


def finite_horizon_reachable_observable_audit(
    frame: ArchiveAuxiliaryFrame,
    scenarios: tuple[FiniteHorizonScenario, ...],
    coordinate_scales: FloatArray,
    *,
    split_times: tuple[float, ...],
    mandatory_dimension: int,
    relative_tolerance: float = 1e-10,
    preparation_weight: float = 1.0,
) -> FiniteHorizonAuxiliaryAudit:
    """Construct split and aggregate finite-horizon Hankel directions."""

    if not scenarios:
        raise ValueError("at least one development scenario is required")
    if not split_times:
        raise ValueError("at least one split time is required")
    if not 0 <= mandatory_dimension <= frame.hidden_dimension:
        raise ValueError("mandatory_dimension is outside the hidden frame")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie between zero and one")
    if preparation_weight < 0.0 or not np.isfinite(preparation_weight):
        raise ValueError("preparation_weight must be finite and nonnegative")
    scales = np.asarray(coordinate_scales, dtype=float)
    if scales.shape != (31,) or not np.all(np.isfinite(scales)):
        raise ValueError("coordinate_scales must have shape (31,)")
    if np.any(scales <= 0.0):
        raise ValueError("coordinate scales must be positive")

    split_audits: list[HankelSplitAudit] = []
    aggregate_reachability = np.zeros(
        (frame.hidden_dimension, frame.hidden_dimension),
        dtype=float,
    )
    aggregate_observability = np.zeros_like(aggregate_reachability)
    for scenario in scenarios:
        _require_scenario(scenario, hidden_dimension=frame.hidden_dimension)
        times = np.asarray(scenario.times, dtype=float)
        intervals = _scenario_intervals(frame, scenario, scales)
        for requested_time in split_times:
            split_index = int(np.argmin(np.abs(times - requested_time)))
            if split_index == 0 or split_index == times.size - 1:
                raise ValueError("split times must lie strictly inside each path")
            if not np.isclose(
                times[split_index],
                requested_time,
                atol=1e-10,
                rtol=0.0,
            ):
                raise ValueError(
                    f"scenario {scenario.label!r} does not sample split time "
                    f"{requested_time}"
                )
            reachability, observability = _split_gramians(
                scenario,
                intervals,
                split_index,
                preparation_weight=preparation_weight,
            )
            values, primal, dual, reach_rank, observe_rank = _balanced_directions(
                reachability,
                observability,
                relative_tolerance=relative_tolerance,
            )
            split_audits.append(
                HankelSplitAudit(
                    scenario_label=scenario.label,
                    split_index=split_index,
                    split_time=float(times[split_index]),
                    reachability_rank=reach_rank,
                    observability_rank=observe_rank,
                    reachability_gramian=reachability,
                    observability_gramian=observability,
                    hankel_singular_values=values,
                    primal_directions=primal,
                    dual_directions=dual,
                )
            )
            aggregate_reachability += reachability
            aggregate_observability += observability

    scale = 1.0 / float(len(split_audits))
    values, primal, dual, _, _ = _balanced_directions(
        scale * aggregate_reachability,
        scale * aggregate_observability,
        relative_tolerance=relative_tolerance,
    )
    return FiniteHorizonAuxiliaryAudit(
        hidden_dimension=frame.hidden_dimension,
        mandatory_dimension=mandatory_dimension,
        relative_tolerance=relative_tolerance,
        split_audits=tuple(split_audits),
        aggregate_hankel_singular_values=values,
        aggregate_primal_directions=primal,
        aggregate_dual_directions=dual,
    )
