"""Offline viability audit for a smooth reciprocal auxiliary atlas.

The module consumes split-local finite-horizon factors and returns geometry,
capture, section, and normal-transport diagnostics.  It does not construct an
online atlas or propagate an autonomous model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svdvals

from .archive_auxiliary_memory import ArchiveAuxiliaryFrame, ArchiveField
from .finite_horizon_auxiliary import (
    FiniteHorizonAuxiliaryAudit,
    FiniteHorizonScenario,
    hidden_output_matrix,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LocalAtlasOrderAudit:
    """Diagnostics for one number of local balanced proposal pairs."""

    pair_count: int
    minimum_local_order: int
    maximum_local_order: int
    fixed_order_across_splits: bool
    worst_reachability_residual: float
    worst_observability_residual: float
    maximum_section_relative_residual: float
    maximum_input_leakage_ratio: float
    maximum_neighbor_principal_angle_degrees: float
    maximum_reference_principal_angle_degrees: float
    minimum_neighbor_blend_gap: float
    minimum_reference_overlap_eigenvalue: float
    maximum_projector_rate: float
    maximum_normal_transport_rate: float
    maximum_terminal_leakage_ratio: float
    maximum_output_visible_terminal_ratio: float


@dataclass(frozen=True)
class MovingFrameViabilityAudit:
    """Complete local-frame order curve without an autonomous rollout."""

    hidden_dimension: int
    mandatory_dimension: int
    scenario_labels: tuple[str, ...]
    orders: tuple[LocalAtlasOrderAudit, ...]

    @property
    def first_full_pair_count(self) -> int | None:
        for audit in self.orders:
            if audit.minimum_local_order == self.hidden_dimension:
                return audit.pair_count
        return None


def _maximum_principal_angle(left: FloatArray, right: FloatArray) -> float:
    singular_values = np.clip(svdvals(left.T @ right), 0.0, 1.0)
    if singular_values.size == 0:
        return 0.0
    return float(np.degrees(np.arccos(float(np.min(singular_values)))))


def _align_frame(reference: FloatArray, candidate: FloatArray) -> FloatArray:
    left, _, right_adjoint = np.linalg.svd(
        candidate.T @ reference,
        full_matrices=False,
    )
    return candidate @ (left @ right_adjoint)


def _neighbor_blend_gap(left: FloatArray, right: FloatArray) -> float:
    order = left.shape[1]
    if order == left.shape[0]:
        return 1.0
    projector = 0.5 * (left @ left.T + right @ right.T)
    values = np.linalg.eigvalsh(projector)
    descending = values[::-1]
    return float(max(0.0, descending[order - 1] - descending[order]))


def moving_frame_viability_audit(
    frame: ArchiveAuxiliaryFrame,
    finite_horizon: FiniteHorizonAuxiliaryAudit,
    scenarios: tuple[FiniteHorizonScenario, ...],
    coordinate_scales: FloatArray,
    archive_field: ArchiveField,
) -> MovingFrameViabilityAudit:
    """Evaluate every distinct local reciprocal-union order.

    Local frames use identical trial and test spaces.  Neighbor geometry is
    evaluated only when every audited split has the same order, as required by
    a fixed-rank smooth atlas.  Pair counts stop after every local frame spans
    the complete construction envelope; this is rank saturation, not a
    user-imposed dimension cap.
    """

    scenario_by_label = {scenario.label: scenario for scenario in scenarios}
    if len(scenario_by_label) != len(scenarios):
        raise ValueError("scenario labels must be unique")
    unknown = {
        split.scenario_label for split in finite_horizon.split_audits
    } - set(scenario_by_label)
    if unknown:
        raise ValueError(f"missing scenarios for split labels: {sorted(unknown)}")
    scales = np.asarray(coordinate_scales, dtype=float)
    if scales.shape != (31,) or np.any(scales <= 0.0):
        raise ValueError("coordinate_scales must be positive with shape (31,)")
    if not finite_horizon.split_audits:
        raise ValueError("finite_horizon must contain split-local audits")

    supported = min(
        split.hankel_singular_values.size
        for split in finite_horizon.split_audits
    )
    results: list[LocalAtlasOrderAudit] = []
    previous_order_signature: tuple[int, ...] | None = None
    for pair_count in range(supported + 1):
        local_frames = tuple(
            split.orthogonal_frame(
                hidden_dimension=frame.hidden_dimension,
                mandatory_dimension=finite_horizon.mandatory_dimension,
                pair_count=pair_count,
                relative_tolerance=finite_horizon.relative_tolerance,
            )
            for split in finite_horizon.split_audits
        )
        order_signature = tuple(local.shape[1] for local in local_frames)
        if order_signature == previous_order_signature:
            continue
        previous_order_signature = order_signature
        fixed_order = len(set(order_signature)) == 1

        reachability_residuals = []
        observability_residuals = []
        section_residuals = []
        input_leakage = []
        for split, local in zip(
            finite_horizon.split_audits,
            local_frames,
            strict=True,
        ):
            reachability, observability = split.projection_residuals(local)
            reachability_residuals.append(reachability)
            observability_residuals.append(observability)
            scenario = scenario_by_label[split.scenario_label]
            closed = np.asarray(
                scenario.closed_coordinates[split.split_index],
                dtype=float,
            )
            drive_value = float(scenario.drive_values[split.split_index])
            projected = frame.orthogonal_projection(local)
            certificate = projected.section(
                closed,
                archive_field(closed, drive_value),
                drive_value=drive_value,
                relative_tolerance=finite_horizon.relative_tolerance,
            )
            section_residuals.append(
                certificate.centered_section_relative_residual
            )
            blocks = frame.blocks(drive_value=drive_value)
            omitted_input = (
                np.eye(frame.hidden_dimension) - local @ local.T
            ) @ blocks.hidden_resolved
            input_leakage.append(
                float(
                    np.linalg.norm(omitted_input)
                    / max(
                        np.linalg.norm(blocks.hidden_resolved),
                        np.finfo(float).tiny,
                    )
                )
            )

        neighbor_angles: list[float] = []
        reference_angles: list[float] = []
        blend_gaps: list[float] = []
        reference_overlaps: list[float] = []
        projector_rates: list[float] = []
        normal_rates: list[float] = []
        terminal_leakage: list[float] = []
        output_terminal: list[float] = []
        if fixed_order:
            reference = local_frames[0]
            for local in local_frames:
                reference_angles.append(_maximum_principal_angle(reference, local))
                overlap_values = svdvals(reference.T @ local)
                reference_overlaps.append(
                    float(np.min(overlap_values) ** 2)
                    if overlap_values.size
                    else 1.0
                )

            for scenario in scenarios:
                indexed = [
                    (split, local)
                    for split, local in zip(
                        finite_horizon.split_audits,
                        local_frames,
                        strict=True,
                    )
                    if split.scenario_label == scenario.label
                ]
                indexed.sort(key=lambda item: item[0].split_time)
                for (left_split, left), (right_split, right) in zip(
                    indexed[:-1],
                    indexed[1:],
                    strict=True,
                ):
                    step = right_split.split_time - left_split.split_time
                    aligned = _align_frame(left, right)
                    frame_rate = (aligned - left) / step
                    projector = left @ left.T
                    normal = (
                        np.eye(frame.hidden_dimension) - projector
                    ) @ frame_rate
                    scenario_state = np.asarray(
                        scenario.closed_coordinates[left_split.split_index],
                        dtype=float,
                    )
                    drive_value = float(
                        scenario.drive_values[left_split.split_index]
                    )
                    blocks = frame.blocks(drive_value=drive_value)
                    generator_normal = (
                        np.eye(frame.hidden_dimension) - projector
                    ) @ (blocks.hidden_hidden @ left - frame_rate)
                    output = hidden_output_matrix(
                        frame,
                        scenario_state,
                        drive_value,
                        scales,
                    )

                    neighbor_angles.append(_maximum_principal_angle(left, right))
                    blend_gaps.append(_neighbor_blend_gap(left, right))
                    projector_rates.append(
                        float(
                            np.linalg.norm(right @ right.T - projector)
                            / step
                        )
                    )
                    normal_rates.append(float(np.linalg.norm(normal)))
                    terminal_leakage.append(
                        float(
                            np.linalg.norm(generator_normal)
                            / max(
                                np.linalg.norm(blocks.hidden_hidden @ left),
                                np.linalg.norm(frame_rate),
                                np.finfo(float).tiny,
                            )
                        )
                    )
                    output_terminal.append(
                        float(
                            np.linalg.norm(output @ generator_normal)
                            / max(
                                np.linalg.norm(output @ left),
                                np.finfo(float).tiny,
                            )
                        )
                    )

        results.append(
            LocalAtlasOrderAudit(
                pair_count=pair_count,
                minimum_local_order=min(order_signature),
                maximum_local_order=max(order_signature),
                fixed_order_across_splits=fixed_order,
                worst_reachability_residual=max(reachability_residuals),
                worst_observability_residual=max(observability_residuals),
                maximum_section_relative_residual=max(section_residuals),
                maximum_input_leakage_ratio=max(input_leakage),
                maximum_neighbor_principal_angle_degrees=max(
                    neighbor_angles,
                    default=float("nan"),
                ),
                maximum_reference_principal_angle_degrees=max(
                    reference_angles,
                    default=float("nan"),
                ),
                minimum_neighbor_blend_gap=min(
                    blend_gaps,
                    default=float("nan"),
                ),
                minimum_reference_overlap_eigenvalue=min(
                    reference_overlaps,
                    default=float("nan"),
                ),
                maximum_projector_rate=max(
                    projector_rates,
                    default=float("nan"),
                ),
                maximum_normal_transport_rate=max(
                    normal_rates,
                    default=float("nan"),
                ),
                maximum_terminal_leakage_ratio=max(
                    terminal_leakage,
                    default=float("nan"),
                ),
                maximum_output_visible_terminal_ratio=max(
                    output_terminal,
                    default=float("nan"),
                ),
            )
        )
        if min(order_signature) == frame.hidden_dimension:
            break

    return MovingFrameViabilityAudit(
        hidden_dimension=frame.hidden_dimension,
        mandatory_dimension=finite_horizon.mandatory_dimension,
        scenario_labels=tuple(scenario.label for scenario in scenarios),
        orders=tuple(results),
    )


__all__ = [
    "LocalAtlasOrderAudit",
    "MovingFrameViabilityAudit",
    "moving_frame_viability_audit",
]
