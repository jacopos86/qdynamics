"""Controller telemetry payload helpers for static ADAPT.

This module serializes controller/branch state only. It deliberately does not
own route identity, scoring, pruning, checkpoint writing, optimizer behavior,
or oracle/noise execution.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from pipelines.scaffold.hh_continuation_types import PhaseControllerSnapshot

__all__ = [
    "_branch_state_summary_payload",
    "_controller_snapshot_dict",
    "_controller_snapshot_payload",
    "_controller_telemetry_summary_payload",
]


def _controller_snapshot_dict(snapshot: Any | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "step_index": int(getattr(snapshot, "step_index", 0)),
        "depth_local": int(getattr(snapshot, "depth_local", 0)),
        "depth_left": int(getattr(snapshot, "depth_left", 0)),
        "runway_ratio": float(getattr(snapshot, "runway_ratio", 0.0)),
        "early_coordinate": float(getattr(snapshot, "early_coordinate", 0.0)),
        "late_coordinate": float(getattr(snapshot, "late_coordinate", 0.0)),
        "frontier_ratio": float(getattr(snapshot, "frontier_ratio", 1.0)),
        "u_stag": float(getattr(snapshot, "u_stag", 0.0)),
        "m_t": float(getattr(snapshot, "m_t", 0.0)),
        "s_t": float(getattr(snapshot, "s_t", 0.0)),
        "rho_t": float(getattr(snapshot, "rho_t", 1.0)),
        "gamma_t": float(getattr(snapshot, "gamma_t", 0.0)),
        "u_front": float(getattr(snapshot, "u_front", 1.0)),
        "n_rem_hat": float(getattr(snapshot, "n_rem_hat", 0.0)),
        "useful_horizon": float(getattr(snapshot, "useful_horizon", 0.0)),
        "runway_fraction": float(getattr(snapshot, "runway_fraction", 0.0)),
        "H_t": float(getattr(snapshot, "H_t", 0.0)),
        "phase_thresholds": dict(getattr(snapshot, "phase_thresholds", {})),
        "phase_caps": dict(getattr(snapshot, "phase_caps", {})),
        "phase_shots": dict(getattr(snapshot, "phase_shots", {})),
        "phase_uncertainty": dict(getattr(snapshot, "phase_uncertainty", {})),
        "snapshot_version": str(getattr(snapshot, "snapshot_version", "phase123_controller_v1")),
        "depth_runway_ratio": float(getattr(snapshot, "depth_runway_ratio", getattr(snapshot, "runway_ratio", 0.0))),
        "n_rem_low": float(getattr(snapshot, "n_rem_low", getattr(snapshot, "n_rem_hat", 0.0))),
        "n_rem_high": float(getattr(snapshot, "n_rem_high", getattr(snapshot, "n_rem_hat", 0.0))),
        "confidence_ratio": float(getattr(snapshot, "confidence_ratio", 0.0)),
        "phase_live": dict(getattr(snapshot, "phase_live", {})),
        "terminal_phase": int(getattr(snapshot, "terminal_phase", 3)),
        "phase_null_reasons": dict(getattr(snapshot, "phase_null_reasons", {})),
        "phase_null_streaks": dict(getattr(snapshot, "phase_null_streaks", {})),
        "phase_caps_scheduled": dict(getattr(snapshot, "phase_caps_scheduled", {})),
        "phase_shots_maturity_floor": dict(getattr(snapshot, "phase_shots_maturity_floor", {})),
        "phase_shots_scheduled": dict(getattr(snapshot, "phase_shots_scheduled", {})),
        "phase_shots_snr": dict(getattr(snapshot, "phase_shots_snr", {})),
        "phase_shots_effective": dict(getattr(snapshot, "phase_shots_effective", {})),
        "phase_shot_uplift": dict(getattr(snapshot, "phase_shot_uplift", {})),
        "phase_shot_fraction": dict(getattr(snapshot, "phase_shot_fraction", {})),
        "phase_signal": dict(getattr(snapshot, "phase_signal", {})),
        "phase_signal_floor": dict(getattr(snapshot, "phase_signal_floor", {})),
    }


def _controller_snapshot_payload(snapshot_raw: Any | None) -> dict[str, Any] | None:
    if not isinstance(snapshot_raw, PhaseControllerSnapshot):
        return None
    return {
        "snapshot_version": str(snapshot_raw.snapshot_version),
        "step_index": int(snapshot_raw.step_index),
        "depth_local": int(snapshot_raw.depth_local),
        "depth_left": int(snapshot_raw.depth_left),
        "runway_ratio": float(snapshot_raw.runway_ratio),
        "early_coordinate": float(snapshot_raw.early_coordinate),
        "late_coordinate": float(snapshot_raw.late_coordinate),
        "frontier_ratio": float(snapshot_raw.frontier_ratio),
        "u_stag": float(snapshot_raw.u_stag),
        "m_t": float(snapshot_raw.m_t),
        "s_t": float(snapshot_raw.s_t),
        "rho_t": float(snapshot_raw.rho_t),
        "gamma_t": float(snapshot_raw.gamma_t),
        "u_front": float(snapshot_raw.u_front),
        "n_rem_hat": float(snapshot_raw.n_rem_hat),
        "useful_horizon": float(snapshot_raw.useful_horizon),
        "runway_fraction": float(snapshot_raw.runway_fraction),
        "H_t": float(snapshot_raw.H_t),
        "phase_thresholds": {
            str(k): float(v) for k, v in dict(snapshot_raw.phase_thresholds).items()
        },
        "phase_caps": {
            str(k): int(v) for k, v in dict(snapshot_raw.phase_caps).items()
        },
        "phase_shots": {
            str(k): int(v) for k, v in dict(snapshot_raw.phase_shots).items()
        },
        "phase_uncertainty": {
            str(k): float(v) for k, v in dict(snapshot_raw.phase_uncertainty).items()
        },
        "depth_runway_ratio": float(getattr(snapshot_raw, "depth_runway_ratio", snapshot_raw.runway_ratio)),
        "n_rem_low": float(getattr(snapshot_raw, "n_rem_low", snapshot_raw.n_rem_hat)),
        "n_rem_high": float(getattr(snapshot_raw, "n_rem_high", snapshot_raw.n_rem_hat)),
        "confidence_ratio": float(getattr(snapshot_raw, "confidence_ratio", 0.0)),
        "phase_live": {
            str(k): bool(v) for k, v in dict(getattr(snapshot_raw, "phase_live", {})).items()
        },
        "terminal_phase": int(getattr(snapshot_raw, "terminal_phase", 3)),
        "phase_null_reasons": {
            str(k): str(v) for k, v in dict(getattr(snapshot_raw, "phase_null_reasons", {})).items()
        },
        "phase_null_streaks": {
            str(k): int(v) for k, v in dict(getattr(snapshot_raw, "phase_null_streaks", {})).items()
        },
        "phase_caps_scheduled": {
            str(k): int(v) for k, v in dict(getattr(snapshot_raw, "phase_caps_scheduled", {})).items()
        },
        "phase_shots_maturity_floor": {
            str(k): int(v) for k, v in dict(getattr(snapshot_raw, "phase_shots_maturity_floor", {})).items()
        },
        "phase_shots_scheduled": {
            str(k): int(v) for k, v in dict(getattr(snapshot_raw, "phase_shots_scheduled", {})).items()
        },
        "phase_shots_snr": {
            str(k): int(v) for k, v in dict(getattr(snapshot_raw, "phase_shots_snr", {})).items()
        },
        "phase_shots_effective": {
            str(k): int(v) for k, v in dict(getattr(snapshot_raw, "phase_shots_effective", {})).items()
        },
        "phase_shot_uplift": {
            str(k): float(v) for k, v in dict(getattr(snapshot_raw, "phase_shot_uplift", {})).items()
        },
        "phase_shot_fraction": {
            str(k): float(v) for k, v in dict(getattr(snapshot_raw, "phase_shot_fraction", {})).items()
        },
        "phase_signal": {
            str(k): float(v) for k, v in dict(getattr(snapshot_raw, "phase_signal", {})).items()
        },
        "phase_signal_floor": {
            str(k): float(v) for k, v in dict(getattr(snapshot_raw, "phase_signal_floor", {})).items()
        },
    }


def _controller_telemetry_summary_payload(
    *,
    stage_name: str | None,
    residual_opened: bool,
    last_probe_reason: str | None,
    stage_events: Sequence[Mapping[str, Any]] | None,
    last_snapshot: Any | None,
) -> dict[str, Any]:
    stage_rows = (
        [dict(row) for row in stage_events if isinstance(row, Mapping)]
        if isinstance(stage_events, Sequence)
        else []
    )
    return {
        "telemetry_label": "T_b^ctrl",
        "stage_name": (None if stage_name is None else str(stage_name)),
        "residual_opened": bool(residual_opened),
        "last_probe_reason": (None if last_probe_reason is None else str(last_probe_reason)),
        "stage_event_count": int(len(stage_rows)),
        "last_stage_event": (dict(stage_rows[-1]) if stage_rows else None),
        "last_snapshot": _controller_snapshot_payload(last_snapshot),
    }


def _branch_state_summary_payload(
    *,
    beam_enabled: bool,
    branch_id: int | None,
    parent_branch_id: int | None,
    history_rows: Sequence[Mapping[str, Any]] | None,
    depth_local: int,
    ansatz_depth: int,
    terminated: bool,
    termination_label: str | None,
    cumulative_selector_score: float,
    cumulative_selector_burden: float,
    stage_name: str | None,
    residual_opened: bool,
    last_probe_reason: str | None,
    stage_events: Sequence[Mapping[str, Any]] | None,
    last_snapshot: Any | None,
) -> dict[str, Any]:
    rows = (
        [dict(row) for row in history_rows if isinstance(row, Mapping)]
        if isinstance(history_rows, Sequence)
        else []
    )
    return {
        "branch_state_notation": "\\mathfrak b_*",
        "status": ("terminal" if bool(terminated) else "frontier"),
        "termination_label": (
            str(termination_label) if bool(terminated) and termination_label is not None else None
        ),
        "beam_enabled": bool(beam_enabled),
        "branch_id": (None if branch_id is None else int(branch_id)),
        "parent_branch_id": (
            None if parent_branch_id is None else int(parent_branch_id)
        ),
        "depth_local": int(depth_local),
        "history_step_count": int(len(rows)),
        "ansatz_depth": int(ansatz_depth),
        "cumulative_selector_score": float(cumulative_selector_score),
        "cumulative_selector_burden": float(cumulative_selector_burden),
        "controller_telemetry": _controller_telemetry_summary_payload(
            stage_name=stage_name,
            residual_opened=bool(residual_opened),
            last_probe_reason=last_probe_reason,
            stage_events=stage_events,
            last_snapshot=last_snapshot,
        ),
    }
