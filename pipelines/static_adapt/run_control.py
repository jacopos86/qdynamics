"""Run-control helpers for static ADAPT segment execution.

This module owns segment caps and telemetry shaping only. It deliberately does
not know about scoring, optimizer dispatch, beam materialization, prune
semantics, or problem/pool construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_BENCHMARK_TARGET_HIT_STOP_REASON = "benchmark_abs_delta_e_target"


@dataclass(frozen=True)
class _AdaptSegmentControls:
    segment_id: str | None
    target_depth: int | None
    max_new_admissions: int | None
    wallclock_cap_s: float | None
    target_controller_round: int | None = None


@dataclass
class _AdaptSegmentRunState:
    controls: _AdaptSegmentControls
    start_depth: int
    start_runtime_parameter_count: int
    start_time_s: float
    max_depth_effective: int
    initial_stop_reason: str | None
    source_controller_round: int = 0
    new_admissions_count: int = 0
    cap_truncated_batch: bool = False


@dataclass(frozen=True)
class _AdaptSegmentBatchDecision:
    admit_count: int
    remaining_slots: int
    stop_reason: str | None
    truncated_records: list[dict[str, Any]] | None


def _benchmark_target_error_from_energy(
    *,
    energy_value: Any,
    reference_energy: float,
) -> float | None:
    try:
        energy_f = float(energy_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(energy_f):
        return None
    return float(abs(float(energy_f) - float(reference_energy)))


def _benchmark_target_hit_classification_payload(
    *,
    stop_reason_snapshot: str | None,
    target_error: float | None,
    target_threshold: float | None,
    source: str,
) -> dict[str, Any]:
    stop_key = str(stop_reason_snapshot or "").strip()
    threshold = None if target_threshold is None else float(target_threshold)
    target_configured = bool(threshold is not None and float(threshold) > 0.0)
    error_within_threshold = (
        bool(target_error is not None and threshold is not None and float(target_error) <= float(threshold))
        if target_configured
        else False
    )
    accepted_target_stop = bool(stop_key == _BENCHMARK_TARGET_HIT_STOP_REASON)
    target_hit_success = bool(target_configured and accepted_target_stop and error_within_threshold)
    if not target_configured:
        status = "target_not_requested"
        non_hit_reason = "benchmark_target_abs_delta_e_not_configured"
    elif target_hit_success:
        status = "target_hit_success"
        non_hit_reason = None
    elif accepted_target_stop:
        status = "inconsistent_target_stop_non_hit"
        non_hit_reason = "benchmark_target_stop_without_in_threshold_error"
    elif stop_key == "":
        status = "active_frontier_non_hit"
        non_hit_reason = "active_or_recoverable_frontier_without_benchmark_target_stop"
    else:
        status = "non_hit_diagnostic"
        non_hit_reason = f"terminal_stop_reason_not_target_hit:{stop_key}"
    return {
        "schema_version": "static_adapt_target_hit_classification_v1",
        "source": str(source),
        "target_hit_success": bool(target_hit_success),
        "status": str(status),
        "non_hit_reason": non_hit_reason,
        "terminal_stop_reason": (None if stop_key == "" else stop_key),
        "required_stop_reason": _BENCHMARK_TARGET_HIT_STOP_REASON,
        "target_configured": bool(target_configured),
        "target_error": (None if target_error is None else float(target_error)),
        "target_threshold": threshold,
        "target_error_within_threshold": bool(error_within_threshold),
        "target_error_within_threshold_without_target_stop": bool(
            target_configured and error_within_threshold and not accepted_target_stop
        ),
    }


def _resolve_adapt_segment_controls(
    *,
    adapt_segment_id: str | None,
    adapt_segment_target_depth: int | None,
    adapt_segment_max_new_admissions: int | None,
    adapt_segment_wallclock_cap_s: float | None,
    adapt_segment_target_controller_round: int | None = None,
) -> _AdaptSegmentControls:
    target_depth = None if adapt_segment_target_depth is None else int(adapt_segment_target_depth)
    if target_depth is not None and target_depth < 0:
        raise ValueError("adapt_segment_target_depth must be >= 0 when provided.")

    max_new_admissions = (
        None
        if adapt_segment_max_new_admissions is None
        else int(adapt_segment_max_new_admissions)
    )
    if max_new_admissions is not None and max_new_admissions < 0:
        raise ValueError("adapt_segment_max_new_admissions must be >= 0 when provided.")

    wallclock_cap_s = (
        None
        if adapt_segment_wallclock_cap_s is None
        else float(adapt_segment_wallclock_cap_s)
    )
    if (
        wallclock_cap_s is not None
        and ((not math.isfinite(wallclock_cap_s)) or wallclock_cap_s < 0.0)
    ):
        raise ValueError("adapt_segment_wallclock_cap_s must be finite and >= 0 when provided.")

    target_controller_round = (
        None
        if adapt_segment_target_controller_round is None
        else int(adapt_segment_target_controller_round)
    )
    if target_controller_round is not None and target_controller_round < 0:
        raise ValueError(
            "adapt_segment_target_controller_round must be >= 0 when provided."
        )

    return _AdaptSegmentControls(
        segment_id=(None if adapt_segment_id in {None, ""} else str(adapt_segment_id)),
        target_depth=target_depth,
        max_new_admissions=max_new_admissions,
        wallclock_cap_s=wallclock_cap_s,
        target_controller_round=target_controller_round,
    )


def _initialize_adapt_segment_run(
    *,
    controls: _AdaptSegmentControls,
    current_depth: int,
    current_runtime_parameter_count: int,
    requested_max_depth: int,
    start_time_s: float,
    source_controller_round: int = 0,
) -> _AdaptSegmentRunState:
    max_depth_effective = int(requested_max_depth)
    initial_stop_reason: str | None = None
    source_controller_round = int(source_controller_round)
    if source_controller_round < 0:
        raise ValueError("source_controller_round must be >= 0.")

    if controls.target_controller_round is not None:
        remaining_by_controller_round = (
            int(controls.target_controller_round) - int(source_controller_round)
        )
        if remaining_by_controller_round <= 0:
            max_depth_effective = 0
            initial_stop_reason = "segment_target_controller_round"
        else:
            max_depth_effective = min(
                int(max_depth_effective),
                int(remaining_by_controller_round),
            )

    if controls.target_depth is not None:
        remaining_by_target = int(controls.target_depth) - int(current_depth)
        if remaining_by_target <= 0:
            max_depth_effective = 0
            initial_stop_reason = initial_stop_reason or "segment_target_depth"
        else:
            max_depth_effective = min(int(max_depth_effective), int(remaining_by_target))

    if controls.max_new_admissions is not None:
        if int(controls.max_new_admissions) <= 0:
            max_depth_effective = 0
            initial_stop_reason = initial_stop_reason or "segment_max_new_admissions"
        else:
            max_depth_effective = min(int(max_depth_effective), int(controls.max_new_admissions))

    if controls.wallclock_cap_s is not None and float(controls.wallclock_cap_s) <= 0.0:
        max_depth_effective = 0
        initial_stop_reason = initial_stop_reason or "segment_wallclock_cap"

    return _AdaptSegmentRunState(
        controls=controls,
        start_depth=int(current_depth),
        start_runtime_parameter_count=int(current_runtime_parameter_count),
        start_time_s=float(start_time_s),
        max_depth_effective=int(max_depth_effective),
        initial_stop_reason=initial_stop_reason,
        source_controller_round=int(source_controller_round),
    )


def _adapt_segment_controls_resolved_log_payload(
    *,
    state: _AdaptSegmentRunState,
    requested_max_depth: int,
) -> dict[str, Any]:
    controls = state.controls
    return {
        "segment_id": controls.segment_id,
        "start_depth": int(state.start_depth),
        "requested_max_depth": int(requested_max_depth),
        "effective_max_depth": int(state.max_depth_effective),
        "target_depth": controls.target_depth,
        "source_controller_round": int(state.source_controller_round),
        "target_controller_round": controls.target_controller_round,
        "max_new_admissions": controls.max_new_admissions,
        "wallclock_cap_s": controls.wallclock_cap_s,
        "initial_stop_reason": state.initial_stop_reason,
    }


def _adapt_segment_loop_stop_reason(
    state: _AdaptSegmentRunState,
    *,
    current_depth: int,
    now_s: float,
    current_controller_round: int | None = None,
) -> str | None:
    controls = state.controls
    if controls.target_controller_round is not None:
        if current_controller_round is None:
            raise ValueError(
                "current_controller_round is required when a controller-round "
                "segment target is active."
            )
        if int(current_controller_round) >= int(controls.target_controller_round):
            return "segment_target_controller_round"
    if controls.target_depth is not None and int(current_depth) >= int(controls.target_depth):
        return "segment_target_depth"
    if (
        controls.max_new_admissions is not None
        and int(state.new_admissions_count) >= int(controls.max_new_admissions)
    ):
        return "segment_max_new_admissions"
    if (
        controls.wallclock_cap_s is not None
        and float(now_s) - float(state.start_time_s) >= float(controls.wallclock_cap_s)
    ):
        return "segment_wallclock_cap"
    return None


def _adapt_segment_controller_round(
    state: _AdaptSegmentRunState,
    *,
    segment_round_index: int,
) -> int:
    """Map a zero-based segment loop index to its cumulative controller round."""

    segment_round_index = int(segment_round_index)
    if segment_round_index < 0:
        raise ValueError("segment_round_index must be >= 0.")
    return int(state.source_controller_round) + segment_round_index + 1


def _resolve_adapt_segment_batch_decision(
    state: _AdaptSegmentRunState,
    *,
    records: Sequence[Mapping[str, Any]],
    current_depth: int,
) -> _AdaptSegmentBatchDecision:
    controls = state.controls
    remaining_slots = int(len(records))

    if controls.target_depth is not None:
        remaining_slots = min(
            int(remaining_slots),
            max(0, int(controls.target_depth) - int(current_depth)),
        )

    if controls.max_new_admissions is not None:
        remaining_slots = min(
            int(remaining_slots),
            max(0, int(controls.max_new_admissions) - int(state.new_admissions_count)),
        )

    if remaining_slots <= 0:
        stop_reason = (
            "segment_target_depth"
            if controls.target_depth is not None
            and int(current_depth) >= int(controls.target_depth)
            else "segment_max_new_admissions"
        )
        return _AdaptSegmentBatchDecision(
            admit_count=0,
            remaining_slots=int(remaining_slots),
            stop_reason=str(stop_reason),
            truncated_records=None,
        )

    if int(remaining_slots) < int(len(records)):
        truncated_records = [dict(row) for row in records[: int(remaining_slots)]]
    else:
        truncated_records = None

    return _AdaptSegmentBatchDecision(
        admit_count=int(remaining_slots),
        remaining_slots=int(remaining_slots),
        stop_reason=None,
        truncated_records=truncated_records,
    )


def _adapt_segment_history_fields(
    state: _AdaptSegmentRunState,
    *,
    selected_batch_label_count: int,
) -> dict[str, Any]:
    committed_new = int(selected_batch_label_count) if int(selected_batch_label_count) else 1
    state.new_admissions_count += int(committed_new)
    return {
        "segment_new_admissions_committed": int(committed_new),
        "segment_new_admissions_total": int(state.new_admissions_count),
        "segment_id": state.controls.segment_id,
        "segment_cap_truncated_batch": bool(state.cap_truncated_batch),
    }


def _sync_adapt_segment_new_admissions_from_depth(
    state: _AdaptSegmentRunState,
    *,
    final_depth: int,
    history_rows: Sequence[Mapping[str, Any]] | None = None,
    start_history_length: int = 0,
) -> None:
    depth_count = max(0, int(final_depth) - int(state.start_depth))
    if history_rows is None:
        state.new_admissions_count = int(depth_count)
        return

    history_count = 0
    for row in list(history_rows)[max(0, int(start_history_length)) :]:
        selected_ops = row.get("selected_ops")
        if isinstance(selected_ops, Sequence) and not isinstance(
            selected_ops, (str, bytes)
        ):
            selected_count = len(selected_ops)
        else:
            selected_count = int(row.get("selected_logical_size", 1) or 1)
        history_count += max(1, int(selected_count))
    state.new_admissions_count = max(int(depth_count), int(history_count))


def _build_adapt_segment_payload(
    *,
    state: _AdaptSegmentRunState,
    final_depth: int,
    final_runtime_parameter_count: int,
    new_admission_records: int,
    stop_reason: str,
    resume_enabled: bool,
    resume_mode: str,
    boundary_refit_executed: bool,
    compile_smoke_result: Mapping[str, Any] | None,
    final_controller_round: int | None = None,
) -> dict[str, Any]:
    controls = state.controls
    return {
        "schema_version": "static_hh_adapt_segment_v1",
        "segment_id": controls.segment_id,
        "resume_enabled": bool(resume_enabled),
        "resume_mode": str(resume_mode),
        "base_depth": int(state.start_depth),
        "base_runtime_parameter_count": int(state.start_runtime_parameter_count),
        "final_depth": int(final_depth),
        "final_runtime_parameter_count": int(final_runtime_parameter_count),
        "target_depth": None if controls.target_depth is None else int(controls.target_depth),
        "source_controller_round": int(state.source_controller_round),
        "target_controller_round": (
            None
            if controls.target_controller_round is None
            else int(controls.target_controller_round)
        ),
        "final_controller_round": (
            None if final_controller_round is None else int(final_controller_round)
        ),
        "max_new_admissions": (
            None if controls.max_new_admissions is None else int(controls.max_new_admissions)
        ),
        "wallclock_cap_s": (
            None if controls.wallclock_cap_s is None else float(controls.wallclock_cap_s)
        ),
        "new_admission_records": int(new_admission_records),
        "segment_cap_truncated_batch": bool(state.cap_truncated_batch),
        "stop_reason": str(stop_reason),
        "boundary_refit_executed": bool(boundary_refit_executed),
        "compile_smoke": (
            dict(compile_smoke_result)
            if isinstance(compile_smoke_result, Mapping)
            else None
        ),
        "no_credentials_serialized": True,
    }
