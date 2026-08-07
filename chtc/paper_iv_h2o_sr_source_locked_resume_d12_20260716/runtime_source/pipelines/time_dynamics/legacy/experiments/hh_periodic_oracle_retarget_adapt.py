#!/usr/bin/env python3
"""Periodic oracle-retargeted ADAPT/McLachlan diagnostic for HH realtime dynamics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.legacy.experiments.checkpoint_local_adapt import (
    CheckpointLocalAdaptRuntimeState,
    add_checkpoint_local_adapt_config_args,
    adapt_checkpoint_snapshot_with_state,
    checkpoint_local_adapt_config_from_args,
    frozen_fit_config_from_checkpoint_adapt_args,
)
from pipelines.time_dynamics.fixed_manifold.exact_fit import (
    _reference_energy_total_span_full_run,
)
from pipelines.time_dynamics.legacy.checkpoint_controller import (
    RuntimeTermCarrier,
)
from pipelines.time_dynamics.legacy.checkpoint_types import (
    dataclass_to_payload,
    make_checkpoint_context,
    normalize_reference_mode,
    normalize_realtime_controller_mode,
)
from pipelines.time_dynamics.legacy.checkpoint_exact_audit import (
    build_exact_audit_helper_for_controller,
    exact_v1_pre_action_snapshot,
)
from pipelines.time_dynamics.runners.hh_from_adapt_artifact import (
    _to_jsonable,
    build_controller_bundle_from_args,
    build_parser as build_realtime_parser,
)
from pipelines.time_dynamics.legacy.checkpoint_measurement import (
    DerivedGeometryMemo,
    ExactCheckpointValueCache,
)


@dataclass(frozen=True)
class PeriodicOracleRetargetConfig:
    period_checkpoints: int = 8
    explicit_checkpoints: tuple[int, ...] = ()
    include_initial: bool = True
    target_mode: str = "exact_time_state"


def _parse_int_tuple(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return ()
    text = str(raw).strip()
    if not text:
        return ()
    return tuple(int(chunk.strip()) for chunk in text.split(",") if chunk.strip())


def _retarget_config_from_args(args: argparse.Namespace) -> PeriodicOracleRetargetConfig:
    target_mode = str(getattr(args, "oracle_retarget_target_mode", "exact_time_state")).strip().lower()
    if target_mode != "exact_time_state":
        raise ValueError("v1 supports only --oracle-retarget-target-mode exact_time_state")
    return PeriodicOracleRetargetConfig(
        period_checkpoints=int(getattr(args, "oracle_retarget_period_checkpoints", 8)),
        explicit_checkpoints=tuple(
            int(x) for x in _parse_int_tuple(getattr(args, "oracle_retarget_checkpoints", ""))
        ),
        include_initial=bool(getattr(args, "oracle_retarget_include_initial", True)),
        target_mode=str(target_mode),
    )


def resolve_retarget_schedule(
    *,
    num_checkpoints: int,
    cfg: PeriodicOracleRetargetConfig,
) -> tuple[tuple[int, ...], dict[int, tuple[str, ...]]]:
    n = int(num_checkpoints)
    if n <= 0:
        raise ValueError("num_checkpoints must be positive")
    reasons: dict[int, set[str]] = {}

    def add(idx: int, reason: str) -> None:
        if idx < 0 or idx >= n:
            raise ValueError(f"retarget checkpoint {idx} out of range for {n} checkpoints")
        reasons.setdefault(int(idx), set()).add(str(reason))

    if int(cfg.period_checkpoints) > 0:
        for idx in range(0, n, int(cfg.period_checkpoints)):
            add(int(idx), "periodic")
    for idx in cfg.explicit_checkpoints:
        add(int(idx), "explicit")
    if bool(cfg.include_initial):
        add(0, "initial")
    elif 0 in reasons:
        reasons.pop(0, None)
    if not reasons:
        raise ValueError("resolved retarget schedule is empty")
    ordered = tuple(sorted(int(x) for x in reasons))
    payload_reasons = {
        int(idx): tuple(sorted(values, key=lambda item: (item != "initial", item)))
        for idx, values in reasons.items()
    }
    return ordered, payload_reasons


def _current_statevector(controller: Any) -> np.ndarray:
    return np.asarray(
        controller.current_executor.prepare_state(controller.current_theta, controller.replay_context.psi_ref),
        dtype=complex,
    ).reshape(-1)


def _compute_current_baseline_for_checkpoint(
    controller: Any,
    *,
    checkpoint_index: int,
    time_value: float,
    time_stop: float | None,
    step_hamiltonian: Any,
) -> dict[str, Any]:
    psi_current = _current_statevector(controller)
    checkpoint_ctx = make_checkpoint_context(
        checkpoint_index=int(checkpoint_index),
        time_start=float(time_value),
        time_stop=(None if time_stop is None else float(time_stop)),
        scaffold_labels=controller._current_scaffold_labels(),
        theta=controller.current_theta,
        psi=psi_current,
        logical_count=int(controller.current_layout.logical_parameter_count),
        runtime_count=int(controller.current_layout.runtime_parameter_count),
        resolved_family=str(controller.replay_context.family_info.get("resolved", "unknown")),
        grouping_mode=str(controller.cfg.grouping_mode),
        structure_locked=False,
    )
    cache = ExactCheckpointValueCache(
        checkpoint_id=str(checkpoint_ctx.checkpoint_id),
        grouping_mode=str(controller.cfg.grouping_mode),
    )
    geometry_memo = DerivedGeometryMemo(checkpoint_id=str(checkpoint_ctx.checkpoint_id))
    baseline = controller._baseline_geometry(
        checkpoint_ctx,
        cache,
        geometry_memo,
        step_hamiltonian=step_hamiltonian,
    )
    return {
        "checkpoint_context": checkpoint_ctx,
        "baseline": baseline,
        "exact_cache_summary": dict(cache.summary()),
        "geometry_memo_summary": dict(geometry_memo.summary()),
    }


def _apply_retarget_state_to_controller(
    controller: Any,
    state: CheckpointLocalAdaptRuntimeState,
    *,
    checkpoint_index: int,
) -> None:
    terms = list(state.terms)
    if not all(isinstance(term, RuntimeTermCarrier) for term in terms):
        raise TypeError("Periodic retarget state must preserve RuntimeTermCarrier scaffold terms.")
    theta = np.asarray(state.theta_runtime, dtype=float).reshape(-1)
    if int(theta.size) != int(state.layout.runtime_parameter_count):
        raise ValueError("Retarget theta length does not match returned layout runtime count.")
    if int(len(terms)) != int(state.layout.logical_parameter_count):
        raise ValueError("Retarget term count does not match returned layout logical count.")
    controller.current_terms = list(terms)
    controller.current_layout = state.layout
    controller.current_executor = state.executor
    controller.current_theta = np.asarray(theta, dtype=float).copy()
    controller._planning_audit = controller._build_planning_audit_for_terms(controller.current_terms)
    controller._previous_theta_dot = None
    controller._theta_dot_history = []
    controller._previous_append_position = None
    for name in (
        "_block_birth_checkpoint",
        "_block_cooldown",
        "_block_burden",
        "_block_motion_history",
        "_block_fit_history",
        "_previous_block_theta_snapshot",
    ):
        value = getattr(controller, name, None)
        if isinstance(value, dict):
            value.clear()
    controller._initialize_prune_state()
    for label in controller._current_scaffold_labels():
        controller._block_birth_checkpoint[str(label)] = int(checkpoint_index)


def _snapshot_current_controller(
    *,
    controller: Any,
    exact_helper: Any,
    checkpoint_index: int,
    reference_energy_total_span: float,
) -> dict[str, Any]:
    snapshot = exact_v1_pre_action_snapshot(
        controller,
        exact_helper,
        checkpoint_index=int(checkpoint_index),
    )
    snapshot["prefix_force_stay_checkpoints"] = []
    snapshot["reference_energy_total_span_full_run"] = float(reference_energy_total_span)
    return snapshot


def _primary_density_from_snapshot(controller: Any, obs: Mapping[str, Any]) -> float:
    return float(controller._primary_density_value_from_snapshot(obs))


def _baseline_summary_payload(baseline_payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = baseline_payload.get("summary")
    return dict(dataclass_to_payload(summary)) if summary is not None else {}


def _trajectory_row(
    *,
    controller: Any,
    checkpoint_index: int,
    time_value: float,
    time_stop: float | None,
    step_hamiltonian: Any,
    action_kind: str,
    snapshot_pre: Mapping[str, Any],
    snapshot_post: Mapping[str, Any],
    baseline_pre: Mapping[str, Any],
    baseline_post: Mapping[str, Any],
    baseline_meta_pre: Mapping[str, Any],
    baseline_meta_post: Mapping[str, Any],
    motion: Any,
    predicted_displacement: float,
    retarget_event_index: int | None,
    schedule_reason: Sequence[str],
    retarget_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    post_obs = dict(snapshot_post.get("current_observables", {}))
    exact_obs = dict(snapshot_post.get("exact_observables", {}))
    primary_controller = _primary_density_from_snapshot(controller, post_obs)
    primary_exact = _primary_density_from_snapshot(controller, exact_obs)
    baseline_summary = _baseline_summary_payload(baseline_post)
    baseline_summary_pre = _baseline_summary_payload(baseline_pre)
    theta_dot = np.asarray(baseline_post.get("theta_dot_step", np.zeros(0)), dtype=float).reshape(-1)
    operators_added = 0 if retarget_payload is None else int(retarget_payload.get("operators_added", 0))
    return {
        "checkpoint_index": int(checkpoint_index),
        "time": float(time_value),
        "time_stop": (None if time_stop is None else float(time_stop)),
        "physical_time": float(getattr(step_hamiltonian, "physical_time", float(time_value))),
        "drive_term_count": int(getattr(step_hamiltonian, "drive_term_count", 0)),
        "action_kind": str(action_kind),
        "proposed_action_kind": str(action_kind),
        "retarget_applied": bool(retarget_payload is not None),
        "retarget_event_index": retarget_event_index,
        "retarget_schedule_reason": [str(x) for x in schedule_reason],
        "target_mode": "exact_time_state",
        "checkpoint_adapt_stop_reason": (
            None if retarget_payload is None else str(retarget_payload.get("stop_reason", "unknown"))
        ),
        "checkpoint_adapt_fit_objective_used": (
            None if retarget_payload is None else str(retarget_payload.get("fit_objective_used", "unknown"))
        ),
        "operators_added": int(operators_added),
        "logical_block_count_before_retarget": int(snapshot_pre.get("logical_block_count", 0)),
        "runtime_parameter_count_before_retarget": int(snapshot_pre.get("runtime_parameter_count", 0)),
        "logical_block_count_after_retarget": int(snapshot_post.get("logical_block_count", 0)),
        "runtime_parameter_count_after_retarget": int(snapshot_post.get("runtime_parameter_count", 0)),
        "logical_block_count": int(snapshot_post.get("logical_block_count", 0)),
        "runtime_parameter_count": int(snapshot_post.get("runtime_parameter_count", 0)),
        "scaffold_labels": [str(x) for x in snapshot_post.get("scaffold_labels", [])],
        "final_scaffold_labels": [str(x) for x in snapshot_post.get("scaffold_labels", [])],
        "fidelity_exact_pre_retarget": float(snapshot_pre.get("fidelity_exact", float("nan"))),
        "fidelity_exact_post_retarget": float(snapshot_post.get("fidelity_exact", float("nan"))),
        "fidelity_exact": float(snapshot_post.get("fidelity_exact", float("nan"))),
        "energy_total_controller": float(snapshot_post.get("energy_current", float("nan"))),
        "energy_total_exact": float(snapshot_post.get("energy_exact", float("nan"))),
        "abs_energy_total_error_pre_retarget": float(snapshot_pre.get("abs_energy_total_error", float("nan"))),
        "abs_energy_total_error_post_retarget": float(snapshot_post.get("abs_energy_total_error", float("nan"))),
        "abs_energy_total_error": float(snapshot_post.get("abs_energy_total_error", float("nan"))),
        "site_occupations_controller": [float(x) for x in post_obs.get("site_occupations", [])],
        "site_occupations_exact": [float(x) for x in exact_obs.get("site_occupations", [])],
        "site_occupations_up_controller": [float(x) for x in post_obs.get("n_up_site", [])],
        "site_occupations_up_exact": [float(x) for x in exact_obs.get("n_up_site", [])],
        "site_occupations_dn_controller": [float(x) for x in post_obs.get("n_dn_site", [])],
        "site_occupations_dn_exact": [float(x) for x in exact_obs.get("n_dn_site", [])],
        "site_occupations_abs_error_max_pre_retarget": float(snapshot_pre.get("site_occupations_abs_error_max", float("nan"))),
        "site_occupations_abs_error_max_post_retarget": float(snapshot_post.get("site_occupations_abs_error_max", float("nan"))),
        "site_occupations_abs_error_max": float(snapshot_post.get("site_occupations_abs_error_max", float("nan"))),
        "primary_density_controller": float(primary_controller),
        "primary_density_exact": float(primary_exact),
        "abs_primary_density_error": float(abs(primary_controller - primary_exact)),
        "doublon_controller": float(post_obs.get("doublon", float("nan"))),
        "doublon_exact": float(exact_obs.get("doublon", float("nan"))),
        "abs_doublon_error": float(abs(float(post_obs.get("doublon", float("nan"))) - float(exact_obs.get("doublon", float("nan"))))),
        "staggered_controller": float(post_obs.get("staggered", float("nan"))),
        "staggered_exact": float(exact_obs.get("staggered", float("nan"))),
        "abs_staggered_error": float(abs(float(post_obs.get("staggered", float("nan"))) - float(exact_obs.get("staggered", float("nan"))))),
        "rho_miss_pre_retarget": float(baseline_summary_pre.get("rho_miss", float("nan"))),
        "rho_miss_post_retarget": float(baseline_summary.get("rho_miss", float("nan"))),
        "rho_miss": float(baseline_summary.get("rho_miss", float("nan"))),
        "theta_dot_l2": float(np.linalg.norm(theta_dot)),
        "predicted_displacement": float(predicted_displacement),
        "motion_regime": str(getattr(motion, "regime", "unknown")),
        "motion_direction_cosine": getattr(motion, "direction_cosine", None),
        "motion_rate_change_ratio": getattr(motion, "rate_change_ratio", None),
        "motion_acceleration_l2": getattr(motion, "acceleration_l2", None),
        "motion_curvature_cosine": getattr(motion, "curvature_cosine", None),
        "motion_direction_reversal": bool(getattr(motion, "direction_reversal", False)),
        "motion_curvature_sign_flip": bool(getattr(motion, "curvature_sign_flip", False)),
        "motion_kink_score": float(getattr(motion, "kink_score", 0.0)),
        "baseline_geometry": dict(baseline_summary),
        "baseline_geometry_pre_retarget": dict(baseline_summary_pre),
        "exact_cache_hits": int(baseline_meta_post.get("exact_cache_summary", {}).get("hits", 0)),
        "exact_cache_misses": int(baseline_meta_post.get("exact_cache_summary", {}).get("misses", 0)),
        "geometry_memo_hits": int(baseline_meta_post.get("geometry_memo_summary", {}).get("hits", 0)),
        "geometry_memo_misses": int(baseline_meta_post.get("geometry_memo_summary", {}).get("misses", 0)),
        "exact_cache_hits_pre_retarget": int(baseline_meta_pre.get("exact_cache_summary", {}).get("hits", 0)),
        "exact_cache_misses_pre_retarget": int(baseline_meta_pre.get("exact_cache_summary", {}).get("misses", 0)),
        "requested_mode": "exact_v1",
        "decision_backend": "oracle_retarget_diagnostic",
        "selection_metric": "scheduled_exact_state_fidelity_retarget",
        "forecast_mode": "disabled_live_decision_bypass",
        "secant_decision_authority": False,
    }


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            out.append(float(value))
    return out


def _summary_from_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    retarget_events: Sequence[Mapping[str, Any]],
    schedule: Sequence[int],
    exact_reference_cache: Mapping[str, object],
    reference_energy_total_span: float,
    controller: Any,
) -> dict[str, Any]:
    fidelity = _finite_values(rows, "fidelity_exact")
    energy_err = _finite_values(rows, "abs_energy_total_error")
    site_err = _finite_values(rows, "site_occupations_abs_error_max")
    rho_pre = _finite_values(rows, "rho_miss_pre_retarget")
    rho_post = _finite_values(rows, "rho_miss_post_retarget")
    final = dict(rows[-1]) if rows else {}
    return {
        "status": "completed",
        "mode": "periodic_oracle_retarget_adapt",
        "target_mode": "exact_time_state",
        "propagation_mode": "fixed_scaffold_mclachlan_stay_only",
        "retarget_count": int(len(retarget_events)),
        "retarget_checkpoints": [int(x) for x in schedule],
        "total_operators_added": int(sum(int(evt.get("operators_added", 0)) for evt in retarget_events)),
        "stay_count": int(sum(1 for row in rows if str(row.get("action_kind")) == "stay")),
        "oracle_retarget_count": int(sum(1 for row in rows if str(row.get("action_kind")) == "oracle_retarget")),
        "final_logical_block_count": int(final.get("logical_block_count", controller.current_layout.logical_parameter_count if rows else 0)),
        "final_runtime_parameter_count": int(final.get("runtime_parameter_count", controller.current_layout.runtime_parameter_count if rows else 0)),
        "final_fidelity_exact": (None if not fidelity else float(fidelity[-1])),
        "min_fidelity_exact": (None if not fidelity else float(min(fidelity))),
        "max_fidelity_exact": (None if not fidelity else float(max(fidelity))),
        "final_abs_energy_total_error": (None if not energy_err else float(energy_err[-1])),
        "max_abs_energy_total_error": (None if not energy_err else float(max(energy_err))),
        "final_site_occupations_abs_error_max": (None if not site_err else float(site_err[-1])),
        "max_abs_site_occupations_error": (None if not site_err else float(max(site_err))),
        "max_rho_miss_pre_retarget": (None if not rho_pre else float(max(rho_pre))),
        "max_rho_miss_post_retarget": (None if not rho_post else float(max(rho_post))),
        "exact_reference_cache_entries": int(len(exact_reference_cache)),
        "reference_energy_total_span_full_run": float(reference_energy_total_span),
        "secant_decision_authority": False,
    }


def run_periodic_oracle_retarget_from_args(args: argparse.Namespace) -> dict[str, Any]:
    mode = normalize_realtime_controller_mode(getattr(args, "checkpoint_controller_mode", "exact_v1"))
    if mode != "exact_v1":
        raise ValueError("periodic oracle retarget diagnostic requires checkpoint_controller_mode=exact_v1")
    reference_mode = normalize_reference_mode(getattr(args, "checkpoint_controller_reference_mode", "benchmark_exact"))
    if reference_mode != "benchmark_exact":
        raise ValueError("periodic oracle retarget diagnostic requires benchmark_exact reference routing")

    output_json = Path(args.output_json).expanduser().resolve()
    retarget_cfg = _retarget_config_from_args(args)
    adapt_cfg = checkpoint_local_adapt_config_from_args(args)
    fit_cfg = frozen_fit_config_from_checkpoint_adapt_args(args, adapt_cfg=adapt_cfg)
    exact_reference_cache: dict[str, object] = {}
    bundle = build_controller_bundle_from_args(args, exact_reference_cache=exact_reference_cache)
    loaded = bundle["loaded"]
    controller = bundle["controller"]
    exact_helper = bundle.get("exact_helper") or build_exact_audit_helper_for_controller(
        controller,
        exact_reference_cache=exact_reference_cache,
    )
    exact_helper.ensure_ready()
    reference_energy_total_span = _reference_energy_total_span_full_run(controller, exact_helper)
    times = np.asarray(controller.times, dtype=float).reshape(-1)
    schedule, schedule_reasons = resolve_retarget_schedule(
        num_checkpoints=int(times.size),
        cfg=retarget_cfg,
    )
    schedule_set = {int(x) for x in schedule}

    trajectory: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    retarget_events: list[dict[str, Any]] = []
    try:
        for checkpoint_index, time_value_raw in enumerate(times.tolist()):
            time_value = float(time_value_raw)
            time_stop = None if int(checkpoint_index) + 1 >= int(times.size) else float(times[int(checkpoint_index) + 1])
            step_sample_time = float(controller._projection_sample_time(time_value, time_stop))
            step_hamiltonian = controller._step_hamiltonian_artifacts(step_sample_time)
            snapshot_pre = _snapshot_current_controller(
                controller=controller,
                exact_helper=exact_helper,
                checkpoint_index=int(checkpoint_index),
                reference_energy_total_span=float(reference_energy_total_span),
            )
            baseline_meta_pre = _compute_current_baseline_for_checkpoint(
                controller,
                checkpoint_index=int(checkpoint_index),
                time_value=float(time_value),
                time_stop=time_stop,
                step_hamiltonian=step_hamiltonian,
            )
            baseline_pre = dict(baseline_meta_pre["baseline"])
            retarget_payload: Mapping[str, Any] | None = None
            retarget_event_index: int | None = None
            action_kind = "stay"
            schedule_reason = tuple(schedule_reasons.get(int(checkpoint_index), ()))
            if int(checkpoint_index) in schedule_set:
                result = adapt_checkpoint_snapshot_with_state(
                    snapshot_pre,
                    bundle=bundle,
                    adapt_cfg=adapt_cfg,
                    fit_cfg=fit_cfg,
                )
                retarget_payload = dict(result.payload)
                retarget_event_index = int(len(retarget_events))
                _apply_retarget_state_to_controller(
                    controller,
                    result.state,
                    checkpoint_index=int(checkpoint_index),
                )
                action_kind = "oracle_retarget"
                snapshot_post = _snapshot_current_controller(
                    controller=controller,
                    exact_helper=exact_helper,
                    checkpoint_index=int(checkpoint_index),
                    reference_energy_total_span=float(reference_energy_total_span),
                )
                baseline_meta_post = _compute_current_baseline_for_checkpoint(
                    controller,
                    checkpoint_index=int(checkpoint_index),
                    time_value=float(time_value),
                    time_stop=time_stop,
                    step_hamiltonian=step_hamiltonian,
                )
                retarget_events.append(
                    {
                        "event_index": int(retarget_event_index),
                        "checkpoint_index": int(checkpoint_index),
                        "time": float(time_value),
                        "physical_time": float(step_hamiltonian.physical_time),
                        "schedule_reason": [str(x) for x in schedule_reason],
                        "operators_added": int(retarget_payload.get("operators_added", 0)),
                        "logical_block_count_before": int(snapshot_pre.get("logical_block_count", 0)),
                        "logical_block_count_after": int(snapshot_post.get("logical_block_count", 0)),
                        "runtime_parameter_count_before": int(snapshot_pre.get("runtime_parameter_count", 0)),
                        "runtime_parameter_count_after": int(snapshot_post.get("runtime_parameter_count", 0)),
                        "pre_metrics": {
                            "fidelity_exact": float(snapshot_pre.get("fidelity_exact", float("nan"))),
                            "abs_energy_total_error": float(snapshot_pre.get("abs_energy_total_error", float("nan"))),
                            "site_occupations_abs_error_max": float(snapshot_pre.get("site_occupations_abs_error_max", float("nan"))),
                        },
                        "post_metrics": {
                            "fidelity_exact": float(snapshot_post.get("fidelity_exact", float("nan"))),
                            "abs_energy_total_error": float(snapshot_post.get("abs_energy_total_error", float("nan"))),
                            "site_occupations_abs_error_max": float(snapshot_post.get("site_occupations_abs_error_max", float("nan"))),
                        },
                        "local_adapt": dict(retarget_payload),
                    }
                )
            else:
                snapshot_post = snapshot_pre
                baseline_meta_post = baseline_meta_pre
            baseline_post = dict(baseline_meta_post["baseline"])
            dt = 0.0 if time_stop is None else float(time_stop - time_value)
            predicted_displacement = controller._predicted_displacement(
                dt=float(dt),
                baseline=baseline_post,
            )
            theta_dot = np.asarray(baseline_post.get("theta_dot_step", np.zeros(0)), dtype=float).reshape(-1)
            motion = controller._motion_telemetry(
                theta_dot=theta_dot,
                predicted_displacement=float(predicted_displacement),
            )
            row = _trajectory_row(
                controller=controller,
                checkpoint_index=int(checkpoint_index),
                time_value=float(time_value),
                time_stop=time_stop,
                step_hamiltonian=step_hamiltonian,
                action_kind=str(action_kind),
                snapshot_pre=snapshot_pre,
                snapshot_post=snapshot_post,
                baseline_pre=baseline_pre,
                baseline_post=baseline_post,
                baseline_meta_pre=baseline_meta_pre,
                baseline_meta_post=baseline_meta_post,
                motion=motion,
                predicted_displacement=float(predicted_displacement),
                retarget_event_index=retarget_event_index,
                schedule_reason=schedule_reason,
                retarget_payload=retarget_payload,
            )
            trajectory.append(row)
            ledger.append(
                {
                    "checkpoint_index": int(checkpoint_index),
                    "time": float(time_value),
                    "physical_time": float(step_hamiltonian.physical_time),
                    "action_kind": str(action_kind),
                    "rho_miss": float(row["rho_miss"]),
                    "fidelity_exact": float(row["fidelity_exact"]),
                    "abs_energy_total_error": float(row["abs_energy_total_error"]),
                    "logical_block_count_before": int(row["logical_block_count_before_retarget"]),
                    "logical_block_count_after": int(row["logical_block_count_after_retarget"]),
                    "runtime_parameter_count_before": int(row["runtime_parameter_count_before_retarget"]),
                    "runtime_parameter_count_after": int(row["runtime_parameter_count_after_retarget"]),
                    "retarget_applied": bool(row["retarget_applied"]),
                    "operators_added": int(row["operators_added"]),
                }
            )
            if time_stop is not None:
                controller.current_theta = np.asarray(
                    np.asarray(controller.current_theta, dtype=float).reshape(-1)
                    + float(dt) * theta_dot,
                    dtype=float,
                ).reshape(-1)
                controller._record_theta_dot_history(theta_dot)
                controller._set_previous_block_theta_snapshot()
    finally:
        if hasattr(controller, "_close_oracles"):
            controller._close_oracles()

    replay_context = loaded.replay_context
    summary = _summary_from_rows(
        rows=trajectory,
        retarget_events=retarget_events,
        schedule=schedule,
        exact_reference_cache=exact_reference_cache,
        reference_energy_total_span=float(reference_energy_total_span),
        controller=controller,
    )
    payload = {
        "pipeline": "hh_periodic_oracle_retarget_adapt_v1",
        "run_tag": str(args.run_tag),
        "artifact_json": str(Path(args.artifact_json).expanduser().resolve()),
        "output_json": str(output_json),
        "loader_mode": str(args.loader_mode),
        "loader_summary": {
            "generator_family": str(args.generator_family),
            "fallback_family": str(args.fallback_family),
            "resolved_family": str(getattr(replay_context, "family_info", {}).get("resolved", "unknown")),
            "handoff_state_kind": str(getattr(replay_context, "handoff_state_kind", "unknown")),
            "family_terms_count": int(getattr(replay_context, "family_terms_count", 0)),
            "adapt_depth": int(getattr(replay_context, "adapt_depth", 0)),
        },
        "retarget_config": {
            **asdict(retarget_cfg),
            "resolved_retarget_checkpoints": [int(x) for x in schedule],
            "schedule_reasons": {str(k): [str(x) for x in v] for k, v in schedule_reasons.items()},
        },
        "checkpoint_adapt_config": _to_jsonable(asdict(adapt_cfg)),
        "fit_config": _to_jsonable(asdict(fit_cfg)),
        "controller_config": _to_jsonable(bundle["cfg"]),
        "drive_config": _to_jsonable(bundle["drive_config"]),
        "oracle_config": _to_jsonable(bundle["oracle_config"]),
        "reference": _to_jsonable(exact_helper.reference_payload()),
        "summary": _to_jsonable(summary),
        "trajectory": _to_jsonable(trajectory),
        "ledger": _to_jsonable(ledger),
        "retarget_events": _to_jsonable(retarget_events),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = build_realtime_parser()
    parser.description = (
        "Periodic oracle-retargeted ADAPT/McLachlan diagnostic against exact HH time states."
    )
    parser.set_defaults(
        run_tag="hh_periodic_oracle_retarget_adapt",
        checkpoint_controller_mode="exact_v1",
        checkpoint_controller_reference_mode="benchmark_exact",
    )
    parser.add_argument("--oracle-retarget-period-checkpoints", type=int, default=8)
    parser.add_argument("--oracle-retarget-checkpoints", default="")
    parser.add_argument(
        "--oracle-retarget-include-initial",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--oracle-retarget-target-mode",
        choices=["exact_time_state"],
        default="exact_time_state",
    )
    add_checkpoint_local_adapt_config_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_periodic_oracle_retarget_from_args(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
