#!/usr/bin/env python3
"""Types and identity helpers for HH adaptive realtime checkpoint control."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


HIGH_MISS_NO_ADMIT_POLICY_DEFAULT = "bounded_stay_advance"
HIGH_MISS_NO_ADMIT_POLICY_CANONICAL = (
    "bounded_stay_advance",
    "repair_stop",
    "repair_retry",
)
HIGH_MISS_NO_ADMIT_POLICY_ALIASES = {
    "legacy_advance_stay": HIGH_MISS_NO_ADMIT_POLICY_DEFAULT,
}
HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_REASON = "bounded_high_miss_no_admit_stay_advance"
HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_WARNING = (
    "HighMiss active and no append candidate was admitted; advanced one checkpoint on "
    "the current scaffold under bounded_stay_advance without relaxing append admission thresholds."
)


def normalize_high_miss_no_admit_policy(raw: str | None) -> str:
    """Normalize high-miss/no-admit policy names to canonical output tokens."""
    text = HIGH_MISS_NO_ADMIT_POLICY_DEFAULT if raw is None else str(raw).strip().lower()
    if text == "":
        text = HIGH_MISS_NO_ADMIT_POLICY_DEFAULT
    normalized = HIGH_MISS_NO_ADMIT_POLICY_ALIASES.get(text, text)
    if normalized not in HIGH_MISS_NO_ADMIT_POLICY_CANONICAL:
        allowed = ", ".join((*HIGH_MISS_NO_ADMIT_POLICY_CANONICAL, *HIGH_MISS_NO_ADMIT_POLICY_ALIASES))
        raise ValueError(
            f"high_miss_no_admit_policy must be one of {allowed}; got {raw!r}."
        )
    return str(normalized)


def high_miss_no_admit_soft_fallback_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize bounded high-miss/no-admit soft-fallback telemetry over rows."""
    raw_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    soft_rows = [row for row in raw_rows if bool(row.get("high_miss_no_admit_soft_fallback", False))]
    stay_count = int(sum(1 for row in raw_rows if str(row.get("action_kind")) == "stay"))
    reason_counts: dict[str, int] = {}
    for row in soft_rows:
        reason = row.get("high_miss_no_admit_soft_fallback_reason", None)
        if reason in {None, ""}:
            reason = HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_REASON
        reason_text = str(reason)
        reason_counts[reason_text] = int(reason_counts.get(reason_text, 0)) + 1
    soft_count = int(len(soft_rows))
    return {
        "high_miss_no_admit_soft_fallback_count": int(soft_count),
        "high_miss_no_admit_soft_fallback_warning_count": int(soft_count),
        "ordinary_stay_count": int(max(0, stay_count - soft_count)),
        "high_miss_no_admit_soft_fallback_reason_counts": dict(reason_counts),
    }


def _compact_reason_counts(values: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value in {None, ""}:
            continue
        key = str(value)
        counts[key] = int(counts.get(key, 0)) + 1
    return dict(sorted(counts.items()))


def _high_miss_no_admit_reason(row: Mapping[str, Any]) -> str:
    diag = row.get("repair_no_admit_diagnostics", None)
    if isinstance(diag, Mapping):
        for key in (
            "strict_no_admit_reason",
            "forecast_veto_reason",
            "no_admit_resolution",
        ):
            value = diag.get(key, None)
            if value not in {None, ""}:
                return str(value)
    for key in (
        "append_no_harm_veto_reason",
        "decision_override_reason",
        "exact_v1_selection_reason",
        "selection_reason",
    ):
        value = row.get(key, None)
        if value not in {None, ""}:
            return str(value)
    return "unknown_no_admit"


def _compact_first_high_miss_no_admit_diagnostic(row: Mapping[str, Any]) -> dict[str, Any]:
    diag = row.get("repair_no_admit_diagnostics", None)
    compact = {
        "checkpoint_index": row.get("checkpoint_index"),
        "time": row.get("time"),
        "physical_time": row.get("physical_time"),
        "action_kind": row.get("action_kind"),
        "proposed_action_kind": row.get("proposed_action_kind"),
        "candidate_label": row.get("candidate_label"),
        "proposed_candidate_label": row.get("proposed_candidate_label"),
        "controller_lane": row.get("controller_lane"),
        "controller_lane_reason": row.get("controller_lane_reason"),
        "rho_miss": row.get("rho_miss"),
        "rho_real": row.get("rho_real"),
        "rho_num": row.get("rho_num"),
        "gain_ratio_selected": row.get("gain_ratio_selected"),
        "exact_v1_selection_reason": row.get("exact_v1_selection_reason"),
        "selection_reason": row.get("selection_reason"),
        "decision_override_reason": row.get("decision_override_reason"),
        "append_no_harm_veto_reason": row.get("append_no_harm_veto_reason"),
        "high_miss_no_admit_reason": _high_miss_no_admit_reason(row),
        "high_miss_no_admit_soft_fallback": bool(row.get("high_miss_no_admit_soft_fallback", False)),
    }
    if isinstance(diag, Mapping):
        compact["repair_no_admit_diagnostics"] = dict(diag)
    return compact


"""
Built Math: high_miss_fraction = #(controller_lane=append)/N_decisions; no_admit is the append-lane subset that does not commit an append.
"""
def high_miss_no_admit_diagnostic_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compact JSON-only telemetry for high-miss append-lane no-admit behavior."""
    raw_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    denominator = max(int(len(raw_rows)), 1)

    def _is_high_miss_row(row: Mapping[str, Any]) -> bool:
        if str(row.get("controller_lane", "")) == "append":
            return True
        if bool(row.get("high_miss_no_admit_soft_fallback", False)):
            return True
        diag = row.get("repair_no_admit_diagnostics", None)
        return bool(isinstance(diag, Mapping) and str(diag.get("controller_lane", "")) == "append")

    high_rows = [row for row in raw_rows if _is_high_miss_row(row)]

    def _is_no_admit(row: Mapping[str, Any]) -> bool:
        if bool(row.get("high_miss_no_admit_soft_fallback", False)):
            return True
        if isinstance(row.get("repair_no_admit_diagnostics", None), Mapping):
            return True
        if str(row.get("controller_lane", "")) != "append":
            return False
        action = str(row.get("action_kind", ""))
        proposed = str(row.get("proposed_action_kind", ""))
        return bool(
            action in {"stay", "repair_miss"}
            or (proposed == "append_candidate" and action != "append_candidate")
        )

    no_admit_rows = [row for row in high_rows if _is_no_admit(row)]
    append_veto_rows = [
        row for row in raw_rows if row.get("append_no_harm_veto_reason") not in {None, ""}
    ]
    no_admit_resolution_values: list[Any] = []
    for row in no_admit_rows:
        diag = row.get("repair_no_admit_diagnostics", None)
        if isinstance(diag, Mapping):
            no_admit_resolution_values.append(diag.get("no_admit_resolution", None))
    return {
        "high_miss_count": int(len(high_rows)),
        "high_miss_fraction": float(len(high_rows) / denominator),
        "high_miss_no_admit_count": int(len(no_admit_rows)),
        "high_miss_no_admit_fraction": float(len(no_admit_rows) / denominator),
        "high_miss_no_admit_reason_counts": _compact_reason_counts(
            [_high_miss_no_admit_reason(row) for row in no_admit_rows]
        ),
        "high_miss_no_admit_resolution_counts": _compact_reason_counts(no_admit_resolution_values),
        "append_no_harm_veto_count": int(len(append_veto_rows)),
        "append_no_harm_veto_reason_counts": _compact_reason_counts(
            [row.get("append_no_harm_veto_reason") for row in append_veto_rows]
        ),
        "first_bad_high_miss_no_admit_checkpoint_diagnostic": (
            None
            if not no_admit_rows
            else _compact_first_high_miss_no_admit_diagnostic(no_admit_rows[0])
        ),
    }


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if np.isfinite(numeric) else None


STABLE_EARLY_STOP_PREFIX = "progress_observables_stable:"


def is_successful_stable_early_stop_reason(reason: Any) -> bool:
    text = "" if reason is None else str(reason).strip()
    if text.lower() in {"", "none", "null"}:
        return False
    return bool(text.startswith(STABLE_EARLY_STOP_PREFIX))


"""
Built Math: full_horizon_gate = reached(t_final) ∧ reached(N_expected) ∨ accepted_stable_early_stop.
"""
def full_horizon_completion_fields(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_t_final: float,
    expected_row_count: int,
    early_stop_reason: Any = None,
    stable_early_stop_accepted: bool = False,
    final_time_tol: float = 1.0e-9,
) -> dict[str, Any]:
    """Return explicit full-horizon completion gate fields for result summaries/rankings."""
    physical_rows = physical_trajectory_rows(rows, fallback_to_raw=False)
    final_row = physical_rows[-1] if physical_rows else {}
    final_time = _finite_float_or_none(final_row.get("time", None))
    expected_final = float(expected_t_final)
    expected_count = int(expected_row_count)
    observed_count = int(len(physical_rows))
    early_reason = None if str(early_stop_reason).strip().lower() in {"", "none", "null"} else early_stop_reason
    reached_final_time = bool(
        final_time is not None and final_time >= expected_final - float(final_time_tol)
    )
    reached_expected_rows = bool(observed_count >= expected_count)
    stable_early_stop = bool(
        bool(stable_early_stop_accepted)
        and
        early_reason not in {None, ""}
        and bool(physical_rows)
        and is_successful_stable_early_stop_reason(early_reason)
    )
    if stable_early_stop:
        gate_reason = f"stable_early_stop:{early_reason}"
        completion_kind = "stable_early_stop"
    elif early_reason not in {None, ""}:
        gate_reason = f"early_stop:{early_reason}"
        completion_kind = "failed"
    elif not physical_rows:
        gate_reason = "no_physical_rows"
        completion_kind = "failed"
    elif not reached_final_time:
        gate_reason = "final_time_short"
        completion_kind = "failed"
    elif not reached_expected_rows:
        gate_reason = "row_count_short"
        completion_kind = "failed"
    else:
        gate_reason = "passed"
        completion_kind = "completed"
    passed = bool(str(gate_reason) == "passed" or stable_early_stop)
    return {
        "full_horizon_gate_passed": bool(passed),
        "full_horizon_completion_kind": str(completion_kind),
        "full_horizon_successful_early_stop": bool(stable_early_stop),
        "full_horizon_gate_reason": str(gate_reason),
        "full_horizon_expected_t_final": float(expected_final),
        "full_horizon_final_time": (None if final_time is None else float(final_time)),
        "full_horizon_expected_row_count": int(expected_count),
        "full_horizon_observed_row_count": int(observed_count),
        "full_horizon_reached_final_time": bool(reached_final_time),
        "full_horizon_reached_expected_rows": bool(reached_expected_rows),
        "full_horizon_early_stop_reason": (None if early_reason in {None, ""} else str(early_reason)),
    }


@dataclass(frozen=True)
class MeasurementTierConfig:
    tier_name: str
    exact_mode_behavior: str
    oracle_shots: int | None = None
    oracle_repeats: int | None = None
    oracle_aggregate: str | None = None


@dataclass(frozen=True)
class RealtimeCheckpointConfig:
    mode: str = "off"
    reference_mode: str = "benchmark_exact"
    oracle_selection_policy: str = "measured_gain_commit_veto"
    forecast_score_gain_weight: float = 1.0
    forecast_score_rho_miss_weight: float = 1.0
    forecast_score_step_residual_weight: float = 0.5
    forecast_score_condition_weight: float = 0.1
    forecast_score_theta_velocity_weight: float = 0.1
    forecast_score_displacement_weight: float = 0.1
    forecast_accept_margin: float = 0.0
    candidate_step_scales: tuple[float, ...] = (1.0,)
    exact_forecast_baseline_step_refine_rounds: int = 0
    exact_forecast_baseline_proposal_mode: str = "norm_locked_blend_v1"
    exact_forecast_baseline_blend_weights: tuple[float, ...] = ()
    exact_forecast_baseline_gain_scales: tuple[float, ...] = ()
    exact_forecast_include_tangent_secant_proposal: bool = False
    exact_forecast_tangent_secant_trust_radius: float = 0.0
    exact_forecast_tangent_secant_signed_energy_lead_limit: float = 0.0
    exact_forecast_tracking_horizon_steps: int = 1
    exact_forecast_tracking_horizon_weights: tuple[float, ...] = ()
    exact_forecast_primary_density_target_mode: str = "auto"
    exact_forecast_tracking_fidelity_defect_weight: float = 1.0
    exact_forecast_tracking_primary_density_error_weight: float | None = None
    exact_forecast_tracking_staggered_error_weight: float = 1.0
    exact_forecast_tracking_doublon_error_weight: float = 1.0
    exact_forecast_tracking_site_occupations_error_weight: float = 1.0
    exact_forecast_tracking_energy_total_error_weight: float = 1.0
    exact_forecast_primary_density_scale_floor: float = 1.0e-6
    exact_forecast_density_slope_scale_floor: float = 1.0e-6
    exact_forecast_doublon_scale_floor: float = 1.0e-6
    exact_forecast_site_occupations_scale_floor: float = 1.0e-6
    exact_forecast_energy_total_scale_floor: float = 1.0e-6
    exact_forecast_density_slope_weight: float = 1.0
    exact_forecast_density_curvature_weight: float = 0.0
    exact_forecast_density_excursion_under_weight: float = 0.0
    exact_forecast_density_excursion_over_weight: float = 0.0
    exact_forecast_density_sign_lag_weight: float = 0.0
    exact_forecast_density_postcross_wrong_sign_weight: float = 0.0
    exact_forecast_drive_harmonic_weight: float = 0.0
    exact_forecast_energy_slope_weight: float = 0.0
    exact_forecast_energy_curvature_weight: float = 0.0
    exact_forecast_energy_excursion_under_weight: float = 0.0
    exact_forecast_energy_excursion_over_weight: float = 0.0
    exact_forecast_energy_excursion_rel_tolerance: float = 0.0
    exact_v1_repeat_reopen_mode: str = "off"
    exact_v1_density_first_target_gain_floor: float = 2.0e-2
    exact_v1_below_floor_probe_target_gain_floor: float = 3.0e-2
    exact_v1_sign_lag_window_activation: float = 0.0
    exact_v1_sign_lag_window_target_gain_floor: float | None = None
    exact_v1_postcross_wrong_sign_activation: float = 0.0
    exact_v1_postcross_wrong_sign_target_gain_floor: float | None = None
    exact_v1_postcross_compare_diag: bool = False
    exact_v1_below_floor_energy_safe_turn_escape: bool = False
    exact_v1_below_floor_energy_safe_d_shape_escape: bool = False
    exact_v1_d_shape_turn_window_abs_activation: float = 0.0
    exact_v1_d_shape_outside_turn_below_floor_probe_stall_threshold: int = 0
    exact_v1_d_shape_pre_turn_shadow_bridge: bool = False
    exact_v1_single_surface_commit_law: bool = False
    exact_forecast_total_occupation_error_increase_tol: float = 0.0
    progress_observable_window: int = 16
    progress_early_stop_min_checkpoint: int = 0
    progress_early_stop_site_error_mean_max: float | None = None
    progress_early_stop_primary_density_error_mean_max: float | None = None
    progress_early_stop_energy_error_mean_max: float | None = None
    progress_early_stop_site_span_max: float | None = None
    progress_early_stop_primary_density_span_max: float | None = None
    progress_early_stop_energy_span_max: float | None = None
    exact_forecast_guardrail_mode: str = "off"
    exact_forecast_fidelity_loss_tol: float = 0.0
    exact_forecast_abs_energy_error_increase_tol: float = 0.0
    miss_threshold: float = 0.05
    high_miss_no_admit_policy: str = HIGH_MISS_NO_ADMIT_POLICY_DEFAULT
    repair_retry_max_attempts: int = 2
    repair_retry_escalation_mode: str = "append_budget_then_stabilize_v1"
    repair_retry_admission_policy: str = "strict"
    repair_retry_rescue_min_gain_ratio: float = 0.0
    repair_retry_rescue_attempt: str = "terminal_attempt_only"
    miss_abs_threshold: float = 0.0
    miss_persistence_window: int = 1
    miss_persistence_count: int = 1
    integrator_policy: str = "auto_euler_rk4"
    integrator_columnarity_threshold: float = 0.80
    integrator_curvature_threshold: float = 0.10
    integrator_euler_fs_error_threshold: float = 1.0e-3
    integrator_condition_max: float = 1.0e12
    integrator_euler_min_time_fraction: float = 0.35
    integrator_euler_observable_window: int = 16
    integrator_euler_site_span_max: float | None = None
    integrator_euler_primary_density_span_max: float | None = None
    integrator_euler_energy_span_max: float | None = None
    gain_ratio_threshold: float = 0.02
    append_margin_abs: float = 1e-6
    append_enabled: bool = True
    append_no_harm_guard_enabled: bool = True
    append_no_harm_condition_ratio_cap: float = 1.0
    append_no_harm_displacement_ratio_cap: float = 1.0
    append_no_harm_condition_abs_floor: float = 1.0
    append_no_harm_kink_min_step_gain_delta: float = 1.0e-3
    append_no_harm_kink_max_condition_ratio: float = 1.0
    append_no_harm_kink_max_displacement_ratio: float = 1.0
    append_no_harm_rho_only_min_step_gain_delta: float = 1.0e-3
    append_no_harm_rho_only_condition_ratio_cap: float = 1.5
    append_no_harm_rho_only_step_residual_ratio_cap: float = 1.5
    append_no_harm_rho_only_displacement_ratio_cap: float = 1.5
    shortlist_size: int = 4
    shortlist_fraction: float = 0.15
    active_window_size: int = 3
    measurement_active_window_size: int = 0
    max_probe_positions: int = 4
    regularization_lambda: float = 1e-8
    candidate_regularization_lambda: float = 1e-8
    pinv_rcond: float = 1e-10
    analytic_noise_std: float = 0.0
    analytic_noise_seed: int | None = None
    analytic_noise_model: str = "iid_gaussian_legacy"
    analytic_noise_nominal_shots: int = 2048
    analytic_noise_nominal_repeats: int = 1
    analytic_noise_shot_scale: float = 1.0
    analytic_noise_two_qubit_depth_scale: float = 0.0
    analytic_noise_groups_new_scale: float = 0.0
    analytic_noise_time_corr: float = 0.0
    analytic_noise_bias_energy: float = 0.0
    analytic_noise_bias_doublon: float = 0.0
    analytic_noise_bias_staggered: float = 0.0
    analytic_noise_metric_scale: float = 1.0
    analytic_noise_force_psd: bool = True
    compile_penalty_weight: float = 0.05
    measurement_penalty_weight: float = 0.02
    directional_penalty_weight: float = 0.01
    confirm_score_mode: str = "compressed_whitened_v1"
    confirm_compress_fraction: float = 0.5
    confirm_compress_min_modes: int = 1
    confirm_compress_max_modes: int = 8
    prune_mode: str = "off"
    prune_miss_threshold: float = 0.02
    prune_protection_steps: int = 2
    prune_stagnation_window: int = 3
    prune_stagnation_alpha: float = 0.5
    prune_stale_score_threshold: float = 0.75
    prune_loss_threshold: float = 0.01
    prune_max_candidates: int = 2
    prune_cooldown_steps: int = 2
    prune_safe_miss_increase_tol: float = 0.01
    prune_no_harm_guard_enabled: bool = True
    prune_no_harm_score_increase_tol: float = 0.0
    prune_no_harm_step_residual_ratio_increase_tol: float = 1.0e-6
    prune_schur_ladder_mode: str = "nested_runtime_v1"
    prune_schur_ladder_local_radius: int = 1
    prune_schur_monotonicity_tol: float = 1.0e-9
    prune_loss_norm_epsilon: float = 1.0e-14
    prune_differential_miss_tol: float = 1.0e-2
    prune_high_miss_differential_enabled: bool = True
    prune_projection_mode: str = "state_tangent_ls_v1"
    prune_projection_rounds: int = 2
    prune_projection_max_active_runtime: int = 64
    prune_projection_trust_radius: float = 5.0e-2
    prune_projection_state_weight: float = 1.0
    prune_projection_observable_weight: float = 0.25
    prune_projection_regularization: float = 1.0e-8
    prune_ray_distance_tol: float = 5.0e-2
    prune_shadow_enabled: bool = True
    prune_shadow_horizon_steps: int = 2
    prune_shadow_score_tol: float = 1.0e-2
    prune_shadow_score_increase_tol: float = 0.0
    prune_shadow_scale_floor: float = 1.0e-6
    prune_persistence_window: int = 1
    prune_persistence_required: int = 1
    prune_state_jump_l2_tol: float = 0.05
    prune_state_jump_l2_hard_cap: float = 1.0e-2
    prune_theta_block_tol: float = 0.05
    prune_appended_origin_bias_enabled: bool = True
    prune_appended_origin_target_policy: str = "append_only"
    prune_appended_origin_grace_steps: int = 1
    prune_initial_scaffold_grace_steps: int = 64
    prune_active_block_theta_dot_rel_tol: float = 0.03
    prune_active_block_theta_dot_abs_tol: float = 1.0e-8
    prune_active_block_theta_dot_abs_hard_tol: float = 5.0e-2
    prune_appended_origin_bias_scale: float = 0.10
    prune_appended_origin_bias_max_factor: float = 0.50
    motion_calm_direction_cosine_threshold: float = 0.98
    motion_calm_rate_change_ratio_threshold: float = 0.15
    motion_direction_reversal_cosine_threshold: float = -0.05
    motion_curvature_flip_cosine_threshold: float = -0.10
    motion_acceleration_l2_threshold: float = 0.05
    motion_kink_rate_change_ratio_threshold: float = 0.50
    motion_calm_shortlist_scale: float = 0.5
    motion_kink_shortlist_bonus: int = 2
    motion_calm_oracle_budget_scale: float = 0.5
    motion_kink_oracle_budget_scale: float = 2.0
    position_jump_tie_margin_abs: float = 1e-6
    reconstruction_tol: float = 1e-10
    grouping_mode: str = "qwc_basis_cover_reuse"
    tiers: tuple[MeasurementTierConfig, ...] = field(
        default_factory=lambda: (
            MeasurementTierConfig(tier_name="scout", exact_mode_behavior="proxy_only"),
            MeasurementTierConfig(tier_name="confirm", exact_mode_behavior="incremental_exact"),
            MeasurementTierConfig(tier_name="commit", exact_mode_behavior="commit_exact"),
        )
    )


@dataclass(frozen=True)
class ScaffoldAcceptanceResult:
    accepted: bool
    reason: str
    structure_locked: bool
    source_kind: str


@dataclass(frozen=True)
class CheckpointContext:
    checkpoint_index: int
    time_start: float
    time_stop: float | None
    checkpoint_id: str
    scaffold_hash: str
    theta_hash: str
    state_hash: str
    resolved_family: str
    grouping_mode: str
    branch_id: int
    structure_locked: bool


@dataclass(frozen=True)
class GeometryValueKey:
    checkpoint_id: str
    observable_family: str
    candidate_label: str | None
    position_id: int | None
    runtime_indices: tuple[int, ...]
    group_key: str | None
    grouping_mode: str


@dataclass(frozen=True)
class OracleValueKey:
    checkpoint_id: str
    tier_name: str
    observable_family: str
    candidate_label: str | None
    position_id: int | None


@dataclass(frozen=True)
class DerivedGeometryKey:
    checkpoint_id: str
    memo_family: str
    candidate_label: str | None
    position_id: int | None


@dataclass(frozen=True)
class RawGroupKey:
    checkpoint_id: str
    observable_family: str
    candidate_label: str | None
    position_id: int | None
    group_key: str
    state_key: str | None = None


@dataclass(frozen=True)
class SharedRawGroupKey:
    checkpoint_id: str
    candidate_label: str | None
    position_id: int | None
    state_key: str
    group_key: str


@dataclass(frozen=True)
class TemporalPriorKey:
    candidate_identity: str
    position_id: int


@dataclass(frozen=True)
class TemporalPriorRecord:
    candidate_identity: str
    position_id: int
    last_checkpoint_index: int
    times_selected: int
    last_groups_new: float
    last_gain_ratio: float
    last_predicted_displacement: float
    last_refresh_pressure: str


@dataclass(frozen=True)
class BaselineGeometrySummary:
    energy: float
    variance: float
    epsilon_proj_sq: float
    epsilon_step_sq: float
    rho_miss: float
    rho_real: float
    rho_num: float
    step_objective_value: float
    step_gain_ratio: float
    theta_dot_l2: float
    matrix_rank: int
    condition_number: float
    regularization_lambda: float
    solve_mode: str
    logical_block_count: int
    runtime_parameter_count: int
    planning_summary: dict[str, Any]
    exact_cache_summary: dict[str, Any]


@dataclass(frozen=True)
class CandidateProbeSummary:
    candidate_label: str
    candidate_pool_index: int
    position_id: int
    runtime_insert_position: int
    runtime_block_indices: list[int]
    residual_overlap_l2: float
    gain_exact: float | None
    gain_ratio: float | None
    compile_proxy_total: float
    groups_new: float
    novelty: float | None
    position_jump_penalty: float
    directional_change_l2: float | None
    tier_reached: str
    admissible: bool
    rejection_reason: str | None
    decision_metric: str = "gain_ratio"
    oracle_estimate_kind: str | None = None
    temporal_prior_bonus: float = 0.0
    predicted_noisy_energy_mean: float | None = None
    predicted_noisy_energy_stderr: float | None = None
    predicted_noisy_improvement_abs: float | None = None
    predicted_noisy_improvement_ratio: float | None = None
    selected_step_scale: float | None = None


@dataclass(frozen=True)
class CheckpointLedgerEntry:
    checkpoint_index: int
    time: float
    action_kind: str
    candidate_label: str | None
    position_id: int | None
    rho_miss: float
    rho_real: float
    rho_num: float
    gain_ratio_selected: float
    shortlist_size: int
    tier_reached: str
    logical_block_count_before: int
    logical_block_count_after: int
    runtime_parameter_count_before: int
    runtime_parameter_count_after: int
    rate_change_l2: float | None
    theta_dot_l2: float | None = None
    theta_update_l2: float | None = None
    observable_family: str | None = None
    primary_density_mode: str | None = None
    drive_enabled: bool = False
    drive_operator_label: str | None = None
    drive_family_key: str | None = None
    drive_coefficient: float | None = None
    drive_coefficient_linf: float | None = None
    prune_cached_loss_selected: float | None = None
    selected_prune_loss: float | None = None
    selected_prune_loss_kind: str | None = None
    selected_prune_loss_delta_g_theorem: float | None = None
    selected_prune_loss_delta_k_damped: float | None = None
    selected_prune_loss_legacy_proxy: float | None = None
    selected_prune_loss_denominator: float | None = None
    selected_prune_loss_denominator_kind: str | None = None
    selected_prune_loss_support_kind: str | None = None
    selected_prune_loss_removed_runtime_indices: list[int] | None = None
    selected_prune_loss_support_runtime_indices: list[int] | None = None
    selected_prune_loss_support_size: int | None = None
    selected_prune_loss_matrix_for_selection: str | None = None
    selected_prune_loss_pinv_policy_id: str | None = None
    selected_prune_loss_pinv_rcond: float | None = None
    selected_prune_loss_regularization_lambda: float | None = None
    selected_prune_loss_regularization_source: str | None = None
    selected_prune_rank_score: float | None = None
    selected_prune_rank_score_kind: str | None = None
    selected_prune_rank_score_terms: dict[str, Any] | None = None
    prune_stagnation_score_selected: float | None = None
    post_prune_state_jump_l2: float | None = None
    prune_schur_raw_loss_selected: float | None = None
    prune_schur_normalized_loss_selected: float | None = None
    prune_schur_selected_rung: int | None = None
    prune_schur_monotonicity_status_selected: str | None = None
    prune_differential_miss_selected: float | None = None
    prune_permit_path_selected: str | None = None
    prune_projection_objective_selected: float | None = None
    prune_projected_state_jump_l2_selected: float | None = None
    prune_ray_distance_selected: float | None = None
    prune_shadow_score_selected: float | None = None
    prune_persistence_count_selected: int | None = None
    prune_persistence_required_selected: int | None = None
    prune_persistence_passed_selected: bool | None = None
    prune_origin_kind_selected: str | None = None
    prune_age_checkpoints_selected: int | None = None
    prune_block_theta_dot_norm_selected: float | None = None
    prune_block_theta_dot_rel_selected: float | None = None
    prune_appended_origin_bias_factor_selected: float | None = None
    prune_appended_origin_bias_applied_selected: bool | None = None
    integrator_policy: str = "euler"
    integrator_used: str = "euler"
    integrator_columnarity: float | None = None
    integrator_curvature: float | None = None
    integrator_euler_fs_error: float | None = None
    integrator_geometry_gate_pass: bool | None = None
    integrator_euler_error_pass: bool | None = None
    integrator_auto_policy_schema: str | None = None
    integrator_auto_admit_euler: bool | None = None
    integrator_euler_blockers: list[str] | None = None
    integrator_condition_number: float | None = None
    integrator_condition_pass: bool | None = None
    integrator_rho_miss_pass: bool | None = None
    integrator_time_fraction: float | None = None
    integrator_euler_min_time_fraction: float | None = None
    integrator_euler_time_gate_pass: bool | None = None
    integrator_euler_observable_gate_pass: bool | None = None
    integrator_euler_site_span: float | None = None
    integrator_euler_primary_density_span: float | None = None
    integrator_euler_energy_span: float | None = None
    integrator_error: str | None = None
    proposed_action_kind: str | None = None
    proposed_candidate_label: str | None = None
    controller_lane: str | None = None
    controller_lane_reason: str | None = None
    physical_time: float | None = None
    motion_regime: str | None = None
    motion_direction_cosine: float | None = None
    motion_rate_change_ratio: float | None = None
    motion_acceleration_l2: float | None = None
    motion_curvature_cosine: float | None = None
    motion_direction_reversal: bool = False
    motion_curvature_sign_flip: bool = False
    motion_kink_score: float | None = None
    exact_cache_hits: int = 0
    exact_cache_misses: int = 0
    geometry_memo_hits: int = 0
    geometry_memo_misses: int = 0
    planning_groups_new_selected: float = 0.0
    energy_total_controller: float = 0.0
    energy_total_exact: float = 0.0
    abs_energy_total_error: float = 0.0
    fidelity_exact: float = 0.0
    requested_mode: str = "exact_v1"
    decision_backend: str = "exact"
    decision_noise_mode: str | None = None
    controller_exact_input_mode: str = "benchmark_exact"
    decision_data_flow: str = "unknown"
    uses_reference_for_decision: bool = False
    uses_future_exact_forecast_for_decision: bool = False
    uses_statevector_as_ideal_observable_estimator: bool = False
    strict_measurement_oracle_certified: bool = False
    oracle_decision_used: bool = False
    oracle_attempted: bool = False
    oracle_estimate_kind: str | None = None
    selection_metric: str | None = None
    decision_override_reason: str | None = None
    selection_reason: str | None = None
    forecast_mode: str | None = None
    forecast_error: str | None = None
    exact_forecast_error: str | None = None
    forecast_stay_score_total: float | None = None
    forecast_selected_score_total: float | None = None
    forecast_score_delta_vs_stay: float | None = None
    forecast_score_interpretation: str = "lower_is_better"
    forecast_selected_lower_than_stay: bool | None = None
    forecast_stay_rho_miss_next: float | None = None
    forecast_selected_rho_miss_next: float | None = None
    forecast_stay_step_gain_ratio_next: float | None = None
    forecast_selected_step_gain_ratio_next: float | None = None
    forecast_stay_condition_number_next: float | None = None
    forecast_selected_condition_number_next: float | None = None
    baseline_step_scale: float | None = None
    baseline_blend_weight: float | None = None
    baseline_gain_scale: float | None = None
    baseline_proposal_kind: str | None = None
    selected_step_scale: float | None = None
    forecast_stay_fidelity_exact_next: float | None = None
    forecast_selected_fidelity_exact_next: float | None = None
    forecast_stay_abs_energy_total_error_next: float | None = None
    forecast_selected_abs_energy_total_error_next: float | None = None
    forecast_stay_abs_primary_density_error_next: float | None = None
    forecast_selected_abs_primary_density_error_next: float | None = None
    forecast_stay_abs_primary_density_slope_error_next: float | None = None
    forecast_selected_abs_primary_density_slope_error_next: float | None = None
    forecast_stay_abs_staggered_error_next: float | None = None
    forecast_selected_abs_staggered_error_next: float | None = None
    forecast_stay_abs_doublon_error_next: float | None = None
    forecast_selected_abs_doublon_error_next: float | None = None
    forecast_stay_site_occupations_abs_error_max_next: float | None = None
    forecast_selected_site_occupations_abs_error_max_next: float | None = None
    forecast_stay_predicted_displacement_next: float | None = None
    forecast_selected_predicted_displacement_next: float | None = None
    forecast_stay_epsilon_step_ratio_next: float | None = None
    forecast_selected_epsilon_step_ratio_next: float | None = None
    append_no_harm_veto_reason: str | None = None
    append_no_harm_condition_ratio: float | None = None
    append_no_harm_rho_miss_delta: float | None = None
    append_no_harm_step_gain_delta: float | None = None
    append_no_harm_step_residual_ratio: float | None = None
    append_no_harm_displacement_ratio: float | None = None
    append_no_harm_diagnostics: dict[str, Any] | None = None
    append_no_harm_exact_logging: dict[str, Any] | None = None
    predicted_displacement: float | None = None
    temporal_refresh_pressure: str | None = None
    selected_noisy_energy_mean: float | None = None
    selected_noisy_energy_stderr: float | None = None
    stay_noisy_energy_mean: float | None = None
    stay_noisy_energy_stderr: float | None = None
    selected_noisy_improvement_abs: float | None = None
    selected_noisy_improvement_ratio: float | None = None
    oracle_confirm_limit: int | None = None
    oracle_budget_scale: float | None = None
    oracle_cache_hits: int = 0
    oracle_cache_misses: int = 0
    raw_group_cache_hits: int = 0
    raw_group_cache_misses: int = 0
    raw_group_cache_extensions: int = 0
    drive_term_count: int = 0
    analytic_noise_std: float = 0.0
    analytic_noise_seed: int | None = None
    degraded_reason: str | None = None
    trajectory_sample_kind: str = "state_sample"
    advances_time: bool = True
    repair_attempt_index: int = 0
    repair_max_attempts: int | None = None
    repair_escalation_kind: str | None = None
    repair_retry_next: bool = False
    repair_terminal: bool = False
    repair_failure_reason: str | None = None
    accepted_after_repair: bool = False
    repair_no_admit_diagnostics: dict[str, Any] | None = None
    repair_rescue_candidate_label: str | None = None
    repair_rescue_reason: str | None = None
    repair_rescue_admitted: bool = False
    high_miss_no_admit_soft_fallback: bool = False
    high_miss_no_admit_soft_fallback_policy: str | None = None
    high_miss_no_admit_soft_fallback_reason: str | None = None
    high_miss_no_admit_soft_fallback_warning: str | None = None


def is_physical_trajectory_row(row: Mapping[str, Any]) -> bool:
    """Return True for rows that represent physical state samples."""
    if str(row.get("trajectory_sample_kind", "state_sample")) == "repair_event":
        return False
    if row.get("advances_time", True) is False:
        return False
    return True


def physical_trajectory_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    fallback_to_raw: bool = True,
) -> list[dict[str, Any]]:
    """Filter controller trajectory rows to state samples, with optional repair-only fallback."""
    raw_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    physical = [dict(row) for row in raw_rows if is_physical_trajectory_row(row)]
    if physical:
        return physical
    return raw_rows if bool(fallback_to_raw) else []


def trajectory_repair_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    raw_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    repair_events = [row for row in raw_rows if str(row.get("trajectory_sample_kind", "state_sample")) == "repair_event"]
    state_samples = [row for row in raw_rows if is_physical_trajectory_row(row)]
    return {
        "raw_trajectory_row_count": int(len(raw_rows)),
        "repair_event_row_count": int(len(repair_events)),
        "trajectory_state_sample_count": int(len(state_samples)),
    }


def _hash_jsonable(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


"""
scaffold_hash = sha1(labels, logical count, runtime count)
"""
def hash_scaffold_labels(labels: Sequence[str], *, logical_count: int, runtime_count: int) -> str:
    return _hash_jsonable(
        {
            "labels": [str(label) for label in labels],
            "logical_count": int(logical_count),
            "runtime_count": int(runtime_count),
        }
    )


"""
theta_hash = sha1(round(theta, 12))
"""
def hash_theta_vector(theta: np.ndarray | Sequence[float]) -> str:
    arr = np.asarray(theta, dtype=float).reshape(-1)
    rounded = np.round(arr, decimals=12)
    return hashlib.sha1(np.ascontiguousarray(rounded).tobytes()).hexdigest()


"""
state_hash = sha1(round(Re psi, 12), round(Im psi, 12))
"""
def hash_statevector(psi: np.ndarray | Sequence[complex]) -> str:
    arr = np.asarray(psi, dtype=complex).reshape(-1)
    rounded = np.round(arr.real, decimals=12) + 1.0j * np.round(arr.imag, decimals=12)
    return hashlib.sha1(np.ascontiguousarray(rounded).tobytes()).hexdigest()


"""
measurement_state_hash = sha1(scaffold_hash, theta_hash)
"""
def hash_measurement_state(
    *,
    scaffold_labels: Sequence[str],
    logical_count: int,
    runtime_count: int,
    theta: np.ndarray | Sequence[float],
) -> str:
    scaffold_hash = hash_scaffold_labels(
        scaffold_labels,
        logical_count=int(logical_count),
        runtime_count=int(runtime_count),
    )
    theta_hash = hash_theta_vector(theta)
    return _hash_jsonable(
        {
            "scaffold_hash": str(scaffold_hash),
            "theta_hash": str(theta_hash),
        }
    )


def make_checkpoint_context(
    *,
    checkpoint_index: int,
    time_start: float,
    time_stop: float | None,
    scaffold_labels: Sequence[str],
    theta: np.ndarray | Sequence[float],
    psi: np.ndarray | Sequence[complex],
    logical_count: int,
    runtime_count: int,
    resolved_family: str,
    grouping_mode: str,
    structure_locked: bool,
) -> CheckpointContext:
    scaffold_hash = hash_scaffold_labels(
        scaffold_labels,
        logical_count=int(logical_count),
        runtime_count=int(runtime_count),
    )
    theta_hash = hash_theta_vector(theta)
    state_hash = hash_statevector(psi)
    checkpoint_id = _hash_jsonable(
        {
            "checkpoint_index": int(checkpoint_index),
            "time_start": float(time_start),
            "time_stop": None if time_stop is None else float(time_stop),
            "scaffold_hash": str(scaffold_hash),
            "theta_hash": str(theta_hash),
            "state_hash": str(state_hash),
        }
    )
    return CheckpointContext(
        checkpoint_index=int(checkpoint_index),
        time_start=float(time_start),
        time_stop=(None if time_stop is None else float(time_stop)),
        checkpoint_id=str(checkpoint_id),
        scaffold_hash=str(scaffold_hash),
        theta_hash=str(theta_hash),
        state_hash=str(state_hash),
        resolved_family=str(resolved_family),
        grouping_mode=str(grouping_mode),
        branch_id=0,
        structure_locked=bool(structure_locked),
    )


def make_measurement_checkpoint_context(
    *,
    checkpoint_index: int,
    time_start: float,
    time_stop: float | None,
    scaffold_labels: Sequence[str],
    theta: np.ndarray | Sequence[float],
    logical_count: int,
    runtime_count: int,
    resolved_family: str,
    grouping_mode: str,
    structure_locked: bool,
) -> CheckpointContext:
    """Build a checkpoint identity without materializing an exact statevector.

    Strict QPU-faithful controller lanes must key measurements by scaffold and
    parameters only; they intentionally do not call statevector preparation just
    to populate identity fields.
    """

    scaffold_hash = hash_scaffold_labels(
        scaffold_labels,
        logical_count=int(logical_count),
        runtime_count=int(runtime_count),
    )
    theta_hash = hash_theta_vector(theta)
    state_hash = hash_measurement_state(
        scaffold_labels=scaffold_labels,
        logical_count=int(logical_count),
        runtime_count=int(runtime_count),
        theta=theta,
    )
    checkpoint_id = _hash_jsonable(
        {
            "checkpoint_index": int(checkpoint_index),
            "time_start": float(time_start),
            "time_stop": None if time_stop is None else float(time_stop),
            "scaffold_hash": str(scaffold_hash),
            "theta_hash": str(theta_hash),
            "measurement_state_hash": str(state_hash),
        }
    )
    return CheckpointContext(
        checkpoint_index=int(checkpoint_index),
        time_start=float(time_start),
        time_stop=(None if time_stop is None else float(time_stop)),
        checkpoint_id=str(checkpoint_id),
        scaffold_hash=str(scaffold_hash),
        theta_hash=str(theta_hash),
        state_hash=str(state_hash),
        resolved_family=str(resolved_family),
        grouping_mode=str(grouping_mode),
        branch_id=0,
        structure_locked=bool(structure_locked),
    )


def normalize_realtime_controller_mode(raw: str | None) -> str:
    text = "off" if raw is None else str(raw).strip().lower()
    aliases = {
        "live_v1": "exact_v1",
        "oracle_live_v1": "oracle_v1",
        "ideal_observable_v1": "observable_v1",
        "statevector_observable_v1": "observable_v1",
        "cheap_statevector_v1": "observable_v1",
        "qpu_faithful_statevector_v1": "observable_v1",
    }
    normalized = aliases.get(text, text)
    if normalized not in {"off", "exact_v1", "oracle_v1", "observable_v1"}:
        raise ValueError(f"Unsupported realtime checkpoint controller mode {raw!r}.")
    return str(normalized)


def normalize_reference_mode(raw: str | None) -> str:
    text = "off" if raw is None else str(raw).strip().lower()
    aliases = {
        "disabled": "off",
        "benchmark": "benchmark_exact",
        "exact": "benchmark_exact",
    }
    normalized = aliases.get(text, text)
    if normalized not in {"off", "benchmark_exact"}:
        raise ValueError(f"Unsupported realtime checkpoint controller reference mode {raw!r}.")
    return str(normalized)


DECISION_DATA_FLOW_CONTROLLER_DISABLED = "controller_disabled"
DECISION_DATA_FLOW_EXACT_ASSISTED = "exact_assisted_controller"
DECISION_DATA_FLOW_IDEAL_OBSERVABLE = "ideal_observable_estimator"
DECISION_DATA_FLOW_LOCAL_PREPARED_STATE = "local_prepared_state_geometry"
DECISION_DATA_FLOW_MEASUREMENT_ORACLE = "measurement_oracle"
DECISION_DATA_FLOW_MIXED = "mixed"
DECISION_DATA_FLOW_UNKNOWN = "unknown"


def decision_data_flow_fields(
    *,
    controller_mode: str | None,
    controller_exact_input_mode: str | None,
    decision_backend: str | None,
    decision_noise_mode: str | None,
    strict_qpu_faithful: bool = False,
    uses_reference_for_decision: bool = False,
    uses_future_exact_forecast_for_decision: bool = False,
) -> dict[str, Any]:
    """Return additive decision-data-flow telemetry for one decision row.

    ``local_prepared_state_geometry`` means current prepared scaffold/candidate
    state geometry, not an ED target state or exact future/reference trajectory.
    """

    mode = "" if controller_mode is None else str(controller_mode).strip().lower()
    exact_input_mode = normalize_reference_mode(controller_exact_input_mode)
    backend = "" if decision_backend is None else str(decision_backend).strip().lower()
    noise = "" if decision_noise_mode is None else str(decision_noise_mode).strip().lower()
    uses_reference = bool(uses_reference_for_decision)
    uses_future = bool(uses_future_exact_forecast_for_decision)

    if uses_reference or uses_future:
        flow = DECISION_DATA_FLOW_EXACT_ASSISTED
        uses_statevector = False
    elif mode == "off" or backend == "off":
        flow = DECISION_DATA_FLOW_CONTROLLER_DISABLED
        uses_statevector = False
    elif backend == "ideal_observable":
        flow = DECISION_DATA_FLOW_IDEAL_OBSERVABLE
        uses_statevector = True
    elif backend == "oracle":
        if noise == "ideal":
            flow = DECISION_DATA_FLOW_IDEAL_OBSERVABLE
            uses_statevector = True
        else:
            flow = DECISION_DATA_FLOW_MEASUREMENT_ORACLE
            uses_statevector = False
    elif backend == "exact":
        flow = DECISION_DATA_FLOW_LOCAL_PREPARED_STATE
        uses_statevector = True
    elif "," in backend or backend == "mixed":
        flow = DECISION_DATA_FLOW_MIXED
        uses_statevector = False
    else:
        flow = DECISION_DATA_FLOW_UNKNOWN
        uses_statevector = False

    strict_certified = bool(
        strict_qpu_faithful
        and exact_input_mode == "off"
        and not uses_reference
        and not uses_future
        and backend in {"oracle", "ideal_observable"}
        and flow
        in {
            DECISION_DATA_FLOW_IDEAL_OBSERVABLE,
            DECISION_DATA_FLOW_MEASUREMENT_ORACLE,
        }
    )
    return {
        "controller_exact_input_mode": str(exact_input_mode),
        "decision_data_flow": str(flow),
        "uses_reference_for_decision": bool(uses_reference),
        "uses_future_exact_forecast_for_decision": bool(uses_future),
        "uses_statevector_as_ideal_observable_estimator": bool(uses_statevector),
        "strict_measurement_oracle_certified": bool(strict_certified),
    }


def _strict_contract_meaningful(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "none", "null", "off", "false"}
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return bool(np.isfinite(float(value)) and abs(float(value)) > 0.0)
        except Exception:
            return True
    if isinstance(value, Mapping):
        return any(_strict_contract_meaningful(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_strict_contract_meaningful(item) for item in value)
    return bool(value)


def _strict_contract_string(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _strict_contract_histogram(values: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value in {None, ""}:
            continue
        key = str(value)
        counts[key] = int(counts.get(key, 0)) + 1
    return dict(sorted(counts.items()))


def strict_qpu_faithful_decision_contract(
    *,
    summary: Mapping[str, Any] | None = None,
    reference: Mapping[str, Any] | None = None,
    decision_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Audit the decision-path purity contract for strict QPU-faithful routes.

    This helper intentionally audits only controller decision telemetry. Exact/ED
    observables and overlays may be attached later for diagnostics/reporting, but
    exact backends, benchmark references, exact audit helpers, or exact forecast
    data must never appear in the strict controller decision path.
    """

    summary_map = summary if isinstance(summary, Mapping) else {}
    reference_map = reference if isinstance(reference, Mapping) else {}
    rows = [dict(row) for row in decision_rows if isinstance(row, Mapping)]
    violations: list[str] = []

    def _violate(reason: str) -> None:
        if reason not in violations:
            violations.append(str(reason))

    inert_summary_exact_forecast_config_keys = {
        "exact_forecast_baseline_blend_weights",
        "exact_forecast_baseline_gain_scales",
        "exact_forecast_baseline_proposal_mode",
        "exact_forecast_baseline_step_refine_rounds",
        "exact_forecast_density_curvature_weight",
        "exact_forecast_density_excursion_over_weight",
        "exact_forecast_density_excursion_under_weight",
        "exact_forecast_density_sign_lag_weight",
        "exact_forecast_density_slope_weight",
        "exact_forecast_doublon_scale_floor",
        "exact_forecast_drive_harmonic_weight",
        "exact_forecast_energy_curvature_weight",
        "exact_forecast_energy_excursion_over_weight",
        "exact_forecast_energy_excursion_rel_tolerance",
        "exact_forecast_energy_excursion_under_weight",
        "exact_forecast_energy_slope_weight",
        "exact_forecast_energy_total_scale_floor",
        "exact_forecast_fidelity_loss_tol",
        "exact_forecast_guardrail_mode",
        "exact_forecast_include_tangent_secant_proposal",
        "exact_forecast_primary_density_scale_floor",
        "exact_forecast_primary_density_target_mode",
        "exact_forecast_site_occupations_scale_floor",
        "exact_forecast_staggered_scale_floor",
        "exact_forecast_tangent_secant_signed_energy_lead_limit",
        "exact_forecast_tangent_secant_trust_radius",
        "exact_forecast_total_occupation_error_increase_tol",
        "exact_forecast_tracking_doublon_error_weight",
        "exact_forecast_tracking_energy_total_error_weight",
        "exact_forecast_tracking_fidelity_defect_weight",
        "exact_forecast_tracking_horizon_steps",
        "exact_forecast_tracking_horizon_weights",
        "exact_forecast_tracking_primary_density_error_weight",
        "exact_forecast_tracking_site_occupations_error_weight",
        "exact_forecast_tracking_staggered_error_weight",
        "exact_forecast_veto_count",
    }

    summary_reference_mode = _strict_contract_string(summary_map.get("reference_mode", "off")).lower()
    reference_reference_mode = _strict_contract_string(reference_map.get("reference_mode", "off")).lower()
    summary_controller_exact_input_mode = _strict_contract_string(
        summary_map.get(
            "controller_exact_input_mode",
            summary_map.get("controller_reference_mode", summary_reference_mode or "off"),
        )
    ).lower()
    reference_controller_exact_input_mode = _strict_contract_string(
        reference_map.get(
            "controller_exact_input_mode",
            reference_map.get("controller_reference_mode", reference_reference_mode or "off"),
        )
    ).lower()
    reference_mode = (
        "benchmark_exact"
        if "benchmark_exact" in {summary_reference_mode, reference_reference_mode}
        else (summary_reference_mode or reference_reference_mode or "off")
    )
    summary_reference_enabled = bool(summary_map.get("reference_enabled", False))
    reference_reference_enabled = bool(reference_map.get("reference_enabled", False))
    reference_enabled = bool(summary_reference_enabled or reference_reference_enabled)
    if summary_reference_enabled:
        _violate("summary.reference_enabled=true")
    if reference_reference_enabled:
        _violate("reference.reference_enabled=true")
    if summary_reference_mode == "benchmark_exact":
        _violate("summary.reference_mode=benchmark_exact")
    if reference_reference_mode == "benchmark_exact":
        _violate("reference.reference_mode=benchmark_exact")
    if summary_controller_exact_input_mode == "benchmark_exact":
        _violate("summary.controller_exact_input_mode=benchmark_exact")
    if reference_controller_exact_input_mode == "benchmark_exact":
        _violate("reference.controller_exact_input_mode=benchmark_exact")
    summary_uses_reference = bool(summary_map.get("uses_reference_for_decision", False))
    reference_uses_reference = bool(reference_map.get("uses_reference_for_decision", False))
    summary_uses_future_forecast = bool(
        summary_map.get("uses_future_exact_forecast_for_decision", False)
    )
    reference_uses_future_forecast = bool(
        reference_map.get("uses_future_exact_forecast_for_decision", False)
    )
    if summary_uses_reference:
        _violate("summary.uses_reference_for_decision=true")
    if reference_uses_reference:
        _violate("reference.uses_reference_for_decision=true")
    if summary_uses_future_forecast:
        _violate("summary.uses_future_exact_forecast_for_decision=true")
    if reference_uses_future_forecast:
        _violate("reference.uses_future_exact_forecast_for_decision=true")
    summary_forecast_guardrail_mode = _strict_contract_string(
        summary_map.get("exact_forecast_guardrail_mode", "off")
    ).lower()
    try:
        summary_forecast_veto_count = int(summary_map.get("exact_forecast_veto_count", 0) or 0)
    except Exception:
        summary_forecast_veto_count = 0
        _violate("summary.exact_forecast_veto_count=non_integer")
    summary_exact_forecast_decision_active = bool(
        summary_uses_future_forecast
        or summary_forecast_guardrail_mode not in {"", "off", "none", "null", "false"}
    )

    try:
        summary_exact_decisions = int(summary_map.get("exact_decision_checkpoints", 0) or 0)
    except Exception:
        summary_exact_decisions = 0
        _violate("summary.exact_decision_checkpoints=non_integer")
    if summary_exact_decisions > 0:
        _violate(f"summary.exact_decision_checkpoints={summary_exact_decisions}")

    for key, value in summary_map.items():
        key_text = str(key)
        if key_text in {
            "exact_audit_helper_active",
            "exact_audit_active",
            "exact_audit_enabled",
            "exact_step_forecast_active",
            "state_at_active",
        } and _strict_contract_meaningful(value):
            _violate(f"summary.{key_text}=active")
        if key_text.startswith("exact_forecast_") and _strict_contract_meaningful(value):
            if (
                key_text in inert_summary_exact_forecast_config_keys
                and not summary_exact_forecast_decision_active
            ):
                continue
            _violate(f"summary.{key_text}=present")

    row_backends: list[str] = []
    row_noise_modes: list[str] = []
    row_data_flows: list[str] = []
    row_exact_decisions = 0
    row_oracle_decisions = 0
    row_ideal_observable_decisions = 0
    row_uses_reference = False
    row_uses_future_forecast = False
    row_uses_statevector_estimator = False
    row_strict_certified_values: list[bool] = []
    forbidden_key_tokens = (
        "exact_audit",
        "exact_step_forecast",
        "state_at",
    )
    forbidden_exact_prefixes = (
        "exact_forecast_",
        "exact_v1_selection",
    )
    forbidden_reason_keys = {
        "decision_override_reason",
        "repair_rescue_reason",
        "forecast_veto_reason",
        "strict_no_admit_reason",
    }

    def _audit_value(*, row_index: int, path: str, key: str, value: Any) -> None:
        key_text = str(key)
        key_lower = key_text.lower()
        if key_lower == "exact_reference_used_for_veto" and bool(value):
            _violate(f"row[{row_index}].{path}=true")
        if key_lower == "exact_reference_logging" and _strict_contract_meaningful(value):
            _violate(f"row[{row_index}].{path}=present")
        if any(token in key_lower for token in forbidden_key_tokens) and _strict_contract_meaningful(value):
            _violate(f"row[{row_index}].{path}=present")
        if key_lower.startswith(forbidden_exact_prefixes) and _strict_contract_meaningful(value):
            _violate(f"row[{row_index}].{path}=present")
        if key_text in forbidden_reason_keys:
            text = _strict_contract_string(value).lower()
            if text.startswith("exact_forecast_") or text.startswith("exact_v1_"):
                _violate(f"row[{row_index}].{path}={value}")
        if key_lower == "forecast_mode" and "exact" in _strict_contract_string(value).lower():
            _violate(f"row[{row_index}].{path}={value}")
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                _audit_value(
                    row_index=row_index,
                    path=f"{path}.{child_key}",
                    key=str(child_key),
                    value=child_value,
                )

    for idx, row in enumerate(rows):
        backend_raw = row.get("decision_backend", None)
        backend = _strict_contract_string(backend_raw).lower()
        if backend == "exact":
            row_exact_decisions += 1
        if backend == "oracle":
            row_oracle_decisions += 1
        if backend == "ideal_observable":
            row_ideal_observable_decisions += 1
        if backend:
            row_backends.append(backend)
            if backend not in {"oracle", "ideal_observable"}:
                _violate(f"row[{idx}].decision_backend={backend}")
        else:
            _violate(f"row[{idx}].decision_backend=missing")
        noise_mode = _strict_contract_string(row.get("decision_noise_mode", None)).lower()
        if noise_mode:
            row_noise_modes.append(noise_mode)
        if bool(row.get("reference_enabled", False)):
            _violate(f"row[{idx}].reference_enabled=true")
        if _strict_contract_string(row.get("reference_mode", "off")).lower() == "benchmark_exact":
            _violate(f"row[{idx}].reference_mode=benchmark_exact")
        controller_exact_input_mode = _strict_contract_string(
            row.get(
                "controller_exact_input_mode",
                row.get("controller_reference_mode", row.get("reference_mode", "off")),
            )
        ).lower()
        if controller_exact_input_mode == "benchmark_exact":
            _violate(f"row[{idx}].controller_exact_input_mode=benchmark_exact")
        data_flow = _strict_contract_string(row.get("decision_data_flow", "")).lower()
        if data_flow:
            row_data_flows.append(data_flow)
            if data_flow == DECISION_DATA_FLOW_EXACT_ASSISTED:
                _violate(f"row[{idx}].decision_data_flow={data_flow}")
        if bool(row.get("uses_reference_for_decision", False)):
            row_uses_reference = True
            _violate(f"row[{idx}].uses_reference_for_decision=true")
        if bool(row.get("uses_future_exact_forecast_for_decision", False)):
            row_uses_future_forecast = True
            _violate(f"row[{idx}].uses_future_exact_forecast_for_decision=true")
        if bool(row.get("uses_statevector_as_ideal_observable_estimator", False)):
            row_uses_statevector_estimator = True
        if "strict_measurement_oracle_certified" in row:
            row_strict_certified_values.append(
                bool(row.get("strict_measurement_oracle_certified", False))
            )
        for key, value in row.items():
            _audit_value(row_index=idx, path=str(key), key=str(key), value=value)

    exact_decisions = max(int(summary_exact_decisions), int(row_exact_decisions))
    if row_exact_decisions > 0:
        _violate(f"rows.exact_decision_checkpoints={row_exact_decisions}")

    return {
        "passed": not violations,
        "violations": list(violations),
        "violation_count": int(len(violations)),
        "exact_decision_checkpoints": int(exact_decisions),
        "oracle_decision_checkpoints": int(
            max(int(summary_map.get("oracle_decision_checkpoints", 0) or 0), int(row_oracle_decisions))
        ),
        "ideal_observable_decision_checkpoints": int(
            max(
                int(summary_map.get("ideal_observable_decision_checkpoints", 0) or 0),
                int(row_ideal_observable_decisions),
            )
        ),
        "decision_backend_counts": _strict_contract_histogram(row_backends),
        "decision_noise_mode_counts": _strict_contract_histogram(row_noise_modes),
        "decision_data_flow_counts": _strict_contract_histogram(row_data_flows),
        "uses_reference_for_decision": bool(
            summary_uses_reference or reference_uses_reference or row_uses_reference
        ),
        "uses_future_exact_forecast_for_decision": bool(
            summary_uses_future_forecast
            or reference_uses_future_forecast
            or row_uses_future_forecast
        ),
        "uses_statevector_as_ideal_observable_estimator": bool(
            summary_map.get("uses_statevector_as_ideal_observable_estimator", False)
            or row_uses_statevector_estimator
        ),
        "strict_measurement_oracle_certified": bool(
            not violations
            and (
                not row_strict_certified_values
                or all(bool(value) for value in row_strict_certified_values)
            )
        ),
        "reference_mode": reference_mode or "off",
        "reference_enabled": bool(reference_enabled),
    }


def validate_scaffold_acceptance(payload: Mapping[str, Any] | None) -> ScaffoldAcceptanceResult:
    adapt = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(adapt, Mapping):
        adapt = {}
    pool_type = str(adapt.get("pool_type", "")).strip().lower()
    fixed_scaffold_kind = adapt.get("fixed_scaffold_kind", None)
    structure_locked = bool(adapt.get("structure_locked", False))
    if structure_locked:
        return ScaffoldAcceptanceResult(
            accepted=False,
            reason="structure_locked",
            structure_locked=True,
            source_kind=str(fixed_scaffold_kind or pool_type or "unknown"),
        )
    if pool_type == "fixed_scaffold_locked":
        return ScaffoldAcceptanceResult(
            accepted=False,
            reason="fixed_scaffold_locked",
            structure_locked=True,
            source_kind=str(fixed_scaffold_kind or pool_type or "unknown"),
        )
    if fixed_scaffold_kind not in {None, "", "none"}:
        return ScaffoldAcceptanceResult(
            accepted=False,
            reason="fixed_scaffold_kind_present",
            structure_locked=True,
            source_kind=str(fixed_scaffold_kind),
        )
    return ScaffoldAcceptanceResult(
        accepted=True,
        reason="accepted",
        structure_locked=False,
        source_kind=str(pool_type or "adaptive_unlocked"),
    )


def dataclass_to_payload(value: Any) -> dict[str, Any]:
    return asdict(value)


__all__ = [
    "BaselineGeometrySummary",
    "CandidateProbeSummary",
    "CheckpointContext",
    "CheckpointLedgerEntry",
    "DECISION_DATA_FLOW_CONTROLLER_DISABLED",
    "DECISION_DATA_FLOW_EXACT_ASSISTED",
    "DECISION_DATA_FLOW_IDEAL_OBSERVABLE",
    "DECISION_DATA_FLOW_LOCAL_PREPARED_STATE",
    "DECISION_DATA_FLOW_MEASUREMENT_ORACLE",
    "DECISION_DATA_FLOW_MIXED",
    "DECISION_DATA_FLOW_UNKNOWN",
    "DerivedGeometryKey",
    "GeometryValueKey",
    "OracleValueKey",
    "RawGroupKey",
    "SharedRawGroupKey",
    "HIGH_MISS_NO_ADMIT_POLICY_DEFAULT",
    "HIGH_MISS_NO_ADMIT_POLICY_CANONICAL",
    "HIGH_MISS_NO_ADMIT_POLICY_ALIASES",
    "HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_REASON",
    "HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_WARNING",
    "MeasurementTierConfig",
    "RealtimeCheckpointConfig",
    "decision_data_flow_fields",
    "full_horizon_completion_fields",
    "high_miss_no_admit_diagnostic_counts",
    "high_miss_no_admit_soft_fallback_counts",
    "normalize_high_miss_no_admit_policy",
    "normalize_reference_mode",
    "normalize_realtime_controller_mode",
    "strict_qpu_faithful_decision_contract",
    "ScaffoldAcceptanceResult",
    "TemporalPriorKey",
    "TemporalPriorRecord",
    "dataclass_to_payload",
    "hash_measurement_state",
    "physical_trajectory_rows",
    "is_physical_trajectory_row",
    "trajectory_repair_counts",
    "hash_scaffold_labels",
    "hash_statevector",
    "hash_theta_vector",
    "make_checkpoint_context",
    "make_measurement_checkpoint_context",
    "validate_scaffold_acceptance",
]
