#!/usr/bin/env python3
"""Optuna harness for adaptive realtime checkpoint-controller tuning.

Exact-v1 tuning surfaces remain diagnostic/exact-assisted.  Strict QPU-faithful
surfaces are family-generic and launch through the canonical generic realtime
artifact runner so online Optuna feedback stays measurement-compatible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input  # noqa: E402
from pipelines.time_dynamics.legacy.checkpoint_types import (  # noqa: E402
    full_horizon_completion_fields,
    high_miss_no_admit_diagnostic_counts,
    high_miss_no_admit_soft_fallback_counts,
    normalize_high_miss_no_admit_policy,
    physical_trajectory_rows,
    strict_qpu_faithful_decision_contract,
    trajectory_repair_counts,
)
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (  # noqa: E402
    DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE,
    DYNAMICS_COARSE_TUNING_CLASSES,
    DynamicsBenchmarkCase,
    build_dynamics_tuning_provenance,
)
from pipelines.time_dynamics.runners.hh_from_adapt_artifact import (  # noqa: E402
    build_controller_config,
    build_output_payload,
)
from pipelines.time_dynamics.legacy.checkpoint_route_defaults import (  # noqa: E402
    DRIVE_A_DEFAULT,
    DRIVE_OMEGA_DEFAULT,
    DRIVE_PATTERN_DEFAULT,
    DRIVE_PHI_DEFAULT,
    DRIVE_T0_DEFAULT,
    DRIVE_TBAR_DEFAULT,
    DRIVE_TIME_SAMPLING_DEFAULT,
    ENABLE_DRIVE_DEFAULT,
    EXACT_STEPS_MULTIPLIER_DEFAULT,
    NUM_TIMES_DEFAULT,
    T_FINAL_DEFAULT,
)
from pipelines.time_dynamics.legacy.analysis.hh_time_dynamics_spectra import (  # noqa: E402
    analyze_payload,
    compute_one_sided_amplitude_spectrum,
    load_trajectory_payload,
    render_spectrum_pdf,
)
from pipelines.time_dynamics.runners.generic_from_adapt_artifact import (  # noqa: E402
    build_controller_bundle_from_args as build_generic_controller_bundle_from_args,
    build_controller_seed_from_args,
    build_parser as build_realtime_parser,
    finalize_controller_bundle_from_seed,
    run_from_args as run_generic_realtime_from_args,
)
from pipelines.time_dynamics.legacy.checkpoint_route_policy import (  # noqa: E402
    policy_for_family,
    strict_qpu_faithful_requested,
)

_PIPELINE_NAME = "hh_realtime_optuna_v1"
_LARGE_OBJECTIVE = 1.0e9
PAPER_II_CLASS_SETTINGS_LOCK_JSON = REPO_ROOT / "chtc/generic_time_dynamics_table/input/class_settings/paper_ii_class_settings_lock_v1.json"

_HORIZON_PRESETS: dict[str, tuple[int, tuple[float, ...]]] = {
    "lead2": (2, (2.0, 1.0)),
    "flat2": (2, (1.0, 1.0)),
    "lead3": (3, (3.0, 2.0, 1.0)),
    "flat3": (3, (1.0, 1.0, 1.0)),
    "lead4": (4, (4.0, 2.0, 1.0, 1.0)),
    "high_amp_lead4": (4, (8.0, 4.0, 2.0, 1.0)),
}

_STEP_SCALE_PRESETS: dict[str, str] = {
    "narrow": "0.2,0.3,0.4,0.5,0.6",
    "default": "0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.8,1.0",
    "wide": "0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.8,1.0,1.25",
    "high_amp_dense": "0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.8,1.0,1.2",
}

_BLEND_WEIGHT_PRESETS: dict[str, str] = {
    "off": "",
    "default": "-0.25,-0.125,0.0,0.125,0.25,0.375,0.5,0.75,1.0",
    "positive_only": "0.0,0.125,0.25,0.375,0.5,0.75,1.0",
    "tight": "-0.125,0.0,0.125,0.25,0.375,0.5",
    "high_amp": "-0.5,-0.375,-0.25,-0.125,0.0,0.125,0.25,0.375,0.5,0.75,1.0",
}

_GAIN_SCALE_PRESETS: dict[str, str] = {
    "off": "",
    "mild": "0.75,1.0,1.25",
    "wide": "0.5,0.75,1.0,1.25,1.5",
    "high_amp": "1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4",
}


def _encode_horizon_spec(steps: int, weights: Sequence[float]) -> str:
    return f"{int(steps)}|{','.join(str(float(weight)) for weight in weights)}"


_HORIZON_SPEC_PRESETS: dict[str, str] = {
    name: _encode_horizon_spec(steps, weights)
    for name, (steps, weights) in _HORIZON_PRESETS.items()
}


@dataclass(frozen=True)
class BaseRunConfig:
    artifact_json: Path
    study_profile: str = "generic_l2_exact_v1"
    tuning_class: str | None = None
    class_settings_source: str = DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE
    class_settings_lock_json: str = str(PAPER_II_CLASS_SETTINGS_LOCK_JSON)
    loader_mode: str = "replay_family"
    generator_family: str = "match_adapt"
    fallback_family: str = "full_meta"
    append_pool_family: str = "match_replay"
    lock_fixed_manifold: bool = False
    allow_repeats: bool = False
    t_final: float = T_FINAL_DEFAULT
    num_times: int = NUM_TIMES_DEFAULT
    enable_drive: bool = ENABLE_DRIVE_DEFAULT
    disable_drive: bool = False
    drive_A: float = DRIVE_A_DEFAULT
    drive_omega: float = DRIVE_OMEGA_DEFAULT
    drive_tbar: float = DRIVE_TBAR_DEFAULT
    drive_phi: float = DRIVE_PHI_DEFAULT
    drive_pattern: str = DRIVE_PATTERN_DEFAULT
    drive_custom_weights: str = ""
    drive_include_identity: bool = False
    drive_time_sampling: str = DRIVE_TIME_SAMPLING_DEFAULT
    drive_t0: float = DRIVE_T0_DEFAULT
    exact_steps_multiplier: int = EXACT_STEPS_MULTIPLIER_DEFAULT
    pair: str = "auto"
    objective_window_start: float | None = None
    objective_window_end: float | None = None
    spectra_detrend: str = "linear"
    spectra_window: str = "hann"
    max_peaks: int = 5
    max_harmonic: int = 3
    skip_spectra_pdf: bool = True
    integrator_policy_override: str | None = None


@dataclass(frozen=True)
class OptunaRouteContext:
    problem_family: str
    locked_primary_density_target_mode: str | None = None


@dataclass(frozen=True)
class ObjectiveWeights:
    pair_mae_over_exact_span: float = 1.0
    epsilon_osc_pair: float = 0.5
    dominant_peak_abs_omega_error: float = 0.25
    pair_corr_defect: float = 0.25
    mean_abs_site_occupations_error: float = 0.25
    mean_abs_energy_total_error: float = 0.1
    mean_signed_energy_total_error_abs: float = 0.0
    max_abs_energy_total_error: float = 0.0
    energy_total_excursion_under_response_mean: float = 0.0
    energy_total_excursion_under_response_max: float = 0.0
    fidelity_defect: float = 0.1
    final_runtime_parameter_count: float = 0.01
    append_count: float = 0.05
    prune_count: float = 0.05
    rk4_count: float = 0.0


@dataclass(frozen=True)
class ValidityGates:
    max_mean_abs_energy_total_error: float | None = None
    max_final_abs_energy_total_error: float | None = None
    min_fidelity_exact: float | None = None
    max_mean_total_occupation_abs_error: float | None = None
    max_primary_observable_mae_over_exact_span: float | None = None


@dataclass(frozen=True)
class TrialParams:
    horizon_mode: str = "lead2"
    step_scale_mode: str = "default"
    blend_weight_mode: str = "default"
    gain_scale_mode: str = "off"
    oracle_selection_policy: str | None = None
    candidate_step_scales: str | None = None
    baseline_blend_weights: str | None = None
    baseline_gain_scales: str | None = None
    horizon_spec: str | None = None
    baseline_step_refine_rounds: int = 0
    forecast_score_gain_weight: float | None = None
    forecast_score_rho_miss_weight: float | None = None
    forecast_score_step_residual_weight: float | None = None
    forecast_score_condition_weight: float | None = None
    forecast_score_theta_velocity_weight: float | None = None
    forecast_score_displacement_weight: float | None = None
    forecast_accept_margin: float | None = None
    include_tangent_secant_proposal: bool = False
    baseline_proposal_mode: str | None = None
    primary_density_target_mode: str | None = None
    high_miss_no_admit_policy: str | None = None
    repair_retry_max_attempts: int | None = None
    repair_retry_escalation_mode: str | None = None
    repair_retry_admission_policy: str | None = None
    repair_retry_rescue_min_gain_ratio: float | None = None
    repair_retry_rescue_attempt: str | None = None
    miss_threshold: float = 0.05
    miss_abs_threshold: float | None = None
    miss_persistence_spec: str | None = None
    gain_ratio_threshold: float = 0.02
    append_margin_abs: float | None = None
    append_no_harm_guard_enabled: bool | None = None
    append_no_harm_condition_ratio_cap: float | None = None
    append_no_harm_condition_abs_floor: float | None = None
    append_no_harm_displacement_ratio_cap: float | None = None
    append_no_harm_kink_min_step_gain_delta: float | None = None
    append_no_harm_kink_max_condition_ratio: float | None = None
    append_no_harm_kink_max_displacement_ratio: float | None = None
    append_no_harm_rho_only_min_step_gain_delta: float | None = None
    append_no_harm_rho_only_condition_ratio_cap: float | None = None
    append_no_harm_rho_only_step_residual_ratio_cap: float | None = None
    append_no_harm_rho_only_displacement_ratio_cap: float | None = None
    trust_radius: float = 0.0
    signed_energy_lead_limit: float = 0.0
    shortlist_size: int | None = None
    shortlist_fraction: float | None = None
    active_window_size: int | None = None
    max_probe_positions: int | None = None
    regularization_lambda: float | None = None
    candidate_regularization_lambda: float | None = None
    pinv_rcond: float | None = None
    compile_penalty_weight: float | None = None
    measurement_penalty_weight: float | None = None
    directional_penalty_weight: float | None = None
    confirm_score_mode: str | None = None
    confirm_compress_fraction: float | None = None
    confirm_compress_min_modes: int | None = None
    confirm_compress_max_modes: int | None = None
    primary_density_weight: float = 1.0
    tracking_staggered_weight: float | None = None
    site_weight: float = 1.0
    energy_weight: float = 1.0
    density_slope_weight: float = 1.0
    fidelity_weight: float | None = None
    doublon_weight: float | None = None
    density_curvature_weight: float | None = None
    density_excursion_under_weight: float | None = None
    density_excursion_over_weight: float | None = None
    density_sign_lag_weight: float | None = None
    density_postcross_wrong_sign_weight: float | None = None
    drive_harmonic_weight: float | None = None
    energy_slope_weight: float | None = None
    energy_curvature_weight: float | None = None
    energy_excursion_under_weight: float | None = None
    energy_excursion_over_weight: float | None = None
    energy_excursion_rel_tolerance: float | None = None
    repeat_reopen_mode: str | None = None
    density_first_target_gain_floor: float | None = None
    below_floor_probe_target_gain_floor: float | None = None
    sign_lag_window_activation: float | None = None
    sign_lag_window_target_gain_floor: float | None = None
    postcross_wrong_sign_activation: float | None = None
    postcross_wrong_sign_target_gain_floor: float | None = None
    postcross_compare_diag: bool = False
    below_floor_energy_safe_d_shape_escape: bool = False
    below_floor_energy_safe_turn_escape: bool = False
    d_shape_turn_window_abs_activation: float | None = None
    d_shape_outside_turn_below_floor_probe_stall_threshold: int | None = None
    d_shape_pre_turn_shadow_bridge: bool | None = None
    single_surface_commit_law: bool | None = None
    guardrail_mode: str | None = None
    fidelity_loss_tol: float | None = None
    abs_energy_error_increase_tol: float | None = None
    total_occupation_error_increase_tol: float | None = None
    primary_density_scale_floor: float | None = None
    density_slope_scale_floor: float | None = None
    doublon_scale_floor: float | None = None
    site_occupations_scale_floor: float | None = None
    energy_total_scale_floor: float | None = None
    progress_observable_window: int | None = None
    progress_early_stop_min_checkpoint: int | None = None
    progress_early_stop_site_error_mean_max: float | None = None
    progress_early_stop_primary_density_error_mean_max: float | None = None
    progress_early_stop_energy_error_mean_max: float | None = None
    progress_early_stop_site_span_max: float | None = None
    progress_early_stop_primary_density_span_max: float | None = None
    progress_early_stop_energy_span_max: float | None = None
    prune_mode: str | None = None
    prune_miss_threshold: float | None = None
    prune_loss_threshold: float | None = None
    prune_theta_block_tol: float | None = None
    prune_state_jump_l2_tol: float | None = None
    prune_state_jump_l2_hard_cap: float | None = None
    prune_safe_miss_increase_tol: float | None = None
    prune_schur_ladder_mode: str | None = None
    prune_schur_ladder_local_radius: int | None = None
    prune_schur_monotonicity_tol: float | None = None
    prune_loss_norm_epsilon: float | None = None
    prune_differential_miss_tol: float | None = None
    prune_high_miss_differential_enabled: bool | None = None
    prune_projection_mode: str | None = None
    prune_projection_rounds: int | None = None
    prune_projection_max_active_runtime: int | None = None
    prune_projection_trust_radius: float | None = None
    prune_projection_state_weight: float | None = None
    prune_projection_observable_weight: float | None = None
    prune_projection_regularization: float | None = None
    prune_ray_distance_tol: float | None = None
    prune_shadow_enabled: bool | None = None
    prune_shadow_horizon_steps: int | None = None
    prune_shadow_score_tol: float | None = None
    prune_shadow_score_increase_tol: float | None = None
    prune_shadow_scale_floor: float | None = None
    prune_persistence_window: int | None = None
    prune_persistence_required: int | None = None
    prune_appended_origin_bias_enabled: bool | None = None
    prune_appended_origin_target_policy: str | None = None
    prune_appended_origin_grace_steps: int | None = None
    prune_initial_scaffold_grace_steps: int | None = None
    prune_active_block_theta_dot_rel_tol: float | None = None
    prune_active_block_theta_dot_abs_tol: float | None = None
    prune_active_block_theta_dot_abs_hard_tol: float | None = None
    prune_appended_origin_bias_scale: float | None = None
    prune_appended_origin_bias_max_factor: float | None = None
    prune_protection_steps: int | None = None
    prune_stagnation_window: int | None = None
    prune_stagnation_alpha: float | None = None
    prune_stale_score_threshold: float | None = None
    prune_max_candidates: int | None = None
    prune_cooldown_steps: int | None = None
    integrator_policy: str | None = "auto_euler_rk4"
    integrator_columnarity_threshold: float | None = 0.80
    integrator_curvature_threshold: float | None = 0.10
    integrator_euler_fs_error_threshold: float | None = 1.0e-3
    integrator_condition_max: float | None = 1.0e12
    integrator_euler_min_time_fraction: float | None = 0.35
    integrator_euler_observable_window: int | None = 16
    integrator_euler_site_span_max: float | None = None
    integrator_euler_primary_density_span_max: float | None = None
    integrator_euler_energy_span_max: float | None = None
    motion_calm_direction_cosine_threshold: float | None = None
    motion_calm_rate_change_ratio_threshold: float | None = None
    motion_direction_reversal_cosine_threshold: float | None = None
    motion_curvature_flip_cosine_threshold: float | None = None
    motion_acceleration_l2_threshold: float | None = None
    motion_kink_rate_change_ratio_threshold: float | None = None
    motion_calm_shortlist_scale: float | None = None
    motion_kink_shortlist_bonus: int | None = None
    motion_calm_oracle_budget_scale: float | None = None
    motion_kink_oracle_budget_scale: float | None = None
    position_jump_tie_margin_abs: float | None = None
    reconstruction_tol: float | None = None
    grouping_mode: str | None = None


@dataclass(frozen=True)
class TrialObservation:
    trial_number: int
    params: dict[str, Any]
    objective: float
    status: str
    metrics: dict[str, Any]
    invalid_reasons: list[str]
    output_json: str | None = None
    spectra_json: str | None = None
    spectra_pdf: str | None = None
    trial_progress_json: str | None = None
    trial_partial_payload_json: str | None = None
    elapsed_s: float | None = None
    error: str | None = None


_DENSITY_FIRST_FEASIBLE_PROFILES = frozenset(
    {
        "high_amp_guarded_d_shape_live_v1",
        "high_amp_guarded_d_shape_turnwindow_v1",
        "high_amp_guarded_d_shape_expanded_v1",
        "high_amp_guarded_d_shape_site3_prefsite_autobaseline_v1",
        "high_amp_guarded_fidelity_first_site3_prefsite_autobaseline_v1",
        "high_amp_guarded_site3_full_surface_v2",
    }
)

_SITE3_PREFSITE_AUTOBASELINE_GUARDRAIL_MODES = {
    "high_amp_guarded_d_shape_site3_prefsite_autobaseline_v1": "d_shape_barrier_v1",
    "high_amp_guarded_fidelity_first_site3_prefsite_autobaseline_v1": "fidelity_first_barrier_v1",
}

_FULL_SITE3_ALGORITHM_PROFILES = frozenset({"high_amp_guarded_site3_full_surface_v2"})
_APPEND_PRUNE_NARROW_PROFILE = "append_live_guard_l2_v1"
_APPEND_PRUNE_RECOVERABILITY_PROFILE = "append_prune_recoverability_l2_v1"
_APPEND_PRUNE_TUNE_PROFILES = frozenset(
    {
        "append_prune_noharm_l2_v1",
        _APPEND_PRUNE_NARROW_PROFILE,
        _APPEND_PRUNE_RECOVERABILITY_PROFILE,
    }
)
_STRICT_QPU_FAITHFUL_RECOVERABILITY_PROFILE = "strict_qpu_faithful_recoverability_v1"
_STRICT_QPU_HH_RECOVERABILITY_PROFILE = "strict_qpu_hh_recoverability_v1"
_STRICT_QPU_FAITHFUL_APPEND_PRUNE_PROFILE = "strict_qpu_faithful_append_prune_recoverability_v1"
_STRICT_QPU_FAITHFUL_APPEND_PRUNE_GUARDRAIL_ONLY_PROFILE = (
    "strict_qpu_faithful_append_prune_guardrail_only_v1"
)
_STRICT_QPU_FAITHFUL_APPEND_PRUNE_AGGRESSIVE_PROFILE = (
    "strict_qpu_faithful_append_prune_aggressive_v1"
)
_STRICT_QPU_FAITHFUL_APPEND_PRUNE_PROFILES = frozenset(
    {
        _STRICT_QPU_FAITHFUL_APPEND_PRUNE_PROFILE,
        _STRICT_QPU_FAITHFUL_APPEND_PRUNE_GUARDRAIL_ONLY_PROFILE,
        _STRICT_QPU_FAITHFUL_APPEND_PRUNE_AGGRESSIVE_PROFILE,
    }
)
_STRICT_QPU_FAITHFUL_PROFILES = frozenset(
    {
        _STRICT_QPU_FAITHFUL_RECOVERABILITY_PROFILE,
        _STRICT_QPU_HH_RECOVERABILITY_PROFILE,
        _STRICT_QPU_FAITHFUL_APPEND_PRUNE_PROFILE,
        _STRICT_QPU_FAITHFUL_APPEND_PRUNE_GUARDRAIL_ONLY_PROFILE,
        _STRICT_QPU_FAITHFUL_APPEND_PRUNE_AGGRESSIVE_PROFILE,
    }
)
_STRICT_QPU_HH_PROFILES = frozenset({_STRICT_QPU_HH_RECOVERABILITY_PROFILE})
_STRICT_AUTO_EULER_GUARDRAIL_FIELD_NAMES = frozenset(
    {
        "integrator_euler_site_span_max",
        "integrator_euler_primary_density_span_max",
        "integrator_euler_energy_span_max",
    }
)

# Exact-free auto-Euler guardrails used by strict/QPU-faithful calibration.
# These thresholds match the HH recovery ablation guardrail surface and prevent
# Optuna from disabling observable-span checks while still allowing Euler in
# genuinely calm prepared-state observable regimes.
STRICT_AUTO_EULER_GUARDRAIL_ID = "strict_auto_euler_observable_guardrails_v1"
STRICT_AUTO_EULER_SITE_SPAN_DEFAULT = 1.0e-2
STRICT_AUTO_EULER_PRIMARY_DENSITY_SPAN_DEFAULT = 2.0e-2
STRICT_AUTO_EULER_ENERGY_SPAN_DEFAULT = 2.0e-3
STRICT_AUTO_EULER_SITE_SPAN_CHOICES = [5.0e-3, 1.0e-2, 2.0e-2]
STRICT_AUTO_EULER_PRIMARY_DENSITY_SPAN_CHOICES = [5.0e-3, 1.0e-2, 2.0e-2]
STRICT_AUTO_EULER_ENERGY_SPAN_CHOICES = [1.0e-3, 2.0e-3, 5.0e-3]


def _strict_auto_euler_guardrail_params() -> dict[str, float]:
    return {
        "integrator_euler_site_span_max": STRICT_AUTO_EULER_SITE_SPAN_DEFAULT,
        "integrator_euler_primary_density_span_max": STRICT_AUTO_EULER_PRIMARY_DENSITY_SPAN_DEFAULT,
        "integrator_euler_energy_span_max": STRICT_AUTO_EULER_ENERGY_SPAN_DEFAULT,
    }


def _strict_auto_euler_guardrail_search_space() -> dict[str, list[float]]:
    return {
        "integrator_euler_site_span_max": list(STRICT_AUTO_EULER_SITE_SPAN_CHOICES),
        "integrator_euler_primary_density_span_max": list(
            STRICT_AUTO_EULER_PRIMARY_DENSITY_SPAN_CHOICES
        ),
        "integrator_euler_energy_span_max": list(STRICT_AUTO_EULER_ENERGY_SPAN_CHOICES),
    }

_STRICT_QPU_HH_RECOVERABILITY_REFERENCE: dict[str, Any] = {
    "kind": "metadata_only_recoverability_target",
    "source": "artifacts/agent_runs/20260421_hh_l2_t8_good_trial_compile_audit_regression_v1/result.json",
    "note": (
        "Exact-assisted HH t=8 good-run calibration target. This reference is "
        "recorded only for human recoverability context; strict Optuna trial "
        "metrics, validity, and objective scoring do not consume exact-assisted "
        "or benchmark-exact fields."
    ),
    "controller_decision_use": "forbidden",
    "optuna_objective_feedback_use": "forbidden",
}

_HIDDEN_EXACT_V1_CFG_FIELD_MAP: dict[str, str] = {
    "forecast_score_gain_weight": "forecast_score_gain_weight",
    "forecast_score_rho_miss_weight": "forecast_score_rho_miss_weight",
    "forecast_score_step_residual_weight": "forecast_score_step_residual_weight",
    "forecast_score_condition_weight": "forecast_score_condition_weight",
    "forecast_score_theta_velocity_weight": "forecast_score_theta_velocity_weight",
    "forecast_score_displacement_weight": "forecast_score_displacement_weight",
    "forecast_accept_margin": "forecast_accept_margin",
    "append_no_harm_guard_enabled": "append_no_harm_guard_enabled",
    "append_no_harm_condition_ratio_cap": "append_no_harm_condition_ratio_cap",
    "append_no_harm_condition_abs_floor": "append_no_harm_condition_abs_floor",
    "append_no_harm_displacement_ratio_cap": "append_no_harm_displacement_ratio_cap",
    "append_no_harm_kink_min_step_gain_delta": "append_no_harm_kink_min_step_gain_delta",
    "append_no_harm_kink_max_condition_ratio": "append_no_harm_kink_max_condition_ratio",
    "append_no_harm_kink_max_displacement_ratio": "append_no_harm_kink_max_displacement_ratio",
    "append_no_harm_rho_only_min_step_gain_delta": "append_no_harm_rho_only_min_step_gain_delta",
    "append_no_harm_rho_only_condition_ratio_cap": "append_no_harm_rho_only_condition_ratio_cap",
    "append_no_harm_rho_only_step_residual_ratio_cap": "append_no_harm_rho_only_step_residual_ratio_cap",
    "append_no_harm_rho_only_displacement_ratio_cap": "append_no_harm_rho_only_displacement_ratio_cap",
    "primary_density_scale_floor": "exact_forecast_primary_density_scale_floor",
    "density_slope_scale_floor": "exact_forecast_density_slope_scale_floor",
    "doublon_scale_floor": "exact_forecast_doublon_scale_floor",
    "site_occupations_scale_floor": "exact_forecast_site_occupations_scale_floor",
    "energy_total_scale_floor": "exact_forecast_energy_total_scale_floor",
    "prune_protection_steps": "prune_protection_steps",
    "prune_schur_ladder_mode": "prune_schur_ladder_mode",
    "prune_schur_ladder_local_radius": "prune_schur_ladder_local_radius",
    "prune_schur_monotonicity_tol": "prune_schur_monotonicity_tol",
    "prune_loss_norm_epsilon": "prune_loss_norm_epsilon",
    "prune_state_jump_l2_hard_cap": "prune_state_jump_l2_hard_cap",
    "prune_differential_miss_tol": "prune_differential_miss_tol",
    "prune_high_miss_differential_enabled": "prune_high_miss_differential_enabled",
    "prune_projection_mode": "prune_projection_mode",
    "prune_projection_rounds": "prune_projection_rounds",
    "prune_projection_max_active_runtime": "prune_projection_max_active_runtime",
    "prune_projection_trust_radius": "prune_projection_trust_radius",
    "prune_projection_state_weight": "prune_projection_state_weight",
    "prune_projection_observable_weight": "prune_projection_observable_weight",
    "prune_projection_regularization": "prune_projection_regularization",
    "prune_ray_distance_tol": "prune_ray_distance_tol",
    "prune_shadow_enabled": "prune_shadow_enabled",
    "prune_shadow_horizon_steps": "prune_shadow_horizon_steps",
    "prune_shadow_score_tol": "prune_shadow_score_tol",
    "prune_shadow_score_increase_tol": "prune_shadow_score_increase_tol",
    "prune_shadow_scale_floor": "prune_shadow_scale_floor",
    "prune_persistence_window": "prune_persistence_window",
    "prune_persistence_required": "prune_persistence_required",
    "prune_appended_origin_bias_enabled": "prune_appended_origin_bias_enabled",
    "prune_appended_origin_target_policy": "prune_appended_origin_target_policy",
    "prune_appended_origin_grace_steps": "prune_appended_origin_grace_steps",
    "prune_initial_scaffold_grace_steps": "prune_initial_scaffold_grace_steps",
    "prune_active_block_theta_dot_rel_tol": "prune_active_block_theta_dot_rel_tol",
    "prune_active_block_theta_dot_abs_tol": "prune_active_block_theta_dot_abs_tol",
    "prune_active_block_theta_dot_abs_hard_tol": "prune_active_block_theta_dot_abs_hard_tol",
    "prune_appended_origin_bias_scale": "prune_appended_origin_bias_scale",
    "prune_appended_origin_bias_max_factor": "prune_appended_origin_bias_max_factor",
    "prune_stagnation_window": "prune_stagnation_window",
    "prune_stagnation_alpha": "prune_stagnation_alpha",
    "prune_stale_score_threshold": "prune_stale_score_threshold",
    "prune_max_candidates": "prune_max_candidates",
    "prune_cooldown_steps": "prune_cooldown_steps",
    "motion_calm_direction_cosine_threshold": "motion_calm_direction_cosine_threshold",
    "motion_calm_rate_change_ratio_threshold": "motion_calm_rate_change_ratio_threshold",
    "motion_direction_reversal_cosine_threshold": "motion_direction_reversal_cosine_threshold",
    "motion_curvature_flip_cosine_threshold": "motion_curvature_flip_cosine_threshold",
    "motion_acceleration_l2_threshold": "motion_acceleration_l2_threshold",
    "motion_kink_rate_change_ratio_threshold": "motion_kink_rate_change_ratio_threshold",
    "motion_calm_shortlist_scale": "motion_calm_shortlist_scale",
    "motion_kink_shortlist_bonus": "motion_kink_shortlist_bonus",
    "motion_calm_oracle_budget_scale": "motion_calm_oracle_budget_scale",
    "motion_kink_oracle_budget_scale": "motion_kink_oracle_budget_scale",
    "position_jump_tie_margin_abs": "position_jump_tie_margin_abs",
    "reconstruction_tol": "reconstruction_tol",
    "grouping_mode": "grouping_mode",
}


def _site3_prefsite_autobaseline_guardrail_mode(profile: str) -> str | None:
    return _SITE3_PREFSITE_AUTOBASELINE_GUARDRAIL_MODES.get(str(profile))


def _is_strict_qpu_faithful_profile(profile: str) -> bool:
    return str(profile) in _STRICT_QPU_FAITHFUL_PROFILES


def _is_strict_guardrail_only_profile(profile: str) -> bool:
    return str(profile) == _STRICT_QPU_FAITHFUL_APPEND_PRUNE_GUARDRAIL_ONLY_PROFILE


def _is_strict_append_prune_aggressive_profile(profile: str) -> bool:
    return str(profile) == _STRICT_QPU_FAITHFUL_APPEND_PRUNE_AGGRESSIVE_PROFILE


def _is_strict_append_prune_controller_profile(profile: str) -> bool:
    return str(profile) in _STRICT_QPU_FAITHFUL_APPEND_PRUNE_PROFILES


def _is_strict_qpu_hh_profile(profile: str) -> bool:
    """Return true only for the legacy HH strict profile literal."""

    return str(profile) in _STRICT_QPU_HH_PROFILES


def _is_diagnostic_generic_exact_profile(profile: str) -> bool:
    return str(profile) == "generic_l2_exact_v1" or str(profile) in _APPEND_PRUNE_TUNE_PROFILES


def _summary_declares_successful_stable_early_stop(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("full_horizon_gate_passed") is True
        and summary.get("full_horizon_successful_early_stop") is True
        and str(summary.get("full_horizon_completion_kind", "")) == "stable_early_stop"
    )


def _strict_payload_flags(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = (
        dict(payload.get("summary", {}))
        if isinstance(payload.get("summary", {}), Mapping)
        else {}
    )
    route_config = (
        dict(payload.get("route_config", {}))
        if isinstance(payload.get("route_config", {}), Mapping)
        else {}
    )
    runtime_contract = (
        dict(payload.get("runtime_contract", {}))
        if isinstance(payload.get("runtime_contract", {}), Mapping)
        else {}
    )
    strict_qpu_hh = bool(summary.get("strict_qpu_hh", route_config.get("strict_qpu_hh", False)))
    strict_qpu_faithful = bool(
        summary.get(
            "strict_qpu_faithful",
            route_config.get("strict_qpu_faithful", strict_qpu_hh),
        )
    )
    raw_family = (
        summary.get("strict_qpu_family")
        or route_config.get("strict_qpu_family")
        or route_config.get("problem_family")
        or runtime_contract.get("problem_family")
        or ("hh" if strict_qpu_hh else "unknown")
    )
    strict_qpu_family = str(raw_family or "unknown")
    if strict_qpu_family == "hh" and strict_qpu_faithful:
        strict_qpu_hh = True
    return {
        "strict_qpu_faithful": bool(strict_qpu_faithful),
        "strict_qpu_hh": bool(strict_qpu_hh),
        "strict_qpu_family": strict_qpu_family,
    }


def _strict_recoverability_reference(flags: Mapping[str, Any]) -> dict[str, Any]:
    family = str(flags.get("strict_qpu_family", "unknown") or "unknown")
    if bool(flags.get("strict_qpu_hh", False)) or family == "hh":
        return dict(_STRICT_QPU_HH_RECOVERABILITY_REFERENCE)
    return {
        "kind": "metadata_only_strict_qpu_faithful_diagnostic_context",
        "strict_qpu_family": family,
        "note": (
            "Strict QPU-faithful Optuna feedback for this family is measured-only. "
            "Exact references may be attached after execution for diagnostics, but "
            "they are forbidden as controller or objective feedback."
        ),
        "controller_decision_use": "forbidden",
        "optuna_objective_feedback_use": "forbidden",
    }


def _strict_analysis_status(flags: Mapping[str, Any]) -> str:
    if bool(flags.get("strict_qpu_hh", False)) or str(flags.get("strict_qpu_family")) == "hh":
        return "strict_qpu_hh_exact_analysis_skipped"
    return "strict_qpu_faithful_exact_analysis_skipped"


def _resolve_csv_override(*, explicit: str | None, preset_key: str, preset_map: Mapping[str, str]) -> str:
    if explicit is not None:
        return str(explicit)
    return str(preset_map[str(preset_key)])


def _resolve_horizon_override(params: TrialParams) -> tuple[int, str]:
    if params.horizon_spec not in {None, ""}:
        raw_steps, raw_weights = str(params.horizon_spec).split("|", 1)
        return int(raw_steps), str(raw_weights)
    steps, weights = _HORIZON_PRESETS[str(params.horizon_mode)]
    return int(steps), ",".join(str(float(weight)) for weight in weights)


def _hidden_exact_v1_cfg_overrides(params: TrialParams) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for trial_field, cfg_field in _HIDDEN_EXACT_V1_CFG_FIELD_MAP.items():
        value = getattr(params, str(trial_field))
        if value is not None:
            overrides[str(cfg_field)] = value
    return overrides


def _resolve_optuna_route_context(base_cfg: BaseRunConfig) -> OptunaRouteContext | None:
    if not _is_diagnostic_generic_exact_profile(str(base_cfg.study_profile)):
        return None
    artifact_path = Path(base_cfg.artifact_json).expanduser()
    if not artifact_path.exists():
        return None
    runtime_input = load_scaffold_runtime_input(
        artifact_path,
        loader_mode=str(base_cfg.loader_mode),
        tag=f"optuna_route_context_{artifact_path.stem}",
        generator_family=str(base_cfg.generator_family),
        fallback_family=str(base_cfg.fallback_family),
    )
    resolved_problem = getattr(runtime_input, "resolved_problem", None)
    raw_family = getattr(resolved_problem, "family_key", None)
    problem_family = "hh" if raw_family in {None, ""} else str(raw_family).strip().lower()
    policy = policy_for_family(problem_family)
    locked_primary_density_target_mode = (
        "auto" if policy.exact_v1_requires_reference_mode is not None else None
    )
    return OptunaRouteContext(
        problem_family=str(policy.family_key),
        locked_primary_density_target_mode=locked_primary_density_target_mode,
    )


def _apply_route_context_to_trial_params(
    params: TrialParams,
    route_ctx: OptunaRouteContext | None,
) -> TrialParams:
    if (
        route_ctx is not None
        and str(route_ctx.locked_primary_density_target_mode or "").strip().lower() == "auto"
        and params.primary_density_target_mode != "auto"
    ):
        return replace(params, primary_density_target_mode="auto")
    return params


def _apply_route_context_to_search_space(
    space: Mapping[str, Sequence[Any]],
    route_ctx: OptunaRouteContext | None,
) -> dict[str, list[Any]]:
    locked = (
        None
        if route_ctx is None
        else str(route_ctx.locked_primary_density_target_mode or "").strip().lower()
    )
    out = {str(key): list(values) for key, values in space.items()}
    if locked == "auto":
        out["primary_density_target_mode"] = ["auto"]
    return out


def _build_controller_bundle_with_optuna_overrides(
    *,
    args: argparse.Namespace,
    params: TrialParams,
) -> dict[str, Any]:
    if strict_qpu_faithful_requested(args):
        return build_generic_controller_bundle_from_args(args)
    cfg = build_controller_config(args)
    hidden_overrides = _hidden_exact_v1_cfg_overrides(params)
    if hidden_overrides:
        cfg = replace(cfg, **hidden_overrides)
    seed = build_controller_seed_from_args(args, cfg=cfg)
    return finalize_controller_bundle_from_seed(args, seed=seed)


def _run_realtime_from_args_with_optuna_overrides(
    *,
    args: argparse.Namespace,
    params: TrialParams,
) -> dict[str, Any]:
    if strict_qpu_faithful_requested(args):
        return run_generic_realtime_from_args(args)
    output_json = Path(args.output_json).expanduser().resolve()
    bundle = _build_controller_bundle_with_optuna_overrides(args=args, params=params)
    loaded = bundle["loaded"]
    cfg = bundle["cfg"]
    oracle_config = bundle["oracle_config"]
    resolved_drive_config = bundle["drive_config"]
    controller = bundle["controller"]
    result = controller.run()
    runtime_input = bundle.get("runtime_input")
    resolved_problem = (
        None if runtime_input is None else getattr(runtime_input, "resolved_problem", None)
    )
    problem_family = (
        str(getattr(resolved_problem, "family_key"))
        if getattr(resolved_problem, "family_key", None) not in {None, ""}
        else "hh"
    )
    payload = build_output_payload(
        args=args,
        loaded=loaded,
        cfg=cfg,
        drive_config=resolved_drive_config,
        oracle_config=oracle_config,
        result=result,
        problem_family=str(problem_family),
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _site3_full_surface_search_space() -> dict[str, list[Any]]:
    return {
        "oracle_selection_policy": [
            "measured_gain_commit_veto",
            "measured_topk_oracle_energy",
        ],
        "candidate_step_scales": [
            _STEP_SCALE_PRESETS["default"],
            _STEP_SCALE_PRESETS["wide"],
            _STEP_SCALE_PRESETS["high_amp_dense"],
        ],
        "baseline_blend_weights": [
            _BLEND_WEIGHT_PRESETS["default"],
            _BLEND_WEIGHT_PRESETS["tight"],
            _BLEND_WEIGHT_PRESETS["high_amp"],
        ],
        "baseline_gain_scales": [
            _GAIN_SCALE_PRESETS["off"],
            _GAIN_SCALE_PRESETS["mild"],
            _GAIN_SCALE_PRESETS["wide"],
            _GAIN_SCALE_PRESETS["high_amp"],
        ],
        "horizon_spec": [
            _HORIZON_SPEC_PRESETS["lead3"],
            _HORIZON_SPEC_PRESETS["lead4"],
            _HORIZON_SPEC_PRESETS["high_amp_lead4"],
        ],
        "baseline_step_refine_rounds": [0, 1, 2],
        "include_tangent_secant_proposal": [False, True],
        "baseline_proposal_mode": ["norm_locked_blend_v1", "anticipatory_drive_basis_v1"],
        "primary_density_target_mode": ["auto", "pair_difference", "staggered"],
        "high_miss_no_admit_policy": ["bounded_stay_advance", "repair_stop"],
        "miss_threshold": [0.03, 0.05, 0.08],
        "miss_abs_threshold": [0.0, 1.0e-4, 1.0e-3, 1.0e-2],
        "miss_persistence_spec": ["1:1", "3:2", "3:3", "5:3"],
        "gain_ratio_threshold": [1.0e-6, 1.0e-4, 0.01, 0.02, 0.03],
        "append_margin_abs": [1.0e-6, 1.0e-5, 5.0e-5],
        "trust_radius": [0.85, 1.0, 1.15],
        "signed_energy_lead_limit": [3.0, 3.5, 4.0],
        "shortlist_size": [4, 6, 8],
        "shortlist_fraction": [0.15, 0.25, 0.35],
        "active_window_size": [3, 4, 6],
        "max_probe_positions": [4, 6, 8],
        "regularization_lambda": [1.0e-10, 1.0e-8, 1.0e-6],
        "candidate_regularization_lambda": [1.0e-10, 1.0e-8, 1.0e-6],
        "pinv_rcond": [1.0e-12, 1.0e-10, 1.0e-8],
        "compile_penalty_weight": [0.0, 0.05, 0.1],
        "measurement_penalty_weight": [0.0, 0.02, 0.05],
        "directional_penalty_weight": [0.0, 0.01, 0.03],
        "confirm_score_mode": ["compressed_whitened_v1", "exact_gain_ratio"],
        "confirm_compress_fraction": [0.25, 0.5, 0.75],
        "confirm_compress_min_modes": [1, 2],
        "confirm_compress_max_modes": [4, 8, 12],
        "primary_density_weight": [1.5, 2.0, 2.5, 3.0],
        "tracking_staggered_weight": [0.0, 0.5, 1.0, 2.0],
        "site_weight": [0.0, 0.125, 0.25, 0.5],
        "energy_weight": [0.0, 0.125, 0.25, 0.5],
        "density_slope_weight": [1.0, 2.0, 3.0],
        "fidelity_weight": [0.0, 0.25, 0.5, 1.0],
        "doublon_weight": [0.0, 0.5, 1.0, 2.0],
        "density_curvature_weight": [0.0, 25.0, 50.0, 100.0],
        "density_excursion_under_weight": [0.0, 150.0, 225.0, 300.0],
        "density_excursion_over_weight": [0.0, 50.0, 100.0],
        "density_sign_lag_weight": [0.1, 0.25, 0.5],
        "density_postcross_wrong_sign_weight": [0.0, 0.1, 0.25, 0.5],
        "drive_harmonic_weight": [0.0, 0.1, 0.25],
        "energy_slope_weight": [0.0, 250.0, 500.0, 750.0],
        "energy_curvature_weight": [0.0, 100.0, 200.0, 300.0],
        "energy_excursion_under_weight": [0.0, 450.0, 900.0, 1350.0],
        "energy_excursion_over_weight": [0.0, 60.0, 120.0, 180.0],
        "energy_excursion_rel_tolerance": [0.0, 0.02, 0.03, 0.05],
        "repeat_reopen_mode": ["off", "sign_reversal_window"],
        "density_first_target_gain_floor": [0.015, 0.02, 0.025, 0.03],
        "below_floor_probe_target_gain_floor": [0.02, 0.025, 0.03, 0.035],
        "sign_lag_window_activation": [0.0, 0.05, 0.1, 0.15],
        "sign_lag_window_target_gain_floor": [None, 0.0025, 0.005, 0.01],
        "postcross_wrong_sign_activation": [0.0, 0.05, 0.1],
        "postcross_wrong_sign_target_gain_floor": [None, 0.0025, 0.005, 0.01],
        "postcross_compare_diag": [False, True],
        "below_floor_energy_safe_d_shape_escape": [False, True],
        "below_floor_energy_safe_turn_escape": [False, True],
        "d_shape_turn_window_abs_activation": [0.0, 0.02, 0.03, 0.04, 0.05, 0.06],
        "d_shape_outside_turn_below_floor_probe_stall_threshold": [0, 1, 2, 3, 5, 7],
        "d_shape_pre_turn_shadow_bridge": [False, True],
        "single_surface_commit_law": [False, True],
        "guardrail_mode": ["d_shape_barrier_v1", "fidelity_first_barrier_v1"],
        "fidelity_loss_tol": [0.005, 0.01, 0.015],
        "abs_energy_error_increase_tol": [0.01, 0.02, 0.03],
        "total_occupation_error_increase_tol": [0.005, 0.01, 0.015],
        "primary_density_scale_floor": [1.0e-8, 1.0e-6, 1.0e-4],
        "density_slope_scale_floor": [1.0e-8, 1.0e-6, 1.0e-4],
        "doublon_scale_floor": [1.0e-8, 1.0e-6, 1.0e-4],
        "site_occupations_scale_floor": [1.0e-8, 1.0e-6, 1.0e-4],
        "energy_total_scale_floor": [1.0e-8, 1.0e-6, 1.0e-4],
        "progress_observable_window": [None, 12, 16, 24],
        "progress_early_stop_min_checkpoint": [None, 96, 120, 160],
        "progress_early_stop_site_error_mean_max": [None, 0.25, 0.3, 0.35],
        "progress_early_stop_primary_density_error_mean_max": [None, 0.5, 0.6, 0.7],
        "progress_early_stop_energy_error_mean_max": [None, 0.05, 0.1, 0.15],
        "integrator_policy": ["auto_euler_rk4"],
        "integrator_columnarity_threshold": [0.70, 0.80, 0.90],
        "integrator_curvature_threshold": [0.10, 0.20, 0.35],
        "integrator_euler_fs_error_threshold": [1.0e-3, 2.5e-3, 5.0e-3],
        "integrator_condition_max": [1.0e10, 1.0e12, 1.0e14],
        "integrator_euler_min_time_fraction": [0.35, 0.50, 0.65],
        "integrator_euler_observable_window": [12, 16, 24],
        "integrator_euler_site_span_max": [None, 5.0e-3, 1.0e-2, 2.0e-2, 5.0e-2],
        "integrator_euler_primary_density_span_max": [None, 5.0e-3, 1.0e-2, 2.0e-2, 5.0e-2],
        "integrator_euler_energy_span_max": [None],
        "prune_mode": ["off", "exact_local_v1"],
        "prune_miss_threshold": [0.01, 0.02, 0.05],
        "prune_loss_threshold": [0.005, 0.01, 0.02],
        "prune_theta_block_tol": [0.02, 0.05, 0.1],
        "prune_state_jump_l2_tol": [0.02, 0.05, 0.1],
        "prune_safe_miss_increase_tol": [0.005, 0.01, 0.02],
        "prune_protection_steps": [0, 1, 2, 4],
        "prune_stagnation_window": [1, 3, 5],
        "prune_stagnation_alpha": [0.25, 0.5, 0.75],
        "prune_stale_score_threshold": [0.5, 0.75, 1.0],
        "prune_max_candidates": [1, 2, 4],
        "prune_cooldown_steps": [0, 1, 2, 4],
        "motion_calm_direction_cosine_threshold": [0.95, 0.98, 0.995],
        "motion_calm_rate_change_ratio_threshold": [0.05, 0.15, 0.3],
        "motion_direction_reversal_cosine_threshold": [-0.25, -0.1, -0.05, 0.0],
        "motion_curvature_flip_cosine_threshold": [-0.25, -0.1, -0.05, 0.0],
        "motion_acceleration_l2_threshold": [0.01, 0.05, 0.1],
        "motion_kink_rate_change_ratio_threshold": [0.25, 0.5, 0.75, 1.0],
        "motion_calm_shortlist_scale": [0.25, 0.5, 0.75, 1.0],
        "motion_kink_shortlist_bonus": [0, 1, 2, 4],
        "motion_calm_oracle_budget_scale": [0.25, 0.5, 0.75, 1.0],
        "motion_kink_oracle_budget_scale": [1.0, 1.5, 2.0, 3.0],
        "position_jump_tie_margin_abs": [1.0e-8, 1.0e-6, 1.0e-4],
        "reconstruction_tol": [1.0e-12, 1.0e-10, 1.0e-8],
        "grouping_mode": ["qwc_basis_cover_reuse"],
    }


def _suggest_trial_params_from_space(
    trial: Any,
    *,
    baseline: TrialParams,
    space: Mapping[str, Sequence[Any]],
) -> TrialParams:
    payload = asdict(baseline)
    for key, choices in space.items():
        if key in {"trust_radius", "signed_energy_lead_limit"} and not bool(
            payload.get("include_tangent_secant_proposal", True)
        ):
            continue
        if key in {
            "confirm_compress_fraction",
            "confirm_compress_min_modes",
            "confirm_compress_max_modes",
        } and str(payload.get("confirm_score_mode", "")) == "exact_gain_ratio":
            continue
        if key in {
            "prune_miss_threshold",
            "prune_loss_threshold",
            "prune_theta_block_tol",
            "prune_state_jump_l2_tol",
            "prune_safe_miss_increase_tol",
            "prune_schur_ladder_local_radius",
            "prune_schur_monotonicity_tol",
            "prune_loss_norm_epsilon",
            "prune_differential_miss_tol",
            "prune_high_miss_differential_enabled",
            "prune_projection_mode",
            "prune_projection_rounds",
            "prune_projection_max_active_runtime",
            "prune_projection_trust_radius",
            "prune_projection_regularization",
            "prune_ray_distance_tol",
            "prune_shadow_enabled",
            "prune_shadow_horizon_steps",
            "prune_shadow_score_increase_tol",
            "prune_persistence_window",
            "prune_persistence_required",
            "prune_appended_origin_bias_enabled",
            "prune_appended_origin_target_policy",
            "prune_appended_origin_grace_steps",
            "prune_initial_scaffold_grace_steps",
            "prune_appended_origin_bias_scale",
            "prune_appended_origin_bias_max_factor",
            "prune_protection_steps",
            "prune_stagnation_window",
            "prune_stagnation_alpha",
            "prune_stale_score_threshold",
            "prune_max_candidates",
            "prune_cooldown_steps",
        } and str(payload.get("prune_mode", "off")) == "off":
            continue
        payload[key] = trial.suggest_categorical(str(key), list(choices))
    return TrialParams(**payload)


def _default_objective_weights(profile: str = "generic_l2_exact_v1") -> ObjectiveWeights:
    if _is_strict_append_prune_aggressive_profile(profile):
        return ObjectiveWeights(
            pair_mae_over_exact_span=0.0,
            epsilon_osc_pair=0.0,
            dominant_peak_abs_omega_error=0.0,
            pair_corr_defect=0.0,
            mean_abs_site_occupations_error=0.0,
            mean_abs_energy_total_error=0.0,
            mean_signed_energy_total_error_abs=0.0,
            max_abs_energy_total_error=0.0,
            energy_total_excursion_under_response_mean=0.0,
            energy_total_excursion_under_response_max=0.0,
            fidelity_defect=0.0,
            final_runtime_parameter_count=0.002,
            append_count=0.03,
            prune_count=1.0,
            rk4_count=0.0,
        )
    if _is_strict_append_prune_controller_profile(profile):
        return ObjectiveWeights(
            pair_mae_over_exact_span=0.0,
            epsilon_osc_pair=0.0,
            dominant_peak_abs_omega_error=0.0,
            pair_corr_defect=0.0,
            mean_abs_site_occupations_error=0.0,
            mean_abs_energy_total_error=0.0,
            mean_signed_energy_total_error_abs=0.0,
            max_abs_energy_total_error=0.0,
            energy_total_excursion_under_response_mean=0.0,
            energy_total_excursion_under_response_max=0.0,
            fidelity_defect=0.0,
            final_runtime_parameter_count=0.004,
            append_count=0.08,
            prune_count=0.05,
            rk4_count=0.0,
        )
    if _is_strict_qpu_faithful_profile(profile):
        return ObjectiveWeights(
            pair_mae_over_exact_span=0.0,
            epsilon_osc_pair=0.0,
            dominant_peak_abs_omega_error=0.0,
            pair_corr_defect=0.0,
            mean_abs_site_occupations_error=0.0,
            mean_abs_energy_total_error=0.0,
            mean_signed_energy_total_error_abs=0.0,
            max_abs_energy_total_error=0.0,
            energy_total_excursion_under_response_mean=0.0,
            energy_total_excursion_under_response_max=0.0,
            fidelity_defect=0.0,
            final_runtime_parameter_count=0.005,
            append_count=0.02,
            prune_count=0.0,
            rk4_count=0.0,
        )
    if str(profile) in _APPEND_PRUNE_TUNE_PROFILES:
        return ObjectiveWeights(
            pair_mae_over_exact_span=0.75,
            epsilon_osc_pair=0.5,
            dominant_peak_abs_omega_error=0.25,
            pair_corr_defect=0.25,
            mean_abs_site_occupations_error=2.0,
            mean_abs_energy_total_error=6.0,
            mean_signed_energy_total_error_abs=80.0,
            max_abs_energy_total_error=8.0,
            energy_total_excursion_under_response_mean=35.0,
            energy_total_excursion_under_response_max=12.0,
            fidelity_defect=0.5,
            final_runtime_parameter_count=0.001,
            append_count=0.01,
            prune_count=0.0,
            rk4_count=0.001,
        )
    if str(profile) in _DENSITY_FIRST_FEASIBLE_PROFILES:
        return ObjectiveWeights(
            pair_mae_over_exact_span=1.0,
            epsilon_osc_pair=0.75,
            dominant_peak_abs_omega_error=0.5,
            pair_corr_defect=0.25,
            mean_abs_site_occupations_error=0.25,
            mean_abs_energy_total_error=0.0,
            mean_signed_energy_total_error_abs=0.0,
            max_abs_energy_total_error=0.0,
            energy_total_excursion_under_response_mean=0.0,
            energy_total_excursion_under_response_max=0.0,
            fidelity_defect=0.0,
            final_runtime_parameter_count=0.01,
            append_count=0.02,
            prune_count=0.02,
            rk4_count=0.001,
        )
    return ObjectiveWeights()


def _default_validity_gates(profile: str = "generic_l2_exact_v1") -> ValidityGates:
    if _is_strict_qpu_faithful_profile(profile):
        return ValidityGates()
    if str(profile) in _APPEND_PRUNE_TUNE_PROFILES:
        return ValidityGates(
            max_mean_abs_energy_total_error=0.05,
            max_final_abs_energy_total_error=0.05,
            min_fidelity_exact=0.90,
            max_mean_total_occupation_abs_error=0.05,
        )
    if str(profile) in _DENSITY_FIRST_FEASIBLE_PROFILES:
        return ValidityGates(
            max_mean_abs_energy_total_error=0.25,
            max_final_abs_energy_total_error=0.25,
            min_fidelity_exact=0.90,
            max_mean_total_occupation_abs_error=0.05,
        )
    return ValidityGates()


def _import_optuna() -> Any:
    try:
        import optuna  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Optuna is not installed. Install it before running this harness, for example via `python -m pip install optuna`."
        ) from exc
    return optuna


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_csv(values: str | None) -> list[str]:
    if values in {None, ""}:
        return []
    return [str(x).strip() for x in str(values).split(",") if str(x).strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_class_settings_lock_path(base_cfg: BaseRunConfig) -> Path:
    raw = str(base_cfg.class_settings_lock_json or PAPER_II_CLASS_SETTINGS_LOCK_JSON)
    return Path(raw).expanduser().resolve()


def _load_class_settings_lock(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"class settings lock not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"class settings lock must be a JSON object: {path}")
    return payload


def _locked_class_settings_entry(base_cfg: BaseRunConfig) -> Mapping[str, Any]:
    tuning_class = str(base_cfg.tuning_class or "").strip()
    if not tuning_class:
        raise ValueError(
            "guardrail-only strict profiles require --tuning-class so the class lock can be applied"
        )
    lock_path = _resolve_class_settings_lock_path(base_cfg)
    lock_payload = _load_class_settings_lock(lock_path)
    settings = lock_payload.get("settings", [])
    if not isinstance(settings, Sequence) or isinstance(settings, (str, bytes)):
        raise ValueError(f"class settings lock has no settings list: {lock_path}")
    matches = [
        entry
        for entry in settings
        if isinstance(entry, Mapping) and str(entry.get("tuning_class", "")) == tuning_class
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one settings lock entry for tuning_class={tuning_class!r} in {lock_path}; found {len(matches)}"
        )
    entry = dict(matches[0])
    payload = entry.get("settings_payload", {})
    if not isinstance(payload, Mapping):
        raise ValueError(f"settings lock entry for {tuning_class!r} lacks settings_payload")
    if bool(entry.get("strict_online_feedback_exact_free")) is not True:
        raise ValueError(f"settings lock entry for {tuning_class!r} is not marked exact-free")
    if str(entry.get("algorithm_id", "")) not in {"dyn_controller_full", ""}:
        raise ValueError(f"unexpected algorithm_id for {tuning_class!r}: {entry.get('algorithm_id')!r}")
    if str(entry.get("settings_kind", "")) not in {"controller", ""}:
        raise ValueError(f"unexpected settings_kind for {tuning_class!r}: {entry.get('settings_kind')!r}")
    controller_mode = str(payload.get("checkpoint_controller_mode", "observable_estimator_v1"))
    if controller_mode not in {"observable_estimator_v1", "observable_v1"}:
        raise ValueError(f"settings lock entry for {tuning_class!r} is not observable-estimator mode")
    exact_mode = payload.get("checkpoint_controller_exact_input_mode", payload.get("checkpoint_controller_reference_mode", "off"))
    if str(exact_mode) != "off":
        raise ValueError(f"settings lock entry for {tuning_class!r} uses exact controller input mode: {exact_mode!r}")
    return entry


def _trial_params_from_class_settings_payload(payload: Mapping[str, Any]) -> TrialParams:
    allowed = set(TrialParams.__dataclass_fields__)
    base = asdict(_baseline_trial_params(profile=_STRICT_QPU_FAITHFUL_APPEND_PRUNE_PROFILE))
    for key, value in payload.items():
        if key in allowed:
            base[str(key)] = value
    base.update(_strict_auto_euler_guardrail_params())
    return TrialParams(**base)


def _locked_class_trial_params(base_cfg: BaseRunConfig) -> TrialParams:
    entry = _locked_class_settings_entry(base_cfg)
    payload = entry.get("settings_payload", {})
    if not isinstance(payload, Mapping):
        raise ValueError("class settings lock entry lacks settings_payload")
    return _trial_params_from_class_settings_payload(payload)


def _apply_guardrail_only_class_lock_overrides(params: TrialParams, base_cfg: BaseRunConfig) -> TrialParams:
    if not _is_strict_guardrail_only_profile(base_cfg.study_profile):
        return params
    locked_payload = asdict(_locked_class_trial_params(base_cfg))
    proposed_payload = asdict(params)
    for key in _STRICT_AUTO_EULER_GUARDRAIL_FIELD_NAMES:
        locked_payload[key] = proposed_payload.get(key, locked_payload.get(key))
    return TrialParams(**locked_payload)


def _class_settings_lock_metadata(base_cfg: BaseRunConfig) -> dict[str, Any] | None:
    if not _is_strict_guardrail_only_profile(base_cfg.study_profile):
        return None
    path = _resolve_class_settings_lock_path(base_cfg)
    entry = _locked_class_settings_entry(base_cfg)
    return {
        "class_settings_lock_json": str(path),
        "class_settings_lock_sha256": _sha256_file(path),
        "locked_settings_id": entry.get("settings_id"),
        "locked_study_profile": entry.get("study_profile"),
        "locked_training_role": entry.get("training_role"),
        "frozen_fields_except": sorted(_STRICT_AUTO_EULER_GUARDRAIL_FIELD_NAMES),
    }


def _write_progress(output_dir: Path, payload: Mapping[str, Any]) -> None:
    _write_json(output_dir / "progress.json", payload)


def _resolved_pair(pair_spec: str, *, num_sites: int) -> tuple[int, int] | None:
    text = str(pair_spec).strip().lower()
    if text in {"", "auto"}:
        return (0, 1) if int(num_sites) == 2 else None
    if text in {"none", "off"}:
        return None
    parts = [chunk.strip() for chunk in str(pair_spec).split(",") if chunk.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected pair like '0,1', got {pair_spec!r}.")
    pair = (int(parts[0]), int(parts[1]))
    if min(pair) < 0 or max(pair) >= int(num_sites) or pair[0] == pair[1]:
        raise ValueError(f"Invalid pair {pair!r} for num_sites={num_sites}.")
    return pair


def _eps_osc(ctrl: np.ndarray, exact: np.ndarray) -> float:
    ctrl_fluct = np.asarray(ctrl - np.mean(ctrl), dtype=float)
    exact_fluct = np.asarray(exact - np.mean(exact), dtype=float)
    denom = float(np.linalg.norm(exact_fluct))
    if denom <= 1.0e-12:
        return float("inf")
    return float(np.linalg.norm(ctrl_fluct - exact_fluct) / denom)


def _corrcoef(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.size == 0 or rhs.size == 0:
        return float("nan")
    if float(np.std(lhs)) <= 1.0e-12 or float(np.std(rhs)) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(lhs, rhs)[0, 1])


def _as_finite_float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _safe_min(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    return float(np.min(finite)) if finite.size else float("nan")


def _safe_max(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    return float(np.max(finite)) if finite.size else float("nan")


def _safe_last(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    return float(finite[-1]) if finite.size else float("nan")


def _safe_ptp(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.ptp(finite)) if finite.size else float("nan")


def _safe_total_variation(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return float("nan")
    return float(np.sum(np.abs(np.diff(finite))))


def _safe_roughness(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 3:
        return float("nan")
    return float(np.sum(np.abs(np.diff(finite, n=2))))


def _tail_series_metrics(prefix: str, values: Sequence[float]) -> dict[str, float]:
    return {
        f"{prefix}_range": _safe_ptp(values),
        f"{prefix}_total_variation": _safe_total_variation(values),
        f"{prefix}_roughness": _safe_roughness(values),
    }


def _ratio_or_nan(num: float, denom: float, *, eps: float = 1.0e-12) -> float:
    try:
        lhs = float(num)
        rhs = float(denom)
    except Exception:
        return float("nan")
    if not math.isfinite(lhs) or not math.isfinite(rhs) or abs(rhs) <= float(eps):
        return float("nan")
    return float(lhs / rhs)


def _row_site0_observable(row: Mapping[str, Any]) -> float:
    sites = row.get("site_occupations")
    if isinstance(sites, (list, tuple)) and len(sites) > 0:
        return _as_finite_float_or_nan(sites[0])
    return float("nan")


def _row_energy_observable(row: Mapping[str, Any]) -> float:
    return _as_finite_float_or_nan(row.get("energy_total", row.get("energy_total_controller")))


def _row_primary_observable(row: Mapping[str, Any]) -> float:
    value = _as_finite_float_or_nan(row.get("primary_density"))
    if math.isfinite(value):
        return value
    value = _as_finite_float_or_nan(row.get("staggered"))
    if math.isfinite(value):
        return value
    return _row_site0_observable(row)


def _strict_observable_tail_metrics(
    *,
    trajectory: Sequence[Mapping[str, Any]],
    base_cfg: BaseRunConfig,
) -> dict[str, Any]:
    """Exact-free observable-history diagnostics for strict QPU-faithful trials.

    These metrics deliberately read only controller/prepared-state observable
    fields and integrator telemetry. They are intended for diagnosis and future
    exact-free objective versions; they must not read ED/reference columns.
    """

    rows = [dict(row) for row in trajectory if isinstance(row, Mapping)]
    times = np.asarray([_as_finite_float_or_nan(row.get("time", row.get("physical_time"))) for row in rows])
    finite_time_mask = np.isfinite(times)
    if not bool(np.any(finite_time_mask)):
        return {
            "strict_observable_tail_metrics_schema": "strict_observable_tail_metrics_v1",
            "strict_observable_tail_metrics_exact_free": True,
            "strict_tail_start_time": float("nan"),
            "strict_tail_sample_count": 0,
            "strict_tail_euler_count": 0,
            "strict_tail_rk4_count": 0,
            "strict_tail_euler_fraction": float("nan"),
            "strict_first_euler_time": float("nan"),
            "strict_post_first_euler_sample_count": 0,
            "strict_post_first_euler_euler_count": 0,
            "strict_post_first_euler_euler_fraction": float("nan"),
        }

    expected_t_final = float(base_cfg.t_final)
    final_time = _safe_last(times.tolist())
    horizon = expected_t_final if math.isfinite(expected_t_final) and expected_t_final > 0.0 else final_time
    tail_start = max(0.65 * float(horizon), float(horizon) - 1.5)
    tail_mask = finite_time_mask & (times >= tail_start - 1.0e-12)
    integrators = [str(row.get("integrator_used", "")) for row in rows]
    euler_indices = [
        idx
        for idx, (ok, used) in enumerate(zip(finite_time_mask.tolist(), integrators))
        if bool(ok) and str(used) == "euler"
    ]
    first_euler_time = float(times[euler_indices[0]]) if euler_indices else float("nan")
    post_euler_mask = (
        finite_time_mask & (times >= first_euler_time - 1.0e-12)
        if math.isfinite(first_euler_time)
        else np.zeros(times.shape, dtype=bool)
    )
    pre_euler_mask = (
        finite_time_mask & (times < first_euler_time - 1.0e-12)
        if math.isfinite(first_euler_time)
        else finite_time_mask
    )

    def _values(mask: np.ndarray, getter: Callable[[Mapping[str, Any]], float]) -> list[float]:
        return [getter(row) for row, keep in zip(rows, mask.tolist()) if bool(keep)]

    def _count(mask: np.ndarray, name: str) -> int:
        return int(sum(1 for keep, used in zip(mask.tolist(), integrators) if bool(keep) and str(used) == name))

    out: dict[str, Any] = {
        "strict_observable_tail_metrics_schema": "strict_observable_tail_metrics_v1",
        "strict_observable_tail_metrics_exact_free": True,
        "strict_tail_start_time": float(tail_start),
        "strict_tail_sample_count": int(np.count_nonzero(tail_mask)),
        "strict_tail_euler_count": _count(tail_mask, "euler"),
        "strict_tail_rk4_count": _count(tail_mask, "rk4"),
        "strict_tail_euler_fraction": float(
            _count(tail_mask, "euler") / max(int(np.count_nonzero(tail_mask)), 1)
        ),
        "strict_first_euler_time": float(first_euler_time),
        "strict_post_first_euler_sample_count": int(np.count_nonzero(post_euler_mask)),
        "strict_post_first_euler_euler_count": _count(post_euler_mask, "euler"),
        "strict_post_first_euler_rk4_count": _count(post_euler_mask, "rk4"),
        "strict_post_first_euler_euler_fraction": float(
            _count(post_euler_mask, "euler") / max(int(np.count_nonzero(post_euler_mask)), 1)
        ),
    }
    for name, getter in (
        ("energy", _row_energy_observable),
        ("primary_density", _row_primary_observable),
        ("site0", _row_site0_observable),
    ):
        tail_values = _values(tail_mask, getter)
        pre_values = _values(pre_euler_mask, getter)
        post_values = _values(post_euler_mask, getter)
        out.update(_tail_series_metrics(f"strict_tail_{name}", tail_values))
        out.update(_tail_series_metrics(f"strict_pre_first_euler_{name}", pre_values))
        out.update(_tail_series_metrics(f"strict_post_first_euler_{name}", post_values))
        out[f"strict_post_first_euler_{name}_range_growth_ratio"] = _ratio_or_nan(
            out[f"strict_post_first_euler_{name}_range"],
            out[f"strict_pre_first_euler_{name}_range"],
        )
        out[f"strict_post_first_euler_{name}_variation_growth_ratio"] = _ratio_or_nan(
            out[f"strict_post_first_euler_{name}_total_variation"],
            out[f"strict_pre_first_euler_{name}_total_variation"],
        )
    return out


def _string_histogram(values: Sequence[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        if value in {None, ""}:
            continue
        key = str(value)
        out[key] = int(out.get(key, 0)) + 1
    return dict(sorted(out.items()))


def _search_space(profile: str = "generic_l2_exact_v1") -> dict[str, list[Any]]:
    if _is_strict_guardrail_only_profile(profile):
        return _strict_auto_euler_guardrail_search_space()
    if _is_strict_qpu_faithful_profile(profile):
        space = {
            "oracle_selection_policy": ["measured_gain_commit_veto"],
            "candidate_step_scales": [
                _STEP_SCALE_PRESETS["narrow"],
                _STEP_SCALE_PRESETS["default"],
                _STEP_SCALE_PRESETS["wide"],
            ],
            "high_miss_no_admit_policy": ["bounded_stay_advance"],
            "miss_threshold": [0.03, 0.05, 0.08],
            "miss_abs_threshold": [0.0, 1.0e-4, 1.0e-3],
            "miss_persistence_spec": ["1:1", "3:2", "3:3"],
            "gain_ratio_threshold": [1.0e-4, 1.0e-3, 0.01, 0.02],
            "append_margin_abs": [1.0e-6, 1.0e-5, 5.0e-5],
            "shortlist_size": [4, 6, 8],
            "shortlist_fraction": [0.15, 0.25, 0.35],
            "active_window_size": [3, 4, 6],
            "max_probe_positions": [4, 6, 8],
            "regularization_lambda": [1.0e-10, 1.0e-8, 1.0e-6],
            "candidate_regularization_lambda": [1.0e-10, 1.0e-8, 1.0e-6],
            "pinv_rcond": [1.0e-12, 1.0e-10, 1.0e-8],
            "compile_penalty_weight": [0.0, 0.02, 0.05],
            "measurement_penalty_weight": [0.0, 0.005, 0.02],
            "directional_penalty_weight": [0.0, 0.002, 0.01],
            "confirm_score_mode": ["compressed_whitened_v1"],
            "confirm_compress_fraction": [0.25, 0.5, 0.75],
            "confirm_compress_min_modes": [1, 2],
            "confirm_compress_max_modes": [4, 8, 12],
            "integrator_policy": ["auto_euler_rk4"],
            "integrator_euler_observable_window": [12, 16, 24],
            **_strict_auto_euler_guardrail_search_space(),
            "prune_mode": ["off"],
        }
        if str(profile) == _STRICT_QPU_FAITHFUL_APPEND_PRUNE_PROFILE:
            space.update(
                {
                    "append_no_harm_guard_enabled": [True],
                    "append_no_harm_condition_ratio_cap": [1.25, 1.5, 2.0],
                    "append_no_harm_condition_abs_floor": [1.0, 2.0],
                    "append_no_harm_displacement_ratio_cap": [1.25, 1.5, 2.0],
                    "append_no_harm_kink_min_step_gain_delta": [1.0e-4, 1.0e-3],
                    "append_no_harm_kink_max_condition_ratio": [1.25, 1.5],
                    "append_no_harm_kink_max_displacement_ratio": [1.25, 1.5],
                    "append_no_harm_rho_only_min_step_gain_delta": [1.0e-4, 1.0e-3],
                    "append_no_harm_rho_only_condition_ratio_cap": [1.5, 2.0],
                    "append_no_harm_rho_only_step_residual_ratio_cap": [1.5, 2.0],
                    "append_no_harm_rho_only_displacement_ratio_cap": [1.5, 2.0],
                    "prune_mode": ["off", "schur_projected_shadow_v1"],
                    "prune_miss_threshold": [0.02, 0.05],
                    "prune_loss_threshold": [0.01, 0.02, 0.05],
                    "prune_theta_block_tol": [0.05, 0.1, 0.25],
                    "prune_state_jump_l2_tol": [0.05, 0.1],
                    "prune_safe_miss_increase_tol": [0.01, 0.05],
                    "prune_schur_ladder_local_radius": [1, 2],
                    "prune_differential_miss_tol": [0.005, 0.01],
                    "prune_projection_mode": ["state_tangent_ls_v1"],
                    "prune_projection_rounds": [1, 2],
                    "prune_projection_max_active_runtime": [32, 64],
                    "prune_projection_trust_radius": [0.02, 0.05, 0.1],
                    "prune_projection_regularization": [1.0e-10, 1.0e-8],
                    "prune_ray_distance_tol": [0.02, 0.05],
                    "prune_shadow_enabled": [True],
                    "prune_shadow_horizon_steps": [1, 2],
                    "prune_shadow_score_increase_tol": [0.0, 1.0e-3],
                    "prune_persistence_window": [2, 3],
                    "prune_persistence_required": [1, 2],
                    "prune_appended_origin_bias_enabled": [True],
                    "prune_appended_origin_target_policy": ["prefer_append", "bias_only"],
                    "prune_appended_origin_grace_steps": [1, 2],
                }
            )
        if _is_strict_append_prune_aggressive_profile(profile):
            space.update(
                {
                    "append_no_harm_guard_enabled": [True],
                    "append_no_harm_condition_ratio_cap": [1.25, 1.5, 2.0],
                    "append_no_harm_condition_abs_floor": [1.0, 2.0],
                    "append_no_harm_displacement_ratio_cap": [1.25, 1.5, 2.0],
                    "append_no_harm_kink_min_step_gain_delta": [1.0e-4, 1.0e-3],
                    "append_no_harm_kink_max_condition_ratio": [1.25, 1.5],
                    "append_no_harm_kink_max_displacement_ratio": [1.25, 1.5],
                    "append_no_harm_rho_only_min_step_gain_delta": [1.0e-4, 1.0e-3],
                    "append_no_harm_rho_only_condition_ratio_cap": [1.5, 2.0],
                    "append_no_harm_rho_only_step_residual_ratio_cap": [1.5, 2.0],
                    "append_no_harm_rho_only_displacement_ratio_cap": [1.5, 2.0],
                    "prune_mode": ["schur_projected_shadow_v1"],
                    "prune_miss_threshold": [0.03, 0.05, 0.08, 0.12],
                    "prune_loss_threshold": [0.02, 0.05, 0.08, 0.12],
                    "prune_theta_block_tol": [0.10, 0.25, 0.35, 0.50],
                    "prune_state_jump_l2_tol": [0.10, 0.15, 0.20],
                    "prune_safe_miss_increase_tol": [0.02, 0.05, 0.08, 0.12],
                    "prune_schur_ladder_local_radius": [1, 2],
                    "prune_differential_miss_tol": [0.01, 0.02, 0.05],
                    "prune_projection_mode": ["state_tangent_ls_v1"],
                    "prune_projection_rounds": [1, 2],
                    "prune_projection_max_active_runtime": [32, 64],
                    "prune_projection_trust_radius": [0.05, 0.10, 0.12, 0.15],
                    "prune_projection_regularization": [1.0e-10, 1.0e-8, 1.0e-6],
                    "prune_ray_distance_tol": [0.05, 0.08, 0.12],
                    "prune_shadow_enabled": [True],
                    "prune_shadow_horizon_steps": [1, 2],
                    "prune_shadow_score_increase_tol": [0.0, 1.0e-3, 2.0e-3, 5.0e-3],
                    "prune_persistence_window": [1, 2],
                    "prune_persistence_required": [1],
                    "prune_appended_origin_bias_enabled": [True],
                    "prune_appended_origin_target_policy": ["bias_only", "prefer_append"],
                    "prune_appended_origin_grace_steps": [0, 1],
                    "prune_initial_scaffold_grace_steps": [0],
                    "prune_appended_origin_bias_scale": [0.25, 0.5, 0.75],
                    "prune_appended_origin_bias_max_factor": [0.75, 1.0],
                }
            )
        return space
    if str(profile) == _APPEND_PRUNE_NARROW_PROFILE:
        return {
            "horizon_mode": ["lead2"],
            "step_scale_mode": ["default", "wide"],
            "blend_weight_mode": ["default"],
            "gain_scale_mode": ["off", "mild"],
            "baseline_step_refine_rounds": [0],
            "include_tangent_secant_proposal": [False],
            "baseline_proposal_mode": ["norm_locked_blend_v1"],
            "primary_density_target_mode": ["auto", "pair_difference"],
            "high_miss_no_admit_policy": ["bounded_stay_advance"],
            "miss_threshold": [0.05],
            "miss_abs_threshold": [0.0],
            "miss_persistence_spec": ["3:3"],
            "gain_ratio_threshold": [0.01, 0.02],
            "append_margin_abs": [1.0e-5, 5.0e-5],
            "append_no_harm_guard_enabled": [False, True],
            "append_no_harm_condition_ratio_cap": [1.5, 2.0],
            "append_no_harm_condition_abs_floor": [1.0, 2.0],
            "append_no_harm_displacement_ratio_cap": [1.5, 2.0],
            "append_no_harm_kink_min_step_gain_delta": [1.0e-3, 5.0e-3],
            "append_no_harm_kink_max_condition_ratio": [1.5, 2.0],
            "append_no_harm_kink_max_displacement_ratio": [1.5, 2.0],
            "append_no_harm_rho_only_min_step_gain_delta": [1.0e-3, 5.0e-3],
            "append_no_harm_rho_only_condition_ratio_cap": [1.5, 2.0],
            "append_no_harm_rho_only_step_residual_ratio_cap": [1.5, 2.0],
            "append_no_harm_rho_only_displacement_ratio_cap": [1.5, 2.0],
            "shortlist_size": [4, 6],
            "shortlist_fraction": [0.15, 0.25],
            "active_window_size": [3],
            "max_probe_positions": [4],
            "regularization_lambda": [1.0e-8],
            "candidate_regularization_lambda": [1.0e-8],
            "pinv_rcond": [1.0e-10],
            "compile_penalty_weight": [0.02, 0.05],
            "measurement_penalty_weight": [0.005, 0.02],
            "directional_penalty_weight": [0.002, 0.01],
            "confirm_score_mode": ["exact_gain_ratio"],
            "primary_density_weight": [1.0],
            "tracking_staggered_weight": [1.0],
            "site_weight": [1.0, 2.0],
            "energy_weight": [0.5, 1.0],
            "density_slope_weight": [0.5, 1.0],
            "fidelity_weight": [0.0],
            "doublon_weight": [0.0],
            "integrator_policy": ["auto_euler_rk4"],
            "integrator_columnarity_threshold": [0.70, 0.80],
            "integrator_curvature_threshold": [0.10, 0.20, 0.35],
            "integrator_euler_fs_error_threshold": [1.0e-3, 2.5e-3, 5.0e-3],
            "integrator_condition_max": [1.0e12],
            "integrator_euler_min_time_fraction": [0.35, 0.50],
            "integrator_euler_observable_window": [12, 16, 24],
            "integrator_euler_site_span_max": STRICT_AUTO_EULER_SITE_SPAN_CHOICES,
            "integrator_euler_primary_density_span_max": STRICT_AUTO_EULER_PRIMARY_DENSITY_SPAN_CHOICES,
            "integrator_euler_energy_span_max": STRICT_AUTO_EULER_ENERGY_SPAN_CHOICES,
            "prune_mode": ["off", "exact_local_v1"],
            "prune_miss_threshold": [0.02],
            "prune_loss_threshold": [0.01, 0.05],
            "prune_theta_block_tol": [0.05, 0.1],
            "prune_state_jump_l2_tol": [0.05, 0.1],
            "prune_safe_miss_increase_tol": [0.01, 0.05],
            "prune_appended_origin_bias_enabled": [True],
            "prune_appended_origin_target_policy": ["append_only"],
            "prune_appended_origin_grace_steps": [1, 2],
            "prune_appended_origin_bias_scale": [0.1, 0.5],
            "prune_appended_origin_bias_max_factor": [0.5, 1.0],
            "prune_protection_steps": [2],
            "prune_stagnation_window": [3],
            "prune_stagnation_alpha": [0.5],
            "prune_stale_score_threshold": [0.75],
            "prune_max_candidates": [1, 2],
            "prune_cooldown_steps": [1, 2],
            "motion_calm_direction_cosine_threshold": [0.98],
            "motion_calm_rate_change_ratio_threshold": [0.15],
            "motion_direction_reversal_cosine_threshold": [-0.05],
            "motion_curvature_flip_cosine_threshold": [-0.1],
            "motion_acceleration_l2_threshold": [0.05],
            "motion_kink_rate_change_ratio_threshold": [0.5],
            "motion_calm_shortlist_scale": [0.5],
            "motion_kink_shortlist_bonus": [2],
            "motion_calm_oracle_budget_scale": [0.5],
            "motion_kink_oracle_budget_scale": [2.0],
            "position_jump_tie_margin_abs": [1.0e-6],
            "reconstruction_tol": [1.0e-10],
            "grouping_mode": ["qwc_basis_cover_reuse"],
        }
    if str(profile) == _APPEND_PRUNE_RECOVERABILITY_PROFILE:
        space = dict(_search_space(profile="append_prune_noharm_l2_v1"))
        space.update(
            {
                "prune_mode": ["off", "schur_projected_shadow_v1"],
                "prune_loss_threshold": [0.01, 0.02, 0.05],
                "prune_theta_block_tol": [0.05, 0.1, 0.25],
                "prune_state_jump_l2_tol": [0.05, 0.1],
                "prune_safe_miss_increase_tol": [0.01, 0.05],
                "prune_schur_ladder_local_radius": [1, 2],
                "prune_differential_miss_tol": [0.005, 0.01, 0.02],
                "prune_projection_mode": ["state_tangent_ls_v1"],
                "prune_projection_rounds": [1, 2],
                "prune_projection_max_active_runtime": [32, 64],
                "prune_projection_trust_radius": [0.02, 0.05, 0.1],
                "prune_projection_regularization": [1.0e-10, 1.0e-8, 1.0e-6],
                "prune_ray_distance_tol": [0.02, 0.05, 0.1],
                "prune_shadow_enabled": [True],
                "prune_shadow_horizon_steps": [1, 2],
                "prune_shadow_score_increase_tol": [0.0, 1.0e-3],
                "prune_persistence_window": [2, 3],
                "prune_persistence_required": [2],
                "prune_appended_origin_target_policy": ["prefer_append", "bias_only"],
            }
        )
        return space

    if str(profile) in _APPEND_PRUNE_TUNE_PROFILES:
        return {
            "horizon_mode": ["lead2", "lead3"],
            "step_scale_mode": ["default", "wide", "high_amp_dense"],
            "blend_weight_mode": ["default", "tight"],
            "gain_scale_mode": ["off", "mild"],
            "baseline_step_refine_rounds": [0, 1],
            "include_tangent_secant_proposal": [False],
            "baseline_proposal_mode": ["norm_locked_blend_v1"],
            "primary_density_target_mode": ["auto", "pair_difference", "staggered"],
            "high_miss_no_admit_policy": ["bounded_stay_advance"],
            "miss_threshold": [0.05],
            "miss_abs_threshold": [0.0],
            "miss_persistence_spec": ["3:3"],
            "gain_ratio_threshold": [1.0e-4, 1.0e-3, 0.01, 0.02],
            "append_margin_abs": [1.0e-6, 1.0e-5, 5.0e-5],
            "append_no_harm_guard_enabled": [True, False],
            "append_no_harm_condition_ratio_cap": [1.0, 1.25, 1.5, 2.0],
            "append_no_harm_condition_abs_floor": [0.5, 1.0, 2.0],
            "append_no_harm_displacement_ratio_cap": [1.0, 1.25, 1.5, 2.0],
            "append_no_harm_kink_min_step_gain_delta": [1.0e-4, 1.0e-3, 5.0e-3],
            "append_no_harm_kink_max_condition_ratio": [1.0, 1.25, 1.5, 2.0],
            "append_no_harm_kink_max_displacement_ratio": [1.0, 1.25, 1.5, 2.0],
            "append_no_harm_rho_only_min_step_gain_delta": [1.0e-4, 1.0e-3, 5.0e-3],
            "append_no_harm_rho_only_condition_ratio_cap": [1.0, 1.5, 2.0],
            "append_no_harm_rho_only_step_residual_ratio_cap": [1.0, 1.5, 2.0],
            "append_no_harm_rho_only_displacement_ratio_cap": [1.0, 1.5, 2.0],
            "shortlist_size": [4, 6],
            "shortlist_fraction": [0.15, 0.25],
            "active_window_size": [3, 4],
            "max_probe_positions": [4, 6],
            "regularization_lambda": [1.0e-10, 1.0e-8],
            "candidate_regularization_lambda": [1.0e-10, 1.0e-8],
            "pinv_rcond": [1.0e-12, 1.0e-10],
            "compile_penalty_weight": [0.0, 0.02, 0.05],
            "measurement_penalty_weight": [0.0, 0.005, 0.02],
            "directional_penalty_weight": [0.0, 0.002, 0.01],
            "confirm_score_mode": ["exact_gain_ratio"],
            "primary_density_weight": [1.0, 2.0],
            "tracking_staggered_weight": [0.5, 1.0],
            "site_weight": [1.0, 2.0, 4.0],
            "energy_weight": [0.25, 0.5, 1.0],
            "density_slope_weight": [0.5, 1.0, 2.0],
            "fidelity_weight": [0.0, 0.25],
            "doublon_weight": [0.0, 0.5],
            "integrator_policy": ["auto_euler_rk4"],
            "integrator_columnarity_threshold": [0.70, 0.80, 0.90],
            "integrator_curvature_threshold": [0.10, 0.20, 0.35],
            "integrator_euler_fs_error_threshold": [1.0e-3, 2.5e-3, 5.0e-3],
            "integrator_condition_max": [1.0e10, 1.0e12, 1.0e14],
            "integrator_euler_min_time_fraction": [0.35, 0.50, 0.65],
            "integrator_euler_observable_window": [12, 16, 24],
            "integrator_euler_site_span_max": STRICT_AUTO_EULER_SITE_SPAN_CHOICES,
            "integrator_euler_primary_density_span_max": STRICT_AUTO_EULER_PRIMARY_DENSITY_SPAN_CHOICES,
            "integrator_euler_energy_span_max": STRICT_AUTO_EULER_ENERGY_SPAN_CHOICES,
            "prune_mode": ["off", "exact_local_v1"],
            "prune_miss_threshold": [0.02, 0.05],
            "prune_loss_threshold": [0.01, 0.05, 0.1],
            "prune_theta_block_tol": [0.05, 0.1, 0.25],
            "prune_state_jump_l2_tol": [0.05, 0.1, 0.25],
            "prune_safe_miss_increase_tol": [0.01, 0.05, 0.1],
            "prune_appended_origin_bias_enabled": [True],
            "prune_appended_origin_target_policy": ["append_only"],
            "prune_appended_origin_grace_steps": [1, 2, 3],
            "prune_appended_origin_bias_scale": [0.1, 0.5, 1.0],
            "prune_appended_origin_bias_max_factor": [0.5, 1.0, 2.0],
            "prune_protection_steps": [0, 1, 2],
            "prune_stagnation_window": [1, 3],
            "prune_stagnation_alpha": [0.25, 0.5, 0.75],
            "prune_stale_score_threshold": [0.5, 0.75, 1.0],
            "prune_max_candidates": [1, 2],
            "prune_cooldown_steps": [0, 1, 2],
            "motion_calm_direction_cosine_threshold": [0.95, 0.98],
            "motion_calm_rate_change_ratio_threshold": [0.15, 0.3],
            "motion_direction_reversal_cosine_threshold": [-0.1, -0.05],
            "motion_curvature_flip_cosine_threshold": [-0.1, -0.05],
            "motion_acceleration_l2_threshold": [0.05, 0.1],
            "motion_kink_rate_change_ratio_threshold": [0.5, 1.0],
            "motion_calm_shortlist_scale": [0.5, 1.0],
            "motion_kink_shortlist_bonus": [1, 2],
            "motion_calm_oracle_budget_scale": [0.5, 1.0],
            "motion_kink_oracle_budget_scale": [1.5, 2.0],
            "position_jump_tie_margin_abs": [1.0e-8, 1.0e-6],
            "reconstruction_tol": [1.0e-12, 1.0e-10],
            "grouping_mode": ["qwc_basis_cover_reuse"],
        }
    if str(profile) in _FULL_SITE3_ALGORITHM_PROFILES:
        return _site3_full_surface_search_space()
    site3_prefsite_guardrail_mode = _site3_prefsite_autobaseline_guardrail_mode(profile)
    if site3_prefsite_guardrail_mode is not None:
        return {
            "horizon_mode": ["high_amp_lead4"],
            "step_scale_mode": ["high_amp_dense"],
            "blend_weight_mode": ["high_amp"],
            "gain_scale_mode": ["high_amp"],
            "baseline_step_refine_rounds": [2],
            "include_tangent_secant_proposal": [True],
            "baseline_proposal_mode": ["norm_locked_blend_v1"],
            "primary_density_target_mode": ["pair_difference"],
            "miss_threshold": [0.05],
            "gain_ratio_threshold": [0.02],
            "append_margin_abs": [1.0e-6, 1.0e-5, 5.0e-5],
            "trust_radius": [0.85, 1.0, 1.15],
            "signed_energy_lead_limit": [3.0, 3.5, 4.0],
            "shortlist_size": [4, 6, 8],
            "shortlist_fraction": [0.15, 0.25, 0.35],
            "active_window_size": [3, 4, 6],
            "max_probe_positions": [4, 6, 8],
            "regularization_lambda": [1.0e-10, 1.0e-8, 1.0e-6],
            "candidate_regularization_lambda": [1.0e-10, 1.0e-8, 1.0e-6],
            "pinv_rcond": [1.0e-12, 1.0e-10, 1.0e-8],
            "compile_penalty_weight": [0.0, 0.05, 0.1],
            "measurement_penalty_weight": [0.0, 0.02, 0.05],
            "directional_penalty_weight": [0.0, 0.01, 0.03],
            "confirm_score_mode": ["compressed_whitened_v1", "exact_gain_ratio"],
            "confirm_compress_fraction": [0.25, 0.5, 0.75],
            "confirm_compress_min_modes": [1, 2],
            "confirm_compress_max_modes": [4, 8, 12],
            "primary_density_weight": [2.0],
            "site_weight": [0.0],
            "energy_weight": [0.0],
            "density_slope_weight": [2.0],
            "density_sign_lag_weight": [0.25],
            "drive_harmonic_weight": [0.0],
            "density_first_target_gain_floor": [0.02],
            "below_floor_probe_target_gain_floor": [0.03],
            "fidelity_loss_tol": [0.005, 0.01, 0.015],
            "abs_energy_error_increase_tol": [0.01, 0.02, 0.03],
            "total_occupation_error_increase_tol": [0.005, 0.01, 0.015],
            "guardrail_mode": [site3_prefsite_guardrail_mode],
            "below_floor_energy_safe_d_shape_escape": [False],
            "below_floor_energy_safe_turn_escape": [False],
            "d_shape_turn_window_abs_activation": [0.04],
            "d_shape_outside_turn_below_floor_probe_stall_threshold": [7],
            "d_shape_pre_turn_shadow_bridge": [True],
        }
    if str(profile) == "high_amp_guarded_d_shape_expanded_v1":
        return {
            "horizon_mode": ["high_amp_lead4"],
            "step_scale_mode": ["high_amp_dense"],
            "blend_weight_mode": ["high_amp"],
            "gain_scale_mode": ["high_amp"],
            "baseline_step_refine_rounds": [2],
            "include_tangent_secant_proposal": [False, True],
            "baseline_proposal_mode": ["norm_locked_blend_v1", "anticipatory_drive_basis_v1"],
            "primary_density_target_mode": ["auto", "pair_difference", "staggered"],
            "miss_threshold": [0.05],
            "gain_ratio_threshold": [0.02],
            "append_margin_abs": [1.0e-6, 1.0e-5, 5.0e-5],
            "trust_radius": [0.85, 1.0, 1.15],
            "signed_energy_lead_limit": [3.0, 3.5, 4.0],
            "shortlist_size": [4, 6, 8],
            "shortlist_fraction": [0.15, 0.25, 0.35],
            "active_window_size": [3, 4, 6],
            "max_probe_positions": [4, 6, 8],
            "regularization_lambda": [1.0e-10, 1.0e-8, 1.0e-6],
            "candidate_regularization_lambda": [1.0e-10, 1.0e-8, 1.0e-6],
            "pinv_rcond": [1.0e-12, 1.0e-10, 1.0e-8],
            "compile_penalty_weight": [0.0, 0.05, 0.1],
            "measurement_penalty_weight": [0.0, 0.02, 0.05],
            "directional_penalty_weight": [0.0, 0.01, 0.03],
            "confirm_score_mode": ["compressed_whitened_v1", "exact_gain_ratio"],
            "confirm_compress_fraction": [0.25, 0.5, 0.75],
            "confirm_compress_min_modes": [1, 2],
            "confirm_compress_max_modes": [4, 8, 12],
            "primary_density_weight": [1.5, 2.0, 2.5, 3.0],
            "site_weight": [0.0, 0.125, 0.25],
            "energy_weight": [0.0, 0.125, 0.25],
            "density_slope_weight": [1.0, 2.0, 3.0],
            "density_sign_lag_weight": [0.1, 0.25, 0.5],
            "drive_harmonic_weight": [0.0, 0.1, 0.25],
            "density_first_target_gain_floor": [0.015, 0.02, 0.025],
            "below_floor_probe_target_gain_floor": [0.025, 0.03, 0.035],
            "fidelity_loss_tol": [0.005, 0.01, 0.015],
            "abs_energy_error_increase_tol": [0.01, 0.02, 0.03],
            "total_occupation_error_increase_tol": [0.005, 0.01, 0.015],
            "guardrail_mode": ["d_shape_barrier_v1"],
            "below_floor_energy_safe_d_shape_escape": [False],
            "below_floor_energy_safe_turn_escape": [False],
            "d_shape_turn_window_abs_activation": [0.02, 0.03, 0.04, 0.05, 0.06],
            "d_shape_outside_turn_below_floor_probe_stall_threshold": [0, 1, 2, 3],
            "d_shape_pre_turn_shadow_bridge": [False, True],
        }
    if str(profile) == "high_amp_guarded_d_shape_turnwindow_v1":
        return {
            "horizon_mode": ["high_amp_lead4"],
            "step_scale_mode": ["high_amp_dense"],
            "blend_weight_mode": ["high_amp"],
            "gain_scale_mode": ["high_amp"],
            "baseline_step_refine_rounds": [2],
            "miss_threshold": [0.05],
            "gain_ratio_threshold": [0.02],
            "trust_radius": [0.85, 1.0, 1.15],
            "signed_energy_lead_limit": [3.0, 3.5, 4.0],
            "primary_density_weight": [1.5, 2.0, 2.5, 3.0],
            "site_weight": [0.0, 0.125, 0.25],
            "energy_weight": [0.0, 0.125, 0.25],
            "density_slope_weight": [1.0, 2.0, 3.0],
            "density_sign_lag_weight": [0.1, 0.25, 0.5],
            "drive_harmonic_weight": [0.0, 0.1, 0.25],
            "density_first_target_gain_floor": [0.015, 0.02, 0.025],
            "below_floor_probe_target_gain_floor": [0.025, 0.03, 0.035],
            "fidelity_loss_tol": [0.005, 0.01, 0.015],
            "abs_energy_error_increase_tol": [0.01, 0.02, 0.03],
            "total_occupation_error_increase_tol": [0.005, 0.01, 0.015],
            "guardrail_mode": ["d_shape_barrier_v1"],
            "below_floor_energy_safe_d_shape_escape": [False],
            "below_floor_energy_safe_turn_escape": [False],
            "d_shape_turn_window_abs_activation": [0.02, 0.03, 0.04, 0.05, 0.06],
            "d_shape_outside_turn_below_floor_probe_stall_threshold": [0, 1, 2, 3],
            "d_shape_pre_turn_shadow_bridge": [False, True],
        }
    if str(profile) == "high_amp_guarded_d_shape_live_v1":
        return {
            "horizon_mode": ["high_amp_lead4"],
            "step_scale_mode": ["high_amp_dense"],
            "blend_weight_mode": ["high_amp"],
            "gain_scale_mode": ["high_amp"],
            "baseline_step_refine_rounds": [2],
            "miss_threshold": [0.05],
            "gain_ratio_threshold": [0.02],
            "trust_radius": [0.85, 1.0, 1.15],
            "signed_energy_lead_limit": [3.0, 3.5, 4.0],
            "primary_density_weight": [1.5, 2.0, 2.5, 3.0],
            "site_weight": [0.0, 0.125, 0.25, 0.5],
            "energy_weight": [0.0, 0.125, 0.25, 0.5],
            "density_slope_weight": [1.0, 2.0, 3.0],
            "density_sign_lag_weight": [0.1, 0.25, 0.5],
            "drive_harmonic_weight": [0.0, 0.1, 0.25],
            "density_first_target_gain_floor": [0.015, 0.02, 0.025],
            "below_floor_probe_target_gain_floor": [0.025, 0.03, 0.035],
            "fidelity_loss_tol": [0.005, 0.01, 0.015],
            "abs_energy_error_increase_tol": [0.01, 0.02, 0.03],
            "total_occupation_error_increase_tol": [0.005, 0.01, 0.015],
            "guardrail_mode": ["d_shape_barrier_v1"],
            "below_floor_energy_safe_d_shape_escape": [False],
            "below_floor_energy_safe_turn_escape": [False],
            "d_shape_turn_window_abs_activation": [0.04],
            "d_shape_outside_turn_below_floor_probe_stall_threshold": [0],
            "d_shape_pre_turn_shadow_bridge": [False],
        }
    if str(profile) == "high_amp_guarded_d_shape_v1":
        return {
            "horizon_mode": ["high_amp_lead4"],
            "step_scale_mode": ["high_amp_dense"],
            "blend_weight_mode": ["high_amp"],
            "gain_scale_mode": ["high_amp"],
            "baseline_step_refine_rounds": [2],
            "miss_threshold": [0.05],
            "gain_ratio_threshold": [0.02],
            "trust_radius": [0.85, 1.0, 1.15],
            "signed_energy_lead_limit": [3.0, 3.5, 4.0],
            "primary_density_weight": [1.5, 2.0, 2.5, 3.0],
            "site_weight": [0.0, 0.25, 0.5],
            "energy_weight": [0.0, 0.25, 0.5],
            "density_slope_weight": [1.0, 2.0, 3.0],
            "density_curvature_weight": [25.0, 50.0, 75.0, 100.0],
            "density_excursion_under_weight": [150.0, 225.0, 300.0, 450.0],
            "density_sign_lag_weight": [0.1, 0.25, 0.5],
            "density_postcross_wrong_sign_weight": [0.1, 0.25, 0.5],
            "density_first_target_gain_floor": [0.015, 0.02, 0.025],
            "below_floor_probe_target_gain_floor": [0.025, 0.03, 0.035],
            "sign_lag_window_target_gain_floor": [0.0025, 0.005, 0.01],
            "postcross_wrong_sign_activation": [0.0, 0.05, 0.1],
            "postcross_wrong_sign_target_gain_floor": [None, 0.0025, 0.005],
            "below_floor_energy_safe_d_shape_escape": [False],
            "guardrail_mode": ["d_shape_barrier_v1"],
            "fidelity_loss_tol": [0.005, 0.01, 0.015],
            "abs_energy_error_increase_tol": [0.01, 0.02, 0.03],
            "total_occupation_error_increase_tol": [0.005, 0.01, 0.015],
        }
    if str(profile) == "high_amp_d_shape_v1":
        return {
            "horizon_mode": ["high_amp_lead4"],
            "step_scale_mode": ["high_amp_dense"],
            "blend_weight_mode": ["high_amp"],
            "gain_scale_mode": ["high_amp"],
            "baseline_step_refine_rounds": [2],
            "miss_threshold": [0.05],
            "gain_ratio_threshold": [0.02],
            "trust_radius": [0.75, 1.0, 1.25],
            "signed_energy_lead_limit": [3.0, 3.5, 4.0],
            "primary_density_weight": [2.0],
            "site_weight": [0.0, 1.0, 3.0],
            "energy_weight": [0.0, 0.5, 1.0],
            "density_slope_weight": [1.0, 2.0, 4.0],
            "density_curvature_weight": [0.0, 50.0, 100.0, 200.0],
            "density_excursion_under_weight": [0.0, 225.0, 450.0, 900.0],
            "density_sign_lag_weight": [0.25, 0.5, 1.0],
            "density_postcross_wrong_sign_weight": [0.25, 0.5, 1.0],
            "sign_lag_window_target_gain_floor": [0.0025, 0.005, 0.01],
            "below_floor_energy_safe_d_shape_escape": [False, True],
            "guardrail_mode": ["off", "d_shape_barrier_v1"],
            "fidelity_loss_tol": [0.005, 0.01, 0.02],
            "abs_energy_error_increase_tol": [0.01, 0.02, 0.05],
            "total_occupation_error_increase_tol": [0.005, 0.01, 0.02],
        }
    return {
        "horizon_mode": ["lead2", "flat2", "lead3", "flat3", "lead4"],
        "step_scale_mode": ["narrow", "default", "wide"],
        "blend_weight_mode": ["off", "default", "positive_only", "tight"],
        "gain_scale_mode": ["off", "mild", "wide"],
        "baseline_step_refine_rounds": [0, 1, 2],
        "primary_density_target_mode": ["auto", "pair_difference", "staggered"],
        "high_miss_no_admit_policy": ["bounded_stay_advance", "repair_stop"],
        "miss_threshold": [0.02, 0.03, 0.05, 0.08],
        "miss_abs_threshold": [0.0, 1.0e-4, 1.0e-3, 1.0e-2],
        "miss_persistence_spec": ["1:1", "3:2", "3:3", "5:3"],
        "gain_ratio_threshold": [1.0e-6, 1.0e-4, 1.0e-3, 0.01, 0.02],
        "append_margin_abs": [1.0e-8, 1.0e-6, 1.0e-5],
        "compile_penalty_weight": [0.0, 0.02, 0.05],
        "measurement_penalty_weight": [0.0, 0.005, 0.02],
        "directional_penalty_weight": [0.0, 0.002, 0.01],
        "confirm_score_mode": ["compressed_whitened_v1", "exact_gain_ratio"],
        "trust_radius": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        "signed_energy_lead_limit": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        "primary_density_weight": [1.0, 2.0, 4.0, 8.0],
        "site_weight": [0.5, 1.0, 2.0, 4.0],
        "energy_weight": [0.0, 0.25, 0.5, 1.0],
        "density_slope_weight": [0.5, 1.0, 2.0, 4.0],
    }


def _baseline_trial_params(profile: str = "generic_l2_exact_v1") -> TrialParams:
    if str(profile) == "generic_l2_exact_v1":
        return TrialParams(primary_density_target_mode="auto")
    if _is_strict_guardrail_only_profile(profile):
        baseline = _baseline_trial_params(profile=_STRICT_QPU_FAITHFUL_APPEND_PRUNE_PROFILE)
        return TrialParams(**{**asdict(baseline), **_strict_auto_euler_guardrail_params()})
    if _is_strict_append_prune_aggressive_profile(profile):
        baseline = _baseline_trial_params(profile=_STRICT_QPU_FAITHFUL_RECOVERABILITY_PROFILE)
        return TrialParams(
            **{
                **asdict(baseline),
                "append_no_harm_guard_enabled": True,
                "append_no_harm_condition_ratio_cap": 1.5,
                "append_no_harm_condition_abs_floor": 1.0,
                "append_no_harm_displacement_ratio_cap": 1.5,
                "append_no_harm_kink_min_step_gain_delta": 1.0e-3,
                "append_no_harm_kink_max_condition_ratio": 1.5,
                "append_no_harm_kink_max_displacement_ratio": 1.5,
                "append_no_harm_rho_only_min_step_gain_delta": 1.0e-3,
                "append_no_harm_rho_only_condition_ratio_cap": 1.5,
                "append_no_harm_rho_only_step_residual_ratio_cap": 1.5,
                "append_no_harm_rho_only_displacement_ratio_cap": 1.5,
                "miss_threshold": 0.08,
                "miss_persistence_spec": "3:3",
                "gain_ratio_threshold": 0.01,
                "append_margin_abs": 1.0e-6,
                "active_window_size": 4,
                "shortlist_size": 4,
                "shortlist_fraction": 0.25,
                "max_probe_positions": 4,
                "regularization_lambda": 1.0e-6,
                "candidate_regularization_lambda": 1.0e-10,
                "confirm_compress_fraction": 0.25,
                "confirm_compress_min_modes": 2,
                "confirm_compress_max_modes": 12,
                "prune_mode": "schur_projected_shadow_v1",
                "prune_miss_threshold": 0.05,
                "prune_loss_threshold": 0.08,
                "prune_theta_block_tol": 0.35,
                "prune_state_jump_l2_tol": 0.15,
                "prune_safe_miss_increase_tol": 0.08,
                "prune_differential_miss_tol": 0.02,
                "prune_schur_ladder_local_radius": 1,
                "prune_projection_mode": "state_tangent_ls_v1",
                "prune_projection_rounds": 1,
                "prune_projection_max_active_runtime": 64,
                "prune_projection_trust_radius": 0.12,
                "prune_projection_regularization": 1.0e-8,
                "prune_ray_distance_tol": 0.08,
                "prune_shadow_enabled": True,
                "prune_shadow_horizon_steps": 1,
                "prune_shadow_score_increase_tol": 0.002,
                "prune_persistence_window": 1,
                "prune_persistence_required": 1,
                "prune_appended_origin_bias_enabled": True,
                "prune_appended_origin_target_policy": "bias_only",
                "prune_appended_origin_grace_steps": 0,
                "prune_initial_scaffold_grace_steps": 0,
                "prune_appended_origin_bias_scale": 0.5,
                "prune_appended_origin_bias_max_factor": 1.0,
            }
        )
    if str(profile) == _STRICT_QPU_FAITHFUL_APPEND_PRUNE_PROFILE:
        baseline = _baseline_trial_params(profile=_STRICT_QPU_FAITHFUL_RECOVERABILITY_PROFILE)
        return TrialParams(
            **{
                **asdict(baseline),
                "append_no_harm_guard_enabled": True,
                "append_no_harm_condition_ratio_cap": 1.5,
                "append_no_harm_condition_abs_floor": 1.0,
                "append_no_harm_displacement_ratio_cap": 1.5,
                "append_no_harm_kink_min_step_gain_delta": 1.0e-3,
                "append_no_harm_kink_max_condition_ratio": 1.5,
                "append_no_harm_kink_max_displacement_ratio": 1.5,
                "append_no_harm_rho_only_min_step_gain_delta": 1.0e-3,
                "append_no_harm_rho_only_condition_ratio_cap": 1.5,
                "append_no_harm_rho_only_step_residual_ratio_cap": 1.5,
                "append_no_harm_rho_only_displacement_ratio_cap": 1.5,
                "prune_mode": "schur_projected_shadow_v1",
                "prune_miss_threshold": 0.02,
                "prune_loss_threshold": 0.02,
                "prune_theta_block_tol": 0.10,
                "prune_state_jump_l2_tol": 0.05,
                "prune_safe_miss_increase_tol": 0.01,
                "prune_differential_miss_tol": 0.01,
                "prune_schur_ladder_local_radius": 1,
                "prune_projection_mode": "state_tangent_ls_v1",
                "prune_projection_rounds": 2,
                "prune_projection_max_active_runtime": 64,
                "prune_projection_trust_radius": 0.05,
                "prune_projection_regularization": 1.0e-8,
                "prune_ray_distance_tol": 0.05,
                "prune_shadow_enabled": True,
                "prune_shadow_horizon_steps": 2,
                "prune_shadow_score_increase_tol": 0.0,
                "prune_persistence_window": 2,
                "prune_persistence_required": 1,
                "prune_appended_origin_bias_enabled": True,
                "prune_appended_origin_target_policy": "prefer_append",
                "prune_appended_origin_grace_steps": 1,
            }
        )
    if _is_strict_qpu_faithful_profile(profile):
        return TrialParams(
            oracle_selection_policy="measured_gain_commit_veto",
            candidate_step_scales=_STEP_SCALE_PRESETS["default"],
            baseline_blend_weights="",
            baseline_gain_scales="",
            baseline_step_refine_rounds=0,
            include_tangent_secant_proposal=False,
            high_miss_no_admit_policy="bounded_stay_advance",
            miss_threshold=0.05,
            miss_abs_threshold=0.0,
            miss_persistence_spec="3:3",
            gain_ratio_threshold=0.02,
            append_margin_abs=1.0e-5,
            append_no_harm_guard_enabled=False,
            trust_radius=0.0,
            signed_energy_lead_limit=0.0,
            shortlist_size=4,
            shortlist_fraction=0.15,
            active_window_size=3,
            max_probe_positions=4,
            regularization_lambda=1.0e-8,
            candidate_regularization_lambda=1.0e-8,
            pinv_rcond=1.0e-10,
            compile_penalty_weight=0.02,
            measurement_penalty_weight=0.005,
            directional_penalty_weight=0.002,
            confirm_score_mode="compressed_whitened_v1",
            confirm_compress_fraction=0.5,
            confirm_compress_min_modes=1,
            confirm_compress_max_modes=8,
            progress_observable_window=None,
            progress_early_stop_min_checkpoint=None,
            progress_early_stop_site_error_mean_max=None,
            progress_early_stop_primary_density_error_mean_max=None,
            progress_early_stop_energy_error_mean_max=None,
            prune_mode="off",
            **_strict_auto_euler_guardrail_params(),
        )
    if str(profile) == _APPEND_PRUNE_RECOVERABILITY_PROFILE:
        baseline = _baseline_trial_params(profile="append_prune_noharm_l2_v1")
        return TrialParams(
            **{
                **asdict(baseline),
                "prune_mode": "schur_projected_shadow_v1",
                "prune_loss_threshold": 0.02,
                "prune_theta_block_tol": 0.10,
                "prune_state_jump_l2_tol": 0.05,
                "prune_safe_miss_increase_tol": 0.01,
                "prune_differential_miss_tol": 0.01,
                "prune_schur_ladder_local_radius": 1,
                "prune_projection_mode": "state_tangent_ls_v1",
                "prune_projection_rounds": 2,
                "prune_projection_max_active_runtime": 64,
                "prune_projection_trust_radius": 0.05,
                "prune_projection_regularization": 1.0e-8,
                "prune_ray_distance_tol": 0.05,
                "prune_shadow_enabled": True,
                "prune_shadow_horizon_steps": 2,
                "prune_shadow_score_increase_tol": 0.0,
                "prune_persistence_window": 2,
                "prune_persistence_required": 2,
                "prune_appended_origin_target_policy": "prefer_append",
            }
        )
    if str(profile) == _APPEND_PRUNE_NARROW_PROFILE:
        baseline = _baseline_trial_params(profile="append_prune_noharm_l2_v1")
        return TrialParams(
            **{
                **asdict(baseline),
                "append_no_harm_guard_enabled": False,
                "append_no_harm_condition_ratio_cap": 1.5,
                "append_no_harm_condition_abs_floor": 1.0,
                "append_no_harm_displacement_ratio_cap": 1.5,
                "append_no_harm_kink_max_condition_ratio": 1.5,
                "append_no_harm_kink_max_displacement_ratio": 1.5,
                "append_no_harm_rho_only_condition_ratio_cap": 1.5,
                "append_no_harm_rho_only_step_residual_ratio_cap": 1.5,
                "append_no_harm_rho_only_displacement_ratio_cap": 1.5,
                "prune_mode": "off",
                "prune_appended_origin_target_policy": "append_only",
            }
        )
    if str(profile) in _APPEND_PRUNE_TUNE_PROFILES:
        return TrialParams(
            horizon_mode="lead2",
            step_scale_mode="default",
            blend_weight_mode="default",
            gain_scale_mode="off",
            baseline_step_refine_rounds=0,
            include_tangent_secant_proposal=False,
            baseline_proposal_mode="norm_locked_blend_v1",
            primary_density_target_mode="auto",
            high_miss_no_admit_policy="bounded_stay_advance",
            miss_threshold=0.05,
            miss_abs_threshold=0.0,
            miss_persistence_spec="3:3",
            gain_ratio_threshold=0.02,
            append_margin_abs=1.0e-5,
            append_no_harm_guard_enabled=True,
            append_no_harm_condition_ratio_cap=1.0,
            append_no_harm_condition_abs_floor=1.0,
            append_no_harm_displacement_ratio_cap=1.0,
            append_no_harm_kink_min_step_gain_delta=1.0e-3,
            append_no_harm_kink_max_condition_ratio=1.0,
            append_no_harm_kink_max_displacement_ratio=1.0,
            append_no_harm_rho_only_min_step_gain_delta=1.0e-3,
            append_no_harm_rho_only_condition_ratio_cap=1.5,
            append_no_harm_rho_only_step_residual_ratio_cap=1.5,
            append_no_harm_rho_only_displacement_ratio_cap=1.5,
            trust_radius=0.75,
            signed_energy_lead_limit=1.0,
            shortlist_size=4,
            shortlist_fraction=0.15,
            active_window_size=3,
            max_probe_positions=4,
            regularization_lambda=1.0e-8,
            candidate_regularization_lambda=1.0e-8,
            pinv_rcond=1.0e-10,
            compile_penalty_weight=0.05,
            measurement_penalty_weight=0.02,
            directional_penalty_weight=0.01,
            confirm_score_mode="exact_gain_ratio",
            primary_density_weight=1.0,
            tracking_staggered_weight=1.0,
            site_weight=1.0,
            energy_weight=1.0,
            density_slope_weight=1.0,
            fidelity_weight=0.0,
            doublon_weight=0.0,
            progress_observable_window=16,
            progress_early_stop_min_checkpoint=160,
            progress_early_stop_site_error_mean_max=None,
            progress_early_stop_primary_density_error_mean_max=None,
            progress_early_stop_energy_error_mean_max=None,
            progress_early_stop_site_span_max=2.5e-3,
            progress_early_stop_primary_density_span_max=2.5e-3,
            progress_early_stop_energy_span_max=2.5e-3,
            **_strict_auto_euler_guardrail_params(),
            prune_mode="exact_local_v1",
            prune_miss_threshold=0.02,
            prune_loss_threshold=0.01,
            prune_theta_block_tol=0.05,
            prune_state_jump_l2_tol=0.05,
            prune_safe_miss_increase_tol=0.01,
            prune_appended_origin_bias_enabled=True,
            prune_appended_origin_target_policy="append_only",
            prune_appended_origin_grace_steps=1,
            prune_appended_origin_bias_scale=0.10,
            prune_appended_origin_bias_max_factor=0.50,
            prune_protection_steps=2,
            prune_stagnation_window=3,
            prune_stagnation_alpha=0.5,
            prune_stale_score_threshold=0.75,
            prune_max_candidates=2,
            prune_cooldown_steps=2,
            motion_calm_direction_cosine_threshold=0.98,
            motion_calm_rate_change_ratio_threshold=0.15,
            motion_direction_reversal_cosine_threshold=-0.05,
            motion_curvature_flip_cosine_threshold=-0.10,
            motion_acceleration_l2_threshold=0.05,
            motion_kink_rate_change_ratio_threshold=0.50,
            motion_calm_shortlist_scale=0.5,
            motion_kink_shortlist_bonus=2,
            motion_calm_oracle_budget_scale=0.5,
            motion_kink_oracle_budget_scale=2.0,
            position_jump_tie_margin_abs=1.0e-6,
            reconstruction_tol=1.0e-10,
            grouping_mode="qwc_basis_cover_reuse",
        )
    if str(profile) in _FULL_SITE3_ALGORITHM_PROFILES:
        baseline = _baseline_trial_params(
            profile="high_amp_guarded_d_shape_site3_prefsite_autobaseline_v1"
        )
        return TrialParams(
            **{
                **asdict(baseline),
                "oracle_selection_policy": "measured_gain_commit_veto",
                "candidate_step_scales": _STEP_SCALE_PRESETS["high_amp_dense"],
                "baseline_blend_weights": _BLEND_WEIGHT_PRESETS["high_amp"],
                "baseline_gain_scales": _GAIN_SCALE_PRESETS["high_amp"],
                "horizon_spec": _HORIZON_SPEC_PRESETS["high_amp_lead4"],
                "tracking_staggered_weight": 1.0,
                "single_surface_commit_law": False,
                "primary_density_scale_floor": 1.0e-6,
                "density_slope_scale_floor": 1.0e-6,
                "doublon_scale_floor": 1.0e-6,
                "site_occupations_scale_floor": 1.0e-6,
                "energy_total_scale_floor": 1.0e-6,
                "prune_miss_threshold": 0.02,
                "prune_loss_threshold": 0.01,
                "prune_theta_block_tol": 0.05,
                "prune_state_jump_l2_tol": 0.05,
                "prune_safe_miss_increase_tol": 0.01,
                "prune_protection_steps": 2,
                "prune_stagnation_window": 3,
                "prune_stagnation_alpha": 0.5,
                "prune_stale_score_threshold": 0.75,
                "prune_max_candidates": 2,
                "prune_cooldown_steps": 2,
                "motion_calm_direction_cosine_threshold": 0.98,
                "motion_calm_rate_change_ratio_threshold": 0.15,
                "motion_direction_reversal_cosine_threshold": -0.05,
                "motion_curvature_flip_cosine_threshold": -0.10,
                "motion_acceleration_l2_threshold": 0.05,
                "motion_kink_rate_change_ratio_threshold": 0.50,
                "motion_calm_shortlist_scale": 0.5,
                "motion_kink_shortlist_bonus": 2,
                "motion_calm_oracle_budget_scale": 0.5,
                "motion_kink_oracle_budget_scale": 2.0,
                "position_jump_tie_margin_abs": 1.0e-6,
                "reconstruction_tol": 1.0e-10,
                "grouping_mode": "qwc_basis_cover_reuse",
            }
        )
    site3_prefsite_guardrail_mode = _site3_prefsite_autobaseline_guardrail_mode(profile)
    if site3_prefsite_guardrail_mode is not None:
        return TrialParams(
            horizon_mode="high_amp_lead4",
            step_scale_mode="high_amp_dense",
            blend_weight_mode="high_amp",
            gain_scale_mode="high_amp",
            baseline_step_refine_rounds=2,
            include_tangent_secant_proposal=True,
            baseline_proposal_mode="norm_locked_blend_v1",
            primary_density_target_mode="pair_difference",
            miss_threshold=0.05,
            gain_ratio_threshold=0.02,
            append_margin_abs=1.0e-6,
            trust_radius=1.0,
            signed_energy_lead_limit=3.5,
            shortlist_size=4,
            shortlist_fraction=0.15,
            active_window_size=3,
            max_probe_positions=4,
            regularization_lambda=1.0e-8,
            candidate_regularization_lambda=1.0e-8,
            pinv_rcond=1.0e-10,
            compile_penalty_weight=0.05,
            measurement_penalty_weight=0.02,
            directional_penalty_weight=0.01,
            confirm_score_mode="compressed_whitened_v1",
            confirm_compress_fraction=0.5,
            confirm_compress_min_modes=1,
            confirm_compress_max_modes=8,
            primary_density_weight=2.0,
            site_weight=0.0,
            energy_weight=0.0,
            density_slope_weight=2.0,
            fidelity_weight=0.0,
            doublon_weight=1.0,
            density_curvature_weight=50.0,
            density_excursion_under_weight=225.0,
            density_excursion_over_weight=0.0,
            density_sign_lag_weight=0.25,
            density_postcross_wrong_sign_weight=0.25,
            drive_harmonic_weight=0.0,
            energy_slope_weight=500.0,
            energy_curvature_weight=200.0,
            energy_excursion_under_weight=900.0,
            energy_excursion_over_weight=120.0,
            energy_excursion_rel_tolerance=0.03,
            repeat_reopen_mode="sign_reversal_window",
            density_first_target_gain_floor=0.02,
            below_floor_probe_target_gain_floor=0.03,
            sign_lag_window_activation=0.10,
            sign_lag_window_target_gain_floor=0.005,
            postcross_wrong_sign_activation=0.0,
            postcross_wrong_sign_target_gain_floor=None,
            postcross_compare_diag=True,
            below_floor_energy_safe_d_shape_escape=False,
            below_floor_energy_safe_turn_escape=False,
            d_shape_turn_window_abs_activation=0.04,
            d_shape_outside_turn_below_floor_probe_stall_threshold=7,
            d_shape_pre_turn_shadow_bridge=True,
            guardrail_mode=site3_prefsite_guardrail_mode,
            fidelity_loss_tol=0.01,
            abs_energy_error_increase_tol=0.02,
            total_occupation_error_increase_tol=0.01,
            progress_observable_window=None,
            progress_early_stop_min_checkpoint=None,
            progress_early_stop_site_error_mean_max=None,
            progress_early_stop_primary_density_error_mean_max=None,
            progress_early_stop_energy_error_mean_max=None,
            prune_mode="exact_local_v1",
        )
    if str(profile) == "high_amp_guarded_d_shape_expanded_v1":
        return TrialParams(
            horizon_mode="high_amp_lead4",
            step_scale_mode="high_amp_dense",
            blend_weight_mode="high_amp",
            gain_scale_mode="high_amp",
            baseline_step_refine_rounds=2,
            include_tangent_secant_proposal=True,
            baseline_proposal_mode="norm_locked_blend_v1",
            primary_density_target_mode="auto",
            miss_threshold=0.05,
            gain_ratio_threshold=0.02,
            append_margin_abs=1.0e-6,
            trust_radius=1.0,
            signed_energy_lead_limit=3.5,
            shortlist_size=4,
            shortlist_fraction=0.15,
            active_window_size=3,
            max_probe_positions=4,
            regularization_lambda=1.0e-8,
            candidate_regularization_lambda=1.0e-8,
            pinv_rcond=1.0e-10,
            compile_penalty_weight=0.05,
            measurement_penalty_weight=0.02,
            directional_penalty_weight=0.01,
            confirm_score_mode="compressed_whitened_v1",
            confirm_compress_fraction=0.5,
            confirm_compress_min_modes=1,
            confirm_compress_max_modes=8,
            primary_density_weight=2.0,
            site_weight=0.0,
            energy_weight=0.0,
            density_slope_weight=2.0,
            fidelity_weight=0.0,
            doublon_weight=1.0,
            density_curvature_weight=50.0,
            density_excursion_under_weight=225.0,
            density_excursion_over_weight=0.0,
            density_sign_lag_weight=0.25,
            density_postcross_wrong_sign_weight=0.25,
            drive_harmonic_weight=0.0,
            energy_slope_weight=500.0,
            energy_curvature_weight=200.0,
            energy_excursion_under_weight=900.0,
            energy_excursion_over_weight=120.0,
            energy_excursion_rel_tolerance=0.03,
            repeat_reopen_mode="sign_reversal_window",
            density_first_target_gain_floor=0.02,
            below_floor_probe_target_gain_floor=0.03,
            sign_lag_window_activation=0.10,
            sign_lag_window_target_gain_floor=0.005,
            postcross_wrong_sign_activation=0.0,
            postcross_wrong_sign_target_gain_floor=None,
            postcross_compare_diag=True,
            below_floor_energy_safe_d_shape_escape=False,
            below_floor_energy_safe_turn_escape=False,
            d_shape_turn_window_abs_activation=0.04,
            d_shape_outside_turn_below_floor_probe_stall_threshold=0,
            d_shape_pre_turn_shadow_bridge=False,
            guardrail_mode="d_shape_barrier_v1",
            fidelity_loss_tol=0.01,
            abs_energy_error_increase_tol=0.02,
            total_occupation_error_increase_tol=0.01,
            progress_observable_window=None,
            progress_early_stop_min_checkpoint=None,
            progress_early_stop_site_error_mean_max=None,
            progress_early_stop_primary_density_error_mean_max=None,
            progress_early_stop_energy_error_mean_max=None,
            prune_mode="exact_local_v1",
        )
    if str(profile) == "high_amp_guarded_d_shape_turnwindow_v1":
        return TrialParams(
            horizon_mode="high_amp_lead4",
            step_scale_mode="high_amp_dense",
            blend_weight_mode="high_amp",
            gain_scale_mode="high_amp",
            baseline_step_refine_rounds=2,
            miss_threshold=0.05,
            gain_ratio_threshold=0.02,
            trust_radius=1.0,
            signed_energy_lead_limit=3.5,
            primary_density_weight=2.0,
            site_weight=0.0,
            energy_weight=0.0,
            density_slope_weight=2.0,
            fidelity_weight=0.0,
            doublon_weight=1.0,
            density_curvature_weight=50.0,
            density_excursion_under_weight=225.0,
            density_excursion_over_weight=0.0,
            density_sign_lag_weight=0.25,
            density_postcross_wrong_sign_weight=0.25,
            drive_harmonic_weight=0.0,
            energy_slope_weight=500.0,
            energy_curvature_weight=200.0,
            energy_excursion_under_weight=900.0,
            energy_excursion_over_weight=120.0,
            energy_excursion_rel_tolerance=0.03,
            repeat_reopen_mode="sign_reversal_window",
            density_first_target_gain_floor=0.02,
            below_floor_probe_target_gain_floor=0.03,
            sign_lag_window_activation=0.10,
            sign_lag_window_target_gain_floor=0.005,
            postcross_wrong_sign_activation=0.0,
            postcross_wrong_sign_target_gain_floor=None,
            postcross_compare_diag=True,
            below_floor_energy_safe_d_shape_escape=False,
            below_floor_energy_safe_turn_escape=False,
            d_shape_turn_window_abs_activation=0.04,
            d_shape_outside_turn_below_floor_probe_stall_threshold=0,
            d_shape_pre_turn_shadow_bridge=False,
            guardrail_mode="d_shape_barrier_v1",
            fidelity_loss_tol=0.01,
            abs_energy_error_increase_tol=0.02,
            total_occupation_error_increase_tol=0.01,
            progress_observable_window=None,
            progress_early_stop_min_checkpoint=None,
            progress_early_stop_site_error_mean_max=None,
            progress_early_stop_primary_density_error_mean_max=None,
            progress_early_stop_energy_error_mean_max=None,
            prune_mode="exact_local_v1",
        )
    if str(profile) == "high_amp_guarded_d_shape_live_v1":
        return TrialParams(
            horizon_mode="high_amp_lead4",
            step_scale_mode="high_amp_dense",
            blend_weight_mode="high_amp",
            gain_scale_mode="high_amp",
            baseline_step_refine_rounds=2,
            miss_threshold=0.05,
            gain_ratio_threshold=0.02,
            trust_radius=1.0,
            signed_energy_lead_limit=3.5,
            primary_density_weight=2.0,
            site_weight=0.0,
            energy_weight=0.0,
            density_slope_weight=2.0,
            fidelity_weight=0.0,
            doublon_weight=1.0,
            density_curvature_weight=50.0,
            density_excursion_under_weight=225.0,
            density_excursion_over_weight=0.0,
            density_sign_lag_weight=0.25,
            density_postcross_wrong_sign_weight=0.25,
            drive_harmonic_weight=0.0,
            energy_slope_weight=500.0,
            energy_curvature_weight=200.0,
            energy_excursion_under_weight=900.0,
            energy_excursion_over_weight=120.0,
            energy_excursion_rel_tolerance=0.03,
            repeat_reopen_mode="sign_reversal_window",
            density_first_target_gain_floor=0.02,
            below_floor_probe_target_gain_floor=0.03,
            sign_lag_window_activation=0.10,
            sign_lag_window_target_gain_floor=0.005,
            postcross_wrong_sign_activation=0.0,
            postcross_wrong_sign_target_gain_floor=None,
            postcross_compare_diag=True,
            below_floor_energy_safe_d_shape_escape=False,
            below_floor_energy_safe_turn_escape=False,
            d_shape_turn_window_abs_activation=0.04,
            d_shape_outside_turn_below_floor_probe_stall_threshold=0,
            d_shape_pre_turn_shadow_bridge=False,
            guardrail_mode="d_shape_barrier_v1",
            fidelity_loss_tol=0.01,
            abs_energy_error_increase_tol=0.02,
            total_occupation_error_increase_tol=0.01,
            progress_observable_window=None,
            progress_early_stop_min_checkpoint=None,
            progress_early_stop_site_error_mean_max=None,
            progress_early_stop_primary_density_error_mean_max=None,
            progress_early_stop_energy_error_mean_max=None,
            prune_mode="exact_local_v1",
        )
    if str(profile) == "high_amp_guarded_d_shape_v1":
        return TrialParams(
            horizon_mode="high_amp_lead4",
            step_scale_mode="high_amp_dense",
            blend_weight_mode="high_amp",
            gain_scale_mode="high_amp",
            baseline_step_refine_rounds=2,
            miss_threshold=0.05,
            gain_ratio_threshold=0.02,
            trust_radius=1.0,
            signed_energy_lead_limit=3.5,
            primary_density_weight=2.0,
            site_weight=0.0,
            energy_weight=0.0,
            density_slope_weight=2.0,
            fidelity_weight=0.0,
            doublon_weight=1.0,
            density_curvature_weight=50.0,
            density_excursion_under_weight=225.0,
            density_excursion_over_weight=0.0,
            density_sign_lag_weight=0.25,
            density_postcross_wrong_sign_weight=0.25,
            energy_slope_weight=500.0,
            energy_curvature_weight=200.0,
            energy_excursion_under_weight=900.0,
            energy_excursion_over_weight=120.0,
            energy_excursion_rel_tolerance=0.03,
            repeat_reopen_mode="sign_reversal_window",
            density_first_target_gain_floor=0.02,
            below_floor_probe_target_gain_floor=0.03,
            sign_lag_window_activation=0.10,
            sign_lag_window_target_gain_floor=0.005,
            postcross_wrong_sign_activation=0.0,
            postcross_wrong_sign_target_gain_floor=None,
            postcross_compare_diag=False,
            below_floor_energy_safe_d_shape_escape=False,
            guardrail_mode="d_shape_barrier_v1",
            fidelity_loss_tol=0.01,
            abs_energy_error_increase_tol=0.02,
            total_occupation_error_increase_tol=0.01,
            progress_observable_window=None,
            progress_early_stop_min_checkpoint=None,
            progress_early_stop_site_error_mean_max=None,
            progress_early_stop_primary_density_error_mean_max=None,
            progress_early_stop_energy_error_mean_max=None,
            prune_mode="exact_local_v1",
        )
    if str(profile) == "high_amp_d_shape_v1":
        return TrialParams(
            horizon_mode="high_amp_lead4",
            step_scale_mode="high_amp_dense",
            blend_weight_mode="high_amp",
            gain_scale_mode="high_amp",
            baseline_step_refine_rounds=2,
            miss_threshold=0.05,
            gain_ratio_threshold=0.02,
            trust_radius=1.0,
            signed_energy_lead_limit=3.5,
            primary_density_weight=2.0,
            site_weight=3.0,
            energy_weight=1.0,
            density_slope_weight=2.0,
            fidelity_weight=0.0,
            doublon_weight=1.0,
            density_curvature_weight=0.0,
            density_excursion_under_weight=0.0,
            density_excursion_over_weight=0.0,
            density_sign_lag_weight=1.0,
            density_postcross_wrong_sign_weight=1.0,
            energy_slope_weight=500.0,
            energy_curvature_weight=200.0,
            energy_excursion_under_weight=900.0,
            energy_excursion_over_weight=120.0,
            energy_excursion_rel_tolerance=0.03,
            repeat_reopen_mode="sign_reversal_window",
            sign_lag_window_activation=0.10,
            sign_lag_window_target_gain_floor=0.005,
            postcross_compare_diag=True,
            below_floor_energy_safe_d_shape_escape=False,
            guardrail_mode="off",
            fidelity_loss_tol=0.01,
            abs_energy_error_increase_tol=0.02,
            total_occupation_error_increase_tol=0.01,
            progress_observable_window=16,
            progress_early_stop_min_checkpoint=120,
            progress_early_stop_site_error_mean_max=0.30,
            progress_early_stop_primary_density_error_mean_max=0.60,
            progress_early_stop_energy_error_mean_max=0.05,
            prune_mode="exact_local_v1",
        )
    return TrialParams()


def _profile_seed_trials(profile: str = "generic_l2_exact_v1") -> list[TrialParams]:
    if _is_strict_guardrail_only_profile(profile):
        return [_baseline_trial_params(profile=profile)]
    if _is_strict_append_prune_aggressive_profile(profile):
        baseline = _baseline_trial_params(profile=profile)
        return [
            baseline,
            TrialParams(
                **{
                    **asdict(baseline),
                    "prune_loss_threshold": 0.12,
                    "prune_theta_block_tol": 0.50,
                    "prune_safe_miss_increase_tol": 0.12,
                    "prune_projection_rounds": 2,
                    "prune_projection_trust_radius": 0.15,
                    "prune_shadow_score_increase_tol": 5.0e-3,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "prune_appended_origin_target_policy": "prefer_append",
                    "prune_appended_origin_bias_scale": 0.75,
                    "prune_appended_origin_bias_max_factor": 1.0,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "miss_threshold": 0.03,
                    "gain_ratio_threshold": 1.0e-3,
                    "append_margin_abs": 1.0e-6,
                    "shortlist_size": 6,
                    "shortlist_fraction": 0.35,
                    "max_probe_positions": 6,
                }
            ),
        ]
    if str(profile) == _STRICT_QPU_FAITHFUL_APPEND_PRUNE_PROFILE:
        baseline = _baseline_trial_params(profile=profile)
        return [
            baseline,
            TrialParams(**{**asdict(baseline), "prune_mode": "off"}),
            TrialParams(
                **{
                    **asdict(baseline),
                    "miss_threshold": 0.03,
                    "gain_ratio_threshold": 0.01,
                    "append_margin_abs": 1.0e-6,
                    "shortlist_size": 6,
                    "shortlist_fraction": 0.25,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "prune_loss_threshold": 0.05,
                    "prune_theta_block_tol": 0.25,
                    "prune_safe_miss_increase_tol": 0.05,
                    "prune_shadow_horizon_steps": 1,
                    "prune_persistence_window": 3,
                    "prune_persistence_required": 2,
                }
            ),
        ]
    if _is_strict_qpu_faithful_profile(profile):
        baseline = _baseline_trial_params(profile=profile)
        return [
            baseline,
            TrialParams(
                **{
                    **asdict(baseline),
                    "miss_threshold": 0.03,
                    "miss_persistence_spec": "3:2",
                    "gain_ratio_threshold": 0.01,
                    "append_margin_abs": 1.0e-6,
                    "shortlist_size": 6,
                    "shortlist_fraction": 0.25,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "candidate_step_scales": _STEP_SCALE_PRESETS["wide"],
                    "shortlist_size": 8,
                    "max_probe_positions": 6,
                    "regularization_lambda": 1.0e-10,
                    "candidate_regularization_lambda": 1.0e-10,
                    "confirm_compress_fraction": 0.75,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "measurement_penalty_weight": 0.02,
                    "directional_penalty_weight": 0.01,
                    "compile_penalty_weight": 0.05,
                    "pinv_rcond": 1.0e-8,
                }
            ),
        ]
    if str(profile) == _APPEND_PRUNE_NARROW_PROFILE:
        baseline = _baseline_trial_params(profile=profile)
        return [
            baseline,
            TrialParams(
                **{
                    **asdict(baseline),
                    "prune_mode": "exact_local_v1",
                    "prune_appended_origin_target_policy": "append_only",
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "append_no_harm_guard_enabled": True,
                    "append_no_harm_condition_ratio_cap": 1.5,
                    "append_no_harm_displacement_ratio_cap": 1.5,
                    "append_no_harm_kink_max_condition_ratio": 1.5,
                    "append_no_harm_kink_max_displacement_ratio": 1.5,
                    "append_no_harm_rho_only_condition_ratio_cap": 2.0,
                    "append_no_harm_rho_only_step_residual_ratio_cap": 2.0,
                    "append_no_harm_rho_only_displacement_ratio_cap": 2.0,
                    "prune_mode": "off",
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "append_no_harm_guard_enabled": True,
                    "append_no_harm_condition_ratio_cap": 2.0,
                    "append_no_harm_displacement_ratio_cap": 2.0,
                    "append_no_harm_kink_max_condition_ratio": 2.0,
                    "append_no_harm_kink_max_displacement_ratio": 2.0,
                    "prune_mode": "exact_local_v1",
                    "prune_appended_origin_grace_steps": 2,
                    "prune_appended_origin_bias_scale": 0.5,
                    "prune_appended_origin_bias_max_factor": 1.0,
                }
            ),
        ]
    if str(profile) == _APPEND_PRUNE_RECOVERABILITY_PROFILE:
        baseline = _baseline_trial_params(profile=profile)
        return [
            baseline,
            TrialParams(
                **{
                    **asdict(baseline),
                    "prune_mode": "schur_projected_shadow_v1",
                    "prune_loss_threshold": 0.01,
                    "prune_theta_block_tol": 0.05,
                    "prune_state_jump_l2_tol": 0.05,
                    "prune_safe_miss_increase_tol": 0.01,
                    "prune_shadow_horizon_steps": 1,
                    "prune_persistence_window": 2,
                    "prune_persistence_required": 2,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "prune_mode": "schur_projected_shadow_v1",
                    "prune_loss_threshold": 0.05,
                    "prune_theta_block_tol": 0.25,
                    "prune_state_jump_l2_tol": 0.10,
                    "prune_safe_miss_increase_tol": 0.05,
                    "prune_shadow_horizon_steps": 2,
                    "prune_persistence_window": 3,
                    "prune_persistence_required": 2,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "prune_mode": "off",
                }
            ),
        ]
    if str(profile) in _APPEND_PRUNE_TUNE_PROFILES:
        baseline = _baseline_trial_params(profile=profile)
        return [
            baseline,
            TrialParams(
                **{
                    **asdict(baseline),
                    "append_no_harm_guard_enabled": False,
                    "prune_mode": "exact_local_v1",
                    "prune_appended_origin_target_policy": "append_only",
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "append_no_harm_guard_enabled": True,
                    "append_no_harm_condition_ratio_cap": 1.5,
                    "append_no_harm_displacement_ratio_cap": 1.5,
                    "append_no_harm_kink_max_condition_ratio": 1.5,
                    "append_no_harm_kink_max_displacement_ratio": 1.5,
                    "append_no_harm_rho_only_condition_ratio_cap": 2.0,
                    "append_no_harm_rho_only_step_residual_ratio_cap": 2.0,
                    "append_no_harm_rho_only_displacement_ratio_cap": 2.0,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "append_no_harm_guard_enabled": False,
                    "prune_loss_threshold": 0.05,
                    "prune_theta_block_tol": 0.1,
                    "prune_state_jump_l2_tol": 0.1,
                    "prune_safe_miss_increase_tol": 0.05,
                    "prune_appended_origin_grace_steps": 2,
                    "prune_appended_origin_bias_scale": 0.5,
                    "prune_appended_origin_bias_max_factor": 1.0,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "append_no_harm_guard_enabled": False,
                    "prune_mode": "off",
                }
            ),
        ]
    if str(profile) in _FULL_SITE3_ALGORITHM_PROFILES:
        baseline = _baseline_trial_params(profile=profile)
        return [
            baseline,
            TrialParams(
                **{
                    **asdict(baseline),
                    "guardrail_mode": "fidelity_first_barrier_v1",
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "single_surface_commit_law": True,
                    "confirm_score_mode": "exact_gain_ratio",
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "include_tangent_secant_proposal": False,
                    "candidate_step_scales": _STEP_SCALE_PRESETS["wide"],
                    "baseline_blend_weights": _BLEND_WEIGHT_PRESETS["tight"],
                    "baseline_gain_scales": _GAIN_SCALE_PRESETS["wide"],
                    "horizon_spec": _HORIZON_SPEC_PRESETS["lead4"],
                    "prune_mode": "off",
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "baseline_proposal_mode": "anticipatory_drive_basis_v1",
                    "primary_density_target_mode": "staggered",
                    "tracking_staggered_weight": 2.0,
                    "d_shape_pre_turn_shadow_bridge": False,
                    "below_floor_energy_safe_d_shape_escape": True,
                    "below_floor_energy_safe_turn_escape": True,
                }
            ),
        ]
    if _site3_prefsite_autobaseline_guardrail_mode(profile) is not None:
        baseline = _baseline_trial_params(profile=profile)
        return [
            baseline,
            TrialParams(
                **{
                    **asdict(baseline),
                    "trust_radius": 0.85,
                    "signed_energy_lead_limit": 3.0,
                    "append_margin_abs": 1.0e-5,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "confirm_score_mode": "exact_gain_ratio",
                    "shortlist_size": 6,
                    "shortlist_fraction": 0.25,
                    "active_window_size": 4,
                    "max_probe_positions": 6,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "regularization_lambda": 1.0e-6,
                    "candidate_regularization_lambda": 1.0e-6,
                    "pinv_rcond": 1.0e-8,
                    "compile_penalty_weight": 0.0,
                    "measurement_penalty_weight": 0.0,
                    "directional_penalty_weight": 0.0,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "trust_radius": 1.15,
                    "signed_energy_lead_limit": 4.0,
                    "fidelity_loss_tol": 0.015,
                    "abs_energy_error_increase_tol": 0.03,
                    "total_occupation_error_increase_tol": 0.015,
                }
            ),
        ]
    if str(profile) == "high_amp_guarded_d_shape_expanded_v1":
        baseline = _baseline_trial_params(profile=profile)
        return [
            baseline,
            TrialParams(
                **{
                    **asdict(baseline),
                    "baseline_proposal_mode": "anticipatory_drive_basis_v1",
                    "primary_density_target_mode": "pair_difference",
                    "d_shape_turn_window_abs_activation": 0.03,
                    "d_shape_pre_turn_shadow_bridge": True,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "confirm_score_mode": "exact_gain_ratio",
                    "confirm_compress_fraction": 0.25,
                    "confirm_compress_max_modes": 4,
                    "shortlist_size": 6,
                    "shortlist_fraction": 0.25,
                    "max_probe_positions": 6,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "regularization_lambda": 1.0e-6,
                    "candidate_regularization_lambda": 1.0e-6,
                    "pinv_rcond": 1.0e-8,
                    "compile_penalty_weight": 0.0,
                    "measurement_penalty_weight": 0.0,
                    "directional_penalty_weight": 0.0,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "include_tangent_secant_proposal": False,
                    "append_margin_abs": 1.0e-5,
                    "d_shape_turn_window_abs_activation": 0.02,
                    "d_shape_outside_turn_below_floor_probe_stall_threshold": 2,
                }
            ),
        ]
    if str(profile) == "high_amp_guarded_d_shape_turnwindow_v1":
        baseline = _baseline_trial_params(profile=profile)
        return [
            baseline,
            TrialParams(
                **{
                    **asdict(baseline),
                    "d_shape_turn_window_abs_activation": 0.02,
                    "d_shape_pre_turn_shadow_bridge": True,
                    "d_shape_outside_turn_below_floor_probe_stall_threshold": 1,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "d_shape_turn_window_abs_activation": 0.03,
                    "d_shape_pre_turn_shadow_bridge": True,
                    "d_shape_outside_turn_below_floor_probe_stall_threshold": 2,
                    "drive_harmonic_weight": 0.1,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "d_shape_turn_window_abs_activation": 0.05,
                    "trust_radius": 1.15,
                    "signed_energy_lead_limit": 4.0,
                    "primary_density_weight": 2.5,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "d_shape_turn_window_abs_activation": 0.06,
                    "d_shape_pre_turn_shadow_bridge": False,
                    "d_shape_outside_turn_below_floor_probe_stall_threshold": 0,
                    "density_sign_lag_weight": 0.5,
                }
            ),
        ]
    if str(profile) == "high_amp_guarded_d_shape_live_v1":
        baseline = _baseline_trial_params(profile=profile)
        return [
            baseline,
            TrialParams(
                **{
                    **asdict(baseline),
                    "drive_harmonic_weight": 0.25,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "trust_radius": 1.15,
                    "signed_energy_lead_limit": 4.0,
                    "primary_density_weight": 2.5,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "site_weight": 0.25,
                    "energy_weight": 0.25,
                    "density_slope_weight": 3.0,
                    "density_sign_lag_weight": 0.5,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "trust_radius": 0.85,
                    "signed_energy_lead_limit": 3.0,
                    "primary_density_weight": 1.5,
                    "density_sign_lag_weight": 0.1,
                }
            ),
        ]
    if str(profile) == "high_amp_guarded_d_shape_v1":
        baseline = _baseline_trial_params(profile=profile)
        return [
            TrialParams(
                **{
                    **asdict(baseline),
                    "density_first_target_gain_floor": 0.015,
                    "below_floor_probe_target_gain_floor": 0.025,
                    "sign_lag_window_target_gain_floor": 0.0025,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "trust_radius": 1.15,
                    "signed_energy_lead_limit": 4.0,
                    "density_curvature_weight": 75.0,
                    "density_excursion_under_weight": 300.0,
                }
            ),
            TrialParams(
                **{
                    **asdict(baseline),
                    "postcross_wrong_sign_activation": 0.05,
                    "postcross_wrong_sign_target_gain_floor": 0.005,
                    "density_postcross_wrong_sign_weight": 0.5,
                }
            ),
        ]
    if str(profile) != "high_amp_d_shape_v1":
        return []
    baseline = _baseline_trial_params(profile=profile)
    return [
        TrialParams(
            **{
                **asdict(baseline),
                "density_curvature_weight": 100.0,
                "density_excursion_under_weight": 450.0,
            }
        ),
        TrialParams(
            **{
                **asdict(baseline),
                "below_floor_energy_safe_d_shape_escape": True,
            }
        ),
        TrialParams(
            **{
                **asdict(baseline),
                "guardrail_mode": "d_shape_barrier_v1",
                "site_weight": 0.0,
                "energy_weight": 0.0,
                "density_sign_lag_weight": 0.25,
                "density_postcross_wrong_sign_weight": 0.25,
                "density_curvature_weight": 50.0,
                "density_excursion_under_weight": 225.0,
            }
        ),
    ]


def _trial_to_enqueue_params(
    params: TrialParams,
    *,
    profile: str,
    route_ctx: OptunaRouteContext | None = None,
) -> dict[str, Any]:
    params = _apply_route_context_to_trial_params(params, route_ctx)
    keys = list(
        _apply_route_context_to_search_space(
            _search_space(profile=profile),
            route_ctx,
        ).keys()
    )
    payload = asdict(params)
    return {
        key: payload[key]
        for key in keys
        if key in payload and payload[key] is not None
    }


def _suggest_trial_params(
    trial: Any,
    *,
    profile: str = "generic_l2_exact_v1",
    route_ctx: OptunaRouteContext | None = None,
) -> TrialParams:
    space = _apply_route_context_to_search_space(_search_space(profile=profile), route_ctx)
    if _is_strict_qpu_faithful_profile(profile):
        return _apply_route_context_to_trial_params(
            _suggest_trial_params_from_space(
                trial,
                baseline=_baseline_trial_params(profile=profile),
                space=space,
            ),
            route_ctx,
        )
    if str(profile) in _APPEND_PRUNE_TUNE_PROFILES:
        return _apply_route_context_to_trial_params(
            _suggest_trial_params_from_space(
                trial,
                baseline=_baseline_trial_params(profile=profile),
                space=space,
            ),
            route_ctx,
        )
    if str(profile) in _FULL_SITE3_ALGORITHM_PROFILES:
        return _apply_route_context_to_trial_params(
            _suggest_trial_params_from_space(
                trial,
                baseline=_baseline_trial_params(profile=profile),
                space=space,
            ),
            route_ctx,
        )
    if _site3_prefsite_autobaseline_guardrail_mode(profile) is not None:
        baseline = _baseline_trial_params(profile=profile)
        confirm_score_mode = str(
            trial.suggest_categorical("confirm_score_mode", list(space["confirm_score_mode"]))
        )
        if confirm_score_mode == "compressed_whitened_v1":
            confirm_compress_fraction = float(
                trial.suggest_categorical(
                    "confirm_compress_fraction",
                    list(space["confirm_compress_fraction"]),
                )
            )
            confirm_compress_min_modes = int(
                trial.suggest_categorical(
                    "confirm_compress_min_modes",
                    list(space["confirm_compress_min_modes"]),
                )
            )
            confirm_compress_max_modes = int(
                trial.suggest_categorical(
                    "confirm_compress_max_modes",
                    list(space["confirm_compress_max_modes"]),
                )
            )
        else:
            confirm_compress_fraction = float(baseline.confirm_compress_fraction or 0.5)
            confirm_compress_min_modes = int(baseline.confirm_compress_min_modes or 1)
            confirm_compress_max_modes = int(baseline.confirm_compress_max_modes or 8)
        return TrialParams(
            horizon_mode="high_amp_lead4",
            step_scale_mode="high_amp_dense",
            blend_weight_mode="high_amp",
            gain_scale_mode="high_amp",
            baseline_step_refine_rounds=2,
            include_tangent_secant_proposal=True,
            baseline_proposal_mode="norm_locked_blend_v1",
            primary_density_target_mode="pair_difference",
            miss_threshold=0.05,
            gain_ratio_threshold=0.02,
            append_margin_abs=float(
                trial.suggest_categorical("append_margin_abs", list(space["append_margin_abs"]))
            ),
            trust_radius=float(trial.suggest_categorical("trust_radius", list(space["trust_radius"]))),
            signed_energy_lead_limit=float(
                trial.suggest_categorical("signed_energy_lead_limit", list(space["signed_energy_lead_limit"]))
            ),
            shortlist_size=int(trial.suggest_categorical("shortlist_size", list(space["shortlist_size"]))),
            shortlist_fraction=float(
                trial.suggest_categorical("shortlist_fraction", list(space["shortlist_fraction"]))
            ),
            active_window_size=int(
                trial.suggest_categorical("active_window_size", list(space["active_window_size"]))
            ),
            max_probe_positions=int(
                trial.suggest_categorical("max_probe_positions", list(space["max_probe_positions"]))
            ),
            regularization_lambda=float(
                trial.suggest_categorical("regularization_lambda", list(space["regularization_lambda"]))
            ),
            candidate_regularization_lambda=float(
                trial.suggest_categorical(
                    "candidate_regularization_lambda",
                    list(space["candidate_regularization_lambda"]),
                )
            ),
            pinv_rcond=float(trial.suggest_categorical("pinv_rcond", list(space["pinv_rcond"]))),
            compile_penalty_weight=float(
                trial.suggest_categorical("compile_penalty_weight", list(space["compile_penalty_weight"]))
            ),
            measurement_penalty_weight=float(
                trial.suggest_categorical(
                    "measurement_penalty_weight",
                    list(space["measurement_penalty_weight"]),
                )
            ),
            directional_penalty_weight=float(
                trial.suggest_categorical(
                    "directional_penalty_weight",
                    list(space["directional_penalty_weight"]),
                )
            ),
            confirm_score_mode=confirm_score_mode,
            confirm_compress_fraction=confirm_compress_fraction,
            confirm_compress_min_modes=confirm_compress_min_modes,
            confirm_compress_max_modes=confirm_compress_max_modes,
            primary_density_weight=2.0,
            site_weight=0.0,
            energy_weight=0.0,
            density_slope_weight=2.0,
            fidelity_weight=0.0,
            doublon_weight=1.0,
            density_curvature_weight=50.0,
            density_excursion_under_weight=225.0,
            density_excursion_over_weight=0.0,
            density_sign_lag_weight=0.25,
            density_postcross_wrong_sign_weight=0.25,
            drive_harmonic_weight=0.0,
            energy_slope_weight=500.0,
            energy_curvature_weight=200.0,
            energy_excursion_under_weight=900.0,
            energy_excursion_over_weight=120.0,
            energy_excursion_rel_tolerance=0.03,
            repeat_reopen_mode="sign_reversal_window",
            density_first_target_gain_floor=0.02,
            below_floor_probe_target_gain_floor=0.03,
            sign_lag_window_activation=0.10,
            sign_lag_window_target_gain_floor=0.005,
            postcross_wrong_sign_activation=0.0,
            postcross_wrong_sign_target_gain_floor=None,
            postcross_compare_diag=True,
            below_floor_energy_safe_d_shape_escape=False,
            below_floor_energy_safe_turn_escape=False,
            d_shape_turn_window_abs_activation=0.04,
            d_shape_outside_turn_below_floor_probe_stall_threshold=7,
            d_shape_pre_turn_shadow_bridge=True,
            guardrail_mode=str(baseline.guardrail_mode),
            fidelity_loss_tol=float(
                trial.suggest_categorical("fidelity_loss_tol", list(space["fidelity_loss_tol"]))
            ),
            abs_energy_error_increase_tol=float(
                trial.suggest_categorical(
                    "abs_energy_error_increase_tol",
                    list(space["abs_energy_error_increase_tol"]),
                )
            ),
            total_occupation_error_increase_tol=float(
                trial.suggest_categorical(
                    "total_occupation_error_increase_tol",
                    list(space["total_occupation_error_increase_tol"]),
                )
            ),
            progress_observable_window=None,
            progress_early_stop_min_checkpoint=None,
            progress_early_stop_site_error_mean_max=None,
            progress_early_stop_primary_density_error_mean_max=None,
            progress_early_stop_energy_error_mean_max=None,
            prune_mode="exact_local_v1",
        )
    if str(profile) == "high_amp_guarded_d_shape_expanded_v1":
        baseline = _baseline_trial_params(profile=profile)
        include_tangent_secant_proposal = bool(
            trial.suggest_categorical(
                "include_tangent_secant_proposal",
                list(space["include_tangent_secant_proposal"]),
            )
        )
        if include_tangent_secant_proposal:
            trust_radius = float(trial.suggest_categorical("trust_radius", list(space["trust_radius"])))
            signed_energy_lead_limit = float(
                trial.suggest_categorical(
                    "signed_energy_lead_limit",
                    list(space["signed_energy_lead_limit"]),
                )
            )
        else:
            trust_radius = float(baseline.trust_radius)
            signed_energy_lead_limit = float(baseline.signed_energy_lead_limit)
        confirm_score_mode = str(
            trial.suggest_categorical("confirm_score_mode", list(space["confirm_score_mode"]))
        )
        if confirm_score_mode == "compressed_whitened_v1":
            confirm_compress_fraction = float(
                trial.suggest_categorical(
                    "confirm_compress_fraction",
                    list(space["confirm_compress_fraction"]),
                )
            )
            confirm_compress_min_modes = int(
                trial.suggest_categorical(
                    "confirm_compress_min_modes",
                    list(space["confirm_compress_min_modes"]),
                )
            )
            confirm_compress_max_modes = int(
                trial.suggest_categorical(
                    "confirm_compress_max_modes",
                    list(space["confirm_compress_max_modes"]),
                )
            )
        else:
            confirm_compress_fraction = float(baseline.confirm_compress_fraction or 0.5)
            confirm_compress_min_modes = int(baseline.confirm_compress_min_modes or 1)
            confirm_compress_max_modes = int(baseline.confirm_compress_max_modes or 8)
        return TrialParams(
            horizon_mode=str(trial.suggest_categorical("horizon_mode", list(space["horizon_mode"]))),
            step_scale_mode=str(trial.suggest_categorical("step_scale_mode", list(space["step_scale_mode"]))),
            blend_weight_mode=str(trial.suggest_categorical("blend_weight_mode", list(space["blend_weight_mode"]))),
            gain_scale_mode=str(trial.suggest_categorical("gain_scale_mode", list(space["gain_scale_mode"]))),
            baseline_step_refine_rounds=int(
                trial.suggest_categorical("baseline_step_refine_rounds", list(space["baseline_step_refine_rounds"]))
            ),
            include_tangent_secant_proposal=include_tangent_secant_proposal,
            baseline_proposal_mode=str(
                trial.suggest_categorical("baseline_proposal_mode", list(space["baseline_proposal_mode"]))
            ),
            primary_density_target_mode=str(
                trial.suggest_categorical(
                    "primary_density_target_mode",
                    list(space["primary_density_target_mode"]),
                )
            ),
            miss_threshold=float(trial.suggest_categorical("miss_threshold", list(space["miss_threshold"]))),
            gain_ratio_threshold=float(
                trial.suggest_categorical("gain_ratio_threshold", list(space["gain_ratio_threshold"]))
            ),
            append_margin_abs=float(
                trial.suggest_categorical("append_margin_abs", list(space["append_margin_abs"]))
            ),
            trust_radius=trust_radius,
            signed_energy_lead_limit=signed_energy_lead_limit,
            shortlist_size=int(trial.suggest_categorical("shortlist_size", list(space["shortlist_size"]))),
            shortlist_fraction=float(
                trial.suggest_categorical("shortlist_fraction", list(space["shortlist_fraction"]))
            ),
            active_window_size=int(
                trial.suggest_categorical("active_window_size", list(space["active_window_size"]))
            ),
            max_probe_positions=int(
                trial.suggest_categorical("max_probe_positions", list(space["max_probe_positions"]))
            ),
            regularization_lambda=float(
                trial.suggest_categorical("regularization_lambda", list(space["regularization_lambda"]))
            ),
            candidate_regularization_lambda=float(
                trial.suggest_categorical(
                    "candidate_regularization_lambda",
                    list(space["candidate_regularization_lambda"]),
                )
            ),
            pinv_rcond=float(trial.suggest_categorical("pinv_rcond", list(space["pinv_rcond"]))),
            compile_penalty_weight=float(
                trial.suggest_categorical("compile_penalty_weight", list(space["compile_penalty_weight"]))
            ),
            measurement_penalty_weight=float(
                trial.suggest_categorical(
                    "measurement_penalty_weight",
                    list(space["measurement_penalty_weight"]),
                )
            ),
            directional_penalty_weight=float(
                trial.suggest_categorical(
                    "directional_penalty_weight",
                    list(space["directional_penalty_weight"]),
                )
            ),
            confirm_score_mode=confirm_score_mode,
            confirm_compress_fraction=confirm_compress_fraction,
            confirm_compress_min_modes=confirm_compress_min_modes,
            confirm_compress_max_modes=confirm_compress_max_modes,
            primary_density_weight=float(
                trial.suggest_categorical("primary_density_weight", list(space["primary_density_weight"]))
            ),
            site_weight=float(trial.suggest_categorical("site_weight", list(space["site_weight"]))),
            energy_weight=float(trial.suggest_categorical("energy_weight", list(space["energy_weight"]))),
            density_slope_weight=float(
                trial.suggest_categorical("density_slope_weight", list(space["density_slope_weight"]))
            ),
            fidelity_weight=0.0,
            doublon_weight=1.0,
            density_curvature_weight=50.0,
            density_excursion_under_weight=225.0,
            density_excursion_over_weight=0.0,
            density_sign_lag_weight=float(
                trial.suggest_categorical("density_sign_lag_weight", list(space["density_sign_lag_weight"]))
            ),
            density_postcross_wrong_sign_weight=0.25,
            drive_harmonic_weight=float(
                trial.suggest_categorical("drive_harmonic_weight", list(space["drive_harmonic_weight"]))
            ),
            energy_slope_weight=500.0,
            energy_curvature_weight=200.0,
            energy_excursion_under_weight=900.0,
            energy_excursion_over_weight=120.0,
            energy_excursion_rel_tolerance=0.03,
            repeat_reopen_mode="sign_reversal_window",
            density_first_target_gain_floor=float(
                trial.suggest_categorical(
                    "density_first_target_gain_floor",
                    list(space["density_first_target_gain_floor"]),
                )
            ),
            below_floor_probe_target_gain_floor=float(
                trial.suggest_categorical(
                    "below_floor_probe_target_gain_floor",
                    list(space["below_floor_probe_target_gain_floor"]),
                )
            ),
            sign_lag_window_activation=0.10,
            sign_lag_window_target_gain_floor=0.005,
            postcross_wrong_sign_activation=0.0,
            postcross_wrong_sign_target_gain_floor=None,
            postcross_compare_diag=True,
            below_floor_energy_safe_d_shape_escape=False,
            below_floor_energy_safe_turn_escape=False,
            d_shape_turn_window_abs_activation=float(
                trial.suggest_categorical(
                    "d_shape_turn_window_abs_activation",
                    list(space["d_shape_turn_window_abs_activation"]),
                )
            ),
            d_shape_outside_turn_below_floor_probe_stall_threshold=int(
                trial.suggest_categorical(
                    "d_shape_outside_turn_below_floor_probe_stall_threshold",
                    list(space["d_shape_outside_turn_below_floor_probe_stall_threshold"]),
                )
            ),
            d_shape_pre_turn_shadow_bridge=trial.suggest_categorical(
                "d_shape_pre_turn_shadow_bridge",
                list(space["d_shape_pre_turn_shadow_bridge"]),
            ),
            guardrail_mode="d_shape_barrier_v1",
            fidelity_loss_tol=float(trial.suggest_categorical("fidelity_loss_tol", list(space["fidelity_loss_tol"]))),
            abs_energy_error_increase_tol=float(
                trial.suggest_categorical(
                    "abs_energy_error_increase_tol",
                    list(space["abs_energy_error_increase_tol"]),
                )
            ),
            total_occupation_error_increase_tol=float(
                trial.suggest_categorical(
                    "total_occupation_error_increase_tol",
                    list(space["total_occupation_error_increase_tol"]),
                )
            ),
            progress_observable_window=None,
            progress_early_stop_min_checkpoint=None,
            progress_early_stop_site_error_mean_max=None,
            progress_early_stop_primary_density_error_mean_max=None,
            progress_early_stop_energy_error_mean_max=None,
            prune_mode="exact_local_v1",
        )
    if str(profile) == "high_amp_guarded_d_shape_turnwindow_v1":
        return TrialParams(
            horizon_mode=str(trial.suggest_categorical("horizon_mode", list(space["horizon_mode"]))),
            step_scale_mode=str(trial.suggest_categorical("step_scale_mode", list(space["step_scale_mode"]))),
            blend_weight_mode=str(trial.suggest_categorical("blend_weight_mode", list(space["blend_weight_mode"]))),
            gain_scale_mode=str(trial.suggest_categorical("gain_scale_mode", list(space["gain_scale_mode"]))),
            baseline_step_refine_rounds=int(
                trial.suggest_categorical("baseline_step_refine_rounds", list(space["baseline_step_refine_rounds"]))
            ),
            miss_threshold=float(trial.suggest_categorical("miss_threshold", list(space["miss_threshold"]))),
            gain_ratio_threshold=float(
                trial.suggest_categorical("gain_ratio_threshold", list(space["gain_ratio_threshold"]))
            ),
            trust_radius=float(trial.suggest_categorical("trust_radius", list(space["trust_radius"]))),
            signed_energy_lead_limit=float(
                trial.suggest_categorical("signed_energy_lead_limit", list(space["signed_energy_lead_limit"]))
            ),
            primary_density_weight=float(
                trial.suggest_categorical("primary_density_weight", list(space["primary_density_weight"]))
            ),
            site_weight=float(trial.suggest_categorical("site_weight", list(space["site_weight"]))),
            energy_weight=float(trial.suggest_categorical("energy_weight", list(space["energy_weight"]))),
            density_slope_weight=float(
                trial.suggest_categorical("density_slope_weight", list(space["density_slope_weight"]))
            ),
            fidelity_weight=0.0,
            doublon_weight=1.0,
            density_curvature_weight=50.0,
            density_excursion_under_weight=225.0,
            density_excursion_over_weight=0.0,
            density_sign_lag_weight=float(
                trial.suggest_categorical("density_sign_lag_weight", list(space["density_sign_lag_weight"]))
            ),
            density_postcross_wrong_sign_weight=0.25,
            drive_harmonic_weight=float(
                trial.suggest_categorical("drive_harmonic_weight", list(space["drive_harmonic_weight"]))
            ),
            energy_slope_weight=500.0,
            energy_curvature_weight=200.0,
            energy_excursion_under_weight=900.0,
            energy_excursion_over_weight=120.0,
            energy_excursion_rel_tolerance=0.03,
            repeat_reopen_mode="sign_reversal_window",
            density_first_target_gain_floor=float(
                trial.suggest_categorical(
                    "density_first_target_gain_floor",
                    list(space["density_first_target_gain_floor"]),
                )
            ),
            below_floor_probe_target_gain_floor=float(
                trial.suggest_categorical(
                    "below_floor_probe_target_gain_floor",
                    list(space["below_floor_probe_target_gain_floor"]),
                )
            ),
            sign_lag_window_activation=0.10,
            sign_lag_window_target_gain_floor=0.005,
            postcross_wrong_sign_activation=0.0,
            postcross_wrong_sign_target_gain_floor=None,
            postcross_compare_diag=True,
            below_floor_energy_safe_d_shape_escape=False,
            below_floor_energy_safe_turn_escape=False,
            d_shape_turn_window_abs_activation=float(
                trial.suggest_categorical(
                    "d_shape_turn_window_abs_activation",
                    list(space["d_shape_turn_window_abs_activation"]),
                )
            ),
            d_shape_outside_turn_below_floor_probe_stall_threshold=int(
                trial.suggest_categorical(
                    "d_shape_outside_turn_below_floor_probe_stall_threshold",
                    list(space["d_shape_outside_turn_below_floor_probe_stall_threshold"]),
                )
            ),
            d_shape_pre_turn_shadow_bridge=trial.suggest_categorical(
                "d_shape_pre_turn_shadow_bridge",
                list(space["d_shape_pre_turn_shadow_bridge"]),
            ),
            guardrail_mode="d_shape_barrier_v1",
            fidelity_loss_tol=float(trial.suggest_categorical("fidelity_loss_tol", list(space["fidelity_loss_tol"]))),
            abs_energy_error_increase_tol=float(
                trial.suggest_categorical(
                    "abs_energy_error_increase_tol",
                    list(space["abs_energy_error_increase_tol"]),
                )
            ),
            total_occupation_error_increase_tol=float(
                trial.suggest_categorical(
                    "total_occupation_error_increase_tol",
                    list(space["total_occupation_error_increase_tol"]),
                )
            ),
            progress_observable_window=None,
            progress_early_stop_min_checkpoint=None,
            progress_early_stop_site_error_mean_max=None,
            progress_early_stop_primary_density_error_mean_max=None,
            progress_early_stop_energy_error_mean_max=None,
            prune_mode="exact_local_v1",
        )
    if str(profile) == "high_amp_guarded_d_shape_live_v1":
        return TrialParams(
            horizon_mode=str(trial.suggest_categorical("horizon_mode", list(space["horizon_mode"]))),
            step_scale_mode=str(trial.suggest_categorical("step_scale_mode", list(space["step_scale_mode"]))),
            blend_weight_mode=str(trial.suggest_categorical("blend_weight_mode", list(space["blend_weight_mode"]))),
            gain_scale_mode=str(trial.suggest_categorical("gain_scale_mode", list(space["gain_scale_mode"]))),
            baseline_step_refine_rounds=int(
                trial.suggest_categorical("baseline_step_refine_rounds", list(space["baseline_step_refine_rounds"]))
            ),
            miss_threshold=float(trial.suggest_categorical("miss_threshold", list(space["miss_threshold"]))),
            gain_ratio_threshold=float(
                trial.suggest_categorical("gain_ratio_threshold", list(space["gain_ratio_threshold"]))
            ),
            trust_radius=float(trial.suggest_categorical("trust_radius", list(space["trust_radius"]))),
            signed_energy_lead_limit=float(
                trial.suggest_categorical("signed_energy_lead_limit", list(space["signed_energy_lead_limit"]))
            ),
            primary_density_weight=float(
                trial.suggest_categorical("primary_density_weight", list(space["primary_density_weight"]))
            ),
            site_weight=float(trial.suggest_categorical("site_weight", list(space["site_weight"]))),
            energy_weight=float(trial.suggest_categorical("energy_weight", list(space["energy_weight"]))),
            density_slope_weight=float(
                trial.suggest_categorical("density_slope_weight", list(space["density_slope_weight"]))
            ),
            fidelity_weight=0.0,
            doublon_weight=1.0,
            density_curvature_weight=50.0,
            density_excursion_under_weight=225.0,
            density_excursion_over_weight=0.0,
            density_sign_lag_weight=float(
                trial.suggest_categorical("density_sign_lag_weight", list(space["density_sign_lag_weight"]))
            ),
            density_postcross_wrong_sign_weight=0.25,
            drive_harmonic_weight=float(
                trial.suggest_categorical("drive_harmonic_weight", list(space["drive_harmonic_weight"]))
            ),
            energy_slope_weight=500.0,
            energy_curvature_weight=200.0,
            energy_excursion_under_weight=900.0,
            energy_excursion_over_weight=120.0,
            energy_excursion_rel_tolerance=0.03,
            repeat_reopen_mode="sign_reversal_window",
            density_first_target_gain_floor=float(
                trial.suggest_categorical(
                    "density_first_target_gain_floor",
                    list(space["density_first_target_gain_floor"]),
                )
            ),
            below_floor_probe_target_gain_floor=float(
                trial.suggest_categorical(
                    "below_floor_probe_target_gain_floor",
                    list(space["below_floor_probe_target_gain_floor"]),
                )
            ),
            sign_lag_window_activation=0.10,
            sign_lag_window_target_gain_floor=0.005,
            postcross_wrong_sign_activation=0.0,
            postcross_wrong_sign_target_gain_floor=None,
            postcross_compare_diag=True,
            below_floor_energy_safe_d_shape_escape=bool(
                trial.suggest_categorical(
                    "below_floor_energy_safe_d_shape_escape",
                    list(space["below_floor_energy_safe_d_shape_escape"]),
                )
            ),
            below_floor_energy_safe_turn_escape=bool(
                trial.suggest_categorical(
                    "below_floor_energy_safe_turn_escape",
                    list(space["below_floor_energy_safe_turn_escape"]),
                )
            ),
            d_shape_turn_window_abs_activation=float(
                trial.suggest_categorical(
                    "d_shape_turn_window_abs_activation",
                    list(space["d_shape_turn_window_abs_activation"]),
                )
            ),
            d_shape_outside_turn_below_floor_probe_stall_threshold=int(
                trial.suggest_categorical(
                    "d_shape_outside_turn_below_floor_probe_stall_threshold",
                    list(space["d_shape_outside_turn_below_floor_probe_stall_threshold"]),
                )
            ),
            d_shape_pre_turn_shadow_bridge=trial.suggest_categorical(
                "d_shape_pre_turn_shadow_bridge",
                list(space["d_shape_pre_turn_shadow_bridge"]),
            ),
            guardrail_mode=str(trial.suggest_categorical("guardrail_mode", list(space["guardrail_mode"]))),
            fidelity_loss_tol=float(trial.suggest_categorical("fidelity_loss_tol", list(space["fidelity_loss_tol"]))),
            abs_energy_error_increase_tol=float(
                trial.suggest_categorical(
                    "abs_energy_error_increase_tol",
                    list(space["abs_energy_error_increase_tol"]),
                )
            ),
            total_occupation_error_increase_tol=float(
                trial.suggest_categorical(
                    "total_occupation_error_increase_tol",
                    list(space["total_occupation_error_increase_tol"]),
                )
            ),
            progress_observable_window=None,
            progress_early_stop_min_checkpoint=None,
            progress_early_stop_site_error_mean_max=None,
            progress_early_stop_primary_density_error_mean_max=None,
            progress_early_stop_energy_error_mean_max=None,
            prune_mode="exact_local_v1",
        )
    if str(profile) == "high_amp_guarded_d_shape_v1":
        return TrialParams(
            horizon_mode=str(trial.suggest_categorical("horizon_mode", list(space["horizon_mode"]))),
            step_scale_mode=str(trial.suggest_categorical("step_scale_mode", list(space["step_scale_mode"]))),
            blend_weight_mode=str(trial.suggest_categorical("blend_weight_mode", list(space["blend_weight_mode"]))),
            gain_scale_mode=str(trial.suggest_categorical("gain_scale_mode", list(space["gain_scale_mode"]))),
            baseline_step_refine_rounds=int(
                trial.suggest_categorical("baseline_step_refine_rounds", list(space["baseline_step_refine_rounds"]))
            ),
            miss_threshold=float(trial.suggest_categorical("miss_threshold", list(space["miss_threshold"]))),
            gain_ratio_threshold=float(
                trial.suggest_categorical("gain_ratio_threshold", list(space["gain_ratio_threshold"]))
            ),
            trust_radius=float(trial.suggest_categorical("trust_radius", list(space["trust_radius"]))),
            signed_energy_lead_limit=float(
                trial.suggest_categorical("signed_energy_lead_limit", list(space["signed_energy_lead_limit"]))
            ),
            primary_density_weight=float(
                trial.suggest_categorical("primary_density_weight", list(space["primary_density_weight"]))
            ),
            site_weight=float(trial.suggest_categorical("site_weight", list(space["site_weight"]))),
            energy_weight=float(trial.suggest_categorical("energy_weight", list(space["energy_weight"]))),
            density_slope_weight=float(
                trial.suggest_categorical("density_slope_weight", list(space["density_slope_weight"]))
            ),
            fidelity_weight=0.0,
            doublon_weight=1.0,
            density_curvature_weight=float(
                trial.suggest_categorical("density_curvature_weight", list(space["density_curvature_weight"]))
            ),
            density_excursion_under_weight=float(
                trial.suggest_categorical(
                    "density_excursion_under_weight",
                    list(space["density_excursion_under_weight"]),
                )
            ),
            density_excursion_over_weight=0.0,
            density_sign_lag_weight=float(
                trial.suggest_categorical("density_sign_lag_weight", list(space["density_sign_lag_weight"]))
            ),
            density_postcross_wrong_sign_weight=float(
                trial.suggest_categorical(
                    "density_postcross_wrong_sign_weight",
                    list(space["density_postcross_wrong_sign_weight"]),
                )
            ),
            energy_slope_weight=500.0,
            energy_curvature_weight=200.0,
            energy_excursion_under_weight=900.0,
            energy_excursion_over_weight=120.0,
            energy_excursion_rel_tolerance=0.03,
            repeat_reopen_mode="sign_reversal_window",
            density_first_target_gain_floor=float(
                trial.suggest_categorical(
                    "density_first_target_gain_floor",
                    list(space["density_first_target_gain_floor"]),
                )
            ),
            below_floor_probe_target_gain_floor=float(
                trial.suggest_categorical(
                    "below_floor_probe_target_gain_floor",
                    list(space["below_floor_probe_target_gain_floor"]),
                )
            ),
            sign_lag_window_activation=0.10,
            sign_lag_window_target_gain_floor=float(
                trial.suggest_categorical(
                    "sign_lag_window_target_gain_floor",
                    list(space["sign_lag_window_target_gain_floor"]),
                )
            ),
            postcross_wrong_sign_activation=float(
                trial.suggest_categorical(
                    "postcross_wrong_sign_activation",
                    list(space["postcross_wrong_sign_activation"]),
                )
            ),
            postcross_wrong_sign_target_gain_floor=trial.suggest_categorical(
                "postcross_wrong_sign_target_gain_floor",
                list(space["postcross_wrong_sign_target_gain_floor"]),
            ),
            postcross_compare_diag=False,
            below_floor_energy_safe_d_shape_escape=bool(
                trial.suggest_categorical(
                    "below_floor_energy_safe_d_shape_escape",
                    list(space["below_floor_energy_safe_d_shape_escape"]),
                )
            ),
            guardrail_mode=str(trial.suggest_categorical("guardrail_mode", list(space["guardrail_mode"]))),
            fidelity_loss_tol=float(trial.suggest_categorical("fidelity_loss_tol", list(space["fidelity_loss_tol"]))),
            abs_energy_error_increase_tol=float(
                trial.suggest_categorical(
                    "abs_energy_error_increase_tol",
                    list(space["abs_energy_error_increase_tol"]),
                )
            ),
            total_occupation_error_increase_tol=float(
                trial.suggest_categorical(
                    "total_occupation_error_increase_tol",
                    list(space["total_occupation_error_increase_tol"]),
                )
            ),
            progress_observable_window=None,
            progress_early_stop_min_checkpoint=None,
            progress_early_stop_site_error_mean_max=None,
            progress_early_stop_primary_density_error_mean_max=None,
            progress_early_stop_energy_error_mean_max=None,
            prune_mode="exact_local_v1",
        )
    if str(profile) == "high_amp_d_shape_v1":
        return TrialParams(
            horizon_mode=str(trial.suggest_categorical("horizon_mode", list(space["horizon_mode"]))),
            step_scale_mode=str(trial.suggest_categorical("step_scale_mode", list(space["step_scale_mode"]))),
            blend_weight_mode=str(trial.suggest_categorical("blend_weight_mode", list(space["blend_weight_mode"]))),
            gain_scale_mode=str(trial.suggest_categorical("gain_scale_mode", list(space["gain_scale_mode"]))),
            baseline_step_refine_rounds=int(
                trial.suggest_categorical("baseline_step_refine_rounds", list(space["baseline_step_refine_rounds"]))
            ),
            miss_threshold=float(trial.suggest_categorical("miss_threshold", list(space["miss_threshold"]))),
            gain_ratio_threshold=float(
                trial.suggest_categorical("gain_ratio_threshold", list(space["gain_ratio_threshold"]))
            ),
            trust_radius=float(trial.suggest_categorical("trust_radius", list(space["trust_radius"]))),
            signed_energy_lead_limit=float(
                trial.suggest_categorical("signed_energy_lead_limit", list(space["signed_energy_lead_limit"]))
            ),
            primary_density_weight=float(
                trial.suggest_categorical("primary_density_weight", list(space["primary_density_weight"]))
            ),
            site_weight=float(trial.suggest_categorical("site_weight", list(space["site_weight"]))),
            energy_weight=float(trial.suggest_categorical("energy_weight", list(space["energy_weight"]))),
            density_slope_weight=float(
                trial.suggest_categorical("density_slope_weight", list(space["density_slope_weight"]))
            ),
            fidelity_weight=0.0,
            doublon_weight=1.0,
            density_curvature_weight=float(
                trial.suggest_categorical("density_curvature_weight", list(space["density_curvature_weight"]))
            ),
            density_excursion_under_weight=float(
                trial.suggest_categorical(
                    "density_excursion_under_weight",
                    list(space["density_excursion_under_weight"]),
                )
            ),
            density_excursion_over_weight=0.0,
            density_sign_lag_weight=float(
                trial.suggest_categorical("density_sign_lag_weight", list(space["density_sign_lag_weight"]))
            ),
            density_postcross_wrong_sign_weight=float(
                trial.suggest_categorical(
                    "density_postcross_wrong_sign_weight",
                    list(space["density_postcross_wrong_sign_weight"]),
                )
            ),
            energy_slope_weight=500.0,
            energy_curvature_weight=200.0,
            energy_excursion_under_weight=900.0,
            energy_excursion_over_weight=120.0,
            energy_excursion_rel_tolerance=0.03,
            repeat_reopen_mode="sign_reversal_window",
            sign_lag_window_activation=0.10,
            sign_lag_window_target_gain_floor=float(
                trial.suggest_categorical(
                    "sign_lag_window_target_gain_floor",
                    list(space["sign_lag_window_target_gain_floor"]),
                )
            ),
            postcross_compare_diag=True,
            below_floor_energy_safe_d_shape_escape=bool(
                trial.suggest_categorical(
                    "below_floor_energy_safe_d_shape_escape",
                    list(space["below_floor_energy_safe_d_shape_escape"]),
                )
            ),
            guardrail_mode=str(trial.suggest_categorical("guardrail_mode", list(space["guardrail_mode"]))),
            fidelity_loss_tol=float(trial.suggest_categorical("fidelity_loss_tol", list(space["fidelity_loss_tol"]))),
            abs_energy_error_increase_tol=float(
                trial.suggest_categorical(
                    "abs_energy_error_increase_tol",
                    list(space["abs_energy_error_increase_tol"]),
                )
            ),
            total_occupation_error_increase_tol=float(
                trial.suggest_categorical(
                    "total_occupation_error_increase_tol",
                    list(space["total_occupation_error_increase_tol"]),
                )
            ),
            progress_observable_window=16,
            progress_early_stop_min_checkpoint=120,
            progress_early_stop_site_error_mean_max=0.30,
            progress_early_stop_primary_density_error_mean_max=0.60,
            progress_early_stop_energy_error_mean_max=0.05,
            prune_mode="exact_local_v1",
        )
    return _apply_route_context_to_trial_params(
        TrialParams(
            horizon_mode=str(trial.suggest_categorical("horizon_mode", list(space["horizon_mode"]))),
            step_scale_mode=str(trial.suggest_categorical("step_scale_mode", list(space["step_scale_mode"]))),
            blend_weight_mode=str(trial.suggest_categorical("blend_weight_mode", list(space["blend_weight_mode"]))),
            gain_scale_mode=str(trial.suggest_categorical("gain_scale_mode", list(space["gain_scale_mode"]))),
            baseline_step_refine_rounds=int(
                trial.suggest_categorical("baseline_step_refine_rounds", list(space["baseline_step_refine_rounds"]))
            ),
            primary_density_target_mode=str(
                trial.suggest_categorical("primary_density_target_mode", list(space["primary_density_target_mode"]))
            ),
            high_miss_no_admit_policy=str(
                trial.suggest_categorical("high_miss_no_admit_policy", list(space["high_miss_no_admit_policy"]))
            ),
            miss_threshold=float(trial.suggest_categorical("miss_threshold", list(space["miss_threshold"]))),
            miss_abs_threshold=float(
                trial.suggest_categorical("miss_abs_threshold", list(space["miss_abs_threshold"]))
            ),
            miss_persistence_spec=str(
                trial.suggest_categorical("miss_persistence_spec", list(space["miss_persistence_spec"]))
            ),
            gain_ratio_threshold=float(
                trial.suggest_categorical("gain_ratio_threshold", list(space["gain_ratio_threshold"]))
            ),
            append_margin_abs=float(trial.suggest_categorical("append_margin_abs", list(space["append_margin_abs"]))),
            compile_penalty_weight=float(
                trial.suggest_categorical("compile_penalty_weight", list(space["compile_penalty_weight"]))
            ),
            measurement_penalty_weight=float(
                trial.suggest_categorical("measurement_penalty_weight", list(space["measurement_penalty_weight"]))
            ),
            directional_penalty_weight=float(
                trial.suggest_categorical("directional_penalty_weight", list(space["directional_penalty_weight"]))
            ),
            confirm_score_mode=str(trial.suggest_categorical("confirm_score_mode", list(space["confirm_score_mode"]))),
            trust_radius=float(trial.suggest_categorical("trust_radius", list(space["trust_radius"]))),
            signed_energy_lead_limit=float(
                trial.suggest_categorical(
                    "signed_energy_lead_limit",
                    list(space["signed_energy_lead_limit"]),
                )
            ),
            primary_density_weight=float(
                trial.suggest_categorical("primary_density_weight", list(space["primary_density_weight"]))
            ),
            site_weight=float(trial.suggest_categorical("site_weight", list(space["site_weight"]))),
            energy_weight=float(trial.suggest_categorical("energy_weight", list(space["energy_weight"]))),
            density_slope_weight=float(
                trial.suggest_categorical("density_slope_weight", list(space["density_slope_weight"]))
            ),
        ),
        route_ctx,
    )


def _build_base_run_config(args: argparse.Namespace) -> BaseRunConfig:
    integrator_policy_override = (
        None
        if args.integrator_policy_override in {None, ""}
        else str(args.integrator_policy_override).strip().lower()
    )
    return BaseRunConfig(
        artifact_json=Path(args.artifact_json).expanduser().resolve(),
        study_profile=str(args.study_profile),
        tuning_class=(None if args.tuning_class in {None, ""} else str(args.tuning_class)),
        class_settings_source=str(args.class_settings_source),
        class_settings_lock_json=str(Path(args.class_settings_lock_json).expanduser().resolve()),
        loader_mode=str(args.loader_mode),
        generator_family=str(args.generator_family),
        fallback_family=str(args.fallback_family),
        append_pool_family=str(args.append_pool_family),
        lock_fixed_manifold=bool(args.lock_fixed_manifold),
        allow_repeats=bool(args.allow_repeats),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        enable_drive=bool(args.enable_drive),
        disable_drive=bool(args.disable_drive),
        drive_A=float(args.drive_A),
        drive_omega=float(args.drive_omega),
        drive_tbar=float(args.drive_tbar),
        drive_phi=float(args.drive_phi),
        drive_pattern=str(args.drive_pattern),
        drive_custom_weights=str(args.drive_custom_weights),
        drive_include_identity=bool(args.drive_include_identity),
        drive_time_sampling=str(args.drive_time_sampling),
        drive_t0=float(args.drive_t0),
        exact_steps_multiplier=int(args.exact_steps_multiplier),
        pair=str(args.pair),
        objective_window_start=(
            None if args.objective_window_start is None else float(args.objective_window_start)
        ),
        objective_window_end=(
            None if args.objective_window_end is None else float(args.objective_window_end)
        ),
        spectra_detrend=str(args.spectra_detrend),
        spectra_window=str(args.spectra_window),
        max_peaks=int(args.max_peaks),
        max_harmonic=int(args.max_harmonic),
        skip_spectra_pdf=bool(args.skip_spectra_pdf),
        integrator_policy_override=integrator_policy_override,
    )


def _build_objective_weights(args: argparse.Namespace) -> ObjectiveWeights:
    defaults = _default_objective_weights(profile=str(args.study_profile))
    return ObjectiveWeights(
        pair_mae_over_exact_span=float(
            defaults.pair_mae_over_exact_span
            if args.objective_weight_pair_mae_over_span is None
            else args.objective_weight_pair_mae_over_span
        ),
        epsilon_osc_pair=float(
            defaults.epsilon_osc_pair
            if args.objective_weight_epsilon_osc is None
            else args.objective_weight_epsilon_osc
        ),
        dominant_peak_abs_omega_error=float(
            defaults.dominant_peak_abs_omega_error
            if args.objective_weight_peak_omega is None
            else args.objective_weight_peak_omega
        ),
        pair_corr_defect=float(
            defaults.pair_corr_defect
            if args.objective_weight_pair_corr_defect is None
            else args.objective_weight_pair_corr_defect
        ),
        mean_abs_site_occupations_error=float(
            defaults.mean_abs_site_occupations_error
            if args.objective_weight_site_mae is None
            else args.objective_weight_site_mae
        ),
        mean_abs_energy_total_error=float(
            defaults.mean_abs_energy_total_error
            if args.objective_weight_energy_mae is None
            else args.objective_weight_energy_mae
        ),
        mean_signed_energy_total_error_abs=float(
            defaults.mean_signed_energy_total_error_abs
            if args.objective_weight_energy_bias_abs is None
            else args.objective_weight_energy_bias_abs
        ),
        max_abs_energy_total_error=float(
            defaults.max_abs_energy_total_error
            if args.objective_weight_energy_max_abs is None
            else args.objective_weight_energy_max_abs
        ),
        energy_total_excursion_under_response_mean=float(
            defaults.energy_total_excursion_under_response_mean
            if args.objective_weight_energy_under_response_mean is None
            else args.objective_weight_energy_under_response_mean
        ),
        energy_total_excursion_under_response_max=float(
            defaults.energy_total_excursion_under_response_max
            if args.objective_weight_energy_under_response_max is None
            else args.objective_weight_energy_under_response_max
        ),
        fidelity_defect=float(
            defaults.fidelity_defect
            if args.objective_weight_fidelity_defect is None
            else args.objective_weight_fidelity_defect
        ),
        final_runtime_parameter_count=float(
            defaults.final_runtime_parameter_count
            if args.objective_weight_runtime_count is None
            else args.objective_weight_runtime_count
        ),
        append_count=float(
            defaults.append_count
            if args.objective_weight_append_count is None
            else args.objective_weight_append_count
        ),
        prune_count=float(
            defaults.prune_count
            if args.objective_weight_prune_count is None
            else args.objective_weight_prune_count
        ),
        rk4_count=float(
            defaults.rk4_count
            if args.objective_weight_rk4_count is None
            else args.objective_weight_rk4_count
        ),
    )


def _build_validity_gates(args: argparse.Namespace) -> ValidityGates:
    defaults = _default_validity_gates(profile=str(args.study_profile))
    return ValidityGates(
        max_mean_abs_energy_total_error=(
            defaults.max_mean_abs_energy_total_error
            if args.invalid_max_mean_energy_mae is None
            else float(args.invalid_max_mean_energy_mae)
        ),
        max_final_abs_energy_total_error=(
            defaults.max_final_abs_energy_total_error
            if args.invalid_max_final_energy_mae is None
            else float(args.invalid_max_final_energy_mae)
        ),
        min_fidelity_exact=(
            defaults.min_fidelity_exact
            if args.invalid_min_fidelity is None
            else float(args.invalid_min_fidelity)
        ),
        max_mean_total_occupation_abs_error=(
            defaults.max_mean_total_occupation_abs_error
            if args.invalid_max_mean_total_occupation_mae is None
            else float(args.invalid_max_mean_total_occupation_mae)
        ),
        max_primary_observable_mae_over_exact_span=(
            defaults.max_primary_observable_mae_over_exact_span
            if args.invalid_max_primary_observable_mae_over_span is None
            else float(args.invalid_max_primary_observable_mae_over_span)
        ),
    )


def _build_strict_qpu_faithful_realtime_tokens(
    *,
    base_cfg: BaseRunConfig,
    params: TrialParams,
    output_json: Path,
    run_tag: str,
    progress_json: Path | None = None,
    partial_payload_json: Path | None = None,
) -> list[str]:
    if bool(base_cfg.enable_drive) and bool(base_cfg.disable_drive):
        raise ValueError("BaseRunConfig cannot set both enable_drive and disable_drive.")
    candidate_step_scales = _resolve_csv_override(
        explicit=params.candidate_step_scales,
        preset_key=str(params.step_scale_mode),
        preset_map=_STEP_SCALE_PRESETS,
    )
    strict_prune_mode = "off" if params.prune_mode in {None, ""} else str(params.prune_mode)
    strict_high_miss_policy = (
        "bounded_stay_advance"
        if params.high_miss_no_admit_policy in {None, ""}
        else normalize_high_miss_no_admit_policy(params.high_miss_no_admit_policy)
    )
    strict_oracle_selection_policy = (
        "measured_gain_commit_veto"
        if params.oracle_selection_policy in {None, ""}
        else str(params.oracle_selection_policy)
    )
    tokens = [
        "--artifact-json",
        str(base_cfg.artifact_json),
        "--output-json",
        str(output_json),
        "--run-tag",
        str(run_tag),
        "--loader-mode",
        str(base_cfg.loader_mode),
        "--generator-family",
        str(base_cfg.generator_family),
        "--fallback-family",
        str(base_cfg.fallback_family),
        "--append-pool-family",
        str(base_cfg.append_pool_family),
        "--t-final",
        str(float(base_cfg.t_final)),
        "--num-times",
        str(int(base_cfg.num_times)),
        "--checkpoint-controller-mode",
        "observable_v1",
        "--checkpoint-controller-reference-mode",
        "off",
        "--checkpoint-controller-exact-input-mode",
        "off",
        "--checkpoint-controller-strict-qpu-faithful",
        "--checkpoint-controller-noise-mode",
        "ideal",
        "--checkpoint-controller-integrator-policy",
        str(params.integrator_policy or "auto_euler_rk4"),
        "--checkpoint-controller-prune-mode",
        str(strict_prune_mode),
        "--checkpoint-controller-high-miss-no-admit-policy",
        str(strict_high_miss_policy),
        "--checkpoint-controller-oracle-selection-policy",
        str(strict_oracle_selection_policy),
        "--checkpoint-controller-miss-threshold",
        str(float(params.miss_threshold)),
        "--checkpoint-controller-gain-ratio-threshold",
        str(float(params.gain_ratio_threshold)),
        "--checkpoint-controller-candidate-step-scales",
        str(candidate_step_scales),
    ]
    optional_float_flags = [
        ("--checkpoint-controller-miss-abs-threshold", params.miss_abs_threshold),
        ("--checkpoint-controller-append-margin-abs", params.append_margin_abs),
        ("--checkpoint-controller-append-no-harm-condition-ratio-cap", params.append_no_harm_condition_ratio_cap),
        ("--checkpoint-controller-append-no-harm-condition-abs-floor", params.append_no_harm_condition_abs_floor),
        ("--checkpoint-controller-append-no-harm-displacement-ratio-cap", params.append_no_harm_displacement_ratio_cap),
        ("--checkpoint-controller-append-no-harm-kink-min-step-gain-delta", params.append_no_harm_kink_min_step_gain_delta),
        ("--checkpoint-controller-append-no-harm-kink-max-condition-ratio", params.append_no_harm_kink_max_condition_ratio),
        ("--checkpoint-controller-append-no-harm-kink-max-displacement-ratio", params.append_no_harm_kink_max_displacement_ratio),
        ("--checkpoint-controller-append-no-harm-rho-only-min-step-gain-delta", params.append_no_harm_rho_only_min_step_gain_delta),
        ("--checkpoint-controller-append-no-harm-rho-only-condition-ratio-cap", params.append_no_harm_rho_only_condition_ratio_cap),
        ("--checkpoint-controller-append-no-harm-rho-only-step-residual-ratio-cap", params.append_no_harm_rho_only_step_residual_ratio_cap),
        ("--checkpoint-controller-append-no-harm-rho-only-displacement-ratio-cap", params.append_no_harm_rho_only_displacement_ratio_cap),
        ("--checkpoint-controller-integrator-columnarity-threshold", params.integrator_columnarity_threshold),
        ("--checkpoint-controller-integrator-curvature-threshold", params.integrator_curvature_threshold),
        ("--checkpoint-controller-integrator-euler-fs-error-threshold", params.integrator_euler_fs_error_threshold),
        ("--checkpoint-controller-integrator-condition-max", params.integrator_condition_max),
        ("--checkpoint-controller-integrator-euler-min-time-fraction", params.integrator_euler_min_time_fraction),
        ("--checkpoint-controller-integrator-euler-site-span-max", params.integrator_euler_site_span_max),
        (
            "--checkpoint-controller-integrator-euler-primary-density-span-max",
            params.integrator_euler_primary_density_span_max,
        ),
        ("--checkpoint-controller-integrator-euler-energy-span-max", params.integrator_euler_energy_span_max),
        ("--checkpoint-controller-shortlist-fraction", params.shortlist_fraction),
        ("--checkpoint-controller-regularization-lambda", params.regularization_lambda),
        ("--checkpoint-controller-candidate-regularization-lambda", params.candidate_regularization_lambda),
        ("--checkpoint-controller-pinv-rcond", params.pinv_rcond),
        ("--checkpoint-controller-compile-penalty-weight", params.compile_penalty_weight),
        ("--checkpoint-controller-measurement-penalty-weight", params.measurement_penalty_weight),
        ("--checkpoint-controller-directional-penalty-weight", params.directional_penalty_weight),
        ("--checkpoint-controller-confirm-compress-fraction", params.confirm_compress_fraction),
        ("--checkpoint-controller-prune-miss-threshold", params.prune_miss_threshold),
        ("--checkpoint-controller-prune-loss-threshold", params.prune_loss_threshold),
        ("--checkpoint-controller-prune-theta-block-tol", params.prune_theta_block_tol),
        ("--checkpoint-controller-prune-state-jump-l2-tol", params.prune_state_jump_l2_tol),
        ("--checkpoint-controller-prune-safe-miss-increase-tol", params.prune_safe_miss_increase_tol),
        ("--checkpoint-controller-prune-schur-monotonicity-tol", params.prune_schur_monotonicity_tol),
        ("--checkpoint-controller-prune-loss-norm-epsilon", params.prune_loss_norm_epsilon),
        ("--checkpoint-controller-prune-differential-miss-tol", params.prune_differential_miss_tol),
        ("--checkpoint-controller-prune-projection-trust-radius", params.prune_projection_trust_radius),
        ("--checkpoint-controller-prune-projection-regularization", params.prune_projection_regularization),
        ("--checkpoint-controller-prune-ray-distance-tol", params.prune_ray_distance_tol),
        ("--checkpoint-controller-prune-shadow-score-increase-tol", params.prune_shadow_score_increase_tol),
        ("--checkpoint-controller-prune-appended-origin-bias-scale", params.prune_appended_origin_bias_scale),
        ("--checkpoint-controller-prune-appended-origin-bias-max-factor", params.prune_appended_origin_bias_max_factor),
    ]
    for flag, value in optional_float_flags:
        if value is not None:
            tokens.extend([str(flag), str(float(value))])
    optional_int_flags = [
        ("--checkpoint-controller-shortlist-size", params.shortlist_size),
        ("--checkpoint-controller-active-window-size", params.active_window_size),
        ("--checkpoint-controller-max-probe-positions", params.max_probe_positions),
        ("--checkpoint-controller-confirm-compress-min-modes", params.confirm_compress_min_modes),
        ("--checkpoint-controller-confirm-compress-max-modes", params.confirm_compress_max_modes),
        ("--checkpoint-controller-prune-schur-ladder-local-radius", params.prune_schur_ladder_local_radius),
        ("--checkpoint-controller-prune-projection-rounds", params.prune_projection_rounds),
        ("--checkpoint-controller-prune-projection-max-active-runtime", params.prune_projection_max_active_runtime),
        ("--checkpoint-controller-prune-shadow-horizon-steps", params.prune_shadow_horizon_steps),
        ("--checkpoint-controller-prune-persistence-window", params.prune_persistence_window),
        ("--checkpoint-controller-prune-persistence-required", params.prune_persistence_required),
        ("--checkpoint-controller-prune-appended-origin-grace-steps", params.prune_appended_origin_grace_steps),
        ("--checkpoint-controller-prune-initial-scaffold-grace-steps", params.prune_initial_scaffold_grace_steps),
    ]
    for flag, value in optional_int_flags:
        if value is not None:
            tokens.extend([str(flag), str(int(value))])
    optional_str_flags = [
        ("--checkpoint-controller-prune-projection-mode", params.prune_projection_mode),
        ("--checkpoint-controller-prune-appended-origin-target-policy", params.prune_appended_origin_target_policy),
    ]
    for flag, value in optional_str_flags:
        if value not in {None, ""}:
            tokens.extend([str(flag), str(value)])
    if params.confirm_score_mode not in {None, ""}:
        tokens.extend(["--checkpoint-controller-confirm-score-mode", str(params.confirm_score_mode)])
    if _is_strict_append_prune_controller_profile(base_cfg.study_profile):
        if params.append_no_harm_guard_enabled is True:
            tokens.append("--checkpoint-controller-append-no-harm-guard-enabled")
        elif params.append_no_harm_guard_enabled is False:
            tokens.append("--no-checkpoint-controller-append-no-harm-guard-enabled")
    if params.prune_high_miss_differential_enabled is True:
        tokens.append("--checkpoint-controller-prune-high-miss-differential-enabled")
    elif params.prune_high_miss_differential_enabled is False:
        tokens.append("--no-checkpoint-controller-prune-high-miss-differential-enabled")
    if params.prune_shadow_enabled is True:
        tokens.append("--checkpoint-controller-prune-shadow-enabled")
    elif params.prune_shadow_enabled is False:
        tokens.append("--no-checkpoint-controller-prune-shadow-enabled")
    if params.prune_appended_origin_bias_enabled is True:
        tokens.append("--checkpoint-controller-prune-appended-origin-bias-enabled")
    elif params.prune_appended_origin_bias_enabled is False:
        tokens.append("--no-checkpoint-controller-prune-appended-origin-bias-enabled")
    if params.miss_persistence_spec not in {None, ""}:
        raw_window, raw_count = str(params.miss_persistence_spec).split(":", 1)
        tokens.extend(
            [
                "--checkpoint-controller-miss-persistence-window",
                str(int(raw_window)),
                "--checkpoint-controller-miss-persistence-count",
                str(int(raw_count)),
            ]
        )
    if progress_json is not None:
        tokens.extend(["--progress-json", str(progress_json)])
    if partial_payload_json is not None:
        tokens.extend(["--partial-payload-json", str(partial_payload_json)])
    if bool(base_cfg.lock_fixed_manifold):
        tokens.append("--lock-fixed-manifold")
    if bool(base_cfg.allow_repeats):
        tokens.append("--allow-repeats")
    if bool(base_cfg.disable_drive):
        tokens.append("--disable-drive")
    elif bool(base_cfg.enable_drive):
        tokens.extend(
            [
                "--enable-drive",
                "--drive-A",
                str(float(base_cfg.drive_A)),
                "--drive-omega",
                str(float(base_cfg.drive_omega)),
                "--drive-tbar",
                str(float(base_cfg.drive_tbar)),
                "--drive-phi",
                str(float(base_cfg.drive_phi)),
                "--drive-pattern",
                str(base_cfg.drive_pattern),
                "--drive-custom-weights",
                str(base_cfg.drive_custom_weights),
                "--drive-time-sampling",
                str(base_cfg.drive_time_sampling),
                "--drive-t0",
                str(float(base_cfg.drive_t0)),
            ]
        )
        if bool(base_cfg.drive_include_identity):
            tokens.append("--drive-include-identity")
    return tokens


def _build_realtime_tokens(
    *,
    base_cfg: BaseRunConfig,
    params: TrialParams,
    output_json: Path,
    run_tag: str,
    progress_json: Path | None = None,
    partial_payload_json: Path | None = None,
    route_ctx: OptunaRouteContext | None = None,
) -> list[str]:
    if _is_strict_qpu_faithful_profile(base_cfg.study_profile):
        return _build_strict_qpu_faithful_realtime_tokens(
            base_cfg=base_cfg,
            params=params,
            output_json=output_json,
            run_tag=run_tag,
            progress_json=progress_json,
            partial_payload_json=partial_payload_json,
        )
    params = _apply_route_context_to_trial_params(params, route_ctx)
    horizon_steps, horizon_weights = _resolve_horizon_override(params)
    candidate_step_scales = _resolve_csv_override(
        explicit=params.candidate_step_scales,
        preset_key=str(params.step_scale_mode),
        preset_map=_STEP_SCALE_PRESETS,
    )
    blend_weights = _resolve_csv_override(
        explicit=params.baseline_blend_weights,
        preset_key=str(params.blend_weight_mode),
        preset_map=_BLEND_WEIGHT_PRESETS,
    )
    gain_scales = _resolve_csv_override(
        explicit=params.baseline_gain_scales,
        preset_key=str(params.gain_scale_mode),
        preset_map=_GAIN_SCALE_PRESETS,
    )
    tokens = [
        "--artifact-json",
        str(base_cfg.artifact_json),
        "--output-json",
        str(output_json),
        "--run-tag",
        str(run_tag),
        "--loader-mode",
        str(base_cfg.loader_mode),
        "--generator-family",
        str(base_cfg.generator_family),
        "--fallback-family",
        str(base_cfg.fallback_family),
        "--append-pool-family",
        str(base_cfg.append_pool_family),
        "--t-final",
        str(float(base_cfg.t_final)),
        "--num-times",
        str(int(base_cfg.num_times)),
        "--exact-steps-multiplier",
        str(int(base_cfg.exact_steps_multiplier)),
        "--checkpoint-controller-mode",
        "exact_v1",
        "--checkpoint-controller-reference-mode",
        "benchmark_exact",
        "--checkpoint-controller-integrator-policy",
        str(params.integrator_policy or "auto_euler_rk4"),
        "--checkpoint-controller-miss-threshold",
        str(float(params.miss_threshold)),
        "--checkpoint-controller-gain-ratio-threshold",
        str(float(params.gain_ratio_threshold)),
        "--checkpoint-controller-candidate-step-scales",
        str(candidate_step_scales),
        "--checkpoint-controller-exact-forecast-baseline-step-refine-rounds",
        str(int(params.baseline_step_refine_rounds)),
        "--checkpoint-controller-exact-forecast-tangent-secant-trust-radius",
        str(float(params.trust_radius)),
        "--checkpoint-controller-exact-forecast-tangent-secant-signed-energy-lead-limit",
        str(float(params.signed_energy_lead_limit)),
        "--checkpoint-controller-exact-forecast-horizon-steps",
        str(int(horizon_steps)),
        "--checkpoint-controller-exact-forecast-horizon-weights",
        str(horizon_weights),
        "--checkpoint-controller-exact-forecast-tracking-primary-density-error-weight",
        str(float(params.primary_density_weight)),
        "--checkpoint-controller-exact-forecast-tracking-staggered-error-weight",
        str(float(params.tracking_staggered_weight if params.tracking_staggered_weight is not None else 1.0)),
        "--checkpoint-controller-exact-forecast-tracking-site-occupations-error-weight",
        str(float(params.site_weight)),
        "--checkpoint-controller-exact-forecast-tracking-energy-total-error-weight",
        str(float(params.energy_weight)),
        "--checkpoint-controller-exact-forecast-density-slope-weight",
        str(float(params.density_slope_weight)),
    ]
    if params.oracle_selection_policy not in {None, ""}:
        tokens.extend(
            [
                "--checkpoint-controller-oracle-selection-policy",
                str(params.oracle_selection_policy),
            ]
        )
    optional_float_flags = [
        ("--checkpoint-controller-miss-abs-threshold", params.miss_abs_threshold),
        (
            "--checkpoint-controller-repair-retry-rescue-min-gain-ratio",
            params.repair_retry_rescue_min_gain_ratio,
        ),
        ("--checkpoint-controller-append-margin-abs", params.append_margin_abs),
        ("--checkpoint-controller-integrator-columnarity-threshold", params.integrator_columnarity_threshold),
        ("--checkpoint-controller-integrator-curvature-threshold", params.integrator_curvature_threshold),
        ("--checkpoint-controller-integrator-euler-fs-error-threshold", params.integrator_euler_fs_error_threshold),
        ("--checkpoint-controller-integrator-condition-max", params.integrator_condition_max),
        ("--checkpoint-controller-integrator-euler-min-time-fraction", params.integrator_euler_min_time_fraction),
        ("--checkpoint-controller-integrator-euler-site-span-max", params.integrator_euler_site_span_max),
        (
            "--checkpoint-controller-integrator-euler-primary-density-span-max",
            params.integrator_euler_primary_density_span_max,
        ),
        ("--checkpoint-controller-integrator-euler-energy-span-max", params.integrator_euler_energy_span_max),
        ("--checkpoint-controller-shortlist-fraction", params.shortlist_fraction),
        ("--checkpoint-controller-regularization-lambda", params.regularization_lambda),
        ("--checkpoint-controller-candidate-regularization-lambda", params.candidate_regularization_lambda),
        ("--checkpoint-controller-pinv-rcond", params.pinv_rcond),
        ("--checkpoint-controller-compile-penalty-weight", params.compile_penalty_weight),
        ("--checkpoint-controller-measurement-penalty-weight", params.measurement_penalty_weight),
        ("--checkpoint-controller-directional-penalty-weight", params.directional_penalty_weight),
        ("--checkpoint-controller-confirm-compress-fraction", params.confirm_compress_fraction),
        ("--checkpoint-controller-exact-forecast-tracking-fidelity-defect-weight", params.fidelity_weight),
        ("--checkpoint-controller-exact-forecast-tracking-doublon-error-weight", params.doublon_weight),
        ("--checkpoint-controller-exact-forecast-density-curvature-weight", params.density_curvature_weight),
        ("--checkpoint-controller-exact-forecast-density-excursion-under-weight", params.density_excursion_under_weight),
        ("--checkpoint-controller-exact-forecast-density-excursion-over-weight", params.density_excursion_over_weight),
        ("--checkpoint-controller-exact-forecast-density-sign-lag-weight", params.density_sign_lag_weight),
        ("--checkpoint-controller-exact-forecast-density-postcross-wrong-sign-weight", params.density_postcross_wrong_sign_weight),
        ("--checkpoint-controller-exact-forecast-drive-harmonic-weight", params.drive_harmonic_weight),
        ("--checkpoint-controller-exact-forecast-energy-slope-weight", params.energy_slope_weight),
        ("--checkpoint-controller-exact-forecast-energy-curvature-weight", params.energy_curvature_weight),
        ("--checkpoint-controller-exact-forecast-energy-excursion-under-weight", params.energy_excursion_under_weight),
        ("--checkpoint-controller-exact-forecast-energy-excursion-over-weight", params.energy_excursion_over_weight),
        ("--checkpoint-controller-exact-forecast-energy-excursion-rel-tolerance", params.energy_excursion_rel_tolerance),
        ("--checkpoint-controller-exact-v1-density-first-target-gain-floor", params.density_first_target_gain_floor),
        (
            "--checkpoint-controller-exact-v1-below-floor-probe-target-gain-floor",
            params.below_floor_probe_target_gain_floor,
        ),
        ("--checkpoint-controller-exact-v1-sign-lag-window-activation", params.sign_lag_window_activation),
        ("--checkpoint-controller-exact-v1-sign-lag-window-target-gain-floor", params.sign_lag_window_target_gain_floor),
        (
            "--checkpoint-controller-exact-v1-postcross-wrong-sign-activation",
            params.postcross_wrong_sign_activation,
        ),
        (
            "--checkpoint-controller-exact-v1-postcross-wrong-sign-target-gain-floor",
            params.postcross_wrong_sign_target_gain_floor,
        ),
        (
            "--checkpoint-controller-exact-v1-d-shape-turn-window-abs-activation",
            params.d_shape_turn_window_abs_activation,
        ),
        ("--checkpoint-controller-exact-forecast-fidelity-loss-tol", params.fidelity_loss_tol),
        ("--checkpoint-controller-exact-forecast-abs-energy-error-increase-tol", params.abs_energy_error_increase_tol),
        ("--checkpoint-controller-exact-forecast-total-occupation-error-increase-tol", params.total_occupation_error_increase_tol),
        ("--checkpoint-controller-progress-early-stop-site-error-mean-max", params.progress_early_stop_site_error_mean_max),
        (
            "--checkpoint-controller-progress-early-stop-primary-density-error-mean-max",
            params.progress_early_stop_primary_density_error_mean_max,
        ),
        ("--checkpoint-controller-progress-early-stop-energy-error-mean-max", params.progress_early_stop_energy_error_mean_max),
        ("--checkpoint-controller-progress-early-stop-site-span-max", params.progress_early_stop_site_span_max),
        (
            "--checkpoint-controller-progress-early-stop-primary-density-span-max",
            params.progress_early_stop_primary_density_span_max,
        ),
        ("--checkpoint-controller-progress-early-stop-energy-span-max", params.progress_early_stop_energy_span_max),
        ("--checkpoint-controller-prune-miss-threshold", params.prune_miss_threshold),
        ("--checkpoint-controller-prune-loss-threshold", params.prune_loss_threshold),
        ("--checkpoint-controller-prune-theta-block-tol", params.prune_theta_block_tol),
        ("--checkpoint-controller-prune-state-jump-l2-tol", params.prune_state_jump_l2_tol),
        (
            "--checkpoint-controller-prune-safe-miss-increase-tol",
            params.prune_safe_miss_increase_tol,
        ),
        ("--checkpoint-controller-prune-schur-monotonicity-tol", params.prune_schur_monotonicity_tol),
        ("--checkpoint-controller-prune-loss-norm-epsilon", params.prune_loss_norm_epsilon),
        ("--checkpoint-controller-prune-differential-miss-tol", params.prune_differential_miss_tol),
        ("--checkpoint-controller-prune-projection-trust-radius", params.prune_projection_trust_radius),
        ("--checkpoint-controller-prune-projection-regularization", params.prune_projection_regularization),
        ("--checkpoint-controller-prune-ray-distance-tol", params.prune_ray_distance_tol),
        ("--checkpoint-controller-prune-shadow-score-increase-tol", params.prune_shadow_score_increase_tol),
    ]
    for flag, value in optional_float_flags:
        if value is not None:
            tokens.extend([str(flag), str(float(value))])
    optional_int_flags = [
        ("--checkpoint-controller-shortlist-size", params.shortlist_size),
        ("--checkpoint-controller-repair-retry-max-attempts", params.repair_retry_max_attempts),
        ("--checkpoint-controller-active-window-size", params.active_window_size),
        ("--checkpoint-controller-max-probe-positions", params.max_probe_positions),
        ("--checkpoint-controller-confirm-compress-min-modes", params.confirm_compress_min_modes),
        ("--checkpoint-controller-confirm-compress-max-modes", params.confirm_compress_max_modes),
        ("--checkpoint-controller-progress-observable-window", params.progress_observable_window),
        ("--checkpoint-controller-integrator-euler-observable-window", params.integrator_euler_observable_window),
        ("--checkpoint-controller-progress-early-stop-min-checkpoint", params.progress_early_stop_min_checkpoint),
        ("--checkpoint-controller-prune-schur-ladder-local-radius", params.prune_schur_ladder_local_radius),
        ("--checkpoint-controller-prune-projection-rounds", params.prune_projection_rounds),
        ("--checkpoint-controller-prune-projection-max-active-runtime", params.prune_projection_max_active_runtime),
        ("--checkpoint-controller-prune-shadow-horizon-steps", params.prune_shadow_horizon_steps),
        ("--checkpoint-controller-prune-persistence-window", params.prune_persistence_window),
        ("--checkpoint-controller-prune-persistence-required", params.prune_persistence_required),
        ("--checkpoint-controller-prune-appended-origin-grace-steps", params.prune_appended_origin_grace_steps),
        ("--checkpoint-controller-prune-initial-scaffold-grace-steps", params.prune_initial_scaffold_grace_steps),
        (
            "--checkpoint-controller-exact-v1-d-shape-outside-turn-below-floor-probe-stall-threshold",
            params.d_shape_outside_turn_below_floor_probe_stall_threshold,
        ),
    ]
    for flag, value in optional_int_flags:
        if value is not None:
            tokens.extend([str(flag), str(int(value))])
    explicit_high_miss_policy = (
        None
        if params.high_miss_no_admit_policy in {None, ""}
        else normalize_high_miss_no_admit_policy(params.high_miss_no_admit_policy)
    )
    optional_str_flags = [
        ("--checkpoint-controller-high-miss-no-admit-policy", explicit_high_miss_policy),
        ("--checkpoint-controller-repair-retry-escalation-mode", params.repair_retry_escalation_mode),
        ("--checkpoint-controller-repair-retry-admission-policy", params.repair_retry_admission_policy),
        ("--checkpoint-controller-repair-retry-rescue-attempt", params.repair_retry_rescue_attempt),
        ("--checkpoint-controller-exact-v1-repeat-reopen-mode", params.repeat_reopen_mode),
        ("--checkpoint-controller-exact-forecast-guardrail-mode", params.guardrail_mode),
        ("--checkpoint-controller-prune-mode", params.prune_mode),
        ("--checkpoint-controller-prune-projection-mode", params.prune_projection_mode),
        ("--checkpoint-controller-exact-forecast-baseline-proposal-mode", params.baseline_proposal_mode),
        (
            "--checkpoint-controller-exact-forecast-primary-density-target-mode",
            params.primary_density_target_mode,
        ),
        ("--checkpoint-controller-confirm-score-mode", params.confirm_score_mode),
    ]
    for flag, value in optional_str_flags:
        if value not in {None, ""}:
            tokens.extend([str(flag), str(value)])
    if params.miss_persistence_spec not in {None, ""}:
        raw_window, raw_count = str(params.miss_persistence_spec).split(":", 1)
        tokens.extend(
            [
                "--checkpoint-controller-miss-persistence-window",
                str(int(raw_window)),
                "--checkpoint-controller-miss-persistence-count",
                str(int(raw_count)),
            ]
        )
    if bool(params.include_tangent_secant_proposal):
        tokens.append("--checkpoint-controller-exact-forecast-include-tangent-secant-proposal")
    else:
        tokens.append("--no-checkpoint-controller-exact-forecast-include-tangent-secant-proposal")
    if bool(params.postcross_compare_diag):
        tokens.append("--checkpoint-controller-exact-v1-postcross-compare-diag")
    if bool(params.below_floor_energy_safe_d_shape_escape):
        tokens.append("--checkpoint-controller-exact-v1-below-floor-energy-safe-d-shape-escape")
    if bool(params.below_floor_energy_safe_turn_escape):
        tokens.append("--checkpoint-controller-exact-v1-below-floor-energy-safe-turn-escape")
    if params.d_shape_pre_turn_shadow_bridge is True:
        tokens.append("--checkpoint-controller-exact-v1-d-shape-pre-turn-shadow-bridge")
    elif params.d_shape_pre_turn_shadow_bridge is False:
        tokens.append("--no-checkpoint-controller-exact-v1-d-shape-pre-turn-shadow-bridge")
    if params.single_surface_commit_law is True:
        tokens.append("--checkpoint-controller-exact-v1-single-surface-commit-law")
    elif params.single_surface_commit_law is False:
        tokens.append("--no-checkpoint-controller-exact-v1-single-surface-commit-law")
    if params.prune_appended_origin_bias_enabled is True:
        tokens.append("--checkpoint-controller-prune-appended-origin-bias-enabled")
    elif params.prune_appended_origin_bias_enabled is False:
        tokens.append("--no-checkpoint-controller-prune-appended-origin-bias-enabled")
    if params.prune_high_miss_differential_enabled is True:
        tokens.append("--checkpoint-controller-prune-high-miss-differential-enabled")
    elif params.prune_high_miss_differential_enabled is False:
        tokens.append("--no-checkpoint-controller-prune-high-miss-differential-enabled")
    if params.prune_shadow_enabled is True:
        tokens.append("--checkpoint-controller-prune-shadow-enabled")
    elif params.prune_shadow_enabled is False:
        tokens.append("--no-checkpoint-controller-prune-shadow-enabled")
    tokens.append(f"--checkpoint-controller-exact-forecast-baseline-blend-weights={str(blend_weights)}")
    if gain_scales:
        tokens.append(
            f"--checkpoint-controller-exact-forecast-baseline-gain-scales={str(gain_scales)}"
        )
    if progress_json is not None:
        tokens.extend(["--progress-json", str(progress_json)])
    if partial_payload_json is not None:
        tokens.extend(["--partial-payload-json", str(partial_payload_json)])
    if bool(base_cfg.lock_fixed_manifold):
        tokens.append("--lock-fixed-manifold")
    if bool(base_cfg.allow_repeats):
        tokens.append("--allow-repeats")
    if bool(base_cfg.enable_drive) and bool(base_cfg.disable_drive):
        raise ValueError("BaseRunConfig cannot set both enable_drive and disable_drive.")
    if bool(base_cfg.disable_drive):
        tokens.append("--disable-drive")
    elif bool(base_cfg.enable_drive):
        tokens.extend(
            [
                "--enable-drive",
                "--drive-A",
                str(float(base_cfg.drive_A)),
                "--drive-omega",
                str(float(base_cfg.drive_omega)),
                "--drive-tbar",
                str(float(base_cfg.drive_tbar)),
                "--drive-phi",
                str(float(base_cfg.drive_phi)),
                "--drive-pattern",
                str(base_cfg.drive_pattern),
                "--drive-custom-weights",
                str(base_cfg.drive_custom_weights),
                "--drive-time-sampling",
                str(base_cfg.drive_time_sampling),
                "--drive-t0",
                str(float(base_cfg.drive_t0)),
            ]
        )
        if bool(base_cfg.drive_include_identity):
            tokens.append("--drive-include-identity")
    return tokens


def _controller_settings_payload_from_trial(
    *,
    params: TrialParams,
    base_cfg: BaseRunConfig,
    route_ctx: OptunaRouteContext | None,
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in asdict(params).items()
        if value is not None and value != ""
    }
    payload.update(
        {
            "checkpoint_controller_mode": (
                "observable_v1"
                if _is_strict_qpu_faithful_profile(base_cfg.study_profile)
                else "exact_v1"
            ),
            "checkpoint_controller_exact_input_mode": (
                "off"
                if _is_strict_qpu_faithful_profile(base_cfg.study_profile)
                else "benchmark_exact"
            ),
            "lock_fixed_manifold": bool(base_cfg.lock_fixed_manifold),
            "allow_repeats": bool(base_cfg.allow_repeats),
        }
    )
    return {str(key): value for key, value in payload.items() if value is not None}


def _optuna_class_tuning_provenance(
    *,
    base_cfg: BaseRunConfig,
    settings_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not base_cfg.tuning_class:
        return None
    case = DynamicsBenchmarkCase(
        case_id="optuna_class_tuning_candidate",
        family="optuna_class_tuning",
        table_class="optuna_class_tuning",
        artifact_json="",
        tuning_class=str(base_cfg.tuning_class),
    )
    return build_dynamics_tuning_provenance(
        case=case,
        algorithm_id="dyn_controller_full",
        settings_kind="controller",
        settings_payload=dict(settings_payload or {}),
        settings_source=str(base_cfg.class_settings_source),
        locked=False,
    )


def _finite_metric(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(str(key), None)
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if math.isfinite(numeric) else None


def _finite_1d_array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size <= 0 or not bool(np.all(np.isfinite(arr))):
        return None
    return arr


def _finite_2d_array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
        return None
    if not bool(np.all(np.isfinite(arr))):
        return None
    return arr


def _primary_observable_mode(
    *,
    payload: Mapping[str, Any],
    analysis_metadata: Mapping[str, Any],
    trajectory: Sequence[Mapping[str, Any]],
) -> str:
    for row in trajectory:
        raw = row.get("primary_density_mode", None)
        if raw not in {None, ""}:
            return str(raw)
    for container_key in ("summary", "reference", "route_config"):
        container = payload.get(container_key, {})
        if not isinstance(container, Mapping):
            continue
        for key in (
            "primary_density_mode",
            "final_primary_density_mode",
            "observable_telemetry_primary_density_mode",
        ):
            raw = container.get(key, None)
            if raw not in {None, ""}:
                return str(raw)
    raw_mode = analysis_metadata.get("primary_density_mode", None)
    if raw_mode not in {None, ""}:
        return str(raw_mode)
    if analysis_metadata.get("pair_difference", None) is not None:
        return "pair_difference"
    return "primary_density"


def _primary_observable_series_from_rows(
    trajectory: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    times: list[float] = []
    ctrl: list[float] = []
    exact: list[float] = []
    if not trajectory:
        return None
    for row in trajectory:
        if "primary_density" not in row or "primary_density_exact" not in row:
            return None
        time_raw = row.get("time", row.get("physical_time", None))
        values = (time_raw, row.get("primary_density"), row.get("primary_density_exact"))
        try:
            time_value, ctrl_value, exact_value = (float(value) for value in values)
        except Exception:
            return None
        if not all(math.isfinite(value) for value in (time_value, ctrl_value, exact_value)):
            return None
        times.append(float(time_value))
        ctrl.append(float(ctrl_value))
        exact.append(float(exact_value))
    if not times:
        return None
    return (
        np.asarray(times, dtype=float),
        np.asarray(ctrl, dtype=float),
        np.asarray(exact, dtype=float),
    )


def _primary_observable_series_from_raw_traces(
    raw: Mapping[str, Any],
    *,
    ctrl_key: str,
    exact_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    times = _finite_1d_array_or_none(raw.get("times", None))
    ctrl = _finite_1d_array_or_none(raw.get(ctrl_key, None))
    exact = _finite_1d_array_or_none(raw.get(exact_key, None))
    if times is None or ctrl is None or exact is None:
        return None
    if times.size != ctrl.size or times.size != exact.size:
        return None
    return times, ctrl, exact


def _staggered_from_site_matrix(site_occupations: np.ndarray) -> np.ndarray:
    site_occ = np.asarray(site_occupations, dtype=float)
    signs = np.asarray(
        [1.0 if (idx % 2 == 0) else -1.0 for idx in range(int(site_occ.shape[1]))],
        dtype=float,
    )
    return np.asarray((site_occ @ signs) / float(site_occ.shape[1]), dtype=float)


def _primary_observable_series(
    *,
    payload: Mapping[str, Any],
    analysis: Mapping[str, Any],
    analysis_metadata: Mapping[str, Any],
    trajectory: Sequence[Mapping[str, Any]],
    pair: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str] | None:
    mode = _primary_observable_mode(
        payload=payload,
        analysis_metadata=analysis_metadata,
        trajectory=trajectory,
    )
    direct = _primary_observable_series_from_rows(trajectory)
    if direct is not None:
        times, ctrl, exact = direct
        return times, ctrl, exact, "primary_density", "trajectory.primary_density"

    raw = analysis.get("raw_traces", {}) if isinstance(analysis.get("raw_traces", {}), Mapping) else {}
    pair_traces = _primary_observable_series_from_raw_traces(
        raw,
        ctrl_key="pair_difference",
        exact_key="pair_difference_exact",
    )
    if pair_traces is not None:
        times, ctrl, exact = pair_traces
        return times, ctrl, exact, "pair_difference", "analysis.raw_traces"

    times = _finite_1d_array_or_none(raw.get("times", None))
    site_occ = _finite_2d_array_or_none(raw.get("site_occupations", None))
    site_occ_exact = _finite_2d_array_or_none(raw.get("site_occupations_exact", None))
    if times is None or site_occ is None or site_occ_exact is None:
        return None
    if site_occ.shape != site_occ_exact.shape or int(site_occ.shape[0]) != int(times.size):
        return None
    if pair is not None:
        left, right = int(pair[0]), int(pair[1])
        if max(left, right) < int(site_occ.shape[1]):
            ctrl = np.asarray(site_occ[:, left] - site_occ[:, right], dtype=float)
            exact = np.asarray(site_occ_exact[:, left] - site_occ_exact[:, right], dtype=float)
            return times, ctrl, exact, "pair_difference", "site_occupations.pair_fallback"
    if str(mode).strip().lower() == "staggered":
        return (
            times,
            _staggered_from_site_matrix(site_occ),
            _staggered_from_site_matrix(site_occ_exact),
            "staggered",
            "site_occupations.staggered_fallback",
        )
    return None


def _finite_or_nan(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float("nan")


def _primary_observable_metric_bundle(
    *,
    times: np.ndarray,
    ctrl: np.ndarray,
    exact: np.ndarray,
    base_cfg: BaseRunConfig,
    drive_omega: float | None,
) -> dict[str, Any]:
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    ctrl_arr = np.asarray(ctrl, dtype=float).reshape(-1)
    exact_arr = np.asarray(exact, dtype=float).reshape(-1)
    if (
        times_arr.size == 0
        or times_arr.size != ctrl_arr.size
        or times_arr.size != exact_arr.size
        or not bool(np.all(np.isfinite(times_arr)))
        or not bool(np.all(np.isfinite(ctrl_arr)))
        or not bool(np.all(np.isfinite(exact_arr)))
    ):
        return {
            "primary_observable_num_samples": 0,
            "primary_observable_mae": float("nan"),
            "primary_observable_span_exact": float("nan"),
            "primary_observable_mae_over_exact_span": float("nan"),
            "primary_observable_corr": float("nan"),
            "primary_observable_epsilon_osc": float("nan"),
            "primary_observable_dominant_peak_abs_omega_error": float("nan"),
            "primary_observable_spectrum_available": False,
        }
    err = np.asarray(ctrl_arr - exact_arr, dtype=float)
    mae = float(np.mean(np.abs(err)))
    span_exact = float(np.ptp(exact_arr))
    corr = _finite_or_nan(_corrcoef(ctrl_arr, exact_arr))
    epsilon = _finite_or_nan(_eps_osc(ctrl_arr, exact_arr))
    peak_error = float("nan")
    spectrum_available = False
    if int(times_arr.size) >= 3:
        try:
            ctrl_spec = compute_one_sided_amplitude_spectrum(
                times_arr,
                ctrl_arr,
                detrend=str(base_cfg.spectra_detrend),
                window=str(base_cfg.spectra_window),
                max_peaks=int(base_cfg.max_peaks),
                drive_omega=drive_omega,
                max_harmonic=int(base_cfg.max_harmonic),
            )
            exact_spec = compute_one_sided_amplitude_spectrum(
                times_arr,
                exact_arr,
                detrend=str(base_cfg.spectra_detrend),
                window=str(base_cfg.spectra_window),
                max_peaks=int(base_cfg.max_peaks),
                drive_omega=drive_omega,
                max_harmonic=int(base_cfg.max_harmonic),
            )
            if ctrl_spec.top_peaks and exact_spec.top_peaks:
                peak_error = float(
                    abs(
                        float(ctrl_spec.top_peaks[0]["omega"])
                        - float(exact_spec.top_peaks[0]["omega"])
                    )
                )
                spectrum_available = math.isfinite(float(peak_error))
        except Exception:
            peak_error = float("nan")
            spectrum_available = False
    return {
        "primary_observable_num_samples": int(times_arr.size),
        "primary_observable_mae": float(mae),
        "primary_observable_span_exact": float(span_exact),
        "primary_observable_mae_over_exact_span": float(mae / max(float(span_exact), 1.0e-12)),
        "primary_observable_corr": corr,
        "primary_observable_epsilon_osc": epsilon,
        "primary_observable_dominant_peak_abs_omega_error": _finite_or_nan(peak_error),
        "primary_observable_spectrum_available": bool(spectrum_available),
    }


def _prefixed_metric_bundle(prefix: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _energy_tracking_metric_bundle(
    *,
    ctrl: np.ndarray,
    exact: np.ndarray,
) -> dict[str, Any]:
    ctrl_arr = np.asarray(ctrl, dtype=float).reshape(-1)
    exact_arr = np.asarray(exact, dtype=float).reshape(-1)
    if ctrl_arr.size != exact_arr.size or ctrl_arr.size < 2:
        return {
            "mean_signed_energy_total_error": float("nan"),
            "mean_signed_energy_total_error_abs": float("nan"),
            "max_abs_energy_total_error": float("nan"),
            "energy_total_excursion_under_response_mean": float("nan"),
            "energy_total_excursion_under_response_max": float("nan"),
        }
    finite = np.isfinite(ctrl_arr) & np.isfinite(exact_arr)
    if int(np.count_nonzero(finite)) < 2:
        return {
            "mean_signed_energy_total_error": float("nan"),
            "mean_signed_energy_total_error_abs": float("nan"),
            "max_abs_energy_total_error": float("nan"),
            "energy_total_excursion_under_response_mean": float("nan"),
            "energy_total_excursion_under_response_max": float("nan"),
        }
    ctrl_arr = ctrl_arr[finite]
    exact_arr = exact_arr[finite]
    signed_err = np.asarray(ctrl_arr - exact_arr, dtype=float)
    exact_delta = np.asarray(exact_arr - exact_arr[0], dtype=float)
    ctrl_delta = np.asarray(ctrl_arr - ctrl_arr[0], dtype=float)
    direction = np.sign(exact_delta)
    directed_ctrl_excursion = direction * ctrl_delta
    exact_excursion = np.abs(exact_delta)
    under_response = np.maximum(0.0, exact_excursion - directed_ctrl_excursion)
    return {
        "mean_signed_energy_total_error": float(np.mean(signed_err)),
        "mean_signed_energy_total_error_abs": float(abs(float(np.mean(signed_err)))),
        "max_abs_energy_total_error": float(np.max(np.abs(signed_err))),
        "energy_total_excursion_under_response_mean": float(np.mean(under_response)),
        "energy_total_excursion_under_response_max": float(np.max(under_response)),
    }


def _generic_exact_v1_signal_metric(
    metrics: Mapping[str, Any],
    name: str,
    *,
    objective_window: bool,
) -> float | None:
    objective_keys = {
        "mae_over_exact_span": (
            "objective_primary_observable_mae_over_exact_span",
            "objective_pair_mae_over_exact_span",
        ),
        "corr": ("objective_primary_observable_corr", "objective_pair_corr"),
        "epsilon_osc": (
            "objective_primary_observable_epsilon_osc",
            "objective_epsilon_osc_pair",
        ),
        "dominant_peak_abs_omega_error": (
            "objective_primary_observable_dominant_peak_abs_omega_error",
            "objective_dominant_peak_abs_omega_error",
        ),
    }
    base_keys = {
        "mae_over_exact_span": (
            "primary_observable_mae_over_exact_span",
            "pair_mae_over_exact_span",
        ),
        "corr": ("primary_observable_corr", "pair_corr"),
        "epsilon_osc": ("primary_observable_epsilon_osc", "epsilon_osc_pair"),
        "dominant_peak_abs_omega_error": (
            "primary_observable_dominant_peak_abs_omega_error",
            "dominant_peak_abs_omega_error",
        ),
    }
    keys: list[str] = []
    if bool(objective_window):
        keys.extend(objective_keys.get(str(name), ()))
    keys.extend(base_keys.get(str(name), ()))
    for key in keys:
        value = _finite_metric(metrics, key)
        if value is not None:
            return float(value)
    return None


def _strict_qpu_faithful_metrics_from_payload(
    *,
    payload: Mapping[str, Any],
    base_cfg: BaseRunConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_trajectory = [dict(row) for row in payload.get("trajectory", []) if isinstance(row, Mapping)]
    ledger = [dict(row) for row in payload.get("ledger", []) if isinstance(row, Mapping)]
    decision_rows = ledger if ledger else raw_trajectory
    trajectory = physical_trajectory_rows(raw_trajectory)
    summary = dict(payload.get("summary", {})) if isinstance(payload.get("summary", {}), Mapping) else {}
    reference = dict(payload.get("reference", {})) if isinstance(payload.get("reference", {}), Mapping) else {}
    route_config = (
        dict(payload.get("route_config", {}))
        if isinstance(payload.get("route_config", {}), Mapping)
        else {}
    )
    strict_flags = _strict_payload_flags(payload)
    analysis_status = _strict_analysis_status(strict_flags)
    recoverability_reference = _strict_recoverability_reference(strict_flags)
    repair_counts = trajectory_repair_counts(raw_trajectory)
    soft_fallback_counts = high_miss_no_admit_soft_fallback_counts(decision_rows)
    high_miss_no_admit_counts = high_miss_no_admit_diagnostic_counts(decision_rows)
    early_stop_raw = summary.get("early_stop_reason", payload.get("early_stop_reason"))
    early_stop_reason = None if str(early_stop_raw).strip().lower() in {"", "none", "null"} else early_stop_raw
    time_vals = [_as_finite_float_or_nan(row.get("time")) for row in trajectory]
    fidelity_vals = [_as_finite_float_or_nan(row.get("fidelity_exact")) for row in trajectory]
    energy_err_vals = [
        _as_finite_float_or_nan(row.get("abs_energy_total_error")) for row in trajectory
    ]
    site_err_vals = [
        _as_finite_float_or_nan(row.get("site_occupations_abs_error_max")) for row in trajectory
    ]
    final_time = _safe_last(time_vals)
    full_horizon_fields = full_horizon_completion_fields(
        raw_trajectory,
        expected_t_final=float(base_cfg.t_final),
        expected_row_count=int(base_cfg.num_times),
        early_stop_reason=early_stop_reason,
        stable_early_stop_accepted=_summary_declares_successful_stable_early_stop(summary),
    )
    rho_miss_vals = [_as_finite_float_or_nan(row.get("rho_miss")) for row in trajectory]
    rho_real_vals = [_as_finite_float_or_nan(row.get("rho_real")) for row in trajectory]
    rho_num_vals = [_as_finite_float_or_nan(row.get("rho_num")) for row in trajectory]
    displacement_vals = [
        _as_finite_float_or_nan(row.get("predicted_displacement")) for row in trajectory
    ]
    runtime_vals = [
        _as_finite_float_or_nan(row.get("runtime_parameter_count")) for row in trajectory
    ]
    improvement_abs_vals = [
        _as_finite_float_or_nan(row.get("selected_noisy_improvement_abs")) for row in trajectory
    ]
    improvement_ratio_vals = [
        _as_finite_float_or_nan(row.get("selected_noisy_improvement_ratio")) for row in trajectory
    ]
    proposed_append_count = int(
        sum(1 for row in decision_rows if str(row.get("proposed_action_kind", "")) == "append_candidate")
    )
    proposed_prune_count = int(
        sum(1 for row in decision_rows if str(row.get("proposed_action_kind", "")) == "prune_coordinate")
    )
    prune_candidate_lists = [
        row.get("prune_candidates", [])
        for row in decision_rows
        if isinstance(row.get("prune_candidates", []), list)
    ]
    prune_candidate_count = int(sum(len(items) for items in prune_candidate_lists))
    prune_candidate_checkpoint_count = int(sum(1 for items in prune_candidate_lists if items))
    # Older controller payloads can record prune proposals/accepts without
    # materializing a `prune_candidates` list in each decision row. Treat those
    # rows as prune opportunities for tuning/diagnostics rather than allowing
    # zero-valued summary defaults to hide active prune behavior.
    prune_candidate_count_lower_bound = int(max(prune_candidate_count, proposed_prune_count))
    prune_candidate_checkpoint_count_lower_bound = int(
        max(prune_candidate_checkpoint_count, proposed_prune_count)
    )
    prune_shadow_reject_count = int(
        sum(
            1
            for items in prune_candidate_lists
            for item in items
            if isinstance(item, Mapping)
            and item.get("prune_rejection_reason", item.get("prune_no_harm_reason", None))
            not in {None, ""}
        )
    )
    degraded_rows = [
        row for row in decision_rows if row.get("degraded_reason", None) not in {None, ""}
    ]
    degraded_checkpoint_count = max(
        int(summary.get("degraded_checkpoints", 0)),
        int(len(degraded_rows)),
    )
    trajectory_count = int(len(trajectory))
    strict_contract = strict_qpu_faithful_decision_contract(
        summary=summary,
        reference=reference,
        decision_rows=decision_rows,
    )
    exact_decision_checkpoints = int(strict_contract.get("exact_decision_checkpoints", 0))
    oracle_decision_checkpoints = int(strict_contract.get("oracle_decision_checkpoints", 0))
    row_backend_purity_passed = bool(strict_contract.get("passed", False))
    row_noise_modes = [
        str(row.get("decision_noise_mode"))
        for row in decision_rows
        if row.get("decision_noise_mode", None) not in {None, ""}
    ]
    non_ideal_decision_noise_count = int(
        sum(1 for mode in row_noise_modes if str(mode).strip().lower() != "ideal")
    )
    strict_fail_closed = bool(summary.get("strict_fail_closed", False))
    qpu_expected = bool(summary.get("qpu_faithful_decisions_expected", True))
    summary_qpu_passed = bool(summary.get("qpu_faithful_decisions_passed", True))
    qpu_passed = bool(
        summary_qpu_passed
        and (not strict_fail_closed)
        and exact_decision_checkpoints == 0
        and row_backend_purity_passed
        and non_ideal_decision_noise_count == 0
    )
    reference_mode = str(summary.get("reference_mode", reference.get("reference_mode", "off")))
    reference_enabled = bool(summary.get("reference_enabled", reference.get("reference_enabled", False)))
    decision_noise_mode = summary.get("decision_noise_mode")
    if row_noise_modes:
        unique_noise_modes = sorted(set(row_noise_modes))
        decision_noise_mode = (
            unique_noise_modes[0] if len(unique_noise_modes) == 1 else "mixed"
        )
    mean_rho_miss = _safe_mean(rho_miss_vals)
    max_rho_miss = _safe_max(rho_miss_vals)
    final_rho_miss = _safe_last(rho_miss_vals)
    strict_degraded_fraction = float(degraded_checkpoint_count / max(trajectory_count, 1))
    high_miss_no_admit_soft_fallback_count = int(
        summary.get(
            "high_miss_no_admit_soft_fallback_count",
            soft_fallback_counts["high_miss_no_admit_soft_fallback_count"],
        )
    )
    high_miss_no_admit_fraction = float(
        summary.get(
            "high_miss_no_admit_fraction",
            high_miss_no_admit_counts["high_miss_no_admit_fraction"],
        )
    )
    soft_fallback_fraction = float(high_miss_no_admit_soft_fallback_count / max(trajectory_count, 1))
    finite_components = [
        value
        for value in (
            mean_rho_miss,
            0.25 * max_rho_miss if math.isfinite(max_rho_miss) else float("nan"),
            0.10 * final_rho_miss if math.isfinite(final_rho_miss) else float("nan"),
            0.50 * high_miss_no_admit_fraction,
            0.25 * soft_fallback_fraction,
            10.0 * strict_degraded_fraction,
        )
        if math.isfinite(float(value))
    ]
    strict_measured_degradation_score = (
        float(sum(finite_components)) if finite_components else float("nan")
    )
    strict_tail_metrics = _strict_observable_tail_metrics(
        trajectory=trajectory,
        base_cfg=base_cfg,
    )
    analysis: dict[str, Any] = {
        "metadata": {
            "analysis_status": analysis_status,
            "analysis_skipped": True,
            "analysis_skip_reason": (
                "strict QPU-faithful profile scores measured telemetry only; exact "
                "trajectory/spectra analysis is intentionally disabled"
            ),
            "recoverability_reference": dict(recoverability_reference),
        },
        "spectra": {},
        "raw_traces": {},
    }
    metrics: dict[str, Any] = {
        "strict_qpu_faithful": bool(strict_flags["strict_qpu_faithful"]),
        "strict_qpu_hh": bool(strict_flags["strict_qpu_hh"]),
        "strict_qpu_family": str(strict_flags["strict_qpu_family"]),
        "decision_path_kind": summary.get("decision_path_kind", "strict_qpu_faithful_observable_v1"),
        "trajectory_row_count": int(len(raw_trajectory)),
        **repair_counts,
        **full_horizon_fields,
        "expected_trajectory_row_count": int(base_cfg.num_times),
        "final_time": float(final_time),
        "expected_t_final": float(base_cfg.t_final),
        "trajectory_reached_final_time": bool(full_horizon_fields["full_horizon_reached_final_time"]),
        "trajectory_reached_expected_rows": bool(
            full_horizon_fields["full_horizon_reached_expected_rows"]
        ),
        "analysis_status": analysis_status,
        "analysis_error": None,
        "summary_status": summary.get("status", payload.get("status")),
        "early_stop_reason": early_stop_reason,
        "reference_mode": reference_mode,
        "reference_enabled": bool(reference_enabled),
        "decision_noise_mode": None if decision_noise_mode in {None, ""} else str(decision_noise_mode),
        "qpu_faithful_decisions_expected": bool(qpu_expected),
        "qpu_faithful_decisions_passed": bool(qpu_passed),
        "strict_decision_contract_passed": bool(strict_contract.get("passed", False)),
        "strict_decision_contract_violations": list(strict_contract.get("violations", [])),
        "strict_fail_closed": bool(strict_fail_closed),
        "strict_fail_closed_reason": summary.get("strict_fail_closed_reason"),
        "exact_decision_checkpoints": int(exact_decision_checkpoints),
        "oracle_decision_checkpoints": int(oracle_decision_checkpoints),
        "decision_backend_counts": _string_histogram(
            [row.get("decision_backend", None) for row in decision_rows]
        ),
        "decision_noise_mode_counts": _string_histogram(row_noise_modes),
        "non_ideal_decision_noise_count": int(non_ideal_decision_noise_count),
        "append_count": int(summary.get("append_count", 0)),
        "prune_count": int(summary.get("prune_count", 0)),
        "proposed_append_count": int(summary.get("proposed_append_count", proposed_append_count)),
        "proposed_prune_count": int(summary.get("proposed_prune_count", proposed_prune_count)),
        "append_opportunity_count": int(
            max(
                int(summary.get("high_miss_count", high_miss_no_admit_counts["high_miss_count"])),
                int(summary.get("proposed_append_count", proposed_append_count)),
            )
        ),
        "prune_candidate_count": int(
            max(
                int(summary.get("prune_candidate_count", 0)),
                prune_candidate_count_lower_bound,
                int(summary.get("prune_count", 0)),
            )
        ),
        "prune_candidate_checkpoint_count": int(
            max(
                int(summary.get("prune_candidate_checkpoint_count", 0)),
                prune_candidate_checkpoint_count_lower_bound,
                1 if int(summary.get("prune_count", 0)) > 0 else 0,
            )
        ),
        "prune_opportunity_count": int(
            max(
                int(summary.get("prune_opportunity_count", 0)),
                prune_candidate_checkpoint_count_lower_bound,
                1 if int(summary.get("prune_count", 0)) > 0 else 0,
            )
        ),
        "prune_shadow_reject_count": int(
            summary.get("prune_shadow_reject_count", prune_shadow_reject_count)
        ),
        "integrator_euler_count": int(summary.get("integrator_euler_count", 0)),
        "integrator_rk4_count": int(summary.get("integrator_rk4_count", 0)),
        "final_runtime_parameter_count": int(
            summary.get(
                "final_runtime_parameter_count",
                int(_safe_last(runtime_vals)) if math.isfinite(_safe_last(runtime_vals)) else 0,
            )
        ),
        "mean_runtime_parameter_count": _safe_mean(runtime_vals),
        "high_miss_count": int(
            summary.get("high_miss_count", high_miss_no_admit_counts["high_miss_count"])
        ),
        "high_miss_fraction": float(
            summary.get("high_miss_fraction", high_miss_no_admit_counts["high_miss_fraction"])
        ),
        "high_miss_no_admit_count": int(
            summary.get(
                "high_miss_no_admit_count",
                high_miss_no_admit_counts["high_miss_no_admit_count"],
            )
        ),
        "high_miss_no_admit_fraction": high_miss_no_admit_fraction,
        "high_miss_no_admit_reason_counts": dict(
            summary.get(
                "high_miss_no_admit_reason_counts",
                high_miss_no_admit_counts["high_miss_no_admit_reason_counts"],
            )
        ),
        "high_miss_no_admit_resolution_counts": dict(
            summary.get(
                "high_miss_no_admit_resolution_counts",
                high_miss_no_admit_counts["high_miss_no_admit_resolution_counts"],
            )
        ),
        "high_miss_no_admit_soft_fallback_count": int(high_miss_no_admit_soft_fallback_count),
        "high_miss_no_admit_soft_fallback_fraction": float(soft_fallback_fraction),
        "soft_fallback_reason_counts": dict(
            summary.get(
                "high_miss_no_admit_soft_fallback_reason_counts",
                soft_fallback_counts["high_miss_no_admit_soft_fallback_reason_counts"],
            )
        ),
        "mean_rho_miss": float(mean_rho_miss),
        "max_rho_miss": float(max_rho_miss),
        "final_rho_miss": float(final_rho_miss),
        "mean_rho_real": _safe_mean(rho_real_vals),
        "mean_rho_num": _safe_mean(rho_num_vals),
        "max_predicted_displacement": _safe_max(displacement_vals),
        "mean_predicted_displacement": _safe_mean(displacement_vals),
        "strict_degraded_checkpoint_count": int(degraded_checkpoint_count),
        "strict_degraded_fraction": float(strict_degraded_fraction),
        "strict_degraded_reason_counts": _string_histogram(
            [row.get("degraded_reason", None) for row in degraded_rows]
        ),
        "selected_noisy_improvement_abs_mean": _safe_mean(improvement_abs_vals),
        "selected_noisy_improvement_abs_max": _safe_max(improvement_abs_vals),
        "selected_noisy_improvement_ratio_mean": _safe_mean(improvement_ratio_vals),
        "strict_measured_degradation_score": float(strict_measured_degradation_score),
        # Diagnostic exact-reference errors are emitted for post-run review and
        # aggregate PDFs only.  They are intentionally not part of
        # _objective_value(...) for strict/QPU-faithful profiles, so Optuna's
        # online feedback remains measurement-compatible.
        "diagnostic_min_fidelity_exact": _safe_min(fidelity_vals),
        "diagnostic_mean_fidelity_exact": _safe_mean(fidelity_vals),
        "diagnostic_mean_abs_energy_total_error": _safe_mean(energy_err_vals),
        "diagnostic_final_abs_energy_total_error": _safe_last(energy_err_vals),
        "diagnostic_max_abs_energy_total_error": _safe_max(energy_err_vals),
        "diagnostic_mean_abs_site_occupations_error": _safe_mean(site_err_vals),
        "diagnostic_max_abs_site_occupations_error": _safe_max(site_err_vals),
        **strict_tail_metrics,
        "recoverability_reference": dict(recoverability_reference),
    }
    return analysis, metrics


def _trial_metrics_from_payload(
    *,
    payload: Mapping[str, Any],
    output_json: Path,
    base_cfg: BaseRunConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    strict_flags = _strict_payload_flags(payload)
    if _is_strict_qpu_faithful_profile(base_cfg.study_profile) or bool(
        strict_flags["strict_qpu_faithful"]
    ):
        return _strict_qpu_faithful_metrics_from_payload(payload=payload, base_cfg=base_cfg)
    loaded = load_trajectory_payload(output_json, time_key="time")
    pair = _resolved_pair(str(base_cfg.pair), num_sites=int(loaded.site_occupations.shape[1]))
    analysis = analyze_payload(
        loaded,
        pair=pair,
        detrend=str(base_cfg.spectra_detrend),
        window=str(base_cfg.spectra_window),
        max_peaks=int(base_cfg.max_peaks),
        max_harmonic=int(base_cfg.max_harmonic),
        allow_short=True,
    )
    analysis_metadata = (
        analysis.get("metadata", {}) if isinstance(analysis.get("metadata", {}), Mapping) else {}
    )
    parameter_manifest = (
        analysis_metadata.get("parameter_manifest", {})
        if isinstance(analysis_metadata.get("parameter_manifest", {}), Mapping)
        else {}
    )

    raw_trajectory = [dict(row) for row in payload.get("trajectory", []) if isinstance(row, Mapping)]
    repair_counts = trajectory_repair_counts(raw_trajectory)
    soft_fallback_counts = high_miss_no_admit_soft_fallback_counts(raw_trajectory)
    high_miss_no_admit_counts = high_miss_no_admit_diagnostic_counts(raw_trajectory)
    trajectory = physical_trajectory_rows(raw_trajectory)
    summary = dict(payload.get("summary", {}))
    fidelity_vals = [_as_finite_float_or_nan(row.get("fidelity_exact")) for row in trajectory]
    energy_total_vals = [
        _as_finite_float_or_nan(row.get("energy_total", row.get("energy_total_controller")))
        for row in trajectory
    ]
    energy_total_exact_vals = [
        _as_finite_float_or_nan(row.get("energy_total_exact")) for row in trajectory
    ]
    energy_err_vals = [_as_finite_float_or_nan(row.get("abs_energy_total_error")) for row in trajectory]
    site_err_vals = [_as_finite_float_or_nan(row.get("site_occupations_abs_error_max")) for row in trajectory]
    runtime_vals = [_as_finite_float_or_nan(row.get("runtime_parameter_count")) for row in trajectory]
    d_curvature_vals = [
        _as_finite_float_or_nan(row.get("tracking_d_curvature_abs_error_mean")) for row in trajectory
    ]
    d_exc_under_vals = [
        _as_finite_float_or_nan(row.get("tracking_d_excursion_under_response_mean")) for row in trajectory
    ]
    d_exc_over_vals = [
        _as_finite_float_or_nan(row.get("tracking_d_excursion_over_response_mean")) for row in trajectory
    ]
    tracking_total_occ_vals = [
        _as_finite_float_or_nan(row.get("tracking_total_occupation_abs_error_mean")) for row in trajectory
    ]
    selection_reason_counts = _string_histogram(
        [row.get("exact_v1_selection_reason", None) for row in trajectory]
    )
    decision_override_reason_counts = _string_histogram(
        [row.get("decision_override_reason", None) for row in trajectory]
    )
    soft_fallback_reason_counts = dict(
        summary.get(
            "high_miss_no_admit_soft_fallback_reason_counts",
            soft_fallback_counts["high_miss_no_admit_soft_fallback_reason_counts"],
        )
    )
    high_miss_no_admit_soft_fallback_count = int(
        summary.get(
            "high_miss_no_admit_soft_fallback_count",
            soft_fallback_counts["high_miss_no_admit_soft_fallback_count"],
        )
    )
    protected_horizon_selection_count = int(
        sum(
            1
            for row in trajectory
            if str(row.get("exact_v1_selection_reason", "")) == "d_shape_barrier_protected_horizon"
        )
    )
    proposed_append_count = int(
        sum(1 for row in trajectory if str(row.get("proposed_action_kind", "")) == "append_candidate")
    )
    guardrail_d_shape_override_count = int(
        sum(
            1
            for row in trajectory
            if str(row.get("decision_override_reason", "")).startswith("exact_forecast_d_shape_")
        )
    )

    site_occ = raw_site_occ = None
    site_occ_exact = raw_site_occ_exact = None
    early_stop_raw = summary.get("early_stop_reason", payload.get("early_stop_reason"))
    early_stop_reason = None if str(early_stop_raw).strip().lower() in {"", "none", "null"} else early_stop_raw
    final_time = float(loaded.times[-1]) if loaded.times.size else float("nan")
    expected_t_final = float(base_cfg.t_final)
    expected_row_count = int(base_cfg.num_times)
    full_horizon_fields = full_horizon_completion_fields(
        raw_trajectory,
        expected_t_final=float(expected_t_final),
        expected_row_count=int(expected_row_count),
        early_stop_reason=early_stop_reason,
        stable_early_stop_accepted=_summary_declares_successful_stable_early_stop(summary),
    )
    metrics: dict[str, Any] = {
        "generic_exact_v1_family_objective": _is_diagnostic_generic_exact_profile(str(base_cfg.study_profile)),
        "trajectory_row_count": int(len(raw_trajectory)),
        **repair_counts,
        **full_horizon_fields,
        "expected_trajectory_row_count": int(expected_row_count),
        "final_time": float(final_time),
        "expected_t_final": float(expected_t_final),
        "trajectory_reached_final_time": bool(full_horizon_fields["full_horizon_reached_final_time"]),
        "trajectory_reached_expected_rows": bool(full_horizon_fields["full_horizon_reached_expected_rows"]),
        "analysis_status": analysis_metadata.get("analysis_status", "ok"),
        "analysis_error": analysis_metadata.get("analysis_error"),
        "compiled_count_2q": parameter_manifest.get("compiled_count_2q"),
        "compiled_count_2q_source": parameter_manifest.get("compiled_count_2q_source"),
        "compiled_depth": parameter_manifest.get("compiled_depth"),
        "compiled_depth_source": parameter_manifest.get("compiled_depth_source"),
        "compile_backend": parameter_manifest.get("compile_backend"),
        "compile_note": parameter_manifest.get("compile_note"),
        "append_count": int(summary.get("append_count", 0)),
        "prune_count": int(summary.get("prune_count", 0)),
        "integrator_euler_count": int(summary.get("integrator_euler_count", 0)),
        "integrator_rk4_count": int(summary.get("integrator_rk4_count", 0)),
        "high_miss_count": int(summary.get("high_miss_count", high_miss_no_admit_counts["high_miss_count"])),
        "high_miss_fraction": float(summary.get("high_miss_fraction", high_miss_no_admit_counts["high_miss_fraction"])),
        "high_miss_no_admit_count": int(
            summary.get("high_miss_no_admit_count", high_miss_no_admit_counts["high_miss_no_admit_count"])
        ),
        "high_miss_no_admit_fraction": float(
            summary.get("high_miss_no_admit_fraction", high_miss_no_admit_counts["high_miss_no_admit_fraction"])
        ),
        "high_miss_no_admit_reason_counts": dict(
            summary.get(
                "high_miss_no_admit_reason_counts",
                high_miss_no_admit_counts["high_miss_no_admit_reason_counts"],
            )
        ),
        "high_miss_no_admit_resolution_counts": dict(
            summary.get(
                "high_miss_no_admit_resolution_counts",
                high_miss_no_admit_counts["high_miss_no_admit_resolution_counts"],
            )
        ),
        "first_bad_high_miss_no_admit_checkpoint_diagnostic": summary.get(
            "first_bad_high_miss_no_admit_checkpoint_diagnostic",
            high_miss_no_admit_counts["first_bad_high_miss_no_admit_checkpoint_diagnostic"],
        ),
        "append_no_harm_veto_count": int(
            summary.get("append_no_harm_veto_count", high_miss_no_admit_counts["append_no_harm_veto_count"])
        ),
        "append_no_harm_veto_reason_counts": dict(
            summary.get(
                "append_no_harm_veto_reason_counts",
                high_miss_no_admit_counts["append_no_harm_veto_reason_counts"],
            )
        ),
        "prune_appended_origin_bias_influenced_count": int(
            summary.get("prune_appended_origin_bias_influenced_count", 0)
        ),
        "final_runtime_parameter_count": int(
            summary.get(
                "final_runtime_parameter_count",
                int(runtime_vals[-1]) if runtime_vals and math.isfinite(float(runtime_vals[-1])) else 0,
            )
        ),
        "mean_runtime_parameter_count": _safe_mean(runtime_vals),
        "min_fidelity_exact": _safe_min(fidelity_vals),
        "mean_abs_energy_total_error": _safe_mean(energy_err_vals),
        "final_abs_energy_total_error": _safe_last(energy_err_vals),
        "max_abs_energy_total_error": float(summary.get("max_abs_energy_total_error", np.nan)),
        "mean_abs_site_occupations_error": _safe_mean(site_err_vals),
        "max_abs_site_occupations_error": float(summary.get("max_abs_site_occupations_error", np.nan)),
        "summary_status": summary.get("status", payload.get("status")),
        "early_stop_reason": early_stop_reason,
        "exact_forecast_veto_count": int(summary.get("exact_forecast_veto_count", 0)),
        "decision_override_count": int(summary.get("decision_override_count", 0)),
        "high_miss_no_admit_soft_fallback_count": int(high_miss_no_admit_soft_fallback_count),
        "high_miss_no_admit_soft_fallback_fraction": float(
            high_miss_no_admit_soft_fallback_count / max(len(trajectory), 1)
        ),
        "ordinary_stay_count": int(
            summary.get("ordinary_stay_count", soft_fallback_counts["ordinary_stay_count"])
        ),
        "soft_fallback_reason_counts": dict(soft_fallback_reason_counts),
        "protected_horizon_selection_count": int(protected_horizon_selection_count),
        "proposed_append_count": int(proposed_append_count),
        "append_commit_rate": float(int(summary.get("append_count", 0)) / max(int(proposed_append_count), 1)),
        "guardrail_d_shape_override_count": int(guardrail_d_shape_override_count),
        "protected_horizon_selection_fraction": float(
            protected_horizon_selection_count / max(len(trajectory), 1)
        ),
        "guardrail_veto_fraction": float(int(summary.get("exact_forecast_veto_count", 0)) / max(len(trajectory), 1)),
        "decision_override_fraction": float(int(summary.get("decision_override_count", 0)) / max(len(trajectory), 1)),
        "mean_tracking_d_curvature_abs_error_mean": _safe_mean(d_curvature_vals),
        "mean_tracking_d_excursion_under_response_mean": _safe_mean(d_exc_under_vals),
        "mean_tracking_d_excursion_over_response_mean": _safe_mean(d_exc_over_vals),
        "mean_tracking_total_occupation_abs_error_mean": _safe_mean(tracking_total_occ_vals),
        "selection_reason_counts": selection_reason_counts,
        "decision_override_reason_counts": decision_override_reason_counts,
    }
    metrics.update(
        _energy_tracking_metric_bundle(
            ctrl=np.asarray(energy_total_vals, dtype=float),
            exact=np.asarray(energy_total_exact_vals, dtype=float),
        )
    )

    raw = analysis.get("raw_traces", {}) if isinstance(analysis.get("raw_traces", {}), Mapping) else {}
    raw_site_occ = raw.get("site_occupations", None)
    raw_site_occ_exact = raw.get("site_occupations_exact", None)
    if raw_site_occ is not None and raw_site_occ_exact is not None:
        site_occ = np.asarray(raw_site_occ, dtype=float)
        site_occ_exact = np.asarray(raw_site_occ_exact, dtype=float)
        site_err = np.abs(site_occ - site_occ_exact)
        total_occ_err = np.abs(np.sum(site_occ, axis=1) - np.sum(site_occ_exact, axis=1))
        metrics["mean_abs_site_occupations_error"] = float(np.mean(site_err))
        metrics["max_abs_site_occupations_error"] = float(np.max(site_err))
        metrics["mean_total_occupation_abs_error"] = float(np.mean(total_occ_err))
        metrics["final_total_occupation_abs_error"] = float(total_occ_err[-1])
    else:
        metrics["mean_total_occupation_abs_error"] = _safe_mean(tracking_total_occ_vals)
        metrics["final_total_occupation_abs_error"] = _safe_last(tracking_total_occ_vals)
    pair_ctrl_raw = raw.get("pair_difference", None)
    pair_exact_raw = raw.get("pair_difference_exact", None)
    if pair_ctrl_raw is not None and pair_exact_raw is not None:
        pair_ctrl = np.asarray(pair_ctrl_raw, dtype=float)
        pair_exact = np.asarray(pair_exact_raw, dtype=float)
        pair_err = np.asarray(pair_ctrl - pair_exact, dtype=float)
        pair_span_exact = float(np.ptp(pair_exact))
        metrics.update(
            {
                "pair_corr": _corrcoef(pair_ctrl, pair_exact),
                "pair_mae": float(np.mean(np.abs(pair_err))),
                "pair_span_exact": float(pair_span_exact),
                "pair_mae_over_exact_span": float(
                    float(np.mean(np.abs(pair_err))) / max(float(pair_span_exact), 1.0e-12)
                ),
                "epsilon_osc_pair": _eps_osc(pair_ctrl, pair_exact),
            }
        )
    else:
        metrics.update(
            {
                "pair_corr": float("nan"),
                "pair_mae": float("nan"),
                "pair_span_exact": float("nan"),
                "pair_mae_over_exact_span": float("nan"),
                "epsilon_osc_pair": float("nan"),
            }
        )

    pair_key = None if pair is None else f"pair_difference_{pair[0]}_{pair[1]}"
    spectra = analysis.get("spectra", {}) if isinstance(analysis.get("spectra", {}), Mapping) else {}
    if pair_key is not None and pair_key in spectra and f"{pair_key}_exact" in spectra:
        ctrl_spec = dict(spectra[pair_key])
        exact_spec = dict(spectra[f"{pair_key}_exact"])
        omega = np.asarray(ctrl_spec.get("omega", []), dtype=float)
        ctrl_amp = np.asarray(ctrl_spec.get("amplitude", []), dtype=float)
        exact_amp = np.asarray(exact_spec.get("amplitude", []), dtype=float)
        ctrl_peaks = list(ctrl_spec.get("top_peaks", []))
        exact_peaks = list(exact_spec.get("top_peaks", []))
        if ctrl_peaks and exact_peaks:
            ctrl_dom = dict(ctrl_peaks[0])
            exact_dom = dict(exact_peaks[0])
            exact_dom_idx = int(round(float(exact_dom.get("index", 0.0))))
            exact_dom_idx = min(max(exact_dom_idx, 0), max(0, int(ctrl_amp.size) - 1))
            drive_omega = analysis.get("metadata", {}).get("drive_omega", None)
            drive_idx = None
            if drive_omega is not None and omega.size > 0 and math.isfinite(float(drive_omega)):
                drive_idx = int(np.argmin(np.abs(omega - float(drive_omega))))
            metrics.update(
                {
                    "dominant_peak_ctrl_omega": float(ctrl_dom.get("omega", float("nan"))),
                    "dominant_peak_exact_omega": float(exact_dom.get("omega", float("nan"))),
                    "dominant_peak_abs_omega_error": float(
                        abs(float(ctrl_dom.get("omega", float("nan"))) - float(exact_dom.get("omega", float("nan"))))
                    ),
                    "exact_dominant_bin_ratio_ctrl_over_exact": (
                        float(ctrl_amp[exact_dom_idx] / exact_amp[exact_dom_idx])
                        if exact_amp.size > exact_dom_idx and abs(float(exact_amp[exact_dom_idx])) > 1.0e-12
                        else float("nan")
                    ),
                    "drive_line_ratio_ctrl_over_exact": (
                        float(ctrl_amp[int(drive_idx)] / exact_amp[int(drive_idx)])
                        if drive_idx is not None
                        and exact_amp.size > int(drive_idx)
                        and abs(float(exact_amp[int(drive_idx)])) > 1.0e-12
                        else float("nan")
                    ),
                }
            )
            drive_ratio = metrics.get("drive_line_ratio_ctrl_over_exact", float("nan"))
            try:
                drive_ratio_float = float(drive_ratio)
            except Exception:
                drive_ratio_float = float("nan")
            metrics["drive_line_ratio_defect"] = (
                float(abs(1.0 - drive_ratio_float)) if math.isfinite(drive_ratio_float) else float("nan")
            )
        else:
            metrics.update(
                {
                    "dominant_peak_ctrl_omega": float("nan"),
                    "dominant_peak_exact_omega": float("nan"),
                    "dominant_peak_abs_omega_error": float("nan"),
                    "exact_dominant_bin_ratio_ctrl_over_exact": float("nan"),
                    "drive_line_ratio_ctrl_over_exact": float("nan"),
                    "drive_line_ratio_defect": float("nan"),
                }
            )
    else:
        metrics.update(
            {
                "dominant_peak_ctrl_omega": float("nan"),
                "dominant_peak_exact_omega": float("nan"),
                "dominant_peak_abs_omega_error": float("nan"),
                "exact_dominant_bin_ratio_ctrl_over_exact": float("nan"),
                "drive_line_ratio_ctrl_over_exact": float("nan"),
                "drive_line_ratio_defect": float("nan"),
            }
        )

    primary_series = _primary_observable_series(
        payload=payload,
        analysis=analysis,
        analysis_metadata=analysis_metadata,
        trajectory=trajectory,
        pair=pair,
    )
    if primary_series is not None:
        primary_times, primary_ctrl, primary_exact, primary_name, primary_source = primary_series
        primary_metrics = _primary_observable_metric_bundle(
            times=primary_times,
            ctrl=primary_ctrl,
            exact=primary_exact,
            base_cfg=base_cfg,
            drive_omega=loaded.drive_omega,
        )
        metrics.update(
            {
                "primary_observable_name": str(primary_name),
                "primary_observable_source": str(primary_source),
                **primary_metrics,
            }
        )
    else:
        primary_times = primary_ctrl = primary_exact = None
        metrics.update(
            {
                "primary_observable_name": None,
                "primary_observable_source": None,
                "primary_observable_num_samples": 0,
                "primary_observable_mae": float("nan"),
                "primary_observable_span_exact": float("nan"),
                "primary_observable_mae_over_exact_span": float("nan"),
                "primary_observable_corr": float("nan"),
                "primary_observable_epsilon_osc": float("nan"),
                "primary_observable_dominant_peak_abs_omega_error": float("nan"),
                "primary_observable_spectrum_available": False,
            }
        )
    analysis_metadata_out = (
        dict(analysis.get("metadata", {})) if isinstance(analysis.get("metadata", {}), Mapping) else {}
    )
    analysis_metadata_out.update(
        {
            "primary_observable_name": metrics.get("primary_observable_name"),
            "primary_observable_source": metrics.get("primary_observable_source"),
            "primary_observable_spectrum_available": bool(
                metrics.get("primary_observable_spectrum_available", False)
            ),
        }
    )
    analysis["metadata"] = analysis_metadata_out

    window_start = base_cfg.objective_window_start
    window_end = base_cfg.objective_window_end
    if window_start is not None or window_end is not None:
        times = np.asarray(loaded.times, dtype=float)
        mask = np.ones(times.shape, dtype=bool)
        if window_start is not None:
            mask &= times >= float(window_start) - 1.0e-12
        if window_end is not None:
            mask &= times <= float(window_end) + 1.0e-12
        indices = np.flatnonzero(mask)
        metrics["objective_window_start"] = None if window_start is None else float(window_start)
        metrics["objective_window_end"] = None if window_end is None else float(window_end)
        metrics["objective_window_num_samples"] = int(indices.size)
        if indices.size >= 2:
            fidelity_arr = np.asarray(fidelity_vals, dtype=float)
            energy_err_arr = np.asarray(energy_err_vals, dtype=float)
            site_err_arr = np.asarray(site_err_vals, dtype=float)
            if fidelity_arr.size == times.size:
                metrics["objective_min_fidelity_exact"] = _safe_min(fidelity_arr[indices].tolist())
            if energy_err_arr.size == times.size:
                metrics["objective_mean_abs_energy_total_error"] = _safe_mean(energy_err_arr[indices].tolist())
            energy_total_arr = np.asarray(energy_total_vals, dtype=float)
            energy_total_exact_arr = np.asarray(energy_total_exact_vals, dtype=float)
            if energy_total_arr.size == times.size and energy_total_exact_arr.size == times.size:
                metrics.update(
                    _prefixed_metric_bundle(
                        "objective_",
                        _energy_tracking_metric_bundle(
                            ctrl=energy_total_arr[indices],
                            exact=energy_total_exact_arr[indices],
                        ),
                    )
                )
            if site_err_arr.size == times.size:
                metrics["objective_mean_abs_site_occupations_error"] = _safe_mean(site_err_arr[indices].tolist())
            if pair_ctrl_raw is not None and pair_exact_raw is not None:
                pair_ctrl_obj = np.asarray(pair_ctrl_raw, dtype=float)[indices]
                pair_exact_obj = np.asarray(pair_exact_raw, dtype=float)[indices]
                pair_err_obj = np.asarray(pair_ctrl_obj - pair_exact_obj, dtype=float)
                pair_span_exact_obj = float(np.ptp(pair_exact_obj))
                metrics["objective_pair_corr"] = _corrcoef(pair_ctrl_obj, pair_exact_obj)
                metrics["objective_pair_mae"] = float(np.mean(np.abs(pair_err_obj)))
                metrics["objective_pair_span_exact"] = float(pair_span_exact_obj)
                metrics["objective_pair_mae_over_exact_span"] = float(
                    float(np.mean(np.abs(pair_err_obj))) / max(float(pair_span_exact_obj), 1.0e-12)
                )
                metrics["objective_epsilon_osc_pair"] = _eps_osc(pair_ctrl_obj, pair_exact_obj)
                if indices.size >= 3:
                    window_times = times[indices]
                    try:
                        ctrl_spec_obj = compute_one_sided_amplitude_spectrum(
                            window_times,
                            pair_ctrl_obj,
                            detrend=str(base_cfg.spectra_detrend),
                            window=str(base_cfg.spectra_window),
                            max_peaks=int(base_cfg.max_peaks),
                            drive_omega=loaded.drive_omega,
                            max_harmonic=int(base_cfg.max_harmonic),
                        )
                        exact_spec_obj = compute_one_sided_amplitude_spectrum(
                            window_times,
                            pair_exact_obj,
                            detrend=str(base_cfg.spectra_detrend),
                            window=str(base_cfg.spectra_window),
                            max_peaks=int(base_cfg.max_peaks),
                            drive_omega=loaded.drive_omega,
                            max_harmonic=int(base_cfg.max_harmonic),
                        )
                        if ctrl_spec_obj.top_peaks and exact_spec_obj.top_peaks:
                            metrics["objective_dominant_peak_abs_omega_error"] = float(
                                abs(
                                    float(ctrl_spec_obj.top_peaks[0]["omega"])
                                    - float(exact_spec_obj.top_peaks[0]["omega"])
                                )
                            )
                    except Exception:
                        pass
        if primary_times is not None and primary_ctrl is not None and primary_exact is not None:
            primary_times_arr = np.asarray(primary_times, dtype=float)
            primary_mask = np.ones(primary_times_arr.shape, dtype=bool)
            if window_start is not None:
                primary_mask &= primary_times_arr >= float(window_start) - 1.0e-12
            if window_end is not None:
                primary_mask &= primary_times_arr <= float(window_end) + 1.0e-12
            primary_indices = np.flatnonzero(primary_mask)
            if primary_indices.size >= 2:
                objective_primary_metrics = _primary_observable_metric_bundle(
                    times=np.asarray(primary_times, dtype=float)[primary_indices],
                    ctrl=np.asarray(primary_ctrl, dtype=float)[primary_indices],
                    exact=np.asarray(primary_exact, dtype=float)[primary_indices],
                    base_cfg=base_cfg,
                    drive_omega=loaded.drive_omega,
                )
                metrics.update(_prefixed_metric_bundle("objective_", objective_primary_metrics))
    return analysis, metrics


def _apply_base_run_trial_overrides(params: TrialParams, base_cfg: BaseRunConfig) -> TrialParams:
    integrator_policy = base_cfg.integrator_policy_override
    if integrator_policy in {None, ""}:
        return params
    normalized = str(integrator_policy).strip().lower()
    if normalized not in {"euler", "rk4", "auto_euler_rk4"}:
        raise ValueError(
            "integrator_policy_override must be one of euler, rk4, or auto_euler_rk4; "
            f"got {integrator_policy!r}"
        )
    return replace(params, integrator_policy=normalized)


def _append_non_strict_horizon_and_gate_reasons(
    out: list[str],
    *,
    metrics: Mapping[str, Any],
    gates: ValidityGates | None,
) -> None:
    early_stop_reason = metrics.get("early_stop_reason")
    successful_stable_early_stop = bool(
        metrics.get("full_horizon_gate_passed") is True
        and metrics.get("full_horizon_successful_early_stop") is True
        and str(metrics.get("full_horizon_completion_kind", "")) == "stable_early_stop"
    )
    if early_stop_reason not in {None, "", "none", "None"} and not successful_stable_early_stop:
        out.append(f"trajectory_early_stop:{early_stop_reason}")
    if metrics.get("trajectory_reached_final_time") is False and not successful_stable_early_stop:
        out.append("trajectory_incomplete:final_time")
    if metrics.get("trajectory_reached_expected_rows") is False and not successful_stable_early_stop:
        out.append("trajectory_incomplete:row_count")
    if metrics.get("full_horizon_gate_passed") is False:
        out.append(f"full_horizon_gate:{metrics.get('full_horizon_gate_reason', 'failed')}")
    if gates is not None and gates.max_mean_abs_energy_total_error is not None:
        mean_energy = _finite_metric(metrics, "mean_abs_energy_total_error")
        if mean_energy is not None and float(mean_energy) > float(gates.max_mean_abs_energy_total_error):
            out.append(
                "above_gate:mean_abs_energy_total_error>"
                f"{float(gates.max_mean_abs_energy_total_error):.6g}"
            )
    if gates is not None and gates.max_final_abs_energy_total_error is not None:
        final_energy = _finite_metric(metrics, "final_abs_energy_total_error")
        if final_energy is None:
            out.append("missing_or_nonfinite:final_abs_energy_total_error")
        elif float(final_energy) > float(gates.max_final_abs_energy_total_error):
            out.append(
                "above_gate:final_abs_energy_total_error>"
                f"{float(gates.max_final_abs_energy_total_error):.6g}"
            )
    if gates is not None and gates.min_fidelity_exact is not None:
        min_fidelity = _finite_metric(metrics, "min_fidelity_exact")
        if min_fidelity is None:
            out.append("missing_or_nonfinite:min_fidelity_exact")
        elif float(min_fidelity) < float(gates.min_fidelity_exact):
            out.append(
                "below_gate:min_fidelity_exact<"
                f"{float(gates.min_fidelity_exact):.6g}"
            )
    if gates is not None and gates.max_mean_total_occupation_abs_error is not None:
        mean_total_occ = _finite_metric(metrics, "mean_total_occupation_abs_error")
        if mean_total_occ is None:
            mean_total_occ = _finite_metric(metrics, "mean_tracking_total_occupation_abs_error_mean")
        if mean_total_occ is None:
            out.append("missing_or_nonfinite:mean_total_occupation_abs_error")
        elif float(mean_total_occ) > float(gates.max_mean_total_occupation_abs_error):
            out.append(
                "above_gate:mean_total_occupation_abs_error>"
                f"{float(gates.max_mean_total_occupation_abs_error):.6g}"
            )
    if gates is not None and gates.max_primary_observable_mae_over_exact_span is not None:
        rel_metric = _generic_exact_v1_signal_metric(
            metrics,
            "mae_over_exact_span",
            objective_window=False,
        )
        if rel_metric is None:
            out.append("missing_or_nonfinite:primary_observable_mae_over_exact_span")
        elif float(rel_metric) > float(gates.max_primary_observable_mae_over_exact_span):
            out.append(
                "above_gate:primary_observable_mae_over_exact_span>"
                f"{float(gates.max_primary_observable_mae_over_exact_span):.6g}"
            )


def _invalid_reasons(metrics: Mapping[str, Any], gates: ValidityGates | None = None) -> list[str]:
    if bool(metrics.get("strict_qpu_faithful", False)):
        out: list[str] = []
        if metrics.get("qpu_faithful_decisions_expected") is not True:
            out.append("strict_purity:qpu_faithful_decisions_not_expected")
        if metrics.get("qpu_faithful_decisions_passed") is not True:
            out.append("strict_purity:qpu_faithful_decisions_failed")
        if metrics.get("strict_decision_contract_passed") is False:
            for violation in metrics.get("strict_decision_contract_violations", []) or []:
                out.append(f"strict_purity_contract:{violation}")
        if gates is not None and gates.max_primary_observable_mae_over_exact_span is not None:
            out.append("strict_purity:exact_span_gate_configured")
        if bool(metrics.get("strict_fail_closed", False)):
            reason = metrics.get("strict_fail_closed_reason", "unknown")
            out.append(f"strict_fail_closed:{reason}")
        exact_decisions = _finite_metric(metrics, "exact_decision_checkpoints")
        if exact_decisions is not None and int(exact_decisions) > 0:
            out.append(f"strict_purity:exact_decision_checkpoints>{int(exact_decisions)}")
        if bool(metrics.get("reference_enabled", False)):
            out.append("strict_purity:reference_enabled")
        if str(metrics.get("reference_mode", "off")) != "off":
            out.append(f"strict_purity:reference_mode={metrics.get('reference_mode')}")
        non_ideal_noise = _finite_metric(metrics, "non_ideal_decision_noise_count")
        if non_ideal_noise is not None and int(non_ideal_noise) > 0:
            out.append(f"strict_purity:non_ideal_decision_noise>{int(non_ideal_noise)}")
        if str(metrics.get("decision_noise_mode", "ideal")) != "ideal":
            out.append(f"strict_purity:decision_noise_mode={metrics.get('decision_noise_mode')}")
        early_stop_reason = metrics.get("early_stop_reason")
        if early_stop_reason not in {None, "", "none", "None"}:
            out.append(f"trajectory_early_stop:{early_stop_reason}")
        if metrics.get("trajectory_reached_final_time") is False:
            out.append("trajectory_incomplete:final_time")
        if metrics.get("trajectory_reached_expected_rows") is False:
            out.append("trajectory_incomplete:row_count")
        if metrics.get("full_horizon_gate_passed") is False:
            out.append(f"full_horizon_gate:{metrics.get('full_horizon_gate_reason', 'failed')}")
        if not bool(metrics.get("strict_fail_closed", False)):
            for key in (
                "mean_rho_miss",
                "max_rho_miss",
                "strict_degraded_fraction",
                "final_runtime_parameter_count",
            ):
                if _finite_metric(metrics, key) is None:
                    out.append(f"missing_or_nonfinite:{key}")
        degraded_count = _finite_metric(metrics, "strict_degraded_checkpoint_count")
        if degraded_count is not None and int(degraded_count) > 0:
            out.append(f"strict_degraded_checkpoints:{int(degraded_count)}")
        return out

    if bool(metrics.get("generic_exact_v1_family_objective", False)):
        out: list[str] = []
        if (
            _generic_exact_v1_signal_metric(
                metrics,
                "mae_over_exact_span",
                objective_window=False,
            )
            is None
        ):
            out.append("missing_or_nonfinite:primary_observable_mae_over_exact_span")
        for key in (
            "mean_abs_site_occupations_error",
            "mean_abs_energy_total_error",
            "min_fidelity_exact",
            "final_runtime_parameter_count",
        ):
            if _finite_metric(metrics, key) is None:
                out.append(f"missing_or_nonfinite:{key}")
        _append_non_strict_horizon_and_gate_reasons(out, metrics=metrics, gates=gates)
        return out

    required = (
        "pair_mae_over_exact_span",
        "epsilon_osc_pair",
        "dominant_peak_abs_omega_error",
        "mean_abs_site_occupations_error",
        "mean_abs_energy_total_error",
        "min_fidelity_exact",
        "final_runtime_parameter_count",
    )
    out: list[str] = []
    for key in required:
        if _finite_metric(metrics, key) is None:
            out.append(f"missing_or_nonfinite:{key}")
    _append_non_strict_horizon_and_gate_reasons(out, metrics=metrics, gates=gates)
    return out


def _objective_value(metrics: Mapping[str, Any], weights: ObjectiveWeights) -> float:
    if bool(metrics.get("strict_qpu_faithful", False)):
        if bool(metrics.get("strict_fail_closed", False)) or metrics.get(
            "qpu_faithful_decisions_passed"
        ) is not True:
            return float(_LARGE_OBJECTIVE)

        def _strict_value_or_penalty(key: str) -> float:
            value = _finite_metric(metrics, key)
            return float(_LARGE_OBJECTIVE if value is None else value)

        append_count = _strict_value_or_penalty("append_count")
        prune_count = _strict_value_or_penalty("prune_count")
        append_opportunities = max(
            _strict_value_or_penalty("append_opportunity_count"),
            _strict_value_or_penalty("proposed_append_count"),
        )
        prune_opportunities = max(
            _strict_value_or_penalty("prune_opportunity_count"),
            _strict_value_or_penalty("prune_candidate_checkpoint_count"),
        )
        append_shortfall = 1.0 if append_opportunities > 0.0 and append_count < 1.0 else 0.0
        prune_shortfall = 1.0 if prune_opportunities > 0.0 and prune_count < 1.0 else 0.0
        parts = [
            _strict_value_or_penalty("strict_measured_degradation_score"),
            0.25 * _strict_value_or_penalty("max_rho_miss"),
            0.10 * _strict_value_or_penalty("mean_predicted_displacement"),
            10.0 * _strict_value_or_penalty("strict_degraded_fraction"),
            0.50 * _strict_value_or_penalty("high_miss_no_admit_fraction"),
            0.25 * _strict_value_or_penalty("high_miss_no_admit_soft_fallback_fraction"),
            float(weights.final_runtime_parameter_count)
            * _strict_value_or_penalty("final_runtime_parameter_count"),
            float(weights.append_count) * append_shortfall,
            float(weights.prune_count) * prune_shortfall,
            float(weights.rk4_count) * _strict_value_or_penalty("integrator_rk4_count"),
        ]
        total = float(sum(parts))
        return total if math.isfinite(total) else float(_LARGE_OBJECTIVE)

    def _objective_metric_value(key: str) -> float | None:
        preferred = _finite_metric(metrics, f"objective_{key}")
        return preferred if preferred is not None else _finite_metric(metrics, key)

    def _value_or_penalty(key: str) -> float:
        value = _objective_metric_value(key)
        return float(_LARGE_OBJECTIVE if value is None else value)

    if bool(metrics.get("generic_exact_v1_family_objective", False)):
        signal_mae = _generic_exact_v1_signal_metric(
            metrics,
            "mae_over_exact_span",
            objective_window=True,
        )
        min_fidelity = _objective_metric_value("min_fidelity_exact")
        parts = [
            float(weights.pair_mae_over_exact_span)
            * float(_LARGE_OBJECTIVE if signal_mae is None else signal_mae),
            float(weights.mean_abs_site_occupations_error)
            * _value_or_penalty("mean_abs_site_occupations_error"),
            float(weights.mean_abs_energy_total_error)
            * _value_or_penalty("mean_abs_energy_total_error"),
            float(weights.mean_signed_energy_total_error_abs)
            * _value_or_penalty("mean_signed_energy_total_error_abs"),
            float(weights.max_abs_energy_total_error)
            * _value_or_penalty("max_abs_energy_total_error"),
            float(weights.energy_total_excursion_under_response_mean)
            * _value_or_penalty("energy_total_excursion_under_response_mean"),
            float(weights.energy_total_excursion_under_response_max)
            * _value_or_penalty("energy_total_excursion_under_response_max"),
            float(weights.fidelity_defect) * float(max(0.0, 1.0 - float(min_fidelity)))
            if min_fidelity is not None
            else float(weights.fidelity_defect) * _LARGE_OBJECTIVE,
            float(weights.final_runtime_parameter_count)
            * _value_or_penalty("final_runtime_parameter_count"),
            float(weights.append_count) * _value_or_penalty("append_count"),
            float(weights.prune_count) * _value_or_penalty("prune_count"),
            float(weights.rk4_count) * _value_or_penalty("integrator_rk4_count"),
        ]
        signal_corr = _generic_exact_v1_signal_metric(
            metrics,
            "corr",
            objective_window=True,
        )
        if signal_corr is not None:
            parts.append(
                float(weights.pair_corr_defect) * float(max(0.0, 1.0 - float(signal_corr)))
            )
        signal_epsilon = _generic_exact_v1_signal_metric(
            metrics,
            "epsilon_osc",
            objective_window=True,
        )
        if signal_epsilon is not None:
            parts.append(float(weights.epsilon_osc_pair) * float(signal_epsilon))
        signal_peak = _generic_exact_v1_signal_metric(
            metrics,
            "dominant_peak_abs_omega_error",
            objective_window=True,
        )
        if signal_peak is not None:
            parts.append(float(weights.dominant_peak_abs_omega_error) * float(signal_peak))
        total = float(sum(parts))
        return total if math.isfinite(total) else float(_LARGE_OBJECTIVE)

    pair_corr = _objective_metric_value("pair_corr")
    min_fidelity = _objective_metric_value("min_fidelity_exact")
    parts = [
        float(weights.pair_mae_over_exact_span) * _value_or_penalty("pair_mae_over_exact_span"),
        float(weights.epsilon_osc_pair) * _value_or_penalty("epsilon_osc_pair"),
        float(weights.dominant_peak_abs_omega_error) * _value_or_penalty("dominant_peak_abs_omega_error"),
        float(weights.pair_corr_defect) * float(max(0.0, 1.0 - float(pair_corr)))
        if pair_corr is not None
        else float(weights.pair_corr_defect) * _LARGE_OBJECTIVE,
        float(weights.mean_abs_site_occupations_error) * _value_or_penalty("mean_abs_site_occupations_error"),
        float(weights.mean_abs_energy_total_error) * _value_or_penalty("mean_abs_energy_total_error"),
        float(weights.mean_signed_energy_total_error_abs)
        * _value_or_penalty("mean_signed_energy_total_error_abs"),
        float(weights.max_abs_energy_total_error) * _value_or_penalty("max_abs_energy_total_error"),
        float(weights.energy_total_excursion_under_response_mean)
        * _value_or_penalty("energy_total_excursion_under_response_mean"),
        float(weights.energy_total_excursion_under_response_max)
        * _value_or_penalty("energy_total_excursion_under_response_max"),
        float(weights.fidelity_defect) * float(max(0.0, 1.0 - float(min_fidelity)))
        if min_fidelity is not None
        else float(weights.fidelity_defect) * _LARGE_OBJECTIVE,
        float(weights.final_runtime_parameter_count) * _value_or_penalty("final_runtime_parameter_count"),
        float(weights.append_count) * _value_or_penalty("append_count"),
        float(weights.prune_count) * _value_or_penalty("prune_count"),
        float(weights.rk4_count) * _value_or_penalty("integrator_rk4_count"),
    ]
    total = float(sum(parts))
    return total if math.isfinite(total) else float(_LARGE_OBJECTIVE)


def _observation_to_user_attrs(obs: TrialObservation) -> dict[str, Any]:
    return {
        "status": str(obs.status),
        "objective": float(obs.objective),
        "metrics": dict(obs.metrics),
        "invalid_reasons": list(obs.invalid_reasons),
        "output_json": obs.output_json,
        "spectra_json": obs.spectra_json,
        "spectra_pdf": obs.spectra_pdf,
        "trial_progress_json": obs.trial_progress_json,
        "trial_partial_payload_json": obs.trial_partial_payload_json,
        "elapsed_s": obs.elapsed_s,
        "error": obs.error,
    }


def _evaluate_trial(
    *,
    trial_number: int,
    params: TrialParams,
    base_cfg: BaseRunConfig,
    objective_weights: ObjectiveWeights,
    validity_gates: ValidityGates,
    output_dir: Path,
    route_ctx: OptunaRouteContext | None = None,
) -> TrialObservation:
    params = _apply_route_context_to_trial_params(params, route_ctx)
    params = _apply_guardrail_only_class_lock_overrides(params, base_cfg)
    params = _apply_base_run_trial_overrides(params, base_cfg)
    started = time.time()
    trial_dir = output_dir / "trials" / f"trial_{int(trial_number):04d}"
    result_json = trial_dir / "result.json"
    spectra_json = trial_dir / "spectra.json"
    spectra_pdf = trial_dir / "spectra.pdf"
    trial_progress_json = trial_dir / "realtime_progress.json"
    trial_partial_payload_json = trial_dir / "partial_payload.json"
    trial_summary_json = trial_dir / "trial_summary.json"
    trial_dir.mkdir(parents=True, exist_ok=True)
    tokens = _build_realtime_tokens(
        base_cfg=base_cfg,
        params=params,
        output_json=result_json,
        run_tag=f"trial_{int(trial_number):04d}",
        progress_json=trial_progress_json,
        partial_payload_json=trial_partial_payload_json,
        route_ctx=route_ctx,
    )
    _write_json(
        trial_dir / "input_tokens.json",
        {
            "generated_utc": _now_utc(),
            "tokens": list(tokens),
            "tuning_class": base_cfg.tuning_class,
            "params": asdict(params),
            "route_context": None if route_ctx is None else asdict(route_ctx),
            "class_settings_candidate_payload": _controller_settings_payload_from_trial(
                params=params,
                base_cfg=base_cfg,
                route_ctx=route_ctx,
            ),
            "hidden_exact_v1_cfg_overrides": _hidden_exact_v1_cfg_overrides(params),
            "trial_progress_json": str(trial_progress_json),
            "trial_partial_payload_json": str(trial_partial_payload_json),
        },
    )
    try:
        args = build_realtime_parser().parse_args(tokens)
        payload = _run_realtime_from_args_with_optuna_overrides(args=args, params=params)
        analysis, metrics = _trial_metrics_from_payload(
            payload=payload,
            output_json=result_json,
            base_cfg=base_cfg,
        )
        _write_json(spectra_json, analysis)
        spectra_pdf_path: Path | None = None
        if not bool(base_cfg.skip_spectra_pdf) and not _is_strict_qpu_faithful_profile(
            base_cfg.study_profile
        ):
            render_spectrum_pdf(
                analysis,
                output_pdf=spectra_pdf,
                max_harmonic=int(base_cfg.max_harmonic),
            )
            spectra_pdf_path = spectra_pdf
        invalid_reasons = _invalid_reasons(metrics, gates=validity_gates)
        objective = _objective_value(metrics, objective_weights)
        status = "invalid" if invalid_reasons else "completed"
        obs = TrialObservation(
            trial_number=int(trial_number),
            params=asdict(params),
            objective=float(_LARGE_OBJECTIVE if invalid_reasons else objective),
            status=str(status),
            metrics=dict(metrics),
            invalid_reasons=list(invalid_reasons),
            output_json=str(result_json),
            spectra_json=str(spectra_json),
            spectra_pdf=None if spectra_pdf_path is None else str(spectra_pdf_path),
            trial_progress_json=str(trial_progress_json),
            trial_partial_payload_json=str(trial_partial_payload_json),
            elapsed_s=float(time.time() - started),
        )
    except SystemExit as exc:
        obs = TrialObservation(
            trial_number=int(trial_number),
            params=asdict(params),
            objective=float(_LARGE_OBJECTIVE),
            status="failed",
            metrics={},
            invalid_reasons=["system_exit"],
            output_json=str(result_json),
            spectra_json=None,
            spectra_pdf=None,
            trial_progress_json=str(trial_progress_json),
            trial_partial_payload_json=str(trial_partial_payload_json),
            elapsed_s=float(time.time() - started),
            error=f"SystemExit: {exc}",
        )
    except Exception as exc:
        obs = TrialObservation(
            trial_number=int(trial_number),
            params=asdict(params),
            objective=float(_LARGE_OBJECTIVE),
            status="failed",
            metrics={},
            invalid_reasons=["exception"],
            output_json=str(result_json),
            spectra_json=None,
            spectra_pdf=None,
            trial_progress_json=str(trial_progress_json),
            trial_partial_payload_json=str(trial_partial_payload_json),
            elapsed_s=float(time.time() - started),
            error=str(exc),
        )
    _write_json(trial_summary_json, _observation_to_user_attrs(obs))
    return obs


def _study_progress_snapshot(
    *,
    observations: Sequence[TrialObservation],
    tag: str,
    output_dir: Path,
) -> dict[str, Any]:
    completed = [obs for obs in observations if str(obs.status) == "completed"]
    invalid = [obs for obs in observations if str(obs.status) == "invalid"]
    failed = [obs for obs in observations if str(obs.status) == "failed"]
    best = None if not completed else min(completed, key=lambda item: float(item.objective))
    return {
        "generated_utc": _now_utc(),
        "pipeline": _PIPELINE_NAME,
        "tag": str(tag),
        "output_dir": str(output_dir),
        "completed_trial_count": int(len(completed)),
        "invalid_trial_count": int(len(invalid)),
        "failed_trial_count": int(len(failed)),
        "best_objective": None if best is None else float(best.objective),
        "best_trial_number": None if best is None else int(best.trial_number),
        "done": False,
    }


def _pareto_front(records: Sequence[Mapping[str, Any]], *, x_key: str, y_key: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in records:
        x = _finite_metric(row, x_key)
        y = _finite_metric(row, y_key)
        if x is None or y is None:
            continue
        enriched = dict(row)
        enriched[str(x_key)] = float(x)
        enriched[str(y_key)] = float(y)
        candidates.append(enriched)
    frontier: list[dict[str, Any]] = []
    for row in candidates:
        x = float(row[x_key])
        y = float(row[y_key])
        dominated = False
        for other in candidates:
            if other is row:
                continue
            ox = float(other[x_key])
            oy = float(other[y_key])
            if ox <= x and oy <= y and (ox < x or oy < y):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda item: (float(item[x_key]), float(item[y_key])))


def _best_by_metric(observations: Sequence[TrialObservation], *, key: str, maximize: bool = False) -> dict[str, Any] | None:
    return _best_by_metric_filtered(observations, key=key, maximize=maximize, predicate=None)


def _diagnostic_physical_score(metrics: Mapping[str, Any]) -> float | None:
    """Post-run exact-reference review score for strict diagnostic summaries.

    This score is deliberately separate from the strict Optuna objective.  It
    may use exact/reference diagnostic outputs because it is generated after a
    trial completes for reporting and human review; it is not fed back into the
    controller or Optuna objective used during strict/QPU-faithful studies.
    """

    mean_energy = _finite_metric(metrics, "diagnostic_mean_abs_energy_total_error")
    if mean_energy is None:
        mean_energy = _finite_metric(metrics, "mean_abs_energy_total_error")
    max_energy = _finite_metric(metrics, "diagnostic_max_abs_energy_total_error")
    if max_energy is None:
        max_energy = _finite_metric(metrics, "max_abs_energy_total_error")
    mean_site = _finite_metric(metrics, "diagnostic_mean_abs_site_occupations_error")
    if mean_site is None:
        mean_site = _finite_metric(metrics, "mean_abs_site_occupations_error")
    max_site = _finite_metric(metrics, "diagnostic_max_abs_site_occupations_error")
    if max_site is None:
        max_site = _finite_metric(metrics, "max_abs_site_occupations_error")
    components: list[float] = []
    if mean_energy is not None:
        components.append(6.0 * float(mean_energy))
    if max_energy is not None:
        components.append(2.0 * float(max_energy))
    if mean_site is not None:
        components.append(2.0 * float(mean_site))
    if max_site is not None:
        components.append(float(max_site))
    if not components:
        return None
    score = float(sum(components))
    return score if math.isfinite(score) else None


def _best_by_diagnostic_physical_score(
    observations: Sequence[TrialObservation],
) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for obs in observations:
        if str(obs.status) != "completed":
            continue
        if obs.metrics.get("full_horizon_gate_passed") is False:
            continue
        score = _diagnostic_physical_score(obs.metrics)
        if score is None:
            continue
        rows.append(
            {
                "trial_number": int(obs.trial_number),
                "objective": float(obs.objective),
                "metric_value": float(score),
                "metric_name": "diagnostic_physical_score",
                "params": dict(obs.params),
                "metrics": dict(obs.metrics),
                "output_json": obs.output_json,
                "spectra_json": obs.spectra_json,
                "spectra_pdf": obs.spectra_pdf,
                "trial_progress_json": obs.trial_progress_json,
                "trial_partial_payload_json": obs.trial_partial_payload_json,
                "selection_note": (
                    "post_run_exact_diagnostic_review_only; not strict Optuna online feedback "
                    "and not a controller decision input"
                ),
            }
        )
    if not rows:
        return None
    return dict(min(rows, key=lambda item: float(item["metric_value"])))


def _best_by_metric_filtered(
    observations: Sequence[TrialObservation],
    *,
    key: str,
    maximize: bool = False,
    predicate: Callable[[TrialObservation], bool] | None = None,
) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for obs in observations:
        if str(obs.status) != "completed":
            continue
        if obs.metrics.get("full_horizon_gate_passed") is False:
            continue
        if predicate is not None and not bool(predicate(obs)):
            continue
        value = float(obs.objective) if str(key) == "objective" else _finite_metric(obs.metrics, key)
        if value is None:
            continue
        rows.append(
            {
                "trial_number": int(obs.trial_number),
                "objective": float(obs.objective),
                "metric_value": float(value),
                "metric_name": str(key),
                "params": dict(obs.params),
                "metrics": dict(obs.metrics),
            }
        )
    if not rows:
        return None
    best = max(rows, key=lambda item: float(item["metric_value"])) if maximize else min(
        rows,
        key=lambda item: float(item["metric_value"]),
    )
    return dict(best)


def _run_study(
    *,
    optuna: Any,
    tag: str,
    output_dir: Path,
    base_cfg: BaseRunConfig,
    objective_weights: ObjectiveWeights,
    validity_gates: ValidityGates,
    n_trials: int,
    n_startup_trials: int,
    sampler_seed: int,
    enqueue_baseline: bool,
) -> dict[str, Any]:
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        constant_liar=True,
        seed=int(sampler_seed),
        n_startup_trials=int(max(1, n_startup_trials)),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=str(tag))
    profile = str(base_cfg.study_profile)
    route_ctx = _resolve_optuna_route_context(base_cfg)
    if bool(enqueue_baseline):
        study.enqueue_trial(
            _trial_to_enqueue_params(
                _baseline_trial_params(profile=profile),
                profile=profile,
                route_ctx=route_ctx,
            )
        )
    for seed_params in _profile_seed_trials(profile=profile):
        study.enqueue_trial(
            _trial_to_enqueue_params(seed_params, profile=profile, route_ctx=route_ctx)
        )

    observations: list[TrialObservation] = []
    for step_idx in range(int(max(0, n_trials))):
        trial = study.ask()
        params = _suggest_trial_params(trial, profile=profile, route_ctx=route_ctx)
        obs = _evaluate_trial(
            trial_number=int(trial.number),
            params=params,
            base_cfg=base_cfg,
            objective_weights=objective_weights,
            validity_gates=validity_gates,
            output_dir=output_dir,
            route_ctx=route_ctx,
        )
        for key, value in _observation_to_user_attrs(obs).items():
            trial.set_user_attr(str(key), value)
        study.tell(trial, float(obs.objective))
        observations.append(obs)
        progress = _study_progress_snapshot(observations=observations, tag=tag, output_dir=output_dir)
        progress["last_completed_step"] = int(step_idx)
        _write_progress(output_dir, progress)

    observation_rows = [
        {
            "trial_number": int(obs.trial_number),
            "objective": float(obs.objective),
            "status": str(obs.status),
            "params": dict(obs.params),
            "metrics": dict(obs.metrics),
            "invalid_reasons": list(obs.invalid_reasons),
            "output_json": obs.output_json,
            "spectra_json": obs.spectra_json,
            "spectra_pdf": obs.spectra_pdf,
            "trial_progress_json": obs.trial_progress_json,
            "trial_partial_payload_json": obs.trial_partial_payload_json,
            "elapsed_s": obs.elapsed_s,
            "error": obs.error,
        }
        for obs in observations
    ]
    completed_rows = [row for row in observation_rows if str(row.get("status")) == "completed"]
    payload = {
        "generated_utc": _now_utc(),
        "pipeline": _PIPELINE_NAME,
        "tag": str(tag),
        "output_dir": str(output_dir),
        "tuning_class": base_cfg.tuning_class,
        "tuning_class_source": "cli" if base_cfg.tuning_class else None,
        "search_space": _apply_route_context_to_search_space(
            _search_space(profile=str(base_cfg.study_profile)),
            route_ctx,
        ),
        "route_context": None if route_ctx is None else asdict(route_ctx),
        "class_settings_lock": _class_settings_lock_metadata(base_cfg),
        "base_run_config": asdict(base_cfg),
        "objective_weights": asdict(objective_weights),
        "validity_gates": asdict(validity_gates),
        "n_trials_requested": int(n_trials),
        "n_startup_trials": int(n_startup_trials),
        "enqueue_baseline": bool(enqueue_baseline),
        "observations": observation_rows,
        "feasible_trial_count": int(len(completed_rows)),
        "best_objective_trial": _best_by_metric(observations, key="objective", maximize=False),
        "best_diagnostic_physical_trial": _best_by_diagnostic_physical_score(observations),
        "best_density_feasible_trial": _best_by_metric(observations, key="objective", maximize=False),
        "best_pair_trial": _best_by_metric(observations, key="pair_mae_over_exact_span", maximize=False),
        "best_peak_trial": _best_by_metric(observations, key="dominant_peak_abs_omega_error", maximize=False),
        "best_drive_line_ratio_trial": _best_by_metric(
            observations,
            key="drive_line_ratio_defect",
            maximize=False,
        ),
        "best_pair_corr_trial": _best_by_metric(observations, key="pair_corr", maximize=True),
        "best_density_feasible_protected_horizon_trial": _best_by_metric_filtered(
            observations,
            key="objective",
            maximize=False,
            predicate=lambda obs: int(obs.metrics.get("protected_horizon_selection_count", 0)) > 0,
        ),
        "best_protected_horizon_trial": _best_by_metric_filtered(
            observations,
            key="objective",
            maximize=False,
            predicate=lambda obs: int(obs.metrics.get("protected_horizon_selection_count", 0)) > 0,
        ),
        "most_protected_horizon_trial": _best_by_metric(
            observations,
            key="protected_horizon_selection_count",
            maximize=True,
        ),
        "pair_runtime_frontier": _pareto_front(
            [dict(row, **dict(row.get("metrics", {}))) for row in completed_rows],
            x_key="pair_mae_over_exact_span",
            y_key="final_runtime_parameter_count",
        ),
        "peak_runtime_frontier": _pareto_front(
            [dict(row, **dict(row.get("metrics", {}))) for row in completed_rows],
            x_key="dominant_peak_abs_omega_error",
            y_key="final_runtime_parameter_count",
        ),
    }
    best_objective = payload.get("best_objective_trial")
    best_params = (
        dict(best_objective.get("params", {}))
        if isinstance(best_objective, Mapping) and isinstance(best_objective.get("params"), Mapping)
        else {}
    )
    best_diagnostic_physical = payload.get("best_diagnostic_physical_trial")
    diagnostic_physical_params = (
        dict(best_diagnostic_physical.get("params", {}))
        if isinstance(best_diagnostic_physical, Mapping)
        and isinstance(best_diagnostic_physical.get("params"), Mapping)
        else {}
    )
    if base_cfg.tuning_class:
        settings_payload = _controller_settings_payload_from_trial(
            params=TrialParams(**best_params) if best_params else _baseline_trial_params(profile=profile),
            base_cfg=base_cfg,
            route_ctx=route_ctx,
        )
        diagnostic_physical_settings_payload = (
            _controller_settings_payload_from_trial(
                params=TrialParams(**diagnostic_physical_params),
                base_cfg=base_cfg,
                route_ctx=route_ctx,
            )
            if diagnostic_physical_params
            else None
        )
        class_tuning_provenance = _optuna_class_tuning_provenance(
            base_cfg=base_cfg,
            settings_payload=settings_payload,
        )
        payload["class_tuning_provenance"] = class_tuning_provenance
        guardrail_only_candidate = _is_strict_guardrail_only_profile(base_cfg.study_profile)
        aggressive_prune_candidate = _is_strict_append_prune_aggressive_profile(base_cfg.study_profile)
        payload["class_settings_candidate"] = {
            "schema": "dynamics_class_settings_candidate_v1",
            "tuning_class": str(base_cfg.tuning_class),
            "algorithm_id": "dyn_controller_full",
            "settings_kind": "controller",
            "settings_source": str(base_cfg.class_settings_source),
            "settings_payload": settings_payload,
            "settings_id": None if class_tuning_provenance is None else class_tuning_provenance.get("settings_id"),
            "selected_trial_number": (
                None
                if not isinstance(best_objective, Mapping)
                else best_objective.get("trial_number")
            ),
            "diagnostic_physical_selected_trial_number": (
                None
                if not isinstance(best_diagnostic_physical, Mapping)
                else best_diagnostic_physical.get("trial_number")
            ),
            "diagnostic_physical_score": (
                None
                if not isinstance(best_diagnostic_physical, Mapping)
                else best_diagnostic_physical.get("metric_value")
            ),
            "diagnostic_physical_settings_payload": diagnostic_physical_settings_payload,
            "diagnostic_physical_selection_note": (
                None
                if not isinstance(best_diagnostic_physical, Mapping)
                else best_diagnostic_physical.get("selection_note")
            ),
            "source_summary_json": str(output_dir / "summary.json"),
            "study_profile": str(base_cfg.study_profile),
            "training_role": (
                "hamiltonian_specific_guardrail_candidate"
                if guardrail_only_candidate
                else (
                    "diagnostic_append_seed_prune_probe"
                    if aggressive_prune_candidate
                    else "class_policy_candidate"
                )
            ),
            "controller_settings_scope": (
                "class_lock_frozen_except_auto_euler_guardrails"
                if guardrail_only_candidate
                else (
                    "diagnostic_hh_append_seed_probe_not_promoted"
                    if aggressive_prune_candidate
                    else "class_policy_candidate"
                )
            ),
            "promotion_level": (
                "candidate_only_user_review_required"
                if guardrail_only_candidate
                else (
                    "diagnostic_only_not_paper_evidence"
                    if aggressive_prune_candidate
                    else None
                )
            ),
            "diagnostic_only_not_paper_evidence": bool(aggressive_prune_candidate),
            "class_tuned_result_locked": False,
            "strict_online_feedback_exact_free": bool(_is_strict_qpu_faithful_profile(base_cfg.study_profile)),
            "class_settings_lock": _class_settings_lock_metadata(base_cfg),
        }
    if _is_strict_qpu_faithful_profile(base_cfg.study_profile):
        observed_family = next(
            (
                str(row.get("metrics", {}).get("strict_qpu_family"))
                for row in observation_rows
                if isinstance(row.get("metrics", {}), Mapping)
                and row.get("metrics", {}).get("strict_qpu_family") not in {None, ""}
            ),
            "hh" if _is_strict_qpu_hh_profile(base_cfg.study_profile) else "generic",
        )
        flags = {
            "strict_qpu_faithful": True,
            "strict_qpu_hh": observed_family == "hh",
            "strict_qpu_family": observed_family,
        }
        payload["recoverability_reference"] = _strict_recoverability_reference(flags)
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Optuna harness for adaptive realtime checkpoint-controller tuning."
    )
    p.add_argument("--artifact-json", type=Path, required=True)
    p.add_argument(
        "--study-profile",
        choices=[
            "generic_l2_exact_v1",
            _STRICT_QPU_FAITHFUL_RECOVERABILITY_PROFILE,
            _STRICT_QPU_HH_RECOVERABILITY_PROFILE,
            _STRICT_QPU_FAITHFUL_APPEND_PRUNE_PROFILE,
            _STRICT_QPU_FAITHFUL_APPEND_PRUNE_GUARDRAIL_ONLY_PROFILE,
            _STRICT_QPU_FAITHFUL_APPEND_PRUNE_AGGRESSIVE_PROFILE,
            "append_prune_noharm_l2_v1",
            _APPEND_PRUNE_NARROW_PROFILE,
            _APPEND_PRUNE_RECOVERABILITY_PROFILE,
            "high_amp_d_shape_v1",
            "high_amp_guarded_d_shape_v1",
            "high_amp_guarded_d_shape_live_v1",
            "high_amp_guarded_d_shape_turnwindow_v1",
            "high_amp_guarded_d_shape_expanded_v1",
            "high_amp_guarded_d_shape_site3_prefsite_autobaseline_v1",
            "high_amp_guarded_fidelity_first_site3_prefsite_autobaseline_v1",
            "high_amp_guarded_site3_full_surface_v2",
        ],
        default="generic_l2_exact_v1",
    )
    p.add_argument("--tag", type=str, default=f"hh_realtime_optuna_{_timestamp_slug()}")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--n-trials", type=int, default=24)
    p.add_argument("--n-startup-trials", type=int, default=8)
    p.add_argument("--sampler-seed", type=int, default=7)
    p.add_argument("--no-baseline-trial", action="store_true")
    p.add_argument(
        "--tuning-class",
        choices=list(DYNAMICS_COARSE_TUNING_CLASSES),
        default=None,
        help="Coarse Hamiltonian class for class-tuned McLachlan/controller settings.",
    )
    p.add_argument(
        "--class-settings-source",
        type=str,
        default=DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE,
        help="Settings source label used in class-tuning provenance.",
    )
    p.add_argument(
        "--class-settings-lock-json",
        type=Path,
        default=PAPER_II_CLASS_SETTINGS_LOCK_JSON,
        help=(
            "Class-settings lock used by guardrail-only strict profiles; only "
            "auto-Euler guardrail thresholds are retuned."
        ),
    )
    p.add_argument(
        "--class-settings-output",
        type=Path,
        default=None,
        help="Optional path to write the best class-settings candidate JSON.",
    )

    p.add_argument("--loader-mode", type=str, default="replay_family")
    p.add_argument("--generator-family", type=str, default="match_adapt")
    p.add_argument("--fallback-family", type=str, default="full_meta")
    p.add_argument(
        "--append-pool-family",
        "--candidate-pool-family",
        dest="append_pool_family",
        type=str,
        default="match_replay",
    )
    p.add_argument("--lock-fixed-manifold", action="store_true")
    p.add_argument("--allow-repeats", action="store_true")
    p.add_argument("--t-final", type=float, default=T_FINAL_DEFAULT)
    p.add_argument("--num-times", type=int, default=NUM_TIMES_DEFAULT)
    drive_group = p.add_mutually_exclusive_group()
    drive_group.add_argument(
        "--enable-drive",
        action="store_true",
        default=ENABLE_DRIVE_DEFAULT,
        help="Enable the opt-in time-dependent onsite density drive.",
    )
    drive_group.add_argument(
        "--disable-drive",
        action="store_true",
        help="Deprecated compatibility flag; no flag is already static/no-drive.",
    )
    p.add_argument("--drive-A", type=float, default=DRIVE_A_DEFAULT)
    p.add_argument("--drive-omega", type=float, default=DRIVE_OMEGA_DEFAULT)
    p.add_argument("--drive-tbar", type=float, default=DRIVE_TBAR_DEFAULT)
    p.add_argument("--drive-phi", type=float, default=DRIVE_PHI_DEFAULT)
    p.add_argument("--drive-pattern", type=str, default=DRIVE_PATTERN_DEFAULT)
    p.add_argument("--drive-custom-weights", type=str, default="")
    p.add_argument("--drive-include-identity", action="store_true")
    p.add_argument("--drive-time-sampling", type=str, default=DRIVE_TIME_SAMPLING_DEFAULT)
    p.add_argument("--drive-t0", type=float, default=DRIVE_T0_DEFAULT)
    p.add_argument("--exact-steps-multiplier", type=int, default=EXACT_STEPS_MULTIPLIER_DEFAULT)

    p.add_argument("--pair", type=str, default="auto")
    p.add_argument("--objective-window-start", type=float, default=None)
    p.add_argument("--objective-window-end", type=float, default=None)
    p.add_argument("--spectra-detrend", choices=["constant", "linear"], default="linear")
    p.add_argument("--spectra-window", choices=["hann", "none"], default="hann")
    p.add_argument("--max-peaks", type=int, default=5)
    p.add_argument("--max-harmonic", type=int, default=3)
    p.add_argument(
        "--integrator-policy-override",
        choices=["euler", "rk4", "auto_euler_rk4"],
        default=None,
        help=(
            "Force every trial to use one integrator policy. Intended for local "
            "counterfactual calibration lanes; leave unset for ordinary Optuna search."
        ),
    )
    spectra_pdf_group = p.add_mutually_exclusive_group()
    spectra_pdf_group.add_argument(
        "--skip-spectra-pdf",
        dest="skip_spectra_pdf",
        action="store_true",
        help="Write spectra JSON only; skip per-trial spectra PDF rendering (default).",
    )
    spectra_pdf_group.add_argument(
        "--with-spectra-pdf",
        dest="skip_spectra_pdf",
        action="store_false",
        help="Opt in to rendering a per-trial spectra PDF for each Optuna evaluation.",
    )
    p.set_defaults(skip_spectra_pdf=True)

    p.add_argument("--objective-weight-pair-mae-over-span", type=float, default=None)
    p.add_argument("--objective-weight-epsilon-osc", type=float, default=None)
    p.add_argument("--objective-weight-peak-omega", type=float, default=None)
    p.add_argument("--objective-weight-pair-corr-defect", type=float, default=None)
    p.add_argument("--objective-weight-site-mae", type=float, default=None)
    p.add_argument("--objective-weight-energy-mae", type=float, default=None)
    p.add_argument("--objective-weight-energy-bias-abs", type=float, default=None)
    p.add_argument("--objective-weight-energy-max-abs", type=float, default=None)
    p.add_argument("--objective-weight-energy-under-response-mean", type=float, default=None)
    p.add_argument("--objective-weight-energy-under-response-max", type=float, default=None)
    p.add_argument("--objective-weight-fidelity-defect", type=float, default=None)
    p.add_argument("--objective-weight-runtime-count", type=float, default=None)
    p.add_argument("--objective-weight-append-count", type=float, default=None)
    p.add_argument("--objective-weight-prune-count", type=float, default=None)
    p.add_argument("--objective-weight-rk4-count", type=float, default=None)
    p.add_argument("--invalid-max-mean-energy-mae", type=float, default=None)
    p.add_argument("--invalid-max-final-energy-mae", type=float, default=None)
    p.add_argument("--invalid-min-fidelity", type=float, default=None)
    p.add_argument("--invalid-max-mean-total-occupation-mae", type=float, default=None)
    p.add_argument("--invalid-max-primary-observable-mae-over-span", type=float, default=None)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    optuna = _import_optuna()
    base_cfg = _build_base_run_config(args)
    objective_weights = _build_objective_weights(args)
    validity_gates = _build_validity_gates(args)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (REPO_ROOT / "artifacts" / "agent_runs" / str(args.tag))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _run_study(
        optuna=optuna,
        tag=str(args.tag),
        output_dir=output_dir,
        base_cfg=base_cfg,
        objective_weights=objective_weights,
        validity_gates=validity_gates,
        n_trials=int(max(0, args.n_trials)),
        n_startup_trials=int(max(1, args.n_startup_trials)),
        sampler_seed=int(args.sampler_seed),
        enqueue_baseline=not bool(args.no_baseline_trial),
    )
    best_trial = dict(summary.get("best_objective_trial", {}) or {})
    _write_json(output_dir / "summary.json", summary)
    if args.class_settings_output is not None:
        _write_json(
            Path(args.class_settings_output).expanduser().resolve(),
            summary.get("class_settings_candidate", {}),
        )
    _write_progress(
        output_dir,
        {
            "generated_utc": _now_utc(),
            "pipeline": _PIPELINE_NAME,
            "tag": str(args.tag),
            "output_dir": str(output_dir),
            "done": True,
            "best_objective": best_trial.get("objective"),
            "best_trial_number": best_trial.get("trial_number"),
            "completed_trial_count": int(
                sum(1 for row in summary.get("observations", []) if str(row.get("status")) == "completed")
            ),
        },
    )
    print(f"Wrote summary: {output_dir / 'summary.json'}")


__all__ = [
    "BaseRunConfig",
    "ObjectiveWeights",
    "TrialObservation",
    "TrialParams",
    "ValidityGates",
    "_STRICT_QPU_FAITHFUL_RECOVERABILITY_PROFILE",
    "_STRICT_QPU_FAITHFUL_APPEND_PRUNE_PROFILE",
    "_STRICT_QPU_FAITHFUL_APPEND_PRUNE_GUARDRAIL_ONLY_PROFILE",
    "_STRICT_QPU_FAITHFUL_APPEND_PRUNE_AGGRESSIVE_PROFILE",
    "_STRICT_QPU_HH_RECOVERABILITY_PROFILE",
    "_baseline_trial_params",
    "_build_realtime_tokens",
    "_invalid_reasons",
    "_objective_value",
    "_resolved_pair",
    "_search_space",
    "_trial_metrics_from_payload",
    "build_parser",
    "main",
]


if __name__ == "__main__":
    main()
