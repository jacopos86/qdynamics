#!/usr/bin/env python3
"""Types and identity helpers for HH adaptive realtime checkpoint control."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


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
    oracle_selection_policy: str = "measured_gain_commit_veto"
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
    exact_forecast_tracking_fidelity_defect_weight: float = 1.0
    exact_forecast_tracking_staggered_error_weight: float = 1.0
    exact_forecast_tracking_doublon_error_weight: float = 1.0
    exact_forecast_tracking_site_occupations_error_weight: float = 1.0
    exact_forecast_tracking_energy_total_error_weight: float = 1.0
    exact_forecast_energy_slope_weight: float = 0.0
    exact_forecast_energy_curvature_weight: float = 0.0
    exact_forecast_energy_excursion_under_weight: float = 0.0
    exact_forecast_energy_excursion_over_weight: float = 0.0
    exact_forecast_energy_excursion_rel_tolerance: float = 0.0
    exact_forecast_guardrail_mode: str = "off"
    exact_forecast_fidelity_loss_tol: float = 0.0
    exact_forecast_abs_energy_error_increase_tol: float = 0.0
    miss_threshold: float = 0.05
    gain_ratio_threshold: float = 0.02
    append_margin_abs: float = 1e-6
    shortlist_size: int = 4
    shortlist_fraction: float = 0.15
    active_window_size: int = 3
    max_probe_positions: int = 4
    regularization_lambda: float = 1e-8
    candidate_regularization_lambda: float = 1e-8
    pinv_rcond: float = 1e-10
    analytic_noise_std: float = 0.0
    analytic_noise_seed: int | None = None
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
    prune_state_jump_l2_tol: float = 0.05
    prune_theta_block_tol: float = 0.05
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
    gain_ratio_selected: float
    shortlist_size: int
    tier_reached: str
    logical_block_count_before: int
    logical_block_count_after: int
    runtime_parameter_count_before: int
    runtime_parameter_count_after: int
    rate_change_l2: float | None
    prune_cached_loss_selected: float | None = None
    prune_stagnation_score_selected: float | None = None
    post_prune_state_jump_l2: float | None = None
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
    oracle_decision_used: bool = False
    oracle_attempted: bool = False
    oracle_estimate_kind: str | None = None
    selection_metric: str | None = None
    decision_override_reason: str | None = None
    exact_forecast_error: str | None = None
    baseline_step_scale: float | None = None
    baseline_blend_weight: float | None = None
    baseline_gain_scale: float | None = None
    baseline_proposal_kind: str | None = None
    selected_step_scale: float | None = None
    forecast_stay_fidelity_exact_next: float | None = None
    forecast_selected_fidelity_exact_next: float | None = None
    forecast_stay_abs_energy_total_error_next: float | None = None
    forecast_selected_abs_energy_total_error_next: float | None = None
    forecast_stay_abs_staggered_error_next: float | None = None
    forecast_selected_abs_staggered_error_next: float | None = None
    forecast_stay_abs_doublon_error_next: float | None = None
    forecast_selected_abs_doublon_error_next: float | None = None
    forecast_stay_site_occupations_abs_error_max_next: float | None = None
    forecast_selected_site_occupations_abs_error_max_next: float | None = None
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
    "DerivedGeometryKey",
    "GeometryValueKey",
    "OracleValueKey",
    "RawGroupKey",
    "SharedRawGroupKey",
    "MeasurementTierConfig",
    "RealtimeCheckpointConfig",
    "ScaffoldAcceptanceResult",
    "TemporalPriorKey",
    "TemporalPriorRecord",
    "dataclass_to_payload",
    "hash_measurement_state",
    "hash_scaffold_labels",
    "hash_statevector",
    "hash_theta_vector",
    "make_checkpoint_context",
    "validate_scaffold_acceptance",
]
