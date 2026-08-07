"""Append-first AP-McLachlan trajectory propagation."""

from __future__ import annotations

import itertools
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.append_cost import (
    AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1,
    AP_APPEND_RANK_SCORE_KIND_V1,
    AppendCostSettings,
    append_cost_telemetry_for_family,
    estimate_append_atom_set_cost,
)
from pipelines.time_dynamics.ap_mclachlan.prune_cost import (
    AP_PRUNE_CONDITIONED_RANK_SCORE_KIND_V1,
    AP_PRUNE_RANK_SCORE_KIND_V1,
    PruneCostSettings,
    estimate_prune_atom_set_cost,
    prune_cost_telemetry_for_family,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import (
    FixedMcLachlanStep,
    SolveRepairConfig,
    SolveRepairUnsupportedError,
    solve_fixed_mclachlan_step,
    solve_fixed_mclachlan_step_with_repair,
)
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import (
    GeometryEvaluation,
    evaluate_mclachlan_geometry,
    geometry_evaluation_without_tangent_matrix,
    state_space_velocity_from_evaluation,
)
from pipelines.time_dynamics.ap_mclachlan.geometry import (
    residual_denominator,
    state_space_kink_eta,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.integrators import (
    INTEGRATOR_EULER,
    IntegrationStep,
    aggregate_integration_substeps,
    integrate_theta_step,
    integration_step_with_metadata,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import (
    McLachlanInversePolicy,
    supported_inverse,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    AP_PARAMETERIZATION_PER_PAULI_TERM,
    APMcLachlanState,
    normalize_parameterization_mode,
    state_with_appended_terms,
    state_without_term_labels,
)
from pipelines.time_dynamics.ap_mclachlan.support_atoms import (
    APPEND_OCCURRENCE_POLICY_LAYER_REUSE,
    APPEND_OCCURRENCE_POLICY_UNIQUE_SUPPORT,
    ActiveSupportAtom,
    SupportAtom,
    active_support_atoms,
    appended_origin_atom_labels,
    candidate_append_atoms,
    no_pauli_split_parent_labels,
    normalize_append_occurrence_policy,
    state_with_appended_atoms,
    state_with_support_patch_atoms,
    state_without_active_atoms,
)
from pipelines.time_dynamics.ap_mclachlan.support_frontier import (
    APPEND_MACRO_SCOUT_CHEAP_SCORE_MODES,
    APPEND_MACRO_SCOUT_POLICY_V2,
    APPEND_MACRO_SCOUT_SCORE_MODE_FULL_CHILD_BLOCK_DIAGNOSTIC,
    APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_LINEAR_RESIDUAL_V1,
    APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_TANGENT_SCHUR_GAIN,
    APPEND_MACRO_SCOUT_SCORE_MODES,
    SupportFrontierFailOpen,
    SupportFrontierScore,
    build_append_support_frontier,
    validate_append_macro_scout_score_mode,
)
from pipelines.time_dynamics.ap_mclachlan.support_decision import RungDiagnostics
from pipelines.time_dynamics.ap_mclachlan.support_patch import (
    PATCH_APPEND,
    PATCH_DELETE,
    PATCH_EXCHANGE,
    PATCH_INSERT,
    PATCH_NO_EDIT,
    SupportPatch,
    SupportPatchBeforeCache,
    SupportPatchGeometry,
    SupportPatchScore,
    build_support_patch_after_cache,
    build_support_patch_before_cache,
    prune_conditioning_diagnostics,
    score_support_patch,
)
from src.quantum.ansatz_parameterization import (
    build_parameter_layout,
    iter_runtime_rotation_terms,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_actions import apply_compiled_pauli, compile_pauli_action_exyz
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


ADAPTIVE_TRAJECTORY_SCHEMA_V1 = "ap_mclachlan_append_trajectory_v1"
APPEND_BATCH_SELECTION_POLICY_V1 = "max_rank_score_pool_order_tiebreak_v1"
APPEND_LADDER_SELECTION_POLICY_V1 = "cost_weighted_combinatorial_append_ladder_v1"
APPEND_LADDER_PREFILTER_POLICY_V1 = "cost_weighted_singleton_rank_score_prefilter_v1"
PRUNE_LADDER_SELECTION_POLICY_V1 = "cost_pressure_combinatorial_prune_ladder_v1"
PRUNE_LADDER_PREFILTER_POLICY_V1 = "cost_pressure_singleton_prune_prefilter_v1"
SUPPORT_PATCH_CONTROLLER_PROFILE_V1 = "support_patch_exchange_family_v1"
LEGACY_APPEND_CONTROLLER_PROFILE_V1 = "legacy_append_compat_v1"
FAILED_APPEND_REUSE_POLICY_V1 = "failed_append_search_reuse_v1"
SUPPORT_PATCH_EXCHANGE_SELECTION_POLICY_V1 = "paper_ii_unified_support_patch_exchange_v1"
LEGACY_APPEND_PATCH_KINDS = frozenset({PATCH_APPEND, PATCH_INSERT})
FAILED_APPEND_REOPEN_DIRECT = "direct_threshold"
FAILED_APPEND_REOPEN_MODEL_CHANGE = "model_change_with_direct_fallback"
DEFAULT_APPEND_RESIDUAL_RATIO_THRESHOLD = 1.0e-3
PRUNE_PERSISTENCE_EXACT_BATCH = "exact_batch"
PRUNE_PERSISTENCE_ATOM_HISTORY = "atom_history"
PRUNE_PERSISTENCE_MODES = frozenset(
    {PRUNE_PERSISTENCE_EXACT_BATCH, PRUNE_PERSISTENCE_ATOM_HISTORY}
)
PRUNE_TARGET_ALL_ACTIVE = "all_active"
PRUNE_TARGET_APPENDED_ONLY = "appended_only"
PRUNE_TARGET_REDUNDANT_APPENDED_ONLY = "redundant_appended_only"
PRUNE_TARGET_POLICIES = frozenset(
    {
        PRUNE_TARGET_ALL_ACTIVE,
        PRUNE_TARGET_APPENDED_ONLY,
        PRUNE_TARGET_REDUNDANT_APPENDED_ONLY,
    }
)


@dataclass(frozen=True)
class SupportPatchControllerConfig:
    """Schema-stable support-patch controller settings for Paper-II AP."""

    controller_profile: str = SUPPORT_PATCH_CONTROLLER_PROFILE_V1
    parameterization_mode_default: str = "per_pauli_term"
    exchange_enabled: bool = True
    branch_scoring_enabled: bool = True
    support_patch_scoring_workers: int = 1
    prune_enabled: bool = False
    prune_commit_enabled: bool = False
    append_ladder_mode: str = "combinatorial"
    append_occurrence_policy: str = APPEND_OCCURRENCE_POLICY_LAYER_REUSE
    max_append_batch_size: int = 10
    append_rung_set_cap: int = 64
    append_prefilter_size: int = 12
    append_prefilter_policy: str = APPEND_LADDER_PREFILTER_POLICY_V1
    append_gain_threshold: float = 1.0e-10
    append_batch_score_threshold: float = 1.0e-10
    append_schur_guard_enabled: bool = True
    append_schur_min_rank_fraction: float = 1.0
    append_schur_max_condition_number: float = 1.0e12
    append_schur_novelty_ridge_lambda: float = 0.0
    append_macro_scout_enabled: bool = False
    append_macro_scout_score_mode: str = APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_TANGENT_SCHUR_GAIN
    append_macro_scout_parent_cap: int = 0
    append_macro_scout_score_min: float = 0.0
    append_macro_scout_fail_open: bool = True
    append_macro_scout_expand_if_residual_high: float = 0.0
    append_macro_scout_exchange_fail_open: bool = True
    append_macro_scout_audit_parent_count: int = 0
    append_macro_scout_audit_parent_fraction: float = 0.0
    append_macro_scout_parent_cost_alpha: float = 1.0
    append_min_time: float = 0.0
    failed_append_reuse_enabled: bool = False
    failed_append_reuse_reopen_mode: str = FAILED_APPEND_REOPEN_DIRECT
    failed_append_reuse_tau_min: float = 1.0e-4
    failed_append_reuse_tau_margin_scale: float = 1.0
    failed_append_reuse_tau_max: float = 1.0
    failed_append_reuse_eta_reopen: float = 0.5
    failed_append_reuse_model_l_min: float = 1.0e-12
    failed_append_reuse_naturalization_floor: float = 1.0e-14
    failed_append_reuse_sentinel_count: int = 4
    failed_append_reuse_secant_wait_min: float = 0.0
    failed_append_reuse_secant_wait_max: float = 1.0
    failed_append_reuse_secant_wait_margin_scale: float = 1.0
    failed_append_reuse_secant_positive_safety: float = 0.5
    failed_append_reuse_secant_negative_growth: float = 2.0
    residual_ratio_threshold: float = DEFAULT_APPEND_RESIDUAL_RATIO_THRESHOLD
    max_prune_batch_size: int = 0
    prune_rung_set_cap: int = 0
    prune_prefilter_size: int = 0
    prune_loss_threshold: float = 1.0e-2
    prune_history_window: int = 3
    prune_history_lambda: float = 1.0
    prune_persistence_required: int = 1
    prune_persistence_mode: str = PRUNE_PERSISTENCE_ATOM_HISTORY
    prune_atom_history_fraction: float = 1.0
    prune_appended_origin_target_policy: str = PRUNE_TARGET_ALL_ACTIVE
    prune_cooldown_steps: int = 2
    min_runtime_parameter_count: int = 1
    prune_projection_enabled: bool = True
    prune_projection_rounds: int = 2
    prune_projection_trust_radius: float = 5.0e-2
    prune_projection_regularization: float = 1.0e-8
    prune_ray_distance_tol: float = 5.0e-2
    prune_differential_miss_tol: float = 1.0e-2
    prune_shadow_enabled: bool = True
    prune_shadow_horizon_steps: int = 2
    prune_shadow_score_tol: float = 1.0e-2
    prune_patch_smoothness_enabled: bool = True
    prune_patch_smoothness_eta_max: float = 1.0e-3
    prune_patch_smoothness_cooldown_max_steps: int = 8
    prune_patch_smoothness_severity_scale: float = 1.0
    max_prune_commits: int = 0
    max_exchange_append_branches: int = 3
    max_exchange_prune_branches: int = 3
    max_exchange_pair_count: int = 0
    exchange_append_score_min: float = 0.0
    exchange_prune_score_min: float = 0.0
    exchange_residual_dominance_tol: float = 1.0e-8
    exchange_cost_dominance_tol: float = 1.0e-8
    patch_utility_delta_weight: float = 1.0
    patch_utility_refit_weight: float = 0.0
    patch_utility_velocity_weight: float = 1.0
    patch_utility_threshold: float = 0.0
    cost_model: str = AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1
    cost_required_for_decisions: bool = False
    cost_normalization_mode: str = "family_robust_v1"
    append_cost_alpha: float = 1.0
    append_cost_lambda_2q: float = 0.05
    append_cost_lambda_d: float = 0.05
    append_cost_lambda_1q: float = 0.025
    append_cost_lambda_theta: float = 0.0
    append_cost_lambda_shot: float = 0.02
    append_cost_scale_floor: float = 1.0e-12
    prune_cost_alpha: float = 1.0
    prune_condition_lambda_kappa_rel: float = 0.0
    prune_condition_lambda_schur: float = 0.0
    prune_condition_lambda_kappa_hist: float = 0.0
    prune_condition_lambda_kappa_dam: float = 0.0
    exchange_cost_alpha: float = 1.0
    eps_loss: float = 1.0e-14
    allow_incomplete_candidate_pool: bool = False
    protect_drive_aligned_atoms: bool = True
    uses_reference_for_decision: bool = False
    uses_future_exact_forecast_for_decision: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_append_batch_size",
            "append_rung_set_cap",
            "append_prefilter_size",
            "append_macro_scout_parent_cap",
            "append_macro_scout_audit_parent_count",
            "max_prune_batch_size",
            "prune_rung_set_cap",
            "prune_prefilter_size",
            "prune_history_window",
            "prune_persistence_required",
            "prune_cooldown_steps",
            "min_runtime_parameter_count",
            "prune_projection_rounds",
            "prune_shadow_horizon_steps",
            "prune_patch_smoothness_cooldown_max_steps",
            "max_prune_commits",
            "max_exchange_append_branches",
            "max_exchange_prune_branches",
            "max_exchange_pair_count",
            "failed_append_reuse_sentinel_count",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if int(self.support_patch_scoring_workers) < 1:
            raise ValueError("support_patch_scoring_workers must be positive.")
        for name in (
            "append_gain_threshold",
            "append_batch_score_threshold",
            "append_schur_min_rank_fraction",
            "append_schur_max_condition_number",
            "append_schur_novelty_ridge_lambda",
            "append_macro_scout_score_min",
            "append_macro_scout_expand_if_residual_high",
            "append_macro_scout_audit_parent_fraction",
            "append_macro_scout_parent_cost_alpha",
            "append_min_time",
            "failed_append_reuse_tau_min",
            "failed_append_reuse_tau_margin_scale",
            "failed_append_reuse_tau_max",
            "failed_append_reuse_eta_reopen",
            "failed_append_reuse_model_l_min",
            "failed_append_reuse_naturalization_floor",
            "failed_append_reuse_secant_wait_min",
            "failed_append_reuse_secant_wait_max",
            "failed_append_reuse_secant_wait_margin_scale",
            "failed_append_reuse_secant_positive_safety",
            "failed_append_reuse_secant_negative_growth",
            "residual_ratio_threshold",
            "prune_loss_threshold",
            "prune_history_lambda",
            "prune_atom_history_fraction",
            "prune_projection_trust_radius",
            "prune_projection_regularization",
            "prune_ray_distance_tol",
            "prune_differential_miss_tol",
            "prune_shadow_score_tol",
            "prune_patch_smoothness_eta_max",
            "prune_patch_smoothness_severity_scale",
            "exchange_residual_dominance_tol",
            "exchange_cost_dominance_tol",
            "exchange_append_score_min",
            "exchange_prune_score_min",
            "patch_utility_delta_weight",
            "patch_utility_refit_weight",
            "patch_utility_velocity_weight",
            "patch_utility_threshold",
            "append_cost_alpha",
            "append_cost_lambda_2q",
            "append_cost_lambda_d",
            "append_cost_lambda_1q",
            "append_cost_lambda_theta",
            "append_cost_lambda_shot",
            "append_cost_scale_floor",
            "prune_cost_alpha",
            "prune_condition_lambda_kappa_rel",
            "prune_condition_lambda_schur",
            "prune_condition_lambda_kappa_hist",
            "prune_condition_lambda_kappa_dam",
            "exchange_cost_alpha",
            "eps_loss",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if float(self.append_schur_min_rank_fraction) > 1.0:
            raise ValueError("append_schur_min_rank_fraction must be <= 1.")
        validate_append_macro_scout_score_mode(self.append_macro_scout_score_mode)
        normalize_append_occurrence_policy(self.append_occurrence_policy)
        if float(self.append_macro_scout_audit_parent_fraction) > 1.0:
            raise ValueError("append_macro_scout_audit_parent_fraction must be <= 1.")
        if float(self.prune_atom_history_fraction) > 1.0:
            raise ValueError("prune_atom_history_fraction must be <= 1.")
        if bool(self.prune_patch_smoothness_enabled):
            if float(self.prune_patch_smoothness_eta_max) <= 0.0:
                raise ValueError("prune_patch_smoothness_eta_max must be positive.")
            if float(self.prune_patch_smoothness_severity_scale) <= 0.0:
                raise ValueError("prune_patch_smoothness_severity_scale must be positive.")
        prune_persistence_mode = str(self.prune_persistence_mode).strip().lower()
        if prune_persistence_mode not in PRUNE_PERSISTENCE_MODES:
            raise ValueError(
                "prune_persistence_mode must be one of "
                f"{sorted(PRUNE_PERSISTENCE_MODES)!r}."
            )
        prune_target_policy = str(
            self.prune_appended_origin_target_policy
        ).strip().lower()
        if prune_target_policy not in PRUNE_TARGET_POLICIES:
            raise ValueError(
                "prune_appended_origin_target_policy must be one of "
                f"{sorted(PRUNE_TARGET_POLICIES)!r}."
            )
        if float(self.failed_append_reuse_tau_min) > float(self.failed_append_reuse_tau_max):
            raise ValueError("failed_append_reuse_tau_min must be <= failed_append_reuse_tau_max.")
        if float(self.failed_append_reuse_secant_wait_min) > float(
            self.failed_append_reuse_secant_wait_max
        ):
            raise ValueError(
                "failed_append_reuse_secant_wait_min must be <= "
                "failed_append_reuse_secant_wait_max."
            )
        reopen_mode = str(self.failed_append_reuse_reopen_mode).strip().lower()
        if reopen_mode not in {
            FAILED_APPEND_REOPEN_DIRECT,
            FAILED_APPEND_REOPEN_MODEL_CHANGE,
        }:
            raise ValueError(
                "failed_append_reuse_reopen_mode must be "
                f"{FAILED_APPEND_REOPEN_DIRECT!r} or {FAILED_APPEND_REOPEN_MODEL_CHANGE!r}."
            )
        if bool(self.uses_reference_for_decision):
            raise ValueError("AP support-patch decisions must not use reference trajectories.")
        if bool(self.uses_future_exact_forecast_for_decision):
            raise ValueError("AP support-patch decisions must not use future exact forecasts.")
        if str(self.append_ladder_mode).strip().lower() == "combinatorial":
            AppendCostSettings.from_config(self)
            if bool(self.prune_enabled):
                PruneCostSettings.from_config(self)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "controller_profile": str(self.controller_profile),
            "parameterization_mode_default": str(self.parameterization_mode_default),
            "exchange_enabled": bool(self.exchange_enabled),
            "branch_scoring_enabled": bool(self.branch_scoring_enabled),
            "support_patch_scoring_workers": int(self.support_patch_scoring_workers),
            "prune_enabled": bool(self.prune_enabled),
            "prune_commit_enabled": bool(self.prune_commit_enabled),
            "append_ladder_mode": str(self.append_ladder_mode),
            "append_occurrence_policy": str(self.append_occurrence_policy),
            "max_append_batch_size": int(self.max_append_batch_size),
            "append_rung_set_cap": int(self.append_rung_set_cap),
            "append_prefilter_size": int(self.append_prefilter_size),
            "append_prefilter_policy": str(self.append_prefilter_policy),
            "append_gain_threshold": float(self.append_gain_threshold),
            "append_batch_score_threshold": float(self.append_batch_score_threshold),
            "append_schur_guard_enabled": bool(self.append_schur_guard_enabled),
            "append_schur_min_rank_fraction": float(self.append_schur_min_rank_fraction),
            "append_schur_max_condition_number": float(
                self.append_schur_max_condition_number
            ),
            "append_schur_novelty_ridge_lambda": float(
                self.append_schur_novelty_ridge_lambda
            ),
            "append_macro_scout_enabled": bool(self.append_macro_scout_enabled),
            "append_macro_scout_policy": APPEND_MACRO_SCOUT_POLICY_V2,
            "append_macro_scout_score_mode": str(self.append_macro_scout_score_mode),
            "append_macro_scout_parent_cap": int(self.append_macro_scout_parent_cap),
            "append_macro_scout_score_min": float(self.append_macro_scout_score_min),
            "append_macro_scout_fail_open": bool(self.append_macro_scout_fail_open),
            "append_macro_scout_expand_if_residual_high": float(
                self.append_macro_scout_expand_if_residual_high
            ),
            "append_macro_scout_exchange_fail_open": bool(
                self.append_macro_scout_exchange_fail_open
            ),
            "append_macro_scout_audit_parent_count": int(
                self.append_macro_scout_audit_parent_count
            ),
            "append_macro_scout_audit_parent_fraction": float(
                self.append_macro_scout_audit_parent_fraction
            ),
            "append_macro_scout_parent_cost_alpha": float(
                self.append_macro_scout_parent_cost_alpha
            ),
            "append_min_time": float(self.append_min_time),
            "failed_append_reuse_policy": FAILED_APPEND_REUSE_POLICY_V1,
            "failed_append_reuse_enabled": bool(self.failed_append_reuse_enabled),
            "failed_append_reuse_reopen_mode": str(self.failed_append_reuse_reopen_mode),
            "failed_append_reuse_tau_min": float(self.failed_append_reuse_tau_min),
            "failed_append_reuse_tau_margin_scale": float(
                self.failed_append_reuse_tau_margin_scale
            ),
            "failed_append_reuse_tau_max": float(self.failed_append_reuse_tau_max),
            "failed_append_reuse_eta_reopen": float(self.failed_append_reuse_eta_reopen),
            "failed_append_reuse_model_l_min": float(
                self.failed_append_reuse_model_l_min
            ),
            "failed_append_reuse_naturalization_floor": float(
                self.failed_append_reuse_naturalization_floor
            ),
            "failed_append_reuse_sentinel_count": int(
                self.failed_append_reuse_sentinel_count
            ),
            "failed_append_reuse_secant_wait_min": float(
                self.failed_append_reuse_secant_wait_min
            ),
            "failed_append_reuse_secant_wait_max": float(
                self.failed_append_reuse_secant_wait_max
            ),
            "failed_append_reuse_secant_wait_margin_scale": float(
                self.failed_append_reuse_secant_wait_margin_scale
            ),
            "failed_append_reuse_secant_positive_safety": float(
                self.failed_append_reuse_secant_positive_safety
            ),
            "failed_append_reuse_secant_negative_growth": float(
                self.failed_append_reuse_secant_negative_growth
            ),
            "residual_ratio_threshold": float(self.residual_ratio_threshold),
            "max_prune_batch_size": int(self.max_prune_batch_size),
            "prune_rung_set_cap": int(self.prune_rung_set_cap),
            "prune_prefilter_size": int(self.prune_prefilter_size),
            "prune_loss_threshold": float(self.prune_loss_threshold),
            "prune_history_window": int(self.prune_history_window),
            "prune_history_lambda": float(self.prune_history_lambda),
            "prune_persistence_required": int(self.prune_persistence_required),
            "prune_persistence_mode": str(self.prune_persistence_mode),
            "prune_atom_history_fraction": float(self.prune_atom_history_fraction),
            "prune_appended_origin_target_policy": str(
                self.prune_appended_origin_target_policy
            ),
            "prune_cooldown_steps": int(self.prune_cooldown_steps),
            "min_runtime_parameter_count": int(self.min_runtime_parameter_count),
            "prune_projection_enabled": bool(self.prune_projection_enabled),
            "prune_projection_rounds": int(self.prune_projection_rounds),
            "prune_projection_trust_radius": float(self.prune_projection_trust_radius),
            "prune_projection_regularization": float(self.prune_projection_regularization),
            "prune_ray_distance_tol": float(self.prune_ray_distance_tol),
            "prune_differential_miss_tol": float(self.prune_differential_miss_tol),
            "prune_shadow_enabled": bool(self.prune_shadow_enabled),
            "prune_shadow_horizon_steps": int(self.prune_shadow_horizon_steps),
            "prune_shadow_score_tol": float(self.prune_shadow_score_tol),
            "prune_patch_smoothness_enabled": bool(
                self.prune_patch_smoothness_enabled
            ),
            "prune_patch_smoothness_eta_max": float(
                self.prune_patch_smoothness_eta_max
            ),
            "prune_patch_smoothness_cooldown_max_steps": int(
                self.prune_patch_smoothness_cooldown_max_steps
            ),
            "prune_patch_smoothness_severity_scale": float(
                self.prune_patch_smoothness_severity_scale
            ),
            "max_prune_commits": int(self.max_prune_commits),
            "max_exchange_append_branches": int(self.max_exchange_append_branches),
            "max_exchange_prune_branches": int(self.max_exchange_prune_branches),
            "max_exchange_pair_count": int(self.max_exchange_pair_count),
            "exchange_append_score_min": float(self.exchange_append_score_min),
            "exchange_prune_score_min": float(self.exchange_prune_score_min),
            "exchange_residual_dominance_tol": float(self.exchange_residual_dominance_tol),
            "exchange_cost_dominance_tol": float(self.exchange_cost_dominance_tol),
            "patch_utility_delta_weight": float(self.patch_utility_delta_weight),
            "patch_utility_refit_weight": float(self.patch_utility_refit_weight),
            "patch_utility_velocity_weight": float(self.patch_utility_velocity_weight),
            "patch_utility_threshold": float(self.patch_utility_threshold),
            "cost_model": str(self.cost_model),
            "cost_required_for_decisions": bool(self.cost_required_for_decisions),
            "cost_normalization_mode": str(self.cost_normalization_mode),
            "append_cost_alpha": float(self.append_cost_alpha),
            "append_cost_lambda_2q": float(self.append_cost_lambda_2q),
            "append_cost_lambda_d": float(self.append_cost_lambda_d),
            "append_cost_lambda_1q": float(self.append_cost_lambda_1q),
            "append_cost_lambda_theta": float(self.append_cost_lambda_theta),
            "append_cost_lambda_shot": float(self.append_cost_lambda_shot),
            "append_cost_scale_floor": float(self.append_cost_scale_floor),
            "prune_cost_alpha": float(self.prune_cost_alpha),
            "prune_condition_lambda_kappa_rel": float(
                self.prune_condition_lambda_kappa_rel
            ),
            "prune_condition_lambda_schur": float(self.prune_condition_lambda_schur),
            "prune_condition_lambda_kappa_hist": float(
                self.prune_condition_lambda_kappa_hist
            ),
            "prune_condition_lambda_kappa_dam": float(
                self.prune_condition_lambda_kappa_dam
            ),
            "exchange_cost_alpha": float(self.exchange_cost_alpha),
            "eps_loss": float(self.eps_loss),
            "allow_incomplete_candidate_pool": bool(self.allow_incomplete_candidate_pool),
            "protect_drive_aligned_atoms": bool(self.protect_drive_aligned_atoms),
            "uses_reference_for_decision": bool(self.uses_reference_for_decision),
            "uses_future_exact_forecast_for_decision": bool(
                self.uses_future_exact_forecast_for_decision
            ),
        }


@dataclass(frozen=True)
class AppendControllerConfig:
    """Append controller settings for the first AP-McLachlan route."""

    max_append_candidates: int = 8
    max_prune_candidates: int = 0
    max_total_prunes: int = 0
    append_gain_threshold: float = 1.0e-10
    append_min_time: float = 0.0
    prune_loss_threshold: float = 0.0
    residual_ratio_threshold: float = DEFAULT_APPEND_RESIDUAL_RATIO_THRESHOLD
    min_logical_parameter_count: int = 1
    allow_incomplete_candidate_pool: bool = True

    def __post_init__(self) -> None:
        if int(self.max_append_candidates) < 0:
            raise ValueError("max_append_candidates must be non-negative.")
        if int(self.max_prune_candidates) < 0:
            raise ValueError("max_prune_candidates must be non-negative.")
        if int(self.max_total_prunes) < 0:
            raise ValueError("max_total_prunes must be non-negative.")
        if int(self.min_logical_parameter_count) < 0:
            raise ValueError("min_logical_parameter_count must be non-negative.")
        for name, value in (
            ("append_gain_threshold", self.append_gain_threshold),
            ("append_min_time", self.append_min_time),
            ("prune_loss_threshold", self.prune_loss_threshold),
            ("residual_ratio_threshold", self.residual_ratio_threshold),
        ):
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "max_append_candidates": int(self.max_append_candidates),
            "max_prune_candidates": int(self.max_prune_candidates),
            "max_total_prunes": int(self.max_total_prunes),
            "append_gain_threshold": float(self.append_gain_threshold),
            "append_min_time": float(self.append_min_time),
            "prune_loss_threshold": float(self.prune_loss_threshold),
            "residual_ratio_threshold": float(self.residual_ratio_threshold),
            "min_logical_parameter_count": int(self.min_logical_parameter_count),
            "allow_incomplete_candidate_pool": bool(self.allow_incomplete_candidate_pool),
        }

    def to_support_patch_config(self) -> SupportPatchControllerConfig:
        """Map legacy append settings onto the schema-stable support-patch config."""

        return SupportPatchControllerConfig(
            controller_profile=LEGACY_APPEND_CONTROLLER_PROFILE_V1,
            exchange_enabled=False,
            branch_scoring_enabled=False,
            prune_enabled=bool(int(self.max_prune_candidates) > 0),
            prune_commit_enabled=bool(int(self.max_total_prunes) > 0),
            append_ladder_mode="legacy_singleton",
            append_occurrence_policy=APPEND_OCCURRENCE_POLICY_UNIQUE_SUPPORT,
            max_append_batch_size=1,
            append_rung_set_cap=int(self.max_append_candidates),
            append_prefilter_size=int(self.max_append_candidates),
            append_prefilter_policy=APPEND_BATCH_SELECTION_POLICY_V1,
            append_gain_threshold=float(self.append_gain_threshold),
            append_batch_score_threshold=float(self.append_gain_threshold),
            append_min_time=float(self.append_min_time),
            residual_ratio_threshold=float(self.residual_ratio_threshold),
            max_prune_batch_size=1 if int(self.max_prune_candidates) > 0 else 0,
            prune_rung_set_cap=int(self.max_prune_candidates),
            prune_prefilter_size=int(self.max_prune_candidates),
            prune_loss_threshold=float(self.prune_loss_threshold),
            min_runtime_parameter_count=int(self.min_logical_parameter_count),
            cost_required_for_decisions=False,
            allow_incomplete_candidate_pool=bool(self.allow_incomplete_candidate_pool),
        )


@dataclass(frozen=True)
class PatchDecision:
    """One support-patch decision at a time point."""

    patch_kind: str
    accepted: bool
    candidate_count: int
    scored_count: int
    selected_label: str | None = None
    selected_score: SupportPatchScore | None = None
    reason: str = ""
    batch_evaluation: "PatchBatchEvaluation | None" = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "patch_kind": str(self.patch_kind),
            "accepted": bool(self.accepted),
            "candidate_count": int(self.candidate_count),
            "scored_count": int(self.scored_count),
            "selected_label": None if self.selected_label is None else str(self.selected_label),
            "selected_score": (
                None
                if self.selected_score is None
                else self.selected_score.to_json_dict()
            ),
            "reason": str(self.reason),
            "batch_evaluation": (
                None
                if self.batch_evaluation is None
                else self.batch_evaluation.to_json_dict()
            ),
            "metadata": _json_safe(dict(self.metadata or {})),
        }


@dataclass(frozen=True)
class PatchCandidateScore:
    """One frozen-time candidate score before any patch is committed."""

    candidate_kind: str
    candidate_label: str | None
    patch: SupportPatch
    score: SupportPatchScore | None
    rank_score: float | None
    accepted_eligible: bool
    rejection_reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "candidate_kind": str(self.candidate_kind),
            "candidate_label": None if self.candidate_label is None else str(self.candidate_label),
            "patch": {
                "patch_kind": str(self.patch.kind),
                "removed_runtime_indices": [
                    int(i) for i in self.patch.removed_runtime_indices
                ],
                "appended_count": int(self.patch.inserted_count),
                "appended_labels": [str(label) for label in self.patch.inserted_labels],
                "inserted_count": int(self.patch.inserted_count),
                "inserted_labels": [str(label) for label in self.patch.inserted_labels],
            },
            "score": None if self.score is None else self.score.to_json_dict(),
            "rank_score": _finite_or_none(self.rank_score),
            "accepted_eligible": bool(self.accepted_eligible),
            "rejection_reason": str(self.rejection_reason),
            "metadata": _json_safe(dict(self.metadata or {})),
        }


@dataclass(frozen=True)
class PatchBatchEvaluation:
    """All support-patch condition scores at one frozen time point."""

    time: float
    base_runtime_parameter_count: int
    base_logical_parameter_count: int
    base_residual_ratio: float
    candidate_count: int
    scored_count: int
    candidate_scores: tuple[PatchCandidateScore, ...]
    selected_index: int | None = None
    selected_score: PatchCandidateScore | None = None
    reason: str = ""
    selection_policy: str = APPEND_BATCH_SELECTION_POLICY_V1
    rung_diagnostics: tuple[RungDiagnostics, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "time": float(self.time),
            "base_runtime_parameter_count": int(self.base_runtime_parameter_count),
            "base_logical_parameter_count": int(self.base_logical_parameter_count),
            "base_residual_ratio": float(self.base_residual_ratio),
            "candidate_count": int(self.candidate_count),
            "scored_count": int(self.scored_count),
            "candidate_scores": [
                score.to_json_dict() for score in self.candidate_scores
            ],
            "selected_index": (
                None if self.selected_index is None else int(self.selected_index)
            ),
            "selected_score": (
                None if self.selected_score is None else self.selected_score.to_json_dict()
            ),
            "reason": str(self.reason),
            "selection_policy": str(self.selection_policy),
            "rung_diagnostics": [
                rung.to_json_dict() for rung in self.rung_diagnostics
            ],
            "metadata": _json_safe(dict(self.metadata or {})),
        }


@dataclass(frozen=True)
class AdaptiveTrajectoryPoint:
    """One recorded append-first AP-McLachlan time point."""

    index: int
    time: float
    theta_runtime: np.ndarray
    energy_expectation: float
    runtime_parameter_count: int
    logical_parameter_count: int
    geometry: GeometryEvaluation
    fixed_step: FixedMcLachlanStep
    patch_decision: PatchDecision
    integration_to_next: IntegrationStep | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "time": float(self.time),
            "theta_runtime": [float(x) for x in np.asarray(self.theta_runtime, dtype=float).reshape(-1)],
            "energy_expectation": float(self.energy_expectation),
            "runtime_parameter_count": int(self.runtime_parameter_count),
            "logical_parameter_count": int(self.logical_parameter_count),
            "fixed_step": self.fixed_step.to_json_dict(),
            "patch_decision": self.patch_decision.to_json_dict(),
            "integration_to_next": (
                None
                if self.integration_to_next is None
                else self.integration_to_next.to_json_dict()
            ),
        }


@dataclass(frozen=True)
class _RepairHoldState:
    """Internal return-to-base hysteresis state for a repaired inverse policy."""

    inverse_policy: McLachlanInversePolicy
    required_pass_count: int
    pass_count: int = 0
    origin_kink_eta: float | None = None


@dataclass(frozen=True)
class _LocalSubdivisionRequest:
    """Internal severity-scaled local-subdivision request."""

    reason: str
    severity: float
    min_depth: int


@dataclass(frozen=True)
class _PrunePatchSmoothnessEvaluation:
    """State-space continuity check for one deletion-containing patch."""

    available: bool
    reason: str
    eta: float | None
    eta_threshold: float
    severity: float | None
    passed: bool
    defer: bool
    velocity_jump_l2: float | None
    base_velocity_l2: float | None
    patched_velocity_l2: float | None
    denominator: float
    phase_alignment_abs_overlap: float | None
    refit_mode: str = "zero_transport_only"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "prune_patch_smoothness_available": bool(self.available),
            "prune_patch_smoothness_status": str(self.reason),
            "prune_patch_smoothness_eta": _finite_or_none(self.eta),
            "prune_patch_smoothness_eta_threshold": float(self.eta_threshold),
            "prune_patch_smoothness_severity": _finite_or_none(self.severity),
            "prune_patch_smoothness_passed": bool(self.passed),
            "prune_patch_smoothness_deferred": bool(self.defer),
            "prune_patch_smoothness_velocity_jump_l2": _finite_or_none(
                self.velocity_jump_l2
            ),
            "prune_patch_smoothness_base_velocity_l2": _finite_or_none(
                self.base_velocity_l2
            ),
            "prune_patch_smoothness_patched_velocity_l2": _finite_or_none(
                self.patched_velocity_l2
            ),
            "prune_patch_smoothness_denominator": float(self.denominator),
            "prune_patch_smoothness_phase_overlap": _finite_or_none(
                self.phase_alignment_abs_overlap
            ),
            "prune_patch_refit_mode": str(self.refit_mode),
        }


@dataclass(frozen=True)
class _DeletionPatchSafetyResult:
    """Commit authority for deletion-containing support patches."""

    passed: bool
    reason: str
    metadata: Mapping[str, Any]
    patched_state: APMcLachlanState | None = None
    theta_patched: np.ndarray | None = None
    evaluation: GeometryEvaluation | None = None
    step: FixedMcLachlanStep | None = None
    smoothness: _PrunePatchSmoothnessEvaluation | None = None


@dataclass(frozen=True)
class _PatchFinalist:
    """One checkpoint-local stay/append/prune/exchange finalist."""

    candidate: PatchCandidateScore
    utility: float | None
    passed: bool
    reason: str
    appended_atoms: tuple[SupportAtom, ...] = ()
    pruned_atoms: tuple[ActiveSupportAtom, ...] = ()
    patched_state: APMcLachlanState | None = None
    theta_patched: np.ndarray | None = None
    evaluation: GeometryEvaluation | None = None
    step: FixedMcLachlanStep | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _PruneHistoryReadView:
    """Read-only prune history snapshot for candidate scoring."""

    loss_history: Mapping[str, tuple[tuple[int, float], ...]]
    conditioning_history: Mapping[str, tuple[tuple[int, float], ...]]


@dataclass(frozen=True)
class _ExchangeScoringResult:
    """Pure exchange score before serial deletion-safety replay."""

    candidate: PatchCandidateScore
    appended_atoms: tuple[SupportAtom, ...]
    pruned_atoms: tuple[ActiveSupportAtom, ...]


@dataclass
class _PruneSmoothnessDeferredRecord:
    """Stored noncommitted prune batch that failed smoothness only."""

    candidate_key: str
    atom_ids: tuple[str, ...]
    removed_runtime_indices: tuple[int, ...]
    first_deferred_index: int
    last_deferred_index: int
    attempt_count: int
    cooldown_until_index: int
    last_eta: float | None
    last_severity: float | None
    eta_history: list[tuple[int, float]] = field(default_factory=list)
    severity_history: list[tuple[int, float]] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "prune_patch_smoothness_deferred_key": str(self.candidate_key),
            "prune_patch_smoothness_deferred_atom_ids": list(self.atom_ids),
            "prune_patch_smoothness_deferred_removed_runtime_indices": [
                int(i) for i in self.removed_runtime_indices
            ],
            "prune_patch_smoothness_first_deferred_index": int(
                self.first_deferred_index
            ),
            "prune_patch_smoothness_last_deferred_index": int(
                self.last_deferred_index
            ),
            "prune_patch_smoothness_attempt_count": int(self.attempt_count),
            "prune_patch_smoothness_cooldown_until_index": int(
                self.cooldown_until_index
            ),
            "prune_patch_smoothness_last_eta": _finite_or_none(self.last_eta),
            "prune_patch_smoothness_last_severity": _finite_or_none(
                self.last_severity
            ),
        }


@dataclass(frozen=True)
class _FailedAppendSentinelState:
    """Compact candidate-level secant state for failed append-search reuse."""

    candidate_key: str
    utility: float
    margin: float
    geometry_clock: float
    wait_path: float
    due_geometry_clock: float
    secant_slope: float | None = None


@dataclass(frozen=True)
class _FailedAppendCandidateRecord:
    """Measured rejected candidate/batch utility and margin."""

    candidate_key: str
    candidate_label: str | None
    atom_ids: tuple[str, ...]
    utility: float
    margin: float
    rejection_reason: str
    score_index: int
    rung_size: int


@dataclass(frozen=True)
class _FailedAppendCertificate:
    """Local geometry certificate created after a full append search fails."""

    certificate_id: str
    time_index: int
    time: float
    geometry_clock: float
    support_identity_hash: str
    pool_identity_hash: str
    policy_identity_hash: str
    retained_rank_signature_hash: str
    retained_rank: int
    support_dimension: int
    K: np.ndarray
    f: np.ndarray
    naturalizer: np.ndarray
    best_rejected_margin: float
    best_rejected_utility: float
    utility_change_scale: float | None
    sentinel_utility_drift: float
    sentinel_keys: tuple[str, ...]
    created_reason: str
    scored_count: int


@dataclass
class _FailedAppendReuseState:
    """Mutable append-controller state for one trajectory run."""

    certificate: _FailedAppendCertificate | None = None
    sentinels: dict[str, _FailedAppendSentinelState] = field(default_factory=dict)
    geometry_clock_available: bool = True

    def clear_certificate(self) -> None:
        self.certificate = None


@dataclass
class _PruneControllerRuntimeState:
    """Mutable active-prune state for one trajectory run."""

    support_identity_hash: str | None = None
    loss_history: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    conditioning_history: dict[str, list[tuple[int, float]]] = field(
        default_factory=dict
    )
    atom_seen_history: dict[str, list[int]] = field(default_factory=dict)
    eligible_streak: dict[str, int] = field(default_factory=dict)
    last_seen_index: dict[str, int] = field(default_factory=dict)
    cooldown_until_index: dict[str, int] = field(default_factory=dict)
    smoothness_deferred: dict[str, _PruneSmoothnessDeferredRecord] = field(
        default_factory=dict
    )
    accepted_commit_count: int = 0
    last_support_transition_metadata: dict[str, Any] = field(default_factory=dict)

    def ensure_support_identity(self, state: APMcLachlanState) -> None:
        identity = _support_identity_hash(state)
        if self.support_identity_hash != identity:
            self.clear_for_support(identity)

    def clear_for_support(self, identity: str | None = None) -> None:
        self.support_identity_hash = identity
        self.loss_history.clear()
        self.conditioning_history.clear()
        self.atom_seen_history.clear()
        self.eligible_streak.clear()
        self.last_seen_index.clear()
        self.cooldown_until_index.clear()
        self.smoothness_deferred.clear()
        self.last_support_transition_metadata = {
            "prune_history_transition": "full_clear",
            "prune_atom_history_preserved_count": 0,
            "prune_atom_history_dropped_count": 0,
            "prune_geometry_history_cleared_due_to_support_change": True,
        }

    def reset_after_support_change(self, state: APMcLachlanState) -> None:
        self.clear_for_support(_support_identity_hash(state))

    def update_after_support_change(
        self,
        *,
        new_state: APMcLachlanState,
        theta_runtime: Sequence[float] | np.ndarray | None = None,
        patch_kind: str,
    ) -> dict[str, Any]:
        active_atoms = tuple(active_support_atoms(new_state, theta_runtime))
        active_atom_ids = {
            str(atom.atom_id) for atom in active_atoms if str(atom.atom_id) != ""
        }
        active_history_ids = set(active_atom_ids)
        active_history_ids.update(
            _prune_persistence_atom_id(atom) for atom in active_atoms
        )
        before_atom_keys = set(self.atom_seen_history)
        before_cooldown_keys = set(self.cooldown_until_index)
        self.atom_seen_history = {
            str(key): list(values)
            for key, values in self.atom_seen_history.items()
            if str(key) in active_history_ids
        }
        self.cooldown_until_index = {
            str(key): int(value)
            for key, value in self.cooldown_until_index.items()
            if str(key) in active_atom_ids
        }
        dropped = len(before_atom_keys - set(self.atom_seen_history))

        self.loss_history.clear()
        self.conditioning_history.clear()
        self.eligible_streak.clear()
        self.last_seen_index.clear()
        self.smoothness_deferred.clear()
        self.support_identity_hash = _support_identity_hash(new_state)
        transition = (
            "append_preserved_atom_history"
            if str(patch_kind) == PATCH_APPEND
            else "delete_preserved_surviving_atom_history"
            if str(patch_kind) == PATCH_DELETE
            else "exchange_preserved_surviving_atom_history"
            if str(patch_kind) == PATCH_EXCHANGE
            else "support_change_preserved_atom_history"
        )
        metadata = {
            "prune_history_transition": transition,
            "prune_atom_history_preserved_count": int(len(self.atom_seen_history)),
            "prune_atom_history_dropped_count": int(dropped),
            "prune_geometry_history_cleared_due_to_support_change": True,
            "prune_cooldown_preserved_count": int(len(self.cooldown_until_index)),
            "prune_cooldown_dropped_count": int(
                len(before_cooldown_keys - set(self.cooldown_until_index))
            ),
        }
        self.last_support_transition_metadata = dict(metadata)
        return metadata


@dataclass(frozen=True)
class AppendMclachlanTrajectory:
    """Append-first AP-McLachlan trajectory output."""

    points: tuple[AdaptiveTrajectoryPoint, ...]
    integrator_method: str
    inverse_policy: McLachlanInversePolicy
    controller_config: AppendControllerConfig
    final_state: APMcLachlanState
    support_patch_config: SupportPatchControllerConfig | None = None
    solve_repair_config: SolveRepairConfig | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def final_theta_runtime(self) -> np.ndarray:
        if not self.points:
            return np.zeros(0, dtype=float)
        return np.asarray(self.points[-1].theta_runtime, dtype=float).reshape(-1)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": ADAPTIVE_TRAJECTORY_SCHEMA_V1,
            "integrator_method": str(self.integrator_method),
            "inverse_policy_id": str(self.inverse_policy.policy_id),
            "pinv_rcond": float(self.inverse_policy.pinv_rcond),
            "ridge_lambda": float(self.inverse_policy.ridge_lambda),
            "point_count": int(len(self.points)),
            "controller_config": self.controller_config.to_json_dict(),
            "support_patch_config": (
                None
                if self.support_patch_config is None
                else self.support_patch_config.to_json_dict()
            ),
            "solve_repair_config": (
                None
                if self.solve_repair_config is None
                else self.solve_repair_config.to_json_dict()
            ),
            "points": [point.to_json_dict() for point in self.points],
            "metadata": _json_safe(dict(self.metadata or {})),
        }


def run_append_mclachlan_trajectory(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    times: Sequence[float],
    inverse_policy: McLachlanInversePolicy = McLachlanInversePolicy(),
    integrator_method: str = INTEGRATOR_EULER,
    controller_config: AppendControllerConfig = AppendControllerConfig(),
    support_patch_config: SupportPatchControllerConfig | None = None,
    solve_repair_config: SolveRepairConfig = SolveRepairConfig(),
    metadata: Mapping[str, Any] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> AppendMclachlanTrajectory:
    """Propagate with an append-only support-patch controller."""

    time_grid = _time_grid(times)
    current_state = state
    theta_current = np.asarray(state.theta_runtime, dtype=float).reshape(-1)
    points: list[AdaptiveTrajectoryPoint] = []
    prune_count = 0
    repair_hold_state: _RepairHoldState | None = None
    previous_accepted_theta_dot: np.ndarray | None = None
    append_min_time = _append_min_time(
        controller_config=controller_config,
        support_patch_config=support_patch_config,
    )
    failed_append_reuse_state = _FailedAppendReuseState()
    prune_runtime_state = _PruneControllerRuntimeState()
    geometry_clock = 0.0

    for index, time_value in enumerate(time_grid):
        dt_to_next = (
            None if index + 1 >= len(time_grid) else float(time_grid[index + 1] - time_value)
        )
        evaluation = evaluate_mclachlan_geometry(
            state=current_state,
            hamiltonian=hamiltonian,
            theta_runtime=theta_current,
            time=float(time_value),
            include_tangent_matrix=(
                _use_combinatorial_append_ladder(support_patch_config)
                or _needs_prune_patch_smoothness(support_patch_config)
                or _needs_parent_macro_scout_tangent_matrix(support_patch_config)
            ),
        )
        kink_reference_theta_dot = _same_dimension_theta_dot_or_none(
            previous_accepted_theta_dot,
            int(evaluation.geometry.dimension),
        )
        fixed_step, repair_hold_state = _solve_fixed_step_with_hold_for_trajectory(
            evaluation.geometry,
            inverse_policy=inverse_policy,
            solve_repair_config=solve_repair_config,
            repair_dt=None,
            hold_state=repair_hold_state,
            kink_reference_theta_dot=kink_reference_theta_dot,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "checkpoint_start",
                    "index": int(index),
                    "time": float(time_value),
                    "energy_expectation": float(evaluation.energy_expectation),
                    "runtime_parameter_count": int(current_state.runtime_parameter_count),
                    "logical_parameter_count": int(current_state.logical_parameter_count),
                    "mclachlan_residual_ratio": float(fixed_step.residual_ratio),
                    "theta_dot_l2": float(
                        np.linalg.norm(
                            np.asarray(fixed_step.theta_dot, dtype=float).reshape(-1)
                        )
                    ),
                }
            )
        decision_inverse_policy = fixed_step.inverse_policy
        decision = PatchDecision(
            patch_kind=PATCH_NO_EDIT,
            accepted=False,
            candidate_count=0,
            scored_count=0,
            reason="append_not_considered",
        )

        if index + 1 < len(time_grid):
            if float(time_value) + 1.0e-15 < append_min_time:
                decision = PatchDecision(
                    patch_kind=PATCH_NO_EDIT,
                    accepted=False,
                    candidate_count=0,
                    scored_count=0,
                    reason="append_before_min_time",
                )
            elif _use_unified_support_patch_selector(support_patch_config):
                decision, maybe_state, maybe_theta, maybe_eval, maybe_step = _select_unified_support_patch(
                    state=current_state,
                    hamiltonian=hamiltonian,
                    theta_runtime=theta_current,
                    time=float(time_value),
                    base_evaluation=evaluation,
                    base_step=fixed_step,
                    inverse_policy=decision_inverse_policy,
                    solve_repair_config=solve_repair_config,
                    support_config=support_patch_config,
                    runtime_state=prune_runtime_state,
                    repair_dt=None,
                    time_index=int(index),
                )
            elif _use_combinatorial_append_ladder(support_patch_config):
                decision, maybe_state, maybe_theta, maybe_eval, maybe_step = _select_append_ladder_patch(
                    state=current_state,
                    hamiltonian=hamiltonian,
                    theta_runtime=theta_current,
                    time=float(time_value),
                    base_evaluation=evaluation,
                    base_step=fixed_step,
                    inverse_policy=decision_inverse_policy,
                    solve_repair_config=solve_repair_config,
                    support_config=support_patch_config,
                    repair_dt=None,
                    time_index=int(index),
                    geometry_clock=float(geometry_clock),
                    reuse_state=failed_append_reuse_state,
                )
            else:
                decision, maybe_state, maybe_theta, maybe_eval, maybe_step = _select_append_patch(
                    state=current_state,
                    hamiltonian=hamiltonian,
                    theta_runtime=theta_current,
                    time=float(time_value),
                    base_evaluation=evaluation,
                    base_step=fixed_step,
                    inverse_policy=decision_inverse_policy,
                    solve_repair_config=solve_repair_config,
                    controller_config=controller_config,
                    repair_dt=None,
                )
            if decision.accepted and maybe_state is not None and maybe_theta is not None and maybe_eval is not None and maybe_step is not None:
                failed_append_reuse_state.clear_certificate()
                current_state = maybe_state
                theta_current = maybe_theta
                evaluation = maybe_eval
                fixed_step = maybe_step
                if str(decision.patch_kind) in {PATCH_DELETE, PATCH_EXCHANGE}:
                    prune_count += 1
                    prune_runtime_state.accepted_commit_count += 1
                transition_metadata = prune_runtime_state.update_after_support_change(
                    new_state=current_state,
                    theta_runtime=theta_current,
                    patch_kind=str(decision.patch_kind),
                )
                decision = replace(
                    decision,
                    metadata={
                        **dict(decision.metadata or {}),
                        **transition_metadata,
                    },
                )

        if (
            index + 1 < len(time_grid)
            and not bool(decision.accepted)
            and not _use_unified_support_patch_selector(support_patch_config)
            and _use_active_prune_ladder(support_patch_config)
        ):
            decision, maybe_state, maybe_theta, maybe_eval, maybe_step = _select_prune_ladder_patch(
                state=current_state,
                hamiltonian=hamiltonian,
                theta_runtime=theta_current,
                time=float(time_value),
                base_evaluation=evaluation,
                base_step=fixed_step,
                inverse_policy=fixed_step.inverse_policy,
                solve_repair_config=solve_repair_config,
                support_config=support_patch_config,
                runtime_state=prune_runtime_state,
                repair_dt=None,
                time_index=int(index),
            )
            if decision.accepted and maybe_state is not None and maybe_theta is not None and maybe_eval is not None and maybe_step is not None:
                failed_append_reuse_state.clear_certificate()
                current_state = maybe_state
                theta_current = maybe_theta
                evaluation = maybe_eval
                fixed_step = maybe_step
                prune_count += 1
                prune_runtime_state.accepted_commit_count += 1
                transition_metadata = prune_runtime_state.update_after_support_change(
                    new_state=current_state,
                    theta_runtime=theta_current,
                    patch_kind=str(decision.patch_kind),
                )
                decision = replace(
                    decision,
                    metadata={
                        **dict(decision.metadata or {}),
                        **transition_metadata,
                    },
                )

        if (
            index + 1 < len(time_grid)
            and not bool(decision.accepted)
            and support_patch_config is None
            and prune_count < int(controller_config.max_total_prunes)
        ):
            decision, maybe_state, maybe_theta, maybe_eval, maybe_step = _select_prune_patch(
                state=current_state,
                hamiltonian=hamiltonian,
                theta_runtime=theta_current,
                time=float(time_value),
                base_evaluation=evaluation,
                base_step=fixed_step,
                inverse_policy=fixed_step.inverse_policy,
                solve_repair_config=solve_repair_config,
                controller_config=controller_config,
                repair_dt=None,
            )
            if decision.accepted and maybe_state is not None and maybe_theta is not None and maybe_eval is not None and maybe_step is not None:
                failed_append_reuse_state.clear_certificate()
                current_state = maybe_state
                theta_current = maybe_theta
                evaluation = maybe_eval
                fixed_step = maybe_step
                prune_count += 1

        integration: IntegrationStep | None = None
        if index + 1 < len(time_grid):
            dt = float(time_grid[index + 1] - time_value)
            state_for_rhs = current_state
            force_local_subdivision_request = _checkpoint_local_subdivision_request(
                fixed_step,
                solve_repair_config=solve_repair_config,
            )
            integration = _integrate_interval_with_repair(
                state=state_for_rhs,
                hamiltonian=hamiltonian,
                theta_runtime=theta_current,
                time=float(time_value),
                dt=dt,
                inverse_policy=inverse_policy,
                solve_repair_config=solve_repair_config,
                integrator_method=str(integrator_method),
                force_local_subdivision_request=force_local_subdivision_request,
            )
            theta_next = np.asarray(integration.theta_next, dtype=float).reshape(-1)
            path_increment = _mclachlan_path_increment(
                fixed_step=fixed_step,
                evaluation=evaluation,
                dt=dt,
            )
            if path_increment is None:
                failed_append_reuse_state.geometry_clock_available = False
            elif failed_append_reuse_state.geometry_clock_available:
                geometry_clock += float(path_increment)
        else:
            theta_next = theta_current

        point = AdaptiveTrajectoryPoint(
            index=int(index),
            time=float(time_value),
            theta_runtime=np.asarray(theta_current, dtype=float).reshape(-1),
            energy_expectation=float(evaluation.energy_expectation),
            runtime_parameter_count=int(current_state.runtime_parameter_count),
            logical_parameter_count=int(current_state.logical_parameter_count),
            geometry=geometry_evaluation_without_tangent_matrix(evaluation),
            fixed_step=fixed_step,
            patch_decision=decision,
            integration_to_next=integration,
        )
        points.append(point)
        if progress_callback is not None:
            progress_callback(_progress_payload_from_point(point))
        theta_current = np.asarray(theta_next, dtype=float).reshape(-1)
        previous_accepted_theta_dot = np.asarray(fixed_step.theta_dot, dtype=float).reshape(-1)

    return AppendMclachlanTrajectory(
        points=tuple(points),
        integrator_method=str(integrator_method).lower(),
        inverse_policy=inverse_policy,
        controller_config=controller_config,
        final_state=current_state,
        support_patch_config=support_patch_config,
        solve_repair_config=solve_repair_config,
        metadata={
            "trajectory_kind": "append_first_support_patch",
            "append_ladder_enabled": bool(
                _use_combinatorial_append_ladder(support_patch_config)
            ),
            "append_ladder_mode": (
                "legacy_singleton"
                if support_patch_config is None
                else str(support_patch_config.append_ladder_mode)
            ),
            "uses_reference_for_decision": False,
            "uses_exact_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
            "uses_statevector_as_ideal_observable_estimator": True,
            "solve_repair_enabled": bool(solve_repair_config.enabled),
            **dict(metadata or {}),
        },
    )


def _same_dimension_theta_dot_or_none(
    theta_dot: np.ndarray | Sequence[float] | None,
    dimension: int,
) -> np.ndarray | None:
    if theta_dot is None:
        return None
    theta = np.asarray(theta_dot, dtype=float).reshape(-1)
    if int(theta.size) != int(dimension):
        return None
    if not np.all(np.isfinite(theta)):
        return None
    return theta


def _solve_fixed_step_for_trajectory(
    geometry: Any,
    *,
    inverse_policy: McLachlanInversePolicy,
    solve_repair_config: SolveRepairConfig,
    repair_dt: float | None = None,
    kink_reference_theta_dot: np.ndarray | Sequence[float] | None = None,
) -> FixedMcLachlanStep:
    if bool(solve_repair_config.enabled):
        return solve_fixed_mclachlan_step_with_repair(
            geometry,
            inverse_policy=inverse_policy,
            repair_config=solve_repair_config,
            repair_dt=repair_dt,
            kink_reference_theta_dot=kink_reference_theta_dot,
        )
    return solve_fixed_mclachlan_step(
        geometry,
        inverse_policy=inverse_policy,
        step_dt=repair_dt,
        kink_reference_theta_dot=kink_reference_theta_dot,
    )


def _solve_fixed_step_with_hold_for_trajectory(
    geometry: Any,
    *,
    inverse_policy: McLachlanInversePolicy,
    solve_repair_config: SolveRepairConfig,
    repair_dt: float | None,
    hold_state: _RepairHoldState | None,
    kink_reference_theta_dot: np.ndarray | Sequence[float] | None,
) -> tuple[FixedMcLachlanStep, _RepairHoldState | None]:
    if not bool(solve_repair_config.enabled):
        return (
            solve_fixed_mclachlan_step(
                geometry,
                inverse_policy=inverse_policy,
                step_dt=repair_dt,
                kink_reference_theta_dot=kink_reference_theta_dot,
            ),
            None,
        )
    if hold_state is not None:
        try:
            base_candidate = solve_fixed_mclachlan_step_with_repair(
                geometry,
                inverse_policy=inverse_policy,
                repair_config=solve_repair_config,
                repair_dt=repair_dt,
                kink_reference_theta_dot=kink_reference_theta_dot,
            )
        except SolveRepairUnsupportedError:
            base_candidate = None
        if base_candidate is not None and not bool(base_candidate.solve_repair_applied):
            try:
                held_candidate = solve_fixed_mclachlan_step_with_repair(
                    geometry,
                    inverse_policy=hold_state.inverse_policy,
                    repair_config=solve_repair_config,
                    repair_dt=repair_dt,
                    kink_reference_theta_dot=kink_reference_theta_dot,
                )
            except SolveRepairUnsupportedError:
                return base_candidate, None
            eta_rel = _state_space_release_eta(
                geometry=geometry,
                held_step=held_candidate,
                base_step=base_candidate,
                epsilon=float(inverse_policy.epsilon),
            )
            kink_max = solve_repair_config.state_space_kink_eta_max
            release_threshold = None
            if kink_max is not None:
                release_threshold = float(kink_max) * float(
                    solve_repair_config.release_kink_threshold_scale
                )
            if release_threshold is None or eta_rel <= float(release_threshold):
                pass_count = int(hold_state.pass_count) + 1
                if pass_count >= int(hold_state.required_pass_count):
                    return base_candidate, None
                return held_candidate, _RepairHoldState(
                    inverse_policy=hold_state.inverse_policy,
                    required_pass_count=int(hold_state.required_pass_count),
                    pass_count=pass_count,
                    origin_kink_eta=hold_state.origin_kink_eta,
                )
            return held_candidate, _RepairHoldState(
                inverse_policy=hold_state.inverse_policy,
                required_pass_count=int(hold_state.required_pass_count),
                pass_count=0,
                origin_kink_eta=hold_state.origin_kink_eta,
            )
    step = solve_fixed_mclachlan_step_with_repair(
        geometry,
        inverse_policy=inverse_policy,
        repair_config=solve_repair_config,
        repair_dt=repair_dt,
        kink_reference_theta_dot=kink_reference_theta_dot,
    )
    if bool(step.solve_repair_applied):
        return step, _RepairHoldState(
            inverse_policy=step.inverse_policy,
            required_pass_count=_release_patience_count(
                kink_eta=step.state_space_kink_eta,
                repair_config=solve_repair_config,
            ),
            pass_count=0,
            origin_kink_eta=step.state_space_kink_eta,
        )
    return step, hold_state


def _state_space_release_eta(
    *,
    geometry: Any,
    held_step: FixedMcLachlanStep,
    base_step: FixedMcLachlanStep,
    epsilon: float,
) -> float:
    return state_space_kink_eta(
        geometry,
        held_step.theta_dot,
        base_step.theta_dot,
        epsilon=float(epsilon),
    )


def _release_patience_count(
    *,
    kink_eta: float | None,
    repair_config: SolveRepairConfig,
) -> int:
    p_min = int(repair_config.release_patience_min)
    p_max = int(repair_config.release_patience_max)
    if p_max <= p_min:
        return p_min
    kink_max = repair_config.state_space_kink_eta_max
    if kink_eta is None or kink_max is None:
        return p_min
    eta = float(kink_eta)
    threshold = float(kink_max)
    if not np.isfinite(eta) or not np.isfinite(threshold) or threshold <= 0.0:
        return p_max
    severity_scale = max(float(repair_config.release_kink_severity_scale), 1.0e-12)
    severity = max(0.0, math.sqrt(eta) / math.sqrt(threshold) - 1.0) / severity_scale
    return int(math.ceil(p_min + (p_max - p_min) * min(1.0, severity)))


def _integrate_interval_with_repair(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    dt: float,
    inverse_policy: McLachlanInversePolicy,
    solve_repair_config: SolveRepairConfig,
    integrator_method: str,
    force_local_subdivision_request: _LocalSubdivisionRequest | None = None,
) -> IntegrationStep:
    theta0 = np.asarray(theta_runtime, dtype=float).reshape(-1)

    def try_interval(
        local_theta: np.ndarray,
        local_t: float,
        local_dt: float,
    ) -> tuple[IntegrationStep, tuple[FixedMcLachlanStep, ...], float]:
        fixed_steps: list[FixedMcLachlanStep] = []

        def theta_dot_rhs(theta_value: np.ndarray, time_value: float) -> np.ndarray:
            evaluation = evaluate_mclachlan_geometry(
                state=state,
                hamiltonian=hamiltonian,
                theta_runtime=np.asarray(theta_value, dtype=float).reshape(-1),
                time=float(time_value),
            )
            step = _solve_fixed_step_for_trajectory(
                evaluation.geometry,
                inverse_policy=inverse_policy,
                solve_repair_config=solve_repair_config,
                repair_dt=float(local_dt),
            )
            fixed_steps.append(step)
            return np.asarray(step.theta_dot, dtype=float).reshape(-1)

        step = integrate_theta_step(
            theta=local_theta,
            t=float(local_t),
            dt=float(local_dt),
            rhs=theta_dot_rhs,
            method=str(integrator_method),
        )
        prospective_motion = _prospective_state_motion_l2_step(
            state=state,
            theta_start=local_theta,
            theta_trial=step.theta_next,
        )
        return step, tuple(fixed_steps), float(prospective_motion)

    full_step_candidate: IntegrationStep | None = None
    fixed_steps_candidate: tuple[FixedMcLachlanStep, ...] = ()
    full_prospective_motion: float | None = None
    first_reason: str | None = None
    requested_min_depth = 1
    requested_severity: float | None = None
    try:
        full_step, fixed_steps, prospective_motion = try_interval(
            theta0,
            float(time),
            float(dt),
        )
        full_step_candidate = full_step
        fixed_steps_candidate = fixed_steps
        full_prospective_motion = float(prospective_motion)
        inferred_request = _local_subdivision_request_from_steps(
            fixed_steps,
            solve_repair_config=solve_repair_config,
        )
        prospective_request = _prospective_state_motion_subdivision_request(
            prospective_motion,
            solve_repair_config=solve_repair_config,
        )
        active_request = _strongest_local_subdivision_request(
            force_local_subdivision_request,
            inferred_request,
            prospective_request,
        )
        first_reason = (
            None if active_request is None else str(active_request.reason)
        ) or _local_subdivision_reason_from_steps(fixed_steps)
        if active_request is not None:
            requested_min_depth = max(1, int(active_request.min_depth))
            requested_severity = float(active_request.severity)
        subdivision_requested = (
            active_request is not None
            or _fixed_steps_request_local_subdivision(
                fixed_steps,
                solve_repair_config=solve_repair_config,
            )
        )
        if not subdivision_requested:
            return integration_step_with_metadata(
                full_step,
                local_subdivision_applied=False,
                local_subdivision_depth=0,
                local_substep_count=1,
                local_subdivision_reason=None,
                repair_summary=_repair_summary_with_prospective_state_motion(
                    _repair_summary_from_steps(fixed_steps),
                    initial_motion=prospective_motion,
                    accepted_motions=(prospective_motion,),
                    solve_repair_config=solve_repair_config,
                ),
            )
        if not bool(solve_repair_config.local_subdivision_enabled):
            return integration_step_with_metadata(
                full_step,
                local_subdivision_applied=False,
                local_subdivision_depth=0,
                local_substep_count=1,
                local_subdivision_reason=(
                    f"solve_repair_local_subdivision_disabled:{first_reason}"
                ),
                repair_summary=_repair_summary_with_prospective_state_motion(
                    _repair_summary_with_request(
                        fixed_steps,
                        min_depth=requested_min_depth,
                        severity=requested_severity,
                    ),
                    initial_motion=prospective_motion,
                    accepted_motions=(prospective_motion,),
                    solve_repair_config=solve_repair_config,
                ),
            )
    except SolveRepairUnsupportedError as exc:
        if (
            not bool(solve_repair_config.local_subdivision_enabled)
            or not bool(exc.reducible_by_subdivision)
        ):
            raise
        first_reason = str(exc.reason)

    if full_step_candidate is None:
        try:
            (
                full_step_candidate,
                fixed_steps_candidate,
                full_prospective_motion,
            ) = try_interval(
                theta0,
                float(time),
                float(dt),
            )
        except SolveRepairUnsupportedError:
            full_step_candidate = None
            fixed_steps_candidate = ()
            full_prospective_motion = None
    if first_reason is None:
        first_reason = "state_motion_step_above_max"

    if full_step_candidate is not None and not bool(solve_repair_config.local_subdivision_enabled):
        return integration_step_with_metadata(
            full_step_candidate,
            local_subdivision_applied=False,
            local_subdivision_depth=0,
            local_substep_count=1,
            local_subdivision_reason=f"solve_repair_local_subdivision_disabled:{first_reason}",
            repair_summary=_repair_summary_with_prospective_state_motion(
                _repair_summary_with_request(
                    fixed_steps_candidate,
                    min_depth=requested_min_depth,
                    severity=requested_severity,
                ),
                initial_motion=full_prospective_motion,
                accepted_motions=(full_prospective_motion,),
                solve_repair_config=solve_repair_config,
            ),
        )
    factor = int(solve_repair_config.local_subdivision_factor)
    max_depth = int(solve_repair_config.max_local_subdivisions)
    min_dt = float(solve_repair_config.min_local_dt)
    last_error: SolveRepairUnsupportedError | None = None
    start_depth = max(1, min(max(1, int(requested_min_depth)), max_depth))
    for depth in range(start_depth, max_depth + 1):
        substep_count = int(factor**depth)
        sub_dt = float(dt) / float(substep_count)
        if abs(sub_dt) < min_dt:
            break
        local_theta = np.asarray(theta0, dtype=float).reshape(-1)
        local_t = float(time)
        substeps: list[IntegrationStep] = []
        fixed_steps_all: list[FixedMcLachlanStep] = []
        prospective_motions: list[float] = []
        prospective_motion_failed = False
        try:
            for _ in range(substep_count):
                substep, fixed_steps, prospective_motion = try_interval(
                    local_theta,
                    local_t,
                    sub_dt,
                )
                substeps.append(substep)
                fixed_steps_all.extend(fixed_steps)
                prospective_motions.append(float(prospective_motion))
                if _prospective_state_motion_above_limit(
                    prospective_motion,
                    solve_repair_config=solve_repair_config,
                ):
                    prospective_motion_failed = True
                    break
                local_theta = np.asarray(substep.theta_next, dtype=float).reshape(-1)
                local_t += sub_dt
        except SolveRepairUnsupportedError as exc:
            last_error = exc
            if not bool(exc.reducible_by_subdivision):
                break
            continue
        if prospective_motion_failed:
            continue
        if _fixed_steps_request_local_subdivision(
            fixed_steps_all,
            solve_repair_config=solve_repair_config,
        ):
            continue
        return aggregate_integration_substeps(
            theta_start=theta0,
            substeps=tuple(substeps),
            t=float(time),
            dt=float(dt),
            method=str(integrator_method),
            depth=int(depth),
            reason=_local_subdivision_reason_text(
                first_reason,
                min_depth=start_depth,
                severity=requested_severity,
            ),
            repair_summary=_repair_summary_with_prospective_state_motion(
                _repair_summary_with_request(
                    fixed_steps_all,
                    min_depth=start_depth,
                    severity=requested_severity,
                ),
                initial_motion=full_prospective_motion,
                accepted_motions=prospective_motions,
                solve_repair_config=solve_repair_config,
            ),
        )
    if full_step_candidate is not None:
        return integration_step_with_metadata(
            full_step_candidate,
            local_subdivision_applied=False,
            local_subdivision_depth=0,
            local_substep_count=1,
            local_subdivision_reason=(
                f"solve_repair_local_subdivision_not_cured:{first_reason}"
            ),
            repair_summary=_repair_summary_with_prospective_state_motion(
                _repair_summary_from_steps(fixed_steps_candidate),
                initial_motion=full_prospective_motion,
                accepted_motions=(full_prospective_motion,),
                solve_repair_config=solve_repair_config,
            ),
        )
    if last_error is not None:
        raise last_error
    raise SolveRepairUnsupportedError(
        "McLachlan local subdivision repair failed before reaching an admissible substep.",
        attempts=tuple(),
        reducible_by_subdivision=True,
        reason="local_dt_below_minimum",
    )


def _prospective_state_motion_l2_step(
    *,
    state: APMcLachlanState,
    theta_start: np.ndarray | Sequence[float],
    theta_trial: np.ndarray | Sequence[float],
) -> float:
    """Ray distance of the realized finite ANZATS update."""

    theta0 = np.asarray(theta_start, dtype=float).reshape(-1)
    theta1 = np.asarray(theta_trial, dtype=float).reshape(-1)
    if theta0.shape != theta1.shape:
        raise ValueError(
            "prospective state-motion theta shapes do not match: "
            f"{theta0.shape} vs {theta1.shape}"
        )
    if not np.all(np.isfinite(theta0)) or not np.all(np.isfinite(theta1)):
        raise ValueError("prospective state-motion theta values must be finite")
    psi0 = np.asarray(state.prepare_state(theta0), dtype=complex).reshape(-1)
    psi1 = np.asarray(state.prepare_state(theta1), dtype=complex).reshape(-1)
    norm0 = float(np.linalg.norm(psi0))
    norm1 = float(np.linalg.norm(psi1))
    if not np.isfinite(norm0) or not np.isfinite(norm1) or norm0 <= 0.0 or norm1 <= 0.0:
        raise ValueError("prospective state-motion preparation produced an invalid state")
    psi0 = psi0 / norm0
    psi1 = psi1 / norm1
    overlap = complex(np.vdot(psi0, psi1))
    motion = float(np.linalg.norm(psi1 - overlap * psi0))
    if not np.isfinite(motion):
        raise ValueError("prospective state-motion ray distance is non-finite")
    return motion


def _prospective_state_motion_above_limit(
    motion: float,
    *,
    solve_repair_config: SolveRepairConfig,
) -> bool:
    if not bool(solve_repair_config.enabled):
        return False
    limit = solve_repair_config.state_motion_l2_step_max
    if limit is None:
        return False
    motion_f = float(motion)
    limit_f = float(limit)
    if not np.isfinite(motion_f) or not np.isfinite(limit_f) or limit_f <= 0.0:
        return False
    return bool(motion_f > limit_f)


def _prospective_state_motion_subdivision_request(
    motion: float,
    *,
    solve_repair_config: SolveRepairConfig,
) -> _LocalSubdivisionRequest | None:
    if not _prospective_state_motion_above_limit(
        motion,
        solve_repair_config=solve_repair_config,
    ):
        return None
    limit = float(solve_repair_config.state_motion_l2_step_max)
    severity = float(motion) / limit
    return _LocalSubdivisionRequest(
        reason="prospective_state_motion_step_above_max",
        severity=float(severity),
        min_depth=_local_subdivision_depth_from_severity(
            severity,
            solve_repair_config=solve_repair_config,
        ),
    )


def _strongest_local_subdivision_request(
    *requests: _LocalSubdivisionRequest | None,
) -> _LocalSubdivisionRequest | None:
    available = tuple(request for request in requests if request is not None)
    if not available:
        return None
    return max(
        available,
        key=lambda request: (float(request.severity), int(request.min_depth)),
    )


def _repair_summary_with_prospective_state_motion(
    summary: Mapping[str, Any],
    *,
    initial_motion: float | None,
    accepted_motions: Sequence[float | None],
    solve_repair_config: SolveRepairConfig,
) -> dict[str, Any]:
    payload = dict(summary)
    initial = (
        None
        if initial_motion is None or not np.isfinite(float(initial_motion))
        else float(initial_motion)
    )
    accepted = [
        float(value)
        for value in accepted_motions
        if value is not None and np.isfinite(float(value))
    ]
    limit = solve_repair_config.state_motion_l2_step_max
    limit_f = (
        None
        if limit is None or not np.isfinite(float(limit))
        else float(limit)
    )
    above = bool(
        initial is not None
        and limit_f is not None
        and limit_f > 0.0
        and initial > limit_f
    )
    payload.update(
        {
            "prospective_state_motion_l2_step_initial": initial,
            "max_prospective_state_motion_l2_step": (
                None if not accepted else float(max(accepted))
            ),
            "prospective_state_motion_l2_step_max": limit_f,
            "prospective_state_motion_above_max": above,
            "prospective_state_motion_triggered": bool(
                solve_repair_config.enabled and above
            ),
        }
    )
    return payload


def _fixed_steps_request_local_subdivision(
    fixed_steps: Sequence[FixedMcLachlanStep],
    *,
    solve_repair_config: SolveRepairConfig,
) -> bool:
    if not bool(solve_repair_config.enabled):
        return False
    steps = tuple(fixed_steps)
    if not steps:
        return False
    if not any(bool(step.solve_guard_g_delta or step.solve_guard_g_kink) for step in steps):
        return False
    return not any(
        bool(
            step.solve_guard_g_empty
        )
        for step in steps
    )


def _checkpoint_local_subdivision_request(
    step: FixedMcLachlanStep,
    *,
    solve_repair_config: SolveRepairConfig,
) -> _LocalSubdivisionRequest | None:
    if not bool(step.solve_repair_enabled):
        return None
    if bool(step.solve_guard_g_empty):
        return None
    if bool(step.solve_guard_g_delta):
        reason = "state_motion_step_above_max"
    elif bool(step.solve_guard_g_kink):
        reason = "state_space_temporal_kink_above_max"
    else:
        return None
    schedule = step.solve_repair_response_schedule
    severity = (
        float(schedule.severity)
        if schedule is not None and np.isfinite(float(schedule.severity))
        else _repair_severity_for_step(step, solve_repair_config=solve_repair_config)
    )
    min_depth = (
        int(schedule.local_subdivision_breadth)
        if schedule is not None and int(schedule.local_subdivision_breadth) > 0
        else _local_subdivision_depth_from_severity(
            severity,
            solve_repair_config=solve_repair_config,
        )
    )
    return _LocalSubdivisionRequest(
        reason=reason,
        severity=float(severity),
        min_depth=max(1, min(int(solve_repair_config.max_local_subdivisions), min_depth)),
    )


def _local_subdivision_request_from_steps(
    fixed_steps: Sequence[FixedMcLachlanStep],
    *,
    solve_repair_config: SolveRepairConfig,
) -> _LocalSubdivisionRequest | None:
    if not _fixed_steps_request_local_subdivision(
        fixed_steps,
        solve_repair_config=solve_repair_config,
    ):
        return None
    reason = _local_subdivision_reason_from_steps(fixed_steps)
    if reason is None:
        return None
    severity = max(
        (_local_subdivision_request_severity(step, solve_repair_config) for step in tuple(fixed_steps)),
        default=1.0,
    )
    scheduled_depths = [
        int(step.solve_repair_response_schedule.local_subdivision_breadth)
        for step in tuple(fixed_steps)
        if step.solve_repair_response_schedule is not None
        and int(step.solve_repair_response_schedule.local_subdivision_breadth) > 0
    ]
    min_depth = (
        max(scheduled_depths)
        if scheduled_depths
        else _local_subdivision_depth_from_severity(
            severity,
            solve_repair_config=solve_repair_config,
        )
    )
    return _LocalSubdivisionRequest(
        reason=reason,
        severity=float(severity),
        min_depth=max(1, min(int(solve_repair_config.max_local_subdivisions), int(min_depth))),
    )


def _local_subdivision_request_severity(
    step: FixedMcLachlanStep,
    solve_repair_config: SolveRepairConfig,
) -> float:
    schedule = step.solve_repair_response_schedule
    if schedule is not None and int(schedule.local_subdivision_breadth) > 0:
        severity = float(schedule.severity)
        if np.isfinite(severity):
            return severity
    return _repair_severity_for_step(step, solve_repair_config=solve_repair_config)


def _repair_severity_for_step(
    step: FixedMcLachlanStep,
    *,
    solve_repair_config: SolveRepairConfig,
) -> float:
    """Fallback local subdivision severity from state-motion defects."""

    severities: list[float] = [1.0]
    if step.state_space_kink_eta is not None and solve_repair_config.state_space_kink_eta_max is not None:
        ratio = _positive_ratio(
            step.state_space_kink_eta,
            solve_repair_config.state_space_kink_eta_max,
        )
        if ratio is not None:
            severities.append(float(math.sqrt(max(0.0, ratio))))
    if step.state_motion_l2_step is not None and solve_repair_config.state_motion_l2_step_max is not None:
        ratio = _positive_ratio(
            step.state_motion_l2_step,
            solve_repair_config.state_motion_l2_step_max,
        )
        if ratio is not None:
            severities.append(float(ratio))
    return float(max(severities))


def _local_subdivision_depth_from_severity(
    severity: float,
    *,
    solve_repair_config: SolveRepairConfig,
) -> int:
    max_depth = int(solve_repair_config.max_local_subdivisions)
    if max_depth <= 0:
        return 1
    severity_f = float(severity)
    if not np.isfinite(severity_f) or severity_f <= 1.0:
        return 1
    factor = max(int(solve_repair_config.local_subdivision_factor), 2)
    depth = int(math.ceil(math.log(severity_f) / math.log(float(factor))))
    return int(max(1, min(max_depth, depth)))


def _positive_ratio(value: float, limit: float) -> float | None:
    value_f = float(value)
    limit_f = float(limit)
    if not np.isfinite(value_f) or not np.isfinite(limit_f) or limit_f <= 0.0:
        return None
    return float(max(0.0, value_f / limit_f))


def _local_subdivision_reason_text(
    reason: str | None,
    *,
    min_depth: int,
    severity: float | None,
) -> str:
    base = "unknown" if reason is None else str(reason)
    if severity is None:
        return f"solve_repair_local_subdivision:{base}"
    return (
        f"solve_repair_local_subdivision:{base}:"
        f"severity={float(severity):.6g}:min_depth={int(min_depth)}"
    )


def _repair_summary_with_request(
    fixed_steps: Sequence[FixedMcLachlanStep],
    *,
    min_depth: int,
    severity: float | None,
) -> dict[str, Any]:
    summary = _repair_summary_from_steps(fixed_steps)
    summary["local_subdivision_min_depth_requested"] = int(min_depth)
    summary["local_subdivision_severity"] = (
        None if severity is None or not np.isfinite(float(severity)) else float(severity)
    )
    return summary


def _local_subdivision_reason_from_steps(
    fixed_steps: Sequence[FixedMcLachlanStep],
) -> str | None:
    for step in tuple(fixed_steps):
        if bool(step.solve_guard_g_delta):
            return "state_motion_step_above_max"
        if bool(step.solve_guard_g_kink):
            return "state_space_temporal_kink_above_max"
    return None


def _repair_summary_from_steps(
    fixed_steps: Sequence[FixedMcLachlanStep],
) -> dict[str, Any]:
    steps = tuple(fixed_steps)

    def finite_values(name: str) -> list[float]:
        values: list[float] = []
        for step in steps:
            value = getattr(step, name, None)
            if value is None:
                continue
            out = float(value)
            if np.isfinite(out):
                values.append(out)
        return values

    condition_values = finite_values("condition_number")
    rho_num_values = finite_values("rho_num")
    state_motion_values = finite_values("state_motion_l2_step")
    kink_values = finite_values("state_space_kink_eta")
    return {
        "rhs_solve_count": int(len(steps)),
        "solve_repair_applied_count": int(
            sum(1 for step in steps if bool(step.solve_repair_applied))
        ),
        "solve_repair_unsupported_count": int(
            sum(1 for step in steps if bool(step.solve_repair_unsupported))
        ),
        "solve_repair_attempt_count": int(
            sum(len(step.solve_repair_attempts) for step in steps)
        ),
        "max_condition_number": None if not condition_values else max(condition_values),
        "max_rho_num": None if not rho_num_values else max(rho_num_values),
        "max_state_motion_l2_step": (
            None if not state_motion_values else max(state_motion_values)
        ),
        "max_state_space_kink_eta": None if not kink_values else max(kink_values),
        "effective_ridge_lambdas": [
            float(v)
            for v in sorted({float(step.inverse_policy.ridge_lambda) for step in steps})
        ],
        "effective_pinv_rconds": [
            float(v)
            for v in sorted({float(step.inverse_policy.pinv_rcond) for step in steps})
        ],
        "effective_solve_dampings": [
            float(v)
            for v in sorted({float(step.inverse_policy.solve_damping) for step in steps})
        ],
    }


def _append_min_time(
    *,
    controller_config: AppendControllerConfig,
    support_patch_config: SupportPatchControllerConfig | None,
) -> float:
    if support_patch_config is not None:
        return float(support_patch_config.append_min_time)
    return float(controller_config.append_min_time)


def _use_combinatorial_append_ladder(
    support_patch_config: SupportPatchControllerConfig | None,
) -> bool:
    if support_patch_config is None:
        return False
    mode = str(support_patch_config.append_ladder_mode).strip().lower()
    if mode in {"", "legacy_singleton"}:
        return False
    if mode == "combinatorial":
        return True
    raise ValueError(f"Unsupported append_ladder_mode: {support_patch_config.append_ladder_mode!r}.")


def _use_active_prune_ladder(
    support_patch_config: SupportPatchControllerConfig | None,
) -> bool:
    if support_patch_config is None:
        return False
    if not bool(support_patch_config.prune_enabled):
        return False
    if str(support_patch_config.append_ladder_mode).strip().lower() != "combinatorial":
        return False
    return int(support_patch_config.max_prune_batch_size) > 0


def _use_unified_support_patch_selector(
    support_patch_config: SupportPatchControllerConfig | None,
) -> bool:
    if support_patch_config is None:
        return False
    if not _use_combinatorial_append_ladder(support_patch_config):
        return False
    # Failed-append reuse is an append-only diagnostic optimization. Keep it on
    # the old append-ladder path until the reuse certificate is generalized to
    # branch-coupled stay/append/delete/exchange finalist sets.
    if (
        bool(support_patch_config.failed_append_reuse_enabled)
        and not bool(support_patch_config.prune_enabled)
    ):
        return False
    return True


def _needs_prune_patch_smoothness(
    support_patch_config: SupportPatchControllerConfig | None,
) -> bool:
    if support_patch_config is None:
        return False
    if not _use_active_prune_ladder(support_patch_config):
        return False
    return bool(
        support_patch_config.prune_commit_enabled
        and support_patch_config.prune_patch_smoothness_enabled
    )


def _needs_parent_macro_scout_tangent_matrix(
    support_patch_config: SupportPatchControllerConfig | None,
) -> bool:
    if support_patch_config is None:
        return False
    if not _use_combinatorial_append_ladder(support_patch_config):
        return False
    if not bool(support_patch_config.append_macro_scout_enabled):
        return False
    if int(support_patch_config.append_macro_scout_parent_cap) <= 0:
        return False
    mode = validate_append_macro_scout_score_mode(
        support_patch_config.append_macro_scout_score_mode
    )
    return mode in APPEND_MACRO_SCOUT_CHEAP_SCORE_MODES


def _support_patch_scoring_worker_count(
    support_config: SupportPatchControllerConfig,
    *,
    task_count: int,
) -> int:
    requested = int(support_config.support_patch_scoring_workers)
    if requested < 1:
        raise ValueError("support_patch_scoring_workers must be positive.")
    if int(task_count) <= 1 or requested <= 1:
        return 1
    return min(requested, int(task_count))


def _ordered_parallel_map(
    tasks: Sequence[Any],
    *,
    support_config: SupportPatchControllerConfig,
    score_one: Callable[[Any], Any],
) -> tuple[Any, ...]:
    task_tuple = tuple(tasks)
    worker_count = _support_patch_scoring_worker_count(
        support_config,
        task_count=len(task_tuple),
    )
    if worker_count <= 1:
        return tuple(score_one(task) for task in task_tuple)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return tuple(executor.map(score_one, task_tuple))


def _prune_history_read_view(
    runtime_state: _PruneControllerRuntimeState,
) -> _PruneHistoryReadView:
    return _PruneHistoryReadView(
        loss_history={
            str(key): tuple((int(index), float(value)) for index, value in values)
            for key, values in runtime_state.loss_history.items()
        },
        conditioning_history={
            str(key): tuple((int(index), float(value)) for index, value in values)
            for key, values in runtime_state.conditioning_history.items()
        },
    )


def _validate_append_ladder_config(
    support_config: SupportPatchControllerConfig,
) -> str:
    if str(support_config.append_ladder_mode).strip().lower() != "combinatorial":
        raise ValueError("append ladder scoring requires append_ladder_mode='combinatorial'.")
    AppendCostSettings.from_config(support_config)
    return _append_ladder_prefilter_policy_effective(
        str(support_config.append_prefilter_policy)
    )


def _validate_prune_ladder_config(
    support_config: SupportPatchControllerConfig,
) -> None:
    if str(support_config.append_ladder_mode).strip().lower() != "combinatorial":
        raise ValueError("active prune ladder requires append_ladder_mode='combinatorial'.")
    PruneCostSettings.from_config(support_config)


def _append_ladder_prefilter_policy_effective(policy: str) -> str:
    raw = str(policy).strip()
    if raw in {
        APPEND_LADDER_PREFILTER_POLICY_V1,
        APPEND_BATCH_SELECTION_POLICY_V1,
        "neutral_singleton_rank_score_prefilter_v1",
        "cost_weighted_singleton_score_v1",
    }:
        return APPEND_LADDER_PREFILTER_POLICY_V1
    raise ValueError(f"Unsupported append_prefilter_policy for diagnostic ladder: {policy!r}.")


def _failed_append_reuse_enabled(
    support_config: SupportPatchControllerConfig,
) -> bool:
    return bool(support_config.failed_append_reuse_enabled)


def _metadata_with_failed_append_reuse(
    metadata: Mapping[str, Any],
    reuse_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    out = dict(metadata)
    if reuse_metadata is not None:
        out["failed_append_reuse"] = _json_safe(dict(reuse_metadata))
    return out


def _batch_with_failed_append_reuse_metadata(
    batch: PatchBatchEvaluation,
    reuse_metadata: Mapping[str, Any],
) -> PatchBatchEvaluation:
    return replace(
        batch,
        metadata=_metadata_with_failed_append_reuse(
            dict(batch.metadata or {}),
            reuse_metadata,
        ),
    )


def _failed_append_reuse_pre_search_decision(
    *,
    state: APMcLachlanState,
    atoms: Sequence[SupportAtom],
    support_config: SupportPatchControllerConfig,
    inverse_policy: McLachlanInversePolicy,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    time: float,
    time_index: int,
    geometry_clock: float,
    reuse_state: _FailedAppendReuseState | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "policy": FAILED_APPEND_REUSE_POLICY_V1,
        "enabled": bool(_failed_append_reuse_enabled(support_config)),
        "time": float(time),
        "time_index": int(time_index),
        "geometry_clock": _finite_or_none(geometry_clock),
        "full_search_run": True,
        "reopen_mode": str(support_config.failed_append_reuse_reopen_mode),
        "reopen_route": "disabled",
        "certificate_present": False,
    }
    if not _failed_append_reuse_enabled(support_config):
        return metadata
    if reuse_state is None:
        metadata["reopen_route"] = "invalidated_fail_open"
        metadata["invalidation_reason"] = "missing_reuse_state"
        return metadata
    if not bool(reuse_state.geometry_clock_available):
        reuse_state.clear_certificate()
        metadata["reopen_route"] = "invalidated_fail_open"
        metadata["invalidation_reason"] = "nonfinite_geometry_clock"
        return metadata

    certificate = reuse_state.certificate
    if certificate is None:
        metadata["reopen_route"] = "no_certificate"
        return metadata

    metadata["certificate_present"] = True
    metadata["certificate_id"] = str(certificate.certificate_id)
    identity = _failed_append_reuse_identity(
        state=state,
        atoms=atoms,
        support_config=support_config,
        inverse_policy=inverse_policy,
        base_step=base_step,
    )
    mismatch = _failed_append_identity_mismatch(identity, certificate)
    if mismatch is not None:
        reuse_state.clear_certificate()
        metadata.update(
            {
                "reopen_route": "invalidated_fail_open",
                "invalidation_reason": str(mismatch),
            }
        )
        return metadata

    drift = _failed_append_certificate_drift(
        certificate=certificate,
        base_evaluation=base_evaluation,
        geometry_clock=float(geometry_clock),
    )
    if drift is None:
        reuse_state.clear_certificate()
        metadata.update(
            {
                "reopen_route": "invalidated_fail_open",
                "invalidation_reason": "nonfinite_certificate_drift",
            }
        )
        return metadata

    margin = max(float(certificate.best_rejected_margin), float(support_config.eps_loss))
    tau_reopen = _failed_append_tau_reopen(support_config, margin=margin)
    d_cert = float(drift["D_cert"])
    direct_reopen = bool(d_cert >= float(tau_reopen))
    l_scale = certificate.utility_change_scale
    l_useful = (
        l_scale is not None
        and np.isfinite(float(l_scale))
        and float(l_scale) >= float(support_config.failed_append_reuse_model_l_min)
    )
    model_threshold = float(support_config.failed_append_reuse_eta_reopen) * margin
    model_product = None if not l_useful else float(l_scale) * d_cert
    model_reopen = bool(l_useful and model_product is not None and model_product >= model_threshold)
    mode = str(support_config.failed_append_reuse_reopen_mode).strip().lower()
    if mode == FAILED_APPEND_REOPEN_MODEL_CHANGE:
        if l_useful:
            full_search_run = bool(model_reopen)
            route = "model_change" if full_search_run else "skipped"
            reopen_basis = "model_change"
        else:
            full_search_run = bool(direct_reopen)
            route = "fallback_direct" if full_search_run else "skipped"
            reopen_basis = "fallback_direct"
    else:
        full_search_run = bool(direct_reopen)
        route = "direct" if full_search_run else "skipped"
        reopen_basis = "direct"

    sentinel_due_count = sum(
        1
        for sentinel in reuse_state.sentinels.values()
        if float(geometry_clock) >= float(sentinel.due_geometry_clock)
    )
    metadata.update(
        {
            **drift,
            "full_search_run": bool(full_search_run),
            "reopen_route": route,
            "reopen_basis": reopen_basis,
            "best_rejected_margin": float(certificate.best_rejected_margin),
            "best_rejected_utility": float(certificate.best_rejected_utility),
            "tau_reopen": float(tau_reopen),
            "direct_reopen": bool(direct_reopen),
            "model_change_scale": _finite_or_none(l_scale),
            "model_change_scale_useful": bool(l_useful),
            "eta_reopen": float(support_config.failed_append_reuse_eta_reopen),
            "model_change_threshold": float(model_threshold),
            "model_change_product": _finite_or_none(model_product),
            "model_change_reopen": bool(model_reopen),
            "sentinel_due_count": int(sentinel_due_count),
            "sentinel_due_is_advisory": True,
        }
    )
    return metadata


def _record_failed_append_ladder_search(
    *,
    batch: PatchBatchEvaluation,
    state: APMcLachlanState,
    atoms: Sequence[SupportAtom],
    support_config: SupportPatchControllerConfig,
    inverse_policy: McLachlanInversePolicy,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    time: float,
    time_index: int,
    geometry_clock: float,
    reuse_state: _FailedAppendReuseState | None,
    pre_search_metadata: Mapping[str, Any] | None,
    failure_reason: str,
) -> tuple[PatchBatchEvaluation, dict[str, Any]]:
    metadata = dict(pre_search_metadata or {})
    metadata.update(
        {
            "policy": FAILED_APPEND_REUSE_POLICY_V1,
            "enabled": bool(_failed_append_reuse_enabled(support_config)),
            "full_search_run": True,
            "full_search_failure_reason": str(failure_reason),
            "certificate_created": False,
        }
    )
    if not _failed_append_reuse_enabled(support_config) or reuse_state is None:
        return _batch_with_failed_append_reuse_metadata(batch, metadata), metadata

    records = _failed_append_candidate_records(
        batch=batch,
        support_config=support_config,
    )
    if not records:
        reuse_state.clear_certificate()
        metadata["certificate_failure_reason"] = "no_finite_rejected_candidate_records"
        return _batch_with_failed_append_reuse_metadata(batch, metadata), metadata

    naturalized = _failed_append_naturalizer(
        np.asarray(base_evaluation.geometry.K, dtype=float),
        policy=inverse_policy,
        floor=float(support_config.failed_append_reuse_naturalization_floor),
    )
    if naturalized is None:
        reuse_state.clear_certificate()
        metadata["certificate_failure_reason"] = "naturalizer_unavailable"
        return _batch_with_failed_append_reuse_metadata(batch, metadata), metadata
    naturalizer, retained_rank = naturalized
    identity = _failed_append_reuse_identity(
        state=state,
        atoms=atoms,
        support_config=support_config,
        inverse_policy=inverse_policy,
        base_step=base_step,
    )
    sentinels, secant_slopes, sentinel_utility_drift = _update_failed_append_sentinels(
        records=records,
        support_config=support_config,
        reuse_state=reuse_state,
        geometry_clock=float(geometry_clock),
    )
    best_margin = min(float(record.margin) for record in records)
    best_utility = max(float(record.utility) for record in records)
    utility_change_scale = None
    if secant_slopes:
        utility_change_scale = max(abs(float(value)) for value in secant_slopes)
    certificate_payload = {
        **identity,
        "time": float(time),
        "time_index": int(time_index),
        "geometry_clock": float(geometry_clock),
        "best_rejected_margin": float(best_margin),
        "best_rejected_utility": float(best_utility),
        "sentinel_keys": [str(s.candidate_key) for s in sentinels],
        "scored_count": int(batch.scored_count),
    }
    certificate = _FailedAppendCertificate(
        certificate_id=_stable_json_hash(certificate_payload),
        time_index=int(time_index),
        time=float(time),
        geometry_clock=float(geometry_clock),
        support_identity_hash=str(identity["support_identity_hash"]),
        pool_identity_hash=str(identity["pool_identity_hash"]),
        policy_identity_hash=str(identity["policy_identity_hash"]),
        retained_rank_signature_hash=str(identity["retained_rank_signature_hash"]),
        retained_rank=int(retained_rank),
        support_dimension=int(np.asarray(base_evaluation.geometry.K).shape[0]),
        K=np.asarray(base_evaluation.geometry.K, dtype=float),
        f=np.asarray(base_evaluation.geometry.f, dtype=float).reshape(-1),
        naturalizer=np.asarray(naturalizer, dtype=float),
        best_rejected_margin=float(best_margin),
        best_rejected_utility=float(best_utility),
        utility_change_scale=utility_change_scale,
        sentinel_utility_drift=float(sentinel_utility_drift),
        sentinel_keys=tuple(str(s.candidate_key) for s in sentinels),
        created_reason=str(failure_reason),
        scored_count=int(batch.scored_count),
    )
    reuse_state.certificate = certificate
    metadata.update(
        {
            "certificate_created": True,
            "certificate_id": str(certificate.certificate_id),
            "best_rejected_margin": float(best_margin),
            "best_rejected_utility": float(best_utility),
            "model_change_scale": _finite_or_none(utility_change_scale),
            "sentinel_count": int(len(sentinels)),
            "sentinel_utility_drift": float(sentinel_utility_drift),
            "retained_rank": int(retained_rank),
            "support_dimension": int(certificate.support_dimension),
        }
    )
    return _batch_with_failed_append_reuse_metadata(batch, metadata), metadata


def _accepted_failed_append_reuse_metadata(
    pre_search_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict(pre_search_metadata or {})
    if metadata.get("enabled"):
        metadata.update(
            {
                "full_search_run": True,
                "certificate_created": False,
                "certificate_cleared_by_accept": True,
            }
        )
    return metadata


def _failed_append_reuse_identity(
    *,
    state: APMcLachlanState,
    atoms: Sequence[SupportAtom],
    support_config: SupportPatchControllerConfig,
    inverse_policy: McLachlanInversePolicy,
    base_step: FixedMcLachlanStep,
) -> dict[str, str]:
    support_payload = {
        "parameterization_mode": str(state.parameterization_mode),
        "runtime_coordinate_labels": [str(v) for v in state.runtime_coordinate_labels],
        "runtime_parameter_count": int(state.runtime_parameter_count),
        "logical_parameter_count": int(state.logical_parameter_count),
    }
    pool_payload = {
        "candidate_pool_source": _candidate_pool_source_payload(state),
        "atom_identities": [_support_atom_identity(atom) for atom in tuple(atoms)],
    }
    policy_payload = {
        "append_ladder_mode": str(support_config.append_ladder_mode),
        "max_append_batch_size": int(support_config.max_append_batch_size),
        "append_rung_set_cap": int(support_config.append_rung_set_cap),
        "append_prefilter_size": int(support_config.append_prefilter_size),
        "append_prefilter_policy": str(support_config.append_prefilter_policy),
        "append_gain_threshold": float(support_config.append_gain_threshold),
        "append_batch_score_threshold": float(support_config.append_batch_score_threshold),
        "append_schur_guard_enabled": bool(support_config.append_schur_guard_enabled),
        "append_schur_min_rank_fraction": float(
            support_config.append_schur_min_rank_fraction
        ),
        "append_schur_max_condition_number": float(
            support_config.append_schur_max_condition_number
        ),
        "append_schur_novelty_ridge_lambda": float(
            support_config.append_schur_novelty_ridge_lambda
        ),
        "cost_model": str(support_config.cost_model),
        "cost_normalization_mode": str(support_config.cost_normalization_mode),
        "append_cost_alpha": float(support_config.append_cost_alpha),
        "append_cost_lambda_2q": float(support_config.append_cost_lambda_2q),
        "append_cost_lambda_d": float(support_config.append_cost_lambda_d),
        "append_cost_lambda_1q": float(support_config.append_cost_lambda_1q),
        "append_cost_lambda_theta": float(support_config.append_cost_lambda_theta),
        "append_cost_lambda_shot": float(support_config.append_cost_lambda_shot),
        "append_cost_scale_floor": float(support_config.append_cost_scale_floor),
        "inverse_policy_id": str(inverse_policy.policy_id),
        "pinv_rcond": float(inverse_policy.pinv_rcond),
        "ridge_lambda": float(inverse_policy.ridge_lambda),
        "solve_damping": float(inverse_policy.solve_damping),
        "epsilon": float(inverse_policy.epsilon),
    }
    retained_payload = {
        "support_dimension": int(len(state.runtime_coordinate_labels)),
        "retained_rank": int(base_step.rank),
        "inverse_policy_id": str(base_step.inverse_policy.policy_id),
        "pinv_rcond": float(base_step.inverse_policy.pinv_rcond),
        "ridge_lambda": float(base_step.inverse_policy.ridge_lambda),
        "solve_damping": float(base_step.inverse_policy.solve_damping),
    }
    return {
        "support_identity_hash": _stable_json_hash(support_payload),
        "pool_identity_hash": _stable_json_hash(pool_payload),
        "policy_identity_hash": _stable_json_hash(policy_payload),
        "retained_rank_signature_hash": _stable_json_hash(retained_payload),
    }


def _failed_append_identity_mismatch(
    identity: Mapping[str, str],
    certificate: _FailedAppendCertificate,
) -> str | None:
    for field_name in (
        "support_identity_hash",
        "pool_identity_hash",
        "policy_identity_hash",
        "retained_rank_signature_hash",
    ):
        if str(identity.get(field_name)) != str(getattr(certificate, field_name)):
            return field_name
    return None


def _failed_append_certificate_drift(
    *,
    certificate: _FailedAppendCertificate,
    base_evaluation: GeometryEvaluation,
    geometry_clock: float,
) -> dict[str, Any] | None:
    K = np.asarray(base_evaluation.geometry.K, dtype=float)
    f = np.asarray(base_evaluation.geometry.f, dtype=float).reshape(-1)
    K0 = np.asarray(certificate.K, dtype=float)
    f0 = np.asarray(certificate.f, dtype=float).reshape(-1)
    N = np.asarray(certificate.naturalizer, dtype=float)
    if K.shape != K0.shape or N.shape != K0.shape or f.shape != f0.shape:
        return None
    if not (
        np.all(np.isfinite(K))
        and np.all(np.isfinite(f))
        and np.all(np.isfinite(K0))
        and np.all(np.isfinite(f0))
        and np.all(np.isfinite(N))
    ):
        return None
    try:
        dK = float(np.linalg.norm(N @ (0.5 * ((K - K0) + (K - K0).T)) @ N, ord=2))
    except np.linalg.LinAlgError:
        return None
    f0_nat = np.asarray(N @ f0, dtype=float).reshape(-1)
    df_nat = np.asarray(N @ (f - f0), dtype=float).reshape(-1)
    f_norm = float(np.linalg.norm(f0_nat))
    df = float(np.linalg.norm(df_nat) / max(1.0, f_norm))
    d_path = float(max(0.0, float(geometry_clock) - float(certificate.geometry_clock)))
    d_sentinel = float(max(0.0, certificate.sentinel_utility_drift))
    values = (dK, df, d_path, d_sentinel)
    if any(not np.isfinite(value) for value in values):
        return None
    return {
        "D_cert": float(max(values)),
        "D_cert_gram": float(dK),
        "D_cert_force": float(df),
        "D_cert_path": float(d_path),
        "D_cert_sentinel": float(d_sentinel),
    }


def _failed_append_tau_reopen(
    support_config: SupportPatchControllerConfig,
    *,
    margin: float,
) -> float:
    margin_value = max(0.0, float(margin))
    tau = float(support_config.failed_append_reuse_tau_min) + float(
        support_config.failed_append_reuse_tau_margin_scale
    ) * margin_value
    return float(
        min(
            float(support_config.failed_append_reuse_tau_max),
            max(float(support_config.failed_append_reuse_tau_min), tau),
        )
    )


def _failed_append_candidate_records(
    *,
    batch: PatchBatchEvaluation,
    support_config: SupportPatchControllerConfig,
) -> tuple[_FailedAppendCandidateRecord, ...]:
    records: list[_FailedAppendCandidateRecord] = []
    for candidate in tuple(batch.candidate_scores):
        if bool(candidate.accepted_eligible):
            continue
        utility = _failed_append_candidate_utility(candidate)
        margin = _failed_append_candidate_margin(candidate, support_config=support_config)
        if utility is None or margin is None:
            continue
        metadata = dict(candidate.metadata or {})
        atom_ids = tuple(str(atom_id) for atom_id in metadata.get("atom_ids", ()))
        candidate_key = _failed_append_candidate_key(candidate)
        records.append(
            _FailedAppendCandidateRecord(
                candidate_key=candidate_key,
                candidate_label=candidate.candidate_label,
                atom_ids=tuple(sorted(atom_ids)),
                utility=float(utility),
                margin=float(margin),
                rejection_reason=str(candidate.rejection_reason),
                score_index=int(metadata.get("score_index", len(records))),
                rung_size=int(metadata.get("rung_size", 0)),
            )
        )
    records.sort(
        key=lambda record: (
            float(record.margin),
            -float(record.utility),
            int(record.score_index),
            str(record.candidate_key),
        )
    )
    return tuple(records)


def _failed_append_candidate_utility(
    candidate: PatchCandidateScore,
) -> float | None:
    if candidate.rank_score is not None and np.isfinite(float(candidate.rank_score)):
        return float(candidate.rank_score)
    score = candidate.score
    if score is not None and score.rank_score is not None and np.isfinite(float(score.rank_score)):
        return float(score.rank_score)
    return None


def _failed_append_candidate_margin(
    candidate: PatchCandidateScore,
    *,
    support_config: SupportPatchControllerConfig,
) -> float | None:
    score = candidate.score
    rank_score = candidate.rank_score
    insertion_gain = None if score is None else score.insertion_gain
    margins: list[float] = []
    if rank_score is not None and np.isfinite(float(rank_score)):
        margins.append(
            max(
                0.0,
                float(support_config.append_batch_score_threshold) - float(rank_score),
            )
        )
    if insertion_gain is not None and np.isfinite(float(insertion_gain)):
        margins.append(
            max(
                0.0,
                float(support_config.append_gain_threshold) - float(insertion_gain),
            )
        )
    if not margins:
        return None
    if str(candidate.rejection_reason) not in {
        "append_gain_below_threshold",
        "append_batch_score_below_threshold",
        "nonfinite_rank_score",
        "missing_insertion_gain",
        "candidate_scoring_failed",
    }:
        margins.append(float(support_config.eps_loss))
    margin = max(float(value) for value in margins)
    if margin <= 0.0 and not bool(candidate.accepted_eligible):
        margin = float(support_config.eps_loss)
    return margin if np.isfinite(float(margin)) else None


def _failed_append_candidate_key(candidate: PatchCandidateScore) -> str:
    metadata = dict(candidate.metadata or {})
    atom_ids = tuple(str(atom_id) for atom_id in metadata.get("atom_ids", ()))
    if atom_ids:
        return "atoms:" + "|".join(sorted(atom_ids))
    labels = tuple(str(label) for label in candidate.patch.inserted_labels)
    if labels:
        return "labels:" + "|".join(sorted(labels))
    return f"score_index:{int(metadata.get('score_index', -1))}"


def _update_failed_append_sentinels(
    *,
    records: Sequence[_FailedAppendCandidateRecord],
    support_config: SupportPatchControllerConfig,
    reuse_state: _FailedAppendReuseState,
    geometry_clock: float,
) -> tuple[tuple[_FailedAppendSentinelState, ...], tuple[float, ...], float]:
    limit = int(support_config.failed_append_reuse_sentinel_count)
    if limit <= 0:
        return tuple(), tuple(), 0.0
    selected_records = tuple(records)[:limit]
    updated: list[_FailedAppendSentinelState] = []
    slopes: list[float] = []
    utility_drifts: list[float] = []
    for record in selected_records:
        previous = reuse_state.sentinels.get(record.candidate_key)
        secant = None
        if previous is not None:
            ds = float(geometry_clock) - float(previous.geometry_clock)
            if ds > 0.0 and np.isfinite(ds):
                secant = (float(record.utility) - float(previous.utility)) / ds
                if np.isfinite(secant):
                    slopes.append(float(secant))
                    utility_drifts.append(abs(float(record.utility) - float(previous.utility)))
                else:
                    secant = None
        wait = _failed_append_secant_wait(
            margin=float(record.margin),
            secant_slope=secant,
            previous=previous,
            support_config=support_config,
        )
        sentinel = _FailedAppendSentinelState(
            candidate_key=str(record.candidate_key),
            utility=float(record.utility),
            margin=float(record.margin),
            geometry_clock=float(geometry_clock),
            wait_path=float(wait),
            due_geometry_clock=float(geometry_clock) + float(wait),
            secant_slope=secant,
        )
        reuse_state.sentinels[str(record.candidate_key)] = sentinel
        updated.append(sentinel)
    utility_drift = max(utility_drifts) if utility_drifts else 0.0
    return tuple(updated), tuple(slopes), float(utility_drift)


def _failed_append_secant_wait(
    *,
    margin: float,
    secant_slope: float | None,
    previous: _FailedAppendSentinelState | None,
    support_config: SupportPatchControllerConfig,
) -> float:
    wait_min = float(support_config.failed_append_reuse_secant_wait_min)
    wait_max = float(support_config.failed_append_reuse_secant_wait_max)
    scale = max(
        float(support_config.failed_append_reuse_secant_wait_margin_scale),
        float(support_config.eps_loss),
    )
    margin_value = max(0.0, float(margin))
    base_wait = wait_min + (wait_max - wait_min) * margin_value / (margin_value + scale)
    wait = float(base_wait)
    if secant_slope is not None and np.isfinite(float(secant_slope)):
        slope = float(secant_slope)
        if slope > 0.0:
            wait = min(
                wait,
                max(
                    wait_min,
                    float(support_config.failed_append_reuse_secant_positive_safety)
                    * margin_value
                    / max(slope, float(support_config.eps_loss)),
                ),
            )
        elif previous is not None and slope < 0.0:
            wait = max(
                wait,
                min(
                    wait_max,
                    float(support_config.failed_append_reuse_secant_negative_growth)
                    * float(previous.wait_path),
                ),
            )
    return float(min(wait_max, max(wait_min, wait)))


def _failed_append_naturalizer(
    K: np.ndarray,
    *,
    policy: McLachlanInversePolicy,
    floor: float,
) -> tuple[np.ndarray, int] | None:
    matrix = np.asarray(K, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.all(np.isfinite(matrix)):
        return None
    if int(matrix.shape[0]) == 0:
        return np.zeros((0, 0), dtype=float), 0
    mat = 0.5 * (matrix + matrix.T)
    ridge = float(policy.ridge_lambda)
    if ridge != 0.0:
        mat = mat + ridge * np.eye(int(mat.shape[0]), dtype=float)
    try:
        eigenvalues, vectors = np.linalg.eigh(mat)
    except np.linalg.LinAlgError:
        return None
    abs_eigs = np.abs(eigenvalues)
    max_abs = float(np.max(abs_eigs)) if abs_eigs.size else 0.0
    threshold = float(policy.pinv_rcond) * max_abs
    retained = abs_eigs > threshold
    if not np.any(retained):
        return np.zeros_like(mat), 0
    inv_sqrt = np.zeros_like(eigenvalues, dtype=float)
    inv_sqrt[retained] = 1.0 / np.sqrt(
        np.maximum(abs_eigs[retained], float(floor))
    )
    naturalizer = (vectors * inv_sqrt) @ vectors.T
    if not np.all(np.isfinite(naturalizer)):
        return None
    return np.asarray(naturalizer, dtype=float), int(np.count_nonzero(retained))


def _candidate_pool_source_payload(state: APMcLachlanState) -> Any:
    source = getattr(state, "candidate_pool_source", None)
    if source is None:
        return None
    to_json = getattr(source, "to_json_dict", None)
    if callable(to_json):
        return to_json()
    if hasattr(source, "__dict__"):
        return dict(getattr(source, "__dict__"))
    return str(source)


def _support_atom_identity(atom: SupportAtom) -> dict[str, Any]:
    metadata = dict(atom.metadata or {})
    return {
        "atom_id": str(atom.atom_id),
        "atom_label": str(atom.atom_label),
        "base_atom_id": str(metadata.get("base_atom_id", atom.atom_id)),
        "base_atom_label": str(metadata.get("base_atom_label", atom.atom_label)),
        "occurrence_index": int(metadata.get("occurrence_index", 1)),
        "parent_label": str(atom.parent_label),
        "runtime_coordinate_count": int(atom.runtime_count),
    }


def _stable_json_hash(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    text = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _support_identity_hash(state: APMcLachlanState) -> str:
    return _stable_json_hash(
        {
            "parameterization_mode": str(state.parameterization_mode),
            "runtime_coordinate_labels": [
                str(label) for label in state.runtime_coordinate_labels
            ],
        }
    )


def _select_append_ladder_patch(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    inverse_policy: McLachlanInversePolicy,
    solve_repair_config: SolveRepairConfig,
    support_config: SupportPatchControllerConfig,
    repair_dt: float | None = None,
    time_index: int = 0,
    geometry_clock: float = 0.0,
    reuse_state: _FailedAppendReuseState | None = None,
) -> tuple[
    PatchDecision,
    APMcLachlanState | None,
    np.ndarray | None,
    GeometryEvaluation | None,
    FixedMcLachlanStep | None,
]:
    effective_prefilter_policy = _validate_append_ladder_config(support_config)
    append_cost_settings = AppendCostSettings.from_config(support_config)
    ladder_metadata = {
        "append_ladder_enabled": True,
        "append_ladder_mode": "combinatorial",
        "append_occurrence_policy": str(support_config.append_occurrence_policy),
        "cost_model_effective": AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1,
        "rank_score_kind": AP_APPEND_RANK_SCORE_KIND_V1,
        "cost_settings": append_cost_settings.to_json_dict(),
        "prefilter_policy_requested": str(support_config.append_prefilter_policy),
        "prefilter_policy_effective": effective_prefilter_policy,
    }
    if float(base_step.residual_ratio) < float(support_config.residual_ratio_threshold):
        batch = _empty_patch_batch(
            time=float(time),
            state=state,
            base_step=base_step,
            reason="residual_below_threshold",
            selection_policy=APPEND_LADDER_SELECTION_POLICY_V1,
            metadata=ladder_metadata,
        )
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=0,
                scored_count=0,
                reason="residual_below_threshold",
                batch_evaluation=batch,
            ),
            None,
            None,
            None,
            None,
        )
    if not bool(state.can_structural_edit) and not bool(
        support_config.allow_incomplete_candidate_pool
    ):
        batch = _empty_patch_batch(
            time=float(time),
            state=state,
            base_step=base_step,
            reason="incomplete_candidate_pool",
            selection_policy=APPEND_LADDER_SELECTION_POLICY_V1,
            metadata=ladder_metadata,
        )
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=0,
                scored_count=0,
                reason="incomplete_candidate_pool",
                batch_evaluation=batch,
            ),
            None,
            None,
            None,
            None,
        )
    if int(support_config.max_append_batch_size) <= 0:
        batch = _empty_patch_batch(
            time=float(time),
            state=state,
            base_step=base_step,
            reason="append_batch_size_zero",
            selection_policy=APPEND_LADDER_SELECTION_POLICY_V1,
            metadata=ladder_metadata,
        )
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=0,
                scored_count=0,
                reason="append_batch_size_zero",
                batch_evaluation=batch,
            ),
            None,
            None,
            None,
            None,
        )
    atoms = candidate_append_atoms(
        state,
        allow_incomplete_candidate_pool=bool(
            support_config.allow_incomplete_candidate_pool
        ),
        occurrence_policy=str(support_config.append_occurrence_policy),
    )
    if not atoms:
        batch = _empty_patch_batch(
            time=float(time),
            state=state,
            base_step=base_step,
            reason="no_append_atoms",
            selection_policy=APPEND_LADDER_SELECTION_POLICY_V1,
            metadata=ladder_metadata,
        )
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=0,
                scored_count=0,
                reason="no_append_atoms",
                batch_evaluation=batch,
            ),
            None,
            None,
            None,
            None,
        )

    reuse_pre = _failed_append_reuse_pre_search_decision(
        state=state,
        atoms=atoms,
        support_config=support_config,
        inverse_policy=inverse_policy,
        base_evaluation=base_evaluation,
        base_step=base_step,
        time=float(time),
        time_index=int(time_index),
        geometry_clock=float(geometry_clock),
        reuse_state=reuse_state,
    )
    ladder_metadata = _metadata_with_failed_append_reuse(
        ladder_metadata,
        reuse_pre,
    )
    if not bool(reuse_pre.get("full_search_run", True)):
        batch = _empty_patch_batch(
            time=float(time),
            state=state,
            base_step=base_step,
            reason="failed_append_reuse_skip",
            selection_policy=APPEND_LADDER_SELECTION_POLICY_V1,
            metadata=ladder_metadata,
        )
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=0,
                scored_count=0,
                reason="failed_append_reuse_skip",
                batch_evaluation=batch,
                metadata={"failed_append_reuse": reuse_pre},
            ),
            None,
            None,
            None,
            None,
        )

    batch = _score_append_ladder_batch(
        state=state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_runtime,
        time=float(time),
        base_evaluation=base_evaluation,
        base_step=base_step,
        atoms=atoms,
        inverse_policy=inverse_policy,
        support_config=support_config,
        effective_prefilter_policy=effective_prefilter_policy,
        metadata=ladder_metadata,
    )
    selected = batch.selected_score
    if selected is None or selected.score is None:
        batch, reuse_meta = _record_failed_append_ladder_search(
            batch=batch,
            state=state,
            atoms=atoms,
            support_config=support_config,
            inverse_policy=inverse_policy,
            base_evaluation=base_evaluation,
            base_step=base_step,
            time=float(time),
            time_index=int(time_index),
            geometry_clock=float(geometry_clock),
            reuse_state=reuse_state,
            pre_search_metadata=reuse_pre,
            failure_reason=batch.reason or "no_finite_append_ladder_score",
        )
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                reason=batch.reason or "no_finite_append_ladder_score",
                batch_evaluation=batch,
                metadata={"failed_append_reuse": reuse_meta},
            ),
            None,
            None,
            None,
            None,
        )
    score = selected.score
    insertion_gain = None if score.insertion_gain is None else float(score.insertion_gain)
    rank_score = selected.rank_score
    if insertion_gain is None or rank_score is None or not np.isfinite(float(rank_score)):
        batch, reuse_meta = _record_failed_append_ladder_search(
            batch=batch,
            state=state,
            atoms=atoms,
            support_config=support_config,
            inverse_policy=inverse_policy,
            base_evaluation=base_evaluation,
            base_step=base_step,
            time=float(time),
            time_index=int(time_index),
            geometry_clock=float(geometry_clock),
            reuse_state=reuse_state,
            pre_search_metadata=reuse_pre,
            failure_reason="no_finite_append_ladder_score",
        )
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                selected_label=selected.candidate_label,
                selected_score=score,
                reason="no_finite_append_ladder_score",
                batch_evaluation=batch,
                metadata={"failed_append_reuse": reuse_meta},
            ),
            None,
            None,
            None,
            None,
        )
    if insertion_gain < float(support_config.append_gain_threshold):
        batch, reuse_meta = _record_failed_append_ladder_search(
            batch=batch,
            state=state,
            atoms=atoms,
            support_config=support_config,
            inverse_policy=inverse_policy,
            base_evaluation=base_evaluation,
            base_step=base_step,
            time=float(time),
            time_index=int(time_index),
            geometry_clock=float(geometry_clock),
            reuse_state=reuse_state,
            pre_search_metadata=reuse_pre,
            failure_reason="append_gain_below_threshold",
        )
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                selected_label=selected.candidate_label,
                selected_score=score,
                reason="append_gain_below_threshold",
                batch_evaluation=batch,
                metadata={"failed_append_reuse": reuse_meta},
            ),
            None,
            None,
            None,
            None,
        )
    if float(rank_score) < float(support_config.append_batch_score_threshold):
        batch, reuse_meta = _record_failed_append_ladder_search(
            batch=batch,
            state=state,
            atoms=atoms,
            support_config=support_config,
            inverse_policy=inverse_policy,
            base_evaluation=base_evaluation,
            base_step=base_step,
            time=float(time),
            time_index=int(time_index),
            geometry_clock=float(geometry_clock),
            reuse_state=reuse_state,
            pre_search_metadata=reuse_pre,
            failure_reason="append_batch_score_below_threshold",
        )
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                selected_label=selected.candidate_label,
                selected_score=score,
                reason="append_batch_score_below_threshold",
                batch_evaluation=batch,
                metadata={"failed_append_reuse": reuse_meta},
            ),
            None,
            None,
            None,
            None,
        )
    if not bool(selected.accepted_eligible):
        batch, reuse_meta = _record_failed_append_ladder_search(
            batch=batch,
            state=state,
            atoms=atoms,
            support_config=support_config,
            inverse_policy=inverse_policy,
            base_evaluation=base_evaluation,
            base_step=base_step,
            time=float(time),
            time_index=int(time_index),
            geometry_clock=float(geometry_clock),
            reuse_state=reuse_state,
            pre_search_metadata=reuse_pre,
            failure_reason=str(selected.rejection_reason),
        )
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                selected_label=selected.candidate_label,
                selected_score=score,
                reason=str(selected.rejection_reason),
                batch_evaluation=batch,
                metadata={"failed_append_reuse": reuse_meta},
            ),
            None,
            None,
            None,
            None,
        )

    atoms_by_id = {str(atom.atom_id): atom for atom in atoms}
    selected_atom_ids = tuple(str(v) for v in selected.metadata.get("atom_ids", ()))
    selected_atoms = tuple(atoms_by_id[atom_id] for atom_id in selected_atom_ids)
    materialized = _materialize_append_atom_set(
        state=state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_runtime,
        time=float(time),
        atoms=selected_atoms,
        inverse_policy=inverse_policy,
        solve_repair_config=solve_repair_config,
        repair_dt=repair_dt,
    )
    appended_state, theta_aug, evaluation, step = materialized
    if reuse_state is not None:
        reuse_state.clear_certificate()
    accepted_reuse_meta = _accepted_failed_append_reuse_metadata(reuse_pre)
    batch = _batch_with_failed_append_reuse_metadata(batch, accepted_reuse_meta)
    return (
        PatchDecision(
            patch_kind=PATCH_APPEND,
            accepted=True,
            candidate_count=batch.candidate_count,
            scored_count=batch.scored_count,
            selected_label=selected.candidate_label,
            selected_score=score,
            reason=batch.reason or "accepted_best_append_ladder_gain",
            batch_evaluation=batch,
            metadata={"failed_append_reuse": accepted_reuse_meta},
        ),
        appended_state,
        theta_aug,
        evaluation,
        step,
    )


def _select_unified_support_patch(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    inverse_policy: McLachlanInversePolicy,
    solve_repair_config: SolveRepairConfig,
    support_config: SupportPatchControllerConfig,
    runtime_state: _PruneControllerRuntimeState,
    repair_dt: float | None = None,
    time_index: int = 0,
) -> tuple[
    PatchDecision,
    APMcLachlanState | None,
    np.ndarray | None,
    GeometryEvaluation | None,
    FixedMcLachlanStep | None,
]:
    """Select one Paper-II support patch from stay/append/prune/exchange finalists."""

    effective_prefilter_policy = _validate_append_ladder_config(support_config)
    runtime_state.ensure_support_identity(state)
    append_metadata = _append_ladder_metadata(
        support_config,
        effective_prefilter_policy=effective_prefilter_policy,
    )
    prune_metadata = _prune_ladder_metadata(support_config)
    append_atoms: tuple[SupportAtom, ...] = ()
    append_batch: PatchBatchEvaluation | None = None
    append_candidate_geometry_cache: _AppendCandidateGeometryCache | None = None
    append_score_executor: ThreadPoolExecutor | None = None
    append_score_future: Any | None = None
    family_scoring_config = support_config
    prune_atoms: tuple[ActiveSupportAtom, ...] = ()
    prune_batch: PatchBatchEvaluation | None = None
    before_cache: SupportPatchBeforeCache | None = None

    def get_before_cache() -> SupportPatchBeforeCache:
        nonlocal before_cache
        if before_cache is None:
            before_cache = build_support_patch_before_cache(
                geometry=SupportPatchGeometry(
                    K_before=base_evaluation.geometry.K,
                    f_before=base_evaluation.geometry.f,
                    norm_b_sq=float(base_evaluation.geometry.norm_b_sq),
                ),
                inverse_policy=inverse_policy,
            )
        return before_cache

    def resolve_append_score() -> None:
        nonlocal append_batch, append_score_executor, append_score_future
        if append_score_future is None:
            return
        try:
            append_batch = append_score_future.result()
        finally:
            if append_score_executor is not None:
                append_score_executor.shutdown(wait=True)
            append_score_executor = None
            append_score_future = None

    def cancel_append_score() -> None:
        nonlocal append_score_executor, append_score_future
        if append_score_future is not None:
            append_score_future.cancel()
        if append_score_executor is not None:
            append_score_executor.shutdown(wait=True, cancel_futures=True)
        append_score_executor = None
        append_score_future = None

    finalists: list[_PatchFinalist] = [
        _stay_patch_finalist(
            state=state,
            base_evaluation=base_evaluation,
            base_step=base_step,
            inverse_policy=inverse_policy,
            support_config=support_config,
        )
    ]

    append_reason = None
    if float(base_step.residual_ratio) < float(support_config.residual_ratio_threshold):
        append_reason = "residual_below_threshold"
        append_batch = _empty_patch_batch(
            time=float(time),
            state=state,
            base_step=base_step,
            reason=append_reason,
            selection_policy=APPEND_LADDER_SELECTION_POLICY_V1,
            metadata=append_metadata,
        )
    elif int(support_config.max_append_batch_size) <= 0:
        append_reason = "append_batch_size_zero"
        append_batch = _empty_patch_batch(
            time=float(time),
            state=state,
            base_step=base_step,
            reason=append_reason,
            selection_policy=APPEND_LADDER_SELECTION_POLICY_V1,
            metadata=append_metadata,
        )
    elif not bool(state.can_structural_edit) and not bool(
        support_config.allow_incomplete_candidate_pool
    ):
        append_reason = "incomplete_candidate_pool"
        append_batch = _empty_patch_batch(
            time=float(time),
            state=state,
            base_step=base_step,
            reason=append_reason,
            selection_policy=APPEND_LADDER_SELECTION_POLICY_V1,
            metadata=append_metadata,
        )
    else:
        append_atoms = candidate_append_atoms(
            state,
            allow_incomplete_candidate_pool=bool(
                support_config.allow_incomplete_candidate_pool
            ),
            occurrence_policy=str(support_config.append_occurrence_policy),
        )
        base_K = np.asarray(base_evaluation.geometry.K, dtype=float)
        base_f = np.asarray(base_evaluation.geometry.f, dtype=float).reshape(-1)
        n_before = int(state.runtime_parameter_count)
        macro_scout_mode = validate_append_macro_scout_score_mode(
            support_config.append_macro_scout_score_mode
        )
        cheap_parent_scorer = (
            _make_cheap_parent_scout_scorer(
                state=state,
                theta_runtime=theta_runtime,
                base_evaluation=base_evaluation,
                base_step=base_step,
                inverse_policy=inverse_policy,
                support_config=support_config,
            )
            if (
                bool(support_config.append_macro_scout_enabled)
                and macro_scout_mode in APPEND_MACRO_SCOUT_CHEAP_SCORE_MODES
            )
            else None
        )

        def diagnostic_parent_scorer(
            parent_label: str,
            parent_atoms: tuple[SupportAtom, ...],
            ordinal: int,
        ) -> SupportFrontierScore:
            score = _score_append_atom_set(
                state=state,
                hamiltonian=hamiltonian,
                theta_runtime=theta_runtime,
                time=float(time),
                base_K=base_K,
                base_f=base_f,
                norm_b_sq=float(base_evaluation.geometry.norm_b_sq),
                n_before=n_before,
                atoms=tuple(parent_atoms),
                inverse_policy=inverse_policy,
                support_config=support_config,
                before_cache=get_before_cache(),
                candidate_set_index=int(ordinal),
                score_index=int(ordinal),
            )
            patch_score = score.score
            return SupportFrontierScore(
                parent_label=str(parent_label),
                score=None if score.score is None else score.score.rank_score,
                rank_score=score.rank_score,
                insertion_gain=(
                    None
                    if patch_score is None or patch_score.insertion_gain is None
                    else float(patch_score.insertion_gain)
                ),
                accepted_eligible=bool(score.accepted_eligible),
                rejection_reason=score.rejection_reason,
                metadata={
                    **dict(score.metadata or {}),
                    "diagnostic_mode": (
                        APPEND_MACRO_SCOUT_SCORE_MODE_FULL_CHILD_BLOCK_DIAGNOSTIC
                    ),
                    "measurement_saving": False,
                },
            )

        frontier = build_append_support_frontier(
            atoms=append_atoms,
            enabled=bool(support_config.append_macro_scout_enabled),
            score_mode=str(support_config.append_macro_scout_score_mode),
            parent_cap=int(support_config.append_macro_scout_parent_cap),
            score_min=float(support_config.append_macro_scout_score_min),
            fail_open=bool(support_config.append_macro_scout_fail_open),
            residual_ratio=float(base_step.residual_ratio),
            expand_if_residual_high=float(
                support_config.append_macro_scout_expand_if_residual_high
            ),
            exchange_requested=bool(
                support_config.exchange_enabled and _use_active_prune_ladder(support_config)
            ),
            exchange_fail_open=bool(support_config.append_macro_scout_exchange_fail_open),
            audit_parent_count=int(support_config.append_macro_scout_audit_parent_count),
            audit_parent_fraction=float(
                support_config.append_macro_scout_audit_parent_fraction
            ),
            parent_cost_alpha=float(support_config.append_macro_scout_parent_cost_alpha),
            cheap_parent_scorer=cheap_parent_scorer,
            diagnostic_parent_scorer=diagnostic_parent_scorer,
        )
        append_atoms = frontier.child_append_atoms
        append_metadata = {**dict(append_metadata), **dict(frontier.metadata or {})}
        if not append_atoms:
            append_reason = "no_append_atoms"
            append_batch = _empty_patch_batch(
                time=float(time),
                state=state,
                base_step=base_step,
                reason=append_reason,
                selection_policy=APPEND_LADDER_SELECTION_POLICY_V1,
                metadata=append_metadata,
            )
        else:
            append_candidate_geometry_cache = _build_append_candidate_geometry_cache(
                state=state,
                base_evaluation=base_evaluation,
                atoms=append_atoms,
                schur_inverse_policy=_append_schur_inverse_policy(
                    inverse_policy,
                    support_config=support_config,
                ),
            )
            append_score_kwargs = {
                "state": state,
                "hamiltonian": hamiltonian,
                "theta_runtime": theta_runtime,
                "time": float(time),
                "base_evaluation": base_evaluation,
                "base_step": base_step,
                "atoms": append_atoms,
                "inverse_policy": inverse_policy,
                "effective_prefilter_policy": effective_prefilter_policy,
                "metadata": append_metadata,
                "before_cache": get_before_cache(),
                "candidate_geometry_cache": append_candidate_geometry_cache,
            }
            if (
                _use_active_prune_ladder(support_config)
                and int(support_config.support_patch_scoring_workers) > 1
            ):
                family_scoring_config = replace(
                    support_config,
                    support_patch_scoring_workers=max(
                        1,
                        int(support_config.support_patch_scoring_workers) // 2,
                    ),
                )
                append_score_executor = ThreadPoolExecutor(max_workers=1)
                append_score_future = append_score_executor.submit(
                    _score_append_ladder_batch,
                    **append_score_kwargs,
                    support_config=family_scoring_config,
                )
            else:
                append_batch = _score_append_ladder_batch(
                    **append_score_kwargs,
                    support_config=support_config,
                )

    if _use_active_prune_ladder(support_config):
        prune_atoms = _active_prune_atoms(
            state,
            theta_runtime=theta_runtime,
            support_config=support_config,
            runtime_state=runtime_state,
            time_index=int(time_index),
        )
        if not prune_atoms:
            prune_batch = _empty_patch_batch(
                time=float(time),
                state=state,
                base_step=base_step,
                reason="no_active_prune_atoms",
                selection_policy=PRUNE_LADDER_SELECTION_POLICY_V1,
                metadata=prune_metadata,
            )
        else:
            try:
                prune_batch = _score_prune_ladder_batch(
                    state=state,
                    theta_runtime=theta_runtime,
                    time=float(time),
                    time_index=int(time_index),
                    base_evaluation=base_evaluation,
                    base_step=base_step,
                    atoms=prune_atoms,
                    inverse_policy=inverse_policy,
                    support_config=family_scoring_config,
                    runtime_state=runtime_state,
                    metadata=prune_metadata,
                    before_cache=get_before_cache(),
                )
            except BaseException:
                cancel_append_score()
                raise
            resolve_append_score()
            prune_by_id = {str(atom.atom_id): atom for atom in prune_atoms}
            first_rejected_finalist: _PatchFinalist | None = None
            accepted_prune_finalist: _PatchFinalist | None = None
            for branch in _top_patch_branches(
                prune_batch,
                limit=max(1, int(len(prune_batch.candidate_scores))),
                score_min=0.0,
                require_eligible=False,
            ):
                branch_atoms = _atoms_from_candidate_metadata(branch, prune_by_id)
                safety = _evaluate_deletion_patch_safety(
                    state=state,
                    hamiltonian=hamiltonian,
                    theta_runtime=theta_runtime,
                    time=float(time),
                    base_evaluation=base_evaluation,
                    base_step=base_step,
                    candidate=branch,
                    pruned_atoms=branch_atoms,
                    appended_atoms=(),
                    inverse_policy=inverse_policy,
                    solve_repair_config=solve_repair_config,
                    support_config=support_config,
                    runtime_state=runtime_state,
                    repair_dt=repair_dt,
                    time_index=int(time_index),
                )
                finalist = _deletion_patch_finalist(
                    branch,
                    pruned_atoms=branch_atoms,
                    appended_atoms=(),
                    safety=safety,
                    support_config=support_config,
                )
                if bool(safety.passed):
                    accepted_prune_finalist = finalist
                    break
                if first_rejected_finalist is None:
                    first_rejected_finalist = finalist
            if accepted_prune_finalist is not None:
                finalists.append(accepted_prune_finalist)
            elif first_rejected_finalist is not None:
                finalists.append(first_rejected_finalist)

    resolve_append_score()
    if append_batch is not None:
        append_reason = append_batch.reason
        append_by_id = {str(atom.atom_id): atom for atom in append_atoms}
        for branch in _preferred_append_finalist_branches(
            append_batch,
            score_min=float(support_config.append_batch_score_threshold),
        ):
            branch_atoms = _atoms_from_candidate_metadata(branch, append_by_id)
            finalists.insert(
                1,
                _append_patch_finalist(
                    branch,
                    branch_atoms,
                    support_config=support_config,
                )
            )

    if bool(support_config.exchange_enabled) and append_batch is not None and prune_batch is not None:
        append_by_id = {str(atom.atom_id): atom for atom in append_atoms}
        prune_by_id = {str(atom.atom_id): atom for atom in prune_atoms}
        append_branches = _top_patch_branches(
            append_batch,
            limit=int(support_config.max_exchange_append_branches),
            score_min=float(support_config.exchange_append_score_min),
            require_eligible=False,
        )
        prune_branches = _top_patch_branches(
            prune_batch,
            limit=int(support_config.max_exchange_prune_branches),
            score_min=float(support_config.exchange_prune_score_min),
            require_eligible=False,
        )
        pair_limit = int(support_config.max_exchange_pair_count)
        pair_count = 0
        exchange_tasks: list[
            tuple[
                PatchCandidateScore,
                PatchCandidateScore,
                tuple[SupportAtom, ...],
                tuple[ActiveSupportAtom, ...],
            ]
        ] = []
        for append_branch in append_branches:
            for prune_branch in prune_branches:
                if pair_limit > 0 and pair_count >= pair_limit:
                    break
                pair_count += 1
                inserted = _atoms_from_candidate_metadata(append_branch, append_by_id)
                removed = _atoms_from_candidate_metadata(prune_branch, prune_by_id)
                if not inserted or not removed:
                    continue
                exchange_tasks.append(
                    (
                        append_branch,
                        prune_branch,
                        tuple(inserted),
                        tuple(removed),
                    )
                )
            if pair_limit > 0 and pair_count >= pair_limit:
                break

        def score_exchange_task(
            task: tuple[
                PatchCandidateScore,
                PatchCandidateScore,
                tuple[SupportAtom, ...],
                tuple[ActiveSupportAtom, ...],
            ]
        ) -> _ExchangeScoringResult:
            append_branch, prune_branch, inserted, removed = task
            return _score_exchange_candidate_only(
                state=state,
                hamiltonian=hamiltonian,
                theta_runtime=theta_runtime,
                time=float(time),
                base_evaluation=base_evaluation,
                base_step=base_step,
                append_candidate=append_branch,
                prune_candidate=prune_branch,
                appended_atoms=inserted,
                pruned_atoms=removed,
                inverse_policy=inverse_policy,
                support_config=support_config,
                before_cache=get_before_cache(),
                candidate_geometry_cache=append_candidate_geometry_cache,
            )

        for result in _ordered_parallel_map(
            tuple(exchange_tasks),
            support_config=support_config,
            score_one=score_exchange_task,
        ):
            finalist = _exchange_scoring_result_to_finalist(
                result,
                state=state,
                hamiltonian=hamiltonian,
                theta_runtime=theta_runtime,
                time=float(time),
                base_evaluation=base_evaluation,
                base_step=base_step,
                inverse_policy=inverse_policy,
                solve_repair_config=solve_repair_config,
                support_config=support_config,
                runtime_state=runtime_state,
                repair_dt=repair_dt,
                time_index=int(time_index),
            )
            finalists.append(finalist)

    batch = _batch_from_patch_finalists(
        finalists,
        time=float(time),
        state=state,
        base_step=base_step,
        append_batch=append_batch,
        prune_batch=prune_batch,
        append_reason=append_reason,
        support_config=support_config,
    )
    selected_index = batch.selected_index
    selected_finalist = None if selected_index is None else finalists[int(selected_index)]
    if (
        selected_finalist is None
        or selected_finalist.candidate.candidate_kind == PATCH_NO_EDIT
    ):
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                reason=batch.reason or "stay_selected",
                batch_evaluation=batch,
            ),
            None,
            None,
            None,
            None,
        )
    if not bool(selected_finalist.passed):
        candidate = selected_finalist.candidate
        return (
            PatchDecision(
                patch_kind=str(candidate.candidate_kind),
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                selected_label=candidate.candidate_label,
                selected_score=candidate.score,
                reason=str(selected_finalist.reason),
                batch_evaluation=batch,
                metadata=dict(selected_finalist.metadata or {}),
            ),
            None,
            None,
            None,
            None,
        )
    candidate = selected_finalist.candidate
    if candidate.candidate_kind == PATCH_APPEND:
        materialized = _materialize_append_atom_set(
            state=state,
            hamiltonian=hamiltonian,
            theta_runtime=theta_runtime,
            time=float(time),
            atoms=selected_finalist.appended_atoms,
            inverse_policy=inverse_policy,
            solve_repair_config=solve_repair_config,
            repair_dt=repair_dt,
        )
        patched_state, theta_patched, evaluation, step = materialized
    else:
        patched_state = selected_finalist.patched_state
        theta_patched = selected_finalist.theta_patched
        evaluation = selected_finalist.evaluation
        step = selected_finalist.step
        if (
            patched_state is None
            or theta_patched is None
            or evaluation is None
            or step is None
        ):
            return (
                PatchDecision(
                    patch_kind=PATCH_NO_EDIT,
                    accepted=False,
                    candidate_count=batch.candidate_count,
                    scored_count=batch.scored_count,
                    selected_label=candidate.candidate_label,
                    selected_score=candidate.score,
                    reason="selected_patch_missing_materialization",
                    batch_evaluation=batch,
                    metadata=dict(selected_finalist.metadata or {}),
                ),
                None,
                None,
                None,
                None,
            )
    return (
        PatchDecision(
            patch_kind=str(candidate.candidate_kind),
            accepted=True,
            candidate_count=batch.candidate_count,
            scored_count=batch.scored_count,
            selected_label=candidate.candidate_label,
            selected_score=candidate.score,
            reason=str(selected_finalist.reason),
            batch_evaluation=batch,
            metadata=dict(selected_finalist.metadata or {}),
        ),
        patched_state,
        np.asarray(theta_patched, dtype=float).reshape(-1),
        evaluation,
        step,
    )


def _score_append_ladder_batch(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    atoms: Sequence[SupportAtom],
    inverse_policy: McLachlanInversePolicy,
    support_config: SupportPatchControllerConfig,
    effective_prefilter_policy: str,
    metadata: Mapping[str, Any],
    before_cache: SupportPatchBeforeCache | None = None,
    candidate_geometry_cache: _AppendCandidateGeometryCache | None = None,
) -> PatchBatchEvaluation:
    base_K = np.asarray(base_evaluation.geometry.K, dtype=float)
    base_f = np.asarray(base_evaluation.geometry.f, dtype=float).reshape(-1)
    n_before = int(state.runtime_parameter_count)
    if base_K.shape != (n_before, n_before):
        raise ValueError(
            "base McLachlan geometry size does not match current runtime support: "
            f"got {base_K.shape}, expected ({n_before}, {n_before})."
        )
    if int(base_f.size) != n_before:
        raise ValueError(
            "base McLachlan force size does not match current runtime support: "
            f"got {base_f.size}, expected {n_before}."
        )

    all_atoms = tuple(atoms)
    if candidate_geometry_cache is None:
        candidate_geometry_cache = _build_append_candidate_geometry_cache(
            state=state,
            base_evaluation=base_evaluation,
            atoms=all_atoms,
            schur_inverse_policy=_append_schur_inverse_policy(
                inverse_policy,
                support_config=support_config,
            ),
        )
    max_rung = min(int(support_config.max_append_batch_size), int(len(all_atoms)))
    scores: list[PatchCandidateScore] = []
    rung_diagnostics: list[RungDiagnostics] = []
    scored_count = 0
    best_index: int | None = None
    best_rank_score: float | None = None
    best_eligible_index: int | None = None
    best_eligible_rank_score: float | None = None
    candidate_set_index = 0
    singleton_ranked: list[tuple[SupportAtom, float, int]] = []

    for rung_size in range(1, max_rung + 1):
        if rung_size == 1:
            source_atoms = all_atoms
            before_prefilter_count = int(len(all_atoms))
            after_prefilter_count = int(len(source_atoms))
        else:
            if not singleton_ranked:
                before_prefilter_count = _comb_count_safe(len(all_atoms), rung_size)
                after_prefilter_count = 0
                rung_diagnostics.append(
                    _append_ladder_rung_diagnostics(
                        rung_size=rung_size,
                        before_prefilter_count=before_prefilter_count,
                        after_prefilter_count=after_prefilter_count,
                        attempted_count=0,
                        scored_count=0,
                        best_score=None,
                        best_atom_ids=(),
                        effective_prefilter_policy=effective_prefilter_policy,
                        support_config=support_config,
                        rejection_reason="no_finite_singleton_prefilter_atoms",
                    )
                )
                continue
            prefilter_limit = int(support_config.append_prefilter_size)
            if prefilter_limit > 0:
                source_atoms = tuple(
                    item[0] for item in singleton_ranked[:prefilter_limit]
                )
            else:
                source_atoms = tuple(item[0] for item in singleton_ranked)
            before_prefilter_count = _comb_count_safe(len(all_atoms), rung_size)
            after_prefilter_count = _comb_count_safe(len(source_atoms), rung_size)

        attempted_count = _rung_attempt_count(
            after_prefilter_count,
            cap=int(support_config.append_rung_set_cap),
        )
        rung_score_start = int(len(scores))
        rung_best_score: float | None = None
        rung_best_atom_ids: tuple[str, ...] = ()
        rung_scored_count = 0
        if attempted_count > 0:
            rung_score_start = int(len(scores))
            candidate_set_index_start = int(candidate_set_index)
            tasks = tuple(
                (
                    tuple(atom_set),
                    candidate_set_index_start + offset,
                    rung_score_start + offset,
                )
                for offset, atom_set in enumerate(
                    itertools.islice(
                        itertools.combinations(source_atoms, rung_size),
                        attempted_count,
                    )
                )
            )

            def score_task(task: tuple[tuple[SupportAtom, ...], int, int]) -> PatchCandidateScore:
                atom_set, task_candidate_set_index, task_score_index = task
                try:
                    return _score_append_atom_set(
                        state=state,
                        hamiltonian=hamiltonian,
                        theta_runtime=theta_runtime,
                        time=float(time),
                        base_K=base_K,
                        base_f=base_f,
                        norm_b_sq=float(base_evaluation.geometry.norm_b_sq),
                        n_before=n_before,
                        atoms=tuple(atom_set),
                        inverse_policy=inverse_policy,
                        support_config=support_config,
                        before_cache=before_cache,
                        candidate_geometry_cache=candidate_geometry_cache,
                        candidate_set_index=task_candidate_set_index,
                        score_index=task_score_index,
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    return _failed_append_atom_set_score(
                        atoms=tuple(atom_set),
                        rung_size=rung_size,
                        candidate_set_index=task_candidate_set_index,
                        score_index=task_score_index,
                        error=str(exc),
                    )

            for candidate_score in _ordered_parallel_map(
                tasks,
                support_config=support_config,
                score_one=score_task,
            ):
                scores.append(candidate_score)
                if candidate_score.score is not None:
                    scored_count += 1
                    rung_scored_count += 1
                rank_score = candidate_score.rank_score
                finite_rank = rank_score is not None and np.isfinite(float(rank_score))
                if finite_rank:
                    score_index = int(
                        candidate_score.metadata.get("score_index", len(scores) - 1)
                    )
                    if rung_best_score is None or float(rank_score) > float(rung_best_score):
                        rung_best_score = float(rank_score)
                        rung_best_atom_ids = tuple(
                            str(atom_id)
                            for atom_id in candidate_score.metadata.get("atom_ids", ())
                        )
                    if best_rank_score is None or float(rank_score) > float(best_rank_score):
                        best_rank_score = float(rank_score)
                        best_index = int(score_index)
                    if candidate_score.accepted_eligible and (
                        best_eligible_rank_score is None
                        or float(rank_score) > float(best_eligible_rank_score)
                    ):
                        best_eligible_rank_score = float(rank_score)
                        best_eligible_index = int(score_index)
            candidate_set_index += len(tasks)
        if rung_size == 1:
            _apply_append_cost_scores(
                scores,
                support_config=support_config,
                score_indices=range(rung_score_start, len(scores)),
            )
            singleton_ranked = [
                (
                    all_atoms[int(score.metadata["candidate_set_index"])],
                    float(score.rank_score),
                    int(score.metadata["candidate_set_index"]),
                )
                for score in scores[rung_score_start:]
                if score.rank_score is not None
                and np.isfinite(float(score.rank_score))
                and int(score.metadata.get("rung_size", -1)) == 1
            ]
            singleton_ranked.sort(key=lambda item: (-float(item[1]), int(item[2])))
        rejection_reason = None
        if after_prefilter_count == 0:
            rejection_reason = "no_candidate_sets_after_prefilter"
        elif attempted_count < after_prefilter_count:
            rejection_reason = "append_rung_set_cap_applied"
        rung_diagnostics.append(
            _append_ladder_rung_diagnostics(
                rung_size=rung_size,
                before_prefilter_count=before_prefilter_count,
                after_prefilter_count=after_prefilter_count,
                attempted_count=attempted_count,
                scored_count=rung_scored_count,
                best_score=rung_best_score,
                best_atom_ids=rung_best_atom_ids,
                effective_prefilter_policy=effective_prefilter_policy,
                support_config=support_config,
                rejection_reason=rejection_reason,
            )
        )

    _apply_append_cost_scores(scores, support_config=support_config)
    rung_diagnostics = _refresh_append_ladder_rung_diagnostics(
        rung_diagnostics,
        tuple(scores),
    )
    best_index = None
    best_rank_score = None
    best_eligible_index = None
    best_eligible_rank_score = None
    for score_index, candidate_score in enumerate(tuple(scores)):
        rank_score = candidate_score.rank_score
        finite_rank = rank_score is not None and np.isfinite(float(rank_score))
        if not finite_rank:
            continue
        if best_rank_score is None or float(rank_score) > float(best_rank_score):
            best_rank_score = float(rank_score)
            best_index = int(score_index)
        if candidate_score.accepted_eligible and (
            best_eligible_rank_score is None
            or float(rank_score) > float(best_eligible_rank_score)
        ):
            best_eligible_rank_score = float(rank_score)
            best_eligible_index = int(score_index)

    selected_index = best_eligible_index if best_eligible_index is not None else best_index
    selected_score = None if selected_index is None else scores[int(selected_index)]
    reason = "no_finite_append_ladder_score"
    if selected_score is not None and selected_score.score is not None:
        insertion_gain = selected_score.score.insertion_gain
        rank_score = selected_score.rank_score
        if insertion_gain is None or rank_score is None or not np.isfinite(float(rank_score)):
            reason = "no_finite_append_ladder_score"
        elif float(insertion_gain) < float(support_config.append_gain_threshold):
            reason = "append_gain_below_threshold"
        elif float(rank_score) < float(support_config.append_batch_score_threshold):
            reason = "append_batch_score_below_threshold"
        elif not bool(selected_score.accepted_eligible):
            reason = str(selected_score.rejection_reason)
        else:
            reason = "accepted_best_append_ladder_gain"

    return PatchBatchEvaluation(
        time=float(time),
        base_runtime_parameter_count=int(state.runtime_parameter_count),
        base_logical_parameter_count=int(state.logical_parameter_count),
        base_residual_ratio=float(base_step.residual_ratio),
        candidate_count=int(sum(r.candidate_set_count_before_prefilter for r in rung_diagnostics)),
        scored_count=int(scored_count),
        candidate_scores=tuple(scores),
        selected_index=selected_index,
        selected_score=selected_score,
        reason=reason,
        selection_policy=APPEND_LADDER_SELECTION_POLICY_V1,
        rung_diagnostics=tuple(rung_diagnostics),
        metadata={
            **dict(metadata),
            "candidate_atom_count": int(len(all_atoms)),
            "max_append_batch_size_effective": int(max_rung),
            "append_candidate_geometry_mode": (
                "zero_angle_tangent_cache_v1"
                if candidate_geometry_cache is not None
                else "full_augmented_geometry_compat_v1"
            ),
        },
    )


def _apply_append_cost_scores(
    scores: list[PatchCandidateScore],
    *,
    support_config: SupportPatchControllerConfig,
    score_indices: Sequence[int] | range | None = None,
) -> None:
    indices = (
        tuple(range(len(scores)))
        if score_indices is None
        else tuple(int(index) for index in score_indices)
    )
    valid_indices = tuple(
        index
        for index in indices
        if 0 <= int(index) < len(scores)
        and scores[int(index)].score is not None
        and isinstance(scores[int(index)].metadata, Mapping)
        and "append_cost_raw" in scores[int(index)].metadata
    )
    if not valid_indices:
        return
    settings = AppendCostSettings.from_config(support_config)
    raw_estimates = []
    insertion_gains = []
    for index in valid_indices:
        metadata = dict(scores[index].metadata or {})
        raw_estimates.append(_append_cost_raw_from_metadata(metadata["append_cost_raw"]))
        score = scores[index].score
        insertion_gains.append(None if score is None else score.insertion_gain)
    telemetry = append_cost_telemetry_for_family(
        raw_estimates,
        insertion_gains=insertion_gains,
        settings=settings,
    )
    for index, cost in zip(valid_indices, telemetry):
        candidate = scores[index]
        score = candidate.score
        insertion_gain = None if score is None else score.insertion_gain
        rank_utility = cost.rank_utility
        finite_rank = rank_utility is not None and np.isfinite(float(rank_utility))
        solve_reason = str(
            candidate.metadata.get(
                "augmented_solve_confirmation_reason",
                "eligible",
            )
        )
        schur_reason = str(candidate.metadata.get("schur_guard_reason", "eligible"))
        solve_ok = solve_reason == "eligible"
        schur_ok = schur_reason == "eligible"
        eligible = bool(
            finite_rank
            and insertion_gain is not None
            and float(insertion_gain) >= float(support_config.append_gain_threshold)
            and float(rank_utility) >= float(support_config.append_batch_score_threshold)
            and solve_ok
            and schur_ok
        )
        reason = "eligible"
        if not finite_rank:
            reason = "nonfinite_rank_score"
        elif insertion_gain is None:
            reason = "missing_insertion_gain"
        elif float(insertion_gain) < float(support_config.append_gain_threshold):
            reason = "append_gain_below_threshold"
        elif float(rank_utility) < float(support_config.append_batch_score_threshold):
            reason = "append_batch_score_below_threshold"
        elif not solve_ok:
            reason = solve_reason
        elif not schur_ok:
            reason = schur_reason
        scores[index] = replace(
            candidate,
            rank_score=None if rank_utility is None else float(rank_utility),
            accepted_eligible=eligible,
            rejection_reason=reason,
            metadata={
                **dict(candidate.metadata or {}),
                "append_cost": cost.to_json_dict(),
                "rank_score_kind": AP_APPEND_RANK_SCORE_KIND_V1,
            },
        )


def _append_cost_raw_from_metadata(payload: Mapping[str, Any]) -> Any:
    from pipelines.time_dynamics.ap_mclachlan.append_cost import AppendCostRawEstimate

    return AppendCostRawEstimate(
        raw_components=dict(payload.get("raw_components", {})),
        component_sources=dict(payload.get("component_sources", {})),
        inserted_runtime_count=int(payload.get("inserted_runtime_count", 0)),
        atom_ids=tuple(str(v) for v in payload.get("atom_ids", ())),
    )


def _refresh_append_ladder_rung_diagnostics(
    diagnostics: Sequence[RungDiagnostics],
    scores: Sequence[PatchCandidateScore],
) -> list[RungDiagnostics]:
    refreshed: list[RungDiagnostics] = []
    for rung in tuple(diagnostics):
        rung_size = int(rung.rung_size)
        rung_scores = [
            score
            for score in tuple(scores)
            if int(dict(score.metadata or {}).get("rung_size", -1)) == rung_size
            and score.rank_score is not None
            and np.isfinite(float(score.rank_score))
        ]
        if not rung_scores:
            refreshed.append(rung)
            continue
        best = max(
            rung_scores,
            key=lambda score: (
                float(score.rank_score),
                -int(dict(score.metadata or {}).get("candidate_set_index", 10**12)),
            ),
        )
        refreshed.append(
            replace(
                rung,
                best_score=float(best.rank_score),
                best_atom_ids=tuple(
                    str(atom_id)
                    for atom_id in dict(best.metadata or {}).get("atom_ids", ())
                ),
                metadata={
                    **dict(rung.metadata or {}),
                    "rank_score_kind": AP_APPEND_RANK_SCORE_KIND_V1,
                },
            )
        )
    return refreshed


def _append_schur_inverse_policy(
    inverse_policy: McLachlanInversePolicy,
    *,
    support_config: SupportPatchControllerConfig,
) -> McLachlanInversePolicy:
    ridge = float(support_config.append_schur_novelty_ridge_lambda)
    return McLachlanInversePolicy(
        pinv_rcond=float(inverse_policy.pinv_rcond),
        ridge_lambda=ridge,
        solve_damping=0.0,
        epsilon=float(inverse_policy.epsilon),
        policy_id=str(inverse_policy.policy_id),
    )


def _append_schur_guard_reason(
    score: SupportPatchScore,
    *,
    support_config: SupportPatchControllerConfig,
) -> str:
    if not bool(support_config.append_schur_guard_enabled):
        return "eligible"
    novelty = score.schur_novelty
    if novelty is None:
        return "missing_append_schur_novelty"
    if not bool(novelty.psd_within_tolerance):
        return "append_schur_not_psd"
    required_rank = int(
        math.ceil(
            float(support_config.append_schur_min_rank_fraction)
            * max(0, int(score.inserted_count))
        )
    )
    if int(novelty.rank) < required_rank:
        return "append_schur_rank_deficient"
    max_condition = float(support_config.append_schur_max_condition_number)
    if max_condition > 0.0:
        condition = novelty.condition_number
        if condition is None and required_rank > 0:
            return "append_schur_condition_missing"
        if condition is not None and float(condition) > max_condition:
            return "append_schur_condition_too_large"
    return "eligible"


def _append_augmented_solve_confirmation_reason(score: SupportPatchScore) -> str:
    if int(score.inserted_count) <= 0:
        return "eligible"
    confirmation = score.augmented_solve_confirmation
    if confirmation is None:
        return "missing_append_augmented_solve_confirmation"
    if not bool(confirmation.confirmed):
        return "append_augmented_solve_not_confirmed"
    return "eligible"


def _make_cheap_parent_scout_scorer(
    *,
    state: APMcLachlanState,
    theta_runtime: np.ndarray,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    inverse_policy: McLachlanInversePolicy,
    support_config: SupportPatchControllerConfig,
) -> Callable[[str, tuple[SupportAtom, ...], int], SupportFrontierScore]:
    mode = validate_append_macro_scout_score_mode(
        support_config.append_macro_scout_score_mode
    )
    if mode not in APPEND_MACRO_SCOUT_CHEAP_SCORE_MODES:
        raise ValueError(f"Unsupported cheap parent scout mode: {mode!r}.")
    try:
        context = _parent_scout_base_context(
            state=state,
            theta_runtime=theta_runtime,
            base_evaluation=base_evaluation,
            base_step=base_step,
            score_mode=mode,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        reason = f"{mode}_measurements_unavailable"
        error_text = str(exc)

        def unavailable_scorer(
            parent_label: str,
            parent_atoms: tuple[SupportAtom, ...],
            ordinal: int,
        ) -> SupportFrontierScore:
            raise SupportFrontierFailOpen(
                reason,
                metadata={
                    "macro_scout_measurement_saving_score_available": False,
                    "macro_scout_unavailable_error": error_text,
                },
            )

        return unavailable_scorer

    def scorer(
        parent_label: str,
        parent_atoms: tuple[SupportAtom, ...],
        ordinal: int,
    ) -> SupportFrontierScore:
        return _score_cheap_parent_scout(
            context=context,
            parent_label=parent_label,
            parent_atoms=parent_atoms,
            ordinal=int(ordinal),
            score_mode=mode,
            inverse_policy=inverse_policy,
            support_config=support_config,
        )

    return scorer


@dataclass(frozen=True)
class _ParentScoutBaseContext:
    active_terms: tuple[Any, ...]
    theta_runtime: np.ndarray
    psi_ref: np.ndarray
    psi: np.ndarray
    b_bar: np.ndarray
    T_active: np.ndarray
    base_K: np.ndarray
    base_f: np.ndarray
    norm_b_sq: float
    theta_dot: np.ndarray
    n_before: int
    hilbert_dim: int
    ignore_identity: bool
    coefficient_tolerance: float
    sort_terms: bool


@dataclass(frozen=True)
class _AppendCandidateGeometryCache:
    """Checkpoint-local zero-angle append tangent blocks.

    Tail-appended coordinates are initialized at zero, so they leave the
    prepared state and every active tangent unchanged.  The append ladder can
    therefore reuse the base tangent matrix and evaluate each candidate
    tangent exactly once instead of rebuilding the full augmented executor for
    every candidate set.
    """

    atom_index_by_id: Mapping[str, int]
    tangent_matrix: np.ndarray
    K_active_candidate: np.ndarray
    f_candidate: np.ndarray
    K_active_schur_inverse: np.ndarray | None


def _build_append_candidate_geometry_cache(
    *,
    state: APMcLachlanState,
    base_evaluation: GeometryEvaluation,
    atoms: Sequence[SupportAtom],
    schur_inverse_policy: McLachlanInversePolicy | None = None,
) -> _AppendCandidateGeometryCache | None:
    active_tangents = base_evaluation.tangent_matrix
    if active_tangents is None:
        return None

    psi = np.asarray(base_evaluation.psi, dtype=complex).reshape(-1)
    h_psi = np.asarray(base_evaluation.h_psi, dtype=complex).reshape(-1)
    T_active = np.asarray(active_tangents, dtype=complex)
    n_before = int(state.runtime_parameter_count)
    if T_active.shape != (int(psi.size), n_before):
        raise ValueError(
            "base tangent_matrix shape mismatch for append geometry cache: "
            f"{T_active.shape} vs ({psi.size}, {n_before})."
        )
    if h_psi.shape != psi.shape:
        raise ValueError("base h_psi shape mismatch for append geometry cache.")
    if (
        not np.all(np.isfinite(psi))
        or not np.all(np.isfinite(h_psi))
        or not np.all(np.isfinite(T_active))
    ):
        raise ValueError("base append geometry inputs must be finite.")

    atom_index_by_id: dict[str, int] = {}
    candidate_columns: list[np.ndarray] = []
    for atom in tuple(atoms):
        atom_id = str(atom.atom_id)
        if atom_id in atom_index_by_id:
            raise ValueError(f"duplicate append atom id in geometry cache: {atom_id!r}.")
        atom_index_by_id[atom_id] = int(len(candidate_columns))
        candidate_columns.append(
            _zero_angle_append_atom_tangent(state=state, psi=psi, atom=atom)
        )

    if candidate_columns:
        T_candidate = np.asarray(np.column_stack(candidate_columns), dtype=complex)
        K_active_candidate = np.asarray(
            np.real(T_active.conj().T @ T_candidate), dtype=float
        )
    else:
        T_candidate = np.zeros((int(psi.size), 0), dtype=complex)
        K_active_candidate = np.zeros((n_before, 0), dtype=float)
    b_bar = np.asarray(
        -1.0j * (h_psi - float(base_evaluation.energy_expectation) * psi),
        dtype=complex,
    ).reshape(-1)
    f_candidate = np.asarray(
        np.real(T_candidate.conj().T @ b_bar), dtype=float
    ).reshape(-1)
    K_active_schur_inverse = None
    if schur_inverse_policy is not None:
        K_active_schur_inverse = np.asarray(
            supported_inverse(
                base_evaluation.geometry.K,
                policy=schur_inverse_policy,
            ).inverse,
            dtype=float,
        )
    if (
        not np.all(np.isfinite(T_candidate))
        or not np.all(np.isfinite(K_active_candidate))
        or not np.all(np.isfinite(f_candidate))
        or (
            K_active_schur_inverse is not None
            and not np.all(np.isfinite(K_active_schur_inverse))
        )
    ):
        raise ValueError("append candidate geometry cache contains non-finite values.")
    return _AppendCandidateGeometryCache(
        atom_index_by_id=atom_index_by_id,
        tangent_matrix=T_candidate,
        K_active_candidate=K_active_candidate,
        f_candidate=f_candidate,
        K_active_schur_inverse=K_active_schur_inverse,
    )


def _zero_angle_append_atom_tangent(
    *,
    state: APMcLachlanState,
    psi: np.ndarray,
    atom: SupportAtom,
) -> np.ndarray:
    if int(atom.runtime_count) != 1:
        raise ValueError(
            "zero-angle append geometry cache requires one runtime coordinate "
            f"per support atom; got {atom.runtime_count} for {atom.atom_id!r}."
        )
    specs = iter_runtime_rotation_terms(
        getattr(atom.term, "polynomial"),
        ignore_identity=bool(state.executor.ignore_identity),
        coefficient_tolerance=float(state.executor.coefficient_tolerance),
        sort_terms=bool(state.executor.sort_terms),
    )
    mode = normalize_parameterization_mode(state.parameterization_mode)
    if mode == AP_PARAMETERIZATION_PER_PAULI_TERM and len(specs) != 1:
        raise ValueError(
            "per_pauli_term append atom must contain exactly one runtime Pauli "
            f"term; got {len(specs)} for {atom.atom_id!r}."
        )
    if not specs:
        raise ValueError(f"append atom {atom.atom_id!r} has no runtime rotation terms.")

    raw = np.zeros_like(np.asarray(psi, dtype=complex).reshape(-1))
    for spec in specs:
        expected_dim = 1 << int(spec.nq)
        if int(raw.size) != expected_dim:
            raise ValueError(
                "append atom Hilbert dimension mismatch: "
                f"state={raw.size}, atom={expected_dim}."
            )
        action = state.executor.pauli_action_cache.get(str(spec.pauli_exyz))
        if action is None:
            action = compile_pauli_action_exyz(str(spec.pauli_exyz), int(spec.nq))
            state.executor.pauli_action_cache[str(spec.pauli_exyz)] = action
        raw = raw + (
            -1.0j
            * float(spec.coeff_real)
            * apply_compiled_pauli(psi, action)
        )
    horizontal = np.asarray(raw - psi * np.vdot(psi, raw), dtype=complex).reshape(-1)
    if not np.all(np.isfinite(horizontal)):
        raise ValueError("zero-angle append tangent contains non-finite values.")
    return horizontal


def _append_insert_geometry_from_cache(
    cache: _AppendCandidateGeometryCache,
    atoms: Sequence[SupportAtom],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = tuple(
        int(cache.atom_index_by_id[str(atom.atom_id)]) for atom in tuple(atoms)
    )
    if len(set(indices)) != len(indices):
        raise ValueError("append candidate set contains duplicate support atoms.")
    selected = np.asarray(cache.tangent_matrix[:, list(indices)], dtype=complex)
    K_insert = np.asarray(np.real(selected.conj().T @ selected), dtype=float)
    K_insert = 0.5 * (K_insert + K_insert.T)
    return (
        np.asarray(cache.K_active_candidate[:, list(indices)], dtype=float),
        K_insert,
        np.asarray(cache.f_candidate[list(indices)], dtype=float).reshape(-1),
    )


def _parent_scout_base_context(
    *,
    state: APMcLachlanState,
    theta_runtime: np.ndarray,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    score_mode: str,
) -> _ParentScoutBaseContext:
    if normalize_parameterization_mode(state.parameterization_mode) != AP_PARAMETERIZATION_PER_PAULI_TERM:
        raise ValueError("cheap parent scout requires per_pauli_term active support.")
    theta = np.asarray(theta_runtime, dtype=float).reshape(-1)
    n_before = int(state.runtime_parameter_count)
    if int(theta.size) != n_before:
        raise ValueError(
            f"theta_runtime length mismatch for parent scout: {theta.size} vs {n_before}."
        )
    tangent_matrix = base_evaluation.tangent_matrix
    if tangent_matrix is None:
        raise ValueError("base_evaluation does not include tangent_matrix.")
    T_active = np.asarray(tangent_matrix, dtype=complex)
    psi = np.asarray(base_evaluation.psi, dtype=complex).reshape(-1)
    h_psi = np.asarray(base_evaluation.h_psi, dtype=complex).reshape(-1)
    if T_active.ndim != 2:
        raise ValueError("base tangent_matrix must be rank-2.")
    if T_active.shape != (int(psi.size), n_before):
        raise ValueError(
            "base tangent_matrix shape mismatch for parent scout: "
            f"{T_active.shape} vs ({psi.size}, {n_before})."
        )
    if not np.all(np.isfinite(T_active)) or not np.all(np.isfinite(psi)):
        raise ValueError("base tangent_matrix and psi must be finite.")
    if h_psi.shape != psi.shape or not np.all(np.isfinite(h_psi)):
        raise ValueError("base h_psi must be finite and match psi shape.")
    energy = float(base_evaluation.energy_expectation)
    b_bar = np.asarray(-1.0j * (h_psi - energy * psi), dtype=complex).reshape(-1)
    if not np.all(np.isfinite(b_bar)):
        raise ValueError("base McLachlan drift vector is non-finite.")

    base_K = np.asarray(base_evaluation.geometry.K, dtype=float)
    base_f = np.asarray(base_evaluation.geometry.f, dtype=float).reshape(-1)
    if base_K.shape != (n_before, n_before) or base_f.shape != (n_before,):
        raise ValueError("base McLachlan geometry shape mismatch for parent scout.")
    K_reconstructed = np.asarray(np.real(T_active.conj().T @ T_active), dtype=float)
    f_reconstructed = np.asarray(np.real(T_active.conj().T @ b_bar), dtype=float).reshape(-1)
    tol = _parent_scout_geometry_tolerance(state)
    if not np.allclose(K_reconstructed, base_K, rtol=1.0e-6, atol=tol):
        raise ValueError("base tangent_matrix does not reconstruct McLachlan K.")
    if not np.allclose(f_reconstructed, base_f, rtol=1.0e-6, atol=tol):
        raise ValueError("base tangent_matrix does not reconstruct McLachlan f.")

    theta_dot = np.asarray(base_step.theta_dot, dtype=float).reshape(-1)
    if score_mode == APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_LINEAR_RESIDUAL_V1:
        if theta_dot.shape != (n_before,) or not np.all(np.isfinite(theta_dot)):
            raise ValueError("parent_linear_residual_v1 requires finite base theta_dot.")

    active_atoms = active_support_atoms(state, theta)
    if int(len(active_atoms)) != n_before:
        raise ValueError("active support atom count does not match runtime count.")
    active_terms = tuple(atom.term for atom in active_atoms)
    return _ParentScoutBaseContext(
        active_terms=active_terms,
        theta_runtime=theta,
        psi_ref=np.asarray(state.psi_ref, dtype=complex).reshape(-1),
        psi=psi,
        b_bar=b_bar,
        T_active=T_active,
        base_K=base_K,
        base_f=base_f,
        norm_b_sq=float(base_evaluation.geometry.norm_b_sq),
        theta_dot=theta_dot,
        n_before=n_before,
        hilbert_dim=int(psi.size),
        ignore_identity=bool(state.layout.ignore_identity),
        coefficient_tolerance=float(state.layout.coefficient_tolerance),
        sort_terms=(str(state.layout.term_order).strip().lower() == "sorted"),
    )


def _score_cheap_parent_scout(
    *,
    context: _ParentScoutBaseContext,
    parent_label: str,
    parent_atoms: tuple[SupportAtom, ...],
    ordinal: int,
    score_mode: str,
    inverse_policy: McLachlanInversePolicy,
    support_config: SupportPatchControllerConfig,
) -> SupportFrontierScore:
    atom_tuple = tuple(parent_atoms)
    try:
        parent_term = _parent_scout_term_from_child_atoms(parent_label, atom_tuple)
        parent_tangent = _evaluate_parent_scout_tangent(
            context=context,
            parent_term=parent_term,
            parent_label=parent_label,
        )
        K_cross, K_insert, f_insert = _parent_scout_insert_geometry(
            context=context,
            parent_tangent=parent_tangent,
        )
        patch = SupportPatch(
            inserted_count=1,
            inserted_labels=(f"{str(parent_label)}::macro_scout",),
        )
        patch_geometry = SupportPatchGeometry(
            K_before=context.base_K,
            f_before=context.base_f,
            norm_b_sq=float(context.norm_b_sq),
            K_insert_cross=K_cross,
            K_insert_insert=K_insert,
            f_insert=f_insert,
        )
        score = score_support_patch(
            geometry=patch_geometry,
            patch=patch,
            inverse_policy=inverse_policy,
            schur_inverse_policy=_append_schur_inverse_policy(
                inverse_policy,
                support_config=support_config,
            ),
            schur_candidate_ridge_lambda=float(
                support_config.append_schur_novelty_ridge_lambda
            ),
            cost_terms=None,
            cost_weight=0.0,
        )
        linear_residual = None
        linear_residual_score = None
        if score_mode == APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_LINEAR_RESIDUAL_V1:
            linear_residual = float(f_insert[0] - K_cross[:, 0].T @ context.theta_dot)
            denom = math.sqrt(
                max(float(context.norm_b_sq), float(inverse_policy.epsilon))
                * max(float(K_insert[0, 0]), float(inverse_policy.epsilon))
            )
            linear_residual_score = float(abs(linear_residual) / max(denom, 1.0e-300))
            raw_score = linear_residual_score
        else:
            raw_score = (
                None if score.insertion_gain is None else float(score.insertion_gain)
            )
        rank_score = None
        if raw_score is not None and np.isfinite(float(raw_score)):
            cost_proxy = _parent_scout_cost_proxy(atom_tuple)
            rank_score = float(
                float(raw_score)
                / max(1.0, cost_proxy) ** float(support_config.append_macro_scout_parent_cost_alpha)
            )
        solve_reason = _append_augmented_solve_confirmation_reason(score)
        schur_reason = _append_schur_guard_reason(score, support_config=support_config)
        eligible = bool(
            raw_score is not None
            and np.isfinite(float(raw_score))
            and solve_reason == "eligible"
            and schur_reason == "eligible"
        )
        reason = "eligible"
        if raw_score is None or not np.isfinite(float(raw_score)):
            reason = "nonfinite_parent_score"
        elif solve_reason != "eligible":
            reason = solve_reason
        elif schur_reason != "eligible":
            reason = schur_reason
        metadata = {
            **score.to_json_dict(),
            "parent_scout_score_source": "parent_tangent_statevector_scout_v1",
            "parent_scout_score_mode": str(score_mode),
            "parent_label": str(parent_label),
            "parent_ordinal": int(ordinal),
            "parent_child_count": int(len(atom_tuple)),
            "parent_cost_proxy": float(_parent_scout_cost_proxy(atom_tuple)),
            "parent_cost_alpha": float(support_config.append_macro_scout_parent_cost_alpha),
            "parent_raw_score": None if raw_score is None else float(raw_score),
            "parent_rank_score": None if rank_score is None else float(rank_score),
            "parent_insertion_gain": (
                None if score.insertion_gain is None else float(score.insertion_gain)
            ),
            "parent_linear_residual": linear_residual,
            "parent_linear_residual_score": linear_residual_score,
            "measurement_saving": True,
            "diagnostic_mode": None,
            "augmented_solve_confirmation_reason": solve_reason,
            "schur_guard_reason": schur_reason,
        }
        return SupportFrontierScore(
            parent_label=str(parent_label),
            score=None if raw_score is None else float(raw_score),
            rank_score=None if rank_score is None else float(rank_score),
            insertion_gain=(
                None if score.insertion_gain is None else float(score.insertion_gain)
            ),
            accepted_eligible=eligible,
            rejection_reason=reason,
            metadata=metadata,
        )
    except SupportFrontierFailOpen:
        raise
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise SupportFrontierFailOpen(
            f"{score_mode}_measurements_unavailable",
            metadata={
                "macro_scout_measurement_saving_score_available": False,
                "macro_scout_unavailable_error": str(exc),
            },
        ) from exc


def _parent_scout_term_from_child_atoms(
    parent_label: str,
    atoms: Sequence[SupportAtom],
) -> AnsatzTerm:
    atom_tuple = tuple(atoms)
    if not atom_tuple:
        raise ValueError("parent scout requires at least one child atom.")
    seen: set[str] = set()
    poly = None
    for atom in atom_tuple:
        if str(atom.parent_label) != str(parent_label):
            raise ValueError("parent scout atom group contains mixed parent labels.")
        if str(atom.atom_id) in seen:
            raise ValueError(f"duplicate parent scout child atom: {atom.atom_id!r}.")
        seen.add(str(atom.atom_id))
        if normalize_parameterization_mode(atom.parameterization_mode) != AP_PARAMETERIZATION_PER_PAULI_TERM:
            raise ValueError("cheap parent scout only supports child-level atoms.")
        if int(atom.runtime_count) != 1:
            raise ValueError("cheap parent scout child atoms must have runtime_count=1.")
        child_poly = getattr(atom.term, "polynomial", None)
        if child_poly is None:
            raise ValueError("child atom term is missing polynomial.")
        poly = (1.0 * child_poly) if poly is None else (poly + child_poly)
    if poly is None:
        raise ValueError("empty parent scout polynomial.")
    count = getattr(poly, "count_number_terms", None)
    if callable(count) and int(count()) <= 0:
        raise ValueError("parent scout polynomial has no active terms.")
    return AnsatzTerm(
        label=f"{str(parent_label)}::macro_scout",
        polynomial=poly,
        execution_mode="termwise_product",
    )


def _evaluate_parent_scout_tangent(
    *,
    context: _ParentScoutBaseContext,
    parent_term: AnsatzTerm,
    parent_label: str,
) -> np.ndarray:
    terms = tuple(context.active_terms) + (parent_term,)
    layout = build_parameter_layout(
        terms,
        ignore_identity=bool(context.ignore_identity),
        coefficient_tolerance=float(context.coefficient_tolerance),
        sort_terms=bool(context.sort_terms),
    )
    if int(layout.logical_parameter_count) != int(context.n_before + 1):
        raise ValueError("parent scout logical layout does not append one coordinate.")
    executor = CompiledAnsatzExecutor(
        terms,
        coefficient_tolerance=float(context.coefficient_tolerance),
        ignore_identity=bool(context.ignore_identity),
        sort_terms=bool(context.sort_terms),
        parameterization_mode="logical_shared",
        parameterization_layout=layout,
    )
    theta = np.zeros(int(context.n_before + 1), dtype=float)
    theta[: int(context.n_before)] = np.asarray(context.theta_runtime, dtype=float)
    psi_scout, tangents = executor.prepare_state_with_parameter_tangents(
        theta,
        np.asarray(context.psi_ref, dtype=complex).reshape(-1),
        parameter_indices=(int(context.n_before),),
    )
    psi_scout = np.asarray(psi_scout, dtype=complex).reshape(-1)
    if psi_scout.shape != context.psi.shape:
        raise ValueError("parent scout prepared state dimension mismatch.")
    phase, overlap_abs = _patch_phase_alignment(context.psi, psi_scout)
    if overlap_abs is None or float(overlap_abs) < 1.0 - _parent_scout_phase_tolerance():
        raise ValueError(
            "parent scout active prefix does not reproduce the current state "
            f"for parent {parent_label!r}; phase overlap={overlap_abs}."
        )
    psi_aligned = np.asarray(phase * psi_scout, dtype=complex).reshape(-1)
    if float(np.linalg.norm(psi_aligned - context.psi)) > _parent_scout_phase_tolerance():
        raise ValueError("parent scout active prefix state mismatch after phase alignment.")
    tangent = np.asarray(tangents[int(context.n_before)], dtype=complex).reshape(-1)
    if tangent.shape != context.psi.shape or not np.all(np.isfinite(tangent)):
        raise ValueError("parent scout tangent has invalid shape or nonfinite values.")
    return np.asarray(phase * tangent, dtype=complex).reshape(-1)


def _parent_scout_insert_geometry(
    *,
    context: _ParentScoutBaseContext,
    parent_tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parent_raw = np.asarray(parent_tangent, dtype=complex).reshape(-1)
    parent_horizontal = np.asarray(
        parent_raw - context.psi * np.vdot(context.psi, parent_raw),
        dtype=complex,
    ).reshape(-1)
    if parent_horizontal.shape != (int(context.hilbert_dim),):
        raise ValueError("horizontal parent tangent shape mismatch.")
    if not np.all(np.isfinite(parent_horizontal)):
        raise ValueError("horizontal parent tangent is nonfinite.")
    K_cross = np.asarray(
        np.real(context.T_active.conj().T @ parent_horizontal), dtype=float
    ).reshape(int(context.n_before), 1)
    K_insert = np.asarray(
        [[float(np.real(np.vdot(parent_horizontal, parent_horizontal)))]],
        dtype=float,
    )
    f_insert = np.asarray(
        [float(np.real(np.vdot(parent_horizontal, context.b_bar)))],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(K_cross))
        or not np.all(np.isfinite(K_insert))
        or not np.all(np.isfinite(f_insert))
    ):
        raise ValueError("parent scout insert geometry is nonfinite.")
    return K_cross, K_insert, f_insert


def _parent_scout_cost_proxy(atoms: Sequence[SupportAtom]) -> float:
    return float(sum(max(1, int(atom.runtime_count)) for atom in tuple(atoms)))


def _parent_scout_geometry_tolerance(state: APMcLachlanState) -> float:
    return max(1.0e-8, 100.0 * float(state.layout.coefficient_tolerance))


def _parent_scout_phase_tolerance() -> float:
    return 1.0e-7


def _score_append_atom_set(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    base_K: np.ndarray,
    base_f: np.ndarray,
    norm_b_sq: float,
    n_before: int,
    atoms: Sequence[SupportAtom],
    inverse_policy: McLachlanInversePolicy,
    support_config: SupportPatchControllerConfig,
    candidate_set_index: int,
    score_index: int,
    before_cache: SupportPatchBeforeCache | None = None,
    candidate_geometry_cache: _AppendCandidateGeometryCache | None = None,
) -> PatchCandidateScore:
    atom_tuple = tuple(atoms)
    if candidate_geometry_cache is None:
        appended_state, theta_aug = state_with_appended_atoms(
            state,
            atom_tuple,
            theta_runtime=theta_runtime,
        )
        if appended_state.runtime_coordinate_labels[:n_before] != state.runtime_coordinate_labels:
            raise ValueError(
                "appended atom set does not preserve existing runtime coordinates as a prefix."
            )
        m_insert = int(appended_state.runtime_parameter_count) - int(n_before)
        labels = tuple(appended_state.runtime_coordinate_labels[n_before:])
    else:
        m_insert = int(sum(int(atom.runtime_count) for atom in atom_tuple))
        labels = tuple(str(atom.atom_label) for atom in atom_tuple)
    patch = SupportPatch(
        inserted_count=int(m_insert),
        inserted_labels=labels,
    )
    if m_insert <= 0:
        return PatchCandidateScore(
            candidate_kind=PATCH_APPEND,
            candidate_label=_candidate_set_label(atom_tuple),
            patch=patch,
            score=None,
            rank_score=None,
            accepted_eligible=False,
            rejection_reason="candidate_added_no_runtime_coordinates",
            metadata=_append_atom_set_metadata(
                atom_tuple,
                candidate_set_index=candidate_set_index,
                score_index=score_index,
            ),
        )
    if candidate_geometry_cache is None:
        evaluation = evaluate_mclachlan_geometry(
            state=appended_state,
            hamiltonian=hamiltonian,
            theta_runtime=theta_aug,
            time=float(time),
        )
        K = np.asarray(evaluation.geometry.K, dtype=float)
        f = np.asarray(evaluation.geometry.f, dtype=float).reshape(-1)
        K_insert_cross = K[:n_before, n_before:]
        K_insert_insert = K[n_before:, n_before:]
        f_insert = f[n_before:]
    else:
        K_insert_cross, K_insert_insert, f_insert = _append_insert_geometry_from_cache(
            candidate_geometry_cache,
            atom_tuple,
        )
    patch_geometry = SupportPatchGeometry(
        K_before=base_K,
        f_before=base_f,
        norm_b_sq=float(norm_b_sq),
        K_insert_cross=K_insert_cross,
        K_insert_insert=K_insert_insert,
        f_insert=f_insert,
    )
    score = score_support_patch(
        geometry=patch_geometry,
        patch=patch,
        inverse_policy=inverse_policy,
        schur_inverse_policy=_append_schur_inverse_policy(
            inverse_policy,
            support_config=support_config,
        ),
        schur_candidate_ridge_lambda=float(
            support_config.append_schur_novelty_ridge_lambda
        ),
        cost_terms=None,
        cost_weight=0.0,
        before_cache=before_cache,
        schur_available_inverse=(
            None
            if candidate_geometry_cache is None
            else candidate_geometry_cache.K_active_schur_inverse
        ),
    )
    append_cost_raw = estimate_append_atom_set_cost(atom_tuple)
    rank_score = score.rank_score
    insertion_gain = None if score.insertion_gain is None else float(score.insertion_gain)
    finite_rank = rank_score is not None and np.isfinite(float(rank_score))
    solve_reason = _append_augmented_solve_confirmation_reason(score)
    solve_ok = solve_reason == "eligible"
    schur_reason = _append_schur_guard_reason(score, support_config=support_config)
    schur_ok = schur_reason == "eligible"
    eligible = bool(
        finite_rank
        and insertion_gain is not None
        and insertion_gain >= float(support_config.append_gain_threshold)
        and float(rank_score) >= float(support_config.append_batch_score_threshold)
        and solve_ok
        and schur_ok
    )
    reason = "eligible"
    if not finite_rank:
        reason = "nonfinite_rank_score"
    elif insertion_gain is None:
        reason = "missing_insertion_gain"
    elif insertion_gain < float(support_config.append_gain_threshold):
        reason = "append_gain_below_threshold"
    elif float(rank_score) < float(support_config.append_batch_score_threshold):
        reason = "append_batch_score_below_threshold"
    elif not solve_ok:
        reason = solve_reason
    elif not schur_ok:
        reason = schur_reason
    return PatchCandidateScore(
        candidate_kind=PATCH_APPEND,
        candidate_label=_candidate_set_label(atom_tuple),
        patch=patch,
        score=score,
        rank_score=None if rank_score is None else float(rank_score),
        accepted_eligible=eligible,
        rejection_reason=reason,
        metadata=_append_atom_set_metadata(
            atom_tuple,
            candidate_set_index=candidate_set_index,
            score_index=score_index,
            inserted_runtime_count=int(m_insert),
            augmented_solve_confirmation_reason=solve_reason,
            schur_guard_reason=schur_reason,
            append_cost_raw=append_cost_raw.to_json_dict(),
        ),
    )


def _failed_append_atom_set_score(
    *,
    atoms: Sequence[SupportAtom],
    rung_size: int,
    candidate_set_index: int,
    score_index: int,
    error: str,
) -> PatchCandidateScore:
    atom_tuple = tuple(atoms)
    labels = tuple(str(atom.atom_label) for atom in atom_tuple)
    return PatchCandidateScore(
        candidate_kind=PATCH_APPEND,
        candidate_label=_candidate_set_label(atom_tuple),
        patch=SupportPatch(
            inserted_count=int(len(labels)),
            inserted_labels=labels,
        ),
        score=None,
        rank_score=None,
        accepted_eligible=False,
        rejection_reason="candidate_scoring_failed",
        metadata={
            **_append_atom_set_metadata(
                atom_tuple,
                candidate_set_index=candidate_set_index,
                score_index=score_index,
            ),
            "rung_size": int(rung_size),
            "error": str(error),
        },
    )


def _append_atom_set_metadata(
    atoms: Sequence[SupportAtom],
    *,
    candidate_set_index: int,
    score_index: int,
    inserted_runtime_count: int | None = None,
    augmented_solve_confirmation_reason: str | None = None,
    schur_guard_reason: str | None = None,
    append_cost_raw: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    atom_tuple = tuple(atoms)
    return {
        "rung_size": int(len(atom_tuple)),
        "atom_ids": [str(atom.atom_id) for atom in atom_tuple],
        "atom_labels": [str(atom.atom_label) for atom in atom_tuple],
        "atom_parent_labels": [str(atom.parent_label) for atom in atom_tuple],
        "candidate_set_index": int(candidate_set_index),
        "score_index": int(score_index),
        "inserted_runtime_count": (
            None if inserted_runtime_count is None else int(inserted_runtime_count)
        ),
        "augmented_solve_confirmation_reason": (
            None
            if augmented_solve_confirmation_reason is None
            else str(augmented_solve_confirmation_reason)
        ),
        "schur_guard_reason": (
            None if schur_guard_reason is None else str(schur_guard_reason)
        ),
        "cost_model_effective": AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1,
        "rank_score_kind": AP_APPEND_RANK_SCORE_KIND_V1,
        "append_cost_raw": dict(append_cost_raw or {}),
    }


def _append_ladder_rung_diagnostics(
    *,
    rung_size: int,
    before_prefilter_count: int,
    after_prefilter_count: int,
    attempted_count: int,
    scored_count: int,
    best_score: float | None,
    best_atom_ids: Sequence[str],
    effective_prefilter_policy: str,
    support_config: SupportPatchControllerConfig,
    rejection_reason: str | None,
) -> RungDiagnostics:
    return RungDiagnostics(
        rung_size=int(rung_size),
        candidate_set_count_before_prefilter=int(before_prefilter_count),
        candidate_set_count_scored=int(scored_count),
        prefilter_policy=str(effective_prefilter_policy),
        best_score=best_score,
        best_atom_ids=tuple(str(atom_id) for atom_id in best_atom_ids),
        rejection_reason=rejection_reason,
        metadata={
            "candidate_set_count_after_prefilter": int(after_prefilter_count),
            "candidate_set_count_attempted": int(attempted_count),
            "candidate_set_count_rejected_by_prefilter": int(
                max(0, int(before_prefilter_count) - int(after_prefilter_count))
            ),
            "candidate_set_count_rejected_by_cap": int(
                max(0, int(after_prefilter_count) - int(attempted_count))
            ),
            "append_rung_set_cap": int(support_config.append_rung_set_cap),
            "append_prefilter_size": int(support_config.append_prefilter_size),
            "cost_model_effective": AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1,
            "rank_score_kind": AP_APPEND_RANK_SCORE_KIND_V1,
            "cost_normalization_mode": str(support_config.cost_normalization_mode),
            "append_cost_alpha": float(support_config.append_cost_alpha),
        },
    )


def _rung_attempt_count(candidate_count: int, *, cap: int) -> int:
    count = max(0, int(candidate_count))
    limit = int(cap)
    if limit <= 0:
        return count
    return min(count, limit)


def _comb_count_safe(n: int, k: int) -> int:
    if int(k) < 0 or int(n) < 0 or int(k) > int(n):
        return 0
    return int(math.comb(int(n), int(k)))


def _candidate_set_label(atoms: Sequence[SupportAtom]) -> str:
    return " + ".join(str(atom.atom_label) for atom in tuple(atoms))


def _materialize_append_atom_set(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    atoms: Sequence[SupportAtom],
    inverse_policy: McLachlanInversePolicy,
    solve_repair_config: SolveRepairConfig,
    repair_dt: float | None = None,
) -> tuple[APMcLachlanState, np.ndarray, GeometryEvaluation, FixedMcLachlanStep]:
    appended_state, theta_aug = state_with_appended_atoms(
        state,
        tuple(atoms),
        theta_runtime=theta_runtime,
    )
    theta_aug = np.asarray(theta_aug, dtype=float).reshape(-1)
    evaluation = evaluate_mclachlan_geometry(
        state=appended_state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_aug,
        time=float(time),
    )
    step = _solve_fixed_step_for_trajectory(
        evaluation.geometry,
        inverse_policy=inverse_policy,
        solve_repair_config=solve_repair_config,
        repair_dt=repair_dt,
    )
    return appended_state, theta_aug, evaluation, step


def _append_ladder_metadata(
    support_config: SupportPatchControllerConfig,
    *,
    effective_prefilter_policy: str,
) -> dict[str, Any]:
    append_cost_settings = AppendCostSettings.from_config(support_config)
    return {
        "append_ladder_enabled": True,
        "append_ladder_mode": "combinatorial",
        "append_occurrence_policy": str(support_config.append_occurrence_policy),
        "cost_model_effective": AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1,
        "rank_score_kind": AP_APPEND_RANK_SCORE_KIND_V1,
        "cost_settings": append_cost_settings.to_json_dict(),
        "prefilter_policy_requested": str(support_config.append_prefilter_policy),
        "prefilter_policy_effective": effective_prefilter_policy,
        "macro_scout_enabled": bool(support_config.append_macro_scout_enabled),
        "macro_scout_policy": APPEND_MACRO_SCOUT_POLICY_V2,
        "macro_scout_score_mode": str(support_config.append_macro_scout_score_mode),
        "macro_scout_parent_cap": int(support_config.append_macro_scout_parent_cap),
        "macro_scout_score_min": float(support_config.append_macro_scout_score_min),
        "macro_scout_fail_open": bool(support_config.append_macro_scout_fail_open),
        "macro_scout_expand_if_residual_high": float(
            support_config.append_macro_scout_expand_if_residual_high
        ),
        "macro_scout_exchange_fail_open": bool(
            support_config.append_macro_scout_exchange_fail_open
        ),
        "macro_scout_audit_parent_count": int(
            support_config.append_macro_scout_audit_parent_count
        ),
        "macro_scout_audit_parent_fraction": float(
            support_config.append_macro_scout_audit_parent_fraction
        ),
        "macro_scout_parent_cost_alpha": float(
            support_config.append_macro_scout_parent_cost_alpha
        ),
    }


def _prune_ladder_metadata(
    support_config: SupportPatchControllerConfig,
) -> dict[str, Any]:
    prune_cost_settings = PruneCostSettings.from_config(support_config)
    return {
        "prune_ladder_enabled": True,
        "prune_ladder_mode": "combinatorial",
        "cost_model_effective": AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1,
        "rank_score_kind": _prune_rank_score_kind_for_config(support_config),
        "cost_settings": prune_cost_settings.to_json_dict(),
        "prefilter_policy_effective": PRUNE_LADDER_PREFILTER_POLICY_V1,
        "commit_enabled": bool(support_config.prune_commit_enabled),
        "max_prune_commits": int(support_config.max_prune_commits),
    }


def _stay_patch_finalist(
    *,
    state: APMcLachlanState,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    inverse_policy: McLachlanInversePolicy,
    support_config: SupportPatchControllerConfig,
) -> _PatchFinalist:
    score = score_support_patch(
        geometry=SupportPatchGeometry(
            K_before=np.asarray(base_evaluation.geometry.K, dtype=float),
            f_before=np.asarray(base_evaluation.geometry.f, dtype=float).reshape(-1),
            norm_b_sq=float(base_evaluation.geometry.norm_b_sq),
        ),
        patch=SupportPatch(),
        inverse_policy=inverse_policy,
    )
    candidate = PatchCandidateScore(
        candidate_kind=PATCH_NO_EDIT,
        candidate_label="stay",
        patch=SupportPatch(),
        score=score,
        rank_score=0.0,
        accepted_eligible=True,
        rejection_reason="eligible",
        metadata={
            "support_patch_family": SUPPORT_PATCH_EXCHANGE_SELECTION_POLICY_V1,
            "patch_utility": 0.0,
            "patch_utility_threshold": float(support_config.patch_utility_threshold),
            "runtime_parameter_count": int(state.runtime_parameter_count),
            "mclachlan_residual_ratio": float(base_step.residual_ratio),
        },
    )
    return _PatchFinalist(
        candidate=candidate,
        utility=0.0,
        passed=True,
        reason="stay_selected",
        metadata=dict(candidate.metadata),
    )


def _top_patch_branches(
    batch: PatchBatchEvaluation | None,
    *,
    limit: int,
    score_min: float,
    require_eligible: bool,
) -> tuple[PatchCandidateScore, ...]:
    if batch is None or int(limit) <= 0:
        return ()
    ranked = []
    for score_index, candidate in enumerate(tuple(batch.candidate_scores)):
        if candidate.score is None:
            continue
        rank_score = candidate.rank_score
        if rank_score is None or not np.isfinite(float(rank_score)):
            continue
        if float(rank_score) < float(score_min):
            continue
        if bool(require_eligible) and not bool(candidate.accepted_eligible):
            continue
        ranked.append((candidate, float(rank_score), int(score_index)))
    ranked.sort(key=lambda item: (-float(item[1]), int(item[2])))
    return tuple(item[0] for item in ranked[: max(0, int(limit))])


def _preferred_append_finalist_branches(
    batch: PatchBatchEvaluation | None,
    *,
    score_min: float,
) -> tuple[PatchCandidateScore, ...]:
    """Return the best eligible append, falling back to rejected telemetry.

    A rejected candidate can have a larger raw utility than every admissible
    candidate (for example, when its Schur novelty block is rank deficient).
    It must not hide a lower-ranked eligible append from the unified selector.
    When no candidate is eligible, keep the highest-ranked rejection as the
    checkpoint diagnostic instead of dropping the reason behind the failed
    append attempt.
    """

    eligible = _top_patch_branches(
        batch,
        limit=1,
        score_min=float(score_min),
        require_eligible=True,
    )
    if eligible:
        return eligible
    return _top_patch_branches(
        batch,
        limit=1,
        score_min=float(score_min),
        require_eligible=False,
    )


def _atoms_from_candidate_metadata(
    candidate: PatchCandidateScore,
    atoms_by_id: Mapping[str, Any],
) -> tuple[Any, ...]:
    atom_ids = tuple(str(v) for v in candidate.metadata.get("atom_ids", ()))
    out = []
    for atom_id in atom_ids:
        atom = atoms_by_id.get(atom_id)
        if atom is not None:
            out.append(atom)
    return tuple(out)


def _append_patch_finalist(
    candidate: PatchCandidateScore,
    atoms: Sequence[SupportAtom],
    *,
    support_config: SupportPatchControllerConfig,
) -> _PatchFinalist:
    utility = None if candidate.rank_score is None else float(candidate.rank_score)
    passed = bool(
        candidate.accepted_eligible
        and utility is not None
        and np.isfinite(float(utility))
        and float(utility) >= float(support_config.patch_utility_threshold)
    )
    reason = "accepted_best_append_ladder_gain" if passed else str(candidate.rejection_reason)
    metadata = {
        **dict(candidate.metadata or {}),
        "support_patch_family": SUPPORT_PATCH_EXCHANGE_SELECTION_POLICY_V1,
        "patch_utility": _finite_or_none(utility),
        "patch_utility_threshold": float(support_config.patch_utility_threshold),
        "patch_finalist_kind": PATCH_APPEND,
    }
    return _PatchFinalist(
        candidate=replace(
            candidate,
            rank_score=utility,
            accepted_eligible=passed,
            rejection_reason="eligible" if passed else reason,
            metadata=metadata,
        ),
        utility=utility,
        passed=passed,
        reason=reason,
        appended_atoms=tuple(atoms),
        metadata=metadata,
    )


def _deletion_patch_finalist(
    candidate: PatchCandidateScore,
    *,
    pruned_atoms: Sequence[ActiveSupportAtom],
    appended_atoms: Sequence[SupportAtom],
    safety: _DeletionPatchSafetyResult,
    support_config: SupportPatchControllerConfig,
) -> _PatchFinalist:
    base_utility = None if candidate.rank_score is None else float(candidate.rank_score)
    smoothness = safety.smoothness
    eta = None if smoothness is None else _finite_or_none(smoothness.eta)
    denominator = 1.0
    if eta is not None and str(candidate.candidate_kind) != PATCH_APPEND:
        denominator += float(support_config.patch_utility_velocity_weight) * (
            float(eta)
            / max(float(support_config.prune_patch_smoothness_eta_max), 1.0e-300)
        )
    utility = None if base_utility is None else float(base_utility / denominator)
    passed = bool(
        safety.passed
        and candidate.accepted_eligible
        and utility is not None
        and np.isfinite(float(utility))
        and float(utility) >= float(support_config.patch_utility_threshold)
    )
    reason = (
        "accepted_cost_weighted_exchange"
        if passed and str(candidate.candidate_kind) == PATCH_EXCHANGE
        else "accepted_cost_weighted_prune"
        if passed
        else str(safety.reason or candidate.rejection_reason)
    )
    metadata = {
        **dict(candidate.metadata or {}),
        **dict(safety.metadata or {}),
        "support_patch_family": SUPPORT_PATCH_EXCHANGE_SELECTION_POLICY_V1,
        "patch_utility": _finite_or_none(utility),
        "patch_utility_threshold": float(support_config.patch_utility_threshold),
        "patch_utility_denominator": float(denominator),
        "patch_utility_base_score": _finite_or_none(base_utility),
        "patch_finalist_kind": str(candidate.candidate_kind),
    }
    updated = replace(
        candidate,
        rank_score=utility,
        accepted_eligible=passed,
        rejection_reason="eligible" if passed else reason,
        metadata=metadata,
    )
    return _PatchFinalist(
        candidate=updated,
        utility=utility,
        passed=passed,
        reason=reason,
        appended_atoms=tuple(appended_atoms),
        pruned_atoms=tuple(pruned_atoms),
        patched_state=safety.patched_state,
        theta_patched=safety.theta_patched,
        evaluation=safety.evaluation,
        step=safety.step,
        metadata=metadata,
    )


def _batch_from_patch_finalists(
    finalists: Sequence[_PatchFinalist],
    *,
    time: float,
    state: APMcLachlanState,
    base_step: FixedMcLachlanStep,
    append_batch: PatchBatchEvaluation | None,
    prune_batch: PatchBatchEvaluation | None,
    append_reason: str | None,
    support_config: SupportPatchControllerConfig,
) -> PatchBatchEvaluation:
    finalist_tuple = tuple(finalists)
    candidates = tuple(finalist.candidate for finalist in finalist_tuple)
    selected_index: int | None = None
    selected_utility: float | None = None
    for index, finalist in enumerate(finalist_tuple):
        utility = finalist.utility
        if utility is None or not np.isfinite(float(utility)):
            continue
        if finalist.candidate.candidate_kind == PATCH_NO_EDIT:
            continue
        if not bool(finalist.passed):
            continue
        if selected_utility is None or float(utility) > float(selected_utility):
            selected_utility = float(utility)
            selected_index = int(index)
    if selected_index is None:
        for index, finalist in enumerate(finalist_tuple):
            utility = finalist.utility
            if utility is None or not np.isfinite(float(utility)):
                continue
            if finalist.candidate.candidate_kind == PATCH_NO_EDIT:
                continue
            if selected_utility is None or float(utility) > float(selected_utility):
                selected_utility = float(utility)
                selected_index = int(index)
    if selected_index is None and finalist_tuple:
        selected_index = 0
        selected_utility = finalist_tuple[0].utility
    selected_finalist = None if selected_index is None else finalist_tuple[selected_index]
    selected_score = None if selected_finalist is None else selected_finalist.candidate
    if selected_score is None:
        reason = "no_support_patch_finalist_passed"
    elif selected_score.candidate_kind == PATCH_NO_EDIT:
        reason = append_reason or "stay_selected"
    else:
        reason = str(selected_finalist.reason)
    metadata = {
        "support_patch_family": SUPPORT_PATCH_EXCHANGE_SELECTION_POLICY_V1,
        "exchange_enabled": bool(support_config.exchange_enabled),
        "branch_scoring_enabled": bool(support_config.branch_scoring_enabled),
        "append_batch_reason": None if append_batch is None else append_batch.reason,
        "prune_batch_reason": None if prune_batch is None else prune_batch.reason,
        "append_candidate_score_count": (
            0 if append_batch is None else int(len(append_batch.candidate_scores))
        ),
        "prune_candidate_score_count": (
            0 if prune_batch is None else int(len(prune_batch.candidate_scores))
        ),
        "prefilter_policy_effective": (
            None
            if append_batch is None
            else dict(append_batch.metadata or {}).get("prefilter_policy_effective")
        ),
        "rank_score_kind": (
            None
            if append_batch is None
            else dict(append_batch.metadata or {}).get("rank_score_kind")
        ),
        "cost_model_effective": (
            None
            if append_batch is None
            else dict(append_batch.metadata or {}).get("cost_model_effective")
        ),
        "finalist_kind_counts": _finalist_kind_counts(candidates),
        "max_exchange_append_branches": int(support_config.max_exchange_append_branches),
        "max_exchange_prune_branches": int(support_config.max_exchange_prune_branches),
        "max_exchange_pair_count": int(support_config.max_exchange_pair_count),
    }
    if append_batch is not None:
        append_metadata = dict(append_batch.metadata or {})
        metadata.update(
            {
                str(key): value
                for key, value in append_metadata.items()
                if str(key).startswith("macro_scout_")
            }
        )
    rung_diagnostics = ()
    if append_batch is not None:
        rung_diagnostics = tuple(append_batch.rung_diagnostics)
    return PatchBatchEvaluation(
        time=float(time),
        base_runtime_parameter_count=int(state.runtime_parameter_count),
        base_logical_parameter_count=int(state.logical_parameter_count),
        base_residual_ratio=float(base_step.residual_ratio),
        candidate_count=int(len(candidates)),
        scored_count=int(sum(1 for candidate in candidates if candidate.score is not None)),
        candidate_scores=candidates,
        selected_index=selected_index,
        selected_score=selected_score,
        reason=reason,
        selection_policy=SUPPORT_PATCH_EXCHANGE_SELECTION_POLICY_V1,
        rung_diagnostics=rung_diagnostics,
        metadata=metadata,
    )


def _finalist_kind_counts(candidates: Sequence[PatchCandidateScore]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in tuple(candidates):
        key = str(candidate.candidate_kind)
        counts[key] = int(counts.get(key, 0) + 1)
    return counts


def _materialize_support_patch_atom_set(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    pruned_atoms: Sequence[ActiveSupportAtom],
    appended_atoms: Sequence[SupportAtom],
    inverse_policy: McLachlanInversePolicy,
    solve_repair_config: SolveRepairConfig,
    repair_dt: float | None = None,
    include_tangent_matrix: bool = False,
) -> tuple[
    APMcLachlanState,
    np.ndarray,
    GeometryEvaluation,
    FixedMcLachlanStep,
    Mapping[str, Any],
]:
    removed = tuple(
        sorted({int(idx) for atom in tuple(pruned_atoms) for idx in atom.runtime_indices})
    )
    patched_state, theta_patched = state_with_support_patch_atoms(
        state,
        removed_runtime_indices=removed,
        inserted_atoms=tuple(appended_atoms),
        theta_runtime=theta_runtime,
    )
    theta_patched = np.asarray(theta_patched, dtype=float).reshape(-1)
    theta_patched, refit_metadata = _transport_exact_adjacent_duplicate_angles(
        state,
        theta_runtime=np.asarray(theta_runtime, dtype=float).reshape(-1),
        removed_runtime_indices=removed,
        theta_patched=theta_patched,
    )
    evaluation = evaluate_mclachlan_geometry(
        state=patched_state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_patched,
        time=float(time),
        include_tangent_matrix=bool(include_tangent_matrix),
    )
    step = _solve_fixed_step_for_trajectory(
        evaluation.geometry,
        inverse_policy=inverse_policy,
        solve_repair_config=solve_repair_config,
        repair_dt=repair_dt,
    )
    return patched_state, theta_patched, evaluation, step, refit_metadata


def _transport_exact_adjacent_duplicate_angles(
    state: APMcLachlanState,
    *,
    theta_runtime: np.ndarray,
    removed_runtime_indices: Sequence[int],
    theta_patched: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exactly merge deleted adjacent duplicate Pauli rotations into a survivor."""

    theta_before = np.asarray(theta_runtime, dtype=float).reshape(-1)
    theta_after = np.asarray(theta_patched, dtype=float).reshape(-1).copy()
    removed = frozenset(int(index) for index in removed_runtime_indices)
    if (
        not removed
        or normalize_parameterization_mode(state.parameterization_mode)
        != AP_PARAMETERIZATION_PER_PAULI_TERM
    ):
        return theta_after, {
            "prune_patch_refit_mode": "zero_transport_only",
            "prune_patch_exact_duplicate_transfer_count": 0,
        }

    active = tuple(active_support_atoms(state, theta_before))
    ordered = tuple(
        atom
        for atom in active
        if len(tuple(atom.runtime_indices)) == 1
    )
    transfers: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(ordered):
        first = ordered[cursor]
        first_index = int(first.runtime_indices[0])
        base_atom_id = str(dict(first.metadata).get("base_atom_id", first.atom_id))
        run = [first]
        next_cursor = cursor + 1
        previous_index = first_index
        while next_cursor < len(ordered):
            atom = ordered[next_cursor]
            atom_index = int(atom.runtime_indices[0])
            atom_base_id = str(
                dict(atom.metadata).get("base_atom_id", atom.atom_id)
            )
            if atom_index != previous_index + 1 or atom_base_id != base_atom_id:
                break
            run.append(atom)
            previous_index = atom_index
            next_cursor += 1

        run_indices = tuple(int(atom.runtime_indices[0]) for atom in run)
        deleted = tuple(index for index in run_indices if index in removed)
        surviving = tuple(index for index in run_indices if index not in removed)
        if deleted and surviving:
            target_old_index = int(surviving[0])
            target_new_index = int(
                target_old_index
                - sum(1 for index in removed if int(index) < target_old_index)
            )
            transferred_angle = float(np.sum(theta_before[list(deleted)]))
            theta_after[target_new_index] += transferred_angle
            transfers.append(
                {
                    "base_atom_id": base_atom_id,
                    "deleted_runtime_indices": [int(index) for index in deleted],
                    "surviving_runtime_index_before": int(target_old_index),
                    "surviving_runtime_index_after": int(target_new_index),
                    "transferred_angle": float(transferred_angle),
                }
            )
        cursor = next_cursor

    mode = (
        "exact_adjacent_duplicate_transport"
        if transfers
        else "zero_transport_only"
    )
    return theta_after, {
        "prune_patch_refit_mode": mode,
        "prune_patch_exact_duplicate_transfer_count": int(len(transfers)),
        "prune_patch_exact_duplicate_transfers": transfers,
    }


def _evaluate_deletion_patch_safety(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    candidate: PatchCandidateScore,
    pruned_atoms: Sequence[ActiveSupportAtom],
    appended_atoms: Sequence[SupportAtom],
    inverse_policy: McLachlanInversePolicy,
    solve_repair_config: SolveRepairConfig,
    support_config: SupportPatchControllerConfig,
    runtime_state: _PruneControllerRuntimeState,
    repair_dt: float | None,
    time_index: int,
) -> _DeletionPatchSafetyResult:
    atom_tuple = tuple(pruned_atoms)
    persistence_metadata = _prune_persistence_metadata(
        runtime_state,
        candidate,
        support_config=support_config,
        time_index=int(time_index),
    )
    if not _prune_persistence_passed(persistence_metadata):
        return _DeletionPatchSafetyResult(
            passed=False,
            reason="prune_persistence_wait",
            metadata=dict(persistence_metadata),
        )
    if not bool(support_config.prune_commit_enabled):
        return _DeletionPatchSafetyResult(
            passed=False,
            reason="prune_commit_disabled",
            metadata=dict(persistence_metadata),
        )
    if int(support_config.max_prune_commits) <= 0:
        return _DeletionPatchSafetyResult(
            passed=False,
            reason="prune_commit_limit_zero",
            metadata=dict(persistence_metadata),
        )
    if int(runtime_state.accepted_commit_count) >= int(support_config.max_prune_commits):
        return _DeletionPatchSafetyResult(
            passed=False,
            reason="prune_commit_limit_reached",
            metadata=dict(persistence_metadata),
        )
    if bool(support_config.prune_shadow_enabled):
        return _DeletionPatchSafetyResult(
            passed=False,
            reason="shadow_no_harm_unsupported",
            metadata={**persistence_metadata, "shadow_enabled": True},
        )
    try:
        (
            patched_state,
            theta_patched,
            evaluation,
            step,
            refit_metadata,
        ) = _materialize_support_patch_atom_set(
            state=state,
            hamiltonian=hamiltonian,
            theta_runtime=theta_runtime,
            time=float(time),
            pruned_atoms=atom_tuple,
            appended_atoms=tuple(appended_atoms),
            inverse_policy=inverse_policy,
            solve_repair_config=solve_repair_config,
            repair_dt=repair_dt,
            include_tangent_matrix=bool(support_config.prune_patch_smoothness_enabled),
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        _cooldown_prune_atoms(
            runtime_state,
            atom_tuple,
            time_index=int(time_index),
            support_config=support_config,
        )
        return _DeletionPatchSafetyResult(
            passed=False,
            reason="prune_materialization_failed",
            metadata={**persistence_metadata, "error": str(exc)},
        )
    retry_metadata = _prune_smoothness_retry_metadata(
        runtime_state,
        atom_tuple,
        time_index=int(time_index),
    )
    ray_distance = _state_ray_distance(base_evaluation.psi, evaluation.psi)
    if ray_distance is None or ray_distance > float(support_config.prune_ray_distance_tol):
        _cooldown_prune_atoms(
            runtime_state,
            atom_tuple,
            time_index=int(time_index),
            support_config=support_config,
        )
        return _DeletionPatchSafetyResult(
            passed=False,
            reason="prune_ray_distance_above_tol",
            metadata={
                **persistence_metadata,
                **retry_metadata,
                **refit_metadata,
                "prune_ray_distance": _finite_or_none(ray_distance),
            },
        )
    differential_miss = _step_differential_miss(base_step, step)
    if (
        differential_miss is None
        or differential_miss > float(support_config.prune_differential_miss_tol)
    ):
        _cooldown_prune_atoms(
            runtime_state,
            atom_tuple,
            time_index=int(time_index),
            support_config=support_config,
        )
        return _DeletionPatchSafetyResult(
            passed=False,
            reason="prune_differential_miss_above_tol",
            metadata={
                **persistence_metadata,
                **retry_metadata,
                **refit_metadata,
                "prune_ray_distance": float(ray_distance),
                "prune_differential_miss": _finite_or_none(differential_miss),
            },
        )
    smoothness = _evaluate_prune_patch_smoothness(
        base_evaluation=base_evaluation,
        base_step=base_step,
        patched_evaluation=evaluation,
        patched_step=step,
        support_config=support_config,
        refit_mode=str(refit_metadata["prune_patch_refit_mode"]),
    )
    retry_metadata = _prune_smoothness_retry_metadata(
        runtime_state,
        atom_tuple,
        time_index=int(time_index),
        current_smoothness=smoothness,
    )
    smoothness_metadata = {**retry_metadata, **smoothness.to_metadata()}
    if not bool(smoothness.passed):
        cooldown_steps = _prune_smoothness_cooldown_steps(
            smoothness,
            support_config=support_config,
        )
        _cooldown_prune_atoms(
            runtime_state,
            atom_tuple,
            time_index=int(time_index),
            support_config=support_config,
            cooldown_steps=cooldown_steps,
        )
        record = _record_prune_smoothness_deferred(
            runtime_state,
            atom_tuple,
            time_index=int(time_index),
            cooldown_steps=cooldown_steps,
            smoothness=smoothness,
        )
        return _DeletionPatchSafetyResult(
            passed=False,
            reason=(
                "prune_patch_smoothness_unavailable"
                if not bool(smoothness.available)
                else "prune_patch_smoothness_deferred"
            ),
            metadata={
                **persistence_metadata,
                **smoothness_metadata,
                **refit_metadata,
                **record.to_metadata(),
                "prune_patch_smoothness_cooldown_steps": int(cooldown_steps),
                "prune_ray_distance": float(ray_distance),
                "prune_differential_miss": float(differential_miss),
                "shadow_enabled": False,
            },
            smoothness=smoothness,
        )
    runtime_state.smoothness_deferred.pop(_prune_batch_key(atom_tuple), None)
    return _DeletionPatchSafetyResult(
        passed=True,
        reason="eligible",
        metadata={
            **persistence_metadata,
            **smoothness_metadata,
            **refit_metadata,
            "prune_ray_distance": float(ray_distance),
            "prune_differential_miss": float(differential_miss),
            "shadow_enabled": False,
        },
        patched_state=patched_state,
        theta_patched=theta_patched,
        evaluation=evaluation,
        step=step,
        smoothness=smoothness,
    )


def _score_exchange_candidate_only(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    append_candidate: PatchCandidateScore,
    prune_candidate: PatchCandidateScore,
    appended_atoms: Sequence[SupportAtom],
    pruned_atoms: Sequence[ActiveSupportAtom],
    inverse_policy: McLachlanInversePolicy,
    support_config: SupportPatchControllerConfig,
    before_cache: SupportPatchBeforeCache | None = None,
    candidate_geometry_cache: _AppendCandidateGeometryCache | None = None,
) -> _ExchangeScoringResult:
    append_tuple = tuple(appended_atoms)
    prune_tuple = tuple(pruned_atoms)
    removed = tuple(
        sorted({int(idx) for atom in prune_tuple for idx in atom.runtime_indices})
    )
    base_K = np.asarray(base_evaluation.geometry.K, dtype=float)
    base_f = np.asarray(base_evaluation.geometry.f, dtype=float).reshape(-1)
    n_before = int(state.runtime_parameter_count)
    try:
        appended_state, theta_aug = state_with_appended_atoms(
            state,
            append_tuple,
            theta_runtime=theta_runtime,
        )
        m_insert = int(appended_state.runtime_parameter_count) - int(n_before)
        inserted_labels = tuple(appended_state.runtime_coordinate_labels[n_before:])
        if m_insert <= 0:
            raise ValueError("exchange append side added no runtime coordinates")
        if candidate_geometry_cache is None:
            evaluation_aug = evaluate_mclachlan_geometry(
                state=appended_state,
                hamiltonian=hamiltonian,
                theta_runtime=theta_aug,
                time=float(time),
            )
            K_aug = np.asarray(evaluation_aug.geometry.K, dtype=float)
            f_aug = np.asarray(evaluation_aug.geometry.f, dtype=float).reshape(-1)
            K_cross = K_aug[:n_before, n_before:]
            K_insert = K_aug[n_before:, n_before:]
            f_insert = f_aug[n_before:]
        else:
            K_cross, K_insert, f_insert = _append_insert_geometry_from_cache(
                candidate_geometry_cache,
                append_tuple,
            )
        patch = SupportPatch(
            removed_runtime_indices=removed,
            inserted_count=m_insert,
            inserted_labels=inserted_labels,
        )
        geometry = SupportPatchGeometry(
            K_before=base_K,
            f_before=base_f,
            norm_b_sq=float(base_evaluation.geometry.norm_b_sq),
            K_insert_cross=K_cross,
            K_insert_insert=K_insert,
            f_insert=f_insert,
        )
        exchange_score = score_support_patch(
            geometry=geometry,
            patch=patch,
            inverse_policy=inverse_policy,
            schur_inverse_policy=_append_schur_inverse_policy(
                inverse_policy,
                support_config=support_config,
            ),
            schur_candidate_ridge_lambda=float(
                support_config.append_schur_novelty_ridge_lambda
            ),
            before_cache=before_cache,
        )
        delete_score = prune_candidate.score
        if delete_score is None or tuple(delete_score.removed_runtime_indices) != removed:
            raise ValueError("exchange prune scout score does not match delete branch")
        append_score = append_candidate.score
        if append_score is None or int(append_score.inserted_count) != m_insert:
            raise ValueError("exchange append scout score does not match append branch")
    except (ValueError, np.linalg.LinAlgError) as exc:
        candidate = _failed_exchange_candidate_score(
            append_candidate=append_candidate,
            prune_candidate=prune_candidate,
            appended_atoms=append_tuple,
            pruned_atoms=prune_tuple,
            removed_runtime_indices=removed,
            error=str(exc),
        )
        return _ExchangeScoringResult(
            candidate=candidate,
            appended_atoms=append_tuple,
            pruned_atoms=prune_tuple,
        )

    denom = float(exchange_score.denominator)
    gamma_exchange = exchange_score.after_gain
    gamma_delete = delete_score.after_gain
    gamma_append = append_score.after_gain
    conditional_append_gain = _normalized_positive_delta(
        gamma_exchange,
        gamma_delete,
        denom,
    )
    conditional_deletion_loss = _normalized_positive_delta(
        gamma_append,
        gamma_exchange,
        denom,
    )
    append_cost = append_cost_telemetry_for_family(
        (estimate_append_atom_set_cost(append_tuple),),
        insertion_gains=(conditional_append_gain,),
        settings=AppendCostSettings.from_config(support_config),
    )[0]
    prune_conditioning = _exchange_conditioning_components(
        base_step=base_step,
        exchange_score=exchange_score,
        prune_candidate=prune_candidate,
    )
    prune_cost = prune_cost_telemetry_for_family(
        (estimate_prune_atom_set_cost(prune_tuple),),
        deletion_losses=(conditional_deletion_loss,),
        historical_losses=(
            float(prune_candidate.metadata.get("historical_deletion_loss", 0.0)),
        ),
        history_counts=(int(prune_candidate.metadata.get("history_count", 0)),),
        conditioning_components=(prune_conditioning,),
        settings=PruneCostSettings.from_config(support_config),
    )[0]
    append_rank = append_cost.rank_utility
    prune_rank = prune_cost.rank_utility
    patch_delta = 0.0 if exchange_score.normalized_score is None else float(exchange_score.normalized_score)
    scout = float((append_rank or 0.0) + (prune_rank or 0.0))
    numerator = float(
        scout + float(support_config.patch_utility_delta_weight) * patch_delta
    )
    solve_reason = _append_augmented_solve_confirmation_reason(exchange_score)
    schur_reason = _append_schur_guard_reason(exchange_score, support_config=support_config)
    eligible = bool(
        append_rank is not None
        and prune_rank is not None
        and conditional_append_gain is not None
        and conditional_deletion_loss is not None
        and float(conditional_append_gain) >= float(support_config.append_gain_threshold)
        and float(append_rank) >= float(support_config.exchange_append_score_min)
        and float(conditional_deletion_loss) <= float(support_config.prune_loss_threshold)
        and float(prune_rank) >= float(support_config.exchange_prune_score_min)
        and solve_reason == "eligible"
        and schur_reason == "eligible"
        and np.isfinite(numerator)
    )
    reason = "eligible"
    if append_rank is None or conditional_append_gain is None:
        reason = "exchange_missing_conditional_append_gain"
    elif float(conditional_append_gain) < float(support_config.append_gain_threshold):
        reason = "exchange_conditional_append_gain_below_threshold"
    elif float(append_rank) < float(support_config.exchange_append_score_min):
        reason = "exchange_append_score_below_min"
    elif prune_rank is None or conditional_deletion_loss is None:
        reason = "exchange_missing_conditional_deletion_loss"
    elif float(conditional_deletion_loss) > float(support_config.prune_loss_threshold):
        reason = "exchange_conditional_deletion_loss_above_threshold"
    elif float(prune_rank) < float(support_config.exchange_prune_score_min):
        reason = "exchange_prune_score_below_min"
    elif solve_reason != "eligible":
        reason = solve_reason
    elif schur_reason != "eligible":
        reason = schur_reason
    elif not np.isfinite(numerator):
        reason = "exchange_nonfinite_patch_utility"
    metadata = {
        "support_patch_family": SUPPORT_PATCH_EXCHANGE_SELECTION_POLICY_V1,
        "patch_finalist_kind": PATCH_EXCHANGE,
        "atom_ids": [str(atom.atom_id) for atom in prune_tuple],
        "atom_labels": [str(atom.atom_label) for atom in prune_tuple],
        "append_atom_ids": [str(atom.atom_id) for atom in append_tuple],
        "append_atom_labels": [str(atom.atom_label) for atom in append_tuple],
        "exchange_append_candidate_label": append_candidate.candidate_label,
        "exchange_prune_candidate_label": prune_candidate.candidate_label,
        "conditional_append_gain": _finite_or_none(conditional_append_gain),
        "conditional_deletion_loss": _finite_or_none(conditional_deletion_loss),
        "exchange_patch_delta": _finite_or_none(patch_delta),
        "exchange_scout_score": _finite_or_none(scout),
        "exchange_patch_utility_numerator": _finite_or_none(numerator),
        "append_cost": append_cost.to_json_dict(),
        "prune_cost": prune_cost.to_json_dict(),
        "prune_conditioning": dict(prune_conditioning),
        "augmented_solve_confirmation_reason": solve_reason,
        "schur_guard_reason": schur_reason,
        "candidate_key": _prune_batch_key(prune_tuple),
        "removed_runtime_indices": [int(index) for index in removed],
        "inserted_runtime_count": int(m_insert),
        "rung_size": int(len(append_tuple) + len(prune_tuple)),
    }
    candidate = PatchCandidateScore(
        candidate_kind=PATCH_EXCHANGE,
        candidate_label=(
            f"exchange -{_candidate_set_label(prune_tuple)} +"
            f"{_candidate_set_label(append_tuple)}"
        ),
        patch=patch,
        score=replace(
            exchange_score,
            insertion_gain=conditional_append_gain,
            deletion_loss=conditional_deletion_loss,
            rank_score=numerator,
        ),
        rank_score=numerator,
        accepted_eligible=eligible,
        rejection_reason=reason,
        metadata=metadata,
    )
    return _ExchangeScoringResult(
        candidate=candidate,
        appended_atoms=append_tuple,
        pruned_atoms=prune_tuple,
    )


def _exchange_scoring_result_to_finalist(
    result: _ExchangeScoringResult,
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    inverse_policy: McLachlanInversePolicy,
    solve_repair_config: SolveRepairConfig,
    support_config: SupportPatchControllerConfig,
    runtime_state: _PruneControllerRuntimeState,
    repair_dt: float | None,
    time_index: int,
) -> _PatchFinalist:
    candidate = result.candidate
    append_tuple = tuple(result.appended_atoms)
    prune_tuple = tuple(result.pruned_atoms)
    if candidate.score is None:
        return _PatchFinalist(
            candidate=candidate,
            utility=None,
            passed=False,
            reason="exchange_scoring_failed",
            appended_atoms=append_tuple,
            pruned_atoms=prune_tuple,
            metadata=dict(candidate.metadata or {}),
        )
    safety = _evaluate_deletion_patch_safety(
        state=state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_runtime,
        time=float(time),
        base_evaluation=base_evaluation,
        base_step=base_step,
        candidate=candidate,
        pruned_atoms=prune_tuple,
        appended_atoms=append_tuple,
        inverse_policy=inverse_policy,
        solve_repair_config=solve_repair_config,
        support_config=support_config,
        runtime_state=runtime_state,
        repair_dt=repair_dt,
        time_index=int(time_index),
    )
    return _deletion_patch_finalist(
        candidate,
        pruned_atoms=prune_tuple,
        appended_atoms=append_tuple,
        safety=safety,
        support_config=support_config,
    )


def _score_exchange_finalist(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    append_candidate: PatchCandidateScore,
    prune_candidate: PatchCandidateScore,
    appended_atoms: Sequence[SupportAtom],
    pruned_atoms: Sequence[ActiveSupportAtom],
    inverse_policy: McLachlanInversePolicy,
    solve_repair_config: SolveRepairConfig,
    support_config: SupportPatchControllerConfig,
    runtime_state: _PruneControllerRuntimeState,
    repair_dt: float | None,
    time_index: int,
    before_cache: SupportPatchBeforeCache | None = None,
) -> _PatchFinalist:
    result = _score_exchange_candidate_only(
        state=state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_runtime,
        time=float(time),
        base_evaluation=base_evaluation,
        base_step=base_step,
        append_candidate=append_candidate,
        prune_candidate=prune_candidate,
        appended_atoms=appended_atoms,
        pruned_atoms=pruned_atoms,
        inverse_policy=inverse_policy,
        support_config=support_config,
        before_cache=before_cache,
    )
    return _exchange_scoring_result_to_finalist(
        result,
        state=state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_runtime,
        time=float(time),
        base_evaluation=base_evaluation,
        base_step=base_step,
        inverse_policy=inverse_policy,
        solve_repair_config=solve_repair_config,
        support_config=support_config,
        runtime_state=runtime_state,
        repair_dt=repair_dt,
        time_index=int(time_index),
    )


def _normalized_positive_delta(
    after_value: float | None,
    before_value: float | None,
    denominator: float,
) -> float | None:
    if after_value is None or before_value is None:
        return None
    after = float(after_value)
    before = float(before_value)
    denom = float(denominator)
    if not np.isfinite(after) or not np.isfinite(before) or not np.isfinite(denom) or denom <= 0.0:
        return None
    return float(max(0.0, after - before) / denom)


def _exchange_conditioning_components(
    *,
    base_step: FixedMcLachlanStep,
    exchange_score: SupportPatchScore,
    prune_candidate: PatchCandidateScore,
) -> dict[str, float]:
    before = _finite_or_none(base_step.condition_number)
    confirmation = exchange_score.augmented_solve_confirmation
    after = None if confirmation is None else _finite_or_none(confirmation.condition_number)
    if before is not None and after is not None and before > 0.0 and after > 0.0:
        log_before = math.log(float(before))
        log_after = math.log(float(after))
        d_rel = float(max(0.0, log_before - log_after))
        d_dam = float(max(0.0, log_after - log_before))
    else:
        old = _prune_conditioning_components_from_metadata(prune_candidate.metadata)
        d_rel = float(old["d_kappa_rel"])
        d_dam = float(old["d_kappa_dam"])
    old = _prune_conditioning_components_from_metadata(prune_candidate.metadata)
    return {
        "d_kappa_rel": float(d_rel),
        "d_schur": float(old["d_schur"]),
        "d_kappa_schur_hist": float(old["d_kappa_schur_hist"]),
        "d_kappa_dam": float(d_dam),
    }


def _failed_exchange_candidate_score(
    *,
    append_candidate: PatchCandidateScore,
    prune_candidate: PatchCandidateScore,
    appended_atoms: Sequence[SupportAtom],
    pruned_atoms: Sequence[ActiveSupportAtom],
    removed_runtime_indices: Sequence[int],
    error: str,
) -> PatchCandidateScore:
    append_tuple = tuple(appended_atoms)
    prune_tuple = tuple(pruned_atoms)
    patch = SupportPatch(
        removed_runtime_indices=tuple(int(index) for index in removed_runtime_indices),
        inserted_count=int(sum(max(1, int(atom.runtime_count)) for atom in append_tuple)),
        inserted_labels=tuple(str(atom.atom_label) for atom in append_tuple),
    )
    return PatchCandidateScore(
        candidate_kind=PATCH_EXCHANGE,
        candidate_label=(
            f"exchange -{_candidate_set_label(prune_tuple)} +"
            f"{_candidate_set_label(append_tuple)}"
        ),
        patch=patch,
        score=None,
        rank_score=None,
        accepted_eligible=False,
        rejection_reason="exchange_scoring_failed",
        metadata={
            "support_patch_family": SUPPORT_PATCH_EXCHANGE_SELECTION_POLICY_V1,
            "patch_finalist_kind": PATCH_EXCHANGE,
            "atom_ids": [str(atom.atom_id) for atom in prune_tuple],
            "append_atom_ids": [str(atom.atom_id) for atom in append_tuple],
            "exchange_append_candidate_label": append_candidate.candidate_label,
            "exchange_prune_candidate_label": prune_candidate.candidate_label,
            "error": str(error),
        },
    )


def _select_prune_ladder_patch(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    inverse_policy: McLachlanInversePolicy,
    solve_repair_config: SolveRepairConfig,
    support_config: SupportPatchControllerConfig,
    runtime_state: _PruneControllerRuntimeState,
    repair_dt: float | None = None,
    time_index: int = 0,
) -> tuple[
    PatchDecision,
    APMcLachlanState | None,
    np.ndarray | None,
    GeometryEvaluation | None,
    FixedMcLachlanStep | None,
]:
    _validate_prune_ladder_config(support_config)
    runtime_state.ensure_support_identity(state)
    prune_cost_settings = PruneCostSettings.from_config(support_config)
    rank_score_kind = _prune_rank_score_kind_for_config(support_config)
    ladder_metadata = {
        "prune_ladder_enabled": True,
        "prune_ladder_mode": "combinatorial",
        "cost_model_effective": AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1,
        "rank_score_kind": rank_score_kind,
        "cost_settings": prune_cost_settings.to_json_dict(),
        "prefilter_policy_effective": PRUNE_LADDER_PREFILTER_POLICY_V1,
        "commit_enabled": bool(support_config.prune_commit_enabled),
        "max_prune_commits": int(support_config.max_prune_commits),
    }
    if int(support_config.max_prune_batch_size) <= 0:
        batch = _empty_patch_batch(
            time=float(time),
            state=state,
            base_step=base_step,
            reason="prune_batch_size_zero",
            selection_policy=PRUNE_LADDER_SELECTION_POLICY_V1,
            metadata=ladder_metadata,
        )
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=0,
                scored_count=0,
                reason="prune_batch_size_zero",
                batch_evaluation=batch,
            ),
            None,
            None,
            None,
            None,
        )
    atoms = _active_prune_atoms(
        state,
        theta_runtime=theta_runtime,
        support_config=support_config,
        runtime_state=runtime_state,
        time_index=int(time_index),
    )
    if not atoms:
        batch = _empty_patch_batch(
            time=float(time),
            state=state,
            base_step=base_step,
            reason="no_active_prune_atoms",
            selection_policy=PRUNE_LADDER_SELECTION_POLICY_V1,
            metadata=ladder_metadata,
        )
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=0,
                scored_count=0,
                reason="no_active_prune_atoms",
                batch_evaluation=batch,
            ),
            None,
            None,
            None,
            None,
        )

    batch = _score_prune_ladder_batch(
        state=state,
        theta_runtime=theta_runtime,
        time=float(time),
        time_index=int(time_index),
        base_evaluation=base_evaluation,
        base_step=base_step,
        atoms=atoms,
        inverse_policy=inverse_policy,
        support_config=support_config,
        runtime_state=runtime_state,
        metadata=ladder_metadata,
    )
    selected = batch.selected_score
    if selected is None or selected.score is None:
        return (
            PatchDecision(
                patch_kind=PATCH_DELETE,
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                reason=batch.reason or "no_finite_prune_ladder_score",
                batch_evaluation=batch,
            ),
            None,
            None,
            None,
            None,
        )
    if not bool(selected.accepted_eligible):
        return (
            PatchDecision(
                patch_kind=PATCH_DELETE,
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                selected_label=selected.candidate_label,
                selected_score=selected.score,
                reason=str(selected.rejection_reason),
                batch_evaluation=batch,
            ),
            None,
            None,
            None,
            None,
        )
    persistence_metadata = _prune_persistence_metadata(
        runtime_state,
        selected,
        support_config=support_config,
        time_index=int(time_index),
    )
    batch = _batch_with_selected_metadata(batch, persistence_metadata)
    if not _prune_persistence_passed(persistence_metadata):
        return (
            PatchDecision(
                patch_kind=PATCH_DELETE,
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                selected_label=selected.candidate_label,
                selected_score=selected.score,
                reason="prune_persistence_wait",
                batch_evaluation=batch,
                metadata=dict(persistence_metadata),
            ),
            None,
            None,
            None,
            None,
        )
    if not bool(support_config.prune_commit_enabled):
        return (
            PatchDecision(
                patch_kind=PATCH_DELETE,
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                selected_label=selected.candidate_label,
                selected_score=selected.score,
                reason="prune_commit_disabled",
                batch_evaluation=batch,
                metadata=dict(persistence_metadata),
            ),
            None,
            None,
            None,
            None,
        )
    if int(support_config.max_prune_commits) <= 0:
        return (
            PatchDecision(
                patch_kind=PATCH_DELETE,
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                selected_label=selected.candidate_label,
                selected_score=selected.score,
                reason="prune_commit_limit_zero",
                batch_evaluation=batch,
                metadata=dict(persistence_metadata),
            ),
            None,
            None,
            None,
            None,
        )
    if int(runtime_state.accepted_commit_count) >= int(support_config.max_prune_commits):
        return (
            PatchDecision(
                patch_kind=PATCH_DELETE,
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                selected_label=selected.candidate_label,
                selected_score=selected.score,
                reason="prune_commit_limit_reached",
                batch_evaluation=batch,
                metadata=dict(persistence_metadata),
            ),
            None,
            None,
            None,
            None,
        )
    if bool(support_config.prune_shadow_enabled):
        return (
            PatchDecision(
                patch_kind=PATCH_DELETE,
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                selected_label=selected.candidate_label,
                selected_score=selected.score,
                reason="shadow_no_harm_unsupported",
                batch_evaluation=batch,
                metadata={**persistence_metadata, "shadow_enabled": True},
            ),
            None,
            None,
            None,
            None,
        )

    atoms_by_id = {str(atom.atom_id): atom for atom in atoms}
    selected_atom_ids = tuple(str(v) for v in selected.metadata.get("atom_ids", ()))
    selected_atoms = tuple(atoms_by_id[atom_id] for atom_id in selected_atom_ids)
    try:
        pruned_state, theta_pruned, evaluation, step = _materialize_prune_atom_set(
            state=state,
            hamiltonian=hamiltonian,
            theta_runtime=theta_runtime,
            time=float(time),
            atoms=selected_atoms,
            inverse_policy=inverse_policy,
            solve_repair_config=solve_repair_config,
            repair_dt=repair_dt,
            include_tangent_matrix=bool(
                support_config.prune_patch_smoothness_enabled
            ),
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        _cooldown_prune_atoms(
            runtime_state,
            selected_atoms,
            time_index=int(time_index),
            support_config=support_config,
        )
        return (
            PatchDecision(
                patch_kind=PATCH_DELETE,
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                selected_label=selected.candidate_label,
                selected_score=selected.score,
                reason="prune_materialization_failed",
                batch_evaluation=batch,
                metadata={**persistence_metadata, "error": str(exc)},
            ),
            None,
            None,
            None,
            None,
        )
    retry_metadata = _prune_smoothness_retry_metadata(
        runtime_state,
        selected_atoms,
        time_index=int(time_index),
    )
    ray_distance = _state_ray_distance(base_evaluation.psi, evaluation.psi)
    differential_miss = _step_differential_miss(base_step, step)
    if ray_distance is None or ray_distance > float(support_config.prune_ray_distance_tol):
        _cooldown_prune_atoms(
            runtime_state,
            selected_atoms,
            time_index=int(time_index),
            support_config=support_config,
        )
        return (
            PatchDecision(
                patch_kind=PATCH_DELETE,
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                selected_label=selected.candidate_label,
                selected_score=selected.score,
                reason="prune_ray_distance_above_tol",
                batch_evaluation=batch,
                metadata={
                    **persistence_metadata,
                    **retry_metadata,
                    "prune_ray_distance": _finite_or_none(ray_distance),
                },
            ),
            None,
            None,
            None,
            None,
        )
    if (
        differential_miss is None
        or differential_miss > float(support_config.prune_differential_miss_tol)
    ):
        _cooldown_prune_atoms(
            runtime_state,
            selected_atoms,
            time_index=int(time_index),
            support_config=support_config,
        )
        return (
            PatchDecision(
                patch_kind=PATCH_DELETE,
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                selected_label=selected.candidate_label,
                selected_score=selected.score,
                reason="prune_differential_miss_above_tol",
                batch_evaluation=batch,
                metadata={
                    **persistence_metadata,
                    **retry_metadata,
                    "prune_differential_miss": _finite_or_none(differential_miss),
                },
            ),
            None,
            None,
            None,
            None,
        )
    smoothness = _evaluate_prune_patch_smoothness(
        base_evaluation=base_evaluation,
        base_step=base_step,
        patched_evaluation=evaluation,
        patched_step=step,
        support_config=support_config,
    )
    retry_metadata = _prune_smoothness_retry_metadata(
        runtime_state,
        selected_atoms,
        time_index=int(time_index),
        current_smoothness=smoothness,
    )
    smoothness_metadata = {**retry_metadata, **smoothness.to_metadata()}
    if not bool(smoothness.passed):
        cooldown_steps = _prune_smoothness_cooldown_steps(
            smoothness,
            support_config=support_config,
        )
        _cooldown_prune_atoms(
            runtime_state,
            selected_atoms,
            time_index=int(time_index),
            support_config=support_config,
            cooldown_steps=cooldown_steps,
        )
        record = _record_prune_smoothness_deferred(
            runtime_state,
            selected_atoms,
            time_index=int(time_index),
            cooldown_steps=cooldown_steps,
            smoothness=smoothness,
        )
        smoothness_metadata = {
            **smoothness_metadata,
            **record.to_metadata(),
            "prune_patch_smoothness_cooldown_steps": int(cooldown_steps),
        }
        batch = _batch_with_selected_metadata(
            batch,
            {**persistence_metadata, **smoothness_metadata},
        )
        return (
            PatchDecision(
                patch_kind=PATCH_DELETE,
                accepted=False,
                candidate_count=batch.candidate_count,
                scored_count=batch.scored_count,
                selected_label=selected.candidate_label,
                selected_score=selected.score,
                reason=(
                    "prune_patch_smoothness_unavailable"
                    if not bool(smoothness.available)
                    else "prune_patch_smoothness_deferred"
                ),
                batch_evaluation=batch,
                metadata={**persistence_metadata, **smoothness_metadata},
            ),
            None,
            None,
            None,
            None,
        )
    runtime_state.smoothness_deferred.pop(_prune_batch_key(selected_atoms), None)
    batch = _batch_with_selected_metadata(
        batch,
        {**persistence_metadata, **smoothness_metadata},
    )
    return (
        PatchDecision(
            patch_kind=PATCH_DELETE,
            accepted=True,
            candidate_count=batch.candidate_count,
            scored_count=batch.scored_count,
            selected_label=selected.candidate_label,
            selected_score=selected.score,
            reason="accepted_cost_weighted_prune",
            batch_evaluation=batch,
            metadata={
                **persistence_metadata,
                **smoothness_metadata,
                "prune_ray_distance": float(ray_distance),
                "prune_differential_miss": float(differential_miss),
                "shadow_enabled": False,
            },
        ),
        pruned_state,
        theta_pruned,
        evaluation,
        step,
    )


def _score_prune_ladder_batch(
    *,
    state: APMcLachlanState,
    theta_runtime: np.ndarray,
    time: float,
    time_index: int,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    atoms: Sequence[ActiveSupportAtom],
    inverse_policy: McLachlanInversePolicy,
    support_config: SupportPatchControllerConfig,
    runtime_state: _PruneControllerRuntimeState,
    metadata: Mapping[str, Any],
    before_cache: SupportPatchBeforeCache | None = None,
) -> PatchBatchEvaluation:
    base_K = np.asarray(base_evaluation.geometry.K, dtype=float)
    base_f = np.asarray(base_evaluation.geometry.f, dtype=float).reshape(-1)
    n_before = int(state.runtime_parameter_count)
    if base_K.shape != (n_before, n_before):
        raise ValueError(
            "base McLachlan geometry size does not match current runtime support: "
            f"got {base_K.shape}, expected ({n_before}, {n_before})."
        )
    if int(base_f.size) != n_before:
        raise ValueError(
            "base McLachlan force size does not match current runtime support: "
            f"got {base_f.size}, expected {n_before}."
        )

    all_atoms = tuple(atoms)
    max_rung = min(int(support_config.max_prune_batch_size), int(len(all_atoms)))
    history_read_view = _prune_history_read_view(runtime_state)
    scores: list[PatchCandidateScore] = []
    rung_diagnostics: list[RungDiagnostics] = []
    scored_count = 0
    candidate_set_index = 0
    singleton_ranked: list[tuple[ActiveSupportAtom, float, int]] = []

    for rung_size in range(1, max_rung + 1):
        if rung_size == 1:
            source_atoms = all_atoms
            before_prefilter_count = int(len(all_atoms))
            after_prefilter_count = int(len(source_atoms))
        else:
            if not singleton_ranked:
                before_prefilter_count = _comb_count_safe(len(all_atoms), rung_size)
                rung_diagnostics.append(
                    _prune_ladder_rung_diagnostics(
                        rung_size=rung_size,
                        before_prefilter_count=before_prefilter_count,
                        after_prefilter_count=0,
                        attempted_count=0,
                        scored_count=0,
                        best_score=None,
                        best_atom_ids=(),
                        support_config=support_config,
                        rejection_reason="no_finite_singleton_prefilter_atoms",
                    )
                )
                continue
            prefilter_limit = int(support_config.prune_prefilter_size)
            if prefilter_limit > 0:
                source_atoms = tuple(
                    item[0] for item in singleton_ranked[:prefilter_limit]
                )
            else:
                source_atoms = tuple(item[0] for item in singleton_ranked)
            before_prefilter_count = _comb_count_safe(len(all_atoms), rung_size)
            after_prefilter_count = _comb_count_safe(len(source_atoms), rung_size)

        attempted_count = _rung_attempt_count(
            after_prefilter_count,
            cap=int(support_config.prune_rung_set_cap),
        )
        rung_start = int(len(scores))
        rung_best_score: float | None = None
        rung_best_atom_ids: tuple[str, ...] = ()
        rung_scored_count = 0
        if attempted_count > 0:
            rung_score_start = int(len(scores))
            candidate_set_index_start = int(candidate_set_index)
            tasks = tuple(
                (
                    tuple(atom_set),
                    candidate_set_index_start + offset,
                    rung_score_start + offset,
                )
                for offset, atom_set in enumerate(
                    itertools.islice(
                        itertools.combinations(source_atoms, rung_size),
                        attempted_count,
                    )
                )
            )

            def score_task(
                task: tuple[tuple[ActiveSupportAtom, ...], int, int]
            ) -> PatchCandidateScore:
                atom_set, task_candidate_set_index, task_score_index = task
                try:
                    return _score_prune_atom_set(
                        state=state,
                        theta_runtime=theta_runtime,
                        base_K=base_K,
                        base_f=base_f,
                        norm_b_sq=float(base_evaluation.geometry.norm_b_sq),
                        atoms=tuple(atom_set),
                        inverse_policy=inverse_policy,
                        support_config=support_config,
                        runtime_state=history_read_view,
                        before_cache=before_cache,
                        candidate_set_index=task_candidate_set_index,
                        score_index=task_score_index,
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    return _failed_prune_atom_set_score(
                        atoms=tuple(atom_set),
                        rung_size=rung_size,
                        candidate_set_index=task_candidate_set_index,
                        score_index=task_score_index,
                        error=str(exc),
                    )

            for candidate_score in _ordered_parallel_map(
                tasks,
                support_config=support_config,
                score_one=score_task,
            ):
                scores.append(candidate_score)
                if candidate_score.score is not None:
                    scored_count += 1
                    rung_scored_count += 1
            candidate_set_index += len(tasks)
        if rung_size == 1:
            _apply_prune_cost_scores(
                scores,
                support_config=support_config,
                score_indices=range(rung_start, len(scores)),
            )
            singleton_ranked = [
                (
                    all_atoms[int(score.metadata["candidate_set_index"])],
                    float(score.rank_score),
                    int(score.metadata["candidate_set_index"]),
                )
                for score in scores[rung_start:]
                if score.rank_score is not None
                and np.isfinite(float(score.rank_score))
                and int(score.metadata.get("rung_size", -1)) == 1
            ]
            singleton_ranked.sort(key=lambda item: (-float(item[1]), int(item[2])))
        for score in scores[rung_start:]:
            rank = score.rank_score
            if rank is not None and np.isfinite(float(rank)):
                if rung_best_score is None or float(rank) > float(rung_best_score):
                    rung_best_score = float(rank)
                    rung_best_atom_ids = tuple(
                        str(atom_id) for atom_id in score.metadata.get("atom_ids", ())
                    )
        rejection_reason = None
        if after_prefilter_count == 0:
            rejection_reason = "no_candidate_sets_after_prefilter"
        elif attempted_count < after_prefilter_count:
            rejection_reason = "prune_rung_set_cap_applied"
        rung_diagnostics.append(
            _prune_ladder_rung_diagnostics(
                rung_size=rung_size,
                before_prefilter_count=before_prefilter_count,
                after_prefilter_count=after_prefilter_count,
                attempted_count=attempted_count,
                scored_count=rung_scored_count,
                best_score=rung_best_score,
                best_atom_ids=rung_best_atom_ids,
                support_config=support_config,
                rejection_reason=rejection_reason,
            )
        )

    _apply_prune_cost_scores(scores, support_config=support_config)
    best_index: int | None = None
    best_rank: float | None = None
    best_eligible_index: int | None = None
    best_eligible_rank: float | None = None
    for score_index, candidate_score in enumerate(tuple(scores)):
        _record_prune_loss_history(
            runtime_state,
            candidate_score,
            time_index=int(time_index),
            window=int(support_config.prune_history_window),
        )
        _record_prune_conditioning_history(
            runtime_state,
            candidate_score,
            time_index=int(time_index),
            window=int(support_config.prune_history_window),
        )
        rank_score = candidate_score.rank_score
        finite_rank = rank_score is not None and np.isfinite(float(rank_score))
        if not finite_rank:
            continue
        if best_rank is None or float(rank_score) > float(best_rank):
            best_rank = float(rank_score)
            best_index = int(score_index)
        if candidate_score.accepted_eligible and (
            best_eligible_rank is None or float(rank_score) > float(best_eligible_rank)
        ):
            best_eligible_rank = float(rank_score)
            best_eligible_index = int(score_index)
    selected_index = best_eligible_index if best_eligible_index is not None else best_index
    selected_score = None if selected_index is None else scores[int(selected_index)]
    reason = "no_finite_prune_ladder_score"
    if selected_score is not None and selected_score.score is not None:
        deletion_loss = selected_score.score.deletion_loss
        rank_score = selected_score.rank_score
        if deletion_loss is None or rank_score is None or not np.isfinite(float(rank_score)):
            reason = "no_finite_prune_ladder_score"
        elif float(deletion_loss) > float(support_config.prune_loss_threshold):
            reason = "prune_loss_above_threshold"
        elif not bool(selected_score.accepted_eligible):
            reason = str(selected_score.rejection_reason)
        else:
            reason = "selected_cost_weighted_prune"

    return PatchBatchEvaluation(
        time=float(time),
        base_runtime_parameter_count=int(state.runtime_parameter_count),
        base_logical_parameter_count=int(state.logical_parameter_count),
        base_residual_ratio=float(base_step.residual_ratio),
        candidate_count=int(sum(r.candidate_set_count_before_prefilter for r in rung_diagnostics)),
        scored_count=int(scored_count),
        candidate_scores=tuple(scores),
        selected_index=selected_index,
        selected_score=selected_score,
        reason=reason,
        selection_policy=PRUNE_LADDER_SELECTION_POLICY_V1,
        rung_diagnostics=tuple(rung_diagnostics),
        metadata={
            **dict(metadata),
            "candidate_atom_count": int(len(all_atoms)),
            "max_prune_batch_size_effective": int(max_rung),
        },
    )


def _apply_prune_cost_scores(
    scores: list[PatchCandidateScore],
    *,
    support_config: SupportPatchControllerConfig,
    score_indices: Sequence[int] | range | None = None,
) -> None:
    indices = (
        tuple(range(len(scores)))
        if score_indices is None
        else tuple(int(index) for index in score_indices)
    )
    valid_indices = tuple(
        index
        for index in indices
        if 0 <= int(index) < len(scores)
        and scores[int(index)].score is not None
        and isinstance(scores[int(index)].metadata, Mapping)
        and "prune_cost_raw" in scores[int(index)].metadata
    )
    if not valid_indices:
        return
    settings = PruneCostSettings.from_config(support_config)
    raw_estimates = []
    deletion_losses = []
    historical_losses = []
    history_counts = []
    conditioning_components = []
    for index in valid_indices:
        metadata = dict(scores[index].metadata or {})
        raw_estimates.append(_append_cost_raw_from_metadata(metadata["prune_cost_raw"]))
        score = scores[index].score
        deletion_losses.append(None if score is None else score.deletion_loss)
        historical_losses.append(float(metadata.get("historical_deletion_loss", 0.0)))
        history_counts.append(int(metadata.get("history_count", 0)))
        conditioning_components.append(_prune_conditioning_components_from_metadata(metadata))
    telemetry = prune_cost_telemetry_for_family(
        raw_estimates,
        deletion_losses=deletion_losses,
        historical_losses=historical_losses,
        history_counts=history_counts,
        conditioning_components=conditioning_components,
        settings=settings,
    )
    for index, cost in zip(valid_indices, telemetry):
        candidate = scores[index]
        score = candidate.score
        deletion_loss = None if score is None else score.deletion_loss
        rank_utility = cost.rank_utility
        finite_rank = rank_utility is not None and np.isfinite(float(rank_utility))
        eligible = bool(
            finite_rank
            and deletion_loss is not None
            and float(deletion_loss) <= float(support_config.prune_loss_threshold)
        )
        reason = "eligible"
        if not finite_rank:
            reason = "nonfinite_rank_score"
        elif deletion_loss is None:
            reason = "missing_deletion_loss"
        elif float(deletion_loss) > float(support_config.prune_loss_threshold):
            reason = "prune_loss_above_threshold"
        scores[index] = replace(
            candidate,
            rank_score=None if rank_utility is None else float(rank_utility),
            accepted_eligible=eligible,
            rejection_reason=reason,
            metadata={
                **dict(candidate.metadata or {}),
                "prune_cost": cost.to_json_dict(),
                "rank_score_kind": str(cost.rank_score_kind),
            },
        )


def _score_prune_atom_set(
    *,
    state: APMcLachlanState,
    theta_runtime: np.ndarray,
    base_K: np.ndarray,
    base_f: np.ndarray,
    norm_b_sq: float,
    atoms: Sequence[ActiveSupportAtom],
    inverse_policy: McLachlanInversePolicy,
    support_config: SupportPatchControllerConfig,
    runtime_state: _PruneControllerRuntimeState | _PruneHistoryReadView,
    candidate_set_index: int,
    score_index: int,
    before_cache: SupportPatchBeforeCache | None = None,
) -> PatchCandidateScore:
    atom_tuple = tuple(atoms)
    removed = tuple(sorted({int(idx) for atom in atom_tuple for idx in atom.runtime_indices}))
    if not removed:
        return PatchCandidateScore(
            candidate_kind=PATCH_DELETE,
            candidate_label=_candidate_set_label(atom_tuple),
            patch=SupportPatch(removed_runtime_indices=()),
            score=None,
            rank_score=None,
            accepted_eligible=False,
            rejection_reason="candidate_removed_no_runtime_coordinates",
            metadata=_prune_atom_set_metadata(
                atom_tuple,
                candidate_set_index=candidate_set_index,
                score_index=score_index,
            ),
        )
    if int(state.runtime_parameter_count) - int(len(removed)) < int(
        support_config.min_runtime_parameter_count
    ):
        return PatchCandidateScore(
            candidate_kind=PATCH_DELETE,
            candidate_label=_candidate_set_label(atom_tuple),
            patch=SupportPatch(removed_runtime_indices=removed),
            score=None,
            rank_score=None,
            accepted_eligible=False,
            rejection_reason="below_min_runtime_parameter_count",
            metadata=_prune_atom_set_metadata(
                atom_tuple,
                candidate_set_index=candidate_set_index,
                score_index=score_index,
                removed_runtime_indices=removed,
            ),
        )
    patch = SupportPatch(removed_runtime_indices=removed)
    patch_geometry = SupportPatchGeometry(
        K_before=base_K,
        f_before=base_f,
        norm_b_sq=float(norm_b_sq),
    )
    after_cache = build_support_patch_after_cache(
        geometry=patch_geometry,
        patch=patch,
        inverse_policy=inverse_policy,
    )
    score = score_support_patch(
        geometry=patch_geometry,
        patch=patch,
        inverse_policy=inverse_policy,
        before_cache=before_cache,
        after_cache=after_cache,
    )
    conditioning = prune_conditioning_diagnostics(
        geometry=patch_geometry,
        patch=patch,
        inverse_policy=inverse_policy,
        before_cache=before_cache,
        after_cache=after_cache,
    )
    key = _prune_batch_key(atom_tuple)
    history = tuple(runtime_state.loss_history.get(key, ()))
    if int(support_config.prune_history_window) > 0:
        history = history[-int(support_config.prune_history_window) :]
    history_values = tuple(float(value) for _index, value in history)
    historical_loss = float(np.mean(history_values)) if history_values else 0.0
    conditioning_history = tuple(runtime_state.conditioning_history.get(key, ()))
    if int(support_config.prune_history_window) > 0:
        conditioning_history = conditioning_history[
            -int(support_config.prune_history_window) :
        ]
    conditioning_values = tuple(float(value) for _index, value in conditioning_history)
    historical_conditioning = (
        float(np.mean(conditioning_values)) if conditioning_values else 0.0
    )
    deletion_loss = None if score.deletion_loss is None else float(score.deletion_loss)
    raw_cost = estimate_prune_atom_set_cost(atom_tuple)
    finite_loss = deletion_loss is not None and np.isfinite(float(deletion_loss))
    eligible = bool(
        finite_loss and float(deletion_loss) <= float(support_config.prune_loss_threshold)
    )
    reason = "eligible"
    if not finite_loss:
        reason = "missing_deletion_loss"
    elif float(deletion_loss) > float(support_config.prune_loss_threshold):
        reason = "prune_loss_above_threshold"
    return PatchCandidateScore(
        candidate_kind=PATCH_DELETE,
        candidate_label=_candidate_set_label(atom_tuple),
        patch=patch,
        score=score,
        rank_score=None if score.rank_score is None else float(score.rank_score),
        accepted_eligible=eligible,
        rejection_reason=reason,
        metadata=_prune_atom_set_metadata(
            atom_tuple,
            candidate_set_index=candidate_set_index,
            score_index=score_index,
            removed_runtime_indices=removed,
            deletion_loss_full=deletion_loss,
            historical_deletion_loss=historical_loss,
            history_count=len(history_values),
            historical_conditioning_toxicity=historical_conditioning,
            conditioning_history_count=len(conditioning_values),
            prune_conditioning=conditioning.to_json_dict(),
            prune_cost_raw=raw_cost.to_json_dict(),
        ),
    )


def _failed_prune_atom_set_score(
    *,
    atoms: Sequence[ActiveSupportAtom],
    rung_size: int,
    candidate_set_index: int,
    score_index: int,
    error: str,
) -> PatchCandidateScore:
    atom_tuple = tuple(atoms)
    removed = tuple(sorted({int(idx) for atom in atom_tuple for idx in atom.runtime_indices}))
    return PatchCandidateScore(
        candidate_kind=PATCH_DELETE,
        candidate_label=_candidate_set_label(atom_tuple),
        patch=SupportPatch(removed_runtime_indices=removed),
        score=None,
        rank_score=None,
        accepted_eligible=False,
        rejection_reason="candidate_scoring_failed",
        metadata={
            **_prune_atom_set_metadata(
                atom_tuple,
                candidate_set_index=candidate_set_index,
                score_index=score_index,
                removed_runtime_indices=removed,
            ),
            "rung_size": int(rung_size),
            "error": str(error),
        },
    )


def _prune_atom_set_metadata(
    atoms: Sequence[ActiveSupportAtom],
    *,
    candidate_set_index: int,
    score_index: int,
    removed_runtime_indices: Sequence[int] = (),
    deletion_loss_full: float | None = None,
    historical_deletion_loss: float | None = None,
    history_count: int | None = None,
    historical_conditioning_toxicity: float | None = None,
    conditioning_history_count: int | None = None,
    prune_conditioning: Mapping[str, Any] | None = None,
    prune_cost_raw: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    atom_tuple = tuple(atoms)
    return {
        "rung_size": int(len(atom_tuple)),
        "candidate_key": _prune_batch_key(atom_tuple),
        "atom_ids": [str(atom.atom_id) for atom in atom_tuple],
        "persistence_atom_ids": [
            _prune_persistence_atom_id(atom) for atom in atom_tuple
        ],
        "prune_atom_history_identity_kind": "base_atom_id",
        "atom_labels": [str(atom.atom_label) for atom in atom_tuple],
        "atom_parent_labels": [str(atom.parent_label) for atom in atom_tuple],
        "candidate_set_index": int(candidate_set_index),
        "score_index": int(score_index),
        "removed_runtime_indices": [int(index) for index in removed_runtime_indices],
        "deleted_runtime_count": int(len(tuple(removed_runtime_indices))),
        "deletion_loss_full": _finite_or_none(deletion_loss_full),
        "historical_deletion_loss": (
            0.0
            if historical_deletion_loss is None
            else float(historical_deletion_loss)
        ),
        "history_count": 0 if history_count is None else int(history_count),
        "historical_conditioning_toxicity": (
            0.0
            if historical_conditioning_toxicity is None
            else float(historical_conditioning_toxicity)
        ),
        "conditioning_history_count": (
            0 if conditioning_history_count is None else int(conditioning_history_count)
        ),
        "prune_conditioning": dict(prune_conditioning or {}),
        "cost_model_effective": AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1,
        "rank_score_kind": AP_PRUNE_RANK_SCORE_KIND_V1,
        "prune_cost_raw": dict(prune_cost_raw or {}),
    }


def _prune_ladder_rung_diagnostics(
    *,
    rung_size: int,
    before_prefilter_count: int,
    after_prefilter_count: int,
    attempted_count: int,
    scored_count: int,
    best_score: float | None,
    best_atom_ids: Sequence[str],
    support_config: SupportPatchControllerConfig,
    rejection_reason: str | None,
) -> RungDiagnostics:
    return RungDiagnostics(
        rung_size=int(rung_size),
        candidate_set_count_before_prefilter=int(before_prefilter_count),
        candidate_set_count_scored=int(scored_count),
        prefilter_policy=PRUNE_LADDER_PREFILTER_POLICY_V1,
        best_score=best_score,
        best_atom_ids=tuple(str(atom_id) for atom_id in best_atom_ids),
        rejection_reason=rejection_reason,
        metadata={
            "candidate_set_count_after_prefilter": int(after_prefilter_count),
            "candidate_set_count_attempted": int(attempted_count),
            "candidate_set_count_rejected_by_prefilter": int(
                max(0, int(before_prefilter_count) - int(after_prefilter_count))
            ),
            "candidate_set_count_rejected_by_cap": int(
                max(0, int(after_prefilter_count) - int(attempted_count))
            ),
            "prune_rung_set_cap": int(support_config.prune_rung_set_cap),
            "prune_prefilter_size": int(support_config.prune_prefilter_size),
            "cost_model_effective": AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1,
            "rank_score_kind": _prune_rank_score_kind_for_config(support_config),
            "cost_normalization_mode": str(support_config.cost_normalization_mode),
            "prune_cost_alpha": float(support_config.prune_cost_alpha),
        },
    )


def _active_prune_atoms(
    state: APMcLachlanState,
    *,
    theta_runtime: np.ndarray,
    support_config: SupportPatchControllerConfig,
    runtime_state: _PruneControllerRuntimeState,
    time_index: int,
) -> tuple[ActiveSupportAtom, ...]:
    out: list[ActiveSupportAtom] = []
    target_policy = str(
        support_config.prune_appended_origin_target_policy
    ).strip().lower()
    appended_labels = appended_origin_atom_labels(state)
    active_atoms = tuple(active_support_atoms(state, theta_runtime))
    appended_base_counts: dict[str, int] = {}
    for atom in active_atoms:
        if str(atom.atom_label) not in appended_labels:
            continue
        base_atom_id = str(
            dict(atom.metadata or {}).get("base_atom_id", atom.atom_id)
        )
        appended_base_counts[base_atom_id] = int(
            appended_base_counts.get(base_atom_id, 0) + 1
        )
    for atom in active_atoms:
        if (
            target_policy
            in {PRUNE_TARGET_APPENDED_ONLY, PRUNE_TARGET_REDUNDANT_APPENDED_ONLY}
            and str(atom.atom_label) not in appended_labels
        ):
            continue
        if target_policy == PRUNE_TARGET_REDUNDANT_APPENDED_ONLY:
            base_atom_id = str(
                dict(atom.metadata or {}).get("base_atom_id", atom.atom_id)
            )
            if int(appended_base_counts.get(base_atom_id, 0)) <= 1:
                continue
        if bool(support_config.protect_drive_aligned_atoms) and _is_drive_aligned_atom(atom):
            continue
        if int(state.runtime_parameter_count) - int(atom.runtime_count) < int(
            support_config.min_runtime_parameter_count
        ):
            continue
        cooldown_until = runtime_state.cooldown_until_index.get(str(atom.atom_id))
        if cooldown_until is not None and int(time_index) < int(cooldown_until):
            continue
        out.append(atom)
    return tuple(out)


def _is_drive_aligned_atom(atom: ActiveSupportAtom | SupportAtom) -> bool:
    text = " ".join(
        (
            str(atom.atom_id),
            str(atom.atom_label),
            str(atom.parent_label),
        )
    ).lower()
    return "drive_aligned" in text


def _prune_batch_key(atoms: Sequence[ActiveSupportAtom]) -> str:
    parts = []
    for atom in tuple(atoms):
        runtime = ",".join(str(int(index)) for index in atom.runtime_indices)
        parts.append(f"{atom.atom_id}@{runtime}")
    return "|".join(sorted(parts))


def _prune_conditioning_components_from_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, float]:
    diagnostics = dict(metadata.get("prune_conditioning", {}) or {})
    return {
        "d_kappa_rel": _nonnegative_metadata_float(diagnostics.get("d_kappa_rel")),
        "d_schur": _nonnegative_metadata_float(diagnostics.get("d_schur")),
        "d_kappa_schur_hist": _nonnegative_metadata_float(
            metadata.get("historical_conditioning_toxicity")
        ),
        "d_kappa_dam": _nonnegative_metadata_float(diagnostics.get("d_kappa_dam")),
    }


def _prune_rank_score_kind_for_config(
    support_config: SupportPatchControllerConfig,
) -> str:
    if any(
        float(getattr(support_config, name)) > 0.0
        for name in (
            "prune_condition_lambda_kappa_rel",
            "prune_condition_lambda_schur",
            "prune_condition_lambda_kappa_hist",
            "prune_condition_lambda_kappa_dam",
        )
    ):
        return AP_PRUNE_CONDITIONED_RANK_SCORE_KIND_V1
    return AP_PRUNE_RANK_SCORE_KIND_V1


def _nonnegative_metadata_float(value: Any) -> float:
    finite = _finite_or_none(value)
    return 0.0 if finite is None else float(max(0.0, finite))


def _record_prune_loss_history(
    runtime_state: _PruneControllerRuntimeState,
    candidate: PatchCandidateScore,
    *,
    time_index: int,
    window: int,
) -> None:
    if candidate.score is None or candidate.score.deletion_loss is None:
        return
    key = str(candidate.metadata.get("candidate_key", ""))
    if key == "":
        return
    loss = float(candidate.score.deletion_loss)
    if not np.isfinite(loss):
        return
    history = runtime_state.loss_history.setdefault(key, [])
    history[:] = [(index, value) for index, value in history if int(index) != int(time_index)]
    history.append((int(time_index), float(loss)))
    if int(window) > 0 and len(history) > int(window):
        del history[: max(0, len(history) - int(window))]


def _record_prune_conditioning_history(
    runtime_state: _PruneControllerRuntimeState,
    candidate: PatchCandidateScore,
    *,
    time_index: int,
    window: int,
) -> None:
    key = str(candidate.metadata.get("candidate_key", ""))
    if key == "":
        return
    diagnostics = dict(candidate.metadata.get("prune_conditioning", {}) or {})
    value = _nonnegative_metadata_float(diagnostics.get("d_conditioning_toxicity"))
    if value <= 0.0:
        return
    history = runtime_state.conditioning_history.setdefault(key, [])
    history[:] = [(index, old) for index, old in history if int(index) != int(time_index)]
    history.append((int(time_index), float(value)))
    if int(window) > 0 and len(history) > int(window):
        del history[: max(0, len(history) - int(window))]


def _record_prune_candidate_seen(
    runtime_state: _PruneControllerRuntimeState,
    *,
    candidate_key: str,
    time_index: int,
) -> int:
    key = str(candidate_key)
    previous_index = runtime_state.last_seen_index.get(key)
    previous_streak = int(runtime_state.eligible_streak.get(key, 0))
    streak = previous_streak + 1 if previous_index == int(time_index) - 1 else 1
    runtime_state.last_seen_index[key] = int(time_index)
    runtime_state.eligible_streak[key] = int(streak)
    return int(streak)


def _prune_atom_ids_from_score(candidate: PatchCandidateScore) -> tuple[str, ...]:
    atom_ids = candidate.metadata.get(
        "persistence_atom_ids",
        candidate.metadata.get("atom_ids", ()),
    )
    return tuple(str(atom_id) for atom_id in tuple(atom_ids) if str(atom_id) != "")


def _prune_persistence_atom_id(atom: ActiveSupportAtom) -> str:
    metadata = dict(atom.metadata or {})
    base_atom_id = str(metadata.get("base_atom_id", "")).strip()
    return base_atom_id if base_atom_id else str(atom.atom_id)


def _record_prune_atoms_seen(
    runtime_state: _PruneControllerRuntimeState,
    *,
    atom_ids: Sequence[str],
    time_index: int,
    window: int,
) -> dict[str, int]:
    current_index = int(time_index)
    out: dict[str, int] = {}
    for atom_id in tuple(str(atom_id) for atom_id in atom_ids if str(atom_id) != ""):
        history = runtime_state.atom_seen_history.setdefault(atom_id, [])
        history[:] = [int(index) for index in history if int(index) != current_index]
        history.append(current_index)
        if int(window) > 0:
            lower_bound = current_index - max(0, int(window) - 1)
            history[:] = [int(index) for index in history if int(index) >= lower_bound]
        history.sort()
        out[atom_id] = int(len(history))
    return out


def _prune_atom_history_persistence_metadata(
    runtime_state: _PruneControllerRuntimeState,
    *,
    atom_ids: Sequence[str],
    time_index: int,
    window: int,
    required: int,
    fraction_required: float,
) -> dict[str, Any]:
    atom_tuple = tuple(str(atom_id) for atom_id in atom_ids if str(atom_id) != "")
    counts = _record_prune_atoms_seen(
        runtime_state,
        atom_ids=atom_tuple,
        time_index=int(time_index),
        window=int(window),
    )
    required_count = max(1, int(required))
    total = int(len(atom_tuple))
    passing = sum(1 for atom_id in atom_tuple if int(counts.get(atom_id, 0)) >= required_count)
    fraction = 1.0 if total == 0 else float(passing) / float(total)
    threshold = min(1.0, max(0.0, float(fraction_required)))
    return {
        "prune_persistence_mode": PRUNE_PERSISTENCE_ATOM_HISTORY,
        "prune_persistence_count": int(passing),
        "prune_persistence_required": int(required_count),
        "prune_atom_history_pass_count": int(passing),
        "prune_atom_history_total_count": int(total),
        "prune_atom_history_fraction": float(fraction),
        "prune_atom_history_fraction_required": float(threshold),
        "prune_atom_history_min_count": (
            0 if total == 0 else int(min(counts.get(atom_id, 0) for atom_id in atom_tuple))
        ),
        "prune_atom_history_counts": {str(k): int(v) for k, v in counts.items()},
        "prune_atom_history_passed": bool(fraction >= threshold),
    }


def _prune_persistence_metadata(
    runtime_state: _PruneControllerRuntimeState,
    candidate: PatchCandidateScore,
    *,
    support_config: SupportPatchControllerConfig,
    time_index: int,
) -> dict[str, Any]:
    required = max(1, int(support_config.prune_persistence_required))
    mode = str(support_config.prune_persistence_mode).strip().lower()
    if mode == PRUNE_PERSISTENCE_ATOM_HISTORY:
        return _prune_atom_history_persistence_metadata(
            runtime_state,
            atom_ids=_prune_atom_ids_from_score(candidate),
            time_index=int(time_index),
            window=int(support_config.prune_history_window),
            required=int(required),
            fraction_required=float(support_config.prune_atom_history_fraction),
        )
    streak = _record_prune_candidate_seen(
        runtime_state,
        candidate_key=str(candidate.metadata.get("candidate_key", "")),
        time_index=int(time_index),
    )
    return {
        "prune_persistence_mode": PRUNE_PERSISTENCE_EXACT_BATCH,
        "prune_persistence_count": int(streak),
        "prune_persistence_required": int(required),
        "prune_atom_history_passed": None,
    }


def _prune_persistence_passed(metadata: Mapping[str, Any]) -> bool:
    if str(metadata.get("prune_persistence_mode")) == PRUNE_PERSISTENCE_ATOM_HISTORY:
        return bool(metadata.get("prune_atom_history_passed", False))
    return int(metadata.get("prune_persistence_count", 0)) >= int(
        metadata.get("prune_persistence_required", 1)
    )


def _cooldown_prune_atoms(
    runtime_state: _PruneControllerRuntimeState,
    atoms: Sequence[ActiveSupportAtom],
    *,
    time_index: int,
    support_config: SupportPatchControllerConfig,
    cooldown_steps: int | None = None,
) -> None:
    steps = (
        int(support_config.prune_cooldown_steps)
        if cooldown_steps is None
        else int(cooldown_steps)
    )
    until = int(time_index) + max(0, int(steps))
    for atom in tuple(atoms):
        runtime_state.cooldown_until_index[str(atom.atom_id)] = int(until)


def _batch_with_selected_metadata(
    batch: PatchBatchEvaluation,
    metadata: Mapping[str, Any],
) -> PatchBatchEvaluation:
    selected = batch.selected_score
    if selected is None:
        return batch
    candidate_scores = list(batch.candidate_scores)
    selected_index = batch.selected_index
    if selected_index is None:
        return batch
    candidate_scores[int(selected_index)] = replace(
        selected,
        metadata={**dict(selected.metadata or {}), **dict(metadata)},
    )
    return replace(
        batch,
        candidate_scores=tuple(candidate_scores),
        selected_score=candidate_scores[int(selected_index)],
    )


def _materialize_prune_atom_set(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    atoms: Sequence[ActiveSupportAtom],
    inverse_policy: McLachlanInversePolicy,
    solve_repair_config: SolveRepairConfig,
    repair_dt: float | None = None,
    include_tangent_matrix: bool = False,
) -> tuple[APMcLachlanState, np.ndarray, GeometryEvaluation, FixedMcLachlanStep]:
    pruned_state, theta_pruned = state_without_active_atoms(
        state,
        tuple(atoms),
        theta_runtime=theta_runtime,
    )
    theta_pruned = np.asarray(theta_pruned, dtype=float).reshape(-1)
    evaluation = evaluate_mclachlan_geometry(
        state=pruned_state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_pruned,
        time=float(time),
        include_tangent_matrix=bool(include_tangent_matrix),
    )
    step = _solve_fixed_step_for_trajectory(
        evaluation.geometry,
        inverse_policy=inverse_policy,
        solve_repair_config=solve_repair_config,
        repair_dt=repair_dt,
    )
    return pruned_state, theta_pruned, evaluation, step


def _evaluate_prune_patch_smoothness(
    *,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    patched_evaluation: GeometryEvaluation,
    patched_step: FixedMcLachlanStep,
    support_config: SupportPatchControllerConfig,
    refit_mode: str = "zero_transport_only",
) -> _PrunePatchSmoothnessEvaluation:
    threshold = float(support_config.prune_patch_smoothness_eta_max)
    denom = residual_denominator(
        float(base_evaluation.geometry.norm_b_sq),
        float(base_step.inverse_policy.epsilon),
    )
    if not bool(support_config.prune_patch_smoothness_enabled):
        return _PrunePatchSmoothnessEvaluation(
            available=True,
            reason="disabled",
            eta=None,
            eta_threshold=threshold,
            severity=None,
            passed=True,
            defer=False,
            velocity_jump_l2=None,
            base_velocity_l2=None,
            patched_velocity_l2=None,
            denominator=float(denom),
            phase_alignment_abs_overlap=None,
            refit_mode=str(refit_mode),
        )
    try:
        base_velocity = state_space_velocity_from_evaluation(
            base_evaluation,
            base_step.theta_dot,
        )
        patched_velocity = state_space_velocity_from_evaluation(
            patched_evaluation,
            patched_step.theta_dot,
        )
        phase, overlap_abs = _patch_phase_alignment(
            base_evaluation.psi,
            patched_evaluation.psi,
        )
        patched_aligned = np.asarray(phase * patched_velocity, dtype=complex).reshape(-1)
        if base_velocity.shape != patched_aligned.shape:
            raise ValueError(
                "base and patched state-space velocities must have the same Hilbert "
                f"dimension; got {base_velocity.shape} and {patched_aligned.shape}."
            )
        delta = np.asarray(patched_aligned - base_velocity, dtype=complex).reshape(-1)
        jump_sq = float(np.real(np.vdot(delta, delta)))
        if not np.isfinite(jump_sq):
            raise ValueError("patch velocity jump is non-finite.")
        eta = float(max(0.0, jump_sq) / float(denom))
        severity = float(math.sqrt(max(0.0, eta) / max(threshold, 1.0e-300)))
        passed = bool(severity <= 1.0)
        return _PrunePatchSmoothnessEvaluation(
            available=True,
            reason="passed" if passed else "deferred",
            eta=eta,
            eta_threshold=threshold,
            severity=severity,
            passed=passed,
            defer=not passed,
            velocity_jump_l2=float(math.sqrt(max(0.0, jump_sq))),
            base_velocity_l2=float(np.linalg.norm(base_velocity)),
            patched_velocity_l2=float(np.linalg.norm(patched_aligned)),
            denominator=float(denom),
            phase_alignment_abs_overlap=overlap_abs,
            refit_mode=str(refit_mode),
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _PrunePatchSmoothnessEvaluation(
            available=False,
            reason=f"unavailable:{exc}",
            eta=None,
            eta_threshold=threshold,
            severity=None,
            passed=False,
            defer=True,
            velocity_jump_l2=None,
            base_velocity_l2=None,
            patched_velocity_l2=None,
            denominator=float(denom),
            phase_alignment_abs_overlap=None,
            refit_mode=str(refit_mode),
        )


def _patch_phase_alignment(
    base_psi: np.ndarray,
    patched_psi: np.ndarray,
) -> tuple[complex, float | None]:
    base = np.asarray(base_psi, dtype=complex).reshape(-1)
    patched = np.asarray(patched_psi, dtype=complex).reshape(-1)
    if base.shape != patched.shape or base.size == 0:
        return 1.0 + 0.0j, None
    nb = float(np.linalg.norm(base))
    npatched = float(np.linalg.norm(patched))
    if nb <= 0.0 or npatched <= 0.0:
        return 1.0 + 0.0j, None
    overlap = np.vdot(base / nb, patched / npatched)
    overlap_abs = float(abs(overlap))
    if not np.isfinite(overlap_abs) or overlap_abs <= 0.0:
        return 1.0 + 0.0j, None
    return complex(np.conj(overlap) / overlap_abs), overlap_abs


def _prune_smoothness_cooldown_steps(
    smoothness: _PrunePatchSmoothnessEvaluation,
    *,
    support_config: SupportPatchControllerConfig,
) -> int:
    p_min = max(0, int(support_config.prune_cooldown_steps))
    p_max = max(p_min, int(support_config.prune_patch_smoothness_cooldown_max_steps))
    severity = 1.0 if smoothness.severity is None else float(smoothness.severity)
    scale = max(1.0e-12, float(support_config.prune_patch_smoothness_severity_scale))
    fraction = min(1.0, max(0.0, (severity - 1.0) / scale))
    return int(math.ceil(float(p_min) + float(p_max - p_min) * fraction))


def _record_prune_smoothness_deferred(
    runtime_state: _PruneControllerRuntimeState,
    atoms: Sequence[ActiveSupportAtom],
    *,
    time_index: int,
    cooldown_steps: int,
    smoothness: _PrunePatchSmoothnessEvaluation,
) -> _PruneSmoothnessDeferredRecord:
    atom_tuple = tuple(atoms)
    candidate_key = _prune_batch_key(atom_tuple)
    atom_ids = tuple(str(atom.atom_id) for atom in atom_tuple)
    removed_runtime_indices = tuple(
        sorted({int(idx) for atom in atom_tuple for idx in atom.runtime_indices})
    )
    existing = runtime_state.smoothness_deferred.get(candidate_key)
    eta_history = [] if existing is None else list(existing.eta_history)
    severity_history = [] if existing is None else list(existing.severity_history)
    eta_value = _finite_or_none(smoothness.eta)
    severity_value = _finite_or_none(smoothness.severity)
    if eta_value is not None:
        eta_history.append((int(time_index), float(eta_value)))
    if severity_value is not None:
        severity_history.append((int(time_index), float(severity_value)))
    record = _PruneSmoothnessDeferredRecord(
        candidate_key=candidate_key,
        atom_ids=atom_ids,
        removed_runtime_indices=removed_runtime_indices,
        first_deferred_index=(
            int(time_index)
            if existing is None
            else int(existing.first_deferred_index)
        ),
        last_deferred_index=int(time_index),
        attempt_count=1 if existing is None else int(existing.attempt_count) + 1,
        cooldown_until_index=int(time_index) + max(0, int(cooldown_steps)),
        last_eta=eta_value,
        last_severity=severity_value,
        eta_history=eta_history[-8:],
        severity_history=severity_history[-8:],
    )
    runtime_state.smoothness_deferred[candidate_key] = record
    return record


def _prune_smoothness_retry_metadata(
    runtime_state: _PruneControllerRuntimeState,
    atoms: Sequence[ActiveSupportAtom],
    *,
    time_index: int,
    current_smoothness: _PrunePatchSmoothnessEvaluation | None = None,
) -> dict[str, Any]:
    candidate_key = _prune_batch_key(tuple(atoms))
    record = runtime_state.smoothness_deferred.get(candidate_key)
    if record is None:
        return {
            "prune_patch_smoothness_retry_from_deferred": False,
            "prune_patch_smoothness_attempt_count": 1,
        }
    metadata = {
        **record.to_metadata(),
        "prune_patch_smoothness_retry_from_deferred": True,
        "prune_patch_smoothness_attempt_count": int(record.attempt_count) + 1,
    }
    eta_now = None if current_smoothness is None else _finite_or_none(current_smoothness.eta)
    if eta_now is not None and record.eta_history:
        prev_index, prev_eta = record.eta_history[-1]
        delta_index = int(time_index) - int(prev_index)
        if delta_index > 0:
            slope = (float(eta_now) - float(prev_eta)) / float(delta_index)
            if slope < 0.0:
                threshold = (
                    0.0
                    if current_smoothness is None
                    else float(current_smoothness.eta_threshold)
                )
                predicted = int(
                    math.ceil(
                        float(time_index)
                        + max(0.0, (float(eta_now) - threshold) / abs(slope))
                    )
                )
                trend = "decreasing"
            elif slope > 0.0:
                predicted = None
                trend = "increasing"
            else:
                predicted = None
                trend = "flat"
            metadata.update(
                {
                    "prune_patch_smoothness_trend_direction": trend,
                    "prune_patch_smoothness_trend_slope_per_index": float(slope),
                    "prune_patch_smoothness_predicted_ready_index": predicted,
                }
            )
    return metadata


def _state_ray_distance(psi_a: np.ndarray, psi_b: np.ndarray) -> float | None:
    a = np.asarray(psi_a, dtype=complex).reshape(-1)
    b = np.asarray(psi_b, dtype=complex).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        return None
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return None
    overlap = abs(np.vdot(a / na, b / nb))
    return float(math.acos(min(1.0, max(0.0, float(overlap)))))


def _step_differential_miss(
    base_step: FixedMcLachlanStep,
    pruned_step: FixedMcLachlanStep,
) -> float | None:
    base = _finite_or_none(getattr(base_step, "rho_expr", None))
    pruned = _finite_or_none(getattr(pruned_step, "rho_expr", None))
    if base is None or pruned is None:
        base = _finite_or_none(getattr(base_step, "residual_ratio", None))
        pruned = _finite_or_none(getattr(pruned_step, "residual_ratio", None))
    if base is None or pruned is None:
        return None
    return float(max(0.0, float(pruned) - float(base)))


def _select_append_patch(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    inverse_policy: McLachlanInversePolicy,
    solve_repair_config: SolveRepairConfig,
    controller_config: AppendControllerConfig,
    repair_dt: float | None = None,
) -> tuple[
    PatchDecision,
    APMcLachlanState | None,
    np.ndarray | None,
    GeometryEvaluation | None,
    FixedMcLachlanStep | None,
]:
    if float(base_step.residual_ratio) < float(controller_config.residual_ratio_threshold):
        batch = _empty_patch_batch(
            time=float(time),
            state=state,
            base_step=base_step,
            reason="residual_below_threshold",
        )
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=0,
                scored_count=0,
                reason="residual_below_threshold",
                batch_evaluation=batch,
            ),
            None,
            None,
            None,
            None,
        )
    candidates = _append_candidates(
        state,
        max_candidates=int(controller_config.max_append_candidates),
        allow_incomplete=bool(controller_config.allow_incomplete_candidate_pool),
    )
    if not candidates:
        batch = _empty_patch_batch(
            time=float(time),
            state=state,
            base_step=base_step,
            reason="no_append_candidates",
        )
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=0,
                scored_count=0,
                reason="no_append_candidates",
                batch_evaluation=batch,
            ),
            None,
            None,
            None,
            None,
        )

    batch = _score_append_batch(
        state=state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_runtime,
        time=float(time),
        base_evaluation=base_evaluation,
        base_step=base_step,
        candidates=candidates,
        inverse_policy=inverse_policy,
        controller_config=controller_config,
    )
    selected = batch.selected_score
    if selected is None or selected.score is None:
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=len(candidates),
                scored_count=batch.scored_count,
                reason=batch.reason or "no_finite_append_score",
                batch_evaluation=batch,
            ),
            None,
            None,
            None,
            None,
        )
    selected_index = batch.selected_index
    if selected_index is None:
        raise RuntimeError("append batch selected a score without a selected index.")
    candidate_ordinal = int(selected.metadata.get("candidate_index", selected_index))
    candidate = candidates[candidate_ordinal]
    score = selected.score
    insertion_gain = 0.0 if score.insertion_gain is None else float(score.insertion_gain)
    if insertion_gain < float(controller_config.append_gain_threshold):
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=len(candidates),
                scored_count=batch.scored_count,
                selected_label=str(getattr(candidate, "label", "")),
                selected_score=score,
                reason=batch.reason or "append_gain_below_threshold",
                batch_evaluation=batch,
            ),
            None,
            None,
            None,
            None,
        )
    if not bool(selected.accepted_eligible):
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=len(candidates),
                scored_count=batch.scored_count,
                selected_label=str(getattr(candidate, "label", "")),
                selected_score=score,
                reason=str(selected.rejection_reason),
                batch_evaluation=batch,
            ),
            None,
            None,
            None,
            None,
        )
    materialized = _materialize_append_candidate(
        state=state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_runtime,
        time=float(time),
        candidate=candidate,
        inverse_policy=inverse_policy,
        solve_repair_config=solve_repair_config,
        repair_dt=repair_dt,
    )
    appended_state, theta_aug, evaluation, step = materialized
    return (
        PatchDecision(
            patch_kind=PATCH_APPEND,
            accepted=True,
            candidate_count=len(candidates),
            scored_count=batch.scored_count,
            selected_label=str(getattr(candidate, "label", "")),
            selected_score=score,
            reason=batch.reason or "accepted_best_append_gain",
            batch_evaluation=batch,
        ),
        appended_state,
        theta_aug,
        evaluation,
        step,
    )


def _score_append_batch(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    candidates: Sequence[Any],
    inverse_policy: McLachlanInversePolicy,
    controller_config: AppendControllerConfig,
) -> PatchBatchEvaluation:
    """Score all append candidates against one frozen time point."""

    base_K = np.asarray(base_evaluation.geometry.K, dtype=float)
    base_f = np.asarray(base_evaluation.geometry.f, dtype=float).reshape(-1)
    n_before = int(state.runtime_parameter_count)
    if base_K.shape != (n_before, n_before):
        raise ValueError(
            "base McLachlan geometry size does not match current runtime support: "
            f"got {base_K.shape}, expected ({n_before}, {n_before})."
        )
    if int(base_f.size) != n_before:
        raise ValueError(
            "base McLachlan force size does not match current runtime support: "
            f"got {base_f.size}, expected {n_before}."
        )
    scores: list[PatchCandidateScore] = []
    scored_count = 0
    best_index: int | None = None
    best_rank_score: float | None = None
    best_eligible_index: int | None = None
    best_eligible_rank_score: float | None = None
    for candidate_index, candidate in enumerate(tuple(candidates)):
        candidate_label = str(getattr(candidate, "label", ""))
        try:
            appended_state = state_with_appended_terms(
                state,
                (candidate,),
                theta_runtime=theta_runtime,
            )
            if (
                appended_state.runtime_coordinate_labels[:n_before]
                != state.runtime_coordinate_labels
            ):
                raise ValueError(
                    "appended candidate does not preserve existing runtime coordinates "
                    "as a prefix."
                )
            theta_aug = np.asarray(appended_state.theta_runtime, dtype=float).reshape(-1)
            m_insert = int(appended_state.runtime_parameter_count) - int(n_before)
            if m_insert <= 0:
                scores.append(
                    PatchCandidateScore(
                        candidate_kind=PATCH_APPEND,
                        candidate_label=candidate_label,
                        patch=SupportPatch(inserted_count=0),
                        score=None,
                        rank_score=None,
                        accepted_eligible=False,
                        rejection_reason="candidate_added_no_runtime_coordinates",
                        metadata={"candidate_index": int(candidate_index)},
                    )
                )
                continue
            evaluation = evaluate_mclachlan_geometry(
                state=appended_state,
                hamiltonian=hamiltonian,
                theta_runtime=theta_aug,
                time=float(time),
            )
            K = np.asarray(evaluation.geometry.K, dtype=float)
            f = np.asarray(evaluation.geometry.f, dtype=float).reshape(-1)
            patch = SupportPatch(
                inserted_count=int(m_insert),
                inserted_labels=tuple(appended_state.runtime_coordinate_labels[n_before:]),
            )
            patch_geometry = SupportPatchGeometry(
                K_before=base_K,
                f_before=base_f,
                norm_b_sq=float(base_evaluation.geometry.norm_b_sq),
                K_insert_cross=K[:n_before, n_before:],
                K_insert_insert=K[n_before:, n_before:],
                f_insert=f[n_before:],
            )
            score = score_support_patch(
                geometry=patch_geometry,
                patch=patch,
                inverse_policy=inverse_policy,
            )
            scored_count += 1
            rank_score = score.rank_score
            finite_rank = rank_score is not None and np.isfinite(float(rank_score))
            insertion_gain = (
                None if score.insertion_gain is None else float(score.insertion_gain)
            )
            solve_reason = _append_augmented_solve_confirmation_reason(score)
            solve_ok = solve_reason == "eligible"
            eligible = bool(
                finite_rank
                and insertion_gain is not None
                and insertion_gain >= float(controller_config.append_gain_threshold)
                and solve_ok
            )
            reason = "eligible"
            if not finite_rank:
                reason = "nonfinite_rank_score"
            elif insertion_gain is None:
                reason = "missing_insertion_gain"
            elif insertion_gain < float(controller_config.append_gain_threshold):
                reason = "append_gain_below_threshold"
            elif not solve_ok:
                reason = solve_reason
            candidate_score = PatchCandidateScore(
                candidate_kind=PATCH_APPEND,
                candidate_label=candidate_label,
                patch=patch,
                score=score,
                rank_score=None if rank_score is None else float(rank_score),
                accepted_eligible=eligible,
                rejection_reason=reason,
                metadata={
                    "candidate_index": int(candidate_index),
                    "augmented_solve_confirmation_reason": solve_reason,
                },
            )
            scores.append(candidate_score)
            if finite_rank and (
                best_rank_score is None or float(rank_score) > float(best_rank_score)
            ):
                best_index = int(candidate_index)
                best_rank_score = float(rank_score)
            if candidate_score.accepted_eligible and (
                best_eligible_rank_score is None
                or float(rank_score) > float(best_eligible_rank_score)
            ):
                best_eligible_index = int(candidate_index)
                best_eligible_rank_score = float(rank_score)
        except (ValueError, np.linalg.LinAlgError) as exc:
            scores.append(
                PatchCandidateScore(
                    candidate_kind=PATCH_APPEND,
                    candidate_label=candidate_label,
                    patch=SupportPatch(inserted_count=0),
                    score=None,
                    rank_score=None,
                    accepted_eligible=False,
                    rejection_reason="candidate_scoring_failed",
                    metadata={
                        "candidate_index": int(candidate_index),
                        "error": str(exc),
                    },
                )
            )

    selected_index = best_eligible_index if best_eligible_index is not None else best_index
    selected_score = None if selected_index is None else scores[int(selected_index)]
    reason = "no_finite_append_score"
    if selected_score is not None and selected_score.score is not None:
        insertion_gain = selected_score.score.insertion_gain
        if (
            insertion_gain is not None
            and float(insertion_gain) >= float(controller_config.append_gain_threshold)
            and bool(selected_score.accepted_eligible)
        ):
            reason = "accepted_best_append_gain"
        elif not bool(selected_score.accepted_eligible):
            reason = str(selected_score.rejection_reason)
        else:
            reason = "append_gain_below_threshold"
    return PatchBatchEvaluation(
        time=float(time),
        base_runtime_parameter_count=int(state.runtime_parameter_count),
        base_logical_parameter_count=int(state.logical_parameter_count),
        base_residual_ratio=float(base_step.residual_ratio),
        candidate_count=int(len(candidates)),
        scored_count=int(scored_count),
        candidate_scores=tuple(scores),
        selected_index=selected_index,
        selected_score=selected_score,
        reason=reason,
        selection_policy=APPEND_BATCH_SELECTION_POLICY_V1,
    )


def _select_prune_patch(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    inverse_policy: McLachlanInversePolicy,
    solve_repair_config: SolveRepairConfig,
    controller_config: AppendControllerConfig,
    repair_dt: float | None = None,
) -> tuple[
    PatchDecision,
    APMcLachlanState | None,
    np.ndarray | None,
    GeometryEvaluation | None,
    FixedMcLachlanStep | None,
]:
    """Legacy complete-term singleton prune path; not the Paper-II active route."""

    candidates = _prune_candidates(
        state,
        max_candidates=int(controller_config.max_prune_candidates),
        min_logical_parameter_count=int(controller_config.min_logical_parameter_count),
    )
    if not candidates:
        batch = _empty_patch_batch(
            time=float(time),
            state=state,
            base_step=base_step,
            reason="no_prune_candidates",
        )
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=0,
                scored_count=0,
                reason="no_prune_candidates",
                batch_evaluation=batch,
            ),
            None,
            None,
            None,
            None,
        )

    batch = _score_prune_batch(
        state=state,
        time=float(time),
        base_evaluation=base_evaluation,
        base_step=base_step,
        candidates=candidates,
        inverse_policy=inverse_policy,
        controller_config=controller_config,
    )
    selected = batch.selected_score
    if selected is None or selected.score is None:
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=len(candidates),
                scored_count=batch.scored_count,
                reason=batch.reason or "no_finite_prune_score",
                batch_evaluation=batch,
            ),
            None,
            None,
            None,
            None,
        )
    selected_index = batch.selected_index
    if selected_index is None:
        raise RuntimeError("prune batch selected a score without a selected index.")
    candidate_ordinal = int(selected.metadata.get("candidate_index", selected_index))
    candidate = candidates[candidate_ordinal]
    score = selected.score
    deletion_loss = 0.0 if score.deletion_loss is None else float(score.deletion_loss)
    if deletion_loss > float(controller_config.prune_loss_threshold):
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=len(candidates),
                scored_count=batch.scored_count,
                selected_label=str(candidate["label"]),
                selected_score=score,
                reason=batch.reason or "prune_loss_above_threshold",
                batch_evaluation=batch,
            ),
            None,
            None,
            None,
            None,
        )

    pruned_state, theta_pruned, evaluation, step = _materialize_prune_candidate(
        state=state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_runtime,
        time=float(time),
        candidate=candidate,
        inverse_policy=inverse_policy,
        solve_repair_config=solve_repair_config,
        repair_dt=repair_dt,
    )
    return (
        PatchDecision(
            patch_kind=PATCH_DELETE,
            accepted=True,
            candidate_count=len(candidates),
            scored_count=batch.scored_count,
            selected_label=str(candidate["label"]),
            selected_score=score,
            reason=batch.reason or "accepted_best_prune_loss",
            batch_evaluation=batch,
        ),
        pruned_state,
        theta_pruned,
        evaluation,
        step,
    )


def _score_prune_batch(
    *,
    state: APMcLachlanState,
    time: float,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    candidates: Sequence[Mapping[str, Any]],
    inverse_policy: McLachlanInversePolicy,
    controller_config: AppendControllerConfig,
) -> PatchBatchEvaluation:
    """Legacy complete-block prune scoring; not the Paper-II active route."""

    base_K = np.asarray(base_evaluation.geometry.K, dtype=float)
    base_f = np.asarray(base_evaluation.geometry.f, dtype=float).reshape(-1)
    n_before = int(state.runtime_parameter_count)
    if base_K.shape != (n_before, n_before):
        raise ValueError(
            "base McLachlan geometry size does not match current runtime support: "
            f"got {base_K.shape}, expected ({n_before}, {n_before})."
        )
    if int(base_f.size) != n_before:
        raise ValueError(
            "base McLachlan force size does not match current runtime support: "
            f"got {base_f.size}, expected {n_before}."
        )
    scores: list[PatchCandidateScore] = []
    scored_count = 0
    best_index: int | None = None
    best_rank_score: float | None = None
    for candidate_index, candidate in enumerate(tuple(candidates)):
        candidate_label = str(candidate["label"])
        removed_indices = tuple(int(i) for i in candidate["removed_runtime_indices"])
        try:
            patch = SupportPatch(removed_runtime_indices=removed_indices)
            patch_geometry = SupportPatchGeometry(
                K_before=base_K,
                f_before=base_f,
                norm_b_sq=float(base_evaluation.geometry.norm_b_sq),
            )
            score = score_support_patch(
                geometry=patch_geometry,
                patch=patch,
                inverse_policy=inverse_policy,
            )
            scored_count += 1
            rank_score = score.rank_score
            finite_rank = rank_score is not None and np.isfinite(float(rank_score))
            deletion_loss = (
                None if score.deletion_loss is None else float(score.deletion_loss)
            )
            eligible = bool(
                finite_rank
                and deletion_loss is not None
                and deletion_loss <= float(controller_config.prune_loss_threshold)
            )
            reason = "eligible"
            if not finite_rank:
                reason = "nonfinite_rank_score"
            elif deletion_loss is None:
                reason = "missing_deletion_loss"
            elif deletion_loss > float(controller_config.prune_loss_threshold):
                reason = "prune_loss_above_threshold"
            candidate_score = PatchCandidateScore(
                candidate_kind=PATCH_DELETE,
                candidate_label=candidate_label,
                patch=patch,
                score=score,
                rank_score=None if rank_score is None else float(rank_score),
                accepted_eligible=eligible,
                rejection_reason=reason,
                metadata={
                    "candidate_index": int(candidate_index),
                    "removed_runtime_indices": [int(i) for i in removed_indices],
                },
            )
            scores.append(candidate_score)
            if finite_rank and (
                best_rank_score is None or float(rank_score) > float(best_rank_score)
            ):
                best_index = int(candidate_index)
                best_rank_score = float(rank_score)
        except (ValueError, np.linalg.LinAlgError) as exc:
            scores.append(
                PatchCandidateScore(
                    candidate_kind=PATCH_DELETE,
                    candidate_label=candidate_label,
                    patch=SupportPatch(removed_runtime_indices=()),
                    score=None,
                    rank_score=None,
                    accepted_eligible=False,
                    rejection_reason="candidate_scoring_failed",
                    metadata={
                        "candidate_index": int(candidate_index),
                        "error": str(exc),
                    },
                )
            )

    selected_score = None if best_index is None else scores[int(best_index)]
    reason = "no_finite_prune_score"
    if selected_score is not None and selected_score.score is not None:
        deletion_loss = selected_score.score.deletion_loss
        if deletion_loss is not None and float(deletion_loss) <= float(controller_config.prune_loss_threshold):
            reason = "accepted_best_prune_loss"
        else:
            reason = "prune_loss_above_threshold"
    return PatchBatchEvaluation(
        time=float(time),
        base_runtime_parameter_count=int(state.runtime_parameter_count),
        base_logical_parameter_count=int(state.logical_parameter_count),
        base_residual_ratio=float(base_step.residual_ratio),
        candidate_count=int(len(candidates)),
        scored_count=int(scored_count),
        candidate_scores=tuple(scores),
        selected_index=best_index,
        selected_score=selected_score,
        reason=reason,
        selection_policy=APPEND_BATCH_SELECTION_POLICY_V1,
    )


def _empty_patch_batch(
    *,
    time: float,
    state: APMcLachlanState,
    base_step: FixedMcLachlanStep,
    reason: str,
    selection_policy: str = APPEND_BATCH_SELECTION_POLICY_V1,
    rung_diagnostics: Sequence[RungDiagnostics] = (),
    metadata: Mapping[str, Any] | None = None,
) -> PatchBatchEvaluation:
    return PatchBatchEvaluation(
        time=float(time),
        base_runtime_parameter_count=int(state.runtime_parameter_count),
        base_logical_parameter_count=int(state.logical_parameter_count),
        base_residual_ratio=float(base_step.residual_ratio),
        candidate_count=0,
        scored_count=0,
        candidate_scores=tuple(),
        selected_index=None,
        selected_score=None,
        reason=str(reason),
        selection_policy=str(selection_policy),
        rung_diagnostics=tuple(rung_diagnostics),
        metadata=dict(metadata or {}),
    )


def _materialize_append_candidate(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    candidate: Any,
    inverse_policy: McLachlanInversePolicy,
    solve_repair_config: SolveRepairConfig,
    repair_dt: float | None = None,
) -> tuple[APMcLachlanState, np.ndarray, GeometryEvaluation, FixedMcLachlanStep]:
    appended_state = state_with_appended_terms(
        state,
        (candidate,),
        theta_runtime=theta_runtime,
    )
    theta_aug = np.asarray(appended_state.theta_runtime, dtype=float).reshape(-1)
    evaluation = evaluate_mclachlan_geometry(
        state=appended_state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_aug,
        time=float(time),
    )
    step = _solve_fixed_step_for_trajectory(
        evaluation.geometry,
        inverse_policy=inverse_policy,
        solve_repair_config=solve_repair_config,
        repair_dt=repair_dt,
    )
    return appended_state, theta_aug, evaluation, step


def _materialize_prune_candidate(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    candidate: Mapping[str, Any],
    inverse_policy: McLachlanInversePolicy,
    solve_repair_config: SolveRepairConfig,
    repair_dt: float | None = None,
) -> tuple[APMcLachlanState, np.ndarray, GeometryEvaluation, FixedMcLachlanStep]:
    """Legacy complete-term prune materialization; active route uses support atoms."""

    pruned_state = state_without_term_labels(
        state,
        (str(candidate["label"]),),
        theta_runtime=theta_runtime,
    )
    theta_pruned = np.asarray(pruned_state.theta_runtime, dtype=float).reshape(-1)
    evaluation = evaluate_mclachlan_geometry(
        state=pruned_state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_pruned,
        time=float(time),
    )
    step = _solve_fixed_step_for_trajectory(
        evaluation.geometry,
        inverse_policy=inverse_policy,
        solve_repair_config=solve_repair_config,
        repair_dt=repair_dt,
    )
    return pruned_state, theta_pruned, evaluation, step


def _append_candidates(
    state: APMcLachlanState,
    *,
    max_candidates: int,
    allow_incomplete: bool,
) -> tuple[Any, ...]:
    if int(max_candidates) <= 0:
        return tuple()
    if not bool(allow_incomplete) and not bool(state.can_structural_edit):
        return tuple()
    selected = {str(getattr(term, "label", "")) for term in tuple(state.terms)}
    blocked_parent_labels = (
        no_pauli_split_parent_labels(state)
        if normalize_parameterization_mode(state.parameterization_mode)
        == AP_PARAMETERIZATION_PER_PAULI_TERM
        else set()
    )
    out: list[Any] = []
    for term in tuple(state.candidate_pool_terms):
        label = str(getattr(term, "label", ""))
        if label == "" or label in selected or label in blocked_parent_labels:
            continue
        out.append(term)
        if len(out) >= int(max_candidates):
            break
    return tuple(out)


def _prune_candidates(
    state: APMcLachlanState,
    *,
    max_candidates: int,
    min_logical_parameter_count: int,
) -> tuple[Mapping[str, Any], ...]:
    """Legacy complete-term prune candidates; active route uses support atoms."""

    if int(max_candidates) <= 0:
        return tuple()
    if int(state.logical_parameter_count) <= int(min_logical_parameter_count):
        return tuple()
    out: list[Mapping[str, Any]] = []
    for term_index, term in enumerate(tuple(state.terms)):
        if int(state.logical_parameter_count) - 1 < int(min_logical_parameter_count):
            break
        label = str(getattr(term, "label", ""))
        if label == "":
            continue
        removed = _runtime_indices_for_term_index(state, int(term_index))
        if not removed:
            continue
        out.append(
            {
                "label": label,
                "term_index": int(term_index),
                "removed_runtime_indices": tuple(int(i) for i in removed),
            }
        )
        if len(out) >= int(max_candidates):
            break
    return tuple(out)


def _runtime_indices_for_term_index(state: APMcLachlanState, term_index: int) -> tuple[int, ...]:
    if state.parameterization_mode == "logical_shared":
        return (int(term_index),)
    block = state.layout.blocks[int(term_index)]
    return tuple(range(int(block.runtime_start), int(block.runtime_stop)))


def _time_grid(times: Sequence[float]) -> np.ndarray:
    grid = np.asarray(times, dtype=float).reshape(-1)
    if int(grid.size) == 0:
        raise ValueError("times must contain at least one time point.")
    if not np.all(np.isfinite(grid)):
        raise ValueError("times must contain only finite values.")
    if np.any(np.diff(grid) < 0.0):
        raise ValueError("times must be monotonically nondecreasing.")
    return grid


def _progress_payload_from_point(point: AdaptiveTrajectoryPoint) -> dict[str, Any]:
    decision = point.patch_decision
    selected = decision.selected_score
    batch = decision.batch_evaluation
    selected_candidate = None if batch is None else batch.selected_score
    selected_metadata = (
        dict(selected_candidate.metadata or {}) if selected_candidate is not None else {}
    )
    decision_metadata = dict(decision.metadata or {})
    return {
        "phase": "checkpoint_done",
        "index": int(point.index),
        "time": float(point.time),
        "energy_expectation": float(point.energy_expectation),
        "runtime_parameter_count": int(point.runtime_parameter_count),
        "logical_parameter_count": int(point.logical_parameter_count),
        "mclachlan_residual_ratio": float(point.fixed_step.residual_ratio),
        "theta_dot_l2": float(
            np.linalg.norm(
                np.asarray(point.fixed_step.theta_dot, dtype=float).reshape(-1)
            )
        ),
        "patch_kind": str(decision.patch_kind),
        "patch_accepted": bool(decision.accepted),
        "patch_reason": str(decision.reason),
        "patch_selected_label": decision.selected_label,
        "patch_candidate_count": int(decision.candidate_count),
        "patch_scored_count": int(decision.scored_count),
        "patch_appended_count": (
            0 if selected is None else int(selected.inserted_count)
        ),
        "patch_selected_rung_size": (
            None
            if batch is None or batch.selected_score is None
            else batch.selected_score.metadata.get("rung_size")
        ),
        "patch_prune_smoothness_status": decision_metadata.get(
            "prune_patch_smoothness_status",
            selected_metadata.get("prune_patch_smoothness_status"),
        ),
        "patch_prune_smoothness_eta": decision_metadata.get(
            "prune_patch_smoothness_eta",
            selected_metadata.get("prune_patch_smoothness_eta"),
        ),
        "patch_prune_smoothness_severity": decision_metadata.get(
            "prune_patch_smoothness_severity",
            selected_metadata.get("prune_patch_smoothness_severity"),
        ),
        "patch_prune_smoothness_deferred": decision_metadata.get(
            "prune_patch_smoothness_deferred",
            selected_metadata.get("prune_patch_smoothness_deferred"),
        ),
        "patch_prune_history_transition": decision_metadata.get(
            "prune_history_transition"
        ),
        "patch_prune_atom_history_preserved_count": decision_metadata.get(
            "prune_atom_history_preserved_count"
        ),
        "patch_prune_atom_history_dropped_count": decision_metadata.get(
            "prune_atom_history_dropped_count"
        ),
        "patch_prune_geometry_history_cleared_due_to_support_change": (
            decision_metadata.get(
                "prune_geometry_history_cleared_due_to_support_change"
            )
        ),
        "patch_prune_cooldown_preserved_count": decision_metadata.get(
            "prune_cooldown_preserved_count"
        ),
        "patch_prune_cooldown_dropped_count": decision_metadata.get(
            "prune_cooldown_dropped_count"
        ),
    }


def _mclachlan_path_increment(
    *,
    fixed_step: FixedMcLachlanStep,
    evaluation: GeometryEvaluation,
    dt: float,
) -> float | None:
    theta_dot = np.asarray(fixed_step.theta_dot, dtype=float).reshape(-1)
    K = np.asarray(evaluation.geometry.K, dtype=float)
    if K.shape != (int(theta_dot.size), int(theta_dot.size)):
        return None
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(theta_dot)):
        return None
    quadratic = float(theta_dot @ (K @ theta_dot))
    if not np.isfinite(quadratic):
        return None
    return float(abs(float(dt)) * math.sqrt(max(0.0, quadratic)))


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    return out if np.isfinite(out) else None


__all__ = [
    "ADAPTIVE_TRAJECTORY_SCHEMA_V1",
    "APPEND_MACRO_SCOUT_POLICY_V2",
    "APPEND_MACRO_SCOUT_SCORE_MODE_FULL_CHILD_BLOCK_DIAGNOSTIC",
    "APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_LINEAR_RESIDUAL_V1",
    "APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_TANGENT_SCHUR_GAIN",
    "APPEND_MACRO_SCOUT_SCORE_MODES",
    "APPEND_BATCH_SELECTION_POLICY_V1",
    "FAILED_APPEND_REOPEN_DIRECT",
    "FAILED_APPEND_REOPEN_MODEL_CHANGE",
    "FAILED_APPEND_REUSE_POLICY_V1",
    "APPEND_LADDER_PREFILTER_POLICY_V1",
    "APPEND_LADDER_SELECTION_POLICY_V1",
    "AppendControllerConfig",
    "AppendMclachlanTrajectory",
    "AdaptiveTrajectoryPoint",
    "LEGACY_APPEND_CONTROLLER_PROFILE_V1",
    "PatchBatchEvaluation",
    "PatchCandidateScore",
    "PatchDecision",
    "SUPPORT_PATCH_CONTROLLER_PROFILE_V1",
    "SolveRepairConfig",
    "SupportPatchControllerConfig",
    "run_append_mclachlan_trajectory",
]
