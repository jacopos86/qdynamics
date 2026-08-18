"""Append-first AP-McLachlan trajectory propagation."""

from __future__ import annotations

import itertools
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.append_cost import (
    AP_APPEND_COST_MODEL_PAPER_I_PROXY_V1,
    AppendCostSettings,
)
from pipelines.time_dynamics.ap_mclachlan.prune_cost import (
    PruneCostSettings,
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
)
from pipelines.time_dynamics.ap_mclachlan.geometry import (
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
)
from pipelines.time_dynamics.ap_mclachlan.exchange_integration import (
    select_deletion_conditioned_patch,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_selector import (
    EXCHANGE_SELECTION_POLICY_V1,
    ExchangeSelection,
)
from pipelines.time_dynamics.ap_mclachlan.performance import (
    NULL_PHASE as _NO_PHASE,
    PHASE_FIXED_STEP_SOLVE,
    PHASE_GEOMETRY_EVAL,
    PHASE_INTEGRATE,
    PHASE_UNIFIED_SELECT,
    active_profiler as _active_profiler,
    attribute_nested as _attribute_nested,
    count as _profile_count,
    phase,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    APMcLachlanState,
)
from pipelines.time_dynamics.ap_mclachlan.support_atoms import (
    APPEND_OCCURRENCE_POLICY_LAYER_REUSE,
    APPEND_OCCURRENCE_POLICY_UNIQUE_SUPPORT,
    ActiveSupportAtom,
    SupportAtom,
    active_support_atoms,
    appended_origin_atom_labels,
    normalize_append_occurrence_policy,
)
from pipelines.time_dynamics.ap_mclachlan.support_frontier import (
    APPEND_MACRO_SCOUT_POLICY_V2,
    APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_TANGENT_SCHUR_GAIN,
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
    SupportPatchScore,
)


ADAPTIVE_TRAJECTORY_SCHEMA_V1 = "ap_mclachlan_append_trajectory_v1"
APPEND_BATCH_SELECTION_POLICY_V1 = "max_rank_score_pool_order_tiebreak_v1"
APPEND_LADDER_SELECTION_POLICY_V1 = "cost_weighted_combinatorial_append_ladder_v1"
APPEND_LADDER_PREFILTER_POLICY_V1 = "cost_weighted_singleton_rank_score_prefilter_v1"
PRUNE_LADDER_SELECTION_POLICY_V1 = "cost_pressure_combinatorial_prune_ladder_v1"
PRUNE_LADDER_PREFILTER_POLICY_V1 = "cost_pressure_singleton_prune_prefilter_v1"
SUPPORT_PATCH_CONTROLLER_PROFILE_V1 = "support_patch_exchange_family_v1"
LEGACY_APPEND_CONTROLLER_PROFILE_V1 = "legacy_append_compat_v1"
SUPPORT_PATCH_EXCHANGE_SELECTION_POLICY_V1 = "paper_ii_unified_support_patch_exchange_v1"
LEGACY_APPEND_PATCH_KINDS = frozenset({PATCH_APPEND, PATCH_INSERT})
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
    residual_ratio_threshold: float = DEFAULT_APPEND_RESIDUAL_RATIO_THRESHOLD
    # Deletion-conditioned exchange selector (paper_ii_deletion_conditioned_exchange_v1)
    interaction_frontier_widths: tuple[int, ...] | None = None
    max_insertion_batch_size: int | None = None
    structural_score_floor: float = 0.0
    max_joint_patch_evaluations: int | None = None
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
    certification_refit_enabled: bool = False
    certification_refit_trust_radius: float = 0.1
    certification_refit_max_iterations: int = 15
    max_certification_attempts_per_level: int | None = None
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
            "residual_ratio_threshold": float(self.residual_ratio_threshold),
            "interaction_frontier_widths": (
                None
                if self.interaction_frontier_widths is None
                else [int(w) for w in self.interaction_frontier_widths]
            ),
            "max_insertion_batch_size": (
                None
                if self.max_insertion_batch_size is None
                else int(self.max_insertion_batch_size)
            ),
            "structural_score_floor": float(self.structural_score_floor),
            "max_joint_patch_evaluations": (
                None
                if self.max_joint_patch_evaluations is None
                else int(self.max_joint_patch_evaluations)
            ),
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
            "certification_refit_enabled": bool(self.certification_refit_enabled),
            "certification_refit_trust_radius": float(
                self.certification_refit_trust_radius
            ),
            "certification_refit_max_iterations": int(
                self.certification_refit_max_iterations
            ),
            "max_certification_attempts_per_level": (
                None
                if self.max_certification_attempts_per_level is None
                else int(self.max_certification_attempts_per_level)
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


_EXCHANGE_KIND_TO_PATCH = {
    "insert": PATCH_APPEND,
    "delete": PATCH_DELETE,
    "exchange": PATCH_EXCHANGE,
    "stay": PATCH_NO_EDIT,
}


def _decision_from_exchange_selection(
    selection: ExchangeSelection,
    payload: Mapping[str, Any],
) -> tuple[
    PatchDecision,
    APMcLachlanState | None,
    np.ndarray | None,
    GeometryEvaluation | None,
    FixedMcLachlanStep | None,
]:
    """Map an exchange selection onto the trajectory's PatchDecision contract."""

    scored_count = int(
        (payload.get("work_guard") or {}).get("scored_count", len(selection.attempts))
    )
    metadata = dict(payload)
    if selection.committed is None or selection.certification is None:
        return (
            PatchDecision(
                patch_kind=PATCH_NO_EDIT,
                accepted=False,
                candidate_count=scored_count,
                scored_count=scored_count,
                reason=str(selection.stop_reason),
                metadata=metadata,
            ),
            None,
            None,
            None,
            None,
        )
    committed = selection.committed
    certification = selection.certification
    return (
        PatchDecision(
            patch_kind=_EXCHANGE_KIND_TO_PATCH[str(selection.kind)],
            accepted=True,
            candidate_count=scored_count,
            scored_count=scored_count,
            selected_label=(
                ",".join(a for a, _p in committed.inserted_selection) or None
            ),
            reason=f"accepted_deletion_conditioned_{selection.kind}",
            metadata=metadata,
        ),
        certification.state,
        np.asarray(certification.theta, dtype=float).reshape(-1),
        certification.evaluation,
        certification.step,
    )


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
    prune_runtime_state = _PruneControllerRuntimeState()
    # The deletion-conditioned exchange selector is the single current route;
    # a missing support-patch config resolves to its typed defaults.
    effective_support_config = (
        support_patch_config
        if support_patch_config is not None
        else SupportPatchControllerConfig(append_ladder_mode="combinatorial")
    )

    for index, time_value in enumerate(time_grid):
        dt_to_next = (
            None if index + 1 >= len(time_grid) else float(time_grid[index + 1] - time_value)
        )
        _profile_count("checkpoints")
        with phase(PHASE_GEOMETRY_EVAL):
            evaluation = evaluate_mclachlan_geometry(
                state=current_state,
                hamiltonian=hamiltonian,
                theta_runtime=theta_current,
                time=float(time_value),
                # The exchange selector's structural cache and certification
                # smoothness gate always consume the frozen tangent matrix.
                include_tangent_matrix=True,
            )
        kink_reference_theta_dot = _same_dimension_theta_dot_or_none(
            previous_accepted_theta_dot,
            int(evaluation.geometry.dimension),
        )
        with phase(PHASE_FIXED_STEP_SOLVE):
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
            elif float(fixed_step.residual_ratio) < float(
                effective_support_config.residual_ratio_threshold
            ):
                # Structural-repair predicate is inactive: the realized miss is
                # already below threshold, so no structural family is acquired
                # and the checkpoint pays no candidate solves.
                decision = PatchDecision(
                    patch_kind=PATCH_NO_EDIT,
                    accepted=False,
                    candidate_count=0,
                    scored_count=0,
                    reason="residual_below_threshold",
                )
            else:
                with phase(PHASE_UNIFIED_SELECT):
                    selection, selection_payload = select_deletion_conditioned_patch(
                        state=current_state,
                        hamiltonian=hamiltonian,
                        theta_runtime=theta_current,
                        time=float(time_value),
                        base_evaluation=evaluation,
                        base_step=fixed_step,
                        inverse_policy=decision_inverse_policy,
                        support_config=effective_support_config,
                        runtime_state=prune_runtime_state,
                        time_index=int(index),
                        active_prune_atoms=_active_prune_atoms,
                        solve_repair_config=solve_repair_config,
                    )
                (
                    decision,
                    maybe_state,
                    maybe_theta,
                    maybe_eval,
                    maybe_step,
                ) = _decision_from_exchange_selection(selection, selection_payload)
            if decision.accepted and maybe_state is not None and maybe_theta is not None and maybe_eval is not None and maybe_step is not None:
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

        integration: IntegrationStep | None = None
        if index + 1 < len(time_grid):
            dt = float(time_grid[index + 1] - time_value)
            state_for_rhs = current_state
            force_local_subdivision_request = _checkpoint_local_subdivision_request(
                fixed_step,
                solve_repair_config=solve_repair_config,
            )
            with phase(PHASE_INTEGRATE):
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
            "trajectory_kind": "deletion_conditioned_exchange_support_patch",
            "selection_policy": EXCHANGE_SELECTION_POLICY_V1,
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
    batch_phase: str | None = None,
    task_phase: str | None = None,
) -> tuple[Any, ...]:
    task_tuple = tuple(tasks)
    worker_count = _support_patch_scoring_worker_count(
        support_config,
        task_count=len(task_tuple),
    )
    profiling = _active_profiler() is not None
    if not profiling:
        # Ordinary runs pay nothing for the instrumentation.
        if worker_count <= 1:
            return tuple(score_one(task) for task in task_tuple)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return tuple(executor.map(score_one, task_tuple))

    if batch_phase is not None:
        _profile_count(f"{batch_phase}.tasks", len(task_tuple))
        _profile_count(f"{batch_phase}.worker_count_{int(worker_count)}")

    # Worker threads keep their own nested-phase stacks, so a task timed on a
    # worker is not subtracted from the dispatching phase.  Sum the task spans
    # and hand the total back to the parent after fan-in, otherwise the batch
    # phase reports the whole parallel section as its own exclusive time.
    task_seconds: list[float] = []
    inner = score_one

    def timed_task(task: Any) -> Any:
        start = time.perf_counter()
        try:
            if task_phase is None:
                return inner(task)
            with phase(task_phase):
                return inner(task)
        finally:
            task_seconds.append(time.perf_counter() - start)

    def phased_task(task: Any) -> Any:
        if task_phase is None:
            return inner(task)
        with phase(task_phase):
            return inner(task)

    with phase(batch_phase) if batch_phase is not None else _NO_PHASE:
        if worker_count <= 1:
            # Same thread: nested accounting already applies, so the parent must
            # not be charged a second time.
            return tuple(phased_task(task) for task in task_tuple)
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                return tuple(executor.map(timed_task, task_tuple))
        finally:
            _attribute_nested(sum(task_seconds))
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


@lru_cache(maxsize=512)
def _support_identity_hash_for(
    parameterization_mode: str,
    runtime_coordinate_labels: tuple[str, ...],
) -> str:
    return _stable_json_hash(
        {
            "parameterization_mode": str(parameterization_mode),
            "runtime_coordinate_labels": [
                str(label) for label in runtime_coordinate_labels
            ],
        }
    )


def _support_identity_hash(state: APMcLachlanState) -> str:
    """Stable identity of the ordered runtime support.

    The support changes only when a patch is accepted, but this is consulted
    every checkpoint, so the JSON encode plus SHA-256 over every coordinate
    label was repeated for each unchanged support.  Memoizing on the identity
    inputs returns the same digest without recomputing it; tuple hashing reuses
    the interned label hashes.
    """

    return _support_identity_hash_for(
        str(state.parameterization_mode),
        tuple(str(label) for label in state.runtime_coordinate_labels),
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


def _prune_persistence_atom_id(atom: ActiveSupportAtom) -> str:
    metadata = dict(atom.metadata or {})
    base_atom_id = str(metadata.get("base_atom_id", "")).strip()
    return base_atom_id if base_atom_id else str(atom.atom_id)


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
