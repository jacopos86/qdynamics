#!/usr/bin/env python3
"""Scoring and proxy accounting for HH continuation."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import cmp_to_key
import hashlib
import itertools
import json
import math
import numbers
import os
import threading
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from pipelines.static_adapt.plateau_acquisition import (
    PLATEAU_ACQUISITION_SCORE_FRACTIONAL_RESIDUAL_V1,
    PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1,
    normalize_plateau_acquisition_score,
    plateau_score_formula,
)
from pipelines.static_adapt.paper_i_config import PAPER_I_CANONICAL_COST_WEIGHTS
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_BLOCK_PINV_LEGACY_V1,
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
    JointLinearSolveConfig,
    solve_joint_linear_model,
)
from pipelines.static_adapt.phase3_material_window import (
    Phase3MaterialWindowPolicy,
    build_phase3_material_window,
)
from pipelines.static_adapt.sr_snake_phase12_policy import (
    PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
    PHASE1_ENERGY_MODEL_LEGACY_LAMBDA_F_QUADRATIC_V1,
    PHASE2_CHEAP_CURVATURE_PROXY_POLICY_LEGACY_LAMBDA_F_RATIO_V1,
    PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF,
    PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1,
    PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1,
    PHASE2_DIRECTIONAL_CURVATURE_PROVENANCE_SCHEMA,
    PHASE2_DIRECTIONAL_CURVATURE_RECEIPT_SCHEMA,
    normalize_phase1_energy_model,
    normalize_phase2_cheap_curvature_proxy_policy,
    normalize_phase2_curvature_policy,
)
from pipelines.static_adapt.sr_snake_escape_controller import (
    NonstationaryCertificate,
    OrdinaryCertificate,
    PsdCertificate,
    QuotientRedundantCertificate,
    ReachablePopulationAudit,
    SRControllerDecisionKind,
    SREscapeMode,
    SR_ESCAPE_DISABLED,
    SaddleCertificate,
    StateStationarityCertificate,
    UnresolvedCertificate,
    reachable_population_digest,
    select_sr_escape_path,
    sr_escape_record_id,
)
from pipelines.static_adapt.sr_snake_modeled_minimum import (
    assess_exposed_family_psd,
)
from pipelines.scaffold.hh_continuation_types import (
    CandidateFeatures,
    CompileCostEstimate,
    CurvatureOracle,
    MeasurementCacheStats,
    MeasurementGroupSpec,
    MeasurementPlan,
    OrderedInsertionGeometryOracleProtocol,
)
from pipelines.scaffold.hh_continuation_motifs import motif_bonus_for_generator
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    apply_compiled_polynomial,
    compile_polynomial_action,
)
from src.quantum.pauli_actions import apply_compiled_pauli


CANONICAL_HARDWARE_COST_LAMBDA_2Q = PAPER_I_CANONICAL_COST_WEIGHTS.lambda_2q
CANONICAL_HARDWARE_COST_LAMBDA_D = PAPER_I_CANONICAL_COST_WEIGHTS.lambda_d
CANONICAL_HARDWARE_COST_LAMBDA_1Q = PAPER_I_CANONICAL_COST_WEIGHTS.lambda_1q
CANONICAL_HARDWARE_COST_LAMBDA_THETA = PAPER_I_CANONICAL_COST_WEIGHTS.lambda_theta
CANONICAL_HARDWARE_COST_LAMBDA_SHOT = PAPER_I_CANONICAL_COST_WEIGHTS.lambda_shot

_SR_MODELED_MINIMUM_CORE_TELEMETRY_SCHEMA = (
    "sr_snake_modeled_minimum_core_telemetry_v1"
)
_SR_MODELED_MINIMUM_EXECUTION_BLOCKERS = (
    "canonical_continuation_path_provider_missing",
    "live_nonlinear_active_manifold_distance_provider_missing",
    "uniform_full_path_incumbent_barrier_provider_missing",
    "connected_exclusion_component_witness_provider_missing",
    "disposable_powell_reproducibility_provider_missing",
    "countable_action_cursor_tail_bound_missing",
    "incumbent_working_state_runtime_integration_missing",
    "modeled_minimum_checkpoint_roundtrip_missing",
)

@dataclass(frozen=True)
class SimpleScoreConfig:
    lambda_F: float = 1.0
    lambda_compile: float = 0.05
    lambda_measure: float = 0.02
    lambda_leak: float = 0.0
    z_alpha: float = 0.0
    rho: float = 0.25
    metric_floor: float = 1e-12
    hardware_resolution_mode: str = "ideal"
    manual_b_g_hw: float = 0.0
    manual_b_g_drift: float = 0.0
    wD: float = 0.0
    wG: float = 0.0
    wC: float = 0.0
    wc: float = 0.0
    lambda_2q: float | None = CANONICAL_HARDWARE_COST_LAMBDA_2Q
    lambda_d: float | None = CANONICAL_HARDWARE_COST_LAMBDA_D
    lambda_1q: float | None = CANONICAL_HARDWARE_COST_LAMBDA_1Q
    lambda_theta: float | None = CANONICAL_HARDWARE_COST_LAMBDA_THETA
    lambda_shot: float | None = CANONICAL_HARDWARE_COST_LAMBDA_SHOT
    shot_sigma_star: float = 1.0
    hardware_cost_scale_floor: float = 1e-12
    hardware_cost_normalization_mode: str = "family_robust_v1"
    compile_cx_proxy_weight: float = 1.0
    compile_sq_proxy_weight: float = 0.5
    compile_rotation_step_weight: float = 1.0
    compile_position_shift_weight: float = 1.0
    compile_refit_active_weight: float = 1.0
    measure_groups_weight: float = 1.0
    measure_shots_weight: float = 1.0
    measure_reuse_weight: float = 1.0
    opt_dim_cost_scale: float = 1.0
    family_repeat_cost_scale: float = 1.0
    depth_ref: float = 1.0
    group_ref: float = 1.0
    shot_ref: float = 1.0
    family_ref: float = 1.0
    lifetime_cost_mode: str = "off"
    burden_floor: float = 0.25
    phase1_score_mode: str = "trust_region_v1"
    phase1_energy_model: str = PHASE1_ENERGY_MODEL_LEGACY_LAMBDA_F_QUADRATIC_V1
    resource_weighting_scope: str = "all_phase_resource_weighting_v1"
    score_version: str = "simple_v1"


PHASE3_CANDIDATE_GAIN_JOINT_TOTAL_V1 = "joint_total_gain_v1"
PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1 = (
    "joint_minus_active_only_supported_trust_v1"
)
PHASE3_CANDIDATE_GAIN_POLICIES = frozenset(
    {
        PHASE3_CANDIDATE_GAIN_JOINT_TOTAL_V1,
        PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1,
    }
)
PHASE3_ZERO_CENTERED_SIGNED_FACTOR_CONSUMER_SEMANTIC_VERSION = (
    "paper_i_ra_phase0_proxy_ablation_phase123_qiskit_semantic_closure_v1"
)


@dataclass(frozen=True)
class FullScoreConfig:
    z_alpha: float = 0.0
    lambda_F: float = 1.0
    lambda_H: float = 1e-6
    rho: float = 0.25
    eta_L: float = 0.0
    wD: float = 0.2
    wG: float = 0.15
    wC: float = 0.15
    wP: float = 0.1
    wc: float = 0.1
    lambda_2q: float | None = CANONICAL_HARDWARE_COST_LAMBDA_2Q
    lambda_d: float | None = CANONICAL_HARDWARE_COST_LAMBDA_D
    lambda_1q: float | None = CANONICAL_HARDWARE_COST_LAMBDA_1Q
    lambda_theta: float | None = CANONICAL_HARDWARE_COST_LAMBDA_THETA
    lambda_shot: float | None = CANONICAL_HARDWARE_COST_LAMBDA_SHOT
    resource_weighting_scope: str = "all_phase_resource_weighting_v1"
    active_gradient_policy: str = "measured_residual_response_v1"
    phase3_candidate_gain_policy: str = (
        PHASE3_CANDIDATE_GAIN_JOINT_TOTAL_V1
    )
    shot_sigma_star: float = 1.0
    hardware_cost_scale_floor: float = 1e-12
    hardware_cost_normalization_mode: str = "family_robust_v1"
    phase3_signed_factor_consumer_semantic_version: str | None = None
    compile_cx_proxy_weight: float = 1.0
    compile_sq_proxy_weight: float = 0.5
    compile_rotation_step_weight: float = 1.0
    compile_position_shift_weight: float = 1.0
    compile_refit_active_weight: float = 1.0
    measure_groups_weight: float = 1.0
    measure_shots_weight: float = 1.0
    measure_reuse_weight: float = 1.0
    opt_dim_cost_scale: float = 1.0
    family_repeat_cost_scale: float = 1.0
    depth_ref: float = 1.0
    group_ref: float = 1.0
    shot_ref: float = 1.0
    optdim_ref: float = 1.0
    reuse_ref: float = 1.0
    family_ref: float = 1.0
    deferred_gram_fallback_enabled: bool = False
    deferred_gram_fallback_ridge: float = 1e-6
    cheap_score_eps: float = 1e-12
    phase2_curvature_policy: str = PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1
    phase2_cheap_curvature_proxy_policy: str = (
        PHASE2_CHEAP_CURVATURE_PROXY_POLICY_LEGACY_LAMBDA_F_RATIO_V1
    )
    shortlist_fraction: float = 0.2
    shortlist_size: int = 12
    phase2_frontier_ratio: float = 0.9
    phase3_frontier_ratio: float = 0.9
    batch_target_size: int = 2
    batch_size_cap: int = 3
    batch_near_degenerate_ratio: float = 0.9
    batch_rank_rel_tol: float = 1e-6
    batch_max_gram_condition_number: float = 1e12
    batch_additivity_tol: float = 0.25
    batch_search_pool_size: int | None = None
    batch_search_population_mode: str = "near_degenerate_shell_legacy_v1"
    batch_search_feasibility_policy: str = "raw_ranked_legacy_v1"
    batch_additivity_policy: str = "hard_gate_legacy_v1"
    batch_additivity_lambda: float = 0.0
    batch_score_tie_tolerance: float = 1e-12
    batch_geometry_mode: str = "per_subset_diagonal_hessian_legacy_v1"
    batch_metric_regularization: float = 1e-9
    batch_energy_regularization: float = 1e-9
    batch_joint_linear_solve_policy: str = (
        JOINT_LINEAR_SOLVE_BLOCK_PINV_LEGACY_V1
    )
    batch_state_consistency_tolerance: float = 1e-8
    batch_joint_context_mode: str = "batch_only_diagnostic_v1"
    batch_active_context_indices: tuple[int, ...] | None = None
    batch_selection_mode: str = "reduced_plane"
    duplicate_penalty_weight: float = 0.0
    compat_overlap_weight: float = 0.4
    compat_comm_weight: float = 0.2
    compat_curv_weight: float = 0.2
    compat_sched_weight: float = 0.2
    compat_measure_weight: float = 0.2
    leakage_cap: float = 1e6
    lifetime_cost_mode: str = "off"
    remaining_evaluations_proxy_mode: str = "none"
    lifetime_weight: float = 0.05
    motif_bonus_weight: float = 0.05
    metric_floor: float = 1e-12
    reduced_metric_collapse_rel_tol: float = 1e-8
    ridge_growth_factor: float = 10.0
    ridge_max_steps: int = 12
    phase3_selector_geometry_mode: str = "reduced"
    phase3_window_relaxation_mode: str = "reduced"
    auxiliary_score_mode: str = "tie_break_only"
    burden_floor: float = 0.25
    score_version: str = "full_v2"
    phase2_selector_gain_mode: str = "trust_region_v1"
    hardware_resolution_mode: str = "ideal"
    manual_b_g_hw: float = 0.0
    manual_b_g_drift: float = 0.0


PHASE2_SELECTOR_GAIN_TRUST_REGION_V1 = "trust_region_v1"
PHASE2_SELECTOR_GAIN_UNIT_GAIN_V1 = "unit_gain_v1"
PHASE2_CANONICAL_RAW_SCORE_FORMULA = "DeltaE_TR_raw / (1 + K2)"
PHASE2_UNIT_GAIN_RAW_SCORE_FORMULA = "1 / (1 + K2)"
PHASE2_NO_NOVELTY_RAW_SCORE_FORMULA = PHASE2_CANONICAL_RAW_SCORE_FORMULA
PHASE2_NO_NOVELTY_UNIT_GAIN_RAW_SCORE_FORMULA = (
    PHASE2_UNIT_GAIN_RAW_SCORE_FORMULA
)
PHASE3_CANONICAL_SCORE_FORMULA = "DeltaE_TR / (1 + K3)"
PHASE3_NO_NOVELTY_SCORE_FORMULA = PHASE3_CANONICAL_SCORE_FORMULA
GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1 = "fallback_only_v1"
ORDINARY_NOVELTY_SCORING_RETIRED_V1 = "ordinary_novelty_scoring_retired_v1"
GRAM_NOVELTY_STATUS_NOT_COMPUTED_FOR_ORDINARY_SCORING = (
    "not_computed_for_ordinary_scoring"
)
GRAM_NOVELTY_STATUS_COMPUTED_FOR_GEOMETRY_EXPANSION = (
    "computed_for_geometry_expansion_fallback"
)
PHASE3_AUXILIARY_SCORE_TIE_BREAK_ONLY = "tie_break_only"
PHASE3_AUXILIARY_SCORE_ABLATION_ADDITIVE = "ablation_additive"
PHASE1_SCORE_MODE_TRUST_REGION_V1 = "trust_region_v1"
PHASE1_SCORE_MODE_LEGACY_SIMPLE_V1 = "legacy_simple_v1"
OVERLAP_ORTHOGONAL_BENCHMARK_MAX = 0.15

HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1 = "family_robust_v1"
HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1 = (
    "family_robust_symmetric_arctan_v1"
)
HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1 = (
    "zero_centered_signed_arctan_v1"
)
HARDWARE_COST_MULTIPLICATIVE_SIGNED_FACTOR_POLICIES = frozenset(
    {
        HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1,
        HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1,
    }
)
HARDWARE_COST_NORMALIZATION_RAW_LEGACY_V1 = "raw_legacy_v1"
HARDWARE_COST_NORMALIZATION_SCHEMA = "snake_hardware_cost_family_robust_v1"
HARDWARE_COST_SYMMETRIC_ARCTAN_SCHEMA = (
    "snake_hardware_cost_family_robust_symmetric_arctan_v1"
)
HARDWARE_COST_RAW_LEGACY_SCHEMA = "snake_hardware_cost_raw_legacy_v1"
_HARDWARE_COST_COMPONENTS = ("2q", "d", "1q", "theta", "shot")


def _finite_nonnegative(value: Any, default: float = 0.0) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value_f):
        return float(default)
    return float(max(0.0, value_f))


def normalize_phase1_score_mode(raw_mode: Any) -> str:
    """Normalize Phase-I score mode aliases."""
    mode = (
        PHASE1_SCORE_MODE_TRUST_REGION_V1
        if raw_mode is None
        else str(raw_mode).strip().lower()
    )
    aliases = {
        "": PHASE1_SCORE_MODE_TRUST_REGION_V1,
        "trust_region": PHASE1_SCORE_MODE_TRUST_REGION_V1,
        "trust-region": PHASE1_SCORE_MODE_TRUST_REGION_V1,
        "tr": PHASE1_SCORE_MODE_TRUST_REGION_V1,
        "rho": PHASE1_SCORE_MODE_TRUST_REGION_V1,
        "legacy": PHASE1_SCORE_MODE_LEGACY_SIMPLE_V1,
        "simple": PHASE1_SCORE_MODE_LEGACY_SIMPLE_V1,
        "simple_v1": PHASE1_SCORE_MODE_LEGACY_SIMPLE_V1,
    }
    mode = aliases.get(mode, mode)
    if mode not in {PHASE1_SCORE_MODE_TRUST_REGION_V1, PHASE1_SCORE_MODE_LEGACY_SIMPLE_V1}:
        raise ValueError(
            "phase1_score_mode must be one of "
            f"{PHASE1_SCORE_MODE_TRUST_REGION_V1!r} or "
            f"{PHASE1_SCORE_MODE_LEGACY_SIMPLE_V1!r}."
        )
    return str(mode)


def _optional_nonnegative(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return float(max(0.0, value_f))


def _hardware_cost_scale_floor(cfg: Any) -> float:
    return float(max(_finite_nonnegative(getattr(cfg, "hardware_cost_scale_floor", 1e-12), 1e-12), 1e-12))


def resolve_hardware_cost_lambdas(cfg: Any) -> tuple[dict[str, float], str]:
    """Resolve manuscript hardware-cost lambdas with legacy aliases.

    Explicit ``lambda_*`` fields win component-by-component.  Missing explicit
    fields fall back to the compatibility mapping in the implementation plan.
    """
    explicit = {
        "2q": _optional_nonnegative(getattr(cfg, "lambda_2q", None)),
        "d": _optional_nonnegative(getattr(cfg, "lambda_d", None)),
        "1q": _optional_nonnegative(getattr(cfg, "lambda_1q", None)),
        "theta": _optional_nonnegative(getattr(cfg, "lambda_theta", None)),
        "shot": _optional_nonnegative(getattr(cfg, "lambda_shot", None)),
    }
    legacy_wD = _finite_nonnegative(getattr(cfg, "wD", 0.0))
    legacy_wP = _finite_nonnegative(getattr(cfg, "wP", 0.0))
    legacy_wC = _finite_nonnegative(getattr(cfg, "wC", 0.0))
    legacy_wG = _finite_nonnegative(getattr(cfg, "wG", 0.0))
    legacy_wc = _finite_nonnegative(getattr(cfg, "wc", 0.0))
    compile_pressure = _finite_nonnegative(getattr(cfg, "lambda_compile", legacy_wD))
    measure_pressure = _finite_nonnegative(getattr(cfg, "lambda_measure", max(legacy_wC, legacy_wG, legacy_wc)))
    cx_weight = _finite_nonnegative(getattr(cfg, "compile_cx_proxy_weight", 1.0), 1.0)
    sq_weight = _finite_nonnegative(getattr(cfg, "compile_sq_proxy_weight", 0.5), 0.5)
    if isinstance(cfg, SimpleScoreConfig):
        legacy = {
            "2q": float(compile_pressure * cx_weight),
            "d": float(compile_pressure),
            "1q": float(compile_pressure * sq_weight),
            "theta": 0.0,
            "shot": float(measure_pressure),
        }
    else:
        legacy = {
            "2q": float(legacy_wD * cx_weight),
            "d": float(legacy_wD),
            "1q": float(legacy_wD * sq_weight),
            "theta": float(legacy_wP),
            "shot": float(max(legacy_wC, legacy_wG, legacy_wc)),
        }
    resolved = {
        component: float(explicit[component] if explicit[component] is not None else legacy[component])
        for component in _HARDWARE_COST_COMPONENTS
    }
    source = (
        "explicit_lambda_fields_v1"
        if all(explicit[component] is not None for component in _HARDWARE_COST_COMPONENTS)
        else "mixed_explicit_lambda_legacy_alias_v1"
        if any(explicit[component] is not None for component in _HARDWARE_COST_COMPONENTS)
        else "legacy_alias_mapping_v1"
    )
    return resolved, source


def _strict_nonnegative_hardware_cost(value: Any, *, field_name: str) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"symmetric hardware cost {field_name} must be numeric."
        ) from exc
    if not math.isfinite(value_f) or value_f < 0.0:
        raise ValueError(
            f"symmetric hardware cost {field_name} must be finite and nonnegative."
        )
    return float(value_f)


def _hardware_cost_raw_components(
    feat: CandidateFeatures,
    *,
    strict: bool = False,
) -> dict[str, float]:
    raw_values = {
        "2q": getattr(feat, "c_hat_2q", 0.0),
        "d": getattr(feat, "c_hat_d", 0.0),
        "1q": getattr(feat, "c_hat_1q", 0.0),
        "theta": getattr(feat, "c_hat_theta", 0.0),
        "shot": getattr(feat, "c_hat_shot", 0.0),
    }
    if strict:
        return {
            key: _strict_nonnegative_hardware_cost(
                raw_values[key], field_name=f"c_hat_{key}"
            )
            for key in _HARDWARE_COST_COMPONENTS
        }
    return {
        key: _finite_nonnegative(raw_values[key])
        for key in _HARDWARE_COST_COMPONENTS
    }


def _hardware_cost_bar_components(feat: CandidateFeatures) -> dict[str, float]:
    return {
        "2q": _finite_nonnegative(getattr(feat, "c_bar_2q", 0.0)),
        "d": _finite_nonnegative(getattr(feat, "c_bar_d", 0.0)),
        "1q": _finite_nonnegative(getattr(feat, "c_bar_1q", 0.0)),
        "theta": _finite_nonnegative(getattr(feat, "c_bar_theta", 0.0)),
        "shot": _finite_nonnegative(getattr(feat, "c_bar_shot", 0.0)),
    }


def _signed_compiled_marginal_components(
    feat: CandidateFeatures,
) -> dict[str, float]:
    backend = feat.compiled_position_cost_backend
    if (
        not isinstance(backend, Mapping)
        or backend.get("negative_delta_reward_enabled") is not True
    ):
        raise ValueError(
            "signed Qiskit cost scoring requires authenticated raw marginal "
            "telemetry with negative-delta rewards enabled."
        )
    fields = {
        "2q": "raw_delta_compiled_count_2q",
        "d": "raw_delta_compiled_depth_2q",
        "1q": "raw_delta_compiled_count_1q",
    }
    result: dict[str, float] = {}
    for key, field_name in fields.items():
        try:
            value = float(backend[field_name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "signed Qiskit cost scoring lacks a raw compiled marginal."
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                "signed Qiskit compiled marginals must be finite."
            )
        result[key] = value
    result["theta"] = 0.0
    result["shot"] = 0.0
    return result


def _zero_centered_signed_population_identity(
    feat: CandidateFeatures,
) -> dict[str, str]:
    generator_id = feat.generator_id
    if not isinstance(generator_id, str) or not generator_id.strip():
        raise ValueError(
            "zero-centered signed Qiskit normalization requires a nonempty "
            "generator_id."
        )
    backend = feat.compiled_position_cost_backend
    if not isinstance(backend, Mapping):
        raise ValueError(
            "zero-centered signed Qiskit normalization requires compiled "
            "base/trial ansatz identities."
        )
    structure_keys: dict[str, str] = {}
    for field_name in ("base_structure_key", "trial_structure_key"):
        value = backend.get(field_name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                "zero-centered signed Qiskit normalization requires exact "
                f"64-character lowercase SHA-256 {field_name}."
            )
        structure_keys[field_name] = value
    if (
        structure_keys["base_structure_key"]
        == structure_keys["trial_structure_key"]
    ):
        raise ValueError(
            "zero-centered signed Qiskit normalization requires distinct "
            "base/trial ansatz identities."
        )
    return {
        "generator_id": generator_id,
        **structure_keys,
    }


def _hardware_cost_denominator_payload(feat: CandidateFeatures, cfg: Any) -> dict[str, Any]:
    lambdas, lambda_source = resolve_hardware_cost_lambdas(cfg)
    mode = _hardware_cost_normalization_mode(cfg)
    if (
        mode
        == HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
        and str(getattr(feat, "hardware_cost_policy", ""))
        == HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1
    ):
        return _hardware_cost_denominator_payload(
            feat,
            replace(
                cfg,
                hardware_cost_normalization_mode=(
                    HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1
                ),
            ),
        )
    if mode in HARDWARE_COST_MULTIPLICATIVE_SIGNED_FACTOR_POLICIES:
        # Validate raw costs even when this is the neutral pre-family feature.
        # The family rescore later supplies the population-relative factor.
        _hardware_cost_raw_components(feat, strict=True)
        feature_policy = str(
            getattr(feat, "hardware_cost_policy", "unresolved") or "unresolved"
        )
        if feature_policy not in {
            "unresolved",
            *HARDWARE_COST_MULTIPLICATIVE_SIGNED_FACTOR_POLICIES,
        }:
            raise ValueError(
                "symmetric hardware-cost scoring received a feature normalized "
                f"under incompatible policy {feature_policy!r}."
            )
        if feature_policy in HARDWARE_COST_MULTIPLICATIVE_SIGNED_FACTOR_POLICIES:
            signed_raw = getattr(feat, "hardware_cost_signed_components", {})
            if not isinstance(signed_raw, Mapping):
                raise ValueError(
                    "symmetric hardware-cost feature lacks signed components."
                )
            signed_components: dict[str, float] = {}
            for key in _HARDWARE_COST_COMPONENTS:
                try:
                    value = float(signed_raw[key])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "symmetric hardware-cost feature has incomplete signed "
                        "components."
                    ) from exc
                if not math.isfinite(value) or abs(value) > 1.0 + 1e-12:
                    raise ValueError(
                        "symmetric hardware-cost signed components must be finite "
                        "and lie in [-1, 1]."
                    )
                signed_components[key] = float(max(-1.0, min(1.0, value)))
            signed_index = float(getattr(feat, "hardware_cost_signed_index", 0.0))
            score_factor = float(getattr(feat, "hardware_cost_score_factor", 1.0))
            population_hash = getattr(feat, "hardware_cost_population_hash", None)
            if not math.isfinite(signed_index) or abs(signed_index) > 1.0 + 1e-12:
                raise ValueError(
                    "symmetric hardware-cost signed index must lie in [-1, 1]."
                )
            if (
                not math.isfinite(score_factor)
                or score_factor < 0.5 - 1e-12
                or score_factor > 1.5 + 1e-12
            ):
                raise ValueError(
                    "symmetric hardware-cost score factor must lie in [0.5, 1.5]."
                )
            if (
                not isinstance(population_hash, str)
                or len(population_hash) != 64
                or any(ch not in "0123456789abcdef" for ch in population_hash)
            ):
                raise ValueError(
                    "symmetric hardware-cost feature lacks a valid population SHA-256."
                )
        else:
            signed_components = {key: 0.0 for key in _HARDWARE_COST_COMPONENTS}
            signed_index = 0.0
            score_factor = 1.0
            population_hash = None
        return {
            "lambdas": lambdas,
            "lambda_source": str(lambda_source),
            "bars": signed_components,
            "hardware_cost_excess_sum": 0.0,
            "hardware_cost_denominator": 1.0,
            "hardware_cost_policy": str(mode),
            "hardware_cost_signed_index": float(signed_index),
            "hardware_cost_score_factor": float(score_factor),
            "hardware_cost_population_hash": population_hash,
        }
    bars = _hardware_cost_bar_components(feat)
    excess_sum = float(sum(float(lambdas[key]) * float(bars[key]) for key in _HARDWARE_COST_COMPONENTS))
    denominator = float(max(1.0, 1.0 + max(0.0, excess_sum)))
    return {
        "lambdas": lambdas,
        "lambda_source": str(lambda_source),
        "bars": bars,
        "hardware_cost_excess_sum": float(max(0.0, excess_sum)),
        "hardware_cost_denominator": float(denominator),
        "hardware_cost_policy": str(mode),
        "hardware_cost_signed_index": 0.0,
        "hardware_cost_score_factor": 1.0,
        "hardware_cost_population_hash": None,
    }


def _validated_multiplicative_signed_factor_feature(
    feat: CandidateFeatures,
    cfg: Any,
    *,
    configured_policy: str,
) -> str:
    """Validate one population-normalized multiplicative signed-cost feature."""

    if configured_policy not in HARDWARE_COST_MULTIPLICATIVE_SIGNED_FACTOR_POLICIES:
        raise ValueError(
            "Signed-factor feature validation requires a multiplicative "
            "signed-factor policy."
        )
    feature_policy = str(
        getattr(feat, "hardware_cost_policy", "unresolved") or "unresolved"
    )
    if feature_policy != configured_policy:
        raise ValueError(
            "Signed-factor Phase-III coordinate-model rescore requires a "
            "population-normalized feature whose policy matches the "
            "configured normalization policy."
        )
    normalization = getattr(feat, "hardware_cost_normalization", None)
    if not isinstance(normalization, Mapping) or not normalization:
        raise ValueError(
            "Signed-factor Phase-III feature lacks its normalization receipt."
        )
    if (
        str(normalization.get("schema", ""))
        != HARDWARE_COST_SYMMETRIC_ARCTAN_SCHEMA
        or str(normalization.get("policy", "")) != configured_policy
    ):
        raise ValueError(
            "Signed-factor Phase-III feature was normalized under a different "
            "policy or schema."
        )
    population_hash = getattr(feat, "hardware_cost_population_hash", None)
    if (
        not isinstance(population_hash, str)
        or len(population_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in population_hash)
    ):
        raise ValueError(
            "Signed-factor Phase-III feature lacks a valid population SHA-256."
        )
    phase_cost = getattr(feat, "phase_cost_components", None)
    if not isinstance(phase_cost, Mapping):
        raise ValueError(
            "Signed-factor Phase-III feature lacks cost-component telemetry."
        )
    if (
        normalization.get("population_hash") != population_hash
        or phase_cost.get("hardware_cost_population_hash") != population_hash
        or phase_cost.get("hardware_cost_policy") != configured_policy
    ):
        raise ValueError(
            "Signed-factor Phase-III feature has stale or mixed population "
            "normalization telemetry."
        )

    raw = (
        _signed_compiled_marginal_components(feat)
        if configured_policy
        == HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
        else _hardware_cost_raw_components(feat, strict=True)
    )
    raw_receipt = normalization.get("raw")
    signed_receipt = normalization.get("signed_components")
    feature_signed = getattr(feat, "hardware_cost_signed_components", None)
    if (
        not isinstance(raw_receipt, Mapping)
        or not isinstance(signed_receipt, Mapping)
        or not isinstance(feature_signed, Mapping)
    ):
        raise ValueError(
            "Signed-factor Phase-III feature lacks raw or normalized components."
        )
    medians = normalization.get("medians")
    scales = normalization.get("scales")
    uniform_components = normalization.get("uniform_components")
    if (
        not isinstance(medians, Mapping)
        or not isinstance(scales, Mapping)
        or not isinstance(uniform_components, Mapping)
    ):
        raise ValueError(
            "Signed-factor Phase-III normalization statistics are incomplete."
        )
    scale_floor = _hardware_cost_scale_floor(cfg)
    validated_signed: dict[str, float] = {}
    for key in _HARDWARE_COST_COMPONENTS:
        try:
            raw_value = float(raw[key])
            raw_receipt_value = float(raw_receipt[key])
            median = float(medians[key])
            scale = float(scales[key])
            normalized_value = float(signed_receipt[key])
            feature_value = float(feature_signed[key])
            phase_value = float(phase_cost[f"c_bar_{key}"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Signed-factor Phase-III feature has incomplete normalized "
                "components."
            ) from exc
        if (
            not math.isclose(raw_receipt_value, raw_value, rel_tol=1e-12, abs_tol=1e-12)
            or not math.isfinite(median)
            or not math.isfinite(scale)
            or scale < scale_floor
        ):
            raise ValueError(
                "Signed-factor Phase-III normalization receipt is stale."
            )
        centered = (
            raw_value
            if configured_policy
            == HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
            else raw_value - median
        )
        expected = float((2.0 / math.pi) * math.atan(centered / scale))
        if bool(uniform_components.get(key, False)):
            expected = 0.0
        expected = float(max(-1.0, min(1.0, expected)))
        if not all(
            math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)
            for value in (normalized_value, feature_value, phase_value)
        ):
            raise ValueError(
                "Signed-factor Phase-III normalized components are stale or mixed."
            )
        validated_signed[key] = expected

    lambdas, _lambda_source = resolve_hardware_cost_lambdas(cfg)
    normalization_lambdas = normalization.get("lambdas")
    feature_lambdas = getattr(feat, "hardware_cost_lambdas", None)
    if not isinstance(normalization_lambdas, Mapping) or not isinstance(
        feature_lambdas, Mapping
    ):
        raise ValueError(
            "Signed-factor Phase-III feature lacks normalized cost weights."
        )
    for key in _HARDWARE_COST_COMPONENTS:
        try:
            expected_lambda = float(lambdas[key])
            normalized_lambda = float(normalization_lambdas[key])
            feature_lambda = float(feature_lambdas[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Signed-factor Phase-III feature has incomplete cost weights."
            ) from exc
        if not (
            math.isclose(normalized_lambda, expected_lambda, rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(feature_lambda, expected_lambda, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError(
                "Signed-factor Phase-III feature was normalized with stale cost weights."
            )
    lambda_total = float(sum(float(lambdas[key]) for key in _HARDWARE_COST_COMPONENTS))
    expected_index = (
        0.0
        if lambda_total <= 0.0
        else float(
            sum(
                float(lambdas[key]) * float(validated_signed[key])
                for key in _HARDWARE_COST_COMPONENTS
            )
            / lambda_total
        )
    )
    expected_index = float(max(-1.0, min(1.0, expected_index)))
    expected_factor = float(max(0.5, min(1.5, 1.0 - 0.5 * expected_index)))
    try:
        observed_indices = (
            float(feat.hardware_cost_signed_index),
            float(normalization["signed_index"]),
            float(phase_cost["hardware_cost_signed_index"]),
        )
        observed_factors = (
            float(feat.hardware_cost_score_factor),
            float(normalization["score_factor"]),
            float(phase_cost["hardware_cost_score_factor"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Signed-factor Phase-III feature lacks score-factor closure."
        ) from exc
    if not all(
        math.isclose(value, expected_index, rel_tol=1e-12, abs_tol=1e-12)
        for value in observed_indices
    ) or not all(
        math.isclose(value, expected_factor, rel_tol=1e-12, abs_tol=1e-12)
        for value in observed_factors
    ):
        raise ValueError(
            "Signed-factor Phase-III feature has stale score-factor telemetry."
        )
    return population_hash


def _cheap_burden_total_from_hardware_cost(feat: CandidateFeatures, cfg: Any) -> float:
    return float(_hardware_cost_denominator_payload(feat, cfg)["hardware_cost_denominator"])


def _hardware_cost_normalization_mode(cfg: Any) -> str:
    mode = str(
        getattr(
            cfg,
            "hardware_cost_normalization_mode",
            HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1,
        )
        or HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1
    )
    mode = mode.strip().lower().replace("-", "_")
    aliases = {
        "family": HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1,
        "family_robust": HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1,
        "snake_hardware_cost_family_robust_v1": (
            HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1
        ),
        "symmetric": (
            HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1
        ),
        "symmetric_arctan": (
            HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1
        ),
        "family_robust_symmetric": (
            HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1
        ),
        "snake_hardware_cost_family_robust_symmetric_arctan_v1": (
            HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1
        ),
        "signed_zero_centered": (
            HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
        ),
        "raw": HARDWARE_COST_NORMALIZATION_RAW_LEGACY_V1,
        "legacy_raw": HARDWARE_COST_NORMALIZATION_RAW_LEGACY_V1,
        "raw_legacy": HARDWARE_COST_NORMALIZATION_RAW_LEGACY_V1,
        "snake_hardware_cost_raw_legacy_v1": (
            HARDWARE_COST_NORMALIZATION_RAW_LEGACY_V1
        ),
    }
    mode = aliases.get(mode, mode)
    allowed = {
        HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1,
        HARDWARE_COST_NORMALIZATION_RAW_LEGACY_V1,
        *HARDWARE_COST_MULTIPLICATIVE_SIGNED_FACTOR_POLICIES,
    }
    if mode not in allowed:
        raise ValueError(
            "hardware_cost_normalization_mode must be one of "
            f"{sorted(allowed)}."
        )
    return str(mode)


def require_phase3_signed_factor_consumer_semantic_version(
    cfg: Any,
) -> str | None:
    """Refuse corrected zero-centered Phase-III semantics under old routes."""

    if (
        _hardware_cost_normalization_mode(cfg)
        != HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
    ):
        return None
    observed = str(
        getattr(
            cfg,
            "phase3_signed_factor_consumer_semantic_version",
            "",
        )
        or ""
    )
    if (
        observed
        != PHASE3_ZERO_CENTERED_SIGNED_FACTOR_CONSUMER_SEMANTIC_VERSION
    ):
        raise RuntimeError(
            "Corrected zero-centered Phase-III factor consumption requires "
            "the new Paper-I semantic implementation version; historical "
            "affected route digests are not executable under this consumer."
        )
    return observed


def _legacy_hardware_cost_family_normalization(
    features: Sequence[CandidateFeatures],
    cfg: Any,
) -> dict[str, Any]:
    """Compute manuscript robust positive-excess stats for a live record family."""
    scale_floor = _hardware_cost_scale_floor(cfg)
    raw_by_component: dict[str, list[float]] = {key: [] for key in _HARDWARE_COST_COMPONENTS}
    for feat in features:
        raw = _hardware_cost_raw_components(feat)
        for key in _HARDWARE_COST_COMPONENTS:
            raw_by_component[key].append(float(raw[key]))
    medians: dict[str, float] = {}
    scales: dict[str, float] = {}
    for key in _HARDWARE_COST_COMPONENTS:
        values = [float(v) for v in raw_by_component[key] if math.isfinite(float(v)) and float(v) >= 0.0]
        median = float(np.median(values)) if values else 0.0
        excesses = [float(v - median) for v in values if float(v) > float(median)]
        scale = float(np.median(excesses)) if excesses else scale_floor
        medians[key] = float(max(0.0, median))
        scales[key] = float(max(scale_floor, scale))
    lambdas, lambda_source = resolve_hardware_cost_lambdas(cfg)
    return {
        "schema": HARDWARE_COST_NORMALIZATION_SCHEMA,
        "policy": HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1,
        "medians": medians,
        "scales": scales,
        "scale_floor": float(scale_floor),
        "lambdas": lambdas,
        "lambda_source": str(lambda_source),
    }


def _symmetric_hardware_cost_population_hash(
    *,
    policy: str,
    features: Sequence[CandidateFeatures],
    raw_rows: Sequence[Mapping[str, float]],
    medians: Mapping[str, float],
    scales: Mapping[str, float],
    lambdas: Mapping[str, float],
    scale_floor: float,
) -> str:
    if policy not in HARDWARE_COST_MULTIPLICATIVE_SIGNED_FACTOR_POLICIES:
        raise ValueError(
            "Signed hardware-cost population hash requires a multiplicative "
            "signed-factor policy."
        )
    rows: list[dict[str, Any]] = []
    for feat, raw in zip(features, raw_rows):
        row = {
            "candidate_label": str(feat.candidate_label),
            "candidate_pool_index": int(feat.candidate_pool_index),
            "position_id": int(feat.position_id),
            "raw": {
                key: float(raw[key]) for key in _HARDWARE_COST_COMPONENTS
            },
        }
        if policy == HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1:
            row.update(_zero_centered_signed_population_identity(feat))
        rows.append(row)
    if policy == HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1:
        rows.sort(
            key=lambda row: (
                int(row["candidate_pool_index"]),
                int(row["position_id"]),
                str(row["candidate_label"]),
                str(row["generator_id"]),
                str(row["base_structure_key"]),
                str(row["trial_structure_key"]),
                json.dumps(row["raw"], sort_keys=True, separators=(",", ":")),
            )
        )
    else:
        rows.sort(
            key=lambda row: (
                int(row["candidate_pool_index"]),
                int(row["position_id"]),
                str(row["candidate_label"]),
                json.dumps(row["raw"], sort_keys=True, separators=(",", ":")),
            )
        )
    payload = {
        "schema": HARDWARE_COST_SYMMETRIC_ARCTAN_SCHEMA,
        "policy": str(policy),
        "component_order": list(_HARDWARE_COST_COMPONENTS),
        "rows": rows,
        "medians": {
            key: float(medians[key]) for key in _HARDWARE_COST_COMPONENTS
        },
        "scales": {
            key: float(scales[key]) for key in _HARDWARE_COST_COMPONENTS
        },
        "lambdas": {
            key: float(lambdas[key]) for key in _HARDWARE_COST_COMPONENTS
        },
        "scale_floor": float(scale_floor),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _symmetric_hardware_cost_family_normalization(
    features: Sequence[CandidateFeatures],
    cfg: Any,
) -> dict[str, Any]:
    """Build the signed, bounded population-relative cost normalization."""

    feature_list = list(features)
    scale_floor = _hardware_cost_scale_floor(cfg)
    raw_rows = [
        _hardware_cost_raw_components(feat, strict=True)
        for feat in feature_list
    ]
    raw_by_component = {
        key: [float(raw[key]) for raw in raw_rows]
        for key in _HARDWARE_COST_COMPONENTS
    }
    medians: dict[str, float] = {}
    scales: dict[str, float] = {}
    mad_scales: dict[str, float] = {}
    uniform_components: dict[str, bool] = {}
    for key in _HARDWARE_COST_COMPONENTS:
        values = np.asarray(raw_by_component[key], dtype=float)
        median = float(np.median(values)) if values.size else 0.0
        deviations = np.abs(values - float(median))
        mad = float(np.median(deviations)) if deviations.size else 0.0
        uniform = bool(values.size <= 1 or np.all(values == values[0]))
        medians[key] = float(median)
        mad_scales[key] = float(mad)
        scales[key] = float(max(scale_floor, mad))
        uniform_components[key] = bool(uniform)
    lambdas, lambda_source = resolve_hardware_cost_lambdas(cfg)
    population_hash = _symmetric_hardware_cost_population_hash(
        policy=HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1,
        features=feature_list,
        raw_rows=raw_rows,
        medians=medians,
        scales=scales,
        lambdas=lambdas,
        scale_floor=scale_floor,
    )
    return {
        "schema": HARDWARE_COST_SYMMETRIC_ARCTAN_SCHEMA,
        "policy": (
            HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1
        ),
        "medians": medians,
        "scales": scales,
        "mad_scales": mad_scales,
        "uniform_components": uniform_components,
        "scale_floor": float(scale_floor),
        "lambdas": lambdas,
        "lambda_source": str(lambda_source),
        "population_hash": str(population_hash),
        "component_transform": "(2/pi)*atan((cost-median)/scale)",
        "weighted_index_formula": "sum(lambda_a*u_a)/sum(lambda_a)",
        "score_factor_formula": "1-0.5*weighted_index",
        "score_factor_bounds": [0.5, 1.5],
    }


def _zero_centered_signed_hardware_cost_normalization(
    features: Sequence[CandidateFeatures],
    cfg: Any,
) -> dict[str, Any]:
    """Build a bounded zero-centered transform of signed Qiskit marginals."""

    feature_list = list(features)
    scale_floor = _hardware_cost_scale_floor(cfg)
    raw_rows = [
        _signed_compiled_marginal_components(feat)
        for feat in feature_list
    ]
    scales: dict[str, float] = {}
    uniform_components: dict[str, bool] = {}
    for key in _HARDWARE_COST_COMPONENTS:
        magnitudes = np.asarray(
            [abs(float(raw[key])) for raw in raw_rows],
            dtype=float,
        )
        nonzero = magnitudes[magnitudes > 0.0]
        scale = float(np.median(nonzero)) if nonzero.size else scale_floor
        scales[key] = float(max(scale_floor, scale))
        uniform_components[key] = bool(
            not raw_rows
            or all(float(raw[key]) == 0.0 for raw in raw_rows)
        )
    medians = {key: 0.0 for key in _HARDWARE_COST_COMPONENTS}
    lambdas, lambda_source = resolve_hardware_cost_lambdas(cfg)
    population_hash = _symmetric_hardware_cost_population_hash(
        policy=HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1,
        features=feature_list,
        raw_rows=raw_rows,
        medians=medians,
        scales=scales,
        lambdas=lambdas,
        scale_floor=scale_floor,
    )
    return {
        "schema": HARDWARE_COST_SYMMETRIC_ARCTAN_SCHEMA,
        "policy": (
            HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
        ),
        "medians": medians,
        "scales": scales,
        "mad_scales": dict(scales),
        "uniform_components": uniform_components,
        "scale_floor": float(scale_floor),
        "lambdas": lambdas,
        "lambda_source": str(lambda_source),
        "population_hash": str(population_hash),
        "component_transform": "(2/pi)*atan(raw_signed_delta/scale)",
        "weighted_index_formula": "sum(lambda_a*u_a)/sum(lambda_a)",
        "score_factor_formula": "1-0.5*weighted_index",
        "score_factor_bounds": [0.5, 1.5],
        "zero_centered": True,
    }


def hardware_cost_family_normalization(
    features: Sequence[CandidateFeatures],
    cfg: Any,
) -> dict[str, Any]:
    """Compute the requested population-level hardware-cost normalization."""

    mode = _hardware_cost_normalization_mode(cfg)
    if mode == HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1:
        return _zero_centered_signed_hardware_cost_normalization(
            features, cfg
        )
    if mode == HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_SYMMETRIC_ARCTAN_V1:
        return _symmetric_hardware_cost_family_normalization(features, cfg)
    return _legacy_hardware_cost_family_normalization(features, cfg)


def _hardware_cost_raw_components_from_entry(entry: Any) -> dict[str, float]:
    if isinstance(entry, CandidateFeatures):
        return _hardware_cost_raw_components(entry)
    if isinstance(entry, Mapping):
        return {
            "2q": _finite_nonnegative(entry.get("c_hat_2q", entry.get("2q", 0.0))),
            "d": _finite_nonnegative(entry.get("c_hat_d", entry.get("d", 0.0))),
            "1q": _finite_nonnegative(entry.get("c_hat_1q", entry.get("1q", 0.0))),
            "theta": _finite_nonnegative(entry.get("c_hat_theta", entry.get("theta", 0.0))),
            "shot": _finite_nonnegative(entry.get("c_hat_shot", entry.get("shot", 0.0))),
        }
    return {
        "2q": _finite_nonnegative(getattr(entry, "c_hat_2q", 0.0)),
        "d": _finite_nonnegative(getattr(entry, "c_hat_d", 0.0)),
        "1q": _finite_nonnegative(getattr(entry, "c_hat_1q", 0.0)),
        "theta": _finite_nonnegative(getattr(entry, "c_hat_theta", 0.0)),
        "shot": _finite_nonnegative(getattr(entry, "c_hat_shot", 0.0)),
    }


def hardware_cost_ansatz_entry_denominators(
    entries: Sequence[Any],
    cfg: Any,
) -> dict[str, Any]:
    """Compute the Paper-I hardware denominator over current ansatz entries.

    This mirrors the candidate-family normalization, but the competition set is
    the already-admitted coordinates.  It is intended for prune nomination.
    """

    entry_list = list(entries)
    scale_floor = _hardware_cost_scale_floor(cfg)
    raw_rows = [_hardware_cost_raw_components_from_entry(entry) for entry in entry_list]
    raw_by_component: dict[str, list[float]] = {key: [] for key in _HARDWARE_COST_COMPONENTS}
    for raw in raw_rows:
        for key in _HARDWARE_COST_COMPONENTS:
            raw_by_component[key].append(float(raw[key]))
    medians: dict[str, float] = {}
    scales: dict[str, float] = {}
    for key in _HARDWARE_COST_COMPONENTS:
        values = [
            float(v)
            for v in raw_by_component[key]
            if math.isfinite(float(v)) and float(v) >= 0.0
        ]
        median = float(np.median(values)) if values else 0.0
        excesses = [float(v - median) for v in values if float(v) > float(median)]
        scale = float(np.median(excesses)) if excesses else scale_floor
        medians[key] = float(max(0.0, median))
        scales[key] = float(max(scale_floor, scale))
    lambdas, lambda_source = resolve_hardware_cost_lambdas(cfg)
    rows: list[dict[str, Any]] = []
    denominators: list[float] = []
    for idx, raw in enumerate(raw_rows):
        bars: dict[str, float] = {}
        for key in _HARDWARE_COST_COMPONENTS:
            scale = float(max(float(scales[key]), scale_floor))
            bars[key] = float(math.asinh(max(0.0, float(raw[key]) - float(medians[key])) / scale))
        excess_sum = float(sum(float(lambdas[key]) * float(bars[key]) for key in _HARDWARE_COST_COMPONENTS))
        denominator = float(max(1.0, 1.0 + max(0.0, excess_sum)))
        denominators.append(float(denominator))
        label = (
            str(entry_list[idx].get("label", entry_list[idx].get("candidate_label", idx)))
            if isinstance(entry_list[idx], Mapping)
            else str(getattr(entry_list[idx], "candidate_label", idx))
        )
        rows.append(
            {
                "index": int(idx),
                "label": str(label),
                "raw": dict(raw),
                "bars": dict(bars),
                "hardware_cost_excess_sum": float(max(0.0, excess_sum)),
                "hardware_cost_denominator": float(denominator),
            }
        )
    return {
        "schema": "snake_hardware_cost_ansatz_entry_denominator_v1",
        "scope": "current_ansatz_entries",
        "normalization_schema": HARDWARE_COST_NORMALIZATION_SCHEMA,
        "medians": medians,
        "scales": scales,
        "scale_floor": float(scale_floor),
        "lambdas": lambdas,
        "lambda_source": str(lambda_source),
        "denominators": [float(x) for x in denominators],
        "rows": rows,
    }


def hardware_cost_candidate_record_denominators(
    records: Sequence[Mapping[str, Any]],
    cfg: Any,
) -> dict[str, Any]:
    """Compute the Paper-I cost denominator over a candidate-record family."""

    record_list = [dict(record) for record in records]
    payload = hardware_cost_ansatz_entry_denominators(record_list, cfg)
    payload["schema"] = "snake_hardware_cost_candidate_record_denominator_v1"
    payload["scope"] = "candidate_records"
    rows = [dict(row) for row in payload.get("rows", [])]
    for record, row in zip(record_list, rows):
        row["candidate_pool_index"] = record.get("candidate_pool_index")
        row["position_id"] = record.get("position_id")
    payload["rows"] = rows
    return payload


def apply_hardware_cost_normalization(
    feat: CandidateFeatures,
    cfg: Any,
    normalization: Mapping[str, Any],
) -> CandidateFeatures:
    raw = _hardware_cost_raw_components(feat)
    medians = dict(normalization.get("medians", {})) if isinstance(normalization, Mapping) else {}
    scales = dict(normalization.get("scales", {})) if isinstance(normalization, Mapping) else {}
    bars: dict[str, float] = {}
    for key in _HARDWARE_COST_COMPONENTS:
        median = _finite_nonnegative(medians.get(key, 0.0))
        scale = float(max(_finite_nonnegative(scales.get(key, _hardware_cost_scale_floor(cfg))), _hardware_cost_scale_floor(cfg)))
        bars[key] = float(math.asinh(max(0.0, float(raw[key]) - float(median)) / float(scale)))
    lambdas, lambda_source = resolve_hardware_cost_lambdas(cfg)
    excess_sum = float(sum(float(lambdas[key]) * float(bars[key]) for key in _HARDWARE_COST_COMPONENTS))
    denominator = float(max(1.0, 1.0 + max(0.0, excess_sum)))
    norm_payload = {
        **dict(normalization),
        "schema": str(normalization.get("schema", HARDWARE_COST_NORMALIZATION_SCHEMA)) if isinstance(normalization, Mapping) else HARDWARE_COST_NORMALIZATION_SCHEMA,
        "raw": dict(raw),
        "bars": dict(bars),
        "lambdas": dict(lambdas),
        "lambda_source": str(lambda_source),
        "denominator": float(denominator),
    }
    phase_cost = {
        **dict(feat.phase_cost_components),
        "c_hat_2q": float(raw["2q"]),
        "c_hat_d": float(raw["d"]),
        "c_hat_1q": float(raw["1q"]),
        "c_hat_theta": float(raw["theta"]),
        "c_hat_shot": float(raw["shot"]),
        "c_bar_2q": float(bars["2q"]),
        "c_bar_d": float(bars["d"]),
        "c_bar_1q": float(bars["1q"]),
        "c_bar_theta": float(bars["theta"]),
        "c_bar_shot": float(bars["shot"]),
        "lambda_2q": float(lambdas["2q"]),
        "lambda_d": float(lambdas["d"]),
        "lambda_1q": float(lambdas["1q"]),
        "lambda_theta": float(lambdas["theta"]),
        "lambda_shot": float(lambdas["shot"]),
        "hardware_cost_excess_sum": float(max(0.0, excess_sum)),
        "hardware_cost_denominator": float(denominator),
        "hardware_cost_normalization_schema": HARDWARE_COST_NORMALIZATION_SCHEMA,
    }
    return _replace_feature(
        feat,
        c_bar_2q=float(bars["2q"]),
        c_bar_d=float(bars["d"]),
        c_bar_1q=float(bars["1q"]),
        c_bar_theta=float(bars["theta"]),
        c_bar_shot=float(bars["shot"]),
        hardware_cost_excess_sum=float(max(0.0, excess_sum)),
        hardware_cost_denominator=float(denominator),
        hardware_cost_policy=HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1,
        hardware_cost_signed_components={},
        hardware_cost_signed_index=0.0,
        hardware_cost_score_factor=1.0,
        hardware_cost_population_hash=None,
        hardware_cost_normalization=norm_payload,
        hardware_cost_lambdas={str(k): float(v) for k, v in lambdas.items()},
        hardware_cost_lambda_source=str(lambda_source),
        phase_cost_components=phase_cost,
    )


def apply_symmetric_hardware_cost_normalization(
    feat: CandidateFeatures,
    cfg: Any,
    normalization: Mapping[str, Any],
) -> CandidateFeatures:
    """Apply the bounded signed population-relative cost shaping policy."""

    if not isinstance(normalization, Mapping):
        raise ValueError("symmetric hardware-cost normalization must be a mapping.")
    policy = str(normalization.get("policy", ""))
    schema = str(normalization.get("schema", ""))
    if policy not in HARDWARE_COST_MULTIPLICATIVE_SIGNED_FACTOR_POLICIES:
        raise ValueError(
            "symmetric hardware-cost normalization has an incompatible policy."
        )
    if schema != HARDWARE_COST_SYMMETRIC_ARCTAN_SCHEMA:
        raise ValueError(
            "symmetric hardware-cost normalization has an incompatible schema."
        )
    population_hash = normalization.get("population_hash")
    if (
        not isinstance(population_hash, str)
        or len(population_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in population_hash)
    ):
        raise ValueError(
            "symmetric hardware-cost normalization lacks a valid population SHA-256."
        )

    zero_centered_signed = bool(
        policy
        == HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
    )
    raw = (
        _signed_compiled_marginal_components(feat)
        if zero_centered_signed
        else _hardware_cost_raw_components(feat, strict=True)
    )
    medians = dict(normalization.get("medians", {}))
    scales = dict(normalization.get("scales", {}))
    uniform_components = dict(normalization.get("uniform_components", {}))
    scale_floor = _hardware_cost_scale_floor(cfg)
    signed: dict[str, float] = {}
    for key in _HARDWARE_COST_COMPONENTS:
        try:
            median = float(medians[key])
            scale = float(scales[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "symmetric hardware-cost normalization has incomplete component statistics."
            ) from exc
        if not math.isfinite(median) or (
            not zero_centered_signed and median < 0.0
        ):
            raise ValueError(
                "symmetric hardware-cost medians must be finite and nonnegative."
            )
        if not math.isfinite(scale) or scale < scale_floor:
            raise ValueError(
                "symmetric hardware-cost scales must be finite and respect the floor."
            )
        centered = (
            float(raw[key])
            if zero_centered_signed
            else float(raw[key]) - median
        )
        value = float((2.0 / math.pi) * math.atan(centered / scale))
        if bool(uniform_components.get(key, False)):
            value = 0.0
        signed[key] = float(max(-1.0, min(1.0, value)))

    lambdas, lambda_source = resolve_hardware_cost_lambdas(cfg)
    lambda_total = float(sum(float(lambdas[key]) for key in _HARDWARE_COST_COMPONENTS))
    signed_index = (
        0.0
        if lambda_total <= 0.0
        else float(
            sum(
                float(lambdas[key]) * float(signed[key])
                for key in _HARDWARE_COST_COMPONENTS
            )
            / lambda_total
        )
    )
    signed_index = float(max(-1.0, min(1.0, signed_index)))
    score_factor = float(max(0.5, min(1.5, 1.0 - 0.5 * signed_index)))
    norm_payload = {
        **dict(normalization),
        "raw": dict(raw),
        "signed_components": dict(signed),
        "lambdas": dict(lambdas),
        "lambda_source": str(lambda_source),
        "signed_index": float(signed_index),
        "score_factor": float(score_factor),
        "denominator": 1.0,
    }
    phase_cost = {
        **dict(feat.phase_cost_components),
        "c_hat_2q": float(raw["2q"]),
        "c_hat_d": float(raw["d"]),
        "c_hat_1q": float(raw["1q"]),
        "c_hat_theta": float(raw["theta"]),
        "c_hat_shot": float(raw["shot"]),
        "c_bar_2q": float(signed["2q"]),
        "c_bar_d": float(signed["d"]),
        "c_bar_1q": float(signed["1q"]),
        "c_bar_theta": float(signed["theta"]),
        "c_bar_shot": float(signed["shot"]),
        "lambda_2q": float(lambdas["2q"]),
        "lambda_d": float(lambdas["d"]),
        "lambda_1q": float(lambdas["1q"]),
        "lambda_theta": float(lambdas["theta"]),
        "lambda_shot": float(lambdas["shot"]),
        "hardware_cost_policy": str(policy),
        "hardware_cost_signed_index": float(signed_index),
        "hardware_cost_score_factor": float(score_factor),
        "hardware_cost_population_hash": str(population_hash),
        "hardware_cost_excess_sum": 0.0,
        "hardware_cost_denominator": 1.0,
        "hardware_cost_normalization_schema": HARDWARE_COST_SYMMETRIC_ARCTAN_SCHEMA,
    }
    return _replace_feature(
        feat,
        c_bar_2q=float(signed["2q"]),
        c_bar_d=float(signed["d"]),
        c_bar_1q=float(signed["1q"]),
        c_bar_theta=float(signed["theta"]),
        c_bar_shot=float(signed["shot"]),
        hardware_cost_excess_sum=0.0,
        hardware_cost_denominator=1.0,
        hardware_cost_policy=str(policy),
        hardware_cost_signed_components={str(k): float(v) for k, v in signed.items()},
        hardware_cost_signed_index=float(signed_index),
        hardware_cost_score_factor=float(score_factor),
        hardware_cost_population_hash=str(population_hash),
        hardware_cost_normalization=norm_payload,
        hardware_cost_lambdas={str(k): float(v) for k, v in lambdas.items()},
        hardware_cost_lambda_source=str(lambda_source),
        phase_cost_components=phase_cost,
    )


def apply_raw_legacy_hardware_cost(
    feat: CandidateFeatures,
    cfg: Any,
) -> CandidateFeatures:
    raw = _hardware_cost_raw_components(feat)
    bars = {str(key): float(raw[key]) for key in _HARDWARE_COST_COMPONENTS}
    lambdas, lambda_source = resolve_hardware_cost_lambdas(cfg)
    excess_sum = float(sum(float(lambdas[key]) * float(bars[key]) for key in _HARDWARE_COST_COMPONENTS))
    denominator = float(max(1.0, 1.0 + max(0.0, excess_sum)))
    norm_payload = {
        "schema": HARDWARE_COST_RAW_LEGACY_SCHEMA,
        "mode": "raw_legacy_v1",
        "raw": dict(raw),
        "bars": dict(bars),
        "lambdas": dict(lambdas),
        "lambda_source": str(lambda_source),
        "denominator": float(denominator),
    }
    phase_cost = {
        **dict(feat.phase_cost_components),
        "c_hat_2q": float(raw["2q"]),
        "c_hat_d": float(raw["d"]),
        "c_hat_1q": float(raw["1q"]),
        "c_hat_theta": float(raw["theta"]),
        "c_hat_shot": float(raw["shot"]),
        "c_bar_2q": float(bars["2q"]),
        "c_bar_d": float(bars["d"]),
        "c_bar_1q": float(bars["1q"]),
        "c_bar_theta": float(bars["theta"]),
        "c_bar_shot": float(bars["shot"]),
        "lambda_2q": float(lambdas["2q"]),
        "lambda_d": float(lambdas["d"]),
        "lambda_1q": float(lambdas["1q"]),
        "lambda_theta": float(lambdas["theta"]),
        "lambda_shot": float(lambdas["shot"]),
        "hardware_cost_excess_sum": float(max(0.0, excess_sum)),
        "hardware_cost_denominator": float(denominator),
        "hardware_cost_normalization_schema": HARDWARE_COST_RAW_LEGACY_SCHEMA,
    }
    return _replace_feature(
        feat,
        c_bar_2q=float(bars["2q"]),
        c_bar_d=float(bars["d"]),
        c_bar_1q=float(bars["1q"]),
        c_bar_theta=float(bars["theta"]),
        c_bar_shot=float(bars["shot"]),
        hardware_cost_excess_sum=float(max(0.0, excess_sum)),
        hardware_cost_denominator=float(denominator),
        hardware_cost_policy=HARDWARE_COST_NORMALIZATION_RAW_LEGACY_V1,
        hardware_cost_signed_components={},
        hardware_cost_signed_index=0.0,
        hardware_cost_score_factor=1.0,
        hardware_cost_population_hash=None,
        hardware_cost_normalization=norm_payload,
        hardware_cost_lambdas={str(k): float(v) for k, v in lambdas.items()},
        hardware_cost_lambda_source=str(lambda_source),
        phase_cost_components=phase_cost,
    )


def normalize_hardware_cost_feature_family(
    features: Sequence[CandidateFeatures],
    cfg: Any,
) -> list[CandidateFeatures]:
    mode = _hardware_cost_normalization_mode(cfg)
    if mode == HARDWARE_COST_NORMALIZATION_RAW_LEGACY_V1:
        return [apply_raw_legacy_hardware_cost(feat, cfg) for feat in features]
    if mode == HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1:
        signed_flags = [
            isinstance(feat.compiled_position_cost_backend, Mapping)
            and feat.compiled_position_cost_backend.get(
                "negative_delta_reward_enabled"
            )
            is True
            for feat in features
        ]
        if not any(signed_flags):
            if any(
                str(feat.compile_cost_source) == "backend_transpile_v1"
                for feat in features
            ):
                raise ValueError(
                    "Qiskit-scored population lost signed marginal-cost "
                    "telemetry."
                )
            proxy_cfg = replace(
                cfg,
                hardware_cost_normalization_mode=(
                    HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1
                ),
            )
            normalization = hardware_cost_family_normalization(
                features, proxy_cfg
            )
            return [
                apply_hardware_cost_normalization(
                    feat, proxy_cfg, normalization
                )
                for feat in features
            ]
        if not all(signed_flags):
            raise ValueError(
                "Qiskit-scored population mixes signed and unsigned costs."
            )
    normalization = hardware_cost_family_normalization(features, cfg)
    if mode in HARDWARE_COST_MULTIPLICATIVE_SIGNED_FACTOR_POLICIES:
        return [
            apply_symmetric_hardware_cost_normalization(feat, cfg, normalization)
            for feat in features
        ]
    return [apply_hardware_cost_normalization(feat, cfg, normalization) for feat in features]


def rescore_hardware_cost_family(
    records: Sequence[Mapping[str, Any]],
    cfg: Any,
) -> list[dict[str, Any]]:
    """Normalize hardware costs over records and synchronize selector aliases."""
    feature_records: list[tuple[dict[str, Any], CandidateFeatures]] = []
    for rec in records:
        rec_dict = dict(rec)
        feat = rec_dict.get("feature")
        if isinstance(feat, CandidateFeatures):
            feature_records.append((rec_dict, feat))
    if not feature_records:
        return [dict(rec) for rec in records]
    normalized = normalize_hardware_cost_feature_family([feat for _rec, feat in feature_records], cfg)
    out: list[dict[str, Any]] = []
    norm_iter = iter(normalized)
    for rec in records:
        rec_dict = dict(rec)
        feat = rec_dict.get("feature")
        if isinstance(feat, CandidateFeatures):
            feat_norm = next(norm_iter)
            if isinstance(cfg, SimpleScoreConfig):
                phase1_payload = phase1_score_payload(feat_norm, cfg)
                active_score = float(phase1_payload["active_score"])
                legacy_score = float(phase1_payload["legacy_simple_score"])
                trust_gain = float(phase1_payload["trust_region_gain"])
                trust_score = float(phase1_payload["trust_region_score"])
                burden = float(_cheap_burden_total_from_hardware_cost(feat_norm, cfg))
                cost_factor = float(phase1_payload["hardware_cost_score_factor"])
                feat_norm = _replace_feature(
                    feat_norm,
                    simple_score=float(active_score),
                    cheap_score=float(active_score),
                    cheap_benefit_proxy=float(trust_gain),
                    cheap_burden_total=float(burden),
                    phase1_score_mode=str(phase1_payload["mode"]),
                    phase1_active_score=float(active_score),
                    phase1_legacy_simple_score=float(legacy_score),
                    phase1_trust_region_gain=float(trust_gain),
                    phase1_trust_region_score=float(trust_score),
                    phase1_rho=float(phase1_payload["rho"]),
                    phase1_burden_total=float(burden),
                    selector_score=float(active_score),
                    selector_burden=float(burden),
                    phase_score_components={
                        **dict(feat_norm.phase_score_components),
                        "phase1_legacy_simple_score": float(legacy_score),
                        "phase1_trust_region_score": float(trust_score),
                        "phase1_active_score": float(active_score),
                        "phase1_DeltaE1_TR_hw": float(trust_gain),
                        "phase1_rho": float(phase1_payload["rho"]),
                        "phase1_score": float(active_score),
                        "hardware_cost_score_factor": float(cost_factor),
                        "selector_score": float(active_score),
                    },
                    phase_cost_components={
                        **dict(feat_norm.phase_cost_components),
                        "hardware_cost_denominator": float(burden),
                        "hardware_cost_policy": str(
                            phase1_payload["hardware_cost_policy"]
                        ),
                        "hardware_cost_signed_index": float(
                            phase1_payload["hardware_cost_signed_index"]
                        ),
                        "hardware_cost_score_factor": float(cost_factor),
                        "hardware_cost_population_hash": phase1_payload[
                            "hardware_cost_population_hash"
                        ],
                    },
                )
                rec_dict["feature"] = feat_norm
                rec_dict["simple_score"] = float(active_score)
                rec_dict["cheap_score"] = float(active_score)
                rec_dict["phase1_score_mode"] = str(phase1_payload["mode"])
                rec_dict["phase1_active_score"] = float(active_score)
                rec_dict["phase1_legacy_simple_score"] = float(legacy_score)
                rec_dict["phase1_trust_region_gain"] = float(trust_gain)
                rec_dict["phase1_trust_region_score"] = float(trust_score)
                rec_dict["phase1_rho"] = float(phase1_payload["rho"])
                rec_dict["cheap_burden_total"] = float(burden)
                rec_dict["selector_score"] = float(active_score)
                rec_dict["selector_burden"] = float(burden)
                rec_dict["hardware_cost_policy"] = str(
                    phase1_payload["hardware_cost_policy"]
                )
                rec_dict["hardware_cost_signed_index"] = float(
                    phase1_payload["hardware_cost_signed_index"]
                )
                rec_dict["hardware_cost_score_factor"] = float(cost_factor)
                rec_dict["hardware_cost_population_hash"] = phase1_payload[
                    "hardware_cost_population_hash"
                ]
                out.append(rec_dict)
                continue
            canonical = phase3_canonical_score_components(feat_norm, cfg)
            full_score = float(_full_v2_score_from_components(canonical))
            burden = float(canonical.get("denominator_1_plus_K3", feat_norm.hardware_cost_denominator))
            cost_factor = float(canonical.get("hardware_cost_score_factor", 1.0))
            phase2_trust_gain = float(feat_norm.phase2_raw_trust_gain or 0.0)
            phase2_score = float(
                phase2_trust_gain
                * cost_factor
                / max(float(burden), float(getattr(cfg, "cheap_score_eps", 1e-12)))
            )
            selector_geometry_mode = str(getattr(cfg, "phase3_selector_geometry_mode", "reduced")).strip().lower()
            selector_score = float(phase2_score if selector_geometry_mode == "raw_exact" else full_score)
            selector_burden = float(burden)
            feat_norm = _replace_feature(
                feat_norm,
                phase2_raw_score=float(phase2_score),
                phase2_burden_total=float(burden),
                full_v2_score=float(full_score),
                selector_score=float(selector_score),
                selector_burden=float(selector_burden),
                phase3_primary_score=float(canonical.get("phase3_primary_score", full_score)),
                phase3_burden_total=float(burden),
                phase_score_components={
                    **dict(feat_norm.phase_score_components),
                    "phase3_K3": float(canonical.get("K3", burden - 1.0)),
                    "K3": float(canonical.get("K3", burden - 1.0)),
                    "phase3_denominator_1_plus_K3": float(burden),
                    "denominator_1_plus_K3": float(burden),
                    "phase2_raw_score": float(phase2_score),
                    "phase2_measured_novelty": None,
                    "phase2_novelty_multiplier": None,
                    "phase2_novelty_multiplier_policy": (
                        ORDINARY_NOVELTY_SCORING_RETIRED_V1
                    ),
                    "phase2_gram_novelty_policy": (
                        GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1
                        if _deferred_gram_fallback_enabled(cfg)
                        else "off"
                    ),
                    "phase2_novelty_status": (
                        GRAM_NOVELTY_STATUS_NOT_COMPUTED_FOR_ORDINARY_SCORING
                    ),
                    "phase2_novelty_query_charge": 0,
                    "phase2_novelty_applied": False,
                    "phase2_burden_total": float(burden),
                    "hardware_cost_score_factor": float(cost_factor),
                    "phase3_primary_score": float(canonical.get("phase3_primary_score", full_score)),
                    "selector_score": float(selector_score),
                },
                phase_cost_components={
                    **dict(feat_norm.phase_cost_components),
                    "phase2_burden_total": float(burden),
                    "phase3_burden_total": float(burden),
                    "phase3_denominator_1_plus_K3": float(burden),
                    "hardware_cost_policy": str(
                        canonical.get("hardware_cost_policy", "unresolved")
                    ),
                    "hardware_cost_signed_index": float(
                        canonical.get("hardware_cost_signed_index", 0.0)
                    ),
                    "hardware_cost_score_factor": float(cost_factor),
                    "hardware_cost_population_hash": canonical.get(
                        "hardware_cost_population_hash"
                    ),
                },
            )
            rec_dict["feature"] = feat_norm
            rec_dict["phase2_raw_score"] = float(phase2_score)
            rec_dict["phase2_burden_total"] = float(burden)
            rec_dict["full_v2_score"] = float(full_score)
            rec_dict["selector_score"] = float(selector_score)
            rec_dict["selector_burden"] = float(selector_burden)
            rec_dict["phase3_primary_score"] = float(canonical.get("phase3_primary_score", full_score))
            rec_dict["phase3_burden_total"] = float(burden)
            rec_dict["hardware_cost_policy"] = str(
                canonical.get("hardware_cost_policy", "unresolved")
            )
            rec_dict["hardware_cost_signed_index"] = float(
                canonical.get("hardware_cost_signed_index", 0.0)
            )
            rec_dict["hardware_cost_score_factor"] = float(cost_factor)
            rec_dict["hardware_cost_population_hash"] = canonical.get(
                "hardware_cost_population_hash"
            )
        out.append(rec_dict)
    return out

PHASE2_BATCH_REDUCED_PLANE = "reduced_plane"
PHASE2_BATCH_GREEDY_REDUCED_PLANE = "greedy_reduced_plane"
PHASE2_BATCH_COMBINATORIAL_REDUCED_PLANE = "combinatorial_reduced_plane"
PHASE2_JOINT_RESPONSE_SCORE_FORMULA = "DeltaE_joint_singleton/(1+K2)"
BATCH_SEARCH_POPULATION_RANKED_CHILD_PHASE2_V1 = "ranked_child_phase2_v1"
BATCH_SEARCH_POPULATION_NEAR_DEGENERATE_SHELL_LEGACY_V1 = (
    "near_degenerate_shell_legacy_v1"
)
BATCH_SEARCH_FEASIBILITY_RANK_FEASIBLE_FILL_V1 = "rank_feasible_fill_v1"
BATCH_SEARCH_FEASIBILITY_JOINT_SUBSET_GATE_V1 = "joint_subset_gate_v1"
BATCH_SEARCH_FEASIBILITY_RAW_RANKED_LEGACY_V1 = "raw_ranked_legacy_v1"
BATCH_SEARCH_FEASIBILITY_POLICIES = frozenset(
    {
        BATCH_SEARCH_FEASIBILITY_JOINT_SUBSET_GATE_V1,
        BATCH_SEARCH_FEASIBILITY_RANK_FEASIBLE_FILL_V1,
        BATCH_SEARCH_FEASIBILITY_RAW_RANKED_LEGACY_V1,
    }
)
BATCH_ADDITIVITY_OFF = "off"
BATCH_ADDITIVITY_SOFT_PENALTY_V1 = "soft_penalty_v1"
BATCH_ADDITIVITY_HARD_GATE_LEGACY_V1 = "hard_gate_legacy_v1"
BATCH_ADDITIVITY_POLICIES = frozenset(
    {
        BATCH_ADDITIVITY_OFF,
        BATCH_ADDITIVITY_SOFT_PENALTY_V1,
        BATCH_ADDITIVITY_HARD_GATE_LEGACY_V1,
    }
)
BATCH_GEOMETRY_FULL_RESIDUAL_GRAM_HESSIAN_V1 = (
    "full_residual_gram_hessian_v1"
)
BATCH_GEOMETRY_DIAGONAL_HESSIAN_DIAGNOSTIC_V1 = (
    "diagonal_hessian_diagnostic_v1"
)
BATCH_GEOMETRY_PER_SUBSET_DIAGONAL_HESSIAN_LEGACY_V1 = (
    "per_subset_diagonal_hessian_legacy_v1"
)
BATCH_GEOMETRY_MODES = frozenset(
    {
        BATCH_GEOMETRY_FULL_RESIDUAL_GRAM_HESSIAN_V1,
        BATCH_GEOMETRY_DIAGONAL_HESSIAN_DIAGNOSTIC_V1,
        BATCH_GEOMETRY_PER_SUBSET_DIAGONAL_HESSIAN_LEGACY_V1,
    }
)
BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1 = "full_ansatz_v1"
BATCH_JOINT_CONTEXT_ACTIVE_WINDOW_V1 = "active_window_v1"
BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1 = "batch_only_diagnostic_v1"
BATCH_JOINT_CONTEXT_MODES = frozenset(
    {
        BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        BATCH_JOINT_CONTEXT_ACTIVE_WINDOW_V1,
        BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1,
    }
)
ORDERED_BATCH_BEAM_SELECTION_MODES = frozenset(
    {
        PHASE2_BATCH_GREEDY_REDUCED_PLANE,
        PHASE2_BATCH_COMBINATORIAL_REDUCED_PLANE,
    }
)


def normalize_batch_additivity_policy(raw_policy: Any) -> str:
    policy = str(raw_policy or BATCH_ADDITIVITY_HARD_GATE_LEGACY_V1).strip().lower()
    if policy not in BATCH_ADDITIVITY_POLICIES:
        raise ValueError(
            "batch_additivity_policy must be one of "
            f"{sorted(BATCH_ADDITIVITY_POLICIES)}; got {raw_policy!r}."
        )
    return policy


def normalize_batch_geometry_mode(raw_mode: Any) -> str:
    mode = str(
        raw_mode or BATCH_GEOMETRY_PER_SUBSET_DIAGONAL_HESSIAN_LEGACY_V1
    ).strip().lower()
    if mode not in BATCH_GEOMETRY_MODES:
        raise ValueError(
            f"batch_geometry_mode must be one of {sorted(BATCH_GEOMETRY_MODES)}; "
            f"got {raw_mode!r}."
        )
    return mode


def normalize_batch_joint_context_mode(raw_mode: Any) -> str:
    mode = str(
        raw_mode or BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1
    ).strip().lower()
    if mode not in BATCH_JOINT_CONTEXT_MODES:
        raise ValueError(
            "batch_joint_context_mode must be one of "
            f"{sorted(BATCH_JOINT_CONTEXT_MODES)}; got {raw_mode!r}."
        )
    return mode

_VALID_PHASE2_BATCH_SELECTION_MODES = frozenset(
    {
        PHASE2_BATCH_REDUCED_PLANE,
        PHASE2_BATCH_GREEDY_REDUCED_PLANE,
        PHASE2_BATCH_COMBINATORIAL_REDUCED_PLANE,
        "overlap_orthogonal_benchmark",
        "ceo_commuting_benchmark",
    }
)


def normalize_phase2_batch_selection_mode(raw_mode: Any) -> str:
    """Normalize and validate the phase-2 batch selector mode."""
    mode = "reduced_plane" if raw_mode is None else str(raw_mode).strip().lower()
    aliases = {
        "": PHASE2_BATCH_REDUCED_PLANE,
        "legacy": PHASE2_BATCH_REDUCED_PLANE,
        "default": PHASE2_BATCH_REDUCED_PLANE,
        "greedy": PHASE2_BATCH_GREEDY_REDUCED_PLANE,
        "greedy_reduced": PHASE2_BATCH_GREEDY_REDUCED_PLANE,
        "cost_greedy": PHASE2_BATCH_GREEDY_REDUCED_PLANE,
        "cost_weighted_greedy": PHASE2_BATCH_GREEDY_REDUCED_PLANE,
        "combinatorial": PHASE2_BATCH_COMBINATORIAL_REDUCED_PLANE,
        "combinatorial_reduced": PHASE2_BATCH_COMBINATORIAL_REDUCED_PLANE,
        "combo": PHASE2_BATCH_COMBINATORIAL_REDUCED_PLANE,
        "cost_combinatorial": PHASE2_BATCH_COMBINATORIAL_REDUCED_PLANE,
    }
    mode = aliases.get(mode, mode)
    if mode not in _VALID_PHASE2_BATCH_SELECTION_MODES:
        known = ", ".join(sorted(_VALID_PHASE2_BATCH_SELECTION_MODES))
        raise ValueError(f"phase2_batch_selection_mode must be one of {{{known}}}.")
    return str(mode)



def normalize_phase3_auxiliary_score_mode(raw_mode: Any) -> str:
    """Normalize the Phase3 auxiliary-score mode.

    The canonical/default selector keeps motif, duplicate, family-role, and
    proposal-prior terms out of the primary score.  The additive path remains
    available only as an explicit ablation/diagnostic mode.
    """
    mode = (
        PHASE3_AUXILIARY_SCORE_TIE_BREAK_ONLY
        if raw_mode is None
        else str(raw_mode).strip().lower()
    )
    if mode in {"", "none", "off", "canonical", "tie_break", "tie-break", "tie_break_only"}:
        return PHASE3_AUXILIARY_SCORE_TIE_BREAK_ONLY
    if mode in {"ablation", "additive", "ablation_additive", "legacy_additive"}:
        return PHASE3_AUXILIARY_SCORE_ABLATION_ADDITIVE
    return PHASE3_AUXILIARY_SCORE_TIE_BREAK_ONLY


_SCAFFOLD_OLD_OLD_HESSIAN_SOURCE_SCHEMA = (
    "scaffold_old_old_hessian_source_v1"
)


def _normalize_scaffold_hessian_provenance(
    raw: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return an owned, deterministic, JSON-serializable provenance map."""

    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(
            "old-old Hessian prior provenance must be a mapping."
        )
    if any(not isinstance(key, str) or not key for key in raw):
        raise ValueError(
            "old-old Hessian prior provenance keys must be nonempty strings."
        )
    try:
        encoded = json.dumps(
            dict(raw),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "old-old Hessian prior provenance must be JSON serializable."
        ) from exc
    if not isinstance(normalized, dict):
        raise ValueError(
            "old-old Hessian prior provenance must serialize as an object."
        )
    return dict(normalized)


@dataclass(frozen=True)
class _ScaffoldDerivativeContext:
    psi_state: np.ndarray
    hpsi_state: np.ndarray
    selected_ops: tuple[Any, ...]
    theta: np.ndarray
    psi_ref: np.ndarray
    state_fingerprint: str
    ordered_scaffold_fingerprint: str
    theta_fingerprint: str
    refit_window_indices: tuple[int, ...]
    dpsi_window: tuple[np.ndarray, ...]
    tangents_window: tuple[np.ndarray, ...]
    Q_window: np.ndarray
    H_window_hessian: np.ndarray
    state_reconstruction_delta_norm: float = 0.0
    old_old_geometry_measured: bool = True
    old_old_metric_measured: bool = True
    old_old_hessian_measured: bool = True
    old_old_hessian_fingerprint: str = ""
    old_old_hessian_status: str = ""
    old_old_hessian_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provenance = _normalize_scaffold_hessian_provenance(
            self.old_old_hessian_provenance
        )
        fingerprint = _array_fingerprint(
            np.asarray(self.H_window_hessian, dtype=float)
        )
        supplied_fingerprint = str(self.old_old_hessian_fingerprint).strip()
        if supplied_fingerprint and supplied_fingerprint != fingerprint:
            raise ValueError(
                "old-old Hessian fingerprint does not match the scaffold matrix."
            )
        provenance_status = provenance.get("status")
        supplied_status = str(self.old_old_hessian_status).strip()
        if provenance_status is not None:
            if (
                not isinstance(provenance_status, str)
                or not provenance_status.strip()
            ):
                raise ValueError(
                    "old-old Hessian provenance status must be a nonempty string."
                )
            provenance_status = provenance_status.strip()
            if supplied_status and supplied_status != provenance_status:
                raise ValueError(
                    "old-old Hessian status conflicts with provenance status."
                )
        status = (
            supplied_status
            or provenance_status
            or (
                "exact_measured_v1"
                if bool(self.old_old_hessian_measured)
                else "predicted_prior_unattributed_v1"
            )
        )
        object.__setattr__(self, "old_old_hessian_fingerprint", fingerprint)
        object.__setattr__(self, "old_old_hessian_status", str(status))
        object.__setattr__(self, "old_old_hessian_provenance", provenance)


def _scaffold_hessian_source_telemetry(
    scaffold_context: _ScaffoldDerivativeContext,
) -> dict[str, Any]:
    """Serialize the Hessian source consumed by a candidate scaffold."""

    return {
        "schema": _SCAFFOLD_OLD_OLD_HESSIAN_SOURCE_SCHEMA,
        "status": str(scaffold_context.old_old_hessian_status),
        "hessian_fingerprint": str(
            scaffold_context.old_old_hessian_fingerprint
        ),
        "old_old_hessian_measured": bool(
            scaffold_context.old_old_hessian_measured
        ),
        "provenance": _normalize_scaffold_hessian_provenance(
            scaffold_context.old_old_hessian_provenance
        ),
    }


class Phase1CompileCostOracle:
    """Built-in math expression:
    D_proxy = gate_proxy + shift_span + active_count
    """

    @staticmethod
    def _pauli_weight(label_exyz: str) -> int:
        return int(sum(1 for ch in str(label_exyz) if ch in {"x", "y", "z"}))

    @staticmethod
    def _pauli_xy_count(label_exyz: str) -> int:
        return int(sum(1 for ch in str(label_exyz) if ch in {"x", "y"}))

    @staticmethod
    def _pauli_1q_cost_term(label_exyz: str) -> int:
        label = str(label_exyz)
        weight = int(sum(1 for ch in label if ch in {"x", "y", "z"}))
        if weight <= 0:
            return 0
        x_count = int(sum(1 for ch in label if ch == "x"))
        y_count = int(sum(1 for ch in label if ch == "y"))
        # Direct Pauli rotation template: X uses H/H, Y uses Sdg-H/H-S,
        # and every active word carries one central Rz.
        return int(2 * x_count + 4 * y_count + 1)

    @classmethod
    def _cx_proxy_term(cls, label_exyz: str) -> int:
        return int(2 * max(cls._pauli_weight(label_exyz) - 1, 0))

    @classmethod
    def _sq_proxy_term(cls, label_exyz: str) -> int:
        weight = cls._pauli_weight(label_exyz)
        if weight <= 0:
            return 0
        return int(2 * cls._pauli_xy_count(label_exyz) + 1)

    @classmethod
    def _logical_ladder_span_term(cls, label_exyz: str) -> int:
        return int(max(cls._pauli_weight(label_exyz) - 1, 0))

    def estimate(
        self,
        *,
        candidate_term_count: int,
        position_id: int,
        append_position: int,
        refit_active_count: int,
        candidate_term: Any | None = None,
    ) -> CompileCostEstimate:
        candidate_labels = _pauli_labels_from_term(candidate_term)
        active_labels = [str(lbl) for lbl in candidate_labels if _pauli_weight_exyz(str(lbl)) > 0]
        if active_labels:
            new_pauli_actions = float(len(active_labels))
            new_rotation_steps = float(len(active_labels))
            cx_proxy_total = float(sum(self._cx_proxy_term(lbl) for lbl in active_labels))
            sq_proxy_total = float(sum(self._sq_proxy_term(lbl) for lbl in active_labels))
            gate_proxy_total = float(cx_proxy_total + 0.5 * sq_proxy_total)
            max_pauli_weight = float(max(self._pauli_weight(lbl) for lbl in active_labels))
            c_hat_2q = float(cx_proxy_total)
            c_hat_d = float(sum(2 * self._logical_ladder_span_term(lbl) for lbl in active_labels))
            c_hat_1q = float(sum(self._pauli_1q_cost_term(lbl) for lbl in active_labels))
            c_hat_theta = 1.0
        else:
            fallback_count = float(max(1, int(candidate_term_count)))
            new_pauli_actions = fallback_count
            new_rotation_steps = fallback_count
            cx_proxy_total = fallback_count
            sq_proxy_total = fallback_count
            gate_proxy_total = fallback_count
            max_pauli_weight = 0.0
            c_hat_2q = fallback_count
            c_hat_d = fallback_count
            c_hat_1q = fallback_count
            c_hat_theta = 1.0 if fallback_count > 0.0 else 0.0
        position_shift_span = float(abs(int(append_position) - int(position_id)))
        refit_active = float(max(0, int(refit_active_count)))
        total = float(gate_proxy_total + position_shift_span + refit_active)
        return CompileCostEstimate(
            new_pauli_actions=new_pauli_actions,
            new_rotation_steps=new_rotation_steps,
            position_shift_span=position_shift_span,
            refit_active_count=refit_active,
            proxy_total=total,
            cx_proxy_total=cx_proxy_total,
            sq_proxy_total=sq_proxy_total,
            gate_proxy_total=gate_proxy_total,
            max_pauli_weight=max_pauli_weight,
            c_hat_2q=float(c_hat_2q),
            c_hat_d=float(c_hat_d),
            c_hat_1q=float(c_hat_1q),
            c_hat_theta=float(c_hat_theta),
            hardware_cost_source="proxy_logical_ladder_span_v1",
        )


class MeasurementCacheAudit:
    """Phase 1 accounting-only grouped reuse tracker."""

    def __init__(
        self,
        nominal_shots_per_group: int = 1,
        *,
        plan_version: str = "phase1_qwc_basis_cover_reuse",
        grouping_mode: str = "qwc_basis_cover_reuse",
        sigma_star: float = 1.0,
    ) -> None:
        self._seen_groups: set[str] = set()
        self._nominal_shots = int(max(1, nominal_shots_per_group))
        self._plan_version = str(plan_version)
        self._grouping_mode = str(grouping_mode)
        self._sigma_star = float(max(_finite_nonnegative(sigma_star, 1.0), 1e-12))

    def clone(self) -> "MeasurementCacheAudit":
        cloned = MeasurementCacheAudit(
            nominal_shots_per_group=int(self._nominal_shots),
            plan_version=str(self._plan_version),
            grouping_mode=str(self._grouping_mode),
            sigma_star=float(self._sigma_star),
        )
        cloned._seen_groups = set(self._seen_groups)
        return cloned

    def snapshot(self) -> dict[str, Any]:
        return {
            "seen_groups": sorted(str(x) for x in self._seen_groups),
            "nominal_shots_per_group": int(self._nominal_shots),
            "plan_version": str(self._plan_version),
            "grouping_mode": str(self._grouping_mode),
            "sigma_star": float(self._sigma_star),
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "MeasurementCacheAudit":
        cloned = cls(
            nominal_shots_per_group=int(snapshot.get("nominal_shots_per_group", 1)),
            plan_version=str(snapshot.get("plan_version", "phase1_qwc_basis_cover_reuse")),
            grouping_mode=str(snapshot.get("grouping_mode", "qwc_basis_cover_reuse")),
            sigma_star=float(snapshot.get("sigma_star", 1.0)),
        )
        cloned._seen_groups = {
            str(x)
            for x in snapshot.get("seen_groups", [])
            if str(x) != ""
        }
        return cloned

    def plan_for(self, group_keys: Iterable[Any]) -> MeasurementPlan:
        unique_specs = _compress_measurement_group_specs(group_keys)
        unique_keys = [str(spec.group_key) for spec in unique_specs]
        return MeasurementPlan(
            plan_version=str(self._plan_version),
            group_keys=list(unique_keys),
            nominal_shots_per_group=int(self._nominal_shots),
            grouping_mode=str(self._grouping_mode),
        )

    def estimate(self, group_keys: Iterable[Any]) -> MeasurementCacheStats:
        unique_specs = _compress_measurement_group_specs(group_keys)
        unique_keys = [str(spec.group_key) for spec in unique_specs]

        groups_total = int(len(unique_keys))
        groups_reused = 0
        seen_keys = list(self._seen_groups)
        new_coeff_l2_sum = 0.0
        new_term_count = 0
        source_tags: set[str] = set()
        for spec in unique_specs:
            key = str(spec.group_key)
            source_tags.add(str(spec.source))
            if any(_measurement_basis_key_covers(str(key), str(seen)) for seen in seen_keys):
                groups_reused += 1
                continue
            new_coeff_l2_sum += float(max(0.0, spec.coeff_l2))
            new_term_count += int(max(0, spec.term_count))
        groups_new = int(groups_total - groups_reused)
        shots_reused = float(groups_reused * self._nominal_shots)
        shots_new = float(groups_new * self._nominal_shots)
        reuse_count_cost = float(groups_new)
        # Paper-facing C_shot is the fixed-precision coefficient-norm proxy over
        # newly required measurement groups; count-style work remains in
        # groups_new/shots_new telemetry.
        shot_cost_proxy = float((float(new_coeff_l2_sum) / float(self._sigma_star)) ** 2)
        return MeasurementCacheStats(
            groups_total=groups_total,
            groups_reused=int(groups_reused),
            groups_new=int(groups_new),
            shots_reused=shots_reused,
            shots_new=shots_new,
            reuse_count_cost=reuse_count_cost,
            shot_cost_proxy=float(shot_cost_proxy),
            new_group_coeff_l2_sum=float(new_coeff_l2_sum),
            sigma_star=float(self._sigma_star),
            new_group_term_count=int(new_term_count),
            measurement_cost_source="+".join(sorted(source_tags)) if source_tags else "none",
        )

    def commit(self, group_keys: Iterable[Any]) -> None:
        for spec in _compress_measurement_group_specs(group_keys):
            key = str(spec.group_key)
            key_s = str(key)
            if key_s == "":
                continue
            if any(_measurement_basis_key_covers(key_s, seen) for seen in self._seen_groups):
                continue
            covered = {seen for seen in self._seen_groups if _measurement_basis_key_covers(seen, key_s)}
            if covered:
                self._seen_groups -= covered
            self._seen_groups.add(key_s)

    def summary(self) -> dict[str, float]:
        return {
            "groups_known": float(len(self._seen_groups)),
            "nominal_shots_per_group": float(self._nominal_shots),
            "plan_version": str(self._plan_version),
            "grouping_mode": str(self._grouping_mode),
            "sigma_star": float(self._sigma_star),
        }


def _replace_feature(feat: CandidateFeatures, **updates: Any) -> CandidateFeatures:
    return CandidateFeatures(**{**feat.__dict__, **updates})


def _window_int_list(value: Any, fallback: Sequence[int] | None = None) -> list[int]:
    if value is None:
        return [int(x) for x in (fallback or [])]
    if isinstance(value, (str, bytes, bytearray)):
        return [int(x) for x in (fallback or [])]
    try:
        return [int(x) for x in value]
    except TypeError:
        return [int(x) for x in (fallback or [])]


def _feature_phase2_geometry_window(feat: CandidateFeatures) -> list[int]:
    explicit = _window_int_list(getattr(feat, "phase2_geometry_window_indices", None))
    if explicit:
        return explicit
    return _window_int_list(getattr(feat, "refit_window_indices", None))


def _feature_phase3_schur_window(feat: CandidateFeatures) -> list[int]:
    raw_schur = getattr(feat, "schur_window_indices", None)
    schur = _window_int_list(raw_schur)
    schur_policy = str(getattr(feat, "schur_window_policy", "phase3_geometry_refit_window_alias"))
    p3_policy = str(getattr(feat, "phase3_geometry_window_policy", "legacy_coupled"))
    if raw_schur is not None and (
        schur
        or schur_policy != "phase3_geometry_refit_window_alias"
        or p3_policy == "fixed_local_v1"
    ):
        return schur
    raw_p3 = getattr(feat, "phase3_geometry_refit_window_indices", None)
    p3 = _window_int_list(raw_p3)
    if raw_p3 is not None and (p3 or p3_policy != "legacy_coupled"):
        return p3
    return _feature_phase2_geometry_window(feat)


def _pauli_weight_exyz(label_exyz: str) -> int:
    return int(sum(1 for ch in str(label_exyz) if ch in {"x", "y", "z"}))


def _measurement_basis_key_covers(required_key: str, seen_key: str) -> bool:
    req = str(required_key)
    seen = str(seen_key)
    if len(req) != len(seen):
        return False
    return all((r == "e") or (r == s) for r, s in zip(req, seen))


def _measurement_basis_key_merge(lhs_key: str, rhs_key: str) -> str | None:
    lhs = str(lhs_key)
    rhs = str(rhs_key)
    if len(lhs) != len(rhs):
        return None
    merged: list[str] = []
    for lhs_ch, rhs_ch in zip(lhs, rhs):
        if lhs_ch == "e":
            merged.append(rhs_ch)
            continue
        if rhs_ch in {"e", lhs_ch}:
            merged.append(lhs_ch)
            continue
        return None
    return "".join(merged)


def _compress_measurement_group_keys(group_keys: Iterable[str]) -> list[str]:
    ordered = sorted(
        {str(key) for key in group_keys if str(key) != ""},
        key=lambda key: (-_pauli_weight_exyz(str(key)), str(key)),
    )
    kept: list[str] = []
    for key in ordered:
        if any(_measurement_basis_key_covers(str(key), existing) for existing in kept):
            continue
        kept = [existing for existing in kept if not _measurement_basis_key_covers(existing, str(key))]
        kept.append(str(key))
    return kept


def _measurement_group_spec_from_raw(raw: Any) -> MeasurementGroupSpec | None:
    if isinstance(raw, MeasurementGroupSpec):
        key = str(raw.group_key)
        if key == "":
            return None
        return MeasurementGroupSpec(
            group_key=key,
            coeff_l2=_finite_nonnegative(raw.coeff_l2, 1.0),
            term_count=int(max(1, raw.term_count)),
            source=str(raw.source),
        )
    if isinstance(raw, Mapping):
        key_raw = raw.get("group_key", raw.get("key", None))
        if key_raw is None:
            return None
        key = str(key_raw)
        if key == "":
            return None
        return MeasurementGroupSpec(
            group_key=key,
            coeff_l2=_finite_nonnegative(raw.get("coeff_l2", 1.0), 1.0),
            term_count=int(max(1, raw.get("term_count", 1))),
            source=str(raw.get("source", "mapping_group_spec")),
        )
    key = str(raw)
    if key == "":
        return None
    return MeasurementGroupSpec(group_key=key, coeff_l2=1.0, term_count=1, source="legacy_string_key")


def _compress_measurement_group_specs(group_specs: Iterable[Any]) -> list[MeasurementGroupSpec]:
    specs = [spec for raw in group_specs if (spec := _measurement_group_spec_from_raw(raw)) is not None]
    ordered = sorted(
        specs,
        key=lambda spec: (-_pauli_weight_exyz(str(spec.group_key)), str(spec.group_key)),
    )
    groups: list[dict[str, Any]] = []
    for spec in ordered:
        key = str(spec.group_key)
        coeff_sq = float(max(0.0, spec.coeff_l2)) ** 2
        best_idx: int | None = None
        best_key: str | None = None
        best_delta: tuple[int, int] | None = None
        for idx, group in enumerate(groups):
            merged = _measurement_basis_key_merge(str(group["group_key"]), key)
            if merged is None:
                continue
            delta = (_pauli_weight_exyz(merged) - _pauli_weight_exyz(str(group["group_key"])), idx)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = int(idx)
                best_key = str(merged)
        if best_idx is None or best_key is None:
            groups.append(
                {
                    "group_key": key,
                    "coeff_sq": coeff_sq,
                    "term_count": int(max(1, spec.term_count)),
                    "sources": {str(spec.source)},
                }
            )
        else:
            groups[best_idx]["group_key"] = str(best_key)
            groups[best_idx]["coeff_sq"] = float(groups[best_idx]["coeff_sq"]) + coeff_sq
            groups[best_idx]["term_count"] = int(groups[best_idx]["term_count"]) + int(max(1, spec.term_count))
            groups[best_idx]["sources"] = set(groups[best_idx]["sources"]) | {str(spec.source)}
    final: list[MeasurementGroupSpec] = []
    for group in sorted(groups, key=lambda item: (-_pauli_weight_exyz(str(item["group_key"])), str(item["group_key"]))):
        key = str(group["group_key"])
        if any(_measurement_basis_key_covers(key, str(existing.group_key)) for existing in final):
            continue
        final = [existing for existing in final if not _measurement_basis_key_covers(str(existing.group_key), key)]
        final.append(
            MeasurementGroupSpec(
                group_key=key,
                coeff_l2=float(math.sqrt(max(0.0, float(group["coeff_sq"])))),
                term_count=int(max(1, group["term_count"])),
                source="+".join(sorted(str(x) for x in group["sources"])),
            )
        )
    return final


def compress_measurement_group_keys(group_keys: Iterable[str]) -> list[str]:
    """Public wrapper for QWC measurement group compression.

    Reporting/telemetry code uses this to share the exact basis-cover semantics
    used by the selector measurement cache without depending on private names.
    """

    return [str(spec.group_key) for spec in _compress_measurement_group_specs(group_keys)]


def measurement_basis_key_covers(required_key: str, seen_key: str) -> bool:
    """Public wrapper for selector measurement-basis reuse semantics."""

    return _measurement_basis_key_covers(required_key, seen_key)


def _measurement_group_keys_from_labels(labels: Sequence[str]) -> list[str]:
    active_labels = sorted(
        {str(lbl) for lbl in labels if _pauli_weight_exyz(str(lbl)) > 0},
        key=lambda lbl: (-_pauli_weight_exyz(lbl), lbl),
    )
    groups: list[str] = []
    for label in active_labels:
        best_idx: int | None = None
        best_key: str | None = None
        best_delta: tuple[int, int] | None = None
        for idx, group_key in enumerate(groups):
            merged = _measurement_basis_key_merge(str(group_key), str(label))
            if merged is None:
                continue
            delta = (_pauli_weight_exyz(merged) - _pauli_weight_exyz(str(group_key)), idx)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = int(idx)
                best_key = str(merged)
        if best_idx is None or best_key is None:
            groups.append(str(label))
        else:
            groups[best_idx] = str(best_key)
    return _compress_measurement_group_keys(groups)


def measurement_group_specs_for_term(term: Any) -> list[MeasurementGroupSpec]:
    label_coeffs = _pauli_label_coeffs_from_term(term)
    specs: list[MeasurementGroupSpec] = []
    for label, coeff in label_coeffs:
        label_s = str(label)
        if _pauli_weight_exyz(label_s) <= 0:
            continue
        coeff_abs = abs(complex(coeff))
        if not math.isfinite(float(coeff_abs)) or float(coeff_abs) == 0.0:
            continue
        specs.append(
            MeasurementGroupSpec(
                group_key=label_s,
                coeff_l2=float(coeff_abs),
                term_count=1,
                source="polynomial_coeff_l2_v1",
            )
        )
    if specs:
        return _compress_measurement_group_specs(specs)
    return [
        MeasurementGroupSpec(group_key=str(key), coeff_l2=1.0, term_count=1, source="label_fallback_l2_v1")
        for key in _measurement_group_keys_from_labels(_pauli_labels_from_term(term))
    ]


def measurement_group_keys_for_term(term: Any) -> list[str]:
    return [str(spec.group_key) for spec in measurement_group_specs_for_term(term)]


def _measurement_group_overlap_score(keys_a: Sequence[str], keys_b: Sequence[str]) -> float:
    groups_a = _compress_measurement_group_keys(keys_a)
    groups_b = _compress_measurement_group_keys(keys_b)
    if not groups_a or not groups_b:
        return 1.0

    def _directional(required_groups: Sequence[str], seen_groups: Sequence[str]) -> float:
        if not required_groups:
            return 1.0
        covered = 0
        for req in required_groups:
            if any(_measurement_basis_key_covers(str(req), str(seen)) for seen in seen_groups):
                covered += 1
        return float(covered / len(required_groups))

    return float(
        0.5 * (
            _directional(groups_a, groups_b)
            + _directional(groups_b, groups_a)
        )
    )


def _effective_gate_proxy_total(
    proxy_cost: Mapping[str, Any],
    cfg: SimpleScoreConfig | FullScoreConfig,
) -> float:
    cx_proxy = float(proxy_cost.get("cx_proxy_total", 0.0))
    sq_proxy = float(proxy_cost.get("sq_proxy_total", 0.0))
    rotation_steps = float(proxy_cost.get("new_rotation_steps", 0.0))
    if cx_proxy > 0.0 or sq_proxy > 0.0:
        return float(cfg.compile_cx_proxy_weight) * cx_proxy + float(cfg.compile_sq_proxy_weight) * sq_proxy
    return float(cfg.compile_rotation_step_weight) * rotation_steps


def _effective_compile_proxy_total(
    proxy_cost: Mapping[str, Any],
    cfg: SimpleScoreConfig | FullScoreConfig,
) -> float:
    gate_proxy = _effective_gate_proxy_total(proxy_cost, cfg)
    position_shift = float(proxy_cost.get("position_shift_span", 0.0))
    refit_active = float(proxy_cost.get("refit_active_count", 0.0))
    return float(
        gate_proxy
        + float(cfg.compile_position_shift_weight) * position_shift
        + float(cfg.compile_refit_active_weight) * refit_active
    )


def _effective_depth_cost(
    proxy_cost: Mapping[str, Any],
    cfg: SimpleScoreConfig | FullScoreConfig,
) -> float:
    gate_proxy = _effective_gate_proxy_total(proxy_cost, cfg)
    position_shift = float(proxy_cost.get("position_shift_span", 0.0))
    return float(
        gate_proxy + float(cfg.compile_position_shift_weight) * position_shift
    )


def simple_v1_score(
    feat: CandidateFeatures,
    cfg: SimpleScoreConfig,
) -> float:
    """Return the configured active Phase-I score.

    The historical function name is kept for compatibility.  The active score
    is mode-selected by ``cfg.phase1_score_mode``.
    """
    return float(phase1_score_payload(feat, cfg)["active_score"])


def legacy_simple_v1_score(
    feat: CandidateFeatures,
    cfg: SimpleScoreConfig,
) -> float:
    if not bool(feat.stage_gate_open):
        return float("-inf")
    if not bool(feat.leakage_gate_open):
        return float("-inf")
    if not bool(feat.compile_gate_open):
        return float("-inf")

    hardware_payload = _hardware_cost_denominator_payload(feat, cfg)
    denom = float(hardware_payload["hardware_cost_denominator"])
    cost_factor = float(hardware_payload["hardware_cost_score_factor"])
    return float(float(feat.g_abs) * cost_factor / float(denom))


def phase1_trust_region_gain(
    feat: CandidateFeatures,
    cfg: SimpleScoreConfig,
) -> float:
    if not bool(feat.stage_gate_open):
        return 0.0
    if not bool(feat.leakage_gate_open):
        return 0.0
    if not bool(feat.compile_gate_open):
        return 0.0
    F_measured = float(
        max(
            0.0,
            float(feat.metric_proxy),
            float(feat.F_metric),
        )
    )
    g_hw_lcb = float(_selector_gradient_lcb(feat, cfg))
    energy_model = normalize_phase1_energy_model(
        getattr(
            cfg,
            "phase1_energy_model",
            PHASE1_ENERGY_MODEL_LEGACY_LAMBDA_F_QUADRATIC_V1,
        )
    )
    if energy_model == PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1:
        if g_hw_lcb <= 0.0 or F_measured <= 0.0:
            return 0.0
        return float(
            float(getattr(cfg, "rho", 0.25))
            * float(g_hw_lcb)
            / math.sqrt(float(F_measured))
        )
    F_legacy = float(
        max(
            float(max(0.0, getattr(cfg, "metric_floor", 0.0))),
            float(F_measured),
        )
    )
    return float(
        trust_region_drop(
            g_hw_lcb,
            float(cfg.lambda_F) * float(F_legacy),
            float(F_legacy),
            float(getattr(cfg, "rho", 0.25)),
        )
    )


def phase1_score_payload(
    feat: CandidateFeatures,
    cfg: SimpleScoreConfig,
) -> dict[str, float | str]:
    mode = normalize_phase1_score_mode(getattr(cfg, "phase1_score_mode", None))
    energy_model = normalize_phase1_energy_model(
        getattr(
            cfg,
            "phase1_energy_model",
            PHASE1_ENERGY_MODEL_LEGACY_LAMBDA_F_QUADRATIC_V1,
        )
    )
    hardware_payload = _hardware_cost_denominator_payload(feat, cfg)
    raw_burden = float(hardware_payload["hardware_cost_denominator"])
    raw_cost_factor = float(hardware_payload["hardware_cost_score_factor"])
    resource_weighting_scope = str(
        getattr(
            cfg,
            "resource_weighting_scope",
            "all_phase_resource_weighting_v1",
        )
    ).strip()
    if resource_weighting_scope not in {
        "late_resource_weighting_v1",
        "all_phase_resource_weighting_v1",
    }:
        raise ValueError(
            "resource_weighting_scope must be one of "
            "{'late_resource_weighting_v1',"
            "'all_phase_resource_weighting_v1'}."
        )
    phase1_resource_weighting_active = bool(
        resource_weighting_scope == "all_phase_resource_weighting_v1"
    )
    burden = float(raw_burden if phase1_resource_weighting_active else 1.0)
    cost_factor = float(
        raw_cost_factor if phase1_resource_weighting_active else 1.0
    )
    legacy_score = float(legacy_simple_v1_score(feat, cfg))
    trust_gain = float(phase1_trust_region_gain(feat, cfg))
    gates_open = bool(feat.stage_gate_open) and bool(feat.leakage_gate_open) and bool(feat.compile_gate_open)
    trust_score = (
        float(trust_gain * cost_factor / burden)
        if gates_open
        else float("-inf")
    )
    active_score = trust_score if mode == PHASE1_SCORE_MODE_TRUST_REGION_V1 else legacy_score
    return {
        "mode": str(mode),
        "phase1_energy_model": str(energy_model),
        "phase1_lambda_f_curvature_proxy_applied": bool(
            energy_model
            == PHASE1_ENERGY_MODEL_LEGACY_LAMBDA_F_QUADRATIC_V1
        ),
        "active_score": float(active_score),
        "legacy_simple_score": float(legacy_score),
        "trust_region_gain": float(trust_gain),
        "trust_region_score": float(trust_score),
        "rho": float(getattr(cfg, "rho", 0.25)),
        "burden": float(burden),
        "resource_weighting_scope": str(resource_weighting_scope),
        "phase1_resource_weighting_active": bool(
            phase1_resource_weighting_active
        ),
        "phase1_effective_cost_factor": float(cost_factor),
        "phase1_effective_burden": float(burden),
        "phase1_raw_cost_factor": float(raw_cost_factor),
        "phase1_raw_burden": float(raw_burden),
        "hardware_cost_policy": str(hardware_payload["hardware_cost_policy"]),
        "hardware_cost_signed_index": float(
            hardware_payload["hardware_cost_signed_index"]
        ),
        "hardware_cost_score_factor": float(cost_factor),
        "hardware_cost_population_hash": hardware_payload[
            "hardware_cost_population_hash"
        ],
    }


def normalize_signed(value: float, ref: float) -> float:
    denom = float(ref)
    value_f = float(value)
    if not math.isfinite(value_f):
        return 0.0
    if not math.isfinite(denom) or denom <= 0.0:
        return float(value_f)
    return float(value_f / denom)


def normalize_phase2_selector_gain_mode(raw_mode: Any) -> str:
    mode = (
        PHASE2_SELECTOR_GAIN_TRUST_REGION_V1
        if raw_mode is None
        else str(raw_mode).strip().lower()
    )
    aliases = {
        "": PHASE2_SELECTOR_GAIN_TRUST_REGION_V1,
        "default": PHASE2_SELECTOR_GAIN_TRUST_REGION_V1,
        "trust_region": PHASE2_SELECTOR_GAIN_TRUST_REGION_V1,
        "trust-region": PHASE2_SELECTOR_GAIN_TRUST_REGION_V1,
        "deltae_tr": PHASE2_SELECTOR_GAIN_TRUST_REGION_V1,
        "second_order": PHASE2_SELECTOR_GAIN_TRUST_REGION_V1,
        "second-order": PHASE2_SELECTOR_GAIN_TRUST_REGION_V1,
        "unit": PHASE2_SELECTOR_GAIN_UNIT_GAIN_V1,
        "unit_gain": PHASE2_SELECTOR_GAIN_UNIT_GAIN_V1,
        "novelty_only": PHASE2_SELECTOR_GAIN_UNIT_GAIN_V1,
        "novelty-only": PHASE2_SELECTOR_GAIN_UNIT_GAIN_V1,
        "no_second_order": PHASE2_SELECTOR_GAIN_UNIT_GAIN_V1,
        "no-second-order": PHASE2_SELECTOR_GAIN_UNIT_GAIN_V1,
    }
    mode = aliases.get(mode, mode)
    if mode not in {PHASE2_SELECTOR_GAIN_TRUST_REGION_V1, PHASE2_SELECTOR_GAIN_UNIT_GAIN_V1}:
        raise ValueError(
            "phase2_selector_gain_mode must be one of "
            f"{PHASE2_SELECTOR_GAIN_TRUST_REGION_V1!r} or "
            f"{PHASE2_SELECTOR_GAIN_UNIT_GAIN_V1!r}."
        )
    return str(mode)


def normalize(value: float, ref: float) -> float:
    denom = float(ref)
    if not math.isfinite(denom) or denom <= 0.0:
        return float(max(0.0, value))
    return float(max(0.0, value) / denom)


def trust_region_drop(g_lcb: float, h_eff: float, F: float, rho: float) -> float:
    if float(g_lcb) <= 0.0 or float(F) <= 0.0:
        return 0.0
    h_eff_pos = float(max(0.0, h_eff))
    alpha_max = float(rho) / float(math.sqrt(float(F)))
    if h_eff_pos > 0.0:
        alpha_newton = float(g_lcb) / h_eff_pos
        if alpha_newton <= alpha_max:
            return float(0.5 * float(g_lcb) * float(g_lcb) / h_eff_pos)
    alpha = float(alpha_max)
    return float(float(g_lcb) * alpha - 0.5 * h_eff_pos * alpha * alpha)


def remaining_evaluations_proxy(
    *,
    current_depth: int | None,
    max_depth: int | None,
    mode: str,
    controller_snapshot: Mapping[str, Any] | None = None,
) -> float:
    mode_key = str(mode).strip().lower()
    if mode_key == "none":
        return 0.0
    depth_now = 0 if current_depth is None else int(max(0, current_depth))
    depth_cap = depth_now if max_depth is None else int(max(depth_now, max_depth))
    if mode_key == "remaining_depth":
        snapshot = dict(controller_snapshot) if isinstance(controller_snapshot, Mapping) else {}
        useful_horizon = snapshot.get("useful_horizon")
        if useful_horizon is not None:
            try:
                useful_horizon_val = float(useful_horizon)
            except (TypeError, ValueError):
                useful_horizon_val = float("nan")
            if math.isfinite(useful_horizon_val):
                return float(max(0.0, useful_horizon_val))
        h_t = snapshot.get("H_t")
        if h_t is not None:
            try:
                h_t_val = float(h_t)
            except (TypeError, ValueError):
                h_t_val = float("nan")
            if math.isfinite(h_t_val):
                return float(max(0.0, h_t_val))
        n_rem_hat = snapshot.get("n_rem_hat")
        depth_left = snapshot.get("depth_left")
        if n_rem_hat is not None:
            try:
                n_rem_hat_val = float(n_rem_hat)
            except (TypeError, ValueError):
                n_rem_hat_val = float("nan")
            if math.isfinite(n_rem_hat_val):
                depth_left_val = None
                try:
                    depth_left_val = float(depth_left) if depth_left is not None else None
                except (TypeError, ValueError):
                    depth_left_val = None
                if depth_left_val is None or not math.isfinite(depth_left_val):
                    return float(max(0.0, n_rem_hat_val))
                return float(max(0.0, min(depth_left_val, n_rem_hat_val)))
        return float(max(1, depth_cap - depth_now + 1))
    raise ValueError("remaining_evaluations_proxy_mode must be 'none' or 'remaining_depth'")


def family_repeat_cost_from_history(
    *,
    history_rows: Sequence[Mapping[str, Any]],
    candidate_family: str,
) -> float:
    fam = str(candidate_family).strip()
    if fam == "":
        return 0.0
    tail = [row for row in history_rows if isinstance(row, Mapping) and row.get("candidate_family") is not None]
    if not tail:
        return 0.0
    if str(tail[-1].get("candidate_family", "")).strip() != fam:
        return 0.0
    streak = 0
    for row in reversed(tail):
        if str(row.get("candidate_family", "")).strip() != fam:
            break
        streak += 1
    return float(streak)


def lifetime_weight_components(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> dict[str, float]:
    snapshot = (
        dict(feat.controller_snapshot)
        if isinstance(feat.controller_snapshot, Mapping)
        else {}
    )
    depth_left_raw = snapshot.get("depth_left")
    try:
        depth_left = float(depth_left_raw) if depth_left_raw is not None else None
    except (TypeError, ValueError):
        depth_left = None
    n_rem_hat_raw = snapshot.get("n_rem_hat")
    try:
        n_rem_hat = float(n_rem_hat_raw) if n_rem_hat_raw is not None else float(feat.remaining_evaluations_proxy)
    except (TypeError, ValueError):
        n_rem_hat = float(feat.remaining_evaluations_proxy)
    useful_horizon_raw = snapshot.get("useful_horizon", snapshot.get("H_t", feat.remaining_evaluations_proxy))
    try:
        useful_horizon = float(useful_horizon_raw)
    except (TypeError, ValueError):
        useful_horizon = float(feat.remaining_evaluations_proxy)
    if depth_left is not None and math.isfinite(depth_left):
        useful_horizon = float(min(max(0.0, depth_left), max(0.0, useful_horizon)))
    else:
        useful_horizon = float(max(0.0, useful_horizon))
    h_t = float(useful_horizon)
    if str(cfg.lifetime_cost_mode).strip().lower() == "off":
        return {
            "remaining_evaluations_proxy": float(feat.remaining_evaluations_proxy),
            "n_rem_hat": float(max(0.0, n_rem_hat)),
            "useful_horizon": float(useful_horizon),
            "H_t": float(h_t),
            "compiled": 0.0,
            "measurement": 0.0,
            "optimizer_dim": 0.0,
            "total": 0.0,
        }
    rem = float(useful_horizon)
    compiled = rem * normalize_signed(float(feat.depth_cost), float(cfg.depth_ref))
    measurement = rem * (
        normalize(float(feat.new_group_cost), float(cfg.group_ref))
        + normalize(float(feat.new_shot_cost), float(cfg.shot_ref))
        + normalize(float(feat.reuse_count_cost), float(cfg.reuse_ref))
    )
    optimizer_dim = rem * normalize(float(feat.opt_dim_cost), float(cfg.optdim_ref))
    total = compiled + measurement + optimizer_dim
    return {
        "remaining_evaluations_proxy": float(rem),
        "n_rem_hat": float(max(0.0, n_rem_hat)),
        "useful_horizon": float(useful_horizon),
        "H_t": float(h_t),
        "compiled": float(compiled),
        "measurement": float(measurement),
        "optimizer_dim": float(optimizer_dim),
        "total": float(total),
    }


def _cheap_burden_total(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> float:
    return float(_cheap_burden_total_from_hardware_cost(feat, cfg))


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _phase_confidence_factor(
    g_abs: float,
    sigma_hat: float,
    *,
    z_alpha: float,
    eps: float = 1e-12,
) -> float:
    denom = float(max(abs(float(g_abs)), float(eps)))
    return _clip01(1.0 - float(z_alpha) * float(max(0.0, sigma_hat)) / denom)


def _gradient_resolution_components(
    *,
    g_abs: float,
    sigma_hat: float,
    z_alpha: float,
    hardware_resolution_mode: str,
    manual_b_g_hw: float,
    manual_b_g_drift: float,
) -> dict[str, float | str]:
    """Return Unit-1A hardware-resolution gradient telemetry.

    ``g_lcb`` remains legacy shot-only telemetry. Selector math uses
    ``g_hw_lcb = max(|g| - (z*sigma + b_hw + b_drift), 0)``.
    """
    mode = str(hardware_resolution_mode or "ideal").strip().lower()
    if mode not in {"ideal", "manual"}:
        raise ValueError("hardware_resolution_mode must be one of {'ideal','manual'}.")
    b_hw = float(manual_b_g_hw)
    b_drift = float(manual_b_g_drift)
    if not math.isfinite(b_hw) or b_hw < 0.0:
        raise ValueError("gradient hardware floor must be finite and nonnegative.")
    if not math.isfinite(b_drift) or b_drift < 0.0:
        raise ValueError("gradient drift floor must be finite and nonnegative.")
    if mode == "ideal":
        if b_hw != 0.0 or b_drift != 0.0:
            raise ValueError("ideal hardware_resolution_mode requires zero gradient hardware/drift floors.")
        source = "ideal_zero_floors"
        b_hw = 0.0
        b_drift = 0.0
    else:
        source = "manual_scalar_floors"
    epsilon_g_shot = float(max(0.0, float(z_alpha)) * float(max(0.0, sigma_hat)))
    epsilon_g_res = float(epsilon_g_shot + b_hw + b_drift)
    g_hw_lcb = float(max(float(g_abs) - float(epsilon_g_res), 0.0))
    g_lcb_legacy_shot = float(max(float(g_abs) - float(epsilon_g_shot), 0.0))
    return {
        "epsilon_g_shot": float(epsilon_g_shot),
        "b_g_hw": float(b_hw),
        "b_g_drift": float(b_drift),
        "epsilon_g_res": float(epsilon_g_res),
        "g_hw_lcb": float(g_hw_lcb),
        "g_lcb_legacy_shot": float(g_lcb_legacy_shot),
        "hardware_resolution_mode": str(mode),
        "hardware_resolution_source": str(source),
    }


def phase0_raw_gradient_pilot_components(
    *,
    gradient_signed: float,
    sigma_hat: float | None,
    alpha0: float,
    z_alpha: float,
    hardware_resolution_mode: str,
    manual_b_g_hw: float,
    manual_b_g_drift: float,
) -> dict[str, float | str | bool | None]:
    """Return weak Phase-0 raw-gradient upper-confidence pilot telemetry.

    Unit 1A selector math uses the lower-confidence ``g_hw_lcb``.  The Phase 0
    pilot is intentionally permissive and uses the upper confidence value
    ``|g0_hat| + (z_alpha*sigma0 + b_g_hw + b_g_drift)`` as a one-sided keep
    screen before expensive Phase 1 feature construction.
    """

    gradient_value = float(gradient_signed)
    if not math.isfinite(gradient_value):
        raise ValueError("phase0 gradient_signed must be finite.")
    alpha_value = float(alpha0)
    if (not math.isfinite(alpha_value)) or alpha_value <= 0.0:
        raise ValueError("phase0 alpha0 must be finite and > 0.")
    z_value = float(z_alpha)
    if (not math.isfinite(z_value)) or z_value < 0.0:
        raise ValueError("phase0 z_alpha must be finite and nonnegative.")

    sigma_available = sigma_hat is not None
    if sigma_available:
        try:
            sigma_value = float(sigma_hat)
        except (TypeError, ValueError):
            sigma_available = False
            sigma_value = 0.0
        else:
            if (not math.isfinite(sigma_value)) or sigma_value < 0.0:
                sigma_available = False
                sigma_value = 0.0
    else:
        sigma_value = 0.0

    g_abs = float(abs(gradient_value))
    resolution = _gradient_resolution_components(
        g_abs=float(g_abs),
        sigma_hat=float(sigma_value),
        z_alpha=float(z_value),
        hardware_resolution_mode=str(hardware_resolution_mode),
        manual_b_g_hw=float(manual_b_g_hw),
        manual_b_g_drift=float(manual_b_g_drift),
    )
    epsilon_g_res = float(resolution["epsilon_g_res"])
    g_upper_hw = float(g_abs + epsilon_g_res)
    return {
        "phase0_pilot_schema": "phase0_raw_gradient_upper_v1",
        "phase0_raw_gradient_signed": float(gradient_value),
        "phase0_raw_gradient_abs": float(g_abs),
        "phase0_sigma_hat": float(sigma_value),
        "phase0_sigma_hat_available": bool(sigma_available),
        "phase0_epsilon_g_shot": float(resolution["epsilon_g_shot"]),
        "phase0_b_g_hw": float(resolution["b_g_hw"]),
        "phase0_b_g_drift": float(resolution["b_g_drift"]),
        "phase0_epsilon_g_res": float(epsilon_g_res),
        "phase0_g_upper_hw": float(g_upper_hw),
        "phase0_delta_e_upper_hw": float(alpha_value * g_upper_hw),
        "phase0_alpha": float(alpha_value),
        "phase0_hardware_resolution_mode": str(resolution["hardware_resolution_mode"]),
        "phase0_hardware_resolution_source": str(resolution["hardware_resolution_source"]),
    }



def _feature_has_unit1a_resolution(feat: CandidateFeatures) -> bool:
    return bool(
        float(getattr(feat, "epsilon_g_shot", 0.0) or 0.0) > 0.0
        or float(getattr(feat, "epsilon_g_res", 0.0) or 0.0) > 0.0
        or float(getattr(feat, "b_g_hw", 0.0) or 0.0) > 0.0
        or float(getattr(feat, "b_g_drift", 0.0) or 0.0) > 0.0
        or float(getattr(feat, "g_lcb_legacy_shot", 0.0) or 0.0) > 0.0
        or str(getattr(feat, "hardware_resolution_source", "legacy_unset")) not in {"", "legacy_unset"}
    )


def _selector_gradient_resolution(
    feat: CandidateFeatures,
    cfg: SimpleScoreConfig | FullScoreConfig | None = None,
) -> dict[str, float | str]:
    """Selector-facing Unit-1A resolution with config-aware legacy fallback."""
    if cfg is not None:
        return _gradient_resolution_components(
            g_abs=float(feat.g_abs),
            sigma_hat=float(max(0.0, feat.sigma_hat)),
            z_alpha=float(getattr(cfg, "z_alpha", 0.0)),
            hardware_resolution_mode=str(getattr(cfg, "hardware_resolution_mode", "ideal")),
            manual_b_g_hw=float(getattr(cfg, "manual_b_g_hw", 0.0)),
            manual_b_g_drift=float(getattr(cfg, "manual_b_g_drift", 0.0)),
        )
    if _feature_has_unit1a_resolution(feat):
        return {
            "epsilon_g_shot": float(max(0.0, float(getattr(feat, "epsilon_g_shot", 0.0) or 0.0))),
            "b_g_hw": float(max(0.0, float(getattr(feat, "b_g_hw", 0.0) or 0.0))),
            "b_g_drift": float(max(0.0, float(getattr(feat, "b_g_drift", 0.0) or 0.0))),
            "epsilon_g_res": float(max(0.0, float(getattr(feat, "epsilon_g_res", 0.0) or 0.0))),
            "g_hw_lcb": float(max(0.0, float(getattr(feat, "g_hw_lcb", 0.0) or 0.0))),
            "g_lcb_legacy_shot": float(max(0.0, float(getattr(feat, "g_lcb_legacy_shot", getattr(feat, "g_lcb", 0.0)) or 0.0))),
            "hardware_resolution_mode": str(getattr(feat, "hardware_resolution_mode", "ideal") or "ideal"),
            "hardware_resolution_source": str(getattr(feat, "hardware_resolution_source", "legacy_unset") or "legacy_unset"),
        }
    legacy_lcb = float(max(0.0, float(getattr(feat, "g_lcb", 0.0) or 0.0)))
    return {
        "epsilon_g_shot": 0.0,
        "b_g_hw": 0.0,
        "b_g_drift": 0.0,
        "epsilon_g_res": 0.0,
        "g_hw_lcb": float(legacy_lcb),
        "g_lcb_legacy_shot": float(legacy_lcb),
        "hardware_resolution_mode": "legacy",
        "hardware_resolution_source": "legacy_feature_fallback",
    }


def _selector_gradient_lcb(
    feat: CandidateFeatures,
    cfg: SimpleScoreConfig | FullScoreConfig | None = None,
) -> float:
    """Selector-facing lower-confidence gradient with config-aware legacy fallback.

    When a scoring config is provided, direct/full scoring APIs apply its
    current hardware-resolution mode and floors even to old/directly
    constructed feature snapshots.
    """
    return float(_selector_gradient_resolution(feat, cfg).get("g_hw_lcb", 0.0))


def _deferred_gram_fallback_enabled(cfg: FullScoreConfig) -> bool:
    """Return the sole live Gram-residual policy.

    Ordinary Phase-II/III novelty scoring is retired.  This flag authorizes
    only the all-energy-models-infeasible geometry-expansion fallback.
    """

    return bool(getattr(cfg, "deferred_gram_fallback_enabled", False))


def _deferred_gram_fallback_ridge(cfg: FullScoreConfig) -> float:
    raw = getattr(cfg, "deferred_gram_fallback_ridge", 1e-6)
    try:
        ridge = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "deferred_gram_fallback_ridge must be finite and nonnegative."
        ) from exc
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError(
            "deferred_gram_fallback_ridge must be finite and nonnegative."
        )
    return ridge


def _phase3_window_relaxation_disabled(cfg: FullScoreConfig) -> bool:
    mode = str(getattr(cfg, "phase3_window_relaxation_mode", "reduced")).strip().lower()
    return mode in {"off", "none", "raw", "disabled", "no_relaxation"}


def phase2_raw_geometry_score(
    feat: CandidateFeatures,
    *,
    F_raw: float,
    h_raw: float,
    q_window: Sequence[float],
    Q_window: np.ndarray,
    cfg: FullScoreConfig,
) -> dict[str, Any]:
    """Score Phase II from response gain and cost, without ordinary novelty.

    The Gram cross-block inputs remain in the call contract because the same
    exact geometry is reused by insertion, supported response, and the
    deferred all-models-infeasible fallback.  They are not ranking inputs.
    """

    del q_window, Q_window
    curvature_policy = normalize_phase2_curvature_policy(
        getattr(
            cfg,
            "phase2_curvature_policy",
            PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1,
        )
    )
    cheap_curvature_proxy_policy = (
        normalize_phase2_cheap_curvature_proxy_policy(
            getattr(
                cfg,
                "phase2_cheap_curvature_proxy_policy",
                PHASE2_CHEAP_CURVATURE_PROXY_POLICY_LEGACY_LAMBDA_F_RATIO_V1,
            )
        )
    )
    if (
        curvature_policy
        == PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1
    ):
        try:
            h_raw = float(h_raw)
        except (TypeError, ValueError) as exc:
            raise Phase2CurvatureConstructionError(
                "RA-ADAPT Phase-II measured-required curvature is malformed."
            ) from exc
        if not math.isfinite(float(h_raw)):
            raise Phase2CurvatureConstructionError(
                "RA-ADAPT Phase-II measured-required curvature is nonfinite."
            )
        if (
            cheap_curvature_proxy_policy
            != PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF
        ):
            raise Phase2CurvatureConstructionError(
                "RA-ADAPT Phase-II measured-required curvature cannot be "
                "paired with a lambda-F cheap proxy."
            )

    gain_mode = normalize_phase2_selector_gain_mode(
        getattr(cfg, "phase2_selector_gain_mode", None)
    )
    metric_floor = float(max(float(cfg.metric_floor), 0.0))
    F_raw_nonnegative = float(max(0.0, float(F_raw)))
    F_for_gain = (
        F_raw_nonnegative
        if curvature_policy
        == PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1
        else float(max(F_raw_nonnegative, metric_floor))
    )
    confidence = _phase_confidence_factor(
        float(feat.g_abs),
        float(feat.sigma_hat),
        z_alpha=float(cfg.z_alpha),
    )
    g_hw_lcb = float(_selector_gradient_lcb(feat, cfg))
    trust_region_gain = float(
        trust_region_drop(
            g_hw_lcb,
            float(max(0.0, h_raw)),
            F_for_gain,
            float(cfg.rho),
        )
    )
    score_gain = (
        1.0
        if gain_mode == PHASE2_SELECTOR_GAIN_UNIT_GAIN_V1
        else float(trust_region_gain)
    )
    hardware_payload = _hardware_cost_denominator_payload(feat, cfg)
    burden_total = float(hardware_payload["hardware_cost_denominator"])
    cost_factor = float(hardware_payload["hardware_cost_score_factor"])
    score = float(
        score_gain
        * cost_factor
        / max(burden_total, float(cfg.cheap_score_eps))
    )
    return {
        "phase2_curvature_policy": str(curvature_policy),
        "phase2_cheap_curvature_proxy_policy": str(
            cheap_curvature_proxy_policy
        ),
        "phase2_lambda_f_proxy_applied": False,
        "phase2_missing_curvature_fallback_used": False,
        "confidence_factor": float(confidence),
        "phase2_raw_overlap_max": None,
        "phase2_raw_novelty": None,
        "phase2_measured_novelty": None,
        "phase2_novelty_multiplier": None,
        "phase2_novelty_multiplier_policy": (
            ORDINARY_NOVELTY_SCORING_RETIRED_V1
        ),
        "phase2_gram_novelty_policy": (
            GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1
            if _deferred_gram_fallback_enabled(cfg)
            else "off"
        ),
        "phase2_novelty_status": (
            GRAM_NOVELTY_STATUS_NOT_COMPUTED_FOR_ORDINARY_SCORING
        ),
        "phase2_novelty_query_charge": 0,
        "phase2_novelty_classical_solve_count": 0,
        "phase2_novelty_applied": False,
        "phase2_novelty_mode": ORDINARY_NOVELTY_SCORING_RETIRED_V1,
        "phase2_novelty_source": ORDINARY_NOVELTY_SCORING_RETIRED_V1,
        "phase2_novelty_fallback_reason": None,
        "phase2_span_projection_z": None,
        "phase2_novelty_ridge_used": None,
        "phase2_raw_F_effective": float(F_for_gain),
        "phase2_legacy_pairwise_novelty": None,
        "phase2_confidence_applied": False,
        "phase2_selector_gain_mode": str(gain_mode),
        "phase2_raw_score_formula": str(
            PHASE2_NO_NOVELTY_UNIT_GAIN_RAW_SCORE_FORMULA
            if gain_mode == PHASE2_SELECTOR_GAIN_UNIT_GAIN_V1
            else PHASE2_NO_NOVELTY_RAW_SCORE_FORMULA
        ),
        "phase2_trust_region_gain": float(trust_region_gain),
        "phase2_raw_trust_gain": float(score_gain),
        "phase2_g_hw_lcb": float(g_hw_lcb),
        "phase2_burden_total": float(burden_total),
        "hardware_cost_policy": str(
            hardware_payload["hardware_cost_policy"]
        ),
        "hardware_cost_signed_index": float(
            hardware_payload["hardware_cost_signed_index"]
        ),
        "hardware_cost_score_factor": float(cost_factor),
        "hardware_cost_population_hash": hardware_payload[
            "hardware_cost_population_hash"
        ],
        "phase2_raw_score": float(score),
    }

def phase_shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    threshold: float,
    cap: int,
    frontier_ratio: float,
    tie_break_score_key: str | None = None,
    shortlist_flag: str | None = None,
    score_eps: float = 1e-12,
) -> list[dict[str, Any]]:
    def _record_score(rec: Mapping[str, Any], key: str | None, default: float = float("-inf")) -> float:
        if key is None:
            return 0.0
        raw = rec.get(key, default)
        if raw is None:
            return float(default)
        return float(raw)

    ranked = sorted(
        [dict(rec) for rec in records if float(_record_score(rec, score_key)) >= float(threshold)],
        key=lambda rec: (
            -_record_score(rec, score_key),
            -_record_score(rec, tie_break_score_key),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        ),
    )
    if not ranked:
        return []
    cap_eff = int(max(1, min(int(cap), len(ranked))))
    shortlist_size = int(cap_eff)
    frontier_cut = float(max(0.0, min(1.0, frontier_ratio)))
    # Treat 1.0 as "no frontier cut" so callers can make the frontier nonbinding.
    frontier_enabled = bool(0.0 < frontier_cut < 1.0)
    if cap_eff > 1 and frontier_enabled:
        for idx in range(cap_eff - 1):
            s_cur = float(_record_score(ranked[idx], score_key))
            s_next = float(_record_score(ranked[idx + 1], score_key))
            ratio = float((s_next + float(score_eps)) / (s_cur + float(score_eps)))
            if ratio <= frontier_cut:
                shortlist_size = int(idx + 1)
                break
    out: list[dict[str, Any]] = []
    for idx, rec in enumerate(ranked[:shortlist_size], start=1):
        updated = dict(rec)
        feat = updated.get("feature")
        if isinstance(feat, CandidateFeatures):
            replacement_kwargs: dict[str, Any] = {
                "shortlist_rank": int(idx),
                "shortlist_size": int(shortlist_size),
            }
            if shortlist_flag is not None and hasattr(feat, str(shortlist_flag)):
                replacement_kwargs[str(shortlist_flag)] = True
            updated["feature"] = _replace_feature(feat, **replacement_kwargs)
        out.append(updated)
    return out


def phase3_cheap_ratio_v1(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> dict[str, float | str | None]:
    cheap_proxy_policy = normalize_phase2_cheap_curvature_proxy_policy(
        getattr(
            cfg,
            "phase2_cheap_curvature_proxy_policy",
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY_LEGACY_LAMBDA_F_RATIO_V1,
        )
    )
    if cheap_proxy_policy == PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF:
        raise Phase2CurvatureConstructionError(
            "The Phase-II lambda-F cheap curvature proxy is disabled; "
            "Phase-I survivors must use measured Phase-II curvature."
        )
    metric_source = float(feat.cheap_metric_proxy)
    if metric_source <= 0.0:
        metric_source = float(feat.metric_proxy)
    cheap_metric_proxy = float(max(0.0, metric_source))
    cheap_burden_total = float(_cheap_burden_total(feat, cfg))
    base_payload = {
        "cheap_score_version": "phase3_cheap_ratio_v1",
        "cheap_metric_proxy": float(cheap_metric_proxy),
        "cheap_benefit_proxy": 0.0,
        "cheap_burden_total": float(cheap_burden_total),
    }
    if not bool(feat.stage_gate_open):
        return {**base_payload, "cheap_score": float("-inf")}
    if not bool(feat.leakage_gate_open):
        return {**base_payload, "cheap_score": float("-inf")}
    if not bool(feat.compile_gate_open):
        return {**base_payload, "cheap_score": float("-inf")}

    g_hw_lcb = float(_selector_gradient_lcb(feat, cfg))
    if g_hw_lcb <= 0.0 or cheap_metric_proxy <= 0.0:
        return {**base_payload, "cheap_score": 0.0}

    lambda_F_eff = float(max(float(cfg.lambda_F), float(cfg.cheap_score_eps)))
    cheap_benefit_proxy = float(
        float(g_hw_lcb) * float(g_hw_lcb) / (2.0 * float(lambda_F_eff) * float(cheap_metric_proxy))
    )
    cheap_score = float(
        float(cheap_benefit_proxy)
        / float(float(cheap_burden_total) + float(cfg.cheap_score_eps))
    )
    return {
        **base_payload,
        "cheap_score": float(cheap_score),
        "cheap_benefit_proxy": float(cheap_benefit_proxy),
    }


def _phase3_plateau_novelty_value(feat: CandidateFeatures, plateau_novelty: Any = None) -> float | None:
    raw = plateau_novelty
    if raw is None:
        payload = getattr(feat, "route_c_plateau_acquisition", None)
        if isinstance(payload, Mapping):
            for key in ("phase3_plateau_novelty", "plateau_novelty", "N3_plat", "n3_plat"):
                if payload.get(key) is not None:
                    raw = payload.get(key)
                    break
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return float(min(1.0, max(0.0, value)))


def _phase3_plateau_log_volume_geometry_components(
    *,
    F_raw: Any = None,
    Q_window: Any = None,
    q_window: Any = None,
    novelty_fallback: float | None = None,
    lambda_vol: float = 1e-8,
    metric_floor: float = 1e-12,
    ridge_growth_factor: float = 10.0,
    ridge_max_steps: int = 8,
) -> dict[str, Any]:
    """Return Route-C raw tangent residual and log-volume geometry telemetry.

    The primitive data are the admitted-context Fubini--Study Gram ``Q_window``,
    candidate/context overlap ``q_window``, and candidate norm ``F_raw``.
    ``sigma_perp`` uses the Moore--Penrose projection residual for gates.
    ``sigma_perp_lambda`` uses the ridge residual required by the v1.2
    log-determinant score.
    """

    try:
        F_val = float(F_raw)
    except (TypeError, ValueError):
        if novelty_fallback is None:
            return {
                "phase3_plateau_geometry_available": False,
                "phase3_plateau_geometry_error": "missing_or_malformed_F_raw",
            }
        F_val = float(max(float(metric_floor), 0.0))
    if (not math.isfinite(F_val)) or F_val < 0.0:
        return {
            "phase3_plateau_geometry_available": False,
            "phase3_plateau_geometry_error": "missing_or_malformed_F_raw",
        }
    F_val = float(max(0.0, F_val))
    lambda_val = float(lambda_vol)
    if (not math.isfinite(lambda_val)) or lambda_val <= 0.0:
        return {
            "phase3_plateau_geometry_available": False,
            "phase3_plateau_geometry_error": "malformed_lambda_vol",
        }
    metric_floor_val = float(max(float(metric_floor), 0.0))
    try:
        q_vec = np.asarray([] if q_window is None else q_window, dtype=float).reshape(-1)
        q_len = int(q_vec.size)
        G_raw = np.asarray(
            np.zeros((q_len, q_len), dtype=float) if Q_window is None else Q_window,
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        return {
            "phase3_plateau_geometry_available": False,
            "phase3_plateau_geometry_error": f"malformed_context_geometry:{exc}",
        }
    if G_raw.size == 0 and q_len == 0:
        G = np.zeros((0, 0), dtype=float)
    else:
        if G_raw.ndim != 2 or G_raw.shape[0] != G_raw.shape[1] or G_raw.shape[0] != q_len:
            return {
                "phase3_plateau_geometry_available": False,
                "phase3_plateau_geometry_error": "context_geometry_shape_mismatch",
                "phase3_plateau_context_dimension": int(q_len),
                "phase3_plateau_Q_window_shape": [int(x) for x in G_raw.shape],
            }
        if (not np.all(np.isfinite(G_raw))) or (not np.all(np.isfinite(q_vec))):
            return {
                "phase3_plateau_geometry_available": False,
                "phase3_plateau_geometry_error": "nonfinite_context_geometry",
            }
        G = 0.5 * (G_raw + G_raw.T)

    if q_len == 0:
        sigma_perp = float(max(0.0, F_val))
        sigma_lambda = float(max(0.0, F_val))
        ridge_used = float(lambda_val)
        solve_mode = "empty_context_v1"
        rank_context = 0
        rank_augmented = 1 if F_val > 0.0 else 0
    else:
        try:
            pinv = np.linalg.pinv(G, rcond=1e-10)
            projected = float(q_vec.T @ pinv @ q_vec)
            sigma_perp = float(max(0.0, F_val - projected))
        except np.linalg.LinAlgError as exc:
            if novelty_fallback is None:
                return {
                    "phase3_plateau_geometry_available": False,
                    "phase3_plateau_geometry_error": f"pinv_failed:{exc}",
                }
            sigma_perp = float(max(0.0, float(novelty_fallback) * max(F_val, metric_floor_val)))

        eye = np.eye(q_len, dtype=float)
        ridge_used = float(lambda_val)
        solve_mode = "ridge_context_v1"
        sigma_lambda = None
        for _step in range(int(max(1, ridge_max_steps))):
            try:
                sol = np.linalg.solve(G + ridge_used * eye, q_vec)
                sigma_lambda = float(max(0.0, F_val - float(q_vec.T @ sol)))
                break
            except np.linalg.LinAlgError:
                ridge_used *= float(max(1.1, ridge_growth_factor))
                solve_mode = "ridge_grown_context_v1"
        if sigma_lambda is None:
            return {
                "phase3_plateau_geometry_available": False,
                "phase3_plateau_geometry_error": "ridge_solve_failed",
            }
        try:
            evals = np.linalg.eigvalsh(G)
            max_eval = float(max(float(np.max(evals)), 0.0)) if evals.size else 0.0
            rank_floor = max(1e-12, 1e-10 * max_eval)
            rank_context = int(np.count_nonzero(evals > rank_floor))
            augmented = np.block([[G, q_vec[:, None]], [q_vec[None, :], np.array([[F_val]], dtype=float)]])
            aug_evals = np.linalg.eigvalsh(0.5 * (augmented + augmented.T))
            aug_max_eval = float(max(float(np.max(aug_evals)), 0.0)) if aug_evals.size else 0.0
            aug_rank_floor = max(1e-12, 1e-10 * aug_max_eval)
            rank_augmented = int(np.count_nonzero(aug_evals > aug_rank_floor))
        except np.linalg.LinAlgError:
            rank_context = None
            rank_augmented = None

    F_safe = float(max(F_val, metric_floor_val))
    fractional_residual = 0.0 if F_safe <= 0.0 else float(min(1.0, max(0.0, sigma_perp / F_safe)))
    log_volume_gain = float(math.log1p(float(sigma_lambda) / float(lambda_val)))
    return {
        "phase3_plateau_geometry_available": True,
        "phase3_plateau_geometry_source": "raw_active_dormant_qim_v1",
        "phase3_plateau_geometry_solve_mode": str(solve_mode),
        "phase3_plateau_context_dimension": int(q_len),
        "phase3_plateau_rank_context": None if rank_context is None else int(rank_context),
        "phase3_plateau_rank_augmented": None if rank_augmented is None else int(rank_augmented),
        "phase3_plateau_rank_delta": (
            None
            if rank_context is None or rank_augmented is None
            else int(rank_augmented) - int(rank_context)
        ),
        "phase3_plateau_F_raw": float(F_val),
        "phase3_plateau_F_safe": float(F_safe),
        "phase3_plateau_sigma_perp": float(sigma_perp),
        "phase3_plateau_sigma_perp_lcb": float(sigma_perp),
        "phase3_plateau_sigma_perp_lambda": float(sigma_lambda),
        "phase3_plateau_sigma_perp_lambda_lcb": float(sigma_lambda),
        "phase3_plateau_fractional_residual": float(fractional_residual),
        "phase3_plateau_fractional_residual_lcb": float(fractional_residual),
        "phase3_plateau_log_volume_gain": float(log_volume_gain),
        "phase3_plateau_log_volume_gain_lcb": float(log_volume_gain),
        "phase3_plateau_lambda_vol": float(lambda_val),
        "phase3_plateau_lambda_vol_used": float(ridge_used),
    }


def phase3_plateau_novelty_cost_score_components(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
    *,
    plateau_novelty: Any = None,
    acquisition_score: Any = PLATEAU_ACQUISITION_SCORE_FRACTIONAL_RESIDUAL_V1,
    F_raw: Any = None,
    Q_window: Any = None,
    q_window: Any = None,
    lambda_vol: float = 1e-8,
    sigma_min: float = 0.0,
    nu_min: float = 0.0,
    volume_min: float = 0.0,
    duplicate_blocked: bool = False,
    duplicate_key: Any = None,
    context_indices: Sequence[int] | None = None,
    dormant_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Return Route-C plateau acquisition score components for one candidate.

    Plateau acquisition intentionally ranks by geometry/cost only:

        S_plat = DeltaV_log / (1 + K3)             [log_volume_v1]
        S_plat = N3_plat / (1 + K3)                [compatibility]

    It must not multiply by immediate trust-region gain or gradient lower bounds,
    because the branch is designed for a flat/unresolved Phase-III gain surface.
    The helper still honors real feasibility gates such as stage/leakage/compile
    gates, leakage caps, malformed telemetry, and duplicate blocking.
    """

    score_key = normalize_plateau_acquisition_score(acquisition_score)
    formula = plateau_score_formula(score_key)
    hardware_payload = _hardware_cost_denominator_payload(feat, cfg)
    denominator = float(hardware_payload["hardware_cost_denominator"])
    cost_factor = float(hardware_payload["hardware_cost_score_factor"])
    novelty = _phase3_plateau_novelty_value(feat, plateau_novelty)
    geometry_payload = _phase3_plateau_log_volume_geometry_components(
        F_raw=feat.F_raw if F_raw is None else F_raw,
        Q_window=Q_window,
        q_window=q_window,
        novelty_fallback=novelty,
        lambda_vol=float(lambda_vol),
        metric_floor=float(cfg.metric_floor),
        ridge_growth_factor=float(cfg.ridge_growth_factor),
        ridge_max_steps=int(cfg.ridge_max_steps),
    )
    base: dict[str, Any] = {
        "schema": "route_c_plateau_acquisition_score_v1",
        "score_kind": str(score_key),
        "phase3_plateau_acquisition_score_kind": str(score_key),
        "score_formula": str(formula),
        "phase3_plateau_score_formula": str(formula),
        "phase3_plateau_acquisition_score": 0.0,
        "route_c_plateau_acquisition_score": 0.0,
        "N3_plat": 0.0 if novelty is None else float(novelty),
        "n3_plat": 0.0 if novelty is None else float(novelty),
        "phase3_plateau_novelty": 0.0 if novelty is None else float(novelty),
        "K3": float(denominator - 1.0),
        "k3": float(denominator - 1.0),
        "denominator_1_plus_K3": float(denominator),
        "phase3_plateau_burden_total": float(denominator),
        "hardware_cost_denominator": float(denominator),
        "hardware_cost_excess_sum": float(hardware_payload["hardware_cost_excess_sum"]),
        "hardware_cost_lambda_source": str(hardware_payload["lambda_source"]),
        "hardware_cost_policy": str(hardware_payload["hardware_cost_policy"]),
        "hardware_cost_signed_index": float(
            hardware_payload["hardware_cost_signed_index"]
        ),
        "hardware_cost_score_factor": float(cost_factor),
        "hardware_cost_population_hash": hardware_payload[
            "hardware_cost_population_hash"
        ],
        "duplicate_key": duplicate_key,
        "duplicate_blocked": bool(duplicate_blocked),
        "context_indices": [] if context_indices is None else [int(x) for x in context_indices],
        "dormant_indices": [] if dormant_indices is None else [int(x) for x in dormant_indices],
        "phase3_plateau_sigma_min": float(sigma_min),
        "phase3_plateau_nu_min": float(nu_min),
        "phase3_plateau_volume_min": float(volume_min),
        "eligible": False,
        "block_reason": None,
        **dict(geometry_payload),
    }
    if (not bool(feat.stage_gate_open)) or (not bool(feat.leakage_gate_open)):
        return {
            **base,
            "phase3_plateau_acquisition_score": float("-inf"),
            "route_c_plateau_acquisition_score": float("-inf"),
            "block_reason": "blocked_stage_or_leakage_gate",
        }
    if not bool(feat.compile_gate_open):
        return {
            **base,
            "phase3_plateau_acquisition_score": float("-inf"),
            "route_c_plateau_acquisition_score": float("-inf"),
            "block_reason": "compile_gate_closed",
        }
    if float(feat.leakage_penalty) > float(cfg.leakage_cap):
        return {
            **base,
            "phase3_plateau_acquisition_score": float("-inf"),
            "route_c_plateau_acquisition_score": float("-inf"),
            "block_reason": "leakage_cap",
        }
    if bool(duplicate_blocked):
        return {
            **base,
            "phase3_plateau_acquisition_score": float("-inf"),
            "route_c_plateau_acquisition_score": float("-inf"),
            "block_reason": "exact_candidate_position_duplicate",
        }
    if score_key == PLATEAU_ACQUISITION_SCORE_FRACTIONAL_RESIDUAL_V1 and novelty is None:
        return {
            **base,
            "phase3_plateau_acquisition_score": float("-inf"),
            "route_c_plateau_acquisition_score": float("-inf"),
            "block_reason": "missing_or_malformed_plateau_novelty",
        }
    if score_key == PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1 and not bool(
        geometry_payload.get("phase3_plateau_geometry_available", False)
    ):
        return {
            **base,
            "phase3_plateau_acquisition_score": float("-inf"),
            "route_c_plateau_acquisition_score": float("-inf"),
            "block_reason": f"missing_or_malformed_plateau_geometry:{geometry_payload.get('phase3_plateau_geometry_error')}",
        }
    if (not math.isfinite(denominator)) or denominator <= 0.0:
        return {
            **base,
            "phase3_plateau_acquisition_score": float("-inf"),
            "route_c_plateau_acquisition_score": float("-inf"),
            "block_reason": "malformed_plateau_cost_denominator",
        }
    if score_key == PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1:
        sigma_lcb = float(geometry_payload.get("phase3_plateau_sigma_perp_lcb", 0.0))
        frac_lcb = float(geometry_payload.get("phase3_plateau_fractional_residual_lcb", 0.0))
        volume_lcb = float(geometry_payload.get("phase3_plateau_log_volume_gain_lcb", 0.0))
        if sigma_lcb < float(sigma_min):
            return {
                **base,
                "phase3_plateau_acquisition_score": float("-inf"),
                "route_c_plateau_acquisition_score": float("-inf"),
                "block_reason": "phase3_plateau_sigma_below_min",
            }
        if frac_lcb < float(nu_min):
            return {
                **base,
                "phase3_plateau_acquisition_score": float("-inf"),
                "route_c_plateau_acquisition_score": float("-inf"),
                "block_reason": "phase3_plateau_fractional_residual_below_min",
            }
        if volume_lcb < float(volume_min):
            return {
                **base,
                "phase3_plateau_acquisition_score": float("-inf"),
                "route_c_plateau_acquisition_score": float("-inf"),
                "block_reason": "phase3_plateau_volume_below_min",
            }
        score_numerator = float(volume_lcb)
    else:
        score_numerator = float(novelty)
    score = float(
        float(score_numerator)
        * cost_factor
        / max(float(denominator), float(cfg.cheap_score_eps))
    )
    return {
        **base,
        "phase3_plateau_score_numerator": float(score_numerator),
        "phase3_plateau_acquisition_score": float(score),
        "route_c_plateau_acquisition_score": float(score),
        "eligible": True,
        "block_reason": None,
    }


def attach_route_c_plateau_acquisition_payload(
    feat: CandidateFeatures,
    payload: Mapping[str, Any],
) -> CandidateFeatures:
    """Attach plateau score telemetry without changing canonical selector fields."""

    merged_payload = dict(getattr(feat, "route_c_plateau_acquisition", None) or {})
    merged_payload.update(dict(payload))
    phase_score_components = dict(getattr(feat, "phase_score_components", {}) or {})
    phase_cost_components = dict(getattr(feat, "phase_cost_components", {}) or {})
    for key in (
        "phase3_plateau_acquisition_score",
        "route_c_plateau_acquisition_score",
        "phase3_plateau_novelty",
        "N3_plat",
    ):
        if key in merged_payload:
            try:
                phase_score_components[key] = float(merged_payload[key])
            except (TypeError, ValueError):
                pass
    for key in ("phase3_plateau_burden_total", "denominator_1_plus_K3", "hardware_cost_denominator"):
        if key in merged_payload:
            try:
                phase_cost_components[key] = float(merged_payload[key])
            except (TypeError, ValueError):
                pass
    return _replace_feature(
        feat,
        route_c_plateau_acquisition=merged_payload,
        phase_score_components=phase_score_components,
        phase_cost_components=phase_cost_components,
    )


def _phase3_auxiliary_tie_break_components(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> dict[str, float]:
    motif_tie_break = float(cfg.motif_bonus_weight) * float(max(0.0, feat.motif_bonus))
    duplicate_tie_break = -float(cfg.duplicate_penalty_weight) * float(
        max(0.0, feat.phase3_duplicate_penalty)
    )
    return {
        "motif_tie_break_score": float(motif_tie_break),
        "duplicate_tie_break_score": float(duplicate_tie_break),
        "phase3_tie_break_score": float(motif_tie_break + duplicate_tie_break),
    }


def phase3_canonical_score_components(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> dict[str, Any]:
    """Return canonical Phase3 primary-score components for a candidate.

    Canonical static Phase3 scoring is physics-first:

        S3_primary = DeltaE_TR / (1 + K3)

    In this repo ``_cheap_burden_total()`` already returns the full
    ``1 + K3`` denominator, so this helper never adds another leading one.
    Ordinary novelty multipliers and their gamma schedules are retired.
    Auxiliary motif/duplicate terms are reported separately and only affect the
    returned selector score when ``auxiliary_score_mode='ablation_additive'``.
    """
    curvature_policy = normalize_phase2_curvature_policy(
        getattr(
            cfg,
            "phase2_curvature_policy",
            PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1,
        )
    )
    cheap_curvature_proxy_policy = normalize_phase2_cheap_curvature_proxy_policy(
        getattr(
            cfg,
            "phase2_cheap_curvature_proxy_policy",
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY_LEGACY_LAMBDA_F_RATIO_V1,
        )
    )
    validated_phase2_curvature: float | None = None
    if curvature_policy == PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1:
        validated_phase2_curvature = validate_phase2_feature_curvature(feat, cfg)
    auxiliary_mode = normalize_phase3_auxiliary_score_mode(
        getattr(cfg, "auxiliary_score_mode", PHASE3_AUXILIARY_SCORE_TIE_BREAK_ONLY)
    )
    score_formula = PHASE3_CANONICAL_SCORE_FORMULA
    tie_components = _phase3_auxiliary_tie_break_components(feat, cfg)
    hardware_payload = _hardware_cost_denominator_payload(feat, cfg)
    denominator = float(hardware_payload["hardware_cost_denominator"])
    cost_factor = float(hardware_payload["hardware_cost_score_factor"])
    confidence = _phase_confidence_factor(
        float(feat.g_abs),
        float(feat.sigma_hat),
        z_alpha=float(cfg.z_alpha),
    )
    gradient_resolution = _selector_gradient_resolution(feat, cfg)
    g_hw_lcb = float(gradient_resolution.get("g_hw_lcb", 0.0))
    base: dict[str, Any] = {
        "phase2_curvature_policy": str(curvature_policy),
        "phase2_cheap_curvature_proxy_policy": str(
            cheap_curvature_proxy_policy
        ),
        "phase2_validated_directional_curvature": (
            None
            if validated_phase2_curvature is None
            else float(validated_phase2_curvature)
        ),
        "phase2_lambda_f_proxy_applied": False,
        "phase2_missing_curvature_fallback_used": False,
        "canonical_score_formula": str(score_formula),
        "phase3_canonical_score_formula": str(score_formula),
        "auxiliary_score_mode": str(auxiliary_mode),
        "phase3_auxiliary_score_mode": str(auxiliary_mode),
        "confidence_factor": float(confidence),
        "phase3_confidence_factor": float(confidence),
        "delta_e_tr": 0.0,
        "DeltaE_TR": 0.0,
        "epsilon_g_shot": float(gradient_resolution.get("epsilon_g_shot", 0.0)),
        "b_g_hw": float(gradient_resolution.get("b_g_hw", 0.0)),
        "b_g_drift": float(gradient_resolution.get("b_g_drift", 0.0)),
        "epsilon_g_res": float(gradient_resolution.get("epsilon_g_res", 0.0)),
        "g_hw_lcb": float(g_hw_lcb),
        "g_lcb_legacy_shot": float(gradient_resolution.get("g_lcb_legacy_shot", 0.0)),
        "hardware_resolution_mode": str(gradient_resolution.get("hardware_resolution_mode", "ideal")),
        "hardware_resolution_source": str(gradient_resolution.get("hardware_resolution_source", "legacy_unset")),
        "N3": None,
        "n3": None,
        "phase3_measured_novelty": None,
        "phase3_novelty_multiplier": None,
        "phase3_novelty_multiplier_policy": (
            ORDINARY_NOVELTY_SCORING_RETIRED_V1
        ),
        "phase3_gram_novelty_policy": (
            GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1
            if _deferred_gram_fallback_enabled(cfg)
            else "off"
        ),
        "phase3_novelty_status": (
            GRAM_NOVELTY_STATUS_NOT_COMPUTED_FOR_ORDINARY_SCORING
        ),
        "phase3_novelty_query_charge": 0,
        "phase3_novelty_classical_solve_count": 0,
        "phase3_novelty_applied": False,
        "K3": float(denominator - 1.0),
        "k3": float(denominator - 1.0),
        "denominator_1_plus_K3": float(denominator),
        "phase3_denominator_1_plus_K3": float(denominator),
        "hardware_cost_excess_sum": float(hardware_payload["hardware_cost_excess_sum"]),
        "hardware_cost_denominator": float(denominator),
        "hardware_cost_lambda_2q": float(hardware_payload["lambdas"]["2q"]),
        "hardware_cost_lambda_d": float(hardware_payload["lambdas"]["d"]),
        "hardware_cost_lambda_1q": float(hardware_payload["lambdas"]["1q"]),
        "hardware_cost_lambda_theta": float(hardware_payload["lambdas"]["theta"]),
        "hardware_cost_lambda_shot": float(hardware_payload["lambdas"]["shot"]),
        "hardware_cost_lambda_source": str(hardware_payload["lambda_source"]),
        "hardware_cost_policy": str(hardware_payload["hardware_cost_policy"]),
        "hardware_cost_signed_index": float(
            hardware_payload["hardware_cost_signed_index"]
        ),
        "hardware_cost_score_factor": float(cost_factor),
        "hardware_cost_population_hash": hardware_payload[
            "hardware_cost_population_hash"
        ],
        "c_bar_2q": float(hardware_payload["bars"]["2q"]),
        "c_bar_d": float(hardware_payload["bars"]["d"]),
        "c_bar_1q": float(hardware_payload["bars"]["1q"]),
        "c_bar_theta": float(hardware_payload["bars"]["theta"]),
        "c_bar_shot": float(hardware_payload["bars"]["shot"]),
        "leakage_factor": 0.0,
        "phase3_leakage_factor": 0.0,
        "phase3_primary_score": 0.0,
        "primary_score": 0.0,
        "phase3_tie_break_score": float(tie_components["phase3_tie_break_score"]),
        "tie_break_score": float(tie_components["phase3_tie_break_score"]),
        "motif_tie_break_score": float(tie_components["motif_tie_break_score"]),
        "duplicate_tie_break_score": float(tie_components["duplicate_tie_break_score"]),
        "fallback_mode": "unknown",
        "eligible": False,
        "block_reason": None,
    }
    if (not bool(feat.stage_gate_open)) or (not bool(feat.leakage_gate_open)):
        return {
            **base,
            "phase3_primary_score": float("-inf"),
            "primary_score": float("-inf"),
            "fallback_mode": "blocked_stage_or_leakage_gate",
            "block_reason": "blocked_stage_or_leakage_gate",
        }
    if not bool(feat.compile_gate_open):
        return {
            **base,
            "phase3_primary_score": float("-inf"),
            "primary_score": float("-inf"),
            "fallback_mode": "compile_gate_closed",
            "block_reason": "compile_gate_closed",
        }
    if float(feat.leakage_penalty) > float(cfg.leakage_cap):
        return {
            **base,
            "phase3_primary_score": float("-inf"),
            "primary_score": float("-inf"),
            "fallback_mode": "leakage_cap",
            "block_reason": "leakage_cap",
        }

    if g_hw_lcb <= 0.0:
        return {
            **base,
            "fallback_mode": "nonpositive_gradient",
            "block_reason": "nonpositive_gradient",
        }

    window_relaxation_disabled = _phase3_window_relaxation_disabled(cfg)
    F_red_raw = (
        (feat.F_raw if feat.F_raw is not None else feat.F_metric)
        if window_relaxation_disabled
        else (feat.F_red if feat.F_red is not None else feat.F_raw)
    )
    if F_red_raw is None:
        if float(feat.F_metric) <= 0.0:
            return {
                **base,
                "fallback_mode": "nonpositive_metric",
                "block_reason": "nonpositive_metric",
            }
        F_red = float(max(float(feat.F_metric), float(cfg.metric_floor)))
        h_eff = float(
            max(
                0.0,
                _feature_curvature_or_legacy_lambda_f_proxy(
                    feat,
                    cfg,
                    legacy_metric=float(feat.F_metric),
                ),
            )
        )
        fallback_mode = "legacy_metric_path"
    else:
        h_eff = float(
            _feature_curvature_or_legacy_lambda_f_proxy(
                feat,
                cfg,
                legacy_metric=float(F_red_raw),
            )
            if window_relaxation_disabled
            else (
                feat.h_eff
                if feat.h_eff is not None
                else (feat.h_hat if feat.h_hat is not None else 0.0)
            )
        )
        F_red = float(max(float(F_red_raw), float(cfg.metric_floor)))
        metric_collapse = str(feat.curvature_mode).startswith("append_exact_metric_collapse")
        if metric_collapse and not window_relaxation_disabled:
            return {
                **base,
                "fallback_mode": "reduced_metric_collapse",
                "block_reason": "reduced_metric_collapse",
            }
        if window_relaxation_disabled:
            fallback_mode = "phase3_window_relaxation_disabled"
        else:
            fallback_mode = (
                "append_exact_reduced_path_ridge_grown"
                if feat.ridge_used is not None and float(feat.ridge_used) > float(max(cfg.lambda_H, 0.0))
                else (
                    "append_exact_empty_window"
                    if len(_feature_phase3_schur_window(feat)) == 0
                    else "append_exact_reduced_path"
                )
            )

    delta_e = float(trust_region_drop(g_hw_lcb, float(max(0.0, h_eff)), F_red, float(cfg.rho)))
    if delta_e <= 0.0:
        return {
            **base,
            "fallback_mode": str(fallback_mode),
            "block_reason": "nonpositive_delta_e_tr",
        }

    leakage_factor = float(math.exp(-float(cfg.eta_L) * float(feat.leakage_penalty)))
    primary_score = float(
        float(delta_e)
        * cost_factor
        / max(float(denominator), float(cfg.cheap_score_eps))
    )
    return {
        **base,
        "delta_e_tr": float(delta_e),
        "DeltaE_TR": float(delta_e),
        "g_hw_lcb": float(g_hw_lcb),
        "N3": None,
        "n3": None,
        "phase3_measured_novelty": None,
        "phase3_novelty_multiplier": None,
        "phase3_novelty_multiplier_policy": (
            ORDINARY_NOVELTY_SCORING_RETIRED_V1
        ),
        "phase3_gram_novelty_policy": (
            GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1
            if _deferred_gram_fallback_enabled(cfg)
            else "off"
        ),
        "phase3_novelty_status": (
            GRAM_NOVELTY_STATUS_NOT_COMPUTED_FOR_ORDINARY_SCORING
        ),
        "phase3_novelty_query_charge": 0,
        "phase3_novelty_classical_solve_count": 0,
        "phase3_novelty_applied": False,
        "leakage_factor": float(leakage_factor),
        "phase3_leakage_factor": float(leakage_factor),
        "phase3_primary_score": float(primary_score),
        "primary_score": float(primary_score),
        "fallback_mode": str(fallback_mode),
        "eligible": True,
        "block_reason": None,
    }


def _full_v2_score_from_components(components: Mapping[str, Any]) -> float:
    primary_score = float(components.get("phase3_primary_score", 0.0))
    auxiliary_mode = normalize_phase3_auxiliary_score_mode(
        components.get("phase3_auxiliary_score_mode", components.get("auxiliary_score_mode"))
    )
    if (
        str(auxiliary_mode) == PHASE3_AUXILIARY_SCORE_ABLATION_ADDITIVE
        and math.isfinite(float(primary_score))
    ):
        primary_score = float(primary_score) + float(components.get("phase3_tie_break_score", 0.0))
    return float(primary_score)


def full_v2_score(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> tuple[float, str]:
    components = phase3_canonical_score_components(feat, cfg)
    return float(_full_v2_score_from_components(components)), str(components.get("fallback_mode", "unknown"))


def rescore_candidate_feature(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
    *,
    selector_geometry_mode: str | None = None,
) -> dict[str, Any]:
    """Recompute selector-facing scores from a saved candidate snapshot.

    Built-in math expression:
    selector = full_v2(feat, cfg) or raw_exact(feat, cfg)

    This helper is intentionally snapshot-only: it never rebuilds tangent
    geometry from statevectors. It only reuses the persisted scalar fields on
    ``CandidateFeatures`` plus the supplied ``FullScoreConfig``.
    """
    canonical_components = phase3_canonical_score_components(feat, cfg)
    full_score = float(_full_v2_score_from_components(canonical_components))
    fallback_mode = str(canonical_components.get("fallback_mode", "unknown"))
    phase3_burden_total = float(canonical_components.get("denominator_1_plus_K3", _cheap_burden_total(feat, cfg)))
    cost_factor = float(canonical_components.get("hardware_cost_score_factor", 1.0))

    selector_mode = (
        str(selector_geometry_mode).strip().lower()
        if selector_geometry_mode is not None
        else str(getattr(cfg, "phase3_selector_geometry_mode", "reduced")).strip().lower()
    )
    if selector_mode not in {"reduced", "raw_exact"}:
        selector_mode = "reduced"

    confidence = (
        float(feat.confidence_factor)
        if feat.confidence_factor is not None
        else _phase_confidence_factor(
            float(feat.g_abs),
            float(feat.sigma_hat),
            z_alpha=float(cfg.z_alpha),
        )
    )
    raw_burden_total = (
        float(feat.phase2_burden_total)
        if feat.phase2_burden_total is not None
        else float(_cheap_burden_total(feat, cfg))
    )
    raw_score_available = bool(feat.phase2_raw_score is not None)
    raw_score_recomputed = False
    raw_score_value = (
        float(feat.phase2_raw_score)
        if feat.phase2_raw_score is not None
        else None
    )
    raw_trust_gain_for_cfg: float | None = None
    stored_phase2_formula = str(getattr(feat, "phase2_raw_score_formula", "") or "").strip()
    stored_formula_is_stale = bool(
        stored_phase2_formula == ""
        or "confidence_factor" in stored_phase2_formula
        or stored_phase2_formula
        not in {
            PHASE2_CANONICAL_RAW_SCORE_FORMULA,
            PHASE2_NO_NOVELTY_RAW_SCORE_FORMULA,
            PHASE2_NO_NOVELTY_UNIT_GAIN_RAW_SCORE_FORMULA,
            HISTORICAL_SINGLETON_PHASE2_WHITENED_NO_N2_SCORE_FORMULA,
        }
    )
    feature_lacks_unit1a_resolution = not _feature_has_unit1a_resolution(feat)
    selector_policy_indicates_hw = bool(
        str(getattr(feat, "phase3_selector_policy", "")).strip().lower() == "hardware_resolvable_v1"
        or str(getattr(feat, "phase3_score_policy", "")).strip().lower() == "hardware_resolvable_v1"
    )
    raw_resolution_cfg_forces_recompute = bool(
        str(getattr(cfg, "hardware_resolution_mode", "ideal")).strip().lower() != "ideal"
        or float(getattr(cfg, "manual_b_g_hw", 0.0)) != 0.0
        or float(getattr(cfg, "manual_b_g_drift", 0.0)) != 0.0
        or stored_formula_is_stale
        or feature_lacks_unit1a_resolution
        or selector_policy_indicates_hw
    )
    if raw_resolution_cfg_forces_recompute:
        raw_score_value = None
        raw_score_available = False
    if raw_resolution_cfg_forces_recompute and feat.phase2_raw_F_effective is not None:
        curvature_policy = normalize_phase2_curvature_policy(
            getattr(
                cfg,
                "phase2_curvature_policy",
                PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1,
            )
        )
        F_raw_eff = (
            float(max(float(feat.phase2_raw_F_effective), 0.0))
            if curvature_policy
            == PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1
            else float(
                max(
                    float(feat.phase2_raw_F_effective),
                    float(cfg.metric_floor),
                    0.0,
                )
            )
        )
        h_raw_eff = _feature_curvature_or_legacy_lambda_f_proxy(
            feat,
            cfg,
            legacy_metric=float(F_raw_eff),
        )
        raw_trust_gain_for_cfg = float(
            trust_region_drop(
                float(_selector_gradient_lcb(feat, cfg)),
                float(max(0.0, h_raw_eff)),
                float(F_raw_eff),
                float(cfg.rho),
            )
        )
    if raw_trust_gain_for_cfg is not None:
        raw_score_value = float(
            float(raw_trust_gain_for_cfg)
            * cost_factor
            / max(raw_burden_total, float(cfg.cheap_score_eps))
        )
        raw_score_available = True
        raw_score_recomputed = True
    elif (
        raw_score_value is None
        and feat.phase2_raw_trust_gain is not None
    ):
        raw_score_value = float(
            float(feat.phase2_raw_trust_gain)
            * cost_factor
            / max(raw_burden_total, float(cfg.cheap_score_eps))
        )
        raw_score_available = True
        raw_score_recomputed = True

    selector_score = float(full_score)
    selector_burden = float(phase3_burden_total)
    if selector_mode == "raw_exact":
        selector_score = float(raw_score_value) if raw_score_value is not None else 0.0
        selector_burden = float(raw_burden_total if raw_score_available else phase3_burden_total)

    return {
        "full_v2_score": float(full_score),
        "full_v2_fallback_mode": str(fallback_mode),
        "phase3_burden_total": float(phase3_burden_total),
        "phase3_primary_score": float(canonical_components.get("phase3_primary_score", full_score)),
        "phase3_tie_break_score": float(canonical_components.get("phase3_tie_break_score", 0.0)),
        "phase3_auxiliary_score_mode": str(
            canonical_components.get("phase3_auxiliary_score_mode", PHASE3_AUXILIARY_SCORE_TIE_BREAK_ONLY)
        ),
        "phase3_canonical_score_formula": str(
            canonical_components.get(
                "phase3_canonical_score_formula",
                PHASE3_CANONICAL_SCORE_FORMULA,
            )
        ),
        "phase2_raw_score": (None if raw_score_value is None else float(raw_score_value)),
        "phase2_raw_trust_gain": (
            None if raw_trust_gain_for_cfg is None else float(raw_trust_gain_for_cfg)
        ),
        "phase2_g_hw_lcb": float(_selector_gradient_lcb(feat, cfg)),
        "phase2_raw_available": bool(raw_score_available),
        "phase2_raw_recomputed": bool(raw_score_recomputed),
        "phase2_burden_total": float(raw_burden_total),
        "hardware_cost_policy": str(
            canonical_components.get("hardware_cost_policy", "unresolved")
        ),
        "hardware_cost_signed_index": float(
            canonical_components.get("hardware_cost_signed_index", 0.0)
        ),
        "hardware_cost_score_factor": float(cost_factor),
        "hardware_cost_population_hash": canonical_components.get(
            "hardware_cost_population_hash"
        ),
        "selector_score": float(selector_score),
        "selector_burden": float(selector_burden),
        "selector_geometry_mode": str(selector_mode),
        "confidence_factor": float(confidence),
    }


def shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    score_key: str = "simple_score",
    tie_break_score_key: str | None = "simple_score",
) -> list[dict[str, Any]]:
    def _record_score(rec: Mapping[str, Any], key: str | None, default: float = float("-inf")) -> float:
        if key is None:
            return 0.0
        raw = rec.get(key, default)
        if raw is None:
            return float(default)
        return float(raw)

    ranked = sorted(
        [dict(rec) for rec in records],
        key=lambda rec: (
            -_record_score(rec, score_key),
            -_record_score(rec, tie_break_score_key),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        ),
    )
    if not ranked:
        return []
    total = int(len(ranked))
    target = int(max(1, min(total, cfg.shortlist_size, math.ceil(float(cfg.shortlist_fraction) * total))))
    out: list[dict[str, Any]] = []
    for idx, rec in enumerate(ranked[:target], start=1):
        updated = dict(rec)
        feat = updated.get("feature", None)
        if isinstance(feat, CandidateFeatures):
            updated["feature"] = _replace_feature(
                feat,
                shortlist_rank=int(idx),
                shortlist_size=int(target),
            )
        out.append(updated)
    return out


def _compiled_for_label(
    *,
    label: str,
    polynomial: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None,
    pauli_action_cache: dict[str, Any] | None,
) -> CompiledPolynomialAction:
    cache = compiled_cache if compiled_cache is not None else {}
    key = str(label)
    compiled = cache.get(key)
    if compiled is None:
        compiled = compile_polynomial_action(
            polynomial,
            tol=1e-12,
            pauli_action_cache=pauli_action_cache,
        )
        cache[key] = compiled
    return compiled


def _tangent_data(
    *,
    psi_state: np.ndarray,
    label: str,
    polynomial: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None,
    pauli_action_cache: dict[str, Any] | None,
) -> tuple[np.ndarray, float]:
    psi = np.asarray(psi_state, dtype=complex).reshape(-1)
    compiled = _compiled_for_label(
        label=str(label),
        polynomial=polynomial,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
    )
    apsi = apply_compiled_polynomial(psi, compiled)
    mean = complex(np.vdot(psi, apsi))
    centered = np.asarray(apsi - mean * psi, dtype=complex)
    F = float(max(0.0, np.real(np.vdot(centered, centered))))
    return centered, F


def raw_f_metric_from_state(
    *,
    psi_state: np.ndarray,
    candidate_label: str,
    candidate_term: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
) -> float:
    """Built-in math expression:
    F = ||(A - <A>) psi||^2
    """
    _tangent, F_metric = _tangent_data(
        psi_state=psi_state,
        label=str(candidate_label),
        polynomial=candidate_term.polynomial,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
    )
    return float(F_metric)


def _tangent_overlap_matrix(tangents: Sequence[np.ndarray]) -> np.ndarray:
    n = int(len(tangents))
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = float(np.real(np.vdot(tangents[i], tangents[j])))
            out[i, j] = val
            out[j, i] = val
    return out


def _executor_for_terms(
    terms: Sequence[Any],
    *,
    pauli_action_cache: dict[str, Any] | None,
) -> CompiledAnsatzExecutor:
    return CompiledAnsatzExecutor(
        list(terms),
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        pauli_action_cache=pauli_action_cache,
    )


def _rotation_triplet(vec: np.ndarray, step: Any, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vec_arr = np.asarray(vec, dtype=complex).reshape(-1)
    coeff = float(step.coeff_real)
    pvec = apply_compiled_pauli(vec_arr, step.action)
    phi = float(theta) * coeff
    c = math.cos(phi)
    s = math.sin(phi)
    u_vec = c * vec_arr - 1j * s * pvec
    d_vec = -coeff * s * vec_arr - 1j * coeff * c * pvec
    s_vec = -(coeff * coeff) * u_vec
    return np.asarray(u_vec, dtype=complex), np.asarray(d_vec, dtype=complex), np.asarray(s_vec, dtype=complex)


def _grouped_exact_plan_triplet(
    executor: CompiledAnsatzExecutor,
    plan: Any,
    vec: np.ndarray,
    theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a grouped exponential and its first two logical derivatives."""

    if str(getattr(plan, "execution_mode", "")).strip().lower() != "grouped_exact":
        raise ValueError("grouped-exact plan triplet received a termwise plan.")
    evolved = executor._apply_plan(  # noqa: SLF001 - shared compiled-plan kernel
        np.asarray(vec, dtype=complex).reshape(-1),
        plan,
        float(theta),
    )
    h_evolved = executor._apply_generator_hamiltonian(  # noqa: SLF001
        evolved,
        plan,
    )
    first = -1.0j * np.asarray(h_evolved, dtype=complex)
    second = -executor._apply_generator_hamiltonian(  # noqa: SLF001
        h_evolved,
        plan,
    )
    return (
        np.asarray(evolved, dtype=complex),
        np.asarray(first, dtype=complex),
        np.asarray(second, dtype=complex),
    )


def _horizontal_tangent(psi_state: np.ndarray, dpsi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi_state, dtype=complex).reshape(-1)
    dpsi_vec = np.asarray(dpsi, dtype=complex).reshape(-1)
    overlap = complex(np.vdot(psi, dpsi_vec))
    return np.asarray(dpsi_vec - overlap * psi, dtype=complex)


def _energy_hessian_entry(
    *,
    dpsi_left: np.ndarray,
    dpsi_right: np.ndarray,
    d2psi: np.ndarray,
    hpsi_state: np.ndarray,
    hdpsi_right: np.ndarray,
) -> float:
    return float(
        2.0
        * np.real(
            np.vdot(np.asarray(d2psi, dtype=complex), np.asarray(hpsi_state, dtype=complex))
            + np.vdot(np.asarray(dpsi_left, dtype=complex), np.asarray(hdpsi_right, dtype=complex))
        )
    )


@dataclass(frozen=True, slots=True)
class _ExactInsertionFirstOrderContext:
    prefix_states: tuple[np.ndarray, ...]
    suffix_adjoint_hpsi: tuple[np.ndarray, ...]
    selected_operator_count: int
    state_reconstruction_delta_norm: float


def _apply_executor_plan_adjoint(
    *,
    executor: CompiledAnsatzExecutor,
    plan: Any,
    theta: float,
    vector: np.ndarray,
) -> np.ndarray:
    if str(getattr(plan, "execution_mode", "")).strip().lower() == "grouped_exact":
        return np.asarray(
            executor._apply_plan(  # noqa: SLF001 - shared compiled-plan kernel
                np.asarray(vector, dtype=complex).reshape(-1),
                plan,
                -float(theta),
            ),
            dtype=complex,
        )
    value = np.asarray(vector, dtype=complex).reshape(-1)
    for step in reversed(tuple(getattr(plan, "steps", ()))):
        value = _rotation_triplet(value, step, -float(theta))[0]
    return np.asarray(value, dtype=complex)


def _prepare_exact_insertion_first_order_context(
    *,
    selected_ops: Sequence[Any],
    theta: Sequence[float] | np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    hpsi_state: np.ndarray,
    pauli_action_cache: dict[str, Any] | None,
    state_consistency_tolerance: float,
) -> _ExactInsertionFirstOrderContext:
    """Prepare position-resolved first-order geometry for one accepted state."""

    selected = list(selected_ops)
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    if int(theta_vec.size) != int(len(selected)):
        raise ValueError(
            "Exact insertion first-order context requires one logical "
            "coordinate per selected generator."
        )
    executor = _executor_for_terms(
        selected,
        pauli_action_cache=pauli_action_cache,
    )
    plans = list(getattr(executor, "_plans", ()))
    if len(plans) != len(selected):
        raise ValueError(
            "Exact insertion first-order context received an executor whose "
            "logical plan count differs from the selected ansatz."
        )

    prefixes = [
        np.asarray(psi_ref, dtype=complex).reshape(-1).copy()
    ]
    for index, plan in enumerate(plans):
        prefixes.append(
            np.asarray(
                executor._apply_plan(  # noqa: SLF001 - shared compiled-plan kernel
                    prefixes[-1],
                    plan,
                    float(theta_vec[index]),
                ),
                dtype=complex,
            )
        )
    supplied_state = np.asarray(psi_state, dtype=complex).reshape(-1)
    reconstructed_state = np.asarray(prefixes[-1], dtype=complex).reshape(-1)
    overlap = complex(np.vdot(supplied_state, reconstructed_state))
    phase = overlap / abs(overlap) if abs(overlap) > 0.0 else 1.0 + 0.0j
    state_delta = float(
        np.linalg.norm(reconstructed_state / phase - supplied_state)
    )
    tolerance = float(max(1e-12, state_consistency_tolerance))
    if state_delta > tolerance:
        raise ValueError(
            "Exact insertion first-order context reconstructed a state "
            "inconsistent with the current branch state: "
            f"delta={state_delta:.6g}, tolerance={tolerance:.6g}."
        )
    aligned_prefixes = tuple(
        np.asarray(value / phase, dtype=complex)
        for value in prefixes
    )

    suffix_adjoint: list[np.ndarray | None] = [
        None for _ in range(len(selected) + 1)
    ]
    suffix_adjoint[-1] = np.asarray(
        hpsi_state,
        dtype=complex,
    ).reshape(-1)
    for index in range(len(selected) - 1, -1, -1):
        suffix_adjoint[index] = _apply_executor_plan_adjoint(
            executor=executor,
            plan=plans[index],
            theta=float(theta_vec[index]),
            vector=np.asarray(suffix_adjoint[index + 1], dtype=complex),
        )
    return _ExactInsertionFirstOrderContext(
        prefix_states=aligned_prefixes,
        suffix_adjoint_hpsi=tuple(
            np.asarray(value, dtype=complex)
            for value in suffix_adjoint
        ),
        selected_operator_count=int(len(selected)),
        state_reconstruction_delta_norm=float(state_delta),
    )


def _exact_insertion_first_order_candidate_geometry(
    *,
    context: _ExactInsertionFirstOrderContext,
    candidate_term: Any,
    position_id: int,
    candidate_compiled: CompiledPolynomialAction | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact gradient and Gram diagonal at one insertion position."""

    position = int(position_id)
    if position < 0 or position > int(context.selected_operator_count):
        raise ValueError(
            "Exact insertion first-order geometry received a position outside "
            "the current ordered ansatz."
        )
    compiled = candidate_compiled
    if compiled is None:
        compiled = compile_polynomial_action(
            candidate_term.polynomial,
            pauli_action_cache=pauli_action_cache,
        )
    prefix_state = np.asarray(
        context.prefix_states[position],
        dtype=complex,
    )
    candidate_action = apply_compiled_polynomial(prefix_state, compiled)
    derivative = np.asarray(-1.0j * candidate_action, dtype=complex)
    energy_gradient = float(
        2.0
        * np.real(
            np.vdot(
                derivative,
                np.asarray(
                    context.suffix_adjoint_hpsi[position],
                    dtype=complex,
                ),
            )
        )
    )
    tangent = _horizontal_tangent(prefix_state, derivative)
    metric = float(
        max(0.0, np.real(np.vdot(tangent, tangent)))
    )
    return {
        "schema": "exact_insertion_first_order_candidate_geometry_v1",
        "position_id": int(position),
        "energy_gradient": float(energy_gradient),
        "fubini_study_metric": float(metric),
        "state_reconstruction_delta_norm": float(
            context.state_reconstruction_delta_norm
        ),
    }


def _propagate_executor_derivatives(
    *,
    executor: CompiledAnsatzExecutor,
    theta: np.ndarray,
    psi_ref: np.ndarray,
    active_indices: Sequence[int],
) -> tuple[np.ndarray, list[np.ndarray], list[list[np.ndarray]]]:
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    active = [int(i) for i in active_indices]
    psi = np.asarray(psi_ref, dtype=complex).reshape(-1).copy()
    n_active = int(len(active))
    dpsi = [np.zeros_like(psi, dtype=complex) for _ in range(n_active)]
    d2psi = [[np.zeros_like(psi, dtype=complex) for _ in range(n_active)] for __ in range(n_active)]
    if n_active == 0:
        return executor.prepare_state(theta_vec, psi), dpsi, d2psi

    active_map = {int(global_idx): int(local_idx) for local_idx, global_idx in enumerate(active)}
    plans = list(getattr(executor, "_plans", []))
    if len(plans) != int(theta_vec.size):
        raise ValueError(f"theta length mismatch: got {theta_vec.size}, expected {len(plans)}.")

    for global_idx, plan in enumerate(plans):
        theta_k = float(theta_vec[global_idx])
        local = active_map.get(int(global_idx), None)
        if str(getattr(plan, "execution_mode", "")).strip().lower() == "grouped_exact":
            old_psi = psi
            old_dpsi = dpsi
            old_d2psi = d2psi

            psi_u, psi_d, psi_s = _grouped_exact_plan_triplet(
                executor,
                plan,
                old_psi,
                theta_k,
            )
            psi = psi_u
            next_dpsi: list[np.ndarray] = []
            d_old: list[np.ndarray] = []
            for idx in range(n_active):
                vec_u, vec_d, _vec_s = _grouped_exact_plan_triplet(
                    executor,
                    plan,
                    old_dpsi[idx],
                    theta_k,
                )
                next_dpsi.append(vec_u)
                d_old.append(vec_d)
            if local is not None:
                next_dpsi[int(local)] = np.asarray(
                    next_dpsi[int(local)] + psi_d,
                    dtype=complex,
                )

            next_d2psi: list[list[np.ndarray]] = [
                [np.zeros_like(psi, dtype=complex) for _ in range(n_active)]
                for __ in range(n_active)
            ]
            for row in range(n_active):
                for col in range(n_active):
                    vec_u, _vec_d, _vec_s = _grouped_exact_plan_triplet(
                        executor,
                        plan,
                        old_d2psi[row][col],
                        theta_k,
                    )
                    updated = vec_u
                    if local is not None:
                        if row == int(local):
                            updated = np.asarray(
                                updated + d_old[col],
                                dtype=complex,
                            )
                        if col == int(local):
                            updated = np.asarray(
                                updated + d_old[row],
                                dtype=complex,
                            )
                        if row == int(local) and col == int(local):
                            updated = np.asarray(updated + psi_s, dtype=complex)
                    next_d2psi[row][col] = np.asarray(updated, dtype=complex)
            dpsi = next_dpsi
            d2psi = next_d2psi
            continue
        for step in getattr(plan, "steps", ()):
            old_psi = psi
            old_dpsi = dpsi
            old_d2psi = d2psi

            psi_u, psi_d, psi_s = _rotation_triplet(old_psi, step, theta_k)
            psi = psi_u

            next_dpsi: list[np.ndarray] = []
            d_old: list[np.ndarray] = []
            for idx in range(n_active):
                vec_u, vec_d, _vec_s = _rotation_triplet(old_dpsi[idx], step, theta_k)
                next_dpsi.append(vec_u)
                d_old.append(vec_d)
            if local is not None:
                next_dpsi[int(local)] = np.asarray(next_dpsi[int(local)] + psi_d, dtype=complex)

            next_d2psi: list[list[np.ndarray]] = [
                [np.zeros_like(psi, dtype=complex) for _ in range(n_active)]
                for __ in range(n_active)
            ]
            for row in range(n_active):
                for col in range(n_active):
                    vec_u, _vec_d, _vec_s = _rotation_triplet(old_d2psi[row][col], step, theta_k)
                    updated = vec_u
                    if local is not None:
                        if row == int(local):
                            updated = np.asarray(updated + d_old[col], dtype=complex)
                        if col == int(local):
                            updated = np.asarray(updated + d_old[row], dtype=complex)
                        if row == int(local) and col == int(local):
                            updated = np.asarray(updated + psi_s, dtype=complex)
                    next_d2psi[row][col] = np.asarray(updated, dtype=complex)

            dpsi = next_dpsi
            d2psi = next_d2psi

    return np.asarray(psi, dtype=complex), dpsi, d2psi


def _propagate_executor_sparse_second_derivatives(
    *,
    executor: CompiledAnsatzExecutor,
    theta: np.ndarray,
    psi_ref: np.ndarray,
    active_indices: Sequence[int],
    second_derivative_pairs: Sequence[tuple[int, int]],
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    dict[tuple[int, int], np.ndarray],
]:
    """Propagate all requested first derivatives and only selected Hessian pairs."""

    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    active = [int(index) for index in active_indices]
    n_active = int(len(active))
    normalized_pairs: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for raw_left, raw_right in second_derivative_pairs:
        left = int(raw_left)
        right = int(raw_right)
        if left < 0 or right < 0 or left >= n_active or right >= n_active:
            raise ValueError("Sparse second-derivative pair index is out of range.")
        pair = (min(left, right), max(left, right))
        if pair not in seen_pairs:
            normalized_pairs.append(pair)
            seen_pairs.add(pair)

    psi = np.asarray(psi_ref, dtype=complex).reshape(-1).copy()
    dpsi = [np.zeros_like(psi, dtype=complex) for _ in range(n_active)]
    d2psi = {
        pair: np.zeros_like(psi, dtype=complex) for pair in normalized_pairs
    }
    if n_active == 0:
        return executor.prepare_state(theta_vec, psi), dpsi, d2psi

    active_map = {
        int(global_index): int(local_index)
        for local_index, global_index in enumerate(active)
    }
    plans = list(getattr(executor, "_plans", []))
    if len(plans) != int(theta_vec.size):
        raise ValueError(
            f"theta length mismatch: got {theta_vec.size}, expected {len(plans)}."
        )

    for global_index, plan in enumerate(plans):
        theta_k = float(theta_vec[global_index])
        local = active_map.get(int(global_index))
        if str(getattr(plan, "execution_mode", "")).strip().lower() == "grouped_exact":
            old_psi = psi
            old_dpsi = dpsi
            old_d2psi = d2psi
            psi_u, psi_d, psi_s = _grouped_exact_plan_triplet(
                executor,
                plan,
                old_psi,
                theta_k,
            )
            psi = psi_u

            next_dpsi: list[np.ndarray] = []
            d_old: list[np.ndarray] = []
            for local_index in range(n_active):
                vec_u, vec_d, _vec_s = _grouped_exact_plan_triplet(
                    executor,
                    plan,
                    old_dpsi[local_index],
                    theta_k,
                )
                next_dpsi.append(vec_u)
                d_old.append(vec_d)
            if local is not None:
                next_dpsi[int(local)] = np.asarray(
                    next_dpsi[int(local)] + psi_d,
                    dtype=complex,
                )

            next_d2psi: dict[tuple[int, int], np.ndarray] = {}
            for pair in normalized_pairs:
                left, right = pair
                vec_u, _vec_d, _vec_s = _grouped_exact_plan_triplet(
                    executor,
                    plan,
                    old_d2psi[pair],
                    theta_k,
                )
                updated = vec_u
                if local is not None:
                    if left == int(local):
                        updated = np.asarray(
                            updated + d_old[right],
                            dtype=complex,
                        )
                    if right == int(local):
                        updated = np.asarray(
                            updated + d_old[left],
                            dtype=complex,
                        )
                    if left == right == int(local):
                        updated = np.asarray(updated + psi_s, dtype=complex)
                next_d2psi[pair] = np.asarray(updated, dtype=complex)
            dpsi = next_dpsi
            d2psi = next_d2psi
            continue
        for step in getattr(plan, "steps", ()):
            old_psi = psi
            old_dpsi = dpsi
            old_d2psi = d2psi
            psi_u, psi_d, psi_s = _rotation_triplet(old_psi, step, theta_k)
            psi = psi_u

            next_dpsi: list[np.ndarray] = []
            d_old: list[np.ndarray] = []
            for local_index in range(n_active):
                vec_u, vec_d, _vec_s = _rotation_triplet(
                    old_dpsi[local_index],
                    step,
                    theta_k,
                )
                next_dpsi.append(vec_u)
                d_old.append(vec_d)
            if local is not None:
                next_dpsi[int(local)] = np.asarray(
                    next_dpsi[int(local)] + psi_d,
                    dtype=complex,
                )

            next_d2psi: dict[tuple[int, int], np.ndarray] = {}
            for pair in normalized_pairs:
                left, right = pair
                vec_u, _vec_d, _vec_s = _rotation_triplet(
                    old_d2psi[pair],
                    step,
                    theta_k,
                )
                updated = vec_u
                if local is not None:
                    if left == int(local):
                        updated = np.asarray(updated + d_old[right], dtype=complex)
                    if right == int(local):
                        updated = np.asarray(updated + d_old[left], dtype=complex)
                    if left == right == int(local):
                        updated = np.asarray(updated + psi_s, dtype=complex)
                next_d2psi[pair] = np.asarray(updated, dtype=complex)
            dpsi = next_dpsi
            d2psi = next_d2psi

    return np.asarray(psi, dtype=complex), dpsi, d2psi


def _propagate_append_candidate(
    *,
    candidate_term: Any,
    psi_state: np.ndarray,
    window_dpsi: Sequence[np.ndarray],
    pauli_action_cache: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    cand_exec = _executor_for_terms([candidate_term], pauli_action_cache=pauli_action_cache)
    plan = list(getattr(cand_exec, "_plans", []))
    if not plan:
        zero = np.zeros_like(np.asarray(psi_state, dtype=complex).reshape(-1), dtype=complex)
        return zero, zero, [np.zeros_like(zero) for _ in window_dpsi]
    steps = list(getattr(plan[0], "steps", ()))
    psi = np.asarray(psi_state, dtype=complex).reshape(-1).copy()
    cand_dpsi = np.zeros_like(psi, dtype=complex)
    cand_d2psi = np.zeros_like(psi, dtype=complex)
    win_dpsi = [np.asarray(vec, dtype=complex).reshape(-1).copy() for vec in window_dpsi]
    cand_win_d2 = [np.zeros_like(psi) for _ in window_dpsi]

    for step in steps:
        old_psi = psi
        old_cand_dpsi = cand_dpsi
        old_cand_d2psi = cand_d2psi
        old_win_dpsi = win_dpsi
        old_cand_win_d2 = cand_win_d2

        psi_u, psi_d, psi_s = _rotation_triplet(old_psi, step, 0.0)
        cand_u, cand_d, _cand_s = _rotation_triplet(old_cand_dpsi, step, 0.0)
        cand2_u, _cand2_d, _cand2_s = _rotation_triplet(old_cand_d2psi, step, 0.0)

        psi = psi_u
        cand_dpsi = np.asarray(cand_u + psi_d, dtype=complex)
        cand_d2psi = np.asarray(cand2_u + cand_d + cand_d + psi_s, dtype=complex)

        next_win_dpsi: list[np.ndarray] = []
        next_cand_win_d2: list[np.ndarray] = []
        for idx, win_vec in enumerate(old_win_dpsi):
            win_u, win_d, _win_s = _rotation_triplet(win_vec, step, 0.0)
            cross_u, _cross_d, _cross_s = _rotation_triplet(old_cand_win_d2[idx], step, 0.0)
            next_win_dpsi.append(np.asarray(win_u, dtype=complex))
            next_cand_win_d2.append(np.asarray(cross_u + win_d, dtype=complex))
        win_dpsi = next_win_dpsi
        cand_win_d2 = next_cand_win_d2

    return cand_dpsi, cand_d2psi, cand_win_d2


def _regularized_solve(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    base_ridge: float,
    growth_factor: float,
    max_steps: int,
    require_pd: bool,
) -> tuple[np.ndarray, float, np.ndarray]:
    mat = np.asarray(matrix, dtype=float)
    vec = np.asarray(rhs, dtype=float).reshape(-1)
    n = int(mat.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float), float(max(base_ridge, 0.0)), np.zeros((0, 0), dtype=float)
    eye = np.eye(n, dtype=float)
    ridge = float(max(base_ridge, 0.0))
    if ridge == 0.0:
        ridge = 1e-12
    mat_sym = 0.5 * (mat + mat.T)
    for _ in range(int(max(1, max_steps))):
        trial = mat_sym + ridge * eye
        try:
            if require_pd:
                np.linalg.cholesky(trial)
            sol = np.linalg.solve(trial, vec)
            return np.asarray(sol, dtype=float), float(ridge), np.asarray(trial, dtype=float)
        except Exception:
            ridge *= float(max(growth_factor, 2.0))
    trial = mat_sym + ridge * eye
    if require_pd:
        np.linalg.cholesky(trial)
    sol = np.linalg.solve(trial, vec)
    return np.asarray(sol, dtype=float), float(ridge), np.asarray(trial, dtype=float)


class OrderedInsertionGeometryOracle:
    """Exact ordered-state geometry for an active ansatz plus one insertion.

    The module constructs scaffold tangents and emits the candidate derivative,
    metric, and Hessian inputs consumed by downstream scoring.  It does not
    choose or apply an ordinary novelty multiplier.
    """

    def prepare_scaffold_context(
        self,
        *,
        selected_ops: Sequence[Any],
        theta: np.ndarray,
        psi_ref: np.ndarray,
        psi_state: np.ndarray,
        h_compiled: CompiledPolynomialAction,
        hpsi_state: np.ndarray,
        refit_window_indices: Sequence[int],
        pauli_action_cache: dict[str, Any] | None = None,
        align_reconstructed_global_phase: bool = True,
        old_old_metric_prior: np.ndarray | None = None,
        old_old_hessian_prior: np.ndarray | None = None,
        old_old_hessian_prior_provenance: Mapping[str, Any] | None = None,
        old_old_hessian_prior_status: str | None = None,
    ) -> _ScaffoldDerivativeContext:
        """Build one active scaffold, optionally with predicted old--old blocks.

        Supplying either prior preserves the exact tangent-state construction
        needed for candidate cross/new measurements while preventing an exact
        endpoint old--old Gram or Hessian acquisition from entering the model.
        These arrays are therefore an explicit information-source boundary,
        not a post-hoc overwrite of already consumed exact geometry.
        """

        inherited_window = [int(i) for i in refit_window_indices]
        metric_prior_supplied = bool(old_old_metric_prior is not None)
        hessian_prior_supplied = bool(old_old_hessian_prior is not None)
        prior_supplied = bool(metric_prior_supplied or hessian_prior_supplied)
        hessian_provenance = _normalize_scaffold_hessian_provenance(
            old_old_hessian_prior_provenance
        )
        if not hessian_prior_supplied and (
            hessian_provenance or old_old_hessian_prior_status is not None
        ):
            raise ValueError(
                "old-old Hessian provenance requires an old-old Hessian prior."
            )
        if old_old_hessian_prior_status is not None and not isinstance(
            old_old_hessian_prior_status, str
        ):
            raise TypeError("old-old Hessian prior status must be a string.")
        hessian_status = (
            ""
            if old_old_hessian_prior_status is None
            else old_old_hessian_prior_status.strip()
        )
        if old_old_hessian_prior_status is not None and not hessian_status:
            raise ValueError("old-old Hessian prior status must be nonempty.")
        psi_current = np.asarray(psi_state, dtype=complex).reshape(-1)
        hpsi_current = np.asarray(hpsi_state, dtype=complex).reshape(-1)
        if not inherited_window:
            if hessian_prior_supplied:
                empty_hessian = np.asarray(
                    old_old_hessian_prior, dtype=float
                )
                if empty_hessian.shape != (0, 0):
                    raise ValueError(
                        "old-old Hessian prior does not match the active window."
                    )
                if not np.all(np.isfinite(empty_hessian)):
                    raise ValueError("old-old Hessian prior must be finite.")
            return _ScaffoldDerivativeContext(
                psi_state=psi_current,
                hpsi_state=hpsi_current,
                selected_ops=tuple(selected_ops),
                theta=np.asarray(theta, dtype=float).copy(),
                psi_ref=np.asarray(psi_ref, dtype=complex).copy(),
                state_fingerprint=_array_fingerprint(psi_current),
                ordered_scaffold_fingerprint=_ordered_scaffold_fingerprint(
                    selected_ops
                ),
                theta_fingerprint=_array_fingerprint(
                    np.asarray(theta, dtype=float)
                ),
                refit_window_indices=tuple(),
                dpsi_window=tuple(),
                tangents_window=tuple(),
                Q_window=np.zeros((0, 0), dtype=float),
                H_window_hessian=np.zeros((0, 0), dtype=float),
                old_old_geometry_measured=bool(not prior_supplied),
                old_old_metric_measured=bool(not metric_prior_supplied),
                old_old_hessian_measured=bool(not hessian_prior_supplied),
                old_old_hessian_status=hessian_status,
                old_old_hessian_provenance=hessian_provenance,
            )

        executor = _executor_for_terms(selected_ops, pauli_action_cache=pauli_action_cache)
        reconstructed, dpsi_window, d2psi_window = _propagate_executor_derivatives(
            executor=executor,
            theta=np.asarray(theta, dtype=float),
            psi_ref=np.asarray(psi_ref, dtype=complex),
            active_indices=inherited_window,
        )
        reconstructed_state = np.asarray(
            reconstructed, dtype=complex
        ).reshape(-1)
        overlap = complex(np.vdot(psi_current, reconstructed_state))
        phase = overlap / abs(overlap) if abs(overlap) > 0.0 else 1.0 + 0.0j
        state_reconstruction_delta_norm = float(
            np.linalg.norm(reconstructed_state / phase - psi_current)
        )
        if bool(align_reconstructed_global_phase):
            dpsi_window = [
                np.asarray(value / phase, dtype=complex) for value in dpsi_window
            ]
            d2psi_window = [
                [np.asarray(value / phase, dtype=complex) for value in row]
                for row in d2psi_window
            ]
        tangents_window = [
            _horizontal_tangent(psi_current, dpsi_vec)
            for dpsi_vec in dpsi_window
        ]
        m = int(len(inherited_window))
        if metric_prior_supplied:
            q_window = np.asarray(old_old_metric_prior, dtype=float)
            expected_shape = (m, m)
            if q_window.shape != expected_shape:
                raise ValueError(
                    "old-old metric prior does not match the active window."
                )
            if not np.all(np.isfinite(q_window)):
                raise ValueError("old-old metric prior must be finite.")
            q_window = 0.5 * (q_window + q_window.T)
        else:
            q_window = _tangent_overlap_matrix(tangents_window)
        if hessian_prior_supplied:
            hess = np.asarray(old_old_hessian_prior, dtype=float)
            expected_shape = (m, m)
            if hess.shape != expected_shape:
                raise ValueError(
                    "old-old Hessian prior does not match the active window."
                )
            if not np.all(np.isfinite(hess)):
                raise ValueError("old-old Hessian prior must be finite.")
            hess = 0.5 * (hess + hess.T)
        else:
            hdpsi_window = [
                apply_compiled_polynomial(
                    np.asarray(dpsi_vec, dtype=complex), h_compiled
                )
                for dpsi_vec in dpsi_window
            ]
            hess = np.zeros((m, m), dtype=float)
            for row in range(m):
                for col in range(m):
                    hess[row, col] = _energy_hessian_entry(
                        dpsi_left=dpsi_window[row],
                        dpsi_right=dpsi_window[col],
                        d2psi=d2psi_window[row][col],
                        hpsi_state=hpsi_current,
                        hdpsi_right=hdpsi_window[col],
                    )
            hess = 0.5 * (hess + hess.T)
        return _ScaffoldDerivativeContext(
            psi_state=psi_current,
            hpsi_state=hpsi_current,
            selected_ops=tuple(selected_ops),
            theta=np.asarray(theta, dtype=float).copy(),
            psi_ref=np.asarray(psi_ref, dtype=complex).copy(),
            state_fingerprint=_array_fingerprint(psi_current),
            ordered_scaffold_fingerprint=_ordered_scaffold_fingerprint(
                selected_ops
            ),
            theta_fingerprint=_array_fingerprint(
                np.asarray(theta, dtype=float)
            ),
            refit_window_indices=tuple(inherited_window),
            dpsi_window=tuple(np.asarray(x, dtype=complex) for x in dpsi_window),
            tangents_window=tuple(np.asarray(x, dtype=complex) for x in tangents_window),
            Q_window=np.asarray(q_window, dtype=float),
            H_window_hessian=np.asarray(hess, dtype=float),
            state_reconstruction_delta_norm=float(
                state_reconstruction_delta_norm
            ),
            old_old_geometry_measured=bool(not prior_supplied),
            old_old_metric_measured=bool(not metric_prior_supplied),
            old_old_hessian_measured=bool(not hessian_prior_supplied),
            old_old_hessian_status=hessian_status,
            old_old_hessian_provenance=hessian_provenance,
        )

    def estimate(
        self,
        *,
        scaffold_context: _ScaffoldDerivativeContext,
        candidate_label: str,
        candidate_term: Any,
        compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
        pauli_action_cache: dict[str, Any] | None = None,
        novelty_eps: float = 1e-6,
    ) -> Mapping[str, Any]:
        del compiled_cache, novelty_eps
        cand_dpsi, cand_d2psi, cand_window_d2 = _propagate_append_candidate(
            candidate_term=candidate_term,
            psi_state=scaffold_context.psi_state,
            window_dpsi=list(scaffold_context.dpsi_window),
            pauli_action_cache=pauli_action_cache,
        )
        cand_tangent = _horizontal_tangent(scaffold_context.psi_state, cand_dpsi)
        q_window = np.asarray(
            [
                float(np.real(np.vdot(tang_j, cand_tangent)))
                for tang_j in scaffold_context.tangents_window
            ],
            dtype=float,
        )
        F_raw = float(max(0.0, np.real(np.vdot(cand_tangent, cand_tangent))))
        return {
            "novelty_mode": "append_exact_tangent_context_v1",
            "candidate_dpsi": np.asarray(cand_dpsi, dtype=complex),
            "candidate_d2psi": np.asarray(cand_d2psi, dtype=complex),
            "candidate_window_d2": [np.asarray(x, dtype=complex) for x in cand_window_d2],
            "candidate_tangent": np.asarray(cand_tangent, dtype=complex),
            "F_raw": float(F_raw),
            "Q_window": np.asarray(scaffold_context.Q_window, dtype=float),
            "q_window": np.asarray(q_window, dtype=float),
            "scaffold_old_old_hessian_source": (
                _scaffold_hessian_source_telemetry(scaffold_context)
            ),
        }


# Identity-only compatibility for the out-of-scope Paper-II legacy controller.
# This alias owns no separate scoring behavior and may be removed after that
# caller migrates to OrderedInsertionGeometryOracle.
Phase2NoveltyOracle = OrderedInsertionGeometryOracle


class Phase2CurvatureConstructionError(RuntimeError):
    """A required Phase-II directional curvature could not be certified."""


def _phase2_curvature_failure(reason: str, *, candidate_label: str) -> None:
    raise Phase2CurvatureConstructionError(
        "SR-SNAKE Phase-II curvature construction failed closed for "
        f"candidate {candidate_label!r}: {reason}."
    )


_PHASE2_CURVATURE_BINDING_FIELDS = (
    "state_fingerprint",
    "ordered_scaffold_fingerprint",
    "theta_fingerprint",
    "hamiltonian_fingerprint",
    "candidate_coordinate_fingerprint",
    "candidate_position_id",
    "derivative_convention",
)


def _strict_finite_curvature_scalar(
    raw: Any,
    *,
    candidate_label: str,
    source: str,
) -> float:
    """Return a finite real scalar without coercing strings or booleans."""

    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, numbers.Real):
        _phase2_curvature_failure(
            f"{source} is malformed (expected a real numeric scalar)",
            candidate_label=candidate_label,
        )
    value = float(raw)
    if not math.isfinite(value):
        _phase2_curvature_failure(
            f"{source} is nonfinite",
            candidate_label=candidate_label,
        )
    return value


def _phase2_curvature_binding(
    *,
    scaffold_context: _ScaffoldDerivativeContext,
    h_compiled: CompiledPolynomialAction,
    candidate_term: Any,
    position_id: int,
) -> dict[str, Any]:
    return {
        "state_fingerprint": str(scaffold_context.state_fingerprint),
        "ordered_scaffold_fingerprint": str(
            scaffold_context.ordered_scaffold_fingerprint
        ),
        "theta_fingerprint": str(scaffold_context.theta_fingerprint),
        "hamiltonian_fingerprint": _compiled_polynomial_fingerprint(h_compiled),
        "candidate_coordinate_fingerprint": _candidate_coordinate_fingerprint(
            candidate_term,
            position_id=int(position_id),
        ),
        "candidate_position_id": int(position_id),
        "derivative_convention": (
            "compiled_ansatz_exact_parameter_derivatives_v1"
        ),
    }


def _validate_phase2_curvature_receipt(
    *,
    receipt_raw: Any,
    h_raw: float,
    candidate_label: str,
    expected_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(receipt_raw, Mapping):
        _phase2_curvature_failure(
            "required curvature receipt is absent or malformed",
            candidate_label=candidate_label,
        )
    receipt = dict(receipt_raw)
    if str(receipt.get("schema", "")) != PHASE2_DIRECTIONAL_CURVATURE_RECEIPT_SCHEMA:
        _phase2_curvature_failure(
            "curvature receipt schema is unresolved",
            candidate_label=candidate_label,
        )
    if str(receipt.get("status", "")) != "computed_finite":
        _phase2_curvature_failure(
            "curvature receipt does not certify a finite computation",
            candidate_label=candidate_label,
        )
    receipt_value = _strict_finite_curvature_scalar(
        receipt.get("h_raw"),
        candidate_label=candidate_label,
        source="curvature receipt value",
    )
    if receipt_value != h_raw:
        _phase2_curvature_failure(
            "curvature receipt value does not match the scored value",
            candidate_label=candidate_label,
        )
    provenance_raw = receipt.get("measurement_provenance")
    if not isinstance(provenance_raw, Mapping):
        _phase2_curvature_failure(
            "required measurement provenance is absent",
            candidate_label=candidate_label,
        )
    provenance = dict(provenance_raw)
    if (
        str(provenance.get("schema", ""))
        != PHASE2_DIRECTIONAL_CURVATURE_PROVENANCE_SCHEMA
        or provenance.get("required_primitives_resolved") is not True
        or str(provenance.get("source", "")).strip() == ""
    ):
        _phase2_curvature_failure(
            "measurement provenance is unresolved",
            candidate_label=candidate_label,
        )
    for field in _PHASE2_CURVATURE_BINDING_FIELDS:
        actual = receipt.get(field)
        if actual is None or actual == "":
            _phase2_curvature_failure(
                f"curvature receipt is missing identity binding {field}",
                candidate_label=candidate_label,
            )
        if expected_binding is not None:
            expected = expected_binding.get(field)
            try:
                if field == "candidate_position_id":
                    matches = int(actual) == int(expected)
                else:
                    matches = str(actual) == str(expected)
            except (TypeError, ValueError):
                matches = False
            if not matches:
                _phase2_curvature_failure(
                    f"curvature receipt identity binding {field} does not "
                    "match the current scoring context",
                    candidate_label=candidate_label,
                )
    return receipt


def _validated_phase2_directional_curvature(
    *,
    curvature_info: Mapping[str, Any],
    cfg: FullScoreConfig,
    candidate_label: str,
    expected_binding: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, Any] | None]:
    """Return the Phase-II directional curvature under the resolved policy.

    Historical profiles retain their optional/default behavior.  The v4
    policy requires both a finite value and an explicit receipt proving that
    the value came from the already-computed directional Hessian primitives.
    """

    policy = normalize_phase2_curvature_policy(
        getattr(
            cfg,
            "phase2_curvature_policy",
            PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1,
        )
    )
    cheap_proxy_policy = normalize_phase2_cheap_curvature_proxy_policy(
        getattr(
            cfg,
            "phase2_cheap_curvature_proxy_policy",
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY_LEGACY_LAMBDA_F_RATIO_V1,
        )
    )
    if policy != PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1:
        raw = curvature_info.get("h_raw", 0.0)
        try:
            return float(raw), None
        except (TypeError, ValueError):
            return 0.0, None

    if cheap_proxy_policy != PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF:
        _phase2_curvature_failure(
            "measured-required policy was paired with an active lambda-F proxy",
            candidate_label=candidate_label,
        )
    if "h_raw" not in curvature_info:
        _phase2_curvature_failure(
            "directional curvature value is absent",
            candidate_label=candidate_label,
        )
    raw = curvature_info.get("h_raw")
    if raw is None:
        _phase2_curvature_failure(
            "directional curvature value is None",
            candidate_label=candidate_label,
        )
    h_raw = _strict_finite_curvature_scalar(
        raw,
        candidate_label=candidate_label,
        source="directional curvature",
    )
    receipt = _validate_phase2_curvature_receipt(
        receipt_raw=curvature_info.get("phase2_curvature_receipt"),
        h_raw=float(h_raw),
        candidate_label=candidate_label,
        expected_binding=expected_binding,
    )
    return h_raw, receipt


def validate_phase2_feature_curvature(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
    *,
    expected_binding: Mapping[str, Any] | None = None,
) -> float:
    """Validate a persisted candidate curvature before any v4 rescore/cache use."""

    value, _receipt = _validated_phase2_directional_curvature(
        curvature_info={
            "h_raw": getattr(feat, "h_hat", None),
            "phase2_curvature_receipt": getattr(
                feat,
                "phase2_curvature_receipt",
                None,
            ),
        },
        cfg=cfg,
        candidate_label=str(getattr(feat, "candidate_label", "unresolved")),
        expected_binding=expected_binding,
    )
    return float(value)


def _feature_curvature_or_legacy_lambda_f_proxy(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
    *,
    legacy_metric: float,
) -> float:
    policy = normalize_phase2_curvature_policy(
        getattr(
            cfg,
            "phase2_curvature_policy",
            PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1,
        )
    )
    if policy == PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1:
        return validate_phase2_feature_curvature(feat, cfg)
    return float(
        feat.h_hat
        if feat.h_hat is not None
        else float(cfg.lambda_F) * float(legacy_metric)
    )


class Phase2CurvatureOracle:
    """Exact analytic Hessian blocks for the append-only reduced path."""

    def estimate(
        self,
        *,
        base_feature: CandidateFeatures,
        novelty_info: Mapping[str, Any],
        scaffold_context: _ScaffoldDerivativeContext,
        h_compiled: CompiledPolynomialAction,
        cfg: FullScoreConfig,
        optimizer_memory: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        del optimizer_memory
        F_raw = float(max(0.0, novelty_info.get("F_raw", base_feature.F_raw or base_feature.F_metric)))
        q_window = np.asarray(novelty_info.get("q_window", []), dtype=float).reshape(-1)
        Q_window = np.asarray(novelty_info.get("Q_window", scaffold_context.Q_window), dtype=float)
        cand_dpsi = np.asarray(novelty_info.get("candidate_dpsi"), dtype=complex).reshape(-1)
        cand_d2psi = np.asarray(novelty_info.get("candidate_d2psi"), dtype=complex).reshape(-1)
        cand_window_d2 = [
            np.asarray(x, dtype=complex).reshape(-1)
            for x in novelty_info.get("candidate_window_d2", [])
        ]
        hdpsi_candidate = apply_compiled_polynomial(cand_dpsi, h_compiled)
        h_raw = _energy_hessian_entry(
            dpsi_left=cand_dpsi,
            dpsi_right=cand_dpsi,
            d2psi=cand_d2psi,
            hpsi_state=scaffold_context.hpsi_state,
            hdpsi_right=hdpsi_candidate,
        )
        binding_raw = novelty_info.get("_phase2_curvature_binding")
        binding = dict(binding_raw) if isinstance(binding_raw, Mapping) else {}
        curvature_receipt = {
            "schema": PHASE2_DIRECTIONAL_CURVATURE_RECEIPT_SCHEMA,
            "status": (
                "computed_finite" if math.isfinite(float(h_raw)) else "nonfinite"
            ),
            "h_raw": float(h_raw),
            "negative_curvature": bool(
                math.isfinite(float(h_raw)) and float(h_raw) < 0.0
            ),
            **{
                str(field): binding.get(field)
                for field in _PHASE2_CURVATURE_BINDING_FIELDS
            },
            "measurement_provenance": {
                "schema": PHASE2_DIRECTIONAL_CURVATURE_PROVENANCE_SCHEMA,
                "source": "compiled_directional_energy_hessian_v1",
                "candidate_derivative_source": (
                    "compiled_ansatz_exact_parameter_derivatives_v1"
                ),
                "hamiltonian_action_source": "existing_compiled_hamiltonian_actions_v1",
                "required_primitives_resolved": True,
                "added_query_count": 0,
            },
        }

        b_mixed = np.zeros(len(scaffold_context.refit_window_indices), dtype=float)
        for idx, dpsi_window in enumerate(scaffold_context.dpsi_window):
            if idx >= len(cand_window_d2):
                break
            b_mixed[idx] = _energy_hessian_entry(
                dpsi_left=dpsi_window,
                dpsi_right=cand_dpsi,
                d2psi=cand_window_d2[idx],
                hpsi_state=scaffold_context.hpsi_state,
                hdpsi_right=hdpsi_candidate,
            )

        deferred_fallback_enabled = _deferred_gram_fallback_enabled(cfg)
        H_window = np.asarray(scaffold_context.H_window_hessian, dtype=float)
        if H_window.size == 0:
            minv_b = np.zeros(0, dtype=float)
            h_eff = float(h_raw)
            F_red = float(max(F_raw, float(cfg.metric_floor)))
            ridge_used = float(max(cfg.lambda_H, 0.0))
            mode = "append_exact_empty_window"
            deferred_novelty_payload: dict[str, Any] = {
                "schema": "phase3_deferred_gram_novelty_v1",
                "Q_window": [],
                "q_reduced": [],
                "F_red": float(F_red),
                "metric_collapse": False,
            }
        else:
            minv_b, ridge_used, _M_window = _regularized_solve(
                H_window,
                b_mixed,
                base_ridge=float(max(cfg.lambda_H, 0.0)),
                growth_factor=float(max(cfg.ridge_growth_factor, 2.0)),
                max_steps=int(max(1, cfg.ridge_max_steps)),
                require_pd=True,
            )
            h_eff = float(h_raw - float(b_mixed.T @ minv_b))
            F_red_exact = float(
                F_raw
                - 2.0 * float(q_window.T @ minv_b)
                + float(minv_b.T @ Q_window @ minv_b)
            )
            F_red = float(max(F_red_exact, float(cfg.metric_floor)))
            q_reduced = np.asarray(q_window - Q_window @ minv_b, dtype=float)
            collapse_floor = max(
                float(cfg.metric_floor),
                float(cfg.reduced_metric_collapse_rel_tol) * float(max(F_raw, float(cfg.metric_floor))),
            )
            metric_collapse = bool(F_red_exact <= collapse_floor)
            deferred_novelty_payload = {
                "schema": "phase3_deferred_gram_novelty_v1",
                "Q_window": np.asarray(Q_window, dtype=float).tolist(),
                "q_reduced": np.asarray(q_reduced, dtype=float).tolist(),
                "F_red": float(F_red),
                "metric_collapse": bool(metric_collapse),
            }
            mode = (
                "append_exact_metric_collapse_v1"
                if metric_collapse
                else (
                    "append_exact_window_hessian_ridge_grown_v1"
                    if float(ridge_used) > float(max(cfg.lambda_H, 0.0))
                    else "append_exact_window_hessian_v1"
                )
            )

        return {
            "h_raw": float(h_raw),
            "phase2_curvature_receipt": dict(curvature_receipt),
            "b_mixed": [float(x) for x in b_mixed.tolist()],
            "H_window_hessian": [[float(x) for x in row] for row in H_window.tolist()],
            "schur_window_solve": [float(x) for x in np.asarray(minv_b, dtype=float).reshape(-1).tolist()],
            "h_eff": float(h_eff),
            "F_red": float(F_red),
            "novelty": None,
            "gram_novelty_policy": (
                GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1
                if deferred_fallback_enabled
                else "off"
            ),
            "novelty_status": (
                GRAM_NOVELTY_STATUS_NOT_COMPUTED_FOR_ORDINARY_SCORING
            ),
            "novelty_query_charge": 0,
            "novelty_classical_solve_count": 0,
            "deferred_gram_novelty": deferred_novelty_payload,
            "ridge_used": float(ridge_used),
            "curvature_mode": str(mode),
        }


def _pauli_label_coeffs_from_term(term: Any) -> list[tuple[str, complex]]:
    pairs: list[tuple[str, complex]] = []
    if term is None or not hasattr(term, "polynomial"):
        return pairs
    try:
        poly_terms = list(term.polynomial.return_polynomial())
    except Exception:
        return pairs
    for poly_term in poly_terms:
        try:
            label = str(poly_term.pw2strng())
            coeff = complex(getattr(poly_term, "p_coeff", 1.0))
        except Exception:
            continue
        pairs.append((label, coeff))
    return pairs


def _array_fingerprint(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(tuple(int(x) for x in array.shape)).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _term_fingerprint_payload(term: Any) -> dict[str, Any]:
    return {
        "label": str(getattr(term, "label", "")),
        "execution_mode": str(
            getattr(term, "execution_mode", "termwise_product")
            or "termwise_product"
        ).strip().lower(),
        "terms": [
            {
                "pauli_exyz": str(label),
                "coeff_re": float(complex(coeff).real),
                "coeff_im": float(complex(coeff).imag),
            }
            for label, coeff in sorted(
                _pauli_label_coeffs_from_term(term),
                key=lambda item: (str(item[0]), float(item[1].real), float(item[1].imag)),
            )
        ],
    }


def _ordered_scaffold_fingerprint(selected_ops: Sequence[Any]) -> str:
    payload = [
        {"index": int(index), **_term_fingerprint_payload(term)}
        for index, term in enumerate(selected_ops)
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _candidate_coordinate_fingerprint(
    candidate_term: Any,
    *,
    position_id: int,
) -> str:
    payload = {
        "position_id": int(position_id),
        "candidate": _term_fingerprint_payload(candidate_term),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


EXACT_INSERTION_GEOMETRY_DEFAULT_FULL_V1 = "default_full_v1"
EXACT_INSERTION_GEOMETRY_CANDIDATE_COUPLING_SCREEN_V1 = (
    "candidate_coupling_screen_v1"
)
EXACT_INSERTION_GEOMETRY_ACQUISITION_MODES = frozenset(
    {
        EXACT_INSERTION_GEOMETRY_DEFAULT_FULL_V1,
        EXACT_INSERTION_GEOMETRY_CANDIDATE_COUPLING_SCREEN_V1,
    }
)


def _exact_insertion_joint_geometry_payload(
    *,
    scaffold_context: _ScaffoldDerivativeContext,
    candidate_term: Any,
    position_id: int,
    h_compiled: CompiledPolynomialAction,
    pauli_action_cache: dict[str, Any] | None,
    state_consistency_tolerance: float,
    old_old_geometry_prior: HistoricalSingletonOldOldGeometryPrior | None = None,
    acquisition_mode: str = EXACT_INSERTION_GEOMETRY_DEFAULT_FULL_V1,
) -> dict[str, Any]:
    """Build the reusable A/c blocks in the selector's exact insertion chart."""

    acquisition_mode_key = str(acquisition_mode).strip().lower()
    if acquisition_mode_key not in EXACT_INSERTION_GEOMETRY_ACQUISITION_MODES:
        raise ValueError(
            "Unknown exact insertion geometry acquisition mode: "
            f"{acquisition_mode!r}."
        )
    coupling_screen = bool(
        acquisition_mode_key
        == EXACT_INSERTION_GEOMETRY_CANDIDATE_COUPLING_SCREEN_V1
    )
    if coupling_screen:
        if old_old_geometry_prior is not None:
            raise ValueError(
                "Candidate-coupling screening cannot consume an old--old "
                "geometry prior."
            )
        if (
            bool(scaffold_context.old_old_geometry_measured)
            or bool(scaffold_context.old_old_metric_measured)
            or bool(scaffold_context.old_old_hessian_measured)
        ):
            raise ValueError(
                "candidate_coupling_screen_v1 requires a scaffold context "
                "with old_old_geometry_measured=False."
            )

    selected = list(scaffold_context.selected_ops)
    theta_vec = np.asarray(scaffold_context.theta, dtype=float).reshape(-1)
    position = int(position_id)
    if position < 0 or position > len(selected):
        raise ValueError(
            "Canonical Phase-2 geometry received an insertion position outside "
            "the current ordered ansatz."
        )
    combined_terms = [
        *selected[:position],
        candidate_term,
        *selected[position:],
    ]
    combined_theta = np.insert(theta_vec, position, 0.0)
    source_active = tuple(
        int(index) for index in scaffold_context.refit_window_indices
    )
    active_combined = [
        int(index if index < position else index + 1)
        for index in source_active
    ]
    derivative_indices = [*active_combined, int(position)]
    active_count = int(len(source_active))
    candidate_local_index = int(active_count)
    sparse_pairs = [
        *(tuple((index, candidate_local_index)) for index in range(active_count)),
        (candidate_local_index, candidate_local_index),
    ]
    executor = _executor_for_terms(
        combined_terms,
        pauli_action_cache=pauli_action_cache,
    )
    reconstructed, derivatives, second_derivatives = (
        _propagate_executor_sparse_second_derivatives(
            executor=executor,
            theta=np.asarray(combined_theta, dtype=float),
            psi_ref=np.asarray(scaffold_context.psi_ref, dtype=complex),
            active_indices=derivative_indices,
            second_derivative_pairs=sparse_pairs,
        )
    )
    reconstructed_state = np.asarray(reconstructed, dtype=complex).reshape(-1)
    supplied_state = np.asarray(scaffold_context.psi_state, dtype=complex).reshape(-1)
    overlap = complex(np.vdot(supplied_state, reconstructed_state))
    phase = overlap / abs(overlap) if abs(overlap) > 0.0 else 1.0 + 0.0j
    state_delta_norm = float(
        np.linalg.norm(reconstructed_state / phase - supplied_state)
    )
    tolerance = float(max(1e-12, state_consistency_tolerance))
    if state_delta_norm > tolerance:
        raise ValueError(
            "Canonical Phase-2 exact-insertion geometry reconstructed a state "
            "inconsistent with the current branch state: "
            f"delta={state_delta_norm:.6g}, tolerance={tolerance:.6g}."
        )
    aligned_derivatives = [
        np.asarray(derivative / phase, dtype=complex)
        for derivative in derivatives
    ]
    aligned_second_derivatives = {
        pair: np.asarray(value / phase, dtype=complex)
        for pair, value in second_derivatives.items()
    }
    if (
        len(scaffold_context.dpsi_window) != active_count
        or len(scaffold_context.tangents_window) != active_count
        or (
            old_old_geometry_prior is None
            and (
                np.asarray(scaffold_context.Q_window).shape
                != (active_count, active_count)
                or np.asarray(scaffold_context.H_window_hessian).shape
                != (active_count, active_count)
            )
        )
    ):
        raise ValueError(
            "Canonical Phase-2 shared active geometry does not match the "
            "resolved active context."
        )
    if old_old_geometry_prior is not None:
        if bool(scaffold_context.old_old_geometry_measured):
            raise ValueError(
                "Outer-information prior path received a scaffold context that "
                "already measured old--old geometry."
            )
        if tuple(old_old_geometry_prior.active_indices) != source_active:
            raise ValueError(
                "Outer-information prior active indices differ from the exact "
                "candidate-cross chart."
            )
        for field_name, expected_value in (
            ("state_fingerprint", scaffold_context.state_fingerprint),
            (
                "ordered_scaffold_fingerprint",
                scaffold_context.ordered_scaffold_fingerprint,
            ),
            ("theta_fingerprint", scaffold_context.theta_fingerprint),
            (
                "hamiltonian_fingerprint",
                _compiled_polynomial_fingerprint(h_compiled),
            ),
        ):
            if str(getattr(old_old_geometry_prior, field_name)) != str(
                expected_value
            ):
                raise ValueError(
                    "Outer-information prior fingerprint differs from the exact "
                    f"candidate-cross context: {field_name}."
                )
    psi_current = np.asarray(scaffold_context.psi_state, dtype=complex)
    hpsi_state = np.asarray(scaffold_context.hpsi_state, dtype=complex)
    candidate_dpsi = aligned_derivatives[candidate_local_index]
    candidate_tangent = _horizontal_tangent(psi_current, candidate_dpsi)
    G_AA = None
    if not coupling_screen:
        G_AA = np.asarray(
            scaffold_context.Q_window
            if old_old_geometry_prior is None
            else old_old_geometry_prior.G_AA,
            dtype=float,
        )
    G_A_diagonal = np.asarray(
        [
            max(0.0, float(np.real(np.vdot(tangent, tangent))))
            for tangent in scaffold_context.tangents_window
        ],
        dtype=float,
    )
    G_AB = np.asarray(
        [
            float(np.real(np.vdot(tangent, candidate_tangent)))
            for tangent in scaffold_context.tangents_window
        ],
        dtype=float,
    )
    G_BB = float(
        max(0.0, np.real(np.vdot(candidate_tangent, candidate_tangent)))
    )
    active_dpsi = [
        np.asarray(value, dtype=complex)
        for value in scaffold_context.dpsi_window
    ]
    active_hdpsi = [
        apply_compiled_polynomial(value, h_compiled) for value in active_dpsi
    ]
    candidate_hdpsi = apply_compiled_polynomial(candidate_dpsi, h_compiled)
    H_AA = None
    if not coupling_screen:
        H_AA = np.asarray(
            scaffold_context.H_window_hessian
            if old_old_geometry_prior is None
            else old_old_geometry_prior.H_AA,
            dtype=float,
        )
    H_AB = np.zeros(active_count, dtype=float)
    for active_local in range(active_count):
        mixed_second = aligned_second_derivatives[
            (active_local, candidate_local_index)
        ]
        active_candidate = _energy_hessian_entry(
            dpsi_left=active_dpsi[active_local],
            dpsi_right=candidate_dpsi,
            d2psi=mixed_second,
            hpsi_state=hpsi_state,
            hdpsi_right=candidate_hdpsi,
        )
        candidate_active = _energy_hessian_entry(
            dpsi_left=candidate_dpsi,
            dpsi_right=active_dpsi[active_local],
            d2psi=mixed_second,
            hpsi_state=hpsi_state,
            hdpsi_right=active_hdpsi[active_local],
        )
        H_AB[active_local] = float(
            0.5 * (active_candidate + candidate_active)
        )
    H_BB = _energy_hessian_entry(
        dpsi_left=candidate_dpsi,
        dpsi_right=candidate_dpsi,
        d2psi=aligned_second_derivatives[
            (candidate_local_index, candidate_local_index)
        ],
        hpsi_state=hpsi_state,
        hdpsi_right=candidate_hdpsi,
    )
    if coupling_screen:
        g_A = None
    elif old_old_geometry_prior is None:
        g_A = np.asarray(
            [
                -2.0
                * float(
                    np.real(
                        np.vdot(
                            derivative,
                            hpsi_state,
                        )
                    )
                )
                for derivative in active_dpsi
            ],
            dtype=float,
        )
    else:
        g_A = np.asarray(old_old_geometry_prior.g_A, dtype=float)
    candidate_gradient = float(
        -2.0 * np.real(np.vdot(candidate_dpsi, hpsi_state))
    )
    payload = {
        "schema": "phase2_joint_geometry_reuse_v2",
        "status": "populated",
        "coordinate_chart": "exact_ordered_insertion_zero_angle_v1",
        "derivative_convention": "compiled_ansatz_exact_parameter_derivatives_v1",
        "cache_scope": "branch_local_state_scaffold_theta_context_v1",
        "state_fingerprint": str(scaffold_context.state_fingerprint),
        "ordered_scaffold_fingerprint": str(
            scaffold_context.ordered_scaffold_fingerprint
        ),
        "theta_fingerprint": str(scaffold_context.theta_fingerprint),
        "hamiltonian_fingerprint": _compiled_polynomial_fingerprint(
            h_compiled
        ),
        "candidate_coordinate_fingerprint": _candidate_coordinate_fingerprint(
            candidate_term,
            position_id=position,
        ),
        "candidate_position_id": int(position),
        "active_indices": [int(index) for index in source_active],
        "G_AB": G_AB.tolist(),
        "G_BB": float(G_BB),
        "H_AB": H_AB.tolist(),
        "H_BB": float(H_BB),
        "descent_gradient": float(candidate_gradient),
        "state_reconstruction_delta_norm": float(state_delta_norm),
        "state_consistency_tolerance": float(tolerance),
        "acquisition_mode": str(acquisition_mode_key),
        "active_block_source": (
            EXACT_INSERTION_GEOMETRY_CANDIDATE_COUPLING_SCREEN_V1
            if coupling_screen
            else "shared_scaffold_context_v1"
            if old_old_geometry_prior is None
            else "formal_manifold_outer_information_prior_v1"
        ),
        "active_block_recomputed": False,
        "sparse_second_derivative_pair_count": int(len(sparse_pairs)),
        "dense_second_derivative_entry_count_avoided": int(
            (active_count + 1) ** 2 - len(sparse_pairs)
        ),
        "G_AA_element_count": int(
            0 if coupling_screen else active_count * (active_count + 1) // 2
        ),
        "H_AA_element_count": int(
            0 if coupling_screen else active_count * (active_count + 1) // 2
        ),
        "G_AC_element_count": int(active_count),
        "H_AC_element_count": int(active_count),
        "G_CC_diagonal_element_count": 1,
        "H_CC_diagonal_element_count": 1,
        "scaffold_old_old_hessian_source": (
            _scaffold_hessian_source_telemetry(scaffold_context)
        ),
    }
    if coupling_screen:
        payload.update(
            {
                "schema": "phase3_candidate_coupling_screen_v1",
                "G_A_diagonal": G_A_diagonal.tolist(),
                "G_A_diagonal_element_count": int(active_count),
                "old_old_geometry_measured": False,
                "old_old_metric_element_count_acquired": 0,
                "old_old_direct_curvature_element_count_acquired": 0,
                "old_old_descent_gradient_component_count_acquired": 0,
                "placeholder_old_old_blocks_serialized": False,
            }
        )
    else:
        assert G_AA is not None
        assert H_AA is not None
        assert g_A is not None
        payload.update(
            {
                "G_AA": G_AA.tolist(),
                "H_AA": H_AA.tolist(),
                "g_A": g_A.tolist(),
            }
        )
    if old_old_geometry_prior is not None:
        payload.update(
            {
                "old_old_geometry_prior_used": True,
                "old_old_geometry_prior_fingerprint": str(
                    old_old_geometry_prior.prior_fingerprint
                ),
                "old_old_metric_reacquired": False,
                "old_old_direct_curvature_reacquired": False,
                "old_old_descent_gradient_reacquired": False,
                "old_old_metric_element_count_acquired": 0,
                "old_old_direct_curvature_element_count_acquired": 0,
                "old_old_descent_gradient_component_count_acquired": 0,
            }
        )
    return payload


def _promote_fresh_phase3_joint_geometry_receipt(
    *,
    acquired_payload: Mapping[str, Any],
    scaffold_context: _ScaffoldDerivativeContext,
    candidate_term: Any,
    position_id: int,
    h_compiled: CompiledPolynomialAction,
    state_consistency_tolerance: float,
    active_gradient_policy: str = "measured_residual_response_v1",
) -> dict[str, Any]:
    """Bind already acquired Phase-III blocks into the strict reuse schema.

    ``build_full_candidate_features`` has already evaluated the shared
    scaffold and candidate cross/self blocks when this function is called.
    This function therefore validates and serializes those values; it does not
    invoke another geometry backend.
    """

    source = dict(acquired_payload)
    if str(source.get("schema", "")) != "phase2_joint_geometry_reuse_v1":
        raise ValueError(
            "Fresh Phase-III receipt promotion requires the complete acquired "
            "v1 block payload."
        )
    if not (
        bool(scaffold_context.old_old_geometry_measured)
        and bool(scaffold_context.old_old_metric_measured)
        and bool(scaffold_context.old_old_hessian_measured)
    ):
        raise ValueError(
            "Fresh Phase-III receipt promotion cannot serialize predicted or "
            "unmeasured old--old geometry."
        )
    active = tuple(
        int(index) for index in scaffold_context.refit_window_indices
    )
    active_count = int(len(active))
    tolerance = float(max(1e-12, state_consistency_tolerance))
    state_delta = float(
        scaffold_context.state_reconstruction_delta_norm
    )
    if not math.isfinite(state_delta) or state_delta > tolerance:
        raise ValueError(
            "Fresh Phase-III scaffold state certificate is invalid: "
            f"delta={state_delta:.6g}, tolerance={tolerance:.6g}."
        )

    G_AA = np.asarray(source.get("G_AA", ()), dtype=float).reshape(
        active_count, active_count
    )
    H_AA = np.asarray(source.get("H_AA", ()), dtype=float).reshape(
        active_count, active_count
    )
    G_AB = np.asarray(source.get("G_AB", ()), dtype=float).reshape(
        active_count
    )
    H_AB = np.asarray(source.get("H_AB", ()), dtype=float).reshape(
        active_count
    )
    G_BB = float(source.get("G_BB", float("nan")))
    H_BB = float(source.get("H_BB", float("nan")))
    candidate_gradient = float(
        source.get("descent_gradient", float("nan"))
    )
    gradient_policy = str(active_gradient_policy).strip()
    if gradient_policy == "stationary_source_response_v1":
        # The stationary-source protocol is coupling-only. Do not evaluate
        # active residual contractions and then erase them: no active-gradient
        # value is acquired at all.
        g_A = np.zeros(active_count, dtype=float)
        active_gradient_indices_acquired: list[int] = []
        active_gradient_source = "not_acquired_stationary_source_protocol"
    elif gradient_policy == "measured_residual_response_v1":
        g_A = np.asarray(
            [
                -2.0
                * float(
                    np.real(
                        np.vdot(
                            derivative,
                            scaffold_context.hpsi_state,
                        )
                    )
                )
                for derivative in scaffold_context.dpsi_window
            ],
            dtype=float,
        )
        active_gradient_indices_acquired = [int(index) for index in active]
        active_gradient_source = "measured_active_residual_response_v1"
    else:
        raise ValueError(
            "active_gradient_policy must be one of "
            "{'stationary_source_response_v1',"
            "'measured_residual_response_v1'}."
        )
    for name, block in (
        ("G_AA", G_AA),
        ("H_AA", H_AA),
        ("G_AB", G_AB),
        ("H_AB", H_AB),
        ("g_A", g_A),
        ("G_BB", np.asarray([G_BB], dtype=float)),
        ("H_BB", np.asarray([H_BB], dtype=float)),
        (
            "descent_gradient",
            np.asarray([candidate_gradient], dtype=float),
        ),
    ):
        if not np.all(np.isfinite(block)):
            raise ValueError(
                "Fresh Phase-III acquired block is incomplete or nonfinite: "
                f"{name}."
            )

    return {
        "schema": "phase2_joint_geometry_reuse_v2",
        "status": "populated",
        "coordinate_chart": "exact_ordered_insertion_zero_angle_v1",
        "derivative_convention": (
            "compiled_ansatz_exact_parameter_derivatives_v1"
        ),
        "cache_scope": "same_outer_iteration_phase3_population_v1",
        "state_fingerprint": str(scaffold_context.state_fingerprint),
        "ordered_scaffold_fingerprint": str(
            scaffold_context.ordered_scaffold_fingerprint
        ),
        "theta_fingerprint": str(scaffold_context.theta_fingerprint),
        "hamiltonian_fingerprint": _compiled_polynomial_fingerprint(
            h_compiled
        ),
        "candidate_coordinate_fingerprint": (
            _candidate_coordinate_fingerprint(
                candidate_term,
                position_id=int(position_id),
            )
        ),
        "candidate_position_id": int(position_id),
        "append_position": int(source.get("append_position", position_id)),
        "active_indices": [int(index) for index in active],
        "G_AA": G_AA.tolist(),
        "G_AB": G_AB.tolist(),
        "G_BB": float(G_BB),
        "H_AA": H_AA.tolist(),
        "H_AB": H_AB.tolist(),
        "H_BB": float(H_BB),
        "g_A": g_A.tolist(),
        "active_gradient_policy": gradient_policy,
        "active_gradient_indices_acquired": active_gradient_indices_acquired,
        "active_gradient_source": active_gradient_source,
        "descent_gradient": float(candidate_gradient),
        "phase3_deferred_gram_novelty": (
            None
            if source.get("phase3_deferred_gram_novelty") is None
            else dict(source["phase3_deferred_gram_novelty"])
        ),
        "state_reconstruction_delta_norm": float(state_delta),
        "state_consistency_tolerance": float(tolerance),
        "acquisition_mode": "fresh_projected_phase3_population_v1",
        "acquisition_authority": (
            "fresh_projected_phase3_child_measurement_v1"
        ),
        "active_block_source": "shared_phase3_scaffold_acquisition_v1",
        "active_block_recomputed": False,
        "receipt_promotion_recomputed_geometry": False,
        "G_AA_element_count": int(
            active_count * (active_count + 1) // 2
        ),
        "H_AA_element_count": int(
            active_count * (active_count + 1) // 2
        ),
        "G_AC_element_count": int(active_count),
        "H_AC_element_count": int(active_count),
        "G_CC_diagonal_element_count": 1,
        "H_CC_diagonal_element_count": 1,
        "scaffold_old_old_hessian_source": (
            _scaffold_hessian_source_telemetry(scaffold_context)
        ),
        "same_outer_iteration_consumers": [
            "phase3_supported_response_projection",
            "accepted_refit_supported_fs_gram",
        ],
        "cross_outer_iteration_reuse_permitted": False,
    }


def _pauli_labels_from_term(term: Any) -> list[str]:
    return [str(label) for label, _coeff in _pauli_label_coeffs_from_term(term)]


def _support_set(term: Any) -> set[int]:
    support: set[int] = set()
    labels = _pauli_labels_from_term(term)
    for label in labels:
        for idx, ch in enumerate(str(label)):
            if ch != "e":
                support.add(int(idx))
    return support


def _pauli_strings_commute(lhs: str, rhs: str) -> bool:
    anticomm = 0
    for a, b in zip(str(lhs), str(rhs)):
        if a == "e" or b == "e" or a == b:
            continue
        anticomm += 1
    return bool((anticomm % 2) == 0)


def _polynomials_commute(term_a: Any, term_b: Any) -> bool:
    labels_a = _pauli_labels_from_term(term_a)
    labels_b = _pauli_labels_from_term(term_b)
    if not labels_a or not labels_b:
        return True
    for lhs in labels_a:
        for rhs in labels_b:
            if not _pauli_strings_commute(lhs, rhs):
                return False
    return True


def _normalize_pauli_word_for_commutation(raw_word: Any) -> str | None:
    word = str(raw_word).strip().lower().replace("i", "e")
    if word == "" or any(ch not in {"e", "x", "y", "z"} for ch in word):
        return None
    return word


def _record_pauli_words(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return candidate Pauli words carried by a shortlisted record.

    The CEO benchmark selector intentionally rejects malformed non-leading
    candidates instead of recomputing operator content from other sources.
    """
    term = record.get("candidate_term")
    if term is None or not hasattr(term, "polynomial"):
        return ()
    try:
        raw_words = _pauli_labels_from_term(term)
    except Exception:
        return ()
    words: list[str] = []
    for raw_word in raw_words:
        word = _normalize_pauli_word_for_commutation(raw_word)
        if word is None:
            return ()
        words.append(str(word))
    return tuple(words)


def _single_pauli_word_commutes(a_word: str, b_word: str) -> bool:
    lhs = _normalize_pauli_word_for_commutation(a_word)
    rhs = _normalize_pauli_word_for_commutation(b_word)
    if lhs is None or rhs is None or len(lhs) != len(rhs):
        return False
    anticomm = 0
    for a_ch, b_ch in zip(lhs, rhs):
        if a_ch == "e" or b_ch == "e" or a_ch == b_ch:
            continue
        anticomm += 1
    return bool((anticomm % 2) == 0)


def _records_commute(record_a: Mapping[str, Any], record_b: Mapping[str, Any]) -> bool:
    words_a = _record_pauli_words(record_a)
    words_b = _record_pauli_words(record_b)
    if not words_a or not words_b:
        return False
    return all(_single_pauli_word_commutes(lhs, rhs) for lhs in words_a for rhs in words_b)


def build_full_candidate_features(
    *,
    base_feature: CandidateFeatures,
    candidate_term: Any,
    cfg: FullScoreConfig,
    novelty_oracle: OrderedInsertionGeometryOracleProtocol,
    curvature_oracle: CurvatureOracle,
    scaffold_context: _ScaffoldDerivativeContext,
    phase2_scaffold_context: _ScaffoldDerivativeContext | None = None,
    phase3_scaffold_context: _ScaffoldDerivativeContext | None = None,
    h_compiled: CompiledPolynomialAction,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    optimizer_memory: Mapping[str, Any] | None = None,
    motif_library: Mapping[str, Any] | None = None,
    target_num_sites: int | None = None,
    include_phase3: bool = True,
    emit_fresh_phase3_joint_geometry_receipt: bool = False,
) -> CandidateFeatures:
    phase2_context = phase2_scaffold_context or scaffold_context
    phase3_context = (
        phase3_scaffold_context or scaffold_context
        if bool(include_phase3)
        else phase2_context
    )
    novelty_info_phase2 = novelty_oracle.estimate(
        scaffold_context=phase2_context,
        candidate_label=str(base_feature.candidate_label),
        candidate_term=candidate_term,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
    )
    if tuple(int(i) for i in phase3_context.refit_window_indices) == tuple(
        int(i) for i in phase2_context.refit_window_indices
    ):
        novelty_info_phase3 = novelty_info_phase2
    else:
        novelty_info_phase3 = novelty_oracle.estimate(
            scaffold_context=phase3_context,
            candidate_label=str(base_feature.candidate_label),
            candidate_term=candidate_term,
            compiled_cache=compiled_cache,
            pauli_action_cache=pauli_action_cache,
        )
    phase2_curvature_binding = _phase2_curvature_binding(
        scaffold_context=phase3_context,
        h_compiled=h_compiled,
        candidate_term=candidate_term,
        position_id=int(base_feature.position_id),
    )
    curvature_novelty_info = dict(novelty_info_phase3)
    curvature_novelty_info["_phase2_curvature_binding"] = dict(
        phase2_curvature_binding
    )
    curvature_info = curvature_oracle.estimate(
        base_feature=base_feature,
        novelty_info=curvature_novelty_info,
        scaffold_context=phase3_context,
        h_compiled=h_compiled,
        cfg=cfg,
        optimizer_memory=optimizer_memory,
    )
    if not isinstance(curvature_info, Mapping):
        if (
            normalize_phase2_curvature_policy(
                getattr(
                    cfg,
                    "phase2_curvature_policy",
                    PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1,
                )
            )
            == PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1
        ):
            _phase2_curvature_failure(
                "curvature oracle returned a malformed payload",
                candidate_label=str(base_feature.candidate_label),
            )
        curvature_info = {}
    phase2_h_raw, phase2_curvature_receipt = (
        _validated_phase2_directional_curvature(
            curvature_info=curvature_info,
            cfg=cfg,
            candidate_label=str(base_feature.candidate_label),
            expected_binding=phase2_curvature_binding,
        )
    )
    raw_geometry = phase2_raw_geometry_score(
        base_feature,
        F_raw=float(max(0.0, novelty_info_phase2.get("F_raw", base_feature.F_metric))),
        h_raw=float(phase2_h_raw),
        q_window=novelty_info_phase2.get("q_window", []),
        Q_window=novelty_info_phase2.get("Q_window", phase2_context.Q_window),
        cfg=cfg,
    )
    phase2_state_fingerprint = hashlib.sha256(
        np.asarray(phase2_context.psi_state, dtype=complex).tobytes()
    ).hexdigest()
    phase2_joint_geometry_reuse = {
        "schema": "phase2_joint_geometry_reuse_v1",
        "coordinate_chart": "append_candidate_after_current_ansatz_v1",
        "state_fingerprint": str(phase2_state_fingerprint),
        "active_indices": [
            int(index) for index in phase2_context.refit_window_indices
        ],
        "candidate_position_id": int(base_feature.position_id),
        "append_position": int(base_feature.append_position),
        "G_AA": np.asarray(phase2_context.Q_window, dtype=float).tolist(),
        "G_AB": np.asarray(
            novelty_info_phase2.get("q_window", []),
            dtype=float,
        ).reshape(-1).tolist(),
        "G_BB": float(
            max(
                0.0,
                novelty_info_phase2.get("F_raw", base_feature.F_metric),
            )
        ),
        "H_AA": np.asarray(
            curvature_info.get(
                "H_window_hessian",
                phase2_context.H_window_hessian,
            ),
            dtype=float,
        ).tolist(),
        "H_AB": [
            float(value) for value in curvature_info.get("b_mixed", [])
        ],
        "H_BB": float(phase2_h_raw),
        "phase2_curvature_receipt": (
            None
            if phase2_curvature_receipt is None
            else dict(phase2_curvature_receipt)
        ),
        "descent_gradient": float(-base_feature.g_signed),
        "phase3_deferred_gram_novelty": (
            None
            if curvature_info.get("deferred_gram_novelty") is None
            else dict(curvature_info.get("deferred_gram_novelty"))
        ),
        "scaffold_old_old_hessian_source": (
            _scaffold_hessian_source_telemetry(phase3_context)
        ),
    }
    if bool(emit_fresh_phase3_joint_geometry_receipt):
        if not bool(include_phase3):
            raise ValueError(
                "A fresh Phase-III joint-geometry receipt requires "
                "include_phase3=True."
            )
        phase2_joint_geometry_reuse = (
            _promote_fresh_phase3_joint_geometry_receipt(
                acquired_payload=phase2_joint_geometry_reuse,
                scaffold_context=phase3_context,
                candidate_term=candidate_term,
                position_id=int(base_feature.position_id),
                h_compiled=h_compiled,
                state_consistency_tolerance=float(
                    max(
                        1e-12,
                        getattr(
                            cfg,
                            "batch_state_consistency_tolerance",
                            1e-8,
                        ),
                    )
                ),
                active_gradient_policy=str(
                    getattr(
                        cfg,
                        "active_gradient_policy",
                        "measured_residual_response_v1",
                    )
                ),
            )
        )
    elif not bool(include_phase3):
        phase2_joint_geometry_reuse = _exact_insertion_joint_geometry_payload(
            scaffold_context=phase2_context,
            candidate_term=candidate_term,
            position_id=int(base_feature.position_id),
            h_compiled=h_compiled,
            pauli_action_cache=pauli_action_cache,
            state_consistency_tolerance=float(
                max(
                    1e-12,
                    getattr(cfg, "batch_state_consistency_tolerance", 1e-8),
                )
            ),
        )
    feat = _replace_feature(
        base_feature,
        novelty=(
            None
            if curvature_info.get("novelty") is None
            else float(curvature_info.get("novelty"))
        ),
        novelty_mode=(
            "deferred_for_geometry_expansion_fallback"
            if curvature_info.get("novelty") is None
            else str(
                novelty_info_phase3.get(
                    "novelty_mode", "append_exact_tangent_context_v1"
                )
            )
        ),
        curvature_mode=str(curvature_info.get("curvature_mode", "append_exact_window_hessian_v1")),
        F_metric=float(max(0.0, novelty_info_phase2.get("F_raw", base_feature.F_metric))),
        metric_proxy=float(max(0.0, novelty_info_phase2.get("F_raw", base_feature.metric_proxy))),
        F_raw=float(max(0.0, novelty_info_phase2.get("F_raw", base_feature.F_raw or base_feature.F_metric))),
        h_eff=float(curvature_info.get("h_eff", 0.0)),
        F_red=float(curvature_info.get("F_red", novelty_info_phase3.get("F_raw", 0.0))),
        ridge_used=float(curvature_info.get("ridge_used", max(cfg.lambda_H, 0.0))),
        h_hat=float(phase2_h_raw),
        phase2_curvature_policy=str(
            normalize_phase2_curvature_policy(
                getattr(
                    cfg,
                    "phase2_curvature_policy",
                    PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1,
                )
            )
        ),
        phase2_cheap_curvature_proxy_policy=str(
            normalize_phase2_cheap_curvature_proxy_policy(
                getattr(
                    cfg,
                    "phase2_cheap_curvature_proxy_policy",
                    PHASE2_CHEAP_CURVATURE_PROXY_POLICY_LEGACY_LAMBDA_F_RATIO_V1,
                )
            )
        ),
        phase2_curvature_receipt=(
            {}
            if phase2_curvature_receipt is None
            else dict(phase2_curvature_receipt)
        ),
        phase2_lambda_f_proxy_applied=False,
        phase2_missing_curvature_fallback_used=False,
        b_hat=[float(x) for x in curvature_info.get("b_mixed", [])],
        H_window=[[float(x) for x in row] for row in curvature_info.get("H_window_hessian", [])],
        phase2_joint_geometry_reuse=dict(phase2_joint_geometry_reuse),
        schur_window_solve=[float(x) for x in curvature_info.get("schur_window_solve", [])],
        refit_window_basis="old_pre_geometry_alias",
        phase2_geometry_window_indices=[int(i) for i in phase2_context.refit_window_indices],
        phase3_geometry_refit_window_indices=[int(i) for i in phase3_context.refit_window_indices],
        schur_window_indices=[int(i) for i in phase3_context.refit_window_indices],
        score_version=str(cfg.score_version),
        confidence_factor=float(raw_geometry.get("confidence_factor", 1.0)),
        phase2_raw_overlap_max=(
            None
            if raw_geometry.get("phase2_raw_overlap_max") is None
            else float(raw_geometry.get("phase2_raw_overlap_max"))
        ),
        phase2_raw_novelty=(
            None
            if raw_geometry.get("phase2_raw_novelty") is None
            else float(raw_geometry.get("phase2_raw_novelty"))
        ),
        phase2_novelty_mode=str(
            raw_geometry.get(
                "phase2_novelty_mode",
                ORDINARY_NOVELTY_SCORING_RETIRED_V1,
            )
        ),
        phase2_novelty_source=str(
            raw_geometry.get(
                "phase2_novelty_source",
                ORDINARY_NOVELTY_SCORING_RETIRED_V1,
            )
        ),
        phase2_novelty_fallback_reason=(
            None
            if raw_geometry.get("phase2_novelty_fallback_reason") is None
            else str(raw_geometry.get("phase2_novelty_fallback_reason"))
        ),
        phase2_span_projection_z=(
            None
            if raw_geometry.get("phase2_span_projection_z") is None
            else float(raw_geometry.get("phase2_span_projection_z"))
        ),
        phase2_novelty_ridge_used=(
            None
            if raw_geometry.get("phase2_novelty_ridge_used") is None
            else float(raw_geometry.get("phase2_novelty_ridge_used"))
        ),
        phase2_raw_F_effective=(
            None
            if raw_geometry.get("phase2_raw_F_effective") is None
            else float(raw_geometry.get("phase2_raw_F_effective"))
        ),
        phase2_legacy_pairwise_novelty=(
            None
            if raw_geometry.get("phase2_legacy_pairwise_novelty") is None
            else float(raw_geometry.get("phase2_legacy_pairwise_novelty"))
        ),
        phase2_confidence_applied=bool(raw_geometry.get("phase2_confidence_applied", False)),
        phase2_raw_score_formula=str(raw_geometry.get("phase2_raw_score_formula", PHASE2_CANONICAL_RAW_SCORE_FORMULA)),
        phase2_raw_trust_gain=float(raw_geometry.get("phase2_raw_trust_gain", 0.0)),
        phase2_raw_score=float(raw_geometry.get("phase2_raw_score", 0.0)),
        phase2_burden_total=float(raw_geometry.get("phase2_burden_total", 1.0)),
        placeholder_hooks={
            **dict(base_feature.placeholder_hooks),
            "novelty_oracle": True,
            "curvature_oracle": True,
            "full_v2_score": True,
        },
    )
    if not bool(include_phase3):
        phase2_score = float(feat.phase2_raw_score or 0.0)
        phase2_burden = float(feat.phase2_burden_total or 1.0)
        return _replace_feature(
            feat,
            full_v2_score=float(phase2_score),
            phase3_primary_score=None,
            phase3_tie_break_score=0.0,
            phase3_auxiliary_score_mode="disabled_in_canonical_child12_route",
            phase3_canonical_score_formula="disabled_in_canonical_child12_route",
            phase3_reduced_novelty=0.0,
            phase3_reduced_trust_gain=0.0,
            phase3_burden_total=0.0,
            selector_score=float(phase2_score),
            selector_burden=float(phase2_burden),
            selector_geometry_mode="phase2_only_v1",
            phase_score_components={
                **dict(feat.phase_score_components),
                "phase2_raw_score": float(phase2_score),
                "phase2_raw_trust_gain": float(
                    feat.phase2_raw_trust_gain or 0.0
                ),
                "phase2_raw_novelty": (
                    None
                    if feat.phase2_raw_novelty is None
                    else float(feat.phase2_raw_novelty)
                ),
                "phase2_gram_novelty_policy": str(
                    raw_geometry.get(
                        "phase2_gram_novelty_policy",
                        (
                            GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1
                            if _deferred_gram_fallback_enabled(cfg)
                            else "off"
                        ),
                    )
                ),
                "phase2_novelty_status": str(
                    raw_geometry.get(
                        "phase2_novelty_status",
                        GRAM_NOVELTY_STATUS_NOT_COMPUTED_FOR_ORDINARY_SCORING,
                    )
                ),
                "phase2_novelty_query_charge": int(
                    raw_geometry.get("phase2_novelty_query_charge", 0)
                ),
                "selector_score": float(phase2_score),
            },
            phase_cost_components={
                **dict(feat.phase_cost_components),
                "phase2_burden_total": float(phase2_burden),
            },
            actual_fallback_mode="phase2_only_v1",
        )
    if isinstance(base_feature.generator_metadata, Mapping) and isinstance(motif_library, Mapping):
        motif_bonus, motif_meta = motif_bonus_for_generator(
            generator_metadata=base_feature.generator_metadata,
            motif_library=motif_library,
            target_num_sites=int(max(0, target_num_sites or 0)),
        )
        feat = _replace_feature(
            feat,
            motif_bonus=float(motif_bonus),
            motif_source=(
                str(motif_library.get("source_tag", "payload"))
                if bool(motif_bonus) else str(feat.motif_source)
            ),
            motif_metadata=(dict(motif_meta) if isinstance(motif_meta, Mapping) else feat.motif_metadata),
        )
    feat = _replace_feature(
        feat,
        lifetime_weight_components=dict(lifetime_weight_components(feat, cfg)),
        lifetime_cost_mode=str(cfg.lifetime_cost_mode),
        remaining_evaluations_proxy_mode=str(cfg.remaining_evaluations_proxy_mode),
    )
    canonical_components = phase3_canonical_score_components(feat, cfg)
    score = float(_full_v2_score_from_components(canonical_components))
    fallback_mode = str(canonical_components.get("fallback_mode", "unknown"))
    selector_geometry_mode = str(getattr(cfg, "phase3_selector_geometry_mode", "reduced")).strip().lower()
    if selector_geometry_mode not in {"reduced", "raw_exact"}:
        selector_geometry_mode = "reduced"
    phase3_burden_total = float(canonical_components.get("denominator_1_plus_K3", _cheap_burden_total(feat, cfg)))
    phase3_primary_score = float(canonical_components.get("phase3_primary_score", score))
    phase3_tie_break_score = float(canonical_components.get("phase3_tie_break_score", 0.0))
    phase3_auxiliary_score_mode = str(
        canonical_components.get("phase3_auxiliary_score_mode", PHASE3_AUXILIARY_SCORE_TIE_BREAK_ONLY)
    )
    selector_score = float(score)
    selector_burden = float(phase3_burden_total)
    if selector_geometry_mode == "raw_exact":
        selector_score = float(feat.phase2_raw_score or 0.0)
        selector_burden = float(feat.phase2_burden_total or phase3_burden_total)
    phase_score_components = {
        **dict(feat.phase_score_components),
        "phase2_raw_score": float(feat.phase2_raw_score or 0.0),
        "phase2_raw_trust_gain": float(feat.phase2_raw_trust_gain or 0.0),
        "phase2_g_hw_lcb": float(raw_geometry.get("phase2_g_hw_lcb", _selector_gradient_lcb(feat, cfg))),
        "phase2_raw_novelty": (
            None
            if feat.phase2_raw_novelty is None
            else float(feat.phase2_raw_novelty)
        ),
        "phase2_measured_novelty": (
            None
            if raw_geometry.get("phase2_measured_novelty") is None
            else float(raw_geometry.get("phase2_measured_novelty"))
        ),
        "phase2_novelty_multiplier": (
            None
            if raw_geometry.get("phase2_novelty_multiplier") is None
            else float(raw_geometry.get("phase2_novelty_multiplier"))
        ),
        "phase2_gram_novelty_policy": str(
            raw_geometry.get(
                "phase2_gram_novelty_policy",
                (
                    GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1
                    if _deferred_gram_fallback_enabled(cfg)
                    else "off"
                ),
            )
        ),
        "phase2_novelty_status": str(
            raw_geometry.get(
                "phase2_novelty_status",
                GRAM_NOVELTY_STATUS_NOT_COMPUTED_FOR_ORDINARY_SCORING,
            )
        ),
        "phase2_novelty_query_charge": int(
            raw_geometry.get("phase2_novelty_query_charge", 0)
        ),
        "phase2_novelty_multiplier_policy": str(
            raw_geometry.get(
                "phase2_novelty_multiplier_policy",
                ORDINARY_NOVELTY_SCORING_RETIRED_V1,
            )
        ),
        "phase2_novelty_applied": bool(
            raw_geometry.get("phase2_novelty_applied", True)
        ),
        "phase2_span_projection_z": float(feat.phase2_span_projection_z or 0.0),
        "phase2_raw_F_effective": float(feat.phase2_raw_F_effective or 0.0),
        "phase2_legacy_pairwise_novelty": (
            None
            if feat.phase2_legacy_pairwise_novelty is None
            else float(feat.phase2_legacy_pairwise_novelty)
        ),
        "phase2_confidence_applied": float(1.0 if bool(feat.phase2_confidence_applied) else 0.0),
        "phase3_reduced_score": float(score),
        "phase3_reduced_novelty": (
            None if feat.novelty is None else float(feat.novelty)
        ),
        "phase3_delta_e_tr": float(canonical_components.get("delta_e_tr", 0.0)),
        "DeltaE_TR": float(canonical_components.get("DeltaE_TR", 0.0)),
        "phase3_g_hw_lcb": float(canonical_components.get("g_hw_lcb", _selector_gradient_lcb(feat, cfg))),
        "phase3_confidence_factor": float(canonical_components.get("confidence_factor", 1.0)),
        "confidence_factor": float(canonical_components.get("confidence_factor", 1.0)),
        "phase3_N3": (
            None
            if canonical_components.get("N3") is None
            else float(canonical_components.get("N3"))
        ),
        "N3": (
            None
            if canonical_components.get("N3") is None
            else float(canonical_components.get("N3"))
        ),
        "phase3_measured_novelty": (
            None
            if canonical_components.get("phase3_measured_novelty") is None
            else float(canonical_components.get("phase3_measured_novelty"))
        ),
        "phase3_novelty_multiplier": (
            None
            if canonical_components.get("phase3_novelty_multiplier") is None
            else float(canonical_components.get("phase3_novelty_multiplier"))
        ),
        "phase3_gram_novelty_policy": str(
            canonical_components.get(
                "phase3_gram_novelty_policy",
                (
                    GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1
                    if _deferred_gram_fallback_enabled(cfg)
                    else "off"
                ),
            )
        ),
        "phase3_novelty_status": str(
            canonical_components.get(
                "phase3_novelty_status",
                GRAM_NOVELTY_STATUS_NOT_COMPUTED_FOR_ORDINARY_SCORING,
            )
        ),
        "phase3_novelty_query_charge": int(
            canonical_components.get("phase3_novelty_query_charge", 0)
        ),
        "phase3_novelty_multiplier_policy": str(
            canonical_components.get(
                "phase3_novelty_multiplier_policy",
                ORDINARY_NOVELTY_SCORING_RETIRED_V1,
            )
        ),
        "phase3_novelty_applied": bool(
            canonical_components.get("phase3_novelty_applied", False)
        ),
        "phase3_K3": float(canonical_components.get("K3", 0.0)),
        "K3": float(canonical_components.get("K3", 0.0)),
        "phase3_denominator_1_plus_K3": float(canonical_components.get("denominator_1_plus_K3", phase3_burden_total)),
        "denominator_1_plus_K3": float(canonical_components.get("denominator_1_plus_K3", phase3_burden_total)),
        "phase3_leakage_factor": float(canonical_components.get("leakage_factor", 0.0)),
        "phase3_primary_score": float(phase3_primary_score),
        "phase3_tie_break_score": float(phase3_tie_break_score),
        "phase3_motif_tie_break_score": float(canonical_components.get("motif_tie_break_score", 0.0)),
        "phase3_duplicate_tie_break_score": float(canonical_components.get("duplicate_tie_break_score", 0.0)),
        "selector_score": float(selector_score),
    }
    phase_cost_components = {
        **dict(feat.phase_cost_components),
        "phase2_burden_total": float(feat.phase2_burden_total or 0.0),
        "phase3_burden_total": float(phase3_burden_total),
        "phase3_denominator_1_plus_K3": float(phase3_burden_total),
    }
    return _replace_feature(
        feat,
        full_v2_score=float(score),
        phase3_primary_score=float(phase3_primary_score),
        phase3_tie_break_score=float(phase3_tie_break_score),
        phase3_auxiliary_score_mode=str(phase3_auxiliary_score_mode),
        phase3_canonical_score_formula=str(
            canonical_components.get(
                "phase3_canonical_score_formula",
                PHASE3_CANONICAL_SCORE_FORMULA,
            )
        ),
        phase3_reduced_novelty=(
            None if feat.novelty is None else float(feat.novelty)
        ),
        phase3_reduced_trust_gain=float(canonical_components.get("delta_e_tr", 0.0)),
        phase3_burden_total=float(phase3_burden_total),
        selector_score=float(selector_score),
        selector_burden=float(selector_burden),
        selector_geometry_mode=str(selector_geometry_mode),
        phase_score_components=phase_score_components,
        phase_cost_components=phase_cost_components,
        actual_fallback_mode=str(fallback_mode),
    )


def _compatibility_penalty_components(
    *,
    record_a: Mapping[str, Any],
    record_b: Mapping[str, Any],
    cfg: FullScoreConfig,
    tangent_for_record: Callable[[Mapping[str, Any]], tuple[np.ndarray, float]] | None = None,
) -> dict[str, float]:
    feat_a = record_a.get("feature")
    feat_b = record_b.get("feature")
    term_a = record_a.get("candidate_term")
    term_b = record_b.get("candidate_term")
    if not isinstance(feat_a, CandidateFeatures) or not isinstance(feat_b, CandidateFeatures):
        return {
            "support_overlap": 0.0,
            "noncommutation": 0.0,
            "cross_curvature": 0.0,
            "schedule": 0.0,
            "measurement_mismatch": 0.0,
            "total": 0.0,
        }

    supp_a = _support_set(term_a)
    supp_b = _support_set(term_b)
    union = len(supp_a | supp_b)
    support_overlap = 0.0 if union == 0 else float(len(supp_a & supp_b) / union)
    noncomm = 0.0 if _polynomials_commute(term_a, term_b) else 1.0

    cross_curv = 0.0
    if tangent_for_record is not None and term_a is not None and term_b is not None:
        try:
            tang_a, F_a = tangent_for_record(record_a)
            tang_b, F_b = tangent_for_record(record_b)
            denom = math.sqrt(max(F_a, 0.0) * max(F_b, 0.0))
            if denom > 0.0:
                cross_curv = float(min(1.0, abs(float(np.real(np.vdot(tang_a, tang_b)))) / denom))
        except Exception:
            cross_curv = float(support_overlap)
    elif feat_a.b_hat is not None and feat_b.b_hat is not None:
        vec_a = np.asarray(feat_a.b_hat, dtype=float)
        vec_b = np.asarray(feat_b.b_hat, dtype=float)
        denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if denom > 0.0:
            cross_curv = float(min(1.0, abs(float(vec_a @ vec_b)) / denom))

    win_a = set(int(i) for i in _feature_phase3_schur_window(feat_a))
    win_b = set(int(i) for i in _feature_phase3_schur_window(feat_b))
    union_w = len(win_a | win_b)
    schedule = 0.0 if union_w == 0 else float(len(win_a & win_b) / union_w)
    measurement_overlap = _measurement_group_overlap_score(
        measurement_group_keys_for_term(term_a),
        measurement_group_keys_for_term(term_b),
    )
    measurement_mismatch = float(1.0 - measurement_overlap)
    total = (
        float(cfg.compat_overlap_weight) * float(support_overlap)
        + float(cfg.compat_comm_weight) * float(noncomm)
        + float(cfg.compat_curv_weight) * float(cross_curv)
        + float(cfg.compat_sched_weight) * float(schedule)
        + float(cfg.compat_measure_weight) * float(measurement_mismatch)
    )
    return {
        "support_overlap": float(support_overlap),
        "noncommutation": float(noncomm),
        "cross_curvature": float(cross_curv),
        "schedule": float(schedule),
        "measurement_mismatch": float(measurement_mismatch),
        "total": float(total),
    }


def compatibility_penalty(
    *,
    record_a: Mapping[str, Any],
    record_b: Mapping[str, Any],
    cfg: FullScoreConfig,
    psi_state: np.ndarray | None = None,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
) -> dict[str, float]:
    tangent_for_record: Callable[[Mapping[str, Any]], tuple[np.ndarray, float]] | None = None
    if psi_state is not None:

        def _uncached_tangent_for_record(record: Mapping[str, Any]) -> tuple[np.ndarray, float]:
            feat = record.get("feature")
            term = record.get("candidate_term")
            return _tangent_data(
                psi_state=np.asarray(psi_state, dtype=complex),
                label=str(feat.candidate_label),
                polynomial=term.polynomial,
                compiled_cache=compiled_cache,
                pauli_action_cache=pauli_action_cache,
            )

        tangent_for_record = _uncached_tangent_for_record

    return _compatibility_penalty_components(
        record_a=record_a,
        record_b=record_b,
        cfg=cfg,
        tangent_for_record=tangent_for_record,
    )


class CompatibilityPenaltyOracle:
    def __init__(
        self,
        *,
        cfg: FullScoreConfig,
        psi_state: np.ndarray | None = None,
        compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
        pauli_action_cache: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.psi_state = None if psi_state is None else np.asarray(psi_state, dtype=complex)
        self.compiled_cache = compiled_cache
        self.pauli_action_cache = pauli_action_cache
        self._tangent_cache: dict[str, tuple[np.ndarray, float]] = {}

    @staticmethod
    def _record_cache_key(record: Mapping[str, Any]) -> str:
        feat = record.get("feature")
        if isinstance(feat, CandidateFeatures):
            try:
                return f"feature:{str(feat.candidate_label)}"
            except Exception:
                pass
        raw_label = record.get("candidate_label")
        if raw_label is not None:
            return f"record:{str(raw_label)}"
        term = record.get("candidate_term")
        label = getattr(term, "label", None)
        if label is not None:
            return f"term:{str(label)}"
        return f"id:{id(record)}"

    def _tangent_for_record(self, record: Mapping[str, Any]) -> tuple[np.ndarray, float]:
        cache_key = self._record_cache_key(record)
        cached = self._tangent_cache.get(cache_key)
        if cached is not None:
            return cached
        feat = record.get("feature")
        term = record.get("candidate_term")
        if self.psi_state is None or not isinstance(feat, CandidateFeatures) or term is None:
            raise ValueError("tangent cache requires psi_state, feature, and candidate_term.")
        tangent = _tangent_data(
            psi_state=np.asarray(self.psi_state, dtype=complex),
            label=str(feat.candidate_label),
            polynomial=term.polynomial,
            compiled_cache=self.compiled_cache,
            pauli_action_cache=self.pauli_action_cache,
        )
        self._tangent_cache[cache_key] = tangent
        return tangent

    def penalty(self, record_a: Mapping[str, Any], record_b: Mapping[str, Any]) -> dict[str, float]:
        tangent_for_record = self._tangent_for_record if self.psi_state is not None else None
        return _compatibility_penalty_components(
            record_a=record_a,
            record_b=record_b,
            cfg=self.cfg,
            tangent_for_record=tangent_for_record,
        )


def _batch_sort_key(record: Mapping[str, Any], tie_break_score_key: str) -> tuple[float, float, int, int]:
    full_score = record.get("full_v2_score", float("-inf"))
    if full_score is None:
        full_score = float("-inf")
    tie_score = record.get(tie_break_score_key, float("-inf"))
    if tie_score is None:
        tie_score = float("-inf")
    return (
        -float(full_score),
        -float(tie_score),
        int(record.get("candidate_pool_index", -1)),
        int(record.get("position_id", -1)),
    )


def _solve_joint_trust_region_gain(
    *,
    g_vec: np.ndarray,
    G_mat: np.ndarray,
    H_mat: np.ndarray,
    rho: float,
) -> tuple[float, np.ndarray]:
    g = np.asarray(g_vec, dtype=float).reshape(-1)
    G = 0.5 * (np.asarray(G_mat, dtype=float) + np.asarray(G_mat, dtype=float).T)
    H = 0.5 * (np.asarray(H_mat, dtype=float) + np.asarray(H_mat, dtype=float).T)
    n = int(g.size)
    if n == 0:
        return 0.0, np.zeros(0, dtype=float)
    eye = np.eye(n, dtype=float)
    rho_sq = float(max(0.0, rho)) ** 2

    def _alpha(lam: float) -> np.ndarray:
        trial = H + float(lam) * G + 1e-12 * eye
        sol = np.linalg.solve(trial, g)
        return np.asarray(np.maximum(sol, 0.0), dtype=float)

    def _constraint(alpha: np.ndarray) -> float:
        return float(alpha.T @ G @ alpha)

    alpha0 = _alpha(0.0)
    if _constraint(alpha0) <= rho_sq:
        alpha = alpha0
    else:
        lo = 0.0
        hi = 1.0
        alpha_hi = _alpha(hi)
        while _constraint(alpha_hi) > rho_sq and hi < 1e12:
            lo = hi
            hi *= 2.0
            alpha_hi = _alpha(hi)
        alpha = alpha_hi
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            alpha_mid = _alpha(mid)
            if _constraint(alpha_mid) > rho_sq:
                lo = mid
            else:
                hi = mid
                alpha = alpha_mid
    gain = float(g.T @ alpha - 0.5 * alpha.T @ H @ alpha)
    return float(max(0.0, gain)), np.asarray(alpha, dtype=float)


def _batch_geometry_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    novelty_oracle: Any,
    curvature_oracle: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prepared: list[tuple[dict[str, Any], CandidateFeatures, Any]] = []
    common_window = sorted(
        {
            int(idx)
            for rec in records
            for idx in (
                _feature_phase3_schur_window(rec.get("feature"))
                if isinstance(rec.get("feature"), CandidateFeatures)
                else []
            )
        }
    )
    scaffold_context = novelty_oracle.prepare_scaffold_context(
        selected_ops=list(selected_ops),
        theta=np.asarray(theta, dtype=float),
        psi_ref=np.asarray(psi_ref, dtype=complex),
        psi_state=np.asarray(psi_state, dtype=complex),
        h_compiled=h_compiled,
        hpsi_state=apply_compiled_polynomial(np.asarray(psi_state, dtype=complex), h_compiled),
        refit_window_indices=list(common_window),
        pauli_action_cache=pauli_action_cache,
    )
    H_window = np.asarray(scaffold_context.H_window_hessian, dtype=float)
    Q_window = np.asarray(scaffold_context.Q_window, dtype=float)
    for rec in records:
        feat = rec.get("feature")
        candidate_term = rec.get("candidate_term")
        if not isinstance(feat, CandidateFeatures) or candidate_term is None:
            return {"feasible": False, "reason": "invalid_record"}
        novelty_info = novelty_oracle.estimate(
            scaffold_context=scaffold_context,
            candidate_label=str(feat.candidate_label),
            candidate_term=candidate_term,
            compiled_cache=compiled_cache,
            pauli_action_cache=pauli_action_cache,
        )
        curvature_info = curvature_oracle.estimate(
            base_feature=feat,
            novelty_info=novelty_info,
            scaffold_context=scaffold_context,
            h_compiled=h_compiled,
            cfg=cfg,
            optimizer_memory=None,
        )
        b_vec = np.asarray(curvature_info.get("b_mixed", []), dtype=float).reshape(-1)
        if H_window.size == 0 or b_vec.size == 0:
            v_vec = np.zeros_like(b_vec, dtype=float)
        else:
            v_vec, _ridge, _trial = _regularized_solve(
                H_window,
                b_vec,
                base_ridge=float(max(cfg.lambda_H, 0.0)),
                growth_factor=float(max(cfg.ridge_growth_factor, 2.0)),
                max_steps=int(max(1, cfg.ridge_max_steps)),
                require_pd=True,
            )
        prepared.append(
            (
                dict(rec),
                feat,
                {
                    "novelty_info": novelty_info,
                    "curvature_info": curvature_info,
                    "b_vec": np.asarray(b_vec, dtype=float),
                    "v_vec": np.asarray(v_vec, dtype=float),
                    "candidate_tangent": np.asarray(novelty_info.get("candidate_tangent"), dtype=complex).reshape(-1),
                    "q_vec": np.asarray(novelty_info.get("q_window", []), dtype=float).reshape(-1),
                    "h_eff": float(curvature_info.get("h_eff", 0.0)),
                    "g_hw_lcb": float(_selector_gradient_lcb(feat, cfg)),
                },
            )
        )
    n = int(len(prepared))
    G = np.zeros((n, n), dtype=float)
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        feat_i = prepared[i][1]
        aux_i = prepared[i][2]
        tau_i = np.asarray(aux_i["candidate_tangent"], dtype=complex)
        q_i = np.asarray(aux_i["q_vec"], dtype=float)
        v_i = np.asarray(aux_i["v_vec"], dtype=float)
        H[i, i] = float(max(0.0, aux_i["h_eff"]))
        for j in range(i, n):
            aux_j = prepared[j][2]
            tau_j = np.asarray(aux_j["candidate_tangent"], dtype=complex)
            q_j = np.asarray(aux_j["q_vec"], dtype=float)
            v_j = np.asarray(aux_j["v_vec"], dtype=float)
            c_ij = float(np.real(np.vdot(tau_i, tau_j)))
            if Q_window.size == 0:
                g_ij = float(c_ij)
            else:
                g_ij = float(c_ij - q_i.T @ v_j - q_j.T @ v_i + v_i.T @ Q_window @ v_j)
            G[i, j] = g_ij
            G[j, i] = g_ij
    trace_G = float(np.trace(G))
    lambda_min = float(np.min(np.linalg.eigvalsh(G))) if n > 0 else 0.0
    rank_floor = float(cfg.batch_rank_rel_tol) * float(trace_G / max(1, n))
    if n > 1 and lambda_min < rank_floor:
        return {
            "feasible": False,
            "reason": "rank_gate",
            "lambda_min": float(lambda_min),
            "rank_floor": float(rank_floor),
            "common_window_indices": [int(x) for x in common_window],
        }
    g_vec = np.asarray([float(item[2]["g_hw_lcb"]) for item in prepared], dtype=float)
    try:
        joint_gain, alpha = _solve_joint_trust_region_gain(
            g_vec=g_vec,
            G_mat=G,
            H_mat=H,
            rho=float(cfg.rho),
        )
    except np.linalg.LinAlgError:
        return {
            "feasible": False,
            "reason": "conditioning_gate",
            "lambda_min": float(lambda_min),
            "rank_floor": float(rank_floor),
            "common_window_indices": [int(x) for x in common_window],
        }
    contextual_single = [
        float(trust_region_drop(float(g_vec[i]), float(H[i, i]), float(max(G[i, i], cfg.metric_floor)), float(cfg.rho)))
        for i in range(n)
    ]
    single_total = float(sum(contextual_single))
    additivity_defect = float(max(0.0, 1.0 - joint_gain / (single_total + float(cfg.cheap_score_eps))))
    additivity_policy = normalize_batch_additivity_policy(
        getattr(cfg, "batch_additivity_policy", BATCH_ADDITIVITY_HARD_GATE_LEGACY_V1)
    )
    if (
        n > 1
        and additivity_policy == BATCH_ADDITIVITY_HARD_GATE_LEGACY_V1
        and additivity_defect > float(cfg.batch_additivity_tol)
    ):
        return {
            "feasible": False,
            "reason": "additivity_hard_gate_legacy",
            "joint_gain": float(joint_gain),
            "contextual_single_total": float(single_total),
            "additivity_defect": float(additivity_defect),
            "additivity_policy": str(additivity_policy),
            "common_window_indices": [int(x) for x in common_window],
        }
    mu_tan = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            denom = math.sqrt(max(G[i, i], 0.0) * max(G[j, j], 0.0)) + float(cfg.cheap_score_eps)
            mu_tan = max(mu_tan, abs(float(G[i, j])) / denom)
    return {
        "feasible": True,
        "joint_gain": float(joint_gain),
        "contextual_single_total": float(single_total),
        "additivity_defect": float(additivity_defect),
        "additivity_policy": str(additivity_policy),
        "lambda_min": float(lambda_min),
        "rank_floor": float(rank_floor),
        "mu_tan": float(mu_tan),
        "alpha": [float(x) for x in alpha.tolist()],
        "common_window_indices": [int(x) for x in common_window],
        "schur_window_solves": [
            [float(x) for x in np.asarray(item[2]["v_vec"], dtype=float).reshape(-1).tolist()]
            for item in prepared
        ],
        "G": [[float(x) for x in row] for row in G.tolist()],
    }


@dataclass(frozen=True)
class BatchSelectionProposal:
    records: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    score: float
    delta_e3: float
    k3: float
    denominator_1_plus_k3: float


def _ordered_reduced_plane_batch_limits(cfg: FullScoreConfig) -> tuple[int, int]:
    cap = int(max(1, min(int(max(1, cfg.batch_size_cap)), 5)))
    target = int(max(1, min(int(max(1, cfg.batch_target_size)), cap)))
    return target, cap


def _phase3_record_k3(record: Mapping[str, Any]) -> float:
    feat = record.get("feature")
    candidate_values: list[Any] = []
    if isinstance(feat, CandidateFeatures):
        if isinstance(feat.phase_score_components, Mapping):
            candidate_values.extend(
                [
                    feat.phase_score_components.get("K3"),
                    feat.phase_score_components.get("phase3_K3"),
                ]
            )
        if isinstance(feat.phase_cost_components, Mapping):
            candidate_values.extend(
                [
                    feat.phase_cost_components.get("K3"),
                    feat.phase_cost_components.get("phase3_K3"),
                ]
            )
        for raw in (
            feat.phase3_burden_total,
            feat.selector_burden,
            feat.hardware_cost_denominator,
        ):
            if raw is not None:
                try:
                    candidate_values.append(float(raw) - 1.0)
                except (TypeError, ValueError):
                    pass
    elif isinstance(feat, Mapping):
        phase_score_components = feat.get("phase_score_components")
        if isinstance(phase_score_components, Mapping):
            candidate_values.extend(
                [phase_score_components.get("K3"), phase_score_components.get("phase3_K3")]
            )
        phase_cost_components = feat.get("phase_cost_components")
        if isinstance(phase_cost_components, Mapping):
            candidate_values.extend(
                [phase_cost_components.get("K3"), phase_cost_components.get("phase3_K3")]
            )
        for key in ("phase3_burden_total", "selector_burden", "hardware_cost_denominator"):
            raw = feat.get(key)
            if raw is not None:
                try:
                    candidate_values.append(float(raw) - 1.0)
                except (TypeError, ValueError):
                    pass
    record_phase_score_components = record.get("phase_score_components")
    if isinstance(record_phase_score_components, Mapping):
        candidate_values.extend(
            [
                record_phase_score_components.get("K3"),
                record_phase_score_components.get("phase3_K3"),
            ]
        )
    for key in ("phase3_burden_total", "selector_burden", "hardware_cost_denominator"):
        raw = record.get(key)
        if raw is not None:
            try:
                candidate_values.append(float(raw) - 1.0)
            except (TypeError, ValueError):
                pass
    for value in candidate_values:
        k3 = _finite_nonnegative(value, default=float("nan"))
        if math.isfinite(k3):
            return float(k3)
    return 0.0


def _phase3_batch_k3(records: Sequence[Mapping[str, Any]]) -> float:
    return float(sum(_phase3_record_k3(record) for record in records))


def _phase2_record_k2(record: Mapping[str, Any]) -> float:
    feat = record.get("feature")
    candidate_values: list[Any] = []
    if isinstance(feat, CandidateFeatures):
        for components in (feat.phase_score_components, feat.phase_cost_components):
            if isinstance(components, Mapping):
                candidate_values.extend(
                    [components.get("K2"), components.get("phase2_K2")]
                )
        if feat.phase2_burden_total is not None:
            candidate_values.append(float(feat.phase2_burden_total) - 1.0)
    for key in ("phase2_K2", "K2"):
        candidate_values.append(record.get(key))
    if record.get("phase2_burden_total") is not None:
        candidate_values.append(float(record.get("phase2_burden_total")) - 1.0)
    for value in candidate_values:
        k2 = _finite_nonnegative(value, default=float("nan"))
        if math.isfinite(k2):
            return float(k2)
    return _phase3_record_k3(record)


def _batch_cost_excess(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
) -> float:
    if _canonical_ranked_child_batch_search(cfg):
        return float(sum(_phase2_record_k2(record) for record in records))
    return _phase3_batch_k3(records)


def _batch_record_identity_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(record.get("candidate_pool_index", -1)),
        int(record.get("position_id", -1)),
        str(_batch_record_label(record)),
    )


def _canonical_ranked_child_batch_search(cfg: FullScoreConfig) -> bool:
    return bool(
        str(
            getattr(
                cfg,
                "batch_search_population_mode",
                BATCH_SEARCH_POPULATION_NEAR_DEGENERATE_SHELL_LEGACY_V1,
            )
        )
        == BATCH_SEARCH_POPULATION_RANKED_CHILD_PHASE2_V1
    )


def _batch_proposal_sort_key(proposal: BatchSelectionProposal) -> tuple[float, float, float, int, tuple[tuple[int, int, str], ...]]:
    return (
        -float(proposal.score),
        -float(proposal.delta_e3),
        float(proposal.denominator_1_plus_k3),
        -int(len(proposal.records)),
        tuple(_batch_record_identity_key(record) for record in proposal.records),
    )


def _batch_proposal_compare(
    lhs: BatchSelectionProposal,
    rhs: BatchSelectionProposal,
    *,
    cfg: FullScoreConfig,
) -> int:
    if not _canonical_ranked_child_batch_search(cfg):
        lhs_key = _batch_proposal_sort_key(lhs)
        rhs_key = _batch_proposal_sort_key(rhs)
        return -1 if lhs_key < rhs_key else (1 if lhs_key > rhs_key else 0)
    tolerance = float(max(0.0, getattr(cfg, "batch_score_tie_tolerance", 0.0)))
    if float(lhs.score) > float(rhs.score) + tolerance:
        return -1
    if float(rhs.score) > float(lhs.score) + tolerance:
        return 1
    if float(lhs.k3) < float(rhs.k3) - tolerance:
        return -1
    if float(rhs.k3) < float(lhs.k3) - tolerance:
        return 1
    if len(lhs.records) != len(rhs.records):
        return -1 if len(lhs.records) < len(rhs.records) else 1
    lhs_key = tuple(sorted(_batch_record_identity_key(record) for record in lhs.records))
    rhs_key = tuple(sorted(_batch_record_identity_key(record) for record in rhs.records))
    return -1 if lhs_key < rhs_key else (1 if lhs_key > rhs_key else 0)


def _batch_proposal_better(
    lhs: BatchSelectionProposal,
    rhs: BatchSelectionProposal,
    *,
    cfg: FullScoreConfig,
) -> bool:
    return bool(_batch_proposal_compare(lhs, rhs, cfg=cfg) < 0)


def _sort_batch_proposals(
    proposals: Sequence[BatchSelectionProposal],
    *,
    cfg: FullScoreConfig,
) -> list[BatchSelectionProposal]:
    return sorted(
        proposals,
        key=cmp_to_key(lambda lhs, rhs: _batch_proposal_compare(lhs, rhs, cfg=cfg)),
    )


def _phase3_batch_shell(
    ranked_records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    tie_break_score_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    ranked = sorted([dict(rec) for rec in ranked_records], key=lambda rec: _batch_sort_key(rec, tie_break_score_key))
    if not ranked:
        return [], [], float("-inf")
    top_score = _batch_record_score(ranked[0], "full_v2_score")
    shell = [
        dict(rec)
        for rec in ranked
        if _batch_record_score(rec, "full_v2_score") > 0.0
        and _batch_record_score(rec, "full_v2_score") >= float(cfg.batch_near_degenerate_ratio) * float(top_score)
    ]
    return ranked, shell, float(top_score)


def _phase2_ranked_batch_population(
    ranked_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        [dict(record) for record in ranked_records],
        key=lambda record: (
            -_batch_record_score(record, "phase2_raw_score"),
            -_batch_record_score(record, "phase1_active_score"),
            int(record.get("candidate_pool_index", -1)),
            int(record.get("position_id", -1)),
            str(_batch_record_label(record)),
        ),
    )


def _batch_search_population(
    ranked_records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    tie_break_score_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    if _canonical_ranked_child_batch_search(cfg):
        ranked = _phase2_ranked_batch_population(ranked_records)
        top_score = (
            _batch_record_score(ranked[0], "phase2_raw_score")
            if ranked
            else float("-inf")
        )
        return ranked, list(ranked), float(top_score)
    return _phase3_batch_shell(
        ranked_records,
        cfg=cfg,
        tie_break_score_key=tie_break_score_key,
    )


HISTORICAL_SINGLETON_OLD_OLD_GEOMETRY_PRIOR_SCHEMA = (
    "historical_singleton_old_old_geometry_prior_v1"
)


def _historical_singleton_active_coordinate_identities(
    selected_ops: Sequence[Any],
    active_indices: Sequence[int],
) -> tuple[str, ...]:
    """Return ordered, registry-stable identities for active coordinates."""

    identities: list[str] = []
    selected = list(selected_ops)
    for raw_index in active_indices:
        index = int(raw_index)
        if index < 0 or index >= len(selected):
            raise ValueError(
                "Historical singleton active coordinate index is out of range."
            )
        payload = {
            "schema": "historical_singleton_active_coordinate_identity_v1",
            "ansatz_index": int(index),
            "term": _term_fingerprint_payload(selected[index]),
        }
        identities.append(
            hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest()
        )
    return tuple(identities)


@dataclass(frozen=True)
class HistoricalSingletonOldOldGeometryPrior:
    """Typed old--old Phase-III geometry supplied by an outer FM model.

    ``G_AA``, ``H_AA``, and ``g_A`` use the historical singleton workspace's
    raw active-coordinate ordering.  In particular, ``g_A`` is the SR descent
    gradient (the negative of the raw energy differential).  Candidate cross
    and candidate--candidate blocks are deliberately absent: the authoritative
    Phase-III builder must still acquire those at the current state.
    """

    active_indices: tuple[int, ...]
    active_coordinate_identities: tuple[str, ...]
    G_AA: np.ndarray
    H_AA: np.ndarray
    g_A: np.ndarray
    state_fingerprint: str
    ordered_scaffold_fingerprint: str
    theta_fingerprint: str
    hamiltonian_fingerprint: str
    source_prior_id: str
    source_state_id: str
    source_frame_id: str
    source_support_id: str
    source_geometry_status: str
    source_provenance_ids: tuple[str, ...]
    prior_fingerprint: str = ""
    schema: str = HISTORICAL_SINGLETON_OLD_OLD_GEOMETRY_PRIOR_SCHEMA

    def __post_init__(self) -> None:
        if str(self.schema) != HISTORICAL_SINGLETON_OLD_OLD_GEOMETRY_PRIOR_SCHEMA:
            raise ValueError("Historical singleton old--old prior schema mismatch.")
        active_indices = tuple(int(value) for value in self.active_indices)
        identities = tuple(str(value) for value in self.active_coordinate_identities)
        if len(set(active_indices)) != len(active_indices):
            raise ValueError("Historical singleton old--old prior indices repeat.")
        if len(identities) != len(active_indices) or any(not value for value in identities):
            raise ValueError(
                "Historical singleton old--old prior identities are incomplete."
            )
        dimension = int(len(active_indices))
        metric = np.asarray(self.G_AA, dtype=float)
        curvature = np.asarray(self.H_AA, dtype=float)
        gradient = np.asarray(self.g_A, dtype=float).reshape(-1)
        if metric.shape != (dimension, dimension):
            raise ValueError("Historical singleton old--old metric shape mismatch.")
        if curvature.shape != (dimension, dimension):
            raise ValueError(
                "Historical singleton old--old direct-curvature shape mismatch."
            )
        if gradient.shape != (dimension,):
            raise ValueError(
                "Historical singleton old--old descent-gradient shape mismatch."
            )
        for name, value in (
            ("metric", metric),
            ("direct curvature", curvature),
            ("descent gradient", gradient),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(
                    f"Historical singleton old--old {name} is nonfinite."
                )
        symmetry_tolerance = float(
            4096.0
            * np.finfo(float).eps
            * max(1, dimension)
            * max(
                1.0,
                float(np.linalg.norm(metric, ord=2)) if metric.size else 0.0,
                float(np.linalg.norm(curvature, ord=2)) if curvature.size else 0.0,
            )
        )
        if not np.allclose(
            metric, metric.T, rtol=0.0, atol=symmetry_tolerance
        ):
            raise ValueError("Historical singleton old--old metric is not symmetric.")
        if not np.allclose(
            curvature, curvature.T, rtol=0.0, atol=symmetry_tolerance
        ):
            raise ValueError(
                "Historical singleton old--old direct curvature is not symmetric."
            )
        fingerprint_fields = (
            "state_fingerprint",
            "ordered_scaffold_fingerprint",
            "theta_fingerprint",
            "hamiltonian_fingerprint",
            "source_prior_id",
            "source_state_id",
            "source_frame_id",
            "source_support_id",
            "source_geometry_status",
        )
        if any(not str(getattr(self, name)) for name in fingerprint_fields):
            raise ValueError(
                "Historical singleton old--old prior provenance is incomplete."
            )
        provenance_ids = tuple(str(value) for value in self.source_provenance_ids)
        if any(not value for value in provenance_ids):
            raise ValueError(
                "Historical singleton old--old prior has an empty provenance id."
            )
        metric = np.array(0.5 * (metric + metric.T), dtype=float, copy=True)
        curvature = np.array(
            0.5 * (curvature + curvature.T), dtype=float, copy=True
        )
        gradient = np.array(gradient, dtype=float, copy=True)
        metric.setflags(write=False)
        curvature.setflags(write=False)
        gradient.setflags(write=False)
        payload = {
            "schema": HISTORICAL_SINGLETON_OLD_OLD_GEOMETRY_PRIOR_SCHEMA,
            "active_indices": list(active_indices),
            "active_coordinate_identities": list(identities),
            "G_AA_fingerprint": _array_fingerprint(metric),
            "H_AA_fingerprint": _array_fingerprint(curvature),
            "g_A_fingerprint": _array_fingerprint(gradient),
            "state_fingerprint": str(self.state_fingerprint),
            "ordered_scaffold_fingerprint": str(
                self.ordered_scaffold_fingerprint
            ),
            "theta_fingerprint": str(self.theta_fingerprint),
            "hamiltonian_fingerprint": str(self.hamiltonian_fingerprint),
            "source_prior_id": str(self.source_prior_id),
            "source_state_id": str(self.source_state_id),
            "source_frame_id": str(self.source_frame_id),
            "source_support_id": str(self.source_support_id),
            "source_geometry_status": str(self.source_geometry_status),
            "source_provenance_ids": list(provenance_ids),
        }
        expected_fingerprint = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        if self.prior_fingerprint and str(self.prior_fingerprint) != expected_fingerprint:
            raise ValueError(
                "Historical singleton old--old prior fingerprint mismatch."
            )
        object.__setattr__(self, "active_indices", active_indices)
        object.__setattr__(self, "active_coordinate_identities", identities)
        object.__setattr__(self, "G_AA", metric)
        object.__setattr__(self, "H_AA", curvature)
        object.__setattr__(self, "g_A", gradient)
        object.__setattr__(self, "source_provenance_ids", provenance_ids)
        object.__setattr__(self, "prior_fingerprint", expected_fingerprint)

    def telemetry(self) -> dict[str, Any]:
        dimension = int(len(self.active_indices))
        triangle = int(dimension * (dimension + 1) // 2)
        return {
            "schema": str(self.schema),
            "prior_fingerprint": str(self.prior_fingerprint),
            "source_prior_id": str(self.source_prior_id),
            "source_state_id": str(self.source_state_id),
            "source_frame_id": str(self.source_frame_id),
            "source_support_id": str(self.source_support_id),
            "source_geometry_status": str(self.source_geometry_status),
            "source_provenance_ids": list(self.source_provenance_ids),
            "active_indices": [int(value) for value in self.active_indices],
            "active_coordinate_identities": list(
                self.active_coordinate_identities
            ),
            "old_old_metric_element_count_reused": int(triangle),
            "old_old_direct_curvature_element_count_reused": int(triangle),
            "old_old_descent_gradient_component_count_reused": int(dimension),
            "old_old_metric_element_count_acquired": 0,
            "old_old_direct_curvature_element_count_acquired": 0,
            "old_old_descent_gradient_component_count_acquired": 0,
            "old_old_geometry_reacquired": False,
        }


def build_historical_singleton_old_old_geometry_prior(
    *,
    selected_ops: Sequence[Any],
    active_indices: Sequence[int],
    theta: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    G_AA: np.ndarray,
    H_AA: np.ndarray,
    g_A: np.ndarray,
    source_prior_id: str,
    source_state_id: str,
    source_frame_id: str,
    source_support_id: str,
    source_geometry_status: str,
    source_provenance_ids: Sequence[str] = (),
) -> HistoricalSingletonOldOldGeometryPrior:
    """Bind transported arrays to the exact endpoint/scaffold registry.

    The caller must pass ``g_A`` in the SR descent-gradient convention.  This
    constructor derives every consumer-side fingerprint so adapt-pipeline code
    does not have to duplicate the scoring module's private identity rules.
    """

    active = tuple(int(value) for value in active_indices)
    return HistoricalSingletonOldOldGeometryPrior(
        active_indices=active,
        active_coordinate_identities=(
            _historical_singleton_active_coordinate_identities(
                selected_ops, active
            )
        ),
        G_AA=np.asarray(G_AA, dtype=float),
        H_AA=np.asarray(H_AA, dtype=float),
        g_A=np.asarray(g_A, dtype=float),
        state_fingerprint=_array_fingerprint(
            np.asarray(psi_state, dtype=complex)
        ),
        ordered_scaffold_fingerprint=_ordered_scaffold_fingerprint(
            selected_ops
        ),
        theta_fingerprint=_array_fingerprint(np.asarray(theta, dtype=float)),
        hamiltonian_fingerprint=_compiled_polynomial_fingerprint(h_compiled),
        source_prior_id=str(source_prior_id),
        source_state_id=str(source_state_id),
        source_frame_id=str(source_frame_id),
        source_support_id=str(source_support_id),
        source_geometry_status=str(source_geometry_status),
        source_provenance_ids=tuple(
            str(value) for value in source_provenance_ids
        ),
    )


def _validate_historical_singleton_old_old_geometry_prior(
    prior: HistoricalSingletonOldOldGeometryPrior,
    *,
    selected_ops: Sequence[Any],
    active_indices: Sequence[int],
    theta: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
) -> HistoricalSingletonOldOldGeometryPrior:
    if not isinstance(prior, HistoricalSingletonOldOldGeometryPrior):
        raise TypeError(
            "old_old_geometry_prior must be a "
            "HistoricalSingletonOldOldGeometryPrior."
        )
    active = tuple(int(value) for value in active_indices)
    expected = {
        "active_indices": active,
        "active_coordinate_identities": (
            _historical_singleton_active_coordinate_identities(
                selected_ops, active
            )
        ),
        "state_fingerprint": _array_fingerprint(
            np.asarray(psi_state, dtype=complex)
        ),
        "ordered_scaffold_fingerprint": _ordered_scaffold_fingerprint(
            selected_ops
        ),
        "theta_fingerprint": _array_fingerprint(
            np.asarray(theta, dtype=float)
        ),
        "hamiltonian_fingerprint": _compiled_polynomial_fingerprint(
            h_compiled
        ),
    }
    for field_name, expected_value in expected.items():
        if getattr(prior, field_name) != expected_value:
            raise ValueError(
                "Historical singleton old--old prior "
                f"{field_name} does not match the authoritative Phase-III endpoint."
            )
    return prior


@dataclass
class _Phase2JointGeometryCache:
    active_indices: tuple[int, ...]
    G_AA: np.ndarray
    H_AA: np.ndarray
    g_A: np.ndarray
    G_AB: np.ndarray
    H_AB: np.ndarray
    G_BB_diagonal: np.ndarray
    H_BB_diagonal: np.ndarray
    g_B: np.ndarray
    valid_record_indices: tuple[int, ...]
    valid_gradient_record_indices: tuple[int, ...]
    record_results: list[dict[str, Any]]
    active_block_valid: bool
    state_fingerprint: str
    ordered_scaffold_fingerprint: str
    theta_fingerprint: str
    state_reconstruction_delta_norm_max: float

    @property
    def complete(self) -> bool:
        return bool(
            self.active_block_valid
            and len(self.valid_record_indices) == int(self.G_AB.shape[1])
            and len(self.valid_gradient_record_indices) == int(self.G_AB.shape[1])
        )


def _build_phase2_joint_geometry_cache(
    records: Sequence[Mapping[str, Any]],
    *,
    active_indices: Sequence[int],
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    tolerance: float,
    old_old_geometry_prior: HistoricalSingletonOldOldGeometryPrior | None = None,
) -> _Phase2JointGeometryCache:
    active = tuple(int(index) for index in active_indices)
    state_fingerprint = _array_fingerprint(
        np.asarray(psi_state, dtype=complex)
    )
    scaffold_fingerprint = _ordered_scaffold_fingerprint(selected_ops)
    theta_fingerprint = _array_fingerprint(np.asarray(theta, dtype=float))
    hamiltonian_fingerprint = _compiled_polynomial_fingerprint(h_compiled)
    atol = float(max(1e-10, tolerance))
    active_count = int(len(active))
    candidate_count = int(len(records))
    if old_old_geometry_prior is None:
        G_AA = np.full((active_count, active_count), np.nan, dtype=float)
        H_AA = np.full((active_count, active_count), np.nan, dtype=float)
        g_A = np.full(active_count, np.nan, dtype=float)
    else:
        G_AA = np.asarray(old_old_geometry_prior.G_AA, dtype=float).copy()
        H_AA = np.asarray(old_old_geometry_prior.H_AA, dtype=float).copy()
        g_A = np.asarray(old_old_geometry_prior.g_A, dtype=float).copy()
    G_AB = np.full((active_count, candidate_count), np.nan, dtype=float)
    H_AB = np.full((active_count, candidate_count), np.nan, dtype=float)
    G_BB_diagonal = np.full(candidate_count, np.nan, dtype=float)
    H_BB_diagonal = np.full(candidate_count, np.nan, dtype=float)
    g_B = np.full(candidate_count, np.nan, dtype=float)
    record_results: list[dict[str, Any]] = []
    valid_indices: list[int] = []
    valid_gradient_indices: list[int] = []
    active_reference_initialized = bool(old_old_geometry_prior is not None)
    state_reconstruction_delta_norm_max = 0.0

    def _matches(lhs: Any, rhs: Any) -> bool:
        lhs_array = np.asarray(lhs, dtype=float)
        rhs_array = np.asarray(rhs, dtype=float)
        if lhs_array.size == 0 and rhs_array.size == 0:
            return True
        return bool(
            lhs_array.shape == rhs_array.shape
            and np.allclose(lhs_array, rhs_array, atol=atol, rtol=1e-7)
        )

    for candidate_index, record in enumerate(records):
        feature = record.get("feature")
        payload = (
            feature.phase2_joint_geometry_reuse
            if isinstance(feature, CandidateFeatures)
            else {}
        )
        reasons: list[str] = []
        gradient_reasons: list[str] = []
        if not isinstance(payload, Mapping) or str(payload.get("schema") or "") != (
            "phase2_joint_geometry_reuse_v2"
        ):
            reasons.append("missing_phase2_reuse_payload")
        else:
            state_reconstruction_delta_norm_max = max(
                float(state_reconstruction_delta_norm_max),
                float(payload.get("state_reconstruction_delta_norm", 0.0)),
            )
            if str(payload.get("coordinate_chart") or "") != (
                "exact_ordered_insertion_zero_angle_v1"
            ):
                reasons.append("coordinate_chart_mismatch")
            if int(payload.get("candidate_position_id", -1)) != int(
                record.get("position_id", -1)
            ):
                reasons.append("payload_position_mismatch")
            if str(payload.get("state_fingerprint") or "") != str(
                state_fingerprint
            ):
                reasons.append("state_fingerprint_mismatch")
            if str(payload.get("ordered_scaffold_fingerprint") or "") != str(
                scaffold_fingerprint
            ):
                reasons.append("ordered_scaffold_fingerprint_mismatch")
            if str(payload.get("theta_fingerprint") or "") != str(
                theta_fingerprint
            ):
                reasons.append("theta_fingerprint_mismatch")
            if str(payload.get("hamiltonian_fingerprint") or "") != str(
                hamiltonian_fingerprint
            ):
                reasons.append("hamiltonian_fingerprint_mismatch")
            if float(payload.get("state_reconstruction_delta_norm", math.inf)) > atol:
                reasons.append("state_consistency_tolerance_exceeded")
            candidate_term = record.get("candidate_term")
            expected_candidate_fingerprint = _candidate_coordinate_fingerprint(
                candidate_term,
                position_id=int(record.get("position_id", -1)),
            )
            if str(payload.get("candidate_coordinate_fingerprint") or "") != str(
                expected_candidate_fingerprint
            ):
                reasons.append("candidate_coordinate_fingerprint_mismatch")
            source_active = tuple(
                int(value) for value in payload.get("active_indices", [])
            )
            source_lookup = {
                int(index): int(offset)
                for offset, index in enumerate(source_active)
            }
            if any(index not in source_lookup for index in active):
                reasons.append("active_context_not_available")
            if not reasons:
                offsets = [int(source_lookup[index]) for index in active]
                source_G_AB = np.asarray(payload.get("G_AB", []), dtype=float)
                source_H_AB = np.asarray(payload.get("H_AB", []), dtype=float)
                source_count = int(len(source_active))
                if old_old_geometry_prior is None:
                    source_G_AA = np.asarray(
                        payload.get("G_AA", []), dtype=float
                    )
                    source_H_AA = np.asarray(
                        payload.get("H_AA", []), dtype=float
                    )
                    source_g_A = np.asarray(
                        payload.get("g_A", []), dtype=float
                    )
                    if source_count == 0:
                        source_G_AA = source_G_AA.reshape(0, 0)
                        source_H_AA = source_H_AA.reshape(0, 0)
                    if source_G_AA.shape != (source_count, source_count):
                        reasons.append("G_AA_shape_mismatch")
                    if source_H_AA.shape != (source_count, source_count):
                        reasons.append("H_AA_shape_mismatch")
                    if source_g_A.shape != (source_count,):
                        reasons.append("g_A_shape_mismatch")
                if source_G_AB.shape != (source_count,):
                    reasons.append("G_AB_shape_mismatch")
                if source_H_AB.shape != (source_count,):
                    reasons.append("H_AB_shape_mismatch")
            if not reasons:
                idx = np.asarray(offsets, dtype=int)
                candidate_G_AB = np.asarray(source_G_AB[idx], dtype=float)
                candidate_H_AB = np.asarray(source_H_AB[idx], dtype=float)
                candidate_blocks: list[tuple[str, np.ndarray]] = [
                    ("G_AB", candidate_G_AB),
                    ("H_AB", candidate_H_AB),
                ]
                if old_old_geometry_prior is None:
                    candidate_G_AA = np.asarray(
                        source_G_AA[np.ix_(idx, idx)], dtype=float
                    )
                    candidate_H_AA = np.asarray(
                        source_H_AA[np.ix_(idx, idx)], dtype=float
                    )
                    candidate_g_A = np.asarray(source_g_A[idx], dtype=float)
                    candidate_blocks = [
                        ("G_AA", candidate_G_AA),
                        ("H_AA", candidate_H_AA),
                        ("g_A", candidate_g_A),
                        *candidate_blocks,
                    ]
                for block_name, block_value in candidate_blocks:
                    if not np.all(np.isfinite(np.asarray(block_value))):
                        reasons.append(f"{block_name}_nonfinite")
                if reasons:
                    pass
                elif (
                    old_old_geometry_prior is None
                    and active_reference_initialized
                ):
                    if not _matches(candidate_G_AA, G_AA):
                        reasons.append("G_AA_inconsistent_across_records")
                    if not _matches(candidate_H_AA, H_AA):
                        reasons.append("H_AA_inconsistent_across_records")
                    if not _matches(candidate_g_A, g_A):
                        reasons.append("g_A_inconsistent_across_records")
                elif old_old_geometry_prior is None:
                    G_AA = candidate_G_AA
                    H_AA = candidate_H_AA
                    g_A = candidate_g_A
                    active_reference_initialized = True
                if not reasons:
                    G_AB[:, candidate_index] = candidate_G_AB
                    H_AB[:, candidate_index] = candidate_H_AB
                    G_BB_diagonal[candidate_index] = float(
                        payload.get("G_BB", float("nan"))
                    )
                    H_BB_diagonal[candidate_index] = float(
                        payload.get("H_BB", float("nan"))
                    )
                    if not math.isfinite(G_BB_diagonal[candidate_index]):
                        reasons.append("G_BB_nonfinite")
                    if not math.isfinite(H_BB_diagonal[candidate_index]):
                        reasons.append("H_BB_nonfinite")
                    gradient_value = float(
                        payload.get("descent_gradient", float("nan"))
                    )
                    if math.isfinite(gradient_value):
                        g_B[candidate_index] = gradient_value
                    else:
                        gradient_reasons.append("descent_gradient_nonfinite")
        valid = not reasons
        if valid:
            valid_indices.append(int(candidate_index))
        gradient_valid = bool(valid and not gradient_reasons)
        if gradient_valid:
            valid_gradient_indices.append(int(candidate_index))
        record_results.append(
            {
                "workspace_index": int(candidate_index),
                "valid": bool(valid),
                "reasons": reasons,
                "gradient_valid": bool(gradient_valid),
                "gradient_reasons": gradient_reasons,
            }
        )

    if active_count == 0:
        G_AA = np.zeros((0, 0), dtype=float)
        H_AA = np.zeros((0, 0), dtype=float)
        g_A = np.zeros(0, dtype=float)
        active_reference_initialized = True
    return _Phase2JointGeometryCache(
        active_indices=active,
        G_AA=np.asarray(G_AA, dtype=float),
        H_AA=np.asarray(H_AA, dtype=float),
        g_A=np.asarray(g_A, dtype=float),
        G_AB=np.asarray(G_AB, dtype=float),
        H_AB=np.asarray(H_AB, dtype=float),
        G_BB_diagonal=np.asarray(G_BB_diagonal, dtype=float),
        H_BB_diagonal=np.asarray(H_BB_diagonal, dtype=float),
        g_B=np.asarray(g_B, dtype=float),
        valid_record_indices=tuple(int(index) for index in valid_indices),
        valid_gradient_record_indices=tuple(
            int(index) for index in valid_gradient_indices
        ),
        record_results=record_results,
        active_block_valid=bool(active_reference_initialized),
        state_fingerprint=str(state_fingerprint),
        ordered_scaffold_fingerprint=str(scaffold_fingerprint),
        theta_fingerprint=str(theta_fingerprint),
        state_reconstruction_delta_norm_max=float(
            state_reconstruction_delta_norm_max
        ),
    )


def _batch_joint_active_indices(
    cfg: FullScoreConfig,
    *,
    selected_ops: Sequence[Any],
) -> tuple[int, ...]:
    selected_count = int(len(selected_ops))
    context_mode = normalize_batch_joint_context_mode(
        getattr(cfg, "batch_joint_context_mode", None)
    )
    if context_mode == BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1:
        return tuple(range(selected_count))
    if context_mode == BATCH_JOINT_CONTEXT_BATCH_ONLY_DIAGNOSTIC_V1:
        return ()
    configured_indices = getattr(cfg, "batch_active_context_indices", None)
    if configured_indices is None:
        raise ValueError(
            "active_window_v1 requires explicit batch_active_context_indices."
        )
    active_indices = tuple(int(index) for index in configured_indices)
    if len(set(active_indices)) != len(active_indices):
        raise ValueError("batch_active_context_indices contains duplicates.")
    if any(index < 0 or index >= selected_count for index in active_indices):
        raise ValueError(
            "batch_active_context_indices contains an out-of-range ansatz index."
        )
    return active_indices


def _rank_feasible_child_phase2_population(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    copied = [dict(record) for record in records]
    policy = str(
        getattr(
            cfg,
            "batch_search_feasibility_policy",
            BATCH_SEARCH_FEASIBILITY_RAW_RANKED_LEGACY_V1,
        )
    )
    if policy not in BATCH_SEARCH_FEASIBILITY_POLICIES:
        raise ValueError(
            "batch_search_feasibility_policy must be one of "
            f"{sorted(BATCH_SEARCH_FEASIBILITY_POLICIES)}."
        )
    base_telemetry = {
        "schema": "batch_search_rank_feasibility_prefilter_v1",
        "policy": str(policy),
        "input_record_count": int(len(copied)),
        "phase2_geometry_reuse_only": True,
        "query_chargeable_geometry_element_count": 0,
    }
    if policy in {
        BATCH_SEARCH_FEASIBILITY_JOINT_SUBSET_GATE_V1,
        BATCH_SEARCH_FEASIBILITY_RAW_RANKED_LEGACY_V1,
    } or not copied:
        joint_subset_gate = (
            policy == BATCH_SEARCH_FEASIBILITY_JOINT_SUBSET_GATE_V1
        )
        return copied, {
            **base_telemetry,
            "active": False,
            "reason": (
                "empty_population"
                if not copied
                else (
                    "joint_rank_gate_after_search_pool_v1"
                    if joint_subset_gate
                    else "legacy_raw_ranked_population"
                )
            ),
            "rank_gate_application_stage": (
                "joint_subset_after_search_pool"
                if joint_subset_gate
                else "joint_subset_after_legacy_raw_search_pool"
            ),
            "rank_feasible_record_count": int(len(copied)),
            "rank_rejected_record_count": 0,
            "fail_open_record_count": 0,
            "rejected_records": [],
        }

    active_indices = _batch_joint_active_indices(
        cfg,
        selected_ops=selected_ops,
    )
    tolerance = float(
        max(
            1e-8,
            getattr(cfg, "batch_state_consistency_tolerance", 1e-8),
        )
    )
    cache = _build_phase2_joint_geometry_cache(
        copied,
        active_indices=active_indices,
        selected_ops=selected_ops,
        theta=np.asarray(theta, dtype=float),
        psi_state=np.asarray(psi_state, dtype=complex),
        h_compiled=h_compiled,
        tolerance=tolerance,
    )
    valid_indices = set(int(index) for index in cache.valid_record_indices)
    metric_regularization = float(
        max(0.0, getattr(cfg, "batch_metric_regularization", 1e-9))
    )
    rank_relative_tolerance = float(
        max(0.0, getattr(cfg, "batch_rank_rel_tol", 1e-6))
    )
    active_count = int(len(active_indices))
    active_block_available = bool(cache.active_block_valid)
    if active_count and active_block_available:
        G_AA_regularized = 0.5 * (cache.G_AA + cache.G_AA.T) + float(
            metric_regularization
        ) * np.eye(active_count, dtype=float)
        G_AA_inverse = np.linalg.pinv(
            G_AA_regularized,
            rcond=max(metric_regularization, 1e-15),
        )
    else:
        G_AA_inverse = np.zeros((active_count, active_count), dtype=float)

    retained: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    fail_open_count = 0
    for index, record in enumerate(copied):
        if int(index) not in valid_indices or (
            active_count and not active_block_available
        ):
            retained.append(record)
            fail_open_count += 1
            continue
        raw_gram = float(cache.G_BB_diagonal[int(index)])
        if active_count:
            mixed = np.asarray(cache.G_AB[:, int(index)], dtype=float)
            effective_gram = float(
                raw_gram - mixed.T @ G_AA_inverse @ mixed
            )
        else:
            effective_gram = float(raw_gram)
        raw_gram_scale = float(max(abs(raw_gram), metric_regularization))
        effective_gram_scale = float(
            max(abs(effective_gram), metric_regularization)
        )
        rank_floor = float(
            max(
                10.0 * metric_regularization,
                rank_relative_tolerance * raw_gram_scale,
                rank_relative_tolerance * effective_gram_scale,
            )
        )
        if math.isfinite(effective_gram) and effective_gram > rank_floor:
            retained.append(record)
            continue
        identity_kind, identity_value = _batch_record_generator_identity(record)
        rejected.append(
            {
                "ranked_population_index": int(index),
                "identity_kind": str(identity_kind),
                "identity": str(identity_value),
                "candidate_label": str(_batch_record_label(record)),
                "position_id": int(record.get("position_id", -1)),
                "raw_gram": float(raw_gram),
                "effective_gram": float(effective_gram),
                "rank_floor": float(rank_floor),
            }
        )
    return retained, {
        **base_telemetry,
        "active": True,
        "reason": "hard_singleton_rank_gate_before_search_pool_v1",
        "active_context_indices": [int(index) for index in active_indices],
        "rank_gate_application_stage": "singleton_before_search_pool",
        "phase2_geometry_complete": bool(cache.complete),
        "rank_feasible_record_count": int(len(retained)),
        "rank_rejected_record_count": int(len(rejected)),
        "fail_open_record_count": int(fail_open_count),
        "rejected_records": rejected[:20],
        "rejected_records_truncated": bool(len(rejected) > 20),
    }


def _phase2_joint_geometry_reuse_accounting(
    cache: _Phase2JointGeometryCache,
    *,
    candidate_count: int,
    required_candidate_pairs: Sequence[tuple[int, int]],
    reused_candidate_pair_count: int = 0,
    tolerance: float,
    eager_blocks: Mapping[str, np.ndarray] | None = None,
    old_old_geometry_prior: HistoricalSingletonOldOldGeometryPrior | None = None,
) -> dict[str, Any]:
    active_count = int(len(cache.active_indices))
    atol = float(max(1e-10, tolerance))
    valid = set(int(index) for index in cache.valid_record_indices)
    gradient_valid = set(
        int(index) for index in cache.valid_gradient_record_indices
    )
    records = [dict(row) for row in cache.record_results]

    def _matches(lhs: Any, rhs: Any) -> bool:
        left = np.asarray(lhs, dtype=float)
        right = np.asarray(rhs, dtype=float)
        return bool(
            left.shape == right.shape
            and np.allclose(left, right, atol=atol, rtol=1e-7)
        )

    if eager_blocks is not None:
        active_mismatch = bool(
            not _matches(cache.G_AA, eager_blocks["G_AA"])
            or not _matches(cache.H_AA, eager_blocks["H_AA"])
            or not _matches(cache.g_A, eager_blocks["g_A"])
        )
        if active_mismatch:
            valid.clear()
            gradient_valid.clear()
            for row in records:
                row.setdefault("reasons", []).append(
                    "active_block_eager_parity_mismatch"
                )
                row["valid"] = False
                row["gradient_valid"] = False
        else:
            for index in list(valid):
                block_ok = bool(
                    _matches(cache.G_AB[:, index], eager_blocks["G_AB"][:, index])
                    and _matches(cache.H_AB[:, index], eager_blocks["H_AB"][:, index])
                    and _matches(
                        cache.G_BB_diagonal[index],
                        eager_blocks["G_BB"][index, index],
                    )
                    and _matches(
                        cache.H_BB_diagonal[index],
                        eager_blocks["H_BB"][index, index],
                    )
                )
                if not block_ok:
                    valid.discard(index)
                    gradient_valid.discard(index)
                    records[index].setdefault("reasons", []).append(
                        "candidate_block_eager_parity_mismatch"
                    )
                    records[index]["valid"] = False
                elif index in gradient_valid and not _matches(
                    cache.g_B[index], eager_blocks["g_B"][index]
                ):
                    gradient_valid.discard(index)
                    records[index].setdefault("gradient_reasons", []).append(
                        "descent_gradient_eager_parity_mismatch"
                    )
                    records[index]["gradient_valid"] = False

    active_triangle = int(active_count * (active_count + 1) // 2)
    pair_count = int(len(required_candidate_pairs))
    reused_pair_count = int(
        min(max(0, reused_candidate_pair_count), pair_count)
    )
    full_pair_count = int(candidate_count * (candidate_count - 1) // 2)
    active_reused = bool(
        old_old_geometry_prior is not None
        or (cache.active_block_valid and valid)
    )
    valid_count = int(len(valid))
    required_by_block = {
        "G_AA": int(active_triangle),
        "H_AA": int(active_triangle),
        "G_AC": int(active_count * candidate_count),
        "H_AC": int(active_count * candidate_count),
        "G_CC_diagonal": int(candidate_count),
        "H_CC_diagonal": int(candidate_count),
        "G_CC_off_diagonal": int(pair_count),
        "H_CC_off_diagonal": int(pair_count),
    }
    reused_by_block = {
        "G_AA": int(active_triangle if active_reused else 0),
        "H_AA": int(active_triangle if active_reused else 0),
        "G_AC": int(active_count * valid_count),
        "H_AC": int(active_count * valid_count),
        "G_CC_diagonal": int(valid_count),
        "H_CC_diagonal": int(valid_count),
        "G_CC_off_diagonal": int(reused_pair_count),
        "H_CC_off_diagonal": int(reused_pair_count),
    }
    newly_measured_by_block = {
        key: int(max(0, required_by_block[key] - reused_by_block[key]))
        for key in required_by_block
    }
    required_total = int(sum(required_by_block.values()))
    reused_total = int(sum(reused_by_block.values()))
    newly_measured_total = int(sum(newly_measured_by_block.values()))
    full_eager_total = int(
        (active_count + candidate_count)
        * (active_count + candidate_count + 1)
    )
    invalidation_reasons: dict[str, int] = {}
    for row in records:
        for reason in row.get("reasons", []):
            invalidation_reasons[str(reason)] = int(
                invalidation_reasons.get(str(reason), 0) + 1
            )
    payload = {
        "schema": "phase2_joint_geometry_reuse_validation_v2",
        "status": "validated" if valid_count else "no_valid_reuse",
        "cache_scope": "selector_call_branch_local_v1",
        "state_fingerprint": str(cache.state_fingerprint),
        "ordered_scaffold_fingerprint": str(
            cache.ordered_scaffold_fingerprint
        ),
        "theta_fingerprint": str(cache.theta_fingerprint),
        "comparison_tolerance": float(atol),
        "valid_record_indices": sorted(valid),
        "valid_record_count": int(valid_count),
        "invalid_record_count": int(candidate_count - valid_count),
        "valid_gradient_record_indices": sorted(gradient_valid),
        "valid_gradient_record_count": int(len(gradient_valid)),
        "gradient_repair_record_count": int(
            candidate_count - len(gradient_valid)
        ),
        "query_chargeable_gradient_repair_count": int(
            candidate_count - len(gradient_valid)
        ),
        "records": records,
        "invalidation_reason_counts": invalidation_reasons,
        "required_element_counts": required_by_block,
        "reused_element_counts": reused_by_block,
        "newly_measured_element_counts": newly_measured_by_block,
        "total_mathematically_required_element_count": int(required_total),
        "full_unique_geometry_element_count": int(full_eager_total),
        "reused_active_block_element_count": int(
            reused_by_block["G_AA"] + reused_by_block["H_AA"]
        ),
        "reused_candidate_block_element_count": int(
            reused_total
            - reused_by_block["G_AA"]
            - reused_by_block["H_AA"]
        ),
        "reused_total_element_count": int(reused_total),
        "query_chargeable_unique_geometry_element_count": int(
            newly_measured_total
        ),
        "cache_hit_element_count": int(reused_total),
        "cache_miss_element_count": int(newly_measured_total),
        "required_candidate_pair_count": int(pair_count),
        "constructed_candidate_pair_count": int(pair_count - reused_pair_count),
        "reused_cached_candidate_pair_count": int(reused_pair_count),
        "full_candidate_pair_count": int(full_pair_count),
        "candidate_pair_pruned_before_measurement_count": int(
            full_pair_count - pair_count
        ),
    }
    if old_old_geometry_prior is not None:
        payload["old_old_geometry_prior"] = (
            old_old_geometry_prior.telemetry()
        )
        payload["old_old_geometry_source"] = (
            "formal_manifold_outer_information_prior_v1"
        )
        payload["old_old_geometry_reacquired"] = False
    return payload

def _sr_supported_quotient_summary(
    *,
    full_solve_result: Any,
    shared_support_active_restriction: Mapping[str, Any],
    hessian_cluster_tolerance: float,
) -> dict[str, Any]:
    """Certify minimum-mode quotient participation in the full v2 support.

    This routine deliberately has no raw Gram input.  It can only consume the
    transported metric, Hessian, gradient, and physical active-image basis emitted
    by :func:`_sr_v2_shared_support_active_restriction`.  Consequently it
    cannot silently choose a second support, apply a second ridge, or form a
    coordinate-block pseudoinverse that disagrees with the global trust solve.
    """

    full_telemetry = dict(full_solve_result.telemetry)
    full_provenance = str(
        full_telemetry.get("supported_metric_whitening_provenance_id", "")
    )
    whitened_gradient_norm_raw = full_telemetry.get(
        "supported_gradient_norm"
    )
    whitened_gradient_norm = (
        None
        if whitened_gradient_norm_raw is None
        else float(whitened_gradient_norm_raw)
    )

    def unresolved(reason: str) -> dict[str, Any]:
        return {
            "quotient_geometry_schema": (
                "sr_v2_full_support_transported_quotient_v2"
            ),
            "quotient_geometry_source": (
                "full_v2_shared_support_physical_active_image_transport_v2"
            ),
            "quotient_shared_support_provenance_id": full_provenance,
            "quotient_independent_metric_factorization": False,
            "quotient_independent_metric_pseudoinverse": False,
            "quotient_participation_resolved": False,
            "quotient_participation_reason": str(reason),
            "quotient_participation": 0.0,
            "quotient_participation_lower_bound": 0.0,
            "quotient_participation_tolerance": 0.0,
            "quotient_residual_metric_eigenvalues": [],
            "quotient_redundant_certified": False,
            "whitened_gradient_norm": whitened_gradient_norm,
        }

    if str(
        full_telemetry.get("joint_linear_solve_policy_effective", "")
    ) != JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2:
        return unresolved("quotient_requires_global_trust_v2")
    if str(full_telemetry.get("raw_metric_support_status", "unresolved")) != (
        "resolved"
    ):
        return unresolved(
            str(
                full_telemetry.get(
                    "raw_metric_support_reason", "raw_metric_support_unresolved"
                )
            )
        )
    if not bool(
        full_telemetry.get("raw_metric_null_compatibility_certified", False)
    ):
        return unresolved(
            str(
                full_telemetry.get(
                    "raw_metric_null_compatibility_reason",
                    "raw_metric_null_compatibility_unresolved",
                )
            )
        )
    if bool(
        shared_support_active_restriction.get(
            "active_restriction_independent_metric_factorization", True
        )
    ):
        return unresolved("active_restriction_did_not_reuse_full_v2_support")
    transport = shared_support_active_restriction.get(
        "shared_support_quotient_transport"
    )
    if not isinstance(transport, Mapping):
        return unresolved("shared_support_quotient_transport_missing")
    transport_provenance = str(
        transport.get("full_supported_metric_whitening_provenance_id", "")
    )
    active_provenance = str(
        shared_support_active_restriction.get(
            "full_supported_metric_whitening_provenance_id", ""
        )
    )
    if (
        not full_provenance
        or active_provenance != full_provenance
        or transport_provenance != full_provenance
    ):
        return unresolved("shared_support_quotient_provenance_mismatch")
    if str(transport.get("raw_metric_support_status", "unresolved")) != (
        "resolved"
    ) or not bool(
        transport.get("raw_metric_null_compatibility_certified", False)
    ):
        return unresolved("shared_support_quotient_transport_uncertified")

    M = np.asarray(
        transport.get("raw_metric_in_supported_coordinates", []), dtype=float
    )
    H_w = np.asarray(
        transport.get("global_trust_hessian_supported_coordinates", []),
        dtype=float,
    )
    g_w = np.asarray(
        transport.get("global_trust_gradient_supported_coordinates", []),
        dtype=float,
    ).reshape(-1)
    active_image_basis = np.asarray(
        transport.get(
            "active_restriction_basis_supported_coordinates", []
        ),
        dtype=float,
    )
    supported_rank = int(transport.get("supported_rank", -1))
    batch_count = int(transport.get("batch_coordinate_count", -1))
    if active_image_basis.ndim == 1 and active_image_basis.size == 0:
        active_image_basis = np.zeros((supported_rank, 0), dtype=float)
    if (
        supported_rank <= 0
        or M.shape != (supported_rank, supported_rank)
        or H_w.shape != (supported_rank, supported_rank)
        or g_w.size != supported_rank
        or active_image_basis.ndim != 2
        or active_image_basis.shape[0] != supported_rank
        or batch_count < 0
        or not all(
            np.all(np.isfinite(value))
            for value in (M, H_w, g_w, active_image_basis)
        )
    ):
        return unresolved("shared_support_quotient_transport_malformed")
    if whitened_gradient_norm is None:
        whitened_gradient_norm = float(np.linalg.norm(g_w))
    image_residual = float(
        transport.get("active_restriction_image_projection_residual", math.inf)
    )
    image_tolerance = float(
        max(
            0.0,
            transport.get("active_restriction_image_projection_tolerance", 0.0),
        )
    )
    if (
        not math.isfinite(image_residual)
        or image_residual > image_tolerance
    ):
        return unresolved("active_restriction_physical_image_unresolved")
    try:
        active_image_subspace_rotation_bound = float(
            transport.get("active_image_subspace_rotation_bound", math.inf)
        )
    except (TypeError, ValueError):
        active_image_subspace_rotation_bound = math.inf
    if (
        not bool(
            transport.get(
                "active_image_subspace_rotation_certified", False
            )
        )
        or not math.isfinite(active_image_subspace_rotation_bound)
        or active_image_subspace_rotation_bound < 0.0
        or active_image_subspace_rotation_bound >= 1.0
    ):
        return unresolved("active_image_subspace_rotation_unresolved")

    M = 0.5 * (M + M.T)
    H_w = 0.5 * (H_w + H_w.T)
    machine_factor = float(
        4096.0 * np.finfo(float).eps * max(1, supported_rank)
    )
    metric_scale = float(max(1.0, np.linalg.norm(M, ord=2)))
    metric_resolution = float(machine_factor * metric_scale)
    try:
        metric_cholesky = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return unresolved("transported_raw_metric_not_positive_definite")
    metric_minimum = float(np.min(np.linalg.eigvalsh(M)))
    if metric_minimum <= metric_resolution:
        return unresolved("transported_raw_metric_resolution_unresolved")

    active_subspace_dimension = int(active_image_basis.shape[1])
    if active_subspace_dimension:
        active_orthogonality_residual = float(
            np.linalg.norm(
                active_image_basis.T @ active_image_basis
                - np.eye(active_subspace_dimension, dtype=float),
                ord=2,
            )
        )
        active_orthogonality_tolerance = float(machine_factor)
        if active_orthogonality_residual > active_orthogonality_tolerance:
            return unresolved("active_restriction_basis_not_orthonormal")
        active_metric = 0.5 * (
            active_image_basis.T @ M @ active_image_basis
            + active_image_basis.T @ M.T @ active_image_basis
        )
        try:
            np.linalg.cholesky(active_metric)
            active_projection_coefficients = np.linalg.solve(
                active_metric, active_image_basis.T @ M
            )
        except np.linalg.LinAlgError:
            return unresolved("active_restriction_metric_unresolved")
        quotient_metric = 0.5 * (
            M
            - M @ active_image_basis @ active_projection_coefficients
            + (
                M
                - M @ active_image_basis @ active_projection_coefficients
            ).T
        )
    else:
        active_orthogonality_residual = 0.0
        active_orthogonality_tolerance = float(machine_factor)
        quotient_metric = M.copy()

    try:
        left_normalized_quotient = np.linalg.solve(
            metric_cholesky, quotient_metric
        )
        normalized_quotient = np.linalg.solve(
            metric_cholesky, left_normalized_quotient.T
        ).T
        quotient_eigenvalues = np.linalg.eigvalsh(
            0.5 * (normalized_quotient + normalized_quotient.T)
        )
    except np.linalg.LinAlgError:
        return unresolved("shared_support_quotient_normalization_failed")

    support_rotation_bound = float(
        max(0.0, transport.get("raw_metric_support_rotation_bound", 0.0))
    )
    whitening_error_bound = float(
        max(
            0.0,
            transport.get(
                "raw_metric_support_relative_whitening_error_bound", 0.0
            ),
        )
    )
    support_geometry_error_bound = float(
        4.0
        * (
            support_rotation_bound
            + whitening_error_bound
            + active_image_subspace_rotation_bound
        )
    )
    if support_geometry_error_bound >= 1.0:
        return {
            **unresolved(
                "quotient_resolution_saturated_by_transport_uncertainty"
            ),
            "quotient_support_rotation_bound": support_rotation_bound,
            "quotient_relative_whitening_error_bound": (
                whitening_error_bound
            ),
            "quotient_active_image_subspace_rotation_bound": (
                active_image_subspace_rotation_bound
            ),
            "quotient_support_geometry_error_bound": (
                support_geometry_error_bound
            ),
        }
    quotient_arithmetic_error_bound = float(
        machine_factor
        * max(
            1.0,
            np.linalg.norm(normalized_quotient, ord=2),
        )
    )
    quotient_resolution_floor = float(
        support_geometry_error_bound + quotient_arithmetic_error_bound
    )
    if quotient_resolution_floor >= 1.0:
        return {
            **unresolved(
                "quotient_resolution_saturated_by_transport_uncertainty"
            ),
            "quotient_support_rotation_bound": support_rotation_bound,
            "quotient_relative_whitening_error_bound": (
                whitening_error_bound
            ),
            "quotient_active_image_subspace_rotation_bound": (
                active_image_subspace_rotation_bound
            ),
            "quotient_support_geometry_error_bound": (
                support_geometry_error_bound
            ),
            "quotient_arithmetic_error_bound": (
                quotient_arithmetic_error_bound
            ),
            "quotient_resolution_floor": quotient_resolution_floor,
        }
    minimum_quotient_eigenvalue = float(quotient_eigenvalues[0])
    if minimum_quotient_eigenvalue < -quotient_resolution_floor:
        return unresolved("shared_support_quotient_not_positive_semidefinite")
    maximum_quotient_eigenvalue = float(
        max(0.0, quotient_eigenvalues[-1])
    )

    if batch_count <= 0:
        return {
            "quotient_geometry_schema": (
                "sr_v2_full_support_transported_quotient_v2"
            ),
            "quotient_geometry_source": (
                "full_v2_shared_support_physical_active_image_transport_v2"
            ),
            "quotient_shared_support_provenance_id": full_provenance,
            "quotient_independent_metric_factorization": False,
            "quotient_independent_metric_pseudoinverse": False,
            "quotient_participation_resolved": True,
            "quotient_participation_reason": "no_candidate_coordinates",
            "quotient_participation": 0.0,
            "quotient_participation_lower_bound": 0.0,
            "quotient_participation_tolerance": 0.0,
            "quotient_residual_metric_eigenvalues": [
                float(value) for value in quotient_eigenvalues.tolist()
            ],
            "quotient_resolution_floor": quotient_resolution_floor,
            "quotient_active_image_subspace_rotation_bound": (
                active_image_subspace_rotation_bound
            ),
            "quotient_redundant_certified": True,
            "whitened_gradient_norm": whitened_gradient_norm,
        }

    try:
        h_values, h_vectors = np.linalg.eigh(H_w)
    except np.linalg.LinAlgError:
        return unresolved("global_trust_hessian_eigendecomposition_failed")
    hessian_scale = float(max(1.0, np.max(np.abs(h_values))))
    cluster_tolerance = float(
        max(
            0.0,
            float(hessian_cluster_tolerance),
            256.0
            * np.finfo(float).eps
            * max(1, supported_rank)
            * hessian_scale,
        )
    )
    minimum = float(h_values[0])
    minimum_mask = np.asarray(
        np.abs(h_values - minimum) <= cluster_tolerance,
        dtype=bool,
    )
    minimum_cluster_dimension = int(np.count_nonzero(minimum_mask))
    hessian_error_bound = float(
        max(
            0.0,
            transport.get("supported_hessian_propagated_error_bound", 0.0),
        )
    )
    if minimum_cluster_dimension < supported_rank:
        cluster_gap = float(
            h_values[minimum_cluster_dimension] - h_values[0]
        )
        if cluster_gap <= 2.0 * hessian_error_bound:
            return unresolved("minimum_mode_cluster_separation_unresolved")
        hessian_rotation_bound = float(
            hessian_error_bound / (cluster_gap - hessian_error_bound)
        )
    else:
        cluster_gap = math.inf
        hessian_rotation_bound = 0.0

    minimum_modes = np.asarray(h_vectors[:, minimum_mask], dtype=float)
    numerator = 0.5 * (
        minimum_modes.T @ quotient_metric @ minimum_modes
        + minimum_modes.T @ quotient_metric.T @ minimum_modes
    )
    denominator = 0.5 * (
        minimum_modes.T @ M @ minimum_modes
        + minimum_modes.T @ M.T @ minimum_modes
    )
    try:
        denominator_cholesky = np.linalg.cholesky(denominator)
        left_normalized = np.linalg.solve(denominator_cholesky, numerator)
        normalized_minimum_quotient = np.linalg.solve(
            denominator_cholesky, left_normalized.T
        ).T
        normalized_minimum_quotient = 0.5 * (
            normalized_minimum_quotient + normalized_minimum_quotient.T
        )
        participation_values, participation_vectors = np.linalg.eigh(
            normalized_minimum_quotient
        )
        maximizing_coefficients = np.linalg.solve(
            denominator_cholesky.T, participation_vectors[:, -1]
        )
    except np.linalg.LinAlgError:
        return unresolved("minimum_mode_quotient_metric_unresolved")

    participation_sq = float(
        min(1.0, max(0.0, participation_values[-1]))
    )
    participation = float(math.sqrt(participation_sq))
    constructive_numerator = float(
        maximizing_coefficients.T @ numerator @ maximizing_coefficients
    )
    constructive_denominator = float(
        maximizing_coefficients.T @ denominator @ maximizing_coefficients
    )
    if constructive_denominator <= metric_resolution:
        return unresolved("minimum_mode_constructive_denominator_unresolved")
    constructive_ratio = float(
        min(
            1.0,
            max(0.0, constructive_numerator / constructive_denominator),
        )
    )
    minimum_mode_rotation_error_bound = float(
        min(1.0, 4.0 * max(0.0, hessian_rotation_bound))
    )
    participation_sq_error_bound = float(
        min(
            1.0,
            quotient_resolution_floor + minimum_mode_rotation_error_bound,
        )
    )
    participation_sq_lower_bound = float(
        max(0.0, constructive_ratio - participation_sq_error_bound)
    )
    participation_lower_bound = float(
        math.sqrt(participation_sq_lower_bound)
    )
    participation_tolerance = float(
        max(0.0, participation - participation_lower_bound)
    )
    return {
        "quotient_geometry_schema": (
            "sr_v2_full_support_transported_quotient_v2"
        ),
        "quotient_geometry_source": (
            "full_v2_shared_support_physical_active_image_transport_v2"
        ),
        "quotient_shared_support_provenance_id": full_provenance,
        "quotient_independent_metric_factorization": False,
        "quotient_independent_metric_pseudoinverse": False,
        "quotient_supported_rank": supported_rank,
        "quotient_active_subspace_dimension": active_subspace_dimension,
        "quotient_active_basis_orthogonality_residual": (
            active_orthogonality_residual
        ),
        "quotient_active_basis_orthogonality_tolerance": (
            active_orthogonality_tolerance
        ),
        "quotient_support_rotation_bound": support_rotation_bound,
        "quotient_relative_whitening_error_bound": whitening_error_bound,
        "quotient_active_image_subspace_rotation_bound": (
            active_image_subspace_rotation_bound
        ),
        "quotient_support_geometry_error_bound": (
            support_geometry_error_bound
        ),
        "quotient_arithmetic_error_bound": quotient_arithmetic_error_bound,
        "quotient_minimum_mode_cluster_dimension": minimum_cluster_dimension,
        "quotient_minimum_mode_cluster_gap": (
            None if math.isinf(cluster_gap) else cluster_gap
        ),
        "quotient_minimum_mode_rotation_bound": hessian_rotation_bound,
        "quotient_participation_resolved": True,
        "quotient_participation_reason": (
            "full_v2_shared_support_transported_physical_active_quotient_v2"
        ),
        "quotient_participation": participation,
        "quotient_participation_sq_constructive_rayleigh": (
            constructive_ratio
        ),
        "quotient_participation_sq_error_bound": (
            participation_sq_error_bound
        ),
        "quotient_participation_lower_bound": participation_lower_bound,
        "quotient_participation_tolerance": participation_tolerance,
        "quotient_participation_lower_bound_semantics": (
            "constructive_shared_support_rayleigh_minus_transport_and_rotation_bounds"
        ),
        "quotient_residual_metric_eigenvalues": [
            float(value) for value in quotient_eigenvalues.tolist()
        ],
        "quotient_resolution_floor": quotient_resolution_floor,
        "quotient_redundant_certified": bool(
            maximum_quotient_eigenvalue <= quotient_resolution_floor
        ),
        "whitened_gradient_norm": whitened_gradient_norm,
    }


def _sr_trust_gain_numerical_error_bound(
    *,
    gain: float,
    telemetry: Mapping[str, Any],
    model_scale: float,
    dimension: int,
    radius: float,
    energy_regularization: float,
    direct_objective_residual: float = 0.0,
    subspace_transport_error_bound: float = 0.0,
) -> float:
    """Return a conservative objective-error budget for a certified trust solve."""

    stationarity_tolerance = float(
        max(0.0, telemetry.get("trust_kkt_stationarity_tolerance", 0.0))
    )
    objective_identity_tolerance = float(
        max(
            0.0,
            telemetry.get("trust_kkt_objective_identity_tolerance", 0.0),
        )
    )
    arithmetic_budget = float(
        4096.0
        * np.finfo(float).eps
        * max(1, int(dimension))
        * max(1.0, abs(float(gain)), abs(float(model_scale)))
    )
    return float(
        max(0.0, float(energy_regularization))
        + objective_identity_tolerance
        + stationarity_tolerance * max(1.0, float(radius))
        + max(0.0, float(direct_objective_residual))
        + max(0.0, float(subspace_transport_error_bound))
        + arithmetic_budget
    )


def _sr_v2_shared_support_active_restriction(
    *,
    gram: np.ndarray,
    hessian: np.ndarray,
    gradient: np.ndarray,
    active_count: int,
    full_solve_result: Any,
    rank_relative_tolerance: float,
    energy_regularization: float,
    metric_regularization: float,
    max_fubini_study_step: float,
) -> dict[str, Any]:
    """Solve the physical active image inside the full v2 trust coordinates.

    The full solve has already selected a stable raw-Gram support and formed
    its regularized supported-metric whitening.  The active comparison must
    not make another support decision from ``G_AA``.  Instead this routine
    reconstructs that exact full whitening and transports the physical image
    of the raw active-coordinate subspace into it.  A supported representative
    may have nonzero batch coordinates when the discarded raw-metric nullspace
    mixes active and batch axes; the returned parameter representative is
    therefore mapped back through the certified null quotient and has an
    exactly zero batch block.
    """

    G = np.asarray(gram, dtype=float)
    H = np.asarray(hessian, dtype=float)
    g = np.asarray(gradient, dtype=float).reshape(-1)
    active = int(active_count)
    dimension = int(g.size)
    batch_count = int(dimension - active)
    telemetry = dict(full_solve_result.telemetry)
    applied_metric_ridge = float(
        telemetry.get("metric_whitening_ridge", math.nan)
    )
    source_provenance = str(
        telemetry.get("supported_metric_whitening_provenance_id", "")
    )
    base: dict[str, Any] = {
        "schema": "sr_v2_shared_support_active_restriction_v2",
        "active_restriction_source": (
            "full_v2_supported_metric_physical_active_image_transport_v2"
        ),
        "active_restriction_constraint": (
            "physical_active_coordinate_image_modulo_certified_raw_metric_null"
        ),
        "active_restriction_uses_full_support_decision": True,
        "active_restriction_independent_metric_factorization": False,
        "active_restriction_independent_metric_pseudoinverse": False,
        "active_restriction_metric_ridge_reapplied": False,
        "active_restriction_trust_metric": (
            "identity_in_full_regularized_supported_metric_whitening_coordinates"
        ),
        "full_supported_metric_whitening_provenance_id": source_provenance,
        "full_metric_retained_mask": list(
            telemetry.get("metric_retained_mask", [])
        ),
        "full_metric_support_rank": telemetry.get("metric_support_rank"),
        "full_metric_regularization": applied_metric_ridge,
        "full_configured_legacy_metric_regularization": float(
            metric_regularization
        ),
        "active_coordinate_count": active,
        "batch_coordinate_count": batch_count,
    }
    raw_support_status = str(
        telemetry.get("raw_metric_support_status", "unresolved")
    )
    if raw_support_status != "resolved":
        return {
            **base,
            "valid": False,
            "reason": str(
                telemetry.get(
                    "raw_metric_support_reason",
                    "physical_active_image_requires_resolved_raw_metric_support",
                )
            ),
            "predicted_reduction": 0.0,
            "trust_global_optimality_certified": False,
        }
    if not bool(
        telemetry.get("raw_metric_null_compatibility_certified", False)
    ):
        return {
            **base,
            "valid": False,
            "reason": (
                "physical_active_image_requires_raw_metric_null_compatibility"
            ),
            "raw_metric_null_compatibility_reason": str(
                telemetry.get(
                    "raw_metric_null_compatibility_reason",
                    "raw_metric_null_compatibility_unresolved",
                )
            ),
            "predicted_reduction": 0.0,
            "trust_global_optimality_certified": False,
        }
    raw_eigenvalues_reference = np.asarray(
        telemetry.get("raw_metric_eigenvalues", []), dtype=float
    )
    retained_mask = np.asarray(
        telemetry.get("metric_retained_mask", []), dtype=bool
    )
    whitening_denominators = np.asarray(
        telemetry.get("whitening_denominators", []), dtype=float
    )
    if (
        raw_eigenvalues_reference.size != dimension
        or retained_mask.size != dimension
        or whitening_denominators.size != int(np.count_nonzero(retained_mask))
        or not math.isfinite(applied_metric_ridge)
        or not source_provenance
    ):
        return {
            **base,
            "valid": False,
            "reason": "full_v2_support_telemetry_incomplete",
            "predicted_reduction": 0.0,
            "trust_global_optimality_certified": False,
        }

    try:
        raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(
            0.5 * (G + G.T)
        )
    except np.linalg.LinAlgError:
        return {
            **base,
            "valid": False,
            "reason": "shared_support_reconstruction_failed",
            "predicted_reduction": 0.0,
            "trust_global_optimality_certified": False,
        }
    raw_scale = float(
        max(
            1.0,
            np.max(np.abs(raw_eigenvalues)) if raw_eigenvalues.size else 0.0,
        )
    )
    reconstruction_tolerance = float(
        max(
            float(telemetry.get("raw_metric_support_epsilon_G", 0.0)),
            4096.0
            * np.finfo(float).eps
            * max(1, dimension)
            * raw_scale,
        )
    )
    eigenvalue_residual = float(
        np.max(np.abs(raw_eigenvalues - raw_eigenvalues_reference))
        if dimension
        else 0.0
    )
    expected_denominators = np.asarray(
        raw_eigenvalues[retained_mask] + applied_metric_ridge,
        dtype=float,
    )
    denominator_residual = float(
        np.max(np.abs(expected_denominators - whitening_denominators))
        if whitening_denominators.size
        else 0.0
    )
    if (
        eigenvalue_residual > reconstruction_tolerance
        or denominator_residual > reconstruction_tolerance
        or np.any(whitening_denominators <= 0.0)
    ):
        return {
            **base,
            "valid": False,
            "reason": "full_v2_support_reconstruction_mismatch",
            "predicted_reduction": 0.0,
            "trust_global_optimality_certified": False,
            "support_reconstruction_tolerance": reconstruction_tolerance,
            "support_eigenvalue_residual": eigenvalue_residual,
            "whitening_denominator_residual": denominator_residual,
        }

    retained_vectors = np.asarray(raw_eigenvectors[:, retained_mask], dtype=float)
    whitening = np.asarray(
        retained_vectors @ np.diag(whitening_denominators ** -0.5),
        dtype=float,
    )
    supported_rank = int(whitening.shape[1])
    H_w = 0.5 * (whitening.T @ H @ whitening + whitening.T @ H.T @ whitening)
    g_w = np.asarray(whitening.T @ g, dtype=float)
    raw_metric_in_supported_coordinates = 0.5 * (
        whitening.T @ G @ whitening + whitening.T @ G.T @ whitening
    )
    active_embedding = np.zeros((dimension, active), dtype=float)
    if active:
        active_embedding[:active, :] = np.eye(active, dtype=float)
    active_image_transport = np.asarray(
        np.diag(np.sqrt(whitening_denominators))
        @ retained_vectors[:active, :].T,
        dtype=float,
    )
    try:
        image_u, image_singular_values, image_vh = np.linalg.svd(
            active_image_transport,
            full_matrices=False,
        )
    except np.linalg.LinAlgError:
        return {
            **base,
            "valid": False,
            "reason": "physical_active_image_svd_failed",
            "predicted_reduction": 0.0,
            "trust_global_optimality_certified": False,
        }
    active_image_scale = float(
        max(
            1.0,
            image_singular_values[0]
            if image_singular_values.size
            else 0.0,
        )
    )
    support_rotation_bound = float(
        max(0.0, telemetry.get("raw_metric_support_rotation_bound", 0.0))
    )
    support_whitening_error_bound = float(
        max(
            0.0,
            telemetry.get(
                "raw_metric_support_relative_whitening_error_bound", 0.0
            ),
        )
    )
    active_image_arithmetic_error_bound = float(
        4096.0
        * np.finfo(float).eps
        * max(1, dimension, supported_rank, active)
        * active_image_scale
    )
    active_image_support_error_bound = float(
        active_image_scale
        * min(1.0, support_rotation_bound + support_whitening_error_bound)
    )
    active_image_resolution = float(
        active_image_support_error_bound + active_image_arithmetic_error_bound
    )
    expected_active_image_rank = int(min(supported_rank, active))
    active_image_rank = int(
        np.count_nonzero(image_singular_values > active_image_resolution)
    )
    if active_image_rank != expected_active_image_rank:
        return {
            **base,
            "valid": False,
            "reason": "physical_active_image_rank_unresolved",
            "predicted_reduction": 0.0,
            "trust_global_optimality_certified": False,
            "active_image_singular_values": [
                float(value) for value in image_singular_values.tolist()
            ],
            "active_image_rank": active_image_rank,
            "active_image_expected_full_rank": expected_active_image_rank,
            "active_image_resolution": active_image_resolution,
            "active_image_support_error_bound": (
                active_image_support_error_bound
            ),
            "active_image_arithmetic_error_bound": (
                active_image_arithmetic_error_bound
            ),
        }
    minimum_active_image_singular_value = (
        float(image_singular_values[expected_active_image_rank - 1])
        if expected_active_image_rank
        else None
    )
    active_image_singular_gap_lower_bound = (
        None
        if minimum_active_image_singular_value is None
        else float(
            max(
                0.0,
                minimum_active_image_singular_value
                - active_image_resolution,
            )
        )
    )
    if (
        minimum_active_image_singular_value is not None
        and minimum_active_image_singular_value
        <= 2.0 * active_image_resolution
    ):
        return {
            **base,
            "valid": False,
            "reason": "physical_active_image_subspace_rotation_unresolved",
            "predicted_reduction": 0.0,
            "trust_global_optimality_certified": False,
            "active_image_singular_values": [
                float(value) for value in image_singular_values.tolist()
            ],
            "active_image_rank": active_image_rank,
            "active_image_expected_full_rank": expected_active_image_rank,
            "active_image_resolution": active_image_resolution,
            "active_image_support_error_bound": (
                active_image_support_error_bound
            ),
            "active_image_arithmetic_error_bound": (
                active_image_arithmetic_error_bound
            ),
            "active_image_minimum_retained_singular_value": (
                minimum_active_image_singular_value
            ),
            "active_image_singular_gap_lower_bound": (
                active_image_singular_gap_lower_bound
            ),
            "active_image_subspace_rotation_certified": False,
            "active_image_subspace_rotation_bound": None,
            "active_image_subspace_rotation_bound_semantics": (
                "wedin_transport_error_over_singular_gap_v1"
            ),
        }
    active_image_subspace_rotation_bound = (
        0.0
        if minimum_active_image_singular_value is None
        else float(
            active_image_resolution
            / (
                minimum_active_image_singular_value
                - active_image_resolution
            )
        )
    )
    active_image_basis = np.asarray(
        image_u[:, :active_image_rank], dtype=float
    )
    if active_image_rank:
        active_coefficient_transport = np.asarray(
            image_vh[:active_image_rank, :].T
            @ np.diag(image_singular_values[:active_image_rank] ** -1.0)
            @ active_image_basis.T,
            dtype=float,
        )
    else:
        active_coefficient_transport = np.zeros(
            (active, supported_rank), dtype=float
        )
    active_image_projection_residual = float(
        np.linalg.norm(
            active_image_transport
            - active_image_basis
            @ (active_image_basis.T @ active_image_transport),
            ord=2,
        )
        if active_image_transport.size
        else 0.0
    )
    active_image_transport_residual = float(
        np.linalg.norm(
            active_image_transport
            @ active_coefficient_transport
            @ active_image_transport
            - active_image_transport,
            ord=2,
        )
        if active_image_transport.size
        else 0.0
    )
    canonical_active_representatives = np.asarray(
        whitening @ active_image_transport, dtype=float
    )
    active_image_metric_null_residual = float(
        np.linalg.norm(
            G @ (canonical_active_representatives - active_embedding),
            ord=2,
        )
        if active
        else 0.0
    )
    active_image_projection_tolerance = float(
        max(
            active_image_resolution,
            reconstruction_tolerance
            * max(1.0, np.linalg.norm(active_image_transport, ord=2)),
        )
    )
    active_image_metric_null_tolerance = float(
        max(
            reconstruction_tolerance
            * max(1.0, np.linalg.norm(G, ord=2), active_image_scale),
            active_image_arithmetic_error_bound
            * max(1.0, np.linalg.norm(G, ord=2)),
        )
    )
    if (
        active_image_projection_residual > active_image_projection_tolerance
        or active_image_transport_residual > active_image_projection_tolerance
        or active_image_metric_null_residual
        > active_image_metric_null_tolerance
    ):
        return {
            **base,
            "valid": False,
            "reason": "physical_active_image_transport_unresolved",
            "predicted_reduction": 0.0,
            "trust_global_optimality_certified": False,
            "active_image_projection_residual": (
                active_image_projection_residual
            ),
            "active_image_transport_residual": active_image_transport_residual,
            "active_image_projection_tolerance": (
                active_image_projection_tolerance
            ),
            "active_image_metric_null_residual": (
                active_image_metric_null_residual
            ),
            "active_image_metric_null_tolerance": (
                active_image_metric_null_tolerance
            ),
        }
    restricted_dimension = int(active_image_basis.shape[1])
    projector = np.asarray(
        active_image_basis @ active_image_basis.T, dtype="<f8"
    )
    restriction_digest = hashlib.sha256()
    restriction_digest.update(
        b"sr_v2_shared_support_active_restriction_v2\0"
    )
    restriction_digest.update(source_provenance.encode("ascii"))
    restriction_digest.update(b"\0")
    restriction_digest.update(str(active).encode("ascii"))
    restriction_digest.update(b"\0")
    restriction_digest.update(np.ascontiguousarray(projector).tobytes())
    restriction_provenance = restriction_digest.hexdigest()
    quotient_transport = {
        "schema": "sr_v2_shared_support_quotient_transport_v2",
        "geometry_source": (
            "full_v2_supported_metric_whitening_and_physical_active_image_v2"
        ),
        "full_supported_metric_whitening_provenance_id": source_provenance,
        "independent_metric_factorization": False,
        "independent_metric_pseudoinverse": False,
        "raw_metric_support_status": str(
            telemetry.get("raw_metric_support_status", "unresolved")
        ),
        "raw_metric_null_compatibility_certified": bool(
            telemetry.get("raw_metric_null_compatibility_certified", False)
        ),
        "raw_metric_support_rotation_bound": float(
            max(0.0, telemetry.get("raw_metric_support_rotation_bound", 0.0))
        ),
        "raw_metric_support_relative_whitening_error_bound": float(
            max(
                0.0,
                telemetry.get(
                    "raw_metric_support_relative_whitening_error_bound", 0.0
                ),
            )
        ),
        "supported_hessian_propagated_error_bound": float(
            max(
                0.0,
                telemetry.get("supported_hessian_propagated_error_bound", 0.0),
            )
        ),
        "supported_rank": supported_rank,
        "active_coordinate_count": active,
        "batch_coordinate_count": batch_count,
        "raw_metric_in_supported_coordinates": (
            raw_metric_in_supported_coordinates.tolist()
        ),
        "global_trust_hessian_supported_coordinates": H_w.tolist(),
        "global_trust_gradient_supported_coordinates": [
            float(value) for value in g_w.tolist()
        ],
        "active_restriction_basis_supported_coordinates": (
            active_image_basis.tolist()
        ),
        "active_restriction_image_projection_residual": (
            active_image_projection_residual
        ),
        "active_restriction_image_projection_tolerance": (
            active_image_projection_tolerance
        ),
        "active_image_minimum_retained_singular_value": (
            minimum_active_image_singular_value
        ),
        "active_image_singular_gap_lower_bound": (
            active_image_singular_gap_lower_bound
        ),
        "active_image_subspace_rotation_certified": True,
        "active_image_subspace_rotation_bound": (
            active_image_subspace_rotation_bound
        ),
        "active_image_subspace_rotation_bound_semantics": (
            "wedin_transport_error_over_singular_gap_v1"
        ),
        "active_restriction_provenance_id": restriction_provenance,
    }
    active_image_subspace_gain_error_bound = float(
        2.0
        * float(max_fubini_study_step)
        * active_image_subspace_rotation_bound
        * (
            np.linalg.norm(g_w)
            + float(max_fubini_study_step) * np.linalg.norm(H_w, ord=2)
        )
    )
    shared_payload = {
        **base,
        "support_reconstruction_tolerance": reconstruction_tolerance,
        "support_eigenvalue_residual": eigenvalue_residual,
        "whitening_denominator_residual": denominator_residual,
        "active_image_singular_values": [
            float(value) for value in image_singular_values.tolist()
        ],
        "active_image_rank": active_image_rank,
        "active_image_expected_full_rank": expected_active_image_rank,
        "active_image_resolution": active_image_resolution,
        "active_image_support_error_bound": active_image_support_error_bound,
        "active_image_arithmetic_error_bound": (
            active_image_arithmetic_error_bound
        ),
        "active_image_minimum_retained_singular_value": (
            minimum_active_image_singular_value
        ),
        "active_image_singular_gap_lower_bound": (
            active_image_singular_gap_lower_bound
        ),
        "active_image_subspace_rotation_certified": True,
        "active_image_subspace_rotation_bound": (
            active_image_subspace_rotation_bound
        ),
        "active_image_subspace_rotation_bound_semantics": (
            "wedin_transport_error_over_singular_gap_v1"
        ),
        "active_image_subspace_gain_error_bound": (
            active_image_subspace_gain_error_bound
        ),
        "active_image_subspace_gain_error_bound_semantics": (
            "two_radius_times_rotation_times_gradient_plus_radius_hessian_v1"
        ),
        "active_image_projection_residual": active_image_projection_residual,
        "active_image_projection_tolerance": (
            active_image_projection_tolerance
        ),
        "active_image_transport_residual": active_image_transport_residual,
        "active_image_metric_null_residual": active_image_metric_null_residual,
        "active_image_metric_null_tolerance": (
            active_image_metric_null_tolerance
        ),
        "active_coordinate_transport_solver": (
            "svd_full_rank_physical_active_image_v2"
        ),
        "active_coordinate_transport_is_metric_pseudoinverse": False,
        "active_restriction_supported_dimension": restricted_dimension,
        # Backward-compatible field name; v2 semantics are the physical active
        # image dimension, not the nullity of a lifted batch-row constraint.
        "active_restriction_supported_nullity": restricted_dimension,
        "active_restriction_supported_nullity_semantics": (
            "legacy_alias_of_physical_active_image_dimension_v2"
        ),
        "active_restriction_provenance_id": restriction_provenance,
        "shared_support_quotient_transport": quotient_transport,
        "full_whitened_model_scale": float(
            np.linalg.norm(g_w) * float(max_fubini_study_step)
            + 0.5
            * np.linalg.norm(H_w, ord=2)
            * float(max_fubini_study_step) ** 2
        ),
    }

    if restricted_dimension == 0:
        active_gain = 0.0
        active_step = np.zeros(dimension, dtype=float)
        active_valid = bool(
            active_image_projection_residual
            <= active_image_projection_tolerance
            and active_image_metric_null_residual
            <= active_image_metric_null_tolerance
        )
        return {
            **shared_payload,
            "valid": active_valid,
            "reason": "zero_dimensional_active_restriction",
            "feasible": True,
            "predicted_reduction": active_gain,
            "applied_predicted_reduction": active_gain,
            "joint_step": [float(value) for value in active_step.tolist()],
            "active_parameter_relaxation": [
                float(value) for value in active_step[:active].tolist()
            ],
            "batch_coordinate_step": [
                float(value) for value in active_step[active:].tolist()
            ],
            "fubini_study_displacement_sq": 0.0,
            "joint_fubini_study_displacement_sq": 0.0,
            "trust_global_optimality_certified": active_valid,
            "active_restriction_direct_objective_residual": 0.0,
            "active_restriction_batch_zero_residual": 0.0,
            "active_restriction_model_scale": 0.0,
            "restricted_coordinate_trust_solve": None,
        }

    H_restricted = 0.5 * (
        active_image_basis.T @ H_w @ active_image_basis
        + active_image_basis.T @ H_w.T @ active_image_basis
    )
    g_restricted = np.asarray(active_image_basis.T @ g_w, dtype=float)
    restricted_result = solve_joint_linear_model(
        gram=np.eye(restricted_dimension, dtype=float),
        hessian=H_restricted,
        gradient=g_restricted,
        active_coordinate_count=restricted_dimension,
        config=JointLinearSolveConfig(
            policy=JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
            rank_relative_tolerance=float(rank_relative_tolerance),
            metric_regularization=0.0,
            energy_regularization=float(energy_regularization),
            max_fubini_study_step=float(max_fubini_study_step),
        ),
    )
    restricted_step = np.asarray(restricted_result.joint_step, dtype=float)
    whitened_step = np.asarray(
        active_image_basis @ restricted_step, dtype=float
    )
    active_coefficients = np.asarray(
        active_coefficient_transport @ whitened_step, dtype=float
    )
    active_step = np.zeros(dimension, dtype=float)
    active_step[:active] = active_coefficients
    supported_parameter_representative = np.asarray(
        whitening @ whitened_step, dtype=float
    )
    active_image_step_residual = float(
        np.linalg.norm(
            active_image_transport @ active_coefficients - whitened_step
        )
    )
    active_step_metric_null_residual = float(
        np.linalg.norm(
            G @ (supported_parameter_representative - active_step)
        )
    )
    active_gain = float(restricted_result.predicted_reduction)
    direct_gain = float(g.T @ active_step - 0.5 * active_step.T @ H @ active_step)
    objective_residual = float(abs(direct_gain - active_gain))
    batch_zero_residual = float(np.linalg.norm(active_step[active:]))
    step_scale = float(max(1.0, np.linalg.norm(active_step)))
    batch_zero_tolerance = float(
        4096.0
        * np.finfo(float).eps
        * max(1, dimension)
        * step_scale
    )
    active_image_step_tolerance = float(
        max(
            active_image_projection_tolerance
            * max(
                1.0,
                np.linalg.norm(active_coefficients),
                np.linalg.norm(whitened_step),
            ),
            batch_zero_tolerance,
        )
    )
    active_step_metric_null_tolerance = float(
        max(
            active_image_metric_null_tolerance
            * max(1.0, np.linalg.norm(active_coefficients)),
            batch_zero_tolerance * max(1.0, np.linalg.norm(G, ord=2)),
        )
    )
    objective_tolerance = float(
        max(
            float(energy_regularization),
            restricted_result.telemetry.get(
                "trust_kkt_objective_identity_tolerance", 0.0
            ),
            4096.0
            * np.finfo(float).eps
            * max(1, dimension)
            * max(1.0, abs(active_gain), abs(direct_gain)),
        )
    )
    restricted_solve_payload = restricted_result.as_dict()
    retained_candidate_transport: dict[str, Any] = {
        "schema": "sr_v2_active_restriction_atomic_candidate_transport_v2",
        "indicated": bool(
            restricted_solve_payload.get("hard_case_detected", False)
        ),
        "valid": True,
        "reason": "no_hard_or_reflection_pair_indicated",
        "required_candidate_count": 0,
        "source_candidate_count": 0,
        "transported_candidate_count": 0,
        "source_coordinate_system": (
            "restricted_supported_trust_coordinates_v2"
        ),
        "target_coordinate_system": (
            "full_joint_active_only_parameter_coordinates_v2"
        ),
        "transport_map": (
            "physical_active_image_coefficient_transport_v2"
        ),
        "full_supported_metric_whitening_provenance_id": source_provenance,
        "active_restriction_provenance_id": restriction_provenance,
        "nested_supported_metric_whitening_provenance_id": str(
            restricted_solve_payload.get(
                "supported_metric_whitening_provenance_id", ""
            )
        ),
        "candidate_block_exact_zero_after_certified_transport": True,
        "maximum_candidate_block_residual_before_projection": 0.0,
        "maximum_active_image_step_residual": 0.0,
        "maximum_metric_null_equivalence_residual": 0.0,
        "prediction_authority": (
            "full_joint_active_only_model_recomputed_after_transport_v2"
        ),
        "maximum_prediction_crosscheck_residual": 0.0,
        "transport_provenance_id": None,
    }
    transported_pair_payload: dict[str, Any] = {}
    candidate_transport_valid = True
    if bool(retained_candidate_transport["indicated"]):
        source_candidates_raw = restricted_solve_payload.get(
            "hard_case_sign_candidates_joint", []
        )
        source_predictions_raw = restricted_solve_payload.get(
            "hard_case_sign_candidate_predicted_reductions", []
        )
        source_roles_raw = restricted_solve_payload.get(
            "hard_case_sign_candidate_point_estimate_roles", []
        )
        source_candidates = (
            list(source_candidates_raw)
            if isinstance(source_candidates_raw, Sequence)
            and not isinstance(source_candidates_raw, (str, bytes, bytearray))
            else []
        )
        source_predictions = (
            list(source_predictions_raw)
            if isinstance(source_predictions_raw, Sequence)
            and not isinstance(source_predictions_raw, (str, bytes, bytearray))
            else []
        )
        source_roles = (
            [str(value) for value in source_roles_raw]
            if isinstance(source_roles_raw, Sequence)
            and not isinstance(source_roles_raw, (str, bytes, bytearray))
            else []
        )
        retained_candidate_transport.update(
            {
                "required_candidate_count": 2,
                "source_candidate_count": int(len(source_candidates)),
            }
        )
        transported_candidates: list[list[float]] = []
        transported_predictions: list[float] = []
        maximum_batch_residual = 0.0
        maximum_image_residual = 0.0
        maximum_metric_null_residual = 0.0
        maximum_prediction_residual = 0.0
        transport_reason = "active_restriction_pair_transported"
        if (
            len(source_candidates) != 2
            or len(source_predictions) != 2
            or len(source_roles) != 2
        ):
            candidate_transport_valid = False
            transport_reason = "active_restriction_pair_metadata_count_mismatch"
        else:
            for source_candidate, source_prediction in zip(
                source_candidates,
                source_predictions,
            ):
                try:
                    restricted_candidate = np.asarray(
                        source_candidate, dtype=float
                    ).reshape(-1)
                    prediction = float(source_prediction)
                except (TypeError, ValueError):
                    candidate_transport_valid = False
                    transport_reason = "active_restriction_pair_nonfinite_source"
                    break
                if (
                    restricted_candidate.size != restricted_dimension
                    or not np.all(np.isfinite(restricted_candidate))
                    or not math.isfinite(prediction)
                ):
                    candidate_transport_valid = False
                    transport_reason = "active_restriction_pair_nonfinite_source"
                    break
                supported_candidate = np.asarray(
                    active_image_basis @ restricted_candidate,
                    dtype=float,
                )
                candidate_active_coefficients = np.asarray(
                    active_coefficient_transport @ supported_candidate,
                    dtype=float,
                )
                lifted_candidate = np.zeros(dimension, dtype=float)
                lifted_candidate[:active] = candidate_active_coefficients
                supported_parameter_candidate = np.asarray(
                    whitening @ supported_candidate,
                    dtype=float,
                )
                if (
                    lifted_candidate.size != dimension
                    or not np.all(np.isfinite(lifted_candidate))
                    or not np.all(
                        np.isfinite(candidate_active_coefficients)
                    )
                    or not np.all(
                        np.isfinite(supported_parameter_candidate)
                    )
                ):
                    candidate_transport_valid = False
                    transport_reason = "active_restriction_pair_transport_nonfinite"
                    break
                candidate_image_residual = float(
                    np.linalg.norm(
                        active_image_transport @ candidate_active_coefficients
                        - supported_candidate
                    )
                )
                candidate_metric_null_residual = float(
                    np.linalg.norm(
                        G
                        @ (
                            supported_parameter_candidate
                            - lifted_candidate
                        )
                    )
                )
                if (
                    not math.isfinite(candidate_image_residual)
                    or not math.isfinite(candidate_metric_null_residual)
                ):
                    candidate_transport_valid = False
                    transport_reason = (
                        "active_restriction_pair_transport_nonfinite"
                    )
                    break
                maximum_image_residual = float(
                    max(maximum_image_residual, candidate_image_residual)
                )
                maximum_metric_null_residual = float(
                    max(
                        maximum_metric_null_residual,
                        candidate_metric_null_residual,
                    )
                )
                candidate_batch_residual = float(
                    np.linalg.norm(lifted_candidate[active:])
                )
                maximum_batch_residual = float(
                    max(maximum_batch_residual, candidate_batch_residual)
                )
                candidate_zero_tolerance = float(
                    max(
                        batch_zero_tolerance,
                        4096.0
                        * np.finfo(float).eps
                        * max(1, dimension)
                        * max(1.0, np.linalg.norm(lifted_candidate)),
                    )
                )
                candidate_image_tolerance = float(
                    max(
                        active_image_step_tolerance,
                        active_image_projection_tolerance
                        * max(
                            1.0,
                            np.linalg.norm(candidate_active_coefficients),
                            np.linalg.norm(supported_candidate),
                        ),
                    )
                )
                candidate_metric_null_tolerance = float(
                    max(
                        active_step_metric_null_tolerance,
                        active_image_metric_null_tolerance
                        * max(
                            1.0,
                            np.linalg.norm(candidate_active_coefficients),
                        ),
                    )
                )
                if (
                    candidate_batch_residual > candidate_zero_tolerance
                    or candidate_image_residual > candidate_image_tolerance
                    or candidate_metric_null_residual
                    > candidate_metric_null_tolerance
                ):
                    candidate_transport_valid = False
                    transport_reason = (
                        "active_restriction_pair_physical_transport_unresolved"
                    )
                    break
                transported_prediction = float(
                    g.T @ lifted_candidate
                    - 0.5 * lifted_candidate.T @ H @ lifted_candidate
                )
                prediction_residual = float(
                    abs(transported_prediction - prediction)
                )
                maximum_prediction_residual = float(
                    max(maximum_prediction_residual, prediction_residual)
                )
                candidate_prediction_tolerance = float(
                    max(
                        objective_tolerance,
                        4096.0
                        * np.finfo(float).eps
                        * max(1, dimension)
                        * max(
                            1.0,
                            abs(transported_prediction),
                            abs(prediction),
                        ),
                    )
                )
                if prediction_residual > candidate_prediction_tolerance:
                    candidate_transport_valid = False
                    transport_reason = (
                        "active_restriction_pair_prediction_mismatch"
                    )
                    break
                transported_candidates.append(
                    [float(value) for value in lifted_candidate.tolist()]
                )
                transported_predictions.append(transported_prediction)
        retained_candidate_transport.update(
            {
                "valid": bool(candidate_transport_valid),
                "reason": str(transport_reason),
                "transported_candidate_count": int(
                    len(transported_candidates)
                ),
                "maximum_candidate_block_residual_before_projection": float(
                    maximum_batch_residual
                ),
                "maximum_active_image_step_residual": float(
                    maximum_image_residual
                ),
                "maximum_metric_null_equivalence_residual": float(
                    maximum_metric_null_residual
                ),
                "maximum_prediction_crosscheck_residual": float(
                    maximum_prediction_residual
                ),
            }
        )
        if candidate_transport_valid:
            transport_digest = hashlib.sha256()
            transport_digest.update(
                b"sr_v2_active_restriction_atomic_candidate_transport_v2\0"
            )
            for value in (
                source_provenance,
                restriction_provenance,
                str(
                    retained_candidate_transport[
                        "nested_supported_metric_whitening_provenance_id"
                    ]
                ),
                str(
                    restricted_solve_payload.get(
                        "hard_case_classification", ""
                    )
                ),
            ):
                transport_digest.update(value.encode("utf-8"))
                transport_digest.update(b"\0")
            transport_digest.update(
                np.ascontiguousarray(
                    np.asarray(transported_candidates, dtype="<f8")
                ).tobytes()
            )
            transport_digest.update(b"\0")
            transport_digest.update(
                np.ascontiguousarray(
                    np.asarray(transported_predictions, dtype="<f8")
                ).tobytes()
            )
            transport_digest.update(b"\0")
            for role in source_roles:
                transport_digest.update(role.encode("utf-8"))
                transport_digest.update(b"\0")
            transport_provenance = transport_digest.hexdigest()
            retained_candidate_transport["transport_provenance_id"] = (
                transport_provenance
            )
            transported_pair_payload = {
                "hard_case_detected": True,
                "hard_case_classification": str(
                    restricted_solve_payload.get(
                        "hard_case_classification", ""
                    )
                ),
                "hard_case_uncertain_projection_reflection_retained": bool(
                    restricted_solve_payload.get(
                        "hard_case_uncertain_projection_reflection_retained",
                        False,
                    )
                ),
                "hard_case_orientation_policy": str(
                    restricted_solve_payload.get(
                        "hard_case_orientation_policy", ""
                    )
                ),
                "hard_case_orientation_exact_signs_retained": bool(
                    restricted_solve_payload.get(
                        "hard_case_orientation_exact_signs_retained", True
                    )
                ),
                "hard_case_sign_candidates_joint": transported_candidates,
                "hard_case_sign_candidate_predicted_reductions": (
                    transported_predictions
                ),
                "hard_case_sign_candidate_point_estimate_roles": source_roles,
                "hard_case_selected_sign": restricted_solve_payload.get(
                    "hard_case_selected_sign"
                ),
                "hard_case_sign_pair_atomic_required": True,
                "hard_case_sign_pair_transport_provenance_id": (
                    transport_provenance
                ),
            }
    active_valid = bool(
        restricted_result.feasible
        and restricted_result.telemetry.get(
            "trust_global_optimality_certified", False
        )
        and active_image_step_residual <= active_image_step_tolerance
        and active_step_metric_null_residual
        <= active_step_metric_null_tolerance
        and batch_zero_residual <= batch_zero_tolerance
        and objective_residual <= objective_tolerance
        and candidate_transport_valid
    )
    return {
        **shared_payload,
        "valid": active_valid,
        "reason": (
            "active_restriction_atomic_candidate_transport_failed"
            if not candidate_transport_valid
            else "shared_support_active_restriction_solved"
            if active_valid
            else "shared_support_active_restriction_certificate_failed"
        ),
        "feasible": bool(restricted_result.feasible),
        "predicted_reduction": active_gain,
        "applied_predicted_reduction": active_gain,
        "joint_step": [float(value) for value in active_step.tolist()],
        "active_parameter_relaxation": [
            float(value) for value in active_step[:active].tolist()
        ],
        "batch_coordinate_step": [
            float(value) for value in active_step[active:].tolist()
        ],
        "fubini_study_displacement_sq": float(
            max(0.0, active_step.T @ G @ active_step)
        ),
        "joint_fubini_study_displacement_sq": float(
            max(0.0, active_step.T @ G @ active_step)
        ),
        "trust_global_optimality_certified": bool(active_valid),
        "active_restriction_direct_predicted_reduction": direct_gain,
        "active_restriction_direct_objective_residual": objective_residual,
        "active_restriction_direct_objective_tolerance": objective_tolerance,
        "active_restriction_batch_zero_residual": batch_zero_residual,
        "active_restriction_batch_zero_tolerance": batch_zero_tolerance,
        "active_restriction_image_step_residual": active_image_step_residual,
        "active_restriction_image_step_tolerance": active_image_step_tolerance,
        "active_restriction_metric_null_residual": (
            active_step_metric_null_residual
        ),
        "active_restriction_metric_null_tolerance": (
            active_step_metric_null_tolerance
        ),
        "active_restriction_model_scale": float(
            np.linalg.norm(g_restricted) * float(max_fubini_study_step)
            + 0.5
            * np.linalg.norm(H_restricted, ord=2)
            * float(max_fubini_study_step) ** 2
        ),
        "restricted_coordinate_trust_solve": restricted_solve_payload,
        "active_restriction_atomic_candidate_transport": (
            retained_candidate_transport
        ),
        **transported_pair_payload,
    }


@dataclass
class _BatchFullGeometryWorkspace:
    records: tuple[dict[str, Any], ...]
    record_index: dict[tuple[int, int, str], int]
    ansatz_depth: int
    active_indices: tuple[int, ...]
    active_labels: tuple[str, ...]
    G_AA: np.ndarray
    H_AA: np.ndarray
    G_AB: np.ndarray
    H_AB: np.ndarray
    G_BB: np.ndarray
    H_BB: np.ndarray
    g_A: np.ndarray
    g_B: np.ndarray
    phase2_reported_g_B: np.ndarray
    geometry_mode: str
    joint_context_mode: str
    workspace_fingerprint: str
    metric_regularization: float
    energy_regularization: float
    joint_linear_solve_policy: str
    rank_relative_tolerance: float
    max_gram_condition_number: float
    max_fubini_study_step: float
    state_delta_norm: float
    state_consistency_tolerance: float
    phase2_reuse_validation: dict[str, Any]
    _subset_cache: dict[tuple[int, ...], dict[str, Any]]
    state_fingerprint: str = ""
    theta_fingerprint: str = ""
    ordered_scaffold_fingerprint: str = ""
    hamiltonian_fingerprint: str = ""
    _state_stationarity_cache: dict[str, Any] | None = None
    phase3_candidate_gain_policy: str = (
        PHASE3_CANDIDATE_GAIN_JOINT_TOTAL_V1
    )
    _active_only_gain_baseline_cache: dict[str, Any] | None = None

    def _active_only_gain_baseline(self) -> dict[str, Any]:
        """Solve the candidate-independent active model once per workspace."""

        if self._active_only_gain_baseline_cache is not None:
            return dict(self._active_only_gain_baseline_cache)
        active_count = int(len(self.active_indices))
        base: dict[str, Any] = {
            "schema": "phase3_active_only_supported_trust_baseline_v1",
            "source": "independent_active_block_same_policy_and_radius_v1",
            "candidate_independent": True,
            "active_coordinate_count": int(active_count),
            "joint_linear_solve_policy": str(
                self.joint_linear_solve_policy
            ),
            "rank_relative_tolerance": float(
                self.rank_relative_tolerance
            ),
            "metric_regularization": float(self.metric_regularization),
            "energy_regularization": float(self.energy_regularization),
            "max_fubini_study_step": float(
                self.max_fubini_study_step
            ),
            "classical_quantum_query_charge": 0,
        }
        if active_count == 0:
            payload = {
                **base,
                "feasible": True,
                "reason": "zero_dimensional_active_model",
                "predicted_reduction": 0.0,
                "gain_numerical_error_bound": 0.0,
                "active_parameter_relaxation": [],
                "joint_step": [],
            }
            self._active_only_gain_baseline_cache = dict(payload)
            return payload

        solve_result = solve_joint_linear_model(
            gram=np.asarray(self.G_AA, dtype=float),
            hessian=np.asarray(self.H_AA, dtype=float),
            gradient=np.asarray(self.g_A, dtype=float),
            active_coordinate_count=active_count,
            config=JointLinearSolveConfig(
                policy=str(self.joint_linear_solve_policy),
                rank_relative_tolerance=float(
                    self.rank_relative_tolerance
                ),
                metric_regularization=float(self.metric_regularization),
                energy_regularization=float(self.energy_regularization),
                max_fubini_study_step=float(
                    self.max_fubini_study_step
                ),
            ),
        )
        retained_condition_raw = solve_result.telemetry.get(
            "retained_metric_condition_number"
        )
        retained_condition = (
            None
            if retained_condition_raw is None
            else float(retained_condition_raw)
        )
        feasible = bool(
            solve_result.feasible
            and (
                retained_condition is None
                or retained_condition
                <= float(self.max_gram_condition_number)
            )
        )
        reason = (
            str(solve_result.reason)
            if not bool(solve_result.feasible)
            else "conditioning_gate"
            if not feasible
            else "independent_active_only_supported_trust_solve_v1"
        )
        gain = float(
            max(0.0, solve_result.predicted_reduction)
            if feasible
            else 0.0
        )
        error_bound = float(
            max(
                float(self.energy_regularization),
                4096.0
                * np.finfo(float).eps
                * max(1, active_count)
                * max(1.0, abs(gain)),
            )
        )
        payload = {
            **base,
            **solve_result.as_dict(),
            "feasible": bool(feasible),
            "reason": str(reason),
            "predicted_reduction": float(gain),
            "gain_numerical_error_bound": float(error_bound),
            "retained_metric_condition_number": retained_condition,
            "max_gram_condition_number": float(
                self.max_gram_condition_number
            ),
        }
        self._active_only_gain_baseline_cache = dict(payload)
        return payload

    def _infeasible_incremental_candidate_gain_payload(
        self,
        *,
        reason: str,
        solve_result: Any | None = None,
    ) -> dict[str, Any]:
        """Bind corrected gain semantics even when admission is infeasible.

        Rank, conditioning, or solver gates may reject a record before the
        ordinary feasible-summary path constructs its gain receipt.  The
        record remains in the authenticated Phase-III population, so the
        corrected route must still identify its policy and explicitly record
        that its selection-facing incremental gain is zero.  Legacy summaries
        retain their historical sparse shape.
        """

        policy = str(self.phase3_candidate_gain_policy)
        if policy not in PHASE3_CANDIDATE_GAIN_POLICIES:
            raise ValueError(
                "phase3_candidate_gain_policy must be one of "
                f"{sorted(PHASE3_CANDIDATE_GAIN_POLICIES)}."
            )
        if policy != PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1:
            return {}

        full_model_feasible = bool(
            solve_result is not None
            and bool(getattr(solve_result, "feasible", False))
        )
        full_gain = float(
            max(0.0, float(solve_result.predicted_reduction))
            if full_model_feasible
            else 0.0
        )
        solve_dimension = int(
            np.asarray(
                getattr(solve_result, "joint_step", ()),
                dtype=float,
            ).size
            if solve_result is not None
            else len(self.active_indices)
        )
        full_error_bound = float(
            max(
                float(self.energy_regularization),
                4096.0
                * np.finfo(float).eps
                * max(
                    1,
                    solve_dimension,
                )
                * max(1.0, abs(full_gain)),
            )
        )
        active_only_baseline = self._active_only_gain_baseline()
        active_model_feasible = bool(
            active_only_baseline.get("feasible", False)
        )
        active_gain = float(
            active_only_baseline.get("predicted_reduction", 0.0)
            if active_model_feasible
            else 0.0
        )
        active_error_bound = float(
            active_only_baseline.get("gain_numerical_error_bound", 0.0)
            if active_model_feasible
            else 0.0
        )
        raw_increment = float(full_gain - active_gain)
        comparison_tolerance = float(
            full_error_bound + active_error_bound
        )
        receipt = {
            "schema": "phase3_candidate_gain_receipt_v1",
            "policy": str(policy),
            "joint_gain_semantics": "incremental_candidate_gain_v1",
            "selection_authority": (
                "candidate_infeasible_before_gain_selection_v1"
            ),
            "full_joint_trust_gain": float(full_gain),
            "full_joint_gain_numerical_error_bound": float(
                full_error_bound
            ),
            "active_only_trust_gain": float(active_gain),
            "incremental_candidate_gain_raw": float(raw_increment),
            "incremental_candidate_gain": 0.0,
            "selected_gain": 0.0,
            "comparison_tolerance": float(comparison_tolerance),
            "comparison_status": (
                "selection_infeasible_before_gain_admission_v1"
            ),
            "comparison_feasible": False,
            "selection_infeasible_reason": str(reason),
            "full_joint_model_feasible": bool(full_model_feasible),
            "active_only_model_feasible": bool(active_model_feasible),
            "active_only_baseline": dict(active_only_baseline),
            "baseline_candidate_independent": bool(
                active_only_baseline.get("candidate_independent", False)
            ),
            "classical_quantum_query_charge": 0,
        }
        return {
            "phase3_candidate_gain_policy": str(policy),
            "joint_gain_semantics": "incremental_candidate_gain_v1",
            "phase3_candidate_gain_receipt": receipt,
            "full_joint_gain": float(full_gain),
            "active_only_gain": float(active_gain),
            "incremental_candidate_gain_raw": float(raw_increment),
            "incremental_candidate_gain": 0.0,
            "joint_gain": 0.0,
        }

    def _state_stationarity_summary(self) -> dict[str, Any]:
        """Audit supported stationarity of X independently of every record.

        A quotient-redundant singleton supplies no state-stationarity evidence.
        This shared active-coordinate audit is therefore constructed once from
        the unchanged working state and attached to every record summary only
        as a state-bound token, never as a per-record inertia substitute.
        """

        if self._state_stationarity_cache is not None:
            return dict(self._state_stationarity_cache)
        active_count = int(len(self.active_indices))
        state_fingerprint = str(
            self.state_fingerprint or self.workspace_fingerprint
        )
        base: dict[str, Any] = {
            "schema": "sr_escape_state_stationarity_certificate_v1",
            "state_fingerprint": state_fingerprint,
            "workspace_fingerprint": str(self.workspace_fingerprint),
            "active_coordinate_count": int(active_count),
            "active_coordinate_identities": [
                str(label) for label in self.active_labels
            ],
            "trust_radius": float(self.max_fubini_study_step),
            "joint_linear_solve_policy": str(
                self.joint_linear_solve_policy
            ),
            "comparison_scope": (
                "working_state_active_coordinates_independent_of_singleton_v1"
            ),
        }
        if str(self.joint_linear_solve_policy) != (
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2
        ):
            result = {
                **base,
                "valid": False,
                "reason": "state_stationarity_requires_global_trust_v2",
                "supported_stationarity_status": "unresolved",
            }
            self._state_stationarity_cache = dict(result)
            return result
        if active_count == 0:
            provenance = hashlib.sha256(
                (
                    "sr_escape_zero_dimensional_state_stationarity_v1::"
                    + state_fingerprint
                    + "::"
                    + str(self.workspace_fingerprint)
                    + "::"
                    + repr(float(self.max_fubini_study_step))
                ).encode("utf-8")
            ).hexdigest()
            result = {
                **base,
                "valid": True,
                "reason": "zero_dimensional_active_state_is_stationary",
                "supported_stationarity_status": "stationary",
                "supported_gradient_norm_upper_bound": 0.0,
                "supported_gradient_resolution": 0.0,
                "stationarity_margin": 0.0,
                "raw_metric_support_status": "resolved",
                "raw_metric_null_compatibility_certified": True,
                "support_provenance_digest": provenance,
                "trust_provenance_digest": provenance,
            }
            self._state_stationarity_cache = dict(result)
            return result
        solve_result = solve_joint_linear_model(
            gram=np.asarray(self.G_AA, dtype=float),
            hessian=np.asarray(self.H_AA, dtype=float),
            gradient=np.asarray(self.g_A, dtype=float),
            active_coordinate_count=active_count,
            config=JointLinearSolveConfig(
                policy=(
                    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2
                ),
                rank_relative_tolerance=float(self.rank_relative_tolerance),
                metric_regularization=float(self.metric_regularization),
                energy_regularization=float(self.energy_regularization),
                max_fubini_study_step=float(self.max_fubini_study_step),
            ),
        )
        telemetry = dict(solve_result.telemetry)
        status = str(
            telemetry.get("supported_stationarity_status", "unresolved")
        )
        upper_raw = telemetry.get("supported_gradient_norm_upper_bound")
        resolution_raw = telemetry.get("supported_gradient_resolution")
        try:
            upper = float(upper_raw)
            resolution = float(resolution_raw)
        except (TypeError, ValueError):
            upper = float("nan")
            resolution = float("nan")
        margin = float(upper - resolution)
        raw_support_resolved = bool(
            str(telemetry.get("raw_metric_support_status", "unresolved"))
            == "resolved"
        )
        null_compatible = bool(
            telemetry.get("raw_metric_null_compatibility_certified", False)
        )
        valid = bool(
            raw_support_resolved
            and null_compatible
            and status == "stationary"
            and math.isfinite(upper)
            and math.isfinite(resolution)
            and resolution >= 0.0
            and margin <= 0.0
        )
        support_provenance = str(
            telemetry.get("supported_metric_whitening_provenance_id", "")
        )
        trust_payload = {
            "state_fingerprint": state_fingerprint,
            "workspace_fingerprint": str(self.workspace_fingerprint),
            "support_provenance": support_provenance,
            "trust_radius": float(self.max_fubini_study_step),
            "solver_policy": str(self.joint_linear_solve_policy),
            "metric_whitening_ridge": telemetry.get(
                "metric_whitening_ridge"
            ),
            "metric_retained_mask": telemetry.get("metric_retained_mask"),
            "whitening_denominators": telemetry.get(
                "whitening_denominators"
            ),
        }
        trust_provenance = hashlib.sha256(
            json.dumps(
                trust_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        result = {
            **base,
            "valid": bool(valid),
            "reason": (
                "working_state_supported_stationarity_certified"
                if valid
                else "working_state_supported_stationarity_unresolved"
            ),
            "supported_stationarity_status": status,
            "supported_gradient_norm_upper_bound": (
                float(upper) if math.isfinite(upper) else None
            ),
            "supported_gradient_resolution": (
                float(resolution) if math.isfinite(resolution) else None
            ),
            "stationarity_margin": (
                float(margin) if math.isfinite(margin) else None
            ),
            "raw_metric_support_status": str(
                telemetry.get("raw_metric_support_status", "unresolved")
            ),
            "raw_metric_null_compatibility_certified": bool(
                null_compatible
            ),
            "support_provenance_digest": support_provenance,
            "trust_provenance_digest": trust_provenance,
        }
        self._state_stationarity_cache = dict(result)
        return result

    def old_old_geometry_payload(self) -> dict[str, Any]:
        """Expose the authoritative active block for FM anchor/reuse plumbing."""

        reuse = dict(self.phase2_reuse_validation)
        return {
            "schema": "historical_singleton_workspace_old_old_geometry_v1",
            "workspace_fingerprint": str(self.workspace_fingerprint),
            "state_fingerprint": str(
                self.state_fingerprint or self.workspace_fingerprint
            ),
            "theta_fingerprint": str(self.theta_fingerprint),
            "ordered_scaffold_fingerprint": str(
                self.ordered_scaffold_fingerprint
            ),
            "hamiltonian_fingerprint": str(
                self.hamiltonian_fingerprint
            ),
            "active_indices": [int(value) for value in self.active_indices],
            "active_labels": [str(value) for value in self.active_labels],
            "G_AA": np.asarray(self.G_AA, dtype=float).tolist(),
            "H_AA": np.asarray(self.H_AA, dtype=float).tolist(),
            "g_A": np.asarray(self.g_A, dtype=float).tolist(),
            "gradient_convention": "sr_descent_gradient_v1",
            "old_old_geometry_source": str(
                reuse.get(
                    "old_old_geometry_source",
                    "authoritative_endpoint_geometry_v1",
                )
            ),
            "old_old_geometry_prior": (
                None
                if reuse.get("old_old_geometry_prior") is None
                else dict(reuse.get("old_old_geometry_prior", {}))
            ),
        }

    def build_telemetry(self) -> dict[str, Any]:
        candidate_count = int(len(self.records))
        active_count = int(len(self.active_indices))
        candidate_pair_count = int(candidate_count * (candidate_count - 1) // 2)
        reuse = dict(self.phase2_reuse_validation)
        valid_reuse_count = int(reuse.get("valid_record_count", 0))
        search_pool_records = []
        for workspace_index, record in enumerate(self.records):
            feature = record.get("feature")
            parent_label = str(
                record.get("runtime_split_parent_label")
                or getattr(feature, "runtime_split_parent_label", "")
                or ""
            )
            parent_labels = record.get("route_a_child_parent_labels")
            if isinstance(parent_labels, Sequence) and not isinstance(
                parent_labels, (str, bytes)
            ):
                resolved_parent_labels = [str(value) for value in parent_labels]
            else:
                resolved_parent_labels = [parent_label] if parent_label else []
            search_pool_records.append(
                {
                    "workspace_index": int(workspace_index),
                    "candidate_pool_index": int(
                        record.get("candidate_pool_index", -1)
                    ),
                    "candidate_label": str(_batch_record_label(record)),
                    "position_id": int(record.get("position_id", -1)),
                    "global_child_identity": str(
                        record.get("route_a_global_pauli_identity") or ""
                    ),
                    "parent_labels": resolved_parent_labels,
                    "joint_model_descent_gradient": float(
                        self.g_B[int(workspace_index)]
                    ),
                    "phase2_reported_descent_gradient": float(
                        self.phase2_reported_g_B[int(workspace_index)]
                    ),
                    "gradient_parity_delta": float(
                        self.g_B[int(workspace_index)]
                        - self.phase2_reported_g_B[int(workspace_index)]
                    ),
                }
            )
        return {
            "schema": "batch_full_geometry_workspace_v1",
            "geometry_mode": str(self.geometry_mode),
            "full_geometry_workspace_build_count": 1,
            "search_population_count": int(candidate_count),
            "active_ansatz_depth": int(self.ansatz_depth),
            "active_coordinate_count": int(active_count),
            "active_indices": [int(index) for index in self.active_indices],
            "active_labels": [str(label) for label in self.active_labels],
            "search_pool_records": search_pool_records,
            "joint_batch_context_mode_requested": str(self.joint_context_mode),
            "joint_batch_context_mode_effective": str(self.joint_context_mode),
            "workspace_fingerprint": str(self.workspace_fingerprint),
            "state_fingerprint": str(
                self.state_fingerprint or self.workspace_fingerprint
            ),
            "theta_fingerprint": str(self.theta_fingerprint),
            "ordered_scaffold_fingerprint": str(
                self.ordered_scaffold_fingerprint
            ),
            "hamiltonian_fingerprint": str(
                self.hamiltonian_fingerprint
            ),
            "sr_escape_state_stationarity_summary": (
                self._state_stationarity_summary()
            ),
            "G_search_shape": [int(x) for x in self.G_BB.shape],
            "H_search_shape": [int(x) for x in self.H_BB.shape],
            "G_active_candidate_shape": [int(x) for x in self.G_AB.shape],
            "H_active_candidate_shape": [int(x) for x in self.H_AB.shape],
            "G_AA_dimensions": [int(x) for x in self.G_AA.shape],
            "G_AB_dimensions": [int(x) for x in self.G_AB.shape],
            "G_BB_dimensions": [int(x) for x in self.G_BB.shape],
            "H_AA_dimensions": [int(x) for x in self.H_AA.shape],
            "H_AB_dimensions": [int(x) for x in self.H_AB.shape],
            "H_BB_dimensions": [int(x) for x in self.H_BB.shape],
            "candidate_pair_count": int(candidate_pair_count),
            "required_candidate_pair_count": int(
                reuse.get("required_candidate_pair_count", candidate_pair_count)
            ),
            "constructed_candidate_pair_count": int(
                reuse.get("constructed_candidate_pair_count", candidate_pair_count)
            ),
            "reused_cached_candidate_pair_count": int(
                reuse.get("reused_cached_candidate_pair_count", 0)
            ),
            "candidate_pair_pruned_before_measurement_count": int(
                reuse.get("candidate_pair_pruned_before_measurement_count", 0)
            ),
            "matrix_cache_scope": "selector_round_shared_workspace_v1",
            "joint_pair_cache_scope": str(
                reuse.get(
                    "joint_pair_cache_scope",
                    "state_scaffold_hamiltonian_fingerprinted_lru_v1",
                )
            ),
            "joint_pair_cache_max_entries": int(
                reuse.get("joint_pair_cache_max_entries", 0)
            ),
            "joint_pair_cache_hit_count": int(
                reuse.get("joint_pair_cache_hit_count", 0)
            ),
            "joint_pair_cache_miss_count": int(
                reuse.get("joint_pair_cache_miss_count", 0)
            ),
            "joint_pair_workers_requested": int(
                reuse.get("joint_pair_workers_requested", 1)
            ),
            "joint_pair_workers_effective": int(
                reuse.get("joint_pair_workers_effective", 1)
            ),
            "joint_pair_parallel_enabled": bool(
                reuse.get("joint_pair_parallel_enabled", False)
            ),
            "joint_pair_result_order": str(
                reuse.get(
                    "joint_pair_result_order",
                    "required_candidate_pairs_order_v1",
                )
            ),
            "joint_pair_receipts": [
                dict(receipt)
                for receipt in reuse.get("joint_pair_receipts", ())
                if isinstance(receipt, Mapping)
            ],
            "workspace_build_mode": str(
                reuse.get("workspace_build_mode", "eager_compatibility_v1")
            ),
            "subset_cache_entry_count": int(len(self._subset_cache)),
            "reused_child_phase2_diagonal_count": int(2 * valid_reuse_count),
            "reused_child_phase2_active_mixed_count": int(
                2 * active_count * valid_reuse_count
            ),
            "query_chargeable_unique_geometry_element_count": int(
                reuse.get("query_chargeable_unique_geometry_element_count", 0)
            ),
            "total_mathematically_required_element_count": int(
                reuse.get("total_mathematically_required_element_count", 0)
            ),
            "reused_phase2_element_count": int(
                reuse.get("reused_total_element_count", 0)
            ),
            "newly_measured_element_count": int(
                reuse.get("query_chargeable_unique_geometry_element_count", 0)
            ),
            "required_element_counts": dict(
                reuse.get("required_element_counts", {})
            ),
            "reused_element_counts": dict(
                reuse.get("reused_element_counts", {})
            ),
            "newly_measured_element_counts": dict(
                reuse.get("newly_measured_element_counts", {})
            ),
            "matrix_cache_hit_element_count": int(
                reuse.get("cache_hit_element_count", 0)
            ),
            "matrix_cache_miss_element_count": int(
                reuse.get("cache_miss_element_count", 0)
            ),
            "matrix_cache_invalidation_reason_counts": dict(
                reuse.get("invalidation_reason_counts", {})
            ),
            "query_chargeable_gradient_repair_count": int(
                reuse.get("query_chargeable_gradient_repair_count", 0)
            ),
            "validated_phase2_gradient_reuse_count": int(
                reuse.get("valid_gradient_record_count", 0)
            ),
            "query_charge_policy": (
                "validated_phase2_block_and_gradient_reuse_then_charge_each_"
                "remaining_unique_joint_matrix_element_or_gradient_once"
            ),
            "metric_regularization": float(self.metric_regularization),
            "energy_regularization": float(self.energy_regularization),
            "joint_linear_solve_policy_requested": str(
                self.joint_linear_solve_policy
            ),
            "phase3_candidate_gain_policy": str(
                self.phase3_candidate_gain_policy
            ),
            "active_only_gain_baseline": (
                None
                if self._active_only_gain_baseline_cache is None
                else dict(self._active_only_gain_baseline_cache)
            ),
            "state_reconstruction_delta_norm": float(self.state_delta_norm),
            "state_consistency_tolerance": float(
                self.state_consistency_tolerance
            ),
            "state_consistency_status": "phase_aligned_match",
            "phase2_joint_geometry_reuse_validation": reuse,
            "joint_gradient_source": (
                "outer_information_old_old_plus_exact_candidate_gradient_v1"
                if reuse.get("old_old_geometry_prior") is not None
                else "shared_full_derivative_workspace_v1"
            ),
            "phase2_gradient_reused_as": "parity_telemetry_only",
            "joint_vs_phase2_gradient_linf": float(
                np.max(np.abs(self.g_B - self.phase2_reported_g_B))
                if candidate_count
                else 0.0
            ),
            **(
                {
                    "old_old_geometry_prior": dict(
                        reuse.get("old_old_geometry_prior", {})
                    ),
                    "old_old_geometry_source": str(
                        reuse.get("old_old_geometry_source")
                    ),
                    "old_old_geometry_reacquired": False,
                }
                if reuse.get("old_old_geometry_prior") is not None
                else {}
            ),
        }

    def _separate_schur_summary_for_indices(
        self,
        indices: tuple[int, ...],
    ) -> dict[str, Any]:
        key = tuple(sorted(int(index) for index in indices))
        cached = self._subset_cache.get(key)
        if cached is not None:
            return {**dict(cached), "geometry_workspace_cache_hit": True}
        if not key:
            return {"feasible": False, "reason": "empty_subset"}
        idx = np.asarray(key, dtype=int)
        G_BB_raw = np.asarray(self.G_BB[np.ix_(idx, idx)], dtype=float)
        H_BB_raw = np.asarray(self.H_BB[np.ix_(idx, idx)], dtype=float)
        G_BA = np.asarray(self.G_AB[:, idx].T, dtype=float)
        H_BA = np.asarray(self.H_AB[:, idx].T, dtype=float)
        G_AB = np.asarray(G_BA.T, dtype=float)
        H_AB = np.asarray(H_BA.T, dtype=float)
        if self.G_AA.size:
            G_AA_regularized = 0.5 * (self.G_AA + self.G_AA.T) + float(
                self.metric_regularization
            ) * np.eye(self.G_AA.shape[0], dtype=float)
            G_active_inverse = np.linalg.pinv(
                G_AA_regularized,
                rcond=max(float(self.metric_regularization), 1e-15),
            )
            G_residual = G_BB_raw - G_BA @ G_active_inverse @ G_AB
        else:
            G_AA_regularized = np.zeros((0, 0), dtype=float)
            G_residual = np.asarray(G_BB_raw, dtype=float)
        if self.H_AA.size:
            H_AA_regularized = 0.5 * (self.H_AA + self.H_AA.T) + float(
                self.energy_regularization
            ) * np.eye(self.H_AA.shape[0], dtype=float)
            H_active_inverse = np.linalg.pinv(
                H_AA_regularized,
                rcond=max(float(self.energy_regularization), 1e-15),
            )
            H_residual = H_BB_raw - H_BA @ H_active_inverse @ H_AB
        else:
            H_AA_regularized = np.zeros((0, 0), dtype=float)
            H_residual = np.asarray(H_BB_raw, dtype=float)
        G_residual = 0.5 * (G_residual + G_residual.T)
        H_residual = 0.5 * (H_residual + H_residual.T)
        gram_eigenvalues = np.linalg.eigvalsh(G_residual)
        gram_scale = float(
            max(
                np.max(np.abs(gram_eigenvalues)) if gram_eigenvalues.size else 0.0,
                self.metric_regularization,
            )
        )
        rank_floor = float(
            max(
                self.metric_regularization,
                gram_scale
                * float(
                    max(
                        0.0,
                        self.rank_relative_tolerance,
                    )
                ),
            )
        )
        effective_rank = int(np.count_nonzero(gram_eigenvalues > rank_floor))
        if effective_rank < int(len(key)):
            summary = {
                "feasible": False,
                "reason": "rank_gate",
                "subset_workspace_indices": [int(value) for value in key],
                "gram_eigenvalues": [
                    float(value) for value in gram_eigenvalues.tolist()
                ],
                "effective_rank": int(effective_rank),
                "rank_floor": float(rank_floor),
                "G_BB_raw": G_BB_raw.tolist(),
                "G_residual": G_residual.tolist(),
                "H_BB_raw": H_BB_raw.tolist(),
                "H_residual": H_residual.tolist(),
            }
            self._subset_cache[key] = dict(summary)
            return summary
        positive_eigenvalues = gram_eigenvalues[gram_eigenvalues > rank_floor]
        gram_condition = float(
            np.max(positive_eigenvalues) / np.min(positive_eigenvalues)
            if positive_eigenvalues.size
            else float("inf")
        )
        g_subset = np.asarray(self.g_B[idx], dtype=float)
        G_direction_matrix = G_residual + float(self.metric_regularization) * np.eye(
            len(key), dtype=float
        )
        direction = np.asarray(
            np.linalg.pinv(
                G_direction_matrix,
                rcond=max(float(self.metric_regularization), 1e-15),
            )
            @ g_subset,
            dtype=float,
        )
        geometric_descent = float(max(0.0, g_subset.T @ direction))
        directional_metric = float(max(0.0, direction.T @ G_residual @ direction))
        directional_curvature = float(direction.T @ H_residual @ direction)
        step_norm = float(math.sqrt(directional_metric))
        step_limit = float(
            float(self.max_fubini_study_step) / step_norm
            if step_norm > max(self.metric_regularization, 1e-15)
            else 0.0
        )
        curvature_floor = float(max(self.energy_regularization, 1e-15))
        if directional_curvature > curvature_floor:
            eta_star: float | None = float(
                geometric_descent / directional_curvature
            )
            applied_eta = float(min(eta_star, step_limit))
            eta_clipped = bool(applied_eta < eta_star)
            eta_policy = (
                "positive_curvature_trust_safeguard"
                if eta_clipped
                else "positive_curvature_unclipped_optimum"
            )
        else:
            eta_star = None
            applied_eta = float(step_limit)
            eta_clipped = True
            eta_policy = "nonpositive_curvature_fubini_study_boundary_v1"
        first_order_reduction = float(applied_eta * geometric_descent)
        hessian_correction = float(
            0.5 * applied_eta * applied_eta * directional_curvature
        )
        predicted_gain = float(max(0.0, first_order_reduction - hessian_correction))
        if len(key) == 1:
            singleton_total = float(predicted_gain)
            additivity_defect = 0.0
        else:
            singleton_total = float(
                sum(
                    float(
                        self._summary_for_indices((int(index),)).get(
                            "joint_gain", 0.0
                        )
                    )
                    for index in key
                )
            )
            additivity_defect = float(
                max(
                    0.0,
                    1.0
                    - predicted_gain
                    / (singleton_total + max(self.metric_regularization, 1e-15)),
                )
            )
        labels = [
            str(self.records[int(index)].get("candidate_label", ""))
            for index in key
        ]
        summary = {
            "feasible": True,
            "reason": "shared_full_joint_geometry_v1",
            "geometry_mode": str(self.geometry_mode),
            "subset_workspace_indices": [int(value) for value in key],
            "selected_labels": labels,
            "selected_count": int(len(key)),
            "G_AA_raw": self.G_AA.tolist(),
            "G_AA_regularized": G_AA_regularized.tolist(),
            "G_BA_raw": G_BA.tolist(),
            "G_BB_raw": G_BB_raw.tolist(),
            "G_residual": G_residual.tolist(),
            "H_AA_raw": self.H_AA.tolist(),
            "H_AA_regularized": H_AA_regularized.tolist(),
            "H_BA_raw": H_BA.tolist(),
            "H_BB_raw": H_BB_raw.tolist(),
            "H_residual": H_residual.tolist(),
            "gram_eigenvalues": [
                float(value) for value in gram_eigenvalues.tolist()
            ],
            "effective_rank": int(effective_rank),
            "rank_floor": float(rank_floor),
            "gram_condition_number": float(gram_condition),
            "g_effective": [float(value) for value in g_subset.tolist()],
            "natural_gradient_direction": [
                float(value) for value in direction.tolist()
            ],
            "D_geo": float(geometric_descent),
            "directional_metric": float(directional_metric),
            "directional_hessian_curvature": float(directional_curvature),
            "eta_star_unconstrained": (
                None if eta_star is None else float(eta_star)
            ),
            "eta_applied": float(applied_eta),
            "eta_clipped": bool(eta_clipped),
            "eta_policy": str(eta_policy),
            "first_order_geometric_reduction": float(first_order_reduction),
            "hessian_correction": float(hessian_correction),
            "joint_gain": float(predicted_gain),
            "contextual_single_total": float(singleton_total),
            "additivity_defect": float(additivity_defect),
            "matrix_elements_reused_from_workspace": int(
                2 * len(key) * (len(key) + 1) // 2
                + 2 * len(self.active_indices) * len(key)
            ),
            "geometry_workspace_cache_hit": False,
        }
        self._subset_cache[key] = dict(summary)
        return summary

    def _supported_metric_summary_for_indices(
        self,
        indices: tuple[int, ...],
    ) -> dict[str, Any]:
        key = tuple(sorted(int(index) for index in indices))
        cached = self._subset_cache.get(key)
        if cached is not None:
            return {**dict(cached), "geometry_workspace_cache_hit": True}
        if not key:
            return {
                "feasible": False,
                "reason": "empty_subset",
                **self._infeasible_incremental_candidate_gain_payload(
                    reason="empty_subset"
                ),
            }

        idx = np.asarray(key, dtype=int)
        active_count = int(len(self.active_indices))
        batch_count = int(len(key))
        G_AA = np.asarray(self.G_AA, dtype=float)
        H_AA = np.asarray(self.H_AA, dtype=float)
        G_AB = np.asarray(self.G_AB[:, idx], dtype=float)
        H_AB = np.asarray(self.H_AB[:, idx], dtype=float)
        G_BB = np.asarray(self.G_BB[np.ix_(idx, idx)], dtype=float)
        H_BB = np.asarray(self.H_BB[np.ix_(idx, idx)], dtype=float)
        g_A = np.asarray(self.g_A, dtype=float)
        g_B = np.asarray(self.g_B[idx], dtype=float)
        G_joint = np.block([[G_AA, G_AB], [G_AB.T, G_BB]])
        H_joint = np.block([[H_AA, H_AB], [H_AB.T, H_BB]])
        G_joint = 0.5 * (G_joint + G_joint.T)
        H_joint = 0.5 * (H_joint + H_joint.T)
        g_joint = np.concatenate([g_A, g_B])

        selected_records = [self.records[int(index)] for index in key]
        selected_labels = [
            str(_batch_record_label(record))
            for record in selected_records
        ]
        coordinate_identities = [
            {
                "candidate_label": str(_batch_record_label(record)),
                "candidate_pool_index": int(
                    record.get("candidate_pool_index", -1)
                ),
                "position_id": int(record.get("position_id", -1)),
                "global_child_identity": str(
                    record.get("route_a_global_pauli_identity") or ""
                ),
            }
            for record in selected_records
        ]
        base_matrix_payload = {
            "geometry_mode": str(self.geometry_mode),
            "joint_batch_context_mode": str(self.joint_context_mode),
            "workspace_fingerprint": str(self.workspace_fingerprint),
            "state_fingerprint": str(self.state_fingerprint),
            "theta_fingerprint": str(self.theta_fingerprint),
            "ordered_scaffold_fingerprint": str(
                self.ordered_scaffold_fingerprint
            ),
            "hamiltonian_fingerprint": str(
                self.hamiltonian_fingerprint
            ),
            "active_indices": [
                int(value) for value in self.active_indices
            ],
            "selected_count": int(batch_count),
            "G_AA_raw": G_AA.tolist(),
            "G_AB_raw": G_AB.tolist(),
            "G_BB_raw": G_BB.tolist(),
            "H_AA_raw": H_AA.tolist(),
            "H_AB_raw": H_AB.tolist(),
            "H_BB_raw": H_BB.tolist(),
            "g_A": [float(value) for value in g_A.tolist()],
            "g_B": [float(value) for value in g_B.tolist()],
            "active_coordinate_identities": [
                str(label) for label in self.active_labels
            ],
            "batch_coordinate_identities": coordinate_identities,
        }

        solve_result = solve_joint_linear_model(
            gram=G_joint,
            hessian=H_joint,
            gradient=g_joint,
            active_coordinate_count=active_count,
            config=JointLinearSolveConfig(
                policy=str(self.joint_linear_solve_policy),
                rank_relative_tolerance=float(self.rank_relative_tolerance),
                metric_regularization=float(self.metric_regularization),
                energy_regularization=float(self.energy_regularization),
                max_fubini_study_step=float(self.max_fubini_study_step),
            ),
        )
        solver_payload = solve_result.as_dict()
        global_trust_v2 = bool(
            str(self.joint_linear_solve_policy)
            == JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2
        )
        sr_certificate_payload: dict[str, Any] = {
            "sr_escape_certificate_schema": (
                "sr_supported_global_trust_certificate_v1"
            ),
            "sr_escape_global_trust_active": bool(global_trust_v2),
        }
        ordinary_solve_result = None
        if global_trust_v2:
            ordinary_solve_result = solve_joint_linear_model(
                gram=G_joint,
                hessian=H_joint,
                gradient=g_joint,
                active_coordinate_count=active_count,
                config=JointLinearSolveConfig(
                    policy=(
                        JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
                    ),
                    rank_relative_tolerance=float(
                        self.rank_relative_tolerance
                    ),
                    metric_regularization=float(self.metric_regularization),
                    energy_regularization=float(self.energy_regularization),
                    max_fubini_study_step=float(
                        self.max_fubini_study_step
                    ),
                ),
            )
            active_solve_payload = _sr_v2_shared_support_active_restriction(
                gram=G_joint,
                hessian=H_joint,
                gradient=g_joint,
                active_count=active_count,
                full_solve_result=solve_result,
                rank_relative_tolerance=float(self.rank_relative_tolerance),
                energy_regularization=float(self.energy_regularization),
                metric_regularization=float(self.metric_regularization),
                max_fubini_study_step=float(self.max_fubini_study_step),
            )
            quotient_payload = _sr_supported_quotient_summary(
                full_solve_result=solve_result,
                shared_support_active_restriction=active_solve_payload,
                hessian_cluster_tolerance=float(
                    solve_result.telemetry.get(
                        "global_trust_eigenspace_tolerance", 0.0
                    )
                ),
            )
            active_gain = float(
                active_solve_payload.get("predicted_reduction", 0.0)
            )
            active_valid = bool(active_solve_payload.get("valid", False))
            full_gain = float(solve_result.predicted_reduction)
            full_error_bound = _sr_trust_gain_numerical_error_bound(
                gain=full_gain,
                telemetry=solve_result.telemetry,
                model_scale=float(
                    active_solve_payload.get(
                        "full_whitened_model_scale", abs(full_gain)
                    )
                ),
                dimension=int(G_joint.shape[0]),
                radius=float(self.max_fubini_study_step),
                energy_regularization=float(self.energy_regularization),
            )
            restricted_solve_payload = active_solve_payload.get(
                "restricted_coordinate_trust_solve"
            )
            active_error_bound = _sr_trust_gain_numerical_error_bound(
                gain=active_gain,
                telemetry=(
                    restricted_solve_payload
                    if isinstance(restricted_solve_payload, Mapping)
                    else {}
                ),
                model_scale=float(
                    active_solve_payload.get(
                        "active_restriction_model_scale", abs(active_gain)
                    )
                ),
                dimension=int(
                    active_solve_payload.get(
                        "active_restriction_supported_dimension",
                        active_solve_payload.get(
                            "active_restriction_supported_nullity", 0
                        ),
                    )
                ),
                radius=float(self.max_fubini_study_step),
                energy_regularization=float(self.energy_regularization),
                direct_objective_residual=float(
                    active_solve_payload.get(
                        "active_restriction_direct_objective_residual", 0.0
                    )
                ),
                subspace_transport_error_bound=float(
                    active_solve_payload.get(
                        "active_image_subspace_gain_error_bound", 0.0
                    )
                ),
            )
            comparison_tolerance = float(
                full_error_bound + active_error_bound
            )
            marginal_gain_raw = float(full_gain - active_gain)
            marginal_comparison_valid = bool(
                solve_result.feasible
                and solve_result.telemetry.get(
                    "trust_global_optimality_certified", False
                )
                and active_valid
                and marginal_gain_raw >= -comparison_tolerance
            )
            full_gain_lower_bound = float(
                max(0.0, full_gain - full_error_bound)
            )
            full_gain_upper_bound = float(full_gain + full_error_bound)
            active_gain_lower_bound = float(
                max(0.0, active_gain - active_error_bound)
            )
            active_gain_upper_bound = float(active_gain + active_error_bound)
            marginal_interval_lower = float(
                full_gain_lower_bound - active_gain_upper_bound
            )
            marginal_interval_upper = float(
                full_gain_upper_bound - active_gain_lower_bound
            )
            hessian_values = [
                float(value)
                for value in solve_result.telemetry.get("H_w_eigenvalues", [])
            ]
            hessian_minimum = (
                float(min(hessian_values)) if hessian_values else None
            )
            hessian_lower_bounds = [
                float(value)
                for value in solve_result.telemetry.get(
                    "supported_hessian_eigenvalue_lower_bounds", []
                )
            ]
            hessian_upper_bounds = [
                float(value)
                for value in solve_result.telemetry.get(
                    "supported_hessian_eigenvalue_upper_bounds", []
                )
            ]
            hessian_minimum_lower_bound = (
                float(min(hessian_lower_bounds))
                if hessian_lower_bounds
                else None
            )
            hessian_minimum_upper_bound = (
                float(min(hessian_upper_bounds))
                if hessian_upper_bounds
                else None
            )
            gradient_norm_raw = quotient_payload.get("whitened_gradient_norm")
            gradient_norm = (
                None if gradient_norm_raw is None else float(gradient_norm_raw)
            )
            stationarity_status = str(
                solve_result.telemetry.get(
                    "supported_stationarity_status", "unresolved"
                )
            )
            stationarity_resolution = float(
                max(
                    0.0,
                    solve_result.telemetry.get(
                        "supported_gradient_resolution", 0.0
                    ),
                )
            )
            stationarity_upper_bound_raw = solve_result.telemetry.get(
                "supported_gradient_norm_upper_bound"
            )
            stationarity_upper_bound = (
                None
                if stationarity_upper_bound_raw is None
                else float(stationarity_upper_bound_raw)
            )
            inertia_status = str(
                solve_result.telemetry.get(
                    "supported_inertia_status", "unresolved"
                )
            )
            inertia_label_issued = bool(
                solve_result.telemetry.get(
                    "supported_inertia_label_issued", False
                )
            )
            quotient_participation = float(
                quotient_payload.get("quotient_participation", 0.0)
            )
            quotient_participation_tolerance = float(
                max(
                    0.0,
                    quotient_payload.get(
                        "quotient_participation_tolerance",
                        quotient_participation,
                    ),
                )
            )
            quotient_participation_lower_bound = float(
                max(
                    0.0,
                    quotient_payload.get(
                        "quotient_participation_lower_bound", 0.0
                    ),
                )
            )
            sr_certificate_payload.update(
                {
                    **dict(quotient_payload),
                    "full_trust_gain": float(full_gain),
                    "full_trust_gain_numerical_error_bound": float(
                        full_error_bound
                    ),
                    "full_trust_gain_lower_bound": float(
                        full_gain_lower_bound
                    ),
                    "full_trust_gain_upper_bound": float(
                        full_gain_upper_bound
                    ),
                    "active_restricted_trust_gain": float(active_gain),
                    "active_restricted_trust_gain_numerical_error_bound": float(
                        active_error_bound
                    ),
                    "active_restricted_subspace_transport_error_bound": float(
                        active_solve_payload.get(
                            "active_image_subspace_gain_error_bound", 0.0
                        )
                    ),
                    "active_restricted_trust_gain_lower_bound": float(
                        active_gain_lower_bound
                    ),
                    "active_restricted_trust_gain_upper_bound": float(
                        active_gain_upper_bound
                    ),
                    "marginal_trust_gain_raw": float(marginal_gain_raw),
                    "marginal_trust_gain_interval_lower_raw": float(
                        marginal_interval_lower
                    ),
                    "marginal_trust_gain_interval_upper_raw": float(
                        marginal_interval_upper
                    ),
                    "marginal_trust_gain_lower_bound": float(
                        max(0.0, marginal_interval_lower)
                        if marginal_comparison_valid
                        else 0.0
                    ),
                    "marginal_trust_gain_upper_bound": (
                        float(max(0.0, marginal_interval_upper))
                        if marginal_comparison_valid
                        else None
                    ),
                    "marginal_trust_gain_numerical_error_bound": float(
                        comparison_tolerance
                    ),
                    "marginal_trust_gain_comparison_tolerance": float(
                        comparison_tolerance
                    ),
                    "marginal_trust_gain_comparison_valid": bool(
                        marginal_comparison_valid
                    ),
                    "active_restriction_source": str(
                        active_solve_payload.get(
                            "active_restriction_source",
                            "full_v2_shared_support_restriction_unresolved",
                        )
                    ),
                    "active_restriction_shared_support_provenance_id": str(
                        active_solve_payload.get(
                            "full_supported_metric_whitening_provenance_id",
                            "",
                        )
                    ),
                    "active_restriction_independent_metric_factorization": bool(
                        active_solve_payload.get(
                            "active_restriction_independent_metric_factorization",
                            True,
                        )
                    ),
                    "active_restriction_solve": dict(active_solve_payload),
                    "whitened_gradient_norm": gradient_norm,
                    "supported_stationarity_status": str(
                        stationarity_status
                    ),
                    "stationarity_tolerance": float(
                        stationarity_resolution
                    ),
                    "stationarity_margin": (
                        None
                        if stationarity_upper_bound is None
                        else float(
                            stationarity_upper_bound
                            - stationarity_resolution
                        )
                    ),
                    "stationary_certified": bool(
                        stationarity_status == "stationary"
                    ),
                    "nonstationary_certified": bool(
                        stationarity_status == "certified_nonstationary"
                    ),
                    "minimum_hessian_eigenvalue": hessian_minimum,
                    "minimum_hessian_eigenvalue_lower_bound": (
                        hessian_minimum_lower_bound
                    ),
                    "minimum_hessian_eigenvalue_upper_bound": (
                        hessian_minimum_upper_bound
                    ),
                    "supported_inertia_status": str(inertia_status),
                    "supported_inertia_label_issued": bool(
                        inertia_label_issued
                    ),
                    "negative_curvature_certified": bool(
                        inertia_label_issued and inertia_status == "negative"
                    ),
                    "positive_semidefinite_certified": bool(
                        inertia_label_issued and inertia_status == "psd"
                    ),
                    "quotient_participation_tolerance": float(
                        quotient_participation_tolerance
                    ),
                    "quotient_participation_lower_bound": float(
                        quotient_participation_lower_bound
                    ),
                }
            )
        support_threshold = float(
            solve_result.telemetry.get("metric_support_threshold", 0.0)
        )
        active_metric_eigenvalues = (
            np.linalg.eigvalsh(0.5 * (G_AA + G_AA.T))
            if active_count
            else np.zeros(0, dtype=float)
        )
        active_metric_rank = int(
            np.count_nonzero(active_metric_eigenvalues > support_threshold)
        )
        joint_metric_rank = int(
            solve_result.telemetry.get("metric_support_rank", 0)
        )
        batch_metric_rank_increment = int(
            max(0, joint_metric_rank - active_metric_rank)
        )
        rank_payload = {
            "active_metric_eigenvalues": [
                float(value) for value in active_metric_eigenvalues.tolist()
            ],
            "active_metric_support_rank": int(active_metric_rank),
            "joint_metric_support_rank": int(joint_metric_rank),
            "batch_metric_rank_increment": int(batch_metric_rank_increment),
            "required_batch_metric_rank_increment": int(batch_count),
            "effective_rank": int(batch_metric_rank_increment),
            "rank_floor": float(support_threshold),
            "rank_relative_tolerance": float(self.rank_relative_tolerance),
        }
        retained_condition_raw = solve_result.telemetry.get(
            "retained_metric_condition_number"
        )
        retained_condition = (
            None
            if retained_condition_raw is None
            else float(retained_condition_raw)
        )
        if ordinary_solve_result is not None:
            ordinary_payload = ordinary_solve_result.as_dict()
            ordinary_support_threshold = float(
                ordinary_solve_result.telemetry.get(
                    "metric_support_threshold", 0.0
                )
            )
            ordinary_active_metric_rank = int(
                np.count_nonzero(
                    active_metric_eigenvalues > ordinary_support_threshold
                )
            )
            ordinary_joint_metric_rank = int(
                ordinary_solve_result.telemetry.get(
                    "metric_support_rank", 0
                )
            )
            ordinary_batch_metric_rank_increment = int(
                max(
                    0,
                    ordinary_joint_metric_rank
                    - ordinary_active_metric_rank,
                )
            )
            ordinary_retained_condition_raw = (
                ordinary_solve_result.telemetry.get(
                    "retained_metric_condition_number"
                )
            )
            ordinary_retained_condition = (
                None
                if ordinary_retained_condition_raw is None
                else float(ordinary_retained_condition_raw)
            )
            if not bool(ordinary_solve_result.feasible):
                ordinary_feasible = False
                ordinary_reason = str(ordinary_solve_result.reason)
            elif ordinary_batch_metric_rank_increment < batch_count:
                ordinary_feasible = False
                ordinary_reason = "rank_gate"
            elif (
                ordinary_retained_condition is not None
                and ordinary_retained_condition
                > float(self.max_gram_condition_number)
            ):
                ordinary_feasible = False
                ordinary_reason = "conditioning_gate"
            else:
                ordinary_feasible = True
                ordinary_reason = (
                    "preserved_supported_metric_whitened_ordinary_v1"
                )
            sr_certificate_payload[
                "sr_escape_ordinary_summary"
            ] = {
                **ordinary_payload,
                "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
                "feasible": bool(ordinary_feasible),
                "reason": str(ordinary_reason),
                "joint_batch_context_mode": str(self.joint_context_mode),
                "joint_gain": float(
                    ordinary_solve_result.predicted_reduction
                    if ordinary_feasible
                    else 0.0
                ),
                "joint_gain_numerical_error_bound": float(
                    4096.0
                    * np.finfo(float).eps
                    * max(1, G_joint.shape[0])
                    * max(
                        1.0,
                        abs(
                            float(
                                ordinary_solve_result.predicted_reduction
                            )
                        ),
                    )
                ),
                "joint_gain_lower_bound": float(
                    max(
                        0.0,
                        float(ordinary_solve_result.predicted_reduction)
                        - 4096.0
                        * np.finfo(float).eps
                        * max(1, G_joint.shape[0])
                        * max(
                            1.0,
                            abs(
                                float(
                                    ordinary_solve_result.predicted_reduction
                                )
                            ),
                        ),
                    )
                    if ordinary_feasible
                    else 0.0
                ),
                "joint_linear_solve_policy_requested": (
                    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
                ),
                "joint_linear_solve_policy_effective": (
                    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
                ),
                "ordinary_selection_authority": (
                    "literal_sr_snake_preserved_v1"
                ),
                "batch_metric_rank_increment": int(
                    ordinary_batch_metric_rank_increment
                ),
                "required_batch_metric_rank_increment": int(batch_count),
                "active_metric_support_rank": int(
                    ordinary_active_metric_rank
                ),
                "joint_metric_support_rank": int(
                    ordinary_joint_metric_rank
                ),
                "metric_support_threshold": float(
                    ordinary_support_threshold
                ),
                "retained_metric_condition_number": (
                    ordinary_retained_condition
                ),
                "max_gram_condition_number": float(
                    self.max_gram_condition_number
                ),
            }
        if not bool(solve_result.feasible):
            summary = {
                "feasible": False,
                "reason": str(solve_result.reason),
                "subset_workspace_indices": [int(value) for value in key],
                "selected_labels": selected_labels,
                **base_matrix_payload,
                **rank_payload,
                **sr_certificate_payload,
                **solver_payload,
                **self._infeasible_incremental_candidate_gain_payload(
                    reason=str(solve_result.reason),
                    solve_result=solve_result,
                ),
            }
            self._subset_cache[key] = dict(summary)
            return summary
        if batch_metric_rank_increment < batch_count:
            summary = {
                "subset_workspace_indices": [int(value) for value in key],
                "selected_labels": selected_labels,
                **base_matrix_payload,
                **rank_payload,
                **sr_certificate_payload,
                **solver_payload,
                "feasible": False,
                "reason": "rank_gate",
                **self._infeasible_incremental_candidate_gain_payload(
                    reason="rank_gate",
                    solve_result=solve_result,
                ),
            }
            self._subset_cache[key] = dict(summary)
            return summary

        if (
            retained_condition is not None
            and retained_condition > float(self.max_gram_condition_number)
        ):
            summary = {
                "subset_workspace_indices": [int(value) for value in key],
                "selected_labels": selected_labels,
                "gram_condition_number": float(retained_condition),
                "max_gram_condition_number": float(
                    self.max_gram_condition_number
                ),
                **base_matrix_payload,
                **rank_payload,
                **sr_certificate_payload,
                **solver_payload,
                "feasible": False,
                "reason": "conditioning_gate",
                **self._infeasible_incremental_candidate_gain_payload(
                    reason="conditioning_gate",
                    solve_result=solve_result,
                ),
            }
            self._subset_cache[key] = dict(summary)
            return summary

        metric_eigenvalues, metric_eigenvectors = np.linalg.eigh(G_joint)
        retained_mask = np.asarray(
            solve_result.telemetry.get("metric_retained_mask", []),
            dtype=bool,
        )
        if retained_mask.size != metric_eigenvalues.size:
            retained_mask = metric_eigenvalues > support_threshold
        retained_values = np.asarray(
            metric_eigenvalues[retained_mask], dtype=float
        )
        retained_vectors = np.asarray(
            metric_eigenvectors[:, retained_mask], dtype=float
        )
        projected_generalized_policy = bool(
            str(self.joint_linear_solve_policy)
            == JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
        )
        geo_denominators = retained_values + float(self.metric_regularization)
        if projected_generalized_policy:
            # This diagnostic follows the already solved physical step.  Do
            # not reintroduce a Gram inverse/ridge after the route explicitly
            # selected direct supported generalized trust coordinates.
            d_geo_full = np.asarray(solve_result.joint_step, dtype=float)
        elif retained_values.size and np.all(geo_denominators > 0.0):
            d_geo_full = np.asarray(
                retained_vectors
                @ (
                    (retained_vectors.T @ g_joint)
                    / geo_denominators
                ),
                dtype=float,
            )
        else:
            d_geo_full = np.zeros_like(g_joint)
        d_geo_active = np.asarray(d_geo_full[:active_count], dtype=float)
        d_geo_batch = np.asarray(d_geo_full[active_count:], dtype=float)
        geometric_descent = float(max(0.0, g_joint.T @ d_geo_full))
        directional_metric = float(
            max(0.0, d_geo_full.T @ G_joint @ d_geo_full)
        )
        directional_curvature = float(d_geo_full.T @ H_joint @ d_geo_full)
        geo_step_norm = float(math.sqrt(directional_metric))
        geo_eta_limit = float(
            self.max_fubini_study_step / geo_step_norm
            if geo_step_norm > max(self.metric_regularization, 1e-15)
            else 0.0
        )
        if directional_curvature > max(self.energy_regularization, 1e-15):
            geo_eta_star: float | None = float(
                geometric_descent / directional_curvature
            )
            geo_eta_applied = float(min(geo_eta_star, geo_eta_limit))
            geo_eta_clipped = bool(geo_eta_applied < geo_eta_star)
            geo_eta_policy = (
                "positive_curvature_trust_safeguard"
                if geo_eta_clipped
                else "positive_curvature_unclipped_optimum"
            )
        else:
            geo_eta_star = None
            geo_eta_applied = float(geo_eta_limit)
            geo_eta_clipped = True
            geo_eta_policy = "nonpositive_curvature_fubini_study_boundary_v1"
        geo_first_order = float(geo_eta_applied * geometric_descent)
        geo_hessian_correction = float(
            0.5 * geo_eta_applied * geo_eta_applied * directional_curvature
        )
        geo_predicted_reduction = float(
            max(0.0, geo_first_order - geo_hessian_correction)
        )

        candidate_gain_policy = str(self.phase3_candidate_gain_policy)
        if candidate_gain_policy not in PHASE3_CANDIDATE_GAIN_POLICIES:
            raise ValueError(
                "phase3_candidate_gain_policy must be one of "
                f"{sorted(PHASE3_CANDIDATE_GAIN_POLICIES)}."
            )
        full_joint_gain = float(
            max(0.0, solve_result.predicted_reduction)
        )
        full_gain_error_bound = float(
            max(
                float(self.energy_regularization),
                4096.0
                * np.finfo(float).eps
                * max(1, int(G_joint.shape[0]))
                * max(1.0, abs(full_joint_gain)),
            )
        )
        active_only_baseline: dict[str, Any] | None = None
        active_only_gain = 0.0
        incremental_gain_raw = float(full_joint_gain)
        gain_comparison_tolerance = float(full_gain_error_bound)
        gain_comparison_status = "legacy_total_joint_gain"
        candidate_gain_feasible = True
        candidate_gain_reason: str | None = None
        if candidate_gain_policy == (
            PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
        ):
            active_only_baseline = self._active_only_gain_baseline()
            if not bool(active_only_baseline.get("feasible", False)):
                candidate_gain_feasible = False
                candidate_gain_reason = "active_only_baseline_infeasible"
                gain_comparison_status = "active_only_baseline_infeasible"
            else:
                active_only_gain = float(
                    active_only_baseline.get("predicted_reduction", 0.0)
                )
                active_error_bound = float(
                    active_only_baseline.get(
                        "gain_numerical_error_bound", 0.0
                    )
                )
                gain_comparison_tolerance = float(
                    full_gain_error_bound + active_error_bound
                )
                incremental_gain_raw = float(
                    full_joint_gain - active_only_gain
                )
                if incremental_gain_raw < -gain_comparison_tolerance:
                    candidate_gain_feasible = False
                    candidate_gain_reason = (
                        "joint_gain_below_active_only_baseline"
                    )
                    gain_comparison_status = (
                        "materially_negative_incremental_gain"
                    )
                elif incremental_gain_raw < 0.0:
                    gain_comparison_status = (
                        "roundoff_negative_increment_clamped_to_zero"
                    )
                else:
                    gain_comparison_status = "incremental_gain_resolved"
        predicted_gain = float(
            max(0.0, incremental_gain_raw)
            if candidate_gain_feasible
            else 0.0
        )
        candidate_gain_receipt = {
            "schema": "phase3_candidate_gain_receipt_v1",
            "policy": str(candidate_gain_policy),
            "joint_gain_semantics": (
                "incremental_candidate_gain_v1"
                if candidate_gain_policy
                == PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
                else "full_joint_trust_gain_legacy_v1"
            ),
            "selection_authority": (
                "joint_minus_candidate_independent_active_only_baseline_v1"
                if candidate_gain_policy
                == PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
                else "full_joint_trust_gain_legacy_v1"
            ),
            "full_joint_trust_gain": float(full_joint_gain),
            "full_joint_gain_numerical_error_bound": float(
                full_gain_error_bound
            ),
            "active_only_trust_gain": float(active_only_gain),
            "incremental_candidate_gain_raw": float(
                incremental_gain_raw
            ),
            "incremental_candidate_gain": float(predicted_gain),
            "selected_gain": float(predicted_gain),
            "comparison_tolerance": float(gain_comparison_tolerance),
            "comparison_status": str(gain_comparison_status),
            "comparison_feasible": bool(candidate_gain_feasible),
            "active_only_baseline": (
                None
                if active_only_baseline is None
                else dict(active_only_baseline)
            ),
            "baseline_candidate_independent": bool(
                active_only_baseline is not None
                and active_only_baseline.get(
                    "candidate_independent", False
                )
            ),
            "classical_quantum_query_charge": 0,
        }
        if batch_count == 1:
            singleton_total = float(predicted_gain)
            additivity_defect = 0.0
        else:
            singleton_total = float(
                sum(
                    float(
                        self._supported_metric_summary_for_indices(
                            (int(index),)
                        ).get("joint_gain", 0.0)
                    )
                    for index in key
                )
            )
            additivity_defect = float(
                max(
                    0.0,
                    1.0
                    - predicted_gain
                    / (
                        singleton_total
                        + max(self.metric_regularization, 1e-15)
                    ),
                )
            )
        coordinate_model_reason = candidate_gain_reason or (
            "supported_metric_projected_generalized_full_joint_model_v1"
            if str(self.joint_linear_solve_policy)
            == JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
            else "supported_metric_whitened_full_joint_model_v1"
        )
        summary = {
            "feasible": bool(candidate_gain_feasible),
            "reason": coordinate_model_reason,
            "geometry_mode": str(self.geometry_mode),
            "joint_batch_context_mode": str(self.joint_context_mode),
            "subset_workspace_indices": [int(value) for value in key],
            "selected_labels": selected_labels,
            "selected_count": int(batch_count),
            **base_matrix_payload,
            **rank_payload,
            **sr_certificate_payload,
            "gram_eigenvalues": [
                float(value) for value in metric_eigenvalues.tolist()
            ],
            "gram_condition_number": retained_condition,
            "max_gram_condition_number": float(
                self.max_gram_condition_number
            ),
            "geometric_direction_policy": (
                "accepted_projected_generalized_trust_step_v1"
                if projected_generalized_policy
                else "regularized_supported_natural_gradient_diagnostic_v1"
            ),
            "natural_gradient_active_relaxation": [
                float(value) for value in d_geo_active.tolist()
            ],
            "natural_gradient_direction": [
                float(value) for value in d_geo_batch.tolist()
            ],
            "D_geo": float(geometric_descent),
            "directional_metric": float(directional_metric),
            "directional_hessian_curvature": float(directional_curvature),
            "eta_star_unconstrained": (
                None if geo_eta_star is None else float(geo_eta_star)
            ),
            "eta_applied": float(geo_eta_applied),
            "eta_clipped": bool(geo_eta_clipped),
            "eta_policy": str(geo_eta_policy),
            "first_order_geometric_reduction": float(geo_first_order),
            "hessian_correction": float(geo_hessian_correction),
            "geo_direction_predicted_reduction": float(
                geo_predicted_reduction
            ),
            "joint_solve_policy": str(self.joint_linear_solve_policy),
            **solver_payload,
            "trust_radius_sq": float(self.max_fubini_study_step) ** 2,
            "trust_radius_binding_tolerance_sq": float(
                max(
                    1e-14,
                    (float(self.max_fubini_study_step) ** 2) * 1e-8,
                )
            ),
            "joint_solve_direct_residual": float(
                solve_result.telemetry.get("full_direct_residual", 0.0)
            ),
            "phase3_candidate_gain_policy": str(
                candidate_gain_policy
            ),
            "joint_gain_semantics": str(
                candidate_gain_receipt["joint_gain_semantics"]
            ),
            "phase3_candidate_gain_receipt": dict(
                candidate_gain_receipt
            ),
            "full_joint_gain": float(full_joint_gain),
            "active_only_gain": float(active_only_gain),
            "incremental_candidate_gain_raw": float(
                incremental_gain_raw
            ),
            "joint_gain": float(predicted_gain),
            "contextual_single_total": float(singleton_total),
            "additivity_defect": float(additivity_defect),
            "matrix_elements_reused_from_workspace": int(
                2 * batch_count * (batch_count + 1) // 2
                + 2 * active_count * batch_count
            ),
            "sr_escape_state_stationarity_summary": (
                self._state_stationarity_summary()
            ),
            "geometry_workspace_cache_hit": False,
        }
        self._subset_cache[key] = dict(summary)
        return summary

    def _summary_for_indices(self, indices: tuple[int, ...]) -> dict[str, Any]:
        if str(self.joint_linear_solve_policy) in {
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
        }:
            return self._supported_metric_summary_for_indices(indices)
        if self.phase3_candidate_gain_policy != (
            PHASE3_CANDIDATE_GAIN_JOINT_TOTAL_V1
        ):
            raise ValueError(
                "Incremental Phase-III candidate gain requires a supported "
                "joint trust solver."
            )
        key = tuple(sorted(int(index) for index in indices))
        cached = self._subset_cache.get(key)
        if cached is not None:
            return {**dict(cached), "geometry_workspace_cache_hit": True}
        if not key:
            return {"feasible": False, "reason": "empty_subset"}

        idx = np.asarray(key, dtype=int)
        active_count = int(len(self.active_indices))
        batch_count = int(len(key))
        G_AA = np.asarray(self.G_AA, dtype=float)
        H_AA = np.asarray(self.H_AA, dtype=float)
        G_AB = np.asarray(self.G_AB[:, idx], dtype=float)
        H_AB = np.asarray(self.H_AB[:, idx], dtype=float)
        G_BA = np.asarray(G_AB.T, dtype=float)
        H_BA = np.asarray(H_AB.T, dtype=float)
        G_BB = np.asarray(self.G_BB[np.ix_(idx, idx)], dtype=float)
        H_BB = np.asarray(self.H_BB[np.ix_(idx, idx)], dtype=float)
        g_A = np.asarray(self.g_A, dtype=float)
        g_B = np.asarray(self.g_B[idx], dtype=float)

        if active_count:
            G_AA_regularized = 0.5 * (G_AA + G_AA.T) + float(
                self.metric_regularization
            ) * np.eye(active_count, dtype=float)
            G_AA_inverse = np.linalg.pinv(
                G_AA_regularized,
                rcond=max(float(self.metric_regularization), 1e-15),
            )
            G_effective = G_BB - G_BA @ G_AA_inverse @ G_AB
            g_effective_geo = g_B - G_BA @ G_AA_inverse @ g_A
        else:
            G_AA_regularized = np.zeros((0, 0), dtype=float)
            G_AA_inverse = np.zeros((0, 0), dtype=float)
            G_effective = np.asarray(G_BB, dtype=float)
            g_effective_geo = np.asarray(g_B, dtype=float)
        G_effective = 0.5 * (G_effective + G_effective.T)
        gram_eigenvalues = np.linalg.eigvalsh(G_effective)
        raw_gram_scale = float(
            max(
                np.max(np.abs(np.linalg.eigvalsh(0.5 * (G_BB + G_BB.T))))
                if G_BB.size
                else 0.0,
                self.metric_regularization,
            )
        )
        gram_scale = float(
            max(
                np.max(np.abs(gram_eigenvalues)) if gram_eigenvalues.size else 0.0,
                self.metric_regularization,
            )
        )
        rank_floor = float(
            max(
                10.0 * self.metric_regularization,
                self.rank_relative_tolerance * raw_gram_scale,
                self.rank_relative_tolerance * gram_scale,
            )
        )
        effective_rank = int(np.count_nonzero(gram_eigenvalues > rank_floor))
        residual_to_raw_scale_ratio = float(
            gram_scale / raw_gram_scale if raw_gram_scale > 0.0 else 0.0
        )
        base_matrix_payload = {
            "G_AA_raw": G_AA.tolist(),
            "G_AB_raw": G_AB.tolist(),
            "G_BB_raw": G_BB.tolist(),
            "G_effective": G_effective.tolist(),
            "H_AA_raw": H_AA.tolist(),
            "H_AB_raw": H_AB.tolist(),
            "H_BB_raw": H_BB.tolist(),
            "gram_eigenvalues": [
                float(value) for value in gram_eigenvalues.tolist()
            ],
            "effective_rank": int(effective_rank),
            "rank_floor": float(rank_floor),
            "rank_relative_tolerance": float(self.rank_relative_tolerance),
            "raw_gram_scale": float(raw_gram_scale),
            "effective_gram_scale": float(gram_scale),
            "residual_to_raw_scale_ratio": float(
                residual_to_raw_scale_ratio
            ),
        }
        if effective_rank < batch_count:
            summary = {
                "feasible": False,
                "reason": "rank_gate",
                "subset_workspace_indices": [int(value) for value in key],
                **base_matrix_payload,
            }
            self._subset_cache[key] = dict(summary)
            return summary
        positive_eigenvalues = gram_eigenvalues[gram_eigenvalues > rank_floor]
        gram_condition = float(
            np.max(positive_eigenvalues) / np.min(positive_eigenvalues)
        )
        if gram_condition > float(self.max_gram_condition_number):
            summary = {
                "feasible": False,
                "reason": "conditioning_gate",
                "subset_workspace_indices": [int(value) for value in key],
                "gram_condition_number": float(gram_condition),
                "max_gram_condition_number": float(
                    self.max_gram_condition_number
                ),
                **base_matrix_payload,
            }
            self._subset_cache[key] = dict(summary)
            return summary

        G_effective_regularized = G_effective + float(
            self.metric_regularization
        ) * np.eye(batch_count, dtype=float)
        d_geo_batch = np.asarray(
            np.linalg.pinv(
                G_effective_regularized,
                rcond=max(float(self.metric_regularization), 1e-15),
            )
            @ g_effective_geo,
            dtype=float,
        )
        d_geo_active = (
            np.asarray(
                G_AA_inverse @ (g_A - G_AB @ d_geo_batch),
                dtype=float,
            )
            if active_count
            else np.zeros(0, dtype=float)
        )
        d_geo_full = np.concatenate([d_geo_active, d_geo_batch])
        G_joint = np.block(
            [[G_AA, G_AB], [G_BA, G_BB]]
        )
        H_joint = np.block(
            [[H_AA, H_AB], [H_BA, H_BB]]
        )
        G_joint = 0.5 * (G_joint + G_joint.T)
        H_joint = 0.5 * (H_joint + H_joint.T)
        g_joint = np.concatenate([g_A, g_B])
        geometric_descent = float(
            max(0.0, g_effective_geo.T @ d_geo_batch)
        )
        directional_metric = float(
            max(0.0, d_geo_full.T @ G_joint @ d_geo_full)
        )
        directional_curvature = float(d_geo_full.T @ H_joint @ d_geo_full)
        geo_step_norm = float(math.sqrt(directional_metric))
        geo_eta_limit = float(
            self.max_fubini_study_step / geo_step_norm
            if geo_step_norm > max(self.metric_regularization, 1e-15)
            else 0.0
        )
        if directional_curvature > max(self.energy_regularization, 1e-15):
            geo_eta_star: float | None = float(
                geometric_descent / directional_curvature
            )
            geo_eta_applied = float(min(geo_eta_star, geo_eta_limit))
            geo_eta_clipped = bool(geo_eta_applied < geo_eta_star)
            geo_eta_policy = (
                "positive_curvature_trust_safeguard"
                if geo_eta_clipped
                else "positive_curvature_unclipped_optimum"
            )
        else:
            geo_eta_star = None
            geo_eta_applied = float(geo_eta_limit)
            geo_eta_clipped = True
            geo_eta_policy = "nonpositive_curvature_fubini_study_boundary_v1"
        geo_first_order = float(geo_eta_applied * geometric_descent)
        geo_hessian_correction = float(
            0.5 * geo_eta_applied * geo_eta_applied * directional_curvature
        )
        geo_predicted_reduction = float(
            max(0.0, geo_first_order - geo_hessian_correction)
        )

        total_count = int(active_count + batch_count)
        energy_floor = float(max(self.energy_regularization, 1e-15))

        def _solve_at_lambda(trust_lambda: float) -> dict[str, Any]:
            M_joint = H_joint + float(trust_lambda) * G_joint
            M_joint_regularized = 0.5 * (M_joint + M_joint.T) + energy_floor * np.eye(
                total_count, dtype=float
            )
            if active_count:
                M_AA = M_joint_regularized[:active_count, :active_count]
                M_AB = M_joint_regularized[:active_count, active_count:]
                M_BA = M_joint_regularized[active_count:, :active_count]
                M_BB = M_joint_regularized[active_count:, active_count:]
                M_AA_inverse = np.linalg.pinv(
                    M_AA,
                    rcond=energy_floor,
                )
                M_effective = M_BB - M_BA @ M_AA_inverse @ M_AB
                g_effective = g_B - M_BA @ M_AA_inverse @ g_A
                batch_step = np.asarray(
                    np.linalg.pinv(M_effective, rcond=energy_floor)
                    @ g_effective,
                    dtype=float,
                )
                active_step = np.asarray(
                    M_AA_inverse @ (g_A - M_AB @ batch_step),
                    dtype=float,
                )
            else:
                M_AA = np.zeros((0, 0), dtype=float)
                M_effective = np.asarray(M_joint_regularized, dtype=float)
                g_effective = np.asarray(g_B, dtype=float)
                batch_step = np.asarray(
                    np.linalg.pinv(M_effective, rcond=energy_floor)
                    @ g_effective,
                    dtype=float,
                )
                active_step = np.zeros(0, dtype=float)
            joint_step = np.concatenate([active_step, batch_step])
            displacement_sq = float(
                max(0.0, joint_step.T @ G_joint @ joint_step)
            )
            predicted_reduction = float(
                g_joint.T @ joint_step
                - 0.5 * joint_step.T @ H_joint @ joint_step
            )
            eigenvalues = np.linalg.eigvalsh(M_joint_regularized)
            return {
                "trust_lambda": float(trust_lambda),
                "M_joint_regularized": M_joint_regularized,
                "M_AA_regularized": M_AA,
                "M_effective": 0.5 * (M_effective + M_effective.T),
                "g_effective": np.asarray(g_effective, dtype=float),
                "active_step": np.asarray(active_step, dtype=float),
                "batch_step": np.asarray(batch_step, dtype=float),
                "joint_step": np.asarray(joint_step, dtype=float),
                "displacement_sq": float(displacement_sq),
                "predicted_reduction": float(predicted_reduction),
                "minimum_eigenvalue": float(np.min(eigenvalues)),
                "direct_solve_residual": float(
                    np.linalg.norm(M_joint_regularized @ joint_step - g_joint)
                ),
            }

        radius_sq = float(self.max_fubini_study_step) ** 2

        def _trust_feasible(solution: Mapping[str, Any]) -> bool:
            return bool(
                math.isfinite(float(solution["predicted_reduction"]))
                and float(solution["minimum_eigenvalue"]) > 0.0
                and float(solution["displacement_sq"])
                <= radius_sq * (1.0 + 1e-10)
            )

        unconstrained_solution = _solve_at_lambda(0.0)
        if _trust_feasible(unconstrained_solution):
            applied_solution = unconstrained_solution
            trust_clipped = False
        else:
            low = 0.0
            high = float(max(energy_floor, 1e-12))
            applied_solution: dict[str, Any] | None = None
            for _ in range(80):
                trial = _solve_at_lambda(high)
                if _trust_feasible(trial):
                    applied_solution = trial
                    break
                low = high
                high *= 2.0
            if applied_solution is None:
                summary = {
                    "feasible": False,
                    "reason": "joint_trust_solve_failed",
                    "subset_workspace_indices": [int(value) for value in key],
                    **base_matrix_payload,
                }
                self._subset_cache[key] = dict(summary)
                return summary
            for _ in range(64):
                midpoint = 0.5 * (low + high)
                trial = _solve_at_lambda(midpoint)
                if _trust_feasible(trial):
                    high = midpoint
                    applied_solution = trial
                else:
                    low = midpoint
            trust_clipped = True

        trust_regularization_applied = bool(trust_clipped)
        applied_displacement_sq = float(applied_solution["displacement_sq"])
        trust_radius_binding_tolerance_sq = float(
            max(1e-14, radius_sq * 1e-8)
        )
        trust_radius_binding = bool(
            trust_regularization_applied
            and abs(applied_displacement_sq - radius_sq)
            <= trust_radius_binding_tolerance_sq
        )

        predicted_gain = float(
            max(0.0, float(applied_solution["predicted_reduction"]))
        )
        if batch_count == 1:
            singleton_total = float(predicted_gain)
            additivity_defect = 0.0
        else:
            singleton_total = float(
                sum(
                    float(
                        self._summary_for_indices((int(index),)).get(
                            "joint_gain", 0.0
                        )
                    )
                    for index in key
                )
            )
            additivity_defect = float(
                max(
                    0.0,
                    1.0
                    - predicted_gain
                    / (singleton_total + max(self.metric_regularization, 1e-15)),
                )
            )
        summary = {
            "feasible": True,
            "reason": "shared_full_ansatz_batch_joint_model_v1",
            "geometry_mode": str(self.geometry_mode),
            "joint_batch_context_mode": str(self.joint_context_mode),
            "subset_workspace_indices": [int(value) for value in key],
            "selected_labels": [
                str(self.records[int(index)].get("candidate_label", ""))
                for index in key
            ],
            "selected_count": int(batch_count),
            **base_matrix_payload,
            "G_AA_regularized": G_AA_regularized.tolist(),
            "g_A": [float(value) for value in g_A.tolist()],
            "g_B": [float(value) for value in g_B.tolist()],
            "g_effective_geo": [
                float(value) for value in g_effective_geo.tolist()
            ],
            "gram_condition_number": float(gram_condition),
            "max_gram_condition_number": float(
                self.max_gram_condition_number
            ),
            "natural_gradient_active_relaxation": [
                float(value) for value in d_geo_active.tolist()
            ],
            "natural_gradient_direction": [
                float(value) for value in d_geo_batch.tolist()
            ],
            "D_geo": float(geometric_descent),
            "directional_metric": float(directional_metric),
            "directional_hessian_curvature": float(directional_curvature),
            "eta_star_unconstrained": (
                None if geo_eta_star is None else float(geo_eta_star)
            ),
            "eta_applied": float(geo_eta_applied),
            "eta_clipped": bool(geo_eta_clipped),
            "eta_policy": str(geo_eta_policy),
            "first_order_geometric_reduction": float(geo_first_order),
            "hessian_correction": float(geo_hessian_correction),
            "geo_direction_predicted_reduction": float(geo_predicted_reduction),
            "joint_solve_policy": "schur_of_H_plus_lambda_G_v1",
            "joint_linear_solve_policy_requested": str(
                self.joint_linear_solve_policy
            ),
            "joint_linear_solve_policy_effective": str(
                self.joint_linear_solve_policy
            ),
            "classical_quantum_query_charge": 0,
            "trust_lambda": float(applied_solution["trust_lambda"]),
            "trust_clipped": bool(trust_clipped),
            "trust_regularization_applied": bool(
                trust_regularization_applied
            ),
            "trust_radius_binding": bool(trust_radius_binding),
            "trust_radius_sq": float(radius_sq),
            "trust_radius_binding_tolerance_sq": float(
                trust_radius_binding_tolerance_sq
            ),
            "M_joint_regularized": applied_solution[
                "M_joint_regularized"
            ].tolist(),
            "M_AA_regularized": applied_solution[
                "M_AA_regularized"
            ].tolist(),
            "M_effective": applied_solution["M_effective"].tolist(),
            "g_effective_joint": [
                float(value)
                for value in applied_solution["g_effective"].tolist()
            ],
            "active_parameter_relaxation": [
                float(value)
                for value in applied_solution["active_step"].tolist()
            ],
            "batch_coordinate_step": [
                float(value)
                for value in applied_solution["batch_step"].tolist()
            ],
            "joint_fubini_study_displacement_sq": float(
                applied_solution["displacement_sq"]
            ),
            "unconstrained_predicted_reduction": float(
                unconstrained_solution["predicted_reduction"]
            ),
            "applied_predicted_reduction": float(predicted_gain),
            "joint_solve_direct_residual": float(
                applied_solution["direct_solve_residual"]
            ),
            "joint_gain": float(predicted_gain),
            "contextual_single_total": float(singleton_total),
            "additivity_defect": float(additivity_defect),
            "matrix_elements_reused_from_workspace": int(
                2 * batch_count * (batch_count + 1) // 2
                + 2 * active_count * batch_count
            ),
            "geometry_workspace_cache_hit": False,
        }
        self._subset_cache[key] = dict(summary)
        return summary

    def summary_for_records(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        try:
            indices = tuple(
                int(self.record_index[_batch_record_identity_key(record)])
                for record in records
            )
        except KeyError:
            return {"feasible": False, "reason": "record_missing_from_workspace"}
        return self._summary_for_indices(indices)


@dataclass(frozen=True)
class Phase2JointResponseEvaluation:
    """State-scoped Phase-II singleton evaluation over one shared workspace."""

    records: tuple[dict[str, Any], ...]
    telemetry: dict[str, Any]
    workspace: _BatchFullGeometryWorkspace = field(repr=False, compare=False)


HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA = (
    "historical_singleton_coordinate_model_v1"
)
HISTORICAL_SINGLETON_COORDINATE_MODEL_POPULATION_SCHEMA = (
    "historical_singleton_coordinate_model_population_v1"
)
HISTORICAL_SINGLETON_PHASE2_WHITENED_SCORE_FORMULA = (
    "DeltaE_TR_supported_metric_joint * N2 / (1 + K2)"
)
HISTORICAL_SINGLETON_PHASE2_WHITENED_NO_N2_SCORE_FORMULA = (
    "DeltaE_TR_supported_metric_joint / (1 + K2)"
)


def evaluate_historical_singleton_phase2_coordinate_models(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    pauli_action_cache: dict[str, Any] | None = None,
    scope: str = "historical_phase2_whitening",
) -> Phase2JointResponseEvaluation:
    """Replace the scalar Phase-II benefit with supported joint-model gain.

    Ordinary novelty is not evaluated or applied.  Any historical N2 values
    already present on candidate snapshots are retained as passive provenance
    while ranking uses only supported gain over the saved cost denominator.
    """

    copied = [dict(record) for record in records]
    score_formula = HISTORICAL_SINGLETON_PHASE2_WHITENED_NO_N2_SCORE_FORMULA
    singleton_cfg = replace(
        cfg,
        batch_target_size=1,
        batch_size_cap=1,
    )
    workspace = _build_batch_full_geometry_workspace(
        copied,
        cfg=singleton_cfg,
        selected_ops=selected_ops,
        theta=np.asarray(theta, dtype=float),
        psi_ref=np.asarray(psi_ref, dtype=complex),
        psi_state=np.asarray(psi_state, dtype=complex),
        h_compiled=h_compiled,
        pauli_action_cache=pauli_action_cache,
    )
    evaluated: list[dict[str, Any]] = []
    record_payloads: list[dict[str, Any]] = []
    infeasible_reasons: dict[str, int] = {}
    passive_n2_count = 0
    for input_index, record in enumerate(copied):
        feature = record.get("feature")
        if not isinstance(feature, CandidateFeatures):
            raise ValueError(
                "Historical singleton Phase-II response requires "
                "CandidateFeatures on every retained record."
            )
        if feature.phase2_burden_total is None:
            raise ValueError(
                "Historical singleton Phase-II response requires a saved "
                "1 + K2 denominator."
            )
        denominator = float(feature.phase2_burden_total)
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(
                "Historical singleton Phase-II 1 + K2 must be finite and "
                "positive."
            )
        passive_n2: float | None = None
        if feature.phase2_raw_novelty is not None:
            try:
                candidate_n2 = float(feature.phase2_raw_novelty)
            except (TypeError, ValueError):
                candidate_n2 = float("nan")
            if math.isfinite(candidate_n2):
                passive_n2 = float(candidate_n2)
                passive_n2_count += 1

        raw_summary = dict(workspace.summary_for_records([record]))
        feasible = bool(raw_summary.get("feasible", False))
        reason = str(raw_summary.get("reason", "unknown"))
        if not feasible:
            infeasible_reasons[reason] = int(
                infeasible_reasons.get(reason, 0) + 1
            )
        joint_gain_raw = float(raw_summary.get("joint_gain", 0.0))
        if not math.isfinite(joint_gain_raw) or joint_gain_raw < 0.0:
            raise ValueError(
                "Historical singleton Phase-II joint gain must be finite and "
                "nonnegative."
            )
        joint_gain = float(joint_gain_raw if feasible else 0.0)
        score = (
            float(
                joint_gain
                / max(denominator, float(singleton_cfg.cheap_score_eps))
            )
            if feasible
            else float("-inf")
        )
        model = {
            **raw_summary,
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "scope": str(scope),
            "authority": "phase2_supported_response_gain_only",
            "score_formula": str(score_formula),
            "input_record_index": int(input_index),
            "candidate_label": str(_batch_record_label(record)),
            "candidate_pool_index": int(
                record.get("candidate_pool_index", feature.candidate_pool_index)
            ),
            "position_id": int(record.get("position_id", feature.position_id)),
            "historical_phase2_raw_score": (
                None
                if feature.phase2_raw_score is None
                else float(feature.phase2_raw_score)
            ),
            "historical_phase2_raw_score_formula": str(
                feature.phase2_raw_score_formula
            ),
            "historical_phase2_raw_trust_gain": (
                None
                if feature.phase2_raw_trust_gain is None
                else float(feature.phase2_raw_trust_gain)
            ),
            "preserved_N2": passive_n2,
            "measured_N2": passive_n2,
            "phase2_novelty_authority": "passive_provenance_only",
            "phase2_novelty_multiplier": None,
            "phase2_novelty_multiplier_policy": (
                ORDINARY_NOVELTY_SCORING_RETIRED_V1
            ),
            "phase2_gram_novelty_policy": (
                GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1
                if _deferred_gram_fallback_enabled(cfg)
                else "off"
            ),
            "phase2_novelty_status": (
                GRAM_NOVELTY_STATUS_NOT_COMPUTED_FOR_ORDINARY_SCORING
            ),
            "phase2_novelty_query_charge": 0,
            "phase2_novelty_classical_solve_count": 0,
            "phase2_novelty_applied": False,
            "preserved_denominator_1_plus_K2": float(denominator),
            "phase2_joint_geometry_reuse_preserved": dict(
                feature.phase2_joint_geometry_reuse or {}
            ),
            "whitened_phase2_raw_score": float(score),
        }
        updated_feature = _replace_feature(
            feature,
            historical_singleton_phase2_coordinate_model=dict(model),
            phase2_raw_score_formula=str(score_formula),
            phase2_raw_trust_gain=float(joint_gain),
            phase2_raw_score=float(score),
            selector_score=float(score),
            selector_burden=float(denominator),
            phase_score_components={
                **dict(feature.phase_score_components or {}),
                "phase2_historical_scalar_DeltaE_TR": (
                    None
                    if feature.phase2_raw_trust_gain is None
                    else float(feature.phase2_raw_trust_gain)
                ),
                "phase2_supported_metric_joint_DeltaE_TR": float(joint_gain),
                "phase2_raw_novelty": passive_n2,
                "N2": passive_n2,
                "phase2_measured_novelty": passive_n2,
                "phase2_novelty_authority": "passive_provenance_only",
                "phase2_novelty_multiplier": None,
                "phase2_novelty_multiplier_policy": (
                    ORDINARY_NOVELTY_SCORING_RETIRED_V1
                ),
                "phase2_gram_novelty_policy": (
                    GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1
                    if _deferred_gram_fallback_enabled(cfg)
                    else "off"
                ),
                "phase2_novelty_status": (
                    GRAM_NOVELTY_STATUS_NOT_COMPUTED_FOR_ORDINARY_SCORING
                ),
                "phase2_novelty_query_charge": 0,
                "phase2_novelty_applied": False,
                "phase2_burden_total": float(denominator),
                "denominator_1_plus_K2": float(denominator),
                "phase2_raw_score": float(score),
                "phase2_raw_score_formula": str(score_formula),
            },
        )
        updated = {
            **record,
            "feature": updated_feature,
            "phase2_raw_score": float(score),
            "phase2_raw_score_formula": str(score_formula),
            "phase2_raw_trust_gain": float(joint_gain),
            "phase2_selector_mode": (
                "historical_singleton_supported_metric_response_v1"
            ),
            "historical_singleton_phase2_coordinate_model": dict(model),
            "selector_score": float(score),
            "selector_burden": float(denominator),
        }
        evaluated.append(updated)
        record_payloads.append(dict(model))

    workspace_telemetry = dict(workspace.build_telemetry())
    telemetry = {
        "schema": "historical_singleton_phase2_coordinate_population_v1",
        "scope": str(scope),
        "authority": "phase2_supported_response_gain_only",
        "score_formula": str(score_formula),
        "input_record_count": int(len(copied)),
        "output_record_count": int(len(evaluated)),
        "membership_preserved": True,
        "order_preserved": True,
        "historical_n2_retained_as_passive_provenance": True,
        "passive_n2_record_count": int(passive_n2_count),
        "phase2_gram_novelty_policy": (
            GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1
            if _deferred_gram_fallback_enabled(cfg)
            else "off"
        ),
        "phase2_novelty_status": (
            GRAM_NOVELTY_STATUS_NOT_COMPUTED_FOR_ORDINARY_SCORING
        ),
        "phase2_novelty_query_charge": 0,
        "phase2_novelty_classical_solve_count": 0,
        "phase2_novelty_multiplier_policy": (
            ORDINARY_NOVELTY_SCORING_RETIRED_V1
        ),
        "phase2_novelty_applied": False,
        "measured_n2_retained": bool(passive_n2_count),
        "cost_denominator_preserved": True,
        "phase3_fields_preserved": True,
        "batching_applied": False,
        "effective_batch_size_cap": 1,
        "joint_linear_solve_policy_requested": str(
            singleton_cfg.batch_joint_linear_solve_policy
        ),
        "feasible_record_count": int(
            sum(bool(row.get("feasible", False)) for row in record_payloads)
        ),
        "infeasible_record_count": int(
            sum(not bool(row.get("feasible", False)) for row in record_payloads)
        ),
        "infeasible_reason_counts": dict(sorted(infeasible_reasons.items())),
        "candidate_pair_measurement_count": 0,
        "records": record_payloads,
        "geometry_workspace": workspace_telemetry,
    }
    return Phase2JointResponseEvaluation(
        records=tuple(evaluated),
        telemetry=telemetry,
        workspace=workspace,
    )

def evaluate_historical_singleton_coordinate_models(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    pauli_action_cache: dict[str, Any] | None = None,
    scope: str = "historical_phase3",
    old_old_geometry_prior: HistoricalSingletonOldOldGeometryPrior | None = None,
    require_complete_upstream_geometry: bool = False,
) -> Phase2JointResponseEvaluation:
    """Evaluate retained historical singletons in one shared coordinate model.

    This seam deliberately does *not* invoke the Joint-Response Phase-II
    selector and does not replace ``phase2_raw_score`` or any shortlist field.
    It preserves record membership and order, forces the existing shared
    workspace into its singleton limit, and attaches the complete raw model
    summary to the feature for a later, pure Phase-III benefit substitution.
    """

    copied = [dict(record) for record in records]
    singleton_cfg = replace(
        cfg,
        batch_target_size=1,
        batch_size_cap=1,
    )
    workspace = _build_batch_full_geometry_workspace(
        copied,
        cfg=singleton_cfg,
        selected_ops=selected_ops,
        theta=np.asarray(theta, dtype=float),
        psi_ref=np.asarray(psi_ref, dtype=complex),
        psi_state=np.asarray(psi_state, dtype=complex),
        h_compiled=h_compiled,
        pauli_action_cache=pauli_action_cache,
        old_old_geometry_prior=old_old_geometry_prior,
        require_complete_upstream_geometry=bool(
            require_complete_upstream_geometry
        ),
    )
    active_gradient_policy = str(
        getattr(
            singleton_cfg,
            "active_gradient_policy",
            "measured_residual_response_v1",
        )
    ).strip()
    if active_gradient_policy not in {
        "stationary_source_response_v1",
        "measured_residual_response_v1",
    }:
        raise ValueError(
            "active_gradient_policy must be one of "
            "{'stationary_source_response_v1',"
            "'measured_residual_response_v1'}."
        )
    if active_gradient_policy == "stationary_source_response_v1":
        # The stationary-source model is a coupling-only Schur response.
        # The candidate gradient remains measured; only the active source
        # residual is fixed to zero by protocol.
        workspace.g_A = np.zeros_like(
            np.asarray(workspace.g_A, dtype=float)
        )
        workspace._subset_cache.clear()
        workspace._state_stationarity_cache = None
        workspace._active_only_gain_baseline_cache = None
    evaluated: list[dict[str, Any]] = []
    record_payloads: list[dict[str, Any]] = []
    infeasible_reasons: dict[str, int] = {}
    for input_index, record in enumerate(copied):
        feature = record.get("feature")
        if not isinstance(feature, CandidateFeatures):
            raise ValueError(
                "Historical singleton coordinate-model evaluation requires "
                "CandidateFeatures on every retained record."
            )
        raw_summary = dict(workspace.summary_for_records([record]))
        feasible = bool(raw_summary.get("feasible", False))
        reason = str(raw_summary.get("reason", "unknown"))
        if not feasible:
            infeasible_reasons[reason] = int(
                infeasible_reasons.get(reason, 0) + 1
            )
        prior_reuse = dict(feature.phase2_joint_geometry_reuse or {})
        attached_summary = {
            **raw_summary,
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "scope": str(scope),
            "authority": "historical_phase3_benefit_overlay_only",
            "input_record_index": int(input_index),
            "candidate_label": str(_batch_record_label(record)),
            "candidate_pool_index": int(
                record.get("candidate_pool_index", feature.candidate_pool_index)
            ),
            "position_id": int(record.get("position_id", feature.position_id)),
            "historical_phase2_joint_geometry_reuse": prior_reuse,
        }
        supported_rank_raw = raw_summary.get("joint_metric_support_rank")
        supported_rank = (
            None
            if supported_rank_raw is None
            else int(supported_rank_raw)
        )
        updated = dict(record)
        updated["feature"] = _replace_feature(
            feature,
            phase2_joint_geometry_reuse=dict(attached_summary),
            phase3_response_supported_rank=supported_rank,
        )
        evaluated.append(updated)
        record_payloads.append(dict(attached_summary))

    workspace_telemetry = dict(workspace.build_telemetry())
    telemetry = {
        "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_POPULATION_SCHEMA,
        "scope": str(scope),
        "authority": "historical_phase3_benefit_overlay_only",
        "input_record_count": int(len(copied)),
        "output_record_count": int(len(evaluated)),
        "membership_preserved": True,
        "order_preserved": True,
        "phase2_rescoring_applied": False,
        "batching_applied": False,
        "effective_batch_size_cap": 1,
        "joint_linear_solve_policy_requested": str(
            singleton_cfg.batch_joint_linear_solve_policy
        ),
        "active_gradient_policy": str(active_gradient_policy),
            "active_gradient_indices_acquired": (
                []
                if active_gradient_policy == "stationary_source_response_v1"
                else [
                    int(value)
                    for value in getattr(workspace, "active_indices", ())
                ]
            ),
        "active_source_gradient_fixed_zero": bool(
            active_gradient_policy == "stationary_source_response_v1"
        ),
        "feasible_record_count": int(
            sum(bool(row.get("feasible", False)) for row in record_payloads)
        ),
        "infeasible_record_count": int(
            sum(not bool(row.get("feasible", False)) for row in record_payloads)
        ),
        "infeasible_reason_counts": dict(sorted(infeasible_reasons.items())),
        "records": record_payloads,
        "geometry_workspace": workspace_telemetry,
        **(
            {
                "old_old_geometry_prior": (
                    old_old_geometry_prior.telemetry()
                ),
                "old_old_geometry_reacquired": False,
            }
            if old_old_geometry_prior is not None
            else {}
        ),
    }
    return Phase2JointResponseEvaluation(
        records=tuple(evaluated),
        telemetry=telemetry,
        workspace=workspace,
    )


def evaluate_historical_singleton_material_window_coordinate_models(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    material_window_policy: Phase3MaterialWindowPolicy,
    pauli_action_cache: dict[str, Any] | None = None,
    scope: str = "historical_phase3_material_window",
    prior_retained_support_nullities: Mapping[
        tuple[int, int, str], tuple[int | None, int | None]
    ]
    | None = None,
) -> Phase2JointResponseEvaluation:
    """Evaluate each historical singleton in its exact material window.

    Candidate coupling is screened against every active coordinate without an
    old--old block.  Each record then receives an exact retained ``W x W``
    model while ``W x O`` is acquired as closure evidence.  A planner closure
    failure or retained support-nullity drift acquires ``O x O`` exactly once
    and rebuilds that record in the full active chart.
    """

    if not isinstance(material_window_policy, Phase3MaterialWindowPolicy):
        raise TypeError(
            "material_window_policy must be a Phase3MaterialWindowPolicy."
        )
    copied = [dict(record) for record in records]
    selected = list(selected_ops)
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    if theta_vec.size != len(selected):
        raise ValueError(
            "Material-window coordinate evaluation theta/ansatz size mismatch."
        )
    for record in copied:
        if not isinstance(record.get("feature"), CandidateFeatures):
            raise ValueError(
                "Material-window coordinate evaluation requires "
                "CandidateFeatures on every record."
            )
        if record.get("candidate_term") is None:
            raise ValueError(
                "Material-window coordinate evaluation requires candidate_term."
            )

    singleton_cfg = replace(
        cfg,
        batch_target_size=1,
        batch_size_cap=1,
        batch_joint_context_mode=BATCH_JOINT_CONTEXT_ACTIVE_WINDOW_V1,
        batch_joint_linear_solve_policy=(
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
        ),
    )
    all_active = tuple(range(len(selected)))
    screen_context = _selector_scaffold_context(
        selected_ops=selected,
        theta=theta_vec,
        psi_ref=np.asarray(psi_ref, dtype=complex),
        psi_state=np.asarray(psi_state, dtype=complex),
        active_indices=all_active,
        h_compiled=h_compiled,
        measure_old_old_geometry=False,
    )
    tolerance = float(
        max(
            1e-12,
            getattr(singleton_cfg, "batch_state_consistency_tolerance", 1e-8),
        )
    )
    sparse_cache: dict[
        tuple[tuple[tuple[int, int], ...], tuple[int, ...]], dict[str, Any]
    ] = {}

    def _sparse_acquisition(
        *,
        pairs: Sequence[Sequence[int]],
        gradients: Sequence[int],
        stage: str,
    ) -> dict[str, Any]:
        pair_key = tuple(
            (int(pair[0]), int(pair[1])) for pair in pairs
        )
        gradient_key = tuple(int(value) for value in gradients)
        key = (pair_key, gradient_key)
        cached = sparse_cache.get(key)
        if cached is not None:
            return {
                **dict(cached),
                "acquisition_stage": str(stage),
                "shared_exact_acquisition_reused": True,
            }
        measured = _acquire_sparse_exact_old_old_geometry(
            scaffold_context=screen_context,
            pair_plan=pair_key,
            gradient_indices=gradient_key,
            h_compiled=h_compiled,
            pauli_action_cache=pauli_action_cache,
            state_consistency_tolerance=tolerance,
            acquisition_stage=str(stage),
        )
        measured["shared_exact_acquisition_reused"] = False
        sparse_cache[key] = dict(measured)
        return measured

    def _geometry_maps(
        acquisitions: Sequence[Mapping[str, Any]],
    ) -> tuple[
        dict[tuple[int, int], float],
        dict[tuple[int, int], float],
        dict[int, float],
    ]:
        gram: dict[tuple[int, int], float] = {}
        hessian: dict[tuple[int, int], float] = {}
        gradient: dict[int, float] = {}
        for acquisition in acquisitions:
            for entry in acquisition.get("pair_entries", []):
                raw_pair = entry.get("active_index_pair", [])
                if len(raw_pair) != 2:
                    raise ValueError("Material-window sparse pair entry is malformed.")
                pair = tuple(sorted((int(raw_pair[0]), int(raw_pair[1]))))
                for target, name in ((gram, "G"), (hessian, "H")):
                    value = float(entry.get(name, float("nan")))
                    previous = target.get(pair)
                    if previous is not None and not math.isclose(
                        previous,
                        value,
                        rel_tol=1e-7,
                        abs_tol=tolerance,
                    ):
                        raise ValueError(
                            "Repeated material-window sparse geometry disagrees."
                        )
                    target[pair] = value
            for entry in acquisition.get("gradient_entries", []):
                index = int(entry.get("active_index", -1))
                value = float(entry.get("g", float("nan")))
                previous = gradient.get(index)
                if previous is not None and not math.isclose(
                    previous,
                    value,
                    rel_tol=1e-7,
                    abs_tol=tolerance,
                ):
                    raise ValueError(
                        "Repeated material-window sparse gradient disagrees."
                    )
                gradient[index] = value
        return gram, hessian, gradient

    def _blocks_for_indices(
        indices: Sequence[int],
        *,
        acquisitions: Sequence[Mapping[str, Any]],
        screen: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        target = tuple(int(value) for value in indices)
        gram_entries, hessian_entries, gradients = _geometry_maps(acquisitions)
        dimension = len(target)
        G_AA = np.zeros((dimension, dimension), dtype=float)
        H_AA = np.zeros((dimension, dimension), dtype=float)
        g_A = np.zeros(dimension, dtype=float)
        for left_local, left in enumerate(target):
            if left not in gradients:
                raise ValueError(
                    "Material-window active gradient acquisition is incomplete."
                )
            g_A[left_local] = gradients[left]
            for right_local in range(left_local, dimension):
                right = target[right_local]
                pair = tuple(sorted((left, right)))
                if pair not in gram_entries or pair not in hessian_entries:
                    raise ValueError(
                        "Material-window old--old pair acquisition is incomplete."
                    )
                G_AA[left_local, right_local] = G_AA[
                    right_local, left_local
                ] = gram_entries[pair]
                H_AA[left_local, right_local] = H_AA[
                    right_local, left_local
                ] = hessian_entries[pair]
        active_position = {value: index for index, value in enumerate(all_active)}
        G_AC_all = np.asarray(screen.get("G_AB", []), dtype=float).reshape(-1)
        H_AC_all = np.asarray(screen.get("H_AB", []), dtype=float).reshape(-1)
        G_AC = np.asarray(
            [G_AC_all[active_position[value]] for value in target], dtype=float
        )
        H_AC = np.asarray(
            [H_AC_all[active_position[value]] for value in target], dtype=float
        )
        return G_AA, H_AA, g_A, G_AC, H_AC

    def _retained_omitted_block_closure(
        *,
        retained_indices: Sequence[int],
        omitted_indices: Sequence[int],
        acquisition: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Test the measured W-by-O blocks before allowing omission.

        Candidate-coupling tails select W.  This independent check asks
        whether the retained old-coordinate block is itself sufficiently
        closed against O.  A failed check forces the already planned O-by-O
        acquisition and a full-coordinate solve.
        """

        retained = tuple(int(value) for value in retained_indices)
        omitted = tuple(int(value) for value in omitted_indices)
        gram_entries, hessian_entries, _gradients = _geometry_maps(
            (acquisition,)
        )

        def _matrix(
            rows: Sequence[int],
            cols: Sequence[int],
            entries: Mapping[tuple[int, int], float],
        ) -> np.ndarray:
            block = np.zeros((len(rows), len(cols)), dtype=float)
            for row_local, row in enumerate(rows):
                for col_local, col in enumerate(cols):
                    pair = tuple(sorted((int(row), int(col))))
                    if pair not in entries:
                        raise ValueError(
                            "Material-window closure acquisition is missing "
                            "a requested old--old pair."
                        )
                    block[row_local, col_local] = float(entries[pair])
            return block

        G_WW = _matrix(retained, retained, gram_entries)
        H_WW = _matrix(retained, retained, hessian_entries)
        G_WO = _matrix(retained, omitted, gram_entries)
        H_WO = _matrix(retained, omitted, hessian_entries)
        epsilon = float(material_window_policy.epsilon)

        def _ratio(cross: np.ndarray, retained_block: np.ndarray) -> float:
            if cross.size == 0:
                return 0.0
            return float(
                np.linalg.norm(cross, ord="fro")
                / max(float(np.linalg.norm(retained_block, ord="fro")), epsilon)
            )

        gram_ratio = _ratio(G_WO, G_WW)
        hessian_ratio = _ratio(H_WO, H_WW)
        finite = bool(
            math.isfinite(gram_ratio)
            and math.isfinite(hessian_ratio)
            and np.all(np.isfinite(G_WW))
            and np.all(np.isfinite(H_WW))
            and np.all(np.isfinite(G_WO))
            and np.all(np.isfinite(H_WO))
        )
        gram_satisfied = bool(
            finite
            and gram_ratio
            <= float(material_window_policy.gram_cross_block_tolerance)
        )
        hessian_satisfied = bool(
            finite
            and hessian_ratio
            <= float(material_window_policy.hessian_cross_block_tolerance)
        )
        reasons: list[str] = []
        if not finite:
            reasons.append("nonfinite_retained_omitted_block_closure")
        if finite and not gram_satisfied:
            reasons.append("gram_cross_block_closure_failed")
        if finite and not hessian_satisfied:
            reasons.append("hessian_cross_block_closure_failed")
        return {
            "schema": "phase3_material_window_block_closure_v1",
            "retained_indices": list(retained),
            "omitted_indices": list(omitted),
            "gram_retained_fro_norm": float(np.linalg.norm(G_WW, ord="fro")),
            "hessian_retained_fro_norm": float(
                np.linalg.norm(H_WW, ord="fro")
            ),
            "gram_retained_omitted_fro_norm": float(
                np.linalg.norm(G_WO, ord="fro")
            ),
            "hessian_retained_omitted_fro_norm": float(
                np.linalg.norm(H_WO, ord="fro")
            ),
            "gram_retained_omitted_ratio": float(gram_ratio),
            "hessian_retained_omitted_ratio": float(hessian_ratio),
            "gram_tolerance": float(
                material_window_policy.gram_cross_block_tolerance
            ),
            "hessian_tolerance": float(
                material_window_policy.hessian_cross_block_tolerance
            ),
            "inputs_finite": bool(finite),
            "gram_satisfied": bool(gram_satisfied),
            "hessian_satisfied": bool(hessian_satisfied),
            "closure_satisfied": bool(gram_satisfied and hessian_satisfied),
            "refresh_reasons": list(reasons),
        }

    evaluated: list[dict[str, Any]] = []
    record_payloads: list[dict[str, Any]] = []
    workspaces: list[_BatchFullGeometryWorkspace] = []
    refresh_count = 0
    infeasible_reasons: dict[str, int] = {}
    for input_index, record in enumerate(copied):
        feature = record["feature"]
        assert isinstance(feature, CandidateFeatures)
        position_id = int(record.get("position_id", feature.position_id))
        screen = _exact_insertion_joint_geometry_payload(
            scaffold_context=screen_context,
            candidate_term=record["candidate_term"],
            position_id=position_id,
            h_compiled=h_compiled,
            pauli_action_cache=pauli_action_cache,
            state_consistency_tolerance=tolerance,
            acquisition_mode=(
                EXACT_INSERTION_GEOMETRY_CANDIDATE_COUPLING_SCREEN_V1
            ),
        )
        identity = _batch_record_identity_key(record)
        # These anchors must come from the same retained W / W+c chart.  Full-
        # active nullities are not comparable when coordinates were omitted.
        prior_nullities = (
            (None, None)
            if prior_retained_support_nullities is None
            else prior_retained_support_nullities.get(identity, (None, None))
        )
        if len(prior_nullities) != 2:
            raise ValueError(
                "Material-window prior support-nullity entry must have two values."
            )
        receipt = build_phase3_material_window(
            active_indices=all_active,
            gram_diagonal=screen.get("G_A_diagonal", []),
            candidate_gram_cross=screen.get("G_AB", []),
            candidate_gram_self=float(screen.get("G_BB", float("nan"))),
            candidate_hessian_cross=screen.get("H_AB", []),
            candidate_hessian_self=float(screen.get("H_BB", float("nan"))),
            policy=material_window_policy,
            prior_active_nullity=prior_nullities[0],
            prior_joint_nullity=prior_nullities[1],
        )
        pair_plan = _phase3_material_window_old_old_pair_plan(
            active_indices=all_active,
            retained_indices=receipt.retained_indices,
        )
        initial_acquisition = _sparse_acquisition(
            pairs=pair_plan["initial_pairs"],
            gradients=receipt.retained_indices,
            stage="retained_and_closure_v1",
        )
        block_closure = _retained_omitted_block_closure(
            retained_indices=receipt.retained_indices,
            omitted_indices=receipt.omitted_indices,
            acquisition=initial_acquisition,
        )
        initial_blocks = _blocks_for_indices(
            receipt.retained_indices,
            acquisitions=(initial_acquisition,),
            screen=screen,
        )
        initial_telemetry = {
            "schema": "phase3_material_window_record_v1",
            "screen": dict(screen),
            "pair_plan": dict(pair_plan),
            "initial_sparse_acquisition": dict(initial_acquisition),
            "retained_omitted_block_closure": dict(block_closure),
            "state_reconstruction_delta_norm": float(
                max(
                    float(screen.get("state_reconstruction_delta_norm", 0.0)),
                    float(
                        initial_acquisition.get(
                            "state_reconstruction_delta_norm", 0.0
                        )
                    ),
                )
            ),
        }
        retained_workspace = _build_material_window_singleton_workspace(
            record,
            cfg=singleton_cfg,
            selected_ops=selected,
            theta=theta_vec,
            psi_state=np.asarray(psi_state, dtype=complex),
            h_compiled=h_compiled,
            active_indices=receipt.retained_indices,
            G_AA=initial_blocks[0],
            H_AA=initial_blocks[1],
            g_A=initial_blocks[2],
            G_AC=initial_blocks[3],
            H_AC=initial_blocks[4],
            G_CC=float(screen["G_BB"]),
            H_CC=float(screen["H_BB"]),
            g_C=float(screen["descent_gradient"]),
            material_window_telemetry=initial_telemetry,
        )
        retained_summary = dict(
            retained_workspace.summary_for_records([record])
        )
        active_rank = int(retained_summary.get("active_metric_support_rank", 0))
        joint_rank = int(retained_summary.get("joint_metric_support_rank", 0))
        finalized_receipt = receipt.finalize_with_support_ranks(
            active_supported_rank=active_rank,
            joint_supported_rank=joint_rank,
            additional_refresh_reasons=tuple(
                str(value)
                for value in block_closure.get("refresh_reasons", [])
            ),
        )
        refresh_performed = bool(
            finalized_receipt.requires_full_geometry_refresh
        )
        refresh_acquisition: dict[str, Any] | None = None
        final_workspace = retained_workspace
        raw_summary = retained_summary
        actual_active_indices = tuple(finalized_receipt.retained_indices)
        if refresh_performed:
            refresh_count += 1
            refresh_acquisition = _sparse_acquisition(
                pairs=pair_plan["omitted_omitted_refresh_pairs"],
                gradients=finalized_receipt.omitted_indices,
                stage="full_geometry_refresh_v1",
            )
            full_blocks = _blocks_for_indices(
                all_active,
                acquisitions=(initial_acquisition, refresh_acquisition),
                screen=screen,
            )
            final_workspace = _build_material_window_singleton_workspace(
                record,
                cfg=singleton_cfg,
                selected_ops=selected,
                theta=theta_vec,
                psi_state=np.asarray(psi_state, dtype=complex),
                h_compiled=h_compiled,
                active_indices=all_active,
                G_AA=full_blocks[0],
                H_AA=full_blocks[1],
                g_A=full_blocks[2],
                G_AC=full_blocks[3],
                H_AC=full_blocks[4],
                G_CC=float(screen["G_BB"]),
                H_CC=float(screen["H_BB"]),
                g_C=float(screen["descent_gradient"]),
                material_window_telemetry={
                    **initial_telemetry,
                    "refresh_sparse_acquisition": dict(refresh_acquisition),
                },
            )
            raw_summary = dict(final_workspace.summary_for_records([record]))
            actual_active_indices = all_active

        initial_pairs = [
            [int(pair[0]), int(pair[1])]
            for pair in pair_plan["initial_pairs"]
        ]
        refresh_pairs = (
            []
            if refresh_acquisition is None
            else [
                [int(pair[0]), int(pair[1])]
                for pair in pair_plan["omitted_omitted_refresh_pairs"]
            ]
        )
        active_gradient_indices = [
            int(value) for value in finalized_receipt.retained_indices
        ] + (
            []
            if refresh_acquisition is None
            else [int(value) for value in finalized_receipt.omitted_indices]
        )
        estimator_acquisition_plan = {
            "schema": "phase3_material_window_estimator_acquisition_plan_v1",
            "state_fingerprint": str(screen.get("state_fingerprint", "")),
            "ordered_scaffold_fingerprint": str(
                screen.get("ordered_scaffold_fingerprint", "")
            ),
            "theta_fingerprint": str(screen.get("theta_fingerprint", "")),
            "hamiltonian_fingerprint": str(
                screen.get("hamiltonian_fingerprint", "")
            ),
            "candidate_coordinate_fingerprint": str(
                screen.get("candidate_coordinate_fingerprint", "")
            ),
            "candidate_pool_index": int(
                record.get("candidate_pool_index", feature.candidate_pool_index)
            ),
            "candidate_label": str(_batch_record_label(record)),
            "candidate_position_id": int(position_id),
            "active_indices": [int(value) for value in all_active],
            "screen_gram_diagonal_indices": [
                int(value) for value in all_active
            ],
            "candidate_cross_gram_active_indices": [
                int(value) for value in all_active
            ],
            "candidate_cross_hessian_active_indices": [
                int(value) for value in all_active
            ],
            "candidate_self_gram_acquired": True,
            "candidate_self_hessian_acquired": True,
            "candidate_gradient_acquired": True,
            "retained_indices": [
                int(value) for value in finalized_receipt.retained_indices
            ],
            "omitted_indices": [
                int(value) for value in finalized_receipt.omitted_indices
            ],
            "retained_retained_pairs": list(
                pair_plan["retained_retained_pairs"]
            ),
            "retained_omitted_closure_pairs": list(
                pair_plan["retained_omitted_pairs"]
            ),
            "omitted_omitted_refresh_pairs": refresh_pairs,
            "old_old_metric_pairs_acquired": [
                *initial_pairs,
                *refresh_pairs,
            ],
            "old_old_hessian_pairs_acquired": [
                *initial_pairs,
                *refresh_pairs,
            ],
            "active_gradient_indices_acquired": active_gradient_indices,
            "retained_omitted_closure_acquired": bool(
                pair_plan["retained_omitted_pairs"]
            ),
            "retained_omitted_block_closure": dict(block_closure),
            "screen_gram_diagonal_indices_reused_in_old_old_pairs": [
                int(value) for value in finalized_receipt.retained_indices
            ],
            "full_geometry_refresh_performed": bool(refresh_performed),
            "full_geometry_refresh_count": int(refresh_performed),
            "screen_gram_diagonal_count": int(len(all_active)),
            "candidate_cross_gram_count": int(len(all_active)),
            "candidate_cross_hessian_count": int(len(all_active)),
            "old_old_metric_pair_count": int(
                len(initial_pairs) + len(refresh_pairs)
            ),
            "old_old_hessian_pair_count": int(
                len(initial_pairs) + len(refresh_pairs)
            ),
            "retained_retained_pair_count": int(
                len(pair_plan["retained_retained_pairs"])
            ),
            "retained_omitted_closure_pair_count": int(
                len(pair_plan["retained_omitted_pairs"])
            ),
            "omitted_omitted_refresh_pair_count": int(len(refresh_pairs)),
            "active_gradient_count": int(len(active_gradient_indices)),
        }
        feasible = bool(raw_summary.get("feasible", False))
        reason = str(raw_summary.get("reason", "unknown"))
        if not feasible:
            infeasible_reasons[reason] = int(
                infeasible_reasons.get(reason, 0) + 1
            )
        response_indices = sorted(
            [
                int(value if value < position_id else value + 1)
                for value in actual_active_indices
            ]
            + [int(position_id)]
        )
        attached_summary = {
            **raw_summary,
            "schema": HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "scope": str(scope),
            "authority": "historical_phase3_benefit_overlay_only",
            "input_record_index": int(input_index),
            "candidate_label": str(_batch_record_label(record)),
            "candidate_pool_index": int(
                record.get("candidate_pool_index", feature.candidate_pool_index)
            ),
            "position_id": int(position_id),
            "historical_phase2_joint_geometry_reuse": dict(
                feature.phase2_joint_geometry_reuse or {}
            ),
            "material_window_receipt": finalized_receipt.to_dict(),
            "prior_nullity_comparison_scope": (
                "same_retained_W_and_W_plus_candidate_v1"
            ),
            "material_window_refresh": {
                "performed": bool(refresh_performed),
                "count": int(refresh_performed),
                "reasons": list(finalized_receipt.refresh_reasons),
                "retained_supported_rank": int(active_rank),
                "retained_joint_supported_rank": int(joint_rank),
                "final_active_indices": [
                    int(value) for value in actual_active_indices
                ],
                "refresh_sparse_acquisition": (
                    None
                    if refresh_acquisition is None
                    else dict(refresh_acquisition)
                ),
            },
            "estimator_acquisition_plan": estimator_acquisition_plan,
        }
        supported_rank_raw = raw_summary.get("joint_metric_support_rank")
        supported_rank = (
            None if supported_rank_raw is None else int(supported_rank_raw)
        )
        updated = dict(record)
        updated["feature"] = _replace_feature(
            feature,
            phase2_joint_geometry_reuse=dict(attached_summary),
            phase3_response_coordinate_indices=response_indices,
            phase3_response_pre_support_count=int(len(response_indices)),
            phase3_response_supported_rank=supported_rank,
        )
        updated["material_window_receipt"] = finalized_receipt.to_dict()
        updated["estimator_acquisition_plan"] = estimator_acquisition_plan
        evaluated.append(updated)
        record_payloads.append(dict(attached_summary))
        workspaces.append(final_workspace)

    if workspaces:
        representative_workspace = workspaces[0]
        representative_telemetry = dict(
            representative_workspace.build_telemetry()
        )
    else:
        empty_cfg = replace(
            singleton_cfg,
            batch_active_context_indices=(),
        )
        representative_workspace = _build_batch_full_geometry_workspace(
            (),
            cfg=empty_cfg,
            selected_ops=selected,
            theta=theta_vec,
            psi_ref=np.asarray(psi_ref, dtype=complex),
            psi_state=np.asarray(psi_state, dtype=complex),
            h_compiled=h_compiled,
            pauli_action_cache=pauli_action_cache,
        )
        representative_telemetry = dict(
            representative_workspace.build_telemetry()
        )
    telemetry = {
        "schema": "historical_singleton_material_window_population_v1",
        "scope": str(scope),
        "authority": "historical_phase3_benefit_overlay_only",
        "input_record_count": int(len(copied)),
        "output_record_count": int(len(evaluated)),
        "membership_preserved": True,
        "order_preserved": True,
        "phase2_rescoring_applied": False,
        "batching_applied": False,
        "effective_batch_size_cap": 1,
        "joint_linear_solve_policy_requested": (
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
        ),
        "feasible_record_count": int(
            sum(bool(row.get("feasible", False)) for row in record_payloads)
        ),
        "infeasible_record_count": int(
            sum(not bool(row.get("feasible", False)) for row in record_payloads)
        ),
        "infeasible_reason_counts": dict(sorted(infeasible_reasons.items())),
        "full_geometry_refresh_count": int(refresh_count),
        "records": record_payloads,
        "estimator_acquisition_plans": [
            dict(row["estimator_acquisition_plan"])
            for row in record_payloads
        ],
        "representative_geometry_workspace": representative_telemetry,
        "per_record_geometry_workspaces": [
            dict(workspace.build_telemetry()) for workspace in workspaces
        ],
    }
    return Phase2JointResponseEvaluation(
        records=tuple(evaluated),
        telemetry=telemetry,
        workspace=representative_workspace,
    )


def _historical_phase3_stored_component(
    components: Mapping[str, Any],
    *names: str,
) -> float:
    for name in names:
        if name not in components:
            continue
        try:
            value = float(components[name])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    raise ValueError(
        "Historical Phase-III coordinate-model rescore is missing finite "
        f"component(s): {', '.join(names)}."
    )


def _compute_deferred_phase3_gram_novelty(
    feature: CandidateFeatures,
    *,
    cfg: FullScoreConfig,
) -> dict[str, Any]:
    """Compute N3 only when the geometry-expansion fallback needs it.

    The deferred payload contains only Gram-derived arrays already constructed
    for the candidate's supported coordinate model. This helper performs no
    Hamiltonian or metric estimator request and therefore carries zero query
    charge.
    """

    coordinate_summary = dict(feature.phase2_joint_geometry_reuse or {})
    source_geometry_raw = coordinate_summary.get(
        "historical_phase2_joint_geometry_reuse",
        coordinate_summary,
    )
    if not isinstance(source_geometry_raw, Mapping):
        raise ValueError(
            "Fallback-only Phase-III novelty is missing preserved Gram geometry."
        )
    deferred_raw = source_geometry_raw.get("phase3_deferred_gram_novelty")
    if not isinstance(deferred_raw, Mapping):
        raise ValueError(
            "Fallback-only Phase-III novelty is missing its deferred payload."
        )
    deferred = dict(deferred_raw)
    if str(deferred.get("schema", "")) != "phase3_deferred_gram_novelty_v1":
        raise ValueError(
            "Fallback-only Phase-III novelty has the wrong deferred schema."
        )
    Q_window = np.asarray(deferred.get("Q_window", []), dtype=float)
    q_reduced = np.asarray(deferred.get("q_reduced", []), dtype=float).reshape(-1)
    if q_reduced.size == 0 and Q_window.size == 0:
        Q_window = np.zeros((0, 0), dtype=float)
    F_red = float(deferred.get("F_red", float("nan")))
    if (
        Q_window.ndim != 2
        or Q_window.shape != (int(q_reduced.size), int(q_reduced.size))
        or not np.all(np.isfinite(Q_window))
        or not np.all(np.isfinite(q_reduced))
        or not math.isfinite(F_red)
        or F_red <= 0.0
    ):
        raise ValueError(
            "Fallback-only Phase-III novelty deferred Gram payload is malformed."
        )
    metric_collapse = bool(deferred.get("metric_collapse", False))
    ridge_used: float | None = None
    classical_solve_count = 0
    if metric_collapse:
        novelty = 0.0
        source = "deferred_reduced_metric_collapse_v1"
    elif q_reduced.size == 0:
        novelty = 1.0
        source = "deferred_empty_window_v1"
    else:
        qsol, ridge_used, _Qreg = _regularized_solve(
            Q_window,
            q_reduced,
            base_ridge=float(_deferred_gram_fallback_ridge(cfg)),
            growth_factor=float(max(cfg.ridge_growth_factor, 2.0)),
            max_steps=int(max(1, cfg.ridge_max_steps)),
            require_pd=True,
        )
        novelty_raw = 1.0 - float(q_reduced.T @ qsol) / float(F_red)
        novelty = float(min(1.0, max(0.0, novelty_raw)))
        source = "deferred_collective_span_v1"
        classical_solve_count = 1
    return {
        "schema": "phase3_lazy_gram_novelty_result_v1",
        "value": float(novelty),
        "status": GRAM_NOVELTY_STATUS_COMPUTED_FOR_GEOMETRY_EXPANSION,
        "source": str(source),
        "ridge_used": None if ridge_used is None else float(ridge_used),
        "query_charge": 0,
        "classical_solve_count": int(classical_solve_count),
        "measurement_role": "supported_gram_reuse_for_geometry_expansion",
    }


def rescore_historical_phase3_records_with_coordinate_models(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig | None = None,
    expected_coordinate_solve_policy: str | None = None,
    sr_escape_mode: SREscapeMode | str = SR_ESCAPE_DISABLED,
    sr_escape_ordinary_record_ids: Sequence[str] | None = None,
    sr_escape_reachable_record_ids: Sequence[str] | None = None,
    sr_escape_contradicted_ordinary_record_ids: Sequence[str] | None = None,
    sr_escape_state_fingerprint: str | None = None,
    sr_escape_trust_radius: float | None = None,
    sr_escape_comparison_epoch: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Substitute only the historical Phase-III benefit with joint gain.

    Input membership and order are invariant. Saved historical novelty remains
    passive provenance and never multiplies the ordinary score. The deferred
    Gram residual is computed only when the explicitly authorized
    all-models-infeasible fallback fires. Hardware-cost denominator, tie terms,
    and gate fields remain authoritative; no Phase-II score, shortlist,
    batching, or Joint-Response funnel policy is evaluated here.
    """

    eps = float(
        max(
            1e-15,
            getattr(cfg, "cheap_score_eps", 1e-12) if cfg is not None else 1e-12,
        )
    )
    phase3_novelty_multiplier_policy = (
        ORDINARY_NOVELTY_SCORING_RETIRED_V1
    )
    deferred_fallback_enabled = bool(
        cfg is not None and _deferred_gram_fallback_enabled(cfg)
    )
    phase3_gram_novelty_policy = (
        GRAM_NOVELTY_POLICY_FALLBACK_ONLY_V1
        if deferred_fallback_enabled
        else "off"
    )
    hardware_cost_normalization_mode = (
        HARDWARE_COST_NORMALIZATION_FAMILY_ROBUST_V1
        if cfg is None
        else _hardware_cost_normalization_mode(cfg)
    )
    phase3_signed_factor_consumer_semantic_version = None
    if cfg is not None:
        phase3_signed_factor_consumer_semantic_version = (
            require_phase3_signed_factor_consumer_semantic_version(cfg)
        )
    multiplicative_signed_hardware_cost_active = bool(
        hardware_cost_normalization_mode
        in HARDWARE_COST_MULTIPLICATIVE_SIGNED_FACTOR_POLICIES
    )
    phase3_score_formula = PHASE3_CANONICAL_SCORE_FORMULA
    if multiplicative_signed_hardware_cost_active:
        phase3_score_formula = (
            "DeltaE_TR * hardware_cost_score_factor / (1 + K3)"
        )
    escape_mode = SREscapeMode(sr_escape_mode)
    expected_candidate_gain_policy = str(
        getattr(
            cfg,
            "phase3_candidate_gain_policy",
            PHASE3_CANDIDATE_GAIN_JOINT_TOTAL_V1,
        )
        if cfg is not None
        else PHASE3_CANDIDATE_GAIN_JOINT_TOTAL_V1
    )
    if expected_candidate_gain_policy not in PHASE3_CANDIDATE_GAIN_POLICIES:
        raise ValueError(
            "Historical Phase-III coordinate-model rescore received an "
            "unknown candidate-gain policy."
        )
    combined_mode_requested = bool(
        escape_mode is SREscapeMode.SADDLE_PLUS_MODELED_MINIMUM
    )
    modeled_minimum_eligibility = assess_exposed_family_psd(None)
    modeled_minimum_core_payload: dict[str, Any] = {
        "schema": _SR_MODELED_MINIMUM_CORE_TELEMETRY_SCHEMA,
        "version": 1,
        "combined_mode_requested": combined_mode_requested,
        "mathematical_eligibility": modeled_minimum_eligibility.to_dict(),
        "state_token_digest": None,
        "pure_core_available": True,
        "execution_implemented": False,
        "actionable": False,
        "remaining_provider_runtime_checkpoint_blockers": list(
            _SR_MODELED_MINIMUM_EXECUTION_BLOCKERS
        ),
    }
    if escape_mode is SREscapeMode.DISABLED:
        allowed_policy = str(
            expected_coordinate_solve_policy
            or JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
        )
        if allowed_policy not in {
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
        }:
            raise ValueError(
                "Historical Phase-III coordinate-model rescore received "
                f"an unsupported ordinary solver policy {allowed_policy!r}."
            )
    else:
        allowed_policy = JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2
        if expected_coordinate_solve_policy not in {None, allowed_policy}:
            raise ValueError(
                "SR escape requires the registered global-trust solver; "
                f"got {expected_coordinate_solve_policy!r}."
            )
    geometry_expansion_reason = (
        "all_supported_coordinate_energy_models_infeasible"
        if allowed_policy
        == JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
        else "all_whitened_energy_models_infeasible"
    )
    coordinate_summaries: list[dict[str, Any]] = []
    signed_factor_population_hashes: set[str] = set()
    signed_factor_population_features: list[CandidateFeatures] = []
    for raw_record in records:
        feature = raw_record.get("feature")
        if not isinstance(feature, CandidateFeatures):
            raise ValueError(
                "Historical Phase-III coordinate-model rescore requires "
                "CandidateFeatures on every retained record."
            )
        if multiplicative_signed_hardware_cost_active:
            if cfg is None:  # pragma: no cover - implied by the active-policy gate
                raise RuntimeError(
                    "Signed-factor Phase-III coordinate rescoring requires "
                    "FullScoreConfig."
                )
            signed_factor_population_hashes.add(
                _validated_multiplicative_signed_factor_feature(
                    feature,
                    cfg,
                    configured_policy=hardware_cost_normalization_mode,
                )
            )
            signed_factor_population_features.append(feature)
        summary = dict(feature.phase2_joint_geometry_reuse or {})
        if str(summary.get("schema", "")) != (
            HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA
        ):
            raise ValueError(
                "Historical Phase-III coordinate-model summary is missing or "
                "has the wrong schema."
            )
        effective_policy = str(
            summary.get("joint_linear_solve_policy_effective", "")
        )
        if effective_policy != allowed_policy:
            raise ValueError(
                "Historical Phase-III coordinate-model summary used the wrong "
                f"solver for SR escape mode {escape_mode.value!r}: "
                f"expected {allowed_policy!r}, got {effective_policy!r}."
            )
        observed_candidate_gain_policy = str(
            summary.get(
                "phase3_candidate_gain_policy",
                PHASE3_CANDIDATE_GAIN_JOINT_TOTAL_V1,
            )
        )
        if observed_candidate_gain_policy != expected_candidate_gain_policy:
            raise ValueError(
                "Historical Phase-III coordinate-model summary used the "
                "wrong candidate-gain policy: expected "
                f"{expected_candidate_gain_policy!r}, got "
                f"{observed_candidate_gain_policy!r}."
            )
        if (
            expected_candidate_gain_policy
            == PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
            and not isinstance(
                summary.get("phase3_candidate_gain_receipt"), Mapping
            )
        ):
            raise ValueError(
                "Incremental Phase-III candidate gain requires its "
                "candidate-gain receipt."
            )
        coordinate_summaries.append(summary)
    if multiplicative_signed_hardware_cost_active:
        if cfg is None:  # pragma: no cover - implied by the active-policy gate
            raise RuntimeError(
                "Signed-factor Phase-III coordinate rescoring requires "
                "FullScoreConfig."
            )
        expected_population_normalization = hardware_cost_family_normalization(
            signed_factor_population_features,
            cfg,
        )
        expected_population_hash = expected_population_normalization.get(
            "population_hash"
        )
        if (
            expected_population_normalization.get("policy")
            != hardware_cost_normalization_mode
            or not isinstance(expected_population_hash, str)
            or signed_factor_population_hashes != {expected_population_hash}
        ):
            raise ValueError(
                "Signed-factor Phase-III records do not close to exactly one "
                "configured normalized hardware-cost population."
            )
    geometry_expansion_active = bool(
        deferred_fallback_enabled
        and
        escape_mode is SREscapeMode.DISABLED
        and coordinate_summaries
        and not any(
            bool(summary.get("feasible", False))
            for summary in coordinate_summaries
        )
    )
    rescored: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, Any]] = []
    infeasible_reasons: dict[str, int] = {}
    for input_index, raw_record in enumerate(records):
        record = dict(raw_record)
        feature = record.get("feature")
        if not isinstance(feature, CandidateFeatures):
            raise ValueError(
                "Historical Phase-III coordinate-model rescore requires "
                "CandidateFeatures on every retained record."
            )
        summary = dict(coordinate_summaries[input_index])
        components = dict(feature.phase_score_components or {})
        historical_gain = _historical_phase3_stored_component(
            components,
            "phase3_historical_scalar_DeltaE_TR",
            "historical_scalar_DeltaE_TR",
            "DeltaE_TR",
            "phase3_delta_e_tr",
        )
        lazy_novelty_result: dict[str, Any] | None = None
        if geometry_expansion_active:
            if cfg is None:  # pragma: no cover - implied by authorization
                raise ValueError(
                    "Deferred-Gram fallback requires FullScoreConfig."
                )
            lazy_novelty_result = _compute_deferred_phase3_gram_novelty(
                feature,
                cfg=cfg,
            )
        passive_n3: float | None = None
        for passive_key in ("N3", "phase3_N3"):
            try:
                passive_candidate = float(components[passive_key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(passive_candidate):
                passive_n3 = float(passive_candidate)
                break
        applied_n3 = (
            None
            if lazy_novelty_result is None
            else float(lazy_novelty_result["value"])
        )
        k3 = _historical_phase3_stored_component(
            components,
            "K3",
            "phase3_K3",
        )
        denominator = _historical_phase3_stored_component(
            components,
            "denominator_1_plus_K3",
            "phase3_denominator_1_plus_K3",
        )
        if k3 < 0.0 or denominator <= 0.0:
            raise ValueError(
                "Historical Phase-III K3 and denominator must be nonnegative "
                "with a positive denominator."
            )
        if not math.isclose(
            denominator,
            1.0 + k3,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Historical Phase-III denominator does not match 1 + K3."
            )
        if feature.phase3_burden_total is not None and not math.isclose(
            float(feature.phase3_burden_total),
            denominator,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Historical Phase-III feature burden disagrees with its "
                "saved denominator."
            )
        hardware_cost_score_factor = 1.0
        if multiplicative_signed_hardware_cost_active:
            if cfg is None:  # pragma: no cover - implied by the active-policy gate
                raise RuntimeError(
                    "Signed-factor Phase-III coordinate rescoring requires "
                    "FullScoreConfig."
                )
            feature_cost_policy = str(
                getattr(feature, "hardware_cost_policy", "unresolved")
                or "unresolved"
            )
            if feature_cost_policy != hardware_cost_normalization_mode:
                raise ValueError(
                    "Signed-factor Phase-III coordinate-model rescore requires "
                    "a population-normalized hardware-cost feature whose policy "
                    "matches the configured policy."
                )
            hardware_cost_payload = _hardware_cost_denominator_payload(
                feature,
                cfg,
            )
            payload_denominator = float(
                hardware_cost_payload["hardware_cost_denominator"]
            )
            if not math.isclose(
                payload_denominator,
                denominator,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Signed-factor Phase-III coordinate-model cost payload "
                    "disagrees with its saved denominator."
                )
            hardware_cost_score_factor = float(
                hardware_cost_payload["hardware_cost_score_factor"]
            )

        feasible = bool(summary.get("feasible", False))
        reason = str(summary.get("reason", "unknown"))
        try:
            joint_gain_raw = float(summary.get("joint_gain", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Historical singleton joint_gain must be finite."
            ) from exc
        if not math.isfinite(joint_gain_raw) or joint_gain_raw < 0.0:
            raise ValueError(
                "Historical singleton joint_gain must be finite and "
                "nonnegative."
            )
        joint_gain = float(joint_gain_raw if feasible else 0.0)
        candidate_gain_receipt_raw = summary.get(
            "phase3_candidate_gain_receipt"
        )
        candidate_gain_receipt = (
            dict(candidate_gain_receipt_raw)
            if isinstance(candidate_gain_receipt_raw, Mapping)
            else {
                "schema": "phase3_candidate_gain_receipt_v1",
                "policy": PHASE3_CANDIDATE_GAIN_JOINT_TOTAL_V1,
                "joint_gain_semantics": "full_joint_trust_gain_legacy_v1",
                "full_joint_trust_gain": float(joint_gain_raw),
                "active_only_trust_gain": 0.0,
                "incremental_candidate_gain": float(joint_gain_raw),
                "selected_gain": float(joint_gain_raw),
                "classical_quantum_query_charge": 0,
            }
        )
        if str(candidate_gain_receipt.get("policy", "")) != (
            expected_candidate_gain_policy
        ):
            raise ValueError(
                "Phase-III candidate-gain receipt policy drifted from the "
                "configured rescore policy."
            )
        try:
            full_joint_gain = float(
                candidate_gain_receipt.get(
                    "full_joint_trust_gain", joint_gain_raw
                )
            )
            active_only_gain = float(
                candidate_gain_receipt.get(
                    "active_only_trust_gain", 0.0
                )
            )
            incremental_candidate_gain = float(
                candidate_gain_receipt.get(
                    "incremental_candidate_gain", joint_gain_raw
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Phase-III candidate-gain receipt contains invalid gains."
            ) from exc
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                full_joint_gain,
                active_only_gain,
                incremental_candidate_gain,
            )
        ) or not math.isclose(
            incremental_candidate_gain,
            joint_gain_raw,
            rel_tol=1.0e-10,
            abs_tol=max(
                1.0e-12,
                float(
                    getattr(cfg, "batch_energy_regularization", 0.0)
                )
                if cfg is not None
                else 0.0,
            ),
        ):
            raise ValueError(
                "Phase-III candidate-gain receipt does not close against "
                "the selection-facing gain."
            )
        measured_n3 = (
            float(applied_n3)
            if applied_n3 is not None
            else passive_n3
        )
        energy_primary_score = float(
            joint_gain
            * float(hardware_cost_score_factor)
            / max(denominator, eps)
        )
        geometry_expansion_score = (
            None
            if applied_n3 is None
            else float(
                applied_n3
                * float(hardware_cost_score_factor)
                / max(denominator, eps)
            )
        )
        primary_score = float(
            float(geometry_expansion_score)
            if geometry_expansion_active
            else energy_primary_score
        )
        auxiliary_mode = normalize_phase3_auxiliary_score_mode(
            feature.phase3_auxiliary_score_mode
        )
        tie_score = float(feature.phase3_tie_break_score)
        full_score = float(primary_score)
        if (
            not geometry_expansion_active
            and auxiliary_mode == PHASE3_AUXILIARY_SCORE_ABLATION_ADDITIVE
        ):
            full_score = float(full_score + tie_score)
        if not feasible and not geometry_expansion_active:
            full_score = float("-inf")
            infeasible_reasons[reason] = int(
                infeasible_reasons.get(reason, 0) + 1
            )
        elif not feasible:
            infeasible_reasons[reason] = int(
                infeasible_reasons.get(reason, 0) + 1
            )

        phase_score_components = {
            **components,
            "phase3_historical_scalar_DeltaE_TR": float(historical_gain),
            "historical_scalar_DeltaE_TR": float(historical_gain),
            "phase3_coordinate_model_full_joint_gain": float(
                full_joint_gain
            ),
            "phase3_coordinate_model_active_only_gain": float(
                active_only_gain
            ),
            "phase3_coordinate_model_incremental_candidate_gain": float(
                joint_gain
            ),
            "phase3_candidate_gain_policy": str(
                expected_candidate_gain_policy
            ),
            "phase3_coordinate_model_feasible": float(1.0 if feasible else 0.0),
            "phase3_geometry_expansion_active": float(
                1.0 if geometry_expansion_active else 0.0
            ),
            "phase3_geometry_expansion_score": (
                None
                if geometry_expansion_score is None
                else float(geometry_expansion_score)
            ),
            "phase3_delta_e_tr": float(joint_gain),
            "DeltaE_TR": float(joint_gain),
            "phase3_measured_novelty": (
                None if measured_n3 is None else float(measured_n3)
            ),
            "phase3_novelty_multiplier": (
                None if applied_n3 is None else float(applied_n3)
            ),
            "phase3_novelty_multiplier_policy": str(
                phase3_novelty_multiplier_policy
            ),
            "phase3_gram_novelty_policy": str(
                phase3_gram_novelty_policy
            ),
            "phase3_novelty_status": (
                str(lazy_novelty_result["status"])
                if lazy_novelty_result is not None
                else GRAM_NOVELTY_STATUS_NOT_COMPUTED_FOR_ORDINARY_SCORING
            ),
            "phase3_novelty_query_charge": 0,
            "phase3_novelty_classical_solve_count": int(
                0
                if lazy_novelty_result is None
                else lazy_novelty_result["classical_solve_count"]
            ),
            "phase3_novelty_applied": bool(geometry_expansion_active),
            "phase3_N3": None if applied_n3 is None else float(applied_n3),
            "N3": None if applied_n3 is None else float(applied_n3),
            "phase3_primary_score": float(primary_score),
            "hardware_cost_score_factor": float(
                hardware_cost_score_factor
            ),
            "phase3_reduced_score": float(full_score),
            "selector_score": float(full_score),
        }
        updated_feature = _replace_feature(
            feature,
            full_v2_score=float(full_score),
            phase3_reduced_trust_gain=float(joint_gain),
            phase3_full_joint_trust_gain=float(full_joint_gain),
            phase3_active_only_trust_gain=float(active_only_gain),
            phase3_incremental_candidate_gain=float(joint_gain),
            phase3_candidate_gain_policy=str(
                expected_candidate_gain_policy
            ),
            phase3_candidate_gain_receipt=dict(candidate_gain_receipt),
            phase3_reduced_novelty=feature.phase3_reduced_novelty,
            phase3_primary_score=float(primary_score),
            phase3_canonical_score_formula=str(phase3_score_formula),
            selector_score=float(full_score),
            phase_score_components=phase_score_components,
        )
        updated_record = {
            **record,
            "feature": updated_feature,
            "full_v2_score": float(full_score),
            "selector_score": float(full_score),
            "phase3_primary_score": float(primary_score),
            "hardware_cost_score_factor": float(
                hardware_cost_score_factor
            ),
            "phase3_reduced_trust_gain": float(joint_gain),
            "phase3_full_joint_trust_gain": float(full_joint_gain),
            "phase3_active_only_trust_gain": float(active_only_gain),
            "phase3_incremental_candidate_gain": float(joint_gain),
            "phase3_candidate_gain_policy": str(
                expected_candidate_gain_policy
            ),
            "phase3_candidate_gain_receipt": dict(
                candidate_gain_receipt
            ),
            "phase3_coordinate_model_feasible": bool(feasible),
            "phase3_coordinate_model_reason": str(reason),
        }
        if geometry_expansion_active:
            updated_record.update(
                {
                    "route_a_geometry_expansion_mode": (
                        "collective_span_novelty_over_cost_v1"
                    ),
                    "route_a_geometry_expansion_reason": (
                        geometry_expansion_reason
                    ),
                    "route_a_geometry_expansion_score": float(
                        geometry_expansion_score
                    ),
                    "route_a_geometry_expansion_novelty_status": str(
                        lazy_novelty_result["status"]
                        if lazy_novelty_result is not None
                        else GRAM_NOVELTY_STATUS_NOT_COMPUTED_FOR_ORDINARY_SCORING
                    ),
                    "route_a_geometry_expansion_query_charge": 0,
                    "route_a_coordinate_model_infeasible_reason": str(reason),
                }
            )
        rescored.append(updated_record)
        telemetry_rows.append(
            {
                "input_record_index": int(input_index),
                "candidate_label": str(_batch_record_label(record)),
                "candidate_pool_index": int(
                    record.get("candidate_pool_index", feature.candidate_pool_index)
                ),
                "position_id": int(record.get("position_id", feature.position_id)),
                "feasible": bool(feasible),
                "reason": str(reason),
                "historical_scalar_gain": float(historical_gain),
                "coordinate_model_full_joint_gain": float(
                    full_joint_gain
                ),
                "coordinate_model_active_only_gain": float(
                    active_only_gain
                ),
                "coordinate_model_incremental_candidate_gain": float(
                    joint_gain
                ),
                "phase3_candidate_gain_policy": str(
                    expected_candidate_gain_policy
                ),
                "phase3_candidate_gain_receipt": dict(
                    candidate_gain_receipt
                ),
                "geometry_expansion_score": (
                    None
                    if geometry_expansion_score is None
                    else float(geometry_expansion_score)
                ),
                "N3": None if applied_n3 is None else float(applied_n3),
                "measured_N3": (
                    None if measured_n3 is None else float(measured_n3)
                ),
                "phase3_novelty_multiplier": (
                    None if applied_n3 is None else float(applied_n3)
                ),
                "phase3_novelty_multiplier_policy": str(
                    phase3_novelty_multiplier_policy
                ),
                "phase3_gram_novelty_policy": str(
                    phase3_gram_novelty_policy
                ),
                "phase3_novelty_status": (
                    str(lazy_novelty_result["status"])
                    if lazy_novelty_result is not None
                    else GRAM_NOVELTY_STATUS_NOT_COMPUTED_FOR_ORDINARY_SCORING
                ),
                "phase3_novelty_query_charge": 0,
                "phase3_novelty_classical_solve_count": int(
                    0
                    if lazy_novelty_result is None
                    else lazy_novelty_result["classical_solve_count"]
                ),
                "phase3_novelty_applied": bool(geometry_expansion_active),
                "K3": float(k3),
                "denominator_1_plus_K3": float(denominator),
                "hardware_cost_normalization_mode": str(
                    hardware_cost_normalization_mode
                ),
                "hardware_cost_score_factor": float(
                    hardware_cost_score_factor
                ),
                "phase3_primary_score": float(primary_score),
                "phase3_tie_break_score": float(tie_score),
                "phase3_auxiliary_score_mode": str(auxiliary_mode),
                "full_v2_score": float(full_score),
            }
        )

    sr_decision_payload: dict[str, Any] = {
        "active": bool(escape_mode is not SREscapeMode.DISABLED),
        "mode": str(escape_mode.value),
        "decision_kind": "disabled",
        "reason": "sr_escape_disabled",
        "record_id": None,
        "certificate_record_id": None,
        "actionable": False,
        "consumes_singleton": False,
        "stage_b_eligible": False,
        "acquisition": 0.0,
        "marginal_gain_lower_bound": 0.0,
        "admission_eligible_record_ids": [],
        "modeled_minimum_core": dict(modeled_minimum_core_payload),
    }
    if escape_mode is not SREscapeMode.DISABLED:
        record_ids = tuple(
            sr_escape_record_id(
                _batch_record_label(record),
                int(record.get("candidate_pool_index", -1)),
                int(record.get("position_id", -1)),
            )
            for record in records
        )
        if sr_escape_reachable_record_ids is None:
            raise ValueError(
                "Active SR escape rescoring requires an explicit complete "
                "Phase-III-reachable record-id population."
            )
        reachable_record_ids = tuple(
            str(record_id) for record_id in sr_escape_reachable_record_ids
        )
        if reachable_record_ids != record_ids:
            raise ValueError(
                "Active SR escape rescore input must exactly reproduce the "
                "declared complete reachable population and deterministic order."
            )
        ordinary_record_ids = frozenset(
            str(record_id)
            for record_id in (sr_escape_ordinary_record_ids or ())
        )
        unexpected_ordinary_ids = sorted(
            ordinary_record_ids - set(reachable_record_ids)
        )
        if unexpected_ordinary_ids:
            raise ValueError(
                "SR ordinary population contains records outside the escape "
                f"reachable population: {unexpected_ordinary_ids}"
            )
        contradicted_ordinary_record_ids = frozenset(
            str(record_id)
            for record_id in (
                sr_escape_contradicted_ordinary_record_ids or ()
            )
        )
        unexpected_contradicted_ids = sorted(
            contradicted_ordinary_record_ids - ordinary_record_ids
        )
        if unexpected_contradicted_ids:
            raise ValueError(
                "SR contradicted ordinary population contains records outside "
                f"the ordinary population: {unexpected_contradicted_ids}"
            )
        live_ordinary_record_ids = frozenset(
            ordinary_record_ids - contradicted_ordinary_record_ids
        )
        ordinary_summaries: list[Mapping[str, Any] | None] = []
        ordinary_scores: list[float] = []
        for summary, row in zip(coordinate_summaries, telemetry_rows):
            ordinary_raw = summary.get("sr_escape_ordinary_summary")
            ordinary_summary = (
                ordinary_raw if isinstance(ordinary_raw, Mapping) else None
            )
            ordinary_summaries.append(ordinary_summary)
            ordinary_gain = float(
                ordinary_summary.get("joint_gain", 0.0)
                if ordinary_summary is not None
                else 0.0
            )
            ordinary_score = float(
                ordinary_gain
                * float(1.0 if row["N3"] is None else row["N3"])
                / max(float(row["denominator_1_plus_K3"]), eps)
            )
            if (
                str(row["phase3_auxiliary_score_mode"])
                == PHASE3_AUXILIARY_SCORE_ABLATION_ADDITIVE
            ):
                ordinary_score = float(
                    ordinary_score + float(row["phase3_tie_break_score"])
                )
            ordinary_scores.append(ordinary_score)
        ordinary_indices = [
            index
            for index, ordinary_summary in enumerate(ordinary_summaries)
            if record_ids[index] in live_ordinary_record_ids
            and ordinary_summary is not None
            and bool(ordinary_summary.get("feasible", False))
            and float(
                ordinary_summary.get("joint_gain_lower_bound", 0.0)
            )
            > eps
        ]
        ordinary_certificate: OrdinaryCertificate | None = None
        if ordinary_indices:
            best_ordinary_index = min(
                ordinary_indices,
                key=lambda index: (
                    -float(ordinary_scores[index]),
                    index,
                ),
            )
            best_ordinary_summary = ordinary_summaries[best_ordinary_index]
            if best_ordinary_summary is None:
                raise RuntimeError(
                    "SR ordinary selection lost its preserved v1 summary."
                )
            ordinary_certificate = OrdinaryCertificate(
                record_id=record_ids[best_ordinary_index],
                gain_lower_bound=float(
                    max(
                        eps,
                        best_ordinary_summary.get(
                            "joint_gain_lower_bound", 0.0
                        ),
                    )
                ),
            )

        audit: ReachablePopulationAudit | None = None
        certificate_kind_counts: dict[str, int] = {}
        unresolved_certificate_reason_counts: dict[str, int] = {}
        reachable_certificate_table: list[dict[str, Any]] = []
        state_stationarity_certificate: StateStationarityCertificate | None = None
        state_stationarity_blocker: str | None = "ordinary_precedence"
        if ordinary_certificate is None:
            state_stationarity_blocker = None
            certificates: list[
                NonstationaryCertificate
                | SaddleCertificate
                | PsdCertificate
                | QuotientRedundantCertificate
                | UnresolvedCertificate
            ] = []
            for record_id, summary, row in zip(
                record_ids, coordinate_summaries, telemetry_rows
            ):
                support_status = str(
                    summary.get("raw_metric_support_status", "unresolved")
                )
                null_compatibility_certified = bool(
                    summary.get(
                        "raw_metric_null_compatibility_certified", False
                    )
                )
                if (
                    support_status != "resolved"
                    or not null_compatibility_certified
                ):
                    unresolved_reason = (
                        str(
                            summary.get(
                                "raw_metric_support_reason",
                                summary.get("reason", "raw_support_unresolved"),
                            )
                        )
                        if support_status != "resolved"
                        else str(
                            summary.get(
                                "raw_metric_null_compatibility_reason",
                                summary.get(
                                    "reason",
                                    "raw_metric_null_compatibility_unresolved",
                                ),
                            )
                        )
                    )
                    certificates.append(
                        UnresolvedCertificate(
                            record_id=record_id,
                            reason=unresolved_reason,
                        )
                    )
                    continue
                stationarity_status = str(
                    summary.get("supported_stationarity_status", "unresolved")
                )
                if stationarity_status == "certified_nonstationary":
                    active_restriction_raw = summary.get(
                        "active_restriction_solve"
                    )
                    active_restriction = (
                        active_restriction_raw
                        if isinstance(active_restriction_raw, Mapping)
                        else {}
                    )
                    def finite_nonnegative(raw_value: Any) -> float | None:
                        try:
                            value = float(raw_value)
                        except (TypeError, ValueError):
                            return None
                        if not math.isfinite(value) or value < 0.0:
                            return None
                        return value

                    stationarity_margin = finite_nonnegative(
                        summary.get("stationarity_margin")
                    )
                    active_lower = finite_nonnegative(
                        summary.get(
                            "active_restricted_trust_gain_lower_bound"
                        )
                    )
                    active_upper = finite_nonnegative(
                        summary.get(
                            "active_restricted_trust_gain_upper_bound"
                        )
                    )
                    batch_zero_residual = finite_nonnegative(
                        active_restriction.get(
                            "active_restriction_batch_zero_residual"
                        )
                    )
                    batch_zero_tolerance = finite_nonnegative(
                        active_restriction.get(
                            "active_restriction_batch_zero_tolerance"
                        )
                    )
                    numeric_certificate_valid = bool(
                        stationarity_margin is not None
                        and stationarity_margin > 0.0
                        and active_lower is not None
                        and active_lower > 0.0
                        and active_upper is not None
                        and active_lower <= active_upper
                        and batch_zero_residual is not None
                        and batch_zero_tolerance is not None
                        and batch_zero_residual <= batch_zero_tolerance
                    )
                    active_restriction_valid = bool(
                        numeric_certificate_valid
                        and active_restriction
                        and active_restriction.get("valid", False)
                        and active_restriction.get("feasible", False)
                        and active_restriction.get(
                            "trust_global_optimality_certified", False
                        )
                        and active_restriction.get(
                            "active_restriction_uses_full_support_decision",
                            False,
                        )
                        and not active_restriction.get(
                            "active_restriction_independent_metric_factorization",
                            True,
                        )
                        and str(
                            active_restriction.get(
                                "active_restriction_constraint", ""
                            )
                        )
                        == (
                            "physical_active_coordinate_image_modulo_"
                            "certified_raw_metric_null"
                        )
                    )
                    if active_restriction_valid:
                        assert stationarity_margin is not None
                        assert active_lower is not None
                        assert active_upper is not None
                        certificates.append(
                            NonstationaryCertificate(
                                record_id=record_id,
                                stationarity_margin=stationarity_margin,
                                active_trust_gain_lower_bound=active_lower,
                                active_trust_gain_upper_bound=active_upper,
                            )
                        )
                    else:
                        certificates.append(
                            UnresolvedCertificate(
                                record_id=record_id,
                                reason=(
                                    "certified_nonstationary_numeric_"
                                    "certificate_invalid"
                                    if not numeric_certificate_valid
                                    else "certified_nonstationary_but_active_"
                                    "restriction_unresolved"
                                ),
                            )
                        )
                    continue
                try:
                    stationary_margin = float(
                        summary.get("stationarity_margin", float("nan"))
                    )
                except (TypeError, ValueError):
                    stationary_margin = float("nan")
                if (
                    stationarity_status != "stationary"
                    or not math.isfinite(stationary_margin)
                    or stationary_margin > 0.0
                ):
                    certificates.append(
                        UnresolvedCertificate(
                            record_id=record_id,
                            reason=(
                                "ordinary_unusable_but_stationarity_not_"
                                "certified"
                            ),
                        )
                    )
                    continue
                quotient_eigenvalues = [
                    float(value)
                    for value in summary.get(
                        "quotient_residual_metric_eigenvalues", []
                    )
                ]
                quotient_floor = float(
                    max(0.0, summary.get("quotient_resolution_floor", 0.0))
                )
                if bool(summary.get("quotient_redundant_certified", False)):
                    quotient_upper = float(
                        math.sqrt(
                            max(
                                0.0,
                                max(quotient_eigenvalues)
                                if quotient_eigenvalues
                                else 0.0,
                            )
                        )
                    )
                    certificates.append(
                        QuotientRedundantCertificate(
                            record_id=record_id,
                            quotient_norm_upper_bound=quotient_upper,
                            support_resolution_floor=float(
                                max(quotient_upper, math.sqrt(quotient_floor))
                            ),
                        )
                    )
                    continue
                if (
                    not bool(summary.get("feasible", False))
                    or not bool(
                        summary.get("trust_global_optimality_certified", False)
                    )
                    or not bool(
                        summary.get("marginal_trust_gain_comparison_valid", False)
                    )
                    or not bool(
                        summary.get("quotient_participation_resolved", False)
                    )
                ):
                    certificates.append(
                        UnresolvedCertificate(
                            record_id=record_id,
                            reason=str(summary.get("reason", "numerical_invalid")),
                        )
                    )
                    continue
                stationarity_margin = float(stationary_margin)
                inertia_status = str(
                    summary.get("supported_inertia_status", "unresolved")
                )
                inertia_label_issued = bool(
                    summary.get("supported_inertia_label_issued", False)
                )
                if inertia_label_issued and inertia_status == "negative":
                    certificates.append(
                        SaddleCertificate(
                            record_id=record_id,
                            stationarity_margin=stationarity_margin,
                            minimum_eigenvalue_upper_bound=float(
                                summary.get(
                                    "minimum_hessian_eigenvalue_upper_bound"
                                )
                            ),
                            full_trust_gain_lower_bound=float(
                                max(
                                    0.0,
                                    summary.get(
                                        "full_trust_gain_lower_bound", 0.0
                                    ),
                                )
                            ),
                            active_trust_gain_lower_bound=float(
                                max(
                                    0.0,
                                    summary.get(
                                        "active_restricted_trust_gain_lower_bound",
                                        0.0,
                                    ),
                                )
                            ),
                            active_trust_gain_upper_bound=float(
                                max(
                                    0.0,
                                    summary.get(
                                        "active_restricted_trust_gain_upper_bound",
                                        0.0,
                                    ),
                                )
                            ),
                            quotient_participation_lower_bound=float(
                                max(
                                    0.0,
                                    summary.get(
                                        "quotient_participation_lower_bound", 0.0
                                    ),
                                )
                            ),
                            phase3_cost=float(row["K3"]),
                            novelty_statistic=(
                                None
                                if row["N3"] is None
                                else float(row["N3"])
                            ),
                        )
                    )
                elif inertia_label_issued and inertia_status == "psd":
                    certificates.append(
                        PsdCertificate(
                            record_id=record_id,
                            stationarity_margin=stationarity_margin,
                            minimum_eigenvalue_lower_bound=float(
                                summary.get(
                                    "minimum_hessian_eigenvalue_lower_bound"
                                )
                            ),
                        )
                    )
                else:
                    certificates.append(
                        UnresolvedCertificate(
                            record_id=record_id,
                            reason="supported_hessian_inertia_unresolved",
                        )
                    )
            state_stationarity_summaries: list[dict[str, Any]] = []
            for summary in coordinate_summaries:
                shared_raw = summary.get(
                    "sr_escape_state_stationarity_summary"
                )
                if not isinstance(shared_raw, Mapping):
                    state_stationarity_blocker = (
                        "independent_state_stationarity_summary_missing"
                    )
                    break
                state_stationarity_summaries.append(dict(shared_raw))
            if (
                state_stationarity_blocker is None
                and not state_stationarity_summaries
            ):
                state_stationarity_blocker = "reachable_population_empty"
            shared_stationarity: dict[str, Any] = (
                dict(state_stationarity_summaries[0])
                if state_stationarity_summaries
                else {}
            )
            if state_stationarity_blocker is None:
                shared_digest = hashlib.sha256(
                    json.dumps(
                        shared_stationarity,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()
                other_digests = {
                    hashlib.sha256(
                        json.dumps(
                            payload,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=True,
                            default=str,
                        ).encode("utf-8")
                    ).hexdigest()
                    for payload in state_stationarity_summaries[1:]
                }
                if any(value != shared_digest for value in other_digests):
                    state_stationarity_blocker = (
                        "independent_state_stationarity_summary_mismatch"
                    )
            try:
                independent_margin = float(
                    shared_stationarity.get(
                        "stationarity_margin", float("nan")
                    )
                )
            except (TypeError, ValueError):
                independent_margin = float("nan")
            if (
                state_stationarity_blocker is None
                and (
                    not bool(shared_stationarity.get("valid", False))
                    or str(
                        shared_stationarity.get(
                            "supported_stationarity_status", "unresolved"
                        )
                    )
                    != "stationary"
                    or not math.isfinite(independent_margin)
                    or independent_margin > 0.0
                )
            ):
                state_stationarity_blocker = (
                    "independent_working_state_stationarity_unresolved"
                )
            state_fingerprint = str(sr_escape_state_fingerprint or "").strip()
            comparison_epoch = str(sr_escape_comparison_epoch or "").strip()
            try:
                trust_radius = float(sr_escape_trust_radius)
            except (TypeError, ValueError):
                trust_radius = float("nan")
            if state_stationarity_blocker is None and not state_fingerprint:
                state_stationarity_blocker = "state_fingerprint_missing"
            if state_stationarity_blocker is None and not comparison_epoch:
                state_stationarity_blocker = "comparison_epoch_missing"
            if (
                state_stationarity_blocker is None
                and (not math.isfinite(trust_radius) or trust_radius <= 0.0)
            ):
                state_stationarity_blocker = "trust_radius_invalid"
            try:
                shared_trust_radius = float(
                    shared_stationarity.get("trust_radius", float("nan"))
                )
            except (TypeError, ValueError):
                shared_trust_radius = float("nan")
            radius_match_tolerance = float(
                128.0
                * np.finfo(float).eps
                * max(
                    1.0,
                    abs(trust_radius) if math.isfinite(trust_radius) else 0.0,
                    abs(shared_trust_radius)
                    if math.isfinite(shared_trust_radius)
                    else 0.0,
                )
            )
            if (
                state_stationarity_blocker is None
                and (
                    not math.isfinite(shared_trust_radius)
                    or shared_trust_radius <= 0.0
                    or abs(shared_trust_radius - trust_radius)
                    > radius_match_tolerance
                )
            ):
                state_stationarity_blocker = (
                    "independent_state_stationarity_trust_radius_mismatch"
                )
            support_provenance_digest = str(
                shared_stationarity.get("support_provenance_digest", "")
            ).strip()
            trust_provenance_digest = str(
                shared_stationarity.get("trust_provenance_digest", "")
            ).strip()
            if (
                state_stationarity_blocker is None
                and not support_provenance_digest
            ):
                state_stationarity_blocker = (
                    "independent_state_support_provenance_missing"
                )
            if (
                state_stationarity_blocker is None
                and not trust_provenance_digest
            ):
                state_stationarity_blocker = (
                    "independent_state_trust_provenance_missing"
                )
            if state_stationarity_blocker is None:
                state_stationarity_certificate = StateStationarityCertificate(
                    state_fingerprint=state_fingerprint,
                    reachable_population_digest=reachable_population_digest(
                        reachable_record_ids
                    ),
                    comparison_epoch=comparison_epoch,
                    support_provenance_digest=support_provenance_digest,
                    trust_provenance_digest=trust_provenance_digest,
                    trust_radius=float(trust_radius),
                    stationarity_margin=float(independent_margin),
                )
            audit = ReachablePopulationAudit(
                reachable_record_ids=reachable_record_ids,
                certificates=tuple(certificates),
                state_stationarity=state_stationarity_certificate,
            )
            for certificate in certificates:
                kind = type(certificate).__name__
                certificate_kind_counts[kind] = int(
                    certificate_kind_counts.get(kind, 0) + 1
                )
                if isinstance(certificate, UnresolvedCertificate):
                    reason = str(certificate.reason)
                    unresolved_certificate_reason_counts[reason] = int(
                        unresolved_certificate_reason_counts.get(reason, 0)
                        + 1
                    )
            certificate_by_id = {
                certificate.record_id: certificate
                for certificate in certificates
            }
            for record_id, raw_record, summary in zip(
                reachable_record_ids, records, coordinate_summaries
            ):
                certificate = certificate_by_id[record_id]
                summary_json = json.dumps(
                    dict(summary),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    default=(
                        lambda value: (
                            value.tolist()
                            if isinstance(value, np.ndarray)
                            else value.item()
                            if isinstance(value, np.generic)
                            else str(value)
                        )
                    ),
                )
                certificate_payload: dict[str, Any] = {
                    "schema": "sr_escape_reachable_certificate_record_v1",
                    "record_id": str(record_id),
                    "candidate_label": str(
                        raw_record.get("candidate_label", "")
                    ),
                    "candidate_pool_index": int(
                        raw_record.get("candidate_pool_index", -1)
                    ),
                    "position_id": int(raw_record.get("position_id", -1)),
                    "certificate_kind": type(certificate).__name__,
                    "coordinate_summary_digest": hashlib.sha256(
                        summary_json.encode("utf-8")
                    ).hexdigest(),
                    "working_state_fingerprint": str(
                        sr_escape_state_fingerprint or ""
                    ),
                    "state_stationarity_token_digest": (
                        state_stationarity_certificate.token_digest
                        if state_stationarity_certificate is not None
                        else None
                    ),
                    "trust_radius": (
                        float(state_stationarity_certificate.trust_radius)
                        if state_stationarity_certificate is not None
                        else None
                    ),
                }
                if isinstance(certificate, PsdCertificate):
                    certificate_payload.update(
                        {
                            "stationarity_margin": float(
                                certificate.stationarity_margin
                            ),
                            "minimum_eigenvalue_lower_bound": float(
                                certificate.minimum_eigenvalue_lower_bound
                            ),
                        }
                    )
                elif isinstance(certificate, QuotientRedundantCertificate):
                    certificate_payload.update(
                        {
                            "quotient_norm_upper_bound": float(
                                certificate.quotient_norm_upper_bound
                            ),
                            "support_resolution_floor": float(
                                certificate.support_resolution_floor
                            ),
                        }
                    )
                elif isinstance(certificate, SaddleCertificate):
                    certificate_payload.update(
                        {
                            "stationarity_margin": float(
                                certificate.stationarity_margin
                            ),
                            "minimum_eigenvalue_upper_bound": float(
                                certificate.minimum_eigenvalue_upper_bound
                            ),
                            "marginal_gain_lower_bound": float(
                                certificate.marginal_gain_lower_bound
                            ),
                        }
                    )
                elif isinstance(certificate, NonstationaryCertificate):
                    certificate_payload.update(
                        {
                            "stationarity_margin": float(
                                certificate.stationarity_margin
                            ),
                            "active_trust_gain_lower_bound": float(
                                certificate.active_trust_gain_lower_bound
                            ),
                        }
                    )
                else:
                    assert isinstance(certificate, UnresolvedCertificate)
                    certificate_payload["unresolved_reason"] = str(
                        certificate.reason
                    )
                reachable_certificate_table.append(certificate_payload)

        decision = select_sr_escape_path(
            mode=escape_mode,
            ordinary=ordinary_certificate,
            audit=audit,
        )
        modeled_minimum_eligibility = assess_exposed_family_psd(audit)
        if combined_mode_requested and (
            bool(modeled_minimum_eligibility.eligible)
            != bool(decision.stage_b_eligible)
        ):
            raise RuntimeError(
                "SR modeled-minimum pure-core eligibility diverged from the "
                "combined-mode precedence controller."
            )
        modeled_minimum_core_payload = {
            "schema": _SR_MODELED_MINIMUM_CORE_TELEMETRY_SCHEMA,
            "version": 1,
            "combined_mode_requested": combined_mode_requested,
            "mathematical_eligibility": (
                modeled_minimum_eligibility.to_dict()
            ),
            "state_token_digest": (
                modeled_minimum_eligibility.state_token.digest
                if modeled_minimum_eligibility.eligible
                else None
            ),
            "pure_core_available": True,
            "execution_implemented": False,
            "actionable": False,
            "remaining_provider_runtime_checkpoint_blockers": list(
                _SR_MODELED_MINIMUM_EXECUTION_BLOCKERS
            ),
        }
        if decision.kind is SRControllerDecisionKind.ORDINARY:
            eligible_ids = {record_ids[index] for index in ordinary_indices}
        elif decision.kind is SRControllerDecisionKind.SADDLE_SINGLETON:
            eligible_ids = (
                {str(decision.record_id)}
                if decision.record_id not in {None, ""}
                else set()
            )
        else:
            eligible_ids = set()

        updated_rescored: list[dict[str, Any]] = []
        for index, (record_id, record) in enumerate(zip(record_ids, rescored)):
            updated = dict(record)
            feature = updated.get("feature")
            if not isinstance(feature, CandidateFeatures):
                raise ValueError("SR escape rescore lost CandidateFeatures.")
            eligible = bool(record_id in eligible_ids)
            if (
                decision.kind is SRControllerDecisionKind.SADDLE_SINGLETON
                and eligible
            ):
                score = float(decision.acquisition)
                benefit = float(decision.marginal_gain_lower_bound)
            elif decision.kind is SRControllerDecisionKind.ORDINARY and eligible:
                ordinary_summary = ordinary_summaries[index]
                if ordinary_summary is None:
                    raise RuntimeError(
                        "SR ordinary admission lost its preserved v1 summary."
                    )
                score = float(ordinary_scores[index])
                benefit = float(ordinary_summary.get("joint_gain", 0.0))
            else:
                score = float("-inf")
                benefit = 0.0
            exact_map_transaction_required = bool(
                (
                    decision.kind
                    is SRControllerDecisionKind.SADDLE_SINGLETON
                    and eligible
                )
                or (
                    decision.kind
                    in {
                        SRControllerDecisionKind.ACTIVE_STATIONARITY_CORRECTION,
                        SRControllerDecisionKind.ACTIVE_ONLY_CORRECTION,
                    }
                    and decision.actionable
                    and record_id == decision.certificate_record_id
                )
            )
            components = {
                **dict(feature.phase_score_components or {}),
                "sr_escape_active": 1.0,
                "sr_escape_admission_eligible": float(1.0 if eligible else 0.0),
                "sr_escape_marginal_gain_lower_bound": float(
                    decision.marginal_gain_lower_bound
                    if eligible
                    and decision.kind
                    is SRControllerDecisionKind.SADDLE_SINGLETON
                    else 0.0
                ),
                "phase3_primary_score": float(score),
                "phase3_reduced_score": float(score),
                "selector_score": float(score),
            }
            updated_feature = _replace_feature(
                feature,
                full_v2_score=float(score),
                phase3_reduced_trust_gain=float(benefit),
                phase3_primary_score=float(score),
                selector_score=float(score),
                phase_score_components=components,
            )
            updated.update(
                {
                    "feature": updated_feature,
                    "full_v2_score": float(score),
                    "selector_score": float(score),
                    "phase3_primary_score": float(score),
                    "phase3_reduced_trust_gain": float(benefit),
                    "sr_escape_record_id": str(record_id),
                    "sr_escape_decision_kind": str(decision.kind.value),
                    "sr_escape_admission_eligible": bool(eligible),
                    "sr_escape_local_model_classification": (
                        "stationary_negative_curvature_model"
                        if decision.kind
                        is SRControllerDecisionKind.SADDLE_SINGLETON
                        and eligible
                        else None
                    ),
                    "sr_escape_exact_map_transaction_required": bool(
                        exact_map_transaction_required
                    ),
                    "sr_escape_physical_transition_certified": False,
                    "sr_escape_ordinary_selection_authority": (
                        "literal_sr_snake_preserved_v1"
                        if decision.kind is SRControllerDecisionKind.ORDINARY
                        else None
                    ),
                    "sr_escape_ordinary_model_live": bool(
                        record_id in live_ordinary_record_ids
                    ),
                    "sr_escape_ordinary_model_contradicted": bool(
                        record_id in contradicted_ordinary_record_ids
                    ),
                }
            )
            updated_rescored.append(updated)
        rescored = updated_rescored
        for index, row in enumerate(telemetry_rows):
            row_exact_map_required = bool(
                (
                    decision.kind
                    is SRControllerDecisionKind.SADDLE_SINGLETON
                    and record_ids[index] in eligible_ids
                )
                or (
                    decision.kind
                    in {
                        SRControllerDecisionKind.ACTIVE_STATIONARITY_CORRECTION,
                        SRControllerDecisionKind.ACTIVE_ONLY_CORRECTION,
                    }
                    and decision.actionable
                    and record_ids[index] == decision.certificate_record_id
                )
            )
            row.update(
                {
                    "sr_escape_record_id": str(record_ids[index]),
                    "sr_escape_admission_eligible": bool(
                        record_ids[index] in eligible_ids
                    ),
                    "sr_escape_decision_kind": str(decision.kind.value),
                    "sr_escape_exact_map_transaction_required": (
                        row_exact_map_required
                    ),
                    "sr_escape_ordinary_score": float(
                        ordinary_scores[index]
                    ),
                    "sr_escape_ordinary_summary": (
                        dict(ordinary_summaries[index])
                        if ordinary_summaries[index] is not None
                        else None
                    ),
                    "sr_escape_ordinary_model_live": bool(
                        record_ids[index] in live_ordinary_record_ids
                    ),
                    "sr_escape_ordinary_model_contradicted": bool(
                        record_ids[index]
                        in contradicted_ordinary_record_ids
                    ),
                }
            )
        sr_decision_payload = {
            "active": True,
            "mode": str(escape_mode.value),
            "decision_kind": str(decision.kind.value),
            "reason": str(decision.reason),
            "record_id": decision.record_id,
            "certificate_record_id": decision.certificate_record_id,
            "actionable": bool(decision.actionable),
            "consumes_singleton": bool(decision.consumes_singleton),
            "stage_b_eligible": bool(decision.stage_b_eligible),
            "acquisition": float(decision.acquisition),
            "marginal_gain_lower_bound": float(
                decision.marginal_gain_lower_bound
            ),
            "classification_authority": (
                "supported_local_model_only"
                if decision.kind
                is SRControllerDecisionKind.SADDLE_SINGLETON
                else "controller_precedence"
            ),
            "exact_map_transaction_required": bool(
                decision.actionable
                and decision.kind
                in {
                    SRControllerDecisionKind.SADDLE_SINGLETON,
                    SRControllerDecisionKind.ACTIVE_STATIONARITY_CORRECTION,
                    SRControllerDecisionKind.ACTIVE_ONLY_CORRECTION,
                }
            ),
            "physical_transition_certified": False,
            "admission_eligible_record_ids": sorted(eligible_ids),
            "reachable_population_complete": bool(
                audit.complete if audit is not None else False
            ),
            "reachable_population_digest": (
                audit.expected_population_digest
                if audit is not None
                else reachable_population_digest(reachable_record_ids)
            ),
            "state_stationarity_certified": bool(
                audit.state_stationarity_certified
                if audit is not None
                else False
            ),
            "state_stationarity_blocker": state_stationarity_blocker,
            "state_stationarity_certificate": (
                {
                    "token_digest": (
                        state_stationarity_certificate.token_digest
                    ),
                    "state_fingerprint": (
                        state_stationarity_certificate.state_fingerprint
                    ),
                    "reachable_population_digest": (
                        state_stationarity_certificate.reachable_population_digest
                    ),
                    "comparison_epoch": (
                        state_stationarity_certificate.comparison_epoch
                    ),
                    "support_provenance_digest": (
                        state_stationarity_certificate.support_provenance_digest
                    ),
                    "trust_provenance_digest": (
                        state_stationarity_certificate.trust_provenance_digest
                    ),
                    "trust_radius": float(
                        state_stationarity_certificate.trust_radius
                    ),
                    "stationarity_margin": float(
                        state_stationarity_certificate.stationarity_margin
                    ),
                }
                if state_stationarity_certificate is not None
                else None
            ),
            "reachable_record_ids": list(reachable_record_ids),
            "certificate_kind_counts": dict(
                sorted(certificate_kind_counts.items())
            ),
            "reachable_certificate_table_schema": (
                "sr_escape_reachable_certificate_table_v1"
            ),
            "reachable_certificate_table": list(
                reachable_certificate_table
            ),
            "unresolved_certificate_reason_counts": dict(
                sorted(unresolved_certificate_reason_counts.items())
            ),
            "ordinary_record_ids": sorted(ordinary_record_ids),
            "ordinary_model_live_record_ids": sorted(
                live_ordinary_record_ids
            ),
            "contradicted_ordinary_record_ids": sorted(
                contradicted_ordinary_record_ids
            ),
            "ordinary_model_live_authority": (
                "branch_local_state_fingerprint_and_record_id_ledger_v1"
            ),
            "modeled_minimum_core": dict(modeled_minimum_core_payload),
        }

    telemetry = {
        "schema": "historical_singleton_coordinate_model_phase3_rescore_v1",
        "score_formula": str(phase3_score_formula),
        "authority": "historical_phase3_with_coordinate_model_benefit_only",
        "input_record_count": int(len(records)),
        "output_record_count": int(len(rescored)),
        "membership_preserved": True,
        "order_preserved": True,
        "phase2_rescoring_applied": False,
        "phase3_candidate_gain_policy": str(
            expected_candidate_gain_policy
        ),
        "phase3_candidate_gain_semantics": (
            "full_joint_minus_candidate_independent_active_only_v1"
            if expected_candidate_gain_policy
            == PHASE3_CANDIDATE_GAIN_JOINT_MINUS_ACTIVE_ONLY_V1
            else "full_joint_trust_gain_legacy_v1"
        ),
        "phase3_novelty_multiplier_policy": str(
            phase3_novelty_multiplier_policy
        ),
        "phase3_gram_novelty_policy": str(phase3_gram_novelty_policy),
        "hardware_cost_normalization_mode": str(
            hardware_cost_normalization_mode
        ),
        "phase3_signed_factor_consumer_semantic_version": (
            phase3_signed_factor_consumer_semantic_version
        ),
        "symmetric_hardware_cost_factor_applied": bool(
            multiplicative_signed_hardware_cost_active
        ),
        "multiplicative_signed_hardware_cost_factor_applied": bool(
            multiplicative_signed_hardware_cost_active
        ),
        "phase3_novelty_status": (
            GRAM_NOVELTY_STATUS_COMPUTED_FOR_GEOMETRY_EXPANSION
            if geometry_expansion_active
            else GRAM_NOVELTY_STATUS_NOT_COMPUTED_FOR_ORDINARY_SCORING
        ),
        "phase3_novelty_query_charge": 0,
        "phase3_novelty_classical_solve_count": int(
            sum(
                int(row.get("phase3_novelty_classical_solve_count", 0))
                for row in telemetry_rows
            )
        ),
        "phase3_novelty_applied": bool(geometry_expansion_active),
        "measured_n3_retained": bool(
            any(row.get("measured_N3") is not None for row in telemetry_rows)
        ),
        "batching_applied": False,
        "geometry_expansion_active": bool(geometry_expansion_active),
        "geometry_expansion_mode": (
            "collective_span_novelty_over_cost_v1"
            if geometry_expansion_active
            else "off"
        ),
        "geometry_expansion_reason": (
            geometry_expansion_reason
            if geometry_expansion_active
            else None
        ),
        "geometry_expansion_score_formula": (
            (
                "N3*hardware_cost_score_factor/(1+K3)"
                if geometry_expansion_active
                else "off"
            )
            if multiplicative_signed_hardware_cost_active
            else (
                "N3/(1+K3)"
                if geometry_expansion_active
                else "off"
            )
        ),
        "geometry_expansion_lazy_novelty_activation": bool(
            geometry_expansion_active
        ),
        "sr_escape_controller": dict(sr_decision_payload),
        "feasible_record_count": int(
            sum(bool(row["feasible"]) for row in telemetry_rows)
        ),
        "infeasible_record_count": int(
            sum(not bool(row["feasible"]) for row in telemetry_rows)
        ),
        "infeasible_reason_counts": dict(sorted(infeasible_reasons.items())),
        "records": telemetry_rows,
    }
    return rescored, telemetry


def _phase2_joint_response_record_payload(
    summary: Mapping[str, Any],
    *,
    k2: float,
    denominator: float,
    score: float,
    legacy_score: float | None,
    legacy_formula: str,
    legacy_novelty: float | None,
) -> dict[str, Any]:
    return {
        "schema": "phase2_joint_response_singleton_v1",
        "feasible": bool(summary.get("feasible", False)),
        "reason": str(summary.get("reason", "unknown")),
        "score_formula": PHASE2_JOINT_RESPONSE_SCORE_FORMULA,
        "joint_gain": float(max(0.0, summary.get("joint_gain", 0.0))),
        "K2": float(k2),
        "denominator_1_plus_K2": float(denominator),
        "score": float(score),
        "legacy_product_score": (
            None if legacy_score is None else float(legacy_score)
        ),
        "legacy_product_formula": str(legacy_formula),
        "legacy_novelty_scalar": (
            None if legacy_novelty is None else float(legacy_novelty)
        ),
        "legacy_novelty_authority": "telemetry_only",
        "subset_workspace_indices": [
            int(value) for value in summary.get("subset_workspace_indices", [])
        ],
        "gram_eigenvalues": [
            float(value) for value in summary.get("gram_eigenvalues", [])
        ],
        "effective_rank": (
            None
            if summary.get("effective_rank") is None
            else int(summary.get("effective_rank"))
        ),
        "rank_floor": (
            None
            if summary.get("rank_floor") is None
            else float(summary.get("rank_floor"))
        ),
        "gram_condition_number": (
            None
            if summary.get("gram_condition_number") is None
            else float(summary.get("gram_condition_number"))
        ),
        "trust_lambda": (
            None
            if summary.get("trust_lambda") is None
            else float(summary.get("trust_lambda"))
        ),
        "trust_clipped": bool(summary.get("trust_clipped", False)),
        "trust_regularization_applied": bool(
            summary.get(
                "trust_regularization_applied",
                summary.get("trust_clipped", False),
            )
        ),
        "trust_radius_binding": bool(
            summary.get("trust_radius_binding", False)
        ),
        "active_parameter_relaxation": [
            float(value)
            for value in summary.get("active_parameter_relaxation", [])
        ],
        "candidate_step": [
            float(value) for value in summary.get("batch_coordinate_step", [])
        ],
        "D_geo": float(max(0.0, summary.get("D_geo", 0.0))),
        "unconstrained_predicted_reduction": (
            None
            if summary.get("unconstrained_predicted_reduction") is None
            else float(summary.get("unconstrained_predicted_reduction"))
        ),
        "applied_predicted_reduction": float(
            max(0.0, summary.get("applied_predicted_reduction", 0.0))
        ),
        "joint_solve_policy": str(summary.get("joint_solve_policy", "")),
    }


def evaluate_phase2_joint_response_singletons(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    pauli_action_cache: dict[str, Any] | None = None,
    scope: str = "phase2",
) -> Phase2JointResponseEvaluation:
    """Rank Phase-II records with the singleton limit of the joint solver."""

    copied = [dict(record) for record in records]
    singleton_cfg = replace(
        cfg,
        batch_target_size=1,
        batch_size_cap=1,
    )
    workspace = _build_batch_full_geometry_workspace(
        copied,
        cfg=singleton_cfg,
        selected_ops=selected_ops,
        theta=np.asarray(theta, dtype=float),
        psi_ref=np.asarray(psi_ref, dtype=complex),
        psi_state=np.asarray(psi_state, dtype=complex),
        h_compiled=h_compiled,
        pauli_action_cache=pauli_action_cache,
    )
    evaluated: list[dict[str, Any]] = []
    record_payloads: list[dict[str, Any]] = []
    infeasible_reasons: dict[str, int] = {}
    for record in copied:
        feature = record.get("feature")
        legacy_score = (
            None
            if not isinstance(feature, CandidateFeatures)
            or feature.phase2_raw_score is None
            else float(feature.phase2_raw_score)
        )
        if legacy_score is None and record.get("phase2_raw_score") is not None:
            legacy_score = float(record.get("phase2_raw_score"))
        legacy_formula = (
            str(feature.phase2_raw_score_formula)
            if isinstance(feature, CandidateFeatures)
            else str(record.get("phase2_raw_score_formula", ""))
        )
        legacy_novelty = (
            None
            if not isinstance(feature, CandidateFeatures)
            or feature.phase2_raw_novelty is None
            else float(feature.phase2_raw_novelty)
        )
        summary = workspace.summary_for_records([record])
        feasible = bool(summary.get("feasible", False))
        reason = str(summary.get("reason", "unknown"))
        if not feasible:
            infeasible_reasons[reason] = int(infeasible_reasons.get(reason, 0) + 1)
        joint_gain = float(
            max(0.0, summary.get("joint_gain", 0.0)) if feasible else 0.0
        )
        k2 = float(_phase2_record_k2(record))
        denominator = float(
            max(float(singleton_cfg.cheap_score_eps), 1.0 + float(k2))
        )
        score = float(joint_gain / denominator) if feasible else float("-inf")
        payload = _phase2_joint_response_record_payload(
            summary,
            k2=k2,
            denominator=denominator,
            score=score,
            legacy_score=legacy_score,
            legacy_formula=legacy_formula,
            legacy_novelty=legacy_novelty,
        )
        updated = {
            **record,
            "phase2_raw_score": float(score),
            "phase2_raw_score_formula": PHASE2_JOINT_RESPONSE_SCORE_FORMULA,
            "phase2_selector_mode": "joint_response_singleton_v1",
            "phase2_joint_response_score": float(score),
            "phase2_joint_response_gain": float(joint_gain),
            "phase2_joint_response_feasible": bool(feasible),
            "phase2_joint_response": dict(payload),
            "phase2_legacy_product_score": legacy_score,
            "phase2_legacy_product_formula": str(legacy_formula),
            "phase2_novelty_authority": "telemetry_only",
            "selector_score": float(score),
            "selector_burden": float(denominator),
        }
        if isinstance(feature, CandidateFeatures):
            updated["feature"] = _replace_feature(
                feature,
                phase2_raw_score_formula=PHASE2_JOINT_RESPONSE_SCORE_FORMULA,
                phase2_raw_score=float(score),
                phase2_burden_total=float(denominator),
                selector_score=float(score),
                selector_burden=float(denominator),
                phase_score_components={
                    **dict(feature.phase_score_components),
                    "phase2_selector_mode": "joint_response_singleton_v1",
                    "phase2_joint_response_gain": float(joint_gain),
                    "phase2_joint_response_score": float(score),
                    "phase2_legacy_product_score": legacy_score,
                    "phase2_legacy_novelty_telemetry": legacy_novelty,
                },
                phase_cost_components={
                    **dict(feature.phase_cost_components),
                    "phase2_K2": float(k2),
                    "phase2_burden_total": float(denominator),
                },
            )
        evaluated.append(updated)
        record_payloads.append(
            {
                "candidate_label": str(_batch_record_label(record)),
                "candidate_pool_index": int(record.get("candidate_pool_index", -1)),
                "position_id": int(record.get("position_id", -1)),
                **dict(payload),
            }
        )
    workspace_telemetry = workspace.build_telemetry()
    valid_reuse_count = int(
        workspace_telemetry.get("validated_phase2_gradient_reuse_count", 0)
    )
    telemetry = {
        "schema": "route_a_phase2_joint_response_population_v1",
        "scope": str(scope),
        "selector_mode_requested": "joint_response_singleton_v1",
        "selector_mode_effective": "joint_response_singleton_v1",
        "authority": "joint_ansatz_plus_singleton_response",
        "score_formula": PHASE2_JOINT_RESPONSE_SCORE_FORMULA,
        "legacy_novelty_authority": "telemetry_only",
        "input_record_count": int(len(copied)),
        "feasible_record_count": int(
            sum(bool(row["phase2_joint_response_feasible"]) for row in evaluated)
        ),
        "infeasible_record_count": int(
            sum(not bool(row["phase2_joint_response_feasible"]) for row in evaluated)
        ),
        "infeasible_reason_counts": dict(sorted(infeasible_reasons.items())),
        "workspace_source": (
            "phase2_singleton_blocks_v1"
            if valid_reuse_count == len(copied)
            else "phase2_blocks_plus_state_scoped_repair_v1"
        ),
        "candidate_pair_measurement_count": 0,
        "records": record_payloads,
        "geometry_workspace": workspace_telemetry,
    }
    return Phase2JointResponseEvaluation(
        records=tuple(evaluated),
        telemetry=telemetry,
        workspace=workspace,
    )


def _phase2_reported_candidate_descent_gradient(
    record: Mapping[str, Any],
    cfg: FullScoreConfig,
) -> float:
    feat = record.get("feature")
    if isinstance(feat, CandidateFeatures):
        magnitude = float(max(0.0, _selector_gradient_lcb(feat, cfg)))
        signed_gradient = float(feat.g_signed)
        if signed_gradient > 0.0:
            return -magnitude
        if signed_gradient < 0.0:
            return magnitude
        return 0.0
    return float(max(0.0, _batch_record_score(record, "phase2_raw_score", 0.0)))


def _required_joint_candidate_pairs(
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size_cap: int,
) -> tuple[tuple[int, int], ...]:
    if int(batch_size_cap) <= 1:
        return ()
    pairs: list[tuple[int, int]] = []
    for left in range(len(records)):
        left_identity = _batch_record_generator_identity(records[left])
        for right in range(left + 1, len(records)):
            if left_identity == _batch_record_generator_identity(records[right]):
                continue
            pairs.append((int(left), int(right)))
    return tuple(pairs)


def _selector_scaffold_context(
    *,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    active_indices: Sequence[int],
    h_compiled: CompiledPolynomialAction,
    measure_old_old_geometry: bool = True,
) -> _ScaffoldDerivativeContext:
    """Create the state-scoped context used for selector cache repairs."""

    active_count = int(len(active_indices))
    current = np.asarray(psi_state, dtype=complex).reshape(-1)
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    reference = np.asarray(psi_ref, dtype=complex).reshape(-1)
    hpsi_current = apply_compiled_polynomial(current, h_compiled)
    if active_count:
        executor = _executor_for_terms(selected_ops, pauli_action_cache={})
        if bool(measure_old_old_geometry):
            reconstructed, dpsi_window, d2psi_window = (
                _propagate_executor_derivatives(
                    executor=executor,
                    theta=theta_vec,
                    psi_ref=reference,
                    active_indices=active_indices,
                )
            )
        else:
            reconstructed, dpsi_window, _unused_sparse_second = (
                _propagate_executor_sparse_second_derivatives(
                    executor=executor,
                    theta=theta_vec,
                    psi_ref=reference,
                    active_indices=active_indices,
                    second_derivative_pairs=(),
                )
            )
            d2psi_window = None
        overlap = complex(np.vdot(current, reconstructed))
        phase = overlap / abs(overlap) if abs(overlap) > 0.0 else 1.0 + 0.0j
        state_reconstruction_delta_norm = float(
            np.linalg.norm(
                np.asarray(reconstructed, dtype=complex).reshape(-1)
                / phase
                - current
            )
        )
        dpsi_window = [
            np.asarray(value / phase, dtype=complex) for value in dpsi_window
        ]
        if d2psi_window is not None:
            d2psi_window = [
                [np.asarray(value / phase, dtype=complex) for value in row]
                for row in d2psi_window
            ]
        tangents_window = [
            _horizontal_tangent(current, value) for value in dpsi_window
        ]
        if bool(measure_old_old_geometry):
            assert d2psi_window is not None
            Q_window = _tangent_overlap_matrix(tangents_window)
            hdpsi_window = [
                apply_compiled_polynomial(value, h_compiled)
                for value in dpsi_window
            ]
            H_window = np.zeros((active_count, active_count), dtype=float)
            for row in range(active_count):
                for col in range(active_count):
                    H_window[row, col] = _energy_hessian_entry(
                        dpsi_left=dpsi_window[row],
                        dpsi_right=dpsi_window[col],
                        d2psi=d2psi_window[row][col],
                        hpsi_state=hpsi_current,
                        hdpsi_right=hdpsi_window[col],
                    )
            H_window = 0.5 * (H_window + H_window.T)
        else:
            # The active derivatives are still required to measure exact
            # candidate--active blocks.  Their old--old contractions are not.
            Q_window = np.zeros((active_count, active_count), dtype=float)
            H_window = np.zeros((active_count, active_count), dtype=float)
    else:
        dpsi_window = []
        tangents_window = []
        Q_window = np.zeros((0, 0), dtype=float)
        H_window = np.zeros((0, 0), dtype=float)
        state_reconstruction_delta_norm = 0.0
    return _ScaffoldDerivativeContext(
        psi_state=current,
        hpsi_state=hpsi_current,
        selected_ops=tuple(selected_ops),
        theta=theta_vec.copy(),
        psi_ref=reference.copy(),
        state_fingerprint=_array_fingerprint(current),
        ordered_scaffold_fingerprint=_ordered_scaffold_fingerprint(selected_ops),
        theta_fingerprint=_array_fingerprint(theta_vec),
        refit_window_indices=tuple(int(index) for index in active_indices),
        dpsi_window=tuple(dpsi_window),
        tangents_window=tuple(tangents_window),
        Q_window=np.asarray(Q_window, dtype=float),
        H_window_hessian=np.asarray(H_window, dtype=float),
        state_reconstruction_delta_norm=float(
            state_reconstruction_delta_norm
        ),
        old_old_geometry_measured=bool(measure_old_old_geometry),
        old_old_metric_measured=bool(measure_old_old_geometry),
        old_old_hessian_measured=bool(measure_old_old_geometry),
        old_old_hessian_status=(
            "exact_measured_v1"
            if bool(measure_old_old_geometry)
            else "not_acquired_placeholder_v1"
        ),
    )


def _phase3_material_window_old_old_pair_plan(
    *,
    active_indices: Sequence[int],
    retained_indices: Sequence[int],
) -> dict[str, Any]:
    """Return the exact retained/omitted symmetric old--old pair partition."""

    active = tuple(int(value) for value in active_indices)
    retained = tuple(int(value) for value in retained_indices)
    if len(set(active)) != len(active):
        raise ValueError("Material-window active indices contain duplicates.")
    if len(set(retained)) != len(retained):
        raise ValueError("Material-window retained indices contain duplicates.")
    active_set = set(active)
    if any(value not in active_set for value in retained):
        raise ValueError(
            "Material-window retained indices are not a subset of active indices."
        )
    retained_set = set(retained)
    omitted = tuple(value for value in active if value not in retained_set)
    local = {value: index for index, value in enumerate(active)}

    def _ordered_pair(left: int, right: int) -> tuple[int, int]:
        return (
            (int(left), int(right))
            if local[int(left)] <= local[int(right)]
            else (int(right), int(left))
        )

    retained_retained = tuple(
        _ordered_pair(retained[left], retained[right])
        for left in range(len(retained))
        for right in range(left, len(retained))
    )
    retained_omitted = tuple(
        _ordered_pair(left, right)
        for left in retained
        for right in omitted
    )
    omitted_omitted = tuple(
        _ordered_pair(omitted[left], omitted[right])
        for left in range(len(omitted))
        for right in range(left, len(omitted))
    )
    initial_pairs = tuple([*retained_retained, *retained_omitted])
    return {
        "schema": "phase3_material_window_old_old_pair_plan_v1",
        "active_indices": list(active),
        "retained_indices": list(retained),
        "omitted_indices": list(omitted),
        "retained_retained_pairs": [list(pair) for pair in retained_retained],
        "retained_omitted_pairs": [list(pair) for pair in retained_omitted],
        "omitted_omitted_refresh_pairs": [
            list(pair) for pair in omitted_omitted
        ],
        "initial_pairs": [list(pair) for pair in initial_pairs],
        "initial_pair_count": int(len(initial_pairs)),
        "refresh_pair_count": int(len(omitted_omitted)),
        "full_pair_count": int(len(initial_pairs) + len(omitted_omitted)),
    }


def _acquire_sparse_exact_old_old_geometry(
    *,
    scaffold_context: _ScaffoldDerivativeContext,
    pair_plan: Sequence[tuple[int, int]],
    gradient_indices: Sequence[int],
    h_compiled: CompiledPolynomialAction,
    pauli_action_cache: dict[str, Any] | None,
    state_consistency_tolerance: float,
    acquisition_stage: str,
) -> dict[str, Any]:
    """Acquire only named old--old Gram/Hessian pairs and gradients.

    Pair and gradient indices are logical indices in the scaffold's active
    registry. Missing entries are absent rather than represented by zeros.
    """

    if bool(scaffold_context.old_old_geometry_measured):
        raise ValueError(
            "Sparse old--old acquisition requires a no-old-old scaffold context."
        )
    active = tuple(int(value) for value in scaffold_context.refit_window_indices)
    if len(set(active)) != len(active):
        raise ValueError("Sparse old--old active registry contains duplicates.")
    local = {value: index for index, value in enumerate(active)}
    normalized_pairs: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for raw_left, raw_right in pair_plan:
        left = int(raw_left)
        right = int(raw_right)
        if left not in local or right not in local:
            raise ValueError(
                "Sparse old--old pair references an index outside the active registry."
            )
        local_pair = tuple(sorted((int(local[left]), int(local[right]))))
        if local_pair in seen_pairs:
            continue
        seen_pairs.add(local_pair)
        normalized_pairs.append(local_pair)
    normalized_pairs.sort()

    gradients = tuple(int(value) for value in gradient_indices)
    if len(set(gradients)) != len(gradients):
        raise ValueError("Sparse old--old gradient indices contain duplicates.")
    if any(value not in local for value in gradients):
        raise ValueError(
            "Sparse old--old gradient index is outside the active registry."
        )

    tolerance = float(max(1e-12, state_consistency_tolerance))
    if normalized_pairs:
        executor = _executor_for_terms(
            scaffold_context.selected_ops,
            pauli_action_cache=pauli_action_cache,
        )
        reconstructed, derivatives, second_derivatives = (
            _propagate_executor_sparse_second_derivatives(
                executor=executor,
                theta=np.asarray(scaffold_context.theta, dtype=float),
                psi_ref=np.asarray(scaffold_context.psi_ref, dtype=complex),
                active_indices=active,
                second_derivative_pairs=normalized_pairs,
            )
        )
        current = np.asarray(scaffold_context.psi_state, dtype=complex).reshape(-1)
        reconstructed_state = np.asarray(reconstructed, dtype=complex).reshape(-1)
        overlap = complex(np.vdot(current, reconstructed_state))
        phase = overlap / abs(overlap) if abs(overlap) > 0.0 else 1.0 + 0.0j
        state_delta_norm = float(
            np.linalg.norm(reconstructed_state / phase - current)
        )
        if state_delta_norm > tolerance:
            raise ValueError(
                "Sparse old--old geometry reconstructed an inconsistent state: "
                f"delta={state_delta_norm:.6g}, tolerance={tolerance:.6g}."
            )
        dpsi = tuple(
            np.asarray(value / phase, dtype=complex) for value in derivatives
        )
        d2psi = {
            pair: np.asarray(value / phase, dtype=complex)
            for pair, value in second_derivatives.items()
        }
    else:
        state_delta_norm = 0.0
        dpsi = tuple(
            np.asarray(value, dtype=complex)
            for value in scaffold_context.dpsi_window
        )
        d2psi = {}
    if len(dpsi) != len(active):
        raise ValueError("Sparse old--old derivative registry is incomplete.")
    for local_index, expected in enumerate(scaffold_context.dpsi_window):
        if not np.allclose(
            dpsi[local_index],
            np.asarray(expected, dtype=complex),
            atol=tolerance,
            rtol=1e-7,
        ):
            raise ValueError(
                "Sparse old--old first derivatives differ from the screen context."
            )

    current = np.asarray(scaffold_context.psi_state, dtype=complex)
    hpsi_state = np.asarray(scaffold_context.hpsi_state, dtype=complex)
    tangents = tuple(_horizontal_tangent(current, value) for value in dpsi)
    hdpsi = tuple(
        apply_compiled_polynomial(value, h_compiled) for value in dpsi
    )
    pair_entries: list[dict[str, Any]] = []
    for left_local, right_local in normalized_pairs:
        mixed_second = d2psi[(left_local, right_local)]
        left_right = _energy_hessian_entry(
            dpsi_left=dpsi[left_local],
            dpsi_right=dpsi[right_local],
            d2psi=mixed_second,
            hpsi_state=hpsi_state,
            hdpsi_right=hdpsi[right_local],
        )
        right_left = _energy_hessian_entry(
            dpsi_left=dpsi[right_local],
            dpsi_right=dpsi[left_local],
            d2psi=mixed_second,
            hpsi_state=hpsi_state,
            hdpsi_right=hdpsi[left_local],
        )
        pair_entries.append(
            {
                "local_pair": [int(left_local), int(right_local)],
                "active_index_pair": [
                    int(active[left_local]),
                    int(active[right_local]),
                ],
                "G": float(
                    np.real(np.vdot(tangents[left_local], tangents[right_local]))
                ),
                "H": float(0.5 * (left_right + right_left)),
            }
        )
    gradient_entries = [
        {
            "local_index": int(local[index]),
            "active_index": int(index),
            "g": float(
                -2.0 * np.real(np.vdot(dpsi[int(local[index])], hpsi_state))
            ),
        }
        for index in gradients
    ]
    return {
        "schema": "phase3_sparse_old_old_geometry_acquisition_v1",
        "acquisition_stage": str(acquisition_stage),
        "active_indices": list(active),
        "requested_pairs": [
            list(entry["active_index_pair"]) for entry in pair_entries
        ],
        "requested_gradient_indices": list(gradients),
        "pair_entries": pair_entries,
        "gradient_entries": gradient_entries,
        "metric_pair_count_acquired": int(len(pair_entries)),
        "hessian_pair_count_acquired": int(len(pair_entries)),
        "descent_gradient_component_count_acquired": int(
            len(gradient_entries)
        ),
        "state_reconstruction_delta_norm": float(state_delta_norm),
        "state_consistency_tolerance": float(tolerance),
        "placeholder_entries_serialized": False,
    }


_JOINT_PAIR_GEOMETRY_CACHE_KEY = "__route_a_joint_pair_geometry_cache_v1__"
_JOINT_PAIR_WORKERS_ENV = "STATIC_ADAPT_JOINT_PAIR_WORKERS"
_JOINT_PAIR_CACHE_MAX_ENTRIES_ENV = (
    "STATIC_ADAPT_JOINT_PAIR_CACHE_MAX_ENTRIES"
)
_JOINT_PAIR_CACHE_REGISTRY_LOCK = threading.Lock()


@dataclass
class _JointPairGeometryCache:
    max_entries: int
    values: OrderedDict[str, tuple[float, float, float]] = field(
        default_factory=OrderedDict
    )
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, key: str) -> tuple[float, float, float] | None:
        if int(self.max_entries) <= 0:
            return None
        with self.lock:
            value = self.values.get(str(key))
            if value is None:
                return None
            self.values.move_to_end(str(key))
            return tuple(float(component) for component in value)

    def put(self, key: str, value: tuple[float, float, float]) -> None:
        if int(self.max_entries) <= 0:
            return
        with self.lock:
            self.values[str(key)] = tuple(
                float(component) for component in value
            )
            self.values.move_to_end(str(key))
            while len(self.values) > int(self.max_entries):
                self.values.popitem(last=False)


def _joint_pair_cache_max_entries() -> int:
    raw = os.environ.get(_JOINT_PAIR_CACHE_MAX_ENTRIES_ENV, "4096")
    try:
        return int(max(0, int(raw)))
    except (TypeError, ValueError):
        return 4096


def _joint_pair_geometry_cache(
    pauli_action_cache: dict[str, Any] | None,
) -> _JointPairGeometryCache:
    if pauli_action_cache is None:
        return _JointPairGeometryCache(max_entries=0)
    with _JOINT_PAIR_CACHE_REGISTRY_LOCK:
        existing = pauli_action_cache.get(_JOINT_PAIR_GEOMETRY_CACHE_KEY)
        if isinstance(existing, _JointPairGeometryCache):
            return existing
        created = _JointPairGeometryCache(
            max_entries=_joint_pair_cache_max_entries()
        )
        pauli_action_cache[_JOINT_PAIR_GEOMETRY_CACHE_KEY] = created
        return created


def _resolve_joint_pair_workers(pair_count: int) -> tuple[int, int]:
    raw = os.environ.get(_JOINT_PAIR_WORKERS_ENV, "1")
    try:
        requested = int(max(1, int(raw)))
    except (TypeError, ValueError):
        requested = 1
    effective = int(min(requested, max(1, int(pair_count))))
    return int(requested), int(effective)


def _compiled_polynomial_fingerprint(
    h_compiled: CompiledPolynomialAction,
) -> str:
    payload = [
        {
            "coeff_re": float(complex(term.coeff).real),
            "coeff_im": float(complex(term.coeff).imag),
            "action": (
                None if term.action is None else str(term.action.label_exyz)
            ),
        }
        for term in h_compiled.terms
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _joint_pair_geometry_context_fingerprint(
    *,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
) -> str:
    payload = {
        "schema": "route_a_joint_pair_geometry_context_v1",
        "state_fingerprint": _array_fingerprint(psi_state),
        "reference_fingerprint": _array_fingerprint(psi_ref),
        "ordered_scaffold_fingerprint": _ordered_scaffold_fingerprint(
            selected_ops
        ),
        "theta_fingerprint": _array_fingerprint(theta),
        "hamiltonian_fingerprint": _compiled_polynomial_fingerprint(h_compiled),
        "derivative_convention": (
            "compiled_ansatz_exact_parameter_derivatives_v1"
        ),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _joint_pair_geometry_cache_key(
    *,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    left_record: Mapping[str, Any],
    right_record: Mapping[str, Any],
    h_compiled: CompiledPolynomialAction,
    state_consistency_tolerance: float,
    context_fingerprint: str | None = None,
) -> str:
    payload = {
        "schema": "route_a_joint_pair_geometry_cache_key_v1",
        "context_fingerprint": str(
            context_fingerprint
            or _joint_pair_geometry_context_fingerprint(
                selected_ops=selected_ops,
                theta=theta,
                psi_ref=psi_ref,
                psi_state=psi_state,
                h_compiled=h_compiled,
            )
        ),
        "left_coordinate": _candidate_coordinate_fingerprint(
            left_record.get("candidate_term"),
            position_id=int(left_record.get("position_id", -1)),
        ),
        "right_coordinate": _candidate_coordinate_fingerprint(
            right_record.get("candidate_term"),
            position_id=int(right_record.get("position_id", -1)),
        ),
        "state_consistency_tolerance": float(
            max(1e-12, state_consistency_tolerance)
        ),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _measure_joint_candidate_pair_entry(
    *,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    left_record: Mapping[str, Any],
    right_record: Mapping[str, Any],
    h_compiled: CompiledPolynomialAction,
    pauli_action_cache: dict[str, Any] | None,
    state_consistency_tolerance: float,
) -> tuple[float, float, float]:
    """Measure one compatible G_CC/H_CC off-diagonal pair exactly once."""

    selected = list(selected_ops)
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    records = (dict(left_record), dict(right_record))
    positions = tuple(
        int(record.get("position_id", len(selected))) for record in records
    )
    if any(position < 0 or position > len(selected) for position in positions):
        raise ValueError(
            "Joint candidate-pair geometry received an insertion position "
            "outside the current ansatz boundary."
        )
    candidates_by_position: dict[int, list[int]] = {}
    for local_index, position in enumerate(positions):
        candidates_by_position.setdefault(int(position), []).append(
            int(local_index)
        )
    combined_terms: list[Any] = []
    combined_theta_values: list[float] = []
    candidate_combined_indices: dict[int, int] = {}
    for position in range(len(selected) + 1):
        for local_index in candidates_by_position.get(position, []):
            candidate_term = records[int(local_index)].get("candidate_term")
            if candidate_term is None:
                raise ValueError(
                    "Joint candidate-pair geometry requires candidate_term."
                )
            candidate_combined_indices[int(local_index)] = int(
                len(combined_terms)
            )
            combined_terms.append(candidate_term)
            combined_theta_values.append(0.0)
        if position < len(selected):
            combined_terms.append(selected[int(position)])
            combined_theta_values.append(float(theta_vec[int(position)]))
    executor = _executor_for_terms(
        combined_terms,
        pauli_action_cache=pauli_action_cache,
    )
    reconstructed, derivatives, second_derivatives = (
        _propagate_executor_sparse_second_derivatives(
            executor=executor,
            theta=np.asarray(combined_theta_values, dtype=float),
            psi_ref=np.asarray(psi_ref, dtype=complex),
            active_indices=(
                candidate_combined_indices[0],
                candidate_combined_indices[1],
            ),
            second_derivative_pairs=((0, 1),),
        )
    )
    reconstructed_state = np.asarray(reconstructed, dtype=complex).reshape(-1)
    supplied_state = np.asarray(psi_state, dtype=complex).reshape(-1)
    overlap = complex(np.vdot(supplied_state, reconstructed_state))
    phase = overlap / abs(overlap) if abs(overlap) > 0.0 else 1.0 + 0.0j
    state_delta_norm = float(
        np.linalg.norm(reconstructed_state / phase - supplied_state)
    )
    tolerance = float(max(1e-12, state_consistency_tolerance))
    if state_delta_norm > tolerance:
        raise ValueError(
            "Joint candidate-pair geometry reconstructed a state inconsistent "
            "with the supplied branch state: "
            f"delta={state_delta_norm:.6g}, tolerance={tolerance:.6g}."
        )
    tangents = [
        _horizontal_tangent(reconstructed_state, derivative)
        for derivative in derivatives
    ]
    gram_pair = float(np.real(np.vdot(tangents[0], tangents[1])))
    hpsi_state = apply_compiled_polynomial(reconstructed_state, h_compiled)
    h_derivatives = [
        apply_compiled_polynomial(
            np.asarray(derivative, dtype=complex),
            h_compiled,
        )
        for derivative in derivatives
    ]
    h_left_right = _energy_hessian_entry(
        dpsi_left=derivatives[0],
        dpsi_right=derivatives[1],
        d2psi=second_derivatives[(0, 1)],
        hpsi_state=hpsi_state,
        hdpsi_right=h_derivatives[1],
    )
    h_right_left = _energy_hessian_entry(
        dpsi_left=derivatives[1],
        dpsi_right=derivatives[0],
        d2psi=second_derivatives[(0, 1)],
        hpsi_state=hpsi_state,
        hdpsi_right=h_derivatives[0],
    )
    return (
        float(gram_pair),
        float(0.5 * (h_left_right + h_right_left)),
        float(state_delta_norm),
    )


def _joint_workspace_fingerprint(
    *,
    psi_state: np.ndarray,
    theta: np.ndarray,
    context_mode: str,
    geometry_mode: str,
    active_indices: Sequence[int],
    selected_ops: Sequence[Any],
    records: Sequence[Mapping[str, Any]],
    cfg: FullScoreConfig,
    h_compiled: CompiledPolynomialAction,
    matrices: Sequence[np.ndarray],
) -> str:
    fingerprint_hasher = hashlib.sha256()
    fingerprint_hasher.update(np.asarray(psi_state, dtype=complex).tobytes())
    fingerprint_hasher.update(np.asarray(theta, dtype=float).tobytes())
    fingerprint_hasher.update(str(context_mode).encode("utf-8"))
    fingerprint_hasher.update(str(geometry_mode).encode("utf-8"))
    fingerprint_hasher.update(repr(tuple(int(x) for x in active_indices)).encode("utf-8"))
    fingerprint_hasher.update(
        repr(tuple(str(getattr(term, "label", "")) for term in selected_ops)).encode(
            "utf-8"
        )
    )
    fingerprint_hasher.update(
        repr(
            (
                float(getattr(cfg, "batch_metric_regularization", 1e-9)),
                float(getattr(cfg, "batch_energy_regularization", 1e-9)),
                float(getattr(cfg, "batch_rank_rel_tol", 1e-6)),
                float(getattr(cfg, "batch_max_gram_condition_number", 1e12)),
                float(getattr(cfg, "rho", 0.25)),
                str(
                    getattr(
                        cfg,
                        "phase3_candidate_gain_policy",
                        PHASE3_CANDIDATE_GAIN_JOINT_TOTAL_V1,
                    )
                ),
            )
        ).encode("utf-8")
    )
    fingerprint_hasher.update(
        repr(
            tuple(
                (
                    complex(term.coeff),
                    None if term.action is None else str(term.action.label_exyz),
                )
                for term in h_compiled.terms
            )
        ).encode("utf-8")
    )
    fingerprint_hasher.update(
        repr(tuple(_batch_record_identity_key(record) for record in records)).encode(
            "utf-8"
        )
    )
    for matrix in matrices:
        fingerprint_hasher.update(np.asarray(matrix, dtype=float).tobytes())
    return fingerprint_hasher.hexdigest()


def _build_material_window_singleton_workspace(
    record: Mapping[str, Any],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    active_indices: Sequence[int],
    G_AA: np.ndarray,
    H_AA: np.ndarray,
    g_A: np.ndarray,
    G_AC: np.ndarray,
    H_AC: np.ndarray,
    G_CC: float,
    H_CC: float,
    g_C: float,
    material_window_telemetry: Mapping[str, Any],
) -> _BatchFullGeometryWorkspace:
    """Build one exact retained-coordinate workspace without cache repair."""

    copied = (dict(record),)
    selected = list(selected_ops)
    active = tuple(int(value) for value in active_indices)
    if len(set(active)) != len(active):
        raise ValueError("Material-window workspace active indices repeat.")
    if any(value < 0 or value >= len(selected) for value in active):
        raise ValueError("Material-window workspace active index is out of range.")
    dimension = int(len(active))
    metric = np.asarray(G_AA, dtype=float)
    hessian = np.asarray(H_AA, dtype=float)
    gradient = np.asarray(g_A, dtype=float).reshape(-1)
    metric_cross = np.asarray(G_AC, dtype=float).reshape(-1)
    hessian_cross = np.asarray(H_AC, dtype=float).reshape(-1)
    if metric.shape != (dimension, dimension):
        raise ValueError("Material-window Gram block shape mismatch.")
    if hessian.shape != (dimension, dimension):
        raise ValueError("Material-window Hessian block shape mismatch.")
    for name, value in (
        ("g_A", gradient),
        ("G_AC", metric_cross),
        ("H_AC", hessian_cross),
    ):
        if value.shape != (dimension,):
            raise ValueError(f"Material-window {name} shape mismatch.")
    matrices = (
        metric,
        metric_cross.reshape(dimension, 1),
        np.asarray([[float(G_CC)]], dtype=float),
        hessian,
        hessian_cross.reshape(dimension, 1),
        np.asarray([[float(H_CC)]], dtype=float),
        gradient,
        np.asarray([float(g_C)], dtype=float),
    )
    if any(not np.all(np.isfinite(value)) for value in matrices):
        raise ValueError("Material-window workspace contains nonfinite geometry.")
    geometry_mode = normalize_batch_geometry_mode(
        getattr(cfg, "batch_geometry_mode", None)
    )
    workspace_fingerprint = _joint_workspace_fingerprint(
        psi_state=np.asarray(psi_state, dtype=complex),
        theta=np.asarray(theta, dtype=float),
        context_mode=BATCH_JOINT_CONTEXT_ACTIVE_WINDOW_V1,
        geometry_mode=str(geometry_mode),
        active_indices=active,
        selected_ops=selected,
        records=copied,
        cfg=cfg,
        h_compiled=h_compiled,
        matrices=matrices,
    )
    active_triangle = int(dimension * (dimension + 1) // 2)
    reuse_validation = {
        "schema": "phase3_material_window_geometry_validation_v1",
        "status": "validated_exact_material_window",
        "valid_record_count": 1,
        "valid_gradient_record_count": 1,
        "required_element_counts": {
            "G_AA": active_triangle,
            "H_AA": active_triangle,
            "G_AC": dimension,
            "H_AC": dimension,
            "G_CC_diagonal": 1,
            "H_CC_diagonal": 1,
            "G_CC_off_diagonal": 0,
            "H_CC_off_diagonal": 0,
        },
        "reused_element_counts": {},
        "newly_measured_element_counts": {},
        "material_window": dict(material_window_telemetry),
        "workspace_build_mode": "exact_material_window_singleton_v1",
    }
    return _BatchFullGeometryWorkspace(
        records=copied,
        record_index={_batch_record_identity_key(copied[0]): 0},
        ansatz_depth=int(len(selected)),
        active_indices=active,
        active_labels=tuple(
            str(getattr(selected[index], "label", f"theta_{index}"))
            for index in active
        ),
        G_AA=0.5 * (metric + metric.T),
        H_AA=0.5 * (hessian + hessian.T),
        G_AB=metric_cross.reshape(dimension, 1),
        H_AB=hessian_cross.reshape(dimension, 1),
        G_BB=np.asarray([[float(G_CC)]], dtype=float),
        H_BB=np.asarray([[float(H_CC)]], dtype=float),
        g_A=gradient,
        g_B=np.asarray([float(g_C)], dtype=float),
        phase2_reported_g_B=np.asarray(
            [_phase2_reported_candidate_descent_gradient(copied[0], cfg)],
            dtype=float,
        ),
        geometry_mode=str(geometry_mode),
        joint_context_mode=BATCH_JOINT_CONTEXT_ACTIVE_WINDOW_V1,
        workspace_fingerprint=str(workspace_fingerprint),
        state_fingerprint=_array_fingerprint(
            np.asarray(psi_state, dtype=complex)
        ),
        theta_fingerprint=_array_fingerprint(
            np.asarray(theta, dtype=float)
        ),
        ordered_scaffold_fingerprint=_ordered_scaffold_fingerprint(
            selected
        ),
        hamiltonian_fingerprint=_compiled_polynomial_fingerprint(
            h_compiled
        ),
        metric_regularization=float(
            max(0.0, getattr(cfg, "batch_metric_regularization", 1e-9))
        ),
        energy_regularization=float(
            max(0.0, getattr(cfg, "batch_energy_regularization", 1e-9))
        ),
        joint_linear_solve_policy=str(
            getattr(
                cfg,
                "batch_joint_linear_solve_policy",
                JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
            )
        ),
        rank_relative_tolerance=float(max(0.0, cfg.batch_rank_rel_tol)),
        max_gram_condition_number=float(
            max(1.0, cfg.batch_max_gram_condition_number)
        ),
        max_fubini_study_step=float(max(0.0, cfg.rho)),
        state_delta_norm=float(
            material_window_telemetry.get(
                "state_reconstruction_delta_norm", 0.0
            )
        ),
        state_consistency_tolerance=float(
            max(
                0.0,
                getattr(cfg, "batch_state_consistency_tolerance", 1e-8),
            )
        ),
        phase2_reuse_validation=reuse_validation,
        _subset_cache={},
        phase3_candidate_gain_policy=str(
            getattr(
                cfg,
                "phase3_candidate_gain_policy",
                PHASE3_CANDIDATE_GAIN_JOINT_TOTAL_V1,
            )
        ),
    )


def _build_batch_full_geometry_workspace(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    pauli_action_cache: dict[str, Any] | None,
    old_old_geometry_prior: HistoricalSingletonOldOldGeometryPrior | None = None,
    require_complete_upstream_geometry: bool = False,
    joint_pair_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> _BatchFullGeometryWorkspace:
    copied = tuple(dict(record) for record in records)
    candidate_terms = [record.get("candidate_term") for record in copied]
    if any(term is None for term in candidate_terms):
        raise ValueError("Full batch geometry requires candidate_term on every record.")
    theta_vec = np.asarray(theta, dtype=float).reshape(-1)
    selected = list(selected_ops)
    context_mode = normalize_batch_joint_context_mode(
        getattr(cfg, "batch_joint_context_mode", None)
    )
    active_indices = list(
        _batch_joint_active_indices(cfg, selected_ops=selected)
    )
    active_labels = [
        str(getattr(selected[index], "label", f"theta_{index}"))
        for index in active_indices
    ]
    state_consistency_tolerance = float(
        max(
            0.0,
            getattr(cfg, "batch_state_consistency_tolerance", 1e-8),
        )
    )
    active_count = int(len(active_indices))
    validated_old_old_prior = (
        None
        if old_old_geometry_prior is None
        else _validate_historical_singleton_old_old_geometry_prior(
            old_old_geometry_prior,
            selected_ops=selected,
            active_indices=active_indices,
            theta=theta_vec,
            psi_state=np.asarray(psi_state, dtype=complex),
            h_compiled=h_compiled,
        )
    )
    candidate_count = int(len(copied))
    candidate_positions = [
        int(record.get("position_id", len(selected))) for record in copied
    ]
    if any(position < 0 or position > len(selected) for position in candidate_positions):
        raise ValueError(
            "Full batch geometry received a candidate insertion position outside "
            "the current ansatz boundary."
        )
    required_candidate_pairs = _required_joint_candidate_pairs(
        copied,
        batch_size_cap=int(max(1, getattr(cfg, "batch_size_cap", 1))),
    )
    phase2_cache = _build_phase2_joint_geometry_cache(
        copied,
        active_indices=active_indices,
        selected_ops=selected,
        theta=theta_vec,
        psi_state=np.asarray(psi_state, dtype=complex),
        h_compiled=h_compiled,
        tolerance=max(state_consistency_tolerance, 1e-8),
        old_old_geometry_prior=validated_old_old_prior,
    )
    if bool(require_complete_upstream_geometry):
        if required_candidate_pairs or not phase2_cache.complete:
            invalid_records = [
                dict(result)
                for result in phase2_cache.record_results
                if not bool(result.get("valid", False))
                or not bool(result.get("gradient_valid", False))
            ]
            raise RuntimeError(
                "The projected Phase-III support projection requires a "
                "complete same-iteration upstream geometry population before "
                "workspace construction; refusing lazy measurement repair: "
                f"invalid_records={len(invalid_records)}, "
                f"required_candidate_pairs={len(required_candidate_pairs)}."
            )
    geometry_mode = normalize_batch_geometry_mode(
        getattr(cfg, "batch_geometry_mode", None)
    )
    phase2_reported_g_B = np.asarray(
        [
            _phase2_reported_candidate_descent_gradient(record, cfg)
            for record in copied
        ],
        dtype=float,
    )
    valid_geometry_indices = set(
        int(index) for index in phase2_cache.valid_record_indices
    )
    valid_gradient_indices = set(
        int(index) for index in phase2_cache.valid_gradient_record_indices
    )
    active_block_initialized = bool(phase2_cache.active_block_valid)
    G_AA = np.asarray(phase2_cache.G_AA, dtype=float).copy()
    H_AA = np.asarray(phase2_cache.H_AA, dtype=float).copy()
    g_A = np.asarray(phase2_cache.g_A, dtype=float).copy()
    G_AB = np.zeros((active_count, candidate_count), dtype=float)
    H_AB = np.zeros((active_count, candidate_count), dtype=float)
    G_BB = np.zeros((candidate_count, candidate_count), dtype=float)
    H_BB = np.zeros((candidate_count, candidate_count), dtype=float)
    g_B = np.zeros(candidate_count, dtype=float)
    selector_context: _ScaffoldDerivativeContext | None = None
    state_delta_norm = float(
        phase2_cache.state_reconstruction_delta_norm_max
    )
    for candidate_index, record in enumerate(copied):
        geometry_reused = int(candidate_index) in valid_geometry_indices
        gradient_reused = int(candidate_index) in valid_gradient_indices
        repair_payload: Mapping[str, Any] | None = None
        if geometry_reused:
            G_AB[:, candidate_index] = np.asarray(
                phase2_cache.G_AB[:, candidate_index], dtype=float
            )
            H_AB[:, candidate_index] = np.asarray(
                phase2_cache.H_AB[:, candidate_index], dtype=float
            )
            G_BB[candidate_index, candidate_index] = float(
                phase2_cache.G_BB_diagonal[candidate_index]
            )
            H_BB[candidate_index, candidate_index] = float(
                phase2_cache.H_BB_diagonal[candidate_index]
            )
        if gradient_reused:
            g_B[candidate_index] = float(phase2_cache.g_B[candidate_index])
        if not geometry_reused or not gradient_reused:
            if selector_context is None:
                selector_context = _selector_scaffold_context(
                    selected_ops=selected,
                    theta=theta_vec,
                    psi_ref=np.asarray(psi_ref, dtype=complex),
                    psi_state=np.asarray(psi_state, dtype=complex),
                    active_indices=active_indices,
                    h_compiled=h_compiled,
                    measure_old_old_geometry=(
                        validated_old_old_prior is None
                    ),
                )
            repair_payload = _exact_insertion_joint_geometry_payload(
                scaffold_context=selector_context,
                candidate_term=candidate_terms[candidate_index],
                position_id=candidate_positions[candidate_index],
                h_compiled=h_compiled,
                pauli_action_cache=pauli_action_cache,
                state_consistency_tolerance=max(
                    state_consistency_tolerance,
                    1e-12,
                ),
                old_old_geometry_prior=validated_old_old_prior,
            )
            state_delta_norm = max(
                state_delta_norm,
                float(
                    repair_payload.get(
                        "state_reconstruction_delta_norm",
                        0.0,
                    )
                ),
            )
            repaired_G_AA = np.asarray(
                repair_payload.get("G_AA", []), dtype=float
            ).reshape(active_count, active_count)
            repaired_H_AA = np.asarray(
                repair_payload.get("H_AA", []), dtype=float
            ).reshape(active_count, active_count)
            repaired_g_A = np.asarray(
                repair_payload.get("g_A", []), dtype=float
            ).reshape(active_count)
            if not active_block_initialized:
                G_AA = repaired_G_AA
                H_AA = repaired_H_AA
                g_A = repaired_g_A
                active_block_initialized = True
            elif not (
                np.allclose(
                    G_AA,
                    repaired_G_AA,
                    atol=max(state_consistency_tolerance, 1e-10),
                    rtol=1e-7,
                )
                and np.allclose(
                    H_AA,
                    repaired_H_AA,
                    atol=max(state_consistency_tolerance, 1e-10),
                    rtol=1e-7,
                )
                and np.allclose(
                    g_A,
                    repaired_g_A,
                    atol=max(state_consistency_tolerance, 1e-10),
                    rtol=1e-7,
                )
            ):
                raise ValueError(
                    "Selector geometry repair produced an active block "
                    "inconsistent with the state-scoped Phase-2 cache."
                )
        if not geometry_reused:
            assert repair_payload is not None
            G_AB[:, candidate_index] = np.asarray(
                repair_payload.get("G_AB", []), dtype=float
            ).reshape(active_count)
            H_AB[:, candidate_index] = np.asarray(
                repair_payload.get("H_AB", []), dtype=float
            ).reshape(active_count)
            G_BB[candidate_index, candidate_index] = float(
                repair_payload.get("G_BB", float("nan"))
            )
            H_BB[candidate_index, candidate_index] = float(
                repair_payload.get("H_BB", float("nan"))
            )
        if not gradient_reused:
            assert repair_payload is not None
            g_B[candidate_index] = float(
                repair_payload.get("descent_gradient", float("nan"))
            )

    pair_geometry_cache = _joint_pair_geometry_cache(pauli_action_cache)
    pair_workers_requested, _pair_workers_initial = _resolve_joint_pair_workers(
        len(required_candidate_pairs)
    )
    pair_context_fingerprint = (
        _joint_pair_geometry_context_fingerprint(
            selected_ops=selected,
            theta=theta_vec,
            psi_ref=np.asarray(psi_ref, dtype=complex),
            psi_state=np.asarray(psi_state, dtype=complex),
            h_compiled=h_compiled,
        )
        if required_candidate_pairs
        else ""
    )
    pair_thread_local = threading.local()

    pair_requests: list[tuple[tuple[int, int], str]] = []
    pair_cache_values: dict[
        str,
        tuple[float, float, float] | None,
    ] = {}
    unique_miss_requests: list[tuple[str, tuple[int, int]]] = []
    for raw_pair in required_candidate_pairs:
        pair = (int(raw_pair[0]), int(raw_pair[1]))
        cache_key = _joint_pair_geometry_cache_key(
            selected_ops=selected,
            theta=theta_vec,
            psi_ref=np.asarray(psi_ref, dtype=complex),
            psi_state=np.asarray(psi_state, dtype=complex),
            left_record=copied[pair[0]],
            right_record=copied[pair[1]],
            h_compiled=h_compiled,
            state_consistency_tolerance=max(
                state_consistency_tolerance,
                1e-12,
            ),
            context_fingerprint=pair_context_fingerprint,
        )
        if cache_key not in pair_cache_values:
            cached = pair_geometry_cache.get(cache_key)
            pair_cache_values[cache_key] = cached
            if cached is None:
                unique_miss_requests.append((cache_key, pair))
        pair_requests.append((pair, cache_key))

    pair_workers_effective = int(
        min(
            int(pair_workers_requested),
            max(1, int(len(unique_miss_requests))),
        )
    )

    def _measure_required_pair(
        request: tuple[str, tuple[int, int]],
    ) -> tuple[str, tuple[float, float, float]]:
        cache_key, pair = request
        left, right = pair
        if pair_workers_effective > 1:
            local_action_cache = getattr(
                pair_thread_local,
                "pauli_action_cache",
                None,
            )
            if local_action_cache is None:
                local_action_cache = {}
                pair_thread_local.pauli_action_cache = local_action_cache
        else:
            local_action_cache = pauli_action_cache
        gram_entry, hessian_entry, pair_state_delta = (
            _measure_joint_candidate_pair_entry(
                selected_ops=selected,
                theta=theta_vec,
                psi_ref=np.asarray(psi_ref, dtype=complex),
                psi_state=np.asarray(psi_state, dtype=complex),
                left_record=copied[left],
                right_record=copied[right],
                h_compiled=h_compiled,
                pauli_action_cache=local_action_cache,
                state_consistency_tolerance=max(
                    state_consistency_tolerance,
                    1e-12,
                ),
            )
        )
        return (
            str(cache_key),
            (
                float(gram_entry),
                float(hessian_entry),
                float(pair_state_delta),
            ),
        )

    if unique_miss_requests and pair_workers_effective > 1:
        with ThreadPoolExecutor(
            max_workers=pair_workers_effective,
            thread_name_prefix="route-a-joint-pair",
        ) as pair_pool:
            measured_pair_results = list(
                pair_pool.map(
                    _measure_required_pair,
                    unique_miss_requests,
                )
            )
    else:
        measured_pair_results = [
            _measure_required_pair(request)
            for request in unique_miss_requests
        ]
    measured_by_key = dict(measured_pair_results)
    for cache_key, measured in measured_pair_results:
        pair_geometry_cache.put(cache_key, measured)

    charged_miss_keys: set[str] = set()
    pair_results: list[tuple[int, int, float, float, float, bool]] = []
    for pair, cache_key in pair_requests:
        cached = pair_cache_values[cache_key]
        cache_hit = cached is not None or cache_key in charged_miss_keys
        measured = cached if cached is not None else measured_by_key[cache_key]
        if cached is None:
            charged_miss_keys.add(cache_key)
        pair_results.append(
            (
                int(pair[0]),
                int(pair[1]),
                float(measured[0]),
                float(measured[1]),
                float(measured[2]),
                bool(cache_hit),
            )
        )
    pair_cache_miss_count = int(len(unique_miss_requests))
    pair_cache_hit_count = int(len(pair_results) - pair_cache_miss_count)
    joint_pair_receipts: list[dict[str, Any]] = []
    for (pair, cache_key), result in zip(
        pair_requests,
        pair_results,
        strict=True,
    ):
        joint_pair_receipts.append(
            {
                "left_workspace_index": int(pair[0]),
                "right_workspace_index": int(pair[1]),
                "cache_key": str(cache_key),
                "cache_hit": bool(result[5]),
            }
        )
    for (
        left,
        right,
        gram_entry,
        hessian_entry,
        pair_state_delta,
        _cache_hit,
    ) in pair_results:
        G_BB[left, right] = float(gram_entry)
        G_BB[right, left] = float(gram_entry)
        H_BB[left, right] = float(hessian_entry)
        H_BB[right, left] = float(hessian_entry)
        state_delta_norm = max(state_delta_norm, float(pair_state_delta))
        if joint_pair_observer is not None:
            matching_receipt = next(
                receipt
                for receipt in joint_pair_receipts
                if int(receipt["left_workspace_index"]) == int(left)
                and int(receipt["right_workspace_index"]) == int(right)
            )
            joint_pair_observer(
                {
                    **dict(matching_receipt),
                    "left_record": dict(copied[left]),
                    "right_record": dict(copied[right]),
                    "gram_entry": float(gram_entry),
                    "hessian_entry": float(hessian_entry),
                    "state_reconstruction_delta_norm": float(
                        pair_state_delta
                    ),
                    "physical_evaluation_performed": bool(
                        not matching_receipt["cache_hit"]
                    ),
                }
            )

    for name, block in (
        ("G_AA", G_AA),
        ("H_AA", H_AA),
        ("g_A", g_A),
        ("G_AB", G_AB),
        ("H_AB", H_AB),
        ("G_BB", G_BB),
        ("H_BB", H_BB),
        ("g_B", g_B),
    ):
        if not np.all(np.isfinite(np.asarray(block))):
            raise ValueError(
                f"State-scoped joint geometry workspace contains nonfinite {name}."
            )
    psi_current = np.asarray(psi_state, dtype=complex).reshape(-1)
    if validated_old_old_prior is not None and phase2_cache.complete:
        workspace_build_mode = (
            "outer_information_prior_plus_phase2_candidate_blocks_v1"
        )
    elif validated_old_old_prior is not None:
        workspace_build_mode = (
            "outer_information_prior_plus_exact_candidate_repairs_v1"
        )
    elif phase2_cache.complete and not required_candidate_pairs:
        workspace_build_mode = "lazy_phase2_blocks_no_candidate_pairs_v1"
    elif phase2_cache.complete:
        workspace_build_mode = "phase2_reuse_plus_required_candidate_pairs_v1"
    else:
        workspace_build_mode = "phase2_partial_reuse_with_lazy_repairs_v1"

    if geometry_mode == BATCH_GEOMETRY_DIAGONAL_HESSIAN_DIAGNOSTIC_V1:
        H_BB = np.diag(np.diag(H_BB))
    required_pair_set = {
        (int(left), int(right))
        for left, right in required_candidate_pairs
    }
    for left in range(candidate_count):
        for right in range(left + 1, candidate_count):
            if (int(left), int(right)) in required_pair_set:
                continue
            G_BB[left, right] = G_BB[right, left] = 0.0
            H_BB[left, right] = H_BB[right, left] = 0.0

    phase2_reuse_validation = _phase2_joint_geometry_reuse_accounting(
        phase2_cache,
        candidate_count=candidate_count,
        required_candidate_pairs=required_candidate_pairs,
        reused_candidate_pair_count=pair_cache_hit_count,
        tolerance=max(state_consistency_tolerance, 1e-8),
        eager_blocks=None,
        old_old_geometry_prior=validated_old_old_prior,
    )
    phase2_reuse_validation["workspace_build_mode"] = str(
        workspace_build_mode
    )
    phase2_reuse_validation.update(
        {
            "joint_pair_cache_scope": (
                "state_scaffold_hamiltonian_fingerprinted_lru_v1"
                if pair_geometry_cache.max_entries > 0
                else "disabled_v1"
            ),
            "joint_pair_cache_max_entries": int(
                pair_geometry_cache.max_entries
            ),
            "joint_pair_cache_hit_count": int(pair_cache_hit_count),
            "joint_pair_cache_miss_count": int(pair_cache_miss_count),
            "joint_pair_workers_requested": int(pair_workers_requested),
            "joint_pair_workers_effective": int(pair_workers_effective),
            "joint_pair_parallel_enabled": bool(
                required_candidate_pairs and pair_workers_effective > 1
            ),
            "joint_pair_result_order": (
                "deterministic_lookup_measure_commit_order_v1"
            ),
            "joint_pair_receipts": joint_pair_receipts,
        }
    )
    workspace_fingerprint = _joint_workspace_fingerprint(
        psi_state=np.asarray(psi_current, dtype=complex),
        theta=theta_vec,
        context_mode=str(context_mode),
        geometry_mode=str(geometry_mode),
        active_indices=active_indices,
        selected_ops=selected,
        records=copied,
        cfg=cfg,
        h_compiled=h_compiled,
        matrices=(G_AA, G_AB, G_BB, H_AA, H_AB, H_BB, g_A, g_B),
    )
    workspace = _BatchFullGeometryWorkspace(
        records=copied,
        record_index={
            _batch_record_identity_key(record): int(index)
            for index, record in enumerate(copied)
        },
        ansatz_depth=int(len(selected)),
        active_indices=tuple(int(index) for index in active_indices),
        active_labels=tuple(str(label) for label in active_labels),
        G_AA=0.5 * (G_AA + G_AA.T),
        H_AA=0.5 * (H_AA + H_AA.T),
        G_AB=np.asarray(G_AB, dtype=float),
        H_AB=np.asarray(H_AB, dtype=float),
        G_BB=0.5 * (G_BB + G_BB.T),
        H_BB=0.5 * (H_BB + H_BB.T),
        g_A=np.asarray(g_A, dtype=float),
        g_B=np.asarray(g_B, dtype=float),
        phase2_reported_g_B=np.asarray(phase2_reported_g_B, dtype=float),
        geometry_mode=str(geometry_mode),
        joint_context_mode=str(context_mode),
        workspace_fingerprint=str(workspace_fingerprint),
        state_fingerprint=_array_fingerprint(psi_current),
        theta_fingerprint=_array_fingerprint(theta_vec),
        ordered_scaffold_fingerprint=_ordered_scaffold_fingerprint(
            selected
        ),
        hamiltonian_fingerprint=_compiled_polynomial_fingerprint(
            h_compiled
        ),
        metric_regularization=float(
            max(0.0, getattr(cfg, "batch_metric_regularization", 1e-9))
        ),
        energy_regularization=float(
            max(0.0, getattr(cfg, "batch_energy_regularization", 1e-9))
        ),
        joint_linear_solve_policy=str(
            getattr(
                cfg,
                "batch_joint_linear_solve_policy",
                JOINT_LINEAR_SOLVE_BLOCK_PINV_LEGACY_V1,
            )
        ),
        rank_relative_tolerance=float(max(0.0, cfg.batch_rank_rel_tol)),
        max_gram_condition_number=float(
            max(1.0, cfg.batch_max_gram_condition_number)
        ),
        max_fubini_study_step=float(max(0.0, cfg.rho)),
        state_delta_norm=float(state_delta_norm),
        state_consistency_tolerance=float(state_consistency_tolerance),
        phase2_reuse_validation=dict(phase2_reuse_validation),
        _subset_cache={},
        phase3_candidate_gain_policy=str(
            getattr(
                cfg,
                "phase3_candidate_gain_policy",
                PHASE3_CANDIDATE_GAIN_JOINT_TOTAL_V1,
            )
        ),
    )
    return workspace


def _annotate_phase3_batch_records(
    records: Sequence[Mapping[str, Any]],
    *,
    proposal: BatchSelectionProposal,
    mode: str,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    summary = dict(proposal.summary)
    for order, rec in enumerate(records):
        updated = dict(rec)
        feat = updated.get("feature")
        compatibility_payload = {
            "total": float(summary.get("additivity_defect", 0.0)),
            "joint_gain": float(proposal.delta_e3),
            "contextual_single_total": float(summary.get("contextual_single_total", 0.0)),
            "lambda_min": float(summary.get("lambda_min", 0.0)),
            "rank_floor": float(summary.get("rank_floor", 0.0)),
            "mu_tan": float(summary.get("mu_tan", 0.0)),
            "phase3_batch_score": float(proposal.score),
            "phase3_batch_delta_e3": float(proposal.delta_e3),
            "phase3_batch_K3": float(proposal.k3),
            "phase3_batch_denominator_1_plus_K3": float(proposal.denominator_1_plus_k3),
            "phase3_batch_order": int(order),
            "phase3_batch_size": int(len(records)),
            "phase3_batch_selection_mode": str(mode),
        }
        if isinstance(feat, CandidateFeatures):
            updated_feature = _replace_feature(
                feat,
                compatibility_penalty_total=float(summary.get("additivity_defect", 0.0)),
                phase_score_components={
                    **dict(feat.phase_score_components),
                    "phase3_batch_score": float(proposal.score),
                    "phase3_batch_delta_e3": float(proposal.delta_e3),
                    "phase3_batch_K3": float(proposal.k3),
                    "phase3_batch_denominator_1_plus_K3": float(proposal.denominator_1_plus_k3),
                    "phase3_batch_order": int(order),
                    "phase3_batch_size": int(len(records)),
                    "phase3_batch_selection_mode": str(mode),
                },
                phase_cost_components={
                    **dict(feat.phase_cost_components),
                    "phase3_batch_K3": float(proposal.k3),
                    "phase3_batch_denominator_1_plus_K3": float(proposal.denominator_1_plus_k3),
                },
            )
            updated["feature"] = updated_feature
        updated["compatibility_penalty"] = compatibility_payload
        updated["phase3_batch_score"] = float(proposal.score)
        updated["phase3_batch_delta_e3"] = float(proposal.delta_e3)
        updated["phase3_batch_K3"] = float(proposal.k3)
        updated["phase3_batch_denominator_1_plus_K3"] = float(proposal.denominator_1_plus_k3)
        updated["phase3_batch_order"] = int(order)
        updated["phase3_batch_size"] = int(len(records))
        updated["phase3_batch_selection_mode"] = str(mode)
        annotated.append(updated)
    return annotated


def _evaluate_ordered_reduced_plane_batch_proposal(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    novelty_oracle: Any,
    curvature_oracle: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    mode: str,
    rejection_counts: dict[str, int] | None = None,
    additivity_diagnostics: list[dict[str, Any]] | None = None,
    geometry_workspace: _BatchFullGeometryWorkspace | None = None,
) -> BatchSelectionProposal | None:
    def _record_rejection(reason: str) -> None:
        if rejection_counts is not None:
            rejection_counts[str(reason)] = int(
                rejection_counts.get(str(reason), 0) + 1
            )

    if not records:
        _record_rejection("empty_subset")
        return None
    identities = [_batch_record_generator_identity(record) for record in records]
    if len(set(identities)) != len(identities):
        _record_rejection("duplicate_exact_child_identity")
        return None
    summary = (
        geometry_workspace.summary_for_records(records)
        if geometry_workspace is not None
        else _batch_geometry_summary(
            records,
            cfg=cfg,
            selected_ops=selected_ops,
            theta=theta,
            psi_ref=psi_ref,
            psi_state=psi_state,
            h_compiled=h_compiled,
            novelty_oracle=novelty_oracle,
            curvature_oracle=curvature_oracle,
            compiled_cache=compiled_cache,
            pauli_action_cache=pauli_action_cache,
        )
    )
    if not bool(summary.get("feasible", False)):
        _record_rejection(str(summary.get("reason", "infeasible")))
        if additivity_diagnostics is not None:
            additivity_diagnostics.append(
                {
                    "cardinality": int(len(records)),
                    "eligible": False,
                    "reason": str(summary.get("reason", "infeasible")),
                    "defect": float(summary.get("additivity_defect", 0.0)),
                    "joint_gain": float(summary.get("joint_gain", 0.0)),
                    "singleton_gain_sum": float(
                        summary.get("contextual_single_total", 0.0)
                    ),
                    "subset_workspace_indices": [
                        int(value)
                        for value in summary.get("subset_workspace_indices", [])
                    ],
                    "gram_eigenvalues": [
                        float(value)
                        for value in summary.get("gram_eigenvalues", [])
                    ],
                    "effective_rank": (
                        None
                        if summary.get("effective_rank") is None
                        else int(summary.get("effective_rank"))
                    ),
                    "rank_floor": (
                        None
                        if summary.get("rank_floor") is None
                        else float(summary.get("rank_floor"))
                    ),
                    "raw_gram_scale": (
                        None
                        if summary.get("raw_gram_scale") is None
                        else float(summary.get("raw_gram_scale"))
                    ),
                    "effective_gram_scale": (
                        None
                        if summary.get("effective_gram_scale") is None
                        else float(summary.get("effective_gram_scale"))
                    ),
                    "residual_to_raw_scale_ratio": (
                        None
                        if summary.get("residual_to_raw_scale_ratio") is None
                        else float(summary.get("residual_to_raw_scale_ratio"))
                    ),
                }
            )
        return None
    if geometry_workspace is not None:
        workspace_indices = tuple(
            int(value) for value in summary.get("subset_workspace_indices", [])
        )
        if len(workspace_indices) == len(records):
            records = tuple(
                dict(geometry_workspace.records[index])
                for index in workspace_indices
            )
    delta_e3 = float(max(0.0, float(summary.get("joint_gain", 0.0))))
    k3 = _batch_cost_excess(records, cfg=cfg)
    denominator = float(max(float(cfg.cheap_score_eps), 1.0 + float(k3)))
    base_score = float(delta_e3 / denominator)
    additivity_policy = normalize_batch_additivity_policy(
        getattr(cfg, "batch_additivity_policy", BATCH_ADDITIVITY_HARD_GATE_LEGACY_V1)
    )
    additivity_defect = (
        0.0
        if len(records) == 1
        else float(max(0.0, summary.get("additivity_defect", 0.0)))
    )
    if (
        additivity_policy == BATCH_ADDITIVITY_HARD_GATE_LEGACY_V1
        and additivity_defect > float(max(0.0, cfg.batch_additivity_tol))
    ):
        _record_rejection("additivity_hard_gate_legacy")
        if additivity_diagnostics is not None:
            additivity_diagnostics.append(
                {
                    "cardinality": int(len(records)),
                    "eligible": False,
                    "reason": "additivity_hard_gate_legacy",
                    "defect": float(additivity_defect),
                    "joint_gain": float(delta_e3),
                    "singleton_gain_sum": float(
                        summary.get("contextual_single_total", delta_e3)
                    ),
                }
            )
        return None
    lambda_add = float(max(0.0, getattr(cfg, "batch_additivity_lambda", 0.0)))
    additivity_penalty_denominator = (
        float(1.0 + lambda_add * additivity_defect)
        if additivity_policy == BATCH_ADDITIVITY_SOFT_PENALTY_V1
        else 1.0
    )
    score = float(base_score / additivity_penalty_denominator)
    summary_out = {
        "selection_mode": str(mode),
        **dict(summary),
        "selected": bool(len(records) > 1),
        "selected_count": int(len(records)),
        "selected_labels": [_batch_record_label(record) for record in records],
        "phase3_batch_score": float(score),
        "phase3_batch_delta_e3": float(delta_e3),
        "phase3_batch_K3": float(k3),
        "phase3_batch_denominator_1_plus_K3": float(denominator),
        "base_score": float(base_score),
        "score": float(score),
        "additivity_policy": str(additivity_policy),
        "lambda_add": float(lambda_add),
        "additivity_defect": float(additivity_defect),
        "additivity_penalty_denominator": float(additivity_penalty_denominator),
        "effective_batch_size_cap": int(_ordered_reduced_plane_batch_limits(cfg)[1]),
        "effective_batch_target_size": int(_ordered_reduced_plane_batch_limits(cfg)[0]),
        "same_generator_batch_duplicate_policy": "block_generator_identity_v1",
    }
    if additivity_diagnostics is not None:
        additivity_diagnostics.append(
            {
                "cardinality": int(len(records)),
                "eligible": True,
                "defect": float(additivity_defect),
                "joint_gain": float(delta_e3),
                "singleton_gain_sum": float(
                    summary.get("contextual_single_total", delta_e3)
                ),
                "base_score": float(base_score),
                "score": float(score),
            }
        )
    return BatchSelectionProposal(
        records=tuple(dict(record) for record in records),
        summary=summary_out,
        score=float(score),
        delta_e3=float(delta_e3),
        k3=float(k3),
        denominator_1_plus_k3=float(denominator),
    )


def _fallback_singleton_batch_proposal(
    record: Mapping[str, Any],
    *,
    cfg: FullScoreConfig,
    mode: str,
    reason: str,
) -> BatchSelectionProposal:
    delta_e3 = float(max(0.0, _batch_record_score(record, "full_v2_score", 0.0)))
    k3 = _phase3_record_k3(record)
    denominator = float(max(float(cfg.cheap_score_eps), 1.0 + float(k3)))
    score = float(delta_e3 / denominator)
    summary = {
        "selection_mode": str(mode),
        "selected": False,
        "reason": str(reason),
        "feasible": True,
        "joint_gain": float(delta_e3),
        "contextual_single_total": float(delta_e3),
        "additivity_defect": 0.0,
        "selected_count": 1,
        "selected_labels": [_batch_record_label(record)],
        "phase3_batch_score": float(score),
        "phase3_batch_delta_e3": float(delta_e3),
        "phase3_batch_K3": float(k3),
        "phase3_batch_denominator_1_plus_K3": float(denominator),
        "effective_batch_size_cap": int(_ordered_reduced_plane_batch_limits(cfg)[1]),
        "effective_batch_target_size": int(_ordered_reduced_plane_batch_limits(cfg)[0]),
    }
    proposal = BatchSelectionProposal(
        records=(dict(record),),
        summary=summary,
        score=float(score),
        delta_e3=float(delta_e3),
        k3=float(k3),
        denominator_1_plus_k3=float(denominator),
    )
    return BatchSelectionProposal(
        records=tuple(_annotate_phase3_batch_records(proposal.records, proposal=proposal, mode=mode)),
        summary=dict(proposal.summary),
        score=float(proposal.score),
        delta_e3=float(proposal.delta_e3),
        k3=float(proposal.k3),
        denominator_1_plus_k3=float(proposal.denominator_1_plus_k3),
    )


def _dedupe_batch_proposals(
    proposals: Sequence[BatchSelectionProposal],
    *,
    cfg: FullScoreConfig,
) -> list[BatchSelectionProposal]:
    keep: dict[tuple[tuple[int, int, str], ...], BatchSelectionProposal] = {}
    for proposal in proposals:
        key = tuple(_batch_record_identity_key(record) for record in proposal.records)
        incumbent = keep.get(key)
        if incumbent is None or _batch_proposal_better(
            proposal,
            incumbent,
            cfg=cfg,
        ):
            keep[key] = proposal
    return _sort_batch_proposals(list(keep.values()), cfg=cfg)


def greedy_reduced_plane_batch_proposals(
    ranked_records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    novelty_oracle: Any,
    curvature_oracle: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    tie_break_score_key: str = "phase2_raw_score",
    max_proposals: int = 1,
    joint_pair_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[list[BatchSelectionProposal], dict[str, Any]]:
    mode = PHASE2_BATCH_GREEDY_REDUCED_PLANE
    ranked, shell, _top_score = _batch_search_population(
        ranked_records,
        cfg=cfg,
        tie_break_score_key=tie_break_score_key,
    )
    target_count, cap_count = _ordered_reduced_plane_batch_limits(cfg)
    if not ranked:
        return [], {
            "selection_mode": mode,
            "selected": False,
            "reason": "empty_shortlist",
            "shell_size": 0,
            "candidate_batch_eval_count": 0,
            "effective_batch_size_cap": int(cap_count),
            "effective_batch_target_size": int(target_count),
        }
    if not shell:
        proposal = _fallback_singleton_batch_proposal(
            ranked[0],
            cfg=cfg,
            mode=mode,
            reason="nonpositive_shell",
        )
        return [proposal], {"selection_mode": mode, **dict(proposal.summary), "shell_size": 0, "candidate_batch_eval_count": 0}

    proposal_count_cap = int(max(1, max_proposals))
    seed_count = int(
        len(shell)
        if _canonical_ranked_child_batch_search(cfg)
        else min(len(shell), max(proposal_count_cap, cap_count))
    )
    geometry_mode = normalize_batch_geometry_mode(
        getattr(cfg, "batch_geometry_mode", None)
    )
    geometry_workspace = (
        None
        if geometry_mode
        == BATCH_GEOMETRY_PER_SUBSET_DIAGONAL_HESSIAN_LEGACY_V1
        else _build_batch_full_geometry_workspace(
            shell,
            cfg=cfg,
            selected_ops=selected_ops,
            theta=theta,
            psi_ref=psi_ref,
            psi_state=psi_state,
            h_compiled=h_compiled,
            pauli_action_cache=pauli_action_cache,
            joint_pair_observer=joint_pair_observer,
        )
    )
    proposals: list[BatchSelectionProposal] = []
    candidate_batch_eval_count = 0
    subset_counts_considered: dict[int, int] = {}
    subset_counts_evaluated: dict[int, int] = {}
    subset_counts_feasible: dict[int, int] = {}
    rejection_counts: dict[str, int] = {}
    additivity_diagnostics: list[dict[str, Any]] = []
    duplicate_generator_skip_count = 0
    duplicate_generator_identities: set[tuple[str, str]] = set()
    for seed in shell[:seed_count]:
        batch = [dict(seed)]
        subset_counts_considered[1] = int(subset_counts_considered.get(1, 0) + 1)
        subset_counts_evaluated[1] = int(subset_counts_evaluated.get(1, 0) + 1)
        current = _evaluate_ordered_reduced_plane_batch_proposal(
            batch,
            cfg=cfg,
            selected_ops=selected_ops,
            theta=theta,
            psi_ref=psi_ref,
            psi_state=psi_state,
            h_compiled=h_compiled,
            novelty_oracle=novelty_oracle,
            curvature_oracle=curvature_oracle,
            compiled_cache=compiled_cache,
            pauli_action_cache=pauli_action_cache,
            mode=mode,
            rejection_counts=rejection_counts,
            additivity_diagnostics=additivity_diagnostics,
            geometry_workspace=geometry_workspace,
        )
        candidate_batch_eval_count += 1
        if current is None:
            continue
        subset_counts_feasible[1] = int(subset_counts_feasible.get(1, 0) + 1)
        greedy_size_limit = (
            int(cap_count)
            if _canonical_ranked_child_batch_search(cfg)
            else int(target_count)
        )
        while len(batch) < greedy_size_limit and len(batch) < cap_count:
            best_trial: BatchSelectionProposal | None = None
            best_record: dict[str, Any] | None = None
            batch_keys = {_batch_record_generator_identity(rec) for rec in batch}
            for rec in shell:
                rec_identity = _batch_record_generator_identity(rec)
                if rec_identity in batch_keys:
                    if _batch_record_identity_key(rec) not in {_batch_record_identity_key(x) for x in batch}:
                        duplicate_generator_skip_count += 1
                        duplicate_generator_identities.add(rec_identity)
                    continue
                trial_batch = [dict(x) for x in batch] + [dict(rec)]
                subset_size = int(len(trial_batch))
                subset_counts_considered[subset_size] = int(
                    subset_counts_considered.get(subset_size, 0) + 1
                )
                subset_counts_evaluated[subset_size] = int(
                    subset_counts_evaluated.get(subset_size, 0) + 1
                )
                trial = _evaluate_ordered_reduced_plane_batch_proposal(
                    trial_batch,
                    cfg=cfg,
                    selected_ops=selected_ops,
                    theta=theta,
                    psi_ref=psi_ref,
                    psi_state=psi_state,
                    h_compiled=h_compiled,
                    novelty_oracle=novelty_oracle,
                    curvature_oracle=curvature_oracle,
                    compiled_cache=compiled_cache,
                    pauli_action_cache=pauli_action_cache,
                    mode=mode,
                    rejection_counts=rejection_counts,
                    additivity_diagnostics=additivity_diagnostics,
                    geometry_workspace=geometry_workspace,
                )
                candidate_batch_eval_count += 1
                if trial is None:
                    continue
                subset_counts_feasible[subset_size] = int(
                    subset_counts_feasible.get(subset_size, 0) + 1
                )
                if best_trial is None or _batch_proposal_better(
                    trial,
                    best_trial,
                    cfg=cfg,
                ):
                    best_trial = trial
                    best_record = dict(rec)
            if (
                best_trial is None
                or best_record is None
                or float(best_trial.score) <= float(current.score) + float(cfg.cheap_score_eps)
            ):
                break
            batch.append(best_record)
            current = best_trial
        annotated_records = _annotate_phase3_batch_records(current.records, proposal=current, mode=mode)
        proposals.append(
            BatchSelectionProposal(
                records=tuple(annotated_records),
                summary={
                    **dict(current.summary),
                    "reason": "cost_weighted_greedy_batch" if len(annotated_records) > 1 else "singleton_shell",
                    "shell_size": int(len(shell)),
                    "candidate_batch_eval_count": int(candidate_batch_eval_count),
                    "same_generator_duplicate_skip_count": int(duplicate_generator_skip_count),
                    "same_generator_duplicate_identities": [
                        {"kind": str(kind), "value": str(value)}
                        for kind, value in sorted(duplicate_generator_identities)
                    ],
                },
                score=float(current.score),
                delta_e3=float(current.delta_e3),
                k3=float(current.k3),
                denominator_1_plus_k3=float(current.denominator_1_plus_k3),
            )
        )
    proposals_sorted = _dedupe_batch_proposals(proposals, cfg=cfg)[:proposal_count_cap]
    summary = {
        "selection_mode": mode,
        "selected": bool(proposals_sorted and len(proposals_sorted[0].records) > 1),
        "reason": "cost_weighted_greedy_batch" if proposals_sorted and len(proposals_sorted[0].records) > 1 else "singleton_shell",
        "shell_size": int(len(shell)),
        "selected_count": int(len(proposals_sorted[0].records)) if proposals_sorted else 0,
        "candidate_batch_eval_count": int(candidate_batch_eval_count),
        "subset_evaluation_count": int(candidate_batch_eval_count),
        "subset_counts_considered": {
            str(size): int(count)
            for size, count in sorted(subset_counts_considered.items())
        },
        "subset_counts_feasible": {
            str(size): int(count)
            for size, count in sorted(subset_counts_feasible.items())
        },
        "subset_counts_evaluated": {
            str(size): int(count)
            for size, count in sorted(subset_counts_evaluated.items())
        },
        "reused_child_phase2_singleton_subset_count": int(
            subset_counts_evaluated.get(1, 0)
        ),
        "query_chargeable_batch_subset_count": int(
            sum(
                count
                for size, count in subset_counts_evaluated.items()
                if int(size) > 1
            )
        ),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "rank_gate_rejection_count": int(rejection_counts.get("rank_gate", 0)),
        "conditioning_gate_rejection_count": int(
            rejection_counts.get("conditioning_gate", 0)
        ),
        "additivity_diagnostics": [dict(row) for row in additivity_diagnostics],
        "subset_diagnostics": [dict(row) for row in additivity_diagnostics],
        "additivity_policy": normalize_batch_additivity_policy(
            getattr(cfg, "batch_additivity_policy", None)
        ),
        "lambda_add": float(max(0.0, getattr(cfg, "batch_additivity_lambda", 0.0))),
        "geometry_mode": str(geometry_mode),
        "geometry_workspace": (
            {}
            if geometry_workspace is None
            else geometry_workspace.build_telemetry()
        ),
        "proposal_count": int(len(proposals_sorted)),
        "effective_batch_size_cap": int(cap_count),
        "effective_batch_target_size": int(target_count),
    }
    if proposals_sorted:
        summary.update(dict(proposals_sorted[0].summary))
        summary["proposal_count"] = int(len(proposals_sorted))
        summary["candidate_batch_eval_count"] = int(candidate_batch_eval_count)
        summary["subset_evaluation_count"] = int(candidate_batch_eval_count)
        summary["subset_counts_considered"] = {
            str(size): int(count)
            for size, count in sorted(subset_counts_considered.items())
        }
        summary["subset_counts_evaluated"] = {
            str(size): int(count)
            for size, count in sorted(subset_counts_evaluated.items())
        }
        summary["subset_counts_feasible"] = {
            str(size): int(count)
            for size, count in sorted(subset_counts_feasible.items())
        }
        summary["query_chargeable_batch_subset_count"] = int(
            sum(
                count
                for size, count in subset_counts_evaluated.items()
                if int(size) > 1
            )
        )
        summary["rejection_counts"] = dict(sorted(rejection_counts.items()))
        summary["rank_gate_rejection_count"] = int(
            rejection_counts.get("rank_gate", 0)
        )
        summary["conditioning_gate_rejection_count"] = int(
            rejection_counts.get("conditioning_gate", 0)
        )
        summary["additivity_diagnostics"] = [
            dict(row) for row in additivity_diagnostics
        ]
        summary["subset_diagnostics"] = [
            dict(row) for row in additivity_diagnostics
        ]
        summary["geometry_mode"] = str(geometry_mode)
        summary["geometry_workspace"] = (
            {}
            if geometry_workspace is None
            else geometry_workspace.build_telemetry()
        )
        summary["selected_subset"] = [
            {
                "identity_kind": str(_batch_record_generator_identity(record)[0]),
                "identity": str(_batch_record_generator_identity(record)[1]),
                "position_id": int(record.get("position_id", -1)),
                "label": str(_batch_record_label(record)),
            }
            for record in proposals_sorted[0].records
        ]
        summary["selected_cardinality"] = int(len(proposals_sorted[0].records))
    return proposals_sorted, summary


def combinatorial_reduced_plane_batch_proposals(
    ranked_records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    novelty_oracle: Any,
    curvature_oracle: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    tie_break_score_key: str = "phase2_raw_score",
    max_proposals: int = 1,
    joint_pair_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[list[BatchSelectionProposal], dict[str, Any]]:
    mode = PHASE2_BATCH_COMBINATORIAL_REDUCED_PLANE
    ranked, shell, _top_score = _batch_search_population(
        ranked_records,
        cfg=cfg,
        tie_break_score_key=tie_break_score_key,
    )
    target_count, cap_count = _ordered_reduced_plane_batch_limits(cfg)
    if not ranked:
        return [], {
            "selection_mode": mode,
            "selected": False,
            "reason": "empty_shortlist",
            "shell_size": 0,
            "candidate_batch_eval_count": 0,
            "effective_batch_size_cap": int(cap_count),
            "effective_batch_target_size": int(target_count),
        }
    if not shell:
        proposal = _fallback_singleton_batch_proposal(
            ranked[0],
            cfg=cfg,
            mode=mode,
            reason="nonpositive_shell",
        )
        return [proposal], {"selection_mode": mode, **dict(proposal.summary), "shell_size": 0, "candidate_batch_eval_count": 0}

    requested_search_pool_size = getattr(cfg, "batch_search_pool_size", None)
    search_population, rank_prefilter = _rank_feasible_child_phase2_population(
        shell,
        cfg=cfg,
        selected_ops=selected_ops,
        theta=theta,
        psi_state=psi_state,
        h_compiled=h_compiled,
    )
    if _canonical_ranked_child_batch_search(cfg):
        if requested_search_pool_size is None:
            raise ValueError(
                "Canonical ranked child batch search requires explicit "
                "batch_search_pool_size; use 0 for all survivors."
            )
        requested_search_pool_size = int(requested_search_pool_size)
        if requested_search_pool_size < 0:
            raise ValueError("batch_search_pool_size must be >= 0; 0 means all.")
        search_pool_size = (
            int(len(search_population))
            if requested_search_pool_size == 0
            else int(min(len(search_population), requested_search_pool_size))
        )
        enumeration_size_cap = int(cap_count)
    else:
        search_pool_size = int(
            min(len(search_population), max(2 * cap_count, cap_count), 10)
        )
        enumeration_size_cap = int(target_count)
    search_pool = [dict(rec) for rec in search_population[:search_pool_size]]
    geometry_mode = normalize_batch_geometry_mode(
        getattr(cfg, "batch_geometry_mode", None)
    )
    geometry_workspace = (
        None
        if (
            not search_pool
            or geometry_mode
            == BATCH_GEOMETRY_PER_SUBSET_DIAGONAL_HESSIAN_LEGACY_V1
        )
        else _build_batch_full_geometry_workspace(
            search_pool,
            cfg=cfg,
            selected_ops=selected_ops,
            theta=theta,
            psi_ref=psi_ref,
            psi_state=psi_state,
            h_compiled=h_compiled,
            pauli_action_cache=pauli_action_cache,
            joint_pair_observer=joint_pair_observer,
        )
    )
    proposals: list[BatchSelectionProposal] = []
    candidate_batch_eval_count = 0
    subset_counts_considered: dict[int, int] = {}
    subset_counts_evaluated: dict[int, int] = {}
    subset_counts_feasible: dict[int, int] = {}
    rejection_counts: dict[str, int] = {}
    additivity_diagnostics: list[dict[str, Any]] = []
    duplicate_generator_skip_count = 0
    duplicate_generator_identities: set[tuple[str, str]] = set()
    for size in range(1, int(enumeration_size_cap) + 1):
        for combo in itertools.combinations(search_pool, int(size)):
            subset_counts_considered[int(size)] = int(
                subset_counts_considered.get(int(size), 0) + 1
            )
            identities = [_batch_record_generator_identity(rec) for rec in combo]
            if len(set(identities)) != len(identities):
                duplicate_generator_skip_count += 1
                rejection_counts["duplicate_exact_child_identity"] = int(
                    rejection_counts.get("duplicate_exact_child_identity", 0) + 1
                )
                for identity in identities:
                    if identities.count(identity) > 1:
                        duplicate_generator_identities.add(identity)
                continue
            subset_counts_evaluated[int(size)] = int(
                subset_counts_evaluated.get(int(size), 0) + 1
            )
            proposal = _evaluate_ordered_reduced_plane_batch_proposal(
                [dict(rec) for rec in combo],
                cfg=cfg,
                selected_ops=selected_ops,
                theta=theta,
                psi_ref=psi_ref,
                psi_state=psi_state,
                h_compiled=h_compiled,
                novelty_oracle=novelty_oracle,
                curvature_oracle=curvature_oracle,
                compiled_cache=compiled_cache,
                pauli_action_cache=pauli_action_cache,
                mode=mode,
                rejection_counts=rejection_counts,
                additivity_diagnostics=additivity_diagnostics,
                geometry_workspace=geometry_workspace,
            )
            candidate_batch_eval_count += 1
            if proposal is None:
                continue
            subset_counts_feasible[int(size)] = int(
                subset_counts_feasible.get(int(size), 0) + 1
            )
            annotated_records = _annotate_phase3_batch_records(proposal.records, proposal=proposal, mode=mode)
            proposals.append(
                BatchSelectionProposal(
                    records=tuple(annotated_records),
                    summary={
                        **dict(proposal.summary),
                        "reason": "cost_weighted_combinatorial_batch" if len(annotated_records) > 1 else "singleton_shell",
                        "shell_size": int(len(shell)),
                        "search_pool_size": int(search_pool_size),
                        "search_pool_truncated": bool(
                            len(search_population) > search_pool_size
                        ),
                        "candidate_batch_eval_count": int(candidate_batch_eval_count),
                        "same_generator_duplicate_skip_count": int(duplicate_generator_skip_count),
                        "same_generator_duplicate_identities": [
                            {"kind": str(kind), "value": str(value)}
                            for kind, value in sorted(duplicate_generator_identities)
                        ],
                    },
                    score=float(proposal.score),
                    delta_e3=float(proposal.delta_e3),
                    k3=float(proposal.k3),
                    denominator_1_plus_k3=float(proposal.denominator_1_plus_k3),
                )
            )
    proposals_sorted = _dedupe_batch_proposals(proposals, cfg=cfg)[
        : int(max(1, max_proposals))
    ]
    summary = {
        "selection_mode": mode,
        "selected": bool(proposals_sorted and len(proposals_sorted[0].records) > 1),
        "reason": "cost_weighted_combinatorial_batch" if proposals_sorted and len(proposals_sorted[0].records) > 1 else "singleton_shell",
        "shell_size": int(len(shell)),
        "selected_count": int(len(proposals_sorted[0].records)) if proposals_sorted else 0,
        "search_pool_size": int(search_pool_size),
        "search_pool_truncated": bool(
            len(search_population) > search_pool_size
        ),
        "batch_search_pool_size_requested": (
            None
            if requested_search_pool_size is None
            else int(requested_search_pool_size)
        ),
        "batch_search_pool_size_effective": int(search_pool_size),
        "batch_search_pool_truncated": bool(
            len(search_population) > search_pool_size
        ),
        "child_phase2_survivor_count": int(len(ranked)),
        "child_phase2_search_population_count": int(len(search_population)),
        "child_phase2_rank_feasible_count": int(len(search_population)),
        "child_phase2_rank_feasible_count_semantics": (
            "singleton_prefilter_survivor_count"
            if bool(rank_prefilter.get("active", False))
            else "compatibility_alias_for_unfiltered_search_population_count"
        ),
        "rank_feasibility_prefilter": dict(rank_prefilter),
        "rank_gate_application_stage": str(
            rank_prefilter.get(
                "rank_gate_application_stage",
                "joint_subset_after_search_pool",
            )
        ),
        "rank_prefilter_rejection_count": int(
            rank_prefilter.get("rank_rejected_record_count", 0)
        ),
        "candidate_batch_eval_count": int(candidate_batch_eval_count),
        "subset_evaluation_count": int(candidate_batch_eval_count),
        "subset_counts_considered": {
            str(size): int(count)
            for size, count in sorted(subset_counts_considered.items())
        },
        "subset_counts_feasible": {
            str(size): int(count)
            for size, count in sorted(subset_counts_feasible.items())
        },
        "subset_counts_evaluated": {
            str(size): int(count)
            for size, count in sorted(subset_counts_evaluated.items())
        },
        "reused_child_phase2_singleton_subset_count": int(
            subset_counts_evaluated.get(1, 0)
        ),
        "query_chargeable_batch_subset_count": int(
            sum(
                count
                for size, count in subset_counts_evaluated.items()
                if int(size) > 1
            )
        ),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "rank_gate_rejection_count": int(rejection_counts.get("rank_gate", 0)),
        "conditioning_gate_rejection_count": int(
            rejection_counts.get("conditioning_gate", 0)
        ),
        "additivity_diagnostics": [dict(row) for row in additivity_diagnostics],
        "subset_diagnostics": [dict(row) for row in additivity_diagnostics],
        "additivity_policy": normalize_batch_additivity_policy(
            getattr(cfg, "batch_additivity_policy", None)
        ),
        "lambda_add": float(max(0.0, getattr(cfg, "batch_additivity_lambda", 0.0))),
        "geometry_mode": str(geometry_mode),
        "geometry_workspace": (
            {}
            if geometry_workspace is None
            else geometry_workspace.build_telemetry()
        ),
        "proposal_count": int(len(proposals_sorted)),
        "effective_batch_size_cap": int(cap_count),
        "effective_batch_target_size": int(target_count),
    }
    if proposals_sorted:
        summary.update(dict(proposals_sorted[0].summary))
        summary["proposal_count"] = int(len(proposals_sorted))
        summary["candidate_batch_eval_count"] = int(candidate_batch_eval_count)
        summary["subset_evaluation_count"] = int(candidate_batch_eval_count)
        summary["subset_counts_considered"] = {
            str(size): int(count)
            for size, count in sorted(subset_counts_considered.items())
        }
        summary["subset_counts_evaluated"] = {
            str(size): int(count)
            for size, count in sorted(subset_counts_evaluated.items())
        }
        summary["subset_counts_feasible"] = {
            str(size): int(count)
            for size, count in sorted(subset_counts_feasible.items())
        }
        summary["query_chargeable_batch_subset_count"] = int(
            sum(
                count
                for size, count in subset_counts_evaluated.items()
                if int(size) > 1
            )
        )
        summary["rejection_counts"] = dict(sorted(rejection_counts.items()))
        summary["rank_gate_rejection_count"] = int(
            rejection_counts.get("rank_gate", 0)
        )
        summary["conditioning_gate_rejection_count"] = int(
            rejection_counts.get("conditioning_gate", 0)
        )
        summary["additivity_diagnostics"] = [
            dict(row) for row in additivity_diagnostics
        ]
        summary["subset_diagnostics"] = [
            dict(row) for row in additivity_diagnostics
        ]
        summary["geometry_mode"] = str(geometry_mode)
        summary["geometry_workspace"] = (
            {}
            if geometry_workspace is None
            else geometry_workspace.build_telemetry()
        )
        summary["selected_subset"] = [
            {
                "identity_kind": str(_batch_record_generator_identity(record)[0]),
                "identity": str(_batch_record_generator_identity(record)[1]),
                "position_id": int(record.get("position_id", -1)),
                "label": str(_batch_record_label(record)),
            }
            for record in proposals_sorted[0].records
        ]
        summary["selected_cardinality"] = int(len(proposals_sorted[0].records))
    return proposals_sorted, summary


def select_phase2_batch_record_proposals(
    ranked_records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    novelty_oracle: Any,
    curvature_oracle: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    tie_break_score_key: str = "phase2_raw_score",
    max_proposals: int = 1,
) -> tuple[list[BatchSelectionProposal], dict[str, Any]]:
    mode = normalize_phase2_batch_selection_mode(getattr(cfg, "batch_selection_mode", PHASE2_BATCH_REDUCED_PLANE))
    if mode == PHASE2_BATCH_GREEDY_REDUCED_PLANE:
        return greedy_reduced_plane_batch_proposals(
            ranked_records,
            cfg=cfg,
            selected_ops=selected_ops,
            theta=theta,
            psi_ref=psi_ref,
            psi_state=psi_state,
            h_compiled=h_compiled,
            novelty_oracle=novelty_oracle,
            curvature_oracle=curvature_oracle,
            compiled_cache=compiled_cache,
            pauli_action_cache=pauli_action_cache,
            tie_break_score_key=tie_break_score_key,
            max_proposals=max_proposals,
        )
    if mode == PHASE2_BATCH_COMBINATORIAL_REDUCED_PLANE:
        return combinatorial_reduced_plane_batch_proposals(
            ranked_records,
            cfg=cfg,
            selected_ops=selected_ops,
            theta=theta,
            psi_ref=psi_ref,
            psi_state=psi_state,
            h_compiled=h_compiled,
            novelty_oracle=novelty_oracle,
            curvature_oracle=curvature_oracle,
            compiled_cache=compiled_cache,
            pauli_action_cache=pauli_action_cache,
            tie_break_score_key=tie_break_score_key,
            max_proposals=max_proposals,
        )
    selected, summary = select_phase2_batch_records(
        ranked_records,
        cfg=cfg,
        selected_ops=selected_ops,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        novelty_oracle=novelty_oracle,
        curvature_oracle=curvature_oracle,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
        tie_break_score_key=tie_break_score_key,
    )
    if not selected:
        return [], dict(summary)
    delta_e3 = float(max(0.0, float(summary.get("joint_gain", sum(max(0.0, _batch_record_score(rec, "full_v2_score", 0.0)) for rec in selected)))))
    k3 = _phase3_batch_k3(selected)
    denominator = float(max(float(cfg.cheap_score_eps), 1.0 + float(k3)))
    proposal = BatchSelectionProposal(
        records=tuple(dict(rec) for rec in selected),
        summary=dict(summary),
        score=float(delta_e3 / denominator),
        delta_e3=float(delta_e3),
        k3=float(k3),
        denominator_1_plus_k3=float(denominator),
    )
    return [proposal], dict(summary)


def greedy_reduced_plane_batch_select(
    ranked_records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    novelty_oracle: Any,
    curvature_oracle: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    tie_break_score_key: str = "phase2_raw_score",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proposals, summary = greedy_reduced_plane_batch_proposals(
        ranked_records,
        cfg=cfg,
        selected_ops=selected_ops,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        novelty_oracle=novelty_oracle,
        curvature_oracle=curvature_oracle,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
        tie_break_score_key=tie_break_score_key,
        max_proposals=1,
    )
    return (list(proposals[0].records), dict(summary)) if proposals else ([], dict(summary))


def combinatorial_reduced_plane_batch_select(
    ranked_records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    novelty_oracle: Any,
    curvature_oracle: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    tie_break_score_key: str = "phase2_raw_score",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proposals, summary = combinatorial_reduced_plane_batch_proposals(
        ranked_records,
        cfg=cfg,
        selected_ops=selected_ops,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        novelty_oracle=novelty_oracle,
        curvature_oracle=curvature_oracle,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
        tie_break_score_key=tie_break_score_key,
        max_proposals=1,
    )
    return (list(proposals[0].records), dict(summary)) if proposals else ([], dict(summary))


def reduced_plane_batch_select(
    ranked_records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    novelty_oracle: Any,
    curvature_oracle: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    tie_break_score_key: str = "simple_score",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ranked = sorted([dict(rec) for rec in ranked_records], key=lambda rec: _batch_sort_key(rec, tie_break_score_key))
    if not ranked:
        return [], {"selected": False, "reason": "empty_shortlist"}
    top_score = float(ranked[0].get("full_v2_score", float("-inf")))
    shell = [
        dict(rec)
        for rec in ranked
        if float(rec.get("full_v2_score", float("-inf"))) > 0.0
        and float(rec.get("full_v2_score", float("-inf"))) >= float(cfg.batch_near_degenerate_ratio) * float(top_score)
    ]
    if not shell:
        return [dict(ranked[0])], {"selected": False, "reason": "nonpositive_shell"}
    batch = [dict(shell[0])]
    batch_summary = _batch_geometry_summary(
        batch,
        cfg=cfg,
        selected_ops=selected_ops,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        h_compiled=h_compiled,
        novelty_oracle=novelty_oracle,
        curvature_oracle=curvature_oracle,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
    )
    current_gain = float(batch_summary.get("joint_gain", batch[0].get("full_v2_score", 0.0)))
    duplicate_generator_skip_count = 0
    duplicate_generator_identities: set[tuple[str, str]] = set()
    while len(batch) < int(max(1, cfg.batch_size_cap)) and len(batch) < int(max(1, cfg.batch_target_size)):
        best_candidate: dict[str, Any] | None = None
        best_summary: dict[str, Any] | None = None
        best_marginal = 0.0
        batch_keys = {_batch_record_generator_identity(rec) for rec in batch}
        for rec in shell[1:]:
            rec_key = _batch_record_generator_identity(rec)
            if rec_key in batch_keys:
                duplicate_generator_skip_count += 1
                duplicate_generator_identities.add(rec_key)
                continue
            trial_batch = [dict(x) for x in batch] + [dict(rec)]
            trial_summary = _batch_geometry_summary(
                trial_batch,
                cfg=cfg,
                selected_ops=selected_ops,
                theta=theta,
                psi_ref=psi_ref,
                psi_state=psi_state,
                h_compiled=h_compiled,
                novelty_oracle=novelty_oracle,
                curvature_oracle=curvature_oracle,
                compiled_cache=compiled_cache,
                pauli_action_cache=pauli_action_cache,
            )
            if not bool(trial_summary.get("feasible", False)):
                continue
            marginal = float(trial_summary.get("joint_gain", 0.0)) - float(current_gain)
            if marginal > float(best_marginal):
                best_candidate = dict(rec)
                best_summary = dict(trial_summary)
                best_marginal = float(marginal)
        if best_candidate is None or float(best_marginal) <= 0.0:
            break
        batch.append(best_candidate)
        batch_summary = dict(best_summary) if isinstance(best_summary, Mapping) else batch_summary
        current_gain = float(batch_summary.get("joint_gain", current_gain))
    annotated: list[dict[str, Any]] = []
    for rec in batch:
        updated = dict(rec)
        feat = updated.get("feature")
        if isinstance(feat, CandidateFeatures):
            updated["feature"] = _replace_feature(
                feat,
                compatibility_penalty_total=float(batch_summary.get("additivity_defect", 0.0)),
            )
        updated["compatibility_penalty"] = {
            "total": float(batch_summary.get("additivity_defect", 0.0)),
            "joint_gain": float(batch_summary.get("joint_gain", 0.0)),
            "contextual_single_total": float(batch_summary.get("contextual_single_total", 0.0)),
            "lambda_min": float(batch_summary.get("lambda_min", 0.0)),
            "rank_floor": float(batch_summary.get("rank_floor", 0.0)),
            "mu_tan": float(batch_summary.get("mu_tan", 0.0)),
        }
        annotated.append(updated)
    summary_out = dict(batch_summary)
    summary_out["same_generator_batch_duplicate_policy"] = "block_generator_identity_v1"
    summary_out["same_generator_duplicate_skip_count"] = int(duplicate_generator_skip_count)
    summary_out["same_generator_duplicate_identities"] = [
        {"kind": str(kind), "value": str(value)}
        for kind, value in sorted(duplicate_generator_identities)
    ]
    return annotated, summary_out


def _batch_record_label(record: Mapping[str, Any]) -> str:
    feat = record.get("feature")
    if isinstance(feat, CandidateFeatures):
        return str(feat.candidate_label)
    if record.get("candidate_label") is not None:
        return str(record.get("candidate_label"))
    term = record.get("candidate_term")
    if getattr(term, "label", None) is not None:
        return str(getattr(term, "label"))
    return ""


def _batch_record_generator_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    """Return the batch-local generator identity, intentionally ignoring position.

    Batch admission may consider several insertion positions for the same
    underlying generator.  Those are useful alternatives, not independent
    directions to buy in one batched ADAPT step.  Use the durable generator id
    when present; fall back to pool index, then label, then term label so older
    records still get deterministic duplicate suppression.
    """

    global_pauli_identity = record.get("route_a_global_pauli_identity")
    if global_pauli_identity not in {None, ""}:
        return ("global_pauli_identity", str(global_pauli_identity))

    feat = record.get("feature")
    if isinstance(feat, CandidateFeatures):
        generator_id = getattr(feat, "generator_id", None)
        if generator_id not in {None, ""}:
            return ("generator_id", str(generator_id))
        candidate_pool_index = getattr(feat, "candidate_pool_index", None)
        if candidate_pool_index is not None:
            try:
                return ("candidate_pool_index", str(int(candidate_pool_index)))
            except (TypeError, ValueError):
                return ("candidate_pool_index", str(candidate_pool_index))
        label = getattr(feat, "candidate_label", None)
        if label not in {None, ""}:
            return ("candidate_label", str(label))

    generator_id = record.get("generator_id")
    if generator_id not in {None, ""}:
        return ("generator_id", str(generator_id))
    if record.get("candidate_pool_index") is not None:
        try:
            return ("candidate_pool_index", str(int(record.get("candidate_pool_index"))))
        except (TypeError, ValueError):
            return ("candidate_pool_index", str(record.get("candidate_pool_index")))
    label = _batch_record_label(record)
    if label:
        return ("candidate_label", str(label))
    return ("record", str(id(record)))


def _batch_record_score(
    record: Mapping[str, Any],
    key: str,
    default: float = float("-inf"),
) -> float:
    raw = record.get(key, default)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _annotate_zero_batch_compatibility(
    records: Sequence[Mapping[str, Any]],
    *,
    joint_gain: float,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for rec in records:
        updated = dict(rec)
        feat = updated.get("feature")
        if isinstance(feat, CandidateFeatures):
            updated["feature"] = _replace_feature(
                feat,
                compatibility_penalty_total=0.0,
            )
        updated["compatibility_penalty"] = {
            "support_overlap": 0.0,
            "noncommutation": 0.0,
            "cross_curvature": 0.0,
            "schedule": 0.0,
            "measurement_mismatch": 0.0,
            "total": 0.0,
            "joint_gain": float(joint_gain),
        }
        annotated.append(updated)
    return annotated


def _overlap_sparse_row_norm(row: Mapping[str, float]) -> float:
    return math.sqrt(float(sum(float(value) * float(value) for value in row.values())))


def _operator_sparse_feature_row(term: Any) -> dict[str, float]:
    """Return a sparse operator-row fallback without rebuilding candidate scores."""
    if term is None or not hasattr(term, "polynomial"):
        return {}
    try:
        poly_terms = list(term.polynomial.return_polynomial())
    except Exception:
        return {}
    row: dict[str, float] = {}
    for poly_term in poly_terms:
        try:
            label = str(poly_term.pw2strng())
            coeff = complex(getattr(poly_term, "p_coeff", 1.0))
        except Exception:
            continue
        if math.isfinite(float(coeff.real)) and float(coeff.real) != 0.0:
            key = f"pauli_re:{label}"
            row[key] = float(row.get(key, 0.0) + float(coeff.real))
        if math.isfinite(float(coeff.imag)) and float(coeff.imag) != 0.0:
            key = f"pauli_im:{label}"
            row[key] = float(row.get(key, 0.0) + float(coeff.imag))
    return {key: value for key, value in row.items() if float(value) != 0.0}


def _overlap_feature_row(record: Mapping[str, Any]) -> dict[str, float] | None:
    """Extract the existing candidate feature row used for benchmark overlap screening.

    The preferred row is the persisted phase-2 curvature coupling vector
    (``CandidateFeatures.b_hat`` / telemetry ``b_hat``).  At empty-window depths
    that vector is legitimately absent, so the selector falls back to the
    already-present candidate operator row rather than recomputing or rescoring
    candidate features.
    """
    feat = record.get("feature")
    raw_b_hat: Any = None
    if isinstance(feat, CandidateFeatures):
        raw_b_hat = feat.b_hat
    elif isinstance(feat, Mapping):
        raw_b_hat = feat.get("b_hat")
    if isinstance(raw_b_hat, Sequence) and not isinstance(raw_b_hat, (str, bytes)):
        row: dict[str, float] = {}
        for idx, raw_value in enumerate(raw_b_hat):
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value != 0.0:
                row[f"b_hat:{int(idx)}"] = float(value)
        if _overlap_sparse_row_norm(row) > 0.0:
            return row
    row = _operator_sparse_feature_row(record.get("candidate_term"))
    if _overlap_sparse_row_norm(row) > 0.0:
        return row
    return None


def _normalized_sparse_feature_overlap(lhs: Mapping[str, float], rhs: Mapping[str, float]) -> float:
    lhs_norm = _overlap_sparse_row_norm(lhs)
    rhs_norm = _overlap_sparse_row_norm(rhs)
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        return float("inf")
    if len(lhs) <= len(rhs):
        dot = sum(float(value) * float(rhs.get(key, 0.0)) for key, value in lhs.items())
    else:
        dot = sum(float(lhs.get(key, 0.0)) * float(value) for key, value in rhs.items())
    return float(abs(dot) / (lhs_norm * rhs_norm))


def ceo_commuting_batch_select(
    ranked_records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    tie_break_score_key: str = "phase2_raw_score",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Benchmark-only CEO-ADAPT-style conservative commuting batch selector."""
    ranked = sorted([dict(rec) for rec in ranked_records], key=lambda rec: _batch_sort_key(rec, tie_break_score_key))
    mode = "ceo_commuting_benchmark"
    if not ranked:
        return [], {
            "selection_mode": mode,
            "selected": False,
            "reason": "empty_shortlist",
            "shell_size": 0,
            "selected_count": 0,
            "rejected_noncommuting_count": 0,
            "rejected_invalid_pauli_count": 0,
            "joint_gain": 0.0,
            "additivity_defect": 0.0,
            "selected_labels": [],
        }
    top_score = _batch_record_score(ranked[0], "full_v2_score")
    shell = [
        dict(rec)
        for rec in ranked
        if _batch_record_score(rec, "full_v2_score") > 0.0
        and _batch_record_score(rec, "full_v2_score") >= float(cfg.batch_near_degenerate_ratio) * float(top_score)
    ]
    target_count = int(max(1, min(int(max(1, cfg.batch_target_size)), int(max(1, cfg.batch_size_cap)))))
    if not shell:
        selected = [dict(ranked[0])]
        joint_gain = float(max(0.0, _batch_record_score(selected[0], "full_v2_score", 0.0)))
        annotated = _annotate_zero_batch_compatibility(selected, joint_gain=joint_gain)
        return annotated, {
            "selection_mode": mode,
            "selected": False,
            "reason": "nonpositive_shell",
            "shell_size": 0,
            "selected_count": 1,
            "rejected_noncommuting_count": 0,
            "rejected_invalid_pauli_count": 0,
            "joint_gain": float(joint_gain),
            "additivity_defect": 0.0,
            "selected_labels": [_batch_record_label(rec) for rec in annotated],
        }

    selected: list[dict[str, Any]] = [dict(shell[0])]
    selected_words: list[tuple[str, ...]] = []
    top_words = _record_pauli_words(shell[0])
    if not top_words:
        rejected_invalid_pauli_count = sum(1 for rec in shell[1:] if not _record_pauli_words(rec))
        joint_gain = float(max(0.0, _batch_record_score(selected[0], "full_v2_score", 0.0)))
        annotated = _annotate_zero_batch_compatibility(selected, joint_gain=joint_gain)
        return annotated, {
            "selection_mode": mode,
            "selected": False,
            "reason": "top_invalid_pauli_fallback",
            "shell_size": int(len(shell)),
            "selected_count": 1,
            "rejected_noncommuting_count": 0,
            "rejected_invalid_pauli_count": int(rejected_invalid_pauli_count),
            "joint_gain": float(joint_gain),
            "additivity_defect": 0.0,
            "selected_labels": [_batch_record_label(rec) for rec in annotated],
        }
    selected_words.append(tuple(top_words))

    rejected_noncommuting_count = 0
    rejected_invalid_pauli_count = 0
    for rec in shell[1:]:
        if len(selected) >= target_count:
            break
        words = _record_pauli_words(rec)
        if not words:
            rejected_invalid_pauli_count += 1
            continue
        commutes_with_selected = all(
            _single_pauli_word_commutes(candidate_word, selected_word)
            for candidate_word in words
            for selected_record_words in selected_words
            for selected_word in selected_record_words
        )
        if commutes_with_selected:
            selected.append(dict(rec))
            selected_words.append(tuple(words))
        else:
            rejected_noncommuting_count += 1

    joint_gain = float(sum(max(0.0, _batch_record_score(rec, "full_v2_score", 0.0)) for rec in selected))
    annotated = _annotate_zero_batch_compatibility(selected, joint_gain=joint_gain)
    return annotated, {
        "selection_mode": mode,
        "selected": bool(len(annotated) > 1),
        "reason": "hard_commuting_batch" if len(annotated) > 1 else "singleton_shell",
        "shell_size": int(len(shell)),
        "selected_count": int(len(annotated)),
        "rejected_noncommuting_count": int(rejected_noncommuting_count),
        "rejected_invalid_pauli_count": int(rejected_invalid_pauli_count),
        "joint_gain": float(joint_gain),
        "additivity_defect": 0.0,
        "selected_labels": [_batch_record_label(rec) for rec in annotated],
    }


def overlap_orthogonal_batch_select(
    ranked_records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    tie_break_score_key: str = "phase2_raw_score",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Benchmark-only Overlap-ADAPT-style hard low-overlap batch selector."""
    ranked = sorted([dict(rec) for rec in ranked_records], key=lambda rec: _batch_sort_key(rec, tie_break_score_key))
    mode = "overlap_orthogonal_benchmark"
    threshold = float(OVERLAP_ORTHOGONAL_BENCHMARK_MAX)
    if not ranked:
        return [], {
            "selection_mode": mode,
            "selected": False,
            "reason": "empty_shortlist",
            "shell_size": 0,
            "selected_count": 0,
            "rejected_overlap_count": 0,
            "rejected_invalid_feature_count": 0,
            "overlap_threshold": float(threshold),
            "max_pairwise_overlap": 0.0,
            "joint_gain": 0.0,
            "additivity_defect": 0.0,
            "selected_labels": [],
        }
    top_score = _batch_record_score(ranked[0], "full_v2_score")
    shell = [
        dict(rec)
        for rec in ranked
        if _batch_record_score(rec, "full_v2_score") > 0.0
        and _batch_record_score(rec, "full_v2_score") >= float(cfg.batch_near_degenerate_ratio) * float(top_score)
    ]
    target_count = int(max(1, min(int(max(1, cfg.batch_target_size)), int(max(1, cfg.batch_size_cap)))))
    if not shell:
        selected = [dict(ranked[0])]
        joint_gain = float(max(0.0, _batch_record_score(selected[0], "full_v2_score", 0.0)))
        annotated = _annotate_zero_batch_compatibility(selected, joint_gain=joint_gain)
        return annotated, {
            "selection_mode": mode,
            "selected": False,
            "reason": "nonpositive_shell",
            "shell_size": 0,
            "selected_count": 1,
            "rejected_overlap_count": 0,
            "rejected_invalid_feature_count": 0,
            "overlap_threshold": float(threshold),
            "max_pairwise_overlap": 0.0,
            "joint_gain": float(joint_gain),
            "additivity_defect": 0.0,
            "selected_labels": [_batch_record_label(rec) for rec in annotated],
        }

    selected: list[dict[str, Any]] = [dict(shell[0])]
    selected_rows: list[dict[str, float]] = []
    top_row = _overlap_feature_row(shell[0])
    if top_row is None:
        joint_gain = float(max(0.0, _batch_record_score(selected[0], "full_v2_score", 0.0)))
        annotated = _annotate_zero_batch_compatibility(selected, joint_gain=joint_gain)
        return annotated, {
            "selection_mode": mode,
            "selected": False,
            "reason": "top_invalid_feature_fallback",
            "shell_size": int(len(shell)),
            "selected_count": 1,
            "rejected_overlap_count": 0,
            "rejected_invalid_feature_count": int(max(0, len(shell) - 1)),
            "overlap_threshold": float(threshold),
            "max_pairwise_overlap": 0.0,
            "joint_gain": float(joint_gain),
            "additivity_defect": 0.0,
            "selected_labels": [_batch_record_label(rec) for rec in annotated],
        }
    selected_rows.append(dict(top_row))

    rejected_overlap_count = 0
    rejected_invalid_feature_count = 0
    max_pairwise_overlap = 0.0
    for rec in shell[1:]:
        if len(selected) >= target_count:
            break
        row = _overlap_feature_row(rec)
        if row is None:
            rejected_invalid_feature_count += 1
            continue
        overlaps = [_normalized_sparse_feature_overlap(row, selected_row) for selected_row in selected_rows]
        candidate_overlap = float(max(overlaps) if overlaps else 0.0)
        if math.isfinite(candidate_overlap):
            max_pairwise_overlap = float(max(max_pairwise_overlap, candidate_overlap))
        if candidate_overlap < threshold:
            selected.append(dict(rec))
            selected_rows.append(dict(row))
        else:
            rejected_overlap_count += 1

    joint_gain = float(sum(max(0.0, _batch_record_score(rec, "full_v2_score", 0.0)) for rec in selected))
    annotated = _annotate_zero_batch_compatibility(selected, joint_gain=joint_gain)
    return annotated, {
        "selection_mode": mode,
        "selected": bool(len(annotated) > 1),
        "reason": "hard_overlap_orthogonal_batch" if len(annotated) > 1 else "singleton_shell",
        "shell_size": int(len(shell)),
        "selected_count": int(len(annotated)),
        "rejected_overlap_count": int(rejected_overlap_count),
        "rejected_invalid_feature_count": int(rejected_invalid_feature_count),
        "overlap_threshold": float(threshold),
        "max_pairwise_overlap": float(max_pairwise_overlap),
        "joint_gain": float(joint_gain),
        "additivity_defect": 0.0,
        "selected_labels": [_batch_record_label(rec) for rec in annotated],
    }


def select_phase2_batch_records(
    ranked_records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    selected_ops: Sequence[Any],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    h_compiled: CompiledPolynomialAction,
    novelty_oracle: Any,
    curvature_oracle: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    tie_break_score_key: str = "phase2_raw_score",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Dispatch phase-2 batch admission without branching in the ADAPT loop."""
    mode = normalize_phase2_batch_selection_mode(getattr(cfg, "batch_selection_mode", PHASE2_BATCH_REDUCED_PLANE))
    if mode == PHASE2_BATCH_REDUCED_PLANE:
        selected, summary = reduced_plane_batch_select(
            ranked_records,
            cfg=cfg,
            selected_ops=selected_ops,
            theta=theta,
            psi_ref=psi_ref,
            psi_state=psi_state,
            h_compiled=h_compiled,
            novelty_oracle=novelty_oracle,
            curvature_oracle=curvature_oracle,
            compiled_cache=compiled_cache,
            pauli_action_cache=pauli_action_cache,
            tie_break_score_key=tie_break_score_key,
        )
        return selected, {"selection_mode": mode, **dict(summary)}
    if mode == PHASE2_BATCH_GREEDY_REDUCED_PLANE:
        return greedy_reduced_plane_batch_select(
            ranked_records,
            cfg=cfg,
            selected_ops=selected_ops,
            theta=theta,
            psi_ref=psi_ref,
            psi_state=psi_state,
            h_compiled=h_compiled,
            novelty_oracle=novelty_oracle,
            curvature_oracle=curvature_oracle,
            compiled_cache=compiled_cache,
            pauli_action_cache=pauli_action_cache,
            tie_break_score_key=tie_break_score_key,
        )
    if mode == PHASE2_BATCH_COMBINATORIAL_REDUCED_PLANE:
        return combinatorial_reduced_plane_batch_select(
            ranked_records,
            cfg=cfg,
            selected_ops=selected_ops,
            theta=theta,
            psi_ref=psi_ref,
            psi_state=psi_state,
            h_compiled=h_compiled,
            novelty_oracle=novelty_oracle,
            curvature_oracle=curvature_oracle,
            compiled_cache=compiled_cache,
            pauli_action_cache=pauli_action_cache,
            tie_break_score_key=tie_break_score_key,
        )
    if mode == "overlap_orthogonal_benchmark":
        return overlap_orthogonal_batch_select(
            ranked_records,
            cfg=cfg,
            tie_break_score_key=tie_break_score_key,
        )
    if mode == "ceo_commuting_benchmark":
        return ceo_commuting_batch_select(
            ranked_records,
            cfg=cfg,
            tie_break_score_key=tie_break_score_key,
        )
    raise ValueError(f"Unsupported phase2 batch selection mode: {mode!r}")


def greedy_batch_select(
    ranked_records: Sequence[Mapping[str, Any]],
    compat_oracle: CompatibilityPenaltyOracle,
    cfg: FullScoreConfig,
    tie_break_score_key: str = "simple_score",
) -> tuple[list[dict[str, Any]], float]:
    def _record_score(rec: Mapping[str, Any], key: str, default: float = float("-inf")) -> float:
        raw = rec.get(key, default)
        if raw is None:
            return float(default)
        return float(raw)

    ranked = sorted(
        [dict(rec) for rec in ranked_records],
        key=lambda rec: (
            -_record_score(rec, "full_v2_score"),
            -_record_score(rec, tie_break_score_key),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        ),
    )
    if not ranked:
        return [], 0.0

    batch: list[dict[str, Any]] = []
    total_penalty = 0.0
    top_score = float(ranked[0].get("full_v2_score", float("-inf")))
    duplicate_generator_skip_count = 0
    for rec in ranked:
        if len(batch) >= int(max(1, cfg.batch_size_cap)):
            break
        rec_identity = _batch_record_generator_identity(rec)
        if any(_batch_record_generator_identity(existing) == rec_identity for existing in batch):
            duplicate_generator_skip_count += 1
            continue
        rec_score = float(rec.get("full_v2_score", float("-inf")))
        if not math.isfinite(rec_score) or rec_score <= 0.0:
            continue
        if batch and rec_score < float(cfg.batch_near_degenerate_ratio) * float(top_score):
            continue
        penalty_total = 0.0
        penalty_breakdown = {
            "support_overlap": 0.0,
            "noncommutation": 0.0,
            "cross_curvature": 0.0,
            "schedule": 0.0,
            "measurement_mismatch": 0.0,
        }
        for existing in batch:
            breakdown = compat_oracle.penalty(rec, existing)
            penalty_total += float(breakdown.get("total", 0.0))
            for key in penalty_breakdown:
                penalty_breakdown[key] += float(breakdown.get(key, 0.0))
        if float(rec_score) - float(penalty_total) <= 0.0 and batch:
            continue
        feat = rec.get("feature")
        updated = dict(rec)
        if isinstance(feat, CandidateFeatures):
            updated["feature"] = _replace_feature(
                feat,
                compatibility_penalty_total=float(penalty_total),
            )
        updated["compatibility_penalty"] = {
            **penalty_breakdown,
            "total": float(penalty_total),
        }
        batch.append(updated)
        total_penalty += float(penalty_total)
        if len(batch) >= int(max(1, cfg.batch_target_size)):
            break
    return batch if batch else [dict(ranked[0])], float(total_penalty)


def build_candidate_features(
    *,
    stage_name: str,
    candidate_label: str,
    candidate_family: str,
    candidate_pool_index: int,
    position_id: int,
    append_position: int,
    positions_considered: list[int],
    gradient_signed: float,
    metric_proxy: float,
    sigma_hat: float,
    refit_window_indices: list[int],
    phase2_geometry_window_indices: Sequence[int] | None = None,
    phase2_geometry_window_policy: str = "legacy_refit_window_alias",
    phase3_geometry_window_indices: Sequence[int] | None = None,
    phase3_geometry_active_post_indices: Sequence[int] | None = None,
    phase3_geometry_window_policy: str | None = None,
    phase3_geometry_window_size: int = 0,
    phase3_response_coordinate_scope: str = "legacy_reopt_coupled_v1",
    phase3_response_coordinate_indices: Sequence[int] | None = None,
    phase3_response_pre_support_count: int | None = None,
    phase3_active_logical_coordinate_count: int | None = None,
    schur_window_indices: Sequence[int] | None = None,
    schur_window_policy: str = "phase3_geometry_refit_window_alias",
    inherited_refit_window_indices: Sequence[int] | None = None,
    active_post_refit_indices: Sequence[int] | None = None,
    optimizer_active_refit_indices: Sequence[int] | None = None,
    compile_cost: CompileCostEstimate,
    measurement_stats: MeasurementCacheStats,
    leakage_penalty: float,
    stage_gate_open: bool,
    leakage_gate_open: bool,
    trough_probe_triggered: bool,
    trough_detected: bool,
    family_repeat_cost: float = 0.0,
    cfg: SimpleScoreConfig,
    cheap_score_cfg: FullScoreConfig | None = None,
    generator_metadata: Mapping[str, Any] | None = None,
    symmetry_spec: Mapping[str, Any] | None = None,
    symmetry_mode: str = "none",
    symmetry_mitigation_mode: str = "off",
    motif_metadata: Mapping[str, Any] | None = None,
    motif_bonus: float = 0.0,
    motif_source: str = "none",
    current_depth: int | None = None,
    max_depth: int | None = None,
    lifetime_cost_mode: str = "off",
    remaining_evaluations_proxy_mode: str = "none",
    controller_snapshot: Mapping[str, Any] | None = None,
) -> CandidateFeatures:
    """Built-in math expression:
    g_hw_lcb = max(|g| - (z_alpha * sigma_hat + b_g_hw + b_g_drift), 0)
    """
    g_abs = float(abs(float(gradient_signed)))
    sigma_hat_nonnegative = float(max(0.0, sigma_hat))
    resolution = _gradient_resolution_components(
        g_abs=float(g_abs),
        sigma_hat=float(sigma_hat_nonnegative),
        z_alpha=float(cfg.z_alpha),
        hardware_resolution_mode=str(getattr(cfg, "hardware_resolution_mode", "ideal")),
        manual_b_g_hw=float(getattr(cfg, "manual_b_g_hw", 0.0)),
        manual_b_g_drift=float(getattr(cfg, "manual_b_g_drift", 0.0)),
    )
    g_lcb = float(resolution["g_lcb_legacy_shot"])
    g_hw_lcb = float(resolution["g_hw_lcb"])
    remaining_eval_proxy = remaining_evaluations_proxy(
        current_depth=current_depth,
        max_depth=max_depth,
        mode=str(remaining_evaluations_proxy_mode),
        controller_snapshot=controller_snapshot,
    )
    controller_snapshot_dict = (
        dict(controller_snapshot)
        if isinstance(controller_snapshot, Mapping)
        else None
    )
    refit_window_indices_norm = _window_int_list(refit_window_indices)
    phase2_geometry_window_indices_norm = _window_int_list(
        phase2_geometry_window_indices,
        fallback=refit_window_indices_norm,
    )
    phase3_geometry_window_indices_norm = _window_int_list(
        phase3_geometry_window_indices,
        fallback=phase2_geometry_window_indices_norm,
    )
    schur_window_indices_norm = _window_int_list(
        schur_window_indices,
        fallback=phase3_geometry_window_indices_norm,
    )
    inherited_refit_window_indices_norm = _window_int_list(
        inherited_refit_window_indices,
        fallback=refit_window_indices_norm,
    )
    active_post_refit_indices_norm = _window_int_list(active_post_refit_indices)
    optimizer_active_refit_indices_norm = _window_int_list(
        optimizer_active_refit_indices,
        fallback=active_post_refit_indices_norm,
    )
    phase3_geometry_active_post_indices_norm = _window_int_list(
        phase3_geometry_active_post_indices,
        fallback=active_post_refit_indices_norm,
    )
    phase3_response_coordinate_indices_norm = _window_int_list(
        phase3_response_coordinate_indices,
        fallback=phase3_geometry_active_post_indices_norm,
    )
    phase3_geometry_window_policy_norm = (
        str(phase3_geometry_window_policy)
        if phase3_geometry_window_policy is not None
        else "legacy_coupled"
    )
    cost_cfg = cheap_score_cfg if cheap_score_cfg is not None else cfg
    candidate_gain_policy = str(
        getattr(
            cost_cfg,
            "phase3_candidate_gain_policy",
            PHASE3_CANDIDATE_GAIN_JOINT_TOTAL_V1,
        )
    )
    if candidate_gain_policy not in PHASE3_CANDIDATE_GAIN_POLICIES:
        raise ValueError(
            "phase3_candidate_gain_policy must be one of "
            f"{sorted(PHASE3_CANDIDATE_GAIN_POLICIES)}."
        )
    proxy_cost = (
        dict(compile_cost.proxy_baseline)
        if isinstance(compile_cost.proxy_baseline, Mapping)
        else {
            "new_pauli_actions": float(compile_cost.new_pauli_actions),
            "new_rotation_steps": float(compile_cost.new_rotation_steps),
            "position_shift_span": float(compile_cost.position_shift_span),
            "refit_active_count": float(compile_cost.refit_active_count),
            "cx_proxy_total": float(compile_cost.cx_proxy_total),
            "sq_proxy_total": float(compile_cost.sq_proxy_total),
            "gate_proxy_total": float(compile_cost.gate_proxy_total),
            "max_pauli_weight": float(compile_cost.max_pauli_weight),
            "proxy_total": float(compile_cost.proxy_total),
            "c_hat_2q": float(compile_cost.c_hat_2q),
            "c_hat_d": float(compile_cost.c_hat_d),
            "c_hat_1q": float(compile_cost.c_hat_1q),
            "c_hat_theta": float(compile_cost.c_hat_theta),
        }
    )
    compile_cost_total = (
        float(compile_cost.penalty_total)
        if compile_cost.penalty_total is not None
        else float(_effective_compile_proxy_total(proxy_cost, cost_cfg))
    )
    depth_cost_value = (
        float(compile_cost.depth_surrogate)
        if compile_cost.depth_surrogate is not None
        else float(_effective_depth_cost(proxy_cost, cost_cfg))
    )
    measurement_groups_cost = float(cost_cfg.measure_groups_weight) * float(measurement_stats.groups_new)
    measurement_shots_cost = float(cost_cfg.measure_shots_weight) * float(measurement_stats.shots_new)
    measurement_reuse_cost = float(cost_cfg.measure_reuse_weight) * float(measurement_stats.reuse_count_cost)
    opt_dim_cost_value = float(cost_cfg.opt_dim_cost_scale) * float(len(refit_window_indices_norm))
    family_repeat_cost_value = float(cost_cfg.family_repeat_cost_scale) * float(family_repeat_cost)
    c_hat_2q = _finite_nonnegative(compile_cost.c_hat_2q)
    c_hat_d = _finite_nonnegative(compile_cost.c_hat_d)
    c_hat_1q = _finite_nonnegative(compile_cost.c_hat_1q)
    c_hat_theta = _finite_nonnegative(compile_cost.c_hat_theta)
    c_hat_shot = _finite_nonnegative(measurement_stats.shot_cost_proxy)
    hardware_lambdas, hardware_lambda_source = resolve_hardware_cost_lambdas(cfg)
    feat = CandidateFeatures(
        stage_name=str(stage_name),
        candidate_label=str(candidate_label),
        candidate_family=str(candidate_family),
        candidate_pool_index=int(candidate_pool_index),
        position_id=int(position_id),
        append_position=int(append_position),
        positions_considered=[int(x) for x in positions_considered],
        g_signed=float(gradient_signed),
        g_abs=float(g_abs),
        g_lcb=float(g_lcb),
        sigma_hat=float(sigma_hat_nonnegative),
        epsilon_g_shot=float(resolution["epsilon_g_shot"]),
        b_g_hw=float(resolution["b_g_hw"]),
        b_g_drift=float(resolution["b_g_drift"]),
        epsilon_g_res=float(resolution["epsilon_g_res"]),
        g_hw_lcb=float(g_hw_lcb),
        g_lcb_legacy_shot=float(resolution["g_lcb_legacy_shot"]),
        hardware_resolution_mode=str(resolution["hardware_resolution_mode"]),
        hardware_resolution_source=str(resolution["hardware_resolution_source"]),
        F_metric=float(max(0.0, metric_proxy)),
        metric_proxy=float(max(0.0, metric_proxy)),
        novelty=None,
        curvature_mode=(
            "phase1_first_order_no_energy_curvature_v1"
            if normalize_phase1_energy_model(
                getattr(
                    cfg,
                    "phase1_energy_model",
                    PHASE1_ENERGY_MODEL_LEGACY_LAMBDA_F_QUADRATIC_V1,
                )
            )
            == PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1
            else "lambda_F_metric_proxy_only"
        ),
        novelty_mode="none",
        refit_window_indices=[int(i) for i in refit_window_indices_norm],
        refit_window_basis="old_pre_geometry_alias",
        phase2_geometry_window_indices=[int(i) for i in phase2_geometry_window_indices_norm],
        phase2_geometry_window_policy=str(phase2_geometry_window_policy),
        phase3_geometry_window_policy=str(phase3_geometry_window_policy_norm),
        phase3_geometry_window_size=int(max(0, int(phase3_geometry_window_size))),
        phase3_geometry_refit_window_indices=[int(i) for i in phase3_geometry_window_indices_norm],
        phase3_geometry_active_post_indices=[int(i) for i in phase3_geometry_active_post_indices_norm],
        phase3_response_coordinate_scope=str(phase3_response_coordinate_scope),
        phase3_response_coordinate_indices=[
            int(i) for i in phase3_response_coordinate_indices_norm
        ],
        phase3_response_pre_support_count=int(
            len(phase3_response_coordinate_indices_norm)
            if phase3_response_pre_support_count is None
            else phase3_response_pre_support_count
        ),
        phase3_active_logical_coordinate_count=int(
            max(0, len(phase3_response_coordinate_indices_norm) - 1)
            if phase3_active_logical_coordinate_count is None
            else phase3_active_logical_coordinate_count
        ),
        phase3_candidate_gain_policy=str(candidate_gain_policy),
        schur_window_indices=[int(i) for i in schur_window_indices_norm],
        schur_window_policy=str(schur_window_policy),
        inherited_refit_window_indices=[int(i) for i in inherited_refit_window_indices_norm],
        active_post_refit_indices=[int(i) for i in active_post_refit_indices_norm],
        selection_inherited_old_indices=[int(i) for i in inherited_refit_window_indices_norm],
        optimizer_active_refit_indices=[int(i) for i in optimizer_active_refit_indices_norm],
        optimizer_active_refit_count=int(len(optimizer_active_refit_indices_norm)),
        w3_wopt_decoupled=bool(
            phase3_geometry_window_policy_norm == "fixed_local_v1"
            or tuple(int(i) for i in phase3_geometry_active_post_indices_norm) != tuple(int(i) for i in optimizer_active_refit_indices_norm)
        ),
        compiled_position_cost_proxy={str(k): float(v) for k, v in proxy_cost.items()},
        measurement_cache_stats={
            "groups_total": float(measurement_stats.groups_total),
            "groups_reused": float(measurement_stats.groups_reused),
            "groups_new": float(measurement_stats.groups_new),
            "shots_reused": float(measurement_stats.shots_reused),
            "shots_new": float(measurement_stats.shots_new),
            "reuse_count_cost": float(measurement_stats.reuse_count_cost),
            "shot_cost_proxy": float(measurement_stats.shot_cost_proxy),
            "new_group_coeff_l2_sum": float(measurement_stats.new_group_coeff_l2_sum),
            "sigma_star": float(measurement_stats.sigma_star),
            "new_group_term_count": float(measurement_stats.new_group_term_count),
        },
        leakage_penalty=float(max(0.0, leakage_penalty)),
        stage_gate_open=bool(stage_gate_open),
        leakage_gate_open=bool(leakage_gate_open),
        trough_probe_triggered=bool(trough_probe_triggered),
        trough_detected=bool(trough_detected),
        simple_score=None,
        score_version=str(cfg.score_version),
        cheap_score=None,
        cheap_score_version=str(cfg.score_version),
        cheap_metric_proxy=float(max(0.0, metric_proxy)),
        cheap_benefit_proxy=None,
        cheap_burden_total=None,
        depth_cost=float(depth_cost_value),
        new_group_cost=float(measurement_groups_cost),
        new_shot_cost=float(measurement_shots_cost),
        opt_dim_cost=float(opt_dim_cost_value),
        reuse_count_cost=float(measurement_reuse_cost),
        c_hat_2q=float(c_hat_2q),
        c_hat_d=float(c_hat_d),
        c_hat_1q=float(c_hat_1q),
        c_hat_theta=float(c_hat_theta),
        c_hat_shot=float(c_hat_shot),
        hardware_cost_lambdas={str(k): float(v) for k, v in hardware_lambdas.items()},
        hardware_cost_lambda_source=str(hardware_lambda_source),
        hardware_cost_source=str(compile_cost.hardware_cost_source),
        family_repeat_cost=float(family_repeat_cost_value),
        actual_fallback_mode="simple_v1_only",
        generator_id=(
            str(generator_metadata.get("generator_id"))
            if isinstance(generator_metadata, Mapping) and generator_metadata.get("generator_id") is not None
            else None
        ),
        template_id=(
            str(generator_metadata.get("template_id"))
            if isinstance(generator_metadata, Mapping) and generator_metadata.get("template_id") is not None
            else None
        ),
        is_macro_generator=bool(generator_metadata.get("is_macro_generator", False)) if isinstance(generator_metadata, Mapping) else False,
        parent_generator_id=(
            str(generator_metadata.get("parent_generator_id"))
            if isinstance(generator_metadata, Mapping) and generator_metadata.get("parent_generator_id") is not None
            else None
        ),
        generator_metadata=(dict(generator_metadata) if isinstance(generator_metadata, Mapping) else None),
        symmetry_spec=(dict(symmetry_spec) if isinstance(symmetry_spec, Mapping) else None),
        symmetry_mode=str(symmetry_mode),
        symmetry_mitigation_mode=str(symmetry_mitigation_mode),
        motif_metadata=(dict(motif_metadata) if isinstance(motif_metadata, Mapping) else None),
        motif_bonus=float(max(0.0, motif_bonus)),
        motif_source=str(motif_source),
        remaining_evaluations_proxy=float(remaining_eval_proxy),
        remaining_evaluations_proxy_mode=str(remaining_evaluations_proxy_mode),
        lifetime_cost_mode=str(lifetime_cost_mode),
        lifetime_weight_components={
            "remaining_evaluations_proxy": float(remaining_eval_proxy),
            "n_rem_hat": (
                float(controller_snapshot_dict.get("n_rem_hat", remaining_eval_proxy))
                if isinstance(controller_snapshot_dict, Mapping)
                else float(remaining_eval_proxy)
            ),
            "useful_horizon": (
                float(controller_snapshot_dict.get("useful_horizon", remaining_eval_proxy))
                if isinstance(controller_snapshot_dict, Mapping)
                else float(remaining_eval_proxy)
            ),
            "H_t": (
                float(controller_snapshot_dict.get("H_t", remaining_eval_proxy))
                if isinstance(controller_snapshot_dict, Mapping)
                else float(remaining_eval_proxy)
            ),
        },
        placeholder_hooks={
            "novelty_oracle": False,
            "curvature_oracle": False,
            "full_v2_score": False,
            "qn_spsa_refresh": False,
            "motif_metadata": False,
            "symmetry_metadata": bool(isinstance(symmetry_spec, Mapping)),
            "backend_compile_oracle": bool(str(compile_cost.source_mode) != "proxy"),
        },
        compile_cost_source=str(compile_cost.source_mode),
        compile_cost_total=float(compile_cost_total),
        compile_gate_open=bool(compile_cost.compile_gate_open),
        compile_failure_reason=(
            None if compile_cost.failure_reason is None else str(compile_cost.failure_reason)
        ),
        compiled_position_cost_backend=(
            None
            if str(compile_cost.source_mode) == "proxy"
            else {
                "selected_backend_name": compile_cost.selected_backend_name,
                "selected_resolution_kind": compile_cost.selected_resolution_kind,
                "aggregation_mode": str(compile_cost.aggregation_mode),
                "target_backend_names": [str(x) for x in compile_cost.target_backend_names],
                "successful_target_count": int(compile_cost.successful_target_count),
                "failed_target_count": int(compile_cost.failed_target_count),
                "candidate_label": (
                    None
                    if not isinstance(compile_cost.selected_backend_row, Mapping)
                    else compile_cost.selected_backend_row.get(
                        "candidate_label"
                    )
                ),
                "position_id": (
                    None
                    if not isinstance(compile_cost.selected_backend_row, Mapping)
                    else compile_cost.selected_backend_row.get("position_id")
                ),
                "candidate_polynomial_sha256": (
                    None
                    if not isinstance(compile_cost.selected_backend_row, Mapping)
                    else compile_cost.selected_backend_row.get(
                        "candidate_polynomial_sha256"
                    )
                ),
                "compile_cache_identity": (
                    None
                    if (
                        not isinstance(
                            compile_cost.selected_backend_row,
                            Mapping,
                        )
                        or not isinstance(
                            compile_cost.selected_backend_row.get(
                                "compile_cache_identity"
                            ),
                            Mapping,
                        )
                    )
                    else dict(
                        compile_cost.selected_backend_row[
                            "compile_cache_identity"
                        ]
                    )
                ),
                "compile_cache_identity_sha256": (
                    None
                    if not isinstance(
                        compile_cost.selected_backend_row,
                        Mapping,
                    )
                    else compile_cost.selected_backend_row.get(
                        "compile_cache_identity_sha256"
                    )
                ),
                "base_structure_key": (
                    None
                    if not isinstance(compile_cost.selected_backend_row, Mapping)
                    else compile_cost.selected_backend_row.get(
                        "base_structure_key"
                    )
                ),
                "trial_structure_key": (
                    None
                    if not isinstance(compile_cost.selected_backend_row, Mapping)
                    else compile_cost.selected_backend_row.get(
                        "trial_structure_key"
                    )
                ),
                "base_initial_layout": (
                    None
                    if not isinstance(compile_cost.selected_backend_row, Mapping)
                    else compile_cost.selected_backend_row.get(
                        "base_initial_layout"
                    )
                ),
                "trial_initial_layout": (
                    None
                    if not isinstance(compile_cost.selected_backend_row, Mapping)
                    else compile_cost.selected_backend_row.get(
                        "trial_initial_layout"
                    )
                ),
                "base_logical_to_physical": (
                    []
                    if not isinstance(compile_cost.selected_backend_row, Mapping)
                    else list(
                        compile_cost.selected_backend_row.get(
                            "base_logical_to_physical", []
                        )
                    )
                ),
                "trial_logical_to_physical": (
                    []
                    if not isinstance(compile_cost.selected_backend_row, Mapping)
                    else list(
                        compile_cost.selected_backend_row.get(
                            "trial_logical_to_physical", []
                        )
                    )
                ),
                "base_trial_layout_coupling_policy": (
                    None
                    if not isinstance(compile_cost.selected_backend_row, Mapping)
                    else compile_cost.selected_backend_row.get(
                        "base_trial_layout_coupling_policy"
                    )
                ),
                "raw_delta_compiled_count_1q": (
                    None
                    if not isinstance(compile_cost.selected_backend_row, Mapping)
                    else compile_cost.selected_backend_row.get(
                        "raw_delta_compiled_count_1q"
                    )
                ),
                "delta_compiled_count_1q": (
                    None
                    if not isinstance(compile_cost.selected_backend_row, Mapping)
                    else compile_cost.selected_backend_row.get(
                        "delta_compiled_count_1q"
                    )
                ),
                "raw_delta_compiled_count_2q": compile_cost.raw_delta_compiled_count_2q,
                "delta_compiled_count_2q": compile_cost.delta_compiled_count_2q,
                "raw_delta_compiled_depth": compile_cost.raw_delta_compiled_depth,
                "delta_compiled_depth": compile_cost.delta_compiled_depth,
                "raw_delta_compiled_depth_2q": compile_cost.raw_delta_compiled_depth_2q,
                "delta_compiled_depth_2q": compile_cost.delta_compiled_depth_2q,
                "raw_delta_compiled_size": compile_cost.raw_delta_compiled_size,
                "delta_compiled_size": compile_cost.delta_compiled_size,
                "delta_compiled_cx_count": compile_cost.delta_compiled_cx_count,
                "delta_compiled_ecr_count": compile_cost.delta_compiled_ecr_count,
                "base_compiled_count_2q": compile_cost.base_compiled_count_2q,
                "base_compiled_depth": compile_cost.base_compiled_depth,
                "base_compiled_size": compile_cost.base_compiled_size,
                "trial_compiled_count_2q": compile_cost.trial_compiled_count_2q,
                "trial_compiled_depth": compile_cost.trial_compiled_depth,
                "trial_compiled_size": compile_cost.trial_compiled_size,
                "c_hat_2q": float(compile_cost.c_hat_2q),
                "c_hat_d": float(compile_cost.c_hat_d),
                "c_hat_1q": float(compile_cost.c_hat_1q),
                "c_hat_theta": float(compile_cost.c_hat_theta),
                "hardware_cost_source": str(compile_cost.hardware_cost_source),
                "negative_delta_reward_enabled": bool(
                    isinstance(compile_cost.selected_backend_row, Mapping)
                    and compile_cost.selected_backend_row.get(
                        "negative_delta_reward_enabled"
                    )
                    is True
                ),
                }
        ),
        phase_score_components={},
        phase_cost_components={},
        controller_snapshot=controller_snapshot_dict,
        window_origin="legacy",
        window_new_indices=[int(i) for i in refit_window_indices_norm],
        window_age_indices=[],
        phase1_shortlisted=False,
        phase2_shortlisted=False,
        phase3_shortlisted=False,
        phase3_duplicate_penalty=0.0,
    )
    phase1_payload = phase1_score_payload(feat, cfg)
    score = float(phase1_payload["active_score"])
    hardware_payload = _hardware_cost_denominator_payload(feat, cfg)
    phase_cost_components = {
        "compile_proxy": float(compile_cost_total),
        "compile_cx_proxy_weight": float(cost_cfg.compile_cx_proxy_weight),
        "compile_sq_proxy_weight": float(cost_cfg.compile_sq_proxy_weight),
        "compile_rotation_step_weight": float(cost_cfg.compile_rotation_step_weight),
        "compile_position_shift_weight": float(cost_cfg.compile_position_shift_weight),
        "compile_refit_active_weight": float(cost_cfg.compile_refit_active_weight),
        "measurement_groups_new_raw": float(measurement_stats.groups_new),
        "measurement_shots_new_raw": float(measurement_stats.shots_new),
        "measurement_reuse_cost_raw": float(measurement_stats.reuse_count_cost),
        "measurement_groups_new": float(measurement_groups_cost),
        "measurement_shots_new": float(measurement_shots_cost),
        "measurement_reuse_cost": float(measurement_reuse_cost),
        "measurement_shot_cost_proxy": float(measurement_stats.shot_cost_proxy),
        "measurement_new_group_coeff_l2_sum": float(measurement_stats.new_group_coeff_l2_sum),
        "measurement_sigma_star": float(measurement_stats.sigma_star),
        "opt_dim_cost": float(opt_dim_cost_value),
        "family_repeat_cost": float(family_repeat_cost_value),
        "leakage_penalty": float(max(0.0, leakage_penalty)),
        "c_hat_2q": float(c_hat_2q),
        "c_hat_d": float(c_hat_d),
        "c_hat_1q": float(c_hat_1q),
        "c_hat_theta": float(c_hat_theta),
        "c_hat_shot": float(c_hat_shot),
        "lambda_2q": float(hardware_payload["lambdas"]["2q"]),
        "lambda_d": float(hardware_payload["lambdas"]["d"]),
        "lambda_1q": float(hardware_payload["lambdas"]["1q"]),
        "lambda_theta": float(hardware_payload["lambdas"]["theta"]),
        "lambda_shot": float(hardware_payload["lambdas"]["shot"]),
        "hardware_cost_lambda_source": str(hardware_payload["lambda_source"]),
        "hardware_cost_excess_sum": float(hardware_payload["hardware_cost_excess_sum"]),
        "hardware_cost_denominator": float(hardware_payload["hardware_cost_denominator"]),
        "burden_total": float(hardware_payload["hardware_cost_excess_sum"]),
    }
    phase1_F_raw = float(
        max(
            float(max(0.0, getattr(cfg, "metric_floor", 0.0))),
            float(max(0.0, float(metric_proxy))),
        )
    )
    phase1_delta_e_tr_hw = float(phase1_payload["trust_region_gain"])
    phase_score_components = {
        "phase1_energy_model": str(phase1_payload["phase1_energy_model"]),
        "phase1_lambda_f_curvature_proxy_applied": bool(
            phase1_payload["phase1_lambda_f_curvature_proxy_applied"]
        ),
        "phase1_gradient_abs": float(g_abs),
        "epsilon_g_shot": float(resolution["epsilon_g_shot"]),
        "b_g_hw": float(resolution["b_g_hw"]),
        "b_g_drift": float(resolution["b_g_drift"]),
        "epsilon_g_res": float(resolution["epsilon_g_res"]),
        "g_hw_lcb": float(g_hw_lcb),
        "g_lcb_legacy_shot": float(resolution["g_lcb_legacy_shot"]),
        "phase1_DeltaE1_TR_hw": float(phase1_delta_e_tr_hw),
        "phase1_legacy_simple_score": float(phase1_payload["legacy_simple_score"]),
        "phase1_trust_region_score": float(phase1_payload["trust_region_score"]),
        "phase1_active_score": float(score),
        "phase1_rho": float(phase1_payload["rho"]),
        "phase1_score": float(score),
    }
    feat = _replace_feature(
        feat,
        simple_score=float(score),
        cheap_score=float(score),
        cheap_score_version=str(cfg.score_version),
        cheap_metric_proxy=float(max(0.0, metric_proxy)),
        cheap_benefit_proxy=float(phase1_delta_e_tr_hw),
        cheap_burden_total=float(hardware_payload["hardware_cost_denominator"]),
        phase1_score_mode=str(phase1_payload["mode"]),
        phase1_active_score=float(score),
        phase1_legacy_simple_score=float(phase1_payload["legacy_simple_score"]),
        phase1_trust_region_gain=float(phase1_delta_e_tr_hw),
        phase1_trust_region_score=float(phase1_payload["trust_region_score"]),
        phase1_rho=float(phase1_payload["rho"]),
        phase1_burden_total=float(hardware_payload["hardware_cost_denominator"]),
        phase1_energy_model=str(phase1_payload["phase1_energy_model"]),
        phase1_lambda_f_proxy_applied=bool(
            phase1_payload["phase1_lambda_f_curvature_proxy_applied"]
        ),
        hardware_cost_excess_sum=float(hardware_payload["hardware_cost_excess_sum"]),
        hardware_cost_denominator=float(hardware_payload["hardware_cost_denominator"]),
        hardware_cost_lambdas={str(k): float(v) for k, v in hardware_payload["lambdas"].items()},
        hardware_cost_lambda_source=str(hardware_payload["lambda_source"]),
        selector_score=float(score),
        selector_burden=float(hardware_payload["hardware_cost_denominator"]),
        phase_score_components=dict(phase_score_components),
        phase_cost_components=dict(phase_cost_components),
    )
    return feat
