"""Static SNAKE route identity helpers.

This module intentionally separates *route identity* from operator pool,
optimizer lane, and evaluation convention.  In project language:

* Route A is the current static SNAKE controller route.
* ``full_meta`` is the default broad operator pool for canonical Route A runs.
* SPSA/POWELL/QNSPSA are optimizer lanes, not separate routes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

ROUTE_ID_UNSPECIFIED = "unspecified"
ROUTE_ID_A = "route_a"
ROUTE_ID_B_LEGACY_PAIRWISE = "route_b_legacy_pairwise"
ROUTE_ID_C = "route_c"
ROUTE_ID_CHOICES = (
    ROUTE_ID_UNSPECIFIED,
    ROUTE_ID_A,
    ROUTE_ID_B_LEGACY_PAIRWISE,
    ROUTE_ID_C,
)
FIRST_CLASS_ROUTE_ID_CHOICES = (
    ROUTE_ID_UNSPECIFIED,
    ROUTE_ID_A,
)
LEGACY_ROUTE_ID_CHOICES = (
    ROUTE_ID_B_LEGACY_PAIRWISE,
    ROUTE_ID_C,
)

ROUTE_A_VERSION = "route_a_static_snake_algebraic_nested_batch_v2"
ROUTE_A_META_FEATURE_VERSION = "route_a_static_snake_algebraic_nested_meta_features_v1"
ROUTE_A_PAPER_I_PRODUCTION_VERSION = "route_a_static_snake_paper_i_production_v1"
ROUTE_B_VERSION = "route_b_legacy_pairwise_algebraic_nested_batch_v2"
ROUTE_C_VERSION = "route_c_plateau_acquisition_v1"
ROUTE_IDENTITY_SCHEMA = "static_snake_route_identity_v1"
ROUTE_VARIANT_OBSERVED_COMPONENT_KEYS = (
    "static_lane_route",
    "static_lane_route_is_route_identity",
    "route_variant_id",
)
STATIC_META_FEATURE_PROFILE_OFF = "off"
STATIC_META_FEATURE_PROFILE_SAFE_CORE_V1 = "safe_core_v1"
STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1 = "paper_i_production_v1"
STATIC_META_FEATURE_PROFILE_CHOICES = (
    STATIC_META_FEATURE_PROFILE_OFF,
    STATIC_META_FEATURE_PROFILE_SAFE_CORE_V1,
    STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
)

ROUTE_A_REQUIRED_COMPONENTS: dict[str, Any] = {
    "base_pool_key": "full_meta",
    "continuation_mode": "phase3_v1",
    "phase2_novelty_mode": "collective_span_v1",
    "phase3_selector_policy": "algebraic_nested_v1",
    "phase3_selector_geometry_mode": "reduced",
    "algebraic_shortlisting_enabled": True,
    "hardware_resolution_schema": "gradient_resolution_v1",
    "hardware_resolution_mode": "ideal",
    "phase2_raw_score_formula": "DeltaE_TR_raw * N2 / (1 + K2)",
    "canonical_score_formula": "DeltaE_TR * N3 / (1 + K3)",
    "primary_selector_score_key": "full_v2_score",
    "auxiliary_terms_primary_mode": "tie_break_only",
    # Internal code spelling: "off" means the Phase-3 novelty ablation is off,
    # so current Phase-3 novelty remains active.
    "phase3_novelty_ablation_mode": "off",
    "phase3_window_relaxation_mode": "reduced",
    "phase3_enable_batching": True,
    "phase3_batch_selection_mode": "reduced_plane",
    "phase3_batch_prefilter_mode": "off",
    "phase3_batch_order_selection_mode": "finite_step_v1",
    "phase3_nested_window_application": "composed_batch_window_v1",
    "phase1_prune_enabled": True,
    "phase1_prune_policy": "recoverability_ladder_v1",
    "phase1_prune_mode": "both",
}

ROUTE_B_REQUIRED_COMPONENTS: dict[str, Any] = {
    **ROUTE_A_REQUIRED_COMPONENTS,
    "phase2_novelty_mode": "legacy_pairwise_v1",
    "phase2_raw_score_formula": "DeltaE_TR_raw * N2_pairwise / (1 + K2)",
}

ROUTE_C_REQUIRED_COMPONENTS: dict[str, Any] = {
    **ROUTE_A_REQUIRED_COMPONENTS,
    "phase3_plateau_acquisition_mode": "novelty_cost_v1",
    "phase3_plateau_acquisition_score": "log_volume_v1",
    "phase3_plateau_score_formula": "log(1 + sigma_perp_lambda / lambda_vol) / (1 + K3)",
    "phase3_plateau_duplicate_policy": "block_exact_position_v1",
}

ROUTE_A_SAFE_CORE_OPTIONAL_COMPONENTS: dict[str, str] = {
    "phase3_enable_batching": "sampled_feature_bundle",
    "phase1_prune_enabled": "sampled_feature_bundle",
}
ROUTE_A_SAFE_CORE_PRUNE_DEPENDENT_COMPONENTS = (
    "phase1_prune_policy",
    "phase1_prune_mode",
)
ROUTE_A_PAPER_I_PRODUCTION_OPTIONAL_COMPONENTS: dict[str, str] = {
    "phase3_enable_batching": "paper_i_production_optuna_toggle",
    "phase3_batch_selection_mode": "paper_i_batch_algorithm_choice",
}

ROUTE_A_VARIANT_REQUIRED_COMPONENT_OVERRIDES: dict[str, dict[str, Any]] = {
    "route_a_molecular_restricted_physical_operator_lanes_v1": {
        "base_pool_key": "uccsd",
    },
}

_OBSERVED_ALIASES: dict[str, tuple[str, ...]] = {
    "base_pool_key": (
        "base_pool_key",
        "route_base_pool_key",
        "pool_key",
        "adapt_pool",
        "adapt_pool_requested",
    ),
    "continuation_mode": ("continuation_mode", "adapt_continuation_mode"),
    "phase2_novelty_mode": ("phase2_novelty_mode",),
    "phase3_selector_policy": ("phase3_selector_policy",),
    "phase3_selector_geometry_mode": ("phase3_selector_geometry_mode", "selector_geometry_mode"),
    "algebraic_shortlisting_enabled": ("algebraic_shortlisting_enabled",),
    "hardware_resolution_schema": ("hardware_resolution_schema", "hardware_gradient_resolution_schema"),
    "hardware_resolution_mode": ("hardware_resolution_mode",),
    "phase2_raw_score_formula": ("phase2_raw_score_formula",),
    "canonical_score_formula": ("canonical_score_formula", "phase3_score_formula"),
    "primary_selector_score_key": ("primary_selector_score_key", "phase3_primary_selector_score_key"),
    "auxiliary_terms_primary_mode": ("auxiliary_terms_primary_mode", "phase3_auxiliary_score_mode"),
    "phase3_novelty_ablation_mode": ("phase3_novelty_ablation_mode",),
    "phase3_window_relaxation_mode": ("phase3_window_relaxation_mode",),
    "phase3_enable_batching": ("phase3_enable_batching", "phase2_enable_batching", "batching_enabled"),
    "phase3_batch_selection_mode": ("phase3_batch_selection_mode", "phase2_batch_selection_mode", "batch_selection_mode"),
    "phase3_batch_prefilter_mode": ("phase3_batch_prefilter_mode", "batch_prefilter_mode"),
    "phase3_batch_order_selection_mode": (
        "phase3_batch_order_selection_mode",
        "batch_order_selection_mode",
    ),
    "phase3_nested_window_application": (
        "phase3_nested_window_application",
        "nested_window_application",
    ),
    "phase1_prune_enabled": ("phase1_prune_enabled",),
    "phase1_prune_policy": ("phase1_prune_policy", "prune_policy"),
    "phase1_prune_mode": ("phase1_prune_mode", "prune_mode"),
    "phase1_prune_amplitude_witness_required": (
        "phase1_prune_amplitude_witness_required",
        "amplitude_witness_required",
    ),
    "phase3_plateau_acquisition_mode": ("phase3_plateau_acquisition_mode",),
    "phase3_plateau_acquisition_score": ("phase3_plateau_acquisition_score",),
    "phase3_plateau_score_formula": ("phase3_plateau_score_formula",),
    "phase3_plateau_duplicate_policy": ("phase3_plateau_duplicate_policy",),
    "phase3_plateau_unlock_margin": ("phase3_plateau_unlock_margin",),
    "meta_feature_profile": ("meta_feature_profile", "static_meta_feature_profile"),
}


@dataclass(frozen=True)
class RouteIdentityValidation:
    """Validation result for static SNAKE route metadata."""

    schema: str
    route_id: str
    route_version: str | None
    canonical_snake_eligible: bool
    evidence_role: str
    valid: bool
    noncanonical_reasons: tuple[str, ...]
    observed_components: Mapping[str, Any]
    required_components: Mapping[str, Any]
    meta_feature_profile: str
    optional_components: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "route_id": self.route_id,
            "route_version": self.route_version,
            "canonical_snake_eligible": bool(self.canonical_snake_eligible),
            "evidence_role": self.evidence_role,
            "valid": bool(self.valid),
            "noncanonical_reasons": list(self.noncanonical_reasons),
            "observed_components": dict(self.observed_components),
            "required_components": dict(self.required_components),
            "meta_feature_profile": self.meta_feature_profile,
            "optional_components": dict(self.optional_components),
        }


@dataclass(frozen=True, kw_only=True)
class StaticRouteIdentityConfig:
    declared_route_id: Any
    optimizer_lane: Any
    pool_key: Any
    pool_key_input: Any
    hh_staged_continuation: bool
    static_meta_feature_profile: Any
    continuation_mode: Any
    phase2_novelty_mode: Any
    phase3_selector_policy: Any
    phase3_selector_geometry_mode: Any
    algebraic_shortlisting_enabled: bool
    hardware_resolution_mode: Any
    phase1_score_mode: Any
    phase1_score_formula: Any
    phase1_active_score_key: Any
    phase1_rho: float
    phase2_selector_gain_mode: Any
    phase2_raw_score_formula: Any
    canonical_score_formula: Any
    primary_selector_score_key: Any
    auxiliary_terms_primary_mode: Any
    phase3_novelty_ablation_mode: Any
    phase3_window_relaxation_mode: Any
    adapt_child_pool_expansion_mode: Any
    adapt_child_pool_expansion_symmetry_policy: Any
    adapt_child_pool_expansion_source: Any
    shared_pauli_pool_mode: Any
    shared_pauli_pool_symmetry_policy: Any
    shared_pauli_pool_symmetry_gate_enforced: Any
    shared_pauli_pool_ordered_pool_hash: Any
    shared_pauli_pool_expanded_pool_term_count: Any
    phase3_plateau_acquisition_mode: Any
    phase3_plateau_acquisition_score: Any
    phase3_plateau_score_formula: Any
    phase3_plateau_duplicate_policy: Any
    phase3_plateau_unlock_margin: float
    phase3_plateau_lambda_vol: float
    phase3_plateau_sigma_min: float
    phase3_plateau_nu_min: float
    phase3_plateau_volume_min: float
    phase3_plateau_failed_family_patience: int
    phase3_plateau_trial_optimizer: Any
    phase3_plateau_trial_qngd_maxiter: int
    phase3_plateau_seed_probe_mode: Any
    phase3_plateau_seed_probe_count: int
    phase3_plateau_seed_probe_radius: float
    phase3_plateau_seed_probe_enabled: bool
    phase3_plateau_seed_probe_seed: Any
    phase3_enable_batching: bool
    phase3_batch_selection_mode: Any
    phase3_batch_prefilter_mode: Any
    phase3_batch_order_selection_mode: Any
    phase3_nested_window_application: Any
    phase1_prune_enabled: bool
    phase1_prune_policy: Any
    phase1_prune_mode: Any
    phase1_prune_amplitude_witness_required: bool
    static_lane_route: Any = "algebraic"
    static_lane_route_is_route_identity: bool = True
    route_variant_id: Any = None


def normalize_static_route_id(value: Any, *, default: str = ROUTE_ID_UNSPECIFIED) -> str:
    key = str(default if value is None or value == "" else value).strip().lower().replace("-", "_")
    aliases = {
        "a": ROUTE_ID_A,
        "routea": ROUTE_ID_A,
        "current": ROUTE_ID_A,
        "current_snake": ROUTE_ID_A,
        "newest": ROUTE_ID_A,
        "b": ROUTE_ID_B_LEGACY_PAIRWISE,
        "routeb": ROUTE_ID_B_LEGACY_PAIRWISE,
        "legacy_pairwise": ROUTE_ID_B_LEGACY_PAIRWISE,
        "legacy_pairwise_v1": ROUTE_ID_B_LEGACY_PAIRWISE,
        "c": ROUTE_ID_C,
        "routec": ROUTE_ID_C,
        "plateau": ROUTE_ID_C,
        "plateau_acquisition": ROUTE_ID_C,
        "plateau_acquisition_v1": ROUTE_ID_C,
        "none": ROUTE_ID_UNSPECIFIED,
        "unknown": ROUTE_ID_UNSPECIFIED,
    }
    key = aliases.get(key, key)
    if key not in ROUTE_ID_CHOICES:
        raise ValueError(f"static route id must be one of {ROUTE_ID_CHOICES}; got {value!r}")
    return key


def normalize_static_meta_feature_profile(value: Any, *, default: str = STATIC_META_FEATURE_PROFILE_OFF) -> str:
    key = str(default if value is None or value == "" else value).strip().lower().replace("-", "_")
    aliases = {
        "none": STATIC_META_FEATURE_PROFILE_OFF,
        "strict": STATIC_META_FEATURE_PROFILE_OFF,
        "false": STATIC_META_FEATURE_PROFILE_OFF,
        "0": STATIC_META_FEATURE_PROFILE_OFF,
        "safe": STATIC_META_FEATURE_PROFILE_SAFE_CORE_V1,
        "safe_core": STATIC_META_FEATURE_PROFILE_SAFE_CORE_V1,
        "safe_core_v1": STATIC_META_FEATURE_PROFILE_SAFE_CORE_V1,
        "meta": STATIC_META_FEATURE_PROFILE_SAFE_CORE_V1,
        "paper_i": STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
        "paper_i_production": STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
        "paper_i_prod": STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
        "paper_i_prod_v1": STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
        "production": STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
        "prod": STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
    }
    key = aliases.get(key, key)
    if key not in STATIC_META_FEATURE_PROFILE_CHOICES:
        raise ValueError(
            "static meta-feature profile must be one of "
            f"{STATIC_META_FEATURE_PROFILE_CHOICES}; got {value!r}"
        )
    return key


def route_contract(route_id: Any) -> dict[str, Any]:
    route_key = normalize_static_route_id(route_id)
    if route_key == ROUTE_ID_A:
        return {
            "schema": ROUTE_IDENTITY_SCHEMA,
            "route_id": ROUTE_ID_A,
            "route_version": ROUTE_A_VERSION,
            "canonical_snake_eligible": True,
            "evidence_role": "canonical_current_route",
            "required_components": dict(ROUTE_A_REQUIRED_COMPONENTS),
        }
    if route_key == ROUTE_ID_B_LEGACY_PAIRWISE:
        return {
            "schema": ROUTE_IDENTITY_SCHEMA,
            "route_id": ROUTE_ID_B_LEGACY_PAIRWISE,
            "route_version": ROUTE_B_VERSION,
            "canonical_snake_eligible": False,
            "evidence_role": "legacy_pairwise_control",
            "required_components": dict(ROUTE_B_REQUIRED_COMPONENTS),
        }
    if route_key == ROUTE_ID_C:
        return {
            "schema": ROUTE_IDENTITY_SCHEMA,
            "route_id": ROUTE_ID_C,
            "route_version": ROUTE_C_VERSION,
            "canonical_snake_eligible": False,
            "evidence_role": "route_c_plateau_acquisition_foundation",
            "required_components": dict(ROUTE_C_REQUIRED_COMPONENTS),
        }
    return {
        "schema": ROUTE_IDENTITY_SCHEMA,
        "route_id": ROUTE_ID_UNSPECIFIED,
        "route_version": None,
        "canonical_snake_eligible": False,
        "evidence_role": "undeclared_or_historical",
        "required_components": {},
    }


def _observed_alias_values(observed: Mapping[str, Any], key: str) -> tuple[tuple[str, Any], ...]:
    values: list[tuple[str, Any]] = []
    for alias in _OBSERVED_ALIASES.get(key, (key,)):
        if alias not in observed:
            continue
        value = observed[alias]
        if value is None or value == "":
            continue
        values.append((str(alias), value))
    return tuple(values)


def _get_observed_component(observed: Mapping[str, Any], key: str) -> Any:
    alias_values = _observed_alias_values(observed, key)
    if not alias_values:
        return None
    return alias_values[0][1]


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _component_matches(observed: Any, required: Any) -> bool:
    observed_token = _normalized_component_token(observed, required)
    required_token = _normalized_component_token(required, required)
    return observed_token is not None and required_token is not None and observed_token == required_token


def _route_a_safe_core_required_components(observed: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        key: value
        for key, value in ROUTE_A_REQUIRED_COMPONENTS.items()
        if key not in ROUTE_A_SAFE_CORE_OPTIONAL_COMPONENTS
        and key not in ROUTE_A_SAFE_CORE_PRUNE_DEPENDENT_COMPONENTS
    }
    prune_enabled = _normalize_bool(_get_observed_component(observed, "phase1_prune_enabled"))
    if prune_enabled is not False:
        for key in ROUTE_A_SAFE_CORE_PRUNE_DEPENDENT_COMPONENTS:
            required[key] = ROUTE_A_REQUIRED_COMPONENTS[key]
    return required


def _route_a_paper_i_production_required_components() -> dict[str, Any]:
    return {
        key: value
        for key, value in ROUTE_A_REQUIRED_COMPONENTS.items()
        if key not in ROUTE_A_PAPER_I_PRODUCTION_OPTIONAL_COMPONENTS
    }


def _route_required_components(
    route_key: str,
    observed: Mapping[str, Any],
    *,
    meta_feature_profile: str,
) -> tuple[dict[str, Any], dict[str, Any], str | None, str]:
    contract = route_contract(route_key)
    required = dict(contract.get("required_components", {}))
    optional: dict[str, Any] = {}
    route_version = contract.get("route_version")
    evidence_role = str(contract.get("evidence_role", "undeclared_or_historical"))
    if route_key == ROUTE_ID_A and meta_feature_profile == STATIC_META_FEATURE_PROFILE_SAFE_CORE_V1:
        required = _route_a_safe_core_required_components(observed)
        optional = {
            key: {
                "status": ROUTE_A_SAFE_CORE_OPTIONAL_COMPONENTS[key],
                "observed": _get_observed_component(observed, key),
                "strict_default": ROUTE_A_REQUIRED_COMPONENTS[key],
            }
            for key in ROUTE_A_SAFE_CORE_OPTIONAL_COMPONENTS
        }
        route_version = ROUTE_A_META_FEATURE_VERSION
        evidence_role = "canonical_current_route_meta_feature_safe_core"
    elif route_key == ROUTE_ID_A and meta_feature_profile == STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1:
        required = _route_a_paper_i_production_required_components()
        optional = {
            key: {
                "status": ROUTE_A_PAPER_I_PRODUCTION_OPTIONAL_COMPONENTS[key],
                "observed": _get_observed_component(observed, key),
                "strict_default": ROUTE_A_REQUIRED_COMPONENTS[key],
            }
            for key in ROUTE_A_PAPER_I_PRODUCTION_OPTIONAL_COMPONENTS
        }
        route_version = ROUTE_A_PAPER_I_PRODUCTION_VERSION
        evidence_role = "canonical_current_route_paper_i_production"
    if route_key == ROUTE_ID_A:
        route_variant_id = _get_observed_component(observed, "route_variant_id")
        variant_overrides = ROUTE_A_VARIANT_REQUIRED_COMPONENT_OVERRIDES.get(
            str(route_variant_id or "").strip().lower(),
            {},
        )
        required.update(variant_overrides)
    return required, optional, route_version, evidence_role


def _normalized_component_token(value: Any, required: Any) -> str | None:
    if isinstance(required, bool):
        obs_bool = _normalize_bool(value)
        if obs_bool is None:
            return None
        return "true" if obs_bool else "false"
    token = str(value).strip().lower()
    required_token = str(required).strip().lower()
    if required_token == "full_meta" and token in {
        "full_meta",
        "math_md_full_meta",
        "math_md_full_meta_v1",
    }:
        return "full_meta"
    return token


def _alias_conflict_reason(observed: Mapping[str, Any], key: str, required: Any) -> str | None:
    alias_values = _observed_alias_values(observed, key)
    normalized: dict[str, list[str]] = {}
    for alias, value in alias_values:
        token = _normalized_component_token(value, required)
        if token is None:
            continue
        normalized.setdefault(token, []).append(f"{alias}={value!r}")
    if len(normalized) <= 1:
        return None
    entries = ";".join(item for values in normalized.values() for item in values)
    return f"conflict_alias:{key}:{entries}"


def validate_route_components(
    observed: Mapping[str, Any] | None,
    *,
    declared_route_id: Any = ROUTE_ID_UNSPECIFIED,
) -> RouteIdentityValidation:
    observed_map = dict(observed or {})
    route_key = normalize_static_route_id(declared_route_id)
    contract = route_contract(route_key)
    meta_feature_profile = normalize_static_meta_feature_profile(
        _get_observed_component(observed_map, "meta_feature_profile"),
        default=STATIC_META_FEATURE_PROFILE_OFF,
    )
    required, optional, route_version, evidence_role = _route_required_components(
        route_key,
        observed_map,
        meta_feature_profile=meta_feature_profile,
    )
    observed_components = {
        key: _get_observed_component(observed_map, key)
        for key in required
    }
    for key in optional:
        observed_components[key] = _get_observed_component(observed_map, key)
    for key in ROUTE_VARIANT_OBSERVED_COMPONENT_KEYS:
        value = _get_observed_component(observed_map, key)
        if value is not None:
            observed_components[key] = value
    reasons: list[str] = []
    if route_key == ROUTE_ID_UNSPECIFIED:
        reasons.append("missing_static_route_identity")
    for key, required_value in required.items():
        conflict_reason = _alias_conflict_reason(observed_map, key, required_value)
        if conflict_reason is not None:
            reasons.append(conflict_reason)
        observed_value = observed_components.get(key)
        if observed_value is None or observed_value == "":
            reasons.append(f"missing:{key}")
            continue
        if not _component_matches(observed_value, required_value):
            reasons.append(f"mismatch:{key}:{observed_value!r}!={required_value!r}")
    for key, metadata in optional.items():
        strict_default = metadata.get("strict_default") if isinstance(metadata, Mapping) else None
        if strict_default is None:
            continue
        conflict_reason = _alias_conflict_reason(observed_map, key, strict_default)
        if conflict_reason is not None:
            reasons.append(conflict_reason)
    valid = bool(route_key != ROUTE_ID_UNSPECIFIED and not reasons)
    canonical = bool(route_key == ROUTE_ID_A and valid)
    return RouteIdentityValidation(
        schema=ROUTE_IDENTITY_SCHEMA,
        route_id=route_key,
        route_version=route_version,
        canonical_snake_eligible=canonical,
        evidence_role=evidence_role,
        valid=valid,
        noncanonical_reasons=tuple(reasons),
        observed_components=observed_components,
        required_components=required,
        meta_feature_profile=meta_feature_profile,
        optional_components=optional,
    )


def route_identity_payload(
    observed: Mapping[str, Any] | None,
    *,
    declared_route_id: Any = ROUTE_ID_UNSPECIFIED,
    optimizer_lane: Any = None,
    evaluation_convention: Any = None,
) -> dict[str, Any]:
    validation = validate_route_components(observed, declared_route_id=declared_route_id).as_dict()
    validation["optimizer_lane"] = None if optimizer_lane is None or optimizer_lane == "" else str(optimizer_lane).strip().upper()
    validation["optimizer_is_route_identity"] = False
    validation["evaluation_convention"] = (
        None if evaluation_convention is None or evaluation_convention == "" else str(evaluation_convention)
    )
    validation["evaluation_convention_is_route_identity"] = False
    return validation


def resolve_route_identity_base_pool_key(
    *,
    pool_key: Any,
    pool_key_input: Any,
    hh_staged_continuation: bool,
) -> str:
    if bool(hh_staged_continuation):
        return str(pool_key_input if pool_key_input is not None else "full_meta")
    return str(pool_key)


def build_static_route_identity_observed_components(
    config: StaticRouteIdentityConfig,
) -> dict[str, Any]:
    route_base_pool_key = resolve_route_identity_base_pool_key(
        pool_key=config.pool_key,
        pool_key_input=config.pool_key_input,
        hh_staged_continuation=bool(config.hh_staged_continuation),
    )
    return {
        "base_pool_key": str(route_base_pool_key),
        "pool_type": str(config.pool_key),
        "adapt_pool_requested": (
            None if config.pool_key_input is None else str(config.pool_key_input)
        ),
        "static_meta_feature_profile": str(config.static_meta_feature_profile),
        "static_lane_route": str(config.static_lane_route),
        "static_lane_route_is_route_identity": bool(config.static_lane_route_is_route_identity),
        "route_variant_id": (
            None if config.route_variant_id in {None, ""} else str(config.route_variant_id)
        ),
        "continuation_mode": str(config.continuation_mode),
        "phase2_novelty_mode": str(config.phase2_novelty_mode),
        "phase3_selector_policy": str(config.phase3_selector_policy),
        "phase3_selector_geometry_mode": str(config.phase3_selector_geometry_mode),
        "algebraic_shortlisting_enabled": bool(config.algebraic_shortlisting_enabled),
        "hardware_resolution_schema": "gradient_resolution_v1",
        "hardware_resolution_mode": str(config.hardware_resolution_mode),
        "phase1_score_mode": str(config.phase1_score_mode),
        "phase1_score_formula": str(config.phase1_score_formula),
        "phase1_active_score_key": str(config.phase1_active_score_key),
        "phase1_rho": float(config.phase1_rho),
        "phase2_selector_gain_mode": str(config.phase2_selector_gain_mode),
        "phase2_raw_score_formula": str(config.phase2_raw_score_formula),
        "canonical_score_formula": str(config.canonical_score_formula),
        "primary_selector_score_key": str(config.primary_selector_score_key),
        "auxiliary_terms_primary_mode": str(config.auxiliary_terms_primary_mode),
        "phase3_novelty_ablation_mode": str(config.phase3_novelty_ablation_mode),
        "phase3_window_relaxation_mode": str(config.phase3_window_relaxation_mode),
        "adapt_child_pool_expansion_mode": str(config.adapt_child_pool_expansion_mode),
        "adapt_child_pool_expansion_symmetry_policy": str(
            config.adapt_child_pool_expansion_symmetry_policy
        ),
        "adapt_child_pool_expansion_source": str(config.adapt_child_pool_expansion_source),
        "shared_pauli_pool_mode": str(config.shared_pauli_pool_mode),
        "shared_pauli_pool_symmetry_policy": str(config.shared_pauli_pool_symmetry_policy),
        "shared_pauli_pool_symmetry_gate_enforced": config.shared_pauli_pool_symmetry_gate_enforced,
        "shared_pauli_pool_ordered_pool_hash": config.shared_pauli_pool_ordered_pool_hash,
        "shared_pauli_pool_expanded_pool_term_count": (
            config.shared_pauli_pool_expanded_pool_term_count
        ),
        "phase3_plateau_acquisition_mode": str(config.phase3_plateau_acquisition_mode),
        "phase3_plateau_acquisition_score": str(config.phase3_plateau_acquisition_score),
        "phase3_plateau_score_formula": str(config.phase3_plateau_score_formula),
        "phase3_plateau_duplicate_policy": str(config.phase3_plateau_duplicate_policy),
        "phase3_plateau_unlock_margin": float(config.phase3_plateau_unlock_margin),
        "phase3_plateau_lambda_vol": float(config.phase3_plateau_lambda_vol),
        "phase3_plateau_sigma_min": float(config.phase3_plateau_sigma_min),
        "phase3_plateau_nu_min": float(config.phase3_plateau_nu_min),
        "phase3_plateau_volume_min": float(config.phase3_plateau_volume_min),
        "phase3_plateau_failed_family_patience": int(
            config.phase3_plateau_failed_family_patience
        ),
        "phase3_plateau_trial_optimizer": str(config.phase3_plateau_trial_optimizer),
        "phase3_plateau_trial_qngd_maxiter": int(
            config.phase3_plateau_trial_qngd_maxiter
        ),
        "phase3_plateau_seed_probe_mode": str(config.phase3_plateau_seed_probe_mode),
        "phase3_plateau_seed_probe_count": int(config.phase3_plateau_seed_probe_count),
        "phase3_plateau_seed_probe_radius": float(config.phase3_plateau_seed_probe_radius),
        "phase3_plateau_seed_probe_enabled": bool(config.phase3_plateau_seed_probe_enabled),
        "phase3_plateau_seed_probe_seed": config.phase3_plateau_seed_probe_seed,
        "phase2_enable_batching": bool(config.phase3_enable_batching),
        "phase3_enable_batching": bool(config.phase3_enable_batching),
        "phase3_batch_selection_mode": str(config.phase3_batch_selection_mode),
        "phase3_batch_prefilter_mode": str(config.phase3_batch_prefilter_mode),
        "phase3_batch_order_selection_mode": str(config.phase3_batch_order_selection_mode),
        "phase3_nested_window_application": str(config.phase3_nested_window_application),
        "phase1_prune_enabled": bool(config.phase1_prune_enabled),
        "phase1_prune_policy": str(config.phase1_prune_policy),
        "phase1_prune_mode": str(config.phase1_prune_mode),
        "phase1_prune_amplitude_witness_required": bool(
            config.phase1_prune_amplitude_witness_required
        ),
    }


def validate_declared_static_route_identity(
    payload: Mapping[str, Any],
    *,
    declared_route_id: Any,
) -> None:
    route_key = normalize_static_route_id(declared_route_id)
    if route_key in {ROUTE_ID_A, ROUTE_ID_B_LEGACY_PAIRWISE, ROUTE_ID_C} and not bool(
        payload.get("valid", False)
    ):
        raise ValueError(
            f"Declared static_route_id={route_key} does not match the active ADAPT route: "
            + "; ".join(str(x) for x in payload.get("noncanonical_reasons", ()))
        )


def build_static_route_identity_payload(
    config: StaticRouteIdentityConfig,
    *,
    validate_declared: bool = False,
) -> dict[str, Any]:
    declared_route_id = normalize_static_route_id(config.declared_route_id)
    payload = route_identity_payload(
        build_static_route_identity_observed_components(config),
        declared_route_id=declared_route_id,
        optimizer_lane=str(config.optimizer_lane),
    )
    if bool(validate_declared):
        validate_declared_static_route_identity(payload, declared_route_id=declared_route_id)
    return payload


def static_route_id_from_record(
    row: Mapping[str, Any],
    *,
    record_id: str = "",
    fail_on_route_named_missing: bool = False,
) -> str:
    explicit = row.get("static_route_id") or row.get("route_id") or row.get("static_route")
    if explicit is not None and explicit != "":
        return normalize_static_route_id(explicit)
    record_lower = str(record_id or row.get("record_id") or "").lower()
    variant_lower = str(row.get("algorithm_variant", "") or "").strip().lower()
    route_a_named = "routea" in record_lower or "route_a" in record_lower or variant_lower.startswith("a_current")
    route_c_named = "routec" in record_lower or "route_c" in record_lower or variant_lower.startswith("c_plateau")
    if route_a_named and fail_on_route_named_missing:
        raise ValueError(
            f"record {record_id!r} is Route-A-named but does not declare static_route_id=route_a"
        )
    if route_c_named and fail_on_route_named_missing:
        raise ValueError(
            f"record {record_id!r} is Route-C-named but does not declare static_route_id=route_c"
        )
    novelty = str(row.get("phase2_novelty_mode", "") or "").strip().lower()
    plateau_mode = str(row.get("phase3_plateau_acquisition_mode", "") or "").strip().lower()
    if novelty == "legacy_pairwise_v1" and plateau_mode == "novelty_cost_v1":
        if fail_on_route_named_missing:
            raise ValueError(
                f"record {record_id!r} mixes legacy Route-B pairwise novelty with Route-C plateau mode"
            )
        return ROUTE_ID_UNSPECIFIED
    if novelty == "legacy_pairwise_v1":
        return ROUTE_ID_B_LEGACY_PAIRWISE
    if plateau_mode == "novelty_cost_v1":
        return ROUTE_ID_C
    return ROUTE_ID_UNSPECIFIED


__all__ = [
    "ROUTE_A_REQUIRED_COMPONENTS",
    "ROUTE_A_META_FEATURE_VERSION",
    "ROUTE_A_PAPER_I_PRODUCTION_OPTIONAL_COMPONENTS",
    "ROUTE_A_PAPER_I_PRODUCTION_VERSION",
    "ROUTE_A_SAFE_CORE_OPTIONAL_COMPONENTS",
    "ROUTE_A_VERSION",
    "ROUTE_B_REQUIRED_COMPONENTS",
    "ROUTE_B_VERSION",
    "ROUTE_C_REQUIRED_COMPONENTS",
    "ROUTE_C_VERSION",
    "FIRST_CLASS_ROUTE_ID_CHOICES",
    "LEGACY_ROUTE_ID_CHOICES",
    "ROUTE_ID_A",
    "ROUTE_ID_B_LEGACY_PAIRWISE",
    "ROUTE_ID_C",
    "ROUTE_ID_CHOICES",
    "ROUTE_ID_UNSPECIFIED",
    "ROUTE_IDENTITY_SCHEMA",
    "RouteIdentityValidation",
    "StaticRouteIdentityConfig",
    "STATIC_META_FEATURE_PROFILE_CHOICES",
    "STATIC_META_FEATURE_PROFILE_OFF",
    "STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1",
    "STATIC_META_FEATURE_PROFILE_SAFE_CORE_V1",
    "build_static_route_identity_observed_components",
    "build_static_route_identity_payload",
    "normalize_static_meta_feature_profile",
    "normalize_static_route_id",
    "resolve_route_identity_base_pool_key",
    "route_contract",
    "route_identity_payload",
    "static_route_id_from_record",
    "validate_declared_static_route_identity",
    "validate_route_components",
]
