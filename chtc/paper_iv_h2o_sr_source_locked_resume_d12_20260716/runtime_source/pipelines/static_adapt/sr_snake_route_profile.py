"""Executable route contract for the historical high-accuracy SR-SNAKE v1.

The older SR identity resolver intentionally classified only the local
coordinate/trust overlay.  That was not enough to reproduce the Paper-I
Hubbard--Holstein route: a partially configured invocation could acquire the
canonical profile name while using different optimizer, child, shortlist,
beam, prune, or fallback policies.

This module owns the fail-closed CLI normalization contract.  It contains no
scientific implementation; it only materializes and hashes the already
implemented historical settings before the runtime begins.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SR_ROUTE_PROFILE_REQUEST_OFF = "off"
SR_ROUTE_PROFILE_CANONICAL_V1 = "supported_whitened_adaptive_trust_v1"
SR_ROUTE_PROFILE_CANONICAL_ALIAS = "sr_snake_v1"
SR_ROUTE_PROFILE_REQUEST_CHOICES = (
    SR_ROUTE_PROFILE_REQUEST_OFF,
    SR_ROUTE_PROFILE_CANONICAL_ALIAS,
    SR_ROUTE_PROFILE_CANONICAL_V1,
)

SR_ROUTE_PROFILE_CONTRACT_SCHEMA = "sr_snake_route_profile_contract_v1"
SR_ROUTE_PROFILE_CONTRACT_DIGEST_SCHEMA = (
    "sr_snake_route_profile_contract_sha256_v1"
)

_HISTORICAL_COMMAND_SHA256 = (
    "37751de2805875337cb8a0034a7394b02344c893e1b0a583439b1954c7c8061e"
)
_HISTORICAL_RESULT_SHA256 = (
    "f8d2bb9756d395d7806bb2f365d95a5fcb4c5aa6de55e96f89ecfc35295b10da"
)
_SELF_CONTAINED_ARCHIVE_SHA256 = (
    "c290d9ee1b31cd211e41faad174cd2e311ca65cf351c46bbb84fbaaea9504c6c"
)


# Namespace destinations are used deliberately: this is the one executable
# source for expanding ``--sr-route-profile sr_snake_v1``.  Regime physics
# (U, g, n_ph_max, and the exact same-cutoff reference) remains outside the
# method profile and must be supplied by the regime's source lock.
CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS: dict[str, Any] = {
    "problem": "hh",
    "adapt_pool": "full_meta",
    "adapt_pool_class_filter_json": None,
    "adapt_pool_label_filter_json": None,
    "adapt_selected_logical_source_json": None,
    "adapt_selected_logical_mode": "off",
    "adapt_continuation_mode": "phase3_v1",
    "static_route_id": "route_a",
    "static_meta_feature_profile": "paper_i_production_v1",
    "static_lane_route": "physical_operator_type",
    "physical_lane_shortlist_aggressiveness": 3,
    "adapt_reoptimization_route": "off",
    "adapt_formal_manifold_route_profile": "off",
    "adapt_formal_manifold_config_json": None,
    "historical_singleton_coordinate_solve_policy": (
        "supported_metric_whitened_eigh_v1"
    ),
    "historical_singleton_coordinate_solve_scope": "phase3_only_v1",
    "historical_singleton_trust_region_update_policy": (
        "displacement_calibrated_unbounded_v2"
    ),
    "sr_powell_coordinate_chart_policy": (
        "expanded_runtime_projected_logical_v1"
    ),
    "sr_escape_mode": "disabled",
    "sr_controller_ablation_contract": "off",
    "phase2_gram_novelty_policy": "ordinary_multiplier_v1",
    "phase3_gram_novelty_policy": "ordinary_multiplier_v1",
    "phase3_novelty_ablation_mode": "off",
    "phase2_novelty_mode": "collective_span_v1",
    "phase2_selector_gain_mode": "trust_region_v1",
    "phase2_rho": 0.25,
    "adapt_inner_optimizer": "POWELL",
    "adapt_maxiter": 200,
    "adapt_scipy_maxfev": 0,
    "adapt_state_backend": "compiled",
    "adapt_seed": 7,
    "adapt_reopt_policy": "windowed",
    "adapt_window_size": 3,
    "adapt_window_topk": 0,
    "adapt_full_refit_every": 8,
    "adapt_final_full_refit": "true",
    "adapt_final_refit_maxiter": 200,
    "adapt_insertion_mode": "append_only",
    "adapt_max_depth": 30,
    "adapt_allow_repeats": True,
    "adapt_finite_angle_fallback": True,
    "adapt_finite_angle": 0.1,
    "adapt_finite_angle_min_improvement": 1.0e-12,
    "adapt_disable_hh_seed": False,
    "phase0_pilot_enabled": False,
    "phase1_shortlist_size": 24,
    "phase2_shortlist_size": 12,
    "phase2_shortlist_fraction": 0.25,
    "phase2_enable_batching": False,
    "phase3_enable_batching": False,
    "phase3_runtime_split_mode": "shortlist_pauli_children_v1",
    "allow_archival_phase3_runtime_split": True,
    "phase3_runtime_split_selection_mode": "archival_child_set_forward_v1",
    "phase3_runtime_split_max_subset_size": 1,
    "phase3_runtime_split_subset_sizes": "1",
    "phase3_runtime_split_child_set_symmetry_policy": "hard_guard",
    "phase3_runtime_split_child_padding_policy": "exact_projected_grouped_v1",
    "adapt_child_pool_expansion_mode": "off",
    "adapt_child_pool_expansion_symmetry_policy": "off",
    "shared_pauli_pool_mode": "off",
    "shared_pauli_pool_symmetry_policy": "off",
    "phase1_prune_enabled": True,
    "phase1_prune_policy": "recoverability_ladder_v1",
    "phase1_prune_mode": "both",
    "phase1_prune_fraction": 0.25,
    "phase1_prune_min_candidates": 1,
    "phase1_prune_max_candidates": 6,
    "phase1_prune_max_regression": 1.0e-8,
    "phase1_prune_tolerance_mode": "auto",
    "phase1_prune_tolerance_shot_coeff": 0.0,
    "phase1_prune_tolerance_screen_coeff": 0.01,
    "phase1_prune_tolerance_chem": 0.0,
    "phase1_prune_tolerance_rel_coeff": 0.0,
    "phase1_prune_tolerance_target_energy": None,
    "phase1_prune_retained_gain_ratio": 0.5,
    "phase1_prune_protect_steps": 2,
    "phase1_prune_stale_age": 2,
    "phase1_prune_stagnation_threshold": 0.0,
    "phase1_prune_small_theta_abs": 1.0e-3,
    "phase1_prune_small_theta_relative": 0.5,
    "phase1_prune_cooldown_steps": 2,
    "phase1_prune_local_window_size": 4,
    "phase1_prune_old_fraction": 0.25,
    "phase1_prune_checkpoint_period": 3,
    "phase1_prune_maturity_threshold": 0.5,
    "phase1_prune_snr_threshold": 1.0,
    "phase1_prune_prefilter_policy": "off",
    "phase1_prune_prefilter_json": None,
    "phase1_prune_risk_threshold": 0.0,
    "phase1_prune_prefilter_max_candidates": 1,
    "adapt_beam_live_branches": 3,
    "adapt_beam_children_per_parent": 2,
    "adapt_beam_terminated_keep": None,
    "adapt_beam_terminal_archive_mode": "disabled",
    "adapt_beam_lambda": 0.005,
    "adapt_beam_parent_workers": 1,
    "phase3_selector_policy": "algebraic_nested_v1",
    "phase3_selector_geometry_mode": "reduced",
    "phase3_geometry_window_size": 0,
    "phase3_window_relaxation_mode": "reduced",
    "phase3_backend_cost_mode": "marrakesh_graph_span_v1",
    "phase3_backend_name": "FakeMarrakesh",
    "phase3_backend_transpile_seed": 7,
    "phase3_backend_optimization_level": 1,
    "phase3_hardware_cost_normalization_mode": "family_robust_v1",
    "phase3_lifetime_cost_mode": "phase3_v1",
    "phase3_enable_rescue": False,
    "phase3_symmetry_mitigation_mode": "off",
    "phase3_plateau_acquisition_mode": "off",
    "phase3_plateau_seed_probe_mode": "off",
    "phase3_shadow_legacy_geometry_mode": "off",
    "phase3_shadow_legacy_max_depth": 0,
    "phase3_parent_collapse_debug_max_depth": 0,
    "hardware_resolution_mode": "ideal",
    "gradient_hw_floor": 0.0,
    "gradient_drift_floor": 0.0,
}


_DEST_OPTION_STRINGS: dict[str, tuple[str, ...]] = {
    "problem": ("--problem",),
    "adapt_pool": ("--adapt-pool",),
    "adapt_pool_class_filter_json": ("--adapt-pool-class-filter-json",),
    "adapt_pool_label_filter_json": ("--adapt-pool-label-filter-json",),
    "adapt_selected_logical_source_json": ("--adapt-selected-logical-source-json",),
    "adapt_selected_logical_mode": ("--adapt-selected-logical-mode",),
    "adapt_continuation_mode": ("--adapt-continuation-mode",),
    "static_route_id": ("--static-route-id",),
    "static_meta_feature_profile": ("--static-meta-feature-profile",),
    "static_lane_route": ("--static-lane-route",),
    "physical_lane_shortlist_aggressiveness": (
        "--physical-lane-shortlist-aggressiveness",
    ),
    "adapt_reoptimization_route": ("--adapt-reoptimization-route",),
    "adapt_formal_manifold_route_profile": (
        "--adapt-formal-manifold-route-profile",
    ),
    "adapt_formal_manifold_config_json": (
        "--adapt-formal-manifold-config-json",
    ),
    "historical_singleton_coordinate_solve_policy": (
        "--historical-singleton-coordinate-solve-policy",
    ),
    "historical_singleton_coordinate_solve_scope": (
        "--historical-singleton-coordinate-solve-scope",
    ),
    "historical_singleton_trust_region_update_policy": (
        "--historical-singleton-trust-region-update-policy",
    ),
    "sr_powell_coordinate_chart_policy": (
        "--sr-powell-coordinate-chart-policy",
    ),
    "sr_escape_mode": ("--sr-escape-mode",),
    "sr_controller_ablation_contract": (
        "--sr-controller-ablation-contract",
    ),
    "phase2_gram_novelty_policy": ("--phase2-gram-novelty-policy",),
    "phase3_gram_novelty_policy": ("--phase3-gram-novelty-policy",),
    "phase3_novelty_ablation_mode": ("--phase3-novelty-ablation-mode",),
    "phase2_novelty_mode": ("--phase2-novelty-mode",),
    "phase2_selector_gain_mode": ("--phase2-selector-gain-mode",),
    "phase2_rho": ("--phase2-rho",),
    "adapt_inner_optimizer": ("--adapt-inner-optimizer",),
    "adapt_maxiter": ("--adapt-maxiter",),
    "adapt_scipy_maxfev": ("--adapt-scipy-maxfev",),
    "adapt_state_backend": ("--adapt-state-backend",),
    "adapt_seed": ("--adapt-seed",),
    "adapt_reopt_policy": ("--adapt-reopt-policy",),
    "adapt_window_size": ("--adapt-window-size",),
    "adapt_window_topk": ("--adapt-window-topk",),
    "adapt_full_refit_every": ("--adapt-full-refit-every",),
    "adapt_final_full_refit": ("--adapt-final-full-refit",),
    "adapt_final_refit_maxiter": ("--adapt-final-refit-maxiter",),
    "adapt_insertion_mode": ("--adapt-insertion-mode",),
    "adapt_max_depth": ("--adapt-max-depth",),
    "adapt_allow_repeats": ("--adapt-allow-repeats", "--adapt-no-repeats"),
    "adapt_finite_angle_fallback": (
        "--adapt-finite-angle-fallback",
        "--adapt-no-finite-angle-fallback",
    ),
    "adapt_finite_angle": ("--adapt-finite-angle",),
    "adapt_finite_angle_min_improvement": (
        "--adapt-finite-angle-min-improvement",
    ),
    "adapt_disable_hh_seed": ("--adapt-disable-hh-seed",),
    "phase0_pilot_enabled": ("--phase0-pilot-enabled", "--phase0-no-pilot"),
    "phase1_shortlist_size": ("--phase1-shortlist-size",),
    "phase2_shortlist_size": ("--phase2-shortlist-size",),
    "phase2_shortlist_fraction": ("--phase2-shortlist-fraction",),
    "phase2_enable_batching": (
        "--phase2-enable-batching",
        "--phase2-no-batching",
    ),
    "phase3_enable_batching": (
        "--phase3-enable-batching",
        "--phase3-no-batching",
    ),
    "phase3_runtime_split_mode": ("--phase3-runtime-split-mode",),
    "allow_archival_phase3_runtime_split": (
        "--allow-archival-phase3-runtime-split",
    ),
    "phase3_runtime_split_selection_mode": (
        "--phase3-runtime-split-selection-mode",
    ),
    "phase3_runtime_split_max_subset_size": (
        "--phase3-runtime-split-max-subset-size",
    ),
    "phase3_runtime_split_subset_sizes": (
        "--phase3-runtime-split-subset-sizes",
    ),
    "phase3_runtime_split_child_set_symmetry_policy": (
        "--phase3-runtime-split-child-set-symmetry-policy",
    ),
    "phase3_runtime_split_child_padding_policy": (
        "--phase3-runtime-split-child-padding-policy",
    ),
    "adapt_child_pool_expansion_mode": ("--adapt-child-pool-expansion-mode",),
    "adapt_child_pool_expansion_symmetry_policy": (
        "--adapt-child-pool-expansion-symmetry-policy",
    ),
    "shared_pauli_pool_mode": ("--shared-pauli-pool-mode",),
    "shared_pauli_pool_symmetry_policy": (
        "--shared-pauli-pool-symmetry-policy",
    ),
    "phase1_prune_enabled": ("--phase1-prune-enabled", "--phase1-no-prune"),
    "phase1_prune_policy": ("--phase1-prune-policy",),
    "phase1_prune_mode": ("--phase1-prune-mode",),
    "phase1_prune_fraction": ("--phase1-prune-fraction",),
    "phase1_prune_min_candidates": ("--phase1-prune-min-candidates",),
    "phase1_prune_max_candidates": ("--phase1-prune-max-candidates",),
    "phase1_prune_max_regression": ("--phase1-prune-max-regression",),
    "phase1_prune_tolerance_mode": ("--phase1-prune-tolerance-mode",),
    "phase1_prune_tolerance_shot_coeff": (
        "--phase1-prune-tolerance-shot-coeff",
    ),
    "phase1_prune_tolerance_screen_coeff": (
        "--phase1-prune-tolerance-screen-coeff",
    ),
    "phase1_prune_tolerance_chem": ("--phase1-prune-tolerance-chem",),
    "phase1_prune_tolerance_rel_coeff": (
        "--phase1-prune-tolerance-rel-coeff",
    ),
    "phase1_prune_tolerance_target_energy": (
        "--phase1-prune-tolerance-target-energy",
    ),
    "phase1_prune_retained_gain_ratio": (
        "--phase1-prune-retained-gain-ratio",
    ),
    "phase1_prune_protect_steps": ("--phase1-prune-protect-steps",),
    "phase1_prune_stale_age": ("--phase1-prune-stale-age",),
    "phase1_prune_stagnation_threshold": (
        "--phase1-prune-stagnation-threshold",
    ),
    "phase1_prune_small_theta_abs": ("--phase1-prune-small-theta-abs",),
    "phase1_prune_small_theta_relative": (
        "--phase1-prune-small-theta-relative",
    ),
    "phase1_prune_cooldown_steps": ("--phase1-prune-cooldown-steps",),
    "phase1_prune_local_window_size": (
        "--phase1-prune-local-window-size",
    ),
    "phase1_prune_old_fraction": ("--phase1-prune-old-fraction",),
    "phase1_prune_checkpoint_period": (
        "--phase1-prune-checkpoint-period",
    ),
    "phase1_prune_maturity_threshold": (
        "--phase1-prune-maturity-threshold",
    ),
    "phase1_prune_snr_threshold": ("--phase1-prune-snr-threshold",),
    "phase1_prune_prefilter_policy": ("--phase1-prune-prefilter-policy",),
    "phase1_prune_prefilter_json": ("--phase1-prune-prefilter-json",),
    "phase1_prune_risk_threshold": ("--phase1-prune-risk-threshold",),
    "phase1_prune_prefilter_max_candidates": (
        "--phase1-prune-prefilter-max-candidates",
    ),
    "adapt_beam_live_branches": ("--adapt-beam-live-branches",),
    "adapt_beam_children_per_parent": ("--adapt-beam-children-per-parent",),
    "adapt_beam_terminated_keep": ("--adapt-beam-terminated-keep",),
    "adapt_beam_terminal_archive_mode": (
        "--adapt-beam-terminal-archive-mode",
    ),
    "adapt_beam_lambda": ("--adapt-beam-lambda",),
    "adapt_beam_parent_workers": ("--adapt-beam-parent-workers",),
    "phase3_selector_policy": ("--phase3-selector-policy",),
    "phase3_selector_geometry_mode": ("--phase3-selector-geometry-mode",),
    "phase3_geometry_window_size": ("--phase3-geometry-window-size",),
    "phase3_window_relaxation_mode": ("--phase3-window-relaxation-mode",),
    "phase3_backend_cost_mode": ("--phase3-backend-cost-mode",),
    "phase3_backend_name": ("--phase3-backend-name",),
    "phase3_backend_transpile_seed": ("--phase3-backend-transpile-seed",),
    "phase3_backend_optimization_level": (
        "--phase3-backend-optimization-level",
    ),
    "phase3_hardware_cost_normalization_mode": (
        "--phase3-hardware-cost-normalization-mode",
    ),
    "phase3_lifetime_cost_mode": ("--phase3-lifetime-cost-mode",),
    "phase3_enable_rescue": ("--phase3-enable-rescue", "--phase3-no-rescue"),
    "phase3_symmetry_mitigation_mode": (
        "--phase3-symmetry-mitigation-mode",
    ),
    "phase3_plateau_acquisition_mode": (
        "--phase3-plateau-acquisition-mode",
    ),
    "phase3_plateau_seed_probe_mode": ("--phase3-plateau-seed-probe-mode",),
    "phase3_shadow_legacy_geometry_mode": (
        "--phase3-shadow-legacy-geometry-mode",
    ),
    "phase3_shadow_legacy_max_depth": ("--phase3-shadow-legacy-max-depth",),
    "phase3_parent_collapse_debug_max_depth": (
        "--phase3-parent-collapse-debug-max-depth",
    ),
    "hardware_resolution_mode": ("--hardware-resolution-mode",),
    "gradient_hw_floor": ("--gradient-hw-floor",),
    "gradient_drift_floor": ("--gradient-drift-floor",),
}

_DISALLOWED_BOOLEAN_OPTIONS = frozenset(
    {
        "--adapt-no-repeats",
        "--adapt-no-finite-angle-fallback",
        "--adapt-disable-hh-seed",
        "--phase0-pilot-enabled",
        "--phase1-no-prune",
        "--phase2-enable-batching",
        "--phase3-enable-batching",
        "--phase3-enable-rescue",
    }
)


def _json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def canonical_sr_snake_v1_contract() -> dict[str, Any]:
    """Return a fresh serialization-safe copy of the canonical contract."""

    payload: dict[str, Any] = {
        "schema": SR_ROUTE_PROFILE_CONTRACT_SCHEMA,
        "route_family": "singleton_response_snake",
        "route_profile": SR_ROUTE_PROFILE_CANONICAL_V1,
        "execution_settings": dict(CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS),
        "semantic_invariants": {
            "regime_physics_source": "per_regime_source_lock",
            "same_cutoff_reference_required": True,
            "full_meta_hva_policy": "included_no_filters_v1",
            "phase2_score_policy": "collective_span_novelty_multiplier_v1",
            "phase3_ordinary_novelty_multiplier": "unit_n3_v1",
            "all_energy_models_infeasible_policy": (
                "collective_span_novelty_over_cost_v1"
            ),
            "geometry_expansion_refit_policy": "full_coordinate_refit_v1",
            "geometry_expansion_radius_policy": (
                "realized_fs_displacement_on_descent_hold_on_no_descent_v1"
            ),
            "scalar_unwhitened_fallback_allowed": False,
            "admission_cardinality": 1,
            "admission_rollback_supported": False,
            "repeated_generator_identity_allowed": True,
            "route_a_funnel_active": False,
        },
        "historical_authority": {
            "historical_command_sha256": _HISTORICAL_COMMAND_SHA256,
            "historical_result_sha256": _HISTORICAL_RESULT_SHA256,
            "self_contained_replay_archive_sha256": (
                _SELF_CONTAINED_ARCHIVE_SHA256
            ),
            "weak_weak_absolute_error": 4.472864776339236e-7,
        },
    }
    return json.loads(json.dumps(payload, sort_keys=True))


def canonical_sr_snake_v1_contract_sha256() -> str:
    return _json_sha256(canonical_sr_snake_v1_contract())


def normalize_sr_route_profile_request(raw: Any) -> str:
    key = str(SR_ROUTE_PROFILE_REQUEST_OFF if raw in {None, ""} else raw)
    key = key.strip().lower().replace("-", "_")
    aliases = {
        "none": SR_ROUTE_PROFILE_REQUEST_OFF,
        "disabled": SR_ROUTE_PROFILE_REQUEST_OFF,
        "sr_snake_v1": SR_ROUTE_PROFILE_CANONICAL_V1,
        "canonical_v1": SR_ROUTE_PROFILE_CANONICAL_V1,
        SR_ROUTE_PROFILE_CANONICAL_V1: SR_ROUTE_PROFILE_CANONICAL_V1,
    }
    key = aliases.get(key, key)
    if key not in {
        SR_ROUTE_PROFILE_REQUEST_OFF,
        SR_ROUTE_PROFILE_CANONICAL_V1,
    }:
        raise ValueError(
            "sr_route_profile must be one of "
            f"{list(SR_ROUTE_PROFILE_REQUEST_CHOICES)}; got {raw!r}."
        )
    return key


def _comparable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    return value


def _equivalent(field: str, actual: Any, expected: Any) -> bool:
    actual_value = _comparable(actual)
    expected_value = _comparable(expected)
    if field == "sr_powell_coordinate_chart_policy" and actual_value == "auto":
        return True
    if isinstance(expected_value, float):
        try:
            return float(actual_value) == float(expected_value)
        except (TypeError, ValueError):
            return False
    return actual_value == expected_value


def normalize_sr_route_profile_namespace(namespace: Any) -> Any:
    """Materialize the canonical profile and reject explicit setting drift.

    The parser records option strings that were explicitly present.  Implicit
    generic defaults are replaced by the profile.  Explicit matching values
    are accepted; explicit conflicting values fail closed.
    """

    requested = normalize_sr_route_profile_request(
        getattr(namespace, "sr_route_profile_request", None)
    )
    setattr(namespace, "sr_route_profile_request", requested)
    if requested == SR_ROUTE_PROFILE_REQUEST_OFF:
        setattr(namespace, "sr_route_profile_resolved", None)
        setattr(namespace, "sr_route_profile_contract", None)
        setattr(namespace, "sr_route_profile_contract_sha256", None)
        return namespace

    explicit_raw = getattr(namespace, "_explicit_cli_options", None)
    if explicit_raw is None:
        # A hand-built Namespace has no way to distinguish an implicit parser
        # default from an explicit scientific override.  It must already carry
        # the complete contract instead of being silently rewritten.
        explicit_options: frozenset[str] | None = None
    else:
        explicit_options = frozenset(str(value) for value in explicit_raw)

    conflicts: list[dict[str, Any]] = []
    for field, expected in CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS.items():
        current = getattr(namespace, field, None)
        option_strings = _DEST_OPTION_STRINGS.get(field, ())
        field_explicit = bool(
            explicit_options is not None
            and explicit_options.intersection(option_strings)
        )
        disallowed = sorted(
            set(option_strings).intersection(_DISALLOWED_BOOLEAN_OPTIONS).intersection(
                explicit_options or ()
            )
        )
        if disallowed or (
            (field_explicit or explicit_options is None)
            and not _equivalent(field, current, expected)
        ):
            conflicts.append(
                {
                    "field": field,
                    "explicit_options": sorted(
                        set(option_strings).intersection(explicit_options or ())
                    ),
                    "current": _comparable(current),
                    "required": expected,
                }
            )
            continue
        setattr(namespace, field, expected)

    if conflicts:
        raise ValueError(
            "SR-SNAKE v1 profile conflicts with explicit or untracked "
            "scientific settings: "
            + json.dumps(conflicts, sort_keys=True, default=str)
        )

    contract = canonical_sr_snake_v1_contract()
    digest = canonical_sr_snake_v1_contract_sha256()
    setattr(namespace, "sr_route_profile_resolved", SR_ROUTE_PROFILE_CANONICAL_V1)
    setattr(namespace, "sr_route_profile_contract", contract)
    setattr(namespace, "sr_route_profile_contract_sha256", digest)
    return namespace


def validate_sr_route_profile_contract(
    *,
    profile_request: Any,
    contract: Mapping[str, Any] | None,
    contract_sha256: str | None,
) -> dict[str, Any] | None:
    """Validate an already-normalized runtime/checkpoint contract."""

    requested = normalize_sr_route_profile_request(profile_request)
    if requested == SR_ROUTE_PROFILE_REQUEST_OFF:
        if (
            (contract is not None and bool(dict(contract)))
            or contract_sha256 not in {None, ""}
        ):
            raise ValueError(
                "An SR route contract was supplied while sr_route_profile is off."
            )
        return None
    if not isinstance(contract, Mapping):
        raise ValueError(
            "SR-SNAKE v1 requires its complete route-profile contract."
        )
    payload = dict(contract)
    expected = canonical_sr_snake_v1_contract()
    if payload != expected:
        raise ValueError("SR-SNAKE v1 route-profile contract drifted.")
    actual_digest = _json_sha256(payload)
    expected_digest = canonical_sr_snake_v1_contract_sha256()
    if actual_digest != expected_digest or str(contract_sha256 or "") != expected_digest:
        raise ValueError(
            "SR-SNAKE v1 route-profile contract SHA-256 is missing or drifted."
        )
    return payload


def validate_sr_route_profile_runtime_settings(
    *,
    profile_request: Any,
    contract: Mapping[str, Any] | None,
    contract_sha256: str | None,
    runtime_settings: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Require supplied effective runtime settings to match the profile."""

    payload = validate_sr_route_profile_contract(
        profile_request=profile_request,
        contract=contract,
        contract_sha256=contract_sha256,
    )
    if payload is None:
        return None
    expected = dict(payload["execution_settings"])
    missing = sorted(set(runtime_settings).difference(expected))
    if missing:
        raise ValueError(
            "SR-SNAKE v1 runtime validator received unknown contract fields: "
            + ",".join(missing)
        )
    mismatches = [
        {
            "field": field,
            "runtime": _comparable(value),
            "required": expected[field],
        }
        for field, value in runtime_settings.items()
        if not _equivalent(field, value, expected[field])
    ]
    if mismatches:
        raise ValueError(
            "SR-SNAKE v1 effective runtime settings drifted: "
            + json.dumps(mismatches, sort_keys=True, default=str)
        )
    return payload


__all__ = [
    "CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS",
    "SR_ROUTE_PROFILE_CANONICAL_ALIAS",
    "SR_ROUTE_PROFILE_CANONICAL_V1",
    "SR_ROUTE_PROFILE_CONTRACT_DIGEST_SCHEMA",
    "SR_ROUTE_PROFILE_CONTRACT_SCHEMA",
    "SR_ROUTE_PROFILE_REQUEST_CHOICES",
    "SR_ROUTE_PROFILE_REQUEST_OFF",
    "canonical_sr_snake_v1_contract",
    "canonical_sr_snake_v1_contract_sha256",
    "normalize_sr_route_profile_namespace",
    "normalize_sr_route_profile_request",
    "validate_sr_route_profile_contract",
    "validate_sr_route_profile_runtime_settings",
]
