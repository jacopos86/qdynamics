from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.static_adapt.extensions import PRUNING_RUNTIME_KEYS
from pipelines.static_adapt.resume_scaffold import (
    validate_resume_sr_route_profile_contract,
)
from pipelines.static_adapt.sr_snake.contracts import SRStopPolicy
from pipelines.static_adapt.sr_snake_route_profile import (
    CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS,
    CANONICAL_SR_SNAKE_V2_EXECUTION_SETTINGS,
    CANONICAL_SR_SNAKE_V3_EXECUTION_SETTINGS,
    CANONICAL_SR_SNAKE_V4_EXECUTION_SETTINGS,
    CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS,
    CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1_EXECUTION_SETTINGS,
    CANONICAL_SR_SNAKE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1_EXECUTION_SETTINGS,
    CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_V2_EXECUTION_SETTINGS,
    CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3_EXECUTION_SETTINGS,
    HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS,
    HISTORICAL_SR_SNAKE_V1_ACCEPTED_REFIT_SETTINGS,
    HISTORICAL_SR_SNAKE_RESPONSE_SCOPE_SETTINGS,
    PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1,
    PHASE3_RESPONSE_COORDINATE_SCOPE_LEGACY_REOPT_COUPLED_V1,
    SR_ROUTE_PROFILE_CANONICAL_V1,
    SR_ROUTE_PROFILE_CANDIDATE_V4,
    SR_ROUTE_PROFILE_CONVENTIONAL_V2,
    SR_ROUTE_PROFILE_CONVENTIONAL_V3,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1,
    SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1,
    SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2,
    SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3,
    SR_ROUTE_PROFILE_REQUEST_OFF,
    canonical_sr_snake_v1_contract,
    canonical_sr_snake_v1_contract_sha256,
    canonical_sr_snake_v2_contract,
    canonical_sr_snake_v2_contract_sha256,
    canonical_sr_snake_v3_contract,
    canonical_sr_snake_v3_contract_sha256,
    canonical_sr_snake_v4_contract,
    canonical_sr_snake_v4_contract_sha256,
    canonical_sr_snake_no_prune_symmetric_cost_v1_contract,
    canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256,
    canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract_sha256,
    canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract,
    canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract_sha256,
    canonical_sr_snake_h2o_derivative_resolved_v2_contract,
    canonical_sr_snake_h2o_derivative_resolved_v2_contract_sha256,
    canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract,
    canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract_sha256,
    normalize_sr_route_profile_namespace,
    validate_sr_route_profile_contract,
    validate_sr_route_profile_runtime_settings,
)
from pipelines.static_adapt.sr_snake_phase12_policy import (
    PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
    PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF,
    PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1,
    PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1,
)
from test_support.route_contract_kwargs import (
    assert_route_binding_rejected,
    expected_flat_settings,
    hh_problem_context,
    route_identity,
    route_pruning,
    route_runtime_kwargs,
)

# Structural note: the retired CLI parser carried a large family of
# fail-closed tests (explicit flags that disagreed with a selected route
# profile exited with SystemExit 2).  On the live route there is no
# per-setting override channel — the runtime kwargs are projected directly
# from the authenticated contract — so "an explicit flag cannot drift a
# profile" holds by construction and those tests were removed.  The live
# fail-closed gate is the controller runtime factory's digest set, covered
# by test_candidate_v4_identity_is_readable_but_not_execution_authority.
#
# Structural note: the historical v1/v2/v3 contracts predate the typed
# pruning-extension interview, so they cannot pass through the live kwargs
# builder.  Their tests below assert contract-registry identity (profile ->
# contract -> digest) rather than runtime-kwargs materialization; runtime
# round-trip coverage lives on the v4 and no-prune profiles, which the
# builder accepts.


def _contract_payload(*, location: str = "settings") -> dict[str, object]:
    return {
        location: {
            "sr_route_profile_request": SR_ROUTE_PROFILE_CANONICAL_V1,
            "sr_route_profile_contract": canonical_sr_snake_v1_contract(),
            "sr_route_profile_contract_sha256": (
                canonical_sr_snake_v1_contract_sha256()
            ),
        }
    }


def _v2_contract_payload(*, location: str = "settings") -> dict[str, object]:
    return {
        location: {
            "sr_route_profile_request": SR_ROUTE_PROFILE_CONVENTIONAL_V2,
            "sr_route_profile_contract": canonical_sr_snake_v2_contract(),
            "sr_route_profile_contract_sha256": (
                canonical_sr_snake_v2_contract_sha256()
            ),
        }
    }


def _v3_contract_payload(*, location: str = "settings") -> dict[str, object]:
    return {
        location: {
            "sr_route_profile_request": SR_ROUTE_PROFILE_CONVENTIONAL_V3,
            "sr_route_profile_contract": canonical_sr_snake_v3_contract(),
            "sr_route_profile_contract_sha256": (
                canonical_sr_snake_v3_contract_sha256()
            ),
            "phase3_response_coordinate_scope": (
                PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
            ),
        }
    }


def _route_profile_namespace(
    profile: str,
    *,
    explicit_options: tuple[str, ...] = (),
    overrides: dict[str, object] | None = None,
) -> SimpleNamespace:
    if profile == SR_ROUTE_PROFILE_CANDIDATE_V4:
        settings = dict(CANONICAL_SR_SNAKE_V4_EXECUTION_SETTINGS)
    elif profile == SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1:
        settings = dict(
            CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS
        )
    elif profile == SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1:
        settings = dict(
            CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1_EXECUTION_SETTINGS
        )
    elif profile == SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1:
        settings = {
            **CANONICAL_SR_SNAKE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1_EXECUTION_SETTINGS,
            **HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS,
        }
    elif profile == SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2:
        settings = {
            **CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_V2_EXECUTION_SETTINGS,
            **HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS,
        }
    elif profile == SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3:
        settings = dict(
            CANONICAL_SR_SNAKE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3_EXECUTION_SETTINGS
        )
    elif profile == SR_ROUTE_PROFILE_CONVENTIONAL_V3:
        settings = {
            **CANONICAL_SR_SNAKE_V3_EXECUTION_SETTINGS,
            **HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS,
        }
    elif profile == SR_ROUTE_PROFILE_CONVENTIONAL_V2:
        settings = {
            **CANONICAL_SR_SNAKE_V2_EXECUTION_SETTINGS,
            **HISTORICAL_SR_SNAKE_RESPONSE_SCOPE_SETTINGS,
            **HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS,
        }
    else:
        settings = {
            **CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS,
            **HISTORICAL_SR_SNAKE_V1_ACCEPTED_REFIT_SETTINGS,
            **HISTORICAL_SR_SNAKE_RESPONSE_SCOPE_SETTINGS,
            **HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS,
        }
    settings.update(overrides or {})
    return SimpleNamespace(
        sr_route_profile_request=profile,
        _explicit_cli_options=explicit_options,
        **settings,
    )


@pytest.mark.parametrize(
    "profile_name",
    ["sr_snake_v1", SR_ROUTE_PROFILE_CANONICAL_V1],
)
def test_canonical_sr_route_profile_materializes_complete_contract(
    profile_name: str,
) -> None:
    resolved, contract, digest = route_identity(profile_name)

    assert resolved == SR_ROUTE_PROFILE_CANONICAL_V1
    assert contract == canonical_sr_snake_v1_contract()
    assert digest == canonical_sr_snake_v1_contract_sha256()
    assert contract["execution_settings"] == (
        CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS
    )
    # The historical accepted-refit overlay stays outside the frozen digest.
    assert not set(HISTORICAL_SR_SNAKE_V1_ACCEPTED_REFIT_SETTINGS) & set(
        contract["execution_settings"]
    )


@pytest.mark.parametrize(
    "profile_name",
    ["sr_snake_v2", SR_ROUTE_PROFILE_CONVENTIONAL_V2],
)
def test_historical_v2_sr_route_profile_materializes_complete_contract(
    profile_name: str,
) -> None:
    resolved, contract, digest = route_identity(profile_name)

    assert resolved == SR_ROUTE_PROFILE_CONVENTIONAL_V2
    assert contract == canonical_sr_snake_v2_contract()
    assert digest == canonical_sr_snake_v2_contract_sha256()
    assert contract["execution_settings"] == (
        CANONICAL_SR_SNAKE_V2_EXECUTION_SETTINGS
    )
    # The historical response-scope overlay stays outside the frozen digest.
    assert not set(HISTORICAL_SR_SNAKE_RESPONSE_SCOPE_SETTINGS) & set(
        contract["execution_settings"]
    )


@pytest.mark.parametrize(
    "profile_name",
    ["sr_snake_v3", SR_ROUTE_PROFILE_CONVENTIONAL_V3],
)
def test_conventional_v3_sr_route_profile_materializes_complete_contract(
    profile_name: str,
) -> None:
    resolved, contract, digest = route_identity(profile_name)

    assert resolved == SR_ROUTE_PROFILE_CONVENTIONAL_V3
    assert contract == canonical_sr_snake_v3_contract()
    assert digest == canonical_sr_snake_v3_contract_sha256()
    assert contract["execution_settings"] == (
        CANONICAL_SR_SNAKE_V3_EXECUTION_SETTINGS
    )
    assert contract["execution_settings"][
        "phase3_response_coordinate_scope"
    ] == PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1


@pytest.mark.parametrize(
    "profile_name",
    ["sr_snake_v4", SR_ROUTE_PROFILE_CANDIDATE_V4],
)
def test_candidate_v4_materializes_exact_combined_contract(
    profile_name: str,
) -> None:
    resolved, contract, digest = route_identity(profile_name)

    assert resolved == SR_ROUTE_PROFILE_CANDIDATE_V4
    assert contract == canonical_sr_snake_v4_contract()
    assert digest == canonical_sr_snake_v4_contract_sha256()

    kwargs = route_runtime_kwargs(
        route_contract=contract,
        route_contract_sha256=digest,
        route_profile=resolved,
        route_profile_request=profile_name,
    )
    for field, expected in expected_flat_settings(contract).items():
        assert kwargs[field] == expected, field

    expected_combined_settings = {
        "phase3_response_coordinate_scope": (
            PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
        ),
        "phase2_gram_novelty_policy": "fallback_only_v1",
        "phase3_gram_novelty_policy": "fallback_only_v1",
        "phase3_hardware_cost_normalization_mode": (
            "family_robust_symmetric_arctan_v1"
        ),
        "phase1_prune_enabled": True,
        "phase1_prune_mode": "live",
        "phase1_prune_local_window_size": 0,
        "phase1_prune_schur_nomination_route": (
            "full_logical_fs_trust_delete_refit_v1"
        ),
        "phase1_prune_metric_schur_mu": 0.0,
        "phase1_prune_metric_schur_solve_mode": (
            "affine_deletion_global_trust_v1"
        ),
        "phase1_prune_trust_update_policy": (
            "modeled_local_fs_conservative_v1"
        ),
        "phase1_prune_endpoint_overlap_policy": "off",
        "adapt_finite_angle_fallback": False,
        "adapt_disable_hh_seed": True,
        "phase3_enable_rescue": False,
    }
    for field, expected in expected_combined_settings.items():
        assert contract["execution_settings"][field] == expected, field


def test_candidate_v4_contract_selects_first_order_measured_curvature_triplet() -> None:
    contract = canonical_sr_snake_v4_contract()
    settings = contract["execution_settings"]

    assert settings["phase1_score_mode"] == "trust_region_v1"
    assert settings["phase1_energy_model"] == (
        PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1
    )
    assert settings["phase2_curvature_policy"] == (
        PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1
    )
    assert settings["phase2_cheap_curvature_proxy_policy"] == (
        PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF
    )
    assert contract["semantic_invariants"][
        "phase2_curvature_failure_policy"
    ] == "abort_run_v1"
    assert contract["execution_settings"]["adapt_disable_hh_seed"] is True
    assert contract["semantic_invariants"]["hh_preseed_policy"] == (
        "disabled_singleton_growth_from_reference_v1"
    )


@pytest.mark.parametrize(
    "profile_name",
    [
        "sr_snake_no_prune_symmetric_cost_v1",
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
    ],
)
def test_no_prune_symmetric_cost_profile_materializes_exact_contract(
    profile_name: str,
) -> None:
    resolved, contract, digest = route_identity(profile_name)

    assert resolved == SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1
    assert contract == canonical_sr_snake_no_prune_symmetric_cost_v1_contract()
    assert digest == (
        canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
    )

    kwargs = route_runtime_kwargs(
        route_contract=contract,
        route_contract_sha256=digest,
        route_profile=resolved,
        route_profile_request=profile_name,
        maximum_controller_rounds=30,
    )
    for field, expected in expected_flat_settings(contract).items():
        assert kwargs[field] == expected, field

    assert "adapt_max_depth" not in (
        CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS
    )
    assert kwargs["max_depth"] == 30
    assert kwargs["problem"] == "hh"
    assert kwargs["phase3_response_coordinate_scope"] == (
        PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
    )
    assert kwargs["historical_singleton_coordinate_solve_scope"] == (
        "phase3_only_v1"
    )
    assert kwargs["historical_singleton_trust_region_update_policy"] == (
        "displacement_calibrated_unbounded_v2"
    )
    assert kwargs["adapt_accepted_refit_scope"] == "full_ansatz_v1"
    assert kwargs["adapt_accepted_refit_coordinate_chart"] == (
        "supported_fs_whitened_fixed_v1"
    )
    assert kwargs["adapt_accepted_refit_base_chart_policy"] == (
        "expanded_runtime_projected_logical_v1"
    )
    assert kwargs["phase1_energy_model"] == (
        PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1
    )
    assert kwargs["phase2_curvature_policy"] == (
        PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1
    )
    assert kwargs["phase2_cheap_curvature_proxy_policy"] == (
        PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF
    )
    assert kwargs["phase2_gram_novelty_policy"] == "fallback_only_v1"
    assert kwargs["phase3_gram_novelty_policy"] == "fallback_only_v1"
    assert kwargs["phase3_hardware_cost_normalization_mode"] == (
        "family_robust_symmetric_arctan_v1"
    )
    # No-prune: no pruning extension is composed for this route.
    assert route_pruning(kwargs) is None
    assert contract["execution_settings"]["phase1_prune_enabled"] is False
    assert kwargs["adapt_full_refit_every"] == 0
    assert kwargs["adapt_final_full_refit"] is False
    assert kwargs["adapt_finite_angle_fallback"] is False
    assert kwargs["sr_escape_mode"] == "disabled"


def test_no_prune_symmetric_cost_contract_keeps_only_telemetried_novelty_fallback(
) -> None:
    contract = canonical_sr_snake_no_prune_symmetric_cost_v1_contract()
    invariants = contract["semantic_invariants"]

    assert invariants["ordinary_phase2_novelty_multiplier_active"] is False
    assert invariants["ordinary_phase3_novelty_multiplier_active"] is False
    assert invariants[
        "all_energy_models_infeasible_novelty_fallback_active"
    ] is True
    assert invariants[
        "all_energy_models_infeasible_novelty_fallback_telemetry_required"
    ] is True
    assert invariants["phase2_supported_whitening_active"] is False
    assert invariants["phase3_supported_whitening_active"] is True
    assert invariants["pruning_active"] is False
    assert invariants["phase3_shadow_damping_active"] is False
    assert invariants["periodic_full_refit_active"] is False
    assert invariants["terminal_full_refit_active"] is False


@pytest.mark.parametrize("depth", [30, 50])
def test_no_prune_symmetric_cost_profile_leaves_horizon_source_locked(
    depth: int,
) -> None:
    resolved, contract, digest = route_identity(
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1
    )
    kwargs = route_runtime_kwargs(
        route_contract=contract,
        route_contract_sha256=digest,
        route_profile=resolved,
        route_profile_request=resolved,
        maximum_controller_rounds=depth,
    )

    assert kwargs["max_depth"] == depth
    assert "adapt_max_depth" not in contract["execution_settings"]
    assert contract["semantic_invariants"][
        "controller_horizon_source"
    ] == "per_regime_source_lock"


@pytest.mark.parametrize("depth", [0, -1])
def test_no_prune_symmetric_cost_profile_rejects_nonpositive_horizon(
    depth: int,
) -> None:
    # The live typed stop policy is the enforcement point for a positive
    # controller horizon (the retired parser used to exit with code 2).
    with pytest.raises(ValueError, match="positive integer"):
        SRStopPolicy(maximum_controller_rounds=depth)


def test_no_prune_symmetric_cost_runtime_validator_is_exact() -> None:
    contract = canonical_sr_snake_no_prune_symmetric_cost_v1_contract()
    digest = canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
    runtime_settings = {
        field: CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS[
            field
        ]
        for field in (
            "phase1_energy_model",
            "phase2_curvature_policy",
            "phase2_cheap_curvature_proxy_policy",
            "phase3_response_coordinate_scope",
            "phase3_hardware_cost_normalization_mode",
            "phase1_prune_enabled",
            "adapt_full_refit_every",
            "adapt_final_full_refit",
        )
    }
    runtime_settings["adapt_max_depth"] = 8
    assert validate_sr_route_profile_runtime_settings(
        profile_request=SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        contract=contract,
        contract_sha256=digest,
        runtime_settings=runtime_settings,
    ) == contract

    runtime_settings_without_horizon = dict(runtime_settings)
    runtime_settings_without_horizon.pop("adapt_max_depth")
    with pytest.raises(ValueError, match="requires an explicit positive"):
        validate_sr_route_profile_runtime_settings(
            profile_request=SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
            contract=contract,
            contract_sha256=digest,
            runtime_settings=runtime_settings_without_horizon,
        )

    runtime_settings["phase1_prune_enabled"] = True
    with pytest.raises(ValueError, match="effective runtime settings drifted"):
        validate_sr_route_profile_runtime_settings(
            profile_request=SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
            contract=contract,
            contract_sha256=digest,
            runtime_settings=runtime_settings,
        )

    runtime_settings["phase1_prune_enabled"] = False
    runtime_settings["adapt_max_depth"] = 0
    with pytest.raises(ValueError, match="positive source-locked adapt_max_depth"):
        validate_sr_route_profile_runtime_settings(
            profile_request=SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
            contract=contract,
            contract_sha256=digest,
            runtime_settings=runtime_settings,
        )


def test_no_prune_symmetric_cost_powell_resolution_labels_outer_route() -> None:
    base = adapt_pipeline._resolve_sr_powell_coordinate_chart_runtime_policy(
        historical_singleton_overlay_active=True,
        sr_escape_mode="disabled",
        coordinate_solve_scope="phase3_only_v1",
        requested_policy="expanded_runtime_projected_logical_v1",
        source_locked_replay=False,
    )
    assert base["route_profile"] == SR_ROUTE_PROFILE_CANONICAL_V1
    base_instance = dict(base["route_instance"])
    registered_instance = {
        **base_instance,
        "base_controller_route_profile": SR_ROUTE_PROFILE_CANONICAL_V1,
        "route_profile": SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
    }

    resolved = adapt_pipeline._registered_sr_powell_coordinate_chart_resolution(
        base,
        registered_profile=SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
        route_instance=registered_instance,
    )

    assert resolved["route_profile"] == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1
    )
    assert resolved["base_controller_route_profile"] == (
        SR_ROUTE_PROFILE_CANONICAL_V1
    )
    assert resolved["route_profile_conformance"] == "registered_profile"
    assert resolved["route_instance"]["route_profile"] == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1
    )
    assert resolved["route_instance"]["base_controller_route_profile"] == (
        SR_ROUTE_PROFILE_CANONICAL_V1
    )


def test_h2o_derivative_resolved_profile_changes_only_pool_identity() -> None:
    resolved, contract, digest = route_identity(
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2
    )
    parent = canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract()

    assert resolved == SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2
    assert contract == canonical_sr_snake_h2o_derivative_resolved_v2_contract()
    assert digest == (
        canonical_sr_snake_h2o_derivative_resolved_v2_contract_sha256()
    )
    assert contract["execution_settings"]["adapt_pool"] == (
        "full_meta_derivative_resolved_v2"
    )
    child_settings = dict(contract["execution_settings"])
    parent_settings = dict(parent["execution_settings"])
    child_settings.pop("adapt_pool")
    parent_settings.pop("adapt_pool")
    assert child_settings == parent_settings
    assert contract["lineage_authority"]["parent_contract_sha256"] == (
        canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract_sha256()
    )


def test_h2o_paper_i_profile_is_no_prune_no_beam_source_locked_overlay() -> None:
    resolved, contract, digest = route_identity(
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3
    )
    parent = canonical_sr_snake_no_prune_symmetric_cost_v1_contract()

    assert resolved == SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3
    assert contract == (
        canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract()
    )
    assert digest == (
        canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract_sha256()
    )

    expected_differences = {
        "problem",
        "adapt_max_depth",
        "adapt_pool",
        "phase3_runtime_split_child_padding_policy",
        "phase3_backend_cost_mode",
    }
    child_settings = dict(contract["execution_settings"])
    parent_settings = dict(parent["execution_settings"])
    differences = {
        field
        for field in set(child_settings) | set(parent_settings)
        if child_settings.get(field) != parent_settings.get(field)
    }
    assert differences == expected_differences
    assert contract["lineage_authority"]["parent_contract_sha256"] == (
        canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
    )

    settings = contract["execution_settings"]
    assert settings["problem"] == "molecular_vibronic_h2o_linear_fd"
    assert settings["adapt_max_depth"] == 50
    assert settings["adapt_pool"] == "full_meta_derivative_resolved_v2"
    assert settings["phase1_prune_enabled"] is False
    assert contract["semantic_invariants"]["pruning_active"] is False
    assert contract["semantic_invariants"]["terminal_prune_active"] is False
    assert settings["phase2_enable_batching"] is False
    assert settings["phase3_enable_batching"] is False
    assert settings["phase2_gram_novelty_policy"] == "fallback_only_v1"
    assert settings["phase3_gram_novelty_policy"] == "fallback_only_v1"
    assert settings["phase3_novelty_ablation_mode"] == "off"
    assert settings["phase3_response_coordinate_scope"] == (
        PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
    )
    assert settings["historical_singleton_coordinate_solve_policy"] == (
        "supported_metric_whitened_eigh_v1"
    )
    assert settings["adapt_accepted_refit_scope"] == "full_ansatz_v1"
    assert settings["adapt_full_refit_every"] == 0
    assert settings["adapt_final_full_refit"] == "false"


@pytest.mark.parametrize(
    ("profile", "contract_settings"),
    [
        (SR_ROUTE_PROFILE_CANONICAL_V1, CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS),
        (
            SR_ROUTE_PROFILE_CONVENTIONAL_V2,
            CANONICAL_SR_SNAKE_V2_EXECUTION_SETTINGS,
        ),
        (
            SR_ROUTE_PROFILE_CONVENTIONAL_V3,
            CANONICAL_SR_SNAKE_V3_EXECUTION_SETTINGS,
        ),
    ],
)
def test_historical_phase12_policy_overlay_is_explicit_but_outside_frozen_digest(
    profile: str,
    contract_settings: dict[str, object],
) -> None:
    assert not set(HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS).intersection(
        contract_settings
    )
    namespace = normalize_sr_route_profile_namespace(
        _route_profile_namespace(profile)
    )
    assert not hasattr(namespace, "phase1_energy_model")
    assert namespace.phase2_curvature_policy == (
        PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1
    )
    assert not hasattr(namespace, "phase2_cheap_curvature_proxy_policy")
    contract_by_profile = {
        SR_ROUTE_PROFILE_CANONICAL_V1: canonical_sr_snake_v1_contract,
        SR_ROUTE_PROFILE_CONVENTIONAL_V2: canonical_sr_snake_v2_contract,
        SR_ROUTE_PROFILE_CONVENTIONAL_V3: canonical_sr_snake_v3_contract,
    }
    digest_by_profile = {
        SR_ROUTE_PROFILE_CANONICAL_V1: canonical_sr_snake_v1_contract_sha256,
        SR_ROUTE_PROFILE_CONVENTIONAL_V2: canonical_sr_snake_v2_contract_sha256,
        SR_ROUTE_PROFILE_CONVENTIONAL_V3: canonical_sr_snake_v3_contract_sha256,
    }
    contract = contract_by_profile[profile]()
    assert validate_sr_route_profile_runtime_settings(
        profile_request=profile,
        contract=contract,
        contract_sha256=digest_by_profile[profile](),
        runtime_settings=HISTORICAL_SR_SNAKE_PHASE12_ENERGY_MODEL_SETTINGS,
    ) == contract


def test_candidate_v4_runtime_validator_requires_exact_phase12_triplet() -> None:
    runtime_settings = {
        field: CANONICAL_SR_SNAKE_V4_EXECUTION_SETTINGS[field]
        for field in (
            "phase1_score_mode",
            "phase1_energy_model",
            "phase2_curvature_policy",
            "phase2_cheap_curvature_proxy_policy",
        )
    }
    assert validate_sr_route_profile_runtime_settings(
        profile_request=SR_ROUTE_PROFILE_CANDIDATE_V4,
        contract=canonical_sr_snake_v4_contract(),
        contract_sha256=canonical_sr_snake_v4_contract_sha256(),
        runtime_settings=runtime_settings,
    ) == canonical_sr_snake_v4_contract()

    runtime_settings["phase2_cheap_curvature_proxy_policy"] = "removed_proxy"
    with pytest.raises(ValueError, match="effective runtime settings drifted"):
        validate_sr_route_profile_runtime_settings(
            profile_request=SR_ROUTE_PROFILE_CANDIDATE_V4,
            contract=canonical_sr_snake_v4_contract(),
            contract_sha256=canonical_sr_snake_v4_contract_sha256(),
            runtime_settings=runtime_settings,
        )


def test_unqualified_sr_snake_alias_resolves_to_v3_not_candidate_v4() -> None:
    resolved, contract, _ = route_identity("sr_snake")

    assert resolved == SR_ROUTE_PROFILE_CONVENTIONAL_V3
    assert contract == canonical_sr_snake_v3_contract()
    assert contract != canonical_sr_snake_v4_contract()


def test_candidate_v4_disables_finite_angle_fallback_in_contract_and_runtime() -> None:
    resolved, contract, digest = route_identity(SR_ROUTE_PROFILE_CANDIDATE_V4)
    kwargs = route_runtime_kwargs(
        route_contract=contract,
        route_contract_sha256=digest,
        route_profile=resolved,
        route_profile_request=resolved,
    )

    assert kwargs["finite_angle_fallback"] is False
    assert kwargs["phase3_enable_rescue"] is False
    assert contract["execution_settings"][
        "adapt_finite_angle_fallback"
    ] is False
    assert contract["execution_settings"]["phase3_enable_rescue"] is False
    assert contract["semantic_invariants"][
        "finite_angle_fallback_active"
    ] is False


@pytest.mark.parametrize(
    ("profile_name", "expected_policy"),
    [
        ("sr_snake_v1", "family_robust_v1"),
        ("sr_snake_v2", "family_robust_v1"),
        ("sr_snake_v3", "family_robust_v1"),
        ("sr_snake", "family_robust_v1"),
        ("sr_snake_v4", "family_robust_symmetric_arctan_v1"),
    ],
)
def test_sr_profiles_resolve_expected_hardware_cost_normalization(
    profile_name: str,
    expected_policy: str,
) -> None:
    _, contract, _ = route_identity(profile_name)

    assert contract["execution_settings"][
        "phase3_hardware_cost_normalization_mode"
    ] == expected_policy


def test_versioned_sr_route_profiles_are_distinct_and_v1_digest_is_stable() -> None:
    assert SR_ROUTE_PROFILE_CANONICAL_V1 != SR_ROUTE_PROFILE_CONVENTIONAL_V2
    assert SR_ROUTE_PROFILE_CONVENTIONAL_V2 != SR_ROUTE_PROFILE_CONVENTIONAL_V3
    assert canonical_sr_snake_v1_contract_sha256() == (
        "fab7b5a6c4bd2ab019139367aa2a507356a5c969b6b88cd72d32365ae766e13e"
    )
    assert canonical_sr_snake_v2_contract_sha256() == (
        "32d2bdf2b05818be6f4add74137447a313605d7ed35ffb880651863b793a0f64"
    )
    assert canonical_sr_snake_v3_contract_sha256() == (
        "435910592e88f0136a0d45f611f79fe96b21d75fd25bad58276c871f39dc080e"
    )
    assert canonical_sr_snake_v4_contract_sha256() == (
            "0b36d0c505244809c131c473165db546587c5a38ffa7f885e77d382e4243e4c3"
    )
    assert canonical_sr_snake_v1_contract_sha256() != (
        canonical_sr_snake_v2_contract_sha256()
    )
    assert canonical_sr_snake_v3_contract_sha256() not in {
        canonical_sr_snake_v1_contract_sha256(),
        canonical_sr_snake_v2_contract_sha256(),
    }
    assert canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256() not in {
        canonical_sr_snake_v1_contract_sha256(),
        canonical_sr_snake_v2_contract_sha256(),
        canonical_sr_snake_v3_contract_sha256(),
        canonical_sr_snake_v4_contract_sha256(),
    }
    assert (
        canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract_sha256()
        != canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
    )
    assert canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract_sha256() == (
        "ce283eeb38a7426bcc4c36f55d35265fa99a0029c571343c17803bd6d87df6f5"
    )
    assert canonical_sr_snake_v1_contract()["execution_settings"] == (
        CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS
    )
    assert canonical_sr_snake_v2_contract()["execution_settings"] == (
        CANONICAL_SR_SNAKE_V2_EXECUTION_SETTINGS
    )
    assert canonical_sr_snake_v3_contract()["execution_settings"] == (
        CANONICAL_SR_SNAKE_V3_EXECUTION_SETTINGS
    )
    assert canonical_sr_snake_v3_contract()["semantic_invariants"][
        "phase3_response_pre_support_invariant"
    ] == "response_count_equals_active_logical_count_plus_one_v1"


def test_conventional_sr_route_profile_records_three_weak_holstein_anchors() -> None:
    authority = canonical_sr_snake_v2_contract()["anchor_authority"]

    assert authority["source_manifest_path"].endswith("source_manifest.json")
    assert authority["source_archive_path"].endswith("source_tree.tar.gz")
    anchors = authority["weak_holstein_anchors"]
    assert [anchor["regime"] for anchor in anchors] == [
        "weak-weak",
        "intermediate-weak",
        "strong-weak-u8",
    ]
    assert all(anchor["source_root"].startswith("raw_outputs/") for anchor in anchors)
    assert all(anchor["n_ph_work"] == 2 for anchor in anchors)


def test_canonical_sr_route_profile_round_trips_into_runtime_kwargs() -> None:
    # Structural note: the v1/v3 contracts predate the typed pruning
    # interview and cannot pass through the live kwargs builder; identity
    # round-trip through the live builder is asserted on the v4 route below.
    resolved, contract, digest = route_identity(SR_ROUTE_PROFILE_CANONICAL_V1)

    assert resolved == SR_ROUTE_PROFILE_CANONICAL_V1
    assert contract == canonical_sr_snake_v1_contract()
    assert digest == canonical_sr_snake_v1_contract_sha256()


def test_conventional_sr_route_profile_round_trips_into_runtime_kwargs() -> None:
    resolved, contract, digest = route_identity(SR_ROUTE_PROFILE_CONVENTIONAL_V3)

    assert resolved == SR_ROUTE_PROFILE_CONVENTIONAL_V3
    assert contract == canonical_sr_snake_v3_contract()
    assert digest == canonical_sr_snake_v3_contract_sha256()
    settings = contract["execution_settings"]
    assert settings["phase3_response_coordinate_scope"] == (
        PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
    )
    assert settings["adapt_accepted_refit_scope"] == "full_ansatz_v1"
    assert settings["adapt_accepted_refit_coordinate_chart"] == (
        "supported_fs_whitened_fixed_v1"
    )
    assert settings["adapt_accepted_refit_base_chart_policy"] == (
        "expanded_runtime_projected_logical_v1"
    )


def test_candidate_v4_round_trips_new_cli_fields_into_runtime_kwargs() -> None:
    kwargs = route_runtime_kwargs(
        route_contract=canonical_sr_snake_v4_contract(),
        route_contract_sha256=canonical_sr_snake_v4_contract_sha256(),
        route_profile=SR_ROUTE_PROFILE_CANDIDATE_V4,
        route_profile_request=SR_ROUTE_PROFILE_CANDIDATE_V4,
    )

    assert kwargs["sr_route_profile_request"] == SR_ROUTE_PROFILE_CANDIDATE_V4
    assert kwargs["sr_route_profile_resolved"] == SR_ROUTE_PROFILE_CANDIDATE_V4
    assert kwargs["sr_route_profile_contract"] == canonical_sr_snake_v4_contract()
    assert kwargs["sr_route_profile_contract_sha256"] == (
        canonical_sr_snake_v4_contract_sha256()
    )
    expected_runtime_fields = {
        "phase3_response_coordinate_scope": (
            PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
        ),
        "phase2_gram_novelty_policy": "fallback_only_v1",
        "phase3_gram_novelty_policy": "fallback_only_v1",
        "phase3_hardware_cost_normalization_mode": (
            "family_robust_symmetric_arctan_v1"
        ),
        "finite_angle_fallback": False,
        "phase3_enable_rescue": False,
    }
    for field, expected in expected_runtime_fields.items():
        assert kwargs[field] == expected, field
    # The live projection strips pruning settings from the flat kwargs;
    # only three inert infrastructure policies (pinned "off") remain flat.
    assert "phase1_prune_enabled" not in kwargs
    assert set(kwargs) & PRUNING_RUNTIME_KEYS == {
        "phase1_prune_endpoint_overlap_policy",
        "phase1_prune_metric_mu_update_policy",
        "phase1_prune_trust_update_policy",
    }
    pruning = route_pruning(kwargs)
    assert pruning is not None
    assert pruning["phase1_prune_mode"] == "live"
    assert pruning["phase1_prune_local_window_size"] == 0
    assert pruning["phase1_prune_schur_nomination_route"] == (
        "full_logical_fs_trust_delete_refit_v1"
    )
    assert pruning["phase1_prune_metric_schur_solve_mode"] == (
        "affine_deletion_global_trust_v1"
    )
    assert pruning["phase1_prune_trust_update_policy"] == (
        "modeled_local_fs_conservative_v1"
    )
    assert pruning["phase1_prune_metric_mu_update_policy"] == (
        "same_trial_underprediction_monotone_v1"
    )
    assert pruning["phase1_prune_endpoint_overlap_policy"] == "off"


def test_sr_route_profile_contract_validation_fails_on_tamper() -> None:
    contract = canonical_sr_snake_v1_contract()
    digest = canonical_sr_snake_v1_contract_sha256()
    assert validate_sr_route_profile_contract(
        profile_request=SR_ROUTE_PROFILE_CANONICAL_V1,
        contract=contract,
        contract_sha256=digest,
    ) == contract

    tampered = copy.deepcopy(contract)
    tampered["execution_settings"]["adapt_maxiter"] = 201
    with pytest.raises(ValueError, match="contract drifted"):
        validate_sr_route_profile_contract(
            profile_request=SR_ROUTE_PROFILE_CANONICAL_V1,
            contract=tampered,
            contract_sha256=digest,
        )


def test_conventional_sr_route_profile_contract_validation_fails_on_tamper() -> None:
    contract = canonical_sr_snake_v2_contract()
    digest = canonical_sr_snake_v2_contract_sha256()
    assert validate_sr_route_profile_contract(
        profile_request=SR_ROUTE_PROFILE_CONVENTIONAL_V2,
        contract=contract,
        contract_sha256=digest,
    ) == contract

    tampered = copy.deepcopy(contract)
    tampered["execution_settings"]["adapt_accepted_refit_scope"] = (
        "selector_policy_v1"
    )
    with pytest.raises(ValueError, match="contract drifted"):
        validate_sr_route_profile_contract(
            profile_request=SR_ROUTE_PROFILE_CONVENTIONAL_V2,
            contract=tampered,
            contract_sha256=digest,
        )


def test_conventional_v3_contract_validation_fails_on_response_scope_tamper() -> None:
    contract = canonical_sr_snake_v3_contract()
    digest = canonical_sr_snake_v3_contract_sha256()
    assert validate_sr_route_profile_contract(
        profile_request=SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        contract=contract,
        contract_sha256=digest,
    ) == contract

    tampered = copy.deepcopy(contract)
    tampered["execution_settings"]["phase3_response_coordinate_scope"] = (
        PHASE3_RESPONSE_COORDINATE_SCOPE_LEGACY_REOPT_COUPLED_V1
    )
    with pytest.raises(ValueError, match="contract drifted"):
        validate_sr_route_profile_contract(
            profile_request=SR_ROUTE_PROFILE_CONVENTIONAL_V3,
            contract=tampered,
            contract_sha256=digest,
        )


def test_sr_route_profile_runtime_validation_rejects_effective_drift() -> None:
    with pytest.raises(ValueError, match="effective runtime settings drifted"):
        validate_sr_route_profile_runtime_settings(
            profile_request=SR_ROUTE_PROFILE_CANONICAL_V1,
            contract=canonical_sr_snake_v1_contract(),
            contract_sha256=canonical_sr_snake_v1_contract_sha256(),
            runtime_settings={"adapt_maxiter": 201},
        )


def test_runtime_validation_distinguishes_v1_and_v2_accepted_refit_charts() -> None:
    assert validate_sr_route_profile_runtime_settings(
        profile_request=SR_ROUTE_PROFILE_CANONICAL_V1,
        contract=canonical_sr_snake_v1_contract(),
        contract_sha256=canonical_sr_snake_v1_contract_sha256(),
        runtime_settings=HISTORICAL_SR_SNAKE_V1_ACCEPTED_REFIT_SETTINGS,
    ) == canonical_sr_snake_v1_contract()
    v2_refit = {
        field: CANONICAL_SR_SNAKE_V2_EXECUTION_SETTINGS[field]
        for field in HISTORICAL_SR_SNAKE_V1_ACCEPTED_REFIT_SETTINGS
    }
    assert validate_sr_route_profile_runtime_settings(
        profile_request=SR_ROUTE_PROFILE_CONVENTIONAL_V2,
        contract=canonical_sr_snake_v2_contract(),
        contract_sha256=canonical_sr_snake_v2_contract_sha256(),
        runtime_settings=v2_refit,
    ) == canonical_sr_snake_v2_contract()

    with pytest.raises(ValueError, match="effective runtime settings drifted"):
        validate_sr_route_profile_runtime_settings(
            profile_request=SR_ROUTE_PROFILE_CONVENTIONAL_V2,
            contract=canonical_sr_snake_v2_contract(),
            contract_sha256=canonical_sr_snake_v2_contract_sha256(),
            runtime_settings=HISTORICAL_SR_SNAKE_V1_ACCEPTED_REFIT_SETTINGS,
        )


def test_runtime_validation_requires_full_phase3_response_scope_for_v3() -> None:
    assert validate_sr_route_profile_runtime_settings(
        profile_request=SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        contract=canonical_sr_snake_v3_contract(),
        contract_sha256=canonical_sr_snake_v3_contract_sha256(),
        runtime_settings={
            "phase3_response_coordinate_scope": (
                PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
            )
        },
    ) == canonical_sr_snake_v3_contract()

    with pytest.raises(ValueError, match="effective runtime settings drifted"):
        validate_sr_route_profile_runtime_settings(
            profile_request=SR_ROUTE_PROFILE_CONVENTIONAL_V3,
            contract=canonical_sr_snake_v3_contract(),
            contract_sha256=canonical_sr_snake_v3_contract_sha256(),
            runtime_settings={
                "phase3_response_coordinate_scope": (
                    PHASE3_RESPONSE_COORDINATE_SCOPE_LEGACY_REOPT_COUPLED_V1
                )
            },
        )


def test_resume_contract_accepts_canonical_settings_alias() -> None:
    validation = validate_resume_sr_route_profile_contract(
        _contract_payload(),
        expected_profile_request=SR_ROUTE_PROFILE_CANONICAL_V1,
        expected_contract=canonical_sr_snake_v1_contract(),
        expected_contract_sha256=canonical_sr_snake_v1_contract_sha256(),
    )

    assert validation["status"] == "pass"
    assert validation["artifact_profile_request"] == (
        SR_ROUTE_PROFILE_CANONICAL_V1
    )
    assert validation["contract_sha256"] == (
        canonical_sr_snake_v1_contract_sha256()
    )


@pytest.mark.parametrize("location", ["settings", "checkpoint"])
def test_resume_contract_accepts_conventional_settings_and_checkpoint_aliases(
    location: str,
) -> None:
    validation = validate_resume_sr_route_profile_contract(
        _v2_contract_payload(location=location),
        expected_profile_request=SR_ROUTE_PROFILE_CONVENTIONAL_V2,
        expected_contract=canonical_sr_snake_v2_contract(),
        expected_contract_sha256=canonical_sr_snake_v2_contract_sha256(),
    )

    assert validation["status"] == "pass"
    assert validation["artifact_profile_request"] == (
        SR_ROUTE_PROFILE_CONVENTIONAL_V2
    )
    assert validation["contract_sha256"] == (
        canonical_sr_snake_v2_contract_sha256()
    )


@pytest.mark.parametrize("location", ["settings", "checkpoint"])
def test_resume_contract_accepts_v3_scope_and_contract_round_trip(
    location: str,
) -> None:
    validation = validate_resume_sr_route_profile_contract(
        _v3_contract_payload(location=location),
        expected_profile_request=SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        expected_contract=canonical_sr_snake_v3_contract(),
        expected_contract_sha256=canonical_sr_snake_v3_contract_sha256(),
    )

    assert validation["status"] == "pass"
    assert validation["artifact_profile_request"] == (
        SR_ROUTE_PROFILE_CONVENTIONAL_V3
    )
    assert validation["contract_sha256"] == (
        canonical_sr_snake_v3_contract_sha256()
    )


def test_resume_contract_rejects_v3_artifact_with_legacy_response_scope() -> None:
    payload = _v3_contract_payload()
    payload["settings"]["phase3_response_coordinate_scope"] = (
        PHASE3_RESPONSE_COORDINATE_SCOPE_LEGACY_REOPT_COUPLED_V1
    )

    with pytest.raises(ValueError, match="requires full_active_plus_singleton_v1"):
        validate_resume_sr_route_profile_contract(
            payload,
            expected_profile_request=SR_ROUTE_PROFILE_CONVENTIONAL_V3,
            expected_contract=canonical_sr_snake_v3_contract(),
            expected_contract_sha256=canonical_sr_snake_v3_contract_sha256(),
        )


def test_resume_contract_rejects_v1_v2_profile_mismatch() -> None:
    with pytest.raises(ValueError, match="current invocation requests"):
        validate_resume_sr_route_profile_contract(
            _contract_payload(),
            expected_profile_request=SR_ROUTE_PROFILE_CONVENTIONAL_V2,
            expected_contract=canonical_sr_snake_v2_contract(),
            expected_contract_sha256=canonical_sr_snake_v2_contract_sha256(),
        )


def test_resume_contract_fails_closed_when_canonical_request_is_missing() -> None:
    with pytest.raises(ValueError, match="lacks its complete"):
        validate_resume_sr_route_profile_contract(
            {},
            expected_profile_request=SR_ROUTE_PROFILE_CANONICAL_V1,
            expected_contract=canonical_sr_snake_v1_contract(),
            expected_contract_sha256=canonical_sr_snake_v1_contract_sha256(),
        )


def test_resume_contract_rejects_conflicting_serialized_aliases() -> None:
    payload = _contract_payload()
    tampered = canonical_sr_snake_v1_contract()
    tampered["execution_settings"]["adapt_maxiter"] = 201
    payload["adapt_vqe"] = {
        "sr_route_profile_request": SR_ROUTE_PROFILE_CANONICAL_V1,
        "sr_route_profile_contract": tampered,
        "sr_route_profile_contract_sha256": (
            canonical_sr_snake_v1_contract_sha256()
        ),
    }

    with pytest.raises(ValueError, match="conflicting serialized"):
        validate_resume_sr_route_profile_contract(payload)


def test_resume_contract_rejects_canonical_artifact_when_profile_is_off() -> None:
    with pytest.raises(ValueError, match="did not explicitly request"):
        validate_resume_sr_route_profile_contract(
            _contract_payload(),
            expected_profile_request=SR_ROUTE_PROFILE_REQUEST_OFF,
        )


def test_candidate_v4_identity_is_readable_but_not_execution_authority() -> None:
    """The historical v4 identity remains readable but grants no execution.

    The legacy executor that used to reject it is deleted, so the "no execution"
    half is now enforced by the live controller runtime factory: v4 is not among
    the authorized (profile, digest) pairs, so the factory's gate refuses it.
    """

    kwargs = route_runtime_kwargs(
        route_contract=canonical_sr_snake_v4_contract(),
        route_contract_sha256=canonical_sr_snake_v4_contract_sha256(),
        route_profile=SR_ROUTE_PROFILE_CANDIDATE_V4,
        route_profile_request=SR_ROUTE_PROFILE_CANDIDATE_V4,
        problem_context=hh_problem_context(
            n_ph_max=2,
            u=0.25,
            g_ep=0.353553390593,
        ),
    )

    assert kwargs["sr_route_profile_resolved"] == SR_ROUTE_PROFILE_CANDIDATE_V4
    assert kwargs["sr_route_profile_contract"] == canonical_sr_snake_v4_contract()
    assert kwargs["sr_route_profile_contract_sha256"] == (
        canonical_sr_snake_v4_contract_sha256()
    )

    assert_route_binding_rejected(kwargs)
