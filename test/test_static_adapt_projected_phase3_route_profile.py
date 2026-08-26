from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
)
from pipelines.static_adapt.route_a_schur_selector import (
    ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1_EXECUTION_SETTINGS,
    CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1_EXECUTION_SETTINGS,
    CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1_EXECUTION_SETTINGS,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract_sha256,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_v1_contract,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_v1_contract_sha256,
    canonical_sr_snake_no_prune_symmetric_cost_v1_contract,
    canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256,
    normalize_sr_route_profile_request,
    validate_sr_route_profile_runtime_settings,
)
from test_support.route_contract_kwargs import route_identity, route_runtime_kwargs

_RETIRED_RAW_SINGLETON_PROFILE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_full_insertion_diagnostic_v1"
)
_RETIRED_RAW_SINGLETON_ALIAS = (
    "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_"
    "full_insertion_diagnostic_v1"
)


def test_commutation_reduced_insertion_profile_is_registered_for_complete_runtime_handoff() -> None:
    assert (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1
        in adapt_pipeline._REGISTERED_COMPLETE_SR_ROUTE_PROFILES
    )


def test_commutation_reduced_insertion_profile_owns_nonappend_insertion_gate() -> None:
    assert (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1
        in adapt_pipeline._PROFILE_OWNED_NONAPPEND_INSERTION_PROFILES
    )


def test_full_insertion_profile_is_not_registered_for_runtime_handoff() -> None:
    assert (
        _RETIRED_RAW_SINGLETON_PROFILE
        not in adapt_pipeline._REGISTERED_COMPLETE_SR_ROUTE_PROFILES
    )


def test_full_insertion_profile_does_not_own_nonappend_insertion_gate() -> None:
    assert (
        _RETIRED_RAW_SINGLETON_PROFILE
        not in adapt_pipeline._PROFILE_OWNED_NONAPPEND_INSERTION_PROFILES
    )


def test_full_insertion_profile_has_no_registered_powell_chart_resolution() -> None:
    assert (
        _RETIRED_RAW_SINGLETON_PROFILE
        not in adapt_pipeline._REGISTERED_SR_POWELL_COORDINATE_CHART_PROFILES
    )


def test_projected_phase3_profile_is_one_setting_ablation_of_valid_main_sr() -> None:
    parent = canonical_sr_snake_no_prune_symmetric_cost_v1_contract()
    child = canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_v1_contract()

    assert canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256() == (
        "023bc7ac535ee4d88d78dd5336a59dd2fb0543c133fa0a60b009efab75422c91"
    )
    assert child["lineage_authority"]["parent_contract_sha256"] == (
        canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
    )
    changed = {
        key: (parent["execution_settings"].get(key), value)
        for key, value in child["execution_settings"].items()
        if parent["execution_settings"].get(key) != value
    }
    assert changed == {
        "historical_singleton_coordinate_solve_policy": (
            "supported_metric_whitened_eigh_v1",
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
        )
    }


def test_projected_phase3_profile_keeps_phase2_and_whitened_accepted_refit() -> None:
    contract = canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_v1_contract()
    settings = contract["execution_settings"]
    invariants = contract["semantic_invariants"]

    assert settings["historical_singleton_coordinate_solve_scope"] == "phase3_only_v1"
    assert settings["phase_live_hysteresis_enabled"] is False
    assert settings["adapt_accepted_refit_scope"] == "full_ansatz_v1"
    assert settings["adapt_accepted_refit_coordinate_chart"] == (
        "supported_fs_whitened_fixed_v1"
    )
    assert invariants["phase2_supported_whitening_active"] is False
    assert invariants["phase3_support_projection_active"] is True
    assert invariants["phase3_supported_whitening_active"] is False
    assert invariants["phase3_supported_metric_inverse_sqrt_active"] is False
    assert invariants["phase3_metric_ridge_active"] is False
    assert invariants["accepted_refit_coordinate_chart"] == (
        "supported_fs_whitened_fixed_v1"
    )


def test_projected_phase3_alias_and_contract_round_trip() -> None:
    resolved, contract, digest = route_identity(
        "sr_snake_no_prune_symmetric_cost_projected_phase3_v1"
    )

    assert resolved == SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1
    assert contract == (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_v1_contract()
    )
    assert digest == (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_v1_contract_sha256()
    )

    kwargs = route_runtime_kwargs(
        route_contract=contract,
        route_contract_sha256=digest,
        route_profile=resolved,
        route_profile_request="sr_snake_no_prune_symmetric_cost_projected_phase3_v1",
        maximum_controller_rounds=50,
    )
    assert kwargs["historical_singleton_coordinate_solve_policy"] == (
        JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
    )
    assert kwargs["max_depth"] == 50

    runtime = dict(
        CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1_EXECUTION_SETTINGS
    )
    runtime["adapt_max_depth"] = 50
    assert validate_sr_route_profile_runtime_settings(
        profile_request=resolved,
        contract=contract,
        contract_sha256=digest,
        runtime_settings=runtime,
    ) == contract


# Structural note: the CLI rejection test for a whitened-solver override was
# retired with the argparse surface. The runtime kwargs builder writes
# historical_singleton_coordinate_solve_policy only from the authenticated
# contract's execution_settings, so runtime and contract cannot disagree by
# construction; no per-flag override path remains to reject.


def test_projected_no_overlap_profile_changes_only_trust_update_policy() -> None:
    parent = canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_v1_contract()
    child = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract()
    )

    changed = {
        key: (parent["execution_settings"].get(key), value)
        for key, value in child["execution_settings"].items()
        if parent["execution_settings"].get(key) != value
    }
    assert changed == {
        "historical_singleton_trust_region_update_policy": (
            "displacement_calibrated_unbounded_v2",
            ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1,
        )
    }
    invariants = child["semantic_invariants"]
    assert invariants["phase3_support_projection_active"] is True
    assert invariants["phase3_supported_whitening_active"] is False
    assert invariants["accepted_refit_coordinate_chart"] == (
        "supported_fs_whitened_fixed_v1"
    )
    assert invariants["endpoint_overlap_measurement_active"] is False
    assert invariants["endpoint_overlap_query_charge_required"] == 0


def test_projected_no_overlap_alias_and_contract_round_trip() -> None:
    resolved, contract, digest = route_identity(
        "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1"
    )

    assert resolved == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
    )
    assert contract == (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract()
    )
    assert digest == (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256()
    )

    kwargs = route_runtime_kwargs(
        route_contract=contract,
        route_contract_sha256=digest,
        route_profile=resolved,
        route_profile_request=(
            "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1"
        ),
        maximum_controller_rounds=50,
    )
    assert kwargs["historical_singleton_coordinate_solve_policy"] == (
        JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
    )
    assert kwargs["historical_singleton_trust_region_update_policy"] == (
        ROUTE_A_TRUST_REGION_SOURCE_METRIC_INVERSE_SQRT_NO_OVERLAP_V1
    )
    assert kwargs["max_depth"] == 50

    runtime = dict(
        CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1_EXECUTION_SETTINGS
    )
    runtime["adapt_max_depth"] = 50
    assert validate_sr_route_profile_runtime_settings(
        profile_request=resolved,
        contract=contract,
        contract_sha256=digest,
        runtime_settings=runtime,
    ) == contract


def test_projected_no_overlap_commutation_route_changes_only_insertion() -> None:
    parent = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract()
    )
    child = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract()
    )

    parent_settings = dict(parent["execution_settings"])
    child_settings = dict(child["execution_settings"])
    assert parent_settings.pop("adapt_insertion_mode") == "append_only"
    assert child_settings.pop("adapt_insertion_mode") == "full_commutation_reduced"
    assert child_settings == parent_settings
    assert child["lineage_authority"]["parent_contract_sha256"] == (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256()
    )


def test_projected_no_overlap_commutation_alias_round_trip() -> None:
    request = (
        "sr_snake_no_prune_symmetric_cost_projected_phase3_"
        "no_overlap_trust_commutation_reduced_insertion_v1"
    )
    resolved, contract, digest = route_identity(request)

    assert resolved == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1
    )
    assert contract == (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract()
    )
    assert digest == (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract_sha256()
    )

    kwargs = route_runtime_kwargs(
        route_contract=contract,
        route_contract_sha256=digest,
        route_profile=resolved,
        route_profile_request=request,
        maximum_controller_rounds=50,
    )
    assert kwargs["adapt_insertion_mode"] == "full_commutation_reduced"
    assert kwargs["max_depth"] == 50

    runtime = dict(
        CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1_EXECUTION_SETTINGS
    )
    runtime["adapt_max_depth"] = 50
    assert validate_sr_route_profile_runtime_settings(
        profile_request=resolved,
        contract=contract,
        contract_sha256=digest,
        runtime_settings=runtime,
    ) == contract


def test_projected_no_overlap_full_insertion_alias_is_retired() -> None:
    with pytest.raises(ValueError, match="sr_route_profile must be one of"):
        normalize_sr_route_profile_request(
            _RETIRED_RAW_SINGLETON_ALIAS
        )
