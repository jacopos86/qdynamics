from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
import textwrap

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt.cli_config import _build_adapt_arg_parser
from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt.phase3_material_window import (
    DEFAULT_PHASE3_MATERIAL_WINDOW_POLICY,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1_EXECUTION_SETTINGS,
    CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1_EXECUTION_SETTINGS,
    PHASE3_RESPONSE_COORDINATE_SCOPE_CANDIDATE_MATERIAL_COUPLING_WINDOW_V1,
    PHASE3_RESPONSE_COORDINATE_SCOPE_CHOICES,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1,
    SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
    SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1_contract,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1_contract_sha256,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256,
    canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1_contract,
    canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1_contract_sha256,
    normalize_phase3_response_coordinate_scope,
    validate_sr_route_profile_runtime_settings,
)


TEST1_ALIAS = (
    "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_"
    "material_window_v1"
)
TEST2_ALIAS = (
    "sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_"
    "window_fs_prune_verify_v1"
)

PRUNE_VERIFY_CHANGED_FIELDS = {
    "phase1_prune_enabled",
    "phase1_prune_mode",
    "phase1_prune_max_candidates",
    "phase1_prune_local_window_size",
    "phase1_prune_recovery_trust_radius",
    "phase1_prune_schur_nomination_route",
    "phase1_prune_metric_schur_mu",
    "phase1_prune_metric_schur_solve_mode",
    "phase1_prune_metric_schur_cost_weighting",
    "phase1_prune_trust_update_policy",
    "phase1_prune_metric_mu_update_policy",
    "phase1_prune_endpoint_overlap_policy",
}

RETIRED_PRUNE_RUNTIME_FIELDS = {
    "phase1_prune_stale_age",
    "phase1_prune_stagnation_threshold",
    "phase1_prune_small_theta_abs",
    "phase1_prune_small_theta_relative",
    "phase1_prune_amplitude_witness_required",
    "phase1_prune_collapse_peak_abs_min",
    "phase1_prune_collapse_current_abs_max",
    "phase1_prune_collapse_ratio",
    "phase1_prune_collapse_min_abs_drop",
    "phase1_prune_collapse_min_observations",
}


def _parser():
    return _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-7)


def _policy_payload() -> dict[str, object]:
    policy = DEFAULT_PHASE3_MATERIAL_WINDOW_POLICY
    return {
        "policy_version": policy.policy_version,
        "gram_entry_threshold": policy.gram_entry_threshold,
        "hessian_entry_threshold": policy.hessian_entry_threshold,
        "gram_omitted_l2_tolerance": policy.gram_omitted_l2_tolerance,
        "hessian_omitted_l2_tolerance": policy.hessian_omitted_l2_tolerance,
        "gram_cross_block_tolerance": policy.gram_cross_block_tolerance,
        "hessian_cross_block_tolerance": policy.hessian_cross_block_tolerance,
        "epsilon": policy.epsilon,
    }


def test_material_window_scope_is_explicit_and_normalized() -> None:
    scope = PHASE3_RESPONSE_COORDINATE_SCOPE_CANDIDATE_MATERIAL_COUPLING_WINDOW_V1

    assert scope in PHASE3_RESPONSE_COORDINATE_SCOPE_CHOICES
    assert normalize_phase3_response_coordinate_scope(scope) == scope


def test_test1_changes_only_phase3_geometry_scope_from_no_overlap_parent() -> None:
    parent = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract()
    )
    child = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1_contract()
    )

    assert canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256() == (
        "fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2"
    )
    changed = {
        key
        for key, value in child["execution_settings"].items()
        if parent["execution_settings"].get(key) != value
    }
    assert changed == {"phase3_response_coordinate_scope"}
    assert child["lineage_authority"] == {
        "parent_route_profile": (
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1
        ),
        "parent_contract_sha256": (
            canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1_contract_sha256()
        ),
        "only_intended_parent_setting_change": {
            "phase3_response_coordinate_scope": (
                PHASE3_RESPONSE_COORDINATE_SCOPE_CANDIDATE_MATERIAL_COUPLING_WINDOW_V1
            )
        },
        "scientific_result_anchor_claimed": False,
    }


def test_test1_binds_window_policy_without_changing_trust_or_refit() -> None:
    contract = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1_contract()
    )
    settings = contract["execution_settings"]
    invariants = contract["semantic_invariants"]

    assert invariants["phase3_material_window_policy"] == _policy_payload()
    assert invariants["phase3_material_window_independent_from_powell_refit_window"] is True
    assert invariants["phase3_material_window_closure_diagnostics"] == (
        "measured_retained_to_omitted_gram_and_hessian_blocks_v1"
    )
    assert invariants["phase3_material_window_measurement_accounting"] == (
        "strict_identity_deduplicated_per_round_union_across_all_"
        "evaluated_candidates_v1"
    )
    assert invariants[
        "phase3_material_window_per_candidate_summed_estimate_allowed"
    ] is False
    assert invariants["s_alg_component_order"] == [
        "N_H_outer",
        "N_H_refit",
        "N_grad",
        "N_metric",
    ]
    assert invariants["s_alg_aggregation"] == (
        "strict_identity_deduplicated_component_sum_v1"
    )
    assert settings["historical_singleton_coordinate_solve_policy"] == (
        "supported_metric_projected_generalized_trust_v1"
    )
    assert settings["historical_singleton_trust_region_update_policy"] == (
        "source_metric_inverse_sqrt_no_overlap_v1"
    )
    assert invariants["endpoint_overlap_measurement_active"] is False
    assert invariants["endpoint_overlap_query_charge_required"] == 0
    assert invariants["phase3_supported_whitening_active"] is False
    assert settings["adapt_accepted_refit_scope"] == "full_ansatz_v1"
    assert settings["adapt_accepted_refit_coordinate_chart"] == (
        "supported_fs_whitened_fixed_v1"
    )
    assert settings["phase1_prune_enabled"] is False
    assert settings["adapt_beam_live_branches"] == 1
    assert settings["adapt_beam_children_per_parent"] == 1


def test_test2_adds_only_live_prune_controls_and_keeps_admission_beam_1x1() -> None:
    parent = dict(
        CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1_EXECUTION_SETTINGS
    )
    child = dict(
        CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1_EXECUTION_SETTINGS
    )

    changed = {
        key
        for key, value in child.items()
        if key not in parent or parent[key] != value
    }
    assert changed == PRUNE_VERIFY_CHANGED_FIELDS
    assert child["adapt_beam_live_branches"] == 1
    assert child["adapt_beam_children_per_parent"] == 1
    assert child["adapt_beam_terminated_keep"] == 0
    assert child["adapt_beam_terminal_archive_mode"] == "disabled"


def test_test2_contract_requires_immutable_keep_prune_accounting() -> None:
    contract = (
        canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1_contract()
    )
    invariants = contract["semantic_invariants"]

    assert invariants["pruning_active"] is True
    assert invariants["prune_nomination_count_per_round_max"] == 1
    assert invariants["prune_source_geometry_policy"] == (
        "reuse_measured_source_active_gram_hessian_blocks_v1"
    )
    assert invariants["prune_verification_beam"] == (
        "minimal_immutable_keep_vs_one_delete_refit_sibling_v1"
    )
    assert invariants["prune_keep_branch_mutation_policy"] == (
        "immutable_never_destructively_mutated_v1"
    )
    assert invariants["prune_rollback_classical_query_charge"] == 0
    assert invariants["prune_branch_specific_delete_refit_measurements_are_real_work"] is True
    assert invariants["prune_rejected_branch_measurements_in_all_work_s_alg"] is True
    assert invariants["historical_admission_beam_active"] is False
    assert invariants["beam_shape"] == "effective_1x1_v1"


@pytest.mark.parametrize(
    ("alias", "resolved", "settings", "contract_factory", "digest_factory"),
    [
        (
            TEST1_ALIAS,
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1,
            CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1_EXECUTION_SETTINGS,
            canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1_contract,
            canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1_contract_sha256,
        ),
        (
            TEST2_ALIAS,
            SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1,
            CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1_EXECUTION_SETTINGS,
            canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1_contract,
            canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1_contract_sha256,
        ),
    ],
)
def test_material_window_profiles_cli_and_runtime_round_trip(
    alias: str,
    resolved: str,
    settings: dict[str, object],
    contract_factory,
    digest_factory,
) -> None:
    args = _parser().parse_args(
        ["--sr-route-profile", alias, "--adapt-max-depth", "50"]
    )
    contract = contract_factory()
    digest = digest_factory()

    assert args.sr_route_profile_resolved == resolved
    assert args.sr_route_profile_contract == contract
    assert args.sr_route_profile_contract_sha256 == digest
    assert args.phase3_response_coordinate_scope == (
        PHASE3_RESPONSE_COORDINATE_SCOPE_CANDIDATE_MATERIAL_COUPLING_WINDOW_V1
    )
    runtime = dict(settings)
    runtime.pop("phase_live_hysteresis_enabled", None)
    runtime["adapt_max_depth"] = 50
    assert validate_sr_route_profile_runtime_settings(
        profile_request=resolved,
        contract=contract,
        contract_sha256=digest,
        runtime_settings=runtime,
    ) == contract


@pytest.mark.parametrize(
    "required_field",
    sorted(
        key
        for key in (
            CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1_EXECUTION_SETTINGS
        )
        if key.startswith("phase1_prune_")
        and key not in RETIRED_PRUNE_RUNTIME_FIELDS
    )
    + ["phase3_shadow_damping_policy"],
)
def test_test2_runtime_source_lock_requires_every_detailed_prune_field(
    required_field: str,
) -> None:
    contract = (
        canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1_contract()
    )
    runtime = dict(
        CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1_EXECUTION_SETTINGS
    )
    runtime.pop("phase_live_hysteresis_enabled", None)
    runtime["adapt_max_depth"] = 50
    runtime.pop(required_field)

    with pytest.raises(ValueError, match="every detailed prune/source-lock"):
        validate_sr_route_profile_runtime_settings(
            profile_request=(
                SR_ROUTE_PROFILE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1
            ),
            contract=contract,
            contract_sha256=(
                canonical_sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_window_fs_prune_verify_v1_contract_sha256()
            ),
            runtime_settings=runtime,
        )


def test_test2_runtime_emits_every_required_prune_source_lock_field() -> None:
    source = textwrap.dedent(
        inspect.getsource(adapt_pipeline._run_hardcoded_adapt_vqe)
    )
    tree = ast.parse(source)
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "sr_runtime_settings_for_validation"
            for target in node.targets
        )
    )
    emitted_fields: set[str] = set()

    def _collect(value: ast.AST) -> None:
        if isinstance(value, ast.Dict):
            for key, child in zip(value.keys, value.values, strict=True):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    emitted_fields.add(key.value)
                elif key is None:
                    _collect(child)
        elif isinstance(value, ast.IfExp):
            _collect(value.body)
            _collect(value.orelse)

    _collect(assignment.value)
    expected = (
        CANONICAL_SR_SNAKE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_FS_PRUNE_VERIFY_V1_EXECUTION_SETTINGS
    )
    required = {
        key
        for key in expected
        if key.startswith("phase1_prune_")
        and key not in RETIRED_PRUNE_RUNTIME_FIELDS
    }.union({"phase3_shadow_damping_policy"})
    assert required.issubset(emitted_fields)
    assert RETIRED_PRUNE_RUNTIME_FIELDS.isdisjoint(emitted_fields)


def test_profiles_reject_wrong_prune_or_historical_beam_controls() -> None:
    with pytest.raises(SystemExit, match="2"):
        _parser().parse_args(
            [
                "--sr-route-profile",
                TEST1_ALIAS,
                "--adapt-max-depth",
                "50",
                "--phase1-prune-enabled",
            ]
        )
    with pytest.raises(SystemExit, match="2"):
        _parser().parse_args(
            [
                "--sr-route-profile",
                TEST2_ALIAS,
                "--adapt-max-depth",
                "50",
                "--adapt-beam-live-branches",
                "3",
                "--adapt-beam-children-per-parent",
                "2",
            ]
        )
