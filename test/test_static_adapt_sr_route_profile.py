from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.static_adapt.cli_config import (
    _build_adapt_arg_parser,
    _build_run_hardcoded_adapt_vqe_kwargs,
    _resolve_beam_capacity_policy,
)
from pipelines.static_adapt.resume_scaffold import (
    validate_resume_sr_route_profile_contract,
)
from pipelines.static_adapt.output_artifacts import build_output_payload
from pipelines.static_adapt.sr_snake_route_profile import (
    CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS,
    CANONICAL_SR_SNAKE_V2_EXECUTION_SETTINGS,
    CANONICAL_SR_SNAKE_V3_EXECUTION_SETTINGS,
    CANONICAL_SR_SNAKE_V3_1_EXECUTION_SETTINGS,
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
    SR_ROUTE_PROFILE_CONVENTIONAL_V3_1,
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
    canonical_sr_snake_v3_1_contract,
    canonical_sr_snake_v3_1_contract_sha256,
    canonical_sr_snake_v4_contract,
    canonical_sr_snake_v4_contract_sha256,
    canonical_sr_snake_no_prune_symmetric_cost_v1_contract,
    canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256,
    canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract,
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
    PHASE1_ENERGY_MODEL_LEGACY_LAMBDA_F_QUADRATIC_V1,
    PHASE2_CHEAP_CURVATURE_PROXY_POLICY_LEGACY_LAMBDA_F_RATIO_V1,
    PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF,
    PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1,
    PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1,
)
from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_holstein_hamiltonian,
)


def _parser():
    return _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-7)


_RETIRED_PRUNE_RUNTIME_FIELDS = frozenset(
    {
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
)


def _active_profile_execution_settings(
    settings: dict[str, object],
) -> dict[str, object]:
    return {
        field: expected
        for field, expected in settings.items()
        if field not in _RETIRED_PRUNE_RUNTIME_FIELDS
    }


def _canonical_args(*extra: str):
    return _parser().parse_args(
        ["--sr-route-profile", "sr_snake_v1", *extra]
    )


def _conventional_args(*extra: str):
    return _parser().parse_args(
        ["--sr-route-profile", "sr_snake", *extra]
    )


def _historical_v2_args(*extra: str):
    return _parser().parse_args(
        ["--sr-route-profile", "sr_snake_v2", *extra]
    )


def _candidate_v4_args(*extra: str):
    return _parser().parse_args(
        ["--sr-route-profile", "sr_snake_v4", *extra]
    )


def _no_prune_symmetric_cost_args(*extra: str):
    explicit_horizon = any(
        str(token) == "--adapt-max-depth"
        or str(token).startswith("--adapt-max-depth=")
        for token in extra
    )
    return _parser().parse_args(
        [
            "--sr-route-profile",
            "sr_snake_no_prune_symmetric_cost_v1",
            *(
                ()
                if explicit_horizon
                else ("--adapt-max-depth", "30")
            ),
            *extra,
        ]
    )


def _no_prune_symmetric_cost_beam_args(*extra: str):
    explicit_horizon = any(
        str(token) == "--adapt-max-depth"
        or str(token).startswith("--adapt-max-depth=")
        for token in extra
    )
    return _parser().parse_args(
        [
            "--sr-route-profile",
            "sr_snake_no_prune_symmetric_cost_beam_v1",
            *(("--adapt-max-depth", "30") if not explicit_horizon else ()),
            *extra,
        ]
    )


def _no_novelty_metric_prune_beam_args(*extra: str):
    return _parser().parse_args(
        [
            "--sr-route-profile",
            "sr_snake_no_novelty_metric_prune_beam_v1",
            *extra,
        ]
    )


def _h2o_derivative_resolved_v2_args(*extra: str):
    return _parser().parse_args(
        [
            "--sr-route-profile",
            "sr_snake_h2o_derivative_resolved_v2",
            *extra,
        ]
    )


def _h2o_derivative_resolved_paper_i_v3_args(*extra: str):
    return _parser().parse_args(
        [
            "--sr-route-profile",
            "sr_snake_h2o_derivative_resolved_paper_i_v3",
            *extra,
        ]
    )


def _runtime_kwargs(args) -> dict[str, object]:
    return _build_run_hardcoded_adapt_vqe_kwargs(
        args,
        h_poly=None,
        resolved_problem_context=SimpleNamespace(
            layout=SimpleNamespace(total_qubits=6)
        ),
        cli_adapt_continuation_mode="phase3_v1",
        adapt_ref_base_depth=0,
        psi_ref_override=None,
        psi_ref_source=None,
        psi_ref_handoff_state_kind=None,
        exact_gs_override=0.0,
        phase3_oracle_gradient_config=None,
        final_noise_audit_config=None,
    )


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


def _no_prune_symmetric_cost_beam_contract_payload(
    *, location: str = "settings"
) -> dict[str, object]:
    return {
        location: {
            "sr_route_profile_request": (
                SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1
            ),
            "sr_route_profile_contract": (
                canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract()
            ),
            "sr_route_profile_contract_sha256": (
                canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract_sha256()
            ),
            "phase3_response_coordinate_scope": (
                PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
            ),
            "phase1_score_mode": "trust_region_v1",
            "phase1_energy_model": PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
            "phase2_curvature_policy": (
                PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1
            ),
            "phase2_cheap_curvature_proxy_policy": (
                PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF
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


def test_sr_route_profile_default_is_off_and_does_not_rewrite_defaults() -> None:
    args = _parser().parse_args([])

    assert args.sr_route_profile_request == SR_ROUTE_PROFILE_REQUEST_OFF
    assert args.sr_route_profile_resolved is None
    assert args.sr_route_profile_contract is None
    assert args.sr_route_profile_contract_sha256 is None


@pytest.mark.parametrize(
    "profile_name",
    ["sr_snake_v1", SR_ROUTE_PROFILE_CANONICAL_V1],
)
def test_canonical_sr_route_profile_materializes_complete_contract(
    profile_name: str,
) -> None:
    args = _parser().parse_args(["--sr-route-profile", profile_name])

    assert args.sr_route_profile_request == SR_ROUTE_PROFILE_CANONICAL_V1
    assert args.sr_route_profile_resolved == SR_ROUTE_PROFILE_CANONICAL_V1
    assert args.sr_route_profile_contract == canonical_sr_snake_v1_contract()
    assert args.sr_route_profile_contract_sha256 == (
        canonical_sr_snake_v1_contract_sha256()
    )
    for field, expected in _active_profile_execution_settings(
        CANONICAL_SR_SNAKE_V1_EXECUTION_SETTINGS
    ).items():
        assert getattr(args, field) == expected, field
    assert all(not hasattr(args, field) for field in _RETIRED_PRUNE_RUNTIME_FIELDS)
    for field, expected in HISTORICAL_SR_SNAKE_V1_ACCEPTED_REFIT_SETTINGS.items():
        assert getattr(args, field) == expected, field


@pytest.mark.parametrize(
    "profile_name",
    ["sr_snake_v2", SR_ROUTE_PROFILE_CONVENTIONAL_V2],
)
def test_historical_v2_sr_route_profile_materializes_complete_contract(
    profile_name: str,
) -> None:
    args = _parser().parse_args(["--sr-route-profile", profile_name])

    assert args.sr_route_profile_request == SR_ROUTE_PROFILE_CONVENTIONAL_V2
    assert args.sr_route_profile_resolved == SR_ROUTE_PROFILE_CONVENTIONAL_V2
    assert args.sr_route_profile_contract == canonical_sr_snake_v2_contract()
    assert args.sr_route_profile_contract_sha256 == (
        canonical_sr_snake_v2_contract_sha256()
    )
    for field, expected in _active_profile_execution_settings(
        CANONICAL_SR_SNAKE_V2_EXECUTION_SETTINGS
    ).items():
        assert getattr(args, field) == expected, field
    for field, expected in HISTORICAL_SR_SNAKE_RESPONSE_SCOPE_SETTINGS.items():
        assert getattr(args, field) == expected, field


@pytest.mark.parametrize(
    "profile_name",
    ["sr_snake_v3", SR_ROUTE_PROFILE_CONVENTIONAL_V3],
)
def test_conventional_v3_sr_route_profile_materializes_complete_contract(
    profile_name: str,
) -> None:
    args = _parser().parse_args(["--sr-route-profile", profile_name])

    assert args.sr_route_profile_request == SR_ROUTE_PROFILE_CONVENTIONAL_V3
    assert args.sr_route_profile_resolved == SR_ROUTE_PROFILE_CONVENTIONAL_V3
    assert args.sr_route_profile_contract == canonical_sr_snake_v3_contract()
    assert args.sr_route_profile_contract_sha256 == (
        canonical_sr_snake_v3_contract_sha256()
    )
    for field, expected in _active_profile_execution_settings(
        CANONICAL_SR_SNAKE_V3_EXECUTION_SETTINGS
    ).items():
        assert getattr(args, field) == expected, field
    assert args.phase3_response_coordinate_scope == (
        PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
    )


@pytest.mark.parametrize(
    "profile_name",
    ["sr_snake", "sr_snake_v3_1", SR_ROUTE_PROFILE_CONVENTIONAL_V3_1],
)
def test_conventional_v3_1_materializes_hysteresis_disabled_contract(
    profile_name: str,
) -> None:
    args = _parser().parse_args(["--sr-route-profile", profile_name])

    assert args.sr_route_profile_request == SR_ROUTE_PROFILE_CONVENTIONAL_V3_1
    assert args.sr_route_profile_resolved == SR_ROUTE_PROFILE_CONVENTIONAL_V3_1
    assert args.sr_route_profile_contract == canonical_sr_snake_v3_1_contract()
    assert args.sr_route_profile_contract_sha256 == (
        canonical_sr_snake_v3_1_contract_sha256()
    )
    for field, expected in _active_profile_execution_settings(
        CANONICAL_SR_SNAKE_V3_1_EXECUTION_SETTINGS
    ).items():
        if field == "phase_live_hysteresis_enabled":
            continue
        assert getattr(args, field) == expected, field
    assert not hasattr(args, "phase_live_hysteresis_enabled")


@pytest.mark.parametrize(
    "profile_name",
    ["sr_snake_v4", SR_ROUTE_PROFILE_CANDIDATE_V4],
)
def test_candidate_v4_materializes_exact_combined_contract(
    profile_name: str,
) -> None:
    args = _parser().parse_args(["--sr-route-profile", profile_name])

    assert args.sr_route_profile_request == SR_ROUTE_PROFILE_CANDIDATE_V4
    assert args.sr_route_profile_resolved == SR_ROUTE_PROFILE_CANDIDATE_V4
    assert args.sr_route_profile_contract == canonical_sr_snake_v4_contract()
    assert args.sr_route_profile_contract_sha256 == (
        canonical_sr_snake_v4_contract_sha256()
    )
    for field, expected in _active_profile_execution_settings(
        CANONICAL_SR_SNAKE_V4_EXECUTION_SETTINGS
    ).items():
        if field == "phase_live_hysteresis_enabled":
            continue
        assert getattr(args, field) == expected, field

    expected_combined_settings = {
        "phase3_response_coordinate_scope": (
            PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
        ),
        "phase2_gram_novelty_policy": "fallback_only_v1",
        "phase3_gram_novelty_policy": "fallback_only_v1",
        "phase3_hardware_cost_normalization_mode": (
            "family_robust_symmetric_arctan_v1"
        ),
        "adapt_beam_live_branches": 1,
        "adapt_beam_children_per_parent": 1,
        "adapt_beam_terminated_keep": 0,
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
        "phase3_shadow_damping_policy": "mapped_seed_zero_query_v1",
        "adapt_finite_angle_fallback": False,
        "adapt_disable_hh_seed": True,
        "phase3_enable_rescue": False,
    }
    for field, expected in expected_combined_settings.items():
        assert getattr(args, field) == expected, field


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
        "phase1_phase2_lambda_f_proxy_active"
    ] is False
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
    args = _parser().parse_args(
        [
            "--sr-route-profile",
            profile_name,
            "--adapt-max-depth",
            "30",
        ]
    )

    assert args.sr_route_profile_request == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1
    )
    assert args.sr_route_profile_resolved == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1
    )
    assert args.sr_route_profile_contract == (
        canonical_sr_snake_no_prune_symmetric_cost_v1_contract()
    )
    assert args.sr_route_profile_contract_sha256 == (
        canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
    )
    for field, expected in (
        _active_profile_execution_settings(
            CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS
        ).items()
    ):
        if field == "phase_live_hysteresis_enabled":
            continue
        assert getattr(args, field) == expected, field
    assert not hasattr(args, "phase_live_hysteresis_enabled")

    assert "adapt_max_depth" not in (
        CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS
    )
    assert args.adapt_max_depth == 30
    assert args.problem == "hh"
    assert args.phase3_response_coordinate_scope == (
        PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
    )
    assert args.historical_singleton_coordinate_solve_scope == "phase3_only_v1"
    assert args.historical_singleton_trust_region_update_policy == (
        "displacement_calibrated_unbounded_v2"
    )
    assert args.adapt_accepted_refit_scope == "full_ansatz_v1"
    assert args.adapt_accepted_refit_coordinate_chart == (
        "supported_fs_whitened_fixed_v1"
    )
    assert args.adapt_accepted_refit_base_chart_policy == (
        "expanded_runtime_projected_logical_v1"
    )
    assert args.phase1_energy_model == PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1
    assert args.phase2_curvature_policy == (
        PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1
    )
    assert args.phase2_cheap_curvature_proxy_policy == (
        PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF
    )
    assert args.phase2_gram_novelty_policy == "fallback_only_v1"
    assert args.phase3_gram_novelty_policy == "fallback_only_v1"
    assert args.phase3_hardware_cost_normalization_mode == (
        "family_robust_symmetric_arctan_v1"
    )
    assert args.adapt_beam_live_branches == 1
    assert args.adapt_beam_children_per_parent == 1
    assert args.phase1_prune_enabled is False
    assert args.adapt_full_refit_every == 0
    assert args.adapt_final_full_refit == "false"
    assert args.adapt_finite_angle_fallback is False
    assert args.phase3_shadow_damping_policy == "off"
    assert args.sr_escape_mode == "disabled"


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


@pytest.mark.parametrize(
    "profile_name",
    [
        "sr_snake_no_prune_symmetric_cost_beam_v1",
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1,
    ],
)
def test_no_prune_symmetric_cost_beam_profile_materializes_historical_3x2(
    profile_name: str,
) -> None:
    args = _parser().parse_args(
        [
            "--sr-route-profile",
            profile_name,
            "--adapt-max-depth",
            "50",
        ]
    )

    assert args.sr_route_profile_request == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1
    )
    assert args.sr_route_profile_resolved == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1
    )
    assert args.sr_route_profile_contract == (
        canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract()
    )
    assert args.sr_route_profile_contract_sha256 == (
        canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract_sha256()
    )
    assert args.adapt_max_depth == 50
    assert args.adapt_beam_live_branches == 3
    assert args.adapt_beam_children_per_parent == 2
    assert args.adapt_beam_terminated_keep == 3
    assert args.adapt_beam_terminal_archive_mode == "legacy"
    assert args.adapt_beam_lambda == pytest.approx(0.005)

    beam = _resolve_beam_capacity_policy(
        adapt_beam_live_branches=args.adapt_beam_live_branches,
        adapt_beam_children_per_parent=args.adapt_beam_children_per_parent,
        adapt_beam_terminated_keep=args.adapt_beam_terminated_keep,
        adapt_beam_terminal_archive_mode=args.adapt_beam_terminal_archive_mode,
    )
    assert beam.beam_enabled is True
    assert beam.live_branches_effective == 3
    assert beam.children_per_parent_effective == 2
    assert beam.terminated_keep_effective == 3
    assert beam.source_terminated_keep == "legacy_explicit"


def test_appendix_beam_profile_changes_only_historical_beam_fields() -> None:
    main = CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_V1_EXECUTION_SETTINGS
    appendix = (
        CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1_EXECUTION_SETTINGS
    )
    changed = {
        field
        for field in main
        if main[field] != appendix[field]
    }

    assert changed == {
        "adapt_beam_live_branches",
        "adapt_beam_children_per_parent",
        "adapt_beam_terminated_keep",
        "adapt_beam_terminal_archive_mode",
    }
    assert main["adapt_beam_live_branches"] == 1
    assert main["adapt_beam_children_per_parent"] == 1
    assert main["adapt_beam_terminated_keep"] == 0
    assert main["adapt_beam_terminal_archive_mode"] == "disabled"
    assert main["adapt_beam_lambda"] == pytest.approx(0.005)
    assert appendix["adapt_beam_lambda"] == pytest.approx(0.005)

    contract = canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract()
    assert canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract_sha256() == (
        "49fb8c2f069722ce87cbaaedc8d7d32726a11dad92a624e3326269d75dcd1168"
    )
    assert contract["semantic_invariants"]["appendix_one_factor_ablation"] is True
    assert contract["semantic_invariants"]["beam_expanded_child_cap_per_round"] == 6
    assert (
        contract["semantic_invariants"][
            "beam_parent_stop_terminal_also_materialized"
        ]
        is True
    )
    assert contract["semantic_invariants"]["beam_structural_mode"] == (
        "stop_or_single_admission"
    )
    assert contract["semantic_invariants"]["beam_terminal_archive_accumulation"] == (
        "cumulative_across_rounds"
    )
    assert contract["semantic_invariants"]["beam_terminal_archive_cap"] == 3
    assert contract["lineage_authority"]["parent_route_profile"] == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1
    )
    assert contract["lineage_authority"]["parent_contract_sha256"] == (
        canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
    )
    assert contract["lineage_authority"]["historical_semantics_source_commit"] == (
        "1f1d93c1a0060f0db70da6736cae4ec5ffffc79b^"
    )


@pytest.mark.parametrize(
    "override",
    [
        ("--adapt-beam-live-branches", "2"),
        ("--adapt-beam-children-per-parent", "3"),
        ("--adapt-beam-terminated-keep", "0"),
        ("--adapt-beam-terminal-archive-mode", "disabled"),
        ("--adapt-beam-lambda", "0"),
        ("--phase1-prune-enabled",),
    ],
)
def test_no_prune_symmetric_cost_beam_profile_rejects_drift(
    override: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        _no_prune_symmetric_cost_beam_args(*override)


def test_no_prune_symmetric_cost_beam_profile_requires_explicit_horizon() -> None:
    with pytest.raises(SystemExit, match="2"):
        _parser().parse_args(
            [
                "--sr-route-profile",
                "sr_snake_no_prune_symmetric_cost_beam_v1",
            ]
        )


@pytest.mark.parametrize("depth", [30, 50])
def test_no_prune_symmetric_cost_profile_leaves_horizon_source_locked(
    depth: int,
) -> None:
    args = _no_prune_symmetric_cost_args("--adapt-max-depth", str(depth))

    assert args.adapt_max_depth == depth
    assert "adapt_max_depth" not in args.sr_route_profile_contract[
        "execution_settings"
    ]
    assert args.sr_route_profile_contract["semantic_invariants"][
        "controller_horizon_source"
    ] == "per_regime_source_lock"


def test_no_prune_symmetric_cost_profile_requires_explicit_horizon() -> None:
    with pytest.raises(SystemExit, match="2"):
        _parser().parse_args(
            [
                "--sr-route-profile",
                "sr_snake_no_prune_symmetric_cost_v1",
            ]
        )


@pytest.mark.parametrize("depth", ["0", "-1"])
def test_no_prune_symmetric_cost_profile_rejects_nonpositive_horizon(
    depth: str,
) -> None:
    with pytest.raises(SystemExit, match="2"):
        _parser().parse_args(
            [
                "--sr-route-profile",
                "sr_snake_no_prune_symmetric_cost_v1",
                "--adapt-max-depth",
                depth,
            ]
        )


def test_no_prune_symmetric_cost_accepts_explicit_matching_safety_switches() -> None:
    args = _no_prune_symmetric_cost_args(
        "--phase1-no-prune",
        "--adapt-no-finite-angle-fallback",
        "--phase3-no-rescue",
    )

    assert args.phase1_prune_enabled is False
    assert args.adapt_finite_angle_fallback is False
    assert args.phase3_enable_rescue is False


@pytest.mark.parametrize(
    "override",
    [
        ("--phase1-prune-enabled",),
        ("--adapt-full-refit-every", "8"),
        ("--adapt-final-full-refit", "true"),
        ("--phase2-gram-novelty-policy", "ordinary_multiplier_v1"),
        ("--phase3-gram-novelty-policy", "ordinary_multiplier_v1"),
        (
            "--phase3-hardware-cost-normalization-mode",
            "family_robust_v1",
        ),
        (
            "--historical-singleton-coordinate-solve-scope",
            "phase2_and_phase3_v1",
        ),
        ("--phase3-shadow-damping-policy", "mapped_seed_zero_query_v1"),
        ("--sr-escape-mode", "saddle_only"),
    ],
)
def test_no_prune_symmetric_cost_profile_rejects_scientific_drift(
    override: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        _no_prune_symmetric_cost_args(*override)


@pytest.mark.parametrize("option", ["--phase1-lambda-F", "--phase2-lambda-F"])
def test_no_prune_symmetric_cost_rejects_lambda_f_proxy_controls(
    option: str,
) -> None:
    with pytest.raises(ValueError, match="lambda_f_curvature_proxies"):
        normalize_sr_route_profile_namespace(
            _route_profile_namespace(
                SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_V1,
                explicit_options=(option,),
            )
        )


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
            "phase3_shadow_damping_policy",
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


def test_no_novelty_metric_prune_beam_profile_materializes_exact_contract() -> None:
    args = _no_novelty_metric_prune_beam_args()

    assert args.sr_route_profile_request == (
        SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1
    )
    assert args.sr_route_profile_resolved == (
        SR_ROUTE_PROFILE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1
    )
    assert args.sr_route_profile_contract == (
        canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract()
    )
    assert args.sr_route_profile_contract_sha256 == (
        canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract_sha256()
    )
    for field, expected in (
        _active_profile_execution_settings(
            CANONICAL_SR_SNAKE_NO_NOVELTY_METRIC_PRUNE_BEAM_V1_EXECUTION_SETTINGS
        ).items()
    ):
        assert getattr(args, field) == expected, field

    assert args.phase3_novelty_ablation_mode == "all"
    assert args.phase2_enable_batching is False
    assert args.phase3_enable_batching is False
    assert args.phase3_runtime_split_subset_sizes == "1"
    assert args.phase3_runtime_split_child_padding_policy == (
        "full_binary_code_space_v1"
    )
    assert args.phase3_backend_cost_mode == "proxy"
    assert args.adapt_beam_live_branches == 3
    assert args.adapt_beam_children_per_parent == 2
    assert args.phase1_prune_enabled is True
    assert args.phase1_prune_schur_nomination_route == "metric_regularized_v1"
    assert args.phase1_prune_metric_schur_mu == pytest.approx(0.01)
    assert args.phase3_response_coordinate_scope == (
        PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
    )


def test_h2o_derivative_resolved_profile_changes_only_pool_identity() -> None:
    args = _h2o_derivative_resolved_v2_args()
    contract = canonical_sr_snake_h2o_derivative_resolved_v2_contract()
    parent = canonical_sr_snake_no_novelty_metric_prune_beam_v1_contract()

    assert args.sr_route_profile_request == (
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2
    )
    assert args.sr_route_profile_resolved == (
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_V2
    )
    assert args.sr_route_profile_contract == contract
    assert args.sr_route_profile_contract_sha256 == (
        canonical_sr_snake_h2o_derivative_resolved_v2_contract_sha256()
    )
    assert args.adapt_pool == "full_meta_derivative_resolved_v2"
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
    args = _h2o_derivative_resolved_paper_i_v3_args()
    contract = canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract()
    parent = canonical_sr_snake_no_prune_symmetric_cost_v1_contract()

    assert args.sr_route_profile_request == (
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3
    )
    assert args.sr_route_profile_resolved == (
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3
    )
    assert args.sr_route_profile_contract == contract
    assert args.sr_route_profile_contract_sha256 == (
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

    assert args.problem == "molecular_vibronic_h2o_linear_fd"
    assert args.adapt_max_depth == 50
    assert args.adapt_pool == "full_meta_derivative_resolved_v2"
    assert args.adapt_beam_live_branches == 1
    assert args.adapt_beam_children_per_parent == 1
    assert args.adapt_beam_terminated_keep == 0
    assert args.adapt_beam_terminal_archive_mode == "disabled"
    assert args.phase1_prune_enabled is False
    assert contract["semantic_invariants"]["pruning_active"] is False
    assert contract["semantic_invariants"]["terminal_prune_active"] is False
    assert args.phase2_enable_batching is False
    assert args.phase3_enable_batching is False
    assert args.phase2_gram_novelty_policy == "fallback_only_v1"
    assert args.phase3_gram_novelty_policy == "fallback_only_v1"
    assert args.phase3_novelty_ablation_mode == "off"
    assert args.phase3_response_coordinate_scope == (
        PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
    )
    assert args.historical_singleton_coordinate_solve_policy == (
        "supported_metric_whitened_eigh_v1"
    )
    assert args.adapt_accepted_refit_scope == "full_ansatz_v1"
    assert args.adapt_full_refit_every == 0
    assert args.adapt_final_full_refit == "false"


@pytest.mark.parametrize(
    "override",
    [
        ("--phase3-novelty-ablation-mode", "off"),
        ("--adapt-beam-live-branches", "1"),
        ("--adapt-beam-children-per-parent", "1"),
        ("--phase1-no-prune",),
        ("--phase1-prune-schur-nomination-route", "hessian_coupling_v1"),
        ("--phase1-prune-metric-schur-mu", "1e-6"),
        ("--phase3-runtime-split-subset-sizes", "1,2"),
    ],
)
def test_no_novelty_metric_prune_beam_profile_rejects_scientific_drift(
    override: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        _no_novelty_metric_prune_beam_args(*override)


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
    assert namespace.phase1_energy_model == (
        PHASE1_ENERGY_MODEL_LEGACY_LAMBDA_F_QUADRATIC_V1
    )
    assert namespace.phase2_curvature_policy == (
        PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1
    )
    assert namespace.phase2_cheap_curvature_proxy_policy == (
        PHASE2_CHEAP_CURVATURE_PROXY_POLICY_LEGACY_LAMBDA_F_RATIO_V1
    )
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


@pytest.mark.parametrize("option", ["--phase1-lambda-F", "--phase2-lambda-F"])
def test_candidate_v4_rejects_explicit_lambda_f_proxy_controls(option: str) -> None:
    with pytest.raises(ValueError, match="lambda_f_curvature_proxies"):
        normalize_sr_route_profile_namespace(
            _route_profile_namespace(
                SR_ROUTE_PROFILE_CANDIDATE_V4,
                explicit_options=(option,),
            )
        )


@pytest.mark.parametrize(
    ("option", "field", "legacy_value"),
    [
        (
            "--phase1-score-mode",
            "phase1_score_mode",
            "legacy_simple_v1",
        ),
        (
            "--phase1-energy-model",
            "phase1_energy_model",
            PHASE1_ENERGY_MODEL_LEGACY_LAMBDA_F_QUADRATIC_V1,
        ),
        (
            "--phase2-curvature-policy",
            "phase2_curvature_policy",
            PHASE2_CURVATURE_POLICY_LEGACY_OPTIONAL_V1,
        ),
        (
            "--phase2-cheap-curvature-proxy-policy",
            "phase2_cheap_curvature_proxy_policy",
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY_LEGACY_LAMBDA_F_RATIO_V1,
        ),
    ],
)
def test_candidate_v4_rejects_explicit_legacy_phase12_policy(
    option: str,
    field: str,
    legacy_value: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        normalize_sr_route_profile_namespace(
            _route_profile_namespace(
                SR_ROUTE_PROFILE_CANDIDATE_V4,
                explicit_options=(option,),
                overrides={field: legacy_value},
            )
        )


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

    runtime_settings["phase2_cheap_curvature_proxy_policy"] = (
        PHASE2_CHEAP_CURVATURE_PROXY_POLICY_LEGACY_LAMBDA_F_RATIO_V1
    )
    with pytest.raises(ValueError, match="effective runtime settings drifted"):
        validate_sr_route_profile_runtime_settings(
            profile_request=SR_ROUTE_PROFILE_CANDIDATE_V4,
            contract=canonical_sr_snake_v4_contract(),
            contract_sha256=canonical_sr_snake_v4_contract_sha256(),
            runtime_settings=runtime_settings,
        )


def test_unqualified_sr_snake_alias_advances_to_v3_1_not_candidate_v4() -> None:
    args = _conventional_args()

    assert args.sr_route_profile_request == SR_ROUTE_PROFILE_CONVENTIONAL_V3_1
    assert args.sr_route_profile_resolved == SR_ROUTE_PROFILE_CONVENTIONAL_V3_1
    assert args.sr_route_profile_contract == canonical_sr_snake_v3_1_contract()
    assert not hasattr(args, "phase_live_hysteresis_enabled")
    assert args.sr_route_profile_contract != canonical_sr_snake_v4_contract()


def test_candidate_v4_disables_finite_angle_fallback_in_contract_and_cli() -> None:
    implicit_args = _candidate_v4_args()
    explicit_args = _candidate_v4_args("--adapt-no-finite-angle-fallback")

    for args in (implicit_args, explicit_args):
        assert args.adapt_finite_angle_fallback is False
        assert args.phase3_enable_rescue is False
        assert args.sr_route_profile_contract["execution_settings"][
            "adapt_finite_angle_fallback"
        ] is False
        assert args.sr_route_profile_contract["execution_settings"][
            "phase3_enable_rescue"
        ] is False
        assert args.sr_route_profile_contract["semantic_invariants"][
            "finite_angle_fallback_active"
        ] is False

    assert "--adapt-no-finite-angle-fallback" in (
        explicit_args._explicit_cli_options
    )


def test_candidate_v4_rejects_enabled_finite_angle_fallback() -> None:
    with pytest.raises(SystemExit, match="2"):
        _candidate_v4_args("--adapt-finite-angle-fallback")


@pytest.mark.parametrize(
    "profile_name",
    ["sr_snake_v1", "sr_snake_v2", "sr_snake_v3"],
)
def test_pre_v4_profiles_still_reject_disabled_finite_angle_fallback(
    profile_name: str,
) -> None:
    with pytest.raises(SystemExit, match="2"):
        _parser().parse_args(
            [
                "--sr-route-profile",
                profile_name,
                "--adapt-no-finite-angle-fallback",
            ]
        )


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
    args = _parser().parse_args(["--sr-route-profile", profile_name])

    assert args.phase3_hardware_cost_normalization_mode == expected_policy


def test_versioned_sr_route_profiles_are_distinct_and_v1_digest_is_stable() -> None:
    assert SR_ROUTE_PROFILE_CANONICAL_V1 != SR_ROUTE_PROFILE_CONVENTIONAL_V2
    assert SR_ROUTE_PROFILE_CONVENTIONAL_V2 != SR_ROUTE_PROFILE_CONVENTIONAL_V3
    assert SR_ROUTE_PROFILE_CONVENTIONAL_V3 != SR_ROUTE_PROFILE_CONVENTIONAL_V3_1
    assert canonical_sr_snake_v1_contract_sha256() == (
        "fab7b5a6c4bd2ab019139367aa2a507356a5c969b6b88cd72d32365ae766e13e"
    )
    assert canonical_sr_snake_v2_contract_sha256() == (
        "32d2bdf2b05818be6f4add74137447a313605d7ed35ffb880651863b793a0f64"
    )
    assert canonical_sr_snake_v3_contract_sha256() == (
        "435910592e88f0136a0d45f611f79fe96b21d75fd25bad58276c871f39dc080e"
    )
    assert canonical_sr_snake_v3_1_contract_sha256() == (
        "9b96179935ed80967a3335dfbbf8eece86a04c2d412e6b92aa8a466fa6913542"
    )
    assert canonical_sr_snake_v4_contract_sha256() == (
            "d705671019543f676ee23ea9e1de8a7658183e3926939f2c389b8f015db6fe2f"
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
        "49fb8c2f069722ce87cbaaedc8d7d32726a11dad92a624e3326269d75dcd1168"
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
    assert canonical_sr_snake_v3_1_contract()["execution_settings"] == (
        CANONICAL_SR_SNAKE_V3_1_EXECUTION_SETTINGS
    )
    assert canonical_sr_snake_v3_1_contract()["lineage_authority"][
        "parent_contract_sha256"
    ] == canonical_sr_snake_v3_contract_sha256()
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


def test_canonical_sr_route_profile_accepts_explicit_matching_setting() -> None:
    args = _canonical_args("--adapt-maxiter", "200")

    assert args.adapt_maxiter == 200
    assert args.sr_route_profile_contract_sha256 == (
        canonical_sr_snake_v1_contract_sha256()
    )


def test_candidate_v4_accepts_explicit_matching_disable_hh_seed() -> None:
    args = _candidate_v4_args("--adapt-disable-hh-seed")

    assert args.adapt_disable_hh_seed is True
    assert args.sr_route_profile_contract_sha256 == (
        canonical_sr_snake_v4_contract_sha256()
    )


@pytest.mark.parametrize(
    "args_factory",
    [_canonical_args, _historical_v2_args, _conventional_args],
)
def test_historical_profiles_still_reject_explicit_disable_hh_seed(
    args_factory,
) -> None:
    with pytest.raises(SystemExit, match="2"):
        args_factory("--adapt-disable-hh-seed")


@pytest.mark.parametrize(
    "override",
    [
        ("--adapt-maxiter", "300"),
        (
            "--historical-singleton-coordinate-solve-scope",
            "phase2_and_phase3_v1",
        ),
        (
            "--historical-singleton-trust-region-update-policy",
            "fixed",
        ),
        ("--sr-powell-coordinate-chart-policy", "logical_shared_reduced_v1"),
        ("--phase0-pilot-enabled",),
        ("--phase2-enable-batching",),
        ("--phase3-enable-batching",),
        ("--adapt-no-repeats",),
        ("--phase1-no-prune",),
        ("--sr-escape-mode", "saddle_only"),
        ("--phase3-novelty-ablation-mode", "fallback_only_v1"),
    ],
)
def test_canonical_sr_route_profile_rejects_explicit_scientific_drift(
    override: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        _canonical_args(*override)


@pytest.mark.parametrize(
    "override",
    [
        ("--adapt-accepted-refit-scope", "selector_policy_v1"),
        ("--adapt-accepted-refit-coordinate-chart", "native_v1"),
        (
            "--adapt-accepted-refit-base-chart-policy",
            "logical_shared_reduced_v1",
        ),
        (
            "--historical-singleton-coordinate-solve-scope",
            "phase2_and_phase3_v1",
        ),
        ("--phase3-novelty-ablation-mode", "fallback_only_v1"),
        ("--sr-escape-mode", "saddle_only"),
        (
            "--phase1-prune-schur-nomination-route",
            "metric_regularized_v1",
        ),
    ],
)
def test_conventional_sr_route_profile_rejects_explicit_scientific_drift(
    override: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        _conventional_args(*override)


def test_conventional_v3_rejects_legacy_phase3_response_scope() -> None:
    with pytest.raises(SystemExit, match="2"):
        _conventional_args(
            "--phase3-response-coordinate-scope",
            PHASE3_RESPONSE_COORDINATE_SCOPE_LEGACY_REOPT_COUPLED_V1,
        )


@pytest.mark.parametrize(
    "override",
    [
        (
            "--phase3-response-coordinate-scope",
            PHASE3_RESPONSE_COORDINATE_SCOPE_LEGACY_REOPT_COUPLED_V1,
        ),
        ("--phase2-gram-novelty-policy", "ordinary_multiplier_v1"),
        ("--phase3-gram-novelty-policy", "ordinary_multiplier_v1"),
        (
            "--phase3-hardware-cost-normalization-mode",
            "family_robust_v1",
        ),
        ("--adapt-beam-live-branches", "2"),
        ("--adapt-beam-children-per-parent", "2"),
        ("--adapt-beam-terminated-keep", "1"),
        ("--phase1-no-prune",),
        ("--phase1-prune-mode", "both"),
        ("--phase1-prune-local-window-size", "4"),
        (
            "--phase1-prune-schur-nomination-route",
            "hessian_coupling_v1",
        ),
        (
            "--phase1-prune-metric-schur-solve-mode",
            "stationary_gw_zero_v1",
        ),
        ("--phase1-prune-trust-update-policy", "off"),
        ("--phase1-prune-metric-mu-update-policy", "off"),
        (
            "--phase1-prune-endpoint-overlap-policy",
            "energy_safe_trial_only_v1",
        ),
        ("--phase3-shadow-damping-policy", "off"),
        ("--adapt-final-full-refit", "true"),
    ],
)
def test_candidate_v4_rejects_conflicting_scientific_overrides(
    override: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        _candidate_v4_args(*override)


def test_canonical_sr_route_profile_round_trips_into_runtime_kwargs() -> None:
    args = _canonical_args()
    kwargs = _runtime_kwargs(args)

    assert kwargs["sr_route_profile_request"] == SR_ROUTE_PROFILE_CANONICAL_V1
    assert kwargs["sr_route_profile_resolved"] == SR_ROUTE_PROFILE_CANONICAL_V1
    assert kwargs["sr_route_profile_contract"] == canonical_sr_snake_v1_contract()
    assert kwargs["sr_route_profile_contract_sha256"] == (
        canonical_sr_snake_v1_contract_sha256()
    )


def test_conventional_sr_route_profile_round_trips_into_runtime_kwargs() -> None:
    args = _conventional_args()
    kwargs = _runtime_kwargs(args)

    assert kwargs["sr_route_profile_request"] == SR_ROUTE_PROFILE_CONVENTIONAL_V3_1
    assert kwargs["sr_route_profile_resolved"] == SR_ROUTE_PROFILE_CONVENTIONAL_V3_1
    assert kwargs["sr_route_profile_contract"] == canonical_sr_snake_v3_1_contract()
    assert kwargs["sr_route_profile_contract_sha256"] == (
        canonical_sr_snake_v3_1_contract_sha256()
    )
    assert "phase_live_hysteresis_enabled" not in kwargs
    assert kwargs["phase3_response_coordinate_scope"] == (
        PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
    )
    assert kwargs["adapt_accepted_refit_scope"] == "full_ansatz_v1"
    assert kwargs["adapt_accepted_refit_coordinate_chart"] == (
        "supported_fs_whitened_fixed_v1"
    )
    assert kwargs["adapt_accepted_refit_base_chart_policy"] == (
        "expanded_runtime_projected_logical_v1"
    )


def test_appendix_beam_profile_round_trips_into_runtime_kwargs() -> None:
    args = _no_prune_symmetric_cost_beam_args("--adapt-max-depth", "50")
    kwargs = _runtime_kwargs(args)

    assert kwargs["sr_route_profile_request"] == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1
    )
    assert kwargs["sr_route_profile_resolved"] == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1
    )
    assert kwargs["sr_route_profile_contract"] == (
        canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract()
    )
    assert kwargs["sr_route_profile_contract_sha256"] == (
        canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract_sha256()
    )
    assert kwargs["adapt_beam_live_branches"] == 3
    assert kwargs["adapt_beam_children_per_parent"] == 2
    assert kwargs["adapt_beam_terminated_keep"] == 3
    assert kwargs["adapt_beam_terminal_archive_mode"] == "legacy"
    assert kwargs["adapt_beam_lambda"] == pytest.approx(0.005)
    assert kwargs["phase1_prune_enabled"] is False
    assert kwargs["phase2_enable_batching"] is False


def test_candidate_v4_round_trips_new_cli_fields_into_runtime_kwargs() -> None:
    args = _candidate_v4_args()
    kwargs = _runtime_kwargs(args)

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
        "adapt_beam_live_branches": 1,
        "adapt_beam_children_per_parent": 1,
        "adapt_beam_terminated_keep": 0,
        "phase1_prune_mode": "live",
        "phase1_prune_local_window_size": 0,
        "phase1_prune_schur_nomination_route": (
            "full_logical_fs_trust_delete_refit_v1"
        ),
        "phase1_prune_metric_schur_solve_mode": (
            "affine_deletion_global_trust_v1"
        ),
        "phase1_prune_trust_update_policy": (
            "modeled_local_fs_conservative_v1"
        ),
        "phase1_prune_metric_mu_update_policy": (
            "same_trial_underprediction_monotone_v1"
        ),
        "phase1_prune_endpoint_overlap_policy": "off",
        "phase3_shadow_damping_policy": "mapped_seed_zero_query_v1",
        "finite_angle_fallback": False,
        "phase3_enable_rescue": False,
    }
    for field, expected in expected_runtime_fields.items():
        assert kwargs[field] == expected, field


def test_candidate_v4_identity_is_readable_but_not_execution_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The historical v4 identity remains readable but grants no execution."""

    kwargs = _runtime_kwargs(
        _candidate_v4_args(
            "--problem",
            "hh",
            "--L",
            "2",
            "--u",
            "0.25",
            "--g-ep",
            "0.353553390593",
            "--n-ph-max",
            "2",
        )
    )
    kwargs["resolved_problem_context"] = None
    kwargs["h_poly"] = build_hubbard_holstein_hamiltonian(
        dims=2,
        J=1.0,
        U=0.25,
        omega0=1.0,
        g=0.353553390593,
        n_ph_max=2,
        boson_encoding="binary",
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
        include_zero_point=True,
    )
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_a, **_k: None)

    assert kwargs["sr_route_profile_resolved"] == (
        SR_ROUTE_PROFILE_CANDIDATE_V4
    )
    assert kwargs["sr_route_profile_contract"] == (
        canonical_sr_snake_v4_contract()
    )
    assert kwargs["sr_route_profile_contract_sha256"] == (
        canonical_sr_snake_v4_contract_sha256()
    )
    with pytest.raises(
        ValueError,
        match="historical_paper_i_route_compatibility_inactive",
    ):
        adapt_pipeline._run_hardcoded_adapt_vqe(**kwargs)


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "appendix_beam"])
def test_registered_sr_route_profile_serializes_into_result_settings(
    version: str,
) -> None:
    phase12_policies: dict[str, str] = {}
    if version == "appendix_beam":
        args = _no_prune_symmetric_cost_beam_args(
            "--adapt-max-depth",
            "50",
            "--adapt-resume-compile-smoke",
            "off",
        )
        profile = SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1
        contract = canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract()
        digest = canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract_sha256()
        response_scope = (
            PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
        )
        phase12_policies = {
            "phase1_score_mode": "trust_region_v1",
            "phase1_energy_model": PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
            "phase2_curvature_policy": (
                PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1
            ),
            "phase2_cheap_curvature_proxy_policy": (
                PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF
            ),
        }
    elif version == "v3":
        args = _parser().parse_args(
            [
                "--sr-route-profile",
                "sr_snake_v3",
                "--adapt-resume-compile-smoke",
                "off",
            ]
        )
        profile = SR_ROUTE_PROFILE_CONVENTIONAL_V3
        contract = canonical_sr_snake_v3_contract()
        digest = canonical_sr_snake_v3_contract_sha256()
        response_scope = (
            PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
        )
    elif version == "v2":
        args = _historical_v2_args("--adapt-resume-compile-smoke", "off")
        profile = SR_ROUTE_PROFILE_CONVENTIONAL_V2
        contract = canonical_sr_snake_v2_contract()
        digest = canonical_sr_snake_v2_contract_sha256()
        response_scope = PHASE3_RESPONSE_COORDINATE_SCOPE_LEGACY_REOPT_COUPLED_V1
    else:
        args = _canonical_args("--adapt-resume-compile-smoke", "off")
        profile = SR_ROUTE_PROFILE_CANONICAL_V1
        contract = canonical_sr_snake_v1_contract()
        digest = canonical_sr_snake_v1_contract_sha256()
        response_scope = PHASE3_RESPONSE_COORDINATE_SCOPE_LEGACY_REOPT_COUPLED_V1
    chart = "expanded_runtime_projected_logical_v1"
    adapt_payload = {
        "success": True,
        "energy": -1.0,
        "exact_gs_energy": -1.0,
        "abs_delta_e": 0.0,
        "operators": [],
        "sr_route_profile_request": profile,
        "sr_route_profile_resolved": profile,
        "sr_route_profile_contract": contract,
        "sr_route_profile_contract_sha256": digest,
        "phase3_response_coordinate_scope": response_scope,
        **phase12_policies,
        "static_route_identity": {
            "route_family": "singleton_response_snake",
            "route_profile": profile,
            "powell_coordinate_chart_policy": chart,
            "sr_route_profile_request": profile,
            "sr_route_profile_contract": contract,
            "sr_route_profile_contract_sha256": digest,
            "phase3_response_coordinate_scope": response_scope,
            **phase12_policies,
        },
        "optimizer_coordinate_chart": {
            "powell_coordinate_chart_policy": chart,
        },
    }
    psi = np.array([1.0 + 0.0j, 0.0 + 0.0j])

    payload = build_output_payload(
        args=args,
        cli_adapt_continuation_mode="phase3_v1",
        adapt_payload=adapt_payload,
        ordered_labels_exyz=["e"],
        coeff_map_exyz={"e": 0.0},
        hmat=np.zeros((2, 2), dtype=complex),
        gs_energy_exact=-1.0,
        gs_energy_source="unit",
        psi0=psi,
        ansatz_input_state_for_adapt=psi,
        ansatz_input_state_source="hf",
        ansatz_input_state_kind="reference_state",
        trajectory=[],
        adapt_ref_import=None,
        dense_eigh_enabled=True,
        hilbert_dim=2,
        adapt_ref_base_depth=0,
        initial_state_source_resolved="hf",
        initial_state_kind_resolved="reference_state",
    )

    assert payload["settings"]["sr_route_profile_request"] == profile
    assert payload["settings"]["sr_route_profile_contract"] == contract
    assert payload["settings"]["sr_route_profile_contract_sha256"] == digest
    assert payload["settings"]["phase3_response_coordinate_scope"] == (
        response_scope
    )
    if version == "appendix_beam":
        serialized = payload["settings"]["sr_route_profile_contract"][
            "execution_settings"
        ]
        assert serialized["adapt_beam_live_branches"] == 3
        assert serialized["adapt_beam_children_per_parent"] == 2
        assert serialized["adapt_beam_terminated_keep"] == 3
        assert serialized["adapt_beam_terminal_archive_mode"] == "legacy"
        assert serialized["adapt_beam_lambda"] == pytest.approx(0.005)


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


@pytest.mark.parametrize("location", ["settings", "checkpoint"])
def test_resume_contract_preserves_appendix_beam_identity(
    location: str,
) -> None:
    contract = canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract()
    digest = canonical_sr_snake_no_prune_symmetric_cost_beam_v1_contract_sha256()
    validation = validate_resume_sr_route_profile_contract(
        _no_prune_symmetric_cost_beam_contract_payload(location=location),
        expected_profile_request=(
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1
        ),
        expected_contract=contract,
        expected_contract_sha256=digest,
    )

    assert validation["status"] == "pass"
    assert validation["artifact_profile_request"] == (
        SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_BEAM_V1
    )
    assert validation["contract_sha256"] == digest


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
