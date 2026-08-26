from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.scaffold.hh_continuation_stage_control import StageControllerConfig
from pipelines.static_adapt import sr_snake_route_profile as route_profiles
from pipelines.static_adapt.adapt_pipeline import _phase1_position_probe_plan


_RAW_ROUTE_REQUESTS = (
    "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_"
    "full_insertion_diagnostic_v1",
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_full_insertion_diagnostic_v1",
    "sr_snake_macro_only_physical_lanes_full_insertion_diagnostic_v1",
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "no_prune_macro_only_physical_lanes_full_insertion_diagnostic_v1",
)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ACTIVE_PAPER_I_ALWAYS_PRODUCERS = (
    "chtc/phase3_optuna/generate_paper_i_scaling_matrix_records.py",
    "chtc/phase3_optuna/"
    "generate_paper_i_hh_current_snake_iter50_records.py",
    "chtc/phase3_optuna/preflight_submit.py",
)


@pytest.mark.parametrize(
    ("retired_mode", "match"),
    (
        ("full", "raw full insertion mode is retired"),
        ("always", "'always' insertion mode is retired"),
    ),
)
def test_runtime_rejects_retired_ambiguous_insertion_modes(
    retired_mode: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _phase1_position_probe_plan(
            insertion_mode=retired_mode,
            append_eval={},
            append_position=3,
            n_params=3,
            active_window_indices=[2],
            stage_name="core",
            drop_plateau_hits=0,
            max_grad=1.0,
            eps_grad=1.0e-8,
            finite_angle_fallback=False,
            repeated_family_flat=False,
            cfg=StageControllerConfig(max_probe_positions=4),
        )


@pytest.mark.parametrize(
    "relative_path",
    _ACTIVE_PAPER_I_ALWAYS_PRODUCERS,
)
def test_active_paper_i_producers_emit_only_reduced_always_insertion(
    relative_path: str,
) -> None:
    source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert '"adapt_insertion_mode": "full"' not in source
    assert '"--adapt-insertion-mode": "full"' not in source
    assert '"--adapt-insertion-mode": "always"' not in source
    assert "full_commutation_reduced" in source


@pytest.mark.parametrize("raw_profile", _RAW_ROUTE_REQUESTS)
def test_raw_full_insertion_profiles_are_not_reachable(raw_profile: str) -> None:
    assert raw_profile not in route_profiles.SR_ROUTE_PROFILE_REQUEST_CHOICES
    with pytest.raises(ValueError, match="sr_route_profile must be one of"):
        route_profiles.normalize_sr_route_profile_request(raw_profile)


def test_raw_full_insertion_profiles_are_not_public_exports() -> None:
    assert not any(
        "FULL_INSERTION_DIAGNOSTIC" in exported
        or "full_insertion_diagnostic" in exported
        for exported in route_profiles.__all__
    )


@pytest.mark.parametrize(
    ("contract_factory", "expected_profile"),
    (
        (
            route_profiles.canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_commutation_reduced_insertion_v1_contract,
            route_profiles.SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_COMMUTATION_REDUCED_INSERTION_V1,
        ),
        (
            route_profiles.canonical_sr_snake_macro_only_physical_lanes_commutation_reduced_insertion_diagnostic_v2_contract,
            route_profiles.SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2,
        ),
    ),
)
def test_always_insertion_profiles_lock_exact_commutation_reduction(
    contract_factory,
    expected_profile: str,
) -> None:
    contract = contract_factory()

    assert contract["route_profile"] == expected_profile
    assert (
        contract["execution_settings"]["adapt_insertion_mode"]
        == "full_commutation_reduced"
    )
    assert (
        contract["semantic_invariants"]["insertion_position_scope"]
        == "full_logical_ansatz_commutation_classes_every_depth_v2"
    )
    assert (
        contract["semantic_invariants"]["insertion_equivalence_policy"]
        == "termwise_cross_component_commutation_earliest_representative_v1"
    )
