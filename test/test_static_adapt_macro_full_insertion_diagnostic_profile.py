from __future__ import annotations

import pytest

from pipelines.scaffold.hh_continuation_stage_control import StageControllerConfig
from pipelines.static_adapt.adapt_pipeline import _phase1_position_probe_plan
from pipelines.static_adapt.cli_config import _build_adapt_arg_parser
from pipelines.static_adapt.sr_snake_route_profile import (
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2,
)


def _parser():
    return _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-7)


def test_raw_full_insertion_diagnostic_route_is_retired() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "--sr-route-profile",
                "sr_snake_macro_only_physical_lanes_full_insertion_diagnostic_v1",
                "--adapt-max-depth",
                "15",
            ]
        )


def test_raw_full_insertion_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="raw full insertion mode is retired"):
        _phase1_position_probe_plan(
            insertion_mode="full",
            append_eval={},
            append_position=15,
            n_params=15,
            active_window_indices=[12, 13, 14],
            stage_name="core",
            drop_plateau_hits=0,
            max_grad=1.0,
            eps_grad=1.0e-8,
            finite_angle_fallback=False,
            repeated_family_flat=False,
            cfg=StageControllerConfig(max_probe_positions=2),
        )


def test_commutation_reduced_profile_preserves_v1_and_changes_only_insertion_policy() -> None:
    baseline = _parser().parse_args(
        [
            "--sr-route-profile",
            "sr_snake_macro_only_physical_lanes_v1",
            "--adapt-max-depth",
            "15",
        ]
    )
    diagnostic = _parser().parse_args(
        [
            "--sr-route-profile",
            (
                "sr_snake_macro_only_physical_lanes_"
                "commutation_reduced_insertion_diagnostic_v2"
            ),
            "--adapt-max-depth",
            "15",
        ]
    )

    baseline_settings = dict(
        baseline.sr_route_profile_contract["execution_settings"]
    )
    diagnostic_settings = dict(
        diagnostic.sr_route_profile_contract["execution_settings"]
    )
    assert baseline_settings.pop("adapt_insertion_mode") == "append_only"
    assert (
        diagnostic_settings.pop("adapt_insertion_mode")
        == "full_commutation_reduced"
    )
    assert diagnostic_settings == baseline_settings
    assert diagnostic.sr_route_profile_resolved == (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2
    )
    assert diagnostic.adapt_accepted_refit_coordinate_chart == (
        baseline.adapt_accepted_refit_coordinate_chart
    )
    assert diagnostic.adapt_allow_repeats is baseline.adapt_allow_repeats
