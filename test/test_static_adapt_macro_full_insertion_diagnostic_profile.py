from __future__ import annotations

import pytest

from pipelines.scaffold.hh_continuation_stage_control import StageControllerConfig
from pipelines.static_adapt.adapt_pipeline import _phase1_position_probe_plan
from pipelines.static_adapt.sr_snake_route_profile import (
    SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2,
)
from test_support.route_contract_kwargs import route_identity


@pytest.mark.parametrize(
    ("insertion_mode", "match"),
    [
        ("full", "raw full insertion mode is retired"),
        ("always", "ambiguous capped-domain 'always' insertion mode is retired"),
    ],
)
def test_raw_full_insertion_mode_fails_closed(
    insertion_mode: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _phase1_position_probe_plan(
            insertion_mode=insertion_mode,
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
    _, baseline_contract, _ = route_identity(
        "sr_snake_macro_only_physical_lanes_v1"
    )
    diagnostic_resolved, diagnostic_contract, _ = route_identity(
        "sr_snake_macro_only_physical_lanes_"
        "commutation_reduced_insertion_diagnostic_v2"
    )

    baseline_settings = dict(baseline_contract["execution_settings"])
    diagnostic_settings = dict(diagnostic_contract["execution_settings"])
    assert baseline_settings.pop("adapt_insertion_mode") == "append_only"
    assert (
        diagnostic_settings.pop("adapt_insertion_mode")
        == "full_commutation_reduced"
    )
    assert diagnostic_settings == baseline_settings
    assert diagnostic_resolved == (
        SR_ROUTE_PROFILE_MACRO_ONLY_PHYSICAL_LANES_COMMUTATION_REDUCED_INSERTION_DIAGNOSTIC_V2
    )
