from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.scaffold.hh_continuation_stage_control import (
    StageController,
    StageControllerConfig,
)
from pipelines.static_adapt.cli_config import _build_adapt_arg_parser
from pipelines.static_adapt.sr_snake_route_profile import (
    SR_ROUTE_PROFILE_CONVENTIONAL_V3,
    canonical_sr_snake_v3_contract,
    canonical_sr_snake_v3_contract_sha256,
    canonical_sr_snake_no_prune_symmetric_cost_v1_contract,
    canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256,
)


def _parser():
    return _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-7)


@pytest.mark.parametrize(
    "removed_option",
    [
        "--phase-live-hysteresis-enabled",
        "--phase-live-hysteresis-disabled",
        "--phase2-null-nrem-high-threshold",
        "--phase2-live-nrem-low-threshold",
        "--phase3-null-nrem-high-threshold",
        "--phase3-live-nrem-low-threshold",
        "--phase2-hysteresis-steps",
        "--phase3-hysteresis-steps",
    ],
)
def test_phase_live_runtime_options_are_retired(removed_option: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        _parser().parse_args([removed_option, "1"])


def test_current_conventional_alias_resolves_to_retained_v3() -> None:
    args = _parser().parse_args(["--sr-route-profile", "sr_snake"])

    assert args.sr_route_profile_resolved == SR_ROUTE_PROFILE_CONVENTIONAL_V3
    assert not hasattr(args, "phase_live_hysteresis_enabled")
    assert args.sr_route_profile_contract == canonical_sr_snake_v3_contract()
    assert args.sr_route_profile_contract_sha256 == (
        canonical_sr_snake_v3_contract_sha256()
    )


def test_stage_controller_snapshots_are_fixed_all_live() -> None:
    controller = StageController(
        StageControllerConfig(
            cap_phase1_min=2,
            cap_phase1_max=2,
            cap_phase2_min=3,
            cap_phase2_max=3,
            cap_phase3_min=4,
            cap_phase3_max=4,
            shot_min=2,
            shot_max=2,
        ),
        configured_terminal_phase=2,
    )

    snapshot = controller.pre_step_snapshot(depth_local=0, max_depth=4)

    assert snapshot.phase_live == {
        "phase1": True,
        "phase2": True,
        "phase3": True,
    }
    assert snapshot.terminal_phase == 2
    assert snapshot.phase_null_streaks == {"phase2": 0, "phase3": 0}
    assert set(snapshot.phase_null_reasons.values()) == {
        "phase_live_retired_non_authoritative"
    }
    assert snapshot.phase_caps == {
        "phase1": 2,
        "phase2": 3,
        "phase3": 4,
    }
    assert snapshot.phase_shots == {
        "phase1": 2,
        "phase2": 2,
        "phase3": 2,
    }


def test_old_phase_live_snapshot_bytes_are_passive_on_restore() -> None:
    controller = StageController(
        StageControllerConfig(),
        configured_terminal_phase=3,
    )
    old_snapshot = controller.snapshot()
    old_snapshot.update(
        {
            "phase_live": {
                "phase1": True,
                "phase2": False,
                "phase3": False,
            },
            "phase_null_streaks": {"phase2": 8, "phase3": 9},
            "phase_null_reasons": {
                "phase1": None,
                "phase2": "old_phase2_null",
                "phase3": "old_phase3_null",
            },
        }
    )

    restored = StageController.from_snapshot(old_snapshot)
    snapshot = restored.pre_step_snapshot(depth_local=0, max_depth=2)

    assert snapshot.phase_live == {
        "phase1": True,
        "phase2": True,
        "phase3": True,
    }
    assert snapshot.phase_null_streaks == {"phase2": 0, "phase3": 0}
    assert snapshot.phase_caps["phase2"] > 0
    assert snapshot.phase_caps["phase3"] > 0
    assert snapshot.phase_shots["phase2"] > 0
    assert snapshot.phase_shots["phase3"] > 0


def test_historical_route_contract_digests_remain_byte_stable() -> None:
    assert canonical_sr_snake_v3_contract_sha256() == (
        "435910592e88f0136a0d45f611f79fe96b21d75fd25bad58276c871f39dc080e"
    )
    assert canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256() == (
        "ac973c2cd6f5dec3a6cd3274b9e255da6783196072bb2656ffabecd893e0da08"
    )
    contract = canonical_sr_snake_no_prune_symmetric_cost_v1_contract()
    assert contract["execution_settings"]["phase_live_hysteresis_enabled"] is False
    assert contract["semantic_invariants"]["phase_retirement_policy"] == "disabled_v1"
