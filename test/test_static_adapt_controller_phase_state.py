from __future__ import annotations

from types import SimpleNamespace

import pipelines.static_adapt.adapt_pipeline as hardcoded_adapt
import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.static_adapt import controller_phase_state


def test_controller_phase_state_helpers_remain_available_through_wrappers() -> None:
    helper_names = (
        "_controller_cap",
        "_controller_phase_shots",
        "_controller_threshold",
        "_record_controller_snapshot",
    )
    for name in helper_names:
        assert getattr(adapt_pipeline, name) is getattr(controller_phase_state, name)
        assert getattr(hardcoded_adapt, name) is getattr(controller_phase_state, name)


def test_record_controller_snapshot_prefers_record_mapping() -> None:
    snapshot = {"depth_left": 3}
    payload = controller_phase_state._record_controller_snapshot(
        {"controller_snapshot": snapshot}
    )

    assert payload == {"depth_left": 3}
    assert payload is not snapshot
    assert controller_phase_state._record_controller_snapshot(None) is None
    assert controller_phase_state._record_controller_snapshot({"feature": object()}) is None


def test_controller_phase_shots_ignores_passive_historical_liveness() -> None:
    assert controller_phase_state._controller_phase_shots(None, "phase1", 0) == 1
    assert (
        controller_phase_state._controller_phase_shots(
            {"phase_live": {"phase1": False}, "phase_shots_effective": {"phase1": 99}},
            "phase1",
        )
        == 99
    )
    assert (
        controller_phase_state._controller_phase_shots(
            {
                "phase_live": {"phase1": True},
                "phase_shots_effective": {"phase1": "2.6"},
                "phase_shots": {"phase1": 7},
            },
            "phase1",
        )
        == 3
    )
    assert (
        controller_phase_state._controller_phase_shots(
            SimpleNamespace(
                phase_live={"phase2": True},
                phase_shots_effective={"phase2": "bad"},
                phase_shots={"phase2": 4},
            ),
            "phase2",
        )
        == 4
    )
    assert (
        controller_phase_state._controller_phase_shots(
            {"phase_live": {"phase3": True}},
            "phase3",
            default_value=-5,
        )
        == 1
    )


def test_controller_cap_and_threshold_ignore_passive_historical_liveness() -> None:
    assert controller_phase_state._controller_cap(None, "phase1", 0) == 1
    assert (
        controller_phase_state._controller_cap(
            {"phase_live": {"phase1": False}, "phase_caps": {"phase1": 10}},
            "phase1",
            2,
        )
        == 10
    )
    assert (
        controller_phase_state._controller_cap(
            SimpleNamespace(phase_live={"phase2": True}, phase_caps={"phase2": 5}),
            "phase2",
            2,
        )
        == 5
    )
    assert (
        controller_phase_state._controller_cap(
            {"phase_live": {"phase3": True}, "phase_caps": {"phase3": 0}},
            "phase3",
            2,
        )
        == 1
    )

    assert controller_phase_state._controller_threshold(None, "phase1") == 0.0
    assert (
        controller_phase_state._controller_threshold(
            SimpleNamespace(phase_thresholds={"phase1": "0.25"}),
            "phase1",
        )
        == 0.25
    )
    assert (
        controller_phase_state._controller_threshold(
            {"phase_thresholds": {"phase2": 0.75}},
            "phase2",
        )
        == 0.75
    )
