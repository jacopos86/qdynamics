from __future__ import annotations

from types import SimpleNamespace

import pipelines.static_adapt.adapt_pipeline as hardcoded_adapt
import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.scaffold.hh_continuation_types import PhaseControllerSnapshot
from pipelines.static_adapt import controller_telemetry


def _snapshot() -> PhaseControllerSnapshot:
    return PhaseControllerSnapshot(
        step_index=3,
        depth_local=4,
        depth_left=5,
        runway_ratio=0.25,
        early_coordinate=0.1,
        late_coordinate=0.9,
        frontier_ratio=0.75,
        u_stag=0.2,
        m_t=0.3,
        s_t=0.4,
        rho_t=0.5,
        gamma_t=0.6,
        u_front=0.7,
        n_rem_hat=8.0,
        useful_horizon=9.0,
        runway_fraction=0.33,
        H_t=1.25,
        phase_thresholds={"phase1": 0.1},
        phase_caps={"phase1": 2},
        phase_shots={"phase1": 16},
        phase_uncertainty={"phase1": 0.05},
        snapshot_version="phase123_controller_test",
        depth_runway_ratio=0.125,
        n_rem_low=7.0,
        n_rem_high=10.0,
        confidence_ratio=0.875,
        phase_live={"phase1": True},
        terminal_phase=2,
        phase_null_reasons={"phase3": "not_live"},
        phase_null_streaks={"phase3": 4},
        phase_caps_scheduled={"phase1": 3},
        phase_shots_maturity_floor={"phase1": 8},
        phase_shots_scheduled={"phase1": 32},
        phase_shots_snr={"phase1": 64},
        phase_shots_effective={"phase1": 48},
        phase_shot_uplift={"phase1": 1.5},
        phase_shot_fraction={"phase1": 0.5},
        phase_signal={"phase1": 2.0},
        phase_signal_floor={"phase1": 0.01},
    )


def test_controller_telemetry_helpers_remain_available_through_wrappers() -> None:
    helper_names = (
        "_branch_state_summary_payload",
        "_controller_snapshot_dict",
        "_controller_snapshot_payload",
        "_controller_telemetry_summary_payload",
    )
    for name in helper_names:
        assert getattr(adapt_pipeline, name) is getattr(controller_telemetry, name)
        assert getattr(hardcoded_adapt, name) is getattr(controller_telemetry, name)


def test_controller_snapshot_dict_preserves_attr_defaults_and_dicts() -> None:
    assert controller_telemetry._controller_snapshot_dict(None) is None

    payload = controller_telemetry._controller_snapshot_dict(
        SimpleNamespace(
            step_index="2",
            depth_local="3",
            runway_ratio="0.5",
            phase_thresholds={"phase1": 0.1},
            phase_live={"phase1": True},
        )
    )

    assert payload is not None
    assert payload["step_index"] == 2
    assert payload["depth_local"] == 3
    assert payload["depth_left"] == 0
    assert payload["runway_ratio"] == 0.5
    assert payload["frontier_ratio"] == 1.0
    assert payload["rho_t"] == 1.0
    assert payload["snapshot_version"] == "phase123_controller_v1"
    assert payload["phase_thresholds"] == {"phase1": 0.1}
    assert payload["phase_live"] == {"phase1": True}
    assert payload["terminal_phase"] == 3


def test_controller_snapshot_payload_preserves_phase_controller_shape() -> None:
    assert controller_telemetry._controller_snapshot_payload(None) is None
    assert controller_telemetry._controller_snapshot_payload(SimpleNamespace()) is None

    payload = controller_telemetry._controller_snapshot_payload(_snapshot())

    assert payload == {
        "snapshot_version": "phase123_controller_test",
        "step_index": 3,
        "depth_local": 4,
        "depth_left": 5,
        "runway_ratio": 0.25,
        "early_coordinate": 0.1,
        "late_coordinate": 0.9,
        "frontier_ratio": 0.75,
        "u_stag": 0.2,
        "m_t": 0.3,
        "s_t": 0.4,
        "rho_t": 0.5,
        "gamma_t": 0.6,
        "u_front": 0.7,
        "n_rem_hat": 8.0,
        "useful_horizon": 9.0,
        "runway_fraction": 0.33,
        "H_t": 1.25,
        "phase_thresholds": {"phase1": 0.1},
        "phase_caps": {"phase1": 2},
        "phase_shots": {"phase1": 16},
        "phase_uncertainty": {"phase1": 0.05},
        "depth_runway_ratio": 0.125,
        "n_rem_low": 7.0,
        "n_rem_high": 10.0,
        "confidence_ratio": 0.875,
        "phase_live": {"phase1": True},
        "terminal_phase": 2,
        "phase_null_reasons": {"phase3": "not_live"},
        "phase_null_streaks": {"phase3": 4},
        "phase_caps_scheduled": {"phase1": 3},
        "phase_shots_maturity_floor": {"phase1": 8},
        "phase_shots_scheduled": {"phase1": 32},
        "phase_shots_snr": {"phase1": 64},
        "phase_shots_effective": {"phase1": 48},
        "phase_shot_uplift": {"phase1": 1.5},
        "phase_shot_fraction": {"phase1": 0.5},
        "phase_signal": {"phase1": 2.0},
        "phase_signal_floor": {"phase1": 0.01},
    }


def test_controller_telemetry_and_branch_summary_payloads_preserve_shapes() -> None:
    snapshot = _snapshot()
    telemetry = controller_telemetry._controller_telemetry_summary_payload(
        stage_name="core",
        residual_opened=True,
        last_probe_reason="threshold",
        stage_events=[{"event": "first"}, object(), {"event": "last"}],
        last_snapshot=snapshot,
    )

    assert telemetry["telemetry_label"] == "T_b^ctrl"
    assert telemetry["stage_name"] == "core"
    assert telemetry["residual_opened"] is True
    assert telemetry["last_probe_reason"] == "threshold"
    assert telemetry["stage_event_count"] == 2
    assert telemetry["last_stage_event"] == {"event": "last"}
    assert telemetry["last_snapshot"] == controller_telemetry._controller_snapshot_payload(snapshot)

    branch_payload = controller_telemetry._branch_state_summary_payload(
        beam_enabled=True,
        branch_id=4,
        parent_branch_id=2,
        history_rows=[{"depth": 1}, object(), {"depth": 2}],
        depth_local=7,
        ansatz_depth=9,
        terminated=True,
        termination_label="max_depth",
        cumulative_selector_score=1.25,
        cumulative_selector_burden=2.5,
        stage_name="residual",
        residual_opened=False,
        last_probe_reason=None,
        stage_events=[{"event": "only"}],
        last_snapshot=snapshot,
    )

    assert branch_payload["branch_state_notation"] == "\\mathfrak b_*"
    assert branch_payload["status"] == "terminal"
    assert branch_payload["termination_label"] == "max_depth"
    assert branch_payload["beam_enabled"] is True
    assert branch_payload["branch_id"] == 4
    assert branch_payload["parent_branch_id"] == 2
    assert branch_payload["history_step_count"] == 2
    assert branch_payload["controller_telemetry"]["stage_event_count"] == 1
