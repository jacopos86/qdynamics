from __future__ import annotations

from pipelines.exact_bench.hh_seq_transition_utils import (
    TransitionConfig,
    TransitionState,
    summarize_transition,
    update_transition_state,
)


def test_windowed_slope_transition_triggers_after_patience() -> None:
    cfg = TransitionConfig(
        window_k=4,
        slope_epsilon=5e-5,
        patience=2,
        min_points_before_switch=6,
    )
    state = TransitionState(cfg=cfg)

    seq = [0.10000, 0.09000, 0.08995, 0.08994, 0.08993, 0.08992]
    rec = None
    for val in seq:
        rec = update_transition_state(state, delta_abs=val)

    assert rec is not None
    assert bool(rec["switch_triggered"]) is True
    summary = summarize_transition(state)
    assert bool(summary["switch_triggered"]) is True
    assert int(summary["trigger_index"]) >= int(cfg.min_points_before_switch - 1)


def test_windowed_slope_transition_does_not_trigger_when_still_descending() -> None:
    cfg = TransitionConfig(
        window_k=4,
        slope_epsilon=1e-6,
        patience=2,
        min_points_before_switch=6,
    )
    state = TransitionState(cfg=cfg)

    for val in [1.0, 0.8, 0.65, 0.5, 0.38, 0.29, 0.22]:
        rec = update_transition_state(state, delta_abs=val)

    assert bool(rec["switch_triggered"]) is False
    summary = summarize_transition(state)
    assert bool(summary["switch_triggered"]) is False
