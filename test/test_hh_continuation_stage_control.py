from __future__ import annotations

from pipelines.hardcoded.hh_continuation_stage_control import (
    StageController,
    StageControllerConfig,
    allowed_positions,
    detect_trough,
    should_probe_positions,
)
import pytest


def test_allowed_positions_are_unique_and_bounded() -> None:
    out = allowed_positions(
        n_params=8,
        append_position=8,
        active_window_indices=[7, 3, 3, 0],
        max_positions=5,
    )
    assert len(out) <= 5
    assert out[0] == 8
    assert len(out) == len(set(out))
    assert all(0 <= x <= 8 for x in out)
    assert 4 not in out


def test_should_probe_on_plateau() -> None:
    cfg = StageControllerConfig(plateau_patience=2)
    probe, reason = should_probe_positions(
        stage_name="core",
        drop_plateau_hits=2,
        max_grad=1e-2,
        eps_grad=1e-4,
        append_score=1.0,
        finite_angle_flat=False,
        repeated_family_flat=False,
        cfg=cfg,
    )
    assert probe is True
    assert reason == "drop_plateau"


def test_should_probe_on_eps_grad_only_when_fallback_flat() -> None:
    cfg = StageControllerConfig()
    probe, reason = should_probe_positions(
        stage_name="core",
        drop_plateau_hits=0,
        max_grad=1e-5,
        eps_grad=1e-4,
        append_score=1.0,
        finite_angle_flat=True,
        repeated_family_flat=False,
        cfg=cfg,
    )
    assert probe is True
    assert reason == "eps_grad_flat"

    probe, reason = should_probe_positions(
        stage_name="core",
        drop_plateau_hits=0,
        max_grad=1e-5,
        eps_grad=1e-4,
        append_score=1.0,
        finite_angle_flat=False,
        repeated_family_flat=False,
        cfg=cfg,
    )
    assert probe is False
    assert reason == "default_append_only"


def test_should_probe_on_repeated_family_flat() -> None:
    cfg = StageControllerConfig()
    probe, reason = should_probe_positions(
        stage_name="core",
        drop_plateau_hits=0,
        max_grad=1e-2,
        eps_grad=1e-4,
        append_score=1.0,
        finite_angle_flat=False,
        repeated_family_flat=True,
        cfg=cfg,
    )
    assert probe is True
    assert reason == "family_repeat_flat"


def test_detect_trough_requires_positive_non_append_lcb() -> None:
    assert detect_trough(
        append_score=0.1,
        best_non_append_score=0.2,
        best_non_append_g_lcb=0.0,
        margin_ratio=1.0,
        append_admit_threshold=0.05,
    ) is False
    assert detect_trough(
        append_score=0.1,
        best_non_append_score=0.2,
        best_non_append_g_lcb=1e-3,
        margin_ratio=1.0,
        append_admit_threshold=0.05,
    ) is True


def test_detect_trough_accepts_non_append_when_append_below_floor() -> None:
    assert detect_trough(
        append_score=0.03,
        best_non_append_score=0.06,
        best_non_append_g_lcb=1e-3,
        margin_ratio=2.0,
        append_admit_threshold=0.05,
    ) is True


def test_stage_controller_seed_to_core_to_residual() -> None:
    ctrl = StageController(StageControllerConfig(plateau_patience=2))
    ctrl.start_with_seed()
    stage, reason = ctrl.resolve_stage_transition(
        drop_plateau_hits=0,
        trough_detected=False,
        residual_opened=False,
    )
    assert stage == "core"
    assert reason == "seed_complete"
    stage, reason = ctrl.resolve_stage_transition(
        drop_plateau_hits=2,
        trough_detected=False,
        residual_opened=False,
    )
    assert stage == "residual"
    assert reason == "plateau_without_trough"


def test_stage_controller_snapshot_tracks_runway_and_frontier() -> None:
    ctrl = StageController(
        StageControllerConfig(
            cap_phase1_min=1,
            cap_phase1_max=5,
            shot_min=8,
            shot_max=32,
        )
    )
    pre = ctrl.pre_step_snapshot(depth_local=2, max_depth=10)
    post = ctrl.finalize_step_snapshot(
        pre_snapshot=pre,
        phase1_raw_scores=[1.0, 0.5, 0.25],
        u_sigma_phase2=0.2,
        u_sigma_phase3=0.4,
    )
    assert post.depth_left == 8
    assert 0.0 <= post.runway_ratio <= 1.0
    assert 0.0 <= post.frontier_ratio <= 1.0
    assert post.phase_caps["phase1"] >= 1
    assert post.phase_shots["phase1"] >= 8
    assert post.phase_uncertainty["phase2"] == 0.2
    assert post.phase_uncertainty["phase3"] == 0.4


def test_stage_controller_records_admission_delta() -> None:
    ctrl = StageController(StageControllerConfig())
    ctrl.record_admission(selector_step=1, energy_before=-1.0, energy_after_refit=-1.1)
    snap = ctrl.snapshot()
    assert snap["admission_deltas"] == pytest.approx([0.1])
