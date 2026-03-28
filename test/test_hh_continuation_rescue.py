from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_rescue import (
    RescueConfig,
    rank_rescue_candidates,
    should_trigger_rescue,
)


def test_rescue_is_off_by_default() -> None:
    enabled, reason = should_trigger_rescue(
        enabled=False,
        exact_state_available=True,
        residual_opened=True,
        trough_detected=True,
        history=[{"delta_abs_drop_from_prev": 0.0}, {"delta_abs_drop_from_prev": 0.0}],
        shortlist_records=[{"full_v2_score": 1.0}, {"full_v2_score": 0.99}],
        cfg=RescueConfig(),
    )
    assert enabled is False
    assert reason == "disabled"


def test_rescue_requires_exact_state_and_flat_diagnostics() -> None:
    enabled, reason = should_trigger_rescue(
        enabled=True,
        exact_state_available=True,
        residual_opened=True,
        trough_detected=False,
        history=[{"delta_abs_drop_from_prev": 0.0}, {"delta_abs_drop_from_prev": 0.0}],
        shortlist_records=[{"full_v2_score": 1.0}, {"full_v2_score": 0.98}],
        cfg=RescueConfig(),
    )
    assert enabled is True
    assert reason == "flat_drop_and_shortlist"


def test_rank_rescue_candidates_is_deterministic_and_requires_gain() -> None:
    cfg = RescueConfig(min_overlap_gain=1e-4)
    best, meta = rank_rescue_candidates(
        records=[
            {"candidate_pool_index": 1, "position_id": 0, "full_v2_score": 0.8, "simple_score": 0.8},
            {"candidate_pool_index": 0, "position_id": 1, "full_v2_score": 0.8, "simple_score": 0.8},
        ],
        overlap_gain_fn=lambda rec: 0.2 if int(rec["candidate_pool_index"]) == 0 else 0.1,
        cfg=cfg,
    )
    assert meta["executed"] is True
    assert best is not None
    assert int(best["candidate_pool_index"]) == 0


def test_rank_rescue_candidates_skips_when_gain_too_small() -> None:
    best, meta = rank_rescue_candidates(
        records=[{"candidate_pool_index": 0, "position_id": 0, "full_v2_score": 0.5, "simple_score": 0.5}],
        overlap_gain_fn=lambda _rec: 0.0,
        cfg=RescueConfig(min_overlap_gain=1e-4),
    )
    assert best is None
    assert meta["reason"] == "insufficient_overlap_gain"


def test_rank_rescue_candidates_prefers_cheap_score_over_legacy_simple_tie_break() -> None:
    cfg = RescueConfig(min_overlap_gain=1e-6)
    best, meta = rank_rescue_candidates(
        records=[
            {
                "candidate_pool_index": 1,
                "position_id": 0,
                "cheap_score": 0.9,
                "simple_score": 0.1,
            },
            {
                "candidate_pool_index": 0,
                "position_id": 0,
                "cheap_score": 0.4,
                "simple_score": 9.0,
            },
        ],
        overlap_gain_fn=lambda _rec: 0.2,
        cfg=cfg,
    )
    assert meta["executed"] is True
    assert best is not None
    assert int(best["candidate_pool_index"]) == 1


def test_rank_rescue_candidates_phase3_exact_tie_ignores_legacy_simple_score() -> None:
    cfg = RescueConfig(min_overlap_gain=1e-6)
    best, meta = rank_rescue_candidates(
        records=[
            {
                "candidate_pool_index": 0,
                "position_id": 0,
                "cheap_score": 0.5,
                "cheap_score_version": "phase3_cheap_ratio_v1",
                "simple_score": 0.1,
            },
            {
                "candidate_pool_index": 1,
                "position_id": 0,
                "cheap_score": 0.5,
                "cheap_score_version": "phase3_cheap_ratio_v1",
                "simple_score": 9.0,
            },
        ],
        overlap_gain_fn=lambda _rec: 0.2,
        cfg=cfg,
    )
    assert meta["executed"] is True
    assert best is not None
    assert int(best["candidate_pool_index"]) == 0
