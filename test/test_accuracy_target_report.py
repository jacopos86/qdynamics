from __future__ import annotations

import pytest

from pipelines.time_dynamics.accuracy_target_report import (
    _effective_threshold,
    _next_tighter_cut,
)


@pytest.mark.parametrize(
    ("cut", "expected"),
    (
        (1.0e-2, 3.0e-3),
        (3.0e-3, 1.0e-3),
        (1.0e-3, 3.0e-4),
        (3.0e-6, 1.0e-6),
    ),
)
def test_next_tighter_cut_continues_one_three_ladder(
    cut: float, expected: float
) -> None:
    assert _next_tighter_cut(cut) == pytest.approx(expected)


def test_effective_threshold_uses_the_controller_actually_run() -> None:
    run = {
        "summary": {
            "support_patch_config": {
                "dynamics_policy": "avqds",
                "avqds_l2_cut": 3.0e-4,
                "insertion_l2_cut": 1.0e-3,
            }
        }
    }
    assert _effective_threshold(run, "avqds") == pytest.approx(3.0e-4)
    assert _effective_threshold(run, "exchange") == pytest.approx(1.0e-3)


def test_effective_threshold_fails_closed_when_artifact_omits_it() -> None:
    with pytest.raises(ValueError, match="does not record"):
        _effective_threshold({"summary": {}}, "exchange")
