from __future__ import annotations

import json

from pipelines.reporting import (
    build_paper_i_historical_mean_global_singleton_salvage_rough_page as report,
)


def test_scheduler_depth_is_mapped_to_entering_energy_round(tmp_path) -> None:
    stdout = tmp_path / "worker.out"
    rows = []
    for depth in range(1, 51):
        rows.append(
            "AI_LOG "
            + json.dumps(
                {
                    "event": "hardcoded_adapt_iter",
                    "depth": depth,
                    "energy": 1.0 - depth / 100.0,
                    "selected_position": depth - 1,
                    "best_op": f"op_{depth}",
                    "max_grad": 1.0 / depth,
                }
            )
        )
    stdout.write_text("\n".join(rows) + "\n", encoding="utf-8")

    parsed = report.parse_entering_energy_trace(stdout)

    assert len(parsed) == 50
    assert parsed[0]["transition_depth"] == 1
    assert parsed[0]["round"] == 0
    assert parsed[-1]["transition_depth"] == 50
    assert parsed[-1]["round"] == 49


def test_shared_crossing_uses_worse_horizon_minimum() -> None:
    ra = [
        {"round": 0, "delta_e": 1.0},
        {"round": 1, "delta_e": 0.3},
        {"round": 2, "delta_e": 0.01},
    ]
    append = [
        {"round": 0, "delta_e": 1.0},
        {"round": 1, "delta_e": 0.4},
        {"round": 2, "delta_e": 0.2},
    ]

    selected = report.first_shared_crossings(ra, append)

    assert selected["target_delta_e"] == 0.2
    assert selected["ra_crossing_round"] == 2
    assert selected["append_crossing_round"] == 2
    assert selected["ra_cost_available"] is False
    assert selected["append_matched_prefix_cost_compiled"] is False
