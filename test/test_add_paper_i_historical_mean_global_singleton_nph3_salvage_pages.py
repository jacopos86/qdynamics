from __future__ import annotations

import json

import pytest

from pipelines.reporting import (
    add_paper_i_historical_mean_global_singleton_nph3_salvage_pages as subject,
)


def _event(depth: int, position: int, energy: float) -> str:
    return "AI_LOG " + json.dumps(
        {
            "event": "hardcoded_adapt_iter",
            "depth": depth,
            "selected_position": position,
            "energy": energy,
            "best_op": f"op-{depth}",
            "max_grad": 1.0 / depth,
        }
    )


def test_parse_scheduler_stdout_preserves_fifty_rounds_and_positions(tmp_path) -> None:
    path = tmp_path / "scheduler.out"
    lines = ["unrelated preamble"]
    lines.extend(
        _event(depth, depth - 1 if depth < 13 else depth - 2, -float(depth))
        for depth in range(1, 51)
    )
    lines.append('{"status":"passed"}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = subject.parse_scheduler_stdout(path)

    assert [row["round"] for row in rows] == list(range(1, 51))
    assert rows[11]["selected_position"] == 11
    assert rows[12]["selected_position"] == 11
    assert rows[-1]["energy"] == -50.0


def test_parse_scheduler_stdout_rejects_missing_round(tmp_path) -> None:
    path = tmp_path / "scheduler.out"
    path.write_text(
        "\n".join(
            _event(depth, depth - 1, -float(depth))
            for depth in range(1, 50)
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(subject.SalvagePageError, match="rounds 1..50"):
        subject.parse_scheduler_stdout(path)
