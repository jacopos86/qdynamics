from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.runners.ap_plot_energy_diagnostics import (
    build_energy_diagnostics_plot,
)


def test_energy_diagnostics_plot_writes_png(tmp_path: Path) -> None:
    run_path = tmp_path / "run.json"
    out_path = tmp_path / "plot.png"
    run_path.write_text(
        json.dumps(
            {
                "summary": {
                    "reference_energy_unmatched_count": 0,
                },
                "plot_rows": [
                    {
                        "time": 0.0,
                        "energy_expectation": 1.0,
                        "reference_energy": 1.0,
                        "abs_energy_error": 0.0,
                    },
                    {
                        "time": 0.1,
                        "energy_expectation": 1.1,
                        "reference_energy": 1.0,
                        "abs_energy_error": 0.1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = build_energy_diagnostics_plot(
        input_jsons=(run_path,),
        output_png=out_path,
        title="test",
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert payload["run_count"] == 1


def test_energy_diagnostics_plot_fails_on_missing_reference(tmp_path: Path) -> None:
    run_path = tmp_path / "run.json"
    out_path = tmp_path / "plot.png"
    run_path.write_text(
        json.dumps(
            {
                "summary": {
                    "reference_energy_unmatched_count": 1,
                },
                "plot_rows": [
                    {
                        "time": 0.0,
                        "energy_expectation": 1.0,
                        "reference_energy": None,
                        "abs_energy_error": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unmatched reference energy"):
        build_energy_diagnostics_plot(
            input_jsons=(run_path,),
            output_png=out_path,
        )
