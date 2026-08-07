from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipelines.time_dynamics.diagnostics.avqds_method_comparison import (
    ComparisonCase,
    _event_times,
    _format_count,
    _stabilization_spans,
    load_comparison_case,
)


def _write_rows(path: Path, *, times: list[float], residual_scale: float) -> None:
    rows = []
    for index, time in enumerate(times):
        rows.append(
            {
                "time": time,
                "energy_expectation": -1.0 + 0.1 * index,
                "doublon": 0.5 + 0.01 * index,
                "mclachlan_residual_ratio": residual_scale * (index + 1),
                "reference_energy": -1.1 + 0.1 * index,
                "seed_reference_energy": -1.0 + 0.1 * index,
                "doublon_exact": 0.48 + 0.01 * index,
                "seed_doublon_exact": 0.5 + 0.01 * index,
                "site_occupations": [0.8 + 0.01 * index, 1.2 - 0.01 * index],
                "site_occupations_exact": [
                    0.79 + 0.01 * index,
                    1.21 - 0.01 * index,
                ],
                "seed_site_occupations_exact": [
                    0.8 + 0.01 * index,
                    1.2 - 0.01 * index,
                ],
            }
        )
    path.write_text(json.dumps({"plot_rows": rows}), encoding="utf-8")


def test_load_comparison_case_aligns_three_method_series(tmp_path: Path) -> None:
    ap = tmp_path / "ap.json"
    avqds = tmp_path / "avqds.json"
    avqds_t = tmp_path / "avqds_t.json"
    _write_rows(ap, times=[0.0, 0.5, 1.0], residual_scale=0.1)
    _write_rows(avqds, times=[0.0, 0.5, 1.0], residual_scale=0.01)
    _write_rows(avqds_t, times=[0.0, 0.5, 1.0], residual_scale=0.001)

    loaded = load_comparison_case(
        ComparisonCase(
            key="seed",
            label="Seed",
            ap_json=ap,
            avqds_json=avqds,
            avqds_t_json=avqds_t,
        )
    )

    assert [method.label for method in loaded.methods] == [
        "AP-McLachlan",
        "AVQDS",
        "AVQDS(T)",
    ]
    assert np.allclose(loaded.methods[2].residual, [0.001, 0.002, 0.003])
    assert np.allclose(loaded.exact_energy, [-1.1, -1.0, -0.9])
    assert np.allclose(loaded.seed_exact_doublon, [0.5, 0.51, 0.52])
    assert np.allclose(loaded.methods[0].site_occupations[:, 0], [0.8, 0.81, 0.82])
    assert np.allclose(loaded.exact_site_occupations[:, 1], [1.21, 1.20, 1.19])


def test_load_comparison_case_rejects_mismatched_time_grids(tmp_path: Path) -> None:
    ap = tmp_path / "ap.json"
    avqds = tmp_path / "avqds.json"
    avqds_t = tmp_path / "avqds_t.json"
    _write_rows(ap, times=[0.0, 0.5, 1.0], residual_scale=0.1)
    _write_rows(avqds, times=[0.0, 0.4, 1.0], residual_scale=0.01)
    _write_rows(avqds_t, times=[0.0, 0.5, 1.0], residual_scale=0.001)

    with pytest.raises(ValueError, match="time grids differ"):
        load_comparison_case(
            ComparisonCase(
                key="seed",
                label="Seed",
                ap_json=ap,
                avqds_json=avqds,
                avqds_t_json=avqds_t,
            )
        )


def test_event_markers_and_terminal_cost_count_format() -> None:
    rows = [
        {
            "time": 0.0,
            "patch_accepted": True,
            "patch_kind": "append",
            "patch_appended_count": 3,
            "solve_repair_applied": False,
        },
        {
            "time": 0.5,
            "patch_accepted": False,
            "solve_repair_applied": True,
        },
        {
            "time": 1.0,
            "patch_accepted": True,
            "patch_kind": "prune",
            "patch_deleted_count": 1,
            "solve_repair_applied": True,
        },
    ]

    append_times, prune_times = _event_times(rows)
    spans, stabilized_count = _stabilization_spans(rows)

    assert append_times == (0.0,)
    assert prune_times == (1.0,)
    assert spans == ((0.25, 1.25),)
    assert stabilized_count == 2
    assert _format_count(12996) == "12,996"
