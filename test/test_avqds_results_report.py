from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.diagnostics.avqds_results_report import (
    build_results_report_payload,
)


@pytest.mark.parametrize(
    ("schema", "steps_key", "residual_key", "family"),
    (
        (
            "generic_avqds_benchmark_v1",
            "avqds_steps",
            "rhs_residual_ratio",
            "AVQDS",
        ),
        (
            "generic_avqds_t_benchmark_v1",
            "avqds_t_steps",
            "target_tangent_residual_ratio",
            "PF-target adaptive tangent",
        ),
        (
            "generic_avqds_tetris_benchmark_v1",
            "avqds_tetris_steps",
            "rhs_residual_ratio",
            "AVQDS(T)",
        ),
    ),
)
def test_avqds_results_adapter_marks_append_at_interval_left_endpoint(
    tmp_path: Path,
    schema: str,
    steps_key: str,
    residual_key: str,
    family: str,
) -> None:
    times = (0.0, 0.5, 1.0)
    trajectory = []
    reference_rows = []
    for index, time_value in enumerate(times):
        seed_energy = -1.0 + 0.02 * index
        seed_doublon = 0.2 + 0.01 * index
        trajectory.append(
            {
                "checkpoint_index": index,
                "time": time_value,
                "energy_total": seed_energy + 0.001,
                "energy_total_exact": seed_energy,
                "doublon": seed_doublon + 0.002,
                "doublon_exact": seed_doublon,
                "site_occupations": [1.0 + 0.01 * index],
                "site_occupations_exact": [1.0],
                "runtime_parameter_count": 1 if index < 2 else 2,
                "logical_block_count": 1 if index < 2 else 2,
                residual_key: None if index == 0 else 0.01 / index,
            }
        )
        reference_rows.append(
            {
                "index": index,
                "time": time_value,
                "reference_energy": seed_energy - 0.003,
                "seed_reference_energy": seed_energy,
                "doublon_exact": seed_doublon - 0.004,
                "seed_doublon_exact": seed_doublon,
                "site_occupations_exact": [0.99],
            }
        )
    avqds_payload = {
        "schema_version": schema,
        "case": {"metadata": {"drive": {"enable_drive": True}}},
        "trajectory": trajectory,
        steps_key: [
            {"theta_dot": [0.1]},
            {"theta_dot": [0.1, 0.2]},
        ],
        "append_events": [
            {
                "interval_index": 1,
                "candidate_label": "candidate_a",
                "runtime_parameter_count": 2,
            }
        ],
        "metrics": {
            "candidate_pool_complete": True,
            "candidate_pool_size": 7,
        },
        "provenance": {"exact_reference_controller_inputs": False},
    }
    reference_payload = {"plot_rows": reference_rows}
    raw_path = tmp_path / "raw.json"
    reference_path = tmp_path / "reference.json"
    raw_path.write_text(json.dumps(avqds_payload), encoding="utf-8")
    reference_path.write_text(json.dumps(reference_payload), encoding="utf-8")

    report = build_results_report_payload(
        avqds_payload=avqds_payload,
        reference_ap_payload=reference_payload,
        raw_payload_path=raw_path,
        reference_ap_path=reference_path,
        label="AVQDS unit comparator",
        comparison_runs=(10,),
    )

    rows = report["plot_rows"]
    assert rows[1]["patch_accepted"] is True
    assert rows[1]["time"] == 0.5
    assert rows[1]["patch_appended_count"] == 1
    assert rows[2]["patch_accepted"] is False
    assert report["summary"]["accepted_append_count"] == 1
    assert report["summary"]["comparator_family"] == family
    assert report["summary"]["runtime_parameter_count_final"] == 2
    assert report["reporting_overlay_audit"][
        "seed_exact_overlay_consistency_passed"
    ] is True


def test_avqds_tetris_adapter_aggregates_layers_at_one_checkpoint(
    tmp_path: Path,
) -> None:
    trajectory = [
        {
            "checkpoint_index": 0,
            "time": 0.0,
            "energy_total": -1.0,
            "energy_total_exact": -1.0,
            "doublon": 0.2,
            "doublon_exact": 0.2,
            "site_occupations": [1.0],
            "site_occupations_exact": [1.0],
            "runtime_parameter_count": 10,
            "logical_block_count": 10,
            "rhs_residual_ratio": None,
        },
        {
            "checkpoint_index": 1,
            "time": 0.005,
            "energy_total": -0.999,
            "energy_total_exact": -0.999,
            "doublon": 0.201,
            "doublon_exact": 0.201,
            "site_occupations": [1.0],
            "site_occupations_exact": [1.0],
            "runtime_parameter_count": 15,
            "logical_block_count": 15,
            "rhs_residual_ratio": 0.01,
        },
    ]
    reference_rows = [
        {
            "index": index,
            "time": row["time"],
            "reference_energy": row["energy_total_exact"],
            "seed_reference_energy": row["energy_total_exact"],
            "doublon_exact": row["doublon_exact"],
            "seed_doublon_exact": row["doublon_exact"],
            "site_occupations_exact": row["site_occupations_exact"],
        }
        for index, row in enumerate(trajectory)
    ]
    payload = {
        "schema_version": "generic_avqds_tetris_benchmark_v1",
        "case": {"metadata": {"drive": {"enable_drive": True}}},
        "trajectory": trajectory,
        "avqds_tetris_steps": [{"theta_dot": [0.0] * 15}],
        "append_events": [
            {
                "interval_index": 0,
                "runtime_parameter_count": 12,
                "pauli_terms": ["xx", "yy"],
            },
            {
                "interval_index": 0,
                "runtime_parameter_count": 15,
                "pauli_terms": ["zi", "iz", "zz"],
            },
        ],
        "metrics": {
            "candidate_pool_complete": True,
            "candidate_pool_size": 7,
            "unsupported_checkpoint_count": 0,
        },
        "provenance": {"exact_reference_controller_inputs": False},
    }
    reference_payload = {"plot_rows": reference_rows}
    raw_path = tmp_path / "raw.json"
    reference_path = tmp_path / "reference.json"
    raw_path.write_text(json.dumps(payload), encoding="utf-8")
    reference_path.write_text(json.dumps(reference_payload), encoding="utf-8")

    report = build_results_report_payload(
        avqds_payload=payload,
        reference_ap_payload=reference_payload,
        raw_payload_path=raw_path,
        reference_ap_path=reference_path,
        label="AVQDS(T) unit comparator",
        comparison_runs=(),
    )

    event_row = report["plot_rows"][0]
    assert event_row["patch_accepted"] is True
    assert event_row["patch_appended_count"] == 5
    assert event_row["patch_selected_label"] == "+5 Pauli terms in 2 TETRIS layers"
    assert report["summary"]["accepted_append_count"] == 2
