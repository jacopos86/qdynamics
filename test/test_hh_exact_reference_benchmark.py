from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.legacy.hh_benchmarks import hh_exact_reference_benchmark as bench


def _fake_source_payload() -> dict:
    return {
        "run_tag": "controller_anchor",
        "artifact_json": "seed.json",
        "reference": {
            "reference_mode": "benchmark_exact",
            "reference_enabled": True,
            "kind": "exact_trajectory",
            "reference_method": "unit_exact_method",
            "reference_steps_multiplier": 2,
            "times": [0.0, 1.0, 2.0],
            "projection_time_sampling": "midpoint",
            "geometry_sample_time_policy": "midpoint",
        },
        "drive_config": {
            "enabled": True,
            "drive_A": 0.55,
            "drive_omega": 2.0,
            "drive_tbar": 4.0,
            "drive_phi": 0.125,
            "drive_pattern": "staggered",
            "drive_time_sampling": "midpoint",
            "drive_t0": 4.0,
            "exact_steps_multiplier": 4,
        },
        "trajectory": [
            {
                "checkpoint_index": 0,
                "time": 0.0,
                "physical_time": 4.0,
                "trajectory_sample_kind": "state_sample",
                "energy_total": 999.0,
                "energy_total_controller": 999.0,
                "energy_total_exact": 1.0,
            },
            {
                "checkpoint_index": 100,
                "time": 0.5,
                "trajectory_sample_kind": "repair_event",
                "energy_total_exact": 999.0,
            },
            {
                "checkpoint_index": 1,
                "time": 1.0,
                "physical_time": 5.0,
                "trajectory_sample_kind": "state_sample",
                "energy_total": 999.0,
                "energy_total_controller": 999.0,
                "energy_total_exact": 1.25,
            },
            {
                "checkpoint_index": 200,
                "time": 1.5,
                "advances_time": False,
                "energy_total_exact": 999.0,
            },
            {
                "checkpoint_index": 2,
                "time": 2.0,
                "physical_time": 6.0,
                "trajectory_sample_kind": "state_sample",
                "energy_total": 999.0,
                "energy_total_controller": 999.0,
                "energy_total_exact": 1.5,
            },
        ],
    }


def test_default_case_manifest() -> None:
    case = bench.default_cases()[0]

    assert case.case_id == "hh_l2_t8_anchor_v1"
    assert case.controller_json == bench.overlay.DEFAULT_CONTROLLER_JSON


def test_row_extraction_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _fake_source_payload()
    monkeypatch.setattr(bench.overlay, "_load_source_payload", lambda path: source)
    monkeypatch.setattr(
        bench.overlay,
        "run_overlay",
        lambda *args, **kwargs: pytest.fail("exact reference benchmark must not call run_overlay"),
        raising=False,
    )

    result = bench.run_benchmark(cases=bench.default_cases(), output_dir=tmp_path, command="unit command")

    row = result["rows"][0]
    assert row["case_id"] == "hh_l2_t8_anchor_v1"
    assert row["method_id"] == "hh_td_exact_reference_v1"
    assert row["method_kind"] == "exact_reference"
    assert row["status"] == "ok"
    assert row["exact_reference_method"] == "unit_exact_method"
    assert row["qpu_faithful"] is False
    assert row["diagnostic_exact_assisted"] is False
    assert row["hardware_cost_applicable"] is False
    assert row["final_energy_total"] == pytest.approx(1.5)
    assert row["final_energy_total_exact"] == pytest.approx(1.5)
    assert row["final_abs_energy_total_error"] == 0.0
    assert row["mean_abs_energy_total_error"] == 0.0
    assert row["max_abs_energy_total_error"] == 0.0
    assert row["state_at_time_2q"] is None
    assert row["state_at_time_depth"] is None
    assert row["full_horizon_2q"] is None
    assert row["full_horizon_depth"] is None
    assert row["controller_state_2q"] is None
    assert row["controller_state_depth"] is None
    assert row["reference_steps_multiplier"] == 2
    assert row["exact_steps_multiplier"] == 4
    assert row["controller_energy_fields_read"] is False

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((tmp_path / "rows.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    reference = json.loads((tmp_path / "reference" / "hh_l2_t8_anchor_v1.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == bench.SCHEMA_VERSION
    assert manifest["method_contract"]["method_id"] == "hh_td_exact_reference_v1"
    assert manifest["method_contract"]["hardware_cost_applicable"] is False
    assert rows[0]["artifact_reference_json"] == str(tmp_path / "reference" / "hh_l2_t8_anchor_v1.json")
    assert summary["status_counts"] == {"ok": 1}
    assert summary["key_metrics"][0]["final_abs_energy_total_error"] == 0.0
    assert reference["schema_version"] == bench.REFERENCE_SCHEMA_VERSION
    assert reference["summary"]["num_times"] == 3
    assert reference["summary"]["final_energy_total_exact"] == pytest.approx(1.5)
    assert reference["trajectory"][-1]["energy_total"] == pytest.approx(1.5)
    assert reference["trajectory"][-1]["energy_total_exact"] == pytest.approx(1.5)
    assert reference["trajectory"][-1]["abs_energy_total_error"] == 0.0


def test_missing_provenance_fails_closed() -> None:
    source = _fake_source_payload()
    source["reference"].pop("reference_method")

    with pytest.raises(ValueError, match=r"reference\.reference_method"):
        bench._reference_payload_from_source(source, case=bench.default_cases()[0])


def test_missing_energy_total_exact_fails_closed() -> None:
    source = _fake_source_payload()
    source["trajectory"][2].pop("energy_total_exact")

    with pytest.raises(ValueError, match="energy_total_exact"):
        bench._reference_payload_from_source(source, case=bench.default_cases()[0])


def test_state_sample_filtering() -> None:
    source = _fake_source_payload()
    payload = bench._reference_payload_from_source(source, case=bench.default_cases()[0])

    assert payload["summary"]["num_times"] == 3
    assert [row["time"] for row in payload["trajectory"]] == [0.0, 1.0, 2.0]
    assert [row["checkpoint_index"] for row in payload["trajectory"]] == [0, 1, 2]
    assert [row["energy_total_exact"] for row in payload["trajectory"]] == [1.0, 1.25, 1.5]

