from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.legacy.hh_benchmarks import hh_suzuki_benchmark as bench


def _hardware_row(method: str, scope: str, twoq: int, depth: int, size: int, **extra):
    row = {
        "method": method,
        "scope": scope,
        "basis": scope.replace("_", " "),
        "compiled_count_2q": twoq,
        "compiled_depth": depth,
        "compiled_size": size,
        "horizon_count_2q": None,
        "horizon_depth_serial": None,
    }
    row.update(extra)
    return row


def _fake_payload(
    *,
    final_exact: float = 1.0,
    final_abs: float = 0.01,
    mean_abs: float = 0.005,
    max_abs: float = 0.01,
) -> dict:
    return {
        "schema_version": "hh_realtime_suzuki_overlay_v1",
        "source": {
            "controller_json": "controller.json",
            "source_pdf": "source.pdf",
            "artifact_json": "seed.json",
        },
        "parameter_manifest": {
            "controller_json": "controller.json",
            "source_pdf": "source.pdf",
            "seed_artifact_json": "seed.json",
            "drive_enabled": True,
            "t_final": 8.0,
            "num_times": 161,
            "trotter_steps": 160,
            "compile_backend": "FakeMarrakesh",
            "compile_seed_transpiler": 7,
            "compile_optimization_level": 2,
            "exact_reference_method": "benchmark_exact",
            "exact_steps_multiplier": 2,
            "output_json": "overlay/hh_l2_t8_anchor_v1.json",
            "output_pdf": None,
        },
        "config": {
            "suzuki_orders": [2],
            "trotter_steps": 160,
            "export_compiled_circuits": False,
            "compiled_circuit_dir": None,
        },
        "methods": {
            "suzuki2": {
                "order": 2,
                "trajectory": [
                    {"time": 0.0, "energy_total": 1.0, "energy_total_exact": 1.0},
                    {"time": 8.0, "energy_total": 0.99, "energy_total_exact": final_exact},
                ],
                "summary": {
                    "row_count": 161,
                    "final_energy_total": 0.99,
                    "final_energy_total_exact": final_exact,
                    "final_abs_energy_total_error": final_abs,
                    "mean_abs_energy_total_error": mean_abs,
                    "max_abs_energy_total_error": max_abs,
                },
            }
        },
        "hardware_report_rows": [
            _hardware_row("suzuki2", "seed_prep_only", 66, 151, 295),
            _hardware_row("suzuki2", "per_step_evolution_only", 999, 1999, 2999),
            _hardware_row("suzuki2", "seed_plus_one_step_additive", 113, 277, 532),
            _hardware_row(
                "suzuki2",
                "full_horizon_with_seed_prep",
                1200,
                3400,
                5600,
                horizon_count_2q=1200,
                horizon_depth_serial=3400,
            ),
            _hardware_row("controller", "controller_state_at_time", 85, 188, 366),
        ],
        "written": {
            "output_json": "overlay/hh_l2_t8_anchor_v1.json",
            "output_pdf": None,
            "compiled_circuit_dir": None,
        },
    }


def _fake_payload_with_suzuki1() -> dict:
    payload = _fake_payload()
    payload["config"]["suzuki_orders"] = [1, 2]
    payload["methods"]["suzuki1"] = {
        "order": 1,
        "trajectory": [
            {"time": 0.0, "energy_total": 1.0, "energy_total_exact": 1.0},
            {"time": 8.0, "energy_total": 1.02, "energy_total_exact": 1.0},
        ],
        "summary": {
            "row_count": 161,
            "final_energy_total": 1.02,
            "final_energy_total_exact": 1.0,
            "final_abs_energy_total_error": 0.02,
            "mean_abs_energy_total_error": 0.015,
            "max_abs_energy_total_error": 0.02,
        },
    }
    payload["hardware_report_rows"].extend(
        [
            _hardware_row("suzuki1", "per_step_evolution_only", 54, 121, 240),
            _hardware_row("suzuki1", "seed_plus_one_step_additive", 98, 231, 450),
            _hardware_row(
                "suzuki1",
                "full_horizon_with_seed_prep",
                5000,
                15000,
                42000,
                horizon_count_2q=5000,
                horizon_depth_serial=15000,
            ),
        ]
    )
    return payload


def test_default_case_manifest_and_runner_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = bench.default_cases()[0]
    assert case.case_id == "hh_l2_t8_anchor_v1"
    assert case.controller_json == bench.overlay.DEFAULT_CONTROLLER_JSON
    assert case.source_pdf == bench.overlay.DEFAULT_SOURCE_PDF
    assert case.trotter_steps == 160
    assert case.suzuki_orders == (2,)
    assert case.skip_pdf is True
    assert case.export_compiled_circuits is False

    monkeypatch.setattr(bench.overlay, "_load_source_payload", lambda path: {"controller": str(path)})
    monkeypatch.setattr(
        bench.overlay,
        "_source_compile_defaults",
        lambda payload: {
            "backend_name": "FakeMarrakesh",
            "seed_transpiler": 7,
            "optimization_level": 2,
            "preferred_fake_backends": ("FakeMarrakesh", "FakeNighthawk"),
        },
    )
    calls = []

    def _fake_run_overlay(config, *, command=""):
        calls.append((config, command))
        payload = _fake_payload()
        payload["written"]["output_json"] = str(config.output_json)
        payload["parameter_manifest"]["output_json"] = str(config.output_json)
        payload["parameter_manifest"]["compile_backend"] = config.backend_name
        payload["parameter_manifest"]["compile_seed_transpiler"] = config.seed_transpiler
        payload["parameter_manifest"]["compile_optimization_level"] = config.optimization_level
        payload["config"]["export_compiled_circuits"] = config.export_compiled_circuits
        payload["config"]["compiled_circuit_dir"] = None
        return payload

    monkeypatch.setattr(bench.overlay, "run_overlay", _fake_run_overlay)

    result = bench.run_benchmark(cases=bench.default_cases(), output_dir=tmp_path, command="unit command")

    assert len(calls) == 1
    config, command = calls[0]
    assert command == "unit command"
    assert config.trotter_steps == 160
    assert config.suzuki_orders == (2,)
    assert config.skip_pdf is True
    assert config.export_compiled_circuits is False
    assert config.backend_name == "FakeMarrakesh"
    assert config.seed_transpiler == 7
    assert config.optimization_level == 2
    assert config.preferred_fake_backends == ("FakeMarrakesh", "FakeNighthawk")

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((tmp_path / "rows.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    overlay_payload = json.loads((tmp_path / "overlay" / "hh_l2_t8_anchor_v1.json").read_text(encoding="utf-8"))

    assert result["summary"]["row_count"] == 1
    assert manifest["schema_version"] == bench.SCHEMA_VERSION
    assert manifest["cases"][0]["case"]["case_id"] == "hh_l2_t8_anchor_v1"
    assert manifest["cases"][0]["resolved_overlay_config"]["suzuki_orders"] == [2]
    assert rows[0]["method_id"] == "hh_td_suzuki2"
    assert rows[0]["preferred_fake_backends"] == ["FakeMarrakesh", "FakeNighthawk"]
    assert summary["status_counts"] == {"ok": 1}
    assert overlay_payload["schema_version"] == "hh_realtime_suzuki_overlay_v1"


def test_row_extraction_selects_required_benchmark_scopes() -> None:
    row = bench._row_from_overlay_payload(
        _fake_payload(),
        case_id="hh_l2_t8_anchor_v1",
        artifact_overlay_json=Path("out/overlay/hh_l2_t8_anchor_v1.json"),
    )

    assert row["case_id"] == "hh_l2_t8_anchor_v1"
    assert row["method_id"] == "hh_td_suzuki2"
    assert row["method_kind"] == "suzuki_trotter"
    assert row["order"] == 2
    assert row["status"] == "ok"
    assert row["state_at_time_scope"] == "seed_plus_one_step_additive"
    assert row["state_at_time_2q"] == 113
    assert row["state_at_time_depth"] == 277
    assert row["state_at_time_size"] == 532
    assert row["full_horizon_scope"] == "full_horizon_with_seed_prep"
    assert row["full_horizon_2q"] == 1200
    assert row["full_horizon_depth"] == 3400
    assert row["full_horizon_size"] == 5600
    assert row["controller_state_scope"] == "controller_state_at_time"
    assert row["controller_state_2q"] == 85
    assert row["controller_state_depth"] == 188
    assert row["backend_name"] == "FakeMarrakesh"
    assert row["seed_transpiler"] == 7
    assert row["optimization_level"] == 2
    assert row["artifact_overlay_json"] == "out/overlay/hh_l2_t8_anchor_v1.json"
    assert row["exact_fields_reporting_only"] is True


def test_generic_extractor_can_extract_suzuki1() -> None:
    row = bench._row_from_overlay_method_payload(
        _fake_payload_with_suzuki1(),
        case_id="hh_l2_t8_anchor_v1",
        overlay_method="suzuki1",
        method_id="hh_td_suzuki1",
        method_kind="suzuki_trotter",
        expected_order=1,
        artifact_overlay_json=Path("out/overlay/hh_l2_t8_anchor_v1.json"),
    )

    assert row["case_id"] == "hh_l2_t8_anchor_v1"
    assert row["method_id"] == "hh_td_suzuki1"
    assert row["method_kind"] == "suzuki_trotter"
    assert row["order"] == 1
    assert row["state_at_time_scope"] == "seed_plus_one_step_additive"
    assert row["state_at_time_2q"] == 98
    assert row["state_at_time_depth"] == 231
    assert row["full_horizon_scope"] == "full_horizon_with_seed_prep"
    assert row["full_horizon_2q"] == 5000
    assert row["full_horizon_depth"] == 15000
    assert row["controller_state_scope"] == "controller_state_at_time"
    assert row["controller_state_2q"] == 85


def test_suzuki2_wrapper_matches_generic_extractor() -> None:
    payload = _fake_payload()
    wrapper = bench._row_from_overlay_payload(payload, case_id="hh_l2_t8_anchor_v1")
    generic = bench._row_from_overlay_method_payload(
        payload,
        case_id="hh_l2_t8_anchor_v1",
        overlay_method="suzuki2",
        method_id="hh_td_suzuki2",
        method_kind="suzuki_trotter",
        expected_order=2,
    )

    assert wrapper == generic


def test_missing_required_scope_fails_closed() -> None:
    payload = _fake_payload()
    payload["hardware_report_rows"] = [
        row
        for row in payload["hardware_report_rows"]
        if row["scope"] != "seed_plus_one_step_additive"
    ]

    with pytest.raises(ValueError, match="seed_plus_one_step_additive"):
        bench._row_from_overlay_payload(payload, case_id="hh_l2_t8_anchor_v1")


def test_exact_fields_are_reporting_only_for_row_hardware_selection() -> None:
    base = bench._row_from_overlay_payload(
        _fake_payload(final_exact=1.0, final_abs=0.01, mean_abs=0.005, max_abs=0.01),
        case_id="hh_l2_t8_anchor_v1",
    )
    changed_exact = bench._row_from_overlay_payload(
        _fake_payload(final_exact=2.0, final_abs=1.01, mean_abs=0.505, max_abs=1.01),
        case_id="hh_l2_t8_anchor_v1",
    )

    hardware_keys = [
        "state_at_time_scope",
        "state_at_time_2q",
        "state_at_time_depth",
        "state_at_time_size",
        "full_horizon_scope",
        "full_horizon_2q",
        "full_horizon_depth",
        "full_horizon_size",
        "controller_state_scope",
        "controller_state_2q",
        "controller_state_depth",
        "controller_state_size",
    ]
    for key in hardware_keys:
        assert changed_exact[key] == base[key]

    assert changed_exact["final_energy_total"] == base["final_energy_total"]
    assert changed_exact["final_energy_total_exact"] != base["final_energy_total_exact"]
    assert changed_exact["final_abs_energy_total_error"] != base["final_abs_energy_total_error"]
    assert changed_exact["mean_abs_energy_total_error"] != base["mean_abs_energy_total_error"]
    assert changed_exact["max_abs_energy_total_error"] != base["max_abs_energy_total_error"]
