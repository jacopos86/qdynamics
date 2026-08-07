from __future__ import annotations

import json
from pathlib import Path

import pytest

import pipelines.reporting.hh_fixed_manifold_dynamics_report as report_mod
from pipelines.reporting.hh_fixed_manifold_dynamics_report import (
    load_report_entry,
    write_fixed_manifold_dynamics_pdf,
)


"toy_source_settings = {physics fields}"
def _write_source_artifact(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": {
            "L": 2,
            "t": 1.0,
            "u": 4.0,
            "dv": 0.0,
            "omega0": 1.0,
            "g_ep": 0.5,
            "n_ph_max": 1,
            "boundary": "open",
            "ordering": "blocked",
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


"toy_measured_payload(label) = minimal_fixed_manifold_json"
def _write_measured_artifact(path: Path, *, source_artifact: Path, label: str, condition_number: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": "hh_fixed_manifold_measured_mclachlan_v1",
        "run_name": label,
        "input_artifact_json": str(source_artifact),
        "manifest": {
            "model_family": "Hubbard-Holstein",
            "geometry_backend": "oracle",
            "reference_audit_method": "exponential_midpoint_magnus2_order2",
            "drive_enabled": True,
            "noise_mode": "ideal",
        },
        "loader": {
            "loader_mode": "fixed_scaffold" if "locked" in label else "replay_family",
        },
        "run_config": {
            "t_final": 10.0,
            "num_times": 3,
        },
        "drive_profile": {
            "A": 0.6,
            "omega": 1.0,
            "tbar": 1.0,
            "phi": 0.0,
            "pattern": "staggered",
            "custom_weights": None,
            "time_sampling": "midpoint",
            "t0": 0.0,
        },
        "projection_config": {
            "integrator": "explicit_euler",
        },
        "summary": {
            "runtime_parameter_count": 7 if "locked" in label else 25,
            "final_fidelity_exact_audit": 0.997,
            "min_fidelity_exact_audit": 0.992,
            "max_abs_energy_total_error_exact_audit": 7.7e-3,
            "max_rho_miss": 1.0,
            "max_theta_dot_l2": 1.2e-10,
            "max_condition_number": condition_number,
        },
        "trajectory": [
            {
                "time": 0.0,
                "geometry": {
                    "rho_miss": 1.0,
                    "theta_dot_l2": 1.0e-12,
                    "condition_number": condition_number,
                },
                "audit": {
                    "fidelity_exact_audit": 1.0,
                    "energy_ansatz_exact_audit": 0.1590,
                    "energy_reference_exact_audit": 0.1590,
                    "abs_energy_total_error_exact_audit": 1.0e-16,
                },
            },
            {
                "time": 5.0,
                "geometry": {
                    "rho_miss": 1.0,
                    "theta_dot_l2": 1.0e-10,
                    "condition_number": condition_number,
                },
                "audit": {
                    "fidelity_exact_audit": 0.995,
                    "energy_ansatz_exact_audit": 0.1600,
                    "energy_reference_exact_audit": 0.1550,
                    "abs_energy_total_error_exact_audit": 5.0e-3,
                },
            },
            {
                "time": 10.0,
                "geometry": {
                    "rho_miss": 1.0,
                    "theta_dot_l2": 1.2e-10,
                    "condition_number": condition_number,
                },
                "audit": {
                    "fidelity_exact_audit": 0.997,
                    "energy_ansatz_exact_audit": 0.1595,
                    "energy_reference_exact_audit": 0.1530,
                    "abs_energy_total_error_exact_audit": 6.5e-3,
                },
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


"entry = load_report_entry(json_path)"
def test_load_report_entry_round_trip(tmp_path: Path) -> None:
    source_path = _write_source_artifact(tmp_path / "source.json")
    measured_path = _write_measured_artifact(
        tmp_path / "locked_7term_measured_drive.json",
        source_artifact=source_path,
        label="locked_7term",
        condition_number=15.8,
    )

    entry = load_report_entry(measured_path)

    assert entry.label == "locked_7term"
    assert entry.times.tolist() == [0.0, 5.0, 10.0]
    assert entry.source_settings["L"] == 2
    assert entry.condition_number.tolist() == [15.8, 15.8, 15.8]


"pdf = report(measured_1, measured_2)"
def test_write_fixed_manifold_dynamics_pdf(tmp_path: Path) -> None:
    source_path = _write_source_artifact(tmp_path / "source.json")
    locked_path = _write_measured_artifact(
        tmp_path / "locked_7term_measured_drive.json",
        source_artifact=source_path,
        label="locked_7term",
        condition_number=15.8,
    )
    pareto_path = _write_measured_artifact(
        tmp_path / "pareto_lean_l2_measured_drive.json",
        source_artifact=source_path,
        label="pareto_lean_l2",
        condition_number=7.4e7,
    )
    out_pdf = tmp_path / "fixed_manifold_report.pdf"

    result = write_fixed_manifold_dynamics_pdf(
        input_jsons=[locked_path, pareto_path],
        output_pdf=out_pdf,
        run_command="pytest fixed-manifold-report",
    )

    assert result == out_pdf.resolve()
    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 0


"missing_metric -> fail_closed"
def test_load_report_entry_missing_metric_fails(tmp_path: Path) -> None:
    source_path = _write_source_artifact(tmp_path / "source.json")
    measured_path = _write_measured_artifact(
        tmp_path / "broken_measured_drive.json",
        source_artifact=source_path,
        label="locked_7term",
        condition_number=15.8,
    )
    payload = json.loads(measured_path.read_text(encoding="utf-8"))
    del payload["trajectory"][1]["geometry"]["condition_number"]
    measured_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(KeyError):
        load_report_entry(measured_path)


"ambiguous_relative_source -> fail_closed"
def test_load_report_entry_ambiguous_relative_source_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo_root"
    base_root = tmp_path / "base_root"
    repo_root.mkdir()
    base_root.mkdir()
    source_rel = Path("artifacts/json/source.json")
    _write_source_artifact(repo_root / source_rel)
    _write_source_artifact(base_root / source_rel)
    monkeypatch.setattr(report_mod, "_REPO_ROOT", repo_root)

    measured_path = base_root / "measured.json"
    payload = {
        "pipeline": "hh_fixed_manifold_measured_mclachlan_v1",
        "run_name": "locked_7term",
        "input_artifact_json": str(source_rel),
        "manifest": {
            "drive_enabled": True,
        },
        "loader": {
            "loader_mode": "fixed_scaffold",
        },
        "trajectory": [
            {
                "time": 0.0,
                "geometry": {
                    "rho_miss": 1.0,
                    "theta_dot_l2": 1.0e-12,
                    "condition_number": 15.8,
                },
                "audit": {
                    "fidelity_exact_audit": 1.0,
                    "energy_ansatz_exact_audit": 0.1590,
                    "energy_reference_exact_audit": 0.1590,
                    "abs_energy_total_error_exact_audit": 1.0e-16,
                },
            }
        ],
        "summary": {
            "runtime_parameter_count": 7,
        },
    }
    measured_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Ambiguous relative artifact path"):
        load_report_entry(measured_path)


"missing_source_settings -> fail_closed"
def test_load_report_entry_missing_source_setting_fails(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps({"settings": {"L": 2, "t": 1.0}}), encoding="utf-8")
    measured_path = _write_measured_artifact(
        tmp_path / "locked_7term_measured_drive.json",
        source_artifact=source_path,
        label="locked_7term",
        condition_number=15.8,
    )

    with pytest.raises(KeyError, match="Missing source setting keys"):
        load_report_entry(measured_path)
