#!/usr/bin/env python3
"""Tests for generic static Table-I output checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from chtc.phase3_optuna import check_generic_static_table_outputs as check


def _write_records(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = ("record_id", "family", "case_id", "algorithm_id")
    extras = tuple(key for row in rows for key in row if key not in base)
    fieldnames = tuple(dict.fromkeys((*base, *extras)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_payload(root: Path, record_id: str) -> None:
    result_dir = root / record_id / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "generic_static_single.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "result": {
                    "status": "ok",
                    "energy": -1.0,
                    "abs_delta_e": 0.0,
                    "count_2q": 1,
                    "circuit_depth": 2,
                    "phase3_controller_called": False,
                    "shots_total": 100,
                    "compiled_depth_total": 2,
                    "compiled_count_2q_total": 1,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_benchmark_value_noise_payload(
    root: Path,
    record_id: str,
    *,
    model: str = "gaussian_iid_v1",
    std: float = 1e-6,
    seed: int = 20260514,
    noise_draw: float = 2e-6,
) -> None:
    result_dir = root / record_id / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    energy_ideal = -1.0
    exact_energy = -1.25
    energy = energy_ideal + noise_draw
    delta = abs(energy - exact_energy)
    row_noise = {
        "enabled": True,
        "model": model,
        "std": std,
        "seed": seed,
        "seed_source": "env",
        "semantic": "post_static_result_value_noise_not_physical_shots",
        "noise_draw": noise_draw,
        "energy_pre_benchmark_value_noise": energy_ideal,
        "benchmark_value_noise_energy_ideal": energy_ideal,
        "physical_shots_unchanged": True,
        "scope": {
            "kind": "static_benchmark_row",
            "family": "hh",
            "case_id": "hh_L2",
            "algorithm_id": "static_hea_qiskit_vqe",
        },
        "applied": True,
        "status": "ok",
    }
    payload_noise = {
        "enabled": True,
        "model": model,
        "std": std,
        "seed": seed,
        "seed_source": "env",
        "semantic": "post_static_result_value_noise_not_physical_shots",
        "physical_shots_unchanged": True,
        "scope": "generic_static_benchmark_dispatch_payload",
        "status": "ok",
        "status_counts": {"ok": 1},
        "row_target_count": 1,
        "applied_row_count": 1,
    }
    (result_dir / "generic_static_single.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "benchmark_value_noise_status": "ok",
                "benchmark_value_noise": payload_noise,
                "result": {
                    "status": "ok",
                    "energy": energy,
                    "energy_ideal": -999.0,
                    "energy_pre_benchmark_value_noise": energy_ideal,
                    "benchmark_value_noise_energy_ideal": energy_ideal,
                    "exact_energy": exact_energy,
                    "exact_gs_energy": exact_energy,
                    "delta_E_abs": delta,
                    "delta_E_abs_ideal": 0.25,
                    "abs_delta_e": delta,
                    "abs_delta_e_ideal": 0.25,
                    "count_2q": 1,
                    "circuit_depth": 2,
                    "phase3_controller_called": False,
                    "shots_total": 100,
                    "compiled_depth_total": 2,
                    "compiled_count_2q_total": 1,
                    "benchmark_value_noise_status": "ok",
                    "benchmark_value_noise": row_noise,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _decision_noise_metadata(*, status: str = "unsupported", model: str = "gaussian_iid_v1", std: float = 1e-6, seed: int = 20260515) -> dict[str, object]:
    common = {
        "enabled": True,
        "model": model,
        "std": std,
        "seed": seed,
        "seed_source": "env",
        "semantic": "benchmark_decision_value_noise_not_physical_shots_v1",
        "physical_shots_unchanged": True,
        "algorithmic_measurement_work_schema": "algorithmic_measurement_work_v1",
        "algorithmic_measurement_work_unchanged": True,
        "scope": {"family": "hh", "case_id": "hh_L2", "algorithm_id": "static_hea_qiskit_vqe"},
    }
    if status == "ok":
        return {
            **common,
            "status": "ok",
            "supported": True,
            "applied": True,
            "draw_count_total": 2,
            "draw_count_by_surface": {"objective": 2},
            "surfaces_affected": ["objective"],
            "trace_preview": [],
            "trace_truncated_count": 0,
        }
    return {
        **common,
        "status": "unsupported",
        "supported": False,
        "applied": False,
        "fail_closed": True,
        "reason": "foundation slice unsupported decision-noise runner",
        "dispatch": "generic_static_hea_qiskit_vqe",
        "draw_count_total": 0,
        "draw_count_by_surface": {},
        "surfaces_affected": [],
        "trace_preview": [],
        "trace_truncated_count": 0,
    }


def _write_benchmark_decision_noise_payload(
    root: Path,
    record_id: str,
    *,
    status: str = "unsupported",
    model: str = "gaussian_iid_v1",
    std: float = 1e-6,
    seed: int = 20260515,
) -> None:
    result_dir = root / record_id / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    meta = _decision_noise_metadata(status=status, model=model, std=std, seed=seed)
    row = {
        "status": "ok" if status == "ok" else "skipped_unsupported_decision_noise",
        "energy": -1.0 if status == "ok" else None,
        "abs_delta_e": 0.0 if status == "ok" else None,
        "count_2q": 1 if status == "ok" else None,
        "circuit_depth": 2 if status == "ok" else None,
        "quality_gate_reason": "" if status == "ok" else "benchmark_decision_noise_unsupported",
        "phase3_controller_called": False,
        "shots_total": 100 if status == "ok" else 0,
        "compiled_depth_total": 2 if status == "ok" else 0,
        "compiled_count_2q_total": 1 if status == "ok" else 0,
        "benchmark_decision_noise_status": status,
        "benchmark_decision_noise": dict(meta),
    }
    payload = {
        "status": "completed" if status == "ok" else "skipped_unsupported_decision_noise",
        "benchmark_decision_noise_status": status,
        "benchmark_decision_noise": dict(meta),
        "result": row,
        "rows": [dict(row)],
    }
    for name, content in {
        "generic_static_single.json": payload,
        "result.json": payload,
        "manifest.json": {"schema": "manifest", **payload},
        "rows.json": {"schema": "rows", "benchmark_decision_noise_status": status, "benchmark_decision_noise": dict(meta), "rows": [dict(row)]},
    }.items():
        (result_dir / name).write_text(json.dumps(content) + "\n", encoding="utf-8")


def _write_phase3_value_noise_payload(root: Path, record_id: str, *, model: str = "gaussian_iid_v1", std: float = 1e-6, seed: int = 20260514) -> None:
    result_dir = root / record_id / "result"
    adapt_path = result_dir / "hh_L2" / "json" / "result.json"
    adapt_path.parent.mkdir(parents=True, exist_ok=True)
    adapt_path.write_text(
        json.dumps(
            {
                "continuation": {
                    "oracle_gradient_config": {
                        "value_noise": {
                            "enabled": True,
                            "model": model,
                            "std": std,
                            "seed": seed,
                            "semantic": "post_expectation_value_noise_not_physical_shots",
                        }
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "generic_static_single.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "result": {
                    "benchmark_id": "hh_L2",
                    "result_json": str(adapt_path),
                    "status": "ok",
                    "energy": -1.0,
                    "abs_delta_e": 0.0,
                    "count_2q": 1,
                    "circuit_depth": 2,
                    "phase3_controller_called": False,
                    "shots_total": 100,
                    "compiled_depth_total": 2,
                    "compiled_count_2q_total": 1,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_enrichment(
    root: Path,
    record_id: str,
    *,
    s_status: str = "ok",
    s_alg_status: str | None = None,
) -> None:
    result_dir = root / record_id / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    row_updates = {"S_norm": 5.0} if s_status == "ok" else {}
    statuses = {"S_norm": s_status}
    metrics = {}
    if s_alg_status is not None:
        statuses["S_alg"] = s_alg_status
        if s_alg_status == "ok":
            row_updates.update(
                {
                    "S_alg": 7.0,
                    "S_alg_N_H_outer_eval": 1.0,
                    "S_alg_N_grad_probe": 2.0,
                    "S_alg_N_metric_probe": 3.0,
                    "S_alg_N_H_refit_eval": 1.0,
                    "S_alg_N_other_quantum": 0.0,
                }
            )
            metrics["algorithmic_measurement_work"] = {
                "schema": "algorithmic_measurement_work_v1",
                "status": "ok",
                "source_kind": "explicit_components",
                "components": {
                    "N_H_outer_eval": 1.0,
                    "N_grad_probe": 2.0,
                    "N_metric_probe": 3.0,
                    "N_H_refit_eval": 1.0,
                },
            }
    (result_dir / "generic_static_metric_enrichment.json").write_text(
        json.dumps(
            {
                "schema": "generic_static_metric_enrichment_v1",
                "record_id": record_id,
                "status": "completed",
                "row_updates": row_updates,
                "metric_statuses": statuses,
                "metrics": metrics,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_check_generic_static_table_outputs_require_s_norm_passes(tmp_path: Path) -> None:
    record_id = "static_table__bose_hubbard__bose_hubbard_L2__static_family_informed_vqe"
    records = [{"record_id": record_id, "family": "bose_hubbard", "case_id": "bose_hubbard_L2", "algorithm_id": "static_family_informed_vqe"}]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_payload(tmp_path / "root", record_id)
    _write_enrichment(tmp_path / "enrichment", record_id, s_status="ok")

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        enrichment_root=tmp_path / "enrichment",
        require_s_norm=True,
    )

    assert summary["enrichment_violation_count"] == 0
    assert summary["s_norm_status_counts"] == {"ok": 1}


def test_check_generic_static_table_outputs_require_s_norm_flags_missing(tmp_path: Path) -> None:
    record_id = "static_table__bose_hubbard__bose_hubbard_L2__static_family_informed_vqe"
    records = [{"record_id": record_id, "family": "bose_hubbard", "case_id": "bose_hubbard_L2", "algorithm_id": "static_family_informed_vqe"}]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_payload(tmp_path / "root", record_id)
    _write_enrichment(tmp_path / "enrichment", record_id, s_status="missing_component_breakdown")

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        enrichment_root=tmp_path / "enrichment",
        require_s_norm=True,
    )

    assert summary["enrichment_violation_count"] == 1
    assert summary["s_norm_status_counts"] == {"missing_component_breakdown": 1}
    assert summary["enrichment_violations"][0]["violation"] == "S_norm_required"


def test_check_generic_static_table_outputs_require_s_norm_without_enrichment_root_flags(tmp_path: Path) -> None:
    record_id = "static_table__bose_hubbard__bose_hubbard_L2__static_family_informed_vqe"
    records = [{"record_id": record_id, "family": "bose_hubbard", "case_id": "bose_hubbard_L2", "algorithm_id": "static_family_informed_vqe"}]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_payload(tmp_path / "root", record_id)

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        require_s_norm=True,
    )

    assert summary["enrichment_violation_count"] == 1
    assert summary["s_norm_status_counts"] == {"missing": 1}
    assert summary["enrichment_violations"][0]["violation"] == "S_norm_required_without_enrichment_root"


def test_check_generic_static_table_outputs_contract_allows_native_phase3_called_true() -> None:
    payload = {
        "algorithm_id": "static_family_native_adapt_phase3",
        "result": {
            "phase3_controller_called": True,
            "shots_total": 107,
            "compiled_depth_total": 166,
            "compiled_count_2q_total": 65,
        },
    }

    assert check._contract_violations(payload) == []


def test_check_generic_static_table_outputs_require_value_noise_applied_passes(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2__static_family_native_adapt_phase3"
    records = [
        {
            "record_id": record_id,
            "family": "hh",
            "case_id": "hh_L2",
            "algorithm_id": "static_family_native_adapt_phase3",
            "phase3_oracle_gradient_mode": "aer_density_matrix",
            "phase3_oracle_value_noise_model": "gaussian_iid_v1",
            "phase3_oracle_value_noise_std": "1e-06",
            "phase3_oracle_value_noise_seed": "20260514",
        }
    ]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_phase3_value_noise_payload(tmp_path / "root", record_id)

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        require_value_noise_applied=True,
        expected_value_noise_model="gaussian_iid_v1",
        expected_value_noise_std=1e-6,
        expected_value_noise_seed=20260514,
    )

    assert summary["value_noise_violation_count"] == 0
    assert summary["value_noise_status_counts"] == {"ok": 1}


def test_check_generic_static_table_outputs_require_benchmark_value_noise_applied_passes_non_phase3(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2__static_hea_qiskit_vqe"
    records = [
        {
            "record_id": record_id,
            "family": "hh",
            "case_id": "hh_L2",
            "algorithm_id": "static_hea_qiskit_vqe",
            "benchmark_value_noise_model": "gaussian_iid_v1",
            "benchmark_value_noise_std": "1e-06",
            "benchmark_value_noise_seed": "20260514",
        }
    ]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_benchmark_value_noise_payload(tmp_path / "root", record_id)

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        require_value_noise_applied=True,
        expected_value_noise_model="gaussian_iid_v1",
        expected_value_noise_std=1e-6,
        expected_value_noise_seed=20260514,
    )

    assert summary["value_noise_violation_count"] == 0
    assert summary["value_noise_status_counts"] == {"ok": 1}


def test_check_generic_static_table_outputs_require_decision_noise_accepts_unsupported(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2__static_hea_qiskit_vqe"
    records = [
        {
            "record_id": record_id,
            "family": "hh",
            "case_id": "hh_L2",
            "algorithm_id": "static_hea_qiskit_vqe",
            "benchmark_decision_noise_model": "gaussian_iid_v1",
            "benchmark_decision_noise_std": "1e-06",
            "benchmark_decision_noise_seed": "20260515",
        }
    ]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_benchmark_decision_noise_payload(tmp_path / "root", record_id, status="unsupported")

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        require_decision_noise_handled=True,
        expected_decision_noise_model="gaussian_iid_v1",
        expected_decision_noise_std=1e-6,
        expected_decision_noise_seed=20260515,
    )

    assert summary["decision_noise_violation_count"] == 0
    assert summary["decision_noise_status_counts"] == {"unsupported": 1}
    assert summary["bad_count"] == 0
    assert summary["status_by_algorithm"]["static_hea_qiskit_vqe"]["decision_noise_unsupported"] == 1


def test_check_generic_static_table_outputs_decision_noise_rejects_unsupported_without_fail_closed(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2__static_hea_qiskit_vqe"
    records = [
        {
            "record_id": record_id,
            "family": "hh",
            "case_id": "hh_L2",
            "algorithm_id": "static_hea_qiskit_vqe",
            "benchmark_decision_noise_model": "gaussian_iid_v1",
            "benchmark_decision_noise_std": "1e-06",
            "benchmark_decision_noise_seed": "20260515",
        }
    ]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_benchmark_decision_noise_payload(tmp_path / "root", record_id, status="unsupported")
    payload_path = tmp_path / "root" / record_id / "result" / "generic_static_single.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["benchmark_decision_noise"]["fail_closed"] = False
    payload_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        require_decision_noise_handled=True,
    )

    assert summary["decision_noise_violation_count"] >= 1
    assert summary["decision_noise_status_counts"] == {"decision_unsupported_not_fail_closed": 1}


def test_check_generic_static_table_outputs_require_decision_noise_accepts_future_ok_payload(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2__static_hea_qiskit_vqe"
    records = [
        {
            "record_id": record_id,
            "family": "hh",
            "case_id": "hh_L2",
            "algorithm_id": "static_hea_qiskit_vqe",
            "benchmark_decision_noise_model": "gaussian_iid_v1",
            "benchmark_decision_noise_std": "1e-06",
            "benchmark_decision_noise_seed": "20260515",
        }
    ]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_benchmark_decision_noise_payload(tmp_path / "root", record_id, status="ok")

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        require_decision_noise_handled=True,
    )

    assert summary["decision_noise_violation_count"] == 0
    assert summary["decision_noise_status_counts"] == {"ok": 1}


def test_check_generic_static_table_outputs_decision_noise_rejects_post_result_only_value_noise(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2__static_hea_qiskit_vqe"
    records = [
        {
            "record_id": record_id,
            "family": "hh",
            "case_id": "hh_L2",
            "algorithm_id": "static_hea_qiskit_vqe",
            "benchmark_decision_noise_model": "gaussian_iid_v1",
            "benchmark_decision_noise_std": "1e-06",
            "benchmark_decision_noise_seed": "20260515",
            "benchmark_value_noise_model": "gaussian_iid_v1",
            "benchmark_value_noise_std": "1e-06",
            "benchmark_value_noise_seed": "20260514",
        }
    ]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_benchmark_value_noise_payload(tmp_path / "root", record_id)

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        require_decision_noise_handled=True,
    )

    assert summary["decision_noise_violation_count"] >= 1
    assert summary["decision_noise_status_counts"] == {"missing_benchmark_decision_noise_top_level_payload": 1}
    assert summary["decision_noise_violations"][0]["violation"] == "decision_noise_required"


def test_check_generic_static_table_outputs_decision_noise_rejects_stale_sibling_artifact(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2__static_hea_qiskit_vqe"
    records = [
        {
            "record_id": record_id,
            "family": "hh",
            "case_id": "hh_L2",
            "algorithm_id": "static_hea_qiskit_vqe",
            "benchmark_decision_noise_model": "gaussian_iid_v1",
            "benchmark_decision_noise_std": "1e-06",
            "benchmark_decision_noise_seed": "20260515",
        }
    ]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_benchmark_decision_noise_payload(tmp_path / "root", record_id, status="unsupported")
    stale_rows = {
        "schema": "stale_rows_v1",
        "rows": [
            {
                "status": "skipped_unsupported_decision_noise",
                "phase3_controller_called": False,
                "shots_total": 0,
                "compiled_depth_total": 0,
                "compiled_count_2q_total": 0,
            }
        ],
    }
    stale_path = tmp_path / "root" / record_id / "result" / "rows.json"
    stale_path.write_text(json.dumps(stale_rows) + "\n", encoding="utf-8")

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        require_decision_noise_handled=True,
    )

    assert summary["decision_noise_violation_count"] >= 1
    assert summary["decision_noise_status_counts"] == {"decision_artifact_missing_benchmark_decision_noise_top_level_payload": 1}
    assert "rows.json" in summary["decision_noise_violations"][0]["decision_noise_detail"]["benchmark_decision_artifacts"]["artifact"]


def test_check_generic_static_table_outputs_benchmark_value_noise_rejects_malformed_second_row(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2__static_hea_qiskit_vqe"
    records = [
        {
            "record_id": record_id,
            "family": "hh",
            "case_id": "hh_L2",
            "algorithm_id": "static_hea_qiskit_vqe",
            "benchmark_value_noise_model": "gaussian_iid_v1",
            "benchmark_value_noise_std": "1e-06",
            "benchmark_value_noise_seed": "20260514",
        }
    ]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    result_dir = tmp_path / "root" / record_id / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    def row(*, idx: int, enabled: bool = True) -> dict[str, object]:
        baseline = -1.0 - idx
        exact = baseline - 0.25
        noise_draw = 2e-6
        energy = baseline + noise_draw
        return {
            "status": "ok",
            "energy": energy,
            "energy_pre_benchmark_value_noise": baseline,
            "benchmark_value_noise_energy_ideal": baseline,
            "exact_energy": exact,
            "delta_E_abs": abs(energy - exact),
            "abs_delta_e": abs(energy - exact),
            "count_2q": 1,
            "circuit_depth": 2,
            "phase3_controller_called": False,
            "shots_total": 100,
            "compiled_depth_total": 2,
            "compiled_count_2q_total": 1,
            "benchmark_value_noise_status": "ok",
            "benchmark_value_noise": {
                "enabled": enabled,
                "model": "gaussian_iid_v1",
                "std": 1e-6,
                "seed": 20260514,
                "semantic": "post_static_result_value_noise_not_physical_shots",
                "noise_draw": noise_draw,
                "energy_pre_benchmark_value_noise": baseline,
                "benchmark_value_noise_energy_ideal": baseline,
                "physical_shots_unchanged": True,
                "scope": {"kind": "static_benchmark_row", "family": "hh", "case_id": "hh_L2", "algorithm_id": "static_hea_qiskit_vqe", "row": idx},
                "applied": enabled,
                "status": "ok",
            },
        }

    payload = {
        "status": "completed",
        "benchmark_value_noise_status": "ok",
        "benchmark_value_noise": {
            "enabled": True,
            "model": "gaussian_iid_v1",
            "std": 1e-6,
            "seed": 20260514,
            "semantic": "post_static_result_value_noise_not_physical_shots",
            "physical_shots_unchanged": True,
            "status": "ok",
            "row_target_count": 2,
            "applied_row_count": 2,
        },
        "rows": [row(idx=0), row(idx=1, enabled=False)],
    }
    (result_dir / "generic_static_single.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        require_value_noise_applied=True,
    )

    assert summary["value_noise_violation_count"] == 2
    assert summary["value_noise_status_counts"] == {"benchmark_row_value_noise_not_enabled": 1}
    assert summary["value_noise_violations"][0]["value_noise_status"] == "benchmark_row_value_noise_not_enabled"


def test_check_generic_static_table_outputs_benchmark_value_noise_rejects_stale_sibling_artifact(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2__static_hea_qiskit_vqe"
    records = [
        {
            "record_id": record_id,
            "family": "hh",
            "case_id": "hh_L2",
            "algorithm_id": "static_hea_qiskit_vqe",
            "benchmark_value_noise_model": "gaussian_iid_v1",
            "benchmark_value_noise_std": "1e-06",
            "benchmark_value_noise_seed": "20260514",
        }
    ]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_benchmark_value_noise_payload(tmp_path / "root", record_id)
    stale_rows = {
        "schema": "stale_rows_v1",
        "rows": [
            {
                "status": "ok",
                "energy": -1.0,
                "exact_energy": -1.25,
                "abs_delta_e": 0.25,
                "phase3_controller_called": False,
                "shots_total": 100,
                "compiled_depth_total": 2,
                "compiled_count_2q_total": 1,
            }
        ],
    }
    stale_path = tmp_path / "root" / record_id / "result" / "rows.json"
    stale_path.write_text(json.dumps(stale_rows) + "\n", encoding="utf-8")

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        require_value_noise_applied=True,
    )

    assert summary["value_noise_violation_count"] == 2
    assert summary["value_noise_status_counts"] == {"benchmark_artifact_missing_benchmark_value_noise_top_level_payload": 1}
    assert "rows.json" in summary["value_noise_violations"][0]["value_noise_detail"]["benchmark_artifacts"]["artifact"]


def test_check_generic_static_table_outputs_benchmark_value_noise_rejects_stale_hh_specific_sibling_artifact(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2__static_hea_qiskit_vqe"
    records = [
        {
            "record_id": record_id,
            "family": "hh",
            "case_id": "hh_L2",
            "algorithm_id": "static_hea_qiskit_vqe",
            "benchmark_value_noise_model": "gaussian_iid_v1",
            "benchmark_value_noise_std": "1e-06",
            "benchmark_value_noise_seed": "20260514",
        }
    ]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_benchmark_value_noise_payload(tmp_path / "root", record_id)
    stale_rows = {
        "schema": "stale_hh_rows_v1",
        "rows": [
            {
                "status": "ok",
                "energy": -1.0,
                "exact_energy": -1.25,
                "abs_delta_e": 0.25,
                "phase3_controller_called": False,
                "shots_total": 100,
                "compiled_depth_total": 2,
                "compiled_count_2q_total": 1,
            }
        ],
    }
    stale_path = tmp_path / "root" / record_id / "result" / "hh_static_benchmark_rows.json"
    stale_path.write_text(json.dumps(stale_rows) + "\n", encoding="utf-8")

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        require_value_noise_applied=True,
    )

    assert summary["value_noise_violation_count"] == 2
    assert summary["value_noise_status_counts"] == {"benchmark_artifact_missing_benchmark_value_noise_top_level_payload": 1}
    assert "hh_static_benchmark_rows.json" in summary["value_noise_violations"][0]["value_noise_detail"]["benchmark_artifacts"]["artifact"]



def test_check_generic_static_table_outputs_benchmark_value_noise_rejects_top_level_count_mismatch(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2__static_hea_qiskit_vqe"
    records = [
        {
            "record_id": record_id,
            "family": "hh",
            "case_id": "hh_L2",
            "algorithm_id": "static_hea_qiskit_vqe",
            "benchmark_value_noise_model": "gaussian_iid_v1",
            "benchmark_value_noise_std": "1e-06",
            "benchmark_value_noise_seed": "20260514",
        }
    ]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_benchmark_value_noise_payload(tmp_path / "root", record_id)
    payload_path = tmp_path / "root" / record_id / "result" / "generic_static_single.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["benchmark_value_noise"]["row_target_count"] = 2
    payload_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        require_value_noise_applied=True,
    )

    assert summary["value_noise_violation_count"] == 2
    assert summary["value_noise_status_counts"] == {"benchmark_top_level_count_mismatch": 1}


def test_check_generic_static_table_outputs_require_both_phase3_and_benchmark_value_noise(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2__static_family_native_adapt_phase3"
    records = [
        {
            "record_id": record_id,
            "family": "hh",
            "case_id": "hh_L2",
            "algorithm_id": "static_family_native_adapt_phase3",
            "phase3_oracle_gradient_mode": "aer_density_matrix",
            "phase3_oracle_value_noise_model": "gaussian_iid_v1",
            "phase3_oracle_value_noise_std": "1e-06",
            "phase3_oracle_value_noise_seed": "20260514",
            "benchmark_value_noise_model": "gaussian_iid_v1",
            "benchmark_value_noise_std": "1e-06",
            "benchmark_value_noise_seed": "20260514",
        }
    ]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_phase3_value_noise_payload(tmp_path / "root", record_id)
    payload_path = tmp_path / "root" / record_id / "result" / "generic_static_single.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    energy_ideal = -1.0
    noise_draw = 2e-6
    energy = energy_ideal + noise_draw
    row_noise = {
        "enabled": True,
        "model": "gaussian_iid_v1",
        "std": 1e-6,
        "seed": 20260514,
        "semantic": "post_static_result_value_noise_not_physical_shots",
        "noise_draw": noise_draw,
        "energy_pre_benchmark_value_noise": energy_ideal,
        "benchmark_value_noise_energy_ideal": energy_ideal,
        "physical_shots_unchanged": True,
        "scope": {"kind": "static_benchmark_row", "family": "hh", "case_id": "hh_L2", "algorithm_id": "static_family_native_adapt_phase3"},
        "applied": True,
        "status": "ok",
    }
    payload_noise = {
        "enabled": True,
        "model": "gaussian_iid_v1",
        "std": 1e-6,
        "seed": 20260514,
        "semantic": "post_static_result_value_noise_not_physical_shots",
        "physical_shots_unchanged": True,
        "scope": "generic_static_benchmark_dispatch_payload",
        "status": "ok",
        "status_counts": {"ok": 1},
        "row_target_count": 1,
        "applied_row_count": 1,
    }
    payload["benchmark_value_noise_status"] = "ok"
    payload["benchmark_value_noise"] = payload_noise
    payload["result"].update(
        {
            "energy": energy,
            "energy_ideal": -999.0,
            "energy_pre_benchmark_value_noise": energy_ideal,
            "benchmark_value_noise_energy_ideal": energy_ideal,
            "exact_energy": -1.25,
            "exact_gs_energy": -1.25,
            "delta_E_abs": abs(energy - (-1.25)),
            "abs_delta_e": abs(energy - (-1.25)),
            "benchmark_value_noise_status": "ok",
            "benchmark_value_noise": row_noise,
            "phase3_controller_called": True,
        }
    )
    payload_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        require_value_noise_applied=True,
    )

    assert summary["value_noise_violation_count"] == 0
    assert summary["value_noise_status_counts"] == {"ok": 1}


def test_check_generic_static_table_outputs_allow_incomplete_does_not_hide_malformed_benchmark_value_noise(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2__static_hea_qiskit_vqe"
    records_path = tmp_path / "records.tsv"
    _write_records(
        records_path,
        [
            {
                "record_id": record_id,
                "family": "hh",
                "case_id": "hh_L2",
                "algorithm_id": "static_hea_qiskit_vqe",
                "benchmark_value_noise_model": "gaussian_iid_v1",
                "benchmark_value_noise_std": "1e-06",
                "benchmark_value_noise_seed": "20260514",
            }
        ],
    )
    _write_payload(tmp_path / "root", record_id)

    exit_code = check.main(
        [
            "--records",
            str(records_path),
            "--root",
            str(tmp_path / "root"),
            "--summary",
            str(tmp_path / "summary.json"),
            "--allow-incomplete",
            "--require-value-noise-applied",
        ]
    )

    assert exit_code == 1
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["value_noise_violation_count"] >= 1
    assert summary["value_noise_status_counts"] == {"missing_benchmark_value_noise_top_level_payload": 1}


def test_check_generic_static_table_outputs_require_value_noise_fails_when_no_rows_apply(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2__static_hea_qiskit_vqe"
    records = [{"record_id": record_id, "family": "hh", "case_id": "hh_L2", "algorithm_id": "static_hea_qiskit_vqe"}]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_payload(tmp_path / "root", record_id)

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        require_value_noise_applied=True,
    )

    assert summary["value_noise_violation_count"] == 1
    assert summary["value_noise_status_counts"] == {"not_requested": 1}
    assert summary["value_noise_violations"][0]["violation"] == "value_noise_required_no_ok_rows"


def test_check_generic_static_table_outputs_allow_incomplete_still_fails_metric_requirements(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2__static_hea_qiskit_vqe"
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, [{"record_id": record_id, "family": "hh", "case_id": "hh_L2", "algorithm_id": "static_hea_qiskit_vqe"}])
    _write_payload(tmp_path / "root", record_id)

    exit_code = check.main(
        [
            "--records",
            str(records_path),
            "--root",
            str(tmp_path / "root"),
            "--summary",
            str(tmp_path / "summary.json"),
            "--allow-incomplete",
            "--require-value-noise-applied",
        ]
    )

    assert exit_code == 1


def test_check_generic_static_table_outputs_value_noise_rejects_non_phase3_static_record(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2__static_hea_qiskit_vqe"
    records = [
        {
            "record_id": record_id,
            "family": "hh",
            "case_id": "hh_L2",
            "algorithm_id": "static_hea_qiskit_vqe",
            "phase3_oracle_gradient_mode": "aer_density_matrix",
            "phase3_oracle_value_noise_model": "gaussian_iid_v1",
            "phase3_oracle_value_noise_std": "1e-06",
        }
    ]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_payload(tmp_path / "root", record_id)

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        require_value_noise_applied=True,
    )

    assert summary["value_noise_violation_count"] == 2
    assert summary["value_noise_status_counts"] == {"requested_for_non_phase3_static_adapt": 1}
    assert {item["violation"] for item in summary["value_noise_violations"]} == {
        "value_noise_required",
        "value_noise_required_no_ok_rows",
    }


def test_check_generic_static_table_outputs_require_s_alg_rejects_missing_algorithmic_work_schema(tmp_path: Path) -> None:
    record_id = "static_table__bose_hubbard__bose_hubbard_L2__static_family_informed_vqe"
    records = [{"record_id": record_id, "family": "bose_hubbard", "case_id": "bose_hubbard_L2", "algorithm_id": "static_family_informed_vqe"}]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_payload(tmp_path / "root", record_id)
    _write_enrichment(tmp_path / "enrichment", record_id, s_status="ok", s_alg_status="ok")
    enrichment_path = tmp_path / "enrichment" / record_id / "result" / "generic_static_metric_enrichment.json"
    payload = json.loads(enrichment_path.read_text(encoding="utf-8"))
    payload.pop("metrics")
    enrichment_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        enrichment_root=tmp_path / "enrichment",
        require_s_alg=True,
    )

    assert summary["enrichment_violation_count"] == 1
    assert summary["s_alg_status_counts"] == {"missing_algorithmic_measurement_work": 1}
    assert summary["enrichment_violations"][0]["violation"] == "S_alg_required"


def test_check_generic_static_table_outputs_require_s_alg_passes_and_flags_missing(tmp_path: Path) -> None:
    ok_id = "static_table__bose_hubbard__bose_hubbard_L2__static_family_informed_vqe"
    bad_id = "static_table__bose_hubbard__bose_hubbard_L2__static_qiskit_adapt_vqe"
    records = [
        {"record_id": ok_id, "family": "bose_hubbard", "case_id": "bose_hubbard_L2", "algorithm_id": "static_family_informed_vqe"},
        {"record_id": bad_id, "family": "bose_hubbard", "case_id": "bose_hubbard_L2", "algorithm_id": "static_qiskit_adapt_vqe"},
    ]
    records_path = tmp_path / "records.tsv"
    _write_records(records_path, records)
    _write_payload(tmp_path / "root", ok_id)
    _write_payload(tmp_path / "root", bad_id)
    _write_enrichment(tmp_path / "enrichment", ok_id, s_status="ok", s_alg_status="ok")
    _write_enrichment(tmp_path / "enrichment", bad_id, s_status="ok", s_alg_status="legacy_proxy_not_event_ledger")

    summary = check.validate_outputs(
        records_path=records_path,
        root=tmp_path / "root",
        enrichment_root=tmp_path / "enrichment",
        require_s_alg=True,
    )

    assert summary["enrichment_violation_count"] == 1
    assert summary["s_alg_status_counts"] == {"legacy_proxy_not_event_ledger": 1, "ok": 1}
    assert summary["enrichment_violations"][0]["violation"] == "S_alg_required"
