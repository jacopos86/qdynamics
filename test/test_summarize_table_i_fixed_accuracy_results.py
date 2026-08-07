#!/usr/bin/env python3
"""Tests for calibrated fixed-accuracy Table-I summaries."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from pipelines.exact_bench import summarize_table_i_fixed_accuracy_results as summary


def _write_records(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("record_id", "family", "case_id", "algorithm_id"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_payload(root: Path, record_id: str, result: dict) -> None:
    out = root / record_id / "result"
    out.mkdir(parents=True, exist_ok=True)
    (out / "generic_static_single.json").write_text(
        json.dumps({"status": "completed", "result": result, "rows": [result]}) + "\n",
        encoding="utf-8",
    )


def _write_enrichment(root: Path, record_id: str, payload: dict) -> None:
    out = root / record_id / "result"
    out.mkdir(parents=True, exist_ok=True)
    body = {
        "schema": "generic_static_metric_enrichment_v1",
        "record_id": record_id,
        "status": "completed",
        "row_updates": {},
        "metric_statuses": {},
    }
    body.update(payload)
    (out / "generic_static_metric_enrichment.json").write_text(json.dumps(body) + "\n", encoding="utf-8")


def _hash_payload_without_snake_sidecar(record_result: dict) -> str:
    def _strip(value):  # noqa: ANN001, ANN202
        if isinstance(value, dict):
            return {
                key: _strip(item)
                for key, item in value.items()
                if key
                not in {
                    "paper_i_first_crossing_compiled_cost",
                    "snake_first_crossing_compiled_cost",
                    "source_result_sha256",
                }
            }
        if isinstance(value, list):
            return [_strip(item) for item in value]
        return value

    payload = {"status": "completed", "result": record_result, "rows": [record_result]}
    body = json.dumps(_strip(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def test_default_thresholds_are_clean_paper_i_tau_phys() -> None:
    assert summary.DEFAULT_THRESHOLDS == (2e-4,)
    assert summary._parse_thresholds(None) == (2e-4,)
    assert summary._parse_thresholds("1e-6,1e-8") == (1e-6, 1e-8)


def test_threshold_s_var_entry_rejects_terminal_scope() -> None:
    value, status, _components = summary._s_var_from_threshold_entry(
        {"status": "ok", "S_var": 12.0, "state_scope": "terminal_final_state"}
    )

    assert value is None
    assert status == "missing_threshold_state"


def test_fixed_accuracy_summary_default_clean_threshold_uses_benchmark_first_hits_2e4(tmp_path: Path) -> None:
    record_id = "static_table__hubbard__hubbard_L2_clean_weak__static_pos_geo_adapt_vqe"
    records_path = tmp_path / "records.tsv"
    root = tmp_path / "raw"
    output_dir = tmp_path / "out"
    _write_records(
        records_path,
        [{"record_id": record_id, "family": "hubbard", "case_id": "hubbard_L2_clean_weak", "algorithm_id": "static_pos_geo_adapt_vqe"}],
    )
    _write_payload(
        root,
        record_id,
        {
            "abs_delta_e": 9e-5,
            "benchmark_first_hits": {
                "2e-4": {
                    "source": "native_adaptive_iteration",
                    "first_hit_semantics": "native_first_crossing_after_adapt_iteration",
                    "abs_delta_e": 9e-5,
                    "selected_operator_count": 2,
                    "energy_eval_count_proxy": 11,
                    "gradient_operator_probe_count_proxy": 13,
                    "metric_operator_probe_count_proxy": 17,
                    "compiled_count_2q_total": 8,
                    "compiled_depth_2q_total": 6,
                    "compiled_depth_total": 44,
                    "compiled_circuit_stats_status": "ok",
                    "first_hit_cost_source_kind": "qiskit_compiled_first_hit_ansatz_circuit",
                }
            },
        },
    )

    rc = summary.main(["--records", str(records_path), "--root", str(root), "--output-dir", str(output_dir)])

    assert rc == 0
    loaded = json.loads((output_dir / "table_i_fixed_accuracy_calibrated_summary.json").read_text(encoding="utf-8"))
    row = loaded["row_results"][0]
    assert loaded["thresholds"] == [0.0002]
    assert loaded["target_profile"] == "paper_i_phys_v1"
    assert loaded["threshold_policy"] == "clean_paper_i_tau_phys"
    assert row["threshold_source"] == "benchmark_first_hits[2e-4]"
    assert row["threshold_status"] == "ok_native_first_hit"
    assert row["count_2q"] == 8.0


def test_fixed_accuracy_summary_uses_native_hits_and_terminal_fixed_only(tmp_path: Path) -> None:
    fixed_id = "static_table__hubbard__hubbard_L2__static_family_informed_vqe"
    adapt_id = "static_table__hubbard__hubbard_L2__static_pos_geo_adapt_vqe"
    records = [
        {"record_id": fixed_id, "family": "hubbard", "case_id": "hubbard_L2", "algorithm_id": "static_family_informed_vqe"},
        {"record_id": adapt_id, "family": "hubbard", "case_id": "hubbard_L2", "algorithm_id": "static_pos_geo_adapt_vqe"},
    ]
    records_path = tmp_path / "records.tsv"
    root = tmp_path / "raw"
    output_dir = tmp_path / "out"
    _write_records(records_path, records)
    _write_payload(
        root,
        fixed_id,
        {
            "abs_delta_e": 5e-7,
            "energy_eval_count_proxy": 9,
            "compiled_count_2q_total": 4,
            "compiled_depth_2q_total": 4,
            "compiled_depth_total": 20,
            "compiled_circuit_stats_status": "ok",
            "first_hit_cost_source_kind": "qiskit_compiled_terminal_only_fixed_ansatz",
        },
    )
    native_hit = {
        "source": "native_adaptive_iteration",
        "first_hit_semantics": "native_first_crossing_after_adapt_iteration",
        "abs_delta_e": 4e-7,
        "selected_operator_count": 2,
        "energy_eval_count_proxy": 11,
        "gradient_operator_probe_count_proxy": 13,
        "metric_operator_probe_count_proxy": 17,
        "compiled_count_2q_total": 8,
        "compiled_depth_2q_total": 8,
        "compiled_depth_total": 44,
        "compiled_circuit_stats_status": "ok",
        "first_hit_cost_source_kind": "qiskit_compiled_first_hit_ansatz_circuit",
    }
    _write_payload(
        root,
        adapt_id,
        {
            "abs_delta_e": 1e-9,
            "selected_operator_count": 3,
            "energy_eval_count_proxy": 99,
            "gradient_operator_probe_count_proxy": 99,
            "metric_operator_probe_count_proxy": 99,
            "first_hit_1e6": native_hit,
        },
    )

    rc = summary.main([
        "--records",
        str(records_path),
        "--root",
        str(root),
        "--output-dir",
        str(output_dir),
        "--thresholds",
        "1e-6",
    ])

    assert rc == 0
    loaded = json.loads((output_dir / "table_i_fixed_accuracy_calibrated_summary.json").read_text(encoding="utf-8"))
    by_record = {row["record_id"]: row for row in loaded["row_results"]}
    assert by_record[fixed_id]["threshold_status"] == "ok_terminal_only_method"
    assert by_record[fixed_id]["S_alg"] == 9.0
    assert by_record[fixed_id]["S_norm"] is None
    assert by_record[adapt_id]["threshold_status"] == "ok_native_first_hit"
    assert by_record[adapt_id]["S_alg"] == 41.0
    assert by_record[adapt_id]["S_norm"] is None
    assert by_record[adapt_id]["count_2q"] == 8.0
    agg = [row for row in loaded["aggregate_rows"] if row["class"] == "fermionic" and row["algorithm_id"] == "static_pos_geo_adapt_vqe"][0]
    assert agg["hit_count"] == 1
    assert agg["S_alg_mean"] == 41.0
    assert agg["S_norm_mean"] is None
    assert agg["N_metric_mean_support_only"] == 17.0


def test_fixed_accuracy_summary_carries_threshold_local_s_var_from_enrichment_only(tmp_path: Path) -> None:
    record_id = "static_table__hubbard__hubbard_L2__static_pos_geo_adapt_vqe"
    records_path = tmp_path / "records.tsv"
    root = tmp_path / "raw"
    enrichment_root = tmp_path / "enrichment"
    output_dir = tmp_path / "out"
    _write_records(
        records_path,
        [{"record_id": record_id, "family": "hubbard", "case_id": "hubbard_L2", "algorithm_id": "static_pos_geo_adapt_vqe"}],
    )
    _write_payload(
        root,
        record_id,
        {
            "abs_delta_e": 1e-9,
            "selected_operator_count": 3,
            "energy_eval_count_proxy": 99,
            "gradient_operator_probe_count_proxy": 99,
            "metric_operator_probe_count_proxy": 99,
            "first_hit_1e6": {
                "source": "native_adaptive_iteration",
                "first_hit_semantics": "native_first_crossing_after_adapt_iteration",
                "abs_delta_e": 4e-7,
                "selected_operator_count": 2,
                "energy_eval_count_proxy": 11,
                "gradient_operator_probe_count_proxy": 13,
                "metric_operator_probe_count_proxy": 17,
                "compiled_count_2q_total": 8,
                "compiled_depth_2q_total": 8,
                "compiled_depth_total": 44,
                "compiled_circuit_stats_status": "ok",
                "first_hit_cost_source_kind": "qiskit_compiled_first_hit_ansatz_circuit",
            },
        },
    )
    _write_enrichment(
        enrichment_root,
        record_id,
        {
            "row_updates": {"S_var": 99.0},
            "metric_statuses": {"S_phys_var": "ok"},
            "threshold_metrics": {
                "first_hit_abs_delta_e_le_1e_6": {
                    "status": "ok",
                    "S_var": 23.0,
                    "provenance": "threshold_first_hit_state_sidecar",
                    "S_var_components": {"S_var_grad": 13.0},
                }
            },
        },
    )

    rc = summary.main([
        "--records",
        str(records_path),
        "--root",
        str(root),
        "--enrichment-root",
        str(enrichment_root),
        "--output-dir",
        str(output_dir),
        "--thresholds",
        "1e-6",
    ])

    assert rc == 0
    loaded = json.loads((output_dir / "table_i_fixed_accuracy_calibrated_summary.json").read_text(encoding="utf-8"))
    row = loaded["row_results"][0]
    assert row["threshold_status"] == "ok_native_first_hit"
    assert row["S_alg"] == 41.0
    assert row["S_var"] == 23.0
    assert row["S_var_status"] == "ok"
    assert row["S_var_provenance"] == "threshold_first_hit_state_sidecar"
    assert row["terminal_S_var_upper_bound"] is None
    agg = loaded["aggregate_rows"][0]
    assert agg["S_alg_mean"] == 41.0
    assert agg["S_var_mean"] == 23.0
    assert agg["S_var_available_n"] == 1
    assert agg["S_var_status_counts"] == {"ok": 1}


def test_fixed_accuracy_summary_marks_terminal_s_var_as_upper_bound_not_hit_cost(tmp_path: Path) -> None:
    record_id = "static_table__hubbard__hubbard_L2__static_pos_geo_adapt_vqe"
    records_path = tmp_path / "records.tsv"
    root = tmp_path / "raw"
    enrichment_root = tmp_path / "enrichment"
    output_dir = tmp_path / "out"
    _write_records(
        records_path,
        [{"record_id": record_id, "family": "hubbard", "case_id": "hubbard_L2", "algorithm_id": "static_pos_geo_adapt_vqe"}],
    )
    _write_payload(
        root,
        record_id,
        {
            "abs_delta_e": 5e-7,
            "first_hit_1e6": {
                "source": "terminal_row_fallback",
                "first_hit_semantics": "terminal_upper_bound_not_native_first_hit",
                "abs_delta_e": 5e-7,
                "selected_operator_count": 2,
                "energy_eval_count_proxy": 11,
                "gradient_operator_probe_count_proxy": 13,
                "metric_operator_probe_count_proxy": 17,
            },
        },
    )
    _write_enrichment(
        enrichment_root,
        record_id,
        {
            "row_updates": {"S_var": 77.0},
            "metric_statuses": {"S_phys_var": "ok"},
        },
    )

    rc = summary.main([
        "--records",
        str(records_path),
        "--root",
        str(root),
        "--enrichment-root",
        str(enrichment_root),
        "--output-dir",
        str(output_dir),
        "--thresholds",
        "1e-6",
    ])

    assert rc == 0
    loaded = json.loads((output_dir / "table_i_fixed_accuracy_calibrated_summary.json").read_text(encoding="utf-8"))
    row = loaded["row_results"][0]
    assert row["threshold_status"] == "terminal_upper_bound_missing_native_first_hit"
    assert row["cost_included"] is False
    assert row["S_var"] is None
    assert row["S_var_status"] == "missing_threshold_state"
    assert row["terminal_S_var_upper_bound"] == 77.0
    assert row["terminal_S_var_upper_bound_provenance"] == "terminal_enrichment_S_var"
    agg = loaded["aggregate_rows"][0]
    assert agg["S_var_mean"] is None
    assert agg["terminal_S_var_upper_bound_mean"] == 77.0
    assert agg["terminal_S_var_upper_bound_count"] == 1


def test_fixed_accuracy_summary_includes_final_ansatz_cost_for_not_reached_adaptive_row(tmp_path: Path) -> None:
    record_id = "static_table__hubbard__hubbard_L2__static_pos_geo_adapt_vqe"
    records_path = tmp_path / "records.tsv"
    root = tmp_path / "raw"
    output_dir = tmp_path / "out"
    _write_records(
        records_path,
        [{"record_id": record_id, "family": "hubbard", "case_id": "hubbard_L2", "algorithm_id": "static_pos_geo_adapt_vqe"}],
    )
    _write_payload(
        root,
        record_id,
        {
            "abs_delta_e": 5e-4,
            "source": "terminal_row_fallback",
            "first_hit_semantics": "terminal_upper_bound_not_native_first_hit",
            "selected_operator_count": 2,
            "energy_eval_count_proxy": 11,
            "gradient_operator_probe_count_proxy": 13,
            "metric_operator_probe_count_proxy": 17,
            "compiled_count_2q_total": 8,
            "compiled_depth_2q_total": 8,
            "compiled_depth_total": 44,
            "compiled_circuit_stats_status": "ok",
            "first_hit_cost_source_kind": "qiskit_compiled_final_ansatz_circuit",
            "compiled_resource_qiskit_validated": True,
        },
    )

    rc = summary.main([
        "--records",
        str(records_path),
        "--root",
        str(root),
        "--output-dir",
        str(output_dir),
        "--thresholds",
        "1e-6",
    ])

    assert rc == 0
    loaded = json.loads((output_dir / "table_i_fixed_accuracy_calibrated_summary.json").read_text(encoding="utf-8"))
    row = loaded["row_results"][0]
    assert row["threshold_status"] == "not_reached_final_ansatz"
    assert row["cost_included"] is True
    assert row["S_alg"] == 41.0
    assert row["count_2q"] == 8.0
    assert row["method_cost_semantics"] == "adaptive_qiskit_compiled_first_hit_or_final_ansatz"


def test_fixed_accuracy_summary_promotes_enriched_final_ansatz_cost_for_adaptive_miss(tmp_path: Path) -> None:
    record_id = "static_table__hubbard__hubbard_L2__static_pos_geo_adapt_vqe"
    records_path = tmp_path / "records.tsv"
    root = tmp_path / "raw"
    enrichment_root = tmp_path / "enrich"
    output_dir = tmp_path / "out"
    _write_records(
        records_path,
        [{"record_id": record_id, "family": "hubbard", "case_id": "hubbard_L2", "algorithm_id": "static_pos_geo_adapt_vqe"}],
    )
    _write_payload(
        root,
        record_id,
        {
            "abs_delta_e": 5e-4,
            "source": "terminal_row_fallback",
            "first_hit_semantics": "terminal_upper_bound_not_native_first_hit",
            "selected_operator_count": 2,
            "energy_eval_count_proxy": 11,
            "gradient_operator_probe_count_proxy": 13,
            "metric_operator_probe_count_proxy": 17,
            "compiled_circuit_stats_status": "not_reportable_adaptive_terminal_cost_proxy_demoted",
        },
    )
    _write_enrichment(
        enrichment_root,
        record_id,
        {
            "row_updates": {
                "compiled_count_2q_total": 8,
                "compiled_depth_2q_total": 4,
                "compiled_depth_total": 54,
                "infidelity_same": 0.25,
            },
            "metric_statuses": {
                "compiled_count_2q_total": "ok",
                "compiled_depth_2q_total": "ok",
                "compiled_depth_total": "ok",
                "infidelity_same": "ok",
            },
        },
    )

    rc = summary.main([
        "--records",
        str(records_path),
        "--root",
        str(root),
        "--enrichment-root",
        str(enrichment_root),
        "--output-dir",
        str(output_dir),
        "--thresholds",
        "1e-6",
    ])

    assert rc == 0
    loaded = json.loads((output_dir / "table_i_fixed_accuracy_calibrated_summary.json").read_text(encoding="utf-8"))
    row = loaded["row_results"][0]
    assert row["threshold_status"] == "not_reached_final_ansatz"
    assert row["cost_included"] is True
    assert row["count_2q"] == 8.0
    assert row["depth_2q"] == 4.0
    assert row["circuit_depth"] == 54.0
    assert row["first_hit_cost_source_kind"] == "qiskit_compiled_final_ansatz_circuit"
    assert row["infidelity"] == 0.25
    assert row["fidelity"] == 0.75


def test_fixed_accuracy_summary_preserves_enriched_fixed_ansatz_source_kind(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2_nph2_three_model_strong_weak__static_family_informed_vqe"
    records_path = tmp_path / "records.tsv"
    root = tmp_path / "raw"
    enrichment_root = tmp_path / "enrich"
    output_dir = tmp_path / "out"
    _write_records(
        records_path,
        [{"record_id": record_id, "family": "hh", "case_id": "hh_L2_nph2_three_model_strong_weak", "algorithm_id": "static_family_informed_vqe"}],
    )
    _write_payload(
        root,
        record_id,
        {
            "abs_delta_e": 5e-4,
            "energy_eval_count_proxy": 11,
            "compiled_circuit_stats_status": "not_reportable_missing_terminal_cost",
        },
    )
    _write_enrichment(
        enrichment_root,
        record_id,
        {
            "row_updates": {
                "compiled_count_2q_total": 14,
                "compiled_depth_2q_total": 9,
                "compiled_depth_total": 39,
                "compiled_resource_source_kind": "qiskit_compiled_terminal_only_fixed_ansatz",
                "first_hit_cost_source_kind": "qiskit_compiled_terminal_only_fixed_ansatz",
                "infidelity_same": 0.5,
            },
            "metric_statuses": {
                "compiled_count_2q_total": "ok",
                "compiled_depth_2q_total": "ok",
                "compiled_depth_total": "ok",
                "infidelity_same": "ok",
            },
        },
    )

    rc = summary.main([
        "--records",
        str(records_path),
        "--root",
        str(root),
        "--enrichment-root",
        str(enrichment_root),
        "--output-dir",
        str(output_dir),
        "--thresholds",
        "1e-6",
    ])

    assert rc == 0
    loaded = json.loads((output_dir / "table_i_fixed_accuracy_calibrated_summary.json").read_text(encoding="utf-8"))
    row = loaded["row_results"][0]
    assert row["cost_included"] is True
    assert row["count_2q"] == 14.0
    assert row["first_hit_cost_source_kind"] == "qiskit_compiled_terminal_only_fixed_ansatz"
    assert row["compiled_resource_validation_status"] == "ok"


def test_fixed_accuracy_summary_does_not_promote_enriched_final_cost_for_adaptive_hit(tmp_path: Path) -> None:
    record_id = "static_table__hubbard__hubbard_L2__static_pos_geo_adapt_vqe"
    records_path = tmp_path / "records.tsv"
    root = tmp_path / "raw"
    enrichment_root = tmp_path / "enrich"
    output_dir = tmp_path / "out"
    _write_records(
        records_path,
        [{"record_id": record_id, "family": "hubbard", "case_id": "hubbard_L2", "algorithm_id": "static_pos_geo_adapt_vqe"}],
    )
    _write_payload(
        root,
        record_id,
        {
            "abs_delta_e": 5e-7,
            "source": "terminal_row_fallback",
            "first_hit_semantics": "terminal_upper_bound_not_native_first_hit",
            "selected_operator_count": 2,
            "energy_eval_count_proxy": 11,
            "gradient_operator_probe_count_proxy": 13,
            "metric_operator_probe_count_proxy": 17,
        },
    )
    _write_enrichment(
        enrichment_root,
        record_id,
        {
            "row_updates": {
                "compiled_count_2q_total": 8,
                "compiled_depth_2q_total": 4,
                "compiled_depth_total": 54,
            },
            "metric_statuses": {
                "compiled_count_2q_total": "ok",
                "compiled_depth_2q_total": "ok",
                "compiled_depth_total": "ok",
            },
        },
    )

    rc = summary.main([
        "--records",
        str(records_path),
        "--root",
        str(root),
        "--enrichment-root",
        str(enrichment_root),
        "--output-dir",
        str(output_dir),
        "--thresholds",
        "1e-6",
    ])

    assert rc == 0
    loaded = json.loads((output_dir / "table_i_fixed_accuracy_calibrated_summary.json").read_text(encoding="utf-8"))
    row = loaded["row_results"][0]
    assert row["threshold_status"] == "terminal_upper_bound_missing_native_first_hit"
    assert row["cost_included"] is False
    assert row["count_2q"] is None


def test_fixed_accuracy_summary_does_not_backfill_snake_sidecar_missing_d2q_from_terminal_row(tmp_path: Path) -> None:
    record_id = "static_table__hh__hh_L2_clean_weak__static_family_native_adapt_phase3"
    records_path = tmp_path / "records.tsv"
    root = tmp_path / "raw"
    output_dir = tmp_path / "out"
    _write_records(
        records_path,
        [{"record_id": record_id, "family": "hh", "case_id": "hh_L2_clean_weak", "algorithm_id": "static_family_native_adapt_phase3"}],
    )
    snake_result = {
        "abs_delta_e": 1e-4,
        "compiled_depth_2q_total": 999,
        "paper_i_first_crossing": {
            "schema": "paper_i_first_crossing_v1",
            "status": "reached",
            "reached": True,
            "tau_phys": 2e-4,
            "benchmark_id": "hh_L2_clean_weak",
            "history_position_tau": 3,
            "primary_error_at_crossing": 1e-4,
        },
    }
    source_hash = _hash_payload_without_snake_sidecar(snake_result)
    snake_result["source_result_sha256"] = source_hash
    snake_result["paper_i_first_crossing_compiled_cost"] = {
        "schema": "snake_first_crossing_compiled_cost_v1",
        "benchmark_id": "hh_L2_clean_weak",
        "history_position_tau": 3,
        "current_target_threshold": 2e-4,
        "primary_error_at_crossing": 1e-4,
        "compiled_count_2q_total": 64,
        "compiled_depth_total": 237,
        "compiled_circuit_stats_status": "ok",
        "first_hit_cost_source_kind": "snake_qiskit_compiled_first_hit_ansatz_circuit",
        "source_result_sha256": source_hash,
        "S_alg": 97,
    }
    _write_payload(root, record_id, snake_result)

    rc = summary.main(["--records", str(records_path), "--root", str(root), "--output-dir", str(output_dir)])

    assert rc == 0
    loaded = json.loads((output_dir / "table_i_fixed_accuracy_calibrated_summary.json").read_text(encoding="utf-8"))
    row = loaded["row_results"][0]
    assert row["threshold_status"] == "snake_audited_first_crossing_cost_invalid"
    assert row["cost_included"] is False
    assert row["cost_source"] is None
    assert row["compiled_resource_validation_reason"] == "compiled_cost_required_resource_missing"
    assert row["count_2q"] is None
    assert row["depth_2q"] is None
    assert row["circuit_depth"] is None


def test_fixed_accuracy_summary_marks_adaptive_fallback_as_upper_bound(tmp_path: Path) -> None:
    record_id = "static_table__hubbard__hubbard_L2__static_pos_geo_adapt_vqe"
    records_path = tmp_path / "records.tsv"
    root = tmp_path / "raw"
    output_dir = tmp_path / "out"
    _write_records(
        records_path,
        [{"record_id": record_id, "family": "hubbard", "case_id": "hubbard_L2", "algorithm_id": "static_pos_geo_adapt_vqe"}],
    )
    _write_payload(
        root,
        record_id,
        {
            "abs_delta_e": 5e-7,
            "first_hit_1e6": {
                "source": "terminal_row_fallback",
                "first_hit_semantics": "terminal_upper_bound_not_native_first_hit",
                "abs_delta_e": 5e-7,
                "selected_operator_count": 2,
                "energy_eval_count_proxy": 11,
                "gradient_operator_probe_count_proxy": 13,
                "metric_operator_probe_count_proxy": 17,
                "compiled_count_2q_total": 99,
                "compiled_depth_2q_total": 88,
                "compiled_depth_total": 77,
            },
        },
    )

    rc = summary.main([
        "--records",
        str(records_path),
        "--root",
        str(root),
        "--output-dir",
        str(output_dir),
        "--thresholds",
        "1e-6",
    ])

    assert rc == 0
    loaded = json.loads((output_dir / "table_i_fixed_accuracy_calibrated_summary.json").read_text(encoding="utf-8"))
    row = loaded["row_results"][0]
    assert row["threshold_status"] == "terminal_upper_bound_missing_native_first_hit"
    assert row["cost_included"] is False
    assert row["terminal_S_alg_upper_bound"] == 41.0
    assert row["count_2q"] is None
    assert row["depth_2q"] is None
    assert row["circuit_depth"] is None
    agg = loaded["aggregate_rows"][0]
    assert agg["hit_count"] == 0
    assert agg["terminal_upper_bound_count"] == 1
    assert agg["S_alg_mean"] is None
    assert agg["terminal_upper_bound_S_alg_mean"] == 41.0
