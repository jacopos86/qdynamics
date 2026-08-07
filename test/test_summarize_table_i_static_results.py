#!/usr/bin/env python3
"""Tests for Table-I static result aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pipelines.exact_bench import summarize_table_i_static_results as summarize


_FAMILY_CASES = {
    "hh": "hh_L2",
    "hubbard": "hubbard_L2",
    "ionic_hubbard": "ionic_hubbard_L2",
    "extended_hubbard": "extended_hubbard_L2",
    "ttprime_hubbard": "ttprime_hubbard_L2",
    "spinless_tv": "spinless_tv_L2",
    "spin_boson": "spin_boson_L1",
    "bose_hubbard": "bose_hubbard_L2",
    "harmonic_kerr_chain": "harmonic_kerr_chain_L2",
    "molecular_vibronic_h2": "molecular_vibronic_h2_L2",
}


def _record_id(family: str, case_id: str, algorithm_id: str = "static_family_informed_vqe") -> str:
    return f"static_table__{family}__{case_id}__{algorithm_id}"


def _write_records(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("record_id", "family", "case_id", "algorithm_id"), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_payload(
    root: Path,
    record_id: str,
    *,
    algorithm_id: str = "static_family_informed_vqe",
    status: str = "completed",
    result: dict | None = None,
) -> None:
    result_dir = root / record_id / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "generic_static_family_informed_vqe_v1",
        "status": status,
        "algorithm_id": algorithm_id,
        "result": result or {},
        "rows": [result or {}],
    }
    (result_dir / "generic_static_single.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_table_i_static_result_summary_aggregates_family_informed_by_class(tmp_path: Path) -> None:
    records: list[dict[str, str]] = []
    root = tmp_path / "root"
    for idx, (family, case_id) in enumerate(_FAMILY_CASES.items(), start=1):
        record_id = _record_id(family, case_id)
        records.append({"record_id": record_id, "family": family, "case_id": case_id, "algorithm_id": "static_family_informed_vqe"})
        _write_payload(
            root,
            record_id,
            result={
                "status": "ok",
                "method_id": "static_family_informed_vqe",
                "abs_delta_e": float(idx) * 1e-3,
                "compiled_count_2q_total": float(idx),
                "compiled_depth_total": float(idx + 10),
                "shots_total": float(idx + 100),
            },
        )
    records_path = tmp_path / "records.tsv"
    out = tmp_path / "out"
    _write_records(records_path, records)

    rc = summarize.main(["--records", str(records_path), "--root", str(root), "--output-dir", str(out)])

    assert rc == 0
    summary = json.loads((out / "table_i_static_results_summary.json").read_text(encoding="utf-8"))
    rows = {
        row["class"]: row
        for row in summary["aggregate_rows"]
        if row["algorithm_id"] == "static_family_informed_vqe"
    }
    assert rows["fermionic"]["n"] == 5
    assert rows["bosonic"]["n"] == 3
    assert rows["fermion-boson"]["n"] == 2
    assert rows["all averaged"]["n"] == 10
    assert "family-informed VQE" in (out / "table_i_static_claim_rows.tex").read_text(encoding="utf-8")


def test_table_i_static_result_summary_keeps_geo_qeb_and_full_meta_geo_labels_distinct(tmp_path: Path) -> None:
    root = tmp_path / "root"
    records = [
        {
            "record_id": _record_id("hubbard", "hubbard_L2", "static_geo_qubit_adapt_vqe"),
            "family": "hubbard",
            "case_id": "hubbard_L2",
            "algorithm_id": "static_geo_qubit_adapt_vqe",
        },
        {
            "record_id": _record_id("hubbard", "hubbard_L2", "static_geo_qeb_adapt_vqe"),
            "family": "hubbard",
            "case_id": "hubbard_L2",
            "algorithm_id": "static_geo_qeb_adapt_vqe",
        },
    ]
    for row in records:
        _write_payload(
            root,
            row["record_id"],
            algorithm_id=row["algorithm_id"],
            result={
                "status": "ok",
                "method_id": row["algorithm_id"],
                "abs_delta_e": 1e-3,
                "compiled_count_2q_total": 4,
                "compiled_depth_total": 6,
                "shots_total": 100,
            },
        )
    records_path = tmp_path / "records.tsv"
    out = tmp_path / "out"
    _write_records(records_path, records)

    rc = summarize.main(["--records", str(records_path), "--root", str(root), "--output-dir", str(out)])

    assert rc == 0
    summary = json.loads((out / "table_i_static_results_summary.json").read_text(encoding="utf-8"))
    labels = {row["algorithm_id"]: row["method"] for row in summary["aggregate_rows"]}
    assert labels["static_geo_qubit_adapt_vqe"] == "legacy geometry diagnostic (removed from default Table I)"
    assert labels["static_geo_qeb_adapt_vqe"] == "Geo-ADAPT-VQE (QEB reference)"
    latex = (out / "table_i_static_claim_rows.tex").read_text(encoding="utf-8")
    assert "legacy geometry diagnostic (removed from default Table I)" in latex
    assert "Geo-ADAPT-VQE (QEB reference)" in latex


def test_table_i_static_result_summary_merges_metric_enrichment_sidecars(tmp_path: Path) -> None:
    root = tmp_path / "root"
    enrichment_root = tmp_path / "enrichment"
    record_id = _record_id("bose_hubbard", "bose_hubbard_L2")
    records = [{"record_id": record_id, "family": "bose_hubbard", "case_id": "bose_hubbard_L2", "algorithm_id": "static_family_informed_vqe"}]
    _write_records(tmp_path / "records.tsv", records)
    _write_payload(
        root,
        record_id,
        result={
            "status": "ok",
            "method_id": "static_family_informed_vqe",
            "abs_delta_e": 1e-2,
            "compiled_count_2q_total": 4,
            "compiled_depth_total": 6,
            "shots_total": 100,
        },
    )
    sidecar_dir = enrichment_root / record_id / "result"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    (sidecar_dir / "generic_static_metric_enrichment.json").write_text(
        json.dumps(
            {
                "schema": "generic_static_metric_enrichment_v1",
                "record_id": record_id,
                "status": "completed",
                "row_updates": {
                    "abs_delta_e_same_cutoff": 2e-3,
                    "abs_delta_e_reference": 3e-3,
                    "infidelity_same": 4e-4,
                    "compiled_depth_2q_total": 5,
                },
                "metric_statuses": {
                    "abs_delta_e_same_cutoff": "ok",
                    "abs_delta_e_reference": "ok",
                    "infidelity_same": "ok",
                    "compiled_depth_2q_total": "ok",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = summarize.main([
        "--records",
        str(tmp_path / "records.tsv"),
        "--root",
        str(root),
        "--enrichment-root",
        str(enrichment_root),
        "--output-dir",
        str(tmp_path / "out"),
    ])

    assert rc == 0
    summary = json.loads((tmp_path / "out" / "table_i_static_results_summary.json").read_text(encoding="utf-8"))
    row = summary["aggregate_rows"][0]
    assert summary["enrichment_available_count"] == 1
    assert row["delta_e_same_mean"] == 2e-3
    assert row["delta_e_4_mean"] == 3e-3
    assert row["infidelity_same_mean"] == 4e-4
    assert row["depth_2q_mean"] == 5.0


def test_table_i_static_result_summary_ignores_failed_enrichment_sidecars(tmp_path: Path) -> None:
    root = tmp_path / "root"
    enrichment_root = tmp_path / "enrichment"
    record_id = _record_id("bose_hubbard", "bose_hubbard_L2")
    records = [{"record_id": record_id, "family": "bose_hubbard", "case_id": "bose_hubbard_L2", "algorithm_id": "static_family_informed_vqe"}]
    _write_records(tmp_path / "records.tsv", records)
    _write_payload(
        root,
        record_id,
        result={
            "status": "ok",
            "method_id": "static_family_informed_vqe",
            "abs_delta_e": 1e-2,
            "compiled_count_2q_total": 4,
            "compiled_depth_total": 6,
            "shots_total": 100,
        },
    )
    sidecar_dir = enrichment_root / record_id / "result"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    (sidecar_dir / "generic_static_metric_enrichment.json").write_text(
        json.dumps(
            {
                "schema": "generic_static_metric_enrichment_v1",
                "record_id": record_id,
                "status": "failed",
                "row_updates": {
                    "abs_delta_e_same_cutoff": 1e-9,
                    "S_norm": 12.0,
                    "S_grp_total": 44.0,
                },
                "metric_statuses": {
                    "abs_delta_e_same_cutoff": "ok",
                    "S_norm": "ok",
                    "S_grp": "ok",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = summarize.main([
        "--records",
        str(tmp_path / "records.tsv"),
        "--root",
        str(root),
        "--enrichment-root",
        str(enrichment_root),
        "--output-dir",
        str(tmp_path / "out"),
    ])

    assert rc == 0
    summary = json.loads((tmp_path / "out" / "table_i_static_results_summary.json").read_text(encoding="utf-8"))
    row = summary["row_results"][0]
    aggregate = summary["aggregate_rows"][0]
    assert aggregate["delta_e_same_mean"] == 1e-2
    assert row["S_norm"] is None
    assert row["S_norm_status"] == "failed"
    assert row["S_grp_total"] is None
    assert row["S_grp_status"] == "failed"
    assert aggregate["measurement_work_proxy_mean"] is None
    assert aggregate["legacy_measurement_work_proxy_mean"] is None
    assert aggregate["S_grp_available_n"] == 0


def test_table_i_static_result_summary_reports_missing_and_unusable_records(tmp_path: Path) -> None:
    root = tmp_path / "root"
    missing_id = _record_id("hubbard", "hubbard_L2")
    unusable_id = _record_id("bose_hubbard", "bose_hubbard_L2")
    records = [
        {"record_id": missing_id, "family": "hubbard", "case_id": "hubbard_L2", "algorithm_id": "static_family_informed_vqe"},
        {"record_id": unusable_id, "family": "bose_hubbard", "case_id": "bose_hubbard_L2", "algorithm_id": "static_family_informed_vqe"},
    ]
    _write_records(tmp_path / "records.tsv", records)
    _write_payload(root, unusable_id, result={"status": "ok", "method_id": "static_family_informed_vqe"})

    rc = summarize.main([
        "--records",
        str(tmp_path / "records.tsv"),
        "--root",
        str(root),
        "--output-dir",
        str(tmp_path / "out"),
        "--allow-incomplete",
    ])

    assert rc == 0
    summary = json.loads((tmp_path / "out" / "table_i_static_results_summary.json").read_text(encoding="utf-8"))
    assert summary["expected_count"] == 2
    assert summary["benchmarked_count"] == 0
    assert summary["missing_count"] == 1
    assert summary["unusable_count"] == 1


def test_table_i_static_result_summary_keeps_s_norm_as_legacy_not_paper_facing_cost(tmp_path: Path) -> None:
    root = tmp_path / "root"
    enrichment_root = tmp_path / "enrichment"
    record_id = _record_id("bose_hubbard", "bose_hubbard_L2")
    records = [{"record_id": record_id, "family": "bose_hubbard", "case_id": "bose_hubbard_L2", "algorithm_id": "static_family_informed_vqe"}]
    _write_records(tmp_path / "records.tsv", records)
    _write_payload(
        root,
        record_id,
        result={
            "status": "ok",
            "method_id": "static_family_informed_vqe",
            "abs_delta_e": 1e-2,
            "compiled_count_2q_total": 4,
            "compiled_depth_total": 6,
            "shots_total": 1000,
        },
    )
    sidecar_dir = enrichment_root / record_id / "result"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    (sidecar_dir / "generic_static_metric_enrichment.json").write_text(
        json.dumps(
                {
                    "schema": "generic_static_metric_enrichment_v1",
                    "record_id": record_id,
                    "status": "completed",
                    "row_updates": {"S_norm": 12.0, "S_grp_total": 44.0},
                    "metric_statuses": {"S_norm": "ok", "S_grp": "ok"},
                }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = summarize.main([
        "--records", str(tmp_path / "records.tsv"),
        "--root", str(root),
        "--enrichment-root", str(enrichment_root),
        "--output-dir", str(tmp_path / "out"),
    ])

    assert rc == 0
    summary = json.loads((tmp_path / "out" / "table_i_static_results_summary.json").read_text(encoding="utf-8"))
    raw_row = summary["row_results"][0]
    aggregate = summary["aggregate_rows"][0]
    assert raw_row["S_norm"] == 12.0
    assert raw_row["S_grp_total"] == 44.0
    assert raw_row["S_grp_status"] == "ok"
    assert raw_row["measurement_work_proxy"] is None
    assert raw_row["measurement_work_proxy_source"] is None
    assert raw_row["measurement_work_proxy_status"] == "missing_s_alg_status"
    assert raw_row["legacy_measurement_work_proxy"] == 12.0
    assert raw_row["legacy_measurement_work_proxy_source"] == "S_norm"
    assert raw_row["raw_shot_cost_proxy"] is None
    assert raw_row["raw_shots_total"] == 1000.0
    assert raw_row["legacy_shot_proxy"] == 1000.0
    assert raw_row["shot_cost_proxy"] is None
    assert raw_row["shot_cost_proxy_status"] == "raw_fallback_forbidden"
    assert aggregate["measurement_work_proxy_mean"] is None
    assert aggregate["legacy_measurement_work_proxy_mean"] == 12.0
    assert aggregate["S_grp_total_mean"] == 44.0
    assert aggregate["S_grp_available_n"] == 1
    assert aggregate["S_grp_status_counts"] == {"ok": 1}
    assert aggregate["shot_cost_proxy_mean"] is None
    latex = (tmp_path / "out" / "table_i_static_claim_rows.tex").read_text(encoding="utf-8")
    assert "$12$" not in latex
    assert "$44$" not in latex
    assert "$1000$" not in latex


def test_table_i_static_result_summary_uses_s_alg_for_paper_facing_cost(tmp_path: Path) -> None:
    root = tmp_path / "root"
    enrichment_root = tmp_path / "enrichment"
    record_id = _record_id("bose_hubbard", "bose_hubbard_L2")
    records = [{"record_id": record_id, "family": "bose_hubbard", "case_id": "bose_hubbard_L2", "algorithm_id": "static_family_informed_vqe"}]
    _write_records(tmp_path / "records.tsv", records)
    _write_payload(
        root,
        record_id,
        result={
            "status": "ok",
            "method_id": "static_family_informed_vqe",
            "abs_delta_e": 1e-2,
            "compiled_count_2q_total": 4,
            "compiled_depth_total": 6,
            "shots_total": 1000,
        },
    )
    sidecar_dir = enrichment_root / record_id / "result"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    (sidecar_dir / "generic_static_metric_enrichment.json").write_text(
        json.dumps(
            {
                "schema": "generic_static_metric_enrichment_v1",
                "record_id": record_id,
                "status": "completed",
                "row_updates": {"S_alg": 12.0, "S_norm": 999.0, "S_phys": 44.0, "S_l2": 33.0, "S_var": 22.0},
                "metric_statuses": {
                    "S_alg": "ok",
                    "S_norm": "ok",
                    "S_phys": "ok",
                    "S_l2": "ok",
                    "S_phys_var": "ok",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = summarize.main([
        "--records", str(tmp_path / "records.tsv"),
        "--root", str(root),
        "--enrichment-root", str(enrichment_root),
        "--output-dir", str(tmp_path / "out"),
    ])

    assert rc == 0
    summary = json.loads((tmp_path / "out" / "table_i_static_results_summary.json").read_text(encoding="utf-8"))
    raw_row = summary["row_results"][0]
    aggregate = summary["aggregate_rows"][0]
    assert raw_row["measurement_work_proxy"] == 12.0
    assert raw_row["measurement_work_proxy_source"] == "S_alg"
    assert aggregate["measurement_work_proxy_mean"] == 12.0
    assert aggregate["S_alg_mean"] == 12.0
    assert aggregate["S_phys_mean"] == 44.0
    assert aggregate["S_l2_mean"] == 33.0
    assert aggregate["S_var_mean"] == 22.0
    assert aggregate["S_var_available_n"] == 1
    assert aggregate["S_phys_var_status_counts"] == {"ok": 1}
    latex = (tmp_path / "out" / "table_i_static_claim_rows.tex").read_text(encoding="utf-8")
    assert "$12$" in latex
    assert "$999$" not in latex


def test_table_i_static_result_summary_raw_fallback_reports_source_status(tmp_path: Path) -> None:
    root = tmp_path / "root"
    enrichment_root = tmp_path / "enrichment"
    record_id = _record_id("bose_hubbard", "bose_hubbard_L2")
    records = [{"record_id": record_id, "family": "bose_hubbard", "case_id": "bose_hubbard_L2", "algorithm_id": "static_family_informed_vqe"}]
    _write_records(tmp_path / "records.tsv", records)
    _write_payload(
        root,
        record_id,
        result={
            "status": "ok",
            "method_id": "static_family_informed_vqe",
            "abs_delta_e": 1e-2,
            "compiled_count_2q_total": 4,
            "compiled_depth_total": 6,
            "measurement_shots_proxy": 77,
        },
    )
    sidecar_dir = enrichment_root / record_id / "result"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    (sidecar_dir / "generic_static_metric_enrichment.json").write_text(
        json.dumps(
            {
                "schema": "generic_static_metric_enrichment_v1",
                "record_id": record_id,
                "status": "completed",
                "row_updates": {},
                "metric_statuses": {
                    "S_norm": "missing_component_breakdown",
                    "S_grp": "missing_grouped_measurement_breakdown",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = summarize.main([
        "--records", str(tmp_path / "records.tsv"),
        "--root", str(root),
        "--enrichment-root", str(enrichment_root),
        "--output-dir", str(tmp_path / "out"),
    ])

    assert rc == 0
    summary = json.loads((tmp_path / "out" / "table_i_static_results_summary.json").read_text(encoding="utf-8"))
    raw_row = summary["row_results"][0]
    aggregate = summary["aggregate_rows"][0]
    assert raw_row["measurement_work_proxy"] is None
    assert raw_row["measurement_work_proxy_source"] is None
    assert raw_row["measurement_work_proxy_status"] == "missing_s_alg_status"
    assert raw_row["legacy_measurement_work_proxy"] is None
    assert raw_row["legacy_measurement_work_proxy_source"] is None
    assert raw_row["legacy_measurement_work_proxy_status"] == "unavailable:missing_component_breakdown"
    assert raw_row["raw_measurement_shots_proxy"] == 77.0
    assert raw_row["legacy_shot_proxy"] == 77.0
    assert raw_row["shot_cost_proxy"] is None
    assert raw_row["shot_cost_proxy_status"] == "raw_fallback_forbidden"
    assert aggregate["raw_shot_fallback_n"] == 1
    assert aggregate["S_norm_available_n"] == 0
    assert aggregate["S_grp_available_n"] == 0
    assert aggregate["S_grp_status_counts"] == {"missing_grouped_measurement_breakdown": 1}
