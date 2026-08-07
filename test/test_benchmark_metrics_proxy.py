#!/usr/bin/env python3
"""Tests for wrapper-level benchmark proxy metrics sidecars."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from pipelines.exact_bench.benchmark_metrics_proxy import (
    PROXY_FIELD_ORDER,
    SCHEMA_VERSION,
    extract_proxy_metric_row,
    write_proxy_sidecars,
)


class TestProxyRowExtraction(unittest.TestCase):
    def test_extract_row_with_missing_fields_and_coercion(self) -> None:
        row = extract_proxy_metric_row(
            {
                "name": "HH-Layerwise",
                "case_id": "hh_L3_weak_current_success",
                "category": "conventional_vqe",
                "num_params": "12",
                "elapsed_s": "1.25",
                "abs_delta_e": "1.0e-4",
                "sector_leak_flag": "false",
            },
            defaults={"problem": "hh", "L": "3", "vqe_restarts": "4"},
        )
        self.assertEqual(row.method_id, "HH-Layerwise")
        self.assertEqual(row.hamiltonian_id, "hh_L3_weak_current_success")
        self.assertEqual(row.problem, "hh")
        self.assertEqual(row.L, 3)
        self.assertEqual(row.num_parameters, 12)
        self.assertEqual(row.depth_proxy, 12)
        self.assertAlmostEqual(float(row.runtime_s), 1.25)
        self.assertAlmostEqual(float(row.delta_E_abs), 1.0e-4)
        self.assertFalse(bool(row.sector_leak_flag))
        self.assertEqual(row.vqe_restarts, 4)

    def test_extract_generic_hea_row_preserves_sector_leak_flag(self) -> None:
        row = extract_proxy_metric_row(
            {
                "run_id": "hubbard_L2::static_hea_qiskit_vqe",
                "case_id": "hubbard_L2",
                "status": "ok",
                "method_id": "static_hea_qiskit_vqe",
                "method_kind": "fixed_ansatz_vqe",
                "ansatz_name": "qiskit_hea_linear_ryrz_cx",
                "problem": "hubbard",
                "L": "2",
                "num_parameters": "16",
                "delta_E_abs": "5.0e-4",
                "sector_leak_flag": "true",
            }
        )

        self.assertEqual(row.method_id, "static_hea_qiskit_vqe")
        self.assertTrue(bool(row.sector_leak_flag))
        self.assertAlmostEqual(float(row.delta_E_abs), 5.0e-4)

    def test_extract_adapt_and_conventional_rows(self) -> None:
        adapt = extract_proxy_metric_row(
            {
                "run_id": "m3_adapt_paop_std|seed1",
                "hamiltonian_id": "hh_L2_strong_canonical",
                "status": "ok",
                "method_id": "m3_adapt_paop_std",
                "method_kind": "adapt",
                "pool_name": "paop_std",
                "adapt_depth_reached": "7",
                "num_parameters": "7",
                "selected_operator_count": "5",
                "delta_E_abs": "2.5e-3",
                "nfev": "123",
            }
        )
        conventional = extract_proxy_metric_row(
            {
                "run_id": "m1_hh_hva|seed1",
                "status": "ok",
                "method_id": "m1_hh_hva",
                "method_kind": "conventional",
                "ansatz_name": "hh_hva",
                "num_parameters": "6",
                "vqe_reps": "2",
                "vqe_restarts": "5",
                "vqe_maxiter": "3000",
                "delta_E_abs": "1.2e-2",
            }
        )
        self.assertEqual(adapt.depth_proxy, 7)
        self.assertEqual(adapt.selected_operator_count, 5)
        self.assertEqual(adapt.hamiltonian_id, "hh_L2_strong_canonical")
        self.assertEqual(conventional.depth_proxy, 6)
        self.assertIn("paop", adapt.operator_family_proxy)
        self.assertIn("hva", conventional.operator_family_proxy)
        self.assertEqual(adapt.pool_family_proxy, "paop_std")

    def test_extract_lang_firsov_vqe_row_preserves_pool_and_operator_counts(self) -> None:
        row = extract_proxy_metric_row(
            {
                "run_id": "hh_L2_strong_canonical__hh_lang_firsov_sq_lf_vqe",
                "hamiltonian_id": "hh_L2_strong_canonical",
                "status": "ok",
                "method_id": "hh_lang_firsov_sq_lf_vqe",
                "method_kind": "conventional_vqe",
                "ansatz_name": "hh_lang_firsov_sq_lf",
                "pool_name": "sq_lf_std",
                "num_parameters": "2",
                "selected_operator_count": "2",
                "delta_E_abs": "1.0e-2",
                "nfev": "31",
            }
        )
        self.assertEqual(row.method_kind, "conventional_vqe")
        self.assertEqual(row.ansatz_name, "hh_lang_firsov_sq_lf")
        self.assertEqual(row.pool_name, "sq_lf_std")
        self.assertEqual(row.num_parameters, 2)
        self.assertEqual(row.selected_operator_count, 2)
        self.assertEqual(row.nfev, 31)
        self.assertAlmostEqual(float(row.delta_E_abs), 1.0e-2)
        self.assertEqual(row.pool_family_proxy, "sq_lf_std")

    def test_extract_avqite_row_preserves_selected_operator_count(self) -> None:
        avqite = extract_proxy_metric_row(
            {
                "run_id": "hh_L2_strong_canonical__hh_avqite_uccsd_lifted",
                "hamiltonian_id": "hh_L2_strong_canonical",
                "status": "ok",
                "method_id": "hh_avqite_uccsd_lifted",
                "method_kind": "avqite",
                "ansatz_name": "hh_uccsd_lifted",
                "num_parameters": "3",
                "selected_operator_count": "3",
                "avqite_steps_completed": "57",
                "avqite_stop_reason": "energy_tol",
                "imaginary_time_total": "5.7",
                "delta_E_abs": "1.5e-2",
                "nfev": "77",
            }
        )
        self.assertEqual(avqite.method_kind, "avqite")
        self.assertEqual(avqite.selected_operator_count, 3)
        self.assertEqual(avqite.avqite_steps_completed, 57)
        self.assertEqual(avqite.avqite_stop_reason, "energy_tol")
        self.assertAlmostEqual(float(avqite.imaginary_time_total), 5.7)
        self.assertEqual(avqite.num_parameters, 3)
        self.assertEqual(avqite.nfev, 77)
        self.assertEqual(avqite.operator_family_proxy, "avqite+uccsd")
        self.assertEqual(avqite.pool_family_proxy, "uccsd")

    def test_extract_qsci_row_preserves_subspace_dimension(self) -> None:
        qsci = extract_proxy_metric_row(
            {
                "run_id": "hh_L2_strong_canonical__hh_qsci_sq_lf_std",
                "hamiltonian_id": "hh_L2_strong_canonical",
                "status": "ok",
                "method_id": "hh_qsci_sq_lf_std",
                "method_kind": "qsci",
                "ansatz_name": "hh_qsci_sq_lf_std",
                "pool_name": "sq_lf_std",
                "selected_operator_count": "12",
                "subspace_dimension": "7",
                "delta_E_abs": "2.0e-2",
                "nfev": "12",
            }
        )
        self.assertEqual(qsci.method_kind, "qsci")
        self.assertEqual(qsci.pool_name, "sq_lf_std")
        self.assertEqual(qsci.selected_operator_count, 12)
        self.assertEqual(qsci.subspace_dimension, 7)
        self.assertEqual(qsci.nfev, 12)
        self.assertAlmostEqual(float(qsci.delta_E_abs), 2.0e-2)
        self.assertEqual(qsci.pool_family_proxy, "sq_lf_std")

    def test_extract_sqd_row_preserves_subspace_dimension_and_shots_total(self) -> None:
        sqd = extract_proxy_metric_row(
            {
                "run_id": "hh_L2_strong_canonical__hh_sqd_sq_lf_std",
                "hamiltonian_id": "hh_L2_strong_canonical",
                "status": "ok",
                "method_id": "hh_sqd_sq_lf_std",
                "method_kind": "sqd",
                "ansatz_name": "hh_sqd_sq_lf_std",
                "pool_name": "sq_lf_std",
                "selected_operator_count": "12",
                "subspace_dimension": "7",
                "shots_total": "3072",
                "delta_E_abs": "2.5e-2",
                "nfev": "12",
            }
        )
        self.assertEqual(sqd.method_kind, "sqd")
        self.assertEqual(sqd.pool_name, "sq_lf_std")
        self.assertEqual(sqd.selected_operator_count, 12)
        self.assertEqual(sqd.subspace_dimension, 7)
        self.assertEqual(sqd.shots_total, 3072)
        self.assertEqual(sqd.nfev, 12)
        self.assertAlmostEqual(float(sqd.delta_E_abs), 2.5e-2)
        self.assertEqual(sqd.pool_family_proxy, "sq_lf_std")


class TestProxySidecarWriter(unittest.TestCase):
    def test_sidecar_writer_creates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            sidecars = write_proxy_sidecars(
                [
                    {
                        "run_id": "a",
                        "hamiltonian_id": "hh_L2_strong_canonical",
                        "status": "ok",
                        "method_id": "m1_hh_hva",
                        "method_kind": "conventional",
                        "ansatz_name": "hh_hva",
                        "problem": "hh",
                        "L": 3,
                        "runtime_s": 1.0,
                        "delta_E_abs": 1.0e-3,
                        "selected_operator_count": 6,
                        "subspace_dimension": 4,
                        "shots_total": 512,
                    },
                    {
                        "run_id": "b",
                        "case_id": "hh_L3_weak_current_success",
                        "status": "ok",
                        "method_id": "m3_adapt_paop_std",
                        "method_kind": "adapt",
                        "pool_name": "paop_std",
                        "problem": "hh",
                        "L": 3,
                        "runtime_s": 2.0,
                        "delta_E_abs": 2.0e-3,
                        "adapt_depth_reached": 9,
                    },
                ],
                out_dir,
                summary_extras={"source_composition_proxy": {"A": {"uccsd": 3, "paop": 4, "hva": 0}}},
            )
            self.assertTrue(sidecars["csv"].exists())
            self.assertTrue(sidecars["jsonl"].exists())
            self.assertTrue(sidecars["summary_json"].exists())

            with sidecars["csv"].open("r", encoding="utf-8", newline="") as f_csv:
                reader = csv.DictReader(f_csv)
                self.assertEqual(reader.fieldnames, PROXY_FIELD_ORDER)
                self.assertIn("hamiltonian_id", reader.fieldnames or [])
                self.assertIn("avqite_steps_completed", reader.fieldnames or [])
                self.assertIn("avqite_stop_reason", reader.fieldnames or [])
                self.assertIn("imaginary_time_total", reader.fieldnames or [])
                self.assertIn("subspace_dimension", reader.fieldnames or [])
                self.assertIn("shots_total", reader.fieldnames or [])
                rows = list(reader)
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["hamiltonian_id"], "hh_L2_strong_canonical")
                self.assertEqual(rows[0]["selected_operator_count"], "6")
                self.assertEqual(rows[0]["subspace_dimension"], "4")
                self.assertEqual(rows[0]["shots_total"], "512")
                self.assertEqual(rows[1]["hamiltonian_id"], "hh_L3_weak_current_success")

            payload = json.loads(sidecars["summary_json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], SCHEMA_VERSION)
            self.assertEqual(payload["schema"], "hh_bench_metrics_v5")
            self.assertEqual(payload["row_count"], 2)
            self.assertEqual(payload["shots_total"]["sum"], 512)
            self.assertIn("source_composition_proxy", payload)

if __name__ == "__main__":
    unittest.main()
