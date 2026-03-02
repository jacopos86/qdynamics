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
from pipelines.exact_bench.l3_hh_exp_fidelity_wrapper import RUN_FIELDS as FIDELITY_RUN_FIELDS
from pipelines.exact_bench.overnight_l3_hh_four_method_benchmark import SUMMARY_FIELDS as OVERNIGHT_SUMMARY_FIELDS


class TestProxyRowExtraction(unittest.TestCase):
    def test_extract_row_with_missing_fields_and_coercion(self) -> None:
        row = extract_proxy_metric_row(
            {
                "name": "HH-Layerwise",
                "category": "conventional_vqe",
                "num_params": "12",
                "elapsed_s": "1.25",
                "abs_delta_e": "1.0e-4",
                "sector_leak_flag": "false",
            },
            defaults={"problem": "hh", "L": "3", "vqe_restarts": "4"},
        )
        self.assertEqual(row.method_id, "HH-Layerwise")
        self.assertEqual(row.problem, "hh")
        self.assertEqual(row.L, 3)
        self.assertEqual(row.num_parameters, 12)
        self.assertEqual(row.depth_proxy, 12)
        self.assertAlmostEqual(float(row.runtime_s), 1.25)
        self.assertAlmostEqual(float(row.delta_E_abs), 1.0e-4)
        self.assertFalse(bool(row.sector_leak_flag))
        self.assertEqual(row.vqe_restarts, 4)

    def test_extract_adapt_and_conventional_rows(self) -> None:
        adapt = extract_proxy_metric_row(
            {
                "run_id": "m3_adapt_paop_std|seed1",
                "status": "ok",
                "method_id": "m3_adapt_paop_std",
                "method_kind": "adapt",
                "pool_name": "paop_std",
                "adapt_depth_reached": "7",
                "num_parameters": "7",
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
        self.assertEqual(conventional.depth_proxy, 6)
        self.assertIn("paop", adapt.operator_family_proxy)
        self.assertIn("hva", conventional.operator_family_proxy)
        self.assertEqual(adapt.pool_family_proxy, "paop_std")


class TestProxySidecarWriter(unittest.TestCase):
    def test_sidecar_writer_creates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            sidecars = write_proxy_sidecars(
                [
                    {
                        "run_id": "a",
                        "status": "ok",
                        "method_id": "m1_hh_hva",
                        "method_kind": "conventional",
                        "ansatz_name": "hh_hva",
                        "problem": "hh",
                        "L": 3,
                        "runtime_s": 1.0,
                        "delta_E_abs": 1.0e-3,
                    },
                    {
                        "run_id": "b",
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
                rows = list(reader)
                self.assertEqual(len(rows), 2)

            payload = json.loads(sidecars["summary_json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], SCHEMA_VERSION)
            self.assertEqual(payload["row_count"], 2)
            self.assertIn("source_composition_proxy", payload)


class TestNonRegressionFieldContracts(unittest.TestCase):
    def test_existing_benchmark_fields_unchanged(self) -> None:
        self.assertNotIn("depth_proxy", OVERNIGHT_SUMMARY_FIELDS)
        self.assertNotIn("operator_family_proxy", OVERNIGHT_SUMMARY_FIELDS)
        self.assertNotIn("depth_proxy", FIDELITY_RUN_FIELDS)
        self.assertNotIn("operator_family_proxy", FIDELITY_RUN_FIELDS)


if __name__ == "__main__":
    unittest.main()

