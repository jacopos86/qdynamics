#!/usr/bin/env python3
from __future__ import annotations
import json, unittest
from pathlib import Path
import build_bundle
class GreedyBundleTest(unittest.TestCase):
    def test_bundle(self): self.assertTrue(build_bundle.verify())
    def test_exact_six_rows_and_cutoffs(self):
        for path in sorted((build_bundle.BUNDLE / "jobs").glob("*.json")):
            job=json.loads(path.read_text())
            strong = job["regime_slug"] in {"weak_strong","intermediate_strong","strong_strong_u8"}
            self.assertEqual(job["physics"]["n_ph_work"], 7 if strong else 3)
            self.assertEqual(job["physics"]["n_ph_reference"], 7 if strong else 3)
            self.assertEqual(job["segment"]["target_controller_round"], 50)
    def test_fixed_source_greedy_contract(self):
        job=json.loads(next((build_bundle.BUNDLE / "jobs").glob("*.json")).read_text())
        settings=job["route_identity"]["profile_contract"]["execution_settings"]
        self.assertFalse(settings["phase2_enable_batching"])
        self.assertTrue(settings["phase3_enable_batching"])
        self.assertEqual(settings["phase3_batch_selection_mode"], "greedy_reduced_plane")
        self.assertEqual(settings["phase3_batch_target_size"], 3)
        self.assertEqual(settings["phase3_batch_size_cap"], 3)
if __name__ == "__main__": unittest.main()
