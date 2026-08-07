#!/usr/bin/env python3
import copy
import json
import tempfile
import unittest
from pathlib import Path

import beam_evidence_validation as validation
import revalidate_v6_archive


class HysteresisReceiptRepairTest(unittest.TestCase):
    digest = "d" * 64

    def normalized(self):
        return {
            "command_argv": [
                "python3",
                "-m",
                "pipelines.static_adapt.adapt_pipeline",
                "--phase-live-hysteresis-disabled",
            ],
            "route_identity": {
                "profile_contract_sha256": self.digest,
                "profile_contract": {
                    "execution_settings": {
                        "phase_live_hysteresis_enabled": False,
                    }
                },
            },
        }

    def test_result_settings_do_not_require_unserialized_hysteresis_field(self):
        settings = dict(validation.COMMON_RUNTIME_SETTINGS)
        self.assertNotIn("phase_live_hysteresis_enabled", settings)
        settings["phase3_hardware_cost_normalization_mode"] = (
            "family_robust_symmetric_arctan_v1"
        )
        validation._validate_runtime_settings(
            settings,
            expected_cost_mode="family_robust_symmetric_arctan_v1",
        )

    def test_disabled_command_and_route_receipt_pass(self):
        receipt = validation.validate_hysteresis_disabled_source_receipt(
            self.normalized(), digest=self.digest
        )
        self.assertEqual(receipt["status"], "pass")
        self.assertTrue(receipt["phase_live_hysteresis_disabled"])

    def test_missing_disabled_flag_fails(self):
        payload = self.normalized()
        payload["command_argv"].pop()
        with self.assertRaises(ValueError):
            validation.validate_hysteresis_disabled_source_receipt(
                payload, digest=self.digest
            )

    def test_enabled_flag_fails(self):
        payload = self.normalized()
        payload["command_argv"].append("--phase-live-hysteresis-enabled")
        with self.assertRaises(ValueError):
            validation.validate_hysteresis_disabled_source_receipt(
                payload, digest=self.digest
            )

    def test_route_contract_true_fails(self):
        payload = self.normalized()
        payload["route_identity"]["profile_contract"]["execution_settings"][
            "phase_live_hysteresis_enabled"
        ] = True
        with self.assertRaises(ValueError):
            validation.validate_hysteresis_disabled_source_receipt(
                payload, digest=self.digest
            )

    def test_digest_mismatch_fails(self):
        with self.assertRaises(ValueError):
            validation.validate_hysteresis_disabled_source_receipt(
                self.normalized(), digest="e" * 64
            )

    def test_only_exact_v6_validation_failure_is_repairable(self):
        self.assertEqual(
            revalidate_v6_archive.KNOWN_FAILURE,
            "ValueError: normalized candidate setting drift: "
            "phase_live_hysteresis_enabled",
        )

    def _frozen_output_fixture(self, root: Path):
        output = root / revalidate_v6_archive.V6_BUNDLE_ID / "weak_weak"
        (output / "json").mkdir(parents=True)
        manifest = {
            "route_identity": {"profile_contract_sha256": self.digest},
            "physics": {"regime_slug": "weak_weak"},
            "segment": {"target_controller_round": 50},
        }
        normalized = dict(manifest)
        payloads = {
            output / "normalized_run_manifest.json": normalized,
            output / "json/result.json": {"status": "complete"},
            output / "json/current.json": {"outer_iteration": 50},
            output / "json/estimator_call_ledger.json": {"status": "closed"},
        }
        for path, payload in payloads.items():
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        artifacts = {}
        for key, path in {
            "normalized_runtime_manifest_json": output
            / "normalized_run_manifest.json",
            "result_json": output / "json/result.json",
            "current_json": output / "json/current.json",
            "ledger_json": output / "json/estimator_call_ledger.json",
        }.items():
            artifacts[key] = {
                "exists": True,
                "sha256": revalidate_v6_archive.sha256(path),
                "size_bytes": path.stat().st_size,
            }
        execution = {
            "status": "failed",
            "exit_code": 70,
            "validation_error": revalidate_v6_archive.KNOWN_FAILURE,
            "artifacts": artifacts,
        }
        (output / "execution.json").write_text(
            json.dumps(execution) + "\n", encoding="utf-8"
        )
        return output, manifest

    def test_exact_completed_science_validator_failure_receipt_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, manifest = self._frozen_output_fixture(Path(tmp))
            receipt = revalidate_v6_archive.validate_original_failure(
                output=output, manifest=manifest
            )
            self.assertEqual(receipt["known_failure"], revalidate_v6_archive.KNOWN_FAILURE)
            self.assertEqual(len(receipt["result_sha256"]), 64)

    def test_different_failure_is_not_repairable(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, manifest = self._frozen_output_fixture(Path(tmp))
            execution_path = output / "execution.json"
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            execution["validation_error"] = "RuntimeError: scientific failure"
            execution_path.write_text(json.dumps(execution) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                revalidate_v6_archive.validate_original_failure(
                    output=output, manifest=manifest
                )


if __name__ == "__main__":
    unittest.main()
