#!/usr/bin/env python3
import copy
import unittest

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


if __name__ == "__main__":
    unittest.main()
