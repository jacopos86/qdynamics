#!/usr/bin/env python3
import copy
import json
import tempfile
import unittest
from pathlib import Path

import beam_evidence_validation as validation
import revalidate_v6_archive


class SourceOnlyRuntimeReceiptRepairTest(unittest.TestCase):
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
                    "execution_settings": dict(
                        validation.SOURCE_ONLY_RUNTIME_SETTINGS
                    ),
                    "semantic_invariants": {
                        "beam_expanded_child_cap_per_round": 6,
                    },
                },
            },
        }

    def test_result_settings_do_not_require_source_only_route_fields(self):
        settings = dict(validation.COMMON_RUNTIME_SETTINGS)
        for key in validation.SOURCE_ONLY_RUNTIME_SETTINGS:
            self.assertNotIn(key, settings)
        settings["phase3_hardware_cost_normalization_mode"] = (
            "family_robust_symmetric_arctan_v1"
        )
        validation._validate_runtime_settings(
            settings,
            expected_cost_mode="family_robust_symmetric_arctan_v1",
        )

    def test_disabled_command_and_route_receipt_pass(self):
        receipt = validation.validate_source_only_runtime_settings_receipt(
            self.normalized(), digest=self.digest
        )
        self.assertEqual(receipt["status"], "pass")
        self.assertTrue(receipt["phase_live_hysteresis_disabled"])
        self.assertEqual(
            receipt["source_only_runtime_settings"],
            dict(validation.SOURCE_ONLY_RUNTIME_SETTINGS),
        )
        self.assertEqual(receipt["beam_expanded_child_cap_per_round"], 6)

    def test_derived_beam_fields_are_not_required_from_flat_telemetry(self):
        self.assertNotIn(
            "expanded_child_cap_per_round", validation.EXPECTED_BEAM_RUNTIME
        )
        self.assertNotIn("terminal_archive_mode", validation.EXPECTED_BEAM_RUNTIME)

    def test_missing_disabled_flag_fails(self):
        payload = self.normalized()
        payload["command_argv"].pop()
        with self.assertRaises(ValueError):
            validation.validate_source_only_runtime_settings_receipt(
                payload, digest=self.digest
            )

    def test_enabled_flag_fails(self):
        payload = self.normalized()
        payload["command_argv"].append("--phase-live-hysteresis-enabled")
        with self.assertRaises(ValueError):
            validation.validate_source_only_runtime_settings_receipt(
                payload, digest=self.digest
            )

    def test_each_source_only_route_setting_fails_closed(self):
        for key in validation.SOURCE_ONLY_RUNTIME_SETTINGS:
            with self.subTest(key=key):
                payload = self.normalized()
                payload["route_identity"]["profile_contract"][
                    "execution_settings"
                ].pop(key)
                with self.assertRaises(ValueError):
                    validation.validate_source_only_runtime_settings_receipt(
                        payload, digest=self.digest
                    )

    def test_derived_beam_cap_fails_closed(self):
        payload = self.normalized()
        payload["route_identity"]["profile_contract"]["semantic_invariants"][
            "beam_expanded_child_cap_per_round"
        ] = 5
        with self.assertRaises(ValueError):
            validation.validate_source_only_runtime_settings_receipt(
                payload, digest=self.digest
            )

    def test_digest_mismatch_fails(self):
        with self.assertRaises(ValueError):
            validation.validate_source_only_runtime_settings_receipt(
                self.normalized(), digest="e" * 64
            )

    def test_compact_current_history_crosschecks_full_selected_history(self):
        selected = {
            "depth": 1,
            "branch_id": 4,
            "parent_branch_id": 0,
            "selected_op": "macro::x",
            "selected_position": 0,
            "batch_size": 1,
            "energy_before_opt": -1.0,
            "energy_after_opt": -1.1,
            "delta_energy": -0.1,
            "nfev_total_before_step": 2,
            "nfev_total_after_step": 5,
            "nfev_step_total_delta": 3,
        }
        checkpoint = {"checkpoint_sha256": "a" * 64}
        compact = {**selected, "active_prefix_checkpoint": checkpoint}
        receipt = validation._validate_compact_current_history(
            [compact],
            selected_path={"history": [selected], "checkpoints": [checkpoint]},
            expected_rounds=1,
        )
        self.assertEqual(receipt["status"], "pass")
        compact["selected_op"] = "macro::drift"
        with self.assertRaises(ValueError):
            validation._validate_compact_current_history(
                [compact],
                selected_path={"history": [selected], "checkpoints": [checkpoint]},
                expected_rounds=1,
            )

    def test_one_sided_cost_mode_uses_canonical_runtime_value(self):
        settings = dict(validation.COMMON_RUNTIME_SETTINGS)
        settings["phase3_hardware_cost_normalization_mode"] = "family_robust_v1"
        validation._validate_runtime_settings(
            settings, expected_cost_mode="family_robust_v1"
        )
        with self.assertRaises(ValueError):
            validation._validate_runtime_settings(
                settings, expected_cost_mode="family_robust_penalty_only_v1"
            )

    def test_compact_controller_validator_is_fail_closed(self):
        self.assertTrue(hasattr(validation, "_validate_compact_controller_history"))
        with self.assertRaises(ValueError):
            validation._validate_compact_controller_history(
                [], expected_rounds=50, fallback_rounds=[]
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
