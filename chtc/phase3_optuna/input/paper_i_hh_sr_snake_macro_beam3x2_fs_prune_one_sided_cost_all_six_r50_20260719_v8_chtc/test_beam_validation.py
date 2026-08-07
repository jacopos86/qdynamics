#!/usr/bin/env python3
import copy
import unittest

import beam_evidence_validation as validation
import run_job


class BeamControllerWinnerValidationTest(unittest.TestCase):
    digest = "d" * 64

    def fixture(self):
        relationship = {
            "schema_version": "static_adapt_beam_final_checkpoint_relationship_v1",
            "relationship_present": True,
            "reason": "non_target_terminal_selected_with_recoverable_frontier",
            "checkpoint_branch_policy": "best_frontier_branch",
            "recoverable_frontier_deeper_than_terminal": True,
            "diagnostic_terminal_branch_id": 3,
            "diagnostic_terminal_branch": {
                "branch_id": 3,
                "status": "terminal",
                "terminated": True,
                "depth_local": 2,
                "history_count": 2,
            },
            "recoverable_frontier_branch_id": 140,
            "recoverable_frontier_branch": {
                "branch_id": 140,
                "status": "frontier",
                "terminated": False,
                "depth_local": 50,
                "history_count": 50,
            },
        }
        beam = dict(validation.EXPECTED_BEAM_RUNTIME)
        beam.update({
            "rounds": [{} for _ in range(50)],
            "final_checkpoint_relationship": relationship,
        })
        return {
            "beam": beam,
            "current_adapt": {
                "partial_checkpoint": True,
                "success": False,
                "history_count": 50,
                "history_tail_count": 50,
                "history_checkpoint_complete": True,
                "branch_id": 140,
            },
            "current_checkpoint": {
                "depth": 50,
                "branch_id": 140,
                "parent_branch_id": 137,
                "checkpoint_branch_policy": "best_frontier_branch",
                "sr_route_profile_contract_sha256": self.digest,
            },
            "segment": {
                "source_controller_round": 0,
                "target_controller_round": 50,
                "max_new_admissions": 50,
                "final_controller_round": 2,
                "new_admission_records": 2,
                "final_depth": 2,
            },
        }

    def validate(self, fixture):
        return validation.validate_controller_winner_relationship(
            **fixture,
            digest=self.digest,
            target_round=50,
            target_new_admissions=50,
        )

    def test_round2_winner_with_round50_frontier_passes(self):
        self.assertEqual(self.validate(self.fixture())["selected_round"], 2)

    def test_missing_controller_round50_fails(self):
        fixture = self.fixture()
        fixture["beam"]["rounds"].pop()
        with self.assertRaises(ValueError):
            self.validate(fixture)

    def test_beam_shape_drift_fails(self):
        fixture = self.fixture()
        fixture["beam"]["children_per_parent"] = 3
        with self.assertRaises(ValueError):
            self.validate(fixture)

    def test_current_frontier_branch_mismatch_fails(self):
        fixture = self.fixture()
        fixture["current_checkpoint"]["branch_id"] = 139
        with self.assertRaises(ValueError):
            self.validate(fixture)

    def test_ordinary_round50_winner_passes_with_complete_relationship(self):
        fixture = self.fixture()
        fixture["segment"]["final_controller_round"] = 50
        fixture["segment"]["new_admission_records"] = 50
        relation = fixture["beam"]["final_checkpoint_relationship"]
        relation.update({
            "relationship_present": False,
            "reason": "not_applicable",
            "recoverable_frontier_deeper_than_terminal": False,
            "diagnostic_terminal_branch_id": 140,
            "diagnostic_terminal_branch": {
                "branch_id": 140,
                "status": "frontier",
                "terminated": False,
                "depth_local": 50,
                "history_count": 50,
            },
        })
        self.assertEqual(self.validate(fixture)["selected_round"], 50)

    def test_qiskit_checkpoint_uses_selected_round2(self):
        checkpoint = {
            "outer_iteration": 2,
            "checkpoint_kind": "post_admission_prune",
            "checkpoint_sha256": "a" * 64,
        }
        result = {"adapt_vqe": {"active_prefix_checkpoints": [checkpoint]}}
        self.assertIs(run_job.terminal_checkpoint(result, 2), checkpoint)
        with self.assertRaises(ValueError):
            run_job.terminal_checkpoint(result, 50)


class BeamFailClosedGateTest(unittest.TestCase):
    def settings(self):
        payload = dict(validation.COMMON_RUNTIME_SETTINGS)
        payload["phase3_hardware_cost_normalization_mode"] = (
            "family_robust_symmetric_arctan_v1"
        )
        return payload

    def test_common_runtime_settings_pass(self):
        validation._validate_runtime_settings(
            self.settings(),
            expected_cost_mode="family_robust_symmetric_arctan_v1",
        )

    def test_prune_policy_drift_fails(self):
        payload = self.settings()
        payload["phase1_prune_policy"] = "off"
        with self.assertRaises(ValueError):
            validation._validate_runtime_settings(
                payload,
                expected_cost_mode="family_robust_symmetric_arctan_v1",
            )

    def test_cost_arm_drift_fails(self):
        with self.assertRaises(ValueError):
            validation._validate_runtime_settings(
                self.settings(), expected_cost_mode="family_robust_v1"
            )

    def test_lane_receipt_must_equal_selected_feature_row(self):
        feature = {
            "static_lane_route": "physical_operator_type",
            "physical_operator_lane": "exchange",
            "physical_operator_quality": "healthy",
            "physical_operator_hh_full_meta_class": "exchange",
            "physical_operator_lane_source": "full_meta",
            "physical_operator_lane_health": 1.0,
            "physical_operator_lane_relative_health": 1.0,
            "physical_operator_lane_live": True,
        }
        row = dict(feature)
        row["selected_feature_rows"] = [dict(feature)]
        validation._validate_lane_receipt(row, outer_iteration=1)
        row["physical_operator_lane_health"] = 0.5
        with self.assertRaises(ValueError):
            validation._validate_lane_receipt(row, outer_iteration=1)

    def test_selected_checkpoint_list_is_exact(self):
        checkpoint = {"outer_iteration": 1, "checkpoint_sha256": "a" * 64}
        adapt = {"active_prefix_checkpoints": [copy.deepcopy(checkpoint)]}
        validation._validate_selected_checkpoint_list(
            adapt, path={"checkpoints": [checkpoint]}
        )
        adapt["active_prefix_checkpoints"][0]["checkpoint_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            validation._validate_selected_checkpoint_list(
                adapt, path={"checkpoints": [checkpoint]}
            )

    def test_current_pointer_depth_crosscheck_fails(self):
        receipt = {
            "outer_iteration": 50,
            "branch_id": "140",
            "parent_branch_id": "137",
        }
        checkpoint = {
            "outer_iteration": 50,
            "checkpoint_kind": "post_admission_prune",
            "active_ansatz_depth": 49,
            "estimator_ledger_receipt": receipt,
        }
        current = {
            "schema_version": "static_adapt_current_checkpoint_v1",
            "no_credentials_serialized": True,
        }
        current_adapt = {"ansatz_depth": 49}
        current_checkpoint = {
            "complete": False,
            "reason": "partial",
            "depth": 50,
            "ansatz_depth": 48,
            "branch_id": 140,
            "parent_branch_id": 137,
        }
        with self.assertRaises(ValueError):
            validation._validate_current_pointer(
                current=current,
                current_adapt=current_adapt,
                current_checkpoint=current_checkpoint,
                controller_path={
                    "final_active_depth": 49,
                    "checkpoints": [checkpoint],
                },
                target_round=50,
            )


if __name__ == "__main__":
    unittest.main()
