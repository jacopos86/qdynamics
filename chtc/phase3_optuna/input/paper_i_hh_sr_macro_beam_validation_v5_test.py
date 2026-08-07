#!/usr/bin/env python3
import copy
import unittest

import beam_evidence_validation
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
                "status": "terminal",
                "terminated": True,
                "depth_local": 2,
                "history_count": 2,
            },
            "recoverable_frontier_branch_id": 140,
            "recoverable_frontier_branch": {
                "status": "frontier",
                "terminated": False,
                "depth_local": 50,
                "history_count": 50,
            },
        }
        return {
            "beam": {
                "rounds": [{} for _ in range(50)],
                "final_checkpoint_relationship": relationship,
            },
            "current_adapt": {
                "history_count": 50,
                "history_checkpoint_complete": True,
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
        return beam_evidence_validation.validate_controller_winner_relationship(
            **fixture,
            digest=self.digest,
            target_round=50,
            target_new_admissions=50,
        )

    def test_round2_winner_with_round50_frontier_passes(self):
        receipt = self.validate(self.fixture())
        self.assertEqual(receipt["selected_round"], 2)

    def test_missing_controller_round50_fails(self):
        fixture = self.fixture()
        fixture["beam"]["rounds"].pop()
        with self.assertRaises(ValueError):
            self.validate(fixture)

    def test_missing_relationship_fails(self):
        fixture = self.fixture()
        fixture["beam"]["final_checkpoint_relationship"] = {}
        with self.assertRaises(ValueError):
            self.validate(fixture)

    def test_terminal_branch_history_mismatch_fails(self):
        fixture = self.fixture()
        relation = fixture["beam"]["final_checkpoint_relationship"]
        relation["diagnostic_terminal_branch"]["history_count"] = 1
        with self.assertRaises(ValueError):
            self.validate(fixture)

    def test_current_frontier_branch_mismatch_fails(self):
        fixture = self.fixture()
        fixture["current_checkpoint"]["branch_id"] = 139
        with self.assertRaises(ValueError):
            self.validate(fixture)

    def test_ordinary_round50_winner_passes_without_relationship(self):
        fixture = self.fixture()
        fixture["segment"]["final_controller_round"] = 50
        fixture["segment"]["new_admission_records"] = 50
        fixture["beam"]["final_checkpoint_relationship"] = {
            "relationship_present": False,
            "reason": "not_applicable",
            "checkpoint_branch_policy": "best_frontier_branch",
        }
        receipt = self.validate(fixture)
        self.assertEqual(receipt["selected_round"], 50)

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


if __name__ == "__main__":
    unittest.main()
