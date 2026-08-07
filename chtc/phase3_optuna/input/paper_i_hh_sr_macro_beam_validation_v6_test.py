#!/usr/bin/env python3
"""Focused fail-closed fixtures for the v6 macro/beam evidence validator."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest


try:
    import beam_evidence_validation
except ModuleNotFoundError:
    _HERE = Path(__file__).resolve().parent
    _EVIDENCE = _HERE / (
        "paper_i_hh_sr_snake_macro_beam3x2_fs_prune_one_sided_cost_"
        "all_six_r50_20260719_v5_chtc/evidence_validation.py"
    )
    _SPEC = importlib.util.spec_from_file_location("evidence_validation", _EVIDENCE)
    if _SPEC is None or _SPEC.loader is None:
        raise RuntimeError("unable to load evidence_validation fixture dependency")
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules["evidence_validation"] = _MODULE
    _SPEC.loader.exec_module(_MODULE)
    _VALIDATOR = _HERE / "paper_i_hh_sr_macro_beam_evidence_validation_v6.py"
    _SPEC = importlib.util.spec_from_file_location(
        "beam_evidence_validation", _VALIDATOR
    )
    if _SPEC is None or _SPEC.loader is None:
        raise RuntimeError("unable to load v6 beam evidence validator")
    beam_evidence_validation = importlib.util.module_from_spec(_SPEC)
    sys.modules["beam_evidence_validation"] = beam_evidence_validation
    _SPEC.loader.exec_module(beam_evidence_validation)


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
        return {
            "beam": {
                **beam_evidence_validation.EXPECTED_BEAM_RUNTIME,
                "rounds": [{} for _ in range(50)],
                "final_checkpoint_relationship": relationship,
            },
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
        return beam_evidence_validation.validate_controller_winner_relationship(
            **fixture,
            digest=self.digest,
            target_round=50,
            target_new_admissions=50,
        )

    def test_round2_winner_with_round50_frontier_passes(self):
        receipt = self.validate(self.fixture())
        self.assertEqual(receipt["selected_round"], 2)
        self.assertEqual(receipt["recoverable_frontier_branch_id"], 140)

    def test_beam_3x2_or_terminal_archive_drift_fails(self):
        for field in beam_evidence_validation.EXPECTED_BEAM_RUNTIME:
            with self.subTest(field=field):
                fixture = self.fixture()
                fixture["beam"][field] = None
                with self.assertRaises(ValueError):
                    self.validate(fixture)

    def test_controller_horizon_and_selected_local_depth_are_separate(self):
        fixture = self.fixture()
        fixture["beam"]["rounds"].pop()
        with self.assertRaises(ValueError):
            self.validate(fixture)
        fixture = self.fixture()
        fixture["segment"]["new_admission_records"] = 50
        with self.assertRaises(ValueError):
            self.validate(fixture)


class RuntimeAndLaneValidationTest(unittest.TestCase):
    cost_mode = "family_robust_symmetric_arctan_v1"

    def settings(self):
        return {
            **beam_evidence_validation.COMMON_RUNTIME_SETTINGS,
            "phase3_hardware_cost_normalization_mode": self.cost_mode,
        }

    @staticmethod
    def drift(value):
        if isinstance(value, bool):
            return not value
        if isinstance(value, str):
            return value + "_drift"
        return value + 1

    def test_every_locked_runtime_setting_is_fail_closed(self):
        beam_evidence_validation._validate_runtime_settings(
            self.settings(), expected_cost_mode=self.cost_mode
        )
        self.assertEqual(
            beam_evidence_validation.COMMON_RUNTIME_SETTINGS[
                "phase1_prune_policy"
            ],
            "recoverability_ladder_v1",
        )
        for field, value in self.settings().items():
            with self.subTest(field=field):
                settings = self.settings()
                settings[field] = self.drift(value)
                with self.assertRaises(ValueError):
                    beam_evidence_validation._validate_runtime_settings(
                        settings, expected_cost_mode=self.cost_mode
                    )

    def lane_row(self):
        feature = {
            field: f"value:{field}"
            for field in beam_evidence_validation.LANE_FIELDS
        }
        feature["static_lane_route"] = "physical_operator_type"
        row = dict(feature)
        row["selected_feature_rows"] = [copy.deepcopy(feature)]
        return row

    def test_all_eight_top_level_lane_fields_must_equal_selected_row(self):
        beam_evidence_validation._validate_lane_receipt(
            self.lane_row(), outer_iteration=1
        )
        self.assertEqual(len(beam_evidence_validation.LANE_FIELDS), 8)
        for field in beam_evidence_validation.LANE_FIELDS:
            with self.subTest(field=field):
                row = self.lane_row()
                row[field] = "top-level-drift"
                with self.assertRaises(ValueError):
                    beam_evidence_validation._validate_lane_receipt(
                        row, outer_iteration=1
                    )

    def test_arm_specific_fallback_and_ordinary_novelty_off(self):
        policy = "collective_span_novelty_over_symmetric_cost_v1"
        payload = {
            "schema": "sr_all_energy_models_infeasible_novelty_fallback_telemetry_v1",
            "enabled": True,
            "policy": policy,
            "ordinary_phase2_multiplier_active": False,
            "ordinary_phase3_multiplier_active": False,
            "phase2_curvature_failure_can_trigger": False,
            "paper_reporting_scope": "telemetry_gate_only_v1",
            "query_charge_total": 0,
            "controller_rounds": [],
            "activation_count": 0,
            "fired": False,
        }
        beam_evidence_validation._validate_fallback(
            payload,
            label="fixture",
            expected_policy=policy,
            observed_rounds=[],
        )
        for field, value in (
            ("policy", "collective_span_novelty_over_cost_v1"),
            ("ordinary_phase2_multiplier_active", True),
            ("ordinary_phase3_multiplier_active", True),
        ):
            with self.subTest(field=field):
                drifted = copy.deepcopy(payload)
                drifted[field] = value
                with self.assertRaises(ValueError):
                    beam_evidence_validation._validate_fallback(
                        drifted,
                        label="fixture",
                        expected_policy=policy,
                        observed_rounds=[],
                    )


class CheckpointIdentityValidationTest(unittest.TestCase):
    def test_selected_checkpoint_list_is_identical_to_embedded_history(self):
        checkpoints = [
            {"checkpoint_sha256": "a" * 64, "nested": {"depth": 1}},
            {"checkpoint_sha256": "b" * 64, "nested": {"depth": 2}},
        ]
        adapt = {"active_prefix_checkpoints": copy.deepcopy(checkpoints)}
        path = {"checkpoints": checkpoints}
        beam_evidence_validation._validate_selected_checkpoint_list(adapt, path=path)
        adapt["active_prefix_checkpoints"][1]["nested"]["depth"] = 99
        with self.assertRaises(ValueError):
            beam_evidence_validation._validate_selected_checkpoint_list(
                adapt, path=path
            )

    def current_fixture(self):
        receipt = {"outer_iteration": 50, "branch_id": 140, "parent_branch_id": 137}
        checkpoint = {
            "outer_iteration": 50,
            "checkpoint_kind": "post_admission_prune",
            "active_ansatz_depth": 37,
            "estimator_ledger_receipt": receipt,
        }
        return {
            "current": {
                "schema_version": "static_adapt_current_checkpoint_v1",
                "no_credentials_serialized": True,
            },
            "current_adapt": {"ansatz_depth": 37},
            "current_checkpoint": {
                "complete": False,
                "reason": "beam_round_complete",
                "depth": 50,
                "ansatz_depth": 37,
                "branch_id": 140,
                "parent_branch_id": 137,
            },
            "controller_path": {
                "final_active_depth": 37,
                "checkpoints": [checkpoint],
            },
            "target_round": 50,
        }

    def test_current_pointer_identity_matches_embedded_round50_checkpoint(self):
        fixture = self.current_fixture()
        beam_evidence_validation._validate_current_pointer(**fixture)
        for field, value in (("depth", 49), ("ansatz_depth", 36)):
            with self.subTest(field=field):
                drifted = copy.deepcopy(fixture)
                drifted["current_checkpoint"][field] = value
                with self.assertRaises(ValueError):
                    beam_evidence_validation._validate_current_pointer(**drifted)


class BeamEstimatorReceiptValidationTest(unittest.TestCase):
    components = beam_evidence_validation.COMPONENTS

    def receipt(self, sequence, outer, kind, branch, parent):
        components = {key: int(key == "N_H_outer") for key in self.components}
        cumulative = {
            key: sequence * int(key == "N_H_outer") for key in self.components
        }
        return {
            "schema": "paper_i_active_prefix_estimator_ledger_receipt_v1",
            "enabled": True,
            "status": "complete",
            "checkpoint_sequence": sequence,
            "canonical_same_state_deduplication_active": True,
            "raw_occurrences_preserved": True,
            "outer_iteration": outer,
            "checkpoint_kind": kind,
            "branch_id": branch,
            "parent_branch_id": parent,
            "occurrence_sequence_start_exclusive": sequence - 1,
            "raw_occurrence_delta": {"components": components, "total": 1},
            "unique_primitive_delta": {"components": components, "S_alg": 1},
            "cumulative_raw_occurrences": {
                "components": cumulative,
                "total": sequence,
            },
            "cumulative_unique_primitives": {
                "components": cumulative,
                "S_alg": sequence,
            },
        }

    def fixture(self):
        first = self.receipt(1, 1, "post_admission_prune", "10", "0")
        second = self.receipt(2, 2, "post_admission_prune", "20", "10")
        terminal = self.receipt(
            3, 1, "terminal_post_final_refit_and_prune", "10", "0"
        )
        totals = {key: 3 * int(key == "N_H_outer") for key in self.components}
        closure = {
            "schema": "paper_i_active_prefix_estimator_ledger_closure_v1",
            "enabled": True,
            "status": "complete",
            "passed": True,
            "includes_discarded_branch_checkpoints": True,
            "receipt_count": 3,
            "summed_raw_occurrences": {"components": totals, "total": 3},
            "terminal_raw_occurrences": {"components": totals, "total": 3},
            "summed_unique_primitives": {"components": totals, "S_alg": 3},
            "terminal_unique_primitives": {"components": totals, "S_alg": 3},
        }
        return {
            "adapt": {
                "continuation": {
                    "all_active_prefix_estimator_ledger_receipts": [
                        first,
                        second,
                        terminal,
                    ],
                    "active_prefix_estimator_ledger_closure": closure,
                }
            },
            "controller_checkpoints": [
                {"estimator_ledger_receipt": first},
                {"estimator_ledger_receipt": second},
            ],
            "selected_checkpoints": [{"estimator_ledger_receipt": first}],
            "selected_terminal": {"estimator_ledger_receipt": terminal},
            "selected_round": 1,
            "selected_branch_id": 10,
            "target_round": 2,
            "ledger": {"raw_occurrence_count": 3, "all_branch_s_alg": 3},
        }

    def test_beam_aware_receipt_graph_closes(self):
        summary = beam_evidence_validation._validate_beam_receipts(**self.fixture())
        self.assertEqual(summary["materialized_branch_count"], 2)
        self.assertTrue(summary["closure_passed"])

    def test_orphaned_branch_and_wrong_selected_terminal_fail(self):
        fixture = self.fixture()
        fixture["adapt"]["continuation"][
            "all_active_prefix_estimator_ledger_receipts"
        ][1]["parent_branch_id"] = "missing"
        with self.assertRaises(ValueError):
            beam_evidence_validation._validate_beam_receipts(**fixture)
        fixture = self.fixture()
        fixture["selected_branch_id"] = 20
        with self.assertRaises(ValueError):
            beam_evidence_validation._validate_beam_receipts(**fixture)


if __name__ == "__main__":
    unittest.main()
