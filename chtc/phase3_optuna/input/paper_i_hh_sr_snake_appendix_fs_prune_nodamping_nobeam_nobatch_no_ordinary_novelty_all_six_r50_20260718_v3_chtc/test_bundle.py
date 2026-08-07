#!/usr/bin/env python3
"""Fail-closed tests for the unsubmitted FS-pruning appendix bundle."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

import build_bundle as spec
import evidence_validation as evidence
BUNDLE = Path(__file__).resolve().parent
REPO = BUNDLE.parents[3]
REGIMES = {
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class PruneAppendixBundleTests(unittest.TestCase):
    def test_bundle_verifier_passes_and_submission_is_blocked(self) -> None:
        result = spec.verify_bundle()
        self.assertEqual(result["status"], "pass", result)
        self.assertTrue(spec.SOURCE_FREEZE_COMPLETE)
        self.assertFalse(spec.SUBMISSION_ENABLED)
        self.assertEqual(result["submission_requirements"], "False")
        self.assertTrue(result["remote_image_gate"]["passed"])
        submit = (BUNDLE / "submit.sub").read_text(encoding="utf-8")
        self.assertIn("requirements = False", submit)

    def test_remote_gate_and_explicit_switch_are_both_required(self) -> None:
        self.assertEqual(
            spec._submission_requirements(
                submission_enabled=False,
                remote_gate_passed=False,
            ),
            "False",
        )
        with self.assertRaisesRegex(
            RuntimeError, "remote_execution_gate.json passes"
        ):
            spec._submission_requirements(
                submission_enabled=True,
                remote_gate_passed=False,
            )

        with tempfile.TemporaryDirectory(prefix="sr-prune-remote-gate-") as td:
            root = Path(td)
            missing = spec._remote_execution_gate_status(root)
            self.assertFalse(missing["passed"])
            self.assertEqual(missing["reason"], "missing")

            gate_path = root / "remote_execution_gate.json"
            gate = {
                "schema": spec.REMOTE_EXECUTION_GATE_SCHEMA,
                "status": "pass",
                "remote_execution_preflight": {
                    "image_path": spec.REMOTE_IMAGE_PATH,
                    "image_sha256": "0" * 64,
                    "qiskit_import_passed": True,
                    "qiskit_version": spec.REMOTE_QISKIT_VERSION,
                    "fake_backend_instantiation_passed": True,
                    "fake_backend_resolved": spec.REMOTE_FAKE_BACKEND_RESOLVED,
                    "fake_backend_qubits": spec.REMOTE_FAKE_BACKEND_QUBITS,
                },
            }
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            invalid = spec._remote_execution_gate_status(root)
            self.assertFalse(invalid["passed"])
            self.assertFalse(invalid["checks"]["image_sha256"])

            gate["remote_execution_preflight"]["image_sha256"] = (
                spec.EXPECTED_IMAGE_SHA256
            )
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            valid = spec._remote_execution_gate_status(root)
            self.assertTrue(valid["passed"])
            self.assertEqual(
                spec._submission_requirements(
                    submission_enabled=False,
                    remote_gate_passed=True,
                ),
                "False",
            )
            self.assertEqual(
                spec._submission_requirements(
                    submission_enabled=True,
                    remote_gate_passed=True,
                ),
                "TARGET.HasSIF",
            )

            (root / "submit.sub").write_text(
                "universe = vanilla\nrequirements = False\nqueue\n",
                encoding="utf-8",
            )
            self.assertEqual(
                spec._write_submit_requirements(
                    root,
                    submission_enabled=True,
                    remote_gate_passed=True,
                ),
                "TARGET.HasSIF",
            )
            self.assertIn(
                "requirements = TARGET.HasSIF",
                (root / "submit.sub").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                spec._write_submit_requirements(
                    root,
                    submission_enabled=False,
                    remote_gate_passed=True,
                ),
                "False",
            )
            self.assertIn(
                "requirements = False",
                (root / "submit.sub").read_text(encoding="utf-8"),
            )

    def test_exact_source_archive_and_profile_probe(self) -> None:
        archive = BUNDLE / "source_locked.tar.gz"
        self.assertEqual(spec.sha256(archive), spec.SOURCE_ARCHIVE_SHA256)
        contract = spec.isolated_contract_probe()
        self.assertEqual(contract["route_profile"], spec.PROFILE_RESOLVED)
        self.assertEqual(spec.json_sha256(contract), spec.PROFILE_CONTRACT_SHA256)

    def test_six_fresh_round50_rows_are_same_cutoff(self) -> None:
        self.assertEqual({p.stem for p in (BUNDLE / "jobs").glob("*.json")}, REGIMES)
        for slug in REGIMES:
            job = load(BUNDLE / "jobs" / f"{slug}.json")
            normalized = load(BUNDLE / "normalized_manifests" / f"{slug}.json")
            for payload in (job, normalized):
                physics = payload["physics"]
                segment = payload["segment"]
                self.assertEqual(physics["n_ph_work"], physics["n_ph_reference"])
                self.assertTrue(physics["same_cutoff_reference"])
                self.assertEqual(segment["source_controller_round"], 0)
                self.assertEqual(segment["source_depth"], 0)
                self.assertEqual(segment["target_controller_round"], 50)
                self.assertEqual(segment["target_depth"], 50)
                self.assertFalse(
                    payload["sensitivity_audit"]["submission_authorized"]
                )

    def test_route_diff_is_prune_only_and_undamped(self) -> None:
        contract = spec.isolated_contract_probe()
        execution = contract["execution_settings"]
        semantics = contract["semantic_invariants"]
        self.assertTrue(execution["phase1_prune_enabled"])
        self.assertEqual(execution["phase1_prune_mode"], "live")
        self.assertEqual(execution["phase1_prune_local_window_size"], 0)
        self.assertEqual(execution["phase1_prune_recovery_trust_radius"], 0.125)
        self.assertEqual(execution["phase1_prune_metric_schur_mu"], 0.0)
        self.assertEqual(execution["phase1_prune_metric_mu_update_policy"], "off")
        self.assertEqual(execution["phase3_shadow_damping_policy"], "off")
        self.assertEqual(execution["adapt_beam_live_branches"], 1)
        self.assertFalse(execution["phase2_enable_batching"])
        self.assertFalse(execution["phase3_enable_batching"])
        self.assertTrue(semantics["pruning_active"])
        self.assertFalse(semantics["terminal_prune_active"])
        self.assertFalse(semantics["terminal_full_refit_active"])

        audit = load(BUNDLE / "scientific_settings_audit.json")
        self.assertEqual(
            set(audit["exact_settings_diff"]["changed_fields"]),
            set(spec.PRUNE_ONLY_CHANGED_FIELDS),
        )
        self.assertEqual(
            audit["exact_settings_diff"]["unexpected_non_prune_differences"],
            [],
        )

    def test_worker_validate_only_accepts_every_manifest(self) -> None:
        relative_bundle = BUNDLE.relative_to(REPO)
        with tempfile.TemporaryDirectory(prefix="sr-prune-archive-validate-") as td:
            root = Path(td)
            with tarfile.open(BUNDLE / "source_locked.tar.gz", "r:gz") as archive:
                archive.extractall(root, filter="data")
            staged_bundle = root / relative_bundle
            staged_bundle.mkdir(parents=True, exist_ok=True)
            for name in (
                "run_job.py",
                "evidence_validation.py",
                "source_archive_manifest.json",
                "source_revision_manifest.json",
                "physics_and_exact_reference_lock.json",
                "source_locked.tar.gz",
            ):
                shutil.copy2(BUNDLE / name, staged_bundle / name)
            shutil.copytree(BUNDLE / "jobs", staged_bundle / "jobs")
            env = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
            for path in sorted((staged_bundle / "jobs").glob("*.json")):
                completed = subprocess.run(
                    ["python3", str(staged_bundle / "run_job.py"), "--validate-only", str(path)],
                    cwd=root,
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_supported_rank_is_optional_only_for_exact_zero_query_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, "supported rank missing"):
            evidence.validate_supported_rank_or_exact_fallback(
                {
                    "all_energy_models_infeasible_novelty_fallback_fired": False,
                    "phase3_response_supported_rank": None,
                },
                pre_support=3,
                outer_iteration=4,
                require_supported_rank=True,
            )
        fallback = {
            "all_energy_models_infeasible_novelty_fallback_fired": True,
            "all_energy_models_infeasible_novelty_fallback_mode": (
                "collective_span_novelty_over_cost_v1"
            ),
            "all_energy_models_infeasible_novelty_fallback_reason": (
                "all_whitened_energy_models_infeasible"
            ),
            "all_energy_models_infeasible_novelty_fallback_query_charge": 0,
            "phase3_response_supported_rank": None,
        }
        receipt = evidence.validate_supported_rank_or_exact_fallback(
            fallback,
            pre_support=3,
            outer_iteration=4,
            require_supported_rank=True,
        )
        self.assertTrue(receipt["fallback_fired"])
        self.assertFalse(receipt["supported_rank_recorded"])
        for field, value in (
            ("all_energy_models_infeasible_novelty_fallback_mode", "wrong"),
            ("all_energy_models_infeasible_novelty_fallback_reason", "wrong"),
            ("all_energy_models_infeasible_novelty_fallback_query_charge", 1),
        ):
            malformed = dict(fallback)
            malformed[field] = value
            with self.assertRaisesRegex(ValueError, "fallback receipt"):
                evidence.validate_supported_rank_or_exact_fallback(
                    malformed,
                    pre_support=3,
                    outer_iteration=4,
                    require_supported_rank=True,
                )

    def test_missing_rank_service_modes_remain_scientific_contract_failures(self) -> None:
        for mode, outer_iteration in (
            ("phase2_raw", 27),
            ("eps_grad_suppressed_continue", 39),
        ):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                ValueError,
                f"selection_mode={mode}.*full-response-every-controller-round",
            ):
                evidence.validate_supported_rank_or_exact_fallback(
                    {
                        "selection_mode": mode,
                        "all_energy_models_infeasible_novelty_fallback_fired": False,
                        "phase3_response_supported_rank": None,
                    },
                    pre_support=4,
                    outer_iteration=outer_iteration,
                    require_supported_rank=True,
                )

    def test_active_prefix_receipts_close_to_complete_not_winning_work(self) -> None:
        components = ("N_H_outer", "N_H_refit", "N_grad", "N_metric")

        def component_payload(total: int) -> dict[str, int]:
            return {
                "N_H_outer": total,
                "N_H_refit": 0,
                "N_grad": 0,
                "N_metric": 0,
            }

        def receipt(
            *, sequence: int, kind: str, raw_delta: int, unique_delta: int,
            cumulative_raw: int, cumulative_unique: int,
        ) -> dict[str, object]:
            return {
                "schema": "paper_i_active_prefix_estimator_ledger_receipt_v1",
                "enabled": True,
                "status": "complete",
                "checkpoint_sequence": sequence,
                "outer_iteration": 1,
                "checkpoint_kind": kind,
                "canonical_same_state_deduplication_active": True,
                "raw_occurrences_preserved": True,
                "occurrence_sequence_start_exclusive": (
                    0 if sequence == 1 else cumulative_raw
                ),
                "raw_occurrence_delta": {
                    "components": component_payload(raw_delta),
                    "total": raw_delta,
                },
                "unique_primitive_delta": {
                    "components": component_payload(unique_delta),
                    "S_alg": unique_delta,
                },
                "cumulative_raw_occurrences": {
                    "components": component_payload(cumulative_raw),
                    "total": cumulative_raw,
                },
                "cumulative_unique_primitives": {
                    "components": component_payload(cumulative_unique),
                    "S_alg": cumulative_unique,
                },
            }

        first = receipt(
            sequence=1,
            kind="post_admission_prune",
            raw_delta=277793,
            unique_delta=220815,
            cumulative_raw=277793,
            cumulative_unique=220815,
        )
        terminal = receipt(
            sequence=2,
            kind="terminal_post_final_refit_and_prune",
            raw_delta=0,
            unique_delta=0,
            cumulative_raw=277793,
            cumulative_unique=220815,
        )
        closure = {
            "schema": "paper_i_active_prefix_estimator_ledger_closure_v1",
            "enabled": True,
            "status": "complete",
            "passed": True,
            "receipt_count": 2,
            "summed_raw_occurrences": {
                "components": component_payload(277793),
                "total": 277793,
            },
            "terminal_raw_occurrences": {
                "components": component_payload(277793),
                "total": 277793,
            },
            "summed_unique_primitives": {
                "components": component_payload(220815),
                "S_alg": 220815,
            },
            "terminal_unique_primitives": {
                "components": component_payload(220815),
                "S_alg": 220815,
            },
        }
        adapt = {
            "continuation": {
                "all_active_prefix_estimator_ledger_receipts": [first, terminal],
                "active_prefix_estimator_ledger_closure": closure,
            },
            "active_prefix_checkpoints": [
                {"estimator_ledger_receipt": first}
            ],
            "terminal_active_prefix_checkpoint": {
                "estimator_ledger_receipt": terminal
            },
        }
        summary = evidence.validate_active_prefix_estimator_receipts(
            adapt=adapt,
            ledger_summary={
                "raw_occurrence_count": 277793,
                "winning_lineage_s_alg": 217387,
                "all_branch_s_alg": 220815,
            },
            target_round=1,
        )
        self.assertEqual(summary["all_branch_S_alg"], 220815)
        self.assertEqual(summary["winning_lineage_S_alg"], 217387)
        self.assertEqual(summary["discarded_only_S_alg"], 3428)

        with self.assertRaisesRegex(ValueError, "do not close to exact ledger"):
            evidence.validate_active_prefix_estimator_receipts(
                adapt=adapt,
                ledger_summary={
                    "raw_occurrence_count": 277793,
                    "winning_lineage_s_alg": 220816,
                    "all_branch_s_alg": 220815,
                },
                target_round=1,
            )

    def test_live_prune_receipt_and_depth_accounting(self) -> None:
        closed = {
            "enabled": True,
            "prune_mode": "live",
            "affine_deletion_fs_trust_route_active": True,
            "candidate_count": 0,
            "probe_indices": [],
            "executed": False,
        }
        closed_summary = evidence.validate_live_prune_round(
            closed,
            outer_iteration=1,
        )
        self.assertFalse(closed_summary["executed"])
        self.assertEqual(
            evidence.validate_post_prune_depth(
                previous_active_depth=0,
                pre_support=1,
                checkpoint_depth=1,
                prune_summary=closed_summary,
                outer_iteration=1,
            ),
            1,
        )
        self.assertEqual(
            evidence.validate_post_prune_depth(
                previous_active_depth=5,
                pre_support=6,
                checkpoint_depth=5,
                prune_summary={"executed": True, "accepted": True},
                outer_iteration=6,
            ),
            5,
        )
        with self.assertRaisesRegex(ValueError, "post-prune active depth"):
            evidence.validate_post_prune_depth(
                previous_active_depth=5,
                pre_support=6,
                checkpoint_depth=6,
                prune_summary={"executed": True, "accepted": True},
                outer_iteration=6,
            )
        inactive = dict(closed)
        inactive["enabled"] = False
        with self.assertRaisesRegex(ValueError, "live pruning is disabled"):
            evidence.validate_live_prune_round(inactive, outer_iteration=1)


if __name__ == "__main__":
    unittest.main()
