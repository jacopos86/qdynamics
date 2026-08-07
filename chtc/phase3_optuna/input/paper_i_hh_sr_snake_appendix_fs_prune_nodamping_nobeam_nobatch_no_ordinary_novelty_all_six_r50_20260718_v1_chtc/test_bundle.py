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
    def test_bundle_verifier_passes_and_submission_is_enabled(self) -> None:
        result = spec.verify_bundle()
        self.assertEqual(result["status"], "pass", result)
        self.assertTrue(spec.SOURCE_FREEZE_COMPLETE)
        self.assertTrue(spec.SUBMISSION_ENABLED)
        self.assertEqual(result["submission_requirements"], "TARGET.HasSIF")
        self.assertTrue(result["remote_image_gate"]["passed"])
        submit = (BUNDLE / "submit.sub").read_text(encoding="utf-8")
        self.assertIn("requirements = TARGET.HasSIF", submit)

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


if __name__ == "__main__":
    unittest.main()
