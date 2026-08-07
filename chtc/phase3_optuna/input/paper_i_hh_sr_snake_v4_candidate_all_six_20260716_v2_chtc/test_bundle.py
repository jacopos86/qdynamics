#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import subprocess
import sys
import tarfile
import unittest
from pathlib import Path, PurePosixPath


BUNDLE = Path(__file__).resolve().parent
REPO = BUNDLE.parents[3]
EXPECTED = {
    "weak_weak": (0.25, 2, -0.9183531194991743),
    "intermediate_weak": (1.25, 2, -0.4949956391086595),
    "strong_weak_u8": (8.0, 2, 0.5264587007998427),
    "weak_strong": (0.25, 4, -1.1385792003592516),
    "intermediate_strong": (1.25, 4, -0.6239104048313423),
    "strong_strong_u8": (8.0, 4, 0.5205762777107107),
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive_only_repo(destination: Path) -> Path:
    """Materialize the frozen source and bundle without using the live checkout."""

    archive = BUNDLE / "source_locked.tar.gz"
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or member.issym()
                or member.islnk()
            ):
                raise ValueError(f"unsafe archive member: {member.name}")
        handle.extractall(destination, members=members)
    isolated_bundle = destination / BUNDLE.relative_to(REPO)
    isolated_bundle.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BUNDLE, isolated_bundle)
    return isolated_bundle


class BundleTests(unittest.TestCase):
    def test_six_exact_parent_jobs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sr_v4_archive_only_") as tmp:
            isolated_repo = Path(tmp)
            isolated_bundle = extract_archive_only_repo(isolated_repo)
            for slug, (u, cutoff, exact) in EXPECTED.items():
                job = load(isolated_bundle / "jobs" / f"{slug}.json")
                self.assertEqual(job["route_identity"]["profile_request"], "sr_snake_v4")
                self.assertEqual(job["route_identity"]["profile_contract_sha256"], "7f99682678c1c338ad73142e604c2cc6757dbb4db311a3b0bb8cb7b23e8436be")
                contract = job["route_identity"]["profile_contract"]
                self.assertFalse(
                    contract["execution_settings"]["adapt_finite_angle_fallback"]
                )
                self.assertFalse(
                    contract["semantic_invariants"]["finite_angle_fallback_active"]
                )
                self.assertFalse(contract["execution_settings"]["phase3_enable_rescue"])
                self.assertEqual(job["segment"]["source_controller_round"], 0)
                self.assertEqual(job["segment"]["target_controller_round"], 30)
                self.assertEqual(job["segment"]["target_depth"], 30)
                self.assertEqual(job["segment"]["max_new_admissions"], 30)
                self.assertEqual(job["physics"]["u_over_t"], u)
                self.assertEqual(job["physics"]["n_ph_work"], cutoff)
                self.assertEqual(job["physics"]["n_ph_reference"], cutoff)
                self.assertEqual(job["physics"]["expected_exact_energy"], exact)
                argv = job["command"]["argv"]
                self.assertNotIn("--adapt-max-depth", argv)
                self.assertNotIn("--adapt-exact-gs-override", argv)
                normalized = load(
                    isolated_bundle / "normalized_manifests" / f"{slug}.json"
                )
                normalized_contract = normalized["route_identity"]["profile_contract"]
                self.assertFalse(
                    normalized_contract["execution_settings"][
                        "adapt_finite_angle_fallback"
                    ]
                )
                self.assertFalse(
                    normalized_contract["execution_settings"]["phase3_enable_rescue"]
                )
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(isolated_bundle / "run_job.py"),
                        "--validate-only",
                        str(isolated_bundle / "jobs" / f"{slug}.json"),
                    ],
                    cwd=isolated_repo,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                validate_payload = json.loads(completed.stdout)
                self.assertEqual(validate_payload["job"], f"jobs/{slug}.json")
                self.assertNotIn("/Users/", completed.stdout)

    def test_route_guard_mutations_fail_closed(self) -> None:
        baseline = load(BUNDLE / "jobs" / "weak_weak.json")
        mutations = []

        finite_angle = json.loads(json.dumps(baseline))
        finite_angle["route_identity"]["profile_contract"]["execution_settings"][
            "adapt_finite_angle_fallback"
        ] = True
        mutations.append(("finite_angle", finite_angle))

        rescue = json.loads(json.dumps(baseline))
        rescue["route_identity"]["profile_contract"]["execution_settings"][
            "phase3_enable_rescue"
        ] = True
        mutations.append(("phase3_rescue", rescue))

        oracle = json.loads(json.dumps(baseline))
        oracle["command"]["argv"][-1:-1] = [
            "--phase3-oracle-gradient-mode",
            "ideal",
        ]
        mutations.append(("phase3_oracle_gradient", oracle))

        with tempfile.TemporaryDirectory(prefix="sr_v4_bundle_guard_") as tmp:
            isolated_repo = Path(tmp)
            isolated_bundle = extract_archive_only_repo(isolated_repo)
            for label, payload in mutations:
                path = isolated_repo / f"{label}.json"
                path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(isolated_bundle / "run_job.py"),
                        "--validate-only",
                        str(path),
                    ],
                    cwd=isolated_repo,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0, label)

    def test_archive_is_safe_exact_commit(self) -> None:
        manifest = load(BUNDLE / "source_archive_manifest.json")
        archive = BUNDLE / "source_locked.tar.gz"
        self.assertEqual(sha256(archive), manifest["archive_sha256"])
        source_revision = load(BUNDLE / "source_revision_manifest.json")
        self.assertEqual(manifest["git_commit"], source_revision["git_commit"])
        self.assertEqual(
            manifest["files"][
                "pipelines/exact_bench/table_i_qiskit_resource_compile.py"
            ]["sha256"],
            "cdc182772288593de6087049470a8b6bb47a00c254cd6176276eda63320d19cd",
        )
        self.assertEqual(
            manifest["files"][
                "pipelines/exact_bench/paper_i_hh_recovery_prefix_qiskit_sidecar.py"
            ]["sha256"],
            "5486c285deffcb47fd0f5ef0314a9e3ab2fd1c83ebb7e0bb72d629d6a81dd044",
        )
        self.assertNotIn(
            "test/test_static_adapt_accepted_refit.py", manifest["files"]
        )
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle.getmembers():
                name = PurePosixPath(member.name)
                self.assertFalse(name.is_absolute())
                self.assertNotIn("..", name.parts)
                self.assertFalse(member.issym() or member.islnk())
                self.assertTrue(member.isfile() or member.isdir())
                self.assertFalse(any(part.startswith("._") for part in name.parts))
                self.assertNotIn("__MACOSX", name.parts)
                self.assertNotIn(".DS_Store", name.parts)

    def test_submit_transfers_qiskit_sidecars_and_is_submission_ready(self) -> None:
        submit = (BUNDLE / "submit.sub").read_text(encoding="utf-8")
        self.assertIn("request_cpus = 4", submit)
        self.assertIn("evidence_validation.py", submit)
        self.assertIn("transfer_output_remaps", submit)
        self.assertIn("$(regime_slug)_transfer.tar.gz", submit)
        self.assertIn("requirements = TARGET.HasSIF", submit)
        self.assertNotIn("requirements = False", submit)
        self.assertIn("qiskit", (BUNDLE / "run_job.py").read_text().lower())
        self.assertIn("validate_parent_evidence", (BUNDLE / "run_job.py").read_text())
        self.assertIn("validate_parent_evidence", (BUNDLE / "validate_fetched.py").read_text())
        preflight = load(BUNDLE / "preflight.json")
        self.assertEqual(
            preflight["status"],
            "pass_submission_ready_not_yet_submitted",
        )
        self.assertTrue(preflight["checks"]["phase3_response_supported_rank_recorded"])
        self.assertFalse(preflight["checks"]["shadow_damping_scientific_application_expected"])
        self.assertTrue(preflight["checks"]["shadow_damping_diagnostic_noop_receipt_recorded"])
        self.assertTrue(
            preflight["checks"][
                "production_composition_delete_refit_prune_regression_passed"
            ]
        )
        self.assertTrue(preflight["checks"]["finite_angle_fallback_disabled"])
        self.assertTrue(preflight["checks"]["phase3_rescue_disabled"])
        self.assertTrue(preflight["checks"]["phase3_oracle_gradient_mode_off"])
        self.assertEqual(preflight["scientific_blockers"], [])
        self.assertEqual(preflight["submission_blockers"], [])
        self.assertTrue(preflight["submission_authorized"])
        self.assertFalse(preflight["submission_performed"])
        self.assertTrue(preflight["checks"]["submission_enabled"])
        self.assertTrue(preflight["checks"]["remote_image_sha256_rechecked"])
        self.assertTrue(preflight["checks"]["remote_qiskit_import_passed"])
        self.assertTrue(
            preflight["checks"]["remote_fake_marrakesh_instantiation_passed"]
        )
        self.assertIn("qiskit_sidecar_helper_in_source_archive", preflight["checks"])

    def test_remote_preflight_and_cleanup_receipt(self) -> None:
        receipt = load(BUNDLE / "remote_preflight_and_cleanup_receipt.json")
        self.assertEqual(receipt["status"], "pass")
        remote = receipt["remote_execution_preflight"]
        self.assertEqual(
            remote["image_sha256"],
            "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f",
        )
        self.assertEqual(remote["qiskit_version"], "2.3.1")
        self.assertEqual(remote["fake_backend_resolved"], "fake_marrakesh")
        self.assertEqual(remote["fake_backend_qubits"], 156)
        cleanup = receipt["storage_cleanup"]
        self.assertEqual(cleanup["remote_absence_check"], "CLEANUP_OK")
        self.assertTrue(cleanup["local_preservation_verified"])
        self.assertFalse(cleanup["unrelated_remote_paths_modified"])
        self.assertEqual(len(cleanup["remote_removed_paths"]), 2)
        self.assertEqual(len(cleanup["local_validation_records"]), 2)
        self.assertFalse(receipt["submission_performed"])

    def test_archive_only_preflight_and_relative_parity(self) -> None:
        isolated = load(BUNDLE / "archive_only_preflight.json")
        self.assertEqual(isolated["status"], "pass")
        self.assertEqual(isolated["source_import"]["status"], "pass")
        self.assertTrue(isolated["live_repo_import_excluded"])
        self.assertTrue(isolated["all_six_validate_only_pass"])
        self.assertEqual(len(isolated["six_validate_only_parses"]), 6)
        self.assertTrue(isolated["qiskit_helper"]["help_pass"])
        self.assertTrue(isolated["focused_source_locked_regressions"]["pass"])
        self.assertEqual(
            isolated["focused_source_locked_regressions"]["expected_pass_count"], 45
        )
        for row in isolated["six_validate_only_parses"]:
            self.assertTrue(str(row["job"]).startswith("jobs/"))
            self.assertNotIn("/Users/", json.dumps(row, sort_keys=True))
        for name in ("route_parity.json", "bundle_manifest.json"):
            self.assertNotIn("/Users/", (BUNDLE / name).read_text(encoding="utf-8"))

    def test_future_continuation_is_template_only(self) -> None:
        payload = load(BUNDLE / "future_round30_to_round50_continuation_template.json")
        self.assertEqual(payload["status"], "template_only_not_executable")
        self.assertEqual(payload["eligible_regimes"], [
            "strong_weak_u8", "weak_strong", "intermediate_strong", "strong_strong_u8",
        ])
        segment = payload["segment_contract"]
        self.assertEqual(segment["source_controller_round"], 30)
        self.assertEqual(segment["target_controller_round"], 50)
        self.assertEqual(segment["max_new_admissions"], 20)
        self.assertEqual(segment["source_depth"], "from_authenticated_round30_signed_checkpoint")
        self.assertEqual(segment["boundary_policy"], "verified_checkpoint_no_refit_v1")
        self.assertTrue(payload["compile_smoke"]["required"])
        self.assertEqual(payload["compile_smoke"]["backend"], "FakeMarrakesh")
        for row in payload["rows"]:
            self.assertFalse(row["materialized"])
            self.assertIsNone(row["source_result_sha256"])
            self.assertIsNone(row["source_signed_checkpoint_sha256"])
            self.assertIsNone(row["source_estimator_ledger_sha256"])

    def test_hash_inventory_closes(self) -> None:
        inventory = load(BUNDLE / "submission_artifact_hashes.json")
        actual = {
            path.relative_to(REPO).as_posix()
            for path in BUNDLE.rglob("*")
            if path.is_file()
            and path.name != "submission_artifact_hashes.json"
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        }
        self.assertEqual(set(inventory["artifacts"]), actual)
        for relative, record in inventory["artifacts"].items():
            path = REPO / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(sha256(path), record["sha256"], relative)
            self.assertNotIn("__pycache__", relative)
            self.assertFalse(relative.endswith(".pyc"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
