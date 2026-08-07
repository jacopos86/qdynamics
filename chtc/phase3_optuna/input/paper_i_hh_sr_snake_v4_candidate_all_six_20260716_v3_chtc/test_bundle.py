#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
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
    "weak_weak": (0.25, 3, -0.918380919994822),
    "intermediate_weak": (1.25, 3, -0.4950053491813613),
    "strong_weak_u8": (8.0, 3, 0.5264586847939736),
    "weak_strong": (0.25, 7, -1.1387206380749124),
    "intermediate_strong": (1.25, 7, -0.6239396137518493),
    "strong_strong_u8": (8.0, 7, 0.5205762765682517),
}
PROFILE_DIGEST = "b6331521fb55f4165e177466536b4e2a5834ff09205ab5532ea70de893f156bc"
PHASE1_ENERGY_MODEL = "first_order_fs_trust_v1"
PHASE2_CURVATURE_POLICY = "measured_required_fail_closed_v1"
PHASE2_CHEAP_CURVATURE_PROXY_POLICY = "off"
EXPECTED_HEAD = "6b1a6c619998bb397e382c78c3a9c77572aa4ede"
EXPECTED_TREE = "1eda6eb868f5dd069d484aef9bbfa4c5d50dcd70"
NONSCIENTIFIC_ARCHIVE_OVERLAYS = {
    "pipelines/hardcoded/adapt_pipeline.py": (
        "93c0e91cd01981f5bfa1e9d1434b74296395ca0837f2901ca66ad18ac63dd42f"
    ),
    "pipelines/hardcoded/hh_continuation_scoring.py": (
        "f25b2ae3f4037758c5f1942e6e3e0c75df04f9c5c7008d8b8dacfa9f150aa492"
    ),
    "pipelines/hardcoded/hh_continuation_generators.py": (
        "8c6292c5c71f67312bdc32afc8d3908a83cb550b8f2d8871ed7f7183824e6570"
    ),
    "pipelines/hardcoded/hh_continuation_symmetry.py": (
        "5f61b9c43c253fb81bc354aace4e015f0c4f06a1e8aa8a48b24a43a11b341e01"
    ),
    "pipelines/hardcoded/hh_continuation_types.py": (
        "f24b1a670179ec17c05132b3b65f9541db54ffd888429951f7e17d6aaaf41f4c"
    ),
}
NONSCIENTIFIC_ARCHIVE_OVERLAY_SIZES = {
    "pipelines/hardcoded/adapt_pipeline.py": 1807,
    "pipelines/hardcoded/hh_continuation_scoring.py": 658,
    "pipelines/hardcoded/hh_continuation_generators.py": 664,
    "pipelines/hardcoded/hh_continuation_symmetry.py": 668,
    "pipelines/hardcoded/hh_continuation_types.py": 654,
}
BUILT_BUNDLE = (BUNDLE / "source_locked.tar.gz").is_file()


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


def isolated_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    home = root / "home"
    home.mkdir(exist_ok=True)
    env.update({
        "HOME": str(home),
        "PYTHONPATH": str(root),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


class ScaffoldTests(unittest.TestCase):
    def test_scaffold_is_fail_closed_and_names_corrected_contract(self) -> None:
        builder = (BUNDLE / "build_bundle.py").read_text(encoding="utf-8")
        submit = (BUNDLE / "submit.sub").read_text(encoding="utf-8")
        self.assertIn(PHASE1_ENERGY_MODEL, builder)
        self.assertIn(PHASE2_CURVATURE_POLICY, builder)
        self.assertIn(PHASE2_CHEAP_CURVATURE_PROXY_POLICY, builder)
        self.assertIn(PROFILE_DIGEST, builder)
        for relative, digest in NONSCIENTIFIC_ARCHIVE_OVERLAYS.items():
            self.assertIn(relative, builder)
            self.assertIn(digest, builder)
        if not BUILT_BUNDLE:
            self.assertIn('EXPECTED_HEAD = "6b1a6c619998bb397e382c78c3a9c77572aa4ede"', builder)
            self.assertIn('EXPECTED_TREE = "1eda6eb868f5dd069d484aef9bbfa4c5d50dcd70"', builder)
            self.assertIn("SUBMISSION_ENABLED = False", builder)
            self.assertIn("requirements = False", submit)
            self.assertIn("queue 0", submit)
            self.assertFalse((BUNDLE / "jobs").exists())
        else:
            self.assertNotIn("PENDING_MAIN_AGENT_CONFIRMED", builder)

    def test_validators_enforce_phase12_receipts_and_zero_proxy_counts(self) -> None:
        evidence = (BUNDLE / "evidence_validation.py").read_text(encoding="utf-8")
        runner = (BUNDLE / "run_job.py").read_text(encoding="utf-8")
        for token in (
            PHASE1_ENERGY_MODEL,
            PHASE2_CURVATURE_POLICY,
            "phase2_full_candidate_occurrences",
            "validated_phase2_curvature_receipt_occurrences",
            "phase1_lambda_f_proxy_occurrences",
            "phase2_lambda_f_proxy_occurrences",
            "phase2_missing_curvature_fallback_occurrences",
            "sr_snake_phase2_directional_curvature_receipt_v1",
            "sr_snake_phase2_directional_curvature_provenance_v1",
        ):
            self.assertIn(token, evidence)
        self.assertIn("--phase1-lambda-F", runner)
        self.assertIn("--phase2-lambda-F", runner)


@unittest.skipUnless(BUILT_BUNDLE, "v3 source authority has not been frozen")
class BundleTests(unittest.TestCase):
    def test_six_exact_parent_jobs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sr_v4_archive_only_") as tmp:
            isolated_repo = Path(tmp)
            isolated_bundle = extract_archive_only_repo(isolated_repo)
            for slug, (u, cutoff, exact) in EXPECTED.items():
                job = load(isolated_bundle / "jobs" / f"{slug}.json")
                self.assertEqual(job["route_identity"]["profile_request"], "sr_snake_v4")
                self.assertEqual(job["route_identity"]["profile_contract_sha256"], PROFILE_DIGEST)
                contract = job["route_identity"]["profile_contract"]
                phase12 = job["route_identity"]["phase12_energy_model_contract"]
                self.assertEqual(contract["execution_settings"]["phase1_energy_model"], PHASE1_ENERGY_MODEL)
                self.assertEqual(contract["execution_settings"]["phase2_curvature_policy"], PHASE2_CURVATURE_POLICY)
                self.assertEqual(contract["execution_settings"]["phase2_cheap_curvature_proxy_policy"], PHASE2_CHEAP_CURVATURE_PROXY_POLICY)
                self.assertFalse(contract["semantic_invariants"]["phase1_phase2_lambda_f_proxy_active"])
                self.assertTrue(phase12["lambda_f_proxy_flags_forbidden"])
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
                    env=isolated_env(isolated_repo),
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

        phase1_proxy = json.loads(json.dumps(baseline))
        phase1_proxy["command"]["argv"][-1:-1] = ["--phase1-lambda-F", "1.0"]
        mutations.append(("phase1_lambda_f_proxy", phase1_proxy))

        phase2_proxy = json.loads(json.dumps(baseline))
        phase2_proxy["route_identity"]["phase12_energy_model_contract"][
            "phase2_cheap_curvature_proxy_policy"
        ] = "lambda_f_fallback_v1"
        mutations.append(("phase2_cheap_proxy", phase2_proxy))

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
                    env=isolated_env(isolated_repo),
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
        self.assertEqual(manifest["git_commit"], EXPECTED_HEAD)
        self.assertEqual(manifest["git_tree"], EXPECTED_TREE)
        self.assertEqual(
            manifest["worker_source_mode"],
            "exact_git_archive_plus_hashed_nonscientific_overlays_v1",
        )
        expected_overlay_records = {
            relative: {
                "sha256": digest,
                "size_bytes": NONSCIENTIFIC_ARCHIVE_OVERLAY_SIZES[relative],
                "mode": "0644",
                "classification": "compatibility_import_shim_only",
                "tracked_in_frozen_commit": False,
            }
            for relative, digest in NONSCIENTIFIC_ARCHIVE_OVERLAYS.items()
        }
        self.assertEqual(
            source_revision["non_scientific_archive_overlays"],
            expected_overlay_records,
        )
        self.assertEqual(
            manifest["non_scientific_archive_overlays"],
            expected_overlay_records,
        )
        for relative, digest in NONSCIENTIFIC_ARCHIVE_OVERLAYS.items():
            self.assertEqual(manifest["files"][relative]["sha256"], digest)
            self.assertEqual(
                manifest["files"][relative]["size_bytes"],
                NONSCIENTIFIC_ARCHIVE_OVERLAY_SIZES[relative],
            )
        for relative in (
            "pipelines/exact_bench/table_i_qiskit_resource_compile.py",
            "pipelines/exact_bench/paper_i_hh_recovery_prefix_qiskit_sidecar.py",
            "pipelines/static_adapt/sr_snake_phase12_policy.py",
            "test/test_hh_continuation_scoring.py",
        ):
            self.assertEqual(
                manifest["files"][relative]["sha256"],
                source_revision["critical_source_sha256"][relative],
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

    def test_submit_transfers_qiskit_sidecars_and_matches_submission_gate(self) -> None:
        submit = (BUNDLE / "submit.sub").read_text(encoding="utf-8")
        self.assertIn("request_cpus = 4", submit)
        self.assertIn("evidence_validation.py", submit)
        self.assertIn("transfer_output_remaps", submit)
        self.assertIn("$(regime_slug)_transfer.tar.gz", submit)
        self.assertIn("qiskit", (BUNDLE / "run_job.py").read_text().lower())
        self.assertIn("validate_parent_evidence", (BUNDLE / "run_job.py").read_text())
        self.assertIn("validate_parent_evidence", (BUNDLE / "validate_fetched.py").read_text())
        preflight = load(BUNDLE / "preflight.json")
        submission_enabled = preflight["checks"]["submission_enabled"]
        if submission_enabled:
            self.assertIn("requirements = TARGET.HasSIF", submit)
            self.assertNotIn("requirements = False", submit)
            self.assertEqual(
                preflight["status"], "pass_submission_ready_not_yet_submitted"
            )
            self.assertEqual(preflight["submission_blockers"], [])
            self.assertTrue(preflight["submission_authorized"])
        else:
            self.assertIn("requirements = False", submit)
            self.assertEqual(
                preflight["status"], "pass_bundle_built_submission_blocked"
            )
            self.assertTrue(preflight["submission_blockers"])
            self.assertFalse(preflight["submission_authorized"])
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
        self.assertTrue(preflight["checks"]["phase1_first_order_fs_trust_policy"])
        self.assertTrue(preflight["checks"]["phase2_measured_curvature_required_fail_closed_policy"])
        self.assertTrue(preflight["checks"]["phase2_cheap_curvature_proxy_off"])
        self.assertTrue(preflight["checks"]["phase1_phase2_lambda_f_proxy_inactive"])
        self.assertTrue(preflight["checks"]["smoke_phase2_curvature_receipt_count_closure"])
        self.assertTrue(preflight["checks"]["smoke_lambda_f_proxy_occurrences_zero"])
        self.assertTrue(preflight["checks"]["smoke_missing_curvature_fallback_occurrences_zero"])
        self.assertTrue(
            preflight["checks"][
                "exact_commit_plus_hashed_nonscientific_overlays"
            ]
        )
        self.assertTrue(
            preflight["checks"]["compatibility_overlay_archive_hashes_closed"]
        )
        self.assertTrue(
            preflight["checks"]["isolated_archive_compatibility_aliases"]
        )
        self.assertTrue(
            preflight["checks"]["isolated_archive_live_repo_import_excluded"]
        )
        self.assertEqual(preflight["scientific_blockers"], [])
        self.assertFalse(preflight["submission_performed"])
        if submission_enabled:
            self.assertTrue(preflight["checks"]["remote_image_sha256_rechecked"])
            self.assertTrue(preflight["checks"]["remote_qiskit_import_passed"])
            self.assertTrue(
                preflight["checks"]["remote_fake_marrakesh_instantiation_passed"]
            )
        self.assertIn("qiskit_sidecar_helper_in_source_archive", preflight["checks"])

    def test_remote_preflight_and_cleanup_receipt(self) -> None:
        receipt = load(BUNDLE / "remote_preflight_and_cleanup_receipt.json")
        self.assertIn(
            receipt["status"], {"pass", "blocked_pending_remote_preflight"}
        )
        remote = receipt["remote_execution_preflight"]
        self.assertEqual(
            remote["image_sha256"],
            "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f",
        )
        if receipt["status"] == "pass":
            self.assertEqual(remote["qiskit_version"], "2.3.1")
            self.assertEqual(remote["fake_backend_resolved"], "fake_marrakesh")
            self.assertEqual(remote["fake_backend_qubits"], 156)
        cleanup = receipt["storage_cleanup"]
        self.assertFalse(cleanup["unrelated_remote_paths_modified"])
        self.assertEqual(cleanup["remote_removed_paths"], [])
        self.assertFalse(receipt["submission_performed"])

    def test_archive_only_preflight_and_relative_parity(self) -> None:
        isolated = load(BUNDLE / "archive_only_preflight.json")
        self.assertEqual(isolated["status"], "pass")
        self.assertEqual(isolated["source_import"]["status"], "pass")
        self.assertTrue(isolated["live_repo_import_excluded"])
        source_import = isolated["source_import"]
        self.assertTrue(source_import["adapt_compatibility_alias_is_archived_target"])
        self.assertTrue(source_import["scoring_compatibility_alias_is_archived_target"])
        self.assertEqual(
            source_import["overlay_hashes"], NONSCIENTIFIC_ARCHIVE_OVERLAYS
        )
        self.assertEqual(source_import["project_modules_outside_archive"], [])
        self.assertEqual(source_import["live_repo_sys_path_entries"], 0)
        self.assertTrue(isolated["all_six_validate_only_pass"])
        self.assertEqual(len(isolated["six_validate_only_parses"]), 6)
        self.assertTrue(isolated["qiskit_helper"]["help_pass"])
        self.assertTrue(isolated["focused_source_locked_regressions"]["pass"])
        self.assertEqual(
            isolated["focused_source_locked_regressions"]["returncode"], 0
        )
        for row in isolated["six_validate_only_parses"]:
            self.assertTrue(str(row["job"]).startswith("jobs/"))
            self.assertNotIn("/Users/", json.dumps(row, sort_keys=True))
        for name in ("route_parity.json", "bundle_manifest.json"):
            self.assertNotIn("/Users/", (BUNDLE / name).read_text(encoding="utf-8"))

    def test_v2_to_v3_diff_contains_only_phase12_contract_change(self) -> None:
        audit = load(BUNDLE / "v2_to_v3_scientific_settings_diff.json")
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["unexpected_executable_differences"], [])
        observed = {row["path"] for row in audit["observed_contract_diff"]}
        self.assertIn("execution_settings.phase1_energy_model", observed)
        self.assertIn("execution_settings.phase2_curvature_policy", observed)
        self.assertIn(
            "execution_settings.phase2_cheap_curvature_proxy_policy", observed
        )
        self.assertIn(
            "semantic_invariants.phase1_phase2_lambda_f_proxy_active", observed
        )
        self.assertEqual(len(audit["regime_checks"]), 6)
        for row in audit["regime_checks"]:
            self.assertTrue(all(row["checks"].values()), row["regime_slug"])

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
            and ".pytest_cache" not in path.parts
            and path.suffix != ".pyc"
        }
        self.assertEqual(set(inventory["artifacts"]), actual)
        for relative, record in inventory["artifacts"].items():
            path = REPO / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(sha256(path), record["sha256"], relative)
            self.assertNotIn("__pycache__", relative)
            self.assertNotIn(".pytest_cache", relative)
            self.assertFalse(relative.endswith(".pyc"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
