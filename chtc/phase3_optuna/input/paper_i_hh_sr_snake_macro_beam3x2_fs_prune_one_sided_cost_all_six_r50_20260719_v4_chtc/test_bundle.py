#!/usr/bin/env python3
import hashlib
import json
import tarfile
import unittest

import build_bundle
import run_job
import validate_fetched


class BundleTest(unittest.TestCase):
    def test_immutable_pool_complement_bundle(self):
        self.assertTrue(build_bundle.verify())

    def test_fetched_validator_matches_immutable_source_contract(self):
        revision = json.loads(
            (build_bundle.BUNDLE_DIR / "source_revision_manifest.json").read_text()
        )
        archive = json.loads(
            (build_bundle.BUNDLE_DIR / "source_archive_manifest.json").read_text()
        )
        self.assertEqual(validate_fetched.PROFILE, build_bundle.PROFILE)
        self.assertIs(revision["dirty_live_source_lock"], True)
        self.assertEqual(
            revision["executable_source_authority"],
            validate_fetched.SOURCE_REVISION_EXECUTABLE_AUTHORITY,
        )
        self.assertEqual(
            archive["executable_source_authority"],
            validate_fetched.SOURCE_ARCHIVE_EXECUTABLE_AUTHORITY,
        )
        self.assertEqual(
            archive["worker_source_mode"], validate_fetched.WORKER_SOURCE_MODE
        )
        expected_modules = {
            relative: {
                "sha256": digest,
                "classification": "required_executable_live_source_module",
                "tracked_in_base_commit": False,
            }
            for relative, digest in (
                validate_fetched.REQUIRED_UNTRACKED_SOURCE_MODULES.items()
            )
        }
        self.assertEqual(
            revision["required_untracked_source_modules"], expected_modules
        )
        self.assertEqual(
            archive["required_untracked_source_modules"], expected_modules
        )


    def test_fidelity_source_hashes_match_complete_archive_inventory(self):
        revision = json.loads(
            (build_bundle.BUNDLE_DIR / "source_revision_manifest.json").read_text()
        )
        archive = json.loads(
            (build_bundle.BUNDLE_DIR / "source_archive_manifest.json").read_text()
        )
        expected = validate_fetched.REQUIRED_HASH_LOCKED_FIDELITY_FILES
        self.assertEqual(run_job.REQUIRED_HASH_LOCKED_FIDELITY_FILES, expected)
        for manifest in (revision, archive):
            records = manifest["required_hash_locked_fidelity_files"]
            self.assertEqual(
                {relative: record["sha256"] for relative, record in records.items()},
                expected,
            )
            overlays = manifest["required_untracked_hash_overlays"]
            self.assertEqual(
                {relative: record["sha256"] for relative, record in overlays.items()},
                {relative: expected[relative] for relative in overlays},
            )
        inventory = archive["files"]
        for relative, expected_hash in expected.items():
            self.assertEqual(inventory[relative]["sha256"], expected_hash)
        for kind in ("jobs", "normalized_manifests"):
            for path in sorted((build_bundle.BUNDLE_DIR / kind).glob("*.json")):
                source_lock = json.loads(path.read_text())["source_lock"]
                self.assertEqual(
                    source_lock["required_hash_locked_fidelity_files"],
                    archive["required_hash_locked_fidelity_files"],
                )
                self.assertEqual(
                    source_lock["required_untracked_hash_overlays"],
                    archive["required_untracked_hash_overlays"],
                )
        with tarfile.open(build_bundle.BUNDLE_DIR / "source_locked.tar.gz", "r:gz") as handle:
            for relative, expected_hash in expected.items():
                member = handle.extractfile(relative)
                self.assertIsNotNone(member)
                self.assertEqual(hashlib.sha256(member.read()).hexdigest(), expected_hash)

    def test_prune_trial_consumer_id_is_parent_beam_scoped(self):
        self.assertTrue(build_bundle.verify_prune_consumer_repair())


if __name__ == "__main__":
    unittest.main()
