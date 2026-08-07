#!/usr/bin/env python3
import json
import unittest

import build_bundle
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


if __name__ == "__main__":
    unittest.main()
