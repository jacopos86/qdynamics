#!/usr/bin/env python3
"""Persist the completed v6 validation receipt and refresh artifact hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
INPUT_ROOT = ROOT / "chtc" / "phase3_optuna" / "input"
BUNDLES = (
    "paper_i_hh_sr_snake_macro_beam3x2_fs_prune_symmetric_cost_all_six_r50_20260719_v6_chtc",
    "paper_i_hh_sr_snake_macro_beam3x2_fs_prune_one_sided_cost_all_six_r50_20260719_v6_chtc",
)
SOURCE_SHA256 = "4c40399410b67b34a89f3cadeae59a0fd901c39132ff5cc746101c78e5acccd7"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _refresh_inventory(bundle: Path) -> None:
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path.name == "submission_artifact_hashes.json":
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = str(path.relative_to(ROOT))
        artifacts[relative] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    _dump(
        bundle / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_hh_sr_macro_beam_prune_cost_artifact_hashes_v1",
            "artifacts": artifacts,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-tests", type=int, required=True)
    parser.add_argument("--archive-rows", type=int, required=True)
    parser.add_argument("--focused-tests", type=int, required=True)
    parser.add_argument("--independent-validator-tests", type=int, required=True)
    args = parser.parse_args()
    for bundle_name in BUNDLES:
        bundle = INPUT_ROOT / bundle_name
        archive = bundle / "source_locked.tar.gz"
        if _sha256(archive) != SOURCE_SHA256:
            raise RuntimeError(f"source archive drift: {bundle_name}")
        preflight_path = bundle / "archive_only_preflight.json"
        preflight = json.loads(preflight_path.read_text())
        preflight.update(
            {
                "v6_bundle_tests_passed": args.bundle_tests,
                "v6_archive_only_validate_rows_passed": args.archive_rows,
                "v6_shared_archive_focused_tests_passed": args.focused_tests,
                "v6_independent_validator_fixture_tests_passed": (
                    args.independent_validator_tests
                ),
                "v6_fail_closed_review_status": "pass",
                "status": "pass",
            }
        )
        _dump(preflight_path, preflight)
        _dump(
            bundle / "v6_validation_receipt.json",
            {
                "schema": "paper_i_sr_macro_beam_cost_v6_validation_receipt_v1",
                "status": "pass",
                "bundle_id": bundle_name,
                "source_archive_sha256": SOURCE_SHA256,
                "bundle_tests_passed": args.bundle_tests,
                "archive_only_validate_rows_passed": args.archive_rows,
                "shared_archive_focused_tests_passed": args.focused_tests,
                "independent_validator_fixture_tests_passed": (
                    args.independent_validator_tests
                ),
                "commands": {
                    "bundle": (
                        "PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v "
                        "test_bundle.py test_beam_validation.py"
                    ),
                    "archive_only": "run_job.py --validate-only for all six job manifests",
                    "focused": (
                        "PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q "
                        "test/test_static_adapt_sr_v4_runtime.py "
                        "test/test_static_adapt_macro_beam_prune_cost_profiles.py"
                    ),
                },
            },
        )
        _refresh_inventory(bundle)
        print(bundle)


if __name__ == "__main__":
    main()
