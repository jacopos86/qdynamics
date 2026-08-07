#!/usr/bin/env python3
"""Build Test-2 v14 with the immutable-keep receipt fingerprint repair."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import build_paper_i_hh_sr_material_window_fs_prune_verify_successor_v13_20260722 as previous


core = previous.core
BASE_ID = previous.OUTPUT_ID
BASE_BATCH = previous.OUTPUT_BATCH
BASE = core.INPUT / BASE_ID
OUTPUT_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_fs_prune_verify_"
    "all_six_r50_20260722_v14_chtc"
)
OUTPUT_BATCH = "paper-i-hh-sr-material-window-fsprune-verify-six-r50-20260722-v14"
OUTPUT = core.INPUT / OUTPUT_ID
BASE_SOURCE_SHA256 = "242c26fe2d92ba433f68963bf38593ebb681f7ab4dc5ba8e8d484a5a0fbc1eca"
BASE_ADAPT_SHA256 = "1a9c54dbd993b898fc124b11932e6394b9f34d2d10282cfb35477d2051afaa20"
BASE_TEST_SHA256 = "607312df851a14a1e91457f2c84fdc26d68f46b476c119a5ec67c1ced376da23"

OLD_FINGERPRINT = '''                    "theta_fingerprint": _array_fingerprint(
                        keep_branch_theta_snapshot
                    ),
'''

NEW_FINGERPRINT = '''                    "theta_fingerprint": _candidate_record_array_fingerprint(
                        keep_branch_theta_snapshot
                    ),
'''


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"{label} exact-hunk seam drift")
    return text.replace(old, new, 1)


def _build_source(temp: Path) -> tuple[Path, dict[str, Any]]:
    archive = BASE / "source_locked.tar.gz"
    if core.common.sha256(archive) != BASE_SOURCE_SHA256:
        raise ValueError("Test-2 v13 source archive hash drift")
    source = temp / "source"
    core._extract(archive, source)
    adapt = source / "pipelines/static_adapt/adapt_pipeline.py"
    test = source / "test/test_static_adapt_material_window_prune_source_reuse.py"
    if core.common.sha256(adapt) != BASE_ADAPT_SHA256:
        raise ValueError("Test-2 v13 adapt source hash drift")
    if core.common.sha256(test) != BASE_TEST_SHA256:
        raise ValueError("Test-2 v13 focused-test source hash drift")

    adapt_text = _replace_once(
        adapt.read_text(encoding="utf-8"),
        OLD_FINGERPRINT,
        NEW_FINGERPRINT,
        "immutable keep-branch theta fingerprint helper",
    )
    ast.parse(adapt_text)
    adapt.write_text(adapt_text, encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(source)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "test/test_static_adapt_material_window_prune_source_reuse.py",
            "test/test_static_adapt_material_window_prune_receipt_recovery.py",
            "test/test_static_adapt_prune_source_geometry_threading.py",
            "test/test_static_adapt_material_window_prune_model_index_map.py",
            "test/test_static_adapt_sr_trust_prune.py",
            "test/test_static_adapt_sr_v4_runtime.py",
        ],
        cwd=source,
        env=env,
        check=True,
    )
    core.common.strip_bytecode(source)
    successor = temp / "source_locked.tar.gz"
    core.common.deterministic_archive(source, successor)
    repair = {
        "schema": "paper_i_sr_test2_immutable_keep_receipt_fingerprint_repair_v1",
        "predecessor_bundle": BASE_ID,
        "predecessor_source_archive_sha256": BASE_SOURCE_SHA256,
        "predecessor_cluster": 9309548,
        "detected_by": "three_weak_cutoff_first_accepted_prune_transactions",
        "failure_class": "undefined_receipt_only_array_fingerprint_helper",
        "recovery_authority": "existing_candidate_record_array_fingerprint_v1",
        "changed_paths": ["pipelines/static_adapt/adapt_pipeline.py"],
        "route_contract_sha256_unchanged": core.ROUTE_DIGEST,
        "scientific_settings_changed": False,
        "algorithmic_query_delta": 0,
        "numeric_geometry_modified": False,
        "receipt_only_change": True,
    }
    return successor, repair


def _configure() -> None:
    core.BASE_ID = BASE_ID
    core.BASE_BATCH = BASE_BATCH
    core.BASE = BASE
    core.OUTPUT_ID = OUTPUT_ID
    core.OUTPUT_BATCH = OUTPUT_BATCH
    core.OUTPUT = OUTPUT
    core.BASE_SOURCE_SHA256 = BASE_SOURCE_SHA256
    core.BASE_ADAPT_SHA256 = BASE_ADAPT_SHA256
    core._build_source = _build_source


def main(argv: Sequence[str] | None = None) -> int:
    _configure()
    args = core.parse_args(argv)
    receipt = core.build()
    # The oldest successor builder rewrites the lexical v1 prefix in v13 to
    # v23. Bind that generated segment identity to this immutable v14 bundle.
    core._patch_bundle_text(
        {
            "sr-material-window-fsprune-verify-r0-r50-20260722-v23": (
                "sr-material-window-fsprune-verify-r0-r50-20260722-v14"
            ),
            "sr-material-window-fsprune-verify-r0-r{target}-20260722-v23": (
                "sr-material-window-fsprune-verify-r0-r{target}-20260722-v14"
            ),
        }
    )
    if args.archive_preflight:
        core.archive_preflight()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
