#!/usr/bin/env python3
"""Build Test-2 v13 from the source-identical built-only v12 bundle."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import build_paper_i_hh_sr_material_window_fs_prune_verify_successor_v12_20260722 as previous


core = previous.core
BASE_ID = previous.OUTPUT_ID
BASE_BATCH = previous.OUTPUT_BATCH
BASE = core.INPUT / BASE_ID
OUTPUT_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_fs_prune_verify_"
    "all_six_r50_20260722_v13_chtc"
)
OUTPUT_BATCH = "paper-i-hh-sr-material-window-fsprune-verify-six-r50-20260722-v13"
OUTPUT = core.INPUT / OUTPUT_ID
SOURCE_SHA256 = "242c26fe2d92ba433f68963bf38593ebb681f7ab4dc5ba8e8d484a5a0fbc1eca"


def _build_source(temp: Path) -> tuple[Path, dict[str, Any]]:
    archive = BASE / "source_locked.tar.gz"
    if core.common.sha256(archive) != SOURCE_SHA256:
        raise ValueError("Test-2 v12 source archive hash drift")
    source = temp / "source"
    core._extract(archive, source)
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
    if core.common.sha256(successor) != SOURCE_SHA256:
        raise ValueError("provenance-only v13 changed the source archive")
    repair = {
        "schema": "paper_i_sr_test2_json_empty_shape_and_segment_provenance_repair_v2",
        "predecessor_bundle": BASE_ID,
        "predecessor_source_archive_sha256": SOURCE_SHA256,
        "predecessor_submission_performed": False,
        "failure_class": "built_only_successor_segment_label_not_rebound",
        "recovery_authority": "exact_segment_identity_v13",
        "predecessor_operational_repair": json.loads(
            (BASE / "operational_repair.json").read_text(encoding="utf-8")
        ),
        "changed_paths": [],
        "route_contract_sha256_unchanged": core.ROUTE_DIGEST,
        "source_archive_sha256_unchanged": SOURCE_SHA256,
        "scientific_settings_changed": False,
        "algorithmic_query_delta": 0,
        "numeric_geometry_modified": False,
    }
    return successor, repair


def _configure() -> None:
    core.BASE_ID = BASE_ID
    core.BASE_BATCH = BASE_BATCH
    core.BASE = BASE
    core.OUTPUT_ID = OUTPUT_ID
    core.OUTPUT_BATCH = OUTPUT_BATCH
    core.OUTPUT = OUTPUT
    core.BASE_SOURCE_SHA256 = SOURCE_SHA256
    core._build_source = _build_source


def main(argv: Sequence[str] | None = None) -> int:
    _configure()
    args = core.parse_args(argv)
    receipt = core.build()
    core._patch_bundle_text(
        {
            "sr-material-window-fsprune-verify-r0-r50-20260722-v20": (
                "sr-material-window-fsprune-verify-r0-r50-20260722-v13"
            ),
            "sr-material-window-fsprune-verify-r0-r{target}-20260722-v20": (
                "sr-material-window-fsprune-verify-r0-r{target}-20260722-v13"
            ),
        }
    )
    if args.archive_preflight:
        core.archive_preflight()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
