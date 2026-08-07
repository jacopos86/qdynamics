#!/usr/bin/env python3
"""Build Test-2 v9 with fail-closed inner-cause telemetry."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import build_paper_i_hh_sr_material_window_fs_prune_verify_successor_v8_20260722 as previous


core = previous.core
BASE_ID = previous.OUTPUT_ID
BASE_BATCH = previous.OUTPUT_BATCH
BASE = core.INPUT / BASE_ID
OUTPUT_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_fs_prune_verify_"
    "all_six_r50_20260722_v9_chtc"
)
OUTPUT_BATCH = "paper-i-hh-sr-material-window-fsprune-verify-six-r50-20260722-v9"
OUTPUT = core.INPUT / OUTPUT_ID
BASE_SOURCE_SHA256 = "c1b25dbefd04f979b07bf6ab61f36cf40e54870e3413c7bfecd33437d6539dcc"
BASE_ADAPT_SHA256 = "54cf6e1aca13a5643ba99bbfa351c8f5ce3fb9e40f1d2cbbd4d35f8ca7da14dc"

OLD_ERROR = '''                raise RuntimeError(
                    "SR-SNAKE v4 full-logical FS trust prune model failed; "
                    "refusing to fall back to a historical nomination route."
                ) from exc
'''
NEW_ERROR = '''                raise RuntimeError(
                    "SR-SNAKE v4 full-logical FS trust prune model failed; "
                    "refusing to fall back to a historical nomination route. "
                    f"Cause: {type(exc).__name__}: {exc}"
                ) from exc
'''


def _build_source(temp: Path) -> tuple[Path, dict[str, Any]]:
    archive = BASE / "source_locked.tar.gz"
    if core.common.sha256(archive) != BASE_SOURCE_SHA256:
        raise ValueError("Test-2 v8 source archive hash drift")
    source = temp / "source"
    core._extract(archive, source)
    adapt = source / "pipelines/static_adapt/adapt_pipeline.py"
    if core.common.sha256(adapt) != BASE_ADAPT_SHA256:
        raise ValueError("Test-2 v8 adapt source hash drift")
    text = adapt.read_text(encoding="utf-8")
    if text.count(OLD_ERROR) != 1:
        raise ValueError("affine-prune cause telemetry seam drift")
    text = text.replace(OLD_ERROR, NEW_ERROR, 1)
    ast.parse(text)
    adapt.write_text(text, encoding="utf-8")

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
        "schema": "paper_i_sr_test2_affine_prune_cause_telemetry_repair_v1",
        "predecessor_bundle": BASE_ID,
        "predecessor_source_archive_sha256": BASE_SOURCE_SHA256,
        "predecessor_cluster": None,
        "detected_by": "exact_image_full_horizon_weak_weak_smoke",
        "failure_class": "affine_prune_inner_exception_hidden_by_fail_closed_wrapper",
        "recovery_authority": "exception_type_and_message_telemetry_only_v1",
        "changed_paths": ["pipelines/static_adapt/adapt_pipeline.py"],
        "route_contract_sha256_unchanged": core.ROUTE_DIGEST,
        "scientific_settings_changed": False,
        "algorithmic_query_delta": 0,
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
    core._patch_bundle_text(
        {
            "sr-material-window-fsprune-verify-r0-r50-20260722-v8": (
                "sr-material-window-fsprune-verify-r0-r50-20260722-v9"
            ),
            "sr-material-window-fsprune-verify-r0-r{target}-20260722-v8": (
                "sr-material-window-fsprune-verify-r0-r{target}-20260722-v9"
            ),
        }
    )
    if args.archive_preflight:
        core.archive_preflight()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
