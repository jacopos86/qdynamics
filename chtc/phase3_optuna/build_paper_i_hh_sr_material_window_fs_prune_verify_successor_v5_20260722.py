#!/usr/bin/env python3
"""Build Test-2 v5 with the source-geometry receipt threaded into pruning."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import build_paper_i_hh_sr_material_window_fs_prune_verify_successor_v4_20260722 as previous


core = previous.core
BASE_ID = previous.OUTPUT_ID
BASE_BATCH = previous.OUTPUT_BATCH
BASE = core.INPUT / BASE_ID
OUTPUT_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_fs_prune_verify_"
    "all_six_r50_20260722_v5_chtc"
)
OUTPUT_BATCH = "paper-i-hh-sr-material-window-fsprune-verify-six-r50-20260722-v5"
OUTPUT = core.INPUT / OUTPUT_ID
BASE_SOURCE_SHA256 = "3c510ac495c4055184ff465f07aa66e125625949301f511fd37485b2d968daa3"
BASE_ADAPT_SHA256 = "5bde7028eb35e475881e0da67ee272101201423684a4b13bbfe51baa57ecb439"

OLD_BUILD_SIGNATURE = '''        stage: str,
        affine_trust_state: AffineDeletionFSTrustState | None = None,
    ) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
'''
NEW_BUILD_SIGNATURE = '''        stage: str,
        affine_trust_state: AffineDeletionFSTrustState | None = None,
        source_geometry_workspace: Mapping[str, Any] | None = None,
    ) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
'''

OLD_EXECUTE_SIGNATURE = '''        formal_winning_energy_occurrence_ids_override: list[str] | None = None,
        formal_all_energy_occurrence_ids_override: list[str] | None = None,
    ) -> tuple[list[AnsatzTerm], np.ndarray, float, dict[str, Any], list[ScaffoldCoordinateMetadata], dict[str, int], dict[str, Any]]:
'''
NEW_EXECUTE_SIGNATURE = '''        formal_winning_energy_occurrence_ids_override: list[str] | None = None,
        formal_all_energy_occurrence_ids_override: list[str] | None = None,
        source_geometry_workspace: Mapping[str, Any] | None = None,
    ) -> tuple[list[AnsatzTerm], np.ndarray, float, dict[str, Any], list[ScaffoldCoordinateMetadata], dict[str, int], dict[str, Any]]:
'''

OLD_LIVE_CALL = '''            stage="live_after_admission",
            affine_trust_state=affine_trust_state,
        )
'''
NEW_LIVE_CALL = '''            stage="live_after_admission",
            affine_trust_state=affine_trust_state,
            source_geometry_workspace=source_geometry_workspace,
        )
'''


def _build_source(temp: Path) -> tuple[Path, dict[str, Any]]:
    archive = BASE / "source_locked.tar.gz"
    if core.common.sha256(archive) != BASE_SOURCE_SHA256:
        raise ValueError("Test-2 v4 source archive hash drift")
    source = temp / "source"
    core._extract(archive, source)
    adapt = source / "pipelines/static_adapt/adapt_pipeline.py"
    if core.common.sha256(adapt) != BASE_ADAPT_SHA256:
        raise ValueError("Test-2 v4 adapt source hash drift")
    text = adapt.read_text(encoding="utf-8")
    replacements = (
        (OLD_BUILD_SIGNATURE, NEW_BUILD_SIGNATURE, "nomination signature"),
        (OLD_EXECUTE_SIGNATURE, NEW_EXECUTE_SIGNATURE, "live-prune signature"),
        (OLD_LIVE_CALL, NEW_LIVE_CALL, "live nomination call"),
    )
    for old, new, label in replacements:
        if text.count(old) != 1:
            raise ValueError(f"{label} seam drift")
        text = text.replace(old, new, 1)
    ast.parse(text)
    adapt.write_text(text, encoding="utf-8")

    regression = source / "test/test_static_adapt_prune_source_geometry_threading.py"
    regression.write_text(
        '''from pathlib import Path


def test_source_geometry_receipt_is_threaded_without_measurement_calls() -> None:
    source = Path("pipelines/static_adapt/adapt_pipeline.py").read_text(
        encoding="utf-8"
    )
    execute = source[source.index("def _execute_live_mature_prune_pass("):]
    execute = execute[: execute.index("phase1_prune_metadata_state =")]
    execute_header = execute[: execute.index("summary =")]
    nomination_call = execute[execute.index("prune_schur_scores"):]
    nomination_call = nomination_call[
        : nomination_call.index('summary["schur_surrogate_nomination"]')
    ]
    assert "source_geometry_workspace: Mapping[str, Any] | None = None" in execute_header
    assert "source_geometry_workspace=source_geometry_workspace" in nomination_call
    assert "_record_" not in execute_header + nomination_call

    nomination = source[source.index("def _build_prune_schur_nomination_scores("):]
    nomination = nomination[: nomination.index("def _execute_live_mature_prune_pass(")]
    assert "source_geometry_workspace: Mapping[str, Any] | None = None" in nomination
    assert "selector_summary=source_geometry_workspace" in nomination
''',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "test/test_static_adapt_prune_prefilter_scope_order.py",
            "test/test_static_adapt_prune_tolerance_scope_order.py",
            "test/test_static_adapt_material_window_prune_receipt_recovery.py",
            regression.relative_to(source).as_posix(),
        ],
        cwd=source,
        env=env,
        check=True,
    )
    core.common.strip_bytecode(source)
    successor = temp / "source_locked.tar.gz"
    core.common.deterministic_archive(source, successor)
    repair = {
        "schema": "paper_i_sr_test2_prune_source_geometry_threading_v1",
        "predecessor_bundle": BASE_ID,
        "predecessor_source_archive_sha256": BASE_SOURCE_SHA256,
        "predecessor_cluster": None,
        "detected_by": "exact_uploaded_archive_image_round1_smoke",
        "failure_class": "pre_science_source_geometry_receipt_interface_mismatch",
        "recovery_authority": "already_measured_phase3_source_geometry_workspace_v1",
        "changed_paths": [
            "pipelines/static_adapt/adapt_pipeline.py",
            "test/test_static_adapt_prune_source_geometry_threading.py",
        ],
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
            "sr-material-window-fsprune-verify-r0-r50-20260722-v4": (
                "sr-material-window-fsprune-verify-r0-r50-20260722-v5"
            ),
            "sr-material-window-fsprune-verify-r0-r{target}-20260722-v4": (
                "sr-material-window-fsprune-verify-r0-r{target}-20260722-v5"
            ),
        }
    )
    if args.archive_preflight:
        core.archive_preflight()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
