#!/usr/bin/env python3
"""Build Test-2 v15 with the geometry-expansion prune hold repair."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import build_paper_i_hh_sr_material_window_fs_prune_verify_successor_v14_20260722 as previous


core = previous.core
BASE_ID = previous.OUTPUT_ID
BASE_BATCH = previous.OUTPUT_BATCH
BASE = core.INPUT / BASE_ID
OUTPUT_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_fs_prune_verify_"
    "all_six_r50_20260722_v15_chtc"
)
OUTPUT_BATCH = "paper-i-hh-sr-material-window-fsprune-verify-six-r50-20260722-v15"
OUTPUT = core.INPUT / OUTPUT_ID
BASE_SOURCE_SHA256 = "f1a01a6e8b3a11ee8ab861b7a55ef1307f57ff35c12085a4bb919fe80c5038e6"
BASE_ADAPT_SHA256 = "5686cbebd6e1bf907f4f9bb76a373fab5596b22d9ddef6c6410acb1b84923b8c"
BASE_TEST_SHA256 = "607312df851a14a1e91457f2c84fdc26d68f46b476c119a5ec67c1ced376da23"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"{label} exact-hunk seam drift")
    return text.replace(old, new, 1)


def _build_source(temp: Path) -> tuple[Path, dict[str, Any]]:
    archive = BASE / "source_locked.tar.gz"
    if core.common.sha256(archive) != BASE_SOURCE_SHA256:
        raise ValueError("Test-2 v14 source archive hash drift")
    source = temp / "source"
    core._extract(archive, source)
    adapt = source / "pipelines/static_adapt/adapt_pipeline.py"
    test = source / "test/test_static_adapt_material_window_prune_source_reuse.py"
    if core.common.sha256(adapt) != BASE_ADAPT_SHA256:
        raise ValueError("Test-2 v14 adapt source hash drift")
    if core.common.sha256(test) != BASE_TEST_SHA256:
        raise ValueError("Test-2 v14 focused-test source hash drift")

    text = adapt.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        '''        source_geometry_workspace: Mapping[str, Any] | None = None,
    ) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
''',
        '''        source_geometry_workspace: Mapping[str, Any] | None = None,
        source_geometry_unavailable_reason: str | None = None,
    ) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
''',
        "prune nomination source-unavailable signature",
    )
    text = _replace_once(
        text,
        '''                    if not isinstance(source_geometry_workspace, Mapping):
                        raise RuntimeError(
                            "Material-window prune nomination lost the "
                            "authoritative Phase-III source workspace."
                        )
''',
        '''                    unavailable_reason = str(
                        source_geometry_unavailable_reason or ""
                    ).strip()
                    if unavailable_reason:
                        if unavailable_reason != (
                            "geometry_expansion_no_coordinate_prediction"
                        ):
                            raise RuntimeError(
                                "Material-window prune nomination received an "
                                "unknown source-geometry unavailability reason."
                            )
                        if source_geometry_workspace is not None:
                            raise RuntimeError(
                                "Geometry-expansion prune hold cannot carry a "
                                "measured W-plus-candidate source workspace."
                            )
                        if affine_trust_state is None:
                            raise RuntimeError(
                                "Geometry-expansion prune hold requires the "
                                "branch-local affine trust state."
                            )
                        payload = {
                            **_inactive_prune_schur_nomination_payload(
                                prune_cfg=phase1_prune_cfg,
                                selected_parameterization_mode=(
                                    selected_parameterization_mode
                                ),
                                reason=(
                                    "no_eligible_old_coordinates_in_material_window"
                                ),
                                **inactive_base,
                            ),
                            "active": False,
                            "reason": (
                                "no_eligible_old_coordinates_in_material_window"
                            ),
                            "used_for_nomination": False,
                            "score_count": 0,
                            "stage": str(stage),
                            "coordinate_mode": (
                                "logical_material_window_geometry_expansion_hold"
                            ),
                            "source_geometry_unavailable_reason": (
                                unavailable_reason
                            ),
                            "affine_deletion_fs_trust_active": True,
                            "affine_deletion_fs_trust_state": (
                                _affine_prune_state_payload(
                                    affine_trust_state,
                                    source="geometry_expansion_hold",
                                )
                            ),
                            "affine_deletion_models": [],
                            "affine_deletion_model_count": 0,
                            "affine_deletion_feasible_count": 0,
                            "phase1_prune_source_geometry_reuse": None,
                            "cached_post_admission_geometry": {
                                "G": False,
                                "g": False,
                                "H": False,
                                "fixed_branch": (
                                    "geometry_expansion_no_coordinate_prediction"
                                ),
                                "reuse_policy": (
                                    "ineligible_hold_without_remeasurement_v1"
                                ),
                                "duplicate_measurement_performed": False,
                            },
                        }
                        return {}, payload
                    if not isinstance(source_geometry_workspace, Mapping):
                        raise RuntimeError(
                            "Material-window prune nomination lost the "
                            "authoritative Phase-III source workspace."
                        )
''',
        "geometry-expansion prune source hold",
    )
    text = _replace_once(
        text,
        '''        source_geometry_workspace: Mapping[str, Any] | None = None,
    ) -> tuple[list[AnsatzTerm], np.ndarray, float, dict[str, Any], list[ScaffoldCoordinateMetadata], dict[str, int], dict[str, Any]]:
''',
        '''        source_geometry_workspace: Mapping[str, Any] | None = None,
        source_geometry_unavailable_reason: str | None = None,
    ) -> tuple[list[AnsatzTerm], np.ndarray, float, dict[str, Any], list[ScaffoldCoordinateMetadata], dict[str, int], dict[str, Any]]:
''',
        "live prune source-unavailable signature",
    )
    text = _replace_once(
        text,
        '''            affine_trust_state=affine_trust_state,
            source_geometry_workspace=source_geometry_workspace,
        )
''',
        '''            affine_trust_state=affine_trust_state,
            source_geometry_workspace=source_geometry_workspace,
            source_geometry_unavailable_reason=(
                source_geometry_unavailable_reason
            ),
        )
''',
        "live prune source-unavailable threading",
    )
    text = _replace_once(
        text,
        '''                        source_geometry_workspace=(
                            phase1_prune_source_geometry_workspace
                            if phase1_prune_material_window_source_reuse_active
                            else None
                        ),
''',
        '''                        source_geometry_workspace=(
                            phase1_prune_source_geometry_workspace
                            if phase1_prune_material_window_source_reuse_active
                            and not geometry_expansion_trust_update
                            else None
                        ),
                        source_geometry_unavailable_reason=(
                            "geometry_expansion_no_coordinate_prediction"
                            if phase1_prune_material_window_source_reuse_active
                            and geometry_expansion_trust_update
                            else None
                        ),
''',
        "geometry-expansion prune source eligibility",
    )
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
        "schema": "paper_i_sr_test2_geometry_expansion_prune_hold_repair_v1",
        "predecessor_bundle": BASE_ID,
        "predecessor_source_archive_sha256": BASE_SOURCE_SHA256,
        "predecessor_cluster": 9309670,
        "detected_by": "intermediate_weak_round_26_geometry_expansion",
        "failure_class": "geometry_expansion_lacks_reusable_W_plus_candidate_prune_model",
        "recovery_authority": "explicit_ineligible_zero_query_prune_hold_v1",
        "changed_paths": ["pipelines/static_adapt/adapt_pipeline.py"],
        "route_contract_sha256_unchanged": core.ROUTE_DIGEST,
        "scientific_settings_changed": False,
        "algorithmic_query_delta": 0,
        "numeric_geometry_modified": False,
        "prune_acceptance_policy_modified": False,
        "legacy_nomination_fallback_used": False,
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
            "sr-material-window-fsprune-verify-r0-r50-20260722-v23": (
                "sr-material-window-fsprune-verify-r0-r50-20260722-v15"
            ),
            "sr-material-window-fsprune-verify-r0-r{target}-20260722-v23": (
                "sr-material-window-fsprune-verify-r0-r{target}-20260722-v15"
            ),
        }
    )
    if args.archive_preflight:
        core.archive_preflight()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
