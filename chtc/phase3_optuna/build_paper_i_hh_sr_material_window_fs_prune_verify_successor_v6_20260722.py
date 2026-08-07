#!/usr/bin/env python3
"""Build Test-2 v6 with material-window prune indices mapped to the full ansatz."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import build_paper_i_hh_sr_material_window_fs_prune_verify_successor_v5_20260722 as previous


core = previous.core
BASE_ID = previous.OUTPUT_ID
BASE_BATCH = previous.OUTPUT_BATCH
BASE = core.INPUT / BASE_ID
OUTPUT_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_fs_prune_verify_"
    "all_six_r50_20260722_v6_chtc"
)
OUTPUT_BATCH = "paper-i-hh-sr-material-window-fsprune-verify-six-r50-20260722-v6"
OUTPUT = core.INPUT / OUTPUT_ID
BASE_SOURCE_SHA256 = "623e5135e411ea13119e3409f90f7026209e0d6b9174910f124198bf79cc3d15"
BASE_ADAPT_SHA256 = "bd9b504f0391677db66737ec4da40579296a9d8b6b52f15d357277c489652c62"

OLD_AFFINE_LOOP = '''                    surrogate_scores = {}
                    for idx in range(int(logical_count)):
                        solve_result = solve_full_logical_affine_deletion_fs_trust(
                            theta=np.asarray(theta_logical, dtype=float),
                            gradient=np.asarray(gradient_logical, dtype=float),
                            hessian=np.asarray(hessian, dtype=float),
                            metric=np.asarray(metric_matrix, dtype=float),
                            deletion_index=int(idx),
                            trust_radius=float(affine_trust_state.radius),
                            metric_damping=float(affine_trust_state.metric_damping),
                        )
                        solve_payload = solve_result.as_dict()
                        affine_solver_rows.append(dict(solve_payload))
                        if not bool(solve_result.feasible):
                            continue
                        predicted_change = float(
                            solve_result.predicted_energy_change
                        )
                        surrogate_scores[int(idx)] = {
                            "index": int(idx),
                            "label": str(labels_list[int(idx)]),
'''

NEW_AFFINE_LOOP = '''                    surrogate_scores = {}
                    for model_index, post_index in enumerate(
                        model_post_indices
                    ):
                        solve_result = solve_full_logical_affine_deletion_fs_trust(
                            theta=np.asarray(theta_model, dtype=float),
                            gradient=np.asarray(gradient_logical, dtype=float),
                            hessian=np.asarray(hessian, dtype=float),
                            metric=np.asarray(metric_matrix, dtype=float),
                            deletion_index=int(model_index),
                            trust_radius=float(affine_trust_state.radius),
                            metric_damping=float(affine_trust_state.metric_damping),
                        )
                        solve_payload = solve_result.as_dict()
                        solve_payload["model_coordinate_index"] = int(
                            model_index
                        )
                        solve_payload["post_admission_logical_index"] = int(
                            post_index
                        )
                        affine_solver_rows.append(dict(solve_payload))
                        if not bool(solve_result.feasible):
                            continue
                        predicted_change = float(
                            solve_result.predicted_energy_change
                        )
                        surrogate_scores[int(post_index)] = {
                            "index": int(post_index),
                            "model_coordinate_index": int(model_index),
                            "label": str(labels_list[int(post_index)]),
'''

OLD_BLOCK_RUNTIME_INDEX = '''                            "block_runtime_indices": [int(idx)],
'''
NEW_BLOCK_RUNTIME_INDEX = '''                            "block_runtime_indices": [int(post_index)],
'''


def _build_source(temp: Path) -> tuple[Path, dict[str, Any]]:
    archive = BASE / "source_locked.tar.gz"
    if core.common.sha256(archive) != BASE_SOURCE_SHA256:
        raise ValueError("Test-2 v5 source archive hash drift")
    source = temp / "source"
    core._extract(archive, source)
    adapt = source / "pipelines/static_adapt/adapt_pipeline.py"
    if core.common.sha256(adapt) != BASE_ADAPT_SHA256:
        raise ValueError("Test-2 v5 adapt source hash drift")
    text = adapt.read_text(encoding="utf-8")
    replacements = (
        (OLD_AFFINE_LOOP, NEW_AFFINE_LOOP, "material-window affine solver index map"),
        (OLD_BLOCK_RUNTIME_INDEX, NEW_BLOCK_RUNTIME_INDEX, "full post-admission index receipt"),
    )
    for old, new, label in replacements:
        if text.count(old) != 1:
            raise ValueError(f"{label} seam drift")
        text = text.replace(old, new, 1)
    ast.parse(text)
    adapt.write_text(text, encoding="utf-8")

    regression = source / "test/test_static_adapt_material_window_prune_model_index_map.py"
    regression.write_text(
        '''from pathlib import Path


def test_material_window_affine_model_maps_back_to_full_ansatz_indices() -> None:
    source = Path("pipelines/static_adapt/adapt_pipeline.py").read_text(
        encoding="utf-8"
    )
    nomination = source[source.index("def _build_prune_schur_nomination_scores("):]
    nomination = nomination[: nomination.index("def _default_prune_summary(")]
    assert "for model_index, post_index in enumerate(" in nomination
    assert "model_post_indices" in nomination
    assert "theta=np.asarray(theta_model, dtype=float)" in nomination
    assert "deletion_index=int(model_index)" in nomination
    assert "surrogate_scores[int(post_index)]" in nomination
    assert '"model_coordinate_index": int(model_index)' in nomination
    assert '"block_runtime_indices": [int(post_index)]' in nomination
    assert "for idx in range(int(logical_count))" not in nomination
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
            "test/test_static_adapt_material_window_prune_receipt_recovery.py",
            "test/test_static_adapt_prune_source_geometry_threading.py",
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
        "schema": "paper_i_sr_test2_material_window_prune_index_map_repair_v1",
        "predecessor_bundle": BASE_ID,
        "predecessor_source_archive_sha256": BASE_SOURCE_SHA256,
        "predecessor_cluster": 9308679,
        "detected_by": "six_regime_first_prune_eligible_round_execution",
        "failure_class": "material_window_model_indices_applied_as_full_ansatz_indices",
        "recovery_authority": "authoritative_model_post_indices_v1",
        "changed_paths": [
            "pipelines/static_adapt/adapt_pipeline.py",
            "test/test_static_adapt_material_window_prune_model_index_map.py",
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
            "sr-material-window-fsprune-verify-r0-r50-20260722-v5": (
                "sr-material-window-fsprune-verify-r0-r50-20260722-v6"
            ),
            "sr-material-window-fsprune-verify-r0-r{target}-20260722-v5": (
                "sr-material-window-fsprune-verify-r0-r{target}-20260722-v6"
            ),
        }
    )
    if args.archive_preflight:
        core.archive_preflight()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
