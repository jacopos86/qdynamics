#!/usr/bin/env python3
"""Build Test-2 v7 with candidate-only material prune windows supported."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import build_paper_i_hh_sr_material_window_fs_prune_verify_successor_v6_20260722 as previous


core = previous.core
BASE_ID = previous.OUTPUT_ID
BASE_BATCH = previous.OUTPUT_BATCH
BASE = core.INPUT / BASE_ID
OUTPUT_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_fs_prune_verify_"
    "all_six_r50_20260722_v7_chtc"
)
OUTPUT_BATCH = "paper-i-hh-sr-material-window-fsprune-verify-six-r50-20260722-v7"
OUTPUT = core.INPUT / OUTPUT_ID
BASE_SOURCE_SHA256 = "be10a6caff476b8917aa53cbb38fc1b1510073813e96c9dd8476244a350c0306"
BASE_ADAPT_SHA256 = "ef9772e9476ddd6b7f0e494ed5f6143ff1410a163a6167158d99592ed2b8c80b"

OLD_RETAINED_WINDOW_GUARD = '''    if (
        not active_pre_indices
        or len(set(active_pre_indices)) != len(active_pre_indices)
        or any(index < 0 or index >= pre_count for index in active_pre_indices)
    ):
'''

NEW_RETAINED_WINDOW_GUARD = '''    if (
        len(set(active_pre_indices)) != len(active_pre_indices)
        or any(index < 0 or index >= pre_count for index in active_pre_indices)
    ):
'''

CANDIDATE_ONLY_REGRESSION = '''

def test_material_window_prune_source_reuse_accepts_candidate_only_window():
    payload = _source_workspace()
    plan = payload["estimator_acquisition_plan"]
    accounting = payload["material_window_estimator_accounting"]
    plan.update(
        {
            "active_indices": [0],
            "screen_gram_diagonal_indices": [0],
            "candidate_cross_gram_active_indices": [],
            "candidate_cross_hessian_active_indices": [],
            "old_old_metric_pairs_acquired": [],
            "old_old_hessian_pairs_acquired": [],
            "active_gradient_indices_acquired": [],
        }
    )
    accounting["source_plan"] = copy.deepcopy(plan)
    payload.update(
        {
            "active_coordinate_identities": [],
            "G_AA_raw": np.empty((0, 0), dtype=float),
            "G_AB_raw": np.empty((0, 1), dtype=float),
            "H_AA_raw": np.empty((0, 0), dtype=float),
            "H_AB_raw": np.empty((0, 1), dtype=float),
            "g_A": np.empty((0,), dtype=float),
            "material_window_receipt": {
                "retained_indices": [],
                "omitted_indices": [0],
            },
            "material_window_refresh": {
                "performed": False,
                "final_active_indices": [],
            },
        }
    )

    model = _normalize_sr_material_window_prune_source_geometry(
        selector_summary=payload,
        post_admission_labels=["old-0", "candidate"],
        post_admission_theta=np.asarray([0.1, 0.2]),
    )

    assert model["model_post_indices"] == [1]
    assert np.allclose(model["theta"], [0.2])
    assert np.allclose(model["gradient"], [-0.1])
    assert np.asarray(model["metric"]).shape == (1, 1)
    assert np.asarray(model["hessian"]).shape == (1, 1)
    receipt = model["receipt"]
    assert receipt["active_pre_indices"] == []
    assert receipt["active_post_indices"] == []
    assert receipt["candidate_post_index"] == 1
    assert receipt["incremental_quantum_query_charge"] == 0
    assert receipt["duplicate_measurement_performed"] is False
'''


def _build_source(temp: Path) -> tuple[Path, dict[str, Any]]:
    archive = BASE / "source_locked.tar.gz"
    if core.common.sha256(archive) != BASE_SOURCE_SHA256:
        raise ValueError("Test-2 v6 source archive hash drift")
    source = temp / "source"
    core._extract(archive, source)
    adapt = source / "pipelines/static_adapt/adapt_pipeline.py"
    if core.common.sha256(adapt) != BASE_ADAPT_SHA256:
        raise ValueError("Test-2 v6 adapt source hash drift")
    text = adapt.read_text(encoding="utf-8")
    if text.count(OLD_RETAINED_WINDOW_GUARD) != 1:
        raise ValueError("candidate-only retained-window guard seam drift")
    text = text.replace(
        OLD_RETAINED_WINDOW_GUARD,
        NEW_RETAINED_WINDOW_GUARD,
        1,
    )
    ast.parse(text)
    adapt.write_text(text, encoding="utf-8")

    regression = source / "test/test_static_adapt_material_window_prune_source_reuse.py"
    regression.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        core.ROOT / "test/test_static_adapt_material_window_prune_source_reuse.py",
        regression,
    )
    regression_text = regression.read_text(encoding="utf-8")
    if "test_material_window_prune_source_reuse_accepts_candidate_only_window" not in regression_text:
        raise ValueError("candidate-only regression missing from frozen-source test")

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
        ],
        cwd=source,
        env=env,
        check=True,
    )
    core.common.strip_bytecode(source)
    successor = temp / "source_locked.tar.gz"
    core.common.deterministic_archive(source, successor)
    repair = {
        "schema": "paper_i_sr_test2_candidate_only_material_window_repair_v1",
        "predecessor_bundle": BASE_ID,
        "predecessor_source_archive_sha256": BASE_SOURCE_SHA256,
        "predecessor_cluster": 9309288,
        "detected_by": "six_regime_first_prune_eligible_round_execution",
        "failure_class": "candidate_only_material_window_rejected_as_invalid",
        "recovery_authority": "candidate_only_W_empty_plus_singleton_v1",
        "changed_paths": [
            "pipelines/static_adapt/adapt_pipeline.py",
            "test/test_static_adapt_material_window_prune_source_reuse.py",
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
            "sr-material-window-fsprune-verify-r0-r50-20260722-v6": (
                "sr-material-window-fsprune-verify-r0-r50-20260722-v7"
            ),
            "sr-material-window-fsprune-verify-r0-r{target}-20260722-v6": (
                "sr-material-window-fsprune-verify-r0-r{target}-20260722-v7"
            ),
        }
    )
    if args.archive_preflight:
        core.archive_preflight()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
