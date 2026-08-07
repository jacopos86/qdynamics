#!/usr/bin/env python3
"""Build Test-2 v11 with JSON-empty material block shape recovery."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import build_paper_i_hh_sr_material_window_fs_prune_verify_successor_v10_20260722 as previous


core = previous.core
BASE_ID = previous.OUTPUT_ID
BASE_BATCH = previous.OUTPUT_BATCH
BASE = core.INPUT / BASE_ID
OUTPUT_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_fs_prune_verify_"
    "all_six_r50_20260722_v11_chtc"
)
OUTPUT_BATCH = "paper-i-hh-sr-material-window-fsprune-verify-six-r50-20260722-v11"
OUTPUT = core.INPUT / OUTPUT_ID
BASE_SOURCE_SHA256 = "4af4f220ef0552e0b6499115b14a7077d4d5013f7144972a167548964cca1504"
BASE_ADAPT_SHA256 = "799a8e76f73949b747f6421b738fbb42444cdaa235c3612b51e7718959f742e2"
BASE_TEST_SHA256 = "6b8a48bba61279619c9a1c8664d202f01496f7409a87a0b1374240f01c12046a"

OLD_MATRIX = '''    def _matrix(name: str, shape: tuple[int, ...]) -> np.ndarray:
        value = np.asarray(summary.get(name), dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise RuntimeError(
                f"Prune source reuse {name} is missing, nonfinite, or has "
                f"shape {value.shape!r}; expected {shape!r}."
            )
        return value
'''

NEW_MATRIX = '''    def _matrix(name: str, shape: tuple[int, ...]) -> np.ndarray:
        value = np.asarray(summary.get(name), dtype=float)
        if value.size == 0 and any(dimension == 0 for dimension in shape):
            # JSON preserves that the block is empty but not whether its
            # NumPy shape was (0,), (0, 0), or (0, 1).  The authoritative
            # retained-window and singleton dimensions determine that shape.
            value = np.empty(shape, dtype=float)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise RuntimeError(
                f"Prune source reuse {name} is missing, nonfinite, or has "
                f"shape {value.shape!r}; expected {shape!r}."
            )
        return value
'''

TEST_SEAM = '''

@pytest.mark.parametrize(
'''

TEST_INSERT = '''

def test_material_window_prune_source_reuse_recovers_json_empty_block_shapes():
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
            "G_AA_raw": [],
            "G_AB_raw": [],
            "H_AA_raw": [],
            "H_AB_raw": [],
            "g_A": [],
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
    assert np.asarray(model["metric"]).shape == (1, 1)
    assert np.asarray(model["hessian"]).shape == (1, 1)
    receipt = model["receipt"]
    assert receipt["active_pre_indices"] == []
    assert receipt["incremental_quantum_query_charge"] == 0
    assert receipt["duplicate_measurement_performed"] is False


@pytest.mark.parametrize(
'''


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"{label} exact-hunk seam drift")
    return text.replace(old, new, 1)


def _build_source(temp: Path) -> tuple[Path, dict[str, Any]]:
    archive = BASE / "source_locked.tar.gz"
    if core.common.sha256(archive) != BASE_SOURCE_SHA256:
        raise ValueError("Test-2 v10 source archive hash drift")
    source = temp / "source"
    core._extract(archive, source)
    adapt = source / "pipelines/static_adapt/adapt_pipeline.py"
    test = source / "test/test_static_adapt_material_window_prune_source_reuse.py"
    if core.common.sha256(adapt) != BASE_ADAPT_SHA256:
        raise ValueError("Test-2 v10 adapt source hash drift")
    if core.common.sha256(test) != BASE_TEST_SHA256:
        raise ValueError("Test-2 v10 focused-test source hash drift")

    adapt_text = _replace_once(
        adapt.read_text(encoding="utf-8"),
        OLD_MATRIX,
        NEW_MATRIX,
        "JSON-empty matrix shape recovery",
    )
    ast.parse(adapt_text)
    adapt.write_text(adapt_text, encoding="utf-8")

    test_text = _replace_once(
        test.read_text(encoding="utf-8"),
        TEST_SEAM,
        TEST_INSERT,
        "JSON-empty matrix regression",
    )
    ast.parse(test_text)
    test.write_text(test_text, encoding="utf-8")

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
        "schema": "paper_i_sr_test2_json_empty_matrix_shape_repair_v1",
        "predecessor_bundle": BASE_ID,
        "predecessor_source_archive_sha256": BASE_SOURCE_SHA256,
        "predecessor_cluster": 9309543,
        "detected_by": "six_regime_first_eligible_prune_transaction",
        "failure_class": "json_empty_matrix_shape_erasure",
        "recovery_authority": "authoritative_window_and_singleton_dimensions_v1",
        "changed_paths": [
            "pipelines/static_adapt/adapt_pipeline.py",
            "test/test_static_adapt_material_window_prune_source_reuse.py",
        ],
        "route_contract_sha256_unchanged": core.ROUTE_DIGEST,
        "scientific_settings_changed": False,
        "algorithmic_query_delta": 0,
        "numeric_geometry_modified": False,
        "nonempty_shape_conflicts_remain_fail_closed": True,
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
            "sr-material-window-fsprune-verify-r0-r50-20260722-v10": (
                "sr-material-window-fsprune-verify-r0-r50-20260722-v11"
            ),
            "sr-material-window-fsprune-verify-r0-r{target}-20260722-v10": (
                "sr-material-window-fsprune-verify-r0-r{target}-20260722-v11"
            ),
        }
    )
    if args.archive_preflight:
        core.archive_preflight()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
