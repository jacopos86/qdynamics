#!/usr/bin/env python3
"""Build Test-2 v4 with identity-locked selected accounting recovery."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import build_paper_i_hh_sr_material_window_fs_prune_verify_successor_v3_20260722 as previous


core = previous.prior
BASE_ID = previous.OUTPUT_ID
BASE_BATCH = previous.OUTPUT_BATCH
BASE = core.INPUT / BASE_ID
OUTPUT_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_fs_prune_verify_"
    "all_six_r50_20260722_v4_chtc"
)
OUTPUT_BATCH = "paper-i-hh-sr-material-window-fsprune-verify-six-r50-20260722-v4"
OUTPUT = core.INPUT / OUTPUT_ID
BASE_SOURCE_SHA256 = "4950f1d930c259bd4d786a657809a06aab918960a354fe25d59cdfcbb66b9252"
BASE_ADAPT_SHA256 = "ae814207b08fb85fce54420b4172539899c29692ebddf364890c05bb6485668f"

OLD_SEAM = '''                            source_accounting = selected_trust_record.get(
                                "material_window_estimator_accounting"
                            )
                            if not isinstance(source_accounting, Mapping):
                                raise RuntimeError(
                                    "Material-window prune verification lost "
                                    "the selected estimator accounting receipt."
                                )
'''
NEW_SEAM = '''                            source_accounting = selected_trust_record.get(
                                "material_window_estimator_accounting"
                            )
                            if not isinstance(source_accounting, Mapping):
                                selected_feature = selected_trust_record.get(
                                    "feature"
                                )
                                if not isinstance(
                                    selected_feature, CandidateFeatures
                                ):
                                    raise RuntimeError(
                                        "Material-window prune verification "
                                        "cannot recover estimator accounting "
                                        "without the selected candidate identity."
                                    )
                                selected_identity = (
                                    int(selected_feature.candidate_pool_index),
                                    str(selected_feature.candidate_label),
                                    int(selected_feature.position_id),
                                )
                                accounting_matches = [
                                    dict(accounting_row)
                                    for accounting_row in (
                                        material_window_estimator_accounting
                                    )
                                    if (
                                        int(
                                            accounting_row.get(
                                                "candidate_pool_index", -1
                                            )
                                        ),
                                        str(
                                            accounting_row.get(
                                                "candidate_label", ""
                                            )
                                        ),
                                        int(
                                            accounting_row.get(
                                                "candidate_position_id", -1
                                            )
                                        ),
                                    )
                                    == selected_identity
                                ]
                                if len(accounting_matches) != 1:
                                    raise RuntimeError(
                                        "Material-window prune verification "
                                        "could not uniquely recover the "
                                        "selected estimator accounting receipt."
                                    )
                                source_accounting = accounting_matches[0]
                                selected_trust_record[
                                    "material_window_estimator_accounting"
                                ] = copy.deepcopy(source_accounting)
                            if not isinstance(source_accounting, Mapping):
                                raise RuntimeError(
                                    "Material-window prune verification lost "
                                    "the selected estimator accounting receipt."
                                )
'''


def _build_source(temp: Path) -> tuple[Path, dict[str, Any]]:
    archive = BASE / "source_locked.tar.gz"
    if core.common.sha256(archive) != BASE_SOURCE_SHA256:
        raise ValueError("Test-2 v3 source archive hash drift")
    source = temp / "source"
    core._extract(archive, source)
    adapt = source / "pipelines/static_adapt/adapt_pipeline.py"
    if core.common.sha256(adapt) != BASE_ADAPT_SHA256:
        raise ValueError("Test-2 v3 adapt source hash drift")
    text = adapt.read_text(encoding="utf-8")
    if text.count(OLD_SEAM) != 1:
        raise ValueError("selected estimator-accounting recovery seam drift")
    text = text.replace(OLD_SEAM, NEW_SEAM, 1)
    ast.parse(text)
    adapt.write_text(text, encoding="utf-8")
    regression = source / "test/test_static_adapt_material_window_prune_receipt_recovery.py"
    regression.write_text(
        '''from pathlib import Path


def test_selected_prune_receipt_recovery_is_identity_locked_and_query_free() -> None:
    source = Path("pipelines/static_adapt/adapt_pipeline.py").read_text(
        encoding="utf-8"
    )
    block = source[source.index("selected_identity = ("):]
    block = block[: block.index("phase1_prune_source_geometry_workspace")]
    assert "candidate_pool_index" in block
    assert "candidate_label" in block
    assert "candidate_position_id" in block
    assert "len(accounting_matches) != 1" in block
    assert "_record_" not in block
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
        "schema": "paper_i_sr_test2_selected_prune_receipt_recovery_v1",
        "predecessor_bundle": BASE_ID,
        "predecessor_source_archive_sha256": BASE_SOURCE_SHA256,
        "predecessor_cluster": None,
        "detected_by": "exact_uploaded_archive_image_round1_smoke",
        "failure_class": "pre_science_selected_material_window_estimator_receipt_handoff",
        "recovery_authority": "exact_candidate_pool_label_position_identity_v1",
        "changed_paths": [
            "pipelines/static_adapt/adapt_pipeline.py",
            "test/test_static_adapt_material_window_prune_receipt_recovery.py",
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
            "sr-material-window-fsprune-verify-r0-r50-20260722-v3": (
                "sr-material-window-fsprune-verify-r0-r50-20260722-v4"
            ),
            "sr-material-window-fsprune-verify-r0-r{target}-20260722-v3": (
                "sr-material-window-fsprune-verify-r0-r{target}-20260722-v4"
            ),
        }
    )
    if args.archive_preflight:
        core.archive_preflight()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
