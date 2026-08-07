#!/usr/bin/env python3
"""Build Test-2 v10 with authoritative blank-label receipt binding."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import build_paper_i_hh_sr_material_window_fs_prune_verify_successor_v9_20260722 as previous


core = previous.core
BASE_ID = previous.OUTPUT_ID
BASE_BATCH = previous.OUTPUT_BATCH
BASE = core.INPUT / BASE_ID
OUTPUT_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_fs_prune_verify_"
    "all_six_r50_20260722_v10_chtc"
)
OUTPUT_BATCH = "paper-i-hh-sr-material-window-fsprune-verify-six-r50-20260722-v10"
OUTPUT = core.INPUT / OUTPUT_ID
BASE_SOURCE_SHA256 = "3bf2a93ee2b49ca36b6b436c4f185e961d7123d9a47452a77ff07771e4f1088b"
BASE_ADAPT_SHA256 = "73a207c73909bde772da27e5b5de97108d924f778763b9ed457d6b69edc03f5a"
BASE_TEST_SHA256 = "db1054cf50b1a15a5f2dd761a0875b5158b9e614736e92fe4367f429ca2ca9f4"

OLD_IDENTITY = '''    candidate_identity = dict(candidate_identities[0])
    candidate_position = int(summary.get("position_id", -1))
    candidate_label = str(summary.get("candidate_label", ""))
    candidate_pool_index = int(summary.get("candidate_pool_index", -1))
    for field, expected_value in (
        ("candidate_position_id", candidate_position),
        ("candidate_pool_index", candidate_pool_index),
    ):
        if int(plan.get(field, -2)) != int(expected_value) or int(
            accounting.get(field, -3)
        ) != int(expected_value):
            raise RuntimeError(
                f"Prune source reuse candidate {field} identity drifted."
            )
    if (
        str(plan.get("candidate_label", "")) != candidate_label
        or str(accounting.get("candidate_label", "")) != candidate_label
        or str(candidate_identity.get("candidate_label", "")) != candidate_label
        or int(candidate_identity.get("candidate_pool_index", -2))
        != candidate_pool_index
        or int(candidate_identity.get("position_id", -2)) != candidate_position
    ):
        raise RuntimeError("Prune source reuse candidate identity drifted.")
'''

NEW_IDENTITY = '''    candidate_identity = dict(candidate_identities[0])
    candidate_position = int(summary.get("position_id", -1))
    candidate_label = str(summary.get("candidate_label", ""))
    candidate_pool_index = int(summary.get("candidate_pool_index", -1))
    candidate_identity_label_before_binding = str(
        candidate_identity.get("candidate_label", "")
    )
    for field, expected_value in (
        ("candidate_position_id", candidate_position),
        ("candidate_pool_index", candidate_pool_index),
    ):
        if int(plan.get(field, -2)) != int(expected_value) or int(
            accounting.get(field, -3)
        ) != int(expected_value):
            raise RuntimeError(
                f"Prune source reuse candidate {field} identity drifted."
            )
    candidate_identity_label_placeholder_bound = bool(
        not candidate_identity_label_before_binding
    )
    if candidate_identity_label_placeholder_bound:
        candidate_identity["candidate_label"] = candidate_label
    if (
        str(plan.get("candidate_label", "")) != candidate_label
        or str(accounting.get("candidate_label", "")) != candidate_label
        or str(candidate_identity.get("candidate_label", "")) != candidate_label
        or int(candidate_identity.get("candidate_pool_index", -2))
        != candidate_pool_index
        or int(candidate_identity.get("position_id", -2)) != candidate_position
    ):
        raise RuntimeError("Prune source reuse candidate identity drifted.")
'''

OLD_RECEIPT = '''            "candidate_identity": dict(candidate_identity),
            "state_fingerprint": str(plan.get("state_fingerprint", "")),
'''

NEW_RECEIPT = '''            "candidate_identity": dict(candidate_identity),
            "candidate_identity_binding": {
                "schema": "sr_material_window_prune_candidate_identity_binding_v1",
                "source": (
                    "authoritative_plan_and_estimator_accounting_v1"
                    if candidate_identity_label_placeholder_bound
                    else "coordinate_identity_already_complete_v1"
                ),
                "candidate_label_before_binding": str(
                    candidate_identity_label_before_binding
                ),
                "candidate_label_after_binding": str(
                    candidate_identity["candidate_label"]
                ),
                "placeholder_filled": bool(
                    candidate_identity_label_placeholder_bound
                ),
                "pool_index_and_position_crosschecked": True,
                "numeric_geometry_modified": False,
                "incremental_quantum_query_charge": 0,
            },
            "state_fingerprint": str(plan.get("state_fingerprint", "")),
'''

TEST_SEAM = '''def test_material_window_prune_source_reuse_accepts_candidate_only_window():
'''

TEST_INSERT = '''def test_material_window_prune_source_reuse_binds_only_blank_candidate_label():
    payload = _source_workspace()
    payload["batch_coordinate_identities"][0]["candidate_label"] = ""

    model = _normalize_sr_material_window_prune_source_geometry(
        selector_summary=payload,
        post_admission_labels=["old-0", "candidate", "old-1", "old-2"],
        post_admission_theta=np.asarray([0.1, 0.2, 0.3, 0.4]),
    )

    receipt = model["receipt"]
    assert receipt["candidate_identity"]["candidate_label"] == "candidate"
    binding = receipt["candidate_identity_binding"]
    assert binding["placeholder_filled"] is True
    assert binding["candidate_label_before_binding"] == ""
    assert binding["candidate_label_after_binding"] == "candidate"
    assert binding["pool_index_and_position_crosschecked"] is True
    assert binding["numeric_geometry_modified"] is False
    assert binding["incremental_quantum_query_charge"] == 0


def test_material_window_prune_source_reuse_rejects_conflicting_candidate_label():
    payload = _source_workspace()
    payload["batch_coordinate_identities"][0]["candidate_label"] = "other"

    with pytest.raises(RuntimeError, match="candidate identity drifted"):
        _normalize_sr_material_window_prune_source_geometry(
            selector_summary=payload,
            post_admission_labels=["old-0", "candidate", "old-1", "old-2"],
            post_admission_theta=np.asarray([0.1, 0.2, 0.3, 0.4]),
        )


def test_material_window_prune_source_reuse_accepts_candidate_only_window():
'''


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"{label} exact-hunk seam drift")
    return text.replace(old, new, 1)


def _build_source(temp: Path) -> tuple[Path, dict[str, Any]]:
    archive = BASE / "source_locked.tar.gz"
    if core.common.sha256(archive) != BASE_SOURCE_SHA256:
        raise ValueError("Test-2 v9 source archive hash drift")
    source = temp / "source"
    core._extract(archive, source)
    adapt = source / "pipelines/static_adapt/adapt_pipeline.py"
    test = source / "test/test_static_adapt_material_window_prune_source_reuse.py"
    if core.common.sha256(adapt) != BASE_ADAPT_SHA256:
        raise ValueError("Test-2 v9 adapt source hash drift")
    if core.common.sha256(test) != BASE_TEST_SHA256:
        raise ValueError("Test-2 v9 focused-test source hash drift")

    adapt_text = adapt.read_text(encoding="utf-8")
    adapt_text = _replace_once(
        adapt_text, OLD_IDENTITY, NEW_IDENTITY, "candidate identity binding"
    )
    adapt_text = _replace_once(
        adapt_text, OLD_RECEIPT, NEW_RECEIPT, "candidate identity receipt"
    )
    ast.parse(adapt_text)
    adapt.write_text(adapt_text, encoding="utf-8")

    test_text = test.read_text(encoding="utf-8")
    test_text = _replace_once(
        test_text, TEST_SEAM, TEST_INSERT, "candidate identity focused tests"
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
        "schema": "paper_i_sr_test2_prune_candidate_identity_binding_repair_v1",
        "predecessor_bundle": BASE_ID,
        "predecessor_source_archive_sha256": BASE_SOURCE_SHA256,
        "predecessor_cluster": None,
        "detected_by": "exact_image_full_contract_weak_weak_smoke",
        "failure_class": "blank_coordinate_identity_label_placeholder_not_bound",
        "recovery_authority": (
            "authoritative_plan_accounting_pool_position_exact_binding_v1"
        ),
        "changed_paths": [
            "pipelines/static_adapt/adapt_pipeline.py",
            "test/test_static_adapt_material_window_prune_source_reuse.py",
        ],
        "route_contract_sha256_unchanged": core.ROUTE_DIGEST,
        "scientific_settings_changed": False,
        "algorithmic_query_delta": 0,
        "numeric_geometry_modified": False,
        "nonblank_identity_conflicts_remain_fail_closed": True,
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
            "sr-material-window-fsprune-verify-r0-r50-20260722-v9": (
                "sr-material-window-fsprune-verify-r0-r50-20260722-v10"
            ),
            "sr-material-window-fsprune-verify-r0-r{target}-20260722-v9": (
                "sr-material-window-fsprune-verify-r0-r{target}-20260722-v10"
            ),
        }
    )
    if args.archive_preflight:
        core.archive_preflight()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
