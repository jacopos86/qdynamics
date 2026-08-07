#!/usr/bin/env python3
"""Build Test-2 v8 with old-coordinate-only material-window pruning."""

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

import build_paper_i_hh_sr_material_window_fs_prune_verify_successor_v7_20260722 as previous


core = previous.core
BASE_ID = previous.OUTPUT_ID
BASE_BATCH = previous.OUTPUT_BATCH
BASE = core.INPUT / BASE_ID
OUTPUT_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_fs_prune_verify_"
    "all_six_r50_20260722_v8_chtc"
)
OUTPUT_BATCH = "paper-i-hh-sr-material-window-fsprune-verify-six-r50-20260722-v8"
OUTPUT = core.INPUT / OUTPUT_ID
BASE_SOURCE_SHA256 = "912a72857dcd87d64ec7a8670a4d705fd986e0eb18ce94e9fd9eb143e3733171"
BASE_ADAPT_SHA256 = "f575e11ccc7388e65056e3ad696a1fa972f38e4198f0a653733af516c479b01f"
TEST_SOURCE_SHA256 = "db1054cf50b1a15a5f2dd761a0875b5158b9e614736e92fe4367f429ca2ca9f4"
ADDITIONAL_TEST_SHA256 = {
    "test/test_static_adapt_sr_trust_prune.py": (
        "38c5787ac4362d02fff6413e2a1675aabc7206a974aff2906c4eb619b96c0616"
    ),
    "test/test_static_adapt_sr_v4_runtime.py": (
        "fc53c7a01e8a16bb7cccfc3ab57a8d1ddfa13c9a4a4e5c12d91ada42ecef8015"
    ),
}

DELETION_COORDINATE_HELPER = r'''
def _sr_material_window_prune_deletion_coordinates(
    *,
    model_post_indices: Sequence[int],
    source_geometry_reuse_receipt: Mapping[str, Any],
) -> list[tuple[int, int]]:
    """Map only retained pre-admission coordinates into the prune model.

    The material-window source model is ordered as ``W`` followed by the newly
    admitted singleton.  That singleton is required to complete the measured
    response model, but it is not an old coordinate eligible for deletion in
    the same post-admission prune transaction.  An empty ``W`` is therefore a
    valid zero-candidate prune model rather than permission to delete the
    admitted singleton or fall back to historical ranking.
    """

    model_indices = [int(value) for value in model_post_indices]
    active_post_indices = [
        int(value)
        for value in source_geometry_reuse_receipt.get(
            "active_post_indices", ()
        )
    ]
    try:
        candidate_post_index = int(
            source_geometry_reuse_receipt["candidate_post_index"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Material-window prune source reuse lost the admitted singleton "
            "identity."
        ) from exc
    expected_model_indices = [
        *active_post_indices,
        int(candidate_post_index),
    ]
    if model_indices != expected_model_indices:
        raise RuntimeError(
            "Material-window prune source model order drifted from W plus the "
            "admitted singleton."
        )
    if (
        len(set(active_post_indices)) != len(active_post_indices)
        or int(candidate_post_index) in set(active_post_indices)
    ):
        raise RuntimeError(
            "Material-window prune deletion identities are not unique or "
            "include the admitted singleton."
        )
    return [
        (int(model_index), int(post_index))
        for model_index, post_index in enumerate(active_post_indices)
    ]
'''

NO_ELIGIBLE_HOLD_HELPER = r'''
def _sr_v4_no_eligible_material_window_prune_hold_receipt(
    *,
    route_active: bool,
    nomination_payload: Mapping[str, Any],
    trust_state: AffineDeletionFSTrustState | None,
) -> dict[str, Any] | None:
    """Hold v4 when measured ``W`` contains no old deletion coordinate."""

    if not bool(route_active):
        return None
    payload = dict(nomination_payload)
    if str(payload.get("reason", "")) != (
        "no_eligible_old_coordinates_in_material_window"
    ):
        return None
    if trust_state is None:
        raise RuntimeError(
            "SR-SNAKE v4 cannot hold an empty material-window prune round "
            "without its branch-local trust state."
        )
    rows_raw = payload.get("affine_deletion_models")
    if not isinstance(rows_raw, Sequence) or isinstance(
        rows_raw, (str, bytes, bytearray)
    ):
        raise RuntimeError(
            "Empty material-window prune nomination lacks its model rows."
        )
    if (
        len(list(rows_raw)) != 0
        or int(payload.get("affine_deletion_model_count", -1)) != 0
        or int(payload.get("affine_deletion_feasible_count", -1)) != 0
        or int(payload.get("score_count", -1)) != 0
    ):
        raise RuntimeError(
            "Empty material-window prune nomination has inconsistent counts."
        )
    state_payload = {
        "schema": "affine_deletion_fs_trust_state_v1",
        "radius": float(trust_state.radius),
        "metric_damping": float(trust_state.metric_damping),
        "update_count": int(trust_state.update_count),
    }
    return {
        "schema": "sr_v4_no_eligible_material_window_prune_model_v1",
        "status": "skipped_no_eligible_old_coordinates",
        "reason": "no_eligible_old_coordinates_in_material_window",
        "model_count": 0,
        "feasible_model_count": 0,
        "legacy_nomination_fallback_used": False,
        "admitted_singleton_nominated": False,
        "exact_delete_refit_trial_count": 0,
        "trust_state_action": "hold_exactly",
        "trust_state_before": dict(state_payload),
        "trust_state_after": dict(state_payload),
        "trust_update": {
            "schema": "affine_deletion_fs_trust_state_update_v1",
            "status": "held",
            "reason": "no_eligible_old_coordinates_in_material_window",
            "radius_before": float(trust_state.radius),
            "radius_after": float(trust_state.radius),
            "radius_action": "hold_no_eligible_old_coordinate",
            "metric_damping_before": float(trust_state.metric_damping),
            "metric_damping_after": float(trust_state.metric_damping),
            "metric_damping_action": "hold_no_eligible_old_coordinate",
            "update_count_before": int(trust_state.update_count),
            "update_count_after": int(trust_state.update_count),
            "classical_quantum_query_charge": 0,
        },
        "classical_quantum_query_charge": 0,
    }
'''

OLD_AFFINE_LOOP = '''                    surrogate_scores = {}
                    for model_index, post_index in enumerate(
                        model_post_indices
                    ):
'''
NEW_AFFINE_LOOP = '''                    surrogate_scores = {}
                    affine_deletion_coordinates = list(
                        enumerate(model_post_indices)
                    )
                    if source_geometry_reuse_receipt is not None:
                        affine_deletion_coordinates = (
                            _sr_material_window_prune_deletion_coordinates(
                                model_post_indices=model_post_indices,
                                source_geometry_reuse_receipt=(
                                    source_geometry_reuse_receipt
                                ),
                            )
                        )
                    for model_index, post_index in affine_deletion_coordinates:
'''

OLD_REASON = '''                "reason": "ok" if surrogate_scores else "no_scores",
'''
NEW_REASON = '''                "reason": (
                    "ok"
                    if surrogate_scores
                    else (
                        "no_eligible_old_coordinates_in_material_window"
                        if source_geometry_reuse_receipt is not None
                        and not affine_solver_rows
                        else "no_scores"
                    )
                ),
'''

NO_ELIGIBLE_CALL = '''        no_eligible_hold = (
            _sr_v4_no_eligible_material_window_prune_hold_receipt(
                route_active=bool(phase1_prune_affine_trust_route_active),
                nomination_payload=prune_schur_payload,
                trust_state=affine_trust_state,
            )
        )
        if no_eligible_hold is not None:
            summary["phase1_prune_no_feasible_model"] = dict(
                no_eligible_hold
            )
            summary["phase1_prune_trust_update"] = dict(
                no_eligible_hold["trust_update"]
            )
            summary["permission_reason"] = (
                "no_eligible_old_coordinates_in_material_window"
            )
            summary["candidate_count"] = 0
            summary["probe_indices"] = []
            summary["probe_labels"] = []
            summary["frozen_scores"] = []
            summary["frozen_probe_policy"] = (
                "disabled_no_eligible_old_coordinate_v1"
            )
            summary["nfev_formal_frozen_prune_energy_probes"] = 0
            summary["recoverability_ladder_active"] = False
            summary["recoverability_rung_policy"] = (
                "not_entered_no_eligible_old_coordinate_v1"
            )
            summary["recoverability_ladder_rows"] = []
            summary["typed_prune_ladder_status"] = (
                "no_eligible_old_coordinate"
            )
            return (
                list(ops_now),
                np.asarray(theta_runtime_now, dtype=float),
                float(energy_now),
                dict(optimizer_memory_now),
                metadata_live,
                {str(k): int(v) for k, v in first_seen_steps.items()},
                summary,
            )
'''


def _build_source(temp: Path) -> tuple[Path, dict[str, Any]]:
    archive = BASE / "source_locked.tar.gz"
    if core.common.sha256(archive) != BASE_SOURCE_SHA256:
        raise ValueError("Test-2 v7 source archive hash drift")
    source = temp / "source"
    core._extract(archive, source)
    adapt = source / "pipelines/static_adapt/adapt_pipeline.py"
    if core.common.sha256(adapt) != BASE_ADAPT_SHA256:
        raise ValueError("Test-2 v7 adapt source hash drift")
    text = adapt.read_text(encoding="utf-8")

    insertion_markers = (
        (
            "\n\ndef _sr_v4_prune_estimator_accounting_views(",
            f"\n\n{DELETION_COORDINATE_HELPER}\n\ndef _sr_v4_prune_estimator_accounting_views(",
            "old-coordinate deletion helper",
        ),
        (
            "\n\ndef _phase3_shadow_recommended_mu_from_history(",
            f"\n\n{NO_ELIGIBLE_HOLD_HELPER}\n\ndef _phase3_shadow_recommended_mu_from_history(",
            "empty-window trust-hold helper",
        ),
    )
    for old, new, label in insertion_markers:
        if text.count(old) != 1:
            raise ValueError(f"{label} insertion seam drift")
        text = text.replace(old, new, 1)
    for old, new, label in (
        (OLD_AFFINE_LOOP, NEW_AFFINE_LOOP, "old-coordinate affine loop"),
        (OLD_REASON, NEW_REASON, "empty-window nomination reason"),
    ):
        if text.count(old) != 1:
            raise ValueError(f"{label} seam drift")
        text = text.replace(old, new, 1)
    call_marker = '''        no_feasible_hold = _sr_v4_all_infeasible_prune_hold_receipt(
'''
    if text.count(call_marker) != 1:
        raise ValueError("empty-window hold call seam drift")
    text = text.replace(call_marker, NO_ELIGIBLE_CALL + call_marker, 1)
    ast.parse(text)
    adapt.write_text(text, encoding="utf-8")

    regression_source = (
        core.ROOT / "test/test_static_adapt_material_window_prune_source_reuse.py"
    )
    if core.common.sha256(regression_source) != TEST_SOURCE_SHA256:
        raise ValueError("old-coordinate prune regression source drift")
    regression = source / regression_source.relative_to(core.ROOT)
    shutil.copy2(regression_source, regression)
    for relative_path, expected_sha256 in ADDITIONAL_TEST_SHA256.items():
        test_source = core.ROOT / relative_path
        if core.common.sha256(test_source) != expected_sha256:
            raise ValueError(f"frozen-source focused test drift: {relative_path}")
        test_target = source / relative_path
        test_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(test_source, test_target)
    index_regression = (
        source / "test/test_static_adapt_material_window_prune_model_index_map.py"
    )
    index_regression_text = index_regression.read_text(encoding="utf-8")
    old_index_assertion = (
        '    assert "for model_index, post_index in enumerate(" in nomination\n'
    )
    new_index_assertions = (
        '    assert "for model_index, post_index in affine_deletion_coordinates" '
        'in nomination\n'
        '    assert "_sr_material_window_prune_deletion_coordinates(" in '
        'nomination\n'
    )
    if index_regression_text.count(old_index_assertion) != 1:
        raise ValueError("material-window model-index regression seam drift")
    index_regression.write_text(
        index_regression_text.replace(
            old_index_assertion,
            new_index_assertions,
            1,
        ),
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
            "test/test_static_adapt_material_window_prune_source_reuse.py",
            "test/test_static_adapt_material_window_prune_receipt_recovery.py",
            "test/test_static_adapt_prune_source_geometry_threading.py",
            "test/test_static_adapt_material_window_prune_model_index_map.py",
            "test/test_static_adapt_sr_trust_prune.py",
            "test/test_static_adapt_sr_v4_runtime.py",
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
        "schema": "paper_i_sr_test2_old_coordinate_prune_nomination_repair_v1",
        "predecessor_bundle": BASE_ID,
        "predecessor_source_archive_sha256": BASE_SOURCE_SHA256,
        "predecessor_cluster": 9309508,
        "detected_by": "six_regime_first_prune_eligible_round_execution",
        "failure_class": "admitted_singleton_misclassified_as_prune_candidate",
        "recovery_authority": "retained_old_W_coordinates_only_v1",
        "empty_window_policy": "zero_query_hold_no_nominee_v1",
        "changed_paths": [
            "pipelines/static_adapt/adapt_pipeline.py",
            "test/test_static_adapt_material_window_prune_source_reuse.py",
            "test/test_static_adapt_sr_trust_prune.py",
            "test/test_static_adapt_sr_v4_runtime.py",
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
            "sr-material-window-fsprune-verify-r0-r50-20260722-v7": (
                "sr-material-window-fsprune-verify-r0-r50-20260722-v8"
            ),
            "sr-material-window-fsprune-verify-r0-r{target}-20260722-v7": (
                "sr-material-window-fsprune-verify-r0-r{target}-20260722-v8"
            ),
        }
    )
    if args.archive_preflight:
        core.archive_preflight()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
