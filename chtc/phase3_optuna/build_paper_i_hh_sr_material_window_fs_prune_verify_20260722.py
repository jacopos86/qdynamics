#!/usr/bin/env python3
"""Build Test 2 from the validated Test-1 material-window source lock.

The source overlay is intentionally narrow.  The large live ``adapt_pipeline``
diff is reduced to a reviewed exact-hunk allowlist containing only the
material-window prune-source reuse, immutable keep/delete verification, route
registration, and strict four-view estimator accounting.  The source archive
therefore cannot silently inherit unrelated working-tree changes.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_paper_i_hh_sr_material_window_anchor_20260721 as common


ROOT = common.ROOT
INPUT = common.INPUT
BASE_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_all_six_"
    "r50_20260721_v1_chtc"
)
BASE_BATCH = "paper-i-hh-sr-material-window-six-r50-20260721-v1"
BASE = INPUT / BASE_ID
OUTPUT_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_fs_prune_verify_"
    "all_six_r50_20260722_v1_chtc"
)
OUTPUT_BATCH = "paper-i-hh-sr-material-window-fsprune-verify-six-r50-20260722-v1"
OUTPUT = INPUT / OUTPUT_ID

PARENT_ALIAS = common.CHILD_ALIAS
PARENT_PROFILE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "material_window_symmetric_cost_no_prune_v1"
)
PARENT_DIGEST = common.CHILD_DIGEST
CHILD_ALIAS = (
    "sr_snake_symmetric_cost_projected_phase3_no_overlap_trust_material_"
    "window_fs_prune_verify_v1"
)
CHILD_PROFILE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "material_window_symmetric_cost_fs_prune_keep_verify_v1"
)
CHILD_DIGEST = "b43b23181ab1d93294fd2fb4ab96b32f7669c82db38082c86af39636cdf05201"
BASE_SOURCE_SHA256 = (
    "ced6b10d6bfbe4ae6a54495ff2ef4747a90036fa2027b0386555d016d5869a05"
)
BASE_ADAPT_SHA256 = (
    "472f7e00f677f4b85198170081e71637611933b86d7cd1c21c801b4c26355086"
)
LIVE_ADAPT_SHA256 = (
    "fb29fd88cee109c4cd49277f5a6c8edb70132819d946a9388c78b377d8c33e34"
)
LIVE_FILE_SHA256 = {
    "pipelines/static_adapt/sr_snake_route_profile.py": (
        "ef87ebcebd55b4a1b73dc3d4b32c4ceae6a4beecba0314b1000d4bf650403e59"
    ),
    "pipelines/static_adapt/estimator_call_ledger.py": (
        "d51a7a2da713c82c9e5050394eb0ca77a3f08b1da506e1fcac60c5ab142b2973"
    ),
    "test/test_static_adapt_sr_v4_runtime.py": (
        "fc53c7a01e8a16bb7cccfc3ab57a8d1ddfa13c9a4a4e5c12d91ada42ecef8015"
    ),
    "test/test_static_adapt_phase3_material_window_route_profile.py": (
        "9f158aa6d2b5377f9d0c0299e3199bf530b39b7c7e8d009c8cd8259f33dd290a"
    ),
}

# (old_start, complete unified-hunk SHA-256).  Generated against the exact
# validated Test-1 archive source and reviewed for the Test-2 contract only.
ADAPT_TEST2_HUNK_ALLOWLIST = (
    (863, "55d4a3113e12f86f5c4dfedab3518a915c99e0a0baadb5a61632f7d2ca55fdd6"),
    (1352, "69bc354ad8afd85d6cb73f3f5aafd9fba7c1820a16d26ef798f2ffaee7822e03"),
    (1364, "773f1bd7f40e8cfd4e630c430c9771edb7dd71e1c185cccf0a16b3d47a899d88"),
    (1400, "f47d91c8b906254c3aa5e609063efae2a18f10239da0118a8466bd906a93715d"),
    (1424, "c9fa404967a0fa1d328f1a5db86ee962ec8e682dfc75f863bf84e6fbb2565979"),
    (3511, "6274bff4ddf5fc89b823a7d0c0d19d0edd479e323e44fbe253707573323cbcb5"),
    (6532, "9965a8c8bd6374c96817432fe3fc2b3a026085cfc9bbcf3238b77eb1763dafff"),
    (6569, "c9a2e490af07b39981c2e8e7822d87cc014ddbcd4788f39cbabf605b5b63d63c"),
    (10225, "aa43ae3acdc37de69cdf81f3a31bbb126a833fb8fbb937f99f95242ea84dc19f"),
    (10296, "57b8c298134d591b2167b8cfef52149d67d96815f8116172fdcf3fb8a572edb6"),
    (10324, "419c7977c4afc9cf6fa91c6545fc4671adaa3b0e357ab00ffc8504fe9656766e"),
    (10354, "afcee7c8ce49bf800e9da0a989e73a191bbc8781295a9134f8d8b343d755aebe"),
    (10413, "217ef1dfb690ccf5dfa096caeba97becdd546812c10897cfa75fa6eaf5b1cebf"),
    (16996, "35f9dfaf4cc6a728712c1a49c6f17e4c9322bd51cfc47d325cf297484eae6091"),
    (17669, "01462ca3fa371dcfb8f3dbd5cea2510843fd8916b5c88076adb50ea5b99cac3c"),
    (17761, "483353ceeb6c94739124d5bd5da6495c2152043e47c36cb87fad866c60b541ea"),
    (18207, "3937eb07cbc6d3031dce34348aa59b3c5e19f4fce528b573d2a7e754994cf0a3"),
    (19905, "31f2fcad5964e32ccf7b382c6e5cb5e576a18fd3d03d6267d4a1ae20646d65b4"),
    (19947, "152d3e64d19816a4ba242e2c2f99df1e7cb482ea427181955a87fb356d46c9dc"),
    (20045, "f5a691feaafa2e1bc7a13a1547151bfe93aefb1f099a41e612e38cbfc7507548"),
    (48222, "8ad14c2186afe09a0398ec8c234c99689b0fb19e983430b2833e6977ce4ea98b"),
    (49801, "97d29203fb82a84711571c4dc29e8a34b086c7f1424dd520b4619a905fb2556f"),
    (53663, "a3f8bf337dfd64fd21f85527e4c644787b70f3e3d7fd7d3fb768c053f6012c6a"),
)

CHANGED_EXECUTION_FIELDS = (
    "phase1_prune_enabled",
    "phase1_prune_mode",
    "phase1_prune_max_candidates",
    "phase1_prune_local_window_size",
    "phase1_prune_recovery_trust_radius",
    "phase1_prune_schur_nomination_route",
    "phase1_prune_metric_schur_mu",
    "phase1_prune_metric_schur_solve_mode",
    "phase1_prune_metric_schur_cost_weighting",
    "phase1_prune_trust_update_policy",
    "phase1_prune_metric_mu_update_policy",
    "phase1_prune_endpoint_overlap_policy",
)


def _load(path: Path) -> dict[str, Any]:
    return common.load(path)


def _dump(path: Path, value: Any) -> None:
    common.dump(path, value)


def _extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(destination, filter="data")


def _isolated_contracts(source: Path) -> dict[str, dict[str, Any]]:
    code = (
        "import json\n"
        "from pipelines.static_adapt.sr_snake_route_profile import "
        "canonical_sr_snake_contract, canonical_sr_snake_contract_sha256\n"
        f"aliases={[PARENT_ALIAS, CHILD_ALIAS]!r}\n"
        "print(json.dumps({a:{'digest':canonical_sr_snake_contract_sha256(a),"
        "'contract':canonical_sr_snake_contract(a)} for a in aliases},sort_keys=True))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONNOUSERSITE", None)
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=source, env=env,
        check=True, capture_output=True, text=True,
    )
    value = json.loads(completed.stdout)
    actual = {key: item["digest"] for key, item in value.items()}
    expected = {PARENT_ALIAS: PARENT_DIGEST, CHILD_ALIAS: CHILD_DIGEST}
    if actual != expected:
        raise ValueError(f"isolated Test-2 route digest drift: {actual}")
    return value


def _build_source(temp: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    base_archive = BASE / "source_locked.tar.gz"
    if common.sha256(base_archive) != BASE_SOURCE_SHA256:
        raise ValueError("validated Test-1 source archive hash drift")
    source = temp / "source"
    _extract(base_archive, source)
    adapt_path = source / "pipelines/static_adapt/adapt_pipeline.py"
    live_adapt = ROOT / "pipelines/static_adapt/adapt_pipeline.py"
    if common.sha256(adapt_path) != BASE_ADAPT_SHA256:
        raise ValueError("validated Test-1 adapt source hash drift")
    if common.sha256(live_adapt) != LIVE_ADAPT_SHA256:
        raise ValueError("reviewed live Test-2 adapt source hash drift")
    base_text = adapt_path.read_text(encoding="utf-8")
    live_text = live_adapt.read_text(encoding="utf-8")
    available = common.unified_hunks(base_text, live_text)
    by_identity = {
        (int(item["old_start"]), str(item["sha256"])): item
        for item in available
    }
    missing = sorted(set(ADAPT_TEST2_HUNK_ALLOWLIST).difference(by_identity))
    if missing:
        raise ValueError(f"reviewed Test-2 adapt hunks missing: {missing}")
    selected = [by_identity[key] for key in ADAPT_TEST2_HUNK_ALLOWLIST]
    adapted = common.apply_unified_hunks(base_text, selected)
    for forbidden in (
        "FORMAL_SELECTOR_PHASE_MODEL_SINGLETON_REFEED_V1",
        "def _resolve_parent_sector_filter_policy(",
        "complete authoritative Phase-II population",
        "def _phase2_geometry_payload_in_anchor_order(",
    ):
        if forbidden in adapted:
            raise ValueError(f"unrelated live-tree source drift entered Test 2: {forbidden}")
    adapt_path.write_text(adapted, encoding="utf-8")

    for relative, expected_sha in LIVE_FILE_SHA256.items():
        live = ROOT / relative
        if common.sha256(live) != expected_sha:
            raise ValueError(f"reviewed Test-2 overlay hash drift: {relative}")
        shutil.copy2(live, source / relative)

    contracts = _isolated_contracts(source)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONNOUSERSITE", None)
    subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q",
            "test/test_static_adapt_phase3_material_window_route_profile.py",
            "test/test_static_adapt_sr_v4_runtime.py",
            "test/test_static_adapt_phase3_material_window.py",
            "test/test_static_adapt_phase3_material_window_runtime_contract.py",
        ],
        cwd=source, env=env, check=True,
    )
    common.strip_bytecode(source)
    archive = temp / "source_locked.tar.gz"
    common.deterministic_archive(source, archive)
    overlay = {
        "schema": "paper_i_sr_material_window_fs_prune_verify_source_overlay_v1",
        "parent_source_archive_sha256": BASE_SOURCE_SHA256,
        "changed_paths": sorted([
            "pipelines/static_adapt/adapt_pipeline.py",
            *LIVE_FILE_SHA256,
        ]),
        "adapt_hunk_selection_policy": (
            "exact_old_line_and_full_unified_hunk_sha256_allowlist_v1"
        ),
        "adapt_selected_hunks": [
            {"old_start": int(item["old_start"]), "sha256": str(item["sha256"])}
            for item in selected
        ],
        "scientific_scope": (
            "test1_plus_live_fs_prune_minimal_immutable_keep_delete_verify_v1"
        ),
        "unrelated_live_tree_changes_included": False,
    }
    return archive, contracts, overlay


def _patch_validator(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    old = '        "phase1_prune_enabled": False,\n'
    new = '''        "phase1_prune_enabled": True,
        "phase1_prune_mode": "live",
        "phase1_prune_max_candidates": 1,
        "phase1_prune_local_window_size": 0,
        "phase1_prune_recovery_trust_radius": 0.125,
        "phase1_prune_schur_nomination_route": (
            "full_logical_fs_trust_delete_refit_v1"
        ),
        "phase1_prune_metric_schur_mu": 0.0,
        "phase1_prune_metric_schur_solve_mode": (
            "affine_deletion_global_trust_v1"
        ),
        "phase1_prune_metric_schur_cost_weighting": "off",
        "phase1_prune_trust_update_policy": "modeled_local_fs_conservative_v1",
        "phase1_prune_metric_mu_update_policy": "off",
        "phase1_prune_endpoint_overlap_policy": "off",
'''
    if text.count(old) != 1:
        raise ValueError("Test-2 validator prune-setting seam drift")
    text = text.replace(old, new, 1)
    loop = '''    observed_fallback_rounds: list[int] = []
    for outer_iteration, (history_raw, checkpoint_raw) in enumerate(
'''
    loop_new = '''    observed_fallback_rounds: list[int] = []
    prune_round_summaries: list[dict[str, Any]] = []
    for outer_iteration, (history_raw, checkpoint_raw) in enumerate(
'''
    if text.count(loop) != 1:
        raise ValueError("Test-2 validator history-loop seam drift")
    text = text.replace(loop, loop_new, 1)
    receipt_seam = '''        validate_phase2_curvature_receipt(
            row.get("phase2_curvature_receipt"), outer_iteration=outer_iteration
        )

        active_count = int(row.get("phase3_active_logical_coordinate_count", -1))
'''
    receipt_new = '''        validate_phase2_curvature_receipt(
            row.get("phase2_curvature_receipt"), outer_iteration=outer_iteration
        )
        prune_raw = _mapping(
            row.get("post_admission_prune"),
            field=f"round {outer_iteration} post-admission prune",
        )
        prune_summary = validate_live_prune_round(
            prune_raw, outer_iteration=outer_iteration,
        )
        prune_round_summaries.append(prune_summary)
        reuse = prune_raw.get("phase1_prune_source_geometry_reuse")
        if reuse is not None:
            reuse = _mapping(
                reuse, field=f"round {outer_iteration} source-geometry reuse",
            )
            if (
                reuse.get("schema")
                != "sr_material_window_prune_source_geometry_reuse_v1"
                or int(reuse.get("incremental_quantum_query_charge", -1)) != 0
                or reuse.get("duplicate_measurement_performed") is not False
                or reuse.get("unsupported_logical_coordinates_nominated") is not False
                or not list(reuse.get("primitive_ids", []))
            ):
                raise ValueError(
                    f"round {outer_iteration}: prune source-geometry reuse drift"
                )
        if prune_summary.get("executed") is True:
            beam = _mapping(
                prune_raw.get("minimal_keep_prune_verification_beam"),
                field=f"round {outer_iteration} minimal keep/prune beam",
            )
            keep = _mapping(beam.get("keep_branch"), field="immutable keep branch")
            sibling = _mapping(beam.get("prune_sibling"), field="prune sibling")
            work = _mapping(
                prune_raw.get("phase1_prune_exact_refit_work_accounting"),
                field="exact prune work accounting",
            )
            expected_winner = (
                "prune_sibling" if prune_summary.get("accepted") is True
                else "keep_branch"
            )
            if (
                beam.get("schema") != "minimal_keep_prune_verification_beam_v1"
                or beam.get("historical_admission_beam_used") is not False
                or keep.get("measurement_work") != 0
                or keep.get("classical_snapshot_only") is not True
                or keep.get("intact_before_decision") is not True
                or keep.get("destructively_mutated_then_restored") is not False
                or sibling.get("delete_and_refit_measured") is not True
                or sibling.get("measurement_work_is_real") is not True
                or sibling.get("estimator_trial_branch_id")
                != work.get("estimator_trial_branch_id")
                or beam.get("winner") != expected_winner
                or beam.get(
                    "rejected_sibling_discarded_without_survivor_restore"
                ) is not True
                or int(beam.get("rollback_classical_query_charge", -1)) != 0
            ):
                raise ValueError(
                    f"round {outer_iteration}: minimal keep/prune verification drift"
                )

        active_count = int(row.get("phase3_active_logical_coordinate_count", -1))
'''
    if text.count(receipt_seam) != 1:
        raise ValueError("Test-2 validator Phase-II receipt seam drift")
    text = text.replace(receipt_seam, receipt_new, 1)
    ledger_seam = '''    ledger = validate_ledger(
        ledger_sidecar,
        _mapping(adapt.get("estimator_call_accounting"), field="result accounting"),
    )
    estimator_receipts = validate_active_prefix_estimator_receipts(
        adapt=adapt,
        ledger_summary=ledger,
        target_round=target_round,
    )
    return {
'''
    ledger_new = '''    accounting = _mapping(
        adapt.get("estimator_call_accounting"), field="result accounting",
    )
    ledger = validate_ledger(ledger_sidecar, accounting)
    prune_views = _mapping(
        accounting.get("sr_v4_prune_trial_accounting"),
        field="strict prune S_alg views",
    )
    all_work = _mapping(prune_views.get("all_work"), field="all-work S_alg")
    winning = _mapping(accounting.get("winning_lineage"), field="winning S_alg")
    shared = _mapping(
        prune_views.get("shared_source_state"), field="shared-source S_alg",
    )
    winning_only = _mapping(
        prune_views.get("winning_lineage_excluding_shared_source"),
        field="winning-exclusive S_alg",
    )
    discarded = _mapping(
        prune_views.get("discarded_prune_only_by_unique_set_difference"),
        field="discarded prune S_alg",
    )
    reconciliation = _mapping(
        prune_views.get("primitive_set_reconciliation"),
        field="prune primitive-set reconciliation",
    )
    if (
        prune_views.get("schema") != "sr_v4_prune_estimator_accounting_views_v1"
        or int(all_work.get("S_alg", -1)) != int(ledger["all_branch_s_alg"])
        or int(shared.get("S_alg", -1))
        + int(winning_only.get("S_alg", -1))
        + int(discarded.get("S_alg", -1))
        != int(all_work.get("S_alg", -2))
        or int(winning.get("S_alg", -1))
        != int(shared.get("S_alg", -2)) + int(winning_only.get("S_alg", -2))
        or reconciliation.get("pairwise_disjoint") is not True
        or reconciliation.get("union_equals_all_work") is not True
        or int(reconciliation.get("all_work_S_alg", -1))
        != int(reconciliation.get("partition_S_alg", -2))
        or not list(prune_views.get("shared_source_state_reuse_receipts", []))
    ):
        raise ValueError("strict four-view prune S_alg accounting does not close")
    estimator_receipts = validate_active_prefix_estimator_receipts(
        adapt=adapt,
        ledger_summary=ledger,
        target_round=target_round,
    )
    return {
'''
    if text.count(ledger_seam) != 1:
        raise ValueError("Test-2 validator ledger seam drift")
    text = text.replace(ledger_seam, ledger_new, 1)
    return_seam = '''        "prune_rounds_executed": 0,
        "terminal_state_unchanged_from_last_ordinary_round": True,
        "ledger": ledger,
        "active_prefix_estimator_ledger_receipts": estimator_receipts,
    }
'''
    return_new = '''        "prune_rounds_executed": sum(
            1 for item in prune_round_summaries if item.get("executed") is True
        ),
        "terminal_state_unchanged_from_last_ordinary_round": True,
        "live_prune_rounds_executed": sum(
            1 for item in prune_round_summaries if item.get("executed") is True
        ),
        "live_prune_rounds_accepted": sum(
            1 for item in prune_round_summaries
            if item.get("executed") is True and item.get("accepted") is True
        ),
        "prune_s_alg_views": {
            "all_work": dict(all_work),
            "winning_lineage": dict(winning),
            "rejected_or_discarded_prune_branch": dict(discarded),
            "shared_source_state": dict(shared),
        },
        "ledger": ledger,
        "active_prefix_estimator_ledger_receipts": estimator_receipts,
    }
'''
    if text.count(return_seam) != 1:
        raise ValueError("Test-2 validator return seam drift")
    text = text.replace(return_seam, return_new, 1)
    ast.parse(text)
    path.write_text(text, encoding="utf-8")


def _patch_run_job(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    segment_old = "sr-material-window-r0-r{target}-20260721-v1"
    segment_new = "sr-material-window-fsprune-verify-r0-r{target}-20260722-v1"
    if text.count(segment_old) != 1:
        raise ValueError("Test-2 run-job segment seam drift")
    text = text.replace(segment_old, segment_new, 1)
    execution_old = '        "phase1_prune_enabled": False,\n'
    execution_new = '''        "phase1_prune_enabled": True,
        "phase1_prune_mode": "live",
        "phase1_prune_max_candidates": 1,
        "phase1_prune_local_window_size": 0,
        "phase1_prune_recovery_trust_radius": 0.125,
        "phase1_prune_schur_nomination_route": (
            "full_logical_fs_trust_delete_refit_v1"
        ),
        "phase1_prune_metric_schur_mu": 0.0,
        "phase1_prune_metric_schur_solve_mode": (
            "affine_deletion_global_trust_v1"
        ),
        "phase1_prune_metric_schur_cost_weighting": "off",
        "phase1_prune_trust_update_policy": "modeled_local_fs_conservative_v1",
        "phase1_prune_metric_mu_update_policy": "off",
        "phase1_prune_endpoint_overlap_policy": "off",
'''
    if text.count(execution_old) != 1:
        raise ValueError("Test-2 run-job prune execution seam drift")
    text = text.replace(execution_old, execution_new, 1)
    semantics_old = '        "pruning_active": False,\n'
    semantics_new = '''        "pruning_active": True,
        "prune_execution_scope": "live_only_v1",
        "prune_nomination_count_per_round_max": 1,
        "prune_source_geometry_policy": (
            "reuse_measured_source_active_gram_hessian_blocks_v1"
        ),
        "prune_verification_beam": (
            "minimal_immutable_keep_vs_one_delete_refit_sibling_v1"
        ),
        "prune_keep_branch_mutation_policy": (
            "immutable_never_destructively_mutated_v1"
        ),
        "prune_rollback_classical_query_charge": 0,
        "prune_rejected_branch_measurements_in_all_work_s_alg": True,
        "historical_admission_beam_active": False,
'''
    if text.count(semantics_old) != 1:
        raise ValueError("Test-2 run-job prune semantic seam drift")
    text = text.replace(semantics_old, semantics_new, 1)
    ast.parse(text)
    path.write_text(text, encoding="utf-8")


def _route_job(
    base_job: Mapping[str, Any], *, contract: Mapping[str, Any], archive_sha: str,
    archive_manifest_sha: str, revision_sha: str, physics_sha: str,
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    replacements = {
        BASE_ID: OUTPUT_ID,
        BASE_BATCH: OUTPUT_BATCH,
        PARENT_ALIAS: CHILD_ALIAS,
        PARENT_PROFILE: CHILD_PROFILE,
        PARENT_DIGEST: CHILD_DIGEST,
        BASE_SOURCE_SHA256: archive_sha,
        "sr-material-window-r0-r50-20260721-v1": (
            "sr-material-window-fsprune-verify-r0-r50-20260722-v1"
        ),
    }
    job = common.replace_tree(copy.deepcopy(base_job), replacements)
    job["bundle_id"] = OUTPUT_ID
    job["batch_name"] = OUTPUT_BATCH
    argv = list(job["command"]["argv"])
    argv[argv.index("--sr-route-profile") + 1] = CHILD_ALIAS
    job["command"]["argv"] = argv
    job["route_identity"].update({
        "profile_request": CHILD_ALIAS,
        "profile_resolved": CHILD_PROFILE,
        "profile_contract": copy.deepcopy(contract),
        "profile_contract_sha256": CHILD_DIGEST,
    })
    job["evidence_requirements"].update({
        "live_fs_trust_prune_each_round_required": True,
        "minimal_immutable_keep_delete_refit_verification_required": True,
        "historical_admission_beam_forbidden": True,
        "prune_source_geometry_reuse_required": True,
        "strict_prune_s_alg_four_view_closure_required": True,
    })
    job["source_lock"].update({
        "source_archive": f"chtc/phase3_optuna/input/{OUTPUT_ID}/source_locked.tar.gz",
        "source_archive_sha256": archive_sha,
        "source_archive_manifest": f"chtc/phase3_optuna/input/{OUTPUT_ID}/source_archive_manifest.json",
        "source_archive_manifest_sha256": archive_manifest_sha,
        "source_revision_manifest": f"chtc/phase3_optuna/input/{OUTPUT_ID}/source_revision_manifest.json",
        "source_revision_manifest_sha256": revision_sha,
        "physics_reference_lock": f"chtc/phase3_optuna/input/{OUTPUT_ID}/physics_and_exact_reference_lock.json",
        "physics_reference_lock_sha256": physics_sha,
        "test2_source_overlay": copy.deepcopy(overlay),
    })
    changes = {
        key: contract["execution_settings"][key]
        for key in CHANGED_EXECUTION_FIELDS
    }
    job["source_locked_sensitivity"] = {
        "schema": "source_locked_sensitivity_candidate_row_v1",
        "source_route_contract_sha256": PARENT_DIGEST,
        "candidate_route_contract_sha256": CHILD_DIGEST,
        "only_intended_execution_field_changes": changes,
        "non_swept_settings_diff": [],
    }
    return job


def build() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"immutable Test-2 bundle already exists: {OUTPUT}")
    if not BASE.is_dir():
        raise FileNotFoundError(BASE)
    with tempfile.TemporaryDirectory(prefix="paper-i-material-window-test2-") as raw:
        temp = Path(raw)
        archive, contracts, overlay = _build_source(temp)
        archive_sha = common.sha256(archive)
        shutil.copytree(BASE, OUTPUT, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        common.clean_inherited_bundle_state(OUTPUT)
        shutil.copy2(archive, OUTPUT / "source_locked.tar.gz")
        source_inventory = common.inventory(temp / "source")

    replacements = {
        BASE_ID: OUTPUT_ID,
        BASE_BATCH: OUTPUT_BATCH,
        PARENT_ALIAS: CHILD_ALIAS,
        PARENT_PROFILE: CHILD_PROFILE,
        PARENT_DIGEST: CHILD_DIGEST,
        BASE_SOURCE_SHA256: archive_sha,
        "sr-material-window-r0-r50-20260721-v1": (
            "sr-material-window-fsprune-verify-r0-r50-20260722-v1"
        ),
    }
    archive_manifest = common.replace_tree(
        _load(BASE / "source_archive_manifest.json"), replacements,
    )
    archive_manifest.update({
        "archive": f"chtc/phase3_optuna/input/{OUTPUT_ID}/source_locked.tar.gz",
        "archive_sha256": archive_sha,
        "archive_size_bytes": (OUTPUT / "source_locked.tar.gz").stat().st_size,
        "file_count": len(source_inventory),
        "files": source_inventory,
        "test2_source_overlay": overlay,
    })
    _dump(OUTPUT / "source_archive_manifest.json", archive_manifest)
    archive_manifest_sha = common.sha256(OUTPUT / "source_archive_manifest.json")
    revision = common.replace_tree(
        _load(BASE / "source_revision_manifest.json"), replacements,
    )
    revision.update({
        "profile_request": CHILD_ALIAS,
        "profile_resolved": CHILD_PROFILE,
        "profile_contract_sha256": CHILD_DIGEST,
        "test2_source_overlay": overlay,
        "source_locked_route_transition": {
            "schema": "paper_i_sr_material_window_fs_prune_route_transition_v1",
            "parent_profile_request": PARENT_ALIAS,
            "parent_profile_resolved": PARENT_PROFILE,
            "parent_profile_contract_sha256": PARENT_DIGEST,
            "candidate_profile_request": CHILD_ALIAS,
            "candidate_profile_resolved": CHILD_PROFILE,
            "candidate_profile_contract_sha256": CHILD_DIGEST,
            "changed_execution_fields": list(CHANGED_EXECUTION_FIELDS),
            "non_swept_settings_diff": [],
        },
    })
    _dump(OUTPUT / "source_revision_manifest.json", revision)
    revision_sha = common.sha256(OUTPUT / "source_revision_manifest.json")
    physics = common.replace_tree(
        _load(BASE / "physics_and_exact_reference_lock.json"), replacements,
    )
    _dump(OUTPUT / "physics_and_exact_reference_lock.json", physics)
    physics_sha = common.sha256(OUTPUT / "physics_and_exact_reference_lock.json")

    contract = contracts[CHILD_ALIAS]["contract"]
    jobs: list[str] = []
    normalized_paths: list[str] = []
    for base_job_path in sorted((BASE / "jobs").glob("*.json")):
        job = _route_job(
            _load(base_job_path), contract=contract, archive_sha=archive_sha,
            archive_manifest_sha=archive_manifest_sha, revision_sha=revision_sha,
            physics_sha=physics_sha, overlay=overlay,
        )
        out = OUTPUT / "jobs" / base_job_path.name
        _dump(out, job)
        jobs.append(str(out.relative_to(ROOT)))
        normalized = common.replace_tree(
            _load(BASE / "normalized_manifests" / base_job_path.name), replacements,
        )
        normalized.update({
            "bundle_id": OUTPUT_ID,
            "batch_name": OUTPUT_BATCH,
            "command_argv": copy.deepcopy(job["command"]["argv"]),
            "route_identity": copy.deepcopy(job["route_identity"]),
            "evidence_requirements": copy.deepcopy(job["evidence_requirements"]),
            "source_lock": copy.deepcopy(job["source_lock"]),
            "source_locked_sensitivity": copy.deepcopy(job["source_locked_sensitivity"]),
        })
        normalized_out = OUTPUT / "normalized_manifests" / base_job_path.name
        _dump(normalized_out, normalized)
        normalized_paths.append(str(normalized_out.relative_to(ROOT)))

    for relative in (
        "run_job.py", "evidence_validation.py", "validate_fetched.py",
        "execute_source_locked_job.sh",
    ):
        common.patch_text(OUTPUT / relative, replacements)
    _patch_validator(OUTPUT / "evidence_validation.py")
    _patch_run_job(OUTPUT / "run_job.py")
    queue_text = common.replace_tree(
        (BASE / "queue.tsv").read_text(encoding="utf-8"), replacements,
    )
    (OUTPUT / "queue.tsv").write_text(queue_text, encoding="utf-8")
    queue_rel = f"chtc/phase3_optuna/input/{OUTPUT_ID}/queue.tsv"
    (OUTPUT / "submit.sub").write_text(
        common.submit_text(OUTPUT_ID, OUTPUT_BATCH, archive_sha, queue_rel),
        encoding="utf-8",
    )
    route_parity = {
        "schema": "paper_i_sr_material_window_fs_prune_route_parity_v1",
        "status": "pass",
        "parent_route_contract_sha256": PARENT_DIGEST,
        "candidate_route_contract_sha256": CHILD_DIGEST,
        "changed_execution_fields": list(CHANGED_EXECUTION_FIELDS),
        "non_swept_settings_diff": [],
    }
    _dump(OUTPUT / "route_parity.json", route_parity)
    scientific = {
        "schema": "paper_i_sr_material_window_fs_prune_scientific_audit_v1",
        "status": "pass",
        "phase3_material_window_active": True,
        "phase3_supported_whitening_active": False,
        "accepted_powell_refit_whitening_active": True,
        "endpoint_overlap_measurement_active": False,
        "historical_admission_beam_active": False,
        "live_fs_trust_pruning_active": True,
        "minimal_immutable_keep_delete_refit_verification_active": True,
        "rollback_classical_query_charge": 0,
        "strict_prune_s_alg_views_required": [
            "all_work", "winning_lineage",
            "rejected_or_discarded_prune_branch", "shared_source_state",
        ],
    }
    _dump(OUTPUT / "scientific_settings_audit.json", scientific)
    sensitivity = {
        "schema": "source_locked_sensitivity_audit_v1",
        "source_bundle": BASE_ID,
        "source_archive_sha256": BASE_SOURCE_SHA256,
        "source_route_contract_sha256": PARENT_DIGEST,
        "candidate_bundle": OUTPUT_ID,
        "candidate_archive_sha256": archive_sha,
        "candidate_route_contract_sha256": CHILD_DIGEST,
        "changed_execution_fields": list(CHANGED_EXECUTION_FIELDS),
        "non_swept_settings_diff": [],
        "fanout_authorized": True,
        "status": "test1_validated_test2_fanout_authorized",
    }
    _dump(OUTPUT / "source_locked_sensitivity_audit.json", sensitivity)
    receipt = {
        "schema": "paper_i_sr_material_window_fs_prune_fanout_bundle_v1",
        "bundle_id": OUTPUT_ID,
        "batch_name": OUTPUT_BATCH,
        "source_archive_sha256": archive_sha,
        "source_archive_manifest_sha256": archive_manifest_sha,
        "source_revision_manifest_sha256": revision_sha,
        "physics_reference_lock_sha256": physics_sha,
        "parent_route_contract_sha256": PARENT_DIGEST,
        "candidate_route_contract_sha256": CHILD_DIGEST,
        "job_count": 6,
        "jobs": jobs,
        "normalized_manifests": normalized_paths,
        "submission_performed": False,
    }
    _dump(OUTPUT / "fanout_bundle_receipt.json", receipt)
    _dump(OUTPUT / "bundle_manifest.json", receipt)
    shutil.copy2(
        BASE / "material_window_threshold_source_audit.json",
        OUTPUT / "material_window_threshold_source_audit.json",
    )
    common.patch_text(OUTPUT / "material_window_threshold_source_audit.json", replacements)
    return receipt


def archive_preflight() -> None:
    with tempfile.TemporaryDirectory(prefix="paper-i-material-window-test2-preflight-") as raw:
        root = Path(raw)
        _extract(OUTPUT / "source_locked.tar.gz", root)
        target = root / "chtc/phase3_optuna/input" / OUTPUT_ID
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(OUTPUT, target)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.pop("PYTHONNOUSERSITE", None)
        for job in sorted((target / "jobs").glob("*.json")):
            subprocess.run(
                [sys.executable, str(target / "run_job.py"), "--validate-only", str(job)],
                cwd=root, env=env, check=True,
            )
        subprocess.run(
            [
                sys.executable, "-m", "pytest", "-q",
                "test/test_static_adapt_phase3_material_window_route_profile.py",
                "test/test_static_adapt_sr_v4_runtime.py",
                "test/test_static_adapt_phase3_material_window.py",
                "test/test_static_adapt_phase3_material_window_runtime_contract.py",
            ],
            cwd=root, env=env, check=True,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-preflight", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build()
    if args.archive_preflight:
        archive_preflight()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
