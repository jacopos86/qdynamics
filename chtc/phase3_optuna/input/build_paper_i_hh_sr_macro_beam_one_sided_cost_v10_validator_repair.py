#!/usr/bin/env python3
"""Build the immutable one-sided v10 post-run validator/reporting repair."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
INPUT_ROOT = ROOT / "chtc" / "phase3_optuna" / "input"
V9_BUILDER_PATH = INPUT_ROOT / "build_paper_i_hh_sr_macro_beam_cost_v9_validator_repair.py"
SPEC = importlib.util.spec_from_file_location("_macro_cost_v9_builder", V9_BUILDER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load v9 validator builder helpers")
V9 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V9)
BASE = V9.BASE

SOURCE_SHA256 = "4c40399410b67b34a89f3cadeae59a0fd901c39132ff5cc746101c78e5acccd7"
ROUTE_DIGEST = "e3b9f24af40f3572063dd0d13bcca932870505870a8cd7822453b38e01bf6096"
PREDECESSOR_NAME = (
    "paper_i_hh_sr_snake_macro_beam3x2_fs_prune_"
    "one_sided_cost_all_six_r50_20260719_v9_chtc"
)
SUCCESSOR_NAME = PREDECESSOR_NAME.replace("_v9_chtc", "_v10_chtc")
OLD_BATCH = "paper-i-hh-sr-macro-beam3x2-fsprune-onesided-six-r50-20260719-v9"
NEW_BATCH = OLD_BATCH.removesuffix("-v9") + "-v10"
CREATED_UTC = "2026-07-20T15:50:00Z"


def dump(path: Path, payload: Any) -> None:
    BASE._json_dump(path, payload)


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def patch_run_job(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    old = "expected_cost_mode='family_robust_penalty_only_v1'"
    new = "expected_cost_mode='family_robust_v1'"
    if text.count(old) != 1:
        raise RuntimeError("one-sided canonical cost-mode validator anchor drift")
    text = text.replace(old, new, 1)
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")


def patch_revalidator(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = (
        ("with the v9 validator", "with the v10 validator"),
        ("V9_BUNDLE_ID = run_job.BUNDLE_ID", "V10_BUNDLE_ID = run_job.BUNDLE_ID"),
        (
            'V6_BUNDLE_ID = V9_BUNDLE_ID.replace("_v9_chtc", "_v6_chtc")',
            'V6_BUNDLE_ID = V10_BUNDLE_ID.replace("_v10_chtc", "_v6_chtc")',
        ),
        (
            '"paper_i_sr_macro_beam_cost_v9_v6_archive_revalidation_v1"',
            '"paper_i_sr_macro_beam_cost_v10_v6_archive_revalidation_v1"',
        ),
        ('"v9_validator_bundle_id": V9_BUNDLE_ID', '"v10_validator_bundle_id": V10_BUNDLE_ID'),
    )
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError(f"v10 revalidator anchor drift: {old}")
        text = text.replace(old, new, 1)
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")


def patch_tests(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    anchor = "    def test_only_exact_v6_validation_failure_is_repairable(self):\n"
    addition = '''    def test_one_sided_cost_mode_uses_canonical_runtime_value(self):
        settings = dict(validation.COMMON_RUNTIME_SETTINGS)
        settings["phase3_hardware_cost_normalization_mode"] = "family_robust_v1"
        validation._validate_runtime_settings(
            settings, expected_cost_mode="family_robust_v1"
        )
        with self.assertRaises(ValueError):
            validation._validate_runtime_settings(
                settings, expected_cost_mode="family_robust_penalty_only_v1"
            )

    def test_compact_controller_validator_is_fail_closed(self):
        self.assertTrue(hasattr(validation, "_validate_compact_controller_history"))
        with self.assertRaises(ValueError):
            validation._validate_compact_controller_history(
                [], expected_rounds=50, fallback_rounds=[]
            )

'''
    if text.count(anchor) != 1:
        raise RuntimeError("v10 focused-test insertion anchor drift")
    text = text.replace(anchor, addition + anchor, 1)
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")


def repair_receipt() -> dict[str, Any]:
    return {
        "schema": "paper_i_sr_macro_beam_cost_validator_reporting_repair_v4",
        "classification": "non_scientific_validator_and_reporting_only_v1",
        "predecessor_bundle": PREDECESSOR_NAME,
        "successor_bundle": SUCCESSOR_NAME,
        "failed_cluster": 8900510,
        "root_causes": [
            "one_sided_runtime_cost_mode_is_family_robust_v1_not_descriptive_penalty_only_alias",
            "selected_terminal_may_precede_distinct_signed_recoverable_frontier",
            "current_checkpoint_history_is_intentionally_compact_not_full_selected_history",
        ],
        "repair_contract": {
            "selected_terminal": "validate_complete_selected_branch_history_v1",
            "recoverable_frontier": (
                "validate_signed_compact_checkpoint_refit_prune_history_v1"
            ),
            "route_scope": "validate_immutable_normalized_route_contract_v1",
            "ledger": "validate_materialized_branch_and_selected_prefix_closure_v1",
        },
        "source_archive_sha256_preserved": SOURCE_SHA256,
        "profile_contract_sha256_preserved": ROUTE_DIGEST,
        "scientific_setting_changes": [],
        "scientific_source_changes": [],
        "scientific_rerun_required_after_passing_v10_revalidation": False,
    }


def main() -> None:
    predecessor = INPUT_ROOT / PREDECESSOR_NAME
    successor = INPUT_ROOT / SUCCESSOR_NAME
    if not predecessor.is_dir():
        raise RuntimeError(f"missing v9 predecessor: {predecessor}")
    if successor.exists():
        raise RuntimeError(f"immutable successor already exists: {successor}")
    if BASE._sha256_file(predecessor / "source_locked.tar.gz") != SOURCE_SHA256:
        raise RuntimeError("v9 scientific source archive changed")
    shutil.copytree(
        predecessor,
        successor,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "submission_artifact_hashes.json",
            "SUPERSEDED_DO_NOT_SUBMIT.json",
        ),
    )
    BASE._replace_text_tree(
        successor,
        (
            (PREDECESSOR_NAME, SUCCESSOR_NAME),
            (OLD_BATCH, NEW_BATCH),
            ("20260719-v9", "20260719-v10"),
        ),
    )
    shutil.copy2(
        INPUT_ROOT / "paper_i_hh_sr_macro_beam_cost_v10_one_sided_beam_evidence_validation.py",
        successor / "beam_evidence_validation.py",
    )
    patch_run_job(successor / "run_job.py")
    patch_revalidator(successor / "revalidate_v6_archive.py")
    patch_tests(successor / "test_validator_reporting_repair.py")

    repair = repair_receipt()
    revision_path = successor / "source_revision_manifest.json"
    revision = load(revision_path)
    revision["validator_reporting_repair_v10"] = repair
    dump(revision_path, revision)
    archive_manifest_path = successor / "source_archive_manifest.json"
    archive_manifest = load(archive_manifest_path)
    archive_manifest["validator_reporting_repair_v10"] = repair
    dump(archive_manifest_path, archive_manifest)
    revision_sha = BASE._sha256_file(revision_path)
    archive_manifest_sha = BASE._sha256_file(archive_manifest_path)
    for manifest_dir in ("jobs", "normalized_manifests"):
        for path in sorted((successor / manifest_dir).glob("*.json")):
            payload = load(path)
            source_lock = payload["source_lock"]
            source_lock["source_archive_sha256"] = SOURCE_SHA256
            source_lock["source_revision_manifest_sha256"] = revision_sha
            source_lock["source_archive_manifest_sha256"] = archive_manifest_sha
            source_lock["validator_reporting_repair_v10"] = repair
            dump(path, payload)

    bundle_manifest_path = successor / "bundle_manifest.json"
    bundle_manifest = load(bundle_manifest_path)
    bundle_manifest.update(
        {
            "batch_name": NEW_BATCH,
            "bundle_id": SUCCESSOR_NAME,
            "created_utc": CREATED_UTC,
            "submission_status": "validator_only_do_not_submit",
            "validator_reporting_repair_v10": repair,
        }
    )
    dump(bundle_manifest_path, bundle_manifest)
    for name in (
        "route_parity.json", "scientific_settings_audit.json", "preflight.json",
        "archive_only_preflight.json",
    ):
        path = successor / name
        payload = load(path)
        payload["validator_reporting_repair_v10"] = repair
        dump(path, payload)

    submit_path = successor / "submit.sub"
    submit_text = submit_path.read_text(encoding="utf-8")
    if "requirements = False && TARGET.HasSIF" not in submit_text:
        if "requirements = TARGET.HasSIF" not in submit_text:
            raise RuntimeError("v10 submit fail-closed anchor drift")
        submit_text = submit_text.replace(
            "requirements = TARGET.HasSIF",
            "requirements = False && TARGET.HasSIF",
            1,
        )
        submit_path.write_text(submit_text, encoding="utf-8")
    dump(
        successor / "VALIDATOR_ONLY_DO_NOT_SUBMIT.json",
        {
            "bundle_id": SUCCESSOR_NAME,
            "classification": "post_run_validator_reporting_only_v1",
            "revalidate_command": (
                "python3 revalidate_v6_archive.py /absolute/path/to/"
                "8900510.PROC__REGIME_transfer.tar.gz --output-dir /absolute/output"
            ),
            "scientific_rerun_required": False,
        },
    )
    readme = successor / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8")
        + "\n## v10 one-sided post-run validator repair\n\n"
        + "This immutable validator-only revision recognizes the canonical "
        + "`family_robust_v1` runtime value and validates a complete selected "
        + "terminal independently from a distinct signed compact recoverable "
        + "frontier. It never changes or reruns the frozen v6 science.\n",
        encoding="utf-8",
    )

    for path in (
        successor / "beam_evidence_validation.py",
        successor / "run_job.py",
        successor / "validate_fetched.py",
        successor / "revalidate_v6_archive.py",
        successor / "test_validator_reporting_repair.py",
    ):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    if BASE._sha256_file(successor / "source_locked.tar.gz") != SOURCE_SHA256:
        raise RuntimeError("v10 scientific source archive changed")
    V9._write_inventory(successor)

    dump(
        predecessor / "SUPERSEDED_DO_NOT_SUBMIT.json",
        {
            "bundle_id": PREDECESSOR_NAME,
            "classification": "superseded_one_sided_validator_schema_repair_v1",
            "successor_bundle": SUCCESSOR_NAME,
            "scientific_source_archive_sha256_preserved": SOURCE_SHA256,
            "scientific_setting_changes": [],
        },
    )
    predecessor_submit = predecessor / "submit.sub"
    predecessor_submit_text = predecessor_submit.read_text(encoding="utf-8")
    if "requirements = False && TARGET.HasSIF" not in predecessor_submit_text:
        if "requirements = TARGET.HasSIF" not in predecessor_submit_text:
            raise RuntimeError("v9 one-sided submit fail-closed anchor drift")
        predecessor_submit.write_text(
            predecessor_submit_text.replace(
                "requirements = TARGET.HasSIF",
                "requirements = False && TARGET.HasSIF",
                1,
            ),
            encoding="utf-8",
        )
    print(successor)


if __name__ == "__main__":
    main()
