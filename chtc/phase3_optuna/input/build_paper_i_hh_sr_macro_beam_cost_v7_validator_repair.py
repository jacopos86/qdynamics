#!/usr/bin/env python3
"""Build immutable v7 validator/reporting successors from the v6 bundles."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
INPUT_ROOT = ROOT / "chtc" / "phase3_optuna" / "input"
BASE_BUILDER_PATH = INPUT_ROOT / (
    "build_paper_i_hh_sr_macro_beam3x2_fs_prune_cost_v5_successors.py"
)
SPEC = importlib.util.spec_from_file_location("_macro_cost_builder_base", BASE_BUILDER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load shared cost-arm builder helpers")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

SOURCE_SHA256 = "4c40399410b67b34a89f3cadeae59a0fd901c39132ff5cc746101c78e5acccd7"
CREATED_UTC = "2026-07-20T13:45:00Z"
REPAIR_SCHEMA = "paper_i_sr_macro_beam_cost_validator_reporting_repair_v1"


def _repair_receipt(arm: dict[str, Any], predecessor: str, successor: str) -> dict[str, Any]:
    return {
        "schema": REPAIR_SCHEMA,
        "classification": "non_scientific_validator_and_reporting_only_v1",
        "predecessor_bundle": predecessor,
        "successor_bundle": successor,
        "failed_clusters": [8900509, 8900510],
        "known_failure": (
            "ValueError: normalized candidate setting drift: "
            "phase_live_hysteresis_enabled"
        ),
        "root_cause": (
            "phase_live_hysteresis_enabled_is_not_serialized_in_result_settings"
        ),
        "replacement_gate": (
            "immutable_normalized_command_plus_route_contract_plus_every_round_"
            "full_response_receipt_v1"
        ),
        "source_archive_sha256_preserved": SOURCE_SHA256,
        "profile_contract_sha256_preserved": str(arm["route_digest"]),
        "scientific_setting_changes": [],
        "scientific_source_changes": [],
        "scientific_rerun_required_after_passing_v7_revalidation": False,
    }


def _patch_beam_validator(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    stale = '    "phase_live_hysteresis_enabled": False,\n'
    if text.count(stale) != 1:
        raise RuntimeError("v6 hysteresis result-settings anchor drift")
    text = text.replace(stale, "", 1)
    anchor = "\ndef _validate_phase12(payload_raw: Any, *, label: str) -> int:\n"
    helper = '''
def validate_hysteresis_disabled_source_receipt(
    normalized_manifest: Mapping[str, Any], *, digest: str
) -> dict[str, Any]:
    route = _mapping(
        normalized_manifest.get("route_identity"),
        field="normalized route identity",
    )
    contract = _mapping(route.get("profile_contract"), field="route contract")
    execution = _mapping(
        contract.get("execution_settings"), field="route execution settings"
    )
    argv = [
        str(value)
        for value in _sequence(
            normalized_manifest.get("command_argv"),
            field="normalized command argv",
        )
    ]
    if (
        len(digest) != 64
        or route.get("profile_contract_sha256") != digest
        or execution.get("phase_live_hysteresis_enabled") is not False
        or argv.count("--phase-live-hysteresis-disabled") != 1
        or "--phase-live-hysteresis-enabled" in argv
    ):
        raise ValueError("hysteresis-disabled source receipt drift")
    return {
        "schema": "paper_i_sr_hysteresis_disabled_source_receipt_v1",
        "status": "pass",
        "phase_live_hysteresis_disabled": True,
        "command_flag": "--phase-live-hysteresis-disabled",
        "profile_contract_sha256": digest,
        "result_settings_field_required": False,
        "behavioral_closure": "full_response_validated_each_controller_round_v1",
    }

'''
    if text.count(anchor) != 1:
        raise RuntimeError("v6 Phase-I/II validator insertion anchor drift")
    text = text.replace(anchor, "\n" + helper + anchor.lstrip("\n"), 1)
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")


def _patch_run_job(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    old_import = (
        "from beam_evidence_validation import validate_beam_parent_evidence\n"
    )
    new_import = (
        "from beam_evidence_validation import (\n"
        "    validate_beam_parent_evidence,\n"
        "    validate_hysteresis_disabled_source_receipt,\n"
        ")\n"
    )
    if text.count(old_import) != 1:
        raise RuntimeError("run_job validator import anchor drift")
    text = text.replace(old_import, new_import, 1)
    evidence_anchor = "    evidence = validate_beam_parent_evidence(\n"
    receipt_block = (
        "    normalized_runtime = load(paths[\"normalized_runtime_manifest_json\"])\n"
        "    hysteresis_receipt = validate_hysteresis_disabled_source_receipt(\n"
        "        normalized_runtime, digest=DIGEST\n"
        "    )\n"
    )
    if text.count(evidence_anchor) != 1:
        raise RuntimeError("run_job evidence call anchor drift")
    text = text.replace(evidence_anchor, receipt_block + evidence_anchor, 1)
    return_anchor = '        "scientific_evidence_validation": evidence,\n'
    if text.count(return_anchor) != 1:
        raise RuntimeError("run_job validation return anchor drift")
    text = text.replace(
        return_anchor,
        '        "hysteresis_disabled_source_receipt": hysteresis_receipt,\n'
        + return_anchor,
        1,
    )
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")


def _patch_validate_fetched(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    old_import = (
        "from beam_evidence_validation import validate_beam_parent_evidence\n"
    )
    new_import = (
        "from beam_evidence_validation import (\n"
        "    validate_beam_parent_evidence,\n"
        "    validate_hysteresis_disabled_source_receipt,\n"
        ")\n"
    )
    if text.count(old_import) != 1:
        raise RuntimeError("validate_fetched import anchor drift")
    text = text.replace(old_import, new_import, 1)
    normalized_anchor = '    normalized = load(output / "normalized_run_manifest.json")\n'
    receipt_block = (
        normalized_anchor
        + "    hysteresis_receipt = validate_hysteresis_disabled_source_receipt(\n"
        + "        normalized, digest=digest\n"
        + "    )\n"
    )
    if text.count(normalized_anchor) != 1:
        raise RuntimeError("validate_fetched normalized-manifest anchor drift")
    text = text.replace(normalized_anchor, receipt_block, 1)
    runtime_anchor = "    runtime_evidence = validation.get(\"scientific_evidence_validation\")\n"
    runtime_block = (
        "    if validation.get(\"hysteresis_disabled_source_receipt\") != hysteresis_receipt:\n"
        "        raise ValueError(\"runtime/fetched hysteresis receipt mismatch\")\n"
        + runtime_anchor
    )
    if text.count(runtime_anchor) != 1:
        raise RuntimeError("validate_fetched runtime comparison anchor drift")
    text = text.replace(runtime_anchor, runtime_block, 1)
    return_anchor = '        "scientific_evidence_validation": evidence,\n'
    if text.count(return_anchor) != 1:
        raise RuntimeError("validate_fetched return anchor drift")
    text = text.replace(
        return_anchor,
        '        "hysteresis_disabled_source_receipt": hysteresis_receipt,\n'
        + return_anchor,
        1,
    )
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")


def _write_inventory(bundle: Path) -> None:
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path.name == "submission_artifact_hashes.json":
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = str(path.relative_to(ROOT))
        artifacts[relative] = {
            "sha256": BASE._sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    BASE._json_dump(
        bundle / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_hh_sr_macro_beam_prune_cost_artifact_hashes_v1",
            "artifacts": artifacts,
        },
    )


def _build_arm(arm: dict[str, Any]) -> Path:
    predecessor_name = (
        "paper_i_hh_sr_snake_macro_beam3x2_fs_prune_"
        f"{arm['slug']}_cost_all_six_r50_20260719_v6_chtc"
    )
    successor_name = predecessor_name.replace("_v6_chtc", "_v7_chtc")
    predecessor = INPUT_ROOT / predecessor_name
    successor = INPUT_ROOT / successor_name
    if successor.exists():
        raise RuntimeError(f"immutable successor already exists: {successor}")
    if not predecessor.is_dir():
        raise RuntimeError(f"v6 predecessor missing: {predecessor}")
    if BASE._sha256_file(predecessor / "source_locked.tar.gz") != SOURCE_SHA256:
        raise RuntimeError("v6 source archive changed")
    shutil.copytree(
        predecessor,
        successor,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "submission_artifact_hashes.json",
            "SUPERSEDED_DO_NOT_SUBMIT.json",
            "v6_validation_receipt.json",
        ),
    )
    old_batch = (
        "paper-i-hh-sr-macro-beam3x2-fsprune-"
        f"{arm['short']}-six-r50-20260719-v6"
    )
    new_batch = old_batch.removesuffix("-v6") + "-v7"
    BASE._replace_text_tree(
        successor,
        (
            (predecessor_name, successor_name),
            (old_batch, new_batch),
            ("20260719-v6", "20260719-v7"),
        ),
    )
    _patch_beam_validator(successor / "beam_evidence_validation.py")
    _patch_run_job(successor / "run_job.py")
    _patch_validate_fetched(successor / "validate_fetched.py")
    shutil.copy2(
        INPUT_ROOT / "paper_i_hh_sr_macro_beam_cost_v7_revalidate_v6_archive.py",
        successor / "revalidate_v6_archive.py",
    )
    shutil.copy2(
        INPUT_ROOT / "paper_i_hh_sr_macro_beam_cost_v7_validation_test.py",
        successor / "test_validator_reporting_repair.py",
    )

    repair = _repair_receipt(arm, predecessor_name, successor_name)
    revision_path = successor / "source_revision_manifest.json"
    revision = json.loads(revision_path.read_text())
    revision["validator_reporting_repair"] = repair
    BASE._json_dump(revision_path, revision)
    archive_manifest_path = successor / "source_archive_manifest.json"
    archive_manifest = json.loads(archive_manifest_path.read_text())
    archive_manifest["validator_reporting_repair"] = repair
    BASE._json_dump(archive_manifest_path, archive_manifest)
    revision_sha = BASE._sha256_file(revision_path)
    archive_manifest_sha = BASE._sha256_file(archive_manifest_path)
    for manifest_dir in ("jobs", "normalized_manifests"):
        for path in sorted((successor / manifest_dir).glob("*.json")):
            payload = json.loads(path.read_text())
            source_lock = payload["source_lock"]
            source_lock["source_archive_sha256"] = SOURCE_SHA256
            source_lock["source_revision_manifest_sha256"] = revision_sha
            source_lock["source_archive_manifest_sha256"] = archive_manifest_sha
            source_lock["validator_reporting_repair"] = repair
            BASE._json_dump(path, payload)

    bundle_manifest_path = successor / "bundle_manifest.json"
    bundle_manifest = json.loads(bundle_manifest_path.read_text())
    bundle_manifest.update(
        {
            "batch_name": new_batch,
            "bundle_id": successor_name,
            "created_utc": CREATED_UTC,
            "source_archive_sha256": SOURCE_SHA256,
            "submission_status": "built_not_submitted",
            "validator_reporting_repair": repair,
        }
    )
    BASE._json_dump(bundle_manifest_path, bundle_manifest)
    for name in ("route_parity.json", "scientific_settings_audit.json", "preflight.json"):
        path = successor / name
        payload = json.loads(path.read_text())
        payload["validator_reporting_repair"] = repair
        BASE._json_dump(path, payload)
    archive_preflight_path = successor / "archive_only_preflight.json"
    archive_preflight = json.loads(archive_preflight_path.read_text())
    archive_preflight.update(
        {
            "validator_reporting_repair": repair,
            "v7_bundle_tests_passed": 0,
            "v7_archive_only_validate_rows_passed": 0,
            "v7_validator_repair_tests_passed": 0,
            "v7_frozen_archive_revalidations_passed": 0,
            "status": "pending_validation",
        }
    )
    BASE._json_dump(archive_preflight_path, archive_preflight)
    readme = successor / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8")
        + "\n## v7 validator/reporting repair\n\n"
        + "The scientific source archive, route digest, settings, and command semantics "
        + "are byte-for-byte/scientifically unchanged. The post-run validator now reads "
        + "the hysteresis-disabled policy from the immutable normalized command and route "
        + "contract because that field is not serialized in result.settings. Every-round "
        + "full-response validation remains mandatory. revalidate_v6_archive.py can "
        + "validate and report completed v6 payloads without modifying their raw archives.\n",
        encoding="utf-8",
    )
    upload = successor / "upload_artifact_list.txt"
    upload.write_text(
        upload.read_text(encoding="utf-8")
        + f"{successor.relative_to(ROOT)}/revalidate_v6_archive.py\n",
        encoding="utf-8",
    )
    if BASE._sha256_file(successor / "source_locked.tar.gz") != SOURCE_SHA256:
        raise RuntimeError("v7 scientific source archive changed")
    for path in (
        successor / "beam_evidence_validation.py",
        successor / "run_job.py",
        successor / "validate_fetched.py",
        successor / "revalidate_v6_archive.py",
        successor / "test_validator_reporting_repair.py",
    ):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    _write_inventory(successor)

    marker = {
        "bundle_id": predecessor_name,
        "classification": "superseded_post_run_validator_policy_defect_v1",
        "successor_bundle": successor_name,
        "scientific_outputs_remain_revalidatable": True,
        "scientific_setting_changes": [],
    }
    BASE._json_dump(predecessor / "SUPERSEDED_DO_NOT_SUBMIT.json", marker)
    submit = predecessor / "submit.sub"
    submit_text = submit.read_text(encoding="utf-8")
    if "requirements = False && TARGET.HasSIF" not in submit_text:
        if "requirements = TARGET.HasSIF" not in submit_text:
            raise RuntimeError("v6 submit requirement anchor drift")
        submit.write_text(
            submit_text.replace(
                "requirements = TARGET.HasSIF",
                "requirements = False && TARGET.HasSIF",
                1,
            ),
            encoding="utf-8",
        )
    return successor


def main() -> None:
    for arm in BASE.ARMS:
        print(_build_arm(dict(arm)))


if __name__ == "__main__":
    main()
