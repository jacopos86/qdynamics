#!/usr/bin/env python3
"""Build immutable v9 schema-correct validator/reporting successors from v6."""

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
CREATED_UTC = "2026-07-20T15:20:00Z"
REPAIR_SCHEMA = "paper_i_sr_macro_beam_cost_validator_reporting_repair_v3"

SOURCE_ONLY_RUNTIME_SETTINGS = {
    "phase_live_hysteresis_enabled": False,
    "phase0_pilot_enabled": False,
    "phase3_enable_batching": False,
    "adapt_beam_live_branches": 3,
    "adapt_beam_children_per_parent": 2,
    "adapt_beam_terminated_keep": 3,
    "adapt_beam_terminal_archive_mode": "legacy",
    "adapt_beam_lambda": 0.005,
    "adapt_beam_parent_workers": 1,
    "adapt_accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
}


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
        "root_cause": "resolved_route_fields_are_not_flattened_into_result_settings",
        "source_only_runtime_settings": SOURCE_ONLY_RUNTIME_SETTINGS,
        "replacement_gate": (
            "immutable_normalized_command_plus_route_contract_plus_every_round_"
            "full_response_receipt_v1"
        ),
        "source_archive_sha256_preserved": SOURCE_SHA256,
        "profile_contract_sha256_preserved": str(arm["route_digest"]),
        "scientific_setting_changes": [],
        "scientific_source_changes": [],
        "scientific_rerun_required_after_passing_v9_revalidation": False,
    }


def _patch_beam_validator(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    source_only_lines = (
        '    "phase_live_hysteresis_enabled": False,\n',
        '    "phase0_pilot_enabled": False,\n',
        '    "phase3_enable_batching": False,\n',
        '    "adapt_beam_live_branches": 3,\n',
        '    "adapt_beam_children_per_parent": 2,\n',
        '    "adapt_beam_terminated_keep": 3,\n',
        '    "adapt_beam_terminal_archive_mode": "legacy",\n',
        '    "adapt_beam_lambda": 0.005,\n',
        '    "adapt_beam_parent_workers": 1,\n',
        '    "adapt_accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",\n',
    )
    for stale in source_only_lines:
        if text.count(stale) != 1:
            raise RuntimeError(f"v6 source-only result-settings anchor drift: {stale!r}")
        text = text.replace(stale, "", 1)
    for stale in (
        '    "expanded_child_cap_per_round": 6,\n',
        '    "terminal_archive_mode": "legacy",\n',
    ):
        if text.count(stale) != 1:
            raise RuntimeError(f"v6 derived beam-telemetry anchor drift: {stale!r}")
        text = text.replace(stale, "", 1)
    anchor = "\ndef _validate_phase12(payload_raw: Any, *, label: str) -> int:\n"
    helper = '''
SOURCE_ONLY_RUNTIME_SETTINGS: Mapping[str, Any] = {
    "phase_live_hysteresis_enabled": False,
    "phase0_pilot_enabled": False,
    "phase3_enable_batching": False,
    "adapt_beam_live_branches": 3,
    "adapt_beam_children_per_parent": 2,
    "adapt_beam_terminated_keep": 3,
    "adapt_beam_terminal_archive_mode": "legacy",
    "adapt_beam_lambda": 0.005,
    "adapt_beam_parent_workers": 1,
    "adapt_accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
}


def validate_source_only_runtime_settings_receipt(
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
    semantic = _mapping(
        contract.get("semantic_invariants"), field="route semantic invariants"
    )
    argv = [
        str(value)
        for value in _sequence(
            normalized_manifest.get("command_argv"),
            field="normalized command argv",
        )
    ]
    if len(digest) != 64 or route.get("profile_contract_sha256") != digest:
        raise ValueError("source-only route digest drift")
    for key, expected in SOURCE_ONLY_RUNTIME_SETTINGS.items():
        if execution.get(key) != expected:
            raise ValueError(f"source-only route setting drift: {key}")
    if int(semantic.get("beam_expanded_child_cap_per_round", -1)) != 6:
        raise ValueError("source-only beam expanded-child cap drift")
    if (
        argv.count("--phase-live-hysteresis-disabled") != 1
        or "--phase-live-hysteresis-enabled" in argv
    ):
        raise ValueError("hysteresis-disabled command receipt drift")
    return {
        "schema": "paper_i_sr_source_only_runtime_settings_receipt_v1",
        "status": "pass",
        "phase_live_hysteresis_disabled": True,
        "command_flag": "--phase-live-hysteresis-disabled",
        "profile_contract_sha256": digest,
        "source_only_runtime_settings": dict(SOURCE_ONLY_RUNTIME_SETTINGS),
        "beam_expanded_child_cap_per_round": 6,
        "result_settings_fields_required": False,
        "behavioral_closure": "full_response_validated_each_controller_round_v1",
    }

'''
    if text.count(anchor) != 1:
        raise RuntimeError("v6 Phase-I/II validator insertion anchor drift")
    text = text.replace(anchor, "\n" + helper + anchor.lstrip("\n"), 1)
    compact_anchor = "\ndef _validate_selected_checkpoint_list(\n"
    compact_helper = '''
def _validate_compact_current_history(
    history_raw: Any,
    *,
    selected_path: Mapping[str, Any],
    expected_rounds: int,
) -> dict[str, Any]:
    compact = list(_sequence(history_raw, field="compact current history"))
    selected = list(
        _sequence(selected_path.get("history"), field="selected full history")
    )
    if len(compact) != expected_rounds or len(selected) != expected_rounds:
        raise ValueError("compact/full current history length drift")
    scalar_fields = (
        "depth",
        "branch_id",
        "parent_branch_id",
        "selected_op",
        "selected_position",
        "batch_size",
        "energy_before_opt",
        "energy_after_opt",
        "delta_energy",
        "nfev_total_before_step",
        "nfev_total_after_step",
        "nfev_step_total_delta",
    )
    selected_checkpoints = list(
        _sequence(selected_path.get("checkpoints"), field="selected checkpoints")
    )
    for outer_iteration, (compact_raw, selected_raw, checkpoint_raw) in enumerate(
        zip(compact, selected, selected_checkpoints), start=1
    ):
        compact_row = _mapping(
            compact_raw, field=f"compact current round {outer_iteration}"
        )
        selected_row = _mapping(
            selected_raw, field=f"selected full round {outer_iteration}"
        )
        checkpoint = _mapping(
            compact_row.get("active_prefix_checkpoint"),
            field=f"compact checkpoint round {outer_iteration}",
        )
        selected_checkpoint = _mapping(
            checkpoint_raw, field=f"selected checkpoint round {outer_iteration}"
        )
        for field in scalar_fields:
            if compact_row.get(field) != selected_row.get(field):
                raise ValueError(
                    f"compact/full current history drift at round {outer_iteration}: {field}"
                )
        if checkpoint.get("checkpoint_sha256") != selected_checkpoint.get(
            "checkpoint_sha256"
        ):
            raise ValueError(
                f"compact/full checkpoint drift at round {outer_iteration}"
            )
    return {
        "schema": "paper_i_sr_compact_current_full_winner_crosscheck_v1",
        "status": "pass",
        "rounds": expected_rounds,
    }

'''
    if text.count(compact_anchor) != 1:
        raise RuntimeError("v6 compact-current insertion anchor drift")
    text = text.replace(
        compact_anchor,
        "\n" + compact_helper + compact_anchor.lstrip("\n"),
        1,
    )
    old_controller_block = '''    controller_path = _validate_path(
        current_adapt.get("history"),
        expected_rounds=target_round,
        require_supported_rank=require_supported_rank,
    )
    selected_path = _validate_path(
        adapt.get("history"),
        expected_rounds=selected_round,
        require_supported_rank=require_supported_rank,
    )
'''
    new_controller_block = '''    selected_path = _validate_path(
        adapt.get("history"),
        expected_rounds=selected_round,
        require_supported_rank=require_supported_rank,
    )
    if (
        selected_round != target_round
        or relationship.get("relationship_present") is True
        or int(current_adapt.get("branch_id", -1)) != selected_branch_id
    ):
        raise ValueError(
            "frozen validator repair requires the round-target selected winner "
            "to equal the checkpoint frontier"
        )
    compact_current_receipt = _validate_compact_current_history(
        current_adapt.get("history"),
        selected_path=selected_path,
        expected_rounds=target_round,
    )
    controller_path = selected_path
'''
    if text.count(old_controller_block) != 1:
        raise RuntimeError("v6 controller/full-history validation anchor drift")
    text = text.replace(old_controller_block, new_controller_block, 1)
    return_anchor = '        "active_prefix_estimator_ledger_receipts": receipts,\n'
    if text.count(return_anchor) != 1:
        raise RuntimeError("v6 compact-current receipt return anchor drift")
    text = text.replace(
        return_anchor,
        return_anchor
        + '        "compact_current_history_receipt": compact_current_receipt,\n',
        1,
    )
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
        "    validate_source_only_runtime_settings_receipt,\n"
        ")\n"
    )
    if text.count(old_import) != 1:
        raise RuntimeError("run_job validator import anchor drift")
    text = text.replace(old_import, new_import, 1)
    evidence_anchor = "    evidence = validate_beam_parent_evidence(\n"
    receipt_block = (
        "    normalized_runtime = load(paths[\"normalized_runtime_manifest_json\"])\n"
        "    source_only_receipt = validate_source_only_runtime_settings_receipt(\n"
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
        '        "source_only_runtime_settings_receipt": source_only_receipt,\n'
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
        "    validate_source_only_runtime_settings_receipt,\n"
        ")\n"
    )
    if text.count(old_import) != 1:
        raise RuntimeError("validate_fetched import anchor drift")
    text = text.replace(old_import, new_import, 1)
    normalized_anchor = '    normalized = load(output / "normalized_run_manifest.json")\n'
    receipt_block = (
        normalized_anchor
        + "    source_only_receipt = validate_source_only_runtime_settings_receipt(\n"
        + "        normalized, digest=digest\n"
        + "    )\n"
    )
    if text.count(normalized_anchor) != 1:
        raise RuntimeError("validate_fetched normalized-manifest anchor drift")
    text = text.replace(normalized_anchor, receipt_block, 1)
    runtime_anchor = "    runtime_evidence = validation.get(\"scientific_evidence_validation\")\n"
    runtime_block = (
        "    if validation.get(\"source_only_runtime_settings_receipt\") != source_only_receipt:\n"
        "        raise ValueError(\"runtime/fetched source-only receipt mismatch\")\n"
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
        '        "source_only_runtime_settings_receipt": source_only_receipt,\n'
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
    successor_name = predecessor_name.replace("_v6_chtc", "_v9_chtc")
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
    new_batch = old_batch.removesuffix("-v6") + "-v9"
    BASE._replace_text_tree(
        successor,
        (
            (predecessor_name, successor_name),
            (old_batch, new_batch),
            ("20260719-v6", "20260719-v9"),
        ),
    )
    _patch_beam_validator(successor / "beam_evidence_validation.py")
    _patch_run_job(successor / "run_job.py")
    _patch_validate_fetched(successor / "validate_fetched.py")
    shutil.copy2(
        INPUT_ROOT / "paper_i_hh_sr_macro_beam_cost_v9_revalidate_v6_archive.py",
        successor / "revalidate_v6_archive.py",
    )
    shutil.copy2(
        INPUT_ROOT / "paper_i_hh_sr_macro_beam_cost_v9_validation_test.py",
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
            "v9_bundle_tests_passed": 0,
            "v9_archive_only_validate_rows_passed": 0,
            "v9_validator_repair_tests_passed": 0,
            "v9_frozen_archive_revalidations_passed": 0,
            "status": "pending_validation",
        }
    )
    BASE._json_dump(archive_preflight_path, archive_preflight)
    readme = successor / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8")
        + "\n## v9 validator/reporting repair\n\n"
        + "The scientific source archive, route digest, settings, and command semantics "
        + "are byte-for-byte/scientifically unchanged. The post-run validator now reads "
        + "all route-resolved source-only settings from the immutable normalized command "
        + "and route contract because those fields are not flattened into result.settings. Every-round "
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
        raise RuntimeError("v9 scientific source archive changed")
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
    for incomplete_revision in ("v7", "v8"):
        incomplete = INPUT_ROOT / successor_name.replace(
            "_v9_chtc", f"_{incomplete_revision}_chtc"
        )
        if not incomplete.is_dir():
            continue
        BASE._json_dump(
            incomplete / "SUPERSEDED_DO_NOT_SUBMIT.json",
            {
                "bundle_id": incomplete.name,
                "classification": "superseded_incomplete_validator_schema_repair_v1",
                "successor_bundle": successor_name,
                "scientific_source_archive_sha256_preserved": SOURCE_SHA256,
                "scientific_setting_changes": [],
            },
        )
        incomplete_submit = incomplete / "submit.sub"
        incomplete_submit_text = incomplete_submit.read_text(encoding="utf-8")
        if "requirements = False && TARGET.HasSIF" not in incomplete_submit_text:
            if "requirements = TARGET.HasSIF" not in incomplete_submit_text:
                raise RuntimeError(
                    f"{incomplete_revision} submit requirement anchor drift"
                )
            incomplete_submit.write_text(
                incomplete_submit_text.replace(
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
