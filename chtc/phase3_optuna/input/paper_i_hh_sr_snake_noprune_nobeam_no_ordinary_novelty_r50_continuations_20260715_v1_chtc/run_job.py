#!/usr/bin/env python3
"""Validate or execute one source-locked SR-SNAKE r30 -> r50 row."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


BUNDLE_ID = (
    "paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_"
    "r50_continuations_20260715_v1_chtc"
)
EXPECTED_SCHEMA = "paper_i_hh_sr_r30_to_r50_continuation_job_v1"
SOURCE_DEPTH = 30
TARGET_DEPTH = 50
MAX_NEW_ADMISSIONS = 20
SIGNED_PREFIX_SCHEMA = "static_adapt_signed_active_prefix_resume_sidecar_v1"
SIGNED_PREFIX_CANONICAL_NAME = "signed_active_prefix_checkpoint.json"
OUTPUT_PATH_FLAGS = {
    "--adapt-current-json",
    "--adapt-estimator-call-ledger-json",
    "--output-json",
}
CONTINUATION_VALUES = {
    "--adapt-resume-mode": "scaffold_v1",
    "--adapt-resume-boundary-refit-policy": "verified_checkpoint_no_refit_v1",
    "--adapt-segment-target-depth": "50",
    "--adapt-segment-target-controller-round": "50",
    "--adapt-segment-max-new-admissions": "20",
    "--adapt-resume-compile-smoke": "required",
    "--adapt-resume-smoke-backend": "FakeMarrakesh",
}
CONTINUATION_FLAGS = set(CONTINUATION_VALUES) | {
    "--adapt-resume-scaffold-json",
    "--adapt-segment-id",
}
ALLOWED_DIFF_FLAGS = OUTPUT_PATH_FLAGS | CONTINUATION_FLAGS | {"--adapt-max-depth"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def options(argv: Sequence[str]) -> dict[str, Any]:
    prefix = ["python3", "-m", "pipelines.static_adapt.adapt_pipeline"]
    if list(argv[:3]) != prefix:
        raise ValueError("unexpected command prefix")
    result: dict[str, Any] = {}
    index = 3
    while index < len(argv):
        flag = str(argv[index])
        if not flag.startswith("--") or flag in result:
            raise ValueError(f"invalid or duplicate option: {flag!r}")
        if index + 1 < len(argv) and not str(argv[index + 1]).startswith("--"):
            result[flag] = str(argv[index + 1])
            index += 2
        else:
            result[flag] = True
            index += 1
    return result


def validate(manifest_path: Path, manifest: Mapping[str, Any]) -> list[str]:
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("unexpected job manifest schema")
    if manifest.get("bundle_id") != BUNDLE_ID:
        raise ValueError("unexpected bundle id")
    if manifest.get("run_class") != "candidate_source_locked_continuation":
        raise ValueError("unexpected run class")
    slug = str(manifest.get("regime_slug", ""))
    if not slug or "/" in slug or ".." in slug:
        raise ValueError("unsafe regime slug")

    command = manifest.get("command")
    if not isinstance(command, Mapping):
        raise ValueError("missing command contract")
    source_argv = [str(token) for token in command["source_argv"]]
    execution_argv = [str(token) for token in command["execution_argv"]]
    source = options(source_argv)
    execution = options(execution_argv)
    changed = sorted(
        key
        for key in set(source) | set(execution)
        if source.get(key) != execution.get(key)
    )
    if changed != sorted(command.get("changed_flags", [])):
        raise ValueError("recorded changed flags differ from executable argv")
    if set(changed) - ALLOWED_DIFF_FLAGS:
        raise ValueError(f"non-approved executable drift: {changed}")
    if command.get("unexpected_differences"):
        raise ValueError("manifest records unexpected executable differences")
    source_signature = {
        key: value for key, value in source.items() if key not in ALLOWED_DIFF_FLAGS
    }
    execution_signature = {
        key: value for key, value in execution.items() if key not in ALLOWED_DIFF_FLAGS
    }
    if source_signature != execution_signature:
        raise ValueError("non-horizon route signature differs")
    if source.get("--adapt-max-depth") != str(SOURCE_DEPTH):
        raise ValueError("source max depth is not 30")
    if execution.get("--adapt-max-depth") != str(TARGET_DEPTH):
        raise ValueError("target max depth is not 50")
    for flag, expected in CONTINUATION_VALUES.items():
        if execution.get(flag) != expected:
            raise ValueError(f"continuation control mismatch: {flag}")
    if execution.get("--adapt-segment-id") != f"{slug}-r30-to-r50-v1":
        raise ValueError("segment id mismatch")

    paths = {str(key): Path(value) for key, value in manifest["paths"].items()}
    output_root = Path("raw_outputs") / BUNDLE_ID / slug
    if paths["output_root"] != output_root:
        raise ValueError("isolated output root mismatch")
    for key, path in paths.items():
        if key != "output_root" and output_root not in path.parents:
            raise ValueError(f"path escaped isolated output root: {key}")
    expected_paths = {
        "--adapt-current-json": paths["current_json"].as_posix(),
        "--adapt-estimator-call-ledger-json": paths[
            "estimator_call_ledger_json"
        ].as_posix(),
        "--output-json": paths["result_json"].as_posix(),
        "--adapt-resume-scaffold-json": paths["resume_input_json"].as_posix(),
    }
    for flag, expected in expected_paths.items():
        if execution.get(flag) != expected:
            raise ValueError(f"execution path mismatch: {flag}")

    source_lock = manifest.get("source_lock")
    if not isinstance(source_lock, Mapping):
        raise ValueError("missing source lock")
    for key in (
        "patched_source_archive",
        "no_beam_resume_patch_manifest",
        "no_beam_resume_patch",
        "source_record",
        "transferred_checkpoint",
        "transferred_source_ledger",
        "transferred_signed_prefix_sidecar",
    ):
        path = Path(str(source_lock[key]))
        expected = str(source_lock[f"{key}_sha256"])
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"source-lock hash mismatch: {key}")
    source_record = load_json(Path(str(source_lock["source_record"])))
    if source_record.get("regime_slug") != slug:
        raise ValueError("source record regime mismatch")
    if source_record.get("source_checkpoint_sha256") != source_lock.get(
        "transferred_checkpoint_uncompressed_sha256"
    ):
        raise ValueError("transferred checkpoint differs from source record")
    if int(source_record.get("source_checkpoint_size_bytes", -1)) != int(
        source_lock.get("transferred_checkpoint_uncompressed_size_bytes", -2)
    ):
        raise ValueError("transferred checkpoint size differs from source record")
    if source_lock.get("transferred_checkpoint_compression") != (
        "deterministic_gzip_mtime0_level9_v1"
    ):
        raise ValueError("checkpoint compression contract mismatch")
    if source_record.get("source_estimator_ledger_sha256") != source_lock.get(
        "transferred_source_ledger_uncompressed_sha256"
    ):
        raise ValueError("transferred source ledger differs from source record")
    if int(source_record.get("source_estimator_ledger_size_bytes", -1)) != int(
        source_lock.get("transferred_source_ledger_uncompressed_size_bytes", -2)
    ):
        raise ValueError("transferred source-ledger size differs from source record")
    if source_lock.get("transferred_source_ledger_compression") != (
        "deterministic_gzip_mtime0_level9_v1"
    ):
        raise ValueError("source-ledger compression contract mismatch")
    signed_prefix = load_json(
        Path(str(source_lock["transferred_signed_prefix_sidecar"]))
    )
    if signed_prefix.get("schema") != SIGNED_PREFIX_SCHEMA:
        raise ValueError("signed-prefix sidecar schema mismatch")
    controller_snapshot = signed_prefix.get("controller_snapshot")
    if not isinstance(controller_snapshot, Mapping):
        raise ValueError("signed-prefix sidecar lacks controller snapshot")
    controller_snapshot_sha256 = str(
        signed_prefix.get("controller_snapshot_sha256", "")
    )
    if jsonable_sha256(controller_snapshot) != controller_snapshot_sha256:
        raise ValueError("controller-snapshot digest mismatch")
    if (
        controller_snapshot.get("snapshot_version")
        != "phase123_controller_maturity_v2"
        or int(controller_snapshot.get("step_index", -1)) != SOURCE_DEPTH - 1
    ):
        raise ValueError("controller snapshot is not the round-30 maturity-v2 state")
    controller_state = signed_prefix.get("controller_state")
    if not isinstance(controller_state, Mapping) or controller_state.get(
        "schema"
    ) != "static_adapt_singleton_controller_resume_state_v1":
        raise ValueError("signed-prefix sidecar lacks typed controller state")
    history_evidence = controller_state.get("source_history_row_evidence")
    if not isinstance(history_evidence, Mapping):
        raise ValueError("controller state lacks source-history evidence")
    if jsonable_sha256(history_evidence) != controller_state.get(
        "source_history_row_evidence_sha256"
    ):
        raise ValueError("controller source-history evidence digest mismatch")
    expected_history_evidence = {
        "depth": SOURCE_DEPTH,
        "drop_policy_enabled": False,
        "drop_plateau_hits": 0,
        "stage_name": "core",
        "stage_transition_reason": "stay_core",
        "controller_snapshot_count": 1,
        "selected_feature_row_index": 0,
    }
    if dict(history_evidence) != expected_history_evidence:
        raise ValueError("controller source-history evidence drift")
    if (
        int(controller_state.get("controller_round", -1)) != SOURCE_DEPTH
        or int(controller_state.get("source_max_depth", -1)) != SOURCE_DEPTH
        or controller_state.get("phase1_residual_opened") is not False
        or controller_state.get("phase1_stage_name") != "core"
    ):
        raise ValueError("controller resume-state fields drift")
    selection_state = signed_prefix.get("selection_state")
    if not isinstance(selection_state, Mapping) or selection_state.get(
        "schema"
    ) != "static_adapt_singleton_selection_count_resume_state_v1":
        raise ValueError("signed-prefix sidecar lacks typed selection state")
    pool_size = int(selection_state.get("pool_size", -1))
    parent_indices = selection_state.get("ordered_parent_pool_indices")
    feature_row_counts = selection_state.get(
        "selected_feature_row_count_per_round"
    )
    logical_indices = selection_state.get("ordered_logical_candidate_indices")
    if (
        int(selection_state.get("controller_round", -1)) != SOURCE_DEPTH
        or selection_state.get("seq2p_logical_mode") is not False
        or not isinstance(parent_indices, list)
        or len(parent_indices) != SOURCE_DEPTH
        or not isinstance(feature_row_counts, list)
        or feature_row_counts != [1] * SOURCE_DEPTH
        or pool_size <= 0
        or any(int(index) < 0 or int(index) >= pool_size for index in parent_indices)
        or logical_indices != []
    ):
        raise ValueError("selection-count resume state drift")
    if jsonable_sha256(parent_indices) != selection_state.get(
        "ordered_parent_pool_indices_sha256"
    ) or jsonable_sha256(logical_indices) != selection_state.get(
        "ordered_logical_candidate_indices_sha256"
    ):
        raise ValueError("selection-count resume-state digest mismatch")
    signed_prefix_audit = source_record.get("signed_active_prefix_sidecar")
    if not isinstance(signed_prefix_audit, Mapping):
        raise ValueError("source record lacks signed-prefix sidecar audit")
    if signed_prefix.get("source_result_sha256") != signed_prefix_audit.get(
        "source_result_sha256"
    ):
        raise ValueError("signed-prefix source-result provenance mismatch")
    if source_lock.get("transferred_signed_prefix_sidecar_sha256") != signed_prefix_audit.get(
        "sidecar_sha256"
    ):
        raise ValueError("signed-prefix sidecar differs from source record")
    if controller_snapshot_sha256 != signed_prefix_audit.get(
        "controller_snapshot_sha256"
    ):
        raise ValueError("controller-snapshot digest differs from source record")
    if jsonable_sha256(controller_state) != signed_prefix_audit.get(
        "controller_state_sha256"
    ):
        raise ValueError("controller-state digest differs from source record")
    if jsonable_sha256(selection_state) != signed_prefix_audit.get(
        "selection_state_sha256"
    ):
        raise ValueError("selection-state digest differs from source record")
    parity = source_record.get("checkpoint_parity_and_compile_smoke", {})
    if not isinstance(parity, Mapping) or parity.get("status") != "pass":
        raise ValueError("checkpoint parity record did not pass")
    if not bool(parity.get("compile_smoke", {}).get("success")):
        raise ValueError("source checkpoint compile smoke did not pass")

    environment = {
        str(key): str(value) for key, value in manifest.get("environment", {}).items()
    }
    for key in (
        "MPLCONFIGDIR",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR",
        "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR",
        "STATIC_ADAPT_HH_POOL_CACHE_DIR",
    ):
        if key not in environment or output_root not in Path(environment[key]).parents:
            raise ValueError(f"cache/output path isolation mismatch: {key}")
    resources = manifest.get("resources", {})
    if int(resources.get("request_cpus", -1)) != 4:
        raise ValueError("request_cpus drift")
    if int(resources.get("request_disk_mb", -1)) != 61440:
        raise ValueError("request_disk drift")
    if int(resources.get("max_runtime_s", -1)) != 259200:
        raise ValueError("MaxRuntime drift")
    expected_memory = 32768 if slug == "strong_weak_u8" else 40960
    if int(resources.get("request_memory_mb", -1)) != expected_memory:
        raise ValueError("request_memory drift")
    return execution_argv


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    validate_only = False
    if args and args[0] == "--validate-only":
        validate_only = True
        args = args[1:]
    if len(args) != 1:
        raise SystemExit("usage: run_job.py [--validate-only] JOB_MANIFEST.json")
    manifest_path = Path(args[0])
    manifest = load_json(manifest_path)
    execution_argv = validate(manifest_path, manifest)
    if validate_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "validation_only": True,
                    "job_manifest": manifest_path.as_posix(),
                    "job_manifest_sha256": sha256(manifest_path),
                },
                sort_keys=True,
            )
        )
        return 0

    paths = {str(key): Path(value) for key, value in manifest["paths"].items()}
    for key in (
        "result_json",
        "current_json",
        "estimator_call_ledger_json",
        "execution_manifest_json",
        "normalized_run_manifest_json",
        "resume_input_json",
        "resume_input_ledger_json",
        "resume_input_signed_prefix_json",
        "source_ledger_record_json",
    ):
        paths[key].parent.mkdir(parents=True, exist_ok=True)

    source_lock = manifest["source_lock"]
    source_checkpoint = Path(source_lock["transferred_checkpoint"])
    if sha256(source_checkpoint) != source_lock["transferred_checkpoint_sha256"]:
        raise ValueError("runtime compressed resume-checkpoint hash mismatch")
    temporary_checkpoint = paths["resume_input_json"].with_suffix(".json.tmp")
    with gzip.open(source_checkpoint, "rb") as zipped, temporary_checkpoint.open(
        "wb"
    ) as raw:
        digest = hashlib.sha256()
        size = 0
        for chunk in iter(lambda: zipped.read(1024 * 1024), b""):
            raw.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if digest.hexdigest() != source_lock[
        "transferred_checkpoint_uncompressed_sha256"
    ] or size != int(source_lock["transferred_checkpoint_uncompressed_size_bytes"]):
        temporary_checkpoint.unlink(missing_ok=True)
        raise ValueError("runtime resume-checkpoint decompression hash/size mismatch")
    temporary_checkpoint.replace(paths["resume_input_json"])
    source_ledger = Path(source_lock["transferred_source_ledger"])
    if sha256(source_ledger) != source_lock["transferred_source_ledger_sha256"]:
        raise ValueError("runtime compressed source-ledger hash mismatch")
    temporary_ledger = paths["resume_input_ledger_json"].with_suffix(".json.tmp")
    with gzip.open(source_ledger, "rb") as zipped, temporary_ledger.open("wb") as raw:
        digest = hashlib.sha256()
        size = 0
        for chunk in iter(lambda: zipped.read(1024 * 1024), b""):
            raw.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if digest.hexdigest() != source_lock[
        "transferred_source_ledger_uncompressed_sha256"
    ] or size != int(source_lock["transferred_source_ledger_uncompressed_size_bytes"]):
        temporary_ledger.unlink(missing_ok=True)
        raise ValueError("runtime source-ledger decompression hash/size mismatch")
    temporary_ledger.replace(paths["resume_input_ledger_json"])
    source_signed_prefix = Path(source_lock["transferred_signed_prefix_sidecar"])
    shutil.copy2(source_signed_prefix, paths["resume_input_signed_prefix_json"])
    if sha256(paths["resume_input_signed_prefix_json"]) != source_lock[
        "transferred_signed_prefix_sidecar_sha256"
    ]:
        raise ValueError("runtime signed-prefix sidecar copy hash mismatch")
    if paths["resume_input_signed_prefix_json"].name != SIGNED_PREFIX_CANONICAL_NAME:
        raise ValueError("runtime signed-prefix sidecar canonical name mismatch")
    shutil.copy2(Path(source_lock["source_record"]), paths["source_ledger_record_json"])

    environment = {
        str(key): str(value) for key, value in manifest["environment"].items()
    }
    for key, value in environment.items():
        if key.endswith("_CACHE_DIR") or key == "MPLCONFIGDIR":
            cache_path = Path(value)
            if cache_path.exists():
                raise ValueError(f"job-local cache must begin absent: {cache_path}")
            cache_path.mkdir(parents=True, exist_ok=False)

    normalized = {
        "schema": "paper_i_hh_sr_r30_to_r50_runtime_manifest_v1",
        "job_manifest": manifest_path.as_posix(),
        "job_manifest_sha256": sha256(manifest_path),
        "command_argv": execution_argv,
        "environment": environment,
        "route_identity": manifest["route_identity"],
        "physics": manifest["physics"],
        "scientific_contract": manifest["scientific_contract"],
        "settings_difference": manifest["settings_difference"],
        "source_lock": manifest["source_lock"],
        "started_utc": utc_now(),
    }
    write_json(paths["normalized_run_manifest_json"], normalized)
    execution = {
        **normalized,
        "schema": "paper_i_hh_sr_r30_to_r50_execution_v1",
        "status": "running",
        "exit_code": None,
    }
    write_json(paths["execution_manifest_json"], execution)
    env = os.environ.copy()
    env.update(environment)
    try:
        completed = subprocess.run(execution_argv, env=env, check=False)
        execution["exit_code"] = int(completed.returncode)
        execution["status"] = "completed" if completed.returncode == 0 else "failed"
        return int(completed.returncode)
    finally:
        execution["finished_utc"] = utc_now()
        execution["artifacts"] = {
            key: {
                "path": path.as_posix(),
                "exists": path.is_file(),
                "sha256": sha256(path) if path.is_file() else None,
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
            for key, path in {
                "result_json": paths["result_json"],
                "current_json": paths["current_json"],
                "estimator_call_ledger_json": paths[
                    "estimator_call_ledger_json"
                ],
                "resume_compile_smoke_json": paths["resume_input_json"].with_name(
                    "round30_current_resume_compile_smoke.json"
                ),
            }.items()
        }
        write_json(paths["execution_manifest_json"], execution)


if __name__ == "__main__":
    raise SystemExit(main())
