#!/usr/bin/env python3
"""Execute the single locked SR-SNAKE strong-strong command on CHTC."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


EXPECTED_SCHEMA = "paper_i_hh_sr_snake_strong_strong_r45_chtc_job_v1"
ALLOWED_DIFFS = {
    "--output-json",
    "--adapt-current-json",
    "--adapt-estimator-call-ledger-json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def options(argv: Sequence[str]) -> dict[str, Any]:
    if list(argv[:3]) != ["python3", "-m", "pipelines.static_adapt.adapt_pipeline"]:
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


def validate(manifest: dict[str, Any]) -> list[str]:
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("unexpected job manifest schema")
    source_argv = [str(token) for token in manifest["command"]["source_argv"]]
    execution_argv = [str(token) for token in manifest["command"]["execution_argv"]]
    source = options(source_argv)
    execution = options(execution_argv)
    changed = {key for key in set(source) | set(execution) if source.get(key) != execution.get(key)}
    if changed != ALLOWED_DIFFS:
        raise ValueError(f"execution argv drift: {sorted(changed)}")
    if manifest["command"].get("unexpected_differences"):
        raise ValueError("job manifest records unexpected settings differences")
    source_lock = manifest["source_lock"]
    source_command_path = Path(source_lock["source_command"])
    source_manifest_path = Path(source_lock["source_normalized_manifest"])
    if sha256(source_command_path) != source_lock["source_command_sha256"]:
        raise ValueError("transferred source command hash mismatch")
    if sha256(source_manifest_path) != source_lock["source_normalized_manifest_sha256"]:
        raise ValueError("transferred source normalized manifest hash mismatch")
    source_command = json.loads(source_command_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if [str(token) for token in source_command.get("argv", [])] != source_argv:
        raise ValueError("source command argv does not match job manifest")
    if [str(token) for token in source_manifest.get("argv", [])] != source_argv:
        raise ValueError("source normalized-manifest argv does not match job manifest")
    environment_audit = manifest.get("environment_audit", {})
    if environment_audit.get("execution_environment") != manifest.get("environment"):
        raise ValueError("execution environment audit drift")
    if environment_audit.get("scientific_settings_changed") is not False:
        raise ValueError("cache environment is not classified as operational-only")
    if environment_audit.get("cache_state_contract") != "empty_job_local_no_cross_route_reuse":
        raise ValueError("cache isolation contract drift")
    transfer_contract = manifest.get("transfer_contract", {})
    if transfer_contract.get("mode") != "compressed_output_bundle_v1":
        raise ValueError("compressed transfer contract missing")
    if transfer_contract.get("source_directory") != manifest["paths"].get("output_root"):
        raise ValueError("compressed transfer source-directory drift")
    if transfer_contract.get("when_to_transfer_output") != "ON_EXIT_OR_EVICT":
        raise ValueError("checkpoint transfer timing drift")
    required = {
        "--problem": "hh",
        "--L": "2",
        "--u": "8.0",
        "--n-ph-max": "4",
        "--adapt-max-depth": "45",
        "--adapt-inner-optimizer": "POWELL",
        "--phase3-runtime-split-max-subset-size": "1",
        "--phase3-runtime-split-child-set-symmetry-policy": "hard_guard",
        "--phase3-runtime-split-child-padding-policy": "exact_projected_grouped_v1",
        "--historical-singleton-coordinate-solve-policy": "supported_metric_whitened_eigh_v1",
        "--historical-singleton-trust-region-update-policy": "displacement_calibrated_unbounded_v2",
    }
    mismatches = {
        key: {"expected": value, "actual": execution.get(key)}
        for key, value in required.items()
        if execution.get(key) != value
    }
    for flag in ("--phase0-no-pilot", "--phase2-no-batching", "--phase3-no-batching"):
        if execution.get(flag) is not True:
            mismatches[flag] = {"expected": True, "actual": execution.get(flag)}
    if mismatches:
        raise ValueError(f"scientific contract mismatch: {mismatches}")
    archive = Path(source_lock["source_archive"])
    if sha256(archive) != source_lock["source_archive_sha256"]:
        raise ValueError("source archive hash mismatch in execute sandbox")
    return execution_argv


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("usage: run_job.py JOB_MANIFEST.json")
    manifest_path = Path(args[0])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("job manifest must be a JSON object")
    execution_argv = validate(manifest)
    paths = {key: Path(value) for key, value in manifest["paths"].items()}
    output_root = paths["output_root"]
    expected_root = Path("raw_outputs") / manifest["bundle_id"] / "strong_strong"
    if output_root != expected_root:
        raise ValueError("output root drift")
    for path in paths.values():
        if path != output_root and output_root not in path.parents:
            raise ValueError(f"output path escaped isolated root: {path}")
    for path in (
        paths["result_json"],
        paths["current_json"],
        paths["estimator_call_ledger_json"],
        paths["execution_manifest_json"],
        paths["normalized_run_manifest_json"],
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    environment = {str(key): str(value) for key, value in manifest["environment"].items()}
    for key, value in environment.items():
        if key.endswith("_CACHE_DIR"):
            cache_path = Path(value)
            if cache_path.exists():
                raise ValueError(f"job-local cache must begin absent: {cache_path}")
            cache_path.mkdir(parents=True, exist_ok=False)
    normalized = {
        "schema": "paper_i_hh_sr_snake_strong_strong_r45_chtc_runtime_manifest_v1",
        "job_manifest": manifest_path.as_posix(),
        "job_manifest_sha256": sha256(manifest_path),
        "command_argv": execution_argv,
        "environment": environment,
        "scientific_contract": manifest["scientific_contract"],
        "source_lock": manifest["source_lock"],
        "started_utc": utc_now(),
    }
    write_json(paths["normalized_run_manifest_json"], normalized)
    execution_record = {
        **normalized,
        "schema": "paper_i_hh_sr_snake_strong_strong_r45_chtc_execution_v1",
        "status": "running",
        "exit_code": None,
    }
    write_json(paths["execution_manifest_json"], execution_record)
    env = os.environ.copy()
    env.update(environment)
    try:
        completed = subprocess.run(execution_argv, env=env, check=False)
        execution_record["exit_code"] = int(completed.returncode)
        execution_record["status"] = "completed" if completed.returncode == 0 else "failed"
        return int(completed.returncode)
    finally:
        execution_record["finished_utc"] = utc_now()
        execution_record["artifacts"] = {
            key: {
                "path": path.as_posix(),
                "exists": path.is_file(),
                "sha256": sha256(path) if path.is_file() else None,
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
            for key, path in {
                "result_json": paths["result_json"],
                "current_json": paths["current_json"],
                "estimator_call_ledger_json": paths["estimator_call_ledger_json"],
            }.items()
        }
        write_json(paths["execution_manifest_json"], execution_record)


if __name__ == "__main__":
    raise SystemExit(main())
