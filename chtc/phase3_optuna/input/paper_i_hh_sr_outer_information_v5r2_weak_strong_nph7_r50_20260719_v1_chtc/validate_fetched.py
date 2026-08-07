#!/usr/bin/env python3
"""Validate a fetched control or reuse transfer archive."""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from pair_contract import (
    BUNDLE_ID,
    CONTROL_MODE,
    MODES,
    PAIR_ID,
    REUSE_MODE,
    SOURCE_ARCHIVE_SHA256,
    bundle_dir,
    load_json,
    sha256,
    validate_job,
)
from run_job import validate_completed_outputs


def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsafe transfer member: {member.name}")
        handle.extractall(destination, filter="data")


def validate_output_root(output: Path, mode: str) -> dict[str, Any]:
    manifest_path = bundle_dir() / f"jobs/{mode}.json"
    manifest = load_json(manifest_path)
    validate_job(manifest, expected_mode=mode)
    execution = load_json(output / "execution.json")
    normalized = load_json(output / "normalized_run_manifest.json")
    runtime_validation = load_json(output / "validation.json")
    if execution.get("status") != "completed" or int(
        execution.get("exit_code", -1)
    ) != 0:
        raise ValueError("fetched execution is not terminal-success")
    if (
        execution.get("pair_id") != PAIR_ID
        or execution.get("mode") != mode
        or execution.get("job_manifest_sha256") != sha256(manifest_path)
        or execution.get("image_sha256")
        != manifest["source_lock"]["image_sha256"]
    ):
        raise ValueError("fetched execution provenance drift")
    if (
        normalized.get("pair_id") != PAIR_ID
        or normalized.get("mode") != mode
        or normalized.get("source_lock", {}).get("archive_sha256")
        != SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("fetched normalized manifest drift")
    if mode == REUSE_MODE:
        gate = normalized.get("control_gate_provenance")
        if not isinstance(gate, dict):
            raise ValueError("reuse normalized manifest has no control-gate provenance")
        payload = gate.get("payload", {})
        if (
            payload.get("status") != "pass"
            or payload.get("pair_id") != PAIR_ID
            or gate.get("sha256") is None
        ):
            raise ValueError("reuse consumed an invalid current-runtime control gate")
    artifacts = execution.get("artifacts", {})
    relative = {
        "result_json": "json/result.json",
        "current_json": "json/current.json",
        "ledger_json": "json/estimator_call_ledger.json",
        "normalized_runtime_manifest_json": "normalized_run_manifest.json",
        "validation_json": "validation.json",
        "qiskit_cost_sidecar_json": "qiskit_cost_sidecar.json",
        "repaired_terminal_checkpoint_json": (
            "terminal_checkpoint.execution_order_repaired.json"
        ),
    }
    for key, value in relative.items():
        path = output / value
        record = artifacts.get(key, {})
        if not path.is_file() or record.get("sha256") != sha256(path):
            raise ValueError(f"fetched artifact hash mismatch: {key}")
    validated = validate_completed_outputs(
        manifest,
        output_root_override=output,
        compile_qiskit=False,
    )
    for key in (
        "status",
        "pair_id",
        "mode",
        "result_sha256",
        "current_sha256",
        "ledger_sha256",
        "qiskit_sidecar_sha256",
        "terminal_checkpoint_sha256",
        "S_alg",
        "abs_delta_e",
    ):
        if runtime_validation.get(key) != validated.get(key):
            raise ValueError(f"runtime/fetched validation mismatch: {key}")
    if mode == CONTROL_MODE:
        runtime_gate = load_json(output / "anchor_gate.json")
        if (
            runtime_gate.get("status") != "pass"
            or runtime_gate.get("pair_id") != PAIR_ID
            or runtime_gate.get("control_job_manifest_sha256")
            != sha256(manifest_path)
            or runtime_gate.get("result_sha256") != validated["result_sha256"]
        ):
            raise ValueError("control runtime gate drift")
    return {
        "schema": "paper_i_sr_outer_information_fetched_node_validation_v1",
        "status": "pass",
        "pair_id": PAIR_ID,
        "mode": mode,
        "output_root": str(output),
        "transfer_validation": validated,
    }


def validate_archive(archive: Path, mode: str) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    with tempfile.TemporaryDirectory(prefix=f"{BUNDLE_ID}-{mode}-") as tmp:
        root = Path(tmp)
        safe_extract(archive, root)
        output = root / "raw_outputs" / BUNDLE_ID / mode
        if not output.is_dir():
            raise ValueError(f"fetched archive lacks expected output root: {output}")
        return validate_output_root(output, mode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    receipt = validate_archive(args.archive, args.mode)
    if args.output_json is not None:
        from pair_contract import dump_json

        dump_json(args.output_json, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
