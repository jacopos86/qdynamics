#!/usr/bin/env python3
"""Recover reporting-only artifacts after the Test-1 validator schema defect."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


BUNDLE_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_"
    "all_six_r50_20260721_v1_chtc"
)
PROC_TO_REGIME = {
    0: "weak_weak",
    1: "intermediate_weak",
    2: "strong_weak_u8",
    3: "weak_strong",
    4: "intermediate_strong",
    5: "strong_strong_u8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def safe_extract(archive: Path, root: Path) -> None:
    root_resolved = root.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (root / member.name).resolve()
            if target != root_resolved and root_resolved not in target.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
        handle.extractall(root, filter="data")


def artifact_record(path: Path) -> dict[str, object]:
    return {
        "exists": path.is_file(),
        "path": str(path),
        "sha256": sha256(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def load_worker(bundle: Path):
    sys.path.insert(0, str(bundle))
    spec = importlib.util.spec_from_file_location(
        "material_window_reporting_worker", bundle / "run_job.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load material-window worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()

    source_archive = args.source_archive.resolve()
    bundle = args.bundle.resolve()
    output_root = args.output_root.resolve()
    archives = [path.resolve() for path in args.archives]
    source_root = output_root / "source"
    if source_root.exists():
        shutil.rmtree(source_root)
    source_root.mkdir(parents=True)
    safe_extract(source_archive, source_root)
    recovery_bundle = (
        source_root / "chtc" / "phase3_optuna" / "input" / bundle.name
    )
    recovery_bundle.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle, recovery_bundle)
    worker = load_worker(bundle)
    original_cwd = Path.cwd()
    summaries: list[dict[str, object]] = []
    try:
        os.chdir(source_root)
        for archive in archives:
            proc = int(archive.name.split(".", 1)[1].split("__", 1)[0])
            regime = PROC_TO_REGIME[proc]
            safe_extract(archive, source_root)
            output = source_root / "raw_outputs" / BUNDLE_ID / regime
            execution_path = output / "execution.json"
            original_execution = load(execution_path)
            compact = output_root / "compact_artifacts" / regime
            compact.mkdir(parents=True, exist_ok=True)
            dump(compact / "execution.original_validator_failure.json", original_execution)

            manifest_path = recovery_bundle / "jobs" / f"{regime}.json"
            manifest = worker.load(manifest_path)
            worker.validate(manifest_path, manifest)
            validation = worker.validate_result_and_compile(manifest)
            worker.dump(output / "validation.json", validation)

            paths = {key: Path(value) for key, value in manifest["paths"].items()}
            artifact_paths = {
                "result_json": paths["result_json"],
                "current_json": paths["current_json"],
                "ledger_json": paths["ledger_json"],
                "normalized_runtime_manifest_json": paths["normalized_runtime_manifest_json"],
                "validation_json": paths["validation_json"],
                "qiskit_cost_sidecar_json": paths["qiskit_cost_sidecar_json"],
                "repaired_terminal_checkpoint_json": paths["repaired_terminal_checkpoint_json"],
                "ground_space_fidelity_json": paths["ground_space_fidelity_json"],
            }
            recovered_execution = dict(original_execution)
            recovered_execution.update({
                "status": "completed",
                "exit_code": 0,
                "reporting_only_recovery": {
                    "schema": "paper_i_hh_sr_material_window_reporting_recovery_v1",
                    "status": "pass",
                    "source_cluster": 9308516,
                    "source_process": proc,
                    "source_execution_status": original_execution.get("status"),
                    "source_validation_error": (
                        "ValueError: material-window scope missing from settings"
                    ),
                    "science_rerun": False,
                    "algorithmic_query_delta": 0,
                    "repair_scope": (
                        "read support-change policy from route semantic invariants"
                    ),
                    "recovered_utc": datetime.now(timezone.utc).isoformat(),
                },
                "artifacts": {
                    key: artifact_record(path) for key, path in artifact_paths.items()
                },
            })
            worker.dump(execution_path, recovered_execution)

            fetched_report = compact / f"{regime}_reporting_recovery_fetched_validation.json"
            subprocess.run(
                [
                    sys.executable,
                    str(bundle / "validate_fetched.py"),
                    str(source_root),
                    "--output-json",
                    str(fetched_report),
                ],
                check=True,
                cwd=source_root,
            )
            report = load(fetched_report)
            if report.get("status") != "pass":
                raise ValueError(f"fetched validation failed for {regime}")

            preserved: dict[str, str] = {}
            for name in (
                "execution.json",
                "normalized_run_manifest.json",
                "validation.json",
                "terminal_checkpoint.execution_order_repaired.json",
                "ground_space_projector_fidelity.json",
                "qiskit_cost_sidecar.json",
            ):
                source = output / name
                destination = compact / name
                shutil.copy2(source, destination)
                preserved[name] = sha256(destination)
            receipt = {
                "schema": "paper_i_hh_sr_material_window_reporting_recovery_receipt_v1",
                "status": "pass",
                "cluster_proc": f"9308516.{proc}",
                "regime": regime,
                "raw_archive": {
                    "path": str(archive),
                    "sha256": sha256(archive),
                    "tar_integrity_checked": True,
                },
                "scientific_payload_hashes": {
                    "result_json_sha256": sha256(paths["result_json"]),
                    "current_json_sha256": sha256(paths["current_json"]),
                    "estimator_ledger_sha256": sha256(paths["ledger_json"]),
                },
                "preserved_recovery_artifacts": preserved,
                "algorithmic_query_delta": 0,
                "science_rerun": False,
            }
            dump(compact / "reporting_only_recovery_receipt.json", receipt)
            summaries.append({
                "cluster_proc": f"9308516.{proc}",
                "regime": regime,
                "archive_sha256": sha256(archive),
                "validation_status": report.get("status"),
                "S_alg": report.get("scientific_evidence_validation", {})
                .get("active_prefix_estimator_ledger_receipts", {})
                .get("S_alg"),
                "material_window_validation": report.get("material_window_validation"),
            })
            shutil.rmtree(output)
    finally:
        os.chdir(original_cwd)
    dump(output_root / "reporting_recovery_summary.json", {
        "schema": "paper_i_hh_sr_material_window_reporting_recovery_summary_v1",
        "status": "pass",
        "rows": summaries,
    })
    shutil.rmtree(source_root)
    print(json.dumps({"status": "pass", "rows": summaries}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
