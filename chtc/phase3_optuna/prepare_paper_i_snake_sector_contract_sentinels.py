#!/usr/bin/env python3
"""Prepare six source-locked SNAKE sector-contract sentinels; never submit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chtc.phase3_optuna import generate_paper_i_scaling_matrix_records as scaling  # noqa: E402
from chtc.phase3_optuna import run_paper_i_scaling_matrix_cell as cell_runner  # noqa: E402


BATCH_ID = "paper_i_snake_sector_contract_sentinels_20260713_v1"
SOURCE_BATCH_ID = "paper_i_scaling_matrix_snake_overlay_repair_20260711_v1"
SENTINEL_SCOPE = "snake_sector_correctness_parent_only_sentinel_v1"
DEFAULT_OUTPUT_DIR = ROOT / "chtc" / "phase3_optuna" / "input" / BATCH_ID
DEFAULT_SUBMIT_PATH = ROOT / "chtc" / "phase3_optuna" / f"submit_{BATCH_ID}.sub"
SOURCE_INPUT_DIR = ROOT / "chtc" / "phase3_optuna" / "input" / SOURCE_BATCH_ID
SOURCE_RECORDS = SOURCE_INPUT_DIR / "paper_i_scaling_matrix_records.tsv"
SOURCE_ARCHIVE_DIR = (
    ROOT
    / "output"
    / "chtc_retrievals"
    / "paper_i_append_snake_completed_20260711"
    / "raw_outputs"
    / SOURCE_BATCH_ID
)
PARENT_POLICY = (
    ROOT
    / "output"
    / "diagnostics"
    / "paper_i_snake_sector_repair_20260713"
    / "parent_only_policy.json"
)
PARENT_POLICY_SHA256 = "a07faa738ddfacda2f61c0a929cccc1915873458c82512e58c76c74e3b1f5059"

# Ordered to put the two failing HH rows first, then one HH control, followed
# by the two failing Hubbard rows and one Hubbard control.
SENTINELS: tuple[dict[str, Any], ...] = (
    {
        "family": "hh",
        "case_id": "hh_L4_nph1_scaling_strong_weak",
        "role": "failing_hh_sector_escape",
        "horizon": 10,
        "exact_energy": 0.8794946903883498,
        "cpus": 4,
        "memory_mb": 24576,
        "disk_mb": 40960,
    },
    {
        "family": "hh",
        "case_id": "hh_L4_nph1_scaling_intermediate_weak",
        "role": "hh_l4_non_u8_contrast",
        "horizon": 10,
        "exact_energy": -1.433726322469954,
        "cpus": 4,
        "memory_mb": 24576,
        "disk_mb": 40960,
    },
    {
        "family": "hh",
        "case_id": "hh_L3_nph2_scaling_weak_weak",
        "role": "healthy_hh_control",
        "horizon": 10,
        "exact_energy": -1.2240595038534747,
        "cpus": 4,
        "memory_mb": 24576,
        "disk_mb": 40960,
    },
    {
        "family": "hubbard",
        "case_id": "hubbard_L3_scaling_strong",
        "role": "failing_hubbard_no_accepted_prune",
        "horizon": 8,
        "exact_energy": -0.7077115708616566,
        "cpus": 4,
        "memory_mb": 16384,
        "disk_mb": 32768,
    },
    {
        "family": "hubbard",
        "case_id": "hubbard_L4_scaling_weak",
        "role": "failing_hubbard_l4_no_accepted_prune",
        "horizon": 8,
        "exact_energy": -4.228631633274617,
        "cpus": 4,
        "memory_mb": 16384,
        "disk_mb": 32768,
    },
    {
        "family": "hubbard",
        "case_id": "hubbard_L2_scaling_weak",
        "role": "healthy_hubbard_control",
        "horizon": 8,
        "exact_energy": -1.8789024427351741,
        "cpus": 4,
        "memory_mb": 16384,
        "disk_mb": 32768,
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def _write_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _copy_locked(source: Path, target: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"source-locked artifact is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_sha = _sha256(source)
    target_sha = _sha256(target)
    if source_sha != target_sha:
        raise RuntimeError(f"copied artifact changed bytes: {source} -> {target}")
    return {
        "source": scaling._repo_path(source),
        "packaged": scaling._repo_path(target),
        "sha256": source_sha,
        "size_bytes": target.stat().st_size,
    }


def _load_source_rows() -> tuple[dict[tuple[str, str], dict[str, str]], str]:
    rows = _read_rows(SOURCE_RECORDS)
    by_key = {
        (row["family"], row["case_id"]): row
        for row in rows
        if row.get("method_key") == "snake"
    }
    expected_keys = {(str(item["family"]), str(item["case_id"])) for item in SENTINELS}
    missing = sorted(expected_keys - set(by_key))
    if missing:
        raise ValueError(f"source SNAKE records are missing sentinels: {missing}")
    for item in SENTINELS:
        key = (str(item["family"]), str(item["case_id"]))
        row = by_key[key]
        expected = {
            "batch_id": SOURCE_BATCH_ID,
            "algorithm_id": "static_family_native_adapt_phase3",
            "method_key": "snake",
            "optimizer": "POWELL",
            "adapt_optimizer_kind": "powell",
            "budget": "200",
            "pool_contract": "full_meta_unfiltered",
            "child_policy": "macro_only",
            "phase2_batching": "off",
            "phase3_batching": "off",
        }
        drift = [
            f"{field}={row.get(field)!r}, expected {value!r}"
            for field, value in expected.items()
            if str(row.get(field) or "") != value
        ]
        if float(row["exact_reference_energy"]) != float(item["exact_energy"]):
            drift.append(
                f"exact_reference_energy={row['exact_reference_energy']!r}, "
                f"expected {item['exact_energy']!r}"
            )
        if drift:
            raise ValueError(f"source record drift for {key}: " + "; ".join(drift))
    return by_key, _sha256(SOURCE_RECORDS)


def _stage_source_evidence(
    output_dir: Path,
    source_rows: Mapping[tuple[str, str], Mapping[str, str]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any], Path]:
    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    packaged_source_records = _copy_locked(
        SOURCE_RECORDS,
        output_dir / "source_evidence" / "paper_i_scaling_matrix_source_records.tsv",
    )
    for item in SENTINELS:
        key = (str(item["family"]), str(item["case_id"]))
        source_row = source_rows[key]
        source_record_id = str(source_row["record_id"])
        source_root = SOURCE_ARCHIVE_DIR / source_record_id
        packaged_root = output_dir / "source_evidence" / str(item["case_id"])
        files: dict[str, Any] = {}
        for name, rel in (
            ("result", "result/generic_static_single.json"),
            ("cell_manifest", "cell_manifest.json"),
            ("effective_command", "effective_command.json"),
            ("effective_env_overlay", "effective_env_overlay.json"),
        ):
            files[name] = _copy_locked(source_root / rel, packaged_root / Path(rel).name)
        cell = json.loads((packaged_root / "cell_manifest.json").read_text(encoding="utf-8"))
        if str(cell.get("record_id") or "") != source_record_id:
            raise ValueError(f"archived cell manifest record mismatch for {source_record_id}")
        evidence[key] = {
            "source_record_id": source_record_id,
            "role": str(item["role"]),
            "files": files,
        }
    path = output_dir / "source_evidence_manifest.json"
    _write_json(
        path,
        {
            "schema": "paper_i_snake_sector_contract_source_evidence_v1",
            "source_batch_id": SOURCE_BATCH_ID,
            "source_records": scaling._repo_path(SOURCE_RECORDS),
            "source_records_packaged": packaged_source_records["packaged"],
            "source_records_sha256": _sha256(SOURCE_RECORDS),
            "records": {f"{family}:{case_id}": row for (family, case_id), row in evidence.items()},
            "status": "pass",
        },
    )
    return evidence, packaged_source_records, path


def _stage_hh_caches(output_dir: Path) -> tuple[dict[str, Any], Path]:
    source_manifest_path = SOURCE_INPUT_DIR / "hh_cache_prewarm_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "pass":
        raise ValueError("source HH cache manifest is not pass")
    by_case = {
        str(row.get("case_id") or ""): row
        for row in source_manifest.get("disk_hit_verification", [])
        if isinstance(row, Mapping)
    }
    pool_dir = output_dir / "hh_pool_cache_v1"
    registry_dir = output_dir / "hh_generator_registry_cache_v1"
    rows: list[dict[str, Any]] = []
    for item in SENTINELS:
        if item["family"] != "hh":
            continue
        case_id = str(item["case_id"])
        source = by_case.get(case_id)
        if not isinstance(source, Mapping):
            raise ValueError(f"source HH cache manifest is missing {case_id}")
        pool_source = ROOT / str(source["pool_cache_path"])
        registry_source = ROOT / str(source["generator_registry_cache_path"])
        pool = _copy_locked(pool_source, pool_dir / pool_source.name)
        registry = _copy_locked(registry_source, registry_dir / registry_source.name)
        rows.append(
            {
                "case_id": case_id,
                "source_disk_hit_verified": bool(source.get("disk_hit_verified")),
                "pool": pool,
                "generator_registry": registry,
            }
        )
    payload = {
        "schema": "paper_i_snake_sector_contract_hh_cache_lock_v1",
        "status": "pass",
        "source_manifest": scaling._repo_path(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "case_count": len(rows),
        "cases": rows,
        "pool_cache_dir": scaling._repo_path(pool_dir),
        "generator_registry_cache_dir": scaling._repo_path(registry_dir),
    }
    path = output_dir / "hh_cache_lock_manifest.json"
    _write_json(path, payload)
    return payload, path


def prepare(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    submit_path: Path = DEFAULT_SUBMIT_PATH,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    submit_path = Path(submit_path).expanduser().resolve()
    scaling._ensure_fresh_targets(output_dir, submit_path, force=bool(force))

    source_rows, source_records_sha = _load_source_rows()
    if _sha256(PARENT_POLICY) != PARENT_POLICY_SHA256:
        raise ValueError("parent-only repair policy hash drifted")

    exact_manifest = output_dir / "exact_energy_manifest.json"
    exact_lock = _copy_locked(SOURCE_INPUT_DIR / "exact_energy_manifest.json", exact_manifest)
    policy_path = output_dir / "paper_i_snake_sector_contract_parent_only_policy.json"
    policy_lock = _copy_locked(PARENT_POLICY, policy_path)
    source_evidence, packaged_source_records, source_evidence_manifest = _stage_source_evidence(
        output_dir,
        source_rows,
    )
    cache_payload, cache_manifest = _stage_hh_caches(output_dir)
    code_bundle = scaling._write_code_bundle(output_dir)
    implementation_lock, implementation_lock_path = scaling._write_implementation_lock(
        output_dir,
        code_bundle,
    )
    implementation_lock_sha = _sha256(implementation_lock_path)

    rows: list[dict[str, str]] = []
    for item in SENTINELS:
        key = (str(item["family"]), str(item["case_id"]))
        source = source_rows[key]
        evidence = source_evidence[key]
        horizon = int(item["horizon"])
        record_id = (
            f"{BATCH_ID}__{item['family']}__{item['case_id']}__snake__"
            f"parent_safe_core_powell200__iter{horizon}"
        )
        record_output_dir = f"raw_outputs/{BATCH_ID}/{record_id}"
        row = dict(source)
        row.update(
            {
                "record_id": record_id,
                "batch_id": BATCH_ID,
                "sentinel_contract_id": BATCH_ID,
                "repair_scope": SENTINEL_SCOPE,
                "run_class": "diagnostic",
                "sentinel_role": str(item["role"]),
                "matrix_label": "paper_i_snake_sector_contract_sentinels",
                "matrix_role": "correctness_first_sector_and_runtime_contract_diagnostic",
                "pool_contract": "full_meta_execution_sector_filtered",
                "static_meta_feature_profile": "safe_core_v1",
                "phase1_pruning": "off",
                "structural_rollback": "off",
                "cost_steering": "off",
                "state_sector_audit_required": "true",
                "generator_execution_sector_audit_required": "true",
                "zero_angle_parent_identity_required": "true",
                "strict_replay_required": "true",
                "optimizer_coordinate_contract": "logical_shared",
                "selector_round_admission_depth_telemetry_required": "true",
                "max_depth": str(horizon),
                "phase3_adapt_max_depth": str(horizon),
                "expected_horizon": str(horizon),
                "horizon_source": "correctness_sentinel_depth_cap",
                "request_cpus": str(item["cpus"]),
                "request_memory_mb": str(item["memory_mb"]),
                "request_disk_mb": str(item["disk_mb"]),
                "resource_tier": (
                    "sector_sentinel_hh_24gb" if item["family"] == "hh" else "sector_sentinel_hubbard_16gb"
                ),
                "adapt_parallel_gradient_workers": "1",
                "adapt_beam_parent_workers": "1",
                "phase3_adapt_parallel_gradient_workers": "1",
                "phase3_adapt_beam_parent_workers": "1",
                "phase3_policy_json": scaling._repo_path(policy_path),
                "phase3_policy_json_sha256": policy_lock["sha256"],
                "exact_energy_manifest": scaling._repo_path(exact_manifest),
                "exact_energy_manifest_sha256": exact_lock["sha256"],
                "hh_pool_cache_dir": (
                    str(cache_payload["pool_cache_dir"]) if item["family"] == "hh" else ""
                ),
                "hh_generator_registry_cache_dir": (
                    str(cache_payload["generator_registry_cache_dir"])
                    if item["family"] == "hh"
                    else ""
                ),
                "hh_pool_cache_manifest": (
                    scaling._repo_path(cache_manifest) if item["family"] == "hh" else ""
                ),
                "repair_source_batch_id": SOURCE_BATCH_ID,
                "repair_source_record_id": str(evidence["source_record_id"]),
                "source_records_sha256": source_records_sha,
                "source_records_archived": scaling._repo_path(SOURCE_RECORDS),
                "source_records_packaged": str(packaged_source_records["packaged"]),
                "source_evidence_manifest": scaling._repo_path(source_evidence_manifest),
                "source_result_json_archived": str(evidence["files"]["result"]["source"]),
                "source_result_json_packaged": str(evidence["files"]["result"]["packaged"]),
                "source_result_json_sha256": str(evidence["files"]["result"]["sha256"]),
                "source_cell_manifest_archived": str(evidence["files"]["cell_manifest"]["source"]),
                "source_cell_manifest_packaged": str(evidence["files"]["cell_manifest"]["packaged"]),
                "source_cell_manifest_sha256": str(evidence["files"]["cell_manifest"]["sha256"]),
                "source_settings_status": "archived_scaling_row_locked_for_correctness_diagnostic",
                "settings_reused": "source_physics_case;same_cutoff_exact_energy;Powell200;full_meta_parent_macros",
                "settings_changed": (
                    "sector_contract_execution_guard;safe_core_v1;logical_shared_powell_chart;"
                    "prune_off;structural_rollback_off;cost_steering_off;diagnostic_horizon;new_output_identity"
                ),
                "implementation_contract_id": "snake_sector_runtime_contract_repair_20260713_v1",
                "code_bundle": str(code_bundle["path"]),
                "code_bundle_sha256": str(code_bundle["sha256"]),
                "implementation_lock": scaling._repo_path(implementation_lock_path),
                "implementation_lock_sha256": implementation_lock_sha,
                "record_output_dir": record_output_dir,
                "result_json_rel": f"{record_output_dir}/result/generic_static_single.json",
                "current_json_rel": f"{record_output_dir}/result/{item['case_id']}/json/current.json",
                "stdout_rel": f"{record_output_dir}/stdout.log",
                "stderr_rel": f"{record_output_dir}/stderr.log",
                "cell_manifest_rel": f"{record_output_dir}/cell_manifest.json",
            }
        )
        cell_runner.validate_record(row)
        rows.append(row)

    records_path = output_dir / "paper_i_snake_sector_contract_sentinel_records.tsv"
    _write_rows(records_path, rows)
    ids_path = output_dir / "paper_i_snake_sector_contract_sentinel_record_ids.txt"
    ids_path.write_text("\n".join(row["record_id"] for row in rows) + "\n", encoding="utf-8")
    queue_path = output_dir / "paper_i_snake_sector_contract_sentinel_queue.tsv"
    queue_path.write_text(
        "".join(
            f"{row['record_id']}\t{row['request_cpus']}\t{row['request_memory_mb']}\t{row['request_disk_mb']}\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    scaling._write_submit(
        submit_path=submit_path,
        batch_id=BATCH_ID,
        records_path=records_path,
        queue_path=queue_path,
        output_dir=output_dir,
        job_batch_name="paper-i-snake-sector-contract-sentinels",
        stream_output=False,
        stream_error=False,
    )

    generated_utc = datetime.now(timezone.utc).isoformat()
    audit = {
        "schema": "paper_i_snake_sector_contract_sentinel_submission_audit_v1",
        "generated_utc": generated_utc,
        "status": "prepared_not_submitted",
        "submission_authority": "prepared_only_not_submitted",
        "classification": "correctness_implementation_repair_diagnostic",
        "source_locked_sensitivity_claim": False,
        "batch_id": BATCH_ID,
        "run_class": "diagnostic",
        "record_count": len(rows),
        "sentinel_case_order": [str(item["case_id"]) for item in SENTINELS],
        "records_path": scaling._repo_path(records_path),
        "record_ids_path": scaling._repo_path(ids_path),
        "queue_path": scaling._repo_path(queue_path),
        "submit_path": scaling._repo_path(submit_path),
        "source_records": scaling._repo_path(SOURCE_RECORDS),
        "source_records_sha256": source_records_sha,
        "source_evidence_manifest": scaling._repo_path(source_evidence_manifest),
        "exact_energy_manifest": exact_lock,
        "parent_only_policy": policy_lock,
        "hh_cache_lock_manifest": scaling._repo_path(cache_manifest),
        "code_bundle": code_bundle,
        "implementation_lock": scaling._repo_path(implementation_lock_path),
        "implementation_lock_sha256": implementation_lock_sha,
        "resource_contract": {
            "hh": {"cpus": 4, "memory_mb": 24576, "disk_mb": 40960},
            "hubbard": {"cpus": 4, "memory_mb": 16384, "disk_mb": 32768},
        },
        "submit_log_streaming": {"stdout": False, "stderr": False},
        "preflight_command": (
            "python3 chtc/phase3_optuna/preflight_paper_i_snake_sector_contract_sentinels.py "
            f"--input-dir {scaling._repo_path(output_dir)} --submit {scaling._repo_path(submit_path)}"
        ),
    }
    audit_path = output_dir / "submission_audit.json"
    _write_json(audit_path, audit)
    manifest = {
        "schema": "paper_i_snake_sector_contract_sentinel_manifest_v1",
        **audit,
        "submission_audit": scaling._repo_path(audit_path),
        "implementation_lock_entry_count": len(implementation_lock["entries"]),
    }
    _write_json(output_dir / "sentinel_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--submit-path", type=Path, default=DEFAULT_SUBMIT_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    payload = prepare(
        output_dir=args.output_dir,
        submit_path=args.submit_path,
        force=bool(args.force),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
