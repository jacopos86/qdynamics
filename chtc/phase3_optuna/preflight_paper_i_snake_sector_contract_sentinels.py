#!/usr/bin/env python3
"""Fail-closed preflight for the six SNAKE sector-contract sentinels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chtc.phase3_optuna import generate_paper_i_scaling_matrix_records as scaling  # noqa: E402
from chtc.phase3_optuna import (  # noqa: E402
    prepare_paper_i_snake_sector_contract_sentinels as preparation,
)
from chtc.phase3_optuna import run_paper_i_scaling_matrix_cell as cell_runner  # noqa: E402
from chtc.phase3_optuna import run_task  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def _submit_stream_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip().lower()] = value.strip()
    return fields


def _implementation_blockers(row: Mapping[str, str]) -> list[str]:
    blockers: list[str] = []
    lock_path = _resolve(str(row.get("implementation_lock") or ""))
    bundle_path = _resolve(str(row.get("code_bundle") or ""))
    try:
        if _sha256(lock_path) != str(row.get("implementation_lock_sha256") or ""):
            blockers.append("implementation_lock_sha256_mismatch")
        if _sha256(bundle_path) != str(row.get("code_bundle_sha256") or ""):
            blockers.append("code_bundle_sha256_mismatch")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("status") != "pass":
            blockers.append("implementation_lock_status_not_pass")
        entries = {
            str(item.get("path") or ""): item
            for item in lock.get("entries", [])
            if isinstance(item, Mapping)
        }
        actual_critical = {
            rel for rel, item in entries.items() if item.get("critical_bundle_member") is True
        }
        if actual_critical != set(scaling.CRITICAL_BUNDLE_MEMBERS):
            blockers.append("implementation_lock_critical_set_mismatch")
        for rel, item in entries.items():
            local_path = ROOT / rel
            if not local_path.is_file():
                blockers.append(f"implementation_lock_local_file_missing:{rel}")
            elif _sha256(local_path) != str(item.get("sha256") or ""):
                blockers.append(f"implementation_lock_local_sha_mismatch:{rel}")
        with tarfile.open(bundle_path, "r:gz") as archive:
            names = set(archive.getnames())
            for rel in sorted(scaling.CRITICAL_BUNDLE_MEMBERS):
                item = entries.get(rel)
                if not isinstance(item, Mapping):
                    blockers.append(f"implementation_lock_entry_missing:{rel}")
                    continue
                if rel not in names:
                    blockers.append(f"code_bundle_member_missing:{rel}")
                    continue
                extracted = archive.extractfile(rel)
                if extracted is None:
                    blockers.append(f"code_bundle_member_unreadable:{rel}")
                    continue
                bundle_sha = hashlib.sha256(extracted.read()).hexdigest()
                if bundle_sha != str(item.get("sha256") or ""):
                    blockers.append(f"code_bundle_member_sha_mismatch:{rel}")
    except Exception as exc:
        blockers.append(f"implementation_lock_invalid:{type(exc).__name__}:{exc}")
    return blockers


def _policy_blockers(row: Mapping[str, str]) -> list[str]:
    blockers: list[str] = []
    path = _resolve(str(row.get("phase3_policy_json") or ""))
    try:
        actual_sha = _sha256(path)
        if actual_sha != preparation.PARENT_POLICY_SHA256:
            blockers.append(f"parent_policy_source_hash_mismatch:{actual_sha}")
        if actual_sha != str(row.get("phase3_policy_json_sha256") or ""):
            blockers.append("parent_policy_row_hash_mismatch")
        payload = json.loads(path.read_text(encoding="utf-8"))
        static = payload.get("static")
        inner = payload.get("inner_optimizer")
        pool = payload.get("pool")
        if not isinstance(static, Mapping) or not isinstance(inner, Mapping) or not isinstance(pool, Mapping):
            return [*blockers, "parent_policy_sections_invalid"]
        expected_static: dict[str, Any] = {
            "static_meta_feature_profile": "safe_core_v1",
            "static_route_id": "route_a",
            "adapt_maxiter": 200,
            "adapt_reopt_policy": "full",
            "adapt_full_refit_every": 1,
            "adapt_final_full_refit": True,
            "adapt_allow_repeats": True,
            "adapt_parallel_gradient_workers": 1,
            "adapt_beam_parent_workers": 1,
            "adapt_beam_live_branches": 1,
            "adapt_beam_children_per_parent": 1,
            "phase0_algebraic_lane_mode": "off",
            "phase0_pilot_enabled": False,
            "phase1_prune_enabled": False,
            "phase2_enable_batching": False,
            "phase3_enable_batching": False,
            "phase3_runtime_split_mode": "off",
            "shared_pauli_pool_mode": "off",
            "lambda_compile": 0.0,
            "lambda_measure": 0.0,
            "lambda_leak": 0.0,
            "lambda_2q": 0.0,
            "lambda_d": 0.0,
            "lambda_1q": 0.0,
            "lambda_theta": 0.0,
            "lambda_shot": 0.0,
        }
        for field, expected in expected_static.items():
            if static.get(field) != expected:
                blockers.append(f"parent_policy_static_mismatch:{field}")
        if pool.get("pool_key") != "full_meta":
            blockers.append("parent_policy_pool_key_mismatch")
        for field in ("inner_optimizer", "final_optimizer_type"):
            if inner.get(field) != "POWELL":
                blockers.append(f"parent_policy_optimizer_mismatch:{field}")
        for field in ("refit_maxiter", "final_maxiter"):
            if inner.get(field) != 200:
                blockers.append(f"parent_policy_budget_mismatch:{field}")
    except Exception as exc:
        blockers.append(f"parent_policy_invalid:{type(exc).__name__}:{exc}")
    return blockers


def _source_evidence_blockers(row: Mapping[str, str]) -> list[str]:
    blockers: list[str] = []
    for label, path_field, sha_field in (
        ("result_packaged", "source_result_json_packaged", "source_result_json_sha256"),
        ("cell_packaged", "source_cell_manifest_packaged", "source_cell_manifest_sha256"),
    ):
        try:
            actual = _sha256(_resolve(str(row.get(path_field) or "")))
            expected = str(row.get(sha_field) or "")
            if actual != expected:
                blockers.append(f"source_evidence_sha_mismatch:{label}")
        except Exception as exc:
            blockers.append(f"source_evidence_invalid:{label}:{type(exc).__name__}:{exc}")
    for label, path_field, sha_field in (
        ("result_archived", "source_result_json_archived", "source_result_json_sha256"),
        ("cell_archived", "source_cell_manifest_archived", "source_cell_manifest_sha256"),
    ):
        path = _resolve(str(row.get(path_field) or ""))
        if path.is_file() and _sha256(path) != str(row.get(sha_field) or ""):
            blockers.append(f"source_evidence_sha_mismatch:{label}")
    try:
        cell = json.loads(
            _resolve(str(row["source_cell_manifest_packaged"])).read_text(encoding="utf-8")
        )
        if str(cell.get("record_id") or "") != str(row.get("repair_source_record_id") or ""):
            blockers.append("source_cell_manifest_record_id_mismatch")
    except Exception as exc:
        blockers.append(f"source_cell_manifest_parse_failed:{type(exc).__name__}:{exc}")
    return blockers


def _cache_blockers(rows: Sequence[Mapping[str, str]]) -> list[str]:
    blockers: list[str] = []
    hh_rows = [row for row in rows if row.get("family") == "hh"]
    if len(hh_rows) != 3:
        return [f"hh_sentinel_count_mismatch:{len(hh_rows)}"]
    paths = {str(row.get("hh_pool_cache_manifest") or "") for row in hh_rows}
    if len(paths) != 1:
        return ["hh_cache_manifest_mixed"]
    try:
        payload = json.loads(_resolve(next(iter(paths))).read_text(encoding="utf-8"))
        if payload.get("schema") != "paper_i_snake_sector_contract_hh_cache_lock_v1":
            blockers.append("hh_cache_manifest_schema_mismatch")
        if payload.get("status") != "pass" or int(payload.get("case_count") or 0) != 3:
            blockers.append("hh_cache_manifest_status_or_count_mismatch")
        expected_cases = {str(row["case_id"]) for row in hh_rows}
        actual_cases = {
            str(item.get("case_id") or "")
            for item in payload.get("cases", [])
            if isinstance(item, Mapping)
        }
        if actual_cases != expected_cases:
            blockers.append("hh_cache_case_set_mismatch")
        for item in payload.get("cases", []):
            if not isinstance(item, Mapping):
                blockers.append("hh_cache_case_row_invalid")
                continue
            if item.get("source_disk_hit_verified") is not True:
                blockers.append(f"hh_cache_source_disk_hit_not_verified:{item.get('case_id')}")
            for field in ("pool", "generator_registry"):
                file_row = item.get(field)
                if not isinstance(file_row, Mapping):
                    blockers.append(f"hh_cache_file_row_missing:{item.get('case_id')}:{field}")
                    continue
                path = _resolve(str(file_row.get("packaged") or ""))
                if _sha256(path) != str(file_row.get("sha256") or ""):
                    blockers.append(f"hh_cache_file_sha_mismatch:{item.get('case_id')}:{field}")
    except Exception as exc:
        blockers.append(f"hh_cache_manifest_invalid:{type(exc).__name__}:{exc}")
    return blockers


def build_preflight(
    *,
    input_dir: Path,
    submit_path: Path,
) -> dict[str, Any]:
    input_dir = Path(input_dir).expanduser().resolve()
    submit_path = Path(submit_path).expanduser().resolve()
    records_path = input_dir / "paper_i_snake_sector_contract_sentinel_records.tsv"
    ids_path = input_dir / "paper_i_snake_sector_contract_sentinel_record_ids.txt"
    queue_path = input_dir / "paper_i_snake_sector_contract_sentinel_queue.tsv"
    blockers: list[str] = []
    try:
        rows = _read_rows(records_path)
    except Exception as exc:
        rows = []
        blockers.append(f"records_invalid:{type(exc).__name__}:{exc}")

    expected_by_case = {str(item["case_id"]): item for item in preparation.SENTINELS}
    if len(rows) != 6:
        blockers.append(f"record_count_mismatch:{len(rows)}:expected:6")
    if len({str(row.get("record_id") or "") for row in rows}) != len(rows):
        blockers.append("record_ids_not_unique")
    actual_cases = {str(row.get("case_id") or "") for row in rows}
    if actual_cases != set(expected_by_case):
        blockers.append("sentinel_case_set_mismatch")
    for row in rows:
        case_id = str(row.get("case_id") or "")
        expected = expected_by_case.get(case_id)
        try:
            cell_runner.validate_record(row)
        except Exception as exc:
            blockers.append(f"runner_contract_failed:{case_id}:{type(exc).__name__}:{exc}")
        if expected is None:
            continue
        checks = {
            "family": str(expected["family"]),
            "sentinel_role": str(expected["role"]),
            "expected_horizon": str(expected["horizon"]),
            "request_cpus": str(expected["cpus"]),
            "request_memory_mb": str(expected["memory_mb"]),
            "request_disk_mb": str(expected["disk_mb"]),
        }
        for field, value in checks.items():
            if str(row.get(field) or "") != value:
                blockers.append(f"row_field_mismatch:{case_id}:{field}")
        if float(row.get("exact_reference_energy") or "nan") != float(expected["exact_energy"]):
            blockers.append(f"exact_energy_mismatch:{case_id}")
        try:
            source_records_packaged = _resolve(str(row.get("source_records_packaged") or ""))
            if str(row.get("source_records_sha256") or "") != _sha256(source_records_packaged):
                blockers.append(f"source_records_sha_mismatch:{case_id}")
        except Exception as exc:
            blockers.append(
                f"source_records_packaged_invalid:{case_id}:{type(exc).__name__}:{exc}"
            )
        blockers.extend(f"{case_id}:{reason}" for reason in _source_evidence_blockers(row))

    if rows:
        blockers.extend(_policy_blockers(rows[0]))
        blockers.extend(_implementation_blockers(rows[0]))
        exact_paths = {str(row.get("exact_energy_manifest") or "") for row in rows}
        exact_hashes = {str(row.get("exact_energy_manifest_sha256") or "") for row in rows}
        if len(exact_paths) != 1 or len(exact_hashes) != 1:
            blockers.append("exact_energy_manifest_mixed")
        else:
            try:
                exact_path = _resolve(next(iter(exact_paths)))
                if _sha256(exact_path) != next(iter(exact_hashes)):
                    blockers.append("exact_energy_manifest_sha_mismatch")
                exact = json.loads(exact_path.read_text(encoding="utf-8"))
                for case_id, expected in expected_by_case.items():
                    value = exact["records"][case_id]["exact_energy"]
                    if float(value) != float(expected["exact_energy"]):
                        blockers.append(f"exact_energy_manifest_value_mismatch:{case_id}")
            except Exception as exc:
                blockers.append(f"exact_energy_manifest_invalid:{type(exc).__name__}:{exc}")
        blockers.extend(_cache_blockers(rows))

    try:
        ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if ids != [str(row["record_id"]) for row in rows]:
            blockers.append("record_id_file_order_or_content_mismatch")
    except Exception as exc:
        blockers.append(f"record_id_file_invalid:{type(exc).__name__}:{exc}")
    try:
        queue = [line.split("\t") for line in queue_path.read_text(encoding="utf-8").splitlines() if line]
        expected_queue = [
            [row["record_id"], row["request_cpus"], row["request_memory_mb"], row["request_disk_mb"]]
            for row in rows
        ]
        if queue != expected_queue:
            blockers.append("queue_content_or_resource_mismatch")
    except Exception as exc:
        blockers.append(f"queue_invalid:{type(exc).__name__}:{exc}")

    try:
        contract = run_task.parse_submit_contract(submit_path)
        records_rel = scaling._repo_path(records_path)
        queue_rel = scaling._repo_path(queue_path)
        input_rel = scaling._repo_path(input_dir)
        expected = {
            "executable": "chtc/phase3_optuna/run_paper_i_scaling_matrix_task_apptainer.sh",
            "arguments": (
                f"$(record_id) {records_rel} raw_outputs/{preparation.BATCH_ID}/$(record_id)"
            ),
            "request_cpus": "$(cpus)",
            "request_memory": "$(memory_mb)MB",
            "request_disk": "$(disk_mb)MB",
            "queue_record_id_file": queue_rel,
            "job_batch_name": "paper-i-snake-sector-contract-sentinels",
        }
        for field, value in expected.items():
            if str(contract.get(field) or "") != value:
                blockers.append(f"submit_contract_mismatch:{field}")
        expected_transfers = {
            "chtc/phase3_optuna/image.sif",
            "chtc/phase3_optuna/run_paper_i_scaling_matrix_task_apptainer.sh",
            input_rel,
        }
        if set(contract.get("transfer_input_files") or []) != expected_transfers:
            blockers.append("submit_transfer_input_set_mismatch")
        if contract.get("transfer_output_files") != [
            f"raw_outputs/{preparation.BATCH_ID}/$(record_id)"
        ]:
            blockers.append("submit_transfer_output_mismatch")
        streams = _submit_stream_fields(submit_path)
        if streams.get("stream_output", "").lower() != "false":
            blockers.append("submit_stream_output_not_false")
        if streams.get("stream_error", "").lower() != "false":
            blockers.append("submit_stream_error_not_false")
    except Exception as exc:
        blockers.append(f"submit_contract_invalid:{type(exc).__name__}:{exc}")

    blockers = sorted(set(blockers))
    return {
        "schema": "paper_i_snake_sector_contract_sentinel_preflight_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": preparation.BATCH_ID,
        "status": "pass" if not blockers else "blocked",
        "ok": not blockers,
        "record_count": len(rows),
        "records_path": scaling._repo_path(records_path),
        "submit_path": scaling._repo_path(submit_path),
        "blocking_reasons": blockers,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=preparation.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--submit", type=Path, default=preparation.DEFAULT_SUBMIT_PATH)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    payload = build_preflight(input_dir=args.input_dir, submit_path=args.submit)
    output = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json is not None
        else Path(args.input_dir).expanduser().resolve() / "preflight.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
