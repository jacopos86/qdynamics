#!/usr/bin/env python3
"""Prepare the 34-row SNAKE-only repair for the scaling env-overlay failure.

The repair is source-locked to the submitted 102-row matrix.  It copies every
native SNAKE row, changes only batch/output/provenance paths, rebuilds the exact
code bundle with the corrected runner, and prewarms the same HH caches.  It
does not submit jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chtc.phase3_optuna import generate_paper_i_scaling_matrix_records as generator  # noqa: E402
from chtc.phase3_optuna import preflight_submit  # noqa: E402
from chtc.phase3_optuna import prewarm_paper_i_scaling_hh_pool_cache as prewarm  # noqa: E402


SOURCE_BATCH_ID = "paper_i_scaling_matrix_parent_powell200_20260710_v1"
SOURCE_CLUSTER_ID = 8772847
DEFAULT_BATCH_ID = "paper_i_scaling_matrix_snake_overlay_repair_20260711_v1"
REPAIR_SCOPE = "snake_only_all_34_physical_cases_overlay_plumbing_v1"
SOURCE_HELD_PROCS = (30, 33)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _write_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _prepare_target(output_dir: Path, submit_path: Path, *, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"Repair input directory is not empty: {output_dir}")
    if submit_path.exists() and not force:
        raise FileExistsError(f"Repair submit descriptor already exists: {submit_path}")
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    if force and submit_path.exists():
        submit_path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    submit_path.parent.mkdir(parents=True, exist_ok=True)


def _copy_locked_file(source_dir: Path, output_dir: Path, name: str) -> Path:
    source = source_dir / name
    if not source.is_file():
        raise FileNotFoundError(f"Source-locked repair input is missing: {source}")
    target = output_dir / name
    shutil.copy2(source, target)
    if generator._sha256(source) != generator._sha256(target):
        raise RuntimeError(f"Copied source-locked artifact changed bytes: {name}")
    return target


def prepare(
    *,
    batch_id: str = DEFAULT_BATCH_ID,
    source_batch_id: str = SOURCE_BATCH_ID,
    output_dir: Path | None = None,
    submit_path: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    batch_id = str(batch_id).strip()
    source_batch_id = str(source_batch_id).strip()
    if not batch_id.startswith("paper_i_scaling_matrix_snake_overlay_repair_"):
        raise ValueError("SNAKE overlay repairs require the dedicated repair batch prefix.")
    if batch_id == source_batch_id:
        raise ValueError("Repair batch id must differ from the submitted source batch id.")
    source_dir = ROOT / "chtc" / "phase3_optuna" / "input" / source_batch_id
    output_dir = (
        ROOT / "chtc" / "phase3_optuna" / "input" / batch_id
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    submit_path = (
        ROOT / "chtc" / "phase3_optuna" / f"submit_{batch_id}.sub"
        if submit_path is None
        else Path(submit_path).expanduser().resolve()
    )
    source_records = source_dir / "paper_i_scaling_matrix_records.tsv"
    if not source_records.is_file():
        raise FileNotFoundError(f"Submitted source records are missing: {source_records}")
    source_rows = _read_rows(source_records)
    source_blockers = preflight_submit._paper_i_scaling_matrix_bundle_blockers(source_rows)
    if source_blockers:
        raise ValueError(f"Submitted source matrix no longer satisfies its 102-row contract: {source_blockers}")
    snake_sources = [(proc, row) for proc, row in enumerate(source_rows) if row.get("method_key") == "snake"]
    if len(snake_sources) != 34 or {proc % 3 for proc, _row in snake_sources} != {0}:
        raise ValueError("Expected exactly 34 native SNAKE source rows at original procs divisible by three.")

    _prepare_target(output_dir, submit_path, force=force)
    exact_manifest = _copy_locked_file(source_dir, output_dir, "exact_energy_manifest.json")
    snake_policy = _copy_locked_file(source_dir, output_dir, "paper_i_scaling_matrix_snake_policy.json")
    exact_manifest_sha = generator._sha256(exact_manifest)
    snake_policy_sha = generator._sha256(snake_policy)
    code_bundle = generator._write_code_bundle(output_dir)
    implementation_lock, implementation_lock_path = generator._write_implementation_lock(output_dir, code_bundle)
    implementation_lock_sha = generator._sha256(implementation_lock_path)

    hh_pool_cache_dir = output_dir / "hh_pool_cache_v1"
    hh_registry_cache_dir = output_dir / "hh_generator_registry_cache_v1"
    hh_cache_manifest = output_dir / "hh_cache_prewarm_manifest.json"
    hh_pool_cache_dir.mkdir(parents=True, exist_ok=True)
    hh_registry_cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for source_proc, source_row in snake_sources:
        row = dict(source_row)
        source_record_id = str(row["record_id"])
        record_id = source_record_id.replace(source_batch_id, batch_id, 1)
        if record_id == source_record_id:
            raise ValueError(f"Source record id does not begin with source batch identity: {source_record_id}")
        output_root = f"raw_outputs/{batch_id}/{record_id}"
        row.update(
            {
                "record_id": record_id,
                "batch_id": batch_id,
                "repair_scope": REPAIR_SCOPE,
                "repair_reason": "remove_generic_adapt_runtime_split_env_overlay_from_native_phase3_dispatch",
                "repair_source_batch_id": source_batch_id,
                "repair_source_record_id": source_record_id,
                "repair_source_proc": str(source_proc),
                "repair_scientific_settings_changed": "none",
                "exact_energy_manifest": generator._repo_path(exact_manifest),
                "exact_energy_manifest_sha256": exact_manifest_sha,
                "phase3_policy_json": generator._repo_path(snake_policy),
                "phase3_policy_json_sha256": snake_policy_sha,
                "hh_pool_cache_dir": (
                    generator._repo_path(hh_pool_cache_dir) if row.get("family") == "hh" else ""
                ),
                "hh_pool_cache_manifest": (
                    generator._repo_path(hh_cache_manifest) if row.get("family") == "hh" else ""
                ),
                "hh_generator_registry_cache_dir": (
                    generator._repo_path(hh_registry_cache_dir) if row.get("family") == "hh" else ""
                ),
                "code_bundle": str(code_bundle["path"]),
                "code_bundle_sha256": str(code_bundle["sha256"]),
                "implementation_lock": generator._repo_path(implementation_lock_path),
                "implementation_lock_sha256": implementation_lock_sha,
                "implementation_contract_id": "snake_phase3_env_overlay_dispatch_repair_20260711_v1",
                "settings_changed": "plumbing_only_generic_runtime_split_env_omitted_for_snake;new_output_identity",
                "record_output_dir": output_root,
                "result_json_rel": f"{output_root}/result/generic_static_single.json",
                "current_json_rel": f"{output_root}/result/{row['case_id']}/json/current.json",
                "stdout_rel": f"{output_root}/stdout.log",
                "stderr_rel": f"{output_root}/stderr.log",
                "cell_manifest_rel": f"{output_root}/cell_manifest.json",
            }
        )
        rows.append(row)

    records_path = output_dir / "paper_i_scaling_matrix_records.tsv"
    _write_rows(records_path, rows)
    ids_path = output_dir / "paper_i_scaling_matrix_record_ids.txt"
    ids_path.write_text("\n".join(row["record_id"] for row in rows) + "\n", encoding="utf-8")
    queue_path = output_dir / "paper_i_scaling_matrix_record_queue.tsv"
    queue_path.write_text(
        "".join(
            f"{row['record_id']}\t{row['request_cpus']}\t{row['request_memory_mb']}\t{row['request_disk_mb']}\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    generator._write_submit(
        submit_path=submit_path,
        batch_id=batch_id,
        records_path=records_path,
        queue_path=queue_path,
        output_dir=output_dir,
        job_batch_name="paper-i-scaling-snake-overlay-repair",
    )

    generated_utc = datetime.now(timezone.utc).isoformat()
    common = {
        "generated_utc": generated_utc,
        "batch_id": batch_id,
        "repair_scope": REPAIR_SCOPE,
        "repair_source_batch_id": source_batch_id,
        "repair_source_cluster_id": SOURCE_CLUSTER_ID,
        "repair_source_records_sha256": generator._sha256(source_records),
        "repair_source_procs": [proc for proc, _row in snake_sources],
        "repair_source_failed_procs": [
            proc for proc, _row in snake_sources if proc not in SOURCE_HELD_PROCS
        ],
        "repair_source_held_procs": list(SOURCE_HELD_PROCS),
        "repair_source_proc_coverage_status": "all_34_snake_rows_failed_or_held_covered",
        "record_count": 34,
        "physical_case_count": 34,
        "method_count": 1,
        "methods": ["static_family_native_adapt_phase3"],
        "scientific_settings_changed": [],
        "plumbing_changes": ["omit generic comparator runtime-split env variables from native SNAKE dispatch"],
        "records_path": generator._repo_path(records_path),
        "record_ids_path": generator._repo_path(ids_path),
        "record_queue_path": generator._repo_path(queue_path),
        "submit_path": generator._repo_path(submit_path),
        "exact_energy_manifest": generator._repo_path(exact_manifest),
        "exact_energy_manifest_sha256": exact_manifest_sha,
        "snake_policy_path": generator._repo_path(snake_policy),
        "snake_policy_sha256": snake_policy_sha,
        "code_bundle": code_bundle,
        "implementation_lock": generator._repo_path(implementation_lock_path),
        "implementation_lock_sha256": implementation_lock_sha,
        "hh_pool_cache_dir": generator._repo_path(hh_pool_cache_dir),
        "hh_generator_registry_cache_dir": generator._repo_path(hh_registry_cache_dir),
        "hh_pool_cache_manifest": generator._repo_path(hh_cache_manifest),
        "status": "awaiting_hh_dual_cache_prewarm",
    }
    generator._write_json(
        output_dir / "paper_i_scaling_matrix_manifest.json",
        {"schema": "paper_i_scaling_snake_overlay_repair_manifest_v1", **common},
    )
    generator._write_json(
        output_dir / "submission_audit.json",
        {"schema": "paper_i_scaling_snake_overlay_repair_submission_audit_v1", **common},
    )

    prewarm.prewarm(
        cache_dir=hh_pool_cache_dir,
        generator_registry_cache_dir=hh_registry_cache_dir,
        manifest_path=hh_cache_manifest,
        seed_cache_dir=source_dir / "hh_pool_cache_v1",
        seed_generator_registry_cache_dir=source_dir / "hh_generator_registry_cache_v1",
    )
    manifest_path = output_dir / "paper_i_scaling_matrix_manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--source-batch-id", default=SOURCE_BATCH_ID)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--submit-path", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    payload = prepare(
        batch_id=args.batch_id,
        source_batch_id=args.source_batch_id,
        output_dir=args.output_dir,
        submit_path=args.submit_path,
        force=bool(args.force),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
