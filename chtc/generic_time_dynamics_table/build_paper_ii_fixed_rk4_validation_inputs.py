#!/usr/bin/env python3
"""Build Paper-II fixed-McLachlan RK4 validation inputs.

This keeps the existing Paper-II seed-track benchmark cases intact, selects the
SNAKE track only, and adds an explicit fixed-McLachlan integrator-policy override
for a narrow validation rerun.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
INPUT = Path("chtc/generic_time_dynamics_table/input")
SOURCE_CASE_MANIFEST = INPUT / "paper_ii_seed_tracks_cases_v2.json"
CASE_MANIFEST = INPUT / "paper_ii_fixed_mclachlan_rk4_cases_v1.json"
RECORDS_TSV = INPUT / "paper_ii_fixed_mclachlan_rk4_records_v1.tsv"
RECORD_IDS = INPUT / "paper_ii_fixed_mclachlan_rk4_record_ids_v1.txt"
SMOKE_IDS = INPUT / "paper_ii_fixed_mclachlan_rk4_smoke_record_ids_v1.txt"
CLASS_SETTINGS = "chtc/generic_time_dynamics_table/input/class_settings/paper_ii_class_settings_lock_v1.json"


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _read_json(path: str | Path) -> Any:
    return json.loads(_repo_path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> None:
    out = _repo_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_ids(path: str | Path, ids: list[str]) -> None:
    out = _repo_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def _record_id(case_id: str) -> str:
    return f"paper_ii_fixed_rk4_v1_{case_id}_dyn_fixed_mclachlan"


def _submit_text(*, ids: str, batch: str, runtime: int) -> str:
    return f"""universe = vanilla
executable = chtc/generic_time_dynamics_table/run_task_apptainer.sh
arguments = $(record_id) {RECORDS_TSV}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = pipelines, src, test_support, MATH/Math.md, run_guide.md, AGENTS.md, chtc/generic_time_dynamics_table, chtc/time_dynamics_optuna/image.sif
transfer_output_files = raw_outputs, logs
log = logs/{batch}.$(Cluster).$(Process).log
output = logs/{batch}.$(Cluster).$(Process).out
error = logs/{batch}.$(Cluster).$(Process).err
requirements = TARGET.HasSIF
request_cpus = 1
request_memory = 12GB
request_disk = 40GB
+MaxRuntime = {int(runtime)}
+JobBatchName = \"holstein-{batch}\"
environment = \"GENERIC_TD_TABLE_RECORDS_PATH={RECORDS_TSV} GENERIC_TD_CLASS_SETTINGS_MANIFEST={CLASS_SETTINGS} GENERIC_TD_REQUIRE_LOCKED_CLASS_SETTINGS=1 GENERIC_TD_OUTPUT_ROOT=raw_outputs/generic_time_dynamics_fixed_rk4_v1\"
queue record_id from {ids}
"""


def build_inputs() -> None:
    source = _read_json(SOURCE_CASE_MANIFEST)
    source_cases = source.get("cases", []) if isinstance(source, dict) else []
    cases: list[dict[str, Any]] = []
    records: list[dict[str, str]] = []
    smoke_ids: list[str] = []

    for case in source_cases:
        if not isinstance(case, dict):
            continue
        metadata = dict(case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {})
        seed_lock = metadata.get("seed_lock", {}) if isinstance(metadata.get("seed_lock"), dict) else {}
        if str(seed_lock.get("seed_track", "")).strip().lower() != "snake":
            continue
        fixed_case = dict(case)
        metadata["fixed_mclachlan_integrator_policy"] = "rk4"
        metadata["benchmark_fixed_mclachlan_integrator_policy"] = "rk4"
        metadata["fixed_mclachlan_validation"] = "paper_ii_fixed_mclachlan_rk4_v1"
        metadata["validation_reuses_case_manifest"] = str(SOURCE_CASE_MANIFEST)
        fixed_case["metadata"] = metadata
        cases.append(fixed_case)

        rid = _record_id(str(fixed_case["case_id"]))
        records.append(
            {
                "record_id": rid,
                "kind": "benchmark",
                "family": str(fixed_case["family"]),
                "tuning_class": str(fixed_case.get("tuning_class", "")),
                "case_id": str(fixed_case["case_id"]),
                "algorithm_id": "dyn_fixed_mclachlan",
                "variants": "fixed_rk4_validation",
                "case_manifest": str(CASE_MANIFEST),
            }
        )
        if str(fixed_case["case_id"]) in {
            "table1_hubbard_snake_A0p2_t8_dt321_seedtracks_v2",
            "table1_hh_snake_A0p2_t8_dt321_seedtracks_v2",
        }:
            smoke_ids.append(rid)

    if not records:
        raise SystemExit("No SNAKE fixed-McLachlan RK4 validation records were generated")

    manifest = {
        "manifest_id": "paper_ii_fixed_mclachlan_rk4_cases_v1",
        "schema": "paper_ii_fixed_mclachlan_rk4_validation_cases_v1",
        "source_case_manifest": str(SOURCE_CASE_MANIFEST),
        "case_count": len(cases),
        "record_count": len(records),
        "seed_track": "snake",
        "algorithm_id": "dyn_fixed_mclachlan",
        "integrator_policy": "rk4",
        "same_seed_contract": "inherits paper_ii_seed_tracks_cases_v2 SNAKE seed locks",
        "qpu_faithful_controller_data_contract": "measurement_compatible_prepared_state_observables_only",
        "diagnostic_exact_reference_mode": "benchmark_exact_reporting_only",
        "cases": cases,
    }
    _write_json(CASE_MANIFEST, manifest)

    records_path = _repo_path(RECORDS_TSV)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["record_id", "kind", "family", "tuning_class", "case_id", "algorithm_id", "variants", "case_manifest"]
    with records_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)

    all_ids = [row["record_id"] for row in records]
    _write_ids(RECORD_IDS, all_ids)
    _write_ids(SMOKE_IDS, smoke_ids)

    (_repo_path("chtc/generic_time_dynamics_table/submit_paper_ii_fixed_mclachlan_rk4_validation_v1.sub")).write_text(
        _submit_text(
            ids=str(RECORD_IDS),
            batch="paper-ii-fixed-mclachlan-rk4-v1",
            runtime=28800,
        ),
        encoding="utf-8",
    )
    (_repo_path("chtc/generic_time_dynamics_table/submit_paper_ii_fixed_mclachlan_rk4_validation_smoke_v1.sub")).write_text(
        _submit_text(
            ids=str(SMOKE_IDS),
            batch="paper-ii-fixed-mclachlan-rk4-smoke-v1",
            runtime=3600,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"case_count": len(cases), "record_count": len(records), "smoke_count": len(smoke_ids)}, indent=2))


if __name__ == "__main__":
    build_inputs()
