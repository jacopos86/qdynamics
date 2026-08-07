#!/usr/bin/env python3
"""Build HH-only Paper-II checkpoint-controller recovery ladder inputs.

This diagnostic ladder is intentionally separate from the all-family Paper-II
seedtracks inputs.  Stage 0 points to legacy exact-assisted old-good HH records
for regression comparison.  Stages 1-5 run exact-free prepared-state observable
controller variants on old-good and current seedtracks HH SNAKE seeds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "chtc" / "generic_time_dynamics_table" / "input"
SEEDTRACK_CASES = INPUT / "paper_ii_seed_tracks_cases_v2.json"
CLASS_LOCK = "chtc/generic_time_dynamics_table/input/class_settings/paper_ii_class_settings_lock_v1.json"
SCHEMA = "hh_recovery_ladder_inputs_v1"

OLDGOOD = {
    "A0p2": {
        "amplitude": 0.2,
        "case_id": "hh_recovery_oldgood_A0p2_t8_dt321_v1",
        "artifact_json": "../../time_dynamics_optuna/input/source_seed_artifacts/td_hh_goodseed_A0p2_t8_dt321_prune_recoverability_eulerfix_v1.json",
        "legacy_record_id": "td_hh_goodseed_A0p2_t8_dt321_energybias_v1",
        "legacy_result_json": "raw_outputs/chtc_time_dynamics_optuna/td_hh_goodseed_A0p2_t8_dt321_energybias_v1/run/trials/trial_0020/result.json",
        "mean_abs_energy_total_error": 1.3472362357224444e-4,
        "max_abs_energy_total_error": 1.6620957868407338e-4,
    },
    "A0p6": {
        "amplitude": 0.6,
        "case_id": "hh_recovery_oldgood_A0p6_t8_dt321_v1",
        "artifact_json": "../../time_dynamics_optuna/input/source_seed_artifacts/td_hh_goodseed_A0p6_t8_dt321_prune_recoverability_eulerfix_v1.json",
        "legacy_record_id": "td_hh_goodseed_A0p6_t8_dt321_energybias_v1",
        "legacy_result_json": "raw_outputs/chtc_time_dynamics_optuna/td_hh_goodseed_A0p6_t8_dt321_energybias_v1/run/trials/trial_0035/result.json",
        "mean_abs_energy_total_error": 1.5290267098047527e-3,
        "max_abs_energy_total_error": 2.2456557533092336e-3,
    },
}

STAGES = [
    (1, "hh_recovery_s1_rk4_no_append_no_prune"),
    (2, "hh_recovery_s2_rk4_append_only"),
    (3, "hh_recovery_s3_rk4_append_prune"),
    (4, "hh_recovery_s4_auto_append_prune"),
    (5, "hh_recovery_s5_auto_no_append_no_prune"),
]


def _load_seedtrack_hh_cases() -> dict[float, dict[str, Any]]:
    payload = json.loads(SEEDTRACK_CASES.read_text(encoding="utf-8"))
    cases = {}
    for raw in payload.get("cases", []):
        if raw.get("family") != "hh":
            continue
        drive = ((raw.get("metadata") or {}).get("drive") or {})
        amp = float(drive.get("A"))
        cases[amp] = dict(raw)
    if set(cases) != {0.2, 0.6}:
        raise SystemExit(f"expected HH seedtracks cases for A=0.2 and A=0.6, got {sorted(cases)}")
    return cases


def _oldgood_case(tag: str, info: Mapping[str, Any]) -> dict[str, Any]:
    amp = float(info["amplitude"])
    return {
        "case_id": str(info["case_id"]),
        "family": "hh",
        "table_class": "hubbard_holstein",
        "artifact_json": str(info["artifact_json"]),
        "description": f"HH recovery old-good exact-assisted seed, A={amp}",
        "t_final": 8.0,
        "num_times": 321,
        "loader_mode": "replay_family",
        "generator_family": "full_meta",
        "fallback_family": "full_meta",
        "append_pool_family": "full_meta",
        "tuning_class": "hybrid",
        "metadata": {
            "hh_recovery_ladder_case_manifest": True,
            "diagnostic_only": True,
            "controller_settings_scope": "coarse_hamiltonian_class",
            "qpu_faithful_controller_data_contract": "measurement_compatible_prepared_state_observables_only",
            "enable_drive": True,
            "disable_drive": False,
            "drive": {
                "enable_drive": True,
                "A": amp,
                "omega": 1.0,
                "tbar": 1.0,
                "phi": 0.0,
                "pattern": "staggered",
                "custom_weights": "",
                "include_identity": False,
                "time_sampling": "midpoint",
                "t0": 0.0,
            },
            "seed_lock": {
                "seed_track": "oldgood",
                "normalized_seed_artifact_json": str(info["artifact_json"]),
                "same_seed_comparator_group_id": f"hh_oldgood_{tag}_t8_dt321_recovery_ladder_v1",
                "seed_selection_policy": "recovered_20260429_old_good_exact_assisted_regression_seed",
                "legacy_exact_assisted_record_id": str(info["legacy_record_id"]),
                "legacy_exact_assisted_result_json": str(info["legacy_result_json"]),
                "legacy_mean_abs_energy_total_error": float(info["mean_abs_energy_total_error"]),
                "legacy_max_abs_energy_total_error": float(info["max_abs_energy_total_error"]),
            },
            "diagnostic_exact_reference_mode": "benchmark_exact_reporting_only",
            "time_dependence": "driven_staggered_midpoint",
        },
    }


def _recovery_seedtrack_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    out["case_id"] = str(raw["case_id"]).replace("table1_", "hh_recovery_seedtracks_")
    metadata = dict(out.get("metadata") or {})
    metadata["hh_recovery_ladder_case_manifest"] = True
    metadata["diagnostic_only"] = True
    metadata["current_failed_seedtracks_batch_promoted"] = False
    out["metadata"] = metadata
    return out


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "record_id",
        "kind",
        "family",
        "case_id",
        "algorithm_id",
        "variants",
        "case_manifest",
        "tuning_class",
        "legacy_record_id",
        "legacy_records",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _submit_text(*, name: str, records: str, ids: str, runtime: int, memory: str = "12GB") -> str:
    return f"""universe = vanilla
executable = chtc/generic_time_dynamics_table/run_task_apptainer.sh
arguments = $(record_id) {records}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = pipelines, src, test_support, MATH/Math.md, run_guide.md, AGENTS.md, chtc/generic_time_dynamics_table, chtc/time_dynamics_optuna, chtc/time_dynamics_optuna/image.sif
transfer_output_files = raw_outputs, logs
log = logs/{name}.$(Cluster).$(Process).log
output = logs/{name}.$(Cluster).$(Process).out
error = logs/{name}.$(Cluster).$(Process).err
requirements = TARGET.HasSIF
request_cpus = 1
request_memory = {memory}
request_disk = 40GB
+MaxRuntime = {runtime}
+JobBatchName = "{name}"
environment = "GENERIC_TD_TABLE_RECORDS_PATH={records} GENERIC_TD_CLASS_SETTINGS_MANIFEST={CLASS_LOCK} GENERIC_TD_REQUIRE_LOCKED_CLASS_SETTINGS=1"
queue record_id from {ids}
"""


def main() -> int:
    INPUT.mkdir(parents=True, exist_ok=True)
    seedtrack = _load_seedtrack_hh_cases()
    cases: list[dict[str, Any]] = []
    for tag, info in OLDGOOD.items():
        cases.append(_oldgood_case(tag, info))
    for amp in (0.2, 0.6):
        cases.append(_recovery_seedtrack_case(seedtrack[amp]))

    case_manifest = INPUT / "hh_recovery_ladder_cases_v1.json"
    case_manifest.write_text(json.dumps({"schema": SCHEMA, "cases": cases}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    records: list[dict[str, str]] = []
    stage0_ids: list[str] = []
    oldgood_strict_ids: list[str] = []
    seedtrack_strict_ids: list[str] = []
    rel_manifest = "chtc/generic_time_dynamics_table/input/hh_recovery_ladder_cases_v1.json"
    for tag, info in OLDGOOD.items():
        rid = f"hh_recovery_stage0_exact_assisted_{tag}_v1"
        stage0_ids.append(rid)
        records.append({
            "record_id": rid,
            "kind": "legacy_optuna",
            "family": "hh",
            "case_id": str(info["case_id"]),
            "algorithm_id": "legacy_exact_assisted_hh_recovery",
            "variants": "stage0_exact_assisted_regression",
            "case_manifest": rel_manifest,
            "tuning_class": "hybrid",
            "legacy_record_id": str(info["legacy_record_id"]),
            "legacy_records": "chtc/time_dynamics_optuna/input/records.tsv",
        })
    for case in cases:
        track = "oldgood" if "oldgood" in case["case_id"] else "seedtracks"
        amp = float(((case.get("metadata") or {}).get("drive") or {}).get("A"))
        amp_tag = f"A{str(amp).replace('.', 'p')}"
        for stage, variant in STAGES:
            rid = f"hh_recovery_{track}_{amp_tag}_s{stage}_v1"
            (oldgood_strict_ids if track == "oldgood" else seedtrack_strict_ids).append(rid)
            records.append({
                "record_id": rid,
                "kind": "ablation",
                "family": "hh",
                "case_id": str(case["case_id"]),
                "algorithm_id": "dyn_controller_ablation_matrix",
                "variants": variant,
                "case_manifest": rel_manifest,
                "tuning_class": "hybrid",
                "legacy_record_id": "",
                "legacy_records": "",
            })

    records_path = INPUT / "hh_recovery_ladder_records_v1.tsv"
    _write_tsv(records_path, records)
    files = {
        "hh_recovery_ladder_stage0_record_ids_v1.txt": stage0_ids,
        "hh_recovery_ladder_oldgood_strict_record_ids_v1.txt": oldgood_strict_ids,
        "hh_recovery_ladder_seedtracks_strict_record_ids_v1.txt": seedtrack_strict_ids,
        "hh_recovery_ladder_all_record_ids_v1.txt": [r["record_id"] for r in records],
    }
    for name, ids in files.items():
        (INPUT / name).write_text("\n".join(ids) + "\n", encoding="utf-8")

    records_rel = "chtc/generic_time_dynamics_table/input/hh_recovery_ladder_records_v1.tsv"
    (ROOT / "chtc" / "generic_time_dynamics_table" / "submit_hh_recovery_ladder_stage0_v1.sub").write_text(
        _submit_text(
            name="holstein-hh-recovery-ladder-stage0-v1",
            records=records_rel,
            ids="chtc/generic_time_dynamics_table/input/hh_recovery_ladder_stage0_record_ids_v1.txt",
            runtime=28800,
        ).replace(f" GENERIC_TD_CLASS_SETTINGS_MANIFEST={CLASS_LOCK} GENERIC_TD_REQUIRE_LOCKED_CLASS_SETTINGS=1", ""),
        encoding="utf-8",
    )
    (ROOT / "chtc" / "generic_time_dynamics_table" / "submit_hh_recovery_ladder_oldgood_strict_v1.sub").write_text(
        _submit_text(
            name="holstein-hh-recovery-ladder-oldgood-strict-v1",
            records=records_rel,
            ids="chtc/generic_time_dynamics_table/input/hh_recovery_ladder_oldgood_strict_record_ids_v1.txt",
            runtime=28800,
        ),
        encoding="utf-8",
    )
    (ROOT / "chtc" / "generic_time_dynamics_table" / "submit_hh_recovery_ladder_seedtracks_strict_v1.sub").write_text(
        _submit_text(
            name="holstein-hh-recovery-ladder-seedtracks-strict-v1",
            records=records_rel,
            ids="chtc/generic_time_dynamics_table/input/hh_recovery_ladder_seedtracks_strict_record_ids_v1.txt",
            runtime=28800,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "schema": SCHEMA,
        "case_manifest": str(case_manifest),
        "records": str(records_path),
        "stage0_records": len(stage0_ids),
        "oldgood_strict_records": len(oldgood_strict_ids),
        "seedtracks_strict_records": len(seedtrack_strict_ids),
        "total_records": len(records),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
