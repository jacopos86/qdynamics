#!/usr/bin/env python3
"""Generate the bounded L=2/nph=1 tripartite SPSA A/B CHTC reset batch.

This reset intentionally replaces only the four still-problematic bosonic and
mixed A/B lanes from the previous cluster while preserving the completed
fermionic A/B record IDs and outputs.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input"
SOURCE_RECORDS = INPUT_DIR / "global_tripartite_spsa_ab_records.tsv"
FULL_RECORDS = INPUT_DIR / "global_tripartite_spsa_ab_l2nph1_reset_records.tsv"
FULL_IDS = INPUT_DIR / "global_tripartite_spsa_ab_l2nph1_reset_record_ids.txt"
SMOKE_RECORDS = INPUT_DIR / "global_tripartite_spsa_ab_l2nph1_reset_smoke_records.tsv"
SMOKE_IDS = INPUT_DIR / "global_tripartite_spsa_ab_l2nph1_reset_smoke_record_ids.txt"
MANIFEST = INPUT_DIR / "global_tripartite_spsa_ab_l2nph1_reset_manifest.json"

REPLACED_CLUSTER_ID = "6305147"
BATCH_ID = "global_tripartite_spsa_ab_l2nph1_reset_v1"

PRESERVED_FERMIONIC_RECORD_IDS = (
    "tripartite_spsa_A_current_collective_fermionic_smallrobust_target5e5_v1",
    "tripartite_spsa_B_pre_phase2_novelty_legacy_fermionic_smallrobust_target5e5_v1",
)

RESET_ROWS: tuple[dict[str, str], ...] = (
    {
        "source_id": "tripartite_spsa_A_current_collective_bosonic_smallrobust_target5e5_v1",
        "full_id": "reset6305147_spsa_A_bosonic_bose_hubbard_L2_nph1_smallrobust_target5e5_v1",
        "smoke_id": "reset6305147_spsa_A_bosonic_bose_hubbard_L2_nph1_smoke_v1",
        "families": "bose_hubbard",
        "tuning_class": "bosonic",
        "source_table_class": "bosonic_lattice",
        "policy_search_profile": "bosonic_fullmeta_compact",
    },
    {
        "source_id": "tripartite_spsa_B_pre_phase2_novelty_legacy_bosonic_smallrobust_target5e5_v1",
        "full_id": "reset6305147_spsa_B_bosonic_bose_hubbard_L2_nph1_smallrobust_target5e5_v1",
        "smoke_id": "reset6305147_spsa_B_bosonic_bose_hubbard_L2_nph1_smoke_v1",
        "families": "bose_hubbard",
        "tuning_class": "bosonic",
        "source_table_class": "bosonic_lattice",
        "policy_search_profile": "bosonic_fullmeta_compact",
    },
    {
        "source_id": "tripartite_spsa_A_current_collective_mixed_smallrobust_target5e5_v1",
        "full_id": "reset6305147_spsa_A_mixed_hh_L2_nph1_smallrobust_target5e5_v1",
        "smoke_id": "reset6305147_spsa_A_mixed_hh_L2_nph1_smoke_v1",
        "families": "hh",
        "tuning_class": "mixed_fermion_boson",
        "source_table_class": "mixed_fermion_boson",
        "policy_search_profile": "default",
    },
    {
        "source_id": "tripartite_spsa_B_pre_phase2_novelty_legacy_mixed_smallrobust_target5e5_v1",
        "full_id": "reset6305147_spsa_B_mixed_hh_L2_nph1_smallrobust_target5e5_v1",
        "smoke_id": "reset6305147_spsa_B_mixed_hh_L2_nph1_smoke_v1",
        "families": "hh",
        "tuning_class": "mixed_fermion_boson",
        "source_table_class": "mixed_fermion_boson",
        "policy_search_profile": "default",
    },
)


def read_records(path: Path = SOURCE_RECORDS) -> tuple[list[str], dict[str, dict[str, str]]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a TSV header")
        rows = {str(row.get("record_id") or ""): dict(row) for row in reader if row.get("record_id")}
    return list(reader.fieldnames), rows


def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _reset_row(source: Mapping[str, str], spec: Mapping[str, str], *, smoke: bool) -> dict[str, str]:
    row = dict(source)
    row.update(
        {
            "record_id": spec["smoke_id"] if smoke else spec["full_id"],
            "mode": "global",
            "families": spec["families"],
            "sizes": "2",
            "boson_cutoff": "1",
            "boson_cutoffs": "",
            "exact_reference_boson_cutoff": "0",
            "physics_grid_profile": "canonical" if smoke else "small_robust",
            "oracle_enqueue_limit": "12",
            "objective_profile": "cost_at_accuracy",
            "target_abs_delta_e": "5e-5",
            "objective_energy_weight": "8.0",
            "objective_2q_weight": "2.0",
            "objective_depth_weight": "1.0",
            "objective_parameter_weight": "0.5",
            "objective_shot_weight": "2.0",
            "n_trials": "1" if smoke else "36",
            "n_jobs": "1",
            "benchmarks_per_trial_jobs": "1",
            "trial_timeout_sec": "1200" if smoke else "7200",
            "compile_timeout_sec": "300" if smoke else "900",
            "enqueue_default": _bool_text(smoke),
            "enqueue_historical": "true",
            "fixed_inner_optimizer": "SPSA",
            "canonical_lane": "",
            "canonical_lane_stage": "",
            "policy_search_profile": spec["policy_search_profile"],
            "required_target_profile": "none",
            "required_target_abs_delta_e": "",
            "required_target_penalty": "1000.0",
            "robustness_gate": "off",
        }
    )
    return row


def build_rows(source_path: Path = SOURCE_RECORDS) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    fieldnames, source_rows = read_records(source_path)
    missing = [spec["source_id"] for spec in RESET_ROWS if spec["source_id"] not in source_rows]
    if missing:
        raise ValueError(f"source reset records missing from {source_path}: {missing}")
    full_rows = [_reset_row(source_rows[spec["source_id"]], spec, smoke=False) for spec in RESET_ROWS]
    smoke_rows = [_reset_row(source_rows[spec["source_id"]], spec, smoke=True) for spec in RESET_ROWS]
    validate_reset_rows(full_rows, smoke=False, source_ids=source_rows.keys())
    validate_reset_rows(smoke_rows, smoke=True, source_ids=source_rows.keys())
    return fieldnames, full_rows, smoke_rows


def validate_reset_rows(rows: Sequence[Mapping[str, str]], *, smoke: bool, source_ids: Sequence[str] = ()) -> None:
    if len(rows) != 4:
        raise ValueError(f"expected exactly four {'smoke' if smoke else 'full'} reset rows, got {len(rows)}")
    ids = [str(row.get("record_id") or "") for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate reset record IDs: {ids}")
    source_id_set = {str(x) for x in source_ids}
    overlap = sorted(set(ids) & source_id_set)
    if overlap:
        raise ValueError(f"reset record IDs overlap existing source IDs: {overlap}")
    for row in rows:
        rid = str(row.get("record_id") or "")
        if "fermionic" in rid or str(row.get("canonical_lane") or "").strip() == "fermionic":
            raise ValueError(f"reset row must not requeue fermionic lane: {rid}")
        family = str(row.get("families") or "").strip()
        if family not in {"bose_hubbard", "hh"}:
            raise ValueError(f"{rid}: families must be bose_hubbard or hh, got {family!r}")
        if str(row.get("sizes") or "").strip() != "2":
            raise ValueError(f"{rid}: sizes must be exactly 2")
        if str(row.get("boson_cutoff") or "").strip() != "1":
            raise ValueError(f"{rid}: boson_cutoff must be exactly 1")
        if str(row.get("boson_cutoffs") or "").strip():
            raise ValueError(f"{rid}: boson_cutoffs must be blank")
        if str(row.get("exact_reference_boson_cutoff") or "").strip() != "0":
            raise ValueError(f"{rid}: exact_reference_boson_cutoff must be 0")
        expected_profile = "canonical" if smoke else "small_robust"
        if str(row.get("physics_grid_profile") or "").strip() != expected_profile:
            raise ValueError(f"{rid}: physics_grid_profile must be {expected_profile}")
        if str(row.get("fixed_inner_optimizer") or "").strip().upper() != "SPSA":
            raise ValueError(f"{rid}: fixed_inner_optimizer must be SPSA")
        novelty = str(row.get("phase2_novelty_mode") or "").strip()
        if novelty not in {"collective_span_v1", "legacy_pairwise_v1"}:
            raise ValueError(f"{rid}: phase2_novelty_mode must be explicit A/B route label")
        if family == "bose_hubbard" and "bose_hubbard" not in rid:
            raise ValueError(f"{rid}: bose_hubbard family should be visible in record ID")
        if family == "hh" and "_hh_" not in rid:
            raise ValueError(f"{rid}: hh family should be visible in record ID")


def _tsv_text(fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fieldnames), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return buf.getvalue()


def _ids_text(rows: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(str(row["record_id"]) for row in rows) + "\n"


def _manifest_payload(full_rows: Sequence[Mapping[str, str]], smoke_rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    smoke_by_source = {spec["source_id"]: smoke_rows[idx] for idx, spec in enumerate(RESET_ROWS)}
    records: list[dict[str, object]] = []
    for idx, spec in enumerate(RESET_ROWS):
        full = full_rows[idx]
        smoke = smoke_by_source[spec["source_id"]]
        records.append(
            {
                "record_id": full["record_id"],
                "smoke_record_id": smoke["record_id"],
                "replaced_record_id": spec["source_id"],
                "tuning_class": spec["tuning_class"],
                "source_table_class": spec["source_table_class"],
                "family": full["families"],
                "constraints": {
                    "sizes": [2],
                    "boson_cutoff": 1,
                    "boson_cutoffs": [],
                    "exact_reference_boson_cutoff": 0,
                    "forbidden_sizes": [3],
                    "forbidden_boson_cutoffs": [2],
                },
                "physics_grid_profile": full["physics_grid_profile"],
                "smoke_physics_grid_profile": smoke["physics_grid_profile"],
                "algorithm_variant": full["algorithm_variant"],
                "phase2_novelty_mode": full["phase2_novelty_mode"],
                "objective_profile": full["objective_profile"],
                "objective_weights": {
                    "energy": float(full["objective_energy_weight"]),
                    "count_2q": float(full["objective_2q_weight"]),
                    "depth": float(full["objective_depth_weight"]),
                    "parameters": float(full["objective_parameter_weight"]),
                    "shot": float(full["objective_shot_weight"]),
                    "target_abs_delta_e": full["target_abs_delta_e"],
                },
                "n_trials": int(full["n_trials"]),
                "smoke_n_trials": int(smoke["n_trials"]),
                "timeouts": {
                    "trial_timeout_sec": int(float(full["trial_timeout_sec"])),
                    "compile_timeout_sec": int(float(full["compile_timeout_sec"])),
                    "smoke_trial_timeout_sec": int(float(smoke["trial_timeout_sec"])),
                    "smoke_compile_timeout_sec": int(float(smoke["compile_timeout_sec"])),
                },
                "output_root_pattern": f"raw_outputs/{full['record_id']}",
                "smoke_output_root_pattern": f"raw_outputs/{smoke['record_id']}",
            }
        )
    return {
        "schema": "phase3_tripartite_spsa_ab_l2nph1_reset_manifest_v1",
        "batch_id": BATCH_ID,
        "generated_by": "chtc/phase3_optuna/generate_tripartite_spsa_ab_l2nph1_reset_records.py",
        "source_records": str(SOURCE_RECORDS.relative_to(SCRIPT_DIR.parents[1])),
        "replaces": {
            "cluster_id": REPLACED_CLUSTER_ID,
            "record_ids": [spec["source_id"] for spec in RESET_ROWS],
        },
        "preserves": {"record_ids": list(PRESERVED_FERMIONIC_RECORD_IDS)},
        "records": records,
    }


def render_artifacts(source_path: Path = SOURCE_RECORDS) -> dict[Path, str]:
    fieldnames, full_rows, smoke_rows = build_rows(source_path)
    manifest = _manifest_payload(full_rows, smoke_rows)
    return {
        FULL_RECORDS: _tsv_text(fieldnames, full_rows),
        FULL_IDS: _ids_text(full_rows),
        SMOKE_RECORDS: _tsv_text(fieldnames, smoke_rows),
        SMOKE_IDS: _ids_text(smoke_rows),
        MANIFEST: json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    }


def write_artifacts(source_path: Path = SOURCE_RECORDS) -> None:
    for path, text in render_artifacts(source_path).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def check_artifacts(source_path: Path = SOURCE_RECORDS) -> list[str]:
    errors: list[str] = []
    for path, expected in render_artifacts(source_path).items():
        try:
            actual = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            errors.append(f"missing generated artifact: {path}")
            continue
        if actual != expected:
            errors.append(f"generated artifact is stale: {path}")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-records", type=Path, default=SOURCE_RECORDS)
    parser.add_argument("--check", action="store_true", help="validate committed generated files instead of writing them")
    parser.add_argument("--write", action="store_true", help="write generated reset TSV/ID/manifest files")
    args = parser.parse_args(argv)

    if args.check:
        errors = check_artifacts(args.source_records)
        if errors:
            for error in errors:
                print(error)
            return 1
        print("reset record artifacts are current")
        return 0
    write_artifacts(args.source_records)
    print(f"wrote {len(render_artifacts(args.source_records))} reset record artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
