#!/usr/bin/env python3
"""Generate Paper-I HH SNAKE Pauli-child repair records.

This generator is SNAKE-only.  It separates the Phase-III runtime-split repair
from the pre-Phase-1/global child-pool rerun so the two requested algorithmic
changes are not conflated with the old Powell diagnostic batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chtc.phase3_optuna.generate_paper_i_hh_spsa_budget_ladder_records import (
    FIELDNAMES,
    NATIVE_ENGINE,
    configure_batch,
    output_paths,
    rel_or_abs,
)
from chtc.phase3_optuna.generate_paper_i_hh_native200_depth30_records import (
    REGIME_ORDER,
    SNAKE_RECORDS,
    SUPPORT_JSON,
    manifest_path_for_source,
    read_json,
    read_records,
    support_rows,
    write_submit_file,
)
from pipelines.exact_bench.table_i_canonical_cases import table_i_canonical_spec_by_case_id


DEFAULT_BATCH_ID = "paper_i_hh_snake_pauli_child_repair_20260626_v1"
PHASE3_RUNTIME_SPLIT_MODE = "shortlist_pauli_children_v1"
PHASE0_GLOBAL_CHILD_POOL_MODE = "global_pauli_child_sets_v1"
HARD_GUARD = "hard_guard"
PARENT_POLICY = "parent"
FIXED_HORIZON_MARKER = "forced_depth30_no_early_stop"
LANES = ("phase3_parent_anchor", "phase3_hardguard_qminus1", "phase0_global_qminus1")


def write_lines(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def source_record_index(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in read_records(path):
        if row.get("method_key") != "snake":
            continue
        if row.get("engine_key") != "native_forced" or row.get("budget") != "200":
            continue
        out[str(row.get("display_regime"))] = row
    return out


def qubit_cap_minus_one(source: Mapping[str, str]) -> int:
    spec = table_i_canonical_spec_by_case_id(
        "hh",
        str(source["case_id"]),
        str(source["suite_profile"]),
    )
    return max(1, int(spec.features.n_qubits) - 1)


def changed_fields_for_lane(lane: str) -> list[str]:
    common = [
        "max_depth",
        "adapt_segment_target_depth",
        "adapt_segment_max_new_admissions",
        "disable_drop_stop",
        "disable_benchmark_target_stop",
    ]
    if lane == "phase3_parent_anchor":
        return common + [
            "snake_phase3_runtime_split_mode",
            "snake_phase3_runtime_split_child_set_symmetry_policy",
            "snake_phase3_runtime_split_max_subset_size",
        ]
    if lane == "phase3_hardguard_qminus1":
        return common + [
            "snake_phase3_runtime_split_mode",
            "snake_phase3_runtime_split_child_set_symmetry_policy",
            "snake_phase3_runtime_split_max_subset_size",
        ]
    if lane == "phase0_global_qminus1":
        return common + [
            "snake_phase3_runtime_split_mode",
            "snake_adapt_child_pool_expansion_mode",
            "snake_adapt_child_pool_expansion_symmetry_policy",
            "snake_adapt_child_pool_expansion_max_subset_size",
        ]
    raise ValueError(f"unknown lane: {lane!r}")


def apply_lane_fields(row: dict[str, str], *, lane: str, max_subset_size: int) -> None:
    row["snake_phase3_runtime_split_child_set_symmetry_policy"] = ""
    row["snake_phase3_runtime_split_max_subset_size"] = ""
    row["snake_adapt_child_pool_expansion_mode"] = ""
    row["snake_adapt_child_pool_expansion_symmetry_policy"] = ""
    row["snake_adapt_child_pool_expansion_max_subset_size"] = ""
    if lane == "phase3_parent_anchor":
        row["snake_phase3_runtime_split_mode"] = PHASE3_RUNTIME_SPLIT_MODE
        row["snake_phase3_runtime_split_child_set_symmetry_policy"] = PARENT_POLICY
        row["snake_phase3_runtime_split_max_subset_size"] = "3"
    elif lane == "phase3_hardguard_qminus1":
        row["snake_phase3_runtime_split_mode"] = PHASE3_RUNTIME_SPLIT_MODE
        row["snake_phase3_runtime_split_child_set_symmetry_policy"] = HARD_GUARD
        row["snake_phase3_runtime_split_max_subset_size"] = str(int(max_subset_size))
    elif lane == "phase0_global_qminus1":
        row["snake_phase3_runtime_split_mode"] = "off"
        row["snake_adapt_child_pool_expansion_mode"] = PHASE0_GLOBAL_CHILD_POOL_MODE
        row["snake_adapt_child_pool_expansion_symmetry_policy"] = HARD_GUARD
        row["snake_adapt_child_pool_expansion_max_subset_size"] = str(int(max_subset_size))
    else:
        raise ValueError(f"unknown lane: {lane!r}")


def make_row(
    *,
    batch_id: str,
    regime: str,
    lane: str,
    source: Mapping[str, str],
    anchor: Mapping[str, Any],
) -> dict[str, str]:
    max_subset_size = qubit_cap_minus_one(source)
    record_id = (
        f"{batch_id}__{regime.replace('-', '_')}"
        f"__snake__native_forced__maxiter200__depth30_noearlystop__{lane}"
    )
    row = dict(source)
    row.update(
        {
            "record_id": record_id,
            "batch_id": batch_id,
            "run_class": "candidate",
            "runnable": "true" if str(source.get("runnable") or "") == "true" else "false",
            "blocker": str(source.get("blocker") or ""),
            "method_key": "snake",
            "method_label": "SNAKE",
            "algorithm_id": "static_family_native_adapt_phase3",
            "engine_key": "native_forced",
            "engine_label": "native forced full budget",
            "spsa_refit_engine": NATIVE_ENGINE,
            "budget": "200",
            "max_depth": "30",
            "source_settings_status": f"ok_snake_pauli_child_repair_{lane}",
            "schedule_source_policy": "paper_i_snake_pauli_child_repair",
            "schedule_source_regime": regime,
            "schedule_source_method": "SNAKE",
            "schedule_source_json": str(anchor.get("source_json") or source.get("source_json") or ""),
            "schedule_source_note": (
                f"{FIXED_HORIZON_MARKER}; SNAKE-only Pauli-child repair lane={lane}; "
                "native-forced SPSA maxiter=200; not the Powell diagnostic batch"
            ),
            "anchor_source_json": str(anchor.get("source_json") or ""),
            "anchor_source_sha256": str(anchor.get("source_sha256") or ""),
            "anchor_cell_manifest_rel": manifest_path_for_source(str(anchor.get("source_json") or "")),
            "changed_fields_vs_anchor": ",".join(changed_fields_for_lane(lane)),
            "source_contract_note": (
                f"{FIXED_HORIZON_MARKER}; source-locked SNAKE rerun from the native200/depth30 "
                f"support row. lane={lane}; per-regime child-set cap is q-1={max_subset_size} "
                "for hard-guard lanes."
            ),
        }
    )
    apply_lane_fields(row, lane=lane, max_subset_size=max_subset_size)
    row.update(output_paths(record_id, "snake"))
    for field in FIELDNAMES:
        row.setdefault(field, "")
    return row


def build_records(batch_id: str, lanes: Sequence[str]) -> list[dict[str, str]]:
    anchors = support_rows(SUPPORT_JSON)
    sources = source_record_index(SNAKE_RECORDS)
    rows: list[dict[str, str]] = []
    missing: dict[str, list[str]] = defaultdict(list)
    for regime in REGIME_ORDER:
        anchor = anchors.get((regime, "SNAKE"))
        source = sources.get(regime)
        if anchor is None:
            missing["anchor"].append(f"{regime}/SNAKE")
            continue
        if source is None:
            missing["source_record"].append(f"{regime}/snake")
            continue
        for lane in lanes:
            rows.append(
                make_row(
                    batch_id=batch_id,
                    regime=regime,
                    lane=lane,
                    source=source,
                    anchor=anchor,
                )
            )
    if missing:
        raise SystemExit(json.dumps({"status": "blocked_missing_sources", "missing": missing}, indent=2))
    return rows


def write_records(batch_id: str, records: Sequence[dict[str, str]]) -> dict[str, Any]:
    input_dir = ROOT / "chtc" / "phase3_optuna" / "input" / batch_id
    records_tsv = input_dir / "paper_i_hh_spsa_budget_ladder_records.tsv"
    all_ids = input_dir / "paper_i_hh_spsa_budget_ladder_record_ids.txt"
    smoke_ids = input_dir / "paper_i_hh_spsa_budget_ladder_smoke_record_ids.txt"
    phase3_anchor_ids = input_dir / "paper_i_hh_snake_phase3_parent_anchor_record_ids.txt"
    phase3_repair_ids = input_dir / "paper_i_hh_snake_phase3_hardguard_qminus1_record_ids.txt"
    phase0_global_ids = input_dir / "paper_i_hh_snake_phase0_global_qminus1_record_ids.txt"
    manifest_json = input_dir / "paper_i_hh_spsa_budget_ladder_manifest.json"
    input_dir.mkdir(parents=True, exist_ok=True)
    with records_tsv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDNAMES), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    by_lane: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        lane = str(row["record_id"]).rsplit("__", 1)[-1]
        by_lane[lane].append(dict(row))

    write_lines(all_ids, (row["record_id"] for row in records))
    write_lines(phase3_anchor_ids, (row["record_id"] for row in by_lane["phase3_parent_anchor"]))
    write_lines(phase3_repair_ids, (row["record_id"] for row in by_lane["phase3_hardguard_qminus1"]))
    write_lines(phase0_global_ids, (row["record_id"] for row in by_lane["phase0_global_qminus1"]))
    smoke_source = by_lane["phase3_parent_anchor"] or list(records)
    write_lines(smoke_ids, (smoke_source[0]["record_id"],))

    submit_anchor = ROOT / "chtc" / "phase3_optuna" / f"submit_{batch_id}_phase3_anchor.sub"
    submit_phase3_repair = ROOT / "chtc" / "phase3_optuna" / f"submit_{batch_id}_phase3_hardguard.sub"
    submit_phase0_global = ROOT / "chtc" / "phase3_optuna" / f"submit_{batch_id}_phase0_global.sub"
    write_submit_file(
        batch_id=batch_id,
        submit_path=submit_anchor,
        records_tsv=records_tsv,
        record_ids=phase3_anchor_ids,
        job_batch_suffix="phase3_anchor",
    )
    write_submit_file(
        batch_id=batch_id,
        submit_path=submit_phase3_repair,
        records_tsv=records_tsv,
        record_ids=phase3_repair_ids,
        job_batch_suffix="phase3_hardguard",
    )
    write_submit_file(
        batch_id=batch_id,
        submit_path=submit_phase0_global,
        records_tsv=records_tsv,
        record_ids=phase0_global_ids,
        job_batch_suffix="phase0_global",
    )

    manifest = {
        "schema": "paper_i_hh_snake_pauli_child_repair_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "run_class": "candidate",
        "source_contract": {
            "support_json": rel_or_abs(SUPPORT_JSON),
            "support_schema": read_json(SUPPORT_JSON).get("schema"),
            "source_records_tsv": rel_or_abs(SNAKE_RECORDS),
            "method_scope": ["SNAKE"],
            "regimes": list(REGIME_ORDER),
            "preserved": {
                "spsa_engine": "native_forced",
                "spsa_refit_engine": NATIVE_ENGINE,
                "maxiter": 200,
                "max_depth": 30,
                "fixed_horizon": True,
            },
            "submission_gate": (
                "Submit phase3_parent_anchor first. Submit phase3_hardguard_qminus1 and "
                "phase0_global_qminus1 only after anchor reproduction is reviewed."
            ),
            "phase3_parent_anchor": {
                "runtime_split_mode": PHASE3_RUNTIME_SPLIT_MODE,
                "child_set_symmetry_policy": PARENT_POLICY,
                "max_subset_size": 3,
            },
            "phase3_hardguard_qminus1": {
                "runtime_split_mode": PHASE3_RUNTIME_SPLIT_MODE,
                "child_set_symmetry_policy": HARD_GUARD,
                "max_subset_size_policy": "q_minus_1",
            },
            "phase0_global_qminus1": {
                "runtime_split_mode": "off",
                "child_pool_expansion_mode": PHASE0_GLOBAL_CHILD_POOL_MODE,
                "child_pool_expansion_symmetry_policy": HARD_GUARD,
                "max_subset_size_policy": "q_minus_1",
            },
        },
        "paths": {
            "records_tsv": rel_or_abs(records_tsv),
            "all_record_ids_txt": rel_or_abs(all_ids),
            "smoke_record_ids_txt": rel_or_abs(smoke_ids),
            "phase3_anchor_record_ids_txt": rel_or_abs(phase3_anchor_ids),
            "phase3_hardguard_record_ids_txt": rel_or_abs(phase3_repair_ids),
            "phase0_global_record_ids_txt": rel_or_abs(phase0_global_ids),
            "phase3_anchor_submit": rel_or_abs(submit_anchor),
            "phase3_hardguard_submit": rel_or_abs(submit_phase3_repair),
            "phase0_global_submit": rel_or_abs(submit_phase0_global),
        },
        "record_count": len(records),
        "lane_counts": {lane: len(rows) for lane, rows in sorted(by_lane.items())},
        "records": [
            {
                "record_id": row["record_id"],
                "display_regime": row["display_regime"],
                "lane": row["record_id"].rsplit("__", 1)[-1],
                "runtime_split_mode": row.get("snake_phase3_runtime_split_mode", ""),
                "runtime_split_child_set_symmetry_policy": row.get(
                    "snake_phase3_runtime_split_child_set_symmetry_policy", ""
                ),
                "runtime_split_max_subset_size": row.get("snake_phase3_runtime_split_max_subset_size", ""),
                "child_pool_expansion_mode": row.get("snake_adapt_child_pool_expansion_mode", ""),
                "child_pool_expansion_max_subset_size": row.get(
                    "snake_adapt_child_pool_expansion_max_subset_size", ""
                ),
            }
            for row in records
        ],
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument(
        "--lanes",
        default=",".join(LANES),
        help=f"Comma-separated lanes to materialize. Choices: {','.join(LANES)}.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    batch_id = str(args.batch_id)
    lanes = tuple(lane.strip() for lane in str(args.lanes).split(",") if lane.strip())
    unknown = [lane for lane in lanes if lane not in LANES]
    if unknown:
        raise SystemExit(f"unknown lane(s): {', '.join(unknown)}")
    configure_batch(batch_id)
    records = build_records(batch_id, lanes)
    manifest = write_records(batch_id, records)
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
