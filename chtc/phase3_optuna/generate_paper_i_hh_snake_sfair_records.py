#!/usr/bin/env python3
"""Generate SNAKE-only Paper-I HH fair-shot-accounting repair records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chtc.phase3_optuna.generate_paper_i_hh_monotone_child_schur_records import (
    build_records as build_monotone_records,
)
from chtc.phase3_optuna.generate_paper_i_hh_native_forced_child_matrix_records import (
    build_records as build_native_records,
)
from chtc.phase3_optuna.generate_paper_i_hh_native200_depth30_records import (
    REGIME_ORDER,
    write_submit_file,
)
from chtc.phase3_optuna.generate_paper_i_hh_spsa_budget_ladder_records import (
    FIELDNAMES,
    configure_batch,
    rel_or_abs,
)


DEFAULT_BATCH_ID = "paper_i_hh_snake_sfair_depth30_20260623_v1"


def write_lines(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def _repair_note(row: dict[str, str], page: str) -> dict[str, str]:
    out = dict(row)
    out["source_settings_status"] = f"ok_snake_sfair_repair_{page}"
    out["schedule_source_policy"] = "paper_i_hh_snake_sfair_repair_grid"
    out["schedule_source_note"] = (
        str(out.get("schedule_source_note") or "").rstrip("; ")
        + "; SNAKE fair-shot repair rerun; emits snake_fair_shot_work.json"
    )
    out["source_contract_note"] = (
        str(out.get("source_contract_note") or "").rstrip()
        + " This repair batch preserves the existing page settings and changes only the shot-accounting telemetry: "
        "visible S_fair must come from explicit trajectory-conditioned common-exposure logical operator-probe events."
    )
    return out


def build_records(batch_id: str) -> list[dict[str, str]]:
    configure_batch(batch_id)
    rows: list[dict[str, str]] = []
    for row in build_native_records(batch_id):
        if row.get("method_key") == "snake":
            rows.append(_repair_note(row, "native_forced"))
    for row in build_monotone_records(batch_id):
        if row.get("method_key") == "snake":
            rows.append(_repair_note(row, "monotone_child_schur"))
    return rows


def write_records(batch_id: str, records: Sequence[dict[str, str]]) -> dict[str, Any]:
    input_dir = ROOT / "chtc" / "phase3_optuna" / "input" / batch_id
    records_tsv = input_dir / "paper_i_hh_spsa_budget_ladder_records.tsv"
    record_ids = input_dir / "paper_i_hh_spsa_budget_ladder_record_ids.txt"
    smoke_ids = input_dir / "paper_i_hh_spsa_budget_ladder_smoke_record_ids.txt"
    manifest_json = input_dir / "paper_i_hh_spsa_budget_ladder_manifest.json"
    submit_path = ROOT / "chtc" / "phase3_optuna" / f"submit_{batch_id}.sub"
    input_dir.mkdir(parents=True, exist_ok=True)
    with records_tsv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDNAMES), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    write_lines(record_ids, (row["record_id"] for row in records))
    write_lines(smoke_ids, (records[0]["record_id"],))
    by_engine: dict[str, list[str]] = defaultdict(list)
    by_child: dict[str, list[str]] = defaultdict(list)
    for row in records:
        by_engine[str(row.get("engine_key") or "")].append(row["record_id"])
        child_mode = "polychildren" if str(row.get("snake_phase3_runtime_split_mode") or "") else "no_child"
        if str(row.get("snake_phase3_runtime_split_mode") or "") == "off":
            child_mode = "no_child"
        by_child[child_mode].append(row["record_id"])
    for engine, ids in sorted(by_engine.items()):
        write_lines(input_dir / f"paper_i_hh_snake_sfair_{engine}_record_ids.txt", ids)
    for child_mode, ids in sorted(by_child.items()):
        write_lines(input_dir / f"paper_i_hh_snake_sfair_{child_mode}_record_ids.txt", ids)
    manifest = {
        "schema": "paper_i_hh_snake_sfair_depth30_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "run_class": "diagnostic",
        "repair_objective": (
            "Produce SNAKE snake_fair_shot_work.json sidecars with explicit actual and "
            "trajectory-conditioned common-exposure logical operator-probe ledgers for the visible S_fair."
        ),
        "sidecar_contract": {
            "visible_column": "S_fair",
            "required_sidecar": "snake_fair_shot_work.json",
            "required_status": "ok",
            "fair_work_currency": "expanded_common_candidate_probe_event_count_v1",
            "common_exposure_policy_id": "trajectory_conditioned_full_child_common_exposure_v1",
            "operator_probe_charge_basis": "logical_estimator_request_pre_grouping_v1",
            "legacy_grouped_policy": "preserve old/mixed S only as S_unfair or diagnostic provenance",
        },
        "paths": {
            "records_tsv": rel_or_abs(records_tsv),
            "record_ids_txt": rel_or_abs(record_ids),
            "smoke_record_ids_txt": rel_or_abs(smoke_ids),
            "submit_file": rel_or_abs(submit_path),
        },
        "record_count": len(records),
        "intended_grid": {
            "methods": ["SNAKE"],
            "regimes": list(REGIME_ORDER),
            "child_modes": ["no_child", "polychildren"],
            "page_settings": ["native_forced", "monotone_child_schur"],
            "intended_record_count": len(REGIME_ORDER) * 2 * 2,
        },
        "records": [
            {
                "record_id": row["record_id"],
                "display_regime": row["display_regime"],
                "method_key": row["method_key"],
                "engine_key": row["engine_key"],
                "child_mode": "polychildren"
                if str(row.get("snake_phase3_runtime_split_mode") or "") not in {"", "off"}
                else "no_child",
                "snake_phase3_runtime_split_mode": row.get("snake_phase3_runtime_split_mode"),
                "adapt_schur_warm_start_mode": row.get("adapt_schur_warm_start_mode"),
                "budget": row["budget"],
                "max_depth": row["max_depth"],
            }
            for row in records
        ],
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_submit_file(batch_id=batch_id, submit_path=submit_path, records_tsv=records_tsv, record_ids=record_ids)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    batch_id = str(args.batch_id)
    records = build_records(batch_id)
    manifest = write_records(batch_id, records)
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
