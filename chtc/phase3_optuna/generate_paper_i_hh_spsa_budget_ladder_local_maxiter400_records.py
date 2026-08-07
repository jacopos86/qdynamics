#!/usr/bin/env python3
"""Generate a local serial Paper-I HH SPSA engine diagnostic at maxiter=400."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chtc.phase3_optuna.generate_paper_i_hh_spsa_budget_ladder_records import (  # noqa: E402
    FIELDNAMES,
    RECORDS_TSV as SOURCE_RECORDS_TSV,
)


SOURCE_BATCH_ID = "paper_i_hh_spsa_budget_ladder_20260618_v1"
BATCH_ID = "paper_i_hh_spsa_budget_ladder_local_maxiter400_20260618_v1"
INPUT_DIR = ROOT / "chtc" / "phase3_optuna" / "input" / BATCH_ID
RECORDS_TSV = INPUT_DIR / "paper_i_hh_spsa_budget_ladder_records.tsv"
RECORD_IDS_TXT = INPUT_DIR / "paper_i_hh_spsa_budget_ladder_record_ids.txt"
MANIFEST_JSON = INPUT_DIR / "paper_i_hh_spsa_budget_ladder_manifest.json"
BUDGET = 400
SOURCE_ROW_BUDGET = 800


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def output_paths(record_id: str, method_key: str) -> dict[str, str]:
    record_root = Path("raw_outputs") / BATCH_ID / record_id
    if method_key == "snake":
        result_rel = record_root / "json" / "result.json"
        current_rel = record_root / "current.json"
    else:
        result_rel = record_root / "result" / "generic_static_single.json"
        current_rel = record_root / "adapt_iteration_progress.jsonl"
    return {
        "record_output_dir": str(record_root),
        "result_json_rel": str(result_rel),
        "current_json_rel": str(current_rel),
        "stdout_rel": str(record_root / "stdout.log"),
        "stderr_rel": str(record_root / "stderr.log"),
        "cell_manifest_rel": str(record_root / "cell_manifest.json"),
    }


def write_lines(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def read_source_rows() -> list[dict[str, str]]:
    if not SOURCE_RECORDS_TSV.exists():
        raise FileNotFoundError(f"Missing source records TSV: {rel_or_abs(SOURCE_RECORDS_TSV)}")
    rows: list[dict[str, str]] = []
    with SOURCE_RECORDS_TSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if str(row.get("budget")) != str(SOURCE_ROW_BUDGET):
                continue
            if str(row.get("runnable")) != "true":
                continue
            rows.append({str(k): "" if v is None else str(v) for k, v in row.items()})
    expected = 6 * 3 * 2
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} source rows at budget={SOURCE_ROW_BUDGET}; found {len(rows)}")
    return rows


def clone_row(row: dict[str, str]) -> dict[str, str]:
    source_record_id = str(row["record_id"])
    record_id = source_record_id.replace(SOURCE_BATCH_ID, BATCH_ID).replace(
        f"maxiter{SOURCE_ROW_BUDGET}",
        f"maxiter{BUDGET}",
    )
    if record_id == source_record_id:
        raise RuntimeError(f"Could not derive local record id from {source_record_id!r}")
    cloned = dict(row)
    cloned.update(
        {
            "record_id": record_id,
            "batch_id": BATCH_ID,
            "run_class": "diagnostic",
            "budget": str(BUDGET),
            "adapt_spsa_maxiter": str(BUDGET),
        }
    )
    cloned.update(output_paths(record_id, str(row["method_key"])))
    note = str(cloned.get("schedule_source_note") or "")
    local_note = (
        f"local_serial_diagnostic: cloned non-budget settings from {SOURCE_BATCH_ID} "
        f"budget={SOURCE_ROW_BUDGET}; changed maxiter/budget to {BUDGET}"
    )
    cloned["schedule_source_note"] = f"{note}; {local_note}" if note else local_note
    for field in FIELDNAMES:
        cloned.setdefault(field, "")
    return cloned


def write_records(records: Sequence[dict[str, str]]) -> dict[str, Any]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RECORDS_TSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDNAMES), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    write_lines(RECORD_IDS_TXT, (row["record_id"] for row in records))
    borrowed = [row for row in records if row.get("schedule_source_policy") == "borrowed_intermediate_hubbard_sector"]
    manifest = {
        "schema": "paper_i_hh_spsa_budget_ladder_local_maxiter400_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": BATCH_ID,
        "run_class": "diagnostic",
        "local_serial": True,
        "source_batch_id": SOURCE_BATCH_ID,
        "source_records_tsv": rel_or_abs(SOURCE_RECORDS_TSV),
        "source_row_budget": SOURCE_ROW_BUDGET,
        "budget": BUDGET,
        "record_count": len(records),
        "runnable_record_count": len(records),
        "borrowed_schedule_record_count": len(borrowed),
        "borrowed_schedule_policy": (
            "Same diagnostic fallback as source ladder: missing U/t=8 append/Geo schedules borrow "
            "same-lambda U/t=1.25 schedules until U/t=8 schedule-repair results are available."
        ),
        "intended_grid": {
            "budgets": [BUDGET],
            "engine_keys": ["legacy_monotone", "native_forced"],
            "methods": ["append", "geo", "snake"],
            "regimes": [
                "weak-weak",
                "intermediate-weak",
                "strong-weak",
                "weak-strong",
                "intermediate-strong",
                "strong-strong",
            ],
            "intended_record_count": 36,
        },
        "paths": {
            "records_tsv": rel_or_abs(RECORDS_TSV),
            "record_ids_txt": rel_or_abs(RECORD_IDS_TXT),
            "output_root": f"raw_outputs/{BATCH_ID}",
        },
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    records = [clone_row(row) for row in read_source_rows()]
    manifest = write_records(records)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
