#!/usr/bin/env python3
"""Summarize Paper-I HH SPSA budget ladder outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
BATCH_ID = "paper_i_hh_spsa_budget_ladder_20260618_v1"
DEFAULT_RECORDS = ROOT / "chtc" / "phase3_optuna" / "input" / BATCH_ID / "paper_i_hh_spsa_budget_ladder_records.tsv"
DEFAULT_OUTPUT_ROOT = ROOT / "raw_outputs" / BATCH_ID
DEFAULT_JSON = ROOT / "output" / "pdf" / "paper_i_hh_spsa_budget_ladder_status_20260618.json"


def read_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [{str(k): "" if v is None else str(v) for k, v in row.items()} for row in csv.DictReader(fh, delimiter="\t")]


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def result_path_for(row: dict[str, str], output_root: Path) -> Path:
    rel = Path(row["result_json_rel"])
    try:
        batch_rel = rel.relative_to(Path("raw_outputs") / BATCH_ID)
        return output_root / batch_rel
    except ValueError:
        return ROOT / rel


def manifest_path_for(row: dict[str, str], output_root: Path) -> Path:
    rel = Path(row["cell_manifest_rel"])
    try:
        batch_rel = rel.relative_to(Path("raw_outputs") / BATCH_ID)
        return output_root / batch_rel
    except ValueError:
        return ROOT / rel


def summarize(records: list[dict[str, str]], output_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in records:
        result_path = result_path_for(row, output_root)
        manifest_path = manifest_path_for(row, output_root)
        manifest = read_json(manifest_path) if manifest_path.exists() else None
        if row.get("runnable") != "true":
            status = "blocked_input"
        elif manifest and isinstance(manifest, dict):
            status = str(manifest.get("status") or "manifest_unknown")
        elif result_path.exists():
            status = "result_exists_no_manifest"
        else:
            status = "missing"
        counts[status] += 1
        rows.append(
            {
                "record_id": row["record_id"],
                "method_key": row["method_key"],
                "engine_key": row["engine_key"],
                "budget": int(row["budget"]),
                "display_regime": row["display_regime"],
                "runnable": row["runnable"] == "true",
                "status": status,
                "blocker": row.get("blocker", ""),
                "result_json": str(result_path),
                "cell_manifest": str(manifest_path),
            }
        )
    return {
        "schema": "paper_i_hh_spsa_budget_ladder_status_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": BATCH_ID,
        "record_count": len(records),
        "status_counts": dict(sorted(counts.items())),
        "rows": rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = summarize(read_records(Path(args.records)), Path(args.output_root))
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
