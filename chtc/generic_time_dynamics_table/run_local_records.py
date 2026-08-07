#!/usr/bin/env python3
"""Run generic time-dynamics records sequentially on the local machine."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
RUN_TASK = Path("chtc/generic_time_dynamics_table/run_task.sh")
CLASS_SETTINGS_LOCK = "chtc/generic_time_dynamics_table/input/class_settings/paper_ii_class_settings_lock_v1.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_records(path: Path) -> list[dict[str, str]]:
    with _repo_path(path).open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t")]


def _selected_records(records: Sequence[Mapping[str, str]], ids: Sequence[str] | None) -> list[dict[str, str]]:
    if ids is None:
        return [dict(row) for row in records]
    wanted = [str(item).strip() for item in ids if str(item).strip()]
    by_id = {str(row.get("record_id", "")): dict(row) for row in records}
    missing = [record_id for record_id in wanted if record_id not in by_id]
    if missing:
        raise ValueError(f"record id(s) not present in TSV: {missing}")
    return [by_id[record_id] for record_id in wanted]


def _load_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    return [line.strip() for line in _repo_path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _record_status_path(output_root: Path, record_id: str) -> Path:
    return _repo_path(output_root) / str(record_id) / "chtc_status.json"


def run_records(
    *,
    records_tsv: Path,
    output_root: Path,
    record_ids: Sequence[str] | None = None,
    resume: bool = True,
    stop_on_failure: bool = True,
    class_settings_manifest: str = CLASS_SETTINGS_LOCK,
) -> dict[str, object]:
    records = _selected_records(_read_records(records_tsv), record_ids)
    progress: list[dict[str, object]] = []
    env_base = dict(os.environ)
    env_base["GENERIC_TD_TABLE_RECORDS_PATH"] = str(records_tsv)
    env_base["GENERIC_TD_OUTPUT_ROOT"] = str(output_root)
    env_base["GENERIC_TD_CLASS_SETTINGS_MANIFEST"] = str(class_settings_manifest)
    env_base["GENERIC_TD_REQUIRE_LOCKED_CLASS_SETTINGS"] = "1"
    for index, record in enumerate(records):
        record_id = str(record.get("record_id", "")).strip()
        if not record_id:
            raise ValueError(f"record {index} lacks record_id")
        status_path = _record_status_path(output_root, record_id)
        if bool(resume) and status_path.exists():
            progress.append(
                {
                    "record_id": record_id,
                    "state": "skipped_existing",
                    "status_path": str(status_path),
                    "finished_utc": _utc_now(),
                }
            )
            continue
        cmd = ["bash", str(RUN_TASK), record_id, str(records_tsv)]
        started = _utc_now()
        completed = subprocess.run(cmd, cwd=ROOT, env=env_base)
        item = {
            "record_id": record_id,
            "state": "completed" if completed.returncode == 0 else "failed",
            "return_code": int(completed.returncode),
            "started_utc": started,
            "finished_utc": _utc_now(),
            "status_path": str(status_path),
        }
        progress.append(item)
        if completed.returncode != 0 and bool(stop_on_failure):
            break
    summary = {
        "schema": "generic_time_dynamics_local_runner_summary_v1",
        "generated_utc": _utc_now(),
        "records_tsv": str(records_tsv),
        "output_root": str(output_root),
        "record_count_requested": len(records),
        "record_count_completed": sum(1 for item in progress if item.get("state") in {"completed", "skipped_existing"}),
        "record_count_failed": sum(1 for item in progress if item.get("state") == "failed"),
        "progress": progress,
    }
    out = _repo_path(output_root) / "local_runner_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-tsv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--record-id", action="append", default=None)
    parser.add_argument("--record-id-file", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--class-settings-manifest", default=CLASS_SETTINGS_LOCK)
    args = parser.parse_args(argv)
    ids = list(args.record_id or [])
    file_ids = _load_ids(args.record_id_file)
    if file_ids is not None:
        ids.extend(file_ids)
    summary = run_records(
        records_tsv=args.records_tsv,
        output_root=args.output_root,
        record_ids=ids or None,
        resume=not bool(args.no_resume),
        stop_on_failure=not bool(args.continue_on_failure),
        class_settings_manifest=str(args.class_settings_manifest),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["record_count_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
