#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.paper_iii_excited_dynamics.preflight_inputs import (  # noqa: E402
    DEFAULT_RECORDS,
    MODE_COMPATIBILITY_AUDIT_ONLY,
    MODE_REPORT_ONLY_EXISTING_OUTPUT,
    MODE_STRICT_HH,
    find_record,
    load_records,
    parse_bool,
    repo_path,
)

PIPELINE_MODULE = "pipelines.excited_dynamics.paper_iii_local_science_pilot"
SCHEMA_VERSION = "paper_iii_excited_dynamics_task_result_v1"
DEFAULT_OUTPUT_ROOT = Path("raw_outputs") / "paper_iii_excited_dynamics"


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _as_float_text(value: Any, default: str) -> str:
    text = _clean(value)
    if not text:
        return default
    return str(float(text))


def _as_int_text(value: Any, default: str) -> str:
    text = _clean(value)
    if not text:
        return default
    return str(int(text))


def build_pilot_command(
    row: Mapping[str, str],
    *,
    record_id: str | None = None,
    output_root: str | Path,
    repo_root: str | Path = REPO_ROOT,
) -> tuple[list[str], dict[str, Any]]:
    rid = record_id or _clean(row.get("record_id"))
    repo = Path(repo_root).resolve()
    mode = _clean(row.get("mode"))
    if mode == MODE_COMPATIBILITY_AUDIT_ONLY:
        raise ValueError(
            f"record {rid}: compatibility_audit_only records are audit metadata and are not runnable by run_task; "
            "no local science pilot or dynamics launched"
        )
    if mode not in {MODE_REPORT_ONLY_EXISTING_OUTPUT, MODE_STRICT_HH}:
        raise ValueError(f"record {rid}: unsupported mode {mode!r}")
    artifact_raw = _clean(row.get("artifact_json"))
    if not artifact_raw:
        raise ValueError(f"record {rid}: missing artifact_json")
    output_dir = Path(output_root) / "paper_iii_local_science_pilot"
    command = [
        sys.executable,
        "-u",
        "-m",
        PIPELINE_MODULE,
        "--artifact-json",
        str(repo_path(artifact_raw, repo_root=repo)),
        "--output-dir",
        str(output_dir),
        "--t-final",
        _as_float_text(row.get("t_final"), "1.0"),
        "--num-times",
        _as_int_text(row.get("num_times"), "9"),
        "--timeout-seconds",
        _as_int_text(row.get("timeout_seconds"), "1800"),
        "--run-tag",
        _clean(row.get("run_tag")) or rid,
    ]
    progress_json: Path | None = None
    partial_payload_json: Path | None = None
    existing_strict_output_json: Path | None = None
    if mode == MODE_REPORT_ONLY_EXISTING_OUTPUT:
        existing_raw = _clean(row.get("existing_strict_output_json"))
        if not existing_raw:
            raise ValueError(f"record {rid}: report_only_existing_output requires existing_strict_output_json")
        existing_strict_output_json = repo_path(existing_raw, repo_root=repo)
        command.extend([
            "--report-only-existing-output",
            "--existing-strict-output-json",
            str(existing_strict_output_json),
        ])
    elif mode == MODE_STRICT_HH:
        if not parse_bool(row.get("require_progress_json"), default=True):
            raise ValueError(f"record {rid}: strict_hh requires require_progress_json=true")
        if not parse_bool(row.get("require_partial_payload_json"), default=True):
            raise ValueError(f"record {rid}: strict_hh requires require_partial_payload_json=true")
        progress_json = Path(output_root) / "progress.json"
        partial_payload_json = Path(output_root) / "partial_payload.json"
        command.extend(["--progress-json", str(progress_json)])
        command.extend(["--partial-payload-json", str(partial_payload_json)])
    metadata = {
        "record_id": rid,
        "mode": mode,
        "pilot_output_dir": str(output_dir),
        "report_json": str(output_dir / "paper_iii_local_science_pilot_report.json"),
        "run_manifest_json": str(output_dir / "run_manifest.json"),
        "strict_output_json": str(output_dir / "hh_strict_realtime_pilot.json"),
        "progress_json": None if progress_json is None else str(progress_json),
        "partial_payload_json": None if partial_payload_json is None else str(partial_payload_json),
        "existing_strict_output_json": None if existing_strict_output_json is None else str(existing_strict_output_json),
    }
    return command, metadata


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_record(record_id: str, records_path: str | Path, output_root: str | Path) -> int:
    started = _now_utc()
    records_file = Path(records_path)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    command: list[str] = []
    metadata: dict[str, Any] = {}
    error: str | None = None
    rc = 1
    row: dict[str, str] | None = None
    try:
        row = find_record(load_records(records_file), record_id)
        _write_json(out_root / "record.json", row)
        command, metadata = build_pilot_command(row, record_id=record_id, output_root=out_root)
        (out_root / "command.sh").write_text(shlex.join([str(part) for part in command]) + "\n", encoding="utf-8")
        print("RUN", shlex.join([str(part) for part in command]), flush=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
        rc = int(completed.returncode)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(error, file=sys.stderr, flush=True)
        rc = 2
    finished = _now_utc()
    existing_output_copy: Path | None = None
    existing_source_value = metadata.get("existing_strict_output_json")
    if existing_source_value:
        existing_source = Path(str(existing_source_value))
        if existing_source.exists():
            existing_output_copy = out_root / "existing_strict_output.json"
            shutil.copy2(existing_source, existing_output_copy)

    report_json = Path(metadata.get("report_json", out_root / "paper_iii_local_science_pilot" / "paper_iii_local_science_pilot_report.json"))
    run_manifest_json = Path(metadata.get("run_manifest_json", out_root / "paper_iii_local_science_pilot" / "run_manifest.json"))
    progress_json = metadata.get("progress_json")
    partial_payload_json = metadata.get("partial_payload_json")
    result = {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "return_code": rc,
        "mode": None if row is None else _clean(row.get("mode")),
        "command": command,
        "records_path": str(records_file),
        "output_root": str(out_root),
        "pilot_output_dir": metadata.get("pilot_output_dir"),
        "report_json": str(report_json),
        "report_exists": report_json.exists(),
        "run_manifest_json": str(run_manifest_json),
        "run_manifest_exists": run_manifest_json.exists(),
        "strict_output_json": metadata.get("strict_output_json"),
        "strict_output_exists": bool(metadata.get("strict_output_json") and Path(str(metadata["strict_output_json"])).exists()),
        "source_existing_strict_output_json": metadata.get("existing_strict_output_json"),
        "existing_strict_output_json": None if existing_output_copy is None else str(existing_output_copy),
        "existing_strict_output_exists": bool(existing_output_copy is not None and existing_output_copy.exists()),
        "source_existing_strict_output_exists": bool(
            metadata.get("existing_strict_output_json") and Path(str(metadata["existing_strict_output_json"])).exists()
        ),
        "progress_json": progress_json,
        "progress_exists": bool(progress_json and Path(str(progress_json)).exists()),
        "partial_payload_json": partial_payload_json,
        "partial_payload_exists": bool(partial_payload_json and Path(str(partial_payload_json)).exists()),
        "started_utc": started,
        "finished_utc": finished,
    }
    if error is not None:
        result["error"] = error
    _write_json(out_root / "task_result.json", result)
    return rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Paper III excited-dynamics CHTC readiness record.")
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return run_record(args.record_id, args.records, args.output_root)


__all__ = ["DEFAULT_OUTPUT_ROOT", "PIPELINE_MODULE", "build_pilot_command", "run_record"]


if __name__ == "__main__":
    raise SystemExit(main())
