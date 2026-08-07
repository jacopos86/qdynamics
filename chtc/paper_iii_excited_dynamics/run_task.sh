#!/usr/bin/env bash
set -euo pipefail
RECORD_ID="${1:?record_id required}"
if [[ ! "$RECORD_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid record_id '$RECORD_ID': expected only A-Z, a-z, 0-9, '_', '.', '-'" >&2
  exit 64
fi
ROOT="$PWD"
RECORDS_PATH="${2:-${PAPER_III_EXCITED_DYNAMICS_RECORDS_PATH:-chtc/paper_iii_excited_dynamics/input/records.tsv}}"
OUT_BASE="${PAPER_III_EXCITED_DYNAMICS_OUTPUT_ROOT:-raw_outputs/paper_iii_excited_dynamics}"
OUTDIR="$OUT_BASE/$RECORD_ID"
LOGDIR="$ROOT/logs/$RECORD_ID"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR" "$LOGDIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "JOB START record_id=$RECORD_ID host=$(hostname) pwd=$PWD"
date -u +%Y-%m-%dT%H:%M:%SZ

set +e
python chtc/paper_iii_excited_dynamics/run_task.py \
  --record-id "$RECORD_ID" \
  --records "$RECORDS_PATH" \
  --output-root "$OUTDIR" \
  > "$OUTDIR/run.out" \
  2> "$OUTDIR/run.err"
RC=$?
set -e

cp "$OUTDIR/run.out" "$LOGDIR/run.out"
cp "$OUTDIR/run.err" "$LOGDIR/run.err"

RECORD_ID_FOR_STATUS="$RECORD_ID" RUN_RC="$RC" OUTDIR_FOR_STATUS="$OUTDIR" LOGDIR_FOR_STATUS="$LOGDIR" python - <<'__STATUS_PY__'
from pathlib import Path
import json
import os

record = os.environ["RECORD_ID_FOR_STATUS"]
rc = int(os.environ["RUN_RC"])
out = Path(os.environ["OUTDIR_FOR_STATUS"])
logdir = Path(os.environ["LOGDIR_FOR_STATUS"])
task_result = out / "task_result.json"
record_json = out / "record.json"
report_json = out / "paper_iii_local_science_pilot" / "paper_iii_local_science_pilot_report.json"
run_manifest = out / "paper_iii_local_science_pilot" / "run_manifest.json"
mode = None
if record_json.exists():
    try:
        mode = json.loads(record_json.read_text(encoding="utf-8")).get("mode")
    except Exception:
        mode = None
status = {
    "schema_version": "paper_iii_excited_dynamics_chtc_status_v1",
    "record_id": record,
    "mode": mode,
    "return_code": rc,
    "task_result_exists": task_result.exists(),
    "record_json_exists": record_json.exists(),
    "report_json_exists": report_json.exists(),
    "run_manifest_exists": run_manifest.exists(),
    "run_stdout": str(out / "run.out"),
    "run_stderr": str(out / "run.err"),
    "log_stdout": str(logdir / "run.out"),
    "log_stderr": str(logdir / "run.err"),
}
if task_result.exists():
    try:
        data = json.loads(task_result.read_text(encoding="utf-8"))
        status["task_result_keys"] = sorted(data.keys())[:60]
        status["task_report_exists"] = data.get("report_exists")
        status["task_run_manifest_exists"] = data.get("run_manifest_exists")
        status["task_progress_exists"] = data.get("progress_exists")
        status["task_partial_payload_exists"] = data.get("partial_payload_exists")
    except Exception as exc:
        status["task_result_error"] = f"{type(exc).__name__}: {exc}"
if report_json.exists():
    try:
        report = json.loads(report_json.read_text(encoding="utf-8"))
        status["paper_iii_science_benchmark"] = report.get("paper_iii_science_benchmark")
        status["blocker_count"] = len(report.get("blockers") or []) if isinstance(report.get("blockers") or [], list) else None
        status["strict_validation_passed"] = (
            report.get("runs", {})
            .get("strict_hh_runtime_dynamics", {})
            .get("strict_validation", {})
            .get("passed")
        )
    except Exception as exc:
        status["report_error"] = f"{type(exc).__name__}: {exc}"
(out / "chtc_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(status, indent=2, sort_keys=True))
__STATUS_PY__

date -u +%Y-%m-%dT%H:%M:%SZ
echo "JOB DONE record_id=$RECORD_ID return_code=$RC"
exit "$RC"
