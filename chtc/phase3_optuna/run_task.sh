#!/usr/bin/env bash
set -euo pipefail
RECORD_ID="${1:?record_id required}"
ROOT="$PWD"
OUTDIR="$ROOT/raw_outputs/$RECORD_ID"
LOGDIR="$ROOT/logs/$RECORD_ID"
PROGRESS_DIR="$OUTDIR/progress"
mkdir -p "$OUTDIR" "$LOGDIR" "$PROGRESS_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

append_wrapper_event() {
  local event="$1"
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"schema":"phase3_shell_wrapper_event_v1","timestamp_utc":"%s","record_id":"%s","event":"%s"}\n' "$ts" "$RECORD_ID" "$event" >> "$PROGRESS_DIR/wrapper_events.jsonl" || true
}

write_shell_status() {
  local state="$1"
  local code="$2"
  python - "$OUTDIR" "$RECORD_ID" "$state" "$code" <<'PY' || true
from __future__ import annotations
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
out = Path(sys.argv[1])
record_id = sys.argv[2]
state = sys.argv[3]
code = int(sys.argv[4])
payload = {
    "schema": "phase3_shell_wrapper_status_v1",
    "record_id": record_id,
    "state": state,
    "returncode": code,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "pid": os.getpid(),
    "summary_exists": (out / "summary.json").exists(),
    "heartbeat_exists": (out / "heartbeat.json").exists(),
}
(out / "progress").mkdir(parents=True, exist_ok=True)
(out / "progress" / "shell_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

emit_progress_summary() {
  python - "$OUTDIR" "$RECORD_ID" <<'PY' || true
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
record_id = sys.argv[2]

def read_json_path(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}

def latest_progress_json(filename: str) -> tuple[dict, str | None]:
    progress_dir = out / "progress"
    candidates = [progress_dir / filename]
    try:
        nested = list(progress_dir.glob(f"*/{filename}"))
    except Exception:
        nested = []
    nested.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)
    candidates.extend(nested)
    latest: tuple[dict, str | None] = ({}, None)
    latest_mtime = -1.0
    for path in candidates:
        payload = read_json_path(path)
        if not payload:
            continue
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0.0
        if mtime >= latest_mtime:
            latest = (payload, str(path.relative_to(out)))
            latest_mtime = mtime
    return latest

current, current_path = latest_progress_json("current.json")
current_best, current_best_path = latest_progress_json("current_best.json")
heartbeat = read_json_path(out / "heartbeat.json")
child = current.get("last_child_heartbeat")
if not isinstance(child, dict):
    child = {}
progress = child.get("progress")
if not isinstance(progress, dict):
    progress = {}
freshness = heartbeat.get("progress_freshness")
if not isinstance(freshness, dict):
    freshness = {}
best_attrs = current_best.get("best_user_attrs")
if not isinstance(best_attrs, dict):
    best_attrs = {}

def first_present(*values):
    for value in values:
        if value is not None and value != "":
            return value
    return None

if not current and not current_best and not heartbeat:
    raise SystemExit(0)

payload = {
    "schema": "phase3_condor_tail_progress_v1",
    "record_id": record_id,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "state": current.get("state") or heartbeat.get("state"),
    "trial_number": current.get("trial_number") or current.get("child_trial_number"),
    "benchmark_id": current.get("benchmark_id") or current.get("child_benchmark_id"),
    "current_path": current_path,
    "current_best_path": current_best_path,
    "active_child_count": current.get("active_child_count") or heartbeat.get("active_child_count"),
    "child_event": child.get("last_ai_log_event"),
    "depth": progress.get("depth"),
    "energy": progress.get("energy"),
    "delta_abs_current": progress.get("delta_abs_current"),
    "delta_e": first_present(current.get("delta_e"), current.get("primary_error"), progress.get("delta_abs_current")),
    "primary_error": current.get("primary_error"),
    "first_crossing": current.get("first_crossing"),
    "max_grad": progress.get("max_grad"),
    "stop_reason_so_far": progress.get("stop_reason_so_far"),
    "best_trial_number": current_best.get("best_trial_number") or current_best.get("trial_number"),
    "best_value": current_best.get("best_value") or current_best.get("value"),
    "best_delta_e": first_present(
        current_best.get("best_delta_e"),
        current_best.get("delta_e"),
        current_best.get("best_primary_error"),
        best_attrs.get("telemetry_primary_error"),
    ),
    "best_primary_error": first_present(
        current_best.get("best_primary_error"),
        current_best.get("primary_error"),
        best_attrs.get("telemetry_primary_error"),
    ),
    "best_first_crossing": first_present(
        current_best.get("best_first_crossing"),
        current_best.get("first_crossing"),
        best_attrs.get("telemetry_first_crossing"),
    ),
    "progress_age_s": freshness.get("progress_age_s"),
    "progress_stale": freshness.get("progress_stale"),
    "last_progress_source": freshness.get("last_progress_source"),
}
print("PROGRESS " + json.dumps(payload, sort_keys=True, default=str), flush=True)
PY
}

echo "JOB START record_id=$RECORD_ID host=$(hostname) pwd=$PWD"
date -u +%Y-%m-%dT%H:%M:%SZ
append_wrapper_event "shell_start"
write_shell_status "starting" 0

RECORDS_PATH="${PHASE3_RECORDS_PATH:-chtc/phase3_optuna/input/records.tsv}"
if ! awk -F '\t' -v rid="$RECORD_ID" 'NR > 1 && $1 == rid {found=1} END {exit found ? 0 : 1}' "$RECORDS_PATH" 2>/dev/null; then
  while IFS= read -r candidate; do
    if awk -F '\t' -v rid="$RECORD_ID" 'NR > 1 && $1 == rid {found=1} END {exit found ? 0 : 1}' "$candidate"; then
      RECORDS_PATH="$candidate"
      break
    fi
  done < <(find chtc/phase3_optuna/input -type f -name '*records.tsv' -print 2>/dev/null | LC_ALL=C sort)
fi
if ! awk -F '\t' -v rid="$RECORD_ID" 'NR > 1 && $1 == rid {found=1} END {exit found ? 0 : 1}' "$RECORDS_PATH"; then
  echo "record_id=$RECORD_ID not found in $RECORDS_PATH or alternate input/**/*records.tsv files" >&2
  append_wrapper_event "record_lookup_failed"
  write_shell_status "record_lookup_failed" 2
  exit 2
fi
echo "records_path=$RECORDS_PATH"

python - "$OUTDIR" "$RECORD_ID" "$RECORDS_PATH" <<'PY'
from __future__ import annotations
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
out = Path(sys.argv[1])
payload = {
    "schema": "phase3_shell_wrapper_context_v1",
    "record_id": sys.argv[2],
    "records_path": sys.argv[3],
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "pwd": os.getcwd(),
    "pythonpath": os.environ.get("PYTHONPATH"),
    "thread_env": {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
    },
}
out.mkdir(parents=True, exist_ok=True)
(out / "wrapper_context.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

PY_PID=""
HEARTBEAT_PID=""
START_EPOCH="$(date +%s)"
stop_shell_heartbeat() {
  if [[ -n "${HEARTBEAT_PID:-}" ]]; then
    kill "$HEARTBEAT_PID" 2>/dev/null || true
    wait "$HEARTBEAT_PID" 2>/dev/null || true
    HEARTBEAT_PID=""
  fi
}
start_shell_heartbeat() {
  (
    while true; do
      local_now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      local_epoch="$(date +%s)"
      local_elapsed=$((local_epoch - START_EPOCH))
      local_py_state="not_started"
      if [[ -n "${PY_PID:-}" ]] && kill -0 "$PY_PID" 2>/dev/null; then
        local_py_state="running"
      elif [[ -n "${PY_PID:-}" ]]; then
        local_py_state="exited"
      fi
      echo "HEARTBEAT record_id=$RECORD_ID timestamp_utc=$local_now elapsed_sec=$local_elapsed python_pid=${PY_PID:-} python_state=$local_py_state"
      emit_progress_summary
      sleep "${PHASE3_SHELL_HEARTBEAT_SEC:-60}"
    done
  ) &
  HEARTBEAT_PID=$!
}
on_signal() {
  local sig="$1"
  local code=$((128 + sig))
  append_wrapper_event "shell_signal_${sig}"
  stop_shell_heartbeat
  if [[ -n "${PY_PID:-}" ]]; then
    kill -TERM "$PY_PID" 2>/dev/null || true
    wait "$PY_PID" 2>/dev/null || true
  fi
  write_shell_status "terminated" "$code"
  exit "$code"
}
trap 'on_signal 15' TERM
trap 'on_signal 2' INT

append_wrapper_event "python_wrapper_starting"
set +e
python chtc/phase3_optuna/run_task.py \
  --record-id "$RECORD_ID" \
  --records "$RECORDS_PATH" \
  --output-root "$OUTDIR" \
  > "$OUTDIR/run.out" \
  2> "$OUTDIR/run.err" &
PY_PID=$!
start_shell_heartbeat
wait "$PY_PID"
STATUS=$?
stop_shell_heartbeat
set -e
trap - TERM INT
append_wrapper_event "python_wrapper_exited"
if [[ "$STATUS" -eq 0 ]]; then
  SHELL_STATE="completed"
elif [[ "$STATUS" -ge 128 ]]; then
  SHELL_STATE="terminated"
else
  SHELL_STATE="failed"
fi
write_shell_status "$SHELL_STATE" "$STATUS"

date -u +%Y-%m-%dT%H:%M:%SZ
echo "JOB DONE record_id=$RECORD_ID status=$STATUS"
exit "$STATUS"
