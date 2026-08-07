#!/usr/bin/env bash
set -euo pipefail
ARG_COUNT="$#"
RECORD_ID="${1:?record_id required}"
DEFAULT_RECORDS_PATH="chtc/time_dynamics_optuna/input/records.tsv"
SMOKE_RECORDS_PATH="chtc/time_dynamics_optuna/input/paper_ii_all_algorithm_class_calibration_v1_smoke_records.tsv"
RECORDS_PATH_SOURCE="default"
if [[ "$ARG_COUNT" -ge 2 && -n "${2:-}" ]]; then
  RECORDS_PATH="$2"
  RECORDS_PATH_SOURCE="arg2"
else
  RECORDS_PATH="$DEFAULT_RECORDS_PATH"
fi
ROOT="$PWD"
OUTDIR="$ROOT/raw_outputs/$RECORD_ID"
LOGDIR="$ROOT/logs/$RECORD_ID"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR" "$LOGDIR"

record_in_records_file() {
  local records_file="$1"
  [[ -f "$records_file" ]] && awk -F '\t' -v rid="$RECORD_ID" 'NR > 1 && $1 == rid {found=1} END {exit found ? 0 : 1}' "$records_file"
}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "JOB START record_id=$RECORD_ID host=$(hostname) pwd=$PWD"
date -u +%Y-%m-%dT%H:%M:%SZ
if [[ "$RECORD_ID" == paper_ii_all_algorithm_class_calibration_v1_smoke__* ]] && ! record_in_records_file "$RECORDS_PATH"; then
  if record_in_records_file "$SMOKE_RECORDS_PATH"; then
    echo "records_path_fallback record_id=$RECORD_ID from=$RECORDS_PATH to=$SMOKE_RECORDS_PATH"
    RECORDS_PATH="$SMOKE_RECORDS_PATH"
    RECORDS_PATH_SOURCE="paper_ii_all_algorithm_smoke_fallback"
  fi
fi
echo "JOB ARGS argc=$ARG_COUNT records_path=$RECORDS_PATH records_path_source=$RECORDS_PATH_SOURCE"

set +e
python chtc/time_dynamics_optuna/run_task.py \
  --record-id "$RECORD_ID" \
  --records "$RECORDS_PATH" \
  --output-root "$OUTDIR" \
  > "$OUTDIR/run.out" \
  2> "$OUTDIR/run.err"
RC=$?
set -e

RECORD_ID_FOR_STATUS="$RECORD_ID" RUN_RC="$RC" RECORDS_PATH_FOR_STATUS="$RECORDS_PATH" ARG_COUNT_FOR_STATUS="$ARG_COUNT" RECORDS_PATH_SOURCE_FOR_STATUS="$RECORDS_PATH_SOURCE" python - <<'__STATUS_PY__'
from pathlib import Path
import json, os
record = os.environ["RECORD_ID_FOR_STATUS"]
rc = int(os.environ["RUN_RC"])
try:
    arg_count = int(os.environ.get("ARG_COUNT_FOR_STATUS", "-1"))
except ValueError:
    arg_count = -1
out = Path("raw_outputs") / record
summary = out / "summary.json"
progress = out / "run" / "progress.json"
task_result = out / "task_result.json"
record_json = out / "record.json"
validation_profile = None
if record_json.exists():
    try:
        validation_profile = json.loads(record_json.read_text()).get("validation_profile")
    except Exception:
        validation_profile = None
status = {
    "record_id": record,
    "return_code": rc,
    "arg_count": arg_count,
    "records_path": os.environ.get("RECORDS_PATH_FOR_STATUS", ""),
    "records_path_source": os.environ.get("RECORDS_PATH_SOURCE_FOR_STATUS", ""),
    "summary_exists": summary.exists(),
    "progress_exists": progress.exists(),
    "task_result_exists": task_result.exists(),
    "validation_profile": validation_profile,
    "run_stdout": str(out / "run.out"),
    "run_stderr": str(out / "run.err"),
}
if summary.exists():
    try:
        data = json.loads(summary.read_text())
        status["summary_keys"] = sorted(data.keys())[:40]
    except Exception as exc:
        status["summary_error"] = f"{type(exc).__name__}: {exc}"
(out / "chtc_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
print(json.dumps(status, indent=2, sort_keys=True))
__STATUS_PY__

date -u +%Y-%m-%dT%H:%M:%SZ
echo "JOB DONE record_id=$RECORD_ID return_code=$RC"
exit "$RC"
