#!/usr/bin/env bash
set -euo pipefail
RECORD_ID="${1:?record_id required}"
RECORDS_PATH="${2:?records path required}"
ROOT="$PWD"
OUTDIR="$ROOT/raw_outputs/$RECORD_ID"
LOGDIR="$ROOT/logs/$RECORD_ID"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR" "$LOGDIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "PRUNE COMPILE REPLAY START record_id=$RECORD_ID host=$(hostname) pwd=$PWD"
date -u +%Y-%m-%dT%H:%M:%SZ

set +e
python chtc/time_dynamics_optuna/run_prune_compile_replay_task.py \
  --record-id "$RECORD_ID" \
  --records "$RECORDS_PATH" \
  --output-root "$OUTDIR" \
  > "$OUTDIR/run.out" \
  2> "$OUTDIR/run.err"
RC=$?
set -e

RECORD_ID_FOR_STATUS="$RECORD_ID" RUN_RC="$RC" python - <<'PY'
from pathlib import Path
import json, os
record = os.environ.get("RECORD_ID_FOR_STATUS", "")
out = Path("raw_outputs") / record
task = out / "task_result.json"
status = {
    "record_id": record,
    "return_code": int(os.environ.get("RUN_RC", "0")),
    "task_result_exists": task.exists(),
}
if task.exists():
    try:
        status["task_result"] = json.loads(task.read_text())
    except Exception as exc:
        status["task_result_error"] = f"{type(exc).__name__}: {exc}"
(out / "chtc_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
print(json.dumps(status, indent=2, sort_keys=True))
PY

date -u +%Y-%m-%dT%H:%M:%SZ
echo "PRUNE COMPILE REPLAY DONE record_id=$RECORD_ID return_code=$RC"
exit "$RC"
