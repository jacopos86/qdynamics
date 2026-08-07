#!/usr/bin/env bash
set -euo pipefail

RECORD_ID="${1:-paper_iv_h2o_resume_depth8_to15_wide_vibronic_powell_maxfev20000}"
ROOT="$PWD"
OUTDIR="$ROOT/raw_outputs/$RECORD_ID"
LOGDIR="$ROOT/logs/$RECORD_ID"
PROGRESS_DIR="$OUTDIR/progress"
RESULT_JSON="$OUTDIR/result.json"
CURRENT_JSON="$OUTDIR/current.json"
STDOUT_LOG="$OUTDIR/stdout.log"
STDERR_LOG="$OUTDIR/stderr.log"
COMMAND_TXT="$OUTDIR/command.txt"
MANIFEST_JSON="$OUTDIR/submit_manifest.json"
SHELL_STATUS_JSON="$PROGRESS_DIR/shell_status.json"

DEFAULT_RESUME_SCAFFOLD_JSON="tmp/h2o_linear_fd_valence_psi4_optimized/adapt_paper_i_current_powell_anchor_d15/h2o_depth6_checkpoint_for_chtc_20260705T001649Z.json"
RESUME_SCAFFOLD_JSON="${H2O_RESUME_SCAFFOLD_JSON:-$DEFAULT_RESUME_SCAFFOLD_JSON}"
RESUME_DEPTH="${H2O_RESUME_DEPTH:-8}"
ADAPT_MAX_DEPTH="${H2O_ADAPT_MAX_DEPTH:-15}"
ADAPT_SEGMENT_ID="${H2O_ADAPT_SEGMENT_ID:-paper_iv_h2o_depth8_to15_wide_vibronic_powell_maxfev20000}"
ADAPT_SEGMENT_TARGET_DEPTH="${H2O_ADAPT_SEGMENT_TARGET_DEPTH:-15}"
ADAPT_SEGMENT_MAX_NEW_ADMISSIONS="${H2O_ADAPT_SEGMENT_MAX_NEW_ADMISSIONS:-7}"
PHASE2_ENABLE_BATCHING="${H2O_PHASE2_ENABLE_BATCHING:-1}"
SHORTLIST_CHANGE_REASON="${H2O_SHORTLIST_CHANGE_REASON:-H2O vibronic diagnostic found best vibronic candidates at ranks 25-26; widen shortlists from 24/12 to 48/36.}"

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
  printf '{"schema":"paper_iv_h2o_shell_event_v1","timestamp_utc":"%s","record_id":"%s","event":"%s"}\n' "$ts" "$RECORD_ID" "$event" >> "$PROGRESS_DIR/wrapper_events.jsonl" || true
}

write_shell_status() {
  local state="$1"
  local code="$2"
  python3 - "$SHELL_STATUS_JSON" "$RECORD_ID" "$state" "$code" <<'PY' || true
from __future__ import annotations
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
record_id = sys.argv[2]
state = sys.argv[3]
code = int(sys.argv[4])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    json.dumps(
        {
            "schema": "paper_iv_h2o_shell_status_v1",
            "record_id": record_id,
            "state": state,
            "returncode": code,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

emit_heartbeat() {
  python3 - "$OUTDIR" "$RECORD_ID" <<'PY' || true
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
record_id = sys.argv[2]
current_path = out / "current.json"
payload = {
    "schema": "paper_iv_h2o_condor_progress_v1",
    "record_id": record_id,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "current_json_exists": current_path.exists(),
}
if current_path.exists():
    try:
        current = json.loads(current_path.read_text(encoding="utf-8"))
    except Exception as exc:
        payload["current_json_read_error"] = str(exc)
    else:
        adapt = current.get("adapt_vqe", {}) if isinstance(current, dict) else {}
        progress = adapt.get("progress", {}) if isinstance(adapt, dict) else {}
        ground = current.get("ground_state", {}) if isinstance(current, dict) else {}
        payload.update(
            {
                "ansatz_depth": adapt.get("ansatz_depth") or progress.get("depth"),
                "energy": ground.get("energy") or progress.get("energy"),
                "abs_delta_e": ground.get("abs_delta_e") or progress.get("delta_abs_current"),
                "stop_reason_so_far": progress.get("stop_reason_so_far"),
            }
        )
print("H2O_PROGRESS " + json.dumps(payload, sort_keys=True, default=str), flush=True)
PY
}

CMD=(
  python3
  -u
  -m
  pipelines.static_adapt.adapt_pipeline
  --problem
  molecular_vibronic_h2o_linear_fd
  --molecular-vibronic-h2o-linear-fd-fixture-json
  tmp/h2o_linear_fd_valence_psi4_optimized/h2o_linear_fd_sparse_fixture_nph1_ref2.json
  --n-ph-max
  1
  --term-order
  sorted
  --ordering
  blocked
  --boson-encoding
  binary
  --boundary
  open
  --adapt-pool
  full_meta
  --static-route-id
  route_a
  --adapt-inner-optimizer
  POWELL
  --adapt-maxiter
  300
  --adapt-scipy-maxfev
  20000
  --adapt-max-depth
  "$ADAPT_MAX_DEPTH"
  --adapt-resume-scaffold-json
  "$RESUME_SCAFFOLD_JSON"
  --adapt-segment-id
  "$ADAPT_SEGMENT_ID"
  --adapt-segment-target-depth
  "$ADAPT_SEGMENT_TARGET_DEPTH"
  --adapt-segment-max-new-admissions
  "$ADAPT_SEGMENT_MAX_NEW_ADMISSIONS"
  --adapt-final-full-refit
  true
  --adapt-final-refit-maxiter
  200
  --adapt-full-refit-every
  8
  --adapt-reopt-policy
  windowed
  --adapt-window-size
  50
  --adapt-window-topk
  50
  --adapt-eps-grad
  5e-7
  --adapt-eps-energy
  1e-9
  --adapt-schur-warm-start-mode
  append-prune
  --adapt-seed
  7
  --adapt-state-backend
  compiled
  --phase1-shortlist-size
  48
  --phase1-lambda-F
  1.0
  --phase1-lambda-compile
  0.05
  --phase1-lambda-measure
  0.02
  --phase1-lambda-leak
  0.0
  --phase1-score-mode
  trust_region_v1
  --phase1-probe-max-positions
  999999
  --phase1-prune-enabled
  --phase1-prune-policy
  recoverability_ladder_v1
  --phase1-prune-mode
  both
  --phase1-prune-fraction
  0.4
  --phase1-prune-max-candidates
  6
  --phase1-prune-checkpoint-period
  3
  --phase2-shortlist-size
  36
  --phase2-shortlist-fraction
  0.5
  --phase2-lambda-H
  1e-6
  --phase2-rho
  0.5
  --phase2-gamma-N
  1.0
  --phase2-novelty-mode
  collective_span_v1
  --phase2-selector-gain-mode
  trust_region_v1
  --phase2-batch-target-size
  8
  --phase2-batch-size-cap
  16
  --phase2-batch-near-degenerate-ratio
  0.98
  --phase2-batch-rank-rel-tol
  0.25
  --phase2-batch-additivity-tol
  0.25
  --phase2-w-depth
  0.2
  --phase2-w-group
  0.15
  --phase2-w-shot
  0.05
  --phase2-w-optdim
  0.1
  --phase2-w-reuse
  0.1
  --phase2-w-lifetime
  0.05
  --phase2-eta-L
  0.0
  --phase2-motif-bonus-weight
  0.05
  --phase3-frontier-ratio
  0.9
  --phase3-tie-beam-max-branches
  1
  --phase3-selector-policy
  algebraic_nested_v1
  --allow-archival-phase3-runtime-split
  --phase3-runtime-split-mode
  shortlist_pauli_children_v1
  --phase3-runtime-split-selection-mode
  archival_child_set_forward_v1
  --phase3-runtime-split-max-subset-size
  1
  --phase3-runtime-split-child-set-symmetry-policy
  hard_guard
  --phase3-batch-selection-mode
  reduced_plane
  --phase3-batch-order-selection-mode
  finite_step_v1
  --phase3-batch-prefilter-mode
  off
  --phase3-selector-geometry-mode
  reduced
  --phase3-enable-rescue
  --phase3-lifetime-cost-mode
  off
  --phase3-backend-cost-mode
  auto
  --phase3-backend-name
  FakeMarrakesh
  --phase3-backend-transpile-seed
  7
  --phase3-backend-optimization-level
  1
  --phase3-oracle-inner-objective-mode
  exact
  --hardware-resolution-mode
  ideal
  --shared-pauli-pool-mode
  off
  --adapt-child-pool-expansion-mode
  off
  --output-json
  "$RESULT_JSON"
  --adapt-current-json
  "$CURRENT_JSON"
  --adapt-current-json-every-depth
  1
)

case "${PHASE2_ENABLE_BATCHING,,}" in
  1|true|yes|on)
    CMD+=(--phase2-enable-batching)
    ;;
  0|false|no|off)
    CMD+=(--phase2-no-batching)
    ;;
  *)
    echo "Invalid H2O_PHASE2_ENABLE_BATCHING=$PHASE2_ENABLE_BATCHING" >&2
    exit 2
    ;;
esac

printf '%q ' "${CMD[@]}" > "$COMMAND_TXT"
printf '\n' >> "$COMMAND_TXT"

python3 - "$MANIFEST_JSON" "$RECORD_ID" "$RESULT_JSON" "$CURRENT_JSON" \
  "$RESUME_SCAFFOLD_JSON" "$RESUME_DEPTH" "$ADAPT_SEGMENT_TARGET_DEPTH" \
  "$ADAPT_SEGMENT_MAX_NEW_ADMISSIONS" "$ADAPT_MAX_DEPTH" "$ADAPT_SEGMENT_ID" \
  "$SHORTLIST_CHANGE_REASON" "$PHASE2_ENABLE_BATCHING" <<'PY'
from __future__ import annotations
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
fixture = Path("tmp/h2o_linear_fd_valence_psi4_optimized/h2o_linear_fd_sparse_fixture_nph1_ref2.json")
resume_scaffold = Path(sys.argv[5])

def sha256_file(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

payload = {
    "schema": "paper_iv_h2o_static_snake_chtc_submit_manifest_v1",
    "record_id": sys.argv[2],
    "run_class": "candidate_diagnostic_continuation",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "cwd": os.getcwd(),
    "problem": "molecular_vibronic_h2o_linear_fd",
    "method": "SNAKE / static ADAPT Route A",
    "resume_checkpoint": str(resume_scaffold),
    "resume_checkpoint_sha256": sha256_file(resume_scaffold),
    "fixture_json": str(fixture),
    "fixture_sha256": sha256_file(fixture),
    "resume_depth": int(sys.argv[6]),
    "target_depth": int(sys.argv[7]),
    "max_new_admissions": int(sys.argv[8]),
    "adapt_max_depth": int(sys.argv[9]),
    "adapt_segment_id": str(sys.argv[10]),
    "phase2_enable_batching": str(sys.argv[12]).strip().lower() in {"1", "true", "yes", "on"},
    "optimizer": "POWELL",
    "adapt_maxiter": 300,
    "adapt_scipy_maxfev": 20000,
    "shortlist_change_reason": str(sys.argv[11]),
    "phase1_shortlist_size": 48,
    "phase2_shortlist_size": 36,
    "phase2_shortlist_fraction": 0.5,
    "result_json": sys.argv[3],
    "current_json": sys.argv[4],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "JOB START record_id=$RECORD_ID host=$(hostname) pwd=$PWD"
date -u +%Y-%m-%dT%H:%M:%SZ
append_wrapper_event "shell_start"
write_shell_status "starting" 0

HEARTBEAT_PID=""
START_EPOCH="$(date +%s)"
stop_heartbeat() {
  if [[ -n "${HEARTBEAT_PID:-}" ]]; then
    kill "$HEARTBEAT_PID" 2>/dev/null || true
    wait "$HEARTBEAT_PID" 2>/dev/null || true
    HEARTBEAT_PID=""
  fi
}

start_heartbeat() {
  (
    while true; do
      local_now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      local_epoch="$(date +%s)"
      local_elapsed=$((local_epoch - START_EPOCH))
      echo "HEARTBEAT record_id=$RECORD_ID timestamp_utc=$local_now elapsed_sec=$local_elapsed"
      emit_heartbeat
      sleep "${H2O_SHELL_HEARTBEAT_SEC:-300}"
    done
  ) &
  HEARTBEAT_PID=$!
}

on_signal() {
  local sig="$1"
  local code=$((128 + sig))
  append_wrapper_event "shell_signal_${sig}"
  stop_heartbeat
  write_shell_status "terminated" "$code"
  exit "$code"
}
trap 'on_signal 15' TERM
trap 'on_signal 2' INT

append_wrapper_event "python_starting"
set +e
start_heartbeat
"${CMD[@]}" > >(tee "$STDOUT_LOG") 2> >(tee "$STDERR_LOG" >&2)
STATUS=$?
stop_heartbeat
set -e
trap - TERM INT
append_wrapper_event "python_exited"

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
