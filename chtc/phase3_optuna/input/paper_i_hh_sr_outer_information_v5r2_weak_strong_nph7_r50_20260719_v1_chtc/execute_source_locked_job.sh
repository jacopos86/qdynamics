#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?mode required}"
JOB_MANIFEST="${2:?job manifest required}"
SOURCE_ARCHIVE="${3:?source archive required}"
EXPECTED_SOURCE_SHA256="${4:?source SHA-256 required}"
SOURCE_AUDIT="${5:?source audit required}"
EXPECTED_AUDIT_SHA256="${6:?source-audit SHA-256 required}"
IMAGE="${7:?image required}"
EXPECTED_IMAGE_SHA256="${8:?image SHA-256 required}"

BUNDLE_ID="paper_i_hh_sr_outer_information_v5r2_weak_strong_nph7_r50_20260719_v1_chtc"
RUNTIME_ROOT="extracted_runtime_predictive_prephase3_metric_hessian_reuse_v5_20260719"
BUNDLE_PATH="chtc/phase3_optuna/input/${BUNDLE_ID}"
OUTPUT_ROOT="raw_outputs/${BUNDLE_ID}/${MODE}"
TRANSFER_ARCHIVE="raw_outputs/${BUNDLE_ID}/${MODE}_transfer.tar.gz"
CHECKPOINT_ARCHIVE="raw_outputs/${BUNDLE_ID}/${MODE}_checkpoint.tar.gz"
RESUME_DIR="raw_outputs/${BUNDLE_ID}/resume_input/${MODE}"
RUNTIME_SOURCE="$PWD/runtime_source/${RUNTIME_ROOT}"
CHECKPOINT_EXIT_CODE=85
SIGNAL_STATUS=0
CHILD_PID=""
RESUME_CURRENT=""

[[ "$MODE" == "control" || "$MODE" == "reuse" ]] || {
  echo "mode must be control or reuse" >&2
  exit 2
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

write_exit_receipt() {
  local status="$1"
  mkdir -p "$OUTPUT_ROOT"
  python3 -c 'import json,os,sys,tempfile; p=sys.argv[1]; d={"schema":"paper_i_sr_outer_information_wrapper_exit_receipt_v1","mode":sys.argv[2],"exit_code":int(sys.argv[3]),"signal_status":int(sys.argv[4]),"hostname":os.uname().nodename}; t=p+".tmp"; open(t,"w",encoding="utf-8").write(json.dumps(d,sort_keys=True,indent=2)+"\n"); os.replace(t,p)' \
    "$OUTPUT_ROOT/wrapper_exit_receipt.json" "$MODE" "$status" "$SIGNAL_STATUS"
}

finalize_outputs() {
  local status="$?"
  trap - EXIT TERM INT
  if [[ "$status" -eq "$CHECKPOINT_EXIT_CODE" ]]; then
    exit "$status"
  fi
  write_exit_receipt "$status" || status=74
  mkdir -p "$(dirname "$TRANSFER_ARCHIVE")"
  local files=()
  local relative
  for relative in \
    execution.json \
    normalized_run_manifest.json \
    validation.json \
    anchor_gate.json \
    wrapper_exit_receipt.json \
    json/result.json \
    json/current.json \
    json/estimator_call_ledger.json \
    qiskit_cost_sidecar.json \
    terminal_checkpoint.execution_order_repaired.json; do
    [[ -f "$OUTPUT_ROOT/$relative" ]] && files+=("$OUTPUT_ROOT/$relative")
  done
  shopt -s nullglob
  local checkpoint_ledger
  for checkpoint_ledger in \
    "$OUTPUT_ROOT"/json/current.estimator_call_ledger_checkpoint.*.json; do
    files+=("$checkpoint_ledger")
  done
  shopt -u nullglob
  if [[ "${#files[@]}" -eq 0 ]]; then
    echo "no recoverable output files were produced" >&2
    exit 74
  fi
  if ! tar -czf "${TRANSFER_ARCHIVE}.tmp" "${files[@]}"; then
    rm -f "${TRANSFER_ARCHIVE}.tmp"
    exit 74
  fi
  mv "${TRANSFER_ARCHIVE}.tmp" "$TRANSFER_ARCHIVE"
  exit "$status"
}

forward_signal() {
  SIGNAL_STATUS="$1"
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -TERM -- "-$CHILD_PID" 2>/dev/null || true
  fi
}

trap finalize_outputs EXIT
trap 'forward_signal 143' TERM
trap 'forward_signal 130' INT

for path in "$JOB_MANIFEST" "$SOURCE_ARCHIVE" "$SOURCE_AUDIT" "$IMAGE"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
[[ "$(sha256_file "$SOURCE_ARCHIVE")" == "$EXPECTED_SOURCE_SHA256" ]] || {
  echo "source archive hash mismatch" >&2
  exit 3
}
[[ "$(sha256_file "$SOURCE_AUDIT")" == "$EXPECTED_AUDIT_SHA256" ]] || {
  echo "source audit hash mismatch" >&2
  exit 3
}
[[ "$(sha256_file "$IMAGE")" == "$EXPECTED_IMAGE_SHA256" ]] || {
  echo "image hash mismatch" >&2
  exit 3
}
python3 -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); assert d.get("status")=="pass" and d.get("submission_enabled") is True, d.get("reason")' \
  "$BUNDLE_PATH/submission_gate.json"

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v singularity)"
else
  echo "apptainer/singularity unavailable" >&2
  exit 127
fi
command -v setsid >/dev/null 2>&1 || {
  echo "setsid unavailable; cannot guarantee coherent checkpoint shutdown" >&2
  exit 127
}

python3 "$BUNDLE_PATH/pair_contract.py" inspect "$SOURCE_ARCHIVE" >/dev/null
if [[ -f "$CHECKPOINT_ARCHIVE" ]]; then
  python3 "$BUNDLE_PATH/pair_contract.py" checkpoint-restore \
    "$MODE" "$JOB_MANIFEST" >/dev/null
  if [[ -f "$RESUME_DIR/current.json" ]]; then
    RESUME_CURRENT="$RESUME_DIR/current.json"
  elif [[ ! -f "$RESUME_DIR/checkpoint_manifest.json" ]]; then
    echo "validated checkpoint restore produced no authenticated payload" >&2
    exit 75
  fi
fi
python3 "$BUNDLE_PATH/pair_contract.py" extract "$SOURCE_ARCHIVE" "$PWD/runtime_source" >/dev/null
[[ -d "$RUNTIME_SOURCE" ]] || { echo "fixed runtime root missing" >&2; exit 3; }

ROOT="$PWD"
QISKIT_PREFLIGHT='import qiskit; assert qiskit.__version__ == "2.3.1", qiskit.__version__; from pipelines.qiskit_backend_tools import load_local_fake_backend; backend, resolved = load_local_fake_backend("FakeMarrakesh"); assert backend is not None and "marrakesh" in str(resolved).lower()'

setsid "$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
  bash -lc 'cd "$1" && export BUNDLE_WORK_ROOT=/work BUNDLE_RUNTIME_ROOT="$1" PYTHONPATH="$1" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; if [[ -n "$5" ]]; then export BUNDLE_RESUME_SCAFFOLD_JSON="/work/$5"; fi; python3 -c "$2"; python3 -u "$3" "$4"' \
  -- "/work/runtime_source/${RUNTIME_ROOT}" "$QISKIT_PREFLIGHT" \
  "/work/${BUNDLE_PATH}/run_job.py" "/work/${JOB_MANIFEST}" "$RESUME_CURRENT" &
CHILD_PID="$!"
set +e
wait "$CHILD_PID"
STATUS="$?"
if [[ "$SIGNAL_STATUS" -ne 0 ]]; then
  for _ in $(seq 1 120); do
    if ! kill -0 -- "-$CHILD_PID" 2>/dev/null; then break; fi
    sleep 1
  done
  if kill -0 -- "-$CHILD_PID" 2>/dev/null; then
    kill -KILL -- "-$CHILD_PID" 2>/dev/null || true
  fi
  wait "$CHILD_PID" 2>/dev/null || true
fi
set -e
CHILD_PID=""
if [[ "$SIGNAL_STATUS" -ne 0 ]]; then
  write_exit_receipt "$CHECKPOINT_EXIT_CODE"
  python3 "$BUNDLE_PATH/pair_contract.py" checkpoint-pack \
    "$MODE" "$JOB_MANIFEST" >/dev/null
  STATUS="$CHECKPOINT_EXIT_CODE"
fi
exit "$STATUS"
