#!/usr/bin/env bash
set -euo pipefail

IMAGE="${PROJECT_IMAGE:-chtc/phase3_optuna/image.sif}"
CELL_MANIFEST="${1:?cell manifest required}"
OUTPUT_ROOT="${2:?output root required}"
OUTPUT_RELATIVE_DIR="${3:?output-relative directory required}"
RESUME_SCAFFOLD="${4:?resume scaffold required}"
TARGET_CONTROLLER_ROUND="${5:-}"

for required in "$IMAGE" "$CELL_MANIFEST" "$RESUME_SCAFFOLD" \
  "chtc/phase3_optuna/run_jr_resume_cell.py"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing resume input: $required" >&2
    exit 2
  fi
done

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v singularity)"
else
  echo "Neither apptainer nor singularity is available." >&2
  exit 127
fi

ROOT="$PWD"
CELL_OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_RELATIVE_DIR"
mkdir -p "$CELL_OUTPUT_DIR"

record_failure() {
  local status=$?
  printf '{"status":"wrapper_failed","exit_code":%d}\n' "$status" \
    > "$CELL_OUTPUT_DIR/failure_status.json"
  exit "$status"
}
trap record_failure ERR

"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
  bash -lc '
    cd /work
    export PYTHONHASHSEED=0
    target_args=()
    if [[ -n "$4" ]]; then
      target_args=(--target-controller-round "$4")
    fi
    python -u chtc/phase3_optuna/run_jr_resume_cell.py \
      --cell-manifest "$1" \
      --output-root "$2" \
      --resume-scaffold-json "$3" \
      --gradient-workers 4 \
      --beam-parent-workers 1 \
      --runtime-split-child-workers 0 \
      --joint-pair-workers 4 \
      "${target_args[@]}"
  ' -- "$CELL_MANIFEST" "$OUTPUT_ROOT" "$RESUME_SCAFFOLD" "$TARGET_CONTROLLER_ROUND"

trap - ERR
printf '{"status":"wrapper_complete","exit_code":0}\n' \
  > "$CELL_OUTPUT_DIR/wrapper_status.json"
