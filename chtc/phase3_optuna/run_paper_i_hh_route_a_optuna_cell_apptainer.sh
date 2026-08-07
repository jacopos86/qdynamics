#!/usr/bin/env bash
set -euo pipefail

IMAGE="${PROJECT_IMAGE:-chtc/phase3_optuna/image.sif}"
CELL_MANIFEST="${1:?cell manifest required}"
OUTPUT_ROOT="${2:?output root required}"
OUTPUT_RELATIVE_DIR="${3:?output-relative directory required}"

if [[ ! -f "$IMAGE" ]]; then
  echo "Missing Apptainer image: $IMAGE" >&2
  exit 2
fi
if [[ ! -f "$CELL_MANIFEST" ]]; then
  echo "Missing immutable cell manifest: $CELL_MANIFEST" >&2
  exit 2
fi
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
    python -u -m pipelines.exact_bench.paper_i_hh_route_a_optuna run-cell \
      --cell-manifest "$1" \
      --output-root "$2" \
      --gradient-workers 4 \
      --beam-parent-workers 1 \
      --runtime-split-child-workers 0 \
      --joint-pair-workers 4
  ' -- "$CELL_MANIFEST" "$OUTPUT_ROOT"

trap - ERR
printf '{"status":"wrapper_complete","exit_code":0}\n' \
  > "$CELL_OUTPUT_DIR/wrapper_status.json"
