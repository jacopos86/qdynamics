#!/usr/bin/env bash
set -euo pipefail

IMAGE="${PROJECT_IMAGE:-chtc/phase3_optuna/image.sif}"
if [[ ! -f "$IMAGE" ]]; then
  echo "Missing Apptainer image: $IMAGE" >&2
  exit 2
fi
if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v singularity)"
else
  echo "Neither apptainer nor singularity is available on this execute node." >&2
  exit 127
fi

RECORD_ID="${1:?record_id required}"
RECORDS_PATH="${2:?records_path required}"
OUT_ROOT="${3:?output_root required}"
ROOT="$PWD"
BATCH_DIR="$(basename "$(dirname "$OUT_ROOT")")"
mkdir -p "$(dirname "$OUT_ROOT")" "logs/${BATCH_DIR}/${RECORD_ID}"

"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
  bash -lc 'cd /work && python -u -m chtc.phase3_optuna.run_paper_i_hh_snake_prefix_sidecar_cell "$@"' \
  -- "$RECORD_ID" "$RECORDS_PATH" "$OUT_ROOT"
