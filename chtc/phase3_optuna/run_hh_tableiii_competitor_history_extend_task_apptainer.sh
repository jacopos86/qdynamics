#!/usr/bin/env bash
set -euo pipefail
JOB_ID="${1:?job_id required}"
OUT_ROOT="${2:-raw_outputs/paper_i_hh_tableiii_competitor_history_extend_20260609_v1/${JOB_ID}}"
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
ROOT="$PWD"
"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
  bash -lc 'cd /work && export TABLE_I_STATIC_SUITE_PROFILE=paper_i_three_model_hh_symmetric_20260527_v1 && python -u chtc/phase3_optuna/input/paper_i_hh_tableiii_competitor_history_extend_20260609_v1/run_extend_histories.py "$@"' -- "$JOB_ID" --output-root "$OUT_ROOT"
