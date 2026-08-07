#!/usr/bin/env bash
set -euo pipefail
CASE_ID="${1:?case_id required}"
OUT_ROOT="${2:-raw_outputs/routeA_paper_i_hh_weak_strong_snake_depth42_reprobe_20260613_v1/${CASE_ID}}"
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
  bash -lc 'cd /work && export TABLE_I_STATIC_SUITE_PROFILE=paper_i_three_model_hh_symmetric_20260527_v1 && python -u chtc/phase3_optuna/input/routeA_paper_i_hh_weak_strong_snake_depth42_reprobe_20260613_v1/run_weak_strong_depth42_reprobe.py "$@"' -- "$CASE_ID" --output-root "$OUT_ROOT"
