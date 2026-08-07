#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-geo}"
OUT_ROOT="${2:-raw_outputs/paper_i_hh_u8_strong_strong_geo_no_target_continue_20260614_v1/${METHOD}}"
MAX_DEPTH="${3:-60}"
MAX_SEGMENTS="${4:-}"

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

if [[ "$METHOD" != "geo" ]]; then
  echo "This U/t=8 continuation wrapper only supports method=geo; got ${METHOD}" >&2
  exit 64
fi

ROOT="$PWD"
if [[ -n "$MAX_SEGMENTS" ]]; then
  "$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
    bash -lc 'cd /work && export TABLE_I_STATIC_SUITE_PROFILE=paper_i_three_model_hh_symmetric_u8_20260611_v1 && python -u chtc/phase3_optuna/input/paper_i_hh_u8_strong_strong_geo_no_target_continue_20260614_v1/run_u8_strong_strong_geo_no_target_continue.py "$@"' -- \
      "$METHOD" --output-root "$OUT_ROOT" --max-depth "$MAX_DEPTH" --max-segments "$MAX_SEGMENTS"
else
  "$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
    bash -lc 'cd /work && export TABLE_I_STATIC_SUITE_PROFILE=paper_i_three_model_hh_symmetric_u8_20260611_v1 && python -u chtc/phase3_optuna/input/paper_i_hh_u8_strong_strong_geo_no_target_continue_20260614_v1/run_u8_strong_strong_geo_no_target_continue.py "$@"' -- \
      "$METHOD" --output-root "$OUT_ROOT" --max-depth "$MAX_DEPTH"
fi
