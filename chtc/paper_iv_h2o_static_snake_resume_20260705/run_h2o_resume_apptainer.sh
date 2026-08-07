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

ROOT="$PWD"
APPTAINER_CMD_NAME="${APPTAINER_BIN##*/}"
H2O_ENV_NAMES=(
  H2O_SHELL_HEARTBEAT_SEC
  H2O_RESUME_SCAFFOLD_JSON
  H2O_RESUME_DEPTH
  H2O_ADAPT_MAX_DEPTH
  H2O_ADAPT_SEGMENT_ID
  H2O_ADAPT_SEGMENT_TARGET_DEPTH
  H2O_ADAPT_SEGMENT_MAX_NEW_ADMISSIONS
  H2O_PHASE2_ENABLE_BATCHING
  H2O_SHORTLIST_CHANGE_REASON
  OMP_NUM_THREADS
  OPENBLAS_NUM_THREADS
  MKL_NUM_THREADS
  VECLIB_MAXIMUM_THREADS
  NUMEXPR_NUM_THREADS
)

if [[ "$APPTAINER_CMD_NAME" == singularity* ]]; then
  for name in "${H2O_ENV_NAMES[@]}"; do
    if [[ -n "${!name:-}" ]]; then
      export "SINGULARITYENV_${name}=${!name}"
    fi
  done
else
  for name in "${H2O_ENV_NAMES[@]}"; do
    if [[ -n "${!name:-}" ]]; then
      export "APPTAINERENV_${name}=${!name}"
    fi
  done
fi

"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
  bash -lc 'cd /work && bash chtc/paper_iv_h2o_static_snake_resume_20260705/run_h2o_resume_task.sh "$@"' -- "$@"
