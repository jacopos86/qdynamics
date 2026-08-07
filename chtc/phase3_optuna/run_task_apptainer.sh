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
PHASE3_ENV_NAMES=(
  PHASE3_RECORDS_PATH
  PHASE3_TERMINATE_ON_STALE_PROGRESS
  PHASE3_REQUIRE_FIRST_PROGRESS_WITHIN_SEC
  PHASE3_PROGRESS_STALE_AFTER_SEC
  PHASE3_HEARTBEAT_INTERVAL_SEC
  PHASE3_SHELL_HEARTBEAT_SEC
)
if [[ "$APPTAINER_CMD_NAME" == singularity* ]]; then
  for name in "${PHASE3_ENV_NAMES[@]}"; do
    if [[ -n "${!name:-}" ]]; then
      export "SINGULARITYENV_${name}=${!name}"
    fi
  done
else
  for name in "${PHASE3_ENV_NAMES[@]}"; do
    if [[ -n "${!name:-}" ]]; then
      export "APPTAINERENV_${name}=${!name}"
    fi
  done
fi
"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
  bash -lc 'cd /work && bash chtc/phase3_optuna/run_task.sh "$@"' -- "$@"
