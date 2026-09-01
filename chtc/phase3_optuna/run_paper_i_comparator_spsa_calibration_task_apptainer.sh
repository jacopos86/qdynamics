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

ENV_PREFIX="APPTAINERENV_"
if [[ "$(basename "$APPTAINER_BIN")" == "singularity" ]]; then
  ENV_PREFIX="SINGULARITYENV_"
fi
for var in PAPER_I_COMPARATOR_SPSA_CALIBRATION_RECORDS_PATH PAPER_I_COMPARATOR_SPSA_CALIBRATION_CONFIG_PATH; do
  value="${!var-}"
  if [[ -n "$value" ]]; then
    export "${ENV_PREFIX}${var}=${value}"
  fi
done

ROOT="$PWD"
"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
  bash -lc 'cd /work && bash chtc/phase3_optuna/run_paper_i_comparator_spsa_calibration_task.sh "$@"' -- "$@"
