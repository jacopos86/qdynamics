#!/usr/bin/env bash
set -euo pipefail

IMAGE="${PROJECT_IMAGE:-chtc/phase3_optuna/image.sif}"
CODE_ARCHIVE="${H2O_CODE_ARCHIVE:-chtc/paper_iv_h2o_static_snake_resume_20260705/input/h2o_whitened_nobatch_metric_prune_code_20260713.tgz}"
TASK_SCRIPT="chtc/paper_iv_h2o_static_snake_resume_20260705/run_h2o_resume_whitened_nobatch_metric_prune_task.sh"
EXPECTED_CODE_ARCHIVE_SHA256="331b71dc3a631386833e64ac21fac6b79df83f14cc6fcbe4823db63402e09f49"

if [[ ! -f "$IMAGE" ]]; then
  echo "Missing Apptainer image: $IMAGE" >&2
  exit 2
fi
if [[ ! -f "$CODE_ARCHIVE" ]]; then
  echo "Missing isolated code archive: $CODE_ARCHIVE" >&2
  exit 2
fi
ACTUAL_CODE_ARCHIVE_SHA256="$(sha256sum "$CODE_ARCHIVE" | awk '{print $1}')"
if [[ "$ACTUAL_CODE_ARCHIVE_SHA256" != "$EXPECTED_CODE_ARCHIVE_SHA256" ]]; then
  echo "Code archive SHA-256 mismatch: $ACTUAL_CODE_ARCHIVE_SHA256" >&2
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

tar -xzf "$CODE_ARCHIVE"

ROOT="$PWD"
APPTAINER_CMD_NAME="${APPTAINER_BIN##*/}"
H2O_ENV_NAMES=(
  H2O_SHELL_HEARTBEAT_SEC
  H2O_FIXTURE_JSON
  H2O_RESUME_SCAFFOLD_JSON
  H2O_CODE_ARCHIVE
  H2O_RESUME_DEPTH
  H2O_ADAPT_MAX_DEPTH
  H2O_ADAPT_SEGMENT_ID
  H2O_ADAPT_SEGMENT_MAX_NEW_ADMISSIONS
  STATIC_ADAPT_CANDIDATE_RECORD_CACHE
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
  bash -lc 'cd /work && bash chtc/paper_iv_h2o_static_snake_resume_20260705/run_h2o_resume_whitened_nobatch_metric_prune_task.sh "$@"' -- "$@"
