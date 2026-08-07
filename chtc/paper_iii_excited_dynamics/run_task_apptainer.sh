#!/usr/bin/env bash
set -euo pipefail
IMAGE="${PROJECT_IMAGE:-chtc/time_dynamics_optuna/image.sif}"
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
ENV_ARGS=()
for name in \
  PAPER_III_EXCITED_DYNAMICS_RECORDS_PATH \
  PAPER_III_EXCITED_DYNAMICS_OUTPUT_ROOT; do
  if [[ -n "${!name:-}" ]]; then
    ENV_ARGS+=(--env "$name=${!name}")
  fi
done
"$APPTAINER_BIN" exec --cleanenv "${ENV_ARGS[@]}" --bind "$ROOT:/work" "$IMAGE" \
  bash -lc 'cd /work && bash chtc/paper_iii_excited_dynamics/run_task.sh "$@"' -- "$@"
