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
  GENERIC_TD_TABLE_RECORDS_PATH \
  GENERIC_TD_CLASS_SETTINGS_MANIFEST \
  GENERIC_TD_REQUIRE_LOCKED_CLASS_SETTINGS \
  GENERIC_TD_OUTPUT_ROOT \
  GENERIC_TD_RECORD_ID \
  GENERIC_TD_QISKIT_DYNAMICS_MODE \
  GENERIC_TD_QISKIT_QUBIT_CAP; do
  if [[ -n "${!name:-}" ]]; then
    ENV_ARGS+=(--env "$name=${!name}")
  fi
done
"$APPTAINER_BIN" exec --cleanenv "${ENV_ARGS[@]}" --bind "$ROOT:/work" "$IMAGE" \
  bash -lc 'cd /work && bash chtc/generic_time_dynamics_table/run_task.sh "$@"' -- "$@"
