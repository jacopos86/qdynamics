#!/usr/bin/env bash
set -euo pipefail
mkdir -p raw_outputs
IMAGE="${PROJECT_IMAGE:-chtc/time_dynamics_optuna/image.sif}"
if [[ ! -f "$IMAGE" ]]; then echo "Missing Apptainer image: $IMAGE" >&2; exit 2; fi
if command -v apptainer >/dev/null 2>&1; then APPTAINER_BIN="$(command -v apptainer)";
elif command -v singularity >/dev/null 2>&1; then APPTAINER_BIN="$(command -v singularity)";
else echo "No apptainer/singularity on this execute node." >&2; exit 127; fi
ROOT="$PWD"
"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
  bash -lc 'cd /work && python3 chtc/paper_ii_staged_accuracy_euler_v1_resource_rescue_t5_cut9e-6_frontier340_source_74da5166_20260828/run_action.py "$@"' -- "$@"
