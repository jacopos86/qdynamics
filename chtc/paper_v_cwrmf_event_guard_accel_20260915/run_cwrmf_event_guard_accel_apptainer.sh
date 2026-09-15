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
# The named options are passed through explicitly; --cleanenv drops
# everything else, which is what keeps a segment reproducible.
"$APPTAINER_BIN" exec --cleanenv \
  --env "CWRMF_SOLVER_SETTING_PROFILE=${CWRMF_SOLVER_SETTING_PROFILE:-max_threads_8}" \
  --env "CWRMF_BUNDLE_SEED_POLICY=${CWRMF_BUNDLE_SEED_POLICY:-}" \
  --env "CWRMF_BUNDLE_SEED_RANK_MARGIN=${CWRMF_BUNDLE_SEED_RANK_MARGIN:-}" \
  --env "CWRMF_MAXIMUM_WALL_SECONDS=${CWRMF_MAXIMUM_WALL_SECONDS:-}" \
  --env "OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}" \
  --bind "$ROOT:/work" "$IMAGE" \
  bash -lc 'cd /work && bash chtc/paper_v_cwrmf_event_guard_accel_20260915/run_cwrmf_event_guard_accel.sh "$@"' -- "$@"
