#!/usr/bin/env bash
set -euo pipefail

ACTION_ID="${1:?missing action id}"
IMAGE="${PROJECT_IMAGE:-chtc/time_dynamics_optuna/image.sif}"
EXPECTED_IMAGE_SHA256="fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"

if [[ ! -f "$IMAGE" ]]; then
  echo "Missing Apptainer image: $IMAGE" >&2
  exit 2
fi
if [[ "$(sha256sum "$IMAGE" | awk '{print $1}')" != "$EXPECTED_IMAGE_SHA256" ]]; then
  echo "Apptainer image SHA-256 mismatch" >&2
  exit 2
fi
if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v singularity)"
else
  echo "No apptainer/singularity on this execute node." >&2
  exit 127
fi

ROOT="$PWD"
"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
  bash -lc 'cd /work && PYTHONPATH=. PYTHONUNBUFFERED=1 python3 chtc/siam_nb4_avqds_screen_v1_source_21b129b9_20260827/run_screen.py "$@"' -- "$ACTION_ID"
