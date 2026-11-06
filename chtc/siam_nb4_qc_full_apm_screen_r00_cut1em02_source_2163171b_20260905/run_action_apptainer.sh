#!/usr/bin/env bash
set -euo pipefail

ACTION_ID=siam-nb4-qc-full_apm-screen-r00-cut1em02
if [[ "${1:-}" != "$ACTION_ID" || "$#" -ne 1 ]]; then
  echo "usage: run_action_apptainer.sh $ACTION_ID" >&2
  exit 2
fi
mkdir -p raw_outputs
WORKER=chtc/siam_nb4_qc_full_apm_screen_r00_cut1em02_source_2163171b_20260905/run_action.py
EXPECTED_WORKER_SHA256="918e35065e2dd1ecc0a7ae324ae52c028ef76f49ebf7d82735388399ffed30b0"
IMAGE="${PROJECT_IMAGE:-chtc/time_dynamics_optuna/image.sif}"
EXPECTED_IMAGE_BYTES=216371200
EXPECTED_IMAGE_SHA256="fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"

if [[ ! -f "$WORKER" ]]; then
  echo "Missing digest-bound SIAM worker: $WORKER" >&2
  exit 2
fi
if [[ "$(sha256sum "$WORKER" | awk '{print $1}')" != "$EXPECTED_WORKER_SHA256" ]]; then
  echo "SIAM worker SHA-256 mismatch" >&2
  exit 2
fi
if [[ ! -f "$IMAGE" ]]; then
  echo "Missing Apptainer image: $IMAGE" >&2
  exit 2
fi
if [[ "$(wc -c < "$IMAGE" | tr -d " ")" != "$EXPECTED_IMAGE_BYTES" ]]; then
  echo "Apptainer image byte count mismatch" >&2
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
  bash -lc 'cd /work && PYTHONPATH=. PYTHONUNBUFFERED=1 python3 chtc/siam_nb4_qc_full_apm_screen_r00_cut1em02_source_2163171b_20260905/run_action.py "$@"' -- "$ACTION_ID"
