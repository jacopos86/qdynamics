#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="chtc/paper_ii_corrected_exchange_next_rung3_v1_df9abc5d_20260906"
SOURCE_ARCHIVE="$PACKAGE_DIR/input/source_df9abc5d.tar.gz"
EXPECTED_SOURCE_SHA256="bc41d0dc15774de0881e52b331727d905e509a35276a40a8f7c3ea4c7f832e70"
IMAGE="${PROJECT_IMAGE:-chtc/time_dynamics_optuna/image.sif}"
EXPECTED_IMAGE_SHA256="fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
EXPECTED_IMAGE_SIZE="216371200"

if [[ ! -f "$SOURCE_ARCHIVE" ]]; then
  echo "Missing source archive: $SOURCE_ARCHIVE" >&2
  exit 2
fi
observed_source_sha256="$(sha256sum "$SOURCE_ARCHIVE" | awk '{print $1}')"
if [[ "$observed_source_sha256" != "$EXPECTED_SOURCE_SHA256" ]]; then
  echo "Source archive SHA-256 drift: $observed_source_sha256" >&2
  exit 3
fi
if [[ ! -f "$IMAGE" ]]; then
  echo "Missing Apptainer image: $IMAGE" >&2
  exit 4
fi
observed_image_size="$(wc -c < "$IMAGE" | tr -d '[:space:]')"
if [[ "$observed_image_size" != "$EXPECTED_IMAGE_SIZE" ]]; then
  echo "Apptainer image size drift: $observed_image_size" >&2
  exit 5
fi
observed_image_sha256="$(sha256sum "$IMAGE" | awk '{print $1}')"
if [[ "$observed_image_sha256" != "$EXPECTED_IMAGE_SHA256" ]]; then
  echo "Apptainer image SHA-256 drift: $observed_image_sha256" >&2
  exit 6
fi
if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v singularity)"
else
  echo "No apptainer/singularity on this execute node." >&2
  exit 127
fi

tar -xzf "$SOURCE_ARCHIVE"
mkdir -p raw_outputs archives
ROOT="$PWD"
"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
  bash -lc 'cd /work && env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 PYTHONPATH=/work python3 "$1/run_action.py" "$2"' \
  -- "$PACKAGE_DIR" "$1"
