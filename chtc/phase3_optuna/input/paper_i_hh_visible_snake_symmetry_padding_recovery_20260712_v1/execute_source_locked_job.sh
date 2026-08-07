#!/usr/bin/env bash
set -euo pipefail

JOB_MANIFEST="${1:?job manifest path required}"
SOURCE_ARCHIVE="${2:?exact source archive path required}"
EXPECTED_SOURCE_SHA256="${3:?source archive SHA-256 required}"
IMAGE="${4:?exact Apptainer image path required}"
EXPECTED_IMAGE_SHA256="${5:?Apptainer image SHA-256 required}"

if [[ ! -f "$JOB_MANIFEST" ]]; then
  echo "Missing job manifest: $JOB_MANIFEST" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_ARCHIVE" ]]; then
  echo "Missing exact source archive: $SOURCE_ARCHIVE" >&2
  exit 2
fi
if [[ ! -f "$IMAGE" ]]; then
  echo "Missing Apptainer image: $IMAGE" >&2
  exit 2
fi
if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL_SOURCE_SHA256="$(sha256sum "$SOURCE_ARCHIVE" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  ACTUAL_SOURCE_SHA256="$(shasum -a 256 "$SOURCE_ARCHIVE" | awk '{print $1}')"
else
  echo "No SHA-256 utility is available on the execute node." >&2
  exit 127
fi
if [[ "$ACTUAL_SOURCE_SHA256" != "$EXPECTED_SOURCE_SHA256" ]]; then
  echo "Source archive hash mismatch: expected=$EXPECTED_SOURCE_SHA256 actual=$ACTUAL_SOURCE_SHA256" >&2
  exit 3
fi
if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL_IMAGE_SHA256="$(sha256sum "$IMAGE" | awk '{print $1}')"
else
  ACTUAL_IMAGE_SHA256="$(shasum -a 256 "$IMAGE" | awk '{print $1}')"
fi
if [[ "$ACTUAL_IMAGE_SHA256" != "$EXPECTED_IMAGE_SHA256" ]]; then
  echo "Execution image hash mismatch: expected=$EXPECTED_IMAGE_SHA256 actual=$ACTUAL_IMAGE_SHA256" >&2
  exit 3
fi
if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v singularity)"
else
  echo "Neither apptainer nor singularity is available on this execute node." >&2
  exit 127
fi

# The archive path is explicit and hash-locked.  Never glob for or reuse a
# source archive from another bundle.
tar -xzf "$SOURCE_ARCHIVE"
ROOT="$PWD"
"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
  bash -lc 'cd /work && python3 -u "$1" "$2"' \
  -- \
  "chtc/phase3_optuna/input/paper_i_hh_visible_snake_symmetry_padding_recovery_20260712_v1/run_job.py" \
  "$JOB_MANIFEST"
