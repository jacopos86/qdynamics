#!/usr/bin/env bash
set -euo pipefail

JOB_MANIFEST="${1:?job manifest path required}"
SOURCE_ARCHIVE="${2:?source archive path required}"
EXPECTED_SOURCE_SHA256="${3:?source archive SHA-256 required}"
IMAGE="${4:?Apptainer image path required}"
EXPECTED_IMAGE_SHA256="${5:?Apptainer image SHA-256 required}"
REGIME_SLUG="${6:?regime slug required}"
BUNDLE_ID="paper_i_hh_sr_snake_no_ordinary_novelty_all_six_20260715_v1_chtc"
OUTPUT_ROOT="raw_outputs/${BUNDLE_ID}/${REGIME_SLUG}"
TRANSFER_ARCHIVE="raw_outputs/${BUNDLE_ID}/${REGIME_SLUG}_transfer.tar.gz"

if [[ ! "$REGIME_SLUG" =~ ^[a-z0-9_]+$ ]]; then
  echo "Unsafe regime slug: $REGIME_SLUG" >&2
  exit 2
fi

package_outputs() {
  local status="$?"
  trap - EXIT
  mkdir -p "$OUTPUT_ROOT" "$(dirname "$TRANSFER_ARCHIVE")"
  rm -f "${TRANSFER_ARCHIVE}.tmp"
  if tar -czf "${TRANSFER_ARCHIVE}.tmp" "$OUTPUT_ROOT"; then
    mv "${TRANSFER_ARCHIVE}.tmp" "$TRANSFER_ARCHIVE"
  else
    echo "Failed to package output evidence" >&2
    status=74
  fi
  exit "$status"
}

trap package_outputs EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

for path in "$JOB_MANIFEST" "$SOURCE_ARCHIVE" "$IMAGE"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
done

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

ACTUAL_SOURCE_SHA256="$(sha256_file "$SOURCE_ARCHIVE")"
ACTUAL_IMAGE_SHA256="$(sha256_file "$IMAGE")"
if [[ "$ACTUAL_SOURCE_SHA256" != "$EXPECTED_SOURCE_SHA256" ]]; then
  echo "Source archive hash mismatch" >&2
  exit 3
fi
if [[ "$ACTUAL_IMAGE_SHA256" != "$EXPECTED_IMAGE_SHA256" ]]; then
  echo "Execution image hash mismatch" >&2
  exit 3
fi

if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v singularity)"
else
  echo "Neither apptainer nor singularity is available" >&2
  exit 127
fi

tar -xzf "$SOURCE_ARCHIVE"
ROOT="$PWD"
"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
  bash -lc 'cd /work && python3 -u "$1" "$2"' \
  -- \
  "chtc/phase3_optuna/input/${BUNDLE_ID}/run_job.py" \
  "$JOB_MANIFEST"
