#!/usr/bin/env bash
set -euo pipefail

JOB_MANIFEST="${1:?job manifest required}"
SOURCE_ARCHIVE="${2:?source archive required}"
EXPECTED_SOURCE_SHA256="${3:?source SHA-256 required}"
IMAGE="${4:?image required}"
EXPECTED_IMAGE_SHA256="${5:?image SHA-256 required}"
REGIME_SLUG="${6:?regime slug required}"
BUNDLE_ID="paper_i_hh_sr_snake_no_overlap_trust_parent_anchor_weak_weak_r50_20260720_v2_chtc"
OUTPUT_ROOT="raw_outputs/${BUNDLE_ID}/${REGIME_SLUG}"
TRANSFER_ARCHIVE="raw_outputs/${BUNDLE_ID}/${REGIME_SLUG}_transfer.tar.gz"

[[ "$REGIME_SLUG" =~ ^[a-z0-9_]+$ ]] || { echo "unsafe regime slug" >&2; exit 2; }

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

package_outputs() {
  local status="$?"
  trap - EXIT
  mkdir -p "$OUTPUT_ROOT" "$(dirname "$TRANSFER_ARCHIVE")"
  tar -czf "${TRANSFER_ARCHIVE}.tmp" "$OUTPUT_ROOT" || status=74
  if [[ -f "${TRANSFER_ARCHIVE}.tmp" ]]; then mv "${TRANSFER_ARCHIVE}.tmp" "$TRANSFER_ARCHIVE"; fi
  exit "$status"
}
trap package_outputs EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

for path in "$JOB_MANIFEST" "$SOURCE_ARCHIVE" "$IMAGE"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
[[ "$(sha256_file "$SOURCE_ARCHIVE")" == "$EXPECTED_SOURCE_SHA256" ]] || { echo "source archive hash mismatch" >&2; exit 3; }
[[ "$(sha256_file "$IMAGE")" == "$EXPECTED_IMAGE_SHA256" ]] || { echo "image hash mismatch" >&2; exit 3; }

if command -v apptainer >/dev/null 2>&1; then APPTAINER_BIN="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then APPTAINER_BIN="$(command -v singularity)"
else echo "apptainer/singularity unavailable" >&2; exit 127; fi

tar -xzf "$SOURCE_ARCHIVE"
ROOT="$PWD"
QISKIT_PREFLIGHT='import qiskit; from pipelines.qiskit_backend_tools import load_local_fake_backend; backend, resolved = load_local_fake_backend("FakeMarrakesh"); assert backend is not None and "marrakesh" in str(resolved).lower()'
"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
  bash -lc 'set -euo pipefail; cd /work && export PYTHONPATH=/work PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; python3 -c "$3"; python3 -u "$1" "$2"' \
  -- "chtc/phase3_optuna/input/${BUNDLE_ID}/run_job.py" "$JOB_MANIFEST" "$QISKIT_PREFLIGHT"
