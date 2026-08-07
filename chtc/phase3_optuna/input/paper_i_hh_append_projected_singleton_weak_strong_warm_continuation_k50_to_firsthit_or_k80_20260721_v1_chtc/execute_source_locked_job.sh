#!/usr/bin/env bash
set -euo pipefail

JOB_MANIFEST="$1"
WARM_START_STATE="$2"
SOURCE_ARCHIVE="$3"
EXPECTED_SOURCE_SHA256="$4"
IMAGE="$5"
EXPECTED_IMAGE_SHA256="$6"
JOB_ID="$7"
BUNDLE_ID="paper_i_hh_append_projected_singleton_weak_strong_warm_continuation_k50_to_firsthit_or_k80_20260721_v1_chtc"
OUTPUT_ROOT="raw_outputs/$BUNDLE_ID/$JOB_ID"
TRANSFER_ARCHIVE="raw_outputs/$BUNDLE_ID/$JOB_ID"_transfer.tar.gz

[[ "$JOB_ID" =~ ^[a-z0-9_]+$ ]] || { echo "unsafe job id" >&2; exit 2; }

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

package_outputs() {
  local status="$?"
  trap - EXIT
  mkdir -p "$OUTPUT_ROOT" "$(dirname "$TRANSFER_ARCHIVE")"
  tar -czf "$TRANSFER_ARCHIVE.tmp" "$OUTPUT_ROOT" || status=74
  if [[ -f "$TRANSFER_ARCHIVE.tmp" ]]; then mv "$TRANSFER_ARCHIVE.tmp" "$TRANSFER_ARCHIVE"; fi
  exit "$status"
}
trap package_outputs EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

for path in "$JOB_MANIFEST" "$WARM_START_STATE" "$SOURCE_ARCHIVE" "$IMAGE"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
[[ "$(sha256_file "$SOURCE_ARCHIVE")" == "$EXPECTED_SOURCE_SHA256" ]] || { echo "source archive hash mismatch" >&2; exit 3; }
[[ "$(sha256_file "$IMAGE")" == "$EXPECTED_IMAGE_SHA256" ]] || { echo "image hash mismatch" >&2; exit 3; }

if command -v apptainer >/dev/null 2>&1; then APPTAINER_BIN="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then APPTAINER_BIN="$(command -v singularity)"
else echo "apptainer/singularity unavailable" >&2; exit 127; fi

mkdir -p source_locked "$OUTPUT_ROOT"
tar -xzf "$SOURCE_ARCHIVE" -C source_locked
ROOT="$PWD"
QISKIT_PREFLIGHT='import qiskit; assert qiskit.__version__ == "2.3.1"; from pipelines.qiskit_backend_tools import load_local_fake_backend; backend, resolved = load_local_fake_backend("FakeMarrakesh"); assert backend is not None and "marrakesh" in str(resolved).lower()'
"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \
  bash -lc 'cd /work && export PYTHONPATH=/work/source_locked PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; python3 -c "$4"; python3 -u "$1" "$2" "$3" "$5"' \
  -- "chtc/phase3_optuna/input/$BUNDLE_ID/run_job.py" "$JOB_MANIFEST" "$WARM_START_STATE" "$QISKIT_PREFLIGHT" "$OUTPUT_ROOT"
